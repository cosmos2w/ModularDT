"""Public two-stage hierarchical inverse designer.

Structured request ``R`` and context ``c`` sample compact mechanism ``G``;
then `G,R,c` sample physical modular design ``D``. Frozen HONF verification
recovers realized plan ``G_hat``. An optional corrector may apply exactly one
bounded residual pass. Independent per-stage seeds preserve one-to-many
generation and no method performs iterative optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from .joint_corrector import CorrectionOutput, JointConsistencyCorrector
from .layout_flow import ConditionalLayoutFlow, SampledLayout
from .plan_flow import ConditionalPlanFlow, SampledPlan
from .request_encoder import RequestEncoding, RequestSetEncoder


VerifierAdapter = Callable[..., Any]

_REQUIRED_CHECKPOINT_PROVENANCE = {
    "forward_checkpoint_id",
    "inverse_dataset_version",
    "inverse_dataset_hash",
    "request_schema_version",
    "compact_plan_schema_version",
    "normalization_stats",
}


def _repeat_encoding(encoding: RequestEncoding, repeats: int) -> RequestEncoding:
    return RequestEncoding(
        encoding.global_embedding.repeat_interleave(repeats, dim=0),
        encoding.token_embeddings.repeat_interleave(repeats, dim=0),
        encoding.token_mask.repeat_interleave(repeats, dim=0),
    )


@dataclass(frozen=True)
class HierarchicalSampleBatch:
    plans: torch.Tensor
    layouts: torch.Tensor
    module_present: torch.Tensor
    module_count: torch.Tensor
    plan_indices: torch.Tensor
    plan_seeds: tuple[int, ...]
    layout_seeds: tuple[int, ...]
    verified: Any = None
    corrected: Any = None

    def to_dict(self) -> dict[str, Any]:
        def array(value: torch.Tensor) -> list[Any]:
            return value.detach().cpu().numpy().tolist()

        def serializable(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if torch.is_tensor(value):
                return array(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, Mapping):
                return {str(name): serializable(item) for name, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(item) for item in value]
            if hasattr(value, "to_dict"):
                return serializable(value.to_dict())
            return str(value)

        return {
            "plans": array(self.plans),
            "layouts": array(self.layouts),
            "module_present": array(self.module_present),
            "module_count": array(self.module_count),
            "plan_indices": array(self.plan_indices),
            "plan_seeds": list(self.plan_seeds),
            "layout_seeds": list(self.layout_seeds),
            "verified": serializable(self.verified),
            "corrected": serializable(self.corrected),
        }


class HierarchicalInverseDesigner(nn.Module):
    """Compose request encoder, plan flow, layout flow, and optional corrector."""

    def __init__(
        self,
        request_encoder: RequestSetEncoder,
        plan_flow: ConditionalPlanFlow,
        layout_flow: ConditionalLayoutFlow,
        corrector: JointConsistencyCorrector | None = None,
        *,
        verifier: VerifierAdapter | None = None,
        model_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if plan_flow.num_edges != layout_flow.num_edges:
            raise ValueError("Plan/layout flows must share K.")
        self.request_encoder = request_encoder
        self.plan_flow = plan_flow
        self.layout_flow = layout_flow
        self.corrector = corrector
        self.verifier_adapter = verifier
        self.model_config = dict(model_config or {})

    @property
    def correction_enabled(self) -> bool:
        return self.corrector is not None

    def load_compatible_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load schema-v1 weights, accepting only the additive ordered-plan path.

        Early diagnostic checkpoints predate ``ordered_plan_projection``. A
        zero initialization exactly preserves their pooled-plan behavior while
        keeping every other missing/unexpected key a hard error.
        """

        result = self.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            "layout_flow.ordered_plan_projection.weight",
            "layout_flow.ordered_plan_projection.bias",
        }
        missing = set(result.missing_keys)
        unexpected = set(result.unexpected_keys)
        if unexpected or not missing.issubset(allowed_missing):
            raise ValueError(
                f"Inverse checkpoint state mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        if missing:
            nn.init.zeros_(self.layout_flow.ordered_plan_projection.weight)
            nn.init.zeros_(self.layout_flow.ordered_plan_projection.bias)

    def encode_request(
        self,
        request: Mapping[str, torch.Tensor],
        context: torch.Tensor,
        geometry_constraints: torch.Tensor,
        geometry_constraint_mask: torch.Tensor | None = None,
    ) -> RequestEncoding:
        return self.request_encoder(request, context, geometry_constraints, geometry_constraint_mask)

    def sample_plan(
        self,
        encoding: RequestEncoding,
        *,
        num_plans: int = 1,
        seed: int = 0,
        steps: int | None = None,
    ) -> SampledPlan:
        if encoding.global_embedding.shape[0] != 1:
            raise ValueError("Public sample_plan currently accepts one request at a time.")
        if int(num_plans) <= 0:
            raise ValueError("num_plans must be positive.")
        repeated = _repeat_encoding(encoding, int(num_plans))
        noise = torch.stack(
            [
                torch.randn(
                    (self.plan_flow.num_edges, self.plan_flow.state_dim),
                    device=encoding.global_embedding.device,
                    dtype=encoding.global_embedding.dtype,
                    generator=torch.Generator(device=encoding.global_embedding.device).manual_seed(int(seed) + index),
                )
                for index in range(int(num_plans))
            ],
            dim=0,
        )
        return self.plan_flow.sample(repeated, steps=steps, initial_noise=noise)

    def sample_layout(
        self,
        plans: torch.Tensor,
        encoding: RequestEncoding,
        geometry_constraints: torch.Tensor,
        *,
        layouts_per_plan: int = 1,
        seed: int = 0,
        steps: int | None = None,
    ) -> SampledLayout:
        if plans.ndim != 3 or plans.shape[1:] != (self.plan_flow.num_edges, 12):
            raise ValueError(f"plans must have shape [B,{self.plan_flow.num_edges},12].")
        if int(layouts_per_plan) <= 0:
            raise ValueError("layouts_per_plan must be positive.")
        if encoding.global_embedding.shape[0] not in {1, plans.shape[0]}:
            raise ValueError("Layout encoding batch must be one or match the plan batch.")
        if geometry_constraints.shape[0] not in {1, plans.shape[0]}:
            raise ValueError("Geometry batch must be one or match the plan batch.")
        repeated_plans = plans.repeat_interleave(int(layouts_per_plan), dim=0)
        repeated_encoding = _repeat_encoding(encoding, repeated_plans.shape[0]) if encoding.global_embedding.shape[0] == 1 else _repeat_encoding(encoding, int(layouts_per_plan))
        repeated_geometry = geometry_constraints.repeat_interleave(
            repeated_plans.shape[0] if geometry_constraints.shape[0] == 1 else int(layouts_per_plan), dim=0
        )
        noise = torch.stack(
            [
                torch.randn(
                    (self.layout_flow.max_modules, 3),
                    device=plans.device,
                    dtype=plans.dtype,
                    generator=torch.Generator(device=plans.device).manual_seed(int(seed) + index),
                )
                for index in range(repeated_plans.shape[0])
            ],
            dim=0,
        )
        return self.layout_flow.sample(
            repeated_plans,
            repeated_encoding,
            geometry_constraints=repeated_geometry,
            steps=steps,
            initial_noise=noise,
        )

    def correct_once(
        self,
        *,
        planned_plan: torch.Tensor,
        layout: torch.Tensor,
        module_present: torch.Tensor,
        realized_plan: torch.Tensor,
        request_residuals: torch.Tensor,
        encoding: RequestEncoding,
        enabled: bool = True,
    ) -> CorrectionOutput | None:
        if not enabled or self.corrector is None:
            return None
        return self.corrector(
            planned_plan,
            layout,
            module_present,
            realized_plan,
            request_residuals,
            encoding.global_embedding,
        )

    def verify(self, *args: Any, **kwargs: Any) -> Any:
        if self.verifier_adapter is None:
            raise RuntimeError("No frozen verifier adapter is attached to this inverse designer.")
        if hasattr(self.verifier_adapter, "verify"):
            return self.verifier_adapter.verify(*args, **kwargs)
        if callable(self.verifier_adapter):
            return self.verifier_adapter(*args, **kwargs)
        raise TypeError("Attached verifier adapter must be callable or expose verify(...).")

    def attach_verifier(self, verifier: VerifierAdapter) -> "HierarchicalInverseDesigner":
        """Attach a case-owned exact runtime and return `self` for concise APIs."""

        self.verifier_adapter = verifier
        return self

    @torch.no_grad()
    def sample_candidates(
        self,
        *,
        request: Any,
        context: Any,
        geometry_constraints: torch.Tensor | None = None,
        geometry_constraint_mask: torch.Tensor | None = None,
        num_plans: int = 8,
        layouts_per_plan: int = 4,
        correct_once: bool = False,
        top_k: int = 8,
        seed: int = 0,
        verify: bool = False,
    ) -> Any:
        if not isinstance(request, Mapping) or not torch.is_tensor(context):
            runtime = self.verifier_adapter
            if runtime is None or not hasattr(runtime, "sample_candidates"):
                raise RuntimeError("Structured physical requests require an attached case inference runtime.")
            return runtime.sample_candidates(
                request=request,
                context=context,
                num_plans=num_plans,
                layouts_per_plan=layouts_per_plan,
                correct_once=correct_once,
                top_k=min(int(top_k), num_plans * layouts_per_plan),
                seed=seed,
            )
        if geometry_constraints is None:
            raise ValueError("Tensor-mode sampling requires geometry_constraints.")
        if num_plans <= 0 or layouts_per_plan <= 0:
            raise ValueError("num_plans and layouts_per_plan must be positive.")
        encoding = self.encode_request(request, context, geometry_constraints, geometry_constraint_mask)
        plan_seed = int(seed) + 104729
        layout_seed = int(seed) + 1000003
        sampled_plan = self.sample_plan(encoding, num_plans=num_plans, seed=plan_seed)
        sampled_layout = self.sample_layout(
            sampled_plan.compact_plan,
            encoding,
            geometry_constraints,
            layouts_per_plan=layouts_per_plan,
            seed=layout_seed,
        )
        candidate_plans = sampled_plan.compact_plan.repeat_interleave(layouts_per_plan, dim=0)
        plan_indices = torch.arange(num_plans, device=context.device).repeat_interleave(layouts_per_plan)
        verified = None
        if verify:
            verified = self.verify(
                plans=candidate_plans,
                layouts=sampled_layout.layout,
                module_present=sampled_layout.module_present,
                context=context,
                request=request,
            )
        if correct_once:
            if self.corrector is None:
                raise RuntimeError("correct_once=True but this checkpoint has no corrector.")
            raise RuntimeError(
                "Tensor-mode sampling cannot apply correction before exact verification. "
                "Attach a case inference runtime and pass a structured physical request."
            )
        return HierarchicalSampleBatch(
            plans=candidate_plans,
            layouts=sampled_layout.layout,
            module_present=sampled_layout.module_present,
            module_count=sampled_layout.module_count,
            plan_indices=plan_indices,
            plan_seeds=tuple(plan_seed + index for index in range(num_plans)),
            layout_seeds=tuple(layout_seed + index for index in range(num_plans * layouts_per_plan)),
            verified=verified,
            corrected=None,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, verifier: VerifierAdapter | None = None) -> "HierarchicalInverseDesigner":
        model = dict(config)
        allowed = {
            "num_edges", "max_modules", "request_hidden_dim", "plan_hidden_dim",
            "plan_layers", "plan_sampling_steps", "layout_hidden_dim", "layout_layers",
            "layout_sampling_steps", "corrector_enabled", "corrector_hidden_dim",
            "corrector_blocks", "max_plan_delta", "max_layout_delta", "dropout",
            "matching_mode",
        }
        unknown = sorted(set(model) - allowed)
        if unknown:
            raise ValueError(f"Unknown hierarchical inverse model config keys: {unknown}")
        matching_mode = str(model.get("matching_mode", "canonical"))
        if matching_mode not in {"canonical", "hungarian", "sinkhorn"}:
            raise ValueError(f"Unsupported compact-plan matching mode: {matching_mode!r}")
        model["matching_mode"] = matching_mode
        request_hidden = int(model.get("request_hidden_dim", 128))
        num_edges = int(model["num_edges"])
        max_modules = int(model.get("max_modules", 12))
        dropout = float(model.get("dropout", 0.05))
        encoder = RequestSetEncoder(hidden_dim=request_hidden, dropout=dropout)
        plan = ConditionalPlanFlow(
            num_edges=num_edges,
            condition_dim=request_hidden,
            hidden_dim=int(model.get("plan_hidden_dim", 256)),
            layers=int(model.get("plan_layers", 4)),
            dropout=dropout,
            sampling_steps=int(model.get("plan_sampling_steps", 24)),
        )
        layout = ConditionalLayoutFlow(
            num_edges=num_edges,
            max_modules=max_modules,
            condition_dim=request_hidden,
            hidden_dim=int(model.get("layout_hidden_dim", 256)),
            layers=int(model.get("layout_layers", 4)),
            dropout=dropout,
            sampling_steps=int(model.get("layout_sampling_steps", 24)),
        )
        corrector = None
        if bool(model.get("corrector_enabled", False)):
            corrector = JointConsistencyCorrector(
                num_edges=num_edges,
                max_modules=max_modules,
                condition_dim=request_hidden,
                hidden_dim=int(model.get("corrector_hidden_dim", 256)),
                blocks=int(model.get("corrector_blocks", 2)),
                dropout=dropout,
                max_plan_delta=float(model.get("max_plan_delta", 0.05)),
                max_layout_delta=float(model.get("max_layout_delta", 0.05)),
            )
        return cls(encoder, plan, layout, corrector, verifier=verifier, model_config=model)

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        verifier: VerifierAdapter | None = None,
    ) -> "HierarchicalInverseDesigner":
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_schema_name") != "honf_hierarchical_inverse":
            raise ValueError("Inverse checkpoint schema mismatch.")
        if int(checkpoint.get("checkpoint_schema_version", 0)) != 1:
            raise ValueError("Unsupported inverse checkpoint schema version.")
        provenance = checkpoint.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Inverse checkpoint provenance is missing or invalid.")
        missing = sorted(_REQUIRED_CHECKPOINT_PROVENANCE - set(provenance))
        if missing:
            raise ValueError(f"Inverse checkpoint provenance is incomplete: {missing}")
        if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
            raise ValueError("Inverse checkpoint is missing model_config/model_state_dict.")
        designer = cls.from_config(checkpoint["model_config"], verifier=verifier)
        designer.load_compatible_state_dict(checkpoint["model_state_dict"])
        designer.to(torch.device(device)).eval()
        return designer


__all__ = ["HierarchicalInverseDesigner", "HierarchicalSampleBatch", "VerifierAdapter"]
