"""Exact hierarchical candidate lifecycle for ThermalChannel.

User request ``R`` and context ``c`` sample compact plans ``G`` then physical
designs ``D``. Every raw design is verified once through frozen autonomous
HONF to recover ``G_hat``. If enabled, one bounded correction is applied and
verified once more. Population success is reported before ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from honf_inverse_core.contracts import CompactPlan, NamedContext, PhysicalDesign
from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner
from honf_inverse_core.models.request_encoder import RequestEncoding
from honf_inverse_core.normalization import ScalarStats, VectorStats
from honf_inverse_core.request_schema import StructuredRequest
from honf_inverse_core.sampling.contracts import CandidateRecord, InverseSamplingResult
from honf_inverse_core.sampling.ranking import rank_candidates

from channelthermal.inverse.compact_plan import (
    COMPACT_PLAN_FEATURE_NAMES,
    COMPACT_PLAN_SCHEMA_NAME,
    denormalize_compact_plan,
    validate_compact_plan,
)
from channelthermal.inverse.geometry import decode_generated_design, normalize_geometry_constraints
from channelthermal.inverse.request import make_request_codec
from channelthermal.inverse.verifier import FrozenThermalChannelVerifier

from .scoring import compact_plan_distance, evaluate_request_satisfaction


@dataclass(frozen=True)
class EvaluationNormalizers:
    functional: Mapping[str, ScalarStats]
    context: VectorStats
    active_heat: ScalarStats
    total_heat: ScalarStats


def _repeat_encoding(encoding: RequestEncoding, repeats: int) -> RequestEncoding:
    return RequestEncoding(
        encoding.global_embedding.repeat_interleave(repeats, dim=0),
        encoding.token_embeddings.repeat_interleave(repeats, dim=0),
        encoding.token_mask.repeat_interleave(repeats, dim=0),
    )


def select_one_pass_representatives(
    raw: list[CandidateRecord],
    proposals: list[CandidateRecord],
) -> list[CandidateRecord]:
    """Accept a verified bounded proposal only when it improves its lineage.

    This is a one-pass trust region, not iterative optimization: each raw
    candidate receives at most one corrected proposal and one additional HONF
    verification.  Rejected proposals remain in the result for transparent
    proposal-only diagnostics, while final ranking consumes these accepted
    lineage representatives.
    """

    if not proposals:
        return list(raw)
    if len(raw) != len(proposals):
        raise ValueError("One-pass correction requires exactly one proposal per raw candidate.")
    accepted: list[CandidateRecord] = []
    for base, revised in zip(raw, proposals):
        lineage = (base.source_plan_index, base.source_layout_index)
        revised_lineage = (revised.source_plan_index, revised.source_layout_index)
        if lineage != revised_lineage:
            raise ValueError("One-pass correction proposal lineage does not match its raw candidate.")
        improves = revised.request_violation < base.request_violation
        accepted.append(revised if revised.geometry_valid and improves else base)
    return accepted


class ThermalChannelCandidateEvaluator:
    """Bridge case-neutral inverse tensors to exact physical verification."""

    def __init__(
        self,
        designer: HierarchicalInverseDesigner,
        verifier: FrozenThermalChannelVerifier,
        normalizers: EvaluationNormalizers,
    ) -> None:
        self.designer = designer
        self.verifier = verifier
        self.normalizers = normalizers
        self.device = next(designer.parameters()).device
        self.matching_mode = str(designer.model_config.get("matching_mode", "canonical"))
        if self.matching_mode not in {"canonical", "hungarian", "sinkhorn"}:
            raise ValueError(f"Unsupported compact-plan matching mode: {self.matching_mode!r}")
        self.forward_calls = 0

    def verify(self, *args: Any, **kwargs: Any) -> Any:
        """Expose the attached frozen HONF verifier through the public designer API.

        This path is intentionally a single exact forward call; candidate sampling
        uses the richer private record builder so it can retain call lineage.
        """

        self.forward_calls += 1
        return self.verifier.verify_design(*args, **kwargs)

    def _request_tensors(
        self,
        request: StructuredRequest,
        context: NamedContext,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        codec = make_request_codec(self.normalizers.functional)
        request = codec.materialize_normalized(request)
        tensors = {
            name: torch.as_tensor(value, device=self.device).unsqueeze(0)
            for name, value in codec.tensorize(request).as_dict().items()
        }
        context_tensor = torch.as_tensor(
            self.normalizers.context.normalize(context.vector), device=self.device
        ).unsqueeze(0)
        geometry, mask = normalize_geometry_constraints(
            request.geometry_constraints,
            context,
            max_modules=self.designer.layout_flow.max_modules,
            total_heat_stats=self.normalizers.total_heat,
        )
        return (
            tensors,
            context_tensor.float(),
            torch.as_tensor(geometry, device=self.device).unsqueeze(0).float(),
            torch.as_tensor(mask, device=self.device).unsqueeze(0).float(),
        )

    def _design(
        self,
        layout: np.ndarray,
        present: np.ndarray,
        context: NamedContext,
        request: StructuredRequest,
    ) -> PhysicalDesign:
        return decode_generated_design(
            layout,
            present,
            context,
            request.geometry_constraints,
            heat_mean=self.normalizers.active_heat.mean,
            heat_std=self.normalizers.active_heat.std,
        )

    def _planned(self, normalized: np.ndarray, context: NamedContext) -> CompactPlan:
        raw = denormalize_compact_plan(normalized, context)
        validate_compact_plan(raw, context)
        return CompactPlan(
            raw, normalized, COMPACT_PLAN_FEATURE_NAMES,
            COMPACT_PLAN_SCHEMA_NAME, 1, {"source": "conditional_plan_flow"},
        )

    def _verify_record(
        self,
        *,
        candidate_id: str,
        group: str,
        plan_index: int,
        layout_index: int,
        plan_seed: int,
        layout_seed: int,
        planned_normalized: np.ndarray,
        immutable_base_normalized: np.ndarray,
        layout: np.ndarray,
        present: np.ndarray,
        request: StructuredRequest,
        context: NamedContext,
        correction_used: bool,
        correction_magnitude: float,
        lineage_forward_calls: int,
        prior_forward_call_indices: list[int] | None = None,
    ) -> tuple[CandidateRecord, np.ndarray]:
        design = self._design(layout, present, context, request)
        planned = self._planned(planned_normalized, context)
        regional = [
            (token.request_type, token.region)
            for token in request.tokens
            if token.active and token.region is not None
        ]
        self.forward_calls += 1
        call_index = self.forward_calls
        verified = self.verifier.verify_design(
            design,
            context,
            regional_requests=regional,
            geometry_constraints=request.geometry_constraints,
            return_outputs=("environment", "internal", "interface"),
        )
        terms, violation, satisfied, residual = evaluate_request_satisfaction(
            request, verified.functionals, self.normalizers.functional
        )
        plan_distance = compact_plan_distance(
            planned.normalized,
            verified.compact_plan.normalized,
            matching_mode=self.matching_mode,
        )
        immutable_distance = compact_plan_distance(
            immutable_base_normalized,
            verified.compact_plan.normalized,
            matching_mode=self.matching_mode,
        )
        functional = {name: value.to_dict() for name, value in verified.functionals.items()}
        record = CandidateRecord(
            candidate_id=candidate_id,
            group=group,
            source_plan_index=plan_index,
            source_layout_index=layout_index,
            plan_seed=plan_seed,
            layout_seed=layout_seed,
            planned_compact_raw=planned.raw,
            planned_compact_normalized=planned.normalized,
            design=design.to_dict(),
            geometry=verified.geometry,
            realized_full_plan=verified.full_plan,
            realized_compact_raw=verified.compact_plan.raw,
            realized_compact_normalized=verified.compact_plan.normalized,
            plan_distance=plan_distance,
            immutable_base_plan_distance=immutable_distance,
            functional_values=functional,
            request_terms=terms,
            request_violation=violation,
            request_satisfied=satisfied,
            correction_used=correction_used,
            correction_magnitude=correction_magnitude,
            forward_call_count=lineage_forward_calls,
            forward_call_indices=[*(prior_forward_call_indices or []), call_index],
            outputs=verified.outputs,
        )
        return record, residual

    @torch.no_grad()
    def sample_candidates(
        self,
        *,
        request: StructuredRequest,
        context: NamedContext,
        num_plans: int = 8,
        layouts_per_plan: int = 4,
        correct_once: bool = True,
        top_k: int = 8,
        seed: int = 0,
    ) -> InverseSamplingResult:
        if int(num_plans) <= 0:
            raise ValueError("num_plans must be positive.")
        if int(layouts_per_plan) <= 0:
            raise ValueError("layouts_per_plan must be positive.")
        raw_candidate_count = int(num_plans) * int(layouts_per_plan)
        if int(top_k) <= 0 or int(top_k) > raw_candidate_count:
            raise ValueError(
                "top_k must be positive and no larger than "
                "num_plans * layouts_per_plan."
            )
        self.forward_calls = 0
        request_tensors, context_tensor, geometry, geometry_mask = self._request_tensors(request, context)
        encoding = self.designer.encode_request(request_tensors, context_tensor, geometry, geometry_mask)
        sampled = self.designer.sample_candidates(
            request=request_tensors,
            context=context_tensor,
            geometry_constraints=geometry,
            geometry_constraint_mask=geometry_mask,
            num_plans=num_plans,
            layouts_per_plan=layouts_per_plan,
            correct_once=False,
            seed=seed,
            verify=False,
        )
        plans = sampled.plans.detach().cpu().numpy()
        layouts = sampled.layouts.detach().cpu().numpy()
        present = sampled.module_present.detach().cpu().numpy()
        raw: list[CandidateRecord] = []
        residuals: list[np.ndarray] = []
        for index in range(plans.shape[0]):
            plan_index = int(sampled.plan_indices[index])
            layout_index = index % layouts_per_plan
            record, residual = self._verify_record(
                candidate_id=f"P{plan_index:03d}_L{layout_index:03d}",
                group="raw_unguided",
                plan_index=plan_index,
                layout_index=layout_index,
                plan_seed=sampled.plan_seeds[plan_index],
                layout_seed=sampled.layout_seeds[index],
                planned_normalized=plans[index],
                immutable_base_normalized=plans[index],
                layout=layouts[index],
                present=present[index],
                request=request,
                context=context,
                correction_used=False,
                correction_magnitude=0.0,
                lineage_forward_calls=1,
                prior_forward_call_indices=None,
            )
            raw.append(record)
            residuals.append(residual)
        corrected: list[CandidateRecord] = []
        if correct_once:
            if self.designer.corrector is None:
                raise RuntimeError("Correction requested, but inverse checkpoint has no corrector.")
            realized_tensor = torch.as_tensor(
                np.stack([candidate.realized_compact_normalized for candidate in raw]),
                device=self.device,
            )
            repeated_encoding = _repeat_encoding(encoding, len(raw))
            correction = self.designer.correct_once(
                planned_plan=sampled.plans,
                layout=sampled.layouts,
                module_present=sampled.module_present,
                realized_plan=realized_tensor,
                request_residuals=torch.as_tensor(np.stack(residuals), device=self.device),
                encoding=repeated_encoding,
                enabled=True,
            )
            assert correction is not None
            corrected_plans = correction.corrected_plan.detach().cpu().numpy()
            corrected_layouts = correction.corrected_layout.detach().cpu().numpy()
            magnitudes = correction.magnitude.detach().cpu().numpy()
            for index, base in enumerate(raw):
                record, _ = self._verify_record(
                    candidate_id=f"{base.candidate_id}_C",
                    group="corrected",
                    plan_index=base.source_plan_index,
                    layout_index=base.source_layout_index,
                    plan_seed=base.plan_seed,
                    layout_seed=base.layout_seed,
                    planned_normalized=corrected_plans[index],
                    immutable_base_normalized=plans[index],
                    layout=corrected_layouts[index],
                    present=present[index],
                    request=request,
                    context=context,
                    correction_used=True,
                    correction_magnitude=float(magnitudes[index]),
                    lineage_forward_calls=2,
                    prior_forward_call_indices=list(base.forward_call_indices),
                )
                corrected.append(record)
        accepted = select_one_pass_representatives(raw, corrected)
        ranked = rank_candidates(accepted, top_k=top_k)
        raw_violation = float(np.mean([candidate.request_violation for candidate in raw]))
        accepted_violation = float(np.mean([candidate.request_violation for candidate in accepted]))
        accepted_corrections = [candidate.correction_used for candidate in accepted]
        metadata = {
            "seed": int(seed),
            "num_plans": int(num_plans),
            "layouts_per_plan": int(layouts_per_plan),
            "raw_candidate_count": len(raw),
            "corrected_candidate_count": len(corrected),
            "forward_call_count": self.forward_calls,
            "raw_request_success_fraction": float(np.mean([candidate.request_satisfied for candidate in raw])),
            "raw_geometry_valid_fraction": float(np.mean([candidate.geometry_valid for candidate in raw])),
            "corrected_request_success_fraction": (
                None if not corrected else float(np.mean([candidate.request_satisfied for candidate in corrected]))
            ),
            "corrected_geometry_valid_fraction": (
                None if not corrected else float(np.mean([candidate.geometry_valid for candidate in corrected]))
            ),
            "corrected_group_is_proposal_only": True,
            "accepted_one_pass_count": len(accepted),
            "accepted_correction_count": int(np.sum(accepted_corrections)),
            "accepted_correction_fraction": float(np.mean(accepted_corrections)),
            "accepted_request_success_fraction": float(
                np.mean([candidate.request_satisfied for candidate in accepted])
            ),
            "accepted_request_term_satisfaction_fraction": float(
                np.mean([candidate.request_term_satisfaction_fraction for candidate in accepted])
            ),
            "raw_mean_request_violation": raw_violation,
            "accepted_mean_request_violation": accepted_violation,
            "accepted_relative_request_violation_improvement": (
                (raw_violation - accepted_violation) / max(raw_violation, 1.0e-8)
            ),
            "success_is_population_metric_not_reranked_claim": True,
        }
        return InverseSamplingResult(
            request_summary={"request": request.to_dict(), "context": context.to_dict()},
            metadata=metadata,
            raw_unguided=raw,
            corrected=corrected,
            accepted_one_pass=accepted,
            final_ranked=ranked,
        )


__all__ = [
    "EvaluationNormalizers",
    "ThermalChannelCandidateEvaluator",
    "select_one_pass_representatives",
]
