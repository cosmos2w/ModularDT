"""Hard native values with a separate soft topology gradient for joint fitting.

The shadow is an optimization device. It is never used for inference or as a
physical prediction, and its gradients must not reach the physical model or
its inputs. Callers evaluate the deterministic hard plan at every review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

from .adaptive_interaction_cover import InteractionContext, MechanismPlan
from .core import InterfaceFieldCore
from .input_cover_organizer import InputOnlyCoverOrganizer
from .types import EncodedInterfaceCase, PreparedInterfaceField


@dataclass(frozen=True)
class JointShadowResult:
    """Prediction for optimization and its separately inspectable branches."""

    prediction: torch.Tensor
    hard_prediction: torch.Tensor
    soft_prediction: torch.Tensor
    hard_plans: tuple[MechanismPlan, ...]
    soft_plans: tuple[MechanismPlan, ...]
    hard_prepared: PreparedInterfaceField
    hard_interaction_aux: Mapping[str, torch.Tensor]


def _detach_inputs(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{
            field.name: _detach_inputs(getattr(value, field.name))
            for field in fields(value) if field.init
        })
    if isinstance(value, dict):
        return {key: _detach_inputs(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_inputs(item) for item in value)
    if isinstance(value, list):
        return [_detach_inputs(item) for item in value]
    return value


class _FixedPlanPrediction(nn.Module):
    """Functional-call target so every physical parameter can be detached."""

    def __init__(self, core: InterfaceFieldCore) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        plans: tuple[MechanismPlan, ...],
        query_coords: torch.Tensor,
        query_features: torch.Tensor | None,
        receiver_chunk_size: int | None,
        interaction_context: InteractionContext | None = None,
    ) -> torch.Tensor:
        prepare_kwargs = {"fixed_cover_plans": plans}
        if interaction_context is not None:
            prepare_kwargs["interaction_context"] = interaction_context
        prepared = self.core.prepare(encoded, module_states, **prepare_kwargs)
        return self.core.decode_queries(
            prepared, query_coords, query_features,
            receiver_chunk_size=receiver_chunk_size,
        )["pred_field"]


def hard_value_soft_organizer_forward(
    core: InterfaceFieldCore,
    encoded: EncodedInterfaceCase,
    module_states: torch.Tensor,
    organizer: InputOnlyCoverOrganizer,
    query_coords: torch.Tensor,
    query_features: torch.Tensor | None = None,
    *,
    receiver_chunk_size: int | None = None,
    collect_hard_aux: bool = False,
    budgets: Mapping[str, float | torch.Tensor] | torch.Tensor | None = None,
    frontier_cuts: Sequence[Sequence[int]] | None = None,
    budget_fractions: Mapping[str, float] | None = None,
) -> JointShadowResult:
    """Return exact hard values and physical AD, plus soft organizer AD.

    The caller owns a fresh ``encode_case`` for each physical update. The
    organizer receives only detached, current model-side inputs. Both plans
    use the same case geometry and logits. The hard plan has no straight-
    through path. ``functional_call`` supplies detached physical parameters
    and buffers to the soft execution without changing live requires-grad
    flags. The shadow also receives detached encoded and receiver inputs.
    ``collect_hard_aux`` enables path diagnostics for scheduled hard audits;
    ordinary optimizer steps avoid that per-chunk diagnostic work.
    """

    if core.config.forward_architecture != "dense_pairwise_field":
        raise ValueError("joint native shadow requires the Dense interface-field architecture")
    if not bool(getattr(core.backend, "optional_native_policy", False)):
        raise TypeError("joint native shadow requires the native cover backend")
    if core.backend.cover_mode != "external":
        raise ValueError("joint native shadow requires external cover mode for hard partial access")
    if core.backend.cover_executor != "dense_masked":
        raise ValueError("joint native shadow requires the dense-masked scientific executor")
    detached_encoded = _detach_inputs(encoded)
    detached_states = _detach_inputs(module_states)
    detached_queries = query_coords.detach()
    detached_features = None if query_features is None else query_features.detach()

    trees = core.backend.build_case_trees(detached_encoded)
    score_inputs = {
        "module_states": detached_states,
        "environment_states": detached_encoded.env_tokens,
        "global_state": detached_encoded.global_token,
    }
    if budgets is None:
        scores = organizer.score_cases(detached_encoded, score_inputs, trees)
    else:
        scores = organizer.score_cases(detached_encoded, score_inputs, trees, budgets=budgets)
    planning_options = {
        "frontier_cuts": frontier_cuts,
        "budget_fractions": budget_fractions,
    }
    hard_plans = organizer.plans_from_scores(
        scores, detached_encoded, trees, hard=True, **planning_options
    )
    soft_plans = organizer.plans_from_scores(
        scores, detached_encoded, trees, hard=False, **planning_options
    )
    if any(bool((plan.permission_matrix("QE") <= 0).any()) for plan in soft_plans):
        raise FloatingPointError("soft QE probabilities lost positive support; organizer logits may be saturated")
    if any(plan.split_gates.requires_grad or any(
        permission.requires_grad for permission in plan.permissions.values()
    ) for plan in hard_plans):
        raise RuntimeError("hard plans must carry no straight-through organizer gradient")

    hard_prepared = core.prepare(encoded, module_states, fixed_cover_plans=hard_plans)
    hard_output = core.decode_queries(
        hard_prepared, query_coords, query_features,
        receiver_chunk_size=receiver_chunk_size,
        return_interaction_aux=collect_hard_aux,
    )
    hard_prediction = hard_output["pred_field"]

    shadow = _FixedPlanPrediction(core)
    detached_parameters = {
        name: parameter.detach() for name, parameter in shadow.named_parameters()
    }
    detached_buffers = {
        name: buffer.detach().clone() for name, buffer in shadow.named_buffers()
    }
    # Non-reentrant checkpointing would recompute after functional_call has
    # restored live parameters. The shadow is therefore evaluated without
    # activation checkpointing; the hard physical branch retains its setting.
    checkpointing = bool(core.backend.activation_checkpointing)
    try:
        core.backend.activation_checkpointing = False
        soft_prediction = functional_call(
            shadow,
            (detached_parameters, detached_buffers),
            (
                detached_encoded, detached_states, soft_plans,
                detached_queries, detached_features, receiver_chunk_size,
            ),
            strict=True,
        )
    finally:
        core.backend.activation_checkpointing = checkpointing

    if hard_prediction.shape != soft_prediction.shape:
        raise RuntimeError("hard and soft native predictions must have the same shape")
    prediction = hard_prediction + (soft_prediction - soft_prediction.detach())
    return JointShadowResult(
        prediction=prediction,
        hard_prediction=hard_prediction,
        soft_prediction=soft_prediction,
        hard_plans=hard_plans,
        soft_plans=soft_plans,
        hard_prepared=hard_prepared,
        hard_interaction_aux=hard_output.get("_interaction_aux", {}),
    )
