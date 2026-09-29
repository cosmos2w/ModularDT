"""Exact native direct-pair values with a separate soft scorer gradient.

The caller builds independently scored receiver/source plans for the current
native preparation and requested query panels. No grouped node approximation
or target-derived labels enter this bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.func import functional_call

from .adaptive_interaction_cover import InteractionContext, MechanismPlan
from .core import InterfaceFieldCore
from .native_joint_shadow import _detach_inputs, _FixedPlanPrediction
from .types import EncodedInterfaceCase, PreparedInterfaceField


@dataclass(frozen=True)
class DirectShadowResult:
    prediction: torch.Tensor
    hard_prediction: torch.Tensor
    soft_prediction: torch.Tensor
    hard_plans: tuple[MechanismPlan, ...]
    soft_plans: tuple[MechanismPlan, ...]
    hard_prepared: PreparedInterfaceField


def hard_value_soft_direct_forward(
    core: InterfaceFieldCore,
    encoded: EncodedInterfaceCase,
    module_states: torch.Tensor,
    hard_plans: tuple[MechanismPlan, ...],
    soft_plans: tuple[MechanismPlan, ...],
    query_coords: torch.Tensor,
    query_features: torch.Tensor | None = None,
    *,
    receiver_chunk_size: int | None = None,
    interaction_context: InteractionContext | None = None,
) -> DirectShadowResult:
    """Backpropagate hard physical values and soft direct-pair score gradients.

    Native plan binding must verify every direct receiver panel before either
    branch executes. The soft physical function receives detached parameters,
    inputs and buffers. Only soft access tensors retain a gradient path to the
    independent direct scorer.
    """

    if core.config.forward_architecture != "dense_pairwise_field":
        raise ValueError("native direct shadow requires the Dense interface-field architecture")
    backend = core.backend
    if not bool(getattr(backend, "optional_native_policy", False)) or backend.cover_mode != "external":
        raise ValueError("native direct shadow requires the external native cover backend")
    if backend.cover_executor != "dense_masked":
        raise ValueError("native direct shadow requires the dense-masked scientific executor")
    case_count = int(encoded.module_present.shape[0])
    if len(hard_plans) != case_count or len(soft_plans) != case_count:
        raise ValueError("one hard and one soft direct plan is required for each case")
    if any(hard is soft for hard, soft in zip(hard_plans, soft_plans, strict=True)):
        raise ValueError("hard and soft direct plans must be separate objects")

    hard_options = {"fixed_cover_plans": hard_plans}
    if interaction_context is not None:
        hard_options["interaction_context"] = interaction_context
    hard_prepared = core.prepare(encoded, module_states, **hard_options)
    hard_prediction = core.decode_queries(
        hard_prepared,
        query_coords,
        query_features,
        receiver_chunk_size=receiver_chunk_size,
    )["pred_field"]

    shadow = _FixedPlanPrediction(core)
    detached_parameters = {
        name: parameter.detach() for name, parameter in shadow.named_parameters()
    }
    detached_buffers = {
        name: buffer.detach().clone() for name, buffer in shadow.named_buffers()
    }
    detached_encoded = _detach_inputs(encoded)
    detached_states = _detach_inputs(module_states)
    detached_queries = query_coords.detach()
    detached_features = None if query_features is None else query_features.detach()
    checkpointing = bool(backend.activation_checkpointing)
    try:
        backend.activation_checkpointing = False
        soft_prediction = functional_call(
            shadow,
            (detached_parameters, detached_buffers),
            (
                detached_encoded,
                detached_states,
                soft_plans,
                detached_queries,
                detached_features,
                receiver_chunk_size,
                _detach_inputs(interaction_context),
            ),
            strict=True,
        )
    finally:
        backend.activation_checkpointing = checkpointing
    if hard_prediction.shape != soft_prediction.shape:
        raise RuntimeError("hard and soft direct predictions must have the same shape")
    prediction = hard_prediction + (soft_prediction - soft_prediction.detach())
    return DirectShadowResult(
        prediction,
        hard_prediction,
        soft_prediction,
        hard_plans,
        soft_plans,
        hard_prepared,
    )


__all__ = ["DirectShadowResult", "hard_value_soft_direct_forward"]
