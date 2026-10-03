"""Hard wrapper values/physical AD with a separate positive organizer shadow."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any

import torch
from torch.func import functional_call


def detach_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: detach_tree(getattr(value, field.name))
                                 for field in fields(value) if field.init})
    if isinstance(value, dict):
        return {key: detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_tree(item) for item in value)
    if isinstance(value, list):
        return [detach_tree(item) for item in value]
    return value


def _merge_prediction(hard: Any, soft: Any, *, diagnostic: bool = False) -> Any:
    if torch.is_tensor(hard) and torch.is_tensor(soft):
        if hard.is_floating_point() and hard.shape == soft.shape and not diagnostic:
            # Parentheses preserve the hard value even at low precision.
            return hard + (soft - soft.detach())
        return hard
    if isinstance(hard, dict) and isinstance(soft, dict):
        return {key: _merge_prediction(value, soft.get(key),
                    diagnostic=diagnostic or key.endswith("aux") or key.startswith("_"))
                for key, value in hard.items()}
    return hard


def hard_value_soft_hypergraph_forward(model: torch.nn.Module, *args: Any, **kwargs: Any) -> dict:
    """Two complete physical wrapper calls; inference remains hard only.

    Only ``backend.organizer`` parameters stay live in the soft call. All
    physical inputs and remaining parameters are detached. Activation
    checkpointing is disabled there so later recomputation cannot restore
    live physical parameters. The hard branch retains its normal settings.
    """
    backend = model.core.backend
    if not hasattr(backend, "permission_mode") or not hasattr(backend, "organizer"):
        return model(*args, **kwargs)
    previous = backend.permission_mode
    backend.permission_mode = "hard"
    hard = model(*args, **kwargs)
    parameters = {
        name: parameter if ".backend.organizer." in "." + name else parameter.detach()
        for name, parameter in model.named_parameters()
    }
    buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    checkpointing = [(module, module.activation_checkpointing) for module in model.modules()
                     if hasattr(module, "activation_checkpointing")]
    try:
        backend.permission_mode = "soft"
        for module, _ in checkpointing:
            module.activation_checkpointing = False
        soft = functional_call(model, (parameters, buffers), detach_tree(args), detach_tree(kwargs), strict=True)
        auxiliary = soft.get("interaction_aux", {})
        phase_costs = []
        for key, numerator in auxiliary.items():
            if key.endswith("hypergraph_structural_numerator"):
                denominator = auxiliary.get(key.removesuffix("numerator") + "denominator")
                if torch.is_tensor(numerator) and torch.is_tensor(denominator):
                    phase_costs.append(numerator / denominator.clamp_min(1))
        structural = torch.stack(phase_costs).mean() if phase_costs else getattr(backend, "last_structural_cost", None)
    finally:
        backend.permission_mode = previous
        for module, enabled in checkpointing:
            module.activation_checkpointing = enabled
    result = _merge_prediction(hard, soft)
    if torch.is_tensor(structural):
        result["campaign_structural_cost"] = structural
    result["campaign_shadow_calls"] = 1
    return result


__all__ = ["detach_tree", "hard_value_soft_hypergraph_forward"]
