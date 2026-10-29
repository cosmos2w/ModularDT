"""Hard physical values with explicit whole-wrapper or local organizer gradients."""

from __future__ import annotations

import math
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


def _whole_wrapper_shadow_forward(model: torch.nn.Module, *args: Any, **kwargs: Any) -> dict:
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


def bridge_local_context(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Select the hard value while adding only the local soft derivative."""
    # Cancel before adding, preserving even the hard low-precision value.
    return hard + (soft - soft.detach()).to(hard.dtype)


def local_modulated_sum(messages: torch.Tensor, weight: torch.Tensor,
                        projected_gain: torch.Tensor,
                        source_measure: torch.Tensor | None = None) -> torch.Tensor:
    """Reduce already evaluated messages; projections and masks are explicit.

    The caller detaches messages and physical projection weights for the soft
    branch. Permission/quotient construction remains in FP64 upstream; only
    the completed gain and weight enter the physical message dtype.
    """
    gain = projected_gain.to(messages.dtype)
    weighted = messages * (1.0 + torch.tanh(gain)) * weight.to(messages.dtype)[..., None]
    if source_measure is not None:
        weighted = weighted * source_measure[..., None]
    return weighted.sum(2)


def local_attention_context(query: torch.Tensor, key: torch.Tensor,
                            value: torch.Tensor, geometry_bias: torch.Tensor,
                            source_weight: torch.Tensor, support: torch.Tensor,
                            projected_score: torch.Tensor, projected_gain: torch.Tensor,
                            source_measure: torch.Tensor) -> torch.Tensor:
    """Complete the QE reduction before bridging, including omitted sources.

    Sensitive positive permissions reach the logarithm in their original
    precision. The source support is supplied per branch, so soft restoration
    is never blocked by the hard prior's zero-support mask. All physical
    arguments must be detached by the soft caller.
    """
    safe_weight = torch.where(support, source_measure[:, None] * source_weight,
                              torch.ones_like(source_weight))
    log_weight = safe_weight.log().to(query.dtype)[:, None]
    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    score = score + geometry_bias + log_weight + projected_score.to(query.dtype).permute(0, 3, 1, 2)
    score = score.masked_fill(~support[:, None], torch.finfo(score.dtype).min)
    attention = torch.softmax(score, -1) * support[:, None]
    attention = attention / attention.sum(-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).tiny)
    gain = projected_gain.to(value.dtype).permute(0, 3, 1, 2)
    return torch.matmul(attention * (1.0 + torch.tanh(gain)), value)


def hard_value_soft_hypergraph_forward(model: torch.nn.Module, *args: Any,
                                      gradient_policy: str = "whole_wrapper_shadow_v1",
                                      **kwargs: Any) -> dict:
    """Apply a named organizer derivative policy without changing hard values.

    Local mode executes the ordinary wrapper once. Each phase's backend
    preparation captures the policy and explicit hard/soft tensors; backward
    and activation-checkpoint replay never consult this temporary selection.
    The historical whole-wrapper policy remains available for exact replay.
    """
    if gradient_policy == "whole_wrapper_shadow_v1":
        return _whole_wrapper_shadow_forward(model, *args, **kwargs)
    if gradient_policy != "local_context_shadow_v1":
        raise ValueError(f"Unknown organizer gradient policy: {gradient_policy}")
    backend = model.core.backend
    if not hasattr(backend, "organizer"):
        return model(*args, **kwargs)
    if not getattr(backend, "supports_local_context_shadow", False):
        raise ValueError("Backend does not support explicit local-context shadow reductions")
    previous_mode = backend.training_gradient_mode
    previous_permission = backend.permission_mode
    try:
        backend.training_gradient_mode = gradient_policy
        backend.permission_mode = "hard"
        result = model(*args, **kwargs)
    finally:
        backend.training_gradient_mode = previous_mode
        backend.permission_mode = previous_permission
    auxiliary = result.get("interaction_aux", {})
    phase_costs = []
    for key, numerator in auxiliary.items():
        if key.endswith("hypergraph_structural_numerator"):
            denominator = auxiliary.get(key.removesuffix("numerator") + "denominator")
            if torch.is_tensor(numerator) and torch.is_tensor(denominator):
                phase_costs.append(numerator / denominator.clamp_min(1))
    structural = torch.stack(phase_costs).mean() if phase_costs else backend.last_structural_cost
    if torch.is_tensor(structural):
        result["campaign_structural_cost"] = structural
    result["campaign_shadow_calls"] = 0
    result["campaign_gradient_policy"] = gradient_policy
    return result


__all__ = [
    "bridge_local_context",
    "detach_tree",
    "hard_value_soft_hypergraph_forward",
    "local_attention_context",
    "local_modulated_sum",
]
