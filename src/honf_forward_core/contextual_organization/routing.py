"""Typed nullable group and original-record reads for contextual organization."""
from __future__ import annotations

import torch
from torch import Tensor


def positive_gate(logits: Tensor) -> Tensor:
    """C1 nonnegative gate used by direct membership and group access."""
    positive = logits.clamp_min(0.0)
    square = positive.square()
    return square / (1.0 + square)


def quintic_window(distance: Tensor, outer_radius: Tensor, shell_width: Tensor) -> Tensor:
    """Compact C2 support with the declared quintic transition shell."""
    valid_radius = outer_radius > 0
    outer = outer_radius.clamp_min(torch.finfo(distance.dtype).eps)
    shell = shell_width.clamp_min(torch.finfo(distance.dtype).eps)
    inner = (outer - shell).clamp_min(0.0)
    t = ((distance - inner) / shell).clamp(0.0, 1.0)
    transition = 1.0 - 10.0 * t.pow(3) + 15.0 * t.pow(4) - 6.0 * t.pow(5)
    window = torch.where(distance <= inner, torch.ones_like(distance),
                         torch.where(distance >= outer, torch.zeros_like(distance), transition))
    return window * valid_radius.to(window.dtype)


def smooth_union(windows: Tensor, dim: int = -1) -> Tensor:
    """Smooth union ``1 - product(1 - W_i)`` with an exact empty result."""
    if windows.shape[dim] == 0:
        shape = list(windows.shape)
        del shape[dim]
        return windows.new_zeros(shape)
    return 1.0 - (1.0 - windows).prod(dim=dim)


def null_normalized_weights(gates: Tensor, information_measure: Tensor, scores: Tensor) -> Tensor:
    """Normalize selected original-record weights with a fixed null atom.

    Inputs use ``[..., N]`` axes and must already contain selected records only.
    Empty reads therefore return an exact empty tensor and never request a
    score or value row for an unselected original record.
    """
    if gates.shape != scores.shape:
        raise ValueError("Selected gates and scores must have the same shape.")
    if information_measure.ndim == gates.ndim - 1:
        information_measure = information_measure.unsqueeze(-2)
    try:
        measure = torch.broadcast_to(information_measure, gates.shape)
    except RuntimeError as exc:
        raise ValueError("Information measures do not broadcast to selected gates.") from exc
    if gates.shape[-1] == 0:
        return gates
    valid = (gates > 0) & (measure > 0)
    log_weight = (torch.log(gates.clamp_min(torch.finfo(gates.dtype).tiny))
                  + torch.log(measure.clamp_min(torch.finfo(measure.dtype).tiny)) + scores)
    log_weight = torch.where(valid, log_weight, torch.full_like(log_weight, -torch.inf))
    log_null = torch.zeros_like(log_weight[..., :1])
    denominator = torch.logsumexp(torch.cat((log_null, log_weight), dim=-1), dim=-1, keepdim=True)
    return torch.where(valid, torch.exp(log_weight - denominator), torch.zeros_like(log_weight))


def null_normalized_group_access(mass: Tensor, window: Tensor, scores: Tensor) -> Tensor:
    """Compute independent typed group access with null mass one."""
    if mass.shape != scores.shape or window.shape != scores.shape:
        raise ValueError("Group mass, support window, and score shapes must agree.")
    if scores.shape[-1] == 0:
        return scores
    omega = mass.clamp_min(0.0) * window.clamp(0.0, 1.0) * positive_gate(scores)
    return omega / (1.0 + omega.sum(dim=-1, keepdim=True))
