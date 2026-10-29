"""Case-neutral losses for generic multi-channel HONF outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch


def weighted_channel_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_weights: torch.Tensor | Sequence[float],
    point_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return MSE weighted by explicit channels and optional sample/query weights.

    The final tensor axis is treated only as an ordered channel axis. The caller
    must provide exactly one weight for every channel; this function knows no
    field names, case configuration keys, or distinguished channel indices.
    Channel weights scale errors but do not renormalize the mean, preserving the
    usual ``mean(weight * squared_error)`` convention.
    """

    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes must match, got {tuple(pred.shape)} and {tuple(target.shape)}."
        )
    if pred.ndim < 1:
        raise ValueError("Prediction and target tensors must have a channel axis.")
    weights = torch.as_tensor(channel_weights, device=pred.device, dtype=pred.dtype)
    if weights.ndim != 1 or int(weights.numel()) != int(pred.shape[-1]):
        raise ValueError(
            "channel_weights must be a one-dimensional vector with exactly one "
            f"entry per output channel ({pred.shape[-1]}), got shape {tuple(weights.shape)}."
        )
    per_value = (pred - target).square() * weights
    if point_weights is None:
        return per_value.mean()
    point_weights = torch.as_tensor(point_weights, device=pred.device, dtype=pred.dtype)
    if point_weights.ndim == pred.ndim and point_weights.shape[-1] == 1:
        point_weights = point_weights.squeeze(-1)
    while point_weights.ndim < pred.ndim - 1:
        point_weights = point_weights.unsqueeze(-1)
    try:
        point_weights = torch.broadcast_to(point_weights, pred.shape[:-1])
    except RuntimeError as error:
        raise ValueError(
            f"point_weights shape {tuple(point_weights.shape)} is not broadcastable "
            f"to prediction sample axes {tuple(pred.shape[:-1])}."
        ) from error
    weighted_sum = (per_value * point_weights.unsqueeze(-1)).sum()
    denominator = point_weights.sum() * pred.new_tensor(float(pred.shape[-1]))
    return weighted_sum / denominator.clamp_min(1.0e-6)


__all__ = ["weighted_channel_mse"]
