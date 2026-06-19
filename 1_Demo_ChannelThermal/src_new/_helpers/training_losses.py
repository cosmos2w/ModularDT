"""CHANNELTHERMAL-SPECIFIC training losses.

Inputs are ChannelThermal field predictions, targets, optional point weights,
and loss config dictionaries. Outputs are scalar PyTorch losses for Prompt-3
global-field NewHONF training. This module is specific to ChannelThermal field
channel conventions and legacy loss compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def weighted_field_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: Dict[str, Any],
    point_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    weights = torch.ones(pred.shape[-1], device=pred.device, dtype=pred.dtype)
    if pred.shape[-1] >= 5:
        weights[4] = float(loss_cfg.get("temperature_weight", 1.0))
    if loss_cfg.get("field_channel_weights") is not None:
        custom = torch.as_tensor(loss_cfg["field_channel_weights"], device=pred.device, dtype=pred.dtype)
        weights[: min(custom.numel(), pred.shape[-1])] = custom[: pred.shape[-1]]
    per_value = (pred - target).square() * weights
    if point_weights is None:
        return per_value.mean()
    point_weights = point_weights.to(device=pred.device, dtype=pred.dtype)
    while point_weights.ndim < per_value.ndim:
        point_weights = point_weights.unsqueeze(-1)
    return (per_value * point_weights).sum() / (point_weights.sum() * pred.new_tensor(float(pred.shape[-1]))).clamp_min(1.0e-6)
