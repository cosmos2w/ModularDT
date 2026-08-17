"""ChannelThermal-specific global-field loss policy.

This module owns the meaning of named physical output channels and translates
the case loss configuration into a complete ordered weight vector. The reusable
HONF core receives only that vector and therefore has no knowledge of
temperature or any other physical channel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from honf_forward_core.training.losses import weighted_channel_mse


def channelthermal_field_channel_weights(
    field_names: Sequence[str],
    loss_cfg: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resolve case settings into one explicit weight per named field channel."""

    names = tuple(str(name) for name in field_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("ChannelThermal field_names must be nonempty and unique.")
    weights = torch.ones(len(names), device=device, dtype=dtype)
    if "temperature" not in names:
        raise ValueError("ChannelThermal global fields must include the named 'temperature' channel.")
    weights[names.index("temperature")] = float(loss_cfg.get("temperature_weight", 1.0))

    configured = loss_cfg.get("field_channel_weights")
    if configured is not None:
        explicit = torch.as_tensor(configured, device=device, dtype=dtype)
        if explicit.ndim != 1 or int(explicit.numel()) != len(names):
            raise ValueError(
                "loss.field_channel_weights must provide exactly one value for each "
                f"ChannelThermal field ({len(names)}), got shape {tuple(explicit.shape)}."
            )
        weights = explicit
    return weights


def channelthermal_field_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: Mapping[str, Any],
    *,
    field_names: Sequence[str],
    point_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply ChannelThermal channel policy through the case-neutral core loss."""

    channel_weights = channelthermal_field_channel_weights(
        field_names,
        loss_cfg,
        device=pred.device,
        dtype=pred.dtype,
    )
    return weighted_channel_mse(pred, target, channel_weights, point_weights)


__all__ = ["channelthermal_field_channel_weights", "channelthermal_field_mse"]
