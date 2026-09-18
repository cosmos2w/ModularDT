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


def induced_pair_cost_loss(
    output: Mapping[str, Any],
    *,
    enabled: bool,
    require_components: bool = False,
) -> torch.Tensor:
    """Return the live Eq. (18) induced fine-pair cost surrogate.

    The routed executor supplies a numerator and denominator for each physical
    P0/P1/P2 read.  Numerators already contain the configured fixed module and
    environment marginal costs; only their ratio is formed here.  Prefixes
    such as ``initial_`` and ``provisional_`` identify the other physical
    reads collected by the ChannelThermal coupling, so all available reads
    contribute once to the objective.  The fallback to a direct scalar keeps
    small backend fixtures usable while requiring the canonical components
    for an enabled managed run.
    """

    pred = output.get("pred_field")
    if not torch.is_tensor(pred):
        raise ValueError("Pair-cost loss requires output['pred_field'] tensor metadata.")
    if not enabled:
        return pred.new_zeros(())
    aux = output.get("interaction_aux")
    if not isinstance(aux, Mapping):
        if require_components:
            raise RuntimeError(
                "Enabled induced pair-cost objective requires routed interaction_aux components."
            )
        return pred.new_zeros(())

    numerators: list[torch.Tensor] = []
    denominators: list[torch.Tensor] = []
    direct_values: list[torch.Tensor] = []
    for key, value in aux.items():
        if not torch.is_tensor(value) or value.numel() == 0:
            continue
        key = str(key)
        if key.endswith("routing_paircost_numerator"):
            numerators.append(value)
        elif key.endswith("routing_paircost_denominator"):
            denominators.append(value)
        elif key.endswith("routing_paircost"):
            direct_values.append(value)
    if numerators or denominators:
        if len(numerators) != len(denominators):
            raise RuntimeError(
                "Induced pair-cost numerator/denominator components are incomplete: "
                f"{len(numerators)} numerator(s), {len(denominators)} denominator(s)."
            )
        # Backends may return one scalar per batch case or one scalar for the
        # complete read.  Reduce each component to the Eq. 18 numerator before
        # combining P0/P1/P2; this keeps the public aux contract independent
        # of batch packing while retaining every live numerator edge.
        numerator = torch.stack([value.sum() for value in numerators]).sum()
        denominator = torch.stack([value.sum() for value in denominators]).sum().detach()
        # Valid-source masks and strictly positive configured marginal costs
        # make this denominator positive by construction.  Keep the fallback
        # tensor-only so ordinary optimizer batches do not synchronize the
        # host merely to inspect a scalar diagnostic.
        safe_denominator = torch.where(
            torch.isfinite(denominator) & (denominator > 0.0),
            denominator,
            denominator.new_ones(()),
        )
        return numerator / safe_denominator.to(device=numerator.device, dtype=numerator.dtype)
    if direct_values:
        if require_components:
            raise RuntimeError(
                "Enabled induced pair-cost objective requires numerator/denominator components; "
                "a direct scalar is reporting-only."
            )
        return torch.stack([value.mean() for value in direct_values]).mean()
    if require_components:
        raise RuntimeError(
            "Enabled induced pair-cost objective did not receive routed pair-cost components."
        )
    return pred.new_zeros(())


__all__ = [
    "channelthermal_field_channel_weights",
    "channelthermal_field_mse",
    "induced_pair_cost_loss",
]
