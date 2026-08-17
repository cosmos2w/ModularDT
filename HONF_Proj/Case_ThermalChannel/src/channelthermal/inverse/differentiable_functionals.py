"""Smooth training surrogates for the seven request functionals.

Generated physical design ``D`` and context ``c`` are passed through frozen
HONF. Request ``R`` selects differentiable probes; compact plan ``G`` and
realized ``G_hat`` are compared separately. Exact evaluation remains the
release authority—these smooth reductions exist only for sparse stage-four
gradients.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def smooth_max(values: torch.Tensor, *, temperature: float = 0.05) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("smooth_max received no values.")
    scale = max(float(temperature), 1.0e-6)
    return torch.logsumexp(values / scale, dim=0) * scale


def functional_token_values(
    *,
    pred_field: torch.Tensor,
    query_xy: torch.Tensor,
    pred_internal_temperature: torch.Tensor,
    module_centers: torch.Tensor,
    module_present: torch.Tensor,
    module_radius: torch.Tensor,
    domain_length_x: torch.Tensor,
    domain_length_y: torch.Tensor,
    request: Mapping[str, torch.Tensor],
    temperature_index: int = 4,
    pressure_index: int = 2,
) -> torch.Tensor:
    """Return smooth physical values `[B,L]` for active request tokens."""

    batch, tokens = request["type_id"].shape
    values = pred_field.new_zeros((batch, tokens))
    internal = pred_internal_temperature[..., 0] if pred_internal_temperature.shape[-1] == 1 else pred_internal_temperature
    for row in range(batch):
        x = query_xy[row, :, 0]
        y = query_xy[row, :, 1]
        temperature = pred_field[row, :, temperature_index]
        pressure = pred_field[row, :, pressure_index]
        fluid = torch.ones_like(x, dtype=torch.bool)
        for module in torch.nonzero(module_present[row] > 0.5, as_tuple=False).reshape(-1):
            distance = torch.sqrt(
                (x - module_centers[row, module, 0]).square()
                + (y - module_centers[row, module, 1]).square()
            )
            fluid = fluid & (distance > module_radius[row])
        inlet = fluid & (x <= 0.08 * domain_length_x[row])
        outlet = fluid & (x >= 0.92 * domain_length_x[row])
        active_modules = module_present[row] > 0.5
        internal_peaks = torch.logsumexp(internal[row, active_modules] / 0.05, dim=-1) * 0.05
        for slot in range(tokens):
            if float(request["active_mask"][row, slot]) < 0.5:
                continue
            request_type = int(request["type_id"][row, slot])
            if request_type == 0:
                selected = temperature[fluid]
                values[row, slot] = smooth_max(selected)
            elif request_type == 1:
                values[row, slot] = pressure[inlet].mean() - pressure[outlet].mean()
            elif request_type == 2:
                values[row, slot] = temperature[outlet].std(unbiased=False)
            elif request_type == 3:
                values[row, slot] = smooth_max(internal_peaks)
            elif request_type == 4:
                values[row, slot] = smooth_max(internal_peaks) - (-smooth_max(-internal_peaks))
            elif request_type in {5, 6}:
                x0, y0, x1, y1 = request["region"][row, slot]
                region = (
                    fluid
                    & (x >= x0 * domain_length_x[row])
                    & (x <= x1 * domain_length_x[row])
                    & (y >= y0 * domain_length_y[row])
                    & (y <= y1 * domain_length_y[row])
                )
                selected = temperature[region]
                if selected.numel() == 0:
                    raise ValueError("Differentiable regional probe selected no points.")
                values[row, slot] = selected.mean() if request_type == 5 else smooth_max(selected)
            else:
                raise ValueError(f"Unsupported differentiable request type ID: {request_type}")
    return values


def normalized_request_residuals(
    values_raw: torch.Tensor,
    request: Mapping[str, torch.Tensor],
    functional_mean: torch.Tensor,
    functional_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token normalized violations and their weighted mean."""

    type_id = request["type_id"].long().clamp_min(0)
    mean = functional_mean[type_id]
    std = functional_std[type_id].clamp_min(1.0e-8)
    value = (values_raw - mean) / std
    target = request["target_normalized"].float()
    tolerance = request["tolerance_normalized"].float()
    ranges = request["range_normalized"].float()
    relation = request["relation_id"].long()
    upper = F.relu(value - target - tolerance)
    lower = F.relu(target - value - tolerance)
    target_range = F.relu(value - ranges[..., 1] - tolerance) - F.relu(
        ranges[..., 0] - tolerance - value
    )
    minimize = F.softplus(value)
    residual = torch.where(
        relation == 0,
        upper,
        torch.where(relation == 1, -lower, torch.where(relation == 2, target_range, minimize)),
    )
    active = request["active_mask"].float()
    weight = request["weight"].float() * active
    loss = (residual.square() * weight).sum() / weight.sum().clamp_min(1.0)
    return residual * active, loss


__all__ = ["functional_token_values", "normalized_request_residuals", "smooth_max"]
