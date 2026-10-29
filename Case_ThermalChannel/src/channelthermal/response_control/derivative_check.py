"""Whole-wrapper automatic/finite-difference checks for physical responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch

from channelthermal.interaction_evidence.response_dataset import ResponseStencil

from .contracts import AbsoluteOperator, DesignInput, context_inputs, role_queries_from_stencil
from .thermal import module_peak_temperatures_from_role, pressure_drop_from_field, pressure_section_masks


def check_pressure_peak_ad_fd(
    operator: AbsoluteOperator,
    stencil: ResponseStencil,
    *,
    variable: Literal["heating", "position_x", "position_y"] = "heating",
    module_slot: int = 0,
    step: float | None = None,
    relative_tolerance: float = 0.25,
    absolute_tolerance: float = 1.0e-4,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Compare AD and centered FD through the complete native absolute operator.

    Pressure comparisons require exactly unchanged Eulerian 8% support. The
    solid maximum comparison records the sampled maximizing receiver at both
    FD sides; a changed maximizer reports a nonsmoothness warning even when
    the scalar derivative comparison is close.
    """

    if relative_tolerance <= 0.0 or absolute_tolerance < 0.0:
        raise ValueError("Derivative tolerances must be positive/nonnegative.")
    base = DesignInput.from_state(stencil.baseline.design, device=device, requires_grad=False)
    if not 0 <= module_slot < base.module_positions.shape[0]:
        raise ValueError("module_slot is outside the baseline design.")
    if variable == "heating":
        target = base.module_heating.detach().clone().requires_grad_(True)
        base_design = DesignInput(base.module_positions, target, base.module_present)
        if step is None:
            step = max(abs(float(target[module_slot].detach().cpu())) * 1.0e-3, 1.0e-5)
        if float(target[module_slot].detach().cpu()) <= float(step):
            raise ValueError("Centered heating differences must remain in positive support.")
        target_index = (module_slot,)
    elif variable in {"position_x", "position_y"}:
        target = base.module_positions.detach().clone().requires_grad_(True)
        base_design = DesignInput(target, base.module_heating, base.module_present)
        coordinate = 0 if variable == "position_x" else 1
        if step is None:
            step = 1.0e-3
        target_index = (module_slot, coordinate)
    else:  # pragma: no cover - guarded by Literal for typed callers
        raise ValueError(f"Unsupported finite-difference variable {variable!r}.")
    if not torch.isfinite(torch.tensor(float(step))) or float(step) <= 0.0:
        raise ValueError("Finite-difference step must be positive and finite.")

    role_queries = role_queries_from_stencil(stencil, device=device, dtype=base.module_positions.dtype)
    context: Mapping[str, Any] = context_inputs(stencil.baseline.context)
    fluid_query = role_queries["fluid_fields"]
    solid_query = role_queries["solid_temperature"]

    def values_for(design: DesignInput) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        prediction = operator(design, context, role_queries)
        pressure = pressure_drop_from_field(
            prediction.role_values["fluid_fields"], fluid_query, design, context
        )
        peaks = module_peak_temperatures_from_role(
            prediction.role_values["solid_temperature"], solid_query, design
        )
        peak_values = torch.stack([peaks[key] for key in sorted(peaks)])
        slots = torch.as_tensor(solid_query.receiver_slots, device=peak_values.device, dtype=torch.long)
        channel = solid_query.channel_names.index("temperature") if "temperature" in solid_query.channel_names else 0
        material_values = prediction.role_values["solid_temperature"][:, channel]
        local_peak_row = int(torch.argmax(material_values).detach().cpu())
        local_peak_slot = int(slots[local_peak_row].detach().cpu())
        local_rank = int(torch.nonzero(slots == local_peak_slot, as_tuple=False).reshape(-1).tolist().index(local_peak_row))
        return pressure, peak_values.max(), (local_peak_slot, local_rank)

    pressure_base, peak_base, peak_location_base = values_for(base_design)
    pressure_gradient = torch.autograd.grad(
        pressure_base, target, retain_graph=True, allow_unused=False
    )[0]
    peak_gradient = torch.autograd.grad(
        peak_base, target, retain_graph=False, allow_unused=False
    )[0]
    pressure_ad = float(pressure_gradient[target_index].detach().cpu())
    peak_ad = float(peak_gradient[target_index].detach().cpu())

    if variable == "heating":
        plus_var = base.module_heating.detach().clone()
        minus_var = base.module_heating.detach().clone()
        plus_var[module_slot] += float(step)
        minus_var[module_slot] -= float(step)
        plus_design = DesignInput(base.module_positions, plus_var, base.module_present)
        minus_design = DesignInput(base.module_positions, minus_var, base.module_present)
    else:
        plus_var = base.module_positions.detach().clone()
        minus_var = base.module_positions.detach().clone()
        plus_var[module_slot, target_index[1]] += float(step)
        minus_var[module_slot, target_index[1]] -= float(step)
        plus_design = DesignInput(plus_var, base.module_heating, base.module_present)
        minus_design = DesignInput(minus_var, base.module_heating, base.module_present)

    with torch.no_grad():
        pressure_plus, peak_plus, peak_location_plus = values_for(plus_design)
        pressure_minus, peak_minus, peak_location_minus = values_for(minus_design)
    fd_pressure = float(((pressure_plus - pressure_minus) / (2.0 * float(step))).detach().cpu())
    fd_peak = float(((peak_plus - peak_minus) / (2.0 * float(step))).detach().cpu())
    pressure_masks = []
    for design in (base_design, plus_design, minus_design):
        inlet, outlet, _ = pressure_section_masks(fluid_query, design, context)
        pressure_masks.append((inlet.detach().cpu(), outlet.detach().cpu()))
    support_unchanged = all(
        torch.equal(pressure_masks[0][index], pressure_masks[side][index])
        for side in (1, 2)
        for index in (0, 1)
    )
    peak_location_stable = peak_location_plus == peak_location_minus

    def comparison(ad: float, fd: float) -> dict[str, float | bool]:
        error = abs(ad - fd)
        allowed = float(absolute_tolerance) + float(relative_tolerance) * abs(fd)
        return {"autodiff": ad, "finite_difference": fd, "absolute_error": error, "allowed_error": allowed, "passed": error <= allowed}

    pressure_comparison = comparison(pressure_ad, fd_pressure)
    peak_comparison = comparison(peak_ad, fd_peak)
    return {
        "variable": variable,
        "module_slot": int(module_slot),
        "step": float(step),
        "pressure_drop": pressure_comparison,
        "solid_sampled_peak": peak_comparison,
        "pressure_support_unchanged": support_unchanged,
        "pressure_inlet_count": int(pressure_masks[0][0].sum()),
        "pressure_outlet_count": int(pressure_masks[0][1].sum()),
        "solid_peak_location_base": peak_location_base,
        "solid_peak_location_plus": peak_location_plus,
        "solid_peak_location_minus": peak_location_minus,
        "solid_peak_location_stable_across_fd_sides": peak_location_stable,
        "passed": bool(
            pressure_comparison["passed"]
            and peak_comparison["passed"]
            and support_unchanged
            and peak_location_stable
        ),
        "limits": [
            "sampled solid maxima are piecewise differentiable and can move between receivers",
            "pressure sections are discrete masks and this check requires their support to remain fixed",
            "the check validates the full model and inverse-normalization wrapper locally, not physical accuracy",
        ],
    }


__all__ = ["check_pressure_peak_ad_fd"]
