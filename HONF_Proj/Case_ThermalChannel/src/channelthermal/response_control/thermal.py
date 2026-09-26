"""Differentiable ThermalChannel functionals from physical role predictions."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from channelthermal.interaction_evidence.types import SolveRecord

from .contracts import DesignInput, RoleQuery

PRESSURE_INLET_BAND_FRACTION = 0.08
PRESSURE_OUTLET_BAND_FRACTION = 0.08
NEAR_INTERFACE_DISTANCE = 0.25


def _channel_index(query: RoleQuery, name: str) -> int:
    try:
        return query.channel_names.index(name)
    except ValueError as exc:
        raise ValueError(f"Role {query.role!r} has no channel named {name!r}.") from exc


def pressure_drop_from_field(
    fluid_values: torch.Tensor,
    query: RoleQuery,
    design: DesignInput,
    context: Mapping[str, object],
) -> torch.Tensor:
    """Use the maintained 8% fluid-only sections with geometry-derived support.

    The fluid mask is recomputed from the current design, query coordinates,
    and module radius. No reference validity mask or solved field enters this
    calculation. Quadrature weights are not used because the maintained scalar
    is an unweighted mean over grid points.
    """

    if query.role != "fluid_fields" or fluid_values.ndim != 2:
        raise ValueError("Pressure drop requires [N,C] fluid_fields predictions.")
    if tuple(fluid_values.shape) != (query.query_features.shape[0], len(query.channel_names)):
        raise ValueError("Fluid predictions and pressure receiver coordinates do not align.")
    if query.query_features.shape[1] < 2:
        raise ValueError("Fluid query coordinates must contain x and y.")
    if query.coordinate_kind != "eulerian":
        raise ValueError("Pressure sections require Eulerian fluid queries.")
    inlet, outlet, _fluid = pressure_section_masks(query, design, context)
    if not bool(inlet.any()) or not bool(outlet.any()):
        raise ValueError("Both maintained pressure-drop bands need fluid query points.")
    pressure = fluid_values[:, _channel_index(query, "p")]
    if not bool(torch.isfinite(pressure[inlet]).all()) or not bool(torch.isfinite(pressure[outlet]).all()):
        raise ValueError("Predicted pressure must be finite in both pressure sections.")
    return pressure[inlet].mean() - pressure[outlet].mean()


def pressure_section_masks(
    query: RoleQuery,
    design: DesignInput,
    context: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact maintained 8% inlet/outlet and current fluid masks."""

    if query.role != "fluid_fields" or query.coordinate_kind != "eulerian":
        raise ValueError("Pressure sections require Eulerian fluid receiver coordinates.")
    if query.query_features.shape[1] < 2:
        raise ValueError("Fluid query coordinates must contain x and y.")
    try:
        length_x = float(context["domain_length_x"])
        radius = float(context["module_radius"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Thermal context needs finite domain_length_x and module_radius.") from exc
    if not torch.isfinite(torch.tensor([length_x, radius])).all() or length_x <= 0.0 or radius <= 0.0:
        raise ValueError("Thermal domain length and module radius must be positive and finite.")
    coordinates = query.query_features
    fluid = torch.ones(coordinates.shape[0], dtype=torch.bool, device=coordinates.device)
    active = design.module_present.bool()
    if bool(active.any()):
        centers = design.module_positions[active].to(dtype=coordinates.dtype, device=coordinates.device)
        distance = torch.cdist(coordinates[:, :2], centers)
        fluid = (distance > radius).all(dim=1)
    x = coordinates[:, 0]
    inlet = fluid & (x <= PRESSURE_INLET_BAND_FRACTION * length_x)
    outlet = fluid & (x >= (1.0 - PRESSURE_OUTLET_BAND_FRACTION) * length_x)
    if not bool(inlet.any()) or not bool(outlet.any()):
        raise ValueError("Both maintained pressure-drop bands need fluid query points.")
    return inlet, outlet, fluid


def module_peak_temperatures_from_role(
    solid_values: torch.Tensor,
    query: RoleQuery,
    design: DesignInput,
) -> dict[int, torch.Tensor]:
    """Compute each active module's true sampled material-temperature peak."""

    if query.role != "solid_temperature" or query.receiver_slots is None:
        raise ValueError("Solid peaks require module-indexed solid_temperature queries.")
    if tuple(solid_values.shape) != (query.query_features.shape[0], len(query.channel_names)):
        raise ValueError("Solid predictions and material receiver queries do not align.")
    if "temperature" in query.channel_names:
        channel = query.channel_names.index("temperature")
    elif len(query.channel_names) == 1:
        channel = 0
    else:
        raise ValueError("Solid-temperature role must identify its temperature channel.")
    slots = torch.as_tensor(query.receiver_slots, dtype=torch.long, device=solid_values.device)
    active_slots = torch.nonzero(design.module_present.bool(), as_tuple=False).reshape(-1).tolist()
    peaks: dict[int, torch.Tensor] = {}
    values = solid_values[:, channel]
    for slot in active_slots:
        selected = slots == int(slot)
        if not bool(selected.any()):
            raise ValueError(f"Active module slot {slot} has no material-temperature queries.")
        module_values = values[selected]
        if not bool(torch.isfinite(module_values).all()):
            raise ValueError(f"Predicted solid temperature is non-finite for module slot {slot}.")
        peaks[int(slot)] = module_values.max()
    if not peaks:
        raise ValueError("At least one active module is required for a solid peak.")
    return peaks


def smooth_module_peak(module_peaks: Mapping[int, torch.Tensor], beta: float) -> torch.Tensor:
    """Normalized log-sum-exp over per-module peaks, in temperature units."""

    if not module_peaks or not torch.isfinite(torch.tensor(beta)) or beta <= 0.0:
        raise ValueError("A nonempty module peak map and positive finite beta are required.")
    values = torch.stack(tuple(module_peaks.values()))
    maximum = values.max()
    return maximum + (torch.logsumexp(beta * (values - maximum), dim=0) - values.new_tensor(float(values.numel())).log()) / beta


def target_module_peak_slots(record: SolveRecord) -> dict[int, float]:
    """Align physical peak labels to design slots without exposing IDs to model."""

    if record.output is None:
        raise ValueError("A solved record is required for peak supervision.")
    by_id = {module.module_id: index for index, module in enumerate(record.design.modules)}
    try:
        return {by_id[module_id]: float(value) for module_id, value in record.output.module_peak_temperature.items()}
    except KeyError as exc:
        raise ValueError("A physical module peak names an unknown design module.") from exc


__all__ = [
    "PRESSURE_INLET_BAND_FRACTION",
    "PRESSURE_OUTLET_BAND_FRACTION",
    "module_peak_temperatures_from_role",
    "pressure_drop_from_field",
    "pressure_section_masks",
    "smooth_module_peak",
    "target_module_peak_slots",
]
