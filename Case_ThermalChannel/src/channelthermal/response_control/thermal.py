"""Differentiable ThermalChannel functionals from physical role predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch

from channelthermal.interaction_evidence.types import SolveRecord

from .contracts import AbsolutePrediction, DesignInput, RoleQuery

PRESSURE_INLET_BAND_FRACTION = 0.08
PRESSURE_OUTLET_BAND_FRACTION = 0.08
NEAR_INTERFACE_DISTANCE = 0.25


@dataclass(frozen=True)
class NativeThermalQuantities:
    """Differentiable physical scalars reduced from one absolute prediction."""

    pressure_drop: torch.Tensor
    pressure_drop_units: str
    module_peak_temperature: Mapping[str, torch.Tensor]
    temperature_units: str

    def __post_init__(self) -> None:
        if not isinstance(self.pressure_drop, torch.Tensor) or self.pressure_drop.numel() != 1:
            raise TypeError("Native pressure drop must be a scalar torch tensor.")
        peaks = dict(self.module_peak_temperature)
        if not peaks or any(
            not isinstance(value, torch.Tensor) or value.numel() != 1
            for value in peaks.values()
        ):
            raise TypeError("Native module peaks must be a nonempty mapping of scalar tensors.")
        if not self.pressure_drop_units or not self.temperature_units:
            raise ValueError("Native scalar units must be declared by their role schemas.")
        object.__setattr__(self, "module_peak_temperature", MappingProxyType(peaks))


def reduce_native_thermal_quantities(
    prediction: AbsolutePrediction,
    design: DesignInput,
    role_queries: Mapping[str, RoleQuery],
    context: Mapping[str, object],
    *,
    module_ids: Sequence[str],
    solid_valid_mask: np.ndarray | torch.Tensor | None = None,
) -> NativeThermalQuantities:
    """Reduce maintained pressure and material-temperature quantities.

    ``module_ids`` names the design slots in their existing order. The
    optional solid mask is output-side receiver validity metadata; it is used
    only for the peak reduction and is never passed into the model. When the
    selected material query universe is already valid, leave it unset. All
    reductions remain in the prediction's physical units and retain autograd.
    """

    required_roles = {"fluid_fields", "solid_temperature"}
    if not required_roles.issubset(role_queries) or not required_roles.issubset(prediction.role_values):
        raise ValueError("Native quantity reduction requires fluid_fields and solid_temperature roles.")
    fluid_query = role_queries["fluid_fields"]
    solid_query = role_queries["solid_temperature"]
    fluid_values = prediction.role_values["fluid_fields"]
    solid_values = prediction.role_values["solid_temperature"]
    if fluid_values.device != fluid_query.query_features.device:
        raise ValueError("Fluid prediction and query coordinates must share a device.")
    if solid_values.device != solid_query.query_features.device:
        raise ValueError("Solid prediction and material coordinates must share a device.")

    pressure_drop = pressure_drop_from_field(fluid_values, fluid_query, design, context)
    pressure_channel = _channel_index(fluid_query, "p")
    pressure_units = fluid_query.channel_units[pressure_channel]

    if solid_query.role != "solid_temperature" or solid_query.receiver_slots is None:
        raise ValueError("Material peaks require module-indexed solid_temperature queries.")
    if tuple(solid_values.shape) != (
        solid_query.query_features.shape[0], len(solid_query.channel_names)
    ):
        raise ValueError("Solid predictions and material receiver queries do not align.")
    temperature_channel = (
        solid_query.channel_names.index("temperature")
        if "temperature" in solid_query.channel_names
        else 0 if len(solid_query.channel_names) == 1
        else -1
    )
    if temperature_channel < 0:
        raise ValueError("Solid-temperature role must identify its temperature channel.")
    temperature_units = solid_query.channel_units[temperature_channel]

    if len(module_ids) != int(design.module_positions.shape[0]):
        raise ValueError("module_ids must name every design slot in its existing order.")
    names = tuple(str(value) for value in module_ids)
    active = design.module_present.detach().bool().cpu().tolist()
    active_names = [names[index] for index, enabled in enumerate(active) if enabled]
    if any(not name for name in active_names) or len(active_names) != len(set(active_names)):
        raise ValueError("Every active design slot needs a unique nonempty physical module ID.")

    slots = torch.as_tensor(
        solid_query.receiver_slots, dtype=torch.long, device=solid_values.device
    )
    if slots.numel() and (int(slots.min()) < 0 or int(slots.max()) >= len(names)):
        raise ValueError("Solid receiver slots must index the supplied physical module IDs.")
    if solid_valid_mask is None:
        valid = torch.ones_like(solid_values, dtype=torch.bool)
    elif isinstance(solid_valid_mask, torch.Tensor):
        valid = solid_valid_mask.to(device=solid_values.device, dtype=torch.bool)
    else:
        valid = torch.as_tensor(
            np.array(solid_valid_mask, copy=True), device=solid_values.device, dtype=torch.bool
        )
    if valid.ndim == 1:
        valid = valid[:, None].expand_as(solid_values)
    if tuple(valid.shape) != tuple(solid_values.shape):
        raise ValueError("Solid output validity mask must align with [N,C] material predictions.")

    material_temperature = solid_values[:, temperature_channel]
    temperature_valid = valid[:, temperature_channel]
    peaks: dict[str, torch.Tensor] = {}
    for slot, (module_id, enabled) in enumerate(zip(names, active, strict=True)):
        if not enabled:
            continue
        selected = (slots == slot) & temperature_valid
        if not bool(selected.any()):
            raise ValueError(f"Active module {module_id!r} has no valid material-temperature queries.")
        values = material_temperature[selected]
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"Predicted material temperature is non-finite for module {module_id!r}.")
        peaks[module_id] = values.max()

    return NativeThermalQuantities(
        pressure_drop=pressure_drop,
        pressure_drop_units=pressure_units,
        module_peak_temperature=peaks,
        temperature_units=temperature_units,
    )


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
    "NativeThermalQuantities",
    "module_peak_temperatures_from_role",
    "pressure_drop_from_field",
    "pressure_section_masks",
    "reduce_native_thermal_quantities",
    "smooth_module_peak",
    "target_module_peak_slots",
]
