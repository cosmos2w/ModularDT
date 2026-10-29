"""Exact physical ThermalChannel functionals for inverse requests.

Physical design ``D`` defines disk exclusion masks, context ``c`` defines the
domain, request ``R`` selects from exactly seven functionals, compact plan ``G``
is the intended mechanism, and ``G_hat`` is the verified mechanism. Functional
values are computed only from denormalized frozen-HONF outputs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from honf_inverse_core.contracts import FunctionalValue, NamedContext, PhysicalDesign

from .vocabulary import FUNCTIONAL_UNITS, NONREGIONAL_REQUEST_TYPES, REGIONAL_REQUEST_TYPES


INLET_BAND_FRACTION = 0.08
OUTLET_BAND_FRACTION = 0.08


def _finite(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _channel_index(channel_order: Sequence[str], name: str) -> int:
    try:
        return list(channel_order).index(name)
    except ValueError as exc:
        raise ValueError(f"Forward output does not contain required channel {name!r}.") from exc


def module_and_fluid_masks(
    x_grid: Any,
    y_grid: Any,
    design: PhysicalDesign,
    context: NamedContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact disk-union and complementary fluid masks on a physical grid."""

    x = _finite(x_grid, name="x_grid")
    y = _finite(y_grid, name="y_grid")
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x_grid and y_grid must share a two-dimensional shape.")
    radius = context.as_mapping()["module_radius"]
    module_mask = np.zeros(x.shape, dtype=bool)
    for index in np.flatnonzero(design.module_present > 0.5):
        cx, cy = design.module_centers[index]
        module_mask |= np.hypot(x - float(cx), y - float(cy)) <= radius
    return module_mask, ~module_mask


def _selected_value(
    request_type: str,
    values: np.ndarray,
    selection: np.ndarray,
    *,
    reducer: str,
    metadata: Mapping[str, Any] | None = None,
) -> FunctionalValue:
    count = int(np.sum(selection))
    if count <= 0:
        raise ValueError(f"Functional {request_type!r} selected no grid points.")
    selected = values[selection]
    if reducer == "max":
        result = float(np.max(selected))
    elif reducer == "mean":
        result = float(np.mean(selected))
    elif reducer == "std":
        result = float(np.std(selected, ddof=0))
    else:  # pragma: no cover - internal programming error guard.
        raise ValueError(f"Unsupported reducer: {reducer}")
    return FunctionalValue(
        request_type=request_type,
        value=result,
        valid=True,
        selected_count=count,
        units=FUNCTIONAL_UNITS[request_type],
        metadata=dict(metadata or {}),
    )


def evaluate_nonregional_functionals(
    *,
    pred_field_grid: Any,
    x_grid: Any,
    y_grid: Any,
    channel_order: Sequence[str],
    design: PhysicalDesign,
    context: NamedContext,
    pred_internal_temperature: Any,
) -> dict[str, FunctionalValue]:
    """Evaluate the five nonregional schema-v1 functionals."""

    field = _finite(pred_field_grid, name="pred_field_grid")
    x = _finite(x_grid, name="x_grid")
    y = _finite(y_grid, name="y_grid")
    if field.ndim != 3 or field.shape[:2] != x.shape or x.shape != y.shape:
        raise ValueError("pred_field_grid must be [ny,nx,F] aligned to x_grid/y_grid.")
    temperature = field[..., _channel_index(channel_order, "temperature")]
    pressure = field[..., _channel_index(channel_order, "p")]
    _, fluid = module_and_fluid_masks(x, y, design, context)
    if not np.any(fluid):
        raise ValueError("Generated design leaves no fluid grid points.")
    lx = context.as_mapping()["domain_length_x"]
    inlet = fluid & (x <= INLET_BAND_FRACTION * lx)
    outlet = fluid & (x >= (1.0 - OUTLET_BAND_FRACTION) * lx)
    inlet_count = int(np.sum(inlet))
    outlet_count = int(np.sum(outlet))
    if inlet_count <= 0 or outlet_count <= 0:
        raise ValueError("Pressure/outlet functional bands selected no fluid points.")

    internal = _finite(pred_internal_temperature, name="pred_internal_temperature")
    if internal.ndim >= 1 and internal.shape[-1] == 1:
        internal = internal[..., 0]
    if internal.ndim != 2 or internal.shape[0] != design.max_modules:
        raise ValueError("pred_internal_temperature must have shape [M,Q] or [M,Q,1].")
    active = np.flatnonzero(design.module_present > 0.5)
    if active.size == 0 or internal.shape[1] == 0:
        raise ValueError("Internal-temperature functionals require active modules and local query points.")
    per_module_peak = np.max(internal[active], axis=1)

    return {
        "environment_temperature_max": _selected_value(
            "environment_temperature_max", temperature, fluid, reducer="max"
        ),
        "pressure_drop": FunctionalValue(
            request_type="pressure_drop",
            value=float(np.mean(pressure[inlet]) - np.mean(pressure[outlet])),
            valid=True,
            selected_count=inlet_count + outlet_count,
            units=FUNCTIONAL_UNITS["pressure_drop"],
            metadata={"inlet_count": inlet_count, "outlet_count": outlet_count},
        ),
        "outlet_temperature_nonuniformity": _selected_value(
            "outlet_temperature_nonuniformity",
            temperature,
            outlet,
            reducer="std",
            metadata={"ddof": 0},
        ),
        "internal_temperature_max": FunctionalValue(
            request_type="internal_temperature_max",
            value=float(np.max(per_module_peak)),
            valid=True,
            selected_count=int(active.size * internal.shape[1]),
            units=FUNCTIONAL_UNITS["internal_temperature_max"],
            metadata={"active_module_count": int(active.size)},
        ),
        "internal_temperature_spread": FunctionalValue(
            request_type="internal_temperature_spread",
            value=float(np.max(per_module_peak) - np.min(per_module_peak)) if active.size > 1 else 0.0,
            valid=True,
            selected_count=int(active.size),
            units=FUNCTIONAL_UNITS["internal_temperature_spread"],
            metadata={"active_module_count": int(active.size)},
        ),
    }


def evaluate_regional_functional(
    request_type: str,
    region: Sequence[float],
    *,
    pred_field_grid: Any,
    x_grid: Any,
    y_grid: Any,
    channel_order: Sequence[str],
    design: PhysicalDesign,
    context: NamedContext,
) -> FunctionalValue:
    """Evaluate one normalized rectangular regional-temperature functional."""

    if request_type not in REGIONAL_REQUEST_TYPES:
        raise ValueError(f"Unsupported regional request type: {request_type!r}")
    region_array = _finite(region, name="region").reshape(-1)
    if region_array.shape != (4,) or np.any(region_array < 0.0) or np.any(region_array > 1.0):
        raise ValueError("Region must contain four normalized values in [0,1].")
    x0, y0, x1, y1 = region_array
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Region must have positive width and height.")
    field = _finite(pred_field_grid, name="pred_field_grid")
    x = _finite(x_grid, name="x_grid")
    y = _finite(y_grid, name="y_grid")
    if field.ndim != 3 or field.shape[:2] != x.shape or x.shape != y.shape:
        raise ValueError("pred_field_grid must be [ny,nx,F] aligned to x_grid/y_grid.")
    temperature = field[..., _channel_index(channel_order, "temperature")]
    _, fluid = module_and_fluid_masks(x, y, design, context)
    values = context.as_mapping()
    physical = np.asarray(
        [x0 * values["domain_length_x"], y0 * values["domain_length_y"],
         x1 * values["domain_length_x"], y1 * values["domain_length_y"]],
        dtype=np.float64,
    )
    selection = (
        fluid
        & (x >= physical[0])
        & (x <= physical[2])
        & (y >= physical[1])
        & (y <= physical[3])
    )
    return _selected_value(
        request_type,
        temperature,
        selection,
        reducer="mean" if request_type == "regional_temperature_mean" else "max",
        metadata={"region_normalized": region_array.tolist(), "region_physical": physical.tolist()},
    )


def evaluate_supported_functionals(
    *,
    pred_field_grid: Any,
    x_grid: Any,
    y_grid: Any,
    channel_order: Sequence[str],
    design: PhysicalDesign,
    context: NamedContext,
    pred_internal_temperature: Any,
    regional_requests: Sequence[tuple[str, Sequence[float]]] = (),
) -> dict[str, FunctionalValue]:
    """Evaluate all five global values and explicitly requested regional values."""

    values = evaluate_nonregional_functionals(
        pred_field_grid=pred_field_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        channel_order=channel_order,
        design=design,
        context=context,
        pred_internal_temperature=pred_internal_temperature,
    )
    for request_type, region in regional_requests:
        if request_type in values:
            raise ValueError(f"Duplicate functional request: {request_type}")
        values[request_type] = evaluate_regional_functional(
            request_type,
            region,
            pred_field_grid=pred_field_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            channel_order=channel_order,
            design=design,
            context=context,
        )
    if set(values) - set(NONREGIONAL_REQUEST_TYPES) - set(REGIONAL_REQUEST_TYPES):  # pragma: no cover
        raise AssertionError("Evaluator returned an unsupported request type.")
    return values


__all__ = [
    "INLET_BAND_FRACTION",
    "OUTLET_BAND_FRACTION",
    "evaluate_nonregional_functionals",
    "evaluate_regional_functional",
    "evaluate_supported_functionals",
    "module_and_fluid_masks",
]
