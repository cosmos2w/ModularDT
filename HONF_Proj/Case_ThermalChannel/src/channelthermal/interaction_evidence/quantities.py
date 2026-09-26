"""Physical quantities and response summaries with explicit sampling rules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

CHANNEL_ORDER = ("u", "v", "p", "omega", "temperature")
INLET_BAND_FRACTION = 0.08
OUTLET_BAND_FRACTION = 0.08


def pressure_drop_8pct(
    pressure: np.ndarray,
    x_grid: np.ndarray,
    fluid_mask: np.ndarray,
    domain_length_x: float,
) -> tuple[float, dict[str, int]]:
    """Use the maintained inverse functional's fluid-only 8% inlet/outlet bands."""

    p = np.asarray(pressure, dtype=np.float64)
    x = np.asarray(x_grid, dtype=np.float64)
    fluid = np.asarray(fluid_mask, dtype=bool)
    if p.ndim != 2 or p.shape != x.shape or p.shape != fluid.shape:
        raise ValueError("pressure, x_grid, and fluid_mask must share shape [ny,nx].")
    if not np.isfinite(p[fluid]).all() or not np.isfinite(x).all():
        raise ValueError("Pressure on fluid points and x_grid must be finite.")
    if not np.isfinite(domain_length_x) or domain_length_x <= 0.0:
        raise ValueError("domain_length_x must be positive and finite.")
    inlet = fluid & (x <= INLET_BAND_FRACTION * float(domain_length_x))
    outlet = fluid & (x >= (1.0 - OUTLET_BAND_FRACTION) * float(domain_length_x))
    inlet_count = int(inlet.sum())
    outlet_count = int(outlet.sum())
    if inlet_count == 0 or outlet_count == 0:
        raise ValueError("Both maintained pressure-drop bands need fluid points.")
    value = float(np.mean(p[inlet]) - np.mean(p[outlet]))
    return value, {"inlet_count": inlet_count, "outlet_count": outlet_count}


def module_peak_temperatures(
    internal_temperature: np.ndarray,
    receiver_module_ids: Sequence[str],
    valid_mask: np.ndarray,
) -> dict[str, float]:
    """Return one material-coordinate maximum for each represented active module."""

    values = np.asarray(internal_temperature, dtype=np.float64)
    if values.ndim == 2:
        if values.shape[1] != 1:
            raise ValueError("Internal temperatures must be [S,1] or [M,S].")
        values = values[:, 0]
    ids = np.asarray(receiver_module_ids, dtype=object)
    mask = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 1 or ids.shape != values.shape:
        raise ValueError("One module ID is required for every internal sample.")
    if mask.ndim == 2:
        if mask.shape[1] != 1:
            raise ValueError("Internal validity mask must have one target channel.")
        mask = mask[:, 0]
    if mask.shape != values.shape:
        raise ValueError("Internal validity mask does not match the samples.")
    result: dict[str, float] = {}
    for module_id in dict.fromkeys(str(value) for value in ids):
        selected = (ids == module_id) & mask
        if not np.any(selected):
            raise ValueError(f"Active module {module_id!r} has no resolved internal samples.")
        result[module_id] = float(np.max(values[selected]))
    return result


def smooth_peak_temperature(values: np.ndarray, beta: float) -> float:
    """Stable log-sum-exp peak proxy; beta and input units must be reported."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("A finite nonempty array and positive beta are required.")
    maximum = float(np.max(array))
    return float(maximum + np.log(np.mean(np.exp(beta * (array - maximum)))) / beta)


def field_response_norms(
    baseline: np.ndarray,
    trial: np.ndarray,
    common_mask: np.ndarray,
    channel_names: Sequence[str] = CHANNEL_ORDER,
) -> dict[str, dict[str, float | int]]:
    """Per-channel absolute response norms and the stated relative-L2 ratio.

    ``relative_l2`` is ``||trial - baseline||_2 / ||baseline||_2`` over the
    exact common mask; it is undefined when the masked baseline norm is zero.
    """

    base = np.asarray(baseline, dtype=np.float64)
    changed = np.asarray(trial, dtype=np.float64)
    mask = np.asarray(common_mask, dtype=bool)
    if base.shape != changed.shape or base.ndim != 2:
        raise ValueError("baseline and trial must share shape [N,C].")
    if mask.ndim == 1:
        mask = np.broadcast_to(mask[:, None], base.shape)
    if mask.shape != base.shape or len(channel_names) != base.shape[1]:
        raise ValueError("common_mask or channel_names is not aligned with field arrays.")
    output: dict[str, dict[str, float | int]] = {}
    for channel, name in enumerate(channel_names):
        selected = mask[:, channel]
        if not np.any(selected):
            output[str(name)] = {"count": 0, "response_l2": 0.0, "response_rms": 0.0,
                                 "response_max_abs": 0.0, "relative_l2": float("nan")}
            continue
        before = base[selected, channel]
        delta = changed[selected, channel] - before
        reference_norm = float(np.linalg.norm(before))
        output[str(name)] = {
            "count": int(selected.sum()),
            "response_l2": float(np.linalg.norm(delta)),
            "response_rms": float(np.sqrt(np.mean(np.square(delta)))),
            "response_max_abs": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta) / reference_norm)
            if reference_norm > 0.0 else float("nan"),
        }
    return output


__all__ = [
    "CHANNEL_ORDER",
    "INLET_BAND_FRACTION",
    "OUTLET_BAND_FRACTION",
    "field_response_norms",
    "module_peak_temperatures",
    "pressure_drop_8pct",
    "smooth_peak_temperature",
]
