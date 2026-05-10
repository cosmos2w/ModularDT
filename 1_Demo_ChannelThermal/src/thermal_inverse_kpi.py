from __future__ import annotations

"""KPI utilities for steady ChannelThermal inverse design.

These helpers are intentionally standalone. Training uses them to turn solved
processed cases into target specifications; evaluation uses the same functions
to score generated layouts after frozen-forward verification. They do not feed
solved KPI values into the forward model.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Torch is optional for this utility module.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


DEFAULT_KPI_NAMES: Tuple[str, ...] = (
    "max_solid_temperature",
    "p95_solid_temperature",
    "mean_solid_temperature",
    "max_surface_temperature",
    "module_peak_temperature_spread",
    "module_mean_temperature_std",
    "thermal_resistance_max",
    "pressure_drop",
    "outlet_temperature_rise_mean",
    "outlet_temperature_nonuniformity",
    "max_fluid_temperature",
    "p95_fluid_temperature",
    "hot_fluid_area_fraction",
    "hot_solid_area_fraction",
    "thermal_plume_area",
    "thermal_plume_length",
    "downstream_reheat_index",
    "low_velocity_hotspot_fraction",
    "wall_hot_area_fraction",
    "temperature_gradient_energy",
    "outlet_hot_fraction",
    "max_interface_heat_flux",
    "mean_abs_interface_heat_flux",
    "interface_flux_std",
    "mean_interface_T_env",
    "max_h_effective",
)

TARGET_BLOCKS: Tuple[str, ...] = (
    "values",
    "value_mask",
    "lower_bounds",
    "lower_mask",
    "upper_bounds",
    "upper_mask",
    "weights",
)

CONSTRAINT_NAMES: Tuple[str, ...] = (
    "num_modules_min_scaled",
    "num_modules_max_scaled",
    "min_center_distance_scaled",
    "wall_clearance_scaled",
    "inlet_clearance_scaled",
    "outlet_clearance_scaled",
    "heat_power_total_scaled",
    "heat_power_total_mask",
)

NONNEGATIVE_KPI_NAMES = frozenset(DEFAULT_KPI_NAMES)


def _as_numpy(array: Any) -> np.ndarray:
    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float(default)
    return scalar if math.isfinite(scalar) else float(default)


def _finite_array(value: Any) -> np.ndarray:
    arr = _as_numpy(value).astype(np.float64, copy=False)
    return arr[np.isfinite(arr)]


def _channel_index(channel_order: Optional[Sequence[str]], name: str, fallback: int, field_dim: int) -> int:
    if channel_order:
        normalized = [str(ch).lower() for ch in channel_order]
        if name.lower() in normalized:
            idx = normalized.index(name.lower())
            if 0 <= idx < field_dim:
                return idx
    return min(max(int(fallback), 0), max(int(field_dim) - 1, 0))


def _domain_from_grid(
    x_grid: Optional[Any],
    y_grid: Optional[Any],
    domain: Optional[Mapping[str, Any]],
) -> Tuple[float, float, float, float]:
    if isinstance(domain, Mapping):
        lx = domain.get("domain_length_x", domain.get("lx", domain.get("Lx")))
        ly = domain.get("domain_length_y", domain.get("ly", domain.get("Ly")))
        xmin = _finite_float(domain.get("xmin", 0.0), 0.0)
        ymin = _finite_float(domain.get("ymin", 0.0), 0.0)
        if lx is not None and ly is not None:
            return max(_finite_float(lx, 1.0), 1.0e-8), max(_finite_float(ly, 1.0), 1.0e-8), xmin, ymin
    if x_grid is None or y_grid is None:
        return 1.0, 1.0, 0.0, 0.0
    x = _as_numpy(x_grid).astype(np.float64, copy=False)
    y = _as_numpy(y_grid).astype(np.float64, copy=False)

    def _length(vals: np.ndarray) -> Tuple[float, float]:
        flat = np.unique(vals[np.isfinite(vals)].reshape(-1))
        if flat.size <= 1:
            return 1.0, 0.0
        diffs = np.diff(np.sort(flat))
        step = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
        return float(np.max(flat) - np.min(flat) + step), float(np.min(flat) - 0.5 * step)

    lx, xmin = _length(x)
    ly, ymin = _length(y)
    return max(lx, 1.0e-8), max(ly, 1.0e-8), xmin, ymin


def _axis_masks(x_grid: Optional[Any], y_grid: Optional[Any], shape: Tuple[int, int], domain: Optional[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    h, w = shape
    lx, ly, xmin, ymin = _domain_from_grid(x_grid, y_grid, domain)
    if x_grid is not None and y_grid is not None:
        x = _as_numpy(x_grid).astype(np.float64, copy=False)
        y = _as_numpy(y_grid).astype(np.float64, copy=False)
        if x.shape != shape:
            x = np.resize(x, shape)
        if y.shape != shape:
            y = np.resize(y, shape)
        x_rel = x - xmin
        y_rel = y - ymin
    else:
        x_rel = np.broadcast_to(np.linspace(0.0, lx, w, dtype=np.float64)[None, :], shape)
        y_rel = np.broadcast_to(np.linspace(0.0, ly, h, dtype=np.float64)[:, None], shape)
    inlet = x_rel <= 0.08 * lx
    outlet = x_rel >= 0.92 * lx
    upstream = x_rel <= 0.35 * lx
    midstream = (x_rel >= 0.40 * lx) & (x_rel <= 0.60 * lx)
    downstream = x_rel >= 0.65 * lx
    wall_band = (y_rel <= 0.08 * ly) | (y_rel >= 0.92 * ly)
    return {
        "x_rel": x_rel,
        "y_rel": y_rel,
        "inlet": inlet,
        "outlet": outlet,
        "upstream": upstream,
        "midstream": midstream,
        "downstream": downstream,
        "wall_band": wall_band,
    }


def _module_mask_from_geometry(
    x_grid: Optional[Any],
    y_grid: Optional[Any],
    module_centers: Optional[Any],
    module_present: Optional[Any],
    module_radius: float,
    shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    if x_grid is None or y_grid is None or module_centers is None:
        return None
    x = _as_numpy(x_grid).astype(np.float64, copy=False)
    y = _as_numpy(y_grid).astype(np.float64, copy=False)
    centers = _as_numpy(module_centers).astype(np.float64, copy=False).reshape(-1, 2)
    if module_present is None:
        present = np.ones((centers.shape[0],), dtype=bool)
    else:
        present = _as_numpy(module_present).reshape(-1) > 0.5
    mask = np.zeros(shape, dtype=bool)
    for idx, center in enumerate(centers[: present.shape[0]]):
        if not present[idx]:
            continue
        mask |= np.hypot(x - float(center[0]), y - float(center[1])) <= float(module_radius)
    return mask


def _add_available(out: Dict[str, Any], name: str, value: Any) -> None:
    scalar = _finite_float(value, float("nan"))
    if math.isfinite(scalar):
        out[name] = float(scalar)
        out.setdefault("available_kpis", []).append(name)
    else:
        out[name] = float("nan")
        out.setdefault("unavailable_kpis", []).append(name)


def _hot_threshold(values: np.ndarray, reference: float, explicit: Optional[float] = None) -> float:
    finite = values[np.isfinite(values)]
    if explicit is not None and math.isfinite(float(explicit)):
        return float(explicit)
    if finite.size == 0:
        return float("nan")
    p90 = float(np.percentile(finite, 90.0))
    p98 = float(np.percentile(finite, 98.0))
    ref = float(reference) if math.isfinite(float(reference)) else float(np.percentile(finite, 10.0))
    return max(p90, ref + 0.45 * max(p98 - ref, 0.0))


def _limit_value(limits: Optional[Mapping[str, Any]], key: str) -> Optional[float]:
    if not isinstance(limits, Mapping):
        return None
    value = limits.get(key)
    if value is None:
        return None
    scalar = _finite_float(value, float("nan"))
    return scalar if math.isfinite(scalar) else None


def _configured_temperature_threshold(
    values: np.ndarray,
    reference: float,
    limits: Optional[Mapping[str, Any]],
    *,
    absolute_key: str,
    delta_key: str,
    legacy_explicit: Optional[float] = None,
) -> Tuple[float, str]:
    absolute = _limit_value(limits, absolute_key)
    if absolute is not None:
        return float(absolute), f"{absolute_key}=absolute"
    delta = _limit_value(limits, delta_key)
    if delta is not None and math.isfinite(float(reference)):
        return float(reference + delta), f"{delta_key}=reference_delta"
    if legacy_explicit is not None and math.isfinite(float(legacy_explicit)):
        return float(legacy_explicit), "hot_temperature_threshold=legacy_absolute"
    return _hot_threshold(values, reference, None), "percentile_fallback"


def _masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.shape != mask.shape:
        mask = np.resize(mask.astype(bool), values.shape)
    return values[mask.astype(bool) & np.isfinite(values)]


def _stats(values: np.ndarray) -> Tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.max(finite)), float(np.percentile(finite, 95.0)), float(np.mean(finite))


def compute_steady_thermal_kpis(
    steady_field: Any,
    *,
    x_grid: Optional[Any] = None,
    y_grid: Optional[Any] = None,
    channel_order: Optional[Sequence[str]] = None,
    module_mask: Optional[Any] = None,
    module_centers: Optional[Any] = None,
    module_present: Optional[Any] = None,
    heat_powers: Optional[Any] = None,
    module_internal_temperature: Optional[Any] = None,
    module_internal_mask: Optional[Any] = None,
    interface_target: Optional[Any] = None,
    interface_condition: Optional[Any] = None,
    domain: Optional[Mapping[str, Any]] = None,
    material_params: Optional[Any] = None,
    reference_temperature: Optional[float] = None,
    hot_temperature_threshold: Optional[float] = None,
    temperature_limits: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute steady field, internal, and interface KPIs.

    Missing internal/interface branches are reported in ``unavailable_kpis`` and
    filled with ``NaN``. Scoring skips those entries unless a target marks them
    as explicitly required.
    """

    field = _as_numpy(steady_field).astype(np.float64, copy=False)
    if field.ndim != 3:
        raise ValueError(f"steady_field must have shape [H,W,C], got {field.shape}.")
    if field.shape[-1] < 1:
        raise ValueError("steady_field must contain at least one channel.")

    h, w, c = field.shape
    t_idx = _channel_index(channel_order, "temperature", c - 1, c)
    p_idx = _channel_index(channel_order, "p", min(2, c - 1), c)
    u_idx = _channel_index(channel_order, "u", 0, c)
    v_idx = _channel_index(channel_order, "v", min(1, c - 1), c)
    temperature = field[..., t_idx]
    pressure = field[..., p_idx]
    u = field[..., u_idx]
    v = field[..., v_idx]
    axes = _axis_masks(x_grid, y_grid, (h, w), domain)

    material = _as_numpy(material_params).reshape(-1) if material_params is not None else np.asarray([], dtype=np.float32)
    radius = _finite_float(material[5], 0.45) if material.size > 5 else _finite_float((domain or {}).get("module_radius", 0.45), 0.45)
    if module_mask is None:
        module_mask_np = _module_mask_from_geometry(x_grid, y_grid, module_centers, module_present, radius, (h, w))
    else:
        module_mask_np = _as_numpy(module_mask).astype(bool)
    if module_mask_np is not None and module_mask_np.shape != (h, w):
        module_mask_np = np.resize(module_mask_np, (h, w)).astype(bool)
    fluid_mask = ~module_mask_np if module_mask_np is not None else np.ones((h, w), dtype=bool)
    fluid_t = _masked_values(temperature, fluid_mask)

    inlet_t = _masked_values(temperature, fluid_mask & axes["inlet"])
    outlet_t = _masked_values(temperature, fluid_mask & axes["outlet"])
    ref_t = (
        float(reference_temperature)
        if reference_temperature is not None and math.isfinite(float(reference_temperature))
        else float(np.mean(inlet_t))
        if inlet_t.size
        else float(np.nanmin(temperature))
    )
    fluid_hot_t, fluid_hot_source = _configured_temperature_threshold(
        fluid_t,
        ref_t,
        temperature_limits,
        absolute_key="fluid_hot_temperature",
        delta_key="fluid_hot_delta_T",
        legacy_explicit=hot_temperature_threshold,
    )
    solid_hot_t, solid_hot_source = _configured_temperature_threshold(
        solid_values if "solid_values" in locals() and solid_values.size else fluid_t,
        ref_t,
        temperature_limits,
        absolute_key="solid_hot_temperature",
        delta_key="solid_hot_delta_T",
        legacy_explicit=hot_temperature_threshold,
    )
    outlet_hot_t, outlet_hot_source = _configured_temperature_threshold(
        outlet_t if outlet_t.size else fluid_t,
        ref_t,
        temperature_limits,
        absolute_key="outlet_hot_temperature",
        delta_key="outlet_hot_delta_T",
        legacy_explicit=hot_temperature_threshold,
    )
    wall_hot_t, wall_hot_source = _configured_temperature_threshold(
        fluid_t,
        ref_t,
        temperature_limits,
        absolute_key="wall_hot_temperature",
        delta_key="wall_hot_delta_T",
        legacy_explicit=hot_temperature_threshold,
    )
    hot_fluid = (temperature >= fluid_hot_t) & fluid_mask if math.isfinite(fluid_hot_t) else np.zeros((h, w), dtype=bool)

    out: Dict[str, Any] = {
        "available_kpis": [],
        "unavailable_kpis": [],
        "num_modules": int(np.sum(_as_numpy(module_present).reshape(-1) > 0.5)) if module_present is not None else 0,
        "temperature_thresholds": {
            "reference_temperature": float(ref_t),
            "fluid_hot_temperature": float(fluid_hot_t) if math.isfinite(fluid_hot_t) else float("nan"),
            "fluid_hot_source": fluid_hot_source,
            "solid_hot_temperature": float(solid_hot_t) if math.isfinite(solid_hot_t) else float("nan"),
            "solid_hot_source": solid_hot_source,
            "outlet_hot_temperature": float(outlet_hot_t) if math.isfinite(outlet_hot_t) else float("nan"),
            "outlet_hot_source": outlet_hot_source,
            "wall_hot_temperature": float(wall_hot_t) if math.isfinite(wall_hot_t) else float("nan"),
            "wall_hot_source": wall_hot_source,
        },
    }

    # Solid/internal KPIs.
    active_present = _as_numpy(module_present).reshape(-1) > 0.5 if module_present is not None else None
    solid_values = np.asarray([], dtype=np.float64)
    per_module_peak = []
    per_module_mean = []
    if module_internal_temperature is not None:
        internal = _as_numpy(module_internal_temperature).astype(np.float64, copy=False)
        if internal.ndim >= 2:
            if active_present is None:
                active_present = np.ones((internal.shape[0],), dtype=bool)
            local_mask = _as_numpy(module_internal_mask).astype(bool) if module_internal_mask is not None else None
            active_values = []
            for m in range(min(internal.shape[0], active_present.shape[0])):
                if not active_present[m]:
                    continue
                vals = internal[m]
                if vals.ndim == 3 and vals.shape[-1] == 1:
                    vals = vals[..., 0]
                vals = vals[local_mask] if local_mask is not None and vals.shape[:2] == local_mask.shape else vals.reshape(-1)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    active_values.append(vals)
                    per_module_peak.append(float(np.max(vals)))
                    per_module_mean.append(float(np.mean(vals)))
            if active_values:
                solid_values = np.concatenate(active_values)
    elif module_mask_np is not None:
        solid_values = _masked_values(temperature, module_mask_np)

    solid_hot_t, solid_hot_source = _configured_temperature_threshold(
        solid_values if solid_values.size else fluid_t,
        ref_t,
        temperature_limits,
        absolute_key="solid_hot_temperature",
        delta_key="solid_hot_delta_T",
        legacy_explicit=hot_temperature_threshold,
    )
    out["temperature_thresholds"]["solid_hot_temperature"] = float(solid_hot_t) if math.isfinite(solid_hot_t) else float("nan")
    out["temperature_thresholds"]["solid_hot_source"] = solid_hot_source

    max_solid, p95_solid, mean_solid = _stats(solid_values)
    _add_available(out, "max_solid_temperature", max_solid)
    _add_available(out, "p95_solid_temperature", p95_solid)
    _add_available(out, "mean_solid_temperature", mean_solid)
    _add_available(out, "module_peak_temperature_spread", float(np.ptp(per_module_peak)) if len(per_module_peak) >= 2 else 0.0 if len(per_module_peak) == 1 else float("nan"))
    _add_available(out, "module_mean_temperature_std", float(np.std(per_module_mean)) if len(per_module_mean) >= 2 else 0.0 if len(per_module_mean) == 1 else float("nan"))

    heat = _as_numpy(heat_powers).reshape(-1).astype(np.float64) if heat_powers is not None else np.asarray([], dtype=np.float64)
    if per_module_peak and heat.size:
        n = min(len(per_module_peak), heat.size)
        denom = np.maximum(np.abs(heat[:n]), 1.0e-8)
        resistance = (np.asarray(per_module_peak[:n], dtype=np.float64) - ref_t) / denom
        _add_available(out, "thermal_resistance_max", float(np.nanmax(resistance)))
    else:
        _add_available(out, "thermal_resistance_max", float("nan"))

    if interface_target is not None:
        interface = _as_numpy(interface_target).astype(np.float64, copy=False)
        if active_present is not None and interface.ndim >= 3:
            present_shape = min(interface.shape[0], active_present.shape[0])
            interface = interface[:present_shape][active_present[:present_shape]]
        surf = interface[..., 0] if interface.ndim >= 2 and interface.shape[-1] >= 1 else np.asarray([])
        q = interface[..., 1] if interface.ndim >= 2 and interface.shape[-1] >= 2 else np.asarray([])
        _add_available(out, "max_surface_temperature", float(np.nanmax(surf)) if np.isfinite(surf).any() else float("nan"))
        _add_available(out, "max_interface_heat_flux", float(np.nanmax(np.abs(q))) if np.isfinite(q).any() else float("nan"))
        _add_available(out, "mean_abs_interface_heat_flux", float(np.nanmean(np.abs(q))) if np.isfinite(q).any() else float("nan"))
        _add_available(out, "interface_flux_std", float(np.nanstd(q)) if np.isfinite(q).any() else float("nan"))
    else:
        _add_available(out, "max_surface_temperature", max_solid)
        for name in ("max_interface_heat_flux", "mean_abs_interface_heat_flux", "interface_flux_std"):
            _add_available(out, name, float("nan"))

    if interface_condition is not None:
        cond = _as_numpy(interface_condition).astype(np.float64, copy=False)
        if active_present is not None and cond.ndim >= 3:
            present_shape = min(cond.shape[0], active_present.shape[0])
            cond = cond[:present_shape][active_present[:present_shape]]
        t_env = cond[..., 3] if cond.ndim >= 2 and cond.shape[-1] >= 4 else np.asarray([])
        h_eff = cond[..., 7] if cond.ndim >= 2 and cond.shape[-1] >= 8 else cond[..., 6] if cond.ndim >= 2 and cond.shape[-1] >= 7 else np.asarray([])
        _add_available(out, "mean_interface_T_env", float(np.nanmean(t_env)) if np.isfinite(t_env).any() else float("nan"))
        _add_available(out, "max_h_effective", float(np.nanmax(h_eff)) if np.isfinite(h_eff).any() else float("nan"))
    else:
        _add_available(out, "mean_interface_T_env", float("nan"))
        _add_available(out, "max_h_effective", float("nan"))

    # Global field and outlet KPIs.
    inlet_p = _masked_values(pressure, fluid_mask & axes["inlet"])
    outlet_p = _masked_values(pressure, fluid_mask & axes["outlet"])
    _add_available(out, "pressure_drop", float(np.mean(inlet_p) - np.mean(outlet_p)) if inlet_p.size and outlet_p.size else float("nan"))
    _add_available(out, "outlet_temperature_rise_mean", float(np.mean(outlet_t) - ref_t) if outlet_t.size else float("nan"))
    _add_available(out, "outlet_temperature_nonuniformity", float(np.std(outlet_t)) if outlet_t.size else float("nan"))

    max_fluid, p95_fluid, _ = _stats(fluid_t)
    _add_available(out, "max_fluid_temperature", max_fluid)
    _add_available(out, "p95_fluid_temperature", p95_fluid)
    _add_available(out, "hot_fluid_area_fraction", float(np.mean(hot_fluid[fluid_mask])) if np.any(fluid_mask) else float("nan"))
    if solid_values.size:
        _add_available(out, "hot_solid_area_fraction", float(np.mean(solid_values >= solid_hot_t)) if math.isfinite(solid_hot_t) else float("nan"))
    elif module_mask_np is not None and np.any(module_mask_np):
        _add_available(out, "hot_solid_area_fraction", float(np.mean(temperature[module_mask_np] >= solid_hot_t)) if math.isfinite(solid_hot_t) else float("nan"))
    else:
        _add_available(out, "hot_solid_area_fraction", float("nan"))

    downstream_hot = hot_fluid & axes["downstream"]
    _add_available(out, "thermal_plume_area", float(np.mean(downstream_hot[fluid_mask])) if np.any(fluid_mask) else float("nan"))
    if np.any(hot_fluid):
        hot_x = axes["x_rel"][hot_fluid]
        if module_centers is not None:
            centers = _as_numpy(module_centers).reshape(-1, 2)
            present = active_present if active_present is not None else np.ones((centers.shape[0],), dtype=bool)
            source_x = float(np.min(centers[: present.shape[0]][present[: centers.shape[0]], 0])) if np.any(present[: centers.shape[0]]) else float(np.min(hot_x))
        else:
            source_x = float(np.min(hot_x))
        plume_length = max(float(np.max(hot_x) - source_x), 0.0)
    else:
        plume_length = 0.0
    _add_available(out, "thermal_plume_length", plume_length)

    mid_t = _masked_values(temperature, fluid_mask & axes["midstream"])
    down_t = _masked_values(temperature, fluid_mask & axes["downstream"])
    _add_available(out, "downstream_reheat_index", float(np.mean(down_t) - np.mean(mid_t)) if down_t.size and mid_t.size else float("nan"))
    speed = np.sqrt(u * u + v * v)
    speed_fluid = _masked_values(speed, fluid_mask)
    low_speed_cut = float(np.percentile(speed_fluid, 25.0)) if speed_fluid.size else float("nan")
    low_hot = (temperature >= fluid_hot_t) & fluid_mask & (speed <= low_speed_cut) if math.isfinite(low_speed_cut) and math.isfinite(fluid_hot_t) else np.zeros_like(hot_fluid)
    _add_available(out, "low_velocity_hotspot_fraction", float(np.mean(low_hot[fluid_mask])) if np.any(fluid_mask) else float("nan"))
    wall_hot = (temperature >= wall_hot_t) & fluid_mask & axes["wall_band"] if math.isfinite(wall_hot_t) else np.zeros_like(hot_fluid)
    _add_available(out, "wall_hot_area_fraction", float(np.mean(wall_hot[fluid_mask & axes["wall_band"]])) if np.any(fluid_mask & axes["wall_band"]) else 0.0)

    try:
        grad_y, grad_x = np.gradient(temperature)
        _add_available(out, "temperature_gradient_energy", float(np.nanmean((grad_x[fluid_mask] ** 2 + grad_y[fluid_mask] ** 2))))
    except Exception:
        _add_available(out, "temperature_gradient_energy", float("nan"))
    _add_available(out, "outlet_hot_fraction", float(np.mean((outlet_t >= outlet_hot_t))) if outlet_t.size and math.isfinite(outlet_hot_t) else float("nan"))

    out["available_kpis"] = sorted(set(out.get("available_kpis", [])))
    out["unavailable_kpis"] = sorted(set(name for name in DEFAULT_KPI_NAMES if name not in out["available_kpis"]))
    return out


def kpi_vector_from_dict(kpi_dict: Mapping[str, Any], kpi_names: Sequence[str] = DEFAULT_KPI_NAMES, *, fill_value: float = 0.0) -> np.ndarray:
    unavailable = set(str(name) for name in kpi_dict.get("unavailable_kpis", []))
    values = []
    for name in kpi_names:
        if str(name) in unavailable:
            values.append(float(fill_value))
        else:
            values.append(_finite_float(kpi_dict.get(name, fill_value), fill_value))
    return np.asarray(values, dtype=np.float32)


def _stats_arrays(stats: Optional[Mapping[str, Any]], size: int) -> Tuple[np.ndarray, np.ndarray]:
    if not stats:
        return np.zeros(size, dtype=np.float32), np.ones(size, dtype=np.float32)
    if "mean" in stats and "std" in stats:
        mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
    elif "kpi_mean" in stats and "kpi_std" in stats:
        mean = np.asarray(stats["kpi_mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(stats["kpi_std"], dtype=np.float32).reshape(-1)
    else:
        names = list(stats.get("names", [])) if isinstance(stats, Mapping) else []
        mean = np.zeros(size, dtype=np.float32)
        std = np.ones(size, dtype=np.float32)
        for i, name in enumerate(names[:size]):
            entry = stats.get(name, {}) if isinstance(stats, Mapping) else {}
            if isinstance(entry, Mapping):
                mean[i] = _finite_float(entry.get("mean", 0.0), 0.0)
                std[i] = max(_finite_float(entry.get("std", 1.0), 1.0), 1.0e-8)
    if mean.size != size:
        mean = np.resize(mean, size).astype(np.float32)
    if std.size != size:
        std = np.resize(std, size).astype(np.float32)
    std = np.where(np.abs(std) < 1.0e-8, 1.0, std).astype(np.float32)
    return mean.astype(np.float32), std


def normalize_kpis(kpi_vector: Any, stats: Optional[Mapping[str, Any]]) -> np.ndarray:
    vec = np.asarray(kpi_vector, dtype=np.float32)
    mean, std = _stats_arrays(stats, vec.size)
    return ((vec.reshape(-1) - mean) / std).astype(np.float32).reshape(vec.shape)


def denormalize_kpis(kpi_vector: Any, stats: Optional[Mapping[str, Any]]) -> np.ndarray:
    vec = np.asarray(kpi_vector, dtype=np.float32)
    mean, std = _stats_arrays(stats, vec.size)
    return (vec.reshape(-1) * std + mean).astype(np.float32).reshape(vec.shape)


def augment_kpi_targets_for_training(
    kpi_dict: Mapping[str, Any],
    kpi_names: Sequence[str],
    kpi_stats: Optional[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    rng: Any,
    *,
    return_metadata: bool = False,
) -> Dict[str, Any]:
    def _with_metadata(targets: Dict[str, Dict[str, float | str]]) -> Dict[str, Any]:
        active_names = [str(name) for name in kpi_names if str(name) in targets]
        active_set = set(active_names)
        mode_by_kpi = {str(name): str(targets[name].get("mode", "exact")) for name in active_names}
        if not return_metadata:
            return targets
        return {
            "kpi_targets": targets,
            "active_kpi_names": active_names,
            "dropped_kpi_names": [str(name) for name in kpi_names if str(name) not in active_set],
            "target_modes_by_kpi": mode_by_kpi,
            "active_kpi_count": len(active_names),
            "target_active_fraction": float(len(active_names)) / max(float(len(kpi_names)), 1.0),
            "active_kpi_mask": [1.0 if str(name) in active_set else 0.0 for name in kpi_names],
        }

    names = [str(name) for name in kpi_names if str(name) in kpi_dict and str(name) not in set(kpi_dict.get("unavailable_kpis", []))]
    if not names:
        return _with_metadata({})
    if not bool(cfg.get("enabled", False)):
        targets = {name: {"mode": "exact", "value": _finite_float(kpi_dict.get(name, 0.0), 0.0), "weight": 1.0} for name in names}
    else:
        mode_name = str(cfg.get("mode", "independent_dropout")).lower().strip()
        always = {str(name) for name in cfg.get("always_include", [])}
        drop_p = min(max(_finite_float(cfg.get("drop_probability", 0.0), 0.0), 0.0), 1.0)
        min_active_cfg = min(max(int(cfg.get("min_active_kpis", 1)), 0), len(names))
        max_active_raw = cfg.get("max_active_kpis")
        max_active_cfg = len(names) if max_active_raw is None else min(max(int(max_active_raw), 0), len(names))
        always_active = [name for name in names if name in always]
        max_active_cfg = max(max_active_cfg, len(always_active))
        if mode_name == "bounded_subset":
            max_drop_fraction = min(max(_finite_float(cfg.get("max_drop_fraction", 1.0), 1.0), 0.0), 1.0)
            bounded_min = int(math.ceil(len(names) * (1.0 - max_drop_fraction)))
            min_active = max(min_active_cfg, bounded_min, len(always_active))
            min_active = min(min_active, max_active_cfg)
            max_active = max(max_active_cfg, min_active)
            removable = [name for name in names if name not in always]
            drop_candidates = [name for name in removable if float(rng.random()) < drop_p]
            if drop_candidates:
                drop_candidates = [str(name) for name in np.asarray(rng.permutation(drop_candidates)).reshape(-1)]
            max_drops = max(len(names) - min_active, 0)
            drop_set = set(drop_candidates[:max_drops])
            active = [name for name in names if name not in drop_set]
            if len(active) > max_active:
                optional = [name for name in active if name not in always]
                remove_count = min(len(active) - max_active, len(optional))
                if remove_count > 0:
                    remove = set(str(name) for name in np.asarray(rng.choice(optional, size=remove_count, replace=False)).reshape(-1))
                    active = [name for name in active if name not in remove]
        else:
            min_active = min(max(min_active_cfg, len(always_active)), max_active_cfg)
            active = [name for name in names if name in always or float(rng.random()) >= drop_p]
            if len(active) > max_active_cfg:
                optional = [name for name in active if name not in always]
                remove_count = min(len(active) - max_active_cfg, len(optional))
                if remove_count > 0:
                    remove = set(str(name) for name in np.asarray(rng.choice(optional, size=remove_count, replace=False)).reshape(-1))
                    active = [name for name in active if name not in remove]
        if len(active) < min_active:
            missing = [name for name in names if name not in active]
            if missing:
                chosen = rng.choice(missing, size=min(min_active - len(active), len(missing)), replace=False)
                active.extend([str(name) for name in np.asarray(chosen).reshape(-1)])
        probs = np.asarray(
            [
                max(_finite_float(cfg.get("exact_probability", 0.45), 0.45), 0.0),
                max(_finite_float(cfg.get("range_probability", 0.30), 0.30), 0.0),
                max(_finite_float(cfg.get("upper_probability", 0.20), 0.20), 0.0),
                max(_finite_float(cfg.get("lower_probability", 0.05), 0.05), 0.0),
            ],
            dtype=np.float64,
        )
        if float(np.sum(probs)) <= 0.0:
            probs = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        probs = probs / float(np.sum(probs))
        modes = np.asarray(["exact", "range", "max", "min"], dtype=object)
        _, std_arr = _stats_arrays(kpi_stats, len(kpi_names))
        name_to_idx = {str(name): i for i, name in enumerate(kpi_names)}
        width_frac = max(_finite_float(cfg.get("range_width_std_fraction", 0.35), 0.35), 0.0)
        noise_std = max(_finite_float(cfg.get("target_noise_std", 0.0), 0.0), 0.0)
        targets = {}
        for name in active:
            value = _finite_float(kpi_dict.get(name, 0.0), 0.0)
            idx = name_to_idx.get(name, -1)
            std = float(std_arr[idx]) if 0 <= idx < std_arr.size else max(abs(value), 1.0)
            if noise_std > 0.0:
                value = float(value + rng.normal(0.0, noise_std * std))
            mode = str(rng.choice(modes, p=probs))
            width = max(abs(float(rng.normal(width_frac * std, 0.25 * width_frac * std))), 1.0e-6)
            low = value - 0.5 * width
            high = value + 0.5 * width
            if name in NONNEGATIVE_KPI_NAMES:
                low = max(low, 0.0)
            if mode == "range":
                targets[name] = {"mode": "range", "low": float(low), "high": float(max(high, low)), "weight": 1.0}
            elif mode == "max":
                targets[name] = {"mode": "max", "high": float(max(high, value)), "weight": 1.0}
            elif mode == "min":
                targets[name] = {"mode": "min", "low": float(low), "weight": 1.0}
            else:
                targets[name] = {"mode": "exact", "value": float(value), "weight": 1.0}
    return _with_metadata(targets)


def _normalize_scalar(name: str, value: float, kpi_names: Sequence[str], stats: Optional[Mapping[str, Any]], normalize: bool) -> float:
    if not normalize or stats is None:
        return float(value)
    try:
        idx = list(kpi_names).index(name)
    except ValueError:
        return float(value)
    mean, std = _stats_arrays(stats, len(kpi_names))
    return float((float(value) - float(mean[idx])) / max(float(std[idx]), 1.0e-8))


def _parse_target_entry(name: str, entry: Any) -> Dict[str, float | str | bool]:
    if isinstance(entry, Mapping):
        mode = str(entry.get("mode", "exact")).lower().strip()
        value = entry.get("value", entry.get("target"))
        low = entry.get("low", entry.get("lower"))
        high = entry.get("high", entry.get("upper"))
        return {
            "mode": mode,
            "value": _finite_float(value, float("nan")),
            "low": _finite_float(low, float("nan")),
            "high": _finite_float(high, float("nan")),
            "weight": max(_finite_float(entry.get("weight", 1.0), 1.0), 0.0),
            "scale": _finite_float(entry.get("scale", float("nan")), float("nan")),
            "required": bool(entry.get("required", False)),
        }
    return {
        "mode": "exact",
        "value": _finite_float(entry, 0.0),
        "low": float("nan"),
        "high": float("nan"),
        "weight": 1.0,
        "scale": float("nan"),
        "required": False,
    }


def build_target_spec_vector(
    kpi_dict: Optional[Mapping[str, Any]] = None,
    kpi_names: Optional[Sequence[str]] = None,
    *,
    kpi_targets: Optional[Mapping[str, Any]] = None,
    stats: Optional[Mapping[str, Any]] = None,
    normalize: bool = False,
    num_modules_min: Optional[int] = None,
    num_modules_max: Optional[int] = None,
    min_center_distance: Optional[float] = None,
    wall_clearance: Optional[float] = None,
    inlet_clearance: Optional[float] = None,
    outlet_clearance: Optional[float] = None,
    heat_power_total: Optional[float] = None,
    max_num_modules: int = 8,
    domain_length_scale: float = 12.0,
    heat_power_scale: float = 1.0,
    return_spec: bool = False,
) -> np.ndarray | Dict[str, Any]:
    names = tuple(kpi_names or DEFAULT_KPI_NAMES)
    k = len(names)
    raw_targets: Dict[str, Any] = dict(kpi_targets or {})
    unavailable = set(str(name) for name in (kpi_dict or {}).get("unavailable_kpis", []))
    if kpi_dict:
        for name in names:
            if name not in raw_targets and name in kpi_dict and name not in unavailable:
                raw_targets[name] = {"mode": "exact", "value": float(kpi_dict[name]), "weight": 1.0}

    values = np.zeros(k, dtype=np.float32)
    value_mask = np.zeros(k, dtype=np.float32)
    lower = np.zeros(k, dtype=np.float32)
    lower_mask = np.zeros(k, dtype=np.float32)
    upper = np.zeros(k, dtype=np.float32)
    upper_mask = np.zeros(k, dtype=np.float32)
    weights = np.zeros(k, dtype=np.float32)
    parsed: Dict[str, Dict[str, float | str | bool]] = {}

    for i, name in enumerate(names):
        if name not in raw_targets:
            continue
        entry = _parse_target_entry(name, raw_targets[name])
        parsed[name] = entry
        mode = str(entry["mode"])
        weight = max(float(entry["weight"]), 0.0)
        weights[i] = weight
        val = float(entry["value"])
        lo = float(entry["low"])
        hi = float(entry["high"])
        if mode in {"range", "between"}:
            if math.isfinite(lo):
                lower[i] = _normalize_scalar(name, lo, names, stats, normalize)
                lower_mask[i] = 1.0
            if math.isfinite(hi):
                upper[i] = _normalize_scalar(name, hi, names, stats, normalize)
                upper_mask[i] = 1.0
            if math.isfinite(lo) and math.isfinite(hi):
                values[i] = _normalize_scalar(name, 0.5 * (lo + hi), names, stats, normalize)
                value_mask[i] = 1.0
        elif mode in {"max", "upper", "at_most"}:
            if math.isfinite(hi):
                upper[i] = _normalize_scalar(name, hi, names, stats, normalize)
                upper_mask[i] = 1.0
                values[i] = upper[i]
        elif mode in {"min", "lower", "at_least"}:
            if math.isfinite(lo):
                lower[i] = _normalize_scalar(name, lo, names, stats, normalize)
                lower_mask[i] = 1.0
                values[i] = lower[i]
        elif mode == "minimize":
            value_mask[i] = 1.0
            values[i] = _normalize_scalar(name, 0.0, names, stats, normalize)
        elif mode == "maximize":
            value_mask[i] = 1.0
            values[i] = _normalize_scalar(name, 0.0, names, stats, normalize)
        else:
            if not math.isfinite(val):
                val = 0.5 * (lo + hi) if math.isfinite(lo) and math.isfinite(hi) else 0.0
            values[i] = _normalize_scalar(name, val, names, stats, normalize)
            value_mask[i] = 1.0

    constraints = np.asarray(
        [
            0.0 if num_modules_min is None else float(num_modules_min) / max(float(max_num_modules), 1.0),
            1.0 if num_modules_max is None else float(num_modules_max) / max(float(max_num_modules), 1.0),
            0.0 if min_center_distance is None else float(min_center_distance) / max(float(domain_length_scale), 1.0e-8),
            0.0 if wall_clearance is None else float(wall_clearance) / max(float(domain_length_scale), 1.0e-8),
            0.0 if inlet_clearance is None else float(inlet_clearance) / max(float(domain_length_scale), 1.0e-8),
            0.0 if outlet_clearance is None else float(outlet_clearance) / max(float(domain_length_scale), 1.0e-8),
            0.0 if heat_power_total is None else float(heat_power_total) / max(float(heat_power_scale), 1.0e-8),
            0.0 if heat_power_total is None else 1.0,
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([values, value_mask, lower, lower_mask, upper, upper_mask, weights, constraints]).astype(np.float32)
    if return_spec:
        return {
            "vector": vector,
            "kpi_names": list(names),
            "kpi_targets": parsed,
            "kpi_stats": stats,
            "normalized": bool(normalize),
            "constraints": {
                "num_modules_min": num_modules_min,
                "num_modules_max": num_modules_max,
                "min_center_distance": min_center_distance,
                "wall_clearance": wall_clearance,
                "inlet_clearance": inlet_clearance,
                "outlet_clearance": outlet_clearance,
                "heat_power_total": heat_power_total,
                "max_num_modules": max_num_modules,
                "domain_length_scale": domain_length_scale,
                "heat_power_scale": heat_power_scale,
            },
            "layout": {"blocks": list(TARGET_BLOCKS), "constraint_names": list(CONSTRAINT_NAMES)},
        }
    return vector


def split_target_spec_vector(vector: Any, kpi_names: Sequence[str]) -> Dict[str, np.ndarray]:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    k = len(kpi_names)
    expected = len(TARGET_BLOCKS) * k + len(CONSTRAINT_NAMES)
    if vec.size != expected:
        raise ValueError(f"target vector length {vec.size} does not match expected {expected} for {k} KPIs.")
    out: Dict[str, np.ndarray] = {}
    cursor = 0
    for block in TARGET_BLOCKS:
        out[block] = vec[cursor : cursor + k].copy()
        cursor += k
    out["constraints"] = vec[cursor : cursor + len(CONSTRAINT_NAMES)].copy()
    return out


def _scale_for_error(*values: float) -> float:
    finite = [abs(float(v)) for v in values if math.isfinite(float(v))]
    return max(finite + [1.0])


def _scale_for_target(
    name: str,
    names: Sequence[str],
    stats: Optional[Mapping[str, Any]],
    explicit_scale: float,
    *target_values: float,
) -> float:
    if math.isfinite(float(explicit_scale)) and abs(float(explicit_scale)) > 1.0e-8:
        return abs(float(explicit_scale))
    if stats is not None:
        try:
            idx = list(names).index(str(name))
        except ValueError:
            idx = -1
        if idx >= 0:
            _, std = _stats_arrays(stats, len(names))
            if idx < std.size and math.isfinite(float(std[idx])) and abs(float(std[idx])) > 1.0e-8:
                return abs(float(std[idx]))
    finite_targets = [abs(float(v)) for v in target_values if math.isfinite(float(v)) and abs(float(v)) > 1.0e-8]
    return max(finite_targets + [1.0])


def score_candidate_kpis(kpi_dict: Mapping[str, Any], target_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Score candidate KPIs against exact/range/bound targets.

    Lower ``total_score`` is better. Missing unavailable KPIs are ignored unless
    a target entry includes ``"required": true``.
    """

    names = list(target_spec.get("kpi_names", DEFAULT_KPI_NAMES))
    stats = target_spec.get("kpi_stats")
    target_entries = target_spec.get("kpi_targets", target_spec.get("kpis"))
    if target_entries is None and "vector" in target_spec:
        blocks = split_target_spec_vector(target_spec["vector"], names)
        target_entries = {}
        for idx, name in enumerate(names):
            if blocks["weights"][idx] <= 0.0 and blocks["value_mask"][idx] <= 0.0 and blocks["lower_mask"][idx] <= 0.0 and blocks["upper_mask"][idx] <= 0.0:
                continue
            target_entries[name] = {
                "mode": "vector",
                "value": float(blocks["values"][idx]),
                "value_mask": float(blocks["value_mask"][idx]),
                "low": float(blocks["lower_bounds"][idx]),
                "low_mask": float(blocks["lower_mask"][idx]),
                "high": float(blocks["upper_bounds"][idx]),
                "high_mask": float(blocks["upper_mask"][idx]),
                "weight": float(blocks["weights"][idx]),
            }
    target_entries = dict(target_entries or {})
    unavailable = set(str(name) for name in kpi_dict.get("unavailable_kpis", []))

    per_errors: Dict[str, float] = {}
    per_preference_rewards: Dict[str, float] = {}
    missing_required: list[str] = []
    unavailable_optional: list[str] = []
    weighted_violation = 0.0
    hard_weight_total = 0.0
    weighted_reward = 0.0
    preference_weight_total = 0.0
    for name, raw_entry in target_entries.items():
        entry = _parse_target_entry(str(name), raw_entry)
        weight = max(float(entry["weight"]), 0.0)
        if weight <= 0.0:
            continue
        candidate = _finite_float(kpi_dict.get(name, float("nan")), float("nan"))
        if str(name) in unavailable or not math.isfinite(candidate):
            if bool(entry.get("required", False)):
                missing_required.append(str(name))
                per_errors[str(name)] = float(target_spec.get("missing_kpi_penalty", 5.0))
                weighted_violation += weight * per_errors[str(name)]
                hard_weight_total += weight
            else:
                unavailable_optional.append(str(name))
            continue
        mode = str(entry["mode"])
        value = float(entry["value"])
        low = float(entry["low"])
        high = float(entry["high"])
        scale = float(entry["scale"])
        is_preference = mode in {"minimize", "maximize"}
        reward = 0.0
        if mode in {"range", "between"}:
            if math.isfinite(low) and candidate < low:
                error = (low - candidate) / _scale_for_error(low, high)
            elif math.isfinite(high) and candidate > high:
                error = (candidate - high) / _scale_for_error(low, high)
            else:
                error = 0.0
        elif mode in {"max", "upper", "at_most"}:
            error = max(0.0, candidate - high) / _scale_for_error(high)
        elif mode in {"min", "lower", "at_least"}:
            error = max(0.0, low - candidate) / _scale_for_error(low)
        elif mode == "minimize":
            denom = _scale_for_target(str(name), names, stats, scale, value, low, high)
            if math.isfinite(high) and high > 0.0:
                reward = 1.0 - min(max(candidate / max(high, 1.0e-8), 0.0), 1.0)
            else:
                reward = 1.0 / (1.0 + max(candidate, 0.0) / max(denom, 1.0e-8))
            error = 0.0
        elif mode == "maximize":
            denom = _scale_for_target(str(name), names, stats, scale, value, low, high)
            if math.isfinite(low) and low > 0.0:
                reward = min(max(candidate / max(low, 1.0e-8), 0.0), 1.0)
            else:
                reward = 0.5 + math.atan(candidate / max(denom, 1.0e-8)) / math.pi
            error = 0.0
        elif mode == "vector" and isinstance(raw_entry, Mapping):
            low_mask = _finite_float(raw_entry.get("low_mask", 0.0), 0.0)
            high_mask = _finite_float(raw_entry.get("high_mask", 0.0), 0.0)
            value_mask = _finite_float(raw_entry.get("value_mask", 0.0), 0.0)
            if low_mask > 0.5 and candidate < low:
                error = (low - candidate) / _scale_for_error(low)
            elif high_mask > 0.5 and candidate > high:
                error = (candidate - high) / _scale_for_error(high)
            elif value_mask > 0.5:
                error = abs(candidate - value) / _scale_for_error(value)
            else:
                error = 0.0
        else:
            error = abs(candidate - value) / _scale_for_error(value)
        if is_preference:
            reward = min(max(float(locals().get("reward", 0.0)), 0.0), 1.0)
            per_preference_rewards[str(name)] = reward
            weighted_reward += weight * reward
            preference_weight_total += weight
        else:
            error = max(float(error), 0.0)
            per_errors[str(name)] = error
            weighted_violation += weight * error
            hard_weight_total += weight

    feasibility_penalty = 0.0
    constraints = target_spec.get("constraints", target_spec)
    count = kpi_dict.get("num_modules", kpi_dict.get("count"))
    if count is not None and isinstance(constraints, Mapping):
        count_val = int(round(_finite_float(count, 0.0)))
        n_min = constraints.get("num_modules_min", constraints.get("num_cylinders_min"))
        n_max = constraints.get("num_modules_max", constraints.get("num_cylinders_max"))
        if n_min is not None and count_val < int(n_min):
            feasibility_penalty += float(int(n_min) - count_val)
        if n_max is not None and count_val > int(n_max):
            feasibility_penalty += float(count_val - int(n_max))
    if isinstance(constraints, Mapping):
        for key in ("min_center_distance", "wall_clearance", "inlet_clearance", "outlet_clearance"):
            target_val = constraints.get(key)
            actual_val = kpi_dict.get(key)
            if target_val is not None and actual_val is not None:
                deficit = float(target_val) - _finite_float(actual_val, 0.0)
                if deficit > 0.0:
                    feasibility_penalty += deficit / max(float(target_val), 1.0e-8)
        heat_target = constraints.get("heat_power_total")
        heat_actual = kpi_dict.get("heat_power_total")
        if heat_target is not None and heat_actual is not None:
            feasibility_penalty += abs(float(heat_actual) - float(heat_target)) / max(abs(float(heat_target)), 1.0)
    if bool(kpi_dict.get("valid", True)) is False:
        feasibility_penalty += 10.0

    kpi_violation = weighted_violation / max(hard_weight_total, 1.0e-8)
    preference_reward = weighted_reward / max(preference_weight_total, 1.0e-8) if preference_weight_total > 0.0 else 0.0
    preference_reward_weight = _finite_float(target_spec.get("preference_reward_weight", 0.1), 0.1)
    total_score = float(feasibility_penalty + kpi_violation - preference_reward_weight * preference_reward)
    return {
        "total_score": total_score,
        "feasibility_penalty": float(feasibility_penalty),
        "kpi_violation": float(kpi_violation),
        "preference_reward": float(preference_reward),
        "kpi_score": float(kpi_violation),
        "per_kpi_errors": per_errors,
        "per_preference_rewards": per_preference_rewards,
        "constraint_penalty": float(feasibility_penalty),
        "missing_required_kpis": missing_required,
        "unavailable_optional_kpis": sorted(set(unavailable_optional)),
        "scored_weight": float(hard_weight_total),
        "preference_weight": float(preference_weight_total),
    }
