from __future__ import annotations

"""Design-intent utilities for field-aware ChannelThermal inverse design."""

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from thermal_inverse_kpi import layout_spread_metrics


OBJECTIVE_NAMES: Tuple[str, ...] = (
    "safety",
    "uniformity",
    "pressure",
    "outlet_mixing",
    "wall_protection",
    "plume_avoidance",
    "coverage",
)

FIELD_INTENT_CHANNELS: Tuple[str, ...] = (
    "protected_region_map",
    "wall_risk_map",
    "outlet_profile_weight",
    "plume_avoidance_map",
    "keepout_map",
)

DESIGN_INTENT_SCALAR_NAMES: Tuple[str, ...] = (
    "num_modules_min_scaled",
    "num_modules_max_scaled",
    "min_center_distance_scaled",
    "wall_clearance_scaled",
    "inlet_clearance_scaled",
    "outlet_clearance_scaled",
    "x_span_low_scaled",
    "x_span_high_scaled",
    "y_span_low_scaled",
    "y_span_high_scaled",
    "solid_temperature_max",
    "module_temperature_spread_max",
    "pressure_drop_max",
    "wall_hot_delta_T",
    "outlet_hot_delta_T",
    "heat_power_total_scaled",
    "heat_power_total_mask",
    "avoid_downstream_hot_plumes",
    "protect_wall_band",
    "protect_outlet_uniformity",
    "min_x_coverage_scaled",
    "min_y_coverage_scaled",
    "min_mean_pair_distance_scaled",
    "schema_version",
)

DESIGN_INTENT_DIM = len(DESIGN_INTENT_SCALAR_NAMES)
OBJECTIVE_DIM = len(OBJECTIVE_NAMES)
DEFAULT_FIELD_MAP_SHAPE = (24, 12)

STRUCTURE_FEATURE_NAMES: Tuple[str, ...] = (
    "count_scaled",
    "centroid_x_scaled",
    "centroid_y_scaled",
    "heat_weighted_centroid_x_scaled",
    "heat_weighted_centroid_y_scaled",
    "x_coverage_scaled",
    "y_coverage_scaled",
    "bbox_area_scaled",
    "x_std_scaled",
    "y_std_scaled",
    "min_pair_distance_scaled",
    "mean_pair_distance_scaled",
    "pair_distance_std_scaled",
    "nearest_neighbor_mean_scaled",
    "nearest_neighbor_std_scaled",
    "wall_proximity_mean",
    "wall_proximity_min",
    "upstream_count_fraction",
    "midstream_count_fraction",
    "downstream_count_fraction",
    "upstream_heat_fraction",
    "midstream_heat_fraction",
    "downstream_heat_fraction",
    *tuple(f"x_bin_occupancy_{idx}" for idx in range(6)),
    *tuple(f"y_bin_occupancy_{idx}" for idx in range(4)),
    *tuple(f"pair_distance_hist_{idx}" for idx in range(6)),
    "occupancy_entropy",
    "heat_density_entropy",
    "anisotropy_score",
)
STRUCTURE_INTENT_DIM = len(STRUCTURE_FEATURE_NAMES)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float(default)
    return scalar if math.isfinite(scalar) else float(default)


def _active_layout_arrays(
    centers: Any,
    present: Optional[Any],
    heat_powers: Optional[Any],
    *,
    max_num_modules: int,
) -> Tuple[np.ndarray, np.ndarray]:
    centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    if present is None:
        present_arr = np.ones((centers_arr.shape[0],), dtype=bool)
    else:
        present_arr = np.asarray(present, dtype=np.float32).reshape(-1) > 0.5
    n = min(centers_arr.shape[0], present_arr.shape[0], int(max_num_modules))
    centers_arr = centers_arr[:n]
    present_arr = present_arr[:n]
    active = centers_arr[present_arr] if n else np.zeros((0, 2), dtype=np.float32)
    if heat_powers is None:
        heat = np.ones((n,), dtype=np.float32)
    else:
        heat = np.asarray(heat_powers, dtype=np.float32).reshape(-1)
        if heat.shape[0] < n:
            heat = np.pad(heat, (0, n - heat.shape[0]))
        heat = heat[:n]
    return active.astype(np.float32), np.maximum(heat[present_arr], 0.0).astype(np.float32) if n else np.zeros((0,), dtype=np.float32)


def _entropy(prob: np.ndarray) -> float:
    arr = np.asarray(prob, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size <= 1:
        return 0.0
    return float(-np.sum(arr * np.log(arr)) / max(math.log(float(arr.size)), 1.0e-8))


def compute_layout_structure_features(
    centers: Any,
    present: Optional[Any],
    heat_powers: Optional[Any] = None,
    *,
    domain_length_x: float = 12.0,
    domain_length_y: float = 4.0,
    module_radius: float = 0.45,
    max_num_modules: int = 12,
    x_bins: int = 6,
    y_bins: int = 4,
    pair_distance_bins: int = 6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return normalized layout-structure descriptors and human-readable metadata."""

    lx = max(float(domain_length_x), 1.0e-8)
    ly = max(float(domain_length_y), 1.0e-8)
    scale = max(lx, ly, 1.0e-8)
    active, heat = _active_layout_arrays(centers, present, heat_powers, max_num_modules=max_num_modules)
    count = int(active.shape[0])
    x_bins = max(int(x_bins), 1)
    y_bins = max(int(y_bins), 1)
    pair_distance_bins = max(int(pair_distance_bins), 1)
    names = list(STRUCTURE_FEATURE_NAMES)
    values: Dict[str, float] = {name: 0.0 for name in names}

    values["count_scaled"] = float(count) / max(float(max_num_modules), 1.0)
    if count > 0:
        x = np.clip(active[:, 0].astype(np.float64), 0.0, lx)
        y = np.clip(active[:, 1].astype(np.float64), 0.0, ly)
        heat64 = np.maximum(heat.astype(np.float64), 0.0)
        heat_sum = float(np.sum(heat64))
        weights = heat64 / heat_sum if heat_sum > 1.0e-12 else np.full((count,), 1.0 / float(count), dtype=np.float64)
        centroid = np.asarray([np.mean(x), np.mean(y)], dtype=np.float64)
        heat_centroid = np.asarray([np.sum(weights * x), np.sum(weights * y)], dtype=np.float64)
        x_cov = float(np.max(x) - np.min(x)) if count >= 2 else 0.0
        y_cov = float(np.max(y) - np.min(y)) if count >= 2 else 0.0
        values.update(
            {
                "centroid_x_scaled": float(centroid[0] / lx),
                "centroid_y_scaled": float(centroid[1] / ly),
                "heat_weighted_centroid_x_scaled": float(heat_centroid[0] / lx),
                "heat_weighted_centroid_y_scaled": float(heat_centroid[1] / ly),
                "x_coverage_scaled": float(x_cov / lx),
                "y_coverage_scaled": float(y_cov / ly),
                "bbox_area_scaled": float((x_cov * y_cov) / max(lx * ly, 1.0e-8)),
                "x_std_scaled": float(np.std(x) / lx),
                "y_std_scaled": float(np.std(y) / ly),
                "wall_proximity_mean": float(np.mean(np.minimum(y, ly - y)) / max(0.5 * ly, 1.0e-8)),
                "wall_proximity_min": float(np.min(np.minimum(y, ly - y)) / max(0.5 * ly, 1.0e-8)),
            }
        )
        region_masks = {
            "upstream": x < (lx / 3.0),
            "midstream": (x >= (lx / 3.0)) & (x < (2.0 * lx / 3.0)),
            "downstream": x >= (2.0 * lx / 3.0),
        }
        for region, mask in region_masks.items():
            values[f"{region}_count_fraction"] = float(np.mean(mask)) if count else 0.0
            values[f"{region}_heat_fraction"] = float(np.sum(heat64[mask]) / heat_sum) if heat_sum > 1.0e-12 else values[f"{region}_count_fraction"]
        x_hist, _ = np.histogram(x, bins=x_bins, range=(0.0, lx))
        y_hist, _ = np.histogram(y, bins=y_bins, range=(0.0, ly))
        x_occ = x_hist.astype(np.float64) / max(float(count), 1.0)
        y_occ = y_hist.astype(np.float64) / max(float(count), 1.0)
        for idx in range(6):
            values[f"x_bin_occupancy_{idx}"] = float(x_occ[idx]) if idx < x_occ.size else 0.0
        for idx in range(4):
            values[f"y_bin_occupancy_{idx}"] = float(y_occ[idx]) if idx < y_occ.size else 0.0
        values["occupancy_entropy"] = 0.5 * (_entropy(x_occ) + _entropy(y_occ))
        heat_x_hist, _ = np.histogram(x, bins=x_bins, range=(0.0, lx), weights=heat64)
        heat_y_hist, _ = np.histogram(y, bins=y_bins, range=(0.0, ly), weights=heat64)
        heat_x_prob = heat_x_hist / max(float(np.sum(heat_x_hist)), 1.0e-12)
        heat_y_prob = heat_y_hist / max(float(np.sum(heat_y_hist)), 1.0e-12)
        values["heat_density_entropy"] = 0.5 * (_entropy(heat_x_prob) + _entropy(heat_y_prob))
        values["anisotropy_score"] = float(abs(values["x_std_scaled"] - values["y_std_scaled"]) / max(values["x_std_scaled"] + values["y_std_scaled"], 1.0e-8))

        pair = []
        nearest = []
        for i in range(count):
            dists = []
            for j in range(count):
                if i == j:
                    continue
                d = float(np.linalg.norm(active[i].astype(np.float64) - active[j].astype(np.float64)))
                dists.append(d)
                if j > i:
                    pair.append(d)
            if dists:
                nearest.append(min(dists))
        pair_arr = np.asarray(pair, dtype=np.float64)
        nn_arr = np.asarray(nearest, dtype=np.float64)
        if pair_arr.size:
            values["min_pair_distance_scaled"] = float(np.min(pair_arr) / scale)
            values["mean_pair_distance_scaled"] = float(np.mean(pair_arr) / scale)
            values["pair_distance_std_scaled"] = float(np.std(pair_arr) / scale)
            pair_hist, _ = np.histogram(pair_arr, bins=pair_distance_bins, range=(0.0, scale))
            pair_prob = pair_hist.astype(np.float64) / max(float(np.sum(pair_hist)), 1.0)
            for idx in range(6):
                values[f"pair_distance_hist_{idx}"] = float(pair_prob[idx]) if idx < pair_prob.size else 0.0
        if nn_arr.size:
            values["nearest_neighbor_mean_scaled"] = float(np.mean(nn_arr) / scale)
            values["nearest_neighbor_std_scaled"] = float(np.std(nn_arr) / scale)

    vector = np.asarray([values[name] for name in names], dtype=np.float32)
    metadata = {
        "feature_names": names,
        "feature_values": {name: float(vector[idx]) for idx, name in enumerate(names)},
        "count": count,
        "domain_length_x": lx,
        "domain_length_y": ly,
        "module_radius": float(module_radius),
        "x_bins": x_bins,
        "y_bins": y_bins,
        "pair_distance_bins": pair_distance_bins,
    }
    return vector, metadata


def _soft_layout_density(
    centers: np.ndarray,
    weights: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    sigma: float,
) -> np.ndarray:
    density = np.zeros_like(xx, dtype=np.float32)
    sigma2 = max(float(sigma) ** 2, 1.0e-8)
    for idx, center in enumerate(np.asarray(centers, dtype=np.float32).reshape(-1, 2)):
        w = float(weights[idx]) if idx < weights.size else 1.0
        density += float(w) * np.exp(-0.5 * ((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2) / sigma2).astype(np.float32)
    max_value = float(np.max(density)) if density.size else 0.0
    return density / max(max_value, 1.0e-8) if max_value > 0.0 else density


def build_layout_structure_maps(
    centers: Any,
    present: Optional[Any],
    heat_powers: Optional[Any] = None,
    *,
    domain_length_x: float = 12.0,
    domain_length_y: float = 4.0,
    module_radius: float = 0.45,
    max_num_modules: int = 12,
    shape: Tuple[int, int] = DEFAULT_FIELD_MAP_SHAPE,
    preferred_region_map: Optional[Any] = None,
    keepout_map: Optional[Any] = None,
    reference_layout_soft_map: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build occupancy, heat, preference, keepout, and reference soft maps."""

    width, height = int(shape[0]), int(shape[1])
    xs = np.linspace(0.0, float(domain_length_x), width, dtype=np.float32)
    ys = np.linspace(0.0, float(domain_length_y), height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    active, heat = _active_layout_arrays(centers, present, heat_powers, max_num_modules=max_num_modules)
    occupancy = _soft_layout_density(active, np.ones((active.shape[0],), dtype=np.float32), xx, yy, sigma=max(float(module_radius), 0.1))
    heat_map = _soft_layout_density(active, np.maximum(heat, 0.0), xx, yy, sigma=max(float(module_radius), 0.1))

    def _map(value: Optional[Any]) -> np.ndarray:
        if value is None:
            return np.zeros((height, width), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (height, width):
            arr = np.resize(arr, (height, width)).astype(np.float32)
        finite = np.where(np.isfinite(arr), arr, 0.0)
        return np.clip(finite, 0.0, 1.0).astype(np.float32)

    maps = np.stack(
        [
            occupancy,
            heat_map,
            _map(preferred_region_map),
            _map(keepout_map),
            _map(reference_layout_soft_map),
        ],
        axis=0,
    ).astype(np.float32)
    return maps, {"channel_names": ["occupancy_density_map", "heat_density_map", "preferred_region_map", "keepout_map", "reference_layout_soft_map"], "shape": [width, height]}


def is_design_intent_payload(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in ("scenario", "geometry_constraints", "thermal_limits", "objective_weights", "field_preferences"))


def normalize_intent_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize v2 intent JSONs and legacy KPI JSONs into a common schema."""

    if is_design_intent_payload(payload):
        scenario = dict(payload.get("scenario", {}) or {})
        geometry = dict(payload.get("geometry_constraints", {}) or {})
        thermal = dict(payload.get("thermal_limits", {}) or {})
        objectives = dict(payload.get("objective_weights", {}) or {})
        fields = dict(payload.get("field_preferences", {}) or {})
    else:
        prefs = dict(payload.get("preferences", {}) or {})
        kpis = dict(payload.get("kpis", {}) or {})
        scenario = {
            "num_modules_min": payload.get("num_modules_min", payload.get("num_cylinders_min")),
            "num_modules_max": payload.get("num_modules_max", payload.get("num_cylinders_max")),
            "heat_load_policy": payload.get("heat_load_policy", "preserve_total_heat"),
        }
        geometry = {
            "min_center_distance": payload.get("min_center_distance", prefs.get("min_center_distance")),
            "wall_clearance": payload.get("wall_clearance", prefs.get("wall_clearance")),
            "inlet_clearance": payload.get("inlet_clearance", prefs.get("inlet_clearance")),
            "outlet_clearance": payload.get("outlet_clearance", prefs.get("outlet_clearance")),
            "x_span": prefs.get("x_span"),
            "y_span": prefs.get("y_span"),
            "keepout_boxes": prefs.get("keepout_boxes", []),
            "protected_boxes": prefs.get("protected_boxes", []),
        }
        thermal = {
            "solid_temperature_max": _legacy_bound(kpis.get("max_solid_temperature")),
            "module_temperature_spread_max": _legacy_bound(kpis.get("module_peak_temperature_spread")),
            "pressure_drop_max": _legacy_bound(kpis.get("pressure_drop")),
            "wall_hot_delta_T": payload.get("temperature_limits", {}).get("wall_hot_delta_T") if isinstance(payload.get("temperature_limits"), Mapping) else None,
            "outlet_hot_delta_T": payload.get("temperature_limits", {}).get("outlet_hot_delta_T") if isinstance(payload.get("temperature_limits"), Mapping) else None,
        }
        objectives = {
            "safety": 1.0,
            "uniformity": 0.8,
            "pressure": 0.4,
            "outlet_mixing": 0.5,
            "wall_protection": 0.5 if bool(prefs.get("avoid_wall_hotspots", False)) else 0.0,
            "plume_avoidance": 0.5,
            "coverage": 0.3 if any(key in prefs for key in ("min_x_coverage", "min_y_coverage", "min_mean_pair_distance")) else 0.0,
        }
        fields = {
            "avoid_downstream_hot_plumes": True,
            "protect_wall_band": bool(prefs.get("avoid_wall_hotspots", False)),
            "protect_outlet_uniformity": "outlet_temperature_nonuniformity" in kpis,
            "min_x_coverage": prefs.get("min_x_coverage"),
            "min_y_coverage": prefs.get("min_y_coverage"),
            "min_mean_pair_distance": prefs.get("min_mean_pair_distance"),
        }
    return {
        "name": payload.get("name", "design_intent"),
        "scenario": scenario,
        "geometry_constraints": geometry,
        "thermal_limits": thermal,
        "objective_weights": objectives,
        "field_preferences": fields,
        "legacy_kpis": dict(payload.get("kpis", {}) or {}),
        "source_payload": dict(payload),
        "is_design_intent": bool(is_design_intent_payload(payload)),
    }


def _legacy_bound(entry: Any) -> Optional[float]:
    if not isinstance(entry, Mapping):
        return None
    mode = str(entry.get("mode", "")).lower().strip()
    if mode in {"max", "upper", "at_most", "range", "between"}:
        value = entry.get("high", entry.get("upper"))
        return None if value is None else float(value)
    return None


def _span(value: Any, fallback: Tuple[float, float]) -> Tuple[float, float]:
    arr = np.asarray(value if value is not None else fallback, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return fallback
    lo, hi = sorted((float(arr[0]), float(arr[1])))
    return lo, hi


def _boxes_mask(xx: np.ndarray, yy: np.ndarray, boxes: Any) -> np.ndarray:
    mask = np.zeros_like(xx, dtype=np.float32)
    for box in boxes or []:
        if isinstance(box, Mapping):
            x0, x1 = _span(box.get("x", box.get("x_span")), (float(np.min(xx)), float(np.max(xx))))
            y0, y1 = _span(box.get("y", box.get("y_span")), (float(np.min(yy)), float(np.max(yy))))
        else:
            arr = np.asarray(box, dtype=np.float64).reshape(-1)
            if arr.size < 4:
                continue
            x0, x1 = sorted((float(arr[0]), float(arr[1])))
            y0, y1 = sorted((float(arr[2]), float(arr[3])))
        mask = np.maximum(mask, ((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)).astype(np.float32))
    return mask


def build_field_intent_maps(
    intent: Mapping[str, Any],
    *,
    domain_length_x: float,
    domain_length_y: float,
    shape: Tuple[int, int] = DEFAULT_FIELD_MAP_SHAPE,
) -> np.ndarray:
    width, height = int(shape[0]), int(shape[1])
    xs = np.linspace(0.0, float(domain_length_x), width, dtype=np.float32)
    ys = np.linspace(0.0, float(domain_length_y), height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    geometry = intent.get("geometry_constraints", {}) if isinstance(intent.get("geometry_constraints"), Mapping) else {}
    fields = intent.get("field_preferences", {}) if isinstance(intent.get("field_preferences"), Mapping) else {}
    protected = _boxes_mask(xx, yy, geometry.get("protected_boxes", []))
    keepout = _boxes_mask(xx, yy, geometry.get("keepout_boxes", []))
    wall_band = np.minimum(yy, float(domain_length_y) - yy) / max(float(domain_length_y), 1.0e-8)
    wall_risk = np.clip(1.0 - wall_band / 0.18, 0.0, 1.0) if bool(fields.get("protect_wall_band", False)) else np.zeros_like(xx, dtype=np.float32)
    outlet = np.clip((xx / max(float(domain_length_x), 1.0e-8) - 0.82) / 0.18, 0.0, 1.0) if bool(fields.get("protect_outlet_uniformity", False)) else np.zeros_like(xx, dtype=np.float32)
    plume = np.clip((xx / max(float(domain_length_x), 1.0e-8) - 0.45) / 0.55, 0.0, 1.0) if bool(fields.get("avoid_downstream_hot_plumes", False)) else np.zeros_like(xx, dtype=np.float32)
    return np.stack([protected, wall_risk, outlet, plume, keepout], axis=0).astype(np.float32)


def build_design_intent_arrays(
    payload: Mapping[str, Any],
    *,
    max_num_modules: int,
    domain_length_x: float,
    domain_length_y: float,
    heat_power_scale: float = 1.0,
    field_shape: Tuple[int, int] = DEFAULT_FIELD_MAP_SHAPE,
) -> Dict[str, Any]:
    intent = normalize_intent_payload(payload)
    scenario = intent["scenario"]
    geometry = intent["geometry_constraints"]
    thermal = intent["thermal_limits"]
    fields = intent["field_preferences"]
    scale = max(float(domain_length_x), float(domain_length_y), 1.0e-8)
    x0, x1 = _span(geometry.get("x_span"), (0.0, float(domain_length_x)))
    y0, y1 = _span(geometry.get("y_span"), (0.0, float(domain_length_y)))
    scalars = np.asarray(
        [
            _finite_float(scenario.get("num_modules_min"), 0.0) / max(float(max_num_modules), 1.0),
            _finite_float(scenario.get("num_modules_max"), max_num_modules) / max(float(max_num_modules), 1.0),
            _finite_float(geometry.get("min_center_distance"), 0.0) / scale,
            _finite_float(geometry.get("wall_clearance"), 0.0) / scale,
            _finite_float(geometry.get("inlet_clearance"), 0.0) / scale,
            _finite_float(geometry.get("outlet_clearance"), 0.0) / scale,
            x0 / max(float(domain_length_x), 1.0e-8),
            x1 / max(float(domain_length_x), 1.0e-8),
            y0 / max(float(domain_length_y), 1.0e-8),
            y1 / max(float(domain_length_y), 1.0e-8),
            _finite_float(thermal.get("solid_temperature_max"), 0.0),
            _finite_float(thermal.get("module_temperature_spread_max"), 0.0),
            _finite_float(thermal.get("pressure_drop_max"), 0.0),
            _finite_float(thermal.get("wall_hot_delta_T"), 0.0),
            _finite_float(thermal.get("outlet_hot_delta_T"), 0.0),
            _finite_float(scenario.get("heat_power_total"), 0.0) / max(float(heat_power_scale), 1.0e-8),
            1.0 if scenario.get("heat_power_total") is not None else 0.0,
            1.0 if bool(fields.get("avoid_downstream_hot_plumes", False)) else 0.0,
            1.0 if bool(fields.get("protect_wall_band", False)) else 0.0,
            1.0 if bool(fields.get("protect_outlet_uniformity", False)) else 0.0,
            _finite_float(fields.get("min_x_coverage"), 0.0) / max(float(domain_length_x), 1.0e-8),
            _finite_float(fields.get("min_y_coverage"), 0.0) / max(float(domain_length_y), 1.0e-8),
            _finite_float(fields.get("min_mean_pair_distance"), 0.0) / scale,
            2.0,
        ],
        dtype=np.float32,
    )
    objectives = np.asarray([max(_finite_float(intent["objective_weights"].get(name), 0.0), 0.0) for name in OBJECTIVE_NAMES], dtype=np.float32)
    maps = build_field_intent_maps(intent, domain_length_x=domain_length_x, domain_length_y=domain_length_y, shape=field_shape)
    constraints = {
        "num_modules_min": scenario.get("num_modules_min"),
        "num_modules_max": scenario.get("num_modules_max"),
        "min_center_distance": geometry.get("min_center_distance"),
        "wall_clearance": geometry.get("wall_clearance"),
        "inlet_clearance": geometry.get("inlet_clearance"),
        "outlet_clearance": geometry.get("outlet_clearance"),
        "heat_power_total": scenario.get("heat_power_total"),
    }
    return {
        "intent": intent,
        "design_intent_vector": scalars,
        "objective_weight_vector": objectives,
        "field_intent_maps": maps,
        "constraints": constraints,
        "x_bounds": _span(geometry.get("x_span"), (0.0, float(domain_length_x))),
        "y_bounds": _span(geometry.get("y_span"), (0.0, float(domain_length_y))),
    }


def training_intent_from_record(
    kpi_dict: Mapping[str, Any],
    *,
    true_count: int,
    domain_length_x: float,
    domain_length_y: float,
    max_num_modules: int,
    rng: np.random.Generator,
    distribution_summary: Optional[Mapping[str, Any]] = None,
    augmentation_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = dict(augmentation_cfg or {})
    dropout = min(max(_finite_float(cfg.get("field_preference_dropout"), 0.3), 0.0), 1.0)
    weights = rng.dirichlet(np.ones(len(OBJECTIVE_NAMES), dtype=np.float64)).astype(np.float32)
    def _limit(name: str, fallback: float, q_low: float, q_high: float) -> float:
        summary = (distribution_summary or {}).get("kpis", distribution_summary or {})
        entry = summary.get(name, {}) if isinstance(summary, Mapping) else {}
        if isinstance(entry, Mapping) and entry.get("p10") is not None:
            lo = _finite_float(entry.get(f"p{int(q_low):02d}", entry.get("p10")), fallback)
            hi = _finite_float(entry.get(f"p{int(q_high):02d}", entry.get("p90")), fallback)
            return float(rng.uniform(min(lo, hi), max(lo, hi)))
        return max(_finite_float(kpi_dict.get(name), fallback), fallback)
    payload = {
        "name": "augmented_training_intent",
        "scenario": {"num_modules_min": true_count, "num_modules_max": true_count, "heat_load_policy": "preserve_total_heat"},
        "geometry_constraints": {"min_center_distance": 1.1, "wall_clearance": 0.05, "inlet_clearance": 0.25, "outlet_clearance": 0.25},
        "thermal_limits": {
            "solid_temperature_max": _limit("max_solid_temperature", _finite_float(kpi_dict.get("max_solid_temperature"), 1.0), 10, 90),
            "module_temperature_spread_max": _limit("module_peak_temperature_spread", _finite_float(kpi_dict.get("module_peak_temperature_spread"), 0.1), 10, 90),
            "pressure_drop_max": _limit("pressure_drop", _finite_float(kpi_dict.get("pressure_drop"), 0.1), 10, 90),
            "wall_hot_delta_T": 0.25,
            "outlet_hot_delta_T": 0.20,
        },
        "objective_weights": {name: float(weights[i]) for i, name in enumerate(OBJECTIVE_NAMES)},
        "field_preferences": {
            "avoid_downstream_hot_plumes": bool(rng.random() > dropout),
            "protect_wall_band": bool(rng.random() > dropout),
            "protect_outlet_uniformity": bool(rng.random() > dropout),
            "min_x_coverage": float(rng.uniform(0.20, 0.45) * domain_length_x) if rng.random() > dropout else 0.0,
            "min_y_coverage": float(rng.uniform(0.20, 0.45) * domain_length_y) if rng.random() > dropout else 0.0,
            "min_mean_pair_distance": float(rng.uniform(1.1, 1.8)) if rng.random() > dropout else 0.0,
        },
    }
    return build_design_intent_arrays(payload, max_num_modules=max_num_modules, domain_length_x=domain_length_x, domain_length_y=domain_length_y)


def compute_design_intent_score(
    kpis: Mapping[str, Any],
    design: Mapping[str, Any],
    intent_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    intent = normalize_intent_payload(intent_spec.get("target_payload", intent_spec.get("source_payload", intent_spec)))
    thermal = intent.get("thermal_limits", {})
    objectives = intent.get("objective_weights", {})
    fields = intent.get("field_preferences", {})
    centers = np.asarray(design.get("centers", []), dtype=np.float32).reshape(-1, 2)
    spread = layout_spread_metrics(centers, num_modules=int(kpis.get("num_modules", centers.shape[0])))

    def over(actual_key: str, limit_key: str) -> float:
        limit = thermal.get(limit_key)
        if limit is None:
            return 0.0
        actual = _finite_float(kpis.get(actual_key), float("nan"))
        limit_f = _finite_float(limit, float("nan"))
        if not math.isfinite(actual) or not math.isfinite(limit_f):
            return 0.0
        return max(0.0, actual - limit_f) / max(abs(limit_f), 1.0)

    hard = 0.0 if bool(kpis.get("valid", True)) else 10.0
    hard += over("pressure_drop", "pressure_drop_max")
    hard += over("max_solid_temperature", "solid_temperature_max")
    hard += over("module_peak_temperature_spread", "module_temperature_spread_max")

    components = {
        "safety": over("max_solid_temperature", "solid_temperature_max") + float(kpis.get("hot_solid_area_fraction", 0.0) or 0.0),
        "uniformity": over("module_peak_temperature_spread", "module_temperature_spread_max") + float(kpis.get("module_mean_temperature_std", 0.0) or 0.0),
        "pressure": over("pressure_drop", "pressure_drop_max"),
        "outlet_mixing": float(kpis.get("outlet_temperature_nonuniformity", 0.0) or 0.0) + float(kpis.get("outlet_hot_fraction", 0.0) or 0.0),
        "wall_protection": float(kpis.get("wall_hot_area_fraction", 0.0) or 0.0),
        "plume_avoidance": float(kpis.get("thermal_plume_area", 0.0) or 0.0) + float(kpis.get("thermal_plume_length", 0.0) or 0.0) / 10.0 + max(float(kpis.get("downstream_reheat_index", 0.0) or 0.0), 0.0),
        "coverage": 0.0,
    }
    if fields.get("min_x_coverage") is not None:
        target = _finite_float(fields.get("min_x_coverage"), 0.0)
        components["coverage"] += max(0.0, target - spread["x_coverage"]) / max(target, 1.0)
    if fields.get("min_y_coverage") is not None:
        target = _finite_float(fields.get("min_y_coverage"), 0.0)
        components["coverage"] += max(0.0, target - spread["y_coverage"]) / max(target, 1.0)
    if fields.get("min_mean_pair_distance") is not None:
        target = _finite_float(fields.get("min_mean_pair_distance"), 0.0)
        components["coverage"] += max(0.0, target - spread["mean_pair_distance"]) / max(target, 1.0)

    weighted = 0.0
    weight_total = 0.0
    for name in OBJECTIVE_NAMES:
        weight = max(_finite_float(objectives.get(name), 0.0), 0.0)
        weighted += weight * float(components.get(name, 0.0))
        weight_total += weight
    objective = weighted / max(weight_total, 1.0e-8)
    field_penalties = {
        "protected_region_hot_penalty": float(kpis.get("hot_solid_area_fraction", 0.0) or 0.0) if np.any(centers) else 0.0,
        "wall_band_hot_penalty": float(kpis.get("wall_hot_area_fraction", 0.0) or 0.0) if bool(fields.get("protect_wall_band", False)) else 0.0,
        "outlet_profile_loss": float(kpis.get("outlet_temperature_nonuniformity", 0.0) or 0.0) if bool(fields.get("protect_outlet_uniformity", False)) else 0.0,
        "downstream_plume_shadowing_loss": float(kpis.get("thermal_plume_area", 0.0) or 0.0) if bool(fields.get("avoid_downstream_hot_plumes", False)) else 0.0,
    }
    field_total = float(sum(field_penalties.values()))
    total = float(hard + objective + field_total)
    return {
        "total_score": total,
        "design_intent_score": total,
        "hard_feasibility_penalty": float(hard),
        "objective_score": float(objective),
        "field_penalty": field_total,
        "components": components,
        "field_penalties": field_penalties,
        "objective_weights": {name: float(_finite_float(objectives.get(name), 0.0)) for name in OBJECTIVE_NAMES},
        "layout_spread": spread,
    }
