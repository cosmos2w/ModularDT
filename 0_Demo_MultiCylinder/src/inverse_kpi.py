from __future__ import annotations

"""KPI utilities for the multi-cylinder inverse-design demo.

The functions in this file are intentionally standalone: training uses them to
build supervised inverse targets, and evaluation uses the same definitions to
score generated designs after frozen-forward verification.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Torch is optional for this small utility module.
    import torch
except Exception:  # pragma: no cover - exercised only in minimal environments.
    torch = None  # type: ignore[assignment]


DEFAULT_KPI_NAMES: Tuple[str, ...] = (
    "mean_abs_omega",
    "enstrophy",
    "max_abs_omega",
    "kinetic_energy",
    "pressure_range",
    "wake_deficit",
    "wake_mixing",
    "fluctuation_energy",
    "downstream_omega_area",
    "phase_signal_amplitude",
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
    "re_scaled",
    "re_mask",
    "num_cylinders_min_scaled",
    "num_cylinders_max_scaled",
    "min_center_distance_scaled",
)


def _as_numpy(array: Any) -> np.ndarray:
    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float(0.0 if default is None else default)
    return scalar if math.isfinite(scalar) else float(0.0 if default is None else default)


def _channel_index(channel_order: Optional[Sequence[str]], name: str, fallback: int, field_dim: int) -> int:
    if channel_order:
        normalized = [str(ch).lower() for ch in channel_order]
        if name in normalized:
            idx = normalized.index(name)
            if 0 <= idx < field_dim:
                return idx
    return min(max(int(fallback), 0), max(field_dim - 1, 0))


def _domain_lengths(
    x_grid: Optional[Any],
    y_grid: Optional[Any],
    domain: Optional[Mapping[str, Any]],
) -> Tuple[Optional[float], Optional[float], float, float]:
    if domain:
        lx = domain.get("lx", domain.get("domain_length_x", domain.get("Lx")))
        ly = domain.get("ly", domain.get("domain_length_y", domain.get("Ly")))
        xmin = _finite_float(domain.get("xmin", 0.0), 0.0)
        ymin = _finite_float(domain.get("ymin", 0.0), 0.0)
        if lx is not None and ly is not None:
            return _finite_float(lx, 0.0), _finite_float(ly, 0.0), xmin, ymin

    if x_grid is None or y_grid is None:
        return None, None, 0.0, 0.0

    x_arr = _as_numpy(x_grid).astype(np.float64, copy=False)
    y_arr = _as_numpy(y_grid).astype(np.float64, copy=False)
    if x_arr.size == 0 or y_arr.size == 0:
        return None, None, 0.0, 0.0

    def length_from_grid(arr: np.ndarray, axis: int) -> Tuple[float, float]:
        vals = arr[0, :] if axis == 1 and arr.ndim == 2 else arr[:, 0] if arr.ndim == 2 else arr.reshape(-1)
        vals = np.asarray(vals, dtype=np.float64)
        vals = np.unique(vals[np.isfinite(vals)])
        if vals.size <= 1:
            return float(np.nanmax(arr) - np.nanmin(arr)), float(np.nanmin(arr))
        diffs = np.diff(np.sort(vals))
        step = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
        return float(np.nanmax(vals) - np.nanmin(vals) + step), float(np.nanmin(vals) - 0.5 * step)

    lx, xmin = length_from_grid(x_arr, axis=1)
    ly, ymin = length_from_grid(y_arr, axis=0)
    return lx, ly, xmin, ymin


def _downstream_masks(
    field_cycle: np.ndarray,
    x_grid: Optional[Any],
    y_grid: Optional[Any],
    domain: Optional[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    _, height, width, _ = field_cycle.shape
    lx, ly, xmin, _ = _domain_lengths(x_grid, y_grid, domain)
    if x_grid is not None and lx is not None and lx > 0:
        x_arr = _as_numpy(x_grid).astype(np.float64, copy=False)
        x_rel = np.mod(x_arr - xmin, lx)
        downstream = x_rel > 0.65 * lx
        upstream = x_rel < 0.35 * lx
    else:
        x_idx = np.arange(width)[None, :]
        downstream = np.broadcast_to(x_idx > 0.65 * max(width - 1, 1), (height, width))
        upstream = np.broadcast_to(x_idx < 0.35 * max(width - 1, 1), (height, width))

    if not np.any(downstream):
        downstream = np.ones((height, width), dtype=bool)
    if not np.any(upstream):
        upstream = np.ones((height, width), dtype=bool)
    return downstream.astype(bool), upstream.astype(bool), lx, ly


def compute_cycle_kpis(
    field_cycle: Any,
    x_grid: Optional[Any] = None,
    y_grid: Optional[Any] = None,
    channel_order: Optional[Sequence[str]] = None,
    domain: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Compute field-aware KPIs from a canonical cycle or predicted cycle.

    Args:
        field_cycle: Array shaped ``[T, H, W, C]``. Channel convention defaults
            to ``u, v, p, omega``.
        x_grid/y_grid: Optional physical grids shaped ``[H, W]``.
        channel_order: Optional names for the last dimension.
        domain: Optional physical bounds/lengths. ``lx`` and ``ly`` are used
            when available.
    """

    arr = _as_numpy(field_cycle).astype(np.float64, copy=False)
    if arr.ndim != 4:
        raise ValueError(f"field_cycle must have shape [T,H,W,C], got {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0 or arr.shape[3] < 1:
        raise ValueError(f"field_cycle has empty dimensions: {arr.shape}.")

    field_dim = int(arr.shape[-1])
    u_idx = _channel_index(channel_order, "u", 0, field_dim)
    v_idx = _channel_index(channel_order, "v", 1, field_dim)
    p_idx = _channel_index(channel_order, "p", 2, field_dim)
    omega_idx = _channel_index(channel_order, "omega", 3, field_dim)

    u = arr[..., u_idx]
    v = arr[..., v_idx]
    p = arr[..., p_idx]
    omega = arr[..., omega_idx]
    downstream, upstream, _, _ = _downstream_masks(arr, x_grid, y_grid, domain)

    omega_abs = np.abs(omega)
    enstrophy_t = np.mean(np.square(omega), axis=(1, 2))
    time_mean = np.mean(arr, axis=0, keepdims=True)

    upstream_u = float(np.mean(u[:, upstream]))
    downstream_u = float(np.mean(u[:, downstream]))
    global_u = float(np.mean(u))
    reference_u = upstream_u if abs(upstream_u) > 1.0e-8 else global_u
    wake_deficit = reference_u - downstream_u

    downstream_u_fluct = u[:, downstream] - np.mean(u[:, downstream], axis=0, keepdims=True)
    downstream_v = v[:, downstream]
    wake_mixing = float(np.mean(np.sqrt(np.square(downstream_v) + np.square(downstream_u_fluct))))

    omega_threshold = 0.25 * float(np.max(omega_abs)) if omega_abs.size else 0.0
    downstream_area = float(np.mean(omega_abs[:, downstream] > omega_threshold)) if omega_threshold > 0.0 else 0.0

    return {
        "mean_abs_omega": float(np.mean(omega_abs)),
        "enstrophy": float(np.mean(np.square(omega))),
        "max_abs_omega": float(np.max(omega_abs)),
        "kinetic_energy": float(np.mean(np.square(u) + np.square(v))),
        "pressure_range": float(np.max(p) - np.min(p)),
        "wake_deficit": float(wake_deficit),
        "wake_mixing": float(wake_mixing),
        "fluctuation_energy": float(np.mean(np.square(arr - time_mean))),
        "downstream_omega_area": downstream_area,
        "phase_signal_amplitude": float(np.max(enstrophy_t) - np.min(enstrophy_t)),
    }


def kpi_vector_from_dict(kpi_dict: Mapping[str, Any], kpi_names: Sequence[str]) -> np.ndarray:
    return np.asarray([_finite_float(kpi_dict.get(name, 0.0), 0.0) for name in kpi_names], dtype=np.float32)


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


def _normalize_scalar(name: str, value: float, kpi_names: Sequence[str], stats: Optional[Mapping[str, Any]], normalize: bool) -> float:
    if not normalize or stats is None:
        return float(value)
    try:
        idx = list(kpi_names).index(name)
    except ValueError:
        return float(value)
    mean, std = _stats_arrays(stats, len(kpi_names))
    return float((float(value) - float(mean[idx])) / max(float(std[idx]), 1.0e-8))


def _parse_target_entry(name: str, entry: Any) -> Dict[str, float | str]:
    if isinstance(entry, Mapping):
        mode = str(entry.get("mode", "exact")).lower().strip()
        weight = _finite_float(entry.get("weight", 1.0), 1.0)
        value = entry.get("value", entry.get("target"))
        low = entry.get("low", entry.get("lower"))
        high = entry.get("high", entry.get("upper"))
        scale = entry.get("scale")
        return {
            "mode": mode,
            "value": _finite_float(value, float("nan")),
            "low": _finite_float(low, float("nan")),
            "high": _finite_float(high, float("nan")),
            "weight": weight,
            "scale": _finite_float(scale, float("nan")),
        }
    return {
        "mode": "exact",
        "value": _finite_float(entry, 0.0),
        "low": float("nan"),
        "high": float("nan"),
        "weight": 1.0,
        "scale": float("nan"),
    }


def build_target_spec_vector(
    kpi_dict: Optional[Mapping[str, Any]] = None,
    kpi_names: Optional[Sequence[str]] = None,
    *,
    kpi_targets: Optional[Mapping[str, Any]] = None,
    stats: Optional[Mapping[str, Any]] = None,
    normalize: bool = False,
    re_value: Optional[float] = None,
    num_cylinders_min: Optional[int] = None,
    num_cylinders_max: Optional[int] = None,
    min_center_distance: Optional[float] = None,
    max_num_cylinders: int = 8,
    re_scale: float = 200.0,
    domain_length_scale: float = 24.0,
    return_spec: bool = False,
) -> np.ndarray | Dict[str, Any]:
    """Build the numeric inverse target vector.

    Vector layout is ``[values, value_mask, lower, lower_mask, upper,
    upper_mask, weights, constraints]``. KPI values/bounds can be normalized
    with dataset stats; Re/count/distance constraints are scaled to compact
    numeric ranges.
    """

    names = tuple(kpi_names or DEFAULT_KPI_NAMES)
    k = len(names)
    raw_targets: Dict[str, Any] = dict(kpi_targets or {})
    if kpi_dict:
        for name in names:
            if name not in raw_targets and name in kpi_dict:
                raw_targets[name] = {"mode": "exact", "value": float(kpi_dict[name]), "weight": 1.0}

    values = np.zeros(k, dtype=np.float32)
    value_mask = np.zeros(k, dtype=np.float32)
    lower = np.zeros(k, dtype=np.float32)
    lower_mask = np.zeros(k, dtype=np.float32)
    upper = np.zeros(k, dtype=np.float32)
    upper_mask = np.zeros(k, dtype=np.float32)
    weights = np.zeros(k, dtype=np.float32)
    parsed: Dict[str, Dict[str, float | str]] = {}

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
            upper[i] = values[i]
            upper_mask[i] = 0.0
        elif mode == "maximize":
            value_mask[i] = 1.0
            values[i] = _normalize_scalar(name, 0.0, names, stats, normalize)
            lower[i] = values[i]
            lower_mask[i] = 0.0
        else:
            if not math.isfinite(val):
                val = 0.5 * (lo + hi) if math.isfinite(lo) and math.isfinite(hi) else 0.0
            values[i] = _normalize_scalar(name, val, names, stats, normalize)
            value_mask[i] = 1.0

    constraints = np.asarray(
        [
            0.0 if re_value is None else float(re_value) / max(float(re_scale), 1.0e-8),
            0.0 if re_value is None else 1.0,
            0.0 if num_cylinders_min is None else float(num_cylinders_min) / max(float(max_num_cylinders), 1.0),
            1.0 if num_cylinders_max is None else float(num_cylinders_max) / max(float(max_num_cylinders), 1.0),
            0.0 if min_center_distance is None else float(min_center_distance) / max(float(domain_length_scale), 1.0e-8),
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
                "Re": re_value,
                "num_cylinders_min": num_cylinders_min,
                "num_cylinders_max": num_cylinders_max,
                "min_center_distance": min_center_distance,
                "max_num_cylinders": max_num_cylinders,
                "re_scale": re_scale,
                "domain_length_scale": domain_length_scale,
            },
            "layout": {
                "blocks": list(TARGET_BLOCKS),
                "constraint_names": list(CONSTRAINT_NAMES),
            },
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
    """Return a meaningful target scale for ranking-style KPI objectives."""

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

    Lower ``total_score`` is better. Constraint penalties are additive and kept
    separate so callers can rank by validity first if desired.

    ``range``, ``max``/``upper``, and ``min``/``lower`` are constraint-style
    objectives: candidates inside the acceptable region receive zero KPI error.
    ``minimize`` and ``maximize`` are ranking objectives: they keep ordering
    candidates even when no hard acceptable bound is supplied, so they require a
    meaningful scale from the target entry, KPI stats, or a target bound.
    """

    names = list(target_spec.get("kpi_names", DEFAULT_KPI_NAMES))
    stats = target_spec.get("kpi_stats")
    target_entries = target_spec.get("kpi_targets")
    if target_entries is None and "kpis" in target_spec:
        target_entries = target_spec["kpis"]
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

    per_errors: Dict[str, float] = {}
    weighted_total = 0.0
    weight_total = 0.0
    for name, raw_entry in target_entries.items():
        if name not in kpi_dict:
            continue
        entry = _parse_target_entry(str(name), raw_entry)
        candidate = _finite_float(kpi_dict.get(name, 0.0), 0.0)
        mode = str(entry["mode"])
        weight = max(float(entry["weight"]), 0.0)
        if weight <= 0.0:
            continue
        value = float(entry["value"])
        low = float(entry["low"])
        high = float(entry["high"])
        scale = float(entry["scale"])

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
            error = max(candidate, 0.0) / _scale_for_target(str(name), names, stats, scale, value, low, high)
        elif mode == "maximize":
            denom = _scale_for_target(str(name), names, stats, scale, value, low, high)
            if math.isfinite(low):
                error = max(0.0, low - candidate) / denom
            else:
                error = -candidate / denom
        elif mode == "vector":
            # Vector scoring is mainly for exact supervised targets. Bound masks
            # are respected if present.
            low_mask = _finite_float(raw_entry.get("low_mask", 0.0), 0.0) if isinstance(raw_entry, Mapping) else 0.0
            high_mask = _finite_float(raw_entry.get("high_mask", 0.0), 0.0) if isinstance(raw_entry, Mapping) else 0.0
            value_mask = _finite_float(raw_entry.get("value_mask", 0.0), 0.0) if isinstance(raw_entry, Mapping) else 0.0
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

        per_errors[str(name)] = float(error)
        weighted_total += weight * float(error)
        weight_total += weight

    constraint_penalty = 0.0
    constraints = target_spec.get("constraints", target_spec)
    count = kpi_dict.get("num_cylinders", kpi_dict.get("count"))
    if count is not None:
        count_val = int(round(_finite_float(count, 0.0)))
        n_min = constraints.get("num_cylinders_min") if isinstance(constraints, Mapping) else None
        n_max = constraints.get("num_cylinders_max") if isinstance(constraints, Mapping) else None
        if n_min is not None and count_val < int(n_min):
            constraint_penalty += float(int(n_min) - count_val)
        if n_max is not None and count_val > int(n_max):
            constraint_penalty += float(count_val - int(n_max))
    min_dist_target = constraints.get("min_center_distance") if isinstance(constraints, Mapping) else None
    min_dist_actual = kpi_dict.get("min_center_distance")
    if min_dist_target is not None and min_dist_actual is not None:
        deficit = float(min_dist_target) - _finite_float(min_dist_actual, 0.0)
        if deficit > 0.0:
            constraint_penalty += deficit / max(float(min_dist_target), 1.0e-8)
    if bool(kpi_dict.get("valid", True)) is False:
        constraint_penalty += 10.0

    normalized_total = weighted_total / max(weight_total, 1.0e-8)
    return {
        "total_score": float(normalized_total + constraint_penalty),
        "per_kpi_errors": per_errors,
        "constraint_penalty": float(constraint_penalty),
    }
