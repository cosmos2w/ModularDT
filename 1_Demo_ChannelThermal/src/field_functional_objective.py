from __future__ import annotations

"""Field-functional inverse-design objectives for ChannelThermal.

This module is intentionally independent of the KPI-conditioned inverse model.
It compiles downstream design requirements into deterministic scores over a
frozen forward HONF prediction, decoded geometry, and optional hypergraph
summaries. That makes arbitrary inference-time requirements comparable without
training a new inverse generator for every target specification.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # Torch tensors are accepted if callers already use PyTorch.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class ObjectiveTermResult:
    name: str
    value: float
    raw_value: float
    target: Any
    weight: float
    mode: str
    satisfied: bool
    details: Dict[str, Any]


_DEFAULT_CHANNEL_ORDER: Tuple[str, ...] = ("u", "v", "p", "omega", "temperature")


def _as_array(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if torch is not None and torch.is_tensor(value):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _maybe_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    return value if isinstance(value, Mapping) else None


def _normalize_field_array(value: Any) -> Optional[np.ndarray]:
    try:
        arr = _as_array(value).astype(np.float64, copy=False)
    except Exception:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr
    return None


def _channel_index(field_name: str, field_dim: int, channel_order: Optional[Sequence[Any]]) -> int:
    aliases = {
        "t": "temperature",
        "temp": "temperature",
        "temperature": "temperature",
        "pressure": "p",
    }
    name = aliases.get(str(field_name).lower(), str(field_name).lower())
    order = [str(item).lower() for item in (channel_order or _DEFAULT_CHANNEL_ORDER)]
    if name in order:
        idx = order.index(name)
        if 0 <= idx < field_dim:
            return idx
    if name in {"temperature", "t"}:
        return max(field_dim - 1, 0)
    if name in {"p", "pressure"}:
        return min(2, max(field_dim - 1, 0))
    if name in {"u", "vx"}:
        return 0
    if name in {"v", "vy"}:
        return min(1, max(field_dim - 1, 0))
    return max(field_dim - 1, 0)


def extract_field_array(forward_prediction: Optional[Mapping[str, Any]], field_name: str) -> Optional[np.ndarray]:
    """Extract a dense 2D field from common forward-prediction payload shapes."""

    if not isinstance(forward_prediction, Mapping):
        return None
    payload: Mapping[str, Any] = forward_prediction
    nested = _maybe_mapping(payload.get("forward_prediction"))
    if nested is not None:
        direct = extract_field_array(nested, field_name)
        if direct is not None:
            return direct

    name = str(field_name)
    for key in (name, name.lower(), name.upper(), "T" if name.lower() in {"temperature", "t"} else name):
        if key in payload:
            arr = _normalize_field_array(payload[key])
            if arr is not None:
                return arr[..., _channel_index(name, arr.shape[-1], payload.get("channel_order"))] if arr.ndim == 3 else arr

    for key in ("pred_field_grid", "pred_field", "field", "global_field", "field_grid"):
        if key not in payload:
            continue
        arr = _normalize_field_array(payload[key])
        if arr is None:
            continue
        if arr.ndim == 2:
            return arr
        return arr[..., _channel_index(name, arr.shape[-1], payload.get("channel_order"))]
    return None


def _domain(forward_prediction: Optional[Mapping[str, Any]], layout: Optional[Mapping[str, Any]]) -> Tuple[float, float]:
    for source in (forward_prediction, layout):
        if not isinstance(source, Mapping):
            continue
        domain = source.get("domain") if isinstance(source.get("domain"), Mapping) else source
        lx = domain.get("domain_length_x", domain.get("lx", domain.get("Lx"))) if isinstance(domain, Mapping) else None
        ly = domain.get("domain_length_y", domain.get("ly", domain.get("Ly"))) if isinstance(domain, Mapping) else None
        if lx is not None and ly is not None:
            return max(_finite_float(lx, 12.0), 1.0e-8), max(_finite_float(ly, 4.0), 1.0e-8)
    return 12.0, 4.0


def _grid_coords(shape: Tuple[int, int], forward_prediction: Optional[Mapping[str, Any]], layout: Optional[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = shape
    if isinstance(forward_prediction, Mapping):
        x_grid = forward_prediction.get("x_grid")
        y_grid = forward_prediction.get("y_grid")
        if x_grid is not None and y_grid is not None:
            x = np.resize(_as_array(x_grid).astype(np.float64, copy=False), shape)
            y = np.resize(_as_array(y_grid).astype(np.float64, copy=False), shape)
            return x, y
    lx, ly = _domain(forward_prediction, layout)
    x = np.broadcast_to(np.linspace(0.0, lx, w, dtype=np.float64)[None, :], shape)
    y = np.broadcast_to(np.linspace(0.0, ly, h, dtype=np.float64)[:, None], shape)
    return x, y


def _region_mask(region: Any, shape: Tuple[int, int], forward_prediction: Optional[Mapping[str, Any]], layout: Optional[Mapping[str, Any]]) -> Optional[np.ndarray]:
    h, w = shape
    if region is None or region == "all":
        return np.ones(shape, dtype=bool)
    x, y = _grid_coords(shape, forward_prediction, layout)
    lx, ly = _domain(forward_prediction, layout)
    if isinstance(region, str):
        key = region.lower().strip()
        if key == "outlet_band":
            return x >= 0.92 * lx
        if key == "inlet_band":
            return x <= 0.08 * lx
        if key == "wall_band":
            return (y <= 0.08 * ly) | (y >= 0.92 * ly)
        if key == "lower_wall_band":
            return y <= 0.08 * ly
        if key == "upper_wall_band":
            return y >= 0.92 * ly
        return np.ones(shape, dtype=bool)
    if isinstance(region, Mapping):
        rtype = str(region.get("type", "")).lower().strip()
        if rtype == "box":
            xr = region.get("x_range", [0.0, lx])
            yr = region.get("y_range", [0.0, ly])
            x0, x1 = sorted((_finite_float(xr[0], 0.0), _finite_float(xr[1], lx))) if isinstance(xr, Sequence) and len(xr) >= 2 else (0.0, lx)
            y0, y1 = sorted((_finite_float(yr[0], 0.0), _finite_float(yr[1], ly))) if isinstance(yr, Sequence) and len(yr) >= 2 else (0.0, ly)
            return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if rtype == "mask_file":
            path = Path(str(region.get("path", ""))).expanduser()
            if not path.exists():
                return None
            try:
                if path.suffix.lower() == ".npy":
                    mask = np.load(path)
                else:
                    mask = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
                return np.resize(mask.astype(bool), (h, w))
            except Exception:
                return None
    return np.ones(shape, dtype=bool)


def _finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _apply_numeric_operator(values: np.ndarray, operator: str, term: Mapping[str, Any]) -> Optional[float]:
    finite = _finite_values(values)
    op = str(operator or "identity").lower().strip()
    if finite.size == 0:
        return None
    if op == "identity":
        return float(finite[0]) if finite.size == 1 else float(np.mean(finite))
    if op == "max":
        return float(np.max(finite))
    if op == "mean":
        return float(np.mean(finite))
    if op == "p95":
        return float(np.percentile(finite, 95.0))
    if op == "std":
        return float(np.std(finite))
    if op == "min":
        return float(np.min(finite))
    if op == "mean_above_threshold":
        threshold = _finite_float(term.get("threshold"), float("nan"))
        if not math.isfinite(threshold):
            return None
        above = finite[finite > threshold]
        return float(np.mean(above - threshold)) if above.size else 0.0
    if op == "area_above_threshold":
        threshold = _finite_float(term.get("threshold"), float("nan"))
        if not math.isfinite(threshold):
            return None
        return float(np.mean(finite > threshold))
    if op == "gradient_energy":
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2:
            return None
        grad_y, grad_x = np.gradient(arr)
        energy = grad_x**2 + grad_y**2
        return float(np.nanmean(energy[np.isfinite(arr)]))
    return None


def _centers(layout: Optional[Mapping[str, Any]]) -> np.ndarray:
    if not isinstance(layout, Mapping):
        return np.zeros((0, 2), dtype=np.float64)
    raw = layout.get("centers", layout.get("module_centers", []))
    arr = _as_array(raw).astype(np.float64, copy=False).reshape(-1, 2) if np.asarray(raw).size else np.zeros((0, 2), dtype=np.float64)
    mask = layout.get("mask", layout.get("module_present"))
    if mask is not None and arr.size:
        keep = _as_array(mask).reshape(-1)[: arr.shape[0]] > 0.5
        arr = arr[keep]
    count = layout.get("count")
    if count is not None:
        arr = arr[: max(int(count), 0)]
    return arr[np.all(np.isfinite(arr), axis=1)] if arr.size else arr


def _pair_distances(points: np.ndarray) -> np.ndarray:
    vals = []
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            vals.append(float(np.linalg.norm(points[i] - points[j])))
    return np.asarray(vals, dtype=np.float64)


def _layout_value(layout: Optional[Mapping[str, Any]], operator: str) -> Optional[float]:
    pts = _centers(layout)
    op = str(operator or "count").lower().strip()
    if op == "count":
        return float(pts.shape[0])
    if pts.shape[0] == 0:
        return 0.0
    x_span = float(np.max(pts[:, 0]) - np.min(pts[:, 0])) if pts.shape[0] >= 2 else 0.0
    y_span = float(np.max(pts[:, 1]) - np.min(pts[:, 1])) if pts.shape[0] >= 2 else 0.0
    if op == "bbox_area":
        return float(x_span * y_span)
    if op == "x_span":
        return x_span
    if op == "y_span":
        return y_span
    pair = _pair_distances(pts)
    if op == "mean_pair_distance":
        return float(np.mean(pair)) if pair.size else 0.0
    if op == "min_pair_distance":
        return float(np.min(pair)) if pair.size else 0.0
    return None


def _array_from_mapping(mapping: Optional[Mapping[str, Any]], keys: Sequence[str]) -> np.ndarray:
    if not isinstance(mapping, Mapping):
        return np.zeros((0,), dtype=np.float64)
    for key in keys:
        if key in mapping:
            try:
                return _as_array(mapping[key]).astype(np.float64, copy=False).reshape(-1)
            except Exception:
                return np.zeros((0,), dtype=np.float64)
    summary = mapping.get("summary")
    if isinstance(summary, Mapping):
        return _array_from_mapping(summary, keys)
    decoded = mapping.get("decoded")
    if isinstance(decoded, Mapping):
        return _array_from_mapping(decoded, keys)
    return np.zeros((0,), dtype=np.float64)


def _hypergraph_value(
    operator: str,
    planned: Optional[Mapping[str, Any]],
    realized: Optional[Mapping[str, Any]],
    consistency: Optional[Mapping[str, Any]],
) -> Optional[float]:
    op = str(operator or "").lower().strip()
    if op == "plan_realization_distance":
        if isinstance(consistency, Mapping):
            for key in ("total", "score", "distance", "plan_realization_distance"):
                if key in consistency:
                    return _finite_float(consistency[key], 0.0)
        p = _array_from_mapping(planned, ("vector", "hyper_strength", "strength"))
        r = _array_from_mapping(realized, ("vector", "hyper_strength", "strength"))
        n = min(p.size, r.size)
        return float(np.mean(np.abs(p[:n] - r[:n]))) if n else None
    source = realized if isinstance(realized, Mapping) else planned
    strength = _array_from_mapping(source, ("hyper_strength", "strength", "edge_strength"))
    active = _array_from_mapping(source, ("edge_active", "active", "mask"))
    if active.size == 0:
        active = strength > 1.0e-6 if strength.size else np.zeros((0,), dtype=np.float64)
    if op == "active_edge_count":
        return float(np.sum(active > 0.5))
    if op == "active_edge_entropy":
        vals = strength[np.isfinite(strength)]
        if vals.size == 0:
            return 0.0
        probs = np.abs(vals) / max(float(np.sum(np.abs(vals))), 1.0e-12)
        probs = probs[probs > 0.0]
        return float(-np.sum(probs * np.log(probs)))
    if op == "strength_mean":
        vals = strength[np.isfinite(strength)]
        return float(np.mean(vals)) if vals.size else 0.0
    if op == "strength_max":
        vals = strength[np.isfinite(strength)]
        return float(np.max(vals)) if vals.size else 0.0
    return None


class FieldFunctionalObjective:
    def __init__(self, spec: Mapping[str, Any], *, default_temperature_key: str = "T"):
        self.spec = dict(spec)
        self.default_temperature_key = str(default_temperature_key)
        self.name = str(self.spec.get("name", "field_functional_objective"))
        self.missing_term_penalty = _finite_float(self.spec.get("missing_term_penalty", 0.0), 0.0)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "FieldFunctionalObjective":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    def evaluate(
        self,
        *,
        forward_prediction: Optional[Mapping[str, Any]] = None,
        kpis: Optional[Mapping[str, float]] = None,
        layout: Optional[Mapping[str, Any]] = None,
        planned_hypergraph: Optional[Mapping[str, Any]] = None,
        realized_hypergraph: Optional[Mapping[str, Any]] = None,
        hypergraph_consistency: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        term_results = []
        soft_score = 0.0
        hard_violation_score = 0.0
        for section in ("scalar_kpi_terms", "field_terms", "layout_terms", "hypergraph_terms"):
            for term in self.spec.get(section, []) or []:
                result = self._evaluate_term(
                    section,
                    term,
                    forward_prediction=forward_prediction,
                    kpis=kpis,
                    layout=layout,
                    planned_hypergraph=planned_hypergraph,
                    realized_hypergraph=realized_hypergraph,
                    hypergraph_consistency=hypergraph_consistency,
                )
                term_results.append(result)
                if bool(term.get("hard", False)):
                    hard_violation_score += max(float(result.value), 0.0)
                else:
                    soft_score += float(result.value)

        hard_result = self._evaluate_hard_constraints(layout, self.spec.get("hard_constraints"))
        hard_violation_score += float(hard_result["hard_violation_score"])
        hard_ok = bool(hard_result["satisfied"])
        bounded_results = [r for r in term_results if r.mode in {"exact", "target_range", "upper_bound", "lower_bound", "mean_above_threshold_minimize", "area_above_threshold_minimize"} or bool(r.details.get("hard"))]
        num_satisfied = int(sum(1 for r in term_results if bool(r.satisfied)))
        satisfied = bool(hard_ok and all(r.satisfied for r in bounded_results))
        total_score = float(soft_score + hard_violation_score)
        return _jsonable(
            {
                "total_score": total_score,
                "hard_violation_score": float(hard_violation_score),
                "term_results": [asdict(result) for result in term_results],
                "hard_constraints": hard_result,
                "satisfied": satisfied,
                "num_satisfied": num_satisfied,
                "num_terms": int(len(term_results)),
                "spec_name": self.name,
            }
        )

    def _unavailable(self, term: Mapping[str, Any], reason: str) -> ObjectiveTermResult:
        hard = bool(term.get("hard", False))
        penalty = _finite_float(term.get("missing_term_penalty", 1.0 if hard else self.missing_term_penalty), 1.0 if hard else 0.0)
        weight = _finite_float(term.get("weight", 1.0), 1.0)
        return ObjectiveTermResult(
            name=str(term.get("name", term.get("kpi", term.get("operator", "term")))),
            value=float(weight * penalty),
            raw_value=float("nan"),
            target=_jsonable(term.get("value", term.get("range"))),
            weight=weight,
            mode=str(term.get("mode", "minimize")),
            satisfied=False,
            details={"available": False, "reason": reason, "hard": hard},
        )

    def _evaluate_term(self, section: str, term: Mapping[str, Any], **payload: Any) -> ObjectiveTermResult:
        weight = _finite_float(term.get("weight", 1.0), 1.0)
        mode = str(term.get("mode", "minimize")).lower().strip()
        raw: Optional[float] = None
        details: Dict[str, Any] = {"section": section, "available": True, "hard": bool(term.get("hard", False))}
        if section == "scalar_kpi_terms":
            kpis = payload.get("kpis") if isinstance(payload.get("kpis"), Mapping) else {}
            key = str(term.get("kpi", term.get("name", "")))
            if key in kpis:
                raw = _finite_float(kpis[key], float("nan"))
            else:
                return self._unavailable(term, f"kpi {key!r} unavailable")
        elif section == "field_terms":
            field_name = str(term.get("field", self.default_temperature_key))
            field = extract_field_array(payload.get("forward_prediction"), field_name)
            if field is None and term.get("fallback_kpi") is not None:
                kpis = payload.get("kpis") if isinstance(payload.get("kpis"), Mapping) else {}
                key = str(term.get("fallback_kpi"))
                if key in kpis:
                    raw = _finite_float(kpis[key], float("nan"))
                    details["fallback_kpi"] = key
            if raw is None:
                if field is None:
                    return self._unavailable(term, f"field {field_name!r} unavailable")
                mask = _region_mask(term.get("region", "all"), field.shape, payload.get("forward_prediction"), payload.get("layout"))
                if mask is None:
                    return self._unavailable(term, "region mask unavailable")
                values = np.asarray(field, dtype=np.float64)
                selected = np.where(np.resize(mask, field.shape), values, np.nan)
                raw = _apply_numeric_operator(selected, str(term.get("operator", "mean")), term)
                details["field"] = field_name
                details["region"] = _jsonable(term.get("region", "all"))
        elif section == "layout_terms":
            raw = _layout_value(payload.get("layout"), str(term.get("operator", "count")))
        elif section == "hypergraph_terms":
            raw = _hypergraph_value(str(term.get("operator", "")), payload.get("planned_hypergraph"), payload.get("realized_hypergraph"), payload.get("hypergraph_consistency"))
        if raw is None or not math.isfinite(float(raw)):
            return self._unavailable(term, "operator value unavailable")
        contribution, satisfied, target = self._score_raw_value(float(raw), term, mode, weight)
        return ObjectiveTermResult(
            name=str(term.get("name", term.get("kpi", term.get("operator", "term")))),
            value=float(contribution),
            raw_value=float(raw),
            target=_jsonable(target),
            weight=weight,
            mode=mode,
            satisfied=bool(satisfied),
            details=details,
        )

    def _score_raw_value(self, raw: float, term: Mapping[str, Any], mode: str, weight: float) -> Tuple[float, bool, Any]:
        if mode == "minimize":
            return weight * raw, True, None
        if mode == "maximize":
            return -weight * raw, True, None
        if mode == "exact":
            target = _finite_float(term.get("value", term.get("target")), 0.0)
            tol = max(_finite_float(term.get("tolerance", 0.0), 0.0), 0.0)
            violation = max(abs(raw - target) - tol, 0.0)
            return weight * violation, violation <= 1.0e-12, target
        if mode == "target_range":
            rng = term.get("range", term.get("target", [0.0, 0.0]))
            low, high = sorted((_finite_float(rng[0], 0.0), _finite_float(rng[1], 0.0))) if isinstance(rng, Sequence) and len(rng) >= 2 else (0.0, 0.0)
            violation = max(low - raw, 0.0, raw - high)
            return weight * violation, violation <= 1.0e-12, [low, high]
        if mode == "upper_bound":
            target = _finite_float(term.get("value", term.get("upper", term.get("target"))), 0.0)
            violation = max(raw - target, 0.0)
            return weight * violation, violation <= 1.0e-12, target
        if mode == "lower_bound":
            target = _finite_float(term.get("value", term.get("lower", term.get("target"))), 0.0)
            violation = max(target - raw, 0.0)
            return weight * violation, violation <= 1.0e-12, target
        if mode in {"mean_above_threshold_minimize", "area_above_threshold_minimize"}:
            return weight * raw, raw <= 1.0e-12, term.get("threshold")
        return weight * raw, True, None

    def _evaluate_hard_constraints(self, layout: Optional[Mapping[str, Any]], constraints: Any) -> Dict[str, Any]:
        if not isinstance(constraints, Mapping) or not constraints:
            return {"hard_violation_score": 0.0, "satisfied": True, "violations": {}, "details": {}}
        pts = _centers(layout)
        violations: Dict[str, float] = {}
        count = int(pts.shape[0])
        num_range = constraints.get("num_modules")
        if isinstance(num_range, Sequence) and len(num_range) >= 2:
            lo, hi = int(num_range[0]), int(num_range[1])
            violations["num_modules"] = float(max(lo - count, 0, count - hi))
        for key, actual in self._layout_constraint_values(pts, layout).items():
            if key not in constraints:
                continue
            required = _finite_float(constraints[key], 0.0)
            violations[key] = float(max(required - actual, 0.0))
        total = float(sum(max(v, 0.0) for v in violations.values()))
        return _jsonable({"hard_violation_score": total, "satisfied": total <= 1.0e-12, "violations": violations, "details": {"num_modules": count}})

    def _layout_constraint_values(self, pts: np.ndarray, layout: Optional[Mapping[str, Any]]) -> Dict[str, float]:
        lx, ly = _domain(None, layout)
        radius = _finite_float(layout.get("module_radius", 0.45), 0.45) if isinstance(layout, Mapping) else 0.45
        if pts.shape[0] == 0:
            return {"min_center_distance": float("inf"), "wall_clearance": float("inf"), "inlet_clearance": float("inf"), "outlet_clearance": float("inf")}
        pairs = _pair_distances(pts)
        return {
            "min_center_distance": float(np.min(pairs)) if pairs.size else float("inf"),
            "wall_clearance": float(np.min(np.minimum(pts[:, 1], ly - pts[:, 1]) - radius)),
            "inlet_clearance": float(np.min(pts[:, 0] - radius)),
            "outlet_clearance": float(np.min(lx - pts[:, 0] - radius)),
        }
