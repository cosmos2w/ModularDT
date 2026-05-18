from __future__ import annotations

"""Raw-layout inverse-design baselines using frozen forward verification.

These baselines deliberately do not depend on the forthcoming design atlas
model. Raw CEM is the direct-optimizer baseline needed to show whether a
behavior-aware latent prior adds value beyond optimizing normalized geometry
variables with the same frozen forward HONF verifier.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

try:
    from field_functional_objective import FieldFunctionalObjective
except Exception:  # pragma: no cover - package-style import fallback
    from .field_functional_objective import FieldFunctionalObjective


@dataclass
class LayoutSearchConfig:
    max_num_modules: int = 12
    domain_length_x: float = 12.0
    domain_length_y: float = 4.0
    module_radius: float = 0.45
    min_center_distance: float = 1.1
    wall_clearance: float = 0.05
    inlet_clearance: float = 0.25
    outlet_clearance: float = 0.25
    min_num_modules: int = 1
    max_active_modules: Optional[int] = None
    generate_heat_power: bool = False
    default_heat_power: float = 1.0
    cem_iterations: int = 8
    cem_population: int = 128
    cem_elite_frac: float = 0.15
    cem_init_std: float = 0.35
    cem_min_std: float = 0.03
    cem_smoothing: float = 0.5
    random_seed: int = 0


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _constraint_mapping(context: Optional[Mapping[str, Any]], constraints: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for source in (context, constraints):
        if not isinstance(source, Mapping):
            continue
        if isinstance(source.get("hard_constraints"), Mapping):
            out.update(source["hard_constraints"])  # type: ignore[index]
        elif isinstance(source.get("objective_spec"), Mapping) and isinstance(source["objective_spec"].get("hard_constraints"), Mapping):  # type: ignore[index]
            out.update(source["objective_spec"]["hard_constraints"])  # type: ignore[index]
        else:
            for key in ("num_modules", "num_modules_min", "num_modules_max", "min_center_distance", "wall_clearance", "inlet_clearance", "outlet_clearance"):
                if key in source:
                    out[key] = source[key]
    return out


def _count_range(config: LayoutSearchConfig, context: Optional[Mapping[str, Any]] = None, constraints: Optional[Mapping[str, Any]] = None) -> Tuple[int, int]:
    hard = _constraint_mapping(context, constraints)
    lo = int(hard.get("num_modules_min", config.min_num_modules))
    hi = int(hard.get("num_modules_max", config.max_active_modules or config.max_num_modules))
    if isinstance(hard.get("num_modules"), Sequence) and len(hard["num_modules"]) >= 2:
        lo = int(hard["num_modules"][0])
        hi = int(hard["num_modules"][1])
    lo = max(0, min(lo, int(config.max_num_modules)))
    hi = max(lo, min(hi, int(config.max_num_modules)))
    return lo, hi


def _bounds(config: LayoutSearchConfig, constraints: Optional[Mapping[str, Any]] = None) -> Tuple[float, float, float, float]:
    hard = _constraint_mapping(None, constraints)
    radius = max(float(config.module_radius), 0.0)
    x_min = radius + max(_finite_float(hard.get("inlet_clearance"), config.inlet_clearance), 0.0)
    x_max = float(config.domain_length_x) - radius - max(_finite_float(hard.get("outlet_clearance"), config.outlet_clearance), 0.0)
    y_min = radius + max(_finite_float(hard.get("wall_clearance"), config.wall_clearance), 0.0)
    y_max = float(config.domain_length_y) - radius - max(_finite_float(hard.get("wall_clearance"), config.wall_clearance), 0.0)
    if x_max < x_min:
        x_min, x_max = radius, max(radius, float(config.domain_length_x) - radius)
    if y_max < y_min:
        y_min, y_max = radius, max(radius, float(config.domain_length_y) - radius)
    return x_min, x_max, y_min, y_max


def _centers(layout: Mapping[str, Any]) -> np.ndarray:
    raw = layout.get("centers", layout.get("module_centers", []))
    arr = np.asarray(raw, dtype=np.float64).reshape(-1, 2) if np.asarray(raw).size else np.zeros((0, 2), dtype=np.float64)
    mask = layout.get("mask", layout.get("module_present"))
    if mask is not None and arr.size:
        keep = np.asarray(mask).reshape(-1)[: arr.shape[0]] > 0.5
        arr = arr[keep]
    count = layout.get("count")
    if count is not None:
        arr = arr[: max(int(count), 0)]
    return arr.astype(np.float64, copy=False)


def _sort_xy(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    return points[np.lexsort((points[:, 1], points[:, 0]))]


def sample_random_valid_layout(config: LayoutSearchConfig, rng: np.random.Generator, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    constraints = _constraint_mapping(context)
    lo, hi = _count_range(config, context, constraints)
    count = int(rng.integers(lo, hi + 1)) if hi >= lo else int(lo)
    x_min, x_max, y_min, y_max = _bounds(config, constraints)
    min_dist = max(_finite_float(constraints.get("min_center_distance"), config.min_center_distance), 0.0)
    centers = []
    attempts = 0
    while len(centers) < count and attempts < max(1000, 400 * max(count, 1)):
        attempts += 1
        point = np.asarray([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=np.float64)
        if all(float(np.linalg.norm(point - other)) >= min_dist for other in centers):
            centers.append(point)
    layout: Dict[str, Any] = {
        "centers": _sort_xy(np.asarray(centers, dtype=np.float32).reshape(-1, 2)),
        "count": int(len(centers)),
        "module_radius": float(config.module_radius),
        "domain": {"domain_length_x": float(config.domain_length_x), "domain_length_y": float(config.domain_length_y), "module_radius": float(config.module_radius)},
        "sampling_attempts": int(attempts),
    }
    if config.generate_heat_power:
        layout["heat_powers"] = np.full((layout["count"],), float(config.default_heat_power), dtype=np.float32)
    repaired = repair_layout(layout, config)
    return repaired


def repair_layout(layout: Mapping[str, Any], config: LayoutSearchConfig) -> Dict[str, Any]:
    constraints = _constraint_mapping(layout)
    lo, hi = _count_range(config, layout, constraints)
    rng = np.random.default_rng(int(config.random_seed))
    original = _centers(layout)
    centers = original[:hi].copy()
    x_min, x_max, y_min, y_max = _bounds(config, constraints)
    min_dist = max(_finite_float(constraints.get("min_center_distance"), config.min_center_distance), 0.0)
    if centers.size:
        centers[:, 0] = np.clip(centers[:, 0], x_min, x_max)
        centers[:, 1] = np.clip(centers[:, 1], y_min, y_max)

    def pairs(points: np.ndarray) -> list[tuple[int, int, float]]:
        out = []
        for i in range(points.shape[0]):
            for j in range(i + 1, points.shape[0]):
                dist = float(np.linalg.norm(points[i] - points[j]))
                if dist < min_dist:
                    out.append((i, j, dist))
        return out

    initial_pairs = len(pairs(centers))
    for _ in range(48):
        overlaps = pairs(centers)
        if not overlaps:
            break
        for i, j, dist in overlaps:
            delta = centers[j] - centers[i]
            norm = float(np.linalg.norm(delta))
            if norm < 1.0e-8:
                angle = float(rng.uniform(0.0, 2.0 * math.pi))
                direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
            else:
                direction = delta / norm
            step = 0.55 * (min_dist - norm + 1.0e-3)
            centers[i] -= 0.5 * step * direction
            centers[j] += 0.5 * step * direction
            centers[:, 0] = np.clip(centers[:, 0], x_min, x_max)
            centers[:, 1] = np.clip(centers[:, 1], y_min, y_max)

    kept = []
    dropped = 0
    for point in centers:
        if all(float(np.linalg.norm(point - other)) >= min_dist for other in kept):
            kept.append(point.copy())
        else:
            dropped += 1
    centers = np.asarray(kept, dtype=np.float64).reshape(-1, 2) if kept else np.zeros((0, 2), dtype=np.float64)
    attempts = 0
    while centers.shape[0] < lo and centers.shape[0] < hi and attempts < 5000:
        attempts += 1
        point = np.asarray([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=np.float64)
        if centers.shape[0] == 0 or all(float(np.linalg.norm(point - p)) >= min_dist for p in centers):
            centers = np.concatenate([centers, point[None, :]], axis=0)
    centers = _sort_xy(centers[:hi]).astype(np.float32)
    out = dict(layout)
    out["centers"] = centers
    out["count"] = int(centers.shape[0])
    out["module_radius"] = float(config.module_radius)
    out["domain"] = {"domain_length_x": float(config.domain_length_x), "domain_length_y": float(config.domain_length_y), "module_radius": float(config.module_radius)}
    if config.generate_heat_power:
        heat = np.asarray(layout.get("heat_powers", []), dtype=np.float32).reshape(-1)
        if heat.size < centers.shape[0]:
            heat = np.pad(heat, (0, centers.shape[0] - heat.size), constant_values=float(config.default_heat_power))
        out["heat_powers"] = heat[: centers.shape[0]].astype(np.float32)
    penalties = geometry_penalty(out, config, constraints)
    repair_distance = 0.0
    n = min(original.shape[0], centers.shape[0])
    if n:
        repair_distance = float(np.mean(np.linalg.norm(original[:n] - centers[:n], axis=1)))
    repair_distance += float(abs(original.shape[0] - centers.shape[0]))
    out["repair_distance"] = repair_distance
    out["validity"] = {
        "valid": bool(sum(penalties.values()) <= 1.0e-12 and centers.shape[0] >= lo and centers.shape[0] <= hi),
        "initial_overlap_pairs": int(initial_pairs),
        "dropped_count": int(dropped),
        "added_count": int(max(centers.shape[0] - max(original.shape[0] - dropped, 0), 0)),
        "geometry_penalty": penalties,
    }
    return out


def encode_layout_to_design_vec(layout: Mapping[str, Any], config: LayoutSearchConfig) -> np.ndarray:
    centers = _sort_xy(_centers(layout))[: int(config.max_num_modules)]
    m = int(config.max_num_modules)
    centers_norm = np.zeros((m, 2), dtype=np.float32)
    mask = np.zeros((m,), dtype=np.float32)
    n = min(centers.shape[0], m)
    if n:
        centers_norm[:n, 0] = np.clip(centers[:n, 0] / max(float(config.domain_length_x), 1.0e-8), 0.0, 1.0)
        centers_norm[:n, 1] = np.clip(centers[:n, 1] / max(float(config.domain_length_y), 1.0e-8), 0.0, 1.0)
        mask[:n] = 1.0
    parts = [centers_norm.reshape(-1), mask]
    if config.generate_heat_power:
        heat_norm = np.zeros((m,), dtype=np.float32)
        heat = np.asarray(layout.get("heat_powers", []), dtype=np.float32).reshape(-1)
        if heat.size and n:
            heat_norm[:n] = heat[:n] / max(float(config.default_heat_power), 1.0e-8)
        parts.append(heat_norm)
    return np.concatenate(parts, axis=0).astype(np.float32)


def decode_design_vec_to_layout(design_vec: Any, config: LayoutSearchConfig) -> Dict[str, Any]:
    m = int(config.max_num_modules)
    arr = np.asarray(design_vec, dtype=np.float32).reshape(-1)
    min_dim = 3 * m + (m if config.generate_heat_power else 0)
    if arr.size < min_dim:
        arr = np.pad(arr, (0, min_dim - arr.size))
    arr = np.clip(arr, 0.0, 1.0)
    centers_norm = arr[: 2 * m].reshape(m, 2)
    mask_raw = arr[2 * m : 3 * m]
    active = mask_raw > 0.5
    centers = np.zeros((m, 2), dtype=np.float32)
    centers[:, 0] = centers_norm[:, 0] * float(config.domain_length_x)
    centers[:, 1] = centers_norm[:, 1] * float(config.domain_length_y)
    layout: Dict[str, Any] = {"centers": centers[active], "count": int(np.sum(active)), "mask_scores": mask_raw.astype(np.float32)}
    if config.generate_heat_power:
        heat_norm = arr[3 * m : 4 * m]
        layout["heat_powers"] = (heat_norm[active] * float(config.default_heat_power)).astype(np.float32)
    return repair_layout(layout, config)


def geometry_penalty(layout: Mapping[str, Any], config: LayoutSearchConfig, constraints: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    hard = _constraint_mapping(layout, constraints)
    pts = _centers(layout)
    min_dist_required = max(_finite_float(hard.get("min_center_distance"), config.min_center_distance), 0.0)
    wall_required = max(_finite_float(hard.get("wall_clearance"), config.wall_clearance), 0.0)
    inlet_required = max(_finite_float(hard.get("inlet_clearance"), config.inlet_clearance), 0.0)
    outlet_required = max(_finite_float(hard.get("outlet_clearance"), config.outlet_clearance), 0.0)
    penalties = {"min_center_distance": 0.0, "wall_clearance": 0.0, "inlet_clearance": 0.0, "outlet_clearance": 0.0, "num_modules": 0.0}
    lo, hi = _count_range(config, layout, hard)
    penalties["num_modules"] = float(max(lo - pts.shape[0], 0, pts.shape[0] - hi))
    if pts.shape[0] == 0:
        return penalties
    pairs = []
    for i in range(pts.shape[0]):
        for j in range(i + 1, pts.shape[0]):
            pairs.append(float(np.linalg.norm(pts[i] - pts[j])))
    if pairs:
        penalties["min_center_distance"] = float(max(min_dist_required - min(pairs), 0.0))
    radius = float(config.module_radius)
    wall = float(np.min(np.minimum(pts[:, 1], float(config.domain_length_y) - pts[:, 1]) - radius))
    inlet = float(np.min(pts[:, 0] - radius))
    outlet = float(np.min(float(config.domain_length_x) - pts[:, 0] - radius))
    penalties["wall_clearance"] = float(max(wall_required - wall, 0.0))
    penalties["inlet_clearance"] = float(max(inlet_required - inlet, 0.0))
    penalties["outlet_clearance"] = float(max(outlet_required - outlet, 0.0))
    return penalties


def _forward_eval(forward_evaluator: Any, layout: Mapping[str, Any], design_vec: np.ndarray, context: Mapping[str, Any]) -> Dict[str, Any]:
    if hasattr(forward_evaluator, "evaluate_layout"):
        return dict(forward_evaluator.evaluate_layout(layout, context))
    if callable(forward_evaluator):
        return dict(forward_evaluator(layout, context))
    raise TypeError("forward_evaluator must be callable or expose evaluate_layout(layout_or_design_vec, context).")


def _objective_eval(objective: Any, forward_payload: Mapping[str, Any], layout: Mapping[str, Any]) -> Dict[str, Any]:
    obj = objective if hasattr(objective, "evaluate") else FieldFunctionalObjective(objective)
    prediction = forward_payload.get("forward_prediction", forward_payload)
    return obj.evaluate(
        forward_prediction=prediction if isinstance(prediction, Mapping) else forward_payload,
        kpis=forward_payload.get("kpis", forward_payload.get("verified_kpis")),
        layout=forward_payload.get("layout", layout),
        planned_hypergraph=forward_payload.get("planned_hypergraph"),
        realized_hypergraph=forward_payload.get("realized_hypergraph"),
        hypergraph_consistency=forward_payload.get("hypergraph_consistency"),
    )


def _candidate(
    *,
    method: str,
    layout: Mapping[str, Any],
    design_vec: np.ndarray,
    forward_payload: Mapping[str, Any],
    objective_result: Mapping[str, Any],
    forward_calls: int,
    source: str,
) -> Dict[str, Any]:
    validity = layout.get("validity", {}) if isinstance(layout.get("validity"), Mapping) else {}
    penalties = validity.get("geometry_penalty", {})
    row = {
        "method": method,
        "layout": dict(layout),
        "centers": layout.get("centers"),
        "count": int(layout.get("count", 0)),
        "num_modules": int(layout.get("count", 0)),
        "design_vec": np.asarray(design_vec, dtype=np.float32),
        "forward_prediction": dict(forward_payload),
        "objective_result": dict(objective_result),
        "total_score": float(objective_result.get("total_score", float("inf"))),
        "internal_total_score": float(objective_result.get("total_score", float("inf"))),
        "fair_objective_score": float(objective_result.get("total_score", float("inf"))),
        "objective_score": float(objective_result.get("total_score", float("inf"))),
        "ranking_score": float(objective_result.get("total_score", float("inf"))),
        "ranking_score_key": "fair_objective_score",
        "hard_violation_score": float(objective_result.get("hard_violation_score", 0.0)),
        "satisfied": bool(objective_result.get("satisfied", False)),
        "num_satisfied": int(objective_result.get("num_satisfied", 0)),
        "num_terms": int(objective_result.get("num_terms", 0)),
        "repair_distance": float(layout.get("repair_distance", 0.0)),
        "geometry_penalty": float(sum(float(v) for v in penalties.values())) if isinstance(penalties, Mapping) else 0.0,
        "hypergraph_consistency_score": float(forward_payload.get("hypergraph_consistency_score", forward_payload.get("hypergraph_consistency", {}).get("total", 0.0) if isinstance(forward_payload.get("hypergraph_consistency"), Mapping) else 0.0)),
        "planned_realized_hypergraph_score": float(forward_payload.get("planned_realized_hypergraph_score", forward_payload.get("hypergraph_consistency_score", 0.0))),
        "forward_calls": int(forward_calls),
        "diversity_cluster_id": forward_payload.get("diversity_cluster_id", ""),
        "source": source,
    }
    return row


class RawLayoutCEMOptimizer:
    def __init__(self, config: LayoutSearchConfig, forward_evaluator: Any, objective: Any):
        self.config = config
        self.forward_evaluator = forward_evaluator
        self.objective = objective

    def search(self, context: Optional[Mapping[str, Any]], *, num_return: int = 16) -> Dict[str, Any]:
        ctx: Mapping[str, Any] = context or {}
        rng = np.random.default_rng(int(self.config.random_seed))
        seed_layout = sample_random_valid_layout(self.config, rng, ctx)
        mean = encode_layout_to_design_vec(seed_layout, self.config).astype(np.float64)
        std = np.full_like(mean, max(float(self.config.cem_init_std), float(self.config.cem_min_std)), dtype=np.float64)
        all_candidates = []
        history = []
        calls = 0
        cumulative_best = float("inf")
        pop = max(int(self.config.cem_population), 2)
        elite_n = max(1, int(math.ceil(pop * float(self.config.cem_elite_frac))))
        total_iterations = max(int(self.config.cem_iterations), 1)
        progress = tqdm(range(total_iterations), desc="raw_layout_cem", unit="iter", dynamic_ncols=True)
        for iteration in progress:
            iter_start = time.time()
            iteration_rows = []
            vectors = np.clip(rng.normal(mean[None, :], std[None, :], size=(pop, mean.size)), 0.0, 1.0)
            for vec in tqdm(vectors, desc=f"raw_layout_cem iter {iteration + 1}/{total_iterations}", unit="cand", leave=False, dynamic_ncols=True):
                layout = decode_design_vec_to_layout(vec, self.config)
                design_vec = encode_layout_to_design_vec(layout, self.config)
                forward_payload = _forward_eval(self.forward_evaluator, layout, design_vec, ctx)
                calls += 1
                objective_result = _objective_eval(self.objective, forward_payload, layout)
                row = _candidate(
                    method="raw_layout_cem",
                    layout=layout,
                    design_vec=design_vec,
                    forward_payload=forward_payload,
                    objective_result=objective_result,
                    forward_calls=calls,
                    source=f"cem_iteration_{iteration}",
                )
                iteration_rows.append(row)
            iteration_rows.sort(key=lambda item: float(item["total_score"]))
            elites = iteration_rows[:elite_n]
            elite_vecs = np.stack([np.asarray(row["design_vec"], dtype=np.float64) for row in elites], axis=0)
            smoothing = min(max(float(self.config.cem_smoothing), 0.0), 1.0)
            mean = smoothing * mean + (1.0 - smoothing) * np.mean(elite_vecs, axis=0)
            std = smoothing * std + (1.0 - smoothing) * np.maximum(np.std(elite_vecs, axis=0), float(self.config.cem_min_std))
            all_candidates.extend(iteration_rows)
            round_best = float(iteration_rows[0]["total_score"])
            cumulative_best = min(cumulative_best, round_best)
            history.append(
                {
                    "iteration": int(iteration),
                    "round_best_score": float(round_best),
                    "best_score": float(cumulative_best),
                    "mean_elite_score": float(np.mean([row["total_score"] for row in elites])),
                    "num_forward_calls": int(calls),
                    "runtime_seconds": float(time.time() - iter_start),
                }
            )
            progress.set_postfix(best=f"{cumulative_best:.4g}", calls=calls)
        all_candidates.sort(key=lambda item: float(item["total_score"]))
        for rank, row in enumerate(all_candidates[: max(int(num_return), 0)], start=1):
            row["rank"] = int(rank)
        return {"method": "raw_layout_cem", "best_candidates": all_candidates[: max(int(num_return), 0)], "history": history, "num_forward_calls": int(calls)}


class RandomValidLayoutSampler:
    def __init__(self, config: LayoutSearchConfig, forward_evaluator: Any, objective: Any, *, num_samples: Optional[int] = None):
        self.config = config
        self.forward_evaluator = forward_evaluator
        self.objective = objective
        self.num_samples = num_samples

    def search(self, context: Optional[Mapping[str, Any]], *, num_return: int = 16, num_samples: Optional[int] = None) -> Dict[str, Any]:
        ctx: Mapping[str, Any] = context or {}
        rng = np.random.default_rng(int(self.config.random_seed))
        n = int(num_samples or self.num_samples or max(int(self.config.cem_population), int(num_return)))
        rows = []
        calls = 0
        for idx in tqdm(range(max(n, 0)), desc="random_valid", unit="layout", dynamic_ncols=True):
            layout = sample_random_valid_layout(self.config, rng, ctx)
            design_vec = encode_layout_to_design_vec(layout, self.config)
            forward_payload = _forward_eval(self.forward_evaluator, layout, design_vec, ctx)
            calls += 1
            objective_result = _objective_eval(self.objective, forward_payload, layout)
            rows.append(
                _candidate(
                    method="random_valid_layout",
                    layout=layout,
                    design_vec=design_vec,
                    forward_payload=forward_payload,
                    objective_result=objective_result,
                    forward_calls=calls,
                    source=f"random_{idx}",
                )
            )
        rows.sort(key=lambda item: float(item["total_score"]))
        for rank, row in enumerate(rows[: max(int(num_return), 0)], start=1):
            row["rank"] = int(rank)
        best = float(rows[0]["total_score"]) if rows else float("inf")
        history = [{"iteration": 0, "round_best_score": best, "best_score": best, "mean_elite_score": float(np.mean([r["total_score"] for r in rows[: max(1, min(len(rows), num_return))]])) if rows else float("inf"), "num_forward_calls": int(calls)}]
        return {"method": "random_valid_layout", "best_candidates": rows[: max(int(num_return), 0)], "history": history, "num_forward_calls": int(calls)}
