from __future__ import annotations

"""Utilities for planned-vs-realized ChannelThermal hypergraph diagnostics."""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle


DEFAULT_WEIGHTS: Dict[str, float] = {
    "active_count": 0.10,
    "strength": 1.0,
    "module_mass": 0.5,
    "env_mass": 0.5,
    "source": 1.0,
    "thermal_region": 1.0,
    "A_mh": 1.0,
    "unmatched_edge": 1.0,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _array(plan: Mapping[str, Any], key: str, shape_tail: Tuple[int, ...] = ()) -> np.ndarray:
    arr = np.asarray(plan.get(key, []), dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, *shape_tail), dtype=np.float64)
    if shape_tail:
        try:
            arr = arr.reshape(-1, *shape_tail)
        except ValueError:
            arr = np.zeros((0, *shape_tail), dtype=np.float64)
    else:
        arr = arr.reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def _active_indices(plan: Mapping[str, Any], active_threshold: float) -> np.ndarray:
    strength = _array(plan, "hyper_strength")
    fields = plan.get("fields") if isinstance(plan.get("fields"), Mapping) else {}
    active = _array(fields, "edge_active_or_strength") if isinstance(fields, Mapping) else np.zeros((0,), dtype=np.float64)
    n = max(strength.size, active.size, _array(plan, "source_coords", (2,)).shape[0], _array(plan, "thermal_region_coords", (2,)).shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    score = np.zeros((n,), dtype=np.float64)
    if strength.size:
        score[: min(n, strength.size)] = np.maximum(score[: min(n, strength.size)], strength[: min(n, strength.size)])
    if active.size:
        score[: min(n, active.size)] = np.maximum(score[: min(n, active.size)], active[: min(n, active.size)])
    return np.nonzero(score > float(active_threshold))[0].astype(np.int64)


def _take(arr: np.ndarray, indices: np.ndarray, fill_shape: Tuple[int, ...]) -> np.ndarray:
    out = np.zeros((indices.size, *fill_shape), dtype=np.float64)
    if arr.size == 0 or indices.size == 0:
        return out
    for row, idx in enumerate(indices):
        if 0 <= int(idx) < arr.shape[0]:
            out[row] = arr[int(idx)]
    return out


def _edge_features(plan: Mapping[str, Any], indices: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "strength": _take(_array(plan, "hyper_strength"), indices, ()),
        "module_mass": _take(_array(plan, "module_mass"), indices, ()),
        "env_mass": _take(_array(plan, "env_mass"), indices, ()),
        "source": _take(_array(plan, "source_coords", (2,)), indices, (2,)),
        "thermal_region": _take(_array(plan, "thermal_region_coords", (2,)), indices, (2,)),
    }


def _cost_matrix(planned: Dict[str, np.ndarray], realized: Dict[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    np_edges = planned["strength"].shape[0]
    nr_edges = realized["strength"].shape[0]
    if np_edges == 0 or nr_edges == 0:
        return np.zeros((np_edges, nr_edges), dtype=np.float64)
    strength = np.abs(planned["strength"][:, None] - realized["strength"][None, :])
    module_mass = np.abs(planned["module_mass"][:, None] - realized["module_mass"][None, :])
    env_mass = np.abs(planned["env_mass"][:, None] - realized["env_mass"][None, :])
    source = np.linalg.norm(planned["source"][:, None, :] - realized["source"][None, :, :], axis=-1) / math.sqrt(2.0)
    thermal = np.linalg.norm(planned["thermal_region"][:, None, :] - realized["thermal_region"][None, :, :], axis=-1) / math.sqrt(2.0)
    return (
        float(weights.get("strength", 1.0)) * strength
        + float(weights.get("module_mass", 0.5)) * module_mass
        + float(weights.get("env_mass", 0.5)) * env_mass
        + float(weights.get("source", 1.0)) * source
        + float(weights.get("thermal_region", 1.0)) * thermal
    )


def _greedy_match(cost: np.ndarray) -> List[Tuple[int, int, float]]:
    if cost.size == 0:
        return []
    remaining_rows = set(range(cost.shape[0]))
    remaining_cols = set(range(cost.shape[1]))
    matches: List[Tuple[int, int, float]] = []
    while remaining_rows and remaining_cols:
        best: Optional[Tuple[int, int, float]] = None
        for i in remaining_rows:
            for j in remaining_cols:
                c = float(cost[i, j])
                if best is None or c < best[2]:
                    best = (i, j, c)
        if best is None:
            break
        matches.append(best)
        remaining_rows.remove(best[0])
        remaining_cols.remove(best[1])
    return matches


def _match_edges(cost: np.ndarray, edge_matching: str) -> List[Tuple[int, int, float]]:
    if cost.size == 0:
        return []
    if str(edge_matching).lower().strip() == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore

            rows, cols = linear_sum_assignment(cost)
            return [(int(i), int(j), float(cost[i, j])) for i, j in zip(rows, cols)]
        except Exception:
            pass
    return _greedy_match(cost)


def _matched_a_l1(planned: Mapping[str, Any], realized: Mapping[str, Any], matches: Sequence[Tuple[int, int, float]], p_idx: np.ndarray, r_idx: np.ndarray) -> float:
    a_p = np.nan_to_num(np.asarray(planned.get("A_mh", []), dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    a_r = np.nan_to_num(np.asarray(realized.get("A_mh", []), dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    if a_p.size == 0 or a_r.size == 0 or not matches:
        return 0.0
    if a_p.ndim != 2 or a_r.ndim != 2:
        return 0.0
    vals = []
    module_dim = min(a_p.shape[0], a_r.shape[0])
    for pi, ri, _ in matches:
        ep = int(p_idx[pi]) if pi < p_idx.size else pi
        er = int(r_idx[ri]) if ri < r_idx.size else ri
        if ep < a_p.shape[1] and er < a_r.shape[1] and module_dim > 0:
            vals.append(float(np.mean(np.abs(a_p[:module_dim, ep] - a_r[:module_dim, er]))))
    return float(np.mean(vals)) if vals else 0.0


def compare_hypergraph_plans(
    planned: Mapping[str, Any],
    realized: Mapping[str, Any],
    *,
    weights: Optional[Mapping[str, float]] = None,
    edge_matching: str = "greedy",
    active_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compare decoded planned and realized hypergraph dictionaries.

    The returned score is an error-like quantity where lower is better.  It is
    deliberately finite for missing or empty plans so evaluators can keep
    ranking candidates without special-casing legacy layout-only checkpoints.
    """

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({str(k): float(v) for k, v in weights.items()})
    p_active = _active_indices(planned, active_threshold)
    r_active = _active_indices(realized, active_threshold)
    p_feat = _edge_features(planned, p_active)
    r_feat = _edge_features(realized, r_active)
    cost = _cost_matrix(p_feat, r_feat, w)
    matches = _match_edges(cost, edge_matching)

    def matched_mae(key: str) -> float:
        if not matches:
            return 0.0
        vals = [abs(float(p_feat[key][pi]) - float(r_feat[key][ri])) for pi, ri, _ in matches]
        return float(np.mean(vals)) if vals else 0.0

    def matched_rmse(key: str) -> float:
        if not matches:
            return 0.0
        vals = [float(np.mean((p_feat[key][pi] - r_feat[key][ri]) ** 2)) for pi, ri, _ in matches]
        return float(math.sqrt(max(float(np.mean(vals)), 0.0))) if vals else 0.0

    active_count_error = abs(float(len(p_active) - len(r_active)))
    a_l1 = _matched_a_l1(planned, realized, matches, p_active, r_active)
    unmatched = abs(len(p_active) - len(r_active))
    mean_match_cost = float(np.mean([m[2] for m in matches])) if matches else 0.0
    total = (
        mean_match_cost
        + float(w.get("active_count", 0.10)) * active_count_error
        + float(w.get("A_mh", 1.0)) * a_l1
        + float(w.get("unmatched_edge", 1.0)) * float(unmatched)
    )
    edge_table = []
    for pi, ri, c in matches:
        edge_table.append(
            {
                "planned_edge": int(p_active[pi]) if pi < p_active.size else int(pi),
                "realized_edge": int(r_active[ri]) if ri < r_active.size else int(ri),
                "match_cost": float(c),
                "planned_strength": float(p_feat["strength"][pi]) if p_feat["strength"].size else 0.0,
                "realized_strength": float(r_feat["strength"][ri]) if r_feat["strength"].size else 0.0,
            }
        )
    return _json_safe(
        {
            "total": float(total),
            "active_count_error": float(active_count_error),
            "strength_mae": matched_mae("strength"),
            "module_mass_mae": matched_mae("module_mass"),
            "env_mass_mae": matched_mae("env_mass"),
            "source_rmse": matched_rmse("source"),
            "thermal_region_rmse": matched_rmse("thermal_region"),
            "A_mh_l1": float(a_l1),
            "matched_edges": [
                {
                    "planned_match_index": int(pi),
                    "realized_match_index": int(ri),
                    "planned_edge": int(p_active[pi]) if pi < p_active.size else int(pi),
                    "realized_edge": int(r_active[ri]) if ri < r_active.size else int(ri),
                    "cost": float(c),
                }
                for pi, ri, c in matches
            ],
            "edge_table": edge_table,
            "edge_match_cost_matrix": cost,
            "weights": w,
            "active_threshold": float(active_threshold),
        }
    )


def summarize_hypergraph_plan(decoded: Mapping[str, Any]) -> Dict[str, Any]:
    """Return compact JSON-safe statistics for a decoded hypergraph plan."""

    active = _active_indices(decoded, 0.5)
    strength = _array(decoded, "hyper_strength")
    source = _array(decoded, "source_coords", (2,))
    thermal = _array(decoded, "thermal_region_coords", (2,))
    a_mh = np.nan_to_num(np.asarray(decoded.get("A_mh", []), dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    max_active = int(active.max()) if active.size else -1
    return _json_safe(
        {
            "active_count": int(active.size),
            "mean_strength": float(np.mean(strength[active])) if active.size and strength.size else 0.0,
            "max_strength": float(np.max(strength)) if strength.size else 0.0,
            "mean_source_x": float(np.mean(source[active, 0])) if active.size and source.shape[0] > max_active else 0.0,
            "mean_source_y": float(np.mean(source[active, 1])) if active.size and source.shape[0] > max_active else 0.0,
            "mean_thermal_region_x": float(np.mean(thermal[active, 0])) if active.size and thermal.shape[0] > max_active else 0.0,
            "mean_thermal_region_y": float(np.mean(thermal[active, 1])) if active.size and thermal.shape[0] > max_active else 0.0,
            "A_mh_sparsity": float(np.mean(np.abs(a_mh) <= 1.0e-8)) if a_mh.size else 1.0,
        }
    )


def write_edge_table_csv(edge_table: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write matched-edge diagnostics as a small CSV artifact."""

    keys = sorted({str(k) for row in edge_table for k in row.keys()}) or ["planned_edge", "realized_edge", "match_cost"]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in edge_table:
            writer.writerow({key: row.get(key, "") for key in keys})


def _coords(plan: Mapping[str, Any], key: str, domain_length_x: float, domain_length_y: float) -> np.ndarray:
    arr = _array(plan, key, (2,))
    if arr.size:
        arr = arr.copy()
        arr[:, 0] *= float(domain_length_x)
        arr[:, 1] *= float(domain_length_y)
    return arr


def plot_hypergraph_overlay(
    planned: Mapping[str, Any],
    realized: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    centers: Any,
    domain_length_x: float,
    domain_length_y: float,
    module_radius: float,
    out_path: Path,
) -> None:
    """Plot planned and realized hyperedge landmarks over the generated layout."""

    import matplotlib.pyplot as plt

    p_src = _coords(planned, "source_coords", domain_length_x, domain_length_y)
    p_thr = _coords(planned, "thermal_region_coords", domain_length_x, domain_length_y)
    r_src = _coords(realized, "source_coords", domain_length_x, domain_length_y)
    r_thr = _coords(realized, "thermal_region_coords", domain_length_x, domain_length_y)
    centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(8.8, 3.5), constrained_layout=True)
    ax.set_xlim(0.0, float(domain_length_x))
    ax.set_ylim(0.0, float(domain_length_y))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Planned vs realized hypergraph landmarks")
    for cx, cy in centers_arr:
        ax.add_patch(Circle((float(cx), float(cy)), float(module_radius), fill=False, lw=1.1, color="#6b7280", alpha=0.9))
    if p_src.size:
        ax.scatter(p_src[:, 0], p_src[:, 1], marker="o", s=55, c="#2563eb", label="planned source")
    if p_thr.size:
        ax.scatter(p_thr[:, 0], p_thr[:, 1], marker=MarkerStyle("^"), s=65, c="#2563eb", label="planned thermal")
    if r_src.size:
        ax.scatter(r_src[:, 0], r_src[:, 1], marker="x", s=60, c="#dc2626", label="realized source")
    if r_thr.size:
        ax.scatter(r_thr[:, 0], r_thr[:, 1], marker="s", s=45, facecolors="none", edgecolors="#dc2626", label="realized thermal")
    for match in comparison.get("matched_edges", []) or []:
        pi = int(match.get("planned_edge", -1))
        ri = int(match.get("realized_edge", -1))
        if 0 <= pi < p_src.shape[0] and 0 <= ri < r_src.shape[0]:
            ax.plot([p_src[pi, 0], r_src[ri, 0]], [p_src[pi, 1], r_src[ri, 1]], color="#111827", lw=0.8, alpha=0.35)
            ax.text(p_src[pi, 0], p_src[pi, 1], f"P{pi}", fontsize=8, color="#1d4ed8")
            ax.text(r_src[ri, 0], r_src[ri, 1], f"R{ri}", fontsize=8, color="#b91c1c")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(str(out_path), dpi=170)
    plt.close(fig)


def plot_hypergraph_mismatch_heatmap(comparison: Mapping[str, Any], out_path: Path) -> None:
    """Plot the edge matching cost matrix and annotate chosen matches."""

    import matplotlib.pyplot as plt

    cost = np.asarray(comparison.get("edge_match_cost_matrix", []), dtype=np.float32)
    if cost.ndim != 2:
        cost = np.zeros((0, 0), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    if cost.size:
        im = ax.imshow(cost, aspect="auto", cmap="magma")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="match cost")
    else:
        ax.imshow(np.zeros((1, 1), dtype=np.float32), cmap="Greys", vmin=0.0, vmax=1.0)
    ax.set_title("Hyperedge mismatch cost")
    ax.set_xlabel("realized active edge")
    ax.set_ylabel("planned active edge")
    for match in comparison.get("matched_edges", []) or []:
        ax.text(int(match.get("realized_match_index", 0)), int(match.get("planned_match_index", 0)), "x", ha="center", va="center", color="white", fontweight="bold")
    fig.savefig(str(out_path), dpi=170)
    plt.close(fig)
