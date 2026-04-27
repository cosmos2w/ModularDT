from __future__ import annotations

"""Shared organizer visualizations for deterministic and generative evaluators."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

try:
    import torch
except Exception:  # pragma: no cover - evaluator environments normally have torch.
    torch = None


def _to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _case_array(case: Dict, *keys: str) -> np.ndarray:
    for key in keys:
        if key in case:
            return _to_numpy(case[key])
    raise KeyError(f"Case is missing one of: {keys}")


def _grid_domain_bounds(case: Dict) -> Dict[str, float]:
    x_grid = _case_array(case, "x_grid")
    y_grid = _case_array(case, "y_grid")
    if x_grid.ndim == 3:
        x_grid = x_grid[0]
    if y_grid.ndim == 3:
        y_grid = y_grid[0]

    x_min = float(x_grid.min())
    x_max = float(x_grid.max())
    y_min = float(y_grid.min())
    y_max = float(y_grid.max())
    dx = float(np.mean(np.diff(x_grid[0]))) if x_grid.shape[1] > 1 else 0.0
    dy = float(np.mean(np.diff(y_grid[:, 0]))) if y_grid.shape[0] > 1 else 0.0
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "lx": (x_max - x_min) + dx,
        "ly": (y_max - y_min) + dy,
    }


def _coords_norm_to_physical(coords_norm: np.ndarray, case: Dict) -> np.ndarray:
    bounds = _grid_domain_bounds(case)
    coords_xy = np.asarray(coords_norm, dtype=np.float32).copy()
    coords_xy[..., 0] = bounds["x_min"] + coords_xy[..., 0] * bounds["lx"]
    coords_xy[..., 1] = bounds["y_min"] + coords_xy[..., 1] * bounds["ly"]
    return coords_xy


def _env_coords_to_physical(env_coords_norm: np.ndarray, case: Dict) -> np.ndarray:
    return _coords_norm_to_physical(env_coords_norm, case)


def _as_numpy_first(out: Dict, key: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if key not in out:
        return default
    value = _to_numpy(out[key])
    if value.ndim >= 3:
        return value[0]
    if value.ndim == 2 and key in {
        "hyper_strength",
        "hyper_wake_extent",
        "hyper_module_mass",
        "hyper_env_mass",
        "hyper_active_mask",
        "hyper_collapsed_mask",
        "hyper_duplicate_mask",
        "hyper_edge_score",
        "hyper_env_token_count",
    }:
        return value[0]
    return value


def extract_organization_arrays(out: Dict, case: Dict) -> Dict:
    centers = _case_array(case, "centers", "centers_np").astype(np.float32)
    if centers.ndim == 3:
        centers = centers[0]
    num_cyl = centers.shape[0]

    A_me = _as_numpy_first(out, "A_me")
    A_mh = _as_numpy_first(out, "A_mh")
    A_eh = _as_numpy_first(out, "A_eh")
    env_coords_norm = _as_numpy_first(out, "env_coords")
    hyper_source_norm = _as_numpy_first(out, "hyper_source_coords")
    hyper_wake_norm = _as_numpy_first(out, "hyper_wake_coords", hyper_source_norm)
    hyper_wake_axis = _as_numpy_first(out, "hyper_wake_axis")

    if A_me is None or A_mh is None or A_eh is None or env_coords_norm is None or hyper_source_norm is None:
        raise KeyError("Organization outputs are missing one of A_me, A_mh, A_eh, env_coords, hyper_source_coords.")

    A_me = np.asarray(A_me[:num_cyl], dtype=np.float32)
    A_mh = np.asarray(A_mh[:num_cyl], dtype=np.float32)
    A_eh = np.asarray(A_eh, dtype=np.float32)
    env_coords_norm = np.asarray(env_coords_norm, dtype=np.float32)
    hyper_source_norm = np.asarray(hyper_source_norm, dtype=np.float32)
    hyper_wake_norm = np.asarray(hyper_wake_norm, dtype=np.float32)
    num_hyper = A_eh.shape[1]

    if hyper_wake_axis is None:
        hyper_wake_axis = hyper_wake_norm - hyper_source_norm
    hyper_wake_axis = np.asarray(hyper_wake_axis, dtype=np.float32).reshape(num_hyper, 2)

    hyper_wake_extent = _as_numpy_first(out, "hyper_wake_extent")
    if hyper_wake_extent is None:
        hyper_wake_extent = np.full((num_hyper,), np.nan, dtype=np.float32)
    hyper_wake_extent = np.asarray(hyper_wake_extent, dtype=np.float32).reshape(-1)[:num_hyper]

    hyper_strength = _as_numpy_first(out, "hyper_strength")
    hyper_module_mass = _as_numpy_first(out, "hyper_module_mass")
    hyper_env_mass = _as_numpy_first(out, "hyper_env_mass")
    if hyper_module_mass is None or hyper_env_mass is None:
        module_mass_raw = A_mh.sum(axis=0) / max(float(num_cyl), 1.0)
        env_mass_raw = A_eh.mean(axis=0)
        hyper_module_mass = module_mass_raw / max(float(np.sum(module_mass_raw)), 1e-6)
        hyper_env_mass = env_mass_raw / max(float(np.sum(env_mass_raw)), 1e-6)
    if hyper_strength is None:
        hyper_strength = np.sqrt(np.asarray(hyper_module_mass) * np.asarray(hyper_env_mass) + 1e-6)
    hyper_strength = np.asarray(hyper_strength, dtype=np.float32).reshape(-1)[:num_hyper]
    hyper_module_mass = np.asarray(hyper_module_mass, dtype=np.float32).reshape(-1)[:num_hyper]
    hyper_env_mass = np.asarray(hyper_env_mass, dtype=np.float32).reshape(-1)[:num_hyper]
    hard_env_token_count = np.asarray([(np.argmax(A_eh, axis=1) == k).sum() for k in range(num_hyper)], dtype=np.float32)
    hyper_active_mask = _as_numpy_first(out, "hyper_active_mask")
    hyper_collapsed_mask = _as_numpy_first(out, "hyper_collapsed_mask")
    hyper_duplicate_mask = _as_numpy_first(out, "hyper_duplicate_mask")
    hyper_edge_score = _as_numpy_first(out, "hyper_edge_score")
    hyper_env_token_count = _as_numpy_first(out, "hyper_env_token_count")
    if hyper_active_mask is None:
        hyper_active_mask = np.ones((num_hyper,), dtype=np.float32)
    if hyper_collapsed_mask is None:
        hyper_collapsed_mask = np.zeros((num_hyper,), dtype=bool)
    if hyper_duplicate_mask is None:
        hyper_duplicate_mask = np.zeros((num_hyper,), dtype=bool)
    if hyper_env_token_count is None:
        hyper_env_token_count = hard_env_token_count
    if hyper_edge_score is None:
        hyper_edge_score = hyper_strength + 0.05 * (hard_env_token_count / max(float(A_eh.shape[0]), 1.0))
    hyper_active_mask = np.asarray(hyper_active_mask, dtype=np.float32).reshape(-1)[:num_hyper]
    hyper_collapsed_mask = np.asarray(hyper_collapsed_mask, dtype=bool).reshape(-1)[:num_hyper]
    hyper_duplicate_mask = np.asarray(hyper_duplicate_mask, dtype=bool).reshape(-1)[:num_hyper]
    hyper_edge_score = np.asarray(hyper_edge_score, dtype=np.float32).reshape(-1)[:num_hyper]
    hyper_env_token_count = np.asarray(hyper_env_token_count, dtype=np.float32).reshape(-1)[:num_hyper]

    env_xy = _env_coords_to_physical(env_coords_norm, case)
    hyper_source_xy = _coords_norm_to_physical(hyper_source_norm, case)
    hyper_wake_xy = _coords_norm_to_physical(hyper_wake_norm, case)
    token_group = np.argmax(A_eh, axis=1)
    token_conf = np.max(A_eh, axis=1)
    cmap_name = "tab10" if num_hyper <= 10 else "tab20"
    colors = plt.get_cmap(cmap_name)(np.arange(num_hyper) % plt.get_cmap(cmap_name).N)

    return {
        "centers": centers,
        "cylinder_radius": float(case.get("cylinder_radius", 0.5)),
        "A_me": A_me,
        "A_mh": A_mh,
        "A_eh": A_eh,
        "env_coords_norm": env_coords_norm,
        "env_xy": env_xy,
        "hyper_source_norm": hyper_source_norm,
        "hyper_wake_norm": hyper_wake_norm,
        "hyper_source_xy": hyper_source_xy,
        "hyper_wake_xy": hyper_wake_xy,
        "hyper_wake_axis": hyper_wake_axis,
        "hyper_wake_extent": hyper_wake_extent,
        "hyper_module_mass": hyper_module_mass,
        "hyper_env_mass": hyper_env_mass,
        "hyper_strength": hyper_strength,
        "hyper_active_mask": hyper_active_mask,
        "hyper_collapsed_mask": hyper_collapsed_mask,
        "hyper_duplicate_mask": hyper_duplicate_mask,
        "hyper_edge_score": hyper_edge_score,
        "hyper_env_token_count": hyper_env_token_count,
        "token_group": token_group,
        "token_conf": token_conf,
        "bounds": _grid_domain_bounds(case),
        "colors": colors,
    }


def periodic_min_image_delta_physical(p0: np.ndarray, p1: np.ndarray, bounds: Dict[str, float]) -> tuple[float, float]:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    if bounds["lx"] > 0.0:
        dx = ((dx + 0.5 * bounds["lx"]) % bounds["lx"]) - 0.5 * bounds["lx"]
    if bounds["ly"] > 0.0:
        dy = ((dy + 0.5 * bounds["ly"]) % bounds["ly"]) - 0.5 * bounds["ly"]
    return dx, dy


def periodic_shifted_endpoint(p0: np.ndarray, p1: np.ndarray, bounds: Dict[str, float]) -> np.ndarray:
    dx, dy = periodic_min_image_delta_physical(p0, p1, bounds)
    p0 = np.asarray(p0, dtype=np.float64)
    return np.asarray([p0[0] + dx, p0[1] + dy], dtype=np.float64)


def _wrap_point_to_bounds(point: np.ndarray, bounds: Dict[str, float]) -> np.ndarray:
    wrapped = np.asarray(point, dtype=np.float64).copy()
    wrapped[0] = bounds["x_min"] + ((wrapped[0] - bounds["x_min"]) % bounds["lx"])
    wrapped[1] = bounds["y_min"] + ((wrapped[1] - bounds["y_min"]) % bounds["ly"])
    return wrapped


def draw_periodic_segment(ax, p0: np.ndarray, p1: np.ndarray, bounds: Dict[str, float], **plot_kwargs) -> None:
    shifted = periodic_shifted_endpoint(p0, p1, bounds)
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    wraps = bool(np.linalg.norm(shifted - p1) > 1e-6)
    kwargs = dict(plot_kwargs)
    if wraps:
        kwargs["alpha"] = min(float(kwargs.get("alpha", 1.0)), 0.55)
    ax.plot([p0[0], shifted[0]], [p0[1], shifted[1]], **kwargs)
    if wraps:
        mid = _wrap_point_to_bounds(0.5 * (p0 + shifted), bounds)
        ax.scatter([mid[0]], [mid[1]], marker="x", s=24, c=[kwargs.get("color", "k")], linewidths=0.8, alpha=0.75, zorder=kwargs.get("zorder", 3) + 1)


def topk_cylinder_members(A_mh: np.ndarray, k: int, top_n: int = 3) -> List[Dict]:
    top_idx = np.argsort(-A_mh[:, k])[:max(0, top_n)]
    return [{"id": int(i), "weight": float(A_mh[i, k])} for i in top_idx]


def topk_env_members(A_eh: np.ndarray, k: int, env_xy: np.ndarray, top_n: int = 5) -> List[Dict]:
    top_idx = np.argsort(-A_eh[:, k])[:max(0, top_n)]
    return [
        {"id": int(j), "weight": float(A_eh[j, k]), "x": float(env_xy[j, 0]), "y": float(env_xy[j, 1])}
        for j in top_idx
    ]


def compute_hyperedge_summary(
    org: Dict,
    *,
    case_id: str,
    tau_value: float,
    topk_cylinders: int = 3,
    topk_env: int = 5,
) -> List[Dict]:
    A_mh = org["A_mh"]
    A_eh = org["A_eh"]
    summaries = []
    for k in range(A_eh.shape[1]):
        env_token_count = int(np.sum(org["token_group"] == k))
        summaries.append(
            {
                "case_id": str(case_id),
                "tau": float(tau_value),
                "hyperedge_id": int(k),
                "strength": float(org["hyper_strength"][k]),
                "module_mass": float(org["hyper_module_mass"][k]),
                "env_mass": float(org["hyper_env_mass"][k]),
                "edge_score": float(org["hyper_edge_score"][k]),
                "active": bool(org["hyper_active_mask"][k] > 0.5),
                "collapsed": bool(org["hyper_collapsed_mask"][k]),
                "duplicate": bool(org["hyper_duplicate_mask"][k]),
                "source": {"x": float(org["hyper_source_xy"][k, 0]), "y": float(org["hyper_source_xy"][k, 1])},
                "wake": {"x": float(org["hyper_wake_xy"][k, 0]), "y": float(org["hyper_wake_xy"][k, 1])},
                "wake_axis": {"x": float(org["hyper_wake_axis"][k, 0]), "y": float(org["hyper_wake_axis"][k, 1])},
                "wake_extent": float(org["hyper_wake_extent"][k]),
                "top_cylinders": topk_cylinder_members(A_mh, k, top_n=topk_cylinders),
                "env_token_count": int(round(float(org["hyper_env_token_count"][k]))) if "hyper_env_token_count" in org else env_token_count,
                "env_mass_sum": float(np.sum(A_eh[:, k])),
                "env_mass_mean": float(np.mean(A_eh[:, k])),
                "top_env_tokens": topk_env_members(A_eh, k, org["env_xy"], top_n=topk_env),
            }
        )
    return summaries


def _format_members(prefix: str, members: List[Dict], limit: Optional[int] = None) -> str:
    shown = members if limit is None else members[:limit]
    return ", ".join(f"{prefix}{m['id']}:{m['weight']:.2f}" for m in shown)


def write_organization_summary(save_csv: Path, save_json: Path, summaries: List[Dict]) -> None:
    rows = []
    for item in summaries:
        rows.append(
            {
                "case_id": item["case_id"],
                "tau": item["tau"],
                "hyperedge_id": item["hyperedge_id"],
                "strength": item["strength"],
                "module_mass": item["module_mass"],
                "env_mass": item["env_mass"],
                "edge_score": item["edge_score"],
                "active": int(bool(item["active"])),
                "collapsed": int(bool(item["collapsed"])),
                "duplicate": int(bool(item["duplicate"])),
                "source_x": item["source"]["x"],
                "source_y": item["source"]["y"],
                "wake_x": item["wake"]["x"],
                "wake_y": item["wake"]["y"],
                "axis_x": item["wake_axis"]["x"],
                "axis_y": item["wake_axis"]["y"],
                "extent": item["wake_extent"],
                "env_token_count": item["env_token_count"],
                "env_mass_sum": item["env_mass_sum"],
                "env_mass_mean": item["env_mass_mean"],
                "top_cylinders": ",".join(f"C{m['id']}" for m in item["top_cylinders"]),
                "top_cylinder_weights": ",".join(f"{m['weight']:.6g}" for m in item["top_cylinders"]),
                "top_env_tokens": ",".join(f"E{m['id']}" for m in item["top_env_tokens"]),
                "top_env_weights": ",".join(f"{m['weight']:.6g}" for m in item["top_env_tokens"]),
            }
        )
    with save_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    with save_json.open("w", encoding="utf-8") as f:
        json.dump({"hyperedges": summaries}, f, indent=2)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _padded_polygon(points: np.ndarray, pad: float) -> np.ndarray:
    center = points.mean(axis=0)
    vec = points - center
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return points + pad * vec / np.maximum(norm, 1e-6)


def _ellipse_params(points: np.ndarray, pad: float) -> tuple[np.ndarray, float, float, float]:
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    if len(pts) == 1:
        return center, 2.4 * pad, 2.4 * pad, 0.0
    centered = pts - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    primary = vh[0]
    secondary = np.asarray([-primary[1], primary[0]])
    proj0 = centered @ primary
    proj1 = centered @ secondary
    width = max(float(proj0.max() - proj0.min()) + 2.8 * pad, 2.4 * pad)
    height = max(float(proj1.max() - proj1.min()) + 2.2 * pad, 2.0 * pad)
    angle = float(np.degrees(np.arctan2(primary[1], primary[0])))
    return center, width, height, angle


def _add_region(ax, points: np.ndarray, color, pad: float, *, alpha: float, zorder: int, edge_alpha: float = 0.45) -> None:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) >= 3:
        hull = _convex_hull(pts)
        area = 0.0
        if len(hull) >= 3:
            area = 0.5 * abs(np.dot(hull[:, 0], np.roll(hull[:, 1], -1)) - np.dot(hull[:, 1], np.roll(hull[:, 0], -1)))
        if len(hull) >= 3 and area > 1e-6:
            ax.add_patch(Polygon(_padded_polygon(hull, pad), closed=True, facecolor=color, edgecolor=color, linewidth=1.2, alpha=alpha, zorder=zorder))
            ax.add_patch(Polygon(_padded_polygon(hull, pad), closed=True, facecolor="none", edgecolor=color, linewidth=0.8, alpha=edge_alpha, zorder=zorder + 1))
            return
    center, width, height, angle = _ellipse_params(pts, pad)
    ax.add_patch(Ellipse(center, width=width, height=height, angle=angle, facecolor=color, edgecolor=color, linewidth=1.0, alpha=alpha, zorder=zorder))
    ax.add_patch(Ellipse(center, width=width, height=height, angle=angle, facecolor="none", edgecolor=color, linewidth=0.8, alpha=edge_alpha, zorder=zorder + 1))


def _axis_unit_vectors(org: Dict) -> np.ndarray:
    bounds = org["bounds"]
    axis_phys = np.stack(
        [org["hyper_wake_axis"][:, 0] * bounds["lx"], org["hyper_wake_axis"][:, 1] * bounds["ly"]],
        axis=-1,
    )
    axis_norm = np.linalg.norm(axis_phys, axis=1, keepdims=True)
    return np.where(axis_norm > 1e-8, axis_phys / np.maximum(axis_norm, 1e-8), np.array([[1.0, 0.0]]))


def _edge_is_active(org: Dict, k: int) -> bool:
    return bool(np.asarray(org.get("hyper_active_mask", np.ones((k + 1,), dtype=np.float32)))[k] > 0.5)


def _edge_draw_style(org: Dict, k: int, *, show_disabled_edges: bool) -> tuple[bool, object, float, str]:
    active = _edge_is_active(org, k)
    if active:
        return True, org["colors"][k], 1.0, "-"
    if not show_disabled_edges:
        return False, "0.65", 0.0, "--"
    return True, "0.62", 0.32, "--"


def render_organization_physical_summary(
    save_path: Path,
    org: Dict,
    summaries: List[Dict],
    case: Dict,
    *,
    threshold: float = 0.15,
    topk_me_links: int = 3,
    show_table: bool = True,
    show_disabled_edges: bool = False,
    visualize_disabled_edges: bool = True,
) -> None:
    del topk_me_links
    bounds = org["bounds"]
    extent = (bounds["x_min"], bounds["x_min"] + bounds["lx"], bounds["y_min"], bounds["y_min"] + bounds["ly"])
    num_hyper = org["A_eh"].shape[1]
    fig, axes = plt.subplots(
        1,
        2 if show_table else 1,
        figsize=(17, 7.4) if show_table else (9.5, 7.4),
        dpi=150,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 1.0]} if show_table else None,
    )
    ax = axes[0] if show_table else axes
    ax.set_title("Physical organizer overlay")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    pad = 0.045 * min(bounds["lx"], bounds["ly"])

    for k in range(num_hyper):
        draw, color, alpha_scale, _ = _edge_draw_style(org, k, show_disabled_edges=show_disabled_edges)
        if visualize_disabled_edges and not draw:
            continue
        mask = org["token_group"] == k
        if np.any(mask):
            _add_region(ax, org["env_xy"][mask], color, pad, alpha=0.13 * alpha_scale, zorder=0)

    for k in range(num_hyper):
        draw, color, alpha_scale, _ = _edge_draw_style(org, k, show_disabled_edges=show_disabled_edges)
        if visualize_disabled_edges and not draw:
            continue
        mask = org["token_group"] == k
        if not np.any(mask):
            continue
        conf = np.clip(org["token_conf"][mask], 0.0, 1.0)
        ax.scatter(
            org["env_xy"][mask, 0],
            org["env_xy"][mask, 1],
            s=18 + 95 * conf,
            c=[color],
            alpha=np.clip((0.18 + 0.70 * conf) * alpha_scale, 0.10, 0.88),
            linewidths=0.0,
            zorder=2,
        )

    for item in summaries:
        k = item["hyperedge_id"]
        draw, color, alpha_scale, _ = _edge_draw_style(org, k, show_disabled_edges=show_disabled_edges)
        if visualize_disabled_edges and not draw:
            continue
        for member in item["top_env_tokens"]:
            j = member["id"]
            ax.scatter(
                org["env_xy"][j, 0],
                org["env_xy"][j, 1],
                s=92 + 70 * float(member["weight"]),
                facecolors=[color],
                edgecolors="k",
                linewidths=0.65,
                alpha=0.92 * alpha_scale,
                zorder=6,
            )

    arrow_len = 0.09 * min(bounds["lx"], bounds["ly"])
    axis_phys = _axis_unit_vectors(org)
    for k in range(num_hyper):
        draw, color, alpha_scale, linestyle = _edge_draw_style(org, k, show_disabled_edges=show_disabled_edges)
        if visualize_disabled_edges and not draw:
            continue
        draw_periodic_segment(ax, org["hyper_source_xy"][k], org["hyper_wake_xy"][k], bounds, linestyle=":" if _edge_is_active(org, k) else linestyle, color=color, alpha=0.78 * alpha_scale, linewidth=1.5, zorder=3)
        ax.scatter(org["hyper_source_xy"][k, 0], org["hyper_source_xy"][k, 1], s=105, marker="X", c=[color], edgecolors="k", linewidths=0.75, zorder=7)
        ax.scatter(org["hyper_wake_xy"][k, 0], org["hyper_wake_xy"][k, 1], s=130 + 180 * float(org["hyper_strength"][k]), marker="*", c=[color], edgecolors="k", linewidths=0.75, zorder=7)
        ax.arrow(
            org["hyper_wake_xy"][k, 0],
            org["hyper_wake_xy"][k, 1],
            arrow_len * axis_phys[k, 0],
            arrow_len * axis_phys[k, 1],
            color=color,
            alpha=max(0.15, alpha_scale),
            width=0.0,
            head_width=0.10,
            head_length=0.18,
            length_includes_head=True,
            zorder=8,
        )
        suffix = "" if _edge_is_active(org, k) else " (off)"
        ax.text(org["hyper_source_xy"][k, 0], org["hyper_source_xy"][k, 1], f"S{k}", fontsize=8, ha="right", va="top", zorder=9)
        ax.text(org["hyper_wake_xy"][k, 0], org["hyper_wake_xy"][k, 1], f"H{k}{suffix}", fontsize=9, ha="left", va="bottom", zorder=9)

    for i in range(org["centers"].shape[0]):
        for k in range(num_hyper):
            draw, color, alpha_scale, linestyle = _edge_draw_style(org, k, show_disabled_edges=show_disabled_edges)
            if visualize_disabled_edges and not draw:
                continue
            w = float(org["A_mh"][i, k])
            if w < threshold:
                continue
            draw_periodic_segment(ax, org["centers"][i], org["hyper_source_xy"][k], bounds, linestyle=linestyle if not _edge_is_active(org, k) else "--", color=color, alpha=(0.16 + 0.60 * w) * alpha_scale, linewidth=0.8 + 3.0 * w, zorder=4)

    for i, (cx, cy) in enumerate(org["centers"]):
        ax.add_patch(plt.Circle((cx, cy), org["cylinder_radius"], fill=False, color="k", lw=1.25, zorder=10))
        ax.text(cx, cy, f"C{i}", fontsize=8, ha="center", va="center", zorder=11)

    legend_items = [
        Line2D([0], [0], marker="o", color="k", markerfacecolor="white", lw=0, label="cylinder"),
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray", lw=0, label="env token colored by dominant H"),
        Line2D([0], [0], marker="X", color="k", lw=0, label="source center"),
        Line2D([0], [0], marker="*", color="k", lw=0, label="wake center"),
        Line2D([0], [0], color="k", lw=1.4, label="wake axis"),
        Line2D([0], [0], color="k", lw=1.4, linestyle="--", label="cylinder->hyperedge"),
        Line2D([0], [0], color="k", lw=1.4, linestyle=":", label="source->wake"),
    ]
    if visualize_disabled_edges:
        legend_items.append(Line2D([0], [0], color="0.62", lw=1.4, linestyle="--", label="inactive / disabled hyperedge"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=7, framealpha=0.88)
    ax.text(
        0.01,
        0.01,
        "Periodic shortest-image links are used.\nEnv-token color = dominant hyperedge; size/opacity = confidence.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    if show_table:
        table_ax = axes[1]
        table_ax.axis("off")
        table_ax.set_title("Hyperedge summary")
        row_h = min(0.145, 0.92 / max(len(summaries), 1))
        y = 0.98
        for item in summaries:
            k = item["hyperedge_id"]
            draw, color, alpha_scale, _ = _edge_draw_style(org, k, show_disabled_edges=True)
            y -= row_h
            table_ax.add_patch(Rectangle((0.0, y), 1.0, row_h * 0.86, transform=table_ax.transAxes, facecolor=color, edgecolor=color, alpha=0.16 * alpha_scale, linewidth=1.0))
            status = "on" if item["active"] else "off"
            text = (
                f"H{k} {status} score={item['edge_score']:.3f} strength={item['strength']:.3f} mod={item['module_mass']:.3f} env={item['env_mass']:.3f} n={item['env_token_count']}\n"
                f"collapsed={int(item['collapsed'])} duplicate={int(item['duplicate'])} "
                f"src=({item['source']['x']:.2f},{item['source']['y']:.2f}) wake=({item['wake']['x']:.2f},{item['wake']['y']:.2f}) "
                f"axis=({item['wake_axis']['x']:.2f},{item['wake_axis']['y']:.2f}) extent={item['wake_extent']:.3f}\n"
                f"cyl: {_format_members('C', item['top_cylinders'])}\n"
                f"env: {_format_members('E', item['top_env_tokens'], limit=4)}"
            )
            table_ax.text(0.02, y + row_h * 0.78, text, ha="left", va="top", fontsize=7.6, family="monospace", transform=table_ax.transAxes)

    case_id = case.get("case_id", "?")
    fig.suptitle(f"Case {case_id} | tau organization diagnostics")
    fig.savefig(save_path)
    plt.close(fig)


def _sorted_env_order(org: Dict) -> tuple[np.ndarray, np.ndarray]:
    groups = org["token_group"]
    env_xy = org["env_xy"]
    order = np.lexsort((env_xy[:, 1], env_xy[:, 0], groups))
    sorted_groups = groups[order]
    return order, sorted_groups


def render_organization_matrices(
    save_path: Path,
    org: Dict,
    summaries: List[Dict],
    *,
    show_disabled_edges: bool = False,
    visualize_disabled_edges: bool = True,
) -> None:
    A_mh = org["A_mh"]
    A_eh = org["A_eh"]
    num_hyper = A_eh.shape[1]
    order, sorted_groups = _sorted_env_order(org)
    A_eh_sorted = A_eh[order]
    cols = max(2, int(np.ceil(np.sqrt(num_hyper))))
    rows = int(np.ceil(num_hyper / cols))
    fig = plt.figure(figsize=(max(13.5, cols * 3.1), 7.6 + rows * 2.8), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(2 + rows, cols)
    ax_mh = fig.add_subplot(gs[0, : max(1, cols // 2)])
    eh_gs = gs[0, max(1, cols // 2) :].subgridspec(1, 2, width_ratios=[0.07, 1.0], wspace=0.02)
    ax_strip = fig.add_subplot(eh_gs[0, 0])
    ax_eh = fig.add_subplot(eh_gs[0, 1])

    im_mh = ax_mh.imshow(A_mh, aspect="auto", vmin=0.0, vmax=max(1.0, float(A_mh.max())), cmap="viridis")
    ax_mh.set_title("Module -> Hyperedge assignment A_mh")
    ax_mh.set_xlabel("hyperedge")
    ax_mh.set_ylabel("cylinder")
    ax_mh.set_xticks(np.arange(num_hyper), labels=[f"H{k}" for k in range(num_hyper)])
    ax_mh.set_yticks(np.arange(A_mh.shape[0]), labels=[f"C{i}" for i in range(A_mh.shape[0])])
    if A_mh.size <= 80:
        for i in range(A_mh.shape[0]):
            for k in range(num_hyper):
                ax_mh.text(k, i, f"{A_mh[i, k]:.2f}", ha="center", va="center", fontsize=7, color="white" if A_mh[i, k] > 0.5 else "black")
    if visualize_disabled_edges:
        for k in range(num_hyper):
            if not _edge_is_active(org, k):
                ax_mh.axvspan(k - 0.5, k + 0.5, color="0.75", alpha=0.35, hatch="//")
    fig.colorbar(im_mh, ax=ax_mh, fraction=0.046, pad=0.03)

    group_cmap = ListedColormap(org["colors"])
    ax_strip.imshow(sorted_groups[:, None], aspect="auto", cmap=group_cmap, vmin=-0.5, vmax=num_hyper - 0.5)
    ax_strip.set_title("group", fontsize=8)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])

    im_eh = ax_eh.imshow(A_eh_sorted, aspect="auto", vmin=0.0, vmax=max(1.0, float(A_eh.max())), cmap="viridis")
    ax_eh.set_title("Environment -> Hyperedge assignment A_eh (group-sorted)")
    ax_eh.set_xlabel("hyperedge")
    ax_eh.set_ylabel("env token row")
    ax_eh.set_xticks(np.arange(num_hyper), labels=[f"H{k}" for k in range(num_hyper)])
    if A_eh.shape[0] <= 32:
        ax_eh.set_yticks(np.arange(A_eh.shape[0]), labels=[f"E{int(i)}" for i in order])
    else:
        tick_idx = np.linspace(0, A_eh.shape[0] - 1, min(9, A_eh.shape[0])).astype(int)
        ax_eh.set_yticks(tick_idx, labels=[f"E{int(order[i])}" for i in tick_idx])
    boundaries = np.where(sorted_groups[1:] != sorted_groups[:-1])[0] + 0.5
    for y in boundaries:
        ax_eh.axhline(y, color="white", lw=1.1, alpha=0.95)
        ax_strip.axhline(y, color="white", lw=1.1, alpha=0.95)
    if visualize_disabled_edges:
        for k in range(num_hyper):
            if not _edge_is_active(org, k):
                ax_eh.axvspan(k - 0.5, k + 0.5, color="0.75", alpha=0.35, hatch="//")
    fig.colorbar(im_eh, ax=ax_eh, fraction=0.046, pad=0.03)

    bounds = org["bounds"]
    extent = (bounds["x_min"], bounds["x_min"] + bounds["lx"], bounds["y_min"], bounds["y_min"] + bounds["ly"])
    for k in range(num_hyper):
        ax = fig.add_subplot(gs[1 + (k // cols), k % cols])
        draw, color, alpha_scale, _ = _edge_draw_style(org, k, show_disabled_edges=show_disabled_edges)
        if visualize_disabled_edges and not draw:
            ax.axis("off")
            ax.set_title(f"H{k} (off)", fontsize=8.2, color="0.45")
            continue
        weights = np.clip(A_eh[:, k], 0.0, None)
        dominated = org["token_group"] == k
        ax.scatter(org["env_xy"][:, 0], org["env_xy"][:, 1], c="0.78", s=14, alpha=0.45, linewidths=0.0)
        if np.any(dominated):
            ax.scatter(
                org["env_xy"][dominated, 0],
                org["env_xy"][dominated, 1],
                c=[color],
                s=18 + 85 * weights[dominated],
                alpha=np.clip((0.28 + 0.68 * weights[dominated]) * alpha_scale, 0.12, 0.92),
                linewidths=0.0,
            )
        for member in summaries[k]["top_env_tokens"]:
            j = member["id"]
            ax.scatter(org["env_xy"][j, 0], org["env_xy"][j, 1], s=88 + 65 * float(member["weight"]), facecolors=[color], edgecolors="k", linewidths=0.6, alpha=alpha_scale)
        for cx, cy in org["centers"]:
            ax.add_patch(plt.Circle((cx, cy), org["cylinder_radius"], fill=False, color="k", lw=0.8))
        ax.scatter(org["hyper_source_xy"][k, 0], org["hyper_source_xy"][k, 1], marker="X", s=70, c=[color], edgecolors="k", alpha=alpha_scale)
        ax.scatter(org["hyper_wake_xy"][k, 0], org["hyper_wake_xy"][k, 1], marker="*", s=100, c=[color], edgecolors="k", alpha=alpha_scale)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_title(
            f"H{k}{'' if summaries[k]['active'] else ' (off)'} | score={summaries[k]['edge_score']:.2f} | mod={summaries[k]['module_mass']:.2f} | env={summaries[k]['env_mass']:.2f}",
            fontsize=8.2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Organization matrix diagnostics")
    fig.savefig(save_path)
    plt.close(fig)


def spread_positions(desired_y: np.ndarray, min_gap: float = 0.08, y_min: float = 0.08, y_max: float = 0.92) -> np.ndarray:
    desired_y = np.asarray(desired_y, dtype=np.float64)
    n = desired_y.size
    if n == 0:
        return desired_y
    if n == 1:
        return np.asarray([float(np.clip(desired_y[0], y_min, y_max))])
    order = np.argsort(desired_y)
    sorted_y = np.clip(desired_y[order], y_min, y_max)
    span = y_max - y_min
    gap = min(float(min_gap), span / max(n - 1, 1))
    for i in range(1, n):
        sorted_y[i] = max(sorted_y[i], sorted_y[i - 1] + gap)
    overflow = sorted_y[-1] - y_max
    if overflow > 0:
        sorted_y -= overflow
    for i in range(n - 2, -1, -1):
        sorted_y[i] = min(sorted_y[i], sorted_y[i + 1] - gap)
    underflow = y_min - sorted_y[0]
    if underflow > 0:
        sorted_y += underflow
    sorted_y = np.clip(sorted_y, y_min, y_max)
    out = np.empty_like(sorted_y)
    out[order] = sorted_y
    return out


def _curved_edge(ax, p0: tuple[float, float], p1: tuple[float, float], *, color, linewidth: float, alpha: float, rad: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-",
            connectionstyle=f"arc3,rad={rad}",
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            shrinkA=8,
            shrinkB=8,
            zorder=1,
        )
    )


def render_organization_sankey(
    save_path: Path,
    org: Dict,
    summaries: List[Dict],
    *,
    threshold: float = 0.15,
    min_gap: float = 0.08,
    show_disabled_edges: bool = False,
    visualize_disabled_edges: bool = True,
) -> None:
    centers = org["centers"]
    A_mh = org["A_mh"]
    A_eh = org["A_eh"]
    bounds = org["bounds"]
    num_cyl = centers.shape[0]
    num_hyper = A_eh.shape[1]
    fig, ax = plt.subplots(figsize=(13, 7), dpi=150, constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Sankey-style organizer topology\nC_i = cylinder/module, H_k = interaction hyperedge, EnvGroup_k = dominated environment tokens")

    cyl_desired = np.clip((centers[:, 1] - bounds["y_min"]) / max(bounds["ly"], 1e-6), 0.08, 0.92)
    hyper_desired = np.clip((org["hyper_wake_xy"][:, 1] - bounds["y_min"]) / max(bounds["ly"], 1e-6), 0.08, 0.92)
    env_desired = np.clip(hyper_desired + 0.035 * np.where(np.arange(num_hyper) % 2 == 0, 1.0, -1.0), 0.08, 0.92)
    cyl_y = spread_positions(cyl_desired, min_gap=min_gap * 0.75)
    hyper_y = spread_positions(hyper_desired, min_gap=min_gap)
    env_y = spread_positions(env_desired, min_gap=min_gap)
    module_pos = {i: (0.10, float(cyl_y[i])) for i in range(num_cyl)}
    hyper_pos = {k: (0.50, float(hyper_y[k])) for k in range(num_hyper)}
    env_pos = {k: (0.88, float(env_y[k])) for k in range(num_hyper)}

    for i in range(num_cyl):
        for k in range(num_hyper):
            draw, color, alpha_scale, _ = _edge_draw_style(org, k, show_disabled_edges=show_disabled_edges)
            if visualize_disabled_edges and not draw:
                continue
            w = float(A_mh[i, k])
            if w < threshold:
                continue
            _curved_edge(ax, module_pos[i], hyper_pos[k], color=color, linewidth=0.7 + 4.2 * w, alpha=min(0.95, 0.15 + 0.85 * w) * alpha_scale, rad=0.10 if (i + k) % 2 == 0 else -0.10)
    for k in range(num_hyper):
        draw, color, alpha_scale, _ = _edge_draw_style(org, k, show_disabled_edges=show_disabled_edges)
        if visualize_disabled_edges and not draw:
            continue
        mass = float(A_eh[:, k].mean())
        n_tokens = summaries[k]["env_token_count"]
        width = 1.0 + 8.0 * max(mass, n_tokens / max(float(A_eh.shape[0]), 1.0))
        _curved_edge(ax, hyper_pos[k], env_pos[k], color=color, linewidth=width, alpha=(0.35 + 0.5 * min(1.0, width / 8.0)) * alpha_scale, rad=-0.08)

    bbox = {"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.0}
    for i in range(num_cyl):
        x, y = module_pos[i]
        ax.scatter([x], [y], s=130, c="white", edgecolors="k", zorder=4)
        ax.text(x - 0.035, y, f"C{i}", ha="right", va="center", fontsize=9, bbox=bbox, zorder=5)
    for k in range(num_hyper):
        draw, color, alpha_scale, _ = _edge_draw_style(org, k, show_disabled_edges=show_disabled_edges)
        if visualize_disabled_edges and not draw:
            continue
        x, y = hyper_pos[k]
        ax.scatter([x], [y], s=180, marker="*", c=[color], edgecolors="k", alpha=alpha_scale, zorder=4)
        ax.text(x, y + 0.035, f"H{k}{'' if summaries[k]['active'] else ' (off)'}", ha="center", va="bottom", fontsize=9, bbox=bbox, zorder=5)
    for k in range(num_hyper):
        draw, color, alpha_scale, _ = _edge_draw_style(org, k, show_disabled_edges=show_disabled_edges)
        if visualize_disabled_edges and not draw:
            continue
        x, y = env_pos[k]
        ax.scatter([x], [y], s=145, marker="s", c=[color], edgecolors="k", alpha=alpha_scale, zorder=4)
        ax.text(
            x + 0.035,
            y,
            f"EnvGroup_{k}\nn={summaries[k]['env_token_count']}\nmod={summaries[k]['module_mass']:.2f} env={summaries[k]['env_mass']:.2f}",
            ha="left",
            va="center",
            fontsize=8,
            bbox=bbox,
            zorder=5,
        )

    ax.text(0.03, 0.965, "Line width is proportional to soft assignment weight.", ha="left", va="top", fontsize=9)
    ax.text(0.10, 0.99, "Modules", ha="center", va="top", fontsize=10, weight="bold")
    ax.text(0.50, 0.99, "Interaction hyperedges", ha="center", va="top", fontsize=10, weight="bold")
    ax.text(0.88, 0.99, "Environment groups", ha="center", va="top", fontsize=10, weight="bold")
    fig.savefig(save_path)
    plt.close(fig)


def _schematic_node_positions(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    if n == 0:
        return centers
    span = np.ptp(centers, axis=0)
    if np.all(span > 1e-6):
        x = 0.16 + 0.68 * (centers[:, 0] - centers[:, 0].min()) / span[0]
        y = 0.20 + 0.60 * (centers[:, 1] - centers[:, 1].min()) / span[1]
        pos = np.stack([x, y], axis=1)
    else:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        pos = np.stack([0.5 + 0.28 * np.cos(angles), 0.5 + 0.28 * np.sin(angles)], axis=1)
    if n > 1:
        pos[:, 1] = spread_positions(pos[:, 1], min_gap=0.055, y_min=0.17, y_max=0.83)
    return pos


def render_organization_hypergraph_schematic(
    save_path: Path,
    org: Dict,
    summaries: List[Dict],
    *,
    threshold: float = 0.15,
    show_disabled_edges: bool = False,
    visualize_disabled_edges: bool = True,
) -> None:
    A_mh = org["A_mh"]
    num_cyl, num_hyper = A_mh.shape
    node_pos = _schematic_node_positions(org["centers"])
    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=150, constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Conceptual hypergraph organizer schematic")

    label_items = []
    for k in range(num_hyper):
        draw, color, alpha_scale, _ = _edge_draw_style(org, k, show_disabled_edges=show_disabled_edges)
        if visualize_disabled_edges and not draw:
            continue
        weights = A_mh[:, k]
        members = np.where(weights >= threshold)[0]
        if members.size == 0 and num_cyl:
            members = np.argsort(-weights)[: min(2, num_cyl)]
        elif members.size == 1 and num_cyl > 1:
            runner_up = int(np.argsort(-weights)[1])
            if weights[runner_up] >= 0.5 * max(float(weights[members[0]]), 1e-6):
                members = np.asarray([members[0], runner_up])
        if members.size == 0:
            continue
        pts = node_pos[members]
        jitter = np.array([0.018 * ((k % 3) - 1), 0.018 * (((k + 1) % 3) - 1)])
        region_pts = np.clip(pts + jitter, 0.05, 0.95)
        _add_region(ax, region_pts, color, pad=0.065 + 0.01 * (k % 2), alpha=0.20 * alpha_scale, zorder=1 + k)
        label_xy = np.clip(region_pts.mean(axis=0) + np.array([0.0, 0.095 + 0.015 * ((k % 2) * 2 - 1)]), 0.08, 0.92)
        label_items.append((k, label_xy, color, alpha_scale))

    if label_items:
        raw_y = np.asarray([item[1][1] for item in label_items], dtype=np.float64)
        spread_y = spread_positions(raw_y, min_gap=0.085, y_min=0.10, y_max=0.90)
    else:
        spread_y = np.asarray([], dtype=np.float64)

    for idx, (k, label_xy, color, alpha_scale) in enumerate(label_items):
        label_x = float(np.clip(label_xy[0] + 0.04 * ((idx % 3) - 1), 0.10, 0.90))
        label_y = float(spread_y[idx])
        ax.text(
            label_x,
            label_y,
            f"H{k}{'' if summaries[k]['active'] else ' off'}\nmod={summaries[k]['module_mass']:.2f}\nenv={summaries[k]['env_mass']:.2f}",
            ha="center",
            va="center",
            fontsize=8.3,
            color="black",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": color, "alpha": 0.88 * alpha_scale},
            zorder=20 + k,
        )

    for i, (x, y) in enumerate(node_pos):
        ax.scatter([x], [y], s=310, c="white", edgecolors="black", linewidths=1.25, zorder=50)
        ax.text(x, y, f"C{i}", ha="center", va="center", fontsize=10, weight="bold", zorder=51)

    ax.text(0.02, 0.03, "Colored regions enclose cylinders with strong A_mh membership; fallback uses top cylinders.", ha="left", va="bottom", fontsize=8.2)
    fig.savefig(save_path)
    plt.close(fig)


def render_soft_organization(
    output_dir: Path,
    out: Dict,
    case: Dict,
    *,
    tau_value: float,
    phase_idx: int,
    threshold: float = 0.15,
    topk_me_links: int = 3,
    organization_view: str = "all",
    topk_cylinders: int = 3,
    topk_env: int = 5,
    min_gap: float = 0.08,
    show_table: bool = True,
    show_disabled_edges: bool = False,
    visualize_disabled_edges: bool = True,
) -> Dict[str, str]:
    org = extract_organization_arrays(out, case)
    case_id = str(case.get("case_id", "?"))
    summaries = compute_hyperedge_summary(
        org,
        case_id=case_id,
        tau_value=tau_value,
        topk_cylinders=topk_cylinders,
        topk_env=topk_env,
    )
    render_org = org
    if not visualize_disabled_edges:
        render_org = dict(org)
        render_org["hyper_active_mask"] = np.ones_like(org["hyper_active_mask"], dtype=np.float32)
    base = f"case_{case_id}_tau_{phase_idx:03d}"
    csv_path = output_dir / f"organization_summary_{base}.csv"
    json_path = output_dir / f"organization_summary_{base}.json"
    write_organization_summary(csv_path, json_path, summaries)

    paths = {"summary_csv": str(csv_path), "summary_json": str(json_path)}
    if organization_view in {"all", "physical"}:
        path = output_dir / f"organization_physical_{base}.png"
        render_organization_physical_summary(
            path,
            render_org,
            summaries,
            case,
            threshold=threshold,
            topk_me_links=topk_me_links,
            show_table=show_table,
            show_disabled_edges=show_disabled_edges,
            visualize_disabled_edges=visualize_disabled_edges,
        )
        paths["physical"] = str(path)
    if organization_view in {"all", "matrices"}:
        path = output_dir / f"organization_matrices_{base}.png"
        render_organization_matrices(path, render_org, summaries, show_disabled_edges=show_disabled_edges, visualize_disabled_edges=visualize_disabled_edges)
        paths["matrices"] = str(path)
    if organization_view in {"all", "sankey"}:
        path = output_dir / f"organization_sankey_{base}.png"
        render_organization_sankey(path, render_org, summaries, threshold=threshold, min_gap=min_gap, show_disabled_edges=show_disabled_edges, visualize_disabled_edges=visualize_disabled_edges)
        paths["sankey"] = str(path)
    if organization_view in {"all", "schematic"}:
        path = output_dir / f"organization_hypergraph_schematic_{base}.png"
        render_organization_hypergraph_schematic(path, render_org, summaries, threshold=threshold, show_disabled_edges=show_disabled_edges, visualize_disabled_edges=visualize_disabled_edges)
        paths["schematic"] = str(path)
    return paths
