from __future__ import annotations

"""Presentation-oriented organizer visualizations for Demo 1 Channel Thermal."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon


def _hyperedge_colors(num_h: int) -> List[Tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20", max(int(num_h), 1))
    return [cmap(idx) for idx in range(max(int(num_h), 1))]


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)})
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float32)

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def _domain_extent(sample: Dict[str, Any], env_coords: np.ndarray) -> Tuple[float, float, float, float]:
    if "x_grid" in sample and "y_grid" in sample:
        return (
            float(np.nanmin(sample["x_grid"])),
            float(np.nanmax(sample["x_grid"])),
            float(np.nanmin(sample["y_grid"])),
            float(np.nanmax(sample["y_grid"])),
        )
    if env_coords.size:
        pad = 0.05 * max(float(np.ptp(env_coords[:, 0])), float(np.ptp(env_coords[:, 1])), 1.0)
        return (
            float(np.nanmin(env_coords[:, 0]) - pad),
            float(np.nanmax(env_coords[:, 0]) + pad),
            float(np.nanmin(env_coords[:, 1]) - pad),
            float(np.nanmax(env_coords[:, 1]) + pad),
        )
    return 0.0, 1.0, 0.0, 1.0


def _temperature_image(sample: Dict[str, Any], channel_order: Optional[Sequence[str]]) -> Optional[np.ndarray]:
    field = sample.get("steady_field")
    if field is None:
        return None
    arr = np.asarray(field)
    if arr.ndim < 3 or arr.shape[-1] == 0:
        return None
    if channel_order and "temperature" in channel_order:
        idx = list(channel_order).index("temperature")
    else:
        idx = min(arr.shape[-1] - 1, 4)
    return np.asarray(arr[..., idx], dtype=np.float32)


def _dominant_env(A_eh: np.ndarray, env_count: int) -> Tuple[np.ndarray, np.ndarray]:
    if A_eh.size:
        return A_eh.argmax(axis=-1), A_eh.max(axis=-1)
    return np.zeros((env_count,), dtype=np.int64), np.ones((env_count,), dtype=np.float32)


def _heat_scale(heat: np.ndarray) -> np.ndarray:
    heat_abs = np.abs(np.asarray(heat, dtype=np.float32))
    denom = max(float(np.nanmax(heat_abs)) if heat_abs.size else 0.0, 1.0e-6)
    return heat_abs / denom


def _top_modules(A_mh: np.ndarray, present: np.ndarray, hidx: int, limit: int = 3) -> Tuple[str, str]:
    if not A_mh.size or hidx >= A_mh.shape[1]:
        return "", ""
    valid = [(idx, float(A_mh[idx, hidx])) for idx in np.flatnonzero(present) if idx < A_mh.shape[0]]
    valid.sort(key=lambda item: item[1], reverse=True)
    top = [(idx, value) for idx, value in valid[:limit] if value > 1.0e-6]
    return ", ".join(f"M{idx}" for idx, _ in top), ", ".join(f"{value:.2f}" for _, value in top)


def _summary_rows(arrays: Dict[str, np.ndarray]) -> List[List[str]]:
    A_mh = arrays["A_mh"]
    A_eh = arrays["A_eh"]
    present = arrays["present"]
    strength = arrays["strength"]
    module_mass = arrays["module_mass"]
    env_mass = arrays["env_mass"]
    dominant, _ = _dominant_env(A_eh, arrays["env_coords"].shape[0])
    rows: List[List[str]] = []
    for hidx in range(strength.shape[0]):
        top, _ = _top_modules(A_mh, present, hidx)
        rows.append(
            [
                f"H{hidx}",
                f"{float(strength[hidx]):.2f}",
                f"{float(module_mass[hidx]):.2f}",
                f"{float(env_mass[hidx]):.2f}",
                top or "-",
                str(int(np.sum(dominant == hidx))),
            ]
        )
    return rows


def _draw_module_circles(
    ax: Any,
    arrays: Dict[str, np.ndarray],
    module_radius: float,
    *,
    label: bool = True,
) -> None:
    centers = arrays["centers"]
    present = arrays["present"]
    heat = arrays["heat"]
    scale = _heat_scale(heat)
    for module_idx in np.flatnonzero(present):
        cx, cy = centers[module_idx]
        hot = float(heat[module_idx]) >= 0.0 if module_idx < heat.shape[0] else True
        color = "#d95f02" if hot else "#1f78b4"
        lw = 1.2 + 2.5 * float(scale[module_idx]) if module_idx < scale.shape[0] else 1.2
        alpha = 0.16 + 0.28 * float(scale[module_idx]) if module_idx < scale.shape[0] else 0.16
        ax.add_patch(plt.Circle((float(cx), float(cy)), module_radius, facecolor=color, edgecolor=color, alpha=alpha, lw=0.0, zorder=3))
        ax.add_patch(plt.Circle((float(cx), float(cy)), module_radius, fill=False, edgecolor=color, lw=lw, alpha=0.95, zorder=5))
        if label:
            ax.text(
                float(cx),
                float(cy),
                f"M{module_idx}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                weight="bold",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                zorder=6,
            )


def _draw_region_hulls(ax: Any, arrays: Dict[str, np.ndarray], colors: Sequence[Tuple[float, float, float, float]]) -> None:
    env_coords = arrays["env_coords"]
    A_eh = arrays["A_eh"]
    strength = arrays["strength"]
    if env_coords.size == 0:
        return
    dominant, _ = _dominant_env(A_eh, env_coords.shape[0])
    for hidx in range(strength.shape[0]):
        pts = env_coords[dominant == hidx]
        if pts.shape[0] >= 3:
            hull = _convex_hull(pts)
            if hull.shape[0] >= 3:
                ax.add_patch(Polygon(hull, closed=True, facecolor=colors[hidx], edgecolor=colors[hidx], lw=1.2, alpha=0.12, zorder=1))
                continue
        if 1 <= pts.shape[0] < 3:
            center = np.mean(pts, axis=0)
            ax.add_patch(Ellipse((float(center[0]), float(center[1])), 0.36, 0.18, facecolor=colors[hidx], edgecolor=colors[hidx], lw=1.0, alpha=0.14, zorder=1))


def _draw_env_tokens(ax: Any, arrays: Dict[str, np.ndarray], colors: Sequence[Tuple[float, float, float, float]]) -> None:
    env_coords = arrays["env_coords"]
    if env_coords.size == 0:
        return
    dominant, confidence = _dominant_env(arrays["A_eh"], env_coords.shape[0])
    facecolors = []
    for hidx, conf in zip(dominant, confidence):
        rgba = list(colors[int(hidx) % len(colors)])
        rgba[3] = float(np.clip(0.20 + 0.75 * conf, 0.20, 0.95))
        facecolors.append(tuple(rgba))
    sizes = 18.0 + 70.0 * np.clip(confidence, 0.0, 1.0)
    ax.scatter(env_coords[:, 0], env_coords[:, 1], s=sizes, c=facecolors, edgecolor="white", linewidth=0.35, zorder=4)


def _draw_sources_regions_links(
    ax: Any,
    arrays: Dict[str, np.ndarray],
    colors: Sequence[Tuple[float, float, float, float]],
    link_threshold: float,
) -> None:
    centers = arrays["centers"]
    present = arrays["present"]
    A_mh = arrays["A_mh"]
    strength = arrays["strength"]
    src = arrays["src"]
    dst = arrays["dst"]
    for hidx in range(strength.shape[0]):
        color = colors[hidx]
        alpha = float(np.clip(0.28 + 0.65 * strength[hidx], 0.28, 0.95))
        ax.annotate(
            "",
            xy=(float(dst[hidx, 0]), float(dst[hidx, 1])),
            xytext=(float(src[hidx, 0]), float(src[hidx, 1])),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2 + 2.2 * float(strength[hidx]), "alpha": alpha, "shrinkA": 4, "shrinkB": 5},
            zorder=7,
        )
        ax.scatter(src[hidx, 0], src[hidx, 1], marker="x", s=60, color=color, linewidth=2.0, zorder=8)
        ax.scatter(dst[hidx, 0], dst[hidx, 1], marker="*", s=140, color=color, edgecolor="black", linewidth=0.55, zorder=8)
        label_x = float(dst[hidx, 0]) + 0.06
        label_y = float(dst[hidx, 1]) + 0.06 * (1 if hidx % 2 == 0 else -1)
        ax.text(
            label_x,
            label_y,
            f"H{hidx}",
            fontsize=8,
            color="black",
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": color, "alpha": 0.82},
            zorder=9,
        )
        if A_mh.size:
            for module_idx in np.flatnonzero(present):
                if module_idx >= A_mh.shape[0]:
                    continue
                weight = float(A_mh[module_idx, hidx])
                if weight < float(link_threshold):
                    continue
                ax.plot(
                    [centers[module_idx, 0], src[hidx, 0]],
                    [centers[module_idx, 1], src[hidx, 1]],
                    color=color,
                    lw=0.6 + 3.0 * weight,
                    alpha=0.18 + 0.55 * min(weight, 1.0),
                    zorder=2,
                )


def render_channelthermal_organization_overview(
    output_path: Path,
    sample: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    *,
    module_radius: float,
    channel_order: Optional[Sequence[str]] = None,
    link_threshold: float = 0.25,
) -> None:
    """Render a presentation overview: physical overlay plus hyperedge table."""
    env_coords = arrays["env_coords"]
    strength = arrays["strength"]
    colors = _hyperedge_colors(strength.shape[0])
    extent = _domain_extent(sample, env_coords)
    fig = plt.figure(figsize=(14.2, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.05])
    ax = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])

    temp = _temperature_image(sample, channel_order)
    if temp is not None:
        ax.imshow(temp, origin="lower", extent=extent, cmap="inferno", alpha=0.26, aspect="auto", zorder=0)
    _draw_region_hulls(ax, arrays, colors)
    _draw_sources_regions_links(ax, arrays, colors, link_threshold)
    _draw_env_tokens(ax, arrays, colors)
    _draw_module_circles(ax, arrays, module_radius, label=True)

    ax.set_title("Organizer physical overview")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    legend_items = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#777777", markeredgecolor="white", markersize=7, label="env token color = dominant H"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#777777", markeredgecolor="white", alpha=0.45, markersize=5, label="opacity/size = confidence"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=7, label="source center"),
        Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=10, label="thermal region center"),
        Line2D([0], [0], color="black", lw=2.4, label=f"module link if A_mh >= {link_threshold:.2f}"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.88)

    ax_table.axis("off")
    col_labels = ["H", "S", "M", "E", "top modules", "env n"]
    rows = _summary_rows(arrays)
    table = ax_table.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center", colWidths=[0.10, 0.12, 0.12, 0.12, 0.34, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#f1f1f1")
            cell.set_text_props(weight="bold")
        else:
            rgba = list(colors[(row - 1) % len(colors)])
            rgba[3] = 0.20
            cell.set_facecolor(tuple(rgba))
    ax_table.set_title("Hyperedge summary", pad=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_channelthermal_organization_schematic_presentation(
    output_path: Path,
    sample: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    *,
    link_threshold: float = 0.25,
    min_strength: float = 0.05,
    max_hyperedges: int = 10,
) -> None:
    """Render a clean tripartite module-hyperedge-region graph."""
    del sample
    centers = arrays["centers"]
    present = arrays["present"]
    heat = arrays["heat"]
    A_mh = arrays["A_mh"]
    A_eh = arrays["A_eh"]
    strength = arrays["strength"]
    module_mass = arrays["module_mass"]
    env_mass = arrays["env_mass"]
    colors = _hyperedge_colors(strength.shape[0])
    dominant, _ = _dominant_env(A_eh, arrays["env_coords"].shape[0])
    active = [idx for idx in np.argsort(-strength) if strength[idx] >= min_strength]
    hidden_count = max(0, strength.shape[0] - len(active))
    if len(active) > max_hyperedges:
        hidden_count += len(active) - max_hyperedges
        active = active[:max_hyperedges]
    active = sorted(int(idx) for idx in active)
    if not active and strength.shape[0] > 0:
        active = [int(np.argmax(strength))]
        hidden_count = max(0, strength.shape[0] - 1)

    module_indices = list(np.flatnonzero(present))
    if centers.size and module_indices:
        module_indices.sort(key=lambda idx: (-float(centers[idx, 1]), int(idx)))
    heat_norm = _heat_scale(heat)
    fig, ax = plt.subplots(figsize=(12.0, 6.2), constrained_layout=True)
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Organizer schematic: modules -> hyperedges -> thermal regions")

    def y_positions(count: int) -> List[float]:
        if count <= 1:
            return [0.5]
        return list(np.linspace(0.86, 0.14, count))

    module_y = {idx: y for idx, y in zip(module_indices, y_positions(len(module_indices)))}
    hyper_y = {idx: y for idx, y in zip(active, y_positions(len(active)))}
    region_y = dict(hyper_y)
    module_x, hyper_x, region_x = 0.16, 0.50, 0.84

    for module_idx in module_indices:
        y = module_y[module_idx]
        size = 500.0 + 950.0 * float(heat_norm[module_idx]) if module_idx < heat_norm.shape[0] else 600.0
        ax.scatter(module_x, y, s=size, color="#fdb863", edgecolor="#8c510a", linewidth=1.1, zorder=5)
        ax.text(module_x, y, f"M{module_idx}", ha="center", va="center", fontsize=9, weight="bold", zorder=6)

    for hidx in active:
        y = hyper_y[hidx]
        size = 700.0 + 1800.0 * float(np.clip(strength[hidx], 0.0, 1.0))
        ax.scatter(hyper_x, y, s=size, color=colors[hidx], edgecolor="black", linewidth=0.8, alpha=0.88, zorder=5)
        ax.text(
            hyper_x,
            y,
            f"H{hidx}\nS={strength[hidx]:.2f}\nM={module_mass[hidx]:.2f} E={env_mass[hidx]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            zorder=6,
        )
        env_count = int(np.sum(dominant == hidx))
        width = max(0.18, 0.30 + 0.24 * float(np.clip(env_mass[hidx], 0.0, 1.0)))
        height = 0.075
        ax.add_patch(Ellipse((region_x, region_y[hidx]), width, height, facecolor=colors[hidx], edgecolor="black", lw=0.8, alpha=0.22, zorder=4))
        ax.text(region_x, region_y[hidx], f"R{hidx}\nE={env_mass[hidx]:.2f}\nn={env_count}", ha="center", va="center", fontsize=8, zorder=6)
        edge_lw = 0.8 + 4.2 * float(np.clip(max(env_mass[hidx], env_count / max(A_eh.shape[0], 1)), 0.0, 1.0))
        ax.plot([hyper_x + 0.05, region_x - 0.08], [y, region_y[hidx]], color=colors[hidx], lw=edge_lw, alpha=0.55, zorder=2)

    if A_mh.size:
        for module_idx in module_indices:
            for hidx in active:
                weight = float(A_mh[module_idx, hidx])
                if weight < float(link_threshold):
                    continue
                ax.plot(
                    [module_x + 0.045, hyper_x - 0.06],
                    [module_y[module_idx], hyper_y[hidx]],
                    color=colors[hidx],
                    lw=0.6 + 4.2 * weight,
                    alpha=0.24 + 0.56 * min(weight, 1.0),
                    zorder=1,
                )

    ax.text(module_x, 0.965, "Modules", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(hyper_x, 0.965, "Hyperedges", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(region_x, 0.965, "Thermal regions", ha="center", va="center", fontsize=10, weight="bold")
    legend_items = [
        Line2D([0], [0], color="black", lw=3.0, label="line width = soft assignment weight"),
        Line2D([0], [0], marker="o", linestyle="None", color="black", markerfacecolor="#bbbbbb", markersize=10, label="H node size = hyper_strength"),
        Line2D([0], [0], marker="o", linestyle="None", color="black", markerfacecolor="#dddddd", markersize=9, label="R node = environment/thermal region"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=3, fontsize=8, framealpha=0.90)
    if hidden_count > 0:
        ax.text(0.5, 0.045, "weak hyperedges hidden; see debug matrices", ha="center", va="center", fontsize=8, color="#444444")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_channelthermal_organization_summary_matrices(
    output_path: Path,
    sample: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    *,
    module_radius: float,
    channel_order: Optional[Sequence[str]] = None,
) -> None:
    """Render readable 2x2 assignment, mass, and physical mini-map summary."""
    del channel_order
    centers = arrays["centers"]
    present = arrays["present"]
    env_coords = arrays["env_coords"]
    A_mh = arrays["A_mh"]
    A_eh = arrays["A_eh"]
    strength = arrays["strength"]
    module_mass = arrays["module_mass"]
    env_mass = arrays["env_mass"]
    src = arrays["src"]
    dst = arrays["dst"]
    colors = _hyperedge_colors(strength.shape[0])
    dominant, confidence = _dominant_env(A_eh, env_coords.shape[0])
    sort_idx = np.lexsort((np.arange(A_eh.shape[0]), dominant)) if A_eh.size else np.arange(0)

    fig = plt.figure(figsize=(13.6, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_mh = fig.add_subplot(gs[0, 0])
    ax_eh = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_map = fig.add_subplot(gs[1, 1])

    im_mh = ax_mh.imshow(A_mh, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(np.nanmax(A_mh)) if A_mh.size else 1.0, 1.0e-6))
    ax_mh.set_title("Module -> Hyperedge assignment A_mh")
    ax_mh.set_xlabel("hyperedge")
    ax_mh.set_ylabel("module")
    ax_mh.set_xticks(np.arange(strength.shape[0]))
    ax_mh.set_xticklabels([f"H{i}" for i in range(strength.shape[0])])
    ax_mh.set_yticks(np.arange(centers.shape[0]))
    ax_mh.set_yticklabels([f"M{i}" for i in range(centers.shape[0])])
    if A_mh.shape[0] * A_mh.shape[1] <= 120:
        for i in range(A_mh.shape[0]):
            for j in range(A_mh.shape[1]):
                ax_mh.text(j, i, f"{A_mh[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if A_mh[i, j] > 0.5 else "black")
    fig.colorbar(im_mh, ax=ax_mh, fraction=0.046, pad=0.04)

    A_eh_sorted = A_eh[sort_idx] if A_eh.size else A_eh
    im_eh = ax_eh.imshow(A_eh_sorted, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(np.nanmax(A_eh)) if A_eh.size else 1.0, 1.0e-6))
    ax_eh.set_title("Environment -> Hyperedge assignment A_eh")
    ax_eh.set_xlabel("hyperedge")
    ax_eh.set_ylabel("env tokens sorted by dominant H")
    ax_eh.set_xticks(np.arange(strength.shape[0]))
    ax_eh.set_xticklabels([f"H{i}" for i in range(strength.shape[0])])
    max_ticks = min(8, A_eh_sorted.shape[0])
    if max_ticks > 0:
        tick_pos = np.linspace(0, A_eh_sorted.shape[0] - 1, max_ticks, dtype=int)
        ax_eh.set_yticks(tick_pos)
        ax_eh.set_yticklabels([str(int(idx)) for idx in tick_pos])
    fig.colorbar(im_eh, ax=ax_eh, fraction=0.046, pad=0.04)
    if A_eh.size:
        strip_colors = np.asarray([colors[int(h) % len(colors)] for h in dominant[sort_idx]])[:, None, :]
        inset = ax_eh.inset_axes([-0.055, 0.0, 0.025, 1.0], transform=ax_eh.transAxes)
        inset.imshow(strip_colors, aspect="auto")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("dom", fontsize=7)

    x = np.arange(strength.shape[0])
    width = 0.25
    ax_bar.bar(x - width, module_mass, width, label="module_mass", color="#d95f02", alpha=0.78)
    ax_bar.bar(x, env_mass, width, label="env_mass", color="#1b9e77", alpha=0.78)
    ax_bar.bar(x + width, strength, width, label="hyper_strength", color="#7570b3", alpha=0.78)
    ax_bar.set_title("Hyperedge mass and strength")
    ax_bar.set_xlabel("hyperedge")
    ax_bar.set_ylabel("value")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"H{i}" for i in range(strength.shape[0])])
    ax_bar.set_ylim(0.0, max(1.0, float(np.nanmax([np.nanmax(module_mass) if module_mass.size else 0, np.nanmax(env_mass) if env_mass.size else 0, np.nanmax(strength) if strength.size else 0])) * 1.15))
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.legend(fontsize=8)

    extent = _domain_extent(sample, env_coords)
    _draw_env_tokens(ax_map, arrays, colors)
    _draw_module_circles(ax_map, arrays, module_radius, label=True)
    for hidx in range(strength.shape[0]):
        ax_map.plot([src[hidx, 0], dst[hidx, 0]], [src[hidx, 1], dst[hidx, 1]], color=colors[hidx], lw=1.0 + 2.0 * float(strength[hidx]), alpha=0.55)
        ax_map.scatter(src[hidx, 0], src[hidx, 1], marker="x", s=42, color=colors[hidx], linewidth=1.5)
        ax_map.scatter(dst[hidx, 0], dst[hidx, 1], marker="*", s=90, color=colors[hidx], edgecolor="black", linewidth=0.45)
    ax_map.set_title("Env-token physical mini-map")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.set_xlim(extent[0], extent[1])
    ax_map.set_ylim(extent[2], extent[3])
    ax_map.set_aspect("equal", adjustable="box")
    if confidence.size:
        ax_map.text(0.01, 0.01, "color = dominant H; size/opacity = confidence", transform=ax_map.transAxes, fontsize=8, va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75})

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
