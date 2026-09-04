"""CHANNELTHERMAL-SPECIFIC organizer visualizations.

Inputs are legacy ChannelThermal samples plus organizer arrays from the HONF
wrapper. Outputs are PNG visualizations of hyperedge/module/environment
organization. This module is specific to ChannelThermal presentation geometry
and channel ordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Polygon


def _hyperedge_colors(num_h: int) -> List[Tuple[float, float, float, float]]:
    """Perform the hyperedge colors operation used by this module."""

    cmap = plt.get_cmap("tab20", max(int(num_h), 1))
    return [cmap(idx) for idx in range(max(int(num_h), 1))]


def _active_hyperedge_indices(arrays: Dict[str, np.ndarray], count: int | None = None) -> List[int]:
    """Return hard-active packed mechanisms in extraction/column order.

    A supplied active/effective mask is authoritative.  In particular, do not
    infer residual-organizer support from strength: a padded mechanism can have
    a nonzero analytic strength while remaining hard-inactive.  Historical
    arrays without an explicit mask retain the strength-based compatibility
    fallback.
    """

    strength = np.asarray(arrays.get("strength", arrays.get("hyper_strength", np.zeros((0,)))), dtype=np.float64).reshape(-1)
    if count is None:
        count = int(strength.shape[0])
    count = max(int(count), 0)
    explicit = "active_hyperedge_mask" in arrays or "effective_edge_mask" in arrays
    source = arrays.get("active_hyperedge_mask", arrays.get("effective_edge_mask"))
    if source is not None:
        mask = np.asarray(source, dtype=np.float64).reshape(-1)
        active = [idx for idx in range(count) if idx < mask.shape[0] and mask[idx] > 0.5]
        if active or explicit:
            return active
    active = [idx for idx in range(count) if idx < strength.shape[0] and strength[idx] > 0.05]
    if active:
        return active
    return list(range(count))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Perform the convex hull operation used by this module."""

    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)})
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float32)

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Perform the cross operation used by this module."""

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
    """Perform the domain extent operation used by this module."""

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
    """Perform the temperature image operation used by this module."""

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


def _dominant_env(
    A_eh: np.ndarray,
    env_count: int,
    active_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform the dominant env operation used by this module."""

    if A_eh.size:
        values = np.asarray(A_eh, dtype=np.float64)
        if active_mask is not None and values.ndim == 2:
            mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if mask.shape[0] >= values.shape[-1] and np.any(mask[: values.shape[-1]]):
                values = values.copy()
                values[:, ~mask[: values.shape[-1]]] = -np.inf
            elif mask.shape[0] >= values.shape[-1]:
                return np.zeros((values.shape[0],), dtype=np.int64), np.zeros((values.shape[0],), dtype=np.float32)
        dominant = values.argmax(axis=-1)
        confidence = np.maximum(values.max(axis=-1), 0.0)
        return dominant, confidence.astype(np.float32)
    return np.zeros((env_count,), dtype=np.int64), np.ones((env_count,), dtype=np.float32)


def _collapse_title_suffix(arrays: Dict[str, np.ndarray]) -> str:
    """Perform the collapse title suffix operation used by this module."""

    A_eh = np.asarray(arrays.get("A_eh", np.zeros((0, 0))), dtype=np.float64)
    if A_eh.size == 0:
        return ""
    eps = 1.0e-12
    active = _active_hyperedge_indices(arrays, A_eh.shape[-1])
    if active:
        values = A_eh[:, active]
    else:
        values = np.zeros((A_eh.shape[0], 1), dtype=np.float64)
    dominant_local = values.argmax(axis=-1)
    dominant = np.asarray([active[idx] for idx in dominant_local], dtype=np.int64) if active else np.zeros((A_eh.shape[0],), dtype=np.int64)
    counts = np.bincount(dominant, minlength=A_eh.shape[-1]).astype(np.float64)
    frac = counts / max(float(counts.sum()), eps)
    env_mass = values.mean(axis=0)
    env_mass = env_mass / max(float(env_mass.sum()), eps)
    entropy = -float(np.sum(env_mass * np.log(np.maximum(env_mass, eps))))
    return f" | dom={float(np.max(frac)):.2f}, softEff={float(np.exp(entropy)):.2f}"


def _heat_scale(heat: np.ndarray) -> np.ndarray:
    """Perform the heat scale operation used by this module."""

    heat_abs = np.abs(np.asarray(heat, dtype=np.float32))
    denom = max(float(np.nanmax(heat_abs)) if heat_abs.size else 0.0, 1.0e-6)
    return heat_abs / denom


def _top_modules(A_mh: np.ndarray, present: np.ndarray, hidx: int, limit: int = 3) -> Tuple[str, str]:
    """Perform the top modules operation used by this module."""

    if not A_mh.size or hidx >= A_mh.shape[1]:
        return "", ""
    valid = [(idx, float(A_mh[idx, hidx])) for idx in np.flatnonzero(present) if idx < A_mh.shape[0]]
    valid.sort(key=lambda item: item[1], reverse=True)
    top = [(idx, value) for idx, value in valid[:limit] if value > 1.0e-6]
    return ", ".join(f"M{idx}" for idx, _ in top), ", ".join(f"{value:.2f}" for _, value in top)


def _summary_rows(arrays: Dict[str, np.ndarray]) -> List[List[str]]:
    """Perform the summary rows operation used by this module."""

    A_mh = arrays["A_mh"]
    A_eh = arrays["A_eh"]
    present = arrays["present"]
    strength = arrays["strength"]
    module_mass = arrays["module_mass"]
    env_mass = arrays["env_mass"]
    active = _active_hyperedge_indices(arrays, strength.shape[0])
    dominant, _ = _dominant_env(A_eh, arrays["env_coords"].shape[0], arrays.get("active_hyperedge_mask"))
    rows: List[List[str]] = []
    for hidx in active:
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
    """Perform the draw module circles operation used by this module."""

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
        ax.add_patch(Circle((float(cx), float(cy)), module_radius, facecolor=color, edgecolor=color, alpha=alpha, lw=0.0, zorder=3))
        ax.add_patch(Circle((float(cx), float(cy)), module_radius, fill=False, edgecolor=color, lw=lw, alpha=0.95, zorder=5))
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
    """Perform the draw region hulls operation used by this module."""

    env_coords = arrays["env_coords"]
    A_eh = arrays["A_eh"]
    strength = arrays["strength"]
    if env_coords.size == 0:
        return
    active = _active_hyperedge_indices(arrays, strength.shape[0])
    dominant, _ = _dominant_env(A_eh, env_coords.shape[0], arrays.get("active_hyperedge_mask"))
    for hidx in active:
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
    """Perform the draw env tokens operation used by this module."""

    env_coords = arrays["env_coords"]
    if env_coords.size == 0:
        return
    dominant, confidence = _dominant_env(
        arrays["A_eh"], env_coords.shape[0], arrays.get("active_hyperedge_mask")
    )
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
    """Perform the draw sources regions links operation used by this module."""

    centers = arrays["centers"]
    present = arrays["present"]
    A_mh = arrays["A_mh"]
    strength = arrays["strength"]
    src = arrays["src"]
    dst = arrays["dst"]
    for hidx in _active_hyperedge_indices(arrays, strength.shape[0]):
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

    ax.set_title(f"Organizer physical overview{_collapse_title_suffix(arrays)}")
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
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def render_case_adaptive_residual_summary(
    output_path: Path,
    arrays: Dict[str, np.ndarray],
    *,
    title: str = "Case-adaptive residual mechanism summary",
) -> None:
    """Render one compact residual-extraction summary for a selected case.

    The figure intentionally summarizes the extraction sequence in one figure:
    residual waterfall, coupling/tensor decomposition, marginal gains/support,
    content factors when available, and a source-to-region map. It never writes
    one image per packed mechanism and uses the hard active mask for all
    presentation geometry.
    """

    strength = np.asarray(arrays.get("residual_mechanism_strength", arrays.get("strength", [])), dtype=np.float64).reshape(-1)
    trace = np.asarray(arrays.get("residual_fraction_trace", []), dtype=np.float64).reshape(-1)
    marginal = np.asarray(arrays.get("residual_marginal_explained_fraction", []), dtype=np.float64).reshape(-1)
    survival = np.asarray(
        arrays.get("edge_survival_soft", arrays.get("edge_survival_weight", [])),
        dtype=np.float64,
    ).reshape(-1)
    if strength.size == 0:
        strength = np.asarray(arrays.get("strength", []), dtype=np.float64).reshape(-1)
    packed_count = int(max(strength.size, trace.size - 1, marginal.size, survival.size))
    if packed_count <= 0:
        packed_count = 1
        strength = np.zeros((1,), dtype=np.float64)
    if trace.size == 0:
        trace = np.ones((packed_count + 1,), dtype=np.float64)
    elif trace.size < packed_count + 1:
        trace = np.pad(trace, (0, packed_count + 1 - trace.size), constant_values=float(trace[-1]))
    trace = trace[: packed_count + 1]
    if marginal.size < packed_count:
        marginal = np.pad(marginal, (0, packed_count - marginal.size), constant_values=0.0)
    marginal = marginal[:packed_count]
    if survival.size < packed_count:
        survival = np.pad(survival, (0, packed_count - survival.size), constant_values=0.0)
    survival = survival[:packed_count]
    if strength.size < packed_count:
        strength = np.pad(strength, (0, packed_count - strength.size), constant_values=0.0)
    strength = strength[:packed_count]
    active = _active_hyperedge_indices(arrays, packed_count)
    count_values = np.asarray(arrays.get("case_adaptive_edge_count", [len(active)]), dtype=np.float64).reshape(-1)
    cap_values = np.asarray(arrays.get("case_adaptive_edge_cap", [packed_count]), dtype=np.float64).reshape(-1)
    stop_values = np.asarray(arrays.get("residual_stop_fraction", [0.02]), dtype=np.float64).reshape(-1)
    case_count = int(round(float(count_values[0]))) if count_values.size else len(active)
    case_cap = int(round(float(cap_values[0]))) if cap_values.size else packed_count
    stop_fraction = float(stop_values[0]) if stop_values.size else 0.02

    interaction_tensor = np.asarray(
        arrays.get("residual_interaction_tensor", np.zeros((0, 0, 0))),
        dtype=np.float64,
    )
    if interaction_tensor.ndim == 4 and interaction_tensor.shape[0] == 1:
        interaction_tensor = interaction_tensor[0]
    tensor_mode = bool(interaction_tensor.ndim == 3 and interaction_tensor.size)
    A_me = np.asarray(arrays.get("A_me", np.zeros((0, 0))), dtype=np.float64)
    row_mass = np.asarray(arrays.get("residual_coupling_row_mass", np.zeros((0,))), dtype=np.float64).reshape(-1)
    if tensor_mode:
        # The organizer extracts factors from the normalized residual
        # ``R_0 = C / ||C||_1``.  Normalize the optional raw interaction tensor
        # before summing over content channels so the component heatmaps and
        # final residual are on the same scale as the extracted factors.
        interaction_mass = float(np.sum(interaction_tensor))
        normalized_tensor = (
            interaction_tensor / interaction_mass
            if interaction_mass > 1.0e-12
            else np.zeros_like(interaction_tensor)
        )
        initial = np.sum(normalized_tensor, axis=-1)
    else:
        initial = (
            A_me * row_mass[:, None]
            if A_me.ndim == 2 and A_me.shape[0] == row_mass.size
            else np.zeros((0, 0), dtype=np.float64)
        )
    module_factor = np.asarray(arrays.get("residual_module_factor", np.zeros((0, packed_count))), dtype=np.float64)
    env_factor = np.asarray(arrays.get("residual_environment_factor", np.zeros((0, packed_count))), dtype=np.float64)
    content_factor = np.asarray(arrays.get("residual_content_factor", np.zeros((0, packed_count))), dtype=np.float64)
    reconstructed = np.zeros_like(initial)
    if initial.size and module_factor.ndim == 2 and env_factor.ndim == 2:
        for hidx in active:
            if hidx >= module_factor.shape[1] or hidx >= env_factor.shape[1]:
                continue
            component = strength[hidx] * np.outer(module_factor[:, hidx], env_factor[:, hidx])
            if tensor_mode and content_factor.ndim == 2 and hidx < content_factor.shape[1]:
                component = component * float(np.sum(content_factor[:, hidx]))
            reconstructed += component
    residual = np.maximum(initial - reconstructed, 0.0) if initial.size else np.zeros((0, 0), dtype=np.float64)
    cap_hit = np.asarray(arrays.get("case_adaptive_cap_hit", [0.0]), dtype=np.float64).reshape(-1)
    final_residual = float(trace[min(max(case_count, 0), trace.size - 1)])
    cap_warning = " — CAP HIT" if cap_hit.size and cap_hit[0] > 0.5 else ""

    if tensor_mode:
        fig = plt.figure(figsize=(16.0, 12.0), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.8, 1.15])
        tensor_slot = gs[1, :]
        ax_trace = fig.add_subplot(gs[0, 0])
        ax_gain = fig.add_subplot(gs[0, 1])
        ax_content = fig.add_subplot(gs[2, 0])
        ax_map = fig.add_subplot(gs[2, 1])
    else:
        fig = plt.figure(figsize=(14.0, 9.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)
        tensor_slot = gs[0, 1]
        ax_trace = fig.add_subplot(gs[0, 0])
        ax_gain = fig.add_subplot(gs[1, 0])
        ax_map = fig.add_subplot(gs[1, 1])
    figure_title = f"{title}{cap_warning}"
    if tensor_mode:
        figure_title += " — vector interaction tensor"
    fig.suptitle(figure_title, fontsize=14)

    steps = np.arange(trace.size)
    ax_trace.plot(steps, trace, marker="o", color="#2166ac", lw=2.0, label=r"$\rho_r$")
    ax_trace.axhline(stop_fraction, color="#b2182b", ls="--", lw=1.2, label=f"stop={stop_fraction:.3g}")
    ax_trace.axvline(case_count, color="#4d4d4d", ls=":", lw=1.2, label=f"K_case={case_count}")
    ax_trace.set(title=f"Residual waterfall (K_cap={case_cap})", xlabel="extraction step r", ylabel="unexplained fraction")
    ax_trace.set_ylim(bottom=0.0)
    ax_trace.grid(alpha=0.22)
    ax_trace.legend(fontsize=8, loc="best")

    if initial.size:
        # Keep the complete decomposition in one compact panel grid: initial
        # coupling/tensor, one heatmap per hard-active rank-one component, and
        # the residual left after those selected components.  For Phase 2 the
        # tensor is summarized over content channels in this panel; the
        # companion content-factor panel retains the channel decomposition.
        initial_title = "initial $\\widetilde C$" if not tensor_mode else r"initial normalized $R_0$: $\sum_c C_{ijc}$"
        component_title = r"H{}: $\lambda ab^T$" if not tensor_mode else r"H{}: $\sum_c\lambda ab^Tw_c$"
        components: list[tuple[str, np.ndarray]] = [(initial_title, initial)]
        for hidx in active:
            if (
                module_factor.ndim == 2
                and env_factor.ndim == 2
                and hidx < module_factor.shape[1]
                and hidx < env_factor.shape[1]
            ):
                component = strength[hidx] * np.outer(module_factor[:, hidx], env_factor[:, hidx])
                if tensor_mode and content_factor.ndim == 2 and hidx < content_factor.shape[1]:
                    component = component * float(np.sum(content_factor[:, hidx]))
            else:
                component = np.zeros_like(initial)
            components.append((component_title.format(hidx), component))
        components.append(("final residual", residual))
        n_components = len(components)
        n_cols = min(4, max(1, n_components))
        n_rows = int(np.ceil(n_components / float(n_cols)))
        coupling_grid = tensor_slot.subgridspec(n_rows, n_cols, wspace=0.12, hspace=0.30)
        vmax = max(
            max(float(np.nanmax(matrix)) for _, matrix in components if matrix.size),
            1.0e-8,
        )
        coupling_axes = []
        images = []
        module_count = initial.shape[0]
        environment_count = initial.shape[1]
        module_step = max(1, int(np.ceil(module_count / 12.0)))
        environment_step = max(1, int(np.ceil(environment_count / 12.0)))
        module_ticks = np.arange(0, module_count, module_step)
        environment_ticks = np.arange(0, environment_count, environment_step)
        for component_index, (component_title, matrix) in enumerate(components):
            row_index, column_index = divmod(component_index, n_cols)
            axis = fig.add_subplot(coupling_grid[row_index, column_index])
            image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
            images.append(image)
            coupling_axes.append(axis)
            axis.set_title(component_title, fontsize=8, pad=3)
            if column_index == 0:
                axis.set_yticks(module_ticks)
                axis.set_yticklabels([f"M{i}" for i in module_ticks], fontsize=6)
            else:
                axis.set_yticks([])
            if row_index == n_rows - 1:
                axis.set_xticks(environment_ticks)
                axis.set_xticklabels([f"E{i}" for i in environment_ticks], rotation=90, fontsize=6)
            else:
                axis.set_xticks([])
        for component_index in range(n_components, n_rows * n_cols):
            row_index, column_index = divmod(component_index, n_cols)
            fig.add_subplot(coupling_grid[row_index, column_index]).axis("off")
        fig.colorbar(images[0], ax=coupling_axes, fraction=0.025, pad=0.02, label="coupling")
    else:
        coupling_grid = tensor_slot.subgridspec(1, 1)
        ax_coupling = fig.add_subplot(coupling_grid[0, 0])
        ax_coupling.text(0.5, 0.5, "coupling arrays unavailable", ha="center", va="center")
        ax_coupling.set_xticks([])
        ax_coupling.set_yticks([])
        ax_coupling.set_title("Initial coupling → residual reconstruction")

    if tensor_mode:
        if content_factor.ndim == 2 and content_factor.size:
            content_values = content_factor[:, active] if active else np.zeros((content_factor.shape[0], 1), dtype=np.float64)
            content_image = ax_content.imshow(content_values, aspect="auto", cmap="coolwarm")
            ax_content.set_title("Interaction-content factors $w_{cr}$ (hard-active H)")
            ax_content.set_xlabel("mechanism")
            ax_content.set_ylabel("interaction channel c")
            ax_content.set_xticks(np.arange(len(active)) if active else [0])
            ax_content.set_xticklabels([f"H{i}" for i in active] if active else ["none"])
            channel_step = max(1, int(np.ceil(content_values.shape[0] / 12.0)))
            channel_ticks = np.arange(0, content_values.shape[0], channel_step)
            ax_content.set_yticks(channel_ticks)
            ax_content.set_yticklabels([f"C{i}" for i in channel_ticks], fontsize=7)
            fig.colorbar(content_image, ax=ax_content, fraction=0.046, pad=0.04, label="w")
        else:
            ax_content.text(0.5, 0.5, "interaction-content factors unavailable", ha="center", va="center")
            ax_content.set_xticks([])
            ax_content.set_yticks([])
            ax_content.set_title("Interaction-content factors $w_{cr}$")

    x = np.arange(packed_count)
    width = 0.72
    ax_gain.bar(x, marginal, width=width, color="#67a9cf", alpha=0.88, label="marginal explained")
    ax_gain.set_xlabel("mechanism / extraction order")
    ax_gain.set_ylabel("marginal explained fraction")
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels([f"H{i}" for i in x], rotation=45, ha="right")
    ax_gain.grid(axis="y", alpha=0.22)
    ax_survival = ax_gain.twinx()
    ax_survival.plot(x, survival, color="#ef8a62", marker="o", lw=1.6, label="soft/hard support")
    ax_survival.plot(x, strength, color="#5e3c99", marker="x", lw=1.2, alpha=0.85, label="strength")
    ax_survival.set_ylim(bottom=0.0)
    ax_survival.set_ylabel("support / strength")
    if active:
        ax_gain.axvspan(-0.5, max(active) + 0.5, color="#bdbdbd", alpha=0.08)
    lines, labels = ax_gain.get_legend_handles_labels()
    lines2, labels2 = ax_survival.get_legend_handles_labels()
    ax_gain.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper right")
    ax_gain.set_title("Mechanism gains and support")

    centers = np.asarray(arrays.get("centers", np.zeros((0, 2))), dtype=np.float64)
    present = np.asarray(arrays.get("present", np.ones((centers.shape[0],), dtype=bool))).astype(bool).reshape(-1)
    src = np.asarray(arrays.get("src", np.zeros((packed_count, 2))), dtype=np.float64)
    dst = np.asarray(arrays.get("dst", np.zeros((packed_count, 2))), dtype=np.float64)
    if centers.ndim == 2 and centers.shape[1] >= 2 and centers.size:
        ax_map.scatter(centers[present, 0], centers[present, 1], c="#252525", s=42, label="active module")
    colors = _hyperedge_colors(packed_count)
    for hidx in active:
        if hidx >= src.shape[0] or hidx >= dst.shape[0] or src.shape[1] < 2 or dst.shape[1] < 2:
            continue
        ax_map.plot([src[hidx, 0], dst[hidx, 0]], [src[hidx, 1], dst[hidx, 1]], color=colors[hidx], lw=1.4, alpha=0.72)
        ax_map.scatter(src[hidx, 0], src[hidx, 1], marker="x", color=colors[hidx], s=48)
        ax_map.scatter(dst[hidx, 0], dst[hidx, 1], marker="*", color=colors[hidx], edgecolor="black", s=86)
        ax_map.text(dst[hidx, 0], dst[hidx, 1], f" H{hidx}", fontsize=8, va="center")
    ax_map.set_title(
        f"Active source-to-region mechanisms (K_case={case_count}, K_cap={case_cap}, "
        f"final residual={final_residual:.3g})"
    )
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.grid(alpha=0.18)
    if centers.size or src.size or dst.size:
        ax_map.set_aspect("equal", adjustable="datalim")
    fig.text(0.5, 0.01, f"final residual={final_residual:.4g}{cap_warning}; inactive packed mechanisms are omitted", ha="center", fontsize=9, color="#444444")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=180)
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
    active_mask = _active_hyperedge_indices(arrays, strength.shape[0])
    dominant, _ = _dominant_env(
        A_eh, arrays["env_coords"].shape[0], arrays.get("active_hyperedge_mask")
    )
    # The supplied hard mask is authoritative for residual mechanisms.  Keep
    # ``min_strength`` as a compatibility filter only for historical arrays
    # that have no mask at all.
    if "active_hyperedge_mask" in arrays or "effective_edge_mask" in arrays:
        active = active_mask
    else:
        active = [idx for idx in active_mask if strength[idx] >= min_strength]
    hidden_count = max(0, strength.shape[0] - len(active))
    if len(active) > max_hyperedges:
        hidden_count += len(active) - max_hyperedges
        active = active[:max_hyperedges]
    active = sorted(int(idx) for idx in active)
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
        """Perform the y positions operation used by this module."""

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
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def render_channelthermal_organization_summary_matrices(
    output_path: Path,
    sample: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    *,
    module_radius: float,
    channel_order: Optional[Sequence[str]] = None,
    sort_environment: bool = False,
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
    active = _active_hyperedge_indices(arrays, strength.shape[0])
    display_indices = list(active)
    if not display_indices and strength.shape[0] > 0:
        # Keep empty/malformed debug inputs renderable without presenting an
        # inactive packed column as a real mechanism.
        display_indices = [-1]
    display_strength = (
        strength[display_indices]
        if display_indices and display_indices[0] >= 0
        else np.zeros((1,), dtype=np.float32)
    )
    display_module_mass = (
        module_mass[display_indices]
        if display_indices and display_indices[0] >= 0
        else np.zeros((1,), dtype=np.float32)
    )
    display_env_mass = (
        env_mass[display_indices]
        if display_indices and display_indices[0] >= 0
        else np.zeros((1,), dtype=np.float32)
    )
    display_A_mh = A_mh[:, display_indices] if display_indices and display_indices[0] >= 0 else np.zeros((A_mh.shape[0], 1), dtype=np.float32)
    display_A_eh = A_eh[:, display_indices] if display_indices and display_indices[0] >= 0 else np.zeros((A_eh.shape[0], 1), dtype=np.float32)
    dominant, confidence = _dominant_env(A_eh, env_coords.shape[0], arrays.get("active_hyperedge_mask"))
    sort_idx = (
        np.lexsort((np.arange(A_eh.shape[0]), dominant))
        if sort_environment and A_eh.size
        else np.arange(A_eh.shape[0])
    )
    fig = plt.figure(figsize=(13.6, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_mh = fig.add_subplot(gs[0, 0])
    ax_eh = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_map = fig.add_subplot(gs[1, 1])

    im_mh = ax_mh.imshow(display_A_mh, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(np.nanmax(display_A_mh)) if display_A_mh.size else 1.0, 1.0e-6))
    ax_mh.set_title("Module -> active mechanism assignment A_mh")
    ax_mh.set_xlabel("hyperedge")
    ax_mh.set_ylabel("module")
    ax_mh.set_xticks(np.arange(len(display_indices)))
    ax_mh.set_xticklabels(["none" if i < 0 else f"H{i}" for i in display_indices])
    ax_mh.set_yticks(np.arange(centers.shape[0]))
    ax_mh.set_yticklabels([f"M{i}" for i in range(centers.shape[0])])
    if display_A_mh.shape[0] * display_A_mh.shape[1] <= 120:
        for i in range(display_A_mh.shape[0]):
            for j in range(display_A_mh.shape[1]):
                ax_mh.text(j, i, f"{display_A_mh[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if display_A_mh[i, j] > 0.5 else "black")
    fig.colorbar(im_mh, ax=ax_mh, fraction=0.046, pad=0.04)

    display_sort_idx = sort_idx if display_A_eh.shape[0] else np.arange(display_A_eh.shape[0])
    A_eh_sorted = display_A_eh[display_sort_idx] if display_A_eh.size else display_A_eh
    im_eh = ax_eh.imshow(A_eh_sorted, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(np.nanmax(display_A_eh)) if display_A_eh.size else 1.0, 1.0e-6))
    order_label = "sorted by dominant edge" if sort_environment else "physical token order"
    ax_eh.set_title(
        f"Environment -> active mechanism assignment A_eh ({order_label})"
        f"{_collapse_title_suffix(arrays)}"
    )
    ax_eh.set_xlabel("hyperedge")
    ax_eh.set_ylabel("environment token index" if not sort_environment else "sorted env row")
    ax_eh.set_xticks(np.arange(len(display_indices)))
    ax_eh.set_xticklabels(["none" if i < 0 else f"H{i}" for i in display_indices])
    max_ticks = min(8, A_eh_sorted.shape[0])
    if max_ticks > 0:
        tick_pos = np.linspace(0, A_eh_sorted.shape[0] - 1, max_ticks, dtype=int)
        ax_eh.set_yticks(tick_pos)
        ax_eh.set_yticklabels([str(int(display_sort_idx[idx])) for idx in tick_pos])
    fig.colorbar(im_eh, ax=ax_eh, fraction=0.046, pad=0.04)
    x = np.arange(len(display_indices))
    width = 0.25
    ax_bar.bar(x - width, display_module_mass, width, label="module_mass", color="#d95f02", alpha=0.78)
    ax_bar.bar(x, display_env_mass, width, label="env_mass", color="#1b9e77", alpha=0.78)
    ax_bar.bar(x + width, display_strength, width, label="hyper_strength", color="#7570b3", alpha=0.78)
    ax_bar.set_title("Hyperedge mass and strength")
    ax_bar.set_xlabel("hyperedge")
    ax_bar.set_ylabel("value")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["none" if i < 0 else f"H{i}" for i in display_indices])
    ax_bar.set_ylim(0.0, max(1.0, float(np.nanmax([np.nanmax(display_module_mass) if display_module_mass.size else 0, np.nanmax(display_env_mass) if display_env_mass.size else 0, np.nanmax(display_strength) if display_strength.size else 0])) * 1.15))
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.legend(fontsize=8)

    extent = _domain_extent(sample, env_coords)
    _draw_env_tokens(ax_map, arrays, colors)
    _draw_module_circles(ax_map, arrays, module_radius, label=True)
    for hidx in active:
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
    if active:
        dominant_cmap = ListedColormap([colors[idx] for idx in active])
        dominant_norm = BoundaryNorm(np.arange(len(active) + 1) - 0.5, len(active))
        dominant_mappable = plt.cm.ScalarMappable(cmap=dominant_cmap, norm=dominant_norm)
        dominant_mappable.set_array([])
        cbar = fig.colorbar(dominant_mappable, ax=ax_map, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.arange(len(active)))
        cbar.set_ticklabels([f"H{i}" for i in active])
        cbar.set_label("env token dominant H")

    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)
