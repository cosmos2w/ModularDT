"""CHANNELTHERMAL-SPECIFIC HONF routing visualizations.

Inputs are query-grid routing arrays from `ChannelThermalHONFModel`, a legacy
ChannelThermal sample, and organizer arrays. Outputs are PNG maps and compact
NPZ/JSON diagnostics for query-dependent HONF routing.

This module is ChannelThermal-specific because it overlays the demo channel
geometry, module outlines, and hyperedge source/region coordinates. The CORE
HONF model only provides generic tensors: `alpha_qk`, H-routed pairwise
contributions, and context norms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


def _domain_extent(sample: Dict[str, Any]) -> tuple[float, float, float, float]:
    x_grid = np.asarray(sample.get("x_grid"))
    y_grid = np.asarray(sample.get("y_grid"))
    if x_grid.size and y_grid.size:
        return float(np.nanmin(x_grid)), float(np.nanmax(x_grid)), float(np.nanmin(y_grid)), float(np.nanmax(y_grid))
    return 0.0, 1.0, 0.0, 1.0


def _grid_shape(sample: Dict[str, Any]) -> tuple[int, int]:
    x_grid = np.asarray(sample["x_grid"])
    return int(x_grid.shape[0]), int(x_grid.shape[1])


def _as_grid(values: np.ndarray, sample: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim >= 2 and arr.shape[:2] == _grid_shape(sample):
        return arr
    return arr.reshape(*_grid_shape(sample), *arr.shape[1:])


def _active_edges(arrays: Dict[str, np.ndarray], num_hyperedges: int) -> list[int]:
    mask = np.asarray(arrays.get("active_hyperedge_mask", np.ones((num_hyperedges,), dtype=np.float32))).reshape(-1)
    strength = np.asarray(arrays.get("strength", arrays.get("hyper_strength", np.ones((num_hyperedges,), dtype=np.float32)))).reshape(-1)
    active = [idx for idx in range(num_hyperedges) if idx < mask.shape[0] and mask[idx] > 0.5]
    if not active:
        active = [idx for idx in range(num_hyperedges) if idx < strength.shape[0] and strength[idx] > 0.05]
    return active or list(range(num_hyperedges))


def _colors(num_hyperedges: int) -> list[Any]:
    cmap = plt.get_cmap("tab20", max(int(num_hyperedges), 1))
    return [cmap(idx) for idx in range(max(int(num_hyperedges), 1))]


def _overlay_geometry(ax: Any, arrays: Dict[str, np.ndarray], module_radius: float, colors: Sequence[Any]) -> None:
    centers = np.asarray(arrays.get("centers", np.zeros((0, 2))), dtype=np.float32)
    present = np.asarray(arrays.get("present", np.ones((centers.shape[0],), dtype=bool))).astype(bool)
    for module_idx in np.flatnonzero(present):
        cx, cy = centers[module_idx]
        ax.add_patch(Circle((float(cx), float(cy)), module_radius, fill=False, edgecolor="white", lw=2.0, alpha=0.92, zorder=5))
        ax.add_patch(Circle((float(cx), float(cy)), module_radius, fill=False, edgecolor="black", lw=0.7, alpha=0.85, zorder=6))
        ax.text(float(cx), float(cy), f"M{module_idx}", ha="center", va="center", fontsize=7, color="black", zorder=7)
    src = np.asarray(arrays.get("src", np.zeros((0, 2))), dtype=np.float32)
    dst = np.asarray(arrays.get("dst", np.zeros((0, 2))), dtype=np.float32)
    count = min(src.shape[0], dst.shape[0], len(colors))
    for hidx in range(count):
        ax.scatter(src[hidx, 0], src[hidx, 1], marker="x", s=42, color=colors[hidx], linewidth=1.8, zorder=8)
        ax.scatter(dst[hidx, 0], dst[hidx, 1], marker="*", s=92, color=colors[hidx], edgecolor="black", linewidth=0.45, zorder=8)
        ax.plot([src[hidx, 0], dst[hidx, 0]], [src[hidx, 1], dst[hidx, 1]], color=colors[hidx], lw=0.9, alpha=0.50, zorder=4)


def _format_axis(ax: Any, sample: Dict[str, Any], arrays: Dict[str, np.ndarray], module_radius: float, colors: Sequence[Any], title: str) -> None:
    extent = _domain_extent(sample)
    _overlay_geometry(ax, arrays, module_radius, colors)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")


def _add_geometry_legend(fig: Any, colors: Sequence[Any]) -> None:
    """Explain ChannelThermal overlays without changing the panel data scale."""

    sample_color = colors[0] if colors else "black"
    handles = [
        Line2D([0], [0], color="white", marker="o", markerfacecolor="none", markeredgecolor="black", lw=0, label="module outline / M id"),
        Line2D([0], [0], color=sample_color, marker="x", lw=0, label="H source: module-side center"),
        Line2D([0], [0], color=sample_color, marker="*", markeredgecolor="black", lw=0, label="H region: env-side center"),
        Line2D([0], [0], color=sample_color, lw=1.2, label="source-to-region link"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11)


def _panel_grid(count: int) -> tuple[int, int]:
    cols = min(3, max(1, int(np.ceil(np.sqrt(max(count, 1))))))
    rows = int(np.ceil(max(count, 1) / cols))
    return rows, cols


def _plot_edge_panels(
    output_path: Path,
    sample: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    values: np.ndarray,
    *,
    title_prefix: str,
    cmap: str,
    module_radius: float,
    figure_note: str | None = None,
    colorbar_label: str | None = None,
) -> None:
    num_h = int(values.shape[-1])
    active = _active_edges(arrays, num_h)
    colors = _colors(num_h)
    extent = _domain_extent(sample)
    rows, cols = _panel_grid(len(active))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.35 * rows), squeeze=False, constrained_layout=True)
    if figure_note:
        fig.suptitle(figure_note, fontsize=13)
    vmax = float(np.nanmax(values[..., active])) if values.size and active else 1.0
    vmax = max(vmax, 1.0e-8)
    for panel_idx, hidx in enumerate(active):
        ax = axes[panel_idx // cols][panel_idx % cols]
        im = ax.imshow(values[..., hidx], origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        _format_axis(ax, sample, arrays, module_radius, colors, f"{title_prefix} H{hidx}")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=11)
    for panel_idx in range(len(active), rows * cols):
        axes[panel_idx // cols][panel_idx % cols].axis("off")
    _add_geometry_legend(fig, colors)
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def _plot_dominant(
    output_path: Path,
    sample: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    dominant: np.ndarray,
    *,
    module_radius: float,
) -> None:
    num_h = int(max(np.nanmax(dominant) + 1 if dominant.size else 1, np.asarray(arrays.get("strength", [1])).shape[0]))
    colors = _colors(num_h)
    cmap = plt.get_cmap("tab20", max(num_h, 1))
    extent = _domain_extent(sample)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    im = ax.imshow(dominant, origin="lower", extent=extent, cmap=cmap, vmin=-0.5, vmax=max(num_h - 0.5, 0.5), aspect="auto")
    _format_axis(ax, sample, arrays, module_radius, colors, "Dominant query hyperedge argmax_k alpha_qk")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("hyperedge")
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def _plot_context_norms(
    output_path: Path,
    sample: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    c_h_norm: np.ndarray,
    c_pair_norm: np.ndarray,
    *,
    module_radius: float,
) -> None:
    eps = 1.0e-8
    ratio = c_pair_norm / (c_h_norm + c_pair_norm + eps)
    colors = _colors(int(np.asarray(arrays.get("strength", [1])).shape[0]))
    extent = _domain_extent(sample)
    panels = [
        ("c_H norm", c_h_norm, "magma"),
        ("c_pair norm", c_pair_norm, "viridis"),
        ("c_pair / (c_H + c_pair)", ratio, "coolwarm"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), constrained_layout=True)
    for ax, (title, values, cmap) in zip(axes, panels):
        vmax = float(np.nanmax(values)) if values.size else 1.0
        vmax = max(vmax, 1.0e-8)
        im = ax.imshow(values, origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        _format_axis(ax, sample, arrays, module_radius, colors, title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def save_routing_diagnostics(
    output_dir: Path,
    sample: Dict[str, Any],
    routing_maps: Dict[str, np.ndarray],
    organizer_arrays: Dict[str, np.ndarray],
    *,
    module_radius: float,
    routing_view: str = "all",
) -> Dict[str, str]:
    """Save opt-in HONF query-routing maps and return output paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    alpha = _as_grid(np.asarray(routing_maps["query_hyper_attention"], dtype=np.float32), sample)
    pair = _as_grid(np.asarray(routing_maps["pairwise_edge_contribution"], dtype=np.float32), sample)
    c_h = _as_grid(np.asarray(routing_maps["c_H_norm"], dtype=np.float32), sample)
    c_pair = _as_grid(np.asarray(routing_maps["c_pair_norm"], dtype=np.float32), sample)
    dominant = _as_grid(np.asarray(routing_maps.get("dominant_hyperedge", np.argmax(alpha, axis=-1)), dtype=np.int64), sample)
    entropy = _as_grid(np.asarray(routing_maps.get("hyper_attention_entropy", np.zeros(alpha.shape[:2])), dtype=np.float32), sample)

    npz_path = output_dir / "routing_maps.npz"
    np.savez_compressed(
        npz_path,
        query_hyper_attention=alpha.astype(np.float32),
        pairwise_edge_contribution=pair.astype(np.float32),
        c_H_norm=c_h.astype(np.float32),
        c_pair_norm=c_pair.astype(np.float32),
        dominant_hyperedge=dominant.astype(np.int64),
        hyper_attention_entropy=entropy.astype(np.float32),
    )
    summary = {
        "num_hyperedges": int(alpha.shape[-1]),
        "active_hyperedges": _active_edges(organizer_arrays, int(alpha.shape[-1])),
        "alpha_mean_by_hyperedge": np.nanmean(alpha.reshape(-1, alpha.shape[-1]), axis=0).astype(float).tolist(),
        "pairwise_contribution_mean_by_hyperedge": np.nanmean(pair.reshape(-1, pair.shape[-1]), axis=0).astype(float).tolist(),
        "c_H_norm_mean": float(np.nanmean(c_h)),
        "c_pair_norm_mean": float(np.nanmean(c_pair)),
        "entropy_mean": float(np.nanmean(entropy)),
        "note": "alpha_qk is query-dependent and is recomputed by the HONF decoder for each query grid.",
        "pairwise_edge_contribution_meaning": (
            "Each H panel shows ||g_pair * alpha_qk * edge_pair_context_qk||. "
            "Bright regions are query locations where that hyperedge contributes strong routed module-detail information "
            "to c_pair(q); the value is a diagnostic magnitude, not temperature/flux and not signed."
        ),
    }
    summary_path = output_dir / "routing_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    outputs = {"routing_maps_npz": str(npz_path), "routing_summary": str(summary_path)}
    if routing_view == "none":
        return outputs
    if routing_view in {"summary", "all"}:
        dominant_path = output_dir / "routing_dominant_edge.png"
        context_path = output_dir / "routing_context_norms.png"
        _plot_dominant(dominant_path, sample, organizer_arrays, dominant, module_radius=module_radius)
        _plot_context_norms(context_path, sample, organizer_arrays, c_h, c_pair, module_radius=module_radius)
        outputs["routing_dominant_edge"] = str(dominant_path)
        outputs["routing_context_norms"] = str(context_path)
    if routing_view == "all":
        attention_path = output_dir / "routing_attention_maps.png"
        pair_path = output_dir / "routing_pairwise_contribution_maps.png"
        _plot_edge_panels(
            attention_path,
            sample,
            organizer_arrays,
            alpha,
            title_prefix="alpha_qk",
            cmap="viridis",
            module_radius=module_radius,
            figure_note="Query-to-H routing: bright = this query selects the shown hyperedge more strongly.",
            colorbar_label="attention weight alpha_qk",
        )
        _plot_edge_panels(
            pair_path,
            sample,
            organizer_arrays,
            pair,
            title_prefix="pairwise routed detail",
            cmap="plasma",
            module_radius=module_radius,
            figure_note=(
                "Per-edge pairwise contribution: bright = strong H-routed module-detail contribution to c_pair(q). "
                "Magnitude is ||g_pair * alpha_qk * edge_pair_context_qk||."
            ),
            colorbar_label="diagnostic norm, not physical units",
        )
        outputs["routing_attention_maps"] = str(attention_path)
        outputs["routing_pairwise_contribution_maps"] = str(pair_path)
    return outputs
