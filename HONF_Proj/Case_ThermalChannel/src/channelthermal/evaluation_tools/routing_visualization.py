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
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


def _domain_extent(sample: Dict[str, Any]) -> tuple[float, float, float, float]:
    """Perform the domain extent operation used by this module."""

    x_grid = np.asarray(sample.get("x_grid"))
    y_grid = np.asarray(sample.get("y_grid"))
    if x_grid.size and y_grid.size:
        return float(np.nanmin(x_grid)), float(np.nanmax(x_grid)), float(np.nanmin(y_grid)), float(np.nanmax(y_grid))
    return 0.0, 1.0, 0.0, 1.0


def _grid_shape(sample: Dict[str, Any]) -> tuple[int, int]:
    """Perform the grid shape operation used by this module."""

    x_grid = np.asarray(sample["x_grid"])
    return int(x_grid.shape[0]), int(x_grid.shape[1])


def _as_grid(values: np.ndarray, sample: Dict[str, Any]) -> np.ndarray:
    """Perform the as grid operation used by this module."""

    arr = np.asarray(values)
    if arr.ndim >= 2 and arr.shape[:2] == _grid_shape(sample):
        return arr
    return arr.reshape(*_grid_shape(sample), *arr.shape[1:])


def _active_edges(arrays: Dict[str, np.ndarray], num_hyperedges: int) -> list[int]:
    """Perform the active edges operation used by this module."""

    explicit = "active_hyperedge_mask" in arrays or "effective_edge_mask" in arrays
    mask = np.asarray(
        arrays.get("active_hyperedge_mask", arrays.get("effective_edge_mask", np.ones((num_hyperedges,), dtype=np.float32)))
    ).reshape(-1)
    strength = np.asarray(arrays.get("strength", arrays.get("hyper_strength", np.ones((num_hyperedges,), dtype=np.float32)))).reshape(-1)
    active = [idx for idx in range(num_hyperedges) if idx < mask.shape[0] and mask[idx] > 0.5]
    if active or explicit:
        return active
    if not active:
        active = [idx for idx in range(num_hyperedges) if idx < strength.shape[0] and strength[idx] > 0.05]
    return active or list(range(num_hyperedges))


def _colors(num_hyperedges: int) -> list[Any]:
    """Perform the colors operation used by this module."""

    cmap = plt.get_cmap("tab20", max(int(num_hyperedges), 1))
    return [cmap(idx) for idx in range(max(int(num_hyperedges), 1))]


def _overlay_geometry(ax: Any, arrays: Dict[str, np.ndarray], module_radius: float, colors: Sequence[Any]) -> None:
    """Perform the overlay geometry operation used by this module."""

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
    active = _active_edges(arrays, count)
    for hidx in active:
        ax.scatter(src[hidx, 0], src[hidx, 1], marker="x", s=42, color=colors[hidx], linewidth=1.8, zorder=8)
        ax.scatter(dst[hidx, 0], dst[hidx, 1], marker="*", s=92, color=colors[hidx], edgecolor="black", linewidth=0.45, zorder=8)
        ax.plot([src[hidx, 0], dst[hidx, 0]], [src[hidx, 1], dst[hidx, 1]], color=colors[hidx], lw=0.9, alpha=0.50, zorder=4)


def _format_axis(ax: Any, sample: Dict[str, Any], arrays: Dict[str, np.ndarray], module_radius: float, colors: Sequence[Any], title: str) -> None:
    """Perform the format axis operation used by this module."""

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
    """Perform the panel grid operation used by this module."""

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
    """Perform the plot edge panels operation used by this module."""

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
    """Perform the plot dominant operation used by this module."""

    num_h = int(max(np.nanmax(dominant) + 1 if dominant.size else 1, np.asarray(arrays.get("strength", [1])).shape[0]))
    active = _active_edges(arrays, num_h)
    colors = _colors(num_h)
    if active:
        local_index = {edge: index for index, edge in enumerate(active)}
        mapped = np.full(dominant.shape, -1, dtype=np.int64)
        for edge, index in local_index.items():
            mapped[dominant == edge] = index
        dominant_to_plot = mapped
        cmap = plt.get_cmap("tab20", max(len(active), 1))
        vmax = max(len(active) - 0.5, 0.5)
    else:
        dominant_to_plot = np.full(dominant.shape, -1, dtype=np.int64)
        cmap = plt.get_cmap("Greys", 1)
        vmax = 0.5
    extent = _domain_extent(sample)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    im = ax.imshow(dominant_to_plot, origin="lower", extent=extent, cmap=cmap, vmin=-0.5, vmax=vmax, aspect="auto")
    _format_axis(ax, sample, arrays, module_radius, colors, "Dominant active mechanism argmax_r alpha_qr")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("active mechanism (packed index)")
    if active:
        cbar.set_ticks(np.arange(len(active)))
        cbar.set_ticklabels([f"H{edge}" for edge in active])
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
    """Perform the plot context norms operation used by this module."""

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


def _reader_npz_scalar(values: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Convert one saved receiver vector to its saved query-grid shape."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape[:2] == grid_shape:
        if array.ndim == 2:
            return array
        if array.ndim == 3 and array.shape[-1] == 1:
            return array[..., 0]
    if array.size == int(np.prod(grid_shape)):
        return array.reshape(grid_shape)
    if array.ndim == 2 and array.shape[0] == int(np.prod(grid_shape)):
        return array[:, 0].reshape(grid_shape)
    raise ValueError(f"Saved reader map has shape {array.shape}, expected a scalar map for {grid_shape}.")


def _reader_npz_first(data: Any, keys: Sequence[str]) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key]), key
    return None, None


def _reader_npz_maps(data: Any, grid_shape: tuple[int, int]) -> tuple[dict[str, np.ndarray | None], list[str]]:
    """Read the five compact scalar maps from one saved debug NPZ."""

    notes: list[str] = []
    geometric, geometric_key = _reader_npz_first(
        data,
        (
            "group_read_geometric_availability",
            "interaction__group_read_geometric_availability",
            "group_read_effective_geometric_weight",
            "interaction__group_read_effective_geometric_weight",
        ),
    )
    if geometric is None:
        geometric_weight, _ = _reader_npz_first(
            data,
            ("group_read_geometric_weight", "interaction__group_read_geometric_weight"),
        )
        group_index, _ = _reader_npz_first(
            data,
            ("group_read_group_index", "interaction__group_read_group_index"),
        )
        occupancy, _ = _reader_npz_first(
            data,
            ("group_occupancy_envelope", "interaction__group_occupancy_envelope"),
        )
        if geometric_weight is not None and group_index is not None and occupancy is not None:
            weight = np.asarray(geometric_weight, dtype=np.float64)
            indices = np.asarray(group_index, dtype=np.int64)
            envelope = np.asarray(occupancy, dtype=np.float64).reshape(-1)
            if weight.shape == indices.shape and weight.ndim >= 2 and envelope.size:
                valid = (indices >= 0) & (indices < envelope.size)
                safe_indices = np.where(valid, indices, 0)
                effective = np.where(valid, weight * envelope[safe_indices], 0.0)
                geometric = np.nansum(effective, axis=-1)
                geometric_key = "reconstructed_occupancy_envelope_times_geometric_weight"
                notes.append(
                    "G reconstructed as sum of occupancy_envelope[group] * B with invalid group slots masked"
                )
        if geometric is None:
            geometric_key = None
            notes.append(
                "geometric availability G unavailable; saved B lacks effective G or occupancy-envelope/group-index inputs"
            )
    if geometric is not None and geometric.ndim >= 2 and geometric.shape[:2] != grid_shape:
        geometric = np.nansum(geometric, axis=-1)
    mass, _ = _reader_npz_first(
        data,
        ("group_read_weight_mass", "interaction__group_read_weight_mass"),
    )
    conditional, conditional_key = _reader_npz_first(
        data,
        ("group_read_conditional_weight", "interaction__group_read_conditional_weight"),
    )
    if conditional is None:
        normalized, _ = _reader_npz_first(
            data,
            ("group_read_normalized_weight", "interaction__group_read_normalized_weight"),
        )
        if normalized is not None:
            normalized = np.asarray(normalized, dtype=np.float64)
            if normalized.ndim >= 2 and normalized.shape[:2] != grid_shape:
                positive = np.where(np.isfinite(normalized) & (normalized > 0.0), normalized, 0.0)
                denominator = np.sum(positive, axis=-1)
                conditional = np.full_like(positive, np.nan, dtype=np.float64)
                valid_rows = denominator > 0.0
                conditional[valid_rows] = positive[valid_rows] / denominator[valid_rows, None]
                conditional_key = "reconstructed_conditional_from_normalized_weight"
                notes.append(
                    "conditional pi reconstructed from positive normalized read weights; all-zero rows are unavailable"
                )
            else:
                conditional_key = None
                notes.append("conditional pi unavailable; normalized read weights have no saved slot axis")
    main_context, _ = _reader_npz_first(
        data,
        ("main_context_norm", "interaction__main_context_norm"),
    )
    pred_field, _ = _reader_npz_first(data, ("pred_field_grid",))
    gt_field, _ = _reader_npz_first(data, ("gt_field_grid",))
    if conditional is not None and conditional.ndim >= 2 and conditional.shape[:2] != grid_shape:
        finite_rows = np.isfinite(conditional).any(axis=-1)
        collapsed = np.full(conditional.shape[:-1], np.nan, dtype=np.float64)
        if np.any(finite_rows):
            collapsed[finite_rows] = np.nanmax(conditional[finite_rows], axis=-1)
        conditional = collapsed
    if pred_field is not None and gt_field is not None:
        temperature_error = np.abs(pred_field[..., 4] - gt_field[..., 4])
        fluid_mask, _ = _reader_npz_first(data, ("fluid_mask",))
        if fluid_mask is None:
            x_grid = np.asarray(data["x_grid"], dtype=np.float64) if "x_grid" in data.files else None
            y_grid = np.asarray(data["y_grid"], dtype=np.float64) if "y_grid" in data.files else None
            centers = np.asarray(data["module_centers"], dtype=np.float64) if "module_centers" in data.files else None
            present = np.asarray(data["module_present"], dtype=np.float64).reshape(-1) if "module_present" in data.files else None
            radius_values = np.asarray(data["module_radius"], dtype=np.float64).reshape(-1) if "module_radius" in data.files else np.asarray([], dtype=np.float64)
            radius = float(radius_values[0]) if radius_values.size and np.isfinite(radius_values[0]) else 0.45
            if x_grid is not None and y_grid is not None and centers is not None and present is not None:
                module_mask = np.zeros(x_grid.shape, dtype=bool)
                for module_idx in np.flatnonzero(present > 0.5):
                    if module_idx < centers.shape[0]:
                        module_mask |= np.hypot(x_grid - centers[module_idx, 0], y_grid - centers[module_idx, 1]) <= radius
                fluid_mask = ~module_mask
                notes.append(
                    "canonical fluid_mask absent; temperature error uses a legacy circle mask from saved centers/radius"
                )
        if fluid_mask is not None:
            temperature_error = np.where(
                _reader_npz_scalar(np.asarray(fluid_mask), grid_shape).astype(bool),
                temperature_error,
                np.nan,
            )
        else:
            temperature_error = None
            notes.append("fluid temperature mask unavailable; temperature error panel is omitted")
    else:
        temperature_error = None

    maps: dict[str, np.ndarray | None] = {}
    for name, values in (
        ("G", geometric),
        ("mass", mass),
        ("conditional", conditional),
        ("main", main_context),
        ("temperature", temperature_error),
    ):
        maps[name] = None if values is None else _reader_npz_scalar(values, grid_shape)
    if geometric_key is None:
        notes.append("geometric availability G is absent")
    if conditional_key is None:
        notes.append("conditional pi is absent")
    return maps, notes


def plot_reader_anchor_npz_maps(
    artifacts: Sequence[tuple[str, str, Path]],
    output_path: Path,
    *,
    module_radius: float = 0.45,
) -> None:
    """Plot shared-scale reader maps from saved anchor debug NPZ files.

    Each artifact is ``(model_label, case_id, debug_npz_path)``.  The function
    only reads saved arrays; it does not load checkpoints or run inference.
    Rows may contain old and new reader artifacts, while every metric column
    shares one color scale across all rows.
    """

    if not artifacts:
        raise ValueError("At least one saved reader NPZ is required.")
    loaded: list[dict[str, Any]] = []
    notes: list[str] = []
    for model_label, case_id, path in artifacts:
        with np.load(path, allow_pickle=False) as data:
            x_grid = np.asarray(data["x_grid"], dtype=np.float64)
            y_grid = np.asarray(data["y_grid"], dtype=np.float64)
            grid_shape = tuple(int(value) for value in x_grid.shape)
            maps, artifact_notes = _reader_npz_maps(data, grid_shape)
            centers = np.asarray(data["module_centers"], dtype=np.float64)
            present = np.asarray(data["module_present"], dtype=np.float64)
            radius_values = np.asarray(data["module_radius"], dtype=np.float64).reshape(-1) if "module_radius" in data.files else np.asarray([], dtype=np.float64)
            saved_radius = float(radius_values[0]) if radius_values.size and np.isfinite(radius_values[0]) else float(module_radius)
        loaded.append(
            {
                "label": model_label,
                "case": case_id,
                "x": x_grid,
                "y": y_grid,
                "maps": maps,
                "centers": centers,
                "present": present,
                "radius": saved_radius,
            }
        )
        notes.extend(f"{model_label}/{case_id}: {note}" for note in artifact_notes)

    columns = (
        ("G", "geometric availability G", "viridis", False),
        ("mass", "read mass", "magma", True),
        ("conditional", "maximum conditional/read weight", "plasma", True),
        ("main", "main context norm", "cividis", False),
        ("temperature", "physical fluid temperature absolute error", "inferno", False),
    )
    norms: dict[str, Normalize | LogNorm | None] = {}
    for name, _, _, logarithmic in columns:
        finite = np.concatenate(
            [
                values[np.isfinite(values)]
                for item in loaded
                if (values := item["maps"][name]) is not None
                and np.isfinite(values).any()
            ]
        ) if any(item["maps"][name] is not None and np.isfinite(item["maps"][name]).any() for item in loaded) else np.asarray([])
        if finite.size == 0:
            norms[name] = None
        elif logarithmic and np.any(finite > 0.0):
            positive = finite[finite > 0.0]
            lower = float(np.min(positive))
            upper = float(np.max(positive))
            norms[name] = LogNorm(vmin=lower, vmax=upper if upper > lower else lower * 10.0)
        else:
            lower = 0.0 if name in {"G", "mass", "conditional", "temperature"} else float(np.min(finite))
            upper = max(float(np.max(finite)), lower + 1.0e-8)
            norms[name] = Normalize(vmin=lower, vmax=upper)

    row_count = len(loaded)
    fig, axes = plt.subplots(
        row_count,
        len(columns),
        figsize=(4.1 * len(columns), 3.7 * row_count),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.88, wspace=0.28, hspace=0.36)
    conditional_fallback = any("conditional pi reconstructed" in note for note in notes)
    images: dict[str, Any] = {}
    for row_index, item in enumerate(loaded):
        sample = {"x_grid": item["x"], "y_grid": item["y"]}
        geometry = {"centers": item["centers"], "present": item["present"]}
        for column_index, (name, label, cmap, _) in enumerate(columns):
            axis = axes[row_index][column_index]
            values = item["maps"][name]
            norm = norms[name]
            if values is None or norm is None:
                axis.text(0.5, 0.5, "saved map unavailable", ha="center", va="center", fontsize=8)
                axis.set_axis_off()
                continue
            image = axis.imshow(
                values,
                origin="lower",
                extent=_domain_extent(sample),
                cmap=cmap,
                norm=norm,
                aspect="auto",
            )
            images.setdefault(name, image)
            _format_axis(axis, sample, geometry, float(item["radius"]), [], label)
            if column_index == 0:
                axis.text(
                    -0.08,
                    0.5,
                    f"{item['label']}\ncase {item['case']}",
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                )
    for column_index, (name, label, cmap, _) in enumerate(columns):
        norm = norms[name]
        image = images.get(name)
        if image is not None and norm is not None:
            fig.colorbar(image, ax=axes[:, column_index].tolist(), label=label, fraction=0.03, pad=0.02)
    note = "Shared column scales across saved anchors/models; maps are debug-NPZ reads only."
    if conditional_fallback:
        note += " Conditional column reconstructs pi from positive normalized weights; all-zero rows are unavailable."
    fig.suptitle("Saved reader maps on matched anchors", fontsize=13)
    fig.text(0.5, 0.035, note, ha="center", va="bottom", fontsize=8, color="#555555")
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    supplied_dominant = routing_maps.get("dominant_hyperedge")
    dominant = _as_grid(
        np.asarray(supplied_dominant if supplied_dominant is not None else np.argmax(alpha, axis=-1), dtype=np.int64),
        sample,
    )
    entropy = _as_grid(np.asarray(routing_maps.get("hyper_attention_entropy", np.zeros(alpha.shape[:2])), dtype=np.float32), sample)
    active = _active_edges(organizer_arrays, int(alpha.shape[-1]))
    if active:
        valid = np.zeros((alpha.shape[-1],), dtype=bool)
        valid[active] = True
        alpha = alpha.copy()
        alpha[..., ~valid] = 0.0
        pair = pair.copy()
        pair[..., ~valid] = 0.0
        # If an upstream map was computed before support masking, recompute its
        # dominant index from the surviving route rather than exposing an
        # inactive packed column in the summary.
        masked_dominant = np.argmax(alpha[..., active], axis=-1)
        dominant = np.asarray(active, dtype=np.int64)[masked_dominant]
    else:
        dominant = np.full(dominant.shape, -1, dtype=np.int64)

    npz_path = output_dir / "routing_maps.npz"
    np.savez_compressed(
        npz_path,
        query_hyper_attention=alpha.astype(np.float32),
        pairwise_edge_contribution=pair.astype(np.float32),
        c_H_norm=c_h.astype(np.float32),
        c_pair_norm=c_pair.astype(np.float32),
        dominant_hyperedge=dominant.astype(np.int64),
        hyper_attention_entropy=entropy.astype(np.float32),
        active_hyperedge_mask=np.asarray(
            [1.0 if idx in active else 0.0 for idx in range(alpha.shape[-1])], dtype=np.float32
        ),
    )
    summary = {
        "num_hyperedges": int(alpha.shape[-1]),
        "active_hyperedges": active,
        "active_hyperedge_mask": np.asarray(
            [1.0 if idx in active else 0.0 for idx in range(alpha.shape[-1])], dtype=np.float32
        ).tolist(),
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
