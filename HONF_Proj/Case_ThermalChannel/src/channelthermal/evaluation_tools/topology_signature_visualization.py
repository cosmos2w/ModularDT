"""ChannelThermal views for generic HONF topology-signature arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np


def render_topology_signature_diagnostics(
    output_dir: str | Path,
    sample: Mapping[str, Any],
    signature: Mapping[str, Any],
    *,
    edge_fields: Any | None = None,
    field_names: Sequence[str] | None = None,
    overlap_threshold: float = 0.1,
) -> dict[str, str]:
    """Render active geometry, incidence, overlap, and optional field maps."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "topology_active_regions": str(destination / "topology_active_regions.png"),
        "topology_memberships": str(destination / "topology_memberships.png"),
        "topology_overlap_graph": str(destination / "topology_overlap_graph.png"),
    }
    _plot_active_regions(Path(outputs["topology_active_regions"]), sample, signature)
    _plot_memberships(Path(outputs["topology_memberships"]), signature)
    _plot_overlap_graph(
        Path(outputs["topology_overlap_graph"]), signature, overlap_threshold=float(overlap_threshold)
    )
    if edge_fields is not None:
        contribution_path = destination / "topology_field_contributions.png"
        _plot_field_contributions(
            contribution_path,
            sample,
            signature,
            np.asarray(edge_fields),
            field_names=field_names,
        )
        outputs["topology_field_contributions"] = str(contribution_path)
    return outputs


def _feature(signature: Mapping[str, Any], name: str) -> np.ndarray:
    names = [str(value) for value in np.asarray(signature["edge_feature_names"]).tolist()]
    return np.asarray(signature["edge_features"], dtype=np.float64)[:, names.index(name)]


def _domain_lengths(sample: Mapping[str, Any], signature: Mapping[str, Any]) -> tuple[float, float]:
    structure = sample.get("structure", {})
    lengths = np.asarray(signature.get("domain_lengths", [0.0, 0.0]), dtype=np.float64)
    lx = float(np.asarray(structure.get("domain_length_x", lengths[0])).reshape(-1)[0])
    ly = float(np.asarray(structure.get("domain_length_y", lengths[1])).reshape(-1)[0])
    return lx, ly


def _plot_active_regions(path: Path, sample: Mapping[str, Any], signature: Mapping[str, Any]) -> None:
    mask = np.asarray(signature["edge_mask"]) > 0.5
    source = np.column_stack((_feature(signature, "source_x"), _feature(signature, "source_y")))
    region = np.column_stack((_feature(signature, "region_x"), _feature(signature, "region_y")))
    source_scale = np.column_stack(
        (_feature(signature, "source_scale_x"), _feature(signature, "source_scale_y"))
    )
    region_scale = np.column_stack(
        (_feature(signature, "region_scale_x"), _feature(signature, "region_scale_y"))
    )
    lx, ly = _domain_lengths(sample, signature)
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(int(mask.sum()), 1)))
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.add_patch(Rectangle((0.0, 0.0), lx, ly, fill=False, color="black", linewidth=1.5))
    centers = np.asarray(sample.get("structure", {}).get("module_centers", np.empty((0, 2))))
    present = np.asarray(sample.get("structure", {}).get("module_present", np.ones(centers.shape[0]))) > 0.5
    if centers.ndim == 2 and centers.shape[1] == 2:
        axis.scatter(centers[present, 0], centers[present, 1], marker="s", color="black", s=28, label="modules")
    for color, edge in zip(colors, np.flatnonzero(mask)):
        axis.add_patch(
            Ellipse(
                source[edge],
                width=max(2.0 * source_scale[edge, 0], 1.0e-4),
                height=max(2.0 * source_scale[edge, 1], 1.0e-4),
                fill=False,
                edgecolor=color,
                linewidth=1.4,
                linestyle="--",
            )
        )
        axis.add_patch(
            Ellipse(
                region[edge],
                width=max(2.0 * region_scale[edge, 0], 1.0e-4),
                height=max(2.0 * region_scale[edge, 1], 1.0e-4),
                facecolor=color,
                edgecolor=color,
                alpha=0.18,
                linewidth=1.2,
            )
        )
        axis.annotate("", xy=region[edge], xytext=source[edge], arrowprops={"arrowstyle": "->", "color": color})
        axis.text(source[edge, 0], source[edge, 1], f"e{edge}", color=color, fontsize=8)
    axis.set(xlim=(0.0, lx), ylim=(0.0, ly), xlabel="x", ylabel="y", title="Active source and environment regions")
    axis.set_aspect("equal", adjustable="box")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_memberships(path: Path, signature: Mapping[str, Any]) -> None:
    module = np.asarray(signature["module_incidence"], dtype=np.float64)
    environment = np.asarray(signature["environment_incidence"], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    module_image = axes[0].imshow(module, aspect="auto", interpolation="nearest", vmin=0.0)
    axes[0].set(title="Module membership", xlabel="serialized edge", ylabel="module slot")
    figure.colorbar(module_image, ax=axes[0], fraction=0.046)
    environment_image = axes[1].imshow(environment, aspect="auto", interpolation="nearest", vmin=0.0)
    axes[1].set(title="Environment membership", xlabel="serialized edge", ylabel="environment token")
    figure.colorbar(environment_image, ax=axes[1], fraction=0.046)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_overlap_graph(path: Path, signature: Mapping[str, Any], *, overlap_threshold: float) -> None:
    relation_names = [str(value) for value in np.asarray(signature["relation_feature_names"]).tolist()]
    relations = np.asarray(signature["edge_relations"], dtype=np.float64)
    module_overlap = relations[..., relation_names.index("module_overlap")]
    environment_overlap = relations[..., relation_names.index("environment_overlap")]
    overlap = 0.5 * (module_overlap + environment_overlap)
    active = np.flatnonzero(np.asarray(signature["edge_mask"]) > 0.5)
    count = len(active)
    angles = np.linspace(0.0, 2.0 * np.pi, max(count, 1), endpoint=False)
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for left in range(count):
        for right in range(left + 1, count):
            weight = float(overlap[active[left], active[right]])
            if weight >= overlap_threshold:
                axis.plot(
                    positions[[left, right], 0],
                    positions[[left, right], 1],
                    color="tab:blue",
                    alpha=min(1.0, 0.2 + weight),
                    linewidth=0.5 + 3.0 * weight,
                )
    if count:
        axis.scatter(positions[:, 0], positions[:, 1], s=260, color="white", edgecolor="black", zorder=3)
        for position, edge in zip(positions, active):
            axis.text(position[0], position[1], f"e{edge}", ha="center", va="center", zorder=4)
    axis.set(title="Active-edge overlap graph", xlim=(-1.25, 1.25), ylim=(-1.25, 1.25))
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_field_contributions(
    path: Path,
    sample: Mapping[str, Any],
    signature: Mapping[str, Any],
    edge_fields: np.ndarray,
    *,
    field_names: Sequence[str] | None,
) -> None:
    if edge_fields.ndim != 3:
        raise ValueError("edge_fields must have shape [Q,K_cap,F].")
    permutation = np.asarray(signature["serialization_permutation"], dtype=np.int64)
    edge_fields = edge_fields[:, permutation, :]
    active = np.flatnonzero(np.asarray(signature["edge_mask"]) > 0.5)
    names = list(field_names or np.asarray(signature["field_names"]).tolist())
    x_grid = np.asarray(sample["x_grid"])
    y_grid = np.asarray(sample["y_grid"])
    if edge_fields.shape[0] != x_grid.size:
        raise ValueError("edge_fields query count does not match the evaluation grid.")
    rows = max(len(active), 1)
    columns = max(edge_fields.shape[-1], 1)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 2.4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    if not len(active):
        axes[0, 0].text(0.5, 0.5, "No active edges", ha="center", va="center")
        axes[0, 0].axis("off")
    for row, edge in enumerate(active):
        for channel in range(edge_fields.shape[-1]):
            values = edge_fields[:, edge, channel].reshape(x_grid.shape)
            limit = max(float(np.max(np.abs(values))), 1.0e-8)
            image = axes[row, channel].pcolormesh(
                x_grid, y_grid, values, shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit
            )
            axes[row, channel].set_title(f"e{edge} · {names[channel] if channel < len(names) else channel}")
            axes[row, channel].set_aspect("equal", adjustable="box")
            figure.colorbar(image, ax=axes[row, channel], fraction=0.046)
    figure.savefig(path, dpi=150)
    plt.close(figure)
