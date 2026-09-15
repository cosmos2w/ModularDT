"""Comparative native-slice figures for the two WindFarm forward runs.

This module is deliberately separate from the sampled evaluator.  It selects
validation rows from geometry metadata, reads one native plane at a time, and
asks each checkpoint for predictions in bounded receiver chunks.  The figures
therefore show the actual variable-domain native coordinates; no padded grid,
interpolation, or full-volume prediction is created.

The Classic checkpoint exposes learned hyperedge assignments.  The optional
diagnostic in this module renders those assignments as *latent organization*.
They are learned routing coordinates and attention labels, not a physical wake
topology or a substitute for CFD interpretation.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_runtime.run_store import atomic_write_json

from .data import WindFarmNativeView, case_batch
from .study_spatial import native_plane
from .workflows.evaluate_forward import _compact_metadata, _load_split, load_checkpoint

DEFAULT_OUTPUT_DIR = Path("docs/reports/windfarm_forward_2500_comparison/figures/native_fields")
DEFAULT_HUB_HEIGHT_M = 70.0


@dataclass(frozen=True)
class GeometryRecord:
    """Geometry-only descriptors used for validation-row selection."""

    row: int
    layout_index: int
    case: str
    wind_direction_deg: float
    n_turbines: int
    support_extent_D: tuple[float, float, float]
    support_volume_D3: float
    min_spacing_D: float | None
    mean_spacing_D: float | None

    def descriptor(self) -> np.ndarray:
        """Return a finite descriptor used only by the diversity heuristic."""

        direction = np.deg2rad(self.wind_direction_deg)
        spacing = 0.0 if self.min_spacing_D is None else self.min_spacing_D
        mean_spacing = 0.0 if self.mean_spacing_D is None else self.mean_spacing_D
        return np.asarray(
            [
                float(self.n_turbines),
                *self.support_extent_D,
                float(self.support_volume_D3),
                spacing,
                mean_spacing,
                np.sin(direction),
                np.cos(direction),
            ],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "layout_index": self.layout_index,
            "case": self.case,
            "wind_direction_deg": self.wind_direction_deg,
            "n_turbines": self.n_turbines,
            "support_extent_D": list(self.support_extent_D),
            "support_volume_D3": self.support_volume_D3,
            "min_spacing_D": self.min_spacing_D,
            "mean_spacing_D": self.mean_spacing_D,
        }


def _spacing_D(centers_D: np.ndarray, present: np.ndarray) -> tuple[float | None, float | None]:
    """Return nearest-neighbour spacing in the layout's horizontal plane."""

    centers = np.asarray(centers_D, dtype=np.float64)
    mask = np.asarray(present, dtype=np.float64).reshape(-1) > 0.5
    if centers.ndim != 2 or centers.shape[-1] != 3 or mask.shape != (centers.shape[0],):
        raise ValueError("module centers/presence must have shapes [M,3] and [M]")
    active = centers[mask]
    if active.shape[0] < 2:
        return None, None
    delta = active[:, None, :2] - active[None, :, :2]
    distances = np.sqrt(np.sum(delta * delta, axis=-1))
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(np.min(nearest)), float(np.mean(nearest))


def geometry_record(view: WindFarmNativeView, row: int) -> GeometryRecord:
    """Describe one row without touching any solved field array."""

    case = view.run(int(row))
    min_spacing, mean_spacing = _spacing_D(case.module_centers, case.module_present)
    extent = tuple(float(value) for value in case.support.extent_D)
    return GeometryRecord(
        row=int(row),
        layout_index=int(case.layout_index),
        case=str(case.case),
        wind_direction_deg=float(case.wind_direction_deg),
        n_turbines=int(case.n_turbines),
        support_extent_D=extent,
        support_volume_D3=float(case.support.volume_D3),
        min_spacing_D=min_spacing,
        mean_spacing_D=mean_spacing,
    )


def _representative_layout_records(view: WindFarmNativeView, rows: Iterable[int]) -> list[GeometryRecord]:
    """Keep one deterministic wind direction per layout group."""

    records = [geometry_record(view, int(row)) for row in sorted({int(value) for value in rows})]
    if not records:
        return []
    groups: dict[int, list[GeometryRecord]] = {}
    for record in records:
        groups.setdefault(record.layout_index, []).append(record)
    representatives: list[GeometryRecord] = []
    for layout_index in sorted(groups):
        group = groups[layout_index]
        # A 270-degree row is preferred when present, then the nearest angle,
        # then source row.  This keeps one layout from dominating the sample
        # while preserving a stable direction choice.
        representative = min(
            group,
            key=lambda item: (
                abs(((item.wind_direction_deg - 270.0 + 180.0) % 360.0) - 180.0),
                item.wind_direction_deg,
                item.row,
            ),
        )
        representatives.append(representative)
    return representatives


def select_diverse_validation_rows(
    view: WindFarmNativeView,
    rows: Iterable[int],
    count: int = 3,
) -> list[int]:
    """Select distinct validation layouts by geometry only.

    The candidate set is collapsed to one row per ``layout_index`` before a
    deterministic farthest-point selection in robustly scaled geometry space.
    No target field, sampled error, checkpoint metric, or model output is read.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    candidates = _representative_layout_records(view, rows)
    if not candidates:
        return []
    if count >= len(candidates):
        return [record.row for record in sorted(candidates, key=lambda item: item.row)]
    descriptors = np.stack([record.descriptor() for record in candidates])
    low = np.nanpercentile(descriptors, 5.0, axis=0)
    high = np.nanpercentile(descriptors, 95.0, axis=0)
    scale = np.maximum(high - low, 1e-8)
    scaled = np.clip((descriptors - low) / scale, 0.0, 1.0)

    # The first and last complexity records anchor the range; subsequent
    # choices maximise their distance from the already selected layouts.
    complexity = np.column_stack((scaled[:, 0], scaled[:, 3], scaled[:, 4], scaled[:, 5]))
    order = sorted(range(len(candidates)), key=lambda index: (tuple(complexity[index]), candidates[index].row))
    selected = [order[0]]
    if count > 1:
        selected.append(order[-1])
    while len(selected) < count:
        available = [index for index in range(len(candidates)) if index not in selected]
        next_index = max(
            available,
            key=lambda index: (
                min(float(np.linalg.norm(scaled[index] - scaled[chosen])) for chosen in selected),
                -candidates[index].row,
            ),
        )
        selected.append(next_index)
    return [candidates[index].row for index in sorted(selected, key=lambda index: candidates[index].row)]


def _plane_prediction(
    model: Any,
    case: Any,
    prepared: Any,
    plane: Mapping[str, Any],
    device: torch.device,
    receiver_chunk_size: int,
) -> np.ndarray:
    """Decode one native plane in bounded chunks and return only that plane."""

    if receiver_chunk_size <= 0:
        raise ValueError("receiver_chunk_size must be positive")
    coords_D = np.asarray(plane["coords_D"], dtype=np.float32)
    query = case.geometry_for_queries(coords_D)
    prediction = np.empty((len(coords_D), 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(coords_D), receiver_chunk_size):
            stop = min(start + receiver_chunk_size, len(coords_D))
            coordinates = torch.from_numpy(query["query_xy"][start:stop][None]).to(device)
            features = torch.from_numpy(query["query_features"][start:stop][None]).to(device)
            values = model.predict_physical(
                prepared,
                coordinates,
                features,
                receiver_chunk_size=receiver_chunk_size,
            )
            prediction[start:stop] = values[0].detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Native plane prediction contains non-finite values")
    return prediction.reshape(np.asarray(plane["target"]).shape)


def _channel_limits(
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Use common physical limits for fields and common symmetric limits for errors."""

    field_limits: list[tuple[float, float]] = []
    error_limits: list[tuple[float, float]] = []
    for channel in range(3):
        values = [np.asarray(target)[..., channel]] + [np.asarray(value)[..., channel] for value in predictions.values()]
        finite = np.concatenate([value.reshape(-1) for value in values])
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            field_limits.append((-1.0, 1.0))
        elif channel == 0:
            low, high = float(np.min(finite)), float(np.max(finite))
            field_limits.append((low, high if high > low else low + 1e-8))
        else:
            extent = max(float(np.max(np.abs(finite))), 1e-8)
            field_limits.append((-extent, extent))
        errors = np.concatenate(
            [
                (np.asarray(value)[..., channel] - np.asarray(target)[..., channel]).reshape(-1)
                for value in predictions.values()
            ]
        )
        finite_errors = errors[np.isfinite(errors)]
        extent = max(float(np.max(np.abs(finite_errors))) if finite_errors.size else 0.0, 1e-8)
        error_limits.append((-extent, extent))
    return field_limits, error_limits


def render_comparison_plane(
    plane: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    hubs_D: np.ndarray,
    title: str,
    destination: str | Path,
) -> Path:
    """Render reference, both models, and both errors on one native plane."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = {"classic", "dense"}
    if set(predictions) != required:
        raise ValueError(f"predictions must contain exactly {sorted(required)}")
    target = np.asarray(plane["target"], dtype=np.float32)
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("native plane target must have shape [height,width,3]")
    checked = {name: np.asarray(value, dtype=np.float32) for name, value in predictions.items()}
    if any(value.shape != target.shape for value in checked.values()):
        raise ValueError("comparison predictions must have the native target plane shape")
    field_limits, error_limits = _channel_limits(target, checked)
    values_by_column = (
        ("Reference", target, False),
        ("Classic (Run 2102)", checked["classic"], False),
        ("Dense (Run 2103)", checked["dense"], False),
        ("Classic error", checked["classic"] - target, True),
        ("Dense error", checked["dense"] - target, True),
    )
    fig, axes = plt.subplots(3, 5, figsize=(22, 10), layout="constrained", squeeze=False)
    hubs = np.asarray(hubs_D, dtype=np.float64)
    first = str(plane["horizontal_axis"])
    second = str(plane["vertical_axis"])
    first_index, second_index = "xyz".index(first), "xyz".index(second)
    x_axis = np.asarray(plane["horizontal_D"], dtype=np.float64)
    y_axis = np.asarray(plane["vertical_D"], dtype=np.float64)
    for channel, label in enumerate(("Ux", "Uy", "Uz")):
        for column, (name, values, is_error) in enumerate(values_by_column):
            limits = error_limits[channel] if is_error else field_limits[channel]
            cmap = "RdBu_r" if channel or is_error else "viridis"
            artist = axes[channel, column].pcolormesh(
                x_axis,
                y_axis,
                values[..., channel],
                shading="nearest",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                rasterized=True,
            )
            if hubs.ndim == 2 and hubs.shape[-1] == 3 and hubs.size:
                axes[channel, column].scatter(
                    hubs[:, first_index],
                    hubs[:, second_index],
                    s=12,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.45,
                )
            axes[channel, column].set_aspect("equal", adjustable="box")
            axes[channel, column].set(
                xlabel=f"{first} / D",
                ylabel=f"{second} / D" if column == 0 else "",
                title=f"{name}: {label} [m/s]",
                xlim=(float(x_axis[0]), float(x_axis[-1])),
                ylim=(float(y_axis[0]), float(y_axis[-1])),
            )
            axes[channel, column].tick_params(labelsize=7)
            fig.colorbar(artist, ax=axes[channel, column], shrink=0.74, pad=0.015)
    fig.suptitle(
        f"{title}\nNative {plane['fixed_axis']} = {float(plane['actual_m']):.3f} m; "
        "shared physical scales within each velocity channel",
        fontsize=14,
    )
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, facecolor="white")
    plt.close(fig)
    return destination


def _latent_routing_summary(
    model: Any,
    case: Any,
    prepared: Any,
    plane: Mapping[str, Any],
    device: torch.device,
    receiver_chunk_size: int,
) -> dict[str, Any]:
    """Collect bounded Classic routing labels for a native plane."""

    summary: dict[str, Any] = {
        "available": False,
        "label": "latent organization; not physical wake topology",
        "architecture": getattr(model, "architecture", None),
    }
    if getattr(model, "architecture", None) != "legacy_honf":
        summary["reason"] = "Hyperedge routing is exposed only by the Classic legacy_honf checkpoint."
        return summary
    encoded = getattr(prepared, "encoded", None)
    if not isinstance(encoded, Mapping):
        summary["reason"] = "Classic prepared state does not expose a mapping of organizer tensors."
        return summary
    required = ("hyper_source_coords", "hyper_region_coords", "A_mh")
    if any(key not in encoded for key in required):
        summary["reason"] = f"Classic organizer state lacks required tensors: {required}."
        return summary

    coords_D = np.asarray(plane["coords_D"], dtype=np.float32)
    query = case.geometry_for_queries(coords_D)
    dominant = np.empty(len(coords_D), dtype=np.int64)
    entropy = np.empty(len(coords_D), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(coords_D), receiver_chunk_size):
            stop = min(start + receiver_chunk_size, len(coords_D))
            coordinates = torch.from_numpy(query["query_xy"][start:stop][None]).to(device)
            features = torch.from_numpy(query["query_features"][start:stop][None]).to(device)
            output = model.decode(
                prepared,
                coordinates,
                features,
                receiver_chunk_size=receiver_chunk_size,
                return_routing_maps=True,
            )
            edge = output.get("dominant_hyperedge")
            if edge is None:
                attention = output.get("query_hyper_attention")
                if attention is None:
                    summary["reason"] = "Classic decoder did not expose query hyperedge attention."
                    return summary
                edge = attention.argmax(dim=-1)
            entropy_value = output.get("hyper_attention_entropy_map")
            if entropy_value is None:
                attention = output.get("query_hyper_attention")
                entropy_value = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(dim=-1)
            dominant[start:stop] = edge[0].detach().cpu().numpy().astype(np.int64, copy=False)
            entropy[start:stop] = entropy_value[0].detach().cpu().numpy().astype(np.float32, copy=False)
    source = encoded["hyper_source_coords"].detach().cpu().numpy()[0]
    region = encoded["hyper_region_coords"].detach().cpu().numpy()[0]
    assignment = encoded["A_mh"].detach().cpu().numpy()[0]
    summary.update(
        {
            "available": True,
            "fixed_axis": plane["fixed_axis"],
            "actual_m": float(plane["actual_m"]),
            "grid_shape": list(np.asarray(plane["target"]).shape[:2]),
            "query_coords_D": coords_D.tolist(),
            "dominant_hyperedge": dominant.tolist(),
            "query_entropy": entropy.tolist(),
            "hyper_source_coords_D": source.tolist(),
            "hyper_region_coords_D": region.tolist(),
            "edge_module_mass": assignment.sum(axis=0).tolist(),
            "edge_count": int(assignment.shape[-1]),
        }
    )
    return summary


def render_latent_organization(summary: Mapping[str, Any], case: Any, destination: str | Path) -> Path | None:
    """Render Classic routing as a clearly labelled latent diagnostic."""

    if not summary.get("available"):
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = np.asarray(summary["query_coords_D"], dtype=np.float64)
    dominant = np.asarray(summary["dominant_hyperedge"], dtype=np.int64)
    source = np.asarray(summary["hyper_source_coords_D"], dtype=np.float64)
    region = np.asarray(summary["hyper_region_coords_D"], dtype=np.float64)
    fixed_axis = str(summary["fixed_axis"])
    first, second = ("x", "y") if fixed_axis == "z" else (("x", "z") if fixed_axis == "y" else ("y", "z"))
    first_index, second_index = "xyz".index(first), "xyz".index(second)
    fig = plt.figure(figsize=(15, 6.5), layout="constrained")
    planar = fig.add_subplot(1, 2, 1)
    edge_count = max(int(summary.get("edge_count", 1)), 1)
    scatter = planar.scatter(
        coords[:, first_index],
        coords[:, second_index],
        c=dominant,
        cmap="tab10",
        vmin=-0.5,
        vmax=edge_count - 0.5,
        s=7,
        rasterized=True,
    )
    hubs = np.asarray(case.module_centers, dtype=np.float64)
    planar.scatter(hubs[:, first_index], hubs[:, second_index], marker="x", c="black", s=22, linewidths=0.8)
    planar.set_aspect("equal", adjustable="box")
    planar.set(
        xlabel=f"{first} / D",
        ylabel=f"{second} / D",
        title=(f"Dominant latent hyperedge on native {fixed_axis} slice\n"
               "model routing label; not a physical wake map"),
    )
    fig.colorbar(scatter, ax=planar, ticks=np.arange(edge_count), label="Latent hyperedge index")

    spatial = fig.add_subplot(1, 2, 2, projection="3d")
    spatial.set_box_aspect(case.support.extent_D)
    colors = plt.get_cmap("tab10")
    for edge, (start, stop) in enumerate(zip(source, region)):
        color = colors(edge % 10)
        spatial.scatter(*start, c=[color], marker="o", s=48)
        spatial.scatter(*stop, c=[color], marker="^", s=48)
        spatial.plot(*np.stack((start, stop)).T, c=color, alpha=0.75, linewidth=1.0)
    spatial.scatter(*hubs.T, c="black", marker="x", s=18)
    spatial.set(xlabel="x / D", ylabel="y / D", zlabel="z / D", title="Learned source/region centroids")
    fig.suptitle(
        f"{case.case}: Classic latent organization diagnostic — no physical wake-topology claim",
        fontsize=13,
    )
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, facecolor="white")
    plt.close(fig)
    return destination


def run_comparison(
    *,
    classic_checkpoint: str | Path,
    dense_checkpoint: str | Path,
    volume_path: str | Path,
    derived_view: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str | torch.device = "cpu",
    layout_count: int = 3,
    fixed_axes: Sequence[str] = ("z", "y", "x"),
    receiver_chunk_size: int = 512,
    compact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the Run 2102 versus Run 2103 native comparison."""

    target_device = torch.device(device)
    if receiver_chunk_size <= 0:
        raise ValueError("receiver_chunk_size must be positive")
    axes = tuple(str(axis) for axis in fixed_axes)
    if not axes or any(axis not in {"x", "y", "z"} for axis in axes):
        raise ValueError("fixed_axes must contain one or more of x, y, z")
    view = WindFarmNativeView(
        volume_path,
        compact_metadata=None if compact_path is None else _compact_metadata(compact_path),
    )
    split = _load_split(view, Path(derived_view).expanduser().resolve())
    rows = select_diverse_validation_rows(view, split.validation, count=layout_count)
    if not rows:
        raise ValueError("No validation rows are available for comparison")
    records = [geometry_record(view, row) for row in rows]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "geometry_selection.json", {"rows": [record.as_dict() for record in records]})

    materialization_case = view.run(int(split.train[0]))
    materialization_batch = case_batch(materialization_case, materialization_case.module_centers)
    checkpoints = {"classic": Path(classic_checkpoint).expanduser().resolve(), "dense": Path(dense_checkpoint).expanduser().resolve()}
    models: dict[str, Any] = {}
    payloads: dict[str, Mapping[str, Any]] = {}
    for label, checkpoint in checkpoints.items():
        model, payload = load_checkpoint(
            checkpoint,
            device=target_device,
            materialization_batch=materialization_batch,
        )
        models[label] = model
        payloads[label] = payload
    if models["classic"].architecture != "legacy_honf":
        raise ValueError("classic_checkpoint does not load as legacy_honf")
    if models["dense"].architecture != "dense_pairwise_field":
        raise ValueError("dense_checkpoint does not load as dense_pairwise_field")

    cases: list[dict[str, Any]] = []
    try:
        for row, record in zip(rows, records):
            case = view.run(row)
            initial = case_batch(case, case.module_centers).to(target_device)
            prepared: dict[str, Any] = {}
            for label, model in models.items():
                prepared[label] = model.prepare_case(initial)
            case_output: dict[str, Any] = {"geometry": record.as_dict(), "planes": [], "latent_organization": None}
            for axis in axes:
                requested = DEFAULT_HUB_HEIGHT_M if axis == "z" else 0.0
                plane = native_plane(case.run, axis, requested, diameter=case.diameter_m)
                predictions = {
                    label: _plane_prediction(model, case, prepared[label], plane, target_device, receiver_chunk_size)
                    for label, model in models.items()
                }
                figure = output / f"layout_{case.layout_index:04d}_{case.case}_native_{axis}_comparison.png"
                render_comparison_plane(
                    plane,
                    predictions,
                    case.module_centers,
                    f"WindFarm native velocity comparison — {case.case} | layout {case.layout_index}",
                    figure,
                )
                case_output["planes"].append(
                    {
                        "fixed_axis": axis,
                        "requested_m": requested,
                        "actual_m": float(plane["actual_m"]),
                        "shape": list(np.asarray(plane["target"]).shape),
                        "figure": str(figure),
                    }
                )
                if axis == "z":
                    latent_summary = _latent_routing_summary(
                        models["classic"],
                        case,
                        prepared["classic"],
                        plane,
                        target_device,
                        receiver_chunk_size,
                    )
                    latent_figure = output / f"layout_{case.layout_index:04d}_{case.case}_latent_organization.png"
                    latent_path = render_latent_organization(latent_summary, case, latent_figure)
                    if latent_path is not None:
                        latent_summary = dict(latent_summary)
                        latent_summary["figure"] = str(latent_path)
                    else:
                        latent_summary = dict(latent_summary)
                    # Keep the machine-readable record bounded: the figure is
                    # the native-map artifact; JSON retains no full prediction.
                    case_output["latent_organization"] = {
                        key: value
                        for key, value in latent_summary.items()
                        if key not in {"query_coords_D", "dominant_hyperedge", "query_entropy"}
                    }
            atomic_write_json(output / f"layout_{case.layout_index:04d}_{case.case}_comparison.json", case_output)
            cases.append(case_output)
            del prepared, initial
    finally:
        del models
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "workflow": "windfarm_native_run2102_vs_run2103",
        "split": "validation",
        "selected_rows": rows,
        "selected_layouts": [record.layout_index for record in records],
        "selection_policy": "geometry-only one-row-per-layout deterministic farthest-point selection",
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "checkpoint_epochs": {label: payloads[label].get("epoch") for label in payloads},
        "device": str(target_device),
        "receiver_chunk_size": int(receiver_chunk_size),
        "fixed_axes": list(axes),
        "native_only": True,
        "full_volume_prediction_saved": False,
        "latent_label": "latent organization; not physical wake topology",
        "cases": cases,
    }
    atomic_write_json(output / "comparison.json", result)
    return result


def _parse_axes(value: str) -> tuple[str, ...]:
    axes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not axes:
        raise argparse.ArgumentTypeError("expected comma-separated axes such as z,y,x")
    return axes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classic-checkpoint", required=True)
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--volume-path", required=True)
    parser.add_argument("--derived-view", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layout-count", type=int, default=3)
    parser.add_argument("--fixed-axes", type=_parse_axes, default=("z", "y", "x"))
    parser.add_argument("--receiver-chunk-size", type=int, default=512)
    parser.add_argument("--compact-path")
    args = parser.parse_args(argv)
    result = run_comparison(
        classic_checkpoint=args.classic_checkpoint,
        dense_checkpoint=args.dense_checkpoint,
        volume_path=args.volume_path,
        derived_view=args.derived_view,
        output_dir=args.output_dir,
        device=args.device,
        layout_count=args.layout_count,
        fixed_axes=args.fixed_axes,
        receiver_chunk_size=args.receiver_chunk_size,
        compact_path=args.compact_path,
    )
    print(json.dumps({"output_dir": str(Path(args.output_dir).expanduser().resolve()), "cases": len(result["cases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GeometryRecord",
    "geometry_record",
    "render_comparison_plane",
    "render_latent_organization",
    "run_comparison",
    "select_diverse_validation_rows",
]
