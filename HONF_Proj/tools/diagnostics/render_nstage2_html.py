#!/usr/bin/env python3
"""Render the bounded NStage2 comparison as one offline HTML page.

The page consumes the reducer's CSV/JSON outputs and any already-produced
debug/timing/intervention artifacts.  It does not run models, copy checkpoints,
or invent panels when a candidate artifact has not arrived yet.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Keep the established renderer's Plotly/layout/array handling as the only
# presentation dependency.  This file only adapts the NStage2 artifact names.
from nstage2_reduction import ANCHORS, RUN_LABELS, RUN_TRACK, _resolve_roots  # type: ignore[import-not-found]
from render_honf_maturity_html import (  # type: ignore[import-not-found]
    array_grid,
    bar_trace,
    escape,
    finite,
    js,
    line_trace,
    plot,
    plotly_source,
    read_csv,
    read_json,
    rounded_grid,
    safe_float,
    slug,
)

__all__ = ["render_nstage2"]

CHANNELS = ("u", "v", "p", "omega", "temperature")
CHANNEL_LABELS = {"u": "u velocity", "v": "v velocity", "p": "pressure", "omega": "vorticity", "temperature": "temperature"}
MODEL_ORDER = ("A", "Regional", "Dense", "B", "Reader")
MODEL_RUNS = {"A": "1807", "Regional": "1806", "Dense": "1804", "B": "1808", "Reader": "1805"}
MODEL_LABELS = {
    "A": RUN_LABELS["1807"],
    "Regional": RUN_LABELS["1806"],
    "Dense": RUN_LABELS["1804"],
    "B": RUN_LABELS["1808"],
    "Reader": RUN_LABELS["1805"],
}
HISTORY_ORDER = ("Legacy", "Latent", "Dense", "Reader", "Regional", "A", "B")
RUN_DISPLAY = {"1401": "Legacy", "1801": "Latent", **{run: model for model, run in MODEL_RUNS.items()}}
MODEL_COLORS = {
    "Legacy": "#53606e",
    "Latent": "#2f6f9f",
    "A": "#c05a3f",
    "Regional": "#39826a",
    "Dense": "#bf6b3c",
    "B": "#315f7c",
    "Reader": "#7c5aa6",
}
TIMING_PHASES = (
    "full_forward",
    "physical_preparation_plus_one_query",
    "prepared_decode",
    "encoding_plus_layout_construction",
)
TIMING_PHASE_LABELS = {
    "full_forward": "full forward",
    "physical_preparation_plus_one_query": "physical preparation + one query",
    "prepared_decode": "prepared decode",
    "encoding_plus_layout_construction": "encoding + layout construction",
}
TIMING_REAL_CASES = ("0273", "0653")
TIMING_SYNTHETIC_WORKLOADS = (
    ("synthetic:E=768 M=32 Q=65536", "synthetic E=768 M=32 Q=65536"),
    ("synthetic:E=3072 M=128 Q=262144", "synthetic E=3072 M=128 Q=262144"),
)
TIMING_ARCHITECTURE_LABELS = {
    "legacy_honf": "Legacy",
    "geometry_latent_field": "Latent",
    "dense_pairwise_field": "Dense",
    "sparse_interface_honf": "Reader",
    "regional_response_honf": "Regional",
    "hierarchical_regional_honf": "A",
    "group_mediated_reader": "B",
}

OLD_DEBUG_ROOTS = {
    "Dense": (
        "Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage2_Run1401_1804_1801_1802_Epoch500_20Case/debug_npz",
    ),
    "Reader": ("diagnostics/generated/interface_operator_study/group_reader_recovery/endpoint/debug_npz",),
    "Regional": ("diagnostics/generated/interface_operator_study/regional_response/endpoint500/debug_npz",),
}
MISSING_DEBUG_ROOTS = {
    "Dense": "comparison/missing_dense_anchors/debug_npz",
    "Reader": "comparison/missing_parent_anchors/debug_npz",
    "Regional": "comparison/missing_parent_anchors/debug_npz",
}
OLD_DEBUG_PREFIXES = {
    "Dense": ("Dense_pairwise_adaptation", "Dense_1804", "Run_1804"),
    "Reader": ("Geometry-envelope_sparse_HONF", "Reader_1805", "Run_1805"),
    "Regional": ("Regional_response_HONF", "Regional_1806", "Run_1806"),
}


def _comparison_files(comparison: Path) -> dict[str, list[dict[str, str]]]:
    names = {
        "headline": "exact500_headline.csv",
        "channels": "exact500_channels.csv",
        "physical": "exact500_physical.csv",
        "kpis": "exact500_engineering_kpis.csv",
        "pairs": "exact500_pairs.csv",
        "pair_summary": "exact500_pair_summary.csv",
        "best": "best_field_headline.csv",
        "history": "learning_curves.csv",
    }
    return {key: read_csv(comparison / name) for key, name in names.items()}


def _display_model(row: dict[str, Any]) -> str:
    run = str(row.get("run", ""))
    if run in RUN_DISPLAY:
        return RUN_DISPLAY[run]
    for model, candidate_run in MODEL_RUNS.items():
        if run == candidate_run:
            return model
    value = str(row.get("model", ""))
    for model in MODEL_ORDER:
        if value.startswith(model) or model in value:
            return model
    return value or run


def _label_for_run(run: str) -> str:
    return RUN_DISPLAY.get(str(run), str(run))


def _available_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("status", "available")) == "available"]


def _headline_plot(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    if not rows:
        return None
    labels = [_display_model(row) for row in rows]
    traces = []
    for field, name, color in (
        ("global_field_fluid_norm_pooled_relative_l2", "pooled fluid relative L2", "#315f7c"),
        ("global_field_fluid_norm_equal_case_mean", "equal-case mean", "#d38b54"),
        ("global_field_fluid_norm_equal_case_p95", "equal-case p95", "#7c5aa6"),
    ):
        values = [safe_float(row, field) for row in rows]
        if any(value is not None for value in values):
            traces.append(bar_trace(labels, values, name, color))
    return plot(
        "NStage2 exact epoch 500 endpoint across the matched 90 cases",
        traces,
        height=450,
        barmode="group",
        xaxis={"title": "model / track"},
        yaxis={"title": "reported error", "gridcolor": "#e8edf1"},
    ) if traces else None


def _channel_plot(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    rows = _available_rows(rows)
    if not rows:
        return None
    traces = []
    for channel, color in zip(CHANNELS, ("#315f7c", "#bf6b3c", "#7c5aa6", "#39826a", "#a35a63"), strict=True):
        metric = f"field_{channel}_fluid_norm"
        subset = [row for row in rows if row.get("metric") == metric]
        if not subset:
            continue
        labels = [_display_model(row) for row in subset]
        values = [safe_float(row, "relative_l2") for row in subset]
        if any(value is not None for value in values):
            traces.append(bar_trace(labels, values, CHANNEL_LABELS[channel], color, offsetgroup=channel))
    return plot(
        "Exact endpoint normalized channel relative L2",
        traces,
        height=470,
        barmode="group",
        xaxis={"title": "model / track"},
        yaxis={"title": "pooled relative L2", "gridcolor": "#e8edf1"},
    ) if traces else None


def _physical_plot(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    rows = _available_rows(rows)
    labels = (
        ("global_field_fluid_norm", "fluid field"),
        ("global_field_near_interface_norm", "near interface"),
        ("global_field_far_fluid_norm", "far fluid"),
        ("internal_temperature_physical", "internal T"),
        ("port_t_env_final_physical", "port T final"),
        ("port_t_env_provisional_physical", "port T provisional"),
        ("port_h_effective_final_physical", "port h final"),
        ("port_h_effective_provisional_physical", "port h provisional"),
    )
    traces = []
    for metric, label in labels:
        subset = [row for row in rows if row.get("metric") == metric]
        values = [safe_float(row, "relative_l2") for row in subset]
        if subset and any(value is not None for value in values):
            traces.append(bar_trace([_display_model(row) for row in subset], values, label, MODEL_COLORS.get(_display_model(subset[0]), "#315f7c"), offsetgroup=metric))
    return plot(
        "Exact endpoint near / far / physical field outcomes",
        traces,
        height=460,
        barmode="group",
        xaxis={"title": "model / track"},
        yaxis={"title": "pooled relative L2", "gridcolor": "#e8edf1"},
    ) if traces else None


def _pair_plots(rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    metric = "global_field_fluid_norm"
    case_groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("metric") != metric or row.get("status") != "available":
            continue
        value = safe_float(row, "delta")
        if value is None:
            continue
        pair = str(row.get("pair", ""))
        case_groups[pair].append(value)
    box_traces = []
    for pair, values in case_groups.items():
        box_traces.append({"type": "box", "name": pair, "y": values, "boxmean": True, "marker": {"color": MODEL_COLORS.get("A" if pair.startswith("A") else "B", "#315f7c")}})
    case_plot = plot(
        "Exact endpoint paired per-case deltas (candidate minus baseline)",
        box_traces,
        height=430,
        showlegend=False,
        yaxis={"title": "candidate − baseline relative L2", "zeroline": True, "gridcolor": "#e8edf1"},
        xaxis={"title": "requested matched pair"},
    ) if box_traces else None
    summary = []
    for row in summary_rows:
        if row.get("metric") != metric or row.get("status") != "available":
            continue
        value = safe_float(row, "mean_delta")
        if value is not None:
            summary.append((str(row.get("pair", "")), value))
    summary_plot = plot(
        "Exact endpoint paired mean deltas",
        [bar_trace([item[0] for item in summary], [item[1] for item in summary], "mean candidate − baseline", "#315f7c")],
        height=410,
        showlegend=False,
        yaxis={"title": "mean candidate − baseline", "zeroline": True, "gridcolor": "#e8edf1"},
        xaxis={"title": "requested matched pair"},
    ) if summary else None
    return case_plot, summary_plot


def _kpi_label(metric: str) -> str:
    return (
        metric.replace("_physical", "")
        .replace("pressure_drop_inlet_minus_outlet", "pressure drop")
        .replace("mean_outlet_temperature", "outlet T")
        .replace("mean_active_module_temperature", "active module T")
        .replace("_relative_error", " rel")
        .replace("_abs_error", " abs")
        .replace("_error", " signed")
    )


def _kpi_plot(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    rows = _available_rows(rows)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("metric"):
            groups[row["metric"]].append(row)
    if not groups:
        return None
    traces = []
    for metric, subset in groups.items():
        values = [safe_float(row, "mean") for row in subset]
        if any(value is not None for value in values):
            traces.append(bar_trace([_display_model(row) for row in subset], values, _kpi_label(metric), MODEL_COLORS.get(_display_model(subset[0]), "#315f7c"), offsetgroup=metric))
    return plot(
        "Exact endpoint engineering KPI error distributions",
        traces,
        height=520,
        barmode="group",
        xaxis={"title": "model / track"},
        yaxis={"title": "equal-case mean of evaluator error column", "gridcolor": "#e8edf1"},
    ) if traces else None


def _history_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        epoch = finite(row.get("epoch"))
        run = str(row.get("run", ""))
        if epoch is None or not run:
            continue
        grouped[run].append(row)
    output: list[dict[str, Any]] = []
    for run, items in grouped.items():
        items.sort(key=lambda row: int(float(row["epoch"])))
        cumulative = 0.0
        for row in items:
            train = safe_float(row, "train_wall_seconds", "train_seconds") or 0.0
            validation = safe_float(row, "val_wall_seconds", "validation_wall_seconds", "val_seconds") or 0.0
            cumulative += train + validation
            output.append({
                "run": run,
                "model": _label_for_run(run),
                "epoch": int(float(row["epoch"])),
                "logged_active_seconds": cumulative,
                "val_field_mse": safe_float(row, "val_field_mse", "validation_field_mse"),
                "val_temperature_mse": safe_float(row, "val_temperature_mse", "validation_temperature_mse"),
            })
    return sorted(output, key=lambda row: (HISTORY_ORDER.index(row["model"]) if row["model"] in HISTORY_ORDER else 99, row["epoch"]))


def _history_plot(rows: list[dict[str, Any]], field: str, x_field: str, title: str, x_title: str) -> dict[str, Any] | None:
    traces = []
    for model in HISTORY_ORDER:
        subset = [row for row in rows if row["model"] == model and row.get(field) is not None and row.get(x_field) is not None]
        if subset:
            subset.sort(key=lambda row: row[x_field])
            traces.append(line_trace([row[x_field] for row in subset], [row[field] for row in subset], model, MODEL_COLORS.get(model, "#315f7c")))
    return plot(title, traces, height=430, xaxis={"title": x_title}, yaxis={"title": field, "type": "log", "gridcolor": "#e8edf1"}) if traces else None


def _npz_candidates(project: Path, study: Path, model: str, case_id: str) -> list[Path]:
    """Return only sources whose run/model ownership is known.

    The old comparison directories contain several architectures in one debug
    folder.  A case substring is therefore not enough to identify a source;
    matching is restricted to the maintained filename prefixes below.  More
    than one surviving path is deliberately returned so the caller can report
    an ambiguous source instead of silently choosing alphabetically.
    """

    run = MODEL_RUNS[model]
    paths: list[Path] = []
    if model in {"A", "B"}:
        root = study / RUN_TRACK[run] / "endpoint500" / "debug_npz"
        if root.is_dir():
            paths.extend(sorted(root.rglob("*.npz")))
        token = str(case_id)
        paths = [path for path in paths if token in path.stem]
        # The managed run directory is the ownership boundary.  If a future
        # export includes several copies, an explicit Run_<id> filename wins;
        # otherwise the ambiguity is retained for the caller to report.
        explicit_run = [path for path in paths if re.search(r"Run[_-]?\d+", path.name, re.IGNORECASE)]
        if explicit_run:
            paths = [path for path in explicit_run if re.search(rf"Run[_-]?{run}(?:[_-]|$)", path.name, re.IGNORECASE)]
        return sorted(dict.fromkeys(paths))

    if model not in OLD_DEBUG_ROOTS:
        return []
    roots = [project / relative for relative in OLD_DEBUG_ROOTS[model]]
    roots.append(study / MISSING_DEBUG_ROOTS[model])
    prefixes = OLD_DEBUG_PREFIXES[model]
    token = f"__{case_id}.npz"
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.npz")):
            # Exact suffix avoids 0273 matching 10273, while the prefix keeps
            # Dense/Reader/Regional files from crossing architecture families.
            if not path.name.endswith(token):
                continue
            if not any(path.name.startswith(prefix + "__") for prefix in prefixes):
                continue
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def _relative(path: Path, project: Path) -> str:
    try:
        return str(path.resolve().relative_to(project.resolve()))
    except ValueError:
        return str(path)


def _grid_shape(raw_x: np.ndarray, raw_y: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    if raw_x.ndim == 2:
        return tuple(int(value) for value in raw_x.shape)
    if raw_y.ndim == 2:
        return tuple(int(value) for value in raw_y.shape)
    if raw_x.ndim == 1 and raw_y.ndim == 1 and raw_x.size * raw_y.size == gt[..., 0].size:
        return (int(raw_y.size), int(raw_x.size))
    return tuple(int(value) for value in gt.shape[:2])


def _canonical_mask_for_case(
    project: Path,
    study: Path,
    case_id: str,
) -> dict[str, Any] | None:
    """Load one shared stored mask from aligned Reader/Regional targets."""

    candidates: list[Path] = []
    for model in ("Reader", "Regional"):
        candidates.extend(_npz_candidates(project, study, model, case_id))
    candidates = sorted(dict.fromkeys(candidates))
    if not candidates:
        return None
    reference: dict[str, Any] | None = None
    for path in candidates:
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {"fluid_mask", "x_grid", "y_grid", "gt_field_grid"}
                if not required.issubset(data.files):
                    continue
                gt = np.asarray(data["gt_field_grid"], dtype=float)
                raw_x = np.asarray(data["x_grid"], dtype=float)
                raw_y = np.asarray(data["y_grid"], dtype=float)
                shape = _grid_shape(raw_x, raw_y, gt)
                x = array_grid(raw_x, shape)
                y = array_grid(raw_y, shape)
                mask = array_grid(np.asarray(data["fluid_mask"], dtype=bool), shape).astype(bool)
                candidate = {"x": x, "y": y, "gt": gt, "mask": mask, "source": path}
                if reference is None:
                    reference = candidate
                    continue
                if (
                    reference["x"].shape != x.shape
                    or reference["y"].shape != y.shape
                    or reference["gt"].shape != gt.shape
                    or not np.array_equal(reference["x"], x)
                    or not np.array_equal(reference["y"], y)
                    or not np.array_equal(reference["gt"], gt)
                    or not np.array_equal(reference["mask"], mask)
                ):
                    # Two canonical parent sources disagree.  A shared mask
                    # would hide a source error, so make the Dense panel
                    # unavailable rather than selecting one by filename order.
                    return None
        except (OSError, KeyError, ValueError):
            continue
    return reference


def _canonical_mask_aligned(
    canonical: dict[str, Any] | None,
    x: np.ndarray,
    y: np.ndarray,
    gt: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if canonical is None:
        return None
    target_x = array_grid(np.asarray(canonical["x"], dtype=float), shape)
    target_y = array_grid(np.asarray(canonical["y"], dtype=float), shape)
    target_gt = np.asarray(canonical["gt"], dtype=float)
    if target_gt.shape != gt.shape:
        return None
    if not np.array_equal(x, target_x) or not np.array_equal(y, target_y) or not np.array_equal(gt, target_gt):
        return None
    return array_grid(np.asarray(canonical["mask"], dtype=bool), shape)


def _map_entry(
    path: Path,
    project: Path,
    *,
    model: str,
    canonical_mask: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    record: dict[str, Any] = {"source": _relative(path, project), "keys": [], "status": "unavailable"}
    try:
        with np.load(path, allow_pickle=False) as data:
            record["keys"] = list(data.files)
            gt = np.asarray(data["gt_field_grid"], dtype=float)
            pred = np.asarray(data["pred_field_grid"], dtype=float)
            if gt.ndim != 3 or pred.shape != gt.shape or gt.shape[-1] < len(CHANNELS):
                record["reason"] = f"field arrays have unsupported shapes: gt={gt.shape}, pred={pred.shape}"
                return None, record
            raw_x = np.asarray(data["x_grid"], dtype=float)
            raw_y = np.asarray(data["y_grid"], dtype=float)
            shape = _grid_shape(raw_x, raw_y, gt)
            x = array_grid(raw_x, shape)
            y = array_grid(raw_y, shape)
            canonical = _canonical_mask_aligned(canonical_mask, x, y, gt, shape)
            if canonical is not None:
                mask = canonical
                source = canonical_mask["source"] if canonical_mask is not None else path
                record["mask_source"] = f"canonical stored fluid_mask from {_relative(source, project)} (shared across aligned models)"
            elif "fluid_mask" in data.files:
                mask = np.asarray(data["fluid_mask"], dtype=bool)
                record["mask_source"] = "stored fluid_mask"
            elif {"module_centers", "module_radius"}.issubset(data.files):
                centers = np.asarray(data["module_centers"], dtype=float).reshape(-1, 2)
                present = np.asarray(data["module_present"], dtype=bool).reshape(-1) if "module_present" in data.files else np.ones(len(centers), dtype=bool)
                radius = float(np.asarray(data["module_radius"], dtype=float).reshape(-1)[0])
                mask = np.ones(shape, dtype=bool)
                for center, active in zip(centers, present, strict=False):
                    if active:
                        mask &= (x - center[0]) ** 2 + (y - center[1]) ** 2 >= radius**2
                record["mask_source"] = "derived from stored module_centers/module_radius"
            else:
                # Some maintained Dense Stage2 exports predate the explicit
                # fluid mask.  Preserve their complete stored grid and label
                # the absence rather than fabricating a solid-cell geometry.
                mask = np.ones(shape, dtype=bool)
                record["mask_source"] = "no fluid_mask in source; complete stored grid shown"
            mask_grid = array_grid(mask, shape)
            entry = {
                "source": record["source"],
                "x": [float(value) if math.isfinite(float(value)) else None for value in x[0]],
                "y": [float(value) if math.isfinite(float(value)) else None for value in y[:, 0]],
                "fluid_mask": mask_grid.astype(int).tolist(),
                "reference": [rounded_grid(array_grid(gt[:, :, index], shape), 5) for index in range(len(CHANNELS))],
                "prediction": [rounded_grid(array_grid(pred[:, :, index], shape), 5) for index in range(len(CHANNELS))],
                "error": [rounded_grid(array_grid(pred[:, :, index] - gt[:, :, index], shape), 5) for index in range(len(CHANNELS))],
            }
            record["status"] = "available"
            return entry, record
    except (OSError, KeyError, ValueError) as exc:
        record["reason"] = str(exc)
        return None, record


def _finite_values(data: Any, limit: int = 6000) -> list[float]:
    values = np.asarray(data, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size > limit:
        values = values[np.linspace(0, values.size - 1, limit, dtype=int)]
    return values.astype(float).tolist()


def _array_payload(data: Any, limit: int = 20000) -> Any:
    """Serialize a bounded ndarray while preserving its shape."""

    values = np.asarray(data)
    if values.size > limit:
        flat = values.reshape(-1)
        indices = np.linspace(0, flat.size - 1, limit, dtype=int)
        return {"shape": [int(value) for value in values.shape], "sample": flat[indices].tolist()}
    return values.tolist()


def _first_batch_array(data: Any, *, expected_rank: int | None = None) -> np.ndarray:
    values = np.asarray(data)
    if expected_rank is not None and values.ndim == expected_rank + 1 and values.shape[0] == 1:
        values = values[0]
    return values


def _organization_record(path: Path, project: Path, model: str, case_id: str) -> dict[str, Any]:
    record: dict[str, Any] = {"model": model, "case_id": case_id, "source": _relative(path, project), "status": "unavailable"}
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.files)
            record["keys"] = keys
            if model == "A":
                required = (
                    "tree_levels",
                    "tree_children",
                    "tree_coords",
                    "tree_bounds_min",
                    "tree_bounds_max",
                    "tree_states",
                    "tree_mass",
                    "tree_valid",
                )
                missing = [key for key in required if key not in data.files]
                incidence_keys = sorted(key for key in keys if key.startswith("interaction__hierarchical_incidence_"))
                if missing:
                    record["reason"] = f"A hierarchy export lacks exact keys: {missing}"
                    return record
                states = _first_batch_array(data["tree_states"], expected_rank=2)
                state_norm = np.linalg.norm(np.asarray(states, dtype=float), axis=-1) if states.ndim >= 2 else np.asarray(states, dtype=float)
                tree_levels = _first_batch_array(data["tree_levels"], expected_rank=1)
                tree_children = _first_batch_array(data["tree_children"], expected_rank=2)
                tree_coords = _first_batch_array(data["tree_coords"], expected_rank=2)
                tree_bounds_min = _first_batch_array(data["tree_bounds_min"], expected_rank=2)
                tree_bounds_max = _first_batch_array(data["tree_bounds_max"], expected_rank=2)
                tree_mass = _first_batch_array(data["tree_mass"], expected_rank=1)
                tree_valid = _first_batch_array(data["tree_valid"], expected_rank=1)
                node_count = int(tree_levels.shape[0]) if tree_levels.ndim == 1 else -1
                shape_errors: list[str] = []
                if tree_coords.ndim != 2 or tree_coords.shape != (node_count, 2):
                    shape_errors.append(f"tree_coords={tree_coords.shape}, expected=({node_count}, 2)")
                if tree_children.ndim != 2 or tree_children.shape[0] != node_count:
                    shape_errors.append(f"tree_children={tree_children.shape}, expected first dimension {node_count}")
                if tree_bounds_min.ndim != 2 or tree_bounds_min.shape != (node_count, 2):
                    shape_errors.append(f"tree_bounds_min={tree_bounds_min.shape}, expected=({node_count}, 2)")
                if tree_bounds_max.ndim != 2 or tree_bounds_max.shape != (node_count, 2):
                    shape_errors.append(f"tree_bounds_max={tree_bounds_max.shape}, expected=({node_count}, 2)")
                if tree_mass.ndim != 1 or tree_mass.shape != (node_count,):
                    shape_errors.append(f"tree_mass={tree_mass.shape}, expected=({node_count},)")
                if tree_valid.ndim != 1 or tree_valid.shape != (node_count,):
                    shape_errors.append(f"tree_valid={tree_valid.shape}, expected=({node_count},)")
                if states.ndim != 2 or states.shape[0] != node_count:
                    shape_errors.append(f"tree_states={states.shape}, expected first dimension {node_count}")
                if shape_errors:
                    record["reason"] = "A hierarchy export has misaligned exact arrays: " + "; ".join(shape_errors)
                    return record
                record.update(
                    {
                        "family": "A hierarchical regional tree",
                        "tree_levels": _array_payload(tree_levels),
                        "tree_children": _array_payload(tree_children),
                        "tree_coords": _array_payload(tree_coords),
                        "tree_bounds_min": _array_payload(tree_bounds_min),
                        "tree_bounds_max": _array_payload(tree_bounds_max),
                        "tree_mass": _array_payload(tree_mass),
                        "tree_valid": _array_payload(tree_valid),
                        "tree_response_norm": _array_payload(state_norm),
                        "incidence_keys": incidence_keys,
                        "incidence": {key: _array_payload(data[key]) for key in incidence_keys},
                    }
                )
                if not incidence_keys:
                    record["incidence_note"] = "no exact interaction__hierarchical_incidence_* keys in source"
            elif model == "B":
                support_key = "interaction__support_centres"
                state_key = "interaction__group_state_norm"
                attention_key = "interaction__group_read_max_weight"
                if support_key not in data.files:
                    record["reason"] = f"B source lacks exact {support_key}"
                    return record
                centers = np.asarray(data[support_key], dtype=float)
                if centers.ndim != 2 or centers.shape[-1] != 2:
                    record["reason"] = f"B source has {support_key}={centers.shape}, expected (group_count, 2)"
                    return record
                states_array = np.asarray(data[state_key], dtype=float).reshape(-1) if state_key in data.files else np.empty((0,), dtype=float)
                if states_array.size and states_array.shape != (len(centers),):
                    record["reason"] = f"B source has {state_key}={states_array.shape}, expected ({len(centers)},)"
                    return record
                states = states_array.astype(float).tolist()
                topology_keys = (
                    "interaction__support_group_batch",
                    "interaction__support_case_group_offsets",
                    "interaction__module_group_indices",
                    "interaction__environment_group_indices",
                    "interaction__initial_port_group_read_group_index",
                )
                topology = {
                    key: _array_payload(data[key])
                    for key in topology_keys
                    if key in data.files
                }
                record.update({
                    "family": "B group support/state/read",
                    "support_key": support_key,
                    "state_key": state_key,
                    "attention_key": attention_key,
                    "support_x": centers[:, 0].astype(float).tolist(),
                    "support_y": centers[:, 1].astype(float).tolist(),
                    "state": states,
                    "attention": _finite_values(data[attention_key]) if attention_key in data.files else [],
                    "topology": topology,
                })
            record["status"] = "available"
    except (OSError, KeyError, ValueError) as exc:
        record["reason"] = str(exc)
    return record


def _load_anchor_data(project: Path, study: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data: dict[str, Any] = {"cases": list(ANCHORS), "models": list(MODEL_ORDER), "channels": list(CHANNELS), "entries": {}, "organization": []}
    statuses: list[dict[str, Any]] = []
    for case_id in ANCHORS:
        data["entries"][case_id] = {}
        canonical_mask = _canonical_mask_for_case(project, study, case_id)
        for model in MODEL_ORDER:
            candidates = _npz_candidates(project, study, model, case_id)
            if not candidates:
                statuses.append({"model": model, "case_id": case_id, "status": "unavailable", "reason": "matching debug NPZ has not arrived"})
                continue
            if len(candidates) != 1:
                statuses.append(
                    {
                        "model": model,
                        "case_id": case_id,
                        "status": "unavailable",
                        "reason": (
                            "ambiguous matching debug NPZ sources: "
                            + ", ".join(_relative(path, project) for path in candidates)
                        ),
                    }
                )
                continue
            path = candidates[0]
            entry, status = _map_entry(path, project, model=model, canonical_mask=canonical_mask)
            statuses.append({"model": model, "case_id": case_id, **status})
            if entry is not None:
                data["entries"][case_id][model] = entry
                data["organization"].append(_organization_record(path, project, model, case_id))
    return data, statuses


def _tree_array(record: dict[str, Any], key: str, dtype: Any = float) -> np.ndarray:
    value = record.get(key)
    if value is None or isinstance(value, dict):
        return np.empty((0,), dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _tree_layout_x(children: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Assign deterministic diagram positions from the exported child graph."""

    count = int(valid.shape[0])
    child_map = {
        parent: sorted(
            {
                int(child)
                for child in children[parent].reshape(-1)
                if 0 <= int(child) < count and valid[int(child)] and int(child) != parent
            }
        )
        for parent in range(count)
        if valid[parent]
    }
    child_ids = {child for values in child_map.values() for child in values}
    roots = sorted(index for index in range(count) if valid[index] and index not in child_ids)
    if not roots:
        roots = [index for index in range(count) if valid[index]]
    positions: dict[int, float] = {}
    visiting: set[int] = set()
    next_leaf = 0

    def visit(node: int) -> float:
        nonlocal next_leaf
        if node in positions:
            return positions[node]
        if node in visiting:
            position = float(next_leaf)
            next_leaf += 1
            positions[node] = position
            return position
        visiting.add(node)
        children_for_node = [child for child in child_map.get(node, []) if child not in visiting]
        if children_for_node:
            position = float(np.mean([visit(child) for child in children_for_node]))
        else:
            position = float(next_leaf)
            next_leaf += 1
        visiting.discard(node)
        positions[node] = position
        return position

    for root in roots:
        visit(root)
    for node in range(count):
        if valid[node] and node not in positions:
            visit(node)
    return np.asarray([positions.get(index, np.nan) for index in range(count)], dtype=float)


def _a_hierarchy_plot(records: list[dict[str, Any]], case_id: str | None = None) -> dict[str, Any] | None:
    """Show the exported tree vertically, with edges and level labels."""

    traces: list[dict[str, Any]] = []
    for record in records:
        if record.get("model") != "A" or record.get("status") != "available" or (case_id is not None and record.get("case_id") != case_id):
            continue
        coords = _tree_array(record, "tree_coords")
        levels = _tree_array(record, "tree_levels", int).reshape(-1)
        mass = _tree_array(record, "tree_mass").reshape(-1)
        response = _tree_array(record, "tree_response_norm").reshape(-1)
        valid = _tree_array(record, "tree_valid", bool).reshape(-1)
        children = _tree_array(record, "tree_children", int)
        if coords.ndim != 2 or coords.shape[-1] != 2 or not levels.size:
            continue
        count = len(coords)
        if any(array.shape != (count,) for array in (levels, mass, response, valid)):
            continue
        if children.ndim != 2 or children.shape[0] != count:
            continue
        x = _tree_layout_x(children, valid)
        y = levels[:count].astype(float)
        edges_x: list[float | None] = []
        edges_y: list[float | None] = []
        if children.ndim == 3 and children.shape[0] == 1:
            children = children[0]
        for parent, row in enumerate(children):
            if not valid[parent]:
                continue
            for child in row.reshape(-1):
                child_index = int(child)
                if 0 <= child_index < count and valid[child_index]:
                    edges_x.extend((float(x[parent]), float(x[child_index]), None))
                    edges_y.extend((float(y[parent]), float(y[child_index]), None))
        if edges_x:
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": edges_x,
                    "y": edges_y,
                    "name": f"{record['case_id']} tree edges",
                    "line": {"color": "#c8d3d9", "width": 1},
                    "hoverinfo": "skip",
                }
            )
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": x[valid].tolist(),
                "y": y[valid].tolist(),
                "name": f"{record['case_id']} nodes",
                "customdata": np.column_stack(
                    [
                        np.arange(count)[valid],
                        mass[:count][valid],
                        response[:count][valid],
                        coords[:count, 0][valid],
                        coords[:count, 1][valid],
                    ]
                ).tolist(),
                "marker": {
                    "size": (6.0 + 5.0 * np.sqrt(np.maximum(mass[:count][valid], 0.0))).tolist(),
                    "color": response[:count][valid].tolist(),
                    "colorscale": "Viridis",
                    "showscale": not any(trace.get("marker", {}).get("showscale") for trace in traces),
                    "colorbar": {"title": "response norm"},
                },
                "hovertemplate": "node=%{customdata[0]}<br>level=%{y}<br>mass=%{customdata[1]:.4g}<br>response norm=%{customdata[2]:.4g}<br>actual x=%{customdata[3]:.4g}<br>actual y=%{customdata[4]:.4g}<extra></extra>",
            }
        )
    if not traces:
        return None
    return plot(
        f"Track A vertical hierarchy from exact tree exports{f' · case {case_id}' if case_id else ''}",
        traces,
        height=520,
        xaxis={"title": "deterministic tree-layout order (actual coordinates in hover)", "gridcolor": "#e8edf1", "dtick": 1},
        yaxis={"title": "tree level (higher is coarser)", "dtick": 1, "gridcolor": "#e8edf1"},
    )


def _a_mass_response_plot(
    records: list[dict[str, Any]],
    case_id: str | None = None,
    level: int | None = None,
) -> dict[str, Any] | None:
    traces: list[dict[str, Any]] = []
    for record in records:
        if record.get("model") != "A" or record.get("status") != "available" or (case_id is not None and record.get("case_id") != case_id):
            continue
        coords = _tree_array(record, "tree_coords")
        mass = _tree_array(record, "tree_mass").reshape(-1)
        response = _tree_array(record, "tree_response_norm").reshape(-1)
        levels = _tree_array(record, "tree_levels", int).reshape(-1)
        valid = _tree_array(record, "tree_valid", bool).reshape(-1)
        if coords.ndim != 2 or coords.shape[-1] != 2 or not mass.size or not response.size:
            continue
        count = len(coords)
        if any(array.shape != (count,) for array in (mass, response, levels, valid)):
            continue
        selected = valid if level is None else valid & (levels == level)
        if not bool(selected.any()):
            continue
        name = str(record["case_id"])
        traces.extend(
            [
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": coords[:count, 0][selected].tolist(),
                    "y": coords[:count, 1][selected].tolist(),
                    "name": f"{name} mass",
                    "xaxis": "x1",
                    "yaxis": "y1",
                    "marker": {"size": 8, "color": mass[:count][selected].tolist(), "colorscale": "Blues", "showscale": False},
                    "hovertemplate": "case=" + name + "<br>mass=%{marker.color:.4g}<extra></extra>",
                },
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": coords[:count, 0][selected].tolist(),
                    "y": coords[:count, 1][selected].tolist(),
                    "name": f"{name} response norm",
                    "xaxis": "x2",
                    "yaxis": "y2",
                    "marker": {"size": 8, "color": response[:count][selected].tolist(), "colorscale": "Viridis", "showscale": False},
                    "hovertemplate": "case=" + name + "<br>response norm=%{marker.color:.4g}<extra></extra>",
                },
            ]
        )
    if not traces:
        return None
    level_label = "all levels" if level is None else f"level {level}"
    figure = plot(f"Track A tree mass and response norm maps · {case_id or 'all cases'} · {level_label}", traces, height=500, showlegend=True)
    figure["layout"].update(
        grid={"rows": 1, "columns": 2, "pattern": "independent"},
        xaxis={"title": "tree x coordinate", "domain": [0.0, 0.46], "scaleanchor": "y1", "scaleratio": 1},
        yaxis={"title": "tree y coordinate", "domain": [0, 1]},
        xaxis2={"title": "tree x coordinate", "domain": [0.54, 1.0], "scaleanchor": "y2", "scaleratio": 1},
        yaxis2={"title": "tree y coordinate", "domain": [0, 1]},
    )
    return figure


def _formal_probe_results(study: Path, track: str) -> list[dict[str, Any]]:
    """Read only the final ``probes.json`` contract for a candidate track."""

    payload = read_json(study / track / "probes.json", {})
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return []
    return [row for row in payload["results"] if isinstance(row, dict)]


def _a_receiver_plot(study: Path) -> dict[str, Any] | None:
    traces: list[dict[str, Any]] = []
    for result in _formal_probe_results(study, "track_a"):
        routing = result.get("ordinary_hierarchical_routing")
        if not isinstance(routing, dict):
            continue
        case_id = str(result.get("case_id", "unknown"))
        for receiver_name, label in (("field_probes", "field receiver"), ("physical_ports", "physical port")):
            details = routing.get(receiver_name)
            rows = details.get("rows") if isinstance(details, dict) else None
            if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
                continue
            row = rows[0]
            levels = np.asarray(row.get("levels", []), dtype=float).reshape(-1)
            eta = np.asarray(row.get("eta", []), dtype=float).reshape(-1)
            attention = np.asarray(row.get("learned_attention", []), dtype=float).reshape(-1)
            node_ids = np.asarray(row.get("node_ids", []), dtype=int).reshape(-1)
            if not levels.size or not (levels.shape == eta.shape == attention.shape == node_ids.shape):
                continue
            count = levels.size
            custom = np.column_stack((node_ids[:count], levels[:count], eta[:count], attention[:count])).tolist()
            traces.extend(
                [
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": levels[:count].tolist(),
                        "y": eta[:count].tolist(),
                        "name": f"{case_id} {label} eta",
                        "customdata": custom,
                        "marker": {"color": "#315f7c", "size": 8},
                        "hovertemplate": "node=%{customdata[0]}<br>level=%{x}<br>eta=%{y:.4g}<extra></extra>",
                    },
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": levels[:count].tolist(),
                        "y": attention[:count].tolist(),
                        "name": f"{case_id} {label} learned attention",
                        "customdata": custom,
                        "marker": {"color": "#c05a3f", "symbol": "x", "size": 9},
                        "hovertemplate": "node=%{customdata[0]}<br>level=%{x}<br>learned attention=%{y:.4g}<extra></extra>",
                    },
                ]
            )
    return (
        plot(
            "Track A selected node levels, eta, and learned attention",
            traces,
            height=500,
            xaxis={"title": "selected node level", "dtick": 1},
            yaxis={"title": "routing value"},
        )
        if traces
        else None
    )


def _b_support_plot(records: list[dict[str, Any]], case_id: str | None = None) -> dict[str, Any] | None:
    traces = []
    for record in records:
        if record.get("model") != "B" or not record.get("support_x") or (case_id is not None and record.get("case_id") != case_id):
            continue
        traces.append({"type": "scatter", "mode": "markers", "x": record["support_x"], "y": record["support_y"], "name": f"B case {record['case_id']}", "marker": {"size": 8, "color": record.get("state") or "#2f6f9f", "colorscale": "Viridis", "showscale": True}, "hovertemplate": "support (%{x:.3f}, %{y:.3f})<br>state norm=%{marker.color:.4g}<extra></extra>"})
    return plot(f"B support centers colored by exported group-state norm · {case_id or 'all cases'}", traces, height=470, xaxis={"title": "x", "scaleanchor": "y"}, yaxis={"title": "y"}) if traces else None


def _topology_array(record: dict[str, Any], key: str, dtype: Any) -> np.ndarray:
    topology = record.get("topology", {})
    value = topology.get(key) if isinstance(topology, dict) else None
    if value is None or isinstance(value, dict):
        return np.empty((0,), dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _b_topology_plot(records: list[dict[str, Any]], case_id: str | None = None) -> dict[str, Any] | None:
    """Show the implemented route; measured group identities have their own panel."""

    record = next(
        (
            row for row in records
            if row.get("model") == "B" and row.get("status") == "available"
            and (case_id is None or row.get("case_id") == case_id)
        ),
        None,
    )
    if record is None:
        return None
    # Each box denotes a computation, not one learned state. In particular,
    # the eight coarse seeds are shared by all groups, not indexed by group ID.
    nodes = {
        "modules": (0.0, 3.5, "Encoded modules"),
        "environment": (0.0, 0.0, "Fine environment"),
        "groups": (1.0, 3.5, "Existing group states"),
        "group_attention": (2.0, 3.5, "Group-source attention<br>occupancy weighted"),
        "env_attention": (2.0, 0.0, "Environment-source<br>attention"),
        "seeds": (1.0, 1.7, "8 coarse seeds<br>shared queries + residual"),
        "processor": (3.0, 1.7, "Sum + 1 processor<br>8 coarse states"),
        "coarse_read": (4.0, 1.7, "Coarse read"),
        "local_read": (3.0, 5.2, "Local group read"),
        "receivers": (5.0, 3.5, "Ports / field contexts"),
    }
    routes = (
        ("modules", "groups", "modules → groups", "#315f7c"),
        ("environment", "groups", "fine environment → groups", "#7c5aa6"),
        ("groups", "group_attention", "groups → coarse source", "#39826a"),
        ("environment", "env_attention", "fine environment → coarse source", "#7c5aa6"),
        ("seeds", "group_attention", "shared coarse queries", "#60717e"),
        ("seeds", "env_attention", "shared coarse queries", "#60717e"),
        ("seeds", "processor", "coarse seed residual", "#60717e"),
        ("group_attention", "processor", "group context", "#39826a"),
        ("env_attention", "processor", "environment context", "#7c5aa6"),
        ("processor", "coarse_read", "coarse states → coarse read", "#39826a"),
        ("coarse_read", "receivers", "coarse read → receivers", "#39826a"),
        ("groups", "local_read", "groups → local read", "#d38b54"),
        ("local_read", "receivers", "local read → receivers", "#d38b54"),
    )
    traces = []
    annotations = []
    for source, target, name, color in routes:
        x0, y0, _ = nodes[source]
        x1, y1, _ = nodes[target]
        traces.append({
            "type": "scatter", "mode": "lines", "x": [x0, x1], "y": [y0, y1],
            "name": name, "line": {"color": color, "width": 1.5},
            "hovertemplate": name + "<extra></extra>", "showlegend": False,
        })
        annotations.append({
            "x": x0 + 0.78 * (x1 - x0), "y": y0 + 0.78 * (y1 - y0),
            "ax": x0 + 0.60 * (x1 - x0), "ay": y0 + 0.60 * (y1 - y0),
            "xref": "x", "yref": "y", "axref": "x", "ayref": "y",
            "text": "", "showarrow": True, "arrowhead": 2, "arrowwidth": 1.5,
            "arrowcolor": color,
        })
    for x, y, label in nodes.values():
        annotations.append({
            "x": x, "y": y, "xref": "x", "yref": "y", "text": label,
            "showarrow": False, "bgcolor": "#ffffff", "bordercolor": "#c7d2d9",
            "borderpad": 5, "font": {"size": 10},
        })
    figure = plot(
        "Track B group and coarse computation", traces, height=500, showlegend=False,
        xaxis={"range": [-0.6, 5.6], "showticklabels": False, "showgrid": False, "zeroline": False},
        yaxis={"range": [-0.7, 5.9], "showticklabels": False, "showgrid": False, "zeroline": False},
    )
    figure["layout"]["annotations"] = annotations + [{
        "text": "Architecture schematic. Measured shared group IDs are shown separately.",
        "xref": "paper", "yref": "paper", "x": 0, "y": 1.08,
        "showarrow": False, "font": {"size": 10, "color": "#60717e"},
    }]
    return figure


def _b_attention_plot(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    traces = []
    for record in records:
        values = record.get("attention", []) if record.get("model") == "B" else []
        if values:
            traces.append({"type": "histogram", "x": values, "name": f"B case {record['case_id']}", "opacity": 0.65, "nbinsx": 30})
    return plot("B group-read attention export", traces, height=430, barmode="overlay", xaxis={"title": "attention value"}, yaxis={"title": "count"}) if traces else None


def _b_source_routing_plot(study: Path) -> dict[str, Any] | None:
    traces: list[dict[str, Any]] = []
    for result in _formal_probe_results(study, "track_b"):
        conditional = result.get("conditional_encoded_module")
        connectivity = conditional.get("receiver_connectivity") if isinstance(conditional, dict) else None
        if not isinstance(connectivity, dict):
            continue
        selected = {int(value) for value in connectivity.get("selected_module_group_ids", [])}
        case_id = str(result.get("case_id", "unknown"))
        rows = connectivity.get("receivers", [])
        if not isinstance(rows, list):
            continue
        for receiver_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            receiver = str(row.get("receiver", f"receiver_{receiver_index}"))
            group_ids = [int(value) for value in row.get("group_ids", [])]
            shared = {int(value) for value in row.get("shared_with_selected_module", [])}
            if not group_ids:
                continue
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": group_ids,
                    "y": [receiver] * len(group_ids),
                    "name": f"{case_id} {receiver}",
                    "customdata": [[int(group), int(group in selected), int(group in shared)] for group in group_ids],
                    "marker": {
                        "size": 10,
                        "color": [2 if group in shared else 1 if group in selected else 0 for group in group_ids],
                        "colorscale": [[0.0, "#c8d3d9"], [0.5, "#d38b54"], [1.0, "#315f7c"]],
                        "cmin": 0,
                        "cmax": 2,
                        "showscale": False,
                    },
                    "hovertemplate": "group=%{x}<br>selected-module support=%{customdata[1]}<br>near/far shared=%{customdata[2]}<extra></extra>",
                }
            )
    if not traces:
        return None
    return plot(
        "Track B source routing from fixed receiver supports",
        traces,
        height=480,
        xaxis={"title": "actual support group ID", "dtick": 1},
        yaxis={"title": "receiver role", "categoryorder": "array", "categoryarray": ["near_port", "far_field"]},
    )


def _b_influence_plot(study: Path) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for result in _formal_probe_results(study, "track_b"):
        conditional = result.get("conditional_encoded_module")
        influence = conditional.get("selected_receiver_influence") if isinstance(conditional, dict) else None
        connectivity = conditional.get("receiver_connectivity", {}) if isinstance(conditional, dict) else {}
        shared = connectivity.get("near_far_shared_group_ids", []) if isinstance(connectivity, dict) else []
        if not isinstance(influence, dict):
            continue
        case_id = str(result.get("case_id", "unknown"))
        for receiver in ("near_port", "far_field"):
            receiver_data = influence.get(receiver)
            if not isinstance(receiver_data, dict):
                continue
            for route, color in (("local_group_read", "#d38b54"), ("group_mediated_coarse", "#315f7c")):
                details = receiver_data.get(route)
                steps = details.get("steps") if isinstance(details, dict) else None
                summary = steps.get("h=0.001", {}).get("summary", {}) if isinstance(steps, dict) else {}
                summary_small = steps.get("h=0.0005", {}).get("summary", {}) if isinstance(steps, dict) else {}
                value = safe_float(summary, "jvp_norm_mean") if isinstance(summary, dict) else None
                disagreement = safe_float(summary, "jvp_fd_difference_norm_mean") if isinstance(summary, dict) else None
                disagreement_small = safe_float(summary_small, "jvp_fd_difference_norm_mean") if isinstance(summary_small, dict) else None
                if value is None:
                    continue
                rows.append(
                    {
                        "label": f"{case_id} {receiver} {route}",
                        "receiver": receiver,
                        "route": route,
                        "value": value,
                        "disagreement": disagreement,
                        "disagreement_small": disagreement_small,
                        "color": color,
                        "shared": ",".join(str(item) for item in shared),
                    }
                )
    if not rows:
        return None
    labels = [row["label"] for row in rows]
    traces = [
        bar_trace(
            labels,
            [row["value"] for row in rows],
            "conditional JVP norm mean at h=0.001",
            "#315f7c",
            customdata=[
                [row["receiver"], row["route"], row["shared"], row["disagreement_small"], row["disagreement"]]
                for row in rows
            ],
            hovertemplate="%{x}<br>receiver=%{customdata[0]}<br>route=%{customdata[1]}<br>JVP norm mean=%{y:.4g}<br>shared IDs=%{customdata[2]}<br>AD/FD disagreement h=0.0005=%{customdata[3]:.4g}<br>AD/FD disagreement h=0.001=%{customdata[4]:.4g}<extra></extra>",
        )
    ]
    return plot(
        "Track B conditional influence · near port vs farthest fixed field probe · AD/FD discrepancies shown in hover",
        traces,
        height=520,
        barmode="group",
        xaxis={"title": "receiver and route", "tickangle": -35},
        yaxis={"title": "conditional JVP norm mean (h=0.001)", "type": "log", "gridcolor": "#e8edf1"},
    )


def _formal_intervention_rows(study: Path) -> list[dict[str, Any]]:
    """Read only the final per-track JSON intervention contract.

    The epoch-10 ``*_execution_check`` files intentionally contain four-query
    partial physical errors and must never enter the endpoint intervention
    figure.  The formal filenames are exact and carry full-grid canonical
    deltas when the run has completed.
    """

    rows: list[dict[str, Any]] = []
    for track, model in (("track_a", "A"), ("track_b", "B")):
        path = study / track / "interventions.json"
        payload = read_json(path, {})
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            continue
        checkpoint = payload.get("checkpoint", {})
        checkpoint_label = checkpoint.get("label") if isinstance(checkpoint, dict) else None
        for result in payload["results"]:
            if not isinstance(result, dict):
                continue
            case_id = str(result.get("case_id", "unknown"))
            query_count = result.get("query_count")
            ground_truth = result.get("ground_truth_errors")
            interventions = result.get("interventions")
            if not isinstance(ground_truth, dict) or not isinstance(interventions, dict):
                continue
            normal = ground_truth.get("normal")
            normal_metrics = normal.get("metrics") if isinstance(normal, dict) else None
            for mode, intervention in interventions.items():
                if not isinstance(intervention, dict):
                    continue
                delta = intervention.get("error_deltas")
                if not isinstance(delta, dict) and isinstance(normal_metrics, dict):
                    variant = ground_truth.get(mode)
                    variant_metrics = variant.get("metrics") if isinstance(variant, dict) else None
                    if isinstance(variant_metrics, dict):
                        delta = {
                            key: float(value) - float(normal_metrics[key])
                            for key, value in variant_metrics.items()
                            if key in normal_metrics
                            and isinstance(value, (int, float))
                            and isinstance(normal_metrics[key], (int, float))
                            and math.isfinite(float(value))
                            and math.isfinite(float(normal_metrics[key]))
                        }
                if not isinstance(delta, dict):
                    continue
                for metric, value in delta.items():
                    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        continue
                    rows.append(
                        {
                            "scope": "P1" if str(mode).lower() == "p1_only" else str(mode).upper(),
                            "mode": str(mode),
                            "model": model,
                            "case_id": case_id,
                            "metric": str(metric),
                            "value": float(value),
                            "query_count": query_count,
                            "checkpoint": checkpoint_label,
                            "source": str(path),
                        }
                    )
    return rows


def _intervention_rows(study: Path) -> list[dict[str, Any]]:
    """Compatibility name for callers of the renderer's prior helper."""

    return _formal_intervention_rows(study)


def _intervention_plot(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    preferred = (
        "global_field_fluid_norm_l2",
        "global_field_fluid_norm_pooled_relative_l2",
        "global_field_fluid_norm_relative_l2",
    )
    metric = next((candidate for candidate in preferred if any(row.get("metric") == candidate for row in rows)), None)
    if metric is None:
        metric = str(rows[0].get("metric", ""))
    rows = [row for row in rows if row.get("metric") == metric]
    traces = []
    for scope, color in (("P0", "#315f7c"), ("P1", "#d38b54"), ("P2", "#7c5aa6")):
        subset = [row for row in rows if row["scope"] == scope]
        if subset:
            traces.append(
                bar_trace(
                    [f"{row['model']} · {row['case_id']}" for row in subset],
                    [row["value"] for row in subset],
                    scope,
                    color,
                    offsetgroup=scope,
                    customdata=[[row.get("query_count"), row.get("checkpoint"), row.get("source")] for row in subset],
                    hovertemplate="%{x}<br>mode=" + scope + "<br>query count=%{customdata[0]}<br>checkpoint=%{customdata[1]}<br>delta=%{y:.4g}<extra></extra>",
                )
            )
    return plot(
        f"Formal P0/P1/P2 ground-truth deltas · {metric}",
        traces,
        height=460,
        barmode="group",
        xaxis={"title": "candidate / anchor", "tickangle": -35},
        yaxis={"title": "intervened minus normal ground-truth error", "zeroline": True},
    ) if traces else None


def _timing_stat_rows(
    payload: dict[str, Any],
    source: Path,
    *,
    track: str,
    run: str,
    timing_chunk: int | None,
    context: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = payload.get("models") or []
    if not models and isinstance(payload.get("rows"), list):
        models = [{"architecture": track, "real_cases": payload["rows"]}]
    for model in models:
        if not isinstance(model, dict):
            continue
        checkpoint = model.get("checkpoint") if isinstance(model.get("checkpoint"), dict) else {}
        model_name = str(checkpoint.get("label") or model.get("architecture") or run)
        checkpoint_epoch = checkpoint.get("epoch")
        for kind, names in (("real", ("real_cases", "real_anchors")), ("synthetic", ("synthetic_cases", "synthetic_shapes"))):
            cases: list[Any] = []
            for name in names:
                value = model.get(name)
                if isinstance(value, list):
                    cases.extend(value)
            for case in cases:
                if not isinstance(case, dict):
                    continue
                normal = case.get("normal")
                if not isinstance(normal, dict):
                    continue
                # Mature real anchors store normal.phases.  Mature synthetic
                # shapes store normal directly; candidate protocol files may
                # use either representation, so both are retained.
                phases = normal.get("phases")
                phase_rows = phases.items() if isinstance(phases, dict) else (("full_forward", normal),)
                shape = case.get("shape") if isinstance(case.get("shape"), dict) else {}
                shape_label = ""
                if shape:
                    shape_label = " ".join(f"{key}={shape[key]}" for key in ("E", "M", "Q") if key in shape)
                case_id = str(case.get("case_id", "")) if kind == "real" else ""
                receiver_chunk = case.get("receiver_chunk_size")
                effective_chunk = timing_chunk if timing_chunk is not None else receiver_chunk
                for phase, stats in phase_rows:
                    if not isinstance(stats, dict):
                        continue
                    median = safe_float(stats, "median_ms", "mean_ms")
                    if median is None:
                        continue
                    peak_bytes = safe_float(stats, "peak_allocated_bytes", "incremental_peak_allocated_bytes")
                    rows.append(
                        {
                            "track": track,
                            "run": run,
                            "model": model_name,
                            "architecture": model.get("architecture"),
                            "checkpoint_epoch": checkpoint_epoch,
                            "checkpoint_context": context,
                            "timing_chunk": effective_chunk,
                            "query_count": effective_chunk,
                            "kind": kind,
                            "case_id": case_id,
                            "shape": shape,
                            "shape_label": shape_label,
                            "phase": str(phase),
                            "median_ms": median,
                            "peak_allocated_mib": peak_bytes / (1024.0 * 1024.0) if peak_bytes is not None else None,
                            "source": str(source),
                        }
                    )
    return rows


def _timing_rows(study: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track, run in (("track_a", "1807"), ("track_b", "1808")):
        for query_count, filename in ((128, "timing_chunk128.json"), (2048, "timing_chunk2048.json")):
            path = study / track / filename
            payload = read_json(path, {})
            if isinstance(payload, dict):
                rows.extend(
                    _timing_stat_rows(
                        payload,
                        path,
                        track=track,
                        run=run,
                        timing_chunk=query_count,
                        context="candidate endpoint500",
                    )
                )
    # Existing mature timing artifacts are valid measured full-forward costs,
    # but their checkpoint context is explicit and is never presented as a
    # candidate's endpoint timing.
    mature = study.parent / "five_model_epoch5000/timing/five_model_timing.json"
    payload = read_json(mature, {})
    if isinstance(payload, dict):
        rows.extend(
            _timing_stat_rows(
                payload,
                mature,
                track="parent",
                run="parent",
                timing_chunk=None,
                context="parent mature checkpoint5000",
            )
        )
    exact_parent_timing = study.parent / "regional_response/timing/regional_vs_dense.json"
    payload = read_json(exact_parent_timing, {})
    if isinstance(payload, dict):
        rows.extend(
            _timing_stat_rows(
                payload,
                exact_parent_timing,
                track="parent",
                run="parent",
                timing_chunk=None,
                context="parent exact500",
            )
        )
    parent_timing_chunk128 = study / "comparison/parent_timing_chunk128.json"
    payload = read_json(parent_timing_chunk128, {})
    if isinstance(payload, dict):
        rows.extend(
            _timing_stat_rows(
                payload,
                parent_timing_chunk128,
                track="parent",
                run="parent",
                timing_chunk=128,
                context="parent exact500",
            )
        )
    return rows


def _timing_workload_key(row: dict[str, Any]) -> str:
    if row.get("kind") == "real":
        return f"real:{row.get('case_id', '')}"
    shape = row.get("shape") if isinstance(row.get("shape"), dict) else {}
    values = []
    for key in ("E", "M", "Q"):
        value = shape.get(key)
        try:
            values.append(f"{key}={int(value)}")
        except (TypeError, ValueError):
            values.append(f"{key}={value}")
    return "synthetic:" + " ".join(values)


def _timing_workload_options() -> list[tuple[str, str, str]]:
    options: list[tuple[str, str, str]] = []
    for case_id in TIMING_REAL_CASES:
        workload = f"real:{case_id}"
        for phase in TIMING_PHASES:
            options.append((f"{workload}|{phase}", f"real {case_id} · {TIMING_PHASE_LABELS[phase]}", workload))
    for workload, label in TIMING_SYNTHETIC_WORKLOADS:
        options.append((f"{workload}|full_forward", f"{label} · full forward", workload))
    return options


def _timing_workload_label(workload: str) -> str:
    if workload.startswith("real:"):
        return f"real {workload.split(':', 1)[1]}"
    for key, label in TIMING_SYNTHETIC_WORKLOADS:
        if workload == key:
            return label
    return workload


def _timing_model_label(row: dict[str, Any]) -> str:
    run = str(row.get("run", ""))
    if run in RUN_DISPLAY:
        return RUN_DISPLAY[run]
    architecture = str(row.get("architecture", ""))
    if architecture in TIMING_ARCHITECTURE_LABELS:
        return TIMING_ARCHITECTURE_LABELS[architecture]
    return _display_model(row)


def _timing_context_rank(row: dict[str, Any]) -> int:
    return {
        "candidate endpoint500": 0,
        "parent exact500": 1,
        "parent mature checkpoint5000": 2,
    }.get(str(row.get("checkpoint_context")), 99)


def _dedupe_timing_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one measured row per model/workload/phase/chunk.

    Candidate endpoint rows remain distinct from parent rows.  When a parent
    model is present in both exact500 and mature checkpoint timing artifacts,
    exact500 wins for the same workload and phase.
    """

    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        model_key = str(row.get("run")) if row.get("track") != "parent" else str(row.get("architecture"))
        key = (
            str(row.get("track")),
            model_key,
            _timing_workload_key(row),
            str(row.get("phase")),
            row.get("query_count"),
        )
        previous = selected.get(key)
        if previous is None or _timing_context_rank(row) < _timing_context_rank(previous):
            selected[key] = row
    return list(selected.values())


def _timing_plot(
    rows: list[dict[str, Any]],
    query_count: int,
    workload: str | None = None,
    phase: str = "full_forward",
) -> dict[str, Any] | None:
    if workload is None:
        workload = "real:0273"
    contexts = {"candidate endpoint500", "parent exact500", "parent mature checkpoint5000"}
    subset = [
        row
        for row in _dedupe_timing_rows(rows)
        if row.get("query_count") == query_count
        and row.get("checkpoint_context") in contexts
        and _timing_workload_key(row) == workload
        and row.get("phase") == phase
    ]
    if not subset:
        return None
    model_order = {name: index for index, name in enumerate(HISTORY_ORDER)}
    subset.sort(key=lambda row: (model_order.get(_timing_model_label(row), len(model_order)), _timing_model_label(row)))
    labels = [_timing_model_label(row) for row in subset]
    customdata = [
        [
            row.get("checkpoint_context"),
            Path(str(row.get("source", ""))).name,
            row.get("case_id"),
            row.get("shape_label"),
            row.get("peak_allocated_mib"),
        ]
        for row in subset
    ]
    trace = bar_trace(
        labels,
        [row["median_ms"] for row in subset],
        TIMING_PHASE_LABELS.get(phase, phase),
        "#315f7c" if phase == "full_forward" else "#d38b54",
        customdata=customdata,
        hovertemplate=(
            "%{x}<br>median=%{y:.4g} ms<br>provenance=%{customdata[0]}"
            "<br>source=%{customdata[1]}<br>case=%{customdata[2]}<br>shape=%{customdata[3]}"
            "<br>peak allocated=%{customdata[4]:.4g} MiB<extra></extra>"
        ),
    )
    workload_label = _timing_workload_label(workload)
    context_label = ", ".join(sorted({str(row.get("checkpoint_context")) for row in subset}))
    figure = plot(
        f"{workload_label}<br>{TIMING_PHASE_LABELS.get(phase, phase)} · chunk {query_count}",
        [trace],
        height=460,
        barmode="group",
        xaxis={"title": "model / track"},
        yaxis={"title": "median milliseconds", "gridcolor": "#e8edf1"},
    )
    figure["layout"]["margin"]["t"] = 105
    figure["layout"]["annotations"] = [{
        "text": context_label, "xref": "paper", "yref": "paper", "x": 0,
        "y": 1.10, "xanchor": "left", "showarrow": False,
        "font": {"size": 10, "color": "#60717e"},
    }]
    return figure


def _timing_variant_spec(rows: list[dict[str, Any]], query_count: int) -> dict[str, Any] | None:
    figures: dict[str, dict[str, Any]] = {}
    options = _timing_workload_options()
    for key, _label, workload in options:
        phase = key.rsplit("|", 1)[1]
        figure = _timing_plot(rows, query_count, workload, phase)
        if figure is not None:
            figures[key] = figure
    if not figures:
        return None
    selector_id = f"timing_{query_count}_workload_phase"
    return {
        "figures": figures,
        "selectors": [{"id": selector_id, "label": "workload / phase", "options": [(key, label) for key, label, _workload in options]}],
    }


def _accuracy_cost_plot(headline: list[dict[str, str]], timing_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compare exact-500 accuracy to one fixed measured full-forward shape."""

    accuracy: dict[str, float] = {}
    for row in _available_rows(headline):
        run = str(row.get("run", ""))
        value = safe_float(row, "global_field_fluid_norm_pooled_relative_l2")
        if run in {"1401", "1801", "1804", "1805", "1806", "1807", "1808"} and value is not None:
            accuracy[run] = value
    run_by_architecture = {
        "legacy_honf": "1401",
        "geometry_latent_field": "1801",
        "dense_pairwise_field": "1804",
        "sparse_interface_honf": "1805",
        "regional_response_honf": "1806",
        "hierarchical_regional_honf": "1807",
        "group_mediated_reader": "1808",
    }
    context_priority = {
        "candidate endpoint500": 0,
        "parent exact500": 1,
        "parent mature checkpoint5000": 2,
    }
    selected: dict[str, dict[str, Any]] = {}
    for row in timing_rows:
        shape = row.get("shape") if isinstance(row.get("shape"), dict) else {}
        if (
            row.get("kind") != "synthetic"
            or row.get("phase") != "full_forward"
            or shape.get("E") != 3072
            or shape.get("M") != 128
            or shape.get("Q") != 262144
            or row.get("timing_chunk") != 2048
            or row.get("checkpoint_context") not in context_priority
        ):
            continue
        run = str(row.get("run"))
        if run == "parent":
            run = run_by_architecture.get(str(row.get("architecture")), "")
        if run not in accuracy:
            continue
        cost = safe_float(row, "median_ms")
        if cost is None:
            continue
        old = selected.get(run)
        new_key = (
            context_priority[str(row["checkpoint_context"])],
        )
        old_key = (
            context_priority[str(old["checkpoint_context"])],
        ) if old is not None else None
        if old is None or new_key < old_key:
            selected[run] = row
    points = [(run, safe_float(row, "median_ms"), accuracy[run], row) for run, row in selected.items()]
    if not points:
        return None
    traces = []
    for run, cost, error, row in points:
        traces.append(
            {
                "type": "scatter",
                "mode": "markers+text",
                "x": [cost],
                "y": [error],
                "text": [_label_for_run(run)],
                "textposition": "top center",
                "name": _label_for_run(run),
                "marker": {"size": 11, "color": MODEL_COLORS.get(_label_for_run(run), "#315f7c")},
                "customdata": [[run, row.get("checkpoint_context"), row.get("shape_label"), row.get("source")]],
                "hovertemplate": "%{text}<br>exact500 accuracy=%{y:.4g}<br>measured full-forward=%{x:.4g} ms<br>run=%{customdata[0]}<br>timing=%{customdata[1]}<br>shape=%{customdata[2]}<extra></extra>",
            }
        )
    contexts = ", ".join(sorted({str(row.get("checkpoint_context")) for _, _, _, row in points}))
    return plot(
        f"Accuracy vs measured full-forward cost · fixed synthetic shape E=3072, M=128, Q=262144 · {contexts}",
        traces,
        height=500,
        xaxis={"title": "measured full-forward median (ms)", "type": "log"},
        yaxis={"title": "exact epoch 500 pooled relative L2", "type": "log"},
    )


def _table_card(rows: list[dict[str, str]], parent_selection: dict[str, Any]) -> str:
    headers = ("run", "model", "status", "checkpoint_epoch", "global_field_fluid_norm_pooled_relative_l2", "reason")
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(row.get(header, ''))}</td>" for header in headers) + "</tr>")
    if not body:
        body.append('<tr><td colspan="6">No saved-best candidate rows are available yet.</td></tr>')
    parent_reason = escape(parent_selection.get("reason", "Parent selected comparison unavailable."))
    return f"""
<section class="card"><h2>Saved-best candidate tables</h2>
<p class="muted">This policy is separate from exact epoch 500. Parent saved-best comparison is unavailable: {parent_reason}</p>
<div class="table-wrap"><table><thead><tr>{''.join(f'<th>{escape(header)}</th>' for header in headers)}</tr></thead><tbody>{''.join(body)}</tbody></table></div></section>
"""


def _figure_card(key: str, title: str, figure: dict[str, Any] | None, reason: str = "No matching artifact rows are available yet.") -> str:
    if not figure or not figure.get("data"):
        return f'<section class="card"><h2>{escape(title)}</h2><div class="missing">{escape(reason)}</div></section>'
    return f'<section class="card"><h2>{escape(title)}</h2><div id="fig_{slug(key)}" class="plot"></div></section>'


def _variant_card(
    key: str,
    title: str,
    figure: dict[str, Any] | None,
    spec: dict[str, Any] | None,
) -> str:
    if not spec or not spec.get("figures"):
        return _figure_card(key, title, figure)
    controls = []
    for selector in spec.get("selectors", []):
        options = "".join(
            f'<option value="{escape(value)}">{escape(label)}</option>'
            for value, label in selector.get("options", [])
        )
        controls.append(f'<label>{escape(selector["label"])} <select id="{escape(selector["id"])}">{options}</select></label>')
    return f'<section class="card"><h2>{escape(title)}</h2><div class="controls">{"".join(controls)}</div><div id="fig_{slug(key)}" class="plot"></div></section>'


def _anchor_javascript(anchor_data: dict[str, Any]) -> str:
    return f"""
const anchorData = {js(anchor_data)};
function masked(values, mask) {{ return values.map((row,i)=>row.map((value,j)=>mask[i][j] ? value : null)); }}
function extent(caseId, key, channel, absolute) {{
  let lo=Infinity, hi=-Infinity;
  for (const model of anchorData.models) {{
    const entry=(anchorData.entries[caseId]||{{}})[model]; if (!entry) continue;
    const values=entry[key][channel], mask=entry.fluid_mask;
    for (let i=0;i<values.length;i++) for (let j=0;j<values[i].length;j++) {{
      const value=values[i][j]; if (!mask[i][j] || value===null || !Number.isFinite(value)) continue;
      const shown=absolute ? Math.abs(value) : value; lo=Math.min(lo,shown); hi=Math.max(hi,shown);
    }}
  }}
  return Number.isFinite(lo)&&Number.isFinite(hi)?[lo,hi]:[0,1];
}}
function renderAnchors() {{
  const caseId=document.getElementById('anchor_case').value, model=document.getElementById('anchor_model').value, channel=document.getElementById('anchor_channel').value;
  const target=document.getElementById('anchor_maps'), entry=(anchorData.entries[caseId]||{{}})[model];
  if (!entry) {{ if (target.data) Plotly.purge(target); target.innerHTML='<div class="missing">This model/anchor debug export is unavailable.</div>'; return; }}
  if (target.querySelector('.missing')) target.replaceChildren();
  const c=anchorData.channels.indexOf(channel), ref=masked(entry.reference[c],entry.fluid_mask), pred=masked(entry.prediction[c],entry.fluid_mask), error=masked(entry.error[c],entry.fluid_mask);
  const valueRange=extent(caseId,'reference',c,false).concat(extent(caseId,'prediction',c,false));
  const zmin=Math.min(valueRange[0],valueRange[2]), zmax=Math.max(valueRange[1],valueRange[3]), emax=extent(caseId,'error',c,true)[1]||1;
  const common={{type:'heatmap',x:entry.x,y:entry.y,hoverongaps:false,xgap:0,ygap:0}};
  const traces=[
    Object.assign({{}},common,{{z:ref,colorscale:'Viridis',zmin:zmin,zmax:zmax,showscale:false,xaxis:'x1',yaxis:'y1',name:'ground truth'}}),
    Object.assign({{}},common,{{z:pred,colorscale:'Viridis',zmin:zmin,zmax:zmax,colorbar:{{len:0.8,x:0.62}},xaxis:'x2',yaxis:'y2',name:'prediction'}}),
    Object.assign({{}},common,{{z:error,colorscale:'RdBu',zmin:-emax,zmax:emax,zmid:0,colorbar:{{len:0.8,x:1.02}},xaxis:'x3',yaxis:'y3',name:'signed error'}})
  ];
  const x=entry.x, y=entry.y, dx=(x[1]-x[0])/2, dy=(y[1]-y[0])/2, xr=[x[0]-dx,x[x.length-1]+dx], yr=[y[0]-dy,y[y.length-1]+dy];
  const axis={{zeroline:false,showticklabels:true,constrain:'domain'}};
  Plotly.react(target,traces,{{title:{{text:model+' · case '+caseId+' · '+channel,x:0.02,xanchor:'left'}},height:390,margin:{{l:48,r:90,t:70,b:32}},grid:{{rows:1,columns:3,pattern:'independent'}},xaxis:Object.assign({{}},axis,{{domain:[0,.28],range:xr}}),yaxis:Object.assign({{}},axis,{{domain:[0,1],range:yr,scaleanchor:'x',scaleratio:1}}),xaxis2:Object.assign({{}},axis,{{domain:[.32,.60],range:xr}}),yaxis2:Object.assign({{}},axis,{{domain:[0,1],range:yr,scaleanchor:'x2',scaleratio:1,showticklabels:false}}),xaxis3:Object.assign({{}},axis,{{domain:[.71,.99],range:xr}}),yaxis3:Object.assign({{}},axis,{{domain:[0,1],range:yr,scaleanchor:'x3',scaleratio:1,showticklabels:false}}),annotations:[{{text:'Ground truth',x:.14}},{{text:'Prediction',x:.46}},{{text:'Signed error',x:.85}}].map(a=>Object.assign(a,{{xref:'paper',yref:'paper',y:1.10,xanchor:'center',showarrow:false}})),showlegend:false}},{{responsive:true,displaylogo:false}});
}}
for (const id of ['anchor_case','anchor_model','anchor_channel']) document.getElementById(id).addEventListener('change',renderAnchors);
renderAnchors();
"""


def _render_page(
    output: Path,
    figures: dict[str, dict[str, Any] | None],
    variants: dict[str, dict[str, Any]],
    anchor_data: dict[str, Any],
    source_links: list[str],
    notes: list[str],
    best_rows: list[dict[str, str]],
    parent_selection: dict[str, Any],
    plotly: str,
) -> None:
    cards = [
        _figure_card("headline", "Exact 500 headline", figures.get("headline")),
        _figure_card("channels", "Exact 500 normalized channels", figures.get("channels")),
        _figure_card("pair_case", "Exact 500 requested case-pair distributions", figures.get("pair_case")),
        _figure_card("pair_summary", "Exact 500 requested pair summaries", figures.get("pair_summary")),
        _figure_card("physical", "Exact 500 near / far / physical fields", figures.get("physical")),
        _figure_card("kpis", "Exact 500 physical engineering KPIs", figures.get("kpis")),
        _figure_card("history_epoch", "Validation field histories by epoch", figures.get("history_epoch")),
        _figure_card("history_time", "Validation field histories by logged active time", figures.get("history_time")),
        _figure_card("history_temperature_epoch", "Validation temperature histories by epoch", figures.get("history_temperature_epoch")),
        _figure_card("history_temperature_time", "Validation temperature histories by logged active time", figures.get("history_temperature_time")),
        _variant_card("a_hierarchy", "Track A vertical hierarchy with tree levels and edges", figures.get("a_hierarchy"), variants.get("a_hierarchy")),
        _figure_card("a_receivers", "Track A selected field receiver and physical port routing", figures.get("a_receivers")),
        _variant_card("a_mass_response", "Track A tree mass and response norm maps", figures.get("a_mass_response"), variants.get("a_mass_response")),
        _variant_card("b_support", "Track B support centers and group states", figures.get("b_support"), variants.get("b_support")),
        _variant_card("b_topology", "Track B source-routing topology schematic", figures.get("b_topology"), variants.get("b_topology")),
        _figure_card("b_routing", "Track B measured shared support IDs by receiver", figures.get("b_routing")),
        _figure_card("b_influence", "Track B local versus coarse conditional influence", figures.get("b_influence")),
        _figure_card("intervention", "Formal P0/P1/P2 ground-truth deltas", figures.get("intervention")),
        _variant_card("timing128", "Scientific timing at evaluation chunk 128", figures.get("timing128"), variants.get("timing128")),
        _variant_card("timing2048", "Inference timing at receiver chunk 2048", figures.get("timing2048"), variants.get("timing2048")),
        _figure_card("accuracy_cost", "Accuracy versus measured full-forward cost", figures.get("accuracy_cost")),
    ]
    cases = "".join(f'<option value="{escape(case)}">{escape(case)}</option>' for case in anchor_data["cases"])
    models = "".join(f'<option value="{escape(model)}">{escape(model)} · {escape(MODEL_LABELS[model])}</option>' for model in anchor_data["models"])
    channels = "".join(f'<option value="{escape(channel)}">{escape(CHANNEL_LABELS[channel])}</option>' for channel in anchor_data["channels"])
    links = "".join(f'<li><a href="{escape(link)}">{escape(link)}</a></li>' for link in source_links)
    note_html = "".join(f"<li>{escape(note)}</li>" for note in notes)
    figure_json = js({key: value for key, value in figures.items() if value})
    variant_json = js(variants)
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HONF NStage2 comparison</title>
<style>
:root{{--ink:#23313d;--muted:#60717e;--line:#dce4e9;--panel:#fff;--page:#f4f7f9}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--page);color:var(--ink);font:14px/1.4 Inter,Arial,sans-serif}}
main{{width:min(1500px,96vw);margin:auto;padding:28px 0 70px}} header,.card,details{{background:var(--panel);border:1px solid var(--line);padding:16px 18px;margin:0 0 18px}}
h1{{margin:0 0 6px;font-size:27px}} h2{{margin:0 0 10px;font-size:16px}} p{{color:var(--muted)}} .notice{{background:#eef5f8;border-left:4px solid #315f7c;padding:12px 15px;color:#364d5a}}
.controls{{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:14px 0}} select{{padding:7px 9px;border:1px solid #b8c6ce;background:#fff;border-radius:4px}}
.plot{{min-height:300px}} .missing{{padding:25px 8px;color:var(--muted);background:#f7f9fa;border:1px dashed #c7d2d9}} .muted{{color:var(--muted)}}
.table-wrap{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}} th{{background:#eef5f8}}
ul{{margin:8px 0 4px 18px}} a{{color:#315f7c}} @media(min-width:1000px){{.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.grid .card{{margin-bottom:0}}}}
</style></head><body><main>
<header><h1>HONF NStage2 · matched epoch 500 comparison</h1>
<p>Offline evidence page for Tracks A (hierarchical regional) and B (group-mediated reader). Exact endpoint tables, saved-best tables, maps, timings, and interventions are shown only when their source artifacts exist.</p>
<div class="notice"><strong>Evidence boundary.</strong> A missing candidate table, debug export, timing file, or intervention remains an unavailable panel. Parent saved-best weights are unavailable, so no selected-parent comparison is drawn. Attention, membership, support, and tree values describe model exports; they are not causal influence or physical truth.</div></header>
{_table_card(best_rows, parent_selection)}
<section class="card"><h2>Five-anchor field maps</h2><p class="muted">Ground truth and prediction share a color scale across available models for each case/channel. Signed error uses a symmetric scale. Aligned models share the canonical stored fluid mask from the Reader/Regional source when available.</p>
<div class="controls"><label>anchor <select id="anchor_case">{cases}</select></label><label>model <select id="anchor_model">{models}</select></label><label>channel <select id="anchor_channel">{channels}</select></label></div><div id="anchor_maps" class="plot"></div></section>
<div class="grid">{''.join(cards)}</div>
<details><summary>Sources and unavailable evidence</summary><p class="muted">The reducer's comparison CSV/JSON files remain the source of numerical values. Debug NPZ and timing paths are listed only for auditability.</p><ul>{links}</ul><ul>{note_html}</ul></details>
</main><script>{plotly}</script><script>
const figures={figure_json};
for (const [key,figure] of Object.entries(figures)) {{ const target=document.getElementById('fig_'+key); if(target&&figure&&figure.data&&figure.data.length) Plotly.newPlot(target,figure.data,figure.layout,{{responsive:true,displaylogo:false}}); }}
const variants={variant_json};
function renderVariant(name) {{
  const spec=variants[name]; if(!spec) return;
  const key=spec.selectors.map(selector=>document.getElementById(selector.id).value).join('|');
  const figure=spec.figures[key], target=document.getElementById('fig_'+name);
  if(!figure||!figure.data||!figure.data.length) {{ if(target.data) Plotly.purge(target); target.innerHTML='<div class="missing">This workload/phase scientific export is unavailable.</div>'; return; }}
  if(target.querySelector('.missing')) target.replaceChildren();
  Plotly.react(target,figure.data,figure.layout,{{responsive:true,displaylogo:false}});
}}
for (const [name,spec] of Object.entries(variants)) {{ for (const selector of spec.selectors) document.getElementById(selector.id).addEventListener('change',()=>renderVariant(name)); renderVariant(name); }}
</script><script>{_anchor_javascript(anchor_data)}</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")


def render_nstage2(root: str | Path | None = None, output: str | Path | None = None, plotly_js: str | Path | None = None) -> Path:
    """Render ``nstage2/figures/index.html`` from available study artifacts."""

    project, study = _resolve_roots(root)
    comparison = study / "comparison"
    reduction = _comparison_files(comparison)
    manifest = read_json(comparison / "comparison.json", {})
    anchor_data, statuses = _load_anchor_data(project, study)
    organization = anchor_data.pop("organization", [])
    intervention_rows = _formal_intervention_rows(study)
    timing_rows = _timing_rows(study)
    history = _history_rows(reduction["history"])
    pair_case, pair_summary = _pair_plots(reduction["pairs"], reduction["pair_summary"])
    a_records = [row for row in organization if row.get("model") == "A" and row.get("status") == "available"]
    b_records = [row for row in organization if row.get("model") == "B" and row.get("status") == "available"]
    a_cases = sorted({str(row["case_id"]) for row in a_records})
    b_cases = sorted({str(row["case_id"]) for row in b_records})
    a_hierarchy_figures = {
        case_id: figure
        for case_id in a_cases
        if (figure := _a_hierarchy_plot(organization, case_id)) is not None
    }
    a_mass_figures: dict[str, dict[str, Any]] = {}
    a_mass_levels: set[int] = set()
    for record in a_records:
        levels = _tree_array(record, "tree_levels", int).reshape(-1)
        if levels.size:
            a_mass_levels.update(int(value) for value in np.unique(levels))
    mass_level_values: list[str] = ["all"] + [str(value) for value in sorted(a_mass_levels)]
    for case_id in a_cases:
        for level_value in mass_level_values:
            level = None if level_value == "all" else int(level_value)
            figure = _a_mass_response_plot(organization, case_id, level)
            if figure is not None:
                a_mass_figures[f"{case_id}|{level_value}"] = figure
    b_support_figures = {
        case_id: figure
        for case_id in b_cases
        if (figure := _b_support_plot(organization, case_id)) is not None
    }
    b_topology_figures = {
        case_id: figure
        for case_id in b_cases
        if (figure := _b_topology_plot(organization, case_id)) is not None
    }
    timing128_spec = _timing_variant_spec(timing_rows, 128)
    timing2048_spec = _timing_variant_spec(timing_rows, 2048)
    variants: dict[str, dict[str, Any]] = {}
    if a_hierarchy_figures:
        variants["a_hierarchy"] = {
            "figures": a_hierarchy_figures,
            "selectors": [{"id": "a_hierarchy_case", "label": "case", "options": [(case, case) for case in a_cases]}],
        }
    if a_mass_figures:
        variants["a_mass_response"] = {
            "figures": a_mass_figures,
            "selectors": [
                {"id": "a_mass_case", "label": "case", "options": [(case, case) for case in a_cases]},
                {"id": "a_mass_level", "label": "tree level", "options": [(value, value) for value in mass_level_values]},
            ],
        }
    if b_support_figures:
        variants["b_support"] = {
            "figures": b_support_figures,
            "selectors": [{"id": "b_support_case", "label": "case", "options": [(case, case) for case in b_cases]}],
        }
    if b_topology_figures:
        variants["b_topology"] = {
            "figures": b_topology_figures,
            "selectors": [{"id": "b_topology_case", "label": "case", "options": [(case, case) for case in b_cases]}],
        }
    if timing128_spec:
        variants["timing128"] = timing128_spec
    if timing2048_spec:
        variants["timing2048"] = timing2048_spec
    timing128_initial = next(iter(timing128_spec["figures"].values()), None) if timing128_spec else None
    timing2048_initial = next(iter(timing2048_spec["figures"].values()), None) if timing2048_spec else None
    figures: dict[str, dict[str, Any] | None] = {
        "headline": _headline_plot(reduction["headline"]),
        "channels": _channel_plot(reduction["channels"]),
        "pair_case": pair_case,
        "pair_summary": pair_summary,
        "physical": _physical_plot(reduction["physical"]),
        "kpis": _kpi_plot(reduction["kpis"]),
        "history_epoch": _history_plot(history, "val_field_mse", "epoch", "Validation field MSE by epoch", "training epoch"),
        "history_time": _history_plot(history, "val_field_mse", "logged_active_seconds", "Validation field MSE by logged active time", "cumulative logged train + validation seconds"),
        "history_temperature_epoch": _history_plot(history, "val_temperature_mse", "epoch", "Validation temperature MSE by epoch", "training epoch"),
        "history_temperature_time": _history_plot(history, "val_temperature_mse", "logged_active_seconds", "Validation temperature MSE by logged active time", "cumulative logged train + validation seconds"),
        "a_hierarchy": next(iter(a_hierarchy_figures.values()), None),
        "a_receivers": _a_receiver_plot(study),
        "a_mass_response": next(iter(a_mass_figures.values()), None),
        "b_support": next(iter(b_support_figures.values()), None),
        "b_topology": next(iter(b_topology_figures.values()), None),
        "b_routing": _b_source_routing_plot(study),
        "b_influence": _b_influence_plot(study),
        "intervention": _intervention_plot(intervention_rows),
        "timing128": timing128_initial,
        "timing2048": timing2048_initial,
        "accuracy_cost": _accuracy_cost_plot(reduction["headline"], timing_rows),
    }
    best_rows = reduction["best"]
    parent_selection = manifest.get("parent_best_through_500", {}) if isinstance(manifest, dict) else {}
    source_links = [f"../comparison/{name}" for name in sorted(path.name for path in comparison.glob("*.csv"))]
    for relative in (
        "track_a/probes.json",
        "track_b/probes.json",
        "track_a/interventions.json",
        "track_b/interventions.json",
        "track_a/timing_chunk128.json",
        "track_a/timing_chunk2048.json",
        "track_b/timing_chunk128.json",
        "track_b/timing_chunk2048.json",
        "comparison/parent_timing_chunk128.json",
    ):
        if (study / relative).is_file():
            source_links.append("../" + relative)
    mature_timing = study.parent / "five_model_epoch5000/timing/five_model_timing.json"
    if mature_timing.is_file():
        source_links.append("../../five_model_epoch5000/timing/five_model_timing.json")
    notes = [
        "Exact endpoint values come from exact500_* reducer outputs; candidate rows remain visibly unavailable until the 90-case tables arrive.",
        "Saved-best candidate rows are separate from exact epoch 500. Parent best-through-500 weights are not selected from history minima.",
        "Anchor maps use the five fixed cases 0273, 0653, 0283, 0298, and 0302 and retain shared per-case/channel scales; aligned models use the shared canonical stored fluid mask.",
        "Track A hierarchy panels require exact tree_levels/tree_children/tree_coords/tree_bounds_min/tree_bounds_max/tree_states/tree_mass/tree_valid exports; selected receiver rows come only from track_a/probes.json.",
        "Track B routing and influence panels come only from track_b/probes.json. The far_field point is the farthest fixed field probe and may share support groups with the near port. AD/FD disagreement values are fixed-step diagnostics shown for context; their magnitude is not a confirmed convergence or causal claim.",
        "Timing panels use fixed workload/phase selectors: real 0273 and 0653 at four measured phases, plus the two synthetic shapes at full forward. X axes use short model labels; hover and titles retain exact500 versus mature checkpoint5000 provenance. Parent exact500 replaces a duplicated mature row for the same model/workload/phase/chunk, while candidate endpoint500 rows remain distinct.",
        "Accuracy-versus-cost uses only chunk-2048 measurements at the fixed E=3072, M=128, Q=262144 shape and prefers candidate endpoint500, then parent exact500, then mature checkpoint5000 timing; every point keeps its checkpoint context.",
        "Formal intervention panels read only track_a/interventions.json and track_b/interventions.json; epoch-10 execution-check JSON is excluded.",
    ]
    if not _formal_probe_results(study, "track_a"):
        notes.append("Track A selected receiver diagnostics are unavailable until track_a/probes.json arrives.")
    if not _formal_probe_results(study, "track_b"):
        notes.append("Track B source routing and conditional influence diagnostics are unavailable until track_b/probes.json arrives.")
    if not intervention_rows:
        notes.append("Formal intervention deltas are unavailable until track_a/interventions.json or track_b/interventions.json arrives.")
    if not any(row.get("checkpoint_context") == "candidate endpoint500" for row in timing_rows):
        notes.append("Candidate scientific/inference timing files are unavailable; parent timing remains labeled checkpoint 5000.")
    notes.extend(f"{row['model']} case {row['case_id']}: {row.get('reason', row.get('source', 'available'))}" for row in statuses if row.get("status") != "available")
    notes.extend(f"{row['model']} case {row['case_id']}: {row['mask_source']}" for row in statuses if row.get("mask_source") and row.get("mask_source") != "stored fluid_mask")
    target = Path(output).expanduser().resolve() if output is not None else study / "figures" / "index.html"
    plotly_source_text = plotly_source(Path(plotly_js).expanduser() if plotly_js is not None else None)
    _render_page(target, figures, variants, anchor_data, source_links, notes, best_rows, parent_selection, plotly_source_text)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="repository root or NStage2 study root")
    parser.add_argument("--output", type=Path, default=None, help="offline HTML path (default: nstage2/figures/index.html)")
    parser.add_argument("--plotly-js", type=Path, default=None, help="optional local Plotly JS file")
    args = parser.parse_args(argv)
    path = render_nstage2(args.root, args.output, args.plotly_js)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
