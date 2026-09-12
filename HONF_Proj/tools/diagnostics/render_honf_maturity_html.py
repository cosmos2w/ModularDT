#!/usr/bin/env python3
"""Render the five-model HONF epoch-5000 diagnostic study as one offline HTML file.

The renderer intentionally consumes the reducer/evaluator CSV and JSON artifacts
already written by the study. It copies no checkpoints and needs no web server
or network access: locally installed Plotly JS is embedded into the page.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import importlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODEL_ORDER = ["Legacy", "Latent", "Dense", "Reader", "Regional"]
MODEL_COLORS = {
    "Legacy": "#53606e",
    "Latent": "#2f6f9f",
    "Dense": "#bf6b3c",
    "Reader": "#7c5aa6",
    "Regional": "#39826a",
}
CHANNELS = ["u", "v", "p", "omega", "temperature"]
CHANNEL_LABELS = {
    "u": "u velocity",
    "v": "v velocity",
    "p": "pressure",
    "omega": "vorticity",
    "temperature": "temperature",
}
CHANNEL_COLORS = {"u": "#315f7c", "v": "#bf6b3c", "p": "#7c5aa6", "omega": "#39826a", "temperature": "#a35a63"}
PHYSICAL_COLORS = {"global_field_fluid_norm": "#315f7c", "global_field_near_interface_norm": "#bf6b3c", "global_field_far_fluid_norm": "#39826a", "internal_temperature_physical": "#7c5aa6"}
ANCHOR_CASES = ["0273", "0298", "0302", "0653"]
MODEL_RUNS = {"Legacy": "1401", "Latent": "1801", "Dense": "1804", "Reader": "1805", "Regional": "1806"}
RUN_MODELS = {v: k for k, v in MODEL_RUNS.items()}


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def model_name(row: dict[str, Any]) -> str:
    value = string(row.get("model") or row.get("model_label"))
    for name in MODEL_ORDER:
        if value.startswith(name):
            return name
    run = string(row.get("run") or row.get("model_index"))
    return RUN_MODELS.get(run, value)


def epoch_value(row: dict[str, Any], *keys: str) -> int | None:
    keys = keys or ("epoch", "checkpoint_epoch")
    for key in keys:
        value = finite(row.get(key))
        if value is not None:
            return int(value)
    return None


def safe_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = finite(row.get(key))
        if value is not None:
            return value
    return None


def sort_models(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    return sorted(rows, key=lambda row: (order.get(model_name(row), 99), epoch_value(row) or 0, string(row.get("run"))))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def js(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":")).replace("</", "<" + chr(92) + "/")


def escape(value: Any) -> str:
    return html.escape(string(value), quote=True)


def slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", string(value).lower()).strip("_")


def layout(title: str, height: int = 420, **kwargs: Any) -> dict[str, Any]:
    base = {
        "title": {"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 15, "color": "#23313d"}},
        "height": height,
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"family": "Inter, Arial, sans-serif", "color": "#23313d", "size": 12},
        "margin": {"l": 62, "r": 24, "t": 58, "b": 64},
        "legend": {"orientation": "h", "y": -0.18, "x": 0.02},
        "hovermode": "closest",
        "xaxis": {"gridcolor": "#e8edf1", "zerolinecolor": "#b7c1c9"},
        "yaxis": {"gridcolor": "#e8edf1", "zerolinecolor": "#b7c1c9"},
    }
    base.update(kwargs)
    for key, axis in base.items():
        if key.startswith(("xaxis", "yaxis")) and isinstance(axis, dict) and isinstance(axis.get("title"), str):
            axis["title"] = {"text": axis["title"]}
    return base


def plot(title: str, traces: list[dict[str, Any]], height: int = 420, **kwargs: Any) -> dict[str, Any]:
    return {"data": traces, "layout": layout(title, height=height, **kwargs)}


def line_trace(x: list[Any], y: list[Any], name: str, color: str, **kwargs: Any) -> dict[str, Any]:
    trace = {
        "type": "scatter",
        "mode": "lines",
        "x": x,
        "y": y,
        "name": name,
        "line": {"color": color, "width": 2},
        "marker": {"color": color},
    }
    trace.update(kwargs)
    return trace


def bar_trace(x: list[Any], y: list[Any], name: str, color: str, **kwargs: Any) -> dict[str, Any]:
    trace = {"type": "bar", "x": x, "y": y, "name": name, "marker": {"color": color}}
    trace.update(kwargs)
    return trace


def load_reduction(root: Path) -> dict[str, list[dict[str, str]]]:
    directory = root / "reduction"
    names = [
        "headline",
        "metrics_long",
        "pooled_metrics",
        "strata",
        "worst_cases",
        "paired_case_deltas",
        "paired_summary",
        "channel_mse_contributions",
    ]
    return {name: read_csv(directory / f"{name}.csv") for name in names}


def exact_rows(rows: list[dict[str, str]], epoch: int = 5000) -> list[dict[str, str]]:
    return [row for row in rows if epoch_value(row) == epoch]


def paired_rows(rows: list[dict[str, str]], epoch: int = 5000) -> list[dict[str, str]]:
    return [
        row for row in rows
        if safe_float(row, "candidate_epoch") == epoch and safe_float(row, "baseline_epoch") == epoch
    ]


def endpoint_plots(reduction: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    headline = exact_rows(reduction.get("headline", []))
    by_model = {model_name(row): row for row in headline}
    labels = [name for name in MODEL_ORDER if name in by_model]
    values: dict[str, list[float | None]] = {}
    for metric, source in [
        ("Global fluid error (relative L2)", "relative_l2"),
        ("Equal-case mean fluid error", "equal_case_mean"),
        ("Median case error", "median"),
        ("P95 case error", "p95"),
        ("Worst case error", "max"),
    ]:
        values[metric] = [safe_float(by_model[name], source) for name in labels]
    traces = [
        bar_trace(labels, values["Global fluid error (relative L2)"], "pooled relative L2", "#315f7c"),
        bar_trace(labels, values["Equal-case mean fluid error"], "equal-case mean", "#d38b54"),
        bar_trace(labels, values["P95 case error"], "case P95", "#7c5aa6"),
    ]
    endpoint = plot(
        "Exact epoch 5000 reconstruction error across 90 matched cases",
        traces,
        height=460,
        barmode="group",
        yaxis={"title": "relative L2 / case error", "gridcolor": "#e8edf1"},
        xaxis={"title": "model family"},
    )
    channels = exact_rows(reduction.get("pooled_metrics", []))
    channel_values = defaultdict(dict)
    for row in channels:
        base = string(row.get("base"))
        for channel in CHANNELS:
            if base == f"field_{channel}_fluid_norm":
                value = safe_float(row, "mse")
                if value is not None:
                    channel_values[channel][model_name(row)] = value
    channel_traces = []
    for channel in CHANNELS:
        channel_traces.append(
            bar_trace(
                [name for name in MODEL_ORDER if name in channel_values[channel]],
                [channel_values[channel].get(name) for name in MODEL_ORDER if name in channel_values[channel]],
                CHANNEL_LABELS[channel],
                CHANNEL_COLORS[channel],
                offsetgroup=channel,
            )
        )
    channel_plot = plot(
        "Per-channel pooled field MSE at epoch 5000",
        channel_traces,
        height=460,
        barmode="group",
        yaxis={"title": "MSE contribution", "gridcolor": "#e8edf1"},
        xaxis={"title": "model family"},
    )
    physical_bases = [
        ("global_field_fluid_norm", "fluid field"),
        ("global_field_near_interface_norm", "near-interface field"),
        ("global_field_far_fluid_norm", "far-fluid field"),
        ("internal_temperature_physical", "internal temperature"),
    ]
    pooled_rows = [row for row in exact_rows(reduction.get("pooled_metrics", [])) if string(row.get("base")) in {base for base, _ in physical_bases}]
    physical_traces = []
    for base, label in physical_bases:
        subset = {model_name(row): safe_float(row, "relative_l2") for row in pooled_rows if string(row.get("base")) == base}
        names = [name for name in MODEL_ORDER if name in subset]
        physical_traces.append(bar_trace(names, [subset[name] for name in names], label, PHYSICAL_COLORS[base], offsetgroup=base))
    physical_plot = plot(
        "Exact epoch 5000 near / far / thermal outcomes",
        physical_traces,
        height=470,
        barmode="group",
        yaxis={"title": "pooled relative L2", "gridcolor": "#e8edf1"},
        xaxis={"title": "model family"},
    )
    return {"endpoint": endpoint, "channel": channel_plot, "physical": physical_plot}


def paired_plots(reduction: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    rows = paired_rows(reduction.get("paired_case_deltas", []))
    metric = "global_field_fluid_norm_l2"
    selected = [row for row in rows if string(row.get("metric")) == metric]
    groups: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        if string(row.get("candidate")) == string(row.get("baseline")):
            continue
        candidate = RUN_MODELS.get(string(row.get("candidate")), string(row.get("candidate")))
        baseline = RUN_MODELS.get(string(row.get("baseline")), string(row.get("baseline")))
        delta = safe_float(row, "delta")
        if delta is not None:
            groups[f"{candidate} − {baseline}"].append(delta)
    names = list(groups)
    box = [
        {
            "type": "box",
            "y": groups[name],
            "name": name,
            "marker": {"color": MODEL_COLORS.get(name.split(" − ")[0], "#315f7c")},
            "boxmean": True,
            "hovertemplate": "%{y:.4f}<extra>%{fullData.name}</extra>",
        }
        for name in names
    ]
    distribution = plot(
        "Paired 90-case error deltas at matched epoch 5000",
        box,
        height=480,
        yaxis={"title": "candidate − baseline relative L2", "gridcolor": "#e8edf1", "zeroline": True},
        xaxis={"title": "matched candidate versus baseline"},
        showlegend=False,
    )
    summary_rows = paired_rows(reduction.get("paired_summary", []))
    summary_names: list[str] = []
    summary_values: list[float | None] = []
    for row in summary_rows:
        if string(row.get("metric")) != metric:
            continue
        candidate = RUN_MODELS.get(string(row.get("candidate")), string(row.get("candidate")))
        baseline = RUN_MODELS.get(string(row.get("baseline")), string(row.get("baseline")))
        summary_names.append(f"{candidate} − {baseline}")
        summary_values.append(safe_float(row, "mean_delta", "delta_mean", "equal_case_mean"))
    summary = plot(
        "Paired mean error change at matched epoch 5000",
        [bar_trace(summary_names, summary_values, "mean delta", "#315f7c")],
        height=420,
        yaxis={"title": "mean candidate − baseline", "gridcolor": "#e8edf1", "zeroline": True},
        xaxis={"title": "matched comparison"},
        showlegend=False,
    )
    return {"distribution": distribution, "summary": summary}


def history_rows(root: Path) -> list[dict[str, Any]]:
    directory = root / "history"
    sources = [
        directory / "history_trajectory.csv",
        directory / "history_trajectories.csv",
        directory / "history_checkpoints.csv",
    ]
    source = next((path for path in sources if path.exists()), None)
    rows = read_csv(source) if source else []
    if not rows:
        summary = read_json(directory / "history_summary.json", {})
        for item in summary.get("trajectory", summary.get("rows", [])) if isinstance(summary, dict) else []:
            if isinstance(item, dict):
                rows.append({str(key): string(value) for key, value in item.items()})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        run = string(raw.get("run") or raw.get("run_id"))
        epoch = epoch_value(raw)
        if not run or epoch is None:
            continue
        grouped[run].append({"raw": raw, "epoch": epoch})
    out: list[dict[str, Any]] = []
    for run, items in grouped.items():
        items.sort(key=lambda item: item["epoch"])
        cumulative = 0.0
        for item in items:
            raw = item["raw"]
            train = safe_float(raw, "train_wall_seconds", "train_seconds")
            validation = safe_float(raw, "val_wall_seconds", "validation_wall_seconds", "val_seconds")
            if train is not None:
                cumulative += train
            if validation is not None:
                cumulative += validation
            row: dict[str, Any] = {
                "run": run,
                "model": RUN_MODELS.get(run, run),
                "epoch": item["epoch"],
                "schema": string(raw.get("schema")),
                "val_loss_total": safe_float(raw, "val_loss_total", "validation_loss"),
                "val_field_mse": safe_float(raw, "val_field_mse", "validation_field_mse"),
                "val_temperature_mse": safe_float(raw, "val_temperature_mse", "validation_temperature_mse"),
                "loss_total": safe_float(raw, "loss_total"),
                "field_mse": safe_float(raw, "field_mse"),
                "temperature_mse": safe_float(raw, "temperature_mse"),
                "train_wall_seconds": train,
                "val_wall_seconds": validation,
                "cumulative_logged_wall_seconds": cumulative if cumulative else safe_float(raw, "cumulative_logged_wall_seconds"),
                "peak_memory_mib": safe_float(raw, "peak_cuda_memory_mb", "peak_memory_mib"),
                "sampling_source": "logged trajectory/checkpoint rows",
            }
            out.append(row)
    return sorted(out, key=lambda row: (string(row["model"]), row["epoch"]))


def history_plots(rows: list[dict[str, Any]]) -> dict[str, Any]:
    plots: dict[str, Any] = {}
    for field, title, yaxis in [
        ("val_field_mse", "Validation field MSE versus epoch", "validation field MSE"),
        ("val_temperature_mse", "Validation temperature MSE versus epoch", "validation temperature MSE"),
    ]:
        traces = []
        for name in MODEL_ORDER:
            subset = [row for row in rows if row["model"] == name and row.get(field) is not None]
            subset.sort(key=lambda row: row["epoch"])
            if subset:
                traces.append(line_trace([row["epoch"] for row in subset], [row[field] for row in subset], name, MODEL_COLORS[name]))
        plots[f"{field}_epoch"] = plot(title, traces, height=440, xaxis={"title": "training epoch"}, yaxis={"title": yaxis, "type": "log", "gridcolor": "#e8edf1"})
        traces = []
        for name in MODEL_ORDER:
            subset = [row for row in rows if row["model"] == name and row.get(field) is not None and row.get("cumulative_logged_wall_seconds") is not None]
            subset.sort(key=lambda row: row["cumulative_logged_wall_seconds"])
            if subset:
                traces.append(line_trace([row["cumulative_logged_wall_seconds"] for row in subset], [row[field] for row in subset], name, MODEL_COLORS[name]))
        plots[f"{field}_time"] = plot(title.replace("versus epoch", "versus cumulative logged time"), traces, height=440, xaxis={"title": "cumulative logged training + validation wall seconds"}, yaxis={"title": yaxis, "type": "log", "gridcolor": "#e8edf1"})
    return plots


def timing_rows(root: Path) -> list[dict[str, Any]]:
    data = read_json(root / "timing" / "five_model_timing.json", {})
    rows: list[dict[str, Any]] = []
    for item in data.get("models", []) if isinstance(data, dict) else []:
        checkpoint = item.get("checkpoint", {})
        name = RUN_MODELS.get(string(checkpoint.get("label")).split("_")[0], "")
        if not name:
            label = string(checkpoint.get("label"))
            name = next((candidate for candidate in MODEL_ORDER if label.startswith(candidate)), string(item.get("architecture")))
        anchors = item.get("real_anchors", [])
        real_phases = []
        real_memory = []
        for anchor in anchors:
            phase = anchor.get("normal", {}).get("phases", {}).get("full_forward", {})
            if safe_float(phase, "median_ms") is not None:
                real_phases.append(float(phase["median_ms"]))
            allocated = safe_float(phase, "peak_allocated_bytes")
            if allocated is not None:
                real_memory.append(allocated / 1024.0**2)
        synthetic = item.get("synthetic_shapes", [])
        largest = None
        if synthetic:
            largest = max(synthetic, key=lambda entry: int(entry.get("shape", {}).get("Q", 0)))
        synthetic_phase = (largest or {}).get("normal", {})
        rows.append({
            "model": name,
            "run": MODEL_RUNS.get(name),
            "variant": string(item.get("real_anchors", [{}])[0].get("normal_variant")) if anchors else "",
            "real_anchor_median_ms": float(np.median(real_phases)) if real_phases else None,
            "real_peak_allocated_mib": max(real_memory) if real_memory else None,
            "synthetic_shape": string((largest or {}).get("shape")),
            "synthetic_largest_median_ms": safe_float(synthetic_phase, "median_ms"),
            "synthetic_largest_peak_allocated_mib": (safe_float(synthetic_phase, "peak_allocated_bytes") or 0.0) / 1024.0**2 if safe_float(synthetic_phase, "peak_allocated_bytes") is not None else None,
            "synthetic_largest_q": (largest or {}).get("shape", {}).get("Q"),
            "sampling": "median over measured anchor repetitions; largest synthetic grid uses the largest recorded Q",
        })
    return sort_models(rows)


def timing_plots(rows: list[dict[str, Any]], reduction: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    exact = {model_name(row): safe_float(row, "relative_l2") for row in exact_rows(reduction.get("headline", []))}
    real = [row for row in rows if row.get("real_anchor_median_ms") is not None and exact.get(row["model"]) is not None]
    pareto = {
        "data": [{
            "type": "scatter",
            "mode": "markers+text",
            "x": [row["real_anchor_median_ms"] for row in real],
            "y": [exact[row["model"]] for row in real],
            "text": [row["model"] for row in real],
            "textposition": "top center",
            "marker": {"size": 11, "color": [MODEL_COLORS[row["model"]] for row in real], "line": {"color": "#23313d", "width": 0.7}},
            "customdata": [[row.get("real_peak_allocated_mib")] for row in real],
            "hovertemplate": "%{text}<br>real median %{x:.2f} ms<br>epoch-5000 relative L2 %{y:.4f}<br>peak allocated %{customdata[0]:.1f} MiB<extra></extra>",
            "showlegend": False,
        }],
        "layout": layout("Exact epoch 5000 accuracy versus matched real-anchor timing", 440, xaxis={"title": "full-forward median per anchor (ms)", "gridcolor": "#e8edf1"}, yaxis={"title": "relative L2", "gridcolor": "#e8edf1"}),
    }
    synthetic = [row for row in rows if row.get("synthetic_largest_median_ms") is not None and exact.get(row["model"]) is not None]
    synthetic_plot = {
        "data": [{
            "type": "scatter",
            "mode": "markers+text",
            "x": [row["synthetic_largest_median_ms"] for row in synthetic],
            "y": [exact[row["model"]] for row in synthetic],
            "text": [row["model"] for row in synthetic],
            "textposition": "top center",
            "marker": {"size": 11, "color": [MODEL_COLORS[row["model"]] for row in synthetic], "line": {"color": "#23313d", "width": 0.7}},
            "customdata": [[row.get("synthetic_largest_peak_allocated_mib"), row.get("synthetic_largest_q")] for row in synthetic],
            "hovertemplate": "%{text}<br>largest synthetic median %{x:.2f} ms<br>relative L2 %{y:.4f}<br>peak allocated %{customdata[0]:.1f} MiB<br>Q=%{customdata[1]}<extra></extra>",
            "showlegend": False,
        }],
        "layout": layout("Exact epoch 5000 accuracy versus largest synthetic timing", 440, xaxis={"title": "largest synthetic full-forward median (ms)", "type": "log", "gridcolor": "#e8edf1"}, yaxis={"title": "relative L2", "gridcolor": "#e8edf1"}),
    }
    memory = [row for row in rows if row.get("real_peak_allocated_mib") is not None and exact.get(row["model"]) is not None]
    memory_trace = {
        "type": "scatter",
        "mode": "markers+text",
        "x": [row["real_peak_allocated_mib"] for row in memory],
        "y": [exact[row["model"]] for row in memory],
        "text": [row["model"] for row in memory],
        "textposition": "top center",
        "marker": {"size": 11, "color": [MODEL_COLORS[row["model"]] for row in memory], "line": {"color": "#23313d", "width": 0.7}},
        "customdata": [[row.get("real_anchor_median_ms")] for row in memory],
        "hovertemplate": "%{text}<br>peak allocated %{x:.1f} MiB<br>relative L2 %{y:.4f}<br>real median %{customdata[0]:.2f} ms<extra></extra>",
        "showlegend": False,
    }
    return {"pareto": pareto, "synthetic": synthetic_plot, "memory": plot("Exact epoch 5000 accuracy versus matched real-anchor memory", [memory_trace], height=420, yaxis={"title": "relative L2", "gridcolor": "#e8edf1"}, xaxis={"title": "peak allocated (MiB)", "type": "log", "gridcolor": "#e8edf1"}, showlegend=False)}


def best_selected_plots(root: Path) -> dict[str, Any]:
    directory = root / "reduction"
    rows = read_csv(directory / "best_selected_headline.csv")
    rows = sort_models(rows)
    labels = [model_name(row) for row in rows]
    values = [safe_float(row, "relative_l2") for row in rows]
    epochs = [safe_float(row, "checkpoint_epoch") for row in rows]
    trace = {
        "type": "bar",
        "x": labels,
        "y": values,
        "marker": {"color": [MODEL_COLORS.get(label, "#53606e") for label in labels]},
        "customdata": [[epoch] for epoch in epochs],
        "hovertemplate": "%{x}<br>best selected relative L2 %{y:.4f}<br>checkpoint epoch %{customdata[0]:.0f}<extra></extra>",
        "name": "best validation-field checkpoint",
        "showlegend": False,
    }
    return {"plot": plot("Best-by-validation-field checkpoint sensitivity", [trace], height=420, yaxis={"title": "relative L2", "gridcolor": "#e8edf1"}, xaxis={"title": "model family"}, showlegend=False), "rows": rows}


def anchor_path(project_root: Path, model: str, case_id: str) -> Path | None:
    candidates = [
        project_root / "diagnostics/generated/interface_operator_study/five_model_epoch5000/evaluation/debug_npz" / f"{model}_{MODEL_RUNS[model]}__5000__{case_id}.npz",
        project_root / "diagnostics/generated/interface_operator_study/epoch5000_comparison/evaluation/debug_npz" / f"{model}_{MODEL_RUNS[model]}__5000__{case_id}.npz",
    ]
    return next((path for path in candidates if path.exists()), None)


def array_grid(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array)
    if values.shape == shape:
        return values
    if values.size == shape[0] * shape[1]:
        return values.reshape(shape)
    return np.full(shape, np.nan, dtype=float)


def rounded_grid(array: np.ndarray, digits: int = 6) -> list[list[float | None]]:
    values = np.asarray(array, dtype=float)
    out: list[list[float | None]] = []
    for row in values:
        out.append([round(float(value), digits) if math.isfinite(float(value)) else None for value in row])
    return out


def load_anchors(project_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchors: dict[str, Any] = {"cases": ANCHOR_CASES, "models": MODEL_ORDER, "channels": CHANNELS, "data": {}}
    organization: list[dict[str, Any]] = []
    for case_id in ANCHOR_CASES:
        anchors["data"][case_id] = {}
        reference: dict[str, Any] | None = None
        for model in MODEL_ORDER:
            path = anchor_path(project_root, model, case_id)
            if path is None:
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    gt = np.asarray(data["gt_field_grid"], dtype=float)
                    pred = np.asarray(data["pred_field_grid"], dtype=float)
                    mask = np.asarray(data["fluid_mask"], dtype=bool)
                    x = np.asarray(data["x_grid"], dtype=float)
                    y = np.asarray(data["y_grid"], dtype=float)
                    shape = gt.shape[:2]
                    entry = {
                        "source": str(path.relative_to(project_root)),
                        "x": rounded_grid(x),
                        "y": rounded_grid(y),
                        "fluid_mask": mask.astype(int).tolist(),
                        "reference": [rounded_grid(gt[:, :, index]) for index in range(len(CHANNELS))],
                        "prediction": [rounded_grid(pred[:, :, index]) for index in range(len(CHANNELS))],
                        "error": [rounded_grid(pred[:, :, index] - gt[:, :, index]) for index in range(len(CHANNELS))],
                    }
                    anchors["data"][case_id][model] = entry
                    if reference is None:
                        reference = entry
                    organization.extend(organization_rows(data, model, case_id, shape, x, y))
            except (OSError, KeyError, ValueError):
                continue
        if reference is not None:
            for model in MODEL_ORDER:
                if model in anchors["data"][case_id]:
                    anchors["data"][case_id][model]["reference"] = reference["reference"]
    return anchors, organization


def organization_rows(data: np.lib.npyio.NpzFile, model: str, case_id: str, shape: tuple[int, int], x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    query_x = x.reshape(-1)
    query_y = y.reshape(-1)
    rows: list[dict[str, Any]] = []
    if model == "Latent" and "latent_query_attention" in data:
        attention = np.asarray(data["latent_query_attention"], dtype=float)
        values = np.nanmax(attention, axis=-1).reshape(-1)
        for qx, qy, value in zip(query_x, query_y, values):
            rows.append({"model": model, "case_id": case_id, "family": "latent query attention max (K=16)", "x": float(qx), "y": float(qy), "value": float(value)})
    elif model == "Regional" and "regional_environment_attention" in data:
        attention = np.asarray(data["regional_environment_attention"], dtype=float)
        values = np.nanmax(attention, axis=-1).reshape(-1)
        for qx, qy, value in zip(query_x, query_y, values):
            rows.append({"model": model, "case_id": case_id, "family": "regional query attention max (48 fixed regions)", "x": float(qx), "y": float(qy), "value": float(value)})
        if "regional_coords" in data:
            coords = np.asarray(data["regional_coords"], dtype=float)
            ids = np.asarray(data["regional_ids"]) if "regional_ids" in data else np.arange(len(coords))
            for coord, region_id in zip(coords, ids):
                rows.append({"model": model, "case_id": case_id, "family": "regional cover region id (fixed cover)", "x": float(coord[0]), "y": float(coord[1]), "value": int(region_id)})
            if "regional_response_states" in data:
                state_norm = np.linalg.norm(np.asarray(data["regional_response_states"], dtype=float), axis=1)
                for coord, value in zip(coords, state_norm):
                    rows.append({"model": model, "case_id": case_id, "family": "regional response-state norm", "x": float(coord[0]), "y": float(coord[1]), "value": float(value)})
    elif model == "Reader" and "interaction__group_read_degree" in data:
        values = np.asarray(data["interaction__group_read_degree"], dtype=float).reshape(-1)
        for qx, qy, value in zip(query_x, query_y, values):
            rows.append({"model": model, "case_id": case_id, "family": "sparse reader group-read degree", "x": float(qx), "y": float(qy), "value": float(value)})
        if "interaction__support_centres" in data and "interaction__group_state_norm" in data:
            centres = np.asarray(data["interaction__support_centres"], dtype=float)
            state_norm = np.asarray(data["interaction__group_state_norm"], dtype=float).reshape(-1)
            for centre, value in zip(centres, state_norm):
                rows.append({"model": model, "case_id": case_id, "family": "reader support centres · state norm", "x": float(centre[0]), "y": float(centre[1]), "value": float(value)})
    elif model in {"Dense", "Legacy"} and "module_centers" in data:
        centers = np.asarray(data["module_centers"], dtype=float)
        present = np.asarray(data["module_present"], dtype=float).reshape(-1) if "module_present" in data else np.ones(len(centers))
        for center, is_present in zip(centers, present):
            if is_present > 0:
                rows.append({"model": model, "case_id": case_id, "family": f"{model.lower()} module centers (no attention export)", "x": float(center[0]), "y": float(center[1]), "value": 1.0})
    return rows


def organization_plot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    families = []
    for row in rows:
        family = string(row.get("family"))
        if family not in families:
            families.append(family)
    for index, family in enumerate(families):
        subset = [row for row in rows if row.get("family") == family and row.get("case_id") == ANCHOR_CASES[0]]
        if not subset:
            continue
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": [row["x"] for row in subset],
            "y": [row["y"] for row in subset],
            "name": family,
            "visible": index == 0,
            "marker": {"size": 5 if len(subset) > 100 else 9, "color": [row["value"] for row in subset], "colorscale": "Viridis", "showscale": True, "colorbar": {"title": "family value", "x": 1.02, "thickness": 12}, "line": {"width": 0}},
            "customdata": [[row["model"], row["case_id"], row["value"]] for row in subset],
            "hovertemplate": "%{customdata[0]} · case %{customdata[1]}<br>x=%{x:.3f}, y=%{y:.3f}<br>value=%{customdata[2]:.4g}<extra></extra>",
        })
    traces.append({
        "type": "scatter",
        "mode": "markers",
        "x": [row["x"] for row in rows if row.get("model") == "Regional" and row.get("family") == "regional cover region id (fixed cover)" and row.get("case_id") == ANCHOR_CASES[0]],
        "y": [row["y"] for row in rows if row.get("model") == "Regional" and row.get("family") == "regional cover region id (fixed cover)" and row.get("case_id") == ANCHOR_CASES[0]],
        "name": "regional cover centroids",
        "marker": {"symbol": "x", "size": 7, "color": "#23313d"},
        "showlegend": True,
        "hovertemplate": "fixed regional cover centroid<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
    })
    buttons = []
    for index, family in enumerate(families):
        visible = [False] * (len(traces) - 1) + [True]
        if index < len(traces) - 1:
            visible[index] = True
        buttons.append({"label": family, "method": "update", "args": [{"visible": visible}, {"title": {"text": f"Organization export · {family} · anchor 0273"}}]})
    figure = plot("Organization export · choose one family · anchor 0273", traces, height=560, xaxis={"title": "x coordinate", "gridcolor": "#e8edf1"}, yaxis={"title": "y coordinate", "gridcolor": "#e8edf1", "scaleanchor": "x", "scaleratio": 1})
    figure["layout"]["updatemenus"] = [{"type": "dropdown", "x": 0.02, "y": 1.16, "xanchor": "left", "yanchor": "top", "buttons": buttons, "showactive": True}]
    return figure


def intervention_plot(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_csv(root / "interventions" / "phase_summary.csv")
    selected: list[dict[str, Any]] = []
    for row in rows:
        scope = string(row.get("scope")).lower()
        row_type = string(row.get("row_type")).lower()
        if scope not in {"p0", "p1", "p1_only", "p2"} or row_type not in {"equal_four_mean", "equal-four-mean"}:
            continue
        value = safe_float(row, "fluid", "global_field_fluid_norm_l2")
        if value is None:
            continue
        selected.append({"model": model_name(row), "run": string(row.get("run")), "scope": "P1" if scope == "p1_only" else scope.upper(), "fluid_delta": value, "definition": string(row.get("scope_semantics") or row.get("scope_description"))})
    scopes = ["P0", "P1", "P2"]
    traces = []
    for scope in scopes:
        subset = [row for row in selected if row["scope"] == scope]
        traces.append({
            "type": "bar",
            "x": [row["model"] for row in subset],
            "y": [row["fluid_delta"] for row in subset],
            "name": scope,
            "marker": {"color": {"P0": "#315f7c", "P1": "#d38b54", "P2": "#7c5aa6"}[scope]},
            "offsetgroup": scope,
            "hovertemplate": "%{x}<br>%{y:.4f}<extra>" + scope + "</extra>",
        })
    return plot("P0/P1/P2 matched four-anchor fluid-error effects", traces, height=430, barmode="group", yaxis={"title": "fluid relative-L2 effect (signed)", "gridcolor": "#e8edf1", "zeroline": True}, xaxis={"title": "model family"}), selected


def plotly_source(path: Path | None = None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    try:
        plotly_offline = importlib.import_module("plotly.offline")
        source = plotly_offline.get_plotlyjs()
        if source:
            return source
    except (ImportError, AttributeError, RuntimeError):
        pass
    raise RuntimeError("Plotly JS was not found. Install plotly locally or pass --plotly-js PATH.")


def figure_div(key: str, title: str, figure: dict[str, Any] | None) -> str:
    if not figure or not figure.get("data"):
        return f'<section class="card"><h2>{escape(title)}</h2><div class="missing">No rows were available for this panel.</div></section>'
    return f'<section class="card"><h2>{escape(title)}</h2><div id="fig_{slug(key)}" class="plot"></div></section>'


def anchor_javascript(anchors: dict[str, Any]) -> str:
    return f"""
const anchorData = {js(anchors)};
function maskedGrid(values, mask) {{
  return values.map((row, i) => row.map((v, j) => mask[i][j] ? v : null));
}}
function finiteExtent(caseId, key, channel, absolute) {{
  let lo = Infinity, hi = -Infinity;
  for (const model of anchorData.models) {{
    const entry = anchorData.data[caseId] && anchorData.data[caseId][model];
    if (!entry) continue;
    const values = entry[key][channel];
    const mask = entry.fluid_mask;
    for (let i = 0; i < values.length; i++) {{
      for (let j = 0; j < values[i].length; j++) {{
        if (!mask[i][j] || values[i][j] === null || !Number.isFinite(values[i][j])) continue;
        const value = absolute ? Math.abs(values[i][j]) : values[i][j];
        if (value < lo) lo = value;
        if (value > hi) hi = value;
      }}
    }}
  }}
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  return [lo, hi];
}}
function anchorFigure(caseId, model, channel) {{
  const entry = anchorData.data[caseId] && anchorData.data[caseId][model];
  if (!entry) return null;
  const c = anchorData.channels.indexOf(channel);
  const z0 = maskedGrid(entry.reference[c], entry.fluid_mask);
  const z1 = maskedGrid(entry.prediction[c], entry.fluid_mask);
  const z2 = maskedGrid(entry.error[c], entry.fluid_mask);
  const refRange = finiteExtent(caseId, 'reference', c, false);
  const predRange = finiteExtent(caseId, 'prediction', c, false);
  const valueRange = [Math.min(refRange[0], predRange[0]), Math.max(refRange[1], predRange[1])];
  const errAbs = finiteExtent(caseId, 'error', c, true)[1];
  const x = entry.x[0];
  const y = entry.y.map(row => row[0]);
  const common = {{type:'heatmap', x:x, y:y, hoverongaps:false, colorbar:{{len:0.30}}, xgap:0, ygap:0}};
  return [
    Object.assign({{}}, common, {{z:z0, colorscale:'Viridis', zmin:valueRange[0], zmax:valueRange[1], showscale:false, xaxis:'x1', yaxis:'y1', name:'reference'}}),
    Object.assign({{}}, common, {{z:z1, colorscale:'Viridis', zmin:valueRange[0], zmax:valueRange[1], colorbar:{{len:0.80, x:0.62, thickness:10, tickfont:{{size:10}}}}, xaxis:'x2', yaxis:'y2', name:'prediction'}}),
    Object.assign({{}}, common, {{z:z2, colorscale:'RdBu', zmin:-errAbs, zmax:errAbs, zmid:0, colorbar:{{len:0.80, x:1.02, thickness:10, tickfont:{{size:10}}}}, xaxis:'x3', yaxis:'y3', name:'error'}})
  ];
}}
function renderAnchors() {{
  const caseId = document.getElementById('anchor_case').value;
  const model = document.getElementById('anchor_model').value;
  const channel = document.getElementById('anchor_channel').value;
  const traces = anchorFigure(caseId, model, channel);
  const target = document.getElementById('anchor_maps');
  if (!traces) {{
    target.innerHTML = '<div class="missing">This model/anchor export is unavailable.</div>';
    return;
  }}
  const x=traces[0].x, y=traces[0].y;
  const dx=(x[1]-x[0])/2, dy=(y[1]-y[0])/2;
  const xr=[x[0]-dx,x[x.length-1]+dx], yr=[y[0]-dy,y[y.length-1]+dy];
  const axis = {{gridcolor:'#e8edf1', zeroline:false, showticklabels:true, constrain:'domain'}};
  Plotly.react(target, traces, {{
    title: {{text: model + ' · case ' + caseId + ' · ' + channel, x:0.02, xanchor:'left', font:{{size:15}}}},
    height: 360, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    margin:{{l:48,r:100,t:82,b:34}}, font:{{family:'Inter, Arial, sans-serif', size:11, color:'#23313d'}},
    grid:{{rows:1, columns:3, pattern:'independent'}},
    xaxis:Object.assign({{}}, axis, {{domain:[0.00,0.28], range:xr}}),
    yaxis:Object.assign({{}}, axis, {{domain:[0.0,1.0], range:yr, scaleanchor:'x', scaleratio:1}}),
    xaxis2:Object.assign({{}}, axis, {{domain:[0.32,0.60], range:xr}}),
    yaxis2:Object.assign({{}}, axis, {{domain:[0.0,1.0], range:yr, scaleanchor:'x2', scaleratio:1, showticklabels:false}}),
    xaxis3:Object.assign({{}}, axis, {{domain:[0.71,0.99], range:xr}}),
    yaxis3:Object.assign({{}}, axis, {{domain:[0.0,1.0], range:yr, scaleanchor:'x3', scaleratio:1, showticklabels:false}}),
    annotations:[{{text:'Reference',x:0.14}},{{text:'Prediction',x:0.46}},{{text:'Prediction − reference',x:0.85}}].map(a=>Object.assign(a,{{xref:'paper',yref:'paper',y:1.12,xanchor:'center',showarrow:false}})),
    showlegend:false
  }}, {{responsive:true, displaylogo:false}});
}}
document.getElementById('anchor_case').addEventListener('change', renderAnchors);
document.getElementById('anchor_model').addEventListener('change', renderAnchors);
document.getElementById('anchor_channel').addEventListener('change', renderAnchors);
renderAnchors();
"""


def render_html(
    output: Path,
    figures: dict[str, dict[str, Any]],
    anchors: dict[str, Any],
    source_links: list[str],
    source_notes: list[str],
    plotly: str,
) -> None:
    cards = [
        figure_div("endpoint", "Exact checkpoint reconstruction / near / far / thermal summary", figures.get("endpoint")),
        figure_div("channel", "Per-channel error contribution", figures.get("channel")),
        figure_div("physical", "Near / far / thermal outcomes", figures.get("physical")),
        figure_div("paired_distribution", "Paired 90-case distribution", figures.get("paired_distribution")),
        figure_div("paired_summary", "Paired mean error change", figures.get("paired_summary")),
        figure_div("best_selected", "Best validation-field checkpoint sensitivity", figures.get("best_selected")),
        figure_div("history_field_epoch", "History: validation field MSE versus epoch", figures.get("history_field_epoch")),
        figure_div("history_field_time", "History: validation field MSE versus cumulative logged time", figures.get("history_field_time")),
        figure_div("history_temperature_epoch", "History: validation temperature MSE versus epoch", figures.get("history_temperature_epoch")),
        figure_div("history_temperature_time", "History: validation temperature MSE versus cumulative logged time", figures.get("history_temperature_time")),
        figure_div("timing_pareto", "Accuracy versus matched real-anchor timing", figures.get("timing_pareto")),
        figure_div("timing_synthetic", "Accuracy versus largest synthetic timing", figures.get("timing_synthetic")),
        figure_div("timing_memory", "Matched real-anchor memory", figures.get("timing_memory")),
        figure_div("organization", "Organization / attention exports", figures.get("organization")),
        figure_div("intervention", "P0/P1/P2 effects", figures.get("intervention")),
    ]
    case_options = "".join(f'<option value="{escape(case)}">{escape(case)}</option>' for case in anchors.get("cases", []))
    model_options = "".join(f'<option value="{escape(model)}">{escape(model)}</option>' for model in anchors.get("models", []))
    channel_options = "".join(f'<option value="{escape(channel)}">{escape(CHANNEL_LABELS.get(channel, channel))}</option>' for channel in anchors.get("channels", []))
    links = "".join(f'<li><a href="{escape(link)}">{escape(link)}</a></li>' for link in source_links)
    notes = "".join(f"<li>{escape(note)}</li>" for note in source_notes)
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HONF maturity study · five models · epoch 5000</title>
<style>
:root {{ color-scheme: light; --ink:#23313d; --muted:#60717e; --line:#dce4e9; --panel:#ffffff; --page:#f4f7f9; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,Arial,sans-serif; color:var(--ink); background:var(--page); line-height:1.4; }}
main {{ width:min(1500px, 96vw); margin:0 auto; padding:30px 0 70px; }}
header {{ background:var(--panel); border:1px solid var(--line); padding:24px 28px; margin-bottom:18px; }}
h1 {{ margin:0 0 8px; font-size:27px; }}
h2 {{ margin:0 0 10px; font-size:16px; font-weight:650; }}
p {{ margin:7px 0; color:var(--muted); }}
.notice {{ background:#eef5f8; border-left:4px solid #315f7c; padding:12px 15px; margin:13px 0 0; color:#364d5a; }}
.controls {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; background:var(--panel); border:1px solid var(--line); padding:14px 16px; margin-bottom:18px; }}
label {{ color:var(--muted); font-size:13px; }}
select {{ padding:7px 10px; border:1px solid #b8c6ce; background:#fff; color:var(--ink); border-radius:4px; margin-left:5px; }}
.card {{ background:var(--panel); border:1px solid var(--line); padding:15px 16px 8px; margin:0 0 18px; }}
.plot {{ width:100%; min-height:300px; }}
.anchor-card {{ padding-bottom:16px; }}
#anchor_maps {{ min-height:360px; }}
.missing {{ color:var(--muted); padding:25px 8px; background:#f7f9fa; border:1px dashed #c7d2d9; }}
details {{ background:var(--panel); border:1px solid var(--line); padding:12px 16px; margin-top:18px; }}
summary {{ cursor:pointer; font-weight:650; }}
ul {{ margin:8px 0 4px 19px; padding:0; }}
code {{ font-size:12px; }}
@media (min-width: 1000px) {{ .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; align-items:start; }} .grid2 .card {{ margin-bottom:0; }} }}
</style>
</head>
<body>
<main>
<header>
<h1>HONF maturity study · five models · exact epoch 5000</h1>
<p>Offline, source-backed diagnostic figures for Legacy, Latent, Dense, Reader, and Regional on the matched 90-case evaluation.</p>
<div class="notice"><strong>How to read these figures.</strong> Exact epoch-5000 comparisons are primary: Dense has the lowest pooled fluid relative L2 (0.02966), followed by Regional (0.03119). Best-by-validation-field results are a separate sensitivity view, where Regional at checkpoint 4933 (0.02819) is below Dense at checkpoint 4738 (0.02896). Attention exports describe model-side weighting or read structure; they do not establish causal influence. Regional cover regions are a fixed deterministic cover in this export; they do not demonstrate learned geometric organization. Histories use logged trajectory/checkpoint rows for epoch plots and cumulative train + validation wall seconds for time plots. Timing synthetic rows are execution stress tests, not accuracy evidence. Memory labels are MiB.</div>
</header>
<section class="card anchor-card">
<h2>Four-anchor fluid-masked field maps: reference, prediction, and error</h2>
<p>Dataset physical units. Reference and predictions share a color scale across all five models for the selected case/channel; signed errors share a symmetric scale. White holes are excluded solid-module cells.</p>
<div class="controls">
<label>anchor <select id="anchor_case">{case_options}</select></label>
<label>model <select id="anchor_model">{model_options}</select></label>
<label>channel <select id="anchor_channel">{channel_options}</select></label>
</div>
<div id="anchor_maps"></div>
</section>
<div class="grid2">
{''.join(cards)}
</div>
<details><summary>Inspect source artifacts and definitions</summary>
<p>Normalized source files are written next to this HTML so every chart can be audited without checkpoint copies.</p>
<ul>{links}</ul>
<ul>{notes}</ul>
</details>
</main>
<script>{plotly}</script>
<script>
const figures = {js(figures)};
for (const [key, figure] of Object.entries(figures)) {{
  const target = document.getElementById('fig_' + key);
  if (target && figure && figure.data && figure.data.length) {{
    Plotly.newPlot(target, figure.data, figure.layout, {{responsive:true, displaylogo:false}});
  }}
}}
</script>
<script>{anchor_javascript(anchors)}</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")


def normalize_sources(output_dir: Path, reduction: dict[str, list[dict[str, str]]], history: list[dict[str, Any]], timing: list[dict[str, Any]], organization: list[dict[str, Any]], interventions: list[dict[str, Any]], anchors: dict[str, Any], best_rows: list[dict[str, str]]) -> list[str]:
    channel_rows = [
        row for row in exact_rows(reduction.get("pooled_metrics", []))
        if string(row.get("base")).startswith("field_") and "_fluid_norm" in string(row.get("base"))
    ]
    source_files = {
        "summary_source.csv": exact_rows(reduction.get("headline", [])),
        "paired_source.csv": paired_rows(reduction.get("paired_case_deltas", [])),
        "pooled_source.csv": exact_rows(reduction.get("pooled_metrics", [])),
        "channel_source.csv": channel_rows,
        "channel_delta_source.csv": paired_rows(reduction.get("channel_mse_contributions", [])),
        "best_selected_source.csv": best_rows,
        "history_source.csv": history,
        "timing_source.csv": timing,
        "organization_source.csv": organization,
        "intervention_source.csv": interventions,
    }
    links = []
    for name, rows in source_files.items():
        write_csv(output_dir / name, rows)
        links.append(name)
    (output_dir / "anchors_source.json").write_text(json.dumps(jsonable(anchors), ensure_ascii=False), encoding="utf-8")
    links.append("anchors_source.json")
    return links


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plotly-js", type=Path, default=None)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if (project_root / "HONF_Proj").is_dir() and not (project_root / "diagnostics").is_dir():
        project_root = project_root / "HONF_Proj"
    input_dir = project_root / "diagnostics/generated/interface_operator_study/five_model_epoch5000"
    output_dir = (args.output_dir or input_dir / "figures").resolve()
    reduction = load_reduction(input_dir)
    figures: dict[str, dict[str, Any]] = {}
    endpoint = endpoint_plots(reduction)
    figures.update(endpoint)
    paired = paired_plots(reduction)
    figures["paired_distribution"] = paired["distribution"]
    figures["paired_summary"] = paired["summary"]
    best = best_selected_plots(input_dir)
    figures["best_selected"] = best["plot"]
    history = history_rows(input_dir)
    history_figures = history_plots(history)
    figures.update({
        "history_field_epoch": history_figures.get("val_field_mse_epoch"),
        "history_field_time": history_figures.get("val_field_mse_time"),
        "history_temperature_epoch": history_figures.get("val_temperature_mse_epoch"),
        "history_temperature_time": history_figures.get("val_temperature_mse_time"),
    })
    timing = timing_rows(input_dir)
    timing_figures = timing_plots(timing, reduction)
    figures.update({
        "timing_pareto": timing_figures["pareto"],
        "timing_synthetic": timing_figures["synthetic"],
        "timing_memory": timing_figures["memory"],
    })
    anchors, organization = load_anchors(project_root)
    figures["organization"] = organization_plot(organization)
    intervention_figure, interventions = intervention_plot(input_dir)
    figures["intervention"] = intervention_figure
    output_dir.mkdir(parents=True, exist_ok=True)
    links = normalize_sources(output_dir, reduction, history, timing, organization, interventions, anchors, best["rows"])
    source_notes = [
        "Primary endpoint rows are reducer headline.csv filtered to integer epoch 5000.",
        "Paired distributions use matched case deltas and retain the candidate-minus-baseline sign.",
        "History epoch and cumulative-time panels use different x-axis definitions from the logged trajectory/checkpoint rows.",
        "Real-anchor timing is median full-forward time across measured anchors; synthetic timing uses the largest recorded query shape.",
        "Organization labels preserve exported family semantics; Reader has no full query-incidence matrix here, so no incidence heatmap is fabricated.",
        "Evaluation uses one seed and a development holdout; 16 solver requests remain pending.",
    ]
    plotly_js = plotly_source(args.plotly_js)
    render_html(output_dir / "index.html", figures, anchors, links, source_notes, plotly_js)
    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "renderer": str(Path(__file__).resolve()),
        "primary_scope": "exact integer epoch 5000, 90 matched cases",
        "models": MODEL_ORDER,
        "anchor_cases": ANCHOR_CASES,
        "source_files": links,
        "panel_keys": sorted(figures),
        "plotly_embedded": bool(plotly_js),
    }
    (output_dir / "figure_source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {output_dir / 'index.html'}")
    print(f"normalized sources: {len(links)}")
    print(f"history rows: {len(history)}; timing rows: {len(timing)}; organization rows: {len(organization)}")


if __name__ == "__main__":
    main()
