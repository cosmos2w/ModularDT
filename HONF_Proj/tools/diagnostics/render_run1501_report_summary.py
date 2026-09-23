"""Render compact, evidence-backed Run-1501 report summary figures.

The evaluator writes large per-case tables and population arrays.  This
renderer consumes the maintained evaluator artifacts, including per-case SSE
terms needed for a true pooled relative-L2, so that the report figures can be
regenerated after each endpoint without rerunning a model.  It discovers all
available exact checkpoints under one Run-1501 run
directory, so the same command can render the epoch-50/150 interim view and
the later epoch-500 view.

The figures keep three distinctions visible:

* checkpoint accuracy is read from exact evaluator tables, not sampled
  validation loss;
* unique logical support is plotted separately from rectangular executor
  rows; and
* a selected executor is compared with the rectangular implementation using
  measured latency, memory, and the reported prediction difference.

No model, checkpoint, or dataset is loaded by this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPOCH_RE = re.compile(r"epoch[_-]?0*(\d+)", flags=re.IGNORECASE)
MODEL_TOKENS = ("Run1501", "Run1406", "Run1404", "Run1804", "Run1500")
MODEL_LABELS = {
    "Run1501": "Run 1501 sparse",
    "Run1406": "Run 1406",
    "Run1404": "Run 1404",
    "Run1804": "Run 1804 dense",
    "Run1500": "Run 1500 mass",
}
MODEL_COLORS = {
    "Run1501": "#D2691E",
    "Run1406": "#2563A6",
    "Run1404": "#7B4BA0",
    "Run1804": "#4C78A8",
    "Run1500": "#6B7280",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean_summary(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, dict):
        return _number(value.get("mean"))
    return _number(value)


def _epoch_hint(*values: Any) -> int | None:
    """Extract the first explicit ``epochNNN`` marker from a path or label."""

    for value in values:
        if value is None:
            continue
        match = EPOCH_RE.search(str(value))
        if match:
            return int(match.group(1))
    return None


def _model_key(row: dict[str, str]) -> str | None:
    text = f"{row.get('model_label', '')} {row.get('checkpoint', '')}"
    for token in MODEL_TOKENS:
        if token in text:
            return token
    return None


def _metrics_path(run_dir: Path) -> Path | None:
    for candidate in (run_dir / "metrics.csv", run_dir / "metrics" / "metrics.csv"):
        if candidate.is_file():
            return candidate
    return None


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _axes_style(axes: Iterable[Any]) -> None:
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(alpha=0.2)


def _plot_training(run_dir: Path, output: Path) -> dict[str, Any]:
    """Plot exact recorded train/validation curves and router state signals."""

    source = _metrics_path(run_dir)
    if source is None:
        return {"status": "unavailable", "reason": "metrics.csv not found"}
    rows = _read_csv(source)
    epochs = np.asarray([_number(row.get("epoch")) for row in rows], dtype=float)
    finite_epoch = np.isfinite(epochs)
    epochs = epochs[finite_epoch]
    rows = [row for row, keep in zip(rows, finite_epoch, strict=True) if keep]
    if epochs.size == 0:
        return {"status": "unavailable", "reason": "metrics.csv has no finite epochs"}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    fig.suptitle("Run 1501 recorded training trajectory", fontsize=16, fontweight="bold")
    series = (
        (axes[0, 0], ("loss_total", "val_loss_total"), ("train total", "validation total"), "Total loss"),
        (axes[0, 1], ("loss_field", "val_loss_field"), ("train field", "validation field"), "Field loss"),
        (axes[1, 0], ("temperature_mse", "val_temperature_mse"), ("train temperature MSE", "validation temperature MSE"), "Temperature MSE"),
    )
    for axis, keys, labels, title in series:
        for key, label, color in zip(keys, labels, ("#2563A6", "#D2691E"), strict=True):
            values = np.asarray([_number(row.get(key)) for row in rows], dtype=float)
            keep = np.isfinite(values) & (values > 0)
            if keep.any():
                axis.plot(epochs[keep], values[keep], lw=1.25, color=color, label=label)
        axis.set(title=title, xlabel="Epoch", ylabel="value", yscale="log")
        axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    for key, label, color in (
        ("train_wall_seconds", "training wall time", "#2563A6"),
        ("val_wall_seconds", "validation wall time", "#D2691E"),
    ):
        values = np.asarray([_number(row.get(key)) for row in rows], dtype=float)
        keep = np.isfinite(values) & (values > 0)
        if keep.any():
            axis.plot(epochs[keep], values[keep], lw=1.15, color=color, label=label)
    axis.set(title="Measured per-epoch runtime", xlabel="Epoch", ylabel="Seconds", yscale="log")
    axis.legend(frameon=False, fontsize=8)
    _axes_style(axes.flat)
    _save(fig, output)
    return {
        "status": "complete",
        "source": str(source.resolve()),
        "epochs": [int(value) for value in epochs],
        "output": str(output.resolve()),
    }


def _comparison_rows(eval_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for table in sorted(eval_dir.glob("*/tables/model_summary_metrics.csv")):
        try:
            rows = _read_csv(table)
        except OSError:
            continue
        per_case_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
        per_case_path = table.with_name("per_case_metrics.csv")
        if per_case_path.is_file():
            for case_row in _read_csv(per_case_path):
                per_case_by_label[case_row.get("model_label", "")].append(case_row)
        for row in rows:
            model = _model_key(row)
            epoch = _epoch_hint(row.get("checkpoint"), row.get("model_label"), table.parent.parent.name)
            if model is None or epoch is None:
                continue
            value: float | None = None
            metric_key = "sqrt(sum(global_field_fluid_norm_sse)/sum(global_field_fluid_norm_target_sse))"
            case_rows = per_case_by_label.get(row.get("model_label", ""), [])
            numerator = sum(
                candidate
                for case_row in case_rows
                if (candidate := _number(case_row.get("global_field_fluid_norm_sse"))) is not None
            )
            denominator = sum(
                candidate
                for case_row in case_rows
                if (candidate := _number(case_row.get("global_field_fluid_norm_target_sse"))) is not None
            )
            if case_rows and denominator > 0.0:
                value = math.sqrt(numerator / denominator)
            if value is None:
                continue
            records.append(
                {
                    "model": model,
                    "epoch": epoch,
                    "value": value,
                    "metric_key": metric_key,
                    "label": row.get("model_label", model),
                    "checkpoint": row.get("checkpoint", ""),
                    "table": str(table.resolve()),
                }
            )
    return records


def _cost_rows(eval_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for table in sorted(eval_dir.glob("*/tables/evaluation_cost_summary_metrics.csv")):
        try:
            rows = _read_csv(table)
        except OSError:
            continue
        for row in rows:
            model = _model_key(row)
            epoch = _epoch_hint(row.get("checkpoint"), row.get("model_label"), table.parent.parent.name)
            latency = _number(row.get("evaluation_wall_time_seconds_mean"))
            memory = _number(row.get("evaluation_cuda_incremental_peak_allocated_mib_mean"))
            if model is None or epoch is None or latency is None or memory is None:
                continue
            records.append(
                {
                    "model": model,
                    "epoch": epoch,
                    "latency_seconds": latency,
                    "memory_mib": memory,
                    "label": row.get("model_label", model),
                    "checkpoint": row.get("checkpoint", ""),
                    "table": str(table.resolve()),
                }
            )
    return records


def _deduplicate(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(key) for key in keys)].append(record)
    result = []
    for group in grouped.values():
        # Matched and candidate tables repeat the same exact checkpoint.  Keep
        # the first finite value while retaining every source in the manifest.
        result.append(group[0])
    return sorted(result, key=lambda item: (str(item.get("model")), int(item.get("epoch", 0))))


def _plot_accuracy(run_dir: Path, output: Path) -> dict[str, Any]:
    records = _deduplicate(_comparison_rows(run_dir / "evaluations"), ("model", "epoch"))
    costs = _deduplicate(_cost_rows(run_dir / "evaluations"), ("model", "epoch"))
    if not records:
        return {"status": "unavailable", "reason": "no model_summary_metrics.csv records found"}

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), constrained_layout=True)
    fig.suptitle("Run 1501 exact-checkpoint accuracy and measured evaluation cost", fontsize=16, fontweight="bold")
    axis = axes[0]
    for model in MODEL_TOKENS:
        selected = [record for record in records if record["model"] == model]
        if not selected:
            continue
        selected.sort(key=lambda item: item["epoch"])
        axis.plot(
            [item["epoch"] for item in selected],
            [item["value"] for item in selected],
            marker="o",
            lw=2.0 if model == "Run1501" else 1.15,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        if model == "Run1501":
            for item in selected:
                if item["epoch"] >= 400:
                    suffix = "best total" if "best" in item["label"].lower() else "exact"
                    axis.annotate(
                        f"{suffix} e{item['epoch']}",
                        (item["epoch"], item["value"]),
                        xytext=(-52 if suffix == "best total" else 6, 9 if suffix == "best total" else -15),
                        textcoords="offset points",
                        fontsize=7,
                    )
    axis.set(
        title="Matched full-grid fluid fidelity",
        xlabel="Exact checkpoint epoch",
        ylabel="Pooled relative L2 (lower is better)",
        yscale="log",
    )
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    label_offsets = {
        ("Run1501", 50): (-22, 9),
        ("Run1501", 150): (-30, -16),
        ("Run1501", 444): (-68, 10),
        ("Run1501", 500): (7, -17),
        ("Run1406", 50): (7, -16),
        ("Run1406", 500): (7, 9),
        ("Run1404", 50): (7, -16),
        ("Run1404", 500): (7, 9),
        ("Run1804", 50): (-40, -16),
        ("Run1804", 500): (7, 9),
        ("Run1500", 50): (7, 9),
    }
    for model in MODEL_TOKENS:
        selected = [record for record in costs if record["model"] == model]
        if not selected:
            continue
        color = MODEL_COLORS[model]
        axis.scatter(
            [1000.0 * item["latency_seconds"] for item in selected],
            [item["memory_mib"] for item in selected],
            s=62 if model != "Run1501" else 86,
            color=color,
            label=MODEL_LABELS[model],
        )
        for item in selected:
            offset = label_offsets.get((item["model"], item["epoch"]), (5, 5))
            axis.annotate(
                f"{item['model'].replace('Run', 'R')} e{item['epoch']}",
                (1000.0 * item["latency_seconds"], item["memory_mib"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
            )
    axis.set(
        title="Measured evaluation cost",
        xlabel="Mean synchronized case time (ms)",
        ylabel="Mean incremental peak CUDA memory (MiB)",
    )
    axis.set_ylim(bottom=50.0)
    axis.legend(frameon=False, fontsize=8)
    _axes_style(axes)
    _save(fig, output)
    return {
        "status": "complete",
        "accuracy_records": records,
        "cost_records": costs,
        "output": str(output.resolve()),
    }


def _population_records(eval_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(eval_dir.glob("*/population_summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        epoch = _epoch_hint(path.parent.name)
        if epoch is None:
            continue
        q_values = summary.get("query_count_per_case", [])
        q_per_case = _number(q_values[0]) if q_values else None
        capacity = _number(summary.get("registered_capacity"))
        module_dense = _mean_summary(summary, "module_dense_valid_pairs")
        environment_dense = _mean_summary(summary, "environment_dense_valid_pairs")
        module_actual = _mean_summary(summary, "p2_module_actual_rows")
        environment_actual = _mean_summary(summary, "p2_environment_actual_rows")
        module_unique = _mean_summary(summary, "module_unique_pairs")
        environment_unique = _mean_summary(summary, "environment_unique_pairs")
        records.append(
            {
                "epoch": epoch,
                "path": str(path.resolve()),
                "summary": summary,
                "q_per_case": q_per_case,
                "capacity": capacity,
                "module_support": _mean_summary(summary, "module_RM_support"),
                "environment_support": _mean_summary(summary, "environment_RE_support"),
                "kq_mean": _mean_summary(summary, "query_degree_mean"),
                "effective_groups": _mean_summary(summary, "query_effective_groups_mean"),
                "module_unique": module_unique,
                "environment_unique": environment_unique,
                "module_dense": module_dense,
                "environment_dense": environment_dense,
                "module_actual": module_actual,
                "environment_actual": environment_actual,
                "module_unique_fraction": (module_unique / module_dense if module_unique is not None and module_dense else None),
                "environment_unique_fraction": (environment_unique / environment_dense if environment_unique is not None and environment_dense else None),
                "module_rectangular_fraction": (module_actual / (q_per_case * capacity) if module_actual is not None and q_per_case and capacity else None),
                "environment_rectangular_fraction": (environment_actual / environment_dense if environment_actual is not None and environment_dense else None),
            }
        )
    return sorted(records, key=lambda item: item["epoch"])


def _plot_support(run_dir: Path, output: Path) -> dict[str, Any]:
    records = _population_records(run_dir / "evaluations")
    if not records:
        return {"status": "unavailable", "reason": "no population_summary.json records found"}
    epochs = np.asarray([record["epoch"] for record in records], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)
    fig.suptitle("Run 1501 logical support versus rectangular execution", fontsize=16, fontweight="bold")

    axis = axes[0, 0]
    for key, label, color in (
        ("module_unique_fraction", "module unique / dense valid pairs", "#2563A6"),
        ("environment_unique_fraction", "environment unique / dense valid pairs", "#59A14F"),
        ("module_rectangular_fraction", "module rectangular rows / Q·K", "#D2691E"),
        ("environment_rectangular_fraction", "environment rectangular rows / dense rectangle", "#E45756"),
    ):
        values = np.asarray([record.get(key, np.nan) for record in records], dtype=float)
        keep = np.isfinite(values)
        if keep.any():
            axis.plot(epochs[keep], values[keep], marker="o", lw=1.6, color=color, label=label)
    axis.axhline(1.0, color="#444444", lw=0.8, ls="--")
    axis.set(title="Support fraction and executor occupancy", xlabel="Exact checkpoint epoch", ylabel="Fraction", ylim=(0.0, 1.08))
    axis.legend(frameon=False, fontsize=7)

    axis = axes[0, 1]
    for key, label, color in (("kq_mean", "mean Kq", "#7B4BA0"), ("effective_groups", "mean effective query groups", "#F28E2B")):
        values = np.asarray([record.get(key, np.nan) for record in records], dtype=float)
        keep = np.isfinite(values)
        if keep.any():
            axis.plot(epochs[keep], values[keep], marker="o", lw=1.6, color=color, label=label)
    axis.set(title="Query-local routing degree", xlabel="Exact checkpoint epoch", ylabel="Groups per query")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    for key, label, color in (("module_unique", "module unique pairs", "#2563A6"), ("environment_unique", "environment unique pairs", "#59A14F"), ("module_actual", "module rectangular rows", "#D2691E"), ("environment_actual", "environment rectangular rows", "#E45756")):
        values = np.asarray([record.get(key, np.nan) for record in records], dtype=float)
        keep = np.isfinite(values) & (values > 0)
        if keep.any():
            axis.plot(epochs[keep], values[keep], marker="o", lw=1.35, color=color, label=label)
    axis.set(title="Mean pair/row counts", xlabel="Exact checkpoint epoch", ylabel="Count", yscale="log")
    axis.legend(frameon=False, fontsize=7)

    axis = axes[1, 1]
    bins = sorted({int(key) for record in records for key in record["summary"].get("kq_histogram", {})})
    if bins:
        bottoms = np.zeros(len(records), dtype=float)
        for bucket in bins:
            fractions = []
            for record in records:
                histogram = record["summary"].get("kq_histogram", {})
                total = sum(float(value) for value in histogram.values())
                fractions.append(float(histogram.get(str(bucket), 0.0)) / total if total else np.nan)
            values = np.asarray(fractions, dtype=float)
            axis.bar(epochs, values, bottom=bottoms, width=max(4.0, 0.58 * max(1.0, np.ptp(epochs) / max(len(records), 1))), label=f"Kq={bucket}")
            bottoms += np.nan_to_num(values, nan=0.0)
        axis.set(title="Pooled query-support degree distribution", xlabel="Exact checkpoint epoch", ylabel="Fraction of queries", ylim=(0.0, 1.05))
        axis.legend(frameon=False, fontsize=7, ncols=2)
    else:
        axis.text(0.5, 0.5, "Kq histograms unavailable", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    _axes_style(axes.flat)
    _save(fig, output)
    return {
        "status": "complete",
        "population_records": records,
        "output": str(output.resolve()),
    }


def _selected_records(eval_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(eval_dir.glob("selected_executor_epoch*_q*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        epoch = _number(payload.get("checkpoint_epoch"))
        if epoch is None:
            epoch = _epoch_hint(path.name)
        if epoch is None:
            continue
        for row in payload.get("rows", []):
            records.append(
                {
                    "epoch": int(epoch),
                    "case_id": str(row.get("case_id", "unknown")),
                    "latency_ratio": _number(row.get("selected_over_rectangular_median_latency")),
                    "memory_ratio": _number(row.get("selected_over_rectangular_incremental_peak_allocated")),
                    "prediction_max_difference": _number(row.get("prediction_max_abs_difference")),
                    "source": str(path.resolve()),
                }
            )
    return sorted(records, key=lambda item: (item["epoch"], item["case_id"]))


def _plot_selected(run_dir: Path, output: Path) -> dict[str, Any]:
    records = _selected_records(run_dir / "evaluations")
    if not records:
        return {"status": "unavailable", "reason": "no selected_executor_epoch*_q*.json records found"}
    labels = [f"e{record['epoch']}\n{record['case_id']}" for record in records]
    x = np.arange(len(records))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.3), constrained_layout=True)
    fig.suptitle("Selected versus rectangular executor benchmark", fontsize=16, fontweight="bold")
    for axis, key, title, ylabel in (
        (axes[0], "latency_ratio", "Median latency ratio", "selected / rectangular"),
        (axes[1], "memory_ratio", "Incremental peak memory ratio", "selected / rectangular"),
        (axes[2], "prediction_max_difference", "Reported prediction difference", "maximum absolute difference"),
    ):
        values = np.asarray([record.get(key, np.nan) for record in records], dtype=float)
        colors = ["#D2691E" if record["epoch"] == max(item["epoch"] for item in records) else "#4C78A8" for record in records]
        axis.bar(x, values, color=colors, alpha=0.9)
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.2)
        if key != "prediction_max_difference":
            axis.axhline(1.0, color="#444444", lw=0.9, ls="--", label="equal to rectangular")
            axis.legend(frameon=False, fontsize=8)
    _axes_style(axes)
    _save(fig, output)
    return {
        "status": "complete",
        "records": records,
        "output": str(output.resolve()),
    }


def render(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Render all available summary figures and write a source manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run_dir": str(run_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "figures": {
            "training": _plot_training(run_dir, output_dir / "training_curves.png"),
            "accuracy": _plot_accuracy(run_dir, output_dir / "accuracy_evolution_and_comparators.png"),
            "support": _plot_support(run_dir, output_dir / "support_vs_rectangular_execution.png"),
            "selected_executor": _plot_selected(run_dir, output_dir / "selected_vs_rectangular_benchmark.png"),
        },
        "limitations": [
            "Accuracy curves use exact evaluator tables and are not sampled validation MSE.",
            "Support fractions describe logical unique pairs; rectangular rows are the maintained executor work and are not inferred speedups.",
            "Group labels are permutation-ambiguous; plots do not establish physical causality.",
            "A selected executor benchmark is omitted until its JSON artifact exists.",
        ],
    }
    manifest = output_dir / "summary_manifest.json"
    result["manifest"] = str(manifest.resolve())
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Run-1501 managed run directory")
    parser.add_argument("--output-dir", type=Path, help="Directory for report figures (default: run/evaluations/summary_figures)")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "evaluations" / "summary_figures")).resolve()
    result = render(run_dir, output_dir)
    print(json.dumps({"manifest": result["manifest"], "figure_status": {key: value["status"] for key, value in result["figures"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render"]
