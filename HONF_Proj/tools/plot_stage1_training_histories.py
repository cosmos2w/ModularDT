#!/usr/bin/env python3
"""Plot aligned HONF histories and optional group-reader recovery diagnostics.

The optional reader-recovery figure deliberately keeps the historical checkpoint
audit and the candidate training history on separate panels.  The audit is a
fixed four-case backward replay; the history is the sampled-batch trajectory
recorded during a candidate run.  Keeping those grains visible avoids implying
that the two curves are matched-batch measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-stage1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METRICS = ("loss_total", "field_mse", "val_loss_total", "val_field_mse")
READER_GRADIENT_METRICS = (
    "group_prepare_gradient_norm",
    "group_receiver_gradient_norm",
    "coarse_gradient_norm",
    "local_gradient_norm",
)
READER_HISTORY_METRICS = (
    "interaction_group_read_geometric_availability_mean",
    "interaction_group_read_weight_mass_mean",
    "preclip_gradient_norm_group_prepare",
    "preclip_gradient_norm_group_receiver",
    "preclip_gradient_norm_coarse",
    "preclip_gradient_norm_local",
)
READER_PHASE_LABELS = {
    "P0_initial_port": "P0",
    "P1_refinement": "P1",
    "P2_field": "P2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="LABEL=METRICS_CSV",
        help="Named history; repeat in desired display order.",
    )
    parser.add_argument("--max-epoch", type=int, default=500)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument(
        "--reader-recovery-audit",
        type=Path,
        help="Run-1802 checkpoint-audit JSON for an optional recovery figure.",
    )
    parser.add_argument(
        "--reader-recovery-history",
        action="append",
        default=[],
        metavar="LABEL=METRICS_CSV",
        help=(
            "Candidate history for the optional recovery figure; repeat for "
            "additional sampled-batch histories."
        ),
    )
    parser.add_argument(
        "--reader-recovery-output-figure",
        type=Path,
        help="Output path for the optional group-reader recovery figure.",
    )
    parser.add_argument(
        "--accuracy-cost-summary",
        action="append",
        default=[],
        type=Path,
        help="model_summary_metrics.csv input for the optional accuracy-cost figure; repeat as needed.",
    )
    parser.add_argument(
        "--accuracy-cost-inference-summary",
        action="append",
        default=[],
        type=Path,
        help="evaluation_cost_summary_metrics.csv input; repeat as needed.",
    )
    parser.add_argument(
        "--accuracy-cost-timing",
        action="append",
        default=[],
        type=Path,
        help="Optional timing JSON fallback for inference cost; repeat as needed.",
    )
    parser.add_argument(
        "--accuracy-cost-run-metrics",
        action="append",
        default=[],
        metavar="LABEL=METRICS_CSV",
        help="Optional run metrics override for measured training cost; repeat as needed.",
    )
    parser.add_argument(
        "--accuracy-cost-metric",
        default="global_field_fluid_pooled_relative_l2",
        help=(
            "Accuracy column in model_summary_metrics.csv "
            "(default: pooled global fluid field relative L2)."
        ),
    )
    parser.add_argument(
        "--accuracy-cost-output-figure",
        type=Path,
        help="Output path for the optional accuracy-cost figure.",
    )
    parser.add_argument(
        "--accuracy-cost-inference-axis-label",
        default="mean evaluation wall time per case (s)",
        help="Optional x-axis label for the inference-cost panel.",
    )
    return parser.parse_args()


def _split_label_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected LABEL=PATH, got {raw!r}.")
    label, raw_path = raw.split("=", 1)
    return label, Path(raw_path)


def _load_history(
    raw: str, max_epoch: int, metrics: tuple[str, ...]
) -> tuple[str, list[dict[str, float]]]:
    label, path = _split_label_path(raw)
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            epoch = int(float(source["epoch"]))
            if epoch > max_epoch:
                continue
            row = {"epoch": float(epoch)}
            for metric in metrics:
                value = float(source.get(metric, "nan"))
                row[metric] = value
            rows.append(row)
    if not rows or int(rows[-1]["epoch"]) != max_epoch:
        raise ValueError(f"{path} does not contain the requested epoch {max_epoch} endpoint.")
    return label, rows


def load_series(raw: str, max_epoch: int) -> tuple[str, list[dict[str, float]]]:
    return _load_history(raw, max_epoch, METRICS)


def load_reader_history(raw: str, max_epoch: int) -> tuple[str, list[dict[str, float]]]:
    return _load_history(raw, max_epoch, ("field_mse", "val_field_mse", *READER_HISTORY_METRICS))


def load_reader_audit(path: Path) -> list[dict[str, Any]]:
    """Load the reduced, equal-case rows from the existing checkpoint audit."""

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    rows = payload.get("reduced_diagnosis_table")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no reduced_diagnosis_table rows.")
    required = {
        "epoch",
        "sample_kind",
        "phase",
        "availability_mean",
        "nonnull_mass_mean",
        *READER_GRADIENT_METRICS,
    }
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"{path} contains a reduced row missing reader metrics.")
    return rows


def _mean_reader_rows(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[tuple[int, str, str], dict[str, float]]:
    """Average equal-case rows while retaining epoch, split, and phase labels."""

    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["epoch"]), str(row["sample_kind"]), str(row["phase"]))
        grouped.setdefault(key, []).append(row)
    means: dict[tuple[int, str, str], dict[str, float]] = {}
    for key, members in grouped.items():
        means[key] = {
            metric: sum(float(member[metric]) for member in members) / len(members)
            for metric in keys
        }
    return means


def _mean_reader_epoch(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[int, dict[str, float]]:
    """Average the fixed-batch gradient rows across the audited cases/phases."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["epoch"]), []).append(row)
    return {
        epoch: {
            metric: sum(float(member[metric]) for member in members) / len(members)
            for metric in keys
        }
        for epoch, members in grouped.items()
    }


def _finite_float(row: dict[str, Any], field: str) -> float | None:
    try:
        value = float(row.get(field, "nan"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load_summary_rows(paths: list[Path]) -> dict[str, dict[str, str]]:
    """Merge model summaries in argument order, keeping the latest row per label."""

    summaries: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                label = str(row.get("model_label", "")).strip()
                if label:
                    summaries[label] = row
    if not summaries:
        raise ValueError("No model rows were found in the accuracy-cost summaries.")
    return summaries


def _load_inference_costs(paths: list[Path]) -> dict[str, float]:
    costs: dict[str, float] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                label = str(row.get("model_label", "")).strip()
                value = _finite_float(row, "evaluation_wall_time_seconds_mean")
                if label and value is not None and value > 0.0:
                    costs[label] = value
    return costs


def _timing_cost(path: Path, summaries: dict[str, dict[str, str]]) -> tuple[str, float] | None:
    """Read a full-forward timing JSON, matching its checkpoint to a model row."""

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    label = str(payload.get("model_label", "")).strip()
    checkpoint = str(payload.get("checkpoint", ""))
    if not label and checkpoint:
        for candidate, row in summaries.items():
            run_dir = str(row.get("run_dir", ""))
            if run_dir and run_dir in checkpoint:
                label = candidate
                break
    if not label:
        return None
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    full_forward = [row for row in rows if row.get("phase") == "full_forward"]
    selected = full_forward or rows
    values = []
    for row in selected:
        value = _finite_float(row, "median_ms")
        if value is None:
            value = _finite_float(row, "mean_ms")
        if value is not None and value > 0.0:
            values.append(value / 1000.0)
    if not values:
        value = _finite_float(payload, "evaluation_wall_time_seconds_mean")
        if value is None or value <= 0.0:
            return None
        return label, value
    return label, sum(values) / len(values)


def _load_training_costs(
    summaries: dict[str, dict[str, str]], overrides: list[str], max_epoch: int
) -> dict[str, float]:
    paths: dict[str, Path] = {}
    for raw in overrides:
        label, path = _split_label_path(raw)
        paths[label] = path
    costs: dict[str, float] = {}
    for label, row in summaries.items():
        path = paths.get(label)
        if path is None:
            run_dir = str(row.get("run_dir", "")).strip()
            path = Path(run_dir) / "metrics.csv" if run_dir else None
        if path is None or not path.is_file():
            continue
        total = 0.0
        seen = False
        with path.open(newline="", encoding="utf-8") as stream:
            for metric_row in csv.DictReader(stream):
                epoch = _finite_float(metric_row, "epoch")
                if epoch is not None and epoch > max_epoch:
                    continue
                for wall_key in ("train_wall_seconds", "val_wall_seconds"):
                    value = _finite_float(metric_row, wall_key)
                    if value is not None and value >= 0.0:
                        total += value
                        seen = True
        if seen and total > 0.0:
            costs[label] = total
    return costs


def _short_model_label(label: str) -> str:
    short = label.replace(" legacy HONF", " legacy")
    short = short.replace(" adaptation", "")
    short = short.replace(" @500", "")
    short = short.replace(" mature best-field @4585", " mature @4585")
    return short


def plot_accuracy_cost(
    summary_paths: list[Path],
    inference_paths: list[Path],
    timing_paths: list[Path],
    run_metric_overrides: list[str],
    accuracy_metric: str,
    max_epoch: int,
    output_path: Path,
    inference_axis_label: str = "mean evaluation wall time per case (s)",
) -> None:
    """Compare endpoint error with measured inference and training wall cost."""

    summaries = _load_summary_rows(summary_paths)
    inference_costs = _load_inference_costs(inference_paths)
    for timing_path in timing_paths:
        timing = _timing_cost(timing_path, summaries)
        if timing is not None:
            label, value = timing
            inference_costs.setdefault(label, value)
    training_costs = _load_training_costs(summaries, run_metric_overrides, max_epoch)
    records = []
    for label, row in summaries.items():
        accuracy = _finite_float(row, accuracy_metric)
        if accuracy is None or accuracy <= 0.0:
            continue
        records.append(
            {
                "label": label,
                "accuracy": accuracy,
                "inference": inference_costs.get(label),
                "training": training_costs.get(label),
            }
        )
    if not records:
        raise ValueError(f"No finite {accuracy_metric!r} values were found in model summaries.")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.23, top=0.78, wspace=0.32)
    palette = ["#2d6a9f", "#b27a17", "#b44c6b", "#568c68", "#75507b", "#6c757d"]
    panels = (
        (axes[0], "inference", "Measured inference cost", inference_axis_label),
        (
            axes[1],
            "training",
            f"Measured train + validation cost through epoch {max_epoch}",
            f"cumulative train + validation wall time through epoch {max_epoch} (s)",
        ),
    )
    accuracy_label = (
        "pooled global fluid field relative L2"
        if accuracy_metric == "global_field_fluid_pooled_relative_l2"
        else "mean global fluid field relative L2"
        if accuracy_metric == "global_field_fluid_norm_l2_mean"
        else accuracy_metric
    )
    for axis, cost_key, title, xlabel in panels:
        points = [row for row in records if row[cost_key] is not None and row[cost_key] > 0.0]
        if not points:
            axis.text(0.5, 0.5, "measured cost not supplied", ha="center", va="center")
            axis.set_axis_off()
            continue
        for index, row in enumerate(points):
            color = palette[index % len(palette)]
            axis.scatter(row[cost_key], row["accuracy"], color=color, s=36, zorder=3)
            axis.annotate(
                _short_model_label(row["label"]),
                (row[cost_key], row["accuracy"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                color="#333333",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        costs = [row[cost_key] for row in points]
        low_cost = min(costs)
        high_cost = max(costs)
        if low_cost == high_cost:
            axis.set_xlim(low_cost * 0.5, high_cost * 2.0)
        else:
            axis.set_xlim(low_cost * 0.72, high_cost * 1.55)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(accuracy_label if axis is axes[0] else "")
        axis.set_title(title)
        axis.grid(alpha=0.22)
    fig.suptitle("Endpoint field error versus measured cost", fontsize=13)
    fig.text(
        0.5,
        0.045,
        f"Accuracy: {accuracy_label}; lower is better. Costs are measured through epoch {max_epoch}; training sums train + validation wall time.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reader_recovery(
    audit_path: Path,
    histories: list[tuple[str, list[dict[str, float]]]],
    history_series: list[tuple[str, list[dict[str, float]]]],
    output_path: Path,
) -> None:
    """Plot fixed-case collapse beside sampled-batch reader recovery."""

    audit_rows = load_reader_audit(audit_path)
    support_keys = ("availability_mean", "nonnull_mass_mean")
    phase_rows = _mean_reader_rows(audit_rows, support_keys)
    gradient_rows = _mean_reader_epoch(audit_rows, READER_GRADIENT_METRICS)
    old_epochs = sorted({key[0] for key in phase_rows})
    if not old_epochs:
        raise ValueError(f"{audit_path} has no audit epochs.")

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.9), constrained_layout=False)
    axes = list(axes.ravel())
    fig.subplots_adjust(left=0.06, right=0.95, bottom=0.16, top=0.88, wspace=0.34, hspace=0.48)

    phase_colors = {"P0": "#2d6a9f", "P1": "#b27a17", "P2": "#b44c6b"}
    split_styles = {"holdout": "-", "training": "--"}
    phase_markers = {"P0": "o", "P1": "s", "P2": "^"}
    support_ax = axes[0]
    support_series: dict[tuple[str, str], list[tuple[int, dict[str, float]]]] = {}
    for (epoch, sample_kind, phase), values in phase_rows.items():
        support_series.setdefault((sample_kind, phase), []).append((epoch, values))
    for (sample_kind, phase), series in sorted(support_series.items()):
        phase_label = READER_PHASE_LABELS.get(phase, phase)
        color = phase_colors.get(phase_label, "#3f3f46")
        linestyle = split_styles.get(sample_kind, ":")
        marker = phase_markers.get(phase_label, "o")
        series.sort()
        support_ax.plot(
            [epoch for epoch, _ in series],
            [values["availability_mean"] for _, values in series],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            alpha=0.88,
        )
    support_ax.set_xscale("log")
    support_ax.set_xticks(old_epochs)
    support_ax.set_xticklabels([str(epoch) for epoch in old_epochs])
    support_ax.set_ylim(0.0, 1.05)
    support_ax.set_xlabel("Run 1802 audit epoch")
    support_ax.set_ylabel("geometric availability G")
    support_ax.set_title("Run 1802 fixed-case G")
    support_ax.grid(alpha=0.22)
    support_ax.legend(
        handles=[
            Line2D([0], [0], color="#2d6a9f", marker="o", label="P0"),
            Line2D([0], [0], color="#b27a17", marker="s", label="P1"),
            Line2D([0], [0], color="#b44c6b", marker="^", label="P2"),
            Line2D([0], [0], color="#3f3f46", linestyle="-", label="holdout"),
            Line2D([0], [0], color="#3f3f46", linestyle="--", label="training"),
        ],
        frameon=False,
        fontsize=7,
        loc="lower left",
        ncol=2,
    )

    mass_ax = axes[1]
    for (sample_kind, phase), series in sorted(support_series.items()):
        phase_label = READER_PHASE_LABELS.get(phase, phase)
        color = phase_colors.get(phase_label, "#3f3f46")
        series.sort()
        mass_ax.plot(
            [epoch for epoch, _ in series],
            [max(values["nonnull_mass_mean"], 1e-300) for _, values in series],
            color=color,
            linestyle=split_styles.get(sample_kind, ":"),
            marker=phase_markers.get(phase_label, "o"),
            markersize=4,
            alpha=0.82,
        )
    mass_ax.set_xscale("log")
    mass_ax.set_xticks(old_epochs)
    mass_ax.set_xticklabels([str(epoch) for epoch in old_epochs])
    mass_ax.set_yscale("log")
    mass_ax.set_xlabel("Run 1802 audit epoch")
    mass_ax.set_ylabel("non-null read mass")
    mass_ax.set_title("Run 1802 fixed-case non-null mass")
    mass_ax.grid(alpha=0.22)

    gradient_ax = axes[2]
    gradient_colors = {
        "group_prepare_gradient_norm": "#2d6a9f",
        "group_receiver_gradient_norm": "#6f9fc5",
        "coarse_gradient_norm": "#b27a17",
        "local_gradient_norm": "#b44c6b",
    }
    gradient_labels = {
        "group_prepare_gradient_norm": "group preparation",
        "group_receiver_gradient_norm": "group receiver",
        "coarse_gradient_norm": "coarse",
        "local_gradient_norm": "local",
    }
    for metric in READER_GRADIENT_METRICS:
        gradient_ax.plot(
            sorted(gradient_rows),
            [gradient_rows[epoch][metric] for epoch in sorted(gradient_rows)],
            color=gradient_colors[metric],
            marker="o",
            markersize=4,
            label=gradient_labels[metric],
        )
    gradient_ax.set_xscale("log")
    gradient_ax.set_xticks(old_epochs)
    gradient_ax.set_xticklabels([str(epoch) for epoch in old_epochs])
    gradient_ax.set_yscale("log")
    gradient_ax.set_xlabel("Run 1802 audit epoch")
    gradient_ax.set_ylabel("gradient norm")
    gradient_ax.set_title("Run 1802 fixed-batch gradients")
    gradient_ax.grid(alpha=0.22)
    gradient_ax.legend(frameon=False, fontsize=7)

    candidate_colors = ["#2d6a9f", "#b27a17", "#b44c6b", "#568c68", "#75507b"]
    candidate_linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
    new_support_ax = axes[3]
    new_gradient_ax = axes[4]
    new_support_handles = []
    for index, (label, rows) in enumerate(history_series):
        color = candidate_colors[index % len(candidate_colors)]
        linestyle = candidate_linestyles[index % len(candidate_linestyles)]
        epochs = [row["epoch"] for row in rows]
        support_values = [row["interaction_group_read_geometric_availability_mean"] for row in rows]
        mass_values = [row["interaction_group_read_weight_mass_mean"] for row in rows]
        finite_support = [
            (epoch, value)
            for epoch, value in zip(epochs, support_values)
            if math.isfinite(value)
        ]
        finite_mass = [
            (epoch, max(value, 1e-300))
            for epoch, value in zip(epochs, mass_values)
            if math.isfinite(value) and value > 0.0
        ]
        if finite_support:
            new_support_ax.plot(
                [epoch for epoch, _ in finite_support],
                [value for _, value in finite_support],
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
            )
        if finite_mass:
            new_support_ax.plot(
                [epoch for epoch, _ in finite_mass],
                [value for _, value in finite_mass],
                color=color,
                linestyle=":",
                linewidth=1.2,
            )
        if finite_support or finite_mass:
            new_support_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, label=label))
        for metric, metric_color, metric_label in (
            ("preclip_gradient_norm_group_prepare", "#2d6a9f", "group preparation"),
            ("preclip_gradient_norm_group_receiver", "#6f9fc5", "group receiver"),
            ("preclip_gradient_norm_coarse", "#b27a17", "coarse"),
            ("preclip_gradient_norm_local", "#b44c6b", "local"),
        ):
            finite_gradient = [
                (epoch, value)
                for epoch, value in zip(epochs, [row[metric] for row in rows])
                if math.isfinite(value) and value > 0.0
            ]
            if finite_gradient:
                new_gradient_ax.plot(
                    [epoch for epoch, _ in finite_gradient],
                    [value for _, value in finite_gradient],
                    color=metric_color,
                    linestyle=linestyle,
                    linewidth=1.3,
                    label=f"{metric_label} ({label})" if len(history_series) > 1 else metric_label,
                )
    if history_series:
        new_support_ax.set_xscale("log")
        support_values = [
            value
            for _, rows in history_series
            for row in rows
            for value in (
                row["interaction_group_read_geometric_availability_mean"],
                row["interaction_group_read_weight_mass_mean"],
            )
            if math.isfinite(value)
        ]
        new_support_ax.set_ylim(0.0, max(1.05, max(support_values, default=1.0) * 1.1))
        new_support_ax.set_xlabel("candidate training epoch")
        new_support_ax.set_ylabel("mean G / read mass (sampled field receivers)")
        new_support_ax.set_title("Candidate sampled-batch mean G and mass")
        new_support_ax.grid(alpha=0.22)
        new_support_handles.extend(
            [
                Line2D([0], [0], color="#3f3f46", linestyle="-", label="G"),
                Line2D([0], [0], color="#3f3f46", linestyle=":", label="read mass"),
            ]
        )
        new_support_ax.legend(handles=new_support_handles, frameon=False, fontsize=7)

        new_gradient_ax.set_xscale("log")
        new_gradient_ax.set_yscale("log")
        new_gradient_ax.set_xlabel("candidate training epoch")
        new_gradient_ax.set_ylabel("pre-clip gradient norm")
        new_gradient_ax.set_title("Candidate sampled-batch gradients")
        new_gradient_ax.grid(alpha=0.22)
        new_gradient_ax.legend(frameon=False, fontsize=7)
    else:
        for axis, title in (
            (new_support_ax, "Candidate sampled-batch mean G and mass"),
            (new_gradient_ax, "Candidate sampled-batch gradients"),
        ):
            axis.text(0.5, 0.5, "candidate history not supplied", ha="center", va="center")
            axis.set_title(title)
            axis.set_axis_off()

    history_ax = axes[5]
    history_colors = ["#2d6a9f", "#b27a17", "#b44c6b", "#568c68", "#75507b", "#6c757d"]
    for index, (label, rows) in enumerate(histories):
        finite_validation = [
            (row["epoch"], row["val_field_mse"])
            for row in rows
            if math.isfinite(row["val_field_mse"]) and row["val_field_mse"] > 0.0
        ]
        if finite_validation:
            history_ax.plot(
                [epoch for epoch, _ in finite_validation],
                [value for _, value in finite_validation],
                color=history_colors[index % len(history_colors)],
                linewidth=1.4,
                label=label,
            )
    history_ax.set_xscale("log")
    history_ax.set_yscale("log")
    history_ax.set_xlabel("training epoch")
    history_ax.set_ylabel("development field MSE")
    history_ax.set_title("Validation field error across histories")
    history_ax.grid(alpha=0.22)
    if histories:
        history_ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Group-reader collapse and candidate recovery", fontsize=13)
    fig.text(
        0.5,
        0.045,
        "Top: Run 1802 fixed four-case backward replay; bottom: candidate sampled-batch means and gradients. "
        "The old and new panels are not matched-batch measurements.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.max_epoch < 1:
        raise ValueError("--max-epoch must be positive.")
    histories = [load_series(raw, args.max_epoch) for raw in args.series]
    if args.reader_recovery_output_figure and not args.reader_recovery_audit:
        raise ValueError("--reader-recovery-output-figure requires --reader-recovery-audit.")
    if args.reader_recovery_history and not args.reader_recovery_output_figure:
        raise ValueError("--reader-recovery-history requires --reader-recovery-output-figure.")
    reader_histories = [load_reader_history(raw, args.max_epoch) for raw in args.reader_recovery_history]
    accuracy_cost_requested = any(
        (
            args.accuracy_cost_summary,
            args.accuracy_cost_inference_summary,
            args.accuracy_cost_timing,
            args.accuracy_cost_run_metrics,
            args.accuracy_cost_output_figure,
        )
    )
    if accuracy_cost_requested and not args.accuracy_cost_summary:
        raise ValueError("Accuracy-cost inputs require at least one --accuracy-cost-summary.")
    if accuracy_cost_requested and not args.accuracy_cost_output_figure:
        raise ValueError("Accuracy-cost inputs require --accuracy-cost-output-figure.")
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    panels = (
        ("loss_total", "val_loss_total", "Total physical objective"),
        ("field_mse", "val_field_mse", "Field mean-squared error"),
    )
    for ax, (train_metric, val_metric, title) in zip(axes, panels):
        for label, rows in histories:
            epochs = [row["epoch"] for row in rows]
            line = ax.plot(epochs, [row[val_metric] for row in rows], label=f"{label} — development")[0]
            ax.plot(
                epochs,
                [row[train_metric] for row in rows],
                linestyle="--",
                alpha=0.55,
                color=line.get_color(),
                label=f"{label} — train",
            )
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(f"Matched training histories through epoch {args.max_epoch}")
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)

    fields = ["model_label", "metric", "endpoint_epoch", "endpoint_value", "best_epoch", "best_value"]
    with args.output_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, rows in histories:
            for metric in METRICS:
                finite = [row for row in rows if math.isfinite(row[metric])]
                if not finite:
                    continue
                best = min(finite, key=lambda row: row[metric])
                writer.writerow(
                    {
                        "model_label": label,
                        "metric": metric,
                        "endpoint_epoch": int(rows[-1]["epoch"]),
                        "endpoint_value": rows[-1][metric],
                        "best_epoch": int(best["epoch"]),
                        "best_value": best[metric],
                    }
                )
    if args.reader_recovery_output_figure:
        plot_reader_recovery(
            args.reader_recovery_audit,
            histories,
            reader_histories,
            args.reader_recovery_output_figure,
        )
    if accuracy_cost_requested:
        plot_accuracy_cost(
            args.accuracy_cost_summary,
            args.accuracy_cost_inference_summary,
            args.accuracy_cost_timing,
            args.accuracy_cost_run_metrics,
            args.accuracy_cost_metric,
            args.max_epoch,
            args.accuracy_cost_output_figure,
            inference_axis_label=args.accuracy_cost_inference_axis_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
