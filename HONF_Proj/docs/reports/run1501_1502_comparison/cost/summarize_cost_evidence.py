#!/usr/bin/env python3
"""Aggregate matched inference and historical training cost evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "inference_cost_cuda2.json"
SUMMARY_CSV = HERE / "model_cost_summary.csv"
PHASE_CSV = HERE / "inference_phase_summary.csv"
LEDGER_CSV = HERE / "execution_ledger_summary.csv"
FIGURE = HERE / "figures" / "inference_cost_comparison.png"
RUN_ORDER = ("1404", "1804", "1501", "1502")
PHASES = (
    ("application_evaluator", "predict_case"),
    ("full_forward", "full direct forward"),
    ("preparation_plus_one_query", "prepare + 1 query"),
    ("prepared_p2_decode", "prepared P2 decode"),
)


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def percentile(values: list[float], value: float) -> float | None:
    return float(np.percentile(values, value)) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ledger_counter_stats(ledgers: list[dict[str, Any]], key: str) -> tuple[float | None, float | None, float | None]:
    values = [float(ledger[key]) for ledger in ledgers if key in ledger]
    return mean(values), min(values) if values else None, max(values) if values else None


def group_control_ledger_mean(ledgers: list[dict[str, Any]], prefix: str, part: str, suffix: str) -> float | None:
    key = f"{prefix}{part}_{suffix}"
    values = [float(ledger[key]) for ledger in ledgers if key in ledger]
    return mean(values)


def phase_values(rows: list[dict[str, Any]], phase: str, key: str) -> list[float]:
    values = [row["phases"][phase].get(key) for row in rows]
    return [float(value) for value in values if value is not None]


def aggregate(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoints = data["checkpoints"]
    rows_by_run = {run: [row for row in data["rows"] if row["label"] == run] for run in RUN_ORDER}
    case_ids = [row["case_id"] for row in rows_by_run[RUN_ORDER[0]]]
    expected_ids = set(case_ids)
    for run, rows in rows_by_run.items():
        if len(rows) != len(expected_ids) or {row["case_id"] for row in rows} != expected_ids:
            raise ValueError(f"{run}: expected matched {len(expected_ids)} cases")
        if any(row["runtime_inner_receiver_chunk_override"] is not None for row in rows):
            raise ValueError(f"{run}: runtime receiver chunk override is present")
        if any(row["application_grid_query_count"] != row["direct_query_count"] for row in rows):
            raise ValueError(f"{run}: direct and application query counts differ")

    phase_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run in RUN_ORDER:
        rows = rows_by_run[run]
        checkpoint = checkpoints[run]
        training = checkpoint["training_summary"]
        per_phase: dict[str, dict[str, float | None]] = {}
        for phase_key, phase_label in PHASES:
            latency = phase_values(rows, phase_key, "median_wall_ms")
            p95_by_case = percentile(latency, 95)
            cuda_latency = phase_values(rows, phase_key, "median_cuda_event_ms")
            incremental_bytes = phase_values(rows, phase_key, "peak_incremental_allocated_bytes")
            incremental = [value / (1024.0**2) for value in incremental_bytes]
            baseline = [
                float(row["phases"][phase_key]["samples"][0]["baseline_allocated_bytes"]) / (1024.0**2)
                for row in rows
                if row["phases"][phase_key].get("samples")
                and row["phases"][phase_key]["samples"][0].get("baseline_allocated_bytes") is not None
            ]
            absolute_peak = [base + incr for base, incr in zip(baseline, incremental, strict=False)]
            phase_stats = {
                "mean_of_case_medians_wall_ms": mean(latency),
                "median_of_case_medians_wall_ms": percentile(latency, 50),
                "p95_of_case_medians_wall_ms": p95_by_case,
                "mean_of_case_median_cuda_event_ms": mean(cuda_latency),
                "mean_incremental_peak_allocated_mib": mean(incremental),
                "mean_absolute_peak_allocated_mib": mean(absolute_peak),
            }
            per_phase[phase_key] = phase_stats
            phase_rows.append(
                {
                    "run": run,
                    "checkpoint_epoch": checkpoint["epoch"],
                    "phase": phase_key,
                    "phase_label": phase_label,
                    "case_count": len(latency),
                    **phase_stats,
                }
            )

        total_seconds = training.get("actual_total_epoch_wall_seconds")
        train_seconds = training.get("actual_train_wall_seconds")
        validation_seconds = training.get("actual_validation_wall_seconds")
        app = per_phase["application_evaluator"]
        summaries.append(
            {
                "run": run,
                "architecture": checkpoint["architecture"],
                "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint": checkpoint["path"],
                "parameter_count": checkpoint["parameter_count"],
                "trainable_parameter_count": checkpoint["trainable_parameter_count"],
                "configured_inner_receiver_chunk_size": checkpoint["configured_inner_receiver_chunk_size"],
                "inference_device": data["protocol"]["device"],
                "inference_gpu": data["protocol"]["gpu"]["device_name"],
                "inference_case_count": len(rows),
                "query_count_per_case": data["protocol"]["direct_query_count"],
                "outer_query_batch_size": data["protocol"]["application_query_batch_size"],
                "predict_case_mean_case_median_wall_ms": app["mean_of_case_medians_wall_ms"],
                "predict_case_p95_case_median_wall_ms": app["p95_of_case_medians_wall_ms"],
                "predict_case_mean_case_median_cuda_event_ms": app["mean_of_case_median_cuda_event_ms"],
                "predict_case_mean_incremental_peak_allocated_mib": app["mean_incremental_peak_allocated_mib"],
                "predict_case_mean_absolute_peak_allocated_mib": app["mean_absolute_peak_allocated_mib"],
                "full_forward_mean_case_median_wall_ms": per_phase["full_forward"]["mean_of_case_medians_wall_ms"],
                "prepare_plus_one_query_mean_case_median_wall_ms": per_phase["preparation_plus_one_query"][
                    "mean_of_case_medians_wall_ms"
                ],
                "prepared_p2_decode_mean_case_median_wall_ms": per_phase["prepared_p2_decode"][
                    "mean_of_case_medians_wall_ms"
                ],
                "historical_train_hours_5000_epochs": train_seconds / 3600.0 if train_seconds is not None else None,
                "historical_validation_hours_5000_epochs": validation_seconds / 3600.0
                if validation_seconds is not None
                else None,
                "historical_total_hours_5000_epochs": total_seconds / 3600.0 if total_seconds is not None else None,
                "historical_trainer_peak_cuda_memory_mb_recorded": training.get("peak_cuda_memory_mb_recorded"),
                "historical_training_visible_cuda_device_count": training.get("training_cuda_device_count"),
                "historical_training_torch_version": training.get("training_torch_version"),
                "historical_training_timing_source": training.get("timing_source"),
            }
        )

    base_latency = next(row["predict_case_mean_case_median_wall_ms"] for row in summaries if row["run"] == "1804")
    run1501_latency = next(row["predict_case_mean_case_median_wall_ms"] for row in summaries if row["run"] == "1501")
    for row in summaries:
        latency = row["predict_case_mean_case_median_wall_ms"]
        row["predict_case_latency_ratio_vs_1804"] = latency / base_latency if latency else None
        row["predict_case_speedup_vs_1804"] = base_latency / latency if latency else None
        row["predict_case_latency_change_vs_1501_percent"] = (
            (latency / run1501_latency - 1.0) * 100.0 if latency and run1501_latency else None
        )

    ledger_rows: list[dict[str, Any]] = []
    for run in RUN_ORDER:
        rows = rows_by_run[run]
        ledgers = [row.get("untimed_execution_ledger", {}) for row in rows]
        record: dict[str, Any] = {
            "run": run,
            "case_count": len(rows),
            "ledger_source": "untimed maps-on direct forward; scalar diagnostics only",
            "module_dense_routes_mean": None,
            "module_dense_routes_min": None,
            "module_dense_routes_max": None,
            "module_evaluated_pairs_mean": None,
            "module_evaluated_pairs_min": None,
            "module_evaluated_pairs_max": None,
            "module_gathered_routes_mean": None,
            "module_gathered_routes_min": None,
            "module_gathered_routes_max": None,
            "module_selection_ratio_mean": None,
            "module_unique_pairs_mean": None,
            "module_logical_paths_mean": None,
            "module_rows_forward_mean": None,
            "module_valid_pair_denominator_mean": None,
            "module_executor_selected_mean": None,
            "module_padded_rows_mean": None,
            "environment_unique_pairs_mean": None,
            "environment_logical_paths_mean": None,
            "environment_rows_forward_mean": None,
            "environment_valid_pair_denominator_mean": None,
            "environment_executor_selected_mean": None,
            "environment_content_dot_rows_forward_mean": None,
        }
        if run == "1404":
            for key, field in (
                ("routing_aux.pairwise_dense_route_count", "module_dense_routes"),
                ("routing_aux.pairwise_evaluated_pair_count", "module_evaluated_pairs"),
                ("routing_aux.pairwise_gathered_route_count", "module_gathered_routes"),
            ):
                avg, minimum, maximum = ledger_counter_stats(ledgers, key)
                record[f"{field}_mean"] = avg
                record[f"{field}_min"] = minimum
                record[f"{field}_max"] = maximum
            record["module_selection_ratio_mean"] = mean(
                [
                    float(ledger["routing_aux.pairwise_selection_ratio"])
                    for ledger in ledgers
                    if "routing_aux.pairwise_selection_ratio" in ledger
                ]
            )
            record["environment_unique_pairs_mean"] = None
            record["environment_logical_paths_mean"] = None
            record["environment_rows_forward_mean"] = None
            record["environment_valid_pair_denominator_mean"] = None
            record["environment_content_dot_rows_forward_mean"] = None
            record["executor_selected_mean"] = None
            record["interpretation"] = "Classic route counters are available; no environment pair ledger is exposed."
        elif run in {"1501", "1502"}:
            prefix = "routing_aux.group_control_"

            for part in ("module", "environment"):
                record[f"{part}_unique_pairs_mean"] = group_control_ledger_mean(ledgers, prefix, part, "unique_pairs")
                record[f"{part}_logical_paths_mean"] = group_control_ledger_mean(ledgers, prefix, part, "logical_paths")
                record[f"{part}_rows_forward_mean"] = group_control_ledger_mean(
                    ledgers, prefix, part, "fine_rows_forward"
                )
                record[f"{part}_valid_pair_denominator_mean"] = group_control_ledger_mean(
                    ledgers, prefix, part, "valid_pair_denominator"
                )
                record[f"{part}_executor_selected_mean"] = group_control_ledger_mean(
                    ledgers, prefix, part, "executor_selected"
                )
            record["environment_content_dot_rows_forward_mean"] = group_control_ledger_mean(
                ledgers, prefix, "environment", "content_dot_rows_forward"
            )
            record["module_padded_rows_mean"] = group_control_ledger_mean(ledgers, prefix, "module", "padded_rows")
            record["interpretation"] = (
                "P2 rectangular-reference counters separate unique support from rows actually forwarded."
            )
        else:
            record.update(
                {
                    "module_dense_routes_mean": None,
                    "module_dense_routes_min": None,
                    "module_dense_routes_max": None,
                    "module_evaluated_pairs_mean": None,
                    "module_gathered_routes_mean": None,
                    "environment_unique_pairs_mean": None,
                    "environment_logical_paths_mean": None,
                    "environment_rows_forward_mean": None,
                    "environment_valid_pair_denominator_mean": None,
                    "environment_content_dot_rows_forward_mean": None,
                    "executor_selected_mean": None,
                    "interpretation": "No scalar row ledger is emitted; dense all-source behavior is visible in dense_pairwise.py.",
                }
            )
        ledger_rows.append(record)
    return summaries, phase_rows, ledger_rows


def make_figure(summaries: list[dict[str, Any]], phases: list[dict[str, Any]]) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    runs = [summary["run"] for summary in summaries]
    colors = {"1404": "#4477AA", "1804": "#EE6677", "1501": "#228833", "1502": "#AA3377"}
    display_names = {"1404": "1404 classic", "1804": "1804 dense", "1501": "1501 adaptive", "1502": "1502 sparsemax"}
    x = np.arange(len(runs))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.1))

    ax = axes[0, 0]
    phase_lookup = {(row["run"], row["phase"]): row for row in phases}
    width = 0.34
    app = [phase_lookup[(run, "application_evaluator")]["mean_of_case_medians_wall_ms"] for run in runs]
    full = [phase_lookup[(run, "full_forward")]["mean_of_case_medians_wall_ms"] for run in runs]
    ax.bar(x - width / 2, app, width, label="predict_case", color=[colors[run] for run in runs])
    ax.bar(x + width / 2, full, width, label="one full forward", color=[colors[run] for run in runs], alpha=0.48)
    ax.set_yscale("log")
    ax.set_ylabel("milliseconds, log scale")
    ax.set_title("Whole-grid inference cost")
    ax.set_xticks(x, [display_names[run] for run in runs], rotation=12, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    prep = [phase_lookup[(run, "preparation_plus_one_query")]["mean_of_case_medians_wall_ms"] for run in runs]
    decode = [phase_lookup[(run, "prepared_p2_decode")]["mean_of_case_medians_wall_ms"] for run in runs]
    ax.bar(x - width / 2, prep, width, label="prepare + 1 query", color="#66CCEE")
    ax.bar(x + width / 2, decode, width, label="prepared P2 decode", color="#CCBB44")
    ax.set_yscale("log")
    ax.set_ylabel("milliseconds, log scale")
    ax.set_title("Preparation and decode scopes")
    ax.set_xticks(x, [display_names[run] for run in runs], rotation=12, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    memory = [summary["predict_case_mean_incremental_peak_allocated_mib"] for summary in summaries]
    bars = ax.bar(x, memory, color=[colors[run] for run in runs])
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.set_ylabel("MiB above pre-call allocation")
    ax.set_title("Inference peak allocated memory")
    ax.set_xticks(x, [display_names[run] for run in runs], rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    train = [summary["historical_train_hours_5000_epochs"] for summary in summaries]
    validation = [summary["historical_validation_hours_5000_epochs"] for summary in summaries]
    bars_train = ax.bar(x, train, color="#4477AA", label="train")
    bars_val = ax.bar(x, validation, bottom=train, color="#BBBBBB", label="validation")
    totals = np.asarray(train) + np.asarray(validation)
    for index, total in enumerate(totals):
        ax.text(index, total + 0.35, f"{total:.1f} h", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("historical hours for 5,000 epochs")
    ax.set_title("Recorded training time (different run environments)")
    ax.set_xticks(x, [display_names[run] for run in runs], rotation=12, ha="right")
    ax.legend(handles=[bars_train, bars_val], frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("HONF inference and training cost comparison", fontsize=15)
    fig.text(
        0.5,
        0.012,
        "Inference: same RTX 6000 Ada, 90 test cases, Q=8,192, native inner chunks, maps off; bars are means of per-case medians. "
        "Historical training times are from run summaries and are not same-hardware controlled.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    if data.get("status") != "complete":
        raise ValueError(f"benchmark is not complete: {data.get('status')}")
    summaries, phases, ledgers = aggregate(data)
    write_csv(SUMMARY_CSV, summaries)
    write_csv(PHASE_CSV, phases)
    write_csv(LEDGER_CSV, ledgers)
    make_figure(summaries, phases)
    print(
        json.dumps({"models": len(summaries), "cases_per_model": data["protocol"]["case_count"], "figure": str(FIGURE)})
    )


if __name__ == "__main__":
    main()
