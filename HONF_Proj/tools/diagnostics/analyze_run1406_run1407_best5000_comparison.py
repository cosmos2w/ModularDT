#!/usr/bin/env python3
"""Reduce mature Run-1404/1406/1407/1804 accuracy and training evidence.

The primary checkpoint policy is validation-field-MSE selection.  Exact epoch
5000 is retained as an endpoint sensitivity rather than used to select on the
90-case evaluation population.  Historical resume artifacts are handled
explicitly: replayed epochs keep their later row, and Dense Run 1804's expanded
post-resume CSV rows recover predicted-mode validation values from their final
three fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT / "Trained_Results/ThermalChannel/HONF_Forward_Runs"
DEFAULT_ACCURACY = (
    PROJECT
    / "diagnostics/generated/run1404_1406_1407_1804_best5000_accuracy_20260920"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "diagnostics/generated/run1406_run1407_best5000_comparison_20260920"
)

RUNS = {
    "Run 1404": RUN_ROOT / "Run_1404_20260916_092508_routing_only_pairwise",
    "Run 1406": RUN_ROOT
    / "Run_1406_20260919_132837_low_dimensional_group_control_executor_optimized_rerun",
    "Run 1407": RUN_ROOT
    / "Run_1407_20260919_174751_phase_shared_prototype_group_control",
    "Dense 1804": RUN_ROOT
    / "Run_1804_20260905_081349_dense_pairwise_field_adaptation",
}
BEST_LABELS = {
    "Run 1404": "Run1404_best_field",
    "Run 1406": "Run1406_best_field",
    "Run 1407": "Run1407_best_field",
    "Dense 1804": "Run1804_best_field",
}
ENDPOINT_LABELS = {
    "Run 1404": "Run1404_epoch5000",
    "Run 1406": "Run1406_epoch5000",
    "Run 1407": "Run1407_epoch5000",
    "Dense 1804": "Run1804_epoch5000",
}
COLORS = {
    "Run 1404": "#7A7A7A",
    "Run 1406": "#0072B2",
    "Run 1407": "#CC79A7",
    "Dense 1804": "#D55E00",
}
MARKERS = {"Run 1404": "s", "Run 1406": "o", "Run 1407": "D", "Dense 1804": "^"}

COMPONENTS = {
    "Fluid field": "global_field_fluid_norm",
    "Near interface": "global_field_near_interface_norm",
    "Far fluid": "global_field_far_fluid_norm",
    "U": "field_u_fluid_norm",
    "V": "field_v_fluid_norm",
    "Pressure": "field_p_fluid_norm",
    "Vorticity": "field_omega_fluid_norm",
    "Field temperature": "field_temperature_fluid_norm",
    "Internal temperature": "internal_temperature_physical",
    "Surface temperature": "interface_t_surface_physical",
    "Normal heat flux": "interface_q_normal_physical",
    "Final environment T": "port_t_env_final_physical",
    "Final effective h": "port_h_effective_final_physical",
}


def _finite(value: str | float | None) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_training_rows(path: Path) -> list[dict[str, float | int]]:
    """Read one history, retaining the later copy of replayed epochs.

    Dense 1804 resumed after the CSV schema expanded.  Those appended rows have
    38 more fields than the original header; their final three values are the
    predicted-mode validation total, field MSE, and temperature MSE.  Predicted
    mode is the primary validation mode for this run, so those fields recover
    the intended convergence series without reassigning the shifted columns.
    """

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indexes = {name: header.index(name) for name in header}
        by_epoch: dict[int, dict[str, float | int]] = {}
        for values in reader:
            if not values:
                continue
            epoch = int(float(values[indexes["epoch"]]))
            expanded = len(values) > len(header)
            if expanded:
                val_total = _finite(values[-3])
                val_field = _finite(values[-2])
                val_temperature = _finite(values[-1])
            else:
                val_total = _finite(values[indexes["val_loss_total"]])
                val_field = _finite(values[indexes["val_field_mse"]])
                val_temperature = _finite(values[indexes["val_temperature_mse"]])
            peak_memory = _finite(values[indexes["peak_cuda_memory_mb"]])
            by_epoch[epoch] = {
                "epoch": epoch,
                "val_loss_total": math.nan if val_total is None else val_total,
                "val_field_mse": math.nan if val_field is None else val_field,
                "val_temperature_mse": (
                    math.nan if val_temperature is None else val_temperature
                ),
                "peak_cuda_memory_mb": math.nan if peak_memory is None else peak_memory,
                "expanded_schema_row": int(expanded),
            }
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def trailing_median(values: list[float], window: int = 101) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        finite = [value for value in values[start : index + 1] if math.isfinite(value)]
        result[index] = statistics.median(finite) if finite else math.nan
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checkpoint_selection() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, run_dir in RUNS.items():
        for policy, filename in (
            ("validation_field_best", "best_by_field_mse_model.pt"),
            ("exact_epoch_5000", "epoch_5000_model.pt"),
        ):
            path = run_dir / filename
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            rows.append(
                {
                    "model": model,
                    "policy": policy,
                    "checkpoint": str(path),
                    "epoch": int(checkpoint["epoch"]),
                    "selection_metric": float(checkpoint.get("best_metric", math.nan)),
                    "has_optimizer_state": bool(checkpoint.get("optimizer_state_dict")),
                    "has_rng_state": bool(checkpoint.get("rng_state")),
                }
            )
    return rows


def accuracy_summary(
    per_case: list[dict[str, str]], labels: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    by_model_case: dict[str, dict[str, float]] = {}
    for model, label in labels.items():
        rows = [row for row in per_case if row["model_label"] == label]
        if len(rows) != 90:
            raise ValueError(f"{label} has {len(rows)} cases; expected 90")
        values = [float(row["global_field_fluid_norm_l2"]) for row in rows]
        sse = sum(float(row["global_field_fluid_norm_sse"]) for row in rows)
        target = sum(float(row["global_field_fluid_norm_target_sse"]) for row in rows)
        worst = int(np.argmax(values))
        summaries.append(
            {
                "model": model,
                "label": label,
                "case_count": len(rows),
                "pooled_fluid_relative_l2": math.sqrt(sse / target),
                "equal_case_mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": max(values),
                "worst_case": rows[worst]["case_id"],
            }
        )
        by_model_case[model] = {
            row["case_id"]: float(row["global_field_fluid_norm_l2"]) for row in rows
        }
        component_row: dict[str, Any] = {"model": model}
        for display, prefix in COMPONENTS.items():
            component_sse = sum(float(row[f"{prefix}_sse"]) for row in rows)
            component_target = sum(float(row[f"{prefix}_target_sse"]) for row in rows)
            component_row[display] = math.sqrt(component_sse / component_target)
        components.append(component_row)

    pairs: list[dict[str, Any]] = []
    models = list(labels)
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            cases = sorted(set(by_model_case[left]) & set(by_model_case[right]))
            deltas = [by_model_case[left][case] - by_model_case[right][case] for case in cases]
            pairs.append(
                {
                    "candidate": left,
                    "baseline": right,
                    "candidate_wins": sum(delta < 0.0 for delta in deltas),
                    "baseline_wins": sum(delta > 0.0 for delta in deltas),
                    "ties": sum(delta == 0.0 for delta in deltas),
                    "mean_delta": statistics.fmean(deltas),
                    "median_delta": statistics.median(deltas),
                }
            )
    return summaries, pairs, components


def plot_convergence_accuracy(
    path: Path,
    histories: Mapping[str, list[dict[str, float | int]]],
    selections: list[dict[str, Any]],
    best_summary: list[dict[str, Any]],
    endpoint_summary: list[dict[str, Any]],
    best_per_case: list[dict[str, str]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4), constrained_layout=True)
    ax = axes[0, 0]
    selected = {
        row["model"]: row
        for row in selections
        if row["policy"] == "validation_field_best"
    }
    for model, rows in histories.items():
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        values = [float(row["val_field_mse"]) for row in rows]
        curve = trailing_median(values)
        ax.plot(epochs, values, color=COLORS[model], alpha=0.08, linewidth=0.6)
        ax.plot(epochs, curve, color=COLORS[model], linewidth=1.9, label=model)
        choice = selected[model]
        ax.scatter(
            [choice["epoch"]],
            [choice["selection_metric"]],
            color=COLORS[model],
            marker=MARKERS[model],
            s=48,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
    ax.set_yscale("log")
    ax.set_xlim(0, 5000)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation field MSE")
    ax.set_title("Convergence differs early, then 1406 and 1407 meet")
    ax.grid(True, which="both", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    ordered = list(RUNS)
    best = {row["model"]: row["pooled_fluid_relative_l2"] for row in best_summary}
    endpoint = {
        row["model"]: row["pooled_fluid_relative_l2"] for row in endpoint_summary
    }
    x = np.arange(len(ordered))
    ax.bar(
        x,
        [best[model] for model in ordered],
        color=[COLORS[model] for model in ordered],
        width=0.62,
        alpha=0.88,
        label="Validation-selected best",
    )
    ax.scatter(
        x,
        [endpoint[model] for model in ordered],
        facecolors="white",
        edgecolors=[COLORS[model] for model in ordered],
        marker="o",
        s=65,
        linewidth=1.8,
        zorder=5,
        label="Exact epoch 5000",
    )
    ax.set_xticks(x, ordered)
    ax.set_ylim(0.0, max(max(best.values()), max(endpoint.values())) * 1.18)
    ax.set_ylabel("Pooled fluid relative L2")
    ax.set_title("Checkpoint policy changes the 1406–1407 ordering")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False)

    by_label = {
        label: {
            row["case_id"]: float(row["global_field_fluid_norm_l2"])
            for row in best_per_case
            if row["model_label"] == label
        }
        for label in BEST_LABELS.values()
    }
    comparisons = [
        ("1407 − 1406", "Run 1407", "Run 1406"),
        ("1407 − Dense", "Run 1407", "Dense 1804"),
        ("1406 − Dense", "Run 1406", "Dense 1804"),
    ]
    ax = axes[1, 0]
    delta_sets = []
    for _, left, right in comparisons:
        left_values = by_label[BEST_LABELS[left]]
        right_values = by_label[BEST_LABELS[right]]
        delta_sets.append([left_values[case] - right_values[case] for case in sorted(left_values)])
    parts = ax.violinplot(delta_sets, showmeans=False, showmedians=True, widths=0.75)
    for body, (_, left, _) in zip(parts["bodies"], comparisons, strict=True):
        body.set_facecolor(COLORS[left])
        body.set_edgecolor("#333333")
        body.set_alpha(0.65)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#333333")
        parts[key].set_linewidth(0.9)
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks(range(1, len(comparisons) + 1), [item[0] for item in comparisons])
    ax.set_ylabel("Per-case fluid relative-L2 delta")
    ax.set_title("1407 usually beats 1406, but Dense wins most cases")
    ax.grid(axis="y", color="#E2E2E2", linewidth=0.6)

    ax = axes[1, 1]
    memory_models = ["Run 1404", "Run 1406", "Run 1407"]
    for model in memory_models:
        rows = histories[model]
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        values = np.asarray([float(row["peak_cuda_memory_mb"]) for row in rows]) / 1024.0
        valid = np.isfinite(values)
        ax.plot(epochs[valid], values[valid], color=COLORS[model], linewidth=1.0, label=model)
    ax.axvline(500, color="#333333", linewidth=0.9, linestyle="--")
    ax.text(530, 29.4, "1406 QE checkpoint\nboundary after resume", fontsize=8, va="top")
    ax.set_xlim(0, 5000)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch peak allocated memory (GiB)")
    ax.set_title("Large historical shifts are regime changes, not one smooth trend")
    ax.grid(True, color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False)

    fig.suptitle(
        "Mature accuracy, convergence, and recorded training-memory history",
        fontsize=15,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_components(path: Path, components: list[dict[str, Any]]) -> None:
    by_model = {row["model"]: row for row in components}
    dense = by_model["Dense 1804"]
    metrics = list(COMPONENTS)
    y = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10.8, 7.2), constrained_layout=True)
    offsets = {"Run 1404": -0.18, "Run 1406": 0.0, "Run 1407": 0.18}
    for model in ("Run 1404", "Run 1406", "Run 1407"):
        deltas = [100.0 * (by_model[model][metric] / dense[metric] - 1.0) for metric in metrics]
        ax.scatter(
            deltas,
            y + offsets[model],
            color=COLORS[model],
            marker=MARKERS[model],
            s=42,
            label=model,
            zorder=3,
        )
    ax.axvline(0.0, color="#222222", linewidth=1.1, label="Dense 1804")
    ax.set_yticks(y, metrics)
    ax.invert_yaxis()
    ax.set_xlabel("Relative-L2 difference from Dense 1804 (%) — lower is better")
    ax.set_title("Component accuracy: no routed model dominates every physical output")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-dir", type=Path, default=DEFAULT_ACCURACY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    accuracy = args.accuracy_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    histories = {
        model: read_training_rows(run_dir / "metrics.csv") for model, run_dir in RUNS.items()
    }
    selections = checkpoint_selection()
    best_per_case = _read_csv(accuracy / "tables/per_case_metrics.csv")
    endpoint_per_case = _read_csv(accuracy / "endpoint5000/tables/per_case_metrics.csv")
    best_summary, best_pairs, best_components = accuracy_summary(best_per_case, BEST_LABELS)
    endpoint_summary, endpoint_pairs, endpoint_components = accuracy_summary(
        endpoint_per_case, ENDPOINT_LABELS
    )

    history_rows = [
        {"model": model, **row}
        for model, rows in histories.items()
        for row in rows
    ]
    _write_csv(output / "checkpoint_selection.csv", selections)
    _write_csv(output / "training_history_reconciled.csv", history_rows)
    _write_csv(output / "best_accuracy_summary.csv", best_summary)
    _write_csv(output / "best_paired_cases.csv", best_pairs)
    _write_csv(output / "best_component_accuracy.csv", best_components)
    _write_csv(output / "epoch5000_accuracy_summary.csv", endpoint_summary)
    _write_csv(output / "epoch5000_paired_cases.csv", endpoint_pairs)
    _write_csv(output / "epoch5000_component_accuracy.csv", endpoint_components)
    payload = {
        "checkpoint_policy": {
            "primary": "validation_field_best",
            "sensitivity": "exact_epoch_5000",
            "selection_population": "training-run validation metric",
            "evaluation_population": "90-case test/development holdout",
            "test_population_used_for_selection": False,
        },
        "data_quality": {
            "replayed_epochs_keep_later_row": {"Run 1404": [851, 852, 853, 854], "Run 1407": [137]},
            "scientific_duplicate_fields_identical": True,
            "dense_1804_expanded_schema_recovery": "post-resume rows use final predicted total/field/temperature fields",
        },
        "checkpoint_selection": selections,
        "primary_accuracy": best_summary,
        "primary_pairs": best_pairs,
        "endpoint_accuracy": endpoint_summary,
        "endpoint_pairs": endpoint_pairs,
    }
    _write_json(output / "accuracy_convergence_summary.json", payload)
    plot_convergence_accuracy(
        output / "figures/convergence_accuracy_memory.png",
        histories,
        selections,
        best_summary,
        endpoint_summary,
        best_per_case,
    )
    plot_components(output / "figures/component_accuracy_vs_dense.png", best_components)
    print(json.dumps({"output_dir": str(output), "summary": payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
