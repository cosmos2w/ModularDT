#!/usr/bin/env python3
"""Consolidate the frozen Stage-5 comparison artifacts and render report figures.

This script is evaluation-only.  It reads metrics and previously generated JSON/CSV
artifacts; it does not instantiate or modify a model or checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS = ROOT / "diagnostics"
RUN_ROOT = ROOT / "Trained_Results" / "ThermalChannel" / "HONF_Forward_Runs"
OUT = DIAGNOSTICS / "stage5_final_comparison"

RUNS = {
    "Run 1000 classic": RUN_ROOT / "Run_1000_20260817_214356_enhanced_honf_pairwise",
    "Run 1301 exchangeable": RUN_ROOT / "Run_1301_20260821_142454_stage5_exchangeable_soft_organized",
    "Run 1302 fixed descriptor": RUN_ROOT / "Run_1302_20260821_142505_stage5_fixed_softmax_modern",
    "Run 1303 fixed residual": RUN_ROOT / "Run_1303_20260822_144316_stage5_fixed_residual_concat_modern",
    "Run 1304 fixed residual uniform LR": RUN_ROOT / "Run_1304_20260822_232649_stage5_fixed_residual_concat_uniform_lr3e4",
}
CHECKPOINT_LABELS = {
    "Run 1000 classic": "classic_1000_best",
    "Run 1301 exchangeable": "exchangeable_1301_latest2500",
    "Run 1302 fixed descriptor": "fixed_1302_latest2500",
    "Run 1303 fixed residual": "residual_1303_latest2500",
    # Epoch 2500 isolates the organizer-learning-rate change from extra training budget.
    "Run 1304 fixed residual uniform LR": "uniform_1304_epoch2500",
}
BENCHMARK_LABELS = {
    "Run 1000 classic": "classic_1000_best",
    "Run 1301 exchangeable": "exchangeable_1301_latest2500",
    "Run 1302 fixed descriptor": "fixed_descriptor_1302_latest2500",
    "Run 1303 fixed residual": "fixed_residual_1303_latest2500",
    "Run 1304 fixed residual uniform LR": "fixed_residual_uniform_1304_latest5000",
}
COLORS = {
    "Run 1000 classic": "#4C78A8",
    "Run 1301 exchangeable": "#E45756",
    "Run 1302 fixed descriptor": "#B279A2",
    "Run 1303 fixed residual": "#59A14F",
    "Run 1304 fixed residual uniform LR": "#F28E2B",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def merge_evaluation_json(first: dict, second: dict, sections: tuple[str, ...]) -> dict:
    """Merge independently evaluated checkpoint labels without changing values."""
    merged = dict(first)
    for section in sections:
        merged[section] = {**first.get(section, {}), **second.get(section, {})}
    return merged


def read_metrics(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    duplicate_rows = int(raw.duplicated("epoch", keep=False).sum())
    duplicate_epochs = int(raw.loc[raw.duplicated("epoch", keep=False), "epoch"].nunique())
    dedup = raw.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)
    return dedup, {
        "raw_rows": int(len(raw)),
        "deduplicated_rows": int(len(dedup)),
        "duplicate_rows": duplicate_rows,
        "duplicate_epochs": duplicate_epochs,
    }


def trailing_value(frame: pd.DataFrame, column: str, epoch: int, window: int = 50) -> float:
    values = frame.loc[frame["epoch"].between(epoch - window + 1, epoch), column]
    return float(values.median())


def first_sustained_epoch(frame: pd.DataFrame, column: str, threshold: float, window: int = 25):
    rolling = frame[column].rolling(window, min_periods=window).median()
    hit = frame.loc[rolling <= threshold, "epoch"]
    return None if hit.empty else int(hit.iloc[0])


def metric_median(topology: dict, label: str, metric: str) -> float:
    return float(topology["summary"][f"{label}/final/selected"][metric]["median"])


def accuracy_value(accuracy: dict, label: str, region: str, channel: str, field: str) -> float:
    return float(accuracy["summary"][f"{label}/{region}/{channel}"][field])


def pruning_value(pruning: dict, label: str, region: str, channel: str, field: str) -> float:
    return float(pruning["summary"][f"{label}/{region}/{channel}"][field])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {}
    metric_inventory = {}
    for label, run in RUNS.items():
        metrics[label], metric_inventory[label] = read_metrics(run / "metrics.csv")

    accuracy_path = DIAGNOSTICS / "stage5_final_1301_1302_1000_accuracy" / "accuracy_summary.json"
    topology_path = DIAGNOSTICS / "stage5_final_1301_1302_1000_topology" / "topology_quality_summary.json"
    pruning_path = DIAGNOSTICS / "stage5_final_1301_1302_pruning" / "retained_mass_pruning_summary.json"
    residual_accuracy_path = DIAGNOSTICS / "stage5_final_1303_accuracy" / "accuracy_summary.json"
    residual_topology_path = DIAGNOSTICS / "stage5_final_1303_topology" / "topology_quality_summary.json"
    residual_pruning_path = DIAGNOSTICS / "stage5_final_1303_pruning" / "retained_mass_pruning_summary.json"
    uniform_accuracy_path = DIAGNOSTICS / "stage5_final_1304_accuracy" / "accuracy_summary.json"
    uniform_topology_path = DIAGNOSTICS / "stage5_final_1304_topology" / "topology_quality_summary.json"
    uniform_pruning_path = DIAGNOSTICS / "stage5_final_1304_pruning" / "retained_mass_pruning_summary.json"
    benchmark_path = DIAGNOSTICS / "stage5_final_1000_1304_checkpoint_benchmark.json"
    capacity_accuracy_path = DIAGNOSTICS / "stage5_final_1301_capacity_accuracy" / "accuracy_summary.json"
    capacity_topology_path = DIAGNOSTICS / "stage5_final_1301_capacity6_capacity8" / "topology_quality_summary.json"

    accuracy_original = read_json(accuracy_path)
    accuracy_residual = read_json(residual_accuracy_path)
    accuracy = merge_evaluation_json(accuracy_original, accuracy_residual, ("provenance", "summary"))
    accuracy_uniform = read_json(uniform_accuracy_path)
    accuracy = merge_evaluation_json(accuracy, accuracy_uniform, ("provenance", "summary"))
    topology = merge_evaluation_json(
        read_json(topology_path), read_json(residual_topology_path),
        ("provenance", "summary", "per_edge_summary"),
    )
    topology = merge_evaluation_json(
        topology, read_json(uniform_topology_path),
        ("provenance", "summary", "per_edge_summary"),
    )
    pruning = merge_evaluation_json(
        read_json(pruning_path), read_json(residual_pruning_path),
        ("checkpoints", "summary"),
    )
    pruning = merge_evaluation_json(
        pruning, read_json(uniform_pruning_path), ("checkpoints", "summary")
    )
    benchmark = read_json(benchmark_path)
    capacity_accuracy = read_json(capacity_accuracy_path)
    capacity_topology = read_json(capacity_topology_path)

    milestones = [250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
    convergence = {}
    for label, frame in metrics.items():
        available = [epoch for epoch in milestones if epoch <= int(frame["epoch"].max())]
        convergence[label] = {
            "max_epoch": int(frame["epoch"].max()),
            "trailing50_val_field_mse": {
                str(epoch): trailing_value(frame, "val_field_mse", epoch) for epoch in available
            },
            "trailing50_val_temperature_mse": {
                str(epoch): trailing_value(frame, "val_temperature_mse", epoch) for epoch in available
            },
            "best_val_field_mse": float(frame["val_field_mse"].min()),
            "best_val_field_epoch": int(frame.loc[frame["val_field_mse"].idxmin(), "epoch"]),
            "best_val_temperature_mse": float(frame["val_temperature_mse"].min()),
            "best_val_temperature_epoch": int(frame.loc[frame["val_temperature_mse"].idxmin(), "epoch"]),
            "first_trailing25_below": {
                str(threshold): first_sustained_epoch(frame, "val_field_mse", threshold)
                for threshold in (0.1, 0.05, 0.02, 0.01, 0.005)
            },
        }
        tail = frame.tail(min(500, len(frame)))
        rolling = frame["val_field_mse"].rolling(25, min_periods=10).median()
        aligned = rolling.loc[tail.index]
        convergence[label]["last500_stability"] = {
            "val_field_median": float(tail["val_field_mse"].median()),
            "val_field_p05": float(tail["val_field_mse"].quantile(0.05)),
            "val_field_p95": float(tail["val_field_mse"].quantile(0.95)),
            "val_field_max_over_median": float(tail["val_field_mse"].max() / tail["val_field_mse"].median()),
            "val_field_spikes_over_5x_rolling25": int((tail["val_field_mse"] > 5.0 * aligned).sum()),
            "train_loss_median": float(tail["loss_total"].median()),
            "train_loss_p95_over_median": float(tail["loss_total"].quantile(0.95) / tail["loss_total"].median()),
        }

    channels = ["u", "v", "p", "omega", "temperature"]
    selected_accuracy = {}
    for run_label, checkpoint_label in CHECKPOINT_LABELS.items():
        selected_accuracy[run_label] = {
            "fluid_pooled_normalized_mse": accuracy_value(
                accuracy, checkpoint_label, "fluid", "__all__", "pooled_normalized_mse"
            ),
            "fluid_pooled_relative_l2": accuracy_value(
                accuracy, checkpoint_label, "fluid", "__all__", "pooled_normalized_relative_l2"
            ),
            "channels": {
                channel: accuracy_value(
                    accuracy, checkpoint_label, "fluid", channel, "pooled_normalized_mse"
                )
                for channel in channels
            },
            "near_interface": {
                channel: accuracy_value(
                    accuracy, checkpoint_label, "near_interface_fluid", channel, "pooled_normalized_mse"
                )
                for channel in channels
            },
            "far_field": {
                channel: accuracy_value(
                    accuracy, checkpoint_label, "far_field_fluid", channel, "pooled_normalized_mse"
                )
                for channel in channels
            },
        }

    accuracy_per_case = pd.concat(
        [
            pd.read_csv(accuracy_original["artifacts"]["per_case_csv"]),
            pd.read_csv(accuracy_residual["artifacts"]["per_case_csv"]),
            pd.read_csv(accuracy_uniform["artifacts"]["per_case_csv"]),
        ],
        ignore_index=True,
    )
    case_fluid = accuracy_per_case[
        (accuracy_per_case["region"] == "fluid") & (accuracy_per_case["channel"] != "__all__")
    ]
    case_aggregate = (
        case_fluid.groupby(["checkpoint", "case_id"], as_index=False)["normalized_sse"].sum()
        .merge(
            case_fluid.groupby(["checkpoint", "case_id"], as_index=False)["value_count"].sum(),
            on=["checkpoint", "case_id"],
        )
    )
    case_aggregate["mse"] = case_aggregate["normalized_sse"] / case_aggregate["value_count"]
    case_pivot = case_aggregate.pivot(index="case_id", columns="checkpoint", values="mse")
    case_wins = {
        "exchangeable_beats_fixed_fraction": float(
            (case_pivot["exchangeable_1301_latest2500"] < case_pivot["fixed_1302_latest2500"]).mean()
        ),
        "residual_beats_descriptor_fraction": float(
            (case_pivot["residual_1303_latest2500"] < case_pivot["fixed_1302_latest2500"]).mean()
        ),
        "residual_beats_exchangeable_fraction": float(
            (case_pivot["residual_1303_latest2500"] < case_pivot["exchangeable_1301_latest2500"]).mean()
        ),
        "exchangeable_beats_classic_fraction": float(
            (case_pivot["exchangeable_1301_latest2500"] < case_pivot["classic_1000_best"]).mean()
        ),
        "fixed_beats_classic_fraction": float(
            (case_pivot["fixed_1302_latest2500"] < case_pivot["classic_1000_best"]).mean()
        ),
        "residual_beats_classic_fraction": float(
            (case_pivot["residual_1303_latest2500"] < case_pivot["classic_1000_best"]).mean()
        ),
        "uniform_lr_epoch2500_beats_residual_lr_epoch2500_fraction": float(
            (case_pivot["uniform_1304_epoch2500"] < case_pivot["residual_1303_latest2500"]).mean()
        ),
        "uniform_lr_best_beats_residual_lr_epoch2500_fraction": float(
            (case_pivot["uniform_1304_best"] < case_pivot["residual_1303_latest2500"]).mean()
        ),
        "uniform_lr_best_beats_residual_best_fraction": float(
            (case_pivot["uniform_1304_best"] < case_pivot["residual_1303_best"]).mean()
        ),
        "uniform_lr_latest5000_beats_residual_lr_epoch2500_fraction": float(
            (case_pivot["uniform_1304_latest5000"] < case_pivot["residual_1303_latest2500"]).mean()
        ),
    }

    topology_metrics = [
        "module_edge_column_cosine",
        "module_effective_rank",
        "environment_edge_column_cosine",
        "environment_effective_rank",
        "environment_largest_dominant_occupancy",
        "environment_neighbor_l1",
        "region_separation_normalized",
        "query_edge_column_cosine",
        "query_effective_rank",
        "query_entropy_norm_direct",
        "query_per_edge_spatial_std",
        "pairwise_edge_map_cosine",
        "pairwise_edge_map_effective_rank",
    ]
    topology_table = {
        run_label: {
            metric: metric_median(topology, checkpoint_label, metric) for metric in topology_metrics
        }
        for run_label, checkpoint_label in CHECKPOINT_LABELS.items()
    }
    for run_label, checkpoint_label in CHECKPOINT_LABELS.items():
        if run_label != "Run 1000 classic":
            topology_table[run_label].update(
                {
                    "additive_temperature_edge_map_cosine": metric_median(
                        topology, checkpoint_label, "additive_temperature_edge_map_cosine"
                    ),
                    "additive_temperature_effective_rank": metric_median(
                        topology, checkpoint_label, "additive_temperature_effective_rank"
                    ),
                    "additive_temperature_largest_energy_fraction": metric_median(
                        topology, checkpoint_label, "additive_temperature_largest_energy_fraction"
                    ),
                }
            )

    uniform_checkpoint_progression = {}
    for checkpoint_label in (
        "uniform_1304_epoch0500",
        "uniform_1304_epoch1000",
        "uniform_1304_epoch1500",
        "uniform_1304_epoch2500",
        "uniform_1304_best",
        "uniform_1304_latest5000",
    ):
        uniform_checkpoint_progression[checkpoint_label] = {
            "fluid_pooled_normalized_mse": accuracy_value(
                accuracy, checkpoint_label, "fluid", "__all__", "pooled_normalized_mse"
            ),
            "fluid_pooled_relative_l2": accuracy_value(
                accuracy, checkpoint_label, "fluid", "__all__", "pooled_normalized_relative_l2"
            ),
            "channels": {
                channel: accuracy_value(
                    accuracy, checkpoint_label, "fluid", channel, "pooled_normalized_mse"
                )
                for channel in channels
            },
            "topology_medians": {
                metric: metric_median(topology, checkpoint_label, metric)
                for metric in topology_metrics
            },
        }

    pruning_table = {}
    for run_label, checkpoint_label in [
        ("Run 1301 exchangeable", "exchangeable_1301_latest2500"),
        ("Run 1302 fixed descriptor", "fixed_1302_latest2500"),
        ("Run 1303 fixed residual", "residual_1303_latest2500"),
        ("Run 1304 fixed residual uniform LR @2500", "uniform_1304_epoch2500"),
        ("Run 1304 fixed residual uniform LR @5000", "uniform_1304_latest5000"),
    ]:
        checkpoint_summary = pruning["checkpoints"][checkpoint_label]
        timing = checkpoint_summary["runtime"]
        pruning_table[run_label] = {
            "fluid_mse_degradation": pruning_value(
                pruning, checkpoint_label, "fluid", "__all__", "pooled_relative_mse_degradation"
            ),
            "worst_fluid_channel_mse_degradation": max(
                pruning_value(pruning, checkpoint_label, "fluid", channel, "pooled_relative_mse_degradation")
                for channel in channels
            ),
            "query_route_reduction": pruning_value(
                pruning, checkpoint_label, "fluid", "__all__", "query_route_reduction_mean"
            ),
            "module_route_reduction": pruning_value(
                pruning, checkpoint_label, "fluid", "__all__", "module_route_reduction_mean"
            ),
            "query_retained_mass_p05": float(checkpoint_summary["query_retained_mass"]["p05"]),
            "module_retained_mass_p05": float(
                checkpoint_summary["routed_module_retained_mass"]["p05"]
            ),
            "dense_decoder_median_ms": 1000.0 * float(timing["dense"]["median_seconds"]),
            "pruned_decoder_median_ms": 1000.0
            * float(timing["retained_mass_pruned"]["median_seconds"]),
            "decoder_speedup": float(
                timing["dense"]["median_seconds"]
                / timing["retained_mass_pruned"]["median_seconds"]
            ),
            "dense_peak_allocated_mb": float(timing["dense"]["peak_allocated_bytes"] / 1e6),
            "pruned_peak_allocated_mb": float(
                timing["retained_mass_pruned"]["peak_allocated_bytes"] / 1e6
            ),
        }

    benchmark_table = {}
    for run_label, checkpoint_label in BENCHMARK_LABELS.items():
        result = benchmark["results"][checkpoint_label]
        benchmark_table[run_label] = {
            "parameters": int(result["total_parameters"]),
            "trainable_parameters": int(result["trainable_parameters"]),
            "checkpoint_mb": float(result["checkpoint_size_bytes"] / 1e6),
            "prepared_decoder_median_ms": 1000.0 * float(result["prepared_decoder"]["median_seconds"]),
            "full_forward_median_ms": 1000.0 * float(result["full_forward"]["median_seconds"]),
            "decoder_incremental_peak_mb": float(
                result["prepared_decoder"]["incremental_peak_allocated_bytes"] / 1e6
            ),
            "full_forward_incremental_peak_mb": float(
                result["full_forward"]["incremental_peak_allocated_bytes"] / 1e6
            ),
        }

    capacity = {
        "k6_fluid_normalized_mse": accuracy_value(
            capacity_accuracy, "exchangeable_1301_k6", "fluid", "__all__", "pooled_normalized_mse"
        ),
        "k8_fluid_normalized_mse": accuracy_value(
            capacity_accuracy, "exchangeable_1301_k8", "fluid", "__all__", "pooled_normalized_mse"
        ),
        "k6_environment_effective_rank": metric_median(
            capacity_topology, "exchangeable_1301_k6", "environment_effective_rank"
        ),
        "k8_environment_effective_rank": metric_median(
            capacity_topology, "exchangeable_1301_k8", "environment_effective_rank"
        ),
        "k6_query_effective_rank": metric_median(
            capacity_topology, "exchangeable_1301_k6", "query_effective_rank"
        ),
        "k8_query_effective_rank": metric_median(
            capacity_topology, "exchangeable_1301_k8", "query_effective_rank"
        ),
        "accuracy_case_count": int(
            capacity_accuracy["summary"]["exchangeable_1301_k6/fluid/__all__"]["case_count"]
        ),
        "topology_case_count": int(
            capacity_topology["provenance"]["exchangeable_1301_k6"]["evaluated_case_count"]
        ),
    }

    # Convergence curves: matched first 2500 epochs, with a 25-epoch rolling median.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, frame in metrics.items():
        shown = frame[frame["epoch"] <= 2500]
        for ax, column, title in [
            (axes[0], "val_field_mse", "Validation field MSE"),
            (axes[1], "val_temperature_mse", "Validation temperature MSE"),
        ]:
            curve = shown[column].rolling(25, min_periods=1).median()
            ax.plot(shown["epoch"], curve, label=label, color=COLORS[label], linewidth=1.8)
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("25-epoch trailing median")
            ax.set_title(title)
            ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Matched-budget convergence (epochs 1–2500)")
    fig.savefig(OUT / "matched_budget_convergence.png", dpi=180)
    plt.close(fig)

    # Continuation view for the only two runs that reached epoch 5000.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label in ("Run 1000 classic", "Run 1304 fixed residual uniform LR"):
        frame = metrics[label]
        shown = frame[frame["epoch"] <= 5000]
        for ax, column, title in [
            (axes[0], "val_field_mse", "Validation field MSE"),
            (axes[1], "val_temperature_mse", "Validation temperature MSE"),
        ]:
            curve = shown[column].rolling(25, min_periods=1).median()
            ax.plot(shown["epoch"], curve, label=label, color=COLORS[label], linewidth=1.8)
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("25-epoch trailing median")
            ax.set_title(title)
            ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Five-thousand-epoch convergence: classic Run 1000 vs uniform-LR Run 1304")
    fig.savefig(OUT / "extended_convergence.png", dpi=180)
    plt.close(fig)

    # Organizer structure: raw, independently interpretable metrics (no composite score).
    panels = [
        ("environment_edge_column_cosine", "Environment profile cosine", "lower is more distinct"),
        ("environment_effective_rank", "Environment effective rank", "higher is richer"),
        ("query_edge_column_cosine", "Query profile cosine", "lower is more distinct"),
        ("query_effective_rank", "Query effective rank", "higher is richer"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    labels = list(CHECKPOINT_LABELS)
    short = ["1000\nclassic", "1301\nexch.", "1302\ndesc.", "1303\nresid.", "1304\n@2500"]
    for ax, (metric, title, subtitle) in zip(axes.flat, panels):
        values = [topology_table[label][metric] for label in labels]
        ax.bar(short, values, color=[COLORS[label] for label in labels], width=0.65)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Full 90-case organizer/routing medians: matched Stage-5 epoch 2500")
    fig.savefig(OUT / "organizer_quality.png", dpi=180)
    plt.close(fig)

    # Fluid normalized accuracy by physical channel.
    fig, ax = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    x = np.arange(len(channels))
    width = 0.15
    for index, label in enumerate(labels):
        values = [selected_accuracy[label]["channels"][channel] for channel in channels]
        ax.bar(x + (index - 2.0) * width, values, width, label=label, color=COLORS[label])
    ax.set_xticks(x, channels)
    ax.set_yscale("log")
    ax.set_ylabel("Pooled normalized MSE (log scale)")
    ax.set_title("Complete 90-case accuracy: matched Stage-5 epoch 2500")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "accuracy_by_channel.png", dpi=180)
    plt.close(fig)

    # Identical retained-mass pruning comparison.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    modern = [
        "Run 1301 exchangeable", "Run 1302 fixed descriptor", "Run 1303 fixed residual",
        "Run 1304 fixed residual uniform LR @2500", "Run 1304 fixed residual uniform LR @5000",
    ]
    modern_short = ["1301 exch.", "1302 desc.", "1303 resid.", "1304 @2500", "1304 @5000"]
    modern_colors = [
        COLORS.get(label, COLORS["Run 1304 fixed residual uniform LR"]) for label in modern
    ]
    route_values = [[100 * pruning_table[label][key] for label in modern] for key in (
        "query_route_reduction", "module_route_reduction"
    )]
    xx = np.arange(len(modern))
    axes[0].bar(xx - 0.17, route_values[0], 0.34, label="query-edge", color="#F2CF5B")
    axes[0].bar(xx + 0.17, route_values[1], 0.34, label="module", color="#72B7B2")
    axes[0].set_xticks(xx, modern_short, rotation=12)
    axes[0].set_ylabel("Route reduction (%)")
    axes[0].set_title("Routes removed")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(
        modern_short,
        [100 * pruning_table[label]["fluid_mse_degradation"] for label in modern],
        color=modern_colors,
    )
    axes[1].set_ylabel("Fluid MSE change (%)")
    axes[1].set_title("Accuracy cost")
    axes[1].tick_params(axis="x", rotation=12)
    axes[2].bar(
        modern_short,
        [pruning_table[label]["decoder_speedup"] for label in modern],
        color=modern_colors,
    )
    axes[2].axhline(1.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Dense / pruned decoder time")
    axes[2].set_title("Measured decoder speedup")
    axes[2].tick_params(axis="x", rotation=12)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Same retained-mass pruning rule: query 0.98, module 0.95")
    fig.savefig(OUT / "retained_mass_pruning.png", dpi=180)
    plt.close(fig)

    result = {
        "sources": {
            "accuracy": str(accuracy_path),
            "residual_accuracy": str(residual_accuracy_path),
            "topology": str(topology_path),
            "residual_topology": str(residual_topology_path),
            "pruning": str(pruning_path),
            "residual_pruning": str(residual_pruning_path),
            "uniform_accuracy": str(uniform_accuracy_path),
            "uniform_topology": str(uniform_topology_path),
            "uniform_pruning": str(uniform_pruning_path),
            "benchmark": str(benchmark_path),
            "capacity_accuracy": str(capacity_accuracy_path),
            "capacity_topology": str(capacity_topology_path),
        },
        "metric_inventory": metric_inventory,
        "convergence": convergence,
        "selected_accuracy": selected_accuracy,
        "case_win_fractions": case_wins,
        "topology_medians": topology_table,
        "uniform_checkpoint_progression": uniform_checkpoint_progression,
        "pruning": pruning_table,
        "benchmark": benchmark_table,
        "capacity_override_secondary_screen": capacity,
    }
    (OUT / "comparison_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
