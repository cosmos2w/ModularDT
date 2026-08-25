#!/usr/bin/env python3
"""Consolidate the Stage-7 K-scaling audit and render reproducible report figures."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "Trained_Results" / "ThermalChannel" / "HONF_Forward_Runs"
GENERATED_ROOT = ROOT / "diagnostics" / "generated" / "stage7_k_audit"
INITIAL_ROOT = GENERATED_ROOT / "20260824_143720"
EXTENDED_ROOT = GENERATED_ROOT / "20260824_210713_extended"
OUT = EXTENDED_ROOT / "synthesis"

RUNS = {
    "run1401": RUN_ROOT / "Run_1401_20260823_151126_stage7_modern_structured_context",
    "run1601": RUN_ROOT / "Run_1601_20260824_112400_stage7_fused_query_module",
    "run1602": RUN_ROOT / "Run_1602_20260824_130922_stage7_k4_fused_audit",
    "run1603": RUN_ROOT / "Run_1603_20260824_135513_stage7_k8_fused_audit",
}
DISPLAY = {
    "run1401": "Run 1401 · K=6 baseline",
    "run1601": "Run 1601 · K=6 fused",
    "run1602": "Run 1602 · K=4 fused",
    "run1603": "Run 1603 · K=8 fused",
}
COLORS = {"run1401": "#2F6B9A", "run1601": "#C6922A", "run1602": "#D9782D", "run1603": "#748C36"}
CHANNELS = ("u", "v", "p", "omega", "temperature")
MILESTONES = {
    "run1401": {"e500": "run1401_e500", "e2500": "run1401_e2500", "e5000": "run1401_e5000", "best": "run1401_best"},
    "run1602": {"e500": "run1602_e500", "e2500": "run1602_e2500", "e5000": "run1602_e5000", "best": "run1602_best"},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        by_epoch = {int(row["epoch"]): row for row in csv.DictReader(handle)}
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def trailing_median(rows: list[dict[str, str]], column: str, epoch: int, window: int = 50) -> float:
    values = [float(row[column]) for row in rows if epoch - window + 1 <= int(row["epoch"]) <= epoch]
    if len(values) != window:
        raise ValueError(f"Expected {window} {column} values ending at epoch {epoch}; got {len(values)}")
    return float(statistics.median(values))


def accuracy(payloads: tuple[dict[str, Any], ...], checkpoint: str, region: str, channel: str = "__all__") -> dict[str, Any]:
    key = f"{checkpoint}/{region}/{channel}"
    for payload in payloads:
        if key in payload["summary"]:
            return payload["summary"][key]
    raise KeyError(key)


def provenance(payloads: tuple[dict[str, Any], ...], checkpoint: str) -> dict[str, Any]:
    for payload in payloads:
        if checkpoint in payload["provenance"]:
            return payload["provenance"][checkpoint]
    raise KeyError(checkpoint)


def topology(payloads: tuple[dict[str, Any], ...], checkpoint: str, metric: str) -> float:
    key = f"{checkpoint}/final/selected"
    for payload in payloads:
        if key in payload["summary"]:
            return float(payload["summary"][key][metric]["median"])
    raise KeyError(key)


def checkpoint_summary(accuracy_payloads: tuple[dict[str, Any], ...], topology_payloads: tuple[dict[str, Any], ...], checkpoint: str) -> dict[str, Any]:
    return {
        "provenance": provenance(accuracy_payloads, checkpoint),
        "accuracy": {
            "fluid_pooled_normalized_mse": accuracy(accuracy_payloads, checkpoint, "fluid")["pooled_normalized_mse"],
            "fluid_case_median": accuracy(accuracy_payloads, checkpoint, "fluid")["case_normalized_mse_median"],
            "fluid_case_p95": accuracy(accuracy_payloads, checkpoint, "fluid")["case_normalized_mse_p95"],
            "near_interface_pooled_normalized_mse": accuracy(accuracy_payloads, checkpoint, "near_interface_fluid")["pooled_normalized_mse"],
            "channels": {channel: accuracy(accuracy_payloads, checkpoint, "fluid", channel)["pooled_normalized_mse"] for channel in CHANNELS},
            "near_interface_channels": {channel: accuracy(accuracy_payloads, checkpoint, "near_interface_fluid", channel)["pooled_normalized_mse"] for channel in CHANNELS},
        },
        "topology": {
            metric: topology(topology_payloads, checkpoint, metric)
            for metric in (
                "environment_edge_column_cosine",
                "environment_effective_rank",
                "query_edge_column_cosine",
                "query_effective_rank",
                "region_separation_normalized",
                "pairwise_edge_map_cosine",
                "pairwise_edge_map_effective_rank",
            )
        },
    }


def build_summary() -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    initial_accuracy = read_json(INITIAL_ROOT / "accuracy" / "accuracy_summary.json")
    extended_accuracy = read_json(EXTENDED_ROOT / "accuracy" / "accuracy_summary.json")
    initial_topology = read_json(INITIAL_ROOT / "topology" / "topology_quality_summary.json")
    extended_topology = read_json(EXTENDED_ROOT / "topology" / "topology_quality_summary.json")
    benchmark = read_json(EXTENDED_ROOT / "benchmark.json")
    pruning = read_json(EXTENDED_ROOT / "pruning" / "diagnostics" / "retained_mass_pruning_summary.json")
    accuracy_payloads = (extended_accuracy, initial_accuracy)
    topology_payloads = (extended_topology, initial_topology)
    histories = {label: read_history(path / "metrics.csv") for label, path in RUNS.items()}
    runs: dict[str, Any] = {}
    for run in ("run1401", "run1602"):
        manifest = read_json(RUNS[run] / "run_manifest.json")
        runs[run] = {
            "display": DISPLAY[run],
            "run_directory": str(RUNS[run].resolve()),
            "status": manifest["status"],
            "last_completed_epoch": int(manifest["last_completed_epoch"]),
            "source_commit": manifest["source_state"]["commit"],
            "config_sha256": manifest["config_sha256"],
            "resumed_from": manifest.get("resumed_from"),
            "best_metrics": manifest["best_metrics"],
            "validation_trailing_50": {
                f"e{epoch}": {
                    "field_mse": trailing_median(histories[run], "val_field_mse", epoch),
                    "temperature_mse": trailing_median(histories[run], "val_temperature_mse", epoch),
                }
                for epoch in (500, 2500, 5000)
            },
            "checkpoints": {
                milestone: checkpoint_summary(accuracy_payloads, topology_payloads, checkpoint)
                for milestone, checkpoint in MILESTONES[run].items()
            },
        }
    for label, result in benchmark["results"].items():
        run = label.split("_")[0]
        runs.setdefault(run, {"display": DISPLAY[run]})
        runs[run].setdefault("benchmarks", {})[label.removeprefix(f"{run}_")] = {
            "epoch": result["epoch"],
            "total_parameters": result["total_parameters"],
            "state_dict_key_count": result["state_dict_key_count"],
            "prepared_median_ms": 1000.0 * result["prepared_decoder"]["median_seconds"],
            "prepared_p95_ms": 1000.0 * result["prepared_decoder"]["p95_seconds"],
            "full_median_ms": 1000.0 * result["full_forward"]["median_seconds"],
            "full_p95_ms": 1000.0 * result["full_forward"]["p95_seconds"],
        }
    for label, result in pruning["checkpoints"].items():
        run = label.split("_")[0]
        key = label.removeprefix(f"{run}_")
        row = pruning["summary"][f"{label}/beta_0.980/fluid/__all__"]
        dense = result["runtime"]["dense"]
        gathered = result["runtime"]["beta_0.980"]
        runs[run].setdefault("gathered_beta_098", {})[key] = {
            "epoch": result["epoch"],
            "dense_median_ms": 1000.0 * dense["median_seconds"],
            "gathered_median_ms": 1000.0 * gathered["median_seconds"],
            "runtime_reduction_fraction": 1.0 - gathered["median_seconds"] / dense["median_seconds"],
            "dense_peak_mib": dense["peak_allocated_bytes"] / 2**20,
            "gathered_peak_mib": gathered["peak_allocated_bytes"] / 2**20,
            "peak_memory_reduction_fraction": 1.0 - gathered["peak_allocated_bytes"] / dense["peak_allocated_bytes"],
            "retained_beta_mass_mean": result["policy_retention"]["beta_0.980"]["query_module_retained_beta_mass"]["mean"],
            "module_route_reduction_mean": row["module_route_reduction_mean"],
            "pooled_relative_mse_degradation": row["pooled_relative_mse_degradation"],
            "full_support_max_abs_difference": row["max_no_prune_output_difference"],
        }
    return {
        "initial_evidence_root": str(INITIAL_ROOT.resolve()),
        "extended_evidence_root": str(EXTENDED_ROOT.resolve()),
        "accuracy_scope": {"split": extended_accuracy["split"], "device": extended_accuracy["device"], "case_count": 90},
        "benchmark_scope": {key: benchmark[key] for key in ("device", "gpu_name", "case_id", "query_count", "warmup", "iterations")},
        "runs": runs,
    }, histories


def style_axis(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_convergence(histories: dict[str, list[dict[str, str]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for label, rows in histories.items():
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        for ax, column in zip(axes, ("val_field_mse", "val_temperature_mse"), strict=True):
            raw = np.asarray([float(row[column]) for row in rows])
            rolled = np.full(raw.shape, np.nan)
            for index in range(49, raw.size):
                rolled[index] = np.median(raw[index - 49 : index + 1])
            ax.plot(epochs, rolled, label=DISPLAY[label], color=COLORS[label], linewidth=2.0)
    for ax, title in zip(axes, ("Validation field MSE", "Validation temperature MSE"), strict=True):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("50-epoch trailing median")
        ax.set_title(title, loc="left", fontsize=11, weight="semibold")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.suptitle("Stage-7 K-audit convergence through the completed budgets", x=0.01, y=1.08, ha="left", fontsize=15, weight="bold")
    fig.text(0.01, 1.025, "Run 1602 resumed from its managed epoch-500 state; Runs 1601 and 1603 remain stopped at epoch 500.", fontsize=9, color="#555555")
    fig.savefig(OUT / "training_convergence_extended.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_milestone_accuracy(summary: dict[str, Any]) -> None:
    milestones = ("e500", "e2500", "e5000")
    x = np.arange(len(milestones))
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.7), constrained_layout=True)
    for run in ("run1401", "run1602"):
        values = summary["runs"][run]["checkpoints"]
        axes[0].plot(x, [values[m]["accuracy"]["fluid_pooled_normalized_mse"] for m in milestones], marker="o", linewidth=2.0, color=COLORS[run], label=DISPLAY[run])
        axes[1].plot(x, [values[m]["accuracy"]["near_interface_pooled_normalized_mse"] for m in milestones], marker="o", linewidth=2.0, color=COLORS[run])
    for ax, title in zip(axes[:2], ("Whole-fluid pooled MSE", "Near-interface pooled MSE"), strict=True):
        ax.set_xticks(x, ("500", "2,500", "5,000"))
        ax.set_xlabel("Checkpoint epoch")
        ax.set_ylabel("Normalized MSE")
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=11, weight="semibold")
        style_axis(ax)
    channels = np.arange(len(CHANNELS))
    reference = summary["runs"]["run1401"]["checkpoints"]["best"]["accuracy"]["channels"]
    candidate = summary["runs"]["run1602"]["checkpoints"]["best"]["accuracy"]["channels"]
    deltas = [100.0 * (candidate[channel] / reference[channel] - 1.0) for channel in CHANNELS]
    axes[2].bar(channels, deltas, color=["#748C36" if value < 0 else "#A5413D" for value in deltas], edgecolor="#333333", linewidth=0.6)
    axes[2].axhline(0.0, color="#333333", linewidth=0.9)
    axes[2].set_xticks(channels, CHANNELS, rotation=18, ha="right")
    axes[2].set_ylabel("Run 1602 relative change (%)")
    axes[2].set_title("Best-checkpoint channel tradeoff", loc="left", fontsize=11, weight="semibold")
    style_axis(axes[2])
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.suptitle("Run 1602 accuracy after continuation to epoch 5,000", x=0.01, y=1.08, ha="left", fontsize=15, weight="bold")
    fig.text(0.01, 1.025, "Held-out results cover all 90 test cases; negative channel deltas indicate lower error than accepted Run 1401 best.", fontsize=9, color="#555555")
    fig.savefig(OUT / "milestone_accuracy_extended.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_topology_trajectory(summary: dict[str, Any]) -> None:
    milestones = ("e500", "e2500", "e5000")
    x = np.arange(len(milestones))
    specs = (
        ("environment_effective_rank", "Effective rank", "Environment rank", 3.5),
        ("query_effective_rank", "Effective rank", "Query rank", 3.0),
        ("query_edge_column_cosine", "Cosine", "Query edge cosine", 0.55),
        ("region_separation_normalized", "Normalized separation", "Region separation", None),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    for ax, (metric, ylabel, title, gate) in zip(axes.flat, specs, strict=True):
        for run in ("run1401", "run1602"):
            values = summary["runs"][run]["checkpoints"]
            ax.plot(x, [values[m]["topology"][metric] for m in milestones], marker="o", linewidth=2.0, color=COLORS[run], label=DISPLAY[run])
        if gate is not None:
            ax.axhline(gate, color="#7A1F1F", linestyle="--", linewidth=1.2, label=f"{title} gate {gate:g}")
        ax.set_xticks(x, ("500", "2,500", "5,000"))
        ax.set_xlabel("Checkpoint epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11, weight="semibold")
        style_axis(ax)
    legend_entries: dict[str, Any] = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        legend_entries.update(zip(labels, handles, strict=True))
    fig.legend(legend_entries.values(), legend_entries.keys(), ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("Topology trajectory during the K=4 continuation", x=0.01, y=1.04, ha="left", fontsize=15, weight="bold")
    fig.text(0.01, 1.005, "Medians over the complete 90-case test split; K=4 gains accuracy while query effective rank crosses below the established gate.", fontsize=9, color="#555555")
    fig.savefig(OUT / "topology_trajectory_extended.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_efficiency(summary: dict[str, Any]) -> None:
    labels = ("run1401_best", "run1601_best", "run1602_best", "run1602_e5000", "run1603_best")
    names = ("1401 best", "1601 best", "1602 best", "1602 e5000", "1603 best")
    prepared = [summary["runs"][label.split("_")[0]]["benchmarks"][label.split("_", 1)[1]]["prepared_median_ms"] for label in labels]
    full = [summary["runs"][label.split("_")[0]]["benchmarks"][label.split("_", 1)[1]]["full_median_ms"] for label in labels]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    axes[0].bar(x - width / 2, prepared, width, label="Prepared decoder", color="#2F6B9A")
    axes[0].bar(x + width / 2, full, width, label="Full forward", color="#C6922A")
    axes[0].set_xticks(x, names, rotation=18, ha="right")
    axes[0].set_ylabel("Median milliseconds")
    axes[0].set_title("Controlled dense GPU-1 timing", loc="left", fontsize=11, weight="semibold")
    axes[0].legend(frameon=False)
    style_axis(axes[0])
    prune_labels = ("run1401_best", "run1602_best", "run1602_e5000")
    prune_names = ("1401 best", "1602 best", "1602 e5000")
    speed = [100.0 * summary["runs"][label.split("_")[0]]["gathered_beta_098"][label.split("_", 1)[1]]["runtime_reduction_fraction"] for label in prune_labels]
    memory = [100.0 * summary["runs"][label.split("_")[0]]["gathered_beta_098"][label.split("_", 1)[1]]["peak_memory_reduction_fraction"] for label in prune_labels]
    px = np.arange(len(prune_labels))
    axes[1].bar(px - width / 2, speed, width, label="Runtime reduction", color="#748C36")
    axes[1].bar(px + width / 2, memory, width, label="Peak-memory reduction", color="#7D5A9E")
    axes[1].set_xticks(px, prune_names, rotation=18, ha="right")
    axes[1].set_ylabel("Reduction versus fused dense (%)")
    axes[1].set_title("Gathered beta-0.98 execution", loc="left", fontsize=11, weight="semibold")
    axes[1].legend(frameon=False)
    style_axis(axes[1])
    fig.suptitle("Inference efficiency after the K=4 continuation", x=0.01, y=1.08, ha="left", fontsize=15, weight="bold")
    fig.text(0.01, 1.025, "RTX 6000 Ada, case 0273, 8,192 queries, 10 warmups, and 40 synchronized iterations.", fontsize=9, color="#555555")
    fig.savefig(OUT / "efficiency_extended.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, histories = build_summary()
    (OUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_convergence(histories)
    plot_milestone_accuracy(summary)
    plot_topology_trajectory(summary)
    plot_efficiency(summary)
    print(json.dumps({"output": str(OUT.resolve()), "runs": list(summary["runs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
