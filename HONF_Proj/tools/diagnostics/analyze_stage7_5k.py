#!/usr/bin/env python3
"""Consolidate the matched-budget Stage-7 evaluation and render report figures.

This utility is analysis-only. It reads frozen checkpoints, run histories, and
outputs from the maintained accuracy/topology/benchmark evaluators. It does not
alter models, checkpoints, training state, or inference behavior.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS = ROOT / "diagnostics"
RUN_ROOT = ROOT / "Trained_Results" / "ThermalChannel" / "HONF_Forward_Runs"
OUT = DIAGNOSTICS / "stage7_5k_comparison"

RUNS = {
    "run1000": RUN_ROOT / "Run_1000_20260817_214356_enhanced_honf_pairwise",
    "run1304": RUN_ROOT / "Run_1304_20260822_232649_stage5_fixed_residual_concat_uniform_lr3e4",
    "run1401": RUN_ROOT / "Run_1401_20260823_151126_stage7_modern_structured_context",
}
CHECKPOINTS = {
    "run1000_best": RUNS["run1000"] / "best_by_field_mse_model.pt",
    "run1000_latest": RUNS["run1000"] / "latest_model.pt",
    "run1304_best": RUNS["run1304"] / "best_by_field_mse_model.pt",
    "run1304_e5000": RUNS["run1304"] / "epoch_5000_model.pt",
    "run1304_latest": RUNS["run1304"] / "latest_model.pt",
    "run1401_e500": RUNS["run1401"] / "epoch_0500_model.pt",
    "run1401_e1000": RUNS["run1401"] / "epoch_1000_model.pt",
    "run1401_e2500": RUNS["run1401"] / "epoch_2500_model.pt",
    "run1401_best": RUNS["run1401"] / "best_by_field_mse_model.pt",
    "run1401_e5000": RUNS["run1401"] / "epoch_5000_model.pt",
    "run1401_latest": RUNS["run1401"] / "latest_model.pt",
}
ACCURACY_PATH = DIAGNOSTICS / "stage7_5k_accuracy" / "accuracy_summary.json"
TOPOLOGY_PATH = DIAGNOSTICS / "stage7_5k_topology" / "topology_quality_summary.json"
RUN1304_TRAJECTORY_PATH = (
    DIAGNOSTICS / "stage5_final_1304_topology" / "topology_quality_summary.json"
)
BENCHMARK_PATH = DIAGNOSTICS / "stage7_5k_benchmark.json"

COLORS = {
    "Run 1000": "#4C78A8",
    "Run 1304": "#F28E2B",
    "Run 1401": "#59A14F",
}
CHANNELS = ["u", "v", "p", "omega", "temperature"]
MILESTONES = [500, 1000, 2500, 5000]
TOPOLOGY_METRICS = [
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
    "query_largest_dominant_occupancy",
    "query_global_mean_route_l1",
    "pairwise_edge_map_cosine",
    "pairwise_edge_map_effective_rank",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(member) for member in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_history(path: Path) -> list[dict[str, str]]:
    by_epoch: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_epoch[int(row["epoch"])] = row
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def trailing_median(rows: list[dict[str, str]], column: str, epoch: int, window: int = 50) -> float:
    values = [
        float(row[column])
        for row in rows
        if epoch - window + 1 <= int(row["epoch"]) <= epoch
    ]
    if len(values) != window:
        raise ValueError(f"Expected {window} {column} values ending at epoch {epoch}; got {len(values)}")
    return float(statistics.median(values))


def rolling_median(rows: list[dict[str, str]], column: str, window: int = 50) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=np.int64)
    values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    rolled = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(window - 1, len(values)):
        rolled[index] = np.median(values[index - window + 1 : index + 1])
    return epochs, rolled


def state_dict_equal(left_path: Path, right_path: Path) -> bool:
    left = torch.load(left_path, map_location="cpu", weights_only=False)["model_state_dict"]
    right = torch.load(right_path, map_location="cpu", weights_only=False)["model_state_dict"]
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def checkpoint_provenance(label: str, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    normalization = {
        "config": checkpoint.get("global_normalization_config"),
        "stats": checkpoint.get("global_normalization_stats"),
    }
    normalization_bytes = json.dumps(jsonable(normalization), sort_keys=True).encode("utf-8")
    return {
        "label": label,
        "path": str(path.resolve()),
        "epoch": int(checkpoint["epoch"]),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "dataset_id": checkpoint.get("dataset_id"),
        "dataset_schema": checkpoint.get("dataset_schema"),
        "dataset_fingerprint": checkpoint.get("dataset_fingerprint"),
        "channel_order": checkpoint.get("channel_order"),
        "normalization_sha256": hashlib.sha256(normalization_bytes).hexdigest(),
        "state_dict_key_count": len(checkpoint["model_state_dict"]),
    }


def accuracy_value(payload: dict[str, Any], label: str, region: str, channel: str) -> dict[str, Any]:
    return payload["summary"][f"{label}/{region}/{channel}"]


def topology_values(payload: dict[str, Any], label: str) -> dict[str, Any]:
    values = payload["summary"][f"{label}/final/selected"]
    return {metric: values[metric] for metric in TOPOLOGY_METRICS if metric in values}


def selected_config(config: dict[str, Any]) -> dict[str, Any]:
    core = config["model"]["core_honf"]
    training = config["training"]
    return {
        "organizer_mode": core.get("organizer_mode"),
        "num_hyperedges": core.get("num_hyperedges"),
        "mechanism_state_mode": core.get("mechanism_state_mode"),
        "use_hyper_mechanism_encoder": core.get("use_hyper_mechanism_encoder"),
        "field_assembly_mode": core.get("field_assembly_mode"),
        "module_assignment_normalizer": core.get("module_assignment_normalizer"),
        "environment_assignment_normalizer": core.get("environment_assignment_normalizer"),
        "query_assignment_normalizer": core.get("query_assignment_normalizer"),
        "routing_execution": core.get("routing_execution"),
        "learning_rate": training.get("learning_rate"),
        "organizer_learning_rate": training.get("organizer_learning_rate"),
    }


def build_provenance() -> dict[str, Any]:
    run_payload: dict[str, Any] = {}
    for label, run_path in RUNS.items():
        manifest = read_json(run_path / "run_manifest.json")
        config_path = run_path / "config_resolved.json"
        config = read_json(config_path)
        local_path = Path(config["local_checkpoint_provenance"])
        local_checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)
        run_payload[label] = {
            "run_directory": str(run_path.resolve()),
            "source_state": manifest["source_state"],
            "resolved_config": str(config_path.resolve()),
            "resolved_config_sha256": sha256_file(config_path),
            "selected_config": selected_config(config),
            "dataset": manifest["launch_resources"],
            "stage_a": {
                "path": str(local_path.resolve()),
                "sha256": sha256_file(local_path),
                "epoch": int(local_checkpoint["epoch"]),
            },
        }
    checkpoint_payload = {
        label: checkpoint_provenance(label, path) for label, path in CHECKPOINTS.items()
    }
    normalization_hashes = {
        checkpoint_payload[label]["normalization_sha256"]
        for label in ("run1000_best", "run1304_best", "run1401_best")
    }
    dataset_hashes = {
        checkpoint_payload[label]["dataset_fingerprint"]
        for label in ("run1000_best", "run1304_best", "run1401_best")
    }
    channel_orders = {
        tuple(checkpoint_payload[label]["channel_order"])
        for label in ("run1000_best", "run1304_best", "run1401_best")
    }
    reference_checkpoint = torch.load(
        CHECKPOINTS["run1401_best"], map_location="cpu", weights_only=False
    )
    return {
        "runs": run_payload,
        "checkpoints": checkpoint_payload,
        "consistency": {
            "dataset_fingerprint_identical": len(dataset_hashes) == 1,
            "channel_order_identical": len(channel_orders) == 1,
            "normalization_identical": len(normalization_hashes) == 1,
            "run1304_epoch5000_equals_latest_state_dict": state_dict_equal(
                CHECKPOINTS["run1304_e5000"], CHECKPOINTS["run1304_latest"]
            ),
            "run1401_epoch5000_equals_latest_state_dict": state_dict_equal(
                CHECKPOINTS["run1401_e5000"], CHECKPOINTS["run1401_latest"]
            ),
            "run1000_frozen_epoch5000_checkpoint_exists": any(
                (RUNS["run1000"] / name).exists()
                for name in ("epoch_5000_model.pt", "epoch_05000_model.pt")
            ),
        },
        "normalization": jsonable(
            {
                "channel_order": reference_checkpoint["channel_order"],
                "config": reference_checkpoint["global_normalization_config"],
                "stats": reference_checkpoint["global_normalization_stats"],
            }
        ),
    }


def plot_convergence(histories: dict[str, list[dict[str, str]]]) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.5), constrained_layout=True)
    for key, display in (("run1000", "Run 1000"), ("run1304", "Run 1304"), ("run1401", "Run 1401")):
        epochs, values = rolling_median(histories[key], "val_field_mse")
        ax.plot(epochs, values, color=COLORS[display], linewidth=2.0, label=display)
    for epoch in MILESTONES:
        ax.axvline(epoch, color="#C7C7C7", linewidth=0.7, alpha=0.55)
    ax.set(
        title="Matched-budget validation convergence",
        xlabel="Epoch",
        ylabel="Trailing-50 median validation field MSE",
        yscale="log",
        xlim=(0, 5000),
    )
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(OUT / "matched_budget_convergence.png", dpi=180)
    plt.close(fig)


def plot_accuracy(accuracy: dict[str, Any]) -> None:
    labels = ["Run 1000\nbest (e9655)", "Run 1304\nbest (e4793)", "Run 1304\ne5000", "Run 1401\nbest (e4585)", "Run 1401\ne5000"]
    keys = ["run1000_best", "run1304_best", "run1304_e5000", "run1401_best", "run1401_e5000"]
    colors = [COLORS["Run 1000"], COLORS["Run 1304"], COLORS["Run 1304"], COLORS["Run 1401"], COLORS["Run 1401"]]
    values = [accuracy_value(accuracy, key, "fluid", "__all__")["pooled_normalized_mse"] for key in keys]
    fig, ax = plt.subplots(figsize=(9.5, 5.3), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors)
    for bar, value, alpha in zip(bars, values, [1.0, 1.0, 0.55, 1.0, 0.55]):
        bar.set_alpha(alpha)
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3e}", ha="center", va="bottom", fontsize=9)
    ax.set(title="Frozen-checkpoint complete-split accuracy", ylabel="Pooled normalized fluid MSE")
    ax.grid(True, axis="y", alpha=0.22)
    fig.savefig(OUT / "full_split_accuracy.png", dpi=180)
    plt.close(fig)


def plot_topology_gates(topology: dict[str, Any]) -> None:
    labels = ["Run 1000 best", "Run 1304 best", "Run 1401 best", "Run 1401 e5000"]
    keys = ["run1000_best", "run1304_best", "run1401_best", "run1401_e5000"]
    colors = [COLORS["Run 1000"], COLORS["Run 1304"], COLORS["Run 1401"], COLORS["Run 1401"]]
    metrics = [
        ("environment_edge_column_cosine", "Environment profile cosine", 0.20, "lower"),
        ("environment_effective_rank", "Environment effective rank", 3.5, "higher"),
        ("query_edge_column_cosine", "Query profile cosine", 0.55, "lower"),
        ("query_effective_rank", "Query effective rank", 3.0, "higher"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for ax, (metric, title, threshold, direction) in zip(axes.flat, metrics):
        values = [topology_values(topology, key)[metric]["median"] for key in keys]
        bars = ax.bar(range(len(keys)), values, color=colors)
        bars[-1].set_alpha(0.55)
        ax.axhline(threshold, color="#E45756", linestyle="--", linewidth=1.3, label=f"preferred {direction}: {threshold:g}")
        ax.set_title(title)
        ax.set_xticks(range(len(keys)), labels, rotation=18, ha="right")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Full-split organizer and query-routing decision gates", fontsize=14)
    fig.savefig(OUT / "topology_decision_gates.png", dpi=180)
    plt.close(fig)


def plot_topology_evolution(topology: dict[str, Any], run1304: dict[str, Any]) -> None:
    stage7_labels = {500: "run1401_e500", 1000: "run1401_e1000", 2500: "run1401_e2500", 5000: "run1401_e5000"}
    stage5_labels = {500: "uniform_1304_epoch0500", 1000: "uniform_1304_epoch1000", 2500: "uniform_1304_epoch2500", 5000: "uniform_1304_latest5000"}
    metrics = [
        ("environment_effective_rank", "Environment effective rank"),
        ("environment_edge_column_cosine", "Environment profile cosine"),
        ("query_effective_rank", "Query effective rank"),
        ("query_edge_column_cosine", "Query profile cosine"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        stage7_values = [topology_values(topology, stage7_labels[e])[metric]["median"] for e in MILESTONES]
        stage5_values = [
            run1304["summary"][f"{stage5_labels[e]}/final/selected"][metric]["median"]
            for e in MILESTONES
        ]
        ax.plot(MILESTONES, stage7_values, marker="o", color=COLORS["Run 1401"], linewidth=2, label="Run 1401")
        ax.plot(MILESTONES, stage5_values, marker="s", color=COLORS["Run 1304"], linewidth=2, label="Run 1304")
        ax.set(title=title, xlabel="Epoch")
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False)
    fig.suptitle("Topology evolution at matched milestones", fontsize=14)
    fig.savefig(OUT / "topology_evolution.png", dpi=180)
    plt.close(fig)


def plot_efficiency(benchmark: dict[str, Any]) -> None:
    keys = ["run1000_best", "run1304_best", "run1401_best", "run1401_e5000"]
    labels = ["Run 1000 best", "Run 1304 best", "Run 1401 best", "Run 1401 e5000"]
    colors = [COLORS["Run 1000"], COLORS["Run 1304"], COLORS["Run 1401"], COLORS["Run 1401"]]
    full_ms = [benchmark["results"][key]["full_forward"]["median_seconds"] * 1000 for key in keys]
    decoder_ms = [benchmark["results"][key]["prepared_decoder"]["median_seconds"] * 1000 for key in keys]
    memory_mib = [benchmark["results"][key]["full_forward"]["incremental_peak_allocated_bytes"] / 2**20 for key in keys]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    axes[0].bar(x - 0.18, full_ms, 0.36, color=colors, label="Full forward")
    axes[0].bar(x + 0.18, decoder_ms, 0.36, color=colors, alpha=0.45, label="Prepared decoder")
    axes[0].set(title="Synchronized decoder latency", ylabel="Median milliseconds")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.22)
    memory_bars = axes[1].bar(x, memory_mib, color=colors)
    memory_bars[-1].set_alpha(0.55)
    axes[1].set(title="Incremental peak allocated memory", ylabel="MiB")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].grid(True, axis="y", alpha=0.22)
    fig.savefig(OUT / "controlled_gpu_efficiency.png", dpi=180)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy = read_json(ACCURACY_PATH)
    topology = read_json(TOPOLOGY_PATH)
    run1304_trajectory = read_json(RUN1304_TRAJECTORY_PATH)
    benchmark = read_json(BENCHMARK_PATH)
    histories = {label: read_history(path / "metrics.csv") for label, path in RUNS.items()}

    convergence: dict[str, Any] = {}
    convergence_rows: list[dict[str, Any]] = []
    for label, rows in histories.items():
        convergence[label] = {
            "max_epoch": max(int(row["epoch"]) for row in rows),
            "trailing_window_epochs": 50,
            "val_field_mse": {},
            "val_temperature_mse": {},
        }
        for epoch in MILESTONES:
            field = trailing_median(rows, "val_field_mse", epoch)
            temperature = trailing_median(rows, "val_temperature_mse", epoch)
            convergence[label]["val_field_mse"][str(epoch)] = field
            convergence[label]["val_temperature_mse"][str(epoch)] = temperature
            convergence_rows.append(
                {"run": label, "epoch": epoch, "trailing50_val_field_mse": field, "trailing50_val_temperature_mse": temperature}
            )

    accuracy_selected: dict[str, Any] = {}
    for label in ("run1000_best", "run1304_best", "run1304_e5000", "run1401_best", "run1401_e5000"):
        fluid = accuracy_value(accuracy, label, "fluid", "__all__")
        accuracy_selected[label] = {
            "epoch": accuracy["provenance"][label]["epoch"],
            "pooled_normalized_fluid_mse": fluid["pooled_normalized_mse"],
            "pooled_normalized_relative_l2": fluid["pooled_normalized_relative_l2"],
            "median_case_mse": fluid["case_normalized_mse_median"],
            "p95_case_mse": fluid["case_normalized_mse_p95"],
            "channels": {
                channel: accuracy_value(accuracy, label, "fluid", channel)["pooled_normalized_mse"]
                for channel in CHANNELS
            },
            "near_interface_channels": {
                channel: accuracy_value(accuracy, label, "near_interface_fluid", channel)["pooled_normalized_mse"]
                for channel in CHANNELS
            },
        }

    topology_selected = {
        label: topology_values(topology, label)
        for label in topology["provenance"]
    }
    run1304_trajectory_labels = {
        500: "uniform_1304_epoch0500",
        1000: "uniform_1304_epoch1000",
        2500: "uniform_1304_epoch2500",
        5000: "uniform_1304_latest5000",
    }
    topology_trajectories = {
        "run1401": {
            str(epoch): topology_selected[
                {500: "run1401_e500", 1000: "run1401_e1000", 2500: "run1401_e2500", 5000: "run1401_e5000"}[epoch]
            ]
            for epoch in MILESTONES
        },
        "run1304": {
            str(epoch): {
                metric: run1304_trajectory["summary"][
                    f"{run1304_trajectory_labels[epoch]}/final/selected"
                ][metric]
                for metric in TOPOLOGY_METRICS
                if metric in run1304_trajectory["summary"][
                    f"{run1304_trajectory_labels[epoch]}/final/selected"
                ]
            }
            for epoch in MILESTONES
        },
    }
    efficiency = benchmark["results"]
    run1401_history_ratio = (
        convergence["run1401"]["val_field_mse"]["5000"]
        / convergence["run1000"]["val_field_mse"]["5000"]
    )
    run1401_best = accuracy_selected["run1401_best"]["pooled_normalized_fluid_mse"]
    run1304_best = accuracy_selected["run1304_best"]["pooled_normalized_fluid_mse"]
    run1000_best = accuracy_selected["run1000_best"]["pooled_normalized_fluid_mse"]
    run1401_topology = topology_selected["run1401_best"]
    full_forward_ratio = (
        efficiency["run1401_best"]["full_forward"]["median_seconds"]
        / efficiency["run1000_best"]["full_forward"]["median_seconds"]
    )
    gates = {
        "environment_profile_cosine_below_0p20": run1401_topology["environment_edge_column_cosine"]["median"] < 0.20,
        "environment_effective_rank_above_3p5": run1401_topology["environment_effective_rank"]["median"] > 3.5,
        "environment_largest_occupancy_below_0p50": run1401_topology["environment_largest_dominant_occupancy"]["median"] < 0.50,
        "region_separation_above_0p20": run1401_topology["region_separation_normalized"]["median"] > 0.20,
        "query_profile_cosine_below_0p55": run1401_topology["query_edge_column_cosine"]["median"] < 0.55,
        "query_effective_rank_above_3p0": run1401_topology["query_effective_rank"]["median"] > 3.0,
        "matched_history_ratio_below_1p10": run1401_history_ratio <= 1.10,
        "full_forward_ratio_below_1p10": full_forward_ratio <= 1.10,
        "parameter_count_same_as_run1000": efficiency["run1401_best"]["total_parameters"] == efficiency["run1000_best"]["total_parameters"],
    }

    provenance = build_provenance()
    payload = {
        "analysis": "Stage7 Run1000/1304/1401 matched-budget evaluation",
        "accuracy": accuracy_selected,
        "convergence": convergence,
        "topology": topology_selected,
        "topology_trajectories": topology_trajectories,
        "efficiency": efficiency,
        "ratios": {
            "run1401_to_run1000_trailing50_val_field_mse_at_5000": run1401_history_ratio,
            "run1401_best_to_run1304_best_complete_split_mse": run1401_best / run1304_best,
            "run1401_best_to_run1000_best_complete_split_mse": run1401_best / run1000_best,
            "run1401_to_run1000_full_forward_median": full_forward_ratio,
        },
        "decision_gates": gates,
        "provenance": provenance,
        "raw_artifacts": {
            "accuracy": str(ACCURACY_PATH.resolve()),
            "topology": str(TOPOLOGY_PATH.resolve()),
            "run1304_topology_trajectory": str(RUN1304_TRAJECTORY_PATH.resolve()),
            "benchmark": str(BENCHMARK_PATH.resolve()),
            "representative_cases": str((DIAGNOSTICS / "stage7_5k_representative_cases").resolve()),
        },
    }
    if not all(provenance["consistency"][key] for key in (
        "dataset_fingerprint_identical", "channel_order_identical", "normalization_identical"
    )):
        raise RuntimeError("Dataset/channel/normalization provenance mismatch")
    if not all(gates.values()):
        payload["decision_gate_note"] = "At least one Stage-7 aid did not pass; inspect the report rather than auto-accepting."

    (OUT / "comparison_summary.json").write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "matched_budget_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(convergence_rows[0]))
        writer.writeheader()
        writer.writerows(convergence_rows)

    plot_convergence(histories)
    plot_accuracy(accuracy)
    plot_topology_gates(topology)
    plot_topology_evolution(topology, run1304_trajectory)
    plot_efficiency(benchmark)
    print(f"[done] {OUT / 'comparison_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
