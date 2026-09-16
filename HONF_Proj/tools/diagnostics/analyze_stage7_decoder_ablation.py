#!/usr/bin/env python3
"""Reduce the mature Stage-7 decoder-context ablation study.

The reducer combines current matched evaluator passes for Runs 1401--1404
and 1804 with the previously validated historical trajectory tables.
It performs no model execution and never rewrites source run artifacts.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
STUDY = PROJECT / "diagnostics/generated/interface_operator_study/stage7_decoder_ablation"
RUNS = {
    "1401": ("Legacy", "Run_1401_20260823_151126_stage7_modern_structured_context"),
    "1402": ("No global", "Run_1402_20260915_174336_stage7_ablate_decoder_global"),
    "1403": ("No global/near", "Run_1403_20260915_174336_stage7_ablate_decoder_global_near"),
    "1404": ("Routing-only", "Run_1404_20260916_092508_routing_only_pairwise"),
    "1804": ("Dense", "Run_1804_20260905_081349_dense_pairwise_field_adaptation"),
}
COMPONENTS = {
    "global_field_fluid_norm": "Fluid field",
    "global_field_near_interface_norm": "Near-interface field",
    "global_field_far_fluid_norm": "Far-fluid field",
    "field_u_fluid_norm": "u",
    "field_v_fluid_norm": "v",
    "field_p_fluid_norm": "pressure",
    "field_omega_fluid_norm": "vorticity",
    "field_temperature_fluid_norm": "field temperature",
    "internal_temperature_physical": "internal temperature",
    "interface_t_surface_physical": "surface temperature",
    "interface_q_normal_physical": "normal heat flux",
    "port_t_env_final_physical": "final outside temperature",
}
ROUTING_METRICS = {
    "module_affinity_norm_l2": "module-affinity target relative L2",
    "active_edge_count_target_norm_l2": "active-edge-count target relative L2",
    "static_organization_module_mass_entropy_norm": "module mass entropy / log K",
    "static_organization_env_mass_entropy_norm": "environment mass entropy / log K",
    "static_organization_module_mass_max": "largest module mass",
    "static_organization_env_mass_max": "largest environment mass",
    "base_vs_final_module_mass_shift": "module mass shift",
    "base_vs_final_env_mass_shift": "environment mass shift",
    "routing_query_attention_effective_edges": "effective query hyperedges",
    "routing_query_attention_max": "largest conditional query weight",
    "routing_pairwise_edge_contribution_mean": "pairwise edge contribution norm",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def pooled(rows: list[dict[str, str]], base: str) -> dict[str, float] | None:
    keys = (f"{base}_sse", f"{base}_target_sse", f"{base}_num_values")
    if not rows or any(key not in rows[0] for key in keys):
        return None
    values = np.asarray([[float(row[key]) for key in keys] for row in rows], dtype=np.float64)
    sse, target_sse, count = values.sum(axis=0)
    if target_sse <= 0 or count <= 0:
        return None
    return {
        "sse": float(sse),
        "target_sse": float(target_sse),
        "num_values": int(count),
        "mse": float(sse / count),
        "relative_l2": float(np.sqrt(sse / target_sse)),
    }


def tables_for(run: str, kind: str = "exact") -> Path:
    if run == "1404":
        if kind in {"exact", "best"}:
            return STUDY / "run1404_evaluation/tables"
        if kind == "trajectory":
            return STUDY / "run1404_trajectory_evaluation/tables"
        raise ValueError(kind)
    if kind == "exact":
        name = "historical_evaluation" if run in {"1401", "1804"} else "evaluation"
    elif kind == "best":
        name = "best_field_evaluation"
    elif kind == "trajectory":
        name = "trajectory_evaluation"
    else:
        raise ValueError(kind)
    return STUDY / name / "tables"


def select_run(rows: list[dict[str, str]], run: str, epoch: int | None = None) -> list[dict[str, str]]:
    selected = [row for row in rows if f"Run_{run}_" in row.get("checkpoint", "")]
    if epoch is not None:
        selected = [row for row in selected if f"epoch_{epoch:04d}_model.pt" in row.get("checkpoint", "")]
    return selected


def exact_reconstruction() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, dict[str, str]]]]:
    headline: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    by_run: dict[str, dict[str, dict[str, str]]] = {}
    reference_targets: dict[str, tuple[float, float]] | None = None
    for run, (model, _) in RUNS.items():
        rows = select_run(read_csv(tables_for(run) / "per_case_metrics.csv"), run, 5000)
        if len(rows) != 90 or len({row["case_id"] for row in rows}) != 90:
            raise ValueError(f"Expected 90 unique exact-endpoint cases for Run {run}")
        by_run[run] = {row["case_id"]: row for row in rows}
        targets = {
            row["case_id"]: (
                float(row["global_field_fluid_norm_target_sse"]),
                float(row["global_field_fluid_norm_num_values"]),
            )
            for row in rows
        }
        if reference_targets is None:
            reference_targets = targets
        elif targets != reference_targets:
            raise ValueError(f"Target/count mismatch for Run {run}")
        errors = np.asarray([float(row["global_field_fluid_norm_l2"]) for row in rows])
        result = pooled(rows, "global_field_fluid_norm")
        assert result is not None
        headline.append(
            {
                "run": run,
                "model": model,
                **result,
                "equal_case_mean": float(errors.mean()),
                "median": float(np.median(errors)),
                "p95": float(np.quantile(errors, 0.95)),
                "maximum": float(errors.max()),
                "worst_case": rows[int(errors.argmax())]["case_id"],
            }
        )
        for base, label in COMPONENTS.items():
            result = pooled(rows, base)
            if result is not None:
                components.append({"run": run, "model": model, "base": base, "metric": label, **result})
    return headline, components, by_run


def paired_summary(by_run: dict[str, dict[str, dict[str, str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline, candidate in itertools.combinations(RUNS, 2):
        deltas = np.asarray(
            [
                float(by_run[candidate][case]["global_field_fluid_norm_l2"])
                - float(by_run[baseline][case]["global_field_fluid_norm_l2"])
                for case in sorted(by_run[baseline])
            ]
        )
        rows.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "candidate_wins": int((deltas < 0).sum()),
                "candidate_losses": int((deltas > 0).sum()),
                "mean_case_l2_delta": float(deltas.mean()),
                "median_case_l2_delta": float(np.median(deltas)),
            }
        )
    return rows


def trajectory() -> list[dict[str, Any]]:
    historical = read_csv(
        PROJECT / "diagnostics/generated/interface_operator_study/five_model_epoch5000/reduction/headline.csv"
    )
    rows: list[dict[str, Any]] = []
    for run, (model, _) in RUNS.items():
        for epoch in (500, 2500, 5000):
            if run in {"1401", "1804"}:
                item = next(row for row in historical if row["run"] == run and int(row["epoch"]) == epoch)
                value = float(item["relative_l2"])
            elif run == "1404" and epoch == 500:
                selected = select_run(
                    read_csv(
                        PROJECT
                        / "diagnostics/generated/interface_operator_study/run1404_routing_only/compare_exact_best_90/tables/per_case_metrics.csv"
                    ),
                    run,
                    epoch,
                )
                if len(selected) != 90:
                    raise ValueError("Expected 90 Run 1404 cases at epoch 500")
                result = pooled(selected, "global_field_fluid_norm")
                assert result is not None
                value = result["relative_l2"]
            elif epoch == 5000:
                exact = select_run(read_csv(tables_for(run) / "per_case_metrics.csv"), run, 5000)
                result = pooled(exact, "global_field_fluid_norm")
                assert result is not None
                value = result["relative_l2"]
            else:
                selected = select_run(read_csv(tables_for(run, "trajectory") / "per_case_metrics.csv"), run, epoch)
                if len(selected) != 90:
                    raise ValueError(f"Expected 90 trajectory cases for Run {run} at epoch {epoch}")
                result = pooled(selected, "global_field_fluid_norm")
                assert result is not None
                value = result["relative_l2"]
            rows.append({"run": run, "model": model, "epoch": epoch, "pooled_relative_l2": value})
    return rows


def best_selected() -> list[dict[str, Any]]:
    historical = read_csv(
        PROJECT
        / "diagnostics/generated/interface_operator_study/five_model_epoch5000/reduction/best_selected_headline.csv"
    )
    from honf_runtime.compat import load_trusted_checkpoint

    output: list[dict[str, Any]] = []
    for run, (model, run_name) in RUNS.items():
        if run in {"1401", "1804"}:
            item = next(row for row in historical if row["run"] == run)
            output.append(
                {
                    "run": run,
                    "model": model,
                    "checkpoint_epoch": int(item["checkpoint_epoch"]),
                    "relative_l2": float(item["relative_l2"]),
                    "equal_case_mean": float(item["equal_case_mean"]),
                    "p95": float(item["p95"]),
                    "maximum": float(item["max"]),
                }
            )
            continue
        rows = select_run(read_csv(tables_for(run, "best") / "per_case_metrics.csv"), run)
        if run == "1404":
            rows = [row for row in rows if row["checkpoint"].endswith("best_by_field_mse_model.pt")]
        if len(rows) != 90:
            raise ValueError(f"Expected 90 best-selected cases for Run {run}")
        errors = np.asarray([float(row["global_field_fluid_norm_l2"]) for row in rows])
        result = pooled(rows, "global_field_fluid_norm")
        assert result is not None
        checkpoint = PROJECT / "Trained_Results/ThermalChannel/HONF_Forward_Runs" / run_name / "best_by_field_mse_model.pt"
        payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
        output.append(
            {
                "run": run,
                "model": model,
                "checkpoint_epoch": int(payload["epoch"]),
                "relative_l2": result["relative_l2"],
                "equal_case_mean": float(errors.mean()),
                "p95": float(np.quantile(errors, 0.95)),
                "maximum": float(errors.max()),
            }
        )
    return output


def parse_current_history(run: str, run_name: str) -> dict[str, Any]:
    run_dir = PROJECT / "Trained_Results/ThermalChannel/HONF_Forward_Runs" / run_name
    raw_rows = read_csv(run_dir / "metrics.csv")
    rows_by_epoch = {int(row["epoch"]): row for row in raw_rows}
    rows = [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]
    if [int(row["epoch"]) for row in rows] != list(range(1, 5001)):
        raise ValueError(f"History is not consecutive through epoch 5000: Run {run}")

    def values(key: str) -> list[float]:
        return [value for row in rows if (value := finite(row.get(key))) is not None]

    val_field = values("val_field_mse")
    thresholds = (0.02, 0.01, 0.005, 0.003)
    first_below = {
        threshold: next(
            (int(row["epoch"]) for row in rows if float(row["val_field_mse"]) <= threshold),
            None,
        )
        for threshold in thresholds
    }
    stable_below = next(
        (
            int(rows[index]["epoch"])
            for index in range(len(rows) - 99)
            if max(float(row["val_field_mse"]) for row in rows[index : index + 100]) <= 0.01
        ),
        None,
    )
    inventory = json.loads((run_dir / "optimizer_group_inventory.json").read_text(encoding="utf-8"))
    parameters = sum(int(group["trainable_scalar_count"]) for group in inventory["groups"])
    train_wall = values("train_wall_seconds")
    val_wall = values("val_wall_seconds")
    peak_memory = values("peak_cuda_memory_mb")
    return {
        "run": run,
        "model": RUNS[run][0],
        "final_train_field_mse": float(rows[-1]["field_mse"]),
        "final_val_field_mse": float(rows[-1]["val_field_mse"]),
        "last100_val_field_median": float(statistics.median(val_field[-100:])),
        "best_val_field_mse": float(min(val_field)),
        "first_epoch_val_field_le_0p02": first_below[0.02],
        "first_epoch_val_field_le_0p01": first_below[0.01],
        "first_epoch_val_field_le_0p005": first_below[0.005],
        "first_epoch_val_field_le_0p003": first_below[0.003],
        "first_100_epoch_window_val_field_le_0p01": stable_below,
        "train_wall_hours": None if not train_wall else sum(train_wall) / 3600,
        "val_wall_hours": None if not val_wall else sum(val_wall) / 3600,
        "peak_cuda_memory_mib": None if not peak_memory else max(peak_memory),
        "trainable_parameters": parameters,
        "raw_history_rows": len(raw_rows),
        "canonical_history_rows": len(rows),
        "duplicate_history_rows": len(raw_rows) - len(rows),
    }


def training_summary() -> list[dict[str, Any]]:
    prior = json.loads(
        (
            PROJECT
            / "diagnostics/generated/interface_operator_study/five_model_epoch5000/history/history_summary.json"
        ).read_text(encoding="utf-8")
    )
    historical_thresholds = {
        "1401": {0.02: 461, 0.01: 831, 0.005: 1431, 0.003: 2013},
        "1804": {0.02: 357, 0.01: 651, 0.005: 1109, 0.003: 1764},
    }
    output: list[dict[str, Any]] = []
    for run, (model, run_name) in RUNS.items():
        if run not in historical_thresholds:
            output.append(parse_current_history(run, run_name))
            continue
        entry = prior["runs"][run]
        inventory = json.loads(
            (
                PROJECT
                / "Trained_Results/ThermalChannel/HONF_Forward_Runs"
                / run_name
                / "optimizer_group_inventory.json"
            ).read_text(encoding="utf-8")
        )
        parameters = sum(int(group["trainable_scalar_count"]) for group in inventory["groups"])
        timing = entry["timing_and_cost"]
        endpoint = entry["checkpoint_points"]["5000"]
        thresholds = historical_thresholds[run]
        output.append(
            {
                "run": run,
                "model": model,
                "final_train_field_mse": endpoint["field_mse"],
                "final_val_field_mse": endpoint["val_field_mse"],
                "last100_val_field_median": entry["recent_window"]["medians"]["val_field_mse"],
                "best_val_field_mse": entry["best_finite_metrics_through_5000"]["val_field_mse"]["value"],
                "first_epoch_val_field_le_0p02": thresholds[0.02],
                "first_epoch_val_field_le_0p01": thresholds[0.01],
                "first_epoch_val_field_le_0p005": thresholds[0.005],
                "first_epoch_val_field_le_0p003": thresholds[0.003],
                "first_100_epoch_window_val_field_le_0p01": None,
                "train_wall_hours": None
                if timing["logged_train_wall_seconds_sum"] is None
                else timing["logged_train_wall_seconds_sum"] / 3600,
                "val_wall_hours": None
                if timing["logged_val_wall_seconds_sum"] is None
                else timing["logged_val_wall_seconds_sum"] / 3600,
                "peak_cuda_memory_mib": timing["peak_cuda_memory_mb_max"],
                "trainable_parameters": parameters,
                "raw_history_rows": 5000,
                "canonical_history_rows": 5000,
                "duplicate_history_rows": 0,
            }
        )
    return output


def routing_summary() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run, (model, _) in RUNS.items():
        if run == "1804":
            for key, label in ROUTING_METRICS.items():
                output.append(
                    {
                        "run": run,
                        "model": model,
                        "metric": key,
                        "label": label,
                        "equal_case_mean": None,
                        "n": 0,
                        "status": "not_applicable_dense_has_no_learned_hyperedge_partition",
                    }
                )
            continue
        rows = select_run(read_csv(tables_for(run) / "hypergraph_case_metrics.csv"), run, 5000)
        if len(rows) != 90:
            raise ValueError(f"Expected 90 routing rows for Run {run}")
        for key, label in ROUTING_METRICS.items():
            values = [value for row in rows if (value := finite(row.get(key))) is not None]
            output.append(
                {
                    "run": run,
                    "model": model,
                    "metric": key,
                    "label": label,
                    "equal_case_mean": None if not values else float(np.mean(values)),
                    "minimum": None if not values else float(np.min(values)),
                    "maximum": None if not values else float(np.max(values)),
                    "n": len(values),
                    "status": "measured" if values else "not_exported",
                }
            )
    return output


def timing_summary() -> list[dict[str, Any]]:
    payloads = [
        json.loads((STUDY / "timing/four_model_timing.json").read_text(encoding="utf-8")),
        json.loads((STUDY / "timing/run1404_timing.json").read_text(encoding="utf-8")),
    ]
    labels = {"Legacy1401_at5000": "1401", "NoGlobal1402_at5000": "1402", "NoGlobalNear1403_at5000": "1403", "Routing1404_at5000": "1404", "Dense1804_at5000": "1804"}
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for model_payload in payload["models"]:
            run = labels[model_payload["checkpoint"]["label"]]
            for anchor in model_payload["real_anchors"]:
                normal = anchor["normal"]
                for phase, phase_values in normal["phases"].items():
                    rows.append(
                        {
                            "run": run,
                            "model": RUNS[run][0],
                            "kind": "real_anchor",
                            "case_id": anchor["case_id"],
                            "shape": "",
                            "phase": phase,
                            "median_ms": phase_values["median_ms"],
                            "mean_ms": phase_values["mean_ms"],
                            "incremental_peak_allocated_mib": phase_values["incremental_peak_allocated_bytes"] / 2**20,
                        }
                    )
            for synthetic in model_payload["synthetic_shapes"]:
                normal = synthetic["normal"]
                shape = synthetic["shape"]
                rows.append(
                    {
                        "run": run,
                        "model": RUNS[run][0],
                        "kind": "synthetic",
                        "case_id": "",
                        "shape": f"M{shape['M']}_E{shape['E']}_Q{shape['Q']}",
                        "phase": "full_forward",
                        "median_ms": normal["median_ms"],
                        "mean_ms": normal["mean_ms"],
                        "incremental_peak_allocated_mib": normal["incremental_peak_allocated_bytes"] / 2**20,
                    }
                )
    return rows


def anchor_summary(by_run: dict[str, dict[str, dict[str, str]]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case,
            "run": run,
            "model": RUNS[run][0],
            "field_relative_l2": float(by_run[run][case]["global_field_fluid_norm_l2"]),
        }
        for case in ("0273", "0653", "0298", "0302")
        for run in RUNS
    ]


def topology_quality_summary() -> list[dict[str, Any]]:
    """Extract matched full-holdout final/selected organization summaries."""
    path = STUDY / "topology_quality_full90/topology_quality_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))["summary"]
    payload.update(
        json.loads(
            (STUDY / "run1404_topology_quality_full90/topology_quality_summary.json").read_text(
                encoding="utf-8"
            )
        )["summary"]
    )
    metrics = (
        "module_row_entropy_norm",
        "module_row_effective_edges",
        "module_row_max",
        "module_largest_dominant_occupancy",
        "environment_row_entropy_norm",
        "environment_row_effective_edges",
        "environment_row_max",
        "environment_largest_dominant_occupancy",
        "environment_neighbor_dominant_agreement",
        "environment_neighbor_l1",
        "environment_spatial_smoothness",
        "source_separation_normalized",
        "region_separation_normalized",
        "base_to_final_alignment_cost",
        "base_to_final_module_assignment_l1",
        "base_to_final_environment_assignment_l1",
    )
    output: list[dict[str, Any]] = []
    for run in ("1401", "1402", "1403", "1404"):
        selected = payload[f"{run}_5000/final/selected"]
        for metric in metrics:
            item = selected[metric]
            output.append(
                {
                    "run": run,
                    "model": RUNS[run][0],
                    "metric": metric,
                    "equal_case_mean": item["mean"],
                    "median": item["median"],
                    "p05": item["p05"],
                    "p95": item["p95"],
                    "n": 90,
                }
            )
    return output


def run1404_usefulness_summary() -> list[dict[str, Any]]:
    payload = json.loads((STUDY / "run1404_interventions/summary.json").read_text(encoding="utf-8"))
    return [
        {
            "variant": name,
            "fluid_mse_mean": values["fluid_mse_mean"],
            "fluid_mse_delta_from_normal_mean": values["fluid_mse_delta_from_normal_mean"],
            "prediction_rms_difference_from_normal_mean": values[
                "prediction_rms_difference_from_normal_mean"
            ],
            "field_relative_l2_mean": values["field_relative_l2_mean"],
            "field_relative_l2_delta_from_normal_mean": values[
                "field_relative_l2_delta_from_normal_mean"
            ],
        }
        for name, values in payload["variants"].items()
    ]


def run1404_gradient_summary() -> list[dict[str, Any]]:
    payload = json.loads((STUDY / "run1404_endpoint_gradient.json").read_text(encoding="utf-8"))
    return [
        {
            "group": name,
            "gradient_tensor_count": values["gradient_tensor_count"],
            "grad_norm": values["grad_norm"],
            "finite": values["finite"],
        }
        for name, values in payload["gradient_groups"].items()
    ]


def run1404_retained_mass_summary() -> list[dict[str, Any]]:
    payload = json.loads(
        (
            STUDY
            / "run1404_retained_mass/diagnostics/retained_mass_pruning_summary.json"
        ).read_text(encoding="utf-8")
    )["checkpoints"]["exact5000"]
    return [
        {
            "policy": policy,
            "selected_module_count_mean": values["selected_module_count"]["mean"],
            "selected_module_count_min": values["selected_module_count"]["min"],
            "selected_module_count_max": values["selected_module_count"]["max"],
            "retained_beta_mass_mean": values["query_module_retained_beta_mass"]["mean"],
            "pair_mlp_rows_mean": values["actual_pair_mlp_evaluations_per_case"]["mean"],
            "dense_pair_mlp_rows": payload["dense_actual_pair_mlp_evaluations_per_case"]["mean"],
        }
        for policy, values in sorted(payload["policy_retention"].items())
    ]


def main() -> None:
    output = STUDY / "reduction"
    output.mkdir(parents=True, exist_ok=True)
    headline, components, by_run = exact_reconstruction()
    artifacts = {
        "headline": headline,
        "components": components,
        "paired": paired_summary(by_run),
        "trajectory": trajectory(),
        "best_selected": best_selected(),
        "training": training_summary(),
        "routing": routing_summary(),
        "topology_quality": topology_quality_summary(),
        "timing": timing_summary(),
        "anchors": anchor_summary(by_run),
        "run1404_usefulness": run1404_usefulness_summary(),
        "run1404_gradients": run1404_gradient_summary(),
        "run1404_retained_mass": run1404_retained_mass_summary(),
    }
    for name, rows in artifacts.items():
        write_csv(output / f"{name}.csv", rows)
    summary = {
        "schema_version": 1,
        "scope": "Runs 1401, 1402, 1403, 1404, and 1804 at exact epoch 5000, with matched trajectory and selection sensitivity",
        "case_count": 90,
        "query_count_per_case": 8192,
        "checkpoint_policy_primary": "exact_epoch_5000",
        "artifacts": artifacts,
        "limitations": [
            "The 90 cases are the repeatedly inspected development holdout, not an untouched test set.",
            "This is a one-seed comparison and does not estimate training-seed uncertainty.",
            "Historical training wall times were not collected under controlled identical conditions; controlled inference is reported separately.",
            "Dense has no learned hyperedge partition, so Stage-7 clustering metrics are structurally not applicable.",
            "Routing entropy and target-fit metrics are descriptive; reconstruction error determines usefulness.",
            "Run 1404 history rows 851--854 were duplicated by an interrupted resume; the reducer uses the final row for each epoch without rewriting the source history.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "tables": {name: len(rows) for name, rows in artifacts.items()}}, indent=2))


if __name__ == "__main__":
    main()
