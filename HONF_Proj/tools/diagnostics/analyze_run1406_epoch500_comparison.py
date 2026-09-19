#!/usr/bin/env python3
"""Reduce the matched Run-1406/1804/1404 epoch-500 evidence.

The reducer is deliberately read-only with respect to source runs.  It consumes
the standard 90-case comparison tables, the synchronized GPU-0 benchmark, and
the two untimed Run-1406 group-control maps.  Outputs are compact JSON/CSV
tables and two report figures.  Learned routes are never labelled as physical
causality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_DIR = Path(__file__).resolve().parent
if str(DIAGNOSTIC_DIR) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTIC_DIR))

import group_control_evidence as group_control


LABELS = ("Run1406_epoch500", "Run1804_epoch500", "Run1404_epoch500")
SHORT = {
    "Run1406_epoch500": "1406 Group-control",
    "Run1804_epoch500": "1804 Dense",
    "Run1404_epoch500": "1404 Routing-only",
}
COLORS = {
    "Run1406_epoch500": "#0072B2",
    "Run1804_epoch500": "#D55E00",
    "Run1404_epoch500": "#009E73",
}


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


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _accuracy_rows(per_case: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label in LABELS:
        rows = [row for row in per_case if row["model_label"] == label]
        if len(rows) != 90:
            raise ValueError(f"{label} has {len(rows)} cases; expected 90")
        values = [float(row["global_field_fluid_norm_l2"]) for row in rows]
        sse = sum(float(row["global_field_fluid_norm_sse"]) for row in rows)
        target_sse = sum(float(row["global_field_fluid_norm_target_sse"]) for row in rows)
        worst_index = int(np.argmax(values))
        result.append(
            {
                "model_label": label,
                "display_name": SHORT[label],
                "case_count": len(rows),
                "pooled_relative_l2": math.sqrt(sse / target_sse),
                "equal_case_mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p95": _quantile(values, 0.95),
                "maximum": max(values),
                "worst_case": rows[worst_index]["case_id"],
            }
        )
    return result


def _paired_rows(per_case: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_label = {
        label: {
            row["case_id"]: float(row["global_field_fluid_norm_l2"])
            for row in per_case
            if row["model_label"] == label
        }
        for label in LABELS
    }
    pairs = (
        ("Run1406_epoch500", "Run1804_epoch500"),
        ("Run1406_epoch500", "Run1404_epoch500"),
        ("Run1804_epoch500", "Run1404_epoch500"),
    )
    result: list[dict[str, Any]] = []
    for candidate, baseline in pairs:
        common = sorted(set(by_label[candidate]) & set(by_label[baseline]))
        if len(common) != 90:
            raise ValueError(f"{candidate}/{baseline} share {len(common)} cases; expected 90")
        deltas = [by_label[candidate][case] - by_label[baseline][case] for case in common]
        result.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "mean_delta": statistics.fmean(deltas),
                "median_delta": statistics.median(deltas),
                "candidate_wins": sum(value < 0.0 for value in deltas),
                "baseline_wins": sum(value > 0.0 for value in deltas),
                "ties": sum(value == 0.0 for value in deltas),
            }
        )
    return result


def _component_rows(model_summary: list[dict[str, str]]) -> list[dict[str, Any]]:
    metrics = {
        "fluid": "global_field_fluid_pooled_relative_l2",
        "near_interface": "global_field_near_interface_pooled_relative_l2",
        "far_fluid": "global_field_far_fluid_pooled_relative_l2",
        "u": "field_u_fluid_pooled_relative_l2",
        "v": "field_v_fluid_pooled_relative_l2",
        "pressure": "field_p_fluid_pooled_relative_l2",
        "vorticity": "field_omega_fluid_pooled_relative_l2",
        "field_temperature": "field_temperature_fluid_pooled_relative_l2",
        "internal_temperature": "internal_temperature_physical_pooled_relative_l2",
        "surface_temperature": "interface_t_surface_physical_pooled_relative_l2",
        "normal_heat_flux": "interface_q_normal_physical_pooled_relative_l2",
        "final_environment_temperature": "port_t_env_final_physical_pooled_relative_l2",
        "final_effective_h": "port_h_effective_final_physical_pooled_relative_l2",
    }
    by_label = {row["model_label"]: row for row in model_summary}
    return [
        {
            "metric": display,
            **{label: float(by_label[label][column]) for label in LABELS},
        }
        for display, column in metrics.items()
    ]


def _inference_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    wanted = {"full_forward", "prepared_decode", "physical_preparation_plus_one_query"}
    return [
        {
            "model_label": str(row["label"]).replace("_epoch500", ""),
            "case_id": row["case_id"],
            "phase": row["phase"],
            "median_ms": float(row["median_ms"]),
            "p05_ms": float(row["p05_ms"]),
            "p95_ms": float(row["p95_ms"]),
            "peak_allocated_mib": float(row["peak_allocated_bytes"]) / (1024.0**2),
            "peak_reserved_mib": float(row["peak_reserved_bytes"]) / (1024.0**2),
        }
        for row in payload["rows"]
        if row.get("phase") in wanted and row.get("status") == "ok"
    ]


def _training_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in payload["records"]:
        if record.get("status") != "complete":
            continue
        measurement = record["measurement"]
        latency = measurement["latency_seconds"]
        rows.append(
            {
                "model_label": record["checkpoint"]["label"],
                "bucket": record["bucket"]["label"],
                "median_ms": 1000.0 * float(latency["median"]),
                "min_ms": 1000.0 * float(latency["min"]),
                "max_ms": 1000.0 * float(latency["max"]),
                "peak_allocated_mib": float(measurement["peak_allocated_mib"]),
                "peak_reserved_mib": float(measurement["peak_reserved_mib"]),
                "optimizer_state_policy": measurement["optimizer_state_policy"],
            }
        )
    return rows


def _entropy_effective(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    row_sum = values.sum(axis=1, keepdims=True)
    probability = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum > 0.0)
    log_probability = np.zeros_like(probability)
    np.log(probability, out=log_probability, where=probability > 0.0)
    return np.exp(-(probability * log_probability).sum(axis=1))


def _map_rows(map_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(map_dir.glob("1406__*.npz")):
        arrays, metrics, _, metadata = group_control.load_group_control_npz(path)
        module = arrays.module_incidence[arrays.active_module_mask]
        environment = arrays.environment_incidence
        query = arrays.query_routing
        values = metrics.values
        rows.append(
            {
                "case_id": str(metadata["case_id"]),
                "K": arrays.group_count,
                "M_active": int(arrays.active_module_mask.sum()),
                "E_active": int(len(environment)),
                "module_exact_zero_fraction": float((module == 0.0).mean()),
                "environment_exact_zero_fraction": float((environment == 0.0).mean()),
                "query_exact_zero_fraction": float((query == 0.0).mean()),
                "module_degree_mean": float((module > 0.0).sum(axis=1).mean()),
                "environment_degree_mean": float((environment > 0.0).sum(axis=1).mean()),
                "query_degree_mean": float((query > 0.0).sum(axis=1).mean()),
                "module_effective_groups_mean": float(_entropy_effective(module).mean()),
                "environment_effective_groups_mean": float(_entropy_effective(environment).mean()),
                "query_effective_groups_mean": float(_entropy_effective(query).mean()),
                "empty_module_groups": int(((module > 0.0).sum(axis=0) == 0).sum()),
                "empty_environment_groups": int(((environment > 0.0).sum(axis=0) == 0).sum()),
                "empty_query_groups": int(((query > 0.0).sum(axis=0) == 0).sum()),
                "module_logical_paths": int(values["module_logical_path_count"]),
                "module_unique_pairs": int(values["module_unique_pair_count"]),
                "environment_logical_paths": int(values["environment_logical_path_count"]),
                "environment_unique_pairs": int(values["environment_unique_pair_count"]),
                "module_multiplicity": float(values["module_multiplicity"]),
                "environment_multiplicity": float(values["environment_multiplicity"]),
                "R_M": float(values["module_unique_over_dense_valid"]),
                "R_E": float(values["environment_unique_over_dense_valid"]),
            }
        )
    if len(rows) != 2:
        raise ValueError(f"expected two Run-1406 anchor maps, found {len(rows)} in {map_dir}")
    return rows


def _population_hypergraph(rows: list[dict[str, str]]) -> dict[str, Any]:
    if len(rows) != 90:
        raise ValueError(f"expected 90 group-control case rows, found {len(rows)}")
    metric_names = (
        "module_group_nonempty_count",
        "environment_group_nonempty_count",
        "module_source_degree_mean",
        "environment_source_degree_mean",
        "module_membership_effective_groups_mean",
        "environment_membership_effective_groups_mean",
        "query_group_degree_mean",
        "query_routing_entropy_norm_mean",
        "query_routing_effective_groups_mean",
        "module_multiplicity",
        "environment_multiplicity",
        "module_unique_over_dense_valid",
        "environment_unique_over_dense_valid",
        "module_center_separation_mean",
        "environment_center_separation_mean",
    )
    distributions: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        distributions[name] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    paths: dict[str, dict[str, float | int]] = {}
    for source in ("module", "environment"):
        logical = sum(int(float(row[f"{source}_logical_path_count"])) for row in rows)
        unique = sum(int(float(row[f"{source}_unique_pair_count"])) for row in rows)
        paths[source] = {
            "logical_path_count": logical,
            "unique_pair_count": unique,
            "pooled_multiplicity": logical / unique,
        }
    return {
        "case_count": len(rows),
        "failed_p2_actual_equals_unique": sum(
            row["p2_actual_equals_unique_status"] != "pass" for row in rows
        ),
        "distributions": distributions,
        "pooled_paths": paths,
    }


def _mean_phase(inference: list[dict[str, Any]], label: str, phase: str) -> float:
    values = [row["median_ms"] for row in inference if row["model_label"] == label and row["phase"] == phase]
    if len(values) != 2:
        raise ValueError(f"expected two timing rows for {label}/{phase}, found {len(values)}")
    return statistics.fmean(values)


def _run_number(label: str) -> str:
    return label.removeprefix("Run").removesuffix("_epoch500")


def _plot_summary(
    path: Path,
    accuracy: list[dict[str, Any]],
    inference: list[dict[str, Any]],
    training: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), constrained_layout=True)
    x = np.arange(len(LABELS))
    names = [SHORT[label] for label in LABELS]
    colors = [COLORS[label] for label in LABELS]
    acc = {row["model_label"]: row for row in accuracy}
    axes[0].bar(x, [acc[label]["pooled_relative_l2"] for label in LABELS], color=colors)
    axes[0].set_xticks(x, names, rotation=20, ha="right")
    axes[0].set_ylabel("pooled relative L2")
    axes[0].set_title("A  Matched 90-case accuracy ↓")
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.34
    full = [_mean_phase(inference, _run_number(label), "full_forward") for label in LABELS]
    decode = [_mean_phase(inference, _run_number(label), "prepared_decode") for label in LABELS]
    axes[1].bar(x - width / 2, full, width, label="full forward", color=colors, alpha=0.95)
    axes[1].bar(x + width / 2, decode, width, label="prepared P2", color=colors, alpha=0.45, hatch="//")
    axes[1].set_xticks(x, names, rotation=20, ha="right")
    axes[1].set_ylabel("median CUDA time (ms)")
    axes[1].set_title("B  Q=8192 synchronized inference ↓")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    train = {(row["model_label"], row["bucket"]): row for row in training}
    labels = [_run_number(label) for label in LABELS]
    m1 = [train[(label, "M1")]["median_ms"] for label in labels]
    m12 = [train[(label, "M12")]["median_ms"] for label in labels]
    axes[2].bar(x - width / 2, m1, width, label="M1", color=colors, alpha=0.55)
    axes[2].bar(x + width / 2, m12, width, label="M12", color=colors, alpha=0.95)
    axes[2].set_xticks(x, names, rotation=20, ha="right")
    axes[2].set_ylabel("optimizer-step median (ms)")
    axes[2].set_title("C  B48/Q1024 training step ↓")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("Run 1406 closes much of the accuracy gap, but its efficiency advantage is phase-dependent", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_hypergraph(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    cases = [row["case_id"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    for offset, key, label, color in (
        (-width, "module_degree_mean", "module→group degree", "#0072B2"),
        (0.0, "environment_degree_mean", "environment→group degree", "#009E73"),
        (width, "query_degree_mean", "query→group degree", "#D55E00"),
    ):
        axes[0].bar(x + offset, [row[key] for row in rows], width, label=label, color=color)
    axes[0].set_xticks(x, cases)
    axes[0].set_ylim(0.0, 6.4)
    axes[0].set_ylabel("mean exact nonzero degree")
    axes[0].set_title("A  Sparse source incidence, dense query routing")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    logical_m = [row["module_logical_paths"] / row["module_unique_pairs"] for row in rows]
    logical_e = [row["environment_logical_paths"] / row["environment_unique_pairs"] for row in rows]
    axes[1].bar(x - width / 2, logical_m, width, label="module logical/unique", color="#0072B2")
    axes[1].bar(x + width / 2, logical_e, width, label="environment logical/unique", color="#009E73")
    axes[1].axhline(1.0, color="#444444", linewidth=0.9, linestyle="--", label="one path per unique pair")
    axes[1].set_xticks(x, cases)
    axes[1].set_ylabel("logical path multiplicity")
    axes[1].set_title("B  Group paths coalesce to dense unique pairs")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Run 1406 learned interaction routes (not physical causality)", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=PROJECT / "diagnostics/generated/run1406_1804_1404_epoch500_90case",
    )
    parser.add_argument(
        "--efficiency-dir",
        type=Path,
        default=PROJECT / "diagnostics/generated/interface_operator_study/epoch500_run1406_1804_1404_efficiency_20260919",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=PROJECT / "diagnostics/generated/interface_operator_study/run1406_epoch500_hypergraph/maps/forward",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "diagnostics/generated/interface_operator_study/run1406_epoch500_comparison_report",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    comparison = args.comparison_dir.expanduser().resolve()
    efficiency = args.efficiency_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = comparison / "tables"
    per_case = _read_csv(tables / "per_case_metrics.csv")
    model_summary = _read_csv(tables / "model_summary_metrics.csv")
    group_population_rows = _read_csv(
        comparison / "group_control_1406/per_case_group_metrics.csv"
    )
    accuracy = _accuracy_rows(per_case)
    paired = _paired_rows(per_case)
    components = _component_rows(model_summary)
    inference_payload = json.loads((efficiency / "inference/gpu0_phase_timing.json").read_text(encoding="utf-8"))
    training_payload = json.loads((efficiency / "training/training_benchmark_epoch500.json").read_text(encoding="utf-8"))
    inference = _inference_rows(inference_payload)
    training = _training_rows(training_payload)
    maps = _map_rows(args.map_dir.expanduser().resolve())
    population_hypergraph = _population_hypergraph(group_population_rows)
    if len(inference) != 18:
        raise ValueError(f"expected 18 controlled inference rows, found {len(inference)}")
    if len(training) != 6:
        raise ValueError(f"expected six controlled training rows, found {len(training)}")

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "accuracy_summary.csv", accuracy)
    _write_csv(output / "paired_cases.csv", paired)
    _write_csv(output / "component_metrics.csv", components)
    _write_csv(output / "inference_efficiency.csv", inference)
    _write_csv(output / "training_efficiency.csv", training)
    _write_csv(output / "run1406_hypergraph_anchors.csv", maps)
    _write_json(output / "run1406_hypergraph_population_summary.json", population_hypergraph)
    _plot_summary(output / "accuracy_efficiency_summary.png", accuracy, inference, training)
    _plot_hypergraph(output / "run1406_hypergraph_summary.png", maps)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint_policy": "exact epoch 500 for Runs 1406, 1804, and 1404",
        "accuracy_population": "same 90-case test/development holdout; 8192 queries per case; predicted ports",
        "efficiency_protocol": inference_payload.get("scope"),
        "accuracy": accuracy,
        "paired": paired,
        "components": components,
        "inference": inference,
        "training": training,
        "run1406_hypergraph_anchors": maps,
        "run1406_hypergraph_population": population_hypergraph,
        "semantics": "learned interactions, not physical causality",
        "limitations": [
            "Run-1406 detailed group-control maps cover cases 0273 and 0653; full-population accuracy covers all 90 cases.",
            "Dense Run 1804 has no learned hypergraph, so its topology is non-applicable rather than zero.",
            "Run-1404 legacy soft hyperedge metrics and Run-1406 entmax incidence metrics have different semantics and are not numerically interchangeable.",
            "The held-out split is an established development holdout, not independent CFD validation.",
        ],
    }
    _write_json(output / "summary.json", payload)
    print(json.dumps({"status": "complete", "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
