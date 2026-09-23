"""Reduce the matched Run-1502/1501/Dense-1804 epoch-500 evidence.

The reducer reads maintained evaluator and sparse-incidence population outputs
in place.  It never loads a model or chooses a checkpoint from test error.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from analyze_honf_maturity import number, pooled, read_csv, write_csv

LABELS = (
    "Run1502_exact_epoch500",
    "Run1501_exact_epoch500",
    "Run1804_dense_epoch500",
)
RUN_PREFIXES = {
    "Run1502_exact_epoch500": "Run_1502_",
    "Run1501_exact_epoch500": "Run_1501_",
    "Run1804_dense_epoch500": "Run_1804_",
}
CHANNELS = ("u", "v", "p", "omega", "temperature")
POOLED_BASES = (
    "global_field_fluid_norm",
    "global_field_near_interface_norm",
    "global_field_far_fluid_norm",
    *(f"field_{channel}_fluid_norm" for channel in CHANNELS),
)
CASE_METRICS = (
    "global_field_fluid_norm_l2",
    "global_field_near_interface_norm_l2",
    "global_field_far_fluid_norm_l2",
    "field_temperature_fluid_physical_mae",
    "internal_temperature_physical_mae",
    "interface_t_surface_physical_mae",
    "interface_q_normal_physical_mae",
    "port_t_env_final_physical_mae",
    "port_h_effective_final_physical_mae",
    "mean_outlet_temperature_physical_abs_error",
    "pressure_drop_inlet_minus_outlet_physical_abs_error",
)
COST_FIELDS = (
    "evaluation_wall_time_seconds_mean",
    "evaluation_wall_time_seconds_median",
    "evaluation_wall_time_seconds_p95",
    "evaluation_queries_per_second_mean",
    "evaluation_cuda_incremental_peak_allocated_mib_mean",
    "evaluation_cuda_incremental_peak_reserved_mib_mean",
    "evaluation_cuda_peak_allocated_mib_mean",
)
STRUCTURE_FIELDS = (
    "query_degree_mean",
    "module_RM_support",
    "environment_RE_support",
    "module_source_degree_mean",
    "environment_source_degree_mean",
    "module_effective_groups_mean",
    "environment_effective_groups_mean",
    "module_logical_paths",
    "environment_logical_paths",
    "module_unique_pairs",
    "environment_unique_pairs",
    "module_multiplicity",
    "environment_multiplicity",
    "module_dense_valid_pairs",
    "environment_dense_valid_pairs",
    "p2_module_actual_rows",
    "p2_environment_actual_rows",
    "p2_module_padded_rows",
    "p2_environment_padded_rows",
    "p2_module_geometry_rows",
    "p2_environment_geometry_rows",
    "p2_module_content_rows",
    "p2_environment_content_rows",
    "kappa",
)


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "worst": float(np.max(array)),
    }


def _validated_checkpoint(label: str, raw_path: str) -> Path:
    checkpoint = Path(raw_path).expanduser().resolve()
    if checkpoint.name != "epoch_0500_model.pt":
        raise ValueError(f"{label} is not the exact epoch-500 checkpoint: {checkpoint}")
    expected_prefix = RUN_PREFIXES[label]
    if not checkpoint.parent.name.startswith(expected_prefix):
        raise ValueError(
            f"{label} checkpoint is not beneath the expected {expected_prefix} run: {checkpoint}"
        )
    return checkpoint


def _option_values(arguments: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"comparison manifest has a value-less {option}")
            values.append(arguments[index + 1])
        elif argument.startswith(f"{option}="):
            values.append(argument.split("=", 1)[1])
    return values


def _validate_comparison_manifest(comparison_dir: Path, checkpoints: dict[str, Path]) -> None:
    path = comparison_dir / "comparison_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("kind") != "compare":
        raise ValueError(f"comparison manifest is not a completed compare job: {path}")
    arguments = payload.get("arguments")
    if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
        raise ValueError("comparison manifest arguments are unavailable")
    if (
        "--allow-checkpoint-fallback" in arguments
        or "--return-routing-maps" in arguments
        or _option_values(arguments, "--Run_ID")
    ):
        raise ValueError("comparison must use explicit checkpoints with no fallback")
    labels = _option_values(arguments, "--label")
    requested = [Path(value).expanduser().resolve() for value in _option_values(arguments, "--checkpoint-path")]
    if labels != list(LABELS) or requested != [checkpoints[label] for label in LABELS]:
        raise ValueError("comparison manifest does not request the three labelled exact endpoints")
    expected_options = {
        "--split": ["test"],
        "--case-ratio": ["1.0"],
        "--query-batch-size": ["32768"],
        "--local-port-condition-mode": ["predicted"],
    }
    for option, expected in expected_options.items():
        if _option_values(arguments, option) != expected:
            raise ValueError(f"comparison manifest does not match the required {option} protocol")


def _group_population(path: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, Path]]:
    rows = read_csv(path)
    grouped = {label: [row for row in rows if row.get("model_label") == label] for label in LABELS}
    checkpoints: dict[str, Path] = {}
    for label, selected in grouped.items():
        case_ids = [row.get("case_id", "") for row in selected]
        if len(case_ids) != 90 or len(set(case_ids)) != 90:
            raise ValueError(f"{label} must contain 90 unique cases, found {len(case_ids)}")
        selected_checkpoints = {row.get("checkpoint", "") for row in selected}
        if len(selected_checkpoints) != 1:
            raise ValueError(f"{label} is not one explicit checkpoint: {selected_checkpoints}")
        checkpoints[label] = _validated_checkpoint(label, next(iter(selected_checkpoints)))
    reference = {row["case_id"]: row for row in grouped[LABELS[0]]}
    for label, selected in grouped.items():
        if {row["case_id"] for row in selected} != set(reference):
            raise ValueError(f"{label} does not share the fixed 90-case population")
        for row in selected:
            other = reference[row["case_id"]]
            for key, value in other.items():
                if not key.endswith(("_target_sse", "_num_values")):
                    continue
                left = number(row.get(key))
                right = number(value)
                if left is None and right is None:
                    continue
                if left is None or right is None or not np.isclose(left, right, rtol=1e-7, atol=1e-7):
                    raise ValueError(f"target mismatch for {label}/{row['case_id']}/{key}")
    return grouped, checkpoints


def _summary_value(payload: dict[str, Any], field: str) -> float | None:
    value = payload.get(field)
    if isinstance(value, dict):
        value = value.get("mean")
    return number(value)


def _validated_population_evidence(
    path: Path,
    *,
    label: str,
    expected_checkpoint: Path,
    expected_cases: set[str],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"{label} population evidence is incomplete")
    candidate = payload.get("candidate", {})
    if candidate.get("architecture") != "sparse_incidence_group_control_honf":
        raise ValueError(f"{label} population evidence has the wrong architecture")
    candidate_checkpoint = _validated_checkpoint(label, str(candidate.get("checkpoint", "")))
    if candidate_checkpoint != expected_checkpoint:
        raise ValueError(f"{label} population checkpoint does not match the evaluator checkpoint")
    protocol = payload.get("protocol", {})
    expected_protocol = {
        "split": "test",
        "local_port_condition_mode": "predicted",
        "query_count_requested": 1024,
        "query_batch_size": 1024,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"{label} population protocol mismatch for {key}")
    population = payload.get("population", {})
    if population.get("case_count") != 90:
        raise ValueError(f"{label} population evidence must contain 90 cases")
    if population.get("query_count_per_case") != [1024]:
        raise ValueError(f"{label} population evidence is not the Q1024 protocol")
    protocol_ids = protocol.get("case_ids", [])
    case_ids = [str(case.get("case_id", "")) for case in payload.get("cases", [])]
    if (
        len(protocol_ids) != 90
        or len(set(protocol_ids)) != 90
        or len(case_ids) != 90
        or len(set(case_ids)) != 90
        or set(protocol_ids) != expected_cases
        or set(case_ids) != expected_cases
    ):
        raise ValueError(f"{label} population case IDs do not match the evaluator population")
    return population


def _structure_row(label: str, population: dict[str, Any]) -> dict[str, Any]:
    row = {
        "model_label": label,
        **{field: _summary_value(population, field) for field in STRUCTURE_FIELDS},
    }
    if label == "Run1501_exact_epoch500" and row["p2_module_geometry_rows"] is None:
        for field in (
            "p2_module_padded_rows",
            "p2_environment_padded_rows",
            "p2_module_geometry_rows",
            "p2_environment_geometry_rows",
            "p2_module_content_rows",
            "p2_environment_content_rows",
        ):
            row[field] = "unavailable_legacy_accounting"
    return row


def reduce(
    comparison_dir: Path,
    evidence_1502: Path,
    evidence_1501: Path,
    output_dir: Path,
) -> dict[str, Any]:
    comparison_dir = comparison_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    grouped, checkpoints = _group_population(comparison_dir / "tables" / "per_case_metrics.csv")
    _validate_comparison_manifest(comparison_dir, checkpoints)

    pooled_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    headline: list[dict[str, Any]] = []
    for label, rows in grouped.items():
        fluid = pooled(rows, "global_field_fluid_norm")
        if fluid is None:
            raise ValueError(f"pooled fluid metric is unavailable for {label}")
        fluid_cases = [number(row.get("global_field_fluid_norm_l2")) for row in rows]
        if any(value is None for value in fluid_cases):
            raise ValueError(f"nonfinite fluid case metric for {label}")
        headline.append(
            {
                "model_label": label,
                "checkpoint": rows[0]["checkpoint"],
                "pooled_fluid_relative_l2": fluid["relative_l2"],
                **{f"equal_case_{key}": value for key, value in _distribution(fluid_cases).items()},
            }
        )
        for base in POOLED_BASES:
            value = pooled(rows, base)
            if value is None:
                raise ValueError(f"pooled metric {base} is unavailable for {label}")
            pooled_rows.append({"model_label": label, "metric": base, **value})
        for metric in CASE_METRICS:
            values = [number(row.get(metric)) for row in rows]
            if any(value is None for value in values):
                raise ValueError(f"case metric {metric} is unavailable for {label}")
            distribution_rows.append(
                {"model_label": label, "metric": metric, **_distribution(values)}
            )

    for left_label, right_label in itertools.combinations(LABELS, 2):
        right_by_case = {row["case_id"]: row for row in grouped[right_label]}
        for metric in CASE_METRICS:
            deltas = []
            for left in grouped[left_label]:
                left_value = number(left.get(metric))
                right_value = number(right_by_case[left["case_id"]].get(metric))
                if left_value is None or right_value is None:
                    raise ValueError(f"paired metric {metric} is unavailable")
                deltas.append(left_value - right_value)
            paired_rows.append(
                {
                    "left": left_label,
                    "right": right_label,
                    "metric": metric,
                    "mean_delta_left_minus_right": float(np.mean(deltas)),
                    "left_wins": int(sum(delta < 0.0 for delta in deltas)),
                    "ties": int(sum(delta == 0.0 for delta in deltas)),
                    "right_wins": int(sum(delta > 0.0 for delta in deltas)),
                }
            )

    cost_source = read_csv(comparison_dir / "tables" / "evaluation_cost_summary_metrics.csv")
    cost_by_label = {row["model_label"]: row for row in cost_source if row.get("model_label") in LABELS}
    if set(cost_by_label) != set(LABELS):
        raise ValueError("cost summary does not contain all three exact endpoint labels")
    cost_rows: list[dict[str, Any]] = []
    for label in LABELS:
        source = cost_by_label[label]
        if number(source.get("num_cases")) != 90:
            raise ValueError(f"cost summary for {label} is not a 90-case population")
        cost_checkpoint = _validated_checkpoint(label, source.get("checkpoint", ""))
        if cost_checkpoint != checkpoints[label]:
            raise ValueError(f"cost and per-case checkpoint mismatch for {label}")
        values = {field: number(source.get(field)) for field in COST_FIELDS}
        if any(value is None for value in values.values()):
            raise ValueError(f"cost summary for {label} has missing or nonfinite fields")
        cost_rows.append({"model_label": label, "checkpoint": str(cost_checkpoint), **values})

    expected_cases = {row["case_id"] for row in grouped[LABELS[0]]}
    cost_case_rows = read_csv(comparison_dir / "tables" / "evaluation_cost_case_metrics.csv")
    for label in LABELS:
        selected = [row for row in cost_case_rows if row.get("model_label") == label]
        if len(selected) != 90 or {row.get("case_id") for row in selected} != expected_cases:
            raise ValueError(f"cost case table for {label} does not match the 90-case population")
        for row in selected:
            if _validated_checkpoint(label, row.get("checkpoint", "")) != checkpoints[label]:
                raise ValueError(f"cost case checkpoint mismatch for {label}")
            if (
                row.get("split") != "test"
                or number(row.get("evaluation_grid_query_count")) != 8192
                or number(row.get("query_batch_size")) != 32768
            ):
                raise ValueError(f"cost case protocol mismatch for {label}/{row.get('case_id')}")
    summaries = {
        "Run1502_exact_epoch500": _validated_population_evidence(
            evidence_1502,
            label="Run1502_exact_epoch500",
            expected_checkpoint=checkpoints["Run1502_exact_epoch500"],
            expected_cases=expected_cases,
        ),
        "Run1501_exact_epoch500": _validated_population_evidence(
            evidence_1501,
            label="Run1501_exact_epoch500",
            expected_checkpoint=checkpoints["Run1501_exact_epoch500"],
            expected_cases=expected_cases,
        ),
    }
    structure_rows = [_structure_row(label, payload) for label, payload in summaries.items()]
    structure_rows.append(
        {
            "model_label": "Run1804_dense_epoch500",
            **{field: "unavailable_not_zero" for field in STRUCTURE_FIELDS},
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("headline.csv", headline),
        ("pooled_metrics.csv", pooled_rows),
        ("equal_case_distributions.csv", distribution_rows),
        ("paired_summary.csv", paired_rows),
        ("cost_summary.csv", cost_rows),
        ("sparse_structure_summary.csv", structure_rows),
    ):
        write_csv(output_dir / filename, rows)
    payload = {
        "status": "complete",
        "population": 90,
        "fidelity_query_count_per_case": 8192,
        "structure_query_count_per_case": 1024,
        "checkpoint_policy": "matched exact epoch 500",
        "headline": headline,
        "sources": {
            "comparison": str(comparison_dir),
            "run1502_structure": str(evidence_1502.resolve()),
            "run1501_structure": str(evidence_1501.resolve()),
        },
        "limitations": [
            "The 90 cases are the established development holdout, not untouched final validation.",
            "All models are one training seed; casewise variation is not seed uncertainty.",
            "Sparse-incidence organization is compared only between Runs 1502 and 1501.",
            "Dense Run 1804 sparse-support fields are unavailable, not zero.",
            (
                "The retained Run 1501 structure artifact predates corrected receiver-chunk "
                "padded-row and geometry/content accounting; those fields are unavailable."
            ),
            "Learned supports and group labels do not establish physical causality.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument(
        "--evidence-1502", required=True, type=Path, help="Run-1502 Q1024 evidence.json wrapper."
    )
    parser.add_argument(
        "--evidence-1501", required=True, type=Path, help="Run-1501 Q1024 evidence.json wrapper."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    payload = reduce(
        args.comparison_dir,
        args.evidence_1502,
        args.evidence_1501,
        args.output_dir,
    )
    print(json.dumps({"status": payload["status"], "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
