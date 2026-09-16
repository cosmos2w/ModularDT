#!/usr/bin/env python3
"""Analyze the four ThermalChannel routing runs at the 500-epoch budget.

This is a read-only reducer for already-produced metrics, endpoint tables,
routing ledgers, and the bounded mean-shift diagnostic.  The default path does
not load or execute a model, train, select a checkpoint outside the stated
budget, or rewrite any input artifact.  ``--verify-parameter-inventory`` is an
explicit CPU-only strict reconstruction of the four exact checkpoints and
writes its own generated count tables.  Exact epoch-500 endpoint rows are kept
separate from validation-selected rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT / "Trained_Results/ThermalChannel/HONF_Forward_Runs"
DEFAULT_RUN_DIRS = {
    "1401": RUN_ROOT / "Run_1401_20260823_151126_stage7_modern_structured_context",
    "1804": RUN_ROOT / "Run_1804_20260905_081349_dense_pairwise_field_adaptation",
    "2000": RUN_ROOT / "Run_2000_20260915_225542_routed_module_hubs",
    "2100": RUN_ROOT / "Run_2100_20260916_002148_routed_mean_shift",
}
DEFAULT_BASELINE_TABLE = PROJECT / (
    "Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/"
    "Stage3_Run1401_1804_1801_1802_Epoch500_90Case/tables/per_case_metrics.csv"
)
DEFAULT_ENDPOINTS = {
    "2000": PROJECT / (
        "diagnostics/generated/interface_operator_study/dynamic_sparse_routing/"
        "run2000/endpoint500/endpoint.json"
    ),
    "2100": PROJECT / (
        "diagnostics/generated/interface_operator_study/dynamic_sparse_routing/"
        "run2100/endpoint500/endpoint.json"
    ),
}
DEFAULT_LEDGERS = {
    "2000": PROJECT / (
        "diagnostics/generated/interface_operator_study/dynamic_sparse_routing/"
        "run2000/routing/anchors8192.json"
    ),
    "2100": PROJECT / (
        "diagnostics/generated/interface_operator_study/dynamic_sparse_routing/"
        "run2100/routing/anchors8192.json"
    ),
}
DEFAULT_MEAN_SHIFT = PROJECT / (
    "diagnostics/generated/interface_operator_study/dynamic_sparse_routing/"
    "run2100/routing/mean_shift_epoch500.json"
)
STRATA = (
    "module_count_stratum",
    "spacing_stratum",
    "wall_proximity_stratum",
    "heating_heterogeneity_stratum",
)
CHANNELS = ("u", "v", "p", "omega", "temperature")
PHASES = (
    ("p0_port", "p0_port_module", "p0_port_environment"),
    ("p1_refinement", "p1_refinement_module", "p1_refinement_environment"),
    ("p2_field", "p2_field_module", "p2_field_environment"),
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in materialized:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    result = number(value)
    return int(result) if result is not None else None


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.fmean(values)) if values else None


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.median(values)) if values else None


def finite_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    return [value for row in rows if (value := number(row.get(key))) is not None]


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def slope(rows: list[dict[str, Any]], key: str) -> float | None:
    points = [
        (epoch, value)
        for row in rows
        if (epoch := integer(row.get("epoch"))) is not None
        and (value := number(row.get(key))) is not None
    ]
    if len(points) < 2:
        return None
    x_bar = statistics.fmean(point[0] for point in points)
    y_bar = statistics.fmean(point[1] for point in points)
    denominator = sum((x - x_bar) ** 2 for x, _ in points)
    return float(sum((x - x_bar) * (y - y_bar) for x, y in points) / denominator) if denominator else 0.0


def close(a: Any, b: Any, *, rtol: float = 1e-7, atol: float = 1e-7) -> bool:
    left, right = number(a), number(b)
    if left is None or right is None:
        return a == b
    return math.isclose(left, right, rel_tol=rtol, abs_tol=atol)


def endpoint_table(path: Path) -> Path:
    """Accept either an endpoint JSON or its already-reduced CSV."""
    if path.suffix.lower() != ".json":
        return path
    data = read_json(path)
    table = data.get("evaluation", {}).get("per_case_metrics")
    if not table:
        raise ValueError(f"Endpoint JSON has no evaluation.per_case_metrics: {path}")
    return Path(table)


def checkpoint_epoch(path: Path) -> int | None:
    name = path.name
    marker = "epoch_"
    if marker not in name:
        return None
    suffix = name.split(marker, 1)[1].split("_", 1)[0]
    return int(suffix) if suffix.isdigit() else None


def model_identity(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    core = model.get("core_honf", {}) if isinstance(model, dict) else {}
    interface = core.get("interface_model", {}) if isinstance(core, dict) else {}
    routing = interface.get("routing", {}) if isinstance(interface, dict) else {}
    strategy = routing.get("strategy")
    if strategy is None:
        strategy = core.get("organizer_mode") or core.get("decoder_mode")
    architecture = core.get("forward_architecture") or model.get("forward_architecture")
    if architecture is None:
        architecture = "legacy_honf" if core.get("organizer_mode") else "legacy_or_dense"
    if architecture == "legacy_honf" and strategy is None:
        strategy = core.get("organizer_mode")
    if architecture == "dense_pairwise_field" and strategy is None:
        strategy = "dense_pairwise"
    return {
        "forward_architecture": architecture,
        "routing_strategy": strategy,
        "hidden_dim": core.get("hidden_dim"),
        "interface_hidden_dim": interface.get("message_hidden_dim"),
        "mean_shift_steps": routing.get("mean_shift_steps"),
        "mean_shift_feature_bandwidth": routing.get("mean_shift_feature_bandwidth"),
        "routing_execution": routing.get("execution"),
    }


def training_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    training = config.get("training", {})
    return {
        "seed": training.get("seed"),
        "requested_epochs": training.get("epochs"),
        "batch_size": dataset.get("batch_size"),
        "val_batch_size": dataset.get("val_batch_size"),
        "learning_rate": training.get("learning_rate"),
        "weight_decay": training.get("weight_decay"),
        "gradient_clip_norm": training.get("gradient_clip_norm"),
        "amp": training.get("amp"),
        "train_case_count": dataset.get("train_case_count"),
        "val_case_count": dataset.get("val_case_count"),
        "query_batch_size": config.get("evaluation", {}).get("query_batch_size"),
    }


def select_metric_rows(run_dir: Path, max_epoch: int) -> tuple[list[dict[str, str]], Path]:
    candidates = (run_dir / "metrics/metrics.csv", run_dir / "metrics.csv")
    metrics = next((path for path in candidates if path.exists()), None)
    if metrics is None:
        raise FileNotFoundError(f"No metrics CSV under {run_dir}")
    rows = [
        row
        for row in read_csv(metrics)
        if (epoch := integer(row.get("epoch"))) is not None and epoch <= max_epoch
    ]
    if not rows:
        raise ValueError(f"No metric rows through epoch {max_epoch}: {metrics}")
    rows.sort(key=lambda row: integer(row.get("epoch")) or 0)
    return rows, metrics


def convergence_summary(run: str, run_dir: Path, max_epoch: int) -> dict[str, Any]:
    rows, metrics_path = select_metric_rows(run_dir, max_epoch)
    validation_key = "val_field_mse"
    finite = [(integer(row["epoch"]), number(row.get(validation_key))) for row in rows]
    finite = [(epoch, value) for epoch, value in finite if epoch is not None and value is not None]
    if not finite:
        raise ValueError(f"No finite {validation_key} values: {metrics_path}")
    best_epoch, best_value = min(finite, key=lambda pair: pair[1])
    last_epoch = max(epoch for epoch, _ in finite)
    result: dict[str, Any] = {
        "run": run,
        "metrics": str(metrics_path.resolve()),
        "metric_row_count_through_budget": len(rows),
        "metric_last_epoch_through_budget": last_epoch,
        "best_val_field_mse_through_budget": best_value,
        "best_val_field_mse_epoch_through_budget": best_epoch,
        "exact500_val_field_mse": next((value for epoch, value in finite if epoch == max_epoch), None),
    }
    for width in (25, 50, 100):
        start = max_epoch - width + 1
        window = [row for row in rows if (epoch := integer(row.get("epoch"))) is not None and start <= epoch <= max_epoch]
        values = finite_values(window, validation_key)
        result[f"late{width}_count"] = len(values)
        result[f"late{width}_median_val_field_mse"] = median(values)
        result[f"late{width}_min_val_field_mse"] = min(values) if values else None
        result[f"late{width}_max_val_field_mse"] = max(values) if values else None
        result[f"late{width}_last_val_field_mse"] = values[-1] if values else None
        result[f"late{width}_slope_per_epoch"] = slope(window, validation_key)
    previous_start = max_epoch - 2 * 50 + 1
    previous_end = max_epoch - 50
    previous_window = [
        row
        for row in rows
        if (epoch := integer(row.get("epoch"))) is not None
        and previous_start <= epoch <= previous_end
    ]
    previous_values = finite_values(previous_window, validation_key)
    result["previous50_count"] = len(previous_values)
    result["previous50_median_val_field_mse"] = median(previous_values)
    result["last50_minus_previous50_median_val_field_mse"] = (
        result["late50_median_val_field_mse"] - result["previous50_median_val_field_mse"]
        if result["late50_median_val_field_mse"] is not None
        and result["previous50_median_val_field_mse"] is not None
        else None
    )
    # Epoch rows may have validation temperature and total metrics as well;
    # retaining their endpoint values helps review convergence without mixing
    # them into the field-selection policy.
    endpoint = next((row for row in rows if integer(row.get("epoch")) == max_epoch), {})
    for key in ("val_loss_total", "val_temperature_mse", "val_predicted_field_mse", "val_predicted_temperature_mse"):
        result[f"exact500_{key}"] = number(endpoint.get(key))
    return result


def run_record(run: str, run_dir: Path, max_epoch: int) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config_resolved.json"
    manifest = read_json(manifest_path)
    summary = read_json(summary_path)
    config = read_json(config_path)
    convergence = convergence_summary(run, run_dir, max_epoch)
    identity = model_identity(config)
    train = training_config(config)
    splits = manifest.get("launch_resources", {}).get("dataset splits", {})
    train["train_case_count"] = train.get("train_case_count") or splits.get("train")
    train["val_case_count"] = train.get("val_case_count") or splits.get("test") or splits.get("val")
    exact = run_dir / f"epoch_{max_epoch:04d}_model.pt"
    saved_best = run_dir / "best_by_field_mse_model.pt"
    actual_saved_best_epoch = None
    endpoint_json = DEFAULT_ENDPOINTS.get(run)
    if endpoint_json and endpoint_json.exists():
        endpoint = read_json(endpoint_json)
        for policy in endpoint.get("policies", []):
            if policy.get("policy") == "saved_best":
                actual_saved_best_epoch = integer(policy.get("actual_epoch"))
                break
    if actual_saved_best_epoch is None and run in ("2000", "2100") and saved_best.exists():
        # Do not deserialize a checkpoint in this reducer.  The endpoint
        # manifest is the authoritative saved-best epoch when available.
        actual_saved_best_epoch = None
    wall = {
        key: summary.get(key)
        for key in (
            "actual_train_wall_seconds",
            "actual_validation_wall_seconds",
            "actual_total_epoch_wall_seconds",
            "peak_cuda_memory_mb",
        )
    }
    checkpoint_sizes = {
        "exact500_bytes": exact.stat().st_size if exact.exists() else None,
        "saved_best_bytes": saved_best.stat().st_size if saved_best.exists() else None,
    }
    checkpoint_names = sorted(path.name for path in run_dir.glob("epoch_*_model.pt"))
    return {
        "run": run,
        "run_dir": str(run_dir.resolve()),
        "run_manifest": str(manifest_path.resolve()),
        "summary": str(summary_path.resolve()),
        "config": str(config_path.resolve()),
        "status": manifest.get("status"),
        "last_completed_epoch": manifest.get("last_completed_epoch"),
        "requested_epochs": train.get("requested_epochs"),
        "actual_run_epochs": summary.get("epochs"),
        "display_name": manifest.get("display_name"),
        "created_at": manifest.get("created_at"),
        "started_at": manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        **identity,
        **train,
        **wall,
        **checkpoint_sizes,
        "available_epoch_checkpoints": checkpoint_names,
        "exact500_checkpoint": str(exact.resolve()) if exact.exists() else None,
        "exact500_checkpoint_exists": exact.exists(),
        "selected_policy": "best_validation_field_mse_through500",
        "selected_checkpoint": str(saved_best.resolve()) if saved_best.exists() and run in ("2000", "2100") else None,
        "selected_checkpoint_exists": bool(saved_best.exists() and run in ("2000", "2100")),
        "selected_checkpoint_epoch": actual_saved_best_epoch if run in ("2000", "2100") else None,
        "selected_weights_available_within_budget": bool(saved_best.exists() and run in ("2000", "2100") and actual_saved_best_epoch and actual_saved_best_epoch <= max_epoch),
        "parent_selected_weights_missing_within_budget": run in ("1401", "1804"),
        "convergence": convergence,
        "model_config": summary.get("model_config", {}),
    }


def endpoint_policies(run: str, rows: list[dict[str, str]], endpoint_path: Path | None = None) -> list[dict[str, Any]]:
    """Normalize endpoint rows into exact/saved policy identities."""
    if run in ("2000", "2100"):
        labels = {
            f"exact500:run{run}": ("exact500", 500),
            f"saved_best:run{run}": ("saved_best", None),
        }
        result = []
        for label, (policy, epoch) in labels.items():
            selected = [row for row in rows if row.get("model_label") == label]
            if len(selected) != 90:
                raise ValueError(f"Expected 90 rows for {label} in {endpoint_path or 'endpoint table'}; got {len(selected)}")
            actual = checkpoint_epoch(Path(selected[0].get("checkpoint", "")))
            result.append({"run": run, "policy": policy, "label": label, "epoch": actual if actual is not None else epoch, "rows": selected, "source": str(endpoint_path.resolve()) if endpoint_path else None})
        return result
    raise ValueError(f"Endpoint policy rows are only expected for routed runs: {run}")


def baseline_policies(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    result = []
    for run in ("1401", "1804"):
        selected = [
            row
            for row in rows
            if f"Run_{run}_" in row.get("checkpoint", "")
            and "epoch_0500_model.pt" in row.get("checkpoint", "")
        ]
        if len(selected) != 90 or len({row.get("case_id") for row in selected}) != 90:
            raise ValueError(f"Expected 90 unique baseline rows for Run {run} in {path}; got {len(selected)}")
        result.append({
            "run": run,
            "policy": "exact500",
            "label": f"exact500:run{run}",
            "epoch": 500,
            "rows": selected,
            "source": str(path.resolve()),
        })
    return result


def endpoint_base_keys(row: dict[str, str], suffix: str) -> list[str]:
    bases = []
    for key in row:
        if not key.endswith("_sse"):
            continue
        base = key[:-4]
        if suffix == "_sse" and not base.endswith("_norm"):
            continue
        if suffix == "_physical_sse" and not base.endswith("_physical"):
            continue
        if f"{base}_target_sse" in row and f"{base}_num_values" in row:
            bases.append(base)
    return sorted(set(bases))


def pooled(rows: list[dict[str, str]], base: str) -> dict[str, Any] | None:
    triples = []
    for row in rows:
        values = [number(row.get(base + suffix)) for suffix in ("_sse", "_target_sse", "_num_values")]
        if any(value is None for value in values):
            return None
        triples.append(values)
    sse = sum(row[0] for row in triples)
    target = sum(row[1] for row in triples)
    count = sum(row[2] for row in triples)
    if target <= 0 or count <= 0:
        return None
    return {
        "sse": float(sse),
        "target_sse": float(target),
        "num_values": round(count),
        "mse": float(sse / count),
        "rmse": float(math.sqrt(sse / count)),
        "relative_l2": float(math.sqrt(sse / target)),
    }


def case_distribution(rows: list[dict[str, str]], base: str, value_suffix: str = "_l2") -> dict[str, Any]:
    values = [(row.get("case_id"), number(row.get(base + value_suffix))) for row in rows]
    values = [(case, value) for case, value in values if value is not None]
    ordered = sorted(values, key=lambda pair: pair[1])
    return {
        "equal_case_mean": mean(value for _, value in values),
        "equal_case_median": median(value for _, value in values),
        "equal_case_p95": quantile([value for _, value in values], 0.95),
        "equal_case_min": ordered[0][1] if ordered else None,
        "equal_case_max": ordered[-1][1] if ordered else None,
        "worst_case_id": ordered[-1][0] if ordered else None,
        "finite_case_count": len(values),
    }


def validate_populations(policies: list[dict[str, Any]]) -> dict[str, Any]:
    if not policies:
        raise ValueError("No endpoint policies")
    reference = {row.get("case_id") for row in policies[0]["rows"]}
    checks: dict[str, Any] = {
        "policy_count": len(policies),
        "case_count_each": {},
        "all_case_sets_match": True,
        "target_spaces": sorted({row.get("target_space") for policy in policies for row in policy["rows"]}),
        "target_value_checks": 0,
        "target_value_mismatches": [],
        "strata_checks": 0,
        "strata_mismatches": [],
    }
    if len(reference) != 90:
        raise ValueError(f"Endpoint population must contain 90 cases, got {len(reference)}")
    if checks["target_spaces"] != ["dataset_normalized"]:
        raise ValueError(f"Unexpected endpoint target spaces: {checks['target_spaces']}")
    for policy in policies:
        ids = [row.get("case_id") for row in policy["rows"]]
        checks["case_count_each"][policy["label"]] = len(ids)
        if len(ids) != 90 or len(set(ids)) != 90 or set(ids) != reference:
            checks["all_case_sets_match"] = False
    if not checks["all_case_sets_match"]:
        raise ValueError(f"Endpoint policy populations do not match: {checks['case_count_each']}")
    baseline = policies[0]["rows"]
    baseline_by_case = {row["case_id"]: row for row in baseline}
    target_keys = sorted(
        key
        for key in baseline[0]
        if key.endswith(("_target_sse", "_num_values"))
    )
    for policy in policies[1:]:
        for row in policy["rows"]:
            ref = baseline_by_case[row["case_id"]]
            for key in target_keys:
                checks["target_value_checks"] += 1
                if not close(row.get(key), ref.get(key)):
                    checks["target_value_mismatches"].append({"policy": policy["label"], "case_id": row["case_id"], "key": key, "actual": row.get(key), "expected": ref.get(key)})
            for key in STRATA:
                checks["strata_checks"] += 1
                if row.get(key) != ref.get(key):
                    checks["strata_mismatches"].append({"policy": policy["label"], "case_id": row["case_id"], "key": key, "actual": row.get(key), "expected": ref.get(key)})
    if checks["target_value_mismatches"] or checks["strata_mismatches"]:
        raise ValueError(f"Endpoint target/strata mismatch: {checks}")
    return checks


def endpoint_outputs(policies: list[dict[str, Any]], records: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    normalized = endpoint_base_keys(policies[0]["rows"][0], "_sse")
    physical = endpoint_base_keys(policies[0]["rows"][0], "_physical_sse")
    kpi_bases = sorted(
        key[: -len("_error")]
        for key in policies[0]["rows"][0]
        if key.endswith("_physical_error")
    )
    global_base = "global_field_fluid_norm"
    channel_bases = [f"field_{channel}_fluid_norm" for channel in CHANNELS]
    headline: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    kpi_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for policy in policies:
        rows = policy["rows"]
        run = policy["run"]
        record = records[run]
        checkpoint_epoch_value = policy.get("epoch")
        if checkpoint_epoch_value is None and policy["policy"] == "saved_best":
            checkpoint_epoch_value = record.get("selected_checkpoint_epoch")
        metric = pooled(rows, global_base)
        if metric is None:
            raise ValueError(f"Missing pooled global field metric: {policy['label']}")
        distribution = case_distribution(rows, global_base)
        headline.append({
            "run": run,
            "architecture": record.get("forward_architecture"),
            "routing_strategy": record.get("routing_strategy"),
            "policy": policy["policy"],
            "label": policy["label"],
            "checkpoint_epoch": checkpoint_epoch_value,
            "checkpoint": rows[0].get("checkpoint"),
            **metric,
            **distribution,
        })
        for base in normalized:
            metric = pooled(rows, base)
            if metric is None:
                continue
            channel_rows.append({
                "run": run,
                "architecture": record.get("forward_architecture"),
                "routing_strategy": record.get("routing_strategy"),
                "policy": policy["policy"],
                "label": policy["label"],
                "checkpoint_epoch": checkpoint_epoch_value,
                "metric_base": base,
                "metric_family": "dataset_normalized",
                **metric,
                **case_distribution(rows, base),
            })
        for base in physical:
            metric = pooled(rows, base)
            if metric is None:
                continue
            physical_rows.append({
                "run": run,
                "architecture": record.get("forward_architecture"),
                "routing_strategy": record.get("routing_strategy"),
                "policy": policy["policy"],
                "label": policy["label"],
                "checkpoint_epoch": checkpoint_epoch_value,
                "metric_base": base,
                "metric_family": "physical_native_units",
                **metric,
                **case_distribution(rows, base, "_relative_l2"),
            })
        for base in kpi_bases:
            errors = finite_values(rows, base + "_error")
            absolute_errors = finite_values(rows, base + "_abs_error")
            relative_errors = finite_values(rows, base + "_relative_error")
            if not errors and not absolute_errors and not relative_errors:
                continue
            item = {
                "run": run,
                "architecture": record.get("forward_architecture"),
                "routing_strategy": record.get("routing_strategy"),
                "policy": policy["policy"],
                "label": policy["label"],
                "checkpoint_epoch": checkpoint_epoch_value,
                "metric_base": base,
                "metric_family": "physical_scalar_kpi",
                "error_mean": mean(errors),
                "error_median": median(errors),
                "error_p95": quantile(errors, 0.95),
                "absolute_error_mean": mean(absolute_errors),
                "absolute_error_median": median(absolute_errors),
                "absolute_error_p95": quantile(absolute_errors, 0.95),
                "relative_error_mean": mean(relative_errors),
                "relative_error_median": median(relative_errors),
                "relative_error_p95": quantile(relative_errors, 0.95),
                "finite_error_count": len(errors),
                "finite_absolute_error_count": len(absolute_errors),
                "finite_relative_error_count": len(relative_errors),
            }
            kpi_rows.append(item)
        for stratum in STRATA:
            values = sorted({row.get(stratum, "") for row in rows})
            for value in values:
                subset = [row for row in rows if row.get(stratum, "") == value]
                item: dict[str, Any] = {
                    "run": run,
                    "architecture": record.get("forward_architecture"),
                    "routing_strategy": record.get("routing_strategy"),
                    "policy": policy["policy"],
                    "label": policy["label"],
                    "checkpoint_epoch": checkpoint_epoch_value,
                    "stratum": stratum,
                    "stratum_value": value,
                    "case_count": len(subset),
                }
                for base in [global_base, *channel_bases]:
                    metric = pooled(subset, base)
                    if metric:
                        item[f"{base}_pooled_relative_l2"] = metric["relative_l2"]
                        item[f"{base}_pooled_mse"] = metric["mse"]
                        item[f"{base}_num_values"] = metric["num_values"]
                strata_rows.append(item)
        worst = sorted(rows, key=lambda row: number(row.get(global_base + "_l2")) or -math.inf, reverse=True)[:10]
        for rank, row in enumerate(worst, 1):
            tail_rows.append({
                "run": run,
                "architecture": record.get("forward_architecture"),
                "routing_strategy": record.get("routing_strategy"),
                "policy": policy["policy"],
                "label": policy["label"],
                "checkpoint_epoch": checkpoint_epoch_value,
                "rank_worst": rank,
                "case_id": row.get("case_id"),
                "global_field_fluid_norm_l2": number(row.get(global_base + "_l2")),
                "active_module_count": number(row.get("active_module_count")),
                **{key: row.get(key) for key in STRATA},
            })
    write_csv(output_dir / "endpoint_headline.csv", headline)
    write_csv(output_dir / "endpoint_channel.csv", channel_rows)
    write_csv(output_dir / "endpoint_physical.csv", physical_rows)
    write_csv(output_dir / "endpoint_kpi.csv", kpi_rows)
    write_csv(output_dir / "endpoint_strata.csv", strata_rows)
    write_csv(output_dir / "endpoint_tail.csv", tail_rows)
    return {
        "normalized_metric_bases": normalized,
        "physical_metric_bases": physical,
        "physical_scalar_kpi_bases": kpi_bases,
        "headline": headline,
        "channel_row_count": len(channel_rows),
        "physical_row_count": len(physical_rows),
        "kpi_row_count": len(kpi_rows),
        "strata_row_count": len(strata_rows),
        "tail_row_count": len(tail_rows),
    }


def paired_outputs(policies: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    by_label = {policy["label"]: policy for policy in policies}
    baseline_labels = ["exact500:run1401", "exact500:run1804"]
    candidate_labels = [
        label
        for label in by_label
        if label.startswith(("exact500:run2000", "saved_best:run2000", "exact500:run2100", "saved_best:run2100"))
    ]
    summaries: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    base = "global_field_fluid_norm_l2"
    for candidate_label in sorted(candidate_labels):
        candidate = by_label[candidate_label]
        candidate_by_case = {row["case_id"]: row for row in candidate["rows"]}
        for baseline_label in baseline_labels:
            baseline = by_label[baseline_label]
            baseline_by_case = {row["case_id"]: row for row in baseline["rows"]}
            deltas = []
            for case_id in sorted(candidate_by_case):
                candidate_value = number(candidate_by_case[case_id].get(base))
                baseline_value = number(baseline_by_case[case_id].get(base))
                if candidate_value is None or baseline_value is None:
                    continue
                delta = candidate_value - baseline_value
                deltas.append(delta)
                case_rows.append({
                    "candidate": candidate_label,
                    "baseline": baseline_label,
                    "case_id": case_id,
                    "candidate_l2": candidate_value,
                    "baseline_l2": baseline_value,
                    "candidate_minus_baseline_l2": delta,
                    "candidate_wins": delta < 0,
                })
            summaries.append({
                "candidate": candidate_label,
                "baseline": baseline_label,
                "metric": base,
                "case_count": len(deltas),
                "wins": sum(delta < 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "losses": sum(delta > 0 for delta in deltas),
                "mean_delta_candidate_minus_baseline": mean(deltas),
                "median_delta_candidate_minus_baseline": median(deltas),
                "p95_delta_candidate_minus_baseline": quantile(deltas, 0.95),
            })
    write_csv(output_dir / "endpoint_paired.csv", summaries)
    write_csv(output_dir / "endpoint_paired_cases.csv", case_rows)
    return {"summary": summaries, "case_row_count": len(case_rows)}


def aggregate(values: list[float]) -> dict[str, Any]:
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "count": len(values),
    }


def ledger_metric(row: dict[str, Any], prefix: str, name: str) -> float | None:
    return number(row.get(f"{prefix}_{name}"))


def ledger_outputs(ledger_paths: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    aggregate_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    statuses: dict[str, Any] = {}
    names = (
        "raw_path_count",
        "unique_pair_count",
        "duplicate_expansion",
        "query_hub_support_mean",
        "dense_active_module_pair_reference",
        "dense_environment_pair_reference",
    )
    for run, path in ledger_paths.items():
        if not path.exists():
            statuses[run] = {"status": "missing", "path": str(path.resolve())}
            continue
        data = read_json(path)
        rows = data.get("rows", [])
        if data.get("status") != "ok" or len(rows) != 5 or len({row.get("case_id") for row in rows}) != 5:
            raise ValueError(f"Expected five successful anchor rows in {path}")
        statuses[run] = {"status": data.get("status"), "path": str(path.resolve()), "case_ids": data.get("case_ids", [])}
        for phase, module_prefix, environment_prefix in PHASES:
            phase_values: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                common = {
                    "run": run,
                    "phase": phase,
                    "case_id": row.get("case_id"),
                    "M_active": row.get("M_active"),
                    "Mpack": row.get("Mpack"),
                    "candidate_hub_count": row.get("candidate_hub_count"),
                    "candidate_padded_width": row.get("candidate_padded_width"),
                    "source_incidence_nnz": row.get("source_incidence_nnz"),
                    "raw_two_hop_path_count": row.get("raw_two_hop_path_count"),
                    "unique_receiver_source_pair_count": row.get("unique_receiver_source_pair_count"),
                }
                item = dict(common)
                for role, prefix in (("module", module_prefix), ("environment", environment_prefix)):
                    for name in names:
                        value = ledger_metric(row, prefix, name)
                        item[f"{role}_{name}"] = value
                        if value is not None:
                            phase_values[f"{role}_{name}"].append(value)
                    dense_name = "dense_active_module_pair_reference" if role == "module" else "dense_environment_pair_reference"
                    unique = item[f"{role}_unique_pair_count"]
                    dense = item[f"{role}_{dense_name}"]
                    item[f"{role}_unique_over_dense"] = unique / dense if unique is not None and dense and dense > 0 else None
                    if item[f"{role}_unique_over_dense"] is not None:
                        phase_values[f"{role}_unique_over_dense"].append(item[f"{role}_unique_over_dense"])
                anchor_rows.append(item)
            aggregate_row = {"run": run, "phase": phase, "anchor_count": len(rows)}
            for key, values in phase_values.items():
                aggregate_values = aggregate(values)
                for suffix, value in aggregate_values.items():
                    aggregate_row[f"{key}_{suffix}"] = value
            # The common source columns are repeated per case and should be
            # visible in phase summaries even though they are not phase-prefixed.
            for key in ("M_active", "Mpack", "candidate_hub_count", "candidate_padded_width", "source_incidence_nnz", "raw_two_hop_path_count", "unique_receiver_source_pair_count"):
                values = [value for row in rows if (value := number(row.get(key))) is not None]
                for suffix, value in aggregate(values).items():
                    aggregate_row[f"{key}_{suffix}"] = value
            aggregate_rows.append(aggregate_row)
    write_csv(output_dir / "routing_support_anchors.csv", anchor_rows)
    write_csv(output_dir / "routing_support.csv", aggregate_rows)
    return {"statuses": statuses, "aggregate": aggregate_rows, "anchor_row_count": len(anchor_rows)}


def nested_summary(value: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def space_summary(candidate: dict[str, Any], space: str) -> dict[str, Any]:
    data = candidate.get(space, {})
    final_drift = nested_summary(data, ("final_drift", "summary")) or {}
    final_separation = nested_summary(data, ("minimum_separation", "final", "summary")) or {}
    movement = [
        nested_summary(item, ("summary", "mean"))
        for item in data.get("iteration_movement", [])
        if nested_summary(item, ("summary", "mean")) is not None
    ]
    modes = nested_summary(data, ("approximate_mode_count", "counts_per_iteration_per_batch"))
    mode_counts = []
    if isinstance(modes, list):
        for iteration in modes:
            if isinstance(iteration, list):
                mode_counts.append(iteration[0] if iteration else None)
            else:
                mode_counts.append(iteration)
    # Descriptor concentration is emitted once alongside the space-specific
    # coordinate diagnostics.  It describes the final mean-shift descriptors,
    # so expose it for each space without pretending it is a second geometry.
    concentration = candidate.get("descriptor_concentration", {})
    final_concentration = concentration.get("final", {}) if isinstance(concentration, dict) else {}
    return {
        "final_drift_mean": final_drift.get("mean"),
        "final_drift_median": final_drift.get("median"),
        "final_drift_max": final_drift.get("max"),
        "final_minimum_separation": final_separation.get("mean"),
        "iteration_movement_mean_json": json.dumps(movement, separators=(",", ":")),
        "approximate_mode_counts_per_iteration_json": json.dumps(mode_counts, separators=(",", ":")),
        "approximate_mode_tolerance": nested_summary(data, ("approximate_mode_count", "tolerance")),
        "approximate_mode_space": nested_summary(data, ("approximate_mode_count", "space")),
        "descriptor_norm_final_mean": nested_summary(final_concentration, ("descriptor_norm", "mean")),
        "descriptor_distance_to_centroid_final_mean": nested_summary(final_concentration, ("distance_to_centroid_l2", "mean")),
    }


def intervention_outputs(path: Path, output_dir: Path) -> dict[str, Any]:
    if not path.exists():
        write_csv(output_dir / "routing_candidate_intervention.csv", [])
        return {"status": "missing", "path": str(path.resolve()), "row_count": 0}
    data = read_json(path)
    output: list[dict[str, Any]] = []
    for result in data.get("results", []):
        case_id = result.get("case_id")
        p2 = next((item for item in result.get("preparations", []) if item.get("phase") == "p2_field"), None)
        if not p2:
            continue
        candidate = p2.get("candidates", {})
        intervention = result.get("candidate_intervention_comparison", {})
        errors = intervention.get("query_batch_ground_truth", result.get("query_batch_ground_truth", {}))
        mean_shift_errors = errors.get("mean_shift", {})
        module_hubs_errors = errors.get("module_hubs", {})
        delta_errors = errors.get("module_hubs_minus_mean_shift", {})
        supports = intervention.get("fine_support_statistics", {})
        timing = nested_summary(p2, ("candidate_generation", "module_hubs_reference", "candidate_generation_timing")) or {}
        valid_counts = candidate.get("valid_candidate_count", [])
        output_row: dict[str, Any] = {
            "run": "2100",
            "policy": "exact500",
            "case_id": case_id,
            "source": str(path.resolve()),
            "candidate_shape": json.dumps(candidate.get("shape"), separators=(",", ":")),
            "candidate_padded_width": candidate.get("candidate_count"),
            "valid_candidate_count": valid_counts[0] if valid_counts else candidate.get("valid_candidate_count"),
            "valid_candidate_mask": json.dumps(candidate.get("valid_candidate_mask"), separators=(",", ":")),
            "iteration_count": candidate.get("iteration_count"),
            "length_scale_ell": json.dumps(candidate.get("length_scale_ell"), separators=(",", ":")),
            "candidate_generation_seconds_mean": timing.get("seconds_mean"),
            "candidate_generation_seconds_total": timing.get("seconds_total"),
            "candidate_generation_warmup": timing.get("warmup"),
            "candidate_generation_repeats": timing.get("repeats"),
            "candidate_generation_timed_scope": data.get("candidate_timing", {}).get("timed_scope"),
            "same_model_weights": intervention.get("same_model_weights"),
            "trained_accuracy_comparison": intervention.get("trained_accuracy_comparison"),
            "mean_shift_query_relative_l2": mean_shift_errors.get("relative_l2"),
            "mean_shift_query_sse": mean_shift_errors.get("sse"),
            "mean_shift_query_target_sse": mean_shift_errors.get("target_sse"),
            "module_hubs_query_relative_l2": module_hubs_errors.get("relative_l2"),
            "module_hubs_query_sse": module_hubs_errors.get("sse"),
            "module_hubs_query_target_sse": module_hubs_errors.get("target_sse"),
            "module_hubs_minus_mean_shift_relative_l2": delta_errors.get("relative_l2"),
            "module_hubs_minus_mean_shift_sse": delta_errors.get("sse"),
        }
        for space in ("physical", "scaled", "joint"):
            for key, value in space_summary(candidate, space).items():
                output_row[f"{space}_{key}"] = value
        for strategy in ("mean_shift", "module_hubs"):
            for phase in ("p2_field_module", "p2_field_environment"):
                support = supports.get(strategy, {}).get(phase, {})
                for key in ("raw_path_count", "unique_pair_count", "duplicate_expansion"):
                    output_row[f"{strategy}_{phase}_{key}"] = support.get(key)
                query_support = support.get("query_hub_support", {})
                output_row[f"{strategy}_{phase}_query_hub_support_mean"] = query_support.get("mean") if isinstance(query_support, dict) else None
        output.append(output_row)
    write_csv(output_dir / "routing_candidate_intervention.csv", output)
    return {
        "status": data.get("status"),
        "path": str(path.resolve()),
        "row_count": len(output),
        "case_ids": [row.get("case_id") for row in output],
        "candidate_timing": data.get("candidate_timing"),
    }


def selection_rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in ("1401", "1804", "2000", "2100"):
        record = records[run]
        convergence = record["convergence"]
        rows.append({
            "run": run,
            "display_name": record.get("display_name"),
            "forward_architecture": record.get("forward_architecture"),
            "routing_strategy": record.get("routing_strategy"),
            "requested_epochs": record.get("requested_epochs"),
            "actual_run_epochs": record.get("actual_run_epochs"),
            "last_completed_epoch": record.get("last_completed_epoch"),
            "best_val_field_mse_through500": convergence.get("best_val_field_mse_through_budget"),
            "best_val_field_mse_epoch_through500": convergence.get("best_val_field_mse_epoch_through_budget"),
            "exact500_val_field_mse": convergence.get("exact500_val_field_mse"),
            "late25_median_val_field_mse": convergence.get("late25_median_val_field_mse"),
            "late50_median_val_field_mse": convergence.get("late50_median_val_field_mse"),
            "late100_median_val_field_mse": convergence.get("late100_median_val_field_mse"),
            "previous50_median_val_field_mse": convergence.get("previous50_median_val_field_mse"),
            "last50_minus_previous50_median_val_field_mse": convergence.get(
                "last50_minus_previous50_median_val_field_mse"
            ),
            "late50_slope_per_epoch": convergence.get("late50_slope_per_epoch"),
            "late100_slope_per_epoch": convergence.get("late100_slope_per_epoch"),
            "selected_policy": record.get("selected_policy"),
            "selected_checkpoint_epoch": record.get("selected_checkpoint_epoch"),
            "selected_weights_available_within_budget": record.get("selected_weights_available_within_budget"),
            "parent_selected_weights_missing_within_budget": record.get("parent_selected_weights_missing_within_budget"),
            "seed": record.get("seed"),
            "batch_size": record.get("batch_size"),
            "val_batch_size": record.get("val_batch_size"),
            "learning_rate": record.get("learning_rate"),
            "weight_decay": record.get("weight_decay"),
            "gradient_clip_norm": record.get("gradient_clip_norm"),
            "amp": record.get("amp"),
            "train_case_count": record.get("train_case_count"),
            "val_case_count": record.get("val_case_count"),
            "query_batch_size": record.get("query_batch_size"),
            "actual_train_wall_seconds": record.get("actual_train_wall_seconds"),
            "actual_validation_wall_seconds": record.get("actual_validation_wall_seconds"),
            "actual_total_epoch_wall_seconds": record.get("actual_total_epoch_wall_seconds"),
            "peak_cuda_memory_mb": record.get("peak_cuda_memory_mb"),
            "exact500_checkpoint_bytes": record.get("exact500_bytes"),
            "selected_checkpoint_bytes": record.get("saved_best_bytes"),
        })
    return rows


def verify_parameter_inventory(
    run_dirs: dict[str, Path],
    output_dir: Path,
    *,
    max_epoch: int,
) -> dict[str, Any]:
    """Strictly reconstruct and count each exact checkpoint on CPU."""

    import torch
    from channelthermal.evaluation.loading import load_model

    rows: list[dict[str, Any]] = []
    for run in ("1401", "1804", "2000", "2100"):
        checkpoint = run_dirs[run] / f"epoch_{max_epoch:04d}_model.pt"
        model, payload = load_model(checkpoint, torch.device("cpu"))
        actual_epoch = integer(payload.get("epoch", payload.get("current_epoch")))
        if actual_epoch != max_epoch:
            raise ValueError(f"Run {run} checkpoint reports epoch {actual_epoch}, expected {max_epoch}")
        named = list(model.named_parameters())
        total = sum(int(parameter.numel()) for parameter in model.parameters())
        trainable = sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        core = sum(
            int(parameter.numel())
            for name, parameter in named
            if name.startswith("core.")
        )
        local = sum(
            int(parameter.numel())
            for name, parameter in named
            if name.startswith("local_coupling.")
        )
        fallback = sum(
            int(parameter.numel())
            for name, parameter in named
            if name.startswith("fallback_heads.")
        )
        core_config = payload.get("model_config", {}).get("core_honf", {})
        interface_config = core_config.get("interface_model", {})
        routing_config = interface_config.get("routing", {})
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "run": run,
                "checkpoint": str(checkpoint.resolve()),
                "epoch": actual_epoch,
                "forward_architecture": core_config.get("forward_architecture", "legacy_honf"),
                "routing_strategy": routing_config.get("strategy"),
                "parameters_total": total,
                "parameters_trainable": trainable,
                "parameters_core": core,
                "parameters_local_frozen": local,
                "parameters_fallback_heads": fallback,
                "parameter_tensor_count": len(named),
                "state_dict_key_count": len(model.state_dict()),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": digest.hexdigest(),
            }
        )
        del model, payload
    result = {
        "schema_version": 1,
        "status": "ok",
        "scope": "CPU strict reconstruction of each completed exact epoch-500 checkpoint",
        "device": "cpu",
        "loader": "channelthermal.evaluation.loading.load_model",
        "count_definition": {
            "parameters_total": "sum(model.parameters())",
            "parameters_trainable": "sum(parameters with requires_grad=True)",
            "parameters_core": "sum(named parameters whose name starts with core.)",
            "parameters_local_frozen": "sum(named parameters whose name starts with local_coupling.)",
            "parameters_fallback_heads": "sum(named parameters whose name starts with fallback_heads.)",
        },
        "rows": rows,
    }
    write_json(output_dir / "parameter_inventory_epoch500.json", result)
    write_csv(output_dir / "parameter_inventory_epoch500.csv", rows)
    return result


def rolling_median(values: list[float], width: int = 50) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        window = values[max(0, index - width + 1) : index + 1]
        result.append(float(statistics.median(window)) if len(window) >= width else None)
    return result


def convergence_plot(records: dict[str, dict[str, Any]], output_dir: Path) -> str | None:
    """Render the logged 500-epoch trend and available runtime traces."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    colors = {"1401": "#4c78a8", "1804": "#f58518", "2000": "#54a24b", "2100": "#e45756"}
    labels = {"1401": "1401 legacy", "1804": "1804 dense", "2000": "2000 module hubs", "2100": "2100 mean shift"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axis_field, axis_temperature, axis_time, axis_memory = axes.ravel()
    plotted_time = False
    plotted_memory = False
    for run in ("1401", "1804", "2000", "2100"):
        record = records[run]
        rows = read_csv(Path(record["convergence"]["metrics"]))
        rows = [row for row in rows if (epoch := integer(row.get("epoch"))) is not None and epoch <= 500]
        epochs = [integer(row["epoch"]) for row in rows]
        field = [number(row.get("val_field_mse")) for row in rows]
        temperature = [number(row.get("val_temperature_mse")) for row in rows]
        color = colors[run]
        label = labels[run]
        field_points = [(epoch, value) for epoch, value in zip(epochs, field) if epoch is not None and value is not None]
        temperature_points = [(epoch, value) for epoch, value in zip(epochs, temperature) if epoch is not None and value is not None]
        if field_points:
            x, y = zip(*field_points)
            axis_field.plot(x, y, color=color, alpha=0.27, linewidth=0.8)
            medians = rolling_median([value for _, value in field_points])
            axis_field.plot(x, medians, color=color, linewidth=1.8, label=label)
        if temperature_points:
            x, y = zip(*temperature_points)
            axis_temperature.plot(x, y, color=color, alpha=0.27, linewidth=0.8)
            medians = rolling_median([value for _, value in temperature_points])
            axis_temperature.plot(x, medians, color=color, linewidth=1.8, label=label)
        train_points = [(epoch, value) for epoch, row in zip(epochs, rows) if epoch is not None and (value := number(row.get("train_wall_seconds"))) is not None]
        validation_points = [(epoch, value) for epoch, row in zip(epochs, rows) if epoch is not None and (value := number(row.get("val_wall_seconds"))) is not None]
        memory_points = [(epoch, value) for epoch, row in zip(epochs, rows) if epoch is not None and (value := number(row.get("peak_cuda_memory_mb"))) is not None]
        if train_points:
            axis_time.plot(*zip(*train_points), color=color, linewidth=1.0, label=f"{label} train")
            plotted_time = True
        if validation_points:
            axis_time.plot(*zip(*validation_points), color=color, linestyle="--", linewidth=0.9, label=f"{label} val")
            plotted_time = True
        if memory_points:
            axis_memory.plot(*zip(*memory_points), color=color, linewidth=1.0, label=label)
            plotted_memory = True
    axis_field.set_title("Validation field MSE (log scale; raw + trailing-50 median)")
    axis_temperature.set_title("Validation temperature MSE (log scale; raw + trailing-50 median)")
    axis_time.set_title("Logged per-epoch latency")
    axis_memory.set_title("Logged peak CUDA memory")
    axis_field.set_ylabel("MSE")
    axis_temperature.set_ylabel("MSE")
    axis_time.set_ylabel("seconds")
    axis_memory.set_ylabel("MiB")
    for axis in axes.ravel():
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.2)
        axis.set_xlim(1, 500)
    axis_field.set_yscale("log")
    axis_temperature.set_yscale("log")
    axis_field.legend(fontsize=8, loc="upper right")
    axis_temperature.legend(fontsize=8, loc="upper right")
    if plotted_time:
        axis_time.legend(fontsize=7, ncol=2, loc="upper right")
    else:
        axis_time.text(0.5, 0.5, "No per-epoch latency logged", ha="center", va="center", transform=axis_time.transAxes)
    if plotted_memory:
        axis_memory.legend(fontsize=8, loc="upper right")
    else:
        axis_memory.text(0.5, 0.5, "No peak-memory trace logged", ha="center", va="center", transform=axis_memory.transAxes)
    figure.suptitle("HONF routing runs: convergence and logged runtime traces", fontsize=14)
    path = output_dir / "convergence_500.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epoch", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "diagnostics/generated/interface_operator_study/routing_500_comparison")
    for run in ("1401", "1804", "2000", "2100"):
        parser.add_argument(f"--run{run}-dir", type=Path, default=DEFAULT_RUN_DIRS[run])
    parser.add_argument("--baseline-table", type=Path, default=DEFAULT_BASELINE_TABLE)
    parser.add_argument("--run2000-endpoint", type=Path, default=DEFAULT_ENDPOINTS["2000"])
    parser.add_argument("--run2100-endpoint", type=Path, default=DEFAULT_ENDPOINTS["2100"])
    parser.add_argument("--run2000-ledger", type=Path, default=DEFAULT_LEDGERS["2000"])
    parser.add_argument("--run2100-ledger", type=Path, default=DEFAULT_LEDGERS["2100"])
    parser.add_argument("--mean-shift-json", type=Path, default=DEFAULT_MEAN_SHIFT)
    parser.add_argument(
        "--verify-parameter-inventory",
        action="store_true",
        help="strictly reconstruct all exact checkpoints on CPU and write parameter inventory outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_epoch != 500:
        raise ValueError("This comparison is defined for --max-epoch 500")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {run: getattr(args, f"run{run}_dir").resolve() for run in ("1401", "1804", "2000", "2100")}
    records = {run: run_record(run, path, args.max_epoch) for run, path in run_dirs.items()}
    for run, record in records.items():
        if record.get("status") != "completed":
            raise ValueError(f"Run {run} is not completed: {record.get('status')}")
        if not record.get("exact500_checkpoint_exists"):
            raise FileNotFoundError(f"Run {run} has no exact epoch-500 checkpoint")

    endpoint_paths = {"2000": args.run2000_endpoint.resolve(), "2100": args.run2100_endpoint.resolve()}
    policies: list[dict[str, Any]] = []
    for run, path in endpoint_paths.items():
        table = endpoint_table(path)
        policies.extend(endpoint_policies(run, read_csv(table), table))
    policies.extend(baseline_policies(args.baseline_table.resolve()))
    # Keep the report order stable: parent exact endpoints first, then routed
    # exact/selected policies.  The exact-vs-selected distinction remains in
    # every output row.
    policies.sort(key=lambda policy: (int(policy["run"]), 0 if policy["policy"] == "exact500" else 1))
    validation = validate_populations(policies)
    records_for_outputs = records
    endpoint_info = endpoint_outputs(policies, records_for_outputs, output_dir)
    paired_info = paired_outputs(policies, output_dir)
    ledger_info = ledger_outputs({"2000": args.run2000_ledger.resolve(), "2100": args.run2100_ledger.resolve()}, output_dir)
    intervention_info = intervention_outputs(args.mean_shift_json.resolve(), output_dir)
    parameter_inventory_path = output_dir / "parameter_inventory_epoch500.json"
    parameter_inventory = (
        verify_parameter_inventory(run_dirs, output_dir, max_epoch=args.max_epoch)
        if args.verify_parameter_inventory
        else (read_json(parameter_inventory_path) if parameter_inventory_path.exists() else None)
    )
    selection = selection_rows(records)
    write_csv(output_dir / "selection_summary.csv", selection)
    plot_path = convergence_plot(records, output_dir)

    summary = {
        "schema_version": 1,
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": str(PROJECT),
        "budget": {
            "max_epoch": args.max_epoch,
            "selection_metric": "validation field MSE",
            "selection_direction": "minimum",
            "exact_policy": "epoch_0500_model.pt",
            "selected_policy": "best validation field MSE among metrics epochs <=500",
            "older_parent_selected_weights_available": False,
        },
        "inputs": {
            "run_dirs": {run: str(path) for run, path in run_dirs.items()},
            "baseline_table": str(args.baseline_table.resolve()),
            "endpoint_tables": {run: str(endpoint_table(path).resolve()) for run, path in endpoint_paths.items()},
            "endpoint_json": {run: str(path) for run, path in endpoint_paths.items()},
            "ledgers": {run: str(path.resolve()) for run, path in {"2000": args.run2000_ledger, "2100": args.run2100_ledger}.items()},
            "mean_shift_json": str(args.mean_shift_json.resolve()),
            "parameter_inventory": str(parameter_inventory_path.resolve()),
        },
        "runs": records,
        "selection_summary": selection,
        "population_validation": validation,
        "endpoint": endpoint_info,
        "paired": paired_info,
        "routing_support": ledger_info,
        "candidate_intervention": intervention_info,
        "parameter_inventory": parameter_inventory,
        "convergence_plot": plot_path,
        "definitions": {
            "normalized_metrics": "Endpoint *_norm_sse, *_target_sse, and *_num_values are checkpoint dataset-normalized errors; pooled relative L2 is sqrt(sum SSE / sum target SSE).",
            "physical_metrics": "Endpoint *_physical_sse metrics remain in the evaluator's native physical units and are summarized separately from normalized field metrics.",
            "equal_case_summary": "Per-case L2 means, medians, p95, and worst case are descriptive; pooled values use summed SSE and target SSE/count.",
            "support": "Ledger raw paths are positive two-hop paths before receiver-source deduplication; unique pairs are the actual fine pair keys after coalescing. Route weights are not physical influence.",
            "candidate_intervention": "Run2100 mean-shift and temporary module-hubs forwards use the same epoch-500 weights; this is a frozen candidate-generation intervention, not a separately trained accuracy comparison.",
            "parameter_inventory": "Counts come from strict CPU reconstruction with channelthermal.evaluation.loading.load_model; total includes the frozen local module, while trainable counts use requires_grad=True.",
        },
        "limitations": [
            "The 90-case test split is the established development holdout, not an untouched CFD test set.",
            "Run 1401 and Run 1804 validation-best checkpoint files from the <=500 budget are unavailable; their later mature best files are not substituted.",
            "All four exact endpoint policies use their epoch-500 checkpoints and therefore share the requested 500-epoch assessment budget. Parent selected-through-500 weights for Run 1401 and Run 1804 are unavailable; later mature parent checkpoints are not substituted.",
            "Wall-clock fields are absent from the Run 1401 summary and architecture/evaluation overhead differs, so no cross-run efficiency ranking is inferred from them.",
            "The mean-shift intervention is bounded to five anchor cases and 32 query points per case; its query error is not a 90-case endpoint result.",
            "The common ledger artifacts were produced without routing-map export because the existing CUDA map-to-NumPy path fails; support/count fields remain the measured ledger scope.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "policies": [policy["label"] for policy in policies],
        "endpoint_population": validation,
        "headline": endpoint_info["headline"],
        "paired": paired_info["summary"],
        "routing_support_rows": len(ledger_info["aggregate"]),
        "candidate_intervention_rows": intervention_info.get("row_count", 0),
    }, indent=2))


if __name__ == "__main__":
    main()
