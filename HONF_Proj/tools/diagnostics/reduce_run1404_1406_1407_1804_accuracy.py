#!/usr/bin/env python3
"""Reduce the matched Run-1404/1406/1407/1804 accuracy tables.

The compare workflow already owns model execution and writes the complete raw
tables.  This CPU-only reducer validates the common 90-case population,
computes pooled and equal-case metrics, paired wins, checkpoint metadata, and
the exact-epoch-5000 sensitivity deltas.  It never loads a model onto CUDA or
changes the evaluator tables.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

RUNS = ("Run1404", "Run1406", "Run1407", "Run1804")
BEST_LABELS = tuple(f"{run}_best_field" for run in RUNS)
ENDPOINT_LABELS = tuple(f"{run}_epoch5000" for run in RUNS)
HEADLINE_METRICS = (
    "global_field_fluid_norm",
    "global_field_all_norm",
    "global_field_near_interface_norm",
    "global_field_far_fluid_norm",
    "field_u_fluid_norm",
    "field_v_fluid_norm",
    "field_p_fluid_norm",
    "field_omega_fluid_norm",
    "field_temperature_fluid_norm",
    "internal_temperature_physical",
    "interface_t_surface_physical",
    "interface_q_normal_physical",
    "port_t_env_final_physical",
    "port_h_effective_final_physical",
)
VALIDATION_KEYS = (
    "best_val_loss_total",
    "best_val_field_mse",
    "best_val_temperature_mse",
    "best_val_predicted_loss_total",
    "best_val_predicted_field_mse",
    "best_val_predicted_temperature_mse",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_values(rows: Iterable[Mapping[str, object]], key: str) -> np.ndarray:
    values = [number(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def metric_summary(rows: Sequence[Mapping[str, object]], base: str) -> dict[str, object] | None:
    """Return pooled L2 plus unweighted equal-case summaries for one base metric."""
    # Normalized field metrics use ``*_l2`` while physical endpoint metrics
    # use the evaluator's explicit ``*_relative_l2`` suffix.
    l2_suffix = "_l2" if any(f"{base}_l2" in row for row in rows) else "_relative_l2"
    values = _finite_values(rows, f"{base}{l2_suffix}")
    sse = _finite_values(rows, f"{base}_sse")
    target_sse = _finite_values(rows, f"{base}_target_sse")
    counts = _finite_values(rows, f"{base}_num_values")
    if values.size == 0:
        return None
    result: dict[str, object] = {
        "num_cases": int(values.size),
        "equal_case_mean": float(values.mean()),
        "equal_case_median": float(np.median(values)),
        "equal_case_p95": float(np.quantile(values, 0.95)),
        "equal_case_min": float(values.min()),
        "equal_case_max": float(values.max()),
    }
    if sse.size == len(rows) and target_sse.size == len(rows) and np.all(target_sse > 0):
        result.update(
            pooled_sse=float(sse.sum()),
            pooled_target_sse=float(target_sse.sum()),
            pooled_num_values=int(counts.sum()) if counts.size == len(rows) else None,
            pooled_relative_l2=float(np.sqrt(sse.sum() / target_sse.sum())),
        )
    else:
        result.update(
            pooled_sse=None,
            pooled_target_sse=None,
            pooled_num_values=None,
            pooled_relative_l2=None,
        )
    return result


def paired_summary(
    candidate_rows: Mapping[str, Mapping[str, object]],
    baseline_rows: Mapping[str, Mapping[str, object]],
    metric: str,
) -> dict[str, object]:
    """Compare one metric on the exact common case IDs; lower error wins."""
    cases = sorted(set(candidate_rows) & set(baseline_rows))
    deltas = np.asarray(
        [
            float(candidate_rows[case][f"{metric}_l2"])
            - float(baseline_rows[case][f"{metric}_l2"])
            for case in cases
        ],
        dtype=np.float64,
    )
    return {
        "num_cases": int(deltas.size),
        "wins": int((deltas < 0).sum()),
        "losses": int((deltas > 0).sum()),
        "ties": int((deltas == 0).sum()),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
    }


def _load_checkpoint_metadata(path: Path) -> dict[str, object]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("best_metrics", {})
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_bytes": path.stat().st_size,
        "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "model_family": checkpoint.get("model_family"),
        "workflow": checkpoint.get("workflow"),
        "stage": checkpoint.get("stage"),
        "best_metric": checkpoint.get("best_metric"),
        "validation": {key: number(metrics.get(key)) for key in VALIDATION_KEYS},
        "architecture": checkpoint.get("model_config", {}).get("core_honf", {}).get("forward_architecture"),
    }


def _training_metric_row(run_dir: Path, epoch: int) -> dict[str, object]:
    rows = read_csv(run_dir / "metrics.csv")
    matching = [row for row in rows if number(row.get("epoch")) == epoch]
    if not matching:
        return {"epoch": epoch}
    row = matching[-1]
    return {
        "epoch": epoch,
        **{key: number(row.get(key)) for key in ("val_loss_total", "val_field_mse", "val_temperature_mse", "val_predicted_loss_total", "val_predicted_field_mse", "val_predicted_temperature_mse")},
    }


def _run_dir_from_checkpoint(checkpoint: str) -> Path:
    path = Path(checkpoint)
    return path.parent


def reduce_accuracy(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    output = output.resolve()
    best_tables = root / "tables"
    endpoint_tables = root / "endpoint5000" / "tables"
    best_rows = read_csv(best_tables / "per_case_metrics.csv")
    endpoint_rows = read_csv(endpoint_tables / "per_case_metrics.csv")
    best_summary_rows = read_csv(best_tables / "model_summary_metrics.csv")
    endpoint_summary_rows = read_csv(endpoint_tables / "model_summary_metrics.csv")
    cost_rows = read_csv(best_tables / "evaluation_cost_summary_metrics.csv")

    def grouped(rows: Sequence[Mapping[str, str]], labels: Sequence[str]) -> dict[str, list[dict[str, str]]]:
        result = {label: [dict(row) for row in rows if row.get("model_label") == label] for label in labels}
        for label, group in result.items():
            if len(group) != 90 or len({row["case_id"] for row in group}) != 90:
                raise ValueError(f"{label}: expected 90 unique cases, got {len(group)}")
        return result

    best = grouped(best_rows, BEST_LABELS)
    endpoint = grouped(endpoint_rows, ENDPOINT_LABELS)
    all_case_ids = {row["case_id"] for row in best[BEST_LABELS[0]]}
    for label in BEST_LABELS + ENDPOINT_LABELS:
        rows = best[label] if label in best else endpoint[label]
        if {row["case_id"] for row in rows} != all_case_ids:
            raise ValueError(f"{label}: case population differs from the reference")

    def metric_rows(groups: Mapping[str, Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for label, rows in groups.items():
            for metric in HEADLINE_METRICS:
                summary = metric_summary(rows, metric)
                if summary is not None:
                    out.append({"model_label": label, "metric": metric, **summary})
        return out

    metrics = metric_rows(best)
    endpoint_metrics = metric_rows(endpoint)
    paired: list[dict[str, object]] = []
    for candidate, baseline in itertools.permutations(BEST_LABELS, 2):
        candidate_map = {row["case_id"]: row for row in best[candidate]}
        baseline_map = {row["case_id"]: row for row in best[baseline]}
        for metric in ("global_field_fluid_norm", "global_field_all_norm"):
            paired.append({"candidate": candidate, "baseline": baseline, "metric": metric, **paired_summary(candidate_map, baseline_map, metric)})

    checkpoints: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in best_summary_rows:
        label = row["model_label"]
        if label not in BEST_LABELS or label in seen:
            continue
        seen.add(label)
        metadata = _load_checkpoint_metadata(Path(row["checkpoint"]))
        metadata.update({"model_label": label, "run_dir": row["run_dir"], "table_num_cases": int(row["num_cases"])})
        checkpoints.append(metadata)
    if set(seen) != set(BEST_LABELS):
        raise ValueError(f"Missing best checkpoint metadata: {set(BEST_LABELS) - seen}")

    endpoint_checkpoints: list[dict[str, object]] = []
    seen_endpoint: set[str] = set()
    for row in endpoint_summary_rows:
        label = row["model_label"]
        if label not in ENDPOINT_LABELS or label in seen_endpoint:
            continue
        seen_endpoint.add(label)
        checkpoint = Path(row["checkpoint"])
        metadata = _load_checkpoint_metadata(checkpoint)
        metadata.update({"model_label": label, "run_dir": row["run_dir"], "table_num_cases": int(row["num_cases"]), "training_metrics_at_epoch": _training_metric_row(_run_dir_from_checkpoint(row["checkpoint"]), 5000)})
        endpoint_checkpoints.append(metadata)

    endpoint_delta: list[dict[str, object]] = []
    best_metric_map = {(row["model_label"], row["metric"]): row for row in metrics}
    endpoint_metric_map = {(row["model_label"], row["metric"]): row for row in endpoint_metrics}
    for run in RUNS:
        best_label, endpoint_label = f"{run}_best_field", f"{run}_epoch5000"
        for metric in HEADLINE_METRICS:
            before = best_metric_map[(best_label, metric)]
            after = endpoint_metric_map[(endpoint_label, metric)]
            endpoint_delta.append({
                "run": run,
                "metric": metric,
                "best_pooled_relative_l2": before["pooled_relative_l2"],
                "epoch5000_pooled_relative_l2": after["pooled_relative_l2"],
                "delta_epoch5000_minus_best": (float(after["pooled_relative_l2"]) - float(before["pooled_relative_l2"])) if before["pooled_relative_l2"] is not None and after["pooled_relative_l2"] is not None else None,
                "best_equal_case_mean": before["equal_case_mean"],
                "epoch5000_equal_case_mean": after["equal_case_mean"],
                "delta_equal_case_mean": float(after["equal_case_mean"]) - float(before["equal_case_mean"]),
            })

    worst: list[dict[str, object]] = []
    for label, rows in best.items():
        for rank, row in enumerate(sorted(rows, key=lambda row: float(row["global_field_fluid_norm_l2"]), reverse=True)[:10], 1):
            worst.append({"model_label": label, "rank": rank, "case_id": row["case_id"], "global_field_fluid_norm_l2": float(row["global_field_fluid_norm_l2"]), "global_field_all_norm_l2": float(row["global_field_all_norm_l2"]), "module_count_stratum": row.get("module_count_stratum"), "spacing_stratum": row.get("spacing_stratum"), "wall_proximity_stratum": row.get("wall_proximity_stratum"), "heating_heterogeneity_stratum": row.get("heating_heterogeneity_stratum")})

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metric_summary.csv", metrics)
    write_csv(output / "endpoint5000_metric_summary.csv", endpoint_metrics)
    write_csv(output / "paired_case_summary.csv", paired)
    write_csv(output / "endpoint5000_deltas.csv", endpoint_delta)
    write_csv(output / "worst_cases.csv", worst)
    write_csv(output / "cost_summary.csv", cost_rows)
    write_csv(output / "checkpoint_validation.csv", checkpoints + endpoint_checkpoints)

    summary: dict[str, object] = {
        "schema_version": 1,
        "created_by": "reduce_run1404_1406_1407_1804_accuracy.py",
        "protocol": {
            "population": "90 identical test cases",
            "split": "test",
            "query_count": 8192,
            "query_batch_size": 32768,
            "local_port_condition_mode": "predicted",
            "device": "cuda:0 (physical GPU 0 via CUDA_VISIBLE_DEVICES=0)",
            "selection_policy": "best_by_field_mse_model.pt, selected by logged validation field MSE through epoch 5000",
            "sensitivity": "epoch_5000_model.pt",
        },
        "definitions": {
            "pooled_relative_l2": "sqrt(sum case SSE / sum case target SSE)",
            "equal_case_mean": "arithmetic mean of per-case relative L2; every case has equal weight",
            "paired_delta": "candidate minus baseline per-case relative L2; lower error is better",
        },
        "checkpoints": checkpoints,
        "endpoint5000_checkpoints": endpoint_checkpoints,
        "outputs": {
            "raw_root": str(root),
            "metric_summary": str(output / "metric_summary.csv"),
            "endpoint5000_metric_summary": str(output / "endpoint5000_metric_summary.csv"),
            "paired_case_summary": str(output / "paired_case_summary.csv"),
            "endpoint5000_deltas": str(output / "endpoint5000_deltas.csv"),
            "worst_cases": str(output / "worst_cases.csv"),
            "cost_summary": str(output / "cost_summary.csv"),
            "checkpoint_validation": str(output / "checkpoint_validation.csv"),
        },
        "metrics": metrics,
        "endpoint5000_metrics": endpoint_metrics,
        "paired_case_summary": paired,
        "endpoint5000_deltas": endpoint_delta,
    }
    (output / "accuracy_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    def fmt(value: object) -> str:
        return "n/a" if value is None else f"{float(value):.6f}"

    lines = [
        "# Matched Run-1404/1406/1407/1804 accuracy reduction",
        "",
        "Policy: `best_by_field_mse_model.pt` for each run; exact `epoch_5000_model.pt` is sensitivity-only.",
        "Protocol: 90 identical test cases, Q=8192, predicted ports, query batch 32768.",
        "",
        "## Best validation-field-MSE checkpoints",
        "",
        "| model | checkpoint epoch | pooled fluid relL2 | equal-case fluid mean | pooled all-domain relL2 | equal-case all-domain mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    checkpoint_epoch = {row["model_label"]: row["epoch"] for row in checkpoints}
    for label in BEST_LABELS:
        fluid = best_metric_map[(label, "global_field_fluid_norm")]
        all_domain = best_metric_map[(label, "global_field_all_norm")]
        lines.append(f"| {label} | {checkpoint_epoch[label]} | {fmt(fluid['pooled_relative_l2'])} | {fmt(fluid['equal_case_mean'])} | {fmt(all_domain['pooled_relative_l2'])} | {fmt(all_domain['equal_case_mean'])} |")
    lines += [
        "",
        "## Logged validation selection metrics",
        "",
        "| model | checkpoint epoch | best val loss total | best val field MSE | best val temperature MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    checkpoint_by_label = {row["model_label"]: row for row in checkpoints}
    for label in BEST_LABELS:
        validation = checkpoint_by_label[label]["validation"]
        lines.append(f"| {label} | {checkpoint_by_label[label]['epoch']} | {fmt(validation.get('best_val_loss_total'))} | {fmt(validation.get('best_val_field_mse'))} | {fmt(validation.get('best_val_temperature_mse'))} |")
    lines += [
        "",
        "## Component metrics (best checkpoints)",
        "",
        "The CSV/JSON contain equal-case median, p95, min/max, pooled SSE, and pooled target SSE for every listed metric.",
        "",
        "| model | near-interface | far-fluid | u | v | p | vorticity | temperature | internal T | surface T | q-normal | final env T | final h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_key = {(row["model_label"], row["metric"]): row for row in metrics}
    display_metrics = ("global_field_near_interface_norm", "global_field_far_fluid_norm", "field_u_fluid_norm", "field_v_fluid_norm", "field_p_fluid_norm", "field_omega_fluid_norm", "field_temperature_fluid_norm", "internal_temperature_physical", "interface_t_surface_physical", "interface_q_normal_physical", "port_t_env_final_physical", "port_h_effective_final_physical")
    for label in BEST_LABELS:
        values = [fmt(by_key[(label, metric)]["pooled_relative_l2"]) for metric in display_metrics]
        lines.append("| " + " | ".join([label, *values]) + " |")
    lines += [
        "",
        "## Notes",
        "",
        "- Raw evaluator tables remain under the output root; this reduction does not replace them.",
        "- Generic hypergraph tables are preserved as emitted by the established evaluator. Group-control organization is not inferred from legacy-organizer fields; use the dedicated organization evidence separately.",
        "- Validation checkpoint metrics are the checkpoint's recorded `best_metrics`; exact endpoint validation rows are recorded separately and may contain historical NaNs for Run1804.",
        "",
    ]
    (output / "accuracy_summary.md").write_text("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="matched evaluator output root")
    parser.add_argument("--output", type=Path, required=True, help="summary output directory")
    args = parser.parse_args()
    summary = reduce_accuracy(args.root, args.output)
    print(json.dumps({"models": len(summary["checkpoints"]), "best_metrics": len(summary["metrics"]), "endpoint_metrics": len(summary["endpoint5000_metrics"])}, sort_keys=True))


if __name__ == "__main__":
    main()
