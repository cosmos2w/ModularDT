"""Reduce a routed candidate and existing exact-500 tables without rerunning parents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_honf_maturity import number, pooled, read_csv, write_csv
from nstage2_reduction import (
    ANCHORS,
    CHANNEL_BASES,
    CORE_BASES,
    ENGINEERING_KPI_BASES,
    STRATA,
    _checkpoint_epoch_from_payload,
    _load_trusted_checkpoint,
    _parent_table_dir,
    _raw_metric_key,
)

PROJECT = Path(__file__).resolve().parents[2]


def validate_population(rows: list[dict], reference: list[dict] | None = None) -> None:
    identities = [row["case_id"] for row in rows]
    if len(identities) != 90 or len(set(identities)) != 90:
        raise ValueError("Each endpoint must contain exactly 90 unique development cases")
    required_metrics = [_raw_metric_key(base) for base in CORE_BASES + CHANNEL_BASES + ENGINEERING_KPI_BASES]
    for row in rows:
        for key in required_metrics:
            if number(row.get(key)) is None:
                raise ValueError(f"Required endpoint metric missing/nonfinite: {row['case_id']}/{key}")
        for key in STRATA:
            if not row.get(key):
                raise ValueError(f"Required stratum missing: {row['case_id']}/{key}")
    if reference is None:
        return
    by_case = {row["case_id"]: row for row in reference}
    if set(identities) != set(by_case):
        raise ValueError("Endpoint populations differ")
    # Denominators establish the same physical targets, masks, and units.
    for row in rows:
        other = by_case[row["case_id"]]
        for key in other:
            if not key.endswith(("_target_sse", "_num_values")):
                continue
            if key not in row:
                raise ValueError(f"Required target/count column missing: {key}")
            left, right = number(row[key]), number(other[key])
            if left is None and right is None:
                continue
            if left is None or right is None or not np.isclose(left, right, rtol=1e-7, atol=1e-7):
                raise ValueError(f"Target mismatch: case {row['case_id']} / {key}: {left} vs {right}")
        for key in STRATA:
            if row[key] != other[key]:
                raise ValueError(f"Stratum mismatch: {row['case_id']}/{key}")


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "worst": float(array.max()),
    }


def reduce_endpoint(endpoint_table: Path, output: Path, *, run_id: str = "2000") -> dict:
    candidate_identity = f"Run_{run_id}_"
    candidate_rows = read_csv(endpoint_table)
    groups = {}
    for row in candidate_rows:
        groups.setdefault(row["model_label"], []).append(row)
    if not groups or not any(label.startswith("exact500:") for label in groups):
        raise ValueError(f"A labelled exact500 Run {run_id} evaluation is required")
    sources = {label: str(endpoint_table.resolve()) for label in groups}
    epochs = {}
    for label, rows in groups.items():
        if not label.startswith(("exact500:", "saved_best:")):
            raise ValueError(f"Unknown candidate checkpoint policy: {label}")
        validate_population(rows)
        if any(candidate_identity not in row["checkpoint"] for row in rows):
            raise ValueError(f"Unexpected candidate identity in {label}")
        checkpoints = {row["checkpoint"] for row in rows}
        if len(checkpoints) != 1:
            raise ValueError(f"Mixed checkpoints in candidate label {label}")
        checkpoint = Path(next(iter(checkpoints)))
        epoch = _checkpoint_epoch_from_payload(_load_trusted_checkpoint(checkpoint))
        if epoch is None or not 1 <= epoch <= 500:
            raise ValueError(f"Candidate checkpoint outside the authorized epoch budget: {label}/{epoch}")
        if label.startswith("exact500:") and epoch != 500:
            raise ValueError(f"Exact endpoint has epoch {epoch}, expected 500")
        if label.startswith("saved_best:") and checkpoint.name != "best_by_field_mse_model.pt":
            raise ValueError("Selected candidate must use the designated validation-field checkpoint")
        epochs[label] = epoch
    for run, name in (("1401", "Legacy"), ("1804", "Dense"), ("1806", "Regional")):
        path = _parent_table_dir(PROJECT, run, 500) / "per_case_metrics.csv"
        rows = [
            row
            for row in read_csv(path)
            if f"Run_{run}_" in row["checkpoint"] and "epoch_0500_model.pt" in row["checkpoint"]
        ]
        validate_population(rows)
        label = f"exact500:{name}_{run}"
        groups[label] = rows
        sources[label] = str(path)
        epochs[label] = 500
    reference = groups["exact500:Legacy_1401"]
    for rows in groups.values():
        validate_population(rows, reference)

    pools, distributions, strata, cases, headline = [], [], [], [], []
    bases = CORE_BASES + CHANNEL_BASES
    for label, rows in groups.items():
        identity = {
            "label": label,
            "checkpoint": rows[0]["checkpoint"],
            "checkpoint_epoch": epochs[label],
            "source": sources[label],
        }
        for metric in bases:
            result = pooled(rows, metric)
            pools.append(
                {**identity, "metric": metric, "status": "available" if result else "unavailable", **(result or {})}
            )
        for metric in bases + ENGINEERING_KPI_BASES:
            key = _raw_metric_key(metric)
            values = [number(row.get(key)) for row in rows]
            finite = [value for value in values if value is not None]
            distributions.append(
                {
                    **identity,
                    "metric": metric,
                    "status": "available" if len(finite) == 90 else "incomplete",
                    **(distribution(finite) if finite else {}),
                }
            )
        for axis in STRATA:
            for stratum in sorted({row.get(axis, "") for row in rows}):
                subset = [row for row in rows if row.get(axis, "") == stratum]
                for metric in bases:
                    result = pooled(subset, metric)
                    if result:
                        strata.append(
                            {
                                **identity,
                                "axis": axis,
                                "stratum": stratum,
                                "cases": len(subset),
                                "metric": metric,
                                **result,
                            }
                        )
        ordered = sorted(rows, key=lambda row: float(row["global_field_fluid_norm_l2"]), reverse=True)
        selected = set(ANCHORS) | {row["case_id"] for row in ordered[:10]}
        for row in rows:
            if row["case_id"] in selected:
                cases.append({**identity, "anchor": row["case_id"] in ANCHORS, "worst_ten": row in ordered[:10], **row})
        result = pooled(rows, "global_field_fluid_norm")
        headline.append(
            {
                **identity,
                "pooled_fluid_l2": result["relative_l2"],
                **distribution([float(row["global_field_fluid_norm_l2"]) for row in rows]),
                "worst_case_id": ordered[0]["case_id"],
            }
        )

    paired, paired_summary = [], []
    candidate_labels = sorted(label for label in groups if candidate_identity in groups[label][0]["checkpoint"])
    for candidate in candidate_labels:
        for parent in ("exact500:Legacy_1401", "exact500:Dense_1804", "exact500:Regional_1806"):
            parent_cases = {row["case_id"]: row for row in groups[parent]}
            for metric in bases + ENGINEERING_KPI_BASES:
                key = _raw_metric_key(metric)
                comparisons = []
                for row in groups[candidate]:
                    left, right = number(row.get(key)), number(parent_cases[row["case_id"]].get(key))
                    if left is None or right is None:
                        continue
                    item = {
                        "candidate": candidate,
                        "parent": parent,
                        "metric": metric,
                        "case_id": row["case_id"],
                        "candidate_value": left,
                        "parent_value": right,
                        "delta": left - right,
                    }
                    paired.append(item)
                    comparisons.append(item)
                if comparisons:
                    # Signed engineering errors are not ranked as better/worse.
                    rankable = not (metric.endswith("_physical_error"))
                    paired_summary.append(
                        {
                            "candidate": candidate,
                            "parent": parent,
                            "metric": metric,
                            "cases": len(comparisons),
                            "mean_delta": float(np.mean([x["delta"] for x in comparisons])),
                            "wins": sum(x["delta"] < 0 for x in comparisons) if rankable else "",
                            "ties": sum(x["delta"] == 0 for x in comparisons) if rankable else "",
                            "interpretation": "selected candidate vs exact parent"
                            if candidate.startswith("saved_best:")
                            else "matched exact epoch",
                        }
                    )
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("headline", headline),
        ("pooled_metrics", pools),
        ("equal_case_distributions", distributions),
        ("strata", strata),
        ("difficult_cases", cases),
        ("paired_cases", paired),
        ("paired_summary", paired_summary),
    ):
        write_csv(output / f"{name}.csv", rows)
    summary = {
        "status": "complete",
        "population": 90,
        "sources": sources,
        "headline": headline,
        "limitations": [
            "Saved-best candidate and exact-500 parent comparisons are explicitly different policies.",
            "Matched parent saved-best-through-500 weights are unavailable; mature best files are not substituted.",
            "Sampled validation MSE and full-grid pooled fluid L2 have different definitions.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="2000")
    args = parser.parse_args()
    reduce_endpoint(args.endpoint_table, args.output, run_id=args.run_id)


if __name__ == "__main__":
    main()
