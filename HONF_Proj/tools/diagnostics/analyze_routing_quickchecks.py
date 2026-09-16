#!/usr/bin/env python3
"""Reduce short original-versus-optimized routing histories.
Reads metric CSVs, reports first-10/first-50 coverage and trailing evidence,
and writes JSON/CSV/log-y plots without assigning a scientific score.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT / "diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/optimization/quickchecks"
CAPTURE_EPOCHS = (1, 2, 5, 10, 20, 50)
CORE_METRICS = ("val_field_mse", "val_temperature_mse", "val_loss_total")
RUNS = (
    ("run2000", "module_hubs", "original"),
    ("run2001", "module_hubs", "optimized"),
    ("run2100", "mean_shift", "original"),
    ("run2101", "mean_shift", "optimized"),
)
FIELDS = {
    "val_field_mse": ("val_field_mse", "validation_field_mse"),
    "val_temperature_mse": ("val_temperature_mse", "validation_temperature_mse"),
    "val_loss_total": ("val_loss_total", "val_loss", "validation_loss_total"),
    "train_wall_seconds": ("train_wall_seconds", "train_elapsed_seconds"),
    "val_wall_seconds": ("val_wall_seconds", "val_elapsed_seconds"),
    "epoch_wall_seconds": ("epoch_wall_seconds", "total_wall_seconds"),
    "peak_allocated_mb": ("peak_cuda_memory_mb", "peak_allocated_memory_mb", "peak_allocated_mib"),
    "peak_reserved_mb": ("peak_cuda_memory_reserved_mb", "peak_reserved_memory_mb", "peak_reserved_mib", "peak_cuda_reserved_mb"),
    "preclip_gradient_norm": ("preclip_gradient_norm", "gradient_norm", "grad_norm"),
    "parameter_update_norm": ("parameter_update_norm", "update_norm"),
}


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def locate_metrics(raw: str) -> tuple[Path, Path]:
    supplied = Path(raw).expanduser().resolve()
    if supplied.is_file():
        return supplied, supplied.parent.parent if supplied.parent.name == "metrics" else supplied.parent
    for candidate in (supplied / "metrics/metrics.csv", supplied / "metrics.csv"):
        if candidate.is_file():
            return candidate, supplied
    raise FileNotFoundError(f"No metrics CSV under {supplied}")


def phase_record(run: str, strategy: str, variant: str, rows: list[dict[str, Any]], end: int) -> dict[str, Any]:
    phase = [row for row in rows if row["epoch"] <= end]
    epochs = [row["epoch"] for row in phase]
    unique = set(epochs)
    def values(key: str) -> list[float]:
        return [value for row in phase if (value := number(row.get(key))) is not None]
    def med(key: str) -> float | None:
        data = values(key)
        return float(statistics.median(data)) if data else None
    def maximum(key: str) -> float | None:
        data = values(key)
        return max(data) if data else None
    missing = [item for item in range(1, end + 1) if item not in unique]
    return {
        "row_type": "phase", "run": run, "strategy": strategy, "variant": variant,
        "phase": f"first{end}", "requested_end_epoch": end,
        "available_epoch_count": len(unique), "first_available_epoch": min(unique) if unique else None,
        "last_available_epoch": max(unique) if unique else None,
        "duplicate_epoch_rows": len(epochs) - len(unique), "missing_epoch_count": len(missing),
        "missing_epochs": missing, "complete_through_requested_end": not missing,
        "train_wall_seconds_median": med("train_wall_seconds"),
        "val_wall_seconds_median": med("val_wall_seconds"),
        "epoch_wall_seconds_median": med("epoch_wall_seconds"),
        "peak_allocated_mb_median": med("peak_allocated_mb"), "peak_allocated_mb_max": maximum("peak_allocated_mb"),
        "peak_reserved_mb_median": med("peak_reserved_mb"), "peak_reserved_mb_max": maximum("peak_reserved_mb"),
    }


def summarize(run: str, strategy: str, variant: str, supplied: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics_path, run_dir = locate_metrics(supplied)
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        raw = list(reader)
    fields = list(reader.fieldnames or [])
    selected = {name: next((item for item in candidates if item in fields), None) for name, candidates in FIELDS.items()}
    def epoch(row: dict[str, str]) -> int | None:
        parsed = number(row.get("epoch"))
        return int(parsed) if parsed is not None else None
    def metric(row: dict[str, str], name: str) -> float | None:
        field = selected[name]
        return number(row.get(field)) if field else None
    def plain_evidence(rows: list[dict[str, str]], name: str) -> dict[str, Any]:
        field = selected[name]
        if field is None:
            return {"column": None, "rows_with_value": 0, "finite_rows": 0, "all_available_finite": None}
        observed = sum(bool(str(row.get(field, "")).strip()) for row in rows)
        finite = sum(number(row.get(field)) is not None for row in rows)
        return {"column": field, "rows_with_value": observed, "finite_rows": finite,
                "all_available_finite": bool(observed) and finite == observed}
    def capture_evidence(rows: list[dict[str, str]], name: str) -> dict[str, Any]:
        expected = [item for item in CAPTURE_EPOCHS if item <= 50]
        scheduled = [row for row in rows if epoch(row) in expected]
        recorded = sorted({epoch(row) for row in scheduled})
        result = plain_evidence(scheduled, name)
        result.update({
            "expected_scheduled_epochs": expected, "scheduled_rows": len(scheduled),
            "scheduled_epochs_recorded": recorded,
            "unrecorded_scheduled_epochs": [item for item in expected if item not in recorded],
            "unrecorded_scheduled_rows": len(expected) - len(recorded),
            "unscheduled_rows": len(rows) - len(scheduled),
            "missing_scheduled_value_rows": len(scheduled) - result["rows_with_value"],
            "nonfinite_scheduled_rows": result["rows_with_value"] - result["finite_rows"],
            "all_scheduled_available_finite": len(recorded) == len(expected) and result["finite_rows"] == len(expected),
        })
        return result
    raw50 = [row for row in raw if (item := epoch(row)) is not None and 1 <= item <= 50]
    records: list[dict[str, Any]] = []
    for row in raw50:
        current = epoch(row)
        if current is None:
            continue
        train, validation = metric(row, "train_wall_seconds"), metric(row, "val_wall_seconds")
        total = metric(row, "epoch_wall_seconds")
        if total is None and train is not None and validation is not None:
            total = train + validation
        records.append({
            "row_type": "epoch", "run": run, "strategy": strategy, "variant": variant, "epoch": current,
            "val_field_mse": metric(row, "val_field_mse"), "val_temperature_mse": metric(row, "val_temperature_mse"),
            "val_loss_total": metric(row, "val_loss_total"), "train_wall_seconds": train, "val_wall_seconds": validation,
            "epoch_wall_seconds": total, "peak_allocated_mb": metric(row, "peak_allocated_mb"),
            "peak_reserved_mb": metric(row, "peak_reserved_mb"), "preclip_gradient_norm": metric(row, "preclip_gradient_norm"),
            "parameter_update_norm": metric(row, "parameter_update_norm"),
        })
    records.sort(key=lambda item: item["epoch"])
    trailing = records[-10:]
    def med(key: str) -> float | None:
        data = [value for row in trailing if (value := number(row.get(key))) is not None]
        return float(statistics.median(data)) if data else None
    details: dict[str, Any] = {
        "run": run, "strategy": strategy, "variant": variant,
        "supplied_path": str(Path(supplied).expanduser().resolve()), "run_dir": str(run_dir),
        "metrics_path": str(metrics_path), "metric_rows_read": len(raw), "epoch_rows_through50": len(records),
        "available_epochs_through50": sorted({row["epoch"] for row in records}), "selected_columns": selected,
        "trailing10_requested": 10, "trailing10_rows": len(trailing),
        "trailing10_first_epoch": trailing[0]["epoch"] if trailing else None,
        "trailing10_last_epoch": trailing[-1]["epoch"] if trailing else None,
        "trailing10_median_val_field_mse": med("val_field_mse"),
        "trailing10_median_val_temperature_mse": med("val_temperature_mse"),
        "trailing10_median_val_loss_total": med("val_loss_total"),
        "core_metric_evidence_through50": {name: plain_evidence(raw50, name) for name in CORE_METRICS},
        "gradient_evidence_through50": capture_evidence(raw50, "preclip_gradient_norm"),
        "update_evidence_through50": capture_evidence(raw50, "parameter_update_norm"),
        "epoch_wall_seconds_definition": "recorded epoch_wall_seconds, or train_wall_seconds + val_wall_seconds; excludes I/O/checkpoint time",
    }
    manifest = run_dir / "run_manifest.json"
    if manifest.is_file():
        try:
            details["manifest_status"] = json.loads(manifest.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            details["manifest_status"] = None
    return details, [phase_record(run, strategy, variant, records, end) for end in (10, 50)], records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_history(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = (("val_field_mse", "Validation field MSE"), ("val_temperature_mse", "Validation temperature MSE"),
              ("val_loss_total", "Validation total loss"), ("epoch_wall_seconds", "Measured train + val seconds"),
              ("peak_allocated_mb", "Peak allocated MiB"), ("preclip_gradient_norm", "Recorded preclip gradient norm"))
    colors = {"module_hubs": "tab:blue", "mean_shift": "tab:orange"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for axis, (key, title) in zip(axes.flat, panels, strict=True):
        plotted = False
        for strategy, color in colors.items():
            for variant, linestyle in (("original", "-"), ("optimized", "--")):
                points = [(row["epoch"], value) for row in rows
                          if row["strategy"] == strategy and row["variant"] == variant
                          if (value := number(row.get(key))) is not None and value > 0]
                if points:
                    plotted = True
                    axis.plot(*zip(*points, strict=True), color=color, linestyle=linestyle,
                               label=f"{strategy} {variant}")
        axis.set(title=title, xlabel="epoch", yscale="log")
        axis.grid(True, which="both", alpha=0.25)
        if not plotted:
            axis.text(0.5, 0.5, "No finite positive values", ha="center", va="center", transform=axis.transAxes)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Routing quick checks: original versus optimized")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for run, _, _ in RUNS:
        parser.add_argument(f"--{run}", required=True, help="run directory or metrics CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", default="routing_quickchecks")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    details: dict[str, Any] = {}
    phases: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    for run, strategy, variant in RUNS:
        run_details, run_phases, run_epochs = summarize(run, strategy, variant, getattr(args, run))
        details[run], phases, epochs = run_details, phases + run_phases, epochs + run_epochs
    paths = {suffix: str(output / f"{args.prefix}.{suffix}") for suffix in ("json", "csv", "png")}
    payload = {
        "schema_version": 1, "task": "routing_quickcheck_reduction", "status": "complete",
        "inputs": {run: getattr(args, run) for run, _, _ in RUNS}, "outputs": paths,
        "runs": details, "phase_rows": phases, "epoch_rows": epochs,
        "limitations": [
            "Measured train + val seconds exclude I/O/checkpoint time when no total epoch field is logged.",
            "This reports available history; first50 can be incomplete for a 10-epoch quick check.",
            "Gradient/update counts reflect scheduled logged rows; unscheduled CSV placeholders remain unrecorded.",
            "Reserved memory is retained only when present in the input CSV; allocated memory is not a substitute.",
            "No scientific pass/fail threshold or convergence claim is assigned.",
        ],
    }
    json_path, csv_path, plot_path = (output / f"{args.prefix}.{suffix}" for suffix in ("json", "csv", "png"))
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, phases + epochs)
    plot_history(plot_path, epochs)
    for path in (json_path, csv_path, plot_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
