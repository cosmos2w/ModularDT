#!/usr/bin/env python3
"""Compare best finite training metrics at an equal epoch budget."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BestMetric:
    value: float
    epoch: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Reference loss-history CSV")
    parser.add_argument("candidate", type=Path, help="Candidate loss-history CSV")
    parser.add_argument(
        "--metric",
        action="append",
        required=True,
        help="Lower-is-better metric column; repeat to compare several columns",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Ignore rows after this epoch in both histories",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.0,
        help="Allowed candidate increase relative to abs(baseline); default: 0",
    )
    args = parser.parse_args()
    if args.max_epoch is not None and args.max_epoch < 1:
        parser.error("--max-epoch must be positive")
    if args.relative_tolerance < 0:
        parser.error("--relative-tolerance must be non-negative")
    return args


def best_metric(path: Path, metric: str, max_epoch: int | None) -> BestMetric:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if "epoch" not in fields:
            raise ValueError(f"{path}: missing required 'epoch' column")
        if metric not in fields:
            raise ValueError(f"{path}: missing metric column {metric!r}")

        best: BestMetric | None = None
        for row_number, row in enumerate(reader, start=2):
            try:
                epoch = int(float(row["epoch"]))
                value = float(row[metric])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_number}: invalid epoch or {metric}") from exc
            if max_epoch is not None and epoch > max_epoch:
                continue
            if not math.isfinite(value):
                continue
            if best is None or value < best.value:
                best = BestMetric(value=value, epoch=epoch)

    if best is None:
        budget = f" through epoch {max_epoch}" if max_epoch is not None else ""
        raise ValueError(f"{path}: no finite {metric!r} values{budget}")
    return best


def relative_increase(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate <= 0.0 else math.inf
    return (candidate - baseline) / abs(baseline)


def main() -> int:
    args = parse_args()
    failures = 0
    writer = csv.writer(__import__("sys").stdout, lineterminator="\n")
    writer.writerow(
        [
            "metric",
            "baseline_best",
            "baseline_epoch",
            "candidate_best",
            "candidate_epoch",
            "relative_increase",
            "status",
        ]
    )
    for metric in args.metric:
        baseline = best_metric(args.baseline, metric, args.max_epoch)
        candidate = best_metric(args.candidate, metric, args.max_epoch)
        increase = relative_increase(baseline.value, candidate.value)
        passed = increase <= args.relative_tolerance
        failures += not passed
        writer.writerow(
            [
                metric,
                f"{baseline.value:.17g}",
                baseline.epoch,
                f"{candidate.value:.17g}",
                candidate.epoch,
                f"{increase:.9g}",
                "PASS" if passed else "FAIL",
            ]
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
