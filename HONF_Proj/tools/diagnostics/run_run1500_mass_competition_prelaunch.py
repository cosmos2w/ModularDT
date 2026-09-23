"""Apply the fixed Run-1500 mass competition to frozen Run-1409 evidence.

The input table must be the 90-case population table produced from the latest
occupancy-adaptive Run-1409 epoch-50 checkpoint.  Its ``module_mass`` and
``environment_mass`` columns are the checkpoint's existing geometry-refined
source masses.  This command performs no temperature sweep, model mutation,
training, or checkpoint write.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    kcase = np.asarray([row["kcase"] for row in rows], dtype=np.int64)
    modules = np.asarray([row["active_module_count"] for row in rows], dtype=np.int64)
    frequency = np.asarray(rows[0]["population_prototype_frequency"], dtype=np.int64)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    bins = np.arange(int(kcase.min()) - 0.5, int(kcase.max()) + 1.5, 1.0)
    axes[0].hist(kcase, bins=bins, color="#32648e", edgecolor="white", linewidth=1.0)
    axes[0].set_xticks(np.arange(int(kcase.min()), int(kcase.max()) + 1))
    axes[0].set_xlabel(r"$K_{case}$")
    axes[0].set_ylabel("Cases")
    axes[0].set_title("Frozen 90-case competition")

    offsets = np.linspace(-0.12, 0.12, len(rows))
    axes[1].scatter(modules + offsets, kcase, s=19, alpha=0.72, color="#c15a37")
    lo = min(int(modules.min()), int(kcase.min()))
    hi = max(int(modules.max()), int(kcase.max()))
    axes[1].plot([lo, hi], [lo, hi], color="#565656", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Active modules")
    axes[1].set_ylabel(r"$K_{case}$")
    axes[1].set_title("Capacity is not module count")

    axes[2].bar(np.arange(len(frequency)), frequency, color="#568f61")
    axes[2].set_xticks(np.arange(len(frequency)))
    axes[2].set_xlabel("Original prototype ID")
    axes[2].set_ylabel("Selected cases")
    axes[2].set_title("Prototype selection frequency")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (PROJECT_ROOT / "src",):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch

    from honf_forward_core.interface_fields.mass_competitive_router import mass_competition

    table = Path(args.population_csv).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not table.is_file():
        raise FileNotFoundError(table)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with table.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != int(args.expected_cases):
        raise RuntimeError(
            f"expected {int(args.expected_cases)} frozen cases, found {len(source_rows)}"
        )
    module_mass = torch.tensor(
        [json.loads(row["module_mass"]) for row in source_rows], dtype=torch.float64
    )
    environment_mass = torch.tensor(
        [json.loads(row["environment_mass"]) for row in source_rows], dtype=torch.float64
    )
    pi = 0.5 * (module_mass + environment_mass)
    gamma = mass_competition(pi)
    active = gamma > 0.0
    kcase = active.sum(dim=-1)
    kappa = gamma.square().sum(dim=-1).reciprocal()
    retained = (pi * active).sum(dim=-1)
    module_count = torch.tensor(
        [int(row["active_module_count"]) for row in source_rows], dtype=torch.long
    )
    frequency = active.sum(dim=0).tolist()
    correlation = float(np.corrcoef(kcase.numpy(), module_count.numpy())[0, 1])
    if not math.isfinite(correlation):
        correlation = 0.0
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        rows.append(
            {
                "case_id": str(source["case_id"]),
                "active_module_count": int(module_count[index]),
                "kcase": int(kcase[index]),
                "kappa": float(kappa[index]),
                "retained_precompetition_mass": float(retained[index]),
                "active_prototype_ids": torch.where(active[index])[0].tolist(),
                "pi": pi[index].tolist(),
                "gamma": gamma[index].tolist(),
                "population_prototype_frequency": frequency,
            }
        )
    histogram = {
        str(key): int(value) for key, value in sorted(Counter(kcase.tolist()).items())
    }
    universal_k1 = bool(torch.all(kcase == 1))
    universal_kmax = bool(torch.all(kcase == int(pi.shape[1])))
    summary = {
        "schema_version": 1,
        "task": "run1500_frozen_mass_competition_prelaunch",
        "status": "stop_before_training" if universal_k1 or universal_kmax else "pass",
        "checkpoint": str(checkpoint),
        "checkpoint_selection": "latest occupancy-adaptive Run-1409 exact epoch 50",
        "source_population_csv": str(table),
        "case_count": len(rows),
        "transform": "sparsemax(log(pi + 1e-8))",
        "temperature_sweep": False,
        "kmax": int(pi.shape[1]),
        "kcase_histogram": histogram,
        "kcase": _summary(kcase.numpy().astype(np.float64)),
        "kcase_equals_module_count_cases": int(torch.sum(kcase == module_count)),
        "pearson_kcase_module_count": correlation,
        "prototype_selection_frequency": frequency,
        "kappa": _summary(kappa.numpy()),
        "retained_precompetition_mass": _summary(retained.numpy()),
        "gamma_row_sum_max_error": float((gamma.sum(dim=-1) - 1.0).abs().max()),
        "universal_k1": universal_k1,
        "universal_kmax": universal_kmax,
        "managed_training_allowed": not (universal_k1 or universal_kmax),
        "checkpoint_written": False,
        "model_mutated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    with (output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "case_id",
            "active_module_count",
            "kcase",
            "kappa",
            "retained_precompetition_mass",
            "active_prototype_ids",
            "pi",
            "gamma",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key]) if isinstance(row[key], list) else row[key]
                    for key in fields
                }
            )
    _render(rows, output_dir / "mass_competition_prelaunch.png")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--population-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=90)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run"]
