"""Render the Run-1500 epoch-50 decision board from frozen evaluation tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHYSICAL_METRICS = (
    ("Fluid", "global_field_fluid_pooled_relative_l2"),
    ("Near interface", "global_field_near_interface_pooled_relative_l2"),
    ("Vorticity", "field_omega_fluid_pooled_relative_l2"),
    ("Field T", "field_temperature_fluid_pooled_relative_l2"),
    ("Surface T", "interface_t_surface_physical_pooled_relative_l2"),
    ("Heat flux", "interface_q_normal_physical_pooled_relative_l2"),
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render(population_dir: Path, comparison_dir: Path, output: Path) -> None:
    summary = json.loads((population_dir / "population_summary.json").read_text())
    cases = _rows(population_dir / "population_cases.csv")
    physical = _rows(comparison_dir / "tables" / "model_summary_metrics.csv")
    cost = _rows(comparison_dir / "tables" / "evaluation_cost_summary_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), constrained_layout=True)
    fig.suptitle("Run 1500 epoch-50 decision evidence", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    kcase = np.asarray([int(row["kcase"]) for row in cases])
    kappa = np.asarray([float(row["kappa"]) for row in cases])
    ax.hist(kappa, bins=14, color="#4C78A8", alpha=0.85)
    ax.axvline(float(np.mean(kappa)), color="#E45756", linewidth=2, label=f"mean κ={np.mean(kappa):.3f}")
    ax.set(xlabel="Continuous complexity κ", ylabel="Cases", title="Discrete Kcase is fixed at 2 in all 90 cases")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    support = [
        float(summary["module_RM_support"]["mean"]),
        float(summary["environment_RE_support"]["mean"]),
        float(summary["p2_module_actual_rows"]["mean"]) / (1024.0 * 12.0),
        float(summary["p2_environment_actual_rows"]["mean"]) / (1024.0 * 192.0),
    ]
    labels = ["Module\nunique support", "Environment\nunique support", "Module\nexecuted rows", "Environment\nexecuted rows"]
    bars = ax.bar(labels, support, color=["#59A14F", "#59A14F", "#F28E2B", "#F28E2B"])
    ax.set_ylim(0.9, 1.01)
    ax.set(ylabel="Fraction of dense rectangle", title="Logical support and actual execution remain dense")
    ax.bar_label(bars, labels=[f"{100.0 * value:.2f}%" for value in support], padding=3)

    ax = axes[1, 0]
    x = np.arange(len(PHYSICAL_METRICS))
    width = 0.19
    for index, row in enumerate(physical):
        values = [float(row[key]) for _, key in PHYSICAL_METRICS]
        label = row["model_label"].replace("_epoch50", "").replace("Run1500_mass", "Run1500")
        ax.bar(x + (index - 1.5) * width, values, width, label=label)
    ax.set_xticks(x, [label for label, _ in PHYSICAL_METRICS], rotation=24, ha="right")
    ax.set(ylabel="Pooled relative L2 (lower is better)", title="Matched 90-case physical fidelity")
    ax.legend(frameon=False, fontsize=8, ncols=2)

    ax = axes[1, 1]
    colors = ["#E45756", "#4C78A8", "#59A14F", "#F28E2B"]
    for color, row in zip(colors, cost, strict=True):
        latency = 1000.0 * float(row["evaluation_wall_time_seconds_median"])
        memory = float(row["evaluation_cuda_incremental_peak_allocated_mib_mean"])
        label = row["model_label"].replace("_epoch50", "").replace("Run1500_mass", "Run1500")
        ax.scatter(latency, memory, s=85, color=color)
        ax.annotate(label, (latency, memory), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Synchronized full-case median latency (ms)", ylabel="Incremental peak CUDA memory (MiB)", title="Measured cost; no execution saving")
    ax.grid(alpha=0.22)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-dir", required=True, type=Path)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.population_dir.resolve(), args.comparison_dir.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
