"""Render the compact Run-1502 epoch-500 report figures from validated evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sparse_incidence_evidence import (
    render_kq_case_spread,
    render_representative_hypergraph_overview,
)

LABELS = (
    "Run1502_exact_epoch500",
    "Run1501_exact_epoch500",
    "Run1804_dense_epoch500",
)
DISPLAY_LABELS = ("Run 1502 sparsemax", "Run 1501 entmax-1.5", "Run 1804 dense")
COLORS = ("#0072B2", "#E69F00", "#009E73")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _indexed(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def render_comparison(reduction_dir: Path, output_path: Path) -> None:
    pooled = {
        (row["model_label"], row["metric"]): float(row["relative_l2"])
        for row in _read_csv(reduction_dir / "pooled_metrics.csv")
    }
    distributions = {
        (row["model_label"], row["metric"]): float(row["mean"])
        for row in _read_csv(reduction_dir / "equal_case_distributions.csv")
    }
    costs = _indexed(_read_csv(reduction_dir / "cost_summary.csv"), "model_label")

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), constrained_layout=True)
    figure.suptitle(
        "Exact epoch-500 comparison on the matched 90-case development holdout",
        fontsize=14,
        fontweight="bold",
    )
    width = 0.23
    x = np.arange(3, dtype=np.float64)
    pooled_metrics = (
        "global_field_fluid_norm",
        "global_field_near_interface_norm",
        "global_field_far_fluid_norm",
    )
    for index, (label, display, color) in enumerate(
        zip(LABELS, DISPLAY_LABELS, COLORS, strict=True)
    ):
        values = [pooled[(label, metric)] for metric in pooled_metrics]
        axes[0].bar(x + (index - 1) * width, values, width, label=display, color=color)
    axes[0].set_xticks(x, ("all fluid", "near interface", "far fluid"))
    axes[0].set_ylabel("pooled relative L2 (lower is better)")
    axes[0].set_title("Field reconstruction")
    axes[0].legend(frameon=False, fontsize=8)

    physical_metrics = (
        "field_temperature_fluid_physical_mae",
        "internal_temperature_physical_mae",
        "interface_t_surface_physical_mae",
        "port_h_effective_final_physical_mae",
    )
    physical_names = ("fluid T", "internal T", "surface T", "effective h")
    baseline = np.asarray(
        [distributions[(LABELS[1], metric)] for metric in physical_metrics], dtype=np.float64
    )
    x = np.arange(len(physical_metrics), dtype=np.float64)
    for index, (label, display, color) in enumerate(
        zip(LABELS, DISPLAY_LABELS, COLORS, strict=True)
    ):
        values = np.asarray(
            [distributions[(label, metric)] for metric in physical_metrics], dtype=np.float64
        )
        axes[1].bar(x + (index - 1) * width, values / baseline, width, label=display, color=color)
    axes[1].axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x, physical_names)
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_ylabel("mean MAE / Run 1501 mean MAE")
    axes[1].set_title("Selected thermal and port KPIs")

    cost_fields = (
        "evaluation_wall_time_seconds_mean",
        "evaluation_cuda_peak_allocated_mib_mean",
    )
    cost_names = ("mean wall time", "peak allocated memory")
    cost_baseline = np.asarray(
        [float(costs[LABELS[1]][field]) for field in cost_fields], dtype=np.float64
    )
    x = np.arange(len(cost_fields), dtype=np.float64)
    for index, (label, display, color) in enumerate(
        zip(LABELS, DISPLAY_LABELS, COLORS, strict=True)
    ):
        values = np.asarray([float(costs[label][field]) for field in cost_fields])
        axes[2].bar(x + (index - 1) * width, values / cost_baseline, width, label=display, color=color)
    axes[2].axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    axes[2].set_xticks(x, cost_names)
    axes[2].tick_params(axis="x", rotation=14)
    axes[2].set_ylabel("cost / Run 1501 cost")
    axes[2].set_title("Matched maps-off application cost")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, facecolor="white")
    figure.savefig(output_path.with_suffix(".pdf"), format="pdf", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduction-dir", required=True, type=Path)
    parser.add_argument("--population-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-label", default="Run 1502 exact epoch 500")
    args = parser.parse_args()

    reduction_dir = args.reduction_dir.expanduser().resolve()
    population_dir = args.population_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_comparison(reduction_dir, output_dir / "epoch500_accuracy_cost.png")

    evidence = json.loads((population_dir / "evidence.json").read_text(encoding="utf-8"))
    rows = evidence["cases"]
    render_kq_case_spread(
        rows,
        output_dir / "run1502_kq_case_spread.png",
        run_label=args.run_label,
    )
    render_representative_hypergraph_overview(
        rows,
        population_dir / "arrays",
        output_dir / "run1502_representative_hypergraph__0273__0653.png",
        run_label=args.run_label,
    )
    print(f"wrote report figures to {output_dir}")


if __name__ == "__main__":
    main()
