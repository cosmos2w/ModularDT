"""Render compact Run-1503 epoch-50 decision figures from validated evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sparse_incidence_evidence import (
    render_kq_case_spread,
    render_representative_hypergraph_overview,
)

RUN1503_LABEL = "Run1503_exact_epoch0050"
RUN1502_LABEL = "Run1502_exact_epoch0050"
COLORS = {"Run 1503": "#0072B2", "Run 1502": "#E69F00", "hybrid": "#009E73"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _row(rows: list[dict[str, str]], label: str) -> dict[str, str]:
    return next(row for row in rows if row["model_label"] == label)


def _finite_mean(rows: list[dict[str, str]], field: str, *, epochs: int = 50) -> float:
    values = [
        float(row[field])
        for row in rows[:epochs]
        if row.get(field) not in (None, "") and math.isfinite(float(row[field]))
    ]
    if not values:
        raise ValueError(f"no finite {field!r} values in {epochs} rows")
    return float(np.mean(values))


def _benchmark_scope_means(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scopes = ("full_physical_forward", "prepared_p2_decode", "application_evaluator")
    wall = np.asarray(
        [
            np.mean([float(row["timings"][scope]["median_wall_ms"]) for row in payload["rows"]])
            for scope in scopes
        ],
        dtype=np.float64,
    )
    memory = np.asarray(
        [
            np.mean(
                [
                    float(row["timings"][scope]["peak_incremental_allocated_bytes"])
                    for row in payload["rows"]
                ]
            )
            for scope in scopes
        ],
        dtype=np.float64,
    )
    return wall, memory


def _training_step(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = next(row for row in payload["rows"] if row["phase"] == "training_step")
    return float(row["median_ms"]), float(row["incremental_peak_allocated_bytes"])


def render_accuracy_cost(
    comparison_dir: Path,
    hybrid_comparison_dir: Path,
    output_path: Path,
) -> None:
    model_rows = _read_csv(comparison_dir / "tables" / "model_summary_metrics.csv")
    original_cost_rows = _read_csv(
        comparison_dir / "tables" / "evaluation_cost_summary_metrics.csv"
    )
    hybrid_cost_rows = _read_csv(
        hybrid_comparison_dir / "tables" / "evaluation_cost_summary_metrics.csv"
    )
    run1503 = _row(model_rows, RUN1503_LABEL)
    run1502 = _row(model_rows, RUN1502_LABEL)
    run1502_cost = _row(original_cost_rows, RUN1502_LABEL)
    hybrid_cost = hybrid_cost_rows[0]

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), constrained_layout=True)
    figure.suptitle(
        "Run 1503 fails the epoch-50 accuracy and application-cost gate",
        fontsize=14,
        fontweight="bold",
    )
    width = 0.34
    x = np.arange(3, dtype=np.float64)
    field_metrics = (
        "global_field_fluid_norm_l2_mean",
        "global_field_near_interface_norm_l2_mean",
        "global_field_far_fluid_norm_l2_mean",
    )
    for offset, row, label in (
        (-width / 2, run1503, "Run 1503"),
        (width / 2, run1502, "Run 1502"),
    ):
        axes[0].bar(
            x + offset,
            [float(row[field]) for field in field_metrics],
            width,
            label=label,
            color=COLORS[label],
        )
    axes[0].set_xticks(x, ("all fluid", "near interface", "far fluid"))
    axes[0].set_ylabel("mean relative L2 (lower is better)")
    axes[0].set_title("Matched 90-case field accuracy")
    axes[0].legend(frameon=False)

    physical_metrics = (
        "field_temperature_fluid_physical_mae_mean",
        "internal_temperature_physical_mae_mean",
        "interface_t_surface_physical_mae_mean",
        "interface_q_normal_physical_mae_mean",
        "port_h_effective_final_physical_mae_mean",
    )
    physical_names = ("fluid T", "internal T", "surface T", "heat flux", "effective h")
    ratios = np.asarray(
        [float(run1503[field]) / float(run1502[field]) for field in physical_metrics]
    )
    axes[1].bar(np.arange(len(ratios)), ratios, color=COLORS["Run 1503"])
    axes[1].axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(np.arange(len(ratios)), physical_names, rotation=18)
    axes[1].set_ylabel("Run 1503 / Run 1502 mean MAE")
    axes[1].set_title("Physical KPI regression ratios")

    cost_fields = (
        "evaluation_wall_time_seconds_mean",
        "evaluation_cuda_incremental_peak_allocated_mib_mean",
    )
    cost_names = ("mean wall time", "incremental peak memory")
    cost_ratios = np.asarray(
        [float(hybrid_cost[field]) / float(run1502_cost[field]) for field in cost_fields]
    )
    axes[2].bar(np.arange(2), cost_ratios, color=COLORS["hybrid"])
    axes[2].axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    axes[2].set_xticks(np.arange(2), cost_names, rotation=12)
    axes[2].set_ylabel("Run 1503 hybrid / Run 1502")
    axes[2].set_title("Final maps-off application cost")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, facecolor="white")
    figure.savefig(output_path.with_suffix(".pdf"), format="pdf", facecolor="white")
    plt.close(figure)


def render_executor_diagnostics(
    *,
    run1502_metrics: Path,
    run1503_metrics: Path,
    scalar_chunk32: Path,
    hybrid_chunk32: Path,
    scalar_chunk2048: Path,
    hybrid_chunk2048: Path,
    scalar_training_step: Path,
    hybrid_training_step: Path,
    output_path: Path,
) -> None:
    run1502_rows = _read_csv(run1502_metrics)
    run1503_rows = _read_csv(run1503_metrics)
    formal_fields = ("train_wall_seconds", "val_wall_seconds", "peak_cuda_memory_mb")
    formal_ratios = np.asarray(
        [
            _finite_mean(run1503_rows, field) / _finite_mean(run1502_rows, field)
            for field in formal_fields
        ]
    )
    scalar32_wall, _ = _benchmark_scope_means(scalar_chunk32)
    hybrid32_wall, _ = _benchmark_scope_means(hybrid_chunk32)
    scalar2048_wall, scalar2048_memory = _benchmark_scope_means(scalar_chunk2048)
    hybrid2048_wall, hybrid2048_memory = _benchmark_scope_means(hybrid_chunk2048)
    scalar_step_ms, scalar_step_memory = _training_step(scalar_training_step)
    hybrid_step_ms, hybrid_step_memory = _training_step(hybrid_training_step)

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), constrained_layout=True)
    figure.suptitle(
        "Run 1503 slowdown diagnosis and bounded hybrid-executor correction",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].bar(np.arange(3), formal_ratios, color=("#0072B2", "#56B4E9", "#CC79A7"))
    axes[0].axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    axes[0].set_xticks(np.arange(3), ("train epoch", "validation", "peak memory"))
    axes[0].set_ylabel("formal Run 1503 / Run 1502")
    axes[0].set_title("Observed epochs 1–50")

    scope_names = ("full forward", "prepared P2", "application")
    x = np.arange(3, dtype=np.float64)
    width = 0.34
    axes[1].bar(
        x - width / 2,
        hybrid32_wall / scalar32_wall,
        width,
        label="receiver tile 32",
        color="#009E73",
    )
    axes[1].bar(
        x + width / 2,
        hybrid2048_wall / scalar2048_wall,
        width,
        label="receiver tile 2048",
        color="#E69F00",
    )
    axes[1].axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x, scope_names, rotation=12)
    axes[1].set_ylabel("hybrid / scalar median wall time")
    axes[1].set_title("Exact e50 Q8192 anchor benchmark")
    axes[1].legend(frameon=False, fontsize=8)

    step_ratios = np.asarray(
        [hybrid_step_ms / scalar_step_ms, hybrid_step_memory / scalar_step_memory]
    )
    axes[2].bar(np.arange(2), step_ratios, color=("#009E73", "#CC79A7"))
    axes[2].axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    axes[2].set_xticks(np.arange(2), ("optimizer-step wall time", "incremental peak memory"))
    axes[2].tick_params(axis="x", rotation=12)
    axes[2].set_ylabel("hybrid / scalar")
    axes[2].set_title("Nonpersistent 48×1024 training step")
    axes[2].text(
        0.98,
        0.02,
        f"large-tile memory ratio: {np.mean(hybrid2048_memory / scalar2048_memory):.2f}×",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

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
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--hybrid-comparison-dir", required=True, type=Path)
    parser.add_argument("--run1502-metrics", required=True, type=Path)
    parser.add_argument("--run1503-metrics", required=True, type=Path)
    parser.add_argument("--scalar-chunk32", required=True, type=Path)
    parser.add_argument("--hybrid-chunk32", required=True, type=Path)
    parser.add_argument("--scalar-chunk2048", required=True, type=Path)
    parser.add_argument("--hybrid-chunk2048", required=True, type=Path)
    parser.add_argument("--scalar-training-step", required=True, type=Path)
    parser.add_argument("--hybrid-training-step", required=True, type=Path)
    parser.add_argument("--population-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_accuracy_cost(
        args.comparison_dir.expanduser().resolve(),
        args.hybrid_comparison_dir.expanduser().resolve(),
        output_dir / "run1503_epoch50_accuracy_cost.png",
    )
    render_executor_diagnostics(
        run1502_metrics=args.run1502_metrics.expanduser().resolve(),
        run1503_metrics=args.run1503_metrics.expanduser().resolve(),
        scalar_chunk32=args.scalar_chunk32.expanduser().resolve(),
        hybrid_chunk32=args.hybrid_chunk32.expanduser().resolve(),
        scalar_chunk2048=args.scalar_chunk2048.expanduser().resolve(),
        hybrid_chunk2048=args.hybrid_chunk2048.expanduser().resolve(),
        scalar_training_step=args.scalar_training_step.expanduser().resolve(),
        hybrid_training_step=args.hybrid_training_step.expanduser().resolve(),
        output_path=output_dir / "run1503_executor_diagnostics.png",
    )
    population_dir = args.population_dir.expanduser().resolve()
    evidence = json.loads((population_dir / "evidence.json").read_text(encoding="utf-8"))
    rows = evidence["cases"]
    render_kq_case_spread(
        rows,
        output_dir / "run1503_kq_case_spread.png",
        run_label="Run 1503 exact epoch 50",
    )
    render_representative_hypergraph_overview(
        rows,
        population_dir / "arrays",
        output_dir / "run1503_representative_hypergraph__0273__0653.png",
        run_label="Run 1503 exact epoch 50",
    )
    print(f"wrote report figures to {output_dir}")


if __name__ == "__main__":
    main()
