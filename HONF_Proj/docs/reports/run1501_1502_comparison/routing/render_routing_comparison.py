#!/usr/bin/env python3
"""Render reproducible routing summaries from saved best-checkpoint artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

ROUTING_DIR = Path(__file__).resolve().parent
REPORT_DIR = ROUTING_DIR.parent
ACCURACY_DIR = REPORT_DIR / "accuracy" / "evaluation"
DEBUG_DIR = ACCURACY_DIR / "debug_npz"
TABLE_DIR = ACCURACY_DIR / "tables"
FIGURE_DIR = ROUTING_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = REPORT_DIR.parents[2]

POPULATIONS = {
    "Run 1501": {
        "folder": "population_run1501_best_field_e4689_q1024",
        "checkpoint": "best-field e4689",
        "debug": "Run1501_best_field_e4689__0653.npz",
        "label": "Run1501_best_field_e4689",
        "color": "#2878b5",
        "marker": "o",
    },
    "Run 1502": {
        "folder": "population_run1502_best_field_e4794_q1024",
        "checkpoint": "best-field e4794",
        "debug": "Run1502_best_field_e4794__0653.npz",
        "label": "Run1502_best_field_e4794",
        "color": "#e07b39",
        "marker": "^",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_artifacts(name: str) -> tuple[Path, dict[str, object], list[dict[str, str]]]:
    spec = POPULATIONS[name]
    folder = ROUTING_DIR / str(spec["folder"])
    summary = json.loads((folder / "population_summary.json").read_text())
    cases = read_csv(folder / "population_cases.csv")
    return folder, summary, cases


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Average ranks for tied values.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for idx, count in enumerate(counts):
        if count > 1:
            positions = np.flatnonzero(inverse == idx)
            ranks[positions] = ranks[positions].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def build_distribution_tables() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = [
        {
            "run": "Run 1404",
            "checkpoint": "best-field e4890",
            "registered_capacity_K": 6,
            "Kq_definition": "positive query softmax support; architectural dense support",
            "Kq_min": 6,
            "Kq_max": 6,
            "Kq_mean": 6.0,
            "Kq_case_mean_min": "",
            "Kq_case_mean_max": "",
            "Kq_case_mean_mean": "",
            "case_count": 90,
            "occupied_registered_groups_mean": 6,
            "active_modules_min": "",
            "active_modules_max": "",
            "active_modules_mean": "",
            "environment_source_support_RE_mean": "",
            "module_source_support_RM_mean": "",
            "query_population_note": "K=6 and softmax imply positive support on all six groups; support count is not adaptively selected.",
        }
    ]
    histogram_rows: list[dict[str, object]] = []
    for name, spec in POPULATIONS.items():
        _, summary, cases = model_artifacts(name)
        counts = {int(k): int(v) for k, v in summary["kq_histogram"].items()}
        query_count = sum(counts.values())
        case_means = np.array([float(row["query_degree_mean"]) for row in cases])
        active_modules = np.array([float(row["M_active"]) for row in cases])
        hist_min, hist_max = min(counts), max(counts)
        summary_rows.append(
            {
                "run": name,
                "checkpoint": spec["checkpoint"],
                "registered_capacity_K": summary["registered_capacity"],
                "Kq_definition": "exact positive support count in query_assignment",
                "Kq_min": hist_min,
                "Kq_max": hist_max,
                "Kq_mean": sum(k * v for k, v in counts.items()) / query_count,
                "Kq_case_mean_min": float(case_means.min()),
                "Kq_case_mean_max": float(case_means.max()),
                "Kq_case_mean_mean": float(case_means.mean()),
                "case_count": len(cases),
                "occupied_registered_groups_mean": float(summary["occupied_source_groups"]["mean"]),
                "active_modules_min": float(active_modules.min()),
                "active_modules_max": float(active_modules.max()),
                "active_modules_mean": float(active_modules.mean()),
                "environment_source_support_RE_mean": float(summary["environment_RE_support"]["mean"]),
                "module_source_support_RM_mean": float(summary["module_RM_support"]["mean"]),
                "query_population_note": f"{query_count:,} query assignments; Q=1024 deterministic queries per case across the 90-case test split.",
            }
        )
        for k in range(1, 13):
            count = counts.get(k, 0)
            histogram_rows.append(
                {
                    "run": name,
                    "checkpoint": spec["checkpoint"],
                    "Kq": k,
                    "query_count": count,
                    "fraction": count / query_count,
                    "percent": 100.0 * count / query_count,
                    "total_query_assignments": query_count,
                }
            )
    return summary_rows, histogram_rows


def plot_kq_histogram(histogram_rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, name in zip(axes, POPULATIONS):
        rows = [row for row in histogram_rows if row["run"] == name]
        ks = np.array([int(row["Kq"]) for row in rows])
        pct = np.array([float(row["percent"]) for row in rows])
        mean = float(np.sum(ks * pct) / 100.0)
        ax.bar(ks, pct, color=POPULATIONS[name]["color"], width=0.78, alpha=0.88)
        ax.set_xticks(np.arange(1, 13))
        ax.set_xlim(0.4, 12.6)
        ax.set_xlabel("Exact query support Kq")
        ax.set_title(f"{name} · mean Kq={mean:.3f}")
        ax.grid(axis="y", alpha=0.2)
        ax.axvline(12, color="#555555", ls="--", lw=1)
        ax.text(11.85, max(pct) * 0.87, "capacity 12", rotation=90, ha="right", va="top", color="#444444", fontsize=8)
    axes[0].set_ylabel("Share of sampled held-out queries (%)")
    fig.suptitle("Best-field checkpoint query-support distribution · 90 cases × 1,024 queries")
    fig.text(0.5, 0.01, "Kq counts exactly positive query-router entries. All 12 groups remain registered and occupied; this support statistic does not prove reduced executed rows.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(FIGURE_DIR / "kq_distribution_best_field_q1024.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_fullgrid_kq_histogram() -> None:
    path = ROUTING_DIR / "fullgrid_kq_histogram.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, name in zip(axes, POPULATIONS):
        selected = [row for row in rows if row["run"] == name]
        ks = np.array([int(row["Kq"]) for row in selected])
        counts = np.array([int(row["query_count"]) for row in selected])
        total = int(counts.sum())
        pct = 100.0 * counts / total
        mean = float(np.sum(ks * counts) / total)
        ax.bar(ks, pct, color=POPULATIONS[name]["color"], width=0.78, alpha=0.88)
        ax.set_xticks(np.arange(1, 13))
        ax.set_xlim(0.4, 12.6)
        ax.set_xlabel("Exact query support Kq")
        ax.set_title(f"{name} · mean Kq={mean:.3f}")
        ax.grid(axis="y", alpha=0.2)
        ax.axvline(12, color="#555555", ls="--", lw=1)
        ax.text(11.85, max(pct) * 0.87, "capacity 12", rotation=90, ha="right", va="top", color="#444444", fontsize=8)
    axes[0].set_ylabel("Share of held-out query grid (%)")
    fig.suptitle("Exact full-grid Kq distribution · best-field checkpoints · 90 × 8,192 queries")
    fig.text(0.5, 0.01, "Kq counts exactly positive query-router entries from every original grid point. All 12 groups remain registered and occupied; logical support does not imply skipped executor rows.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(FIGURE_DIR / "kq_distribution_fullgrid_q8192.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_case0653_spatial() -> None:
    old_best = np.load(REPO_ROOT / "diagnostics" / "generated" / "run1404_1406_1407_1804_best5000_accuracy_20260920" / "debug_npz" / "Run1404_best_field__0653.npz")
    old_routes_path = next((ROUTING_DIR / "run1404_best_field_0653").glob("*/routing/routing_maps.npz"))
    old_routes = np.load(old_routes_path)
    datasets: list[dict[str, object]] = [
        {
            "name": "Run 1404 · fixed K=6",
            "type": "dominant edge by pairwise field contribution",
            "groups": old_routes["dominant_hyperedge"],
            "kq": np.full((64, 128), 6, dtype=np.int8),
            "centres": old_best["module_centers"],
            "present": old_best["module_present"].astype(bool),
            "fluid": old_best["fluid_mask"].astype(bool),
        }
    ]
    for name, spec in POPULATIONS.items():
        data = np.load(DEBUG_DIR / str(spec["debug"]))
        assignment = data["interaction__group_control_query_routing"].reshape(64, 128, -1)
        datasets.append(
            {
                "name": f"{name} · registered K=12",
                "type": "dominant query-router group",
                "groups": assignment.argmax(axis=-1),
                "kq": (assignment > 0).sum(axis=-1),
                "centres": data["module_centers"],
                "present": data["module_present"].astype(bool),
                "fluid": data["fluid_mask"].astype(bool),
            }
        )

    group_cmap = ListedColormap(plt.get_cmap("tab20").colors[:12])
    kq_cmap = plt.get_cmap("viridis", 12)
    kq_norm = BoundaryNorm(np.arange(0.5, 13.5, 1), kq_cmap.N)
    fig = plt.figure(figsize=(14.2, 7.3))
    grid = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 0.045), height_ratios=(1, 1), hspace=0.3, wspace=0.12)
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)])
    kq_color_axis = fig.add_subplot(grid[:, 3])
    for col, record in enumerate(datasets):
        groups = np.asarray(record["groups"])
        support = np.asarray(record["kq"])
        fluid = np.asarray(record["fluid"])
        centers = np.asarray(record["centres"])
        present = np.asarray(record["present"], dtype=bool)
        shown_groups = np.ma.masked_where(~fluid, groups)
        shown_support = np.ma.masked_where(~fluid, support)
        ax = axes[0, col]
        ax.set_facecolor("#e4e6e8")
        ax.imshow(shown_groups, origin="lower", extent=(0, 12, 0, 6), interpolation="nearest", cmap=group_cmap, vmin=-0.5, vmax=11.5, aspect="equal")
        ax.set_title(f"{record['name']}\n{record['type']}", fontsize=10)
        for module_id, (x, y) in enumerate(centers[present]):
            actual_id = int(np.flatnonzero(present)[module_id])
            ax.scatter([x], [y], s=68, facecolors="white", edgecolors="#202020", linewidths=1.2, zorder=5)
            ax.text(x, y, str(actual_id), ha="center", va="center", fontsize=7, color="#111111", zorder=6)
        ax = axes[1, col]
        ax.set_facecolor("#e4e6e8")
        ax.imshow(shown_support, origin="lower", extent=(0, 12, 0, 6), interpolation="nearest", cmap=kq_cmap, norm=kq_norm, aspect="equal")
        ax.set_title("Exact per-query Kq support", fontsize=10)
        for module_id, (x, y) in enumerate(centers[present]):
            actual_id = int(np.flatnonzero(present)[module_id])
            ax.scatter([x], [y], s=68, facecolors="white", edgecolors="#202020", linewidths=1.2, zorder=5)
            ax.text(x, y, str(actual_id), ha="center", va="center", fontsize=7, color="#111111", zorder=6)
        for row in range(2):
            axes[row, col].set_xlim(0, 12)
            axes[row, col].set_ylim(0, 6)
            if row == 1:
                axes[row, col].set_xlabel("x")
        axes[0, col].set_ylabel("y")
        axes[1, col].set_ylabel("y")
    kq_mappable = plt.cm.ScalarMappable(cmap=kq_cmap, norm=kq_norm)
    kq_cb = fig.colorbar(kq_mappable, cax=kq_color_axis, ticks=np.arange(1, 8))
    kq_cb.set_label("Kq support")
    fig.suptitle("Case 0653 routing map at each model's best-field checkpoint", y=0.995, fontsize=14)
    fig.text(0.5, 0.025, "Colors and group IDs are model-local (permutation ambiguous). Run 1404 shows dominant contribution; 1501/1502 show query-router argmax. Numbered circles are physical module centers; learned organization is not physical causality.", ha="center", fontsize=8.2)
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.055, right=0.96)
    fig.savefig(FIGURE_DIR / "case0653_spatial_routing_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_module_affinity() -> None:
    old_best_path = REPO_ROOT / "diagnostics" / "generated" / "run1404_1406_1407_1804_best5000_accuracy_20260920" / "debug_npz" / "Run1404_best_field__0653.npz"
    old_best = np.load(old_best_path)
    old_plan_path = next((ROUTING_DIR / "run1404_best_field_0653").glob("*/plans/hypergraph_plan.npz"))
    old_plan = np.load(old_plan_path)
    models: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("Run 1404 · H0–H5", old_plan["A_mh"], old_best["module_present"].astype(bool))
    ]
    for name, spec in POPULATIONS.items():
        arr = np.load(ROUTING_DIR / str(spec["folder"]) / "arrays" / "0653.npz")
        models.append((f"{name} · H0–H11", arr["module_assignment"], arr["module_present"].astype(bool)))
    fig = plt.figure(figsize=(13.4, 5.1))
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.045), wspace=0.2)
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]
    color_axis = fig.add_subplot(grid[0, 3])
    image = None
    for ax, (name, matrix, present) in zip(axes, models):
        image = ax.imshow(matrix, origin="upper", aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        group_count = matrix.shape[1]
        ax.set_xticks(np.arange(group_count))
        ax.set_xticklabels([f"H{i}" for i in range(group_count)], rotation=90, fontsize=8)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([f"M{i}{'*' if present[i] else ''}" for i in range(matrix.shape[0])], fontsize=8)
        ax.set_xlabel("model-local latent group")
        ax.set_title(name)
        ax.set_ylabel("physical module row (* active)")
    fig.colorbar(image, cax=color_axis, label="module-to-group weight")
    fig.suptitle("Case 0653 module clustering / group affinity", y=0.99)
    fig.text(0.5, 0.01, "Rows are physical module IDs; latent group columns and colors are model-specific. Group labels are permutation-ambiguous, so cross-model matching by H index is not meaningful.", ha="center", fontsize=8.5)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.07, right=0.97)
    fig.savefig(FIGURE_DIR / "case0653_module_group_affinity.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_kq_error_scatter() -> None:
    accuracy_rows = read_csv(TABLE_DIR / "per_case_metrics.csv")
    accuracy_labels = {spec["label"]: name for name, spec in POPULATIONS.items()}
    error_by_model_case: dict[tuple[str, str], float] = {}
    for row in accuracy_rows:
        if row["model_label"] in accuracy_labels:
            error_by_model_case[(accuracy_labels[row["model_label"]], row["case_id"])] = float(row["global_field_fluid_norm_l2"])
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    scatter_rows: list[dict[str, object]] = []
    fullgrid_path = ROUTING_DIR / "fullgrid_kq_cases.csv"
    fullgrid_case_means: dict[tuple[str, str], float] = {}
    if fullgrid_path.exists():
        for row in read_csv(fullgrid_path):
            fullgrid_case_means[(row["run"], row["case_id"])] = float(row["Kq_mean"])
    kq_source = "Q=8192 exact full-grid" if fullgrid_case_means else "Q=1024 deterministic sample"
    for name, spec in POPULATIONS.items():
        _, _, cases = model_artifacts(name)
        xs, ys, colors = [], [], []
        for row in cases:
            case_id = row["case_id"]
            key = (name, case_id)
            if key not in error_by_model_case:
                continue
            error = error_by_model_case[key]
            mean_kq = fullgrid_case_means.get(key, float(row["query_degree_mean"]))
            active_modules = int(row["M_active"])
            xs.append(error)
            ys.append(mean_kq)
            colors.append(active_modules)
            scatter_rows.append(
                {
                    "run": name,
                    "checkpoint": spec["checkpoint"],
                    "case_id": case_id,
                    "mean_Kq_fullgrid_Q8192": mean_kq,
                    "active_modules": active_modules,
                    "global_field_fluid_norm_l2_Q8192": error,
                }
            )
        ax.scatter(xs, ys, c=colors, cmap="viridis", vmin=3, vmax=10, s=31, alpha=0.75, marker=spec["marker"], edgecolors="white", linewidths=0.35, label=name)
    ax.set_xlabel("Global fluid normalized relative L2 error (Q=8192 accuracy evaluation)")
    ax.set_ylabel(f"Mean exact query support Kq ({kq_source})")
    ax.grid(alpha=0.22)
    ax.legend(title="best-field checkpoint", frameon=False)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(3, 10)), ax=ax, pad=0.025)
    colorbar.set_label("active physical modules")
    fig.suptitle("Descriptive case-level relation between Kq and fluid error", y=0.99)
    fig.text(0.5, 0.005, f"Descriptive association only: mean Kq comes from {kq_source}; fluid error uses the full Q=8192 evaluation. This plot does not establish that Kq causes error or runtime changes.", ha="center", fontsize=8.2)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(FIGURE_DIR / "case_mean_kq_vs_fluid_error.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    write_csv(ROUTING_DIR / "case_kq_vs_fluid_error.csv", scatter_rows)


def main() -> None:
    summary_rows, histogram_rows = build_distribution_tables()
    write_csv(ROUTING_DIR / "kq_population_summary.csv", summary_rows)
    write_csv(ROUTING_DIR / "kq_histogram.csv", histogram_rows)
    plot_kq_histogram(histogram_rows)
    plot_fullgrid_kq_histogram()
    render_case0653_spatial()
    render_module_affinity()
    render_kq_error_scatter()
    print(f"Wrote routing figures and tables under {ROUTING_DIR}")


if __name__ == "__main__":
    main()
