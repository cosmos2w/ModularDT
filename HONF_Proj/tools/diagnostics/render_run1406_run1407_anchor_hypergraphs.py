#!/usr/bin/env python3
"""Render matched Run-1406/1407 anchor hypergraphs from saved routing maps.

The figure is deliberately a learned-interaction visualization rather than a
physical decomposition.  Group identities are exchangeable, so every panel is
relabelled independently: module-bearing groups come first by module mass and
the remaining groups follow by query mass.  The spatial map and the aggregated
source-to-group-to-query diagram share that panel-local ordering.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTING_DIR = (
    PROJECT
    / "diagnostics/generated/run1406_run1407_best5000_routing_organization_20260920"
)
DEFAULT_CASES = ("0273", "0653")
DEFAULT_RUNS = ("1406", "1407")
GROUP_COUNT = 6
EPS = 1.0e-8
PALETTE = np.asarray(
    [
        to_rgb("#0072B2"),
        to_rgb("#D55E00"),
        to_rgb("#009E73"),
        to_rgb("#CC79A7"),
        to_rgb("#E69F00"),
        to_rgb("#56B4E9"),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PanelData:
    run: str
    case_id: str
    epoch: int
    query_xy: np.ndarray
    query_routing: np.ndarray
    module_present: np.ndarray
    module_centers: np.ndarray
    module_incidence: np.ndarray
    environment_incidence: np.ndarray
    group_order: np.ndarray
    metrics: Mapping[str, float]

    @property
    def active_module_indices(self) -> np.ndarray:
        return np.flatnonzero(self.module_present > 0.5)

    @property
    def ordered_query(self) -> np.ndarray:
        return self.query_routing[:, self.group_order]

    @property
    def ordered_modules(self) -> np.ndarray:
        return self.module_incidence[:, self.group_order]

    @property
    def ordered_environment(self) -> np.ndarray:
        return self.environment_incidence[:, self.group_order]


def canonical_group_order(
    module_incidence: np.ndarray,
    environment_incidence: np.ndarray,
    query_routing: np.ndarray,
) -> np.ndarray:
    """Return a deterministic panel-local order without asserting group identity."""
    module_mass = module_incidence.sum(axis=0)
    environment_mass = environment_incidence.sum(axis=0)
    query_mass = query_routing.mean(axis=0)
    module_groups = [index for index in range(GROUP_COUNT) if module_mass[index] > EPS]
    other_groups = [index for index in range(GROUP_COUNT) if module_mass[index] <= EPS]
    module_groups.sort(
        key=lambda index: (-module_mass[index], -query_mass[index], index)
    )
    other_groups.sort(
        key=lambda index: (-query_mass[index], -environment_mass[index], index)
    )
    return np.asarray(module_groups + other_groups, dtype=np.int64)


def _read_case_metrics(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    wanted = {
        "query_support_degree_mean",
        "query_effective_groups_mean",
        "query_entropy_norm_mean",
        "module_occupied_group_count",
        "environment_occupied_group_count",
        "module_unique_pair_density",
        "environment_unique_pair_density",
    }
    result: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            label = str(row.get("label", ""))
            case_id = str(row.get("case_id", ""))
            if not label or not case_id:
                continue
            result[(label, case_id)] = {
                key: float(row[key]) for key in wanted if row.get(key) not in (None, "")
            }
    return result


def _read_epochs(path: Path) -> dict[str, int]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(label): int(values["epoch"])
        for label, values in manifest.get("provenance", {}).items()
        if "epoch" in values
    }


def _squeeze_group_matrix(array: np.ndarray, expected_rows: int | None = None) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or value.shape[1] != GROUP_COUNT:
        raise ValueError(f"expected [N,{GROUP_COUNT}] group matrix, got {value.shape}")
    if expected_rows is not None and value.shape[0] != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {value.shape[0]}")
    return value


def load_panel(
    routing_dir: Path,
    run: str,
    case_id: str,
    case_metrics: Mapping[tuple[str, str], Mapping[str, float]],
    epochs: Mapping[str, int],
) -> PanelData:
    path = routing_dir / "maps" / f"{run}__{case_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        query_xy = np.asarray(values["query_xy"], dtype=np.float64)
        module_present = np.asarray(values["module_present"], dtype=np.float64).reshape(-1)
        module_centers = np.asarray(values["module_centers"], dtype=np.float64)
        query_routing = _squeeze_group_matrix(
            values["group_control_query_routing"], len(query_xy)
        )
        module_incidence = _squeeze_group_matrix(
            values["group_control_module_incidence"], len(module_present)
        )
        environment_incidence = _squeeze_group_matrix(
            values["group_control_environment_incidence"]
        )
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError(f"expected [Q,2] query coordinates, got {query_xy.shape}")
    if module_centers.shape != (len(module_present), 2):
        raise ValueError("module centers do not match module_present")
    active = module_present > 0.5
    order = canonical_group_order(
        module_incidence[active], environment_incidence, query_routing
    )
    return PanelData(
        run=run,
        case_id=case_id,
        epoch=int(epochs.get(run, -1)),
        query_xy=query_xy,
        query_routing=query_routing,
        module_present=module_present,
        module_centers=module_centers,
        module_incidence=module_incidence,
        environment_incidence=environment_incidence,
        group_order=order,
        metrics=case_metrics.get((run, case_id), {}),
    )


def _query_grid(panel: PanelData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.unique(panel.query_xy[:, 0])
    y_values = np.unique(panel.query_xy[:, 1])
    if len(x_values) * len(y_values) != len(panel.query_xy):
        raise ValueError("query coordinates are not a complete rectangular grid")
    x_index = np.searchsorted(x_values, panel.query_xy[:, 0])
    y_index = np.searchsorted(y_values, panel.query_xy[:, 1])
    routing = panel.ordered_query
    dominant = np.argmax(routing, axis=1)
    confidence = np.max(routing, axis=1)
    support = np.sum(routing > EPS, axis=1)
    dominant_grid = np.empty((len(y_values), len(x_values)), dtype=np.int64)
    confidence_grid = np.empty_like(dominant_grid, dtype=np.float64)
    support_grid = np.empty_like(dominant_grid, dtype=np.float64)
    dominant_grid[y_index, x_index] = dominant
    confidence_grid[y_index, x_index] = confidence
    support_grid[y_index, x_index] = support
    return x_values, y_values, dominant_grid, confidence_grid, support_grid


def _draw_query_map(axis: plt.Axes, panel: PanelData) -> None:
    x_values, y_values, dominant, confidence, support = _query_grid(panel)
    baseline = 1.0 / GROUP_COUNT
    strength = np.clip((confidence - baseline) / (1.0 - baseline), 0.0, 1.0)
    strength = 0.24 + 0.76 * strength
    rgb = np.ones((*dominant.shape, 3), dtype=np.float64)
    for group in range(GROUP_COUNT):
        selected = dominant == group
        rgb[selected] = (
            1.0 - strength[selected, None]
        ) + strength[selected, None] * PALETTE[group]
    dx = float(np.median(np.diff(x_values))) if len(x_values) > 1 else 1.0
    dy = float(np.median(np.diff(y_values))) if len(y_values) > 1 else 1.0
    axis.imshow(
        rgb,
        origin="lower",
        extent=(
            float(x_values[0] - dx / 2),
            float(x_values[-1] + dx / 2),
            float(y_values[0] - dy / 2),
            float(y_values[-1] + dy / 2),
        ),
        interpolation="nearest",
        aspect="equal",
    )
    levels = [value + 0.5 for value in np.unique(support)[:-1]]
    if levels:
        axis.contour(
            x_values,
            y_values,
            support,
            levels=levels,
            colors="#31343A",
            linewidths=0.75,
            linestyles="--",
            alpha=0.8,
        )
    active_indices = panel.active_module_indices
    active_incidence = panel.ordered_modules[active_indices]
    for position, (source_index, memberships) in enumerate(
        zip(active_indices, active_incidence, strict=True), start=1
    ):
        center = panel.module_centers[source_index]
        group = int(np.argmax(memberships))
        axis.scatter(
            [center[0]],
            [center[1]],
            s=135,
            marker="o",
            facecolors="white",
            edgecolors=[PALETTE[group]],
            linewidths=2.4,
            zorder=5,
        )
        axis.text(
            center[0],
            center[1],
            f"M{position}",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="#20242A",
            zorder=6,
        )
    axis.set_xlim(float(x_values[0] - dx / 2), float(x_values[-1] + dx / 2))
    axis.set_ylim(float(y_values[0] - dy / 2), float(y_values[-1] + dy / 2))
    axis.set_xticks([0, 3, 6, 9, 12])
    axis.set_yticks([0, 3, 6])
    axis.tick_params(labelsize=7, length=2.5, colors="#525A65")
    axis.set_xlabel("x", fontsize=8)
    axis.set_ylabel("y", fontsize=8)
    axis.set_title("Dominant query group and confidence", fontsize=9, pad=5)
    for spine in axis.spines.values():
        spine.set_color("#A8AFB8")
        spine.set_linewidth(0.7)


def _curve(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: Sequence[float] | str,
    width: float,
    alpha: float,
    bend: float = 0.0,
    zorder: int = 1,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-",
            connectionstyle=f"arc3,rad={bend}",
            linewidth=width,
            color=color,
            alpha=alpha,
            capstyle="round",
            zorder=zorder,
        )
    )


def _draw_hypergraph(axis: plt.Axes, panel: PanelData) -> None:
    module_rows = panel.ordered_modules[panel.active_module_indices]
    environment_rows = panel.ordered_environment
    query_rows = panel.ordered_query
    environment_mass = environment_rows.sum(axis=0)
    query_mass = query_rows.mean(axis=0)
    group_y = np.linspace(0.88, 0.12, GROUP_COUNT)
    module_y = np.linspace(0.82, 0.34, len(module_rows))
    source_x, group_x, query_x = 0.04, 0.52, 0.96
    environment_y = 0.10
    for group, y_value in enumerate(group_y):
        env_share = environment_mass[group] / max(environment_mass.sum(), EPS)
        if env_share > EPS:
            _curve(
                axis,
                (source_x + 0.025, environment_y),
                (group_x - 0.032, y_value),
                color=PALETTE[group],
                width=0.35 + 9.0 * env_share,
                alpha=0.28,
                bend=0.04 * (group - 2.5),
                zorder=1,
            )
        if query_mass[group] > EPS:
            _curve(
                axis,
                (group_x + 0.032, y_value),
                (query_x - 0.034, 0.50),
                color=PALETTE[group],
                width=0.35 + 9.0 * query_mass[group],
                alpha=0.58,
                bend=-0.035 * (group - 2.5),
                zorder=2,
            )
    for module_index, (y_value, memberships) in enumerate(
        zip(module_y, module_rows, strict=True), start=1
    ):
        for group, weight in enumerate(memberships):
            if weight <= EPS:
                continue
            _curve(
                axis,
                (source_x + 0.024, float(y_value)),
                (group_x - 0.032, float(group_y[group])),
                color=PALETTE[group],
                width=0.45 + 3.6 * float(weight),
                alpha=0.78,
                bend=0.025 * (group - 2.5),
                zorder=3,
            )
        axis.scatter(
            [source_x],
            [y_value],
            s=62,
            marker="s",
            facecolors="white",
            edgecolors="#333A43",
            linewidths=1.0,
            zorder=6,
        )
        axis.text(
            source_x - 0.038,
            y_value,
            f"M{module_index}",
            ha="right",
            va="center",
            fontsize=7,
            color="#252A31",
        )
    axis.scatter(
        [source_x],
        [environment_y],
        s=78,
        marker="D",
        facecolors="#F3F4F6",
        edgecolors="#333A43",
        linewidths=1.0,
        zorder=6,
    )
    axis.text(
        source_x - 0.038,
        environment_y,
        f"E\n{len(environment_rows)}",
        ha="right",
        va="center",
        fontsize=6.6,
        color="#252A31",
        linespacing=0.9,
    )
    for group, y_value in enumerate(group_y):
        size = 155 + 620 * float(query_mass[group])
        axis.scatter(
            [group_x],
            [y_value],
            s=size,
            marker="o",
            facecolors="white",
            edgecolors=[PALETTE[group]],
            linewidths=2.0,
            zorder=7,
        )
        axis.text(
            group_x,
            y_value,
            f"H{group + 1}",
            ha="center",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color="#20242A",
            zorder=8,
        )
    axis.scatter(
        [query_x],
        [0.50],
        s=105,
        marker="o",
        facecolors="#252A31",
        edgecolors="white",
        linewidths=0.8,
        zorder=7,
    )
    axis.text(
        query_x,
        0.50,
        "Q",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="white",
        zorder=8,
    )
    axis.text(
        query_x,
        0.43,
        f"{len(query_rows):,}\nqueries",
        ha="center",
        va="top",
        fontsize=6.5,
        color="#525A65",
        linespacing=0.9,
    )
    axis.text(
        0.28,
        0.97,
        "source membership",
        ha="center",
        va="top",
        fontsize=6.8,
        color="#626A75",
    )
    axis.text(
        0.75,
        0.97,
        "query mass",
        ha="center",
        va="top",
        fontsize=6.8,
        color="#626A75",
    )
    axis.set_xlim(-0.12, 1.07)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()
    axis.set_title("Aggregated learned hypergraph", fontsize=9, pad=5)


def _group_rows(panel: PanelData) -> list[dict[str, Any]]:
    module_rows = panel.ordered_modules[panel.active_module_indices]
    environment_rows = panel.ordered_environment
    query_rows = panel.ordered_query
    output: list[dict[str, Any]] = []
    for rank, raw_group in enumerate(panel.group_order, start=1):
        local = rank - 1
        output.append(
            {
                "run": panel.run,
                "case_id": panel.case_id,
                "epoch": panel.epoch,
                "panel_group": f"H{rank}",
                "raw_group_index": int(raw_group),
                "module_membership_mass": float(module_rows[:, local].sum()),
                "module_positive_sources": int((module_rows[:, local] > EPS).sum()),
                "environment_membership_mass": float(
                    environment_rows[:, local].sum()
                ),
                "environment_positive_sources": int(
                    (environment_rows[:, local] > EPS).sum()
                ),
                "mean_query_mass": float(query_rows[:, local].mean()),
                "dominant_query_count": int(
                    (np.argmax(query_rows, axis=1) == local).sum()
                ),
            }
        )
    return output


def render_anchor_hypergraphs(
    routing_dir: Path,
    output_path: Path,
    summary_path: Path,
    group_csv_path: Path,
    *,
    runs: Sequence[str] = DEFAULT_RUNS,
    cases: Sequence[str] = DEFAULT_CASES,
) -> dict[str, Any]:
    metrics = _read_case_metrics(routing_dir / "case_metrics.csv")
    epochs = _read_epochs(routing_dir / "summary.json")
    panels = {
        (run, case_id): load_panel(
            routing_dir, run, case_id, metrics, epochs
        )
        for case_id in cases
        for run in runs
    }
    fig = plt.figure(figsize=(18.0, 10.6), facecolor="white")
    outer = fig.add_gridspec(
        len(cases),
        len(runs),
        left=0.045,
        right=0.985,
        top=0.84,
        bottom=0.105,
        wspace=0.11,
        hspace=0.26,
    )
    for row_index, case_id in enumerate(cases):
        for column_index, run in enumerate(runs):
            panel = panels[(run, case_id)]
            nested = outer[row_index, column_index].subgridspec(
                1, 2, width_ratios=(1.45, 1.0), wspace=0.08
            )
            map_axis = fig.add_subplot(nested[0, 0])
            graph_axis = fig.add_subplot(nested[0, 1])
            _draw_query_map(map_axis, panel)
            _draw_hypergraph(graph_axis, panel)
            support = panel.metrics.get(
                "query_support_degree_mean",
                float(np.mean(np.sum(panel.query_routing > EPS, axis=1))),
            )
            effective = panel.metrics.get("query_effective_groups_mean", float("nan"))
            environment_density = panel.metrics.get(
                "environment_unique_pair_density", float("nan")
            )
            title = f"Run {run} · Case {case_id} · epoch {panel.epoch}"
            subtitle = (
                f"mean query support {support:.2f}/6  ·  effective groups "
                f"{effective:.2f}  ·  environment logical support "
                f"{environment_density:.1%}"
            )
            x0 = map_axis.get_position().x0
            x1 = graph_axis.get_position().x1
            y1 = max(map_axis.get_position().y1, graph_axis.get_position().y1)
            fig.text(
                (x0 + x1) / 2,
                y1 + 0.037,
                title,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="#20242A",
            )
            fig.text(
                (x0 + x1) / 2,
                y1 + 0.016,
                subtitle,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#626A75",
            )
    fig.suptitle(
        "Mature learned hypergraphs: phase sharing sharpens query organization but not module support",
        x=0.515,
        y=0.975,
        fontsize=17,
        fontweight="bold",
        color="#1D2228",
    )
    fig.text(
        0.515,
        0.941,
        "Best-field checkpoints. H1–H6 are relabelled independently in each panel; colors compare structure, not physical group identity.",
        ha="center",
        va="center",
        fontsize=10,
        color="#5A626D",
    )
    legend = (
        "Query-map hue = dominant panel-local group; intensity = maximum routing weight; dashed boundaries = changes in positive support degree.  "
        "Hypergraph edge width = membership/query mass; E aggregates 192 environment sources for readability."
    )
    fig.text(
        0.515,
        0.044,
        legend,
        ha="center",
        va="center",
        fontsize=9,
        color="#4D5560",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, facecolor="white")
    plt.close(fig)

    group_rows = [
        row
        for case_id in cases
        for run in runs
        for row in _group_rows(panels[(run, case_id)])
    ]
    group_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with group_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    payload = {
        "schema_version": 1,
        "task": "run1406_run1407_anchor_hypergraph_visualization",
        "source": {
            "routing_directory": str(routing_dir.resolve()),
            "maps": [
                str((routing_dir / "maps" / f"{run}__{case_id}.npz").resolve())
                for case_id in cases
                for run in runs
            ],
            "case_metrics": str((routing_dir / "case_metrics.csv").resolve()),
        },
        "cases": list(cases),
        "runs": list(runs),
        "group_relabeling": (
            "panel-local only: module-bearing groups sorted by module membership "
            "mass, then remaining groups sorted by mean query mass"
        ),
        "environment_rendering": (
            "192 environment sources are visually aggregated; edge widths retain "
            "their summed membership mass"
        ),
        "artifacts": {
            "figure": str(output_path.resolve()),
            "group_summary_csv": str(group_csv_path.resolve()),
        },
        "panels": [
            {
                "run": panel.run,
                "case_id": panel.case_id,
                "epoch": panel.epoch,
                "module_count": len(panel.active_module_indices),
                "group_order_raw_indices": panel.group_order.tolist(),
                "query_support_degree_mean": float(
                    panel.metrics.get(
                        "query_support_degree_mean",
                        np.mean(np.sum(panel.query_routing > EPS, axis=1)),
                    )
                ),
                "query_effective_groups_mean": float(
                    panel.metrics.get("query_effective_groups_mean", float("nan"))
                ),
                "query_entropy_norm_mean": float(
                    panel.metrics.get("query_entropy_norm_mean", float("nan"))
                ),
                "module_occupied_group_count": float(
                    panel.metrics.get("module_occupied_group_count", float("nan"))
                ),
                "environment_occupied_group_count": float(
                    panel.metrics.get(
                        "environment_occupied_group_count", float("nan")
                    )
                ),
                "module_unique_pair_density": float(
                    panel.metrics.get("module_unique_pair_density", float("nan"))
                ),
                "environment_unique_pair_density": float(
                    panel.metrics.get(
                        "environment_unique_pair_density", float("nan")
                    )
                ),
            }
            for case_id in cases
            for run in runs
            for panel in (panels[(run, case_id)],)
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-dir", type=Path, default=DEFAULT_ROUTING_DIR)
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--case-id", action="append", dest="cases")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--group-csv-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    routing_dir = args.routing_dir.expanduser().resolve()
    output = args.output or routing_dir / "figures/anchor_hypergraph_1406_1407_cases0273_0653.png"
    summary = args.summary_output or routing_dir / "anchor_hypergraph_summary.json"
    group_csv = args.group_csv_output or routing_dir / "anchor_hypergraph_group_summary.csv"
    payload = render_anchor_hypergraphs(
        routing_dir,
        output.expanduser().resolve(),
        summary.expanduser().resolve(),
        group_csv.expanduser().resolve(),
        runs=tuple(args.runs or DEFAULT_RUNS),
        cases=tuple(args.cases or DEFAULT_CASES),
    )
    print(json.dumps(payload["artifacts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
