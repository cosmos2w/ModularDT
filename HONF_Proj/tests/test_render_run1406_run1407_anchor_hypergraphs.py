from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from tools.diagnostics.render_run1406_run1407_anchor_hypergraphs import (
    canonical_group_order,
    render_anchor_hypergraphs,
)


def test_canonical_group_order_puts_module_groups_first() -> None:
    modules = np.asarray(
        [
            [0.0, 0.2, 0.0, 0.0, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    environment = np.full((8, 6), 1.0 / 6.0)
    queries = np.asarray(
        [
            [0.05, 0.05, 0.10, 0.20, 0.40, 0.20],
            [0.05, 0.05, 0.20, 0.10, 0.40, 0.20],
        ]
    )
    order = canonical_group_order(modules, environment, queries)
    assert order[:2].tolist() == [4, 1]
    assert sorted(order.tolist()) == list(range(6))


def _write_fixture(root: Path) -> None:
    maps = root / "maps"
    maps.mkdir(parents=True)
    x_values = np.linspace(0.5, 5.5, 6)
    y_values = np.linspace(0.5, 3.5, 4)
    query_xy = np.asarray([(x, y) for y in y_values for x in x_values])
    module_present = np.asarray([1.0, 1.0, 1.0, 0.0])
    module_centers = np.asarray(
        [[1.5, 1.2], [3.0, 2.4], [4.8, 1.8], [0.0, 0.0]]
    )
    for run in ("1406", "1407"):
        for case_id in ("0273", "0653"):
            module_incidence = np.zeros((1, 4, 6), dtype=np.float64)
            module_incidence[0, 0, 4] = 1.0
            module_incidence[0, 1, 5] = 1.0
            module_incidence[0, 2, 4] = 0.3
            module_incidence[0, 2, 5] = 0.7
            environment_incidence = np.zeros((1, 12, 6), dtype=np.float64)
            for index in range(12):
                environment_incidence[0, index, index % 4] = 1.0
            query = np.zeros((1, len(query_xy), 6), dtype=np.float64)
            if run == "1406":
                query[:] = 0.04
                for index, (x, _y) in enumerate(query_xy):
                    query[0, index, 4 if x < 3.0 else 5] = 0.55
                query /= query.sum(axis=-1, keepdims=True)
            else:
                for index, (x, _y) in enumerate(query_xy):
                    query[0, index, 4 if x < 3.0 else 5] = 0.75
                    query[0, index, 2] = 0.25
            np.savez_compressed(
                maps / f"{run}__{case_id}.npz",
                query_xy=query_xy,
                module_present=module_present,
                module_centers=module_centers,
                group_control_query_routing=query,
                group_control_module_incidence=module_incidence,
                group_control_environment_incidence=environment_incidence,
            )
    fields = [
        "label",
        "case_id",
        "query_support_degree_mean",
        "query_effective_groups_mean",
        "query_entropy_norm_mean",
        "module_occupied_group_count",
        "environment_occupied_group_count",
        "module_unique_pair_density",
        "environment_unique_pair_density",
    ]
    with (root / "case_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in ("1406", "1407"):
            for case_id in ("0273", "0653"):
                writer.writerow(
                    {
                        "label": run,
                        "case_id": case_id,
                        "query_support_degree_mean": 6.0 if run == "1406" else 2.0,
                        "query_effective_groups_mean": 4.2 if run == "1406" else 1.8,
                        "query_entropy_norm_mean": 0.8 if run == "1406" else 0.4,
                        "module_occupied_group_count": 2.0,
                        "environment_occupied_group_count": 4.0,
                        "module_unique_pair_density": 1.0,
                        "environment_unique_pair_density": 1.0 if run == "1406" else 0.7,
                    }
                )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "1406": {"epoch": 4797},
                    "1407": {"epoch": 4782},
                }
            }
        )
    )


def test_render_anchor_hypergraphs_writes_reviewable_artifacts(tmp_path: Path) -> None:
    routing_dir = tmp_path / "routing"
    _write_fixture(routing_dir)
    figure = routing_dir / "figures" / "anchors.png"
    summary = routing_dir / "anchor_summary.json"
    group_csv = routing_dir / "anchor_groups.csv"
    payload = render_anchor_hypergraphs(
        routing_dir,
        figure,
        summary,
        group_csv,
    )
    assert figure.is_file() and figure.stat().st_size > 10_000
    image = mpimg.imread(figure)
    assert image.shape[0] > 1_000
    assert image.shape[1] > image.shape[0]
    assert len(payload["panels"]) == 4
    assert payload["panels"][0]["epoch"] == 4797
    assert summary.is_file()
    with group_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert {row["panel_group"] for row in rows} == {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
    }
