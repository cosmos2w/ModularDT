from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from windfarm.comparative_visualization import (
    _latent_routing_summary,
    _plane_prediction,
    geometry_record,
    render_comparison_plane,
    render_latent_organization,
    select_diverse_validation_rows,
)


class _FakeView:
    def __init__(self) -> None:
        self.cases = {}
        for row, layout, direction, count, extent in (
            (0, 0, 270.0, 4, (10.0, 4.0, 2.0)),
            (1, 0, 285.0, 4, (11.0, 4.0, 2.0)),
            (2, 0, 300.0, 4, (10.0, 5.0, 2.0)),
            (3, 1, 270.0, 7, (20.0, 5.0, 2.0)),
            (4, 1, 285.0, 7, (20.0, 5.0, 2.0)),
            (5, 2, 270.0, 12, (30.0, 8.0, 2.0)),
            (6, 2, 285.0, 12, (30.0, 8.0, 2.0)),
        ):
            centers = np.column_stack(
                (
                    np.arange(count, dtype=np.float32),
                    np.zeros(count, dtype=np.float32),
                    np.full(count, 0.875, dtype=np.float32),
                )
            )
            self.cases[row] = SimpleNamespace(
                index=row,
                case=f"case_{row:02d}",
                layout_index=layout,
                wind_direction_deg=direction,
                n_turbines=count,
                module_centers=centers,
                module_present=np.ones(count, dtype=np.float32),
                support=SimpleNamespace(extent_D=np.asarray(extent), volume_D3=float(np.prod(extent))),
                diameter_m=80.0,
                run=SimpleNamespace(),
            )

    def run(self, row: int):
        return self.cases[int(row)]


def _plane() -> dict[str, object]:
    coords = np.asarray(
        [[-1.0, -0.5, 0.875], [0.0, -0.5, 0.875], [1.0, -0.5, 0.875],
         [-1.0, 0.5, 0.875], [0.0, 0.5, 0.875], [1.0, 0.5, 0.875]],
        dtype=np.float32,
    )
    target = np.asarray(
        [[[8.0, -0.2, 0.1], [9.0, 0.0, 0.2], [10.0, 0.2, 0.3]],
         [[8.5, -0.1, 0.0], [9.5, 0.1, 0.1], [10.5, 0.3, 0.2]]],
        dtype=np.float32,
    )
    return {
        "fixed_axis": "z",
        "actual_m": 70.0,
        "horizontal_axis": "x",
        "vertical_axis": "y",
        "horizontal_D": np.asarray([-1.0, 0.0, 1.0]),
        "vertical_D": np.asarray([-0.5, 0.5]),
        "coords_D": coords,
        "target": target,
    }


def test_geometry_selection_is_one_row_per_layout_and_does_not_need_fields() -> None:
    view = _FakeView()
    selected = select_diverse_validation_rows(view, range(7), count=3)
    assert len(selected) == 3
    assert len({view.run(row).layout_index for row in selected}) == 3
    assert selected == select_diverse_validation_rows(view, range(7), count=3)
    record = geometry_record(view, selected[0])
    assert record.n_turbines == view.run(selected[0]).n_turbines


def test_plane_prediction_uses_bounded_chunks() -> None:
    calls: list[int] = []

    class FakeModel:
        def predict_physical(self, prepared, coordinates, features, *, receiver_chunk_size):
            del prepared, features, receiver_chunk_size
            calls.append(int(coordinates.shape[1]))
            return coordinates.new_zeros((1, coordinates.shape[1], 3))

    case = SimpleNamespace(
        geometry_for_queries=lambda coordinates: {
            "query_xy": np.asarray(coordinates, dtype=np.float32),
            "query_features": np.zeros((len(coordinates), 7), dtype=np.float32),
        }
    )
    plane = _plane()
    prediction = _plane_prediction(FakeModel(), case, object(), plane, torch.device("cpu"), 2)
    assert prediction.shape == (2, 3, 3)
    assert calls == [2, 2, 2]


def test_comparison_and_latent_figures_are_explicitly_labelled(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    plane = _plane()
    target = np.asarray(plane["target"])
    predictions = {"classic": target + 0.1, "dense": target - 0.2}
    comparison = render_comparison_plane(
        plane,
        predictions,
        np.asarray([[0.0, 0.0, 0.875]], dtype=np.float32),
        "comparison",
        tmp_path / "comparison.png",
    )
    assert comparison.is_file() and comparison.stat().st_size > 0

    view = _FakeView()
    case = view.run(0)
    encoded = {
        "hyper_source_coords": torch.zeros((1, 2, 3)),
        "hyper_region_coords": torch.ones((1, 2, 3)),
        "A_mh": torch.ones((1, case.n_turbines, 2)),
    }

    class FakeClassic:
        architecture = "legacy_honf"

        def decode(self, prepared, coordinates, features, *, receiver_chunk_size, return_routing_maps):
            del prepared, features, receiver_chunk_size, return_routing_maps
            count = int(coordinates.shape[1])
            return {
                "dominant_hyperedge": torch.zeros((1, count), dtype=torch.long),
                "hyper_attention_entropy_map": torch.zeros((1, count)),
            }

    case_value = case

    class CaseForQueries:
        case = case_value.case
        module_centers = case_value.module_centers
        support = case_value.support

        @staticmethod
        def geometry_for_queries(coords):
            return {"query_xy": np.asarray(coords, dtype=np.float32), "query_features": np.zeros((len(coords), 7), dtype=np.float32)}

    summary = _latent_routing_summary(
        FakeClassic(),
        CaseForQueries(),
        SimpleNamespace(encoded=encoded),
        plane,
        torch.device("cpu"),
        4,
    )
    assert summary["available"] is True
    assert "not physical wake topology" in summary["label"]
    latent = render_latent_organization(summary, CaseForQueries(), tmp_path / "latent.png")
    assert latent is not None and latent.is_file() and latent.stat().st_size > 0
