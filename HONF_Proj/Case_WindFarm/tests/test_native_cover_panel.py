from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from windfarm.workflows.native_cover_panel import (
    TrainingLayout,
    freeze_organizer_layout_split,
    make_disjoint_native_probes,
    select_training_layouts,
    write_organizer_split_lock,
)


class _PanelView:
    def __init__(self, layout_count: int = 20) -> None:
        self.n_cases = layout_count * 3
        layout_ids = np.repeat(np.arange(layout_count, dtype=np.int64), 3)
        self.metadata = {
            "layout_index": layout_ids,
            "n_turbines": np.repeat(np.arange(5, 5 + layout_count, dtype=np.int64), 3),
            "turbine_xy_D": np.zeros((self.n_cases, 30, 2), dtype=np.float64),
        }
        for layout in range(layout_count):
            count = 5 + layout
            x = np.linspace(0.0, 1.0 + layout * 0.25, count)
            y = np.sin(np.arange(count) * (0.1 + layout * 0.01)) * (1.0 + layout * 0.1)
            self.metadata["turbine_xy_D"][layout * 3 : layout * 3 + 3, :count, 0] = x
            self.metadata["turbine_xy_D"][layout * 3 : layout * 3 + 3, :count, 1] = y


def _native_case() -> SimpleNamespace:
    diameter = 80.0
    x = np.linspace(-10.0, 20.0, 61) * diameter
    y = np.linspace(-8.0, 8.0, 33) * diameter
    z = np.linspace(0.0, 6.25, 51) * diameter
    nx, ny, nz = len(x), len(y), len(z)
    run = SimpleNamespace(
        x_m=x,
        y_m=y,
        z_m=z,
        nx=nx,
        ny=ny,
        nz=nz,
        cell_count=nx * ny * nz,
        U=np.zeros((nx * ny * nz, 3), dtype=np.float32),
    )
    # Values are irrelevant to sampling but finite physical placeholders make
    # the probe object suitable for downstream metric code in later tests.
    run.U[:, 0] = 9.0
    hubs = np.asarray([[-2.0, 0.0, 0.875], [0.0, 0.5, 0.875], [2.0, -0.5, 0.875]])
    return SimpleNamespace(
        index=7,
        layout_index=4,
        run=run,
        module_centers=hubs,
        diameter_m=diameter,
        hub_height_m=70.0,
        support=SimpleNamespace(
            lower_D=np.asarray([-10.0, -8.0, 0.0]),
            upper_D=np.asarray([20.0, 8.0, 6.25]),
        ),
    )


def test_training_layout_selection_keeps_all_direction_rows_and_spans_geometry() -> None:
    view = _PanelView()
    train_rows = np.arange(15 * 3, dtype=np.int64)
    selected = select_training_layouts(view, train_rows, count=12)

    assert len(selected) == 12
    assert len({item.layout_index for item in selected}) == 12
    assert all(item.rows == tuple(range(item.layout_index * 3, item.layout_index * 3 + 3)) for item in selected)
    assert all(set(item.rows).issubset(set(train_rows.tolist())) for item in selected)
    assert len({item.turbine_count for item in selected}) > 4
    assert len({item.feature_vector[3] for item in selected}) > 4


def test_receiver_local_split_is_geometry_frozen_and_keeps_same_m_layouts_across_partitions(
    tmp_path,
) -> None:
    turbine_counts = {
        4: 14,
        24: 6,
        70: 27,
        81: 29,
        88: 7,
        103: 6,
        107: 13,
        115: 18,
        119: 8,
        174: 29,
        177: 30,
        196: 15,
    }
    layouts = tuple(
        TrainingLayout(
            layout_index=layout,
            rows=(3 * layout, 3 * layout + 1, 3 * layout + 2),
            turbine_count=count,
            feature_vector=(float(count), float(layout), 1.0, float(count) / max(layout, 1)),
        )
        for layout, count in turbine_counts.items()
    )

    split = freeze_organizer_layout_split(layouts)
    repeat = freeze_organizer_layout_split(tuple(reversed(layouts)))

    assert split["training_layout_indices"] == [4, 24, 70, 88, 115, 119, 174, 177]
    assert split["development_layout_indices"] == [81, 103, 107, 196]
    assert len(split["training_rows_direction_order"]) == 24
    assert len(split["development_rows_direction_order"]) == 12
    assert split["same_turbine_count_cross_split"]["6"] == {
        "training_layouts": [24],
        "development_layouts": [103],
    }
    assert split["same_turbine_count_cross_split"]["29"] == {
        "training_layouts": [174],
        "development_layouts": [81],
    }
    assert split["split_sha256"] == repeat["split_sha256"]
    first_lock = write_organizer_split_lock(
        layouts,
        checkpoint_sha256="a" * 64,
        output_dir=tmp_path,
    )
    repeated_lock = write_organizer_split_lock(
        tuple(reversed(layouts)),
        checkpoint_sha256="a" * 64,
        output_dir=tmp_path,
    )
    assert first_lock["frozen_at_utc"] == repeated_lock["frozen_at_utc"]
    assert first_lock["artifact_sha256"] == repeated_lock["artifact_sha256"]


def test_receiver_local_split_rejects_a_changed_twelve_layout_panel() -> None:
    wrong = tuple(
        TrainingLayout(layout, (3 * layout, 3 * layout + 1, 3 * layout + 2), 6, (6.0, 1.0, 1.0, 6.0))
        for layout in range(12)
    )

    with pytest.raises(ValueError, match="frozen twelve layout identities"):
        freeze_organizer_layout_split(wrong)


def test_native_oracle_probes_are_disjoint_and_keep_protected_roles() -> None:
    case = _native_case()
    search, verify = make_disjoint_native_probes(case, query_count=32, seed=410)

    assert search.split == "search"
    assert verify.split == "verification"
    assert search.flat_indices.shape == verify.flat_indices.shape == (32,)
    assert np.intersect1d(search.flat_indices, verify.flat_indices).size == 0
    for probe in (search, verify):
        assert probe.coordinates_D.shape == probe.target_mps.shape == (32, 3)
        assert probe.quadrature_weights_D3.shape == (32,)
        assert np.all(probe.quadrature_weights_D3 > 0)
        assert all(bool(mask.any()) for mask in probe.roles.values())
        assert probe.roles["volume"].all()
        assert probe.roles["hub_slab"].sum() >= 6
        assert probe.roles["near_turbine"].sum() >= 1
        assert probe.roles["downstream_envelope"].sum() >= 1
