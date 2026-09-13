from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from windfarm.io import WindFarmDataset
from windfarm.profile import profile_dataset, write_profile_outputs
from windfarm.splits import make_group_split
from windfarm.visualize import default_figure_path, render_case_showcase, select_representative_indices


def _write_array(root: Path, name: str, value: np.ndarray) -> None:
    np.save(root / f"{name}.npy", value)


def make_fixture(tmp_path: Path, runs: int = 6) -> Path:
    root = tmp_path / "wind_farm" / "family_volume"
    root.mkdir(parents=True)
    shapes = np.asarray([(5 + i % 2, 4, 3) for i in range(runs)], dtype=np.int32)
    cell_counts = np.prod(shapes.astype(np.int64), axis=1)
    cell_offsets = np.concatenate(([0], np.cumsum(cell_counts))).astype(np.int64)
    x_lengths = shapes[:, 0]
    y_lengths = shapes[:, 1]
    z_lengths = shapes[:, 2]
    x_offsets = np.concatenate(([0], np.cumsum(x_lengths))).astype(np.int64)
    y_offsets = np.concatenate(([0], np.cumsum(y_lengths))).astype(np.int64)
    z_offsets = np.concatenate(([0], np.cumsum(z_lengths))).astype(np.int64)

    x_values = np.concatenate([np.linspace(-2, 2, n, dtype=np.float32) for n in x_lengths])
    y_values = np.concatenate([np.linspace(-1.5, 1.5, n, dtype=np.float32) for n in y_lengths])
    z_values = np.concatenate([np.linspace(5, 95, n, dtype=np.float32) for n in z_lengths])
    total = int(cell_offsets[-1])
    u = np.zeros((total, 3), dtype=np.float32)
    u[:, 0] = np.linspace(7, 10, total, dtype=np.float32)
    u[:, 1] = 0.1
    u[:, 2] = 0.02
    _write_array(root, "U", u)
    _write_array(root, "p", np.linspace(-1, 1, total, dtype=np.float32))
    _write_array(root, "k", np.full(total, 0.2, dtype=np.float32))
    _write_array(root, "epsilon", np.full(total, 0.01, dtype=np.float32))
    _write_array(root, "case", np.asarray([f"gen_{i:04d}_wd{270 + 15 * (i % 3)}" for i in range(runs)]))
    _write_array(root, "layout_index", np.asarray([i // 2 for i in range(runs)], dtype=np.int16))
    _write_array(root, "wd_deg", np.asarray([270 + 15 * (i % 3) for i in range(runs)], dtype=np.float32))
    _write_array(root, "source_time", np.full(runs, 100, dtype=np.float32))
    _write_array(root, "completed", np.ones(runs, dtype=np.uint8))
    _write_array(root, "run_shape", shapes)
    _write_array(root, "run_cell_offsets", cell_offsets)
    _write_array(root, "run_x_offsets", x_offsets)
    _write_array(root, "run_y_offsets", y_offsets)
    _write_array(root, "run_z_offsets", z_offsets)
    _write_array(root, "x_cell_m", x_values)
    _write_array(root, "y_cell_m", y_values)
    _write_array(root, "z_cell_m", z_values)
    return root.parent


def test_dataset_uses_mmap_and_run_views(tmp_path: Path) -> None:
    dataset = WindFarmDataset(make_fixture(tmp_path))
    assert dataset.n_runs == 6
    assert isinstance(dataset.array("U"), np.memmap)
    run = dataset.run(1)
    assert run.U_structured.shape == (3, 4, 6, 3)
    assert run.p_structured.shape == (3, 4, 6)
    assert run.cell_count == 72


def test_optional_npz_fallback_reads_metadata_only(tmp_path: Path) -> None:
    root = make_fixture(tmp_path)
    np.savez(
        root / "family_tensor.npz",
        case=np.asarray(["compact-case"]),
        layout_index=np.asarray([7], dtype=np.int16),
        wd_deg=np.asarray([285], dtype=np.float32),
        D_m=np.asarray(80.0, dtype=np.float32),
        hub_height_m=np.asarray(70.0, dtype=np.float32),
        U_hub=np.ones((1, 2, 2, 2), dtype=np.float32),
    )
    dataset = WindFarmDataset(root)
    metadata = dataset.compact_metadata(allow_npz_fallback=True)
    assert metadata["case"].tolist() == ["compact-case"]
    assert metadata["layout_index"].tolist() == [7]
    assert float(metadata["D_m"]) == 80.0
    assert "U_hub" not in metadata
    geometry_only = dataset.compact_metadata(
        allow_npz_fallback=True,
        names=("case", "layout_index", "wd_deg", "D_m"),
    )
    assert set(geometry_only) == {"case", "layout_index", "wd_deg", "D_m"}
    assert "wake_loss_pct" not in geometry_only


def test_profile_has_row_aware_shapes_and_three_direction_groups(tmp_path: Path) -> None:
    dataset = WindFarmDataset(make_fixture(tmp_path))
    profile = profile_dataset(dataset, sample_points_per_run=5)
    assert profile["validation"]["ok"]
    assert profile["validation"]["warnings"] == []
    shape_values = {tuple(item["value"]): item["count"] for item in profile["categories"]["shape_nxyz"]}
    assert shape_values[(5, 4, 3)] == 3
    assert shape_values[(6, 4, 3)] == 3
    assert {item["value"]: item["count"] for item in profile["categories"]["layout_row_counts"]} == {2: 3}
    assert profile["field_statistics"]["U"]["count"] == 6 * 5 * 3


def test_profile_accepts_three_direction_rows_per_layout(tmp_path: Path) -> None:
    root = make_fixture(tmp_path)
    np.save(root / "family_volume" / "layout_index.npy", np.repeat([0, 1], 3).astype(np.int16))
    dataset = WindFarmDataset(root)
    profile = profile_dataset(dataset, sample_points_per_run=2)
    assert profile["validation"]["warnings"] == []
    assert {item["value"]: item["count"] for item in profile["categories"]["layout_row_counts"]} == {3: 2}


def test_group_split_is_deterministic_and_disjoint() -> None:
    groups = np.repeat(np.arange(10, dtype=np.int16), 3)
    first = make_group_split(groups, seed=42)
    second = make_group_split(groups, seed=42)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.test, second.test)
    assert not (set(groups[first.train]) & set(groups[first.validation]))
    assert not (set(groups[first.train]) & set(groups[first.test]))
    assert not (set(groups[first.validation]) & set(groups[first.test]))
    assert np.array_equal(np.sort(np.concatenate((first.train, first.validation, first.test))), np.arange(30))


def test_profile_outputs_are_machine_readable(tmp_path: Path) -> None:
    dataset = WindFarmDataset(make_fixture(tmp_path))
    profile = profile_dataset(dataset, sample_points_per_run=3)
    paths = write_profile_outputs(profile, tmp_path / "diagnostics", group_values=dataset.array("layout_index"))
    assert {path.name for path in paths.values()} == {
        "windfarm_profile.json",
        "windfarm_cases.csv",
        "windfarm_splits.json",
        "windfarm_split_indices.npz",
    }
    loaded = json.loads(paths["profile"].read_text(encoding="utf-8"))
    assert loaded["dataset"]["runs"] == 6
    with np.load(paths["split_indices"]) as indices:
        assert set(indices.files) == {"train", "validation", "test"}


def test_visual_showcase_has_explicit_cuts_and_stable_name(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    dataset = WindFarmDataset(make_fixture(tmp_path))
    assert select_representative_indices(dataset) == [0, 3, 5]
    output = default_figure_path(tmp_path / "generated", dataset, 0)
    rendered = render_case_showcase(
        dataset,
        0,
        output,
        max_points=20,
        dpi=60,
        turbine_xy_m=np.asarray([[0.0, 0.0], [1.0, -1.0]]),
        color_limits=(6.0, 11.0),
    )
    assert rendered == output
    assert rendered.name == "windfarm_case_0000_gen_0000_wd270.png"
    assert rendered.stat().st_size > 0
