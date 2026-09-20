"""Unit-level reducers and artifact contracts for the Run-1408 evidence tool."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools" / "diagnostics"))

from run_run1408_evidence import (
    ARCHITECTURE,
    DEFAULT_CASE_IDS,
    _compact_optimizer_inventory,
    _measure_phase,
    build_parser,
    build_plan,
    execution_ledger,
    render_sampled_cells,
)


def _args(tmp_path: Path, *extra: str):
    return build_parser().parse_args(
        [
            "--output",
            str(tmp_path / "run1408.json"),
            *extra,
        ]
    )


def test_run1408_evidence_plan_keeps_fixed_sample_and_measurement_contract(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))

    assert plan["status"] == "plan_only"
    assert plan["architecture"] == ARCHITECTURE
    assert plan["scientific_contract"] == {
        "group_count": 6,
        "group_control_dim": 16,
        "samples_per_group": 4,
        "sample_slots_per_receiver": 24,
        "environment_only_replacement": True,
        "dense_mm_me_em_preparation": True,
        "phase_shared_p0_controller": True,
        "three_term_assembly": "Cg+CM+CE",
    }
    assert plan["protocol"]["case_ids"] == list(DEFAULT_CASE_IDS)
    assert plan["protocol"]["query_count"] == 8192
    assert plan["protocol"]["receiver_chunk_size"] == 2048
    assert plan["protocol"]["training_buckets"] == ["M1", "M12"]
    assert plan["protocol"]["predicted_port_condition"] is True
    assert plan["evidence_fields"] == [
        "sample_slots",
        "nonzero_sample_masses",
        "fine_content_dot_rows",
        "fine_geometry_rows",
        "interpolation_corner_loads",
        "unique_accessed_cells",
        "sampled_bank_actual_tensor_bytes",
        "checkpoint_recomputations",
    ]


def test_run1408_execution_ledger_counts_sites_not_coordinate_scalars() -> None:
    batch, queries, groups, sites = 1, 2, 6, 4
    coordinates = np.zeros((batch, queries, groups, sites, 2), dtype=np.float32)
    beta = np.full((batch, queries, groups, sites), 0.25, dtype=np.float32)
    masses = np.ones_like(beta)
    masses[0, 0, 0, 0] = 0.0
    corners = np.arange(batch * queries * groups * sites * 4, dtype=np.int64).reshape(
        batch, queries, groups, sites, 4
    )
    weights = np.full_like(corners, 0.25, dtype=np.float32)
    cells = np.arange(batch * queries * groups * sites, dtype=np.int64).reshape(batch, queries, groups, sites)
    ledger = execution_ledger(
        {
            "group_control_environment_sample_coordinates": torch.from_numpy(coordinates),
            "group_control_environment_sample_beta": torch.from_numpy(beta),
            "group_control_environment_sample_lambda": torch.from_numpy(masses),
            "group_control_environment_interpolation_corner_indices": torch.from_numpy(corners),
            "group_control_environment_interpolation_corner_weights": torch.from_numpy(weights),
            "group_control_environment_interpolation_cell_indices": torch.from_numpy(cells),
            "group_control_environment_sampled_bank_bytes": torch.tensor(1234.0),
            "group_control_environment_checkpoint_recomputations": torch.tensor(2.0),
        }
    )

    assert ledger["status"] == "actual_maps"
    assert ledger["sample_slots"] == batch * queries * groups * sites == 48
    assert ledger["nonzero_sample_masses"] == 47
    assert ledger["fine_content_geometry_rows"] == 48
    assert ledger["fine_content_dot_rows"] == 48 * 4
    assert ledger["fine_geometry_rows"] == 48
    assert ledger["interpolation_corner_loads"] == 48 * 4
    assert ledger["unique_accessed_cells"] == 48
    assert ledger["sampled_bank_actual_tensor_bytes"] == 1234
    assert ledger["checkpoint_recomputations"] == 2
    assert ledger["map_shapes"]["sample_coordinates"] == [1, 2, 6, 4, 2]


def test_run1408_execution_ledger_accepts_phase_prefixed_maps_and_explicit_counters() -> None:
    prefixed = {
        "initial_port_group_control_environment_sample_coordinates": np.zeros((1, 1, 6, 4, 2)),
        "initial_port_group_control_environment_sample_lambda": np.ones((1, 1, 6, 4)),
    }
    ledger = execution_ledger(prefixed, phase="P0")
    assert ledger["status"] == "actual_maps"
    assert ledger["sample_slots"] == 24

    counters = execution_ledger(
        {
            "group_control_environment_sample_slots": torch.tensor(24.0),
            "group_control_environment_nonzero_sample_masses": torch.tensor(19.0),
            "group_control_environment_fine_rows_forward": torch.tensor(24.0),
            "group_control_environment_content_dot_rows_forward": torch.tensor(96.0),
            "group_control_environment_geometry_rows_forward": torch.tensor(24.0),
            "group_control_environment_interpolation_corner_loads": torch.tensor(96.0),
            "group_control_environment_sampled_bank_bytes": torch.tensor(4096.0),
        }
    )
    assert counters["status"] == "explicit_counter"
    assert counters["sample_slots"] == 24
    assert counters["nonzero_sample_masses"] == 19
    assert counters["fine_content_dot_rows"] == 96
    assert counters["sampled_bank_actual_tensor_bytes"] == 4096
    assert counters["unique_accessed_cells"] is None


def test_run1408_measurement_keeps_cpu_timing_and_peak_schema() -> None:
    measurement = _measure_phase(lambda: torch.ones((4, 4)).sum(), torch.device("cpu"), warmups=1, repetitions=2)

    assert measurement["status"] == "complete"
    assert len(measurement["samples"]) == 2
    assert measurement["median_seconds"] is not None
    assert "peak_allocated_bytes" in measurement["samples"][0]
    assert "peak_reserved_bytes" in measurement["samples"][0]


def test_run1408_optimizer_evidence_keeps_counts_without_metadata_or_name_dump() -> None:
    compact = _compact_optimizer_inventory(
        {
            "mode": "split",
            "weight_decay": 1.0e-5,
            "unwanted_metadata": "omit",
            "groups": [
                {
                    "name": "prediction",
                    "learning_rate": 2.0e-4,
                    "parameter_tensor_count": 3,
                    "trainable_scalar_count": 12,
                    "scalar_count_complete": True,
                    "parameter_names": ["large", "name", "dump"],
                    "other_metadata": "omit",
                }
            ],
        }
    )

    assert compact == {
        "mode": "split",
        "weight_decay": 1.0e-5,
        "groups": [
            {
                "name": "prediction",
                "learning_rate": 2.0e-4,
                "parameter_tensor_count": 3,
                "trainable_scalar_count": 12,
                "scalar_count_complete": True,
            }
        ],
    }


def test_run1408_accessed_cell_plot_consumes_actual_map_arrays(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    queries, groups, sites, grid_tokens = 2, 6, 4, 9
    map_path = tmp_path / "actual_map.npz"
    sample_coordinates = np.zeros((queries, groups, sites, 2), dtype=np.float32)
    for query in range(queries):
        sample_coordinates[query, :, :, 0] = 0.2 + 0.1 * query
        sample_coordinates[query, :, :, 1] = 0.4
    corner_indices = np.tile(np.array([0, 1, 3, 4], dtype=np.int64), (queries, groups, sites, 1))
    np.savez_compressed(
        map_path,
        query_xy=np.array([[0.4, 0.4], [0.6, 0.4]], dtype=np.float32),
        env_coords=np.array(
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0, 0.5], [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )[:grid_tokens],
        sample_coordinates=sample_coordinates,
        sample_beta=np.full((queries, groups, sites), 0.25, dtype=np.float32),
        corner_indices=corner_indices,
        corner_weights=np.full((queries, groups, sites, 4), 0.25, dtype=np.float32),
        module_centers=np.array([[0.25, 0.25]], dtype=np.float32),
        module_present=np.array([1.0], dtype=np.float32),
    )
    output = render_sampled_cells(map_path, tmp_path / "accessed_cells.png", selected_queries=(0, 1))

    assert output["status"] == "complete"
    assert Path(output["path"]).is_file()
    assert output["selected_queries"] == [0, 1]
