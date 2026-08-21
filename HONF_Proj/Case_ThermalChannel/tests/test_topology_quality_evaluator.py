from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from channelthermal.evaluation_tools.organizer_visualization import (
    render_channelthermal_organization_summary_matrices,
)
from channelthermal.workflows.evaluate_forward import extract_organization_arrays


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_diagnostic(name: str):
    path = PROJECT_ROOT / "diagnostics" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample() -> dict[str, object]:
    x, y = np.meshgrid(np.linspace(0.0, 4.0, 6), np.linspace(0.0, 2.0, 4))
    return {
        "x_grid": x.astype(np.float32),
        "y_grid": y.astype(np.float32),
        "steady_field": np.zeros((*x.shape, 5), dtype=np.float32),
        "structure": {
            "module_centers": np.asarray([[1.0, 0.8], [3.0, 1.2]], dtype=np.float32),
            "module_present": np.ones(2, dtype=np.float32),
            "heat_powers": np.asarray([1.0, -1.0], dtype=np.float32),
        },
    }


def test_extract_organization_arrays_preserves_effective_active_mask() -> None:
    sample = _sample()
    aux = {
        "A_mh": np.asarray([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0]], dtype=np.float32),
        "A_eh": np.asarray([[0.7, 0.3, 0.0], [0.2, 0.8, 0.0]], dtype=np.float32),
        "env_coords": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "hyper_strength": np.asarray([0.5, 0.5, 0.0], dtype=np.float32),
        "edge_active_mask": np.ones(3, dtype=np.float32),
        "effective_edge_mask": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
    }

    arrays = extract_organization_arrays(sample, aux)

    np.testing.assert_array_equal(
        arrays["active_hyperedge_mask"], np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    )


def test_environment_matrix_exports_physical_and_explicit_sorted_views(tmp_path) -> None:
    sample = _sample()
    env_coords = np.stack(
        np.meshgrid(np.linspace(0.0, 4.0, 4), np.linspace(0.0, 2.0, 3)), axis=-1
    ).reshape(-1, 2).astype(np.float32)
    A_eh = np.zeros((env_coords.shape[0], 2), dtype=np.float32)
    A_eh[:, 0] = np.linspace(0.05, 0.95, env_coords.shape[0])
    A_eh[:, 1] = 1.0 - A_eh[:, 0]
    arrays = extract_organization_arrays(
        sample,
        {
            "A_mh": np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32),
            "A_eh": A_eh,
            "env_coords": env_coords,
            "hyper_strength": np.ones(2, dtype=np.float32),
            "hyper_module_mass": np.asarray([0.5, 0.5], dtype=np.float32),
            "hyper_env_mass": np.asarray([0.5, 0.5], dtype=np.float32),
            "hyper_source_coords": np.asarray([[1.0, 0.8], [3.0, 1.2]], dtype=np.float32),
            "hyper_region_coords": np.asarray([[1.0, 1.0], [3.0, 1.0]], dtype=np.float32),
        },
    )

    render_channelthermal_organization_summary_matrices(
        tmp_path / "physical.png", sample, arrays, module_radius=0.4, sort_environment=False
    )
    render_channelthermal_organization_summary_matrices(
        tmp_path / "sorted.png", sample, arrays, module_radius=0.4, sort_environment=True
    )

    assert (tmp_path / "physical.png").is_file()
    assert (tmp_path / "sorted.png").is_file()


def test_topology_metrics_distinguish_rank_one_from_organized_assignments() -> None:
    evaluator = _load_diagnostic("evaluate_topology_quality")
    organized = np.asarray(
        [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]], dtype=np.float64
    )
    rank_one = np.tile(np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64), (3, 1))

    good = evaluator.assignment_metrics(organized, "test")
    collapsed = evaluator.assignment_metrics(rank_one, "test")

    assert good["test_effective_rank"] > 2.0
    assert collapsed["test_effective_rank"] == 1.0
    assert good["test_edge_column_cosine"] < collapsed["test_edge_column_cosine"]


def test_topology_evaluator_reads_source_and_config_provenance(tmp_path) -> None:
    evaluator = _load_diagnostic("evaluate_topology_quality")
    manifest = {
        "config_sha256": "config-hash",
        "source_state": {"commit": "source-sha"},
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluator.run_manifest_provenance(tmp_path / "best_model.pt")

    assert result == {"source_sha": "source-sha", "config_sha256": "config-hash"}


def test_retained_mass_distribution_reports_exact_min_p05_and_mean() -> None:
    evaluator = _load_diagnostic("evaluate_retained_mass_pruning")
    result = evaluator.distribution(np.asarray([0.5, 0.75, 1.0], dtype=np.float64))

    assert result["min"] == 0.5
    assert result["p05"] == 0.525
    assert result["mean"] == 0.75
