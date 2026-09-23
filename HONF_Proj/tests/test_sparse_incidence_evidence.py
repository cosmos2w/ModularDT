from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = PROJECT / "tools" / "diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

from sparse_incidence_evidence import (
    canonicalize_case,
    render_population_figures,
    summarize_population,
)


def _payload() -> dict[str, np.ndarray | dict[str, object]]:
    module = np.zeros((3, 12), dtype=np.float32)
    environment = np.zeros((4, 12), dtype=np.float32)
    query = np.zeros((5, 12), dtype=np.float32)
    module[np.arange(3), np.arange(3)] = 1.0
    environment[np.arange(4), np.arange(4)] = 1.0
    query[:, 0] = 0.6
    query[:, 1] = 0.4
    module_measure = np.full(3, 1.0 / 3.0, dtype=np.float32)
    environment_measure = np.full(4, 0.25, dtype=np.float32)
    module_mass = (module_measure[:, None] * module).sum(axis=0)
    environment_mass = (environment_measure[:, None] * environment).sum(axis=0)
    pi = 0.5 * (module_mass + environment_mass)
    return {
        "module_coords": np.asarray([[[0, 0], [1, 0], [2, 0]]], dtype=np.float32),
        "environment_coords": np.asarray([[[0, 1], [1, 1], [2, 1], [3, 1]]], dtype=np.float32),
        "query_xy": np.asarray([[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]], dtype=np.float32),
        "module_present": np.ones((1, 3), dtype=np.float32),
        "group_control_module_incidence": module[None, ...],
        "group_control_environment_incidence": environment[None, ...],
        "group_control_module_measure": module_measure[None, ...],
        "group_control_environment_measure": environment_measure[None, ...],
        "sparse_incidence_query_routing": query[None, ...],
        "sparse_incidence_phase_occupied": ((module_mass + environment_mass) > 0)[None, ...],
        "sparse_incidence_pi": pi[None, ...],
        "sparse_incidence_kappa": np.asarray([1.0 / np.sum(pi * pi)], dtype=np.float32),
        "occupancy_group_phase_ledger": {
            "P2": {
                "module": {"actual_rows": 60, "padded_rows": 0},
                "environment": {"actual_rows": 80, "padded_rows": 0},
            }
        },
    }


def test_canonical_record_separates_capacity_occupancy_and_kq() -> None:
    record = canonicalize_case(_payload(), query_count=5, kmax=12)
    case = record["case"]
    assert case["registered_capacity"] == 12
    assert case["occupied_source_groups"] == 4
    assert case["query_degree_histogram"] == {"2": 5}
    assert case["query_degree_mean"] == 2.0
    assert case["kappa_formula_max_abs"] < 1.0e-5
    assert "kplan" not in case


def test_population_summary_never_introduces_kplan() -> None:
    case = canonicalize_case(_payload(), query_count=5, kmax=12)["case"]
    summary = summarize_population([{"case_id": "synthetic", **case}], expected_cases=1)
    assert summary["registered_capacity"] == 12
    assert summary["kq_histogram"] == {"2": 5}
    assert summary["continuation_gate"]["query_support_nontrivial"]
    assert "kplan_histogram" not in summary


def test_population_figures_include_case_spread_and_representative_overview(tmp_path: Path) -> None:
    first = canonicalize_case(_payload(), query_count=5, kmax=12)
    second = canonicalize_case(_payload(), query_count=5, kmax=12)
    rows = [
        {"case_id": "0273", **first["case"]},
        {"case_id": "0653", **second["case"]},
    ]
    array_dir = tmp_path / "arrays"
    figure_dir = tmp_path / "figures"
    from sparse_incidence_evidence import save_case_arrays

    save_case_arrays(array_dir / "0273.npz", first["maps"])
    save_case_arrays(array_dir / "0653.npz", second["maps"])
    paths = render_population_figures(rows, figure_dir)
    assert "kq_case_spread" in paths
    assert "kq_case_coordinate_regions" in paths
    assert "representative_hypergraph_organization" in paths
    for key in (
        "kq_case_spread",
        "kq_case_coordinate_regions",
        "representative_hypergraph_organization",
    ):
        output = Path(paths[key])
        assert output.is_file()
        assert output.stat().st_size > 10_000
