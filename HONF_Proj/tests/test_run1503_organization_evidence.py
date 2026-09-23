"""Focused CPU contracts for the Run-1503 organization evidence adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = PROJECT / "tools" / "diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

import sparse_incidence_evidence
from run_run1503_organization_evidence import (
    DEFAULT_CASE_IDS,
    _prepared_payload,
    canonicalize_candidate_case,
    render_candidate_figures,
    summarize_candidate_population,
)


def _payload() -> dict[str, object]:
    source = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.25, 0.75, 0.0],
            [0.0, 0.0, 1.0],
            [0.50, 0.0, 0.50],
            [0.0, 0.40, 0.60],
        ],
        dtype=np.float64,
    )
    assignment = np.pad(source, ((0, 0), (0, 9)))
    query = np.pad(
        np.asarray(
            [
                [0.7, 0.3, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.4, 0.6],
                [0.5, 0.0, 0.5],
            ],
            dtype=np.float64,
        ),
        ((0, 0), (0, 9)),
    )
    measure = np.asarray([0.10, 0.20, 0.30, 0.20, 0.20], dtype=np.float64)
    mass = measure @ assignment
    pi = 0.5 * (mass + mass)
    p = np.zeros_like(query)
    p[:, :3] = 1.0 / 3.0
    coordinates = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [8.0, 0.0], [0.0, 2.0], [8.0, 2.0]],
        dtype=np.float64,
    )
    return {
        "module_coords": coordinates,
        "environment_coords": coordinates,
        "query_xy": np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        "module_present": np.ones(5, dtype=bool),
        "module_assignment": assignment,
        "environment_assignment": assignment,
        "module_measure": measure,
        "environment_measure": measure,
        "active_mask": np.r_[np.ones(3, dtype=bool), np.zeros(9, dtype=bool)],
        "pi": pi,
        "kappa": 1.0 / np.sum(pi * pi),
        "query_assignment": query,
        "coarse_assignment": p,
        "opening_blend": query,
        "fine_work_ratio": 0.95,
        "fine_rows": 19.0,
        "full_rectangle_rows": 20.0,
        "phase_ledger": {
            "P2": {
                "environment": {
                    "support_pairs": 11.0,
                    "executed_rows": 19.0,
                    "padded_rows": 0.0,
                    "recomputed_rows": 0.0,
                }
            }
        },
    }


def test_candidate_record_keeps_distributions_work_and_empty_counts() -> None:
    record = canonicalize_candidate_case(
        _payload(),
        case_id="0273",
        module_count=3,
        query_count=4,
    )
    row = record["case"]
    assert row["case_id"] == "0273"
    assert row["registered_capacity"] == 12
    assert row["empty_group_count"] == 9
    assert row["empty_query_count"] == 0
    assert row["empty_source_count"] == 0
    assert row["fine_work_ratio"] == 0.95
    assert row["p_distribution"]["count"] == 48
    assert row["alpha_distribution"]["positive_fraction"] == 7.0 / 48.0
    assert record["maps"]["p_assignment"].shape == (4, 12)
    assert record["maps"]["opening_blend"].shape == (4, 12)


def test_prepared_payload_adapts_live_run1503_control_aliases() -> None:
    raw = _payload()
    controls = SimpleNamespace(
        module_membership=np.asarray(raw["module_assignment"])[None, ...],
        environment_membership=np.asarray(raw["environment_assignment"])[None, ...],
        module_measure=np.asarray(raw["module_measure"])[None, ...],
        environment_measure=np.asarray(raw["environment_measure"])[None, ...],
        module_mass=np.asarray(raw["module_assignment"])[None, ...].sum(axis=1),
        environment_mass=np.asarray(raw["environment_assignment"])[None, ...].sum(axis=1),
        phase_occupied=np.asarray(raw["active_mask"])[None, ...],
        pi=np.asarray(raw["pi"])[None, ...],
        kappa=np.asarray([raw["kappa"]]),
        module_centres=np.zeros((1, 12, 2)),
        environment_centres=np.zeros((1, 12, 2)),
        joint_centres=np.zeros((1, 12, 2)),
    )
    encoded = SimpleNamespace(
        env_coords=np.asarray(raw["environment_coords"])[None, ...],
        env_weights=np.asarray(raw["environment_measure"])[None, ...],
    )
    prepared = SimpleNamespace(encoded=encoded, backend_state={"group_control_state": controls})
    prediction = {
        "interaction_aux": {
            "group_control_adaptive_environment_p": np.asarray(raw["coarse_assignment"]),
            "group_control_adaptive_environment_alpha": np.asarray(raw["query_assignment"]),
            "group_control_adaptive_opening_blend": np.asarray(raw["opening_blend"]),
        },
        "_prepared_state": SimpleNamespace(prepared=prepared),
    }
    sample = {
        "structure": {
            "module_centers": np.asarray(raw["module_coords"]),
            "module_present": np.asarray(raw["module_present"]),
        },
        "query_source_count": 4,
        "query_selection": "full_original_grid",
    }
    payload = _prepared_payload(sample, prediction, np.asarray(raw["query_xy"]))
    assert payload["query_assignment"].shape == (4, 12)
    assert payload["environment_membership"].shape == (1, 5, 12)
    assert payload["environment_coords"].shape == (1, 5, 2)
    assert "phase_ledger" not in payload


def test_batched_padded_module_measure_wins_over_unbatched_alias() -> None:
    raw = _payload()
    module_assignment = np.pad(np.asarray(raw["module_assignment"]), ((0, 7), (0, 0)))
    module_measure = np.pad(np.asarray(raw["module_measure"]), (0, 7))[None, ...]
    module_present = np.r_[np.ones(5, dtype=bool), np.zeros(7, dtype=bool)]
    module_coords = np.pad(np.asarray(raw["module_coords"]), ((0, 7), (0, 0)))
    controls = SimpleNamespace(
        module_membership=module_assignment[None, ...],
        environment_membership=np.asarray(raw["environment_assignment"])[None, ...],
        module_measure=module_measure,
        environment_measure=np.asarray(raw["environment_measure"])[None, ...],
        module_mass=module_assignment[None, ...].sum(axis=1),
        environment_mass=np.asarray(raw["environment_assignment"])[None, ...].sum(axis=1),
        phase_occupied=np.asarray(raw["active_mask"])[None, ...],
        pi=np.asarray(raw["pi"])[None, ...],
        kappa=np.asarray([raw["kappa"]]),
        module_centres=np.zeros((1, 12, 2)),
        environment_centres=np.zeros((1, 12, 2)),
        joint_centres=np.zeros((1, 12, 2)),
    )
    encoded = SimpleNamespace(
        env_coords=np.asarray(raw["environment_coords"])[None, ...],
        env_weights=np.asarray(raw["environment_measure"])[None, ...],
    )
    prepared = SimpleNamespace(encoded=encoded, backend_state={"group_control_state": controls})
    prediction = {
        "interaction_aux": {
            # This is the live stale alias shape that previously won lookup.
            "group_control_module_measure": np.asarray([0.25]),
            "group_control_adaptive_environment_p": np.asarray(raw["coarse_assignment"]),
            "group_control_adaptive_environment_alpha": np.asarray(raw["query_assignment"]),
            "group_control_adaptive_opening_blend": np.asarray(raw["opening_blend"]),
        },
        "_prepared_state": SimpleNamespace(prepared=prepared),
    }
    sample = {
        "structure": {"module_centers": module_coords, "module_present": module_present},
        "query_source_count": 4,
        "query_selection": "full_original_grid",
    }
    payload = _prepared_payload(sample, prediction, np.asarray(raw["query_xy"]))
    assert payload["module_measure"].shape == (1, 12)
    assert payload["group_control_module_measure"].shape == (1, 12)

    record = canonicalize_candidate_case(payload, case_id="0273", module_count=5, query_count=4)
    assert record["maps"]["module_measure"].shape == (12,)
    assert record["case"]["empty_module_source_count"] == 7


def test_fixed_panel_summary_and_required_pngs(tmp_path: Path) -> None:
    records = []
    for case_id in DEFAULT_CASE_IDS:
        record = canonicalize_candidate_case(
            _payload(), case_id=case_id, module_count=3, query_count=4
        )
        records.append(record)
        sparse_incidence_evidence.save_case_arrays(
            tmp_path / "arrays" / f"{case_id}.npz", record["maps"]
        )
    summary = summarize_candidate_population([record["case"] for record in records], expected_cases=2)
    assert summary["case_count"] == 2
    assert summary["empty_query_count"]["total"] == 0
    assert summary["fine_work_ratio"]["mean"] == 0.95
    figures = render_candidate_figures([record["case"] for record in records], tmp_path)
    for key in (
        "representative_hypergraph_organization",
        "kq_case_spread",
        "kq_histogram",
        "source_degree",
        "active_mass",
        "p_alpha_opening_distributions",
        "fine_work_ratio",
        "empty_counts",
    ):
        output = Path(figures[key])
        assert output.is_file()
        assert output.stat().st_size > 10_000
