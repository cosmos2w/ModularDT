"""CPU contracts for the bounded Run-1503 diagnostic tooling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = PROJECT / "tools" / "diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

from run_run1503_diagnostics import (
    DEFAULT_CASE_IDS,
    MODULE_PANEL_CASE_IDS,
    REPRESENTATIVE_CASE_IDS,
    Run1503DiagnosticError,
    _require_exact_epoch500,
    benchmark_application_scopes,
    build_parser,
    build_plan,
    collect_case_diagnostic,
    expected_aux_contract,
    group_geometry,
    opening_diagnostics,
    opening_locality_diagnostics,
)


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    membership = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.25, 0.75, 0.0],
            [0.0, 0.0, 1.0],
            [0.50, 0.0, 0.50],
            [0.0, 0.40, 0.60],
        ],
        dtype=np.float64,
    )
    measure = np.asarray([0.10, 0.20, 0.30, 0.20, 0.20], dtype=np.float64)
    coordinates = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [8.0, 0.0], [0.0, 2.0], [8.0, 2.0]],
        dtype=np.float64,
    )
    query = np.asarray(
        [
            [0.7, 0.3, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.4, 0.6],
            [0.5, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    return membership, measure, coordinates, query


def test_group_geometry_preserves_measure_and_is_permutation_invariant() -> None:
    membership, measure, coordinates, _query = _arrays()
    geometry = group_geometry(
        membership,
        measure,
        coordinates,
        giant_mass_fraction=0.6,
        giant_source_fraction=0.8,
    )

    np.testing.assert_allclose(geometry["mass"], [0.25, 0.23, 0.52])
    assert geometry["mass_sum"] == pytest.approx(1.0)
    assert geometry["empty_groups"] == []
    assert geometry["giant_groups"] == []
    assert geometry["row_normalization_max_abs_error"] == pytest.approx(0.0)

    permutation = np.asarray([2, 0, 1])
    permuted = group_geometry(membership[:, permutation], measure, coordinates)
    np.testing.assert_allclose(permuted["mass"], geometry["mass"][permutation])
    np.testing.assert_allclose(permuted["centroid"], geometry["centroid"][permutation])
    np.testing.assert_allclose(permuted["radius"], geometry["radius"][permutation])


def test_geometry_reports_empty_and_giant_groups_without_division_nan() -> None:
    membership, measure, coordinates, _query = _arrays()
    membership = np.concatenate([membership, np.zeros((5, 1))], axis=1)
    geometry = group_geometry(
        membership,
        measure,
        coordinates,
        giant_mass_fraction=0.4,
        giant_source_fraction=0.8,
    )
    assert geometry["empty_groups"] == [3]
    assert geometry["giant_groups"] == [2]
    assert np.isfinite(geometry["centroid"]).all()
    assert np.isfinite(geometry["radius"]).all()


def test_opening_diagnostics_keeps_qk_ek_and_actual_rows_distinct() -> None:
    membership, measure, _coordinates, query = _arrays()
    coarse = np.full_like(query, 1.0 / query.shape[1])
    blend = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.25, 1.0], [0.5, 0.0, 1.0]],
        dtype=np.float64,
    )
    opening = opening_diagnostics(
        query,
        membership,
        measure,
        coarse_assignment=coarse,
        opening_blend=blend,
        receiver_chunk_size=3,
    )
    np.testing.assert_array_equal(opening["Q_k"], [2, 2, 3])
    np.testing.assert_array_equal(opening["E_k"], [3, 2, 3])
    assert opening["fine_block_area"] == pytest.approx(19.0)
    assert opening["full_QE_area"] == pytest.approx(20.0)
    assert opening["fine_block_ratio"] == pytest.approx(0.95)
    assert opening["opened_fine_block_area"] == pytest.approx(14.0)
    assert opening["coarse_assignment_present"] is True
    assert opening["coarse_global_availability"]["status"] == "observed_assignment"
    assert opening["chunk_support"] == {
        "receiver_chunk_size": 3,
        "query_chunk_count": 2,
        "timing_matched_to_receiver_chunk": True,
    }


def test_opening_locality_distinguishes_near_opened_from_far_closed_groups() -> None:
    query_coordinates = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    centroids = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    radii = np.asarray([1.0, 1.0], dtype=np.float64)
    assignment = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    locality = opening_locality_diagnostics(
        query_coordinates,
        assignment,
        centroids,
        radii,
    )
    assert locality["all_queries_have_open_group"] is True
    assert locality["nearest_centroid_open_fraction"] == pytest.approx(1.0)
    assert locality["opened_distance_summary"]["mean"] == pytest.approx(0.0)
    assert locality["closed_distance_summary"]["mean"] == pytest.approx(10.0)
    assert locality["opened_to_closed_mean_distance_ratio"] == pytest.approx(0.0)


def test_collect_case_diagnostic_accepts_nested_current_aliases_and_marks_inference() -> None:
    membership, measure, coordinates, query = _arrays()
    payload = {
        "routing_maps": {"sparse_incidence_query_routing": query[None, ...]},
        "prepared_state": {
            "backend_state": {
                "group_control_state": {
                    "environment_membership": membership[None, ...],
                    "environment_measure": measure[None, ...],
                }
            },
            "encoded": {"env_coords": coordinates[None, ...]},
        },
    }
    result = collect_case_diagnostic(payload, case_id="0273", module_count=3)
    assert result["case_id"] == "0273"
    assert result["module_count"] == 3
    assert result["geometry"]["mass_sum"] == pytest.approx(1.0)
    assert result["opening"]["opening_status"] == "inferred_from_sparse_query_support"
    assert result["opening"]["coarse_global_availability"]["status"] == "unavailable"
    assert result["executor_ledger"]["status"] == "unavailable"


def test_collect_case_diagnostic_uses_explicit_ledger_for_actual_rows() -> None:
    membership, measure, coordinates, query = _arrays()
    payload = {
        "adaptive_hyperedge_environment_membership": membership,
        "adaptive_hyperedge_environment_measure": measure,
        "adaptive_hyperedge_environment_coordinates": coordinates,
        "adaptive_hyperedge_query_sparse_assignment": query,
        "adaptive_hyperedge_query_coarse_assignment": np.full_like(query, 1.0 / 3.0),
        "adaptive_hyperedge_opening_blend": query,
        "adaptive_hyperedge_coarse_available": np.ones_like(query, dtype=bool),
        "adaptive_hyperedge_phase_ledger": {
            "P2": {
                "environment": {
                    "support_pairs": 17,
                    "executed_rows": 19,
                    "padded_rows": 24,
                    "recomputed_rows": 19,
                    "fine_group_rows": 19,
                }
            }
        },
    }
    result = collect_case_diagnostic(payload, case_id="0653")
    assert result["opening"]["opening_status"] == "explicit_blend_support"
    assert result["opening"]["coarse_global_availability"]["status"] == "explicit"
    assert result["executor_ledger"] == {
        "support_pairs": 17.0,
        "executed_rows": 19.0,
        "padded_rows": 24.0,
        "recomputed_rows": 19.0,
        "coarse_rows": None,
        "fine_group_rows": 19.0,
        "status": "ok",
        "source": "explicit_normal_executor_phase_ledger",
    }


def test_collect_case_diagnostic_accepts_implemented_flat_adaptive_aux_keys() -> None:
    membership, measure, coordinates, query = _arrays()
    payload = {
        "interaction_aux": {
            "group_control_environment_incidence": membership,
            "group_control_environment_measure": measure,
            "group_control_adaptive_environment_p": np.full_like(query, 1.0 / 3.0),
            "group_control_adaptive_environment_alpha": query,
            "group_control_adaptive_opening_blend": query,
            "group_control_adaptive_fine_group_rows": np.asarray([6.0, 4.0, 9.0]),
            "group_control_adaptive_fine_rows": 19.0,
            "group_control_adaptive_coarse_rows": 12.0,
            "group_control_adaptive_full_rectangle_rows": 20.0,
            "group_control_adaptive_fine_work_ratio": 0.95,
            "group_control_environment_fine_rows_padded": 0.0,
            "group_control_environment_fine_rows_recompute": 0.0,
            "group_control_environment_unique_pairs": 11.0,
            "group_control_adaptive_coarse_contribution": np.ones((4, 3, 2)),
            "group_control_adaptive_fine_contribution": np.ones((4, 3, 2)),
        },
        "encoded": {"env_coords": coordinates},
    }
    result = collect_case_diagnostic(payload, case_id="0273")
    assert result["opening"]["coarse_assignment_present"] is True
    assert result["opening"]["opening_status"] == "explicit_blend_support"
    assert result["opening"]["observed_execution"]["status"] == "observed_flat_executor_aux"
    assert result["opening"]["observed_execution"]["fine_work_ratio"] == pytest.approx(0.95)
    assert result["executor_ledger"]["status"] == "ok"
    assert result["executor_ledger"]["source"] == "flat_normal_executor_aux"
    assert result["executor_ledger"]["executed_rows"] == 19.0
    assert result["branch_finite"] == {"coarse_contribution": True, "fine_contribution": True}


def test_opening_rejects_nonfinite_or_negative_inputs() -> None:
    membership, measure, _coordinates, query = _arrays()
    bad = query.copy()
    bad[0, 0] = -1.0
    with pytest.raises(Run1503DiagnosticError, match="query_assignment"):
        opening_diagnostics(bad, membership, measure)


def test_cpu_timing_scopes_use_one_matched_chunk_and_report_memory_fields() -> None:
    seen: list[int] = []

    def scope(chunk: int) -> np.ndarray:
        seen.append(chunk)
        return np.ones((chunk, 2), dtype=np.float32)

    result = benchmark_application_scopes(
        {
            "full_physical_forward": scope,
            "prepared_p2_decode": lambda chunk: np.zeros((chunk, 1), dtype=np.float32),
            "application_evaluator": None,
        },
        receiver_chunk_size=7,
        warmups=1,
        repetitions=2,
    )
    assert result["status"] == "partial"
    assert seen == [7, 7, 7]
    full = result["scopes"]["full_physical_forward"]
    assert full["status"] == "complete"
    assert full["receiver_chunk_size"] == 7
    assert len(full["samples"]) == 2
    assert all(sample["cuda_event_elapsed_seconds"] is None for sample in full["samples"])
    assert all("python_tracemalloc_peak_bytes" in sample for sample in full["samples"])
    assert result["scopes"]["application_evaluator"]["status"] == "unavailable"


def test_plan_is_cpu_safe_and_exposes_future_aux_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "Run_1502" / "epoch_0500_model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"placeholder")
    output = tmp_path / "diagnostic.json"
    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--plan-only",
        ]
    )
    plan = build_plan(args)
    assert plan["status"] == "plan_only"
    assert plan["protocol"]["case_ids"] == list(DEFAULT_CASE_IDS)
    assert plan["protocol"]["representative_case_ids"] == list(REPRESENTATIVE_CASE_IDS)
    assert plan["protocol"]["module_panel_case_ids"] == list(MODULE_PANEL_CASE_IDS)
    assert plan["protocol"]["receiver_chunk_size"] == 2048
    assert plan["protocol"]["timed_scopes"] == [
        "full_physical_forward",
        "prepared_p2_decode",
        "application_evaluator",
    ]
    contract = expected_aux_contract()
    assert "adaptive_hyperedge_query_sparse_assignment" in contract["required"]
    assert "adaptive_hyperedge_opening_blend" in contract["recommended"]
    assert "actual_rows" in contract["semantics"]
    json.dumps(plan)


def test_plan_rejects_non_cpu_device_and_nonpositive_chunk(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch_0500_model.pt"
    checkpoint.write_bytes(b"placeholder")
    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "out.json"),
            "--device",
            "cuda:0",
        ]
    )
    with pytest.raises(ValueError, match="CPU-only"):
        build_plan(args)
    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "out.json"),
            "--receiver-chunk-size",
            "0",
        ]
    )
    with pytest.raises(ValueError, match="chunk sizes"):
        build_plan(args)


def test_frozen_loader_requires_unambiguous_payload_epoch_500() -> None:
    assert _require_exact_epoch500({"epoch": 500}) == 500
    assert _require_exact_epoch500(
        {"epoch": 500, "selection_state": {"epoch": 500, "total_epochs": 5000}}
    ) == 500
    with pytest.raises(Run1503DiagnosticError, match="exact epoch 500"):
        _require_exact_epoch500({"epoch": 499})
    with pytest.raises(Run1503DiagnosticError, match="exact epoch 500"):
        _require_exact_epoch500({"epoch": 500, "selection_state": {"epoch": 499}})
    with pytest.raises(Run1503DiagnosticError, match="exact epoch 500"):
        _require_exact_epoch500({})


def test_predicted_port_smoke_requires_min_max_steps_and_finite_updates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import run_run1503_diagnostics as diagnostics

    calls: dict[str, argparse.Namespace] = {}

    def fake_run_smoke(args: argparse.Namespace) -> dict[str, object]:
        calls["args"] = args
        metrics = {
            "loss_total": 1.25,
            "preclip_gradient_norm": 2.5,
            "parameter_update_norm": 0.125,
            "preclip_gradient_norm_group_prepare": 1.5,
            "parameter_update_norm_group_prepare": 0.025,
        }
        return {
            "steps": {
                "small_module_batch": {
                    "module_counts": [3],
                    "metrics": metrics,
                    "canonical_loss_config": {
                        "key_count": 18,
                        "predicted_consistency_weight_configured": 0.05,
                        "predicted_consistency_weight_used": 0.0005,
                    },
                    "parameters_finite": True,
                    "optimizer_state_finite": True,
                    "optimizer_update_applied": True,
                    "branch_gradients": {
                        path: {"status": "ok"}
                        for path in ("coarse_path", "fine_path", "module_reader")
                    },
                },
                "large_module_batch": {
                    "module_counts": [10],
                    "metrics": metrics,
                    "canonical_loss_config": {
                        "key_count": 18,
                        "predicted_consistency_weight_configured": 0.05,
                        "predicted_consistency_weight_used": 0.0005,
                    },
                    "parameters_finite": True,
                    "optimizer_state_finite": True,
                    "optimizer_update_applied": True,
                    "branch_gradients": {
                        path: {"status": "ok"}
                        for path in ("coarse_path", "fine_path", "module_reader")
                    },
                },
            }
        }

    monkeypatch.setattr(diagnostics, "_run_predicted_port_runtime", fake_run_smoke)
    checkpoint = tmp_path / "epoch_0500_model.pt"
    checkpoint.write_bytes(b"placeholder")
    output = tmp_path / "smoke.json"
    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--smoke",
            "--device",
            "cpu",
            "--profile",
            "project://src/config_core/forward/adaptive_hyperedge_opening_honf_context.json",
        ]
    )

    result = diagnostics.run_predicted_port_smoke(args)

    assert result["status"] == "complete"
    assert result["module_count_range_observed"] == [3, 10]
    assert result["gradient_update_evidence"]["small_module_batch"]["status"] == "ok"
    assert result["gradient_update_evidence"]["large_module_batch"]["status"] == "ok"
    assert result["gradient_update_evidence"]["small_module_batch"]["canonical_loss_config_ok"] is True
    smoke_args = calls["args"]
    assert smoke_args.device == "cpu"
    assert smoke_args.profile == "project://src/config_core/forward/adaptive_hyperedge_opening_honf_context.json"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"
