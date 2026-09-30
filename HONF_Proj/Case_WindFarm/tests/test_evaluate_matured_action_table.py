"""Focused CPU checks for the selected-checkpoint Wind action table."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_matured_action_table.py"
SPEC = importlib.util.spec_from_file_location("wind_matured_action_table", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TABLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TABLE)


def _four_action_rows(checkpoint_sha: str) -> list[dict[str, object]]:
    rows = []
    common = {
        "case_key": "layout004_row016_fixed",
        "row_id": 16,
        "layout_key": "4",
        "family_key": "4",
        "split": "train_fit",
        "query_panel": "fixed",
    }
    for action, k, work in (
        ("root", 1, 8.0),
        ("two_packet", 2, 15.0),
        ("four_packet", 4, 23.0),
        ("full_access", None, 40.0),
    ):
        rows.append({
            **common,
            "action_key": action,
            "forward_checkpoint_sha256": checkpoint_sha,
            "full_access": action == "full_access",
            "nonredundant_k": k,
            "exact_work": work,
            "full_work": 40.0,
        })
    return rows


def test_checkpoint_binding_fields_do_not_label_durability_as_review() -> None:
    durability = {
        "run_id": "2112", "arm": "g_packet", "update_count": 1750,
        "checkpoint_kind": "durability_only", "checkpoint": "/tmp/g_u1750.pt",
        "checkpoint_sha256": "a" * 64,
    }
    fields = TABLE._checkpoint_binding_fields(
        "g", durability, "b" * 64, "current_durability_checkpoint"
    )
    assert fields["g_checkpoint_binding_source"] == "current_durability_checkpoint"
    assert fields["g_checkpoint_binding_record"] == durability
    assert fields["g_checkpoint_review_record"] is None
    assert fields["g_checkpoint_review_record_line_sha256"] is None

    scheduled = {**durability, "checkpoint_kind": "scheduled_review"}
    fields = TABLE._checkpoint_binding_fields(
        "p", scheduled, "c" * 64, "append_only_scheduled_review"
    )
    assert fields["p_checkpoint_binding_source"] == "append_only_scheduled_review"
    assert fields["p_checkpoint_review_record"] == scheduled
    assert fields["p_checkpoint_review_record_line_sha256"] == "c" * 64


def test_effective_packet_feature_quotient_deduplicates_typed_sources_and_drops_empty() -> None:
    features = np.asarray([[1.0, 0.0], [3.0, 2.0], [9.0, 9.0], [5.0, 7.0]], dtype=np.float32)
    mm = np.asarray([[1, 0], [1, 0], [0, 0], [0, 1]], dtype=np.uint8)
    qe = np.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)

    packet_rows, effective_mm, effective_qe, signatures = TABLE._canonical_packet_features(
        features, mm, qe
    )

    assert packet_rows.shape == (2, 2)
    np.testing.assert_allclose(packet_rows, [[2.0, 1.0], [5.0, 7.0]])
    np.testing.assert_array_equal(effective_mm, [[1, 0], [0, 1]])
    np.testing.assert_array_equal(effective_qe, [[0, 0, 1], [0, 0, 0]])
    assert len(signatures) == 2
    assert len(set(signatures)) == 2


def test_budget_feature_uses_realized_full_access_budget() -> None:
    np.testing.assert_allclose(TABLE._action_budget_vector("root"), [0.90, 0.95], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(TABLE._action_budget_vector("full_access"), [1.0, 1.0])


def test_qe_full_recipe_does_not_count_old_capacity_as_mature_exposure(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [16, 17]
    ledger = []
    for pass_id, qe, covered in (
        (2, 0.95, rows),
        (3, 0.95, rows),
        (4, 1.0, rows),
        (5, 1.0, rows[:1]),
    ):
        for row in covered:
            ledger.append({
                "update_count": 700 + len(ledger),
                "optimizer_attempt_key": f"accepted:{len(ledger)}",
                "row": row,
                "requested_action": "root",
                "requested_cut_paths": [""],
                "realized_cut_paths": [""],
                "realized_nonredundant_k": 1,
                "capacity_vector": {"MM": 0.90, "QE": qe},
                "phase": "action_family",
                "primary_pass": pass_id,
                "full_access_replay": False,
            })
    monkeypatch.setattr(TABLE, "PRIMARY_CAPACITY", {"QE": 1.0, "MM": 0.90})
    exposure = TABLE._summarize_action_exposure(
        ledger, training_rows=rows, selected_update=800
    )["root"]
    assert exposure["completed_primary_pass_ids"] == [4]
    assert exposure["eligible_action_family_update_count"] == 3
    assert exposure["trained_action"] is False
    assert exposure["primary_capacity"] == {"MM": 0.90, "QE": 1.0}
    np.testing.assert_allclose(TABLE._action_budget_vector("root"), [0.90, 1.0])


def test_training_exposure_requires_two_complete_primary_action_passes() -> None:
    rows = [16, 17]
    ledger = []
    for action in ("root", "two_packet", "four_packet"):
        for pass_id in (2, 3):
            covered = rows if action != "two_packet" or pass_id == 2 else rows[:1]
            ledger.extend({
                    "update_count": 700 + len(ledger),
                    "optimizer_attempt_key": f"accepted:{len(ledger)}",
                "row": row,
                "requested_action": action,
                "requested_cut_paths": list(TABLE.WIND_ACTION_PATHS[action]),
                "realized_cut_paths": list(TABLE.WIND_ACTION_PATHS[action]),
                "realized_nonredundant_k": len(TABLE.WIND_ACTION_PATHS[action]),
                "capacity_vector": {"MM": 0.90, "QE": 0.95},
                "phase": "action_family",
                "primary_pass": pass_id,
                "full_access_replay": False,
            } for row in covered)
    ledger.extend([
        {
            "update_count": 749, "row": 16, "requested_action": "two_packet",
            "optimizer_attempt_key": "accepted:replay",
            "requested_cut_paths": list(TABLE.WIND_ACTION_PATHS["two_packet"]),
            "realized_cut_paths": list(TABLE.WIND_ACTION_PATHS["two_packet"]),
            "realized_nonredundant_k": 2,
            "capacity_vector": {"MM": 0.90, "QE": 0.95}, "phase": "action_family",
            "primary_pass": 4, "full_access_replay": True,
        },
        {
            "update_count": 751, "row": 16, "requested_action": "four_packet",
            "optimizer_attempt_key": "accepted:future",
            "requested_cut_paths": list(TABLE.WIND_ACTION_PATHS["four_packet"]),
            "realized_cut_paths": list(TABLE.WIND_ACTION_PATHS["four_packet"]),
            "realized_nonredundant_k": 4,
            "capacity_vector": {"MM": 0.90, "QE": 0.95}, "phase": "action_family",
            "primary_pass": 4, "full_access_replay": False,
        },
        *[
            {
                "update_count": 730 + row, "row": row,
                "optimizer_attempt_key": f"orphaned:{row}",
                "requested_action": "root",
                "requested_cut_paths": list(TABLE.WIND_ACTION_PATHS["root"]),
                "realized_cut_paths": list(TABLE.WIND_ACTION_PATHS["root"]),
                "realized_nonredundant_k": 1,
                "capacity_vector": {"MM": 0.90, "QE": 0.95}, "phase": "action_family",
                "primary_pass": 5, "full_access_replay": False,
            }
            for row in rows
        ],
    ])

    exposure = TABLE._summarize_action_exposure(
        ledger, training_rows=rows, selected_update=750,
        archived_optimizer_attempt_keys=["orphaned:16", "orphaned:17"],
    )

    assert exposure["root"]["completed_primary_pass_ids"] == [2, 3]
    assert exposure["root"]["trained_action"] is True
    assert exposure["two_packet"]["completed_primary_action_passes"] == 1
    assert exposure["two_packet"]["trained_action"] is False
    assert exposure["four_packet"]["completed_primary_action_passes"] == 2
    assert TABLE._trained_sparse_action(
        "root", exact_work=20.0, full_work=40.0, exposure_record=exposure["root"],
        row_id=16, realized_cut_paths=[""], nonredundant_k=1,
    ) is True
    assert TABLE._trained_sparse_action(
        "root", exact_work=40.0, full_work=40.0, exposure_record=exposure["root"],
        row_id=16, realized_cut_paths=[""], nonredundant_k=1,
    ) is False
    assert TABLE._trained_sparse_action(
        "two_packet", exact_work=20.0, full_work=40.0, exposure_record=exposure["two_packet"],
        row_id=16, realized_cut_paths=["L", "R"], nonredundant_k=2,
    ) is False


def test_exposure_audit_reports_path_k_overlap_without_inventing_exact_k_maturity() -> None:
    ledger = []
    for pass_id, records in (
        (2, [(1, ["L", "R"], 2), (2, [""], 1)]),
        (3, [(1, [""], 1), (2, ["L", "R"], 2)]),
    ):
        for row_id, paths, k in records:
            ledger.append({
                "update_count": 700 + len(ledger),
                "optimizer_attempt_key": f"accepted:{len(ledger)}",
                "row": row_id,
                "requested_action": "two_packet",
                "requested_cut_paths": ["L", "R"],
                "realized_cut_paths": paths,
                "realized_nonredundant_k": k,
                "capacity_vector": {"MM": 0.90, "QE": 0.95},
                "phase": "action_family",
                "primary_pass": pass_id,
                "full_access_replay": False,
            })
    exposure = TABLE._summarize_action_exposure(
        ledger, training_rows=[1, 2], selected_update=800,
    )["two_packet"]
    assert exposure["trained_action"] is True
    assert exposure["realized_cut_paths_by_primary_pass"]["2"]["resolved_path_families"][
        '["L","R"]'
    ]["realized_nonredundant_k_counts"] == {"2": 1}
    assert len(exposure["resolved_path_exposure_by_row"]["1"]) == 2
    row_audit = TABLE._row_action_exposure(
        exposure, action="two_packet", row_id=1,
        realized_cut_paths=["L", "R"], nonredundant_k=2,
    )
    assert row_audit["same_realized_path_primary_pass_ids"] == [2]
    assert row_audit["same_realized_path_and_k_primary_pass_ids"] == [2]
    assert TABLE._trained_sparse_action(
        "two_packet", exact_work=10.0, full_work=20.0, exposure_record=exposure,
        row_id=1, realized_cut_paths=["L", "R"], nonredundant_k=2,
    ) is True


def test_missing_or_invalid_realized_k_prevents_a_complete_training_pass() -> None:
    ledger = []
    for pass_id in (2, 3):
        for row_id in (1, 2):
            ledger.append({
                "update_count": 700 + len(ledger),
                "optimizer_attempt_key": f"accepted:{len(ledger)}",
                "row": row_id,
                "requested_action": "root",
                "requested_cut_paths": [""],
                "realized_cut_paths": [""],
                "realized_nonredundant_k": (None if pass_id == 2 and row_id == 2 else 1),
                "capacity_vector": {"MM": 0.90, "QE": 0.95},
                "phase": "action_family",
                "primary_pass": pass_id,
                "full_access_replay": False,
            })

    exposure = TABLE._summarize_action_exposure(
        ledger, training_rows=[1, 2], selected_update=800,
    )["root"]
    assert exposure["invalid_realized_cut_path_update_count"] == 0
    assert exposure["invalid_realized_nonredundant_k_update_count"] == 1
    assert exposure["invalid_realized_execution_update_count"] == 1
    assert exposure["completed_primary_pass_ids"] == [3]
    assert exposure["trained_action"] is False


def test_exposure_lineage_hashes_the_same_complete_ledger_snapshot_it_parses(tmp_path: Path) -> None:
    ledger = tmp_path / "arms" / "g_packet" / "updates.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger_bytes = b"\n"
    ledger.write_bytes(ledger_bytes)
    recovery = ledger.parent / "recovery_events.jsonl"
    recovery_bytes = b'{"orphaned_attempt_keys":["discarded"]}\n'
    recovery.write_bytes(recovery_bytes)

    exposure, lineage = TABLE._read_exposure(ledger, training_rows=[1], update_count=1630)
    assert exposure["root"]["completed_primary_action_passes"] == 0
    assert lineage["update_ledger_sha256"] == hashlib.sha256(ledger_bytes).hexdigest()
    assert lineage["update_ledger_snapshot_bytes"] == len(ledger_bytes)
    assert lineage["update_ledger_snapshot_record_count"] == 0
    assert lineage["recovery_ledger_sha256"] == hashlib.sha256(recovery_bytes).hexdigest()

    ledger.write_bytes(b'{"partial":')
    with pytest.raises(ValueError, match="partial JSONL record"):
        TABLE._read_exposure(ledger, training_rows=[1], update_count=1630)


def test_query_repeat_summary_uses_measured_overlap_and_preserves_partial_overlap() -> None:
    summary = TABLE._query_repeat_overlap_summary([
        {
            "case_key": f"repeat_{row_id}", "row_id": row_id,
            "query_panel": "query_repeat", "query_repeat_overlap_count": overlap,
            "query_repeat_disjoint_from_fixed": overlap == 0,
            "query_repeat_candidate_attempt_count": 1024,
        }
        for row_id, overlap in ((1, 0), (2, 3))
    ])
    assert summary["measured_repeat_case_count"] == 2
    assert summary["disjoint_repeat_case_count"] == 1
    assert summary["overlapping_repeat_case_count"] == 1
    assert summary["total_intersecting_query_indices"] == 3
    assert summary["all_measured_repeats_disjoint"] is False


def test_geometry_support_keeps_realized_packet_sizes_and_ignores_invalid_sources() -> None:
    selected = TABLE._nearest_geometry_membership(
        packet_centers=np.asarray([[0.0, 0.0], [10.0, 0.0]]),
        source_coordinates=np.asarray([[0.1, 0.0], [8.0, 0.0], [10.2, 0.0], [0.0, 0.0]]),
        source_validity=np.asarray([True, True, True, False]),
        packet_sizes=np.asarray([1, 2]),
        coordinate_scale=np.asarray([1.0, 1.0]),
    )

    np.testing.assert_array_equal(selected.sum(axis=1), [1, 2])
    np.testing.assert_array_equal(selected.sum(axis=0), [1, 1, 1, 0])
    np.testing.assert_array_equal(selected, [[1, 0, 0, 0], [0, 1, 1, 0]])


def test_degree_size_rewire_preserves_both_marginals_and_reports_changed_links() -> None:
    support = np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    rewired, swaps, changed_links = TABLE._degree_size_preserving_rewire(support, seed=2)

    assert swaps >= 1
    assert changed_links == int(np.count_nonzero(rewired != support)) == 4 * swaps
    np.testing.assert_array_equal(rewired.sum(axis=1), support.sum(axis=1))
    np.testing.assert_array_equal(rewired.sum(axis=0), support.sum(axis=0))
    assert not np.array_equal(rewired, support)

    dense, no_swaps, no_changed_links = TABLE._degree_size_preserving_rewire(
        np.ones((2, 2), dtype=np.uint8), seed=2
    )
    np.testing.assert_array_equal(dense, np.ones((2, 2), dtype=np.uint8))
    assert no_swaps == no_changed_links == 0


def test_count_matched_pair_projection_uses_stable_exact_top_n_and_fails_closed() -> None:
    scores = torch.tensor([[2.0, 2.0], [3.0, -100.0]])
    eligible = torch.tensor([[True, True], [True, False]])

    selected = TABLE._top_count_pair_mask(scores, eligible, 2)

    assert int(selected.sum()) == 2
    torch.testing.assert_close(selected, torch.tensor([[True, False], [True, False]]))
    with pytest.raises(ValueError, match="exceeds the eligible live-pair population"):
        TABLE._top_count_pair_mask(scores, eligible, 4)


def test_training_population_feature_means_are_train_case_weighted_and_typed() -> None:
    records = [
        {
            "module_states": torch.tensor([[1.0, 2.0], [99.0, 99.0]]),
            "module_valid": torch.tensor([True, False]),
            "environment_states": torch.tensor([[0.0, 1.0], [0.0, 3.0]]),
            "environment_weights": torch.tensor([1.0, 3.0]),
            "global_state": torch.tensor([[2.0, 2.0]]),
        },
        {
            "module_states": torch.tensor([[5.0, 6.0]]),
            "module_valid": torch.tensor([True]),
            "environment_states": torch.tensor([[0.0, 5.0], [0.0, 7.0]]),
            "environment_weights": torch.tensor([1.0, 1.0]),
            "global_state": torch.tensor([[4.0, 4.0]]),
        },
    ]
    means = TABLE._training_population_feature_means(records)
    encoded = SimpleNamespace(
        module_tokens=torch.zeros(1, 3, 2),
        env_tokens=torch.zeros(1, 2, 2),
        global_token=torch.zeros(1, 2),
    )

    torch.testing.assert_close(means["module_state"], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(means["environment_state"], torch.tensor([0.0, 4.25]))
    torch.testing.assert_close(means["global_state"], torch.tensor([3.0, 3.0]))
    context = TABLE._fixed_population_context(encoded, means)
    assert tuple(context["module_states"].shape) == (1, 3, 2)
    assert tuple(context["environment_states"].shape) == (1, 2, 2)
    assert tuple(context["global_state"].shape) == (1, 2)
    torch.testing.assert_close(context["module_states"], means["module_state"].expand(1, 3, 2))
    torch.testing.assert_close(
        context["environment_states"], means["environment_state"].expand(1, 2, 2)
    )


def test_floor_and_query_allowance_are_rolewise_train_family_calibrations() -> None:
    roles = ("a", "b")
    floor = TABLE._numerical_floor_role_mps([
        {"family_key": "one", "wfull_role_rmse_mps": {"a": 0.01, "b": 0.02}},
        {"family_key": "one", "wfull_role_rmse_mps": {"a": 0.03, "b": 0.04}},
        {"family_key": "two", "wfull_role_rmse_mps": {"a": 0.02, "b": 0.06}},
    ], roles=roles)
    assert floor == pytest.approx([2.0e-8, 4.5e-8])

    allowance, details = TABLE._sampling_allowance_role_mps({
        "one": {"a": [0.001, 0.003], "b": [0.004, 0.006]},
        "two": {"a": [0.005], "b": [0.008]},
    }, roles=roles)
    assert allowance == pytest.approx([0.00485, 0.00785])
    assert details["train_fit_family_count"] == 2


def test_action_row_contract_requires_four_g_rows_and_exact_selected_sha() -> None:
    sha = hashlib.sha256(b"selected G checkpoint").hexdigest()
    rows = _four_action_rows(sha)

    TABLE._validate_action_rows(rows, expected_sha=sha)
    with pytest.raises(ValueError, match="selected G checkpoint"):
        TABLE._validate_action_rows(rows, expected_sha="0" * 64)
    with pytest.raises(ValueError, match="Non-G action"):
        TABLE._validate_action_rows([*rows, {**rows[0], "action_key": "p_direct_primary"}], expected_sha=sha)
    with pytest.raises(ValueError, match="exactly four G actions"):
        TABLE._validate_action_rows(rows[:-1], expected_sha=sha)
