"""Focused CPU checks for the selected-checkpoint Wind action table."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


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
            "capacity_vector": {"MM": 0.90, "QE": 0.95}, "phase": "action_family",
            "primary_pass": 4, "full_access_replay": True,
        },
        {
            "update_count": 751, "row": 16, "requested_action": "four_packet",
            "optimizer_attempt_key": "accepted:future",
            "requested_cut_paths": list(TABLE.WIND_ACTION_PATHS["four_packet"]),
            "capacity_vector": {"MM": 0.90, "QE": 0.95}, "phase": "action_family",
            "primary_pass": 4, "full_access_replay": False,
        },
        *[
            {
                "update_count": 730 + row, "row": row,
                "optimizer_attempt_key": f"orphaned:{row}",
                "requested_action": "root",
                "requested_cut_paths": list(TABLE.WIND_ACTION_PATHS["root"]),
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
        "root", exact_work=20.0, full_work=40.0, exposure_record=exposure["root"]
    ) is True
    assert TABLE._trained_sparse_action(
        "root", exact_work=40.0, full_work=40.0, exposure_record=exposure["root"]
    ) is False
    assert TABLE._trained_sparse_action(
        "two_packet", exact_work=20.0, full_work=40.0, exposure_record=exposure["two_packet"]
    ) is False


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
