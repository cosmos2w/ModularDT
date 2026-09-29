"""Checkpoint-bound Wind action table and train-only policy calibration."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_matured_action_selector.py"
SPEC = importlib.util.spec_from_file_location("wind_action_selector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _table(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "selected_g.pt"
    checkpoint.write_bytes(b"exact selected forward checkpoint fixture")
    feature_path = tmp_path / "action_features.npz"
    mask_path = tmp_path / "action_masks.npz"
    roles = ["background", "downstream_envelope", "hub_slab", "near_turbine", "volume"]
    arrays = {}
    masks = {}
    rows = []
    for family, split in (("layout_1", "train_fit"), ("layout_2", "train_fit"), ("layout_3", "dev")):
        for action_key, count, work in (("root", 1, 10.0), ("two_packet", 2, 20.0),
                                        ("four_packet", 4, 30.0), ("full_access", None, 40.0)):
            prefix = f"{family}_{action_key}"
            packet = f"packet_{prefix}"
            budget = f"budget_{prefix}"
            descriptors = f"roles_{prefix}"
            arrays[packet] = np.full((count or 1, 3), 0.1 + work / 100, dtype=np.float32)
            arrays[budget] = np.asarray([0.9, 0.95], dtype=np.float32)
            arrays[descriptors] = np.ones((5, 2), dtype=np.float32)
            masks[f"mask_{prefix}"] = np.ones((count or 1, 5), dtype=np.uint8)
            rows.append({
                "split": split, "family_key": family, "case_key": f"{family}_fixed",
                "row_id": int(family[-1]), "layout_id": int(family[-1]),
                "module_count": 10 + int(family[-1]),
                "query_panel": "fixed", "action_key": action_key,
                "forward_checkpoint_sha256": SELECTOR._sha256(checkpoint),
                "full_access": action_key == "full_access",
                "trained_sparse": action_key != "full_access",
                "nonredundant_k": count, "exact_work": work, "full_work": 40.0,
                "candidate_role_error": {role: 0.01 + work / 1000 for role in roles},
                "incumbent_role_error": {role: 0.04 for role in roles},
                "packet_rows_npz_key": packet, "budget_vector_npz_key": budget,
                "receiver_role_features_npz_key": descriptors,
            })
    np.savez(feature_path, **arrays)
    np.savez(mask_path, **masks)
    table = {
        "format_version": 1,
        "forward_checkpoint_sha256": SELECTOR._sha256(checkpoint),
        "feature_npz_sha256": SELECTOR._sha256(feature_path),
        "mask_npz_sha256": SELECTOR._sha256(mask_path),
        "feature_sha256_by_npz_key": {
            key: _sha(np.ascontiguousarray(value).tobytes()) for key, value in arrays.items()
        },
        "roles": roles,
        "numerical_floor_role_mps": [1e-8] * 5,
        "absolute_allowance_role_mps": [0.0] * 5,
        "absolute_allowance_calibration": {
            "calibration_kind": "zero extra physical-risk allowance for paired same-query comparison"
        },
        "rows": rows,
    }
    table_path = tmp_path / "action_table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    return table_path, feature_path, mask_path, checkpoint


def test_action_fit_keeps_held_labels_out_of_risk_weights_and_gate(tmp_path: Path) -> None:
    paths = _table(tmp_path)
    rows, metadata = SELECTOR._load_action_table(*paths)
    first, model = SELECTOR.fit_and_evaluate(rows, metadata, updates=2, seed=19)
    poisoned = [replace(row, candidate_role_error=torch.full((5,), 0.9))
                if row.family_key == "layout_3" else row for row in rows]
    second, altered = SELECTOR.fit_and_evaluate(poisoned, metadata, updates=2, seed=19)
    assert model is not None and altered is not None
    assert first["fixed_train_role_log_limits"] == second["fixed_train_role_log_limits"]
    assert first["neural_empirical_upper_margin_by_role"] == second["neural_empirical_upper_margin_by_role"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, altered.state_dict()[key], rtol=0, atol=0)
    assert first["split_results"]["dev"]["neural"]["per_case"][0]["selected_action_key"] == (
        second["split_results"]["dev"]["neural"]["per_case"][0]["selected_action_key"]
    )
    assert first["split_results"]["dev"]["neural"]["per_case"][0]["measured_adequate_sparse"] != (
        second["split_results"]["dev"]["neural"]["per_case"][0]["measured_adequate_sparse"]
    )


def test_action_table_rejects_stale_checkpoint_and_changed_feature_bytes(tmp_path: Path) -> None:
    table, features, masks, checkpoint = _table(tmp_path)
    checkpoint.write_bytes(b"different forward weights")
    with pytest.raises(ValueError, match="checkpoint differs"):
        SELECTOR._load_action_table(table, features, masks, checkpoint)
    checkpoint.write_bytes(b"exact selected forward checkpoint fixture")
    arrays = dict(np.load(features, allow_pickle=False))
    arrays["packet_layout_1_root"][0, 0] = 99.0
    np.savez(features, **arrays)
    with pytest.raises(ValueError, match="archive differs"):
        SELECTOR._load_action_table(table, features, masks, checkpoint)


def test_paired_native_gate_rejects_sampling_variation_as_extra_allowance(tmp_path: Path) -> None:
    table_path, features, masks, checkpoint = _table(tmp_path)
    table = json.loads(table_path.read_text(encoding="utf-8"))
    table["absolute_allowance_role_mps"] = [0.005] * 5
    table_path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(ValueError, match="query-sampling variability"):
        SELECTOR._load_action_table(table_path, features, masks, checkpoint)


def test_train_gate_uses_fixed_directions_within_each_layout(tmp_path: Path) -> None:
    rows, metadata = SELECTOR._load_action_table(*_table(tmp_path))
    original = next(row for row in rows if row.family_key == "layout_1" and row.full_access)
    extra = [
        replace(original, case_key=f"layout_1_direction_{index}",
                incumbent_role_error=torch.full((5,), error))
        for index, error in ((2, 0.06), (3, 0.08))
    ]
    case_metadata = dict(metadata["case_metadata"])
    for row in extra:
        case_metadata[row.case_key] = {**case_metadata[original.case_key], "query_panel": "fixed"}
    _limits, calibration = SELECTOR._train_fixed_limits(
        [*rows, *extra], metadata["split_families"]["train_fit"],
        case_metadata, metadata["allowance"],
    )
    layout = next(item for item in calibration if item["family_key"] == "layout_1")
    assert layout["fixed_direction_count"] == 3
    assert layout["incumbent_role_rmse_mps_median"] == pytest.approx([0.06] * 5)


def test_no_exposed_sparse_action_is_explicit_fallback_not_success(tmp_path: Path) -> None:
    rows, metadata = SELECTOR._load_action_table(*_table(tmp_path))
    unexposed = [replace(row, trained_sparse=False) if not row.full_access else row for row in rows]
    result, model = SELECTOR.fit_and_evaluate(unexposed, metadata, updates=2)
    assert model is None and result["optimizer_calls"] == 0
    assert result["selector_fit"] == "unavailable_no_exposed_train_sparse_action"
    assert result["split_results"]["dev"]["fixed_action"]["root"]["exposed_sparse_rows"] == 0
