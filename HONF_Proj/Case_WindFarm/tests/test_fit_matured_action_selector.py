"""Checkpoint-bound Wind action table and train-only policy calibration."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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


def _table(tmp_path: Path, *, include_repeat: bool = False) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "selected_g.pt"
    checkpoint.write_bytes(b"exact selected forward checkpoint fixture")
    feature_path = tmp_path / "action_features.npz"
    mask_path = tmp_path / "action_masks.npz"
    roles = ["background", "downstream_envelope", "hub_slab", "near_turbine", "volume"]
    arrays = {}
    masks = {}
    rows = []
    exposure_by_action = {}
    for family, split in (("layout_1", "train_fit"), ("layout_2", "train_fit"), ("layout_3", "dev")):
        row_id = int(family[-1])
        for action_key, count, work in (("root", 1, 10.0), ("two_packet", 2, 20.0),
                                        ("four_packet", 4, 30.0), ("full_access", None, 40.0)):
            requested_paths = SELECTOR.ACTION_PATHS.get(action_key, ())
            realized_paths = list(requested_paths)
            if action_key != "full_access" and action_key not in exposure_by_action:
                row_path_map = {}
                for exposed_row in (1, 2):
                    row_path_map[str(exposed_row)] = [{
                        "realized_cut_paths": list(SELECTOR.ACTION_PATHS[action_key]),
                        "primary_pass_observations": [
                            {"primary_pass": 2, "realized_nonredundant_k": count},
                            {"primary_pass": 3, "realized_nonredundant_k": count},
                        ],
                        "primary_pass_ids": [2, 3],
                        "realized_nonredundant_k_counts": {str(count): 2},
                    }]
                exposure_by_action[action_key] = {
                    "completed_primary_action_passes": 2,
                    "completed_primary_pass_ids": [2, 3],
                    "required_complete_passes": 2,
                    "trained_action": True,
                    "training_row_count": 2,
                    "invalid_realized_cut_path_update_count": 0,
                    "invalid_realized_nonredundant_k_update_count": 0,
                    "invalid_realized_execution_update_count": 0,
                    "realized_cut_paths_by_primary_pass": {
                        str(pass_id): {
                            "eligible_row_count": 2, "distinct_row_count": 2,
                            "complete_primary_pass": True, "expected_row_count": 2,
                            "resolved_path_families": {
                                json.dumps(list(SELECTOR.ACTION_PATHS[action_key]), separators=(",", ":")): {
                                    "realized_nonredundant_k_counts": {str(count): 2}
                                }
                            },
                        } for pass_id in (2, 3)
                    },
                    "resolved_path_exposure_by_row": row_path_map,
                }
            panels = ("fixed", "query_repeat") if include_repeat and family == "layout_1" else ("fixed",)
            for query_panel in panels:
                prefix = f"{family}_{query_panel}_{action_key}"
                packet = f"packet_{prefix}"
                budget = f"budget_{prefix}"
                descriptors = f"roles_{prefix}"
                arrays[packet] = np.full((count or 1, 3), 0.1 + work / 100, dtype=np.float32)
                arrays[budget] = np.asarray([0.9, 0.95], dtype=np.float32)
                arrays[descriptors] = np.ones((5, 2), dtype=np.float32)
                masks[f"mask_{prefix}"] = np.ones((count or 1, 5), dtype=np.uint8)
                row_exposure = {
                    "action_key": action_key,
                    "required_complete_passes": 2,
                    "completed_primary_action_passes": 2,
                    "completed_primary_pass_ids": [2, 3],
                    "trained_action": action_key != "full_access",
                    "training_row_count": 2,
                    "invalid_realized_cut_path_update_count": 0,
                    "invalid_realized_nonredundant_k_update_count": 0,
                    "invalid_realized_execution_update_count": 0,
                    "resolved_path_exposure_for_case": (
                        exposure_by_action[action_key]["resolved_path_exposure_by_row"].get(str(row_id), [])
                        if action_key != "full_access" else []
                    ),
                    "same_realized_path_primary_pass_ids": (
                        [2, 3] if action_key != "full_access" and split == "train_fit" else []
                    ),
                    "same_realized_path_and_k_primary_pass_ids": (
                        [2, 3] if action_key != "full_access" and split == "train_fit" else []
                    ),
                }
                rows.append({
                    "split": split, "family_key": family, "case_key": f"{family}_{query_panel}",
                    "row_id": row_id, "layout_id": row_id,
                    "module_count": 10 + row_id,
                    "query_panel": query_panel,
                    "selector_primary_fit_row": query_panel == "fixed",
                    "query_repeat_overlap_count": (4 if query_panel == "query_repeat" else None),
                    "query_repeat_disjoint_from_fixed": (False if query_panel == "query_repeat" else None),
                    "action_key": action_key,
                    "forward_checkpoint_sha256": SELECTOR._sha256(checkpoint),
                    "full_access": action_key == "full_access",
                    "trained_sparse": action_key != "full_access",
                    "trained_action_after_two_complete_passes": action_key != "full_access",
                    "training_exposure": row_exposure,
                    "requested_cut_paths": list(requested_paths),
                    "realized_cut_paths": realized_paths,
                    "nonredundant_k": count, "exact_work": work, "full_work": 40.0,
                    "candidate_role_error": {role: 0.01 + work / 1000 for role in roles},
                    "incumbent_role_error": {role: 0.04 for role in roles},
                    "packet_rows_npz_key": packet, "budget_vector_npz_key": budget,
                    "receiver_role_features_npz_key": descriptors,
                })
    np.savez(feature_path, **arrays)
    np.savez(mask_path, **masks)
    table = {
        "format_version": 2,
        "run_id": "2112",
        "selected_update_count": 211,
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
        "action_exposure_by_key": exposure_by_action,
        "run_manifest_path": str((tmp_path / "run2112" / "run_manifest.json").resolve()),
        "g_checkpoint_binding_source": "append_only_scheduled_review",
        "g_checkpoint_binding_record_sha256": "b" * 64,
        "g_checkpoint_binding_record": {
            "arm": "g_packet", "update_count": 211,
            "checkpoint_sha256": SELECTOR._sha256(checkpoint),
            "checkpoint": str(checkpoint), "checkpoint_kind": "scheduled_review",
        },
        "g_checkpoint_review_record_line_sha256": "b" * 64,
        "g_checkpoint_review_record": {
            "arm": "g_packet", "update_count": 211,
            "checkpoint_sha256": SELECTOR._sha256(checkpoint),
            "checkpoint": str(checkpoint), "checkpoint_kind": "scheduled_review",
        },
        "g_action_exposure_lineage": {
            "update_ledger_path": str((tmp_path / "run2112" / "arms" / "g_packet" / "updates.jsonl").resolve()),
            "update_ledger_sha256": "c" * 64,
            "update_ledger_snapshot_bytes": 128,
            "update_ledger_snapshot_record_count": 2,
            "selected_update_count": 211,
        },
        "rows": rows,
    }
    table_path = tmp_path / "action_table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    return table_path, feature_path, mask_path, checkpoint


def test_action_fit_keeps_held_labels_out_of_risk_weights_and_gate(tmp_path: Path) -> None:
    paths = _table(tmp_path)
    rows, metadata = SELECTOR._load_action_table(*paths)
    assert "3" not in metadata["action_exposure_by_key"]["root"]["resolved_path_exposure_by_row"]
    assert all(row.trained_sparse for row in rows if row.family_key == "layout_3" and not row.full_access)
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


def test_selector_dev_layout_with_forward_training_exposure_requires_own_path_and_k(tmp_path: Path) -> None:
    table_path, features, masks, checkpoint = _table(tmp_path)
    table = json.loads(table_path.read_text(encoding="utf-8"))
    for action_key, exposure in table["action_exposure_by_key"].items():
        own_exposure = deepcopy(exposure["resolved_path_exposure_by_row"]["2"])
        exposure["resolved_path_exposure_by_row"]["3"] = own_exposure
        exposure["training_row_count"] = 3
        path_key = json.dumps(list(SELECTOR.ACTION_PATHS[action_key]), separators=(",", ":"))
        for pass_audit in exposure["realized_cut_paths_by_primary_pass"].values():
            for count_key in ("eligible_row_count", "distinct_row_count", "expected_row_count"):
                pass_audit[count_key] = 3
            pass_audit["resolved_path_families"][path_key]["realized_nonredundant_k_counts"][
                str(len(SELECTOR.ACTION_PATHS[action_key]))
            ] = 3
    for row in table["rows"]:
        row["source_partition"] = "native_training"
        lineage = row["training_exposure"]
        lineage["training_row_count"] = 3
        if row["split"] == "dev" and not row["full_access"]:
            lineage["resolved_path_exposure_for_case"] = table["action_exposure_by_key"][
                row["action_key"]
            ]["resolved_path_exposure_by_row"]["3"]
            lineage["same_realized_path_primary_pass_ids"] = [2, 3]
            lineage["same_realized_path_and_k_primary_pass_ids"] = [2, 3]
    table_path.write_text(json.dumps(table), encoding="utf-8")

    rows, metadata = SELECTOR._load_action_table(table_path, features, masks, checkpoint)
    assert metadata["split_families"]["dev"] == ("layout_3",)
    assert all(row.trained_sparse for row in rows if row.family_key == "layout_3" and not row.full_access)

    # Complete action passes elsewhere cannot qualify this physical training
    # row when its own execution did not realize the advertised packet K.
    dev_root = next(row for row in table["rows"] if row["split"] == "dev" and row["action_key"] == "root")
    own_root = table["action_exposure_by_key"]["root"]["resolved_path_exposure_by_row"]["3"]
    for observation in own_root[0]["primary_pass_observations"]:
        observation["realized_nonredundant_k"] = 2
    root_path_key = json.dumps([""], separators=(",", ":"))
    for pass_audit in table["action_exposure_by_key"]["root"]["realized_cut_paths_by_primary_pass"].values():
        k_counts = pass_audit["resolved_path_families"][root_path_key]["realized_nonredundant_k_counts"]
        k_counts.update({"1": 2, "2": 1})
    dev_root["training_exposure"]["same_realized_path_and_k_primary_pass_ids"] = []
    table_path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(ValueError, match="trained_sparse flag"):
        SELECTOR._load_action_table(table_path, features, masks, checkpoint)


def test_action_table_rejects_stale_checkpoint_and_changed_feature_bytes(tmp_path: Path) -> None:
    table, features, masks, checkpoint = _table(tmp_path)
    checkpoint.write_bytes(b"different forward weights")
    with pytest.raises(ValueError, match="checkpoint differs"):
        SELECTOR._load_action_table(table, features, masks, checkpoint)
    checkpoint.write_bytes(b"exact selected forward checkpoint fixture")
    arrays = dict(np.load(features, allow_pickle=False))
    arrays["packet_layout_1_fixed_root"][0, 0] = 99.0
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


def test_trained_sparse_flag_must_match_nested_path_k_exposure_and_strict_work(tmp_path: Path) -> None:
    table_path, features, masks, checkpoint = _table(tmp_path)
    table = json.loads(table_path.read_text(encoding="utf-8"))
    root = next(row for row in table["rows"] if row["action_key"] == "root")
    root["exact_work"] = root["full_work"]
    root["trained_sparse"] = True
    table_path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(ValueError, match="trained_sparse flag"):
        SELECTOR._load_action_table(table_path, features, masks, checkpoint)


def test_action_table_binds_exposure_snapshot_to_selected_g_endpoint(tmp_path: Path) -> None:
    table_path, features, masks, checkpoint = _table(tmp_path)
    table = json.loads(table_path.read_text(encoding="utf-8"))
    table["g_action_exposure_lineage"]["selected_update_count"] = 210
    table_path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(ValueError, match="selected G checkpoint binding, update, and exposure ledger snapshot"):
        SELECTOR._load_action_table(table_path, features, masks, checkpoint)


def test_legacy_v1_scheduled_review_table_remains_readable(tmp_path: Path) -> None:
    table_path, features, masks, checkpoint = _table(tmp_path)
    payload = json.loads(table_path.read_text(encoding="utf-8"))
    payload["format_version"] = 1
    payload["g_review_record"] = payload.pop("g_checkpoint_binding_record")
    payload["g_review_record_line_sha256"] = payload.pop("g_checkpoint_binding_record_sha256")
    payload.pop("g_checkpoint_binding_source")
    payload.pop("g_checkpoint_review_record")
    payload.pop("g_checkpoint_review_record_line_sha256")
    table_path.write_text(json.dumps(payload), encoding="utf-8")

    _rows, metadata = SELECTOR._load_action_table(table_path, features, masks, checkpoint)

    assert metadata["checkpoint_binding_source"] == "append_only_scheduled_review"
    assert metadata["checkpoint_binding_record"]["checkpoint_kind"] == "scheduled_review"


def test_durability_checkpoint_binding_is_explicit_and_not_a_review(tmp_path: Path) -> None:
    table_path, features, masks, checkpoint = _table(tmp_path)
    payload = json.loads(table_path.read_text(encoding="utf-8"))
    record = payload["g_checkpoint_binding_record"]
    record.update({"checkpoint_kind": "durability_only", "run_id": "2112"})
    payload["g_checkpoint_binding_source"] = "current_durability_checkpoint"
    payload["g_checkpoint_binding_record_sha256"] = "d" * 64
    payload["g_checkpoint_review_record"] = None
    payload["g_checkpoint_review_record_line_sha256"] = None
    table_path.write_text(json.dumps(payload), encoding="utf-8")

    _rows, metadata = SELECTOR._load_action_table(table_path, features, masks, checkpoint)

    assert metadata["checkpoint_binding_source"] == "current_durability_checkpoint"
    assert metadata["checkpoint_binding_record"]["checkpoint_kind"] == "durability_only"
    assert metadata["checkpoint_binding_record_sha256"] == "d" * 64

    payload["g_checkpoint_review_record"] = dict(record)
    payload["g_checkpoint_review_record_line_sha256"] = "d" * 64
    table_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit G checkpoint source"):
        SELECTOR._load_action_table(table_path, features, masks, checkpoint)


def test_policy_summary_counts_decisions_separately_from_candidate_mismatches() -> None:
    results = [
        SimpleNamespace(
            case_key="selected_bad", family_key="f1", selected_action_key="root",
            measured_oracle_action_key="two_packet", unsupported_at_budget=False,
            selected_supported_sparse=False, measured_adequate_sparse=("two_packet",),
            predicted_safe_sparse=("root",), false_safe_sparse=("root",),
            false_reject_sparse=("two_packet",), selected_work=10.0,
            measured_oracle_sparse_work=15.0, selected_nonredundant_k=1,
        ),
        SimpleNamespace(
            case_key="fallback_bad", family_key="f2", selected_action_key="full_access",
            measured_oracle_action_key="root", unsupported_at_budget=True,
            selected_supported_sparse=False, measured_adequate_sparse=("root",),
            predicted_safe_sparse=(), false_safe_sparse=(), false_reject_sparse=("root",),
            selected_work=40.0, measured_oracle_sparse_work=8.0,
            selected_nonredundant_k=None,
        ),
        SimpleNamespace(
            case_key="selected_unsupported", family_key="f3", selected_action_key="two_packet",
            measured_oracle_action_key=None, unsupported_at_budget=False,
            selected_supported_sparse=False, measured_adequate_sparse=(),
            predicted_safe_sparse=("root", "two_packet"),
            false_safe_sparse=("root", "two_packet"), false_reject_sparse=(),
            selected_work=20.0, measured_oracle_sparse_work=None,
            selected_nonredundant_k=2,
        ),
        SimpleNamespace(
            case_key="selected_good", family_key="f4", selected_action_key="root",
            measured_oracle_action_key="root", unsupported_at_budget=False,
            selected_supported_sparse=True, measured_adequate_sparse=("root",),
            predicted_safe_sparse=("root",), false_safe_sparse=(), false_reject_sparse=(),
            selected_work=8.0, measured_oracle_sparse_work=8.0,
            selected_nonredundant_k=1,
        ),
    ]
    full_rows = [
        SimpleNamespace(case_key=row.case_key, exact_work=40.0, full_access=True)
        for row in results
    ]
    metadata = {
        row.case_key: {"module_count": 10 + index}
        for index, row in enumerate(results)
    }

    summary = SELECTOR._policy_summary(results, full_rows, metadata)

    assert summary["false_safe_choice_count"] == 2
    assert summary["false_reject_choice_count"] == 1
    assert summary["candidate_false_safe_count"] == 3
    assert summary["candidate_false_reject_count"] == 2
    by_case = {row["case_key"]: row for row in summary["per_case"]}
    assert by_case["selected_bad"]["selected_false_safe"] is True
    assert by_case["fallback_bad"]["full_fallback_with_adequate_sparse"] is True
    assert by_case["selected_unsupported"]["selected_false_safe"] is True
    assert by_case["selected_good"]["selected_false_safe"] is False


def test_query_repeats_are_audited_but_do_not_enter_primary_fit(tmp_path: Path) -> None:
    rows, metadata = SELECTOR._load_action_table(*_table(tmp_path, include_repeat=True))
    original, model = SELECTOR.fit_and_evaluate(rows, metadata, updates=2, seed=19)
    poisoned = [
        replace(row, candidate_role_error=torch.full((5,), 0.8))
        if metadata["case_metadata"][row.case_key]["query_panel"] == "query_repeat" else row
        for row in rows
    ]
    changed, changed_model = SELECTOR.fit_and_evaluate(poisoned, metadata, updates=2, seed=19)
    assert model is not None and changed_model is not None
    assert original["primary_fit_action_row_count"] == 12
    assert original["query_repeat_action_row_count_excluded_from_primary_fit"] == 4
    assert original["query_repeat_audit"]["query_repeat_case_count"] == 1
    assert original["query_repeat_audit"]["overlapping_repeat_case_count"] == 1
    assert original["query_repeat_audit"]["used_for_primary_selector_fit"] is False
    diagnostic = original["query_repeat_diagnostic"]
    changed_diagnostic = changed["query_repeat_diagnostic"]
    assert diagnostic["status"] == "evaluated"
    assert diagnostic["used_for_primary_threshold_calibration"] is False
    assert diagnostic["overlap_cases"][0]["overlap_count"] == 4
    assert diagnostic["neural"]["case_count"] == 1
    assert diagnostic["neural"]["paired_fixed_repeat_stability"]["paired_case_count"] == 1
    assert diagnostic["neural"]["paired_fixed_repeat_stability"]["per_pair"][0][
        "query_repeat_overlap_count"
    ] == 4
    assert diagnostic["neural"]["per_case"][0]["measured_adequate_sparse"] != (
        changed_diagnostic["neural"]["per_case"][0]["measured_adequate_sparse"]
    )
    assert original["split_results"] == changed["split_results"]
    assert original["split_results"]["train_fit"]["case_count"] == 2
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, changed_model.state_dict()[key], rtol=0, atol=0)


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
