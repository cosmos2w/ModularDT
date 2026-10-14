from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import wind_native_reference_evaluate as evaluator


def test_endpoint_role_counts_come_from_actual_q8192_role_ids() -> None:
    names = ("volume", "hub_slab", "downstream_envelope", "near_turbine", "background")
    one_row = np.repeat(np.arange(len(names), dtype=np.int8), [1640, 1640, 1640, 1640, 1632])
    role_ids = np.stack((one_row, one_row.copy()))

    assert evaluator.endpoint_role_query_counts(
        role_ids, names, expected_query_count=8192
    ) == {
        "volume": 1640,
        "hub_slab": 1640,
        "downstream_envelope": 1640,
        "near_turbine": 1640,
        "background": 1632,
    }


def test_endpoint_role_counts_reject_row_mismatch_and_wrong_query_count() -> None:
    names = ("volume", "hub_slab", "downstream_envelope", "near_turbine", "background")
    row = np.repeat(np.arange(len(names), dtype=np.int8), [4, 3, 3, 3, 3])
    other_row = row.copy()
    other_row[0] = 1
    mismatched = np.stack((row, other_row))
    try:
        evaluator.endpoint_role_query_counts(mismatched, names, expected_query_count=16)
    except ValueError as error:
        assert "same role counts" in str(error)
    else:
        raise AssertionError("Endpoint metadata must reject inconsistent per-row role allocation.")

    try:
        evaluator.endpoint_role_query_counts(np.stack((row, row)), names, expected_query_count=15)
    except ValueError as error:
        assert "declared per-row query count" in str(error)
    else:
        raise AssertionError("Endpoint metadata must reject an incorrect declared query count.")


class _Run:
    def __init__(self) -> None:
        self.nx, self.ny, self.nz = 5, 3, 4
        self.shape_nxyz = (self.nx, self.ny, self.nz)
        self.x_m = np.asarray([0, 80, 160, 240, 320], dtype=np.float64)
        self.y_m = np.asarray([-80, 0, 80], dtype=np.float64)
        self.z_m = np.asarray([50, 60, 70, 90], dtype=np.float64)
        self.U = np.arange(self.nx * self.ny * self.nz * 3, dtype=np.float32).reshape(-1, 3)


def _case() -> SimpleNamespace:
    return SimpleNamespace(run=_Run(), hub_height_m=70.0, diameter_m=80.0)


def test_hub_plane_uses_exact_stored_z_and_native_c_order_ids() -> None:
    case = _case()
    plane = evaluator.native_hub_plane(case)
    expected = np.arange(2 * case.run.nx * case.run.ny, 3 * case.run.nx * case.run.ny).reshape(
        case.run.ny, case.run.nx
    )

    assert plane["z_index"] == 2
    assert plane["actual_z_m"] == 70.0
    np.testing.assert_array_equal(plane["flat_indices"], expected)
    np.testing.assert_array_equal(
        plane["target_mps"], case.run.U[expected.reshape(-1)].reshape(case.run.ny, case.run.nx, 3)
    )
    assert np.all(plane["coordinates_D"][..., 2] == 70.0 / 80.0)


def test_crosswind_cuts_start_from_stable_geometry_selected_upstream_source() -> None:
    case = _case()
    scene = SimpleNamespace(
        module_centers_D=np.asarray([[1.0, 0.0, 0.875], [0.0, 2.0, 0.875], [0.0, -2.0, 0.875]]),
        module_present=np.ones(3, dtype=np.float32),
        module_ids=np.asarray([7, 9, 2], dtype=np.int64),
    )
    cuts = evaluator.native_downstream_crosswind_cuts(case, scene)

    assert cuts["source_index"] == 2
    assert cuts["source_id"] == 2
    assert cuts["source_center_D"].tolist() == [0.0, -2.0, 0.875]
    assert [cut["requested_offset_D"] for cut in cuts["cuts"]] == [2.0, 4.0, 8.0]
    # The final requested station falls beyond the stored x support, so the
    # record must retain the exact nearest native station and its actual offset.
    assert [cut["x_index"] for cut in cuts["cuts"]] == [2, 4, 4]
    assert [cut["actual_offset_D"] for cut in cuts["cuts"]] == [2.0, 4.0, 4.0]
    for cut in cuts["cuts"]:
        assert np.all(cut["coordinates_D"][:, 0] == cut["actual_x_m"] / 80.0)
        assert np.all(cut["coordinates_D"][:, 2] == 70.0 / 80.0)
        np.testing.assert_array_equal(cut["target_mps"], case.run.U[cut["flat_indices"]])


def test_fixed_dev_row_selection_prefers_valid_extremes_then_stable_ties() -> None:
    rows = (10, 69, 35, 426, 90)
    counts = {10: 8, 69: 3, 35: 8, 426: 20, 90: 3}

    assert evaluator.select_low_high_development_rows(rows, counts) == {"low_M": 69, "high_M": 426}
    assert evaluator.select_low_high_development_rows((10, 35, 90), counts) == {"low_M": 90, "high_M": 35}


def test_role_summary_reports_row_and_layout_tails_with_worst_ids() -> None:
    def record(row: int, layout: int, value: float) -> dict[str, object]:
        return {
            "row_index": row,
            "layout_index": layout,
            "roles": {
                "volume": {
                    "component_rmse_mps": [value, value / 2, value / 4],
                    "vector_rmse_mps": value * 1.2,
                }
            },
        }

    result = evaluator.summarize_role_metrics((record(1, 4, 1.0), record(2, 4, 3.0), record(3, 5, 2.0)))
    volume = result["volume"]

    assert volume["worst_row_id_by_component"] == [2, 2, 2]
    assert volume["worst_vector_row_id"] == 2
    assert volume["layout_tail"]["layout_count"] == 2
    assert volume["layout_tail"]["worst_layout_index"] == 4
    assert volume["layout_tail"]["worst_layout_mean_vector_rmse_mps"] == 2.4


def test_named_checkpoint_arguments_reject_ambiguous_or_duplicate_labels(tmp_path: Path) -> None:
    first = tmp_path / "one.pt"
    second = tmp_path / "two.pt"

    parsed = evaluator.parse_named_paths((f"A_D={first}", f"A_H6={second}"), "--new-checkpoint")

    assert parsed == {"A_D": first.resolve(), "A_H6": second.resolve()}
    try:
        evaluator.parse_named_paths((f"A_D={first}", f"A_D={second}"), "--new-checkpoint")
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("duplicate model labels must fail before evaluation")


def test_cpu_cost_runner_measures_two_warmups_and_five_rotated_repeats() -> None:
    calls = {"A_D": 0, "A_H6": 0}

    def infer(label: str) -> np.ndarray:
        calls[label] += 1
        return np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    result, orders = evaluator._measure_alternating_cpu_cost_trials({
        "A_D": lambda: infer("A_D"),
        "A_H6": lambda: infer("A_H6"),
    })

    assert calls == {"A_D": 7, "A_H6": 7}
    assert all(item["repeat_count"] == 5 for item in result.values())
    assert all(item["warmup_count"] == 2 for item in result.values())
    assert all(item["all_predictions_bitwise_identical"] is True for item in result.values())
    assert all(len(item["repeat_seconds"]) == 5 for item in result.values())
    assert result["A_D"]["median_seconds"] > 0
    assert orders[0] == ["A_D", "A_H6"]
    assert orders[1] == ["A_H6", "A_D"]
    assert orders[2] == ["A_H6", "A_D"]
    assert orders[3] == ["A_D", "A_H6"]


def test_cpu_cost_summary_rejects_non_five_repeat_contract() -> None:
    try:
        evaluator.summarize_cpu_cost_trials((0.1, 0.11), (0.2, 0.3, 0.4))
    except ValueError as error:
        assert "exactly five" in str(error)
    else:
        raise AssertionError("CPU cost summaries must reject any repeat count other than five")


def test_cpu_cost_summary_rejects_wrong_warmup_count() -> None:
    try:
        evaluator.summarize_cpu_cost_trials((0.1,), (0.2, 0.3, 0.4, 0.5, 0.6))
    except ValueError as error:
        assert "exactly two" in str(error)
    else:
        raise AssertionError("CPU cost summaries must require the prespecified two warmups")


def _full_scope_fixture() -> tuple[dict, dict, np.ndarray, np.ndarray, dict]:
    train = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    validation = np.asarray([6, 7, 8], dtype=np.int64)
    test = np.asarray([9, 10, 11], dtype=np.int64)
    train_hash = evaluator._int64_indices_sha256(train)
    validation_hash = evaluator._int64_indices_sha256(validation)
    test_hash = evaluator._int64_indices_sha256(test)
    profile = "wind_full_validation_scope_fixture"
    manifest = {
        "subset_id": profile,
        "manifest_sha256": "fixture-manifest-sha",
        "train_row_indices": train.tolist(),
        "validation_row_indices": validation.tolist(),
        "test_row_indices": test.tolist(),
        "train_row_indices_sha256": train_hash,
        "validation_row_indices_sha256": validation_hash,
        "test_row_indices_sha256": test_hash,
        "train_layout_indices": [0, 1],
        "validation_layout_indices": [2],
        "test_layout_indices": [3],
        "test_row_count": 3,
        "full_train_followup1000": True,
        "wind_test_target_values_read": False,
    }
    identity = {
        "dataset_profile": profile,
        "subset_id": profile,
        "protocol": "full_train1000",
        "subset_manifest_sha256": manifest["manifest_sha256"],
        "train_rows": train.tolist(),
        "validation_rows": validation.tolist(),
        "train_row_indices_sha256": train_hash,
        "validation_row_indices_sha256": validation_hash,
        "test_row_indices_sha256": test_hash,
        "train_layout_indices": manifest["train_layout_indices"],
        "validation_layout_indices": manifest["validation_layout_indices"],
        "test_layout_indices": manifest["test_layout_indices"],
        "recipe": {"protocol": "full_train1000", "approved_stop_after": 1000},
        "architecture": "A-H6",
        "model_config": {"architecture": "A-H6"},
        "training": {"total_epochs": 5000},
        "normalizer_and_background_source": {
            "kind": "verified_full_train_followup_train_fit_cache",
            "training_rows_sha256": train_hash,
        },
        "wind_test_target_values_read": False,
        "unexposed_test_target_values_read": False,
    }
    expected = {
        "profile_id": profile,
        "protocol": "full_train1000",
        "train_count": 6,
        "validation_count": 3,
        "test_count": 3,
        "train_layout_count": 2,
        "validation_layout_count": 1,
        "test_layout_count": 1,
        "total_row_count": 12,
        "train_rows_sha256": train_hash,
        "validation_rows_sha256": validation_hash,
        "test_rows_sha256": test_hash,
    }
    return identity, manifest, train, validation, expected


def test_full_validation_scope_guard_checks_all_partitions_without_targets() -> None:
    identity, manifest, train, validation, expected = _full_scope_fixture()

    evaluator._validate_full_train1000_provider_payload(
        identity, manifest, train, validation, expected_contract=expected
    )


def test_full_validation_scope_binds_v2_recipe_to_organized_v2_architecture() -> None:
    identity, manifest, train, validation, expected = _full_scope_fixture()
    identity = dict(identity, architecture="A2-H6",
                    recipe=dict(identity["recipe"], schema="content_hypergraph_recipe_v2"),
                    model_config=dict(identity["model_config"], architecture="A2-H6"))
    evaluator._validate_full_train1000_provider_payload(
        identity, manifest, train, validation, expected_contract=expected
    )
    mismatched = dict(identity, model_config=dict(identity["model_config"], architecture="A-H6"))
    try:
        evaluator._validate_full_train1000_provider_payload(
            mismatched, manifest, train, validation, expected_contract=expected
        )
    except ValueError as error:
        assert "full_train1000 model" in str(error)
    else:
        raise AssertionError("A v2 recipe cannot qualify a v1 native model.")


def test_full_provider_guard_passes_only_the_metadata_fixture(monkeypatch) -> None:
    identity, manifest, train, validation, expected = _full_scope_fixture()
    provider = SimpleNamespace(
        identity_payload=lambda: identity,
        manifest=manifest,
        train_rows=train,
        validation_rows=validation,
    )
    monkeypatch.setattr(evaluator, "_full_followup_scope_contract", lambda: expected)

    assert evaluator._assert_full_train1000_provider(provider) is identity


def test_full_validation_scope_contract_is_the_original_seed42_partition() -> None:
    contract = evaluator._full_followup_scope_contract()

    assert contract["profile_id"] == "wind_original420_train_fullVALID90_followup1000_v1"
    assert (contract["train_count"], contract["validation_count"], contract["test_count"]) == (420, 90, 90)
    assert (contract["train_layout_count"], contract["validation_layout_count"], contract["test_layout_count"]) == (140, 30, 30)
    assert contract["train_rows_sha256"] == "a2170bb15349462160679632fc074d7f685d7e5d17d09aeeab1797a824c8a282"
    assert contract["validation_rows_sha256"] == "ed7295dd2650687c599c13e29a6bae3492508ec8346517e39f4cacf60d0eff72"
    assert contract["test_rows_sha256"] == "e11734b05e51691680fcae088ef07be8608951660eab9761d05dc29fe2fef998"


def test_full_validation_scope_guard_rejects_profile_or_membership_mismatch() -> None:
    identity, manifest, train, validation, expected = _full_scope_fixture()
    wrong_identity = dict(identity, dataset_profile="wind_shared_fixed24_v1")
    try:
        evaluator._validate_full_train1000_provider_payload(
            wrong_identity, manifest, train, validation, expected_contract=expected
        )
    except ValueError as error:
        assert "profile" in str(error) or "manifest" in str(error)
    else:
        raise AssertionError("Full VALID90 scope guard must reject a fixed DEV24 identity")

    wrong_manifest = dict(manifest, test_row_indices=[9, 10, 12])
    try:
        evaluator._validate_full_train1000_provider_payload(
            identity, wrong_manifest, train, validation, expected_contract=expected
        )
    except ValueError as error:
        assert "TEST metadata" in str(error) or "memberships" in str(error)
    else:
        raise AssertionError("Full VALID90 scope guard must reject altered locked TEST membership")


def test_full_validation_scope_guard_rejects_any_test_target_access_claim() -> None:
    identity, manifest, train, validation, expected = _full_scope_fixture()
    accessed_identity = dict(identity, unexposed_test_target_values_read=True)
    try:
        evaluator._validate_full_train1000_provider_payload(
            accessed_identity, manifest, train, validation, expected_contract=expected
        )
    except ValueError as error:
        assert "target values" in str(error)
    else:
        raise AssertionError("Full VALID90 scope guard must preserve the WindTEST target lock")
