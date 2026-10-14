from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCore, QueryBatch, SceneBatch
from tools.content_hypergraph_inspect import (
    _condition_trace_arrays,
    _fp64_identity_error,
    _select_thermal_cases,
    _select_wind_rows,
    _validate_checkpoint_scope,
    _validate_thermal_full_binding,
    _validate_wind_full_binding,
    _validation_panel_metadata,
    local_branch_zero_hook,
    rank_thermal_edge,
    rank_wind_edge,
    removed_edge_gate,
    select_thermal_probes,
    select_wind_probes,
    slice_query,
)


def _scene() -> SceneBatch:
    return SceneBatch(
        module_features=torch.tensor([[[0.2, 0.4], [0.7, -0.3]]]),
        environment_features=torch.tensor([[[0.1, 0.2, 0.3], [0.4, -0.2, 0.5], [-0.5, 0.7, 0.1]]]),
        module_coordinates=torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
        environment_coordinates=torch.tensor([[[-0.1, 0.3], [0.5, 0.7], [1.1, 0.2]]]),
        module_mask=torch.ones((1, 2), dtype=torch.bool),
        environment_mask=torch.ones((1, 3), dtype=torch.bool),
        module_measure=torch.tensor([[0.7, 1.1]]),
        environment_measure=torch.tensor([[0.2, 0.4, 0.3]]),
        module_ids=torch.tensor([[13, 29]]),
        environment_ids=torch.tensor([[101, 107, 131]]),
        local_radius=torch.full((1, 2), 0.8),
        operating_features=torch.tensor([[0.3, -0.2]]),
        module_measure_reference=torch.tensor([1.0]),
        environment_measure_reference=torch.tensor([0.3]),
        scene_key="inspection-scene",
        normalizer_key="inspection-normalizer",
        coordinate_frame_key="inspection-frame",
    )


def _query() -> QueryBatch:
    return QueryBatch(
        coordinates=torch.tensor([[[0.1, 0.0], [0.5, 0.08], [1.1, 0.1]]]),
        features=torch.tensor([[[0.1, 0.5], [-0.3, 0.7], [0.4, -0.1]]]),
        query_ids=torch.tensor([[3, 5, 8]]),
        scene_key="inspection-scene",
        coordinate_frame_key="inspection-frame",
    )


def _core(thermal_rank: int) -> ContentHypergraphCore:
    return ContentHypergraphCore(
        module_input_dim=2,
        environment_input_dim=3,
        coordinate_dim=2,
        query_input_dim=2,
        operating_input_dim=2,
        output_dim=3,
        architecture="A-H6",
        context_dim=16,
        num_heads=4,
        context_depth=1,
        num_hyperedges=6,
        thermal_response_rank=thermal_rank,
        local_hidden_dim=8,
        init_seed=42,
    ).eval()


def test_thermal_edge_rank_uses_signed_field_contributions_not_attention_mass():
    contributions = np.zeros((3, 6, 2), dtype=np.float64)
    contributions[:, 0, :] = 0.1
    contributions[:, 2, :] = -1.5
    selected, scores = rank_thermal_edge(contributions)

    assert selected == 2
    assert scores[2] > scores[0]


def test_wind_edge_rank_uses_weighted_messages_not_receiver_occupancy():
    access = np.asarray([[0.95, 0.05, 0.0], [0.95, 0.05, 0.0]])
    messages = np.zeros((2, 3, 4), dtype=np.float64)
    messages[:, 0] = 0.01
    messages[:, 1] = 8.0
    selected, scores = rank_wind_edge(access, messages)

    assert selected == 1
    assert scores[1] > scores[0]


def _thermal_dev_provider(case_ids: tuple[str, ...]) -> SimpleNamespace:
    identity = {
        "protocol": "development",
        "validation_scope": "fixed25_v1_DEV22_exposed",
        "dataset_profile": "fixed25_v1",
        "validation_case_ids": list(case_ids),
        "validation_case_ids_sha256": "ids-digest",
        "subset_manifest_sha256": "fixed-manifest",
        "base_fixed25_manifest_sha256": "fixed-manifest",
        "source_binding": {
            "dataset_id": "thermal_channel_global_v1",
            "catalog_metadata_sha256": "catalog-digest",
            "base_manifest_sha256": "fixed-manifest",
            "protocol": "development",
            "normalization_fit_scope": "selected original TRAIN150 only",
            "original_test_targets_outside_declared_panel_read": False,
        },
    }
    return SimpleNamespace(
        protocol="development",
        validation_case_ids=case_ids,
        manifest={"manifest_sha256": "fixed-manifest"},
        base_manifest_sha256="fixed-manifest",
        identity_payload=lambda: identity,
    )


def test_thermal_inspection_selects_only_bound_exposed_dev_ids_without_source_split_heuristic():
    case_ids = ("0291", "0687", *(f"dev-{index:02d}" for index in range(20)))
    provider = _thermal_dev_provider(case_ids)
    with (
        patch("channelthermal.data.development_split.development_case_ids", return_value=case_ids),
        patch("channelthermal.training.unified_task.FIXED25_FINGERPRINT", "fixed-manifest"),
    ):
        assert _select_thermal_cases(provider) == ("0291", "0687")


def test_thermal_inspection_rejects_cases_outside_the_fixed_dev_panel():
    case_ids = ("0291", *(f"dev-{index:02d}" for index in range(21)))
    provider = _thermal_dev_provider(case_ids)
    with (
        patch("channelthermal.data.development_split.development_case_ids", return_value=case_ids),
        patch("channelthermal.training.unified_task.FIXED25_FINGERPRINT", "fixed-manifest"),
    ):
        try:
            _select_thermal_cases(provider)
        except ValueError as error:
            assert "0291 and 0687" in str(error)
        else:
            raise AssertionError("Missing fixed Thermal DEV case was accepted.")


def _thermal_full_provider() -> tuple[SimpleNamespace, dict[str, object]]:
    from channelthermal.training.content_hypergraph_task import (
        THERMAL_CONTENT_FULL_TRAIN_DATASET,
        THERMAL_CONTENT_FULL_TRAIN_PROTOCOL,
    )

    training_ids = tuple(f"train-{index:03d}" for index in range(600))
    validation_ids = ("0291", "0687", *(f"valid-{index:03d}" for index in range(87)))
    membership: dict[str, object] = {
        "training_case_count": 600,
        "training_membership_sha256": "train-membership",
        "validation_case_count": 89,
        "validation_membership_sha256": "validation-membership",
        "validation_targets_are_original_h5_test_split_exposed": True,
        "unexposed_original_test_targets_read": False,
    }
    source_binding = {
        "protocol": "full_train1000",
        "normalization_fit_scope": "exact original TRAIN600 only",
        "original_test_targets_outside_declared_panel_read": False,
        "base_manifest_sha256": "fixed-manifest",
        "full_followup_membership": dict(membership),
    }
    followup_manifest = {
        "execution_protocol": THERMAL_CONTENT_FULL_TRAIN_PROTOCOL,
        "dataset_protocol": THERMAL_CONTENT_FULL_TRAIN_DATASET,
        "base_manifest_sha256": "fixed-manifest",
        "training_case_ids": list(training_ids),
        "training_membership_sha256": membership["training_membership_sha256"],
        "validation_case_ids": list(validation_ids),
        "validation_membership_sha256": membership["validation_membership_sha256"],
        "excluded_validation_id": "0273",
    }
    identity = {
        "protocol": "full_train1000",
        "dataset_profile": THERMAL_CONTENT_FULL_TRAIN_DATASET,
        "validation_scope": "canonical89_exposed_original_test_excluding_0273",
        "training_case_ids": list(training_ids),
        "training_case_ids_sha256": membership["training_membership_sha256"],
        "validation_case_ids": list(validation_ids),
        "validation_case_ids_sha256": membership["validation_membership_sha256"],
        "subset_manifest_sha256": "followup-manifest",
        "base_fixed25_manifest_sha256": "fixed-manifest",
        "source_binding": source_binding,
    }
    provider = SimpleNamespace(
        protocol="full_train1000",
        training_case_ids=training_ids,
        validation_case_ids=validation_ids,
        manifest={"manifest_sha256": "followup-manifest", "full_train_followup1000": followup_manifest},
        base_manifest_sha256="fixed-manifest",
        identity_payload=lambda: identity,
    )
    return provider, membership


def _thermal_full_contract_patches(provider: SimpleNamespace, membership: dict[str, object]):
    from channelthermal.training.content_hypergraph_task import THERMAL_CONTENT_FULL_TRAIN_DATASET

    return (
        patch(
            "channelthermal.training.joint_task.validate_full_train_followup_membership",
            return_value=membership,
        ),
        patch("channelthermal.training.unified_task.FIXED25_FINGERPRINT", "fixed-manifest"),
        patch(
            "channelthermal.training.content_hypergraph_task.THERMAL_CONTENT_FULL_TRAIN_DATASET",
            THERMAL_CONTENT_FULL_TRAIN_DATASET,
        ),
    )


def test_thermal_full_followup_binding_and_case_selection_use_metadata_only():
    provider, membership = _thermal_full_provider()
    patches = _thermal_full_contract_patches(provider, membership)
    with patches[0], patches[1], patches[2]:
        binding = _validate_thermal_full_binding(provider)
        assert binding["protocol"] == "full_train1000"
        assert binding["validation_scope"] == "canonical89_exposed_original_test_excluding_0273"
        assert binding["normalization_fit_scope"] == "exact original TRAIN600 only"
        assert binding["full_followup_membership"] == membership
        assert binding["unexposed_original_test_targets_read"] is False
        assert _select_thermal_cases(provider, "full_train1000") == ("0291", "0687")


@pytest.mark.parametrize(
    "mutate",
    (
        lambda provider: provider.identity_payload()["source_binding"].update(
            normalization_fit_scope="selected original TRAIN150 only"
        ),
        lambda provider: provider.identity_payload()["source_binding"].update(
            original_test_targets_outside_declared_panel_read=True
        ),
        lambda provider: provider.identity_payload().update(base_fixed25_manifest_sha256="wrong-base"),
        lambda provider: provider.identity_payload()["source_binding"]["full_followup_membership"].update(
            unexposed_original_test_targets_read=True
        ),
    ),
)
def test_thermal_full_followup_rejects_scope_binding_or_target_lock_drift(mutate):
    provider, membership = _thermal_full_provider()
    mutate(provider)
    patches = _thermal_full_contract_patches(provider, membership)
    with patches[0], patches[1], patches[2], pytest.raises(ValueError, match="TRAIN600/canonical89"):
        _validate_thermal_full_binding(provider)


def test_thermal_full_followup_falls_back_to_bound_ids_when_dev_representatives_are_absent():
    provider, membership = _thermal_full_provider()
    provider.validation_case_ids = tuple(f"full-{index:02d}" for index in range(89))
    identity = provider.identity_payload()
    identity["validation_case_ids"] = list(provider.validation_case_ids)
    followup = provider.manifest["full_train_followup1000"]
    followup["validation_case_ids"] = list(provider.validation_case_ids)
    patches = _thermal_full_contract_patches(provider, membership)
    with patches[0], patches[1], patches[2]:
        assert _select_thermal_cases(provider, "full_train1000") == ("full-00", "full-01")


def test_wind_full_followup_selection_reuses_the_strict_provider_guard():
    rows = np.asarray([69, 426, 10], dtype=np.int64)
    counts = np.zeros(427, dtype=np.int64)
    counts[69], counts[426], counts[10] = 3, 30, 8
    identity = {
        "protocol": "full_train1000",
        "validation_rows": rows.tolist(),
        "validation_row_indices_sha256": "valid90-membership",
        "subset_manifest_sha256": "full-manifest",
        "dataset_profile": "wind_full_profile",
        "unexposed_test_target_values_read": False,
        "wind_test_target_values_read": False,
    }
    provider = SimpleNamespace(
        recipe={"protocol": "full_train1000"},
        validation_rows=rows,
        view=SimpleNamespace(metadata={"n_turbines": counts}),
    )
    with patch(
        "wind_native_reference_evaluate._assert_full_train1000_provider", return_value=identity
    ) as strict_guard:
        binding = _validate_wind_full_binding(provider)
        assert binding["validation_scope"] == "seed42_full_train1000_VALID90_exposed"
        assert binding["validation_row_indices_sha256"] == "valid90-membership"
        assert binding["unexposed_test_target_values_read"] is False
        assert _select_wind_rows(provider, "full_train1000") == (69, 426)
        assert strict_guard.call_count == 2


@pytest.mark.parametrize("lock_field", ("wind_test_target_values_read", "unexposed_test_target_values_read"))
def test_wind_full_followup_inspection_rejects_test_target_lock_drift(lock_field):
    rows = np.asarray([69, 426], dtype=np.int64)
    identity = {
        "protocol": "full_train1000",
        "validation_rows": rows.tolist(),
        "validation_row_indices_sha256": "valid90-membership",
        "subset_manifest_sha256": "full-manifest",
        "dataset_profile": "wind_full_profile",
        "wind_test_target_values_read": False,
        "unexposed_test_target_values_read": False,
    }
    identity[lock_field] = True
    provider = SimpleNamespace(recipe={"protocol": "full_train1000"}, validation_rows=rows)
    with (
        patch("wind_native_reference_evaluate._assert_full_train1000_provider", return_value=identity),
        pytest.raises(ValueError, match="strict full VALID90 metadata check"),
    ):
        _validate_wind_full_binding(provider)


def test_full_panel_metadata_labels_exposed_validation_and_test_lock_separately():
    thermal_binding = {
        "validation_scope": "canonical89_exposed_original_test_excluding_0273",
        "validation_case_ids": ["0291", "0687", *(f"case-{i}" for i in range(87))],
        "validation_case_ids_sha256": "thermal-valid89",
    }
    thermal = _validation_panel_metadata("thermal", "full_train1000", thermal_binding, ("0291", "0687"))
    assert thermal["bound_validation_count"] == 89
    assert thermal["exposed_validation_target_values_read_for_metrics"] is True
    assert thermal["unexposed_original_test_targets_read"] is False
    assert "DEV22" not in thermal["scope"]

    wind_binding = {
        "validation_scope": "seed42_full_train1000_VALID90_exposed",
        "validation_rows": list(range(90)),
        "validation_row_indices_sha256": "wind-valid90",
    }
    wind = _validation_panel_metadata("wind", "full_train1000", wind_binding, (69, 426))
    assert wind["bound_validation_count"] == 90
    assert wind["exposed_validation_target_values_read_for_metrics"] is True
    assert wind["wind_test_target_values_read"] is False
    assert wind["source_partition"] == "original_full_train1000_VALID90_exposed"


def test_full_followup_inspection_requires_literal_approved_epoch_1000():
    recipe = {"protocol": "full_train1000", "approved_stop_after": 1000, "architecture": "A-H6"}
    assert _validate_checkpoint_scope("thermal", recipe, {"epoch": 1000}) == "full_train1000"
    for bad_recipe, bad_payload in (
        (recipe, {"epoch": 999}),
        ({**recipe, "approved_stop_after": 2000}, {"epoch": 1000}),
        ({**recipe, "architecture": "A-D"}, {"epoch": 1000}),
        ({**recipe, "approved_stop_after": "1000"}, {"epoch": 1000}),
    ):
        with pytest.raises(ValueError, match="literal approved epoch 1000"):
            _validate_checkpoint_scope("wind", bad_recipe, bad_payload)


def test_fp64_native_identity_uses_frozen_software_tolerance_and_reports_raw_error():
    actual = torch.tensor([1.0, -0.125], dtype=torch.float64)
    expected = torch.tensor([1.0 + 2.0e-15, -0.125], dtype=torch.float64)

    result = _fp64_identity_error(actual, expected)

    assert result["comparison_dtype"] == "float64"
    assert np.isclose(result["max_abs_error"], 2.0e-15, rtol=1.0e-3)
    assert result["software_tolerance"] == {"atol": 1.0e-9, "rtol": 1.0e-7}
    assert result["max_abs_software_tolerance"] >= 1.0e-9
    assert result["passed"]


def test_local_zero_keeps_pair_work_and_edge_access_but_removes_exact_local_heat():
    core = _core(thermal_rank=2)
    organization = core.prepare(_scene())
    query = _query()
    controls = torch.tensor([[0.7, -0.4]])
    full = core.decode(organization, query, controls, return_trace=True)

    with local_branch_zero_hook(core):
        local_zero = core.decode(organization, query, controls, return_trace=True)

    trace = full.trace
    pair_batch = trace["local_pair_batch_indices"]
    pair_query = trace["local_pair_query_indices"]
    pair_module = trace["local_pair_module_indices"]
    local_temperature = trace["local_pair_support_weights"] * trace["local_temperature_coefficients"] * controls[
        pair_batch, pair_module
    ]
    expected = torch.zeros_like(full.thermal_temperature)
    expected.index_put_((pair_batch, pair_query), local_temperature, accumulate=True)

    assert full.work_counts["local_rows"] > 0
    assert local_zero.work_counts["local_rows"] == full.work_counts["local_rows"]
    torch.testing.assert_close(
        local_zero.trace["local_pair_query_indices"], full.trace["local_pair_query_indices"]
    )
    torch.testing.assert_close(
        local_zero.trace["local_pair_module_indices"], full.trace["local_pair_module_indices"]
    )
    torch.testing.assert_close(full.trace["edge_used_access"], local_zero.trace["edge_used_access"])
    torch.testing.assert_close(full.thermal_temperature - local_zero.thermal_temperature, expected)
    assert not torch.allclose(full.nonlinear_field, local_zero.nonlinear_field)


def test_removed_edge_keeps_other_access_weights_and_thermal_delta_is_factor_exact():
    core = _core(thermal_rank=2)
    organization = core.prepare(_scene())
    query = _query()
    controls = torch.tensor([[0.7, -0.4]])
    full = core.decode(organization, query, controls, return_trace=True)
    edge_index, _ = rank_thermal_edge(full.trace["edge_temperature_contributions"][0].sum(dim=-1)[:, :, None])
    gates = removed_edge_gate(1, organization.num_hyperedges, edge_index, device="cpu", dtype=torch.float32)
    removed = core.decode(organization, query, controls, edge_gates=gates, return_trace=True)

    expected_access = full.trace["edge_base_access"].clone()
    expected_access[:, :, edge_index] = 0.0
    expected_delta = full.trace["edge_temperature_contributions"][:, :, edge_index].sum(dim=-1)
    torch.testing.assert_close(removed.trace["edge_used_access"], expected_access)
    torch.testing.assert_close(full.thermal_temperature - removed.thermal_temperature, expected_delta)


def test_wind_edge_gate_recomputes_nonlinear_field_with_unrenormalized_access():
    core = _core(thermal_rank=0)
    organization = core.prepare(_scene())
    query = _query()
    full = core.decode(organization, query, return_trace=True)
    edge_index, _ = rank_wind_edge(
        full.trace["edge_used_access"][0],
        full.trace["edge_messages"][0],
    )
    gates = removed_edge_gate(1, organization.num_hyperedges, edge_index, device="cpu", dtype=torch.float32)
    removed = core.decode(organization, query, edge_gates=gates, return_trace=True)
    expected_access = full.trace["edge_base_access"].clone()
    expected_access[:, :, edge_index] = 0.0

    torch.testing.assert_close(removed.trace["edge_used_access"], expected_access)
    assert removed.work_counts["edge_rows"] == full.work_counts["edge_rows"]
    assert not torch.allclose(removed.nonlinear_field, full.nonlinear_field)


def test_query_subset_remaps_packed_pairs_and_preserves_source_kernel_rows():
    core = _core(thermal_rank=2)
    organization = core.prepare(_scene())
    original = QueryBatch(
        coordinates=torch.tensor([[[0.1, 0.0], [0.5, 0.08], [1.1, 0.1]]]),
        features=torch.tensor([[[0.1, 0.5], [-0.3, 0.7], [0.4, -0.1]]]),
        query_ids=torch.tensor([[3, 5, 8]]),
        local_batch_indices=torch.tensor([0, 0, 0]),
        local_query_indices=torch.tensor([0, 1, 2]),
        local_module_indices=torch.tensor([0, 0, 1]),
        local_weights=torch.tensor([0.7, 0.4, 0.8]),
        scene_key="inspection-scene",
        coordinate_frame_key="inspection-frame",
    )
    full_kernels = core.export_source_kernels(organization, original)
    subset = slice_query(original, [2, 0])
    subset_kernels = core.export_source_kernels(organization, subset)

    assert subset.query_ids.tolist() == [[8, 3]]
    assert subset.local_query_indices.tolist() == [1, 0]
    torch.testing.assert_close(subset_kernels, full_kernels[:, [2, 0]])


def test_trace_export_keeps_edge_gate_axis_and_slices_per_query_rows():
    core = _core(thermal_rank=2)
    organization = core.prepare(_scene())
    output = core.decode(organization, _query(), torch.tensor([[0.7, -0.4]]), return_trace=True)
    arrays = {}
    _condition_trace_arrays(arrays, "fixture", output, np.asarray([0, 2]))

    assert arrays["fixture__edge_gates"].shape == (1, 6)
    assert arrays["fixture__edge_access"].shape == (1, 2, 6)
    assert arrays["fixture__edge_messages"].shape == (1, 2, 6, 8)


def test_probe_selection_uses_only_native_geometry_and_role_membership():
    thermal_points = np.asarray([[0.2, 0.0], [0.8, 0.0], [3.0, 0.0], [-0.2, 0.0]])
    thermal = select_thermal_probes(
        thermal_points,
        np.arange(4),
        np.ones(4, dtype=bool),
        np.asarray([[0.0, 0.0], [0.7, 0.0]]),
        np.asarray([0.2, 0.2]),
        np.asarray([0.9, 0.9]),
    )
    assert thermal == {"near_body": 0, "downstream_overlap": 1, "background": 2}

    wind_points = np.asarray(
        [[0.0, 0.49, 0.875], [1.0, 0.0, 0.875], [1.1, 0.0, 0.875], [20.0, 10.0, 2.0]]
    )
    wind = select_wind_probes(
        wind_points,
        np.asarray([10, 20, 21, 99]),
        np.asarray([3, 2, 2, 4]),
        np.asarray([[0.0, 0.0, 0.875], [1.0, 0.0, 0.875]]),
        np.asarray([0.5, 0.5]),
    )
    assert wind == {"near_body": 0, "downstream_overlap": 2, "background": 3}
