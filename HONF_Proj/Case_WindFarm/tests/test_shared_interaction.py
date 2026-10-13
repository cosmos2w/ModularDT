from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from honf_forward_core.interface_fields.interaction_core import NonlinearFieldReadout
from honf_forward_core.interface_fields.interaction_refinement import RefinedNonlinearFieldReadout

from windfarm.geometry import environment_representation, global_geometry_features, support_geometry
from windfarm.interaction_preserving import (
    InteractionPreservingWindModel,
    load_interaction_preserving_wind_checkpoint,
)
from windfarm.joint_regional import WindFarmJointRegionalModel, wind_receiver_features
from windfarm.model import (
    REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE,
    SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE,
    build_windfarm_model,
)
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.shared_interaction import (
    WIND_INTERACTION_DEPENDENCY,
    WindFarmRefinedInteractionModel,
    WindFarmSharedInteractionModel,
    wind_scene_from_case,
)
from windfarm.training.unified_task import _as_scene_batch, _scene_inputs


def _case() -> SimpleNamespace:
    support = support_geometry(
        np.asarray([-400.0, 0.0, 800.0], dtype=np.float32),
        np.asarray([-320.0, 0.0, 320.0], dtype=np.float32),
        np.asarray([20.0, 70.0, 130.0], dtype=np.float32),
    )
    environment = environment_representation(support, token_shape=(2, 2, 2))
    centers = np.asarray([[0.0, 0.0, 0.875], [3.0, 0.5, 0.875]], dtype=np.float32)
    return SimpleNamespace(
        case="gen_0000_wd270",
        layout="layout-0",
        layout_index=0,
        wind_direction_deg=270.0,
        n_turbines=2,
        support=support,
        module_centers=centers,
        module_present=np.ones(2, dtype=np.float32),
        module_features=np.asarray([[0.5, 0.875], [0.5, 0.875]], dtype=np.float32),
        global_context=global_geometry_features(support, 270.0, 2),
        env_coords=environment.coords_D,
        env_features=environment.features,
        env_weights=environment.weights_D3,
    )


def _normalizer() -> VelocityNormalizer:
    return VelocityNormalizer(
        mean=np.asarray([1.0, 0.0, 0.0]),
        std=np.asarray([0.2, 0.1, 0.1]),
        safe_std=np.asarray([0.2, 0.1, 0.1]),
        source_rows=3,
    )


def _profile() -> VerticalProfileBaseline:
    return VerticalProfileBaseline(
        bin_centers_D=np.asarray([0.0, 2.0], dtype=np.float64),
        values_mps=np.asarray([[4.0, 0.1, -0.2], [12.0, 0.5, 0.2]], dtype=np.float64),
        counts=np.asarray([100, 100], dtype=np.int64),
        z_min_D=0.0,
        z_max_D=2.0,
    )


def test_wind_adapter_executes_shared_three_dimensional_nonlinear_readout() -> None:
    torch.manual_seed(42)
    case = _case()
    model = WindFarmSharedInteractionModel(velocity_transform=_normalizer(), hidden=16, message=12)
    assert isinstance(model.core, NonlinearFieldReadout)
    assert model.core.spatial_dim == 3
    assert model.core.output_width == 3

    scene = wind_scene_from_case(case)
    assert scene.dependency == WIND_INTERACTION_DEPENDENCY
    assert scene.dependency.dataset == "WindFarm"
    assert scene.dependency.output_law == "nonlinear"
    assert scene.dependency.units == ("m/s", "m/s", "m/s")
    assert "target_field" not in scene.tensors()
    assert scene.centers.shape == (1, 2, 3)
    assert scene.environment_coords.shape == (1, 8, 3)
    torch.testing.assert_close(scene.source_lengths, torch.ones_like(scene.present))

    prepared = model.prepare_case(case)
    receivers = torch.tensor([[[0.5, 0.2, 0.875], [4.0, 1.0, 1.1]]])
    prediction = model.predict_physical_case(case, prepared, receivers, chunk_size=1)
    assert prediction.shape == (1, 2, 3)
    assert torch.isfinite(prediction).all()

    with pytest.raises(ValueError, match="no exact affine finite-increment"):
        model.apply_increment(prepared, torch.zeros((1, 2)))


def test_native_model_factory_has_an_explicit_shared_core_family() -> None:
    model = build_windfarm_model(
        {
            "forward_architecture": SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE,
            "hidden": 16,
            "message": 12,
            "max_sources": 8,
        },
        velocity_transform=_normalizer(),
    )
    assert isinstance(model, WindFarmSharedInteractionModel)
    assert isinstance(model.core, NonlinearFieldReadout)
    with pytest.raises(ValueError, match="requires a train-only velocity transform"):
        build_windfarm_model({"forward_architecture": SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE})
    with pytest.raises(ValueError, match="Unsupported .* recipe fields"):
        build_windfarm_model(
            {"forward_architecture": SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE, "field_dim": 3},
            velocity_transform=_normalizer(),
        )


def test_refined_factory_uses_real_refined_nonlinear_core_and_fixed_profile() -> None:
    model = build_windfarm_model(
        {
            "forward_architecture": REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE,
            "hidden": 16,
            "message": 12,
            "max_sources": 8,
            "base_width": 16,
            "router_hidden": 8,
        },
        velocity_transform=_normalizer(),
        background_profile=_profile(),
    )
    assert isinstance(model, WindFarmRefinedInteractionModel)
    assert isinstance(model.core, RefinedNonlinearFieldReadout)
    with pytest.raises(ValueError, match="height profile"):
        build_windfarm_model(
            {"forward_architecture": REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE},
            velocity_transform=_normalizer(),
        )


def test_wind_train_profile_and_normalized_residual_round_trip_with_live_height_gradient() -> None:
    model = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(),
        background_profile=_profile(),
        hidden=16,
        message=12,
    )
    receivers = torch.tensor([[[0.5, 0.2, 0.5], [0.5, 0.2, 1.5]]], requires_grad=True)
    residual = torch.tensor([[[0.5, -0.25, 0.75], [-0.1, 0.2, 0.3]]], requires_grad=True)
    background = model.profile_at_receivers(receivers)
    physical = model._physical_from_standardized(residual, receivers)
    normalized_total = model.normalized_field_from_physical(physical, receivers)
    normalized_background = model.normalize_tensor(background)
    torch.testing.assert_close(normalized_total, normalized_background + residual)
    torch.testing.assert_close(model.residual_from_physical(physical, receivers), residual)
    height_gradient = torch.autograd.grad(physical.sum(), receivers)[0][..., 2]
    assert torch.isfinite(height_gradient).all()
    assert torch.all(height_gradient > 0)


def test_refined_context_rejects_mutated_train_profile_and_jvp_restores_fixed_route() -> None:
    torch.manual_seed(12)
    case = _case()
    model = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(),
        background_profile=_profile(),
        hidden=16,
        message=12,
    )
    prepared = model.prepare_case(case)
    receivers = torch.tensor([[[0.5, 0.2, 0.875]]])
    model.core.set_execution(mode="adaptive", phase="hard", training_signal=True)
    original_policy = model.core.refinement_policy
    linearized = model.linearize_case(
        prepared,
        receivers,
        torch.ones_like(prepared.scene.centers),
        wrt="centers",
    )
    assert torch.isfinite(linearized["jvp"]).all()
    assert model.core.refinement_policy == original_policy

    profile = model.background_profile
    assert profile is not None
    profile.values_mps[0, 0] += 0.5
    with pytest.raises(ValueError, match="background profile changed; rebuild"):
        model.predict_physical_case(case, prepared, receivers)


def test_pr2_wind_nonlinear_config_and_strict_state_roundtrip() -> None:
    torch.manual_seed(42)
    model = InteractionPreservingWindModel(
        velocity_transform=_normalizer(),
        background_profile=_profile(),
        mode="P-R2",
        seed=42,
    ).eval()
    config = model.export_config()
    assert config["core"]["native_routing"] == {
        "schema": "native_interaction_routing_v2",
        "mode": "P-R2",
        "policy": "learned_density",
        "initialization_seed": 42,
        "score_hidden": 32,
        "score_bound": 2.0,
        "views": 2,
        "feature_blocks": [64, 64],
    }
    restored = InteractionPreservingWindModel.from_config(copy.deepcopy(config)).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.export_config() == config
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0.0, atol=0.0)

    case = _case()
    receivers = torch.tensor([[[0.5, 0.2, 0.875], [4.0, 1.0, 1.1]]])
    scene = wind_scene_from_case(case)
    prepared = model.prepare_scene(scene)
    restored_prepared = restored.prepare_scene(scene)
    prediction = model.predict_standardized(prepared, receivers)
    restored_prediction = restored.predict_standardized(restored_prepared, receivers)
    assert prediction.shape == (1, 2, 3)
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(restored_prediction, prediction, rtol=0.0, atol=0.0)


def test_wind_read_plan_adapter_requires_bound_features_and_preserves_physical_default():
    torch.manual_seed(81)
    model = InteractionPreservingWindModel(
        velocity_transform=_normalizer(),
        background_profile=_profile(),
        mode="P-R2",
        seed=42,
    ).eval()
    prepared = model.prepare_scene(wind_scene_from_case(_case()))
    config_before = copy.deepcopy(model.export_config())
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    source_ids_before = prepared.core_context.source_ids.detach().clone()
    receivers = torch.tensor([[[0.5, 0.2, 0.875], [2.5, 0.4, 1.125]]])
    features = wind_receiver_features(prepared.scene, receivers)
    receiver_ids = torch.tensor([[19, 23]], dtype=torch.long)
    policy_identity, policy_generation, receiver_generation = "adapter_read_test_v1", 2, 7
    model.core.bind_read_policy_generation(policy_identity, policy_generation)
    active = (prepared.core_context.present > 0)[:, None, :].expand(
        receivers.shape[0], receivers.shape[1], -1).clone()
    plan = model.core.prepare_read_plan(
        prepared.core_context,
        receivers,
        features,
        mode="selected",
        selected_pairs=active,
        policy_identity=policy_identity,
        policy_generation=policy_generation,
        receiver_generation=receiver_generation,
        receiver_ids=receiver_ids,
    )

    dense_standardized = model.predict_standardized(
        prepared, receivers, receiver_features=features,
    )
    planned_standardized = model.predict_standardized(
        prepared,
        receivers,
        receiver_features=features,
        read_plan=plan,
        read_policy_identity=policy_identity,
        read_policy_generation=policy_generation,
        receiver_generation=receiver_generation,
        receiver_ids=receiver_ids,
    )
    torch.testing.assert_close(planned_standardized, dense_standardized, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="caller-supplied receiver_features"):
        model.predict_standardized(
            prepared,
            receivers,
            read_plan=plan,
            read_policy_identity=policy_identity,
            read_policy_generation=policy_generation,
            receiver_generation=receiver_generation,
            receiver_ids=receiver_ids,
        )

    default_physical = model.predict_physical(prepared, receivers)
    inherited_physical = WindFarmJointRegionalModel.predict_physical(model, prepared, receivers)
    torch.testing.assert_close(default_physical, inherited_physical, rtol=0.0, atol=0.0)
    planned_physical = model.predict_physical(
        prepared,
        receivers,
        receiver_features=features,
        read_plan=plan,
        read_policy_identity=policy_identity,
        read_policy_generation=policy_generation,
        receiver_generation=receiver_generation,
        receiver_ids=receiver_ids,
    )
    expected_scale = planned_standardized.new_tensor(model.velocity_transform.safe_std)
    expected_physical = model.profile_at_receivers(receivers) + planned_standardized * expected_scale * float(
        model.velocity_transform.u_ref_mps
    )
    torch.testing.assert_close(planned_physical, expected_physical, rtol=0.0, atol=0.0)
    assert model.export_config() == config_before
    assert set(model.state_dict()) == set(state_before)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in state_before.items())
    assert torch.equal(prepared.core_context.source_ids, source_ids_before)


@pytest.mark.parametrize(
    ("name", "bad_value"),
    (("score_hidden", 32.0), ("score_bound", 2), ("views", 2.0),
     ("feature_blocks", [64.0, 64]), ("feature_blocks", [64, True])),
)
def test_pr2_wind_loader_rejects_numeric_equivalent_v2_metadata_types(name, bad_value) -> None:
    model = InteractionPreservingWindModel(
        velocity_transform=_normalizer(),
        background_profile=_profile(),
        mode="P-R2",
        seed=42,
    )
    config = model.export_config()
    config["core"]["native_routing"][name] = bad_value
    with pytest.raises(TypeError):
        InteractionPreservingWindModel.from_config(config)


def _fresh_wind_engine_checkpoint(mode="P"):
    from honf_runtime.unified_training import fresh_matched_initialization_receipt

    honf_root = Path(__file__).resolve().parents[2]
    tools_path = str(honf_root / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    import joint_regional_train as fresh_runner

    normalizer = VelocityNormalizer(
        mean=np.asarray([1.0, 0.0, 0.0]),
        std=np.asarray([0.2, 0.1, 0.1]),
        safe_std=np.asarray([0.2, 0.1, 0.1]),
        source_rows=72,
        seed=42,
    )
    model = InteractionPreservingWindModel(
        velocity_transform=normalizer,
        background_profile=_profile(),
        mode=mode,
        seed=42,
    )
    model.core.set_consolidated_native_validation(True)
    provider_identity = {
        "dataset": "WindFarm",
        "family": model.family_id,
        "mode": mode,
        "subset_manifest_sha256": "a" * 64,
        "normalizer_and_background_source": {"source_sha256": "b" * 64},
        "standalone_model_config": model.export_config(),
        "fresh_matched_comparison": True,
        "objective": {
            "component_scales_mps": {role: [1.0, 1.0, 1.0] for role in (
                "background", "downstream_envelope", "hub_slab", "near_turbine", "volume"
            )},
            "scale_rule": "synthetic fresh-loader fixture",
            "scale_source_sha256": "c" * 64,
            "scale_calibration": {"schema": "synthetic"},
            "scale_cache_binding": {"sha256": "d" * 64},
        },
    }
    recipe = {
        "schema_version": 1,
        "comparison_protocol": "fresh_segmented_multiview_2500_v1",
        "task": "wind",
        "mode": mode,
        "fresh_matched_comparison": True,
        "dataset_protocol": "wind_shared_fixed24_v1",
        "seed": 42,
        "total_epochs": 2500,
        "native_validation_policy": "consolidated_route_flags_v1",
        "engine_schedule": {"sampling_version": "case_epoch_v1"},
        "model_contract": fresh_runner._fresh_matched_model_contract("wind", mode),
    }
    engine_config = {
        "seed": 42,
        "total_epochs": 2500,
        "sampling_version": "case_epoch_v1",
        "deterministic_case_order": True,
        "training_mode": "joint",
    }
    marker = fresh_matched_initialization_receipt(
        model,
        provider_identity,
        active_mode=mode,
        seed=42,
        sampling_version="case_epoch_v1",
        deterministic_case_order=True,
    )
    identity = {
        "workflow": "fresh_segmented_multiview_2500_v1",
        "recipe": recipe,
        "engine_config": engine_config,
        "provider_identity": provider_identity,
        "native_validation_policy": "consolidated_route_flags_v1",
        "initialization": "fresh_common_trunk_private_router_rng",
        "inherited_parameter_names": [],
        "optimizer_state_start": "fresh_zero_moments",
        "external_learned_field_files": [],
        "fresh_initialization": marker,
        "initial_model_state_sha256": marker["initial_model_state_sha256"],
    }
    return model, {
        "checkpoint_schema_version": 1,
        "workflow": "fresh_segmented_multiview_2500_v1",
        "experiment_identity": identity,
        "arm": mode,
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "branch_parent": None,
    }


@pytest.mark.parametrize("mode", ("P", "P-R", "P-R2"))
def test_fresh_wind_engine_checkpoint_roundtrip_reactivates_validation_policy(mode) -> None:
    source_model, payload = _fresh_wind_engine_checkpoint(mode)
    loaded = load_interaction_preserving_wind_checkpoint(payload)
    assert loaded.export_config() == source_model.export_config()
    assert loaded.core.consolidated_native_validation is True
    for name, tensor in source_model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (("receipt_hash", "Fresh initialization receipt differs"),
     ("receipt_schema", "Fresh matched initialization schema"),
     ("receipt_digest_format", "malformed SHA-256"),
     ("common_digest_format", "malformed SHA-256"),
     ("receipt_exclusion", "common digest must exclude"),
     ("receipt_sampler", "sampler namespace is not bound"),
     ("recipe_route_type", "recipe route metadata has noncanonical numeric types"),
     ("provider_route_type", "provider P-R2 metadata must use exact integer types"),
     ("recipe_mode", "Fresh Wind engine arm, provider and recipe"),
     ("workflow", "not from the fresh segmented multiview workflow")),
)
def test_fresh_wind_engine_checkpoint_rejects_malformed_identity_markers(mutation, message) -> None:
    _, payload = _fresh_wind_engine_checkpoint("P-R2")
    identity = payload["experiment_identity"]
    if mutation == "receipt_hash":
        identity["fresh_initialization"]["common_initialization_sha256"] = "0" * 64
    elif mutation == "receipt_schema":
        identity["fresh_initialization"]["schema"] = "matched_fresh_multiview_initialization_v0"
    elif mutation == "receipt_digest_format":
        identity["fresh_initialization"]["initial_model_state_sha256"] = "not-a-digest"
    elif mutation == "common_digest_format":
        identity["fresh_initialization"]["common_initialization_sha256"] = "not-a-digest"
    elif mutation == "receipt_exclusion":
        identity["fresh_initialization"]["common_parameter_exclusion_prefixes"] = ["core."]
    elif mutation == "receipt_sampler":
        identity["fresh_initialization"]["sampler_namespace"]["dataset_id"] = "other:dataset"
    elif mutation == "recipe_route_type":
        identity["recipe"]["model_contract"]["native_routing"]["views"] = 2.0
    elif mutation == "provider_route_type":
        identity["provider_identity"]["standalone_model_config"]["core"]["native_routing"]["views"] = 2.0
    elif mutation == "recipe_mode":
        identity["recipe"]["mode"] = "P-R"
    else:
        payload["workflow"] = "forward"
    with pytest.raises((TypeError, ValueError), match=message):
        load_interaction_preserving_wind_checkpoint(payload)


def test_fresh_wind_direct_p_engine_loader_rejects_bad_initialization_schema() -> None:
    _, payload = _fresh_wind_engine_checkpoint("P")
    payload["experiment_identity"]["fresh_initialization"]["schema"] = "unknown"
    with pytest.raises(ValueError, match="Fresh matched initialization schema"):
        load_interaction_preserving_wind_checkpoint(payload)


def _compact_c1_adapter_fixture():
    torch.manual_seed(991)
    case = _case()
    model = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(),
        background_profile=_profile(),
        hidden=16,
        message=12,
    )
    scene = wind_scene_from_case(case)
    receivers = torch.tensor(
        [[[8.0, 2.0, 1.0], [9.0, 1.8, 1.2], [10.0, 2.2, 1.4], [11.0, 2.0, 1.6]]]
    )
    # Retain deterministic pair variation while shifting its median probability
    # to 0.5. The resulting C1 route has both zero and fractional corrections.
    with torch.no_grad():
        router_output = model.core.refinement.router[-1]
        router_output.weight.copy_(torch.randn_like(router_output.weight) * 4.0)
        router_output.bias.zero_()
    _, initial = model.predict_refined_batch(
        scene,
        receivers,
        execution_mode="adaptive",
        phase="hard",
        training_signal=False,
        gate_version="compact_c1_v1",
        gate_transition=(0.35, 0.65),
        collect_pair_arrays=True,
    )
    with torch.no_grad():
        logits = torch.logit(initial["probability"].clamp(1e-6, 1.0 - 1e-6))
        model.core.refinement.router[-1].bias.sub_(torch.median(logits))
    _, routed = model.predict_refined_batch(
        scene,
        receivers,
        execution_mode="adaptive",
        phase="hard",
        training_signal=False,
        gate_version="compact_c1_v1",
        gate_transition=(0.35, 0.65),
        collect_pair_arrays=True,
    )
    assert routed["fine_rows"] < routed["active_pairs"]
    assert torch.any((routed["gate_weight"] > 0) & (routed["gate_weight"] < 1))
    assert torch.any(routed["gate_weight"] == 0)
    centers = torch.as_tensor(case.module_centers[None].copy()).requires_grad_()
    prepared = model.prepare_case(case, centers=centers)
    return case, model, prepared, receivers


@pytest.mark.parametrize("wrt", ("centers", "receivers"))
def test_public_compact_c1_linearization_matches_finite_difference_and_restores_policy(wrt: str) -> None:
    case, model, prepared, receivers = _compact_c1_adapter_fixture()
    model.core.set_execution(
        mode="adaptive",
        phase="hard",
        temperature=1.0,
        training_signal=True,
        execution_backend="selected",
        gate_version="compact_c1_v1",
        gate_transition=(0.35, 0.65),
    )
    original_policy = model.core.refinement_policy
    if wrt == "centers":
        tangent = torch.zeros_like(prepared.scene.centers)
        tangent[0, 0, 0] = 1.0
        base = prepared.scene.centers.detach().clone()
    else:
        tangent = torch.zeros_like(receivers)
        tangent[0, 1, 0] = 1.0
        base = receivers.clone()

    linearized = model.linearize_case(
        prepared,
        receivers,
        tangent,
        wrt=wrt,
    )
    assert model.core.refinement_policy is original_policy
    assert original_policy.gate_version == "compact_c1_v1"
    assert original_policy.training_signal is True

    def predict_at(changed: torch.Tensor) -> torch.Tensor:
        if wrt == "centers":
            centers = changed
            query = receivers
        else:
            centers = prepared.scene.centers.detach()
            query = changed
        scene = wind_scene_from_case(case, centers=centers)
        prediction, _ = model.predict_refined_batch(
            scene,
            query,
            execution_mode="adaptive",
            phase="hard",
            training_signal=False,
            gate_version="compact_c1_v1",
            gate_transition=(0.35, 0.65),
            collect_pair_arrays=False,
        )
        return model._physical_from_standardized(prediction.values, query)

    epsilon = 2e-3
    expected_value = predict_at(base)
    finite_difference = (predict_at(base + epsilon * tangent) - predict_at(base - epsilon * tangent)) / (
        2.0 * epsilon
    )
    torch.testing.assert_close(linearized["values"], expected_value, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(linearized["jvp"], finite_difference, atol=2e-4, rtol=8e-3)


def test_refined_wind_adapter_omits_pair_exports_by_default_and_can_collect_them_explicitly() -> None:
    case = _case()
    model = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(), background_profile=_profile(), hidden=16, message=12
    )
    scene = wind_scene_from_case(case)
    receivers = torch.tensor([[[8.0, 2.0, 1.0], [9.0, 2.0, 1.1]]])
    _, ordinary = model.predict_refined_batch(
        scene,
        receivers,
        execution_mode="adaptive",
        phase="hard",
        training_signal=False,
    )
    assert ordinary["pair_arrays_collected"] is False
    assert all(name not in ordinary for name in ("base", "fine", "probability", "keep", "protected"))
    _, exported = model.predict_refined_batch(
        scene,
        receivers,
        execution_mode="adaptive",
        phase="hard",
        training_signal=False,
        collect_pair_arrays=True,
    )
    assert exported["pair_arrays_collected"] is True
    assert all(name in exported for name in ("base", "fine", "probability", "keep", "protected"))


def test_pair_export_flag_preserves_training_objective_and_input_gradients() -> None:
    torch.manual_seed(20261008)
    initial = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(), background_profile=_profile(), hidden=16, message=12
    )
    models = (initial, copy.deepcopy(initial))
    case = _case()
    base_centers = torch.as_tensor(case.module_centers[None].copy())
    base_receivers = torch.tensor([[[8.0, 2.0, 1.0], [9.0, 2.0, 1.1]]])
    observed = []
    parameter_names = tuple(name for name, _ in initial.named_parameters())
    for model, collect_pair_arrays in zip(models, (False, True), strict=True):
        centers = base_centers.clone().requires_grad_()
        scene = wind_scene_from_case(case, centers=centers)
        receivers = base_receivers.clone().requires_grad_()
        prediction, auxiliary = model.predict_refined_batch(
            scene,
            receivers,
            execution_mode="adaptive",
            phase="hard",
            training_signal=True,
            gate_version="compact_c1_v1",
            gate_transition=(0.35, 0.65),
            collect_pair_arrays=collect_pair_arrays,
        )
        assert prediction.full_values is not None
        assert auxiliary["pair_arrays_collected"] is collect_pair_arrays
        loss = (
            prediction.values.square().sum()
            + prediction.full_values.square().sum()
            + 0.1 * auxiliary["base_numerator"]
            + 0.2 * auxiliary["expected_work_numerator"]
            + 0.05 * auxiliary["router_importance_numerator"]
        )
        parameters = tuple(parameter for _, parameter in model.named_parameters())
        gradients = torch.autograd.grad(
            loss, (receivers, centers, *parameters), allow_unused=True
        )
        observed.append((prediction, auxiliary, gradients))
    ordinary, exported = observed
    torch.testing.assert_close(ordinary[0].values, exported[0].values, atol=0, rtol=0)
    torch.testing.assert_close(ordinary[0].full_values, exported[0].full_values, atol=0, rtol=0)
    for key in (
        "base_numerator",
        "base_denominator",
        "expected_work_numerator",
        "expected_work_denominator",
        "router_importance_numerator",
        "router_importance_denominator",
    ):
        torch.testing.assert_close(ordinary[1][key], exported[1][key], atol=0, rtol=0)
    assert len(parameter_names) == len(ordinary[2]) - 2
    for left, right in zip(ordinary[2], exported[2], strict=True):
        if left is None or right is None:
            assert left is None and right is None
        else:
            assert torch.isfinite(left).all() and torch.isfinite(right).all()
            torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_wind_native_scene_batch_is_target_free_and_pads_only_inactive_sources() -> None:
    case_a = _case()
    case_a.index = 0
    case_b = _case()
    case_b.index = 1
    case_b.module_centers = case_b.module_centers[:1].copy()
    case_b.module_present = case_b.module_present[:1].copy()
    case_b.module_features = case_b.module_features[:1].copy()
    case_b.global_context = global_geometry_features(case_b.support, 270.0, 1)
    inputs_a = _scene_inputs(case_a)
    inputs_b = _scene_inputs(case_b)
    assert not hasattr(inputs_a, "target_field")
    assert not hasattr(inputs_a, "velocity_mps")
    scene = _as_scene_batch((inputs_a, inputs_b), torch.device("cpu"))
    assert scene.centers.shape == (2, 2, 3)
    torch.testing.assert_close(scene.present, torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
    assert set(scene.tensors()) == {
        "sources",
        "context",
        "centers",
        "present",
        "lengths",
        "source_lengths",
        "source_measures",
        "environment_tokens",
        "environment_coords",
        "environment_present",
        "environment_measures",
        "source_ids",
        "environment_embedding",
    }


def test_refined_wind_batch_emits_exact_auxiliary_counts_for_shared_engine() -> None:
    torch.manual_seed(13)
    cases = (_case(), _case())
    for index, case in enumerate(cases):
        case.index = index
    scene = _as_scene_batch(tuple(_scene_inputs(case) for case in cases), torch.device("cpu"))
    model = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(),
        background_profile=_profile(),
        hidden=16,
        message=12,
    )
    receivers = torch.tensor(
        [
            [[0.5, 0.2, 0.875], [4.0, 1.0, 1.1]],
            [[0.8, 0.1, 0.875], [3.5, 0.8, 1.1]],
        ]
    )
    predictions, auxiliary = model.predict_refined_batch(
        scene,
        receivers,
        execution_mode="all_fine",
        phase="warmup",
        training_signal=True,
    )
    assert predictions.values.shape == (2, 2, 3)
    assert predictions.full_values is not None
    assert auxiliary["base_denominator"].item() == 2 * 2 * 2 * 12
    assert auxiliary["expected_work_denominator"].item() == 2 * 2 * 2
    assert torch.isfinite(predictions.values).all()


def test_wind_readout_has_finite_live_geometry_and_query_gradients() -> None:
    torch.manual_seed(7)
    case = _case()
    model = WindFarmSharedInteractionModel(velocity_transform=_normalizer(), hidden=16, message=12)
    centers = torch.as_tensor(case.module_centers[None].copy()).requires_grad_()
    prepared = model.prepare_case(case, centers=centers)
    receivers = torch.tensor([[[0.5, 0.2, 0.875], [4.0, 1.0, 1.1]]], requires_grad=True)
    prediction = model.predict_physical_case(case, prepared, receivers)
    center_gradient, query_gradient = torch.autograd.grad(
        prediction.square().sum(), (centers, receivers), allow_unused=False
    )
    assert center_gradient.shape == centers.shape
    assert query_gradient.shape == receivers.shape
    assert torch.isfinite(center_gradient).all() and center_gradient.abs().sum() > 0
    assert torch.isfinite(query_gradient).all() and query_gradient.abs().sum() > 0

    linearized = model.linearize_case(
        prepared,
        receivers.detach(),
        torch.ones_like(centers),
        wrt="centers",
    )
    assert linearized["wrt"] == "centers"
    assert linearized["units"] == "m/s"
    assert "local AD linearization" in linearized["semantics"]
    assert torch.isfinite(linearized["jvp"]).all()


def test_prepared_wind_context_rejects_mutated_inputs_and_wrong_case() -> None:
    torch.manual_seed(11)
    case = _case()
    model = WindFarmSharedInteractionModel(velocity_transform=_normalizer(), hidden=16, message=12)
    prepared = model.prepare_case(case)
    receivers = torch.tensor([[[0.5, 0.2, 0.875]]])

    other_case = _case()
    other_case.case = "different-row"
    with pytest.raises(ValueError, match="different native row"):
        model.predict_case(other_case, prepared, receivers)

    case.module_centers[0, 0] += 0.25
    with pytest.raises(ValueError, match="configuration/context inputs changed"):
        model.predict_case(case, prepared, receivers)

    prepared.scene.context.add_(0.1)
    with pytest.raises(ValueError, match="changed; rebuild"):
        model.predict_standardized(prepared, receivers)


@pytest.mark.parametrize(
    "field",
    (
        "sources",
        "context",
        "centers",
        "present",
        "lengths",
        "source_lengths",
        "source_measures",
        "environment_tokens",
        "environment_coords",
        "environment_present",
        "environment_measures",
        "source_ids",
    ),
)
def test_prepared_wind_context_rejects_reassigned_scene_tensors(field: str) -> None:
    model = WindFarmSharedInteractionModel(velocity_transform=_normalizer(), hidden=16, message=12)
    prepared = model.prepare_case(_case())
    receivers = torch.tensor([[[0.5, 0.2, 0.875]]])
    value = getattr(prepared.scene, field)
    setattr(prepared.scene, field, value.clone())

    with pytest.raises(ValueError, match="metadata changed; rebuild"):
        model.predict_standardized(prepared, receivers)


def test_prepared_wind_context_rejects_dependency_and_output_metadata_changes() -> None:
    model = WindFarmSharedInteractionModel(velocity_transform=_normalizer(), hidden=16, message=12)
    prepared = model.prepare_case(_case())
    receivers = torch.tensor([[[0.5, 0.2, 0.875]]])

    prepared.scene.dependency = replace(WIND_INTERACTION_DEPENDENCY, output_law="affine")
    with pytest.raises(ValueError, match="metadata changed; rebuild"):
        model.predict_standardized(prepared, receivers)

    prepared = model.prepare_case(_case())
    prepared.context.dependency = replace(WIND_INTERACTION_DEPENDENCY, units=("km/s",) * 3)
    with pytest.raises(ValueError, match="metadata changed; rebuild"):
        model.linearize_case(prepared, receivers, torch.ones_like(prepared.scene.centers))

    prepared = model.prepare_case(_case())
    model.core.output_law = "affine"
    with pytest.raises(ValueError, match="metadata changed; rebuild"):
        model.predict_standardized(prepared, receivers)


def test_prepared_wind_context_rejects_replaced_context_and_derived_tensor() -> None:
    model = WindFarmSharedInteractionModel(velocity_transform=_normalizer(), hidden=16, message=12)
    prepared = model.prepare_case(_case())
    receivers = torch.tensor([[[0.5, 0.2, 0.875]]])

    prepared.context = model.core.prepare(prepared.scene)
    with pytest.raises(ValueError, match="metadata changed; rebuild"):
        model.predict_standardized(prepared, receivers)

    prepared = model.prepare_case(_case())
    prepared.context.source_states = prepared.context.source_states.clone()
    with pytest.raises(ValueError, match="metadata changed; rebuild"):
        model.predict_standardized(prepared, receivers)

    prepared = model.prepare_case(_case())
    prepared.context.global_state.add_(0.5)
    with pytest.raises(ValueError, match="metadata changed; rebuild"):
        model.predict_standardized(prepared, receivers)


def test_linearize_rejects_reassigned_scene_before_local_ad() -> None:
    model = WindFarmSharedInteractionModel(velocity_transform=_normalizer(), hidden=16, message=12)
    prepared = model.prepare_case(_case())
    prepared.scene.centers = prepared.scene.centers.clone()
    receivers = torch.tensor([[[0.5, 0.2, 0.875]]])

    with pytest.raises(ValueError, match="metadata changed; rebuild"):
        model.linearize_case(prepared, receivers, torch.ones_like(prepared.scene.centers))


@pytest.mark.parametrize(
    ("mutation", "entrypoint"),
    (
        ("mean_value", "endpoint"),
        ("safe_std_value", "jvp"),
        ("u_ref_and_identity", "endpoint"),
        ("identity_only", "vjp"),
    ),
)
def test_prepared_wind_request_rejects_stale_physical_normalizer(mutation: str, entrypoint: str) -> None:
    case = _case()
    normalizer = _normalizer()
    model = WindFarmSharedInteractionModel(velocity_transform=normalizer, hidden=16, message=12)
    prepared = model.prepare_case(case)
    receivers = torch.tensor([[[0.5, 0.2, 0.875]]])

    if mutation == "mean_value":
        normalizer.mean[0] += 0.25
    elif mutation == "safe_std_value":
        normalizer.safe_std[0] *= 1.5
    elif mutation == "u_ref_and_identity":
        model.velocity_transform = replace(normalizer, u_ref_mps=12.0)
    elif mutation == "identity_only":
        model.velocity_transform = replace(normalizer)
    else:
        raise AssertionError(f"unknown test mutation {mutation!r}")

    with pytest.raises(ValueError, match="velocity transform changed; rebuild"):
        if entrypoint == "endpoint":
            model.predict_physical_case(case, prepared, receivers)
        elif entrypoint == "jvp":
            model.linearize_case(
                prepared,
                receivers,
                torch.ones_like(prepared.scene.centers),
                wrt="centers",
            )
        else:
            prediction = model.predict_physical_case(case, prepared, receivers.requires_grad_())
            torch.autograd.grad(prediction.sum(), receivers)
