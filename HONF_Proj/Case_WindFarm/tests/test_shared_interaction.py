from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from honf_forward_core.interface_fields.interaction_core import NonlinearFieldReadout
from honf_forward_core.interface_fields.interaction_refinement import RefinedNonlinearFieldReadout

from windfarm.geometry import environment_representation, global_geometry_features, support_geometry
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
