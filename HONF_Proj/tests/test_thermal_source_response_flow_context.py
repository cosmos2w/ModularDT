"""Predicted flow is an optional, heat-independent Thermal context edge."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from channelthermal.dependency_flow import ThermalFlowReader
from channelthermal.source_response import SourceResponseThermalModel, ThermalSourceResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from thermal_source_response_flow_context_fit import (
    Q_PROXY_COEFFICIENT,
    _inherited_objective_contract,
    _optimizer_for,
    _scene_signature,
    learning_rate,
)


def _structure(*, requires_grad=False, with_poison=False):
    centers = torch.tensor([[[2.0, 1.5], [7.2, 3.2]]])
    if requires_grad:
        centers.requires_grad_()
    result = {
        "module_centers": centers,
        "module_present": torch.ones(1, 2),
        "material_params": torch.tensor([[0.018, 0.01, 0.02, 1.0, 2.0, 0.45]]),
        "re": torch.tensor([[50.0]]),
        "u_in": torch.ones(1, 1),
        "domain_length_x": torch.tensor([[12.0]]),
        "domain_length_y": torch.tensor([[6.0]]),
    }
    if with_poison:
        result["heat_powers"] = _Poison()
        result["steady_field"] = _Poison()
        result["target_flow"] = _Poison()
    return result


class _Poison:
    def __getattribute__(self, name):
        raise AssertionError(f"Forbidden target/input read: {name}")


def _stats():
    return {
        "field_mean_by_channel": np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.float32),
        "field_std_by_channel": np.array([2.0, 3.0, 4.0, 5.0, 6.0], np.float32),
    }


def _models(*, flow_context):
    torch.manual_seed(19)
    geometry = ThermalSourceResponse(
        {"hidden": 8, "message": 8, "mode": "direct"},
        nx=16, ny=8, environment_nx=3, environment_ny=2,
    )
    adapter = geometry.adapter_config()
    if flow_context:
        adapter["environment_flow_context"] = True
    contextual = ThermalSourceResponse(
        geometry.core_config, **adapter,
    )
    common = geometry.state_dict()
    contextual_state = contextual.state_dict()
    contextual_state.update(common)
    contextual.load_state_dict(contextual_state, strict=True)
    torch.manual_seed(23)
    flow = ThermalFlowReader("D-sep", {"hidden": 8, "message": 8})
    stats = _stats()
    return geometry, contextual, SourceResponseThermalModel(contextual, flow, stats), flow


def test_zero_projection_recovers_geometry_parent_exactly_on_native_outputs():
    geometry, contextual, composed, _flow = _models(flow_context=True)
    parent_composed = SourceResponseThermalModel(geometry, composed.flow, _stats())
    structure = _structure()
    fluid_xy = torch.tensor([[[1.1, 1.0], [4.3, 2.4], [9.1, 4.2]]])
    local = torch.zeros(1, 2, 2)
    parent_prepared = parent_composed.prepare_native(structure, fluid_xy, local_query_points=local, ntheta=4)
    child_prepared = composed.prepare_native(structure, fluid_xy, local_query_points=local, ntheta=4)
    assert parent_prepared["thermal"].context.dependency == parent_composed.interaction_dependency
    assert child_prepared["thermal"].context.dependency == composed.interaction_dependency
    parent = parent_composed.apply_native(parent_prepared, torch.tensor([[0.8, 1.2]]))
    child = composed.apply_native(child_prepared, torch.tensor([[0.8, 1.2]]))
    for key in ("fluid_temperature", "pred_interface", "pred_internal_temperature", "module_material_peak"):
        assert torch.equal(parent[key], child[key]), key
    assert torch.equal(parent["pred_field"][..., :4], child["pred_field"][..., :4])
    assert torch.count_nonzero(contextual.environment_flow_projection.weight) == 0


def test_thermal_composition_exposes_affine_dependency_and_native_control_units():
    geometry, _contextual, composed_context, flow = _models(flow_context=True)
    geometry_contract = SourceResponseThermalModel(geometry, flow, _stats())
    geom_spec = geometry_contract.interaction_dependency
    flow_spec = composed_context.interaction_dependency
    for spec in (geom_spec, flow_spec):
        assert spec.dataset == "ThermalChannel"
        assert spec.output_law == "affine"
        assert spec.configuration_inputs == (
            "prescribed_geometry_and_source_ids", "material_coefficients", "domain_geometry",
            "Reynolds_and_inlet_boundary_context", "inlet_and_wall_temperature_boundary_context",
        )
        assert spec.applicable_controls == ("heat",)
        assert spec.output_roles == ("temperature",)
        assert spec.units == ("packed_dataset_native_temperature",)
        assert spec.prepared_nodes == ("flow_context", "thermal_context")
        assert tuple(edge for edge in spec.edges if edge[0] == "heat") == (("heat", "temperature"),)
    assert ("predicted_flow", "thermal_context") not in geom_spec.edges
    assert ("predicted_flow", "thermal_context") in flow_spec.edges
    assert geometry_contract.interaction_control_units == {
        "heat": "packed_dataset_native_heating_rate",
    }
    geometry_context = geometry.prepare_context(_structure())
    flow_structure = _structure()
    flow_features = composed_context.predicted_environment_flow_features(flow_structure)
    flow_context = composed_context.thermal.prepare_context(
        flow_structure, environment_flow_features=flow_features,
    )
    assert geometry.core.output_law == geometry_context.dependency.output_law == "affine"
    assert composed_context.thermal.core.output_law == flow_context.dependency.output_law == "affine"
    assert geometry_context.dependency == geom_spec
    assert flow_context.dependency == flow_spec


def test_predicted_flow_context_never_reads_heat_or_stored_targets_and_masks_solid_donors():
    _geometry, _contextual, composed, _flow = _models(flow_context=True)
    poisoned = _structure(with_poison=True)
    features = composed.predicted_environment_flow_features(poisoned)
    assert features.shape == (1, 6, 3)
    assert torch.equal(features[0, 0], torch.zeros(3))
    assert torch.equal(features[..., 2].sum(), torch.tensor(5.0))

    changed = _structure(with_poison=True)
    changed["heat_powers"] = torch.tensor([[99.0, -70.0]])
    changed["steady_field"] = torch.full((1, 8, 16, 5), 123.0)
    changed["target_flow"] = torch.full((1, 8, 16, 4), -88.0)
    assert torch.equal(features, composed.predicted_environment_flow_features(changed))


def test_public_composition_keeps_predicted_flow_geometry_vjp_live():
    _geometry, contextual, composed, _flow = _models(flow_context=True)
    with torch.no_grad():
        contextual.environment_flow_projection.weight.fill_(0.03)
    structure = _structure(requires_grad=True)
    fluid_xy = torch.tensor([[[1.1, 1.0], [4.3, 2.4], [9.1, 4.2]]])
    heat = torch.tensor([[0.8, 1.2]])

    live = composed.apply_native(composed.prepare_native(structure, fluid_xy, ntheta=4), heat)
    live_gradient = torch.autograd.grad(live["fluid_temperature"].sum(), structure["module_centers"],
        retain_graph=False)[0]

    # The low-level adapter is used here only to isolate what the public live
    # provider adds to the ordinary geometry derivative.
    structure_detached = _structure(requires_grad=True)
    frozen_features = composed.predicted_environment_flow_features(structure_detached).detach()
    detached_prepared = contextual.prepare_native(structure_detached, fluid_xy, ntheta=4,
        environment_flow_features=frozen_features)
    detached = contextual.apply_native(detached_prepared, heat)
    detached_gradient = torch.autograd.grad(detached["fluid_temperature"].sum(),
        structure_detached["module_centers"])[0]
    assert torch.isfinite(live_gradient).all() and torch.isfinite(detached_gradient).all()
    assert torch.linalg.vector_norm(live_gradient - detached_gradient) > 1.0e-8
    assert torch.linalg.vector_norm(live_gradient) > 0


def test_prepared_thermal_context_rejects_projection_weight_update():
    _geometry, contextual, composed, _flow = _models(flow_context=True)
    structure = _structure()
    fluid_xy = torch.tensor([[[1.1, 1.0], [4.3, 2.4], [9.1, 4.2]]])
    heat = torch.tensor([[0.8, 1.2]])
    features = composed.predicted_environment_flow_features(structure)
    prepared = contextual.prepare_native(structure, fluid_xy, ntheta=4,
        environment_flow_features=features)
    before = contextual.apply_native(prepared, heat)
    with torch.no_grad():
        contextual.environment_flow_projection.weight.add_(0.5)
    with pytest.raises(ValueError, match="adapter weights changed"):
        contextual.apply_native(prepared, heat)
    updated_features = composed.predicted_environment_flow_features(structure)
    updated = contextual.apply_native(contextual.prepare_native(structure, fluid_xy, ntheta=4,
        environment_flow_features=updated_features), heat)
    assert not torch.equal(before["fluid_temperature"], updated["fluid_temperature"])


def test_public_prepared_context_rejects_frozen_flow_weight_update():
    _geometry, _contextual, composed, flow = _models(flow_context=True)
    structure = _structure()
    fluid_xy = torch.tensor([[[1.1, 1.0], [4.3, 2.4], [9.1, 4.2]]])
    prepared = composed.prepare_native(structure, fluid_xy, ntheta=4)
    with torch.no_grad():
        next(flow.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="frozen flow weights changed"):
        composed.apply_native(prepared, torch.tensor([[0.8, 1.2]]))


def _record_preparation(composed):
    prepared = composed.prepare_native(
        _structure(), torch.tensor([[[1.1, 1.0], [4.3, 2.4], [9.1, 4.2]]]),
        local_query_points=torch.zeros(1, 2, 2), ntheta=4,
    )
    prepared.update(
        interface_rows=[np.arange(4), np.arange(4, 8)], interface_row_count=8,
        material_rows=[np.arange(2), np.arange(2, 4)], material_row_count=4,
    )
    return prepared


@pytest.mark.parametrize('stat_key', ['field_mean_by_channel', 'field_std_by_channel'])
@pytest.mark.parametrize('increment', [False, True])
def test_prepared_physical_reads_reject_changed_train_normalization(stat_key, increment):
    _, _, composed, _ = _models(flow_context=True)
    prepared = _record_preparation(composed)
    control = torch.tensor([[0.01, -0.01]])
    apply = composed.apply_record_increment if increment else composed.apply_native
    before = apply(prepared, control)
    assert all(torch.isfinite(value).all() for value in before.values() if torch.is_tensor(value))
    composed.normalization_stats[stat_key][0] += 0.1
    with pytest.raises(ValueError, match='TRAIN normalization changed'):
        apply(prepared, control)
    rebuilt = apply(_record_preparation(composed), control)
    assert all(torch.isfinite(value).all() for value in rebuilt.values() if torch.is_tensor(value))


def test_record_increment_rejects_changed_frozen_flow_context():
    _, _, composed, flow = _models(flow_context=True)
    prepared = _record_preparation(composed)
    control = torch.tensor([[0.01, -0.01]])
    composed.apply_record_increment(prepared, control)
    with torch.no_grad():
        next(flow.parameters()).add_(0.1)
    with pytest.raises(ValueError, match='frozen flow weights changed'):
        composed.apply_record_increment(prepared, control)


def test_geometry_only_adapter_rejects_flow_features():
    geometry, _contextual, _composed, _flow = _models(flow_context=False)
    with pytest.raises(ValueError, match="does not accept environment flow features"):
        geometry.prepare_native(_structure(), torch.zeros(1, 1, 2),
            environment_flow_features=torch.zeros(1, 6, 3))


def test_flow_context_continuation_preserves_parent_adamw_moments_and_starts_projection_empty():
    geometry, contextual, _composed, _flow = _models(flow_context=True)
    parent_optimizer = torch.optim.AdamW(geometry.parameters(), lr=3.0e-6, weight_decay=1.0e-5)
    sum(parameter.square().sum() for parameter in geometry.parameters()).backward()
    parent_optimizer.step()
    optimizer = _optimizer_for(contextual, parent_optimizer.state_dict(), geometry, flow_context=True)
    child_parameters = dict(contextual.named_parameters())
    for name, parent_parameter in geometry.named_parameters():
        parent_state = parent_optimizer.state[parent_parameter]
        child_state = optimizer.state[child_parameters[name]]
        assert parent_state.keys() == child_state.keys()
        for key in parent_state:
            left, right = parent_state[key], child_state[key]
            assert torch.equal(left, right) if torch.is_tensor(left) else left == right
    assert contextual.environment_flow_projection.weight not in optimizer.state


def test_fixed_input_feature_cache_hashes_scene_and_rejects_geometry_gradients():
    structure = {key: value.detach().numpy() for key, value in _structure().items()}
    original = _scene_signature(structure)
    changed = dict(structure)
    changed["module_centers"] = structure["module_centers"].copy()
    changed["module_centers"][0, 0, 0] += 0.01
    assert original != _scene_signature(changed)
    differentiable = dict(structure)
    differentiable["module_centers"] = torch.tensor(structure["module_centers"], requires_grad=True)
    with pytest.raises(ValueError, match="fixed-input scenes without geometry gradients"):
        _scene_signature(differentiable)


def test_flow_context_schedule_has_frozen_warmup_hold_and_cosine_endpoint():
    assert learning_rate(1) == pytest.approx(3.0e-6 + (5.0e-5 - 3.0e-6) / 20)
    assert learning_rate(20) == pytest.approx(5.0e-5)
    assert learning_rate(1000) == pytest.approx(5.0e-5)
    assert learning_rate(2500) == pytest.approx(3.0e-6)


def test_pair_objective_contract_copies_all_parent_calibration_terms():
    parent_recipe = {
        "primary_loss": "mean standardized fluid/surface/material T MSE + .05 standardized q_proxy MSE",
        "operator_decision": {"operator_constraint": "qualified", "scope": "TRAIN"},
        "calibration": {
            "response_coefficient": 1.0,
            "response_scales": {"fluid": 0.28, "surface": 0.44, "material": 0.55},
            "operator_coefficient": 1.0,
            "operator_constraint": "qualified",
        },
    }
    objective = _inherited_objective_contract(parent_recipe)
    assert objective == {
        "q_proxy_coefficient": Q_PROXY_COEFFICIENT,
        "response_coefficient": 1.0,
        "response_scales": {"fluid": 0.28, "surface": 0.44, "material": 0.55},
        "operator_coefficient": 1.0,
        "operator_decision": {"operator_constraint": "qualified", "scope": "TRAIN"},
    }
    parent_recipe["calibration"].pop("response_scales")
    with pytest.raises(ValueError, match="response scales are missing"):
        _inherited_objective_contract(parent_recipe)
