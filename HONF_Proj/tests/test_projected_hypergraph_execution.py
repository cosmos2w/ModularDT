"""Same-operator action lowering, objective caching and diagnostic contracts."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.typed_hypergraph_state import (
    ProjectedSourceAccess,
    TypedSourceAccess,
    prepare_projected_action,
    prepare_structural_cost,
    source_moments,
    structural_cost,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _encoded(dimension=2):
    generator = torch.Generator().manual_seed(731)
    return EncodedInterfaceCase(
        module_tokens=torch.randn(2, 4, 8, generator=generator),
        env_tokens=torch.randn(2, 7, 8, generator=generator),
        global_token=torch.randn(2, 8, generator=generator),
        module_centers=torch.rand(2, 4, dimension, generator=generator),
        env_coords=torch.rand(2, 7, dimension, generator=generator),
        module_present=torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]]),
        module_features=torch.randn(2, 4, 3, generator=generator), env_features=None,
        env_weights=torch.tensor([[1., .5, 2., 1., 3., .75, 1.]]).expand(2, -1),
        coordinate_scale=torch.ones(1, 1, dimension))



@pytest.mark.parametrize("near", [False, True])
def test_projected_mixture_matches_all_float64_operand_gradients(near):
    torch.manual_seed(57)
    operands = [torch.rand(2, 5, 3, dtype=torch.float64),
                torch.rand(2, 3, 4, dtype=torch.float64),
                torch.randn(2, 3, 16, dtype=torch.float64),
                torch.randn(3, 16, dtype=torch.float64),
                torch.randn(3, dtype=torch.float64)]
    valid = torch.tensor([[True, True, False, True], [True, True, True, True]])
    pair_valid = torch.ones(2, 5, 4, dtype=torch.bool)
    pair_valid[:, 0] = False
    operands[0][:, 1] = 0  # empty row projects the zero control to bias
    measure = torch.rand(2, 4, dtype=torch.float64)
    envelope = torch.rand(2, 5, 4, dtype=torch.float64) if near else None
    results, gradients = [], []
    for projected in (False, True):
        edge, member, controls, weight, bias = [x.clone().requires_grad_() for x in operands]
        action = prepare_projected_action(member, controls, weight, bias) if projected else None
        access = source_moments(edge, member, controls, measure, valid,
            pair_valid=pair_valid, near=envelope, prepared_action=action)
        value = access.projected if projected else torch.nn.functional.linear(access.control, weight, bias)
        results.append(value)
        gradients.append(torch.autograd.grad(value.square().sum(), (edge, member, controls, weight, bias)))
        torch.testing.assert_close(value[:, 0], bias.expand(2, 4, -1))
        if not near:
            torch.testing.assert_close(value[:, 1], bias.expand(2, 4, -1))
    torch.testing.assert_close(results[1], results[0], rtol=1e-12, atol=1e-12)
    for actual, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("architecture", ["faithful_receiver_hypergraph_honf", "local_overlap_hypergraph_honf"])
@pytest.mark.parametrize("mode", ["normal", "full_access", "root_union", "control_identity"])
def test_learned_native_projection_matches_reference_values_and_first_gradients(architecture, mode):
    torch.manual_seed(713)
    encoded = _encoded()
    encoded = replace(encoded, env_characteristic_lengths=torch.full_like(encoded.env_weights, .03))
    reference = TypedHypergraphField(8, 12, 2, 2, architecture=architecture,
        spatial_dim=2, module_characteristic_length=.03).eval()
    query, features = torch.rand(2, 9, 2), torch.rand(2, 9, 6)
    reference.read(reference.prepare(encoded, encoded.module_tokens), encoded, query, features)
    with torch.no_grad():
        for module in (*reference.control_gain.values(), reference.control_score):
            module.weight.normal_(0, .25)
            module.bias.normal_(0, .15)
    reference.last_structural_cost = None
    lowered = deepcopy(reference)
    reference.control_execution = "full_control"
    outputs, gradients, parameter_gradients = [], [], []
    for model in (reference, lowered):
        model.set_plan_intervention(mode)
        module = encoded.module_tokens.clone().requires_grad_()
        environment = encoded.env_tokens.clone().requires_grad_()
        coordinates = query.clone().requires_grad_()
        record = replace(encoded, module_tokens=module, env_tokens=environment)
        state = model.prepare(record, module)
        value, _ = model.read(state, record, coordinates, features)
        value.square().sum().backward()
        outputs.append(value)
        gradients.append((module.grad, environment.grad, coordinates.grad))
        parameter_gradients.append({name: p.grad for name, p in model.named_parameters()})
    torch.testing.assert_close(outputs[1], outputs[0], rtol=2e-5, atol=2e-6)
    for actual, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    for name, expected in parameter_gradients[0].items():
        actual = parameter_gradients[1][name]
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6, msg=name)


@pytest.mark.parametrize("version", [1, 2])
def test_cached_structural_objective_preserves_uneven_chunk_weighting_and_gradients(version):
    torch.manual_seed(923)
    model = TypedHypergraphField(8, 12, 2, 2, architecture="faithful_receiver_hypergraph_honf",
        spatial_dim=2, module_characteristic_length=.03,
        options={"structural_measure_policy_version": version}).train()
    encoded = _encoded()
    plan = model.organizer.prepare(encoded, encoded.module_tokens, phase=0, soft=True)
    preparation = {tau: model.organizer.access(plan, coords, tau, soft=True)
                   for tau, coords in (("MM", encoded.module_centers),
                                       ("ME", encoded.module_centers), ("EM", encoded.env_coords))}
    cache = prepare_structural_cost(preparation, plan)
    queries = torch.rand(2, 11, 2)
    costs = [[], []]
    for start, stop in ((0, 3), (3, 8), (8, 11)):
        access = {**preparation, **{tau: model.organizer.access(plan, queries[:, start:stop], tau, soft=True)
                                    for tau in ("QM", "QE")}}
        for index, cached in enumerate((None, cache)):
            cost, _ = structural_cost(access, plan, preparation=cached)
            costs[index].append(cost * (stop - start) / 11)
    original, cached = [torch.stack(parts).sum() for parts in costs]
    torch.testing.assert_close(cached, original, rtol=1e-12, atol=1e-12)
    parameters = tuple(model.organizer.parameters())
    expected = torch.autograd.grad(original, parameters, retain_graph=True, allow_unused=True)
    actual = torch.autograd.grad(cached, parameters, allow_unused=True)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-10, atol=1e-10)


def test_numerical_read_omits_unmeasured_diagnostics_and_export_keeps_full_controls():
    encoded = _encoded()
    model = TypedHypergraphField(8, 12, 2, 2, architecture="faithful_receiver_hypergraph_honf",
        spatial_dim=2, module_characteristic_length=.03).eval()
    query, features = torch.rand(2, 9, 2), torch.rand(2, 9, 6)
    state = model.prepare(encoded, encoded.module_tokens)
    value, aux = model.read(state, encoded, query, features)
    assert model.last_structural_cost is None
    assert "hypergraph_structural_numerator" not in aux
    assert "hypergraph_QE_repeated_paths_removed" not in aux
    assert state["hypergraph_structural_preparation"] is None
    action = model._numerical_access(state["hypergraph_plan"], query, "QE", state["hypergraph_actions"])
    assert isinstance(action, ProjectedSourceAccess)
    assert action.projected.shape == (2, 9, 7, 4)
    assert not hasattr(action, "control")
    exported = model.export_typed_state(state)["receiver_access"](query, "QE")
    assert isinstance(exported, TypedSourceAccess)
    assert exported.control.shape == (2, 9, 7, 16)
    assert "repeated_paths_removed" in exported.diagnostics
    diagnostic, measured = model.read(state, encoded, query, features, return_routing_maps=True)
    torch.testing.assert_close(diagnostic, value)
    assert "hypergraph_QE_repeated_paths_removed" in measured


def test_checkpoint_restore_rebuilds_phase_action_and_preserves_parameter_schema():
    torch.manual_seed(37)
    encoded = _encoded()
    model = TypedHypergraphField(8, 12, 2, 2, architecture="faithful_receiver_hypergraph_honf",
        spatial_dim=2, module_characteristic_length=.03).eval()
    query, features = torch.rand(2, 5, 2), torch.rand(2, 5, 6)
    model.read(model.prepare(encoded, encoded.module_tokens), encoded, query, features)
    reference = deepcopy(model)
    reference.control_execution = "full_control"
    assert tuple(reference.state_dict()) == tuple(model.state_dict())
    with torch.no_grad():
        model.control_gain["QM"].weight.fill_(.1)
    state = model.prepare(encoded, encoded.module_tokens)
    prediction, _ = model.read(state, encoded, query, features)
    reference.load_state_dict(model.state_dict(), strict=True)
    restored = deepcopy(model)
    restored.load_state_dict(reference.state_dict(), strict=True)
    restored_state = restored.prepare(encoded, encoded.module_tokens)
    actual, _ = restored.read(restored_state, encoded, query, features)
    torch.testing.assert_close(actual, prediction)
    assert restored_state["hypergraph_actions"]["QM"] is not state["hypergraph_actions"]["QM"]


def test_soft_native_projection_preserves_live_permission_and_parameter_gradients():
    torch.manual_seed(103)
    encoded = _encoded()
    reference = TypedHypergraphField(8, 12, 2, 2, architecture="faithful_receiver_hypergraph_honf",
        spatial_dim=2, module_characteristic_length=.03).train()
    reference.organizer.eval()
    reference.permission_mode = "soft"
    query, features = torch.rand(2, 6, 2), torch.rand(2, 6, 6)
    reference.read(reference.prepare(encoded, encoded.module_tokens), encoded, query, features)
    with torch.no_grad():
        for module in (*reference.control_gain.values(), reference.control_score):
            module.weight.normal_(0, .3)
            module.bias.normal_(0, .1)
    reference.last_structural_cost = None
    lowered = deepcopy(reference)
    reference.control_execution = "full_control"
    predictions, input_gradients, parameter_gradients = [], [], []
    for model in (reference, lowered):
        states = encoded.module_tokens.clone().requires_grad_()
        env = encoded.env_tokens.clone().requires_grad_()
        coordinates = query.clone().requires_grad_()
        record = replace(encoded, module_tokens=states, env_tokens=env)
        prepared = model.prepare(record, states)
        value, aux = model.read(prepared, record, coordinates, features)
        (value.square().sum() + .1 * aux["hypergraph_structural_numerator"] / 6).backward()
        predictions.append(value)
        input_gradients.append((states.grad, env.grad, coordinates.grad))
        parameter_gradients.append({name: p.grad for name, p in model.named_parameters()})
    torch.testing.assert_close(predictions[1], predictions[0], rtol=2e-5, atol=2e-6)
    for actual, expected in zip(input_gradients[1], input_gradients[0]):
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
    for name, expected in parameter_gradients[0].items():
        torch.testing.assert_close(parameter_gradients[1][name], expected, rtol=5e-5, atol=5e-6, msg=name)


def test_projected_extreme_permission_quotients_keep_finite_gradients():
    edge = torch.tensor([[[1e-45, 2e-45], [0., 0.]]], dtype=torch.float64, requires_grad=True)
    member = torch.tensor([[[1e-40, 2e-40], [3e-40, 1e-40]]], dtype=torch.float64, requires_grad=True)
    controls = torch.randn(1, 2, 16, requires_grad=True)
    weight = torch.randn(4, 16, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)
    action = prepare_projected_action(member, controls, weight, bias)
    access = source_moments(edge, member, controls, torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool),
                            prepared_action=action, include_diagnostics=False)
    assert access.weight.dtype == torch.float64
    assert access.projected.dtype == torch.float32
    objective = (access.projected.square() * access.weight[..., None]).sum()
    gradients = torch.autograd.grad(objective, (edge, member, controls, weight, bias))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    torch.testing.assert_close(access.projected[:, 1], bias.expand(1, 2, -1))


@pytest.mark.parametrize("architecture", ["faithful_receiver_hypergraph_honf", "overlap_control_hypergraph_honf"])
def test_hard_training_preserves_detached_permission_boundary_and_physical_gradients(architecture):
    torch.manual_seed(989)
    encoded = _encoded()
    reference = TypedHypergraphField(8, 12, 2, 2, architecture=architecture,
        spatial_dim=2, module_characteristic_length=.03).train()
    reference.organizer.eval()
    query, features = torch.rand(2, 6, 2), torch.rand(2, 6, 6)
    with torch.no_grad():
        reference.read(reference.prepare(encoded, encoded.module_tokens), encoded, query, features)
        for gain in reference.control_gain.values():
            gain.weight.normal_(0, .25)
        reference.control_score.weight.normal_(0, .25)
    reference.last_structural_cost = None
    lowered = deepcopy(reference)
    reference.control_execution = "full_control"
    results, gradients, parameter_gradients = [], [], []
    for model in (reference, lowered):
        states = encoded.module_tokens.clone().requires_grad_()
        coords = query.clone().requires_grad_()
        record = replace(encoded, module_tokens=states)
        value, _ = model.read(model.prepare(record, states), record, coords, features)
        value.square().sum().backward()
        results.append(value)
        gradients.append((states.grad, coords.grad))
        parameter_gradients.append({name: parameter.grad for name, parameter in model.named_parameters()})
    torch.testing.assert_close(results[1], results[0], rtol=2e-5, atol=2e-6)
    for actual, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    for name, expected in parameter_gradients[0].items():
        torch.testing.assert_close(parameter_gradients[1][name], expected, rtol=3e-5, atol=3e-6, msg=name)
