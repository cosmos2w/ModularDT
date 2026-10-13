"""Shared nonlinear execution, dependency ownership and live input derivatives."""
from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.interaction_core import (
    DependencySpec,
    InteractionContextCore,
    InteractionScene,
    NonlinearFieldReadout,
    _geometry,
)
from honf_forward_core.interface_fields.source_response_operator import SourceResponseOperator


def scene(dimension):
    generator = torch.Generator().manual_seed(17)
    random = lambda *shape: torch.rand(*shape, generator=generator, dtype=torch.float64)
    dependency = DependencySpec('native_test', 'nonlinear', ('centers', 'context'), (),
                                ('velocity',), ('m/s',))
    return InteractionScene(random(1, 3, 4), random(1, 2), random(1, 3, dimension),
        torch.ones(1, 3, dtype=torch.float64), torch.ones(1, dimension, dtype=torch.float64),
        torch.full((1, 3), .03, dtype=torch.float64), dependency,
        environment_tokens=random(1, 5, 6), environment_coords=random(1, 5, dimension))


@pytest.mark.parametrize('dimension', [2, 3])
def test_actual_shared_primitives_and_nonlinear_local_derivatives(dimension):
    torch.set_num_threads(2)
    field = NonlinearFieldReadout(4, 2, 6, spatial_dim=dimension, hidden=8,
                                 message=6, output_width=1).double()
    request = scene(dimension)
    receivers = torch.rand(1, 7, dimension, dtype=torch.float64)
    assert SourceResponseOperator._read_features is InteractionContextCore._read_features
    assert NonlinearFieldReadout._read_features is InteractionContextCore._read_features
    prepared = field.prepare(request)
    whole = field.predict(prepared, receivers)
    torch.testing.assert_close(whole, field.predict(prepared, receivers, chunk_size=2))
    with pytest.raises(ValueError, match='no exact affine'):
        field.apply_increment(prepared, torch.ones(1, 3))
    tangent = torch.ones_like(request.centers) * .01
    local = field.linearize(request, receivers, tangent)
    step = 1e-4
    finite = (field.predict(field.prepare(replace(request, centers=request.centers + step * tangent)), receivers)
              - field.predict(field.prepare(replace(request, centers=request.centers - step * tangent)), receivers)) / (2 * step)
    torch.testing.assert_close(local['jvp'], finite, atol=1e-8, rtol=1e-5)
    assert torch.isfinite(local['jvp']).all() and local['jvp'].abs().sum() > 0
    with torch.no_grad():
        request.centers.add_(.01)
    with pytest.raises(ValueError, match='changed'):
        field.predict(prepared, receivers)


def test_zero_embedding_recovers_parent_and_tracks_optional_dependency():
    torch.manual_seed(9)
    model = SourceResponseOperator(4, 2, 6, hidden=8, message=6).double()
    request = scene(2)
    arguments = request.tensors()
    ordinary = model.prepare_context(**arguments)
    embedding = torch.zeros(1, 5, 8, dtype=torch.float64, requires_grad=True)
    injected = model.prepare_context(**{**arguments, 'environment_embedding': embedding})
    for name in ('source_states', 'environment_states', 'global_state'):
        assert torch.equal(getattr(ordinary, name), getattr(injected, name))
    gradient = torch.autograd.grad(injected.global_state.sum(), embedding)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    with torch.no_grad():
        embedding.add_(.01)
    with pytest.raises(ValueError, match='changed'):
        injected.assert_fresh()


def test_nonlinear_prepared_ownership_and_dependency_law():
    field = NonlinearFieldReadout(4, 2, 6, hidden=8, message=6, output_width=1).double()
    other = NonlinearFieldReadout(4, 2, 6, hidden=8, message=6, output_width=1).double()
    prepared = field.prepare(scene(3))
    with pytest.raises(ValueError, match='another operator'):
        other.predict(prepared, torch.rand(1, 2, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match='rebuilding'):
        DependencySpec('wind', 'nonlinear', ('yaw',), ('thrust',), ('u',), ('m/s',))
    with pytest.raises(ValueError, match='both'):
        DependencySpec('thermal', 'affine', ('heat',), ('heat',), ('T',), ('K',))


def test_dependency_rejects_coupled_update_and_control_dependent_affine_cache():
    with pytest.raises(ValueError, match='Cyclic coupled'):
        DependencySpec('coupled', 'nonlinear', ('geometry',), (), ('u',), ('m/s',),
            edges=(('flow', 'temperature'), ('temperature', 'flow')))
    with pytest.raises(ValueError, match='reaches prepared context'):
        DependencySpec('invalid_affine', 'affine', ('geometry',), ('heat',), ('T',), ('native',),
            edges=(('heat', 'flow'), ('flow', 'thermal_context')),
            prepared_nodes=('thermal_context',))
    valid = DependencySpec('thermal', 'affine', ('geometry',), ('heat',), ('T',), ('native',),
        edges=(('geometry', 'thermal_context'), ('thermal_context', 'temperature'), ('heat', 'temperature')),
        prepared_nodes=('thermal_context',))
    assert valid.applicable_controls == ('heat',)


@pytest.mark.parametrize('replace_parameter', [False, True])
def test_prepared_context_rejects_updated_or_replaced_model_weights(replace_parameter):
    field = NonlinearFieldReadout(4, 2, 6, hidden=8, message=6, output_width=1).double()
    request = scene(3)
    receivers = torch.rand(1, 2, 3, dtype=torch.float64)
    prepared = field.prepare(request)
    field.predict(prepared, receivers)
    with torch.no_grad():
        if replace_parameter:
            field.source_encoder[0].weight = torch.nn.Parameter(field.source_encoder[0].weight + .1)
        else:
            field.source_encoder[0].weight.add_(.1)
    with pytest.raises(ValueError, match='model weights changed'):
        field.predict(prepared, receivers)
    assert torch.isfinite(field.predict(field.prepare(request), receivers)).all()


def test_affine_prepared_response_rejects_post_prepare_readout_update():
    field = SourceResponseOperator(4, 2, 6, hidden=8, message=6).double()
    request = scene(2)
    context = field.prepare_context(**request.tensors())
    receivers = torch.rand(1, 2, 2, dtype=torch.float64)
    response = field.prepare_receivers(context, receivers)
    with torch.no_grad():
        field.far_head[0].weight.add_(.1)
    with pytest.raises(ValueError, match='model weights changed'):
        field.apply_increment(response, torch.ones(1, 3, dtype=torch.float64))


def test_factorized_affine_execution_matches_context_read_and_live_derivatives():
    torch.manual_seed(713)
    request = scene(3)
    receivers = torch.rand(1, 7, 3, dtype=torch.float64)
    reference = NonlinearFieldReadout(4, 2, 6, spatial_dim=3, hidden=8,
        message=6, output_width=1, factorized_affine_execution=False).double()
    factored = NonlinearFieldReadout(4, 2, 6, spatial_dim=3, hidden=8,
        message=6, output_width=1, factorized_affine_execution=True).double()
    factored.load_state_dict(reference.state_dict(), strict=True)
    assert tuple(reference.state_dict()) == tuple(factored.state_dict())
    assert sum(parameter.numel() for parameter in reference.parameters()) == sum(
        parameter.numel() for parameter in factored.parameters())
    assert 'factorized_affine_execution' not in reference.config
    assert factored.config['factorized_affine_execution'] is True

    def evaluate(model):
        current = replace(request,
            sources=request.sources.detach().clone().requires_grad_(True),
            centers=request.centers.detach().clone().requires_grad_(True),
            context=request.context.detach().clone().requires_grad_(True))
        query = receivers.detach().clone().requires_grad_(True)
        prepared = model.prepare(current)
        values, messages = model.predict(prepared, query, return_messages=True)
        loss = values.square().mean() + messages.square().mean()
        gradients = torch.autograd.grad(loss,
            tuple(model.parameters()) + (current.sources, current.centers, current.context, query))
        return current, query, prepared, values, messages, gradients

    old = evaluate(reference)
    new = evaluate(factored)
    for actual, expected in zip(new[3:5], old[3:5], strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(new[5], old[5], strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)

    tangent = torch.full_like(receivers, 0.03)
    old_jvp = torch.autograd.functional.jvp(
        lambda query: reference.predict(old[2], query), receivers, tangent)[1]
    new_jvp = torch.autograd.functional.jvp(
        lambda query: factored.predict(new[2], query), receivers, tangent)[1]
    torch.testing.assert_close(new_jvp, old_jvp, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(factored.predict(new[2], receivers, chunk_size=2),
                               new[3], rtol=1e-12, atol=1e-12)

    stale = factored.prepare(request)
    factored.set_factorized_affine_execution(False)
    with pytest.raises(ValueError, match='execution mode changed'):
        factored.predict(stale, receivers)


def test_factorized_execution_toggleback_invalidates_prepared_context_without_state():
    request = scene(3)
    receivers = torch.rand(1, 3, 3, dtype=torch.float64)
    model = NonlinearFieldReadout(4, 2, 6, spatial_dim=3, hidden=8,
        message=6, output_width=1, factorized_affine_execution=True).double()
    prepared = model.prepare(request)
    state_keys = tuple(model.state_dict())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    generation = model._factorized_affine_generation

    model.set_factorized_affine_execution(True)
    assert model._factorized_affine_generation == generation
    model.set_factorized_affine_execution(False)
    model.set_factorized_affine_execution(True)

    assert tuple(model.state_dict()) == state_keys
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
    with pytest.raises(ValueError, match='execution mode changed'):
        model.predict(prepared, receivers)


def test_factorized_affine_execution_handles_empty_environment_and_zero_query_width():
    torch.manual_seed(44)
    request = scene(2)
    request.environment_tokens = request.sources.new_empty(1, 0, 6)
    request.environment_coords = request.centers.new_empty(1, 0, 2)
    reference = NonlinearFieldReadout(4, 2, 6, spatial_dim=2, hidden=8,
        message=6, output_width=1, query_width=0).double()
    factored = NonlinearFieldReadout(4, 2, 6, spatial_dim=2, hidden=8,
        message=6, output_width=1, query_width=0,
        factorized_affine_execution=True).double()
    factored.load_state_dict(reference.state_dict(), strict=True)
    receivers = torch.rand(1, 5, 2, dtype=torch.float64)
    old_context, new_context = reference.prepare(request), factored.prepare(request)
    old = reference.predict(old_context, receivers)
    new = factored.predict(new_context, receivers)
    torch.testing.assert_close(new, old, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(factored.predict(new_context, receivers, chunk_size=1),
                               new, rtol=1e-12, atol=1e-12)


def _read_plan_kwargs(policy_identity, policy_generation, receiver_generation):
    return {
        'read_policy_identity': policy_identity,
        'read_policy_generation': policy_generation,
        'receiver_generation': receiver_generation,
    }


def test_active_dense_plan_removes_only_padding_and_preserves_legacy_reduction():
    torch.manual_seed(43)
    field = NonlinearFieldReadout(4, 2, 6, spatial_dim=2, hidden=8,
        message=6, output_width=2).double()
    state_keys = tuple(field.state_dict())
    request = scene(2)
    request.present = torch.tensor([[1., 0., 1.]], dtype=torch.float64)
    prepared = field.prepare(request)
    receivers = torch.rand(1, 7, 2, dtype=torch.float64)
    legacy, legacy_messages = field.predict(prepared, receivers, return_messages=True)
    plan = field.prepare_read_plan(prepared, receivers, mode='active_dense',
        policy_identity='active_dense_v1', policy_generation=0, receiver_generation=9)
    actual, messages = field.predict(prepared, receivers, return_messages=True,
        read_plan=plan, chunk_size=3,
        **_read_plan_kwargs('active_dense_v1', 0, 9))
    torch.testing.assert_close(actual, legacy, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(messages, legacy_messages, rtol=0, atol=0)
    torch.testing.assert_close(field.predict(prepared, receivers, read_plan=plan,
        chunk_size=1, **_read_plan_kwargs('active_dense_v1', 0, 9)), actual,
        rtol=1e-12, atol=1e-12)
    assert plan.active_pair_count == 14
    assert plan.evaluated_pair_count == 14
    assert plan.padding_pair_count == 7
    assert plan.omitted_active_pair_count == 0
    assert plan.reduction_workspace_elements(chunk_size=3, message_width=6) == 54
    assert plan.reduction_workspace_elements(chunk_size=1, message_width=6) == 18
    assert tuple(field.state_dict()) == state_keys


def test_selected_plan_gathers_before_mlp_and_keeps_original_measure_weights():
    torch.manual_seed(61)
    field = NonlinearFieldReadout(4, 2, 6, spatial_dim=2, hidden=8,
        message=6, output_width=2).double()
    field.bind_read_policy_generation('selected_test_v1', 3)
    request = scene(2)
    request.present = torch.tensor([[1., 0., 1.]], dtype=torch.float64)
    prepared = field.prepare(request)
    receivers = torch.rand(1, 5, 2, dtype=torch.float64)
    selected = torch.zeros(1, 5, 3, dtype=torch.bool)
    selected[0, 0, 0] = True
    selected[0, 1, 2] = True
    selected[0, 2, 0] = True
    selected[0, 2, 2] = True
    selected[0, 4, 0] = True
    plan = field.prepare_read_plan(prepared, receivers, mode='selected',
        selected_pairs=selected, policy_identity='selected_test_v1',
        policy_generation=3, receiver_generation=12)
    observed_rows = []
    hook = field.source_read[0].register_forward_pre_hook(
        lambda _module, inputs: observed_rows.append(inputs[0].shape[0]))
    actual, messages = field.predict(prepared, receivers, read_plan=plan,
        return_messages=True, chunk_size=2,
        **_read_plan_kwargs('selected_test_v1', 3, 12))
    hook.remove()
    assert sum(observed_rows) == int(selected.sum()) == 5
    assert plan.omitted_active_pair_count == 5
    assert plan.reduction_workspace_elements(chunk_size=2, message_width=6) == 36

    dense_fine = field.source_read(field._read_features(prepared, receivers,
        prepared.source_states, prepared.centers, prepared.source_lengths,
        receivers.new_empty(1, receivers.shape[1], 0)))
    weighted = (dense_fine * prepared.source_measure[:, None, :, None]
                * selected[..., None]).sum(2)
    expected = field.field_head(torch.cat((weighted,
        prepared.global_state[:, None].expand(-1, receivers.shape[1], -1),
        _geometry(receivers / prepared.lengths[:, None]),
        receivers.new_empty(1, receivers.shape[1], 0)), -1))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    expected_messages = dense_fine * selected[..., None] * prepared.present[:, None, :, None]
    torch.testing.assert_close(messages, expected_messages, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(field.predict(prepared, receivers, read_plan=plan,
        chunk_size=5, **_read_plan_kwargs('selected_test_v1', 3, 12)),
        actual, rtol=1e-12, atol=1e-12)


def test_read_plan_rejects_illegal_pairs_and_stale_owner_policy_inputs_and_weights():
    torch.manual_seed(71)
    field = NonlinearFieldReadout(4, 2, 6, spatial_dim=2, hidden=8,
        message=6, output_width=1).double()
    field.bind_read_policy_generation('selected_guard_v1', 4)
    request = scene(2)
    request.present = torch.tensor([[1., 0., 1.]], dtype=torch.float64)
    prepared = field.prepare(request)
    receivers = torch.rand(1, 3, 2, dtype=torch.float64)
    support = torch.zeros(1, 3, 3, dtype=torch.bool)
    support[0, 0, 0] = True
    illegal = support.clone()
    illegal[0, 1, 1] = True
    with pytest.raises(ValueError, match='absent or padded'):
        field.prepare_read_plan(prepared, receivers, mode='selected',
            selected_pairs=illegal, policy_identity='selected_guard_v1',
            policy_generation=4, receiver_generation=2)
    plan = field.prepare_read_plan(prepared, receivers, mode='selected',
        selected_pairs=support, policy_identity='selected_guard_v1',
        policy_generation=4, receiver_generation=2)
    def call(model=field, context=prepared, query=receivers,
             policy_identity='selected_guard_v1', policy_generation=4,
             receiver_generation=2):
        return model.predict(context, query, read_plan=plan,
            **_read_plan_kwargs(policy_identity, policy_generation, receiver_generation))
    other = NonlinearFieldReadout(4, 2, 6, spatial_dim=2, hidden=8,
        message=6, output_width=1).double()
    with pytest.raises(ValueError, match='another operator'):
        call(model=other)
    with pytest.raises(ValueError, match='input policy changed'):
        call(policy_identity='stale')
    with pytest.raises(ValueError, match='receiver generation changed'):
        call(receiver_generation=3)
    changed_receivers = receivers.clone()
    with pytest.raises(ValueError, match='receiver inputs changed'):
        call(query=changed_receivers)
    field.bind_read_policy_generation('selected_guard_v1', 5)
    with pytest.raises(ValueError, match='model policy generation changed'):
        call()

    field.bind_read_policy_generation('selected_guard_v2', 0)
    fresh_prepared = field.prepare(request)
    fresh_plan = field.prepare_read_plan(fresh_prepared, receivers, mode='selected',
        selected_pairs=support, policy_identity='selected_guard_v2',
        policy_generation=0, receiver_generation=2)
    with torch.no_grad():
        field.source_read[0].weight.add_(0.01)
    with pytest.raises(ValueError, match='model weights changed'):
        field.predict(fresh_prepared, receivers, read_plan=fresh_plan,
            **_read_plan_kwargs('selected_guard_v2', 0, 2))


def test_read_plan_keeps_input_gradients_live_and_rejects_mutated_bindings():
    torch.manual_seed(83)
    field = NonlinearFieldReadout(4, 2, 6, spatial_dim=2, hidden=8,
        message=6, output_width=1, query_width=1).double()
    field.bind_read_policy_generation('selected_grad_v1', 1)
    request = scene(2)
    request.sources = request.sources.detach().clone().requires_grad_(True)
    request.centers = request.centers.detach().clone().requires_grad_(True)
    prepared = field.prepare(request)
    receivers = torch.rand(1, 3, 2, dtype=torch.float64, requires_grad=True)
    features = torch.rand(1, 3, 1, dtype=torch.float64, requires_grad=True)
    selected = torch.zeros(1, 3, 3, dtype=torch.bool)
    selected[:, :, 0] = True
    selected[:, :, 2] = True
    plan = field.prepare_read_plan(prepared, receivers, features, mode='selected',
        selected_pairs=selected, policy_identity='selected_grad_v1',
        policy_generation=1, receiver_generation=5)
    output = field.predict(prepared, receivers, features, read_plan=plan,
        chunk_size=1, **_read_plan_kwargs('selected_grad_v1', 1, 5))
    gradients = torch.autograd.grad(output.square().sum(),
        (request.sources, request.centers, receivers, features))
    assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0
               for gradient in gradients)

    mutated_indices = field.prepare_read_plan(prepared, receivers, features,
        mode='selected', selected_pairs=selected, policy_identity='selected_grad_v1',
        policy_generation=1, receiver_generation=5)
    mutated_indices.pair_source.add_(0)
    with pytest.raises(ValueError, match='pairs changed'):
        field.predict(prepared, receivers, features, read_plan=mutated_indices,
            **_read_plan_kwargs('selected_grad_v1', 1, 5))

    changed_receivers = receivers.detach().clone()
    bound_features = features.detach()
    stale = field.prepare_read_plan(prepared, changed_receivers, bound_features,
        mode='selected', selected_pairs=selected, policy_identity='selected_grad_v1',
        policy_generation=1, receiver_generation=6)
    with torch.no_grad():
        changed_receivers.add_(0.01)
    with pytest.raises(ValueError, match='receiver generation changed'):
        field.predict(prepared, changed_receivers, bound_features, read_plan=stale,
            **_read_plan_kwargs('selected_grad_v1', 1, 6))


def test_batched_all_active_plan_preserves_native_ids_order_and_vjps():
    torch.manual_seed(97)
    field = NonlinearFieldReadout(4, 2, 6, spatial_dim=2, hidden=8,
        message=6, output_width=2).double()
    base = scene(2)
    repeated = {
        name: (None if value is None else value.expand(2, *value.shape[1:]).clone())
        for name, value in base.tensors().items()
    }
    repeated.update(
        present=torch.tensor([[1., 0., 1.], [0.5, 1., 0.]], dtype=torch.float64),
        source_measures=torch.tensor([[0.5, 0., 1.], [2., 0.75, 0.]], dtype=torch.float64),
        source_ids=torch.tensor([[108, 3, 77], [41, 900, 12]], dtype=torch.long))
    request = replace(base, **repeated)
    request.sources.requires_grad_(True)
    request.centers.requires_grad_(True)
    prepared = field.prepare(request)
    receivers = torch.rand(2, 6, 2, dtype=torch.float64, requires_grad=True)
    receiver_ids = torch.tensor([[900, 4, 900, 2, 77, 5],
                                 [3, 3, 100, 8, 12, 12]], dtype=torch.long)
    active = (prepared.present > 0)[:, None, :].expand(2, 6, -1).clone()
    plan = field.prepare_read_plan(prepared, receivers, mode='selected',
        selected_pairs=active, policy_identity='active_dense_v1',
        policy_generation=0, receiver_generation=14, receiver_ids=receiver_ids)
    assert plan.source_ids is prepared.source_ids
    assert torch.equal(plan.source_ids, request.source_ids)
    assert torch.equal(plan.pair_batch, torch.tensor([0] * 12 + [1] * 12))
    assert torch.equal(plan.pair_receiver[:2], torch.tensor([0, 0]))
    assert torch.equal(plan.pair_source[:2], torch.tensor([0, 2]))
    assert torch.equal(plan.source_ids[plan.pair_batch, plan.pair_source],
        torch.tensor([108, 77] * 6 + [41, 900] * 6))
    assert torch.equal(receiver_ids[plan.pair_batch, plan.pair_receiver],
        torch.tensor([900, 900, 4, 4, 900, 900, 2, 2, 77, 77, 5, 5,
                      3, 3, 3, 3, 100, 100, 8, 8, 12, 12, 12, 12]))
    assert plan.evaluated_pair_count == 24
    assert plan.active_pair_count == 24
    assert plan.padding_pair_count == 12
    assert plan.omitted_active_pair_count == 0
    assert plan.reduction_workspace_elements(chunk_size=2, message_width=6) == 72

    legacy = field.predict(prepared, receivers)
    selected = field.predict(prepared, receivers, read_plan=plan,
        receiver_ids=receiver_ids, chunk_size=2,
        **_read_plan_kwargs('active_dense_v1', 0, 14))
    torch.testing.assert_close(selected, legacy, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(legacy)
    legacy_grad = torch.autograd.grad((legacy * probe).sum(),
        (request.sources, request.centers, receivers), retain_graph=True)
    selected_grad = torch.autograd.grad((selected * probe).sum(),
        (request.sources, request.centers, receivers))
    for actual, expected in zip(selected_grad, legacy_grad):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)

    with pytest.raises(ValueError, match='receiver IDs changed'):
        field.predict(prepared, receivers, read_plan=plan,
            receiver_ids=receiver_ids.clone(),
            **_read_plan_kwargs('active_dense_v1', 0, 14))
    receiver_ids[0, 0] += 1
    with pytest.raises(ValueError, match='receiver IDs changed'):
        field.predict(prepared, receivers, read_plan=plan,
            receiver_ids=receiver_ids,
            **_read_plan_kwargs('active_dense_v1', 0, 14))

    fresh = field.prepare(request)
    fresh_receivers = receivers.detach().clone()
    fresh_plan = field.prepare_read_plan(fresh, fresh_receivers, mode='active_dense',
        policy_identity='active_dense_v1', policy_generation=0,
        receiver_generation=15)
    fresh.source_ids = fresh.source_ids.clone()
    with pytest.raises(ValueError, match='source IDs changed'):
        field.predict(fresh, fresh_receivers, read_plan=fresh_plan,
            **_read_plan_kwargs('active_dense_v1', 0, 15))

    mutated = field.prepare(request)
    mutated_plan = field.prepare_read_plan(mutated, fresh_receivers, mode='active_dense',
        policy_identity='active_dense_v1', policy_generation=0,
        receiver_generation=16)
    with torch.no_grad():
        mutated.source_ids[0, 0] += 1
    with pytest.raises(ValueError, match='changed'):
        field.predict(mutated, fresh_receivers, read_plan=mutated_plan,
            **_read_plan_kwargs('active_dense_v1', 0, 16))
