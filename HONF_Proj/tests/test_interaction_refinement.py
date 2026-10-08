"""Recovery, affine precision, actual routing and nonlinear refinement laws."""
from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.interaction_core import NonlinearFieldReadout
from honf_forward_core.interface_fields.interaction_refinement import (
    RefinedNonlinearFieldReadout,
    RefinedSourceResponseOperator,
    RefinementPolicy,
    SourceRefinement,
)
from honf_forward_core.interface_fields.source_response_operator import SourceResponseOperator


def context(model, *, dimension=2, gradients=False):
    torch.manual_seed(19)
    centers = torch.tensor([[[1., 1.], [3., 1.], [0., 0.]]])
    lengths = torch.tensor([[10., 6.]])
    if dimension == 3:
        centers = torch.cat((centers, torch.ones(1, 3, 1)), -1)
        lengths = torch.tensor([[10., 6., 3.]])
    centers.requires_grad_(gradients)
    return model.prepare_context(torch.rand(1, 3, 2), torch.rand(1, 3), centers,
        torch.tensor([[1., 1., 0.]]), lengths, torch.full((1, 3), .2),
        environment_tokens=torch.rand(1, 2, 1),
        environment_coords=torch.ones(1, 2, dimension),
        source_ids=torch.tensor([[17, 4, 9]]))


def thermal_pair():
    torch.manual_seed(7)
    fine = SourceResponseOperator(2, 3, 1, hidden=8, message=8,
                                 forcing_scale=2., far_hidden=12, max_sources=3)
    refined = RefinedSourceResponseOperator.from_fine(fine)
    return fine, refined


def points(dimension=2, gradients=False):
    value = torch.tensor([[[8., 5.], [7., 4.], [1.1, 1.]]])
    if dimension == 3:
        value = torch.cat((value, torch.ones(1, 3, 1)), -1)
    return value.requires_grad_(gradients)


def test_all_open_retains_original_thermal_outputs_and_parameter_names():
    fine, refined = thermal_pair()
    assert set(fine.state_dict()) == {k for k in refined.state_dict() if not k.startswith('refinement.')}
    heat = torch.tensor([[.7, .3, 0.]])
    old = fine.prepare_receivers(context(fine), points())
    for training_signal in (False, True):
        refined.set_execution('all_fine', training_signal=training_signal)
        current = refined.prepare_receivers(context(refined), points())
        assert torch.equal(fine.apply_forcing(old, heat), refined.apply_forcing(current, heat))
        assert torch.equal(old.dense_kernel(), current.dense_kernel())
    refined.set_execution('adaptive', threshold=0)
    opened = refined.prepare_receivers(context(refined), points())
    assert torch.equal(fine.apply_forcing(old, heat), refined.apply_forcing(opened, heat))


def test_unrefined_source_still_contributes_and_heating_does_not_change_kernel():
    _, refined = thermal_pair()
    with torch.no_grad():
        refined.refinement.base_output.weight.zero_()
        refined.refinement.base_output.bias.fill_(.3)
    refined.set_execution('all_base')
    request = refined.prepare_receivers(context(refined), points()[:, :2])
    assert not request.keep.any()
    assert torch.allclose(request.dense_kernel()[0, :, :2, 0], torch.full((2, 2), .15))
    a = refined.apply_forcing(request, torch.tensor([[1., 0., 0.]]))
    b = refined.apply_forcing(request, torch.tensor([[1., 2., 0.]]))
    assert torch.allclose(b - a, torch.full_like(a, .3))
    assert request.refinement_aux['fine_rows'] == 0


def test_thermal_route_receipt_counts_padded_reads_and_selected_detail_rows():
    _, refined = thermal_pair()
    refined.set_execution('adaptive', phase='hard', threshold=0.5)
    request = refined.prepare_receivers(context(refined), points())
    active_capacity = request.receivers.shape[0] * request.receivers.shape[1] * request.context.centers.shape[1]
    assert request.refinement_aux['cheap_rows'] == active_capacity
    assert request.refinement_aux['gate_rows'] == active_capacity
    assert request.refinement_aux['padded_fine_capacity'] == active_capacity
    assert request.refinement_aux['selected_detail_rows'] == int(request.keep.sum())
    assert request.refinement_aux['fine_rows'] == request.refinement_aux['selected_detail_rows']
    assert request.refinement_aux['near_rows'] == int(request.protected.sum())


def test_precise_fixed_route_assembles_factors_before_contraction():
    _, refined = thermal_pair()
    refined.set_execution('adaptive')
    route = torch.tensor([[[True, False, False], [False, True, False], [False, False, False]]])
    request = refined.prepare_receivers(context(refined), points(), fixed_route=route)
    heat = torch.tensor([[.123456789, 1.234567891, 0.]], dtype=torch.float64)
    delta = torch.tensor([[1e-9, -1e-9, 0.]], dtype=torch.float64)
    endpoint = refined.apply_forcing(request, heat, accumulation_dtype=torch.float64)
    changed = refined.apply_forcing(request, heat + delta, accumulation_dtype=torch.float64)
    increment = refined.apply_increment(request, delta, accumulation_dtype=torch.float64)
    torch.testing.assert_close(changed - endpoint, increment, atol=2e-16, rtol=2e-6)
    kernel = request.dense_kernel(accumulation_dtype=torch.float64)
    expected_far = torch.where(request.keep[..., None], request.fine_far.double(), request.base_far.double())
    expected = expected_far * (1 - request.near_weight.double()[..., None])
    batch, query, source = request.near_indices
    expected = expected.index_put((batch, query, source),
        expected[batch, query, source] + request.near_weight.double()[batch, query, source, None] * request.near_values.double()) / 2.
    assert torch.equal(kernel, expected)


@pytest.mark.parametrize('execution', [
    {'mode': 'all_fine'},
    {'mode': 'adaptive', 'phase': 'open'},
    {'mode': 'adaptive', 'phase': 'soft'},
    {'mode': 'adaptive', 'training_signal': True},
])
def test_fixed_route_cannot_override_full_soft_or_training_policy(execution):
    _, refined = thermal_pair()
    refined.set_execution(**execution)
    route = torch.zeros(1, 3, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match='adaptive hard inference'):
        refined.prepare_receivers(context(refined), points(), fixed_route=route)
    nonlinear = RefinedNonlinearFieldReadout(2, 3, 1, hidden=8, message=8, spatial_dim=3)
    nonlinear.set_execution(**execution)
    with pytest.raises(ValueError, match='adaptive hard inference'):
        nonlinear.read_refinement(context(nonlinear, dimension=3), points(3), fixed_route=route)


def test_fixed_route_input_gradients_are_actual_finite_difference_derivatives():
    _, refined = thermal_pair()
    refined.double().set_execution('adaptive')
    query = points(gradients=True).double().detach().requires_grad_(True)
    route = torch.tensor([[[True, False, False], [False, True, False], [True, True, False]]])
    heat = torch.tensor([[.7, .3, 0.]], dtype=torch.float64, requires_grad=True)
    # Build all context inputs in the model's precision.
    c = context(refined.float(), gradients=True)
    refined.double()
    c = refined.prepare_context(torch.ones(1, 3, 2, dtype=torch.float64), torch.ones(1, 3, dtype=torch.float64),
        c.centers.double(), c.present.double(), c.lengths.double(), c.source_lengths.double(),
        environment_tokens=torch.ones(1, 2, 1, dtype=torch.float64),
        environment_coords=torch.ones(1, 2, 2, dtype=torch.float64))
    def value(q):
        r = refined.prepare_receivers(c, q, fixed_route=route)
        return refined.apply_forcing(r, heat, accumulation_dtype=torch.float64).sum()
    actual = value(query)
    derivative = torch.autograd.grad(actual, (query, heat, c.centers), retain_graph=True)
    assert all(torch.isfinite(item).all() for item in derivative)
    tangent = torch.zeros_like(query); tangent[0, 0, 0] = 1
    eps = 1e-6
    finite = (value(query + eps * tangent) - value(query - eps * tangent)) / (2 * eps)
    torch.testing.assert_close((derivative[0] * tangent).sum(), finite, atol=1e-8, rtol=1e-5)


def test_router_surrogate_restores_signal_but_does_not_train_omitted_fine_rows():
    _, refined = thermal_pair()
    with torch.no_grad():
        refined.refinement.router[-1].bias.fill_(-2.)
    refined.set_execution('adaptive', training_signal=True)
    request = refined.prepare_receivers(context(refined), points()[:, :2])
    full_far = request.full_response.far_kernel
    full_far.retain_grad()
    loss = refined.apply_forcing(request, torch.tensor([[.7, .3, 0.]])).square().sum()
    loss.backward()
    assert not request.keep.any()
    assert full_far.grad is None or torch.count_nonzero(full_far.grad) == 0
    assert refined.refinement.router[-1].bias.grad.abs().sum() > 0
    assert request.refinement_aux['fine_rows'] == 1 * 2 * 3
    assert refined.auxiliary_terms()['cheap_rows'] == 1 * 2 * 3


def test_router_training_surrogate_does_not_change_context_or_base_gradients():
    _, surrogate = thermal_pair()
    with torch.no_grad():
        surrogate.refinement.router[-1].bias.fill_(-2.)
    deployed = copy.deepcopy(surrogate)
    surrogate.set_execution('adaptive', training_signal=True)
    deployed.set_execution('adaptive', training_signal=False)
    for model in (surrogate, deployed):
        request = model.prepare_receivers(context(model), points()[:, :2])
        model.apply_forcing(request, torch.tensor([[.7, .3, 0.]])).square().sum().backward()
    a, b = dict(surrogate.named_parameters()), dict(deployed.named_parameters())
    for name, parameter in a.items():
        if '.router.' in name:
            continue
        if parameter.grad is None and b[name].grad is None:
            continue
        left = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
        right = torch.zeros_like(b[name]) if b[name].grad is None else b[name].grad
        torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-8)
    assert surrogate.refinement.router[-1].bias.grad.abs().sum() > 0
    assert deployed.refinement.router[-1].bias.grad is None


@pytest.mark.parametrize('replace_route', [False, True])
def test_prepared_refinement_route_mutation_or_replacement_is_rejected(replace_route):
    _, refined = thermal_pair()
    refined.set_execution('all_base')
    request = refined.prepare_receivers(context(refined), points())
    if replace_route:
        request.keep = request.keep.clone()
    else:
        request.keep[0, 0, 0] = True
    with pytest.raises(ValueError, match='changed|replaced'):
        request.dense_kernel(accumulation_dtype=torch.float64)


def test_prepared_refinement_rejects_a_backend_policy_replacement():
    _, refined = thermal_pair()
    refined.set_execution('adaptive', execution_backend='selected')
    request = refined.prepare_receivers(context(refined), points())
    request.policy = replace(request.policy, execution_backend='dense_masked')
    with pytest.raises(ValueError, match='replaced'):
        request.dense_kernel(accumulation_dtype=torch.float64)


def test_nonlinear_all_fine_recovers_and_coarse_read_preserves_physical_sources():
    torch.manual_seed(13)
    fine = NonlinearFieldReadout(2, 3, 1, hidden=8, message=8, spatial_dim=3)
    refined = RefinedNonlinearFieldReadout(2, 3, 1, hidden=8, message=8, spatial_dim=3)
    result = refined.load_state_dict(copy.deepcopy(fine.state_dict()), strict=False)
    assert not result.unexpected_keys
    old = fine.predict(context(fine, dimension=3), points(3))
    current = refined.predict(context(refined, dimension=3), points(3))
    assert torch.equal(old, current)
    refined.set_execution('adaptive', threshold=0)
    assert torch.equal(old, refined.predict(context(refined, dimension=3), points(3)))
    refined.set_execution('all_base')
    reduced = refined.read_refinement(context(refined, dimension=3), points(3)[:, :2])
    assert reduced.auxiliary['fine_rows'] == 0
    assert not reduced.auxiliary['keep'].any()
    assert torch.count_nonzero(reduced.auxiliary['base'][..., :2, :]) > 0
    with pytest.raises(ValueError, match='no exact affine'):
        refined.apply_increment(None)


def test_nonlinear_output_only_read_does_not_retain_pair_exports_or_change_gradients():
    torch.manual_seed(13)
    model = RefinedNonlinearFieldReadout(2, 3, 1, hidden=8, message=8, spatial_dim=3)
    model.set_execution('adaptive')
    prepared = context(model, dimension=3, gradients=True)
    query = points(3, gradients=True)
    exported = model.read_refinement(prepared, query, chunk_size=2)
    normal = model.read_refinement(prepared, query, chunk_size=2, collect_pair_arrays=False)
    assert torch.equal(exported.values, normal.values)
    assert all(name not in normal.auxiliary for name in ('base', 'fine', 'keep', 'probability', 'protected'))
    for name in ('fine_rows', 'cheap_rows', 'selected_detail_rows', 'near_rows'):
        assert exported.auxiliary[name] == normal.auxiliary[name]
    original_grad = torch.autograd.grad(exported.values.sum(), (query, prepared.centers), retain_graph=True)
    normal_grad = torch.autograd.grad(normal.values.sum(), (query, prepared.centers))
    for original, actual in zip(original_grad, normal_grad, strict=True):
        assert torch.equal(original, actual)


def test_soft_read_prices_full_fine_and_training_replay_is_shared():
    _, refined = thermal_pair()
    refined.set_execution('adaptive', phase='soft')
    request = refined.prepare_receivers(context(refined), points()[:, :2])
    assert request.refinement_aux['fine_rows'] == 6
    assert request.refinement_aux['complete_fine_values']
    refined.set_execution('all_fine', training_signal=True)
    full = refined.prepare_receivers(context(refined), points())
    assert full.full_response is full


def test_thermal_dense_masked_backend_preserves_the_refinement_kernel_and_increment():
    _, refined = thermal_pair()
    prepared_context = context(refined)
    receivers = points()
    refined.set_execution('adaptive', threshold=0.5, execution_backend='selected')
    selected = refined.prepare_receivers(prepared_context, receivers, chunk_size=2)
    refined.set_execution('adaptive', threshold=0.5, execution_backend='dense_masked')
    dense = refined.prepare_receivers(prepared_context, receivers, chunk_size=2)

    torch.testing.assert_close(dense.keep, selected.keep)
    torch.testing.assert_close(dense.protected, selected.protected)
    torch.testing.assert_close(
        dense.dense_kernel(accumulation_dtype=torch.float64),
        selected.dense_kernel(accumulation_dtype=torch.float64),
        atol=2e-7,
        rtol=2e-6,
    )
    assert selected.refinement_aux['fine_rows'] == selected.refinement_aux['selected_detail_rows']
    assert dense.refinement_aux['fine_rows'] == receivers.shape[0] * receivers.shape[1] * prepared_context.centers.shape[1]
    heat = torch.tensor([[0.7, 0.3, 0.0]])
    delta = torch.tensor([[0.001, -0.001, 0.0]])
    torch.testing.assert_close(
        refined.apply_forcing(dense, heat),
        refined.apply_forcing(selected, heat),
        atol=2e-7,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        refined.apply_increment(dense, delta, accumulation_dtype=torch.float64),
        refined.apply_increment(selected, delta, accumulation_dtype=torch.float64),
        atol=2e-10,
        rtol=2e-6,
    )


def test_wind_selected_and_dense_masked_backends_match_masks_values_and_input_derivatives():
    torch.manual_seed(13)
    model = RefinedNonlinearFieldReadout(2, 3, 1, hidden=8, message=8, spatial_dim=3)
    with torch.no_grad():
        for parameter in model.refinement.router.parameters():
            parameter.zero_()
        model.refinement.router[-1].bias.fill_(-2.0)
    prepared = context(model, dimension=3, gradients=True)
    receivers = points(3, gradients=True)

    model.set_execution('adaptive', phase='hard', threshold=0.5, execution_backend='selected')
    selected = model.read_refinement(prepared, receivers, chunk_size=2)
    selected_gradients = torch.autograd.grad(
        selected.values.square().sum(), (prepared.centers, receivers), retain_graph=True
    )
    model.set_execution('adaptive', phase='hard', threshold=0.5, execution_backend='dense_masked')
    dense = model.read_refinement(prepared, receivers, chunk_size=2)
    dense_gradients = torch.autograd.grad(dense.values.square().sum(), (prepared.centers, receivers))

    torch.testing.assert_close(dense.auxiliary['keep'], selected.auxiliary['keep'])
    torch.testing.assert_close(dense.auxiliary['protected'], selected.auxiliary['protected'])
    torch.testing.assert_close(dense.values, selected.values, atol=2e-6, rtol=2e-6)
    assert selected.auxiliary['selected_detail_rows'] < dense.auxiliary['fine_rows']
    assert dense.auxiliary['fine_rows'] == receivers.shape[0] * receivers.shape[1] * prepared.centers.shape[1]
    for actual, expected in zip(dense_gradients, selected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_wind_dense_masked_backend_is_source_permutation_safe_with_padded_sources():
    torch.manual_seed(23)
    model = RefinedNonlinearFieldReadout(2, 3, 1, hidden=8, message=8, spatial_dim=3)
    with torch.no_grad():
        for parameter in model.refinement.router.parameters():
            parameter.zero_()
        model.refinement.router[-1].bias.fill_(-2.0)
    receivers = points(3)
    permutation = torch.tensor([2, 0, 1])
    torch.manual_seed(19)
    sources = torch.rand(1, 3, 2)
    context_values = torch.rand(1, 3)
    centers = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])
    lengths = torch.tensor([[10.0, 6.0, 3.0]])
    present = torch.tensor([[1.0, 1.0, 0.0]])
    source_lengths = torch.full((1, 3), 0.2)
    environment_tokens = torch.rand(1, 2, 1)
    environment_coords = torch.ones(1, 2, 3)
    source_ids = torch.tensor([[17, 4, 9]])
    original_context = model.prepare_context(sources, context_values, centers, present, lengths,
        source_lengths, environment_tokens=environment_tokens, environment_coords=environment_coords,
        source_ids=source_ids)
    permuted_context = model.prepare_context(
        sources=sources[:, permutation],
        context=context_values,
        centers=centers[:, permutation],
        present=present[:, permutation],
        lengths=lengths,
        source_lengths=source_lengths[:, permutation],
        environment_tokens=environment_tokens,
        environment_coords=environment_coords,
        source_ids=source_ids[:, permutation],
    )

    model.set_execution('adaptive', threshold=0.5, execution_backend='dense_masked')
    original = model.read_refinement(original_context, receivers, chunk_size=2)
    permuted = model.read_refinement(permuted_context, receivers, chunk_size=2)
    torch.testing.assert_close(permuted.auxiliary['keep'], original.auxiliary['keep'][:, :, permutation])
    torch.testing.assert_close(permuted.values, original.values, atol=2e-6, rtol=2e-6)


def test_wind_route_provenance_is_independent_of_larger_misaligned_fine_tiles():
    torch.manual_seed(37)
    model = RefinedNonlinearFieldReadout(2, 3, 1, hidden=8, message=8, spatial_dim=3)
    with torch.no_grad():
        for parameter in model.refinement.router.parameters():
            parameter.zero_()
        model.refinement.router[-1].bias.zero_()
    prepared = context(model, dimension=3, gradients=True)
    receivers = points(3, gradients=True).repeat(1, 4, 1)

    model.set_execution('adaptive', phase='hard', threshold=0.5)
    legacy = model.read_refinement(prepared, receivers, chunk_size=2, route_chunk_size=2)
    legacy_gradient, = torch.autograd.grad(legacy.values.square().sum(), receivers, retain_graph=True)
    larger = model.read_refinement(prepared, receivers, chunk_size=3, route_chunk_size=2)
    larger_gradient, = torch.autograd.grad(larger.values.square().sum(), receivers)

    assert torch.equal(larger.auxiliary['keep'], legacy.auxiliary['keep'])
    assert torch.equal(larger.auxiliary['protected'], legacy.auxiliary['protected'])
    assert torch.equal(larger.auxiliary['probability'], legacy.auxiliary['probability'])
    torch.testing.assert_close(larger.values, legacy.values, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(larger_gradient, legacy_gradient, atol=2e-5, rtol=2e-5)


def test_thermal_route_provenance_is_independent_of_larger_misaligned_fine_tiles():
    _, refined = thermal_pair()
    refined.set_execution('adaptive', phase='hard', threshold=0.5)
    receivers = torch.tensor([[[8., 5.], [7., 4.], [1.1, 1.], [6., 2.], [9., 3.]]])
    scene = context(refined)
    legacy = refined.prepare_receivers(scene, receivers, chunk_size=2, route_chunk_size=2)
    larger = refined.prepare_receivers(scene, receivers, chunk_size=3, route_chunk_size=2)

    assert torch.equal(larger.keep, legacy.keep)
    assert torch.equal(larger.protected, legacy.protected)
    assert torch.equal(larger.probability, legacy.probability)
    torch.testing.assert_close(larger.dense_kernel(), legacy.dense_kernel(), atol=2e-6, rtol=2e-6)
    delta = torch.tensor([[1.0e-8, -1.0e-8, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        refined.apply_increment(larger, delta, accumulation_dtype=torch.float64),
        refined.apply_increment(legacy, delta, accumulation_dtype=torch.float64),
        atol=2e-12,
        rtol=2e-6,
    )


def test_compact_c1_gate_has_exact_support_annulus_blend_and_zero_endpoint_slopes():
    lower, upper = 0.35, 0.65
    policy = RefinementPolicy(mode='adaptive', gate_version='compact_c1_v1',
                              gate_transition=(lower, upper))
    probability = torch.tensor([0.20, lower, 0.50, upper, 0.80], dtype=torch.float64,
                               requires_grad=True)
    near = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    gate = SourceRefinement.continuous_weight(probability, near, policy)
    torch.testing.assert_close(gate, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0], dtype=torch.float64))
    slope, = torch.autograd.grad(gate.sum(), probability)
    assert slope[1] == 0 and slope[3] == 0

    # The existing geometric near taper remains an independent, C1 support
    # contribution: it is fully fine inside, partial through the annulus, and
    # exactly zero outside when the optional score is below its lower bound.
    near = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
    low_score = torch.full_like(near, 0.20)
    gate = SourceRefinement.continuous_weight(low_score, near, policy)
    torch.testing.assert_close(gate, near, atol=0, rtol=0)

    torch.manual_seed(43)
    model = RefinedNonlinearFieldReadout(2, 3, 1, hidden=8, message=8, spatial_dim=3)
    with torch.no_grad():
        for parameter in model.refinement.router.parameters():
            parameter.zero_()
    receivers = torch.tensor([[[1.1, 1.0, 1.0], [1.6, 1.0, 1.0],
                               [1.9, 1.0, 1.0]]])
    for score in (0.20, lower, 0.50, upper, 0.80):
        with torch.no_grad():
            model.refinement.router[-1].bias.fill_(torch.logit(torch.tensor(score)).item())
        prepared = context(model, dimension=3)
        model.set_execution('adaptive', gate_version='compact_c1_v1',
                            gate_transition=(lower, upper), execution_backend='selected')
        result = model.read_refinement(prepared, receivers, collect_pair_arrays=True)
        expected_gate = SourceRefinement.continuous_weight(
            result.auxiliary['probability'], model.near_weight(
                receivers, prepared.centers, prepared.source_lengths, prepared.present),
            model.refinement_policy,
        ) * (prepared.present[:, None] > 0)
        torch.testing.assert_close(result.auxiliary['gate_weight'], expected_gate, atol=0, rtol=0)
        assert torch.equal(result.auxiliary['keep'], expected_gate > 0)
        assert result.auxiliary['fine_rows'] == int(result.auxiliary['keep'].sum())
        values, messages = model.predict(prepared, receivers, return_messages=True)
        mixed = (result.auxiliary['base'] + expected_gate[..., None] *
                 (result.auxiliary['fine'] - result.auxiliary['base']))
        expected_messages = torch.where(expected_gate[..., None] == 1,
                                        result.auxiliary['fine'], mixed)
        expected_messages = expected_messages * prepared.present[:, None, :, None]
        torch.testing.assert_close(messages, expected_messages, atol=0, rtol=0)
        assert torch.isfinite(values).all()


def test_compact_c1_selected_and_dense_masked_match_live_input_vjps():
    torch.manual_seed(47)
    model = RefinedNonlinearFieldReadout(2, 3, 1, hidden=8, message=8, spatial_dim=3).double()
    dtype = torch.float64
    centers = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [0.0, 0.0, 1.0]]],
                           dtype=dtype, requires_grad=True)
    torch.manual_seed(19)
    sources = torch.rand(1, 3, 2, dtype=dtype)
    source_values = torch.rand(1, 3, dtype=dtype)
    environment_tokens = torch.rand(1, 2, 1, dtype=dtype)
    lengths = torch.tensor([[10.0, 6.0, 3.0]], dtype=dtype)
    present = torch.tensor([[1.0, 1.0, 0.0]], dtype=dtype)
    source_lengths = torch.full((1, 3), 0.2, dtype=dtype)
    environment_coords = torch.ones(1, 2, 3, dtype=dtype)
    source_ids = torch.tensor([[17, 4, 9]])

    def prepare(center_values):
        return model.prepare_context(sources, source_values, center_values, present, lengths,
            source_lengths, environment_tokens=environment_tokens,
            environment_coords=environment_coords, source_ids=source_ids)

    prepared = prepare(centers)
    receivers = torch.tensor([[[1.6, 1.0, 1.0]]], dtype=dtype, requires_grad=True)
    features = receivers.new_empty(1, 1, 0)
    source_features = model.refinement.source_features(prepared)
    _, initial_probability = model.refinement.read(prepared, receivers, features,
        source_features, temperature=1.0)
    with torch.no_grad():
        model.refinement.router[-1].bias.sub_(torch.logit(initial_probability[0, 0, 0]))
    prepared = prepare(centers)
    fixed_route = torch.tensor([[[True, True, False]]])

    def evaluate(query_values, center_values):
        local = model.read_refinement(prepare(center_values), query_values,
            fixed_route=fixed_route, collect_pair_arrays=True)
        return local.values.sum(), local.auxiliary['gate_weight'][0, 0, 0]

    model.set_execution('adaptive', gate_version='compact_c1_v1',
                        gate_transition=(0.35, 0.65), execution_backend='selected')
    result = model.read_refinement(prepared, receivers, fixed_route=fixed_route,
                                   collect_pair_arrays=True)
    assert 0.7 < result.auxiliary['gate_weight'][0, 0, 0] < 0.8
    value_grad_query, value_grad_centers = torch.autograd.grad(
        result.values.sum(), (receivers, centers), retain_graph=True)
    gate_grad_query, gate_grad_centers = torch.autograd.grad(
        result.auxiliary['gate_weight'][0, 0, 0], (receivers, centers), retain_graph=True)

    eps = 1e-5
    query_direction = torch.zeros_like(receivers); query_direction[0, 0, 0] = 1
    center_direction = torch.zeros_like(centers); center_direction[0, 0, 0] = 1
    query_plus, gate_plus = evaluate(receivers.detach() + eps * query_direction, centers.detach())
    query_minus, gate_minus = evaluate(receivers.detach() - eps * query_direction, centers.detach())
    finite_value_query = (query_plus - query_minus) / (2 * eps)
    finite_gate_query = (gate_plus - gate_minus) / (2 * eps)
    center_plus, center_gate_plus = evaluate(receivers.detach(), centers.detach() + eps * center_direction)
    center_minus, center_gate_minus = evaluate(receivers.detach(), centers.detach() - eps * center_direction)
    finite_value_center = (center_plus - center_minus) / (2 * eps)
    finite_gate_center = (center_gate_plus - center_gate_minus) / (2 * eps)
    torch.testing.assert_close(value_grad_query[0, 0, 0], finite_value_query,
                               atol=2e-6, rtol=2e-4)
    torch.testing.assert_close(value_grad_centers[0, 0, 0], finite_value_center,
                               atol=2e-6, rtol=2e-4)
    torch.testing.assert_close(gate_grad_query[0, 0, 0], finite_gate_query,
                               atol=2e-7, rtol=2e-4)
    torch.testing.assert_close(gate_grad_centers[0, 0, 0], finite_gate_center,
                               atol=2e-7, rtol=2e-4)

    repeated = receivers.detach().repeat(1, 4, 1).requires_grad_(True)
    parity_centers = centers.detach().clone().requires_grad_(True)
    prepared_for_parity = prepare(parity_centers)
    features = repeated.new_empty(1, 4, 0)
    source_features = model.refinement.source_features(prepared_for_parity)
    _, probabilities = model.refinement.read(prepared_for_parity, repeated, features,
        source_features, temperature=1.0)
    with torch.no_grad():
        # Center the router's active-pair logit distribution in the C1 annulus.
        active_logits = torch.logit(probabilities[..., prepared_for_parity.present[0] > 0])
        model.refinement.router[-1].bias.sub_(active_logits.median())
    prepared_for_parity = prepare(parity_centers)
    model.set_execution('adaptive', gate_version='compact_c1_v1', execution_backend='selected')
    selected = model.read_refinement(prepared_for_parity, repeated, chunk_size=3,
                                     route_chunk_size=2, collect_pair_arrays=True)
    selected_grads = torch.autograd.grad(selected.values.square().sum(),
        (repeated, prepared_for_parity.centers), retain_graph=True)
    model.set_execution('adaptive', gate_version='compact_c1_v1', execution_backend='dense_masked')
    dense = model.read_refinement(prepared_for_parity, repeated, chunk_size=3,
                                  route_chunk_size=2, collect_pair_arrays=True)
    dense_grads = torch.autograd.grad(dense.values.square().sum(),
        (repeated, prepared_for_parity.centers))
    assert torch.equal(dense.auxiliary['keep'], selected.auxiliary['keep'])
    assert torch.equal(dense.auxiliary['protected'], selected.auxiliary['protected'])
    torch.testing.assert_close(dense.auxiliary['gate_weight'], selected.auxiliary['gate_weight'],
                               atol=0, rtol=0)
    torch.testing.assert_close(dense.values, selected.values, atol=2e-9, rtol=2e-8)
    assert selected.auxiliary['fine_rows'] == int(selected.auxiliary['keep'].sum())
    assert selected.auxiliary['fine_rows'] < dense.auxiliary['fine_rows']
    for actual, expected in zip(dense_grads, selected_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_compact_c1_equal_work_controls_preserve_positive_optional_weight_multiset():
    class RouteContext:
        pass

    route_context = RouteContext()
    route_context.present = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    route_context.centers = torch.tensor([[[-2.0, 0.0], [0.0, 0.0], [3.0, 0.0],
                                           [6.0, 0.0], [9.0, 0.0], [14.0, 0.0]]])
    query = torch.tensor([[[4.0, 0.0], [5.5, 0.0]]])
    near = torch.tensor([[[0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.25, 0.0, 0.0]]])
    protected = near > 0
    probability = torch.tensor([[[0.50, 0.10, 0.10, 0.10, 0.80, 0.99],
                                 [0.80, 0.10, 0.10, 0.25, 0.50, 0.99]]])
    active = (route_context.present[:, None] > 0).expand_as(probability)
    adaptive_policy = RefinementPolicy(mode='adaptive', gate_version='compact_c1_v1')
    adaptive_keep = SourceRefinement.route(route_context, query, probability, protected,
        adaptive_policy, near_weight=near)
    adaptive_weight = SourceRefinement.deployed_weight(route_context, probability, near,
        protected, adaptive_keep, adaptive_policy)

    for mode in ('nearest', 'upstream', 'shuffle'):
        policy = replace(adaptive_policy, mode=mode)
        keep = SourceRefinement.route(route_context, query, probability, protected,
            policy, near_weight=near)
        weight = SourceRefinement.deployed_weight(route_context, probability, near,
            protected, keep, policy)
        assert torch.equal(keep.sum(-1), adaptive_keep.sum(-1))
        assert torch.equal(weight[protected], adaptive_weight[protected])
        for query_index in range(query.shape[1]):
            optional = active[0, query_index] & ~protected[0, query_index]
            expected = adaptive_weight[0, query_index, optional]
            expected = expected[expected > 0].sort().values
            actual = weight[0, query_index, optional]
            actual = actual[actual > 0].sort().values
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert not torch.equal(keep, adaptive_keep)


def test_thermal_compact_c1_keeps_affine_heating_nulls_fp64_increments_and_stale_guard():
    _, refined = thermal_pair()
    with torch.no_grad():
        for parameter in refined.refinement.router.parameters():
            parameter.zero_()
        refined.refinement.router[-1].bias.fill_(torch.logit(torch.tensor(0.47)).item())
    refined.set_execution('adaptive', gate_version='compact_c1_v1',
                          gate_transition=(0.35, 0.65), execution_backend='selected')
    prepared_context = context(refined, gradients=True)
    receivers = points(gradients=True)
    request = refined.prepare_receivers(prepared_context, receivers, chunk_size=2,
                                        route_chunk_size=2)
    assert request.gate_weight is not None
    mixed_far = request.base_far.double() + request.gate_weight.double()[..., None] * (
        request.fine_far.double() - request.base_far.double())
    far = torch.where(request.gate_weight.double()[..., None] == 1,
                      request.fine_far.double(), mixed_far)
    expected = far * (1 - request.near_weight.double()[..., None])
    batch, query, source = request.near_indices
    expected = expected.index_put((batch, query, source), expected[batch, query, source] +
        request.near_weight.double()[batch, query, source, None] * request.near_values.double())
    expected = expected / request.forcing_scale
    torch.testing.assert_close(request.dense_kernel(accumulation_dtype=torch.float64),
                               expected, atol=0, rtol=0)
    recomputed_gate = SourceRefinement.continuous_weight(request.probability.double(),
        request.near_weight.double(), request.policy)
    recomputed_gate = recomputed_gate * (request.context.present[:, None] > 0)
    assert torch.max(torch.abs(request.gate_weight.double() - recomputed_gate)) > 0

    heat = torch.tensor([[0.123456789, 1.234567891, 0.0]], dtype=torch.float64)
    delta = torch.tensor([[1e-9, -1e-9, 0.0]], dtype=torch.float64)
    zero = torch.zeros_like(heat)
    endpoint = refined.apply_forcing(request, heat, accumulation_dtype=torch.float64)
    changed = refined.apply_forcing(request, heat + delta, accumulation_dtype=torch.float64)
    increment = refined.apply_increment(request, delta, accumulation_dtype=torch.float64)
    null_endpoint = refined.apply_forcing(request, zero, accumulation_dtype=torch.float64)
    null_increment = refined.apply_increment(request, zero, accumulation_dtype=torch.float64)
    torch.testing.assert_close(changed - endpoint, increment, atol=2e-16, rtol=2e-6)
    assert torch.count_nonzero(null_endpoint) == 0
    assert torch.count_nonzero(null_increment) == 0

    stale = refined.prepare_receivers(prepared_context, receivers)
    stale.gate_weight = stale.gate_weight.clone()
    with pytest.raises(ValueError, match='replaced'):
        stale.dense_kernel(accumulation_dtype=torch.float64)


def test_compact_c1_closed_recovery_is_router_only_and_open_keeps_router_signal():
    torch.manual_seed(59)
    deployed = RefinedNonlinearFieldReadout(2, 3, 1, hidden=8, message=8, spatial_dim=3)
    with torch.no_grad():
        for parameter in deployed.refinement.router.parameters():
            parameter.zero_()
        deployed.refinement.router[-1].bias.fill_(torch.logit(torch.tensor(0.20)).item())
    training = copy.deepcopy(deployed)

    def run(model, *, phase, training_signal):
        model.set_execution('adaptive', phase=phase, gate_version='compact_c1_v1',
                            training_signal=training_signal)
        prepared = context(model, dimension=3, gradients=True)
        receivers = torch.tensor([[[8.0, 5.0, 2.0], [7.0, 4.0, 2.0]]],
                                 requires_grad=True)
        result = model.read_refinement(prepared, receivers, collect_pair_arrays=True)
        loss = result.values.square().sum()
        loss.backward()
        return result, receivers.grad, prepared.centers.grad

    closed_inference, closed_input, closed_centers = run(
        deployed, phase='hard', training_signal=False)
    closed_training, training_input, training_centers = run(
        training, phase='hard', training_signal=True)
    assert not closed_training.auxiliary['keep'].any()
    assert torch.count_nonzero(closed_training.auxiliary['gate_weight']) == 0
    torch.testing.assert_close(closed_training.values, closed_inference.values, atol=0, rtol=0)
    torch.testing.assert_close(training_input, closed_input, atol=0, rtol=0)
    torch.testing.assert_close(training_centers, closed_centers, atol=0, rtol=0)
    inference_parameters = dict(deployed.named_parameters())
    training_parameters = dict(training.named_parameters())
    for name, parameter in training_parameters.items():
        if '.router.' in name:
            continue
        left = parameter.grad
        right = inference_parameters[name].grad
        if left is None and right is None:
            continue
        left = torch.zeros_like(parameter) if left is None else left
        right = torch.zeros_like(inference_parameters[name]) if right is None else right
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert training.refinement.router[-1].bias.grad.abs().sum() > 0
    inference_router_grad = deployed.refinement.router[-1].bias.grad
    assert inference_router_grad is None or torch.count_nonzero(inference_router_grad) == 0
    for parameter in training.source_read.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0

    open_inference, open_input, open_centers = run(
        deployed, phase='open', training_signal=False)
    open_training, open_training_input, open_training_centers = run(
        training, phase='open', training_signal=True)
    torch.testing.assert_close(open_training.values, open_inference.values, atol=0, rtol=0)
    torch.testing.assert_close(open_training_input, open_input, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(open_training_centers, open_centers, atol=2e-7, rtol=2e-6)
    assert training.refinement.router[-1].bias.grad.abs().sum() > 0
    open_inference_router_grad = deployed.refinement.router[-1].bias.grad
    assert open_inference_router_grad is None or torch.count_nonzero(open_inference_router_grad) == 0
