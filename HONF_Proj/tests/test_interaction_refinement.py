"""Recovery, affine precision, actual routing and nonlinear refinement laws."""
from __future__ import annotations

import copy

import pytest
import torch

from honf_forward_core.interface_fields.interaction_core import NonlinearFieldReadout
from honf_forward_core.interface_fields.interaction_refinement import (
    RefinedNonlinearFieldReadout,
    RefinedSourceResponseOperator,
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
