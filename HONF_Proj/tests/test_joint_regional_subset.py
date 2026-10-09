"""Parity, work and conditional derivatives of actual collective subset reads."""
from __future__ import annotations

import pytest
import torch
from test_joint_regional_core import _core, _inputs, _prepare


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('mode', ['J-H', 'J-geometry'])
def test_subset_matches_same_truncated_function_and_preserves_source_path(dimension, mode):
    values = _inputs(dimension=dimension)
    core = _core(mode, dimension=dimension).eval()
    context = _prepare(core, values)
    counts = torch.arange(7)[None].expand(2, -1) % 4
    kwargs = {'retained_edge_counts': counts, 'return_organization': True, 'chunk_size': 3}
    dense, dense_receipt = core.predict_fields(
        context, values['receivers'], values['receiver_features'], **kwargs)
    subset, subset_receipt = core.predict_fields(
        context, values['receivers'], values['receiver_features'], receiver_edge_executor='subset', **kwargs)
    torch.testing.assert_close(subset, dense, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(subset_receipt['receiver_access'], dense_receipt['receiver_access'], rtol=0, atol=0)
    work = subset_receipt['executor_receipt']
    assert work['receiver_edge_value_pairs'] == int(counts.sum())
    assert work['receiver_edge_value_pairs'] < dense_receipt['executor_receipt']['receiver_edge_value_pairs']
    assert work['receiver_edge_score_pairs'] == 2 * 7 * context.group_states.shape[1]
    assert work['physical_source_value_pairs'] == 2 * 7 * 4
    assert work['executors'] == ['packed-subset'] and not work['full_access_fallback']
    responses = [core.prepare_receivers(
        context, values['receivers'], values['receiver_features'],
        retained_edge_counts=counts, receiver_edge_executor=executor, chunk_size=2)
        for executor in ('dense', 'subset')]
    torch.testing.assert_close(responses[0].dense_kernel(), responses[1].dense_kernel(), rtol=2e-6, atol=2e-6)
    assert responses[1].dense_kernel().shape == (2, 7, 4, 1)
    assert responses[1].dense_kernel()[0, :, -1].count_nonzero() == 0
    torch.testing.assert_close(responses[1].context.source_ids, values['source_ids'], rtol=0, atol=0)
    # Even receivers selecting no collective edge still read physical sources.
    zero_edge_read = core.read_receiver(
        context, values['receivers'], values['receiver_features'],
        retained_edge_counts=torch.zeros_like(counts), receiver_edge_executor='subset')
    assert zero_edge_read['executor_receipt']['receiver_edge_value_pairs'] == 0
    assert zero_edge_read['source_features'].abs().sum() > 0


def test_full_access_subset_request_is_explicit_dense_fallback():
    values = _inputs()
    core = _core().eval()
    context = _prepare(core, values)
    expected = core.predict_fields(context, values['receivers'], values['receiver_features'])
    for mass in (None, 1.0):
        actual, receipt = core.predict_fields(
            context, values['receivers'], values['receiver_features'],
            retained_access_mass=mass, receiver_edge_executor='subset', return_organization=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert receipt['executor_receipt']['full_access_fallback']
        assert receipt['executor_receipt']['executors'] == ['dense']


@pytest.mark.parametrize('dimension', [2, 3])
def test_fixed_subset_coordinate_derivatives_match_dense_and_two_fd_steps(dimension):
    values = _inputs(batch=1, dimension=dimension)
    values = {name: value.double() if value.is_floating_point() else value for name, value in values.items()}
    core = _core(dimension=dimension).double().eval()
    context = _prepare(core, values)
    mask = core.read_receiver(context, values['receivers'], values['receiver_features'],
                              retained_edge_counts=torch.full((1, 7), 3))['receiver_access'].detach() > 0
    center_size = values['centers'].numel()
    length_size = values['source_lengths'].numel()
    base = torch.cat((values['centers'].flatten(), values['source_lengths'].flatten(), values['receivers'].flatten()))
    generator = torch.Generator().manual_seed(53071)
    tangent = torch.randn(base.shape, dtype=base.dtype, generator=generator)
    tangent /= tangent.norm()

    def forward(actual_inputs, executor):
        current = {**values,
            'centers': actual_inputs[:center_size].reshape_as(values['centers']),
            'source_lengths': actual_inputs[center_size:center_size + length_size].reshape_as(values['source_lengths'])}
        receivers = actual_inputs[center_size + length_size:].reshape_as(values['receivers'])
        prepared = _prepare(core, current)
        kwargs = {'fixed_receiver_edge_mask': mask, 'receiver_edge_executor': executor, 'chunk_size': 3}
        fields = core.predict_fields(prepared, receivers, values['receiver_features'], **kwargs)
        response = core.prepare_receivers(prepared, receivers, values['receiver_features'], **kwargs)
        heat = core.apply_forcing(response, torch.tensor([[.7, 1.2, .8, 0]], dtype=base.dtype))
        return torch.cat((fields.flatten(), heat.flatten()))

    endpoint, jvp = torch.autograd.functional.jvp(lambda x: forward(x, 'subset'), base, tangent)
    dense_endpoint, dense_jvp = torch.autograd.functional.jvp(lambda x: forward(x, 'dense'), base, tangent)
    torch.testing.assert_close(endpoint, dense_endpoint, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(jvp, dense_jvp, rtol=1e-10, atol=1e-11)
    covector = torch.randn(endpoint.shape, dtype=base.dtype, generator=generator)
    _, vjp = torch.autograd.functional.vjp(lambda x: forward(x, 'subset'), base, covector)
    torch.testing.assert_close((jvp * covector).sum(), (tangent * vjp).sum(), rtol=1e-10, atol=1e-11)
    for epsilon in (1e-4, 5e-5):
        fd = (forward(base + epsilon * tangent, 'subset') - forward(base - epsilon * tangent, 'subset')) / (2 * epsilon)
        assert (fd - jvp).norm() / jvp.norm() < 2e-3


def test_fixed_support_validation_and_mutation_invalidates_prepared_operator():
    values = _inputs()
    core = _core().eval()
    context = _prepare(core, values)
    mask = core.read_receiver(context, values['receivers'], values['receiver_features'],
                              retained_edge_counts=torch.full((2, 7), 2))['receiver_access'].detach() > 0
    with pytest.raises(ValueError, match='Choose one'):
        core.predict_fields(context, values['receivers'], values['receiver_features'],
                            fixed_receiver_edge_mask=mask, retained_access_mass=.99)
    with pytest.raises(ValueError, match='absent edge'):
        core.predict_fields(context, values['receivers'], values['receiver_features'],
                            fixed_receiver_edge_mask=torch.ones_like(mask))
    with pytest.raises(ValueError, match='positive access mass'):
        core.predict_fields(context, values['receivers'], values['receiver_features'],
                            fixed_receiver_edge_mask=torch.zeros_like(mask))
    response = core.prepare_receivers(context, values['receivers'], values['receiver_features'],
                                      fixed_receiver_edge_mask=mask, receiver_edge_executor='subset')
    with torch.no_grad():
        mask[0, 0, 0] = ~mask[0, 0, 0]
    with pytest.raises(ValueError, match='changed'):
        core.apply_forcing(response, torch.ones(2, 4))
