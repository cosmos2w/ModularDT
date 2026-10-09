"""Actual-coordinate derivatives of both joint output blocks."""
from __future__ import annotations

import pytest
import torch
from test_joint_regional_core import _core, _inputs, _prepare


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('attached', [False, True], ids=['eulerian', 'source-attached'])
def test_joint_coordinate_jvp_vjp_and_two_finite_difference_scales(dimension, attached):
    # Double neural arithmetic isolates the derivative contract from FP32
    # subtraction. Scientific FP32 checkpoints receive their own qualification.
    values = _inputs(batch=1, dimension=dimension)
    values = {name: value.double() if value.is_floating_point() else value
              for name, value in values.items()}
    core = _core(dimension=dimension).double().eval()
    center_size = values['centers'].numel()
    length_size = values['source_lengths'].numel()
    base = torch.cat((values['centers'].flatten(), values['source_lengths'].flatten(),
                      values['receivers'].flatten()))
    generator = torch.Generator().manual_seed(91237)
    tangent = torch.randn(base.shape, dtype=base.dtype, generator=generator)
    tangent /= tangent.norm()
    forcing = torch.tensor([[0.7, 1.2, 0.8, 0.0]], dtype=base.dtype)

    def forward(actual_inputs):
        current = dict(values)
        current['centers'] = actual_inputs[:center_size].reshape_as(values['centers'])
        current['source_lengths'] = actual_inputs[center_size:center_size + length_size].reshape_as(
            values['source_lengths'])
        receivers = actual_inputs[center_size + length_size:].reshape_as(values['receivers'])
        if attached:
            receivers = receivers + current['centers'][:, :1] - values['centers'][:, :1]
        context = _prepare(core, current)
        fields = core.predict_fields(context, receivers, values['receiver_features'])
        response = core.prepare_receivers(context, receivers, values['receiver_features'])
        temperature = core.apply_forcing(response, forcing, accumulation_dtype=torch.float64)
        return torch.cat((fields.flatten(), temperature.flatten()))

    endpoint, jvp = torch.autograd.functional.jvp(forward, base, tangent)
    assert torch.isfinite(endpoint).all() and torch.isfinite(jvp).all()
    assert jvp.norm() > 1e-6
    covector = torch.randn(endpoint.shape, dtype=base.dtype, generator=generator)
    _, vjp = torch.autograd.functional.vjp(forward, base, covector)
    torch.testing.assert_close((jvp * covector).sum(), (tangent * vjp).sum(), rtol=1e-10, atol=1e-11)
    # The native multiscale positional embedding resolves small distances;
    # choose FD steps within its local linear regime rather than loosen error.
    for epsilon in (1e-4, 5e-5):
        difference = (forward(base + epsilon * tangent) - forward(base - epsilon * tangent)) / (2 * epsilon)
        relative_error = (difference - jvp).norm() / jvp.norm()
        assert relative_error < 2e-3, (epsilon, float(relative_error))


def test_attached_receiver_motion_is_a_different_derivative_from_eulerian_motion():
    values = _inputs(batch=1)
    core = _core().double().eval()
    values = {name: value.double() if value.is_floating_point() else value
              for name, value in values.items()}
    direction = torch.zeros_like(values['centers'])
    direction[:, 0, 0] = 1

    def read(centers, attached):
        current = {**values, 'centers': centers}
        receivers = values['receivers']
        if attached:
            receivers = receivers + centers[:, :1] - values['centers'][:, :1]
        return core.predict_fields(_prepare(core, current), receivers, values['receiver_features'])

    _, eulerian = torch.autograd.functional.jvp(lambda centers: read(centers, False), values['centers'], direction)
    _, attached = torch.autograd.functional.jvp(lambda centers: read(centers, True), values['centers'], direction)
    assert (attached - eulerian).norm() > 1e-5


def test_locality_configuration_changes_invalidate_fields_and_prepared_heat_operator():
    values = _inputs()
    core = _core().eval()
    prepared = _prepare(core, values)
    response = core.prepare_receivers(prepared, values['receivers'], values['receiver_features'])
    core.locality_prior_strength = 1.0
    with pytest.raises(ValueError, match='locality configuration changed'):
        core.predict_fields(prepared, values['receivers'], values['receiver_features'])
    with pytest.raises(ValueError, match='locality configuration changed'):
        core.apply_forcing(response, torch.ones(2, 4))
    rebuilt = _prepare(core, values)
    assert torch.isfinite(core.predict_fields(rebuilt, values['receivers'], values['receiver_features'])).all()
