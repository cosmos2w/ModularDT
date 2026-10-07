"""Shared nonlinear execution, dependency ownership and live input derivatives."""
from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.interaction_core import (
    DependencySpec,
    InteractionContextCore,
    InteractionScene,
    NonlinearFieldReadout,
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
