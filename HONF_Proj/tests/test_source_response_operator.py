"""Synthetic generic response contracts; no generator or physical solve."""
from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.source_response_operator import SourceResponseOperator


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inputs(dimension=2, modules=3, environments=5):
    generator = torch.Generator().manual_seed(41)
    dtype = torch.float64
    centers = torch.rand(2, modules, dimension, generator=generator, dtype=dtype) + .2
    arguments = {'sources': torch.rand(2, modules, 4, generator=generator, dtype=dtype),
        'context': torch.rand(2, 5, generator=generator, dtype=dtype), 'centers': centers,
        'present': torch.ones(2, modules, dtype=dtype), 'lengths': torch.full((2, dimension), 2., dtype=dtype),
        'source_lengths': torch.full((2, modules), .15, dtype=dtype),
        'source_measures': torch.rand(2, modules, generator=generator, dtype=dtype) + .2,
        'environment_tokens': torch.rand(2, environments, 6, generator=generator, dtype=dtype),
        'environment_coords': torch.rand(2, environments, dimension, generator=generator, dtype=dtype) * 2,
        'environment_present': torch.ones(2, environments, dtype=dtype),
        'environment_measures': torch.rand(2, environments, generator=generator, dtype=dtype) + .2,
        'source_ids': torch.arange(modules)[None].expand(2, -1)}
    query = torch.rand(2, 7, dimension, generator=generator, dtype=dtype) * 2
    if modules:
        query[:, 0] = centers[:, 0]
        query[:, 1] = centers[:, 0] + .45
    return arguments, query


def model(mode='group', dimension=2, zero_offset=True):
    torch.manual_seed(0)
    return SourceResponseOperator(4, 5, 6, mode=mode, spatial_dim=dimension,
        hidden=16, message=12, output_width=2, zero_offset=zero_offset, forcing_scale=3.).double()


def test_geometry_factor_control_preserves_degrees_and_actual_near_subtraction():
    operator = model()
    arguments, query = inputs()
    response = operator.prepare_receivers(operator.prepare_context(**arguments), query)
    membership = operator.geometry_matched_membership(response)
    torch.testing.assert_close(membership.sum(-1), response.group_present, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(membership.sum(1), response.source_membership.sum(1), atol=1e-6, rtol=1e-6)
    controlled = operator.intervene_groups(response, source_membership=membership)
    assert torch.equal(controlled.receiver_functions, response.receiver_functions)
    forcing = torch.tensor([[1.,2.,4.],[3.,2.,1.]], dtype=torch.float64)
    expected = torch.einsum('bqmo,bm->bqo', controlled.dense_kernel(), forcing)
    torch.testing.assert_close(operator.apply_forcing(controlled, forcing), expected, atol=2e-15, rtol=2e-13)
    # A source-centered receiver retains the exact own-source near kernel.
    torch.testing.assert_close(controlled.dense_kernel()[:,0,0], response.dense_kernel()[:,0,0])
    with pytest.raises(ValueError, match='iterations'): operator.geometry_matched_membership(response, 513)


@pytest.mark.parametrize('mode',['direct','group'])
@pytest.mark.parametrize('batch,queries,modules',[(8,1024,1),(8,1024,12),(1,8192,1),(1,8192,12)])
def test_declared_native_receiver_and_source_dimensions(mode,batch,queries,modules):
    arguments,_ = inputs(modules=modules,environments=2)
    arguments={key:value[:1].expand(batch,*value.shape[1:]).clone().float() for key,value in arguments.items()}
    operator=SourceResponseOperator(4,5,6,mode=mode,hidden=8,message=8,output_width=1).float()
    receivers=torch.rand(batch,queries,2)*2
    with torch.no_grad():
        response=operator.prepare_receivers(operator.prepare_context(**arguments),receivers,chunk_size=512)
        value=operator.apply_forcing(response,torch.ones(batch,modules))
    assert value.shape==(batch,queries,1) and bool(value.isfinite().all())
    assert (response.far_kernel is None)==(mode=='group')


@pytest.mark.parametrize('mode', ['direct', 'group'])
@pytest.mark.parametrize('dimension', [2, 3])
def test_affinity_actual_physical_derivative_and_explicit_forcing_boundary(mode, dimension):
    operator = model(mode, dimension, zero_offset=False)
    arguments, query = inputs(dimension)
    context = operator.prepare_context(**arguments)
    response = operator.prepare_receivers(context, query, chunk_size=2)
    forcing = torch.tensor([[1., 2., 4.], [3., 1., 2.]], dtype=torch.float64, requires_grad=True)
    delta = forcing.new_tensor([[.1, -.2, .1], [-.1, .3, -.2]])
    baseline = operator.apply_forcing(response, forcing)
    plus, minus = operator.apply_forcing(response, forcing + delta), operator.apply_forcing(response, forcing - delta)
    kernel = operator.export_response_operator(response)['K']
    expected = torch.einsum('bqmo,bm->bqo', kernel, delta)
    torch.testing.assert_close(plus - baseline, expected, atol=2e-15, rtol=2e-13)
    torch.testing.assert_close(plus + minus, baseline * 2, atol=2e-15, rtol=2e-13)
    torch.testing.assert_close(operator.apply_increment(response, delta), expected, atol=2e-15, rtol=2e-13)
    gradient, = torch.autograd.grad(baseline.sum(), forcing, create_graph=True)
    torch.testing.assert_close(gradient, kernel.sum((1, 3)), atol=2e-15, rtol=2e-13)
    # K itself has no forcing ancestry; the first derivative is constant in h.
    assert torch.autograd.grad(gradient.sum(), forcing, allow_unused=True)[0] is None
    with pytest.raises(TypeError, match='heat'):
        operator.prepare_context(**arguments, heat=forcing)
    with pytest.raises(TypeError, match='target'):
        operator.prepare_context(**arguments, target=torch.ones_like(forcing))
    repeated = operator.prepare_receivers(context, query)
    torch.testing.assert_close(repeated.dense_kernel(), kernel, atol=2e-15, rtol=2e-13)
    assert not any('channelthermal' in name or 'windfarm' in name for name in operator.__dict__)


def test_common_initialization_and_active_parameter_matching():
    direct, grouped = model('direct'), model('group')
    common = ('source_encoder.', 'environment_encoder.', 'module_messages.', 'environment_messages.',
              'source_updates.', 'environment_updates.', 'global_encoder.', 'near_head.')
    for name, value in direct.state_dict().items():
        if name.startswith(common): assert torch.equal(value, grouped.state_dict()[name])
    counts = [sum(parameter.numel() for parameter in operator.parameters()) for operator in (direct, grouped)]
    assert abs(counts[0] / counts[1] - 1) < .1
    assert direct.config['far_hidden'] > grouped.config['far_hidden']
    arguments, query = inputs()
    for operator in (direct, grouped):
        response = operator.prepare_receivers(operator.prepare_context(**arguments), query)
        value = operator.apply_forcing(response, torch.ones(2, 3, dtype=torch.float64))
        gradients = torch.autograd.grad(value.square().sum(), tuple(operator.parameters()))
        assert all(bool(torch.isfinite(gradient).all()) and bool(gradient.abs().max() > 0) for gradient in gradients)


@pytest.mark.parametrize('mode', ['direct', 'group'])
def test_source_permutation_padding_query_chunks_and_unequal_environment_splitting(mode):
    operator = model(mode)
    arguments, query = inputs()
    forcing = torch.randn(2, 3, dtype=torch.float64)
    original = operator.prepare_receivers(operator.prepare_context(**arguments), query)
    expected = operator.apply_forcing(original, forcing)
    permutation = torch.tensor([2, 0, 1])
    permuted = dict(arguments)
    for key in ('sources', 'centers', 'present', 'source_lengths', 'source_measures', 'source_ids'):
        permuted[key] = arguments[key][:, permutation]
    permuted_response = operator.prepare_receivers(operator.prepare_context(**permuted), query)
    torch.testing.assert_close(operator.apply_forcing(permuted_response, forcing[:, permutation]), expected, atol=2e-15, rtol=2e-13)
    torch.testing.assert_close(permuted_response.dense_kernel(), original.dense_kernel()[:, :, permutation], atol=2e-15, rtol=2e-13)
    padded = dict(arguments)
    for key in ('sources', 'centers', 'present', 'source_lengths', 'source_measures', 'source_ids'):
        value = arguments[key]
        shape = list(value.shape); shape[1] = 2
        padding = value.new_zeros(shape)
        if key in ('sources', 'centers'): padding = padding + 8
        padded[key] = torch.cat((value, padding), 1)
    padded_response = operator.prepare_receivers(operator.prepare_context(**padded), query)
    padded_forcing = torch.cat((forcing, forcing.new_full((2, 2), 999)), 1)
    torch.testing.assert_close(operator.apply_forcing(padded_response, padded_forcing), expected, atol=2e-15, rtol=2e-13)
    assert torch.count_nonzero(padded_response.dense_kernel()[:, :, 3:]) == 0
    split = dict(arguments)
    fractions = forcing.new_tensor([.2, .8]).repeat(arguments['environment_tokens'].shape[1])
    for key in ('environment_tokens', 'environment_coords', 'environment_present', 'environment_measures'):
        split[key] = arguments[key].repeat_interleave(2, 1)
    split['environment_measures'] = split['environment_measures'] * fractions
    split_response = operator.prepare_receivers(operator.prepare_context(**split), query)
    torch.testing.assert_close(operator.apply_forcing(split_response, forcing), expected, atol=2e-15, rtol=2e-13)
    reordered = torch.tensor([4, 1, 6, 0, 5, 2, 3])
    context = operator.prepare_context(**arguments)
    chunked = operator.prepare_receivers(context, query[:, reordered], chunk_size=1)
    torch.testing.assert_close(operator.apply_forcing(chunked, forcing)[:, reordered.argsort()], expected, atol=2e-15, rtol=2e-13)
    subset = operator.prepare_receivers(context, query[:, :2])
    torch.testing.assert_close(subset.dense_kernel(), original.dense_kernel()[:, :2], atol=2e-15, rtol=2e-13)


@pytest.mark.parametrize('modules,environments', [(0, 5), (1, 0), (12, 4)])
@pytest.mark.parametrize('mode', ['direct', 'group'])
def test_absent_types_and_native_source_sample_dimensions(modules, environments, mode):
    operator = model(mode)
    arguments, query = inputs(modules=modules, environments=environments)
    context = operator.prepare_context(**arguments)
    response = operator.prepare_receivers(context, query)
    forcing = torch.ones(2, modules, dtype=torch.float64)
    output = operator.apply_forcing(response, forcing)
    assert output.shape == (2, 7, 2) and torch.isfinite(output).all()
    assert response.dense_kernel().shape == (2, 7, modules, 2)
    if modules == 0: assert torch.count_nonzero(output) == 0


def test_group_efficient_dense_near_subtraction_values_input_gradients_and_mode_controls():
    operator = model()
    arguments, query = inputs()
    arguments['centers'] = arguments['centers'].clone().requires_grad_()
    arguments['environment_tokens'] = arguments['environment_tokens'].clone().requires_grad_()
    query = query.clone().requires_grad_()
    forcing = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    response = operator.prepare_receivers(operator.prepare_context(**arguments), query)
    efficient = operator.apply_forcing(response, forcing)
    dense = response.offset + torch.einsum('bqmo,bm->bqo', response.dense_kernel(), forcing)
    torch.testing.assert_close(efficient, dense, atol=2e-15, rtol=2e-13)
    leaves = (arguments['centers'], arguments['environment_tokens'], query, forcing)
    for actual, expected in zip(torch.autograd.grad(efficient.square().sum(), leaves, retain_graph=True),
                                torch.autograd.grad(dense.square().sum(), leaves, retain_graph=True)):
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-12)
        assert torch.isfinite(actual).all() and actual.norm() > 0
    assert response.far_kernel is None  # Production grouped read keeps factors.
    permuted = operator.intervene_groups(response, permutation=torch.tensor([3, 1, 0, 2]))
    torch.testing.assert_close(operator.apply_forcing(permuted, forcing), efficient, atol=2e-15, rtol=2e-13)
    removed = operator.intervene_groups(response, remove_mode=0)
    removed_output = operator.apply_forcing(removed, forcing)
    torch.testing.assert_close(removed_output,
        torch.einsum('bqmo,bm->bqo', removed.dense_kernel(), forcing), atol=2e-15, rtol=2e-13)
    assert (removed_output - efficient).abs().max() > 0
    # Exactly protected near coefficients remain unchanged by far-mode removal.
    protected = response.near_weight == 1
    torch.testing.assert_close(removed.dense_kernel()[protected], response.dense_kernel()[protected], atol=0, rtol=0)


def test_whole_layout_source_zero_is_distinct_from_removing_geometry_and_stale_guard():
    operator = model()
    arguments, query = inputs()
    context = operator.prepare_context(**arguments)
    original = operator.prepare_receivers(context, query)
    moved = dict(arguments); moved['centers'] = arguments['centers'].clone(); moved['centers'][:, 1] += .2
    changed = operator.prepare_receivers(operator.prepare_context(**moved), query)
    assert (changed.dense_kernel()[:, :, 0] - original.dense_kernel()[:, :, 0]).abs().max() > 1e-8
    forcing = torch.ones(2, 3, dtype=torch.float64); forcing[:, 1] = 0
    removed = dict(arguments); removed['present'] = arguments['present'].clone(); removed['present'][:, 1] = 0
    removed_response = operator.prepare_receivers(operator.prepare_context(**removed), query)
    assert (operator.apply_forcing(original, forcing) - operator.apply_forcing(removed_response, forcing)).abs().max() > 1e-8
    with torch.no_grad(): arguments['centers'].add_(.01)
    with pytest.raises(ValueError, match='changed'):
        operator.apply_forcing(original, forcing)


def test_geometry_near_support_and_labeled_nonlinear_forcing_conformance():
    operator = model()
    arguments, query = inputs()
    centers = torch.zeros(1, 1, 2, dtype=torch.float64)
    receivers = torch.tensor([[[0., 0.], [.2, 0.], [.3, 0.], [.4, 0.], [.5, 0.]]], dtype=torch.float64)
    weight = operator.near_weight(receivers, centers, torch.tensor([[.1]], dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64))
    torch.testing.assert_close(weight[0, :, 0], torch.tensor([1., 1., .5, 0., 0.], dtype=torch.float64), atol=1e-15, rtol=0)
    response = operator.prepare_receivers(operator.prepare_context(**arguments), query)
    forcing = torch.ones(2, 3, dtype=torch.float64)
    delta = forcing * .1
    # Synthetic nonlinear constitutive forcing; no claim of Wind/thermal affinity.
    mapping = lambda value: value.square()
    application = operator.apply_forcing(response, forcing, forcing_map=mapping, return_contributions=True)
    assert application.receipt['forcing_law'] == 'explicit caller nonlinear map'
    torch.testing.assert_close(application.values, application.source_contributions.sum(2), atol=2e-15, rtol=2e-13)
    expected = operator.apply_forcing(response, forcing + delta, forcing_map=mapping) - application.values
    torch.testing.assert_close(operator.apply_increment(response, delta, forcing_map=mapping, baseline_forcing=forcing), expected, atol=2e-15, rtol=2e-13)
    with pytest.raises(ValueError, match='baseline'):
        operator.apply_increment(response, delta, forcing_map=mapping)


def test_response_compression_bounds_use_effective_near_weights_and_keep_absolute_baseline():
    operator = model()
    arguments, query = inputs()
    response = operator.prepare_receivers(operator.prepare_context(**arguments), query)
    radii = torch.ones(2, 3, dtype=torch.float64)
    delta = radii.new_tensor([[.3, -.2, -.1], [-.2, .4, -.2]])
    exact = operator.apply_increment(response, delta)
    zero = operator.compress_response(response, radii, 0.)
    assert zero.keep.all()
    torch.testing.assert_close(operator.apply_increment(response, delta, compression=zero), exact, atol=0, rtol=0)
    compressed = operator.compress_response(response, radii, .01)
    approximation = operator.apply_increment(response, delta, compression=compressed)
    assert ((approximation - exact).abs() <= compressed.omitted_bound + 1e-14).all()
    assert (compressed.omitted_bound <= compressed.budget + 1e-14).all()
    baseline = operator.apply_forcing(response, radii)
    torch.testing.assert_close(operator.apply_forcing(response, radii), baseline, atol=0, rtol=0)
    with pytest.raises(ValueError, match='balanced'):
        operator.apply_increment(response, radii / 2, compression=compressed)
    with pytest.raises(ValueError, match='radii'):
        operator.apply_increment(response, delta * 10, compression=compressed)


def test_explicit_query_features_offset_and_checkpoint_roundtrip():
    arguments, query = inputs(3)
    operator = SourceResponseOperator(4, 5, 6, mode='group', spatial_dim=3, hidden=16,
        output_width=3, query_width=2, zero_offset=False).double()
    features = torch.randn(2, 7, 2, dtype=torch.float64)
    response = operator.prepare_receivers(operator.prepare_context(**arguments), query, receiver_features=features)
    forcing = torch.randn(2, 3, dtype=torch.float64)
    restored = SourceResponseOperator(**operator.config).double()
    restored.load_state_dict(operator.state_dict(), strict=True)
    replay = restored.prepare_receivers(restored.prepare_context(**arguments), query, receiver_features=features)
    torch.testing.assert_close(restored.apply_forcing(replay, forcing), operator.apply_forcing(response, forcing), atol=0, rtol=0)
    assert response.offset.abs().max() > 0
    with pytest.raises(ValueError, match='receiver_features'):
        operator.prepare_receivers(operator.prepare_context(**arguments), query)
    changed = replace(response, receiver_ids=torch.arange(7)[None].expand(2, -1))
    assert operator.export_response_operator(changed)['receiver_ids'].shape == (2, 7)
