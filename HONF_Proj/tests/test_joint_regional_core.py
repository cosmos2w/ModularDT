"""Focused contracts for the source-preserving joint regional core."""
from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from honf_forward_core.interface_fields.interaction_core import DependencySpec, InteractionScene
from honf_forward_core.interface_fields.joint_regional import JointRegionalFieldCore


def _inputs(*, batch=2, modules=4, environments=5, dimension=2, query_width=3):
    generator = torch.Generator().manual_seed(42017)
    sources = torch.randn(batch, modules, 4, generator=generator)
    context = torch.randn(batch, 2, generator=generator)
    centers = torch.rand(batch, modules, dimension, generator=generator) * 0.75 + 0.1
    present = torch.ones(batch, modules)
    if modules:
        present[0, -1] = 0
    lengths = torch.ones(batch, dimension)
    domain_origin = torch.zeros(batch, dimension)
    source_lengths = torch.full((batch, modules), 0.16)
    source_measures = torch.linspace(0.5, 1.5, max(modules, 1))[:modules][None].expand(batch, -1).clone()
    source_ids = (torch.arange(modules)[None] + 20).expand(batch, -1).clone()
    environment_tokens = torch.randn(batch, environments, 3, generator=generator)
    environment_coords = torch.rand(batch, environments, dimension, generator=generator)
    environment_present = torch.ones(batch, environments)
    if environments:
        environment_present[0, -1] = 0
    environment_measures = torch.linspace(0.3, 1.7, max(environments, 1))[:environments][None].expand(batch, -1).clone()
    receivers = torch.rand(batch, 7, dimension, generator=generator)
    receiver_features = torch.randn(batch, 7, query_width, generator=generator)
    return {
        'sources': sources, 'context': context, 'centers': centers, 'present': present,
        'lengths': lengths, 'domain_origin': domain_origin,
        'source_lengths': source_lengths, 'source_measures': source_measures,
        'environment_tokens': environment_tokens, 'environment_coords': environment_coords,
        'environment_present': environment_present, 'environment_measures': environment_measures,
        'source_ids': source_ids, 'receivers': receivers, 'receiver_features': receiver_features,
    }


def _core(mode='J-H', *, affine_outputs=1, field_outputs=2, dimension=2, query_width=3, seed=321):
    return JointRegionalFieldCore(
        4, 2, 3, spatial_dim=dimension, hidden=16, message=12, mode=mode,
        regional_anchors=4, depth=2, field_outputs=field_outputs,
        affine_outputs=affine_outputs, query_width=query_width,
        initialization_seed=seed)


def _prepare(core, values):
    kwargs = {name: values[name] for name in (
        'sources', 'context', 'centers', 'present', 'lengths', 'source_lengths',
        'source_measures', 'environment_tokens', 'environment_coords',
        'environment_present', 'environment_measures', 'source_ids', 'domain_origin')}
    return core.prepare_context(**kwargs)


def test_shared_parameters_initialize_identically_across_geometry_and_learned_modes():
    learned = _core('J-H', seed=987)
    geometry = _core('J-geometry', seed=987)
    learned_parameters = dict(learned.named_parameters())
    geometry_parameters = dict(geometry.named_parameters())
    common = learned_parameters.keys() & geometry_parameters.keys()
    assert common
    for name in common:
        torch.testing.assert_close(learned_parameters[name], geometry_parameters[name], rtol=0, atol=0)


def test_both_output_heads_train_collective_membership_and_receiver_access():
    values = _inputs()
    core = _core().train()
    prepared = _prepare(core, values)
    fields = core.predict_fields(prepared, values['receivers'], values['receiver_features'])
    response = core.prepare_receivers(prepared, values['receivers'], values['receiver_features'])
    thermal = core.apply_forcing(response, torch.tensor([[0.5, 1.2, 0.8, 0.0], [1.0, 0.2, 0.7, 1.4]]))
    assert fields.shape == (2, 7, 2)
    assert thermal.shape == (2, 7)
    assert response.dense_kernel().shape == (2, 7, 4, 1)
    assert response.context.source_ids is values['source_ids']

    organizer_parameters = [
        *core.source_membership_score.parameters(),
        *core.environment_membership_score.parameters(),
        *core.edge_updates.parameters(),
        *core.receiver_query.parameters(), *core.edge_key.parameters(),
        *core.access_geometry.parameters(),
    ]
    flow_gradients = torch.autograd.grad(fields.square().mean(), organizer_parameters,
                                         allow_unused=True, retain_graph=True)
    thermal_gradients = torch.autograd.grad(thermal.square().mean(), organizer_parameters,
                                            allow_unused=True)
    for gradients in (flow_gradients, thermal_gradients):
        assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 1e-8
    assert core.source_membership_score[-1].bias is None
    assert core.environment_membership_score[-1].bias is None
    assert core.access_geometry[-1].bias is None


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('mode', ['J-H', 'J-geometry', 'J-direct'])
def test_source_environment_and_receiver_permutations_preserve_outputs(mode, dimension):
    values = _inputs(dimension=dimension)
    core = _core(mode, affine_outputs=0, dimension=dimension).eval()
    prepared = _prepare(core, values)
    expected = core.predict_fields(prepared, values['receivers'], values['receiver_features'])

    source_order = torch.tensor([2, 0, 3, 1])
    source_values = dict(values)
    for name in ('sources', 'centers', 'present', 'source_lengths', 'source_measures', 'source_ids'):
        source_values[name] = values[name][:, source_order]
    source_result = core.predict_fields(_prepare(core, source_values), values['receivers'], values['receiver_features'])
    permuted_context = _prepare(core, source_values)

    environment_order = torch.tensor([3, 1, 4, 0, 2])
    environment_values = dict(values)
    for name in ('environment_tokens', 'environment_coords', 'environment_present', 'environment_measures'):
        environment_values[name] = values[name][:, environment_order]
    environment_result = core.predict_fields(_prepare(core, environment_values), values['receivers'], values['receiver_features'])

    query_order = torch.tensor([5, 1, 6, 0, 2, 4, 3])
    query_result = core.predict_fields(prepared, values['receivers'][:, query_order],
                                       values['receiver_features'][:, query_order])
    torch.testing.assert_close(source_result, expected, rtol=3e-5, atol=3e-6)
    if mode != 'J-direct':
        expected_edge_ids = torch.where(source_values['present'] > 0,
                                        source_values['source_ids'],
                                        torch.full_like(source_values['source_ids'], -1))
        torch.testing.assert_close(permuted_context.edge_source_ids[:, :4], expected_edge_ids, rtol=0, atol=0)
    torch.testing.assert_close(environment_result, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(query_result, expected[:, query_order], rtol=2e-6, atol=2e-6)


def test_chunking_and_affine_application_keep_receiver_and_source_shapes():
    values = _inputs()
    core = _core().eval()
    prepared = _prepare(core, values)
    whole = core.predict_fields(prepared, values['receivers'], values['receiver_features'], chunk_size=16)
    chunked = core.predict_fields(prepared, values['receivers'], values['receiver_features'], chunk_size=2)
    torch.testing.assert_close(chunked, whole, rtol=2e-6, atol=2e-6)
    whole_response = core.prepare_receivers(prepared, values['receivers'], values['receiver_features'], chunk_size=16)
    chunked_response = core.prepare_receivers(prepared, values['receivers'], values['receiver_features'], chunk_size=2)
    torch.testing.assert_close(chunked_response.dense_kernel(), whole_response.dense_kernel(), rtol=2e-6, atol=2e-6)
    forcing = torch.tensor([[0.5, 1.2, 0.8, 0.0], [1.0, 0.2, 0.7, 1.4]])
    delta = torch.tensor([[0.1, -0.2, 0.0, 0.0], [-0.1, 0.2, -0.3, 0.4]])
    kernel = core.read_kernel(prepared, values['receivers'], receiver_features=values['receiver_features'])
    assert kernel.shape == (2, 7, 4)
    applied = core.apply_forcing(whole_response, forcing)
    increment = core.apply_increment(whole_response, delta, accumulation_dtype=torch.float64)
    torch.testing.assert_close(applied, torch.einsum('bqm,bm->bq', kernel, forcing), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(increment, torch.einsum('bqm,bm->bq', kernel.double(), delta.double()),
                               rtol=1e-10, atol=1e-10)
    precise = core.apply_forcing(whole_response, forcing, accumulation_dtype=torch.float64)
    torch.testing.assert_close(precise, torch.einsum('bqm,bm->bq', kernel.double(), forcing.double()),
                               rtol=1e-10, atol=1e-10)


def test_empty_environment_and_masked_types_produce_finite_valid_state():
    values = _inputs(environments=0)
    core = _core().eval()
    prepared = _prepare(core, values)
    output = core.predict_fields(prepared, values['receivers'], values['receiver_features'])
    assert prepared.environment_membership.shape == (2, 8, 0)
    assert torch.isfinite(output).all()
    assert prepared.source_membership[0, :, -1].count_nonzero() == 0

    all_masked = dict(values)
    all_masked['present'] = torch.zeros_like(values['present'])
    all_masked['source_lengths'] = torch.zeros_like(values['source_lengths'])
    all_masked['environment_present'] = torch.zeros_like(values['environment_present'])
    masked_context = _prepare(core, all_masked)
    masked = core.predict_fields(masked_context, values['receivers'], values['receiver_features'])
    assert torch.isfinite(masked).all()
    assert masked_context.source_membership.count_nonzero() == 0
    assert masked_context.environment_membership.count_nonzero() == 0


def test_splitting_an_environment_atom_preserves_measure_weighted_prediction():
    values = _inputs(batch=1, environments=4)
    core = _core().eval()
    base = core.predict_fields(_prepare(core, values), values['receivers'], values['receiver_features'])
    split = dict(values)
    split['environment_tokens'] = torch.cat((values['environment_tokens'][:, :1],
                                             values['environment_tokens'][:, :1],
                                             values['environment_tokens'][:, 1:]), 1)
    split['environment_coords'] = torch.cat((values['environment_coords'][:, :1],
                                             values['environment_coords'][:, :1],
                                             values['environment_coords'][:, 1:]), 1)
    split['environment_present'] = torch.cat((values['environment_present'][:, :1],
                                              values['environment_present'][:, :1],
                                              values['environment_present'][:, 1:]), 1)
    split['environment_measures'] = torch.cat((values['environment_measures'][:, :1] * 0.5,
                                               values['environment_measures'][:, :1] * 0.5,
                                               values['environment_measures'][:, 1:]), 1)
    split_result = core.predict_fields(_prepare(core, split), values['receivers'], values['receiver_features'])
    torch.testing.assert_close(split_result, base, rtol=4e-5, atol=4e-6)
    base_context = _prepare(core, values)
    split_context = _prepare(core, split)
    torch.testing.assert_close(split_context.environment_membership.sum(-1),
                               base_context.environment_membership.sum(-1), rtol=4e-5, atol=4e-6)
    torch.testing.assert_close(split_context.environment_membership_density[:, :, :1],
                               base_context.environment_membership_density[:, :, :1],
                               rtol=4e-5, atol=4e-6)


def test_registered_anchors_respect_shifted_domain_and_physical_source_ids():
    values = _inputs(batch=2)
    origin = torch.tensor([[-2.0, -1.25], [3.5, -4.0]])
    values['domain_origin'] = origin
    for name in ('centers', 'environment_coords', 'receivers'):
        values[name] = origin[:, None] + values[name]
    core = _core().eval()
    prepared = _prepare(core, values)
    regional = prepared.group_centers[:, 4:]
    assert torch.all(regional >= origin[:, None])
    assert torch.all(regional <= origin[:, None] + values['lengths'][:, None])
    expected_edge_ids = torch.where(values['present'] > 0, values['source_ids'],
                                    torch.full_like(values['source_ids'], -1))
    torch.testing.assert_close(prepared.edge_source_ids[:, :4], expected_edge_ids, rtol=0, atol=0)


def test_interventions_change_executed_path_and_restore_original_prediction():
    values = _inputs()
    core = _core().eval()
    original = _prepare(core, values)
    expected = core.predict_fields(original, values['receivers'], values['receiver_features'])
    removed_content = core.intervene_context(original, remove_collective_content=True)
    content_result = core.predict_fields(removed_content, values['receivers'], values['receiver_features'])
    geometry_membership = core.intervene_context(original, incidence_mode='geometry')
    geometry_result = core.predict_fields(geometry_membership, values['receivers'], values['receiver_features'])
    removed_update = core.intervene_context(original, remove_collective_update=True)
    update_result = core.predict_fields(removed_update, values['receivers'], values['receiver_features'])
    shuffled = core.intervene_context(original, matched_mass_shuffle=True)
    shuffle_result = core.predict_fields(shuffled, values['receivers'], values['receiver_features'])
    geometry_access = core.intervene_context(original, receiver_access='geometry')
    geometry_access_response = core.prepare_receivers(geometry_access, values['receivers'], values['receiver_features'])
    catalogue = core.average_receiver_access(
        original, values['receivers'], values['receiver_features'],
        receiver_ids=torch.arange(7)[None].expand(2, -1))
    scene_access = core.intervene_context(original, receiver_access='scene-average',
                                          scene_average_access=catalogue)
    scene_result = core.predict_fields(scene_access, values['receivers'], values['receiver_features'])
    uniform_access = core.intervene_context(original, receiver_access='uniform-average')
    uniform_result = core.predict_fields(uniform_access, values['receivers'], values['receiver_features'])
    restored = core.restore(removed_content)
    restored_result = core.predict_fields(restored, values['receivers'], values['receiver_features'])
    assert not torch.allclose(content_result, expected, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(geometry_result, expected, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(update_result, expected, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(shuffle_result, expected, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(scene_result, expected, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(uniform_result, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(shuffled.source_membership.sum(-1), original.source_membership.sum(-1))
    assert geometry_access_response.receiver_access.shape == (2, 7, 8)
    assert not torch.allclose(geometry_access_response.receiver_access,
                              core.prepare_receivers(original, values['receivers'], values['receiver_features']).receiver_access)
    torch.testing.assert_close(restored_result, expected, rtol=0, atol=0)
    response = core.prepare_receivers(original, values['receivers'], values['receiver_features'])
    export = core.export_organization(original, response)
    assert export['receiver_access'].shape == (2, 7, 8)
    assert export['source_membership_density'].shape == export['source_membership_mass'].shape
    assert export['edge_to_source'].shape == (2, 4, core.hidden)
    assert export['edge_to_environment'].shape == (2, 5, core.hidden)
    assert len(export['source_membership_mass_history']) == core.depth
    assert len(export['environment_membership_density_history']) == core.depth
    assert len(export['edge_state_history']) == core.depth
    assert len(export['edge_to_source_history']) == core.depth


def test_scene_contract_control_exclusion_and_stale_state_rejection():
    values = _inputs()
    core = _core()
    dependency = DependencySpec(
        dataset='toy', output_law='affine', configuration_inputs=('geometry',),
        applicable_controls=('heating',), output_roles=('temperature',), units=('K',))
    scene = InteractionScene(
        sources=values['sources'], context=values['context'], centers=values['centers'],
        present=values['present'], lengths=values['lengths'], source_lengths=values['source_lengths'],
        source_measures=values['source_measures'], environment_tokens=values['environment_tokens'],
        environment_coords=values['environment_coords'], environment_present=values['environment_present'],
        environment_measures=values['environment_measures'], source_ids=values['source_ids'],
        dependency=dependency)
    prepared = core.prepare(scene)
    assert prepared.dependency is dependency
    assert core.output_law == 'joint'
    assert core.field_output_law == 'nonlinear'
    assert core.response_output_law == 'affine'
    with pytest.raises(TypeError):
        core.prepare_context(**{**{key: values[key] for key in (
            'sources', 'context', 'centers', 'present', 'lengths', 'source_lengths')}, 'heat': values['source_measures']})

    values = _inputs()
    prepared = _prepare(core, values)
    with torch.no_grad():
        values['domain_origin'].add_(0.01)
    with pytest.raises(ValueError, match='changed'):
        core.assert_owned(prepared)

    values = _inputs()
    prepared = _prepare(core, values)
    with torch.no_grad():
        next(core.parameters()).add_(0.01)
    with pytest.raises(ValueError, match='weights changed'):
        core.assert_owned(prepared)

    values = _inputs()
    prepared = _prepare(core, values)
    with torch.no_grad():
        values['source_ids'][0, 0] += 100
    with pytest.raises(ValueError, match='changed'):
        core.assert_owned(prepared)


def test_direct_control_has_no_edges_and_wind_can_disable_affine_head():
    values = _inputs()
    direct = _core('J-direct', affine_outputs=0, field_outputs=3).eval()
    assert direct.output_law == 'joint'
    assert direct.response_output_law is None
    assert direct.field_output_law == 'nonlinear'
    prepared = _prepare(direct, values)
    prediction = direct.predict(prepared, values['receivers'], values['receiver_features'])
    assert prediction.shape == (2, 7, 3)
    assert prepared.group_states.shape == (2, 0, direct.hidden)
    assert direct.export_organization(prepared)['receiver_access'] is None
    gradients = torch.autograd.grad(prediction.square().mean(), tuple(direct.parameters()), allow_unused=True)
    assert all(gradient is not None for gradient in gradients)
    assert all(float(gradient.abs().sum()) > 0 for gradient in gradients)
    with pytest.raises(ValueError, match='no separately applicable'):
        direct.prepare_receivers(prepared, values['receivers'], values['receiver_features'])


def test_scene_average_uses_fixed_catalogue_and_is_query_batch_independent():
    values = _inputs()
    core = _core().eval()
    prepared = _prepare(core, values)
    ids = torch.arange(7)[None].expand(2, -1)
    average_one = core.average_receiver_access(
        prepared, values['receivers'], values['receiver_features'], ids, chunk_size=1)
    average_many = core.average_receiver_access(
        prepared, values['receivers'], values['receiver_features'], ids, chunk_size=5)
    torch.testing.assert_close(average_one['access'], average_many['access'], rtol=2e-6, atol=2e-6)

    order = torch.tensor([4, 1, 6, 0, 5, 3, 2])
    average_permuted = core.average_receiver_access(
        prepared, values['receivers'][:, order], values['receiver_features'][:, order],
        ids[:, order], chunk_size=3)
    torch.testing.assert_close(average_one['access'], average_permuted['access'], rtol=2e-6, atol=2e-6)

    scene = core.intervene_context(prepared, receiver_access='scene-average',
                                   scene_average_access=average_one)
    prediction_whole = core.predict_fields(scene, values['receivers'], values['receiver_features'],
                                           chunk_size=7)
    prediction_chunked = core.predict_fields(scene, values['receivers'], values['receiver_features'],
                                             chunk_size=2)
    query_order = torch.tensor([6, 0, 4, 2, 5, 1, 3])
    prediction_permuted = core.predict_fields(
        scene, values['receivers'][:, query_order], values['receiver_features'][:, query_order],
        chunk_size=3)
    torch.testing.assert_close(prediction_whole, prediction_chunked, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(prediction_permuted, prediction_whole[:, query_order],
                               rtol=2e-6, atol=2e-6)
    restored = core.restore(scene)
    baseline = core.predict_fields(prepared, values['receivers'], values['receiver_features'])
    torch.testing.assert_close(core.predict_fields(restored, values['receivers'], values['receiver_features']),
                               baseline, rtol=0, atol=0)
    with pytest.raises(ValueError, match='explicit fixed-catalogue'):
        core.intervene_context(prepared, receiver_access='scene-average')


def test_dense_access_mass_retention_receipts_and_same_k_geometry_control():
    values = _inputs()
    core = _core().eval()
    prepared = _prepare(core, values)
    full_fields = core.predict_fields(prepared, values['receivers'], values['receiver_features'])
    full_fields_100, full_export = core.predict_fields(
        prepared, values['receivers'], values['receiver_features'],
        retained_access_mass=1.0, return_organization=True)
    torch.testing.assert_close(full_fields_100, full_fields, rtol=0, atol=0)
    assert full_export['execution'].startswith('dense receiver-edge')
    counts_by_level = []
    retained_by_level = {}
    for level in (1.0, 0.99, 0.95, 0.90):
        fields, organization = core.predict_fields(
            prepared, values['receivers'], values['receiver_features'], chunk_size=2,
            retained_access_mass=level, return_organization=True)
        response = core.prepare_receivers(
            prepared, values['receivers'], values['receiver_features'], chunk_size=3,
            retained_access_mass=level)
        torch.testing.assert_close(fields, core.predict_fields(
            prepared, values['receivers'], values['receiver_features'], chunk_size=7,
            retained_access_mass=level), rtol=2e-6, atol=2e-6)
        assert torch.isfinite(fields).all()
        assert organization['retained_edge_counts'].shape == (2, 7)
        assert response.retention_receipt['retained_edge_counts'].shape == (2, 7)
        assert (organization['retained_access_mass_actual'] >= level - 1e-6).all()
        assert organization['execution'].startswith('dense receiver-edge')
        counts_by_level.append(organization['retained_edge_counts'])
        retained_by_level[level] = organization
    for higher, lower in pairwise(counts_by_level):
        assert (higher >= lower).all()

    learned_counts = retained_by_level[0.95]['retained_edge_counts']
    geometry_fields, geometry_export = core.predict_fields(
        prepared, values['receivers'], values['receiver_features'], access_mode='geometry',
        retained_edge_counts=learned_counts, return_organization=True)
    assert torch.isfinite(geometry_fields).all()
    torch.testing.assert_close(geometry_export['retained_edge_counts'], learned_counts)
    assert (geometry_export['retained_edge_counts'] == learned_counts).all()

    exact_counts = torch.full((2, 7), 2, dtype=torch.long)
    tied_read = core.read_receiver(
        prepared, values['receivers'], values['receiver_features'],
        access_mode='uniform-average', retained_edge_counts=exact_counts)
    source_order = torch.tensor([2, 0, 3, 1])
    permuted_values = dict(values)
    for name in ('sources', 'centers', 'present', 'source_lengths', 'source_measures', 'source_ids'):
        permuted_values[name] = values[name][:, source_order]
    permuted_context = _prepare(core, permuted_values)
    permuted_read = core.read_receiver(
        permuted_context, values['receivers'], values['receiver_features'],
        access_mode='uniform-average', retained_edge_counts=exact_counts)
    for selected_context, selected_read in ((prepared, tied_read), (permuted_context, permuted_read)):
        selected_ids = []
        for batch_index in range(2):
            active = selected_read['receiver_access'][batch_index, 0, :4] > 0
            selected_ids.append(sorted(selected_context.edge_source_ids[batch_index, :4][active].tolist()))
        assert selected_ids == [[20, 21], [20, 21]]

    retained_read = core.read_receiver(
        prepared, values['receivers'], values['receiver_features'], retained_edge_counts=torch.zeros(2, 7, dtype=torch.long))
    assert retained_read['receiver_access'].count_nonzero() == 0
    assert retained_read['source_features'].abs().sum() > 0
    retained_response = core.prepare_receivers(
        prepared, values['receivers'], values['receiver_features'],
        retained_access_mass=0.95)
    exported = core.export_organization(prepared, retained_response)
    assert exported['receiver_retention']['retained_edge_counts'].shape == (2, 7)
