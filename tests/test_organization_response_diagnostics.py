"""Finite original-member changes, actual stages and bounded controls."""
from dataclasses import fields, replace

import pytest
import torch

from honf_forward_core.interaction_reused_member_routing.detail import DetailAnchoredMemberRoutingCore
from honf_forward_core.interaction_reused_member_routing.organization_response import (
    capture_applied_relationships, contextual_coefficients_trace_request, finite_read_decomposition,
    prepare_relation_removal, select_context_relations,
)
from tests.anchored_fixtures import _base_config
from tests.test_e_anchored_packed_execution import fixture


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def one_scene():
    _, scene, query = fixture('local_e_cover')
    scene = replace(scene, **{f.name: value[:1].clone() for f in fields(scene)
                              if isinstance((value := getattr(scene, f.name)), torch.Tensor)
                              and value.ndim and value.shape[0] == scene.batch_size})
    query = replace(query, coordinates=query.coordinates[:1].detach(),
                    anchor_indices=query.anchor_indices[:1], anchor_weights=query.anchor_weights[:1],
                    query_mask=query.query_mask[:1], query_ids=query.query_ids[:1])
    base = {**_base_config(), 'output_dim': 7}
    model = DetailAnchoredMemberRoutingCore(base, query_tile_size=4, retained_mass=.42).double().eval()
    # Change two prescribed inputs while preserving individual original IDs.
    changed_features = scene.module_features.clone()
    changed_features[0, 0] += 3.0
    changed_environment = scene.environment_features.clone()
    changed_environment[0, 1] -= 4.0
    coordinates = scene.module_coordinates.clone()
    coordinates[0, 0] += torch.tensor([.15, -.1], dtype=coordinates.dtype)
    edited = replace(scene, module_features=changed_features, environment_features=changed_environment,
                     module_coordinates=coordinates)
    return model, scene, edited, query


@pytest.mark.parametrize('mode', ['full', 'restricted'])
def test_exact_decomposition_and_nonlinear_head_for_seven_fields(mode):
    model, scene, edited, query = one_scene()
    parameters_before = {name: value.clone() for name, value in model.state_dict().items()}
    left, right = model.prepare(scene, return_trace=True), model.prepare(edited, return_trace=True)
    result = finite_read_decomposition(model, left, query, right, query, identity_confirmed=True, read_mode=mode)
    assert result['available']
    torch.testing.assert_close(result['context_reconstruction_residual'],
                               torch.zeros_like(result['context_reconstruction_residual']), atol=1e-12, rtol=0)
    torch.testing.assert_close(result['field_allocation_residual'],
                               torch.zeros_like(result['field_allocation_residual']), atol=1e-12, rtol=0)
    assert result['field_before'].shape == (1, 4, 7)
    assert sum(result['work']['diagnostic_message_recomputation'][r]['union_pairs'] for r in ('M', 'E')) > 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, parameters_before[name], rtol=0, atol=0)


def test_union_messages_are_from_both_original_states_even_when_support_changes():
    model, scene, edited, query = one_scene()
    # Route to different actual E-address rows while keeping the same query.
    weights_before = torch.zeros_like(query.anchor_weights)
    weights_after = torch.zeros_like(query.anchor_weights)
    weights_before[..., 0] = 1
    weights_after[..., 1] = 1
    query_before = replace(query, anchor_weights=weights_before)
    query_after = replace(query, anchor_weights=weights_after)
    left, right = model.prepare(scene), model.prepare(edited)
    result = finite_read_decomposition(model, left, query_before, right, query_after,
                                       identity_confirmed=True, read_mode='restricted')
    assert result['available']
    counts = result['work']['diagnostic_message_recomputation']
    assert sum(counts[r]['missing_from_before_support'] + counts[r]['missing_from_after_support']
               for r in ('M', 'E')) > 0
    for role in ('M', 'E'):
        values = result['roles'][role]
        assert torch.isfinite(values['message_before']).all() and torch.isfinite(values['message_after']).all()
        assert counts[role]['diagnostic_message_pairs_before'] == counts[role]['union_pairs']
        assert counts[role]['diagnostic_message_pairs_after'] == counts[role]['union_pairs']
    torch.testing.assert_close(result['context_reconstruction_residual'],
                               torch.zeros_like(result['context_reconstruction_residual']), atol=1e-12, rtol=0)


def test_moving_queries_incompatible_identity_and_stale_states_are_never_attributed():
    model, scene, edited, query = one_scene()
    left, right = model.prepare(scene), model.prepare(edited)
    moving = replace(query, coordinates=query.coordinates + .01)
    result = finite_read_decomposition(model, left, query, right, moving, identity_confirmed=True)
    assert not result['available'] and 'changed' in result['reason']
    changed_id = edited.module_ids.clone(); changed_id[0, 0] += 1000
    incompatible = model.prepare(replace(edited, module_ids=changed_id))
    result = finite_read_decomposition(model, left, query, incompatible, query, identity_confirmed=True)
    assert not result['available'] and 'Insertion' in result['reason']
    with pytest.raises(ValueError, match='confirm persistent'):
        finite_read_decomposition(model, left, query, right, query)
    left.member_values_M.add_(1.)
    with pytest.raises(ValueError, match='projections changed'):
        finite_read_decomposition(model, left, query, right, query, identity_confirmed=True)


def test_actual_stage_selection_recomputation_and_functional_access_control():
    model, scene, edited, query = one_scene()
    left = model.prepare(scene, trace_request=contextual_coefficients_trace_request(model))
    relationships = capture_applied_relationships(model, left)
    assert {'mm', 'em', 'ee', 'me'} <= {x['route'] for x in relationships['contextual_stages']}
    for stage in relationships['contextual_stages']:
        assert stage['stage'].endswith('/' + stage['route'])
        assert stage['receiver_role'].lower() == stage['route'][0]
        assert stage['donor_role'].lower() == stage['route'][1]
        assert stage['coefficients'] is not None
        torch.testing.assert_close(stage['coefficients'].sum(-1).transpose(1, 2), stage['message_mass'],
                                   rtol=1e-10, atol=1e-11)
    choices = select_context_relations(relationships, seed=8, count=4)
    assert choices == select_context_relations(relationships, seed=8, count=4)
    no_coefficients = capture_applied_relationships(model, model.prepare(scene, return_trace=True))
    with pytest.raises(ValueError, match='Request actual contextual coefficients'):
        select_context_relations(no_coefficients, count=1)
    # A role with no executed relation stays an explicit zero-support record;
    # it must not prevent selecting relations on other roles.
    zero_stage = relationships['contextual_stages'][0]
    zero_stage['gate'].zero_()
    zero_stage['coefficients'] = None
    assert select_context_relations(relationships, seed=8, count=1)
    for relation in choices:
        altered = prepare_relation_removal(model, scene, relation)
        prepare_relation_removal(model, edited, relation)
        assert altered is not left
    with pytest.raises(ValueError, match='stage'):
        prepare_relation_removal(model, scene, {**choices[0], 'stage': 'cycle999/' + choices[0]['route']})
    # Existing functional access controls operate on fresh reads without
    # modifying the prepared directory or changing model parameters.
    before = model.decode(left, query, intervention={'read_mode': 'restricted'})
    control = model.decode(left, query, intervention={'read_mode': 'restricted',
                                                       'within_role_access_prior': 'measure_only'})
    after = model.decode(left, query, intervention={'read_mode': 'restricted'})
    torch.testing.assert_close(before.joint_fields, after.joint_fields, atol=0, rtol=0)
    assert control.work_counts['intervention_measure_only_access_entries'] > 0


def test_saved_capture_bridge_preserves_real_stage_ids_and_prepared_rows(tmp_path):
    import json
    import numpy as np
    from organization_response_to_global import convert
    from thermal_organization_response_diagnostic import _flatten, _geometry
    from honf_viz.organization_data import load_scene

    model, scene, _, query = one_scene()
    prepared = model.prepare(scene, return_trace=True)
    relation = capture_applied_relationships(model, prepared)
    geometry = _geometry(scene, {'domain': {'lx': 1.3, 'ly': 1., 'module_radius': .1}})
    arrays = {}
    capture = _flatten({'relationships': {'before': relation}, 'geometry': {'before': geometry}}, arrays)
    source = tmp_path / 'capture.json'
    np.savez_compressed(tmp_path / 'capture.npz', **arrays)
    source.write_text(json.dumps({'schema': 'thermal_saved_edit_organization_response_v1',
        'arrays': 'capture.npz', 'capture': capture, 'family': 'fixture', 'action': 'edit', 'role': 'fit',
        'epoch': 5, 'checkpoint': 'fixture_checkpoint', 'query_selection': 'declared fixed positions',
        'read_mode': 'restricted', 'within_role_access_prior': 'learned', 'retained_mass_setting': .42}))
    destination = tmp_path / 'global.npz'
    convert(source, destination, 'before')
    result = load_scene('Fixture', destination)
    assert result.m_ids == tuple('M' + str(i) for i in scene.module_ids[0].tolist())
    assert result.e_ids == tuple('E' + str(i) for i in scene.environment_ids[0].tolist())
    assert set(result.gates) == {'mm', 'em', 'ee', 'me'}
    final = next(s for s in relation['contextual_stages'] if s['stage'] == 'cycle1/em')
    np.testing.assert_array_equal(result.gates['em'], final['gate'][0].numpy())
    restricted = prepared.e_access_directory.restricted
    for row in range(scene.num_environment_tokens):
        values = restricted.row(0, row)
        np.testing.assert_array_equal(result.C[row, values.member_index.numpy()], values.access_weight.detach().numpy())
    assert json.loads(destination.with_suffix('.json').read_text())['capture_source'] == 'capture.json'
