"""Same-checkpoint observation and access-work evaluation on a real core."""
from copy import deepcopy
import json

import pytest
import torch

from honf_forward_core.content_hypergraph.query_chunks_v1 import slice_query_v1
from honf_runtime.access_prior_evaluation import _timed, evaluate_access_prior_panel
from honf_runtime.observation_response_task import build_observation_response_task
from tools.observation_response_evaluate import evaluate_observation_panel


def _recipe(width=7):
    return {'dataset': 'analytic_fixture', 'seed': 3, 'architecture': 'relation_anchored', 'total_epochs': 2,
            'training_profile': 'observation_response', 'fixture': {'output_dim': width},
            'observation_response': {'variant': 'spatial_contrasts', 'seeds_per_pair': 8}}


@pytest.mark.parametrize('width', [2, 7])
def test_same_prepared_scene_policies_match_existing_observation_evaluator_and_trace(monkeypatch, width):
    torch.set_num_threads(1)
    model, provider = build_observation_response_task(_recipe(width), device='cpu')
    model.train()
    calls, preparations = [], []
    original_prepare, original_pair, original_decode = model.core.prepare, model.core.decode_paired, model.core.decode

    def prepare(scene, **kwargs):
        result = original_prepare(scene, **kwargs)
        preparations.append((scene.scene_key, result))
        return result

    def paired(prepared, query, **kwargs):
        calls.append(('paired', prepared, query, kwargs.get('intervention')))
        return original_pair(prepared, query, **kwargs)

    def decode(prepared, query, **kwargs):
        calls.append(('restricted', prepared, query, kwargs.get('intervention')))
        return original_decode(prepared, query, **kwargs)

    monkeypatch.setattr(model.core, 'prepare', prepare)
    monkeypatch.setattr(model.core, 'decode_paired', paired)
    monkeypatch.setattr(model.core, 'decode', decode)
    pairs = [('fixture_0', 'fixture_1'), ('fixture_0', 'fixture_2')]
    result = evaluate_access_prior_panel(model, provider, pairs)
    assert model.training  # evaluation restores the caller's mode
    assert result['pair_count'] == 2 and result['scene_preparations'] == 3
    assert len(preparations) == 3 and len({key for key, _ in preparations}) == 3
    assert len(calls) == 12
    for start in range(0, len(calls), 4):
        group = calls[start:start + 4]
        assert [call[0] for call in group] == ['paired', 'restricted', 'paired', 'restricted']
        assert len({id(call[1]) for call in group}) == len({id(call[2]) for call in group}) == 1
        assert group[0][3] is None and group[1][3] == {'read_mode': 'restricted'}
        assert group[2][3] == {'within_role_access_prior': 'measure_only'}
        assert group[3][3] == {'read_mode': 'restricted', 'within_role_access_prior': 'measure_only'}

    for scene in result['scenes']:
        assert scene['preparation']['seconds'] >= 0 and scene['preparation_work']['prepared_e_directory_rows'] > 0
        assert scene['support_audit']['selected_members_identical']
        assert scene['support_audit']['located_rows']
        for row in scene['support_audit']['located_rows']:
            for role in row['roles'].values():
                assert len(role['original_member_ids']) == role['selected_count']
                assert role['mass_consistent'] and role['mass_absolute_difference'] < 1e-6
                assert role['maximum_measure_formula_absolute_error'] < 1e-6
        for policy in scene['policies'].values():
            assert policy['paired_full_restricted']['work']['shared']['query_unique_union_raw_score_pairs_E'] > 0
            assert policy['standalone_restricted']['work']['query_unique_union_raw_score_pairs_E'] > 0
            assert policy['standalone_restricted']['matches_paired_fields']
            assert policy['standalone_restricted']['seconds'] >= 0
    assert result['cuda_allocator_peak_allocated_bytes'] is None
    assert result['device'] == 'cpu'
    for pair in result['pairs']:
        assert pair['draw_count'] == 8 and len(pair['draws']) == 8 and pair['dependencies']
        for policy in pair['policies'].values():
            estimates = policy['paired_full']['component_estimates']
            assert any(len(fields) == width for fields in estimates.values())
            assert all('absolute_mse_physical' in field for fields in estimates.values() for field in fields.values())
    json.dumps(result, allow_nan=False)

    # Confirm the saved row audit matches actual decoder traces on the same prepared state.
    _, prepared, query, _ = calls[0]
    first_query = slice_query_v1(query, 0, 1)
    full_learned, _, _ = original_pair(prepared, first_query, return_trace=True)
    full_control, _, _ = original_pair(prepared, first_query,
        intervention={'within_role_access_prior': 'measure_only'}, return_trace=True)
    learned_trace, control_trace = full_learned.trace, full_control.trace
    torch.testing.assert_close(learned_trace['anchor_read_original_id'], control_trace['anchor_read_original_id'], rtol=0, atol=0)
    scene = result['scenes'][0]
    anchor = int(first_query.anchor_indices[0, 0, 0])
    audit = next(row for row in scene['support_audit']['located_rows']
                 if row['anchor_slot'] == anchor and row['read_mode'] == 'full')
    for role, module in (('M', True), ('E', False)):
        mask = learned_trace['anchor_read_is_module'] == module
        assert learned_trace['anchor_read_original_id'][mask].tolist() == audit['roles'][role]['original_member_ids']
        assert float(learned_trace['anchor_read_access_prior'][mask].sum()) == pytest.approx(audit['roles'][role]['learned_prior_mass'])
        assert float(control_trace['anchor_read_access_prior'][mask].sum()) == pytest.approx(audit['roles'][role]['measure_only_prior_mass'])

    # Global union queries give the same learned-prior metrics as the existing pair evaluator.
    baseline = evaluate_observation_panel(model, provider, pairs)
    for view in ('full', 'restricted'):
        for name in ('extra_absolute', 'point_response', 'spatial_response'):
            assert result['summary']['learned']['paired_' + view][name] == pytest.approx(
                baseline['summary'][view][name], rel=2e-5, abs=1e-7)


def test_access_evaluation_rejects_non_primary_or_implicit_panels():
    model, provider = build_observation_response_task(_recipe(), device='cpu')
    for pairs in ([], ['ab'], [('fixture_0', 'fixture_0')], [('fixture_0', 'fixture_1')] * 2):
        with pytest.raises(ValueError, match='explicitly selected pairs'):
            evaluate_access_prior_panel(model, provider, pairs)
    model.core.within_role_access_prior = 'measure_only'
    with pytest.raises(ValueError, match='learned-prior checkpoint'):
        evaluate_access_prior_panel(model, provider, [('fixture_0', 'fixture_1')])


def test_cuda_measurement_uses_the_requested_device_and_reports_allocator_peak(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append(('sync', device)))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: calls.append(('reset', device)))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 10)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 20)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 30)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 40)
    device = torch.device('cuda:1')
    result, measurement = _timed(device, lambda: calls.append(('execute', device)) or 'result')
    assert result == 'result' and measurement['seconds'] >= 0
    assert calls == [('sync', device), ('reset', device), ('execute', device), ('sync', device)]
    assert measurement['cuda_allocator'] == {'baseline': {'allocated_bytes': 10, 'reserved_bytes': 20},
                                              'peak_allocated_bytes': 30, 'peak_reserved_bytes': 40}


def test_cli_restores_checkpoint_and_saves_declared_panel(tmp_path, monkeypatch):
    from tools import observation_access_prior_evaluate as cli
    from tools.relation_anchored_train import build

    torch.set_num_threads(1)
    recipe = _recipe()
    model, provider = build(recipe, device='cpu')
    with torch.no_grad():
        model.core.member_router.joint_head.bias.add_(.314)
    expected = evaluate_access_prior_panel(model, provider, [('fixture_0', 'fixture_1')])
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'epoch': 1000, 'experiment_identity': {'recipe': deepcopy(recipe),
        'provider_identity': provider.identity_payload()}, 'model_state_dict': model.state_dict(),
        'provider_training_state': provider.training_state_dict()}, checkpoint)
    recipe_path, panel_path, output = tmp_path / 'recipe.json', tmp_path / 'panel.json', tmp_path / 'result.json'
    recipe_path.write_text(json.dumps(recipe))
    panel_path.write_text(json.dumps({'name': 'small_training_panel', 'pairs': [['fixture_0', 'fixture_1']]}))
    monkeypatch.setattr(cli, '_ensure_workspace_storage_bindings', lambda: None)
    monkeypatch.setattr(cli, 'verify_storage', lambda *args, **kwargs: {'test_temporary_storage': True})
    assert cli.main(['--recipe', str(recipe_path), '--run', str(tmp_path), '--checkpoint', str(checkpoint), '--pair-panel', str(panel_path),
                     '--output', str(output), '--device', 'cpu']) == 0
    result = json.loads(output.read_text())
    assert result['panel']['name'] == 'small_training_panel' and result['epoch'] == 1000
    assert result['checkpoint'] == str(checkpoint.resolve())
    assert result['scene_preparations'] == 2
    for policy in ('learned', 'measure_only'):
        for view in ('paired_full', 'paired_restricted', 'standalone_restricted'):
            for name, value in expected['summary'][policy][view].items():
                assert result['summary'][policy][view][name] == pytest.approx(value, rel=1e-6, abs=1e-7)
    with pytest.raises(FileExistsError):
        cli.main(['--recipe', str(recipe_path), '--run', str(tmp_path), '--checkpoint', str(checkpoint), '--pair-panel', str(panel_path),
                  '--output', str(output), '--device', 'cpu'])
