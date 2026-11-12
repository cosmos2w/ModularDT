import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.training import r16_8_corpus as module


def manifest(tmp_path):
    config = {'domain': {'nx': 2}, 'thermal': {'k': 1}, 'local_module': {},
              'layout': {'num_modules': 1, 'seed': 0, 'centers': [[1., 2.]], 'heat_powers': [3.]},
              'flow': {'re': 10.}}
    case = {'case_id': '0077', 'role': 'fit', 'source_case_dir': 'raw-base', 'source_config': config}
    actions = [{'case_id': '0077', 'role': 'fit', 'action': f'a{i}', 'status': 'converged',
                'raw_case_dir': f'raw-{i}', 'centers': [[1. + i / 10, 2.]],
                'physical_delta': {'J': float(i), 'P': -float(i)}} for i in range(4)]
    data = {'schema': 'thermal_r16_8_corpus_v1', 'cases': [case], 'actions': actions}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(data))
    return path, data


def test_membership_order_roles_and_finite_labels(tmp_path):
    path, data = manifest(tmp_path)
    corpus = module.ThermalR168Corpus(path)
    assert corpus.family_ids('fit') == ('0077',)
    assert corpus.actions_for_family('0077') == ('a0', 'a1', 'a2', 'a3')
    assert corpus.physical_delta('0077', 'a2') == {'J': 2., 'P': -2.}
    with pytest.raises(ValueError, match='role'):
        corpus.physical_delta('0077', 'a2', role='native_seen')
    data['actions'][1]['status'] = 'failed'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='converged'):
        module.ThermalR168Corpus(path)
    data['actions'][1]['status'] = 'converged'
    data['actions'][1]['physical_delta']['P'] = float('nan')
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='finite'):
        module.ThermalR168Corpus(path)


def test_both_endpoints_use_raw_and_preserve_member_identity(tmp_path, monkeypatch):
    path, data = manifest(tmp_path)
    corpus = module.ThermalR168Corpus(path)
    base = data['cases'][0]['source_config']
    edit = json.loads(json.dumps(base)); edit['layout']['centers'] = data['actions'][1]['centers']
    raw = lambda config: {'config': config, 'material_mask': np.ones((1, 1), dtype=bool)}
    monkeypatch.setattr(module, 'load_raw_endpoint', lambda name: raw(base if name == 'raw-base' else edit))
    import channelthermal.interaction_evidence.reference_adapter as ref
    monkeypatch.setattr(ref, 'operating_context_from_config', lambda config: SimpleNamespace(values={'re': config['flow']['re']}))
    import channelthermal.data.datasets as datasets
    monkeypatch.setattr(datasets, '_local_disk_query_points', lambda mask: np.array([[0., 0.]]))
    calls = []
    def endpoint(provider, original, raw_case, label):
        calls.append((label, raw_case['config']['layout']['centers']))
        return SimpleNamespace(receivers=SimpleNamespace(heat=torch.tensor([3.])))
    monkeypatch.setattr(module, '_native_endpoint_batch', endpoint)
    provider = SimpleNamespace(_load_native_cases=lambda ids: [{'structure': {'module_centers': np.array([[1., 2.]], np.float32),
                    'heat_powers': np.array([3.], np.float32)}, 'module_internal_query_points': np.array([[0., 0.]])}])
    pair = corpus.pair(provider, '0077', 'a1')
    assert calls == [('baseline', [[1., 2.]]), ('a1', [[1.1, 2.]])]
    assert pair['raw_baseline']['config'] == base and pair['raw_edited']['config'] == edit
    edit['layout']['heat_powers'] = [4.]
    with pytest.raises(ValueError, match='heat_powers'):
        corpus.pair(provider, '0077', 'a1')


def test_v2_fit_preview_preserves_v1_and_excludes_evaluation_roles(tmp_path, monkeypatch):
    path, data = manifest(tmp_path)
    case = data['cases'][0]
    actual_base = json.loads(json.dumps(case['source_config']))
    actual_edit = json.loads(json.dumps(actual_base))
    actual_edit['layout']['centers'] = data['actions'][1]['centers']
    actual_edit['thermal']['max_solve_time'] = 800
    seen = []
    import channelthermal.interaction_evidence.numerical_policy as policy
    def validate(expected, actual, **kwargs):
        seen.append((expected, actual, kwargs))
        assert expected['layout']['centers'] == actual['layout']['centers']
        assert kwargs == {'policy': 'thermal_time_cap_v1', 'require_converged': True}
        return {'max_time': actual['thermal'].get('max_solve_time', 200)}
    monkeypatch.setattr(policy, 'validate_numerical_policy', validate)
    case.update(actual_solver_config=actual_base, numerical_policy={'max_time': 200})
    entry = data['actions'][1]
    entry.update(actual_solver_config=actual_edit, numerical_policy={'max_time': 800})
    data.update(schema='thermal_r16_9_corpus_v2', scope='fit', preview=True, actions=[entry])
    path.write_text(json.dumps(data)); corpus = module.ThermalR168Corpus(path)
    assert corpus.family_ids('fit') == ('0077',) and not corpus.family_ids('native_seen')
    raw = lambda cfg: {'config': cfg, 'material_mask': np.ones((1, 1), dtype=bool)}
    monkeypatch.setattr(module, 'load_raw_endpoint', lambda name: raw(actual_base if name == 'raw-base' else actual_edit))
    import channelthermal.data.datasets as datasets
    monkeypatch.setattr(datasets, '_local_disk_query_points', lambda mask: np.array([[0., 0.]]))
    calls = []
    def endpoint(provider, original, raw_case, label):
        calls.append((label, raw_case['config']))
        return SimpleNamespace(receivers=SimpleNamespace(heat=torch.tensor([3.])))
    monkeypatch.setattr(module, '_native_endpoint_batch', endpoint)
    provider = SimpleNamespace(_load_native_cases=lambda ids: [{'structure': {'module_centers': np.array([[1., 2.]], np.float32),
        'heat_powers': np.array([3.], np.float32)}, 'module_internal_query_points': np.array([[0., 0.]])}])
    corpus.pair(provider, '0077', 'a1')
    assert len(seen) == 2 and seen[1][0]['thermal'] == case['source_config']['thermal']
    assert calls[0][0] == 'baseline' and calls[1][0] == 'a1'
    with pytest.raises(ValueError, match='role'):
        corpus.pair(provider, '0077', 'a1', role='native_seen')
    data['cases'][0]['role'] = 'native_seen'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='role'):
        module.ThermalR168Corpus(path)


def test_v2_rejects_mixed_roles_and_unmarked_partial_packs(tmp_path):
    path, data = manifest(tmp_path)
    case = data['cases'][0]
    case.update(actual_solver_config=case['source_config'], numerical_policy={'schema': 'thermal_time_cap_v1'})
    for entry in data['actions']:
        entry.update(actual_solver_config=case['source_config'], numerical_policy={'schema': 'thermal_time_cap_v1'})
    data.update(schema='thermal_r16_9_corpus_v2', scope='fit')
    data['actions'][0]['role'] = 'native_seen'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='roles differ'):
        module.ThermalR168Corpus(path)
    data['actions'][0]['role'] = 'fit'; data['actions'] = data['actions'][:1]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='action count'):
        module.ThermalR168Corpus(path)
