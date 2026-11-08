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
