"""Evaluator preserves role, physical labels, masks and read views."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r16_evaluator', ROOT / 'tools/thermal_state_response_evaluate.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


@pytest.mark.parametrize('capability,modes', [('anchored', {'restricted', 'full'}), ('all_native', {'all_native'})])
def test_eval_uses_common_masks_and_saved_physical_labels(monkeypatch, capability, modes):
    case = {'module_count': 2, 'amplitude': .1, 'joint_kind': 'co', 'selection': {'member_ids': [0, 1]}}
    corpus = SimpleNamespace(cases={'0001': case})
    model = SimpleNamespace(core=SimpleNamespace(reader_capability=capability))
    actions = evaluator.ACTIONS
    calls = []
    def pair(provider, corpus, family, action, *, role):
        calls.append((family, action, role))
        return {'baseline': 'b', 'edited': 'e', 'raw_baseline': {'tag': 'b'}, 'raw_edited': {'tag': 'e'}}
    def observation(provider, model, batch, raw, tau, mode):
        edited = batch == 'e'
        return ({'field': {'prediction': np.array([3., 4.]) if edited else np.array([1., 1.]),
                           'reference': np.array([2., 4.]) if edited else np.array([1., 1.]),
                           'mask': np.array([True, False]) if edited else np.array([True, True])}},
                {'J': 2. if edited else 0., 'P': 2. if edited else 0.}, {})
    monkeypatch.setattr(evaluator, 'response_pair', pair)
    monkeypatch.setattr(evaluator, 'native_observation', observation)
    monkeypatch.setattr(evaluator, 'source_strata', lambda pair: {})
    labels = {('0001', action): {'J': 1., 'P': .1} for action in actions}
    result = evaluator.evaluate(None, model, corpus, 'response_holdout', ('0001',), labels, .4, {'J': .5, 'P': .5})
    assert set(result) == modes
    assert calls == [('0001', action, 'response_holdout') for action in actions]
    for value in result.values():
        assert value['actions'][0]['fields']['field']['response']['count'] == 1
        assert value['actions'][0]['fields']['field']['response']['rmse'] == 1.
        assert value['functionals']['J']['resolved_sign_count'] == 4
        assert value['functionals']['P']['near_null_family_count'] == 1
        assert value['actions'][0]['delta']['J'] == {'model': 2., 'physical': 1.}


def test_field_summary_weights_families_equally():
    def row(family, error):
        return {'case_id': family, 'fields': {'t': {k: {'rmse': error} for k in ('baseline', 'edited', 'response')}}}
    result = evaluator.summarize_fields([row('a', 1), row('a', 3), row('b', 10)])
    assert result['t']['response']['equal_family_mean_rmse'] == 6.
    assert result['t']['response']['family_worst_rmse'] == 10.
    assert result['t']['response']['family_worst_id'] == 'b'


def test_restore_requires_both_exact_identities_before_loading(monkeypatch, tmp_path):
    class Model:
        loaded = False
        def load_state_dict(self, state, strict):
            self.loaded = True
            assert strict
        def to(self, device): return self
        def eval(self): return self
    model = Model()
    provider = SimpleNamespace(identity_payload=lambda: {'membership': ['a']})
    monkeypatch.setattr(evaluator, 'build', lambda recipe, device: (model, provider))
    monkeypatch.setattr(evaluator, 'resolve_checkpoint', lambda run, checkpoint: tmp_path / 'checkpoint.pt')
    saved = {'experiment_identity': {'recipe': {'key': 1}, 'provider_identity': {'membership': ['b']}},
             'model_state_dict': {}, 'epoch': 100}
    monkeypatch.setattr(evaluator, 'load_trusted_checkpoint', lambda *args, **kwargs: saved)
    with pytest.raises(ValueError, match='differs'):
        evaluator.restore_model({'key': 1}, tmp_path, 'latest', 'cpu')
    assert not model.loaded
    saved['experiment_identity']['provider_identity'] = {'membership': ['a']}
    evaluator.restore_model({'key': 1}, tmp_path, 'latest', 'cpu')
    assert model.loaded
