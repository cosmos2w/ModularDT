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


def test_field_effect_ratio_uses_equal_family_averages_and_zero_coverage():
    def row(family, error, effect):
        return {'case_id': family, 'fields': {'t': {k: {'rmse': error, 'reference_rms': effect}
                                                    for k in ('baseline', 'edited', 'response')}}}
    result = evaluator.summarize_fields([row('a', 1, 0), row('a', 3, 2), row('b', 10, 4)])['t']['response']
    assert result['equal_family_mean_rmse'] == 6
    assert result['equal_family_mean_reference_rms'] == 2.5
    assert result['equal_family_rmse_over_reference_rms'] == pytest.approx(2.4)
    assert result['exact_zero_reference_actions'] == 1
    assert result['reference_action_count'] == 3
    null = evaluator.summarize_fields([row('a', 1, 0)])['t']['response']
    assert null['equal_family_rmse_over_reference_rms'] is None


def test_saved_action_stratifier_preserves_views_and_primary_exposure():
    spec = importlib.util.spec_from_file_location('r16_stratifier', ROOT / 'tools/thermal_response_stratify.py')
    stratifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stratifier)
    def row(family, action, delta, effect):
        metric = {'rmse': 1., 'reference_rms': effect}
        return {'case_id': family, 'action': action,
                'strata': {'represented_source_delta': delta, 'amplitude': .0675, 'module_count': 2},
                'fields': {'t': {name: metric for name in ('baseline', 'edited', 'response')}},
                'delta': {name: {'model': 1., 'physical': effect} for name in ('J', 'P')}}
    rows = [row('seen', 'single_plus', 0., 0.), row('unseen', 'joint_minus', .1, 2.)]
    result = stratifier.stratify({'role': 'response_holdout', 'checkpoint': 'saved', 'epoch': 1,
                                 'functional_floors': {'J': .01, 'P': .01},
                                 'views': {'restricted': {'actions': rows}, 'full': {'actions': rows}}}, ['seen'])
    assert set(result['views']) == {'restricted', 'full'}
    groups = result['views']['restricted']['strata']
    assert groups['baseline_exposure']['primary_seen']['family_ids'] == ['seen']
    assert groups['baseline_exposure']['primary_unseen']['family_ids'] == ['unseen']
    assert groups['represented_source_change']['unchanged']['action_count'] == 1
    assert groups['action_type']['joint']['fields']['t']['response']['equal_family_rmse_over_reference_rms'] == .5
    assert groups['action_type']['single']['fields']['t']['response']['equal_family_rmse_over_reference_rms'] is None


@pytest.mark.parametrize('size,expected', [(150, 'primary_unseen'), (600, 'primary_seen')])
def test_exposure_uses_supplied_primary_membership(size, expected):
    spec = importlib.util.spec_from_file_location('r16_stratifier_scope', ROOT / 'tools/thermal_response_stratify.py')
    stratifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stratifier)
    primary = [f'{i:04d}' for i in range(size)]
    metric = {'rmse': 1., 'reference_rms': 1.}
    row = {'case_id': '0500', 'action': 'single_plus',
           'strata': {'represented_source_delta': 0., 'amplitude': .0675, 'module_count': 2},
           'fields': {'t': {name: metric for name in ('baseline', 'edited', 'response')}},
           'delta': {name: {'model': 1., 'physical': 1.} for name in ('J', 'P')}}
    evaluation = {'role': 'response_holdout', 'checkpoint': 'saved', 'epoch': 1,
                  'functional_floors': {'J': .01, 'P': .01}, 'views': {'restricted': {'actions': [row]}}}
    result = stratifier.stratify(evaluation, primary)
    assert result['primary_membership_count'] == size
    assert result['views']['restricted']['strata']['baseline_exposure'][expected]['family_ids'] == ['0500']


def test_restore_requires_both_exact_identities_before_loading(monkeypatch, tmp_path):
    class Model:
        loaded = False
        def load_state_dict(self, state, strict):
            self.loaded = True
            assert strict
        def to(self, device): return self
        def eval(self): return self
    model = Model()
    scales = {'fixed': 'saved objective constants'}
    provider = SimpleNamespace(identity_payload=lambda: {'membership': ['a'], 'response_scales': scales})
    def build(recipe, *, device, response_scales):
        assert response_scales is scales
        return model, provider
    monkeypatch.setattr(evaluator, 'build', build)
    monkeypatch.setattr(evaluator, 'resolve_checkpoint', lambda run, checkpoint: tmp_path / 'checkpoint.pt')
    saved = {'experiment_identity': {'recipe': {'key': 1}, 'provider_identity': {'membership': ['b'], 'response_scales': scales}},
             'model_state_dict': {}, 'epoch': 100}
    monkeypatch.setattr(evaluator, 'load_trusted_checkpoint', lambda *args, **kwargs: saved)
    with pytest.raises(ValueError, match='differs'):
        evaluator.restore_model({'key': 1}, tmp_path, 'latest', 'cpu')
    assert not model.loaded
    saved['experiment_identity']['provider_identity'] = {'membership': ['a'], 'response_scales': scales}
    evaluator.restore_model({'key': 1}, tmp_path, 'latest', 'cpu')
    assert model.loaded
