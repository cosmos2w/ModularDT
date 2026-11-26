import copy
from types import SimpleNamespace

import pytest
import torch

from channelthermal.training import native_checkpoint as module


def _saved():
    model = torch.nn.Linear(2, 1)
    recipe = {
        'training_profile': 'state_response', 'dataset': {'name': 'thermal'},
        'member_routing': {'arm': 'R14-Detail-LocalECover',
                           'model': {'schema': module.DETAIL_CORE_SCHEMA}},
        'evaluation': {'validation_case_ids': ['0281']},
        'state_response': {'enabled': True},
        'thermal_joint_response': {
            'native_ports': {'angle_count': 64},
            'h5': {'path': '/unavailable/finite_responses_v2.h5'},
            'paired': {'roi_profile': '/unavailable/profile.json'},
        },
    }
    return {'experiment_identity': {'recipe': recipe, 'provider_identity': {
                'adapter_state': {'normalization_stats': {'mean': [1.25]}},
                'source_binding': {'normalization_source': 'saved TRAIN fit'}}},
            'model_state_dict': model.state_dict()}


@pytest.mark.parametrize('profile', [
    'state_response', 'native_pair', 'field_response',
    'local_multisecant_response', 'observation_response', 'historical_r16_10_response',
])
def test_native_only_inference_uses_saved_normalization_and_separate_panel(monkeypatch, profile):
    saved = _saved()
    if profile == 'historical_r16_10_response':
        saved['experiment_identity']['recipe']['training_profile'] = None
        saved['experiment_identity']['recipe']['r16_10_family'] = {
            'schema': 'thermal_r16_10_family_v1', 'arm': 'R'}
    else:
        saved['experiment_identity']['recipe']['training_profile'] = profile
    original = copy.deepcopy(saved['experiment_identity'])
    calls = []
    provider = SimpleNamespace(validation_case_ids=('0644', '0645'))
    provider.configure_native = lambda: calls.append('configure_native')

    def build(recipe, **kwargs):
        ports = ({'angle_count': 64} if profile == 'state_response'
                 else {'angle_count': 64, 'outside_temperature_weight': .05})
        assert recipe['thermal_joint_response'] == {'native_ports': ports}
        assert recipe['state_response'] == {'enabled': False}
        assert recipe['training_profile'] == 'state_response'
        assert recipe['evaluation']['validation_case_ids'] == ['0644', '0645']
        assert kwargs['saved_adapter_state'] == original['provider_identity']['adapter_state']
        assert kwargs['saved_normalization_source'] == 'saved TRAIN fit'
        return torch.nn.Linear(2, 1), provider

    monkeypatch.setattr(module, '_build_thermal_member_routing_task', build)
    model, actual_provider = module.build_native_checkpoint_inference(
        saved, validation_case_ids=['0644', '0645'])
    assert actual_provider is provider and calls == ['configure_native']
    assert not model.training
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in saved['model_state_dict'].items())
    assert saved['experiment_identity'] == original


@pytest.mark.parametrize('panel', [[], ['0644', '0644']])
def test_native_inference_rejects_invalid_panel_before_build(panel, monkeypatch):
    monkeypatch.setattr(module, '_build_thermal_member_routing_task', lambda *a, **kw: pytest.fail('unexpected build'))
    with pytest.raises(ValueError, match='unique original'):
        module.build_native_checkpoint_inference(_saved(), validation_case_ids=panel)


def test_native_inference_requires_strict_matching_architecture(monkeypatch):
    saved = _saved()
    saved['model_state_dict']['weight'] = torch.ones(1, 3)
    provider = SimpleNamespace(validation_case_ids=('0644',), configure_native=lambda: None)
    monkeypatch.setattr(module, '_build_thermal_member_routing_task',
                        lambda *a, **kw: (torch.nn.Linear(2, 1), provider))
    with pytest.raises(RuntimeError, match='size mismatch'):
        module.build_native_checkpoint_inference(saved, validation_case_ids=['0644'])


def test_original_heldout_uses_existing_population_in_source_order(monkeypatch):
    from channelthermal.data import development_split

    saved = _saved()
    recipe = saved['experiment_identity']['recipe']
    recipe['training'] = {'membership': 'full_train600'}
    recipe['dataset']['data_path'] = '/source/packed_dataset.h5'
    expected = tuple(f'{number:04d}' for number in range(100, 189))
    catalog = ([{'case_id': '0001', 'split': 'train'}, {'case_id': '0273', 'split': 'test'}]
               + [{'case_id': value, 'split': 'test'} for value in expected])
    monkeypatch.setattr(development_split, 'read_case_catalog', lambda path: ({}, catalog))
    original = copy.deepcopy(saved['experiment_identity'])
    assert module.original_heldout_case_ids(saved) == expected
    assert saved['experiment_identity'] == original


def test_historical_variant_with_changed_predictor_is_not_silently_loaded(monkeypatch):
    saved = _saved()
    saved['experiment_identity']['recipe']['training_profile'] = None
    saved['experiment_identity']['recipe']['r16_10_family'] = {
        'schema': 'thermal_r16_10_family_v1', 'arm': 'V'}
    monkeypatch.setattr(module, '_build_thermal_member_routing_task', lambda *a, **kw: pytest.fail('unexpected build'))
    with pytest.raises(ValueError, match='native joint-field'):
        module.build_native_checkpoint_inference(saved, validation_case_ids=['0644'])


@pytest.mark.parametrize('deterministic', [True, False])
def test_observation_inference_applies_saved_deterministic_policy_before_build(monkeypatch, deterministic):
    saved = _saved()
    saved['experiment_identity']['recipe'].update(
        training_profile='observation_response', execution_config={'deterministic_algorithms': deterministic})
    calls = []
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(module.torch, 'use_deterministic_algorithms', lambda value: calls.append(value))
    provider = SimpleNamespace(validation_case_ids=('0644',), configure_native=lambda: None)

    def build(*args, **kwargs):
        assert calls == [deterministic]
        assert module.os.environ.get('CUBLAS_WORKSPACE_CONFIG') == (':4096:8' if deterministic else None)
        return torch.nn.Linear(2, 1), provider

    monkeypatch.setattr(module, '_build_thermal_member_routing_task', build)
    module.build_native_checkpoint_inference(saved, validation_case_ids=['0644'])
