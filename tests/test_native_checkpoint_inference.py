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


def test_native_only_inference_uses_saved_normalization_and_separate_panel(monkeypatch):
    saved = _saved()
    original = copy.deepcopy(saved['experiment_identity'])
    calls = []
    provider = SimpleNamespace(validation_case_ids=('0644', '0645'))
    provider.configure_native = lambda: calls.append('configure_native')

    def build(recipe, **kwargs):
        assert recipe['thermal_joint_response'] == {'native_ports': {'angle_count': 64}}
        assert recipe['state_response'] == {'enabled': False}
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
