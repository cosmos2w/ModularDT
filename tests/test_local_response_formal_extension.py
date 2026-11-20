"""No-fitting tests for the explicitly bounded TRAIN600 continuation."""
import copy
import json

import pytest

from honf_runtime.unified_training import _engine_config_payload, _resume_identity_amendment
from tests.test_relation_anchored_horizon import native_pair_formal_identities
from tools.relation_anchored_train import engine_for


@pytest.mark.parametrize('variant', ['state_anchors', 'uniform_response', 'multisecant_response', 'soft_inverse_response',
                                     'trace_matched_uniform_response'])
def test_local_formal_extension_keeps_scientific_identity_and_saved_recording(variant):
    old, _ = native_pair_formal_identities()
    old['recipe']['training_profile'] = 'local_multisecant_response'
    old['recipe']['thermal_local_response'] = {'variant': variant, 'seed': 0, 'input_definitions': 'shared.json'}
    if variant == 'trace_matched_uniform_response':
        old['recipe']['thermal_local_response']['target_definitions'] = 'train600_conditioned_targets.json'
    old['provider_identity']['local_response'] = {'variant': variant, 'scales': [1., 2.], 'weights': [.5, .5]}
    old['provider_identity']['recording'] = {'columns': ['native', 'response'], 'meaning': {'type': 'float', 'unit': '1'}}
    new = json.loads(json.dumps(old, sort_keys=True))
    new['engine_config'] = _engine_config_payload(engine_for(old['recipe'], device='cpu', execution_horizon=500).config)
    receipt = _resume_identity_amendment(old, new, allow_horizon_extension=True)
    assert receipt['kind'] == 'explicit_local_multisecant_response_formal_horizon_extension'
    assert receipt['optimizer_schedule_unchanged']
    assert engine_for(old['recipe'], device='cpu', execution_horizon=500).config.checkpoint_epochs == (100, 200, 300, 400, 500)
    for path, value in [(('provider_identity', 'local_response', 'weights'), [.4, .6]),
                        (('recipe', 'thermal_local_response', 'seed'), 1),
                        (('recipe', 'thermal_local_response', 'target_definitions'), 'changed_targets.json'),
                        (('optimizer_schedule_contract', 0, 'schedule', 'hold_through_epoch'), 999),
                        (('provider_identity', 'recording', 'columns'), ['response', 'native']),
                        (('engine_config', 'seed'), 7)]:
        changed = copy.deepcopy(new)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            _resume_identity_amendment(old, changed, allow_horizon_extension=True)
    with pytest.raises(ValueError):
        engine_for(old['recipe'], device='cpu', execution_horizon=100)
    assert engine_for(old['recipe'], device='cpu', execution_horizon=1000).config.total_epochs == 1000
