import copy
from types import SimpleNamespace

import pytest

from tools.relation_anchored_train import provider_identity_matches_for_inference
from tools import thermal_state_response_evaluate as evaluator


def identities():
    recipe = {'training_profile': 'native_pair', 'thermal_native_pair': {'variant': 'state_anchors'},
              'training': {'membership': 'fixed25_train150_dev22', 'local_horizon': 12,
                           'optimizer_schedule_horizon': 2000}}
    old = {'training': {'optimizer_schedule': {'peak_lr': .0003, 'warmup_start_lr': .0003,
           'warmup_epochs': 0, 'hold_through_epoch': 1999, 'total_epochs': 2000, 'final_lr': .0003}},
           'normalization': {'mean': 0.5}, 'membership': ['001'], 'adapter_state': {'scale': 1}}
    current = copy.deepcopy(old)
    current['training']['optimizer_schedule'].update(hold_through_epoch=1000, final_lr=.00003)
    return recipe, old, current


def test_only_known_schedule_difference_is_permitted_for_read_only_inference():
    recipe, old, current = identities()
    original = copy.deepcopy(old)
    assert provider_identity_matches_for_inference(recipe, old, current)
    assert old == original
    for field, value in [('normalization', {'mean': 0.6}), ('membership', ['002']), ('adapter_state', {'scale': 2})]:
        changed = copy.deepcopy(current); changed[field] = value
        assert not provider_identity_matches_for_inference(recipe, old, changed)
    changed = copy.deepcopy(current); changed['training']['optimizer_schedule']['final_lr'] = .00004
    assert not provider_identity_matches_for_inference(recipe, old, changed)
    wrong_recipe = copy.deepcopy(recipe); wrong_recipe['training']['membership'] = 'full_train600'
    assert not provider_identity_matches_for_inference(wrong_recipe, old, current)
    wrong_recipe['training_profile'] = 'state_response'
    assert not provider_identity_matches_for_inference(wrong_recipe, old, current)


def test_response_loader_restores_weights_and_rejects_changed_recipe_or_normalization(monkeypatch):
    recipe, old, current = identities()
    state = {'weight': 'saved'}
    saved = {'epoch': 12, 'experiment_identity': {'recipe': recipe, 'provider_identity': old},
             'model_state_dict': state, 'provider_training_state': {'counter': 3}}
    class Model:
        def load_state_dict(self, value, strict):
            assert value == state and strict
        def to(self, device):
            assert device == 'cpu'
        def eval(self):
            self.evaluated = True
    class Provider:
        def identity_payload(self): return current
        def load_training_state_dict(self, value): self.restored = value
    model, provider = Model(), Provider()
    monkeypatch.setattr(evaluator, 'resolve_checkpoint', lambda *args: 'checkpoint')
    monkeypatch.setattr(evaluator, 'load_trusted_checkpoint', lambda *args, **kwargs: saved)
    monkeypatch.setattr(evaluator, 'build', lambda *args, **kwargs: (model, provider))
    loaded, restored, _, _ = evaluator.restore_model(recipe, 'run', 'latest', 'cpu')
    assert loaded.evaluated and restored.restored == {'counter': 3}
    changed_recipe = copy.deepcopy(recipe); changed_recipe['training']['local_horizon'] = 13
    with pytest.raises(ValueError, match='recipe differs'):
        evaluator.restore_model(changed_recipe, 'run', 'latest', 'cpu')
    current['normalization']['mean'] = 0.6
    with pytest.raises(ValueError, match='native provider differs'):
        evaluator.restore_model(recipe, 'run', 'latest', 'cpu')
