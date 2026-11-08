import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import torch

from channelthermal.training import r16_10_health as health


class FakeSchedule:
    families = {'original': tuple(f'{i:04}' for i in range(15)),
                'new': tuple(f'{i:04}' for i in range(15, 40))}
    def __init__(self):
        self.cursor = 5
    def state_dict(self):
        return {'cursor': self.cursor}
    def load_state_dict(self, state):
        self.cursor = state['cursor']


class FakeObjective:
    def __init__(self):
        self.schedule = FakeSchedule()
    def family_components(self, model, source, family):
        self.schedule.cursor += 1
        p = model.weight.squeeze()
        return {'endpoint': (p - 1).square(), 'response': (p + 2).square(),
                'decision': (p - 3).square()}, {}


class FakeProvider:
    training_case_ids = tuple(f'{i:04}' for i in range(150))
    def __init__(self):
        self.family_objective = FakeObjective()
        self.calls = []
    def after_optimizer_step(self, model, optimizer, norm, *, epoch, update_index):
        self.calls.append(('step', epoch, update_index))
    def completed_training_epoch(self, model, engine, epoch, output, row):
        self.calls.append(('epoch', epoch))
    def native_evaluation_batches(self, *, split, case_keys, microbatch_size):
        assert split == 'train' and len(case_keys) == 1 and microbatch_size == 1
        yield SimpleNamespace(scene_inputs=None, receivers=None)
    def predict_native(self, model, scene, receivers, arm, phase, epoch):
        assert model.training
        return model.weight.squeeze(), {}
    def endpoint_state_loss(self, prediction, batch):
        return (prediction - 4).square()


def test_callbacks_restore_state_and_record_actual_update(tmp_path, monkeypatch):
    monkeypatch.setattr('channelthermal.training.r16_5_decision_task._parameter_groups', lambda model: {})
    coefficients = tmp_path / 'coefficients.json'
    coefficients.write_text(json.dumps({'coefficients': {'endpoint': .2, 'response': .1, 'decision': .05}}))
    recipe = {'r16_10_family': {'population': 150, 'arm': 'R', 'coefficients_path': str(coefficients)},
              'optimizer': {'gradient_clip': 1.0}}
    provider = health.attach_update_diagnostics(FakeProvider(), recipe)
    model = torch.nn.Linear(1, 1, bias=False)
    model.architecture = 'test'
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    model.weight.grad = torch.full_like(model.weight, 7.)
    model.eval()
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_rng = torch.get_rng_state().clone()
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    provider.after_optimizer_step(model, optimizer, torch.tensor(2.), epoch=50, update_index=0)
    provider.after_optimizer_step(model, optimizer, torch.tensor(.5), epoch=50, update_index=1)
    row = {}
    provider.completed_training_epoch(model, SimpleNamespace(config=SimpleNamespace(gradient_clip=1.)), 50, None, row)
    result = row['r16_10_actual_update']
    assert result['observed_epoch_clipping_fraction'] == .5
    assert result['combined_weights'] == {'native': 1., 'endpoint': .2, 'response': .1}
    assert set(result['updates']) == {'native_only', 'combined'}
    assert len(result['native_calibration_panel']) == 15 and len(result['family_calibration_panel']) == 4
    assert provider.family_objective.schedule.cursor == 5
    assert not model.training and torch.equal(model.weight, before_model['weight'])
    assert torch.equal(model.weight.grad, torch.full_like(model.weight, 7.))
    assert optimizer.state_dict()['param_groups'] == before_optimizer['param_groups']
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(optimizer.state_dict()['state'][0][key], before_optimizer['state'][0][key])
    assert torch.equal(torch.get_rng_state(), before_rng) and random.getstate() == py_rng
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), np_rng))
    assert provider.calls == [('step', 50, 0), ('step', 50, 1), ('epoch', 50)]


def test_only_declared_epochs_trigger(monkeypatch):
    provider = health.attach_update_diagnostics(FakeProvider(), {'r16_10_family': {'population': 600, 'arm': 'B0'}, 'optimizer': {'gradient_clip': 1.}})
    monkeypatch.setattr(health, '_diagnose', lambda *args: {})
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters())
    engine = SimpleNamespace(config=SimpleNamespace(gradient_clip=1.))
    for epoch in (49, 50, 100, 1000):
        provider.after_optimizer_step(model, opt, torch.tensor(.5), epoch=epoch, update_index=0)
        row = {}
        provider.completed_training_epoch(model, engine, epoch, None, row)
        assert ('r16_10_actual_update' in row) == (epoch in (50, 100))
