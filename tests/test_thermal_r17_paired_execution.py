import json
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.training.thermal_joint_response_task import _attach_paired_response, calibrate_paired_response
from channelthermal.training.r13_6_member_routing_task import _response_validation_population
from honf_runtime.state_response import ResponseObjectiveConfig


class Provider:
    training_case_ids = tuple(f'{i:04}' for i in range(40))

    def identity_payload(self):
        return {'objective': {}}

    def loss_metadata(self):
        return {}


class Objective:
    def __init__(self):
        self.corpus = SimpleNamespace(path='manifest.json', cases={
            f'{i:04}': {'source': 'original' if i < 15 else 'new'} for i in range(40)})
        self.corpus.family_ids = lambda: tuple(self.corpus.cases)
        self.response_scales = {name: 1. for name in (
            'flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid',
            'temperature/surface', 'temperature/material', 'q_proxy', 'J', 'P')}
        self.auxiliary_fluid_measure = 'grid'
        self.tau = .4

    def actions(self, source, family):
        return ('a', 'b', 'c', 'd')

    def family_components(self, model, source, family, components):
        return {name: model.weight.sum() for name in components}, {}


def test_pending_calibration_source_cursors_and_executor_resume(tmp_path):
    path, decision = tmp_path / 'calibration.json', tmp_path / 'decision.json'
    spec = {'calibration_path': str(path), 'qualified_tile': 512, 'executor_decision_path': str(decision)}
    provider, objective = Provider(), Objective()
    _attach_paired_response(provider, objective, ResponseObjectiveConfig(), spec)
    assert provider.identity_payload()['thermal_joint_response']['counts'] == {'original': 1, 'new': 2}
    model = torch.nn.Linear(1, 1)
    model.core = SimpleNamespace(query_tile_size=512)
    assert provider.begin_training_epoch(model, 51)['active_tile'] == 128
    assert provider.begin_training_epoch(model, 56)['active_tile'] == 512
    assert provider.begin_training_epoch(model, 66)['active_tile'] == 128
    with pytest.raises(FileNotFoundError):
        list(provider.auxiliary_update_terms(model, epoch=11, update_index=0, update_count=3))
    # A real run has 150 native cases; retain 40 eligible families within it.
    provider.training_case_ids = tuple(f'{i:04}' for i in range(150))
    def panel(values):
        selected = set(np.random.default_rng(20261108).choice(len(values), round(.1 * len(values)), replace=False))
        return [value for i, value in enumerate(values) if i in selected]
    families = [(s, f) for s in ('original', 'new') for f in objective.schedule.families[s]]
    receipt = {'native_case_ids': panel(provider.training_case_ids),
               'fit_families': panel(families),
               'epoch': 10, 'measure': 'grid', 'response_scales': objective.response_scales,
               'coefficients': {'endpoint': .2, 'response': .1}}
    path.write_text(json.dumps(receipt))
    for slot in range(3):
        assert len(list(provider.auxiliary_update_terms(model, epoch=11, update_index=slot, update_count=3))) == 1
    saved = provider.training_state_dict()
    restored = Provider()
    restored.training_case_ids = provider.training_case_ids
    other = Objective()
    _attach_paired_response(restored, other, ResponseObjectiveConfig(), spec)
    restored.load_training_state_dict(saved)
    assert other.schedule.cursor == {'original': 1, 'new': 2}
    decision.write_text(json.dumps({'qualified_tile': 512, 'selected_tile': 128}))
    assert restored.begin_training_epoch(model, 71)['active_tile'] == 128
    saved = restored.training_state_dict()
    third = Provider()
    third.training_case_ids = provider.training_case_ids
    _attach_paired_response(third, Objective(), ResponseObjectiveConfig(), spec)
    third.load_training_state_dict(saved)
    decision.write_text(json.dumps({'qualified_tile': 512, 'selected_tile': 512}))
    with pytest.raises(ValueError, match='differs'):
        third.begin_training_epoch(model, 72)


def test_canonical_validation_population_is_explicit_and_keeps_default():
    catalog = [{'case_id': f'{i:04}', 'split': 'test'} for i in range(89)]
    catalog.append({'case_id': '0273', 'split': 'test'})
    assert _response_validation_population({}, catalog, ('old',)) == ('old',)
    recipe = {'training': {'membership': 'fixed25_train150_dev22'},
              'evaluation': {'validation_population': 'canonical_VAL89'}}
    assert len(_response_validation_population(recipe, catalog, ('old',))) == 89


def test_calibration_preserves_rng_gradients_and_unattached_optimizer():
    model = torch.nn.Linear(1, 1)
    model.architecture = 'joint'
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    model.weight.grad = torch.full_like(model.weight, 3.)
    model.eval()
    before_model, before_opt = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    before_grad = model.weight.grad.clone()
    before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    objective = Objective()
    objective.schedule = SimpleNamespace(families={'original': tuple(f'{i:04}' for i in range(15)),
                                                   'new': tuple(f'{i:04}' for i in range(15, 40))})
    def family_components(model, source, family, components):
        random.random(); np.random.random(); torch.rand(1)
        factor = 2. if objective.auxiliary_fluid_measure == 'catalogue' else 1.
        return {'endpoint': factor * (model.weight + 2).square().sum(),
                'response': factor * (model.bias + 3).square().sum()}, {}
    objective.family_components = family_components
    provider = Provider()
    provider.family_objective = objective
    provider.native_evaluation_batches = lambda **kw: (SimpleNamespace(scene_inputs=None, receivers=None),)
    provider.predict_native = lambda model, *a, **kw: (model(torch.ones(1, 1)), {})
    provider.endpoint_state_loss = lambda prediction, batch: (prediction + 4).square().sum()
    panel = [('original', '0000'), ('original', '0001'), ('new', '0015'), ('new', '0016')]
    receipt = calibrate_paired_response(model, provider, epoch=10,
                                        native_ids=provider.training_case_ids[:15], fit_families=panel)
    assert receipt['coefficients']['endpoint'] > 0 and receipt['coefficients']['response'] > 0
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before_model.items())
    assert torch.equal(model.weight.grad, before_grad) and not model.training
    assert random.getstate() == before_python and torch.equal(torch.get_rng_state(), before_torch)
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and np.array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert before_opt['param_groups'] == optimizer.state_dict()['param_groups']
    for key, saved in before_opt['state'].items():
        assert all(torch.equal(value, optimizer.state_dict()['state'][key][name])
                   for name, value in saved.items())
