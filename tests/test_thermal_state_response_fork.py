import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from channelthermal.training.state_response_fork import (
    PROJECTION, insert_area_column, make_fork_payload, validate_fork_recipe,
)
from honf_runtime.unified_training import (
    EngineConfig, OptimizerGroupSpec, ScheduleSpec, SelectionPolicy, TrainingEngine,
    _optimizer_spec_payload, named_optimizer_state,
)


def recipe_pair(area):
    root = Path(__file__).resolve().parents[1]
    parent = json.loads((root / 'configs/research/thermal_r16_1_h0_train150.json').read_text())
    suffix = 'raster_area' if area else 'area_control'
    child = json.loads((root / f'configs/research/thermal_state_response_{suffix}_train150.json').read_text())
    return parent, child


class Model(nn.Module):
    def __init__(self, area):
        super().__init__()
        self.core = nn.Module()
        self.core.module_encoder = nn.Sequential(nn.Linear(12 + int(area), 4))


class Provider:
    def __init__(self, recipe, model):
        self.recipe, self.model = recipe, model
        self.loaded = None

    def identity_payload(self):
        return {'recipe': copy.deepcopy(self.recipe), 'response_scales': {'J': {'scale': 2.}},
                'normalization': [2., 3.],
                'adapter_state': {'module_input_dim': self.recipe['member_routing']['model']['base_model_config']['module_input_dim']}}

    def load_training_state_dict(self, state):
        self.loaded = copy.deepcopy(state)

    def optimizer_groups(self, model, arm, phase):
        age = self.recipe['training']['optimizer_schedule_horizon']
        return [OptimizerGroupSpec('joint', tuple(dict(model.named_parameters())),
                    ScheduleSpec(peak_lr=.0003, warmup_start_lr=.0003, warmup_epochs=0,
                                 hold_through_epoch=age-1, total_epochs=age, final_lr=.0003),
                    weight_decay=.00001)]


@pytest.mark.parametrize('area', [False, True])
def test_fork_preserves_common_parameters_moments_history_and_clocks(area):
    parent_recipe, child_recipe = recipe_pair(area)
    parent, child = Model(False), Model(area)
    pp, cp = Provider(parent_recipe, parent), Provider(child_recipe, child)
    optimizer = torch.optim.AdamW(parent.parameters(), lr=.0003)
    optimizer.param_groups[0]['group_name'] = 'joint'
    x = torch.randn(3, 12)
    parent.core.module_encoder(x).square().mean().backward(); optimizer.step()
    state = named_optimizer_state(parent, optimizer)
    assert state['state_by_name'][PROJECTION]['exp_avg'].abs().sum() > 0
    source = {'epoch': 1000, 'current_epoch': 1000, 'arm': 'R14-Detail-LocalECover',
              'model_state_dict': copy.deepcopy(parent.state_dict()),
              'optimizer_state_by_name': state,
              'experiment_identity': {'recipe': parent_recipe, 'provider_identity': pp.identity_payload()},
              'history': [{'epoch': i} for i in range(1, 1001)],
              'provider_training_state': {'families': 4000, 'actions': 16000},
              'rng_state': {'torch_cpu': torch.get_rng_state().clone()},
              'sampler_state': {'next_epoch': 1001},
              'optimizer_schedule_contract': [_optimizer_spec_payload(s) for s in pp.optimizer_groups(parent, '', '')]}
    engine = TrainingEngine(EngineConfig(seed=0, microbatch_cases=1, effective_cases=1,
                                        total_epochs=1250), device='cpu', selection=SelectionPolicy())
    result, receipt = make_fork_payload(source, child_recipe, child, cp, engine,
                                       child_recipe['checkpoint_fork']['source_checkpoint'])
    assert receipt['additional_epochs'] == 250
    assert result['history'] == source['history'] and cp.loaded == source['provider_training_state']
    assert result['sampler_state'] == source['sampler_state']
    assert torch.equal(result['rng_state']['torch_cpu'], source['rng_state']['torch_cpu'])
    for name, value in source['model_state_dict'].items():
        actual = result['model_state_dict'][name]
        if area and name == PROJECTION:
            assert torch.equal(actual, insert_area_column(value)) and not actual[:, 9].count_nonzero()
        else:
            assert torch.equal(actual, value)
    for name, states in source['optimizer_state_by_name']['state_by_name'].items():
        for key, value in states.items():
            actual = result['optimizer_state_by_name']['state_by_name'][name][key]
            expected = insert_area_column(value) if area and name == PROJECTION and key != 'step' else value
            assert torch.equal(actual, expected)
    child_x = torch.cat((x[:, :9], torch.ones(3, 1), x[:, 9:]), -1) if area else x
    torch.testing.assert_close(parent.core.module_encoder(x), child.core.module_encoder(child_x), atol=1e-7, rtol=1e-6)
    assert source['model_state_dict'][PROJECTION].shape[1] == 12


def test_fork_rejects_other_scientific_changes():
    parent, child = recipe_pair(True)
    child['objective']['native_weights']['flow/u'] *= 2
    with pytest.raises(ValueError, match='only its label'):
        validate_fork_recipe(parent, child, child['checkpoint_fork']['source_checkpoint'])
