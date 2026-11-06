import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from channelthermal.training.r13_8_anchored_task import ARM_DETAIL
from channelthermal.training.r16_5_warm_fork import (
    SOURCE_AGE,
    SOURCE_CHECKPOINT,
    ThermalR165WarmForkEngine,
    initialize_shared_weights,
    make_warm_fork_recipe,
    validate_source_and_recipe,
)
from channelthermal.training.thermal_state_response_task import FIELD_ROLES
from relation_anchored_train import engine_for
from torch import nn

from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    TaskBatch,
    _sampling_dataset_id,
)


def identities():
    h_recipe = json.loads((Path(__file__).resolve().parents[1] /
                           'configs/research/thermal_state_response_h0_train600.json').read_text())
    p_recipe = copy.deepcopy(h_recipe)
    p_recipe.pop('thermal_state_response')
    p_recipe['initialization']['source_parent_age'] = 5000
    p_recipe['training']['local_horizon'] = 100
    p_recipe['training']['optimizer_schedule_horizon'] = 100
    p_recipe['optimizer']['lr'] = 1e-4
    p_recipe['thermal_port_response'] = {'schema': 'thermal_port_response_v1', 'enabled': True,
                                         'arm': 'P64', 'source_epoch': 5000}
    p_recipe['objective'].update(response_fit=False, response_training_families=[])
    fit = sorted(h_recipe['objective']['response_training_families'])
    binding = {'dataset_path': '/dataset', 'dataset_size_bytes': 1, 'dataset_mtime_ns': 2,
               'catalog_metadata_sha256': 'catalog', 'manifest_sha256': 'manifest'}
    adapter = {'normalization_stats': {'x': [1.]}, 'heat_scale': 1., 'native_grid_nx_ny': [128, 64],
               'environment_grid_nx_ny': [24, 8]}
    provider = {'dataset': 'ThermalChannel', 'subset_manifest_sha256': 'manifest', 'source_binding': binding,
                'adapter_state': adapter, 'training_case_ids': [str(i) for i in range(600)],
                'validation_case_ids': [str(i) for i in range(89)]}
    floors = h_recipe['thermal_state_response']['functional_floors']
    scales = {name: {'scale': 2., 'fit_rms': 2., 'floor': floors.get(name, 1.)}
              for name in (*FIELD_ROLES, 'J', 'P')}
    h_provider = {**copy.deepcopy(provider), 'response_training_family_ids': fit, 'response_scales': scales}
    engine = {'sampling_version': SamplingKey.CASE_EPOCH_VERSION, 'deterministic_case_order': True,
              'seed': 0, 'training_mode': 'joint'}
    p64 = {'recipe': p_recipe, 'provider_identity': provider, 'engine_config': engine,
           'sampling_dataset_id': _sampling_dataset_id(provider)}
    h0 = {'recipe': h_recipe, 'provider_identity': h_provider, 'engine_config': engine,
          'sampling_dataset_id': _sampling_dataset_id(h_provider)}
    metadata = {'sensitivity': dict(floors), 'research_scales': {'J': .2, 'P': .001},
                'response_scales': {name: scales[name]['scale'] for name in ('J', 'P')}}
    return p64, h0, metadata


def recipe(p64, h0, metadata, arm='PDelta'):
    return make_warm_fork_recipe(p64, h0, metadata, arm=arm, name='test', output_root='/tmp',
                                 membership='full_train600', local_horizon=100, first_segment_epochs=100)


def test_literal_p64_and_h0_recipe_sources_and_research_receipt():
    p64, h0, metadata = identities()
    source = {'epoch': 100, 'current_epoch': 100, 'arm': ARM_DETAIL,
              'history': [{'epoch': i} for i in range(1, 101)], 'experiment_identity': p64}
    control, decision = recipe(p64, h0, metadata), recipe(p64, h0, metadata, 'PDeltaD')
    for result in (control, decision):
        validate_source_and_recipe(source, result, p64, h0)
        assert result['initialization']['source_checkpoint'] == SOURCE_CHECKPOINT
        assert result['initialization']['source_parent_age'] == 5100
        assert result['optimizer'] == p64['recipe']['optimizer']
        assert result['thermal_port_response'] == p64['recipe']['thermal_port_response']
        assert result['r16_5_decision']['research_scales'] == metadata['research_scales']
    changed = copy.deepcopy(control)
    changed['objective']['native_weights']['flow/u'] *= 2
    with pytest.raises(ValueError, match='differs from the derived'):
        validate_source_and_recipe(source, changed, p64, h0)
    changed = copy.deepcopy(source)
    changed['epoch'] = changed['current_epoch'] = 99
    with pytest.raises(ValueError, match='literal P64'):
        validate_source_and_recipe(changed, control, p64, h0)
    mismatched = copy.deepcopy(metadata)
    mismatched['sensitivity']['P'] *= 2
    with pytest.raises(ValueError, match='shared FIT'):
        recipe(p64, h0, mismatched)
    changed_h0 = copy.deepcopy(h0)
    changed_h0['provider_identity']['adapter_state']['heat_scale'] *= 2
    with pytest.raises(ValueError, match='heat_scale'):
        recipe(p64, changed_h0, metadata)


def test_complete_tensor_inventory_shape_and_dtype_are_strict():
    model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3))
    source = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    receipt = initialize_shared_weights(model, source)
    assert receipt['tensor_names'] == sorted(source)
    for bad in ('missing', 'shape', 'dtype'):
        changed = copy.deepcopy(source)
        if bad == 'missing':
            changed.pop('0.bias')
        elif bad == 'shape':
            changed['0.bias'] = torch.zeros(4)
        else:
            changed['0.bias'] = changed['0.bias'].double()
        with pytest.raises(ValueError, match='inventories|shape or dtype'):
            initialize_shared_weights(model, changed)


def test_only_named_final_candidate_can_extend_the_formal_lineage():
    p64, h0, metadata = identities()
    control = recipe(p64, h0, metadata, 'PDelta')
    candidate = recipe(p64, h0, metadata, 'PDeltaD')
    with pytest.raises(ValueError, match='Unsupported explicit'):
        engine_for(control, device='cpu', execution_horizon=500)
    extended = engine_for(candidate, device='cpu', execution_horizon=500)
    assert extended.config.total_epochs == 500
    assert extended.config.checkpoint_epochs == (100, 200, 300, 400, 500)


class ToyProvider:
    def identity_payload(self): return {'dataset': 'toy', 'manifest': 'fixed'}
    def epoch_cases(self, epoch, seed): return tuple(range(5))
    def fit_sampler_epoch(self, epoch): return SOURCE_AGE + epoch
    def response_epoch_family_ids(self, epoch): return [f'fit-{self.fit_sampler_epoch(epoch)}']
    def response_epoch_action_ids(self, epoch): return [{'case_id': f'fit-{self.fit_sampler_epoch(epoch)}', 'action': 'a'}]
    def make_batch(self, cases, key):
        x = torch.tensor(cases, dtype=torch.float32).reshape(-1, 1)
        return TaskBatch(x, None, 2 * x + 1, {}, tuple(cases))
    def loss_denominators(self, batches, phase, arm):
        return {'native': float(sum(batch.targets.numel() for batch in batches))}
    def make_scene(self, scene): return scene
    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature): return model(scene), None
    def loss_terms(self, prediction, targets, phase, auxiliary):
        return {'native': LossTerm((prediction - targets).square().sum(), targets.numel())}
    def validation_batches(self):
        return [self.make_batch((0, 1), SamplingKey(0, 1, 0, 0, 'joint', 'A'))]
    def validation_metrics(self, prediction, targets, auxiliary): return {'mse': float((prediction - targets).square().mean())}
    def reduce_native_metrics(self, records): return {'field_score': np.mean([row['mse'] for row in records])}
    def optimizer_groups(self, model, arm, stage):
        return (OptimizerGroupSpec('joint', tuple(name for name, _ in model.named_parameters()),
            ScheduleSpec(peak_lr=.0001, warmup_start_lr=.0001, warmup_epochs=0, hold_through_epoch=1,
                         total_epochs=2, final_lr=.0001), weight_decay=1e-5, betas=(.9, .999), eps=1e-8),)
    def work_counts(self, batch, prediction, auxiliary): return {}


def test_local_history_sampler_and_adam_rng_resume(tmp_path):
    config = EngineConfig(seed=0, microbatch_cases=2, effective_cases=3, total_epochs=2,
        training_mode='joint', sampling_version=SamplingKey.CASE_EPOCH_VERSION, monitor_every=100,
        checkpoint_epochs=(2,), latest_every=1, curve_every=None, write_initial_artifacts=False,
        progress_enabled=False, gradient_clip=1.)
    engine = ThermalR165WarmForkEngine(config, device='cpu')
    torch.manual_seed(14)
    initial = nn.Sequential(nn.Linear(1, 2), nn.Dropout(.2), nn.Linear(2, 1))
    full, resumed = copy.deepcopy(initial), copy.deepcopy(initial)
    optimizer, _ = engine._make_optimizer(full, ToyProvider(), 'A', 1, None)
    assert not optimizer.state
    assert optimizer.param_groups[0]['lr'] == .0001
    assert optimizer.param_groups[0]['weight_decay'] == 1e-5
    identity = {'workflow': 'test_r16_5', 'recipe': {'name': 'same'}}
    torch.manual_seed(91)
    engine.fit(full, ToyProvider(), tmp_path / 'full', identity=identity, arm='A', stop_after=2)
    torch.manual_seed(91)
    engine.fit(resumed, ToyProvider(), tmp_path / 'resumed', identity=identity, arm='A', stop_after=1)
    engine.fit(resumed, ToyProvider(), tmp_path / 'resumed', identity=identity, arm='A', stop_after=2,
               resume_checkpoint='latest')
    saved = torch.load(resolve_checkpoint(tmp_path / 'resumed', 'latest'), map_location='cpu', weights_only=False)
    assert [row['epoch'] for row in saved['history']] == [1, 2]
    assert [row['sampler_epoch'] for row in saved['history']] == [5101, 5102]
    assert [row['fit_sampler_epoch'] for row in saved['history']] == [5101, 5102]
    assert saved['sampler_state']['next_fit_sampling_epoch'] == 5103
    assert all(len(row['case_ids']) == 5 for row in saved['history'])
    assert all(torch.equal(full.state_dict()[name], resumed.state_dict()[name]) for name in full.state_dict())
