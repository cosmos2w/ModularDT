import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from channelthermal.member_routing import ThermalDetailAnchoredMemberRoutingAdapter, ThermalHyperedgeNonlinearReadAdapter
from channelthermal.training import r13_6_member_routing_task as member_task
from channelthermal.training import thermal_state_response_task as response_task
from channelthermal.training.r16_4_warm_fork import (
    ThermalR164WarmForkEngine, ThermalR164WarmForkTask, initialize_shared_weights,
    make_warm_fork_recipe, validate_source_and_recipe,
)
from channelthermal.training.thermal_state_response_task import ACTIONS, FIELD_ROLES, ThermalResponseCorpus
from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.unified_training import (
    EngineConfig, LossTerm, OptimizerGroupSpec, SamplingKey, ScheduleSpec, TaskBatch, TrainingEngine,
    _sampling_dataset_id,
)


def source_fixture():
    recipe = json.loads((Path(__file__).resolve().parents[1] /
                         'configs/research/thermal_r16_1_h0_train150.json').read_text())
    recipe['training']['membership'] = 'full_train600'
    provider = {'dataset': 'ThermalChannel', 'subset_manifest_sha256': 'fixed-manifest',
                'training_case_ids': [str(i) for i in range(600)],
                'response_training_family_ids': [str(i) for i in range(48)],
                'response_scales': {name: {'scale': 2., 'fit_rms': 2., 'floor': 1.}
                                    for name in (*FIELD_ROLES, 'J', 'P')}}
    identity = {'recipe': recipe, 'provider_identity': provider,
                'engine_config': {'sampling_version': 'case_epoch_v1', 'deterministic_case_order': True,
                                  'seed': 0, 'training_mode': 'joint'},
                'sampling_dataset_id': _sampling_dataset_id(provider)}
    return {'epoch': 1000, 'current_epoch': 1000, 'arm': 'R14-Detail-LocalECover',
            'history': [{'epoch': i} for i in range(1, 1001)], 'experiment_identity': identity}


def fork_recipe(source, arm):
    return make_warm_fork_recipe(source['experiment_identity']['recipe'], arm=arm,
        name='test', output_root='/tmp', membership='fixed25_train150_dev22',
        local_horizon=3, first_segment_epochs=2)


def test_declared_source_recipe_preserves_scientific_settings():
    source = source_fixture()
    for arm in 'BCG':
        recipe = fork_recipe(source, arm)
        assert validate_source_and_recipe(source, recipe)['response_scales'] == source['experiment_identity']['provider_identity']['response_scales']
        assert recipe['optimizer'] == source['experiment_identity']['recipe']['optimizer']
        assert recipe['training']['optimizer_schedule_horizon'] == source['experiment_identity']['recipe']['training']['optimizer_schedule_horizon']
        changed = copy.deepcopy(recipe)
        changed['objective']['native_weights']['flow/u'] *= 2
        with pytest.raises(ValueError, match='scientific setting objective'):
            validate_source_and_recipe(source, changed)
        changed = copy.deepcopy(recipe)
        changed['initialization']['source_checkpoint'] = '/another/checkpoint.pt'
        with pytest.raises(ValueError, match='literal H0'):
            validate_source_and_recipe(source, changed)


def test_actual_adapters_strict_inventory_and_matched_new_initialization():
    source = source_fixture()
    stats = {'field_mean_by_channel': [0.] * 5, 'field_std_by_channel': [1.] * 5,
             'interface_targets_mean': [0.] * 2, 'interface_targets_std': [1.] * 2,
             'internal_temperature_mean': [0.], 'internal_temperature_std': [1.]}
    base = ThermalDetailAnchoredMemberRoutingAdapter(stats,
        core_config=source['experiment_identity']['recipe']['member_routing']['model'], heat_scale=3.)
    source_weights = {key: value.clone() for key, value in base.state_dict().items()}
    new = []
    for arm in 'BCG':
        recipe = fork_recipe(source, arm)
        adapter = ThermalDetailAnchoredMemberRoutingAdapter if arm == 'B' else ThermalHyperedgeNonlinearReadAdapter
        model = adapter(stats, core_config=recipe['member_routing']['model'], heat_scale=3.)
        receipt = initialize_shared_weights(model, source_weights, arm)
        assert len(receipt['new_tensor_names']) == (0 if arm == 'B' else 4)
        assert model.state_payload()['normalization_stats'] == base.state_payload()['normalization_stats']
        for name, value in source_weights.items():
            assert torch.equal(model.state_dict()[name], value)
        if arm != 'B':
            new.append({name: model.state_dict()[name].clone() for name in receipt['new_tensor_names']})
    assert all(torch.equal(new[0][name], new[1][name]) for name in new[0])
    assert not new[0]['core.hyperedge_read.output.weight'].count_nonzero()
    incomplete = dict(source_weights)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match='inventories'):
        initialize_shared_weights(base, incomplete, 'B')


def test_inherited_normalizer_and_heat_scale_are_used_without_refitting(monkeypatch, tmp_path):
    dataset = tmp_path / 'placeholder'
    dataset.touch()
    monkeypatch.setattr(member_task, '_load_data_binding', lambda *args: ({'manifest_sha256': 'manifest'}, dataset, None))
    monkeypatch.setattr(member_task, 'read_case_catalog', lambda *args: ({'metadata_sha256': 'metadata'}, []))
    monkeypatch.setattr(member_task, '_training_membership', lambda *args: (('train',), ('val',)))
    seen = []
    def read_cases(path, ids, *, split, normalizer, **kwargs):
        seen.append((split, normalizer.stats['field_std_by_channel'].copy()))
        return [{'case_id': ids[0]}]
    monkeypatch.setattr(member_task, '_read_selected_cases', read_cases)
    monkeypatch.setattr(member_task, 'fit_global_normalizer', lambda *args: pytest.fail('source statistics must not be refit'))
    class Adapter(nn.Module):
        def __init__(self, stats, *, core_config, heat_scale):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.stats, self.heat_scale = stats, heat_scale
    class Provider:
        def __init__(self, **kwargs): self.source_binding = kwargs['source_binding']
    recipe = {'member_routing': {'schema': 'section', 'arm': 'A',
               'model': {'schema': 'core', 'architecture': 'A', 'base_model_config': {'module_input_dim': 9}}},
              'initialization': {'source_kind': 'weight_only', 'seed': 0}, 'dataset': {}, 'training': {}}
    adapter, provider = member_task._build_thermal_member_routing_task(recipe, arm='A', task_type=Provider,
        variants={'core': {'arm': 'A', 'adapter': Adapter}}, section_schema='section',
        source_kind='weight_only', lineage='test', required_address_mode=None,
        saved_adapter_state={'normalization_stats': {'field_std_by_channel': [2., 3.]}, 'heat_scale': 4.})
    assert adapter.heat_scale == 4.
    assert all(np.array_equal(values, [2., 3.]) for _, values in seen)
    assert provider.source_binding['normalization_source'] == 'inherited source TRAIN600 adapter state'


def test_original_fit_selection_uses_source_age_and_records_local_actions(monkeypatch):
    corpus = object.__new__(ThermalResponseCorpus)
    corpus.cases = {str(i): {'role': 'fit', 'feasible': True} for i in range(48)}
    def provider():
        result = object.__new__(ThermalR164WarmForkTask)
        result.response_corpus, result.seed = corpus, 0
        result.response_spec, result.response_scales = {'tau': 1.}, {}
        result.response_labels = {(family, action): {} for family in corpus.family_ids('fit') for action in ACTIONS}
        result.response_finite_response_coefficient = .1
        result._response_epoch, result._response_epoch_families, result._response_epoch_actions = None, [], []
        return result
    monkeypatch.setattr(response_task, 'response_pair', lambda p, c, family, action, role: (family, action))
    monkeypatch.setattr(response_task, 'predict_response_pair', lambda *args, **kwargs:
        {'endpoint': torch.tensor(1.), 'field_delta': torch.tensor(1.), 'J_delta': torch.tensor(1.),
         'P_delta': torch.tensor(1.), 'work': {}})
    for local_epoch in (1, 2):  # second instance represents a same-lineage resume
        task = provider()
        for update in range(4):
            list(task.auxiliary_update_terms(None, epoch=local_epoch, update_index=update, update_count=4))
        expected = corpus.fit_families_for_epoch(1000 + local_epoch, 0)
        assert task.response_epoch_family_ids(local_epoch) == expected
        assert [(r['case_id'], r['action']) for r in task.response_epoch_action_ids(local_epoch)] == [
            (family, action) for family in expected for action in ACTIONS]


class ClockProvider:
    def identity_payload(self): return {'dataset': 'toy', 'manifest': 'fixed'}
    def epoch_cases(self, epoch, seed): return tuple(range(5))
    def fit_sampler_epoch(self, epoch): return epoch + 1000
    def response_epoch_family_ids(self, epoch): return [f'fit-{self.fit_sampler_epoch(epoch)}']
    def response_epoch_action_ids(self, epoch): return [{'case_id': f'fit-{self.fit_sampler_epoch(epoch)}', 'action': 'a'}]
    def make_batch(self, cases, key):
        x = torch.tensor(cases, dtype=torch.float32).reshape(-1, 1)
        digest = hashlib.sha256(str(key.seed_for('fluid')).encode()).hexdigest()
        return TaskBatch(x, None, 2 * x + 1, {'query_sampling_sha256': digest}, tuple(cases))
    def loss_denominators(self, batches, phase, arm):
        return {'native': float(sum(batch.targets.numel() for batch in batches))}
    def make_scene(self, scene): return scene
    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature): return model(scene), None
    def loss_terms(self, prediction, targets, phase, auxiliary): return {'native': LossTerm((prediction - targets).square().sum(), targets.numel())}
    def validation_batches(self):
        return [self.make_batch((0, 1), SamplingKey(0, 1, 0, 0, 'joint', 'A', sampling_version='case_epoch_v1', dataset_id=_sampling_dataset_id(self.identity_payload())))]
    def validation_metrics(self, prediction, targets, auxiliary): return {'mse': float((prediction - targets).square().mean())}
    def reduce_native_metrics(self, records): return {'field_score': np.mean([r['mse'] for r in records])}
    def optimizer_groups(self, model, arm, stage):
        return (OptimizerGroupSpec('joint', ('weight', 'bias'), ScheduleSpec(peak_lr=.01, warmup_start_lr=.01,
            warmup_epochs=0, hold_through_epoch=9, total_epochs=10, final_lr=.01)),)
    def work_counts(self, batch, prediction, auxiliary): return {}


def test_native_age_resume_and_fresh_optimizer(tmp_path):
    config = EngineConfig(seed=0, microbatch_cases=2, effective_cases=3, total_epochs=2,
        training_mode='joint', sampling_version='case_epoch_v1', monitor_every=100,
        checkpoint_epochs=(2,), latest_every=1, curve_every=None, write_initial_artifacts=False,
        progress_enabled=False)
    provider, engine = ClockProvider(), ThermalR164WarmForkEngine(config, device='cpu')
    plain = TrainingEngine(config, device='cpu')
    assert engine.epoch_case_order(provider, 1) == plain.epoch_case_order(provider, 1001)
    model = nn.Linear(1, 1)
    optimizer, _ = engine._make_optimizer(model, provider, 'A', 1, None)
    assert not optimizer.state
    identity = {'workflow': 'test_warm_fork', 'recipe': {'name': 'same'}}
    engine.fit(model, provider, tmp_path, identity=identity, arm='A', stop_after=1)
    engine.fit(model, ClockProvider(), tmp_path, identity=identity, arm='A', stop_after=2, resume_checkpoint='latest')
    saved = torch.load(resolve_checkpoint(tmp_path, 'latest'), map_location='cpu', weights_only=False)
    assert [r['epoch'] for r in saved['history']] == [1, 2]
    assert [r['sampler_epoch'] for r in saved['history']] == [1001, 1002]
    assert [r['fit_sampler_epoch'] for r in saved['history']] == [1001, 1002]
    assert [r['response_fit_family_ids'] for r in saved['history']] == [['fit-1001'], ['fit-1002']]
    assert all(len(r['case_ids']) == 5 for r in saved['history'])
    assert saved['sampler_state']['fit_sampling_epoch'] == 1002
    assert saved['sampler_state']['next_fit_sampling_epoch'] == 1003
