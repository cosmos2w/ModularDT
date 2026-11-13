import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.training import thermal_family_objective as family_module
from channelthermal.training.joint_member_routing_task import ThermalJointPairPredictions
from channelthermal.training.thermal_family_objective import ThermalFamilyObjectiveMixin, family_reference_roi_deltas
from channelthermal.training.thermal_joint_response_task import (
    ThermalH5FitFamilyCorpus, _attach_r17_2_response, _r17_2_panel, calibrate_r17_2_response,
    fit_training_roi_scales,
)
from honf_runtime.state_response import ResponseObjectiveConfig


def _family():
    xy = torch.tensor([[[.3, .2], [.4, .2], [.7, .2], [.3, .7], [.4, .7], [.7, .7]]])
    batches = []
    for index in range(5):
        mask = torch.ones((1, 6))
        if index == 1:
            mask[0, 1] = 0
        if index == 2:
            mask[0, 5] = 0
        physical_delta = torch.tensor([1., 1., 1., 2., 2., 2.]) * index
        target = torch.zeros((1, 6, 5))
        target[..., 4] = 10. + physical_delta
        batches.append(SimpleNamespace(scene_inputs=SimpleNamespace(index=index),
            receivers=SimpleNamespace(fluid_xy=xy),
            targets=SimpleNamespace(field_targets=target, point_weights=mask)))
    raw = {'config': {'domain': {'lx': 1., 'ly': 1., 'module_radius': .1},
                      'layout': {'centers': [[.5, .5]]}}}
    return batches, raw


def test_common_roi_support_and_train_rms_use_all_edits_and_resolution():
    batches, raw = _family()
    support, deltas = family_reference_roi_deltas(batches, raw)
    assert support['A'].tolist() == [[True, False, True, False, False, False]]
    assert support['B'].tolist() == [[False, False, False, True, True, False]]
    assert [float(value) for value in deltas['A']] == [1., 2., 3., 4.]
    assert [float(value) for value in deltas['B']] == [2., 4., 6., 8.]
    class Corpus:
        def family_ids(self): return ('train1',)
        def actions(self, family): return ('a', 'b', 'c', 'd')
        def family_context(self, family): return nullcontext()
        def pair(self, provider, family, action):
            return {'baseline': batches[0], 'edited': batches[('a', 'b', 'c', 'd').index(action) + 1],
                    'raw_baseline': raw}
    provider = SimpleNamespace(training_case_ids=('train1',))
    reference = {'role': 'train', 'view': 'r17_1_raw', 'source': 'saved repeats', 'native_case_ids': ['train1']}
    profile = fit_training_roi_scales(provider, Corpus(), population='TRAIN150', view='r17_1_raw',
                                      resolution={'A': 1., 'B': .1}, reference=reference)
    assert profile['rms'] == pytest.approx({'A': np.sqrt(7.5), 'B': np.sqrt(30.)})
    assert profile['scales'] == pytest.approx({'A': 4., 'B': np.sqrt(30.)})
    assert profile['support_counts'] == {'train1': {'A': 2, 'B': 2}}
    assert profile['support_runs'] == {'train1': {'A': [[0, 1], [2, 3]], 'B': [[3, 5]]}}
    reference['role'] = 'validation'
    with pytest.raises(ValueError, match='TRAIN'):
        fit_training_roi_scales(provider, Corpus(), population='TRAIN150', view='r17_1_raw',
                                resolution={'A': 1., 'B': .1}, reference=reference)


def test_endpoint_only_skips_response_and_functional_reuses_five_predictions(monkeypatch):
    batches, raw = _family()
    calls = []
    class Provider:
        native_weights = {}
        def with_raw_port_targets(self, batch, raw): return batch
        def endpoint_state_loss(self, prediction, batch): return prediction.full.native_main['pred_field'].square().mean()
        def predict_native(self, model, scene, receivers, *args, **kwargs):
            calls.append(scene.index)
            base = model.weight * (scene.index * .2 + receivers.fluid_xy[..., 0] * .1)
            def branch(factor):
                field = torch.stack([base * factor] * 5, dim=-1)
                return SimpleNamespace(native_main={'pred_field': field},
                    native_prepared=SimpleNamespace(stencils={'fluid': SimpleNamespace(valid=torch.ones_like(base, dtype=torch.bool))}))
            return ThermalJointPairPredictions(branch(1.), branch(2.), {}), {'work': {}}
    class Objective(ThermalFamilyObjectiveMixin):
        provider = Provider()
        response_scales = {'J': 2., 'P': 4.}
        roi_scales = {'A': 2., 'B': 4.}
        tau = .4
        sensitivity = research_scales = None
        def actions(self, source, family): return ('a', 'b', 'c', 'd')
        def pair(self, source, family, action):
            index = self.actions(source, family).index(action) + 1
            return {'baseline': batches[0], 'edited': batches[index], 'raw_baseline': raw, 'raw_edited': raw}
        def label(self, source, family, action):
            index = self.actions(source, family).index(action) + 1
            return {'J': .5 * index, 'P': .4 * index}
    class Model(torch.nn.Module):
        architecture = 'joint'
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
    model, objective = Model(), Objective()
    def forbidden(*args, **kwargs): raise AssertionError('endpoint-only performed response work')
    monkeypatch.setattr(family_module, '_endpoint_functionals', forbidden)
    monkeypatch.setattr(family_module, '_field_components', forbidden)
    result, _ = objective.family_components(model, 'fit', 'train1', batched=False, components=('endpoint',))
    assert set(result) == {'endpoint'} and calls == [0, 1, 2, 3, 4]
    calls.clear()
    def functional(prediction, batch, raw, tau):
        index = batch.scene_inputs.index
        return torch.stack([torch.stack([model.weight * factor * index,
                                         2 * model.weight * factor * index]) for factor in (1., 2.)])
    monkeypatch.setattr(family_module, '_endpoint_functionals', functional)
    result, _ = objective.family_components(model, 'fit', 'train1', batched=False,
                                             components=('endpoint', 'functional'))
    expected = []
    for factor in (1., 2.):
        for index in range(1, 5):
            expected.append((((factor * index - .5 * index) / 2) ** 2
                + ((2 * factor * index - .4 * index) / 4) ** 2
                + ((.2 * factor * index - index) / 2) ** 2
                + ((.2 * factor * index - 2 * index) / 4) ** 2) / 4)
    assert result['functional'].item() == pytest.approx(np.mean(expected), rel=1e-6)
    result['functional'].backward()
    assert torch.isfinite(model.weight.grad) and model.weight.grad != 0
    assert calls == [0, 1, 2, 3, 4]


class _Provider:
    training_case_ids = tuple(f'{i:04}' for i in range(150))
    seed = 0
    sampling_version = 'case_epoch_v1'
    sampling_dataset_id = 'selected-train'
    def identity_payload(self): return {'objective': {}}
    def loss_metadata(self): return {}


class _Objective:
    def __init__(self):
        self.corpus = SimpleNamespace(path='source', cases={
            f'{i:04}': {'source': 'original' if i < 15 else 'new'} for i in range(40)})
        self.corpus.family_ids = lambda: tuple(self.corpus.cases)
        self.response_scales = {name: 1. for name in ('flow/u', 'flow/v', 'flow/p', 'flow/omega',
            'temperature/fluid', 'temperature/surface', 'temperature/material', 'q_proxy', 'J', 'P')}
        self.tau = .4
        self.calls = []
    def actions(self, source, family): return ('a', 'b', 'c', 'd')
    def family_components(self, model, source, family, *, components):
        self.calls.append(tuple(components))
        return {name: model.weight.square().sum() for name in components}, {}


def _profile(objective):
    return {'schema': 'thermal_training_roi_scales_v1', 'population': 'TRAIN150', 'view': 'r17_1_raw',
        'fit_family_ids': list(objective.corpus.family_ids()), 'resolution': {'A': .1, 'B': .1},
        'floors': {'A': .4, 'B': .4}, 'rms': {'A': .8, 'B': .9}, 'scales': {'A': .8, 'B': .9},
        'resolution_reference': {'role': 'train', 'view': 'r17_1_raw', 'source': 'saved repeats',
                                 'native_case_ids': ['0000']},
        'support_counts': {family: {'A': 2, 'B': 2} for family in objective.corpus.family_ids()},
        'support_runs': {family: {'A': [[0, 2]], 'B': [[3, 5]]} for family in objective.corpus.family_ids()}}


@pytest.mark.parametrize('variant,component,coefficient', [('R-G', 'response', .2), ('E', None, None),
                                                           ('C', 'response', .3), ('F', 'functional', .4)])
def test_recipe_uses_fixed_coefficients_and_restores_schedule(tmp_path, variant, component, coefficient):
    provider, objective = _Provider(), _Objective()
    path = tmp_path / 'calibration.json'
    spec = {'variant': variant, 'calibration_path': str(path), 'roi_profile': _profile(objective)}
    config = ResponseObjectiveConfig(field_response_weight=.8, functional_weights={'J': .1, 'P': .1})
    _attach_r17_2_response(provider, objective, config, spec,
                           population='TRAIN150', view='r17_1_raw')
    families = tuple((source, family) for source in ('original', 'new')
                     for family in objective.schedule.families[source])
    receipt = {'schema': 'thermal_r17_2_calibration_v1', 'epoch': 10,
        'native_case_ids': _r17_2_panel(provider.training_case_ids), 'fit_families': _r17_2_panel(families),
        'response_scales': objective.response_scales, 'roi_scales': objective.roi_scales,
        'coefficients': {'endpoint': .1, 'response_grid': .2, 'response_catalogue': .3, 'functional': .4}}
    model = torch.nn.Linear(1, 1)
    model.core = SimpleNamespace(query_tile_size=999)
    assert provider.begin_training_epoch(model, 11)['active_tile'] == 128
    assert list(provider.auxiliary_update_terms(model, epoch=10, update_index=0, update_count=3)) == []
    path.write_text(json.dumps(receipt))
    rows = [entry for slot in range(3) for entry in
            provider.auxiliary_update_terms(model, epoch=11, update_index=slot, update_count=3)]
    expected_names = ('endpoint',) if component is None else ('endpoint', component)
    assert len(rows) == 3 and objective.calls == [expected_names] * 3
    assert rows[0][0]['family/endpoint'].weight == pytest.approx(.025 * .1)
    if component is not None:
        assert rows[0][0]['family/' + component].weight == pytest.approx(.025 * coefficient)
    else:
        assert set(rows[0][0]) == {'family/endpoint'}
    saved = provider.training_state_dict()
    other, restored = _Objective(), _Provider()
    _attach_r17_2_response(restored, other, config, spec,
                           population='TRAIN150', view='r17_1_raw')
    restored.load_training_state_dict(copy.deepcopy(saved))
    assert other.schedule.state_dict() == objective.schedule.state_dict()
    assert restored.identity_payload() == provider.identity_payload()
    receipt['coefficients']['endpoint'] = .5
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='differs'):
        restored.load_training_state_dict(saved)


def test_common_calibration_measures_each_response_component_and_preserves_state():
    from honf_runtime.unified_training import SamplingKey
    model, provider, objective = torch.nn.Linear(1, 1, bias=False), _Provider(), _Objective()
    with torch.no_grad(): model.weight.fill_(1.)
    model.architecture = 'joint'
    model.eval()
    model.weight.grad = torch.full_like(model.weight, 7.)
    objective.schedule = SimpleNamespace(families={'original': tuple(f'{i:04}' for i in range(15)),
                                                   'new': tuple(f'{i:04}' for i in range(15, 40))})
    objective.auxiliary_fluid_measure, objective.roi_scales = 'grid', {'A': .8, 'B': .9}
    def family_components(model, source, family, components):
        component = components[0]
        factor = {'endpoint': 2., 'functional': 16.}.get(component)
        if component == 'response':
            factor = 8. if objective.auxiliary_fluid_measure == 'catalogue' else 4.
        return {component: factor * model.weight.square().sum()}, {}
    objective.family_components = family_components
    provider.family_objective = objective
    provider.seed, provider.primary_queries, provider.port_angle_count = 0, 1024, 64
    provider.sampling_version, provider.sampling_dataset_id = 'case_epoch_v1', 'selected-train'
    provider.recipe = {'training': {'sampling_arm': 'R13-3-Relative'}}
    provider.training_samples = 23
    seen = []
    def make_batch(case_keys, key):
        assert len(case_keys) == 1 and key.epoch == 10 and key.seed == 0
        assert key.sampling_version == 'case_epoch_v1' and key.dataset_id == 'selected-train'
        case = case_keys[0]
        # The singleton calibration packing must reproduce the ordinary
        # epoch/case query stream despite a different engine microbatch slot.
        training_key = SamplingKey(0, 10, 2, 1, 'joint', 'any-arm',
                                   sampling_version='case_epoch_v1', dataset_id='selected-train')
        for size, count, role in ((8192, 1024, 'fluid'), (100, 32, 'material')):
            assert np.array_equal(key.native_indices(size, count, case, role, 'primary_native_queries'),
                                  training_key.native_indices(size, count, case, role, 'primary_native_queries'))
        seen.append(case)
        provider.training_samples += 1
        return SimpleNamespace(case_keys=case_keys, scene_inputs=None,
            receivers=SimpleNamespace(ntheta=64),
            targets=SimpleNamespace(field_targets=torch.zeros(1, 1024, 5),
                material_targets=torch.zeros(1, 2, 32), interface_target=torch.zeros(1, 2, 64, 2)),
            auxiliary={})
    provider.make_batch = make_batch
    provider.native_evaluation_batches = lambda **kwargs: pytest.fail('full native evaluation was used')
    provider.predict_native = lambda model, *args, **kwargs: (model(torch.ones(1, 1)), {})
    provider.endpoint_state_loss = lambda prediction, batch: (prediction + 1.).square().sum()
    families = tuple((source, family) for source in ('original', 'new')
                     for family in objective.schedule.families[source])
    before_rng = torch.get_rng_state().clone()
    receipt = calibrate_r17_2_response(model, provider, epoch=10,
        native_ids=_r17_2_panel(provider.training_case_ids), fit_families=_r17_2_panel(families))
    assert receipt['coefficients'] == pytest.approx({'endpoint': .25, 'response_grid': .05,
        'response_catalogue': .025, 'functional': .0125})
    assert seen == list(_r17_2_panel(provider.training_case_ids))
    assert receipt['native_sampling']['dataset_id'] == 'selected-train'
    assert receipt['native_sampling']['seed'] == 0 and receipt['native_sampling']['epoch'] == 10
    assert receipt['native_sampling']['fluid_queries_per_case'] == 1024
    assert receipt['native_sampling']['material_queries_per_module'] == 32
    assert receipt['native_sampling']['surface_and_outside_angles'] == 64
    assert provider.training_samples == 23
    assert not model.training and torch.equal(model.weight.grad, torch.full_like(model.weight, 7.))
    assert torch.equal(before_rng, torch.get_rng_state())


def test_direct_fit_corpus_is_role_scoped_and_never_reads_raw_paths(monkeypatch):
    from channelthermal.data import finite_responses
    from channelthermal.data.datasets import _local_disk_query_points
    from channelthermal.training import thermal_joint_response_task as task_module

    config = {'layout': {'centers': [[.4, .5]], 'heat_powers': [2.], 'num_modules': 1}}
    mask = np.array([[True, False], [True, False]])
    raw = {'config': config, 'material_mask': mask, 'field': np.zeros((2, 2, 5), dtype=np.float32)}
    calls = []
    class Store:
        def __init__(self, path): self.path = path
        def family_ids(self, **kwargs):
            assert kwargs == {'role': 'fit', 'population': 'TRAIN150'}
            return ('train1',)
        def family_metadata(self, family, **kwargs):
            assert kwargs == {'role': 'fit', 'population': 'TRAIN150'}
            return {'source': 'original', 'source_config': config, 'source_case_dir': '/unavailable/raw'}
        def actions(self, family, **kwargs):
            assert kwargs == {'role': 'fit', 'population': 'TRAIN150'}
            return ('a', 'b')
        def action_metadata(self, family, action, **kwargs):
            assert kwargs == {'role': 'fit', 'population': 'TRAIN150'}
            return {'centers': [[.4, .5]], 'raw_case_dir': '/unavailable/' + action}
        def physical_delta(self, family, action, **kwargs):
            assert kwargs == {'role': 'fit', 'population': 'TRAIN150', 'kind': 'saved'}
            return {'J': 1., 'P': 2.}
        def endpoint(self, family, action, **kwargs):
            assert kwargs == {'role': 'fit', 'population': 'TRAIN150', 'view': 'r17_1_raw'}
            calls.append(action)
            return copy.deepcopy(raw)
    monkeypatch.setattr(finite_responses, 'FiniteResponseStore', Store)
    monkeypatch.setattr(task_module, 'load_raw_endpoint', lambda path: pytest.fail('raw path opened'))
    monkeypatch.setattr(ThermalH5FitFamilyCorpus, '_verify_saved', lambda *args: None)
    monkeypatch.setattr(task_module, '_native_endpoint_batch', lambda *args: SimpleNamespace(
        receivers=SimpleNamespace(heat=torch.tensor([[2.]]))))
    original = {'structure': {'module_centers': np.array([[.4, .5]], dtype=np.float32),
        'heat_powers': np.array([2.], dtype=np.float32)},
        'module_internal_query_points': _local_disk_query_points(mask)}
    provider = SimpleNamespace(_load_native_cases=lambda families: [original])
    corpus = ThermalH5FitFamilyCorpus('/packed.h5', ('train1',), population='TRAIN150')
    with corpus.family_context('train1'):
        corpus.pair(provider, 'train1', 'a')
        corpus.pair(provider, 'train1', 'b')
    assert calls == ['baseline', 'a', 'b']
    assert corpus.physical_delta('train1', 'a') == {'J': 1., 'P': 2.}
    with pytest.raises(KeyError):
        corpus.pair(provider, 'train1', 'withheld')
