import numpy as np
import pytest
import torch

from honf_runtime.state_response import (
    NativePairScene, native_functional_contrast, scheduled_native_pairs,
    select_native_pairs, validate_common_coordinates,
)


def test_input_matching_is_disjoint_split_safe_and_order_independent():
    scenes = [NativePairScene(str(i), split, 'same', float(i)) for i, split in
              [(1, 'train'), (2, 'train'), (3, 'train'), (4, 'train'), (5, 'test')]]
    distance = lambda x, y: abs(x - y)
    eligible = lambda x, y: (x.case_id, y.case_id) != ('1', '2')
    pairs, unmatched = select_native_pairs(scenes, distance, eligible)
    assert (pairs, unmatched) == select_native_pairs(reversed(scenes), distance, eligible)
    assert pairs == (('2', '3'), ('1', '4'))
    assert unmatched == ('5',)


def test_pair_schedule_independent_of_global_rng_and_resume():
    pairs = tuple((str(i), str(i + 20)) for i in range(7))
    first = scheduled_native_pairs(pairs, 5, 3, 12)
    np.random.seed(77)
    np.random.random(500)
    assert first == scheduled_native_pairs(pairs, 5, 3, 12)
    for epoch in range(1, 40):
        assert len(set(scheduled_native_pairs(pairs, epoch, 3, 12))) == 3


def test_coordinates_and_common_support_are_required():
    xy = np.array([[0., 1., 2.], [1., 1., 2.]])
    assert validate_common_coordinates(xy, xy.copy(), ([True, True], [False, True])).tolist() == [False, True]
    with pytest.raises(ValueError):
        validate_common_coordinates(xy, xy + .01, ([True, True],))
    with pytest.raises(ValueError):
        validate_common_coordinates(xy, xy, ([False, False],))


def test_contrast_is_zero_at_truth_and_keeps_gradients():
    truth = torch.tensor([[[1., 2., 3., 4.], [2., 4., 6., 8.]]], dtype=torch.float64)
    pred = truth.unsqueeze(0).repeat(2, 1, 1, 1).requires_grad_()
    loss = native_functional_contrast(pred, truth, [1., 2., 3., 4.])
    assert loss.item() == 0.
    offset = pred + 3
    assert native_functional_contrast(offset, truth, [1., 2., 3., 4.]).item() == 0.
    assert (offset - truth).square().mean().item() == 9.
    shifted = pred.clone()
    shifted[:, :, 1, :] = shifted[:, :, 1, :] + 1
    native_functional_contrast(shifted, truth, [1., 2., 3., 4.]).backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_recipe_removes_edited_corpus_and_rejects_inherited_weights():
    from channelthermal.training.thermal_native_pair_task import build_native_pair_recipe
    base = {'initialization': {'source_checkpoint': None, 'pretrained_weights': False},
            'state_response': {'enabled': True},
            'thermal_joint_response': {'h5': {'path': 'edited.h5'}},
            'member_routing': {'model': {'architecture': 'joint'}}}
    new = build_native_pair_recipe(base, variant='functional_contrast', validation_pairs=[('001', '002')])
    assert new['training_profile'] == 'native_pair'
    assert 'thermal_joint_response' not in new and 'state_response' not in new
    assert new['member_routing'] == base['member_routing']
    assert base['thermal_joint_response']['h5']['path'] == 'edited.h5'
    base['initialization']['source_checkpoint'] = 'historical.pt'
    with pytest.raises(ValueError, match='fresh'):
        build_native_pair_recipe(base, variant='functional_contrast')


def test_native_resolution_floors_are_plain_json_scalars(monkeypatch):
    import json
    from types import SimpleNamespace
    from channelthermal.training import thermal_native_pair_task as native
    monkeypatch.setattr(native, 'native_pair_references', lambda *args: torch.ones(1, 2, 4))
    provider = SimpleNamespace(data_path='native.h5', model=SimpleNamespace(field_std=torch.ones(5)))
    objective = native.ThermalNativePairObjective(provider, [('0001', '0002')])
    assert all(type(value) is float for value in objective.floors)
    assert json.loads(json.dumps({'floors': objective.floors}))['floors'] == objective.floors


def test_pair_provider_exposure_and_calibration_resume(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import channelthermal.training.thermal_native_pair_task as native
    import channelthermal.training.thermal_joint_response_task as historical
    import channelthermal.training.r16_10_diagnostics as diagnostics

    class Provider:
        def __init__(self, model):
            self.model = model
            self.training_case_ids = tuple(f'{i:03}' for i in range(150))
            self.validation_case_ids = ('v1', 'v2')
            self.training_samples = 0
            self.seed = 0
            self.sampling_version = 'case_epoch_v1'
            self.sampling_dataset_id = 'native'
            self.recipe = {'training': {}}
            self.native_weights = {'field': 1.}
            self.primary_queries = 1024
            self.outside_temperature_weight = .05
            self.port_angle_count = 64
            self.data_path = 'native.h5'
        def identity_payload(self):
            return {}
        def loss_metadata(self):
            return {}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.core = SimpleNamespace(query_tile_size=0)
            self.architecture = 'joint'
        def state_payload(self):
            return {'normalization': {'scale': 1.}}

    class Objective:
        def __init__(self, provider, pairs, *, training_scales=None):
            self.pairs = pairs
            self.scales = torch.ones(4) if training_scales is None else training_scales
            self.floors = [1.] * 4
        def components(self, model, pair, *, epoch):
            return {'absolute': model.weight.square(), 'contrast': 2 * model.weight.square()}, {'case_pair': pair}

    monkeypatch.setattr(historical, 'build_thermal_joint_response_task',
                        lambda *a, **k: (lambda model: (model, Provider(model)))(Model()))
    pairs = tuple((f'{2*i:03}', f'{2*i+1:03}') for i in range(8))
    monkeypatch.setattr(native, 'select_thermal_native_pairs', lambda *a, **k: (pairs, ()))
    monkeypatch.setattr(native, 'ThermalNativePairObjective', Objective)
    monkeypatch.setattr(diagnostics, 'gradient_diagnostics', lambda *a, **k:
                        {'groups': {'global': {'norms': {'native': 4., 'absolute': 2., 'contrast': 1.}}}})
    base = {'initialization': {}, 'training': {'optimizer_schedule_horizon': 2000}, 'thermal_native_pair':
            {'variant': 'state_anchors', 'validation_pairs': [['v1', 'v2']], 'pair_seed': 9}}
    model, anchor = native.build_thermal_native_pair_task(base)
    contrast_recipe = {**base, 'thermal_native_pair': {**base['thermal_native_pair'], 'variant': 'functional_contrast'}}
    contrast_model, contrast = native.build_thermal_native_pair_task(contrast_recipe)
    shared_path = tmp_path / 'shared.json'
    anchor.native_pair_calibration_output = str(shared_path)
    anchor.native_pair_calibration_source_run = '/anchor/run'
    anchor.native_pair_calibration_source_checkpoint = '/anchor/run/checkpoints/latest_model.pt'
    pre_calibration = anchor.training_state_dict()
    anchor.begin_training_epoch(model, 11)
    recovered_model, recovered_anchor = native.build_thermal_native_pair_task(base)
    recovered_anchor.load_training_state_dict(pre_calibration)
    recovered_anchor.native_pair_calibration_output = str(shared_path)
    recovered_anchor.native_pair_calibration_source_run = '/anchor/run'
    recovered_anchor.native_pair_calibration_source_checkpoint = '/anchor/run/checkpoints/latest_model.pt'
    monkeypatch.setattr(diagnostics, 'gradient_diagnostics', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('must reuse the already published calibration')))
    recovered_anchor.begin_training_epoch(recovered_model, 11)
    assert recovered_anchor.training_state_dict()['native_pair']['calibration']['coefficients'] == anchor.training_state_dict()['native_pair']['calibration']['coefficients']
    contrast.native_pair_calibration_input = str(shared_path)
    contrast.begin_training_epoch(contrast_model, 11)
    assert (anchor.training_state_dict()['native_pair']['calibration']['coefficients']
            == contrast.training_state_dict()['native_pair']['calibration']['coefficients'])
    import copy, json
    record = json.loads(shared_path.read_text())
    for key, replacement in (('training_case_ids', ['wrong']), ('pairs', [['bad', 'pair']]),
                             ('scales', [5., 1., 1., 1.]), ('definition', {}),
                             ('adapter_state', {'normalization': {'scale': 2.}})):
        changed = copy.deepcopy(record); changed[key] = replacement
        path = tmp_path / f'bad_{key}.json'; path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='does not match'):
            contrast.read_native_pair_calibration(path)
    changed = copy.deepcopy(record); changed['calibration']['coefficients']['absolute'] *= 2
    path = tmp_path / 'bad_coefficient.json'; path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='invalid epoch, pairs, gradients or coefficients'):
        contrast.read_native_pair_calibration(path)
    def presentations(provider, model):
        return [row for slot in range(4) for row in provider.auxiliary_update_terms(
            model, epoch=50, update_index=slot, update_count=4)]
    rows, contrast_rows = presentations(anchor, model), presentations(contrast, contrast_model)
    assert len(rows) == len(contrast_rows) == 3
    assert [work['case_pair'] for _, work in rows] == [work['case_pair'] for _, work in contrast_rows]
    assert all(a['native_pair/absolute'].weight == b['native_pair/absolute'].weight
               for (a, _), (b, _) in zip(rows, contrast_rows))
    assert all('native_pair/contrast' in terms for terms, _ in contrast_rows)
    saved = anchor.training_state_dict()
    resumed_model, resumed = native.build_thermal_native_pair_task(base)
    resumed.load_training_state_dict(saved)
    resumed_path = tmp_path / 'resumed_anchor.json'
    resumed.native_pair_calibration_output = str(resumed_path)
    resumed.native_pair_calibration_source_run = '/anchor/run'
    resumed.native_pair_calibration_source_checkpoint = '/anchor/run/checkpoints/latest_model.pt'
    resumed.begin_training_epoch(resumed_model, 101)
    assert json.loads(resumed_path.read_text())['source']['completed_epoch'] == 100
    assert resumed.read_native_pair_calibration(resumed_path)['coefficients'] == saved['native_pair']['calibration']['coefficients']
    resumed.begin_training_epoch(resumed_model, 51)
    assert [w['case_pair'] for _, w in presentations(resumed, resumed_model)] == [w['case_pair'] for _, w in rows]
    old_contrast = contrast.training_state_dict()
    old_contrast['native_pair']['calibration']['coefficients']['absolute'] = .9
    old_contrast['native_pair']['adoptions'] = []
    resumed_contrast_model, resumed_contrast = native.build_thermal_native_pair_task(contrast_recipe)
    resumed_contrast.load_training_state_dict(old_contrast)
    adoption = resumed_contrast.adopt_native_pair_calibration(shared_path, source_epoch=100)
    assert adoption['previous_calibration']['coefficients']['absolute'] == .9
    assert adoption['adopted_calibration']['coefficients'] == anchor.training_state_dict()['native_pair']['calibration']['coefficients']
    assert resumed_contrast.adopt_native_pair_calibration(shared_path, source_epoch=100) is None
    different_path = tmp_path / 'different_source.json'
    different_path.write_text(shared_path.read_text())
    with pytest.raises(ValueError, match='different shared'):
        resumed_contrast.adopt_native_pair_calibration(different_path, source_epoch=100)
    original_record = shared_path.read_text()
    changed_source = json.loads(original_record); changed_source['source']['run'] = '/another/anchor'
    shared_path.write_text(json.dumps(changed_source))
    with pytest.raises(ValueError, match='different shared'):
        resumed_contrast.adopt_native_pair_calibration(shared_path, source_epoch=100)
    shared_path.write_text(original_record)
    normal_model, normal_resume = native.build_thermal_native_pair_task(contrast_recipe)
    normal_resume.load_training_state_dict(resumed_contrast.training_state_dict())
    normal_resume.begin_training_epoch(normal_model, 101)
    assert (normal_resume.training_state_dict()['native_pair']['calibration']['coefficients']
            == anchor.training_state_dict()['native_pair']['calibration']['coefficients'])


def test_case_pair_colocation_preserves_each_visit_and_stable_order():
    from channelthermal.training.thermal_native_pair_task import colocate_native_pairs
    order = ('unpaired', 'third', 'first', 'fourth', 'second')
    pairs = (('first', 'second'), ('third', 'fourth'))
    assert colocate_native_pairs(order, pairs) == ('unpaired', 'third', 'fourth', 'first', 'second')
    with pytest.raises(ValueError):
        colocate_native_pairs(order, (('first', 'second'), ('first', 'third')))


def test_train150_schedule_amendment_only_allows_equal_rate_identity_correction():
    import copy
    from channelthermal.training.thermal_native_pair_task import native_pair_schedule_amendment
    from honf_runtime.unified_training import ScheduleSpec
    old_schedule = {'peak_lr': .0003, 'warmup_start_lr': .0003, 'warmup_epochs': 0,
                    'hold_through_epoch': 1999, 'total_epochs': 2000, 'final_lr': .0003}
    new_schedule = {**old_schedule, 'hold_through_epoch': 1000, 'final_lr': .00003}
    identity = {'workflow': 'relation_anchored_case_training_v1',
                'recipe': {'training_profile': 'native_pair',
                           'thermal_native_pair': {'variant': 'state_anchors'},
                           'training': {'membership': 'fixed25_train150_dev22', 'local_horizon': 1000,
                                        'optimizer_schedule_horizon': 2000}},
                'provider_identity': {'training': {'optimizer_schedule': old_schedule}, 'data': 'fixed'},
                'optimizer_schedule_contract': [{'name': 'joint', 'parameter_names': ['weight'],
                    'schedule': old_schedule, 'weight_decay': 1e-5, 'betas': [.9, .999], 'eps': 1e-8}]}
    corrected = copy.deepcopy(identity)
    corrected['provider_identity']['training']['optimizer_schedule'] = copy.deepcopy(new_schedule)
    corrected['optimizer_schedule_contract'][0]['schedule'] = copy.deepcopy(new_schedule)
    result = native_pair_schedule_amendment(identity, corrected, source_epoch=100, stop_after=500)
    assert result['kind'] == 'native_pair_train150_schedule_correction'
    assert all(ScheduleSpec(**old_schedule).value(e) == ScheduleSpec(**new_schedule).value(e)
               for e in range(1, 1001))
    assert ScheduleSpec(**old_schedule).value(1001) != ScheduleSpec(**new_schedule).value(1001)
    with pytest.raises(ValueError):
        native_pair_schedule_amendment(identity, corrected, source_epoch=100, stop_after=1001)
    modified = copy.deepcopy(corrected)
    modified['provider_identity']['data'] = 'different'
    with pytest.raises(ValueError):
        native_pair_schedule_amendment(identity, modified, source_epoch=100, stop_after=500)
    modified = copy.deepcopy(corrected)
    modified['optimizer_schedule_contract'][0]['betas'] = [.8, .999]
    with pytest.raises(ValueError):
        native_pair_schedule_amendment(identity, modified, source_epoch=100, stop_after=500)
    modified = copy.deepcopy(corrected)
    modified['recipe']['training']['membership'] = 'full_train600'
    with pytest.raises(ValueError):
        native_pair_schedule_amendment(identity, modified, source_epoch=100, stop_after=500)
