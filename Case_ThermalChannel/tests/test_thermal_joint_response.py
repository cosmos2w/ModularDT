import copy
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.interaction_evidence import reference_config
from channelthermal.physical_provenance import validate_thermal_endpoint_config
from channelthermal.training.r13_6_member_routing_task import _response_validation_ids
from channelthermal.training.thermal_joint_response_task import (
    FAMILY_SCHEMA, ThermalFitFamilyCorpus, _attach_response, build_thermal_joint_response_config,
    build_thermal_joint_response_task,
)
from channelthermal.training.thermal_native_port_task import ThermalNativePortTask
from honf_runtime.state_response import ResponseObjectiveConfig


def test_native_only_config_never_opens_response_files(monkeypatch):
    import channelthermal.training.thermal_joint_response_task as module
    recipe = build_thermal_joint_response_config({'member_routing': {'arm': 'example'}}, enabled=False,
                                                  fit_manifest='/not/a/manifest')
    provider = SimpleNamespace(configure_native=lambda: None, identity_payload=lambda: {})
    monkeypatch.setattr(module, '_build_thermal_member_routing_task', lambda *args, **kwargs: ('model', provider))
    monkeypatch.setattr(module, 'ThermalFitFamilyCorpus', lambda *args, **kwargs: pytest.fail('opened FIT corpus'))
    model, result = build_thermal_joint_response_task(recipe)
    assert model == 'model' and result is provider
    assert recipe['state_response']['enabled'] is False
    assert result.identity_payload()['state_response']['enabled'] is False
    assert result.state_response_metadata() == {'calibration': None}
    assert not hasattr(provider, 'auxiliary_update_terms')


def test_enabled_config_requires_manifest_and_scale_choice():
    with pytest.raises(ValueError, match='FIT manifest'):
        build_thermal_joint_response_config({}, enabled=True)
    with pytest.raises(ValueError, match='requires a FIT family manifest'):
        build_thermal_joint_response_task({'thermal_joint_response': {'schema': 'thermal_joint_response_v1'}})


def test_fit_manifest_requires_training_members_and_physical_labels(tmp_path):
    path = tmp_path / 'families.json'
    case = {'case_id': 'a', 'source_config': {}, 'source_case_dir': 'base', 'role': 'fit'}
    action = {'case_id': 'a', 'action': 'edit', 'centers': [[1., 1.]], 'raw_case_dir': 'edit'}
    def write(case_row, action_row):
        path.write_text(json.dumps({'schema': FAMILY_SCHEMA, 'cases': [case_row], 'actions': [action_row]}))
    write(case, action)
    with pytest.raises(ValueError, match='J/P response labels'):
        ThermalFitFamilyCorpus(path, ['a'])
    action['physical_delta'] = {'J': 1., 'P': -2.}
    write(case, action)
    corpus = ThermalFitFamilyCorpus(path, ['a'])
    assert corpus.family_ids() == ('a',)
    assert corpus.physical_delta('a', 'edit') == {'J': 1., 'P': -2.}
    assert corpus.cases['a']['source_case_dir'] == str(tmp_path / 'base')
    with pytest.raises(ValueError, match='TRAIN selection'):
        ThermalFitFamilyCorpus(path, ['other'])
    action['role'] = 'holdout'
    write(case, action)
    with pytest.raises(ValueError, match='FIT actions'):
        ThermalFitFamilyCorpus(path, ['a'])


def test_native_ports_16_angles_and_already_sampled_response_endpoint():
    from channelthermal.training.content_hypergraph_task import ThermalContentTargets, ThermalContentReceivers
    from honf_runtime.unified_training import TaskBatch
    task = ThermalNativePortTask.__new__(ThermalNativePortTask)
    task.recipe = {'thermal_joint_response': {'native_ports': {'angle_count': 16, 'outside_temperature_weight': .2}}}
    task.training_cases = task.validation_cases = ()
    task.configure_native()
    iface = torch.arange(128., dtype=torch.float32).reshape(1, 1, 64, 2)
    targets = ThermalContentTargets(('a',), (0,), 0, torch.zeros(1, 1, 5), torch.ones(1, 1),
                                    iface, torch.ones(1, 1, 64), torch.zeros(1, 1, 1))
    batch = TaskBatch(SimpleNamespace(structure={'module_present': torch.ones(1, 1)}),
                      ThermalContentReceivers(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), torch.ones(1, 1)),
                      targets, {}, ('a',))
    outside = np.arange(64, dtype=np.float32).reshape(1, 1, 64)
    valid = np.ones((1, 1, 64), dtype=bool)
    sampled = task._attach_ports(batch, iface, np.ones((1, 1, 64)), outside, valid)
    assert sampled.receivers.ntheta == 16
    assert torch.equal(sampled.targets.interface_target, iface[..., ::4, :])
    raw = {'interface_target': np.zeros((1, 64, 2)), 'teacher_ports': np.zeros((1, 64, 5)),
           'outside_valid': np.ones((1, 64), dtype=bool)}
    raw['teacher_ports'][0, :, 3] = np.arange(64)
    reattached = task.with_raw_port_targets(sampled, raw)
    assert reattached.targets.interface_target.shape[-2:] == (16, 2)
    assert reattached.targets.outside_temperature[0, 0].tolist() == list(range(0, 64, 4))


def test_validation_selection_is_reproducible_and_input_only():
    ids = tuple(str(i) for i in range(89))
    recipe = {'evaluation': {'sample_fraction': .1, 'sample_seed': 20261108}}
    selected = _response_validation_ids(recipe, ids)
    assert len(selected) == 9 and selected == _response_validation_ids(recipe, ids)
    assert selected == tuple(case for case in ids if case in selected)
    assert _response_validation_ids({}, ids) == ids[:9]
    with pytest.raises(ValueError, match='validation population'):
        _response_validation_ids({'evaluation': {'validation_case_ids': ['missing']}}, ids)


def test_train_gradient_calibration_is_saved_and_inactive_component_is_skipped():
    model = torch.nn.Linear(1, 1, bias=False)
    calls = []
    class Corpus:
        path = 'fit.json'
        def family_ids(self): return ('a', 'b')
    class Objective:
        corpus = Corpus()
        response_scales = {'J': 1., 'P': 1.}
        tau = 1.
        def family_components(self, model, source, family, *, components):
            calls.append(tuple(components))
            value = model.weight.sum()
            return {name: (value.square() if name == 'response' else 2 * value.square()) for name in components}, {}
        def actions(self, source, family): return ('one',)
    def provider():
        return SimpleNamespace(training_case_ids=('n1', 'n2'), identity_payload=lambda: {'objective': {}},
                               loss_metadata=lambda: {}, training_state_dict=lambda: {'n': 3},
                               load_training_state_dict=lambda value: None)
    config = ResponseObjectiveConfig(endpoint_weight=0., response_weight=1.,
        functional_weights={'J': .1, 'P': .1}, warmup_epochs=0, ramp_epochs=1)
    obj = Objective()
    first = _attach_response(provider(), obj, config, count=1, seed=7)
    assert first.needs_native_gradient_norm(epoch=1, update_index=0)
    first.record_native_gradient_norm(2., case_keys=('n1', 'n2'))
    list(first.auxiliary_update_terms(model, epoch=1, update_index=0, update_count=1))
    assert calls == [('response',)]
    saved = first.training_state_dict()
    calibration = saved['thermal_joint_response']['calibration']
    assert calibration['native_gradient_norm'] == 2.
    assert calibration['native_case_ids'] == ['n1', 'n2']
    assert first.state_response_metadata()['calibration'] == calibration
    assert calibration['component_gradient_norms']['endpoint'] == 0.
    assert calibration['coefficients']['endpoint'] == 0.
    restored = _attach_response(provider(), obj, config, count=1, seed=7)
    restored.load_training_state_dict(saved)
    assert not restored.needs_native_gradient_norm(epoch=2, update_index=0)
    assert restored.training_state_dict()['thermal_joint_response'] == saved['thermal_joint_response']


def _physical_config():
    result = asdict(reference_config.SimulationConfig())
    result['layout'].update(num_modules=1, centers=[[2., 2.]], heat_powers=[1.])
    return result


@pytest.mark.parametrize(('section', 'key', 'value'), [
    ('domain', 'module_radius', .5), ('flow', 'u_in', 2.), ('thermal', 'solid_k', 2.),
    ('layout', 'heat_powers', [3.]), ('local_module', 'solver_iterations', 3000),
])
def test_truth_join_checks_all_prescribed_controls(section, key, value):
    source = _physical_config()
    actual = copy.deepcopy(source)
    actual['layout']['centers'] = [[2.2, 2.]]
    validate_thermal_endpoint_config(source, actual, centers=[[2.2, 2.]])
    actual[section][key] = value
    with pytest.raises(ValueError, match='controls differ'):
        validate_thermal_endpoint_config(source, actual, centers=[[2.2, 2.]])


def test_reference_schema_known_defaults_and_optional_legacy_parity():
    defaults = asdict(reference_config.SimulationConfig())
    assert defaults['domain']['module_radius'] == .45
    assert defaults['flow']['flow_model'] == 'analytic_wake'
    assert defaults['thermal']['solid_k'] == 1.
    assert defaults['local_module']['n_interface_points'] == 64
    assert defaults['execution'] == {'device': 'cpu', 'gpu_id': 0}
    root = Path(__file__).resolve().parents[2]
    path = Path(os.environ.get('THERMAL_LEGACY_SCHEMA_PATH',
                str(root / '1_Demo_ChannelThermal/src/_helpers_forward/channelthermal_common.py')))
    if not path.exists():
        return
    name = '_test_legacy_thermal_schema'
    spec = importlib.util.spec_from_file_location(name, path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[name] = legacy
    try:
        spec.loader.exec_module(legacy)
        older = asdict(legacy.SimulationConfig())
        defaults['save'].pop('root_dir')
        older['save'].pop('root_dir')
        assert defaults == older
        config = {'layout': {'num_modules': 1, 'centers': [[2., 2.]], 'heat_powers': [1.]},
                  'flow': {'u_in': 2.}, 'execution': {'device': ' GPU '}}
        current = asdict(reference_config.config_from_dict(config))
        historic = asdict(legacy.config_from_dict(config))
        current['save'].pop('root_dir')
        historic['save'].pop('root_dir')
        assert current == historic
    finally:
        del sys.modules[name]


def test_thermal_candidate_pool_uses_supplied_physical_caps_and_sensitivity():
    from channelthermal.physical_response import evaluate_thermal_candidate_pool
    predicted = {'baseline': {'J': 0., 'P': 0.},
                 'over_cap': {'J': -2., 'P': .05}, 'near_null': {'J': -.5, 'P': .11}}
    physical = {'baseline': {'J': 0., 'P': 0.},
                'over_cap': {'J': -1., 'P': .3}, 'near_null': {'J': -.5, 'P': .005}}
    options = dict(objective='J', constraint_caps={'P': .1}, objective_scale=1.,
                   sensitivity={'J': .1, 'P': .01})
    result = evaluate_thermal_candidate_pool(predicted, physical, **options)
    assert result['model_selected_action'] == 'over_cap'
    assert result['raw_constraint_violation'] is True
    assert result['accepted_action'] == 'baseline'
    assert result['near_null_false_rejected_actions'] == ['near_null']
    assert result['physical_best_action'] == 'near_null'
    assert result['regret'] == pytest.approx(.5)
    pending = evaluate_thermal_candidate_pool(predicted, None, **options)
    assert pending['physical_verification'] == 'unavailable' and pending['accepted_action'] is None
    with pytest.raises(ValueError, match='J/P'):
        evaluate_thermal_candidate_pool(predicted, physical, **{**options, 'objective': 'energy'})


def test_calibration_file_is_bound_to_selected_train_and_fit(tmp_path, monkeypatch):
    import channelthermal.training.thermal_joint_response_task as module
    from channelthermal.training.thermal_state_response_task import FIELD_ROLES
    path = tmp_path / 'calibration.json'
    scales = {name: 1. for name in (*FIELD_ROLES, 'J', 'P')}
    payload = {'response_scales': scales, 'coefficients': {'endpoint': .2, 'response': .1},
               'native_case_ids': ['held_out'], 'fit_family_ids': ['family']}
    path.write_text(json.dumps(payload))
    recipe = build_thermal_joint_response_config({'member_routing': {'arm': 'example'}},
        fit_manifest='fit.json', calibration_path=path, families_per_epoch=1)
    class Corpus:
        def __init__(self, path, ids): self.path = str(path)
        def verify_training_controls(self, provider): pass
        def family_ids(self): return ('family',)
    def build(*args, **kwargs):
        return 'model', SimpleNamespace(training_case_ids=('native_train',), configure_native=lambda: None,
            identity_payload=lambda: {'objective': {}}, loss_metadata=lambda: {})
    monkeypatch.setattr(module, 'ThermalFitFamilyCorpus', Corpus)
    monkeypatch.setattr(module, '_build_thermal_member_routing_task', build)
    with pytest.raises(ValueError, match='TRAIN/FIT'):
        build_thermal_joint_response_task(recipe)
    payload['native_case_ids'] = ['native_train']
    path.write_text(json.dumps(payload))
    _, provider = build_thermal_joint_response_task(recipe)
    calibration = provider.training_state_dict()['thermal_joint_response']['calibration']
    assert calibration['source'] == 'explicit calibration file'
    assert calibration['coefficients'] == {'endpoint': .2, 'response': .1}
    assert calibration['native_case_ids'] == ['native_train']
    assert calibration['fit_family_ids'] == ['family']
    disabled = {**payload, 'coefficients': {'endpoint': 0., 'response': .1}}
    baseline = SimpleNamespace(training_case_ids=('native_train',), identity_payload=lambda: {'objective': {}},
                               loss_metadata=lambda: {})
    module._attach_response(baseline, provider.family_objective, module._response_config(recipe),
                            count=1, seed=0, external_calibration=disabled)
    assert baseline.auxiliary_loss_names == ('family/response',)
    payload['coefficients']['response'] = .3
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Saved calibration differs'):
        module._attach_response(provider, provider.family_objective, module._response_config(recipe), count=1, seed=0,
                                restored_calibration=calibration, external_calibration=payload)


def test_new_recipe_rejects_legacy_selector_and_drops_old_response_flags():
    with pytest.raises(ValueError, match='legacy selectors'):
        build_thermal_joint_response_config({'r16_10_family': {'arm': 'B0'}}, enabled=False)
    recipe = build_thermal_joint_response_config({'objective': {'response_fit': False,
        'response_training_families': [], 'organizer_objective': False}}, enabled=False)
    assert recipe['objective'] == {'organizer_objective': False}


def test_catalog_train_uses_native_split_membership_without_fixed25(tmp_path):
    import h5py
    from channelthermal.training.r13_6_member_routing_task import _response_catalog_binding
    path = tmp_path / 'native.h5'
    ids = ['t1', 't2', 't3', 'v1', 'v2', 'v3', 'test1']
    splits = ['train', 'train', 'train', 'validation', 'validation', 'validation', 'test']
    with h5py.File(path, 'w') as h5:
        h5['case_ids'] = np.asarray(ids, dtype='S')
        h5['splits'] = np.asarray(splits, dtype='S')
        cases = h5.create_group('cases')
        for case_id in ids:
            group = cases.create_group(case_id)
            group['module_present'] = np.array([1])
            group['heat_powers'] = np.array([1.])
            group['module_centers'] = np.array([[2., 2.]])
            group.create_group('material_parameters').attrs['re'] = 50.
    recipe = {'dataset': {'data_path': str(path)}, 'training': {'membership': 'catalog_train'},
              'evaluation': {}}
    manifest, result_path, source, catalog, training, validation, split = _response_catalog_binding(recipe)
    assert result_path == path and len(catalog) == 7
    assert training == ('t1', 't2', 't3') and validation == ('v1', 'v2', 'v3')
    assert split == 'validation' and manifest['manifest_sha256'] == source['metadata_sha256']
    selected = copy.deepcopy(recipe)
    selected['training']['train_case_ids'] = ['t3', 't1']
    assert _response_catalog_binding(selected)[4] == ('t3', 't1')
    selected['training']['train_case_ids'] = ['test1']
    with pytest.raises(ValueError, match='TRAIN split'):
        _response_catalog_binding(selected)
    selected = copy.deepcopy(recipe)
    selected['evaluation']['validation_split'] = 'missing'
    with pytest.raises(ValueError, match='No native cases'):
        _response_catalog_binding(selected)
    selected['evaluation']['validation_split'] = 'test'
    assert _response_catalog_binding(selected)[5] == ('test1',)


def test_catalog_builder_loads_only_selected_train_and_sampled_validation(tmp_path, monkeypatch):
    from channelthermal.training import r13_6_member_routing_task as builder
    dataset = tmp_path / 'native.h5'
    dataset.touch()
    catalog = [{'case_id': key, 'split': split} for key, split in
               [('t1', 'train'), ('t2', 'train'), ('v1', 'validation'),
                ('v2', 'validation'), ('v3', 'validation'), ('v4', 'validation'), ('x', 'test')]]
    source = {'metadata_sha256': 'existing-catalogue-id'}
    monkeypatch.setattr(builder, '_response_catalog_binding', lambda recipe: (
        {'manifest_sha256': 'existing-catalogue-id'}, dataset, source, catalog,
        ('t1', 't2'), ('v1', 'v2', 'v3', 'v4'), 'validation'))
    monkeypatch.setattr(builder, '_load_data_binding', lambda *a, **k: pytest.fail('fixed25 binding used'))
    monkeypatch.setattr(builder, '_training_membership', lambda *a, **k: pytest.fail('fixed membership used'))
    monkeypatch.setattr(builder, 'fit_global_normalizer', lambda *a, **k: SimpleNamespace(stats={}))
    calls = []
    def read_cases(path, ids, *, split, **kwargs):
        calls.append((tuple(ids), split))
        return [{'case_id': case_id, 'structure': {'heat_powers': np.array([1.]),
                 'module_present': np.array([1.])}} for case_id in ids]
    monkeypatch.setattr(builder, '_read_selected_cases', read_cases)
    monkeypatch.setattr(builder, '_fresh_member_routing_adapter', lambda *a, **k: 'model')
    recipe = {'training_profile': 'state_response', 'dataset': {'data_path': str(dataset)},
              'training': {'membership': 'catalog_train'}, 'evaluation': {},
              'initialization': {'source_kind': 'fresh', 'seed': 0},
              'member_routing': {'schema': 'section', 'arm': 'arm', 'model': {
                  'schema': 'core', 'architecture': 'arm', 'base_model_config': {'module_input_dim': 9}}}}
    model, provider = builder._build_thermal_member_routing_task(recipe, arm='arm',
        task_type=lambda **kwargs: SimpleNamespace(**kwargs), variants={'core': {'arm': 'arm', 'adapter': object}},
        section_schema='section', source_kind='fresh', lineage='test', required_address_mode=None)
    assert model == 'model' and provider.protocol == 'catalog_train'
    assert calls == [(('t1', 't2'), 'train'), (('v1',), 'validation')]
    assert provider.source_binding['manifest_path'] is None


def test_catalog_native_loader_preserves_partition_and_requested_order(monkeypatch):
    import channelthermal.training.joint_task as loader
    task = ThermalNativePortTask.__new__(ThermalNativePortTask)
    task.recipe = {'training': {'membership': 'catalog_train'}}
    task.training_case_ids, task.validation_case_ids = ('t1',), ('v1', 'v2')
    task.case_split_by_id = {'t1': 'train', 'v1': 'validation', 'v2': 'validation', 'x': 'test'}
    task.data_path = '/unused'
    task.adapter = SimpleNamespace(normalization_stats={}, core=SimpleNamespace(module_input_dim=9))
    calls = []
    def read(path, ids, *, split, **kwargs):
        calls.append((tuple(ids), split))
        return [{'case_id': case_id} for case_id in ids]
    monkeypatch.setattr(loader, '_read_selected_cases', read)
    assert [row['case_id'] for row in task._load_native_cases(('v2', 't1', 'v1'))] == ['v2', 't1', 'v1']
    assert calls == [(('v2', 'v1'), 'validation'), (('t1',), 'train')]
    with pytest.raises(ValueError, match='selected TRAIN/validation'):
        task._load_native_cases(('x',))


def test_new_profile_seed_uses_init_for_lineage_and_train_seed_for_sampling(monkeypatch):
    from channelthermal.training.r13_8_anchored_task import ThermalR138AnchoredMemberRoutingTask
    captured = []
    def validate(parent, initialization):
        captured.append(parent.seed)
    monkeypatch.setattr(ThermalR138AnchoredMemberRoutingTask, '_validate_initialization', validate)
    task = ThermalNativePortTask.__new__(ThermalNativePortTask)
    task.recipe = {'training_profile': 'state_response', 'training': {'seed': 17}}
    task.seed = 0
    task._validate_initialization({'seed': 9})
    assert captured == [9] and task.seed == 17
    task.recipe = {'training': {'seed': 17}}
    task.seed = 0
    task._validate_initialization({'seed': 9})
    assert captured == [9, 0] and task.seed == 0


def test_catalog_profile_identity_records_explicit_test_exposure(monkeypatch):
    import channelthermal.training.thermal_joint_response_task as module
    base = {'member_routing': {'arm': 'example'}, 'training': {'membership': 'catalog_train'},
            'evaluation': {'validation_split': 'test'}}
    recipe = build_thermal_joint_response_config(base, enabled=False)
    provider = SimpleNamespace(configure_native=lambda: None,
                               identity_payload=lambda: {'test_read': False})
    monkeypatch.setattr(module, '_build_thermal_member_routing_task', lambda *args, **kwargs: ('model', provider))
    _, result = build_thermal_joint_response_task(recipe)
    assert result.identity_payload()['validation_source_split'] == 'test'
    assert result.identity_payload()['test_read'] is True
