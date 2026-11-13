"""Self-contained packing, role isolation, legacy views, and audit regression."""
import copy
import csv
import json
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest

import thermal_pack_finite_responses as pack
from channelthermal.data.finite_responses import FiniteResponseStore
from channelthermal.interaction_evidence.reference_config import SimulationConfig
from channelthermal.training.thermal_response_raw import load_raw_endpoint


def _endpoint(path, center, *, record=False):
    path.mkdir(parents=True)
    (path / 'scene').mkdir()
    config = asdict(SimulationConfig())
    config['domain'].update(nx=16, ny=8, lx=16., ly=8., module_radius=.4)
    config['layout'].update(centers=[[center, 4.]], heat_powers=[1.25], num_modules=1)
    config['thermal'].update(convergence_window=10, convergence_tol=.001, convergence_rel_tol=.001)
    (path / 'case_config.json').write_text(json.dumps(config))
    with (path / 'frame_index.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['file', 'step', 'heat_active', 'converged', 'delta_inf', 'delta_l2_rel'])
        writer.writeheader()
        for i in range(10):
            writer.writerow({'file': 'final.npz', 'step': i + 1, 'heat_active': 1, 'converged': int(i == 9),
                             'delta_inf': .0001, 'delta_l2_rel': .0001})
    names = np.asarray(['theta', 'normal_x', 'normal_y', 'T_surface', 'T_outside', 'q_normal'], dtype='S9')
    interface = np.zeros((1, 64, 6), np.float32); interface[..., 3] = 3; interface[..., 4] = 1; interface[..., 5] = 2
    field = np.arange(128, dtype=np.float32).reshape(8, 16)
    frame = {k: field + i for i, k in enumerate(('u', 'v', 'p', 'omega', 'temperature'))}
    frame.update(module_mask=np.zeros((8, 16), np.uint8), module_id=np.full((8, 16), -1, np.int32),
                 module_internal_temperature=np.arange(4, dtype=np.float32).reshape(1, 2, 2),
                 module_internal_mask=np.ones((2, 2), np.uint8), interface_response=interface,
                 interface_feature_names=names)
    np.savez_compressed(path / 'scene/final.npz', **frame)
    np.savez_compressed(path / 'grid.npz', x_grid=field, y_grid=field + 1)
    record_path = None
    if record:
        record_path = path.parent / (path.name + '_record'); record_path.mkdir()
        roles = [{'role': 'fluid_fields', 'channel_names': ['u'], 'channel_units': ['unit'], 'query_ids': ['a'],
                  'receiver_module_ids': None, 'coordinate_kind': 'eulerian',
                  'arrays': {'query_features': 'q', 'values': 'v', 'valid_mask': 'm', 'quadrature_weights': 'w'}}]
        roles.extend([{**roles[0], 'role': name} for name in ('interface', 'solid_temperature')])
        metadata = {'schema_version': 1, 'record_id': 'edit', 'source': 'reference_solver', 'status': 'converged',
                    'elapsed_seconds': 1., 'design': {'anchor_id': '0001', 'physical_family_id': 'f', 'split': 'train',
                    'modules': [{'module_id': '0001:module:0', 'position_xy': [center, 4.], 'heating': 1.25, 'active': True}]},
                    'context': {}, 'provenance': {}, 'output': {'roles': roles,
                    'quantities': {'pressure_drop': {'value': 0., 'units': 'unit', 'resolved': True}},
                    'active_module_ids': ['0001:module:0'], 'module_peak_temperature': {'0001:module:0': 1.},
                    'units_metadata': {}, 'case_dir': str(path)}}
        (record_path / 'record.json').write_text(json.dumps(metadata))
        np.savez_compressed(record_path / 'physical_output.npz', q=np.array([[.25, .5]], np.float64),
                            v=np.array([[1.0000000000001]], np.float64), m=np.ones((1, 1), bool),
                            w=np.array([1.], np.float64))
    return config, record_path


def _fixture(tmp_path):
    native_root = tmp_path / 'raw'; baseline = native_root / 'train/case_0001_fixture'
    edit = tmp_path / 'edit'
    config, _ = _endpoint(baseline, 5.)
    edited_config, record = _endpoint(edit, 5.1250000001, record=True)
    case = {'case_id': '0001', 'source': 'original', 'source_case_dir': str(baseline), 'source_config': config, 'role': 'fit'}
    action = {'case_id': '0001', 'action': 'edit', 'source': 'original', 'role': 'fit',
              'centers': edited_config['layout']['centers'], 'raw_case_dir': str(edit),
              'record_directory': str(record), 'physical_delta': {'J': .1, 'P': .2}}
    family = {'family_id': '0001', 'role': 'fit', 'source': 'original', 'populations': ['TRAIN150', 'TRAIN600'],
              'metadata': case, 'actions': [{'action': 'edit', 'role': 'fit', 'populations': ['TRAIN150', 'TRAIN600'],
              'metadata': action, 'saved_delta': action['physical_delta'],
              'saved_delta_provenance': {'source': 'action_manifest', 'path': 'fixture'}}]}
    families = [family]; sources = pack.endpoint_sources(families)
    native = tmp_path / 'native.h5'
    with h5py.File(native, 'w') as h5:
        h5['case_ids'] = np.array([b'0001']); h5['splits'] = np.array([b'train'])
        for key in ('channel_order', 'interface_feature_names', 'interface_condition_feature_names',
                    'interface_target_names', 'sampled_point_feature_names'):
            h5[key] = np.array([b'unit'])
        h5.create_group('normalization')['field_mean'] = np.array([2.], np.float32)
        case_group = h5.create_group('cases').create_group('0001')
        case_group['module_centers'] = np.array([[5., 4.]], np.float32)
        case_group['heat_powers'] = np.array([1.25], np.float32)
        case_group['module_present'] = np.array([1.], np.float32)
        case_group['steady_field'] = np.arange(3, dtype=np.float32)
        case_group['sampled_points'] = np.array([[.25, .5], [.25, .5]], np.float32)
        case_group['sampled_point_weights'] = np.array([1., 2.], np.float32)
    def save(name, data):
        p = tmp_path / name; p.write_text(json.dumps(data)); return p
    args = Namespace(native_h5=native, dataset_manifest=save('dataset.json', {'schema': 'example'}),
        native_manifest=save('native.json', {'partitions': {'train': {'case_ids': ['0001']}}}),
        panels=save('panels.json', {'native_case_ids': ['0001'], 'validation_case_ids': []}), raw_native_root=native_root,
        calibration150=None, calibration600=None,
        train150_manifest=save('train150.json', {'cases': [case], 'actions': [action]}),
        train600_manifest=save('train600.json', {'cases': [case], 'actions': [action]}),
        historical_manifest='historical', native_seen_manifest='seen', fresh_manifest='fresh', ledger=[],
        original_holdout_labels=None, historical_new_labels=None, packer_revision='fixture', maximum_bytes=2**30)
    refs = pack.reference_sources(sources, args)
    manifest = {'cases': [case], 'actions': [action], 'source_original_recipe': 'fixture-original',
                'source_new_manifest': 'fixture-new', 'source_original_ledger': 'fixture-ledger'}
    output = tmp_path / 'responses.h5'; summary = pack.census(families, sources)
    pack.write_pack(output, families, sources, refs, manifest, manifest, args, summary)
    return output, families, sources, refs, args, baseline, edit, record


def test_pack_exact_views_roles_and_source_unavailable(tmp_path):
    path, families, sources, refs, args, baseline, edit, record = _fixture(tmp_path)
    audit = pack.audit(path, families, sources, refs, args)
    assert (audit['endpoints'], audit['families'], audit['actions'], audit['record_metadata']) == (2, 1, 1, 1)
    before = load_raw_endpoint(edit)
    store = FiniteResponseStore(path)
    assert 'saved_delta' not in store._rows['0001']['actions'][0]
    assert store.family_ids(role='fit', population='TRAIN150', source='original') == ('0001',)
    assert store.actions('0001', role='fit', population='TRAIN600') == ('edit',)
    assert store.physical_delta('0001', 'edit', role='fit', population='TRAIN150', kind='saved') == {'J': .1, 'P': .2}
    assert store.physical_delta_provenance('0001', 'edit', role='fit')['source'] == 'action_manifest'
    assert 'tau' in store.physical_delta_provenance('0001', 'edit', role='fit', kind='derived')
    with pytest.raises(ValueError): store.actions('0001', role='response_holdout')
    with pytest.raises(ValueError): store.endpoint('0001', 'edit', role='fit', view='unsupported')
    with pytest.raises(ValueError): store.observation_record('0001', 'baseline', role='fit')
    baseline.rename(baseline.with_name('hidden_base')); edit.rename(edit.with_name('hidden_edit'))
    record.rename(record.with_name('hidden_record'))
    actual = store.endpoint('0001', 'edit', role='fit', population='TRAIN150')
    for key in ('field', 'fluid_mask', 'material', 'interface_target', 'teacher_ports', 'teacher_h_valid', 'outside_valid'):
        np.testing.assert_array_equal(actual[key], before[key])
    assert actual['config']['layout']['centers'][0][0] == 5.1250000001
    legacy = store.legacy_view_components('0001', 'edit', role='fit', population='TRAIN150')
    assert legacy['profile'] == 'native_baseline_record_edit'
    assert legacy['edited_record'].output.roles['fluid_fields'].values.dtype == np.float64
    assert legacy['native_baseline']['sampled_point_weights'].tolist() == [1., 2.]
    np.testing.assert_array_equal(store.reference_baseline('0001')['field'], store.endpoint('0001', role='fit')['field'])
    assert store.reference_provenance('0001')['convergence']['status'] == 'passed'
    assert 'delta_l2_rel' in store.reference_frame_index('0001')


def test_audit_detects_changed_saved_delta(tmp_path):
    path, families, sources, refs, args, *_ = _fixture(tmp_path)
    with h5py.File(path, 'r+') as h5:
        dataset = h5['families/f0000/actions/a0000/saved_delta_json']
        dataset[0] = ord('x')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        pack.audit(path, families, sources, refs, args)


def test_convergence_rejects_nonfinite_inside_last_window():
    config = {'thermal': {'convergence_window': 2, 'convergence_tol': .001, 'convergence_rel_tol': .001}}
    rows = [{'step': '1', 'converged': '0', 'heat_active': '1', 'delta_inf': '.0001', 'delta_l2_rel': '.0001'},
            {'step': '2', 'converged': '0', 'heat_active': '1', 'delta_inf': 'nan', 'delta_l2_rel': '.0001'},
            {'step': '3', 'converged': '1', 'heat_active': '1', 'delta_inf': '.0001', 'delta_l2_rel': '.0001'}]
    assert pack.convergence(rows, config)['status'] == 'failed_or_unavailable'


def test_historical_saved_labels_are_attributed_and_consistent(tmp_path):
    row = {'case_id': 'x', 'action': 'move', 'delta': {'J': {'physical': .1}, 'P': {'physical': -.2}}}
    path = tmp_path / 'original.json'
    path.write_text(json.dumps({'models': {'a': {'full': {'actions': [row]}},
                                           'b': {'full': {'actions': [copy.deepcopy(row)]}}}}))
    assert pack.saved_original_labels(path) == {('x', 'move'): {'J': .1, 'P': -.2}}
    data = json.loads(path.read_text()); data['models']['b']['full']['actions'][0]['delta']['J']['physical'] = .2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='differ'):
        pack.saved_original_labels(path)
    path.write_text(json.dumps({'schema': 'r16_5_physical_candidate_truth_v1',
                                'truth': {'x': {'actions': {'move': {'delta': {'J': .3, 'P': -.4}}}}}}))
    assert pack.saved_historical_new_labels(path) == {('x', 'move'): {'J': .3, 'P': -.4}}
