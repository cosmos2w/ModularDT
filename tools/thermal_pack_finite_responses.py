#!/usr/bin/env python3
"""Stream the declared Thermal finite-response sources into one HDF5 pack.

All input paths and storage limits are explicit. Use --inventory-only before a
build; --audit compares every packed frame, grid, record and native case array
against the source. Publishing uses a same-filesystem no-clobber hard link.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'Case_ThermalChannel/src'))
import h5py
import numpy as np
from channelthermal.data.finite_responses import SCHEMA, raw_endpoint_from_arrays

ACTIONS = ('single_plus', 'single_minus', 'joint_plus', 'joint_minus')
TAU = .4149511814117432


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def put_bytes(group, name, data):
    array = np.frombuffer(data, dtype=np.uint8)
    group.create_dataset(name, data=array, compression='gzip', compression_opts=1, shuffle=True)


def put_json(group, name, value):
    put_bytes(group, name, json_bytes(value))


def put_array(group, name, data):
    data = np.asarray(data)
    kwargs = {'compression': 'gzip', 'compression_opts': 1, 'shuffle': True} if data.ndim and data.size else {}
    group.create_dataset(name, data=data, **kwargs)


def ledger_rows(paths):
    by_raw, attempts = {}, {}
    for path in paths:
        path = Path(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('event') == 'finished' and row.get('raw_case_dir'):
                raw = str(Path(row['raw_case_dir']).resolve())
                if row.get('record_directory') and row.get('serialization_status') == 'stored':
                    by_raw.setdefault(raw, []).append(row)
            if row.get('event') in ('started', 'finished') and row.get('attempt') is not None:
                attempts.setdefault((str(path), row['attempt']), {})[row['event']] = row
    return by_raw, attempts


def saved_original_labels(path):
    data = read_json(path)
    views = [view for model in data['models'].values() for view in model.values()]
    if not views:
        raise ValueError('Historical original evaluation has no saved views.')
    def labels(view):
        result = {}
        for row in view['actions']:
            key = (str(row['case_id']), str(row['action']))
            if key in result:
                raise ValueError('Duplicate historical original action label.')
            result[key] = {name: float(row['delta'][name]['physical']) for name in ('J', 'P')}
        return result
    result = labels(views[0])
    if any(labels(view) != result for view in views[1:]):
        raise ValueError('Historical original physical labels differ between saved model views.')
    return result


def saved_historical_new_labels(path):
    data = read_json(path)
    if data.get('schema') != 'r16_5_physical_candidate_truth_v1':
        raise ValueError('Unexpected historical-new physical candidate truth schema.')
    return {(str(family), str(action)): {name: float(values['delta'][name]) for name in ('J', 'P')}
            for family, case in data['truth'].items() for action, values in case['actions'].items()}


def source_index(args):
    train150, train600 = read_json(args.train150_manifest), read_json(args.train600_manifest)
    if train150.get('schema') != 'thermal_response_families_v1' or train600.get('schema') != 'thermal_response_families_v1':
        raise ValueError('Expected established training family manifests.')
    original_recipe = Path(train600['source_original_recipe'])
    original_ledger = Path(train600['source_original_ledger'])
    ledgers = list(dict.fromkeys([str(original_ledger), *args.ledger]))
    records_by_raw, attempts = ledger_rows(ledgers)
    eligible = {}
    for population, manifest in (('TRAIN150', train150), ('TRAIN600', train600)):
        eligible[population] = {(str(x['case_id']), str(x['action'])) for x in manifest['actions']}
        if len(eligible[population]) != len(manifest['actions']):
            raise ValueError('Duplicate FIT action in population manifest.')
    if not eligible['TRAIN150'] <= eligible['TRAIN600']:
        raise ValueError('TRAIN150 actions must be a subset of the established TRAIN600 actions.')
    families = []
    def add(case, actions, role, source):
        case = copy.deepcopy(case); family = str(case['case_id'])
        case['source'] = source
        if not case.get('source_config'):
            case['source_config'] = read_json(Path(case['source_case_dir']) / 'case_config.json')
        entries = []
        for item in actions:
            item = copy.deepcopy(item); action = str(item['action'])
            item.setdefault('source', source)
            memberships = [name for name, keys in eligible.items() if (family, action) in keys]
            entries.append({'action': action, 'role': role, 'populations': memberships, 'metadata': item})
        if not entries or len({x['action'] for x in entries}) != len(entries):
            raise ValueError(f'Family {family} has missing or duplicate actions.')
        families.append({'family_id': family, 'role': role, 'source': source,
                         'populations': [p for p in eligible if all(p in a['populations'] for a in entries)],
                         'metadata': case, 'actions': entries})
    for case in train600['cases']:
        family = str(case['case_id'])
        add(case, [a for a in train600['actions'] if str(a['case_id']) == family], 'fit', case['source'])
    recipe = read_json(original_recipe)
    originals = {str(c['case_id']): c for c in recipe['cases'] if c['role'] == 'response_holdout' and c.get('feasible')}
    for family in sorted(originals):
        case = originals[family]; actions = []
        for action in ACTIONS:
            found = [(entry.get('started'), entry.get('finished')) for (ledger, _), entry in attempts.items()
                     if ledger == str(original_ledger) and entry.get('started', {}).get('case_id') == family
                     and entry.get('started', {}).get('action') == action
                     and entry.get('started', {}).get('allocation') == 'edited'
                     and entry.get('finished', {}).get('status') == 'converged']
            if len(found) != 1:
                raise ValueError(f'Original withheld action {family}/{action} has {len(found)} bindings.')
            start, end = found[0]
            actions.append({'case_id': family, 'action': action, 'role': 'response_holdout',
                            'source': 'original', 'centers': start['centers'],
                            'raw_case_dir': end['raw_case_dir'], 'record_directory': end.get('record_directory'),
                            'status': end['status'], 'physical_attempt': {'ledger': str(original_ledger), 'attempt': start['attempt']}})
        add(case, actions, 'response_holdout', 'original')
    for name, path, role in (('historical_new', args.historical_manifest, 'historical_new_withheld'),
                              ('native_seen', args.native_seen_manifest, 'native_seen'),
                              ('fresh_layout', args.fresh_manifest, 'fresh_layout')):
        manifest = read_json(path)
        for case in manifest['cases']:
            family = str(case['case_id'])
            actions = [a for a in manifest['actions'] if str(a['case_id']) == family]
            if name == 'historical_new':
                actions = [a for a in actions if a['role'] == 'evaluation_only']
            add(case, actions, role, name)
    if len({x['family_id'] for x in families}) != len(families):
        raise ValueError('Response cohorts contain duplicate family identities.')
    # Historic physical values are separate source evidence, never added to the
    # original action metadata and never embedded in the role-selection index.
    for role, path, reader, origin in (
            ('response_holdout', args.original_holdout_labels, saved_original_labels, 'original_response_evaluation'),
            ('historical_new_withheld', args.historical_new_labels, saved_historical_new_labels, 'historical_candidate_truth')):
        selected = [f for f in families if f['role'] == role]
        if not path:
            if selected and not args.allow_fixture:
                raise ValueError(f'Existing saved physical labels must be supplied for {role}.')
            continue
        labels = reader(path)
        label_source = read_json(path)
        if origin == 'original_response_evaluation':
            source_context = {k: label_source[k] for k in ('recipe', 'ledger', 'partition', 'units', 'masks') if k in label_source}
            extraction = 'physical J/P values, identical in every saved model/access view'
        else:
            source_context = {k: label_source[k] for k in ('schema', 'protocol', 'selection_scope') if k in label_source}
            extraction = 'saved physical candidate truth action delta J/P'
        expected = {(f['family_id'], a['action']) for f in selected for a in f['actions']}
        if set(labels) != expected:
            raise ValueError(f'Historical saved labels do not exactly cover {role}.')
        for family in selected:
            for action in family['actions']:
                action['saved_delta'] = labels[(family['family_id'], action['action'])]
                action['saved_delta_provenance'] = {'source': origin, 'path': str(path),
                                                     'extraction': extraction, 'source_context': source_context}
    for family in families:
        for action in family['actions']:
            saved = action.get('saved_delta', action['metadata'].get('physical_delta'))
            if saved is not None:
                if set(saved) != {'J', 'P'} or any(not math.isfinite(float(saved[k])) for k in ('J', 'P')):
                    raise ValueError('Saved physical delta must contain finite J and P values.')
                if 'saved_delta' not in action:
                    action['saved_delta'] = saved
                    source_path = (args.train600_manifest if family['role'] == 'fit' else
                                   args.native_seen_manifest if family['role'] == 'native_seen' else args.fresh_manifest)
                    action['saved_delta_provenance'] = {'source': 'action_manifest', 'path': str(source_path)}
    # Resolve baseline record bindings only from exact raw paths in saved ledgers.
    for family in families:
        for item, rawkey in [(family['metadata'], 'source_case_dir'), *((a['metadata'], 'raw_case_dir') for a in family['actions'])]:
            raw = str(Path(item[rawkey]).resolve())
            if not item.get('record_directory'):
                candidates = records_by_raw.get(raw, [])
                unique = {r['record_directory'] for r in candidates}
                if len(unique) == 1:
                    item['record_directory'] = unique.pop()
                elif len(unique) > 1:
                    raise ValueError(f'Ambiguous physical record bindings for {raw}.')
    expected = {'fit': (144, 576), 'response_holdout': (12, 48),
                'historical_new_withheld': (24, 144), 'native_seen': (12, 48), 'fresh_layout': (12, 48)}
    if not args.allow_fixture:
        for role, (fc, ac) in expected.items():
            selected = [f for f in families if f['role'] == role]
            if (len(selected), sum(len(f['actions']) for f in selected)) != (fc, ac):
                raise ValueError(f'Cohort {role} does not match its declared complete coverage.')
    return families, train150, train600, ledgers


def endpoint_sources(families):
    """Build stable identity from exact raw and record bindings, never lossy dedup."""
    sources, keys = [], {}
    for fidx, family in enumerate(families):
        family['key'] = f'f{fidx:04d}'
        items = [(family['metadata'], None)] + [(a['metadata'], a) for a in family['actions']]
        for item, action in items:
            raw = str(Path(item['source_case_dir' if action is None else 'raw_case_dir']).resolve())
            rec = item.get('record_directory')
            rec = str(Path(rec).resolve()) if rec else None
            identity = (raw, rec)
            if identity not in keys:
                keys[identity] = f'e{len(keys):05d}'
                prescribed = copy.deepcopy(family['metadata']['source_config'])
                if action is not None:
                    prescribed['layout']['centers'] = copy.deepcopy(item['centers'])
                sources.append({'key': keys[identity], 'raw_case_dir': raw, 'record_directory': rec,
                                'prescribed': prescribed, 'saved_endpoint_metadata': item,
                                'training_required': family['role'] == 'fit'})
            elif family['role'] == 'fit':
                next(source for source in sources if source['key'] == keys[identity])['training_required'] = True
            if action is None:
                family['baseline_endpoint'] = keys[identity]
            else:
                action['endpoint'] = keys[identity]
                action['key'] = f'a{len([a for a in family["actions"] if "key" in a]):04d}'
    return sources


def reference_sources(sources, args):
    existing_raw = {s['raw_case_dir']: s['key'] for s in sources}
    panels = read_json(args.panels)
    references = []
    for partition, ids in (('train', panels['native_case_ids']), ('validation', panels['validation_case_ids'])):
        for native_id in map(str, ids):
            matches = list(Path(args.raw_native_root, 'train' if partition == 'train' else 'test').glob(f'case_{native_id}_*'))
            if len(matches) != 1:
                raise ValueError(f'Cannot uniquely bind H reference baseline {native_id}.')
            raw = str(matches[0].resolve()); endpoint = existing_raw.get(raw)
            if endpoint is None:
                endpoint = f'e{len(sources):05d}'
                sources.append({'key': endpoint, 'raw_case_dir': raw, 'record_directory': None,
                                'prescribed': read_json(Path(raw) / 'case_config.json'),
                                'saved_endpoint_metadata': {'reference_only': True}, 'training_required': False})
                existing_raw[raw] = endpoint
            references.append({'native_case_id': native_id, 'partition': partition,
                               'role': 'reference_only', 'endpoint': endpoint})
    if len({r['native_case_id'] for r in references}) != len(references):
        raise ValueError('Reference panel contains duplicate case IDs.')
    return references


def raw_parts(path):
    path = Path(path)
    config_text = (path / 'case_config.json').read_bytes()
    config = json.loads(config_text)
    csv_text = (path / 'frame_index.csv').read_bytes()
    rows = list(csv.DictReader(csv_text.decode().splitlines()))
    if not rows:
        raise ValueError(f'No frames for {path}')
    frame_path = path / 'scene' / rows[-1]['file']
    with np.load(frame_path, allow_pickle=False) as z:
        frame = {key: z[key].copy() for key in z.files}
    with np.load(path / 'grid.npz', allow_pickle=False) as z:
        grid = {key: z[key].copy() for key in z.files}
    return config_text, config, csv_text, rows, frame, grid, frame_path


def convergence(rows, config):
    """Audit the recorded saved window and retain failures, without inventing evidence."""
    thermal = config['thermal']; window = int(thermal.get('convergence_window', 0))
    result = {'window': window, 'status': 'unavailable'}
    try:
        tail = rows[-window:]
        if window < 1 or len(tail) != window:
            raise ValueError('saved window is incomplete')
        if any(not math.isfinite(float(r['delta_inf'])) or not math.isfinite(float(r['delta_l2_rel'])) for r in tail):
            raise ValueError('saved window contains nonfinite convergence metrics')
        result.update(max_delta_inf=max(float(r['delta_inf']) for r in tail),
                      max_delta_l2_rel=max(float(r['delta_l2_rel']) for r in tail),
                      first_pass_step=int(rows[-1]['step']))
        if (rows[-1]['converged'] != '1' or any(r['converged'] == '1' for r in rows[:-1])
                or rows[-1]['heat_active'] != '1'
                or result['max_delta_inf'] > float(thermal['convergence_tol'])
                or result['max_delta_l2_rel'] > float(thermal['convergence_rel_tol'])):
            raise ValueError('saved first-pass window does not satisfy declared policy')
        result['status'] = 'passed'
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        result.update(status='failed_or_unavailable', reason=str(exc))
    return result


def physical_functionals(raw):
    config = raw['config']; d = config['domain']; tau = TAU
    material = raw['material'][:, raw['material_mask']].astype(np.float64)
    values = material[np.isfinite(material)]
    mask = raw['fluid_mask'] & np.isfinite(raw['field'][..., 2])
    x = (np.arange(int(d['nx']), dtype=np.float64) + .5) * float(d['lx']) / int(d['nx'])
    left = mask & (x[None, :] <= .08 * float(d['lx']))
    right = mask & (x[None, :] >= .92 * float(d['lx']))
    if not len(values) or not left.any() or not right.any():
        raise ValueError('No physical functional support.')
    # Match the documented normalized log-sum-exp and FP64 band means.
    maximum = np.max(values / tau)
    j = tau * (maximum + np.log(np.exp(values / tau - maximum).sum()) - np.log(len(values)))
    p = raw['field'][..., 2].astype(np.float64)
    return {'J': float(j), 'P': float(p[left].mean() - p[right].mean())}


def verify_record_binding(metadata, source):
    if metadata.get('status') != 'converged' or metadata.get('output') is None:
        raise ValueError('A bound physical record is not a converged observation.')
    record_path = metadata['output'].get('case_dir')
    if not record_path or Path(record_path).resolve() != Path(source['raw_case_dir']).resolve():
        raise ValueError('Physical record and raw endpoint paths do not identify the same solve.')
    modules = metadata['design']['modules']
    layout = source['prescribed']['layout']
    if ([m['position_xy'] for m in modules] != layout['centers']
            or [m['heating'] for m in modules] != layout['heat_powers']
            or not all(m['active'] for m in modules)):
        raise ValueError('Physical record member order, positions or heat differ from the endpoint.')


def verify_controls(source, actual):
    from channelthermal.physical_provenance import validate_thermal_endpoint_config
    saved = source['saved_endpoint_metadata']
    validate_thermal_endpoint_config(source['prescribed'], actual,
        centers=source['prescribed']['layout']['centers'],
        expected_actual_config=saved.get('actual_solver_config'), numerical_policy=saved.get('numerical_policy'))


def census(families, sources):
    totals = {'families': len(families), 'actions': sum(len(f['actions']) for f in families),
              'endpoints': len(sources), 'raw_archive_uncompressed_bytes': 0,
              'record_archive_uncompressed_bytes': 0, 'record_metadata_bytes': 0,
              'missing_records': [], 'cohorts': {}}
    for f in families:
        c = totals['cohorts'].setdefault(f['role'], {'families': 0, 'actions': 0, 'saved_deltas': 0})
        c['families'] += 1; c['actions'] += len(f['actions'])
        c['saved_deltas'] += sum('saved_delta' in a for a in f['actions'])
    for source in sources:
        path = Path(source['raw_case_dir'])
        with (path / 'frame_index.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for archive in (path / 'grid.npz', path / 'scene' / rows[-1]['file']):
            with zipfile.ZipFile(archive) as z:
                totals['raw_archive_uncompressed_bytes'] += sum(x.file_size for x in z.infolist())
        record = source['record_directory']
        if record:
            record = Path(record)
            totals['record_metadata_bytes'] += (record / 'record.json').stat().st_size
            archive = record / 'physical_output.npz'
            if archive.exists():
                with zipfile.ZipFile(archive) as z:
                    totals['record_archive_uncompressed_bytes'] += sum(x.file_size for x in z.infolist())
        else:
            totals['missing_records'].append(source['key'])
    return totals


def pack_endpoint(group, source):
    config_text, actual, csv_text, rows, frame, grid, frame_path = raw_parts(source['raw_case_dir'])
    prescribed = source['prescribed']
    if (actual['layout']['centers'] != prescribed['layout']['centers']
            or actual['layout']['heat_powers'] != prescribed['layout']['heat_powers']):
        raise ValueError(f'Prescribed member controls differ for {source["raw_case_dir"]}.')
    verify_controls(source, actual)
    cfg = group.create_group('configuration')
    put_bytes(cfg, 'actual_json', config_text); put_json(cfg, 'prescribed_json', prescribed)
    put_array(cfg, 'centers_source_float64', np.asarray(prescribed['layout']['centers'], dtype=np.float64))
    put_array(cfg, 'heat_source_float64', np.asarray(prescribed['layout']['heat_powers'], dtype=np.float64))
    raw = group.create_group('raw_frame'); arrays = raw.create_group('arrays'); grids = raw.create_group('grid')
    for name, data in frame.items(): put_array(arrays, name, data)
    for name, data in grid.items(): put_array(grids, name, data)
    put_bytes(raw, 'frame_index_csv', csv_text)
    provenance = {'raw_case_dir': source['raw_case_dir'], 'frame_file': str(frame_path),
                  'record_directory': source['record_directory'], 'convergence': convergence(rows, actual),
                  'saved_endpoint_metadata': source['saved_endpoint_metadata']}
    put_json(group, 'provenance_json', provenance)
    views = group.create_group('effective_views')
    put_json(views, 'r17_1_raw_json', {'implementation': 'channelthermal.data.finite_responses.raw_endpoint_from_arrays',
            'field': 'raw_frame/arrays/{u,v,p,omega,temperature} in saved dtype',
            'fluid_mask': 'logical not raw_frame/arrays/module_mask',
            'material': 'raw_frame/arrays/module_internal_temperature with module_internal_mask',
            'interface': 'T_surface,q_normal from interface_response by interface_feature_names',
            'outside_valid': 'source-float64 centers and cell-center stencil bounds',
            'teacher_h_valid': 'FP32 q/(T_surface-T_outside), epsilon 1e-3, finite and [0,1e4]',
            'legacy_original': 'native baseline catalogue plus edited observation record; outside ports use raw frame'})
    record = source['record_directory']
    if record:
        record = Path(record); target = group.create_group('observation_record')
        record_text = (record / 'record.json').read_bytes()
        verify_record_binding(json.loads(record_text), source)
        put_bytes(target, 'metadata_json', record_text)
        arrays = target.create_group('arrays'); archive = record / 'physical_output.npz'
        if archive.exists():
            with np.load(archive, allow_pickle=False) as z:
                for name in z.files: put_array(arrays, name, z[name])
    effective = raw_endpoint_from_arrays(actual, frame, raw_case_dir=source['raw_case_dir'])
    functionals = physical_functionals(effective)
    put_json(views, 'derived_functionals_json', functionals)
    return functionals, provenance['convergence']


def write_pack(path, families, sources, references, train150, train600, args, summary):
    with h5py.File(args.native_h5, 'r') as native, h5py.File(path, 'x', track_order=True) as out:
        out.attrs.update(schema=SCHEMA, complete=False, created_unix=time.time())
        meta = out.create_group('meta'); out.create_group('index'); groups = out.create_group('families')
        endpoints = out.create_group('endpoints'); catalogues = out.create_group('catalogues')
        populations = out.create_group('populations'); refs = out.create_group('reference_baselines')
        native_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in native['case_ids'][...]]
        splits = [x.decode() if isinstance(x, bytes) else str(x) for x in native['splits'][...]]
        split_by_id = dict(zip(native_ids, splits))
        put_json(meta, 'native_identity_json', {'path': str(Path(args.native_h5).resolve()),
                 'size_bytes': Path(args.native_h5).stat().st_size, 'dataset_manifest': read_json(args.dataset_manifest),
                 'attrs': {key: value.item() if isinstance(value, np.generic) else value for key, value in native.attrs.items()},
                 'case_ids': native_ids, 'splits': splits})
        for name in ('channel_order', 'interface_feature_names', 'interface_condition_feature_names',
                     'interface_target_names', 'sampled_point_feature_names'):
            native.copy(name, meta)
        native.copy('normalization', meta, name='native_source_normalization')
        put_json(meta, 'physical_conventions_json', {
            'field_channels': ['u', 'v', 'p', 'omega', 'temperature'],
            'field_units': ['dataset velocity units', 'dataset velocity units', 'dataset pressure units',
                            'dataset vorticity units', 'dataset temperature units'],
            'position_units': 'dataset length units', 'surface_temperature_units': 'dataset temperature units',
            'q_normal_units': 'dataset heat-flux proxy units; no SI metadata identified',
            'q_normal_semantics': 'source q proxy, not an independently solved conservative interface flux',
            'fluid_coordinates': 'saved cell-center grid, original array orientation',
            'solid_coordinates': 'original-member attached normalized local-mask order',
            'interface_coordinates': 'original-member attached 64-angle source ordering',
            'outside_coordinates': 'source-FP64 centers plus radius and bounded stencil offset',
            'masks': 'endpoint fluid, material, outside and teacher-h validity are distinct'})
        for family in families:
            family['native_case_id'] = family['family_id'] if family['family_id'] in split_by_id else None
            family['native_split'] = split_by_id.get(family['family_id'])
        # One native baseline group per mapped family preserves historical arrays and catalogue order/mass.
        for family in families:
            native_id = family['native_case_id']
            if native_id and native_id not in catalogues:
                case = native['cases'][native_id]
                layout = family['metadata']['source_config']['layout']
                count = len(layout['centers'])
                if (not np.array_equal(case['module_centers'][:count], np.asarray(layout['centers'], dtype=np.float32))
                        or not np.array_equal(case['heat_powers'][:count], np.asarray(layout['heat_powers'], dtype=np.float32))
                        or not np.array_equal(np.flatnonzero(case['module_present'][...] > .5), np.arange(count))):
                    raise ValueError(f'Native family mapping disagrees with prescribed controls: {native_id}.')
                native.copy(native['cases'][native_id], catalogues, name=native_id)
        train150_native = read_json(args.native_manifest)['partitions']['train']['case_ids']
        for name, manifest, native_members in (('TRAIN150', train150, train150_native),
                                               ('TRAIN600', train600, [i for i, s in zip(native_ids, splits) if s == 'train'])):
            pop = populations.create_group(name)
            calibration_path = args.calibration150 if name == 'TRAIN150' else args.calibration600
            calibration = read_json(calibration_path) if calibration_path else None
            metadata = {'native_case_ids': native_members, 'eligible_fit_families': [str(c['case_id']) for c in manifest['cases']],
                        'eligible_fit_actions': [[str(a['case_id']), a['action']] for a in manifest['actions']],
                        'native_normalization': None if calibration is None else calibration.get('adapter_state', {}).get('normalization_stats'),
                        'native_normalization_provenance': None if calibration is None else {'path': str(calibration_path), 'source': calibration.get('normalization_source')},
                        'historical_response_scales': None if calibration is None else calibration.get('response_scales'),
                        'historical_response_scale_provenance': None if calibration is None else str(calibration_path),
                        'new_objective_coefficients': 'pending epoch-10 calibration'}
            put_json(pop, 'metadata_json', metadata)
        functionals = {}; convergence_results = {}
        for source in sources:
            group = endpoints.create_group(source['key'])
            functionals[source['key']], convergence_results[source['key']] = pack_endpoint(group, source)
            if out.id.get_filesize() > args.maximum_bytes:
                raise ValueError('Shared H5 exceeds the declared disk budget.')
        for family in families:
            group = groups.create_group(family['key']); put_json(group, 'metadata_json', family['metadata'])
            actions = group.create_group('actions'); base = functionals[family['baseline_endpoint']]
            for item in family['actions']:
                action = actions.create_group(item['key']); put_json(action, 'metadata_json', item['metadata'])
                saved = item.get('saved_delta')
                if saved is not None:
                    put_json(action, 'saved_delta_json', saved)
                    put_json(action, 'saved_delta_provenance_json', item['saved_delta_provenance'])
                current = functionals[item['endpoint']]
                put_json(action, 'derived_delta_json', {k: current[k] - base[k] for k in ('J', 'P')})
            put_json(group, 'functional_definition_json', {'tau': TAU, 'J': 'tau * (logsumexp(material/tau) - log(N))',
                'P': 'mean pressure for x <= 0.08 lx minus mean pressure for x >= 0.92 lx',
                'effective_view': 'r17_1_raw', 'reduction': 'FP64 over finite own-endpoint physical support'})
        # Reference-only cases are endpoint data, not training-family additions.
        for reference in references:
            native_id = reference['native_case_id']
            group = refs.create_group(native_id)
            put_json(group, 'metadata_json', reference)
            if native_id not in catalogues:
                native.copy(native['cases'][native_id], catalogues, name=native_id)
        serializable = []
        for family in families:
            serializable.append({k: v for k, v in family.items() if k != 'metadata' and k != 'actions'} |
                                {'actions': [{k: v for k, v in a.items() if k not in ('metadata', 'saved_delta', 'saved_delta_provenance')}
                                             for a in family['actions']]})
        put_json(out['index'], 'families_json', serializable)
        summary['convergence'] = {name: sum(v['status'] == name for v in convergence_results.values())
                                  for name in sorted({v['status'] for v in convergence_results.values()})}
        summary['training_convergence_failures'] = [source['key'] for source in sources
            if source['training_required'] and convergence_results[source['key']]['status'] != 'passed']
        summary['reference_baselines'] = len(refs)
        put_json(meta, 'coverage_json', summary)
        put_json(meta, 'creation_json', {'packer_revision': args.packer_revision, 'created_unix': time.time(),
                 'source_manifests': [str(args.train150_manifest), str(args.train600_manifest), str(args.historical_manifest),
                                      str(args.native_seen_manifest), str(args.fresh_manifest),
                                      train600['source_original_recipe'], train600['source_new_manifest']],
                 'historical_label_sources': [str(p) for p in (args.original_holdout_labels, args.historical_new_labels) if p],
                 'ledgers': [train600['source_original_ledger'], *args.ledger],
                 'observation_missing': 'An absent observation_record means no exact compatible saved binding was found.',
                 'reference_scope': 'TRAIN15 and VAL9, input-selected; no TEST labels.'})
        out.attrs['complete'] = True
        out.flush()


def arrays_equal(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right, equal_nan=True) if left.dtype.kind in 'fc' else left.dtype == right.dtype and np.array_equal(left, right)


def audit(path, families, sources, references, args):
    counts = {'endpoints': 0, 'frame_arrays': 0, 'grid_arrays': 0, 'record_arrays': 0,
              'record_metadata': 0, 'families': 0, 'actions': 0, 'native_catalogues': 0,
              'native_arrays': 0, 'reference_baselines': 0, 'populations': 0}
    functionals = {}
    def equal_json(dataset, value):
        return json.loads(dataset[...].tobytes()) == value
    def compare_group(source, copied):
        if set(source) != set(copied) or set(source.attrs) != set(copied.attrs):
            raise ValueError(f'Native catalogue group differs: {source.name}.')
        for key in source.attrs:
            if not np.array_equal(source.attrs[key], copied.attrs[key]):
                raise ValueError(f'Native catalogue attribute differs: {source.name}/{key}.')
        for key in source:
            a, b = source[key], copied[key]
            if isinstance(a, h5py.Group):
                compare_group(a, b)
            elif not arrays_equal(np.asarray(a[()]), np.asarray(b[()])):
                raise ValueError(f'Native catalogue array differs: {a.name}.')
            else:
                counts['native_arrays'] += 1
    with h5py.File(path, 'r') as h5, h5py.File(args.native_h5, 'r') as native:
        if h5.attrs.get('schema') != SCHEMA or not h5.attrs.get('complete'):
            raise ValueError('Pack is not closed and complete.')
        index = json.loads(h5['index/families_json'][...].tobytes())
        expected_index = [{k: v for k, v in family.items() if k not in ('metadata', 'actions')} |
                          {'actions': [{k: v for k, v in a.items() if k not in ('metadata', 'saved_delta', 'saved_delta_provenance')}
                                       for a in family['actions']]}
                          for family in families]
        if index != expected_index:
            raise ValueError('Family/action order, roles, memberships or endpoint mappings differ.')
        def check_links(group):
            for name in group:
                if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                    raise ValueError('Packed dataset contains an external or symbolic HDF5 link.')
                if isinstance(group[name], h5py.Group):
                    check_links(group[name])
        check_links(h5)
        for source in sources:
            group = h5['endpoints'][source['key']]
            cfg, actual, csv_bytes, rows, frame, grid, frame_path = raw_parts(source['raw_case_dir'])
            if group['configuration/actual_json'][...].tobytes() != cfg or group['raw_frame/frame_index_csv'][...].tobytes() != csv_bytes:
                raise ValueError(f'Config or frame history differs: {source["key"]}')
            if not equal_json(group['configuration/prescribed_json'], source['prescribed']):
                raise ValueError('Prescribed source controls differ.')
            layout = source['prescribed']['layout']
            for name, values in (('centers_source_float64', layout['centers']),
                                 ('heat_source_float64', layout['heat_powers'])):
                if not arrays_equal(group[f'configuration/{name}'][...], np.asarray(values, dtype=np.float64)):
                    raise ValueError('Typed source-precision member controls differ.')
            expected_provenance = {'raw_case_dir': source['raw_case_dir'], 'frame_file': str(frame_path),
                'record_directory': source['record_directory'], 'convergence': convergence(rows, actual),
                'saved_endpoint_metadata': source['saved_endpoint_metadata']}
            if not equal_json(group['provenance_json'], expected_provenance):
                raise ValueError('Endpoint convergence or source provenance differs.')
            if set(group['raw_frame/arrays']) != set(frame) or set(group['raw_frame/grid']) != set(grid):
                raise ValueError('Raw frame or grid array keys differ.')
            for name, expected in frame.items():
                if not arrays_equal(group['raw_frame/arrays'][name][...], expected): raise ValueError(f'Frame array differs: {source["key"]}/{name}')
                counts['frame_arrays'] += 1
            for name, expected in grid.items():
                if not arrays_equal(group['raw_frame/grid'][name][...], expected): raise ValueError(f'Grid differs: {source["key"]}/{name}')
                counts['grid_arrays'] += 1
            if source['record_directory']:
                record = Path(source['record_directory']); g = group['observation_record']
                if g['metadata_json'][...].tobytes() != (record / 'record.json').read_bytes(): raise ValueError('Record metadata differs.')
                counts['record_metadata'] += 1
                archive = record / 'physical_output.npz'
                if archive.exists():
                    with np.load(archive, allow_pickle=False) as z:
                        if set(g['arrays']) != set(z.files): raise ValueError('Record array keys differ.')
                        for name in z.files:
                            if not arrays_equal(g['arrays'][name][...], z[name]): raise ValueError(f'Record array differs: {source["key"]}/{name}')
                            counts['record_arrays'] += 1
            elif 'observation_record' in group:
                raise ValueError('Unexpected record substitution.')
            functionals[source['key']] = physical_functionals(raw_endpoint_from_arrays(actual, frame, raw_case_dir=source['raw_case_dir']))
            if not equal_json(group['effective_views/derived_functionals_json'], functionals[source['key']]):
                raise ValueError('Derived FP64 functional differs.')
            counts['endpoints'] += 1
        for family in families:
            group = h5['families'][family['key']]
            if not equal_json(group['metadata_json'], family['metadata']):
                raise ValueError('Family source metadata differs.')
            for item in family['actions']:
                action = group['actions'][item['key']]
                if not equal_json(action['metadata_json'], item['metadata']):
                    raise ValueError('Action source metadata differs.')
                saved = item.get('saved_delta')
                if ('saved_delta_json' in action) != (saved is not None):
                    raise ValueError('Saved delta availability differs.')
                if saved is not None and not equal_json(action['saved_delta_json'], saved):
                    raise ValueError('Saved physical delta differs.')
                if saved is not None and not equal_json(action['saved_delta_provenance_json'], item['saved_delta_provenance']):
                    raise ValueError('Saved physical delta provenance differs.')
                base, edited = functionals[family['baseline_endpoint']], functionals[item['endpoint']]
                if not equal_json(action['derived_delta_json'], {k: edited[k] - base[k] for k in ('J', 'P')}):
                    raise ValueError('Derived physical delta differs.')
                counts['actions'] += 1
            counts['families'] += 1
        native_ids = {family['native_case_id'] for family in families if family['native_case_id'] is not None}
        native_ids.update(ref['native_case_id'] for ref in references)
        if set(h5['catalogues']) != native_ids:
            raise ValueError('Native catalogue coverage differs.')
        for native_id in native_ids:
            compare_group(native['cases'][native_id], h5['catalogues'][native_id]); counts['native_catalogues'] += 1
        compare_group(native['normalization'], h5['meta/native_source_normalization'])
        for reference in references:
            if not equal_json(h5[f"reference_baselines/{reference['native_case_id']}/metadata_json"], reference):
                raise ValueError('Reference-only baseline mapping differs.')
            counts['reference_baselines'] += 1
        for name, calibration_path in (('TRAIN150', args.calibration150), ('TRAIN600', args.calibration600)):
            pop = json.loads(h5[f'populations/{name}/metadata_json'][...].tobytes())
            manifest = read_json(args.train150_manifest if name == 'TRAIN150' else args.train600_manifest)
            if name == 'TRAIN150':
                expected_native = read_json(args.native_manifest)['partitions']['train']['case_ids']
            else:
                ids = [x.decode() if isinstance(x, bytes) else str(x) for x in native['case_ids'][...]]
                splits = [x.decode() if isinstance(x, bytes) else str(x) for x in native['splits'][...]]
                expected_native = [i for i, split in zip(ids, splits) if split == 'train']
            if (pop['native_case_ids'] != expected_native
                    or pop['eligible_fit_families'] != [str(c['case_id']) for c in manifest['cases']]
                    or pop['eligible_fit_actions'] != [[str(a['case_id']), a['action']] for a in manifest['actions']]):
                raise ValueError('Population native membership or ordered FIT family/action membership differs.')
            if calibration_path:
                original = read_json(calibration_path)
                if (pop['native_normalization'] != original['adapter_state']['normalization_stats']
                        or pop['historical_response_scales'] != original['response_scales']):
                    raise ValueError('Population normalization or historical response scales differ.')
            counts['populations'] += 1
    return counts


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ('train150-manifest', 'train600-manifest', 'historical-manifest', 'native-seen-manifest',
                   'fresh-manifest', 'native-h5', 'dataset-manifest', 'native-manifest', 'panels', 'raw-native-root'):
        p.add_argument('--' + option, required=True, type=Path)
    p.add_argument('--ledger', action='append', default=[], help='Additional saved attempt ledger; repeat as needed.')
    p.add_argument('--calibration150', type=Path); p.add_argument('--calibration600', type=Path)
    p.add_argument('--original-holdout-labels', type=Path)
    p.add_argument('--historical-new-labels', type=Path)
    p.add_argument('--destination', type=Path); p.add_argument('--receipt', type=Path)
    p.add_argument('--packer-revision', required=True); p.add_argument('--maximum-bytes', type=int, required=True)
    p.add_argument('--minimum-free-bytes', type=int, required=True)
    p.add_argument('--inventory-only', action='store_true'); p.add_argument('--audit', action='store_true')
    p.add_argument('--convergence-only', action='store_true', help='Audit all source convergence windows without creating output.')
    p.add_argument('--binding-only', action='store_true', help='Check every physical-record/raw/config binding before writing.')
    p.add_argument('--controls-only', action='store_true', help='Check every actual endpoint against its complete prescribed controls.')
    p.add_argument('--allow-fixture', action='store_true', help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    families, train150, train600, ledgers = source_index(args)
    sources = endpoint_sources(families)
    references = reference_sources(sources, args)
    summary = census(families, sources)
    if args.controls_only:
        failures = []
        for source in sources:
            try:
                verify_controls(source, read_json(Path(source['raw_case_dir']) / 'case_config.json'))
            except (KeyError, TypeError, ValueError) as exc:
                failures.append({'endpoint': source['key'], 'reason': str(exc)})
        print(json.dumps({'checked': len(sources), 'failures': failures}, indent=2), flush=True)
        return
    if args.binding_only:
        failures, checked = [], 0
        for source in sources:
            if source['record_directory']:
                try:
                    verify_record_binding(read_json(Path(source['record_directory']) / 'record.json'), source)
                    checked += 1
                except (KeyError, TypeError, ValueError) as exc:
                    failures.append({'endpoint': source['key'], 'reason': str(exc)})
        print(json.dumps({'checked': checked, 'failures': failures}, indent=2), flush=True)
        return
    if args.convergence_only:
        failures = []
        for source in sources:
            raw = Path(source['raw_case_dir'])
            with (raw / 'frame_index.csv').open(newline='') as stream:
                result = convergence(list(csv.DictReader(stream)), read_json(raw / 'case_config.json'))
            if result['status'] != 'passed': failures.append({'endpoint': source['key'], 'raw': str(raw), **result})
        print(json.dumps({'checked': len(sources), 'failures': failures}, indent=2), flush=True)
        return
    print(json.dumps(summary, indent=2), flush=True)
    if args.inventory_only: return
    if args.destination is None or args.receipt is None:
        raise ValueError('Build needs --destination and --receipt.')
    if not args.audit:
        raise ValueError('Publication requires --audit.')
    if args.destination.exists() or args.receipt.exists():
        raise FileExistsError('Destination or receipt exists; choose a non-clobbering path.')
    if args.destination.parent.stat().st_dev != args.receipt.parent.stat().st_dev:
        raise ValueError('Destination and receipt must be on the prepared DATA2 filesystem.')
    if shutil.disk_usage(args.destination.parent).free - args.maximum_bytes < args.minimum_free_bytes:
        raise ValueError('Insufficient declared DATA2 headroom for the H5 budget.')
    temporary = args.destination.with_name(args.destination.name + f'.building-{os.getpid()}')
    if temporary.exists(): raise FileExistsError(temporary)
    write_pack(temporary, families, sources, references, train150, train600, args, summary)
    if temporary.stat().st_size > args.maximum_bytes: raise ValueError('Packed file exceeds its budget.')
    summary['source_audit'] = audit(temporary, families, sources, references, args)
    summary.update(destination=str(args.destination), size_bytes=temporary.stat().st_size,
                   data2_free_after_bytes=shutil.disk_usage(args.destination.parent).free,
                   source_ledgers=ledgers)
    if summary['training_convergence_failures']:
        raise ValueError('A FIT endpoint lacks passing source convergence evidence; no H5 was published.')
    with open(temporary, 'rb') as stream: os.fsync(stream.fileno())
    os.link(temporary, args.destination)  # Atomic, no-clobber publication.
    temporary.unlink()
    args.receipt.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'published': str(args.destination), 'audit': summary['source_audit'],
                      'bytes': summary['size_bytes']}, indent=2))


if __name__ == '__main__':
    main()
