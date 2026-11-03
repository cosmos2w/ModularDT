"""Read R15-8 raw physical endpoints without touching held-out calibration."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_raw_endpoint(path):
    """Read the converged final frame, complete native receivers and P64 teacher.

    Values retain their saved FP32 precision. The outside stencil's geometric
    validity is separate from the preprocessing-derived h-valid mask.
    """
    path = Path(path)
    config = json.loads((path / 'case_config.json').read_text())
    with (path / 'frame_index.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or int(float(rows[-1]['converged'])) != 1:
        raise ValueError('A converged final physical frame is required')
    with np.load(path / 'scene' / rows[-1]['file'], allow_pickle=False) as frame:
        raw = {key: frame[key].copy() for key in frame.files}
    field = np.stack([raw[k] for k in ('u', 'v', 'p', 'omega', 'temperature')], axis=-1)
    response = raw['interface_response']
    names = [s.decode() if isinstance(s, bytes) else str(s) for s in raw['interface_feature_names']]
    indices = [names.index(s) for s in ('theta', 'normal_x', 'normal_y', 'T_outside', 'T_surface', 'q_normal')]
    theta, nxv, nyv, outside, surface, q = (response[..., i] for i in indices)
    difference = surface - outside
    denominator = np.where(np.abs(difference) < 1.e-3, np.where(difference < 0, -1.e-3, 1.e-3), difference).astype(np.float32)
    h = q / denominator
    h_valid = (np.abs(difference) >= 1.e-3) & np.isfinite(q) & np.isfinite(h) & (h >= 0) & (h <= 1.e4)
    teacher = np.stack((theta, nxv, nyv, outside, np.clip(np.nan_to_num(h, nan=0., posinf=1.e4, neginf=0.), 0., 1.e4)), -1).astype(np.float32)
    d = config['domain']; centers = np.asarray(config['layout']['centers'], dtype=np.float64)
    radius = float(d['module_radius']); delta = min(d['lx']/d['nx'], d['ly']/d['ny'], .15*radius)
    angle = np.arange(response.shape[1], dtype=np.float64) * 2*np.pi / response.shape[1]
    xy = centers[:, None, :] + (radius + delta) * np.stack([np.cos(angle), np.sin(angle)], -1)[None]
    outside_valid = ((xy[..., 0] >= .5*d['lx']/d['nx']) & (xy[..., 0] <= d['lx']-.5*d['lx']/d['nx']) &
                     (xy[..., 1] >= .5*d['ly']/d['ny']) & (xy[..., 1] <= d['ly']-.5*d['ly']/d['ny']))
    return {'raw_case_dir': str(path), 'config': config, 'field': field, 'fluid_mask': ~raw['module_mask'].astype(bool),
            'module_id_grid': raw['module_id'], 'material': raw['module_internal_temperature'],
            'material_mask': raw['module_internal_mask'].astype(bool), 'interface_target': np.stack((surface, q), -1),
            'teacher_ports': teacher, 'teacher_h_valid': h_valid, 'outside_valid': outside_valid}


def fitting_cases(recipe_path):
    """Return only fitting entries; never inspect holdout outcomes for calibration."""
    recipe = json.loads(Path(recipe_path).read_text())
    return tuple(case for case in recipe['cases'] if case['role'] == 'fit' and case['feasible'])


def _functionals(endpoint, tau):
    values = np.asarray(endpoint['material'][:, endpoint['material_mask']], dtype=np.float64).ravel()
    x = values / float(tau)
    m = x.max()
    objective = float(tau * (m + np.log(np.exp(x-m).sum()) - np.log(len(x))))
    ny, nx = endpoint['field'].shape[:2]
    length = float(endpoint['config']['domain']['lx'])
    coords = np.broadcast_to((np.arange(nx) + .5) * length / nx, (ny, nx))
    fluid = endpoint['fluid_mask']
    pressure = endpoint['field'][..., 2].astype(np.float64)
    left = fluid & (coords <= .08 * length); right = fluid & (coords >= .92 * length)
    if not left.any() or not right.any():
        raise ValueError('Physical pressure bands have no fluid receivers')
    return objective, float(pressure[left].mean() - pressure[right].mean())


def fit_response_scales(recipe_path, ledger_path, absolute_scales, resolution_floors, tau):
    """Equal-family RMS by channel and action class using fitting labels only.

    Missing/unconverged actions remain absent and are counted. The caller must
    determine fixed TRAIN-only observation/convergence resolution floors first.
    """
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('A positive inherited material smoothing scale is required')
    names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid',
             'temperature/surface', 'temperature/material', 'q_proxy', 'J', 'P')
    if set(names) - set(absolute_scales) or set(names) - set(resolution_floors):
        raise ValueError('Every response channel needs an absolute scale and TRAIN resolution floor')
    rows = [json.loads(s) for s in Path(ledger_path).read_text().splitlines()]
    starts = {r['attempt']: r for r in rows if r['event'] == 'started' and r['allocation'] == 'edited'}
    records = {(starts[r['attempt']]['case_id'], starts[r['attempt']]['action']): r
               for r in rows if r['event'] == 'finished' and r['attempt'] in starts}
    families = {'single': {n: [] for n in names}, 'joint': {n: [] for n in names}}
    retained = {'single': 0, 'joint': 0}
    missing = []
    for case in fitting_cases(recipe_path):
        base = load_raw_endpoint(case['source_case_dir'])
        base_jp = _functionals(base, tau)
        for kind in ('single', 'joint'):
            sums = {n: [] for n in names}
            for sign in ('plus', 'minus'):
                action = f'{kind}_{sign}'; finished = records.get((case['case_id'], action))
                if not finished or finished['status'] != 'converged' or finished['serialization_status'] != 'stored':
                    missing.append({'case_id': case['case_id'], 'action': action})
                    continue
                trial = load_raw_endpoint(finished['raw_case_dir'])
                common = base['fluid_mask'] & trial['fluid_mask']
                if not common.any() or not np.array_equal(base['material_mask'], trial['material_mask']):
                    raise ValueError('Physical paired receiver alignment failed')
                delta = trial['field'].astype(np.float64) - base['field'].astype(np.float64)
                for channel, name in enumerate(names[:5]):
                    sums[name].append(float(np.mean(delta[..., channel][common]**2)))
                for name, channel in (('temperature/surface', 0), ('q_proxy', 1)):
                    mask = base['outside_valid'] & trial['outside_valid'] if channel else np.ones_like(base['outside_valid'], dtype=bool)
                    change = trial['interface_target'][..., channel].astype(np.float64) - base['interface_target'][..., channel].astype(np.float64)
                    sums[name].append(float(np.mean(change[mask]**2)))
                change = trial['material'].astype(np.float64)-base['material'].astype(np.float64)
                sums['temperature/material'].append(float(np.mean(change[:, base['material_mask']]**2)))
                trial_jp = _functionals(trial, tau)
                for name, a, b in zip(('J', 'P'), base_jp, trial_jp):
                    sums[name].append((b-a)**2)
            if sums['J']:
                retained[kind] += 1
                for name in names:
                    families[kind][name].append(float(np.mean(sums[name])))
    scales = {}
    for kind in ('single', 'joint'):
        scales[kind] = {}
        for name in names:
            values = families[kind][name]
            if not values:
                raise ValueError(f'No fitting response data for {kind}/{name}')
            floor = max(.01 * float(absolute_scales[name]), float(resolution_floors[name]))
            if not np.isfinite(floor) or floor <= 0:
                raise ValueError(f'Invalid response floor for {name}')
            scales[kind][name] = max(float(np.sqrt(np.mean(values))), floor)
    return {'scale_by_action_class': scales, 'families_with_actions': retained, 'missing_actions': missing,
            'definition': 'equal-family RMS of signed actions; max with 1% absolute and TRAIN resolution floors',
            'resolution_floors': dict(resolution_floors)}


def training_resolution_candidates(recipe_path, ledger_path, observation_path, functional_path, tau, *, paired_observation_path):
    """Combine FIT-only same-field differences and available tighter solves.

    Tighter nonconvergence is retained as unavailable, never treated as an
    accuracy bound. Values are empirical sensitivity indicators in native units.
    """
    names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid',
             'temperature/surface', 'temperature/material', 'q_proxy', 'J', 'P')
    floor = {n: 0. for n in names}
    observation = json.loads(Path(observation_path).read_text())['families']
    functionals = json.loads(Path(functional_path).read_text())['families']
    paired = json.loads(Path(paired_observation_path).read_text())
    for case_id in ('0001', '0318', '0333', '0348'):
        row = observation[case_id]
        for endpoint in ('baseline', 'trial'):
            for name, key in (('temperature/surface', 'surface'), ('temperature/material', 'material')):
                floor[name] = max(floor[name], row[endpoint][key]['numpy_attached']['max_abs'])
            floor['q_proxy'] = max(floor['q_proxy'], row[endpoint]['q_attached']['max_abs'])
        for name in ('temperature/surface', 'temperature/material', 'q_proxy'):
            floor[name] = max(floor[name], *(paired[case_id][name][key] for key in ('baseline_max_abs', 'trial_max_abs', 'paired_max_abs')))
        jrow = functionals[case_id]
        floor['J'] = max(floor['J'], abs(jrow['paired_J_resample_minus_attached']))
        floor['P'] = max(floor['P'], abs(jrow['paired_pressure_fp64_minus_fp32']))
        for endpoint in ('baseline', 'trial'):
            item = jrow[endpoint]
            floor['J'] = max(floor['J'], abs(item['J_resampled_fp64'] - item['J_attached_fp64']),
                             abs(item['J_attached_fp64'] - item['J_attached_fp32']))
            floor['P'] = max(floor['P'], abs(item['pressure_fp64'] - item['pressure_fp32']))
    rows = [json.loads(s) for s in Path(ledger_path).read_text().splitlines()]
    starts = {r['attempt']: r for r in rows if r['event'] == 'started'}
    results = {(starts[r['attempt']]['case_id'], starts[r['attempt']]['action'], starts[r['attempt']]['allocation']): r
               for r in rows if r['event'] == 'finished' and r['attempt'] in starts}
    fit = {c['case_id']: c for c in fitting_cases(recipe_path)}
    checks = []
    for (case_id, action, allocation), tight in results.items():
        if allocation != 'tight_audit' or case_id not in fit:
            continue
        check = {'case_id': case_id, 'action': action, 'status': tight['status']}
        if tight['status'] != 'converged' or tight['serialization_status'] != 'stored':
            checks.append(check); continue
        ordinary = None if action == 'baseline' else results.get((case_id, action, 'edited'))
        if action != 'baseline' and (not ordinary or ordinary['status'] != 'converged'):
            check['ordinary_unavailable'] = True; checks.append(check); continue
        base = load_raw_endpoint(fit[case_id]['source_case_dir'] if action == 'baseline' else ordinary['raw_case_dir'])
        strict = load_raw_endpoint(tight['raw_case_dir'])
        common = base['fluid_mask'] & strict['fluid_mask']
        field_delta = strict['field'].astype(np.float64)-base['field'].astype(np.float64)
        differences = {name: float(np.sqrt(np.mean(field_delta[..., i][common]**2))) for i, name in enumerate(names[:5])}
        for name, index in (('temperature/surface', 0), ('q_proxy', 1)):
            mask = base['outside_valid'] & strict['outside_valid'] if index else np.ones_like(base['outside_valid'], bool)
            delta = strict['interface_target'][..., index].astype(np.float64)-base['interface_target'][..., index]
            differences[name] = float(np.sqrt(np.mean(delta[mask]**2)))
        delta = strict['material'].astype(np.float64)-base['material'].astype(np.float64)
        differences['temperature/material'] = float(np.sqrt(np.mean(delta[:, base['material_mask']]**2)))
        for name, a, b in zip(('J', 'P'), _functionals(base, tau), _functionals(strict, tau)):
            differences[name] = abs(b-a)
        for name in names:
            floor[name] = max(floor[name], differences[name])
        check['differences'] = differences; checks.append(check)
    return {'floors': floor, 'tight_checks': checks, 'historical_fit_cases': ['0001', '0318', '0333', '0348'],
            'meaning': 'maximum available FIT-only same-field arithmetic and converged tight-solve differences; sensitivity indicator, not a continuum error bound'}
