#!/usr/bin/env python3
"""Matched, frozen-predictor Thermal joint-edit finite-pool study.

Generate FIT first, review its fixed defaults, then generate all DEV nominations
before physical verification. Physical execution is a separate explicit command.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'Case_ThermalChannel/src')]

import numpy as np
import torch
from tqdm import tqdm
from honf_runtime.compat import load_trusted_checkpoint
from honf_inverse_core.contracts import NamedContext, PhysicalDesign
from honf_inverse_core.request_schema import GeometryConstraints
from channelthermal.inverse.joint_edit_reference import load_eligible_atlas, native_reference_quantities
from channelthermal.inverse.joint_edit_sampling import (common_draws, make_loadings, propose,
    request_energies, finite_pool_weights, nominate)

FIT = ('0001', '0318', '0333', '0348')
DEV = ('0304', '0320', '0335', '0350')
ARMS = ('I', 'G', 'H')


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, default=jsonable, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)


def source_records(checkpoint, atlas):
    saved = load_trusted_checkpoint(checkpoint, map_location='cpu')
    identity = saved['experiment_identity']
    ids = identity['provider_identity']['training_case_ids']
    dataset = identity['recipe']['dataset']['data_path']
    records = {}
    for case in FIT + DEV:
        records[case] = load_eligible_atlas(dataset, case, ids, atlas / f'train_{case}_responses.npz')[0].baseline
    return saved, records


def geometry_inputs(record, domain):
    c = record.context.values
    design = PhysicalDesign(np.asarray([m.position_xy for m in record.design.modules]),
                            np.asarray([m.active for m in record.design.modules], dtype=float),
                            np.asarray([m.heating for m in record.design.modules]))
    names = ('module_radius', 'domain_length_x', 'domain_length_y')
    context = NamedContext(names, np.asarray([c[k] for k in names]), 'thermal_joint_edit')
    constraints = GeometryConstraints(design.module_count, design.module_count,
        2*c['module_radius'] + domain['min_gap'], domain['wall_margin'],
        domain['inlet_margin'], domain['outlet_margin'])
    return design, context, constraints


def design_key(centers):
    return tuple(np.asarray(centers, dtype=np.float32).reshape(-1).tolist())


def generate(args):
    from channelthermal.inverse.joint_edit_predictor import FrozenJointEditPredictor
    output = args.output
    phase_path = output / 'proposals' / args.phase
    if phase_path.exists():
        raise FileExistsError(f'Preserve existing proposal evidence: {phase_path}')
    if args.phase == 'dev' and not (output / 'recipes/fit_review.json').is_file():
        raise ValueError('Record the FIT review before generating DEV.')
    phase_path.mkdir(parents=True)
    started = time.perf_counter()
    cpu_started = time.process_time()
    saved, records = source_records(args.checkpoint, args.atlas)
    if int(saved['epoch']) != 100 or saved['experiment_identity']['recipe']['thermal_port_response']['arm'] != 'P64':
        raise ValueError('This study requires literal frozen P64 epoch 100.')
    predictor = FrozenJointEditPredictor(args.checkpoint, args.device)
    baselines = {}
    for case, record in records.items():
        path = output / 'physical' / 'baselines' / f'{case}.json'
        if not path.is_file():
            raise FileNotFoundError(f'Physical baseline extraction required: {path}')
        baselines[case] = json.loads(path.read_text())
    pressure_rms = float(np.sqrt(np.mean([baselines[c]['pressure_drop'] ** 2 for c in FIT])))
    families = FIT if args.phase == 'fit' else DEV
    manifest = json.loads((args.atlas.parent / 'split_anchor_manifest_frozen_20260926.json').read_text())
    domains = {r['anchor_id']: r['case_config']['domain'] for r in manifest['families']}
    inlets = {r['anchor_id']: r['case_config']['thermal']['t_in'] for r in manifest['families']}
    rows, nominations, counts, costs = [], [], {}, []
    recipe = {'checkpoint': str(args.checkpoint), 'phase': args.phase, 'families': families,
              'arms': ARMS, 'raw_per_cell': 64, 'first_cell_attempts': args.first_cell_attempts,
              'first_cell': {'case': families[0], 'request': 'A', 'arms': ARMS},
              'blocks': 2, 'block_size': 32,
              'seed_formula': '157000 + int(case)*100 + request_index*10 + block',
              'sigma_over_radius': .05, 'max_norm_over_radius': .15, 'rho': 1/np.sqrt(2),
              'kappa': .25, 'temperature': 1., 'fit_pressure_rms': pressure_rms,
              'geometry': 'exact family domain margins and minimum gap; reject without repair',
              'known_physical_baseline_required': True, 'optimizer_updates': 0, 'training_epochs': 0}
    write_json(output / 'recipes' / f'{args.phase}.json', recipe)
    for case in families:
        record = records[case]
        design, context, constraints = geometry_inputs(record, domains[case])
        base = predictor.evaluate(record, design.module_centers, save_path=phase_path / f'{case}_baseline.npz')
        org = base['organization']
        write_json(phase_path / f'{case}_organization.json', asdict(org))
        mids, eids = base['module_ids'], base['environment_ids']
        exy = base['environment_coordinates']
        c = record.context.values
        tin = inlets[case]
        j0, p0 = baselines[case]['objective'], baselines[case]['pressure_drop']
        scale = predictor.temperature_scale
        loadings = {arm: make_loadings(arm, design.module_centers, exy,
                    [c['domain_length_x'], c['domain_length_y']], org) for arm in ARMS}
        np.savez_compressed(phase_path / f'{case}_loadings.npz', **loadings)
        # Synchronized repeat is a measured warm diagnostic, not another arm.
        warm = predictor.evaluate(record, design.module_centers)
        costs.append({'case': case, 'kind': 'baseline_cold', **base['timing'], 'work': base['work']})
        costs.append({'case': case, 'kind': 'baseline_warm', **warm['timing'], 'work': warm['work']})
        for ri, request in enumerate(('A', 'B')):
            draws = [common_draws(157000 + int(case)*100 + ri*10 + block,
                      mids, mids, eids, 32) for block in range(2)]
            np.savez_compressed(phase_path / f'{case}_{request}_common_draws.npz',
                pairs=np.asarray([[d.pair for d in block] for block in draws]),
                epsilon=np.asarray([[d.epsilon for d in block] for block in draws]),
                z=np.asarray([[d.z for d in block] for block in draws]),
                uniforms=np.asarray([[d.nomination_uniform for d in block] for block in draws]))
            for arm in ARMS:
                pool, cache = [], {}
                for block in range(2):
                    for di, draw in enumerate(tqdm(draws[block], desc=f'{case} {request} {arm} b{block+1}', disable=args.no_progress)):
                        if case == families[0] and request == 'A' and 32 * block + di >= args.first_cell_attempts:
                            continue
                        t = time.perf_counter()
                        trial, rejection = propose(design, context, constraints, mids, loadings[arm], draw)
                        geometry_seconds = time.perf_counter() - t
                        centers = trial.module_centers
                        key = design_key(centers)
                        row = {'case': case, 'phase': args.phase, 'request': request, 'arm': arm,
                               'block': block+1, 'draw': di, 'pair': draw.pair, 'centers': centers,
                               'delta': centers - design.module_centers, 'rejection': rejection,
                               'geometry_seconds': geometry_seconds, 'duplicate': key in cache}
                        if rejection is None:
                            if key not in cache:
                                result = predictor.evaluate(record, centers)
                                cache[key] = {k: result[k] for k in ('objective', 'pressure', 'sampled_max', 'timing', 'work')}
                                costs.append({'case': case, 'kind': 'candidate', 'arm': arm,
                                              **result['timing'], 'work': result['work']})
                            result = cache[key]
                            j = j0 + result['objective'] - base['objective']
                            p = p0 + result['pressure'] - base['pressure']
                            energies = {r: float(request_energies(r, j, p, j0, p0, tin, scale, pressure_rms)) for r in ('A','B')}
                            row.update(model_j=result['objective'], model_p=result['pressure'],
                                       anchored_j=j, anchored_p=p, energies=energies)
                            pool.append(row)
                        rows.append(row)
                cell = f'{case}_{request}_{arm}'
                attempted = sum(r['case']==case and r['request']==request and r['arm']==arm for r in rows)
                counts[cell] = {'attempted': attempted, 'valid': len(pool), 'unique': len(cache), 'scored': len(cache)}
                blocks = [pool] if args.phase == 'fit' else [[r for r in pool if r['block'] == b] for b in (1,2)]
                for slot, candidates in enumerate(blocks):
                    weighted = slot == 0
                    u = draws[slot][0].nomination_uniform
                    keys = [design_key(r['centers']) for r in candidates]
                    en = [r['energies'][request] for r in candidates]
                    chosen = nominate(keys, en, u, weighted=weighted)
                    first = list(dict.fromkeys(keys))
                    unique = [candidates[keys.index(k)] for k in first]
                    weights = finite_pool_weights([r['energies'][request] for r in unique]) if weighted else np.full(len(unique), 1/max(len(unique),1))
                    write_json(phase_path / f'{cell}_slot{slot+1}_distribution.json',
                               {'uniform': u, 'weighted': weighted, 'candidates': unique, 'probabilities': weights,
                                'request_swap_probabilities': {r: finite_pool_weights([x['energies'][r] for x in unique]) for r in ('A','B')}})
                    nomination = {'id': f'{cell}_slot{slot+1}', 'case': case, 'phase': args.phase,
                                  'request': request, 'arm': arm, 'slot': slot+1,
                                  'policy': 'weighted' if weighted else 'uniform', 'uniform': u,
                                  'empty': chosen is None, 'j0': j0, 'p0': p0,
                                  'sT': max(j0-tin, .01*scale), 'sP': max(abs(p0), .01*pressure_rms)}
                    if chosen is not None:
                        row = candidates[chosen]
                        nomination.update({k: row[k] for k in ('centers','pair','block','draw','model_j','model_p','anchored_j','anchored_p','energies')})
                        replay = predictor.evaluate(record, row['centers'], save_path=phase_path / (nomination['id'] + '.npz'))
                        write_json(phase_path / (nomination['id'] + '_organization.json'), asdict(replay['organization']))
                        costs.append({'case': case, 'kind': 'nominee_replay', 'arm': arm,
                                      **replay['timing'], 'work': replay['work']})
                    nominations.append(nomination)
    write_json(phase_path / 'raw_proposals.json', rows)
    write_json(phase_path / 'nominations.json', nominations)
    write_json(phase_path / 'costs.json', costs)
    write_json(phase_path / 'summary.json', {'counts': counts, 'nominations':len(nominations),
        'elapsed_seconds':time.perf_counter()-started, 'loading_seconds':predictor.load_seconds,
        'cpu_seconds':time.process_time()-cpu_started,
        'model_state_unchanged':predictor.unchanged(),
        'optional_surrogate_arms':'omitted to prioritize matched primary verification and complete diagnostics'})
    print(json.dumps({'phase': args.phase, 'nominations': len(nominations), 'seconds': time.perf_counter()-started}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--atlas', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('fit','dev'), required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--no-progress', action='store_true')
    p.add_argument('--first-cell-attempts', type=int, choices=range(1, 65), default=64,
                   metavar='1..64', help='raw draws per arm in the first case/request cell (default: 64)')
    args=p.parse_args(argv)
    torch.set_num_threads(1)
    generate(args)


if __name__ == '__main__':
    main()
