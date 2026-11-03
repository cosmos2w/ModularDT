#!/usr/bin/env python3
"""Input-only R15-8 Thermal response recipe and raw endpoint extraction."""
from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np

EXCLUDED = {"0001", "0318", "0333", "0348", "0304", "0320", "0335", "0350", "0273"}
DIRECTIONS = np.asarray([[1., 0.], [0., 1.], [2 ** -.5, 2 ** -.5], [2 ** -.5, -(2 ** -.5)]])


def _feasible(centers, radius, domain):
    xy = np.asarray(centers)
    lo = np.array([radius + domain['inlet_margin'], radius + domain['wall_margin']])
    hi = np.array([domain['lx'] - radius - domain['outlet_margin'], domain['ly'] - radius - domain['wall_margin']])
    if np.any(xy < lo) or np.any(xy > hi):
        return False
    distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    distances += np.eye(len(xy)) * 1.e6
    return bool(np.all(distances >= 2 * radius + domain['min_gap']))


def _ordered_strata(rows, rng):
    """Round-robin seeded strata with source-ID order within each stratum."""
    if not rows:
        return []
    thresholds = np.quantile([r['heat_mean'] for r in rows], [1 / 3, 2 / 3])
    groups = {}
    for row in sorted(rows, key=lambda x: x['case_id']):
        key = (row['module_count'], int(np.searchsorted([70., 130.], row['re'])),
               int(np.searchsorted(thresholds, row['heat_mean'])))
        groups.setdefault(key, []).append(row)
    keys = sorted(groups)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    result = []
    while keys:
        following = []
        for key in keys:
            result.append(groups[key].pop(0))
            if groups[key]:
                following.append(key)
        keys = following
    return result


def build_recipe(dataset, identity_path, manifest_path, seed=15808):
    identity = json.loads(Path(identity_path).read_text())
    ordered_ids = identity['provider_identity']['training_case_ids']
    subset = set(json.loads(Path(manifest_path).read_text())['partitions']['train']['case_ids'])
    rng = np.random.default_rng(seed)
    rows = []
    with h5py.File(dataset, 'r') as handle:
        if handle.attrs['target_mode'] != 'converged_final':
            raise ValueError('Dataset baseline convention changed')
        for case_id in ordered_ids:
            if case_id in EXCLUDED:
                continue
            group = handle['cases'][case_id]
            if group.attrs['split'] != 'train':
                raise ValueError('TRAIN membership disagrees with source split')
            present = group['module_present'][...] > .5
            if present.sum() < 2:
                continue
            config = json.loads(group['case_config_json'][()].decode())
            # Preserve the original solver geometry and source precision: the
            # packed FP32 inputs are not identical to the source coordinates.
            centers = np.asarray(config['layout']['centers'], dtype=np.float64)
            heat = np.asarray(config['layout']['heat_powers'], dtype=np.float64)
            if len(centers) != int(present.sum()) or len(heat) != len(centers):
                raise ValueError('Original solver layout disagrees with packed membership')
            rows.append(dict(case_id=case_id, module_count=int(present.sum()), re=float(config['flow']['re']),
                             heat_mean=float(heat.mean()), centers=centers.tolist(), original_ids=np.flatnonzero(present).tolist(),
                             source_case_dir=str(group.attrs['source_case_dir']), selected_frame_ids=group['selected_frame_ids'][...].tolist(),
                             domain=config['domain']))
    qual = _ordered_strata([r for r in rows if r['case_id'] in subset], rng)[:12]
    used = {r['case_id'] for r in qual}
    rest = _ordered_strata([r for r in rows if r['case_id'] not in used], rng)
    selected = qual + rest[:48]
    for index, row in enumerate(selected):
        role = 'fit' if index < 48 else 'response_holdout'
        within = index if role == 'fit' else index - 48
        radius = float(row['domain']['module_radius'])
        amplitude = radius * (.15 if within % 2 == 0 else .35)
        joint = 'co' if (within // 2) % 2 == 0 else 'opposing'
        xy = np.asarray(row['centers'])
        candidates = list(itertools.product(itertools.combinations(range(len(xy)), 2), range(4)))
        options = [candidates[i] for i in rng.permutation(len(candidates))]
        chosen = None
        for (i, j), d in options:
            v = DIRECTIONS[d] * amplitude
            states = {}
            for name, sign, both in [('single_plus', 1, False), ('single_minus', -1, False), ('joint_plus', 1, True), ('joint_minus', -1, True)]:
                moved = xy.copy(); moved[i] += sign * v
                if both:
                    moved[j] += sign * v * (1 if joint == 'co' else -1)
                states[name] = moved.tolist()
            if all(_feasible(c, radius, row['domain']) for c in states.values()):
                chosen = dict(member_ids=[row['original_ids'][i], row['original_ids'][j]], direction=DIRECTIONS[d].tolist(), actions=states)
                break
        row.update(role=role, qualification=index < len(qual), amplitude=amplitude, joint_kind=joint,
                   feasible=chosen is not None, selection=chosen)
    return dict(schema_version=1, seed=seed, dataset=str(dataset), source_membership='original ordered TRAIN600',
                input_strata='module count, Re bands 70/130, within-pool mean-heat tertiles; seeded stratum round-robin, source-ID order within stratum',
                eligible_train150=sum(r['case_id'] in subset for r in rows), cases=selected)


def run_edit(recipe_path, root, case_id, action, *, tight=False):
    """Start one declared edit; write the ledger entry before invoking the solver."""
    begin = time.monotonic(); cpu_begin = time.process_time()
    project = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(project / 'src'), str(project / 'Case_ThermalChannel' / 'src')]
    from channelthermal.interaction_evidence.reference_adapter import AnalyticWakeReferenceAdapter, operating_context_from_config
    from channelthermal.interaction_evidence.storage import persist_solve_record
    from channelthermal.interaction_evidence.types import DesignState, ModuleState

    root = Path(root).resolve()
    recipe = json.loads(Path(recipe_path).read_text())
    case = next(x for x in recipe['cases'] if x['case_id'] == case_id)
    if not case['feasible'] or (action not in case['selection']['actions'] and not (tight and action == 'baseline')):
        raise ValueError('Only the preselected feasible actions may run')
    baseline_audit = json.loads((root / 'evidence' / 'baseline_source_parity.json').read_text())
    audited = next(x for x in baseline_audit['cases'] if x['case_id'] == case_id)
    if not (audited['converged'] and all(audited['config_equal'].values()) and all(audited['raw_packed_equal'].values())):
        raise ValueError('Baseline was not verified for reuse')
    with h5py.File(recipe['dataset'], 'r') as h:
        config = json.loads(h['cases'][case_id]['case_config_json'][()].decode())
    if not config['thermal']['stop_on_convergence'] or config['thermal']['max_solve_time'] != 200:
        raise ValueError('The original convergence and max-time recipe must remain unchanged')
    if tight:
        if case['role'] != 'fit':
            raise ValueError('Tight audits are restricted to selected fitting families')
        config['thermal']['convergence_tol'] *= .1
        config['thermal']['convergence_rel_tol'] *= .1
    baseline = DesignState(case_id, f'r158:{case_id}', 'train', tuple(
        ModuleState(f'{case_id}:module:{slot}', tuple(xy), float(q))
        for slot, xy, q in zip(case['original_ids'], config['layout']['centers'], config['layout']['heat_powers'])))
    coordinates = case['centers'] if action == 'baseline' else case['selection']['actions'][action]
    moved = DesignState(case_id, f'r158:{case_id}', 'train', tuple(
        ModuleState(m.module_id, tuple(xy), m.heating) for m, xy in zip(baseline.modules, coordinates)))
    context = operating_context_from_config(config)
    ledger = root / 'references' / 'attempts.jsonl'
    def append(row):
        with ledger.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
    with (root / 'references' / '.serial.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
        started = [x for x in rows if x['event'] == 'started']; finished = {x['attempt']: x for x in rows if x['event'] == 'finished'}
        if any(x['attempt'] not in finished for x in started):
            raise RuntimeError('Inspect the unfinished physical attempt before another start')
        allocation = 'tight_audit' if tight else 'edited'
        if any(x['case_id'] == case_id and x['action'] == action and x['allocation'] == allocation for x in started):
            raise ValueError('This action already has an attempt; no automatic retry')
        if len(started) >= 320 or sum(x['allocation'] == allocation for x in started) >= (8 if tight else 240):
            raise RuntimeError(f'{allocation} allocation exhausted')
        remaining = 5400. - sum(x['wall_seconds'] for x in finished.values()) - (time.monotonic() - begin)
        if remaining <= 0:
            raise RuntimeError('Serial physical wall budget exhausted')
        attempt = len(started) + 1
        append(dict(event='started', attempt=attempt, case_id=case_id, action=action, role=case['role'],
                    recipe=str(Path(recipe_path).resolve()), centers=coordinates,
                    baseline_raw=case['source_case_dir'], allocation=allocation))
        def timeout(_signum, _frame):
            raise TimeoutError('Serial physical wall budget exhausted')
        old = signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, remaining)
        record = None; saved = None; error = None
        try:
            adapter = AnalyticWakeReferenceAdapter(demo_root=project / '1_Demo_ChannelThermal', output_root=root / 'references' / 'raw' / case_id, case_template=config)
            record = adapter.solve(moved, context, record_id=f'r158_{case_id}_{action}_{allocation}', baseline_design=baseline)
            saved = persist_solve_record(record, root / 'references' / 'records' / f'attempt_{attempt:03d}')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)
        outcome = dict(event='finished', attempt=attempt, status=record.status.value if record else 'failed',
                       failure=record.failure_reason if record else error, wall_seconds=time.monotonic()-begin,
                       cpu_seconds=time.process_time()-cpu_begin, serialization_status=saved.serialization_status if saved else None,
                       raw_case_dir=saved.raw_case_dir if saved else None, record_directory=str(saved.destination) if saved else None)
        append(outcome)
        return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset')
    parser.add_argument('--identity')
    parser.add_argument('--manifest')
    parser.add_argument('--output')
    parser.add_argument('--recipe')
    parser.add_argument('--study-root')
    parser.add_argument('--case-id')
    parser.add_argument('--action')
    parser.add_argument('--all-edits', action='store_true')
    parser.add_argument('--tight-audits', action='store_true')
    args = parser.parse_args()
    if args.recipe:
        if args.tight_audits:
            if not args.study_root or args.all_edits or args.case_id or args.action:
                parser.error('--tight-audits requires only --recipe and --study-root')
            recipe = json.loads(Path(args.recipe).read_text())
            fit = [r for r in recipe['cases'] if r['role'] == 'fit' and r['feasible']]
            low = min(r['module_count'] for r in fit); high = max(r['module_count'] for r in fit)
            chosen = [next(r for r in fit if r['module_count'] == n) for n in (low, high)]
            ledger = Path(args.study_root) / 'references' / 'attempts.jsonl'
            rows = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
            seen = {(r['case_id'], r['action']) for r in rows if r['event'] == 'started' and r['allocation'] == 'tight_audit'}
            for case in chosen:
                for action in ('baseline', 'single_plus', 'single_minus', 'joint_plus'):
                    if (case['case_id'], action) in seen:
                        continue
                    result = run_edit(args.recipe, args.study_root, case['case_id'], action, tight=True)
                    print(json.dumps(result), flush=True)
                    if result['status'] not in ('converged', 'unconverged') or result['serialization_status'] != 'stored':
                        raise RuntimeError('Tight audit execution failed; inspect before continuing')
            return
        if args.all_edits:
            if not args.study_root or args.case_id or args.action:
                parser.error('--all-edits requires --study-root and no individual case/action')
            recipe = json.loads(Path(args.recipe).read_text())
            ledger = Path(args.study_root) / 'references' / 'attempts.jsonl'
            rows = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
            if any(row['event'] == 'finished' and (row['status'] not in ('converged', 'unconverged') or row['serialization_status'] != 'stored') for row in rows):
                raise RuntimeError('An earlier failed attempt requires inspection before continuing')
            completed = {(row['case_id'], row['action']) for row in rows if row['event'] == 'started'}
            for case in recipe['cases']:
                if not case['feasible']:
                    continue
                for action in ('single_plus', 'single_minus', 'joint_plus', 'joint_minus'):
                    if (case['case_id'], action) in completed:
                        continue
                    result = run_edit(args.recipe, args.study_root, case['case_id'], action)
                    print(json.dumps(result), flush=True)
                    if result['status'] not in ('converged', 'unconverged') or result['serialization_status'] != 'stored':
                        raise RuntimeError('Physical attempt failed to execute or persist; inspect before continuing')
            return
        if not all((args.study_root, args.case_id, args.action)):
            parser.error('--recipe requires --study-root, --case-id and --action')
        print(json.dumps(run_edit(args.recipe, args.study_root, args.case_id, args.action), indent=2))
        return
    if not all((args.dataset, args.identity, args.manifest, args.output)):
        parser.error('recipe generation requires --dataset, --identity, --manifest and --output')
    result = build_recipe(args.dataset, args.identity, args.manifest)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"selected={len(result['cases'])} feasible={sum(r['feasible'] for r in result['cases'])} eligible_train150={result['eligible_train150']}")


if __name__ == '__main__':
    main()
