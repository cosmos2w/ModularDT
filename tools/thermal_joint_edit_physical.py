#!/usr/bin/env python3
"""Execute one budgeted, recorded Thermal analytic-wake reference attempt."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'Case_ThermalChannel' / 'src')]

import numpy as np
import torch
from channelthermal.interaction_evidence.reference_adapter import AnalyticWakeReferenceAdapter, operating_context_from_config
from channelthermal.interaction_evidence.storage import persist_solve_record, load_solve_record, _json_value
from channelthermal.inverse.joint_edit_reference import edited_design, load_eligible_atlas, native_reference_quantities

CHECKPOINT = Path('/data2/wanglz/ModularDT/r15_thermal_wind/r15_6/thermal/train600/thermal_r15_6_p64_grid192_train600_r14e5000_v1/checkpoints/epoch_0100_model.pt')
ATLAS = ROOT / 'diagnostics/generated/interactions/physical_response_atlas_20260926'
FAMILIES = {'fit': {'0001', '0318', '0333', '0348'}, 'dev': {'0304', '0320', '0335', '0350'}}
LIMITS = {'baseline': 8, 'fit': 24, 'dev': 48, 'retry': 16}


def _append(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _timeout(_signum, _frame):
    raise TimeoutError('Physical serial time allocation exhausted')


def _save_baseline(root, case_id, record, checkpoint):
    scale = float(checkpoint['model_state_dict']['internal_temperature_std'].reshape(-1)[0])
    quantities = native_reference_quantities(record, .05 * scale)
    directory = root / 'baselines'
    directory.mkdir(exist_ok=True)
    path = directory / f'{case_id}.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(quantities, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_reference(*, output_root, case_id, allocation, centers=None, fresh=False):
    """Run or reuse a reference. Caller must supply physical output root and allocation."""
    invocation_started = time.monotonic()
    case_id = str(case_id)
    if allocation not in LIMITS or case_id not in FAMILIES['fit'] | FAMILIES['dev']:
        raise ValueError('Unknown allocation or out-of-scope family.')
    if allocation in FAMILIES and case_id not in FAMILIES[allocation]:
        raise ValueError('Family does not belong to the specified FIT/DEV allocation.')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    identity = checkpoint['experiment_identity']
    train_ids = identity['provider_identity']['training_case_ids']
    dataset = identity['recipe']['dataset']['data_path']
    stencil, _ = load_eligible_atlas(dataset, case_id, train_ids, ATLAS / 'families' / f'train_{case_id}_responses.npz')
    manifest = json.loads((ATLAS / 'split_anchor_manifest_frozen_20260926.json').read_text())
    entry = next(f for f in manifest['families'] if f['anchor_id'] == case_id)
    config = entry['case_config']
    context = operating_context_from_config(config)
    baseline = stencil.baseline.design
    design = baseline if centers is None else edited_design(baseline, np.asarray(centers, dtype=np.float64))
    if allocation == 'baseline' and design != baseline:
        raise ValueError('Baseline allocation only supports the unchanged design.')
    # The family manifest is the R15-4 solver recipe. This retains the actual
    # convergence controls and excludes output names from the matching identity.
    thermal = config['thermal']
    if not thermal.get('stop_on_convergence') or float(thermal.get('max_solve_time', 0)) != 200:
        raise ValueError('Frozen family recipe does not retain the required convergence and maximum time.')
    key = {'case_id': case_id, 'modules': [{'id': m.module_id, 'xy': list(m.position_xy), 'heat': m.heating, 'active': m.active} for m in design.modules],
           'config': {k: v for k, v in config.items() if k not in ('save', 'execution', 'runtime')},
           'context': dict(context.values), 'extraction': 'reference_adapter_native_roles_v1'}
    root = Path(output_root).expanduser().resolve() / 'physical'
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / 'attempts.jsonl'
    with (root / '.serial.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = _read(ledger)
        started = [r for r in rows if r['event'] == 'started']
        finished = {r['attempt']: r for r in rows if r['event'] == 'finished'}
        if any(r['attempt'] not in finished for r in started):
            raise RuntimeError('Unfinished physical attempt requires inspection before another start.')
        matches = [r for r in started if r['key'] == key and finished[r['attempt']]['status'] == 'converged'
                   and finished[r['attempt']]['serialization_status'] == 'stored']
        if matches and not fresh:
            original = matches[0]
            saved = root / 'records' / f"attempt_{original['attempt']:03d}"
            record = load_solve_record(saved)
            if record.design != design or dict(record.context.values) != dict(context.values):
                raise RuntimeError('Saved physical result does not match its recorded design/context.')
            if allocation == 'baseline':
                _save_baseline(root, case_id, record, checkpoint)
            row = {'event': 'reused', 'case_id': case_id, 'allocation': allocation,
                   'original_attempt': original['attempt'], 'record_directory': str(saved)}
            _append(ledger, row)
            return row
        if allocation == 'retry' and not any(r['key'] == key and finished[r['attempt']]['status'] != 'converged' for r in started):
            raise ValueError('Retry reserve is limited to an existing failed/unconverged identical design and recipe.')
        if len(started) >= 96 or sum(r['allocation'] == allocation for r in started) >= LIMITS[allocation]:
            raise RuntimeError('Physical attempt allocation exhausted.')
        if allocation == 'baseline' and any(r['case_id'] == case_id and r['allocation'] == 'baseline' for r in started):
            raise RuntimeError('Baseline family already received its allocated attempt; use retry only after failure.')
        elapsed_prior = sum(float(finished[r['attempt']]['serial_seconds']) for r in started)
        remaining = 1800.0 - elapsed_prior - (time.monotonic() - invocation_started)
        if remaining <= 0:
            raise RuntimeError('Physical serial time allocation exhausted.')
        attempt = len(started) + 1
        _append(ledger, {'event': 'started', 'attempt': attempt, 'case_id': case_id, 'allocation': allocation,
                         'key': key, 'checkpoint': str(CHECKPOINT), 'dataset_path': dataset})
        status, error, record, persistence = 'failed', None, None, None
        previous = signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            adapter = AnalyticWakeReferenceAdapter(demo_root=ROOT / '1_Demo_ChannelThermal', output_root=root / 'raw' / case_id, case_template=config)
            record = adapter.solve(design, context, record_id=f'r157_{case_id}_attempt_{attempt:03d}', baseline_design=baseline)
            status = record.status.value
            persistence = persist_solve_record(record, root / 'records' / f'attempt_{attempt:03d}')
            if allocation == 'baseline' and status == 'converged':
                _save_baseline(root, case_id, record, checkpoint)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        row = {'event': 'finished', 'attempt': attempt, 'status': status,
               'serial_seconds': time.monotonic() - invocation_started, 'failure': error if record is None else record.failure_reason,
               'serialization_status': persistence.serialization_status if persistence else 'unavailable',
               'serialization_error': persistence.error if persistence else None,
               'record_directory': str(persistence.destination) if persistence else None,
               'raw_case_dir': persistence.raw_case_dir if persistence else None,
               'solver_invoked': record.provenance.get('solver_invoked') if record else None,
               'raw_solver_completed': record.provenance.get('raw_solver_completed') if record else None,
               'provenance': _json_value(record.provenance) if record else None}
        _append(ledger, row)
        return row


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--allocation', required=True, choices=tuple(LIMITS))
    parser.add_argument('--centers-json', help='JSON array of original-slot [x,y] centers; omit for baseline')
    parser.add_argument('--fresh', action='store_true', help='Start a fresh allocated attempt rather than reuse')
    args = parser.parse_args()
    centers = json.loads(Path(args.centers_json).read_text()) if args.centers_json else None
    result = run_reference(output_root=args.output_root, case_id=args.case_id, allocation=args.allocation,
                           centers=centers, fresh=args.fresh)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get('status', 'converged') == 'converged' else 1


if __name__ == '__main__':
    raise SystemExit(main())
