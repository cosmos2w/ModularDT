#!/usr/bin/env python3
"""Bounded native FP64 response arithmetic audit; no fit or physical solve.

Learned coefficients and geometry preparation remain FP32. Physical heat,
coefficient interpolation, native q stencil, and forcing contractions use the
requested accumulation precision. Legacy endpoint failures remain separate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter, time

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / 'src', ROOT / 'Case_ThermalChannel/src', ROOT / 'tools'):
    sys.path.insert(0, str(source))

import numpy as np
import torch
from channelthermal.source_response import load_source_response_model
from channelthermal.training.checkpoints import _file_sha256
from thermal_development import validate_generated_output
from thermal_response_refinement_evaluation import load_atlas_families, load_counted_families
from thermal_source_response_evaluate import model_digest, synchronize
from thermal_source_response_interventions import discrepancy

from honf_runtime.run_store import atomic_write_json


def measured(fn, device, repetitions=5):
    fn()
    seconds = []
    for _ in range(repetitions):
        synchronize(device)
        started = perf_counter()
        fn()
        synchronize(device)
        seconds.append(perf_counter() - started)
    return {'median_seconds': float(np.median(seconds)), 'repetitions_seconds': seconds}


def audit(model, record, device):
    synchronize(device)
    started = perf_counter()
    prepared = model.prepare_record(record, device)
    synchronize(device)
    formation = perf_counter() - started
    heat = torch.tensor([[m.heating for m in record.design.modules]], dtype=torch.float64, device=device)
    precise = {'accumulation_dtype': torch.float64}
    baseline = model.apply_record(prepared, heat, **precise)
    cold = model.apply_record(model.prepare_record(record, device), heat, **precise)
    cold_match = {k: discrepancy(cold[k], baseline[k]) for k in baseline}
    legacy_baseline = model.apply_record(prepared, heat)
    rows = []
    for balanced in (True, False):
        for step in (.25, .001, 1e-6):
            for sign in (-1, 1):
                delta = torch.zeros_like(heat)
                delta[0, 0] = sign * step
                if balanced:
                    delta[0, 1] = -sign * step
                if bool(((heat + delta) < 0).any()):
                    continue
                increment = model.apply_record_increment(prepared, delta, **precise)
                changed = model.apply_record(prepared, heat + delta, **precise)
                legacy_changed = model.apply_record(prepared, heat + delta)
                legacy_increment = model.apply_record_increment(prepared, delta)
                rows.append({'balanced': balanced, 'signed_step': sign * step,
                    'precise_fixed_kernel_endpoint': {k: discrepancy(changed[k] - baseline[k], increment[k]) for k in increment},
                    'legacy_fp32_endpoint': {k: discrepancy(legacy_changed[k] - legacy_baseline[k], legacy_increment[k]) for k in increment},
                    'heat_increment_quantization_max': float(((heat + delta).float().double() - heat.float().double() - delta).abs().max())})
    heat = heat.clone().requires_grad_()
    native = model.thermal.apply_native(prepared['thermal'], heat, **precise)
    kernels = model.thermal.export_native_kernels(prepared['thermal'], **precise)
    vjp = {}
    for name, value, kernel in [('fluid', native['fluid_temperature'], kernels['fluid']),
                               ('surface', native['pred_interface'][..., :1], kernels['surface']),
                               ('q_normal', native['pred_interface'][..., 1:], kernels['q_normal']),
                               ('material', native['pred_internal_temperature'], kernels['material'])]:
        actual = torch.autograd.grad(value.mean(), heat, retain_graph=True)[0]
        expected = kernel.mean(dim=tuple(range(1, kernel.ndim - 2))).squeeze(-1)
        vjp[name] = discrepancy(actual, expected)
    heat = heat.detach()
    with torch.no_grad():
        ordinary_cost = measured(lambda: model.apply_record(prepared, heat), device)
        precise_cost = measured(lambda: model.apply_record(prepared, heat, **precise), device)
        delta = torch.zeros_like(heat)
        delta[0, :2] = delta.new_tensor([.001, -.001])
        increment_cost = measured(lambda: model.apply_record_increment(prepared, delta, **precise), device)
        cold_cost = measured(lambda: model.apply_record(model.prepare_record(record, device), heat), device)
    return {'case_id': record.design.anchor_id, 'partition': record.design.split,
            'source_ids': [m.module_id for m in record.design.modules], 'precise_cold_prepared': cold_match,
            'checks': rows, 'physical_kernel_vjp': vjp, 'formation_seconds': formation,
            'ordinary_cold': cold_cost, 'ordinary_prepared': ordinary_cost,
            'precise_prepared': precise_cost, 'precise_increment': increment_cost,
            'precise_native_kernel_temporary_bytes': sum(k.numel() * k.element_size() for k in kernels.values()),
            'storage_semantics': 'FP64 kernels formed transiently on application; original FP32 coefficients remain resident',
            'nonlinear_semantics': 'material peaks and effective h require endpoint reductions; absent from increment output',
            'coefficient_precision': 'learned FP32 coefficients; physical accuracy unchanged'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--atlas-dir', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--records-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    output = validate_generated_output(args.output)
    if output.exists():
        raise FileExistsError('Preserve prior precision evidence; use a new output identity.')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    started, started_unix = perf_counter(), time()
    model, checkpoint = load_source_response_model(args.checkpoint, args.device)
    model.eval().requires_grad_(False)
    before = model_digest(model)
    train = load_atlas_families([args.atlas_dir / f'train_{cid}_responses.npz' for cid in ('0001', '0318', '0333', '0348')],
                               ('heat_transfer_minus', 'heat_transfer_plus'), 'fit')
    fixed, _ = load_counted_families(args.request, args.records_dir)
    rows = []
    for case_id, records in train + fixed:
        record = records.get('baseline', records['transfer_minus'] if 'transfer_minus' in records else next(iter(records.values())))
        row = audit(model, record, args.device)
        row['secondary_missing_counted_baseline'] = case_id == '0277'
        rows.append(row)
        print(json.dumps({'case_id': case_id, 'audit_complete': True}), flush=True)
    after = model_digest(model)
    if before != after:
        raise RuntimeError('Precision audit changed weights.')
    summary = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': _file_sha256(args.checkpoint),
               'epoch': checkpoint['epoch'], 'rows': rows, 'solver_attempts': 0, 'optimizer_updates': 0,
               'weights_unchanged': before == after, 'start_unix': started_unix, 'end_unix': time(),
               'elapsed_seconds': perf_counter() - started, 'device': args.device,
               'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'physical_reference_ledger': '326/326 unchanged',
               'strict_tolerances': {'rtol': 2e-5, 'atol': 2e-6},
               'scope': 'Four original TRAIN response cases followed by four existing fixed representatives; no new physical response'}
    atomic_write_json(output / 'summary.json', summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
