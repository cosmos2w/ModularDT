#!/usr/bin/env python3
"""Capture one predeclared saved native reference scene without fitting or solves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in ('src', 'Case_ThermalChannel/src', 'Case_WindFarm/src', 'tools'):
    sys.path.insert(0, str(ROOT / directory))

import numpy as np
import torch
from channelthermal.data.finite_responses import FiniteResponseStore
from channelthermal.native_design import build_native_design
from channelthermal.training.native_checkpoint import build_native_checkpoint_inference
from honf_forward_core.interaction_reused_member_routing.organization_response import (
    capture_applied_relationships, contextual_coefficients_trace_request,
)
from honf_runtime.compat import load_trusted_checkpoint
from thermal_organization_response_diagnostic import _flatten, _geometry, _scene_and_query, _select_fixed_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--pack', required=True, type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--read-mode', default='restricted', choices=('restricted', 'full'))
    parser.add_argument('--query-count', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--loader-validation-case', default='0679',
                        help='Exposed validation case for loading when the captured reference is a TRAIN case.')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not 1 <= args.query_count <= 64:
        raise ValueError('Choose 1–64 fixed fluid positions.')
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    source = FiniteResponseStore(args.pack)
    metadata = source.reference_metadata(args.case)
    raw = source.reference_baseline(args.case)
    saved = load_trusted_checkpoint(args.checkpoint, map_location='cpu')
    loader_case = args.case if metadata['partition'] == 'validation' else args.loader_validation_case
    model, _ = build_native_checkpoint_inference(saved, validation_case_ids=(loader_case,), device=args.device)
    model.eval()
    if args.query_count > model.core.query_tile_size:
        raise ValueError('The capture must fit within one reader query tile.')
    design = build_native_design(raw['config'], args.case, device=args.device)
    if not np.array_equal(design.module_id_grid, raw['module_id_grid']):
        raise ValueError('Saved physical ownership differs from the input-derived scene.')
    selected = _select_fixed_positions(design, design, args.query_count, args.seed)
    scene, query = _scene_and_query(model, design, selected)
    with torch.no_grad():
        prepared = model.core.prepare(scene, trace_request=contextual_coefficients_trace_request(model.core))
        relations = capture_applied_relationships(model.core, prepared)
        result = model.core.decode(prepared, query, intervention={'read_mode': args.read_mode}, return_trace=True)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.synchronize(args.device)
    arrays = {}
    capture = _flatten({'relationships': {'before': relations}, 'geometry': {'before': _geometry(scene, raw['config'])},
                        'query_trace': result.trace}, arrays)
    output = Path(args.output); array_path = output.with_suffix('.npz')
    if output.exists() or array_path.exists():
        raise FileExistsError('Choose a new output path.')
    payload = {'schema': 'thermal_saved_edit_organization_response_v1', 'family': args.case, 'action': 'baseline',
               'role': metadata['partition'], 'reference_metadata': metadata,
               'loader_only_validation_case': loader_case,
               'checkpoint': str(args.checkpoint.resolve()), 'epoch': int(saved['epoch']),
               'pack': str(args.pack.resolve()), 'read_mode': args.read_mode, 'device': args.device,
               'seed': args.seed, 'selected_cell_ids': selected.tolist(),
               'query_selection': 'Seeded input-only fluid sample, with inlet/outlet bands.',
               'within_role_access_prior': model.core.within_role_access_prior,
               'retained_mass_setting': float(model.core.retained_mass),
               'prepare_work': dict(prepared.work_counts), 'decode_work': dict(result.work_counts),
               'complete_seconds': time.perf_counter() - start, 'physical_solves': 0, 'fitting_epochs': 0,
               'arrays': array_path.name, 'capture': capture}
    output.parent.mkdir(parents=True, exist_ok=True)
    with array_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'case': args.case, 'partition': metadata['partition'], 'output': str(output),
                      'arrays': str(array_path), 'epoch': saved['epoch']}))


if __name__ == '__main__':
    main()
