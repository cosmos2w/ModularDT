#!/usr/bin/env python3
"""Evaluate a saved Thermal joint-field checkpoint without opening response data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / 'src', PROJECT / 'Case_ThermalChannel/src', PROJECT / 'tools'):
    sys.path.insert(0, str(directory))

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.unified_training import _atomic_json
from channelthermal.training.native_checkpoint import build_native_checkpoint_inference, original_heldout_case_ids
from thermal_state_response_native_evaluate import evaluate_native_panel, near_object_errors
import torch


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--cases', help='Comma-separated original validation IDs, at most nine')
    selection.add_argument('--all-original-held-out', action='store_true',
                           help='Explicitly evaluate the existing canonical 89 original held-out cases')
    parser.add_argument('--export-cases', default='', help='At most four IDs from the selected panel')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--query-tile-size', type=int, default=1024)
    parser.add_argument('--read-mode', choices=('restricted', 'full', 'both'), default='restricted')
    parser.add_argument('--near-object-errors', action='store_true',
                        help='Retain per-case physical fluid errors near module boundaries')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    saved = load_trusted_checkpoint(args.checkpoint, map_location='cpu')
    cases = original_heldout_case_ids(saved) if args.all_original_held_out else tuple(args.cases.split(','))
    exports = tuple(filter(None, args.export_cases.split(',')))
    if (not cases or len(set(cases)) != len(cases) or (not args.all_original_held_out and len(cases) > 9)
            or len(exports) > 4 or not set(exports) <= set(cases)):
        parser.error('Use unique canonical validation cases and at most four selected exports.')
    if args.query_tile_size < 1:
        parser.error('The inference query tile size must be positive.')
    model, provider = build_native_checkpoint_inference(saved, validation_case_ids=cases, device=args.device)
    model.core.query_tile_size = args.query_tile_size
    # The native provider applies its own inference tile from this working
    # evaluation recipe; changing the model field alone would be overwritten.
    provider.recipe['evaluation']['inference_query_tile_size'] = args.query_tile_size
    configs = {}
    if args.near_object_errors:
        import h5py
        with h5py.File(provider.data_path, 'r') as h5:
            configs = {case: json.loads(h5['cases'][case]['case_config_json'][()]) for case in cases}
    modes = ('restricted', 'full') if args.read_mode == 'both' else (args.read_mode,)
    views = {}
    for mode in modes:
        protected = {}
        def observe(case_id, prediction, batch):
            protected[case_id] = near_object_errors(prediction, batch, configs[case_id])
        destination = (args.output.with_name(f'{args.output.stem}_{mode}{args.output.suffix}')
                       if len(modes) > 1 else args.output)
        view = evaluate_native_panel(model, provider, epoch=int(saved['epoch']), split='validation',
            case_ids=cases, mode=mode, export_cases=exports, output=destination,
            case_observer=observe if args.near_object_errors else None)
        if args.near_object_errors:
            view['protected_near_object'] = {
                'definition': 'Valid fluid cell centres within two maximum cell widths outside any prescribed circular module boundary.',
                'per_case': protected,
            }
        view.update(schema='thermal_native_checkpoint_evaluation_v1', checkpoint=str(args.checkpoint.resolve()),
                    epoch=int(saved['epoch']), read_mode=mode,
                    inference_query_tile_size=args.query_tile_size,
                    native_normalization='Literal saved adapter state', response_binding_opened=False,
                    inference_only=True)
        views[mode] = view
    result = (views[modes[0]] if len(modes) == 1 else {
        'schema': 'thermal_native_checkpoint_both_readers_v1', 'checkpoint': str(args.checkpoint.resolve()),
        'epoch': int(saved['epoch']), 'case_ids': list(cases), 'case_count': len(cases), 'views': views,
        'interpretation': 'Historically exposed original held-out cases; not an independent TEST claim.',
    })
    device = torch.device(args.device)
    result['execution'] = {
        'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch_version': str(torch.__version__),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'process_seconds_before_json_write': time.perf_counter() - started,
    }
    if device.type == 'cuda':
        result['execution'].update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    _atomic_json(args.output, result)
    print(json.dumps({'output': str(args.output), 'epoch': int(saved['epoch']), 'cases': len(cases)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
