#!/usr/bin/env python3
"""Evaluate a historical Thermal joint-field checkpoint without response data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / 'src', PROJECT / 'Case_ThermalChannel/src', PROJECT / 'tools'):
    sys.path.insert(0, str(directory))

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.unified_training import _atomic_json
from channelthermal.training.native_checkpoint import build_native_checkpoint_inference
from thermal_state_response_native_evaluate import evaluate_native_panel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', required=True, help='Comma-separated original validation IDs')
    parser.add_argument('--export-cases', default='', help='At most four IDs from the selected panel')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--query-tile-size', type=int, default=1024)
    parser.add_argument('--read-mode', choices=('restricted', 'full'), default='restricted')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    cases = tuple(args.cases.split(','))
    exports = tuple(filter(None, args.export_cases.split(',')))
    if not 1 <= len(cases) <= 9 or len(exports) > 4 or not set(exports) <= set(cases):
        parser.error('Use at most nine canonical validation cases and four selected exports.')
    saved = load_trusted_checkpoint(args.checkpoint, map_location='cpu')
    model, provider = build_native_checkpoint_inference(saved, validation_case_ids=cases, device=args.device)
    if args.query_tile_size < 1:
        parser.error('The inference query tile size must be positive.')
    model.core.query_tile_size = args.query_tile_size
    result = evaluate_native_panel(model, provider, epoch=int(saved['epoch']), split='validation',
        case_ids=cases, mode=args.read_mode, export_cases=exports, output=args.output)
    result.update(schema='thermal_native_checkpoint_evaluation_v1', checkpoint=str(args.checkpoint.resolve()),
                  epoch=int(saved['epoch']), read_mode=args.read_mode,
                  inference_query_tile_size=args.query_tile_size,
                  native_normalization='Literal saved adapter state', response_binding_opened=False,
                  inference_only=True)
    _atomic_json(args.output, result)
    print(json.dumps({'output': str(args.output), 'epoch': int(saved['epoch']), 'cases': len(cases)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
