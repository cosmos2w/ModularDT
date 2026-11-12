#!/usr/bin/env python3
"""Evaluate the complete native Thermal observation panel of a saved run."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from thermal_state_response_evaluate import restore_model, select_evaluation_ids

from honf_runtime.contextual_frontier_evaluation import (
    _known_module_counts,
    _native_export_arrays,
    _summarize_nine_observables,
)


def evaluate_native_panel(model, provider, *, epoch, split, case_ids, mode, export_cases, output):
    """Use the maintained native receiver builder, metrics and reductions."""
    records, counts = [], {}
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), model.core.read_execution(mode):
        for batch in provider.native_evaluation_batches(split=split, case_keys=case_ids, microbatch_size=1):
            prediction, auxiliary = provider.predict_native(
                model, batch.scene_inputs, batch.receivers, model.architecture, 'joint', epoch=epoch)
            records.append(provider.validation_metrics(prediction, batch.targets, auxiliary))
            for name, value in provider.work_counts(batch, prediction, auxiliary).items():
                counts[name] = counts.get(name, 0.) + float(value)
            case_id = str(batch.targets.case_ids[0])
            if case_id in export_cases:
                arrays = _native_export_arrays(prediction, batch, case_id)
                arrays.update(fluid_xy=batch.receivers.fluid_xy[0].cpu().numpy(),
                              module_centers=batch.scene_inputs.structure['module_centers'][0].cpu().numpy(),
                              local_material_xy=batch.receivers.local_query_points[0].cpu().numpy(),
                              material_params=batch.scene_inputs.structure['material_params'][0].cpu().numpy())
                directory = output.parent / (output.stem + '_fields')
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / f'{case_id}.npz', **arrays)
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    reduced = provider.reduce_native_evaluation_metrics(records, split=split, case_keys=case_ids)
    return {'case_ids': list(case_ids), 'case_count': len(case_ids), 'split': split,
            'nine_observables': _summarize_nine_observables(
                reduced['per_case'], known_module_counts=_known_module_counts(provider)),
            'maintained_native_reduction': reduced, 'work_counts': counts,
            'complete_panel_seconds': time.perf_counter() - started,
            'timing_scope': 'case loading, complete native preparation/decode/extraction, metrics and selected exports',
            'receivers': ('all 8192 fluid grid cells with physical mask; '
                          f'{getattr(provider, "port_angle_count", 64)} surface angles; all stored material points'),
            'interpretation': 'Exposed development evidence; no independent TEST or design-reliability claim.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='latest')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'train'), default='validation')
    panel = parser.add_mutually_exclusive_group()
    panel.add_argument('--cases', help='Explicit comma-separated case IDs')
    panel.add_argument('--all-cases', action='store_true', help='Evaluate all available cases in this provider')
    parser.add_argument('--read-mode', choices=('restricted', 'full'), default='restricted')
    parser.add_argument('--export-cases', help='At most four explicit IDs from the evaluated panel')
    args = parser.parse_args(argv)
    recipe = json.loads(args.recipe.read_text())
    model, provider, saved, checkpoint = restore_model(recipe, args.run, args.checkpoint, args.device)
    available = provider.training_case_ids if args.split == 'train' else provider.validation_case_ids
    presampled = args.split == 'validation' and recipe.get('training_profile') == 'state_response'
    cases = select_evaluation_ids(available, requested=tuple(args.cases.split(',')) if args.cases is not None else None,
                                  all_ids=args.all_cases, presampled=presampled)
    exports = tuple(args.export_cases.split(',')) if args.export_cases else ()
    if (not cases or len(set(cases)) != len(cases) or not set(cases).issubset(available)
            or len(exports) > 4 or not set(exports).issubset(cases)):
        raise ValueError('Evaluation/export IDs must belong to the declared native population.')
    result = evaluate_native_panel(model, provider, epoch=int(saved['epoch']), split=args.split,
        case_ids=cases, mode=args.read_mode, export_cases=exports, output=args.output)
    result.update(schema='thermal_state_response_native_evaluation_v1', checkpoint=str(checkpoint),
                  epoch=int(saved['epoch']), recipe=str(args.recipe),
                  read_mode='all_native' if getattr(model.core, 'reader_capability', None) == 'all_native' else args.read_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps({'output': str(args.output), 'epoch': saved['epoch'], 'cases': len(cases)}))


if __name__ == '__main__':
    main()
