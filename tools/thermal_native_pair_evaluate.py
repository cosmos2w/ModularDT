#!/usr/bin/env python3
"""Measure native fields and input-defined finite changes on a fixed case panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / 'src', PROJECT / 'Case_ThermalChannel/src', PROJECT / 'tools'):
    sys.path.insert(0, str(directory))

import h5py
import numpy as np
import torch

from channelthermal.training.thermal_family_objective import _endpoint_functionals
from channelthermal.training.thermal_native_pair_task import (
    OBSERVABLES, TAU, ThermalNativePairObjective, describe_native_scene,
    native_input_distance, native_pair_references, native_pair_support,
)
from honf_runtime.unified_training import _atomic_json
from thermal_state_response_evaluate import restore_model
from thermal_state_response_native_evaluate import evaluate_native_panel


def summarize_changes(rows, scales):
    """Equal-pair physical errors and the predeclared four-observable aggregate."""
    if not rows:
        return {'available': False, 'pair_count': 0}
    error = np.asarray([row['change_error'] for row in rows], dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    if error.shape[1:] != (4,) or scale.shape != (4,) or np.any(scale <= 0):
        raise ValueError('Four positive native-TRAIN scales are required.')
    return {'available': True, 'pair_count': len(rows),
            'contributing_case_count': len({cid for row in rows for cid in row['pair']}),
            'normalized_change_rmse': float(np.sqrt(np.mean(np.square(error / scale)))),
            'observables': {name: {'mae': float(np.abs(error[:, i]).mean()),
                                   'rmse': float(np.sqrt(np.square(error[:, i]).mean())),
                                   'unit': 'dataset pressure' if name == 'P' else 'dataset temperature'}
                            for i, name in enumerate(OBSERVABLES)}}


def evaluate_pair_panel(model, provider, *, epoch, split, case_ids, pairs, scales, floors,
                        mode, export_cases, output):
    """Reuse each native prediction for its scalar and common-fluid diagnostics."""
    pairs = tuple(tuple(map(str, pair)) for pair in pairs)
    cases = tuple(map(str, case_ids))
    if len(set(pairs)) != len(pairs) or len(set(cases)) != len(cases):
        raise ValueError('Evaluation cases and pairs must be unique.')
    if any(len(pair) != 2 or pair[0] >= pair[1] or not set(pair) <= set(cases) for pair in pairs):
        raise ValueError('Oriented pairs must lie within this evaluation panel.')
    needed = {cid for pair in pairs for cid in pair}
    cache = {}
    with h5py.File(provider.data_path, 'r') as source:
        configs = {cid: json.loads(source['cases'][cid]['case_config_json'][()]) for cid in cases}
    for left, right in pairs:
        descriptions = [describe_native_scene(cid, split, configs[cid]) for cid in (left, right)]
        if descriptions[0].compatibility_key != descriptions[1].compatibility_key:
            raise ValueError('Pair domain, controls, material and module count must be compatible.')

    def observe(case_id, prediction, batch):
        if case_id not in needed:
            return
        raw = {'config': configs[case_id]}
        prepared = prediction.native_prepared
        material_valid = prepared.stencils['material'].valid.reshape_as(batch.targets.material_targets)
        required_material = (prepared.source_present[..., None].bool()
                             & torch.isfinite(batch.targets.material_targets))
        if not bool(material_valid[required_material].all()):
            raise ValueError('J requires every saved valid material receiver.')
        fluid_valid = prepared.stencils['fluid'].valid.reshape_as(batch.targets.point_weights)
        x = batch.receivers.fluid_xy[..., 0]
        bands = (x <= .08 * raw['config']['domain']['lx']) | (x >= .92 * raw['config']['domain']['lx'])
        pressure_required = (bands & batch.targets.point_weights.bool()
                             & torch.isfinite(batch.targets.field_targets[..., 2]))
        if not bool(fluid_valid[pressure_required].all()):
            raise ValueError('P requires every saved valid pressure-band receiver.')
        jp = _endpoint_functionals(prediction, batch, raw, TAU)
        if jp.shape != (1, 2):
            raise ValueError('Scientific inference must use exactly one declared read path.')
        field = prediction.native_main['pred_field'][0].detach().cpu().numpy()
        cache[case_id] = {
            'jp': jp[0].cpu().numpy(), 'field': field,
            'target': batch.targets.field_targets[0].cpu().numpy(),
            'mask': (batch.targets.point_weights[0].bool()
                     & prediction.native_prepared.stencils['fluid'].valid[0]).cpu().numpy(),
            'xy': batch.receivers.fluid_xy[0].cpu().numpy(),
        }

    model.eval()
    started = time.perf_counter()
    native = evaluate_native_panel(model, provider, epoch=epoch, split=split,
        case_ids=cases, mode=mode, export_cases=export_cases, output=output, case_observer=observe)
    references = native_pair_references(provider.data_path, pairs).cpu().numpy() if pairs else []
    rows = []
    with h5py.File(provider.data_path, 'r') as source:
        for index, pair in enumerate(pairs):
            left, right = (cache[cid] for cid in pair)
            if not np.array_equal(left['xy'], right['xy']):
                raise ValueError('Common residuals require identical physical coordinates.')
            regions = native_pair_support(source['cases'][pair[0]], source['cases'][pair[1]], configs[pair[0]])
            observations = []
            for scene in (left, right):
                if any(not bool(scene['mask'][regions[name].reshape(-1)].all()) for name in ('A', 'B')):
                    raise ValueError('Regional observations require every common physical receiver.')
                observations.append(np.r_[scene['jp'], [scene['field'][regions[name].reshape(-1), 4].astype(float).mean()
                                                       for name in ('A', 'B')]])
            values = np.asarray(observations)
            reference = references[index]
            delta = reference[1] - reference[0]
            error = (values[1] - values[0]) - delta
            common = left['mask'] & right['mask']
            residuals = []
            for component, name in enumerate(('u', 'v', 'p', 'omega', 'temperature')):
                mask = common & np.isfinite(left['target'][:, component]) & np.isfinite(right['target'][:, component])
                a = left['field'][mask, component].astype(float) - left['target'][mask, component].astype(float)
                b = right['field'][mask, component].astype(float) - right['target'][mask, component].astype(float)
                if not len(a):
                    raise ValueError('No common valid fluid positions for residual diagnostics.')
                common_mse = float(np.square((a + b) / 2).mean())
                change_mse = float(np.square(b - a).mean())
                endpoint_mse = float(((np.square(a) + np.square(b)) / 2).mean())
                residuals.append({'field': name, 'common_rmse': common_mse ** .5,
                                  'change_rmse': change_mse ** .5, 'equal_scene_rmse': endpoint_mse ** .5,
                                  'squared_error_identity_residual': endpoint_mse - common_mse - .25 * change_mse,
                                  'valid_positions': int(mask.sum())})
            config = configs[pair[0]]
            area = config['domain']['lx'] * config['domain']['ly'] / (config['domain']['nx'] * config['domain']['ny'])
            descriptions = [describe_native_scene(cid, split, configs[cid]) for cid in pair]
            changed = [label for key, label in (('centers', 'module positions'), ('heat_powers', 'prescribed module heats'))
                       if not np.array_equal(configs[pair[0]]['layout'][key], configs[pair[1]]['layout'][key])]
            rows.append({'pair': list(pair), 'prediction': values.tolist(), 'reference': reference.tolist(),
                         'reference_change': delta.tolist(), 'change_error': error.tolist(),
                         'near_zero_reference': (np.abs(delta) <= 4 * np.asarray(floors)).tolist(),
                         'near_zero_rule': 'Absolute native change no larger than four source-resolution floors',
                         'input_distance': native_input_distance(descriptions[0].descriptor, descriptions[1].descriptor),
                         'changed_inputs': changed,
                         'module_count': config['layout']['num_modules'], 'reynolds_number': config['flow']['re'],
                         'region_defining_case': pair[0],
                         'regional_physical_area': {name: float(regions[name].sum() * area) for name in ('A', 'B')},
                         'common_field_residuals': residuals})
    return {'native': native, 'responses': summarize_changes(rows, scales), 'pairs': rows,
            'observable_order': list(OBSERVABLES), 'native_train_scales': list(map(float, scales)),
            'source_resolution_floors': list(map(float, floors)), 'read_mode': mode, 'epoch': epoch,
            'complete_review_seconds': time.perf_counter() - started,
            'prediction_reuse': 'The same case predictions supply native metrics, four observables and common/change errors.',
            'interpretation': 'Matched measurements on the declared panel; no fixed-heat edit or inverse-design claim.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='latest')
    parser.add_argument('--panels', type=Path, required=True)
    parser.add_argument('--stage', choices=('development', 'formal'), required=True)
    parser.add_argument('--split', choices=('train', 'validation'), default='validation')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--read-mode', choices=('restricted', 'full'), default='restricted')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    recipe = json.loads(args.recipe.read_text())
    panel = json.loads(args.panels.read_text())[args.stage]
    model, provider, saved, checkpoint = restore_model(recipe, args.run, args.checkpoint, args.device)
    objective = getattr(provider, 'native_pair_objective', None)
    if objective is None:
        objective = ThermalNativePairObjective(provider, panel['training_pairs'])
    cases = tuple(panel[args.split + '_case_ids'])
    pairs = tuple(panel[args.split + '_pairs'])
    result = evaluate_pair_panel(model, provider, epoch=int(saved['epoch']), split=args.split,
        case_ids=cases, pairs=pairs, scales=objective.scales, floors=objective.floors,
        mode=args.read_mode, export_cases=cases[:1] if args.split == 'validation' else (), output=args.output)
    result.update(checkpoint=str(checkpoint), recipe=str(args.recipe), stage=args.stage)
    _atomic_json(args.output, result)
    print(json.dumps({'output': str(args.output), 'epoch': int(saved['epoch']), 'cases': len(cases),
                      'normalized_change_rmse': result['responses'].get('normalized_change_rmse')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
