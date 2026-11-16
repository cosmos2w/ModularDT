#!/usr/bin/env python3
"""Measure native fields and physical field changes on a declared case panel."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    from .thermal_native_pair_evaluate import evaluate_pair_panel
    from .thermal_state_response_native_evaluate import evaluate_native_panel
    from .thermal_state_response_evaluate import restore_model
except ImportError:
    from thermal_native_pair_evaluate import evaluate_pair_panel
    from thermal_state_response_native_evaluate import evaluate_native_panel
    from thermal_state_response_evaluate import restore_model

import h5py
import numpy as np
import torch

from channelthermal.native_observations import native_cell_center_stencil
from channelthermal.training.thermal_native_pair_task import ThermalNativePairObjective
from channelthermal.training.thermal_field_response import sample_thermal_field_observations
from honf_runtime.state_response import FieldResponsePair, field_response_losses
from honf_runtime.unified_training import _atomic_json

CHANNELS = ('u', 'v', 'p', 'omega', 'temperature')
UNITS = ('dataset velocity', 'dataset velocity', 'dataset pressure',
         'dataset vorticity', 'dataset temperature')
ASSESSMENT9_CASES = ('0686', '0687', '0645', '0646', '0281', '0283', '0284', '0294', '0297')
ASSESSMENT9_PAIRS = (('0686', '0687'), ('0645', '0646'), ('0283', '0284'), ('0294', '0297'))


def bind_assessment9(provider):
    """Bind the exposed original-test panel to a shallow evaluation-only provider."""
    if any(provider.case_split_by_id.get(case_id) != 'test' for case_id in ASSESSMENT9_CASES):
        raise ValueError('Assessment9 must contain only the declared original test records.')
    if set(ASSESSMENT9_CASES) & set(provider.training_case_ids):
        raise ValueError('Assessment9 cannot change or overlap training membership.')
    bound = copy.copy(provider)
    bound.validation_case_ids = ASSESSMENT9_CASES
    bound._validation_ids = ASSESSMENT9_CASES
    bound._validation_index_by_id = {case_id: i for i, case_id in enumerate(ASSESSMENT9_CASES)}
    bound._validation_batches_cache = None
    return bound


def summarize_physical_field_changes(rows, scales, weights):
    """Average per-pair physical RMSE within each channel, then combine channels."""
    scales, weights = np.asarray(scales, dtype=float), np.asarray(weights, dtype=float)
    if (scales.shape != (5,) or weights.shape != (5,) or not np.isfinite(scales).all()
            or not np.isfinite(weights).all() or np.any(scales <= 0) or np.any(weights < 0)):
        raise ValueError('Five positive TRAIN scales and nonnegative native weights are required.')
    components = {}
    for index, name in enumerate(CHANNELS):
        eligible = [next(item for item in row['common_field_residuals'] if item['field'] == name)
                    for row in rows]
        eligible = [item for item in eligible if item['valid_positions'] > 0]
        if not eligible:
            components[name] = {'available': False, 'eligible_pairs': 0, 'unit': UNITS[index]}
            continue
        rmse = float(np.mean([item['change_rmse'] for item in eligible]))
        normalized_mse = float(np.mean([(item['change_rmse'] / scales[index])**2 for item in eligible]))
        components[name] = {'available': True, 'eligible_pairs': len(eligible), 'unit': UNITS[index],
            'mean_pair_change_rmse': rmse,
            'mean_pair_change_mae': float(np.mean([item['change_mae'] for item in eligible])),
            'normalized_rmse': rmse / scales[index], 'normalized_change_mse': normalized_mse,
            'weighted_normalized_change_mse': weights[index] * normalized_mse,
            'native_weight': float(weights[index]), 'native_train_scale': float(scales[index]),
            'mean_pair_common_rmse': float(np.mean([item['common_rmse'] for item in eligible]))}
    denominator = sum(weights[i] for i, name in enumerate(CHANNELS) if components[name]['available'])
    score = (sum(weights[i] * components[name]['normalized_rmse'] for i, name in enumerate(CHANNELS)
                 if components[name]['available']) / denominator if denominator > 0 else None)
    return {'available': score is not None, 'D_delta': float(score) if score is not None else None,
            'channels': components,
            'definition': 'Cell-area RMSE for each pair/channel, equal-pair mean within channel, native-weighted normalized channel mean; not a pooled MSE square root.'}


def evaluate_field_response_panel(model, provider, *, epoch, split, case_ids, pairs,
                                  mode, export_cases=(), output, include_thermal_functionals=False):
    """Reuse the complete physical-cell predictions for field and case measurements."""
    scales = provider.model.field_std.detach().cpu().reshape(-1).double().numpy()[:5]
    weights = np.asarray([provider.native_weights[name] for name in
        ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid')])
    cache, targets, masks, coordinates, protected = {}, {}, {}, {}, {}
    with h5py.File(provider.data_path, 'r') as h5:
        configs = {case_id: json.loads(h5['cases'][case_id]['case_config_json'][()]) for case_id in case_ids}

    def observe(case_id, prediction, batch):
        field = prediction.native_main['pred_field'][0].detach().cpu()
        target = batch.targets.field_targets[0].detach().cpu()
        valid = (batch.targets.point_weights[0].bool()
                 & prediction.native_prepared.stencils['fluid'].valid[0]).cpu()
        xy = batch.receivers.fluid_xy[0].cpu().double().numpy()
        config = configs[case_id]
        domain = config['domain']
        centers = np.asarray(config['layout']['centers'])
        distance = np.linalg.norm(xy[:, None] - centers[None], axis=-1).min(1) - domain['module_radius']
        width = 2 * max(domain['lx'] / domain['nx'], domain['ly'] / domain['ny'])
        near = (distance >= 0) & (distance <= width) & valid.numpy()
        protected[case_id] = {}
        for i, name in enumerate(CHANNELS):
            mask = near & torch.isfinite(target[:, i]).numpy()
            if mask.any():
                error = field[mask, i].double().numpy() - target[mask, i].double().numpy()
                protected[case_id][name] = {'rmse': float(np.sqrt(np.mean(error**2))),
                    'absolute_error_p95': float(np.quantile(np.abs(error), .95)),
                    'valid_positions': int(mask.sum()), 'unit': UNITS[i]}
        cache[case_id] = field
        targets[case_id], masks[case_id], coordinates[case_id] = target, valid, xy

    # These optional Thermal summaries retain their existing definitions; they
    # are neither required channels of the shared interface nor training terms.
    if include_thermal_functionals:
        optional = ThermalNativePairObjective(provider, provider.field_response_pairs)
        result = evaluate_pair_panel(model, provider, epoch=epoch, split=split,
            case_ids=case_ids, pairs=pairs, scales=optional.scales, floors=optional.floors,
            mode=mode, export_cases=export_cases, output=output, case_observer=observe,
            allow_missing_field_channels=True)
        for index, name in enumerate(('J', 'P', 'A', 'B')):
            resolved = [row for row in result['pairs'] if not row['near_zero_reference'][index]]
            null = [row for row in result['pairs'] if row['near_zero_reference'][index]]
            metrics = result['responses'].get('observables', {}).get(name)
            if metrics is not None:
                metrics.update(resolved_sign_total=len(resolved),
                    resolved_sign_correct=sum(int(np.sign(row['prediction'][1][index] - row['prediction'][0][index])
                        == np.sign(row['reference_change'][index])) for row in resolved),
                    near_null_pairs=len(null),
                    near_null_change_mae=(float(np.mean([abs(row['change_error'][index]) for row in null]))
                                          if null else None))
    else:
        native = evaluate_native_panel(model, provider, epoch=epoch, split=split,
            case_ids=case_ids, mode=mode, export_cases=export_cases, output=output, case_observer=observe)
        rows = []
        for left, right in pairs:
            if not np.array_equal(coordinates[left], coordinates[right]):
                raise ValueError('Field changes require identical physical cell centres.')
            residuals = []
            for i, name in enumerate(CHANNELS):
                mask = masks[left] & masks[right] & torch.isfinite(targets[left][:, i]) & torch.isfinite(targets[right][:, i])
                if not bool(mask.any()):
                    residuals.append({'field': name, 'available': False, 'valid_positions': 0})
                    continue
                a = cache[left][mask, i].double() - targets[left][mask, i].double()
                b = cache[right][mask, i].double() - targets[right][mask, i].double()
                residuals.append({'field': name, 'valid_positions': int(mask.sum()),
                    'change_rmse': float((b-a).square().mean().sqrt()),
                    'change_mae': float((b-a).abs().mean()),
                    'common_rmse': float(((a+b)/2).square().mean().sqrt())})
            rows.append({'pair': [left, right], 'common_field_residuals': residuals})
        result = {'native': native, 'pairs': rows, 'thermal_functionals': {'available': False, 'reason': 'Not requested.'}}
    result['physical_field_response'] = summarize_physical_field_changes(result['pairs'], scales, weights)
    result['protected_near_object'] = {'definition': 'Valid fluid cell centres within two maximum cell widths outside any prescribed circular module boundary.',
                                       'per_case': protected}
    observations = []
    with h5py.File(provider.data_path, 'r') as h5:
        for pair in pairs:
            groups = [h5['cases'][case_id] for case_id in pair]
            domain = configs[pair[0]]['domain']
            observation = sample_thermal_field_observations(groups, (domain['lx'], domain['ly']),
                seed=0, sample_per_side=max(len(group['sampled_points']) for group in groups))
            stencil = native_cell_center_stencil(observation.coordinates[None],
                torch.tensor([[domain['lx'], domain['ly']]], dtype=torch.float64),
                nx=domain['nx'], ny=domain['ny'])
            predicted = torch.stack([(cache[case_id][stencil.indices[0]].double()
                * stencil.weights[0][..., None]).sum(1) for case_id in pair])[None]
            observations.append(FieldResponsePair(predicted, observation.references,
                                                   observation.mask, observation.masses))
    if observations:
        reduced = field_response_losses(observations, scales, weights)
        result['saved_receiver_measure'] = {name: float(reduced[name]) for name in ('absolute', 'difference')}
        result['saved_receiver_measure']['definition'] = ('Exact symmetric finite population of both saved receiver lists, retaining duplicate mass; predictions interpolated from the already measured physical-cell fields. Distinct from the cell-area primary score.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--checkpoint', default='latest')
    parser.add_argument('--panels', type=Path)
    parser.add_argument('--assessment9', action='store_true', help='Use the fixed historically exposed nine-case assessment.')
    parser.add_argument('--stage', required=True, choices=('development', 'formal'))
    parser.add_argument('--split', default='validation', choices=('train', 'validation'))
    parser.add_argument('--read-mode', default='restricted', choices=('restricted', 'full'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--export-case', action='append', default=[])
    parser.add_argument('--thermal-functionals', action='store_true',
                        help='Also measure optional case-owned J/P and regional-temperature changes.')
    args = parser.parse_args(argv)
    recipe = json.loads(args.recipe.read_text())
    if args.assessment9:
        if args.panels is not None or args.split != 'validation':
            parser.error('--assessment9 requires validation and excludes --panels')
        cases, pairs = ASSESSMENT9_CASES, ASSESSMENT9_PAIRS
    else:
        if args.panels is None:
            parser.error('--panels is required unless --assessment9 is selected')
        panels = json.loads(args.panels.read_text())[args.stage]
        cases, pairs = panels[args.split + '_case_ids'], panels[args.split + '_pairs']
    if len(args.export_case) > 4 or not set(args.export_case) <= set(cases):
        raise ValueError('Export at most four declared panel cases.')
    model, provider, saved, checkpoint = restore_model(recipe, args.run, args.checkpoint, args.device)
    if args.assessment9:
        provider = bind_assessment9(provider)
    model.core.query_tile_size = recipe['evaluation'].get('inference_query_tile_size', 1024)
    result = evaluate_field_response_panel(model, provider, epoch=saved['epoch'], split=args.split,
        case_ids=cases, pairs=pairs, mode=args.read_mode, export_cases=args.export_case, output=args.output,
        include_thermal_functionals=args.thermal_functionals)
    result.update(checkpoint=str(checkpoint), recipe=str(args.recipe), read_mode=args.read_mode,
                  epoch=saved['epoch'], stage=args.stage, evaluation_case_fraction='approximately 10% fixed IDs')
    if args.assessment9:
        result.update(panel='historically_exposed_assessment9', original_h5_split='test',
                      exposure='historically exposed development evidence; not independent TEST',
                      selection='fixed stratified panel; not a population estimate')
    _atomic_json(args.output, result)
    print(json.dumps({'output': str(args.output), 'epoch': saved['epoch'],
                      'D_delta': result['physical_field_response']['D_delta']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
