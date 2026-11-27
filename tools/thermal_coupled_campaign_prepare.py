#!/usr/bin/env python3
"""Create matched fresh Thermal coupled-native recipes from saved source recipes.

The source supplies the actual native dataset, membership, pairs, observation
budgets and normalizer. No model is built, no fitting is performed and no data
or cache is copied by this recipe-only command.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def prepare_recipes(template, schedule_source, *, cohort, output_root, initial_state,
                    round_name, candidate_scope='organization_parameters',
                    max_output_bytes, peak_temporary_bytes, minimum_free_bytes):
    """Return matched recipes whose only learning difference is response scope."""
    expected = {'train150': ('fixed25_train150_dev22', 1000, 47, 15, False),
                'train600': ('full_train600', 500, 240, 60, True)}
    if cohort not in expected:
        raise ValueError('Choose the train150 or train600 cohort.')
    membership, horizon, pair_count, calibration_count, formal = expected[cohort]
    source = copy.deepcopy(template)
    observation = source.get('observation_response', {})
    initialization = source.get('initialization', {})
    training = source.get('training', {})
    if (source.get('training_profile') != 'observation_response'
            or observation.get('mode') != 'coupled_native'
            or training.get('membership') != membership
            or len(observation.get('pairs', ())) != pair_count
            or len(observation.get('calibration_case_ids', ())) != calibration_count
            or initialization.get('source_checkpoint') is not None
            or initialization.get('pretrained_weights') is not False
            or initialization.get('source_parent_age') != 0):
        raise ValueError('The source is not the expected fresh coupled-native cohort recipe.')
    optimizer = schedule_source['optimizer']
    if (optimizer.get('schedule') != 'hold1000_cosine2000'
            or float(optimizer.get('lr', 0)) != 3e-4
            or float(optimizer.get('final_lr', 0)) != 3e-5
            or schedule_source['training'].get('optimizer_schedule_horizon') != 2000):
        raise ValueError('The supplied source does not contain the expected saved R17-4 schedule.')
    if (candidate_scope not in ('all_parameters', 'organization_parameters')
            or not round_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in round_name)):
        raise ValueError('Provide a descriptive round name and supported candidate scope.')
    if any(type(value) is not int or value <= 0 for value in
           (max_output_bytes, peak_temporary_bytes, minimum_free_bytes)):
        raise ValueError('Output, temporary and free-space limits must be positive bytes.')
    output_root, initial_state = Path(output_root).expanduser().absolute(), Path(initial_state).expanduser().absolute()
    source['optimizer']['schedule'] = optimizer['schedule']
    source['optimizer']['lr'] = optimizer['lr']
    source['optimizer']['final_lr'] = optimizer['final_lr']
    training.update(local_horizon=horizon, first_segment_epochs=100, optimizer_schedule_horizon=2000,
                    checkpoint_epochs=list(range(100, horizon + 1, 100)), curve_every=10)
    source['member_routing']['model']['query_tile_size'] = 512
    source['execution']['precision'] = 'fp32'
    source.setdefault('execution_config', {})['native_retention_fraction'] = 0.0
    source['storage'].update(output_root=str(output_root), max_output_bytes=max_output_bytes,
        max_peak_temporary_bytes=peak_temporary_bytes, minimum_free_headroom_bytes=minimum_free_bytes,
        formal_forward_run_alias=formal)
    observation.update(epoch_zero_checkpoint=str(initial_state), reader_policy='sampled_equal_probability',
                       full_reader_probability=0.5)
    observation.pop('response_weight', None)
    suffix = f'_Train600_{round_name}' if formal else f'_{round_name}'
    scopes = [('Native_State_Reference' + suffix, 'disabled')]
    if formal:
        stem = 'Organization_Focused_Response' if candidate_scope == 'organization_parameters' else 'All_Parameter_Response'
        scopes.append((stem + suffix, candidate_scope))
    else:
        scopes.extend([('All_Parameter_Response' + suffix, 'all_parameters'),
                       ('Organization_Focused_Response' + suffix, 'organization_parameters')])
    result = {}
    for name, scope in scopes:
        recipe = copy.deepcopy(source)
        recipe['name'] = name
        options = recipe['observation_response']
        options['variant'] = 'native_control' if scope == 'disabled' else 'observed_response'
        options['response_update_scope'] = scope
        options.pop('response_gradient_composition', None)
        if scope != 'disabled':
            options['response_gradient_composition'] = {
                'kind': 'native_priority_projection', 'maximum_native_norm_ratio': 0.1}
        result[name] = recipe
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--schedule-source', type=Path, required=True)
    parser.add_argument('--cohort', choices=('train150', 'train600'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--recipe-dir', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--round-name', required=True)
    parser.add_argument('--candidate-scope', choices=('organization_parameters', 'all_parameters'),
                        default='organization_parameters')
    parser.add_argument('--max-output-bytes', type=int, required=True)
    parser.add_argument('--peak-temporary-bytes', type=int, required=True)
    parser.add_argument('--minimum-free-bytes', type=int, required=True)
    args = parser.parse_args(argv)
    recipes = prepare_recipes(json.loads(args.template.read_text()), json.loads(args.schedule_source.read_text()),
        cohort=args.cohort, output_root=args.output_root, initial_state=args.initial_state,
        round_name=args.round_name, candidate_scope=args.candidate_scope, max_output_bytes=args.max_output_bytes,
        peak_temporary_bytes=args.peak_temporary_bytes, minimum_free_bytes=args.minimum_free_bytes)
    args.recipe_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.recipe_dir / (name + '.json') for name in recipes]
    if any(path.exists() for path in paths):
        raise FileExistsError('Preserve existing recipes; choose a new recipe directory.')
    for path, recipe in zip(paths, recipes.values()):
        path.write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'recipes': [str(path) for path in paths], 'initial_state': str(args.initial_state),
                      'epochs_per_model': 1000 if args.cohort == 'train150' else 500}))


if __name__ == '__main__':
    main()
