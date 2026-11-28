#!/usr/bin/env python3
"""Evaluate a saved Thermal checkpoint on fixed historical response boards.

The preparation command records family/action identities from existing boards.
Evaluation reuses original configurations and physical outcomes, with no new
training or physical solves. Both readers' model choices are saved before the
physical outcome join.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / 'src', PROJECT / 'Case_ThermalChannel/src', PROJECT / 'tools'):
    sys.path.insert(0, str(directory))

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.unified_training import _atomic_json
from channelthermal.training.native_checkpoint import build_native_checkpoint_inference, original_heldout_case_ids
from channelthermal.saved_response_evaluation import family_board, family_board_inputs, select_saved_actions

ROLES = ('fit', 'original_withheld', 'historical_new_withheld', 'native_seen', 'fresh_layout')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, value)


def _actions_from_board(board, mode):
    rows = board['views'][mode]['actions']
    family_ids = list(map(str, board['family_ids']))
    if not family_ids or len(set(family_ids)) != len(family_ids):
        raise ValueError('Historical board families must be nonempty and unique.')
    result = {family: [str(row['action']) for row in rows if str(row['family']) == family]
              for family in family_ids}
    if (any(not actions or len(set(actions)) != len(actions) for actions in result.values())
            or sum(map(len, result.values())) != len(rows)):
        raise ValueError('Historical board actions must belong to exactly one family.')
    return result


def prepare_panel(reference_boards, historical_recipe, training_inputs, strict_protocol, strict_scales, *, source_paths):
    """Recover fixed case/action identities; do not select by current training IDs."""
    reference_boards = Path(reference_boards).resolve(strict=True)
    historical_recipe = Path(historical_recipe).resolve(strict=True)
    training_inputs = Path(training_inputs).resolve(strict=True)
    strict_protocol = Path(strict_protocol).resolve(strict=True)
    strict_scales = Path(strict_scales).resolve(strict=True)
    source_paths = {role: str(Path(source_paths[role]).resolve(strict=True)) for role in
                    ('historical_new_withheld', 'native_seen', 'fresh_layout')}
    recipe = _read(historical_recipe)
    recipe = recipe.get('recipe', recipe)
    roles = {}
    for role in ROLES:
        board_path = reference_boards / f'{role}.json'
        board = _read(board_path)
        if board.get('role') != role:
            raise ValueError(f'Historical {role} board has the wrong role.')
        restricted = _actions_from_board(board, 'restricted')
        if restricted != _actions_from_board(board, 'full'):
            raise ValueError(f'The historical {role} reader panels differ.')
        cases, _ = family_board_inputs(recipe, role, (), selected_families=tuple(restricted), source_paths=source_paths)
        if {case['family']: [item['action'] for item in case['actions']] for case in cases} != restricted:
            raise ValueError(f'The saved {role} input sources differ from the historical board.')
        roles[role] = {'source_board': str(board_path), 'historical_population_count': board['population_count'],
                       'families': [{'family': case['family'], 'source': case['source'],
                                     'baseline_path': case['baseline_path'],
                                     'actions': restricted[case['family']],
                                     'action_raw_paths': {item['action']: item['raw_path'] for item in case['actions']}}
                                    for case in cases]}
    family_count = sum(len(item['families']) for item in roles.values())
    action_count = sum(len(family['actions']) for item in roles.values() for family in item['families'])
    all_families = [family['family'] for item in roles.values() for family in item['families']]
    if family_count != 19 or action_count != 80 or len(set(all_families)) != family_count:
        raise ValueError('The reference must contain the established 80 edits and 19 distinct families.')
    old_strict = _read(reference_boards / 'strict.json')
    protocol = _read(strict_protocol)
    request = old_strict['request']
    matches = [item for item in protocol['requests'] if item['request_id'] == request['request_id']]
    if (len(matches) != 1 or matches[0] != request or len(request['candidate_ids']) != 7
            or request['baseline_action_id'] != request['candidate_ids'][0]
            or len(set(request['candidate_ids'])) != 7):
        raise ValueError('The saved strict request differs from its original seven-candidate protocol.')
    return {'schema': 'thermal_fixed_saved_response_panel_v1',
        'description': 'Historically exposed fixed response and inverse evaluation; no new physical solves.',
        'family_count': family_count, 'edit_count': action_count, 'roles': roles,
        'historical_recipe': str(historical_recipe), 'training_inputs': str(training_inputs),
        'source_paths': source_paths,
        'strict_protocol': str(strict_protocol), 'strict_scales_identity': str(strict_scales),
        'strict_request': request, 'strict_reference_board': str(reference_boards / 'strict.json')}


def native_exposure(panel, training_case_ids):
    """Describe actual native baseline membership separately from historical role."""
    training = set(map(str, training_case_ids))
    rows = []
    for role in ROLES:
        for family in panel['roles'][role]['families']:
            case_id = None if role == 'fresh_layout' else family['family']
            rows.append({'family': family['family'], 'historical_role': role, 'source': family['source'],
                         'native_case_id': case_id,
                         'baseline_in_current_native_training': None if case_id is None else case_id in training})
    return {'definition': 'Membership of the same original baseline case in this checkpoint native TRAIN set; '
                          'fresh layout is not a native case. Historical edited-scene FIT exposure is separate.',
            'families': rows}


def select_panel(panel, selections):
    """Select saved input identities without reading model or physical outcomes."""
    if set(selections) != set(ROLES):
        raise ValueError('Declare a family/action selection for every historical role.')
    result = copy.deepcopy(panel)
    if panel.get('schema') != 'thermal_fixed_saved_response_panel_v1':
        raise ValueError('Expected a saved response panel.')
    for role in ROLES:
        chosen = selections[role]
        families = panel['roles'][role]['families']
        source = {item['family']: item for item in families}
        if not chosen or not set(chosen) <= set(source):
            raise ValueError(f'The {role} selection must contain known saved families.')
        selected = []
        for family, actions in chosen.items():
            item = source[family]
            if (not actions or len(set(actions)) != len(actions)
                    or not set(actions) <= set(item['actions'])):
                raise ValueError(f'The {role}/{family} selection contains unknown or repeated actions.')
            selected.append({**item, 'actions': list(actions),
                             'action_raw_paths': {action: item['action_raw_paths'][action] for action in actions}})
        result['roles'][role]['families'] = selected
    result['family_count'] = sum(len(role['families']) for role in result['roles'].values())
    result['edit_count'] = sum(len(family['actions']) for role in result['roles'].values()
                               for family in role['families'])
    result['action_subset'] = True
    result['description'] = ('Explicit input-selected saved edits; each keeps its original baseline and historical role. '
                             'The separate strict request keeps its complete candidate board.')
    return result


def strict_request(model, checkpoint, epoch, panel, output, *, device):
    """Evaluate the saved seven-candidate request and persist choices before truth."""
    import torch
    from channelthermal.native_design import build_native_design, score_native_design

    protocol = _read(panel['strict_protocol'])
    request = panel['strict_request']
    if request not in protocol['requests']:
        raise ValueError('The fixed strict request is absent from its source protocol.')
    action_manifest_path = Path(protocol['action_manifest'])
    manifest = _read(action_manifest_path)
    case_id, actions = request['case_id'], request['candidate_ids']
    case = next(item for item in manifest['cases'] if item['case_id'] == case_id)
    centers = {item['action']: item['centers'] for item in manifest['actions']
               if item['case_id'] == case_id and item['action'] in actions}
    if not set(actions[1:]) <= set(centers):
        raise ValueError('The fixed strict request is missing input geometry.')
    case = {key: copy.deepcopy(case[key]) for key in
            ('source_config', 'baseline_centers', 'baseline_heat_powers', 'member_ids')}
    del manifest
    states, selections = {}, {}
    model.eval()
    model.core.query_tile_size = 1024
    with torch.inference_mode():
        for mode in ('restricted', 'full'):
            states[mode] = {}
            for action in actions:
                config = copy.deepcopy(case['source_config'])
                config['layout']['centers'] = (case['baseline_centers'] if action == request['baseline_action_id']
                                               else centers[action])
                config['layout']['heat_powers'] = case['baseline_heat_powers']
                design = build_native_design(config, case_id, member_ids=case['member_ids'], device=device)
                prediction = score_native_design(model, design, tau=protocol['tau'], mode=mode)
                states[mode][action] = {'functionals': {key: float(value) for key, value in
                                                        prediction['functionals'].items()},
                                        'work': prediction['work']}
            baseline = states[mode][request['baseline_action_id']]['functionals']
            deltas = {action: {key: states[mode][action]['functionals'][key] - baseline[key] for key in ('J', 'P')}
                      for action in actions}
            chosen = min((action for action in actions
                          if deltas[action][request['constraint']] <= request['constraint_max_delta']),
                         key=lambda action: (deltas[action][request['objective']],
                                             deltas[action][request['constraint']], action))
            selections[mode] = {'action': chosen, 'predicted_delta': deltas[chosen]}
    output = Path(output)
    prediction_path = output.with_name(output.stem + '_predictions.json')
    _write_new(prediction_path, {'checkpoint': str(checkpoint), 'request': request,
                                 'states': states, 'model_only_selections': selections})
    # Re-open physical outcomes only after all choices are persisted.
    physical_manifest = _read(action_manifest_path)
    entries = {(item['case_id'], item['action']): item for item in physical_manifest['actions']}
    physical = {request['baseline_action_id']: {'J': 0., 'P': 0.},
                **{action: entries[(case_id, action)]['physical_delta'] for action in actions[1:]}}
    scales = _read(panel['strict_scales_identity'])['provider_identity']['response_scales']
    views = {}
    for mode in ('restricted', 'full'):
        chosen = selections[mode]['action']
        objective, constraint, cap = request['objective'], request['constraint'], request['constraint_max_delta']
        best = min((action for action in actions if physical[action][constraint] <= cap),
                   key=lambda action: (physical[action][objective], physical[action][constraint], action))
        violation = physical[chosen][constraint] > cap
        accepted = request['baseline_action_id'] if violation else chosen
        regret = physical[accepted][objective] - physical[best][objective]
        raw_regret = None if violation else physical[chosen][objective] - physical[best][objective]
        views[mode] = {'selected_action': chosen, 'selected_predicted_delta': selections[mode]['predicted_delta'],
            'selected_physical_delta': physical[chosen], 'raw_constraint_violation': violation,
            'raw_physically_feasible': not violation,
            'raw_physically_improving': physical[chosen][objective] < 0.,
            'raw_numerically_resolved_improvement':
                physical[chosen][objective] < -4. * float(scales[objective]['floor']),
            'raw_feasible_regret': raw_regret,
            'raw_feasible_normalized_regret': None if violation else raw_regret / float(scales[objective]['scale']),
            'accepted_action': accepted, 'accepted_physical_delta': physical[accepted],
            'offline_truth_based_fallback_used': violation,
            'physical_optimum_action': best, 'physical_optimum_delta': physical[best],
            'accepted_feasible_regret': regret,
            'accepted_normalized_regret': regret / float(scales[objective]['scale']),
            'physical_candidates': physical}
    return {'population_request_count': len(protocol['requests']), 'sample_request_count': 1,
            'request': request, 'candidate_count': len(actions), 'epoch': int(epoch),
            'checkpoint': str(checkpoint), 'prediction_manifest': str(prediction_path), 'views': views,
            'selection_order': 'All predictions and model choices persisted before physical truth join',
            'fallback_interpretation': 'Offline fallback uses physical outcomes unavailable at deployment; '
                                       'it does not certify a safe deployed decision.'}


def evaluate(panel_path, checkpoint, output_dir, *, device, roles=None, include_strict=True):
    panel_path, checkpoint = Path(panel_path).resolve(strict=True), Path(checkpoint).resolve(strict=True)
    panel = _read(panel_path)
    subset = panel.get('action_subset') is True
    families = [family for role in panel.get('roles', {}).values() for family in role.get('families', ())]
    if (panel.get('schema') != 'thermal_fixed_saved_response_panel_v1'
            or set(panel.get('roles', {})) != set(ROLES)
            or panel.get('family_count') != len(families)
            or panel.get('edit_count') != sum(len(family['actions']) for family in families)
            or len({family['family'] for family in families}) != len(families)
            or (not subset and (len(families) != 19 or panel['edit_count'] != 80))):
        raise ValueError('Expected a valid fixed saved response panel.')
    requested = tuple(ROLES if roles is None else roles)
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(ROLES):
        raise ValueError('Select unique saved response roles.')
    historical = _read(panel['historical_recipe'])
    historical = historical.get('recipe', historical)
    for role in requested:
        families = panel['roles'][role]['families']
        cases, _ = family_board_inputs(historical, role, (),
            selected_families=tuple(item['family'] for item in families), source_paths=panel['source_paths'])
        if subset:
            cases = select_saved_actions(cases, {item['family']: item['actions'] for item in families})
        actual = [{'family': case['family'], 'source': case['source'], 'baseline_path': case['baseline_path'],
                   'actions': [item['action'] for item in case['actions']],
                   'action_raw_paths': {item['action']: item['raw_path'] for item in case['actions']}}
                  for case in cases]
        if actual != families:
            raise ValueError(f'The {role} physical input sources differ from the saved panel.')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty evaluation output directory.')
    saved = load_trusted_checkpoint(checkpoint, map_location='cpu')
    validation_case = original_heldout_case_ids(saved)[0]
    model, provider = build_native_checkpoint_inference(saved, validation_case_ids=(validation_case,), device=device)
    exposure = native_exposure(panel, provider.training_case_ids)
    exposure.update(checkpoint=str(checkpoint), epoch=int(saved['epoch']), panel=str(panel_path))
    _write_new(output / 'native_exposure.json', exposure)
    for role in requested:
        families = panel['roles'][role]['families']
        ids = tuple(item['family'] for item in families)
        expected = {item['family']: item['actions'] for item in families}
        destination = output / f'{role}.json'
        board = family_board(model, provider, historical, role, panel['training_inputs'], destination,
                             selected_families=ids, expected_actions=expected, source_paths=panel['source_paths'],
                             allow_action_subset=subset)
        board.update(checkpoint=str(checkpoint), epoch=int(saved['epoch']), fixed_panel=str(panel_path),
                     historical_population_count=panel['roles'][role]['historical_population_count'],
                     exposure_record=str(output / 'native_exposure.json'))
        _write_new(destination, board)
        print(json.dumps({'role': role, 'families': len(ids), 'status': 'complete'}), flush=True)
    if include_strict:
        strict_path = output / 'strict.json'
        strict = strict_request(model, checkpoint, saved['epoch'], panel, strict_path, device=device)
        strict['native_baseline_in_current_training'] = panel['strict_request']['case_id'] in set(map(str, provider.training_case_ids))
        _write_new(strict_path, strict)
    evaluated_families = sum(len(panel['roles'][role]['families']) for role in requested)
    evaluated_edits = sum(len(family['actions']) for role in requested for family in panel['roles'][role]['families'])
    _write_new(output / 'evaluation_summary.json', {'checkpoint': str(checkpoint), 'epoch': int(saved['epoch']),
        'panel': str(panel_path), 'roles': list(requested), 'family_count': evaluated_families,
        'edit_count': evaluated_edits, 'fixed_panel_family_count': panel['family_count'],
        'fixed_panel_edit_count': panel['edit_count'], 'action_subset': subset,
        'readers': ['restricted', 'full'],
        'strict_request': panel['strict_request']['request_id'] if include_strict else None, 'new_physical_solves': 0})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare = subparsers.add_parser('prepare', help='Record the exact saved family/action panel')
    prepare.add_argument('--reference-boards', type=Path, required=True)
    prepare.add_argument('--historical-recipe', type=Path, required=True)
    prepare.add_argument('--training-inputs', type=Path, required=True)
    prepare.add_argument('--strict-protocol', type=Path, required=True)
    prepare.add_argument('--strict-scales-identity', type=Path, required=True)
    prepare.add_argument('--historical-new-manifest', type=Path, required=True)
    prepare.add_argument('--native-seen-manifest', type=Path, required=True)
    prepare.add_argument('--fresh-layout-manifest', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    select = subparsers.add_parser('select', help='Select explicit saved family/action identities')
    select.add_argument('--panel', type=Path, required=True)
    select.add_argument('--selection', type=Path, required=True,
                        help='JSON mapping historical role to family IDs and lists of saved actions')
    select.add_argument('--output', type=Path, required=True)
    run = subparsers.add_parser('evaluate', help='Evaluate one checkpoint on both readers')
    run.add_argument('--panel', type=Path, required=True)
    run.add_argument('--checkpoint', type=Path, required=True)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--device', default='cpu')
    run.add_argument('--role', choices=ROLES, action='append', help='Evaluate only these saved roles; repeat as needed')
    run.add_argument('--skip-strict-request', action='store_true', help='Omit the separate saved seven-candidate request')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        panel = prepare_panel(args.reference_boards, args.historical_recipe, args.training_inputs,
            args.strict_protocol, args.strict_scales_identity, source_paths={
                'historical_new_withheld': args.historical_new_manifest, 'native_seen': args.native_seen_manifest,
                'fresh_layout': args.fresh_layout_manifest})
        _write_new(args.output, panel)
        print(json.dumps({'output': str(args.output), 'families': panel['family_count'],
                          'edits': panel['edit_count']}))
    elif args.command == 'select':
        panel = select_panel(_read(args.panel), _read(args.selection))
        _write_new(args.output, panel)
        print(json.dumps({'output': str(args.output), 'families': panel['family_count'],
                          'edits': panel['edit_count']}))
    else:
        evaluate(args.panel, args.checkpoint, args.output_dir, device=args.device,
                 roles=args.role, include_strict=not args.skip_strict_request)


if __name__ == '__main__':
    main()
