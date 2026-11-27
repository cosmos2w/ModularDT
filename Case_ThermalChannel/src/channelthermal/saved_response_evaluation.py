"""Reusable saved-scene Thermal response and inverse-board evaluation.

The evaluator operates on explicit existing physical records and can preserve
a fixed family/action panel independently of current native TRAIN membership.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def sampled_ids(ids, *, seed=20261108):
    """Select a reproducible input-only rounded ten percent in source order."""
    ids = tuple(map(str, ids))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Population IDs must be nonempty and unique')
    count = max(1, int(np.floor(.1 * len(ids) + .5)))
    selected = set(np.random.default_rng(seed).choice(len(ids), size=count, replace=False).tolist())
    return tuple(value for index, value in enumerate(ids) if index in selected)


def family_board_inputs(recipe, role, native_ids, *, selected_families=None, source_paths=None):
    """Resolve input-only board cases, optionally using explicit family IDs.

    Explicit FIT families are evaluated independently of the current model's
    native TRAIN membership. Without an explicit selection, the historical
    input-only sampling and membership rules are unchanged.
    """
    import copy
    from channelthermal.training.thermal_state_response_task import ThermalResponseCorpus, ACTIONS
    spec = recipe['r16_10_family']
    original = ThermalResponseCorpus(spec['original_input_recipe'], spec['original_ledger'])
    catalog = []
    def original_row(family, source_role):
        case = original.cases[family]
        config = json.loads((Path(case['source_case_dir']) / 'case_config.json').read_text())
        return {'family': family, 'source': 'original', 'config': config,
                'baseline_path': case['source_case_dir'], 'module_count': case['module_count'],
                'actions': [{'action': a, 'centers': case['selection']['actions'][a],
                             'raw_path': original.action_entry(family, a, role=source_role)['raw_case_dir']}
                            for a in ACTIONS]}
    if role == 'fit':
        from channelthermal.training.r16_8_corpus import ThermalR168Corpus
        new = ThermalR168Corpus(spec['new_fit_manifest'])
        if selected_families is None:
            # Historical selection remains restricted to native TRAIN membership.
            from channelthermal.training.r16_10_corpus import EligibleFamilySchedule
            schedule = EligibleFamilySchedule(original.family_ids('fit'), new.family_ids('fit'), native_ids)
            population = [(source, f) for source in ('original', 'new') for f in schedule.families[source]]
            selected = set(sampled_ids([f'{source}:{f}' for source, f in population]))
            chosen = [(source, family) for source, family in population if f'{source}:{family}' in selected]
        else:
            population = [('original', f) for f in original.family_ids('fit')] + [
                ('new', f) for f in new.family_ids('fit')]
            sources = {family: source for source, family in population}
            selected = tuple(map(str, selected_families))
            if (len(sources) != len(population) or not selected or len(set(selected)) != len(selected)
                    or not set(selected) <= set(sources)):
                raise ValueError('Explicit FIT families must be unique, unambiguous source FIT IDs.')
            chosen = [(sources[family], family) for family in selected]
        for source, family in chosen:
            if source == 'original':
                catalog.append(original_row(family, 'fit'))
            else:
                case = new.cases[family]
                catalog.append({'family': family, 'source': 'new', 'config': case['source_config'],
                    'baseline_path': case['source_case_dir'],
                    'module_count': case['source_config']['layout']['num_modules'],
                    'actions': [{'action': a, 'centers': new.action_entry(family, a, role='fit')['centers'],
                                 'raw_path': new.action_entry(family, a, role='fit')['raw_case_dir']}
                                for a in new.actions_for_family(family)]})
        return catalog, len(population)
    elif role in ('native_seen', 'fresh_layout'):
        from channelthermal.training.r16_8_corpus import ThermalR168Corpus
        path = Path((source_paths or {})[role])
        corpus = ThermalR168Corpus(path)
        if corpus.scope != role or corpus.preview:
            raise ValueError('Unexpected physical corpus role or preview')
        population = tuple(corpus.cases)  # manifest source order
        selected = sampled_ids(population) if selected_families is None else tuple(map(str, selected_families))
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(population):
            raise ValueError(f'Selected families must belong to the {role} cohort.')
        for family in selected:
            case = corpus.cases[family]
            catalog.append({'family': family, 'source': role, 'config': case['source_config'],
                'baseline_path': case['source_case_dir'],
                'module_count': case['source_config']['layout']['num_modules'],
                'actions': [{'action': action,
                    'centers': corpus.action_entry(family, action, role=role)['centers'],
                    'raw_path': corpus.action_entry(family, action, role=role)['raw_case_dir']}
                    for action in corpus.actions_for_family(family)]})
        return catalog, len(population)
    elif role == 'original_withheld':
        population = original.family_ids('response_holdout')
        selected = sampled_ids(population) if selected_families is None else tuple(selected_families)
        if not set(selected) <= set(population) or len(set(selected)) != len(selected):
            raise ValueError('Selected families must belong to the original withheld cohort.')
        return [original_row(f, 'response_holdout') for f in selected], len(population)
    elif role == 'historical_new_withheld':
        path = Path((source_paths or {})[role])
        data = json.loads(path.read_text())
        cases = {str(c['case_id']): c for c in data['cases']}
        population = tuple(cases)
        selected = sampled_ids(population) if selected_families is None else tuple(selected_families)
        if not set(selected) <= set(population) or len(set(selected)) != len(selected):
            raise ValueError('Selected families must belong to the historical withheld cohort.')
        for family in selected:
            case = cases[family]
            actions = [a for a in data['actions'] if str(a['case_id']) == family and a['role'] == 'evaluation_only']
            if len(actions) != 6:
                raise ValueError('Historical withheld family must have six actions')
            catalog.append({'family': family, 'source': role, 'config': case['source_config'],
                'baseline_path': case['source_case_dir'], 'module_count': case['module_count'],
                'actions': [{'action': a['action'], 'centers': a['centers'], 'raw_path': a['raw_case_dir']} for a in actions]})
        return catalog, len(population)
    raise ValueError('Unknown board role')


def _research_decisions(predicted, physical, actions, sensitivity, research_scales, response_scales):
    """Model-only selection, then physical acceptance with baseline fallback."""
    requests = []
    for objective, constraint in (('J', 'P'), ('P', 'J')):
        for multiplier in (0., .5, 1., 1.5, 2.):
            cap = multiplier * float(research_scales[constraint])
            ranked = sorted((a for a in actions if predicted[a][constraint] <= cap),
                            key=lambda a: (predicted[a][objective], predicted[a][constraint], a))
            chosen = ranked[0]  # Baseline delta is zero and therefore always eligible.
            eligible = [a for a in actions if physical[a][constraint] <= cap]
            best = min(eligible, key=lambda a: (physical[a][objective], physical[a][constraint], a))
            admitted_physical = [a for a in ranked if physical[a][constraint] <= cap]
            best_admitted = min(admitted_physical, key=lambda a: (physical[a][objective], physical[a][constraint], a))
            violation = physical[chosen][constraint] > cap
            accepted = 'baseline' if violation else chosen
            requests.append({'objective': objective, 'constraint': constraint, 'multiplier': multiplier,
                'role': 'shortlist' if multiplier in (.5, 1.5) else 'training_band_diagnostic',
                'constraint_max_delta': cap, 'model_selected_action': chosen,
                'selected_predicted_delta': predicted[chosen], 'selected_physical_delta': physical[chosen],
                'raw_constraint_violation': violation,
                'objective_ordering_failure': not violation and physical[chosen][objective] > physical[best_admitted][objective],
                'best_admitted_physically_feasible_action': best_admitted,
                'physical_optimum_excluded_by_model': predicted[best][constraint] > cap,
                'false_feasible_actions': [a for a in actions[1:] if predicted[a][constraint] <= cap and physical[a][constraint] > cap],
                'false_rejected_actions': [a for a in actions[1:] if predicted[a][constraint] > cap and physical[a][constraint] <= cap],
                'near_null_false_rejected_actions': [a for a in actions[1:] if predicted[a][constraint] > cap and physical[a][constraint] <= cap and abs(physical[a][constraint]) <= float(sensitivity[constraint])],
                'raw_constraint_excess': max(0., physical[chosen][constraint] - cap),
                'accepted_action': accepted, 'raw_improvement': -physical[accepted][objective],
                'physically_improving': physical[accepted][objective] < 0,
                'numerically_resolved_improvement': physical[accepted][objective] < -4. * float(sensitivity[objective]),
                'physical_best_action': best,
                'regret': physical[accepted][objective] - physical[best][objective],
                'normalized_regret': (physical[accepted][objective] - physical[best][objective]) / float(response_scales[objective]),
                'resolved_objective': abs(physical[chosen][objective]) > float(sensitivity[objective])})
    return requests


def physical_response_error(baseline, edited, *, detailed=False, newly_valid_state=False):
    """Compare changes on shared valid positions, preserving zero-change context."""
    bp, bt, bm = map(np.asarray, baseline)
    ep, et, em = map(np.asarray, edited)
    if len({x.shape for x in (bp, bt, bm, ep, et, em)}) != 1:
        raise ValueError('Response receiver identity differs')
    mask = bm.astype(bool) & em.astype(bool)
    if not mask.any():
        raise ValueError('Empty response mask')
    if not detailed:
        error = (ep - bp) - (et - bt)
        if not np.isfinite(error[mask]).all():
            raise ValueError('Non-finite response on valid physical positions')
        return {'rmse': float(np.sqrt(np.mean(np.square(error[mask])))), 'count': int(mask.sum())}
    reference_change = et[mask].astype(np.float64) - bt[mask].astype(np.float64)
    predicted_change = ep[mask].astype(np.float64) - bp[mask].astype(np.float64)
    error = predicted_change - reference_change
    if not np.isfinite(error).all():
        raise ValueError('Non-finite response on valid physical positions')
    historical_error = ((ep - bp) - (et - bt))[mask]
    historical_rmse = float(np.sqrt(np.mean(np.square(historical_error))))
    rms = lambda value: float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))
    precision = lambda value: float(np.finfo(value.dtype).eps) if np.issubdtype(value.dtype, np.floating) else 0.
    reference_rms, predicted_rms = rms(reference_change), rms(predicted_change)
    reference_floor = precision(bt) * rms(bt[mask]) + precision(et) * rms(et[mask])
    predicted_floor = precision(bp) * rms(bp[mask]) + precision(ep) * rms(ep[mask])
    reference_resolved = reference_rms > reference_floor
    prediction_resolved = predicted_rms > predicted_floor
    result = {'rmse': historical_rmse, 'count': int(mask.sum()), 'zero_change_reference_rmse': reference_rms,
              'reference_change_rms': reference_rms, 'predicted_change_rms': predicted_rms,
              'reference_rounding_floor_rms': reference_floor, 'prediction_rounding_floor_rms': predicted_floor,
              'reference_resolved': reference_resolved, 'prediction_resolved': prediction_resolved,
              'relative_to_zero_change_rmse': historical_rmse / reference_rms if reference_resolved else None,
              'magnitude_ratio': predicted_rms / reference_rms if reference_resolved else None,
              'direction_cosine': (float(np.mean(reference_change * predicted_change)) /
                                   (reference_rms * predicted_rms)
                                   if reference_resolved and prediction_resolved else None)}
    if newly_valid_state:
        newly_valid = em.astype(bool) & ~bm.astype(bool)
        if newly_valid.any():
            difference = ep[newly_valid].astype(np.float64) - et[newly_valid].astype(np.float64)
            if not np.isfinite(difference).all():
                raise ValueError('Non-finite prediction on newly valid positions')
            result['newly_valid_edited_state'] = {'count': int(newly_valid.sum()), 'rmse': rms(difference)}
        else:
            result['newly_valid_edited_state'] = {'count': 0, 'rmse': None}
    return result


def family_board(model, provider, recipe, role, input_path, output, *, selected_families=None,
                 physical_only=False, expected_actions=None, source_paths=None):
    """Save direct-joint predictions before opening raw endpoints and scoring."""
    import copy
    import torch
    from channelthermal.native_design import build_native_design, score_native_design
    from channelthermal.native_functionals import native_design_functionals
    from channelthermal.physical_response import summarize_actions
    from channelthermal.training.thermal_response_raw import load_raw_endpoint

    started = time.perf_counter()
    cases, population = family_board_inputs(recipe, role, provider.training_case_ids,
        selected_families=selected_families, source_paths=source_paths)
    if expected_actions is not None:
        actual = {case['family']: [item['action'] for item in case['actions']] for case in cases}
        if actual != dict(expected_actions):
            raise ValueError('The resolved family/action identities differ from the fixed evaluation panel.')
    if physical_only:
        sensitivity = {'J': 0.0013635576450248266, 'P': 1.3318313563104311e-08}
        research, scales = {}, {}
    else:
        metadata = json.loads(Path(input_path).read_text())
        sensitivity, research = metadata['sensitivity'], metadata['research_scales']
        scales = {k: float(v['scale'] if isinstance(v, dict) else v) for k, v in metadata['response_scales'].items()}
    tau = .4149511814117432
    field_names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid',
                   'temperature/surface', 'q_proxy', 'temperature/material')
    output = Path(output)
    array_dir = output.parent / (output.stem + '_predictions')
    array_dir.mkdir(parents=True, exist_ok=True)
    predicted = {}
    cost = {mode: {'geometry_preparation_seconds':0., 'model_preparation_and_decode_seconds':0.} for mode in ('restricted','full')}
    device = next(model.parameters()).device
    model.eval(); model.core.query_tile_size = 1024
    # Only configurations and geometry are used on this pass, and every state is
    # persisted before opening the physical endpoint arrays below.
    with torch.inference_mode():
        for case in cases:
            family = case['family']
            for mode in ('restricted', 'full'):
                for item in [{'action': 'baseline', 'centers': case['config']['layout']['centers']}, *case['actions']]:
                    config = copy.deepcopy(case['config']); config['layout']['centers'] = item['centers']
                    t0 = time.perf_counter()
                    design = build_native_design(config, family, device=device)
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    cost[mode]['geometry_preparation_seconds'] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    result = score_native_design(model, design, tau=tau, mode=mode)
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    cost[mode]['model_preparation_and_decode_seconds'] += time.perf_counter() - t0
                    fields = {f'{name}__{kind}': value[kind].detach().cpu().numpy()
                              for name, value in result['fields'].items() for kind in ('prediction', 'mask')}
                    key = (family, item['action'], mode)
                    dest = array_dir / f"{family}_{item['action']}_{mode}.npz"
                    if dest.exists():
                        raise FileExistsError(dest)
                    np.savez_compressed(dest, **fields)
                    predicted[key] = {'path': str(dest), 'functionals': {k: float(v) for k,v in result['functionals'].items()},
                                      'work': result['work']}
    prediction_manifest = output.parent / (output.stem + '_prediction_manifest.json')
    if prediction_manifest.exists(): raise FileExistsError(prediction_manifest)
    model_choices = []
    for case in cases:
        family = case['family']
        actions = ['baseline'] + [item['action'] for item in case['actions']]
        for mode in ('restricted', 'full'):
            base = predicted[(family, 'baseline', mode)]['functionals']
            deltas = {a: {k: predicted[(family,a,mode)]['functionals'][k] - base[k] for k in ('J','P')} for a in actions}
            for objective, constraint in (() if physical_only else (('J','P'), ('P','J'))):
                for multiplier in (0., .5, 1., 1.5, 2.):
                    cap = multiplier * float(research[constraint])
                    selected = min((a for a in actions if deltas[a][constraint] <= cap),
                        key=lambda a: (deltas[a][objective], deltas[a][constraint], a))
                    model_choices.append({'family':family, 'mode':mode, 'objective':objective,
                                          'multiplier':multiplier, 'selected_action':selected})
    prediction_manifest.write_text(json.dumps({'states':[{'family':k[0], 'endpoint':k[1], 'mode':k[2], **v} for k,v in predicted.items()],
                                                'model_choices':model_choices}, indent=2, allow_nan=False) + '\n')
    views = {}
    for mode in ('restricted', 'full'):
        rows, decisions = [], []
        for case in cases:
            family = case['family']; physical = {}; observations = {}; baseline_fluid_xy = None
            entries = [{'action': 'baseline', 'centers': case['config']['layout']['centers'],
                        'raw_path': case['baseline_path']}, *case['actions']]
            for item in entries:
                raw = load_raw_endpoint(item['raw_path'])
                if (raw['config']['layout']['centers'] != item['centers']
                        or raw['config']['layout']['heat_powers'] != case['config']['layout']['heat_powers']
                        or raw['config']['domain'] != case['config']['domain']
                        or raw['config']['flow']['re'] != case['config']['flow']['re']):
                    raise ValueError('Raw endpoint geometry or prescribed physical controls differ')
                config = copy.deepcopy(case['config']); config['layout']['centers'] = item['centers']
                d = build_native_design(config, family)
                if baseline_fluid_xy is None:
                    baseline_fluid_xy = d.fluid_xy_source.copy()
                elif physical_only and not np.array_equal(baseline_fluid_xy, d.fluid_xy_source):
                    raise ValueError('Edited fluid coordinates differ from baseline coordinates')
                if not np.array_equal(raw['module_id_grid'], d.module_id_grid):
                    raise ValueError('Raw endpoint mask differs from model geometry')
                f = raw['field'].reshape(1, -1, 5)
                interface = raw['interface_target'][None]; material = raw['material'][:, raw['material_mask']][None]
                refs = {name: f[...,i] for i,name in enumerate(field_names[:5])}
                refs.update({'temperature/surface': interface[...,0], 'q_proxy':interface[...,1], 'temperature/material':material})
                masks = {name: d.fluid_valid[None] for name in field_names[:5]}
                masks.update({'temperature/surface': d.surface_valid[None],
                    'q_proxy': d.surface_valid[None] & d.outside_valid[None] & raw['outside_valid'][None],
                    'temperature/material': d.material_valid[None]})
                for name in field_names: masks[name] = masks[name] & np.isfinite(refs[name])
                j,p = native_design_functionals(torch.as_tensor(material, dtype=torch.float64), torch.as_tensor(masks['temperature/material']),
                    torch.as_tensor(f[...,2], dtype=torch.float64), torch.as_tensor(masks['flow/p']),
                    torch.as_tensor(d.fluid_xy_source[:,0][None], dtype=torch.float64), d.domain_length_x, tau)
                physical[item['action']] = {'J': float(j), 'P': float(p)}
                with np.load(predicted[(family,item['action'],mode)]['path'], allow_pickle=False) as arrays:
                    observations[item['action']] = {name: (arrays[f'{name}__prediction'], refs[name],
                         arrays[f'{name}__mask'] & masks[name]) for name in field_names}
            pred_delta = {'baseline': {'J': 0., 'P': 0.}}
            true_delta = {'baseline': {'J': 0., 'P': 0.}}
            base_pred = predicted[(family,'baseline',mode)]['functionals']
            for item in case['actions']:
                action = item['action']; current = predicted[(family,action,mode)]['functionals']
                pred_delta[action] = {k: current[k]-base_pred[k] for k in ('J','P')}
                true_delta[action] = {k: physical[action][k]-physical['baseline'][k] for k in ('J','P')}
                field_errors = {}
                for name in (field_names[:5] if physical_only else field_names):
                    field_errors[name] = physical_response_error(observations['baseline'][name], observations[action][name],
                                                                 detailed=expected_actions is not None,
                                                                 newly_valid_state=name in field_names[:5])
                functions = {k: {'prediction': pred_delta[action][k], 'physical': true_delta[action][k],
                    'absolute_error': abs(pred_delta[action][k]-true_delta[action][k]),
                    'resolved': abs(true_delta[action][k]) > sensitivity[k],
                    'resolved_sign_correct': bool(np.sign(pred_delta[action][k]) == np.sign(true_delta[action][k])) if abs(true_delta[action][k]) > sensitivity[k] else None,
                    'near_null': abs(true_delta[action][k]) <= sensitivity[k],
                    'empirical_radius': 4. * sensitivity[k],
                    'radius_resolved': abs(true_delta[action][k]) > 4. * sensitivity[k],
                    'radius_near_null': abs(true_delta[action][k]) <= 4. * sensitivity[k]} for k in ('J','P')}
                rows.append({'family':family, 'action':action, 'module_count':case['module_count'],
                             'field_response':field_errors, 'functionals':functions,
                             'work': {'baseline': predicted[(family,'baseline',mode)]['work'],
                                      'edited': predicted[(family,action,mode)]['work']}})
            actions = ['baseline'] + [item['action'] for item in case['actions']]
            computed = [] if physical_only else _research_decisions(pred_delta,true_delta,actions,sensitivity,research,scales)
            for request in computed:
                saved = next(x for x in model_choices if x['family'] == family and x['mode'] == mode
                             and x['objective'] == request['objective'] and x['multiplier'] == request['multiplier'])
                if request['model_selected_action'] != saved['selected_action']:
                    raise ValueError('Physical join changed the saved model-only selection')
                if expected_actions is not None:
                    objective, chosen = request['objective'], request['model_selected_action']
                    raw_feasible = not request['raw_constraint_violation']
                    raw_regret = (physical[chosen][objective] - physical[request['physical_best_action']][objective]
                                  if raw_feasible else None)
                    request.update(raw_physically_feasible=raw_feasible,
                        raw_physically_improving=physical[chosen][objective] < 0.,
                        raw_numerically_resolved_improvement=physical[chosen][objective] < -4. * sensitivity[objective],
                        raw_feasible_regret=raw_regret,
                        raw_feasible_normalized_regret=raw_regret / scales[objective] if raw_feasible else None,
                        offline_truth_based_fallback_used=request['raw_constraint_violation'])
            decisions += [{'family': family, **request} for request in computed]
        functional_summary = {}
        for name in ('J', 'P'):
            measures = [row['functionals'][name] for row in rows]
            resolved = [m for m in measures if m['resolved']]
            near_null = [m for m in measures if m['near_null']]
            family_maes = [np.mean([r['functionals'][name]['absolute_error'] for r in rows if r['family'] == c['family']]) for c in cases]
            functional_summary[name] = {'equal_family_mean_absolute_error': float(np.mean(family_maes)),
                'action_count': len(measures), 'resolved_count': len(resolved),
                'resolved_sign_correct': sum(int(m['resolved_sign_correct']) for m in resolved),
                'near_null_count': len(near_null),
                'near_null_action_mae': float(np.mean([m['absolute_error'] for m in near_null])) if near_null else None}
        shortlist = [x for x in decisions if x['role'] == 'shortlist']
        decision_summary = {'requests':len(shortlist),
            'raw_violations':sum(x['raw_constraint_violation'] for x in shortlist),
            'fallback_to_baseline':sum(x['accepted_action'] == 'baseline' and x['raw_constraint_violation'] for x in shortlist),
            'physically_improving_exact':sum(x['physically_improving'] for x in shortlist),
            'numerically_resolved_improving':sum(x['numerically_resolved_improvement'] for x in shortlist),
            'objective_ordering_failures':sum(x['objective_ordering_failure'] for x in shortlist),
            'physical_optimum_excluded_by_model':sum(x['physical_optimum_excluded_by_model'] for x in shortlist),
            'false_rejected_actions':sum(len(x['false_rejected_actions']) for x in shortlist),
            'near_null_false_rejections':sum(len(x['near_null_false_rejected_actions']) for x in shortlist),
            'mean_normalized_regret':float(np.mean([x['normalized_regret'] for x in shortlist])) if shortlist else None}
        if physical_only:
            decision_summary = None
        field_summary = {}
        for name in (field_names[:5] if physical_only else field_names):
            means = [float(np.mean([r['field_response'][name]['rmse'] for r in rows if r['family'] == c['family']])) for c in cases]
            field_summary[name] = {'equal_family_mean_rmse':float(np.mean(means)),
                                   'family_p90_rmse':float(np.quantile(means, .9)),
                                   'family_count':len(means)}
        summary_rows = [{'case_id':r['family'], 'delta':{k:{'model':r['functionals'][k]['prediction'],
            'physical':r['functionals'][k]['physical']} for k in ('J','P')}} for r in rows]
        family_ids = [c['family'] for c in cases]
        views[mode] = {'actions': rows, 'field_response_summary':field_summary, 'functionals':functional_summary, 'decisions': decisions,
            'functional_original_sensitivity':summarize_actions(summary_rows, sensitivity, family_ids=family_ids),
            'functional_empirical_radius':summarize_actions(summary_rows, {k:4.*v for k,v in sensitivity.items()}, family_ids=family_ids),
            'decision_summary':decision_summary}
    result = {'schema':'thermal_r16_10_sampled_family_board_v1', 'role':role,
            'family_ids':[c['family'] for c in cases], 'family_count':len(cases), 'population_count':population,
            'units':{'J':'dataset temperature units', 'P':'dataset pressure units', 'flow/u':'dataset velocity units',
                     'flow/v':'dataset velocity units', 'flow/p':'dataset pressure units', 'flow/omega':'dataset vorticity units',
                     'temperature/fluid':'dataset temperature units', 'temperature/surface':'dataset temperature units',
                     'temperature/material':'dataset temperature units', 'q_proxy':'dataset q-proxy units'},
            'functional_resolution':{'original_sensitivity':sensitivity,'empirical_radius':{k:4.*v for k,v in sensitivity.items()}},
            'cost':{'views':cost, 'complete_seconds':time.perf_counter()-started,
                    'scope':'geometry construction, model preparation and decode, file writes, physical join and metrics'},
            'prediction_manifest':str(prediction_manifest), 'prediction_files':[v['path'] for v in predicted.values()], 'views':views,
            'selection': ('explicit saved family panel; physical truth joined after prediction files saved'
                          if expected_actions is not None else
                          'declared R17-5 historical sample; physical truth joined after prediction files saved'
                          if selected_families is not None else
                          'input-only seed20261108; physical truth joined after prediction files saved')}
    if expected_actions is not None:
        result['field_response_definition'] = (
            'RMSE uses the historical stored-precision subtraction; direction and magnitudes use float64 '
            'differences on common valid physical receivers. The zero-change reference is the reference-change RMS. '
            'Newly valid fluid positions report edited-state RMSE separately. '
            'Surface and material observations retain their attached receiver positions.')
        result['decision_interpretation'] = (
            'Accepted-action fallback uses physical outcomes after prediction and does not certify deployed safety. '
            'Model-only selections and raw constraint violations are recorded separately.')
    return result
