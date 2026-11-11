#!/usr/bin/env python3
"""Sampled direct-joint native screening for R16-10 saved checkpoints.

Family response/choice boards require their own explicit physical corpus reader;
this command intentionally does not invoke legacy full-cohort evaluators.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / 'src', PROJECT / 'Case_ThermalChannel/src', PROJECT / 'Case_WindFarm/src', PROJECT / 'tools'):
    sys.path.insert(0, str(directory))

import numpy as np
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.run_layout import resolve_checkpoint
from relation_anchored_train import build
from thermal_state_response_native_evaluate import evaluate_native_panel


def sampled_ids(ids, *, seed=20261108):
    """Select a reproducible input-only rounded ten percent in source order."""
    ids = tuple(map(str, ids))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Population IDs must be nonempty and unique')
    count = max(1, int(np.floor(.1 * len(ids) + .5)))
    selected = set(np.random.default_rng(seed).choice(len(ids), size=count, replace=False).tolist())
    return tuple(value for index, value in enumerate(ids) if index in selected)


def load_run(recipe, run, checkpoint, device):
    path = resolve_checkpoint(run, checkpoint)
    saved = load_trusted_checkpoint(path, map_location='cpu')
    identity = saved['experiment_identity']
    if identity['recipe'] != recipe:
        raise ValueError('Recipe differs from checkpoint')
    model, provider = build(recipe, device=device, resume_payload=saved)
    if identity['provider_identity'] != dict(provider.identity_payload()):
        raise ValueError('Provider identity differs from checkpoint')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.to(device).eval()
    restore = getattr(provider, 'load_training_state_dict', None)
    if callable(restore):
        restore(saved.get('provider_training_state', {}))
    return model, provider, saved, path


def evaluate_sampled_native(model, provider, *, epoch, split, case_ids, output):
    """Evaluate both access modes on the identical explicit ordered IDs."""
    case_ids = tuple(map(str, case_ids))
    available = provider.training_case_ids if split == 'train' else provider.validation_case_ids
    expected_count = max(1, int(np.floor(.1 * len(available) + .5))) if split == 'train' else 9
    if (len(case_ids) != expected_count or len(set(case_ids)) != len(case_ids)
            or not set(case_ids).issubset(available)):
        raise ValueError('Invalid sampled IDs or population size')
    return {mode: evaluate_native_panel(model, provider, epoch=epoch, split=split,
              case_ids=case_ids, mode=mode, export_cases=(), output=output)
            for mode in ('restricted', 'full')}


def cache_family_predictions(families, actions_for_family, predict, save):
    """Persist each unique baseline/edit prediction before any truth join.

    ``predict(family, endpoint, mode)`` returns a prediction-only payload;
    ``save(key, payload)`` persists it and returns its locator. The caller owns
    physical corpus adaptation, truth joining and request-level metrics.
    """
    families = tuple(map(str, families))
    if not families or len(set(families)) != len(families):
        raise ValueError('Sampled family IDs must be nonempty and unique')
    locators = {}
    for family in families:
        actions = tuple(actions_for_family(family))
        if not actions or len(set(actions)) != len(actions) or 'baseline' in actions:
            raise ValueError('Each family needs unique edit actions')
        for mode in ('restricted', 'full'):
            for endpoint in ('baseline', *actions):
                key = (family, endpoint, mode)
                locators[key] = save(key, predict(family, endpoint, mode))
    return locators


def calibrate_b0(model, provider, recipe, inputs):
    """Calibrate from input-only TRAIN panels; inputs contain eligible labels/scales."""
    from channelthermal.training.r16_10_diagnostics import calibrate_components, preserved_state
    from channelthermal.training.r16_5_decision_task import _parameter_groups
    from channelthermal.training.r16_10_family_task import R1610FamilyObjective
    from channelthermal.training.r16_8_corpus import ThermalR168Corpus
    from channelthermal.training.thermal_state_response_task import ThermalResponseCorpus, ACTIONS

    spec = recipe['r16_10_family']
    if spec['arm'] != 'B0':
        raise ValueError('Calibration requires the B0 checkpoint')
    original = ThermalResponseCorpus(spec['original_input_recipe'], spec['original_ledger'])
    new = ThermalR168Corpus(spec['new_fit_manifest'])
    labels = {(str(row['family']), row['action']): row['delta'] for row in inputs['original_labels']}
    objective = R1610FamilyObjective(provider, original, new, provider.training_case_ids, labels,
        response_scales=inputs['response_scales'], sensitivity=inputs['sensitivity'],
        research_scales=inputs['research_scales'])
    expected = {(family, action) for family in objective.schedule.families['original'] for action in ACTIONS}
    if set(labels) != expected:
        raise ValueError('Input labels must be exactly eligible original FIT actions')
    panel = tuple((source, family) for source in ('original', 'new') for family in objective.schedule.families[source])
    selected_keys = set(sampled_ids(tuple(f'{source}:{family}' for source, family in panel)))
    selected = tuple(item for item in panel if f'{item[0]}:{item[1]}' in selected_keys)
    native = sampled_ids(provider.training_case_ids)
    if len(native) != {150: 15, 600: 60}[int(spec['population'])] or len(selected) != {150: 4, 600: 14}[int(spec['population'])]:
        raise ValueError('Calibration population sizes do not match the declared protocol')

    def native_loss():
        for batch in provider.native_evaluation_batches(split='train', case_keys=native, microbatch_size=1):
            prediction, _ = provider.predict_native(model, batch.scene_inputs, batch.receivers,
                                                     model.architecture, 'joint', epoch=10)
            yield provider.endpoint_state_loss(prediction, batch) / len(native)

    def family_loss(component):
        def pieces():
            for source, family in selected:
                values, _ = objective.family_components(model, source, family)
                yield values[component] / len(selected)
        return pieces

    loss_fns = {'native': native_loss, **{key: family_loss(key) for key in ('endpoint', 'response', 'decision')}}
    with preserved_state(model):
        model.train()
        groups = {name: [key for key, _ in pairs] for name, pairs in _parameter_groups(model).items() if pairs}
        result = calibrate_components(model, loss_fns, groups)
    result.update(native_ids=native, family_ids=selected, native_population=len(provider.training_case_ids),
                  family_population=len(panel), sampling='seed20261108_input_only_random_source_order',
                  adapter_state=provider.identity_payload()['adapter_state'],
                  normalization_source=provider.identity_payload()['source_binding']['normalization_source'],
                  original_labels=inputs['original_labels'], response_scales=inputs['response_scales'],
                  sensitivity=inputs['sensitivity'], research_scales=inputs['research_scales'])
    return result


def family_board_inputs(recipe, role, native_ids):
    """Build input-only sampled action catalogues for the requested boards."""
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
        # Original and new IDs are disjoint by construction; intersect input membership.
        from channelthermal.training.r16_10_corpus import EligibleFamilySchedule
        schedule = EligibleFamilySchedule(original.family_ids('fit'), new.family_ids('fit'), native_ids)
        population = [(source, f) for source in ('original', 'new') for f in schedule.families[source]]
        selected = set(sampled_ids([f'{source}:{f}' for source, f in population]))
        for source, family in population:
            if f'{source}:{family}' not in selected:
                continue
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
        path = Path('/data2/wanglz/ModularDT/r16_9_reference_recovery_20261107/references') / f'{role}_corpus.json'
        corpus = ThermalR168Corpus(path)
        if corpus.scope != role or corpus.preview:
            raise ValueError('Unexpected physical corpus role or preview')
        population = tuple(corpus.cases)  # manifest source order
        for family in sampled_ids(population):
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
        return [original_row(f, 'response_holdout') for f in sampled_ids(population)], len(population)
    elif role == 'historical_new_withheld':
        path = Path('/data2/wanglz/ModularDT/r16_5_decision_fields_20261106/recipes/action_panel.json')
        data = json.loads(path.read_text())
        cases = {str(c['case_id']): c for c in data['cases']}
        population = tuple(cases)
        for family in sampled_ids(population):
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


def family_board(model, provider, recipe, role, input_path, output):
    """Save direct-joint predictions before opening raw endpoints and scoring."""
    import copy
    import torch
    from channelthermal.native_design import build_native_design, score_native_design
    from channelthermal.native_functionals import native_design_functionals
    from channelthermal.physical_response import summarize_actions
    from channelthermal.training.thermal_response_raw import load_raw_endpoint

    started = time.perf_counter()
    cases, population = family_board_inputs(recipe, role, provider.training_case_ids)
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
            for objective, constraint in (('J','P'), ('P','J')):
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
            family = case['family']; physical = {}; observations = {}
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
                for name in field_names:
                    bp,bt,bm = observations['baseline'][name]; ep,et,em = observations[action][name]
                    if bp.shape != ep.shape or bp.shape != bt.shape or ep.shape != et.shape:
                        raise ValueError('Response receiver identity differs')
                    mask = bm & em
                    if not mask.any(): raise ValueError('Empty response mask')
                    error = (ep-bp)-(et-bt)
                    field_errors[name] = {'rmse': float(np.sqrt(np.mean(np.square(error[mask])))), 'count': int(mask.sum())}
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
            computed = _research_decisions(pred_delta,true_delta,actions,sensitivity,research,scales)
            for request in computed:
                saved = next(x for x in model_choices if x['family'] == family and x['mode'] == mode
                             and x['objective'] == request['objective'] and x['multiplier'] == request['multiplier'])
                if request['model_selected_action'] != saved['selected_action']:
                    raise ValueError('Physical join changed the saved model-only selection')
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
            'mean_normalized_regret':float(np.mean([x['normalized_regret'] for x in shortlist]))}
        field_summary = {}
        for name in field_names:
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
    return {'schema':'thermal_r16_10_sampled_family_board_v1', 'role':role,
            'family_ids':[c['family'] for c in cases], 'family_count':len(cases), 'population_count':population,
            'units':{'J':'dataset temperature units', 'P':'dataset pressure units', 'flow/u':'dataset velocity units',
                     'flow/v':'dataset velocity units', 'flow/p':'dataset pressure units', 'flow/omega':'dataset vorticity units',
                     'temperature/fluid':'dataset temperature units', 'temperature/surface':'dataset temperature units',
                     'temperature/material':'dataset temperature units', 'q_proxy':'dataset q-proxy units'},
            'functional_resolution':{'original_sensitivity':sensitivity,'empirical_radius':{k:4.*v for k,v in sensitivity.items()}},
            'cost':{'views':cost, 'complete_seconds':time.perf_counter()-started,
                    'scope':'geometry construction, model preparation and decode, file writes, physical join and metrics'},
            'prediction_manifest':str(prediction_manifest), 'prediction_files':[v['path'] for v in predicted.values()], 'views':views,
            'selection':'input-only seed20261108; physical truth joined after prediction files saved'}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', default='latest')
    p.add_argument('--device', default='cpu')
    p.add_argument('--split', choices=('train', 'validation'))
    p.add_argument('--calibration-input', type=Path, help='Eligible TRAIN-only original labels and fitted scales JSON')
    p.add_argument('--board', choices=('fit', 'original_withheld', 'historical_new_withheld', 'native_seen', 'fresh_layout'))
    p.add_argument('--training-inputs', type=Path, help='Training-only scales for family boards')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    recipe = json.loads(args.recipe.read_text())
    model, provider, saved, path = load_run(recipe, args.run, args.checkpoint, args.device)
    if args.board:
        if args.training_inputs is None:
            p.error('--board requires --training-inputs')
        population = recipe['r16_10_family']['population']
        epoch = int(saved['epoch'])
        if population == 150:
            allowed = args.board in ('fit', 'original_withheld', 'historical_new_withheld') and epoch in ((100, 500, 1000) if args.board == 'fit' else (500, 1000))
        else:
            allowed = (population == 600 and epoch == 2000 and
                       recipe['r16_10_family']['arm'] in ('B0', 'R') and
                       path.name == 'epoch_2000_model.pt')
        if not allowed:
            raise ValueError('Board is outside the authorized milestone cadence')
        payload = family_board(model, provider, recipe, args.board, args.training_inputs, args.output)
        payload.update(checkpoint=str(path), epoch=int(saved['epoch']))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, args.output)
        print(json.dumps({'output': str(args.output), 'family_count': payload['family_count']}))
        return
    if args.calibration_input:
        if int(saved['epoch']) != 10:
            raise ValueError('Calibration requires literal B0 epoch 10')
        payload = calibrate_b0(model, provider, recipe, json.loads(args.calibration_input.read_text()))
        payload.update(schema='thermal_r16_10_calibration_v1', checkpoint=str(path), epoch=10)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, args.output)
        print(json.dumps({'output': str(args.output), 'coefficients': payload['coefficients']}))
        return
    if not args.split:
        p.error('--split is required for native evaluation')
    if args.split == 'train':
        population = tuple(provider.training_case_ids)
        cases = sampled_ids(population)
        denominator = len(population)
        sampling = 'seed20261108_input_only_random_source_order'
    else:
        # R16-10 provider already selects the fixed stratified VAL9 from VAL89.
        cases = tuple(provider.validation_case_ids)
        if len(cases) != 9 or provider.validation_scope != 'input_stratified_seed20261108_VAL9_of_89':
            raise ValueError('Expected the fixed R16-10 VAL9 panel')
        denominator = 89
        sampling = provider.validation_scope
    views = evaluate_sampled_native(model, provider, epoch=int(saved['epoch']), split=args.split,
                                    case_ids=cases, output=args.output)
    payload = {'schema': 'thermal_r16_10_sampled_native_v1', 'checkpoint': str(path),
               'epoch': int(saved['epoch']), 'split': args.split, 'case_ids': cases,
               'sample_count': len(cases), 'population_count': denominator,
               'sampling': sampling, 'views': views}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps({'output': str(args.output), 'count': len(cases), 'epoch': saved['epoch']}))


if __name__ == '__main__':
    main()
