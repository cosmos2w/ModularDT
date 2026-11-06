#!/usr/bin/env python3
"""Evaluate saved R16 physical edit responses at a trained checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / 'src', PROJECT / 'Case_ThermalChannel/src', PROJECT / 'Case_WindFarm/src', PROJECT / 'tools'):
    sys.path.insert(0, str(directory))

import numpy as np
import torch

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.run_layout import resolve_checkpoint
from relation_anchored_train import build
from channelthermal.native_functionals import native_design_functionals
from channelthermal.physical_response import compare_native_response, summarize_actions
from channelthermal.training.thermal_state_response_task import (
    ACTIONS, ThermalResponseCorpus, _field_components, load_saved_functional_responses, response_pair,
)

def functional_label_path(section, role):
    root = Path(section['recipe_path']).parent.parent / 'evidence'
    return root / {'fit': 'matched200/epoch200/fit/response_evaluation.json',
                   'response_holdout': 'source_response/response_evaluation.json'}[role]


def restore_model(recipe, run, checkpoint, device):
    path = resolve_checkpoint(run, checkpoint)
    saved = load_trusted_checkpoint(path, map_location='cpu')
    identity = saved['experiment_identity']
    if identity['recipe'] != recipe:
        raise ValueError('The recipe differs from the saved run.')
    native_identity = identity['provider_identity']
    options = {'response_scales': native_identity['response_scales']}
    if recipe.get('r16_4_warm_fork') is None and 'adapter_state' in native_identity:
        options['saved_adapter_state'] = native_identity['adapter_state']
        options['saved_normalization_source'] = native_identity['source_binding']['normalization_source']
    model, provider = build(recipe, device=device, **options)
    if identity['provider_identity'] != dict(provider.identity_payload()):
        raise ValueError('The recipe or native provider differs from the saved run.')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.to(device)
    restore = getattr(provider, 'load_training_state_dict', None)
    if callable(restore):
        restore(saved.get('provider_training_state', {}))
    model.eval()
    return model, provider, saved, path


def native_observation(provider, model, batch, raw, tau, mode):
    with model.core.read_execution(mode):
        prediction, auxiliary = provider.predict_native(model, batch.scene_inputs, batch.receivers,
                                                         model.architecture, 'joint', epoch=0)
    fields = _field_components(prediction, batch)
    converted = {name: {'prediction': pred.detach().cpu().numpy(), 'reference': target.detach().cpu().numpy(),
                        'mask': mask.detach().cpu().numpy().astype(bool)}
                 for name, (pred, target, mask) in fields.items()}
    material = prediction.native_main['pred_internal_temperature'][..., 0]
    pressure = prediction.native_main['pred_field'][..., 2]
    prepared = prediction.native_prepared
    mmask = prepared.stencils['material'].valid.reshape(material.shape) & prepared.source_present[..., None].bool()
    fmask = batch.targets.point_weights.bool() & prepared.stencils['fluid'].valid.reshape(pressure.shape)
    if pressure.numel() != 128 * 64:
        raise ValueError('The physical pressure functional requires the complete 128x64 grid.')
    j, p = native_design_functionals(material, mmask, pressure, fmask, batch.receivers.fluid_xy[..., 0],
                                     raw['config']['domain']['lx'], tau)
    work = {k: float(v) for k, v in auxiliary.get('work', {}).items() if isinstance(v, (int, float))}
    return converted, {'J': float(j.item()), 'P': float(p.item())}, work


def source_strata(pair):
    base, edit = pair['raw_baseline'], pair['raw_edited']
    def q(raw):
        d = raw['config']['domain']; ids = raw['module_id_grid']
        heat = raw['config']['layout']['heat_powers']
        return float(d['lx'] / d['nx'] * d['ly'] / d['ny'] *
                     sum(float(h) * int((ids == i).sum()) for i, h in enumerate(heat)))
    a, b = q(base), q(edit)
    return {'ownership_changed_cells': int((base['module_id_grid'] != edit['module_id_grid']).sum()),
            'represented_source_baseline': a, 'represented_source_edited': b,
            'represented_source_delta': b - a,
            'routing_support_change': None,
            'routing_support_note': 'Not traced by this evaluator; work counters are not support identities.',
            'represented_source_units': 'dataset cellwise heating times cell area; not SI power'}


def summarize_fields(rows):
    """First average actions within a family, then describe family distributions."""
    output = {}
    for field in rows[0]['fields']:
        output[field] = {}
        for kind in ('baseline', 'edited', 'response'):
            families = {}
            for row in rows:
                families.setdefault(row['case_id'], []).append(row['fields'][field][kind]['rmse'])
            family_means = {family: float(np.mean(v)) for family, v in families.items()}
            values = np.asarray(list(family_means.values()), dtype=np.float64)
            worst = max(family_means, key=family_means.get)
            output[field][kind] = {'equal_family_mean_rmse': float(values.mean()),
                                   'family_p90_rmse': float(np.quantile(values, .9)),
                                   'family_worst_rmse': float(values.max()), 'family_worst_id': worst,
                                   'family_count': len(values)}
            entries = [row['fields'][field][kind] for row in rows]
            if all('reference_rms' in entry for entry in entries):
                by_family = {}
                for row in rows:
                    by_family.setdefault(row['case_id'], []).append(row['fields'][field][kind]['reference_rms'])
                effect = float(np.mean([np.mean(values) for values in by_family.values()]))
                output[field][kind].update({
                    'equal_family_mean_reference_rms': effect,
                    'equal_family_rmse_over_reference_rms': float(values.mean()) / effect if effect > 0 else None,
                    'exact_zero_reference_actions': sum(entry['reference_rms'] == 0 for entry in entries),
                    'reference_action_count': len(entries),
                    'relative_error_note': 'Ratio of equally weighted family-average RMSE to family-average reference RMS; zero reference gives null. No field resolution floor is assumed.',
                })
    return output


def summarize_strata(rows, floors):
    groups = {}
    for row in rows:
        action = row['action']
        for key, value in (('edit_kind', action.split('_')[0]), ('sign', action.split('_')[1]),
                           ('module_count', str(row['strata']['module_count'])),
                           ('amplitude', str(row['strata']['amplitude']))):
            groups.setdefault(key, {}).setdefault(value, []).append(row)
    return {key: {value: {'functionals': summarize_actions(group, floors),
                          'fields': summarize_fields(group)}
                  for value, group in values.items()} for key, values in groups.items()}


def evaluate(provider, model, corpus, role, families, labels, tau, floors, save_fields=None):
    all_native = getattr(model.core, 'reader_capability', None) == 'all_native'
    modes = ('all_native',) if all_native else ('restricted', 'full')
    result = {mode: [] for mode in modes}
    with torch.inference_mode():
        for family in families:
            case = corpus.cases[family]
            for action in ACTIONS:
                pair = response_pair(provider, corpus, family, action, role=role)
                strata = source_strata(pair)
                strata.update(module_count=int(case['module_count']), amplitude=float(case['amplitude']),
                              joint_kind=case['joint_kind'], member_ids=case['selection']['member_ids'])
                for mode in modes:
                    # All-native core ignores the directory restriction; the native API still accepts restricted.
                    execution_mode = 'restricted' if mode == 'all_native' else mode
                    b, bf, bw = native_observation(provider, model, pair['baseline'], pair['raw_baseline'], tau, execution_mode)
                    e, ef, ew = native_observation(provider, model, pair['edited'], pair['raw_edited'], tau, execution_mode)
                    fields = compare_native_response(b, e)
                    delta = {name: {'model': ef[name] - bf[name], 'physical': labels[(family, action)][name]}
                             for name in ('J', 'P')}
                    row = {'case_id': family, 'action': action, 'strata': strata, 'fields': fields,
                           'delta': delta, 'baseline_functionals': bf, 'edited_functionals': ef,
                           'work': {'baseline': bw, 'edited': ew}}
                    result[mode].append(row)
                    if save_fields is not None:
                        arrays = {}
                        for name in b:
                            for endpoint, obj in (('baseline', b[name]), ('edited', e[name])):
                                for key, value in obj.items():
                                    arrays[f'{name.replace("/", "_")}__{endpoint}__{key}'] = value
                        np.savez_compressed(save_fields / f'{family}_{action}_{mode}.npz', **arrays)
    return {mode: {'actions': rows, 'functionals': summarize_actions(rows, floors, family_ids=families),
                   'field_summary': summarize_fields(rows), 'strata': summarize_strata(rows, floors)}
            for mode, rows in result.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='latest')
    parser.add_argument('--role', required=True, choices=('fit', 'response_holdout'))
    parser.add_argument('--output', type=Path, required=True, help='Output JSON path')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--families', help='Comma-separated original family IDs; defaults to full role')
    parser.add_argument('--save-fields', action='store_true', help='Save fields for the explicitly selected small family panel')
    args = parser.parse_args(argv)
    recipe = json.loads(args.recipe.read_text()); section = recipe['thermal_state_response']
    if not section.get('enabled'):
        raise ValueError('The selected recipe must enable the physical response task.')
    corpus = ThermalResponseCorpus(section['recipe_path'], section['ledger_path'])
    eligible = corpus.family_ids(args.role)
    families = tuple(args.families.split(',')) if args.families else eligible
    if len(set(families)) != len(families) or not families or not set(families).issubset(eligible):
        raise ValueError('Selected families must be unique IDs in the requested role.')
    if args.save_fields and (not args.families or len(families) > 4):
        raise ValueError('Field exports require an explicit panel of at most four families.')
    label_path = functional_label_path(section, args.role)
    labels = load_saved_functional_responses(label_path, corpus, role=args.role)
    model, provider, saved, path = restore_model(recipe, args.run, args.checkpoint, args.device)
    field_dir = None
    if args.save_fields:
        field_dir = args.output.parent / (args.output.stem + '_fields')
        field_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate(provider, model, corpus, args.role, families, labels, float(section['tau']),
                       section['functional_floors'], field_dir)
    payload = {'schema': 'thermal_state_response_evaluation_v1', 'role': args.role, 'families': list(families),
               'checkpoint': str(path), 'epoch': int(saved['epoch']), 'recipe': str(args.recipe),
               'physical_functional_labels': str(label_path), 'tau': float(section['tau']),
               'functional_floors': section['functional_floors'], 'views': results,
               'interpretation': 'TRAIN-origin edited response families; not an unseen-layout or independent TEST evaluation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps({'output': str(args.output), 'role': args.role, 'families': len(families), 'epoch': saved['epoch']}))


if __name__ == '__main__':
    main()
