#!/usr/bin/env python3
"""Inspect one saved Thermal edit at common fluid cell centres; never solve or fit.

Run one explicit family/action at a time. The compressed capture separates
actual directed context gates, prepared E-address weights, final original-member
coefficients, and the exact finite read/head decomposition. Optional named
relation interventions are selected only on TRAIN inputs and rebuild both
scenes. This sampled field diagnostic does not compute the full J/P functionals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in ('src', 'Case_ThermalChannel/src', 'Case_WindFarm/src', 'tools'):
    sys.path.insert(0, str(ROOT / directory))

import numpy as np
import torch

from channelthermal.data.finite_responses import FiniteResponseStore
from channelthermal.e_anchor_locator import attach_thermal_e_anchor_query
from channelthermal.native_design import build_native_design
from channelthermal.training.native_checkpoint import build_native_checkpoint_inference
from honf_forward_core.content_hypergraph.contracts import QueryBatch
from honf_forward_core.interaction_reused_member_routing.organization_response import (
    capture_applied_relationships, contextual_coefficients_trace_request, finite_read_decomposition, prepare_relation_removal,
    select_context_relations,
)
from honf_runtime.compat import load_trusted_checkpoint

FIELDS = ('flow/u', 'flow/v', 'flow/p', 'temperature/fluid')
SOURCE_COLUMNS = (0, 1, 2, 4)
UNITS = ('dataset velocity', 'dataset velocity', 'dataset pressure', 'dataset temperature')


def _select_fixed_positions(before, after, count, seed):
    """Select common-fluid coordinates from geometry alone, including both bands."""
    if not np.array_equal(before.fluid_xy_source, after.fluid_xy_source):
        raise ValueError('Saved scenes do not share physical fluid cell centres.')
    common = before.fluid_valid & after.fluid_valid
    xy = before.fluid_xy_source
    bands = ((xy[:, 0] <= .08 * before.domain_length_x),
             (xy[:, 0] >= .92 * before.domain_length_x))
    rng = np.random.default_rng(seed)
    selected = []
    for band in bands:
        eligible = np.flatnonzero(common & band)
        take = min(len(eligible), count // 4)
        selected.extend(rng.choice(eligible, size=take, replace=False).tolist())
    remaining = np.setdiff1d(np.flatnonzero(common), np.asarray(selected, dtype=np.int64), assume_unique=False)
    if len(remaining) < count - len(selected):
        raise ValueError('Not enough common valid fluid positions for the requested panel.')
    selected.extend(rng.choice(remaining, size=count - len(selected), replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def _scene_and_query(model, design, selected):
    structure = {**design.structure, 'heat_powers': design.heat}
    scene = model.scene_batch(structure, scene_key=(design.case_id,))
    xy = design.fluid_xy.index_select(1, torch.as_tensor(selected, device=design.fluid_xy.device))
    query = QueryBatch(coordinates=xy, query_mask=torch.ones(xy.shape[:2], dtype=torch.bool, device=xy.device),
                       query_ids=torch.as_tensor(selected, device=xy.device, dtype=torch.long)[None],
                       scene_key=scene.scene_key, coordinate_frame_key=scene.coordinate_frame_key,
                       receiver_key=('fixed_saved_edit_fluid_cell_centres',))
    query = attach_thermal_e_anchor_query(query, scene, model.core.locator_mode,
                                           environment_nx=model.environment_nx, environment_ny=model.environment_ny)
    return scene, query


def _verify_saved_edit(before, after, action_meta):
    left, right = before['config'], after['config']
    if left['domain'] != right['domain'] or left['thermal'] != right['thermal']:
        raise ValueError('The saved edit changes physical domain or material definition.')
    for key in ('re', 'u_in'):
        if left['flow'].get(key) != right['flow'].get(key):
            raise ValueError(f'The saved edit changes prescribed flow input {key}.')
    # Some stored configurations spell out viscosity while their original
    # record derives the identical value from Reynolds number and radius.
    def effective_viscosity(config):
        flow = config['flow']
        if flow.get('nu') is not None:
            return float(flow['nu'])
        return (float(flow.get('viscosity_scale', 1.0)) * float(flow['u_in'])
                * 2.0 * float(config['domain']['module_radius']) / max(float(flow['re']), 1e-12))
    if not np.isclose(effective_viscosity(left), effective_viscosity(right), rtol=1e-12, atol=0):
        raise ValueError('The saved edit changes effective viscosity.')
    for key in ('n_interface_points', 'local_grid_size'):
        if left['local_module'][key] != right['local_module'][key]:
            raise ValueError('The saved edit changes attached-member type or reference sampling.')
    for key in ('num_modules', 'heat_powers'):
        if left['layout'][key] != right['layout'][key]:
            raise ValueError('The saved edit changes the original-member inventory or heat controls.')
    if action_meta.get('centers') != right['layout']['centers']:
        raise ValueError('Saved action slot coordinates disagree with its edited scene.')
    if len(left['layout']['centers']) != len(right['layout']['centers']):
        raise ValueError('Insertion or removal has no two-factor attribution.')


def _to_physical(model, fields):
    columns = torch.tensor(SOURCE_COLUMNS, dtype=torch.long, device=fields.device)
    scale = model.field_std.index_select(0, columns).to(fields)
    mean = model.field_mean.index_select(0, columns).to(fields)
    return (fields * scale + mean).detach().cpu().numpy()[0]


def _geometry(scene, config):
    names = ('module_coordinates', 'module_ids', 'module_mask', 'module_measure',
             'environment_coordinates', 'environment_ids', 'environment_mask', 'environment_measure')
    result = {name: getattr(scene, name).detach().clone() for name in names}
    domain = config['domain']
    result['domain_bounds'] = torch.tensor([0., domain['lx'], 0., domain['ly']], dtype=scene.module_coordinates.dtype)
    result['radius'] = float(domain['module_radius'])
    return result


def _applied_relation(relationships, relation):
    stages = [s for s in relationships['contextual_stages']
              if s['stage'] == relation['stage'] and s['route'] == relation['route']]
    if len(stages) != 1:
        raise ValueError('The selected contextual stage was not executed exactly once.')
    stage = stages[0]
    receiver = (stage['receiver_ids'][0] == int(relation['destination_id'])).nonzero().flatten()
    donor = (stage['donor_ids'][0] == int(relation['donor_id'])).nonzero().flatten()
    if len(receiver) != 1 or len(donor) != 1:
        raise ValueError('The selected contextual relation does not identify original members.')
    coefficients = stage['coefficients']
    return {'gate': float(stage['gate'][0, receiver[0], donor[0]]),
            'mean_coefficient': (None if coefficients is None else
                                 float(coefficients[0, :, receiver[0], donor[0]].mean()))}


def _physical_metrics(model, pred_before, pred_after, raw_before, raw_after, selected, xy, length):
    before = _to_physical(model, pred_before)
    after = _to_physical(model, pred_after)
    references = [np.asarray(raw['field']).reshape(-1, 5)[selected][:, SOURCE_COLUMNS].astype(np.float64)
                  for raw in (raw_before, raw_after)]
    results = {}
    for channel, (name, unit) in enumerate(zip(FIELDS, UNITS)):
        ref0, ref1 = references[0][:, channel], references[1][:, channel]
        pred0, pred1 = before[:, channel].astype(np.float64), after[:, channel].astype(np.float64)
        valid = np.isfinite(ref0) & np.isfinite(ref1) & np.isfinite(pred0) & np.isfinite(pred1)
        if not valid.any():
            results[name] = {'available': False, 'unit': unit, 'valid_positions': 0}
            continue
        target = (ref1 - ref0)[valid]
        predicted = (pred1 - pred0)[valid]
        rmse = lambda value: float(np.sqrt(np.mean(value * value)))
        target_rms, pred_rms = rmse(target), rmse(predicted)
        results[name] = {'available': True, 'unit': unit, 'valid_positions': int(valid.sum()),
                         'before_rmse': rmse((pred0-ref0)[valid]), 'after_rmse': rmse((pred1-ref1)[valid]),
                         'change_rmse': rmse(predicted-target), 'zero_change_rmse': target_rms,
                         'predicted_change_rms': pred_rms, 'physical_change_rms': target_rms,
                         'direction_cosine': (float(np.mean(predicted*target)/(target_rms*pred_rms))
                                              if target_rms > 0 and pred_rms > 0 else None)}
    pressure = {}
    for label, mask in (('inlet', xy[:, 0] <= .08 * length), ('outlet', xy[:, 0] >= .92 * length)):
        valid = mask & np.isfinite(references[0][:, 2]) & np.isfinite(references[1][:, 2])
        pressure[label] = {'sampled_positions': int(valid.sum()),
                           'predicted_mean_change': float(np.mean((after-before)[valid, 2])) if valid.any() else None,
                           'physical_mean_change': float(np.mean((references[1]-references[0])[valid, 2])) if valid.any() else None}
    return {'fields': results, 'sampled_pressure_bands': pressure,
            'scope': 'Selected common-fluid cell centres only. Band means do not replace the full pressure functional. '
                     'Surface and material queries move and have no two-factor allocation here.'}


def _flatten(value, arrays, path='capture'):
    if isinstance(value, torch.Tensor):
        key = path.replace('/', '__')
        arrays[key] = value.detach().cpu().numpy()
        return {'array_key': key, 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _flatten(v, arrays, f'{path}/{k}') for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_flatten(v, arrays, f'{path}/{i}') for i, v in enumerate(value)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(args):
    if not 1 <= args.query_count <= 64 or not 0 <= args.relations <= 4:
        raise ValueError('Use 1–64 queries and at most four contextual relations.')
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    store = FiniteResponseStore(args.pack)
    if args.action not in store.actions(args.family, role=args.role):
        raise ValueError('The action is outside the declared saved family/role.')
    before = store.endpoint(args.family, role=args.role)
    after = store.endpoint(args.family, args.action, role=args.role)
    action_metadata = store.action_metadata(args.family, args.action, role=args.role)
    _verify_saved_edit(before, after, action_metadata)
    saved = load_trusted_checkpoint(args.checkpoint, map_location='cpu')
    model, provider = build_native_checkpoint_inference(
        saved, validation_case_ids=(args.loader_validation_case,), device=args.device)
    model.eval()
    if args.query_count > model.core.query_tile_size:
        raise ValueError('The bounded capture must fit within the checkpoint reader tile.')
    if args.relations and args.relations_from:
        raise ValueError('Choose input-based relation selection or a saved relation list, not both.')
    if (args.relations or args.relations_from) and (
            args.role != 'fit' or args.family not in set(map(str, provider.training_case_ids))):
        raise ValueError('Relation selection and interventions require a saved FIT family in this checkpoint TRAIN membership.')
    designs = [build_native_design(raw['config'], args.family, device=args.device) for raw in (before, after)]
    for design, raw in zip(designs, (before, after)):
        if not np.array_equal(design.module_id_grid, raw['module_id_grid']):
            raise ValueError('Saved physical ownership differs from the input-derived scene.')
        if not np.array_equal(design.fluid_valid, np.asarray(raw['fluid_mask']).reshape(-1)):
            raise ValueError('Saved fluid mask differs from the input-derived scene.')
    selected = _select_fixed_positions(*designs, args.query_count, args.seed)
    left_scene, left_query = _scene_and_query(model, designs[0], selected)
    right_scene, right_query = _scene_and_query(model, designs[1], selected)
    with torch.no_grad():
        if torch.device(args.device).type == 'cuda':
            torch.cuda.synchronize(args.device)
        prep_start = time.perf_counter()
        trace_request = contextual_coefficients_trace_request(model.core)
        left = model.core.prepare(left_scene, trace_request=trace_request)
        right = model.core.prepare(right_scene, trace_request=trace_request)
        if torch.device(args.device).type == 'cuda':
            torch.cuda.synchronize(args.device)
        prep_seconds = time.perf_counter() - prep_start
        relationships = {'before': capture_applied_relationships(model.core, left),
                         'after': capture_applied_relationships(model.core, right)}
        decomposition = finite_read_decomposition(model.core, left, left_query, right, right_query,
                                                  identity_confirmed=True, read_mode=args.read_mode)
        if not decomposition['available']:
            raise ValueError(decomposition['reason'])
        xy = designs[0].fluid_xy_source[selected]
        nominal = _physical_metrics(model, decomposition['field_before'], decomposition['field_after'],
                                    before, after, selected, xy, designs[0].domain_length_x)
        selected_relations = select_context_relations(relationships['before'], seed=args.seed, count=args.relations) if args.relations else []
        prior = None
        if args.relations_from:
            prior = json.loads(args.relations_from.read_text())
            if prior.get('family') != args.family or prior.get('role') != 'fit':
                raise ValueError('The saved relation list must use the same TRAIN family.')
            selected_relations = [entry['relation'] for entry in prior['interventions']
                                  if entry['kind'] == 'contextual_relation_removed']
            if not 1 <= len(selected_relations) <= 4:
                raise ValueError('The saved relation list must contain one to four contextual relations.')
        interventions = []
        for relation in selected_relations:
            start = time.perf_counter()
            a = prepare_relation_removal(model.core, left_scene, relation)
            b = prepare_relation_removal(model.core, right_scene, relation)
            out_a = model.core.decode(a, left_query, intervention={'read_mode': args.read_mode})
            out_b = model.core.decode(b, right_query, intervention={'read_mode': args.read_mode})
            if torch.device(args.device).type == 'cuda':
                torch.cuda.synchronize(args.device)
            interventions.append({'kind': 'contextual_relation_removed', 'relation': relation,
                                  'actual_context_before_removal': _applied_relation(relationships['before'], relation),
                                  'actual_context_after_edit_before_removal': _applied_relation(relationships['after'], relation),
                                  'field_metrics': _physical_metrics(model, out_a.joint_fields, out_b.joint_fields,
                                                                     before, after, selected, xy, designs[0].domain_length_x),
                                  'prepare_work_before': dict(a.work_counts), 'prepare_work_after': dict(b.work_counts),
                                  'decode_work_before': dict(out_a.work_counts), 'decode_work_after': dict(out_b.work_counts),
                                  'seconds_including_repreparation': time.perf_counter() - start})
        if selected_relations:
            choices = []
            for role, values in decomposition['roles'].items():
                for identity in values['original_id'].unique().tolist():
                    mass = float(values['beta_before'][values['original_id'] == identity].sum())
                    choices.append((mass, role, int(identity)))
            choices.sort(reverse=True)
            controls = [('within_role_measure_only', {'within_role_access_prior': 'measure_only'})]
            prior_member = ([entry['control'] for entry in prior['interventions']
                             if entry['kind'] == 'highest_selected_member_removed'] if prior else [])
            if prior_member:
                controls.append(('highest_selected_member_removed', prior_member[0]))
            elif choices:
                # Prefer M: E-address structural self membership is not removed
                # by the existing member intervention.
                _, role, identity = next((item for item in choices if item[1] == 'M'), choices[0])
                controls.append(('highest_selected_member_removed', {'access_intervention': 'remove_member',
                                                                    'member_role': role, 'member_id': identity}))
            for label, control in controls:
                start = time.perf_counter()
                out_a = model.core.decode(left, left_query, intervention={**control, 'read_mode': args.read_mode})
                out_b = model.core.decode(right, right_query, intervention={**control, 'read_mode': args.read_mode})
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.synchronize(args.device)
                interventions.append({'kind': label, 'control': control,
                                      'field_metrics': _physical_metrics(model, out_a.joint_fields, out_b.joint_fields,
                                                                         before, after, selected, xy, designs[0].domain_length_x),
                                      'decode_work_before': dict(out_a.work_counts),
                                      'decode_work_after': dict(out_b.work_counts),
                                      'seconds': time.perf_counter() - start})
    if torch.device(args.device).type == 'cuda':
        torch.cuda.synchronize(args.device)
    arrays = {}
    capture = _flatten({'relationships': relationships, 'decomposition': decomposition,
                        'geometry': {'before': _geometry(left_scene, before['config']),
                                     'after': _geometry(right_scene, after['config'])}}, arrays)
    output = Path(args.output)
    array_path = output.with_suffix('.npz')
    if output.exists() or array_path.exists():
        raise FileExistsError('Diagnostic output already exists; choose a new location.')
    payload = {'schema': 'thermal_saved_edit_organization_response_v1',
               'checkpoint': str(Path(args.checkpoint).resolve()), 'epoch': int(saved['epoch']),
               'pack': str(Path(args.pack).resolve()), 'family': args.family, 'action': args.action, 'role': args.role,
               'loader_only_validation_case': args.loader_validation_case,
               'read_mode': args.read_mode, 'device': args.device, 'seed': args.seed, 'selected_cell_ids': selected.tolist(),
               'within_role_access_prior': model.core.within_role_access_prior,
               'retained_mass_setting': float(model.core.retained_mass),
               'query_selection': 'Seeded input-only common-fluid sample, with inlet/outlet bands.',
               'identity': 'Saved action slot ordering and role-preserving IDs, checked against saved geometry.',
               'changed_module_positions': [i for i, (x, y) in enumerate(zip(before['config']['layout']['centers'], after['config']['layout']['centers'])) if x != y],
               'physical_solves': 0, 'fitting_epochs': 0, 'prepare_seconds': prep_seconds,
               'prepare_work_before': dict(left.work_counts), 'prepare_work_after': dict(right.work_counts),
               'diagnostic_seconds_including_preparation': time.perf_counter() - started,
               'nominal_physical_metrics': nominal, 'interventions': interventions,
               'arrays': array_path.name, 'capture': capture}
    output.parent.mkdir(parents=True, exist_ok=True)
    with array_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'arrays': str(array_path), 'family': args.family,
                      'action': args.action, 'epoch': saved['epoch'], 'read_mode': args.read_mode,
                      'interventions': len(interventions)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--pack', required=True, type=Path)
    parser.add_argument('--family', required=True)
    parser.add_argument('--action', required=True)
    parser.add_argument('--role', required=True, choices=('fit', 'response_holdout', 'historical_new_withheld', 'native_seen', 'fresh_layout'))
    parser.add_argument('--output', required=True, type=Path, help='New JSON path; compressed arrays use the same basename.')
    parser.add_argument('--read-mode', default='restricted', choices=('restricted', 'full'))
    parser.add_argument('--query-count', default=32, type=int)
    parser.add_argument('--relations', default=0, type=int, help='Up to four input-selected TRAIN contextual relations, plus member and measure controls.')
    parser.add_argument('--relations-from', type=Path, help='Reuse contextual relations selected on the same TRAIN family by an earlier capture.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cpu-threads', default=4, type=int)
    parser.add_argument('--loader-validation-case', default='0679',
                        help='Existing exposed native validation case used only to construct the checkpoint provider.')
    parser.add_argument('--device', default='cpu')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
