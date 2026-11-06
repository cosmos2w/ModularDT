"""Thermal R16 paired-response corpus, alignment and objective primitives.

Labels and model observations are kept separate: receiver identity determines
alignment, while coordinates may move for member-attached observations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from honf_runtime.unified_training import LossTerm
from ..interaction_evidence.storage import load_solve_record

ACTIONS = ('single_plus', 'single_minus', 'joint_plus', 'joint_minus')
FIELD_ROLES = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid',
               'temperature/surface', 'temperature/material', 'q_proxy')


class ThermalResponseCorpus:
    """Index the original FIT and holdout roles without reading holdout labels."""

    def __init__(self, recipe_path, ledger_path):
        self.recipe_path, self.ledger_path = str(recipe_path), str(ledger_path)
        recipe = json.loads(Path(recipe_path).read_text())
        self.cases = {}
        for case in recipe['cases']:
            case_id = str(case['case_id'])
            if case_id in self.cases:
                raise ValueError('Duplicate response family ID.')
            self.cases[case_id] = case
        starts, ends = {}, {}
        for line in Path(ledger_path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['event'] == 'started':
                starts[row['attempt']] = row
            elif row['event'] == 'finished':
                ends[row['attempt']] = row
        self.records = {}
        for attempt, start in starts.items():
            if start.get('allocation') != 'edited':
                continue
            end = ends.get(attempt, {})
            if end.get('status') != 'converged' or end.get('serialization_status') != 'stored':
                continue
            key = (str(start['case_id']), start['action'])
            if key in self.records:
                raise ValueError(f'Duplicate converged response action {key}.')
            if key[0] not in self.cases or start['role'] != self.cases[key[0]]['role']:
                raise ValueError(f'Response ledger role differs from family {key}.')
            self.records[key] = {'raw_case_dir': end['raw_case_dir'],
                                 'record_directory': end['record_directory']}
        for role, expected in (('fit', 48), ('response_holdout', 12)):
            families = self.family_ids(role)
            if len(families) != expected:
                raise ValueError(f'Expected {expected} {role} families, got {len(families)}.')
            missing = [(family, action) for family in families for action in ACTIONS
                       if (family, action) not in self.records]
            if missing:
                raise ValueError(f'Missing {len(missing)} {role} response records.')

    def family_ids(self, role):
        if role not in ('fit', 'response_holdout'):
            raise ValueError('Unknown response role.')
        return tuple(sorted(key for key, case in self.cases.items()
                            if case['role'] == role and case['feasible']))

    def fit_families_for_epoch(self, epoch, seed=0):
        """Four families per epoch; seeded permutations cover all in 12 epochs."""
        if int(epoch) < 1:
            raise ValueError('Epoch numbering begins at one.')
        families = self.family_ids('fit')
        cycle, offset = divmod(int(epoch) - 1, 12)
        order = np.random.default_rng(np.random.SeedSequence([int(seed), cycle])).permutation(len(families))
        return tuple(families[int(i)] for i in order[offset * 4:(offset + 1) * 4])

    def fit_scales(self, families, state_scales, repeatability, functional_floors):
        """Fit from exactly the original FIT cohort, never holdout outcomes."""
        if set(map(str, families)) != set(self.family_ids('fit')):
            raise ValueError('Response scales require exactly the original FIT families.')
        return fit_response_scales(families, state_scales, repeatability, functional_floors)

    def action_record(self, family_id, action, *, role):
        entry = self.action_entry(family_id, action, role=role)
        return load_solve_record(entry['record_directory'])

    def action_entry(self, family_id, action, *, role):
        family_id = str(family_id)
        if self.cases[family_id]['role'] != role or action not in ACTIONS:
            raise ValueError('Response action is outside the requested role.')
        return self.records[(family_id, action)]


class ThermalActionCorpus:
    """Load explicit action records without changing the original FIT corpus.

    Roles belong to actions, not layouts: one familiar layout may contain both
    fitted and withheld edits. Geometry and observations use the same native
    endpoint loader as the historical four-action corpus.
    """

    fit_cells = (('v0', 'a0'), ('v1', 'a0'), ('v0', 'a1'))

    def __init__(self, manifest_path):
        self.manifest_path = str(manifest_path)
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get('schema') != 'thermal_response_actions_v1':
            raise ValueError('Expected thermal_response_actions_v1 action manifest.')
        self.cases, self.records = {}, {}
        for case in manifest['cases']:
            case_id = str(case['case_id'])
            if case_id in self.cases or not case.get('source_case_dir'):
                raise ValueError('Action layouts need unique IDs and baseline paths.')
            self.cases[case_id] = case
        for row in manifest['actions']:
            key = (str(row['case_id']), row['action'])
            if key[0] not in self.cases or key in self.records or not key[1]:
                raise ValueError('Action records need unique IDs and an existing layout.')
            if row['role'] not in ('fit', 'diagnostic'):
                raise ValueError('Explicit actions require fit or diagnostic roles.')
            if row['role'] == 'fit':
                cell = row.get('cell') or {}
                if (row.get('kind') != 'first_member'
                        or (cell.get('direction'), cell.get('amplitude')) not in self.fit_cells):
                    raise ValueError('Withheld cells, second-member and joint edits cannot be fitted.')
            self.records[key] = row

    def action_entry(self, family_id, action, *, role):
        row = self.records[(str(family_id), action)]
        if role not in ('fit', 'diagnostic') or row['role'] != role:
            raise ValueError('Response action is outside the requested role.')
        if not row.get('raw_case_dir') or not row.get('record_directory'):
            raise ValueError('Response action has no completed physical record binding.')
        return row

    def action_record(self, family_id, action, *, role):
        return load_solve_record(self.action_entry(family_id, action, role=role)['record_directory'])

    def physical_delta(self, family_id, action, *, role):
        row = self.action_entry(family_id, action, role=role)
        labels = row.get('physical_delta', {})
        if set(labels) != {'J', 'P'} or any(not math.isfinite(float(v)) for v in labels.values()):
            raise ValueError('A fitted physical action requires finite native J/P deltas.')
        return {name: float(labels[name]) for name in ('J', 'P')}

    def pilot_actions_for_epoch(self, arm, epoch):
        """Two signs per layout, with equal input-only exposure in both arms.

        Single-member layouts always use v0/a0. In the coverage arm each other
        layout cycles the three fitting cells with a stable layout offset.
        Selection consumes no NumPy or Torch RNG and resumes from epoch alone.
        """
        if arm not in ('control', 'coverage') or int(epoch) != epoch or int(epoch) < 1:
            raise ValueError('Pilot sampling requires control/coverage and a positive epoch.')
        selected, multi_index = [], 0
        for family, case in sorted(self.cases.items()):
            count = case.get('module_count', len(case.get('member_ids', ())))
            if count < 1:
                raise ValueError('Pilot layout needs its original member count.')
            cell = self.fit_cells[0]
            if count > 1:
                if arm == 'coverage':
                    cell = self.fit_cells[(int(epoch) - 1 + multi_index) % len(self.fit_cells)]
                multi_index += 1
            for sign in (1, -1):
                matches = [(key, row) for key, row in self.records.items()
                           if key[0] == family and row['role'] == 'fit'
                           and row.get('kind') == 'first_member' and row.get('sign') == sign
                           and (row.get('cell', {}).get('direction'),
                                row.get('cell', {}).get('amplitude')) == cell]
                if len(matches) != 1:
                    raise ValueError(f'Pilot needs one action for {family}/{cell}/{sign}.')
                selected.append(matches[0][0])
        return tuple(selected)


def align_role_values(baseline, edited):
    """Align original receiver IDs; retain each endpoint mask and common mask.

    Member-attached coordinates may differ as a member moves. Eulerian receiver
    coordinates must remain identical; identity alone is insufficient there.
    """
    if (baseline.role != edited.role or baseline.channel_names != edited.channel_names
            or baseline.coordinate_kind != edited.coordinate_kind):
        raise ValueError('Response role, channels and coordinate kind must match.')
    ids_b, ids_e = tuple(baseline.query_ids), tuple(edited.query_ids)
    if len(set(ids_b)) != len(ids_b) or len(set(ids_e)) != len(ids_e) or set(ids_b) != set(ids_e):
        raise ValueError('Response receiver IDs must be unique and match.')
    edited_index = {receiver_id: index for index, receiver_id in enumerate(ids_e)}
    order = np.asarray([edited_index[i] for i in ids_b])
    if baseline.receiver_module_ids is not None:
        if edited.receiver_module_ids is None or tuple(baseline.receiver_module_ids) != tuple(edited.receiver_module_ids[i] for i in order):
            raise ValueError('Original receiver member IDs differ.')
    elif edited.receiver_module_ids is not None:
        raise ValueError('Original receiver member IDs differ.')
    if baseline.coordinate_kind == 'eulerian' and not np.array_equal(baseline.query_features, edited.query_features[order]):
        raise ValueError('Eulerian response receiver coordinates differ.')
    b, e = np.asarray(baseline.values), np.asarray(edited.values)[order]
    bm, em = np.asarray(baseline.valid_mask, bool), np.asarray(edited.valid_mask, bool)[order]
    if b.shape != e.shape or b.shape != bm.shape or b.shape != em.shape:
        raise ValueError('Response values and masks must have matching channel shapes.')
    return {'baseline': b, 'edited': e, 'baseline_mask': bm,
            'edited_mask': em, 'common_mask': bm & em, 'query_ids': ids_b}


def fit_response_scales(families, state_scales, repeatability, functional_floors):
    """Fit per-role RMS over equally weighted FIT families and their actions.

    ``families`` maps family IDs to four action dictionaries. Each role value is
    (delta, valid mask); J and P are scalar deltas. Caller supplies only FIT.
    """
    names = (*FIELD_ROLES, 'J', 'P')
    family_squares = {name: [] for name in names}
    if not families:
        raise ValueError('Response scale fit needs FIT families.')
    for actions in families.values():
        if set(actions) != set(ACTIONS):
            raise ValueError('Each fitting family must contain all four actions.')
        squares = {name: [] for name in names}
        for action in ACTIONS:
            for name in FIELD_ROLES:
                values, mask = actions[action][name]
                values, mask = np.asarray(values, dtype=np.float64), np.asarray(mask, dtype=bool)
                if values.shape != mask.shape or not mask.any() or not np.isfinite(values[mask]).all():
                    raise ValueError(f'No valid aligned FIT observations for {name}.')
                squares[name].append(float(np.mean(values[mask] ** 2)))
            for name in ('J', 'P'):
                value = float(actions[action][name])
                if not math.isfinite(value):
                    raise ValueError(f'Nonfinite FIT functional response {name}.')
                squares[name].append(value * value)
        for name in names:
            family_squares[name].append(float(np.mean(squares[name])))
    result = {}
    for name in names:
        rms = math.sqrt(float(np.mean(family_squares[name])))
        floor = (max(float(repeatability[name]), .01 * float(state_scales[name]))
                 if name in FIELD_ROLES else float(functional_floors[name]))
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError(f'Invalid response scale floor for {name}.')
        result[name] = {'scale': max(rms, floor), 'fit_rms': rms, 'floor': floor}
    return result


def _response_coefficient(value):
    """Validate the finite-response coefficient without changing legacy defaults."""
    from numbers import Real

    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite-response coefficient must be finite and nonnegative.')
    return float(value)


def _response_objective_weights(coefficient):
    # Keep historical constants exact: evaluating 0.1 * 0.8 instead would
    # change their floating-point representation in existing training runs.
    coefficient = _response_coefficient(coefficient)
    if coefficient == .1:
        return {'endpoint': .25, 'field_delta': .08, 'J_delta': .01, 'P_delta': .01}
    return {'endpoint': .25, 'field_delta': .8 * coefficient,
            'J_delta': .1 * coefficient, 'P_delta': .1 * coefficient}


def response_loss_terms(endpoint_loss, field_losses, functional_losses, *, multiplier=1.0,
                        finite_response_coefficient=.1):
    """Attach the common endpoint and finite-response weights to scalar losses.

    Inputs already average family/action and paired full/restricted views as
    appropriate; Q0 supplies one view. Epoch scheduling supplies ``multiplier``.
    """
    if set(functional_losses) != {'J', 'P'}:
        raise ValueError('Both complete native functionals are required.')
    weights = _response_objective_weights(finite_response_coefficient)
    if finite_response_coefficient == 0:
        field_losses = field_losses.detach()
        functional_losses = {name: value.detach() for name, value in functional_losses.items()}
    return {
        'response/endpoint': LossTerm(endpoint_loss, 1, weights['endpoint'] * multiplier),
        'response/field_delta': LossTerm(field_losses, 1, weights['field_delta'] * multiplier),
        'response/J_delta': LossTerm(functional_losses['J'], 1, weights['J_delta'] * multiplier),
        'response/P_delta': LossTerm(functional_losses['P'], 1, weights['P_delta'] * multiplier),
    }


def load_saved_functional_responses(path, corpus, *, role):
    """Read historical physical J/P deltas from the named response evaluation.

    These are the source study's originally evaluated labels, not a silent
    recomputation from FP32 fields. This function opens only the requested
    role's evaluation file; callers fitting scales must request ``fit``.
    """
    data = json.loads(Path(path).read_text())
    models = data['models']
    if not models:
        raise ValueError('Response evaluation has no model views.')
    expected = {(family, action) for family in corpus.family_ids(role) for action in ACTIONS}
    result = None
    for views in models.values():
        for view in views.values():
            rows = view['actions']
            current = {}
            for row in rows:
                key = (str(row['case_id']), row['action'])
                if key in current:
                    raise ValueError('Duplicate historical functional action.')
                current[key] = {name: float(row['delta'][name]['physical']) for name in ('J', 'P')}
                if any(not math.isfinite(value) for value in current[key].values()):
                    raise ValueError('Historical functional label is nonfinite.')
            if set(current) != expected:
                raise ValueError(f'Historical functional labels do not match {role} membership.')
            if result is not None and current != result:
                raise ValueError('Historical physical labels differ between model views.')
            result = current
    if result is None:
        raise ValueError('Response evaluation has no model views.')
    return result


def _native_endpoint_batch(provider, original, raw, label, record=None):
    """Rebuild the moved scene, preserving saved physical receiver labels."""
    import copy
    from ..data.datasets import _local_disk_query_points
    from .content_hypergraph_task import ThermalContentHypergraphTask

    case = copy.deepcopy(original)
    family = str(original['case_id'])
    count = len(raw['config']['layout']['centers'])
    present = np.asarray(case['structure']['module_present']) > .5
    if not np.array_equal(np.flatnonzero(present), np.arange(count)):
        raise ValueError('Physical endpoints must preserve original module slots.')
    case['case_id'] = f'{family}:{label}'
    case['structure']['module_centers'][:count] = np.asarray(raw['config']['layout']['centers'], np.float32)
    if 'raster_module_centers' in case['structure']:
        from ..data.datasets import prescribed_raster_geometry
        centers64, raster_geometry = prescribed_raster_geometry(raw['config'], case['structure']['module_present'])
        case['structure']['raster_module_centers'] = centers64
        case['structure']['raster_geometry'] = raster_geometry
    case['structure']['heat_powers'][:count] = np.asarray(raw['config']['layout']['heat_powers'], np.float32)
    case['steady_field'] = raw['field'].copy()
    case['module_mask'] = (~raw['fluid_mask']).astype(np.uint8)
    case['module_internal_query_points'] = _local_disk_query_points(raw['material_mask'])
    material = raw['material'][:, raw['material_mask']].copy()
    interface = raw['interface_target'].copy()
    surface_valid = np.isfinite(interface).any(axis=-1)
    if record is not None:
        roles = record.output.roles
        fluid, surface, solid = (roles[name] for name in ('fluid_fields', 'interface', 'solid_temperature'))
        ny, nx = raw['fluid_mask'].shape
        expected_grid = tuple(f'grid:{y}:{x}' for y in range(ny) for x in range(nx))
        if tuple(fluid.query_ids) != expected_grid or fluid.values.shape != (ny * nx, 5):
            raise ValueError('Saved fluid receiver IDs do not match the native grid.')
        if surface.values.shape != (count * 64, 2) or solid.values.shape != (material.size, 1):
            raise ValueError('Saved attached receivers do not match the native catalogue.')
        expected_surface = tuple(f'{family}:module:{m}:port:{i}' for m in range(count) for i in range(64))
        per_module = material.shape[-1]
        expected_solid = tuple(f'{family}:module:{m}:solid:{i}' for m in range(count) for i in range(per_module))
        if tuple(surface.query_ids) != expected_surface or tuple(solid.query_ids) != expected_solid:
            raise ValueError('Saved attached receiver IDs changed original ordering.')
        field = np.where(fluid.valid_mask, fluid.values, np.nan).reshape(ny, nx, 5)
        case['steady_field'] = np.asarray(field, dtype=np.float32)
        interface = np.where(surface.valid_mask, surface.values, np.nan).reshape(count, 64, 2).astype(np.float32)
        surface_valid = surface.valid_mask.reshape(count, 64, 2).any(axis=-1)
        material = np.where(solid.valid_mask, solid.values, np.nan).reshape(material.shape).astype(np.float32)
    width = len(present)
    case['module_internal_temperature_points'] = np.zeros((width, material.shape[-1]), dtype=np.float32)
    case['module_internal_temperature_points'][:count] = material
    case['interface_target'] = np.zeros((width, 64, 2), dtype=np.float32)
    case['interface_target'][:count] = interface
    case['interface_condition_valid_mask'] = np.zeros((width, 64), dtype=np.float32)
    case['interface_condition_valid_mask'][:count] = surface_valid
    batch = ThermalContentHypergraphTask._native_case_batch(provider, [case], [0], split='train', ntheta=64)
    if record is not None:
        # The record is the original physical observation, including its FP64
        # sampling. Preserve it separately from the FP32 case collation buffers.
        from dataclasses import replace
        device = batch.targets.field_targets.device
        field64 = torch.as_tensor(np.where(fluid.valid_mask, fluid.values, np.nan)[None], device=device)
        surface64 = torch.full(batch.targets.interface_target.shape, float('nan'), device=device, dtype=torch.float64)
        surface64[:, :count] = torch.as_tensor(np.where(surface.valid_mask, surface.values, np.nan).reshape(count, 64, 2), device=device)
        material64 = torch.full(batch.targets.material_targets.shape, float('nan'), device=device, dtype=torch.float64)
        material64[:, :count] = torch.as_tensor(np.where(solid.valid_mask, solid.values, np.nan).reshape(count, -1), device=device)
        batch = replace(batch, targets=replace(batch.targets, field_targets=field64,
                        interface_target=surface64, material_targets=material64))
    return batch


def response_pair(provider, corpus, family_id, action, *, role):
    """Prepare complete native baseline/edited inputs and preserved observations."""
    from .thermal_response_raw import load_raw_endpoint
    family_id = str(family_id)
    case = corpus.cases[family_id]
    entry = corpus.action_entry(family_id, action, role=role)
    original = provider._load_native_cases([family_id])[0]
    raw_b = load_raw_endpoint(case['source_case_dir'])
    raw_e = load_raw_endpoint(entry['raw_case_dir'])
    if not np.array_equal(raw_b['material_mask'], raw_e['material_mask']):
        raise ValueError('Attached material coordinates differ between endpoints.')
    from .content_hypergraph_task import ThermalContentHypergraphTask
    from ..data.datasets import _local_disk_query_points
    if not np.array_equal(np.asarray(original['module_internal_query_points']), _local_disk_query_points(raw_b['material_mask'])):
        raise ValueError('Source H5 and raw attached material coordinates differ.')
    baseline = ThermalContentHypergraphTask._native_case_batch(provider, [original], [0], split='train', ntheta=64)
    edited = _native_endpoint_batch(provider, original, raw_e, action,
                                    corpus.action_record(family_id, action, role=role))
    if not torch.equal(baseline.receivers.heat, edited.receivers.heat):
        raise ValueError('A geometry action changed prescribed heat.')
    return {'family_id': family_id, 'action': action, 'role': role,
            'baseline': baseline, 'edited': edited,
            'raw_baseline': raw_b, 'raw_edited': raw_e}


def _field_components(prediction, batch):
    """Native predictions, physical targets and per-role own-endpoint masks."""
    p, t, prep = prediction.native_main, batch.targets, prediction.native_prepared
    field = p['pred_field']; interface = p['pred_interface']; material = p['pred_internal_temperature'][..., 0]
    field_valid = t.point_weights.bool() & prep.stencils['fluid'].valid.reshape(t.point_weights.shape)
    present = prep.source_present[..., None].bool()
    surface = prep.stencils['surface'].valid.reshape(interface.shape[:-1]) & present & t.interface_valid_mask.bool()
    outside = prep.stencils['outside'].valid.reshape(interface.shape[:-1])
    material_valid = prep.stencils['material'].valid.reshape(material.shape) & present
    values = {}
    for i, name in enumerate(FIELD_ROLES[:5]):
        target = t.field_targets[..., i].to(device=field.device)
        values[name] = (field[..., i], target, field_valid & torch.isfinite(target))
    for name, index in (('temperature/surface', 0), ('q_proxy', 1)):
        target = t.interface_target[..., index].to(device=interface.device)
        mask = surface & torch.isfinite(target)
        if index == 1:
            mask = mask & outside
        values[name] = (interface[..., index], target, mask)
    target = t.material_targets.to(device=material.device)
    values['temperature/material'] = (material, target, material_valid & torch.isfinite(target))
    return values


def _equal_module_mean(error, mask):
    """Match native role reduction: equal receivers within each active module."""
    if error.ndim == 2:
        return error[mask].square().mean()
    count = mask.sum(-1)
    active = count > 0
    if not bool(active.any()):
        raise ValueError('No common attached response receivers.')
    per_module = torch.where(mask, error, torch.zeros_like(error)).square().sum(-1) / count.clamp_min(1)
    return per_module[active].mean()


def predict_response_pair(provider, model, pair, scales, physical_delta, tau, *, response_grad=True):
    """Run full native endpoints and return common loss scalars.

    Complete predicted J/P use each endpoint's own support; field deltas use
    common support. When ``response_grad`` is false, response reductions are
    detached diagnostics while endpoint losses keep their ordinary gradients.
    The caller owns backward and may discard each action graph.
    """
    from .joint_member_routing_task import ThermalJointPairPredictions
    from ..native_functionals import native_design_functionals
    outputs = {}
    work = {}
    for endpoint in ('baseline', 'edited'):
        batch = pair[endpoint]
        predicted, aux = provider.predict_native(model, batch.scene_inputs, batch.receivers,
            model.architecture, 'joint', epoch=0)
        views = ([predicted.full, predicted.restricted]
                 if isinstance(predicted, ThermalJointPairPredictions) else [predicted])
        # An all-native read has identical views and is decoded only once by the core.
        if getattr(model.core, 'reader_capability', None) == 'all_native':
            views = views[:1]
        outputs[endpoint] = views
        for name, value in aux.get('work', {}).items():
            if isinstance(value, (int, float)):
                work[f'{endpoint}_{name}'] = value
    if len(outputs['baseline']) != len(outputs['edited']):
        raise ValueError('Endpoint read views differ.')
    endpoint_losses, field_losses, j_losses, p_losses = [], [], [], []
    observations = []
    for base, edit in zip(outputs['baseline'], outputs['edited']):
        base_batch, edit_batch = pair['baseline'], pair['edited']
        endpoint_losses.append(sum(float(provider.native_weights[n]) * v.mean()
            for branch, batch in ((base, base_batch), (edit, edit_batch))
            for n, v in provider._role_losses(branch, batch.targets).items()) / 2)
        with torch.set_grad_enabled(torch.is_grad_enabled() and response_grad):
            b, e = _field_components(base, base_batch), _field_components(edit, edit_batch)
            weighted = []
            for name in FIELD_ROLES:
                bp, bt, bm = b[name]; ep, et, em = e[name]
                if bp.shape != ep.shape or bt.shape != et.shape:
                    raise ValueError(f'Original response receiver shape changed for {name}.')
                mask = bm & em
                if not bool(mask.any()):
                    raise ValueError(f'No common physical response support for {name}.')
                scale = float(scales[name]['scale'] if isinstance(scales[name], dict) else scales[name])
                error = (ep - bp - (et - bt)) / scale
                weighted.append(float(provider.native_weights[name]) * _equal_module_mean(error, mask))
            field_losses.append(sum(weighted) / sum(float(provider.native_weights[n]) for n in FIELD_ROLES))
            functions = []
            for branch, batch, raw in ((base, base_batch, pair['raw_baseline']), (edit, edit_batch, pair['raw_edited'])):
                material = branch.native_main['pred_internal_temperature'][..., 0]
                pressure = branch.native_main['pred_field'][..., 2]
                prep = branch.native_prepared
                matmask = (prep.stencils['material'].valid.reshape(material.shape) & prep.source_present[..., None].bool()
                           & torch.isfinite(batch.targets.material_targets))
                fluidmask = (batch.targets.point_weights.bool() & prep.stencils['fluid'].valid.reshape(pressure.shape)
                             & torch.isfinite(batch.targets.field_targets[..., 2]))
                functions.append(native_design_functionals(material, matmask, pressure, fluidmask,
                    batch.receivers.fluid_xy[..., 0], raw['config']['domain']['lx'], tau))
            losses = []
            for index, name in enumerate(('J', 'P')):
                scale = float(scales[name]['scale'] if isinstance(scales[name], dict) else scales[name])
                losses.append(((functions[1][index] - functions[0][index] - physical_delta[name]) / scale).square())
            j_losses.append(losses[0]); p_losses.append(losses[1])
            observations.append({'baseline': b, 'edited': e, 'functionals': functions})
    mean = lambda items: sum(items) / len(items)
    return {'endpoint': mean(endpoint_losses), 'field_delta': mean(field_losses),
            'J_delta': mean(j_losses), 'P_delta': mean(p_losses),
            'observations': observations, 'work': work}


def _reference_delta_squares(pair):
    """Physical common-support mean square changes, without model predictions."""
    b, e = pair['baseline'].targets, pair['edited'].targets
    result = {}
    for i, name in enumerate(FIELD_ROLES[:5]):
        bv, ev = b.field_targets[..., i], e.field_targets[..., i]
        mask = b.point_weights.bool() & e.point_weights.bool() & torch.isfinite(bv) & torch.isfinite(ev)
        result[name] = (ev - bv, mask)
    for name, channel in (('temperature/surface', 0), ('q_proxy', 1)):
        bv, ev = b.interface_target[..., channel], e.interface_target[..., channel]
        mask = b.interface_valid_mask.bool() & e.interface_valid_mask.bool() & torch.isfinite(bv) & torch.isfinite(ev)
        if channel == 1:
            for endpoint, raw in (('baseline', pair['raw_baseline']), ('edited', pair['raw_edited'])):
                valid = torch.as_tensor(raw['outside_valid'], device=mask.device)
                padded = torch.zeros_like(mask)
                padded[:, :valid.shape[0]] = valid
                mask &= padded
        result[name] = (ev - bv, mask)
    bv, ev = b.material_targets, e.material_targets
    present = pair['baseline'].scene_inputs.structure['module_present'].bool()[..., None]
    mask = present & torch.isfinite(bv) & torch.isfinite(ev)
    result['temperature/material'] = (ev - bv, mask)
    return result


def _fit_native_response_scales(provider, corpus, physical_labels, repeatability, floors):
    """Fit response RMS on all FIT actions only; retained in memory and metadata."""
    stats = {}
    for family in corpus.family_ids('fit'):
        actions = {}
        for action in ACTIONS:
            pair = response_pair(provider, corpus, family, action, role='fit')
            fields = _reference_delta_squares(pair)
            item = {}
            for name, (values, mask) in fields.items():
                if not bool(mask.any()):
                    raise ValueError(f'No physical common-support receiver for {family}/{action}/{name}.')
                value = float(values[mask].double().square().mean().sqrt().cpu())
                item[name] = (np.asarray([value]), np.asarray([True]))
            item.update(physical_labels[(family, action)])
            actions[action] = item
        stats[family] = actions
    model = provider.model
    state_scales = {name: float(model.field_std[i]) for i, name in enumerate(FIELD_ROLES[:5])}
    state_scales.update({'temperature/surface': float(model.interface_target_std[0]),
                         'temperature/material': float(model.internal_temperature_std[0]),
                         'q_proxy': float(model.interface_target_std[1])})
    return corpus.fit_scales(stats, state_scales, repeatability, floors)


from .r13_8_anchored_task import (ThermalR138AnchoredMemberRoutingTask, ARM_DETAIL,
    DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA, DETAIL_SOURCE_KIND, _VARIANTS)
from .r13_6_member_routing_task import _build_thermal_member_routing_task


def _restore_response_scales(scales):
    """Restore the original loss denominators without device-dependent refitting."""
    import copy
    from numbers import Real

    if not isinstance(scales, dict) or set(scales) != {*FIELD_ROLES, 'J', 'P'}:
        raise ValueError('Saved response scales must contain exactly the ten native roles.')
    for name, item in scales.items():
        if not isinstance(item, dict) or set(item) != {'scale', 'fit_rms', 'floor'}:
            raise ValueError(f'Invalid saved response scale fields for {name}.')
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
               for value in item.values()):
            raise ValueError(f'Nonfinite or nonnumeric saved response scale for {name}.')
        if (item['scale'] <= 0 or item['floor'] <= 0 or item['fit_rms'] < 0
                or item['scale'] != max(item['fit_rms'], item['floor'])):
            raise ValueError(f'Inconsistent saved response scale for {name}.')
    return copy.deepcopy(scales)


class ThermalStateResponseTask(ThermalR138AnchoredMemberRoutingTask):
    """Fresh native state model with four FIT families of response per epoch."""
    auxiliary_loss_names = ('response/endpoint', 'response/field_delta', 'response/J_delta', 'response/P_delta')

    def configure_state_response(self, spec, *, response_scales=None):
        self.response_spec = dict(spec)
        declared_scales = spec.get('response_scales')
        if declared_scales is not None:
            declared_scales = _restore_response_scales(declared_scales)
            if response_scales is not None and _restore_response_scales(response_scales) != declared_scales:
                raise ValueError('Restored response scales differ from the recipe denominators.')
            response_scales = declared_scales
        if spec.get('pilot') is not None and response_scales is None:
            raise ValueError('Pilot supervision must restore the original FIT response scales.')
        self.response_finite_response_coefficient = _response_coefficient(spec.get('finite_response_coefficient', .1))
        self._response_epoch = None
        self._response_epoch_families = []
        self._response_epoch_actions = []
        self.response_corpus = ThermalResponseCorpus(spec['recipe_path'], spec['ledger_path'])
        root = Path(spec['recipe_path']).parent.parent
        fit_labels = root / 'evidence/matched200/epoch200/fit/response_evaluation.json'
        self.response_labels = load_saved_functional_responses(fit_labels, self.response_corpus, role='fit')
        if response_scales is None:
            repeatability = json.loads((root / 'evidence/train_resolution_candidates.json').read_text())['floors']
            self.response_scales = _fit_native_response_scales(self, self.response_corpus, self.response_labels,
                                                               repeatability, spec['functional_floors'])
        else:
            self.response_scales = _restore_response_scales(response_scales)
        self.response_pilot = None
        if spec.get('pilot') is not None:
            pilot = spec['pilot']
            if pilot.get('arm') not in ('control', 'coverage'):
                raise ValueError('Pilot supervision requires a control or coverage arm.')
            coefficient = _response_coefficient(pilot['coefficient'])
            if coefficient <= 0:
                raise ValueError('Pilot supervision coefficient must be positive.')
            self.response_pilot = ThermalActionCorpus(pilot['manifest_path'])
            self.response_pilot_coefficient = coefficient
            self.auxiliary_loss_names = type(self).auxiliary_loss_names + tuple(
                name.replace('response/', 'response_pilot/') for name in type(self).auxiliary_loss_names)

    def _record_response_action(self, epoch, family, action, role):
        if self._response_epoch != epoch:
            self._response_epoch, self._response_epoch_families = epoch, []
            self._response_epoch_actions = []
        if not hasattr(self, '_response_epoch_actions'):
            self._response_epoch_actions = []
        row = {'case_id': str(family), 'action': action, 'role': role}
        if row not in self._response_epoch_actions:
            self._response_epoch_actions.append(row)

    def fit_sampler_epoch(self, local_epoch):
        """Return the FIT family-selection age while preserving local bookkeeping."""
        if type(local_epoch) is not int or local_epoch < 1:
            raise ValueError('FIT sampler requires a positive local epoch.')
        return local_epoch

    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        if update_count < 1:
            raise ValueError('Auxiliary response requires at least one native update.')
        families = self.response_corpus.fit_families_for_epoch(self.fit_sampler_epoch(epoch), self.seed)
        slots = np.linspace(0, update_count - 1, 4, dtype=int)
        for family, slot in zip(families, slots):
            if int(slot) != int(update_index):
                continue
            for action in ACTIONS:
                pair = response_pair(self, self.response_corpus, family, action, role='fit')
                losses = predict_response_pair(self, model, pair, self.response_scales,
                    self.response_labels[(family, action)], self.response_spec['tau'],
                    response_grad=self.response_finite_response_coefficient != 0)
                terms = response_loss_terms(losses['endpoint'], losses['field_delta'],
                    {'J': losses['J_delta'], 'P': losses['P_delta']}, multiplier=update_count / 16,
                    finite_response_coefficient=self.response_finite_response_coefficient)
                work = dict(losses['work'])
                work.update(response_actions=1, response_families=int(action == ACTIONS[0]))
                self._record_response_action(epoch, family, action, 'original_fit')
                if action == ACTIONS[0] and family not in self._response_epoch_families:
                    self._response_epoch_families.append(family)
                yield terms, work
                del pair, losses, terms, work
        pilot = getattr(self, 'response_pilot', None)
        if pilot is not None:
            arm = self.response_spec['pilot']['arm']
            selected = pilot.pilot_actions_for_epoch(arm, epoch)
            slots = np.linspace(0, update_count - 1, len(selected), dtype=int)
            for (family, action), slot in zip(selected, slots):
                if int(slot) != int(update_index):
                    continue
                pair = response_pair(self, pilot, family, action, role='fit')
                losses = predict_response_pair(self, model, pair, self.response_scales,
                    pilot.physical_delta(family, action, role='fit'), self.response_spec['tau'],
                    response_grad=self.response_finite_response_coefficient != 0)
                terms = response_loss_terms(losses['endpoint'], losses['field_delta'],
                    {'J': losses['J_delta'], 'P': losses['P_delta']},
                    multiplier=self.response_pilot_coefficient * update_count / len(selected),
                    finite_response_coefficient=self.response_finite_response_coefficient)
                terms = {name.replace('response/', 'response_pilot/'): term for name, term in terms.items()}
                work = dict(losses['work'])
                work.update(response_pilot_actions=1)
                self._record_response_action(epoch, family, action, f'pilot_{arm}')
                yield terms, work
                del pair, losses, terms, work

    def response_epoch_family_ids(self, epoch):
        """Return the families actually processed so far in the requested epoch."""
        return tuple(self._response_epoch_families) if self._response_epoch == epoch else ()

    def response_epoch_action_ids(self, epoch):
        """Return the actual action IDs processed in this epoch, including pilot."""
        return tuple(dict(row) for row in self._response_epoch_actions) if self._response_epoch == epoch else ()

    def auxiliary_diagnostic_values(self, terms, *, update_count):
        """Per-action diagnostics, normalized for the trainer's update mean."""
        pilot_terms = next(iter(terms)).startswith('response_pilot/')
        count = (len(self.response_pilot.pilot_actions_for_epoch(self.response_spec['pilot']['arm'], self._response_epoch))
                 if pilot_terms else 16)
        return {name: float(term.numerator.detach()) * (update_count / count)
                for name, term in terms.items() if not name.endswith('/endpoint')}

    def loss_metadata(self):
        result = {}
        weights = _response_objective_weights(self.response_finite_response_coefficient)
        for name, label, key in (
                ('response/endpoint', 'paired endpoint state', 'endpoint'),
                ('response/field_delta', 'aligned field response', 'field_delta'),
                ('response/J_delta', 'material smooth-maximum response', 'J_delta'),
                ('response/P_delta', 'pressure-drop response', 'P_delta')):
            diagnostic_only = key != 'endpoint' and self.response_finite_response_coefficient == 0
            if diagnostic_only:
                label += ' (diagnostic only)'
            result[name] = {'panel': 'physics', 'label': label, 'unit': 'scaled squared error',
                            'formula': 'equal FIT family and action mean; paired native reads where applicable',
                            'weight': str(weights[key])}
        if getattr(self, 'response_pilot', None) is not None:
            for name, item in tuple(result.items()):
                result[name.replace('response/', 'response_pilot/')] = {
                    **item, 'label': 'pilot ' + item['label'],
                    'formula': 'equal auxiliary action mean, separately from original FIT mean',
                    'weight': str(float(item['weight']) * self.response_pilot_coefficient)}
        return result

    def identity_payload(self):
        result = super().identity_payload()
        result['native_only_training'] = False
        result['response_training_family_ids'] = list(self.response_corpus.family_ids('fit'))
        result['response_supervision'] = 'original FIT48 edited labels; holdout edited labels excluded'
        result['objective'].update(response_training_families=list(self.response_corpus.family_ids('fit')), response_fit=True, paired_full_restricted=getattr(self.model.core, 'reader_capability', None) != 'all_native')
        all_native = getattr(self.model.core, 'reader_capability', None) == 'all_native'
        result['loss_lineage']['paired_paths'] = (['all_native'] if all_native else ['full_positive_gate', 'restricted_joint95'])
        result['training_reader_objective'] = ('all-native once' if all_native else 'mean full positive-gate and restricted joint95 native loss')
        result['restricted_evaluation_execution'] = 'all_native' if all_native else 'joint95'
        result['joint_member_coverage'] = None if all_native else 0.95
        result['thermal_state_response'] = dict(self.response_spec)
        if 'finite_response_coefficient' in self.response_spec:
            result['response_objective_weights'] = _response_objective_weights(self.response_finite_response_coefficient)
        result['response_scales'] = self.response_scales
        result['response_fit_scope'] = 'original FIT48, all four actions; TRAIN-only state scales'
        if getattr(self, 'response_pilot', None) is not None:
            result['response_pilot_sampling'] = {
                'arm': self.response_spec['pilot']['arm'],
                'actions_per_epoch': len(self.response_pilot.pilot_actions_for_epoch(
                    self.response_spec['pilot']['arm'], 1)),
                'coefficient': self.response_pilot_coefficient,
                'reduction': 'original FIT mean plus coefficient times separate equal pilot action mean',
                'selection': 'epoch-derived three-cell cycle with stable multi-member layout offsets; M1 v0/a0',
                'response_scales': 'restored original FIT48 only'}
        return result


def build_thermal_state_response_task(recipe, *, arm=ARM_DETAIL, device='cpu', response_scales=None,
                                      saved_adapter_state=None, saved_normalization_source=None):
    """Build the maintained native task with fresh TRAIN normalizer and FIT data."""
    from ..data.datasets import H5Normalizer
    from .joint_task import _read_selected_cases
    spec = recipe['thermal_state_response']
    if spec.get('schema') != 'thermal_state_response_v1' or spec.get('enabled') is not True or spec.get('families_per_epoch') != 4:
        raise ValueError('Expected enabled thermal_state_response_v1 with four families per epoch.')
    model, provider = _build_thermal_member_routing_task(recipe, arm=arm, device=device,
        task_type=ThermalStateResponseTask, variants={DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], 'arm': arm}},
        section_schema=DETAIL_CONFIG_SCHEMA, source_kind=DETAIL_SOURCE_KIND,
        lineage='Thermal state response', required_address_mode=None,
        saved_adapter_state=saved_adapter_state, saved_normalization_source=saved_normalization_source)
    # Development TRAIN150 membership does not imply DEV22 validation here.
    val_ids = tuple(case_id for case_id in provider.catalog_case_ids
                    if provider.case_split_by_id[case_id] == 'test' and case_id != '0273')
    if len(val_ids) != 89:
        raise ValueError('Expected the canonical exposed VAL89 population.')
    normalizer = H5Normalizer({k: np.asarray(v, dtype=np.float32) for k, v in model.normalization_stats.items()})
    provider.validation_cases = _read_selected_cases(provider.data_path, val_ids, split='test', normalizer=normalizer, include_grid=False,
                                                    include_raster_geometry=model.core.module_input_dim == 10)
    provider.validation_case_ids = val_ids
    provider._validation_index_by_id = {value: index for index, value in enumerate(val_ids)}
    provider.validation_scope = 'canonical89_exposed_original_test_excluding_0273'
    provider._validation_batches_cache = None
    provider.configure_state_response(spec, response_scales=response_scales)
    return model, provider
