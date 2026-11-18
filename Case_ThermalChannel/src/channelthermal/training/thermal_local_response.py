"""Thermal input neighborhoods and common-fluid observations for R17-7.

The source charts retain prescribed JSON precision. Assignments align geometry
only and carry each heat with its original module; they do not imply identity.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from types import MethodType

import h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..native_observations import native_cell_center_stencil


@dataclass(frozen=True)
class ThermalNeighborhoodObservation:
    coordinates: torch.Tensor
    references: torch.Tensor
    mask: torch.Tensor
    masses: torch.Tensor
    eligible_counts: tuple
    total_masses: torch.Tensor
    retained_weight_fractions: tuple


def thermal_compatibility_key(config):
    """Compare saved global physics, excluding only documented irrelevant fields."""
    flow = copy.deepcopy(config['flow'])
    re = float(flow.pop('re'))
    nu = flow.pop('nu', None)
    if re <= 0:
        raise ValueError('Reynolds number must be positive.')
    expected = (2 * float(config['domain']['module_radius']) * float(flow['u_in'])
                * float(flow.get('viscosity_scale', 1.0)) / re)
    runtime_nu = config.get('runtime', {}).get('nu')
    for value in (nu, runtime_nu):
        if value is not None and not np.isclose(float(value), expected, rtol=1e-6, atol=1e-12):
            raise ValueError('Saved viscosity is inconsistent with the prescribed Reynolds number.')
    local = config['local_module']
    selected_local = {k: local[k] for k in ('local_grid_size', 'n_interface_points')}
    value = {'domain': config['domain'], 'flow': flow, 'thermal': config['thermal'],
             'local_module': selected_local, 'count': int(config['layout']['num_modules']),
             'physical_assumptions': config.get('physical_assumptions'),
             'target_converged': config.get('runtime', {}).get('converged'),
             'pressure_saved': config.get('save', {}).get('save_pressure'),
             'vorticity_saved': config.get('save', {}).get('save_vorticity'),
             'temperature_saved': config.get('save', {}).get('save_temperature'),
             'final_window_frames': config.get('save', {}).get('final_window_frames')}
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def thermal_chart(center, candidate, *, reynolds_range):
    """Return complete dimensionless input and FP32 representation discrepancy."""
    if thermal_compatibility_key(center) != thermal_compatibility_key(candidate):
        raise ValueError('Incompatible Thermal prescribed physics.')
    m = int(center['layout']['num_modules'])
    scale = np.array([center['domain']['lx'], center['domain']['ly']], dtype=np.float64)
    heat_scale = float(center['thermal']['heat_power_max'] - center['thermal']['heat_power_min'])
    if m < 1 or np.any(scale <= 0) or heat_scale <= 0 or reynolds_range <= 0:
        raise ValueError('Chart scales must be positive.')
    anchor = np.asarray(center['layout']['centers'], dtype=np.float64)
    points = np.asarray(candidate['layout']['centers'], dtype=np.float64)
    heat = np.asarray(candidate['layout']['heat_powers'], dtype=np.float64)
    if anchor.shape != (m, 2) or points.shape != (m, 2) or heat.shape != (m,) or not np.isfinite(points).all() or not np.isfinite(heat).all():
        raise ValueError('Malformed prescribed Thermal inputs.')
    cost = np.square((anchor[:, None, :] - points[None, :, :]) / scale).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    best = float(cost[rows, cols].sum())
    alternatives = []
    if m > 1:
        for row, col in zip(rows, cols):
            other = cost.copy(); other[row, col] = np.inf
            r, c = linear_sum_assignment(other)
            alternatives.append(float(other[r, c].sum()))
    gap = min(alternatives) - best if alternatives else None
    root = math.sqrt(m)
    def chart(p, h, reynolds):
        return np.concatenate(((p[cols] / scale / root).reshape(-1), h[cols] / heat_scale / root,
                               [reynolds / reynolds_range]))
    re = float(candidate['flow']['re'])
    source = chart(points, heat, re)
    represented = chart(points.astype(np.float32).astype(np.float64),
                        heat.astype(np.float32).astype(np.float64), float(np.float32(re)))
    return source, source - represented, cols.tolist(), gap


def build_thermal_input_definitions(data_path, case_ids, *, reynolds_range=None, max_scenes=12):
    """Build deterministic input-only neighborhoods, with exact stored grids checked."""
    from honf_runtime.local_response_contrasts import build_local_contrast, build_uniform_contrast
    ids = tuple(map(str, case_ids))
    if len(set(ids)) != len(ids) or len(ids) < 2 or max_scenes < 2:
        raise ValueError('Unique case IDs and at least two possible scenes are required.')
    with h5py.File(data_path, 'r') as h5:
        groups = h5['cases']
        configs = {i: json.loads(groups[i]['case_config_json'][()]) for i in ids}
        grids = {i: tuple(np.asarray(groups[i][axis]) for axis in ('x_grid', 'y_grid')) for i in ids}
        interpretation = {i: tuple(str(groups[i].attrs.get(k)) for k in
            ('channel_order', 'field_dim', 'target_mode', 'converged')) for i in ids}
    keys = {i: thermal_compatibility_key(configs[i]) for i in ids}
    re_values = [float(configs[i]['flow']['re']) for i in ids]
    re_range = float(reynolds_range if reynolds_range is not None else max(re_values) - min(re_values))
    if re_range <= 0:
        raise ValueError('TRAIN Reynolds range must be positive.')
    neighborhoods = []
    for center in ids:
        comparisons = []
        for candidate in ids:
            if keys[center] != keys[candidate] or interpretation[center] != interpretation[candidate] or any(not np.array_equal(a, b) for a, b in zip(grids[center], grids[candidate])):
                continue
            chart, error, assignment, gap = thermal_chart(configs[center], configs[candidate], reynolds_range=re_range)
            comparisons.append((candidate, chart, error, assignment, gap))
        anchor = next(item for item in comparisons if item[0] == center)
        neighbors = sorted((item for item in comparisons if item[0] != center),
                           key=lambda item: (float(np.linalg.norm(item[1] - anchor[1])), item[0]))
        selected = [anchor, *neighbors[:max_scenes - 1]]
        if len(selected) < 2:
            continue
        # Centering and Helmert projection cannot amplify the error matrix's
        # spectral norm beyond this bound, including anchor subtraction.
        errors = np.stack([item[2] for item in selected])
        floor = float(2 * np.linalg.norm(errors, ord=2) / math.sqrt(len(selected) - 1))
        local = build_local_contrast(np.stack([item[1] for item in selected]), absolute_floor=floor)
        uniform = build_uniform_contrast(len(selected))
        neighborhoods.append({'uniform_operator': uniform.operator.tolist(),
            'multisecant_operator': local.operator.tolist(), 'rank': local.rank,
            'singular_values': local.singular_values.tolist(), 'numerical_floor': local.numerical_floor,
            'center_id': center, 'scene_ids': [item[0] for item in selected],
            'chart': [item[1].tolist() for item in selected], 'representation_floor': floor,
            'assignments': [item[3] for item in selected], 'assignment_cost_gaps': [item[4] for item in selected]})
    return {'version': 1, 'data_path': str(data_path), 'case_ids': list(ids), 'reynolds_range': re_range,
            'chart_axes': 'assigned x/lx/sqrt(M), y/ly/sqrt(M), heat/heat_range/sqrt(M), Re/TRAIN_range',
            'input_scales': [list(values) for values in sorted({(float(configs[i]['domain']['lx']), float(configs[i]['domain']['ly']),
                float(configs[i]['thermal']['heat_power_max'] - configs[i]['thermal']['heat_power_min'])) for i in ids})],
            'max_scenes': max_scenes,
            'assignment': 'geometry-only; cost gap to best alternative; not physical identity',
            'neighborhoods': neighborhoods}


def save_thermal_input_definitions(path, definitions):
    path = Path(path)
    payload = json.dumps(definitions, separators=(',', ':'), allow_nan=False)
    if len(payload.encode()) > 32 * 1024 * 1024:
        raise ValueError('Input definition exceeds its 32 MiB budget.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + '\n')


def load_thermal_input_definitions(path):
    definitions = json.loads(Path(path).read_text())
    if definitions.get('version') != 1 or not definitions.get('neighborhoods'):
        raise ValueError('Invalid Thermal neighborhood definitions.')
    return definitions


def scheduled_thermal_neighborhoods(definitions, epoch, count, seed):
    """Continue seeded shuffled decks across epochs without dropping the tail."""
    items = definitions['neighborhoods']
    if epoch < 11 or count < 1 or not items:
        raise ValueError('Positive epoch, count and neighborhood deck required.')
    start = (epoch - 11) * count
    result = []
    for position in range(start, start + count):
        deck, index = divmod(position, len(items))
        order = np.random.default_rng(np.random.SeedSequence([int(seed), deck])).permutation(len(items))
        result.append(items[int(order[index])])
    return result


def sample_thermal_neighborhood_observations(groups, lengths, *, seed, total_positions, remainder_rotation=0):
    """Sample each source list against its full common-fluid, per-channel mass."""
    n = len(groups)
    if n < 2 or total_positions < n:
        raise ValueError('At least two scenes and one requested position per source are required.')
    fields = [torch.as_tensor(np.asarray(g['steady_field'])) for g in groups]
    if any(f.shape != fields[0].shape for f in fields) or fields[0].ndim != 3 or fields[0].shape[-1] < 5:
        raise ValueError('Scenes must share one native physical field shape.')
    ny, nx = fields[0].shape[:2]
    lengths = torch.as_tensor(lengths, dtype=torch.float64).reshape(1, 2)
    fluids = [~torch.as_tensor(np.asarray(g['module_mask']), dtype=torch.bool).reshape(-1) for g in groups]
    if any(f.numel() != nx * ny for f in fluids):
        raise ValueError('Module masks must match the physical field.')
    flat = [f[..., :5].reshape(-1, 5) for f in fields]
    rng = np.random.default_rng(seed)
    selected, counts, totals, fractions = [], [], [], []
    base, extra = divmod(total_positions, n)
    for source, group in enumerate(groups):
        rows = np.asarray(group['sampled_points'])
        if rows.ndim != 2 or rows.shape[1] < 2 or not len(rows):
            raise ValueError('Malformed saved receiver list.')
        xy = torch.from_numpy(np.array(rows[:, :2], copy=True)).double()
        weights = torch.as_tensor(np.asarray(group['sampled_point_weights']) if 'sampled_point_weights' in group else np.ones(len(rows)), dtype=torch.float64)
        if weights.shape != (len(rows),) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
            raise ValueError('Saved receiver weights must be finite, nonnegative and have positive mass.')
        stencil = native_cell_center_stencil(xy[None], lengths, nx=nx, ny=ny)
        indices, coeff = stencil.indices[0], stencil.weights[0]
        required = coeff != 0
        geometry = stencil.valid[0]
        for fluid in fluids:
            geometry = geometry & (fluid[indices] | ~required).all(-1)
        eligible = torch.where(geometry)[0]
        count = len(eligible)
        k = base + int((source - remainder_rotation) % n < extra)
        if count < k:
            raise ValueError('Insufficient common-fluid rows for fixed observation budget.')
        valid = geometry[:, None].expand(-1, 5).clone()
        refs = []
        for field in flat:
            corner = field[indices]
            valid &= (torch.isfinite(corner) | ~required[..., None]).all(1)
            safe = torch.where(torch.isfinite(corner), corner, torch.zeros_like(corner))
            refs.append((safe.double() * coeff[..., None]).sum(1))
        totals.append((weights[:, None] * valid).sum(0))
        draw = eligible[torch.from_numpy(rng.choice(count, k, replace=False))]
        selected.append((xy[draw], torch.stack([v[draw] for v in refs]), valid[draw], weights[draw]))
        counts.append(count)
        fractions.append(float(weights[eligible].sum() / weights.sum()))
    totals = torch.stack(totals)
    available = (totals > 0).all(0)
    masks, masses = [], []
    for (xy, _, valid, weights), count, total in zip(selected, counts, totals):
        mask = valid & available
        mass = (count / len(xy) / n) * weights[:, None] / total.clamp_min(torch.finfo(torch.float64).tiny)
        masks.append(mask); masses.append(torch.where(mask, mass, torch.zeros_like(mass)))
    return ThermalNeighborhoodObservation(torch.cat([p[0] for p in selected]),
        torch.cat([p[1] for p in selected], dim=1), torch.cat(masks), torch.cat(masses),
        tuple(counts), totals, tuple(fractions))


class ThermalLocalResponseObjective:
    def __init__(self, provider, *, seed, variant):
        self.provider, self.seed, self.variant = provider, int(seed), variant
        self.scales = provider.model.field_std.detach().reshape(-1)[:5].cpu().double()
        names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid')
        self.weights = torch.tensor([provider.native_weights[name] for name in names], dtype=torch.float64)

    def components(self, model, neighborhood, *, epoch, force_uniform=False, force_multisecant=False):
        from honf_runtime.local_response_contrasts import (CommonObservations, LocalContrast, neighborhood_field_losses)
        from .thermal_family_objective import _stack_family_batches
        ids = tuple(map(str, neighborhood['scene_ids']))
        provider = self.provider
        if len(set(ids)) != len(ids) or len(ids) < 2 or not set(ids) <= set(map(str, provider.training_case_ids)):
            raise ValueError('Neighborhood must contain distinct declared TRAIN cases.')
        if len({provider.case_split_by_id[i] for i in ids}) != 1:
            raise ValueError('A neighborhood cannot cross data partitions.')
        digest = hashlib.blake2b(f'{self.seed}:{epoch}:{ids[0]}'.encode(), digest_size=8).digest()
        with h5py.File(provider.data_path, 'r') as h5:
            groups = [h5['cases'][i] for i in ids]
            configs = [json.loads(g['case_config_json'][()]) for g in groups]
            if len({thermal_compatibility_key(c) for c in configs}) != 1:
                raise ValueError('Neighborhood has incompatible Thermal physics.')
            if any(not np.array_equal(np.asarray(groups[0][axis]), np.asarray(g[axis])) for g in groups[1:] for axis in ('x_grid', 'y_grid')):
                raise ValueError('Neighborhood physical coordinates differ.')
            n = len(ids)
            total = int(round(1536 / n))
            observation = sample_thermal_neighborhood_observations(groups,
                (configs[0]['domain']['lx'], configs[0]['domain']['ly']),
                seed=int.from_bytes(digest, 'little'), total_positions=total,
                remainder_rotation=(epoch - 1) % n)
        split = provider.case_split_by_id[ids[0]]
        cases = provider._load_native_cases(ids)
        batches = [provider._native_case_batch(cases, [i], split=split) for i in range(len(ids))]
        xy = observation.coordinates.to(device=batches[0].receivers.fluid_xy.device,
                                        dtype=batches[0].receivers.fluid_xy.dtype)[None]
        batches = [replace(batch, receivers=replace(batch.receivers, fluid_xy=xy, local_query_points=None)) for batch in batches]
        batch = _stack_family_batches(batches)
        prepared = model.prepare_native(batch.scene_inputs.structure, batch.receivers.fluid_xy,
            physical_heat=batch.receivers.heat, local_query_points=None, ntheta=batch.receivers.ntheta,
            scene_key=batch.scene_inputs.case_ids, field_only=True)
        full, restricted, shared = model.core.decode_paired(prepared.organization, prepared.query_batch)
        predictions = torch.stack([model._extract_native_roles(prepared, view.joint_fields[..., 3:4],
                                     view.joint_fields[..., :3])['pred_field'] for view in (full, restricted)])
        is_uniform = force_uniform or (self.variant != 'multisecant_response' and not force_multisecant)
        contrast = LocalContrast(torch.as_tensor(neighborhood['uniform_operator' if is_uniform else 'multisecant_operator'], dtype=torch.float64),
            0 if is_uniform else neighborhood['rank'], torch.as_tensor(neighborhood['singular_values'], dtype=torch.float64),
            neighborhood['numerical_floor'])
        obs = CommonObservations(observation.references.to(predictions.device), observation.mask.to(predictions.device),
                                 observation.masses.to(predictions.device))
        if not bool(observation.mask.any()):
            zero = predictions.double().sum() * 0
            absolute, response = zero, zero
        else:
            losses = neighborhood_field_losses(predictions, obs, contrast, self.scales, self.weights)
            absolute, response = losses.absolute, losses.response
        counts = {f'local_response_preparation_{k}': int(v) for k, v in prepared.organization.work_counts.items()}
        for name in ('score_rows', 'positive_value_edges', 'eligible_edges'):
            counts['local_response_preparation_context_' + name] = sum(
                int(prepared.organization.work_counts.get(f'context_{route}_{name}', 0))
                for route in ('mm', 'em', 'ee', 'me'))
        counts.update({f'local_response_shared_{k}': int(v) for k, v in shared.items()})
        for name, view in (('full', full), ('restricted', restricted)):
            counts.update({f'local_response_{name}_{k}': int(v) for k, v in view.work_counts.items()})
        counts.update(local_response_episodes=1, local_response_scenes=len(ids),
                      local_response_positions=len(ids) * len(xy[0]),
                      local_response_unique_core_cells=int(prepared.union.unique_counts.sum()),
                      local_response_curl_unique_cells=int(prepared.curl_stencil.unique_counts.sum()),
                      local_response_curl_stencil_entries=int(prepared.curl_stencil.gather_indices.numel()),
                      local_response_interpolation_corners=int((prepared.stencils['fluid'].weights != 0).sum()),
                      local_response_retained_weight=float(np.mean(observation.retained_weight_fractions)))
        for i, name in enumerate(('u', 'v', 'p', 'omega', 'temperature')):
            counts['local_response_eligible_' + name] = int(bool((observation.total_masses[:, i] > 0).all()))
        return {'absolute': absolute, 'response': response}, counts


def build_thermal_local_response_task(recipe, *, arm=None, device='cpu', resume_payload=None, **kwargs):
    """Attach one shared input-selected episode schedule to the native trainer."""
    from honf_runtime.state_response import ResponseObjectiveConfig
    from honf_runtime.unified_training import LossTerm, SamplingKey
    from honf_runtime.gradient_diagnostics import gradient_diagnostics, preserved_state
    from .thermal_native_pair_task import build_thermal_native_pair_task

    spec = recipe.get('thermal_local_response', {})
    variant = spec.get('variant')
    if variant not in ('state_anchors', 'uniform_response', 'multisecant_response'):
        raise ValueError('Unknown Thermal local-response variant.')
    base = copy.deepcopy(recipe)
    base['training_profile'] = 'native_pair'
    base['thermal_native_pair'] = {'variant': 'field_only', 'prediction_batching': 'paired'}
    model, provider = build_thermal_native_pair_task(base, arm=arm, device=device, resume_payload=resume_payload,
                                                     _field_response_base=True)
    path = str(spec['input_definitions'])
    definitions = load_thermal_input_definitions(path)
    reconstructed = build_thermal_input_definitions(definitions['data_path'], definitions['case_ids'])
    if reconstructed != definitions:
        raise ValueError('Saved neighborhood inputs and operators do not match their prescribed source data.')
    if definitions['case_ids'] != list(map(str, provider.training_case_ids)) or str(definitions['data_path']) != str(provider.data_path):
        raise ValueError('Input definitions do not match the TRAIN membership and data source.')
    if len(definitions['neighborhoods']) != len(definitions['case_ids']):
        raise ValueError('Every declared TRAIN center needs a compatible neighborhood.')
    seed = int(spec.get('seed', 0))
    count = {150: 1, 600: 4}.get(len(definitions['case_ids']))
    if count is None:
        raise ValueError('Local response requires declared TRAIN150 or TRAIN600 membership.')
    objective = ThermalLocalResponseObjective(provider, seed=seed, variant=variant)
    names = ('absolute',) if variant == 'state_anchors' else ('absolute', 'response')
    provider.auxiliary_loss_names = tuple('local_response/' + name for name in names)
    state = {'calibration': None}
    binding = {'source_epoch': 10, 'data_path': str(provider.data_path),
        'case_ids': definitions['case_ids'], 'input_definitions': path,
        'neighborhood_ids': [[item['center_id'], item['scene_ids']] for item in definitions['neighborhoods']],
        'chart_axes': definitions['chart_axes'], 'input_scales': definitions['input_scales'],
        'max_scenes': definitions['max_scenes'], 'reynolds_range': definitions['reynolds_range'],
        'seed': seed, 'output_scales': objective.scales.tolist(), 'channel_weights': objective.weights.tolist(),
        'objective': {'absolute_target_ratio': .25, 'response_target_ratio': .10,
                      'response_calibration': 'uniform', 'ramp': '11-50', 'nominal_scene_positions': 1536}}
    def calibration_centers():
        if len(definitions['case_ids']) == 150:
            return [min((item for item in definitions['neighborhoods'] if (len(item['chart'][0]) - 1) // 3 == target),
                        key=lambda item: item['center_id']) for target in (6, 10)]
        return sorted(definitions['neighborhoods'], key=lambda item: item['center_id'])[:4]
    def validate_local_calibration(self, record, *, source_checkpoint):
        if not isinstance(record, dict) or record.get('binding') != {**binding, 'source_checkpoint': str(source_checkpoint)}:
            raise ValueError('Local-response calibration source, data, neighborhoods, scales or objective do not match.')
        current_definitions = load_thermal_input_definitions(path)
        if current_definitions != definitions:
            raise ValueError('Loaded neighborhood definitions changed during this session.')
        values = record.get('coefficients', {})
        if set(values) != {'absolute', 'response'} or any(not math.isfinite(float(v)) or float(v) < 0 for v in values.values()):
            raise ValueError('Local-response calibration coefficients must be finite and nonnegative.')
        norms = record.get('diagnostics', {}).get('groups', {}).get('global', {}).get('norms', {})
        if record.get('calibration_centers') != [item['center_id'] for item in calibration_centers()]:
            raise ValueError('Calibration centers do not match the input-only selection.')
        if set(norms) != {'native', 'absolute', 'uniform', 'multisecant'} or any(
                not math.isfinite(float(value)) or float(value) <= 0 for value in norms.values()):
            raise ValueError('Calibration requires finite, positive measured gradient norms.')
        if not math.isclose(float(values['absolute']), .25 * norms['native'] / norms['absolute'], rel_tol=1e-12) or not math.isclose(
                float(values['response']), .10 * norms['native'] / norms['uniform'], rel_tol=1e-12):
            raise ValueError('Calibration coefficients do not match declared gradient targets.')
        return copy.deepcopy(record)
    provider.validate_local_calibration = MethodType(validate_local_calibration, provider)
    def initialize_local_response_state(self, native_state, validated_calibration):
        return {'native': copy.deepcopy(native_state['native']), 'local_response': {'calibration': copy.deepcopy(validated_calibration)}}
    provider.initialize_local_response_state = MethodType(initialize_local_response_state, provider)
    old_state = getattr(provider, 'training_state_dict', lambda: {})
    old_load = getattr(provider, 'load_training_state_dict', lambda saved: None)
    def training_state_dict(self):
        return {'native': old_state(), 'local_response': copy.deepcopy(state)}
    def load_training_state_dict(self, saved):
        old_load(saved['native'])
        incoming = copy.deepcopy(saved['local_response'])
        record = incoming.get('calibration')
        if record is not None:
            self.validate_local_calibration(record, source_checkpoint=record['binding']['source_checkpoint'])
        state.update(incoming)
    provider.training_state_dict = MethodType(training_state_dict, provider)
    provider.load_training_state_dict = MethodType(load_training_state_dict, provider)
    def calibrate_local_response(self, current_model, *, source_checkpoint):
        selected = calibration_centers()
        case_ids = tuple(dict.fromkeys(i for item in selected for i in item['scene_ids']))
        key = SamplingKey(seed=self.seed, epoch=10, update_index=0, microbatch_index=0, stage='joint',
            arm=self.recipe.get('training', {}).get('sampling_arm', 'R13-3-Relative'),
            sampling_version=self.sampling_version, dataset_id=self.sampling_dataset_id)
        def native():
            for case_id in case_ids:
                batch = self.make_batch((case_id,), key)
                prediction, extra = self.predict_native(current_model, batch.scene_inputs, batch.receivers,
                    current_model.architecture, 'joint', epoch=10)
                terms = self.loss_terms(prediction, batch.targets, 'joint', extra)
                denominators = self.loss_denominators([batch], phase='joint', arm=current_model.architecture)
                yield sum(term.numerator * float(term.weight) / float(denominators[name]) for name, term in terms.items()) / len(case_ids)
        def auxiliary(name, uniform, multisecant=False):
            def pieces():
                for item in selected:
                    components, _ = objective.components(current_model, item, epoch=10, force_uniform=uniform, force_multisecant=multisecant)
                    yield components[name] / len(selected)
            return pieces
        samples = self.training_samples
        try:
            with preserved_state(current_model):
                current_model.train()
                from .r16_5_decision_task import _parameter_groups
                groups = {key: [name for name, _ in members] for key, members in _parameter_groups(current_model).items() if members}
                diagnostic = gradient_diagnostics(current_model, {'native': native, 'absolute': auxiliary('absolute', True),
                    'uniform': auxiliary('response', True), 'multisecant': auxiliary('response', False, True)},
                    parameter_groups=groups, require_nonzero=True)
        finally:
            self.training_samples = samples
        norms = diagnostic['groups']['global']['norms']
        coefficients = {'absolute': .25 * norms['native'] / norms['absolute'],
                        'response': .10 * norms['native'] / norms['uniform']}
        record = {'binding': {**binding, 'source_checkpoint': str(source_checkpoint)}, 'coefficients': coefficients,
                  'calibration_centers': [x['center_id'] for x in selected], 'diagnostics': diagnostic}
        return self.validate_local_calibration(record, source_checkpoint=source_checkpoint)
    provider.calibrate_local_response = MethodType(calibrate_local_response, provider)
    previous_identity = provider.identity_payload
    def identity_payload(self):
        return {**previous_identity(), 'local_response': {'variant': variant, 'input_definitions': path, 'seed': seed,
                'scales': objective.scales.tolist(), 'weights': objective.weights.tolist()}}
    provider.identity_payload = MethodType(identity_payload, provider)
    old_columns = provider.training_history_columns
    def history_columns(self):
        existing = tuple(x for x in old_columns() if not x.startswith('work_counts.native_pair_'))
        counts = ('episodes', 'scenes', 'positions', 'unique_core_cells', 'curl_unique_cells', 'curl_stencil_entries',
                  'interpolation_corners', 'retained_weight', *(f'eligible_{x}' for x in ('u','v','p','omega','temperature')))
        preparation = ('context_score_rows', 'context_positive_value_edges', 'context_eligible_edges',
            'cached_scene_projection_rows_M', 'cached_scene_projection_rows_E',
            'member_key_projection_rows_M', 'member_key_projection_rows_E',
            'member_value_projection_rows_M', 'member_value_projection_rows_E',
            'prepared_e_directory_compile_pair_rows', 'prepared_e_directory_positive_exchange_entries',
            'prepared_e_directory_total_access_entries', 'prepared_e_directory_rows',
            'prepared_e_directory_full_entries', 'prepared_e_directory_joint95_entries',
            'prepared_e_directory_sort_entries')
        return (*existing, *(f'work_counts.local_response_{x}' for x in counts),
                *(f'work_counts.local_response_preparation_{key}' for key in preparation),
                *(f'work_counts.local_response_{stage}_{key}' for stage in ('shared', 'full', 'restricted')
                  for key in model.core._ANCHOR_COUNT_KEYS))
    provider.training_history_columns = MethodType(history_columns, provider)
    old_definitions = getattr(provider, 'training_column_definitions', lambda: {})
    def training_column_definitions(self):
        columns = self.training_history_columns()
        result = {k: v for k, v in old_definitions().items() if k in columns}
        for column in columns:
            if column.startswith('work_counts.local_response_'):
                fractional = column.endswith('retained_weight')
                result[column] = {'dtype': 'float' if fractional else 'int', 'unit': '1' if fractional else 'count',
                                  'meaning': 'Sum across episodes of ' + column.removeprefix('work_counts.').replace('_', ' ')}
        return result
    provider.training_column_definitions = MethodType(training_column_definitions, provider)
    old_metadata = provider.loss_metadata
    def loss_metadata(self):
        result = dict(old_metadata())
        for name in names:
            result['local_response/' + name] = {'panel': 'physics', 'label': 'local response ' + name,
                'unit': 'scaled squared error', 'formula': 'common-fluid weighted fields, equal scenes and views',
                'weight': 'epoch-10 TRAIN calibration and linear ramp'}
        return result
    provider.loss_metadata = MethodType(loss_metadata, provider)
    config = ResponseObjectiveConfig()
    def auxiliary_update_terms(self, current_model, *, epoch, update_index, update_count):
        ramp = config.ramp(epoch)
        if not ramp:
            return
        if state['calibration'] is None:
            raise ValueError('Local-response TRAIN calibration is missing.')
        chosen = scheduled_thermal_neighborhoods(definitions, epoch, count, seed)
        slots = np.linspace(0, update_count - 1, len(chosen), dtype=int)
        for item, slot in zip(chosen, slots):
            if int(slot) == update_index:
                terms, work = objective.components(current_model, item, epoch=epoch)
                yield ({'local_response/' + name: LossTerm(terms[name], 1., ramp *
                        state['calibration']['coefficients'][name] * update_count / len(chosen)) for name in names}, work)
    provider.auxiliary_update_terms = MethodType(auxiliary_update_terms, provider)
    return model, provider
