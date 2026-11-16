"""Sparse common-position Thermal observations from the existing native H5.

Each side contributes half the measure. Denominators are computed over its
entire eligible receiver list before sampling, including duplicate rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from types import MethodType

import h5py
import numpy as np
import torch

from ..native_observations import native_cell_center_stencil


@dataclass(frozen=True)
class ThermalFieldObservation:
    coordinates: torch.Tensor
    references: torch.Tensor
    mask: torch.Tensor
    masses: torch.Tensor
    eligible_counts: tuple[int, int]
    total_masses: torch.Tensor
    retained_weight_fractions: tuple[float, float]


def sample_thermal_field_observations(groups, lengths, *, seed, sample_per_side=128):
    """Sample the symmetric fixed-denominator measure without copying fields.

    ``groups`` are two H5-like scene groups on the same native grid. Validity
    requires all nonzero interpolation corners to be fluid in both scenes;
    reference finiteness is assessed separately for each channel.
    """
    if len(groups) != 2 or type(sample_per_side) is not int or sample_per_side < 1:
        raise ValueError('A pair and a positive per-side sample count are required.')
    fields = [torch.as_tensor(np.asarray(g['steady_field'])) for g in groups]
    if fields[0].shape != fields[1].shape or fields[0].ndim != 3 or fields[0].shape[-1] < 5:
        raise ValueError('Pair fields must share one [ny,nx,channel] physical grid.')
    ny, nx = fields[0].shape[:2]
    lengths = torch.as_tensor(lengths, dtype=torch.float64).reshape(1, 2)
    fluid = [~torch.as_tensor(np.asarray(g['module_mask']), dtype=torch.bool).reshape(-1) for g in groups]
    if any(mask.numel() != nx * ny for mask in fluid):
        raise ValueError('Native geometry masks must match the physical field grid.')
    flat = [field[..., :5].reshape(-1, 5) for field in fields]
    rng = np.random.default_rng(seed)
    selected = []
    counts = []
    totals = []
    retained = []
    for group in groups:
        rows = np.asarray(group['sampled_points'])
        if rows.ndim != 2 or rows.shape[1] < 2 or not len(rows):
            raise ValueError('Native sampled receiver rows are missing or malformed.')
        xy = torch.from_numpy(np.array(rows[:, :2], copy=True)).to(torch.float64)
        weights = torch.as_tensor(np.asarray(group['sampled_point_weights']) if 'sampled_point_weights' in group
                                  else np.ones(len(rows)), dtype=torch.float64)
        if weights.shape != (len(rows),) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError('Native observation weights must be finite and nonnegative.')
        stencil = native_cell_center_stencil(xy[None], lengths, nx=nx, ny=ny)
        indices, coefficients = stencil.indices[0], stencil.weights[0]
        required = coefficients != 0
        geometry = stencil.valid[0]
        for mask in fluid:
            geometry = geometry & (mask[indices] | ~required).all(-1)
        eligible = torch.where(geometry)[0]
        count = len(eligible)
        if not count:
            raise ValueError('A source receiver list has no common physical fluid support.')
        if not bool(weights.sum() > 0):
            raise ValueError('A source receiver list must have positive total physical weight.')
        retained.append(float(weights[eligible].sum() / weights.sum()))
        valid = torch.ones((len(rows), 5), dtype=torch.bool)
        references = []
        for field in flat:
            corner = field[indices]
            valid &= (torch.isfinite(corner) | ~required[..., None]).all(1)
            safe = torch.where(torch.isfinite(corner), corner, torch.zeros_like(corner))
            references.append((safe * coefficients.to(safe)[..., None]).sum(1))
        valid &= geometry[:, None]
        total = (weights[:, None] * valid).sum(0)
        draw_count = min(sample_per_side, count)
        draw = eligible[torch.from_numpy(rng.choice(count, draw_count, replace=False))]
        selected.append((xy[draw], torch.stack([ref[draw] for ref in references]), valid[draw], weights[draw]))
        counts.append(count)
        totals.append(total)
    total_masses = torch.stack(totals)
    channel_available = (total_masses > 0).all(0)
    coordinates = torch.cat([part[0] for part in selected])
    references = torch.cat([part[1] for part in selected], dim=1)
    masks, masses = [], []
    for (xy, _, valid, weights), count, total in zip(selected, counts, totals):
        mask = valid & channel_available
        mass = .5 * (count / len(xy)) * weights[:, None] / total.clamp_min(torch.finfo(total.dtype).tiny)
        masks.append(mask)
        masses.append(torch.where(mask, mass, torch.zeros_like(mass)))
    return ThermalFieldObservation(coordinates, references, torch.cat(masks), torch.cat(masses),
                                   tuple(counts), total_masses, tuple(retained))


class ThermalFieldResponseObjective:
    """Prepare two scenes and decode both access views at shared positions."""
    def __init__(self, provider, *, seed):
        self.provider, self.seed = provider, int(seed)
        self.scales = provider.model.field_std.detach().reshape(-1)[:5].cpu().double()
        names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid')
        self.weights = torch.tensor([provider.native_weights[name] for name in names], dtype=torch.float64)

    def components(self, model, pair, *, epoch):
        from honf_runtime.state_response import FieldResponsePair, field_response_losses
        from .thermal_family_objective import _stack_family_batches
        from .thermal_native_pair_task import describe_native_scene

        pair = tuple(map(str, pair))
        provider = self.provider
        if (len(pair) != 2 or len(set(pair)) != 2 or
                len({provider.case_split_by_id[x] for x in pair}) != 1 or
                not set(pair) <= (set(provider.training_case_ids) | set(provider.validation_case_ids))):
            raise ValueError('Field-response pair must lie within one selected TRAIN or validation panel.')
        split = provider.case_split_by_id[pair[0]]
        digest = hashlib.blake2b(f'{self.seed}:{epoch}:{pair[0]}:{pair[1]}'.encode(), digest_size=8).digest()
        with h5py.File(provider.data_path, 'r') as h5:
            groups = [h5['cases'][case_id] for case_id in pair]
            configs = [json.loads(group['case_config_json'][()]) for group in groups]
            descriptions = [describe_native_scene(case_id, split, config)
                            for case_id, config in zip(pair, configs)]
            if descriptions[0].compatibility_key != descriptions[1].compatibility_key:
                raise ValueError('Thermal field pair has incompatible prescribed physics or module count.')
            for axis in ('x_grid', 'y_grid'):
                if not np.array_equal(np.asarray(groups[0][axis]), np.asarray(groups[1][axis])):
                    raise ValueError('Thermal pair coordinates do not share a physical frame.')
            lengths = (configs[0]['domain']['lx'], configs[0]['domain']['ly'])
            if any((c['domain']['lx'], c['domain']['ly']) != lengths for c in configs):
                raise ValueError('Thermal pair domain lengths differ.')
            observation = sample_thermal_field_observations(groups, lengths,
                seed=int.from_bytes(digest, 'little'))
        cases = provider._load_native_cases(pair)
        batches = [provider._native_case_batch(cases, [i], split=split) for i in range(2)]
        xy = observation.coordinates.to(device=batches[0].receivers.fluid_xy.device,
                                        dtype=batches[0].receivers.fluid_xy.dtype)[None]
        batches = [replace(batch, receivers=replace(batch.receivers, fluid_xy=xy,
                                                     local_query_points=None)) for batch in batches]
        batch = _stack_family_batches(batches)
        prepared = model.prepare_native(batch.scene_inputs.structure, batch.receivers.fluid_xy,
            physical_heat=batch.receivers.heat, local_query_points=None, ntheta=batch.receivers.ntheta,
            scene_key=batch.scene_inputs.case_ids, field_only=True)
        full, restricted, work = model.core.decode_paired(prepared.organization, prepared.query_batch)
        predictions = torch.stack([model._extract_native_roles(prepared, view.joint_fields[..., 3:4],
                                     view.joint_fields[..., :3])['pred_field'] for view in (full, restricted)])
        response = FieldResponsePair(predictions, observation.references.to(predictions.device),
            observation.mask.to(predictions.device), observation.masses.to(predictions.device))
        losses = field_response_losses([response], self.scales, self.weights)
        counts = {f'field_response_shared_{k}': int(v) for k, v in work.items()}
        for view_name, view in (('full', full), ('restricted', restricted)):
            counts.update({f'field_response_{view_name}_{key}': int(value)
                           for key, value in view.work_counts.items()})
        counts.update(field_response_pairs=1, field_response_positions=int(len(xy[0]) * 2))
        counts.update(field_response_unique_core_cells=int(prepared.union.unique_counts.sum()),
                      field_response_curl_unique_cells=int(prepared.curl_stencil.unique_counts.sum()),
                      field_response_curl_stencil_entries=int(prepared.curl_stencil.gather_indices.numel()),
                      field_response_interpolation_corners=int((prepared.stencils['fluid'].weights != 0).sum()))
        counts.update(field_response_retained_weight_left=observation.retained_weight_fractions[0],
                      field_response_retained_weight_right=observation.retained_weight_fractions[1])
        for i, name in enumerate(('u', 'v', 'p', 'omega', 'temperature')):
            available = bool((observation.total_masses[:, i] > 0).all())
            counts[f'field_response_eligible_{name}'] = int(available)
            counts[f'field_response_effective_weight_{name}'] = float(self.weights[i]) if available else 0.
        beta = 1.0
        fitted = getattr(provider, 'field_response_weighting', None)
        if fitted is not None:
            from honf_runtime.state_response import input_pair_weight
            from .thermal_native_pair_task import describe_structure_control_pair
            description = describe_structure_control_pair(*pair, *configs)
            beta = input_pair_weight(description, fitted)
            counts['field_response_difference_unweighted'] = float(losses['difference'].detach())
            counts['field_response_difference_pair_weight'] = beta
        difference = losses['difference'] if beta == 1.0 else losses['difference'] * beta
        return {'absolute': losses['absolute'], 'difference': difference}, counts


def select_thermal_field_pairs(data_path, case_ids, *, case_split_by_id):
    """Choose input-local compatible edges from the actual declared membership."""
    from honf_runtime.state_response import select_degree_two_native_pairs
    from .thermal_native_pair_task import describe_native_scene, native_input_distance

    ids = tuple(map(str, case_ids))
    if len(set(ids)) != len(ids) or any(case_split_by_id.get(case_id) not in ('train', 'validation', 'val') for case_id in ids):
        raise ValueError('Pair graph requires unique actual TRAIN or validation IDs.')
    with h5py.File(data_path, 'r') as h5:
        groups = h5['cases']
        configs = {case_id: json.loads(groups[case_id]['case_config_json'][()]) for case_id in ids}
        scenes = [describe_native_scene(case_id, case_split_by_id[case_id], configs[case_id]) for case_id in ids]
        def eligible(left, right):
            first, second = groups[left.case_id], groups[right.case_id]
            if any(not np.array_equal(np.asarray(first[axis]), np.asarray(second[axis]))
                   for axis in ('x_grid', 'y_grid')):
                return False
            domain = configs[left.case_id]['domain']
            try:
                result = sample_thermal_field_observations((first, second), (domain['lx'], domain['ly']),
                                                            seed=0, sample_per_side=1)
            except ValueError:
                return False
            return bool(((result.total_masses > 0).all(0)).any())
        return select_degree_two_native_pairs(scenes, native_input_distance, eligible)


def build_thermal_field_response_task(recipe, *, device='cpu', resume_payload=None):
    """Bind field supervision without changing the ordinary native case order."""
    import copy
    from honf_runtime.state_response import (ResponseObjectiveConfig, attach_field_response,
                                             scheduled_native_pairs, fit_input_pair_weights)
    from honf_runtime.unified_training import LossTerm, SamplingKey
    from .thermal_native_pair_task import build_thermal_native_pair_task
    from honf_runtime.gradient_diagnostics import gradient_diagnostics, preserved_state

    spec = recipe.get('thermal_field_response', {})
    variant = spec.get('variant')
    if variant not in ('field_only', 'field_anchors', 'field_response'):
        raise ValueError('Unknown Thermal field-response variant.')
    base = copy.deepcopy(recipe)
    base['training_profile'] = 'native_pair'
    base['thermal_native_pair'] = {'variant': 'field_only', 'prediction_batching': 'paired'}
    model, provider = build_thermal_native_pair_task(base, device=device, resume_payload=resume_payload,
                                                     _field_response_base=True)
    pairs = tuple(tuple(map(str, pair)) for pair in spec.get('pairs', ()))
    if not pairs:
        _, pairs, _ = select_thermal_field_pairs(provider.data_path, provider.training_case_ids,
                                                  case_split_by_id=provider.case_split_by_id)
    population = len(provider.training_case_ids)
    count = {150: 3, 600: 12}.get(population)
    allowed = set(map(str, provider.training_case_ids))
    if count is None or len(pairs) < count or any(len(p) != 2 or p[0] >= p[1] or not set(p) <= allowed for p in pairs):
        raise ValueError('Declare enough distinct ordered pairs from the actual TRAIN membership.')
    if len(set(pairs)) != len(pairs):
        raise ValueError('Field-response pairs must be unique.')
    provider.field_response_pairs = pairs
    amplitude = spec.get('difference_weight_amplitude')
    if amplitude is not None:
        if variant != 'field_response':
            raise ValueError('Difference weighting requires the field_response variant.')
        from .thermal_native_pair_task import describe_structure_control_pair
        with h5py.File(provider.data_path, 'r') as h5:
            configs = {case_id: json.loads(h5['cases'][case_id]['case_config_json'][()])
                       for pair in pairs for case_id in pair}
            cells = {}
            for pair in pairs:
                group = h5['cases'][pair[0]]
                spacing = [np.min(np.diff(np.unique(np.asarray(group[axis], dtype=np.float64))))
                           for axis in ('x_grid', 'y_grid')]
                cells[pair] = float(max(spacing))
        descriptions = [describe_structure_control_pair(*pair, *(configs[case_id] for case_id in pair),
                                                        source_cell_size=cells[pair])
                        for pair in pairs]
        provider.field_response_weighting = fit_input_pair_weights(descriptions, amplitude)
    else:
        provider.field_response_weighting = None
    seed = int(spec.get('pair_seed', 0))
    objective = ThermalFieldResponseObjective(provider, seed=seed)
    names = (() if variant == 'field_only' else ('absolute',) if variant == 'field_anchors'
             else ('absolute', 'difference'))
    provider.auxiliary_loss_names = tuple('field_response/' + name for name in names)
    old_columns = provider.training_history_columns
    def history_columns(self):
        native_columns = tuple(column for column in old_columns()
                               if not column.startswith('work_counts.native_pair_'))
        work = tuple('work_counts.field_response_' + name for name in (
            'pairs', 'positions', 'unique_core_cells', 'curl_unique_cells', 'curl_stencil_entries',
            'interpolation_corners', 'retained_weight_left', 'retained_weight_right',
            *('eligible_' + channel for channel in ('u', 'v', 'p', 'omega', 'temperature')),
            *('effective_weight_' + channel for channel in ('u', 'v', 'p', 'omega', 'temperature'))))
        if provider.field_response_weighting is not None:
            work += ('work_counts.field_response_difference_unweighted',
                     'work_counts.field_response_difference_pair_weight')
        core = tuple(f'work_counts.field_response_{view}_{key}'
                     for view in ('shared', 'full', 'restricted') for key in model.core._ANCHOR_COUNT_KEYS)
        return (*native_columns, *work, *core)
    provider.training_history_columns = MethodType(history_columns, provider)
    old_column_definitions = getattr(provider, 'training_column_definitions', lambda: {})
    def training_column_definitions(self):
        columns = self.training_history_columns()
        result = {name: definition for name, definition in old_column_definitions().items() if name in columns}
        for column in columns:
            if not column.startswith('work_counts.field_response_'):
                continue
            name = column.removeprefix('work_counts.field_response_')
            fractional = name.startswith(('retained_weight_', 'effective_weight_', 'difference_'))
            if name.startswith('retained_weight_'):
                meaning = 'Sum across executed pairs of the fraction of the source receiver-list weight on common fluid support'
            elif name.startswith('effective_weight_'):
                meaning = 'Sum across executed pairs of the native channel weight when both receiver lists have valid channel mass'
            elif name.startswith('eligible_'):
                meaning = 'Number of executed pairs with valid channel mass on both source receiver lists'
            elif name == 'difference_unweighted':
                meaning = 'Sum of unweighted field difference losses across executed pairs'
            elif name == 'difference_pair_weight':
                meaning = 'Sum of TRAIN-fitted difference-only pair weights across executed pairs'
            else:
                meaning = 'Sum across executed field-response pairs of ' + name.replace('_', ' ')
            result[column] = {'dtype': 'float' if fractional else 'int',
                              'unit': '1' if fractional else 'count', 'meaning': meaning}
        return result
    provider.training_column_definitions = MethodType(training_column_definitions, provider)
    validation_pairs = tuple(tuple(map(str, pair)) for pair in spec.get('validation_pairs', ()))
    state = {'calibration': copy.deepcopy(spec.get('calibration'))}
    old_identity = provider.identity_payload
    def identity_payload(self):
        response_identity = {'variant': variant, 'pairs': pairs, 'pair_seed': seed,
                'validation_pairs': validation_pairs, 'sample_per_side': 128,
                'scales': objective.scales.tolist(), 'weights': objective.weights.tolist()}
        if self.field_response_weighting is not None:
            response_identity['difference_weighting'] = self.field_response_weighting
        return {**old_identity(), 'field_response': response_identity}
    provider.identity_payload = MethodType(identity_payload, provider)
    old_state = getattr(provider, 'training_state_dict', lambda: {})
    old_load = getattr(provider, 'load_training_state_dict', lambda saved: None)
    def training_state_dict(self):
        return {'native': old_state(), 'field_response': copy.deepcopy(state)}
    def load_training_state_dict(self, saved):
        old_load(saved['native'])
        state.update(copy.deepcopy(saved['field_response']))
    provider.training_state_dict = MethodType(training_state_dict, provider)
    provider.load_training_state_dict = MethodType(load_training_state_dict, provider)
    def calibrate_common_prefix(self, current_model, *, epoch=10):
        if epoch != 10:
            raise ValueError('Field-response calibration requires the completed epoch-10 common state.')
        chosen = scheduled_native_pairs(pairs, epoch, count, seed)
        case_ids = tuple(case_id for pair in chosen for case_id in pair)
        key = SamplingKey(seed=self.seed, epoch=epoch, update_index=0, microbatch_index=0,
            stage='joint', arm=self.recipe.get('training', {}).get('sampling_arm', 'R13-3-Relative'),
            sampling_version=self.sampling_version, dataset_id=self.sampling_dataset_id)
        def native():
            for case_id in case_ids:
                batch = self.make_batch((case_id,), key)
                prediction, extra = self.predict_native(current_model, batch.scene_inputs, batch.receivers,
                    current_model.architecture, 'joint', epoch=epoch)
                terms = self.loss_terms(prediction, batch.targets, 'joint', extra)
                denominators = self.loss_denominators([batch], phase='joint', arm=current_model.architecture)
                yield sum(term.numerator * float(term.weight) / float(denominators[name])
                          for name, term in terms.items()) / len(case_ids)
        def auxiliary(name):
            def pieces():
                for pair in chosen:
                    values, _ = objective.components(current_model, pair, epoch=epoch)
                    yield values[name] / len(chosen)
            return pieces
        training_samples = self.training_samples
        try:
            with preserved_state(current_model):
                current_model.train()
                diagnostic = gradient_diagnostics(current_model,
                    {'native': native, 'absolute': auxiliary('absolute'), 'difference': auxiliary('difference')},
                    require_nonzero=True)
        finally:
            self.training_samples = training_samples
        norms = diagnostic['groups']['global']['norms']
        state['calibration'] = {'epoch': epoch, 'pairs': chosen, 'norms': norms,
            'coefficients': {'absolute': .25 * norms['native'] / norms['absolute'],
                             'difference': .10 * norms['native'] / norms['difference']}}
        return copy.deepcopy(state['calibration'])
    provider.calibrate_common_prefix = MethodType(calibrate_common_prefix, provider)
    old_metadata = provider.loss_metadata
    def loss_metadata(self):
        result = dict(old_metadata())
        for name in names:
            result['field_response/' + name] = {'panel': 'physics', 'label': 'field response ' + name,
                'unit': 'scaled squared error', 'formula': 'symmetric weighted common positions, equal views',
                'weight': 'epoch-10 TRAIN gradient calibration and linear ramp'}
            if name == 'difference' and self.field_response_weighting is not None:
                result['field_response/' + name]['formula'] += '; TRAIN-fitted pair-weighted difference'
        return result
    provider.loss_metadata = MethodType(loss_metadata, provider)
    if names:
        config = ResponseObjectiveConfig()
        attach_field_response(provider, pairs=pairs, count=count, seed=seed, component_names=names,
            components=objective.components,
            calibration=lambda: (state['calibration']['coefficients'] if state['calibration'] else None),
            ramp=config.ramp)
        if not validation_pairs or any(len(pair) != 2 or pair[0] >= pair[1] or
                                       not set(pair) <= set(map(str, provider.validation_case_ids))
                                       for pair in validation_pairs):
            raise ValueError('Declare fixed pairs within the selected validation panel.')
        def validation_auxiliary_terms(self, current_model, *, epoch, arm):
            if arm != current_model.architecture:
                raise ValueError('Field-response validation model differs from the selected arm.')
            ramp = config.ramp(epoch)
            calibration = state['calibration']
            if ramp and not calibration:
                raise ValueError('Field-response validation requires shared TRAIN calibration.')
            rows = [objective.components(current_model, pair, epoch=10)[0] for pair in validation_pairs]
            coefficients = calibration['coefficients'] if calibration else {}
            return {'field_response/' + name: LossTerm(sum(row[name] for row in rows), len(rows),
                    ramp * float(coefficients.get(name, 0.))) for name in names}
        provider.validation_auxiliary_terms = MethodType(validation_auxiliary_terms, provider)
    return model, provider
