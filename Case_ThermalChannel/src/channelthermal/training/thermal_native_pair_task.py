"""Native Thermal pair selection and exact physical observations.

The H5 is the ordinary physical-field dataset. This module never loads edited
scenes, and its input-only assignment does not imply material identity.
"""
from __future__ import annotations

import json
import math
import h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from honf_runtime.state_response import NativePairScene, select_native_pairs, validate_common_coordinates
from ..native_functionals import native_design_functionals
from ..inverse.request_search import roi_mask

TAU = 0.4149511814117432
OBSERVABLES = ('J', 'P', 'A', 'B')
NATIVE_LOSS_ORDER = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid',
                     'temperature/surface', 'temperature/material', 'q_proxy')


def native_calibration_record(provider, calibration, *, pairs, seed, scales, floors, source=None):
    """Compact, source-specific definition of one native TRAIN calibration."""
    from pathlib import Path
    return {'schema': 'thermal_native_pair_calibration_v1',
            'training_case_ids': list(map(str, provider.training_case_ids)),
            'native_data_path': str(Path(provider.data_path).expanduser().resolve()),
            'sampling_dataset_id': provider.sampling_dataset_id,
            'sampling_version': provider.sampling_version, 'native_seed': int(provider.seed),
            'native_weights': dict(provider.native_weights),
            'outside_temperature_weight': float(provider.outside_temperature_weight),
            'port_angle_count': int(provider.port_angle_count),
            'architecture': provider.model.architecture,
            'adapter_state': provider.model.state_payload(),
            'pairs': [list(pair) for pair in pairs], 'pair_seed': int(seed),
            'scales': list(map(float, scales)), 'floors': list(map(float, floors)),
            'tau': TAU, 'native_queries': int(provider.primary_queries),
            'definition': {'epoch': 10, 'absolute_gradient_ratio': .25,
                           'contrast_gradient_ratio': .10, 'warmup_epochs': 10,
                           'ramp_epochs': 40, 'views': ['full', 'restricted'],
                           'observables': list(OBSERVABLES)},
            'source': source, 'calibration': calibration}


def validate_native_calibration_record(provider, record, *, pairs, seed, scales, floors):
    """Require matching native membership, complete pairs, scales and definition."""
    import copy
    if not isinstance(record, dict):
        raise ValueError('Shared native calibration must be a JSON object.')
    candidate = copy.deepcopy(record)
    calibration = candidate.pop('calibration', None)
    source = candidate.pop('source', None)
    expected = native_calibration_record(provider, None, pairs=pairs, seed=seed,
                                         scales=scales, floors=floors)
    expected.pop('calibration')
    expected.pop('source')
    if (not isinstance(source, dict) or not isinstance(source.get('run'), str)
            or not source['run'] or not isinstance(source.get('checkpoint'), str)
            or not source['checkpoint'] or type(source.get('completed_epoch')) is not int
            or source['completed_epoch'] < 10):
        raise ValueError('Shared calibration requires its source run and completed epoch.')
    if candidate != expected or not isinstance(calibration, dict):
        raise ValueError('Shared calibration does not match the native TRAIN membership, pairs, scales or definition.')
    from honf_runtime.state_response import scheduled_native_pairs
    expected_pairs = scheduled_native_pairs(pairs, 10, {150: 3, 600: 12}[len(provider.training_case_ids)], seed)
    try:
        actual_pairs = tuple(tuple(pair) for pair in calibration['pairs'])
        norms, coefficients = calibration['norms'], calibration['coefficients']
        if (calibration['epoch'] != 10 or actual_pairs != expected_pairs
                or set(norms) != {'native', 'absolute', 'contrast'}
                or set(coefficients) != {'absolute', 'contrast'}
                or any(not math.isfinite(float(value)) or float(value) <= 0
                       for value in (*norms.values(), *coefficients.values()))
                or float(coefficients['absolute']) != .25 * float(norms['native']) / float(norms['absolute'])
                or float(coefficients['contrast']) != .10 * float(norms['native']) / float(norms['contrast'])):
            raise ValueError('Shared native calibration has invalid epoch, pairs, gradients or coefficients.')
    except (KeyError, TypeError, ZeroDivisionError):
        raise ValueError('Shared native calibration is incomplete.') from None
    return copy.deepcopy(calibration)


def build_native_pair_recipe(base_recipe, *, variant, validation_pairs=(), pair_seed=0):
    """Create a fresh base-data recipe from the native architecture prescription."""
    import copy
    labels = {'field_only': 'Native_Field_Only_R17_3', 'state_anchors': 'Native_State_Anchors_R17_3',
              'functional_contrast': 'Native_Functional_Contrast_R17_3'}
    if variant not in labels:
        raise ValueError('Unknown native-pair comparison.')
    result = copy.deepcopy(base_recipe)
    initialization = result.get('initialization', {})
    if initialization.get('pretrained_weights') or initialization.get('source_checkpoint'):
        raise ValueError('Native pair comparisons require fresh initialization.')
    result.pop('state_response', None)
    result.pop('thermal_joint_response', None)
    result['training_profile'] = 'native_pair'
    result['thermal_native_pair'] = {'variant': variant, 'label': labels[variant],
        'pair_seed': pair_seed, 'validation_pairs': [list(pair) for pair in validation_pairs],
        'prediction_batching': 'paired'}
    return result


def _validate_prediction_batching(value):
    if value not in ('serial', 'paired'):
        raise ValueError("thermal_native_pair.prediction_batching must be 'serial' or 'paired'.")
    return value


def native_pair_schedule_amendment(saved, current, *, source_epoch, stop_after):
    """Validate the R17-3 TRAIN150 schedule correction at an exact resume.

    Existing TRAIN150 starts used a flat schedule. Both schedules have exactly
    the same rate through the authorized 1000-epoch development horizon. This
    changes only how a later authorized extension would decay the rate.
    """
    import copy
    from honf_runtime.unified_training import ScheduleSpec
    old, new = copy.deepcopy(saved), copy.deepcopy(current)
    recipe = new.get('recipe', {})
    training = recipe.get('training', {})
    if (recipe.get('training_profile') != 'native_pair'
            or recipe.get('thermal_native_pair', {}).get('variant') not in ('field_only', 'state_anchors', 'functional_contrast')
            or training.get('membership') != 'fixed25_train150_dev22'
            or training.get('local_horizon') != 1000
            or training.get('optimizer_schedule_horizon') != 2000
            or type(source_epoch) is not int or not 1 <= source_epoch <= 1000
            or type(stop_after) is not int or not source_epoch < stop_after <= 1000):
        raise ValueError('Schedule correction is limited to the existing R17-3 TRAIN150 development lineage through epoch 1000.')
    expected_old = {'peak_lr': .0003, 'warmup_start_lr': .0003, 'warmup_epochs': 0,
                    'hold_through_epoch': 1999, 'total_epochs': 2000, 'final_lr': .0003}
    expected_new = {'peak_lr': .0003, 'warmup_start_lr': .0003, 'warmup_epochs': 0,
                    'hold_through_epoch': 1000, 'total_epochs': 2000, 'final_lr': .00003}
    try:
        old_provider_schedule = old['provider_identity']['training']['optimizer_schedule']
        new_provider_schedule = new['provider_identity']['training']['optimizer_schedule']
        old_groups, new_groups = old['optimizer_schedule_contract'], new['optimizer_schedule_contract']
        if (old_provider_schedule != expected_old or new_provider_schedule != expected_new
                or len(old_groups) != 1 or len(new_groups) != 1
                or old_groups[0]['schedule'] != expected_old or new_groups[0]['schedule'] != expected_new):
            raise ValueError('Unexpected native optimizer schedule contract.')
        old['provider_identity']['training']['optimizer_schedule'] = copy.deepcopy(expected_new)
        old_groups[0]['schedule'] = copy.deepcopy(expected_new)
    except (KeyError, TypeError, IndexError):
        raise ValueError('Missing native optimizer schedule identity.') from None
    if old != new:
        raise ValueError('Schedule correction cannot change model, data, sampling, objective or other identity.')
    old_schedule, new_schedule = ScheduleSpec(**expected_old), ScheduleSpec(**expected_new)
    if any(old_schedule.value(epoch) != new_schedule.value(epoch) for epoch in range(1, 1001)):
        raise ValueError('Learning rates are not exactly equal throughout the authorized development horizon.')
    return {'kind': 'native_pair_train150_schedule_correction',
            'schedule_before': expected_old, 'schedule_after': expected_new,
            'identical_learning_rates_through_epoch': 1000,
            'trajectory_note': 'All executed and authorized development learning rates are identical; the corrected cosine schedule applies only beyond epoch 1000 if separately authorized. Model, optimizer moments, RNG, sampler, data and objective are restored from the checkpoint.'}


def native_pair_inference_identity_matches(recipe, saved, current):
    """Accept the known old TRAIN150 schedule for read-only inference only."""
    import copy
    training = recipe.get('training', {})
    if (recipe.get('training_profile') != 'native_pair'
            or recipe.get('thermal_native_pair', {}).get('variant') not in ('field_only', 'state_anchors', 'functional_contrast')
            or training.get('membership') != 'fixed25_train150_dev22'
            or training.get('optimizer_schedule_horizon') != 2000):
        return False
    old_schedule = {'peak_lr': .0003, 'warmup_start_lr': .0003, 'warmup_epochs': 0,
                    'hold_through_epoch': 1999, 'total_epochs': 2000, 'final_lr': .0003}
    new_schedule = {'peak_lr': .0003, 'warmup_start_lr': .0003, 'warmup_epochs': 0,
                    'hold_through_epoch': 1000, 'total_epochs': 2000, 'final_lr': .00003}
    try:
        if (saved['training']['optimizer_schedule'] != old_schedule
                or current['training']['optimizer_schedule'] != new_schedule):
            return False
        comparison = copy.deepcopy(saved)
        comparison['training']['optimizer_schedule'] = new_schedule
        return comparison == current
    except (KeyError, TypeError):
        return False


def describe_native_scene(case_id, split, config):
    """Describe compatibility using only prescribed physical inputs."""
    layout = config['layout']
    count = int(layout['num_modules'])
    centers = np.asarray(layout['centers'], dtype=np.float64)
    heat = np.asarray(layout['heat_powers'], dtype=np.float64)
    if centers.shape != (count, 2) or heat.shape != (count,) or not np.isfinite(centers).all() or not np.isfinite(heat).all():
        raise ValueError('Malformed prescribed module positions or heat controls.')
    # Exclude layout seed, runtime and solver output. The full prescribed domain,
    # flow, thermal and local material/boundary settings define compatibility.
    key = json.dumps({name: config[name] for name in ('domain', 'flow', 'thermal', 'local_module')}, sort_keys=True)
    domain = config['domain']
    xy_scale = np.asarray([domain['lx'], domain['ly']], dtype=np.float64)
    heat_scale = float(config['thermal']['heat_power_max'] - config['thermal']['heat_power_min'])
    if np.any(xy_scale <= 0) or heat_scale <= 0:
        raise ValueError('Native input scales must be positive.')
    descriptor = np.column_stack((centers / xy_scale, heat / heat_scale))
    return NativePairScene(str(case_id), str(split), (key, count), descriptor)


def native_input_distance(left, right):
    """Minimum assignment distance; assignments are never material identities."""
    delta = left[:, None, :] - right[None, :, :]
    cost = np.square(delta).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].mean()))


def describe_structure_control_pair(left_id, right_id, left_config, right_config, *, source_cell_size=None):
    """Compare positions by geometry-only assignment, then assigned heat controls."""
    from honf_runtime.state_response import InputPairDescription
    left = describe_native_scene(left_id, 'train', left_config)
    right = describe_native_scene(right_id, 'train', right_config)
    if left.compatibility_key != right.compatibility_key:
        raise ValueError('Pair requires equal prescribed physics and module count.')
    lx = left_config['layout']
    rx = right_config['layout']
    domain = left_config['domain']
    scale = np.asarray([domain['lx'], domain['ly']], dtype=np.float64)
    first = np.asarray(lx['centers'], dtype=np.float64)
    second = np.asarray(rx['centers'], dtype=np.float64)
    delta = (first[:, None, :] - second[None, :, :]) / scale
    cost = np.square(delta).sum(axis=-1)
    rows, cols = linear_sum_assignment(cost)
    displacement = cost[rows, cols]
    heat_scale = float(left_config['thermal']['heat_power_max'] - left_config['thermal']['heat_power_min'])
    heats = (np.asarray(lx['heat_powers'], dtype=np.float64)[rows]
             - np.asarray(rx['heat_powers'], dtype=np.float64)[cols]) / heat_scale
    participation = (float(displacement.sum() ** 2 / (len(rows) * np.square(displacement).sum()))
                     if displacement.any() else None)
    details = {'frame': 'physical domain coordinates', 'position_units': 'domain lengths',
        'control': 'assigned heat powers', 'control_scale': heat_scale,
        'assignment': 'geometry-only; not persistent member identity',
        'displacement_participation': participation,
        'other_prescribed_inputs': 'domain, flow, thermal and local_module match exactly'}
    if source_cell_size is not None:
        cell = float(source_cell_size)
        if not math.isfinite(cell) or cell <= 0:
            raise ValueError('Source-cell resolution must be finite and positive.')
        details['source_cell_size'] = cell
        details['members_moved_over_source_cell'] = int((np.linalg.norm(first[rows] - second[cols], axis=1) > cell).sum())
    return InputPairDescription((str(left_id), str(right_id)), len(rows),
        float(np.sqrt(displacement.mean())), float(np.sqrt(np.square(heats).mean())),
        details)


def colocate_native_pairs(case_order, pairs):
    """Place matched partners together while preserving first appearance order."""
    order = tuple(case_order)
    if len(set(order)) != len(order):
        raise ValueError('Native case order must contain unique original IDs.')
    partner = {}
    for left, right in pairs:
        if left == right or left in partner or right in partner or left not in order or right not in order:
            raise ValueError('Native matching must be disjoint and lie in the case order.')
        partner[left], partner[right] = right, left
    seen, result = set(), []
    for case_id in order:
        if case_id in seen:
            continue
        result.append(case_id)
        seen.add(case_id)
        paired = partner.get(case_id)
        if paired is not None:
            result.append(paired)
            seen.add(paired)
    return tuple(result)


def _coordinates(group):
    x, y = np.asarray(group['x_grid']), np.asarray(group['y_grid'])
    if x.ndim == y.ndim == 1:
        x, y = np.meshgrid(x, y, indexing='xy')
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError('Malformed native physical coordinates.')
    return np.stack((x, y), axis=-1)


def native_pair_support(left, right, region_config):
    """Pair-defined regions, using the lower-ID scene geometry for both."""
    xy = _coordinates(left)
    other_xy = _coordinates(right)
    field_left, field_right = np.asarray(left['steady_field']), np.asarray(right['steady_field'])
    mask_left = ~np.asarray(left['module_mask'], dtype=bool) & np.isfinite(field_left[..., 4])
    mask_right = ~np.asarray(right['module_mask'], dtype=bool) & np.isfinite(field_right[..., 4])
    common = validate_common_coordinates(xy, other_xy, (mask_left, mask_right))
    domain, layout = region_config['domain'], region_config['layout']
    centers = np.asarray(layout['centers'], dtype=np.float64)
    result = {}
    for name in ('A', 'B'):
        region = roi_mask(xy.reshape(-1, 2), (domain['lx'], domain['ly']), centers,
                          np.ones(len(centers), dtype=bool), float(domain['module_radius']), name)
        result[name] = common & np.asarray(region).reshape(common.shape)
        if not result[name].any():
            raise ValueError(f'Pair has no common physical fluid support in region {name}.')
    return result


def native_reference_observations(group, config, regions):
    """J/P use each scene's support; A/B use the pair's common support."""
    xy, field = _coordinates(group), np.asarray(group['steady_field'])
    fluid = ~np.asarray(group['module_mask'], dtype=bool) & np.isfinite(field[..., 2])
    count = int(config['layout']['num_modules'])
    material = np.asarray(group['module_internal_temperature'])[:count]
    mask = np.broadcast_to(np.asarray(group['module_internal_mask'], dtype=bool), material.shape) & np.isfinite(material)
    j, p = native_design_functionals(torch.from_numpy(material), torch.from_numpy(mask),
        torch.from_numpy(field[..., 2]), torch.from_numpy(fluid), torch.from_numpy(xy[..., 0]),
        config['domain']['lx'], TAU)
    regional = [torch.from_numpy(field[..., 4][regions[name]]).double().mean() for name in ('A', 'B')]
    return torch.stack((j, p, *regional))


def select_thermal_native_pairs(data_path, case_ids, *, split='train'):
    """Input-only nearest disjoint compatible pairs with complete native support."""
    ids = tuple(map(str, case_ids))
    with h5py.File(data_path, 'r') as h5:
        groups = h5['cases']
        configs = {case_id: json.loads(groups[case_id]['case_config_json'][()]) for case_id in ids}
        scenes = [describe_native_scene(case_id, split, configs[case_id]) for case_id in ids]
        def eligible(left, right):
            try:
                regions = native_pair_support(groups[left.case_id], groups[right.case_id], configs[left.case_id])
                native_reference_observations(groups[left.case_id], configs[left.case_id], regions)
                native_reference_observations(groups[right.case_id], configs[right.case_id], regions)
            except ValueError:
                return False
            return True
        return select_native_pairs(scenes, native_input_distance, eligible)


def native_pair_references(data_path, pairs):
    """Return source-derived per-pair observations with no stored field copies."""
    rows = []
    with h5py.File(data_path, 'r') as h5:
        for left_id, right_id in pairs:
            if str(left_id) >= str(right_id):
                raise ValueError('Pairs must be oriented by ascending original case ID.')
            left, right = h5['cases'][str(left_id)], h5['cases'][str(right_id)]
            left_config = json.loads(left['case_config_json'][()])
            right_config = json.loads(right['case_config_json'][()])
            regions = native_pair_support(left, right, left_config)
            rows.append(torch.stack((native_reference_observations(left, left_config, regions),
                                     native_reference_observations(right, right_config, regions))))
    if not rows:
        raise ValueError('At least one native pair is required.')
    return torch.stack(rows)


def native_pair_scales(references, floors):
    """TRAIN pair RMS changes with case-declared native-resolution floors."""
    if references.ndim != 3 or references.shape[1:] != (2, 4) or not len(references):
        raise ValueError('Expected pair, scene, four-observable reference values.')
    floor = torch.as_tensor(floors, dtype=torch.float64)
    if floor.shape != (4,) or not bool(torch.isfinite(floor).all()) or bool((floor <= 0).any()):
        raise ValueError('All four TRAIN-derived physical floors must be positive.')
    scales = torch.maximum((references[:, 1] - references[:, 0]).double().square().mean(0).sqrt(), floor)
    if not bool(torch.isfinite(scales).all()):
        raise ValueError('Native pair references must be finite.')
    return scales


class ThermalNativePairObjective:
    """Dense absolute and four-observable losses on native scenes."""
    def __init__(self, provider, pairs, *, training_scales=None):
        self.provider = provider
        self.prediction_batching = _validate_prediction_batching(
            getattr(provider, 'native_pair_prediction_batching', 'serial'))
        self.pairs = tuple(tuple(pair) for pair in pairs)
        self.references = native_pair_references(provider.data_path, self.pairs)
        field_std = provider.model.field_std.detach().cpu().reshape(-1).double()
        # The saved physical arrays are FP32. Resolution floors use the native
        # normalization scale times one source-precision machine epsilon.
        eps = float(np.finfo(np.float32).eps)
        self.floors = [float(field_std[4]) * eps, float(field_std[2]) * eps,
                       float(field_std[4]) * eps, float(field_std[4]) * eps]
        self.scales = (native_pair_scales(self.references, self.floors) if training_scales is None
                       else torch.as_tensor(training_scales, dtype=torch.float64))
        self._index = {pair: i for i, pair in enumerate(self.pairs)}

    def observe_pair(self, model, pair, *, epoch):
        """Return prediction views and native references in J/P/A/B order."""
        from .thermal_family_objective import (
            _endpoint_functionals, _split_family_prediction, _stack_family_batches, _views,
        )
        pair = tuple(pair)
        splits = {self.provider.case_split_by_id[case_id] for case_id in pair}
        if len(splits) != 1:
            raise ValueError('A native pair cannot cross dataset splits.')
        cases = self.provider._load_native_cases(pair)
        batches = [self.provider._native_case_batch(cases, [i], split=next(iter(splits))) for i in range(2)]
        with h5py.File(self.provider.data_path, 'r') as h5:
            groups = [h5['cases'][case_id] for case_id in pair]
            configs = [json.loads(group['case_config_json'][()]) for group in groups]
            regions = native_pair_support(groups[0], groups[1], configs[0])
        predictions, work = [], {}
        batching = _validate_prediction_batching(self.prediction_batching)
        prediction_batches = [_stack_family_batches(batches)] if batching == 'paired' else batches
        for batch in prediction_batches:
            pred, auxiliary = self.provider.predict_native(model, batch.scene_inputs, batch.receivers,
                model.architecture, 'joint', epoch=epoch)
            if batching == 'paired':
                predictions.extend(_split_family_prediction(pred, len(batches), allow_single=True))
            else:
                predictions.append(pred)
            for name, value in auxiliary.get('work', {}).items():
                if isinstance(value, (int, float)):
                    key = 'native_pair_' + name
                    work[key] = work.get(key, 0.) + value
        absolute = sum(self.provider.endpoint_state_loss(p, b) for p, b in zip(predictions, batches)) / 2
        by_scene = []
        for pred, batch, config in zip(predictions, batches, configs):
            functions = _endpoint_functionals(pred, batch, {'config': config}, TAU)
            branch_rows = []
            for view_index, branch in enumerate(_views(pred)):
                prepared = branch.native_prepared
                material = branch.native_main['pred_internal_temperature'][..., 0]
                material_target = batch.targets.material_targets
                material_required = (prepared.source_present[..., None].bool()
                                     & torch.isfinite(material_target))
                material_valid = prepared.stencils['material'].valid.reshape_as(material)
                if not bool(material_valid[material_required].all()):
                    raise ValueError('Native pair J requires every saved valid material receiver.')
                field = branch.native_main['pred_field'][..., 4]
                valid = branch.native_prepared.stencils['fluid'].valid.reshape_as(field)
                x = batch.receivers.fluid_xy[..., 0]
                pressure_band = (x <= .08 * config['domain']['lx']) | (x >= .92 * config['domain']['lx'])
                pressure_required = (pressure_band & batch.targets.point_weights.bool()
                                     & torch.isfinite(batch.targets.field_targets[..., 2]))
                if not bool(valid[pressure_required].all()):
                    raise ValueError('Native pair P requires all saved valid pressure-band receivers.')
                regional = []
                for name in ('A', 'B'):
                    mask = torch.as_tensor(regions[name].reshape(1, -1), device=field.device)
                    if field.shape != mask.shape or not bool(valid[mask].all()):
                        raise ValueError('Native pair regional prediction stencil is invalid.')
                    regional.append(field[mask].double().mean())
                branch_rows.append(torch.cat((functions[view_index], torch.stack(regional))))
            by_scene.append(torch.stack(branch_rows))
        # [view, pair=1, scene=2, observable=4]
        observations = torch.stack(by_scene, dim=1).unsqueeze(1)
        reference = self.references[self._index[pair]].unsqueeze(0).to(observations.device)
        work.update(native_pair_presentations=1, native_pair_scenes=2)
        return observations[:, 0], reference[0], absolute, work

    def components(self, model, pair, *, epoch):
        from honf_runtime.state_response import native_functional_contrast
        observations, reference, absolute, work = self.observe_pair(model, pair, epoch=epoch)
        if observations.shape[0] != 2:
            raise ValueError('Native pair fitting needs full and restricted predictions.')
        contrast = native_functional_contrast(observations.unsqueeze(1), reference.unsqueeze(0), self.scales)
        return {'absolute': absolute, 'contrast': contrast}, work


def build_thermal_native_pair_task(recipe, *, arm=None, device='cpu', resume_payload=None,
                                   _field_response_base=False, _fresh_coupled_base=False):
    """Build fresh native-only, matched-anchor or matched-contrast training."""
    import copy
    from types import MethodType
    from honf_runtime.state_response import ResponseObjectiveConfig, scheduled_native_pairs
    from honf_runtime.unified_training import LossTerm, SamplingKey
    from .thermal_joint_response_task import build_thermal_joint_response_task
    from honf_runtime.gradient_diagnostics import gradient_diagnostics, preserved_state

    spec = recipe.get('thermal_native_pair', {})
    variant = spec.get('variant')
    if variant not in ('field_only', 'state_anchors', 'functional_contrast'):
        raise ValueError('Choose a declared native-pair variant.')
    prediction_batching = _validate_prediction_batching(spec.get('prediction_batching', 'serial'))
    clean = copy.deepcopy(recipe)
    clean['training_profile'] = 'state_response'
    clean['state_response'] = {'enabled': False}
    clean['thermal_joint_response'] = {'schema': 'thermal_joint_response_v1',
        'native_ports': {'angle_count': 64, 'outside_temperature_weight': .05}}
    if clean.get('training', {}).get('membership') in ('full_train600', 'scratch_train600_exposed_val89'):
        # Select the historical formal native schedule without attaching an
        # edited-family corpus or inheriting any historical weights.
        clean['thermal_joint_response']['paired'] = {}
    model, provider = build_thermal_joint_response_task(clean, arm=arm, device=device, resume_payload=resume_payload)
    provider.native_pair_prediction_batching = prediction_batching
    if (not _fresh_coupled_base
            and int(clean.get('training', {}).get('optimizer_schedule_horizon', 0)) != 2000):
        raise ValueError('R17-3 native pair training requires its 2000-epoch optimizer schedule.')
    def native_pair_schedule(self):
        lr = float(self.recipe['optimizer']['lr'])
        if _fresh_coupled_base:
            horizon = int(self.recipe['training']['optimizer_schedule_horizon'])
            if horizon < int(self.recipe['training']['local_horizon']):
                raise ValueError('The fresh coupled schedule must hold through the declared fit horizon.')
            return {'peak_lr': lr, 'warmup_start_lr': lr, 'warmup_epochs': 0,
                    'hold_through_epoch': horizon, 'total_epochs': horizon + 1, 'final_lr': lr}
        final_lr = float(self.recipe['optimizer'].get('final_lr', 3e-5))
        return {'peak_lr': lr, 'warmup_start_lr': lr, 'warmup_epochs': 0,
                'hold_through_epoch': 1000, 'total_epochs': 2000, 'final_lr': final_lr}
    provider._schedule_payload = MethodType(native_pair_schedule, provider)
    provider.resume_identity_amendment = MethodType(
        lambda self, saved, current, *, source_epoch, stop_after: native_pair_schedule_amendment(
            saved, current, source_epoch=source_epoch, stop_after=stop_after), provider)
    provider.training_output_policy = 'csv_v3'
    provider.sampling_audit = bool(spec.get('sampling_audit', False))
    def history_columns(self):
        native_names = (*NATIVE_LOSS_ORDER, 'port/outside_temperature')
        work_names = ('native_pair_presentations', 'native_pair_scenes', 'gradient_norm',
            'paired_query_unique_union_raw_score_pairs_M', 'paired_query_unique_union_raw_score_pairs_E',
            'paired_query_relative_value_pairs_M', 'paired_query_relative_value_pairs_E',
            'paired_query_value_projection_rows_M', 'paired_query_value_projection_rows_E',
            'paired_query_M_member_query_affine_rows', 'paired_query_E_member_query_affine_rows',
            'paired_mediated_source_eligible_alternatives', 'paired_mediated_conditioned_anchor_M_rows',
            'paired_mediated_conditioned_positive_ME_edges', 'paired_mediated_expanded_ME_edge_rows',
            'paired_mediated_source_prefix_sort_entries', 'paired_mediated_E_prefix_sort_entries')
        auxiliary_work_names = tuple('native_pair_' + name for name in work_names if name.startswith('paired_'))
        return (tuple('train_losses.' + name for name in (*native_names, *self.auxiliary_loss_names))
                + tuple('work_counts.' + name for name in (*work_names, *auxiliary_work_names)))
    provider.auxiliary_loss_names = ()
    provider.training_history_columns = MethodType(history_columns, provider)
    def validation_objective_predictions(self, current_model, batch, *, arm, epoch):
        mode = current_model.training
        try:
            current_model.train(True)
            return self.predict_native(current_model, batch.scene_inputs, batch.receivers,
                                       arm, 'joint', epoch=epoch)
        finally:
            current_model.train(mode)
    provider.validation_objective_predictions = MethodType(validation_objective_predictions, provider)
    if _field_response_base:
        return model, provider
    pairs, unmatched = select_thermal_native_pairs(provider.data_path, provider.training_case_ids)
    population = len(provider.training_case_ids)
    count = {150: 3, 600: 12}.get(population)
    if count is None or len(pairs) < count:
        raise ValueError('Native matching cannot provide the declared TRAIN150/600 pair exposure.')
    def reorder_epoch_cases(self, ordered_cases, *, epoch, seed):
        del epoch, seed
        return colocate_native_pairs(ordered_cases, pairs)
    provider.reorder_epoch_cases = MethodType(reorder_epoch_cases, provider)
    if variant == 'field_only':
        return model, provider
    objective = ThermalNativePairObjective(provider, pairs)
    provider.native_pair_objective = objective
    state = {'calibration': None, 'adoptions': []}
    schedule_seed = int(spec.get('pair_seed', 0))
    config = ResponseObjectiveConfig()
    names = ('absolute', 'contrast') if variant == 'functional_contrast' else ('absolute',)
    provider.auxiliary_loss_names = tuple('native_pair/' + name for name in names)
    def shared_record(self, calibration, *, source_epoch):
        return native_calibration_record(self, calibration, pairs=pairs, seed=schedule_seed,
            scales=objective.scales, floors=objective.floors,
            source={'run': str(self.native_pair_calibration_source_run),
                    'checkpoint': str(self.native_pair_calibration_source_checkpoint),
                    'checkpoint_note': 'The latest checkpoint alias may advance after publication.',
                    'completed_epoch': int(source_epoch)})
    def read_shared(self, path):
        from pathlib import Path
        record = json.loads(Path(path).expanduser().read_text())
        return validate_native_calibration_record(self, record, pairs=pairs, seed=schedule_seed,
                                                   scales=objective.scales, floors=objective.floors)
    provider.native_pair_calibration_record = MethodType(shared_record, provider)
    provider.read_native_pair_calibration = MethodType(read_shared, provider)

    def adopt_shared(self, path, *, source_epoch):
        from pathlib import Path
        before = copy.deepcopy(state['calibration'])
        after = self.read_native_pair_calibration(path)
        source_record = json.loads(Path(path).expanduser().read_text())['source']
        if state['adoptions']:
            previous = state['adoptions'][-1]
            canonical = str(Path(path).expanduser().resolve())
            if (previous.get('source') != canonical
                    or previous.get('calibration_source') != source_record
                    or json.dumps(before, sort_keys=True) != json.dumps(after, sort_keys=True)
                    or json.dumps(previous.get('adopted_calibration'), sort_keys=True) != json.dumps(after, sort_keys=True)):
                raise ValueError('This lineage already adopted a different shared native calibration.')
            return None
        if source_epoch >= 11 and before is None:
            raise ValueError('A completed auxiliary epoch requires the saved original calibration.')
        entry = {'kind': 'native_pair_shared_calibration_adoption',
                 'source': str(Path(path).expanduser().resolve()), 'source_epoch': int(source_epoch),
                 'calibration_source': source_record,
                 'previous_calibration': before, 'adopted_calibration': copy.deepcopy(after)}
        state['calibration'] = after
        state['adoptions'].append(entry)
        return copy.deepcopy(entry)
    provider.adopt_native_pair_calibration = MethodType(adopt_shared, provider)
    validation_pairs = tuple(tuple(map(str, pair)) for pair in spec.get('validation_pairs', ()))
    allowed_validation = set(map(str, provider.validation_case_ids))
    if not validation_pairs or any(len(pair) != 2 or not set(pair) <= allowed_validation for pair in validation_pairs):
        raise ValueError('Declare at least one pair within the selected validation panel.')
    validation_objective = ThermalNativePairObjective(provider, validation_pairs, training_scales=objective.scales)

    def begin_training_epoch(self, current_model, epoch):
        current_model.core.query_tile_size = 128
        destination = getattr(self, 'native_pair_calibration_output', None)
        if destination and state['calibration'] is not None:
            from pathlib import Path
            from honf_runtime.unified_training import _atomic_json
            path = Path(destination).expanduser()
            record = self.native_pair_calibration_record(state['calibration'], source_epoch=epoch - 1)
            if path.exists():
                existing = json.loads(path.read_text())
                self.read_native_pair_calibration(path)
                if (existing['source']['run'] != record['source']['run']
                        or json.dumps(existing['calibration'], sort_keys=True)
                        != json.dumps(record['calibration'], sort_keys=True)):
                    raise ValueError('Existing shared calibration differs from this anchor calibration.')
            else:
                _atomic_json(path, record)
            self.native_pair_calibration_output = None
        if epoch != 11 or state['calibration'] is not None:
            return None
        source = getattr(self, 'native_pair_calibration_input', None)
        if source:
            self.adopt_native_pair_calibration(source, source_epoch=10)
            self.native_pair_calibration_input = None
            return None
        if destination:
            from pathlib import Path
            path = Path(destination).expanduser()
            if path.exists():
                if json.loads(path.read_text()).get('source', {}).get('run') != str(self.native_pair_calibration_source_run):
                    raise ValueError('Existing shared calibration belongs to a different anchor run.')
                state['calibration'] = self.read_native_pair_calibration(path)
                return None
        selected = scheduled_native_pairs(pairs, 10, count, schedule_seed)
        case_ids = tuple(case_id for pair in selected for case_id in pair)
        key = SamplingKey(seed=self.seed, epoch=10, update_index=0, microbatch_index=0,
            stage='joint', arm=self.recipe.get('training', {}).get('sampling_arm', 'R13-3-Relative'),
            sampling_version=self.sampling_version, dataset_id=self.sampling_dataset_id)
        def native():
            for case_id in case_ids:
                batch = self.make_batch((case_id,), key)
                prediction, auxiliary = self.predict_native(current_model, batch.scene_inputs, batch.receivers,
                    current_model.architecture, 'joint', epoch=10)
                terms = self.loss_terms(prediction, batch.targets, 'joint', auxiliary)
                planned = self.loss_denominators([batch], phase='joint', arm=current_model.architecture)
                yield sum(term.numerator * float(term.weight) / float(planned[name])
                          for name, term in terms.items()) / len(case_ids)
        def auxiliary(name):
            def pieces():
                for pair in selected:
                    components, _ = objective.components(current_model, pair, epoch=10)
                    yield components[name] / len(selected)
            return pieces
        training_samples = self.training_samples
        try:
            with preserved_state(current_model):
                current_model.train()
                diagnostics = gradient_diagnostics(current_model,
                    {'native': native, 'absolute': auxiliary('absolute'), 'contrast': auxiliary('contrast')},
                    require_nonzero=True)
        finally:
            self.training_samples = training_samples
        norms = diagnostics['groups']['global']['norms']
        state['calibration'] = {'epoch': 10, 'pairs': selected, 'norms': norms,
            'coefficients': {'absolute': .25 * norms['native'] / norms['absolute'],
                             'contrast': .10 * norms['native'] / norms['contrast']}}
        if destination:
            from honf_runtime.unified_training import _atomic_json
            _atomic_json(path, self.native_pair_calibration_record(state['calibration'], source_epoch=10))
            self.native_pair_calibration_output = None
        return None
    provider.begin_training_epoch = MethodType(begin_training_epoch, provider)

    def auxiliary_update_terms(self, current_model, *, epoch, update_index, update_count):
        ramp = config.ramp(epoch)
        if not ramp:
            return
        if state['calibration'] is None:
            raise ValueError('Epoch-10 TRAIN calibration is missing.')
        selected = scheduled_native_pairs(pairs, epoch, count, schedule_seed)
        slots = np.linspace(0, update_count - 1, len(selected), dtype=int)
        for pair, slot in zip(selected, slots):
            if int(slot) != update_index:
                continue
            components, work = objective.components(current_model, pair, epoch=epoch)
            yield ({'native_pair/' + name: LossTerm(components[name], 1., ramp *
                state['calibration']['coefficients'][name] * update_count / len(selected)) for name in names}, work)
    provider.auxiliary_update_terms = MethodType(auxiliary_update_terms, provider)
    def validation_auxiliary_terms(self, current_model, *, epoch, arm):
        if arm != current_model.architecture:
            raise ValueError('Native pair validation model does not match the requested arm.')
        ramp = config.ramp(epoch)
        if ramp and state['calibration'] is None:
            raise ValueError('Native pair validation requires the saved TRAIN calibration.')
        current_model.train(True)
        rows = [validation_objective.components(current_model, pair, epoch=epoch)[0] for pair in validation_pairs]
        weights = state['calibration']['coefficients'] if state['calibration'] else {name: 0. for name in names}
        return {'native_pair/' + name: LossTerm(sum(row[name] for row in rows), len(rows),
                    ramp * float(weights[name])) for name in names}
    provider.validation_auxiliary_terms = MethodType(validation_auxiliary_terms, provider)
    old_state = getattr(provider, 'training_state_dict', lambda: {})
    old_load = getattr(provider, 'load_training_state_dict', lambda saved: None)
    def training_state_dict(self):
        return {'native': old_state(), 'native_pair': {'calibration': copy.deepcopy(state['calibration']),
                                                       'adoptions': copy.deepcopy(state['adoptions'])}}
    def load_training_state_dict(self, saved):
        old_load(saved['native'])
        state['calibration'] = copy.deepcopy(saved['native_pair']['calibration'])
        state['adoptions'] = copy.deepcopy(saved['native_pair'].get('adoptions', []))
    provider.training_state_dict = MethodType(training_state_dict, provider)
    provider.load_training_state_dict = MethodType(load_training_state_dict, provider)
    previous_identity = provider.identity_payload
    def identity_payload(self):
        return {**previous_identity(), 'native_pair': {'variant': variant, 'pairs': pairs,
            'unmatched': unmatched, 'pair_seed': schedule_seed, 'pairs_per_epoch': count,
            'validation_pairs': validation_pairs,
            'scales': objective.scales.tolist(), 'floors': objective.floors, 'tau': TAU}}
    provider.identity_payload = MethodType(identity_payload, provider)
    old_metadata = provider.loss_metadata
    def loss_metadata(self):
        result = dict(old_metadata())
        for name in names:
            result['native_pair/' + name] = {'panel': 'physics', 'label': 'native pair ' + name,
                'unit': 'scaled squared error', 'formula': 'equal scene and full/restricted view mean',
                'weight': 'epoch-10 TRAIN gradient calibration and linear ramp'}
        return result
    provider.loss_metadata = MethodType(loss_metadata, provider)
    return model, provider
