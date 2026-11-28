#!/usr/bin/env python3
"""Review finite physical observations from already-decoded joint fields.

The generic reducer consumes explicit scene descriptors and decoded values; it
does not call a model. Thermal's observer plugs into the existing full-case
native evaluator so its fixed, alternate and complete-source summaries reuse the
same physical fields. Different scenes never require matching member IDs.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / 'src', PROJECT / 'Case_ThermalChannel/src', PROJECT / 'tools'):
    sys.path.insert(0, str(directory))

import numpy as np
import torch

from honf_runtime.finite_observations import FiniteObservationScene, evaluate_observation_scene, extract_family_predictions
from honf_runtime.unified_training import _atomic_json


def _number(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_number(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _number(item) for key, item in value.items()}
    return value


def summarize_decoded_scene(scene, dependency_lookup, *, configured_weights, frames=None,
                            basis='coordinate', include_row_ids=True, member_of_row=None):
    """Reduce one scene using a callback that returns physical decoded fields.

    ``dependency_lookup(dependencies)`` returns [view, dependency, output], or
    None when any required query is unavailable. An unavailable family remains
    explicit and never receives fabricated values. All results use the fixed
    configured-family denominator, including when a family is missing.
    """
    available, predictions, missing = [], {}, {}
    for family in scene.families:
        physical = dependency_lookup(family.extraction.dependencies)
        if physical is None:
            missing[family.name] = 'one or more required decoded dependency cells are unavailable'
            continue
        if not torch.is_tensor(physical) or physical.ndim != 3 or physical.shape[1] != len(family.extraction.dependencies):
            raise ValueError('Decoded family fields must have [view, dependency, output] shape.')
        values = extract_family_predictions(family, physical)
        if not bool(torch.isfinite(values).all()):
            missing[family.name] = 'required decoded values or their extraction are nonfinite'
            continue
        available.append(family)
        predictions[family.name] = values
    reduced_scene = FiniteObservationScene(scene.scene_id, tuple(available))
    result = evaluate_observation_scene(reduced_scene, predictions, configured_weights=configured_weights,
                                        frames=frames, basis=basis)
    # Zero references turn the same shared reducer into normalized predicted
    # and reference observations, without changing the physical row measure.
    zero_scene = FiniteObservationScene(scene.scene_id, tuple(
        replace(family, references=np.zeros_like(family.references)) for family in available))
    reference_values = {family.name: torch.as_tensor(family.references, device=predictions[family.name].device)
                        .expand_as(predictions[family.name]) for family in available}
    predicted = evaluate_observation_scene(zero_scene, predictions, configured_weights=configured_weights,
                                           frames=frames, basis=basis)
    reference = evaluate_observation_scene(zero_scene, reference_values, configured_weights=configured_weights,
                                           frames=frames, basis=basis)
    family_results = {}
    for family in available:
        actual = result.families[family.name]
        pred = predicted.families[family.name].moments
        ref = reference.families[family.name].moments
        record = {'available': True, 'row_count': len(family.row_ids),
            'measure_sum': float(np.asarray(family.measures, dtype=np.float64).sum()),
            'unit': family.component.unit, 'support': _number(family.support),
            'frame': _number(family.frame), 'time': _number(family.time),
            'extraction': _number(family.extraction_id), 'basis': actual.basis,
            'native_scale': family.component.scale, 'native_weight': family.native_weight,
            'reference_dtype': str(np.asarray(family.references).dtype),
            'coordinate_dtype': str(np.asarray(family.coordinates).dtype),
            'out_of_frame_count': actual.out_of_frame_count,
            'normalized_point_mse': _number(actual.anchor),
            'physical_point_rmse': _number(actual.anchor.sqrt() * family.component.scale),
            'normalized_predicted_moments': _number(pred), 'normalized_reference_moments': _number(ref),
            'normalized_residual_moments': _number(actual.moments),
            'physical_predicted_moments': _number(pred * family.component.scale),
            'physical_reference_moments': _number(ref * family.component.scale)}
        if include_row_ids:
            record['source_row_ids'] = _number(family.row_ids)
            record['source_references'] = _number(family.references)
            record['source_coordinates'] = _number(family.coordinates)
            record['source_measures'] = _number(family.measures)
        if member_of_row is not None:
            members = {member for row in family.row_ids if (member := member_of_row(row)) is not None}
            record['local_member_count'] = len(members) if members else None
        family_results[family.name] = record
    for name in configured_weights:
        if name not in family_results:
            family_results[name] = {'available': False, 'reason': missing.get(name, 'no valid source observations')}
    return {'scene_id': _number(scene.scene_id), 'families': family_results,
            'missing_families': [name for name in configured_weights if not family_results[name]['available']],
            'normalized_weighted_anchor_mse': _number(result.anchor)}


def summarize_scene_changes(left, right, *, configured_weights):
    """Compare scene summaries without matching rows or dividing by changes."""
    fields, total, denominator = {}, None, sum(map(float, configured_weights.values()))
    for name, weight in configured_weights.items():
        first, second = left['families'][name], right['families'][name]
        if not first['available'] or not second['available']:
            fields[name] = {'available': False, 'reason': 'missing source or decoded support in one scene'}
            continue
        identity = ('unit', 'support', 'frame', 'time', 'extraction', 'basis', 'native_scale', 'native_weight')
        if any(first[key] != second[key] for key in identity):
            raise ValueError(f'Incompatible physical observation meaning for {name!r}.')
        predicted = (np.asarray(second['normalized_predicted_moments']) -
                     np.asarray(first['normalized_predicted_moments']))
        reference = (np.asarray(second['normalized_reference_moments']) -
                     np.asarray(first['normalized_reference_moments']))
        error = predicted - reference
        scale = float(first['native_scale'])
        family_mse = np.square(error).mean(axis=-1)
        contribution = family_mse * (float(weight) / denominator)
        total = contribution if total is None else total + contribution
        fields[name] = {'available': True, 'unit': first['unit'], 'support': first['support'],
            'frame': first['frame'], 'time': first['time'], 'basis': first['basis'], 'native_scale': scale,
            'normalized_predicted_change': predicted.tolist(), 'normalized_reference_change': reference.tolist(),
            'normalized_change_error': error.tolist(), 'physical_predicted_change': (predicted * scale).tolist(),
            'physical_reference_change': (reference * scale).tolist(),
            'normalized_change_mse': family_mse.tolist()}
    return {'pair': [left['scene_id'], right['scene_id']], 'families': fields,
            'normalized_weighted_change_mse': None if total is None else total.tolist(),
            'missing_families': [name for name in configured_weights if not fields[name]['available']]}


def _approximation(left, right, configured_weights):
    result = {}
    for name in configured_weights:
        first, second = left['families'][name], right['families'][name]
        if not first['available'] or not second['available']:
            result[name] = {'available': False}
            continue
        result[name] = {'available': True,
            'normalized_predicted_moment_difference': (np.asarray(first['normalized_predicted_moments']) -
                                                        np.asarray(second['normalized_predicted_moments'])).tolist(),
            'normalized_reference_moment_difference': (np.asarray(first['normalized_reference_moments']) -
                                                        np.asarray(second['normalized_reference_moments'])).tolist(),
            'normalized_residual_moment_difference': (np.asarray(first['normalized_residual_moments']) -
                                                       np.asarray(second['normalized_residual_moments'])).tolist()}
    return result


class ThermalDecodedFields:
    """Locate native cell dependencies in one existing physical field prediction."""

    def __init__(self, model, prediction, batch):
        coords = batch.receivers.fluid_xy[0].detach().cpu()
        fields = prediction.native_main['pred_field'][0].detach().cpu()
        valid = prediction.native_prepared.stencils['fluid'].valid[0].detach().cpu()
        lengths = prediction.native_prepared.lengths[0].detach().cpu().to(coords)
        if coords.ndim != 2 or coords.shape[1] != 2 or fields.shape != (len(coords), 5) or valid.shape != (len(coords),):
            raise ValueError('Decoded Thermal physical fields must align with native fluid query rows.')
        self.values = fields[:, [0, 1, 2, 4]]
        self.coordinates = coords
        self.rows = {}
        spacing = lengths / lengths.new_tensor([model.nx, model.ny])
        location = coords / spacing - .5
        rounded = location.round()
        tolerance = min(8 * torch.finfo(location.dtype).eps * max(model.nx, model.ny), 1.e-3)
        good = (torch.isfinite(location).all(-1) & ((location-rounded).abs() <= tolerance).all(-1)
                & (rounded >= 0).all(-1) & (rounded < rounded.new_tensor([model.nx, model.ny])).all(-1)
                & valid.bool())
        cells = rounded.long()
        for row in torch.nonzero(good, as_tuple=True)[0].tolist():
            cell = int(cells[row, 1]) * model.nx + int(cells[row, 0])
            self.rows.setdefault(cell, row)
        self.tolerance = float(spacing.abs().max()) * tolerance + 1.e-7

    def __call__(self, dependencies):
        indices = []
        for dependency in dependencies:
            row = self.rows.get(dependency.native_id)
            if row is None or not np.allclose(self.coordinates[row].numpy(), dependency.coordinates,
                                              rtol=0, atol=self.tolerance):
                return None
            indices.append(row)
        return self.values[indices][None]


def _thermal_member(row_id):
    parts = str(row_id).split(':')
    return parts[1] if len(parts) >= 2 and parts[0] == 'module' else None


def _thermal_reference_source(case_id, family_name):
    group = f'cases/{case_id}/'
    if family_name.startswith('flow/') or family_name == 'temperature/fluid':
        return group + 'sampled_points'
    if family_name == 'temperature/material':
        return group + 'module_internal_temperature with module_internal_mask'
    if family_name in ('temperature/surface', 'q_proxy'):
        return group + 'interface_target with interface_condition_valid_mask'
    if family_name == 'port/outside_temperature':
        return group + 'interface_condition[T_outside]'
    return None


class ThermalFiniteObservationObserver:
    """Composable ``case_observer`` for the existing native or pair evaluator."""

    def __init__(self, binding, model, case_ids, pairs, *, probe_seed=130013, alternate_seed=130014,
                 basis='coordinate', expected_family_scales=None):
        self.binding, self.model = binding, model
        self.case_ids = tuple(map(str, case_ids))
        self.pairs = tuple(tuple(map(str, pair)) for pair in pairs)
        if (not self.case_ids or len(set(self.case_ids)) != len(self.case_ids) or
                len(set(self.pairs)) != len(self.pairs) or
                any(len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(self.case_ids)
                    for pair in self.pairs)):
            raise ValueError('Finite review pairs must be distinct and lie within the explicit case panel.')
        if type(probe_seed) is not int or type(alternate_seed) is not int or min(probe_seed, alternate_seed) < 0 or probe_seed == alternate_seed:
            raise ValueError('Review panels require two different nonnegative seeds.')
        self.probe_seed, self.alternate_seed, self.basis = probe_seed, alternate_seed, basis
        self.weights = binding.finite_observation_family_weights()
        self.expected_family_scales = expected_family_scales
        frame = binding.finite_observation_frame()
        self.frame = frame
        self.frames = {} if frame is None else {frame.identity: frame}
        self.cases = {}

    def __call__(self, case_id, prediction, batch):
        case_id = str(case_id)
        if case_id not in self.case_ids or case_id in self.cases:
            raise ValueError('A finite review case must occur exactly once in its declared panel.')
        if tuple(map(str, batch.targets.case_ids)) != (case_id,):
            raise ValueError('Finite review requires one correctly identified native case at a time.')
        decoded = ThermalDecodedFields(self.model, prediction, batch)
        summaries = {}
        for name, seed in (('fixed', self.probe_seed), ('alternate', self.alternate_seed)):
            scene = self.binding.finite_observation_scene(case_id, count=64, seed=seed)
            self._validate_family_scales(scene)
            summaries[name] = summarize_decoded_scene(scene, decoded, configured_weights=self.weights,
                frames=self.frames, basis=self.basis, member_of_row=_thermal_member)
        complete_builder = getattr(self.binding, 'complete_source_observation_scene_for_evaluation', None)
        if callable(complete_builder):
            # This object is reduced now, never retained in the panel result.
            complete = complete_builder(case_id)
            self._validate_family_scales(complete)
            summaries['complete_source'] = summarize_decoded_scene(complete, decoded, configured_weights=self.weights,
                frames=self.frames, basis=self.basis, include_row_ids=False, member_of_row=_thermal_member)
        else:
            summaries['complete_source'] = None
        for summary in summaries.values():
            if summary is None:
                continue
            for name, record in summary['families'].items():
                record['reference_source'] = _thermal_reference_source(case_id, name)
        self.cases[case_id] = summaries

    def _validate_family_scales(self, scene):
        if self.expected_family_scales is None:
            return
        for family in scene.families:
            expected = self.expected_family_scales.get(family.name)
            if expected is None or float(family.component.scale) != expected:
                raise ValueError(f'Thermal family {family.name!r} disagrees with its saved source scale.')

    def finish(self):
        if set(self.cases) != set(self.case_ids):
            raise ValueError('Finite review did not receive every declared case prediction.')
        panels = ('fixed', 'alternate', 'complete_source')
        pair_results = {panel: [summarize_scene_changes(self.cases[left][panel], self.cases[right][panel],
                                    configured_weights=self.weights)
                                for left, right in self.pairs
                                if self.cases[left][panel] is not None and self.cases[right][panel] is not None]
                        for panel in panels}
        approximations = {}
        for case in self.case_ids:
            rows = self.cases[case]
            approximations[case] = {'fixed_minus_alternate': _approximation(rows['fixed'], rows['alternate'], self.weights),
                'fixed_minus_complete_source': None if rows['complete_source'] is None else
                    _approximation(rows['fixed'], rows['complete_source'], self.weights)}
        return {'case_ids': list(self.case_ids), 'pairs': [list(pair) for pair in self.pairs],
            'fixed_probe_count': 64, 'fixed_probe_seed': self.probe_seed, 'alternate_probe_seed': self.alternate_seed,
            'basis': self.basis, 'frame': None if self.frame is None else asdict(self.frame),
            'configured_family_weights': self.weights, 'unsupported': self.binding.finite_observation_unsupported(),
            'source_measure': ('Sampled fluid native row weights times physical cell area; material disk area per '
                               'valid local-grid source row; port circumference per native angle.'),
            'interpretation': ('The finite panels are exact on their selected existing rows; complete-source '
                               'summaries describe the saved source population, not an unbiased domain integral. '
                               'Natural scene differences can change multiple prescribed inputs and supports.'),
            'cases': self.cases, 'pair_results': pair_results, 'panel_approximations': approximations}


def _verify_review_storage(output):
    """Reserve a bounded local review on the declared mounted DATA2 filesystem."""
    from honf_runtime.storage_policy import _ensure_workspace_storage_bindings

    bindings = _ensure_workspace_storage_bindings()
    mount = Path('/data2')
    root = (mount / 'wanglz/ModularDT').resolve()
    destination = Path(output).expanduser().resolve(strict=False)
    if not os.path.ismount(mount) or not destination.is_relative_to(root):
        raise ValueError('Finite review outputs must resolve beneath the mounted /data2/wanglz/ModularDT root.')
    parent = destination.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if parent.stat().st_dev != mount.stat().st_dev:
        raise ValueError('Finite review output is not on the declared DATA2 filesystem.')
    usage = os.statvfs(mount)
    free = int(usage.f_bavail * usage.f_frsize)
    budget, peak_temporary, headroom = 64 * 1024**2, 64 * 1024**2, 2 * 1024**3
    if free - budget - peak_temporary < headroom:
        raise RuntimeError('Finite review output and temporary space would breach DATA2 headroom.')
    return {'free_bytes_before': free, 'new_output_budget_bytes': budget,
            'peak_temporary_budget_bytes': peak_temporary, 'minimum_free_headroom_bytes': headroom,
            'workspace_bindings': bindings}


def _recorded_pairs(saved):
    identity = saved['experiment_identity']
    finite = identity.get('provider_identity', {}).get('finite_observations', {})
    recipe = identity['recipe']
    for pairs, source in ((finite.get('eligible_pairs'), 'checkpoint finite-observation pairs'),
                          (recipe.get('finite_observations', {}).get('pairs'), 'saved finite-observation recipe'),
                          (recipe.get('observation_response', {}).get('pairs'), 'saved observation-response recipe'),
                          (recipe.get('thermal_field_response', {}).get('pairs'), 'saved field-response recipe')):
        if pairs:
            return tuple(tuple(map(str, pair)) for pair in pairs), source
    return (), 'no inherited natural pairs recorded'


def verify_saved_thermal_observation_identity(binding, saved):
    """Validate checkpoint family weights and saved physical normalization.

    Finite family weights are recorded in the finite objective identity, while
    source scales and physical output offsets belong to the native adapter's
    normalization statistics. The returned family mapping can be supplied to
    ``ThermalFiniteObservationObserver`` to check each emitted scene as well.
    This validates definitions, not selected row or extraction identity.
    """
    identity = saved['experiment_identity']['provider_identity']
    finite = identity.get('finite_observations')
    if finite:
        actual_weights = {name: float(weight) for name, weight in
                          binding.finite_observation_family_weights().items()}
        saved_weights = finite.get('family_weights')
        if not isinstance(saved_weights, dict) or actual_weights != saved_weights:
            raise ValueError('Thermal observation family weights differ from the saved finite identity.')
    stats = identity.get('adapter_state', {}).get('normalization_stats', {})
    model = binding.provider.model
    definitions = (
        ('field_mean', 'field_mean_by_channel', 5, False),
        ('field_std', 'field_std_by_channel', 5, True),
        ('internal_temperature_std', 'internal_temperature_std', 1, True),
        ('interface_target_std', 'interface_targets_std', 2, True),
    )
    normalized = {}
    for attribute, source, size, positive in definitions:
        stored = stats.get(source)
        if stored is None and source == 'interface_targets_std':
            stored = stats.get('interface_target_std')
        if stored is None:
            raise ValueError(f'Thermal checkpoint lacks saved physical normalization {source!r}.')
        saved_values = np.asarray(stored, dtype=np.float32).reshape(-1)
        live_values = getattr(model, attribute).detach().cpu().numpy().reshape(-1)
        if (saved_values.shape != (size,) or live_values.shape != (size,) or
                not np.isfinite(saved_values).all() or (positive and np.any(saved_values <= 0)) or
                not np.array_equal(saved_values, live_values)):
            raise ValueError(f'Thermal restored {attribute!r} differs from saved physical normalization.')
        normalized[attribute] = saved_values.tolist()
    field = normalized['field_std']
    surface = normalized['interface_target_std']
    scales = {'flow/u': field[0], 'flow/v': field[1], 'flow/p': field[2], 'flow/omega': field[3],
              'temperature/fluid': field[4], 'temperature/material': normalized['internal_temperature_std'][0],
              'temperature/surface': surface[0], 'q_proxy': surface[1],
              'port/outside_temperature': field[4]}
    return {'family_weights': 'matched saved finite identity' if finite else 'no finite identity in checkpoint',
            'source_scales_and_field_offsets': 'matched saved native adapter normalization',
            'family_scales': scales, 'selected_row_and_extraction_identity': 'not compared'}


def _validate_train_panel(cases, pairs, membership, *, named_pair_complete_panel=False):
    cases = tuple(map(str, cases))
    if not cases or len(set(cases)) != len(cases) or not set(cases) <= set(map(str, membership)):
        raise ValueError('TRAIN review cases must be distinct members of the saved training membership.')
    ceiling = max(1, math.ceil(.1 * len(membership)))
    # The declared TRAIN150 review panel retains eight complete inherited
    # pairs (16/150 = 10.7%) instead of discarding one case of a pair.
    if (len(membership) == 150 and len(cases) == 16 and named_pair_complete_panel and
            len(pairs) == 8 and {case for pair in pairs for case in pair} == set(cases)):
        ceiling = 16
    if len(cases) > ceiling:
        raise ValueError(f'TRAIN review is bounded to {ceiling} cases for this declared panel.')


def main(argv=None):
    from channelthermal.training.native_checkpoint import build_native_checkpoint_inference
    from channelthermal.training.observation_response import ThermalObservationBinding
    from honf_runtime.compat import load_trusted_checkpoint
    from thermal_field_response_evaluate import ASSESSMENT9_CASES, ASSESSMENT9_PAIRS, evaluate_field_response_panel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    panel = parser.add_mutually_exclusive_group(required=True)
    panel.add_argument('--cases', help='Explicit comma-separated TRAIN case IDs')
    panel.add_argument('--panels', type=Path, help='Saved matched-panel JSON with development/formal TRAIN cases and pairs')
    panel.add_argument('--assessment9', action='store_true', help='Use the existing nine exposed original cases')
    parser.add_argument('--stage', choices=('development', 'formal'), help='Required with --panels')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--query-tile-size', type=int, default=1024)
    parser.add_argument('--basis', choices=('coordinate', 'constant', 'tensor_hats'),
                        help='Override the saved finite-observation basis for this review')
    parser.add_argument('--export-cases', default='', help='At most two case IDs from the selected panel')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if (args.panels is None) != (args.stage is None):
        parser.error('--panels and --stage must be supplied together.')
    if args.query_tile_size < 1:
        parser.error('The inference query tile size must be positive.')
    if args.output.exists():
        raise FileExistsError('Retain the existing finite review instead of replacing it.')
    storage = _verify_review_storage(args.output)
    saved = load_trusted_checkpoint(args.checkpoint, map_location='cpu')
    recipe = saved['experiment_identity']['recipe']
    finite = saved['experiment_identity'].get('provider_identity', {}).get('finite_observations', {})
    basis = args.basis or finite.get('basis') or recipe.get('finite_observations', {}).get('basis', 'coordinate')
    if basis not in ('coordinate', 'constant', 'tensor_hats'):
        raise ValueError('The saved finite-observation basis is unsupported.')
    if args.assessment9:
        cases, pairs = tuple(ASSESSMENT9_CASES), tuple(ASSESSMENT9_PAIRS)
        split, pair_source, named_panel = 'validation', 'declared historically exposed assessment9 pairs', False
        stage_record = {}
    elif args.panels:
        stage_record = json.loads(args.panels.read_text())[args.stage]
        cases = tuple(map(str, stage_record['train_case_ids']))
        pairs = tuple(tuple(map(str, pair)) for pair in stage_record['train_pairs'])
        split, pair_source, named_panel = 'train', f'declared {args.stage} matched panel', True
    else:
        stage_record = {}
        cases = tuple(args.cases.split(','))
        inherited, pair_source = _recorded_pairs(saved)
        pairs = tuple(pair for pair in inherited if set(pair) <= set(cases))
        split, named_panel = 'train', False
    exports = tuple(filter(None, args.export_cases.split(',')))
    if (not cases or len(set(cases)) != len(cases) or len(exports) > 2 or len(set(exports)) != len(exports)
            or not set(exports) <= set(cases) or len(set(pairs)) != len(pairs)
            or any(len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(cases) for pair in pairs)):
        parser.error('Use unique selected cases and at most two exports; inherited pairs must lie within the panel.')
    export_paths = [args.output.parent / f'{args.output.stem}_{mode}_fields' / f'{case}.npz'
                    for mode in ('full', 'restricted') for case in exports]
    if any(path.exists() for path in export_paths):
        raise FileExistsError('A selected field export already exists; preserve the previous review.')
    model, provider = build_native_checkpoint_inference(saved, validation_case_ids=ASSESSMENT9_CASES, device=args.device)
    if split == 'train':
        _validate_train_panel(cases, pairs, provider.training_case_ids, named_pair_complete_panel=named_panel)
    model.core.query_tile_size = args.query_tile_size
    provider.recipe.setdefault('evaluation', {})['inference_query_tile_size'] = args.query_tile_size
    recorded_training_pairs, _ = _recorded_pairs(saved)
    functional_pairs = stage_record.get('training_pairs') or recorded_training_pairs
    functional_pair_source = ('matched panel TRAIN scale pairs' if stage_record.get('training_pairs') else
                              'checkpoint inherited TRAIN pairs' if recorded_training_pairs else
                              'existing input-selected TRAIN scale-pair rule')
    if functional_pairs:
        provider.field_response_pairs = tuple(tuple(map(str, pair)) for pair in functional_pairs)
    evaluation_recipe = copy.deepcopy(recipe)
    if evaluation_recipe.get('training_profile') not in ('local_multisecant_response', 'observation_response'):
        # The existing field evaluator can derive input-only TRAIN scale pairs
        # for native checkpoints using the same ordinary observation profile.
        evaluation_recipe['training_profile'] = 'observation_response'
    binding = ThermalObservationBinding(provider, recipe)
    if args.assessment9:
        binding = binding.for_evaluation(pairs)
    saved_identity_verification = verify_saved_thermal_observation_identity(binding, saved)
    for pair in pairs:
        binding.validate_pair(pair)
    model.eval()
    views = {}
    for mode in ('full', 'restricted'):
        observer = ThermalFiniteObservationObserver(binding, model, cases, pairs, basis=basis,
            expected_family_scales=saved_identity_verification['family_scales'])
        destination = args.output.with_name(f'{args.output.stem}_{mode}{args.output.suffix}')
        native = evaluate_field_response_panel(model, provider, epoch=int(saved['epoch']), split=split,
            case_ids=cases, pairs=pairs, mode=mode, export_cases=exports, output=destination,
            include_thermal_functionals=True, local_contrast_diagnostics=False, recipe=evaluation_recipe,
            case_observer=observer)
        views[mode] = {'native_and_natural_response': native, 'finite_observations': observer.finish()}
    result = {'schema': 'thermal_finite_observation_review_v1', 'checkpoint': str(args.checkpoint.resolve()),
        'epoch': int(saved['epoch']), 'dataset_path': str(provider.data_path), 'case_ids': list(cases),
        'case_count': len(cases), 'training_membership_count': len(provider.training_case_ids),
        'panel': 'assessment9' if args.assessment9 else args.stage or 'explicit_train', 'split': split,
        'pairs': [list(pair) for pair in pairs], 'pair_source': pair_source,
        'functional_scale_pair_source': functional_pair_source, 'device': str(args.device),
        'readers': ['full', 'restricted'], 'views': views, 'export_cases': list(exports),
        'saved_training_probe_recipe_matches_fixed': (None if not finite else
            finite.get('probe_count') == 64 and finite.get('probe_seed') == 130013),
        'saved_training_basis_matches_review': None if not finite else finite.get('basis') == basis,
        'saved_observation_identity_verification': saved_identity_verification,
        'prediction_reuse': 'One native full-case prediction per case and reader supplies all reported field and observation summaries.',
        'storage': storage}
    result['storage']['actual_export_bytes'] = sum(path.stat().st_size for path in export_paths if path.exists())
    json_size = len(json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode('utf-8')) + 1
    if json_size + result['storage']['actual_export_bytes'] > storage.get('new_output_budget_bytes', 64 * 1024**2):
        raise RuntimeError('Finite review output exceeded its declared new-byte budget before JSON publication.')
    _atomic_json(args.output, result)
    print(json.dumps({'output': str(args.output), 'epoch': int(saved['epoch']), 'cases': len(cases), 'pairs': len(pairs)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
