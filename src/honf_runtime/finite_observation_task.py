"""Optional finite-observation objectives on an ordinary joint native task.

The adapter owns existing observations, units and linear extraction. This
attachment owns the fixed objective, pair accounting, calibration and recovery.
It never requires pointwise correspondence or a particular native sampling law.
"""
from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from .coupled_response_task import _fixed_pair_order, _metadata_size
from .finite_observations import (compatible_scene_families, evaluate_observation_scene,
                                  extract_family_predictions, finite_scene_contrast)
from .gradient_diagnostics import gradient_diagnostics, preserved_state
from .unified_training import LossTerm, SamplingKey, _atomic_json


_LOSSES = ('finite_observations/anchor', 'finite_observations/contrast')
_TARGETS = {'anchor': .10, 'contrast': .05}
_COUNTS = ('scenes', 'pairs', 'observation_rows', 'requested_dependencies', 'dependencies',
           'neural_queries', 'padded_queries', 'padding_queries', 'local_pairs',
           'scene_preparations', 'reused_scene_preparations',
           'planning_seconds', 'prediction_seconds', 'anchor_raw_sum', 'contrast_raw_sum',
           'reader_full_batches', 'reader_restricted_batches', 'reader_dual_batches')


class _CalibrationKey(SamplingKey):
    """Separate calibration draws while preserving native identity checks."""

    def seed_for(self, *coordinates):
        return super().seed_for('finite_observation_calibration', *coordinates)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def execute_finite_dependencies(binding, model, scenes, *, read_mode=None,
                                native_batch=None, native_predictions=None):
    """Decode each scene's dependency union, preserving the live forward graph.

    An optional adapter callback reuses a batched native preparation. The
    general fallback prepares individual scenes and explicitly charges it.
    Returned values are physical joint fields, before any sparse extraction.
    """
    dependencies, mappings = {}, {}
    requested = rows = 0
    for scene in scenes:
        union, lookup, family_maps = [], {}, {}
        for family in scene.families:
            indices = []
            rows += len(family.row_ids)
            requested += len(family.extraction.dependencies)
            for dependency in family.extraction.dependencies:
                if dependency not in lookup:
                    lookup[dependency] = len(union)
                    union.append(dependency)
                indices.append(lookup[dependency])
            family_maps[family.name] = np.asarray(indices, dtype=np.int64)
        dependencies[str(scene.scene_id)] = tuple(union)
        mappings[str(scene.scene_id)] = family_maps
    work = {'requested_dependencies': requested, 'dependencies': sum(map(len, dependencies.values())),
            'observation_rows': rows}
    callback = getattr(binding, 'execute_finite_observation_dependencies', None)
    if callable(callback):
        physical, measured = callback(model, tuple(dependencies), dependencies, read_mode=read_mode,
                                      native_batch=native_batch, native_predictions=native_predictions)
        work.update(measured)
    else:
        physical = {}
        work.update(neural_queries=0, padded_queries=0, scene_preparations=0,
                    reused_scene_preparations=0)
        for case_id, union in dependencies.items():
            if not union:
                continue
            scene, query, scale, offset = binding.field_request(model, case_id, union)
            prepared = model.core.prepare(scene)
            if read_mode is None:
                full, restricted, shared = model.core.decode_paired(prepared, query)
                raw = torch.stack((full.joint_fields[0], restricted.joint_fields[0]))
                counts = {f'paired_{key}': value for key, value in (shared or {}).items()}
            else:
                with model.core.read_execution(read_mode):
                    output = model.core.decode(prepared, query)
                raw = output.joint_fields
                counts = {f'single_{read_mode}_{key}': value
                          for key, value in (output.work_counts or {}).items()}
            if raw.ndim != 3 or raw.shape[1] != len(union):
                raise ValueError('Finite dependency predictions do not match their union.')
            scale, offset = (torch.as_tensor(value, device=raw.device) for value in (scale, offset))
            dtype = torch.promote_types(raw.dtype, torch.promote_types(scale.dtype, offset.dtype))
            physical[case_id] = raw.to(dtype) * scale.to(dtype) + offset.to(dtype)
            work['neural_queries'] += len(union)
            work['padded_queries'] += int(query.coordinates.shape[0] * query.coordinates.shape[1])
            work['scene_preparations'] += 1
            for key, value in {**{f'prepare_{key}': value for key, value in
                                  (prepared.work_counts or {}).items()}, **counts}.items():
                if isinstance(value, (int, float)):
                    work[key] = work.get(key, 0) + value
    extracted = {}
    for scene in scenes:
        case_id = str(scene.scene_id)
        extracted[case_id] = {
            family.name: extract_family_predictions(family, physical[case_id],
                            dependency_indices=mappings[case_id][family.name])
            for family in scene.families}
    return extracted, work


def attach_finite_observation_task(recipe, model, provider, binding):
    """Add an ordinary scalar objective, without projection or detached paths."""
    spec = copy.deepcopy(dict(recipe['finite_observations']))
    variant = spec.get('variant', 'contrasts')
    if variant not in ('disabled', 'anchors', 'contrasts'):
        raise ValueError('finite_observations.variant must be disabled, anchors or contrasts.')
    if getattr(provider, 'native_priority_composition', None) is not None:
        raise ValueError('Finite observations use an ordinary scalar loss, without a gradient composer.')
    required = ('finite_observation_family_weights', 'finite_observation_scene',
                'training_pairs', 'validation_pairs', 'validate_pair', 'field_request')
    missing = [name for name in required if not callable(getattr(binding, name, None))]
    if missing:
        raise ValueError('Finite observations need adapter capabilities: ' + ', '.join(missing))
    count, seed = spec.get('probe_count', 64), spec.get('probe_seed', 130013)
    limit = spec.get('metadata_cache_bytes', 256 * 1024**2)
    if (type(count) is not int or not 1 <= count <= 64 or type(seed) is not int or seed < 0
            or type(limit) is not int or not 0 < limit <= 256 * 1024**2):
        raise ValueError('Finite observations require 1–64 rows, a nonnegative seed and <=256 MiB metadata.')
    basis = spec.get('basis', 'coordinate')
    if basis not in ('constant', 'coordinate', 'tensor_hats'):
        raise ValueError('Unknown finite observation basis.')
    weights = dict(binding.finite_observation_family_weights())
    if not weights or any(not math.isfinite(float(v)) or float(v) <= 0 for v in weights.values()):
        raise ValueError('Declare positive finite native family importance.')
    frame_fn = getattr(binding, 'finite_observation_frame', None)
    frame = frame_fn() if callable(frame_fn) else None
    frames = {} if frame is None else {frame.identity: frame}
    membership = tuple(map(str, provider.epoch_cases(1, int(provider.seed))))
    training = recipe.get('training', recipe.get('execution', {}))
    microbatch = int(training.get('microbatch_cases', recipe.get('microbatch_size', 2)))
    effective = int(training.get('effective_batch_size', recipe.get('effective_batch_size', microbatch)))
    pairs = tuple(tuple(map(str, pair)) for pair in
                  (spec.get('pairs') or recipe.get('observation_response', {}).get('pairs') or binding.training_pairs()))
    order = _fixed_pair_order(membership, pairs, microbatch, effective)
    validation_ids = tuple(map(str, getattr(provider, 'validation_case_ids', ())))
    if not validation_ids:
        validation_ids = tuple(dict.fromkeys(case for pair in binding.validation_pairs() for case in pair))
    validation_pairs = tuple(tuple(map(str, pair)) for pair in binding.validation_pairs())
    all_ids = tuple(dict.fromkeys((*membership, *validation_ids)))
    scenes, metadata_bytes = {}, 0
    for case_id in all_ids:
        selected = binding.finite_observation_scene(case_id, count=count, seed=seed)
        # Several supported observations can read the same joint-field cell.
        # Share those immutable dependency objects, retaining every source row
        # and every sparse coefficient at its original precision.
        dependencies = {}
        families = []
        for family in selected.families:
            canonical = tuple(dependencies.setdefault(item, item) for item in family.extraction.dependencies)
            families.append(replace(family, extraction=replace(family.extraction, dependencies=canonical)))
        selected = replace(selected, families=tuple(families))
        if str(selected.scene_id) != case_id:
            raise ValueError('Selected observations changed scene identity.')
        metadata_bytes += _metadata_size(selected)
        if metadata_bytes > limit:
            raise MemoryError(f'Finite-observation metadata uses {metadata_bytes} bytes, exceeding {limit}.')
        scenes[case_id] = selected
    pair_records = []
    def eligible(source, allowed):
        result = []
        for pair in source:
            binding.validate_pair(pair)
            if len(pair) != 2 or not set(pair) <= set(allowed):
                raise ValueError('Finite pairs must preserve the declared partition and membership.')
            actual = compatible_scene_families(scenes[pair[0]], scenes[pair[1]], configured_weights=weights)
            pair_records.append({'pair': list(pair), 'eligible_families': list(actual.eligible_families),
                                 'missing_families': list(actual.missing_families)})
            if actual.eligible_families:
                result.append(pair)
        return tuple(result)
    eligible_pairs = eligible(pairs, membership)
    eligible_validation = eligible(validation_pairs, validation_ids)
    if variant == 'contrasts' and not eligible_pairs:
        raise ValueError('No compatible finite-observation TRAIN pair is available.')
    calibration_ids = tuple(map(str, spec.get('calibration_case_ids', binding.calibration_case_ids())))
    calibration_pairs = tuple(pair for pair in eligible_pairs if set(pair) <= set(calibration_ids))
    if (not calibration_ids or len(set(calibration_ids)) != len(calibration_ids)
            or not set(calibration_ids) <= set(membership) or not calibration_pairs
            or set(calibration_ids) != {case for pair in calibration_pairs for case in pair}):
        raise ValueError('Finite calibration requires a fixed, pair-complete TRAIN panel.')
    old_identity = provider.identity_payload
    old_state = getattr(provider, 'training_state_dict', lambda: {})
    old_load = getattr(provider, 'load_training_state_dict', lambda state: None)
    old_begin = getattr(provider, 'begin_native_update', None)
    old_reuse = getattr(provider, 'native_microbatch_terms', None)
    old_names = tuple(getattr(provider, 'native_reuse_loss_names', ()))
    if old_names:
        raise ValueError('Finite observation comparisons require a native-only underlying objective.')
    old_columns = provider.training_history_columns
    old_definitions = getattr(provider, 'training_column_definitions', lambda: {})
    old_metadata = provider.loss_metadata
    state = {'schema': 'finite_observation_training_state_v1', 'calibration': copy.deepcopy(spec.get('calibration')),
             'case_presentations': 0, 'pair_presentations': 0, 'full_draws': 0, 'restricted_draws': 0,
             'last_epoch': 0, 'last_update': None, 'last_reader': None}
    provider.finite_observation_reader_mode = None
    provider.observation_binding = binding
    provider.native_reuse_loss_names = (() if variant == 'disabled' else
                                       _LOSSES[:1] if variant == 'anchors' else _LOSSES)
    provider.auxiliary_loss_names = ()
    provider.auxiliary_update_terms = None
    provider.training_output_policy = 'csv_v3'
    reader_seed = int(recipe.get('observation_response', {}).get('reader_seed', spec.get('reader_seed', 170011)))

    def identity_payload(self):
        return {**old_identity(), 'finite_observations': {
            'schema': 'finite_observation_objective_v1', 'variant': variant,
            'probe_count': count, 'probe_seed': seed, 'basis': basis,
            'family_weights': weights, 'frame': None if frame is None else asdict(frame),
            'case_order': list(order), 'eligible_pairs': [list(pair) for pair in eligible_pairs],
            'validation_case_ids': list(validation_ids),
            'eligible_validation_pairs': [list(pair) for pair in eligible_validation],
            'calibration_case_ids': list(calibration_ids), 'reader_seed': reader_seed,
            'reader_policy': 'sampled_equal_probability', 'reader_weights': {'full': .5, 'restricted': .5},
            'ramp': 'min(epoch/50,1)', 'pair_normalization': 'equal mean over fixed eligible pairs',
            'missing_families': 'omit without renormalizing configured family weights',
            'unsupported': getattr(binding, 'finite_observation_unsupported', lambda: {})()}}

    def training_state_dict(self):
        return {'native': old_state(), 'finite_observations': copy.deepcopy(state)}

    def load_training_state_dict(self, saved):
        if not isinstance(saved, Mapping) or saved.get('finite_observations', {}).get('schema') != state['schema']:
            raise ValueError('Missing finite-observation recovery state.')
        old_load(saved['native'])
        state.clear(); state.update(copy.deepcopy(saved['finite_observations']))
        self.finite_observation_reader_mode = None

    def load_native_epoch_zero_state(self, saved):
        old_load(saved)

    def native_epoch_zero_state(self):
        return old_state()

    def reorder_epoch_cases(self, ordered_cases, *, epoch, seed):
        del epoch, seed
        if len(ordered_cases) != len(order) or set(map(str, ordered_cases)) != set(order):
            raise ValueError('Finite observation ordering changed native TRAIN membership.')
        return order

    def begin_native_update(self, *, epoch, update_index, update_count):
        work = dict(old_begin(epoch=epoch, update_index=update_index, update_count=update_count)) if callable(old_begin) else {}
        if callable(old_begin):
            mode = getattr(self, 'coupled_native_reader_mode', None)
            if mode not in ('full', 'restricted'):
                raise ValueError('The finite comparison requires one sampled reader per effective batch.')
        else:
            key = SamplingKey(reader_seed, int(epoch), 0, 0, 'joint', 'reader',
                              sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=self.sampling_dataset_id)
            mode = 'full' if key.numpy_rng('finite_observation_reader', int(update_index)).random() < .5 else 'restricted'
        self.finite_observation_reader_mode = mode
        state[mode + '_draws'] += 1
        state.update(last_epoch=int(epoch), last_update=int(update_index), last_reader=mode)
        work['finite_observations_reader_' + mode + '_batches'] = 1
        return work

    def finite_observation_components(self, current_model, case_ids, *, read_mode=None,
                                      native_batch=None, native_predictions=None):
        started = time.perf_counter()
        chosen = tuple(scenes[str(case_id)] for case_id in case_ids)
        planned = time.perf_counter()
        extracted, work = execute_finite_dependencies(binding, current_model, chosen, read_mode=read_mode,
                                  native_batch=native_batch, native_predictions=native_predictions)
        observations = {str(scene.scene_id): evaluate_observation_scene(scene, extracted[str(scene.scene_id)],
                        configured_weights=weights, frames=frames, basis=basis) for scene in chosen}
        available = [row.anchor.mean() for row in observations.values() if row.anchor is not None]
        anchor = torch.stack(available).sum() if available else None
        contrasts = []
        for pair in (*eligible_pairs, *eligible_validation):
            if set(pair) <= set(observations):
                loss = finite_scene_contrast(observations[pair[0]], observations[pair[1]], configured_weights=weights).loss
                if loss is None:
                    raise RuntimeError('A fixed eligible pair unexpectedly lost its observation support.')
                contrasts.append(loss.mean())
        contrast = torch.stack(contrasts).sum() if contrasts else None
        work.update(scenes=len(chosen), pairs=len(contrasts), planning_seconds=planned-started,
                    prediction_seconds=time.perf_counter()-planned)
        return {'anchor': anchor, 'contrast': contrast}, work, observations

    def native_microbatch_terms(self, batch, predictions, auxiliary_state, *, epoch, update_index, update_count):
        terms, work = old_reuse(batch, predictions, auxiliary_state, epoch=epoch,
                update_index=update_index, update_count=update_count) if callable(old_reuse) else ({}, {})
        terms, work = dict(terms), dict(work)
        if variant == 'disabled':
            return terms, work
        calibration = state['calibration']
        if not isinstance(calibration, Mapping):
            raise ValueError('Finite objectives require their shared epoch-zero calibration.')
        ids = tuple(map(str, batch.case_keys))
        for pair in eligible_pairs:
            if len(set(pair).intersection(ids)) == 1:
                raise ValueError(f'Finite response pair {pair} was split across native microbatches.')
        components, measured, _ = finite_observation_components(self, model, ids,
            read_mode=self.finite_observation_reader_mode, native_batch=batch, native_predictions=predictions)
        ramp = min(float(epoch)/50., 1.)
        if components['anchor'] is not None:
            terms[_LOSSES[0]] = LossTerm(components['anchor'], 1.,
                float(calibration['coefficients']['anchor']) * ramp * update_count / len(membership))
            measured['anchor_raw_sum'] = float(components['anchor'].detach())
        if variant == 'contrasts' and components['contrast'] is not None:
            terms[_LOSSES[1]] = LossTerm(components['contrast'], 1.,
                float(calibration['coefficients']['contrast']) * ramp * update_count / len(eligible_pairs))
            measured['contrast_raw_sum'] = float(components['contrast'].detach())
        state['case_presentations'] += len(ids)
        state['pair_presentations'] += int(measured['pairs'])
        work.update({'finite_observations_' + key: value for key, value in measured.items()})
        return terms, work

    def validation_auxiliary_terms(self, current_model, *, epoch, arm):
        if arm != current_model.architecture:
            raise ValueError('Finite validation requires the current joint predictor.')
        values = {'anchor': [], 'contrast': []}
        paired = {case for pair in eligible_validation for case in pair}
        chunks = [*eligible_validation, *((case,) for case in validation_ids if case not in paired)]
        for chunk in chunks:
            component, _, _ = finite_observation_components(self, current_model, chunk)
            for name in values:
                if component[name] is not None:
                    values[name].append(component[name])
        coefficients = state['calibration']['coefficients']
        ramp = min(float(epoch)/50., 1.)
        result = {}
        if values['anchor']:
            result[_LOSSES[0]] = LossTerm(torch.stack(values['anchor']).sum(), len(validation_ids), coefficients['anchor']*ramp)
        if variant == 'contrasts' and values['contrast']:
            result[_LOSSES[1]] = LossTerm(torch.stack(values['contrast']).sum(), len(eligible_validation), coefficients['contrast']*ramp)
        return result

    def loss_metadata(self):
        result = dict(old_metadata())
        for name in self.native_reuse_loss_names:
            short = name.rsplit('/', 1)[-1]
            result[name] = {'panel': 'physics', 'label': 'finite observation ' + short,
                'unit': 'weighted dimensionless squared error',
                'formula': ('mean scene weighted point residual squared' if short == 'anchor' else
                            'equal-pair mean of weighted scene-summary residual differences squared'),
                'weight': f'epoch-zero coefficient; min(epoch/50,1); fixed {short} normalization; equal-probability reader'}
        return result

    decode_keys = tuple(dict.fromkeys(getattr(getattr(model, 'core', model), '_additional_decode_count_keys', ())))
    preparation_keys = ('cached_scene_projection_rows_M', 'cached_scene_projection_rows_E',
        'member_key_projection_rows_M', 'member_value_projection_rows_M',
        'member_key_projection_rows_E', 'member_value_projection_rows_E',
        'normalized_member_key_rows_M', 'normalized_member_key_rows_E',
        'prepared_e_directory_rows', 'prepared_e_directory_full_entries',
        'prepared_e_directory_joint95_entries', 'prepared_e_directory_sort_entries',
        'prepared_e_directory_compile_pair_rows', 'prepared_e_directory_positive_exchange_entries',
        'prepared_e_directory_total_access_entries', 'query_anchor_locator_rows', 'query_anchor_index_rows',
        *(f'context_{route}_{metric}' for route in ('mm', 'em', 'ee', 'me')
          for metric in ('score_rows', 'positive_value_edges', 'eligible_edges')))
    work_names = (*_COUNTS, *(f'{view}_{key}' for view in
        ('paired', 'paired_full', 'paired_restricted', 'single_full', 'single_restricted') for key in decode_keys),
        *(f'prepare_{key}' for key in preparation_keys))

    def training_history_columns(self):
        return tuple(dict.fromkeys((*old_columns(), *(f'train_losses.{name}' for name in self.native_reuse_loss_names),
                                    *(f'work_counts.finite_observations_{name}' for name in work_names))))

    def training_column_definitions(self):
        result = dict(old_definitions())
        for name in work_names:
            result['work_counts.finite_observations_' + name] = {'dtype': 'float',
                'unit': 's' if name.endswith('_seconds') else 'dimensionless squared error' if name.endswith('_raw_sum') else 'count',
                'meaning': 'Epoch sum of finite-observation ' + name.replace('_', ' ')}
        return result

    def set_finite_observation_calibration(self, record):
        if (not isinstance(record, Mapping) or record.get('schema') != 'finite_observation_calibration_v1'
                or record.get('epoch') != 0 or tuple(record.get('calibration_case_ids', ())) != calibration_ids):
            raise ValueError('Expected this finite objective\'s TRAIN-only epoch-zero calibration.')
        norms = record.get('gradient_diagnostics', {}).get('groups', {}).get('global', {}).get('norms', {})
        for name, target in _TARGETS.items():
            value = float(record.get('coefficients', {}).get(name, 0))
            expected = target * float(norms.get('native', 0)) / float(norms.get(name, 1))
            if not math.isfinite(value) or value <= 0 or not math.isclose(value, expected, rel_tol=1.e-10):
                raise ValueError(f'Finite {name} coefficient does not match the recorded gradient calibration.')
        state['calibration'] = copy.deepcopy(dict(record))

    def calibrate_finite_observations(self, current_model):
        before = copy.deepcopy(self.training_state_dict())
        core = getattr(current_model, 'core', current_model)
        clocks = {name: getattr(core, name) for name in ('_joint_epoch', '_joint_update_index') if hasattr(core, name)}
        previous = self.finite_observation_reader_mode
        coupled_previous = getattr(self, 'coupled_native_reader_mode', None)
        # A distinct namespace never consumes native training query/reader RNG.
        key = _CalibrationKey(int(self.seed), 1, 0, 0, 'joint', current_model.architecture,
                          sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=self.sampling_dataset_id)
        chunks = list(calibration_pairs)
        try:
            with preserved_state(current_model):
                current_model.train(True)
                self._coupled_calibration_panel = True
                self.finite_observation_reader_mode = None
                if hasattr(self, 'coupled_native_reader_mode'):
                    self.coupled_native_reader_mode = None
                batches = [self.make_batch(chunk, key) for chunk in chunks]
                for name, value in clocks.items():
                    setattr(core, name, value)
                denominators = self.loss_denominators(batches, phase='joint', arm=current_model.architecture)
                def native():
                    for batch in batches:
                        prediction, auxiliary = self.predict_native(current_model, self.make_scene(batch.scene_inputs),
                            batch.receivers, current_model.architecture, 'joint', epoch=0, temperature=1.)
                        terms = self.loss_terms(prediction, batch.targets, 'joint', auxiliary)
                        yield sum(term.numerator * float(term.weight) / float(denominators[name])
                                  for name, term in terms.items())
                def auxiliary(name):
                    for chunk in chunks:
                        components, _, _ = finite_observation_components(self, current_model, chunk)
                        value = components[name]
                        if value is not None:
                            yield value / (len(calibration_ids) if name == 'anchor' else len(calibration_pairs))
                from .organization_parameters import partition_organization_parameters
                ownership = partition_organization_parameters(current_model)
                all_names = {name for name, parameter in current_model.named_parameters() if parameter.requires_grad}
                organization = tuple(ownership.organization_names)
                other = tuple(sorted(all_names-set(organization)))
                groups = {'organization': organization, 'content_and_output': other}
                diagnostic = gradient_diagnostics(current_model, {'native': native,
                    'anchor': lambda: auxiliary('anchor'), 'contrast': lambda: auxiliary('contrast')},
                    parameter_groups=groups, require_nonzero=True)
        finally:
            self._coupled_calibration_panel = False
            self.load_training_state_dict(before)
            self.finite_observation_reader_mode = previous
            if hasattr(self, 'coupled_native_reader_mode'):
                self.coupled_native_reader_mode = coupled_previous
            for name, value in clocks.items():
                setattr(core, name, value)
        norms = diagnostic['groups']['global']['norms']
        coefficients = {name: ratio * norms['native']/norms[name] for name, ratio in _TARGETS.items()}
        parameter_count = sum(p.numel() for p in current_model.parameters() if p.requires_grad)
        return {'schema': 'finite_observation_calibration_v1', 'epoch': 0, 'coefficients': coefficients,
                'target_gradient_rms_ratios': dict(_TARGETS), 'gradient_rms_parameter_count':
                    parameter_count,
                'gradient_rms': {name: value/math.sqrt(parameter_count) for name, value in norms.items()},
                'calibration_case_ids': list(calibration_ids), 'calibration_pairs': [list(pair) for pair in calibration_pairs],
                'basis': basis, 'probe_count': count, 'probe_seed': seed, 'gradient_diagnostics': diagnostic,
                'native_calibration_stream': 'independent finite_observation_calibration namespace'}

    def write_finite_observation_definitions(self, output):
        """Save compact immutable source maps once, referenced by every arm."""
        import json
        from .run_layout import RunLayout
        configured = spec.get('definitions_path')
        if not configured and spec.get('epoch_zero_checkpoint'):
            zero = Path(spec['epoch_zero_checkpoint'])
            configured = zero.with_name(zero.stem + '_observations.json')
        shared = Path(configured) if configured else RunLayout(output).write_path('finite_observations.json', category='configs')
        records = []
        for case_id in all_ids:
            families = []
            for family in scenes[case_id].families:
                extraction = family.extraction
                actual_frame = frames.get(family.frame)
                actual_basis = basis if actual_frame is not None else 'constant'
                if basis == 'constant':
                    actual_basis = 'constant'
                out_of_frame = 0
                if actual_frame is not None and actual_basis != 'constant':
                    coordinates = np.asarray(family.coordinates, dtype=np.float64)
                    out_of_frame = int(np.count_nonzero(((coordinates < np.asarray(actual_frame.lower)) |
                                                        (coordinates > np.asarray(actual_frame.upper))).any(axis=1)))
                families.append({'name': family.name, 'component': asdict(family.component),
                    'support': family.support, 'frame': family.frame, 'time': family.time,
                    'extraction_id': family.extraction_id, 'native_weight': family.native_weight,
                    'row_ids': family.row_ids, 'coordinates': family.coordinates, 'measures': family.measures,
                    'actual_basis': actual_basis, 'frame_definition': None if actual_frame is None else asdict(actual_frame),
                    'out_of_frame_count': out_of_frame,
                    'reference': 'existing adapter source arrays indexed by these row_ids; no duplicate field arrays',
                    'extraction': asdict(extraction)})
            records.append({'scene_id': case_id, 'families': families})
        definition = _json_value({'schema': 'finite_observation_source_definitions_v1',
            'probe_count': count, 'probe_seed': seed, 'family_weights': weights,
            'frame': None if frame is None else asdict(frame), 'pair_support': pair_records,
            'training_case_ids': membership, 'validation_case_ids': validation_ids, 'scenes': records})
        if shared.exists():
            if json.loads(shared.read_text()) != definition:
                raise ValueError('Saved finite probe IDs, measures or extraction definitions differ.')
        else:
            _atomic_json(shared, definition, indent=None)
        pointer = {'schema': 'finite_observation_run_definition_v1', 'source_definitions': str(shared.resolve()),
                   'objective': identity_payload(self)['finite_observations'],
                   'calibration': copy.deepcopy(state['calibration']), 'metadata_bytes': metadata_bytes}
        target = RunLayout(output).write_path('finite_observation_objective.json', category='configs')
        if target.exists():
            saved = json.loads(target.read_text())
            incoming = _json_value(pointer)
            # Python object sharing and allocator overhead vary between fresh
            # processes. The creation-time RAM measurement is evidence, not a
            # definition of the finite observations or their objective.
            saved.pop('metadata_bytes', None)
            incoming.pop('metadata_bytes', None)
            if saved != incoming:
                added, missing = sorted(set(incoming)-set(saved)), sorted(set(saved)-set(incoming))
                changed = sorted(key for key in set(saved)&set(incoming) if saved[key] != incoming[key])
                raise ValueError(f'Saved finite objective definitions differ: added={added}, '
                                 f'missing={missing}, changed={changed}.')
        else:
            _atomic_json(target, _json_value(pointer))
        return pointer

    for method in (identity_payload, training_state_dict, load_training_state_dict,
                   load_native_epoch_zero_state, native_epoch_zero_state, reorder_epoch_cases,
                   begin_native_update, finite_observation_components, native_microbatch_terms,
                   loss_metadata, training_history_columns, training_column_definitions,
                   set_finite_observation_calibration, calibrate_finite_observations,
                   write_finite_observation_definitions):
        setattr(provider, method.__name__, MethodType(method, provider))
    if variant != 'disabled':
        provider.validation_auxiliary_terms = MethodType(validation_auxiliary_terms, provider)
    provider.finite_observation_metadata_bytes = metadata_bytes
    provider.finite_observation_scenes = scenes
    provider.finite_observation_pairs = eligible_pairs
    provider.finite_observation_spec = spec
    return model, provider
