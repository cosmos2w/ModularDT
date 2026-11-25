"""Attach native-panel response reuse to an existing observation capability.

The binding supplies native population metadata and extracted native values;
this module owns pair ordering, independent streams, reader weighting and the
response loss. No second scene preparation or decoder call is made here.
"""
from __future__ import annotations

import copy
import math
import sys
import time
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from numbers import Real
from types import MethodType

import numpy as np
import torch

from .coupled_observations import prepare_native_coupling, sample_native_coupling, coupled_response_loss
from .gradient_diagnostics import gradient_diagnostics, preserved_state
from .unified_training import LossTerm, SamplingKey


def _metadata_size(value, seen=None):
    """Conservative recursive RAM charge for one disposable coupling plan."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return max(size, int(value.nbytes) + 128)
    if is_dataclass(value) and not isinstance(value, type):
        namespace = getattr(value, '__dict__', None)
        return (size + (sys.getsizeof(namespace) if namespace is not None else 0)
                + sum(_metadata_size(getattr(value, field.name), seen) for field in fields(value)))
    if isinstance(value, Mapping):
        return size + sum(_metadata_size(k, seen) + _metadata_size(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(_metadata_size(item, seen) for item in value)
    return size


def _fixed_pair_order(membership, pairs, microbatch, effective):
    """Pack disjoint pairs within the same actual microbatch, visiting all cases once."""
    membership = tuple(map(str, membership))
    if (len(set(membership)) != len(membership) or microbatch < 2 or effective < microbatch):
        raise ValueError('Coupled training needs unique cases and a microbatch capacity of at least two.')
    position = {case: index for index, case in enumerate(membership)}
    used = set()
    ordered_pairs = []
    for pair in pairs:
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(membership) or used.intersection(pair):
            raise ValueError('Coupled TRAIN pairs must be disjoint members of the native TRAIN cohort.')
        used.update(pair)
        ordered_pairs.append(tuple(sorted(pair, key=position.__getitem__)))
    ordered_pairs.sort(key=lambda pair: min(position[case] for case in pair))
    singles = [case for case in membership if case not in used]
    result = []
    pair_index = single_index = 0
    while pair_index < len(ordered_pairs) or single_index < len(singles):
        within_update = len(result) % effective
        capacity = min(microbatch, effective - within_update)
        remaining = len(membership) - len(result)
        capacity = min(capacity, remaining)
        chunk = []
        while capacity - len(chunk) >= 2 and pair_index < len(ordered_pairs):
            chunk.extend(ordered_pairs[pair_index]); pair_index += 1
        while len(chunk) < capacity and single_index < len(singles):
            chunk.append(singles[single_index]); single_index += 1
        if len(chunk) != capacity:
            raise ValueError('The fixed effective/microbatch capacities cannot place every pair without splitting it.')
        result.extend(chunk)
    return tuple(result)


def attach_coupled_native_task(recipe, model, provider, binding):
    spec = dict(recipe.get('observation_response', {}))
    variant = spec.get('variant')
    if variant not in ('native_control', 'observed_response', 'sampled_reader_response'):
        raise ValueError('Choose native_control, observed_response or sampled_reader_response.')
    response_weight = spec.get('response_weight', 1.0)
    if isinstance(response_weight, bool) or not isinstance(response_weight, Real):
        raise ValueError('observation_response.response_weight must be a finite positive number.')
    try:
        response_weight = float(response_weight)
    except OverflowError:
        raise ValueError('observation_response.response_weight must be a finite positive number.') from None
    if not math.isfinite(response_weight) or response_weight <= 0:
        raise ValueError('observation_response.response_weight must be a finite positive number.')
    capability = getattr(binding, 'coupled_native_capability', None)
    if callable(capability):
        report = capability()
        if not isinstance(report, Mapping):
            raise TypeError('Native coupling capability must be a mapping.')
        if report.get('available') is not True:
            raise ValueError('Native coupling is unavailable for this binding: ' + str(report.get('reason')))
    required = ('coupled_native_population', 'coupled_native_budget', 'coupled_native_extraction',
                'coupled_view_loss_names', 'coupled_native_sampling_law')
    missing = [name for name in required if not callable(getattr(binding, name, None))]
    if missing:
        raise ValueError('Native coupling is unavailable for this binding: missing ' + ', '.join(missing))
    pairs = tuple(tuple(map(str, pair)) for pair in (spec.get('pairs') or binding.training_pairs()))
    membership = tuple(map(str, provider.epoch_cases(1, int(provider.seed))))
    training = recipe.get('training', recipe.get('execution', {}))
    microbatch = int(training.get('microbatch_cases', recipe.get('microbatch_size', 2)))
    effective = int(training.get('effective_batch_size', recipe.get('effective_batch_size', microbatch)))
    order = _fixed_pair_order(membership, pairs, microbatch, effective)
    anchors = int(spec.get('anchor_count', 128))
    pair_seed, reader_seed = int(spec.get('pair_seed', 0)), int(spec.get('reader_seed', 0))
    probability = float(spec.get('full_reader_probability', .5))
    if not 1 <= anchors <= 128 or pair_seed < 0 or reader_seed < 0 or not 0 < probability < 1:
        raise ValueError('Coupled anchors/seeds and reader probabilities are invalid.')
    if getattr(provider, 'sampling_version', SamplingKey.CASE_EPOCH_VERSION) != SamplingKey.CASE_EPOCH_VERSION:
        raise ValueError('Coupled panels require independent case_epoch_v1 query namespaces.')
    for pair in pairs:
        binding.validate_pair(pair)
    validation_pairs = tuple(tuple(map(str, pair)) for pair in
                             (spec.get('validation_pairs') or binding.validation_pairs()))
    if not validation_pairs or not callable(getattr(binding, 'coupled_validation_batch', None)):
        raise ValueError('Coupled training needs declared native validation pairs and a validation binding.')
    for pair in validation_pairs:
        binding.validate_pair(pair)
    cache_limit = int(spec.get('metadata_cache_bytes', 1792 * 1024 ** 2))
    if not 0 <= cache_limit <= 2 * 1024 ** 3:
        raise ValueError('Native coupling metadata cache must be between zero and 2 GiB per process.')
    cache = OrderedDict()
    unavailable = {}
    live_plans = {}
    cache_usage = {'bytes': 0, 'peak_bytes': 0, 'live_bytes': 0, 'peak_live_bytes': 0,
                   'largest_plan_bytes': 0, 'builds': 0, 'build_seconds': 0.}
    def refresh_live():
        for key in tuple(live_plans):
            if live_plans[key][0]() is None:
                del live_plans[key]
        cache_usage['live_bytes'] = sum(size for _, size in live_plans.values())
        cache_usage['peak_live_bytes'] = max(cache_usage['peak_live_bytes'], cache_usage['live_bytes'])
    def get_plan(pair):
        if pair in cache:
            plan, size = cache.pop(pair); cache[pair] = (plan, size)
            return plan
        # Evicted plans can still be owned by prepared microbatches. Leave
        # enough room for a new plan based on measured prior plans, with a
        # conservative first-plan allowance, before allocating it.
        refresh_live()
        reserve = max(64 * 1024 ** 2, cache_usage['largest_plan_bytes'])
        while cache and cache_usage['live_bytes'] + reserve > 2 * 1024 ** 3:
            removed_entry = cache.popitem(last=False)[1]
            cache_usage['bytes'] -= removed_entry[1]
            del removed_entry
            refresh_live()
        if cache_usage['live_bytes'] + reserve > 2 * 1024 ** 3:
            raise MemoryError('Active native coupling plans leave insufficient metadata headroom.')
        left_law, right_law = (binding.coupled_native_sampling_law(case) for case in pair)
        if left_law != right_law:
            unavailable[pair] = 'Scene sampling laws differ.'
            return None
        if left_law != 'uniform_without_replacement':
            unavailable[pair] = f'Unsupported native sampling law: {left_law}'
            return None
        started = time.perf_counter()
        plan = prepare_native_coupling(binding.coupled_native_population(pair[0]),
                                       binding.coupled_native_population(pair[1]), sampling_law=left_law)
        cache_usage['builds'] += 1
        if not plan.available:
            unavailable[pair] = plan.reason
            return None
        # Account for allocator overhead and shared population references too.
        size = math.ceil(1.30 * _metadata_size(plan))
        cache_usage['largest_plan_bytes'] = max(cache_usage['largest_plan_bytes'], size)
        live_plans[id(plan)] = (weakref.ref(plan), size)
        # Keep the admitted fixed-order prefix when the full inventory cannot
        # fit. Replacing it on every sequential pass would evict each next plan
        # before use and make an otherwise useful bounded cache thrash.
        if cache_usage['bytes'] + size <= cache_limit:
            cache[pair] = (plan, size)
            cache_usage['bytes'] += size
            cache_usage['peak_bytes'] = max(cache_usage['peak_bytes'], cache_usage['bytes'])
        refresh_live()
        while cache and cache_usage['live_bytes'] > 2 * 1024 ** 3:
            removed_entry = cache.popitem(last=False)[1]
            cache_usage['bytes'] -= removed_entry[1]
            del removed_entry
            refresh_live()
        if cache_usage['live_bytes'] > 2 * 1024 ** 3:
            raise MemoryError('Live coupling plans exceed the 2 GiB metadata budget, including active batches.')
        cache_usage['build_seconds'] += time.perf_counter() - started
        return plan
    def has_target(plan):
        return plan is not None and any(mass > 0 and component.weight > 0
                   for mass, component in zip(plan.eligible_mass, plan.left.components))
    eligible_pairs = tuple(pair for pair in pairs if has_target(get_plan(pair)))
    if not eligible_pairs:
        raise ValueError('No scheduled pair has a supported physical observation component.')
    eligible_validation_pairs = tuple(pair for pair in validation_pairs if has_target(get_plan(pair)))
    unavailable_inventory = tuple((pair, reason) for pair, reason in unavailable.items())
    positions = {case: i for i, case in enumerate(order)}
    pair_set = {case: pair for pair in pairs for case in pair}
    for pair in pairs:
        first, second = sorted(positions[case] for case in pair)
        if (first // effective != second // effective or
                (first % effective) // microbatch != (second % effective) // microbatch):
            raise RuntimeError('Pair-aware order split a pair across actual native microbatches.')
    view_names = tuple(binding.coupled_view_loss_names())
    if not view_names or len(set(view_names)) != len(view_names):
        raise ValueError('The binding must explicitly identify the view-dependent native loss terms.')
    old_identity = provider.identity_payload
    old_state = getattr(provider, 'training_state_dict', lambda: {})
    old_load = getattr(provider, 'load_training_state_dict', lambda state: None)
    old_terms, old_metadata = provider.loss_terms, provider.loss_metadata
    old_validation_terms = provider.validation_loss_terms
    old_columns = provider.training_history_columns
    old_definitions = getattr(provider, 'training_column_definitions', lambda: {})
    old_validation = getattr(provider, 'validation_objective_predictions', None)
    state = {'schema': 'coupled_native_training_state_v1', 'calibration': copy.deepcopy(spec.get('calibration')),
             'full_draws': 0, 'restricted_draws': 0, 'dual_draws': 0, 'pair_presentations': 0,
             'last_reader': None, 'last_epoch': 0, 'last_update': None}
    cache_report = {'builds': cache_usage['builds'], 'seconds': cache_usage['build_seconds']}
    provider.coupled_native_reader_mode = None
    provider.native_reuse_loss_names = (() if variant == 'native_control' else ('coupled_native/response',))
    provider.auxiliary_loss_names = ()
    provider.auxiliary_update_terms = None
    provider.validation_auxiliary_terms = None
    provider.training_output_policy = 'csv_v3'

    def identity_payload(self):
        result = dict(old_identity())
        # Some bindings retain their source recipe as part of the native
        # identity. Treat an explicit default exactly like an omitted default.
        source_recipe = result.get('recipe')
        if response_weight == 1.0 and isinstance(source_recipe, dict):
            source_observation = source_recipe.get('observation_response')
            if (isinstance(source_observation, dict)
                    and source_observation.get('response_weight') == 1.0):
                source_recipe = copy.deepcopy(source_recipe)
                source_recipe['observation_response'].pop('response_weight')
                result['recipe'] = source_recipe
        result['observation_response'] = {
            'schema': 'coupled_native_observation_response_v1', 'mode': 'coupled_native', 'variant': variant,
            'pairs': [list(p) for p in pairs], 'eligible_pairs': [list(p) for p in eligible_pairs],
            'unavailable_pairs': [{'pair': list(pair), 'reason': reason} for pair, reason in unavailable_inventory],
            'case_order': list(order), 'anchor_count': anchors, 'pair_seed': pair_seed,
            'reader_seed': reader_seed, 'full_reader_probability': probability,
            'validation_pairs': [list(p) for p in validation_pairs],
            'eligible_validation_pairs': [list(p) for p in eligible_validation_pairs],
            'sampling': 'uniform native row panel without replacement; coupled iid anchors and independent completion/order',
            'binding': binding.identity_payload(), 'reader_weights': {'full': .5, 'restricted': .5},
            'calibration_epoch': 0, 'response_ramp': 'min(epoch/50,1)',
            'validation_objective_complete': variant == 'native_control' or bool(eligible_validation_pairs),
        }
        if response_weight != 1.0:
            result['observation_response']['response_weight'] = response_weight
        return result

    def training_state_dict(self):
        return {'native': old_state(), 'coupled_native': copy.deepcopy(state)}

    def load_training_state_dict(self, saved):
        if not isinstance(saved, Mapping) or not isinstance(saved.get('coupled_native'), Mapping):
            raise ValueError('Missing coupled native training state.')
        old_load(saved['native'])
        incoming = copy.deepcopy(saved['coupled_native'])
        if incoming.get('schema') != state['schema']:
            raise ValueError('Invalid coupled native training state schema.')
        state.clear(); state.update(incoming)
        self.coupled_native_reader_mode = None
        cache_report.update(builds=cache_usage['builds'], seconds=cache_usage['build_seconds'])

    def reorder_epoch_cases(self, ordered_cases, *, epoch, seed):
        del epoch, seed
        if set(map(str, ordered_cases)) != set(order) or len(ordered_cases) != len(order):
            raise ValueError('Coupled ordering must preserve the original TRAIN membership.')
        return order

    def sample_coupled_native_rows(self, case_keys, key):
        keys = tuple(map(str, case_keys)); present = set(keys)
        rows, carriers = {}, []
        for pair in pairs:
            overlap = present.intersection(pair)
            if len(overlap) == 1 and not getattr(self, '_coupled_calibration_panel', False):
                raise ValueError(f'Native pair {pair} was split between microbatches.')
            if not overlap:
                continue
            plan = get_plan(pair)
            if plan is None:
                continue
            prefix = ('coupled_native', pair_seed, pair)
            rng = lambda stream: key.numpy_rng(*prefix, stream)
            sample = sample_native_coupling(plan, binding.coupled_native_budget(pair[0]),
                binding.coupled_native_budget(pair[1]), anchor_count=anchors,
                anchor_rng=rng('pair_anchors'), left_completion_rng=rng('left_completion'),
                right_completion_rng=rng('right_completion'), left_order_rng=rng('left_order'),
                right_order_rng=rng('right_order'))
            if pair[0] in overlap:
                rows[pair[0]] = sample.left_rows
            if pair[1] in overlap:
                rows[pair[1]] = sample.right_rows
            if len(overlap) == 2:
                carriers.append((pair, sample))
        return rows, tuple(carriers)

    def begin_native_update(self, *, epoch, update_index, update_count):
        del update_count
        if variant == 'sampled_reader_response':
            key = SamplingKey(reader_seed, int(epoch), 0, 0, 'joint', 'reader',
                              sampling_version=SamplingKey.CASE_EPOCH_VERSION,
                              dataset_id=self.sampling_dataset_id)
            mode = 'full' if key.numpy_rng('coupled_reader', int(update_index)).random() < probability else 'restricted'
            state[mode + '_draws'] += 1
        else:
            mode = None
            state['dual_draws'] += 1
        state.update(last_reader=mode or 'dual', last_epoch=int(epoch), last_update=int(update_index))
        self.coupled_native_reader_mode = mode
        previous_builds = cache_report['builds']
        previous_seconds = cache_report['seconds']
        cache_report.update(builds=cache_usage['builds'], seconds=cache_usage['build_seconds'])
        return {'coupled_reader_' + (mode or 'dual') + '_batches': 1,
                'coupled_plan_builds': cache_usage['builds'] - previous_builds,
                'coupled_plan_build_seconds': cache_usage['build_seconds'] - previous_seconds}

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        terms = dict(old_terms(predictions, targets, phase, auxiliary_state))
        mode = self.coupled_native_reader_mode
        if mode is not None:
            factor = .5 / (probability if mode == 'full' else 1. - probability)
            missing = set(view_names) - set(terms)
            if missing:
                raise ValueError(f'Sampled native reader omitted declared view-dependent losses: {sorted(missing)}')
            terms = {name: LossTerm(term.numerator, term.denominator,
                                    float(term.weight) * (factor if name in view_names else 1.))
                     for name, term in terms.items()}
        return terms

    def estimates(batch, predictions):
        result = []
        for pair, sample in (batch.auxiliary or {}).get('coupled_native', ()):
            left = batch.case_keys.index(pair[0]); right = batch.case_keys.index(pair[1])
            left_values, left_dependencies = binding.coupled_native_extraction(predictions, left, batch)
            right_values, right_dependencies = binding.coupled_native_extraction(predictions, right, batch)
            estimate = coupled_response_loss(sample.plan, sample, left_values, right_values,
                left_dependencies=left_dependencies, right_dependencies=right_dependencies)
            result.append((pair, estimate))
        return result

    def native_microbatch_terms(self, batch, predictions, auxiliary_state, *, epoch, update_index, update_count):
        del auxiliary_state, update_index
        work = {'coupled_pair_presentations': 0., 'coupled_common_draws': 0.,
                'coupled_common_probability_sum': 0., 'coupled_eligible_mass_sum': 0.,
                'coupled_inverse_weight_sum': 0., 'coupled_inverse_weight_count': 0.}
        values = []
        if variant == 'native_control':
            with torch.no_grad():
                observed = estimates(batch, predictions)
        else:
            observed = estimates(batch, predictions)
        for pair, estimate in observed:
            work['coupled_pair_presentations'] += 1
            work['coupled_common_draws'] += estimate.realized_common_draws
            work['coupled_common_probability_sum'] += estimate.common_probability
            work['coupled_eligible_mass_sum'] += sum(estimate.eligible_mass.values())
            for weights in estimate.inverse_probability_weights.values():
                work['coupled_inverse_weight_sum'] += float(weights.sum())
                work['coupled_inverse_weight_count'] += int(np.count_nonzero(weights))
            if pair in eligible_pairs and estimate.loss is not None:
                values.append(estimate.loss)
        state['pair_presentations'] += int(work['coupled_pair_presentations'])
        if variant == 'native_control' or not values:
            return {}, work
        calibration = state['calibration']
        if not isinstance(calibration, Mapping):
            raise ValueError('Coupled response requires the shared epoch-zero calibration.')
        coefficient = float(calibration['coefficient'])
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError('Coupled response calibration coefficient must be positive and finite.')
        mode = self.coupled_native_reader_mode
        reader_weight = (1. if mode is None else .5 / (probability if mode == 'full' else 1. - probability))
        weight = coefficient * min(float(epoch) / 50., 1.) * update_count / len(eligible_pairs) * reader_weight
        if response_weight != 1.0:
            weight *= response_weight
        return {'coupled_native/response': LossTerm(torch.stack(values).sum(), 1., weight)}, work

    def validation_objective_predictions(self, current_model, batch, *, arm, epoch):
        previous = self.coupled_native_reader_mode
        self.coupled_native_reader_mode = None
        try:
            if callable(old_validation):
                return old_validation(current_model, batch, arm=arm, epoch=epoch)
            mode = current_model.training
            try:
                current_model.train(True)
                return self.predict_native(current_model, self.make_scene(batch.scene_inputs), batch.receivers,
                                           arm, 'joint', epoch=epoch, temperature=1.)
            finally:
                current_model.train(mode)
        finally:
            self.coupled_native_reader_mode = previous

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch, arm):
        previous = self.coupled_native_reader_mode
        self.coupled_native_reader_mode = None
        try:
            return old_validation_terms(predictions, targets, auxiliary_state, batch=batch, arm=arm)
        finally:
            self.coupled_native_reader_mode = previous

    def validation_reader_records(self, predictions, batch, auxiliary):
        if hasattr(predictions, 'full') and hasattr(predictions, 'restricted'):
            return {name: self.validation_metrics(getattr(predictions, name), batch.targets, auxiliary)
                    for name in ('full', 'restricted')}
        # The fixture carries the view axis on each scene tensor.
        if isinstance(predictions, tuple) and predictions and torch.is_tensor(predictions[0]):
            if any(item.shape[0] != 2 for item in predictions):
                raise ValueError('Dual-reader validation needs two executed reader views.')
            return {name: self.validation_metrics(tuple(item[index:index+1] for item in predictions),
                                                 batch.targets, auxiliary)
                    for index, name in enumerate(('full', 'restricted'))}
        raise TypeError('Binding does not expose both validation reader predictions.')

    def validation_auxiliary_terms(self, current_model, *, epoch, arm):
        if arm != current_model.architecture:
            raise ValueError('Coupled validation must use the current native model.')
        calibration = state['calibration']
        if not isinstance(calibration, Mapping):
            raise ValueError('Coupled validation requires the shared epoch-zero calibration.')
        key = SamplingKey(pair_seed, 1, 0, 0, 'validation', arm,
                          sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=self.sampling_dataset_id)
        previous, previous_mode = self.coupled_native_reader_mode, current_model.training
        values = []
        try:
            self.coupled_native_reader_mode = None
            current_model.train(True)
            for pair in eligible_validation_pairs:
                plan = get_plan(pair)
                if plan is None:
                    continue
                prefix = ('coupled_validation', pair_seed, pair)
                rng = lambda stream: key.numpy_rng(*prefix, stream)
                sample = sample_native_coupling(plan, binding.coupled_native_budget(pair[0]),
                    binding.coupled_native_budget(pair[1]), anchor_count=anchors,
                    anchor_rng=rng('pair_anchors'), left_completion_rng=rng('left_completion'),
                    right_completion_rng=rng('right_completion'), left_order_rng=rng('left_order'),
                    right_order_rng=rng('right_order'))
                batch = binding.coupled_validation_batch(pair, sample)
                prediction, _ = self.predict_native(current_model, self.make_scene(batch.scene_inputs), batch.receivers,
                                                    arm, 'joint', epoch=epoch, temperature=1.)
                values.extend(estimate.loss for _, estimate in estimates(batch, prediction) if estimate.loss is not None)
        finally:
            current_model.train(previous_mode)
            self.coupled_native_reader_mode = previous
        if not values:
            return {}
        weight = float(calibration['coefficient']) * min(float(epoch) / 50., 1.)
        if response_weight != 1.0:
            weight *= response_weight
        return {'coupled_native/response': LossTerm(torch.stack(values).sum(), len(values), weight)}

    def loss_metadata(self):
        result = {name: value for name, value in old_metadata().items()
                  if not name.startswith(('observation_response/', 'native_pair/', 'field_response/'))}
        for name in view_names:
            if name in result:
                result[name] = dict(result[name])
                result[name]['formula'] = (str(result[name].get('formula', 'Native objective.')) +
                    (' TRAIN executes one probability-weighted sampled reader per effective batch; '
                     'validation executes and averages both readers.' if variant == 'sampled_reader_response'
                     else ' TRAIN and validation average full and restricted readers.'))
        if variant != 'native_control':
            result['coupled_native/response'] = {'panel': 'physics', 'label': 'coupled native observed response',
                'unit': 'weighted dimensionless squared error',
                'formula': 'fixed-measure inverse-probability mean of paired native residual differences',
                'weight': 'epoch-zero TRAIN gradient calibration, epochs 1–50 ramp, equal-pair and reader weights'}
            if response_weight != 1.0:
                result['coupled_native/response']['weight'] += (
                    f'; post-calibration response multiplier {response_weight!r}')
                result['coupled_native/response']['meaning'] = (
                    'Weighted coupled native response contribution, including the calibrated coefficient, '
                    f'epoch ramp, equal-pair and reader weights, and response_weight={response_weight!r}.')
        return result

    decode_keys = tuple(dict.fromkeys(getattr(getattr(model, 'core', model), '_additional_decode_count_keys', ())))
    execution_work_names = tuple(f'{prefix}_{key}' for prefix in
        ('paired', 'paired_full', 'paired_restricted', 'single_full', 'single_restricted') for key in decode_keys)
    preparation_names = ('cached_scene_projection_rows_M', 'cached_scene_projection_rows_E',
        'member_key_projection_rows_M', 'member_value_projection_rows_M',
        'member_key_projection_rows_E', 'member_value_projection_rows_E',
        'normalized_member_key_rows_M', 'normalized_member_key_rows_E',
        'prepared_e_directory_rows', 'prepared_e_directory_full_entries',
        'prepared_e_directory_joint95_entries', 'prepared_e_directory_sort_entries',
        'prepared_e_directory_compile_pair_rows', 'prepared_e_directory_positive_exchange_entries',
        'prepared_e_directory_total_access_entries', 'query_anchor_locator_rows', 'query_anchor_index_rows',
        *(f'context_{route}_{metric}' for route in ('mm', 'em', 'ee', 'me')
          for metric in ('score_rows', 'positive_value_edges', 'eligible_edges')))
    work_names = ('coupled_reader_full_batches', 'coupled_reader_restricted_batches', 'coupled_reader_dual_batches',
                  'coupled_pair_presentations', 'coupled_common_draws', 'coupled_common_probability_sum',
                  'coupled_eligible_mass_sum', 'coupled_inverse_weight_sum', 'coupled_inverse_weight_count',
                  'coupled_plan_builds', 'coupled_plan_build_seconds', 'native_fluid_queries',
                  'native_union_rows', 'native_local_pairs', *execution_work_names,
                  *(f'core_prepare_primary_{name}' for name in preparation_names))
    def training_history_columns(self):
        native = tuple(c for c in old_columns() if not c.startswith(('train_losses.observation_response/',
            'work_counts.observation_response_', 'train_losses.native_pair/', 'work_counts.native_pair_',
            'train_losses.field_response/', 'work_counts.field_response_')))
        return tuple(dict.fromkeys((*native, *(f'train_losses.{name}' for name in self.native_reuse_loss_names),
                                   *(f'work_counts.{name}' for name in work_names))))

    def training_column_definitions(self):
        columns = self.training_history_columns()
        result = {name: definition for name, definition in old_definitions().items() if name in columns}
        for name in work_names:
            result['work_counts.' + name] = {'dtype': 'float', 'unit': ('s' if name.endswith('_seconds') else
                'count' if not name.endswith('_sum') else '1'),
                'meaning': 'Epoch sum of ' + name.replace('_', ' ')}
        return result

    def set_coupled_calibration(self, record):
        if (not isinstance(record, Mapping) or record.get('schema') != 'coupled_native_calibration_v1'
                or record.get('epoch') != 0 or not math.isfinite(float(record.get('coefficient', 0)))
                or float(record['coefficient']) <= 0):
            raise ValueError('Expected a finite positive epoch-zero coupled calibration.')
        state['calibration'] = copy.deepcopy(dict(record))

    def calibrate_coupled_native(self, current_model):
        """Measure both reader gradients on the unchanged common epoch-zero state."""
        ids = tuple(map(str, binding.calibration_case_ids()))
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(membership):
            raise ValueError('Calibration IDs must be unique members of the native TRAIN cohort.')
        selected = tuple(pair for pair in eligible_pairs if set(pair) <= set(ids))
        if not selected:
            raise ValueError('The selected TRAIN calibration panel has no supported complete pair.')
        paired_ids = {case for pair in selected for case in pair}
        chunks = [*selected, *((case,) for case in ids if case not in paired_ids)]
        before = copy.deepcopy(self.training_state_dict())
        previous_mode = self.coupled_native_reader_mode
        core = getattr(current_model, 'core', current_model)
        clocks = {name: getattr(core, name) for name in ('_joint_epoch', '_joint_update_index') if hasattr(core, name)}
        key = SamplingKey(int(self.seed), 1, 0, 0, 'joint', current_model.architecture,
                          sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=self.sampling_dataset_id)
        try:
            with preserved_state(current_model):
                self._coupled_calibration_panel = True
                self.coupled_native_reader_mode = None
                current_model.train(True)
                batches = [self.make_batch(chunk, key) for chunk in chunks]
                # Some native bindings advance their organization clock while
                # sampling. Draw the epoch-one panels but calibrate the actual
                # untrained epoch-zero function, with its original clocks.
                for name, value in clocks.items():
                    setattr(core, name, value)
                denominators = self.loss_denominators(batches, phase='joint', arm=current_model.architecture)
                def predict(batch):
                    return self.predict_native(current_model, self.make_scene(batch.scene_inputs), batch.receivers,
                                               current_model.architecture, 'joint', epoch=0, temperature=1.)
                def native():
                    for batch in batches:
                        predictions, auxiliary = predict(batch)
                        terms = self.loss_terms(predictions, batch.targets, 'joint', auxiliary)
                        yield sum(term.numerator * float(term.weight) / float(denominators[name])
                                  for name, term in terms.items())
                def response():
                    for batch in batches[:len(selected)]:
                        predictions, _ = predict(batch)
                        values = [estimate.loss for _, estimate in estimates(batch, predictions)
                                  if estimate.loss is not None]
                        if len(values) != 1:
                            raise ValueError('Calibration pair lacks one supported native response estimate.')
                        yield values[0] / len(selected)
                diagnostic = gradient_diagnostics(current_model, {'native': native, 'response': response},
                                                  require_nonzero=True)
        finally:
            self._coupled_calibration_panel = False
            self.load_training_state_dict(before)
            self.coupled_native_reader_mode = previous_mode
            for name, value in clocks.items():
                setattr(core, name, value)
        norms = diagnostic['groups']['global']['norms']
        coefficient = .10 * float(norms['native']) / float(norms['response'])
        if not math.isfinite(coefficient) or not 0 < coefficient < 1.e6:
            raise ValueError('Epoch-zero response gradient calibration is numerically unusable.')
        return {'schema': 'coupled_native_calibration_v1', 'epoch': 0, 'coefficient': coefficient,
                'target_gradient_ratio': .10, 'calibration_case_ids': list(ids),
                'calibration_pairs': [list(pair) for pair in selected], 'gradient_diagnostics': diagnostic}

    for method in (identity_payload, training_state_dict, load_training_state_dict, reorder_epoch_cases,
                   sample_coupled_native_rows, begin_native_update, loss_terms, native_microbatch_terms,
                   validation_objective_predictions, validation_loss_terms, validation_reader_records, loss_metadata,
                   training_history_columns, training_column_definitions, set_coupled_calibration,
                   calibrate_coupled_native):
        setattr(provider, method.__name__, MethodType(method, provider))
    if variant != 'native_control':
        provider.validation_auxiliary_terms = MethodType(validation_auxiliary_terms, provider)
    provider.coupled_native_plan = get_plan
    provider.coupled_native_cache_usage = cache_usage
    provider.coupled_native_pairs = eligible_pairs
    provider.coupled_native_estimates = estimates
    return model, provider
