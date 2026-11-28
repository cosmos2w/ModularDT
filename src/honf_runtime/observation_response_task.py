"""Shared observation-response training attachment and capability dispatch.

A capability constructs an ordinary native task and describes existing physical
observations and their linear query dependencies.  All auxiliary scheduling,
reductions, calibration and recovery are owned here; adapters supply no loss.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import math
import time
from collections.abc import Mapping
from pathlib import Path
from types import MethodType

import numpy as np

from .gradient_diagnostics import gradient_diagnostics, preserved_state
from .state_response import ResponseObjectiveConfig, scheduled_native_pairs
from .unified_training import LossTerm, SamplingKey


_CAPABILITIES = {
    'thermal': 'channelthermal.training.observation_response:build_observation_capability',
    'wind': 'windfarm.training.observation_response:build_observation_capability',
    'analytic_fixture': 'honf_runtime.observation_response_fixture:build_observation_capability',
}
_NAMES = ('extra_absolute', 'point_response', 'spatial_response')
_TARGETS = {'extra_absolute': .25, 'point_response': .10, 'spatial_response': .05}
_ACTIVE = {
    'anchors': (1., 0., 0.),
    'contrasts': (1., 1., 0.),
    'spatial_contrasts': (1., 1., 1.),
}


def register_observation_capability(name, factory):
    """Register a case's native-task/observation binding without name branching.

    Factory signature is ``(recipe, *, device, resume_payload)`` and returns
    ``(model, native_provider, binding)``. See the collaborator guide.
    """
    if not isinstance(name, str) or not name or not callable(factory) or name in _CAPABILITIES:
        raise ValueError('A new observation capability requires a unique name and callable factory.')
    _CAPABILITIES[name] = factory


def _factory(name):
    found = _CAPABILITIES.get(name)
    if found is None:
        raise ValueError(f'Observation capability {name!r} is unavailable; register a native-task and observation binding.')
    if callable(found):
        return found
    if not isinstance(found, str) or found.count(':') != 1:
        raise ValueError('An observation factory must be specified as importable.module:function.')
    module, function = found.split(':', 1)
    if not module or not function:
        raise ValueError('An observation factory needs both an importable module and function name.')
    factory = getattr(importlib.import_module(module), function)
    if not callable(factory):
        raise TypeError('The declared observation factory is not callable.')
    return factory


def _case_seed(seed, epoch, pair):
    text = '\0'.join((str(seed), str(epoch), *map(str, pair))).encode()
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), 'little')


def _normalise_spec(recipe, binding):
    spec = dict(recipe.get('observation_response', {}))
    variant = spec.get('variant', 'spatial_contrasts')
    if variant not in _ACTIVE:
        raise ValueError('Observation variant must be anchors, contrasts, or spatial_contrasts.')
    seed = spec.get('pair_seed', 0)
    count = spec.get('seeds_per_pair', 128)
    presentations = spec.get('pairs_per_epoch', 3)
    if type(seed) is not int or seed < 0 or type(count) is not int or not 1 <= count <= 128:
        raise ValueError('Observation sampling requires a nonnegative seed and 1–128 draws per pair.')
    if type(presentations) is not int or presentations < 1:
        raise ValueError('pairs_per_epoch must be a positive integer.')
    pairs = tuple(tuple(map(str, pair)) for pair in (spec.get('pairs') or binding.training_pairs()))
    if len(pairs) < presentations or len(set(pairs)) != len(pairs):
        raise ValueError('Training needs enough distinct prescribed-input pair identities.')
    if any(len(p) != 2 or p[0] == p[1] for p in pairs):
        raise ValueError('Every observation pair must contain two distinct cases.')
    validation = tuple(tuple(map(str, pair)) for pair in spec.get('validation_pairs', ()))
    if not validation:
        validation = tuple(tuple(map(str, pair)) for pair in binding.validation_pairs())
    if not validation:
        raise ValueError('A fixed validation observation pair must be declared.')
    return spec, variant, seed, count, presentations, pairs, validation


def execute_observation_dependencies(binding, model, pair, union):
    """Prepare each requested scene once and decode its joint field union.

    A binding supplies prescribed scene/query carriers and the affine physical
    scale/offset of each output.  The returned predictions preserve union order
    and both full and restricted access views in one autograd graph.
    """
    import torch
    result = None
    work = {'neural_queries': 0, 'scene_preparations': 0}
    groups = {}
    for index, dependency in enumerate(union.dependencies):
        if dependency.scene_id not in pair:
            raise ValueError('An observation dependency is outside its declared scene pair.')
        key = (dependency.scene_id, dependency.frame, dependency.time)
        groups.setdefault(key, []).append(index)
    # Retain pair order for the ordinary single-frame path. Different frames
    # or times may require different scene controls or query carriers.
    ordered = [(key, positions) for case_id in pair
               for key, positions in groups.items() if key[0] == case_id]
    for (case_id, _frame, _time), positions in ordered:
        dependencies = tuple(union.dependencies[i] for i in positions)
        scene, query, scale, offset = binding.field_request(model, case_id, dependencies)
        organization = model.core.prepare(scene)
        full, restricted, shared = model.core.decode_paired(organization, query)
        raw = torch.stack((full.joint_fields[0], restricted.joint_fields[0]))
        if raw.ndim != 3 or raw.shape[1] != len(positions):
            raise ValueError('Joint decoder output does not match the observation dependency union.')
        scale = torch.as_tensor(scale, device=raw.device)
        offset = torch.as_tensor(offset, device=raw.device)
        if not scale.is_floating_point() or not offset.is_floating_point():
            raise TypeError('Physical output scales and offsets must be floating-point values.')
        dtype = torch.promote_types(raw.dtype, torch.promote_types(scale.dtype, offset.dtype))
        physical = raw.to(dtype) * scale.to(dtype) + offset.to(dtype)
        if physical.shape != raw.shape:
            raise ValueError('Physical output scale and offset must preserve the joint-field shape.')
        if result is None:
            result = physical.new_zeros((2, len(union.dependencies), raw.shape[-1]))
        elif result.shape[-1] != raw.shape[-1]:
            raise ValueError('All scenes in an observation pair need the same configured output schema.')
        dtype = torch.promote_types(result.dtype, physical.dtype)
        result = result.to(dtype).index_copy(1, torch.as_tensor(positions, device=raw.device), physical.to(dtype))
        work['neural_queries'] += len(positions)
        work['scene_preparations'] += 1
        for prefix, counts in (('paired', shared), ('full', full.work_counts), ('restricted', restricted.work_counts)):
            for key, value in (counts or {}).items():
                if isinstance(value, (int, float)):
                    name = f'{prefix}_{key}'
                    work[name] = work.get(name, 0) + value
    if result is None:
        raise ValueError('The selected observation dependencies contain no requested scene.')
    return result, work


def build_observation_response_task(recipe, *, device, resume_payload=None, _coupled_bootstrap=False):
    """Construct a native task with the shared generic observation objective."""
    if 'finite_observations' in recipe:
        # Independent scene rows need neither matched-row native capability
        # nor a uniform native sampling law.
        base = copy.deepcopy(recipe)
        finite = base.pop('finite_observations')
        sampling = finite.get('native_sampling', 'native')
        if sampling not in ('native', 'coupled_native'):
            raise ValueError('finite_observations.native_sampling must be native or coupled_native.')
        if sampling == 'native':
            base.setdefault('observation_response', {})['enabled'] = False
        model, provider = build_observation_response_task(base, device=device, resume_payload=resume_payload)
        from .finite_observation_task import attach_finite_observation_task
        return attach_finite_observation_task(recipe, model, provider, getattr(provider, 'observation_binding', None))
    enabled = recipe.get('observation_response', {}).get('enabled', True)
    if type(enabled) is not bool:
        raise ValueError('observation_response.enabled must be a boolean.')
    if (enabled and recipe.get('observation_response', {}).get('mode') == 'coupled_native'
            and not _coupled_bootstrap):
        # Keep the established independent observation evaluator available,
        # while disabling its auxiliary fitting path in the new attachment.
        base = copy.deepcopy(recipe)
        base['observation_response']['variant'] = 'anchors'
        base['observation_response']['pairs_per_epoch'] = 1
        base['observation_response']['seeds_per_pair'] = min(
            int(base['observation_response'].get('anchor_count', 128)), 128)
        base['observation_response'].pop('calibration', None)
        model, provider = build_observation_response_task(
            base, device=device, resume_payload=resume_payload, _coupled_bootstrap=True)
        from .coupled_response_task import attach_coupled_native_task
        return attach_coupled_native_task(recipe, model, provider, provider.observation_binding)
    dataset = recipe.get('dataset')
    inferred = dataset.get('name') if isinstance(dataset, Mapping) else dataset
    name = recipe.get('observation_response', {}).get('adapter', inferred)
    factory_path = recipe.get('observation_response', {}).get('adapter_factory')
    if factory_path is not None:
        if not isinstance(factory_path, str) or factory_path.count(':') != 1:
            raise ValueError('adapter_factory must name an importable module:function.')
        module, function = factory_path.split(':', 1)
        factory = getattr(importlib.import_module(module), function)
        if not callable(factory):
            raise TypeError('The declared observation factory is not callable.')
        name = factory_path
    else:
        factory = _factory(name)
    model, provider, binding = factory(recipe, device=device, resume_payload=resume_payload)
    if binding is not None:
        provider.observation_binding = binding
    if not enabled:
        return model, provider
    if binding is None:
        raise ValueError('Enabled observation training requires an observation binding.')
    spec, variant, seed, count, presentations, pairs, validation = _normalise_spec(recipe, binding)
    block_names = tuple(binding.block_names())
    if not block_names or len(set(block_names)) != len(block_names) or any(
            not isinstance(item, str) or not item for item in block_names):
        raise ValueError('Observation bindings must declare their stable named-block inventory.')
    for pair in (*pairs, *validation):
        binding.validate_pair(pair)
    state = {'calibration': copy.deepcopy(spec.get('calibration')), 'presentations': 0}
    active = dict(zip(_NAMES, _ACTIVE[variant]))
    provider.observation_binding = binding
    provider.observation_pairs = pairs
    provider.auxiliary_loss_names = tuple('observation_response/' + item for item in _NAMES)
    provider.training_output_policy = 'csv_v3'
    old_identity = provider.identity_payload
    old_state = getattr(provider, 'training_state_dict', lambda: {})
    old_load = getattr(provider, 'load_training_state_dict', lambda saved: None)
    old_metadata = provider.loss_metadata
    old_columns = provider.training_history_columns
    old_definitions = getattr(provider, 'training_column_definitions', lambda: {})
    anchor_keys = tuple(getattr(getattr(model, 'core', model), '_ANCHOR_COUNT_KEYS', ()))
    work_names = ('presentations', 'draws', 'dependencies', 'requested_dependencies', 'extraction_terms',
                  'covered_blocks', 'observation_records', 'spatial_edges', 'neural_queries',
                  'scene_preparations', 'planning_seconds', 'prediction_seconds') + tuple(
        f'{view}_{key}' for view in ('paired', 'full', 'restricted') for key in anchor_keys) + tuple(
        f'block_{block}_{measure}' for block in block_names for measure in ('draws', 'spatial_edges'))

    def identity_payload(self):
        return {**old_identity(), 'observation_response': {
            'capability': name, 'variant': variant, 'pairs': [list(p) for p in pairs],
            'validation_pairs': [list(p) for p in validation], 'pair_seed': seed,
            'seeds_per_pair': count, 'pairs_per_epoch': presentations,
            'sampling': 'weighted blocks and uniform seeds with replacement, up to four nearest compatible neighbors',
            'binding': binding.identity_payload(), 'block_names': list(block_names),
            'validation_objective_complete': True,
            'prediction_views': 'equal full and restricted joint-field reads when provided by the capability',
        }}

    def training_state_dict(self):
        return {'native': old_state(), 'observation_response': copy.deepcopy(state)}

    def load_training_state_dict(self, saved):
        if not isinstance(saved, Mapping) or not isinstance(saved.get('observation_response'), Mapping):
            raise ValueError('Missing saved observation-response provider state.')
        old_load(saved['native'])
        state.clear()
        state.update(copy.deepcopy(saved['observation_response']))

    def initialize_observation_response_state(self, native_state, checked):
        native_state = copy.deepcopy(native_state)
        if 'field_response' in native_state:
            if native_state['field_response'].get('calibration') is not None:
                raise ValueError('The source prefix has already fitted an auxiliary response.')
            native_state = native_state['native']
        return {'native': native_state, 'observation_response': {'calibration': copy.deepcopy(checked),
                                                                  'presentations': 0}}

    def validate_observation_calibration(self, record, *, source_checkpoint):
        if not isinstance(record, Mapping) or record.get('schema') != 'observation_response_calibration_v1':
            raise ValueError('Shared observation calibration record has an unknown schema.')
        if int(record.get('epoch', -1)) != 10 or Path(record.get('source_checkpoint', '')).resolve() != Path(source_checkpoint).resolve():
            raise ValueError('Shared calibration must identify the exact epoch-10 source checkpoint.')
        if record.get('binding') != binding.identity_payload() or record.get('pair_seed') != seed or record.get('seeds_per_pair') != count:
            raise ValueError('Calibration observation schema, seed, or sampling budget differs.')
        coefficients = record.get('coefficients', {})
        norms = record.get('gradient_diagnostics', {}).get('groups', {}).get('global', {}).get('norms', {})
        if set(coefficients) != set(_NAMES) or any(not math.isfinite(float(coefficients[n])) or float(coefficients[n]) <= 0 for n in _NAMES):
            raise ValueError('Calibration needs three finite positive measured coefficients.')
        if any(not math.isfinite(float(norms.get(n, 0))) or float(norms[n]) <= 0 for n in ('native', *_NAMES)):
            raise ValueError('Calibration has a zero, unresolved, or nonfinite gradient norm.')
        for n in _NAMES:
            expected = _TARGETS[n] * float(norms['native']) / float(norms[n])
            if not math.isclose(float(coefficients[n]), expected, rel_tol=1e-10, abs_tol=0.):
                raise ValueError(f'Coefficient for {n} does not match its recorded gradient calibration.')
        ids = tuple(map(str, record.get('calibration_case_ids', ())))
        if ids != tuple(map(str, binding.calibration_case_ids())):
            raise ValueError('Calibration case identities differ from the declared input-selected panel.')
        return copy.deepcopy(dict(record))

    def components(self, current_model, pair, *, epoch, details=False):
        from .observation_blocks import (prepare_observation_sampling, sample_observation_blocks,
            union_observation_dependencies, extract_observations, observation_response_losses)
        started = time.perf_counter()
        active_binding = self.observation_binding
        blocks = tuple(active_binding.observation_blocks(pair))
        plan = prepare_observation_sampling(blocks, max_neighbors=4)
        draws = sample_observation_blocks(plan, count, np.random.default_rng(_case_seed(seed, epoch, pair)))
        union = union_observation_dependencies(blocks, draws)
        planned = time.perf_counter()
        predicted, work = execute_observation_dependencies(active_binding, current_model, pair, union)
        predictions = []
        for block_index, block in enumerate(blocks):
            scenes = [extract_observations(block, scene, predicted, union.mappings[block_index][scene])
                      for scene in range(2)]
            import torch
            predictions.append(torch.stack(scenes, dim=1))
        result = observation_response_losses(blocks, predictions, draws)
        losses = {name: result[name] for name in _NAMES}
        coverage = result['coverage']
        if not set(coverage) <= set(block_names):
            raise ValueError('An observation block is missing from the binding recording inventory.')
        ended = time.perf_counter()
        record = {'observation_response_presentations': 1,
                  'observation_response_draws': count,
                  'observation_response_dependencies': len(union.dependencies),
                  'observation_response_planning_seconds': planned - started,
                  'observation_response_prediction_seconds': ended - planned}
        record.update(observation_response_covered_blocks=sum(row['draws'] > 0 for row in coverage.values()),
                      observation_response_observation_records=sum(row['endpoints'] for row in coverage.values()),
                      observation_response_spatial_edges=sum(row['edges'] for row in coverage.values()),
                      observation_response_requested_dependencies=union.requested_dependencies,
                      observation_response_extraction_terms=union.requested_terms)
        for block in block_names:
            actual = coverage.get(block, {})
            record[f'observation_response_block_{block}_draws'] = int(actual.get('draws', 0))
            record[f'observation_response_block_{block}_spatial_edges'] = int(actual.get('edges', 0))
        record.update({f'observation_response_{k}': v for k, v in work.items()
                       if isinstance(v, (int, float))})
        if details:
            return losses, record, {'blocks': blocks, 'draws': draws, 'union': union,
                                   'predictions': tuple(predictions), 'loss_result': result}
        return losses, record

    def calibrate_observation_response(self, current_model, *, source_checkpoint):
        ids = tuple(map(str, binding.calibration_case_ids()))
        chosen = tuple(pair for pair in pairs if set(pair) <= set(ids))
        if not chosen:
            raise ValueError('The declared TRAIN calibration panel has no compatible prescribed-input pairs.')
        before = copy.deepcopy(self.training_state_dict())
        sample_counters = {name: copy.deepcopy(getattr(self, name)) for name in
                           ('training_samples', 'validation_samples') if hasattr(self, name)}
        key = SamplingKey(seed=self.seed, epoch=10, update_index=0, microbatch_index=0,
            stage='joint', arm=self.recipe.get('training', {}).get('sampling_arm', getattr(current_model, 'architecture', 'relation_anchored')),
            sampling_version=self.sampling_version, dataset_id=self.sampling_dataset_id)
        def native():
            for case_id in ids:
                batch = self.make_batch((binding.native_case_key(case_id),), key)
                scene = self.make_scene(batch.scene_inputs)
                prediction, auxiliary = self.predict_native(current_model, scene, batch.receivers,
                    current_model.architecture, 'joint', epoch=10, temperature=1.)
                terms = self.loss_terms(prediction, batch.targets, 'joint', auxiliary)
                denominators = self.loss_denominators([batch], phase='joint', arm=current_model.architecture)
                yield sum(term.numerator * float(term.weight) / float(denominators[name])
                          for name, term in terms.items()) / len(ids)
        def auxiliary(name):
            def pieces():
                for pair in chosen:
                    values, _ = components(self, current_model, pair, epoch=10)
                    yield values[name] / len(chosen)
            return pieces
        try:
            with preserved_state(current_model):
                current_model.train(True)
                diagnostic = gradient_diagnostics(current_model,
                    {'native': native, **{name: auxiliary(name) for name in _NAMES}}, require_nonzero=True)
        finally:
            self.load_training_state_dict(before)
            for counter, value in sample_counters.items():
                setattr(self, counter, value)
        norms = diagnostic['groups']['global']['norms']
        coefficients = {n: _TARGETS[n] * norms['native'] / norms[n] for n in _NAMES}
        return {'schema': 'observation_response_calibration_v1', 'epoch': 10,
                'source_checkpoint': str(Path(source_checkpoint).resolve()),
                'calibration_case_ids': list(ids), 'calibration_pairs': [list(pair) for pair in chosen],
                'binding': binding.identity_payload(), 'pair_seed': seed, 'seeds_per_pair': count,
                'gradient_diagnostics': diagnostic, 'coefficients': coefficients}

    def auxiliary_update_terms(self, current_model, *, epoch, update_index, update_count):
        ramp = 1. if spec.get('fixture_no_ramp', False) else ResponseObjectiveConfig().ramp(epoch)
        if not ramp:
            return
        calibration = state['calibration']
        if not isinstance(calibration, Mapping) or not isinstance(calibration.get('coefficients'), Mapping):
            raise ValueError('The common TRAIN observation calibration is missing.')
        selected = scheduled_native_pairs(pairs, epoch, presentations, seed)
        slots = np.linspace(0, update_count - 1, len(selected), dtype=int)
        for pair, slot in zip(selected, slots):
            if int(slot) == update_index:
                values, work = components(self, current_model, pair, epoch=epoch)
                state['presentations'] += 1
                yield ({'observation_response/' + n: LossTerm(values[n], 1.,
                    ramp * active[n] * float(calibration['coefficients'][n]) * update_count / len(selected))
                    for n in _NAMES}, work)

    def validation_auxiliary_terms(self, current_model, *, epoch, arm):
        if arm != current_model.architecture:
            raise ValueError('Observation validation must use the current native architecture.')
        ramp = 1. if spec.get('fixture_no_ramp', False) else ResponseObjectiveConfig().ramp(epoch)
        calibration = state['calibration']
        if ramp and not calibration:
            raise ValueError('Observation validation needs the saved calibration.')
        mode = current_model.training
        try:
            current_model.train(True)
            rows = [components(self, current_model, pair, epoch=10)[0] for pair in validation]
        finally:
            current_model.train(mode)
        return {'observation_response/' + n: LossTerm(sum(row[n] for row in rows), len(rows),
            ramp * active[n] * float((calibration or {}).get('coefficients', {}).get(n, 0.))) for n in _NAMES}

    def with_observation_evaluation_pairs(self, evaluation_pairs):
        """Return an evaluation-only view with explicitly admitted case IDs."""
        pairs_for_evaluation = tuple(tuple(map(str, pair)) for pair in evaluation_pairs)
        copier = getattr(self.observation_binding, 'for_evaluation', None)
        if not callable(copier):
            for pair in pairs_for_evaluation:
                self.observation_binding.validate_pair(pair)
            return self
        view = copy.copy(self)
        view.observation_binding = copier(pairs_for_evaluation)
        view.observation_components = MethodType(components, view)
        return view

    def loss_metadata(self):
        result = dict(old_metadata())
        for item in _NAMES:
            result['observation_response/' + item] = {'panel': 'physics', 'label': item.replace('_', ' '),
                'unit': 'dimensionless squared error', 'formula': 'fixed physical measure with block and receiver sampling correction; mean of declared views',
                'weight': 'shared TRAIN gradient calibration, candidate coefficient and epochs 11–50 ramp'}
        return result

    def training_history_columns(self):
        native = tuple(c for c in old_columns() if not c.startswith(('train_losses.observation_response/', 'work_counts.native_pair_', 'work_counts.field_response_')))
        # Some existing native column methods inspect self.auxiliary_loss_names.
        native = tuple(c for c in native if not c.startswith('train_losses.observation_response/'))
        return (*native, *(f'train_losses.observation_response/{n}' for n in _NAMES),
                *(f'work_counts.observation_response_{n}' for n in work_names))

    def training_column_definitions(self):
        columns = self.training_history_columns()
        result = {k: v for k, v in old_definitions().items() if k in columns}
        for n in work_names:
            column = 'work_counts.observation_response_' + n
            result[column] = {'dtype': 'float',
                              'unit': 's' if n.endswith('_seconds') else 'count',
                              'meaning': 'Sum across observation pair presentations of ' + n.replace('_', ' ')}
        return result

    for method in (identity_payload, training_state_dict, load_training_state_dict,
                   initialize_observation_response_state, validate_observation_calibration,
                   calibrate_observation_response, auxiliary_update_terms, validation_auxiliary_terms,
                   with_observation_evaluation_pairs, loss_metadata, training_history_columns, training_column_definitions):
        setattr(provider, method.__name__, MethodType(method, provider))
    provider.observation_components = MethodType(components, provider)
    return model, provider
