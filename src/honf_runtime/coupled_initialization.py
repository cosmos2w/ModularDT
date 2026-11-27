"""Save and apply one actual, shared zero-epoch state for coupled training."""
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import torch

from .unified_training import (_cpu_tree, _optimizer_spec_payload, capture_rng_state,
                               named_optimizer_state, restore_rng_state)


def _same(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right, equal_nan=True)
    if isinstance(left, Mapping):
        return isinstance(right, Mapping) and left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(map(lambda p: _same(*p), zip(left, right)))
    return left == right


def _initial_provider_identity(provider_identity):
    """Exclude native activation storage from the shared initial-state contract."""
    identity = copy.deepcopy(dict(provider_identity))
    identity.pop('native_retention_execution', None)
    recipe = identity.get('recipe')
    if isinstance(recipe, dict):
        execution = recipe.get('execution_config')
        if isinstance(execution, dict):
            execution.pop('native_retention_fraction', None)
            # An absent and an empty execution mapping both request defaults.
            if not execution:
                recipe.pop('execution_config')
    return identity


def common_contract(model, provider, engine):
    identity = _initial_provider_identity(provider.identity_payload())
    observation = identity.get('observation_response')
    if not isinstance(observation, dict) or observation.get('mode') != 'coupled_native':
        raise ValueError('A common epoch-zero state requires the explicit coupled native profile.')
    observation.pop('variant', None)
    observation.pop('validation_objective_complete', None)
    # The multiplier changes the later objective, not the common initial state
    # or the measured, unscaled epoch-zero calibration shared by all arms.
    for key in ('response_weight', 'response_update_scope', 'response_gradient_composition'):
        observation.pop(key, None)
    # The provider embeds its source recipe for provenance. A descriptive run
    # name does not change the initialized model, native objective or samples.
    source_recipe = identity.get('recipe')
    if isinstance(source_recipe, dict):
        source_recipe.pop('name', None)
        source_observation = source_recipe.get('observation_response')
        if isinstance(source_observation, dict):
            for key in ('variant', 'response_weight', 'response_update_scope',
                        'response_gradient_composition'):
                source_observation.pop(key, None)
    config = engine.config
    return {'provider': identity,
            'optimizer_groups': [_optimizer_spec_payload(spec) for spec in
                                 provider.optimizer_groups(model, model.architecture, 'joint')],
            'sampling': {'seed': config.seed, 'version': config.sampling_version,
                         'microbatch_cases': config.microbatch_cases, 'effective_cases': config.effective_cases,
                         'deterministic_case_order': config.deterministic_case_order},
            'horizon': config.total_epochs}


def prepare_epoch_zero(model, provider, engine):
    """Create a calibration and snapshot without an optimizer update or clock advance."""
    contract = common_contract(model, provider, engine)
    core = getattr(model, 'core', model)
    clocks = {name: int(getattr(core, name, 0)) for name in ('_joint_epoch', '_joint_update_index')}
    if any(clocks.values()):
        raise ValueError(f'Fresh organization clocks must begin at zero: {clocks}')
    optimizer, _ = engine._make_optimizer(model, provider, model.architecture, 1, None)
    optimizer_state = named_optimizer_state(model, optimizer)
    model_before = copy.deepcopy(model.state_dict())
    provider_before = copy.deepcopy(provider.training_state_dict())
    rng_before = capture_rng_state()
    calibration = provider.calibrate_coupled_native(model)
    if (not _same(model_before, model.state_dict()) or not _same(provider_before, provider.training_state_dict())
            or not _same(rng_before, capture_rng_state()) or not _same(optimizer_state, named_optimizer_state(model, optimizer))
            or any(int(getattr(core, name, 0)) != value for name, value in clocks.items())):
        raise RuntimeError('Epoch-zero calibration changed initialized model, optimizer, provider, RNG or clocks.')
    provider.set_coupled_calibration(calibration)
    return {'schema': 'coupled_native_epoch_zero_v1', 'epoch': 0, 'contract': contract,
            'model_state_dict': _cpu_tree(model.state_dict()), 'optimizer_state_by_name': optimizer_state,
            'provider_training_state': provider.training_state_dict(), 'rng_state': rng_before,
            'sampler_state': {'completed_epoch': 0, 'next_epoch': 1, 'next_case_cursor': 0,
                              'query_seed': engine.config.seed,
                              'pair_seed': contract['provider']['observation_response']['pair_seed'],
                              'reader_seed': contract['provider']['observation_response']['reader_seed'],
                              'stream_keying': 'independent named deterministic case/epoch streams'},
            'organization_clocks': clocks, 'scheduler_epoch': 0, 'selector_epoch': 0,
            'calibration': calibration,
            'metadata_cache_at_preparation': copy.deepcopy(getattr(provider, 'coupled_native_cache_usage', {}))}


def apply_epoch_zero(model, provider, engine, payload):
    """Load the same saved tensors and recovery state into one fresh comparison."""
    if (not isinstance(payload, Mapping) or payload.get('schema') != 'coupled_native_epoch_zero_v1'
            or payload.get('epoch') != 0 or payload.get('scheduler_epoch') != 0
            or payload.get('selector_epoch') != 0 or payload.get('sampler_state', {}).get('next_epoch') != 1
            or any(payload.get('organization_clocks', {}).values())):
        raise ValueError('The coupled initialization is not a genuine zero-epoch state.')
    saved_contract = copy.deepcopy(payload.get('contract'))
    if isinstance(saved_contract, dict) and isinstance(saved_contract.get('provider'), Mapping):
        saved_contract['provider'] = _initial_provider_identity(saved_contract['provider'])
    if not _same(saved_contract, common_contract(model, provider, engine)):
        raise ValueError('Epoch-zero native cohort, model, objective, sampling or optimizer contract differs.')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    provider.load_training_state_dict(payload['provider_training_state'])
    provider.set_coupled_calibration(payload['calibration'])
    core = getattr(model, 'core', model)
    for name, value in payload['organization_clocks'].items():
        setattr(core, name, int(value))
    restore_rng_state(payload['rng_state'])
    return copy.deepcopy(payload['optimizer_state_by_name'])
