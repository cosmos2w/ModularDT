"""Explicit state-preserving input forks; ordinary resumes remain strict."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import torch

from honf_runtime.unified_training import (
    _engine_config_payload, _optimizer_spec_payload,
)

PROJECTION = 'core.module_encoder.0.weight'


def insert_area_column(value):
    """Keep the nine original inputs and all positional columns verbatim."""
    if value.ndim != 2 or value.shape[1] < 9:
        raise ValueError('Expected the original module input projection matrix.')
    return torch.cat((value[:, :9], value.new_zeros(value.shape[0], 1), value[:, 9:]), dim=1)


def validate_fork_recipe(parent, child, source_path):
    declaration = child.get('checkpoint_fork', {})
    if (declaration.get('schema') != 'thermal_state_response_input_fork_v1'
            or type(declaration.get('source_epoch')) is not int or declaration['source_epoch'] < 1
            or type(declaration.get('additional_epochs')) is not int or declaration['additional_epochs'] < 1
            or type(declaration.get('raster_area_feature')) is not bool
            or Path(declaration.get('source_checkpoint', '')).resolve() != Path(source_path).resolve()):
        raise ValueError('Declare the exact source checkpoint, its age and the additional epoch budget.')
    stop = declaration['source_epoch'] + declaration['additional_epochs']
    amended = copy.deepcopy(child)
    amended.pop('checkpoint_fork')
    amended['name'] = parent['name']
    for name in ('local_horizon', 'first_segment_epochs', 'optimizer_schedule_horizon', 'checkpoint_epochs'):
        amended['training'][name] = parent['training'][name]
    width = child['member_routing']['model']['base_model_config']['module_input_dim']
    if width != (10 if declaration['raster_area_feature'] else 9):
        raise ValueError('The declared feature and module input width disagree.')
    amended['member_routing']['model']['base_model_config']['module_input_dim'] = 9
    checkpoints = sorted(set([*range(100, stop + 1, 100), stop]))
    if (amended != parent or child['training']['local_horizon'] != stop
            or child['training']['optimizer_schedule_horizon'] != stop
            or child['training']['first_segment_epochs'] != stop
            or child['training']['checkpoint_epochs'] != checkpoints
            or parent['training']['membership'] != 'fixed25_train150_dev22'
            or parent['member_routing']['model'].get('reader_capability') != 'anchored'):
        raise ValueError('An input fork may change only its label, declared horizon and tenth input.')
    optimizer = parent['optimizer']
    if optimizer['schedule'] != 'constant_config_lr':
        raise ValueError('This fork preserves the existing constant AdamW schedule only.')
    return declaration


def _original_width(value):
    if isinstance(value, dict):
        return {key: (9 if key == 'module_input_dim' and item == 10 else _original_width(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_original_width(item) for item in value]
    return value


def make_fork_payload(source, recipe, model, provider, engine, source_path):
    """Initialize a distinct child checkpoint without advancing scientific age.

    The new projection column inherits its enclosing parameter's AdamW clock;
    its first/second moments start at zero. Every common tensor/moment, RNG,
    provider counter, history row and absolute sampler age remains unchanged.
    """
    parent_identity = source['experiment_identity']
    declaration = validate_fork_recipe(parent_identity['recipe'], recipe, source_path)
    age = declaration['source_epoch']
    stop = age + declaration['additional_epochs']
    if (source['epoch'] != age or source['current_epoch'] != age
            or len(source['history']) != age
            or [row['epoch'] for row in source['history']] != list(range(1, age + 1))):
        raise ValueError('The source must contain the complete declared trajectory.')
    current_provider = copy.deepcopy(dict(provider.identity_payload()))
    checked_provider = _original_width(copy.deepcopy(current_provider))
    checked_provider['recipe'] = parent_identity['provider_identity']['recipe']
    # Rebind only the declared horizon of the same constant learning law.
    if 'training' in checked_provider:
        old_training = parent_identity['provider_identity']['training']
        checked_provider['training']['total_epochs'] = old_training['total_epochs']
        for key in ('total_epochs', 'hold_through_epoch'):
            checked_provider['training']['optimizer_schedule'][key] = old_training['optimizer_schedule'][key]
    if checked_provider != parent_identity['provider_identity']:
        changed = sorted(key for key in set(checked_provider) | set(parent_identity['provider_identity'])
                         if checked_provider.get(key) != parent_identity['provider_identity'].get(key))
        raise ValueError(f'Input fork changed other native/provider semantics: {changed}')
    target = model.state_dict()
    weights = copy.deepcopy(source['model_state_dict'])
    moments = copy.deepcopy(source['optimizer_state_by_name'])
    if declaration['raster_area_feature']:
        weights[PROJECTION] = insert_area_column(weights[PROJECTION])
        for name, value in moments['state_by_name'][PROJECTION].items():
            if torch.is_tensor(value) and name != 'step':
                moments['state_by_name'][PROJECTION][name] = insert_area_column(value)
    if set(target) != set(weights) or any(target[key].shape != weights[key].shape for key in target):
        raise ValueError('Input fork changed a tensor outside the declared projection.')
    model.load_state_dict(weights, strict=True)
    provider.load_training_state_dict(copy.deepcopy(source['provider_training_state']))
    for name, state in moments['state_by_name'].items():
        parameter = dict(model.named_parameters())[name]
        for key, value in state.items():
            if torch.is_tensor(value) and key != 'step' and value.shape != parameter.shape:
                raise ValueError(f'Optimizer moment shape differs for {name}.{key}.')
    specs = tuple(provider.optimizer_groups(model, source['arm'], 'joint'))
    schedules = [_optimizer_spec_payload(spec) for spec in specs]
    original_schedules = source['optimizer_schedule_contract']
    checked_schedules = copy.deepcopy(schedules)
    for new, old in zip(checked_schedules, original_schedules):
        new['schedule']['total_epochs'] = old['schedule']['total_epochs']
        new['schedule']['hold_through_epoch'] = old['schedule']['hold_through_epoch']
    if checked_schedules != original_schedules:
        raise ValueError('An input fork changed optimizer groups or its constant learning law.')
    identity = {**copy.deepcopy(parent_identity), 'recipe': copy.deepcopy(recipe),
                'provider_identity': current_provider, 'engine_config': _engine_config_payload(engine.config),
                'selection_policy': asdict(engine.selection), 'optimizer_schedule_contract': schedules}
    child = copy.deepcopy(source)
    child.update(experiment_identity=identity, model_state_dict=weights,
                 optimizer_state_by_name=moments, optimizer_schedule_contract=schedules,
                 best_field_score=float('inf'), best_response_guarded_score=float('inf'))
    receipt = {'kind': 'state_preserving_prescribed_input_fork', 'source_checkpoint': str(Path(source_path).resolve()),
               'source_epoch': age, 'first_new_epoch': age + 1, 'stop_epoch': stop,
               'additional_epochs': declaration['additional_epochs'], 'raster_area_feature': declaration['raster_area_feature'],
               'original_initialization': 'recipe initialization retains the source scratch origin; this child is a checkpoint fork',
               'projection': PROJECTION, 'inserted_input_column': 9 if declaration['raster_area_feature'] else None,
               'new_column_weights_and_moments': 'zero; enclosing parameter AdamW clock inherited' if declaration['raster_area_feature'] else 'no new entries',
               'common_state': 'all other weights/moments, normalization, objective scales, RNG, provider state and complete history inherited',
               'sampler': f'absolute epochs {age + 1}–{stop}; original case/query namespace and FIT48 family cycle',
               'selection': 'new child validations only; parent scores and history preserved as historical evidence'}
    child['branch_parent'] = receipt
    return child, receipt
