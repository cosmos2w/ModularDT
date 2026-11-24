"""Explicit new CSV lineages from a completed native-only epoch-ten state.

This initializes a new run using the existing checkpoint and recording formats.
It never changes an existing run or relaxes ordinary resume validation.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

from .compat import load_trusted_checkpoint
from .run_layout import RunLayout
from .scalar_history import (VALIDATION_COLUMNS, append_scalar_rows, read_scalar_rows,
                             training_recording_definitions)
from .unified_training import (_atomic_json, _atomic_torch_save, _engine_config_payload,
                              _optimizer_spec_payload, _render_loss_curves,
                              _sampling_dataset_id, read_training_history, SamplingKey)


def _native_recipe(recipe):
    result = copy.deepcopy(recipe)
    if result.get('training_profile') == 'observation_response':
        result['training_profile'] = 'field_response'
        result.pop('observation_response', None)
        # The declared organization control changes query priors only. It
        # inherits the same model tensors and native optimizer state.
        model = result.get('member_routing', {}).get('model', {})
        if model.get('within_role_access_prior') == 'measure_only':
            model.pop('within_role_access_prior')
        execution = result.get('execution_config', {})
        if execution.get('deterministic_algorithms') is True:
            execution.pop('deterministic_algorithms')
    if result.get('training_profile') == 'local_multisecant_response':
        result['training_profile'] = 'field_response'
        result.pop('thermal_local_response', None)
    for key in ('name', 'storage', 'field_response_parent'):
        result.pop(key, None)
    response = result.get('thermal_field_response', {})
    for key in ('variant', 'label', 'calibration', 'difference_weight_amplitude'):
        response.pop(key, None)
    result.get('execution_config', {}).pop('native_retention_fraction', None)
    if result.get('execution_config') == {}:
        result.pop('execution_config')
    if 'training' in result and 'microbatch_cases' in result['training']:
        result['training']['microbatch_cases'] = 24
    return result


def initialize_native_prefix_child(source_checkpoint, output, recipe, model, provider, engine,
                                   *, calibration=None, render_curves=True):
    """Initialize a fresh child, inheriting native training state and scalar rows.

    The caller builds the child through its maintained entry point. Calibration
    is one previously measured TRAIN-only record shared by both paired children.
    No model prediction or optimizer update is performed here.
    """
    source_path = Path(source_checkpoint).expanduser().resolve(strict=True)
    if source_path.name != 'epoch_0010_model.pt':
        raise ValueError('Use the literal epoch_0010_model.pt snapshot, never an advancing selector.')
    source_run = source_path.parent.parent
    output = Path(output).expanduser().resolve()
    if output == source_run or (output.exists() and any(output.iterdir())):
        raise ValueError('A common-prefix child requires a distinct empty output directory.')
    before = source_path.stat()
    source = load_trusted_checkpoint(source_path, map_location='cpu')
    if engine.config.sampling_version != SamplingKey.CASE_EPOCH_VERSION:
        raise ValueError('Common-prefix field-response branching requires case_epoch_v1 sampling.')
    parent_identity = source['experiment_identity']
    parent_recipe = parent_identity['recipe']
    amplitude = recipe.get('thermal_field_response', {}).get('difference_weight_amplitude', 0)
    if amplitude not in (0, 0.5, 0.7):
        raise ValueError('The inactive prefix permits only the declared auxiliary weighting strengths.')
    parent_batch = parent_recipe.get('training', {}).get('microbatch_cases')
    child_batch = recipe.get('training', {}).get('microbatch_cases')
    local_profile = recipe.get('training_profile') == 'local_multisecant_response'
    observation_profile = recipe.get('training_profile') == 'observation_response'
    if (local_profile or observation_profile) and parent_batch != child_batch:
        raise ValueError('General-response children must preserve the native prefix microbatch.')
    if observation_profile:
        prior = recipe.get('member_routing', {}).get('model', {}).get('within_role_access_prior')
        if prior not in (None, 'measure_only'):
            raise ValueError('A native-prefix organization control supports only within-role measure-only priors.')
    if parent_batch != child_batch and not (parent_batch == 24 and child_batch in (6, 12)
            and recipe.get('training', {}).get('effective_batch_size') == 48):
        raise ValueError('Only the declared effective-batch-preserving microbatch changes are supported.')
    if (source.get('epoch') != 10 or parent_recipe.get('training_profile') != 'field_response'
            or parent_recipe.get('thermal_field_response', {}).get('variant') != 'field_only'
            or not ((recipe.get('training_profile') == 'field_response'
                     and recipe.get('thermal_field_response', {}).get('variant') in
                     ('field_anchors', 'field_response'))
                    or (local_profile and recipe.get('thermal_local_response', {}).get('variant') in
                        ('state_anchors', 'uniform_response', 'multisecant_response', 'soft_inverse_response',
                         'targeted_reynolds_response', 'trace_matched_uniform_response'))
                    or (observation_profile and recipe.get('observation_response', {}).get('variant') in
                        ('anchors', 'contrasts', 'spatial_contrasts')))):
        raise ValueError('Field-response children require the completed native-only epoch-ten prefix.')
    if _native_recipe(parent_recipe) != _native_recipe(recipe):
        raise ValueError('Common-prefix branching cannot change native data, model, objective, sampling or schedule.')
    if calibration is None:
        raise ValueError('Paired children require the one shared TRAIN-only prefix calibration.')
    parent_layout = RunLayout(source_run)
    parent_rows = read_scalar_rows(parent_layout.read_path('history.csv'))[:10]
    committed = source.get('committed_validation_rows')
    if not isinstance(committed, int) or committed < 0:
        raise ValueError('The prefix checkpoint needs a committed validation row count.')
    all_validation = read_scalar_rows(parent_layout.read_path('validation.csv'))
    if len(all_validation) < committed:
        raise ValueError('Prefix validation records differ from the committed checkpoint.')
    validation_rows = all_validation[:committed]
    if [int(row['epoch']) for row in parent_rows] != list(range(1, 11)):
        raise ValueError('The prefix requires its ten complete committed scalar rows.')
    if any(int(row['epoch']) > 10 for row in validation_rows):
        raise ValueError('Prefix validation contains measurements beyond its checkpoint.')
    sampler = source.get('sampler_state', {})
    if (sampler.get('completed_epoch') != 10 or sampler.get('next_epoch') != 11
            or sampler.get('next_case_cursor') != 0
            or any(int(row['case_visits']) != sampler.get('case_visits_this_epoch')
                   or int(row['optimizer_updates']) != sampler.get('optimizer_updates_this_epoch')
                   for row in parent_rows)):
        raise ValueError('Prefix history visit/update counters differ from the checkpoint sampler.')
    native_state = copy.deepcopy(source.get('provider_training_state', {}))
    if native_state.get('field_response', {}).get('calibration') is not None:
        raise ValueError('The common prefix must have no auxiliary calibration or fitting.')
    if observation_profile:
        checked = provider.validate_observation_calibration(calibration, source_checkpoint=source_path)
        provider_state = provider.initialize_observation_response_state(native_state, checked)
    elif local_profile:
        checked = provider.validate_local_calibration(calibration, source_checkpoint=source_path)
        provider_state = provider.initialize_local_response_state(native_state, checked)
    else:
        provider_state = {**native_state, 'field_response': {'calibration': copy.deepcopy(calibration)}}
    provider.load_training_state_dict(provider_state)
    model.load_state_dict(source['model_state_dict'], strict=True)
    arm = source['arm']
    specs = tuple(provider.optimizer_groups(model, arm, engine.config.stage_for_epoch(11)))
    schedules = [_optimizer_spec_payload(spec) for spec in specs]
    if schedules != source['optimizer_schedule_contract']:
        raise ValueError('Child optimizer groups or absolute schedules changed.')
    provider_identity = dict(provider.identity_payload())
    identity = {'workflow': parent_identity['workflow'], 'recipe': copy.deepcopy(recipe),
                'provider_identity': provider_identity,
                'engine_config': _engine_config_payload(engine.config),
                'selection_policy': asdict(engine.selection),
                'optimizer_schedule_contract': schedules}
    epoch_contract = engine.sampler_epoch_contract()
    if epoch_contract is not None:
        identity['sampler_epoch_mapping'] = dict(epoch_contract)
    identity['sampling_dataset_id'] = _sampling_dataset_id(provider_identity)
    for key in ('sampling_dataset_id', 'sampler_epoch_mapping', 'engine_config', 'selection_policy'):
        if key == 'engine_config' and parent_batch != child_batch:
            parent_engine, child_engine = copy.deepcopy(parent_identity.get(key)), copy.deepcopy(identity.get(key))
            if parent_engine.get('microbatch_cases') != parent_batch or child_engine.get('microbatch_cases') != child_batch:
                raise ValueError('The engine microbatch does not match its recipe.')
            child_engine['microbatch_cases'] = parent_batch
            if child_engine == parent_engine:
                continue
        if identity.get(key) != parent_identity.get(key):
            raise ValueError(f'Common-prefix branching changed {key}.')
    base_columns = ('epoch', 'phase', 'arm', 'temperature', 'case_visits', 'optimizer_updates',
                    'microbatches', 'partial_update_cases', 'train_seconds', 'train_loss_total')
    from .state_priority import recording_columns_and_definitions
    provider_columns, provider_definitions = recording_columns_and_definitions(provider)
    columns = (*base_columns, *(f'learning_rates.{name}' for name in source['optimizer_group_names']),
               *provider_columns)
    definitions = training_recording_definitions(columns, dict(provider.loss_metadata()),
        provider_definitions)
    parent_definitions = json.loads(parent_layout.read_path('metric_definitions.json', category='configs').read_text())
    parent_loss_order = parent_definitions['training_loss_order']
    if definitions['training_loss_order'][:len(parent_loss_order)] != parent_loss_order:
        raise ValueError('Child changed the native loss accumulation order.')
    for name, definition in parent_definitions['training_column_definitions'].items():
        if name in definitions['training_column_definitions']:
            if definitions['training_column_definitions'][name] != definition:
                raise ValueError(f'Child reinterprets inherited scalar {name}.')
    score = source.get('best_field_score')
    if score is None:
        raise ValueError('The inherited snapshot needs its own native validation selector value.')
    receipt = {'kind': 'native_only_epoch_ten_common_prefix', 'path': str(source_path),
               'epoch': 10, 'inherited_epochs': 10,
               'history_scope': 'Ten native-only rows; added auxiliary columns are unmeasured.',
               'selectors': 'Inherit the selector value stored in the epoch-ten checkpoint.',
               'calibration': copy.deepcopy(calibration)}
    payload = copy.deepcopy(source)
    payload.update(experiment_identity=identity, provider_training_state=provider_state,
                   best_field_score=float(score), branch_parent=receipt,
                   pending_publications={'best_field': True, 'milestone': True})
    payload.pop('history', None)
    payload.pop('best_response_guarded_selection', None)
    # Auxiliary future columns remain blank; inherited totals retain their
    # original native meaning and exact numerical representations.
    rows = [{name: row.get(name, '') for name in columns} for row in parent_rows]
    after = source_path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('The common-prefix checkpoint changed while branching.')
    layout = RunLayout(output)
    _atomic_json(output / 'artifact_layout.json', {'artifact_layout_version': 1,
        'run_dir': str(output), 'training_output_policy': 'csv_v3'})
    _atomic_json(layout.write_path('resolved_config.json'), recipe)
    _atomic_json(layout.write_path('experiment_identity.json'), identity)
    _atomic_json(layout.write_path('metric_definitions.json', category='configs'), definitions)
    _atomic_json(layout.write_path('native_prefix_branch.json', category='configs'), receipt)
    append_scalar_rows(layout.write_path('history.csv'), columns, rows)
    append_scalar_rows(layout.write_path('validation.csv'), VALIDATION_COLUMNS, validation_rows)
    _atomic_torch_save(layout.write_path('latest_model.pt'), payload)
    for name in ('epoch_0010_model.pt', 'best_by_field_mse_model.pt'):
        os.link(layout.write_path('latest_model.pt'), layout.write_path(name))
    if render_curves:
        _render_loss_curves(read_training_history(layout.write_path('history.csv')),
            layout.write_path('loss_curves.png').parent, engine.selection.field_metric,
            provider_identity=provider_identity,
            metadata_output=layout.write_path('loss_curves_metadata.json').parent,
            separate_totals=True)
    return receipt
