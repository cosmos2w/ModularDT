"""Explicit R16-7 prescribed-domain fork of the P64 source."""
from __future__ import annotations
import copy
import json
import os
import numpy as np
import torch
from .r16_6_balanced_task import (ThermalR166BalancedTask, make_balanced_recipe)
from .r16_5_warm_fork import *
from .r16_5_warm_fork import (_read_identity, _build_thermal_member_routing_task, _VARIANTS,
    _read_selected_cases, _restore_response_scales, _sampling_dataset_id)
from ..data.datasets import H5Normalizer
from ..member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from honf_runtime.compat import load_trusted_checkpoint

COEFFICIENT = 0.0014978225391285025

def make_prescribed_recipe(p64, h0, *, arm, name, output_root, membership,
                           local_horizon, first_segment_epochs, wall_clock_budget_seconds=86400,
                           clean_save_reserve_seconds=600):
    if arm not in ('P0', 'P1'):
        raise ValueError('Expected P0 or P1.')
    if membership == 'full_train600':
        if (local_horizon, first_segment_epochs) != (500, 100):
            raise ValueError('Formal R16-7 requires a 500 epoch horizon and 100 epoch first review.')
        base_horizon, base_first = 100, 100
    elif membership == 'fixed25_train150_dev22':
        if type(local_horizon) is not int or not 1 <= first_segment_epochs <= local_horizon <= 30:
            raise ValueError('Qualification must declare a horizon within 30 epochs.')
        base_horizon, base_first = local_horizon, first_segment_epochs
    else:
        raise ValueError('Unknown R16-7 membership.')
    recipe = make_balanced_recipe(p64, h0, arm='C', coefficient=COEFFICIENT, name=name,
        output_root=output_root, membership=membership, local_horizon=base_horizon,
        first_segment_epochs=base_first, wall_clock_budget_seconds=wall_clock_budget_seconds,
        clean_save_reserve_seconds=clean_save_reserve_seconds)
    recipe.pop('r16_6_balanced')
    recipe['r16_7_prescribed'] = {'arm': arm, 'coefficient': COEFFICIENT, 'source_age': SOURCE_AGE,
        'action_placement': 'floor(a*N/16)', 'action_weight': 'N/16'}
    recipe['member_routing']['model']['base_model_config']['environment_input_dim'] = 342
    recipe['training'].update(local_horizon=local_horizon, first_segment_epochs=first_segment_epochs,
        checkpoint_epochs=list(range(100, 501, 100)) if membership == 'full_train600' else [local_horizon],
        monitor_every=100)
    return recipe

def validate_prescribed_recipe(source, recipe, p64, h0):
    spec, t = recipe['r16_7_prescribed'], recipe['training']
    expected = make_prescribed_recipe(p64, h0, arm=spec['arm'], name=recipe['name'],
        output_root=recipe['storage']['output_root'], membership=t['membership'],
        local_horizon=t['local_horizon'], first_segment_epochs=t['first_segment_epochs'],
        wall_clock_budget_seconds=t['wall_clock_budget_seconds'], clean_save_reserve_seconds=t['clean_save_reserve_seconds'])
    if recipe != expected:
        raise ValueError('R16-7 recipe differs from the declared prescribed-domain source recipe.')
    if source is not None and (source.get('epoch') != 100 or source.get('current_epoch') != 100
        or source.get('arm') != ARM_DETAIL or [r.get('epoch') for r in source.get('history', [])] != list(range(1, 101))
        or json.dumps(source.get('experiment_identity'), sort_keys=True) != json.dumps(p64, sort_keys=True)):
        raise ValueError('Expected literal completed P64 source.')
    return validate_source_identities(p64, h0)

def initialize_prescribed_weights(model, source_state):
    target = model.state_dict()
    if set(target) != set(source_state):
        raise ValueError('Unexpected source tensor inventory.')
    names = [name for name in target if name.endswith('environment_encoder.0.weight')]
    if len(names) != 1:
        raise ValueError('Expected one E encoder input weight.')
    key = names[0]
    mapped = {}
    for name, value in source_state.items():
        dst = target[name]
        if value.dtype != dst.dtype:
            raise ValueError(f'Source tensor dtype mismatch: {name}')
        if name == key:
            if value.shape != (128, 22) or dst.shape != (128, 358):
                raise ValueError('Unexpected E encoder dimensions.')
            result = torch.zeros_like(dst)
            result[:, :6] = value[:, :6]
            result[:, 342:] = value[:, 6:]
            mapped[name] = result
        else:
            if value.shape != dst.shape:
                raise ValueError(f'Source tensor mismatch: {name}')
            mapped[name] = value
    model.load_state_dict(mapped, strict=True)
    return {'inherited_names': sorted(source_state), 'new_names': [], 'expanded_name': key}

def build_thermal_r16_7_task(recipe, *, arm=None, device='cpu', resume_payload=None,
                                       response_scales=None):
    """Build a literal P64 fork, or reconstruct an existing fork from its own state."""
    configure_r16_5_execution(recipe, device=device)

    p64, h0 = _read_identity(P64_IDENTITY), _read_identity(H0_IDENTITY)
    source = None if resume_payload is not None else load_trusted_checkpoint(SOURCE_CHECKPOINT, map_location='cpu')
    references = validate_prescribed_recipe(source, recipe, p64, h0)
    if arm is not None and arm != ARM_DETAIL:
        raise ValueError('R16-7 must retain the literal P64 model arm.')
    if response_scales is not None and _restore_response_scales(response_scales) != references['response_scales']:
        raise ValueError('Saved run response scales differ from original H0 FIT scales.')
    if resume_payload is not None:
        saved_identity = resume_payload.get('experiment_identity', {})
        if (saved_identity.get('recipe') != recipe or resume_payload.get('arm') != ARM_DETAIL
                or resume_payload.get('epoch', 0) < 1):
            raise ValueError('Resume requires this fork\'s own saved recipe, arm and local checkpoint.')
        saved_provider = saved_identity.get('provider_identity', {})
        if (saved_provider.get('adapter_state', {}).get('prescribed_patch_mode') != {'P0': 'scaffold', 'P1': 'physical'}[recipe['r16_7_prescribed']['arm']]
                or saved_provider.get('response_scales') != references['response_scales']
                or any(saved_provider.get('adapter_state', {}).get(k) != references['p64_provider']['adapter_state'].get(k)
                       for k in ('normalization_stats', 'heat_scale', 'native_grid_nx_ny', 'environment_grid_nx_ny'))):
            raise ValueError('Resume checkpoint changed the inherited P64 adapter or original FIT scales.')
    source_adapter = references['p64_provider']['adapter_state']
    grid = source_adapter['environment_grid_nx_ny']

    mode = {'P0': 'scaffold', 'P1': 'physical'}[recipe['r16_7_prescribed']['arm']]
    class GridAdapter(ThermalDetailAnchoredMemberRoutingAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, 'environment_nx': grid[0], 'environment_ny': grid[1], 'prescribed_patch_mode': mode})

    model, provider = _build_thermal_member_routing_task(
        recipe, arm=ARM_DETAIL, device=device, task_type=ThermalR166BalancedTask,
        variants={DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], 'adapter': GridAdapter, 'arm': ARM_DETAIL}},
        section_schema=DETAIL_CONFIG_SCHEMA, source_kind=SOURCE_KIND, lineage='R16-7 P64 weight-only fork',
        required_address_mode=None, saved_adapter_state=source_adapter)
    adapter = model.state_payload()
    for key in ('normalization_stats', 'heat_scale', 'native_grid_nx_ny', 'environment_grid_nx_ny'):
        if adapter[key] != source_adapter[key]:
            raise ValueError(f'Fork failed to preserve P64 adapter {key}.')
    for key in ('dataset_path', 'dataset_size_bytes', 'dataset_mtime_ns', 'catalog_metadata_sha256', 'manifest_sha256'):
        if provider.source_binding[key] != references['p64_provider']['source_binding'][key]:
            raise ValueError(f'Fork data binding differs from P64 {key}.')
    source_ids = references['p64_provider']['training_case_ids']
    if (recipe['training']['membership'] == 'full_train600' and list(provider.training_case_ids) != source_ids
            or recipe['training']['membership'] == 'fixed25_train150_dev22'
            and (len(provider.training_case_ids) != 150 or not set(provider.training_case_ids) <= set(source_ids))):
        raise ValueError('Fork TRAIN membership differs from its declared source population.')
    val_ids = tuple(case_id for case_id in provider.catalog_case_ids
                    if provider.case_split_by_id[case_id] == 'test' and case_id != '0273')
    if list(val_ids) != references['p64_provider']['validation_case_ids']:
        raise ValueError('Fork validation membership differs from P64 VAL89.')
    normalizer = H5Normalizer({key: np.asarray(value, dtype=np.float32)
                               for key, value in source_adapter['normalization_stats'].items()})
    provider.validation_cases = _read_selected_cases(provider.data_path, val_ids, split='test', normalizer=normalizer,
                                                      include_grid=False, include_raster_geometry=False, include_prescribed_patch=True)
    provider.validation_case_ids = val_ids
    provider._validation_index_by_id = {value: index for index, value in enumerate(val_ids)}
    provider.validation_scope = 'canonical89_exposed_original_test_excluding_0273'
    provider._validation_batches_cache = None
    provider.configure_port_response(recipe)
    provider.source_parent_age = SOURCE_AGE
    provider.configure_state_response(recipe['thermal_state_response'], response_scales=references['response_scales'])
    provider.configure_balanced(recipe['r16_7_prescribed'])
    if recipe['execution'].get('deterministic_algorithms') is True:
        provider.r16_5_execution = {
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    if resume_payload is None:
        provider.r16_5_source_tensor_inventory = initialize_prescribed_weights(model, source['model_state_dict'])
    else:
        model.load_state_dict(resume_payload['model_state_dict'], strict=True)
        provider.r16_5_source_tensor_inventory = copy.deepcopy(saved_provider['r16_6_source_tensor_inventory'])
    if (list(provider.response_corpus.family_ids('fit')) != references['h0_provider']['response_training_family_ids']
            or _sampling_dataset_id(provider.identity_payload()) != p64['sampling_dataset_id']):
        raise ValueError('Fork changed original FIT48 or native sampling namespace.')
    return model, provider
