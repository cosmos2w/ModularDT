"""Round9 stage, ancestry, sampler and resource contracts (independent of Round8)."""
from __future__ import annotations
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from .native_budgeted_protocol import canonical_sha256, file_sha256

JOINT_NATIVE_FAMILY = 'joint_native_hypergraph_v1'
JOINT_NATIVE_WORKFLOW = 'joint_native_train_v1'
JOINT_NATIVE_PROTOCOL = 'round9_joint_native_v1'
JOINT_NATIVE_RECIPE_SCHEMA = 'round9_joint_native_recipe_v1'
JOINT_NATIVE_CHILD_SCHEMA = 'round9_joint_native_child_v1'
JOINT_NATIVE_ARMS = ('D9-M', 'D9-ME-warm', 'D9-ME', 'H9-ME', 'S9-ME')
STAGES = {'D9-M': (2500, 5000), 'D9-ME-warm': (2500, 3000),
          'D9-ME': (3000, 5000), 'H9-ME': (3000, 5000), 'S9-ME': (3000, 5000)}
GPU_UUID = 'GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39'


def _source(path):
    from .compat import load_trusted_checkpoint
    return load_trusted_checkpoint(path, map_location='cpu')


def validate_source(payload, *, arm, dataset):
    start, _ = STAGES[arm]
    identity = payload.get('experiment_identity', {})
    recipe = identity.get('recipe', {})
    if payload.get('epoch') != start or payload.get('arm') != 'A2-D':
        raise ValueError('Round9 source must have the literal stage boundary and original A2-D stream.')
    if start == 2500:
        if (identity.get('family') != 'native_budgeted_dispatch_v2'
                or recipe.get('treatment') != 'D-control' or recipe.get('dataset') != dataset):
            raise ValueError('Round9 requires segmented D8 e2500, never G8 or a best alias.')
        base = recipe.get('base_recipe')
    else:
        if (identity.get('family') != JOINT_NATIVE_FAMILY
                or recipe.get('arm') != 'D9-ME-warm' or recipe.get('dataset') != dataset):
            raise ValueError('Round9 children require their common literal dense-ME e3000 warm-up.')
        base = recipe.get('base_recipe')
    if not isinstance(base, Mapping) or not isinstance(payload.get('optimizer_state_by_name'), Mapping):
        raise ValueError('Round9 parent needs native recipe and named Adam moments.')
    return copy.deepcopy(dict(base))


def make_joint_native_recipe(source_checkpoint, *, dataset, arm, microbatch_cases,
                             resource_reservation, model_options=None, calibration_receipt=None):
    path = Path(source_checkpoint).expanduser().resolve(strict=True)
    payload = _source(path)
    base = validate_source(payload, arm=arm, dataset=dataset)
    identity = payload['experiment_identity']
    value = dict(schema=JOINT_NATIVE_RECIPE_SCHEMA, family=JOINT_NATIVE_FAMILY,
        workflow=JOINT_NATIVE_WORKFLOW, protocol=JOINT_NATIVE_PROTOCOL, dataset=dataset,
        arm=arm, source_checkpoint=str(path), source_checkpoint_sha256=file_sha256(path),
        source_epoch=STAGES[arm][0], source_recipe_sha256=identity['recipe_sha256'],
        source_provider_identity_sha256=canonical_sha256(identity['provider_identity']),
        base_recipe=base, microbatch_cases=microbatch_cases, effective_cases=48 if dataset=='thermal' else 16,
        final_epoch=STAGES[arm][1], sampling_arm='A2-D', sampling_version='case_epoch_v1',
        model_options=dict(model_options or {}), calibration=None if calibration_receipt is None else {'path':str(Path(calibration_receipt).resolve(strict=True)),'sha256':file_sha256(calibration_receipt)}, resource_reservation=str(Path(resource_reservation).resolve()),
        schedule_extension='retain_exact_source_terminal_lr_through_e5000',
        execution=dict(checkpoint_every=100, curve_every=10, progress_enabled=True))
    return validate_joint_native_recipe(value, verify_source=True)


def validate_joint_native_recipe(value, *, verify_source=False):
    if not isinstance(value, Mapping):
        raise TypeError('Round9 recipe must be a mapping.')
    recipe = copy.deepcopy(dict(value))
    required = {'schema','family','workflow','protocol','dataset','arm','source_checkpoint',
        'source_checkpoint_sha256','source_epoch','source_recipe_sha256','source_provider_identity_sha256',
        'base_recipe','microbatch_cases','effective_cases','final_epoch','sampling_arm','sampling_version',
        'model_options','calibration','resource_reservation','schedule_extension','execution'}
    if set(recipe) != required:
        raise ValueError('Round9 recipe field inventory differs from the registered contract.')
    for key, expected in [('schema',JOINT_NATIVE_RECIPE_SCHEMA),('family',JOINT_NATIVE_FAMILY),
            ('workflow',JOINT_NATIVE_WORKFLOW),('protocol',JOINT_NATIVE_PROTOCOL),
            ('sampling_arm','A2-D'),('sampling_version','case_epoch_v1'),
            ('schedule_extension','retain_exact_source_terminal_lr_through_e5000')]:
        if recipe[key] != expected:
            raise ValueError(f'Invalid Round9 {key}.')
    if recipe['dataset'] not in ('thermal','wind') or recipe['arm'] not in STAGES:
        raise ValueError('Unknown Round9 dataset/arm.')
    if not isinstance(recipe['model_options'],Mapping):
        raise ValueError('Round9 value execution options must be a mapping.')
    if recipe['arm']=='D9-M' and recipe['model_options']:
        raise ValueError('D9-M retains original dense M implementation; no joint-reader options apply.')
    if set(recipe['model_options'])-{'native_backend','routing_seed','checkpoint_values','query_tile_size'}:
        raise ValueError('Round9 model options may configure execution/RNG, never override stage or value semantics.')
    start, final = STAGES[recipe['arm']]
    if recipe['source_epoch'] != start or recipe['final_epoch'] != final:
        raise ValueError('Round9 stage ages may not be relabelled.')
    micro = recipe['microbatch_cases']
    effective = 48 if recipe['dataset']=='thermal' else 16
    if type(micro) is not int or not 1 <= micro <= effective or recipe['effective_cases'] != effective:
        raise ValueError('Round9 microbatch invalid or effective native batch changed.')
    for key in ('source_checkpoint','resource_reservation'):
        if not Path(recipe[key]).is_absolute() or str(Path(recipe[key]).resolve()) != recipe[key]:
            raise ValueError(f'Round9 {key} must be canonical and absolute.')
    for key in ('source_checkpoint_sha256','source_recipe_sha256','source_provider_identity_sha256'):
        if not isinstance(recipe[key], str) or len(recipe[key]) != 64 or any(c not in '0123456789abcdef' for c in recipe[key]):
            raise ValueError(f'Invalid Round9 {key}.')
    calibration=recipe['calibration']
    if calibration is not None:
        if not isinstance(calibration,Mapping) or set(calibration)!={'path','sha256'}:
            raise ValueError('Round9 calibration must pin its exact path and bytes.')
        if verify_source and file_sha256(calibration['path'])!=calibration['sha256']:
            raise ValueError('Round9 TRAIN calibration receipt bytes changed.')
    if recipe['execution'] != dict(checkpoint_every=100,curve_every=10,progress_enabled=True):
        raise ValueError('Round9 output cadence changed.')
    if verify_source:
        if file_sha256(recipe['source_checkpoint']) != recipe['source_checkpoint_sha256']:
            raise ValueError('Round9 source bytes changed.')
        payload = _source(recipe['source_checkpoint'])
        base = validate_source(payload,arm=recipe['arm'],dataset=recipe['dataset'])
        identity = payload['experiment_identity']
        if recipe['source_epoch']==3000:
            warm=identity['recipe']
            if warm['microbatch_cases']!=recipe['microbatch_cases'] or warm['model_options']!=recipe['model_options']:
                raise ValueError('Round9 children must retain the shared warm-up microbatch and value execution settings.')
        if (base != recipe['base_recipe'] or identity['recipe_sha256'] != recipe['source_recipe_sha256']
                or canonical_sha256(identity['provider_identity']) != recipe['source_provider_identity_sha256']):
            raise ValueError('Round9 source recipe/provider seal changed.')
    return recipe


def joint_native_child_declaration(recipe):
    value = validate_joint_native_recipe(recipe)
    return {'schema':JOINT_NATIVE_CHILD_SCHEMA,'recipe_sha256':canonical_sha256(value),
            'source_checkpoint_sha256':value['source_checkpoint_sha256'], 'source_epoch':value['source_epoch'],
            'arm':value['arm'], 'final_epoch':value['final_epoch']}


def validate_joint_native_stop(recipe, stop_after):
    start, final = STAGES[recipe['arm']]
    if (type(stop_after) is not int or not start < stop_after <= final
            or (stop_after % 100 and stop_after not in (2750,3500,4500))):
        raise ValueError('Round9 stop must be a declared hundred-epoch or scientific review boundary.')
