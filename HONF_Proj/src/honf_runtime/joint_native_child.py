"""Round9 by-name model/Adam inheritance and exact stream boundaries."""
from __future__ import annotations
import copy
from collections.abc import Mapping
from pathlib import Path
import torch
from .joint_native_protocol import (canonical_sha256, file_sha256, validate_joint_native_recipe,
    validate_joint_native_stop, joint_native_child_declaration)


def prepare_joint_native_child(model, provider, source_payload, sealed_identity, declaration, *, source_path, stop_after):
    recipe = validate_joint_native_recipe(sealed_identity['recipe'], verify_source=True)
    validate_joint_native_stop(recipe,stop_after)
    if dict(declaration) != joint_native_child_declaration(recipe):
        raise ValueError('Round9 child declaration differs from its sealed recipe.')
    if str(Path(source_path).resolve()) != recipe['source_checkpoint'] or file_sha256(source_path) != recipe['source_checkpoint_sha256']:
        raise ValueError('Round9 branch source differs from pinned literal boundary.')
    old = source_payload['model_state_dict']; current = model.state_dict()
    if set(old)-set(current):
        raise ValueError('Round9 dropped original model state names.')
    with torch.no_grad():
        for name,saved in old.items():
            target=current[name]
            if torch.is_tensor(saved):
                if not torch.is_tensor(target) or target.shape != saved.shape or target.dtype != saved.dtype:
                    raise ValueError(f'Round9 model state changed shape/dtype at {name}.')
                target.copy_(saved.to(target.device))
            elif target != saved:
                raise ValueError(f'Round9 parent extra state differs at {name}.')
    for name,saved in old.items():
        actual=model.state_dict()[name]
        same=torch.equal(actual.detach().cpu(),saved.detach().cpu()) if torch.is_tensor(saved) else actual==saved
        if not same:raise ValueError('Round9 parent model inheritance failed.')
    source_names = {name for group in source_payload['optimizer_state_by_name']['groups'] for name in group['parameter_names']}
    new_names = set(dict(model.named_parameters()))-source_names
    if source_names-set(dict(model.named_parameters())):
        raise ValueError('Round9 dropped a named Adam parameter.')
    if recipe['arm']=='D9-M' and new_names:
        raise ValueError('D9-M may not add parameters.')
    from .unified_training import _sampling_dataset_id
    source_identity=source_payload['experiment_identity']; current_identity=sealed_identity
    if (source_identity.get('sampling_dataset_id') != _sampling_dataset_id(current_identity['provider_identity'])
            or source_identity['engine_config']['seed'] != current_identity['engine_config']['seed']
            or source_identity['engine_config']['effective_cases'] != current_identity['engine_config']['effective_cases']):
        raise ValueError('Round9 changed native sampler namespace, seed, or effective batch.')
    from .native_budgeted_runtime import _round8_provider_pair
    _round8_provider_pair(source_identity['provider_identity'],current_identity['provider_identity'],recipe['dataset'])
    return {**dict(declaration),'kind':'joint_native_child','added_state_keys':sorted(set(current)-set(old)),
        'new_parameter_names':sorted(new_names),'exact_source_model_state_by_name':True,
        'schedule_extension':recipe['schedule_extension'],'shared_microbatch_cases':recipe['microbatch_cases']}


def verify_joint_native_optimizer(model,optimizer,source,amendment):
    from .unified_training import named_optimizer_state
    current=named_optimizer_state(model,optimizer)['state_by_name']
    for name,state in source['state_by_name'].items():
        if name not in current or set(current[name]) != set(state):
            raise ValueError(f'Round9 lost Adam state at {name}.')
        for key,expected in state.items():
            actual=current[name][key]
            same=(torch.equal(actual.detach().cpu(),expected.detach().cpu()) if torch.is_tensor(expected) else actual==expected)
            if not same: raise ValueError(f'Round9 changed Adam state at {name}.{key}.')
    if any(name in current for name in amendment.get('new_parameter_names',())):
        raise ValueError('New Round9 parameters must begin without Adam moments.')


def verify_joint_native_resume(payload, *, recipe, stop_after):
    validate_joint_native_stop(recipe,stop_after)
    identity=payload.get('experiment_identity',{})
    if identity.get('recipe_sha256') != canonical_sha256(recipe) or payload.get('arm') != 'A2-D':
        raise ValueError('Round9 exact resume recipe/stream differs.')
    if not recipe['source_epoch'] < payload.get('epoch',-1) < stop_after:
        raise ValueError('Round9 resume requires forward progress from own completed epoch.')
