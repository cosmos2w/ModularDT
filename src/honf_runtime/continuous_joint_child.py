"""Exact named common-state inheritance with an explicit unified summary drop."""
from __future__ import annotations
import copy
from pathlib import Path
import torch
from .continuous_joint_protocol import *


def _summary_name(name):
    from honf_forward_core.continuous_joint.unified import SUMMARY_ROOTS
    return name.removeprefix('core.').split('.')[0] in SUMMARY_ROOTS


def _allowed_dropped_names(source_names,current_names,recipe):
    missing=set(source_names)-set(current_names)
    allowed={name for name in missing if _summary_name(name)} if recipe['arm'].startswith('U10-') else set()
    if missing!=allowed:
        raise ValueError(f'Continuous child lost undeclared common state: {sorted(missing-allowed)}')
    return sorted(allowed)


def continuous_optimizer_source_payload(payload,model,recipe):
    """Filter only explicitly removed summary entries for group construction."""
    names=set(dict(model.named_parameters()))
    saved=copy.deepcopy(payload)
    inherited={name for group in saved['optimizer_schedule_contract'] for name in group['parameter_names']}
    dropped=set(_allowed_dropped_names(inherited,names,recipe))
    for group in saved['optimizer_schedule_contract']:
        group['parameter_names']=[name for name in group['parameter_names'] if name not in dropped]
    saved['optimizer_schedule_contract']=[group for group in saved['optimizer_schedule_contract'] if group['parameter_names']]
    return saved


def filtered_continuous_optimizer_seed(source,amendment):
    """The scientific parent remains untouched; only declared dropped names go."""
    dropped=set(amendment.get('dropped_optimizer_parameter_names',()))
    if any(not _summary_name(name) for name in dropped):
        raise ValueError('Continuous Adam drop inventory contains a nonsummary parameter.')
    value=copy.deepcopy(source)
    value['state_by_name']={name:state for name,state in value['state_by_name'].items() if name not in dropped}
    for group in value['groups']:
        group['parameter_names']=[name for name in group['parameter_names'] if name not in dropped]
    value['groups']=[group for group in value['groups'] if group['parameter_names']]
    return value


def prepare_continuous_joint_child(model,provider,source_payload,sealed_identity,declaration,*,source_path,stop_after):
    recipe=validate_continuous_joint_recipe(sealed_identity['recipe'],verify_source=True)
    validate_continuous_joint_stop(recipe,stop_after)
    if declaration!=continuous_joint_child_declaration(recipe) or str(Path(source_path).resolve())!=recipe['source_checkpoint']:
        raise ValueError('Continuous child differs from literal common boundary.')
    core=getattr(model,'core',model)
    expected_mode={'C10-H':'hypergraph','C10-S':'direct','U10-D':'dense','U10-H':'hypergraph','U10-S':'direct'}[recipe['arm']]
    config=core.get_config()
    if core.routing_mode!=expected_mode or config.get('schema')!=recipe['family']:
        raise ValueError('Continuous child model mode/family differs from its sealed arm.')
    if recipe['arm'].startswith('U10-') and (core.input_dataset!=recipe['dataset']
        or core.local_input_contract!=recipe['input_feature_contract'] or core.input_capability!=recipe['input_capability']):
        raise ValueError('Unified child model local-input capability differs from its sealed recipe.')
    options=dict(native_backend='packed',checkpoint_pipeline=True,query_tile_size=256,entmax_temperature=1.)
    options.update(recipe['model_options'])
    if recipe['calibration'] is not None:
        import json
        calibrated=json.loads(Path(recipe['calibration']['path']).read_text())
        options['entmax_temperature']=calibrated['temperature']['selected']
    if any(config.get(name)!=value for name,value in options.items()):
        raise ValueError('Continuous child execution/calibration settings differ from its sealed recipe.')
    old=source_payload['model_state_dict'];current=model.state_dict()
    dropped_state=_allowed_dropped_names(old,current,recipe)
    with torch.no_grad():
        for name,value in old.items():
            if name in dropped_state:continue
            if torch.is_tensor(value):
                if value.shape!=current[name].shape or value.dtype!=current[name].dtype:raise ValueError('Common state shape/dtype differs.')
                current[name].copy_(value.to(current[name].device))
            elif value!=current[name]:raise ValueError('Common extra-state differs.')
    source_names={name for group in source_payload['optimizer_state_by_name']['groups'] for name in group['parameter_names']}
    dropped_optimizer=_allowed_dropped_names(source_names,dict(model.named_parameters()),recipe)
    from .unified_training import _sampling_dataset_id
    source=source_payload['experiment_identity']
    if (source['sampling_dataset_id']!=_sampling_dataset_id(sealed_identity['provider_identity'])
        or source['engine_config']['seed']!=sealed_identity['engine_config']['seed']
        or source['engine_config']['effective_cases']!=sealed_identity['engine_config']['effective_cases']):
        raise ValueError('Continuous child changed native exposure or seed.')
    from .native_budgeted_runtime import _round8_provider_pair
    _round8_provider_pair(source['provider_identity'],sealed_identity['provider_identity'],recipe['dataset'])
    return {**declaration,'kind':'continuous_joint_child','added_state_keys':sorted(set(current)-set(old)),
        'new_parameter_names':sorted(set(dict(model.named_parameters()))-source_names),
        'dropped_summary_state_names':dropped_state,'dropped_optimizer_parameter_names':dropped_optimizer,
        'exact_compatible_source_model_state_by_name':True,'shared_microbatch_cases':8,
        'validated_model_config_sha256':canonical_sha256(config),
        'input_capability':recipe.get('input_capability','legacy_records'),
        'input_feature_contract':recipe.get('input_feature_contract')}


def verify_continuous_joint_optimizer(model,optimizer,source,amendment):
    from .joint_native_child import verify_joint_native_optimizer
    return verify_joint_native_optimizer(model,optimizer,filtered_continuous_optimizer_seed(source,amendment),amendment)


def verify_continuous_joint_resume(payload,*,recipe,stop_after):
    validate_continuous_joint_stop(recipe,stop_after)
    if (payload.get('experiment_identity',{}).get('recipe_sha256')!=canonical_sha256(recipe)
        or payload.get('arm')!='A2-D' or not 3000<payload.get('epoch',-1)<stop_after):
        raise ValueError('Continuous resume recipe/stream/age differs.')
