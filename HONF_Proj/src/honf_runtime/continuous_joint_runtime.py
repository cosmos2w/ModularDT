"""Continuous native binding and standalone data-free checkpoint reconstruction."""
from __future__ import annotations
import copy
from pathlib import Path
from .continuous_joint_protocol import *


def continuous_core_from_parent(parent_model,*,dataset,recipe):
    from honf_forward_core.continuous_joint.model import ContinuousJointCore
    parent=getattr(parent_model,'core',parent_model)
    unified=recipe['arm'].startswith('U10-')
    if unified:
        from honf_forward_core.continuous_joint.unified import UnifiedJointCore
        core=UnifiedJointCore.from_parent(parent,input_dataset=dataset,
            input_contract_sha256=recipe['input_feature_contract']['sha256'],
            routing_mode={'U10-D':'dense','U10-H':'hypergraph','U10-S':'direct'}[recipe['arm']],**recipe['model_options'])
    else:
        core=ContinuousJointCore.from_parent(parent,routing_mode={'C10-H':'hypergraph','C10-S':'direct'}[recipe['arm']],**recipe['model_options'])
    if unified and recipe.get('context_initialization') is not None:
        import json
        binding=recipe['context_initialization']
        if file_sha256(binding['path'])!=binding['sha256']:raise ValueError('Shared context initialization receipt changed.')
        initialized=json.loads(Path(binding['path']).read_text())
        expected=dict(schema='round10_continuous_calibration_v1',dataset=dataset,arm='U10-H',
            source_checkpoint_sha256=recipe['source_checkpoint_sha256'],optimizer_updates=0,
            input_feature_contract_sha256=recipe['input_feature_contract']['sha256'])
        if any(initialized.get(key)!=value for key,value in expected.items()):
            raise ValueError('U-S contextual initialization is not its matched input-only U-H source.')
        import torch
        with torch.no_grad():
            for name,parameter in core.named_parameters():
                if name.startswith('router.incidence.') and name.endswith('.bias'):
                    parameter.fill_(initialized['biases'][name])
    if recipe['calibration'] is not None:
        from .continuous_joint_calibration import apply_continuous_calibration
        apply_continuous_calibration(core,recipe)
    if dataset=='thermal':parent_model.core=core;return parent_model
    return core


def bind_continuous_sampling_extension(provider,recipe):
    from .joint_native_runtime import JointNativeSamplingExtension
    from .unified_training import _sampling_dataset_id
    value=validate_continuous_joint_recipe(recipe,verify_source=True)
    extension=JointNativeSamplingExtension(provider,value['dataset'],_sampling_dataset_id(provider.identity_payload()),2500,5000,canonical_sha256(value))
    provider._joint_native_sampling_extension=extension
    return extension


def load_continuous_joint_checkpoint(path,*,device='cpu'):
    from .compat import load_trusted_checkpoint
    from honf_forward_core.continuous_joint.model import ContinuousJointCore
    payload=load_trusted_checkpoint(path,map_location='cpu');identity=payload.get('experiment_identity',{})
    if identity.get('family') not in (CONTINUOUS_FAMILY,UNIFIED_FAMILY) or identity.get('workflow')!=CONTINUOUS_WORKFLOW:
        raise ValueError('Checkpoint is not a continuous native member.')
    recipe=validate_continuous_joint_recipe(identity.get('recipe',{}))
    if identity.get('recipe_sha256')!=canonical_sha256(recipe) or payload.get('arm')!='A2-D' or not 3000<=payload.get('epoch',-1)<=5000:
        raise ValueError('Continuous embedded identity/clock differs.')
    provider=identity['provider_identity'];state=payload['model_state_dict']
    if recipe['dataset']=='thermal':
        from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
        model=ThermalContentHypergraphAdapterV2.from_state_payload(provider['adapter_state'],device=device,state_dict=state);adapter=model
    else:
        from windfarm.content_hypergraph import WindContentHypergraphAdapterV3
        config=provider['model_config']
        if config.get('schema')==UNIFIED_FAMILY:
            from honf_forward_core.continuous_joint.unified import UnifiedJointCore
            model=UnifiedJointCore.from_config(config).to(device)
        else:model=ContinuousJointCore.from_config(config).to(device)
        model.load_state_dict(state,strict=True)
        adapter=WindContentHypergraphAdapterV3.from_state_payload(provider['adapter_state'],device=device)
    if model.get_config()!=provider['model_config']:raise ValueError('Continuous inference constructor differs.')
    core=getattr(model,'core',model)
    if int(core.joint_epoch)!=payload['epoch']:raise ValueError('Continuous literal routing clock differs.')
    model.eval();return model,adapter,payload


def export_continuous_joint_checkpoint(source,target):
    from .compat import load_trusted_checkpoint
    from .unified_training import _atomic_torch_save
    load_continuous_joint_checkpoint(source)
    payload=load_trusted_checkpoint(source,map_location='cpu')
    value={key:copy.deepcopy(payload[key]) for key in ('workflow','epoch','arm','experiment_identity','model_state_dict')}
    value['inference_export']=dict(schema='round10_continuous_standalone_v1',source_sha256=file_sha256(source))
    target=Path(target).resolve()
    if target.exists():raise FileExistsError(target)
    _atomic_torch_save(target,value);load_continuous_joint_checkpoint(target)
    return target


def load_continuous_evaluation_components(checkpoint,device='cpu',runtime_root=None):
    """Reopen the saved native provider; no ancestor checkpoint is consulted."""
    from .native_budgeted_factory import _configure_root
    _configure_root(runtime_root)
    from content_hypergraph_train import build
    model,adapter,payload=load_continuous_joint_checkpoint(checkpoint,device=device)
    identity=payload['experiment_identity'];recipe=validate_continuous_joint_recipe(identity['recipe'])
    unused,provider=build_continuous_native_provider(recipe,identity['provider_identity'],device=device);del unused
    provider.model=model;provider.adapter=adapter;provider._allow_continuous_joint=True
    # Provider identities include constructor and TRAIN-fitted native scales.
    _, aliases = reconcile_evaluation_path_aliases(
        dict(provider.identity_payload()), identity['provider_identity'])
    provider.evaluation_path_aliases = tuple(aliases)
    return model,provider,payload


def reconcile_evaluation_path_aliases(actual, expected):
    """Accept relocated path labels only for the same exact physical inputs.

    This changes evaluation identity comparison, never a saved checkpoint or
    transform. Callers retain the aliases in provenance; all other identity
    fields must match. Equal contents in a different file are insufficient.
    """
    actual = copy.deepcopy(actual)
    aliases = []
    old_rows = expected.get('response_family_sources', [])
    new_rows = actual.get('response_family_sources', [])
    if len(old_rows) != len(new_rows):
        raise ValueError('Continuous evaluation response-family membership differs.')
    for row, old in zip(new_rows, old_rows):
        if {key: value for key, value in row.items() if key != 'path'} != {
                key: value for key, value in old.items() if key != 'path'}:
            raise ValueError('Continuous evaluation response-family identity differs.')
        if row.get('path') != old.get('path'):
            current, inherited = Path(row['path']), Path(old['path'])
            if not current.is_file() or not current.samefile(inherited):
                raise ValueError('Relocated response reference is not the same physical file.')
            if file_sha256(current) != old.get('sha256'):
                raise ValueError('Relocated response-reference bytes differ from the saved hash.')
            aliases.append(dict(kind='response_family_source', family_id=old['family_id'],
                actual_path=str(current), inherited_path=str(inherited), sha256=old['sha256']))
            row['path'] = old['path']
    old_cache = expected.get('native_role_catalogue_cache', {})
    new_cache = actual.get('native_role_catalogue_cache', {})
    old_directory = old_cache.get('persistent_directory')
    new_directory = new_cache.get('persistent_directory')
    if old_directory != new_directory and old_directory and new_directory:
        current, inherited = Path(new_directory), Path(old_directory)
        if not current.is_dir() or not current.samefile(inherited):
            raise ValueError('Relocated native catalogue is not the same physical directory.')
        aliases.append(dict(kind='native_role_catalogue_cache', actual_path=str(current),
                            inherited_path=str(inherited), same_directory_inode=True))
        new_cache['persistent_directory'] = old_directory
    if canonical_sha256(actual) != canonical_sha256(expected):
        raise ValueError('Continuous evaluation provider differs from saved native data/transforms.')
    return actual, aliases


def build_continuous_native_provider(recipe,provider_identity,*,device):
    """Reuse the parent's exact cache identity across source worktrees."""
    import os
    from content_hypergraph_train import build
    name='HONF_WIND_CATALOGUE_CACHE_DIR'
    previous=os.environ.get(name)
    cache=provider_identity.get('native_role_catalogue_cache',{}).get('persistent_directory')
    if recipe['dataset']=='wind' and cache is not None:
        if not Path(cache).is_dir():raise FileNotFoundError(f'Inherited shared Wind cache is missing: {cache}')
        os.environ[name]=cache
    try:return build(recipe['base_recipe'],device=str(device))
    finally:
        if previous is None:os.environ.pop(name,None)
        else:os.environ[name]=previous
