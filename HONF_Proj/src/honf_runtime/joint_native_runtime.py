"""Round9 provider capability, named groups, and ancestor-free inference loading."""
from __future__ import annotations
import copy
from collections.abc import Mapping
from dataclasses import replace
from types import MethodType
from pathlib import Path
import torch
from .joint_native_protocol import *


def joint_core_from_parent(parent_model, *, dataset, recipe):
    from honf_forward_core.joint_native.model import JointNativeCore
    parent=getattr(parent_model,'core',parent_model)
    arm=recipe['arm']
    mode={'D9-ME-warm':'dense','D9-ME':'dense','H9-ME':'joint','S9-ME':'direct'}[arm]
    opts=dict(recipe['model_options']);opts['routing_mode']=mode
    if hasattr(parent,'get_config') and parent.get_config().get('schema')=='joint_native_hypergraph_v1':
        config=parent.get_config();config['routing_mode']=mode;config.update(opts)
        core=JointNativeCore.from_config(config)
        result=core.load_state_dict(parent.state_dict(),strict=False)
        allowed={name for name in core.state_dict() if name.startswith(('router.','joint_budget.'))}
        if result.unexpected_keys or not set(result.missing_keys).issubset(allowed):
            raise ValueError('Round9 fork may add only fresh router and typed budget state.')
    else:
        core=JointNativeCore.from_a2d(parent,**opts)
    if recipe.get('calibration') is not None:
        from .joint_native_calibration import apply_joint_native_calibration
        apply_joint_native_calibration(core,recipe)
    if dataset=='thermal':
        parent_model.core=core.to(next(parent.parameters()).device)
        return parent_model
    return core.to(next(parent.parameters()).device)


def install_joint_optimizer_groups(provider, model, source_payload):
    """Freeze original names/membership, append separate E and router groups."""
    from .unified_training import OptimizerGroupSpec, ScheduleSpec
    old=copy.deepcopy(source_payload['optimizer_schedule_contract'])
    names=tuple(dict(model.named_parameters()))
    inherited={name for group in old for name in group['parameter_names']}
    if inherited-set(names): raise ValueError('Round9 model lost original optimizer names.')
    fresh=[name for name in names if name not in inherited]
    e_names=tuple(name for name in fresh if 'router' not in name and 'budget' not in name)
    r_names=tuple(name for name in fresh if name not in e_names)
    groups=[]
    for group in old:
        config=copy.deepcopy(group['schedule'])
        schedule=ScheduleSpec(**config)
        terminal=schedule.value(2500)
        # Continue the verified terminal policy; extending cosine would rewrite past ages.
        if schedule.value(2501) != terminal:
            raise ValueError('Source terminal learning-rate policy is not constant beyond e2500.')
        config['total_epochs']=5000
        if config['peak_lr'] != config['final_lr']:
            # Constant terminal continuation while retaining the explicit parent policy in lineage.
            config=dict(peak_lr=terminal,warmup_start_lr=terminal,warmup_epochs=0,
                        hold_through_epoch=4999,total_epochs=5000,final_lr=terminal)
        groups.append(OptimizerGroupSpec(group['name'],tuple(group['parameter_names']),ScheduleSpec(**config),
            weight_decay=group['weight_decay'],betas=tuple(group['betas']),eps=group['eps']))
    source=groups[0]
    for label,added,decay in [('joint_native_environment',e_names,source.weight_decay),('joint_native_router',r_names,0.)]:
        if added: groups.append(OptimizerGroupSpec(label,added,source.schedule,weight_decay=decay,betas=source.betas,eps=source.eps))
    def optimizer_groups(self,candidate,arm,stage):
        if candidate is not model or arm!='A2-D' or stage!='joint':
            raise ValueError('Round9 optimizer groups bind one exact model/native arm.')
        return tuple(groups)
    provider.optimizer_groups=MethodType(optimizer_groups,provider)
    return tuple(groups)


def load_joint_native_checkpoint(path, *, device='cpu'):
    """Reconstruct inference exclusively from embedded config, normalizer and tensor bytes."""
    from .compat import load_trusted_checkpoint
    from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
    from honf_forward_core.joint_native.model import JointNativeCore
    payload=load_trusted_checkpoint(path,map_location='cpu')
    identity=payload.get('experiment_identity',{})
    if identity.get('family') != JOINT_NATIVE_FAMILY or identity.get('workflow') != JOINT_NATIVE_WORKFLOW:
        raise ValueError('Checkpoint is not a Round9 joint native member.')
    recipe=validate_joint_native_recipe(identity.get('recipe',{}))
    if identity.get('recipe_sha256') != canonical_sha256(recipe):
        raise ValueError('Round9 embedded recipe seal differs.')
    if (payload.get('workflow')!=JOINT_NATIVE_WORKFLOW or payload.get('arm')!='A2-D'
            or type(payload.get('epoch')) is not int
            or not recipe['source_epoch']<=payload['epoch']<=recipe['final_epoch']):
        raise ValueError('Round9 checkpoint workflow, native arm or literal stage age differs.')
    provider=identity['provider_identity'];state=payload['model_state_dict']
    if recipe['dataset']=='thermal':
        from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
        model=ThermalContentHypergraphAdapterV2.from_state_payload(provider['adapter_state'],device=device,state_dict=state)
        adapter=model
    else:
        from windfarm.content_hypergraph import WindContentHypergraphAdapterV3
        config=provider['model_config']
        cls=JointNativeCore if config.get('schema')=='joint_native_hypergraph_v1' else ContentHypergraphCoreV2
        model=cls.from_config(config).to(device);model.load_state_dict(state,strict=True)
        adapter=WindContentHypergraphAdapterV3.from_state_payload(provider['adapter_state'],device=device)
    if model.get_config()!=provider['model_config']:
        raise ValueError('Round9 embedded inference constructor differs.')
    core=getattr(model,'core',model)
    if hasattr(core,'joint_epoch') and int(core.joint_epoch)!=payload['epoch']:
        raise ValueError('Round9 saved routing clock differs from the literal checkpoint age.')
    model.eval()
    if hasattr(adapter,'eval'): adapter.eval()
    return model,adapter,payload


def export_joint_native_checkpoint(source, target):
    from .compat import load_trusted_checkpoint
    from .unified_training import _atomic_torch_save
    # No external parent, admission or dataset file is read by the loader.
    payload=load_trusted_checkpoint(source,map_location='cpu')
    load_joint_native_checkpoint(source,device='cpu')
    exported={key:copy.deepcopy(payload[key]) for key in ('workflow','epoch','arm','experiment_identity','model_state_dict')}
    exported['inference_export']={'schema':'round9_standalone_inference_v1','source_sha256':file_sha256(source)}
    target=Path(target).resolve()
    if target.exists(): raise FileExistsError(target)
    _atomic_torch_save(target,exported)
    load_joint_native_checkpoint(target,device='cpu')
    return target


from dataclasses import dataclass

@dataclass(frozen=True)
class JointNativeSamplingExtension:
    """Provider ownership and a separately sealed sampling horizon extension."""
    provider: object
    dataset: str
    sampling_dataset_id: str
    original_horizon: int
    final_horizon: int
    recipe_sha256: str


def bind_joint_native_sampling_extension(provider, recipe):
    value=validate_joint_native_recipe(recipe,verify_source=True)
    from .unified_training import _sampling_dataset_id
    namespace=_sampling_dataset_id(provider.identity_payload())
    extension=JointNativeSamplingExtension(provider,value['dataset'],namespace,2500,5000,canonical_sha256(value))
    provider._joint_native_sampling_extension=extension
    return extension


def joint_native_sampling_epoch_limit(provider, extension):
    if (not isinstance(extension,JointNativeSamplingExtension) or extension.provider is not provider
            or extension.dataset!='thermal' or extension.original_horizon!=provider.total_epochs
            or extension.original_horizon!=2500 or extension.final_horizon!=5000
            or extension.sampling_dataset_id!=provider.sampling_dataset_id):
        raise ValueError('Round9 sampling extension has stale ownership, membership or horizon.')
    return extension.final_horizon
