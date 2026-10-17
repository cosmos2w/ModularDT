"""Round9 task construction reusing original providers and TrainingEngine."""
from __future__ import annotations
import copy
from dataclasses import replace
from pathlib import Path
import torch
from .joint_native_protocol import *
from .native_budgeted_factory import _configure_root, _loaded_source_binding
from .native_budgeted_runtime import _round8_provider_pair
from .joint_native_runtime import joint_core_from_parent, install_joint_optimizer_groups


def build_joint_native_task(recipe, *, device='cpu', runtime_root=None):
    value=validate_joint_native_recipe(recipe,verify_source=True)
    project=_configure_root(runtime_root)
    from content_hypergraph_train import build, engine_for
    from .native_budgeted_runtime import load_native_budgeted_checkpoint
    from .joint_native_runtime import load_joint_native_checkpoint
    from .unified_training import _sampling_dataset_id
    if value['source_epoch']==2500:
        source_model,adapter,payload=load_native_budgeted_checkpoint(value['source_checkpoint'],device=device)
    else:
        source_model,adapter,payload=load_joint_native_checkpoint(value['source_checkpoint'],device=device)
    # Build with the exact inherited provider recipe; the common microbatch only changes engine accumulation.
    fresh,provider=build(value['base_recipe'],device=str(device));del fresh
    parent_identity=payload['experiment_identity']
    _round8_provider_pair(parent_identity['provider_identity'],dict(provider.identity_payload()),value['dataset'])
    model=source_model
    if value['arm']!='D9-M': model=joint_core_from_parent(model,dataset=value['dataset'],recipe=value)
    provider.model=model;provider.adapter=model if value['dataset']=='thermal' else adapter
    if value['arm']!='D9-M': provider._allow_joint_native=True
    install_joint_optimizer_groups(provider,model,payload)
    from .joint_native_runtime import bind_joint_native_sampling_extension
    bind_joint_native_sampling_extension(provider,value)
    # Validation retains exactly the source case/receiver panel with smaller packing.
    # Packing is sealed in Round9 identity and does not rewrite historical provider metadata.
    from types import MethodType
    original_validation=provider.validation_batches
    validation_cache=[]
    def validation_batches(self):
        if not validation_cache:
            if value['dataset']=='wind':
                validation_cache.extend(self.validation_batches_for(self.VALIDATION_QUERY_COUNT,
                    microbatch_size=value['microbatch_cases']))
            else:
                previous=self.microbatch_size
                previous_cache=self._validation_batches_cache
                try:
                    self.microbatch_size=value['microbatch_cases'];self._validation_batches_cache=None
                    validation_cache.extend(original_validation())
                finally:
                    self.microbatch_size=previous;self._validation_batches_cache=previous_cache
        return iter(validation_cache)
    provider.validation_batches=MethodType(validation_batches,provider)
    if value['arm'] in ('H9-ME','S9-ME'):
        from .joint_native_review import JointNativeTrainReview
        provider.joint_native_review=JointNativeTrainReview()
        original_state=getattr(provider,'training_state_dict',None)
        original_load=getattr(provider,'load_training_state_dict',None)
        def training_state_dict(self):
            state={} if original_state is None else dict(original_state())
            state['joint_native_review']=self.joint_native_review.state_dict()
            return state
        def load_training_state_dict(self,state):
            inherited=dict(state)
            review_state=inherited.pop('joint_native_review',None)
            if original_load is not None:original_load(inherited)
            elif inherited:raise ValueError('Round9 provider has unrecognized inherited state.')
            if review_state is not None:self.joint_native_review.load_state_dict(review_state)
        provider.training_state_dict=MethodType(training_state_dict,provider)
        provider.load_training_state_dict=MethodType(load_training_state_dict,provider)
    provider_identity=dict(provider.identity_payload())
    _round8_provider_pair(parent_identity['provider_identity'],provider_identity,value['dataset'])
    engine=engine_for(value['base_recipe'],str(device))
    engine.config=replace(engine.config,microbatch_cases=value['microbatch_cases'],effective_cases=value['effective_cases'],
        total_epochs=5000,checkpoint_epochs=tuple(sorted(set(range(2600,5001,100))|{2750})),latest_every=100,
        curve_every=10,progress_enabled=True,monitor_epochs=tuple(sorted(set(range(2600,5001,100))|{2750})))
    original=parent_identity['engine_config']
    if (engine.config.seed!=original['seed'] or engine.config.sampling_version!=original['sampling_version']
            or not engine.config.deterministic_case_order):
        raise ValueError('Round9 changed original sampler seed/version/order.')
    sampling_id=_sampling_dataset_id(provider_identity)
    if sampling_id!=parent_identity['sampling_dataset_id']:
        raise ValueError('Round9 changed original case/query namespace.')
    identity=dict(family=JOINT_NATIVE_FAMILY,model_family=JOINT_NATIVE_FAMILY,workflow=JOINT_NATIVE_WORKFLOW,
        protocol=JOINT_NATIVE_PROTOCOL,treatment=value['arm'],recipe=copy.deepcopy(value),recipe_sha256=canonical_sha256(value),
        provider_identity=provider_identity,sampling_identity=dict(sampling_version='case_epoch_v1',
        source_sampling_dataset_id=sampling_id,arm='A2-D',fit_epoch_is_absolute=True,sampler_epoch_mapping='identity'),
        loaded_source=_loaded_source_binding(project), training_scope=dict(native_objective_unchanged=True,
        sampling_horizon_extension=dict(original_horizon=2500,extended_horizon=5000,namespace_unchanged=True),
        absolute_start_epoch=value['source_epoch']+1,final_epoch=value['final_epoch'],
        shared_microbatch_cases=value['microbatch_cases'],validation_microbatch_cases=value['microbatch_cases'],effective_cases=value['effective_cases'],
        direct_environment_value_reader=value['arm']!='D9-M'))
    return dict(model=model,provider=provider,engine=engine,identity=identity,provider_identity=provider_identity,
        source_checkpoint=value['source_checkpoint'],source_payload=payload,joint_native_child=joint_native_child_declaration(value))
