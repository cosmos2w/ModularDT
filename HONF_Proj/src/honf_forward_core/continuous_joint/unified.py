"""Unified preparation capability; independent from legacy input semantics."""
from __future__ import annotations

import torch
from torch import nn

SUMMARY_ROOTS = ("scene_summary_net", "module_scene_projection", "environment_scene_projection",
                 "module_scene_gate", "environment_scene_gate")


class AllOpenContextPolicy(nn.Module):
    all_open = True

    def forward(self, raw_logits, allowed, route_name, **metadata):
        return allowed.to(raw_logits.dtype)


def local_input_adapter(dataset):
    if dataset == "thermal":
        from channelthermal.local_records import feature_contract, local_records_scene
    elif dataset == "wind":
        from windfarm.local_records import feature_contract, local_records_scene
    else:
        raise ValueError("Unified local records require an explicit thermal/wind input capability.")
    return feature_contract, local_records_scene


class UnifiedPreparationMixin:
    """Place before ContinuousJointCore in the MRO; no pooled neural value path."""
    input_capability = "local_records_context_v1"

    def _initialize_unified_preparation(self, dataset):
        self.input_dataset = dataset
        contract, _ = local_input_adapter(dataset)
        self.local_input_contract = contract()
        self.dropped_parent_parameter_names = tuple(
            name for name in self.state_dict() if name.split('.')[0] in SUMMARY_ROOTS)
        for name in SUMMARY_ROOTS:
            delattr(self, name)
        self._all_open_context_policy = AllOpenContextPolicy()
        self._context_entries = []
        self._unified_context_plan = None
        if self.routing_mode != 'dense':
            from ..native_retrieval.budget_state import BudgetDualState
            self.joint_budget.duals['context'] = BudgetDualState(0.,rng_identity='round10_unified_context')

    def _uses_scene_summary(self):
        return False

    def _context_value_policy(self):
        if self._unified_context_plan is not None:
            from .preparation import FrozenContextPolicy
            return FrozenContextPolicy(self._unified_context_plan)
        return (self._all_open_context_policy if self.routing_mode == "dense"
                else self.router.incidence)

    def _captures_native_routes(self):
        return True

    def _native_capture_route_names(self):
        return ("mm", "em", "ee", "me")

    def prepare(self, scene, **options):
        full_context_trace = bool(options.get('return_trace')) or (options.get('trace_request') is not None and options.get('trace_request') is not False)
        if full_context_trace:
            from ..content_hypergraph.model_v2 import _TRACE_LIMIT
            request = options.get('trace_request')
            cap = min(request.get('max_elements',_TRACE_LIMIT),_TRACE_LIMIT) if isinstance(request,dict) else _TRACE_LIMIT
            expected = self.context_depth*scene.batch_size*(scene.num_modules+scene.num_environment_tokens)**2*(self.num_heads+1)
            if expected > cap:
                raise ValueError('Full applied context trace exceeds its element budget; use a smaller named scene panel.')
        _, transform = local_input_adapter(self.input_dataset)
        transformed = transform(scene)
        prepared = super().prepare(transformed, **options)
        prepared.local_input_source_scene = scene
        prepared.local_input_source_versions = scene.tensor_version_stamp()
        prepared.local_input_feature_contract = dict(self.local_input_contract)
        if self.training and torch.is_grad_enabled() and self.routing_mode != 'dense':
            keys = scene.scene_key
            keys = tuple(repr(key) for key in keys) if isinstance(keys,(tuple,list)) and len(keys)==scene.batch_size else tuple(
                (repr(keys),index) for index in range(scene.batch_size))
            self._context_entries.append((keys,prepared.context_work_summary))
        return prepared

    def _validate_prepared_state(self, organization):
        super()._validate_prepared_state(organization)
        if organization.local_input_source_versions != organization.local_input_source_scene.tensor_version_stamp():
            raise ValueError("Raw local-input scene changed after preparation.")

    def _finalize_native_preparation(self, organization, cycle_trace):
        from ..joint_native.model import JointNativeCore
        # Original record storage is reusable; no legacy hard graph or fresh
        # calibration is run for unified retrieval.
        prepared = JointNativeCore._finalize_native_preparation(self, organization, cycle_trace)
        if self.routing_mode == 'hypergraph':
            capture = {name:cycle_trace[name+'_native_route_snapshot'] for name in self._native_capture_route_names()}
            prepared.continuous_incidence,prepared.continuous_typed_incidence = applied_native_directory(
                capture,prepared.joint_records)
            from ..native_retrieval.contracts import tensor_versions
            if cycle_trace.get("applied_context_stages"):
                prepared.continuous_capture = capture
            prepared.continuous_incidence_versions = tensor_versions(prepared.continuous_incidence)
        stages = tuple(cycle_trace.get("applied_context_stages", ()))
        prepared.applied_context_stages = stages
        prepared.context_work_summary = cycle_trace.get("context_work_summary")
        prepared.context_stage_metadata = cycle_trace.get("context_stage_metadata", ())
        if self._unified_context_plan is not None:
            from .preparation import FrozenSDPAContextPlan
            prepared.context_stage_metadata=tuple(dict(row,control_origin=(
                'captured_SDPA_Q_K_additive_mask' if isinstance(
                    self._unified_context_plan.stages[row['stage']],FrozenSDPAContextPlan)
                else 'captured_explicit_coefficients_and_normalizers'))
                for row in prepared.context_stage_metadata)
        provenance = []
        for cycle, stage in enumerate(stages):
            for route in ("mm", "em", "ee", "me"):
                snapshot = stage[f"{route}_native_route_snapshot"]
                provenance.append(dict(stage=snapshot.stage_identity, route=route,
                    source_state_version=f"cycle{cycle}/" + dict(mm="boundary_M",em="after_MM_M",
                        ee="after_EM_E",me="after_EE_FFN_E")[route],
                    application="contextual_value_reduction", destination_ids=snapshot.destination_ids,
                    donor_ids=snapshot.donor_ids, incidence=snapshot.applied_incidence,
                    owner_token=snapshot.owner_token, parameter_versions=snapshot.parameter_versions,
                    scene_tensor_versions=snapshot.scene_tensor_versions))
        prepared.context_value_provenance = tuple(provenance)
        for route in ("mm", "em", "ee", "me"):
            prepared.work_counts[f"context_{route}_positive_value_edges"] = sum(
                p["positive_value_edges"] for p in prepared.context_stage_metadata if p['route'] == route)
        return prepared


    @torch.no_grad()
    def capture_unified_context_plan(self,scene,*,max_elements=None):
        """Capture live complete stage reductions on one bounded native scene panel."""
        from .preparation import (FrozenContextPlan,FrozenSDPAContextPlan,
                                  FrozenUnifiedContextPlan,context_plan_tensors)
        from ..native_retrieval.contracts import tensor_versions
        from ..content_hypergraph.model_v2 import _TRACE_LIMIT
        if self.training:raise ValueError('Frozen unified context capture requires eval mode.')
        cap=_TRACE_LIMIT if max_elements is None else max_elements
        if type(cap) is not int or not 0<cap<=2*_TRACE_LIMIT:
            raise ValueError('Frozen context plan needs a positive bounded element budget.')
        m,e=scene.num_modules,scene.num_environment_tokens
        expected=self.context_depth*scene.batch_size*((m+e)**2*(self.num_heads+1)
            +2*(m+e)*self.num_heads)
        expected+=3*scene.batch_size*(m+e)  # IDs, presence and information masks.
        if self.routing_mode=='dense':
            expected+=self.context_depth*scene.batch_size*(4*(m+e)*self.context_dim+2*(m+e))
        if expected>cap:
            raise ValueError('Frozen context controls exceed their element budget; use a smaller named scene panel or explicit bounded budget.')
        include=[f'cycle{cycle}/{route}_attention' for cycle in range(self.context_depth)
                 for route in ('mm','em','ee','me')]
        prepared=self.prepare(scene,return_trace=True,trace_request={'include':include})
        stages={}
        clone=lambda value:value.detach().clone()
        for cycle in prepared.applied_context_stages:
            for route in ('mm','em','ee','me'):
                snapshot=cycle[route+'_native_route_snapshot']
                if self.routing_mode=='dense':
                    valid=snapshot.donor_measure>0
                    measure=torch.where(valid,snapshot.donor_measure,torch.ones_like(snapshot.donor_measure))
                    additive=snapshot.geometry_bias+measure.log()[:,None,None,:]
                    additive=additive.masked_fill(~snapshot.allowed[:,None],-torch.inf)
                    present=snapshot.allowed.any(-1)
                    additive=torch.where(present[:,None,:,None],additive,torch.zeros_like(additive))
                    stages[snapshot.stage_identity]=FrozenSDPAContextPlan(clone(snapshot.applied_incidence),
                        clone(snapshot.query_scaled),clone(snapshot.key_normalized),clone(additive),
                        clone(present),clone(snapshot.applied_message_mass))
                else:
                    coefficients=clone(prepared.trace[snapshot.stage_identity+'_attention'].permute(0,2,3,1))
                    stages[snapshot.stage_identity]=FrozenContextPlan(clone(snapshot.applied_incidence),
                        coefficients,clone(snapshot.applied_message_mass))
        base=[clone(scene.module_ids),clone(scene.environment_ids),clone(scene.module_mask),
              clone(scene.environment_mask),clone(scene.module_measure>0),clone(scene.environment_measure>0)]
        tensors=list(base)
        for stage in sorted(stages):
            tensors.extend(context_plan_tensors(stages[stage]))
        if sum(t.numel() for t in tensors)>cap:
            raise ValueError('Captured frozen context controls exceed their element budget.')
        return FrozenUnifiedContextPlan(self._owner_token,self._parameter_version_stamp(),self._buffer_version_stamp(),
            scene.coordinate_frame_key,scene.normalizer_key,*base,stages,tensor_versions(*tensors))

    def prepare_with_context_plan(self,scene,plan,**options):
        """Fresh raw records with complete frozen contextual coefficients/normalizers."""
        from .preparation import FrozenUnifiedContextPlan
        if self.training or not isinstance(plan,FrozenUnifiedContextPlan):
            raise ValueError('Frozen unified context plans are explicit eval-only interventions.')
        plan.assert_current(self,scene)
        previous=self._unified_context_plan;self._unified_context_plan=plan
        try:return self.prepare(scene,**options)
        finally:self._unified_context_plan=previous

    def joint_constraint_kinds(self):
        return ('M','E') if self.routing_mode == 'dense' else ('M','E','context')

    def joint_begin_update(self,epoch,update_index):
        state = super().joint_begin_update(epoch,update_index)
        target,_ = context_schedule(epoch)
        if self.routing_mode != 'dense':
            self.joint_budget.duals['context'].target_removable_retention.fill_(target)
        state['target_context'] = target
        return state

    def joint_begin_microbatch(self):
        super().joint_begin_microbatch()
        self._context_entries = []

    def joint_microbatch_result(self):
        result = super().joint_microbatch_result()
        if self.routing_mode == 'dense':
            return result
        target,pressure = context_schedule(self._joint_epoch)
        cases = {}
        for keys,work in self._context_entries:
            for key,row in zip(keys,work):
                cases[key] = cases.get(key,0) + row
        zero = next(self.router.incidence.parameters()).sum()*0
        residuals,controllers = [],[]
        for proxy,actual,possible in cases.values():
            residuals.append(torch.where(possible>0,(proxy-target*possible)/possible.clamp_min(1),proxy*0)*pressure)
            controllers.append(torch.where(possible>0,(actual-target*possible)/possible.clamp_min(1),actual*0).detach())
        result['constraints']['context'] = dict(residual_numerator=sum(residuals,zero),denominator=len(cases),
            controller_numerator=sum(controllers,zero.detach()))
        result['telemetry'].update(unified_context_target=target,
            unified_context_actual_edges=sum((r[1].detach() for r in cases.values()),zero.detach()),
            unified_context_proxy_edges=sum((r[0].detach() for r in cases.values()),zero.detach()))
        self._context_entries = []
        return result

    def joint_after_update(self,epoch,update_index,residuals):
        result = super().joint_after_update(epoch,update_index,residuals)
        if self.routing_mode != 'dense':
            dual = self.joint_budget.duals['context']
            dual.observe_residual(residuals['context'],int(dual.macro_step)+1)
        return result


def context_schedule(epoch):
    age = max(0,int(epoch)-3000)
    target = 1. if age==0 else .98-.08*min(max(age-1,0)/999,1)
    return target,min(max(age-1,0)/199,1)


def applied_native_directory(capture,records):
    """Reuse actual contextual b; retrieval self-membership is a separate law."""
    from ..joint_native.graph import _validated_raw
    records.assert_current()
    snapshots = tuple(capture.values())
    for field in ('owner_token','parameter_versions','buffer_versions','scene_identity',
                  'scene_tensor_versions','model_config_key'):
        if any(getattr(snapshot,field)!=getattr(snapshots[0],field) for snapshot in snapshots[1:]):
            raise ValueError('Applied native stages must share owner, source state and configuration.')
    if len({snapshot.stage_identity.rsplit('/',1)[0] for snapshot in snapshots}) != 1:
        raise ValueError('Applied native directory requires one real contextual cycle.')
    m,n=records.module_count,records.ids.shape[1]
    slices={'m':slice(0,m),'e':slice(m,n)}
    blocks={}
    for name in ('mm','me','em','ee'):
        destination,donor=slices[name[0]],slices[name[1]]
        _validated_raw(capture,records,name,destination,donor)
        gates=capture[name].applied_incidence
        if gates is None:raise ValueError('Unified retrieval requires actual applied contextual incidence.')
        allowed=records.present[:,destination,None]&records.eligible[:,None,donor]
        gates=gates*allowed
        if name[0]==name[1]:
            identity=torch.eye(gates.shape[-1],device=gates.device,dtype=torch.bool)[None]&allowed
            gates=torch.where(identity,torch.ones_like(gates),gates)
        blocks[name]=gates
    return torch.cat((torch.cat((blocks['mm'],blocks['me']),-1),
                      torch.cat((blocks['em'],blocks['ee']),-1)),1),blocks


def unified_parent_inventory(parent, model):
    """Exact common/drop/fresh state map; callers reuse it for named Adam state."""
    source, target = parent.state_dict(), model.state_dict()
    dropped = sorted(name for name in source if name.split('.')[0] in SUMMARY_ROOTS)
    unexpected = set(source) - set(target) - set(dropped)
    if unexpected:
        raise ValueError(f"Unexpected incompatible parent state: {sorted(unexpected)}")
    common = sorted(set(source) & set(target))
    for name in common:
        if hasattr(source[name], 'shape') and source[name].shape != target[name].shape:
            raise ValueError(f"Parent tensor shape differs: {name}")
    return dict(copied_state_names=common, dropped_summary_state_names=dropped,
                fresh_state_names=sorted(set(target)-set(source)),
                dropped_optimizer_parameter_names=sorted(name for name, _ in parent.named_parameters()
                    if name.split('.')[0] in SUMMARY_ROOTS))


# Imported after the mixin definitions so preparation primitives remain usable
# independently while the continuous retrieval implementation is qualified.
from .model import ContinuousJointCore


class UnifiedJointCore(UnifiedPreparationMixin, ContinuousJointCore):
    schema = "unified_continuous_joint_native_v1"

    def __init__(self, base_model_config, *, input_dataset, input_capability="local_records_context_v1",
                 input_contract_sha256=None, **options):
        if input_capability != self.input_capability:
            raise ValueError("Unified input capability differs from local_records_context_v1.")
        self.input_dataset = input_dataset
        super().__init__(base_model_config, **options)
        self._initialize_unified_preparation(input_dataset)
        if self.dropout_probability != 0.:
            raise ValueError('The unified v1 scientific capability requires the inherited zero dropout.')
        if input_contract_sha256 is not None and input_contract_sha256 != self.local_input_contract['sha256']:
            raise ValueError('Unified local-input named feature contract differs from the saved schema hash.')

    @classmethod
    def validate_config(cls,config):
        extra={'input_dataset','input_capability','input_contract_sha256'}
        if config.get('schema')!=cls.schema or not extra.issubset(config):
            raise ValueError('Unified constructor/input capability inventory differs.')
        contract,_=local_input_adapter(config['input_dataset'])
        if config['input_capability']!='local_records_context_v1' or config['input_contract_sha256']!=contract()['sha256']:
            raise ValueError('Unified named input feature contract differs.')
        translated={key:value for key,value in config.items() if key not in extra}
        translated['schema']=ContinuousJointCore.schema
        return ContinuousJointCore.validate_config(translated)

    def get_config(self):
        config = super().get_config()
        config.update(input_dataset=self.input_dataset, input_capability=self.input_capability,
                      input_contract_sha256=self.local_input_contract['sha256'])
        return config

    def _config_key(self):
        return (*super()._config_key(), self.input_dataset, self.input_capability,
                self.local_input_contract['sha256'])

    @classmethod
    def from_parent(cls, parent, **options):
        from ..joint_native.model import JointNativeCore
        if not isinstance(parent, JointNativeCore) or parent.routing_mode != 'dense':
            raise ValueError("Unified children require the literal shared dense M+E parent.")
        model = cls(parent.get_config()['base_model_config'], **options).to(next(parent.parameters()))
        inventory = unified_parent_inventory(parent, model)
        source, state = parent.state_dict(), model.state_dict()
        for name in inventory['copied_state_names']:
            state[name] = source[name]
        model.load_state_dict(state, strict=True)
        model.set_training_clock(3000, 0)
        model.parent_initialization_inventory = inventory
        model.train(parent.training)
        return model
