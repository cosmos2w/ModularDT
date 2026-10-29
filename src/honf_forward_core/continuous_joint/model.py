"""E-first continuous retrieval with a checkpoint over the complete query tile.

The original local/native physical readout is reused. No hard-router epochs,
random dispatch, mandatory winner, or nearest-donor protection are inherited.
"""
from __future__ import annotations
from types import SimpleNamespace
from dataclasses import replace
from ..native_retrieval.contracts import tensor_versions
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ..content_hypergraph.model_v2 import ContentHypergraphCoreV2
from ..content_hypergraph.readers_v2 import DirectSourceReaderV2
from ..content_hypergraph.decoding import pack_local_pairs
from ..content_hypergraph.query_chunks_v1 import slice_query_v1
from ..content_hypergraph.contracts import CoreOutput
from ..joint_native.model import JointNativeCore
from ..joint_native.readers import raw_scores, read_typed
from .routing import ContinuousRouter
from .reader import read_continuous
from .budget import ContinuousBudgetState, compact_work, microbatch_result


class ContinuousJointCore(ContentHypergraphCoreV2):
    schema='continuous_joint_native_v1'
    def __init__(self,base_model_config,*,routing_mode='hypergraph',native_backend='packed',
                 checkpoint_pipeline=True,query_tile_size=256,entmax_temperature=1.):
        if routing_mode not in ('dense','hypergraph','direct') or native_backend not in (
                'dense','packed','source_order','segment','dense_weighted'):
            raise ValueError('Invalid continuous routing/execution capability.')
        if type(query_tile_size) is not int or query_tile_size<1 or type(checkpoint_pipeline) is not bool:
            raise ValueError('Invalid full-pipeline checkpoint settings.')
        base=dict(base_model_config)
        if base.pop('schema',None)!=ContentHypergraphCoreV2.schema or base.get('architecture')!='A2-D':
            raise ValueError('Continuous native retrieval requires the original A2-D base.')
        self.routing_mode=routing_mode;self.native_backend=native_backend
        self.checkpoint_pipeline=checkpoint_pipeline;self.query_tile_size=query_tile_size
        self.checkpoint_mode='tile' if checkpoint_pipeline else 'off'
        self.context_execution_backend='segment'
        self._entmax_temperature_value=float(entmax_temperature)
        self._joint_epoch=3000;self._joint_update_index=0;self._joint_macro_clock=0
        self._joint_entries=[];self._joint_route_override=None;self._continuous_frozen_plan=None
        super().__init__(**base)
        self.register_buffer('joint_epoch',torch.tensor(3000,dtype=torch.long))
        self.register_buffer('joint_update_index',torch.tensor(0,dtype=torch.long))
        self.register_buffer('joint_macro_clock',torch.tensor(0,dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._stable_seed('environment_reader'))
            self.environment_reader=DirectSourceReaderV2(self.local_hidden_dim,self.coordinate_dim,
                self.num_heads,thermal=False,source_value_conditioning=self.source_value_conditioning)
            self.environment_conditioning=nn.Linear(self.context_dim,self.context_dim,bias=False)
        self._initialize_named_parameters()
        with torch.no_grad():self.environment_conditioning.weight.zero_()
        self.router=ContinuousRouter(self.num_heads,mode=routing_mode,temperature=entmax_temperature)
        self._entmax_temperature_value=float(self.router.entmax_temperature)
        self.joint_budget=ContinuousBudgetState()

    @classmethod
    def validate_config(cls,config):
        allowed={'schema','base_model_config','routing_mode','native_backend','checkpoint_pipeline','query_tile_size','entmax_temperature'}
        if set(config)!=allowed or config.get('schema')!=cls.schema:
            raise ValueError('Continuous configuration inventory differs.')
        translated=dict(schema=JointNativeCore.schema,base_model_config=config['base_model_config'],routing_mode='dense',
            native_backend='dense_ME',routing_seed=0,environment_value_reader=True,checkpoint_values=False,query_tile_size=None)
        return JointNativeCore.validate_config(translated)

    def get_config(self):
        base=ContentHypergraphCoreV2.get_config(self);base['schema']=ContentHypergraphCoreV2.schema
        return dict(schema=self.schema,base_model_config=base,routing_mode=self.routing_mode,
            native_backend=self.native_backend,checkpoint_pipeline=self.checkpoint_pipeline,
            query_tile_size=self.query_tile_size,entmax_temperature=self._entmax_temperature_value)
    @classmethod
    def from_config(cls,config):
        c=dict(config)
        if c.pop('schema',None)!=cls.schema:raise ValueError('Continuous model schema differs.')
        return cls(**c)
    @classmethod
    def from_parent(cls,parent,**options):
        if not isinstance(parent,JointNativeCore) or parent.routing_mode!='dense':
            raise ValueError('Continuous children require the literal common dense M+E parent.')
        core=cls(parent.get_config()['base_model_config'],**options).to(next(parent.parameters()))
        state=core.state_dict();source=parent.state_dict()
        if set(source)-set(state):raise ValueError('Parent has incompatible state.')
        state.update(source);core.load_state_dict(state,strict=True)
        core.set_training_clock(3000,0);core.train(parent.training)
        return core
    def _config_key(self):
        return (tuple(sorted(ContentHypergraphCoreV2.get_config(self).items())),self.routing_mode,
            self.native_backend,self.checkpoint_pipeline,self.query_tile_size,self._joint_epoch,
            self._joint_update_index,self._joint_macro_clock,self._entmax_temperature_value,
            self.checkpoint_mode,self.context_execution_backend)

    def set_execution_options(self,*,backend=None,query_tile_size=None,checkpoint=None,context_backend=None):
        """Select execution without changing trained gates or physical laws.

        Change these before preparing a scene: the prepared-state key includes
        the options so an old prepared graph cannot silently cross a change.
        """
        if backend is not None:
            if backend not in ('dense','packed','source_order','segment','dense_weighted'):
                raise ValueError('Unknown continuous execution backend.')
            self.native_backend=backend
        if query_tile_size is not None:
            if type(query_tile_size) is not int or query_tile_size<1:
                raise ValueError('Query tile size must be a positive integer.')
            self.query_tile_size=query_tile_size
        if checkpoint is not None:
            if checkpoint not in ('off','tile','values'):
                raise ValueError('Checkpoint execution must be off, values or tile.')
            self.checkpoint_mode=checkpoint
            self.checkpoint_pipeline=checkpoint!='off'
        if context_backend is not None:
            if context_backend not in ('segment','dense_weighted'):
                raise ValueError('Unknown contextual reduction backend.')
            self.context_execution_backend=context_backend
        if hasattr(self,'router'):
            self.router.incidence.execution_backend=self.context_execution_backend
        return self
    def _captures_native_routes(self):return self.routing_mode=='hypergraph'
    def _native_capture_route_names(self):return ('mm','me','em','ee')
    def _uses_native_retrieval(self):return True

    def _finalize_native_preparation(self,organization,cycle_trace):
        # Reuse only owner-bound original record storage, not Round9 graph logic.
        JointNativeCore._finalize_native_preparation(self,organization,cycle_trace)
        if self.routing_mode=='hypergraph':
            capture={name:cycle_trace[name+'_native_route_snapshot'] for name in self._native_capture_route_names()}
            organization.continuous_incidence,organization.continuous_typed_incidence=self.router.incidence.retrieval(capture,organization.joint_records)
            organization.continuous_incidence_versions=tensor_versions(organization.continuous_incidence)
        return organization
    def _validate_prepared_state(self,organization):
        ContentHypergraphCoreV2._validate_prepared_state(self,organization)
        if tensor_versions(organization.joint_environment_keys,organization.joint_environment_values)!=organization.joint_projection_versions:
            raise ValueError('Prepared E projections changed after preparation.')
        organization.joint_records.assert_current()
        organization.native_token_store.assert_current(model_owner=self._owner_token,
            model_parameter_versions=self._parameter_version_stamp(),model_buffer_versions=self._buffer_version_stamp(),
            scene_tensor_versions=organization.scene.tensor_version_stamp(),
            scene_identity=(organization.scene.scene_key,organization.scene.normalizer_key,organization.scene.coordinate_frame_key),
            model_config_key=self._config_key())
        if hasattr(organization,'continuous_incidence') and tensor_versions(organization.continuous_incidence)!=organization.continuous_incidence_versions:
            raise ValueError('Prepared continuous incidence changed after preparation.')

    @torch.no_grad()
    def capture_continuous_trace(self,organization,query):
        """Bounded owner-validated coefficients, original-ID paths and actual values.

        This is a live coefficient trace, not a frozen causal execution plan.
        Ordinary inference does not construct or retain this payload.
        """
        self.validate_prepared_state(organization,query)
        if query.num_queries>self.query_tile_size:
            raise ValueError('Continuous trace exceeds the declared query-tile bound.')
        qstates,ctx=self._prepare_native_decode(organization,query,self._query_states(organization,query))
        gates,e,seeds,work=ctx
        _,_,envelope,qmask,_=self._continuous_route_tile(organization,query,self._query_states(organization,query))
        work=compact_work(gates,envelope,organization.joint_records,qmask,self.joint_budget.schedule()['delta'])
        m=self._read('module',organization,query,qstates,gates[...,:organization.scene.num_modules])
        compact_trace=hasattr(organization, 'compact_groups')
        result=dict(gates=gates,seeds=None if compact_trace else seeds,module_ids=organization.scene.module_ids,
            environment_ids=organization.scene.environment_ids,query_ids=query.query_ids,
            local_module_envelope=envelope,
            environment_access=e.alpha,environment_context=e.context,conditioned_query=qstates,
            module_access=m.alpha,source_coefficients=m.response,compact_work=work,
            module_work_counts=dict(m.counts),environment_work_counts=dict(e.counts),
            value_execution_backend='dense_ME' if self.routing_mode=='dense' else self.native_backend)
        if compact_trace:
            compact=organization.compact_groups
            query_valid_mask=(query.query_mask if query.query_mask is not None
                              else torch.ones(gates.shape[:2],device=gates.device,dtype=torch.bool))
            result.update(compact_group_weights=seeds,compact_membership=compact.membership,
                compact_group_mask=compact.group_mask,compact_group_types=compact.group_types,
                compact_group_controls=compact.controls,compact_group_centroids=compact.centroids,
                compact_group_lengths=compact.lengths,compact_group_extents=compact.extents,
                compact_live_K=compact.live_counts,compact_allocated_K=compact.allocated_count,
                compact_lineage=tuple(row.as_dict() for row in compact.topology),
                query_valid_mask=query_valid_mask,
                original_record_present=organization.joint_records.present,
                original_record_eligible=organization.joint_records.eligible,
                original_record_count=organization.joint_records.present.sum(-1))
        if hasattr(organization,'continuous_incidence'):result['incidence']=organization.continuous_incidence
        return result

    @torch.no_grad()
    def capture_continuous_plan(self,organization,query):
        from .contracts import FrozenContinuousReadPlan
        if self.training or query.query_ids is None:raise ValueError('Frozen continuous plans require eval mode and native receiver IDs.')
        trace=self.capture_continuous_trace(organization,query);scene=organization.scene
        qmask=query.query_mask if query.query_mask is not None else torch.ones_like(query.query_ids,dtype=torch.bool)
        clone=lambda x:x.detach().clone()
        return FrozenContinuousReadPlan(self._owner_token,self._parameter_version_stamp(),self._buffer_version_stamp(),
            scene.coordinate_frame_key,scene.normalizer_key,clone(scene.module_ids),clone(scene.environment_ids),
            clone(query.query_ids),clone(scene.module_mask),clone(scene.environment_mask),clone(qmask),
            clone(trace['gates']),clone(trace['environment_access']),clone(trace['module_access']))

    def decode_with_plan(self,organization,query,plan,controls=None,**options):
        """Fixed value-plane coefficients on validated freshly prepared inputs.

        Geometry/latent values may change while original typed/query IDs, owner,
        weights and frame remain bound. Live routing is restored even on error.
        """
        from .contracts import FrozenContinuousReadPlan
        if self.training or not isinstance(plan,FrozenContinuousReadPlan):raise ValueError('Explicit frozen plans are eval-only interventions.')
        self.validate_prepared_state(organization,query);plan.assert_current(self,organization,query)
        if query.num_queries>self.query_tile_size:raise ValueError('Frozen read plan exceeds bounded query tile.')
        previous=self._continuous_frozen_plan;self._continuous_frozen_plan=plan
        try:return self.decode(organization,query,controls,**options)
        finally:self._continuous_frozen_plan=previous

    def _checkpoint_native_tile(self):
        # Frozen plans are scoped to the forward call. Checkpoint backward runs
        # after that scope closes, so a frozen-plan diagnostic retains its
        # bounded tile graph instead of recomputing with live routing.
        return self.checkpoint_mode=='tile' and torch.is_grad_enabled() and self._continuous_frozen_plan is None

    def _read(self,kind,organization,query,qstates,gate,scores=None,return_messages=False):
        scene=organization.scene;is_m=kind=='module'
        valid=getattr(scene,kind+'_mask')
        if not is_m:valid=valid & (scene.environment_measure>0)
        if (self.routing_mode=='dense' and self.native_backend in ('packed','dense','source_order')
                and self._continuous_frozen_plan is None):
            # The all-open control keeps the literal e3000 parent's typed
            # normalizer and reduction order. This still executes entirely
            # inside the full query-tile checkpoint; H/S use the guarded law.
            result=read_typed(self.direct_reader if is_m else self.environment_reader,
                query,qstates,coordinates=getattr(scene,kind+'_coordinates'),
                lengths=getattr(scene,kind+'_characteristic_lengths'),valid=valid,
                measure=getattr(scene,kind+'_measure'),
                reference=self._measure_references(scene,qstates.dtype)[0 if is_m else 1],
                frame_scale=organization.frame_scale,
                keys=organization.direct_source_keys if is_m else organization.joint_environment_keys,
                values=organization.direct_source_values if is_m else organization.joint_environment_values,
                gate=gate,backend='dense_ME',return_messages=return_messages)
            qmask=query.query_mask if query.query_mask is not None else torch.ones(
                qstates.shape[:2],device=qstates.device,dtype=torch.bool)
            result.counts['available_present_rows']=int((valid[:,None]&qmask[...,None]).sum())
            return result
        return read_continuous(self.direct_reader if is_m else self.environment_reader,query,qstates,
            coordinates=getattr(scene,kind+'_coordinates'),lengths=getattr(scene,kind+'_characteristic_lengths'),
            valid=valid,measure=getattr(scene,kind+'_measure'),frame_scale=organization.frame_scale,
            keys=organization.direct_source_keys if is_m else organization.joint_environment_keys,
            values=organization.direct_source_values if is_m else organization.joint_environment_values,
            gate=gate,scores=scores,backend=self.native_backend,
            return_messages=return_messages,
            checkpoint_values=self.checkpoint_mode=='values',
            reference=self._measure_references(scene,qstates.dtype)[0 if is_m else 1],
            frozen_coefficients=(None if self._continuous_frozen_plan is None else
                self._continuous_frozen_plan.module_coefficients if is_m else self._continuous_frozen_plan.environment_coefficients))

    def _continuous_route_tile(self,organization,query,query_states):
        scene=organization.scene;batch,queries=query_states.shape[:2]
        qmask=query.query_mask if query.query_mask is not None else torch.ones((batch,queries),device=query_states.device,dtype=torch.bool)
        local=pack_local_pairs(scene,query,organization.frame_scale)
        envelope=query_states.new_zeros(batch,queries,scene.num_modules)
        if local.count:envelope=envelope.index_put((local.batch_index,local.query_index,local.module_index),local.support_weight)
        if self._continuous_frozen_plan is not None:
            self._continuous_frozen_plan.assert_current(self,organization,query)
            gates=self._continuous_frozen_plan.gates;seeds=None;raw_e=None
        elif self.routing_mode=='dense':
            gates=(organization.joint_records.eligible[:,None]&qmask[...,None]).to(query_states)
            seeds=None;raw_e=None
        elif hasattr(organization, 'compact_groups'):
            # The compact route scores Q against live K controls and applies
            # the direct K-by-N membership matrix. Original E information
            # scores remain the legitimate first physical read.
            raw_e=raw_scores(self.environment_reader,query,query_states,scene.environment_coordinates,
                scene.environment_characteristic_lengths,organization.frame_scale,organization.joint_environment_keys)
            gates,seeds,_group_logits=self.compact_retrieval.query_access(
                organization.compact_groups,organization,query,query_states,envelope,qmask)
        else:
            raw_m=raw_scores(self.direct_reader,query,query_states,scene.module_coordinates,
                scene.module_characteristic_lengths,organization.frame_scale,organization.direct_source_keys)
            raw_e=raw_scores(self.environment_reader,query,query_states,scene.environment_coordinates,
                scene.environment_characteristic_lengths,organization.frame_scale,organization.joint_environment_keys)
            gates,seeds=self.router(raw_m,raw_e,organization.joint_records,envelope,qmask,
                getattr(organization,'continuous_incidence',None))
        return gates,seeds,envelope,qmask,raw_e

    def _prepare_native_decode(self,organization,query,query_states,*,trace_enabled=False):
        gates,seeds,envelope,qmask,raw_e=self._continuous_route_tile(organization,query,query_states)
        environment=self._read('environment',organization,query,query_states,gates[...,organization.scene.num_modules:],scores=raw_e)
        environment=SimpleNamespace(**vars(environment),trace_enabled=trace_enabled)
        conditioned=query_states+self.environment_conditioning(environment.context)
        # Physical and work checkpoint regions have independent saved-tensor
        # lifetimes. A work-only VJP must never recompute or retain wide values.
        work=query_states.new_zeros(query.batch_size,2,8)
        return conditioned,(gates,environment,seeds,work)

    def _checkpoint_work(self,organization,query):
        if not(self.training and torch.is_grad_enabled()) or self.routing_mode=='dense':
            return organization.module_states.new_zeros(query.batch_size,2,8)
        def function(coords):
            tile=replace(query,coordinates=coords)
            qstates=self._query_states(organization,tile)
            gates,seeds,envelope,qmask,raw_e=self._continuous_route_tile(organization,tile,qstates)
            return compact_work(gates,envelope,organization.joint_records,qmask,self.joint_budget.schedule()['delta'])
        return checkpoint(function,query.coordinates,use_reentrant=False) if self.checkpoint_pipeline else function(query.coordinates)

    def _decode_native_values(self,organization,query,query_states,local_pairs,*,return_messages=False,decode_context=None):
        gates,environment,seeds,work=decode_context
        m=self._read('module',organization,query,query_states,gates[...,:organization.scene.num_modules],return_messages=return_messages)
        counts={'joint_M_'+k:v for k,v in m.counts.items()};counts.update({'joint_E_'+k:v for k,v in environment.counts.items()})
        # M is scored once for routing, then again after E conditioning for
        # information attention. E routing scores are reused by its reader.
        compact_route=hasattr(organization, 'compact_groups') and self._continuous_frozen_plan is None
        counts['joint_pre_E_M_score_rows']=(m.counts['candidate_rows']
            if self.routing_mode!='dense' and self._continuous_frozen_plan is None and not compact_route else 0)
        counts['joint_pre_E_group_score_rows']=(
            query.batch_size * query.num_queries * organization.compact_groups.score_slot_count
            if compact_route else 0)
        counts.update(fine_executed_rows=m.counts['value_rows'],fine_selected_rows=m.counts['selected_rows'],
            fine_message_projection_rows=m.counts['message_projection_rows'],fine_scalar_projection_rows=m.counts['scalar_projection_rows'],
            fine_fourier_feature_entries=m.counts['fourier_entries'],fine_hidden_entries=m.counts['hidden_entries'],
            source_score_rows=m.counts['score_rows'],native_environment_consumed_ids=environment.counts['value_rows'],native_environment_metadata_ids=0)
        trace=dict(joint_hard_support=gates>0,native_hard_support=gates[...,:organization.scene.num_modules]>0,
            joint_environment_support=gates[...,organization.scene.num_modules:]>0,
            joint_environment_access=environment.alpha,joint_environment_context=environment.context,
            joint_conditioned_query=query_states) if return_messages or getattr(environment,'trace_enabled',False) else {}
        return SimpleNamespace(context=m.context,alpha=m.alpha,response=m.response,messages=m.messages,counts=counts,trace=trace,work=work)

    def _tile(self,organization,query,controls,accumulation_dtype,local_gates=None,return_affine=False):
        qstates,ctx=self._prepare_native_decode(organization,query,self._query_states(organization,query))
        local_context,pairs,coeff=self.local_reader(organization.scene,query,organization.module_states,qstates,
            organization.frame_scale,self._module_measure_reference(organization.scene,qstates.dtype),
            local_gates=local_gates,execution_backend=self.reader_execution_backend)
        result=self._decode_native_values(organization,query,qstates,pairs,decode_context=ctx)
        field=self.field_head(qstates,result.context,local_context)
        if query.query_mask is not None:field=field*query.query_mask[...,None]
        temp=field.new_empty((0,))
        if controls is not None:
            dtype=self._accumulation_dtype(accumulation_dtype,controls.dtype)
            direct=torch.einsum('bqm,bm->bq',result.response.to(dtype),controls.to(dtype))
            temp=direct+self._local_apply(pairs,coeff,controls,query.batch_size,query.num_queries,dtype=dtype)
            if query.query_mask is not None:temp=temp*query.query_mask
        # Only fixed-size work summaries and compact physical outputs escape.
        counts=self._compact_execution_counts(result.counts,qstates)
        if return_affine:return field,result.response,coeff,result.work,counts
        return field,temp,result.work,counts

    def _response_tile(self,organization,query,local_gates=None):
        qstates,ctx=self._prepare_native_decode(organization,query,self._query_states(organization,query))
        local_context,pairs,coeff=self.local_reader(organization.scene,query,organization.module_states,qstates,
            organization.frame_scale,self._module_measure_reference(organization.scene,qstates.dtype),
            local_gates=local_gates,execution_backend=self.reader_execution_backend)
        result=self._decode_native_values(organization,query,qstates,pairs,decode_context=ctx)
        counts=self._compact_execution_counts(result.counts,qstates)
        return result.response,coeff,result.work,counts

    def _read_for_response(self,organization,query,*,edge_gates=None,uniform_access=False,local_gates=None):
        self.validate_prepared_state(organization,query)
        self._validate_edge_intervention(organization,edge_gates,uniform_access)
        self._validate_local_intervention(organization,local_gates)
        coefficients=[];locals_=[];total_work=None;total_counts=None
        for start in range(0,query.num_queries,self.query_tile_size):
            tile=slice_query_v1(query,start,min(start+self.query_tile_size,query.num_queries))
            def function(coords,tile=tile):
                return self._response_tile(organization,replace(tile,coordinates=coords),local_gates)
            direct,local,work,counts=checkpoint(function,tile.coordinates,use_reentrant=False) if self._checkpoint_native_tile() else function(tile.coordinates)
            work=self._checkpoint_work(organization,tile)
            coefficients.append(direct);locals_.append(local)
            total_counts=counts if total_counts is None else total_counts+counts
            total_work=work if total_work is None else total_work+work
        pairs=pack_local_pairs(organization.scene,query,organization.frame_scale)
        # pack_local_pairs is ordered by query-tile traversal only after sorting
        # by (query tile,batch,query,module). Build an explicit compact row join.
        if locals_:
            tile_coeff=torch.cat(locals_)
            tile_rows=[]
            for start in range(0,query.num_queries,self.query_tile_size):
                tile=slice_query_v1(query,start,min(start+self.query_tile_size,query.num_queries))
                p=pack_local_pairs(organization.scene,tile,organization.frame_scale)
                tile_rows.append(torch.stack((p.batch_index,p.query_index+start,p.module_index),-1))
            rows=torch.cat(tile_rows)
            key=lambda b,q,m:(b*query.num_queries+q)*organization.scene.num_modules+m
            order=key(*rows.unbind(-1)).argsort()
            wanted=key(pairs.batch_index,pairs.query_index,pairs.module_index)
            locations=torch.searchsorted(key(*rows.unbind(-1))[order],wanted)
            local_coeff=tile_coeff[order[locations]]
        else:local_coeff=organization.module_states.new_empty((0,))
        if self.training and torch.is_grad_enabled() and total_work is not None and self.routing_mode!='dense':self._record_compact_work(organization,total_work)
        direct=torch.cat(coefficients,1) if coefficients else organization.module_states.new_zeros(query.batch_size,0,organization.scene.num_modules)
        return None,direct,pairs,local_coeff,self._work_count_dict(total_counts,query)

    @staticmethod
    def _compact_execution_counts(counts,reference):
        keys=[f'joint_{kind}_{metric}' for metric in (
            'value_rows','selected_rows','available_present_rows','candidate_rows',
            'message_projection_rows','head_reduction_projection_rows','scalar_projection_rows')
            for kind in ('M','E')]
        keys += ['joint_pre_E_M_score_rows','joint_pre_E_group_score_rows']
        return torch.stack([torch.as_tensor(counts.get(key,0),device=reference.device,dtype=torch.long)
                            for key in keys])

    def _work_count_dict(self,total_counts,query):
        values=[0]*16 if total_counts is None else total_counts.detach().cpu().tolist()
        if len(values)==8:values += [values[0],values[1],0,0,values[0] if self.thermal_response_rank else 0,0,0,0]
        elif len(values)==14:values += [0,0]
        result=dict(receiver_rows=query.batch_size*query.num_queries,
            fine_executed_rows=values[0],fine_selected_rows=values[2],fine_message_projection_rows=values[8],
            fine_head_reduction_projection_rows=values[10],
            fine_scalar_projection_rows=values[12],
            fine_fourier_feature_entries=values[0]*7*self.coordinate_dim,fine_hidden_entries=values[0]*self.local_hidden_dim,
            source_score_rows=values[6],native_environment_consumed_ids=values[1],native_environment_metadata_ids=0)
        for index,kind in enumerate(('M','E')):
            v,selected,available,candidate=values[index],values[index+2],values[index+4],values[index+6]
            result.update({'joint_'+kind+'_value_rows':v,'joint_'+kind+'_selected_rows':selected,
                'joint_'+kind+'_available_present_rows':available,'joint_'+kind+'_candidate_rows':candidate,
                'joint_'+kind+'_score_rows':candidate,'joint_'+kind+'_message_projection_rows':values[8+index],
                'joint_'+kind+'_hidden_value_rows':v,
                'joint_'+kind+'_head_reduction_projection_rows':values[10+index],
                'joint_'+kind+'_scalar_projection_rows':values[12+index],
                'joint_'+kind+'_fourier_entries':v*7*self.coordinate_dim,'joint_'+kind+'_hidden_entries':v*self.local_hidden_dim})
        result['joint_pre_E_M_score_rows']=values[14]
        result['joint_pre_E_group_score_rows']=values[15]
        if self.training and torch.is_grad_enabled() and self.routing_mode!='dense':
            result['continuous_work_replay_score_rows_M']=values[6]
            result['continuous_work_replay_score_rows_E']=values[7]
        return result

    def _decode_features(self,organization,query,*,return_trace=False,trace_request=None,
                         edge_gates=None,uniform_access=False,local_gates=None):
        # Thermal's primary native API prepares affine factors directly. Tile
        # this entry point too: tiling decode alone leaves its actual loss path
        # with full-request score/value activation lifetimes.
        if return_trace or trace_request is not None:
            if query.num_queries>self.query_tile_size:raise ValueError('Continuous full trace requires a bounded query tile.')
            return ContentHypergraphCoreV2._decode_features(self,organization,query,return_trace=return_trace,
                trace_request=trace_request,edge_gates=edge_gates,uniform_access=uniform_access,local_gates=local_gates)
        fields=[];directs=[];local_coefficients=[];tile_rows=[];total_work=None;total_counts=None
        for start in range(0,query.num_queries,self.query_tile_size):
            tile=slice_query_v1(query,start,min(start+self.query_tile_size,query.num_queries))
            def function(coords,tile=tile):
                return self._tile(organization,replace(tile,coordinates=coords),None,None,local_gates,return_affine=True)
            field,direct,local,work,counts=checkpoint(function,tile.coordinates,use_reentrant=False) if self._checkpoint_native_tile() else function(tile.coordinates)
            work=self._checkpoint_work(organization,tile)
            fields.append(field);directs.append(direct);local_coefficients.append(local)
            p=pack_local_pairs(organization.scene,tile,organization.frame_scale)
            tile_rows.append(torch.stack((p.batch_index,p.query_index+start,p.module_index),-1))
            total_work=work if total_work is None else total_work+work
            total_counts=counts if total_counts is None else total_counts+counts
        pairs=pack_local_pairs(organization.scene,query,organization.frame_scale)
        if local_coefficients:
            rows=torch.cat(tile_rows);coeff=torch.cat(local_coefficients)
            key=lambda b,q,m:(b*query.num_queries+q)*organization.scene.num_modules+m
            keys=key(*rows.unbind(-1));order=keys.argsort()
            locations=torch.searchsorted(keys[order],key(pairs.batch_index,pairs.query_index,pairs.module_index))
            local=coeff[order[locations]]
        else:local=organization.module_states.new_empty((0,))
        if self.training and torch.is_grad_enabled() and total_work is not None and self.routing_mode!='dense':self._record_compact_work(organization,total_work)
        field=torch.cat(fields,1) if fields else organization.module_states.new_zeros(query.batch_size,0,self.output_dim)
        direct=torch.cat(directs,1) if self.thermal_response_rank and directs else None
        return field,None,None,None,direct,local,pairs,{},self._work_count_dict(total_counts,query)

    def _record_compact_work(self,organization,work):
        keys=organization.scene.scene_key
        if not isinstance(keys,(tuple,list)) or len(keys)!=organization.scene.batch_size:keys=tuple((repr(keys),i) for i in range(organization.scene.batch_size))
        else:keys=tuple(repr(k) for k in keys)
        self._joint_entries.append((keys,work))

    def _decode_validated(self,organization,query,controls=None,*,return_basis=False,return_trace=False,
            trace_request=None,accumulation_dtype=None,edge_gates=None,uniform_access=False,local_gates=None):
        self._validate_edge_intervention(organization,edge_gates,uniform_access)
        self._validate_local_intervention(organization,local_gates)
        if controls is not None:
            if not self.thermal_response_rank or tuple(controls.shape)!=(query.batch_size,organization.scene.num_modules):
                raise ValueError('Controls must preserve physical M source correspondence.')
            self._validate_source_read_endpoint(controls,organization.scene.module_mask)
        if return_trace or trace_request is not None:
            if query.num_queries>self.query_tile_size:raise ValueError('Continuous full trace requires a bounded query tile.')
            return ContentHypergraphCoreV2._decode_validated(self,organization,query,controls,
                return_basis=return_basis,return_trace=return_trace,trace_request=trace_request,
                accumulation_dtype=accumulation_dtype,edge_gates=edge_gates,uniform_access=uniform_access,local_gates=local_gates)
        fields=[];temps=[];total_work=None;total_counts=None
        for start in range(0,query.num_queries,self.query_tile_size):
            tile=slice_query_v1(query,start,min(start+self.query_tile_size,query.num_queries))
            # Freeze each closure's tile: backward may occur after the loop ends.
            def function(coords,tile=tile):
                return self._tile(organization,replace(tile,coordinates=coords),controls,accumulation_dtype,local_gates)
            result=checkpoint(function,tile.coordinates,use_reentrant=False) if self._checkpoint_native_tile() else function(tile.coordinates)
            field,temp,work,counts=result
            work=self._checkpoint_work(organization,tile)
            fields.append(field)
            if controls is not None:temps.append(temp)
            total_work=work if total_work is None else total_work+work
            total_counts=counts if total_counts is None else total_counts+counts
        if self.training and torch.is_grad_enabled() and total_work is not None and self.routing_mode!='dense':
            self._record_compact_work(organization,total_work)
        field=torch.cat(fields,1) if fields else organization.module_states.new_zeros(query.batch_size,0,self.output_dim)
        temp=torch.cat(temps,1) if controls is not None and temps else None
        counts=self._work_count_dict(total_counts,query)
        return CoreOutput(nonlinear_field=field,thermal_temperature=temp,thermal_basis=None,trace=None,work_counts=counts)

    set_training_clock=JointNativeCore.set_training_clock
    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)
        self._joint_epoch=int(self.joint_epoch);self._joint_update_index=int(self.joint_update_index)
        self._joint_macro_clock=int(self.joint_macro_clock)
        self._entmax_temperature_value=float(state_dict.get(prefix+'router.entmax_temperature',self.router.entmax_temperature).detach().cpu())
    def joint_begin_update(self,epoch,update_index):
        self._joint_epoch=int(epoch);self._joint_update_index=int(update_index)
        self.joint_epoch.fill_(epoch);self.joint_update_index.fill_(update_index);self.joint_begin_microbatch()
        state=self.joint_budget.set_clock(epoch,update_index)
        if self.routing_mode=='dense':state['constraints_enabled']=False
        return state
    def joint_begin_microbatch(self):self._joint_entries=[]
    def joint_microbatch_result(self):return microbatch_result(self)
    def joint_constraint_kinds(self):return ('M','E')
    def joint_constraint_duals(self):return self.joint_budget.duals
    def joint_after_update(self,epoch,update_index,residuals):
        self.joint_budget.set_clock(epoch,update_index)
        if self.routing_mode=='dense':return {}
        for kind in ('M','E'):
            dual=self.joint_budget.duals[kind]
            dual.observe_residual(residuals[kind],int(dual.macro_step)+1)
        self._joint_macro_clock+=1;self.joint_macro_clock.fill_(self._joint_macro_clock)
        return {}
