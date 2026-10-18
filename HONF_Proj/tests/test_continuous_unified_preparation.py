"""Round10 applied contextual paths, local-input capability and shortcut guards."""
from dataclasses import replace
import copy

import pytest
import torch
from torch import nn

from honf_forward_core.content_hypergraph.staged_attention_v2 import SpatialRouteAttentionV2
from honf_forward_core.content_hypergraph.model_v2 import ContentHypergraphCoreV2
from honf_forward_core.continuous_joint.preparation import FrozenContextPlan
from honf_forward_core.continuous_joint.unified import AllOpenContextPolicy, UnifiedPreparationMixin, SUMMARY_ROOTS
from tests.test_content_hypergraph_v2_core import _scene, _core, _query


class ConstantPolicy(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, raw_logits, allowed, route_name, **metadata):
        return allowed.to(raw_logits.dtype) * self.value


def run_attention(attention, scene, states, policy, **options):
    return attention(states, states, scene.module_coordinates, scene.module_coordinates,
        scene.module_characteristic_lengths, scene.module_characteristic_lengths,
        scene.module_measure, scene.module_measure_reference, scene.module_mask, scene.module_mask,
        scene.coordinate_scale, remove_self=True, route_name='mm', stage_identity='cycle0/mm',
        destination_ids=scene.module_ids, donor_ids=scene.module_ids,
        value_policy=policy, **options)


def double_scene(scene):
    return replace(scene, **{key:value.double() for key,value in vars(scene).items()
                             if isinstance(value, torch.Tensor) and value.is_floating_point()})


def test_open_context_preserves_sdpa_and_closed_context_has_no_bias():
    scene = double_scene(_scene(1))
    layer = SpatialRouteAttentionV2(32, 4, 2).double().eval()
    states = torch.randn(1, 3, 32, dtype=torch.double, requires_grad=True)
    expected = run_attention(layer, scene, states, None, return_attention=True)
    opened = run_attention(layer, scene, states, AllOpenContextPolicy(), return_attention=True)
    torch.testing.assert_close(opened[0], expected[0], rtol=3e-13, atol=3e-13)
    torch.testing.assert_close(opened[1], expected[1], rtol=3e-13, atol=3e-13)
    zero = run_attention(layer, scene, states, ConstantPolicy(0.0))[0]
    assert zero.count_nonzero() == 0
    assert torch.autograd.grad(zero.sum(), layer.output.bias)[0].count_nonzero() == 0


def test_context_gate_gradient_and_frozen_full_plan_component_isolation():
    scene = double_scene(_scene(1))
    layer = SpatialRouteAttentionV2(32, 4, 2).double().eval()
    states = torch.randn(1, 3, 32, dtype=torch.double, requires_grad=True)
    class ParameterPolicy(nn.Module):
        def __init__(self):
            super().__init__(); self.gate = nn.Parameter(torch.tensor(.3, dtype=torch.double))
        def forward(self, logits, allowed, route_name, **metadata):
            return self.gate * allowed
    policy = ParameterPolicy()
    live, coefficients = run_attention(layer,scene,states,policy,return_attention=True)
    derivative = torch.autograd.grad(live.square().sum(), policy.gate)[0]
    step = 1e-5
    with torch.no_grad():
        policy.gate.add_(step); plus = run_attention(layer,scene,states,policy)[0].square().sum()
        policy.gate.sub_(2*step); minus = run_attention(layer,scene,states,policy)[0].square().sum()
        policy.gate.add_(step)
    torch.testing.assert_close(derivative,(plus-minus)/(2*step),rtol=2e-7,atol=2e-8)
    # Two disconnected value components, with fully frozen normalization.
    gates = torch.zeros(1,3,3,dtype=torch.double)
    gates[0,0,1] = gates[0,1,0] = .4
    coeff = coefficients.permute(0,2,3,1).detach().clone() * (gates > 0)[...,None]
    plan = FrozenContextPlan(gates, coeff, coeff.sum(2))
    class Frozen(nn.Module):
        def forward(self,*args,**kwargs):return plan
    frozen = Frozen()
    original = run_attention(layer,scene,states,frozen)[0]
    perturbed = states.detach().clone();perturbed[0,0] += 100
    changed = run_attention(layer,scene,perturbed,frozen)[0]
    assert torch.equal(original[0,2],changed[0,2])
    assert not torch.equal(original[0,1],changed[0,1])
    # Live dense routing separately exposes control/value dependency.
    before = run_attention(layer,scene,states,AllOpenContextPolicy())[0]
    after = run_attention(layer,scene,perturbed,AllOpenContextPolicy())[0]
    assert not torch.equal(before[0,2],after[0,2])


@pytest.mark.parametrize('dataset',['thermal','wind'])
def test_local_input_named_drops_and_physical_metadata_preserved(dataset):
    if dataset == 'thermal':
        from channelthermal.local_records import local_records_scene, feature_contract
        width, operating, dropped = 8,14,[10,11]
    else:
        from windfarm.local_records import local_records_scene, feature_contract
        width, operating, dropped = 7,11,[3]
    scene = replace(_scene(1),environment_features=torch.randn(1,4,width),
                    operating_features=torch.arange(operating,dtype=torch.float)[None])
    original = scene.environment_features.clone()
    local = local_records_scene(scene)
    assert local.operating_features[...,dropped].count_nonzero() == 0
    for name in ('module_features','module_coordinates','module_measure','module_mask',
                 'coordinate_scale','environment_coordinates','environment_measure'):
        assert getattr(local,name) is getattr(scene,name)
    assert torch.equal(scene.environment_features,original)
    assert feature_contract()['schema'] == 'local_records_context_v1'
    assert len(feature_contract()['sha256']) == 64
    if dataset == 'thermal':
        assert local.environment_features[...,2:4].count_nonzero() == 0
        torch.testing.assert_close(local.environment_features[...,[0,1,4,5,6,7]],
                                   original[...,[0,1,4,5,6,7]])
    else:
        assert local.environment_features is scene.environment_features


def UnifiedContextFixture(routing_mode='dense'):
    from honf_forward_core.joint_native.model import JointNativeCore
    from honf_forward_core.continuous_joint.unified import UnifiedJointCore
    parent=JointNativeCore.from_a2d(_core('A2-D',environment_input_dim=8,operating_input_dim=14))
    return UnifiedJointCore.from_parent(parent,input_dataset='thermal',routing_mode=routing_mode,
                                       checkpoint_pipeline=False)


def test_unified_each_stage_applies_gates_and_summary_modules_are_absent():
    core=UnifiedContextFixture().double().eval()
    scene=replace(double_scene(_scene(1)),environment_features=torch.randn(1,4,8,dtype=torch.double),
                  operating_features=torch.randn(1,14,dtype=torch.double))
    assert not any(name.split('.')[0] in SUMMARY_ROOTS for name in core.state_dict())
    def forbidden(*args,**kwargs):raise AssertionError('summary executed')
    core._scene_summary=forbidden
    with torch.no_grad():
        prepared=core.prepare(scene,return_trace=True)
    assert prepared.scene_summary is None
    assert prepared.work_counts['scene_summary_module_entries']==0
    assert prepared.work_counts['scene_reinjection_rows']==0
    assert [p['stage'] for p in prepared.context_value_provenance] == [
        f'cycle{cycle}/{route}' for cycle in range(2) for route in ('mm','em','ee','me')]
    for stage in prepared.applied_context_stages:
        for route in ('mm','em','ee','me'):
            snapshot=stage[route+'_native_route_snapshot']
            assert torch.equal(snapshot.applied_incidence,snapshot.allowed.to(torch.double))
            snapshot.assert_current()
    before=prepared.module_states.clone()
    with torch.no_grad():repeated=core.prepare(scene)
    assert torch.equal(before,repeated.module_states)
    assert repeated.applied_context_stages == ()
    assert repeated.context_value_provenance == ()
    assert len(repeated.context_stage_metadata) == 8
    with torch.no_grad():scene.operating_features.add_(1)
    with pytest.raises(ValueError,match='Raw local-input'):
        core._validate_prepared_state(prepared)


def test_applied_incidence_mutation_is_rejected():
    scene=double_scene(_scene(1));layer=SpatialRouteAttentionV2(32,4,2).double()
    states=torch.randn(1,3,32,dtype=torch.double)
    _,_,snapshot=run_attention(layer,scene,states,ConstantPolicy(.5),capture_native_route_snapshot=True)
    with torch.no_grad():snapshot.applied_incidence.add_(.01)
    with pytest.raises(ValueError,match='changed in place'):snapshot.assert_current()


def test_open_context_gradients_empty_donors_and_bias_recover_legacy():
    scene=double_scene(_scene(1));layer=SpatialRouteAttentionV2(32,4,2).double().eval()
    states=torch.randn(1,3,32,dtype=torch.double,requires_grad=True)
    expected=run_attention(layer,scene,states,None)[0]
    actual=run_attention(layer,scene,states,AllOpenContextPolicy())[0]
    parameters=(states,*layer.parameters())
    old=torch.autograd.grad(expected.square().sum(),parameters,retain_graph=True,allow_unused=True)
    new=torch.autograd.grad(actual.square().sum(),parameters,retain_graph=True,allow_unused=True)
    for left,right in zip(old,new):
        assert (left is None)==(right is None)
        if left is not None:torch.testing.assert_close(left,right,rtol=5e-12,atol=5e-12)
    absent=replace(scene,module_measure=scene.module_measure*0)
    assert run_attention(layer,absent,states,None)[0].count_nonzero()==0
    assert run_attention(layer,absent,states,AllOpenContextPolicy())[0].count_nonzero()==0
    assert run_attention(layer,absent,states,ConstantPolicy(.7))[0].count_nonzero()==0


@pytest.mark.parametrize('mode',['dense','hypergraph','direct'])
def test_unified_shared_parent_native_law_and_exact_applied_directory(mode):
    core=UnifiedContextFixture(mode).double().eval()
    scene=replace(double_scene(_scene(1)),environment_features=torch.randn(1,4,8,dtype=torch.double),
                  operating_features=torch.randn(1,14,dtype=torch.double))
    query=_query(1)
    query=replace(query,coordinates=query.coordinates.double(),features=query.features.double())
    heat=torch.tensor([[1.,-.5,2.]],dtype=torch.double)
    with torch.no_grad():
        prepared=core.prepare(scene,return_trace=True)
        actual=core.decode(prepared,query,heat)
        zero=core.decode(prepared,query,heat*0)
        kernel=core.export_source_kernels(prepared,query)
        torch.testing.assert_close(actual.thermal_temperature,torch.einsum('bqm,bm->bq',kernel,heat),rtol=5e-12,atol=5e-12)
        assert zero.thermal_temperature.count_nonzero()==0
        assert torch.equal(actual.nonlinear_field,zero.nonlinear_field)
        if mode=='hypergraph':
            for route in ('mm','me','em','ee'):
                applied=prepared.applied_context_stages[-1][route+'_applied_incidence']
                retrieved=prepared.continuous_typed_incidence[route]
                if route[0]!=route[1]:assert torch.equal(applied,retrieved)
                else:
                    diagonal=torch.eye(applied.shape[-1],dtype=torch.bool)[None]
                    assert torch.equal(applied.masked_select(~diagonal),retrieved.masked_select(~diagonal))
                    assert (retrieved.diagonal(dim1=-2,dim2=-1)==1).all()
        restored=type(core).from_config(core.get_config()).double().eval()
        restored.load_state_dict(core.state_dict(),strict=True)
        again=restored.decode(restored.prepare(scene),query,heat)
        torch.testing.assert_close(again.thermal_temperature,actual.thermal_temperature,rtol=0,atol=0)
    inventory=core.parent_initialization_inventory
    assert all(name.split('.')[0] in SUMMARY_ROOTS for name in inventory['dropped_summary_state_names'])
    assert set(inventory['dropped_optimizer_parameter_names'])==set(core.dropped_parent_parameter_names)


def test_context_schedule_and_serialized_dual_is_used_in_macro_objective():
    from honf_forward_core.continuous_joint.unified import context_schedule
    from honf_runtime.joint_native_objective import JointMacroObjective
    assert context_schedule(3000)[0]==1.
    assert context_schedule(3001)[0]==.98
    assert context_schedule(4000)[0]==pytest.approx(.90)
    assert context_schedule(5000)[0]==pytest.approx(.90)
    core=UnifiedContextFixture('direct').double().train()
    for dual in core.joint_budget.duals.values():dual.set_calibrated_lambda0(.1)
    optimizer=torch.optim.AdamW(core.parameters(),lr=1e-4)
    objective=JointMacroObjective(core,optimizer,4000,0)
    objective.begin_microbatch()
    scene=replace(double_scene(_scene(1)),environment_features=torch.randn(1,4,8,dtype=torch.double),
                  operating_features=torch.randn(1,14,dtype=torch.double))
    query=_query(1);query=replace(query,coordinates=query.coordinates.double(),features=query.features.double())
    prepared=core.prepare(scene)
    field=core.decode(prepared,query).nonlinear_field
    objective.observe(expected_case_count=1)
    field.square().sum().backward()
    objective.apply();objective.after_update()
    assert 'joint_work_penalty_context' in objective.losses
    assert core.joint_budget.duals['context'].macro_step==1
    assert core._context_entries==[]
    assert core.router.incidence.routes['mm'].bias.grad is not None
    restored=type(core).from_config(core.get_config()).double()
    restored.load_state_dict(core.state_dict())
    assert restored.joint_budget.duals['context'].macro_step==1


def test_unified_trace_budget_and_saved_feature_hash_are_enforced():
    core=UnifiedContextFixture('dense')
    scene=replace(_scene(1),environment_features=torch.randn(1,4,8),operating_features=torch.randn(1,14))
    with pytest.raises(ValueError,match='element budget'):
        core.prepare(scene,trace_request={'include':['module_states'],'max_elements':10})
    config=core.get_config();config['input_contract_sha256']='wrong'
    with pytest.raises(ValueError,match='schema hash'):type(core).from_config(config)


@pytest.mark.parametrize('bad_kinds',[('M','E','context','context'),('M',),('M','E',[]), 'M/E'])
def test_macro_rejects_invalid_declared_constraint_capability(bad_kinds):
    from honf_runtime.joint_native_objective import JointMacroObjective
    core=UnifiedContextFixture('direct')
    core.joint_constraint_kinds=lambda:bad_kinds
    with pytest.raises(ValueError,match='constraint'):
        JointMacroObjective(core,torch.optim.AdamW(core.parameters()),4000,0)


def test_macro_rejects_missing_declared_context_and_extra_legacy_constraint():
    from honf_runtime.joint_native_objective import JointMacroObjective
    core=UnifiedContextFixture('direct')
    objective=JointMacroObjective(core,torch.optim.AdamW(core.parameters()),4000,0)
    core.joint_microbatch_result=lambda:{'constraints':{'M':{},'E':{}}}
    with pytest.raises(ValueError,match='kinds differ'):objective.observe()
    # The unchanged default remains exactly M/E when no extra capability exists.
    from honf_forward_core.joint_native.model import JointNativeCore
    legacy=JointNativeCore.from_a2d(_core('A2-D'),routing_mode='joint')
    objective=JointMacroObjective(legacy,torch.optim.AdamW(legacy.parameters()),4000,0)
    legacy.joint_microbatch_result=lambda:{'constraints':{'M':{},'E':{},'context':{}}}
    with pytest.raises(ValueError,match='kinds differ'):objective.observe()


def test_frozen_complete_context_plan_blocks_cross_component_values_at_every_stage():
    core=UnifiedContextFixture('direct').double().eval()
    scene=replace(double_scene(_scene(1)),environment_features=torch.randn(1,4,8,dtype=torch.double),
                  operating_features=torch.randn(1,14,dtype=torch.double))
    class Components(nn.Module):
        def forward(self,raw_logits,allowed,route_name,**metadata):
            # M13/M29 and E101/E107 form A; M47 and E131/E149 form B.
            destinations=metadata['destination_ids']
            donors=metadata['donor_ids']
            def component(ids):return (ids==47)|(ids==131)|(ids==149)
            return allowed * (component(destinations)[...,None]==component(donors)[:,None,:])*.4
    core.router.incidence=Components()
    include=[f'cycle{cycle}/{route}_attention' for cycle in range(2) for route in ('mm','em','ee','me')]
    with torch.no_grad():
        original=core.prepare(scene,return_trace=True,trace_request={'include':include})
    plans={}
    for stage in original.applied_context_stages:
        for route in ('mm','em','ee','me'):
            snapshot=stage[route+'_native_route_snapshot']
            coefficients=original.trace[snapshot.stage_identity+'_attention'].permute(0,2,3,1).detach()
            plans[snapshot.stage_identity]=FrozenContextPlan(snapshot.applied_incidence.detach(),coefficients,
                                                           coefficients.sum(2))
    class FrozenStages(nn.Module):
        # Even an accidentally retained marker cannot enable the all-open
        # branch while a complete FrozenContextPlan is active.
        all_open=True
        def forward(self,*args,**metadata):return plans[metadata['stage_identity']]
    core.router.incidence=FrozenStages()
    changed_features=scene.module_features.clone();changed_features[0,0]+=3.
    perturbed=replace(scene,module_features=changed_features)
    with torch.no_grad():
        frozen=core.prepare(perturbed,return_trace=True)
    assert torch.equal(original.raw_module_states[:,2],frozen.raw_module_states[:,2])
    assert torch.equal(original.raw_environment_states[:,2:],frozen.raw_environment_states[:,2:])
    for cycle in range(2):
        # Each attention result is recorded before the next typed stage.
        for route in ('mm','em','ee','me'):
            assert torch.equal(original.applied_context_stages[cycle][route+'_applied_incidence'],
                               frozen.applied_context_stages[cycle][route+'_applied_incidence'])
    assert torch.equal(original.module_states[:,2],frozen.module_states[:,2])
    assert torch.equal(original.environment_states[:,2:],frozen.environment_states[:,2:])


def test_frozen_context_and_reader_plan_isolates_E_M_messages_and_final_field():
    core=UnifiedContextFixture('direct').double().eval()
    scene=replace(double_scene(_scene(1)),environment_features=torch.randn(1,4,8,dtype=torch.double),
                  operating_features=torch.randn(1,14,dtype=torch.double))
    class Components(nn.Module):
        def forward(self,raw_logits,allowed,route_name,**metadata):
            def component(ids):return (ids==47)|(ids==131)|(ids==149)
            return allowed*(component(metadata['destination_ids'])[...,None]==component(metadata['donor_ids'])[:,None,:])*.4
    core.router.incidence=Components()
    include=[f'cycle{cycle}/{route}_attention' for cycle in range(2) for route in ('mm','em','ee','me')]
    with torch.no_grad():original=core.prepare(scene,return_trace=True,trace_request={'include':include})
    plans={}
    for stage in original.applied_context_stages:
        for route in ('mm','em','ee','me'):
            snapshot=stage[route+'_native_route_snapshot']
            coeff=original.trace[snapshot.stage_identity+'_attention'].permute(0,2,3,1).detach()
            plans[snapshot.stage_identity]=FrozenContextPlan(snapshot.applied_incidence.detach(),coeff,coeff.sum(2))
    class Frozen(nn.Module):
        def forward(self,*args,**metadata):return plans[metadata['stage_identity']]
    core.router.incidence=Frozen()
    query=_query(1)
    query=replace(query,coordinates=query.coordinates[:,2:3].double(),features=query.features[:,2:3].double(),
                  query_ids=query.query_ids[:,2:3])
    with torch.no_grad():
        plan=core.capture_continuous_plan(original,query)
        gates=plan.gates.clone();gates[...,:2]=0;gates[...,3:5]=0
        emass=plan.environment_coefficients.clone();emass[:,:,:2]=0
        mmass=plan.module_coefficients.clone();mmass[:,:,:2]=0
        plan=replace(plan,gates=gates,environment_coefficients=emass,module_coefficients=mmass)
        features=scene.module_features.clone();features[0,0]+=3.
        changed=core.prepare(replace(scene,module_features=features))
        heat=torch.ones(1,3,dtype=torch.double)
        options={'trace_request':{'include':['joint_environment_context','joint_conditioned_query','direct_source_messages']}}
        before=core.decode_with_plan(original,query,plan,heat,**options)
        after=core.decode_with_plan(changed,query,plan,heat,**options)
    assert torch.equal(before.nonlinear_field,after.nonlinear_field)
    assert torch.equal(before.thermal_temperature,after.thermal_temperature)
    for name in ('joint_environment_context','joint_conditioned_query','direct_source_messages'):
        assert torch.equal(before.trace[name],after.trace[name])


def test_public_context_plan_binds_owner_ids_normalizer_and_full_coefficients():
    core=UnifiedContextFixture('direct').double().eval()
    scene=replace(double_scene(_scene(1)),environment_features=torch.randn(1,4,8,dtype=torch.double),
                  operating_features=torch.randn(1,14,dtype=torch.double))
    class Components(nn.Module):
        def forward(self,raw_logits,allowed,route_name,**metadata):
            def component(ids):return (ids==47)|(ids==131)|(ids==149)
            return allowed*(component(metadata['destination_ids'])[...,None]==component(metadata['donor_ids'])[:,None,:])*.4
    core.router.incidence=Components()
    plan=core.capture_unified_context_plan(scene)
    with torch.no_grad():original=core.prepare_with_context_plan(scene,plan)
    features=scene.module_features.clone();features[0,0]+=2
    changed=replace(scene,module_features=features)
    with torch.no_grad():frozen=core.prepare_with_context_plan(changed,plan)
    assert torch.equal(original.module_states[:,2],frozen.module_states[:,2])
    assert torch.equal(original.environment_states[:,2:],frozen.environment_states[:,2:])
    assert core._unified_context_plan is None
    assert len(plan.stages)==8
    other=UnifiedContextFixture('direct').double().eval()
    with pytest.raises(ValueError,match='owner'):other.prepare_with_context_plan(scene,plan)
    with pytest.raises(ValueError,match='normalizer'):
        core.prepare_with_context_plan(replace(scene,normalizer_key='other'),plan)
    with torch.no_grad():plan.stages['cycle0/mm'].coefficients.add_(1e-4)
    with pytest.raises(ValueError,match='changed after capture'):core.prepare_with_context_plan(scene,plan)


def test_public_all_open_context_plan_replays_SDPA_with_declared_fp64_envelope():
    core=UnifiedContextFixture('dense').double().eval()
    scene=replace(double_scene(_scene(1)),environment_features=torch.randn(1,4,8,dtype=torch.double),
                  operating_features=torch.randn(1,14,dtype=torch.double))
    with torch.no_grad():
        ordinary=core.prepare(scene)
        plan=core.capture_unified_context_plan(scene)
        frozen=core.prepare_with_context_plan(scene,plan)
        torch.testing.assert_close(frozen.module_states,ordinary.module_states,rtol=5e-12,atol=5e-12)
        torch.testing.assert_close(frozen.environment_states,ordinary.environment_states,rtol=5e-12,atol=5e-12)
        query=_query(1);query=replace(query,coordinates=query.coordinates.double(),features=query.features.double())
        heat=torch.ones(1,3,dtype=torch.double)
        before=core.decode(ordinary,query,heat);after=core.decode(frozen,query,heat)
        torch.testing.assert_close(after.nonlinear_field,before.nonlinear_field,rtol=5e-12,atol=5e-12)
        torch.testing.assert_close(after.thermal_temperature,before.thermal_temperature,rtol=5e-12,atol=5e-12)


def test_U_D_FP32_frozen_SDPA_replays_actual_controls_reads_fresh_values_and_rejects_mutation():
    from honf_forward_core.continuous_joint.preparation import FrozenSDPAContextPlan
    core=UnifiedContextFixture('dense').float().eval()
    scene=replace(_scene(1),environment_features=torch.randn(1,4,8),operating_features=torch.randn(1,14))
    with torch.no_grad():
        ordinary=core.prepare(scene)
        plan=core.capture_unified_context_plan(scene)
        assert all(isinstance(stage,FrozenSDPAContextPlan) for stage in plan.stages.values())
        replay=core.prepare_with_context_plan(scene,plan)
        assert all(row['control_origin']=='captured_SDPA_Q_K_additive_mask'
                   for row in replay.context_stage_metadata)
        assert torch.equal(ordinary.module_states,replay.module_states)
        assert torch.equal(ordinary.environment_states,replay.environment_states)
        changed=scene.module_features.clone();changed[:,0,0]+=.25
        changed_scene=replace(scene,module_features=changed)
        fresh=core.prepare_with_context_plan(changed_scene,plan)
        assert not torch.equal(ordinary.module_states,fresh.module_states)
        plan.stages['cycle0/mm'].query.add_(1e-4)
    with pytest.raises(ValueError,match='changed after capture'):
        core.prepare_with_context_plan(scene,plan)


def test_frozen_SDPA_context_plan_budget_is_explicit_and_checked_before_prepare():
    core=UnifiedContextFixture('dense').float().eval()
    scene=replace(_scene(1),environment_features=torch.randn(1,4,8),operating_features=torch.randn(1,14))
    with pytest.raises(ValueError,match='element budget'):
        core.capture_unified_context_plan(scene,max_elements=1)
    with pytest.raises(ValueError,match='bounded element budget'):
        core.capture_unified_context_plan(scene,max_elements=1_000_001)


def test_frozen_SDPA_controls_preserve_fresh_V_gradients_and_closed_component():
    from honf_forward_core.continuous_joint.preparation import FrozenSDPAContextPlan
    from torch.nn import functional as F
    scene=_scene(1)
    layer=SpatialRouteAttentionV2(32,4,2).float().eval()
    states=torch.randn(1,3,32,requires_grad=True)
    with torch.no_grad():
        _,_,snapshot=run_attention(layer,scene,states,AllOpenContextPolicy(),
            return_attention=True,capture_native_route_snapshot=True)
        gates=torch.zeros(1,3,3);gates[0,0,1]=gates[0,1,0]=1
        present=gates.bool().any(-1)
        measure=torch.where(snapshot.donor_measure>0,snapshot.donor_measure,1.)
        additive=snapshot.geometry_bias+measure.log()[:,None,None]
        additive=additive.masked_fill(~gates.bool()[:,None],-torch.inf)
        additive=torch.where(present[:,None,:,None],additive,0.)
        plan=FrozenSDPAContextPlan(gates,snapshot.query_scaled.detach().clone(),
            snapshot.key_normalized.detach().clone(),additive,present,
            present[...,None].expand(-1,-1,layer.heads).float())
    class Frozen(nn.Module):
        def forward(self,*args,**kwargs):return plan
    actual=run_attention(layer,scene,states,Frozen())[0]
    values=layer.value(layer.key_norm(states)).view(1,3,layer.heads,layer.head_width).transpose(1,2)
    attended=F.scaled_dot_product_attention(plan.query,plan.key,values,
        attn_mask=plan.additive_mask,dropout_p=0.,scale=1.)
    expected=layer.output(attended.transpose(1,2).reshape(1,3,layer.width))*present[...,None]
    assert torch.equal(actual,expected)
    actual_grad=torch.autograd.grad(actual.sum(),states,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected.sum(),states,retain_graph=True)[0]
    assert torch.equal(actual_grad,expected_grad)
    qgrad,kgrad=torch.autograd.grad(actual.sum(),(layer.query.weight,layer.key.weight),
        allow_unused=True,retain_graph=True)
    assert qgrad is None and kgrad is None
    changed=states.detach().clone();changed[:,0]+=100
    perturbed=run_attention(layer,scene,changed,Frozen())[0]
    assert torch.equal(actual[:,2],perturbed[:,2]) and not torch.equal(actual[:,1],perturbed[:,1])


@pytest.mark.parametrize('mode',['dense','hypergraph','direct'])
def test_unified_empty_environment_has_zero_E_values_and_finite_native_fields(mode):
    core=UnifiedContextFixture(mode).double().eval()
    scene=replace(double_scene(_scene(1)),environment_features=torch.empty(1,0,8,dtype=torch.double),
        environment_coordinates=torch.empty(1,0,2,dtype=torch.double),environment_mask=torch.empty(1,0,dtype=torch.bool),
        environment_measure=torch.empty(1,0,dtype=torch.double),environment_ids=torch.empty(1,0,dtype=torch.long),
        environment_characteristic_lengths=torch.empty(1,0,2,dtype=torch.double),operating_features=torch.randn(1,14,dtype=torch.double))
    query=_query(1);query=replace(query,coordinates=query.coordinates.double(),features=query.features.double())
    with torch.no_grad():
        output=core.decode(core.prepare(scene),query,torch.ones(1,3,dtype=torch.double))
    assert torch.isfinite(output.nonlinear_field).all() and torch.isfinite(output.thermal_temperature).all()
    assert output.work_counts['joint_E_value_rows']==0
