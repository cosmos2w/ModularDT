"""Numerical contracts of the continuous original-record value function."""
import copy
from dataclasses import replace
import pytest
import torch
from honf_forward_core.continuous_joint.primitives import positive_quadratic_gate,entmax15,guarded_weights
from honf_forward_core.continuous_joint.model import ContinuousJointCore
from honf_forward_core.joint_native.model import JointNativeCore
from tests.test_content_hypergraph_v2_core import _core,_scene,_query
from tests.test_budgeted_native_core import _double


def _reference(scores,mask):
    rows=[]
    for s,m in zip(scores.reshape(-1,scores.shape[-1]),mask.reshape(-1,mask.shape[-1])):
        if not m.any():rows.append(torch.zeros_like(s));continue
        lo=s[m].min()-2;hi=s[m].max()
        for _ in range(100):
            tau=(lo+hi)/2
            mass=((s-tau).clamp_min(0)/2).square()*m
            if mass.sum()>1:lo=tau
            else:hi=tau
        rows.append(((s-(lo+hi)/2).clamp_min(0)/2).square()*m)
    return torch.stack(rows).reshape_as(scores)


def test_entmax_independent_reference_ties_masks_and_support_gradients():
    scores=torch.tensor([[0.,0.,-4.,2.],[1.,1.,1.,1.],[1.,2.,3.,4.]],dtype=torch.double,requires_grad=True)
    mask=torch.tensor([[1,1,0,1],[1,1,1,1],[0,0,0,0]],dtype=torch.bool)
    actual=entmax15(scores,mask)
    torch.testing.assert_close(actual,_reference(scores.detach(),mask),rtol=2e-13,atol=2e-13)
    torch.testing.assert_close(actual.sum(-1),torch.tensor([1.,1.,0.],dtype=torch.double))
    assert actual[0,0]==actual[0,1] and actual[1].unique().numel()==1
    assert torch.isfinite(torch.autograd.grad(actual.square().sum(),scores)[0]).all()
    torch.autograd.gradcheck(lambda x:entmax15(x,mask),scores,atol=2e-6,rtol=2e-5)


def test_gate_at_zero_stability_guarded_empty_and_allopen():
    x=torch.tensor([-1.,0.,1e-6,1.,1e200],dtype=torch.double,requires_grad=True)
    g=positive_quadratic_gate(x)
    assert torch.isfinite(g).all() and g[0]==g[1]==0 and g[-1]==1
    derivative=torch.autograd.grad(g.sum(),x)[0]
    assert derivative[0]==derivative[1]==0 and torch.isfinite(derivative).all()
    scores=torch.randn(2,2,3,4,dtype=torch.double)
    measure=torch.tensor([[1.,0.,2.,1.],[0.,0.,0.,0.]],dtype=torch.double)
    valid=torch.ones((2,4),dtype=torch.bool);qmask=torch.ones((2,3),dtype=torch.bool)
    c,mass=guarded_weights(scores,torch.ones(2,3,4,dtype=torch.double),measure,valid,qmask)
    torch.testing.assert_close(c[0].sum(1),torch.ones((3,2),dtype=torch.double))
    assert c[1].count_nonzero()==mass[1].count_nonzero()==0
    zero,_=guarded_weights(scores,torch.zeros(2,3,4,dtype=torch.double),measure,valid,qmask)
    assert zero.count_nonzero()==0


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_guarded_allopen_coefficients_are_exact_parent_typed_arithmetic(dtype):
    from honf_forward_core.joint_native.readers import _multiplicative_weights
    generator=torch.Generator().manual_seed(23)
    scores=torch.randn((2,4,5,3),generator=generator,dtype=dtype)
    measure=torch.tensor([[0.,.003,.008],[.004,.002,0.]],dtype=dtype)
    reference=torch.tensor([.002,.005],dtype=dtype)
    valid=torch.ones((2,3),dtype=torch.bool)
    qmask=torch.ones((2,5),dtype=torch.bool);qmask[1,-1]=False
    gates=torch.ones((2,5,3),dtype=dtype)
    expected=_multiplicative_weights(scores,gates,measure,valid,qmask,reference)
    actual,mass=guarded_weights(scores,gates,measure,valid,qmask,reference=reference)
    assert torch.equal(actual,expected)
    assert torch.equal(mass,qmask[...,None].expand_as(mass).to(dtype))


@pytest.mark.parametrize('dtype,scale,reference',[
    (torch.float32,1e-30,1e30),(torch.float32,1e30,1e-30),
    (torch.float64,1e-300,1e300),(torch.float64,1e300,1e-300)])
def test_guarded_extreme_measure_fallback_preserves_finite_input_vjp(dtype,scale,reference):
    scores=torch.tensor([[[[.5,-.3]]]],dtype=dtype,requires_grad=True)
    measure=torch.tensor([[scale,2*scale]],dtype=dtype,requires_grad=True)
    ref=torch.tensor([reference],dtype=dtype,requires_grad=True)
    gates=torch.tensor([[[0.,.65]]],dtype=dtype,requires_grad=True)
    alpha,mass=guarded_weights(scores,gates,measure,torch.ones((1,2),dtype=torch.bool),
        torch.ones((1,1),dtype=torch.bool),reference=ref)
    loss=(alpha*alpha.new_tensor([2.,3.])[None,None,:,None]).sum()+mass.sum()
    gradients=torch.autograd.grad(loss,(scores,measure,ref,gates))
    assert torch.isfinite(alpha).all() and torch.isfinite(mass).all()
    assert all(torch.isfinite(g).all() for g in gradients)


def _model(mode='dense',checkpoint=False,tile=2):
    parent=JointNativeCore.from_a2d(_core('A2-D').double()).double()
    with torch.no_grad():parent.environment_conditioning.weight.copy_(torch.eye(parent.context_dim,dtype=torch.double)*.2)
    return parent,ContinuousJointCore.from_parent(parent,routing_mode=mode,checkpoint_pipeline=checkpoint,query_tile_size=tile).double()


def test_allopen_parent_fields_source_k_physical_laws_and_common_gradients():
    parent,model=_model();parent.eval();model.eval()
    scene,query=_double(_scene(1)),_double(_query(1))
    heat=torch.tensor([[1.,-.5,2.]],dtype=torch.double)
    a=parent.decode(parent.prepare(scene),query,heat)
    b=model.decode(model.prepare(scene),query,heat)
    torch.testing.assert_close(a.nonlinear_field,b.nonlinear_field,rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(a.thermal_temperature,b.thermal_temperature,rtol=2e-12,atol=2e-12)
    ka=parent.export_source_kernels(parent.prepare(scene),query)
    kb=model.export_source_kernels(model.prepare(scene),query)
    torch.testing.assert_close(ka,kb,rtol=2e-12,atol=2e-12)
    def gradients(core,output):
        loss=output.nonlinear_field.square().sum()+output.thermal_temperature.square().sum()
        return dict(zip(dict(core.named_parameters()),torch.autograd.grad(loss,tuple(core.parameters()),allow_unused=True)))
    ga,gb=gradients(parent,a),gradients(model,b)
    for name in ga:
        if ga[name] is not None:
            torch.testing.assert_close(ga[name],gb[name],rtol=2e-10,atol=2e-12,msg=name)
    with torch.no_grad():
        org=model.prepare(scene)
        zero=model.decode(org,query,heat*0)
        doubled=model.decode(org,query,heat*2)
        assert zero.thermal_temperature.count_nonzero()==0
        torch.testing.assert_close(doubled.thermal_temperature,2*b.thermal_temperature)
        torch.testing.assert_close(zero.nonlinear_field,b.nonlinear_field)


@pytest.mark.parametrize('mode',['hypergraph','direct'])
def test_dense_packed_checkpointed_gradients_and_actual_rows(mode):
    _,packed=_model(mode);packed.train();packed.joint_begin_update(3400,0)
    with torch.no_grad():
        for calibration in packed.router.incidence.routes.values():calibration.bias.fill_(.1)
        packed.router.direct_parameters.module.bias.fill_(.1)
        packed.router.direct_parameters.environment.bias.fill_(.1)
    dense=copy.deepcopy(packed);dense.native_backend='dense'
    recompute=copy.deepcopy(packed);recompute.checkpoint_pipeline=True;recompute.query_tile_size=1
    scene,query=_double(_scene(1)),_double(_query(1));heat=torch.tensor([[1.,-.5,2.]],dtype=torch.double)
    models=[packed,dense,recompute];outputs=[];grads=[]
    for model in models:
        output=model.decode(model.prepare(scene),query,heat)
        work=model.joint_microbatch_result()['constraints']
        loss=output.nonlinear_field.square().sum()+output.thermal_temperature.square().sum()+sum(r['residual_numerator'] for r in work.values())
        grads.append(torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True));outputs.append(output)
    for output in outputs[1:]:
        torch.testing.assert_close(output.nonlinear_field,outputs[0].nonlinear_field,rtol=2e-12,atol=2e-12)
        torch.testing.assert_close(output.thermal_temperature,outputs[0].thermal_temperature,rtol=2e-12,atol=2e-12)
    for gs in grads[1:]:
        for a,b in zip(gs,grads[0]):
            if a is not None and b is not None:torch.testing.assert_close(a,b,rtol=2e-9,atol=2e-11)
    assert outputs[0].work_counts['joint_M_value_rows']<=outputs[1].work_counts['joint_M_value_rows']

@pytest.mark.parametrize('mode',['hypergraph','direct'])
def test_closed_gate_executes_no_wide_values_and_shared_threshold_can_reopen(mode):
    _,model=_model(mode,tile=10);model.eval()
    with torch.no_grad():
        for c in model.router.incidence.routes.values():c.bias.fill_(-100.)
        model.router.direct_parameters.module.bias.fill_(-100.)
        model.router.direct_parameters.environment.bias.fill_(-100.)
        if mode=='hypergraph':model.router.entmax_temperature.fill_(.01)
    scene,query=_double(_scene(1)),_double(_query(1))
    org=model.prepare(scene)
    trace=model.capture_continuous_trace(org,query)
    dense=copy.deepcopy(model);dense.native_backend='dense'
    with torch.no_grad():
        a=model.decode(org,query,torch.ones((1,3),dtype=torch.double))
        b=dense.decode(dense.prepare(scene),query,torch.ones((1,3),dtype=torch.double))
    assert a.work_counts['joint_M_value_rows']+a.work_counts['joint_E_value_rows'] < b.work_counts['joint_M_value_rows']+b.work_counts['joint_E_value_rows']
    torch.testing.assert_close(a.thermal_temperature,b.thermal_temperature,rtol=2e-12,atol=2e-12)
    before=int((trace['gates']>0).sum())
    with torch.no_grad():
        for c in model.router.incidence.routes.values():c.bias.fill_(10.)
        model.router.direct_parameters.module.bias.fill_(10.)
        model.router.direct_parameters.environment.bias.fill_(10.)
    after=model.capture_continuous_trace(model.prepare(scene),query)
    assert int((after['gates']>0).sum())>before


@pytest.mark.parametrize('mode',['direct','dense'])
def test_full_frozen_plan_owner_ids_mutation_restoration_and_E_conditioning(mode):
    _,model=_model(mode,tile=10);model.eval()
    scene,query=_double(_scene(1)),_double(_query(1))
    query=replace(query,query_ids=torch.arange(query.num_queries)[None])
    heat=torch.ones((1,3),dtype=torch.double)
    with torch.no_grad():
        org=model.prepare(scene);plan=model.capture_continuous_plan(org,query)
        before=model.decode(org,query,heat)
        fixed=model.decode_with_plan(org,query,plan,heat)
        torch.testing.assert_close(before.thermal_temperature,fixed.thermal_temperature,rtol=0,atol=0)
        torch.testing.assert_close(before.nonlinear_field,fixed.nonlinear_field,rtol=0,atol=0)
        trace=model.capture_continuous_trace(org,query)
        expected=model._query_states(org,query)+model.environment_conditioning(trace['environment_context'])
        torch.testing.assert_close(trace['conditioned_query'],expected,rtol=0,atol=0)
        with pytest.raises(ValueError,match='IDs'):
            model.decode_with_plan(org,replace(query,query_ids=query.query_ids+100),plan,heat)
        assert model._continuous_frozen_plan is None
        after=model.decode(org,query,heat)
        torch.testing.assert_close(before.thermal_temperature,after.thermal_temperature,rtol=0,atol=0)
        plan.gates.add_(.01)
        with pytest.raises(ValueError,match='changed after capture'):
            model.decode_with_plan(org,query,plan,heat)


def test_frozen_plan_coordinate_gradient_after_checkpointed_forward():
    _,model=_model('direct',checkpoint=True,tile=10);model.eval()
    scene,query=_double(_scene(1)),_double(_query(1))
    query=replace(query,coordinates=query.coordinates.clone().requires_grad_(True),
                  query_ids=torch.arange(query.num_queries)[None])
    heat=torch.ones((1,3),dtype=torch.double)
    with torch.no_grad():
        org=model.prepare(scene)
        plan=model.capture_continuous_plan(org,query)
    output=model.decode_with_plan(org,query,plan,heat)
    gradient=torch.autograd.grad(output.thermal_temperature.sum()+output.nonlinear_field.sum(),
                                 query.coordinates)[0]
    assert torch.isfinite(gradient).all()
    assert model._continuous_frozen_plan is None
    assert model.checkpoint_pipeline


@pytest.mark.parametrize('tile',[1,2,8])
def test_response_tile_kernel_heat_adjoint_and_coordinate_gradients(tile):
    _,model=_model('direct',checkpoint=True,tile=tile);model.eval()
    scene,query=_double(_scene(2)),_double(_query(2))
    query=replace(query,coordinates=query.coordinates.clone().requires_grad_(True))
    heat=torch.tensor([[1.,-.5,2.],[.2,1.,.7]],dtype=torch.double)
    organization=model.prepare(scene)
    output=model.decode(organization,query,heat)
    kernel=model.export_source_kernels(organization,query)
    torch.testing.assert_close(output.thermal_temperature,torch.einsum('bqm,bm->bq',kernel,heat),rtol=2e-12,atol=2e-12)
    g1=torch.autograd.grad(output.thermal_temperature.sum(),query.coordinates,retain_graph=True)[0]
    g2=torch.autograd.grad(torch.einsum('bqm,bm->bq',kernel,heat).sum(),query.coordinates,retain_graph=True)[0]
    torch.testing.assert_close(g1,g2,rtol=2e-10,atol=2e-11)
    cot=torch.arange(query.num_queries,dtype=torch.double)[None].expand(2,-1)
    # A consumed prepared graph is rejected by owner/store validation; rebuild
    # the state for the separately measured heat-adjoint API.
    with torch.no_grad():
        adjoint=model.heat_vjp(model.prepare(scene),query,cot)
        torch.testing.assert_close(adjoint,torch.einsum('bq,bqm->bm',cot,kernel),rtol=2e-12,atol=2e-12)


def test_entmax_extreme_relative_mask_cannot_outrank_valid_donors():
    scores=torch.tensor([[0.,-1e8,1e20]],dtype=torch.double)
    mask=torch.tensor([[True,True,False]])
    actual=entmax15(scores,mask)
    torch.testing.assert_close(actual,torch.tensor([[1.,0.,0.]],dtype=torch.double))


def test_macro_empty_type_and_all_compulsory_floor_preserve_case_denominator():
    from honf_runtime.joint_native_objective import JointMacroObjective
    _,model=_model('direct');model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    objective=JointMacroObjective(model,optimizer,3200,0)
    objective.begin_microbatch()
    # Two actual cases, M fully compulsory, E absent. Every case contributes
    # zero residual to the same equal-case macro denominator.
    work=next(model.router.parameters()).sum()*0+torch.tensor([
        [[3.,3.,3.,3.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.]],
        [[1.,1.,1.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.]]],dtype=torch.double)
    model._joint_entries=[(('case0','case1'),work)]
    objective.observe(expected_case_count=2)
    assert objective.denominators=={'M':2,'E':2}
    assert all(float(sum(x))==0 for x in objective.residuals.values())


def test_zero_local_pair_response_tiles_are_defined_and_native_law_holds():
    _,model=_model('direct',checkpoint=True,tile=1);model.eval()
    scene,query=_double(_scene(2)),_double(_query(2))
    query=replace(query,coordinates=query.coordinates+100.)
    heat=torch.ones((2,3),dtype=torch.double)
    with torch.no_grad():
        org=model.prepare(scene);k=model.export_source_kernels(org,query);output=model.decode(org,query,heat)
        torch.testing.assert_close(output.thermal_temperature,torch.einsum('bqm,bm->bq',k,heat),rtol=2e-12,atol=2e-12)

@pytest.mark.parametrize('mode',['hypergraph','direct','dense'])
def test_actual_affine_native_entry_and_response_counts_share_full_tile_checkpoint(mode):
    _,model=_model(mode,checkpoint=True,tile=2);model.eval()
    scene,query=_double(_scene(2)),_double(_query(2))
    heat=torch.tensor([[1.,-.5,2.],[.2,1.,.7]],dtype=torch.double)
    org=model.prepare(scene)
    direct=model.decode(org,query,heat)
    affine=model.prepare_affine_query(org,query)
    cached=model.apply_prepared_affine(affine,heat)
    response,counts=model.apply_increment(org,query,heat,return_work_counts=True)
    torch.testing.assert_close(cached.thermal_temperature,direct.thermal_temperature,rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(cached.nonlinear_field,direct.nonlinear_field,rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(response,direct.thermal_temperature,rtol=2e-12,atol=2e-12)
    assert counts['source_read_fine_executed_rows']==counts['source_read_joint_M_value_rows']
    assert counts['source_read_joint_E_available_present_rows']>0
    g1=torch.autograd.grad(cached.thermal_temperature.sum()+cached.nonlinear_field.sum(),tuple(model.parameters()),retain_graph=True,allow_unused=True)
    g2=torch.autograd.grad(direct.thermal_temperature.sum()+direct.nonlinear_field.sum(),tuple(model.parameters()),retain_graph=True,allow_unused=True)
    for a,b in zip(g1,g2):
        if a is not None and b is not None:torch.testing.assert_close(a,b,rtol=2e-9,atol=2e-11)


def test_dense_control_counts_rectangular_values_and_physical_typed_eligibility():
    _,model=_model('dense',checkpoint=True,tile=2);model.eval()
    scene,query=_double(_scene(2)),_double(_query(2))
    mm=scene.module_mask.clone();mm[1,-1]=False
    mu=scene.module_measure.clone();mu[0,0]=0
    eu=scene.environment_measure.clone();eu[:,0]=0
    scene=replace(scene,module_mask=mm,module_measure=mu,environment_measure=eu)
    qm=torch.ones_like(query.query_mask) if query.query_mask is not None else torch.ones((2,query.num_queries),dtype=torch.bool)
    qm[1,-1]=False;query=replace(query,query_mask=qm)
    org=model.prepare(scene)
    response,counts=model.apply_increment(org,query,torch.ones_like(mu),return_work_counts=True)
    assert torch.isfinite(response).all()
    assert model.native_backend=='packed'  # The sparse option is inactive in dense mode.
    for kind,valid,n in (('M',mm,scene.num_modules),('E',scene.environment_mask&(eu>0),scene.num_environment_tokens)):
        expected=int((valid[:,None]&qm[...,None]).sum())
        assert counts['source_read_joint_'+kind+'_available_present_rows']==expected
        assert counts['source_read_joint_'+kind+'_selected_rows']==expected
        assert counts['source_read_joint_'+kind+'_value_rows']==scene.batch_size*query.num_queries*n


def test_work_only_checkpoint_backward_never_recomputes_wide_E_values():
    _,model=_model('hypergraph',checkpoint=True,tile=2);model.train();model.joint_begin_update(3400,0)
    scene,query=_double(_scene(1)),_double(_query(1));heat=torch.ones((1,3),dtype=torch.double)
    calls=[]
    hook=model.environment_reader.message_output.register_forward_pre_hook(lambda module,args:calls.append(args[0].shape[0]))
    output=model.decode(model.prepare(scene),query,heat)
    forward_calls=len(calls)
    constraints=model.joint_microbatch_result()['constraints']
    work=sum(row['residual_numerator'] for row in constraints.values())
    gradients=torch.autograd.grad(work,tuple(model.parameters()),retain_graph=True,allow_unused=True)
    assert len(calls)==forward_calls
    torch.autograd.grad(output.nonlinear_field.square().sum()+output.thermal_temperature.square().sum(),tuple(model.parameters()),allow_unused=True)
    assert len(calls)>forward_calls
    hook.remove()
