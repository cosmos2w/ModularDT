"""Mechanism tests for typed incidence, sparse-control floors and saved clocks."""
from dataclasses import replace
import io

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from honf_forward_core.joint_native.contracts import TypedNativeRecords, CanonicalGeometry
from honf_forward_core.joint_native.graph import (JointGraphParameters, JointNativeRouter,
    build_joint_native_graph, final_support, calibrate_bias, calibrate_graph_birth)
from honf_forward_core.joint_native.routing import (JointDispatchParameters, DirectGateParameters,
    canonical_winner, dispatch_joint_native_seed, direct_typed_gates, nearest_environment_protection)
from honf_forward_core.joint_native.budget import (JointBudgetSchedule, JointBudgetState,
    expected_typed_work, direct_typed_work, calibrated_dual_weight, aggregate_case_constraints)
from honf_forward_core.content_hypergraph.staged_attention_v2 import StagedContextCycleV2
from tests.test_content_hypergraph_v2_core import _scene


def records(m=3, e=4, batch=1):
    def coordinates(n, shift):
        return (torch.arange(n, dtype=torch.float64)*1.31+shift)[None, :, None].expand(batch, n, 2).clone()
    mc, ec = coordinates(m, 0), coordinates(e, .21)
    return TypedNativeRecords(torch.arange(m)[None].expand(batch, m).clone(),
        torch.arange(e)[None].expand(batch, e).clone(), torch.ones((batch,m), dtype=torch.bool),
        torch.ones((batch,e), dtype=torch.bool), mc, ec, torch.ones_like(mc), torch.ones_like(ec),
        torch.ones((batch,m), dtype=torch.float64), torch.ones((batch,e), dtype=torch.float64))


def graph(r=None, **options):
    r = records() if r is None else r
    torch.manual_seed(75)
    b, m, e = r.ids.shape[0], r.module_count, r.environment_ids.shape[1]
    captures = {name: torch.randn(b, 2, rows, cols, dtype=torch.float64)
                for name, rows, cols in (("mm",m,m),("me",m,e),("em",e,m),("ee",e,e))}
    parameters = JointGraphParameters(2, dtype=torch.float64)
    with torch.no_grad():
        for parameter in parameters.routes.values():
            parameter.bias.fill_(-1.)
    return build_joint_native_graph(captures, r, parameters, **options), parameters, captures


def test_typed_ids_remain_disjoint_and_geometry_stamps_reject_mutation():
    r = records()
    assert r.ids[0,0] == r.ids[0,r.module_count]
    assert r.types[0,0] != r.types[0,r.module_count]
    r.module_ids[0,0] = 31
    with pytest.raises(ValueError, match="changed"):
        r.assert_current()


@pytest.mark.parametrize("m,e", [(0,0),(0,3),(3,0),(2,5)])
def test_all_native_seeds_empty_types_and_physical_zero_measure_modules(m,e):
    r = records(m,e)
    if m:
        r.module_measure[0,0] = 0
        r = replace(r)
    g, _, _ = graph(r, deterministic=True)
    assert g.membership_hard.shape == (1,m+e,m+e)
    assert torch.equal(g.self_membership.diagonal(dim1=1,dim2=2), r.eligible)
    if m and e:
        assert g.cross_membership.sum(-1).eq(1).all()
        assert g.membership_hard[0,0,0]  # zero information measure remains a heater
    if m+e:
        selected = torch.arange(m+e)[None]
        support = final_support(selected,g,torch.zeros((1,m+e,m+e),dtype=torch.bool))
        assert torch.equal(support,g.membership_hard)


def test_positive_E_eligibility_nearest_metric_and_padding():
    r = records(2,3)
    r.environment_measure[0,0] = 0
    r.environment_present[0,2] = False
    r = replace(r)
    protection = nearest_environment_protection(r, r.environment_coordinates[:,:1])
    assert protection.tolist() == [[[False,True,False]]]
    g,_,_ = graph(r,deterministic=True)
    assert not g.membership_hard[...,2].any()  # E:0 starts at M=2
    assert not g.membership_hard[:,4].any()  # padded E cannot seed


def test_exact_expected_work_uses_actual_mandatory_union_and_matches_monte_carlo():
    g,_,_ = graph(deterministic=True)
    n,m = g.records.ids.shape[-1],g.records.module_count
    pi = torch.tensor([[[.05,.15,.1,.2,.1,.25,.15],[.4,.1,.1,.1,.1,.1,.1]]],dtype=torch.float64)
    protected = torch.zeros((1,2,n),dtype=torch.bool)
    protected[0,0,1] = True; protected[0,1,m+1] = True
    work = expected_typed_work(g,pi,protected)
    samples = 30000
    generator = torch.Generator().manual_seed(741)
    memberships = torch.rand((samples,n,n),generator=generator,dtype=torch.float64)<g.membership_probabilities[0]
    dispatch = torch.multinomial(pi[0], samples, replacement=True,generator=generator).T
    chosen = memberships[torch.arange(samples)[:,None],dispatch] | protected[0]
    for kind,sl in (("M",slice(None,m)),("E",slice(m,None))):
        torch.testing.assert_close(chosen[...,sl].double().mean(0),work[kind].expected_support[0],atol=.012,rtol=0)
        mandatory = g.mandatory_membership[0,:,sl].double()
        literal_floor = sum(pi[0,:,seed,None] * (mandatory[seed].bool()[None] | protected[0,:,sl]).double()
                            for seed in range(n))
        torch.testing.assert_close(work[kind].floor_rows_per_query[0],literal_floor.sum(-1))


class NoQuerySeedPairs(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        values = result if isinstance(result,(tuple,list)) else (result,)
        assert all(not isinstance(value,torch.Tensor) or value.ndim <= 3 for value in values)
        return result


def test_duplicate_ties_match_literal_reference_without_query_seed_pair_tensor():
    torch.manual_seed(812)
    keys = torch.tensor([[[1.,1.,0.],[0.,1.,0.],[1.,1.,0.],[1.,1.,1.],[4.,1.,1.]]])
    geometry = CanonicalGeometry.build(keys)
    scores = torch.randint(0,3,(1,17,5)).float()
    allowed = torch.rand(1,17,5)>.2
    with NoQuerySeedPairs():
        winner,tied,unresolved = canonical_winner(scores,allowed,geometry)
    for q in range(scores.shape[1]):
        top = [i for i in range(5) if bool(allowed[0,q,i]) and scores[0,q,i]==scores[0,q][allowed[0,q]].max()]
        duplicates = any(torch.equal(keys[0,i],keys[0,j]) for i in top for j in top if i<j)
        assert bool(unresolved[0,q]) == duplicates
        expected = -1 if duplicates or not top else min(top,key=lambda i:tuple(keys[0,i].tolist()))
        assert int(winner[0,q]) == expected


def test_dispatch_permutation_geometry_not_ID_and_type_collision():
    r = records(3,2)
    raw_m,raw_e = torch.zeros(1,2,4,3,dtype=torch.float64),torch.zeros(1,2,4,2,dtype=torch.float64)
    parameters = JointDispatchParameters(2,dtype=torch.float64)
    plan = dispatch_joint_native_seed(raw_m,raw_e,r,parameters,deterministic=True)
    assert plan.dispatch_seed.eq(0).all()
    perm = torch.tensor([2,0,1])
    other = replace(r,module_ids=r.module_ids[:,perm],module_coordinates=r.module_coordinates[:,perm],
        module_lengths=r.module_lengths[:,perm],module_present=r.module_present[:,perm],module_measure=r.module_measure[:,perm])
    new = dispatch_joint_native_seed(raw_m[...,perm],raw_e,other,parameters,deterministic=True)
    assert torch.equal(plan.dispatch_seed_ids,new.dispatch_seed_ids)
    assert new.dispatch_seed.eq(1).all()


def test_direct_control_floors_and_dense_omitted_gate_restoration_gradients():
    r = records(3,4)
    raw_m = torch.tensor([[[[0.,-2.,-4.]],[[0.,-1.,-3.]]]],dtype=torch.float64)
    raw_e = torch.tensor([[[[0.,-1.,-2.,-3.]],[[0.,-2.,-3.,-4.]]]],dtype=torch.float64)
    parameters = DirectGateParameters(2,dtype=torch.float64)
    with torch.no_grad():
        parameters.module.bias.fill_(-2); parameters.environment.bias.fill_(-2)
    protected = torch.zeros(1,1,7,dtype=torch.bool); protected[...,2]=True
    plan = direct_typed_gates(raw_m,raw_e,r,parameters,protected,deterministic=True,straight_through=True)
    assert plan.hard_support.tolist() == [[[True,False,True,True,False,False,False]]]
    assert torch.equal(plan.value_gates,plan.hard_support.double())
    # The omitted original values are evaluated by the dense estimator; a hard
    # zero gate still receives its task derivative through this shared bias.
    loss = (plan.value_gates * torch.arange(7,dtype=torch.float64)).sum()
    loss.backward()
    assert parameters.module.bias.grad.abs()>0 and parameters.environment.bias.grad.abs()>0
    work = direct_typed_work(plan,r)
    assert work["M"].floor_rows_per_query.item()==2
    assert work["E"].floor_rows_per_query.item()==1


def test_balanced_four_epoch_case_coverage_and_exact_mid_window_resume():
    for sizes in ([48,48,48,6],[16,16,16,16,8]):
        visits = sum(sizes)*4
        dense_visits = sum(size for epoch in range(3201,3205) for macro,size in enumerate(sizes)
            if JointBudgetSchedule.state_at_epoch(epoch,macro)["dense_router_step"])
        assert dense_visits*4==visits
    assert JointBudgetSchedule.state_at_epoch(3001,3)["stochastic_expectation_exact"]
    assert JointBudgetSchedule.state_at_epoch(4001,3)["deterministic"]
    state = JointBudgetState(.1,.2)
    state.set_clock(3371,4)
    for step in range(1,8):
        state.observe_residuals(torch.tensor(.2),torch.tensor(-.1),step)
    buffer=io.BytesIO();torch.save(state.state_dict(),buffer);buffer.seek(0)
    restored=JointBudgetState();restored.load_state_dict(torch.load(buffer,weights_only=False))
    assert restored.schedule()==state.schedule()
    for step in range(8,11):
        state.observe_residuals(torch.tensor(.2),torch.tensor(-.1),step)
        restored.observe_residuals(torch.tensor(.2),torch.tensor(-.1),step)
    for name in ("M","E"):
        torch.testing.assert_close(restored.duals[name].dual_lambda,state.duals[name].dual_lambda,rtol=0,atol=0)
    with pytest.raises(ValueError,match="sequence"):
        restored.observe_residuals(torch.tensor(0.),torch.tensor(0.),10)


def test_birth_bias_and_unsupported_dual_calibration():
    scores=torch.tensor([-8.,-4.,-1.,0.],dtype=torch.float64)
    bias=calibrate_bias(scores,torch.ones_like(scores,dtype=torch.bool))
    assert float((scores+bias).sigmoid().mean())==pytest.approx(.98,abs=1e-12)
    assert calibrated_dual_weight(4.,2.)==pytest.approx(.1)
    with pytest.raises(ValueError,match="supported"):
        calibrated_dual_weight(0.,1.)


def test_birth_four_route_calibration_excludes_exact_floors_and_bounds_batches():
    directory,parameters,captures=graph(deterministic=True)
    result=calibrate_graph_birth([directory],parameters)
    assert all(row["status"]=="calibrated" for row in result.values())
    rebuilt=build_joint_native_graph(captures,directory.records,parameters,deterministic=True)
    m=rebuilt.records.module_count
    for name,row in result.items():
        ds=slice(None,m) if name[0]=="m" else slice(m,None)
        ss=slice(None,m) if name[1]=="m" else slice(m,None)
        eligible=rebuilt.records.present[:,ds,None] & rebuilt.records.eligible[:,None,ss]
        eligible &= ~rebuilt.mandatory_membership[:,ds,ss]
        assert float(rebuilt.membership_probabilities[:,ds,ss][eligible].mean())==pytest.approx(.98,abs=1e-12)
    with pytest.raises(ValueError,match="eight"):
        calibrate_graph_birth([directory]*9,parameters)


def test_four_native_score_capture_changes_no_attention_outputs_and_preserves_legacy():
    scene=_scene(1)
    cycle=StagedContextCycleV2(16,2,2).eval()
    torch.manual_seed(8)
    m,e=torch.randn(1,3,16),torch.randn(1,4,16)
    frame=scene.coordinate_scale
    base=cycle(m,e,scene,frame)
    captured=cycle(m,e,scene,frame,capture_native_routes=True,native_capture_route_names=("mm","me","em","ee"),
                   cycle_index=1,owner_token=775,scene_identity=("scene", "normalizer", "frame"))
    torch.testing.assert_close(base[0],captured[0],atol=0,rtol=0)
    torch.testing.assert_close(base[1],captured[1],atol=0,rtol=0)
    captures={name:captured[2][name+"_native_route_snapshot"] for name in ("mm","me","em","ee")}
    assert {snap.stage_identity for snap in captures.values()}=={"cycle1/"+name for name in captures}
    assert all(snap.owner_token==775 for snap in captures.values())
    legacy=cycle(m,e,scene,frame,export_route_scores=True)
    assert legacy[2]["em_score_snapshot"].route=="em"
    assert legacy[2]["ee_score_snapshot"].route=="ee"
    directory=build_joint_native_graph(captures,TypedNativeRecords.from_scene(scene),JointGraphParameters(2),deterministic=True)
    assert directory.membership_hard.shape==(1,7,7)
