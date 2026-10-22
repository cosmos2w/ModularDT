"""Same-function contextual execution and adapter capability regressions."""
import pytest
import torch
from honf_forward_core.input_capabilities import LocalInputCapability, register_local_input_capability
from honf_forward_core.continuous_joint.preparation import FrozenContextPlan
from tests.test_continuous_unified_preparation import ConstantPolicy, run_attention, double_scene
from tests.test_content_hypergraph_v2_core import _scene
from honf_forward_core.content_hypergraph.staged_attention_v2 import SpatialRouteAttentionV2

@pytest.mark.parametrize('gate',[0.,.37,1.])
def test_live_context_segment_dense_weighted_forward_and_parameter_gradients(gate):
    torch.manual_seed(911)
    scene=double_scene(_scene(1))
    layer=SpatialRouteAttentionV2(32,4,2).double().eval()
    source=torch.randn(1,3,32,dtype=torch.double)
    outputs=[];gradients=[]
    for backend in ('segment','dense_weighted'):
        policy=ConstantPolicy(gate);policy.execution_backend=backend
        states=source.clone().requires_grad_()
        output=run_attention(layer,scene,states,policy)[0]
        outputs.append(output)
        gradients.append(torch.autograd.grad(output.square().sum(),(states,*layer.parameters()),allow_unused=True))
    torch.testing.assert_close(*outputs,rtol=2e-12,atol=2e-13)
    for a,b in zip(*gradients):
        if a is None or b is None:assert a is b
        else:torch.testing.assert_close(a,b,rtol=3e-11,atol=3e-12)
    if gate==0:assert outputs[0].count_nonzero()==0

def test_frozen_context_dense_weighted_disconnected_values_and_guarded_bias_mass():
    torch.manual_seed(912)
    scene=double_scene(_scene(1))
    layer=SpatialRouteAttentionV2(32,4,2).double().eval()
    states=torch.randn(1,3,32,dtype=torch.double)
    incidence=torch.zeros(1,3,3,dtype=torch.double)
    incidence[0,0,1]=.2;incidence[0,1,0]=.7
    coefficients=incidence[...,None].expand(-1,-1,-1,4)/3
    plan=FrozenContextPlan(incidence,coefficients,coefficients.sum(2))
    class Frozen(torch.nn.Module):
        execution_backend='dense_weighted'
        def forward(self,*args,**kwargs):return plan
    policy=Frozen()
    original=run_attention(layer,scene,states,policy)[0]
    changed=states.clone();changed[:,2]+=100
    perturbed=run_attention(layer,scene,changed,policy)[0]
    torch.testing.assert_close(original,perturbed,rtol=0,atol=0)
    assert original[:,2].count_nonzero()==0
    policy.execution_backend='segment'
    packed=run_attention(layer,scene,states,policy)[0]
    torch.testing.assert_close(original,packed,rtol=2e-13,atol=2e-13)

def test_registered_adapter_needs_no_generic_dataset_branch():
    transform=lambda scene:scene
    cap=LocalInputCapability({'schema':'local_records_context_v1','sha256':'test'},transform)
    register_local_input_capability('unit_adapter_capability',cap)
    from honf_forward_core.continuous_joint.unified import local_input_adapter
    contract,actual=local_input_adapter('unit_adapter_capability')
    assert actual is transform and contract()==dict(cap.contract)
    with pytest.raises(ValueError,match='registered differently'):
        register_local_input_capability('unit_adapter_capability',LocalInputCapability(
            {'schema':'local_records_context_v1','sha256':'different'},transform))


def test_work_schedule_python_clock_survives_state_load_and_execution_options_invalidate_preparation():
    from honf_forward_core.continuous_joint.budget import ContinuousBudgetState
    state=ContinuousBudgetState();state.set_clock(5000,17)
    restored=ContinuousBudgetState();restored.load_state_dict(state.state_dict())
    assert restored.schedule()==state.schedule()
    from tests.test_continuous_joint import _model
    from tests.test_budgeted_native_core import _double
    from tests.test_content_hypergraph_v2_core import _query
    _,model=_model('hypergraph',tile=8);model.eval()
    scene,query=_double(_scene(1)),_double(_query(1))
    prepared=model.prepare(scene)
    model.set_execution_options(backend='segment',checkpoint='values',context_backend='dense_weighted')
    with pytest.raises(ValueError):model.decode(prepared,query)
    assert model.get_config()['native_backend']=='segment'
    assert model.checkpoint_mode=='values'


@pytest.mark.parametrize('backend',['segment','dense_weighted'])
def test_explicit_executor_is_used_by_all_open_control_without_changing_its_law(backend):
    from tests.test_continuous_joint import _model
    from tests.test_budgeted_native_core import _double
    from tests.test_content_hypergraph_v2_core import _query
    _,model=_model('dense',tile=8);model.eval()
    scene,query=_double(_scene(1)),_double(_query(1))
    heat=torch.tensor([[1.,-.5,2.]],dtype=torch.double)
    baseline=model.decode(model.prepare(scene),query,heat)
    model.set_execution_options(backend=backend,checkpoint='off')
    improved=model.decode(model.prepare(scene),query,heat)
    torch.testing.assert_close(improved.nonlinear_field,baseline.nonlinear_field,rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(improved.thermal_temperature,baseline.thermal_temperature,rtol=2e-12,atol=2e-12)
    assert improved.work_counts['joint_M_message_projection_rows']==0
    assert improved.work_counts['joint_M_head_reduction_projection_rows']>0
    assert improved.work_counts['joint_pre_E_M_score_rows']==0


@pytest.mark.parametrize('backend',['segment','dense_weighted'])
def test_improved_executor_keeps_signed_heat_affinity_zero_heat_and_source_order(backend):
    from tests.test_continuous_joint import _model
    from tests.test_budgeted_native_core import _double
    from tests.test_content_hypergraph_v2_core import _query
    from tests.test_continuous_joint_native_contracts import _permuted
    _,model=_model('hypergraph',tile=8);model.eval()
    model.set_execution_options(backend=backend,checkpoint='off')
    scene,query=_double(_scene(1)),_double(_query(1))
    h1=torch.tensor([[1.,-.5,2.]],dtype=torch.double)
    h2=torch.tensor([[-.2,.3,-1.]],dtype=torch.double)
    with torch.no_grad():
        organization=model.prepare(scene)
        a,b,total,zero=[model.decode(organization,query,h) for h in (h1,h2,h1+h2,h1*0)]
        assert a.work_counts['joint_pre_E_M_score_rows']==a.work_counts['joint_M_candidate_rows']
        assert a.work_counts['joint_M_score_rows']==a.work_counts['joint_M_candidate_rows']
        torch.testing.assert_close(total.thermal_temperature,a.thermal_temperature+b.thermal_temperature,rtol=2e-12,atol=2e-12)
        assert zero.thermal_temperature.count_nonzero()==0
        assert torch.equal(a.nonlinear_field,b.nonlinear_field)
        order=torch.tensor([2,0,1])
        permuted=_permuted(scene,'module',order)
        changed=model.decode(model.prepare(permuted),query,h1[:,order])
        torch.testing.assert_close(changed.nonlinear_field,a.nonlinear_field,rtol=2e-12,atol=2e-12)
        torch.testing.assert_close(changed.thermal_temperature,a.thermal_temperature,rtol=2e-12,atol=2e-12)
