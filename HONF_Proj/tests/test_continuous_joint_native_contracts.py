"""Named physical, source inventory, and literal-loop continuous contracts."""
from dataclasses import replace
import torch
import pytest
from torch.nn import functional as F
from honf_forward_core.continuous_joint.reader import read_continuous
from honf_forward_core.joint_native.readers import raw_scores
from tests.test_continuous_joint import _model
from tests.test_content_hypergraph_v2_core import _scene,_query
from tests.test_budgeted_native_core import _double


def _permuted(scene,kind,index):
    names=(kind+'_features',kind+'_coordinates',kind+'_mask',kind+'_measure',kind+'_ids',kind+'_characteristic_lengths')
    return replace(scene,**{name:getattr(scene,name).index_select(1,index) for name in names})


@pytest.mark.parametrize('mode',['hypergraph','direct'])
def test_signed_heat_affinity_source_permutation_zero_measure_and_empty_E(mode):
    _,model=_model(mode,tile=8);model.eval()
    scene,query=_double(_scene(1)),_double(_query(1))
    measure=scene.module_measure.clone();measure[:,0]=0
    emeasure=scene.environment_measure.clone();emeasure[:,0]=0
    scene=replace(scene,module_measure=measure,environment_measure=emeasure)
    h1=torch.tensor([[1.,-.5,2.]],dtype=torch.double);h2=torch.tensor([[-.2,.3,-1.]],dtype=torch.double)
    with torch.no_grad():
        org=model.prepare(scene)
        a=model.decode(org,query,h1);b=model.decode(org,query,h2);summed=model.decode(org,query,h1+h2)
        zero=model.decode(org,query,h1*0);scaled=model.decode(org,query,2*h1)
        torch.testing.assert_close(summed.thermal_temperature,a.thermal_temperature+b.thermal_temperature,rtol=2e-12,atol=2e-12)
        torch.testing.assert_close(scaled.thermal_temperature,2*a.thermal_temperature,rtol=2e-12,atol=2e-12)
        assert zero.thermal_temperature.count_nonzero()==0
        assert torch.equal(a.nonlinear_field,b.nonlinear_field)
        trace=model.capture_continuous_trace(org,query)
        assert trace['module_access'][:,:,0].count_nonzero()==0
        assert trace['source_coefficients'][:,:,0].abs().sum()>0
        assert trace['environment_access'][:,:,0].count_nonzero()==0
        mi=torch.tensor([2,0,1]);ei=torch.tensor([2,0,3,1])
        permuted=_permuted(_permuted(scene,'module',mi),'environment',ei)
        p=model.decode(model.prepare(permuted),query,h1[:,mi])
        torch.testing.assert_close(p.nonlinear_field,a.nonlinear_field,rtol=2e-12,atol=2e-12)
        torch.testing.assert_close(p.thermal_temperature,a.thermal_temperature,rtol=2e-12,atol=2e-12)
        empty=_permuted(scene,'environment',torch.empty(0,dtype=torch.long))
        empty_output=model.decode(model.prepare(empty),query,h1)
        assert torch.isfinite(empty_output.nonlinear_field).all() and torch.isfinite(empty_output.thermal_temperature).all()
        assert empty_output.work_counts['joint_E_value_rows']==0


def _literal(reader,organization,query,qstates,gate,kind):
    scene=organization.scene;is_m=kind=='module'
    coordinates=getattr(scene,kind+'_coordinates');lengths=getattr(scene,kind+'_characteristic_lengths')
    measure=getattr(scene,kind+'_measure');valid=getattr(scene,kind+'_mask')
    keys=organization.direct_source_keys if is_m else organization.joint_environment_keys
    values=organization.direct_source_values if is_m else organization.joint_environment_values
    scores=raw_scores(reader,query,qstates,coordinates,lengths,organization.frame_scale,keys)
    contexts=[];kernels=[]
    for b in range(query.batch_size):
        cq=[];kq=[]
        for q in range(query.num_queries):
            eligible=valid[b]&(measure[b]>0)
            log_weight=scores[b,:,q]+torch.where(eligible,measure[b],1).log()[None]
            if eligible.any():
                w=torch.exp(log_weight-log_weight[:,eligible].max(-1,keepdim=True).values)*eligible
                w=w/w.sum(-1,keepdim=True)
            else:w=torch.zeros_like(log_weight)
            u=w*gate[b,q][None];z=u.sum(-1,keepdim=True)
            alpha=u/(z+1e-3*(1-z).square())
            messages=[];coeff=[]
            for n in range(coordinates.shape[1]):
                delta=query.coordinates[b,q]-coordinates[b,n]
                relative=delta/organization.frame_scale[b]
                features=[relative]
                for frequency in (1.,2.,4.):
                    phase=2*torch.pi*frequency*relative;features.extend((phase.sin(),phase.cos()))
                relative_value=reader.relative_value(torch.cat(features))
                if reader.source_scale_projection is not None:
                    relative_value=relative_value+reader.source_scale_projection(torch.asinh(delta/lengths[b,n]))
                message=reader.message_output(F.silu(reader.query_value(qstates[b,q])+values[b,n]+relative_value))
                messages.append(message.reshape(reader.heads,reader.head_width))
                if reader.thermal_output is not None:coeff.append(reader.thermal_output(message).squeeze()*gate[b,q,n])
            cq.append((torch.stack(messages)*alpha.T[...,None]).sum(0).flatten())
            if coeff:kq.append(torch.stack(coeff))
        contexts.append(torch.stack(cq))
        if kq:kernels.append(torch.stack(kq))
    return torch.stack(contexts),torch.stack(kernels) if kernels else None


@pytest.mark.parametrize('mode',['hypergraph','direct'])
def test_literal_fp64_E_then_M_guarded_values_signed_K_and_coordinate_vjp(mode):
    _,model=_model(mode,tile=8);model.eval()
    scene,query=_double(_scene(1)),_double(_query(1))
    query=replace(query,coordinates=query.coordinates.clone().requires_grad_(True))
    org=model.prepare(scene);q0=model._query_states(org,query)
    gates,seeds,envelope,qmask,_=model._continuous_route_tile(org,query,q0)
    # Include exact zero, tiny positive, intermediate and all-open native paths.
    gm=torch.tensor([[[0.,1e-7,.4]]],dtype=torch.double).expand(1,query.num_queries,3)
    ge=torch.tensor([[[0.,.01,.7,1.]]],dtype=torch.double).expand(1,query.num_queries,4)
    ec,_=_literal(model.environment_reader,org,query,q0,ge,'environment')
    e=model._read('environment',org,query,q0,ge)
    torch.testing.assert_close(e.context,ec,rtol=2e-13,atol=2e-13)
    qe=q0+model.environment_conditioning(ec)
    mc,mk=_literal(model.direct_reader,org,query,qe,gm,'module')
    m=model._read('module',org,query,qe,gm)
    torch.testing.assert_close(m.context,mc,rtol=2e-13,atol=2e-13)
    torch.testing.assert_close(m.response,mk,rtol=2e-13,atol=2e-13)
    expected=torch.autograd.grad(mc.square().sum()+mk.square().sum(),query.coordinates,retain_graph=True)[0]
    actual=torch.autograd.grad(m.context.square().sum()+m.response.square().sum(),query.coordinates)[0]
    torch.testing.assert_close(actual,expected,rtol=2e-11,atol=2e-12)


def test_shared_bias_learning_reopens_closed_route_without_straight_through_gradient():
    from honf_forward_core.continuous_joint.primitives import positive_quadratic_gate
    bias=torch.nn.Parameter(torch.tensor(0.,dtype=torch.double))
    preactivation=torch.tensor([-.1,1.],dtype=torch.double)
    before=positive_quadratic_gate(preactivation+bias)
    closed_grad=torch.autograd.grad(before[0],bias,retain_graph=True)[0]
    assert before[0]==0 and closed_grad==0
    # Only the already-open query supplies the exact gradient on the shared
    # scalar; the closed route receives no invented restoration derivative.
    optimizer=torch.optim.SGD([bias],lr=.5)
    (-before[1]).backward();optimizer.step()
    after=positive_quadratic_gate(preactivation+bias)
    assert after[0]>0 and bias.detach()>0
