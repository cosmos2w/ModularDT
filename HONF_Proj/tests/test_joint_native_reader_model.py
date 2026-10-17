"""Round9 physical value, true gather, conditioning and retained-state checks."""
from dataclasses import replace

import pytest
import torch

from honf_forward_core.joint_native.model import JointNativeCore
from honf_forward_core.joint_native.readers import read_typed
from tests.test_content_hypergraph_v2_core import _core, _query, _scene
from tests.test_budgeted_native_core import _double
from tests.test_native_budgeted_reader import _fixture, _literal_a2d


def _read(reader, organization, query, states, support, backend, **options):
    s = organization.scene
    return read_typed(reader, query, states, coordinates=s.module_coordinates,
        lengths=s.module_characteristic_lengths, valid=s.module_mask, measure=s.module_measure,
        reference=s.module_measure_reference, frame_scale=organization.frame_scale,
        keys=organization.direct_source_keys, values=organization.direct_source_values,
        gate=support, backend=backend, **options)


@pytest.mark.parametrize("backend", ["flat_typed_selected", "dense_typed_reference"])
def test_independent_literal_fp64_and_zero_measure_source(backend):
    reader, org, query, states, _ = _fixture(batch=1)
    org.scene = replace(org.scene, module_measure=org.scene.module_measure.detach().clone())
    org.scene.module_measure[0,0] = 0
    support = torch.tensor([[[1,1,0,1]]*5], dtype=torch.bool)
    expected_context, expected_alpha, expected_coeff = _literal_a2d(reader, org, query, states, support)
    actual = _read(reader,org,query,states,support,backend)
    coeff = actual.response if isinstance(actual.response,torch.Tensor) else actual.response.explicit_dense_kernel()
    torch.testing.assert_close(actual.context,expected_context,rtol=2e-13,atol=2e-13)
    torch.testing.assert_close(actual.alpha,expected_alpha,rtol=2e-13,atol=2e-13)
    torch.testing.assert_close(coeff,expected_coeff,rtol=2e-13,atol=2e-13)
    assert actual.alpha[:,:,0].count_nonzero()==0
    assert coeff[:,:,0].abs().sum()>0


def test_flat_value_hook_is_actual_selected_rows_and_dense_gate_can_restore_omission():
    reader, org, query, states, _ = _fixture(batch=1)
    gate = torch.tensor([[[1.,0.,1.,0.]]*5],dtype=states.dtype,requires_grad=True)
    observed=[]
    hook=reader.relative_value.register_forward_pre_hook(lambda _module,args: observed.append(args[0].shape))
    result=_read(reader,org,query,states,gate,"flat_typed_selected")
    hook.remove()
    assert observed==[torch.Size([10,14])]
    assert result.counts['value_rows']==10
    dense=_read(reader,org,query,states,gate,"dense_typed_reference")
    gradient=torch.autograd.grad(dense.context.square().sum()+dense.response.square().sum(),gate)[0]
    assert gradient[:,:,1::2].abs().sum()>0


@pytest.mark.parametrize("backend", ["flat_typed_selected", "dense_typed_reference"])
def test_checkpoint_includes_reduction_and_preserves_fixed_support_vjp(backend):
    reader,org,query,states,_=_fixture(batch=1)
    support=torch.tensor([[[1,0,1,1]]*5],dtype=torch.bool)
    plain=_read(reader,org,query,states,support,backend,query_tile_size=2)
    recomputed=_read(reader,org,query,states,support,backend,query_tile_size=2,checkpoint_values=True)
    torch.testing.assert_close(plain.context,recomputed.context,rtol=0,atol=0)
    parameters=(states,org.direct_source_values,org.scene.module_coordinates)
    def loss(result):
        response=result.response if isinstance(result.response,torch.Tensor) else result.response.explicit_dense_kernel()
        return result.context.square().sum()+response.square().sum()
    expected=torch.autograd.grad(loss(plain),parameters,retain_graph=True)
    actual=torch.autograd.grad(loss(recomputed),parameters,retain_graph=True)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=2e-13,atol=2e-13)


def test_parent_exact_boundary_and_non_dead_E_to_native_K_optimizer_path():
    parent=_core('A2-D').double().eval()
    model=JointNativeCore.from_a2d(parent).eval()
    scene,query=_double(_scene(1)),_double(_query(1))
    heat=torch.tensor([[1.,-.5,2.]],dtype=torch.double)
    old=parent.decode(parent.prepare(scene),query,heat)
    new=model.decode(model.prepare(scene),query,heat)
    assert torch.equal(old.nonlinear_field,new.nonlinear_field)
    assert torch.equal(old.thermal_temperature,new.thermal_temperature)
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    for index in range(2):
        optimizer.zero_grad(set_to_none=True)
        output=model.decode(model.prepare(scene),query,heat)
        output.thermal_temperature.square().sum().backward()
        assert model.environment_conditioning.weight.grad.abs().sum()>0
        if index:
            assert model.environment_reader.message_output.weight.grad.abs().sum()>0
            assert model.environment_reader.source_key.weight.grad.abs().sum()>0
        optimizer.step()


@pytest.mark.parametrize('routing_mode',['dense','joint','direct'])
def test_all_physical_apis_share_conditioning_and_stale_owner_rules(routing_mode):
    model=JointNativeCore.from_a2d(_core('A2-D').double(),routing_mode=routing_mode).eval()
    with torch.no_grad():
        model.environment_conditioning.weight.copy_(torch.eye(model.context_dim,dtype=torch.double)*.2)
    scene,query=_double(_scene(1)),_double(_query(1))
    heat=torch.tensor([[1.,-.5,2.]],dtype=torch.double)
    with torch.no_grad():
        organization=model.prepare(scene)
        output=model.decode(organization,query,heat)
        zero=model.decode(organization,query,heat*0)
        prepared=model.prepare_affine_query(organization,query)
        cached=model.apply_prepared_affine(prepared,heat)
        torch.testing.assert_close(output.thermal_temperature,cached.thermal_temperature,rtol=2e-13,atol=2e-13)
        kernel=model.export_source_kernels(organization,query)
        torch.testing.assert_close(torch.einsum('bqm,bm->bq',kernel,heat),output.thermal_temperature,rtol=2e-13,atol=2e-13)
        torch.testing.assert_close(model.heat_jvp(organization,query,heat),output.thermal_temperature,rtol=2e-13,atol=2e-13)
        cotangent=torch.arange(query.num_queries,dtype=torch.double)[None]
        torch.testing.assert_close(model.heat_vjp(organization,query,cotangent),
                                   torch.einsum('bq,bqm->bm',cotangent,kernel),rtol=2e-13,atol=2e-13)
        assert zero.thermal_temperature.count_nonzero()==0
        assert torch.equal(output.nonlinear_field,zero.nonlinear_field)
        restored=JointNativeCore.from_config(model.get_config()).double()
        restored.load_state_dict(model.state_dict())
        with pytest.raises(ValueError,match='different model'):
            restored.decode(organization,query,heat)
        organization.joint_environment_values.add_(1)
        with pytest.raises(ValueError,match='projections changed|mutated'):
            model.decode(organization,query,heat)


def test_backward_consumed_and_recursive_clock_roundtrip():
    model=JointNativeCore.from_a2d(_core('A2-D').double()).train()
    model.joint_begin_update(2760,2)
    scene,query=_double(_scene(1)),_double(_query(1))
    prepared=model.prepare(scene)
    model.decode(prepared,query).nonlinear_field.sum().backward()
    with pytest.raises(ValueError,match='consumed'):
        model.decode(prepared,query)
    wrapper=torch.nn.ModuleList([model])
    restored=JointNativeCore.from_config(model.get_config()).double()
    torch.nn.ModuleList([restored]).load_state_dict(wrapper.state_dict())
    assert restored._joint_epoch==2760 and restored._joint_update_index==2


def test_checkpoint_does_not_save_wide_dense_messages_between_tiles():
    reader,org,query,states,_=_fixture(batch=1)
    support=torch.ones((1,5,4),dtype=torch.bool)
    shapes=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: (shapes.append(tuple(tensor.shape)) or tensor),lambda tensor:tensor):
        _read(reader,org,query,states,support,'dense_typed_reference',query_tile_size=2,checkpoint_values=True)
    assert not any(len(shape)==4 and shape[-1]==reader.width for shape in shapes)


def test_continuous_fixed_support_primitive_gradcheck():
    reader,org,query,states,_=_fixture(batch=1,queries=2,modules=2,width=8,heads=2)
    support=torch.ones((1,2,2),dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda qs: _read(reader,org,query,qs,support,"flat_typed_selected").context,
                                   (states,),fast_mode=True,atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('routing_mode',['joint','direct'])
def test_receiver_chunk_permutation_and_fresh_state_counterfactual(routing_mode):
    model=JointNativeCore.from_a2d(_core('A2-D').double(),routing_mode=routing_mode).eval()
    model.set_training_clock(4001,0)
    scene,query=_double(_scene(1)),_double(_query(1))
    with torch.no_grad():
        model.environment_conditioning.weight.copy_(torch.eye(model.context_dim,dtype=torch.double)*.2)
        organization=model.prepare(scene)
        full=model.decode(organization,query)
        chunks=[]
        for start,stop in ((0,2),(2,5)):
            part=replace(query,coordinates=query.coordinates[:,start:stop],features=query.features[:,start:stop],
                         query_ids=query.query_ids[:,start:stop])
            chunks.append(model.decode(organization,part).nonlinear_field)
        torch.testing.assert_close(torch.cat(chunks,1),full.nonlinear_field,rtol=2e-12,atol=2e-12)
        routes=model.capture_read_routes(organization,query)
        changed=replace(scene,environment_features=scene.environment_features+.2)
        counterfactual=model.decode_counterfactual_routes(changed,query,routes)
        assert torch.isfinite(counterfactual.nonlinear_field).all()
        assert not torch.equal(counterfactual.nonlinear_field,full.nonlinear_field)
        order=torch.tensor([4,1,3,0,2])
        permuted=replace(query,coordinates=query.coordinates[:,order],features=query.features[:,order],query_ids=query.query_ids[:,order])
        actual=model.decode_counterfactual_routes(changed,permuted,routes)
        torch.testing.assert_close(actual.nonlinear_field,counterfactual.nonlinear_field[:,order],rtol=2e-12,atol=2e-12)
        routes.support.logical_not_()
        with pytest.raises(ValueError,match='changed after capture'):
            model.decode_counterfactual_routes(changed,query,routes)


@pytest.mark.parametrize('routing_mode',['joint','direct'])
def test_dense_router_native_task_gradient_and_typed_budget_restore(routing_mode):
    model=JointNativeCore.from_a2d(_core('A2-D').double(),routing_mode=routing_mode).train()
    model.joint_begin_update(3001,0)
    scene,query=_double(_scene(1)),_double(_query(1))
    output=model.decode(model.prepare(scene),query,torch.ones(1,3,dtype=torch.double))
    result=model.joint_microbatch_result()
    assert set(result['constraints'])=={'M','E'}
    (output.nonlinear_field.square().sum()+output.thermal_temperature.square().sum()).backward()
    active=[parameter.grad for parameter in model.router.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert sum(gradient.abs().sum() for gradient in active)>0
    model.joint_after_update(3001,0,{'M':torch.tensor(.1),'E':torch.tensor(.2)})
    restored=JointNativeCore.from_config(model.get_config()).double()
    restored.load_state_dict(model.state_dict())
    assert restored._joint_macro_clock==1
    assert int(restored.joint_budget.duals['M'].macro_step)==1
