"""Numerical restrictions and actual flow learning, without reference solves."""
import copy

import pytest
import torch
from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.dependency_flow import DependencySeparatedThermalModel, ThermalFlowReader
from channelthermal.model import ChannelThermalHONFModel


def inputs(modules=4):
    torch.manual_seed(11)
    return {"module_centers": torch.rand(2, modules, 2) * torch.tensor([12., 4.]),
                "module_present": torch.ones(2, modules), "heat_powers": torch.rand(2, modules),
                "re": torch.tensor([[80.], [140.]]), "u_in": torch.tensor([[.8], [1.4]]),
                "material_params": torch.tensor([[.01,.1,.2,1.,2.,.2]]).expand(2,-1),
                "domain_length_x": torch.full((2,1),12.), "domain_length_y": torch.full((2,1),4.)}


@pytest.mark.parametrize("modules", [1, 12])
def test_matched_init_and_actual_optimizer_update(modules):
    torch.manual_seed(0); separated = ThermalFlowReader("D-sep")
    torch.manual_seed(0); opened = ThermalFlowReader("D-open")
    assert all(torch.equal(value, opened.state_dict()[name]) for name,value in separated.state_dict().items())
    structure = inputs(modules); query = torch.rand(2,17,2)
    assert torch.equal(separated(structure,query),opened(structure,query))
    for model in (separated,opened):
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
        model(structure,query).square().mean().backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step()
        assert any(not torch.equal(value,model.state_dict()[name]) for name,value in before.items())


def test_heat_absent_with_geometry_context_query_live():
    torch.manual_seed(0); model = ThermalFlowReader("D-sep")
    structure = inputs()
    for name in ("heat_powers", "module_centers", "u_in"):
        structure[name] = structure[name].clone().requires_grad_()
    query = torch.rand(2,17,2,requires_grad=True)
    value = model(structure,query)
    gradient = torch.autograd.grad(value.square().mean(),
        [structure["heat_powers"],structure["module_centers"],structure["u_in"],query],allow_unused=True)
    assert gradient[0] is None
    assert all(torch.isfinite(item).all() and item.abs().sum() > 0 for item in gradient[1:])
    changed = dict(structure,heat_powers=structure["heat_powers"]+7)
    left,right = model.prepare_flow(structure),model.prepare_flow(changed)
    assert torch.equal(left.source_inputs,right.source_inputs)
    assert torch.equal(left.source_states,right.source_states)
    assert torch.equal(value,model(changed,query))
    forbidden = dict(changed, predicted_ports=torch.randn(2,4,8,5),
                     thermal_hidden_state=torch.randn(2,4,128),
                     global_heat_statistics=torch.randn(2,4))
    forbidden["material_params"] = structure["material_params"].clone()
    forbidden["material_params"][:,1:5] += 100
    assert torch.equal(value,model(forbidden,query))


def test_permutation_padding_chunking_prepared():
    torch.manual_seed(0); model = ThermalFlowReader("D-sep")
    structure=inputs(); query=torch.rand(2,31,2)
    expected=model(structure,query)
    permutation=torch.tensor([3,1,0,2])
    changed=dict(structure)
    for name in ("module_centers","module_present","heat_powers"):
        changed[name]=structure[name][:,permutation]
    torch.testing.assert_close(model(changed,query),expected,atol=1e-7,rtol=1e-6)
    padded=dict(structure)
    for name in ("module_centers","module_present","heat_powers"):
        value=structure[name]
        padded[name]=torch.cat((value,value.new_zeros((2,3,*value.shape[2:]))),1)
    torch.testing.assert_close(model(padded,query),expected,atol=1e-7,rtol=1e-6)
    prepared=model.prepare_flow(structure)
    torch.testing.assert_close(model.read_flow(prepared,query,7),expected,atol=1e-7,rtol=1e-6)


def test_heat_batch_rejects_geometry_changes_and_stale_local_inputs():
    parent = ChannelThermalHONFModel(ChannelThermalHONFConfig(), attach_local_from_checkpoint=False)
    first = inputs()
    changed = dict(first, module_centers=first["module_centers"] + .1)
    query = torch.rand(2, 17, 2)
    # Exercise the native CPU wrapper to initialize its lazy encoders before
    # attaching the frozen composition; no mocked thermal outputs are used.
    parent(first, query, local_query_points=torch.rand(2, 4, 3, 2))
    model = DependencySeparatedThermalModel(parent, ThermalFlowReader("D-sep"))
    with pytest.raises(ValueError, match="identical physical layout"):
        model.evaluate_heat_batch([first, changed], query)
    with pytest.raises(ValueError, match="stale own heat"):
        model.evaluate_heat_batch([first], query, local_module_params=torch.zeros(2, 4, 7))
    model.train()
    assert model.flow.training and not model.thermal.training
    assert not any(parameter.requires_grad for parameter in model.thermal.parameters())
