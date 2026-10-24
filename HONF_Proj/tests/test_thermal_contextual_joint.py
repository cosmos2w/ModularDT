"""Joint Thermal control preparation and native physical-observation regressions."""
import pytest
import torch
from channelthermal.contextual_joint import ThermalContextualJointAdapter
from tests.test_thermal_content_hypergraph_v2 import _stats, _structure, _receivers


def _model():
    return ThermalContextualJointAdapter(_stats(), heat_scale=2., nx=8, ny=6,
        core_config={"schema":"contextual_organization_joint_v1","architecture":"J-D",
            "module_input_dim":9,"environment_input_dim":6,"operating_input_dim":12,
            "coordinate_dim":2,"output_dim":4,"context_dim":16,"local_hidden_dim":16,
            "num_heads":4,"context_depth":1})


def test_joint_native_control_and_reload():
    torch.set_num_threads(1)
    model = _model().eval()
    structure = _structure()
    fluid, local = _receivers()
    heat = torch.tensor([[1., 2.]], requires_grad=True)
    prepared = model.prepare_native(structure, fluid, physical_heat=heat,
        local_query_points=local, ntheta=16, scene_key=("fixture",))
    assert prepared.scene_batch.module_features.shape[-1] == 9
    assert prepared.scene_batch.environment_features.shape[-1] == 6
    assert prepared.scene_batch.operating_features.shape[-1] == 12
    assert prepared.scene_batch.module_ids.tolist() == [[31,47]]
    output = model.apply_native(prepared, heat)
    assert output["pred_field"].shape == (1,48,5)
    loss = output["pred_field"].square().mean()
    loss.backward()
    assert heat.grad is not None and torch.isfinite(heat.grad).all() and heat.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="Heat changed"):
        model.apply_native(prepared, heat.detach()+1)
    with pytest.raises(ValueError, match="separate complete"):
        model.apply_native(prepared, heat, increment=True)
    restored = ThermalContextualJointAdapter.from_state_payload(model.state_payload(), state_dict=model.state_dict()).eval()
    other = restored.prepare_native(structure,fluid,physical_heat=heat.detach(),local_query_points=local,ntheta=16,scene_key=("fixture",))
    torch.testing.assert_close(output["pred_field"].detach(), restored.apply_native(other,heat.detach())["pred_field"])


def test_temperature_inverse_transform_has_mean_and_boundary_is_prescribed():
    model=_model()
    assert model._physical_temperature(torch.zeros(1)).item()==10.
    structure={**_structure(),"t_in":torch.ones(1,1)*2,"t_wall":torch.ones(1,1)*3,
               "heat_powers":torch.zeros(1,2)}
    scene=model.scene_batch(structure)
    torch.testing.assert_close(scene.operating_features[...,-2:],torch.tensor([[2.,3.]]))


def test_matched_leaf_and_frontier_models_share_initial_named_parameters():
    leaf=_model()
    config={**leaf.core.get_config(),"architecture":"J-H"}
    adaptive=ThermalContextualJointAdapter(_stats(),core_config=config,heat_scale=2.,nx=8,ny=6)
    left,right=dict(leaf.named_parameters()),dict(adaptive.named_parameters())
    assert left.keys()==right.keys()
    for name in left:
        torch.testing.assert_close(left[name],right[name],rtol=0,atol=0)
