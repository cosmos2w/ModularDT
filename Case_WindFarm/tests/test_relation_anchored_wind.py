import io
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.relation_anchored import WindRelationAnchoredAdapter, attach_wind_e_cover


def _adapter(shape=(4,4,4)):
    normalizer = VelocityNormalizer(mean=np.zeros(3), std=np.ones(3), safe_std=np.ones(3))
    profile = VerticalProfileBaseline(bin_centers_D=np.array([0., 2.]), values_mps=np.array([[5.,0.,0.],[7.,0.,0.]]), counts=np.array([2,2]))
    base = ContentHypergraphCoreV2(module_input_dim=2, environment_input_dim=7, coordinate_dim=3, query_input_dim=7, operating_input_dim=11, output_dim=3, architecture='A2-D', context_dim=8, num_heads=2, context_depth=2, local_hidden_dim=8, dropout=0., init_seed=17).get_config()
    return WindRelationAnchoredAdapter(normalizer, profile, base_model_config=base, environment_shape=shape).double().eval()


def _case():
    return SimpleNamespace(index=1, case='example', layout_index=1, wind_direction_deg=270.,
        module_centers=np.array([[.2,.3,.8],[1.2,1.,.8]]), module_features=np.array([[.5,.8],[.5,.8]]), module_present=np.ones(2), global_context=np.zeros(11),
        support=SimpleNamespace(lower_D=np.zeros(3), extent_D=np.array([2.,2.,2.])))


def test_cover_permuted_slots_outer_cells_and_singleton():
    model = _adapter((4,2,1)); scene = model.make_scene(_case())
    q = torch.tensor([[[.5,1.,1.], [0.,0.,0.], [2.,2.,2.]]], dtype=torch.double)
    query = model.make_query(scene, q)
    perm = torch.tensor([5,0,6,2,4,1,7,3])
    shuffled = replace(scene, environment_coordinates=scene.environment_coordinates[:,perm], environment_features=scene.environment_features[:,perm], environment_ids=scene.environment_ids[:,perm], environment_measure=scene.environment_measure[:,perm], environment_mask=scene.environment_mask[:,perm], environment_characteristic_lengths=scene.environment_characteristic_lengths[:,perm])
    alt = attach_wind_e_cover(query, shuffled, model.environment_shape)
    for i in range(3):
        def actual(s, c):
            active = c.anchor_weights[0,i] > 0
            return {int(s.environment_ids[0,k]): float(w) for k,w in zip(c.anchor_indices[0,i,active],c.anchor_weights[0,i,active])}
        assert actual(scene, query) == actual(shuffled, alt)
        torch.testing.assert_close(alt.anchor_weights[0,i].sum(), torch.tensor(1., dtype=torch.double))
    with pytest.raises(ValueError, match='outside'):
        model.make_query(scene, torch.tensor([[[2.01,1.,1.]]],dtype=torch.double))
    assert scene.environment_measure.sum().item() == 8.


def test_complete_native_derivative_and_strict_roundtrip():
    model = _adapter(); case = _case()
    q = torch.tensor([[[.6,.7,.9], [1.4,1.2,1.1]]],dtype=torch.double)
    centers = torch.tensor(case.module_centers,dtype=torch.double,requires_grad=True)
    direction = torch.tensor([[.3,-.2,0.],[-.1,.15,0.]],dtype=torch.double)
    value = model(case,q,centers)[0].sum()
    grad = torch.autograd.grad(value,centers)[0]
    directional = (grad*direction).sum()
    assert torch.isfinite(grad).all() and directional.abs() > 1.e-12
    for h in (1.e-3, 3.e-4):
        with torch.no_grad():
            fd = (model(case,q,centers+h*direction)[0].sum()-model(case,q,centers-h*direction)[0].sum())/(2*h)
        torch.testing.assert_close(fd, directional, rtol=5.e-4, atol=1.e-7)
    buffer = io.BytesIO()
    torch.save({"payload": model.state_payload(), "state_dict": model.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=True)
    copy = WindRelationAnchoredAdapter.from_state_payload(saved["payload"]).double().eval()
    copy.load_state_dict(saved["state_dict"], strict=True)
    torch.testing.assert_close(model(case,q,centers)[0],copy(case,q,centers)[0],rtol=0,atol=0)


def test_cover_continuity_masks_and_coordinate_derivative():
    model = _adapter(); scene = model.make_scene(_case())
    # Within a cell, interpolation of a linear coordinate must recover it.
    q = torch.tensor([[[.74,.8,1.], [1.,1.,1.]]], dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[True, False]])
    carrier = model.make_query(scene, q, query_mask=mask)
    indices = carrier.anchor_indices.clamp_min(0)
    positions = scene.environment_coordinates[0, indices[0], 0]
    interpolated = (positions * carrier.anchor_weights[0]).sum(-1)
    torch.testing.assert_close(interpolated[0], q[0,0,0])
    derivative = torch.autograd.grad(interpolated[0], q)[0][0,0,0]
    torch.testing.assert_close(derivative, torch.ones_like(derivative))
    assert carrier.anchor_weights[0,1].sum() == 0
    for x in (.749999, .750001):
        a = model.make_query(scene, torch.tensor([[[x,.8,1.]]],dtype=torch.double))
        selected = scene.environment_coordinates[0,a.anchor_indices.clamp_min(0)[0],0]
        torch.testing.assert_close((selected*a.anchor_weights[0]).sum(), torch.tensor(x,dtype=torch.double))
    duplicate = scene.environment_coordinates.clone(); duplicate[:,1] = duplicate[:,0]
    with pytest.raises(ValueError, match="one-to-one"):
        model.make_query(replace(scene,environment_coordinates=duplicate),q.detach())


def test_shape_measure_and_horizontal_design_contract():
    case = _case()
    for shape in ((4,4,4),(8,4,2),(8,8,2),(8,4,4),(8,8,4)):
        model = _adapter(shape); scene = model.make_scene(case)
        torch.testing.assert_close(scene.environment_measure.sum(), torch.tensor(8.,dtype=torch.double))
        expected = torch.tensor([2. / n for n in shape], dtype=torch.double)
        torch.testing.assert_close(scene.environment_characteristic_lengths[0, 0], expected)
    moved = torch.tensor(case.module_centers,dtype=torch.double); moved[0,2] += .1
    with pytest.raises(ValueError, match="horizontal"):
        model.make_scene(case,moved)
