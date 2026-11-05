"""Prescribed raster geometry enters only the opted-in original M records."""
import copy
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

from channelthermal.data.datasets import prescribed_raster_geometry
from channelthermal.data.collation import ChannelThermalBatchCollator
from channelthermal.member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from channelthermal.native_observations import prescribed_raster_area_ratio
from channelthermal.training.unified_task import _structure_only
from tests.anchored_fixtures import _base_config
from tests.test_r14_runtime_integration import STATS

@pytest.fixture
def source_raster():
    # The original generator is a local, ignored reference asset. Other feature
    # tests remain runnable in a clean checkout without that asset.
    source = Path(__file__).resolve().parents[1] / '1_Demo_ChannelThermal/src'
    if not (source / '_helpers_forward/channelthermal_common.py').is_file():
        pytest.skip('The original local raster generator is unavailable.')
    sys.path.insert(0, str(source))
    from _helpers_forward.channelthermal_common import DomainConfig, LayoutConfig, SimulationConfig, module_id_map
    return DomainConfig, LayoutConfig, SimulationConfig, module_id_map


def _structure():
    config = {'domain': {'nx': 12, 'ny': 6, 'lx': 12., 'ly': 6., 'module_radius': 1.},
              'layout': {'centers': [[2.5, 2.5], [3.5, 2.5], [10.5, 5.5]]}}
    present = np.array([1, 1, 1, 0], np.float32)
    centers, geometry = prescribed_raster_geometry(config, present)
    structure = {'module_centers': torch.tensor(centers[None], dtype=torch.float32),
                 'module_present': torch.tensor(present[None]), 'heat_powers': torch.tensor([[1., 2., 3., 0.]]),
                 'material_params': torch.tensor([[.01,.02,.03,1.,1.,1.]]),
                 'domain_length_x': torch.tensor([[12.]]), 'domain_length_y': torch.tensor([[6.]]),
                 'raster_module_centers': torch.tensor(centers[None]), 'raster_geometry': torch.tensor(geometry[None])}
    return structure, config


def test_source_raster_ownership_overlap_boundary_and_mask(source_raster):
    DomainConfig, LayoutConfig, SimulationConfig, module_id_map = source_raster
    structure, config = _structure()
    actual = prescribed_raster_area_ratio(structure['raster_module_centers'], structure['module_present'],
                                           structure['raster_geometry'], nx=12, ny=6).numpy()[0]
    source = SimulationConfig(domain=DomainConfig(nx=12,ny=6,lx=12.,ly=6.,module_radius=1.),
                              layout=LayoutConfig(centers=config['layout']['centers']))
    owner = module_id_map(source)
    expected = np.array([(owner == i).sum()/np.pi for i in range(4)])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    assert actual[-1] == 0 and actual[0] != actual[1]
    # A later original slot owns shared cells, and an out-of-domain disk is clipped.
    assert owner[2,3] == 1 and actual[2] < 5 / np.pi


def test_fractional_grid_and_boundary_follow_generator(source_raster):
    DomainConfig, LayoutConfig, SimulationConfig, module_id_map = source_raster
    source = SimulationConfig(domain=DomainConfig(nx=7,ny=5,lx=7.3,ly=5.2,module_radius=1.04),
                              layout=LayoutConfig(centers=[[.5, .52], [1.54, .52], [6.9, 4.8]]))
    owner = module_id_map(source)
    centers = torch.tensor([source.layout.centers],dtype=torch.float64)
    geometry = torch.tensor([[7.3,5.2,1.04,7.,5.]],dtype=torch.float64)
    actual = prescribed_raster_area_ratio(centers,torch.ones(1,3,dtype=torch.bool),geometry,nx=7,ny=5)[0].numpy()
    expected = np.array([(owner == i).sum() * (7.3/7) * (5.2/5) / (np.pi*1.04**2) for i in range(3)])
    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=0)


def test_opt_in_config_zero_column_preserves_h0_and_roundtrips():
    old_config = _base_config(); new_config = copy.deepcopy(old_config); new_config['module_input_dim'] = 10
    parent = ThermalDetailAnchoredMemberRoutingAdapter(STATS, core_config={
        'schema':'interaction_reused_detail_anchored_member_routing_v1','architecture':'R14-Detail-LocalECover',
        'locator_mode':'local_e_cover','base_model_config':old_config,'input_dataset':'thermal',
        'input_capability':'joint_prescribed_records_v1','query_tile_size':32,'retained_mass':.95,'score_calibration':'inherited'}, nx=12,ny=6,environment_nx=4,environment_ny=2).eval()
    candidate_config = copy.deepcopy(parent.core.get_config()); candidate_config['base_model_config'] = new_config
    child = ThermalDetailAnchoredMemberRoutingAdapter(STATS,core_config=candidate_config,nx=12,ny=6,environment_nx=4,environment_ny=2).eval()
    assert torch.count_nonzero(child.core.module_encoder[0].weight[:, 9]) == 0
    state = parent.state_dict(); key='core.module_encoder.0.weight'; weight=state[key]
    state[key] = torch.cat((weight[:,:9],torch.zeros_like(weight[:,:1]),weight[:,9:]), dim=1)
    child.load_state_dict(state,strict=True)
    structure,_ = _structure(); scene0=parent.scene_batch(structure); scene1=child.scene_batch(structure)
    torch.testing.assert_close(scene0.module_features,scene1.module_features[...,:9],rtol=0,atol=0)
    assert scene1.module_features[0,-1,-1] == 0
    assert scene1.module_ids.tolist() == scene0.module_ids.tolist()
    assert 'raster_module_centers' in _structure_only(structure)
    query=torch.tensor([[[.5,.5],[4.5,3.5],[11.5,5.5]]]);local=torch.tensor([[[0.,0.],[.5,0.]]])
    with torch.no_grad():
        a=parent.apply_native(parent.prepare_native(structure,query,physical_heat=structure['heat_powers'],local_query_points=local,ntheta=16),structure['heat_powers'])
        b=child.apply_native(child.prepare_native(structure,query,physical_heat=structure['heat_powers'],local_query_points=local,ntheta=16),structure['heat_powers'])
    for k in a: torch.testing.assert_close(a[k],b[k],atol=1e-6,rtol=1e-6)
    prepared=child.prepare_native(structure,query,physical_heat=structure['heat_powers'],local_query_points=local,ntheta=16)
    child.apply_native(prepared,structure['heat_powers'])['pred_field'].square().sum().backward()
    grad = child.core.module_encoder[0].weight.grad[:, 9]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    loaded=ThermalDetailAnchoredMemberRoutingAdapter.from_state_payload(child.state_payload(),state_dict=child.state_dict())
    assert loaded.core.module_input_dim==10
    with pytest.raises(ValueError,match='source-precision'):
        child.scene_batch({k:v for k,v in structure.items() if not k.startswith('raster_')})


def test_collation_preserves_fp64_geometry_and_original_slot_alignment():
    structure,_ = _structure()
    sample={k:v[0].clone() for k,v in structure.items()}
    sample['module_present'] = torch.tensor([1.,0.,1.,0.])
    sample['module_source_ids'] = torch.tensor([7,-1,31,-1])
    result=ChannelThermalBatchCollator()([{'structure':sample}])['structure']
    assert result['module_source_ids'].tolist() == [[7,31]]
    assert result['raster_module_centers'].dtype == torch.float64
    torch.testing.assert_close(result['raster_module_centers'][0],sample['raster_module_centers'][[0,2]])
    assert 'raster_geometry' in _structure_only(result)


def test_fresh_builder_matches_seed_zero_nine_feature_initialization():
    from channelthermal.training.r13_6_member_routing_task import _fresh_member_routing_adapter

    core_config = {
        'schema': 'interaction_reused_detail_anchored_member_routing_v1',
        'architecture': 'R14-Detail-LocalECover', 'locator_mode': 'local_e_cover',
        'base_model_config': _base_config(), 'input_dataset': 'thermal',
        'input_capability': 'joint_prescribed_records_v1', 'query_tile_size': 32,
        'retained_mass': .95, 'score_calibration': 'inherited',
    }
    expanded_config = copy.deepcopy(core_config)
    expanded_config['base_model_config']['module_input_dim'] = 10
    saved_rng = torch.random.get_rng_state().clone()
    original = _fresh_member_routing_adapter(ThermalDetailAnchoredMemberRoutingAdapter, STATS,
                                             core_config, heat_scale=2., seed=0, device='cpu')
    amended = _fresh_member_routing_adapter(ThermalDetailAnchoredMemberRoutingAdapter, STATS,
                                            expanded_config, heat_scale=2., seed=0, device='cpu')
    assert torch.equal(torch.random.get_rng_state(), saved_rng)
    original_state, amended_state = original.state_dict(), amended.state_dict()
    assert original_state.keys() == amended_state.keys()
    for key, value in original_state.items():
        candidate = amended_state[key]
        if key == 'core.module_encoder.0.weight':
            assert torch.equal(value, torch.cat((candidate[:, :9], candidate[:, 10:]), dim=1))
            assert torch.count_nonzero(candidate[:, 9]) == 0
        else:
            assert torch.equal(value, candidate), key
    assert core_config['base_model_config']['module_input_dim'] == 9
    assert expanded_config['base_model_config']['module_input_dim'] == 10
