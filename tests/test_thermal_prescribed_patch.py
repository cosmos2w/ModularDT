"""The E patches reconstruct the prescribed source raster, including edits."""
import copy

import numpy as np
import torch

from channelthermal.data.datasets import prescribed_patch_inputs
from channelthermal.prescribed_patch import prescribed_environment_patch


def config():
    return {"domain": {"nx": 128, "ny": 64, "lx": 12., "ly": 6., "module_radius": .45},
            "layout": {"centers": [[2.5000001, 2.5], [2.8, 2.5]], "heat_powers": [1.234567891, 2.5]},
            "thermal": {"solid_alpha": .01, "fluid_alpha": .02, "solid_k": 3., "fluid_k": 1.}}


def make(cfg, physical=True):
    arrays = prescribed_patch_inputs(cfg, np.array([1, 1, 0]))
    args = [torch.as_tensor(arrays[k])[None] for k in ("raster_module_centers", "raster_geometry", "raster_heat_powers", "raster_material_params")]
    return prescribed_environment_patch(args[0], torch.tensor([[1, 1, 0]]), *args[1:], heat_scale=2., physical=physical)[0].numpy()


def unpack(patch):
    out = np.zeros((64, 128, 7))
    seen = np.zeros((64, 128), dtype=int)
    for ey in range(8):
        for ex in range(24):
            xi = np.flatnonzero(((2 * np.arange(128) + 1) * 24 // 256) == ex)
            out[ey * 8:(ey + 1) * 8, xi] = patch[ey * 24 + ex, :, :len(xi)]
            seen[ey * 8:(ey + 1) * 8, xi] += 1
            assert np.all(patch[ey * 24 + ex, :, len(xi):] == 0)
    assert np.all(seen == 1)
    return out


def test_reconstruct_source_ownership_and_padding():
    cfg = config()
    patch = make(cfg)
    maps = unpack(patch)
    x = np.linspace(0, 12., 128, endpoint=False) + .5 * 12. / 128
    y = np.linspace(0, 6., 64, endpoint=False) + .5 * 6. / 64
    xx, yy = np.meshgrid(x, y)
    owner = np.full((64, 128), -1)
    for i, (cx, cy) in enumerate(cfg['layout']['centers']):
        owner[np.hypot(xx - cx, yy - cy) <= .45] = i
    expected = np.zeros(owner.shape)
    for i, value in enumerate(cfg['layout']['heat_powers']):
        expected[owner == i] = value / 2.
    np.testing.assert_array_equal(maps[..., 3], owner >= 0)
    np.testing.assert_array_equal(maps[..., 6], expected)
    assert np.all(maps[..., 0] == 1) and np.any(maps[..., 3] == 0)
    np.testing.assert_array_equal(maps[..., 4], (owner >= 0) * .5)
    scaffold = make(cfg, False)
    np.testing.assert_array_equal(scaffold[..., :3], patch[..., :3])
    assert not np.any(scaffold[..., 3:])


def test_edited_prescriptions_rebuild_physical_channels():
    baseline = config()
    edited = copy.deepcopy(baseline)
    edited['layout']['centers'][0][0] += .3
    edited['layout']['heat_powers'][1] += 1.
    edited['thermal']['solid_k'] = 5.
    first, second = make(baseline), make(edited)
    np.testing.assert_array_equal(first[..., :3], second[..., :3])
    assert np.any(first[..., 3] != second[..., 3])
    assert np.any(first[..., 4] != second[..., 4])
    assert np.any(first[..., 6] != second[..., 6])


def test_adapter_mode_and_saved_payload():
    from channelthermal.member_routing import ThermalDetailAnchoredMemberRoutingAdapter
    from tests.anchored_fixtures import _base_config
    from tests.test_r14_runtime_integration import STATS

    core = _base_config()
    core['environment_input_dim'] = 342
    model = ThermalDetailAnchoredMemberRoutingAdapter(STATS, core_config={
        'schema': 'interaction_reused_detail_anchored_member_routing_v1',
        'architecture': 'R14-Detail-LocalECover', 'locator_mode': 'local_e_cover',
        'base_model_config': core, 'input_dataset': 'thermal',
        'input_capability': 'joint_prescribed_records_v1', 'query_tile_size': 32,
        'retained_mass': .95, 'score_calibration': 'inherited'},
        heat_scale=2., prescribed_patch_mode='physical')
    cfg = config()
    cfg.update({'flow': {'re': 20., 'u_in': 1., 'nu': .01},
                'local_module': {'local_grid_size': 8, 'n_interface_points': 64}})
    cfg['layout']['num_modules'] = 2
    cfg['thermal'].update({'t_in': 0., 't_wall': 0.})
    from channelthermal.native_design import build_native_design
    design = build_native_design(cfg, 'case')
    scene = model.scene_batch({**design.structure, 'heat_powers': design.heat})
    assert scene.environment_features.shape == (1, 192, 342)
    assert scene.module_features.shape == (1, 2, 9)
    loaded = ThermalDetailAnchoredMemberRoutingAdapter.from_state_payload(model.state_payload(), state_dict=model.state_dict())
    torch.testing.assert_close(scene.environment_features, loaded.scene_batch({**design.structure, 'heat_powers': design.heat}).environment_features)
    assert loaded.prescribed_patch_mode == 'physical'
