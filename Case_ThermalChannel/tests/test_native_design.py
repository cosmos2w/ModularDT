"""Target-free Thermal design inputs preserve native geometry and scoring."""
import copy

import numpy as np
import torch

from channelthermal.native_design import build_native_design, score_native_design
from channelthermal.native_functionals import native_design_functionals


def config():
    return {
        'domain': {'nx': 12, 'ny': 6, 'lx': 12., 'ly': 6., 'module_radius': 1.},
        'layout': {'num_modules': 2, 'centers': [[2.5000001, 2.5], [3.5, 2.5]], 'heat_powers': [1., 2.]},
        'flow': {'re': 20., 'u_in': 1., 'viscosity_scale': 1., 'nu': None},
        'thermal': {'solid_alpha': .01, 'fluid_alpha': .02, 'solid_k': 1., 'fluid_k': 1.,
                    't_in': 0., 't_wall': 0.},
        'local_module': {'local_grid_size': 8, 'n_interface_points': 64},
    }


def test_source_precision_ownership_and_original_attached_identity():
    cfg = config()
    first = build_native_design(cfg, 'example')
    # FP64 excludes this cell, although rounding the first center to FP32
    # would put it exactly on the inclusive disk boundary.
    assert first.module_id_grid[2, 1] == -1
    assert np.float32(cfg['layout']['centers'][0][0]) == np.float32(2.5)
    assert first.module_id_grid[2, 3] == 1  # last overlapping member owns the cell
    assert first.structure['raster_module_centers'].dtype == torch.float64
    assert first.structure['module_centers'].dtype == torch.float32
    moved = copy.deepcopy(cfg)
    moved['layout']['centers'][0][0] += .2
    second = build_native_design(moved, 'example')
    assert first.fluid_ids == second.fluid_ids
    assert first.surface_ids == second.surface_ids
    assert first.material_ids == second.material_ids
    np.testing.assert_array_equal(first.fluid_xy_source, second.fluid_xy_source)
    np.testing.assert_allclose(second.surface_xy_source[0, :, 0] - first.surface_xy_source[0, :, 0], .2)
    np.testing.assert_allclose(second.material_xy_source[0, :, 0] - first.material_xy_source[0, :, 0], .2)
    np.testing.assert_array_equal(first.surface_xy_source[1], second.surface_xy_source[1])


def test_real_small_native_model_scoring_without_targets():
    from channelthermal.member_routing import ThermalDetailAnchoredMemberRoutingAdapter
    from tests.anchored_fixtures import _base_config
    from tests.test_r14_runtime_integration import STATS

    model = ThermalDetailAnchoredMemberRoutingAdapter(STATS, core_config={
        'schema': 'interaction_reused_detail_anchored_member_routing_v1',
        'architecture': 'R14-Detail-LocalECover', 'locator_mode': 'local_e_cover',
        'base_model_config': _base_config(), 'input_dataset': 'thermal',
        'input_capability': 'joint_prescribed_records_v1', 'query_tile_size': 32,
        'retained_mass': .95, 'score_calibration': 'inherited',
    }, nx=12, ny=6, environment_nx=4, environment_ny=2).eval()
    design = build_native_design(config(), 'example')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            actual = score_native_design(model, design, tau=.4149511814117432, mode='restricted')
            prepared = model.prepare_native(design.structure, design.fluid_xy, physical_heat=design.heat,
                local_query_points=design.local_query_points, ntheta=64, scene_key=(design.case_id,))
            with model.core.read_execution('restricted'):
                native, counts, _ = model.apply_native_with_work_counts(prepared, design.heat)
            field, material = native['pred_field'], native['pred_internal_temperature'][..., 0]
            fmask = torch.as_tensor(design.fluid_valid)[None] & prepared.stencils['fluid'].valid
            mmask = torch.as_tensor(design.material_valid)[None] & prepared.stencils['material'].valid.reshape(material.shape)
            j, p = native_design_functionals(material, mmask, field[..., 2], fmask,
                                             design.fluid_xy[..., 0], design.domain_length_x, .4149511814117432)
    finally:
        torch.set_num_threads(old_threads)
    for key in ('pred_field', 'pred_interface', 'pred_internal_temperature'):
        torch.testing.assert_close(actual['native'][key], native[key], rtol=0, atol=0)
    torch.testing.assert_close(actual['functionals']['J'], j, rtol=0, atol=0)
    torch.testing.assert_close(actual['functionals']['P'], p, rtol=0, atol=0)
    assert torch.equal(actual['fields']['flow/p']['mask'], fmask)
    assert torch.equal(actual['fields']['temperature/material']['mask'], mmask)
    for key, value in counts.items():
        assert actual['work']['core_decode_primary_' + key] == value
