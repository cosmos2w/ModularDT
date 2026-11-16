import numpy as np
import torch

from channelthermal.training.thermal_field_response import sample_thermal_field_observations


def test_common_measure_keeps_duplicate_masses_and_fixed_denominators():
    field = np.arange(3 * 3 * 5, dtype=np.float64).reshape(3, 3, 5)
    rows = np.array([[.5, .5], [.5, .5], [1.5, 1.5], [1., 1.]], dtype=np.float64)
    groups = [dict(steady_field=field, module_mask=np.zeros((3, 3), bool),
                   sampled_points=rows, sampled_point_weights=np.array([1., 2., 3., 4.])),
              dict(steady_field=field + 2, module_mask=np.zeros((3, 3), bool),
                   sampled_points=rows, sampled_point_weights=np.array([1., 2., 3., 4.]))]
    observation = sample_thermal_field_observations(groups, (3., 3.), seed=12, sample_per_side=4)
    assert observation.eligible_counts == (4, 4)
    assert observation.coordinates.dtype == torch.float64
    torch.testing.assert_close(observation.references[1] - observation.references[0],
                               torch.full((8, 5), 2., dtype=torch.float64))
    torch.testing.assert_close(observation.masses.sum(0), torch.ones(5, dtype=torch.float64))
    assert (observation.coordinates == torch.tensor([.5, .5])).all(-1).sum() == 4


def test_missing_channel_and_nonzero_corner_geometry():
    field = np.ones((3, 3, 5), dtype=np.float64)
    other = field.copy()
    other[..., 2] = np.nan
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    rows = np.array([[.5, .5], [1., 1.], [2.5, 2.5]])
    groups = [dict(steady_field=field, module_mask=np.zeros_like(mask), sampled_points=rows),
              dict(steady_field=other, module_mask=mask, sampled_points=rows)]
    observation = sample_thermal_field_observations(groups, (3., 3.), seed=0)
    assert observation.eligible_counts == (2, 2)
    assert not observation.mask[:, 2].any()
    assert not observation.masses[:, 2].any()
    assert not (observation.coordinates == torch.tensor([1., 1.])).all(-1).any()


def test_field_only_matches_full_roles_and_interpolates_off_centre_curl():
    from test_thermal_content_hypergraph_v2 import _model, _receivers, _structure
    from channelthermal.native_observations import interpolate_native, native_cell_center_stencil

    model = _model()
    fluid, _ = _receivers()
    field = model.prepare_native(_structure(), fluid, ntheta=16, scene_key=('test',), field_only=True)
    full = model.prepare_native(_structure(), fluid, ntheta=16, scene_key=('test',))
    assert set(field.stencils) == {'fluid'}
    def extract(prepared):
        xy = prepared.union.grid_coordinates
        flow = torch.stack((xy[..., 0] * .1, xy[..., 1] * .2, xy[..., 0] + xy[..., 1]), -1)
        temperature = (xy[..., 0] - xy[..., 1])[..., None]
        return model._extract_native_roles(prepared, temperature, flow)['pred_field']
    dense = extract(full)
    torch.testing.assert_close(extract(field), dense)
    offcentre = torch.tensor([[[2.3, 2.1], [7.1, 4.2], [2.3, 2.1]]])
    sparse = model.prepare_native(_structure(), offcentre, ntheta=16, scene_key=('test',), field_only=True)
    stencil = native_cell_center_stencil(offcentre, torch.tensor([[12., 6.]]), nx=8, ny=6)
    torch.testing.assert_close(extract(sparse), interpolate_native(dense, stencil), rtol=1e-5, atol=1e-5)
