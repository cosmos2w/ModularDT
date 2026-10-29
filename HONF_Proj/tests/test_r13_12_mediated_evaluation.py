"""Evidence reductions must distinguish live source mass from actual reads."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from honf_runtime.r13_12_mediated_evaluation import (
    QUERY_IDS, TRAIN24_CASES, _evidence_native_arrays, _flatten_arrays, summarize_mediated_trace,
)


def test_empty_route_mass_and_two_restrictions_use_actual_composition():
    trace = {
        "anchor_read_indices": torch.tensor([[0, 0, 0], [0, 1, 0]]),
        "anchor_read_E_slots": torch.tensor([0, 1]),
        "anchor_read_member_read_index": torch.tensor([0, 0, 0, 1, 1]),
        "anchor_read_member_index": torch.tensor([0, 2, 3, 1, 3]),
        "anchor_read_access_prior": torch.tensor([.5, .4, .2, .7, .1]),
        "anchor_read_probabilities": torch.tensor([1., .75, .25, 1., 1.]),
        "anchor_read_strength": torch.tensor([[.4, .6], [0., .8]]),
        "mediated_source_alpha": torch.tensor([[.2, .5, .3], [.1, .2, .7]]),
        "mediated_source_selected": torch.tensor([[True, True, False], [True, False, True]]),
        "mediated_environment_prior": torch.tensor([[.4, .3], [0., .1]]),
        "mediated_conditional_rows": [(torch.tensor([0, 1]), torch.tensor([[.6, .4], [0., 0.]]), torch.tensor([4., 0.]))],
    }
    result = summarize_mediated_trace(trace, module_count=2, full_environment_mass=np.array([.7, .3]))
    assert result["source_mass_assigned_to_empty_ME_rows"]["mean"] == pytest.approx(.5)
    assert result["omitted_E_path_mass_due_to_source_masking"]["mean"] == pytest.approx(.1)
    assert result["retained_E_path_mass_relative_to_full_composition"]["mean"] == pytest.approx((.6 / .7 + .1 / .3) / 2)
    assert result["zero_M_source_fraction"] == 0
    assert result["zero_M_value_strength_fraction"] == .5
    assert result["final_own_E_beta_before_locator_blend"]["mean"] == pytest.approx((.75 * .6 + .8) / 2)


def test_ragged_trace_export_preserves_ids_without_pickle_objects():
    arrays, metadata = {}, {}
    _flatten_arrays([(torch.tensor([37, 11]), torch.tensor([[0., 1.], [1., 0.]])), "cycle1/me"],
                    "conditional", arrays, metadata)
    np.testing.assert_array_equal(arrays["conditional_0_0"], [37, 11])
    assert metadata["conditional_1"] == "cycle1/me"
    assert all(array.dtype.kind != "O" for array in arrays.values())
    assert QUERY_IDS.shape == (512,) and len(np.unique(QUERY_IDS)) == 512
    assert len(TRAIN24_CASES) == len(set(TRAIN24_CASES)) == 24


@pytest.mark.parametrize("ntheta,shared_local", [(16, True), (64, False)])
def test_native_receiver_export_preserves_physical_catalogue_and_historical_arrays(ntheta, shared_local):
    from channelthermal.contextual_joint import ThermalContextualJointAdapter
    from channelthermal.native_observations import interpolate_native
    from honf_runtime.member_routing_evaluation import _native_arrays_with_residuals

    stats = {"field_mean_by_channel": np.zeros(5), "field_std_by_channel": np.ones(5),
             "interface_targets_std": np.ones(2), "internal_temperature_std": np.ones(1)}
    model = ThermalContextualJointAdapter(stats, nx=8, ny=6, core=torch.nn.Identity())
    structure = {"module_centers": torch.tensor([[[.1, 2.], [7.5, 4.], [0., 0.]]]),
                 "module_present": torch.tensor([[1., 1., 0.]]),
                 "module_source_ids": torch.tensor([[37, 8, 99]]),
                 "material_params": torch.tensor([[.02, .01, .02, 1., 2., .37]]),
                 "heat_powers": torch.tensor([[.4, .8, 0.]]),
                 "re": torch.tensor([[50.]]), "u_in": torch.ones(1, 1)}
    scene = model.scene_batch(structure)
    fluid = torch.tensor([[[2.25, 2.5], [3.75, 3.5]]])
    local = torch.tensor([[[-.4, 0.], [0., .7], [.6, -.3]]])
    if not shared_local:
        local = local[:, None].expand(-1, 3, -1, -1).clone()
        local[:, 1, :, 0] *= -.5
    query, union, stencils, _, _, _, delta, conductivity, _ = model._query_union(scene, fluid, local, ntheta)
    prepared = SimpleNamespace(scene_batch=scene, query_batch=query, stencils=stencils,
                               source_present=scene.module_mask, local_query_points=local,
                               ntheta=ntheta, interface_delta=delta, interface_conductivity=conductivity)
    targets = SimpleNamespace(field_targets=torch.zeros(1, 2, 5), point_weights=torch.ones(1, 2),
                              interface_target=torch.zeros(1, 3, ntheta, 2),
                              interface_valid_mask=torch.ones(1, 3, ntheta),
                              material_targets=torch.zeros(1, 3, 3))
    batch = SimpleNamespace(targets=targets, receivers=SimpleNamespace(fluid_xy=fluid, heat=torch.ones(1, 3)))
    prediction = SimpleNamespace(native_prepared=prepared, native_main={
        "pred_field": torch.ones(1, 2, 5), "pred_interface": torch.ones(1, 3, ntheta, 2),
        "pred_internal_temperature": torch.ones(1, 3, 3, 1)})
    historical = _native_arrays_with_residuals(prediction, batch, "geometry-fixture")
    arrays = _evidence_native_arrays(prediction, batch, "geometry-fixture")
    for key, value in historical.items():
        np.testing.assert_array_equal(arrays[key], value)
    np.testing.assert_array_equal(arrays["module_ids"], [37, 8, -1])
    expanded = local[:, None].expand(-1, 3, -1, -1) if shared_local else local
    np.testing.assert_array_equal(arrays["material_local_query_points"], expanded[0])
    for role in ("surface", "outside", "material"):
        xy = arrays[role + "_receiver_xy"].reshape(-1, 2)
        valid = stencils[role].valid[0].numpy()
        extracted = interpolate_native(union.grid_coordinates, stencils[role])[0].numpy()
        np.testing.assert_allclose(xy[valid], extracted[valid], atol=1e-6)
    assert arrays["surface_receiver_xy"][0, ntheta // 2, 0] < 0
    assert not arrays["surface_valid"][0, ntheta // 2]
    assert arrays["surface_receiver_xy"].shape == (3, ntheta, 2)
    assert arrays["material_receiver_xy"].shape == (3, 3, 2)
    assert all(value.dtype.kind != "O" for value in arrays.values())
