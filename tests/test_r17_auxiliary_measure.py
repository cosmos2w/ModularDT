from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from channelthermal.native_observations import native_cell_center_stencil
from channelthermal.training.thermal_native_port_task import AuxiliaryFluidMeasure, ThermalNativePortTask
from channelthermal.training.joint_member_routing_task import ThermalJointPairPredictions
from channelthermal.training.thermal_family_objective import _endpoint_functionals
from channelthermal.training.thermal_state_response_task import _native_endpoint_batch
from honf_runtime.state_response import masked_weighted_mse, normalized_response_mse


def test_weighted_duplicate_interpolation_and_gradient():
    xy = torch.tensor([[[1., 1.], [1., 1.], [.5, .5]]], dtype=torch.float64)
    stencil = native_cell_center_stencil(xy, torch.tensor([[2., 2.]], dtype=torch.float64), nx=2, ny=2)
    measure = AuxiliaryFluidMeasure(stencil, torch.tensor([[1., 2., 1.]], dtype=torch.float64))
    residual = torch.tensor([[1., 3., 5., 7.]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones_like(residual, dtype=torch.bool)
    loss = measure.mse(residual, mask)
    assert loss.item() == pytest.approx((3 * 4**2 + 1) / 4)
    loss.backward()
    assert residual.grad.tolist()[0] == pytest.approx([2., 1.5, 1.5, 1.5])
    # An endpoint with a blocked corner excludes both interpolated rows; the
    # common mask therefore differs from the other endpoint's own support.
    edited_mask = torch.tensor([[True, False, True, True]])
    assert measure.mse(residual, edited_mask).item() == pytest.approx(1.)
    assert measure.mse(residual, mask & edited_mask).item() == pytest.approx(1.)
    with pytest.raises(ValueError, match='positive mass'):
        measure.mse(residual, torch.zeros_like(mask))


def test_uniform_weight_agrees_with_existing_response_reduction():
    base = torch.tensor([[1., 2., 9.]])
    edit = torch.tensor([[3., 5., 4.]])
    reference = torch.zeros_like(base)
    mask = torch.tensor([[True, True, False]])
    actual = masked_weighted_mse((edit - base) / 2, mask, torch.ones_like(base))
    expected = normalized_response_mse(base, edit, reference, reference, mask, 2)
    torch.testing.assert_close(actual, expected)


def test_catalogue_loader_preserves_rows_weights_and_precision(tmp_path):
    path = tmp_path / 'native.h5'
    rows = np.array([[.5, .5, 11.], [.5, .5, 12.], [1., 1., 13.]], dtype=np.float64)
    with h5py.File(path, 'w') as h5:
        group = h5.create_group('cases/a')
        group['sampled_points'] = rows
        group['sampled_point_weights'] = [1., 2., 4.]
    provider = SimpleNamespace(training_case_ids=('a',), data_path=path, adapter=SimpleNamespace(nx=2, ny=2))
    batch = SimpleNamespace(receivers=SimpleNamespace(fluid_xy=torch.zeros(1, 4, 2)),
        scene_inputs=SimpleNamespace(structure={'domain_length_x': torch.tensor([[2.]]),
                                                'domain_length_y': torch.tensor([[2.]])}))
    measure = ThermalNativePortTask.auxiliary_catalogue_measure(provider, 'a', batch)
    assert measure.stencil.weights.dtype == torch.float64
    assert measure.weights.tolist() == [[1., 2., 4.]]
    assert measure.stencil.indices.shape == (1, 3, 4)
    with pytest.raises(ValueError, match='TRAIN membership'):
        ThermalNativePortTask.auxiliary_catalogue_measure(provider, 'b', batch)


def test_only_fluid_terms_change_in_both_views():
    names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid',
             'temperature/surface', 'temperature/material', 'q_proxy')
    task = ThermalNativePortTask.__new__(ThermalNativePortTask)
    task.native_weights = dict.fromkeys(names, 1.)
    task.model = SimpleNamespace(field_std=torch.ones(5))
    task.outside_temperature_weight = .05
    task._role_losses = lambda prediction, targets: {name: torch.tensor([7.]) for name in names}
    def branch(value):
        stencil = {'fluid': SimpleNamespace(valid=torch.ones(1, 4, dtype=torch.bool)),
                   'outside': SimpleNamespace(valid=torch.ones(1, 1, 1, dtype=torch.bool)),
                   'material': SimpleNamespace(valid=torch.ones(1, 1, 1, dtype=torch.bool))}
        return SimpleNamespace(native_prepared=SimpleNamespace(source_present=torch.ones(1, 1), stencils=stencil),
            native_main={'pred_field': torch.full((1, 4, 5), value),
                         'pred_internal_temperature': torch.tensor([[[[2.]]]]),
                         'pred_outside_temperature': torch.tensor([[[3.]]])})
    predictions = ThermalJointPairPredictions(branch(2.), branch(4.), {})
    targets = SimpleNamespace(case_ids=('a',), field_targets=torch.zeros(1, 4, 5),
        point_weights=torch.ones(1, 4), outside_temperature=torch.ones(1, 1, 1),
        outside_valid=torch.ones(1, 1, 1, dtype=torch.bool), material_targets=torch.ones(1, 1, 1))
    stencil = native_cell_center_stencil(torch.tensor([[[.5, .5]]]), torch.tensor([[2., 2.]]), nx=2, ny=2)
    measure = AuxiliaryFluidMeasure(stencil, torch.ones(1, 1))
    batch = SimpleNamespace(targets=targets, receivers=SimpleNamespace(
        fluid_xy=torch.tensor([[[.01, 0.], [.5, 0.], [1.5, 0.], [1.99, 0.]]])))
    raw = {'config': {'domain': {'lx': 2.}}}
    functionals_before = _endpoint_functionals(predictions, batch, raw, 1.)
    original = task._native_terms(predictions, targets)
    measured = task._native_terms(predictions, targets, measure)
    torch.testing.assert_close(_endpoint_functionals(predictions, batch, raw, 1.), functionals_before)
    for name in names[:5]:
        assert measured[name].numerator.item() == pytest.approx((2**2 + 4**2) / 2)
        assert original[name].numerator.item() == pytest.approx(7.)
    for name in (*names[5:], 'port/outside_temperature'):
        torch.testing.assert_close(measured[name].numerator, original[name].numerator)
    assert targets.point_weights.tolist() == [[1., 1., 1., 1.]]


def test_endpoint_binding_retains_source_precision_and_neural_view(monkeypatch):
    from channelthermal.training.content_hypergraph_task import ThermalContentHypergraphTask
    center = [0.700000000123, 1.100000000456]
    heat = 1.200000000789
    config = {'layout': {'centers': [center], 'heat_powers': [heat]},
              'domain': {'lx': 2., 'ly': 2., 'module_radius': .45, 'nx': 2, 'ny': 2},
              'thermal': {'solid_alpha': .01, 'fluid_alpha': .02, 'solid_k': 1., 'fluid_k': 1.}}
    structure = {'module_present': np.ones(1, np.float32), 'module_centers': np.zeros((1, 2), np.float32),
                 'heat_powers': np.zeros(1, np.float32), 'material_params': np.ones(6, np.float32),
                 'raster_heat_powers': np.zeros(1, np.float64)}
    original = {'case_id': 'a', 'structure': structure}
    raw = {'config': config, 'field': np.zeros((2, 2, 5), np.float32),
           'fluid_mask': np.ones((2, 2), bool), 'material_mask': np.ones((2, 2), bool),
           'material': np.zeros((1, 2, 2), np.float32), 'interface_target': np.zeros((1, 64, 2), np.float32)}
    monkeypatch.setattr(ThermalContentHypergraphTask, '_native_case_batch',
                        lambda provider, cases, indices, **kwargs: cases[0])
    case = _native_endpoint_batch(None, original, raw, 'baseline')
    actual = case['structure']
    assert actual['raster_module_centers'].dtype == np.float64
    assert np.array_equal(actual['raster_module_centers'][0], np.asarray(center, np.float64))
    assert actual['raster_heat_powers'][0] == heat
    assert actual['module_centers'].dtype == np.float32
    assert np.array_equal(actual['module_centers'][0], np.asarray(center, np.float32))
    assert actual['heat_powers'][0] == np.float32(heat)
    assert not np.array_equal(actual['raster_module_centers'][0], actual['module_centers'][0].astype(np.float64))
    assert np.array_equal(original['structure']['module_centers'], np.zeros((1, 2), np.float32))
