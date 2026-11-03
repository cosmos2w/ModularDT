import json

import numpy as np
import pytest

from channelthermal.training.thermal_finite_corpus import _functionals, fitting_cases, load_raw_endpoint


def test_raw_endpoint_preserves_temperature_and_separate_h_validity(tmp_path):
    (tmp_path / 'scene').mkdir()
    config = {'domain': {'lx': 12., 'ly': 6., 'nx': 4, 'ny': 2, 'module_radius': .45},
              'layout': {'centers': [[6., 3.]], 'heat_powers': [1.]}}
    (tmp_path / 'case_config.json').write_text(json.dumps(config))
    (tmp_path / 'frame_index.csv').write_text('file,converged\nframe.npz,1\n')
    response = np.zeros((1, 64, 9), np.float32)
    response[..., 3] = 2.; response[..., 4] = 1.; response[..., 5] = 3.
    response[0, 0, 3] = 1.  # Unresolved h, but outside T remains a valid teacher.
    np.savez(tmp_path / 'scene' / 'frame.npz', u=np.zeros((2, 4)), v=np.zeros((2, 4)),
             p=np.zeros((2, 4)), omega=np.zeros((2, 4)), temperature=np.ones((2, 4)),
             module_mask=np.zeros((2, 4)), module_id=np.full((2, 4), -1),
             module_internal_temperature=np.ones((1, 2, 2)), module_internal_mask=np.ones((2, 2)),
             interface_response=response, interface_feature_names=np.asarray(
                 ['theta', 'normal_x', 'normal_y', 'T_surface', 'T_outside', 'q_normal', 'u_normal', 'u_tangent', 'h_proxy'], dtype='S'))
    result = load_raw_endpoint(tmp_path)
    assert result['teacher_ports'][0, 0, 3] == 1
    assert result['outside_valid'][0, 0] and not result['teacher_h_valid'][0, 0]
    assert result['teacher_h_valid'][0, 1] and result['teacher_ports'][0, 1, 4] == 3
    (tmp_path / 'frame_index.csv').write_text('file,converged\nframe.npz,0\n')
    with pytest.raises(ValueError, match='converged'):
        load_raw_endpoint(tmp_path)


def test_fitting_selection_excludes_holdout_and_infeasible(tmp_path):
    p = tmp_path / 'recipe.json'
    p.write_text(json.dumps({'cases': [{'case_id': '1', 'role': 'fit', 'feasible': True},
        {'case_id': '2', 'role': 'response_holdout', 'feasible': True},
        {'case_id': '3', 'role': 'fit', 'feasible': False}]}))
    assert [case['case_id'] for case in fitting_cases(p)] == ['1']


def test_full_material_and_design_specific_pressure_bands():
    field = np.zeros((1, 100, 5), dtype=np.float32)
    field[..., 2] = np.arange(100)
    mask = np.ones((1, 100), dtype=bool)
    mask[0, 0] = False
    state = {'field': field, 'fluid_mask': mask, 'material': np.array([[[1., 3.], [5., 100.]]]),
             'material_mask': np.array([[True, True], [True, False]]),
             'config': {'domain': {'lx': 10.}}}
    objective, pressure = _functionals(state, 2.)
    expected = 2 * (np.log(np.exp(np.array([1., 3., 5.]) / 2).mean()))
    assert objective == pytest.approx(expected)
    assert pressure == pytest.approx(np.arange(1, 8).mean() - np.arange(92, 100).mean())
