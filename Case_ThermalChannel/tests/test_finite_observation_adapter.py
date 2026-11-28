"""Thermal source-row selection and native linear extraction contracts."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from channelthermal.flow_curl import build_native_curl_read_stencil, physical_uvp_native_curl
from channelthermal.native_observations import (
    build_native_solid_mask, interpolate_native, native_cell_center_stencil,
)
from channelthermal.training.observation_response import ThermalObservationBinding
from honf_runtime.finite_observations import extract_family_predictions


@pytest.fixture
def binding(tmp_path):
    path = tmp_path / 'thermal_sources.h5'
    domain = {'nx': 8, 'ny': 6, 'lx': 8., 'ly': 6., 'module_radius': .8}
    with h5py.File(path, 'w') as data:
        data['interface_condition_feature_names'] = np.asarray(
            ['theta', 'normal_x', 'normal_y', 'T_outside', 'u_normal', 'u_tangent', 'h_proxy', 'h_effective'],
            dtype='S')
        cases = data.create_group('cases')
        for case_index, case_id in enumerate(('first_scene', 'second_scene')):
            group = cases.create_group(case_id)
            group['case_config_json'] = json.dumps({'domain': domain, 'thermal': {'solid_k': 2., 'fluid_k': 1.}})
            group['module_centers'] = np.array([[2.5, 2.5], [5.5, 3.5], [3.5, 3.5]], dtype=np.float64)
            group['module_present'] = ([1, 1, 0] if case_index == 0 else [0, 0, 1])
            cells = np.array([[x + .5, y + .5] for y in range(1, 5) for x in range(1, 7)], dtype=np.float64)
            xy = np.tile(cells, (4, 1))
            values = np.arange(len(xy) * 5, dtype=np.float64).reshape(-1, 5) / 7 + case_index
            values[2, 2] = np.nan
            group['sampled_points'] = np.column_stack((xy, values))
            point_weights = np.ones(len(xy), dtype=np.float64)
            point_weights[0] = 0
            point_weights[1::3] = 2
            group['sampled_point_weights'] = point_weights
            group['module_internal_mask'] = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            group['module_internal_temperature'] = np.arange(27, dtype=np.float64).reshape(3, 3, 3) + case_index
            interface = np.arange(3 * 64 * 2, dtype=np.float64).reshape(3, 64, 2) / 13 + case_index
            interface[0, 2, 1] = np.nan
            group['interface_target'] = interface
            valid = np.ones((3, 64), dtype=np.float32)
            valid[0, 3] = 0
            group['interface_condition_valid_mask'] = valid
            condition = np.zeros((3, 64, 8), dtype=np.float64)
            condition[:, :, 3] = np.arange(3 * 64).reshape(3, 64) / 11 + case_index
            group['interface_condition'] = condition
            material = group.create_group('material_parameters')
            material.attrs['solid_k'], material.attrs['fluid_k'] = 2., 1.
    names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid',
             'temperature/material', 'temperature/surface', 'q_proxy')
    model = SimpleNamespace(nx=8, ny=6, field_std=torch.tensor([2., 3., 4., 5., 6.]),
        internal_temperature_std=torch.tensor([7.]), interface_target_std=torch.tensor([8., 9.]))
    provider = SimpleNamespace(data_path=path, model=model, training_case_ids=('first_scene', 'second_scene'),
        validation_case_ids=(), native_weights={name: .2 for name in names}, outside_temperature_weight=.05)
    return ThermalObservationBinding(provider, {})


def _family_map(scene):
    return {family.name: family for family in scene.families}


def test_selected_rows_preserve_source_identity_values_measures_and_unmatched_members(binding):
    first = binding.finite_observation_scene('first_scene', seed=12)
    second = binding.finite_observation_scene('second_scene', seed=12)
    families = _family_map(first)
    assert set(families) == set(binding.finite_observation_family_weights())
    assert binding.finite_observation_frame().upper == (8., 6.)
    assert 'module_peak_temperature' in binding.finite_observation_unsupported()
    assert all(0 < len(f.row_ids) <= 64 and np.all(f.measures > 0) for f in first.families)
    assert all(f.frame == 'thermal_physical_xy' and f.time == 'converged_final_window' for f in first.families)
    first_fluid = families['flow/u']
    assert len(first_fluid.row_ids) == 64 and len(set(first_fluid.row_ids)) == 64
    assert len(np.unique(first_fluid.coordinates, axis=0)) < len(first_fluid.row_ids)
    assert first_fluid.references.dtype == np.float64 and first_fluid.coordinates.dtype == np.float64
    assert first_fluid.row_ids == families['flow/v'].row_ids == families['temperature/fluid'].row_ids
    with h5py.File(binding.provider.data_path, 'r') as data:
        source = data['cases']['first_scene']
        index = np.array([int(row.split(':')[-1]) for row in first_fluid.row_ids])
        np.testing.assert_array_equal(first_fluid.references, source['sampled_points'][...][index, 2])
        np.testing.assert_array_equal(first_fluid.measures, source['sampled_point_weights'][...][index])
        assert 0 not in index
        material = families['temperature/material']
        source_material = source['module_internal_temperature'][...].reshape(3, -1)
        expected_material = [source_material[int(row.split(':')[1]), int(row.split(':')[3])]
                             for row in material.row_ids]
        np.testing.assert_array_equal(material.references, expected_material)
    assert all(row.startswith(('module:0:', 'module:1:')) for row in families['temperature/material'].row_ids)
    assert all(row.startswith('module:2:') for row in _family_map(second)['temperature/material'].row_ids)
    assert len(families['temperature/material'].row_ids) == 10
    assert all(dependency.scene_id == 'first_scene' for family in first.families
               for dependency in family.extraction.dependencies)
    repeat = binding.finite_observation_scene('first_scene', seed=12)
    assert [f.row_ids for f in first.families] == [f.row_ids for f in repeat.families]
    different = binding.finite_observation_scene('first_scene', seed=13)
    assert _family_map(different)['flow/u'].row_ids != first_fluid.row_ids
    assert max(len(f.extraction.dependencies) for f in first.families) <= 48


def test_physical_affine_conversion_and_complete_native_stencils(binding):
    families = _family_map(binding.finite_observation_scene('first_scene', count=32, seed=21))
    lengths = torch.tensor([[8., 6.]])
    centers = torch.tensor([[[2.5, 2.5], [5.5, 3.5], [3.5, 3.5]]])
    present = torch.tensor([[True, True, False]])
    solid = build_native_solid_mask(centers, present, torch.tensor([[.8]]), lengths, nx=8, ny=6)
    index = torch.arange(48)
    x = (index % 8 + .5).double()
    y = (index // 8 + .5).double()
    raw = torch.stack((x + y.square(), 2 * x.square() - y, x - 3 * y, x * y + 2), dim=-1)[None]
    scale = torch.tensor([2., 3., 4., 6.], dtype=torch.float64)
    offset = torch.tensor([10., -2., 1., 7.], dtype=torch.float64)
    physical = raw * scale + offset

    for name, family in families.items():
        dependency_ids = torch.tensor([d.native_id for d in family.extraction.dependencies])
        actual = extract_family_predictions(family, raw[0, dependency_ids], output_scale=scale, output_offset=offset)
        coords = torch.tensor(family.coordinates[None], dtype=torch.float32)
        if name.startswith('flow/'):
            stencil = build_native_curl_read_stencil(coords, lengths, solid, nx=8, ny=6)
            expected_fields = physical_uvp_native_curl(raw[:, stencil.node_indices[0], :3], stencil, offset, scale)
            expected = expected_fields[0, :, {'flow/u': 0, 'flow/v': 1, 'flow/p': 2, 'flow/omega': 3}[name]]
        else:
            stencil = native_cell_center_stencil(coords, lengths, nx=8, ny=6)
            expected = interpolate_native(physical[:, :, 3:4], stencil)[0, :, 0]
            if name == 'q_proxy':
                slot = torch.tensor([int(row.split(':')[1]) for row in family.row_ids])
                port = torch.tensor([int(row.split(':')[3]) for row in family.row_ids])
                angles = torch.arange(64, dtype=torch.float32) * (2 * torch.pi / 64)
                normals = torch.stack((angles.cos(), angles.sin()), dim=-1)[port]
                delta = min(1., .15 * .8)
                outside = centers[0, slot] + (.8 + delta) * normals
                outside_stencil = native_cell_center_stencil(outside[None], lengths, nx=8, ny=6)
                outside_value = interpolate_native(physical[:, :, 3:4], outside_stencil)[0, :, 0]
                conductivity = torch.tensor(2 * 2. * 1. / (2. + 1.), dtype=torch.float32)
                expected = (expected - outside_value) * (conductivity / torch.tensor(delta))
        torch.testing.assert_close(actual, expected, rtol=2.e-6, atol=2.e-5)


def test_missing_family_and_restrictions_are_explicit(binding):
    with h5py.File(binding.provider.data_path, 'a') as data:
        data['cases']['second_scene']['interface_target'][:, :, 1] = np.nan
    scene = binding.finite_observation_scene('second_scene', seed=4)
    assert 'q_proxy' not in _family_map(scene)
    assert 'q_proxy' in binding.finite_observation_family_weights()
    for bad_count in (0, 65):
        with pytest.raises(ValueError, match='1–64'):
            binding.finite_observation_scene('first_scene', count=bad_count)
    with pytest.raises(ValueError, match='declared TRAIN or validation'):
        binding.finite_observation_scene('unlisted_scene')


def test_complete_source_support_is_a_separate_review_only_request(binding):
    probes = binding.finite_observation_scene('first_scene')
    complete = binding.complete_source_observation_scene_for_evaluation('first_scene')
    assert len(_family_map(probes)['flow/u'].row_ids) == 64
    assert len(_family_map(complete)['flow/u'].row_ids) == 95
    with pytest.raises(ValueError, match='1–64'):
        binding.finite_observation_scene('first_scene', count=95)


def test_review_observer_reduces_full_existing_fields_without_new_prediction(binding):
    from finite_observation_evaluate import ThermalFiniteObservationObserver, verify_saved_thermal_observation_identity

    cases = ('first_scene', 'second_scene')
    model = binding.provider.model
    model.field_mean = torch.zeros(5)
    saved = {'experiment_identity': {'provider_identity': {
        'finite_observations': {'family_weights': binding.finite_observation_family_weights()},
        'adapter_state': {'normalization_stats': {'field_mean_by_channel': model.field_mean.tolist(),
            'field_std_by_channel': model.field_std.tolist(),
            'internal_temperature_std': model.internal_temperature_std.tolist(),
            'interface_targets_std': model.interface_target_std.tolist()}}}}}
    identity = verify_saved_thermal_observation_identity(binding, saved)
    observer = ThermalFiniteObservationObserver(binding, model, cases, (cases,),
                                                expected_family_scales=identity['family_scales'])
    cells = torch.arange(48)
    coords = torch.stack(((cells % 8 + .5), (cells // 8 + .5)), dim=-1)
    x, y = coords.unbind(-1)
    field = torch.stack((x, y, x + y, x - y, 2*x + y), dim=-1)[None]
    prepared = SimpleNamespace(stencils={'fluid': SimpleNamespace(valid=torch.ones((1, 48), dtype=torch.bool))},
                               lengths=torch.tensor([[8., 6.]]))
    for index, case in enumerate(cases):
        prediction = SimpleNamespace(native_main={'pred_field': field + index}, native_prepared=prepared)
        batch = SimpleNamespace(receivers=SimpleNamespace(fluid_xy=coords[None]),
                                targets=SimpleNamespace(case_ids=(case,)))
        observer(case, prediction, batch)
    result = observer.finish()
    first = result['cases']['first_scene']
    assert first['fixed']['families']['flow/u']['row_count'] == 64
    assert first['complete_source']['families']['flow/u']['row_count'] == 95
    assert 'source_row_ids' not in first['complete_source']['families']['flow/u']
    assert first['fixed']['families']['temperature/material']['local_member_count'] == 2
    assert first['complete_source']['families']['q_proxy']['available'] is True
    assert result['pair_results']['fixed'][0]['pair'] == list(cases)
    assert result['pair_results']['fixed'][0]['families']['temperature/material']['available'] is True
    assert first['fixed']['families']['flow/u']['reference_source'].endswith('/sampled_points')


def test_magnitudes_do_not_change_selected_rows_or_historical_fluid_population(binding):
    with h5py.File(binding.provider.data_path, 'r') as data:
        group = data['cases']['first_scene']
        rows = group['sampled_points'][...]
        point_weights = group['sampled_point_weights'][...]
    binding.provider.training_cases = ({'case_id': 'first_scene', 'query_xy': rows[:, :2].astype(np.float32),
        'field_targets': rows[:, 2:].astype(np.float32), 'point_weights': point_weights.astype(np.float32),
        'structure': {'domain_length_x': np.array([8.]), 'domain_length_y': np.array([6.])},
        'module_mask': np.zeros((6, 8), dtype=bool)},)
    binding.provider.validation_cases = ()
    before_population = binding.coupled_native_population('first_scene')
    before = binding.finite_observation_scene('first_scene', seed=8)
    with h5py.File(binding.provider.data_path, 'a') as data:
        points = data['cases']['first_scene']['sampled_points']
        changed = points[...]
        changed[:, 2:] *= 5
        points[...] = changed
    after = binding.finite_observation_scene('first_scene', seed=8)
    after_population = binding.coupled_native_population('first_scene')
    assert [family.row_ids for family in before.families] == [family.row_ids for family in after.families]
    assert before_population.keys == after_population.keys
    assert before_population.dependencies == after_population.dependencies
    np.testing.assert_array_equal(before_population.references, after_population.references)
    np.testing.assert_array_equal(before_population.measures, after_population.measures)
    np.testing.assert_array_equal(before_population.valid, after_population.valid)


@pytest.mark.parametrize('read_mode', ['full', 'restricted', 'paired'])
def test_batched_execution_reuses_native_organization_and_matches_separate_gradients(binding, read_mode):
    from channelthermal.member_routing import ThermalDetailAnchoredMemberRoutingAdapter
    from honf_forward_core.interaction_reused_member_routing import DetailAnchoredMemberRoutingCore
    from tests.anchored_fixtures import _base_config

    torch.set_num_threads(1)
    core = DetailAnchoredMemberRoutingCore(_base_config(), input_dataset='thermal', query_tile_size=32,
                                           retained_mass=.7)
    stats = {'field_mean_by_channel': [1., -2., .5, 3., 10.],
             'field_std_by_channel': [2., 3., 4., 5., 6.],
             'interface_targets_std': [1., 1.], 'internal_temperature_std': [1.]}
    model = ThermalDetailAnchoredMemberRoutingAdapter(stats, core=core, nx=8, ny=6,
                                                      environment_nx=2, environment_ny=2).train()
    binding.provider.model = model
    cases = ('first_scene', 'second_scene')
    structure = {
        'module_centers': torch.tensor([[[2.5, 2.5], [5.5, 3.5], [3.5, 3.5]],
                                         [[2.5, 2.5], [5.5, 3.5], [3.5, 3.5]]]),
        'module_present': torch.tensor([[1., 1., 0.], [0., 0., 1.]]),
        'material_params': torch.tensor([[.02, .01, .02, 2., 1., .8], [.02, .01, .02, 2., 1., .8]]),
        're': torch.tensor([[50.], [50.]]), 'u_in': torch.ones(2, 1),
        'domain_length_x': torch.full((2, 1), 8.), 'domain_length_y': torch.full((2, 1), 6.),
    }
    heat = torch.tensor([[1., 2., 0.], [0., 0., 3.]])
    native_batch = SimpleNamespace(scene_inputs=SimpleNamespace(structure=structure, case_ids=cases),
                                   receivers=SimpleNamespace(heat=heat), targets=SimpleNamespace(case_ids=cases))
    fluid = torch.tensor([[[1.5, 1.5], [3.5, 2.5]], [[1.5, 1.5], [4.5, 3.5]]])
    prepared = model.prepare_native(structure, fluid, physical_heat=heat, ntheta=64, scene_key=cases)
    native_predictions = SimpleNamespace(native_prepared=prepared)
    binding.provider.case_split_by_id = {case: 'train' for case in cases}
    binding.provider._load_native_cases = lambda ids: list(ids)

    def one_case_batch(selected, indices, *, split):
        del split
        chosen = [cases.index(selected[index]) for index in indices]
        selected_cases = tuple(selected[index] for index in indices)
        scene_structure = {name: value[chosen] for name, value in structure.items()}
        return SimpleNamespace(scene_inputs=SimpleNamespace(structure=scene_structure, case_ids=selected_cases),
                               receivers=SimpleNamespace(heat=heat[chosen]),
                               targets=SimpleNamespace(case_ids=selected_cases))

    binding.provider._native_case_batch = one_case_batch
    selected = {case: _family_map(binding.finite_observation_scene(case, count=8, seed=7))['flow/omega']
                for case in cases}
    dependencies = {cases[0]: selected[cases[0]].extraction.dependencies[:8],
                    cases[1]: selected[cases[1]].extraction.dependencies[:3]}
    decoder = 'decode_paired' if read_mode == 'paired' else 'decode'
    with patch.object(model.core, 'prepare', wraps=model.core.prepare) as prepare, \
            patch.object(model.core, decoder, wraps=getattr(model.core, decoder)) as decode:
        batched, work = binding.execute_finite_observation_dependencies(
            model, cases, dependencies, read_mode=read_mode,
            native_batch=native_batch, native_predictions=native_predictions)
        assert prepare.call_count == 0
        assert decode.call_count == 1
        query = decode.call_args.args[1]
        assert query.query_mask.tolist() == [[True] * 8, [True] * 3 + [False] * 5]
        assert torch.all(query.query_ids[~query.query_mask] == 0)
        assert torch.all(query.anchor_indices >= 0)
        assert torch.all(query.anchor_weights[~query.query_mask] == 0)
    assert work['reused_scene_preparations'] == 1
    assert work['scene_preparations'] == 0
    assert work['neural_queries'] == 11
    assert work['padding_queries'] == 5
    assert work['padded_queries'] == 16
    assert any('member' in name or 'pairs' in name for name in work if name.startswith(('single_', 'paired_')))
    with patch.object(model.core, 'prepare', wraps=model.core.prepare) as prepare:
        fallback, fallback_work = binding.execute_finite_observation_dependencies(
            model, cases, dependencies, read_mode=read_mode)
        assert prepare.call_count == 1
    assert fallback_work['scene_preparations'] == 1
    assert fallback_work['reused_scene_preparations'] == 0
    for case in cases:
        torch.testing.assert_close(fallback[case], batched[case], rtol=3.e-5, atol=2.e-6)
    expected = {}
    for case in cases:
        scene, query, scale, offset = binding.field_request(model, case, dependencies[case])
        organization = model.core.prepare(scene)
        if read_mode == 'paired':
            full, restricted, _ = model.core.decode_paired(organization, query)
            raw = torch.stack((full.joint_fields[0], restricted.joint_fields[0]))
        else:
            output = model.core.decode(organization, query, intervention={'read_mode': read_mode})
            raw = output.joint_fields
        expected[case] = raw * scale + offset
        torch.testing.assert_close(batched[case], expected[case], rtol=3.e-5, atol=2.e-6)
    parameters = tuple(model.parameters())
    actual_loss = sum(value.square().sum() for value in batched.values())
    expected_loss = sum(value.square().sum() for value in expected.values())
    actual_gradients = torch.autograd.grad(actual_loss, parameters, allow_unused=True)
    expected_gradients = torch.autograd.grad(expected_loss, parameters, allow_unused=True)
    for actual, reference in zip(actual_gradients, expected_gradients):
        if actual is None or reference is None:
            assert actual is reference
        else:
            torch.testing.assert_close(actual, reference, rtol=5.e-4, atol=1.e-4)

    reordered = SimpleNamespace(scene_inputs=SimpleNamespace(structure=structure, case_ids=tuple(reversed(cases))),
                                receivers=native_batch.receivers,
                                targets=SimpleNamespace(case_ids=tuple(reversed(cases))))
    with pytest.raises(ValueError, match='ordered batch'):
        binding.execute_finite_observation_dependencies(model, cases, dependencies, read_mode=read_mode,
                                                         native_batch=reordered, native_predictions=native_predictions)
