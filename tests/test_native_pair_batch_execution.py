"""Independent-scene batching preserves native pair supervision and gradients."""
from __future__ import annotations

import copy
import json
from dataclasses import fields, replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from channelthermal.training import thermal_native_pair_task as native
from channelthermal.training.content_hypergraph_task import ThermalContentReceivers, ThermalContentSceneInputs
from channelthermal.training.joint_member_routing_task import ThermalJointPairPredictions
from channelthermal.training.thermal_native_port_task import NativePortTargets, ThermalNativePortTask
from honf_runtime.state_response import native_functional_contrast
from honf_runtime.unified_training import TaskBatch


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = torch.nn.Parameter(torch.tensor([0.8, -0.3], dtype=torch.double))
        self.register_buffer('field_std', torch.tensor([1., 1.2, 1.4, 1.6, 1.8], dtype=torch.double))
        self.register_buffer('interface_target_std', torch.tensor([1.1, 1.3], dtype=torch.double))
        self.register_buffer('internal_temperature_std', torch.tensor([1.5], dtype=torch.double))
        self.architecture = 'fixture_joint'


class _Provider(ThermalNativePortTask):
    """Small physical observations with case-specific masks and labels."""
    def __init__(self, model, path):
        self.model, self.data_path = model, path
        self.native_weights = {name: 0.5 + 0.1 * i for i, name in enumerate(native.NATIVE_LOSS_ORDER)}
        self.outside_temperature_weight = 0.05
        self.case_split_by_id = {'scene_first': 'train', 'scene_second': 'train'}
        self.calls = []

    def _load_native_cases(self, pair):
        return [{'case_id': case_id} for case_id in pair]

    def _native_case_batch(self, cases, indices, *, split):
        assert split == 'train'
        index = indices[0]
        case_id = cases[index]['case_id']
        factor = 1. if case_id == 'scene_first' else 1.7
        tensor = lambda value: torch.tensor(value, dtype=torch.double)
        coordinates = tensor([[[.02, .5], [.3, .5], [.6, .5], [.98, .5]]])
        point_weights = tensor([[1., 1., 1., 1.] if index == 0 else [1., 1., 0., 1.]])
        field_targets = (torch.arange(20, dtype=torch.double).reshape(1, 4, 5) / 10) * factor
        interface = (torch.arange(8, dtype=torch.double).reshape(1, 2, 2, 2) / 9) * factor
        material = tensor([[[.3, .7], [1.1, 1.5]]]) * factor
        if index == 1:
            material[0, 1, 1] = float('nan')
        interface_mask = torch.tensor([[[True, True], [True, index == 0]]])
        outside_mask = torch.tensor([[[True, index == 0], [True, True]]])
        structure = {'module_present': tensor([[1., 1.]]), 'control': tensor([[factor]]),
                     'fluid_stencil': torch.ones(1, 4, dtype=torch.bool),
                     'material_stencil': torch.ones(1, 4, dtype=torch.bool),
                     'surface_stencil': torch.ones(1, 4, dtype=torch.bool),
                     'outside_stencil': torch.ones(1, 4, dtype=torch.bool)}
        targets = NativePortTargets(case_ids=(case_id,), case_indices=(index,), epoch=0,
            field_targets=field_targets, point_weights=point_weights, interface_target=interface,
            interface_valid_mask=interface_mask, material_targets=material,
            outside_temperature=interface[..., 0] + .2, outside_valid=outside_mask)
        return TaskBatch(scene_inputs=ThermalContentSceneInputs(structure, case_ids=(case_id,)),
            receivers=ThermalContentReceivers(coordinates, tensor([[[.1, .2], [.2, .3]]]),
                                               tensor([[.4 * factor, .7 * factor]]), ntheta=2),
            targets=targets, auxiliary={'evaluation_split': split}, case_keys=(case_id,))

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0):
        assert model is self.model and execution_mode == model.architecture and phase == 'joint'
        self.calls.append(scene.case_ids)
        structure = scene.structure
        stencils = {name: SimpleNamespace(valid=structure[name + '_stencil'])
                    for name in ('fluid', 'material', 'surface', 'outside')}
        prepared = SimpleNamespace(stencils=stencils, source_present=structure['module_present'])
        x = receivers.fluid_xy[..., 0]
        first, second = model.coefficients
        base = first * x + second * structure['control']
        heat = receivers.heat

        def branch(multiplier):
            field = torch.stack([multiplier * (base * (i + 1) + .04 * i) for i in range(5)], -1)
            material = multiplier * (first * heat[..., None] + second * heat[..., None].square())
            material = material.expand(-1, -1, 2)[..., None]
            surface = multiplier * (first * heat[..., None] + second * torch.tensor([.2, .7]))
            interface = torch.stack((surface, surface * .6 + .1), -1)
            return SimpleNamespace(native_prepared=prepared, native_main={'pred_field': field,
                'pred_internal_temperature': material, 'pred_interface': interface,
                'pred_outside_temperature': surface + .15})

        full = branch(1.)
        prediction = (ThermalJointPairPredictions(full, branch(.85), {}) if model.training else full)
        return prediction, {'work': {'fixture_queries': int(x.numel())}}


@pytest.fixture
def pair_objective(tmp_path, monkeypatch):
    path = tmp_path / 'native.h5'
    pair = ('scene_first', 'scene_second')
    with h5py.File(path, 'w') as h5:
        cases = h5.create_group('cases')
        for case_id in pair:
            cases.create_group(case_id).create_dataset('case_config_json',
                data=json.dumps({'domain': {'lx': 1.}}).encode())
    monkeypatch.setattr(native, 'native_pair_support', lambda *args: {
        'A': np.array([[False, True, False, False]]),
        'B': np.array([[False, False, False, True]]),
    })
    monkeypatch.setattr(native, 'native_pair_references', lambda *args: torch.tensor(
        [[[1., 2., 3., 4.], [1.3, 1.6, 3.5, 4.4]]], dtype=torch.double))
    model = _Model()
    provider = _Provider(model, str(path))
    objective = native.ThermalNativePairObjective(provider, [pair])
    return model, provider, objective, pair


def test_pair_batching_preserves_per_scene_losses_observations_and_gradients(pair_objective):
    model, provider, objective, pair = pair_objective
    results = []
    for mode in ('serial', 'paired'):
        model.zero_grad(set_to_none=True)
        provider.calls.clear()
        objective.prediction_batching = mode
        observed, reference, absolute, work = objective.observe_pair(model, pair, epoch=50)
        contrast = native_functional_contrast(observed.unsqueeze(1), reference.unsqueeze(0), objective.scales)
        (absolute + .37 * contrast).backward()
        results.append((observed.detach(), reference.detach(), absolute.detach(), contrast.detach(),
                        model.coefficients.grad.clone(), tuple(provider.calls), work))
    serial, batched = results
    for expected, actual in zip(serial[:4], batched[:4]):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(batched[4], serial[4], rtol=1e-9, atol=1e-8)
    assert serial[5] == (('scene_first',), ('scene_second',))
    assert batched[5] == (('scene_first', 'scene_second'),)
    assert serial[6] == batched[6] == {'native_pair_fixture_queries': 8,
                                      'native_pair_presentations': 1, 'native_pair_scenes': 2}


def test_eval_single_prediction_and_runtime_mode_validation(pair_objective):
    model, provider, objective, pair = pair_objective
    model.eval()
    observed = []
    for mode in ('serial', 'paired'):
        provider.calls.clear()
        objective.prediction_batching = mode
        with torch.no_grad():
            value, reference, absolute, work = objective.observe_pair(model, pair, epoch=1)
        assert value.shape == (1, 2, 4)
        observed.append((value, reference, absolute, work, tuple(provider.calls)))
    for expected, actual in zip(observed[0][:3], observed[1][:3]):
        torch.testing.assert_close(actual, expected)
    assert observed[0][3] == observed[1][3]
    assert len(observed[0][4]) == 2 and len(observed[1][4]) == 1
    objective.prediction_batching = 'invalid'
    with pytest.raises(ValueError, match='prediction_batching'):
        objective.observe_pair(model, pair, epoch=1)


def test_recipe_defaults_and_invalid_modes_preserve_legacy_behavior(pair_objective):
    _, _, objective, _ = pair_objective
    assert objective.prediction_batching == 'serial'
    recipe = native.build_native_pair_recipe({'initialization': {}}, variant='state_anchors')
    assert recipe['thermal_native_pair']['prediction_batching'] == 'paired'
    for invalid in ('invalid', '', None, True):
        with pytest.raises(ValueError, match='prediction_batching'):
            native.build_thermal_native_pair_task({'thermal_native_pair': {
                'variant': 'state_anchors', 'prediction_batching': invalid}})


@pytest.mark.parametrize(('declared_mode', 'expected_mode'), [
    (None, 'serial'), ('serial', 'serial'), ('paired', 'paired'),
])
def test_task_builder_selects_mode_without_changing_legacy_recipes(monkeypatch, declared_mode, expected_mode):
    from channelthermal.training import thermal_joint_response_task

    provider = SimpleNamespace(recipe={'training': {}}, data_path='fixture',
                               training_case_ids=tuple(str(i) for i in range(150)))
    model = object()
    monkeypatch.setattr(thermal_joint_response_task, 'build_thermal_joint_response_task',
                        lambda *args, **kwargs: (model, provider))
    monkeypatch.setattr(native, 'select_thermal_native_pairs',
                        lambda *args: ((('00', '01'), ('02', '03'), ('04', '05')), ()))
    spec = {'variant': 'field_only'}
    if declared_mode is not None:
        spec['prediction_batching'] = declared_mode
    result_model, result_provider = native.build_thermal_native_pair_task({
        'thermal_native_pair': spec, 'training': {'optimizer_schedule_horizon': 2000}})
    assert result_model is model and result_provider is provider
    assert provider.native_pair_prediction_batching == expected_mode


@pytest.mark.parametrize('variant', ['state_anchors', 'functional_contrast'])
def test_fresh_formal_cli_can_reuse_existing_calibration(tmp_path, monkeypatch, variant):
    import tools.relation_anchored_train as cli
    from tests.test_relation_anchored_horizon import native_pair_formal_identities

    identity, _ = native_pair_formal_identities()
    recipe = identity['recipe']
    recipe['thermal_native_pair']['variant'] = variant
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    calibration = tmp_path / 'shared_calibration.json'
    provider = SimpleNamespace()
    model = object()
    monkeypatch.setattr(cli, 'build', lambda *args, **kwargs: (model, provider))
    monkeypatch.setattr(cli, 'bind_training_budget', lambda *args: None)
    monkeypatch.setattr(cli, 'prepare_profile_fit', lambda *args, **kwargs: None)
    calls = []
    def fit(engine, actual_model, actual_provider, output, **kwargs):
        assert actual_model is model and actual_provider is provider
        calls.append(kwargs)
        assert provider.native_pair_calibration_input == str(calibration.resolve())
        assert not hasattr(provider, 'native_pair_calibration_output')
        return {'completed_epoch': 100}
    monkeypatch.setattr(cli.TrainingEngine, 'fit', fit)
    args = ['fit', '--recipe', str(path), '--output', str(tmp_path / 'run'),
            '--stop-after', '100', '--native-pair-calibration-input', str(calibration)]
    assert cli.main(args) == 0
    assert len(calls) == 1 and calls[0]['resume_checkpoint'] is None
    with pytest.raises(SystemExit):
        cli.main(args + ['--native-pair-calibration-output', str(tmp_path / 'other.json')])


def test_checkpointed_core_keeps_ragged_scenes_independent():
    from tests.test_e_anchored_packed_execution import fixture

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, scene, query = fixture('local_e_cover')
        model.train()
        model.query_tile_size = 2
        serial = copy.deepcopy(model)
        mask = query.query_mask.clone()
        mask[1, 3] = False
        weights = query.anchor_weights.detach().clone()
        weights[1, 3] = 0
        query = replace(query, coordinates=query.coordinates.detach(), query_mask=mask, anchor_weights=weights)
        module_fields = {'module_features', 'module_coordinates', 'module_mask', 'module_measure',
                         'module_ids', 'local_radius', 'module_characteristic_lengths'}
        environment_fields = {'environment_features', 'environment_coordinates', 'environment_mask',
                              'environment_measure', 'environment_ids', 'environment_characteristic_lengths'}

        def scene_part(index):
            changes = {}
            for field in fields(scene):
                value = getattr(scene, field.name)
                if torch.is_tensor(value) and value.ndim and value.shape[0] == 2:
                    value = value[index:index + 1]
                    if index == 1 and field.name in module_fields:
                        value = value[:, :1]
                    elif index == 1 and field.name in environment_fields:
                        value = value[:, :2]
                    changes[field.name] = value
            return replace(scene, **changes)

        def query_part(index):
            changes = {}
            for field in fields(query):
                value = getattr(query, field.name)
                if torch.is_tensor(value) and value.ndim and value.shape[0] == 2:
                    value = value[index:index + 1]
                    if index == 1 and value.ndim >= 2 and value.shape[1] == 4:
                        value = value[:, :3]
                    changes[field.name] = value
            return replace(query, **changes)

        full, restricted, _ = model.decode_paired(model.prepare(scene), query)
        separate = [serial.decode_paired(serial.prepare(scene_part(i)), query_part(i)) for i in range(2)]
        for view, batched in enumerate((full, restricted)):
            for i, count in enumerate((4, 3)):
                torch.testing.assert_close(batched.joint_fields[i, :count], separate[i][view].joint_fields[0],
                                           rtol=1e-9, atol=1e-10)
        sum((value.joint_fields * mask[..., None]).square().sum() for value in (full, restricted)).backward()
        sum(value.joint_fields.square().sum() for pair in separate for value in pair[:2]).backward()
        for (name, value), (_, reference) in zip(model.named_parameters(), serial.named_parameters()):
            assert (value.grad is None) == (reference.grad is None), name
            if value.grad is not None:
                torch.testing.assert_close(value.grad, reference.grad, rtol=1e-8, atol=1e-9, msg=name)
    finally:
        torch.set_num_threads(old_threads)
