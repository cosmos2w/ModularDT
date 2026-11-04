from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from channelthermal.training.thermal_port_response_task import (
    ThermalPortResponseTask, _anchored_port_residual, _module_mean,
    load_thermal_port_response_checkpoint,
)
from channelthermal.training import thermal_port_response_task as port_task
from channelthermal.thermal_coupling import native_angle_view
from channelthermal.native_observations import NativeStencil


def test_anchored_loss_keeps_input_gradient_and_stops_teacher():
    outside = torch.tensor([2.0], requires_grad=True)
    direct = torch.tensor([3.0], requires_grad=True)
    teacher = torch.tensor([17.0], requires_grad=True)
    loss = _anchored_port_residual(direct, torch.tensor([3.0]), 4 * outside + 9, teacher, 2.0)
    loss.backward()
    assert direct.grad.item() == 0.5
    assert outside.grad.item() == -2
    assert teacher.grad is None
    assert torch.equal(_module_mean(torch.tensor([[[1., 1.], [3., 99.]]]),
                                    torch.tensor([[[True, True], [True, False]]])), torch.tensor([5.]))


def test_native_angle_view_selects_each_original_module():
    ids = torch.arange(128).reshape(1, 128, 1)
    stencil = NativeStencil(ids, ids.float(), ids[..., 0].bool())
    from dataclasses import dataclass
    @dataclass
    class Prepared:
        source_present: object
        stencils: object
        role_shapes: object
        ntheta: int
    prepared = Prepared(torch.ones(1, 2, dtype=torch.bool), {'surface': stencil, 'outside': stencil},
                        {'surface': (2, 64), 'outside': (2, 64)}, 64)
    result = native_angle_view(prepared, 16)
    assert result.stencils['outside'].indices.flatten().tolist() == list(range(0, 64, 4)) + list(range(64, 128, 4))
    assert result.role_shapes['outside'] == (2, 16)


def test_one_preparation_and_one_decode_per_mode(monkeypatch):
    calls = []
    @contextmanager
    def read_execution(mode):
        calls.append(('mode', mode))
        yield
    core = SimpleNamespace(read_execution=read_execution, set_training_clock=lambda e, i: calls.append(('clock', e, i)), _joint_update_index=0)
    prepared = SimpleNamespace(organization=SimpleNamespace(work_counts={}))
    class Model:
        architecture = 'detail'
        training = True
        def __init__(self): self.core = core
        def prepare_native(self, *args, **kwargs): calls.append(('prepare', kwargs['ntheta'])); return prepared
        def apply_native_with_work_counts(self, *args):
            calls.append(('decode',))
            return {}, {}, SimpleNamespace(joint_fields=torch.zeros(1, 1, 4))
        def _extract_native_roles(self, *args): return {}
    model = Model()
    provider = object.__new__(ThermalPortResponseTask)
    provider.model, provider.source_parent_age, provider.local_operator = model, 5000, object()
    monkeypatch.setattr(port_task, 'native_angle_view', lambda p, n: p)
    monkeypatch.setattr(port_task, 'outside_from_native_output', lambda *args: torch.zeros(1))
    monkeypatch.setattr(port_task, 'local_response_from_outside', lambda *args: {})
    scene = SimpleNamespace(structure={}, case_ids=('1',))
    receivers = SimpleNamespace(fluid_xy=None, heat=None, local_query_points=None, ntheta=16)
    prediction, _ = provider.predict_native(model, scene, receivers, 'detail', 'joint', epoch=100)
    assert calls.count(('prepare', 64)) == 1
    assert calls.count(('decode',)) == 2
    assert ('clock', 5100, 0) in calls
    assert prediction.work['local_calls'] == 2


def test_loader_rejects_inconsistent_identity_before_local_asset(monkeypatch):
    monkeypatch.setattr(port_task, 'load_trusted_checkpoint', lambda *args, **kwargs: {'experiment_identity': {}})
    with pytest.raises(ValueError, match='self-consistent'):
        load_thermal_port_response_checkpoint('unused')


def test_loader_restores_saved_clock_and_explicit_local_asset(monkeypatch):
    captured = {}
    model = SimpleNamespace(core=SimpleNamespace(set_training_clock=lambda epoch, update: captured.update(clock=(epoch, update))),
                            eval=lambda: None)
    class Local:
        def __init__(self, path): captured['local_path'] = path
        def to(self, device): return self
        def eval(self): return self
    adapter = {'schema': port_task.ThermalDetailAnchoredMemberRoutingAdapter.SCHEMA,
               'environment_grid_nx_ny': [24, 8]}
    recipe = {'thermal_port_response': {'schema': 'thermal_port_response_v1', 'enabled': True,
              'arm': 'P64', 'source_epoch': 5000, 'local_checkpoint': 'saved-asset'},
              'member_routing': {'arm': 'detail'}}
    payload = {'epoch': 100, 'arm': 'detail', 'model_state_dict': {}, 'experiment_identity': {
        'workflow': 'relation_anchored_case_training_v1', 'recipe': recipe,
        'provider_identity': {'adapter_state': adapter}}}
    monkeypatch.setattr(port_task, 'load_trusted_checkpoint', lambda *a, **kw: payload)
    monkeypatch.setattr(port_task.ThermalDetailAnchoredMemberRoutingAdapter, 'from_state_payload',
                        lambda *a, **kw: model)
    monkeypatch.setattr(port_task, 'FrozenThermalLocal', Local)
    _, _, loaded = load_thermal_port_response_checkpoint('p64', local_checkpoint='relocated-asset')
    assert loaded is payload
    assert captured == {'clock': (5100, 0), 'local_path': 'relocated-asset'}
    load_thermal_port_response_checkpoint('p64')
    assert captured['local_path'] == 'saved-asset'


def test_identity_does_not_include_feedback(monkeypatch):
    from channelthermal.training.r13_8_anchored_task import ThermalR138AnchoredMemberRoutingTask
    monkeypatch.setattr(ThermalR138AnchoredMemberRoutingTask, 'identity_payload', lambda self: {'base': 'identity'})
    provider = object.__new__(ThermalPortResponseTask)
    provider.recipe = {'thermal_port_response': {'schema': 'thermal_port_response_v1', 'arm': 'P64'}}
    identity = provider.identity_payload()
    assert identity['base'] == 'identity'
    assert identity['thermal_port_response']['preparations_per_design'] == 1
    assert identity['thermal_port_response']['local_asset_exposure'] == 'TRAIN600/DEV22/VAL89'
    assert 'thermal_feedback' not in identity
