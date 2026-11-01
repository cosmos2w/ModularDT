"""Focused contracts for the Thermal response record and explicit transfer."""
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from channelthermal.thermal_coupling import response_scene, transplant_widened_model


@dataclass
class Scene:
    module_features: torch.Tensor
    module_mask: torch.Tensor


class Encoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.module_encoder = nn.Sequential(nn.Linear(width + 16, 8), nn.SiLU())
        self.other = nn.Linear(8, 3)


def test_transfer_preserves_original_geometry_and_zero_feedback_columns():
    torch.manual_seed(13)
    source, widened = Encoder(9), Encoder(138)
    assert transplant_widened_model(widened, source.state_dict()) == 'module_encoder.0.weight'
    features, geometry = torch.randn(4, 9), torch.randn(4, 16)
    response = torch.randn(4, 129, requires_grad=True)
    original = source.module_encoder(torch.cat((features, geometry), -1))
    actual = widened.module_encoder(torch.cat((features, response, geometry), -1))
    torch.testing.assert_close(actual, original)
    gradient, = torch.autograd.grad(actual.sum(), response, retain_graph=True)
    assert torch.count_nonzero(gradient) == 0
    actual.sum().backward()
    assert widened.module_encoder[0].weight.grad[:, 9:138].abs().sum() > 0
    copy = Encoder(138)
    copy.load_state_dict(widened.state_dict(), strict=True)
    torch.testing.assert_close(copy.module_encoder(torch.cat((features, response, geometry), -1)), actual)


def test_angular_order_mask_permutation_and_nonfinite_failure():
    scene = Scene(torch.randn(1, 3, 9), torch.tensor([[True, False, True]]))
    response = torch.randn(1, 3, 64, 2)
    out = response_scene(scene, response, [2., 3.], [4., 5.])
    torch.testing.assert_close(out.module_features[0, 0, 9:73], (response[0, 0, :, 0] - 2) / 4)
    torch.testing.assert_close(out.module_features[0, 0, 73:137], (response[0, 0, :, 1] - 3) / 5)
    assert torch.count_nonzero(out.module_features[0, 1, 9:]) == 0
    perm = torch.tensor([2, 0, 1])
    permuted = response_scene(Scene(scene.module_features[:, perm], scene.module_mask[:, perm]), response[:, perm], [2., 3.], [4., 5.])
    torch.testing.assert_close(permuted.module_features, out.module_features[:, perm])
    response[0, 0, 4, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        response_scene(scene, response, [2., 3.], [4., 5.])


def test_same_mean_angular_responses_can_have_distinct_encodings():
    scene = Scene(torch.zeros(1, 1, 9), torch.ones(1, 1, dtype=torch.bool))
    first = torch.zeros(1, 1, 64, 2)
    first[0, 0, 0, 0], first[0, 0, 1, 0] = 1, -1
    second = first.roll(1, dims=2)
    assert first.mean() == second.mean()
    x = response_scene(scene, first, [0., 0.], [1., 1.]).module_features
    y = response_scene(scene, second, [0., 0.], [1., 1.]).module_features
    encoder = nn.Linear(138, 1, bias=False)
    with torch.no_grad():
        encoder.weight.zero_()
        encoder.weight[0, 9] = 1
    assert not torch.equal(encoder(x), encoder(y))


def test_native_16_view_selects_each_modules_aligned_64_stencil():
    from channelthermal.native_observations import NativeStencil
    from channelthermal.thermal_coupling import native_angle_view
    from dataclasses import make_dataclass

    Prepared = make_dataclass('Prepared', ['source_present', 'stencils', 'role_shapes', 'ntheta'])
    values = torch.arange(2 * 3 * 64).reshape(2, 3 * 64)
    stencil = NativeStencil(values[..., None].expand(-1, -1, 4), values[..., None].expand(-1, -1, 4).float(), values.bool())
    original = Prepared(torch.ones(2, 3, dtype=torch.bool), {'surface': stencil, 'outside': stencil},
                        {'surface': (3, 64), 'outside': (3, 64)}, 64)
    view = native_angle_view(original, 16)
    expected = values.reshape(2, 3, 64)[:, :, ::4].reshape(2, 48)
    assert torch.equal(view.stencils['outside'].indices[..., 0], expected)
    assert torch.equal(original.stencils['outside'].indices[..., 0], values)
    assert view.role_shapes['surface'] == (3, 16)


def test_native_nonfinite_predictions_fail_on_reference_masks_with_identity():
    from types import SimpleNamespace as S
    from channelthermal.training.thermal_response_feedback_task import ThermalResponseFeedbackTask
    valid = lambda shape: S(valid=torch.ones(shape, dtype=torch.bool))
    prepared = S(source_present=torch.tensor([[True]]), scene_batch=S(module_ids=torch.tensor([[42]])),
                 stencils={'fluid': valid((1, 2)), 'surface': valid((1, 1, 2)),
                           'outside': valid((1, 1, 2)), 'material': valid((1, 1, 2))})
    targets = S(case_ids=('case-7',), point_weights=torch.ones(1, 2), field_targets=torch.ones(1, 2, 5),
                interface_target=torch.ones(1, 1, 2, 2), interface_valid_mask=torch.ones(1, 1, 2),
                material_targets=torch.ones(1, 1, 2))
    output = {'pred_field': torch.ones(1, 2, 5), 'pred_interface': torch.ones(1, 1, 2, 2),
              'pred_internal_temperature': torch.ones(1, 1, 2, 1)}
    pred = S(native_prepared=prepared, native_main=output)
    ThermalResponseFeedbackTask._check_native_finite(pred, targets, label='final')
    output['pred_internal_temperature'][0, 0, 1, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='case case-7, module 42'):
        ThermalResponseFeedbackTask._check_native_finite(pred, targets, label='final')
    output['pred_internal_temperature'][0, 0, 1, 0] = 1
    output['pred_field'][0, 1, 0] = float('inf')
    with pytest.raises(FloatingPointError, match='final u.*case case-7'):
        ThermalResponseFeedbackTask._check_native_finite(pred, targets, label='final')


def test_paired_control_reuses_native_and_local_work(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace as S
    from channelthermal.training import thermal_response_feedback_task as task

    class Core:
        mode = None
        @contextmanager
        def read_execution(self, mode):
            self.mode = mode
            yield
        def __init__(self):
            self.reads = []
    class Model:
        architecture = 'arm'
        training = True
        def __init__(self):
            self.core = Core()
        def prepare_native(self, *args, **kwargs):
            return S(organization=S(work_counts={'pairs': 11}))
        def apply_native_with_work_counts(self, prepared, heat):
            self.core.reads.append(self.core.mode)
            return {'path': self.core.mode}, {'pairs': 7}, self.core.mode
    model = Model()
    provider = object.__new__(task.ThermalResponseFeedbackTask)
    provider.model = model
    provider.feedback = False
    provider.local_operator = object()
    calls = []
    monkeypatch.setattr(task, 'native_angle_view', lambda prepared, n: prepared)
    monkeypatch.setattr(task, 'outside_from_native_output', lambda model, prepared, raw: raw)
    def local(model, operator, prepared, heat, points, outside):
        calls.append(outside)
        return {'interface_pred': outside}
    monkeypatch.setattr(task, 'local_response_from_outside', local)
    scene = S(structure={}, case_ids=('a',))
    receivers = S(fluid_xy=None, heat=None, local_query_points=None, ntheta=64)
    pred, _ = provider.predict_native(model, scene, receivers, 'arm', 'joint')
    assert model.core.reads == calls == ['full', 'restricted']
    assert pred.full is pred.initial.full and pred.restricted is pred.initial.restricted
    assert pred.local0 == pred.local1
    assert pred.work['preparations'] == 1 and pred.work['local_calls'] == 2
    assert pred.work['port_geometry_builds'] == pred.work['port_decodes'] == 0
    assert pred.work['stage1_full_reused_stage0'] == 1
    assert not any(k.startswith('stage1_full_decode_') for k in pred.work)


def test_feedback_paired_paths_prepare_from_their_own_local_responses(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace as S
    from channelthermal.training import thermal_response_feedback_task as task

    class Core:
        mode = None
        @contextmanager
        def read_execution(self, mode):
            self.mode = mode
            yield
    class Model:
        architecture = 'arm'
        training = True
        def __init__(self):
            self.core = Core()
            self.prepares = 0
        def prepare_native(self, *args, **kwargs):
            self.prepares += 1
            return S(label='stage0', organization=S(work_counts={'pairs': 1}))
        def apply_native_with_work_counts(self, prepared, heat):
            return {'source': (prepared.label, self.core.mode)}, {'rows': 1}, self.core.mode
    model = Model()
    provider = object.__new__(task.ThermalResponseFeedbackTask)
    provider.model, provider.feedback, provider.local_operator = model, True, object()
    provider.response_mean, provider.response_scale = [0., 0.], [1., 1.]
    monkeypatch.setattr(task, 'native_angle_view', lambda prepared, n: prepared)
    monkeypatch.setattr(task, 'outside_from_native_output', lambda model, prepared, raw: raw)
    builds = []
    def prepare(model, first, response, mean, scale):
        builds.append((first.label, response))
        return S(label='stage1-' + response, organization=S(work_counts={'pairs': 1}))
    def local(model, operator, prepared, heat, points, mode):
        if prepared.label != 'stage0':
            assert prepared.label == 'stage1-' + mode
        return {'interface_pred': mode if prepared.label == 'stage0' else prepared.label}
    monkeypatch.setattr(task, 'prepare_response', prepare)
    monkeypatch.setattr(task, 'local_response_from_outside', local)
    scene = S(structure={}, case_ids=('a',))
    receivers = S(fluid_xy=None, heat=None, local_query_points=None, ntheta=64)
    pred, _ = provider.predict_native(model, scene, receivers, 'arm', 'joint')
    assert model.prepares == 1 and builds == [('stage0', 'full'), ('stage0', 'restricted')]
    assert pred.initial.full.native_main['source'] == ('stage0', 'full')
    assert pred.initial.restricted.native_main['source'] == ('stage0', 'restricted')
    assert pred.full.native_main['source'] == ('stage1-full', 'full')
    assert pred.restricted.native_main['source'] == ('stage1-restricted', 'restricted')
    assert pred.local1['full']['interface_pred'] == 'stage1-full'
    assert pred.local1['restricted']['interface_pred'] == 'stage1-restricted'
    assert pred.work['preparations'] == 3 and pred.work['port_geometry_builds'] == 0
    assert pred.work['local_calls'] == 4 and pred.work['port_decodes'] == 0


def test_feedback_inference_defers_initial_native_read_until_validation_loss(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace as S
    from channelthermal.training import thermal_response_feedback_task as task

    class Core:
        @contextmanager
        def read_execution(self, mode):
            yield
    class Model:
        architecture = 'arm'
        training = False
        core = Core()
        def __init__(self):
            self.reads = []
        def prepare_native(self, *args, **kwargs):
            return S(label='initial', organization=S(work_counts={}))
        def apply_native_with_work_counts(self, prepared, heat):
            self.reads.append(prepared.label)
            return {'source': prepared.label}, {'rows': 1}, prepared.label
    model = Model()
    provider = object.__new__(task.ThermalResponseFeedbackTask)
    provider.model, provider.feedback, provider.local_operator = model, True, object()
    provider.response_mean, provider.response_scale = [0., 0.], [1., 1.]
    monkeypatch.setattr(task, 'native_angle_view', lambda prepared, n: prepared)
    monkeypatch.setattr(task, 'outside_geometry', lambda model, prepared: object())
    monkeypatch.setattr(task, 'local_response', lambda *args: ({'interface_pred': 'response0'}, {'rows': 2}))
    monkeypatch.setattr(task, 'prepare_response', lambda *args: S(label='final', organization=S(work_counts={})))
    monkeypatch.setattr(task, 'outside_from_native_output', lambda model, prepared, raw: raw)
    monkeypatch.setattr(task, 'local_response_from_outside', lambda *args: {'interface_pred': 'response1'})
    monkeypatch.setattr(provider, '_terms', lambda prediction, targets: prediction.initial.native_main)
    scene = S(structure={}, case_ids=('a',))
    receivers = S(fluid_xy=None, heat=None, local_query_points=None, ntheta=16)
    prediction, aux = provider.predict_native(model, scene, receivers, 'arm', 'joint')
    assert model.reads == ['final'] and prediction.initial.native_main is None
    assert provider.validation_loss_terms(prediction, None, aux) == {'source': 'initial'}
    assert model.reads == ['final', 'initial']
    provider.validation_loss_terms(prediction, None, aux)
    assert model.reads == ['final', 'initial']
