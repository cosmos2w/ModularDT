import pytest
import torch
from types import SimpleNamespace

from channelthermal.training.thermal_port_response_task import (
    ThermalPortResponseTask, _anchored_port_residual, _module_mean,
)
from channelthermal.training.thermal_response_feedback_task import ThermalResponseFeedbackTask


def test_anchored_loss_uses_local_input_gradient_and_stops_teacher():
    outside = torch.tensor([2.0], requires_grad=True)
    direct = torch.tensor([3.0], requires_grad=True)
    teacher_outside = torch.tensor([2.0], requires_grad=True)
    reference = torch.tensor([3.0])
    local = 4 * outside + 9
    teacher = 4 * teacher_outside + 9  # biased local operator
    residual = _anchored_port_residual(direct, reference, local, teacher, 2.0)
    assert residual.item() == 0
    residual.backward()
    assert direct.grad.item() == 0.5
    assert outside.grad.item() == -2
    assert teacher_outside.grad is None


def test_module_mean_equal_modules_and_rejects_nonfinite_valid_values():
    errors = torch.tensor([[[1., 1.], [3., 99.]]])
    mask = torch.tensor([[[True, True], [True, False]]])
    assert torch.equal(_module_mean(errors, mask), torch.tensor([5.]))
    errors[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        _module_mean(errors, mask)


def test_epoch_context_preserves_source_age(monkeypatch):
    def inherited(self, model, engine, epoch, output, row):
        row['reader_epoch_context'] = {'absolute_training_epoch': epoch,
                                       'additional_epochs_after_parent': 0}
    monkeypatch.setattr(ThermalResponseFeedbackTask, 'completed_training_epoch', inherited)
    provider = object.__new__(ThermalPortResponseTask)
    provider.source_parent_age = 2000
    provider.port_arm = 'C0'
    row = {}
    provider.completed_training_epoch(None, None, 20, None, row)
    assert row['reader_epoch_context'] == {'absolute_training_epoch': 2020,
                                           'additional_epochs_after_parent': 20}

    class Core:
        _joint_update_index = 3

        def set_training_clock(self, epoch, update_index):
            self.clock = (epoch, update_index)

    class Model:
        architecture = 'R14-Detail-LocalECover'
        core = Core()

        def prepare_native(self, *args, **kwargs):
            raise RuntimeError('stopped after clock update')

    provider.model = Model()
    scene = SimpleNamespace(structure={}, case_ids=('0002',))
    receivers = SimpleNamespace(fluid_xy=None, heat=None, local_query_points=None)
    with pytest.raises(RuntimeError, match='stopped after clock update'):
        provider.predict_native(provider.model, scene, receivers, provider.model.architecture,
                                'joint', epoch=20)
    assert provider.model.core.clock == (2020, 3)
    assert row['reader_epoch_context']['absolute_training_epoch'] == provider.model.core.clock[0]
