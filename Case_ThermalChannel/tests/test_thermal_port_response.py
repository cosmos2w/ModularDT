import pytest
import torch

from channelthermal.training.thermal_port_response_task import _anchored_port_residual, _module_mean


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
