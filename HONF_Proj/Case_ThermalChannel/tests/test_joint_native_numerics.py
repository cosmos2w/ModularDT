"""Complete native joint temperature/flux arithmetic and heat derivatives."""
from __future__ import annotations

import torch
from test_thermal_joint_regional import _adapter, _prepare


def test_joint_native_precise_kernels_cover_flux_and_all_temperature_roles():
    model = _adapter('J-H').eval()
    _, _, _, prepared = _prepare(model, 2)
    heat = torch.tensor([[0.7, 1.1]], dtype=torch.float64, requires_grad=True)
    delta = torch.tensor([[0.125, -0.25]], dtype=torch.float64)
    output = model.apply_native(prepared, heat, accumulation_dtype=torch.float64)
    kernels = model.export_native_kernels(prepared, accumulation_dtype=torch.float64)
    # Independently contract each native output block, including the already
    # widened conductivity/stencil factors in the flux kernel.
    expected = {role: torch.einsum('b...mo,bm->b...o', kernel, heat) for role, kernel in kernels.items()}
    torch.testing.assert_close(output['fluid_temperature'], expected['fluid'], rtol=0, atol=0)
    torch.testing.assert_close(output['pred_interface'][..., :1], expected['surface'], rtol=0, atol=0)
    torch.testing.assert_close(output['pred_interface'][..., 1:], expected['q_normal'], rtol=0, atol=0)
    torch.testing.assert_close(output['pred_internal_temperature'], expected['material'], rtol=0, atol=0)

    def forward(control):
        native = model.apply_native(prepared, control, accumulation_dtype=torch.float64)
        return torch.cat((native['pred_field'].flatten(), native['pred_interface'].flatten(),
                          native['pred_internal_temperature'].flatten()))

    _, jvp = torch.autograd.functional.jvp(forward, heat, delta)
    increment = model.apply_native(prepared, delta, increment=True, accumulation_dtype=torch.float64)
    expected_increment = torch.cat((increment['pred_field'].flatten(), increment['pred_interface'].flatten(),
                                    increment['pred_internal_temperature'].flatten()))
    torch.testing.assert_close(jvp, expected_increment, rtol=1e-12, atol=1e-12)
    for epsilon in (0.1, 0.01):
        difference = (forward(heat + epsilon * delta) - forward(heat - epsilon * delta)) / (2 * epsilon)
        torch.testing.assert_close(difference, expected_increment, rtol=1e-10, atol=1e-11)
