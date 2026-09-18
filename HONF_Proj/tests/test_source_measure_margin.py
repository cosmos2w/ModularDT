"""Source-measure threshold diagnosis, including both sides of a support event."""
import pytest
import torch

from honf_forward_core.interface_fields.routing_index.sparse_projection import source_measure_sparsemax


@pytest.mark.parametrize('temperature,expected', [
    (1., [0.488888888888889, 0.322222222222222, 0.188888888888889]),
    (.5, [0.644444444444444, 0.311111111111111, 0.044444444444444]),
    (.25, [0.833333333333333, 0.166666666666667, 0.]),
    (.125, [1., 0., 0.]),
])
def test_uniform_measure_temperature_example(temperature, expected):
    logits = torch.tensor([[[.9, .4, 0.]]], dtype=torch.float64)
    mu = torch.full((1, 3), 1/3, dtype=torch.float64)
    result = source_measure_sparsemax(logits / temperature, mu)
    torch.testing.assert_close(result.probability.flatten(), torch.tensor(expected, dtype=torch.float64),
                               rtol=1e-12, atol=1e-14)


def test_all_active_margin_uses_actual_mass_and_ignores_unoccupied_hub():
    mu = torch.tensor([[.3, .70000004, 0]], dtype=torch.float64)
    u = torch.tensor([[[.3, -.2, -99.], [3., -.2, -99.]]], dtype=torch.float64)
    occupied = mu[:, None] > 0
    tau = ((mu[:, None]*u).sum(-1)-1)/mu.sum(-1)[:, None]
    margin = u.masked_fill(~occupied, torch.inf).amin(-1)-tau
    p = source_measure_sparsemax(u, mu)
    torch.testing.assert_close(margin > 0, (p.support == occupied).all(-1))


@pytest.mark.parametrize('offset', [-1e-3, 1e-3])
def test_projection_gradient_step_sweep_on_both_sides_of_support_transition(offset):
    # At z0-z1=2 and mu=(1/2,1/2), the second coordinate leaves support.
    z = torch.tensor([[[2.+offset, 0.]]], dtype=torch.float64, requires_grad=True)
    mu = torch.tensor([[.5, .5]], dtype=torch.float64)
    def objective(value):
        p = source_measure_sparsemax(value, mu).probability
        return (p * p.new_tensor([.3, 1.7])).sum()
    grad = torch.autograd.grad(objective(z), z)[0]
    for step in (1e-5, 1e-6, 1e-7):
        direction = torch.zeros_like(z)
        direction[..., 0] = step
        fd = (objective(z.detach()+direction)-objective(z.detach()-direction))/(2*step)
        torch.testing.assert_close(fd, grad[..., 0].squeeze(), rtol=1e-7, atol=1e-9)


def test_one_hub_temperature_independence_and_measure_gradient():
    theta = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    mu = torch.ones(1, 1, dtype=torch.float64)
    projection = source_measure_sparsemax(torch.tensor([[[1.2]]], dtype=torch.float64)/theta.exp(), mu)
    torch.testing.assert_close(projection.density, torch.ones_like(projection.density))
    grad = torch.autograd.grad(projection.density.sum(), theta)[0]
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-14, rtol=0)
