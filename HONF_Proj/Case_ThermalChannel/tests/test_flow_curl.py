import numpy as np
import torch
from channelthermal.flow_curl import (
    build_near_obstacle_curl_stencil,
    generator_discrete_vorticity,
    torch_normalized_curl_from_stencil,
)


def test_generator_curl_uses_signed_cell_centered_gradient_and_zeros_solids():
    nx, ny = 9, 7
    dx, dy = 0.2, 0.3
    xx, yy = np.meshgrid(np.arange(nx, dtype=np.float64) * dx,
                         np.arange(ny, dtype=np.float64) * dy)
    u = 2.0 * xx + 4.0 * yy
    v = 7.0 * xx - 3.0 * yy
    solid = np.zeros((ny, nx), dtype=bool)
    solid[3, 4] = True
    u[solid] = 0.0
    v[solid] = 0.0

    actual = generator_discrete_vorticity(u, v, solid, dx, dy)
    expected = np.gradient(v, dx, axis=1, edge_order=1) - np.gradient(u, dy, axis=0, edge_order=1)
    expected[solid] = 0.0
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert actual[0, 0] == 3.0
    assert actual[3, 4] == 0.0
    assert actual[2, 4] != 3.0  # The adjacent solid velocity participates in the native stencil.


def test_near_curl_stencil_is_deterministic_and_keeps_native_neighbours():
    ny, nx = 24, 32
    dx, dy = 0.1, 0.12
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    x_grid, y_grid = np.meshgrid(xs, ys)
    center = np.asarray([[1.6, 1.2]], dtype=np.float32)
    radius = 0.25
    solid = np.hypot(x_grid - center[0, 0], y_grid - center[0, 1]) <= radius
    target = np.zeros_like(x_grid, dtype=np.float32)
    first = build_near_obstacle_curl_stencil(
        x_grid=x_grid, y_grid=y_grid, module_centers=center, module_present=np.asarray([1]),
        module_mask=solid, omega_target=target, radius=radius, dx=dx, dy=dy,
        max_centers=20, seed=19,
    )
    second = build_near_obstacle_curl_stencil(
        x_grid=x_grid, y_grid=y_grid, module_centers=center, module_present=np.asarray([1]),
        module_mask=solid, omega_target=target, radius=radius, dx=dx, dy=dy,
        max_centers=20, seed=19,
    )

    assert first.candidate_count > first.valid.sum() == 20
    assert np.array_equal(first.center_indices, second.center_indices)
    assert np.array_equal(first.query_xy, second.query_xy)
    assert first.query_xy.shape == (100, 2)
    assert first.solid_nodes.shape == (20, 5)
    assert not first.solid_nodes[:, 0].any()
    assert np.all(first.x_span > 0.0) and np.all(first.y_span > 0.0)


def test_torch_stencil_curl_uses_physical_scales_and_keeps_gradients():
    # Physical u=2x+4y and v=7x-3y have dv/dx-du/dy=3 everywhere.
    field_mean = np.asarray([1.2, -0.5, 2.0, 0.7], dtype=np.float32)
    field_std = np.asarray([2.0, 0.25, 3.0, 0.5], dtype=np.float32)
    physical = torch.tensor(
        [[[[4.0, 1.0, 0.0, 3.0], [3.6, 0.3, 0.0, 3.0],
           [4.4, 1.7, 0.0, 3.0], [3.4, 1.0, 0.0, 3.0],
           [4.6, 1.0, 0.0, 3.0]]]], dtype=torch.float32,
    )
    normalized = ((physical - torch.tensor(field_mean)) / torch.tensor(field_std)).requires_grad_()
    omega, curl = torch_normalized_curl_from_stencil(
        normalized,
        torch.zeros((1, 1, 5), dtype=torch.bool),
        torch.tensor([[0.2]]), torch.tensor([[0.3]]), field_mean, field_std,
    )
    torch.testing.assert_close(omega, torch.tensor([[((3.0 - 0.7) / 0.5)]]))
    torch.testing.assert_close(curl, torch.tensor([[((3.0 - 0.7) / 0.5)]]))
    curl.sum().backward()
    assert normalized.grad is not None and torch.isfinite(normalized.grad).all()
