"""Focused CPU checks for the Run-1408 regular-grid sampling seam."""

import pytest
import torch

from honf_forward_core.interface_fields.environment_sampling import (
    RegularGridLayout,
    sample_regular_grid,
    sample_regular_grid_reference,
)


def _layout(*, dtype: torch.dtype = torch.float64) -> RegularGridLayout:
    x = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=dtype)
    y = torch.tensor([0.25, 0.75, 1.25], dtype=dtype)
    token_to_grid = torch.cartesian_prod(
        torch.arange(y.numel(), dtype=torch.long),
        torch.arange(x.numel(), dtype=torch.long),
    )
    weights = torch.arange(1, token_to_grid.shape[0] + 1, dtype=dtype)
    return RegularGridLayout(
        physical_bounds=torch.tensor([[0.0, 0.0], [4.0, 1.5]], dtype=dtype),
        centre_axes=(x, y),
        token_to_grid=token_to_grid,
        weights=weights,
    )


def _affine_grid(layout: RegularGridLayout, *, batch: int = 1) -> torch.Tensor:
    x, y = layout.centre_axes
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    # A channel dimension lets the same check exercise packed K/V banks.
    field = torch.stack((2.0 * xx - 0.5 * yy + 1.0, -xx + 3.0 * yy - 2.0), dim=0)
    return field.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()


def test_token_mapping_is_explicit_and_permutation_invariant() -> None:
    layout = _layout()
    token_values = torch.arange(layout.num_tokens, dtype=torch.float64).reshape(1, -1, 1)
    canonical = layout.tokens_to_grid(token_values)

    permutation = torch.tensor([7, 0, 9, 4, 2, 10, 1, 8, 5, 3, 11, 6])
    permuted_values = token_values[:, permutation]
    permuted_layout = RegularGridLayout(
        physical_bounds=layout.physical_bounds,
        centre_axes=layout.centre_axes,
        token_to_grid=layout.token_to_grid[permutation],
        weights=layout.weights[permutation],
    )
    torch.testing.assert_close(canonical, permuted_layout.tokens_to_grid(permuted_values))
    torch.testing.assert_close(layout.weights_grid, permuted_layout.weights_grid)
    torch.testing.assert_close(
        token_values,
        layout.grid_to_tokens(canonical),
    )


def test_nodes_cell_centres_constant_and_affine_match_four_corner_reference() -> None:
    layout = _layout()
    values = _affine_grid(layout)
    yy, xx = torch.meshgrid(layout.centre_axes[1], layout.centre_axes[0], indexing="ij")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    expected_nodes = values[0].permute(1, 2, 0).reshape(-1, 2)
    sampled = sample_regular_grid(values, nodes, layout)
    reference = sample_regular_grid_reference(values, nodes, layout)
    torch.testing.assert_close(sampled[0], expected_nodes)
    torch.testing.assert_close(sampled, reference)

    points = torch.tensor([[0.75, 0.50], [2.25, 1.00], [3.25, 0.30]], dtype=torch.float64)
    expected = torch.stack((2.0 * points[:, 0] - 0.5 * points[:, 1] + 1.0,
                            -points[:, 0] + 3.0 * points[:, 1] - 2.0), dim=-1)
    torch.testing.assert_close(sample_regular_grid(values, points, layout)[0], expected)

    constant = torch.full((1, 1, *layout.grid_shape), 3.25, dtype=torch.float64)
    torch.testing.assert_close(
        sample_regular_grid(constant, points, layout),
        torch.full((1, 3, 1), 3.25, dtype=torch.float64),
    )


def test_incidence_reports_four_corners_and_weights_and_grid_sample_agrees() -> None:
    layout = _layout()
    points = torch.tensor([[0.75, 0.50], [3.5, 1.25]], dtype=torch.float64)
    result = sample_regular_grid(_affine_grid(layout), points, layout, return_incidence=True)
    incidence = result.incidence
    assert incidence.corner_indices.shape == (1, 2, 4, 2)
    assert incidence.corner_weights.shape == (1, 2, 4)
    torch.testing.assert_close(incidence.corner_weights.sum(dim=-1), torch.ones(1, 2, dtype=torch.float64))
    assert int(incidence.corner_indices[0, 0, 0, 0]) == 0
    assert int(incidence.corner_indices[0, 0, 0, 1]) == 0
    assert int(incidence.corner_indices[0, 0, 1, 1]) == 1
    torch.testing.assert_close(result.values, sample_regular_grid_reference(_affine_grid(layout), points, layout))


def test_singleton_axes_are_constant_and_physical_map_is_inward() -> None:
    x = torch.tensor([1.0], dtype=torch.float64)
    y = torch.tensor([0.25, 0.75, 1.25], dtype=torch.float64)
    mapping = torch.tensor([[row, 0] for row in range(3)], dtype=torch.long)
    layout = RegularGridLayout(
        physical_bounds=torch.tensor([[0.0, 0.0], [2.0, 1.5]], dtype=torch.float64),
        centre_axes=(x, y),
        token_to_grid=mapping,
        weights=torch.ones(3, dtype=torch.float64),
    )
    values = torch.arange(3, dtype=torch.float64).reshape(1, 1, 3, 1)
    points = torch.tensor([[0.0, 0.5], [2.0, 1.0]], dtype=torch.float64)
    mapped = layout.physical_to_centre_points(points)
    expected = ((mapped[:, 1] - y[0]) / (y[1] - y[0])).reshape(1, -1, 1)
    torch.testing.assert_close(sample_regular_grid(values, mapped, layout), expected)
    torch.testing.assert_close(mapped[:, 0], torch.ones(2, dtype=torch.float64))


def test_first_coordinate_gradients_match_affine_field_away_from_knots() -> None:
    layout = _layout()
    points = torch.tensor([[1.1, 0.62], [2.2, 1.08]], dtype=torch.float64, requires_grad=True)
    values = _affine_grid(layout)
    output = sample_regular_grid_reference(values, points, layout)
    objective = output[..., 0].sum()
    (gradient,) = torch.autograd.grad(objective, points)
    torch.testing.assert_close(gradient, torch.tensor([[2.0, -0.5], [2.0, -0.5]], dtype=torch.float64))

    points_grid = points.detach().clone().requires_grad_()
    objective_grid = sample_regular_grid(values, points_grid, layout)[..., 0].sum()
    (gradient_grid,) = torch.autograd.grad(objective_grid, points_grid)
    torch.testing.assert_close(gradient_grid, gradient)


def test_unsupported_layouts_and_out_of_hull_points_fail_loudly() -> None:
    with pytest.raises(ValueError, match="uniformly spaced"):
        RegularGridLayout(
            physical_bounds=torch.tensor([[0.0, 0.0], [3.0, 1.0]]),
            centre_axes=(torch.tensor([0.0, 1.0, 2.2]), torch.tensor([0.0, 1.0])),
            token_to_grid=torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [1, 2]]),
            weights=torch.ones(6),
        )
    with pytest.raises(ValueError, match="one-to-one"):
        RegularGridLayout(
            physical_bounds=torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
            centre_axes=(torch.tensor([0.5, 1.5]), torch.tensor([0.5, 1.5])),
            token_to_grid=torch.tensor([[0, 0], [0, 0], [1, 0], [1, 1]]),
            weights=torch.ones(4),
        )
    layout = _layout()
    with pytest.raises(ValueError, match="centre hull"):
        sample_regular_grid(_affine_grid(layout), torch.tensor([[0.1, 0.5]]), layout)
