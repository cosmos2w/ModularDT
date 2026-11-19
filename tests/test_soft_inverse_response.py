"""Algebra and physical-reduction checks; no model fitting or inference."""
import numpy as np
import pytest
import torch

from honf_runtime.local_response_contrasts import (
    CommonObservations, build_local_contrast, build_soft_inverse_contrast,
    build_uniform_contrast, neighborhood_field_losses,
)


def kernel(contrast):
    return contrast.operator.T @ contrast.operator


def helmert_chart(singular):
    """Construct a scene chart with exactly the specified centered spectrum."""
    n = len(singular) + 1
    helmert = build_uniform_contrast(n).operator / (2 / (n - 1)) ** .5
    return helmert.T @ torch.diag(torch.as_tensor(singular, dtype=torch.float64) * (n - 1) ** .5)


def test_soft_inverse_weights_trace_and_tight_uniform_bounds():
    chart = helmert_chart([12., 3., 1.])
    local = build_soft_inverse_contrast(chart)
    uniform = kernel(build_uniform_contrast(4))
    gram = kernel(local)
    assert local.rank == 3 and not local.operator.requires_grad
    assert torch.allclose(gram @ torch.ones(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64), atol=1e-14)
    assert torch.trace(gram).item() == pytest.approx(2., abs=1e-14)
    assert torch.linalg.eigvalsh(gram - .75 * uniform).min().item() >= -1e-14
    assert torch.linalg.eigvalsh(1.5 * uniform - gram).min().item() >= -1e-14
    z = 1 / (1 + (torch.tensor([12., 3., 1.], dtype=torch.float64) / 12).square())
    expected = .5 + .5 * 3 * z / z.sum()
    helmert = build_uniform_contrast(4).operator / (2 / 3) ** .5
    directions = helmert @ gram @ helmert.T / (2 / 3)
    assert torch.allclose(directions, torch.diag(expected), atol=1e-14)
    assert not torch.allclose(gram, kernel(build_local_contrast(chart)))


def test_soft_inverse_permutation_scale_shift_and_detachment():
    chart = helmert_chart([10., 3., 1.]).requires_grad_()
    gram = kernel(build_soft_inverse_contrast(chart))
    permutation = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(kernel(build_soft_inverse_contrast(chart[permutation])),
                          gram[permutation][:, permutation], atol=1e-14)
    assert torch.allclose(kernel(build_soft_inverse_contrast(chart * 1e-10 + 2e-10)), gram, atol=1e-14)
    assert chart.grad is None


@pytest.mark.parametrize('chart', [torch.tensor([[0., 0.], [1., 3.]], dtype=torch.float64),
                                  torch.tensor([[0.], [1.], [3.]], dtype=torch.float64),
                                  torch.eye(4, dtype=torch.float64)])
def test_pair_rank_one_and_equal_spectrum_are_uniform(chart):
    assert torch.allclose(kernel(build_soft_inverse_contrast(chart)), kernel(build_uniform_contrast(len(chart))), atol=1e-14)


def test_tied_cutoff_and_floor_do_not_split_or_invert_unsupported_modes():
    chart = helmert_chart([30., 1. + 2e-15, 1. - 2e-15])
    contrast = build_soft_inverse_contrast(chart)
    assert contrast.rank == 1
    assert torch.allclose(kernel(contrast), kernel(build_uniform_contrast(4)), atol=1e-14)
    precise = np.array([[1., 0.], [1. + 1e-10, 0.], [1., 2e-10]], dtype=np.float64)
    assert build_soft_inverse_contrast(precise).rank == 2
    assert build_soft_inverse_contrast(precise, absolute_floor=1e-9).rank == 0
    assert torch.allclose(kernel(build_soft_inverse_contrast(precise, absolute_floor=1e-9)),
                          kernel(build_uniform_contrast(3)), atol=1e-14)
    with pytest.raises(ValueError, match='absolute_floor'):
        build_soft_inverse_contrast(precise, absolute_floor=float('nan'))


def test_small_physical_fixture_fixed_mass_and_masked_channel_gradients():
    chart = helmert_chart([3., 1., .2])
    contrast = build_soft_inverse_contrast(chart)
    # Two valid positions have unequal fixed mass; the unavailable channel is
    # masked before subtraction, including its deliberately nonfinite truth.
    references = torch.tensor([[[1., float('nan')], [2., float('nan')]],
                               [[2., float('nan')], [4., float('nan')]],
                               [[3., float('nan')], [5., float('nan')]],
                               [[4., float('nan')], [6., float('nan')]]], dtype=torch.float64)
    mask = torch.tensor([[True, False], [True, False]])
    mass = torch.tensor([[.2, 0.], [.3, 0.]], dtype=torch.float64)
    residual = torch.tensor([1., 2., -1., 3.], dtype=torch.float64)
    prediction = references.clone().unsqueeze(0)
    prediction[..., 0] += residual.reshape(1, 4, 1)
    prediction.requires_grad_()
    values = neighborhood_field_losses(prediction, CommonObservations(references, mask, mass), contrast,
                                       [2., 3.], [1.5, 4.])
    expected_absolute = .5 * 1.5 * (residual / 2).square().mean()
    expected_response = .5 * 1.5 * (contrast.operator @ (residual / 2)).square().sum()
    assert values.absolute.item() == pytest.approx(expected_absolute.item())
    assert values.response.item() == pytest.approx(expected_response.item())
    assert values.channel_available.tolist() == [True, False]
    values.response.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[..., 1].abs().sum() == 0
    assert prediction.grad[..., 0].sum().item() == pytest.approx(0., abs=1e-14)
