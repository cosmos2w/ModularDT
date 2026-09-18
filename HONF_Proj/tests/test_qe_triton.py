"""Optional fused QE reader checks.

The reference tests run on every host.  CUDA tests are skipped when Triton or
the selected CUDA device is unavailable; importing the optional kernel must
never make the CPU test suite require a GPU runtime.
"""

from __future__ import annotations

import pytest
import torch

from honf_forward_core.interface_fields.kernels.qe_triton import (
    TRITON_AVAILABLE,
    fused_qe_reader,
    is_triton_qe_available,
    qe_reader_reference,
)


def _complete_fixture(*, dtype: torch.dtype = torch.float32, device: str = "cpu"):
    generator = torch.Generator(device=device).manual_seed(204)
    batch, heads, queries, sources, head_dim = 2, 2, 3, 5, 4
    query = torch.randn(batch, heads, queries, head_dim, generator=generator, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(batch, heads, sources, head_dim, generator=generator, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(batch, heads, sources, head_dim, generator=generator, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(batch, heads, queries, sources, generator=generator, device=device, dtype=dtype, requires_grad=True)
    prior = torch.rand(batch, queries, sources, generator=generator, device=device, dtype=torch.float64).add_(0.1).requires_grad_()
    return query, key, value, bias, prior


def test_optional_import_and_cpu_reference_complete_backward() -> None:
    assert isinstance(TRITON_AVAILABLE, bool)
    assert not is_triton_qe_available("cpu")
    query, key, value, bias, prior = _complete_fixture()
    output = qe_reader_reference(query, key, value, bias, prior, mode="complete")
    assert output.shape == (2, 2, 3, 4)
    assert output.dtype == query.dtype
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    for tensor in (query, key, value, bias, prior):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_cpu_fused_request_fails_explicitly_and_csr_empty_rows_are_exact() -> None:
    query, key, value, bias, prior = _complete_fixture()
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        fused_qe_reader(query, key, value, bias, prior, mode="complete")

    # CSR rows are flattened in (batch, receiver) order.  Rows 1 and 4 are
    # empty and must retain the exact zero-context semantics.
    row_offsets = torch.tensor([0, 2, 2, 5, 6, 6, 8], dtype=torch.long)
    source_indices = torch.tensor([0, 4, 1, 2, 4, 3, 0, 2], dtype=torch.long)
    pair_prior = torch.tensor([0.4, 0.6, 0.2, 0.3, 0.5, 1.0, 0.7, 0.3], dtype=torch.float64, requires_grad=True)
    pair_bias = torch.randn(8, query.shape[1], dtype=query.dtype, requires_grad=True)
    output = qe_reader_reference(
        query,
        key,
        value,
        pair_bias,
        pair_prior,
        mode="csr",
        row_offsets=row_offsets,
        source_indices=source_indices,
    )
    assert torch.equal(output[0, :, 1, :], torch.zeros_like(output[0, :, 1, :]))
    assert torch.equal(output[1, :, 1, :], torch.zeros_like(output[1, :, 1, :]))
    output.square().sum().backward()
    assert pair_prior.grad is not None and torch.isfinite(pair_prior.grad).all()
    assert pair_bias.grad is not None and torch.isfinite(pair_bias.grad).all()


def test_reference_complete_source_mask_keeps_invalid_slots_implicit() -> None:
    query, key, value, bias, prior = _complete_fixture()
    source_mask = torch.tensor(
        [[True, False, True, False, True], [False, True, True, False, True]],
        dtype=torch.bool,
    )
    masked_prior = torch.where(source_mask[:, None, :], prior, torch.ones_like(prior))
    output = qe_reader_reference(
        query,
        key,
        value,
        bias,
        masked_prior,
        mode="complete",
        source_mask=source_mask,
    )
    expected_scores = torch.einsum("bhqd,bhnd->bhqn", query, key) / (float(query.shape[-1]) ** 0.5)
    expected_scores = (expected_scores + bias).double() + masked_prior.double().log()[:, None, :, :]
    expected_scores = expected_scores.masked_fill(~source_mask[:, None, None, :], torch.finfo(torch.float64).min)
    expected_weights = torch.softmax(expected_scores, dim=-1)
    expected_weights = expected_weights * source_mask[:, None, None, :].to(expected_weights.dtype)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    expected = torch.einsum("bhqn,bhnd->bhqd", expected_weights.to(value.dtype), value)
    torch.testing.assert_close(output, expected)


def test_reference_csr_coordinate_bias_has_live_coordinate_gradient() -> None:
    query, key, value, _, prior = _complete_fixture()
    coordinates = torch.randn(2, 5, 3, dtype=query.dtype, requires_grad=True)
    receivers = torch.randn(2, 3, 3, dtype=query.dtype)
    batch_index = torch.arange(2).repeat_interleave(4)
    receiver_index = torch.tensor([0, 0, 2, 2, 0, 1, 1, 2], dtype=torch.long)
    source_index = torch.tensor([0, 4, 1, 2, 3, 0, 2, 4], dtype=torch.long)
    relative = receivers[batch_index, receiver_index] - coordinates[batch_index, source_index]
    pair_bias = relative.sum(dim=-1, keepdim=True).expand(-1, query.shape[1])
    pair_prior = prior[batch_index, receiver_index, source_index].detach().clone().requires_grad_()
    row_offsets = torch.tensor([0, 2, 2, 4, 5, 7, 8], dtype=torch.long)
    output = qe_reader_reference(
        query,
        key,
        value,
        pair_bias,
        pair_prior,
        mode="csr",
        row_offsets=row_offsets,
        source_indices=source_index,
    )
    output.square().sum().backward()
    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()
    assert coordinates.grad.abs().sum() > 0


cuda_qe = pytest.mark.skipif(
    not is_triton_qe_available("cuda"),
    reason="Triton QE requires an available CUDA device",
)


def _relative_max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compare gradients without overflowing an L2 norm for tiny priors."""

    scale = torch.maximum(actual.detach().abs().amax(), expected.detach().abs().amax())
    scale = scale.clamp_min(torch.finfo(actual.dtype).tiny)
    return float((actual.detach() - expected.detach()).abs().amax() / scale)


@cuda_qe
def test_triton_complete_forward_backward_matches_reference_and_live_inputs() -> None:
    query, key, value, bias, prior = _complete_fixture(device="cuda")
    reference = qe_reader_reference(query, key, value, bias, prior, mode="complete")
    fused = fused_qe_reader(query, key, value, bias, prior, mode="complete")
    torch.testing.assert_close(fused, reference, rtol=3e-5, atol=3e-5)
    probe = torch.randn_like(fused)
    fused_loss = (fused * probe).sum()
    fused_grads = torch.autograd.grad(fused_loss, (query, key, value, bias, prior), retain_graph=True)
    ref_loss = (reference * probe).sum()
    ref_grads = torch.autograd.grad(ref_loss, (query, key, value, bias, prior))
    for actual, expected in zip(fused_grads, ref_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
    assert all(torch.isfinite(value).all() for value in fused_grads)


@cuda_qe
def test_triton_csr_backward_reaches_prior_value_bias_and_coordinates() -> None:
    torch.manual_seed(211)
    device = "cuda"
    dtype = torch.float32
    batch, heads, queries, sources, head_dim = 2, 2, 3, 5, 4
    query = torch.randn(batch, heads, queries, head_dim, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(batch, heads, sources, head_dim, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(batch, heads, sources, head_dim, device=device, dtype=dtype, requires_grad=True)
    coordinates = torch.randn(batch, sources, 3, device=device, dtype=dtype, requires_grad=True)
    receivers = torch.randn(batch, queries, 3, device=device, dtype=dtype)
    row_offsets = torch.tensor([0, 2, 2, 5, 6, 8, 8], device=device, dtype=torch.long)
    source_indices = torch.tensor([0, 4, 1, 2, 4, 3, 0, 2], device=device, dtype=torch.long)
    pair_batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], device=device, dtype=torch.long)
    pair_receiver = torch.tensor([0, 0, 2, 0, 1, 1, 2, 2], device=device, dtype=torch.long)
    relative = receivers[pair_batch, pair_receiver] - coordinates[pair_batch, source_indices]
    pair_bias = relative.sum(dim=-1, keepdim=True).expand(-1, heads).contiguous()
    pair_prior = torch.rand(8, device=device, dtype=torch.float64).add_(0.2).requires_grad_()
    fused = fused_qe_reader(
        query,
        key,
        value,
        pair_bias,
        pair_prior,
        mode="csr",
        row_offsets=row_offsets,
        source_indices=source_indices,
    )
    reference = qe_reader_reference(
        query,
        key,
        value,
        pair_bias,
        pair_prior,
        mode="csr",
        row_offsets=row_offsets,
        source_indices=source_indices,
    )
    torch.testing.assert_close(fused, reference, rtol=3e-5, atol=3e-5)
    probe = torch.randn_like(fused)
    fused_loss = (fused * probe).sum()
    fused_gradients = torch.autograd.grad(
        fused_loss,
        (query, key, value, pair_bias, pair_prior, coordinates),
        retain_graph=True,
    )
    reference_loss = (reference * probe).sum()
    reference_gradients = torch.autograd.grad(
        reference_loss,
        (query, key, value, pair_bias, pair_prior, coordinates),
    )
    for fused_gradient, reference_gradient in zip(fused_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(fused_gradient, reference_gradient, rtol=4e-4, atol=4e-4)
    for gradient in fused_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


@cuda_qe
def test_triton_model_sized_tiny_prior_gradients_are_scale_aware() -> None:
    """Exercise the production head/tile shape and the stable dPi formula."""

    torch.manual_seed(217)
    device = "cuda"
    batch, heads, queries, sources, head_dim = 1, 4, 128, 192, 64
    query = torch.randn(batch, heads, queries, head_dim, device=device, requires_grad=True)
    key = torch.randn(batch, heads, sources, head_dim, device=device, requires_grad=True)
    value = torch.randn(batch, heads, sources, head_dim, device=device, requires_grad=True)
    bias = torch.randn(batch, heads, queries, sources, device=device, requires_grad=True)
    prior = torch.full(
        (batch, queries, sources),
        1.0e-300,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    fused = fused_qe_reader(query, key, value, bias, prior, mode="complete")
    reference = qe_reader_reference(query, key, value, bias, prior, mode="complete")
    probe = torch.randn_like(fused)
    fused_gradients = torch.autograd.grad(
        (fused * probe).sum(),
        (query, key, value, bias, prior),
        retain_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        (reference * probe).sum(),
        (query, key, value, bias, prior),
    )
    assert _relative_max_error(fused, reference) < 1.0e-4
    for fused_gradient, reference_gradient in zip(fused_gradients, reference_gradients, strict=True):
        assert torch.isfinite(fused_gradient).all()
        assert _relative_max_error(fused_gradient, reference_gradient) < 1.0e-4


@cuda_qe
def test_triton_fp64_variant_and_higher_order_fallback_remain_connected() -> None:
    """FP64 neural inputs use the scalar variant; create_graph uses Torch."""

    torch.manual_seed(218)
    device = "cuda"
    query = torch.randn(1, 2, 3, 7, device=device, dtype=torch.float64, requires_grad=True)
    key = torch.randn(1, 2, 5, 7, device=device, dtype=torch.float64, requires_grad=True)
    value = torch.randn(1, 2, 5, 7, device=device, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(1, 2, 3, 5, device=device, dtype=torch.float64, requires_grad=True)
    prior = torch.rand(1, 3, 5, device=device, dtype=torch.float64).add_(0.2).requires_grad_()
    fused = fused_qe_reader(query, key, value, bias, prior, mode="complete")
    reference = qe_reader_reference(query, key, value, bias, prior, mode="complete")
    assert _relative_max_error(fused, reference) < 1.0e-7
    probe = torch.randn_like(fused)
    fused_gradients = torch.autograd.grad(
        (fused * probe).sum(),
        (query, key, value, bias, prior),
        retain_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        (reference * probe).sum(),
        (query, key, value, bias, prior),
    )
    for fused_gradient, reference_gradient in zip(fused_gradients, reference_gradients, strict=True):
        assert _relative_max_error(fused_gradient, reference_gradient) < 1.0e-6

    query = torch.randn(1, 1, 2, 5, device=device, requires_grad=True)
    key = torch.randn(1, 1, 4, 5, device=device, requires_grad=True)
    value = torch.randn(1, 1, 4, 5, device=device, requires_grad=True)
    bias = torch.randn(1, 1, 2, 4, device=device, requires_grad=True)
    prior = torch.rand(1, 2, 4, device=device, dtype=torch.float64).add_(0.1).requires_grad_()
    output = fused_qe_reader(query, key, value, bias, prior, mode="complete")
    first_gradient = torch.autograd.grad(output.square().sum(), query, create_graph=True)[0]
    second_gradient = torch.autograd.grad(first_gradient.square().sum(), query)[0]
    assert first_gradient.requires_grad
    assert torch.isfinite(second_gradient).all()
    assert second_gradient.abs().sum() > 0
