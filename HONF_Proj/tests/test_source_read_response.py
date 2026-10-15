from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode

from honf_forward_core.content_hypergraph.source_read_response import SourceReadAffineFactorsV1


_DTYPE = torch.float64
_ENVELOPE = 2.5


def _make_factors(
    *,
    batch: int = 2,
    queries: int = 3,
    sources: int = 5,
    completion: bool = True,
    requires_grad: bool = False,
) -> SourceReadAffineFactorsV1:
    torch.manual_seed(8317 + batch + queries + sources)
    present = torch.ones((batch, sources), dtype=torch.bool)
    if sources > 2:
        present[0, -1] = False
    if batch > 1 and sources > 1:
        present[1, 1] = False
    query_mask = torch.ones((batch, queries), dtype=torch.bool)
    if queries > 1:
        query_mask[0, -1] = False

    valid = present[:, None, :] & query_mask[:, :, None]
    rows = valid.nonzero(as_tuple=False)[::2]
    selected_batch, selected_query, selected_source = rows.unbind(dim=1)

    def leaf(shape: tuple[int, ...]) -> Tensor:
        return torch.randn(shape, dtype=_DTYPE, requires_grad=requires_grad)

    coeff = leaf((rows.shape[0],))
    if completion:
        q_factor = leaf((batch, queries, 16))
        s_factor = leaf((batch, sources, 16))
        rank_weight = leaf((16,))
        bias = torch.tensor(-0.37, dtype=_DTYPE, requires_grad=requires_grad)
    else:
        q_factor = s_factor = rank_weight = None
        bias = 0.0
    return SourceReadAffineFactorsV1(
        batch_size=batch,
        query_count=queries,
        source_count=sources,
        selected_batch=selected_batch,
        selected_query=selected_query,
        selected_source=selected_source,
        residual_coefficients=coeff,
        present_mask=present,
        query_mask=query_mask,
        completion_query=q_factor,
        completion_source=s_factor,
        thermal_rank_weight=rank_weight,
        scalar_bias=bias,
        heating_envelope=_ENVELOPE,
    )


def _controls(factors: SourceReadAffineFactorsV1, *, requires_grad: bool = False) -> Tensor:
    values = torch.rand((factors.batch_size, factors.source_count), dtype=_DTYPE)
    values = values * (_ENVELOPE * 0.7)
    if factors.source_count:
        values = torch.where(factors.present_mask, values, torch.full_like(values, 7.0))
    return values.requires_grad_(requires_grad)


def _independent_completion_kernel(factors: SourceReadAffineFactorsV1) -> Tensor:
    assert factors.completion_query is not None
    assert factors.completion_source is not None
    assert factors.thermal_rank_weight is not None
    rows = []
    for b in range(factors.batch_size):
        query_rows = []
        for q in range(factors.query_count):
            source_values = []
            for m in range(factors.source_count):
                value = factors.residual_coefficients.new_zeros(())
                if bool(factors.present_mask[b, m]) and (
                    factors.query_mask is None or bool(factors.query_mask[b, q])
                ):
                    value = (
                        factors.completion_query[b, q]
                        * factors.completion_source[b, m]
                        * factors.thermal_rank_weight
                    ).sum() + factors.scalar_bias
                    for p in range(factors.residual_coefficients.numel()):
                        if (
                            int(factors.selected_batch[p]) == b
                            and int(factors.selected_query[p]) == q
                            and int(factors.selected_source[p]) == m
                        ):
                            value = value + factors.residual_coefficients[p]
                source_values.append(value)
            query_rows.append(
                torch.stack(source_values)
                if source_values
                else factors.residual_coefficients.new_empty((0,))
            )
        rows.append(
            torch.stack(query_rows)
            if query_rows
            else factors.residual_coefficients.new_empty((0, factors.source_count))
        )
    return torch.stack(rows) if rows else factors.residual_coefficients.new_empty(
        (0, factors.query_count, factors.source_count)
    )


def test_completion_apply_transpose_and_dense_export_match_fp64_reference() -> None:
    factors = _make_factors(completion=True)
    controls = _controls(factors)
    cotangent = torch.randn((factors.batch_size, factors.query_count), dtype=_DTYPE)
    dense = factors.explicit_dense_kernel(torch.float64)
    reference = _independent_completion_kernel(factors)

    torch.testing.assert_close(dense, reference, rtol=1e-12, atol=1e-12)
    applied = factors.apply(controls, accumulation_dtype=torch.float64)
    transposed = factors.transpose(cotangent, accumulation_dtype=torch.float64)
    torch.testing.assert_close(applied, (reference * controls[:, None, :]).sum(dim=-1))
    torch.testing.assert_close(transposed, (reference * cotangent[:, :, None]).sum(dim=1))
    torch.testing.assert_close((applied * cotangent).sum(), (controls * transposed).sum())


def test_utility_coefficients_include_gate_and_signed_scalar_bias_once() -> None:
    factors = _make_factors(completion=False)
    gate = torch.linspace(0.1, 0.9, factors.residual_coefficients.numel(), dtype=_DTYPE)
    weighted_fine = torch.linspace(-1.2, 1.1, gate.numel(), dtype=_DTYPE)
    scalar_bias = torch.tensor(0.43, dtype=_DTYPE)
    coefficients = gate * (weighted_fine + scalar_bias)
    utility = SourceReadAffineFactorsV1(
        batch_size=factors.batch_size,
        query_count=factors.query_count,
        source_count=factors.source_count,
        selected_batch=factors.selected_batch,
        selected_query=factors.selected_query,
        selected_source=factors.selected_source,
        residual_coefficients=coefficients,
        present_mask=factors.present_mask,
        query_mask=factors.query_mask,
        heating_envelope=_ENVELOPE,
    )
    controls = _controls(utility)
    dense = utility.explicit_dense_kernel(torch.float64)
    expected = torch.zeros_like(dense)
    expected[utility.selected_batch, utility.selected_query, utility.selected_source] = coefficients
    expected = expected * utility.present_mask[:, None, :]
    expected = expected * utility.query_mask[:, :, None]

    torch.testing.assert_close(dense, expected)
    torch.testing.assert_close(
        utility.apply(controls), (expected * controls[:, None, :]).sum(dim=-1)
    )


def test_completion_selected_coefficients_are_gated_fine_minus_base_residuals() -> None:
    template = _make_factors(completion=True)
    gate = torch.linspace(0.05, 0.95, template.residual_coefficients.numel(), dtype=_DTYPE)
    weighted_fine = torch.linspace(-0.8, 1.3, gate.numel(), dtype=_DTYPE)
    weighted_base = torch.linspace(0.4, -0.6, gate.numel(), dtype=_DTYPE)
    residual = gate * (weighted_fine - weighted_base)
    factors = SourceReadAffineFactorsV1(
        batch_size=template.batch_size,
        query_count=template.query_count,
        source_count=template.source_count,
        selected_batch=template.selected_batch,
        selected_query=template.selected_query,
        selected_source=template.selected_source,
        residual_coefficients=residual,
        present_mask=template.present_mask,
        query_mask=template.query_mask,
        completion_query=template.completion_query,
        completion_source=template.completion_source,
        thermal_rank_weight=template.thermal_rank_weight,
        scalar_bias=template.scalar_bias,
        heating_envelope=_ENVELOPE,
    )
    controls = _controls(factors)
    dense = factors.explicit_dense_kernel()
    base = torch.einsum(
        "bqr,bmr,r->bqm",
        factors.completion_query,
        factors.completion_source,
        factors.thermal_rank_weight,
    )
    base = base + factors.scalar_bias * factors.present_mask[:, None, :]
    expected = base.clone()
    expected[factors.selected_batch, factors.selected_query, factors.selected_source] += residual
    expected *= factors.present_mask[:, None, :]
    expected *= factors.query_mask[:, :, None]
    torch.testing.assert_close(dense, expected)
    torch.testing.assert_close(
        factors.apply(controls), (expected * controls[:, None, :]).sum(dim=-1)
    )


def test_completion_gradient_matches_dense_kernel_and_transpose() -> None:
    factors = _make_factors(completion=True, requires_grad=True)
    controls = _controls(factors, requires_grad=True)
    cotangent = torch.randn((factors.batch_size, factors.query_count), dtype=_DTYPE)
    leaves = [
        controls,
        factors.residual_coefficients,
        factors.completion_query,
        factors.completion_source,
        factors.thermal_rank_weight,
        factors.scalar_bias,
    ]
    sparse_value = (factors.apply(controls) * cotangent).sum()
    sparse_grads = torch.autograd.grad(sparse_value, leaves, retain_graph=True)
    dense_value = (factors.explicit_dense_kernel() * controls[:, None, :] * cotangent[:, :, None]).sum()
    dense_grads = torch.autograd.grad(dense_value, leaves)
    for sparse_grad, dense_grad in zip(sparse_grads, dense_grads):
        torch.testing.assert_close(sparse_grad, dense_grad, rtol=1e-11, atol=1e-11)

    expected_control_grad = factors.transpose(cotangent)
    actual_control_grad = sparse_grads[0]
    torch.testing.assert_close(actual_control_grad, expected_control_grad, rtol=1e-11, atol=1e-11)


class _RejectDenseBQM(TorchDispatchMode):
    def __init__(self, shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.shape = shape

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        leaves, _ = tree_flatten(output)
        if any(isinstance(value, Tensor) and tuple(value.shape) == self.shape for value in leaves):
            raise AssertionError(f"ordinary response path materialized dense kernel {self.shape}")
        return output


def test_apply_and_transpose_do_not_materialize_bqm() -> None:
    factors = _make_factors(batch=2, queries=3, sources=7, completion=True)
    controls = _controls(factors)
    cotangent = torch.randn((2, 3), dtype=_DTYPE)
    with _RejectDenseBQM((2, 3, 7)):
        factors.apply(controls)
        factors.transpose(cotangent)


@pytest.mark.parametrize("sources", [0, 1, 7])
def test_arbitrary_source_count_and_empty_source_axis(sources: int) -> None:
    factors = _make_factors(batch=2, queries=3, sources=sources, completion=True)
    controls = _controls(factors)
    cotangent = torch.randn((2, 3), dtype=_DTYPE)
    dense = factors.explicit_dense_kernel()
    torch.testing.assert_close(factors.apply(controls), (dense * controls[:, None, :]).sum(-1))
    torch.testing.assert_close(factors.transpose(cotangent), (dense * cotangent[:, :, None]).sum(1))
    assert dense.shape == (2, 3, sources)
    if sources == 0:
        assert torch.count_nonzero(factors.apply(controls)) == 0
        assert factors.transpose(cotangent).shape == (2, 0)


def test_empty_query_axis_returns_empty_response_and_zero_transpose() -> None:
    factors = _make_factors(batch=2, queries=0, sources=4, completion=True)
    controls = _controls(factors)
    cotangent = torch.empty((2, 0), dtype=_DTYPE)
    assert factors.apply(controls).shape == (2, 0)
    assert torch.equal(factors.transpose(cotangent), torch.zeros((2, 4), dtype=_DTYPE))
    assert factors.explicit_dense_kernel().shape == (2, 0, 4)


def test_zero_heat_padding_query_mask_and_endpoint_envelope() -> None:
    factors = _make_factors(completion=True)
    zero = torch.zeros((factors.batch_size, factors.source_count), dtype=_DTYPE)
    assert torch.equal(factors.apply(zero), torch.zeros((factors.batch_size, factors.query_count), dtype=_DTYPE))

    endpoint = _controls(factors)
    endpoint[factors.present_mask] = _ENVELOPE
    endpoint[~factors.present_mask] = 999.0
    result = factors.apply(endpoint)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(factors.apply(-endpoint), -result)

    invalid_low = endpoint.clone()
    invalid_low[0, 0] = -_ENVELOPE - 0.01
    with pytest.raises(ValueError, match="abs\\(h\\)"):
        factors.apply(invalid_low)
    invalid_high = endpoint.clone()
    invalid_high[0, 0] = _ENVELOPE + 0.01
    with pytest.raises(ValueError, match="abs\\(h\\)"):
        factors.apply(invalid_high)

    tangent = torch.full_like(endpoint, 7.0)
    tangent[~factors.present_mask] = -8.0
    tangent_result = factors.apply(tangent, check_envelope=False)
    torch.testing.assert_close(
        tangent_result,
        (factors.explicit_dense_kernel() * tangent[:, None, :]).sum(-1),
    )


def test_factor_version_is_stable_and_tracks_in_place_changes() -> None:
    factors = _make_factors(completion=True)
    names = tuple(factors.tensor_values())
    assert names == (
        "selected_batch",
        "selected_query",
        "selected_source",
        "residual_coefficients",
        "present_mask",
        "query_mask",
        "completion_query",
        "completion_source",
        "thermal_rank_weight",
        "scalar_bias",
        "heating_envelope",
    )
    before = factors.version()
    assert before == factors.version()
    with torch.no_grad():
        factors.residual_coefficients.add_(1.0)
    assert factors.version() != before
    replaced = factors.version()
    object.__setattr__(factors, "residual_coefficients", factors.residual_coefficients.clone())
    assert factors.version() != replaced


def test_invalid_selected_padding_and_incomplete_completion_are_rejected() -> None:
    factors = _make_factors(completion=True)
    with pytest.raises(ValueError, match="present sources"):
        SourceReadAffineFactorsV1(
            batch_size=1,
            query_count=1,
            source_count=1,
            selected_batch=torch.tensor([0]),
            selected_query=torch.tensor([0]),
            selected_source=torch.tensor([0]),
            residual_coefficients=torch.ones(1, dtype=_DTYPE),
            present_mask=torch.zeros((1, 1), dtype=torch.bool),
            heating_envelope=_ENVELOPE,
        )
    with pytest.raises(ValueError, match="supplied together"):
        SourceReadAffineFactorsV1(
            batch_size=1,
            query_count=1,
            source_count=1,
            selected_batch=torch.empty(0, dtype=torch.long),
            selected_query=torch.empty(0, dtype=torch.long),
            selected_source=torch.empty(0, dtype=torch.long),
            residual_coefficients=torch.empty(0, dtype=_DTYPE),
            present_mask=torch.ones((1, 1), dtype=torch.bool),
            heating_envelope=_ENVELOPE,
            completion_query=torch.zeros((1, 1, 16), dtype=_DTYPE),
        )
