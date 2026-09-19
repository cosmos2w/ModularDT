"""Focused exact-support tests for the Run-1407 K=6 metadata helper."""

from __future__ import annotations

import pytest
import torch

from honf_forward_core.interface_fields.group_control_support import (
    QUERY_MASK_COUNT,
    build_six_bit_support_index,
    live_pair_values,
    six_bit_mask,
)


def _mask_incidence(masks: torch.Tensor, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    powers = torch.ones(6, dtype=torch.long, device=masks.device)
    powers = powers.bitwise_left_shift(torch.arange(6, dtype=torch.long, device=masks.device))
    return (masks[..., None].bitwise_and(powers) != 0).to(dtype)


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_membership = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.2, 0.3, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.1, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.7, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float64,
    )
    source_valid = torch.tensor(
        [[True, True, True, False, True], [True, False, True, True, True]],
        dtype=torch.bool,
    )
    # A positive-membership source with zero measure is also inactive.  This
    # exercises the module-padding and valid-source path independently.
    source_measure = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    masks = torch.arange(QUERY_MASK_COUNT, dtype=torch.long).repeat(2, 1)
    query_assignment = _mask_incidence(masks)
    return source_membership, query_assignment, source_valid, source_measure


def test_six_bit_masks_and_all_64_query_supports_match_dense_boolean_reference() -> None:
    source_membership, query_assignment, source_valid, source_measure = _fixture()
    index = build_six_bit_support_index(
        source_membership,
        query_assignment,
        source_valid=source_valid,
        source_measure=source_measure,
    )

    # The source masks include validity and positive-measure filtering, while
    # query masks preserve every one of the 64 possible signatures.
    assert index.source_masks.tolist() == [[1, 2, 0, 4, 24], [0, 32, 17, 0, 4]]
    assert index.active_source_masks.tolist() == [[1, 2, 0, 0, 0], [0, 0, 17, 0, 4]]
    assert index.query_masks.tolist() == [list(range(64)), list(range(64))]
    assert index.support_table.shape == (2, 64, 5)
    assert index.support_table.dtype == torch.bool

    expected_source_bits = (source_membership > 0) & source_valid[..., None] & (source_measure > 0)[..., None]
    expected_query_bits = query_assignment > 0
    expected = torch.zeros_like(index.support_table)
    for mask in range(64):
        query_bits = expected_query_bits[:, mask]
        expected[:, mask] = (expected_source_bits & query_bits[:, None, :]).any(dim=-1)
    torch.testing.assert_close(index.support_table, expected)
    torch.testing.assert_close(index.support_count_table, expected.sum(dim=-1, dtype=torch.long))
    # Empty query signatures and sources with no positive group incidence are
    # exact empty support, not a fabricated fallback source.
    assert not bool(index.support_table[:, 0].any())
    assert index.support_pairs_per_query[:, 0].tolist() == [0, 0]
    assert index.logical_paths_per_query[:, 0].tolist() == [0, 0]
    # Raw positive A incidence remains a logical path even when its source is
    # invalid/padded; executable support correctly excludes that row.
    assert index.logical_paths_per_query[0, 4].item() == 1
    assert index.support_pairs_per_query[0, 4].item() == 0


def test_query_signatures_preserve_order_and_return_unique_source_blocks() -> None:
    source_membership, _, source_valid, source_measure = _fixture()
    masks = torch.tensor([[5, 1, 5, 0, 63], [4, 4, 2, 0, 4]], dtype=torch.long)
    query_assignment = _mask_incidence(masks)
    index = build_six_bit_support_index(
        source_membership,
        query_assignment,
        source_valid=source_valid,
        source_measure=source_measure,
    )

    first = index.signature_selection(0, 5)
    assert first.query_index.tolist() == [0, 2]
    assert first.source_index.tolist() == [0]
    assert first.support_pair_count == 2
    empty = index.signature_selection(0, 0)
    assert empty.query_index.tolist() == [3]
    assert empty.source_index.tolist() == []

    second = index.signature_selection(1, 4)
    assert second.query_index.tolist() == [0, 1, 4]
    assert second.source_index.tolist() == [4]
    assert second.support_pair_count == 3

    pairs = index.selected_pair_indices()
    assert pairs.ndim == 2 and tuple(pairs.shape[1:]) == (3,)
    assert int(torch.unique(pairs, dim=0).shape[0]) == int(pairs.shape[0])
    # There is no source entry for the empty query signature.
    assert not bool(((pairs[:, 0] == 0) & (pairs[:, 1] == 3)).any())


def test_live_pair_values_keep_rho_and_moments_differentiable() -> None:
    source_membership, query_assignment, source_valid, source_measure = _fixture()
    source_membership = source_membership.clone().requires_grad_()
    query_assignment = query_assignment.clone().requires_grad_()
    group_control = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)
    index = build_six_bit_support_index(
        source_membership,
        query_assignment,
        source_valid=source_valid,
        source_measure=source_measure,
    )
    pairs = index.selected_pair_indices()
    rho, moments = live_pair_values(
        query_assignment,
        source_membership,
        group_control,
        pairs,
        chunk_size=3,
    )
    batches, queries, sources = pairs.unbind(dim=-1)
    expected_rho = (query_assignment[batches, queries] * source_membership[batches, sources]).sum(dim=-1)
    expected_moments = torch.einsum(
        "pk,pk,pkd->pd",
        query_assignment[batches, queries],
        source_membership[batches, sources],
        group_control[batches],
    )
    torch.testing.assert_close(rho, expected_rho)
    torch.testing.assert_close(moments, expected_moments)
    assert rho.requires_grad and moments.requires_grad
    loss = rho.square().sum() + moments.square().sum()
    gradients = torch.autograd.grad(loss, (query_assignment, source_membership, group_control))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert any(bool(value.abs().sum() > 0) for value in gradients)


def test_group_permutation_preserves_support_table_and_logical_counts() -> None:
    source_membership, query_assignment, source_valid, source_measure = _fixture()
    permutation = torch.tensor([2, 0, 5, 4, 1, 3], dtype=torch.long)
    original = build_six_bit_support_index(
        source_membership,
        query_assignment,
        source_valid=source_valid,
        source_measure=source_measure,
    )
    permuted = build_six_bit_support_index(
        source_membership[..., permutation],
        query_assignment[..., permutation],
        source_valid=source_valid,
        source_measure=source_measure,
    )
    original_query_support = original.support_table.gather(
        1,
        original.query_masks[..., None].expand(-1, -1, original.source_count),
    )
    permuted_query_support = permuted.support_table.gather(
        1,
        permuted.query_masks[..., None].expand(-1, -1, permuted.source_count),
    )
    torch.testing.assert_close(permuted_query_support, original_query_support)
    torch.testing.assert_close(permuted.support_pairs_per_query, original.support_pairs_per_query)
    torch.testing.assert_close(permuted.logical_paths_per_query, original.logical_paths_per_query)


def test_execution_ledger_does_not_conflate_support_with_measured_rows() -> None:
    source_membership, query_assignment, source_valid, source_measure = _fixture()
    index = build_six_bit_support_index(
        source_membership,
        query_assignment,
        source_valid=source_valid,
        source_measure=source_measure,
    )
    executed = index.support_pairs_per_query.to(torch.float64) + 2.0
    padded = torch.full_like(executed, 2.0)
    recomputed = torch.ones_like(executed)
    ledger = index.execution_counts(executed, padded, recomputed_rows=recomputed)
    torch.testing.assert_close(ledger.support_pairs, index.support_pairs_per_query)
    torch.testing.assert_close(ledger.executed_rows, executed)
    torch.testing.assert_close(ledger.padded_rows, padded)
    torch.testing.assert_close(ledger.recomputed_rows, recomputed)
    assert not torch.equal(ledger.support_pairs, ledger.executed_rows)


def test_six_bit_mask_rejects_non_k6_and_negative_assignment() -> None:
    with pytest.raises(ValueError, match="exactly K=6"):
        six_bit_mask(torch.ones(1, 2, 5))
    with pytest.raises(ValueError, match="nonnegative"):
        six_bit_mask(torch.tensor([[[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]]]))
