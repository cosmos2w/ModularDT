"""Tests for the union-to-CSR exact routed executor."""

from __future__ import annotations

import copy

import pytest
import torch

from honf_forward_core.interface_fields.routed_pairwise import RoutedPairwiseField
from honf_forward_core.interface_fields.routing_index import (
    build_typed_source_incidence,
    compile_two_hop_pairs_compiled,
    pair_join,
)
from honf_forward_core.interface_fields.routing_index.types import (
    MeasureQueryProjection,
    PackedPairs,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _dense_prior(
    weights: torch.Tensor,
    membership: torch.Tensor,
    density: torch.Tensor,
) -> torch.Tensor:
    return weights[:, None, :] * torch.einsum(
        "bqk,bnk->bqn",
        density,
        membership,
    )


def _unpack_compiled(
    pairs,
    *,
    receiver_count: int,
    source_count: int,
) -> torch.Tensor:
    """Place complete and CSR rows into a dense audit buffer."""

    output = pairs.complete_prior.new_zeros(
        pairs.batch_count * receiver_count * source_count,
    )
    for position, row in enumerate(pairs.complete_row_index):
        batch = int(row) // receiver_count
        valid = pairs.source_valid[batch]
        keys = int(row) * source_count + torch.arange(
            source_count,
            device=output.device,
        )[valid]
        output.index_copy_(0, keys, pairs.complete_prior[position, valid])
    row_ptr = pairs.partial_row_ptr
    for row in range(pairs.batch_count * receiver_count):
        start, end = int(row_ptr[row]), int(row_ptr[row + 1])
        if start == end:
            continue
        keys = row * source_count + pairs.partial_source_index[start:end]
        output.index_copy_(0, keys, pairs.partial_prior[start:end])
    return output.reshape(pairs.batch_count, receiver_count, source_count)


def test_compiled_union_promotes_exact_complete_rows_without_path_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing hub can be redundant when its sources also use a selected hub."""

    def fail_path_expansion(*args, **kwargs):
        del args, kwargs
        raise AssertionError("compiled union must not enumerate two-hop paths")

    monkeypatch.setattr(pair_join, "_join_path_indices", fail_path_expansion)
    monkeypatch.setattr(pair_join, "_join_paths", fail_path_expansion)
    weights = torch.tensor([[0.25, 0.35, 0.40]], dtype=torch.float64)
    membership = torch.tensor(
        [[[0.5, 0.5], [1.0, 0.0], [0.5, 0.5]]],
        dtype=torch.float64,
    )
    density = torch.tensor(
        [[[1.0, 0.0], [0.5, 0.5]]],
        dtype=torch.float64,
    )
    incidence = build_typed_source_incidence(weights, membership)
    pairs = compile_two_hop_pairs_compiled(
        density,
        incidence,
        support_tile_size=2,
        scalar_tile_size=2,
    )
    expected = _dense_prior(weights, membership, density)
    assert pairs.complete_rows.tolist() == [[True, True]]
    assert pairs.partial_pair_count == 0
    assert pairs.complete_row_index.tolist() == [0, 1]
    actual = _unpack_compiled(pairs, receiver_count=2, source_count=3)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-14)


def test_compiled_partial_csr_has_exact_union_and_no_duplicate_sources() -> None:
    weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    membership = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]],
        dtype=torch.float64,
    )
    density = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float64,
    )
    incidence = build_typed_source_incidence(weights, membership)
    pairs = compile_two_hop_pairs_compiled(
        density,
        incidence,
        support_tile_size=2,
        scalar_tile_size=3,
    )
    # Source 2 belongs to both hubs, so row 0 and row 1 each retain it; the
    # source union remains row-owned and no source is repeated in a row.
    assert pairs.complete_rows.tolist() == [[False, False]]
    assert pairs.partial_row_ptr.tolist() == [0, 2, 4]
    assert pairs.partial_source_index.tolist() == [0, 2, 1, 2]
    actual = _unpack_compiled(pairs, receiver_count=2, source_count=3)
    expected = _dense_prior(weights, membership, density)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-14)
    assert pairs.raw_path_count == 4
    assert pairs.unique_pair_count == 4


def test_compiled_multiword_union_keeps_high_hub_support_exact() -> None:
    """Support words beyond bit 62 still compile directly to CSR rows."""

    weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    membership = torch.zeros(1, 3, 130, dtype=torch.float64)
    membership[0, 0, 0] = 0.4
    membership[0, 0, 70] = 0.6
    membership[0, 1, 70] = 1.0
    membership[0, 2, 129] = 1.0
    density = torch.zeros(1, 2, 130, dtype=torch.float64)
    density[0, 0, 70] = 1.0
    density[0, 1, 0] = 0.5
    density[0, 1, 129] = 0.5
    incidence = build_typed_source_incidence(weights, membership)
    pairs = compile_two_hop_pairs_compiled(
        density,
        incidence,
        support_tile_size=5,
        scalar_tile_size=130,
    )
    assert pairs.complete_rows.tolist() == [[False, False]]
    assert pairs.partial_row_ptr.tolist() == [0, 2, 4]
    assert pairs.partial_source_index.tolist() == [0, 1, 0, 2]
    actual = _unpack_compiled(pairs, receiver_count=2, source_count=3)
    expected = _dense_prior(weights, membership, density)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-14)


def test_compiled_priors_keep_live_gradients_and_mask_inactive_entries() -> None:
    weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64, requires_grad=True)
    membership = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    density = torch.tensor(
        [[[0.7, 0.0], [0.0, 0.8]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    incidence = build_typed_source_incidence(weights, membership)
    pairs = compile_two_hop_pairs_compiled(
        density,
        incidence,
        support_tile_size=2,
        scalar_tile_size=2,
    )
    probe = torch.tensor([1.0, 1.3, 1.7, 2.1], dtype=torch.float64)
    loss = (pairs.partial_prior * probe).sum()
    gradients = torch.autograd.grad(loss, (weights, membership, density))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert gradients[0][0, 0].item() != 0.0
    # Zero route coordinates never enter a selected prior graph.
    assert gradients[1][0, 0, 1].item() == 0.0
    assert gradients[1][0, 1, 0].item() == 0.0
    assert gradients[2][0, 0, 1].item() == 0.0
    assert gradients[2][0, 1, 0].item() == 0.0


def test_compiled_underflow_is_reported_instead_of_dropped() -> None:
    weights = torch.tensor([[1.0]], dtype=torch.float64)
    membership = torch.tensor([[[1.0e-200]]], dtype=torch.float64)
    density = torch.tensor([[[1.0e-200]]], dtype=torch.float64)
    incidence = build_typed_source_incidence(weights, membership)
    with pytest.raises(FloatingPointError, match="nonpositive or nonfinite"):
        compile_two_hop_pairs_compiled(
            density,
            incidence,
            support_tile_size=2,
            scalar_tile_size=2,
        )


def test_single_candidate_shortcut_keeps_live_non_normalized_prior_and_coordinate_gradients() -> None:
    """The singleton shortcut must retain the exact ``omega*A*d`` graph."""

    weights = torch.tensor(
        [[0.2, 0.7, 0.0], [2.0, 3.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    membership = torch.tensor(
        [[[0.4], [0.6], [0.0]], [[0.75], [0.25], [0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    coordinates = torch.tensor(
        [[[0.4], [0.8]], [[1.2], [1.6]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    density = coordinates.square() + 0.5
    incidence = build_typed_source_incidence(weights, membership)
    projection = MeasureQueryProjection(
        density=density,
        probability=torch.zeros_like(density),
        threshold=torch.zeros(density.shape[:2], dtype=density.dtype),
        support=torch.ones_like(density, dtype=torch.bool),
    )
    pairs = RoutedPairwiseField._single_candidate_pairs(projection, incidence)
    assert pairs is not None
    expected = (
        weights[:, None, :]
        * membership[..., 0][:, None, :]
        * density[..., 0][:, :, None]
    ).reshape(-1, weights.shape[1])
    torch.testing.assert_close(
        pairs.complete_prior,
        expected.index_select(0, pairs.complete_row_index),
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    assert pairs.complete_row_index.tolist() == [0, 1, 2, 3]
    assert pairs.unique_pair_count == 8

    gradients = torch.autograd.grad(pairs.complete_prior.sum(), (weights, membership, coordinates))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert gradients[0][0, 0].item() != 0.0
    assert gradients[0][1, 1].item() != 0.0
    assert gradients[1][0, 0, 0].item() != 0.0
    assert gradients[1][1, 1, 0].item() != 0.0
    assert gradients[2].abs().sum().item() > 0.0
    assert torch.equal(gradients[0][:, 2], torch.zeros_like(gradients[0][:, 2]))
    assert torch.equal(gradients[1][:, 2], torch.zeros_like(gradients[1][:, 2]))


def _qm_fixture() -> tuple[EncodedInterfaceCase, torch.Tensor, torch.Tensor, object]:
    torch.manual_seed(91)
    batch, query_count, source_count, hidden = 2, 3, 4, 6
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn(batch, source_count, hidden, dtype=torch.float64),
        env_tokens=torch.randn(batch, 1, hidden, dtype=torch.float64),
        global_token=torch.randn(batch, hidden, dtype=torch.float64),
        module_centers=torch.randn(batch, source_count, 2, dtype=torch.float64),
        env_coords=torch.randn(batch, 1, 2, dtype=torch.float64),
        module_present=torch.ones(batch, source_count, dtype=torch.float64),
        module_features=torch.zeros(batch, source_count, 1, dtype=torch.float64),
        env_features=None,
        env_weights=torch.ones(batch, 1, dtype=torch.float64),
        coordinate_scale=torch.ones(2, dtype=torch.float64),
    )
    weights = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        dtype=torch.float64,
    )
    membership = torch.tensor(
        [
            [[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.5, 0.5]],
            [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]],
        ],
        dtype=torch.float64,
    )
    density = torch.ones(batch, query_count, 2, dtype=torch.float64)
    incidence = build_typed_source_incidence(weights, membership)
    pairs = compile_two_hop_pairs_compiled(
        density,
        incidence,
        support_tile_size=16,
        scalar_tile_size=16,
    )
    receivers = torch.randn(batch, query_count, 2, dtype=torch.float64)
    return encoded, receivers, density, pairs


def test_compiled_qm_reuses_materialized_lazy_weight_and_keeps_qm_parameters_live() -> None:
    encoded, receivers, _, compiled_pairs = _qm_fixture()
    source_count = int(encoded.module_tokens.shape[1])
    rows = compiled_pairs.complete_row_index
    _, query_count = receivers.shape[:2]
    row_batch = torch.div(rows, query_count, rounding_mode="floor")
    row_receiver = torch.remainder(rows, query_count)
    valid = compiled_pairs.source_valid[row_batch]
    pair_batch = row_batch[:, None].expand(-1, source_count)[valid].flatten()
    pair_receiver = row_receiver[:, None].expand(-1, source_count)[valid].flatten()
    pair_source = torch.arange(source_count).expand(rows.numel(), -1)[valid].flatten()
    pair_prior = compiled_pairs.complete_prior[valid]
    packed = PackedPairs(
        pair_batch,
        pair_receiver,
        pair_source,
        pair_prior,
        compiled_pairs.unique_pair_count,
        compiled_pairs.unique_pair_count,
    )
    field = RoutedPairwiseField(
        hidden_dim=6,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
        routing_config={"fine_pair_chunk_size": 3},
    ).double().eval()
    # Materialize the historical lazy first layer, then mimic a checkpoint
    # whose metadata retained in_features=0 while its weight is concrete.
    field.read_module_pairs(
        {"module_tokens": encoded.module_tokens},
        encoded,
        receivers,
        packed,
    )
    reference = copy.deepcopy(field)
    first = field.query_module_message.net[0]
    first.in_features = 0
    module_tokens = encoded.module_tokens.detach().clone().requires_grad_()
    global_token = encoded.global_token.detach().clone().requires_grad_()
    centers = encoded.module_centers.detach().clone().requires_grad_()
    input_encoded = EncodedInterfaceCase(
        module_tokens=encoded.module_tokens,
        env_tokens=encoded.env_tokens,
        global_token=global_token,
        module_centers=centers,
        env_coords=encoded.env_coords,
        module_present=encoded.module_present,
        module_features=encoded.module_features,
        env_features=None,
        env_weights=encoded.env_weights,
        coordinate_scale=encoded.coordinate_scale,
    )
    output = field.read_module_pairs(
        {"module_tokens": module_tokens},
        input_encoded,
        receivers,
        compiled_pairs,
    )
    expected = reference.read_module_pairs(
        {"module_tokens": module_tokens},
        input_encoded,
        receivers,
        packed,
    )
    torch.testing.assert_close(output, expected, rtol=1.0e-11, atol=1.0e-12)
    loss = output.square().sum()
    parameters = [
        parameter
        for name, parameter in field.named_parameters()
        if name.startswith("query_module_message.")
    ]
    gradients = torch.autograd.grad(loss, parameters + [module_tokens, global_token, centers])
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
