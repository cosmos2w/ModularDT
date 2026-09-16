"""Equivalence checks for the tiled routed-pair compiler.

The optimized compiler changes the lifetime of route scalars.  It must keep
the same support decisions, pair ordering-independent values, and derivatives
as the historical path.  These tests deliberately exercise the values which
are *not* in the positive incidence as well: a support mask is an integer
topology decision, so an inactive ``A`` or ``d`` entry must not acquire a
spurious derivative when the unique-pair prior is reconstructed.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from honf_forward_core.interface_fields.routed_pairwise import RoutedPairwiseField
from honf_forward_core.interface_fields.routing_index import (
    PackedPairs,
    build_typed_source_incidence,
    compile_two_hop_pairs_optimized,
    compile_two_hop_pairs_reference,
)
from honf_forward_core.interface_fields.routing_index.pair_join import (
    compile_two_hop_pairs_batched,
)

Compiler = Callable[..., PackedPairs]


def _canonical(
    pairs: PackedPairs,
    *,
    batch_count: int,
    receiver_count: int,
    source_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pair keys and priors in a stable, order-independent order."""

    keys = (
        (pairs.batch_index * receiver_count + pairs.receiver_index) * source_count
        + pairs.source_index
    )
    order = torch.argsort(keys, stable=True)
    del batch_count  # The count is part of the explicit key contract above.
    return keys[order], pairs.prior[order]


def _partial_fixture(
    *,
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a two-batch fixture with zero, invalid, and shared-hub entries."""

    weights = torch.tensor(
        [[0.20, 0.30, 0.00, 0.50], [0.40, 0.00, 0.60, 0.00]],
        dtype=dtype,
        requires_grad=requires_grad,
    )
    membership = torch.tensor(
        [
            [[0.70, 0.30, 0.00], [0.00, 0.40, 0.60], [0.10, 0.20, 0.70], [0.20, 0.00, 0.80]],
            [[0.50, 0.00, 0.50], [0.20, 0.80, 0.00], [0.00, 0.60, 0.40], [0.30, 0.30, 0.40]],
        ],
        dtype=dtype,
        requires_grad=requires_grad,
    )
    density = torch.tensor(
        [
            [[0.60, 0.40, 0.00], [0.00, 0.20, 0.80], [0.00, 0.00, 0.00]],
            [[0.30, 0.00, 0.70], [0.00, 0.50, 0.50], [0.00, 0.00, 0.00]],
        ],
        dtype=dtype,
        requires_grad=requires_grad,
    )
    source_valid = torch.tensor(
        [[True, True, True, False], [True, False, True, True]],
        dtype=torch.bool,
    )
    return weights, membership, density, source_valid


def _run_with_gradients(
    compiler: Compiler,
    *,
    pair_tile_size: int = 2,
) -> tuple[PackedPairs, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    weights, membership, density, source_valid = _partial_fixture(requires_grad=True)
    incidence = build_typed_source_incidence(
        weights,
        membership,
        source_valid=source_valid,
    )
    if compiler is compile_two_hop_pairs_optimized:
        pairs = compiler(
            density,
            incidence,
            pair_tile_size=pair_tile_size,
            use_checkpoint=True,
        )
    elif compiler is compile_two_hop_pairs_batched:
        pairs = compiler(
            density,
            incidence,
            scalar_tile_size=32,
            use_checkpoint=False,
        )
    else:
        pairs = compiler(density, incidence)
    probe = (
        1.0
        + pairs.batch_index.to(torch.float64) * 0.07
        + pairs.receiver_index.to(torch.float64) * 0.11
        + pairs.source_index.to(torch.float64) * 0.13
    )
    gradients = torch.autograd.grad(
        (pairs.prior * probe).sum(),
        (weights, membership, density),
        allow_unused=False,
    )
    return pairs, gradients


def test_optimized_forward_matches_dense_prior_oracle_with_partial_masks() -> None:
    weights, membership, density, source_valid = _partial_fixture()
    incidence = build_typed_source_incidence(
        weights,
        membership,
        source_valid=source_valid,
    )
    reference = compile_two_hop_pairs_reference(density, incidence)
    optimized = compile_two_hop_pairs_optimized(
        density,
        incidence,
        pair_tile_size=2,
        use_checkpoint=False,
    )

    effective_membership = torch.where(
        source_valid[..., None] & (weights > 0.0)[..., None],
        membership,
        torch.zeros_like(membership),
    )
    dense_prior = weights[:, None, :] * torch.einsum(
        "bqk,bnk->bqn",
        density,
        effective_membership,
    )
    expected_keys = torch.nonzero(dense_prior > 0.0, as_tuple=False)
    expected_key_values = (
        (expected_keys[:, 0] * density.shape[1] + expected_keys[:, 1]) * membership.shape[1]
        + expected_keys[:, 2]
    )
    expected_prior = dense_prior[dense_prior > 0.0]

    batched = compile_two_hop_pairs_batched(
        density,
        incidence,
        scalar_tile_size=32,
        use_checkpoint=False,
    )

    for candidate in (reference, optimized, batched):
        keys, prior = _canonical(
            candidate,
            batch_count=int(density.shape[0]),
            receiver_count=int(density.shape[1]),
            source_count=int(membership.shape[1]),
        )
        expected_order = torch.argsort(expected_key_values, stable=True)
        assert torch.equal(keys, expected_key_values[expected_order])
        torch.testing.assert_close(
            prior,
            expected_prior[expected_order],
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        assert candidate.raw_path_count == reference.raw_path_count
        assert candidate.unique_pair_count == reference.unique_pair_count


def test_optimized_and_reference_gradients_match_and_inactive_entries_stay_zero() -> None:
    reference, reference_gradients = _run_with_gradients(compile_two_hop_pairs_reference)
    optimized, optimized_gradients = _run_with_gradients(compile_two_hop_pairs_optimized)
    batched, batched_gradients = _run_with_gradients(compile_two_hop_pairs_batched)

    reference_keys, reference_prior = _canonical(
        reference,
        batch_count=2,
        receiver_count=3,
        source_count=4,
    )
    optimized_keys, optimized_prior = _canonical(
        optimized,
        batch_count=2,
        receiver_count=3,
        source_count=4,
    )
    batched_keys, batched_prior = _canonical(
        batched,
        batch_count=2,
        receiver_count=3,
        source_count=4,
    )
    assert torch.equal(reference_keys, optimized_keys)
    assert torch.equal(reference_keys, batched_keys)
    torch.testing.assert_close(optimized_prior, reference_prior, rtol=1.0e-12, atol=1.0e-14)
    torch.testing.assert_close(batched_prior, reference_prior, rtol=1.0e-12, atol=1.0e-14)
    for expected, actual in zip(reference_gradients, optimized_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1.0e-11, atol=1.0e-13)
    for expected, actual in zip(reference_gradients, batched_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1.0e-11, atol=1.0e-13)

    weights, membership, density, source_valid = _partial_fixture()
    source_route = source_valid[..., None] & (weights > 0.0)[..., None] & (membership > 0.0)
    query_route = density > 0.0
    # A zero/invalid support coordinate must not be differentiated merely
    # because it shares a source or receiver row with a selected pair.
    for gradient, mask in (
        (reference_gradients[0], weights > 0.0),
        (reference_gradients[1], source_route),
        (reference_gradients[2], query_route),
        (optimized_gradients[0], weights > 0.0),
        (optimized_gradients[1], source_route),
        (optimized_gradients[2], query_route),
        (batched_gradients[0], weights > 0.0),
        (batched_gradients[1], source_route),
        (batched_gradients[2], query_route),
    ):
        assert torch.equal(gradient[~mask], torch.zeros_like(gradient[~mask]))


def test_optimized_empty_query_and_empty_source_masks_return_empty_pairs() -> None:
    weights, membership, density, source_valid = _partial_fixture()
    empty_density = torch.zeros_like(density)
    incidence = build_typed_source_incidence(
        weights,
        membership,
        source_valid=source_valid,
    )
    for compiler in (
        compile_two_hop_pairs_reference,
        compile_two_hop_pairs_optimized,
        compile_two_hop_pairs_batched,
    ):
        if compiler is compile_two_hop_pairs_optimized:
            query_pairs = compiler(empty_density, incidence, pair_tile_size=2, use_checkpoint=False)
        elif compiler is compile_two_hop_pairs_batched:
            query_pairs = compiler(empty_density, incidence, scalar_tile_size=32, use_checkpoint=False)
        else:
            query_pairs = compiler(empty_density, incidence)
        assert query_pairs.raw_path_count == 0
        assert query_pairs.unique_pair_count == 0
        assert query_pairs.prior.numel() == 0

    no_source_incidence = build_typed_source_incidence(
        torch.zeros_like(weights),
        membership,
        source_valid=source_valid,
    )
    for compiler in (
        compile_two_hop_pairs_reference,
        compile_two_hop_pairs_optimized,
        compile_two_hop_pairs_batched,
    ):
        if compiler is compile_two_hop_pairs_optimized:
            source_pairs = compiler(density, no_source_incidence, pair_tile_size=2, use_checkpoint=False)
        elif compiler is compile_two_hop_pairs_batched:
            source_pairs = compiler(density, no_source_incidence, scalar_tile_size=32, use_checkpoint=False)
        else:
            source_pairs = compiler(density, no_source_incidence)
        assert source_pairs.raw_path_count == 0
        assert source_pairs.unique_pair_count == 0
        assert source_pairs.prior.numel() == 0


def test_optimized_tiny_positive_fp64_prior_and_gradients_remain_finite() -> None:
    weights = torch.tensor([[0.75]], dtype=torch.float64, requires_grad=True)
    membership = torch.tensor([[[1.0e-200]]], dtype=torch.float64, requires_grad=True)
    density = torch.tensor([[[1.0e-100]]], dtype=torch.float64, requires_grad=True)
    reference = compile_two_hop_pairs_reference(
        density,
        build_typed_source_incidence(weights, membership),
    )
    optimized = compile_two_hop_pairs_optimized(
        density,
        build_typed_source_incidence(weights, membership),
        pair_tile_size=1,
        use_checkpoint=True,
    )
    batched = compile_two_hop_pairs_batched(
        density,
        build_typed_source_incidence(weights, membership),
        scalar_tile_size=2,
        use_checkpoint=False,
    )
    expected = torch.tensor([7.5e-301], dtype=torch.float64)
    torch.testing.assert_close(reference.prior, expected, rtol=1.0e-12, atol=0.0)
    torch.testing.assert_close(optimized.prior, expected, rtol=1.0e-12, atol=0.0)
    torch.testing.assert_close(batched.prior, expected, rtol=1.0e-12, atol=0.0)
    assert torch.isfinite(optimized.prior).all()
    loss = optimized.prior.sum()
    grad_weights, grad_membership, grad_density = torch.autograd.grad(
        loss,
        (weights, membership, density),
    )
    assert torch.isfinite(grad_weights).all()
    assert torch.isfinite(grad_membership).all()
    assert torch.isfinite(grad_density).all()
    assert grad_weights.item() > 0.0
    assert grad_membership.item() > 0.0
    assert grad_density.item() > 0.0

    batched_loss = batched.prior.sum()
    batched_gradients = torch.autograd.grad(batched_loss, (weights, membership, density))
    for gradient in batched_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.item() > 0.0


def test_optimized_source_and_query_permutations_preserve_canonical_pairs() -> None:
    weights, membership, density, source_valid = _partial_fixture()
    original = compile_two_hop_pairs_optimized(
        density,
        build_typed_source_incidence(weights, membership, source_valid=source_valid),
        pair_tile_size=2,
        use_checkpoint=False,
    )
    source_permutation = torch.tensor([3, 0, 2, 1])
    query_permutation = torch.tensor([2, 0, 1])
    permuted = compile_two_hop_pairs_optimized(
        density[:, query_permutation],
        build_typed_source_incidence(
            weights[:, source_permutation],
            membership[:, source_permutation],
            source_valid=source_valid[:, source_permutation],
        ),
        pair_tile_size=2,
        use_checkpoint=False,
    )
    remapped_source = source_permutation[permuted.source_index]
    remapped_receiver = query_permutation[permuted.receiver_index]
    original_keys = (
        (original.batch_index * 3 + original.receiver_index) * 4 + original.source_index
    )
    permuted_keys = (
        (permuted.batch_index * 3 + remapped_receiver) * 4 + remapped_source
    )
    original_order = torch.argsort(original_keys, stable=True)
    permuted_order = torch.argsort(permuted_keys, stable=True)
    assert torch.equal(original_keys[original_order], permuted_keys[permuted_order])
    torch.testing.assert_close(
        original.prior[original_order],
        permuted.prior[permuted_order],
        rtol=1.0e-12,
        atol=1.0e-14,
    )


@pytest.mark.parametrize("pair_tile_size", [1, 2, 64])
def test_optimized_tile_size_does_not_change_values(pair_tile_size: int) -> None:
    weights, membership, density, source_valid = _partial_fixture()
    incidence = build_typed_source_incidence(
        weights,
        membership,
        source_valid=source_valid,
    )
    baseline = compile_two_hop_pairs_optimized(
        density,
        incidence,
        pair_tile_size=pair_tile_size,
        use_checkpoint=False,
    )
    keys, prior = _canonical(baseline, batch_count=2, receiver_count=3, source_count=4)
    assert keys.numel() == baseline.unique_pair_count
    assert torch.isfinite(prior).all()


@pytest.mark.parametrize("scalar_tile_size", [2, 8, 32])
def test_batched_scalar_tiles_match_masked_duplicate_path_union(scalar_tile_size: int) -> None:
    """Batched FP64 blocks preserve masked supports and duplicate coalescing."""

    weights, membership, density, source_valid = _partial_fixture()
    incidence = build_typed_source_incidence(
        weights,
        membership,
        source_valid=source_valid,
    )
    reference = compile_two_hop_pairs_reference(density, incidence)
    batched = compile_two_hop_pairs_batched(
        density,
        incidence,
        scalar_tile_size=scalar_tile_size,
        use_checkpoint=False,
    )
    reference_keys, reference_prior = _canonical(
        reference,
        batch_count=2,
        receiver_count=3,
        source_count=4,
    )
    batched_keys, batched_prior = _canonical(
        batched,
        batch_count=2,
        receiver_count=3,
        source_count=4,
    )
    assert torch.equal(batched_keys, reference_keys)
    torch.testing.assert_close(batched_prior, reference_prior, rtol=1.0e-12, atol=1.0e-14)
    assert batched.raw_path_count == reference.raw_path_count
    assert batched.unique_pair_count == reference.unique_pair_count
    assert batched.duplicate_expansion > 1.0


def _full_support_reader_fixture() -> tuple[RoutedPairwiseField, dict[str, torch.Tensor], SimpleNamespace, torch.Tensor, PackedPairs]:
    """Create a small projected QE state with complete positive support."""

    torch.manual_seed(19)
    field = RoutedPairwiseField(
        hidden_dim=4,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
        routing_config={"strategy": "module_hubs"},
        activation_checkpointing=False,
    ).double()
    batch, query_count, source_count, head_dim = 2, 3, 4, 2
    receivers = torch.randn(batch, query_count, 2, dtype=torch.float64, requires_grad=True)
    coordinates = torch.randn(batch, source_count, 2, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([1.2, 0.8], dtype=torch.float64, requires_grad=True)
    key = torch.randn(batch, 2, source_count, head_dim, dtype=torch.float64, requires_grad=True)
    value = torch.randn(batch, 2, source_count, head_dim, dtype=torch.float64, requires_grad=True)
    prior = torch.rand(batch, query_count, source_count, dtype=torch.float64).add_(0.1).requires_grad_()
    pair_batch = torch.arange(batch).repeat_interleave(query_count * source_count)
    pair_receiver = torch.arange(query_count).repeat_interleave(source_count).repeat(batch)
    pair_source = torch.arange(source_count).repeat(query_count * batch)
    pairs = PackedPairs(
        pair_batch,
        pair_receiver,
        pair_source,
        prior.reshape(-1),
        raw_path_count=batch * query_count * source_count,
        unique_pair_count=batch * query_count * source_count,
    )
    encoded = SimpleNamespace(env_coords=coordinates, coordinate_scale=scale)
    state = {"env_keys": key, "env_values": value}
    return field, state, encoded, receivers, pairs


def _reader_output_and_gradients(
    field: RoutedPairwiseField,
    state: dict[str, torch.Tensor],
    encoded: SimpleNamespace,
    receivers: torch.Tensor,
    pairs: PackedPairs,
    receiver_features: torch.Tensor,
    *,
    dense: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    field.dense_environment_fast_path = dense
    output = field.read_environment_pairs(
        state,
        encoded,
        receivers,
        receiver_features,
        pairs,
    )
    probe = torch.arange(output.numel(), dtype=output.dtype).reshape_as(output) / 17.0 + 0.5
    loss = (output * probe).sum()
    differentiable = {
        "receivers": receivers,
        "receiver_features": receiver_features,
        "coordinates": encoded.env_coords,
        "scale": encoded.coordinate_scale,
        "key": state["env_keys"],
        "value": state["env_values"],
        "prior": pairs.prior,
    }
    initialized_parameters = [
        (name, parameter)
        for name, parameter in field.named_parameters()
        if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
    ]
    names = tuple(differentiable) + tuple(name for name, _ in initialized_parameters)
    values = tuple(differentiable.values()) + tuple(parameter for _, parameter in initialized_parameters)
    gradients = dict(zip(names, torch.autograd.grad(loss, values, allow_unused=True), strict=True))
    return output.detach(), gradients


def test_complete_support_qe_reader_matches_packed_output_and_gradients() -> None:
    field, state, encoded, receivers, pairs = _full_support_reader_fixture()
    # Initialize all lazy reader layers before cloning so both variants share
    # exactly the same parameter tensors and inferred input widths.
    field.eval()
    field.dense_environment_fast_path = False
    warmup_features = torch.randn(2, 3, 4, dtype=torch.float64)
    field.read_environment_pairs(state, encoded, receivers, warmup_features, pairs)
    dense_field = copy.deepcopy(field)
    reference_state = {key: value.detach().clone().requires_grad_() for key, value in state.items()}
    dense_state = {key: value.detach().clone().requires_grad_() for key, value in state.items()}
    reference_encoded = SimpleNamespace(
        env_coords=encoded.env_coords.detach().clone().requires_grad_(),
        coordinate_scale=encoded.coordinate_scale.detach().clone().requires_grad_(),
    )
    dense_encoded = SimpleNamespace(
        env_coords=encoded.env_coords.detach().clone().requires_grad_(),
        coordinate_scale=encoded.coordinate_scale.detach().clone().requires_grad_(),
    )
    reference_receivers = receivers.detach().clone().requires_grad_()
    dense_receivers = receivers.detach().clone().requires_grad_()
    receiver_features = torch.randn(2, 3, 4, dtype=torch.float64)
    reference_pairs = PackedPairs(
        pairs.batch_index,
        pairs.receiver_index,
        pairs.source_index,
        pairs.prior.detach().clone().requires_grad_(),
        pairs.raw_path_count,
        pairs.unique_pair_count,
    )
    dense_pairs = PackedPairs(
        pairs.batch_index,
        pairs.receiver_index,
        pairs.source_index,
        pairs.prior.detach().clone().requires_grad_(),
        pairs.raw_path_count,
        pairs.unique_pair_count,
    )
    reference_output, reference_gradients = _reader_output_and_gradients(
        field,
        reference_state,
        reference_encoded,
        reference_receivers,
        reference_pairs,
        receiver_features.detach().clone().requires_grad_(),
        dense=False,
    )
    dense_output, dense_gradients = _reader_output_and_gradients(
        dense_field,
        dense_state,
        dense_encoded,
        dense_receivers,
        dense_pairs,
        receiver_features.detach().clone().requires_grad_(),
        dense=True,
    )
    torch.testing.assert_close(dense_output, reference_output, rtol=1.0e-10, atol=1.0e-11)
    for name in reference_gradients:
        expected = reference_gradients[name]
        actual = dense_gradients[name]
        if expected is None or actual is None:
            assert expected is None and actual is None, name
        else:
            torch.testing.assert_close(actual, expected, rtol=1.0e-9, atol=1.0e-10, msg=name)


def test_complete_support_qe_reader_does_not_run_for_partial_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    field, state, encoded, receivers, pairs = _full_support_reader_fixture()
    field.eval()
    receiver_features = torch.randn(2, 3, 4, dtype=torch.float64)
    keep = ~(
        (
            (pairs.batch_index == 0)
            & (pairs.receiver_index == 2)
            & (pairs.source_index == 3)
        )
        | (
            (pairs.batch_index == 1)
            & (pairs.receiver_index == 2)
            & (pairs.source_index == 3)
        )
    )
    partial_pairs = PackedPairs(
        pairs.batch_index[keep],
        pairs.receiver_index[keep],
        pairs.source_index[keep],
        pairs.prior[keep],
        pairs.raw_path_count - 2,
        pairs.unique_pair_count - 2,
    )
    calls = {"complete": 0}

    def fail_if_called(*args: object, **kwargs: object) -> torch.Tensor:
        calls["complete"] += 1
        del args, kwargs
        raise AssertionError("complete-support QE callback ran for a partial pair set")

    monkeypatch.setattr(field, "_environment_complete_tile", fail_if_called)
    field.dense_environment_fast_path = True
    field.read_environment_pairs(state, encoded, receivers, receiver_features, partial_pairs)
    assert calls["complete"] == 0


def test_hybrid_complete_and_partial_batches_match_packed_output_and_gradients() -> None:
    field, state, encoded, receivers, pairs = _full_support_reader_fixture()
    field.eval()
    warmup_features = torch.randn(2, 3, 4, dtype=torch.float64)
    field.read_environment_pairs(state, encoded, receivers, warmup_features, pairs)
    optimized_field = copy.deepcopy(field)
    keep = ~(
        (pairs.batch_index == 1)
        & (pairs.receiver_index == 2)
        & (pairs.source_index == 3)
    )
    mixed_pairs = PackedPairs(
        pairs.batch_index[keep],
        pairs.receiver_index[keep],
        pairs.source_index[keep],
        pairs.prior[keep],
        pairs.raw_path_count - 1,
        pairs.unique_pair_count - 1,
    )
    reference_state = {key: value.detach().clone().requires_grad_() for key, value in state.items()}
    optimized_state = {key: value.detach().clone().requires_grad_() for key, value in state.items()}
    reference_encoded = SimpleNamespace(
        env_coords=encoded.env_coords.detach().clone().requires_grad_(),
        coordinate_scale=encoded.coordinate_scale.detach().clone().requires_grad_(),
    )
    optimized_encoded = SimpleNamespace(
        env_coords=encoded.env_coords.detach().clone().requires_grad_(),
        coordinate_scale=encoded.coordinate_scale.detach().clone().requires_grad_(),
    )
    reference_receivers = receivers.detach().clone().requires_grad_()
    optimized_receivers = receivers.detach().clone().requires_grad_()
    reference_pairs = PackedPairs(
        mixed_pairs.batch_index,
        mixed_pairs.receiver_index,
        mixed_pairs.source_index,
        mixed_pairs.prior.detach().clone().requires_grad_(),
        mixed_pairs.raw_path_count,
        mixed_pairs.unique_pair_count,
    )
    optimized_pairs = PackedPairs(
        mixed_pairs.batch_index,
        mixed_pairs.receiver_index,
        mixed_pairs.source_index,
        mixed_pairs.prior.detach().clone().requires_grad_(),
        mixed_pairs.raw_path_count,
        mixed_pairs.unique_pair_count,
    )
    receiver_features = torch.randn(2, 3, 4, dtype=torch.float64)
    reference_output, reference_gradients = _reader_output_and_gradients(
        field,
        reference_state,
        reference_encoded,
        reference_receivers,
        reference_pairs,
        receiver_features.detach().clone().requires_grad_(),
        dense=False,
    )
    optimized_output, optimized_gradients = _reader_output_and_gradients(
        optimized_field,
        optimized_state,
        optimized_encoded,
        optimized_receivers,
        optimized_pairs,
        receiver_features.detach().clone().requires_grad_(),
        dense=True,
    )
    torch.testing.assert_close(optimized_output, reference_output, rtol=1.0e-10, atol=1.0e-11)
    for name in reference_gradients:
        expected = reference_gradients[name]
        actual = optimized_gradients[name]
        if expected is None or actual is None:
            assert expected is None and actual is None, name
        else:
            torch.testing.assert_close(actual, expected, rtol=1.0e-9, atol=1.0e-10, msg=name)


def test_hybrid_reader_handles_broadcast_coordinates_and_an_empty_partial_batch() -> None:
    field, state, encoded, receivers, pairs = _full_support_reader_fixture()
    field.eval()
    warmup_features = torch.randn(2, 3, 4, dtype=torch.float64)
    field.read_environment_pairs(state, encoded, receivers, warmup_features, pairs)
    optimized_field = copy.deepcopy(field)
    broadcast_coordinates = encoded.env_coords[:1].detach().clone().requires_grad_()
    keep = pairs.batch_index == 0
    empty_partial_pairs = PackedPairs(
        pairs.batch_index[keep],
        pairs.receiver_index[keep],
        pairs.source_index[keep],
        pairs.prior[keep],
        pairs.raw_path_count // 2,
        pairs.unique_pair_count // 2,
    )
    reference_state = {key: value.detach().clone().requires_grad_() for key, value in state.items()}
    optimized_state = {key: value.detach().clone().requires_grad_() for key, value in state.items()}
    reference_encoded = SimpleNamespace(
        env_coords=broadcast_coordinates.detach().clone().requires_grad_(),
        coordinate_scale=encoded.coordinate_scale.detach().clone().requires_grad_(),
    )
    optimized_encoded = SimpleNamespace(
        env_coords=broadcast_coordinates.detach().clone().requires_grad_(),
        coordinate_scale=encoded.coordinate_scale.detach().clone().requires_grad_(),
    )
    reference_receivers = receivers.detach().clone().requires_grad_()
    optimized_receivers = receivers.detach().clone().requires_grad_()
    reference_pairs = PackedPairs(
        empty_partial_pairs.batch_index,
        empty_partial_pairs.receiver_index,
        empty_partial_pairs.source_index,
        empty_partial_pairs.prior.detach().clone().requires_grad_(),
        empty_partial_pairs.raw_path_count,
        empty_partial_pairs.unique_pair_count,
    )
    optimized_pairs = PackedPairs(
        empty_partial_pairs.batch_index,
        empty_partial_pairs.receiver_index,
        empty_partial_pairs.source_index,
        empty_partial_pairs.prior.detach().clone().requires_grad_(),
        empty_partial_pairs.raw_path_count,
        empty_partial_pairs.unique_pair_count,
    )
    receiver_features = torch.randn(2, 3, 4, dtype=torch.float64)
    reference_output, reference_gradients = _reader_output_and_gradients(
        field,
        reference_state,
        reference_encoded,
        reference_receivers,
        reference_pairs,
        receiver_features.detach().clone().requires_grad_(),
        dense=False,
    )
    optimized_output, optimized_gradients = _reader_output_and_gradients(
        optimized_field,
        optimized_state,
        optimized_encoded,
        optimized_receivers,
        optimized_pairs,
        receiver_features.detach().clone().requires_grad_(),
        dense=True,
    )
    torch.testing.assert_close(optimized_output, reference_output, rtol=1.0e-10, atol=1.0e-11)
    for name in reference_gradients:
        expected = reference_gradients[name]
        actual = optimized_gradients[name]
        if expected is None or actual is None:
            assert expected is None and actual is None, name
        else:
            torch.testing.assert_close(actual, expected, rtol=1.0e-9, atol=1.0e-10, msg=name)
