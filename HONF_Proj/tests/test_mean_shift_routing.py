"""Focused tests for Goal 2 fixed-data mean-shift routing candidates."""

from __future__ import annotations

from dataclasses import replace

import torch

from honf_forward_core.interface_fields.routed_pairwise import RoutedPairwiseField
from honf_forward_core.interface_fields.routing_index.router import fixed_data_mean_shift
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _encoded(
    *,
    centers: torch.Tensor | None = None,
    module_tokens: torch.Tensor | None = None,
    module_present: torch.Tensor | None = None,
    coordinate_scale: torch.Tensor | None = None,
) -> EncodedInterfaceCase:
    """Build a small dimension-neutral encoded case without a case adapter."""

    dtype = torch.float64
    if centers is None:
        centers = torch.tensor(
            [[[0.0, 0.0], [0.4, 0.0], [8.0, 0.0], [11.0, 0.0]]],
            dtype=dtype,
        )
    if module_tokens is None:
        generator = torch.Generator().manual_seed(17)
        module_tokens = torch.randn(1, centers.shape[1], 6, generator=generator, dtype=dtype)
    if module_present is None:
        module_present = torch.ones(1, centers.shape[1], dtype=dtype)
    if coordinate_scale is None:
        coordinate_scale = torch.ones(1, 1, centers.shape[-1], dtype=dtype)
    environment_count = 3
    env_coords = torch.tensor(
        [[[0.0, 1.0], [4.0, 1.0], [10.0, 1.0]]], dtype=dtype
    )
    env_tokens = torch.tensor(
        [[[0.2, -0.1, 0.3, -0.4, 0.5, -0.6],
          [0.4, -0.2, 0.1, -0.3, 0.6, -0.5],
          [-0.3, 0.2, 0.5, 0.1, -0.4, 0.7]]],
        dtype=dtype,
    )
    assert env_tokens.shape[1] == environment_count
    return EncodedInterfaceCase(
        module_tokens=module_tokens,
        env_tokens=env_tokens,
        global_token=torch.tensor([[0.15, -0.2, 0.25, -0.1, 0.35, -0.3]], dtype=dtype),
        module_centers=centers,
        env_coords=env_coords,
        module_present=module_present,
        module_features=torch.zeros(1, centers.shape[1], 2, dtype=dtype),
        env_features=None,
        env_weights=torch.tensor([[0.5, 1.0, 1.5]], dtype=dtype),
        coordinate_scale=coordinate_scale,
    )


def _backend(*, steps: int = 3, feature_bandwidth: float = 1.0) -> RoutedPairwiseField:
    return RoutedPairwiseField(
        hidden_dim=6,
        message_hidden_dim=5,
        num_heads=1,
        fourier_frequencies=1,
        routing_config={
            "strategy": "mean_shift",
            "descriptor_dim": 5,
            "router_hidden_dim": 9,
            "source_normalizer": "sparsemax",
            "query_normalizer": "source_measure_sparsemax",
            "temperature": 1.0,
            "content_scale": 2.0,
            "geometry_scale": 0.25,
            "propensity_scale": 0.25,
            "resistance_mode": "adapter",
            "execution": "gathered",
            "fine_pair_chunk_size": 32,
            "mean_shift_steps": steps,
            "mean_shift_feature_bandwidth": feature_bandwidth,
        },
    ).double().eval()


def _candidates(
    backend: RoutedPairwiseField, encoded: EncodedInterfaceCase
):
    candidates, candidate_geometry, module_geometry = backend._build_candidates(
        encoded, encoded.module_tokens
    )
    return candidates, candidate_geometry, module_geometry


def _fixed_data_reference(
    backend: RoutedPairwiseField,
    encoded: EncodedInterfaceCase,
    steps: int,
    feature_bandwidth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equation (20) with fixed source samples b_j at every iteration."""

    module_geometry = backend._geometry(encoded, encoded.module_centers)
    source_descriptors = backend.router.module_map(
        encoded.module_tokens, module_geometry, encoded.global_token
    )
    length_scale = backend._length_scale(encoded, encoded.module_centers)
    scale = length_scale.reshape(1, 1, -1)
    source_coords = encoded.module_centers / scale
    attractor_coords = source_coords.clone()
    attractor_descriptors = source_descriptors.clone()
    for _ in range(steps):
        coordinate_delta = attractor_coords[:, :, None, :] - source_coords[:, None, :, :]
        descriptor_delta = attractor_descriptors[:, :, None, :] - source_descriptors[:, None, :, :]
        logits = -0.5 * coordinate_delta.square().sum(dim=-1)
        logits = logits - descriptor_delta.square().sum(dim=-1) / (2.0 * feature_bandwidth**2)
        probabilities = torch.softmax(logits, dim=-1)
        attractor_coords = torch.einsum("bkm,bmd->bkd", probabilities, source_coords)
        attractor_descriptors = torch.einsum(
            "bkm,bmd->bkd", probabilities, source_descriptors
        )
    return attractor_coords * scale, attractor_descriptors


def _blurring_reference(
    backend: RoutedPairwiseField,
    encoded: EncodedInterfaceCase,
    steps: int,
    feature_bandwidth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The moving-sample/blurring update, kept as a negative reference."""

    module_geometry = backend._geometry(encoded, encoded.module_centers)
    descriptors = backend.router.module_map(
        encoded.module_tokens, module_geometry, encoded.global_token
    )
    length_scale = backend._length_scale(encoded, encoded.module_centers)
    scale = length_scale.reshape(1, 1, -1)
    moving_coords = encoded.module_centers / scale
    moving_descriptors = descriptors
    for _ in range(steps):
        coordinate_delta = moving_coords[:, :, None, :] - moving_coords[:, None, :, :]
        descriptor_delta = moving_descriptors[:, :, None, :] - moving_descriptors[:, None, :, :]
        logits = -0.5 * coordinate_delta.square().sum(dim=-1)
        logits = logits - descriptor_delta.square().sum(dim=-1) / (2.0 * feature_bandwidth**2)
        probabilities = torch.softmax(logits, dim=-1)
        moving_coords = torch.einsum("bkm,bmd->bkd", probabilities, moving_coords)
        moving_descriptors = torch.einsum("bkm,bmd->bkd", probabilities, moving_descriptors)
    return moving_coords * scale, moving_descriptors


def test_mean_shift_zero_steps_are_module_seeded() -> None:
    encoded = _encoded()
    backend = _backend()
    module_geometry = backend._geometry(encoded, encoded.module_centers)
    source_descriptors = backend.router.module_map(
        encoded.module_tokens, module_geometry, encoded.global_token
    )
    length_scale = backend._length_scale(encoded, encoded.module_centers)
    zero_step_state, trajectory = fixed_data_mean_shift(
        encoded.module_centers,
        source_descriptors,
        length_scale,
        source_valid=encoded.module_present > 0.5,
        steps=0,
        feature_bandwidth=1.0,
        return_trajectory=True,
    )
    scale = length_scale.reshape(1, 1, -1)
    torch.testing.assert_close(zero_step_state[..., :2] * scale, encoded.module_centers)
    torch.testing.assert_close(zero_step_state[..., 2:], source_descriptors)
    assert trajectory is not None
    assert trajectory.shape == (1, 1, encoded.module_centers.shape[1], 2 + source_descriptors.shape[-1])
    torch.testing.assert_close(trajectory[:, 0], zero_step_state)


def test_mean_shift_source_permutation_equivariance() -> None:
    encoded = _encoded()
    backend = _backend(steps=3)
    candidates, _, _ = _candidates(backend, encoded)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = replace(
        encoded,
        module_tokens=encoded.module_tokens[:, permutation],
        module_centers=encoded.module_centers[:, permutation],
        module_present=encoded.module_present[:, permutation],
        module_features=encoded.module_features[:, permutation],
    )
    permuted_candidates, _, _ = _candidates(backend, permuted)
    torch.testing.assert_close(
        permuted_candidates.coords, candidates.coords[:, permutation]
    )
    torch.testing.assert_close(
        permuted_candidates.descriptors, candidates.descriptors[:, permutation]
    )
    torch.testing.assert_close(
        permuted_candidates.propensity, candidates.propensity[:, permutation]
    )


def test_mean_shift_uses_fixed_source_data_and_not_blurring_samples() -> None:
    encoded = _encoded()
    backend = _backend(feature_bandwidth=1.0)
    candidates, _, _ = _candidates(backend, encoded)
    fixed_coords, fixed_descriptors = _fixed_data_reference(backend, encoded, 3, 1.0)
    blurring_coords, blurring_descriptors = _blurring_reference(
        backend, encoded, 3, 1.0
    )
    torch.testing.assert_close(candidates.coords, fixed_coords, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(
        candidates.descriptors, fixed_descriptors, rtol=1e-11, atol=1e-12
    )
    assert torch.max((fixed_coords - blurring_coords).abs()) > 1.0e-6
    assert torch.max((fixed_descriptors - blurring_descriptors).abs()) > 1.0e-6


def test_mean_shift_multi_step_gradients_keep_live_source_coordinates_and_descriptors() -> None:
    base = _encoded()
    centers = base.module_centers.clone().requires_grad_()
    module_tokens = base.module_tokens.clone().requires_grad_()
    encoded = replace(base, module_centers=centers, module_tokens=module_tokens)
    backend = _backend(steps=3)
    candidates, _, _ = _candidates(backend, encoded)
    objective = (
        candidates.coords.square().sum()
        + 0.7 * candidates.descriptors.square().sum()
        + 0.2 * candidates.propensity.square().sum()
    )
    center_gradient, token_gradient = torch.autograd.grad(
        objective, (centers, module_tokens)
    )
    for gradient in (center_gradient, token_gradient):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_mean_shift_fixed_data_source_path_passes_float64_gradcheck() -> None:
    source_coords = torch.tensor(
        [[[0.0, -0.2], [0.7, 0.35], [2.1, -0.55]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    source_descriptors = torch.tensor(
        [[[0.1, -0.4, 0.3], [0.8, 0.2, -0.6], [-0.5, 0.9, 0.25]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    length_scale = torch.tensor([1.3, 0.85], dtype=torch.float64)
    probe = torch.tensor(
        [[[0.2, -0.7, 0.35, 0.6, -0.25],
          [0.45, 0.1, -0.55, 0.3, 0.75],
          [-0.2, 0.5, 0.65, -0.4, 0.15]]],
        dtype=torch.float64,
    )

    def objective(coords: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        shifted, _ = fixed_data_mean_shift(
            coords,
            descriptors,
            length_scale,
            source_valid=torch.ones(1, 3, dtype=torch.bool),
            steps=3,
            feature_bandwidth=0.9,
        )
        return (shifted * probe).sum()

    assert torch.autograd.gradcheck(
        objective,
        (source_coords, source_descriptors),
        eps=1.0e-6,
        atol=1.0e-5,
        rtol=1.0e-4,
    )


def test_mean_shift_separated_modes_and_broad_bandwidth_consensus_are_observable() -> None:
    separated = _encoded(
        centers=torch.tensor(
            [[[0.0, 0.0], [0.2, 0.0], [8.0, 0.0], [8.2, 0.0]]], dtype=torch.float64
        ),
        module_tokens=torch.zeros(1, 4, 6, dtype=torch.float64),
    )
    separated_candidates, _, _ = _candidates(_backend(steps=3), separated)
    separated_coords = separated_candidates.coords[0, :, 0]
    assert torch.abs(separated_coords[1] - separated_coords[0]) < 0.25
    assert torch.abs(separated_coords[3] - separated_coords[2]) < 0.25
    assert torch.abs(separated_coords[:2].mean() - separated_coords[2:].mean()) > 3.0

    consensus = _encoded(
        centers=torch.tensor(
            [[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0]]], dtype=torch.float64
        ),
        module_tokens=torch.zeros(1, 4, 6, dtype=torch.float64),
    )
    consensus_candidates, _, _ = _candidates(
        _backend(steps=3, feature_bandwidth=100.0), consensus
    )
    initial_spread = torch.pdist(consensus.module_centers[0]).max()
    final_spread = torch.pdist(consensus_candidates.coords[0]).max()
    assert final_spread < initial_spread
    # This fixture records the broad-bandwidth consensus behavior; the model
    # must not be tested under an assumption that every run yields many modes.
    assert final_spread < 0.5 * initial_spread


def test_mean_shift_padding_and_empty_module_sources_are_finite_and_unselected() -> None:
    padded = _encoded(
        centers=torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float64
        ),
        module_tokens=torch.zeros(1, 3, 6, dtype=torch.float64),
        module_present=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    backend = _backend(steps=3)
    candidates, _, _ = _candidates(backend, padded)
    assert torch.isfinite(candidates.coords).all()
    assert torch.isfinite(candidates.descriptors).all()
    state = backend.prepare(padded, padded.module_tokens)
    module_membership = state["routing_index"].module_incidence.membership
    assert torch.count_nonzero(module_membership[:, 1:]) == 0
    assert torch.isfinite(module_membership).all()

    empty = replace(padded, module_present=torch.zeros_like(padded.module_present))
    empty_state = backend.prepare(empty, empty.module_tokens)
    empty_index = empty_state["routing_index"]
    assert empty_index.candidates.valid.sum(dim=-1).tolist() == [1]
    assert empty_index.candidates.candidate_origin[0, empty_index.candidates.valid[0]].tolist() == [-1]
    assert torch.count_nonzero(empty_index.module_incidence.membership) == 0
    torch.testing.assert_close(
        empty_index.environment_incidence.hub_measure.sum(dim=-1),
        torch.ones(1, dtype=torch.float64),
    )
    assert torch.isfinite(empty_index.candidates.coords).all()


def test_prepare_keeps_original_module_descriptors_for_mean_shift_source_logits() -> None:
    encoded = _encoded()
    backend = _backend(steps=3)
    state = backend.prepare(encoded, encoded.module_tokens)
    index = state["routing_index"]
    diagnostics = index.diagnostics
    source_geometry = backend._geometry(encoded, encoded.module_centers)
    original_source_descriptors = backend.router.module_map(
        state["module_tokens"], source_geometry, encoded.global_token
    )
    candidate_descriptors = index.candidates.descriptors[:, : encoded.module_centers.shape[1]]

    # Mean-shift descriptors are moved candidate values.  They are valid hubs,
    # but must not replace the original per-module source descriptors used by
    # the source-to-hub logits and diagnostics.
    torch.testing.assert_close(
        diagnostics["module_descriptors"], original_source_descriptors,
        rtol=1e-11, atol=1e-12,
    )
    assert torch.max((candidate_descriptors - original_source_descriptors).abs()) > 1.0e-7
    expected_logits = backend._source_logits(
        encoded,
        encoded.module_centers,
        original_source_descriptors,
        index.candidates,
        "source_module",
    )
    torch.testing.assert_close(
        diagnostics["module_logits"], expected_logits, rtol=1e-11, atol=1e-12
    )
