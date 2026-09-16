"""Focused algebra and differentiability checks for the routed pairwise index."""

from __future__ import annotations

import pytest
import torch

from honf_forward_core.interface_fields.routing_index import (
    RoutingDescriptorMaps,
    build_typed_source_incidence,
    compile_two_hop_pairs,
    geometry_bounds,
    geometry_features,
    geometry_length_scale,
    geometry_resistance,
    induced_hub_measure,
    ordinary_source_sparsemax,
    routing_affinity,
    segment_resistance,
    source_measure_sparsemax,
    stable_descriptor_norm,
)


class _Geometry:
    bounds = ((-1.0, 2.0), (0.0, 3.0), (-2.0, 2.0))
    length_scale = (1.0, 2.0, 4.0)

    def features(self, points: torch.Tensor) -> torch.Tensor:
        return torch.cat((points, points.square(), points[..., :1]), dim=-1)

    def resistance(self, a: torch.Tensor, b: torch.Tensor, relation_type: str) -> torch.Tensor:
        assert relation_type == "routing"
        return (a - b).square().sum(dim=-1) * 0.125


class _FieldGeometry(_Geometry):
    def __init__(self) -> None:
        self.field_calls = 0

    def resistance_field(self, points: torch.Tensor, relation_type: str) -> torch.Tensor:
        assert relation_type == "source_environment"
        self.field_calls += 1
        return 0.5 + points.square().sum(dim=-1)


def test_ordinary_sparsemax_mask_shift_and_singleton_jacobian() -> None:
    logits = torch.tensor([[2.0, 0.2, -0.5, 9.0]], dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True, True, True, False]])
    projected = ordinary_source_sparsemax(logits, valid)
    shifted = ordinary_source_sparsemax(logits + 17.0, valid)
    assert torch.allclose(projected, shifted)
    assert torch.all(projected[..., 3] == 0.0)
    assert torch.all(projected[..., :3] >= 0.0)
    assert torch.allclose(projected.sum(dim=-1), torch.ones(1, dtype=projected.dtype))

    singleton = torch.tensor([[3.0, -4.0]], dtype=torch.float64, requires_grad=True)
    singleton_output = ordinary_source_sparsemax(singleton, torch.tensor([[True, False]]))
    singleton_output.sum().backward()
    assert torch.equal(singleton_output, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    assert torch.equal(singleton.grad, torch.zeros_like(singleton))


def test_source_measure_sparsemax_kkt_dense_limit_and_shift_invariance() -> None:
    logits = torch.tensor([[[0.8, 0.2, -0.5], [1.1, 0.4, -0.2]]], dtype=torch.float64)
    measure = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)
    result = source_measure_sparsemax(logits, measure)
    assert torch.allclose(result.probability.sum(dim=-1), torch.ones(1, 2, dtype=torch.float64))
    assert torch.allclose(
        (measure[:, None, :] * result.density).sum(dim=-1),
        torch.ones(1, 2, dtype=torch.float64),
    )
    assert torch.allclose(result.probability, source_measure_sparsemax(logits + 13.0, measure).probability)

    equal_logits = torch.full((1, 2, 3), 4.0, dtype=torch.float64)
    equal_result = source_measure_sparsemax(equal_logits, measure)
    assert torch.allclose(equal_result.density, torch.ones_like(equal_result.density))
    assert torch.allclose(equal_result.probability, measure[:, None, :].expand_as(equal_result.probability))

    zero_measure = source_measure_sparsemax(
        torch.tensor([[[1.0, 0.0, -2.0]]], dtype=torch.float64),
        torch.tensor([[0.7, 0.3, 0.0]], dtype=torch.float64),
    )
    assert torch.all(zero_measure.probability[..., 2] == 0.0)
    assert torch.all(zero_measure.density[..., 2] == 0.0)


def test_source_measure_sparsemax_gradients_include_logits_and_measure() -> None:
    logits = torch.tensor([[[0.8, 0.25, -0.2]]], dtype=torch.float64, requires_grad=True)
    raw_measure = torch.tensor([[0.4, 0.35, 0.25]], dtype=torch.float64, requires_grad=True)
    result = source_measure_sparsemax(logits, raw_measure)
    loss = (result.probability.square() + 0.3 * result.density).sum()
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert raw_measure.grad is not None and torch.isfinite(raw_measure.grad).all()
    assert torch.any(logits.grad != 0.0)
    assert torch.any(raw_measure.grad != 0.0)


def test_source_measure_sparsemax_tiny_fp32_measure_stays_finite() -> None:
    # The query density for an extremely small positive hub can be large, but
    # scalar route arithmetic must not overflow before its weighted probability
    # is formed.
    logits = torch.tensor([[[0.0, -0.25]]], dtype=torch.float32, requires_grad=True)
    measure = torch.tensor([[1.0e-44, 1.0]], dtype=torch.float32, requires_grad=True)
    result = source_measure_sparsemax(logits, measure)
    assert result.density.dtype == torch.float64
    assert torch.isfinite(result.density).all()
    assert torch.isfinite(result.probability).all()
    assert torch.allclose(result.probability.sum(dim=-1), torch.ones(1, 1, dtype=torch.float64))
    result.probability.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(measure.grad).all()


def test_two_hop_pairs_coalesce_paths_and_keep_live_gradients() -> None:
    membership = torch.tensor(
        [[[0.7, 0.3, 0.0], [0.0, 0.4, 0.6], [0.3, 0.3, 0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    source_weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64, requires_grad=True)
    incidence = build_typed_source_incidence(source_weights, membership)
    density = torch.tensor(
        [[[0.55, 0.45, 0.0], [0.0, 0.25, 0.75]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    pairs = compile_two_hop_pairs(density, incidence)
    assert pairs.raw_path_count == 10
    assert pairs.unique_pair_count == 6
    dense_prior = source_weights[:, None, :] * torch.einsum("bqk,bnk->bqn", density, membership)
    expected = dense_prior[dense_prior > 0.0]
    assert torch.allclose(torch.sort(pairs.prior).values, torch.sort(expected).values)
    assert torch.equal(pairs.batch_index, torch.zeros_like(pairs.batch_index))
    assert pairs.duplicate_expansion > 1.0

    (pairs.prior.square().sum() + incidence.hub_measure.sum() * 0.1).backward()
    assert membership.grad is not None and torch.isfinite(membership.grad).all()
    assert source_weights.grad is not None and torch.isfinite(source_weights.grad).all()
    assert density.grad is not None and torch.isfinite(density.grad).all()


def test_two_hop_pair_join_is_source_permutation_equivariant() -> None:
    membership = torch.tensor([[[0.8, 0.2], [0.0, 1.0], [0.3, 0.7]]], dtype=torch.float64)
    weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    density = torch.tensor([[[0.6, 0.4], [0.2, 0.8]]], dtype=torch.float64)
    original = compile_two_hop_pairs(density, build_typed_source_incidence(weights, membership))
    permutation = torch.tensor([2, 0, 1])
    permuted = compile_two_hop_pairs(
        density,
        build_typed_source_incidence(weights[:, permutation], membership[:, permutation]),
    )
    remapped_source = permutation[permuted.source_index]
    original_keys = torch.stack((original.receiver_index, original.source_index), dim=1)
    permuted_keys = torch.stack((permuted.receiver_index, remapped_source), dim=1)
    original_order = torch.argsort(original_keys[:, 0] * 3 + original_keys[:, 1], stable=True)
    permuted_order = torch.argsort(permuted_keys[:, 0] * 3 + permuted_keys[:, 1], stable=True)
    assert torch.equal(original_keys[original_order], permuted_keys[permuted_order])
    assert torch.allclose(original.prior[original_order], permuted.prior[permuted_order])


def test_geometry_protocol_and_affinity_are_dimension_neutral() -> None:
    provider = _Geometry()
    points = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    features = geometry_features(provider, points, feature_dim=7)
    assert features.shape == (2, 4, 7)
    assert torch.equal(geometry_bounds(provider, points), points.new_tensor(provider.bounds))
    assert torch.equal(geometry_length_scale(provider, points), points.new_tensor(provider.length_scale))
    resistance = geometry_resistance(provider, points[:, :, None, :], points[:, None, :, :])
    assert resistance.shape == (2, 4, 4)
    source = stable_descriptor_norm(torch.randn(2, 4, 5, dtype=torch.float64))
    hubs = stable_descriptor_norm(torch.randn(2, 3, 5, dtype=torch.float64))
    logits = routing_affinity(
        source,
        points,
        hubs,
        points[:, :3],
        torch.zeros(2, 3, dtype=torch.float64),
        provider.length_scale,
        resistance=resistance[:, :, :3],
    )
    assert logits.shape == (2, 4, 3)
    logits.sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()


def test_segment_resistance_midpoint_quadrature_reversal_type_and_gradients() -> None:
    provider = _FieldGeometry()
    starts = torch.tensor([[0.0, 0.2, -0.1], [0.5, 0.0, 0.3]], dtype=torch.float64, requires_grad=True)
    ends = torch.tensor([[1.0, 0.8, 0.4], [-0.2, 0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    value = segment_resistance(
        provider,
        starts,
        ends,
        relation_type="source_environment",
        samples=8,
    )
    reversed_value = segment_resistance(
        provider,
        ends,
        starts,
        relation_type="source_environment",
        samples=8,
    )
    assert provider.field_calls == 2
    assert torch.allclose(value, reversed_value)
    value.sum().backward()
    assert starts.grad is not None and torch.isfinite(starts.grad).all()
    assert ends.grad is not None and torch.isfinite(ends.grad).all()


def test_split_source_measure_preserves_aggregate_two_hop_prior() -> None:
    membership = torch.tensor([[[0.7, 0.3], [0.2, 0.8]]], dtype=torch.float64)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    density = torch.tensor([[[0.4, 0.6], [0.8, 0.2]]], dtype=torch.float64)
    original = compile_two_hop_pairs(density, build_typed_source_incidence(weights, membership))
    duplicated_membership = torch.cat((membership[:, :1], membership[:, :1], membership[:, 1:]), dim=1)
    duplicated_weights = torch.cat((weights[:, :1] * 0.4, weights[:, :1] * 0.6, weights[:, 1:]), dim=1)
    duplicated = compile_two_hop_pairs(
        density,
        build_typed_source_incidence(duplicated_weights, duplicated_membership),
    )
    original_by_receiver = torch.zeros(2, dtype=torch.float64).index_add(
        0,
        original.receiver_index,
        original.prior,
    )
    duplicated_by_receiver = torch.zeros(2, dtype=torch.float64).index_add(
        0,
        duplicated.receiver_index,
        duplicated.prior,
    )
    assert torch.allclose(original_by_receiver, duplicated_by_receiver)


def test_composed_sparse_routing_has_live_float64_gradients_away_from_support_events() -> None:
    source_logits = torch.tensor(
        [[[0.9, 0.3], [0.4, 0.7], [0.8, 0.6]]], dtype=torch.float64, requires_grad=True
    )
    query_logits = torch.tensor([[[0.6, 0.4], [0.5, 0.7]]], dtype=torch.float64, requires_grad=True)
    source_weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    membership = ordinary_source_sparsemax(source_logits)
    incidence = build_typed_source_incidence(source_weights, membership)
    projection = source_measure_sparsemax(query_logits, incidence.hub_measure)
    pairs = compile_two_hop_pairs(projection.density, incidence)

    def composed(z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        memberships = ordinary_source_sparsemax(z)
        typed = build_typed_source_incidence(source_weights, memberships)
        query = source_measure_sparsemax(q, typed.hub_measure)
        compiled = compile_two_hop_pairs(query.density, typed)
        # A nonconstant probe keeps this a meaningful scalar route check while
        # remaining independent of any field-value implementation.
        probe = 0.2 * compiled.source_index.to(torch.float64) + 1.0
        return compiled.prior * probe

    assert pairs.unique_pair_count == 6
    assert torch.autograd.gradcheck(
        composed,
        (source_logits, query_logits),
        eps=1.0e-6,
        atol=1.0e-5,
        rtol=1.0e-4,
    )


def test_module_hub_router_maps_use_geometry_only_for_routing_and_are_bounded() -> None:
    torch.manual_seed(4)
    router = RoutingDescriptorMaps(source_dim=6).double()
    states = torch.randn(2, 4, 6, dtype=torch.float64, requires_grad=True)
    coords = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    global_features = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    geometry = _Geometry()
    module_geometry = geometry.features(coords)
    module_descriptors = router.module_map(states, module_geometry, global_features)
    propensity = router.hub_propensity(module_descriptors, module_geometry, global_features)
    environment_states = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
    environment_geometry = geometry.features(torch.randn(2, 3, 3, dtype=torch.float64))
    environment_descriptors = router.environment_map(environment_states, environment_geometry, global_features)
    query_states = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)
    query_geometry = geometry.features(torch.randn(2, 5, 3, dtype=torch.float64))
    query_descriptors = router.query_map(query_states, query_geometry, global_features)
    assert module_descriptors.shape == (2, 4, 32)
    assert environment_descriptors.shape == (2, 3, 32)
    assert query_descriptors.shape == (2, 5, 32)
    assert propensity.shape == (2, 4)
    assert torch.all(module_descriptors.norm(dim=-1) < 1.0)
    assert torch.all(environment_descriptors.norm(dim=-1) < 1.0)
    assert torch.all(query_descriptors.norm(dim=-1) < 1.0)
    assert torch.all(propensity.abs() <= 1.0)
    loss = module_descriptors.square().sum() + environment_descriptors.square().sum()
    loss = loss + query_descriptors.square().sum() + propensity.square().sum()
    loss.backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    assert coords.grad is not None and torch.isfinite(coords.grad).all()
    assert global_features.grad is not None and torch.isfinite(global_features.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_induced_hub_measure_preserves_unit_mass(dtype: torch.dtype) -> None:
    weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=dtype)
    membership = ordinary_source_sparsemax(
        torch.tensor([[[1.0, 0.0], [0.3, 0.4], [0.0, 2.0]]], dtype=dtype)
    )
    measure = induced_hub_measure(weights, membership)
    assert torch.allclose(measure.sum(dim=-1), torch.ones(1, dtype=measure.dtype))
