"""Focused checks for the bounded regional-response study harness."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "diagnostics"))

from run_regional_response_study import (
    _family_for_architecture,
    _phase_variant_context,
    _regular_environment,
    _regular_grid_shape,
    _timing_geometry_organization,
    backend_read_zero,
    build_region_partition,
    capture_environment_contexts,
    count_backend_operations,
    dense_projection_cache,
    frozen_dense_read,
    frozen_projected_attention,
)

from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
from honf_forward_core.interface_fields.regional_response import pool_region_weighted
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


class _FakeRegionalBackend:
    def read(self, *_args, **_kwargs):
        return torch.full((1, 2, 3), 9.0), {"route": "combined"}

    def read_module(self, *_args, **_kwargs):
        return torch.full((1, 2, 3), 2.0)

    def read_regional(self, *_args, **_kwargs):
        return torch.full((1, 2, 3), 4.0), torch.ones((1, 1, 2, 2))


class _FakeLatentBackend:
    latent_count = 16

    def read(self, *_args, **_kwargs):
        return torch.full((1, 2, 3), 7.0), {"route": "latent"}


class _CounterAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 2
        self.key = torch.nn.Linear(1, 1)
        self.value = torch.nn.Linear(1, 1)


class _FakeHierarchicalCounterBackend(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.env_attention = _CounterAttention()
        self._read_calls = 0

    def _segmented_read(self, state, encoded, receivers, receiver_features, incidence, **_kwargs):
        del state, encoded, receiver_features
        return receivers.new_zeros(receivers.shape[0], receivers.shape[1], 1), None

    def read(self, state, encoded, receivers, receiver_features, **kwargs):
        self._read_calls += 1
        incidence = SimpleNamespace(
            eta=receivers.new_tensor([0.25, 1.0, 0.5]),
        )
        value, _ = self._segmented_read(
            state,
            encoded,
            receivers,
            receiver_features,
            incidence,
            **kwargs,
        )
        return value, {
            "hierarchical_traversal_rows": receivers.new_tensor(4.0),
            "hierarchical_incidence_rows": receivers.new_tensor(3.0),
        }


def _fake_model() -> SimpleNamespace:
    backend = _FakeRegionalBackend()
    core = SimpleNamespace(backend=backend, _interface_read_role="p2_field")
    config = SimpleNamespace(
        core_honf=SimpleNamespace(forward_architecture="regional_response_honf")
    )
    return SimpleNamespace(core=core, config=config)


def _fake_latent_model() -> SimpleNamespace:
    backend = _FakeLatentBackend()
    core = SimpleNamespace(backend=backend, _interface_read_role="p2_field")
    config = SimpleNamespace(
        core_honf=SimpleNamespace(forward_architecture="geometry_latent_field")
    )
    return SimpleNamespace(core=core, config=config)


def _fake_hierarchical_counter_model() -> SimpleNamespace:
    backend = _FakeHierarchicalCounterBackend()
    core = SimpleNamespace(backend=backend)
    config = SimpleNamespace(
        core_honf=SimpleNamespace(forward_architecture="hierarchical_regional_honf")
    )
    return SimpleNamespace(core=core, config=config)


def test_regional_removal_keeps_direct_module_read() -> None:
    model = _fake_model()
    with backend_read_zero(model, roles=("p2_field",), component="regional"):
        value, aux = model.core.backend.read(None, None, None, None)
    torch.testing.assert_close(value, torch.full((1, 2, 3), 2.0))
    assert aux["dense_module_context_norm"].shape == (1, 2)


def test_module_removal_keeps_regional_read() -> None:
    model = _fake_model()
    with backend_read_zero(model, roles=("p2_field",), component="module"):
        value, aux = model.core.backend.read(None, None, None, None)
    torch.testing.assert_close(value, torch.full((1, 2, 3), 4.0))
    assert aux["regional_context_norm"].shape == (1, 2)


def test_regional_removal_is_role_scoped() -> None:
    model = _fake_model()
    model.core._interface_read_role = "p0_port"
    with backend_read_zero(model, roles=("p2_field",), component="regional"):
        value, aux = model.core.backend.read(None, None, None, None)
    torch.testing.assert_close(value, torch.full((1, 2, 3), 9.0))
    assert aux == {"route": "combined"}


def test_latent_phase_family_uses_shared_role_scoped_read_removal() -> None:
    model = _fake_latent_model()
    assert _family_for_architecture("geometry_latent_field") == "latent"
    with _phase_variant_context(model, "latent", "p2"):
        value, aux = model.core.backend.read(None, None, None, None)
    torch.testing.assert_close(value, torch.zeros((1, 2, 3)))
    assert aux == {"route": "latent"}

    model.core._interface_read_role = "p1_refinement"
    with _phase_variant_context(model, "latent", "p2"):
        value, _ = model.core.backend.read(None, None, None, None)
    torch.testing.assert_close(value, torch.full((1, 2, 3), 7.0))


def test_hierarchical_operation_counter_counts_transition_and_reference_rows() -> None:
    model = _fake_hierarchical_counter_model()
    backend = model.core.backend
    state = {
        "tree_level_offsets": torch.tensor([0, 1, 3]),
        "tree_levels": torch.tensor([0, 1, 1]),
        "tree_valid": torch.tensor([[True, True, True]]),
        "env_tokens": torch.zeros((1, 5, 1)),
    }
    encoded = SimpleNamespace()
    receivers_two = torch.zeros((1, 2, 1))
    features_two = torch.zeros_like(receivers_two)
    receivers_three = torch.zeros((1, 3, 1))
    features_three = torch.zeros_like(receivers_three)
    original_read = backend.read
    original_segmented_read = backend._segmented_read

    with count_backend_operations(model) as counts:
        backend.read(state, encoded, receivers_two, features_two)
        backend.read(state, encoded, receivers_three, features_three)

    assert counts["hierarchical_traversal_rows"] == 8
    assert counts["hierarchical_incidence_rows"] == 6
    assert counts["hierarchical_overlap_transition_rows"] == 4
    assert counts["hierarchical_head_dot_products"] == 12
    assert counts["reference_regional_environment_pairs"] == 10
    assert counts["reference_fine_environment_pairs"] == 25
    assert counts["reference_regional_environment_head_dot_products"] == 20
    assert counts["reference_fine_environment_head_dot_products"] == 50
    assert counts["environment_projected_key_rows"] == 0
    assert counts["environment_projected_value_rows"] == 0
    assert backend.read.__func__ is original_read.__func__
    assert backend._segmented_read.__func__ is original_segmented_read.__func__

    # Restoration is part of the counter contract: a later read must not
    # mutate the already-collected operation totals.
    backend.read(state, encoded, receivers_two, features_two)
    assert counts["hierarchical_incidence_rows"] == 6


def test_timing_geometry_summary_does_not_claim_latent_source_topology() -> None:
    summary = _timing_geometry_organization(_fake_latent_model())
    assert summary["latent_count"] == 16
    assert summary["deterministic_support"] is False
    assert "learned" in summary["topology_export"]


def test_study_partition_accepts_duplicate_coordinates_with_explicit_ids() -> None:
    coordinates = torch.tensor(
        [[[0.5, 0.5], [0.5, 0.5], [1.5, 0.5], [1.5, 0.5]]], dtype=torch.float64
    )
    weights = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    partition = build_region_partition(
        coordinates,
        weights,
        region_ids=torch.tensor([[0, 0, 1, 1]]),
    )
    torch.testing.assert_close(partition.mass, torch.tensor([[3.0, 7.0]], dtype=torch.float64))
    torch.testing.assert_close(
        partition.centroids,
        torch.tensor([[[0.5, 0.5], [1.5, 0.5]]], dtype=torch.float64),
    )


def test_regular_timing_layouts_are_complete_even_grids_with_region_ids() -> None:
    for env_count, expected_grid, expected_regions in (
        (192, (24, 8), 48),
        (768, (48, 16), 192),
        (3072, (96, 32), 768),
    ):
        assert _regular_grid_shape(env_count, 12.0, 6.0) == expected_grid
        environment = _regular_environment(
            env_count,
            12.0,
            6.0,
            1,
            torch.device("cpu"),
            torch.float64,
        )
        assert environment.env_region_ids is not None
        assert tuple(environment.env_coords.shape) == (1, env_count, 2)
        assert int(torch.unique(environment.env_region_ids[0]).numel()) == expected_regions


def test_weighted_pool_padding_has_finite_gradients_and_no_invalid_source_gradients() -> None:
    generator = torch.Generator().manual_seed(19)
    values = torch.randn(2, 5, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    coordinates = torch.randn(2, 5, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 0.5], [1.0, 0.0, 2.0, 3.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    region_ids = torch.tensor([[0, 0, 1, -1, 2], [0, 1, 1, 1, -1]])

    pooled = pool_region_weighted(values, region_ids, weights, coordinates)
    loss = pooled.values.square().sum() + pooled.mass.sum() + pooled.centroids.square().sum()
    loss.backward()

    assert pooled.valid.tolist() == [[True, True, True], [True, True, False]]
    for gradient in (values.grad, weights.grad, coordinates.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
    torch.testing.assert_close(values.grad[0, 3], torch.zeros_like(values.grad[0, 3]))
    torch.testing.assert_close(values.grad[1, 4], torch.zeros_like(values.grad[1, 4]))
    torch.testing.assert_close(weights.grad[0, 3], torch.zeros_like(weights.grad[0, 3]))
    torch.testing.assert_close(weights.grad[1, 4], torch.zeros_like(weights.grad[1, 4]))


def _manual_projected_read(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(query, key.transpose(-1, -2)) / (key.shape[-1] ** 0.5)
    attention = torch.softmax(scores + torch.log(weights)[:, None, None, :], dim=-1)
    return torch.matmul(attention, value), attention


def test_frozen_projected_attention_singleton_regions_match_fine_read() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(1, 2, 3, 4, generator=generator, dtype=torch.float64)
    key = torch.randn(1, 2, 5, 4, generator=generator, dtype=torch.float64)
    value = torch.randn(1, 2, 5, 4, generator=generator, dtype=torch.float64)
    coordinates = torch.arange(10, dtype=torch.float64).reshape(1, 5, 2)
    weights = torch.tensor([[0.25, 1.5, 2.0, 0.75, 3.0]], dtype=torch.float64)
    partition = build_region_partition(
        coordinates,
        weights,
        region_ids=torch.arange(5),
    )

    frozen, frozen_attention = frozen_projected_attention(query, key, value, partition)
    fine, fine_attention = _manual_projected_read(query, key, value, weights)
    torch.testing.assert_close(frozen, fine)
    torch.testing.assert_close(frozen_attention, fine_attention)


def test_frozen_projected_attention_preserves_unequal_region_mass_for_constant_sources() -> None:
    query = torch.tensor([[[[0.3, -0.2]], [[-0.4, 0.5]]]], dtype=torch.float64)
    key = torch.tensor(
        [[[[0.2, 0.4], [0.2, 0.4], [-0.7, 0.1], [-0.7, 0.1]],
          [[0.2, 0.4], [0.2, 0.4], [-0.7, 0.1], [-0.7, 0.1]]]],
        dtype=torch.float64,
    )
    value = torch.tensor(
        [[[[1.0, 2.0], [1.0, 2.0], [8.0, -3.0], [8.0, -3.0]],
          [[-4.0, 5.0], [-4.0, 5.0], [2.5, 6.0], [2.5, 6.0]]]],
        dtype=torch.float64,
    )
    coordinates = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
        dtype=torch.float64,
    )
    weights = torch.tensor([[1.0, 3.0, 2.0, 4.0]], dtype=torch.float64)
    partition = build_region_partition(
        coordinates,
        weights,
        region_ids=torch.tensor([[9, 9, 13, 13]]),
    )

    frozen, frozen_attention = frozen_projected_attention(query, key, value, partition)
    fine, fine_attention = _manual_projected_read(query, key, value, weights)
    torch.testing.assert_close(frozen, fine)
    fine_region_attention = torch.cat(
        (
            fine_attention[..., :2].sum(dim=-1, keepdim=True),
            fine_attention[..., 2:].sum(dim=-1, keepdim=True),
        ),
        dim=-1,
    )
    torch.testing.assert_close(frozen_attention, fine_region_attention)
    torch.testing.assert_close(partition.mass, torch.tensor([[4.0, 6.0]], dtype=torch.float64))


def _dense_study_fixture() -> tuple[SimpleNamespace, EncodedInterfaceCase, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(23)
    dtype = torch.float64
    coordinates = torch.tensor(
        [[[0.125, 0.25], [0.375, 0.25], [0.625, 0.25], [0.875, 0.25],
          [0.125, 0.75], [0.375, 0.75], [0.625, 0.75], [0.875, 0.75]]],
        dtype=dtype,
    )
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn(1, 2, 8, generator=generator, dtype=dtype),
        env_tokens=torch.randn(1, 8, 8, generator=generator, dtype=dtype),
        global_token=torch.randn(1, 8, generator=generator, dtype=dtype),
        module_centers=torch.tensor([[[0.2, 0.3], [0.8, 0.7]]], dtype=dtype),
        env_coords=coordinates,
        module_present=torch.ones(1, 2, dtype=dtype),
        module_features=torch.randn(1, 2, 3, generator=generator, dtype=dtype),
        env_features=None,
        env_weights=torch.arange(1, 9, dtype=dtype).reshape(1, 8),
        coordinate_scale=torch.ones(1, 1, 2, dtype=dtype),
    )
    backend = DensePairwiseField(8, 8, 2, 2).to(dtype=dtype).eval()
    model = SimpleNamespace(
        core=SimpleNamespace(backend=backend, _interface_read_role="p2_field"),
        config=SimpleNamespace(
            core_honf=SimpleNamespace(forward_architecture="dense_pairwise_field")
        ),
    )
    module_states = torch.randn(1, 2, 8, generator=generator, dtype=dtype)
    receivers = torch.rand(1, 3, 2, generator=generator, dtype=dtype)
    receiver_features = torch.randn(1, 3, 8, generator=generator, dtype=dtype)
    return model, encoded, module_states, receivers, receiver_features


def test_frozen_dense_read_refreshes_source_projection_for_each_prepared_identity() -> None:
    model, encoded, module_states, receivers, receiver_features = _dense_study_fixture()
    backend = model.core.backend
    first_state = backend.prepare(encoded, module_states)
    second_state = backend.prepare(encoded, module_states + 0.25)

    with frozen_dense_read(model, roles=("p2_field",)) as frozen_state:
        first, _ = backend.read(first_state, encoded, receivers, receiver_features)
        repeated, _ = backend.read(first_state, encoded, receivers, receiver_features)
        second, _ = backend.read(second_state, encoded, receivers, receiver_features)

    torch.testing.assert_close(first, repeated)
    assert not torch.allclose(first, second)
    assert frozen_state["prepared_source_count"] == 2


def test_dense_projection_cache_refreshes_source_projection_for_each_prepared_identity() -> None:
    model, encoded, module_states, receivers, receiver_features = _dense_study_fixture()
    backend = model.core.backend
    first_state = backend.prepare(encoded, module_states)
    second_state = backend.prepare(encoded, module_states + 0.25)

    with dense_projection_cache(model) as cache_state:
        first, _ = backend.read(first_state, encoded, receivers, receiver_features)
        repeated, _ = backend.read(first_state, encoded, receivers, receiver_features)
        second, _ = backend.read(second_state, encoded, receivers, receiver_features)

    torch.testing.assert_close(first, repeated)
    assert not torch.allclose(first, second)
    assert cache_state["prepared_projection_count"] == 2


def test_dense_projection_cache_fields_follow_prepared_state_across_contexts() -> None:
    model, encoded, module_states, receivers, receiver_features = _dense_study_fixture()
    backend = model.core.backend
    prepared = backend.prepare(encoded, module_states)

    with dense_projection_cache(model) as first_cache:
        backend.read(prepared, encoded, receivers, receiver_features)
    assert first_cache["prepared_projection_count"] == 1

    # A new cache context may reuse an already prepared state without a second
    # projection; a genuinely fresh preparation still receives its own field.
    with dense_projection_cache(model) as second_cache:
        backend.read(prepared, encoded, receivers, receiver_features)
    assert second_cache["prepared_projection_count"] == 0

    fresh = backend.prepare(encoded, module_states + 0.5)
    with dense_projection_cache(model) as fresh_cache:
        backend.read(fresh, encoded, receivers, receiver_features)
    assert fresh_cache["prepared_projection_count"] == 1


def test_frozen_dense_read_composes_with_capture_environment_contexts() -> None:
    model, encoded, module_states, receivers, receiver_features = _dense_study_fixture()
    backend = model.core.backend
    prepared = backend.prepare(encoded, module_states)

    # The frozen reader must be outermost so capture wraps its live custom
    # backend boundary rather than the bypassed Dense original.
    with frozen_dense_read(model, roles=("p2_field",)), capture_environment_contexts(
        model, probe_count=2
    ) as capture:
        backend.read(prepared, encoded, receivers, receiver_features)

    assert capture["records"].get("p2_field")
    assert capture["timing"]["backend_read_seconds"].get("p2_field", 0.0) > 0.0
    assert capture["timing"]["projected_environment_read_seconds"].get("p2_field", 0.0) > 0.0
