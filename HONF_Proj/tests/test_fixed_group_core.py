"""CPU-focused scientific contracts for the Run-1405 core backend."""

from __future__ import annotations

import torch
from torch import nn

from honf_forward_core.interface_fields.fixed_group_pairwise import FixedGroupPairwiseField
from honf_forward_core.interface_fields.fixed_group_router import (
    FixedGroupQueryRoute,
    FixedGroupRouter,
    FixedGroupState,
)
from honf_forward_core.interface_fields.three_term_context import ThreeTermInterfaceContext
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


class _FixedGroupScalarLogits(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        pattern = self.values.to(device=values.device, dtype=values.dtype)
        return pattern.view(1, 1, -1, 1).expand(*values.shape[:-1], 1)


class _FixedGroupQueryLogits(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        pattern = self.values.to(device=values.device, dtype=values.dtype)
        return pattern.view(1, 1, -1, 1).expand(*values.shape[:-1], 1)


class _ZeroGeometryBias(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.new_zeros(*values.shape[:-1], 1)


def _encoded(
    *,
    batch: int = 1,
    modules: int = 4,
    environments: int = 5,
    hidden: int = 8,
    spatial_dim: int = 2,
    all_modules_inactive: bool = False,
) -> EncodedInterfaceCase:
    generator = torch.Generator().manual_seed(1405)
    module_centers = torch.rand(batch, modules, spatial_dim, generator=generator)
    env_coords = torch.rand(batch, environments, spatial_dim, generator=generator)
    present = torch.ones(batch, modules)
    if all_modules_inactive:
        present.zero_()
    return EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, hidden, generator=generator),
        env_tokens=torch.randn(batch, environments, hidden, generator=generator),
        global_token=torch.randn(batch, hidden, generator=generator),
        module_centers=module_centers,
        env_coords=env_coords,
        module_present=present,
        module_features=torch.randn(batch, modules, 3, generator=generator),
        env_features=None,
        env_weights=torch.ones(batch, environments) / float(environments),
        coordinate_scale=torch.ones(1, 1, spatial_dim),
    )


def test_fixed_group_assignments_are_entmax_normalized_and_masked() -> None:
    encoded = _encoded()
    router = FixedGroupRouter(hidden_dim=8, spatial_dim=2, fourier_frequencies=2)
    module_states = torch.randn(1, 4, 8)
    environment_states = torch.randn(1, 5, 8)
    state = router.prepare(encoded, module_states, environment_states)

    assert state.module_membership.shape == (1, 4, 6)
    assert state.environment_membership.shape == (1, 5, 6)
    assert torch.all(state.module_membership >= 0)
    assert torch.all(state.environment_membership >= 0)
    torch.testing.assert_close(
        state.module_membership.sum(dim=-1),
        torch.ones(1, 4),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        state.environment_membership.sum(dim=-1),
        torch.ones(1, 5),
        rtol=0.0,
        atol=1.0e-6,
    )

    encoded_inactive = _encoded(all_modules_inactive=True)
    inactive_state = router.prepare(
        encoded_inactive,
        torch.randn(1, 4, 8),
        torch.randn(1, 5, 8),
    )
    assert torch.equal(inactive_state.module_membership, torch.zeros_like(inactive_state.module_membership))
    assert torch.isfinite(inactive_state.module_centres).all()
    assert torch.isfinite(inactive_state.environment_centres).all()


def test_empty_groups_have_zero_query_route() -> None:
    encoded = _encoded()
    router = FixedGroupRouter(hidden_dim=8, spatial_dim=2, fourier_frequencies=2)
    fallback = router._fallback_centres(encoded)
    empty = FixedGroupState(
        module_membership=torch.zeros(1, 4, 6),
        environment_membership=torch.zeros(1, 5, 6),
        module_mass=torch.zeros(1, 6),
        environment_mass=torch.zeros(1, 6),
        module_centres=fallback,
        environment_centres=fallback,
        group_state=torch.zeros(1, 6, 8),
        valid=torch.zeros(1, 6, dtype=torch.bool),
        module_support=torch.zeros(1, 6, dtype=torch.long),
        environment_support=torch.zeros(1, 6, dtype=torch.long),
    )
    route = router.route_queries(
        encoded,
        empty,
        torch.rand(1, 3, 2),
        torch.randn(1, 3, 10),
    )
    assert torch.equal(route.assignment, torch.zeros_like(route.assignment))
    assert torch.isfinite(route.descriptor).all()


def test_router_exact_zero_supports_and_partial_empty_groups() -> None:
    encoded = _encoded()
    router = FixedGroupRouter(hidden_dim=8, spatial_dim=2, fourier_frequencies=2)
    # Make the numerical support contract deterministic: groups 0 and 1 are
    # active, while groups 2..5 are exact entmax zeros for all three routes.
    router.module_score = _FixedGroupScalarLogits([8.0, 8.0, 0.0, 0.0, 0.0, 0.0])
    router.environment_score = _FixedGroupScalarLogits([8.0, 8.0, 0.0, 0.0, 0.0, 0.0])
    router.environment_geometry_bias = _ZeroGeometryBias()
    router.query_score = _FixedGroupQueryLogits([8.0, 8.0, 0.0, 0.0, 0.0, 0.0])
    state = router.prepare(encoded, torch.randn(1, 4, 8), torch.randn(1, 5, 8))
    route = router.route_queries(
        encoded,
        state,
        torch.rand(1, 3, 2),
        torch.randn(1, 3, 10),
    )
    assert torch.all(state.module_membership[..., 2:] == 0)
    assert torch.all(state.environment_membership[..., 2:] == 0)
    assert torch.all(route.assignment[..., 2:] == 0)
    torch.testing.assert_close(state.module_membership.sum(-1), torch.ones(1, 4), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(state.environment_membership.sum(-1), torch.ones(1, 5), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(route.assignment.sum(-1), torch.ones(1, 3), atol=1.0e-6, rtol=0.0)
    assert torch.equal(state.valid, torch.tensor([[True, True, False, False, False, False]]))
    assert torch.isfinite(state.group_state).all()


def test_group_state_conditions_triads_environment_kv_and_query_features() -> None:
    torch.manual_seed(1405)
    encoded = _encoded()
    field = FixedGroupPairwiseField(8, 16, 2, 2, spatial_dim=2)
    source_states = torch.randn(1, 4, 8)
    prepared = field.prepare(encoded, source_states)
    original = prepared["router_state"]
    changed_group_state = original.group_state.clone()
    changed_group_state[:, 0, :] = changed_group_state[:, 0, :] + 0.75
    changed = type(original)(
        module_membership=original.module_membership,
        environment_membership=original.environment_membership,
        module_mass=original.module_mass,
        environment_mass=original.environment_mass,
        module_centres=original.module_centres,
        environment_centres=original.environment_centres,
        group_state=changed_group_state,
        valid=original.valid,
        module_support=original.module_support,
        environment_support=original.environment_support,
    )
    module_codes_a = field._build_module_source_codes(encoded, prepared["module_tokens"], original)
    module_codes_b = field._build_module_source_codes(encoded, prepared["module_tokens"], changed)
    env_codes_a = field._build_environment_source_codes(encoded, prepared["env_tokens"], original)
    env_codes_b = field._build_environment_source_codes(encoded, prepared["env_tokens"], changed)
    keys_a, values_a = field._prepare_environment_kv(env_codes_a)
    keys_b, values_b = field._prepare_environment_kv(env_codes_b)
    assert not torch.allclose(module_codes_a[:, :, 0], module_codes_b[:, :, 0])
    assert not torch.allclose(env_codes_a[:, :, 0], env_codes_b[:, :, 0])
    assert not torch.allclose(keys_a[:, 0], keys_b[:, 0])
    assert not torch.allclose(values_a[:, 0], values_b[:, 0])

    receivers = torch.rand(1, 3, 2)
    receiver_features = torch.randn(1, 3, 10)
    route_a = field.router.route_queries(encoded, original, receivers, receiver_features)
    route_b = field.router.route_queries(encoded, changed, receivers, receiver_features)
    assert not torch.allclose(route_a.descriptor[:, :, 0], route_b.descriptor[:, :, 0])
    changed_state = dict(prepared)
    changed_state.update(
        {
            "router_state": changed,
            "module_source_codes": module_codes_b,
            "environment_source_codes": env_codes_b,
            "environment_keys": keys_b,
            "environment_values": values_b,
        }
    )
    fixed_assignment_route = FixedGroupQueryRoute(
        descriptor=route_b.descriptor,
        assignment=route_a.assignment,
        logits=route_a.logits,
    )
    psi_a, _, _ = field._read_module_grouped(prepared, encoded, receivers, route_a)
    psi_b, _, _ = field._read_module_grouped(changed_state, encoded, receivers, fixed_assignment_route)
    assert not torch.allclose(psi_a, psi_b)

    context = ThreeTermInterfaceContext(hidden_dim=8, field_dim=3)
    assert not any("group_value" in name for name, _ in context.named_parameters())
    assert not hasattr(context, "group_state")


def test_fixed_group_reader_has_triads_shared_route_and_finite_gradients() -> None:
    torch.manual_seed(1405)
    encoded = _encoded()
    field = FixedGroupPairwiseField(
        hidden_dim=8,
        message_hidden_dim=16,
        num_heads=2,
        fourier_frequencies=2,
        spatial_dim=2,
    )
    module_states = torch.randn(1, 4, 8, requires_grad=True)
    state = field.prepare(encoded, module_states)
    receivers = torch.rand(1, 3, 2)
    receiver_features = torch.randn(1, 3, 10)
    context, aux = field.read(
        state,
        encoded,
        receivers,
        receiver_features,
        return_routing_maps=True,
    )
    assert context.shape == (1, 3, 8)
    assert torch.isfinite(context).all()
    assert aux["fixed_group_query_routing"].shape == (1, 3, 6)
    assert "fixed_group_module_incidence" in aux
    assert "fixed_group_environment_incidence" in aux
    assert "fixed_group_module_triple_source" in aux
    assert "fixed_group_environment_triple_source" in aux

    # Keeping q/source/alpha fixed while changing the group-conditioned query
    # descriptor changes the triadic psi_M response.
    route = field._route(state, encoded, receivers, receiver_features)
    changed_descriptor = route.descriptor.clone()
    changed_descriptor[:, :, 0, :] = changed_descriptor[:, :, 0, :] + 0.75
    changed_route = FixedGroupQueryRoute(
        descriptor=changed_descriptor,
        assignment=route.assignment,
        logits=route.logits,
    )
    baseline, _, _ = field._read_module_grouped(state, encoded, receivers, route)
    changed, _, _ = field._read_module_grouped(state, encoded, receivers, changed_route)
    assert not torch.allclose(baseline, changed)

    loss = context.square().mean()
    loss.backward()
    for name, parameter in field.named_parameters():
        if any(
            part in name
            for part in (
                "router.module_score",
                "router.environment_score",
                "router.group_state",
                "router.query_descriptor",
                "module_source_code",
                "environment_source_code",
                "module_pair_message",
                "environment_query_projection",
                "environment_key_projection",
                "environment_value_projection",
            )
        ):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_fixed_group_reader_is_permutation_invariant() -> None:
    torch.manual_seed(1405)
    encoded = _encoded()
    field = FixedGroupPairwiseField(8, 16, 2, 2, spatial_dim=2)
    module_states = torch.randn(1, 4, 8)
    receivers = torch.rand(1, 3, 2)
    receiver_features = torch.randn(1, 3, 10)
    first_state = field.prepare(encoded, module_states)
    first, _ = field.read(first_state, encoded, receivers, receiver_features)

    module_order = torch.tensor([2, 0, 3, 1])
    environment_order = torch.tensor([3, 1, 4, 0, 2])
    permuted = EncodedInterfaceCase(
        module_tokens=encoded.module_tokens[:, module_order],
        env_tokens=encoded.env_tokens[:, environment_order],
        global_token=encoded.global_token,
        module_centers=encoded.module_centers[:, module_order],
        env_coords=encoded.env_coords[:, environment_order],
        module_present=encoded.module_present[:, module_order],
        module_features=encoded.module_features[:, module_order],
        env_features=None,
        env_weights=encoded.env_weights[:, environment_order],
        coordinate_scale=encoded.coordinate_scale,
    )
    second_state = field.prepare(permuted, module_states[:, module_order])
    second, _ = field.read(second_state, permuted, receivers, receiver_features)
    torch.testing.assert_close(first, second, rtol=2.0e-5, atol=2.0e-5)


def test_three_term_context_has_exact_zero_local_branch_and_context_only_head() -> None:
    torch.manual_seed(1405)
    context = ThreeTermInterfaceContext(hidden_dim=8, field_dim=3)
    module_states = torch.randn(1, 2, 8)
    coarse_state = context.prepare_coarse(
        module_states,
        torch.randn(1, 4, 8),
        torch.ones(1, 2),
        torch.ones(1, 4),
    )
    receivers = torch.rand(1, 3, 2)
    zeros, counts = context.read_local(
        receivers,
        module_states,
        torch.rand(1, 2, 2),
        torch.randn(1, 2, 3),
        torch.ones(1, 2),
        torch.ones(1, 1, 2),
        1.0,
    )
    assert zeros.numel() == 3 * 8
    assert torch.equal(zeros, torch.zeros_like(zeros))
    assert torch.equal(counts, torch.zeros_like(counts))

    receiver_features = torch.randn(1, 3, 10)
    global_token = torch.randn(1, 8)
    background = context.read_coarse(receiver_features, global_token, coarse_state)
    field_a = context.predict_field(receivers, receiver_features, background, global_token, None)
    field_b = context.predict_field(
        receivers,
        torch.randn_like(receiver_features),
        background,
        torch.randn_like(global_token),
        torch.randn(1, 3, 2),
    )
    torch.testing.assert_close(field_a, field_b, rtol=0.0, atol=0.0)
