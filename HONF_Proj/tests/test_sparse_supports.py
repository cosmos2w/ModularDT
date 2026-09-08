"""Arithmetic and geometry checks for Stage-2 sparse cubic supports."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import product

import torch

from honf_forward_core.interface_fields.group_operator import (
    SparseGroupState,
    SparseInterfaceHONF,
    SparseLayoutCache,
)
from honf_forward_core.interface_fields.supports import (
    SparseSupportLayout,
    build_sparse_support_layout,
    cubic_bspline,
    lookup_sparse_supports,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _geometry(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, ...]:
    ports = torch.tensor(
        [[[[0.15, 0.25], [0.35, 0.25], [0.25, 0.45]],
          [[1.55, 0.70], [1.75, 0.70], [1.65, 0.90]]]],
        dtype=dtype,
    )
    present = torch.ones((1, 2), dtype=dtype)
    port_weights = torch.tensor([[[1.0, 2.0, 1.0], [2.0, 1.0, 3.0]]], dtype=dtype)
    environment = torch.tensor(
        [[[-0.20, 0.10], [0.50, 0.40], [1.20, 0.80], [2.10, 1.10]]],
        dtype=dtype,
    )
    environment_weights = torch.tensor([[0.20, 0.30, 0.40, 0.10]], dtype=dtype)
    return ports, present, port_weights, environment, environment_weights


def _dense_b3(value: torch.Tensor) -> torch.Tensor:
    distance = value.abs()
    inner = (4.0 - 6.0 * distance**2 + 3.0 * distance**3) / 6.0
    outer = (2.0 - distance).clamp_min(0.0) ** 3 / 6.0
    return torch.where(distance < 1.0, inner, torch.where(distance < 2.0, outer, 0.0))


def _dense_reference(
    ports: torch.Tensor,
    present: torch.Tensor,
    port_weights: torch.Tensor,
    environment: torch.Tensor,
    environment_weights: torch.Tensor,
    spacing: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intentionally dense tiny reference; never used by the implementation."""

    assert ports.shape[0] == 1
    dimension = int(ports.shape[-1])
    normalized = ports[0] / spacing
    lower = torch.floor(normalized.amin(dim=(0, 1))).to(torch.long) - 1
    upper = torch.floor(normalized.amax(dim=(0, 1))).to(torch.long) + 2
    ranges = [range(int(lower[axis]), int(upper[axis]) + 1) for axis in range(dimension)]
    keys = torch.tensor(list(product(*ranges)), device=ports.device, dtype=torch.long)
    centres = keys.to(ports.dtype) * spacing
    port_phi = _dense_b3((ports[0, :, :, None, :] - centres[None, None, :, :]) / spacing).prod(-1)
    normalized_quadrature = port_weights[0] / port_weights[0].sum(-1, keepdim=True)
    module_weight = (port_phi * normalized_quadrature[:, :, None]).sum(1)
    module_weight = module_weight * present[0, :, None]
    occupancy = module_weight.sum(0)
    occupied = occupancy > 0.0
    keys = keys[occupied]
    module_weight = module_weight[:, occupied]
    occupancy = occupancy[occupied]
    env_phi = _dense_b3(
        (environment[0, :, None, :] - centres[None, occupied, :]) / spacing
    ).prod(-1)
    environment_weight = environment_weights[0, :, None] * env_phi
    return keys, module_weight, environment_weight, occupancy


def _module_matrix(layout: object) -> torch.Tensor:
    indices = layout.module_group_indices.clone()
    indices[0] = torch.remainder(indices[0], layout.module_width)
    return torch.sparse_coo_tensor(
        indices,
        layout.module_support_weight,
        (layout.module_width, layout.num_groups),
    ).to_dense()


def _environment_matrix(layout: object) -> torch.Tensor:
    indices = layout.environment_group_indices.clone()
    indices[0] = torch.remainder(indices[0], layout.environment_width)
    return torch.sparse_coo_tensor(
        indices,
        layout.environment_geometric_weight,
        (layout.environment_width, layout.num_groups),
    ).to_dense()


def _receiver_matrix(layout: object, receivers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lookup = lookup_sparse_supports(layout, receivers)
    matrix = torch.sparse_coo_tensor(
        lookup.receiver_group_indices,
        lookup.support_weight,
        (receivers.shape[0] * receivers.shape[1], layout.num_groups),
    ).to_dense()
    return matrix.reshape(receivers.shape[0], receivers.shape[1], layout.num_groups), lookup.degree


def _learned_operator_case(
    *, group_read_mode: str = "null_softmax"
) -> tuple[
    SparseInterfaceHONF,
    EncodedInterfaceCase,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return a deterministic one-case system small enough for a dense oracle."""

    torch.manual_seed(37)
    hidden_dim = 8
    ports, present, port_weights, environment, environment_weights = _geometry()
    module_centers = torch.tensor([[[0.25, 0.31], [1.65, 0.77]]], dtype=torch.float64)
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn((1, 2, hidden_dim), dtype=torch.float64),
        env_tokens=torch.randn((1, 4, hidden_dim), dtype=torch.float64),
        global_token=torch.randn((1, hidden_dim), dtype=torch.float64),
        module_centers=module_centers,
        env_coords=environment,
        module_present=present,
        module_features=torch.randn((1, 2, 3), dtype=torch.float64),
        env_features=None,
        env_weights=environment_weights,
        coordinate_scale=torch.tensor([[[3.0, 2.0]]], dtype=torch.float64),
    )
    operator = SparseInterfaceHONF(
        hidden_dim=hidden_dim,
        message_hidden_dim=12,
        fourier_frequencies=2,
        support_spacing_factor=1.0,
        group_read_mode=group_read_mode,
    ).double().eval()
    module_states = torch.randn((1, 2, hidden_dim), dtype=torch.float64, requires_grad=True)
    receivers = torch.tensor(
        [[[0.10, 0.20], [0.80, 0.55], [1.45, 0.75], [2.20, 1.15]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    return operator, encoded, ports, port_weights, module_states, receivers


def _receiver_features(receivers: torch.Tensor) -> torch.Tensor:
    """Small differentiable stand-in for the facade's positional features."""

    return torch.cat([receivers, receivers.square(), torch.sin(receivers)], dim=-1)


def _dense_learned_prepare_read(
    operator: SparseInterfaceHONF,
    encoded: EncodedInterfaceCase,
    module_states: torch.Tensor,
    layout: object,
    receivers: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Intentionally dense test-only evaluation of every source/group pair."""

    assert layout.batch_size == 1
    module_count = int(module_states.shape[1])
    environment_count = int(encoded.env_tokens.shape[1])
    group_count = int(layout.group_count)
    hidden_dim = int(operator.hidden_dim)

    module_geometric = module_states.new_zeros(module_count, group_count)
    module_source, module_group = layout.module_group_indices
    module_geometric[module_source, module_group] = layout.module_support_weight
    module_relative = (
        encoded.module_centers[0, :, None, :] - layout.centres[None, :, :]
    ) / layout.spacing
    module_relative_features = operator.relative_fourier(module_relative)
    expanded_states = module_states[0, :, None, :].expand(-1, group_count, -1)
    expanded_features = encoded.module_features[0, :, None, :].expand(-1, group_count, -1)
    module_inputs = torch.cat(
        [expanded_states, expanded_features, module_relative_features], dim=-1
    )
    module_membership = torch.sigmoid(operator.module_membership(module_inputs).squeeze(-1))
    module_messages = operator.module_message(module_inputs)
    module_weights = module_geometric * module_membership
    module_pool = torch.einsum("mk,mkh->kh", module_weights, module_messages)
    module_pool = module_pool / (1.0 + layout.occupancy[:, None])

    environment_geometric = module_states.new_zeros(environment_count, group_count)
    environment_source, environment_group = layout.environment_group_indices
    environment_geometric[environment_source, environment_group] = (
        layout.environment_geometric_weight
    )
    environment_relative = (
        encoded.env_coords[0, :, None, :] - layout.centres[None, :, :]
    ) / layout.spacing
    environment_relative_features = operator.relative_fourier(environment_relative)
    expanded_environment = encoded.env_tokens[0, :, None, :].expand(-1, group_count, -1)
    environment_inputs = torch.cat(
        [expanded_environment, environment_relative_features], dim=-1
    )
    environment_membership = torch.sigmoid(
        operator.environment_membership(environment_inputs).squeeze(-1)
    )
    environment_messages = operator.environment_message(environment_inputs)
    environment_weights = environment_geometric * environment_membership
    environment_pool = torch.einsum(
        "ek,ekh->kh", environment_weights, environment_messages
    ) / layout.spacing.pow(layout.spatial_dim)

    position_features = operator.group_position_fourier(
        layout.centres / encoded.coordinate_scale.reshape(1, layout.spatial_dim)
    )
    group_inputs = torch.cat(
        [
            module_pool,
            environment_pool,
            torch.log1p(layout.occupancy)[:, None],
            layout.covered_volume_ratio[:, None],
            position_features,
            encoded.global_token.index_select(0, layout.group_batch),
        ],
        dim=-1,
    )
    group_base = operator.group_input(group_inputs)
    group_state = group_base + operator.group_residual(group_base)
    normalized_group = operator.group_norm(group_state)
    group_keys = operator.group_key(normalized_group)
    group_values = operator.group_value(normalized_group)

    receiver_features = _receiver_features(receivers)
    receiver_query = operator.receiver_query(
        torch.cat(
            [
                receiver_features,
                encoded.global_token[:, None, :].expand(-1, receivers.shape[1], -1),
            ],
            dim=-1,
        )
    )[0]
    receiver_relative = (
        receivers[0, :, None, :] - layout.centres[None, :, :]
    ) / layout.spacing
    receiver_logits = (
        torch.einsum("qh,kh->qk", receiver_query, group_keys)
        / math.sqrt(float(hidden_dim))
        + operator.receiver_bias(operator.relative_fourier(receiver_relative)).squeeze(-1)
    )
    receiver_geometric = _dense_b3(receiver_relative).prod(-1)
    receiver_weight = (
        layout.occupancy_envelope[None, :]
        * receiver_geometric
        * torch.exp(receiver_logits)
    )
    receiver_context = torch.einsum("qk,kh->qh", receiver_weight, group_values)
    receiver_context = receiver_context / (1.0 + receiver_weight.sum(dim=-1, keepdim=True))
    return {
        "module_membership": module_membership,
        "environment_membership": environment_membership,
        "module_pool": module_pool,
        "environment_pool": environment_pool,
        "group_state": group_state,
        "group_keys": group_keys,
        "group_values": group_values,
        "context": receiver_context.unsqueeze(0),
    }


def test_cubic_basis_values_partition_and_generic_three_dimensional_candidates() -> None:
    locations = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    expected = torch.tensor([0.0, 1.0 / 6.0, 2.0 / 3.0, 23.0 / 48.0, 1.0 / 6.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(cubic_bspline(locations), expected)

    point = torch.tensor([0.37, -1.21, 2.73], dtype=torch.float64)
    base = torch.floor(point).to(torch.long)
    keys = base + torch.tensor(list(product((-1, 0, 1, 2), repeat=3)), dtype=torch.long)
    weights = cubic_bspline(point - keys).prod(-1)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert int((weights > 0.0).sum()) == 4**3


def test_sparse_geometry_and_coordinate_gradients_match_tiny_dense_reference() -> None:
    ports, present, port_weights, environment, environment_weights = _geometry()
    sparse_ports = ports.clone().requires_grad_(True)
    sparse_environment = environment.clone().requires_grad_(True)
    layout = build_sparse_support_layout(
        sparse_ports,
        present,
        sparse_environment,
        environment_weights,
        spacing=1.0,
        port_quadrature_weights=port_weights,
    )
    sparse_module = _module_matrix(layout)
    sparse_environment_weight = _environment_matrix(layout)

    dense_ports = ports.clone().requires_grad_(True)
    dense_environment = environment.clone().requires_grad_(True)
    keys, dense_module, dense_environment_weight, dense_occupancy = _dense_reference(
        dense_ports,
        present,
        port_weights,
        dense_environment,
        environment_weights,
        1.0,
    )
    torch.testing.assert_close(layout.lattice_keys, keys)
    torch.testing.assert_close(sparse_module, dense_module)
    torch.testing.assert_close(sparse_environment_weight, dense_environment_weight)
    torch.testing.assert_close(layout.occupancy, dense_occupancy)
    torch.testing.assert_close(
        layout.occupancy_envelope,
        -torch.expm1(-(4.0**2) * dense_occupancy),
    )
    torch.testing.assert_close(layout.covered_volume_ratio, dense_environment_weight.sum(0))
    assert torch.all((layout.occupancy_envelope > 0.0) & (layout.occupancy_envelope <= 1.0))

    group_coefficient = torch.sin(keys.to(torch.float64) @ torch.tensor([0.31, -0.47], dtype=torch.float64))
    module_coefficient = torch.arange(1, 3, dtype=torch.float64)[:, None] * group_coefficient
    environment_coefficient = torch.arange(1, 5, dtype=torch.float64)[:, None] * group_coefficient
    sparse_loss = (
        (sparse_module * module_coefficient).sum()
        + (sparse_environment_weight * environment_coefficient).sum()
        + 0.2 * (layout.occupancy_envelope * group_coefficient).sum()
    )
    dense_envelope = -torch.expm1(-(4.0**2) * dense_occupancy)
    dense_loss = (
        (dense_module * module_coefficient).sum()
        + (dense_environment_weight * environment_coefficient).sum()
        + 0.2 * (dense_envelope * group_coefficient).sum()
    )
    sparse_gradients = torch.autograd.grad(sparse_loss, (sparse_ports, sparse_environment))
    dense_gradients = torch.autograd.grad(dense_loss, (dense_ports, dense_environment))
    torch.testing.assert_close(sparse_loss, dense_loss)
    torch.testing.assert_close(sparse_gradients[0], dense_gradients[0], rtol=2.0e-12, atol=2.0e-12)
    torch.testing.assert_close(sparse_gradients[1], dense_gradients[1], rtol=2.0e-12, atol=2.0e-12)


def test_learned_sparse_prepare_read_and_gradients_match_dense_reference() -> None:
    operator, encoded, ports, port_weights, module_states, receivers = _learned_operator_case()
    cache = operator.build_layout(
        encoded,
        ports,
        module_radius=1.0,
        port_quadrature_weights=port_weights,
    )
    sparse_state = operator.prepare(encoded, module_states, cache)
    sparse_context, _ = operator.read(
        sparse_state,
        encoded,
        receivers,
        _receiver_features(receivers),
    )
    dense = _dense_learned_prepare_read(
        operator,
        encoded,
        module_states,
        cache.layout,
        receivers,
    )

    module_source, module_group = cache.layout.module_group_indices
    environment_source, environment_group = cache.layout.environment_group_indices
    torch.testing.assert_close(
        sparse_state.module_learned_membership,
        dense["module_membership"][module_source, module_group],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert sparse_state.cache.environment is not None
    torch.testing.assert_close(
        sparse_state.cache.environment.learned_membership,
        dense["environment_membership"][environment_source, environment_group],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    for sparse_value, dense_key in (
        (sparse_state.module_pool, "module_pool"),
        (sparse_state.cache.environment.pool, "environment_pool"),
        (sparse_state.group_state, "group_state"),
        (sparse_state.group_keys, "group_keys"),
        (sparse_state.group_values, "group_values"),
        (sparse_context, "context"),
    ):
        torch.testing.assert_close(sparse_value, dense[dense_key], rtol=2.0e-11, atol=2.0e-12)

    context_coefficient = torch.linspace(
        -0.7,
        0.9,
        sparse_context.numel(),
        dtype=sparse_context.dtype,
    ).reshape_as(sparse_context)
    group_coefficient = torch.linspace(
        0.4,
        -0.2,
        sparse_state.group_state.numel(),
        dtype=sparse_state.group_state.dtype,
    ).reshape_as(sparse_state.group_state)
    sparse_loss = (
        sparse_context * context_coefficient
    ).sum() + 0.17 * (sparse_state.group_state * group_coefficient).sum()
    dense_loss = (
        dense["context"] * context_coefficient
    ).sum() + 0.17 * (dense["group_state"] * group_coefficient).sum()
    named_parameters = tuple(operator.named_parameters())
    differentiation_targets = (
        module_states,
        receivers,
        *(parameter for _, parameter in named_parameters),
    )
    sparse_gradients = torch.autograd.grad(sparse_loss, differentiation_targets)
    dense_gradients = torch.autograd.grad(dense_loss, differentiation_targets)
    gradient_names = ("module_states", "receivers", *(name for name, _ in named_parameters))
    for name, sparse_gradient, dense_gradient in zip(
        gradient_names,
        sparse_gradients,
        dense_gradients,
        strict=True,
    ):
        assert torch.isfinite(sparse_gradient).all(), name
        torch.testing.assert_close(
            sparse_gradient,
            dense_gradient,
            rtol=2.0e-9,
            atol=2.0e-10,
            msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
        )


def test_environment_quadrature_duplication_preserves_learned_preparation_and_read() -> None:
    operator, encoded, ports, port_weights, module_states, receivers = _learned_operator_case()
    original_cache = operator.build_layout(
        encoded,
        ports,
        module_radius=1.0,
        port_quadrature_weights=port_weights,
    )
    original_state = operator.prepare(encoded, module_states, original_cache)
    original_context, _ = operator.read(
        original_state,
        encoded,
        receivers,
        _receiver_features(receivers),
    )

    duplicated_encoded = replace(
        encoded,
        env_tokens=encoded.env_tokens.repeat_interleave(2, dim=1),
        env_coords=encoded.env_coords.repeat_interleave(2, dim=1),
        env_weights=encoded.env_weights.repeat_interleave(2, dim=1) / 2.0,
    )
    duplicated_cache = operator.build_layout(
        duplicated_encoded,
        ports,
        module_radius=1.0,
        port_quadrature_weights=port_weights,
    )
    duplicated_state = operator.prepare(duplicated_encoded, module_states, duplicated_cache)
    duplicated_context, _ = operator.read(
        duplicated_state,
        duplicated_encoded,
        receivers,
        _receiver_features(receivers),
    )

    torch.testing.assert_close(
        duplicated_cache.layout.lattice_keys,
        original_cache.layout.lattice_keys,
    )
    assert original_state.cache.environment is not None
    assert duplicated_state.cache.environment is not None
    for original, duplicated in (
        (original_state.cache.environment.pool, duplicated_state.cache.environment.pool),
        (original_state.module_pool, duplicated_state.module_pool),
        (original_state.group_state, duplicated_state.group_state),
        (original_state.group_keys, duplicated_state.group_keys),
        (original_state.group_values, duplicated_state.group_values),
        (original_context, duplicated_context),
    ):
        torch.testing.assert_close(original, duplicated, rtol=2.0e-12, atol=2.0e-12)


def test_module_permutation_padding_flat_batch_offsets_and_boundary_centres() -> None:
    ports, present, port_weights, environment, environment_weights = _geometry()
    original = build_sparse_support_layout(
        ports,
        present,
        environment,
        environment_weights,
        spacing=1.0,
        port_quadrature_weights=port_weights,
    )
    padding = torch.tensor(
        [[[[90.0, 90.0], [91.0, 90.0], [90.0, 91.0]],
          [[-80.0, -80.0], [-81.0, -80.0], [-80.0, -81.0]],
          [[40.0, -40.0], [41.0, -40.0], [40.0, -41.0]]]],
        dtype=ports.dtype,
    )
    padded_ports = torch.cat([ports, padding], dim=1)
    padded_present = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=ports.dtype)
    padded_weights = torch.cat([port_weights, torch.zeros((1, 3, 3), dtype=ports.dtype)], dim=1)
    padded = build_sparse_support_layout(
        padded_ports,
        padded_present,
        environment,
        environment_weights,
        spacing=1.0,
        port_quadrature_weights=padded_weights,
    )
    torch.testing.assert_close(padded.lattice_keys, original.lattice_keys)
    torch.testing.assert_close(padded.occupancy, original.occupancy)
    torch.testing.assert_close(_module_matrix(padded)[:2], _module_matrix(original))
    assert torch.count_nonzero(_module_matrix(padded)[2:]) == 0

    permutation = torch.tensor([4, 1, 3, 0, 2])
    permuted = build_sparse_support_layout(
        padded_ports[:, permutation],
        padded_present[:, permutation],
        environment,
        environment_weights,
        spacing=1.0,
        port_quadrature_weights=padded_weights[:, permutation],
    )
    torch.testing.assert_close(permuted.lattice_keys, original.lattice_keys)
    torch.testing.assert_close(permuted.occupancy, original.occupancy)
    restored = torch.zeros_like(_module_matrix(permuted))
    restored[permutation] = _module_matrix(permuted)
    torch.testing.assert_close(restored[:2], _module_matrix(original))
    assert torch.count_nonzero(restored[2:]) == 0

    # Port overlap, rather than a clipped domain lattice, owns support keys.
    assert bool((original.centres < 0.0).any())

    second_ports = padded_ports + torch.tensor([3.0, 2.0], dtype=ports.dtype)
    batched = build_sparse_support_layout(
        torch.cat([padded_ports, second_ports], dim=0),
        torch.cat([padded_present, padded_present], dim=0),
        torch.cat([environment, environment + torch.tensor([3.0, 2.0], dtype=ports.dtype)], dim=0),
        torch.cat([environment_weights, environment_weights], dim=0),
        spacing=1.0,
        port_quadrature_weights=torch.cat([padded_weights, padded_weights], dim=0),
    )
    assert batched.case_group_offsets.tolist() == [0, original.num_groups, batched.num_groups]
    module_batch = torch.div(batched.module_group_indices[0], batched.module_width, rounding_mode="floor")
    environment_batch = torch.div(
        batched.environment_group_indices[0], batched.environment_width, rounding_mode="floor"
    )
    torch.testing.assert_close(module_batch, batched.group_batch[batched.module_group_indices[1]])
    torch.testing.assert_close(environment_batch, batched.group_batch[batched.environment_group_indices[1]])


def test_environment_quadrature_duplication_and_receiver_chunking_are_invariant() -> None:
    ports, present, port_weights, environment, environment_weights = _geometry()
    layout = build_sparse_support_layout(
        ports,
        present,
        environment,
        environment_weights,
        spacing=1.0,
        port_quadrature_weights=port_weights,
    )
    duplicated = build_sparse_support_layout(
        ports,
        present,
        environment.repeat_interleave(2, dim=1),
        environment_weights.repeat_interleave(2, dim=1) / 2.0,
        spacing=1.0,
        port_quadrature_weights=port_weights,
    )
    torch.testing.assert_close(duplicated.lattice_keys, layout.lattice_keys)
    torch.testing.assert_close(duplicated.occupancy, layout.occupancy)
    torch.testing.assert_close(duplicated.covered_volume_ratio, layout.covered_volume_ratio)

    receivers = torch.tensor(
        [[[-0.25, 0.05], [0.10, 0.20], [0.80, 0.50], [1.40, 0.75], [2.30, 1.20]]],
        dtype=ports.dtype,
    )
    whole, whole_degree = _receiver_matrix(layout, receivers)
    first, first_degree = _receiver_matrix(layout, receivers[:, :2])
    second, second_degree = _receiver_matrix(layout, receivers[:, 2:])
    torch.testing.assert_close(whole, torch.cat([first, second], dim=1))
    torch.testing.assert_close(whole_degree, torch.cat([first_degree, second_degree], dim=1))
    assert int(whole_degree.max()) <= 4**2

    dense_receiver = _dense_b3(
        (receivers[:, :, None, :] - layout.centres[None, None, :, :]) / layout.spacing
    ).prod(-1)
    torch.testing.assert_close(whole, dense_receiver)


def test_layout_scoped_support_search_matches_uncached_exact_lookup() -> None:
    ports, present, port_weights, environment, environment_weights = _geometry()
    layout = build_sparse_support_layout(
        ports,
        present,
        environment,
        environment_weights,
        spacing=1.0,
        port_quadrature_weights=port_weights,
    )
    assert layout.support_search is not None
    receivers = torch.tensor(
        [[[-20.0, -20.0], [-0.25, 0.05], [0.10, 0.20], [2.30, 1.20], [20.0, 20.0]]],
        dtype=ports.dtype,
    )
    cached = lookup_sparse_supports(layout, receivers)
    uncached = lookup_sparse_supports(replace(layout, support_search=None), receivers)
    torch.testing.assert_close(cached.receiver_group_indices, uncached.receiver_group_indices)
    torch.testing.assert_close(cached.support_weight, uncached.support_weight)
    torch.testing.assert_close(cached.degree, uncached.degree)


def test_support_transition_is_smooth_and_matches_directional_finite_difference() -> None:
    environment = torch.tensor([[[-1.5], [-0.5], [0.5], [1.5]]], dtype=torch.float64)
    environment_weights = torch.full((1, 4), 0.5, dtype=torch.float64)
    present = torch.ones((1, 1), dtype=torch.float64)

    def evaluate(displacement: float, *, with_gradient: bool = False) -> tuple[float, float | None]:
        # The sole footprint port translates with its module centre.  At zero,
        # the outer keys -2 and +2 exchange at zero cubic weight.
        port = torch.tensor([[[[displacement]]]], dtype=torch.float64, requires_grad=with_gradient)
        layout = build_sparse_support_layout(
            port,
            present,
            environment,
            environment_weights,
            spacing=1.0,
        )
        coefficient = torch.sin(0.7 * layout.lattice_keys[:, 0]) + 0.2 * layout.lattice_keys[:, 0]
        value = (layout.occupancy_envelope * coefficient).sum()
        if not with_gradient:
            return float(value), None
        gradient = torch.autograd.grad(value, port)[0]
        return float(value), float(gradient[0, 0, 0, 0])

    centre_value, analytic = evaluate(0.0, with_gradient=True)
    step = 1.0e-4
    positive, _ = evaluate(step)
    negative, _ = evaluate(-step)
    finite_difference = (positive - negative) / (2.0 * step)
    assert abs(positive - negative) < 5.0e-4
    assert analytic is not None
    torch.testing.assert_close(
        torch.tensor(analytic), torch.tensor(finite_difference), rtol=2.0e-5, atol=2.0e-6
    )
    assert abs(centre_value) < 1.0e-14


def _geometry_envelope_reference_read(
    operator: SparseInterfaceHONF,
    state: object,
    encoded: EncodedInterfaceCase,
    receivers: torch.Tensor,
    receiver_features: torch.Tensor,
    *,
    logit_shift: float = 0.0,
) -> torch.Tensor:
    """Tiny dense oracle for the conditional geometry-envelope formula."""

    layout = state.cache.layout
    lookup = lookup_sparse_supports(layout, receivers)
    flat_count = int(receivers.shape[0] * receivers.shape[1])
    query = operator.receiver_query(
        torch.cat(
            [
                receiver_features,
                encoded.global_token[:, None, :].expand(-1, receivers.shape[1], -1),
            ],
            dim=-1,
        )
    ).reshape(flat_count, operator.hidden_dim)
    receiver_index, group_index = lookup.receiver_group_indices
    relative = (
        receivers.reshape(flat_count, layout.spatial_dim)[receiver_index]
        - layout.centres[group_index]
    ) / layout.spacing
    logits = (
        (query[receiver_index] * state.group_keys[group_index]).sum(dim=-1)
        / math.sqrt(float(operator.hidden_dim))
        + operator.receiver_bias(operator.relative_fourier(relative)).squeeze(-1)
        + float(logit_shift)
    )
    geometric = layout.occupancy_envelope[group_index] * lookup.geometric_weight
    contexts = []
    for receiver in range(flat_count):
        rows = receiver_index == receiver
        if not bool(rows.any()):
            contexts.append(state.group_values.new_zeros(operator.hidden_dim))
            continue
        weights = geometric[rows]
        conditional = torch.softmax(torch.log(weights) + logits[rows], dim=0)
        availability = weights.sum()
        contexts.append(
            availability * (conditional[:, None] * state.group_values[group_index[rows]]).sum(dim=0)
        )
    return torch.stack(contexts, dim=0).reshape(
        receivers.shape[0], receivers.shape[1], operator.hidden_dim
    )


def test_geometry_envelope_reader_matches_dense_formula_and_preserves_available_mass() -> None:
    operator, encoded, ports, port_weights, module_states, receivers = _learned_operator_case(
        group_read_mode="geometry_envelope_attention"
    )
    cache = operator.build_layout(
        encoded,
        ports,
        module_radius=1.0,
        port_quadrature_weights=port_weights,
    )
    state = operator.prepare(encoded, module_states, cache)
    receiver_features = _receiver_features(receivers)
    context, aux = operator.read(
        state,
        encoded,
        receivers,
        receiver_features,
        return_routing_maps=True,
    )
    expected = _geometry_envelope_reference_read(
        operator, state, encoded, receivers, receiver_features
    )
    torch.testing.assert_close(context, expected, rtol=2.0e-11, atol=2.0e-12)
    weights = aux["group_read_normalized_weight"]
    availability = aux["group_read_geometric_availability"]
    torch.testing.assert_close(weights.sum(dim=-1), availability, rtol=2.0e-11, atol=2.0e-12)
    torch.testing.assert_close(aux["group_read_weight_mass"], availability)
    torch.testing.assert_close(aux["group_read_null_weight_mass"], torch.zeros_like(availability))
    assert torch.isfinite(context).all()


def test_geometry_envelope_reader_is_invariant_to_common_logit_shift() -> None:
    operator, encoded, ports, port_weights, module_states, receivers = _learned_operator_case(
        group_read_mode="geometry_envelope_attention"
    )
    cache = operator.build_layout(
        encoded,
        ports,
        module_radius=1.0,
        port_quadrature_weights=port_weights,
    )
    state = operator.prepare(encoded, module_states, cache)
    receiver_features = _receiver_features(receivers)
    original = _geometry_envelope_reference_read(
        operator, state, encoded, receivers, receiver_features
    )
    shifted = _geometry_envelope_reference_read(
        operator,
        state,
        encoded,
        receivers,
        receiver_features,
        logit_shift=-60.0,
    )
    torch.testing.assert_close(original, shifted, rtol=2.0e-12, atol=2.0e-12)


def test_geometry_envelope_reader_returns_zero_for_unsupported_receivers_and_has_finite_boundary_gradients() -> None:
    operator, encoded, ports, port_weights, module_states, receivers = _learned_operator_case(
        group_read_mode="geometry_envelope_attention"
    )
    cache = operator.build_layout(
        encoded,
        ports,
        module_radius=1.0,
        port_quadrature_weights=port_weights,
    )
    state = operator.prepare(encoded, module_states, cache)
    boundary = cache.layout.centres[:1].detach() + torch.tensor(
        [[2.0, 0.0]], dtype=receivers.dtype
    ) * cache.layout.spacing
    query = torch.cat(
        [boundary.reshape(1, 1, 2), torch.tensor([[[100.0, 100.0]]], dtype=receivers.dtype)],
        dim=1,
    ).requires_grad_(True)
    context, aux = operator.read(
        state,
        encoded,
        query,
        _receiver_features(query),
        return_routing_maps=False,
    )
    assert torch.equal(context[:, 1], torch.zeros_like(context[:, 1]))
    assert torch.equal(aux["group_read_geometric_availability"][:, 1], torch.zeros_like(aux["group_read_geometric_availability"][:, 1]))
    gradient = torch.autograd.grad(context[:, 0].sum(), query)[0]
    assert torch.isfinite(gradient).all()


def test_geometry_envelope_reader_is_mixed_batch_and_chunk_safe() -> None:
    operator, encoded, ports, port_weights, module_states, receivers = _learned_operator_case(
        group_read_mode="geometry_envelope_attention"
    )
    shift = torch.tensor([3.0, 2.0], dtype=encoded.module_centers.dtype)
    batched_encoded = replace(
        encoded,
        module_tokens=encoded.module_tokens.repeat(2, 1, 1),
        env_tokens=encoded.env_tokens.repeat(2, 1, 1),
        global_token=encoded.global_token.repeat(2, 1),
        module_centers=torch.cat([encoded.module_centers, encoded.module_centers + shift], dim=0),
        env_coords=torch.cat([encoded.env_coords, encoded.env_coords + shift], dim=0),
        module_present=encoded.module_present.repeat(2, 1),
        module_features=encoded.module_features.repeat(2, 1, 1),
        env_weights=encoded.env_weights.repeat(2, 1),
        coordinate_scale=encoded.coordinate_scale.repeat(2, 1, 1),
    )
    batched_ports = torch.cat([ports, ports + shift], dim=0)
    batched_weights = port_weights.repeat(2, 1, 1)
    batched_states = module_states.detach().repeat(2, 1, 1).requires_grad_(True)
    batched_receivers = torch.cat([receivers, receivers + shift], dim=0)
    cache = operator.build_layout(
        batched_encoded,
        batched_ports,
        module_radius=1.0,
        port_quadrature_weights=batched_weights,
    )
    state = operator.prepare(batched_encoded, batched_states, cache)
    context, aux = operator.read(
        state,
        batched_encoded,
        batched_receivers,
        _receiver_features(batched_receivers),
        return_routing_maps=True,
    )
    assert context.shape == (2, receivers.shape[1], operator.hidden_dim)
    assert aux["group_read_geometric_availability"].shape == (2, receivers.shape[1])
    torch.testing.assert_close(
        aux["group_read_normalized_weight"].sum(dim=-1),
        aux["group_read_geometric_availability"],
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    gradient = torch.autograd.grad(context.sum(), batched_states)[0]
    assert torch.isfinite(gradient).all()


def test_geometry_envelope_reader_handles_float32_tiny_availability_and_extreme_logits() -> None:
    operator, encoded, ports, port_weights, module_states, receivers = _learned_operator_case(
        group_read_mode="geometry_envelope_attention"
    )
    operator = operator.float()
    encoded = replace(
        encoded,
        module_tokens=encoded.module_tokens.float(),
        env_tokens=encoded.env_tokens.float(),
        global_token=encoded.global_token.float(),
        module_centers=encoded.module_centers.float(),
        env_coords=encoded.env_coords.float(),
        module_present=encoded.module_present.float(),
        module_features=encoded.module_features.float(),
        env_weights=encoded.env_weights.float(),
        coordinate_scale=encoded.coordinate_scale.float(),
    )
    ports = ports.float()
    port_weights = port_weights.float()
    module_states = module_states.detach().float().requires_grad_(True)
    receivers = receivers.detach().float().requires_grad_(True)
    cache = operator.build_layout(
        encoded,
        ports,
        module_radius=1.0,
        port_quadrature_weights=port_weights,
    )
    tiny_envelope = torch.full_like(cache.layout.occupancy_envelope, 1.0e-40).requires_grad_()
    cache = replace(
        cache,
        layout=replace(
            cache.layout,
            occupancy_envelope=tiny_envelope,
        ),
    )
    state = operator.prepare(encoded, module_states, cache)
    # Initialize the lazy receiver modules before applying a large common
    # bias to exercise the stable log-sum-exp path at both signs.
    receiver_features = _receiver_features(receivers)
    context, _ = operator.read(
        state, encoded, receivers, receiver_features, return_routing_maps=False
    )
    receiver_last = operator.receiver_bias.net[-1]
    assert isinstance(receiver_last, torch.nn.Linear)
    with torch.no_grad():
        receiver_last.bias.fill_(1000.0)
    positive, _ = operator.read(
        state, encoded, receivers, _receiver_features(receivers), return_routing_maps=False
    )
    with torch.no_grad():
        receiver_last.bias.fill_(-1000.0)
    negative, _ = operator.read(
        state, encoded, receivers, _receiver_features(receivers), return_routing_maps=False
    )
    # A linear projection keeps the tiny context's coordinate/envelope
    # derivative observable; squaring it would underflow to zero in FP32.
    loss = context.sum() + positive.sum() + negative.sum()
    gradients = torch.autograd.grad(loss, (module_states, receivers, tiny_envelope))
    assert torch.isfinite(positive).all()
    assert torch.isfinite(negative).all()
    assert all(torch.isfinite(value).all() for value in gradients)
    assert torch.count_nonzero(gradients[-1]) > 0


def test_geometry_envelope_reader_has_the_single_group_interpolation_limit() -> None:
    dtype = torch.float64
    operator = SparseInterfaceHONF(
        hidden_dim=2,
        message_hidden_dim=4,
        fourier_frequencies=1,
        support_spacing_factor=1.0,
        group_read_mode="geometry_envelope_attention",
    ).double().eval()
    layout = SparseSupportLayout(
        lattice_keys=torch.zeros((1, 2), dtype=torch.long),
        centres=torch.zeros((1, 2), dtype=dtype),
        group_batch=torch.zeros((1,), dtype=torch.long),
        case_group_offsets=torch.tensor([0, 1], dtype=torch.long),
        module_group_indices=torch.tensor([[0], [0]], dtype=torch.long),
        module_support_weight=torch.ones((1,), dtype=dtype),
        environment_group_indices=torch.tensor([[0], [0]], dtype=torch.long),
        environment_geometric_weight=torch.ones((1,), dtype=dtype),
        occupancy=torch.ones((1,), dtype=dtype),
        occupancy_envelope=torch.tensor([0.75], dtype=dtype),
        covered_volume_ratio=torch.ones((1,), dtype=dtype),
        origin=torch.zeros((1, 2), dtype=dtype),
        spacing=torch.tensor(1.0, dtype=dtype),
        batch_size=1,
        module_width=1,
        environment_width=1,
        spatial_dimension=2,
    )
    cache = SparseLayoutCache(layout=layout)
    values = torch.tensor([[1.25, -0.75]], dtype=dtype)
    state = SparseGroupState(
        cache=cache,
        group_state=torch.zeros((1, 2), dtype=dtype),
        group_keys=torch.zeros((1, 2), dtype=dtype),
        group_values=values,
        module_learned_membership=torch.ones((1,), dtype=dtype),
        module_pool=torch.zeros((1, 2), dtype=dtype),
    )
    encoded = EncodedInterfaceCase(
        module_tokens=torch.zeros((1, 1, 2), dtype=dtype),
        env_tokens=torch.zeros((1, 1, 2), dtype=dtype),
        global_token=torch.zeros((1, 2), dtype=dtype),
        module_centers=torch.zeros((1, 1, 2), dtype=dtype),
        env_coords=torch.zeros((1, 1, 2), dtype=dtype),
        module_present=torch.ones((1, 1), dtype=dtype),
        module_features=torch.zeros((1, 1, 1), dtype=dtype),
        env_features=None,
        env_weights=torch.ones((1, 1), dtype=dtype),
        coordinate_scale=torch.ones((1, 1, 2), dtype=dtype),
    )
    receivers = torch.zeros((1, 1, 2), dtype=dtype)
    receiver_features = _receiver_features(receivers)
    context, aux = operator.read(state, encoded, receivers, receiver_features)
    expected_availability = torch.tensor(0.75 * (2.0 / 3.0) ** 2, dtype=dtype)
    torch.testing.assert_close(aux["group_read_geometric_availability"], expected_availability.reshape(1, 1))
    torch.testing.assert_close(context[0, 0], expected_availability * values[0])
