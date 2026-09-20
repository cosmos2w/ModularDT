"""Focused Run-1408 quadrature math and executed-work checks."""

from __future__ import annotations

import torch
from channelthermal.environment import ChannelThermalEnvironmentBuilder

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import InterfaceFieldCore
from honf_forward_core.interface_fields.environment_sampling import RegularGridLayout
from honf_forward_core.interface_fields.group_control_router import GroupQueryRoute
from honf_forward_core.interface_fields.hypergraph_quadrature import (
    HypergraphQuadratureField,
    interpolate_regular_grid,
    mass_weighted_response,
)


def _payload(*, activation_checkpointing: bool = False) -> dict[str, object]:
    return {
        "forward_architecture": "hypergraph_quadrature_honf",
        "field_dim": 3,
        "hidden_dim": 16,
        "coordinate_scale": [8.0, 4.0],
        "boundary_feature_mode": "none",
        "interface_model": {
            "message_hidden_dim": 16,
            "attention_heads": 4,
            "relative_fourier_frequencies": 2,
            "receiver_chunk_size": 8,
            "group_count": 6,
            "source_normalizer": "entmax15",
            "query_normalizer": "entmax15",
            "module_temperature": 1.0,
            "environment_temperature": 1.0,
            "query_temperature": 1.0,
            "group_control_dim": 16,
            "samples_per_group": 4,
            "activation_checkpointing": activation_checkpointing,
        },
    }


def _batch(seed: int = 1408, *, query_count: int = 5) -> tuple[BatchData, RegularGridLayout]:
    generator = torch.Generator().manual_seed(seed)
    extent = torch.tensor([8.0, 4.0])
    environment = ChannelThermalEnvironmentBuilder()(
        batch_size=2,
        num_env_tokens_x=4,
        num_env_tokens_y=3,
        domain_length_x=8.0,
        domain_length_y=4.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    batch = BatchData(
        module_centers=torch.rand(2, 3, 2, generator=generator, dtype=torch.float32) * extent,
        module_present=torch.ones(2, 3, dtype=torch.float32),
        module_features=torch.randn(2, 3, 3, generator=generator, dtype=torch.float32),
        global_context=torch.randn(2, 5, generator=generator, dtype=torch.float32),
        query_xy=torch.rand(2, query_count, 2, generator=generator, dtype=torch.float32) * extent,
        query_time=None,
        target_field=None,
        case_name="run1408-test",
        metadata={},
        env_coords=environment.env_coords,
        env_features=environment.env_features,
        env_weights=environment.env_weights,
        sampler_layout=environment.sampler_layout,
    )
    return batch, environment.sampler_layout


def test_dense_node_mass_weighted_identity_and_exact_empty_branch() -> None:
    generator = torch.Generator().manual_seed(17)
    scores = torch.randn(1, 2, 3, 5, generator=generator, dtype=torch.float64)
    values = torch.randn(1, 2, 3, 5, 4, generator=generator, dtype=torch.float64)
    masses = torch.rand(1, 3, 5, generator=generator, dtype=torch.float64) + 0.1
    response, valid = mass_weighted_response(scores, values, masses)
    expected_weights = torch.softmax(scores + masses.log()[:, None], dim=-1)
    expected = torch.einsum("bhqj,bhqjd->bhqd", expected_weights, values)
    torch.testing.assert_close(response, expected, rtol=1.0e-12, atol=1.0e-12)
    assert valid.all()

    empty_values = values.detach().clone().requires_grad_(True)
    empty_scores = scores.detach().clone().requires_grad_(True)
    empty_masses = torch.zeros_like(masses)
    empty_response, empty_valid = mass_weighted_response(empty_scores, empty_values, empty_masses)
    assert not empty_valid.any()
    assert torch.equal(empty_response, torch.zeros_like(empty_response))
    empty_response.square().sum().backward()
    assert torch.isfinite(empty_scores.grad).all()
    assert torch.isfinite(empty_values.grad).all()


def test_dense_node_group_regrouping_matches_parent_rho_operator() -> None:
    generator = torch.Generator().manual_seed(1408)
    batch, heads, queries, sources, groups, width = 2, 2, 3, 5, 4, 3
    scores = torch.randn(batch, heads, queries, sources, generator=generator, dtype=torch.float64)
    values = torch.randn(batch, heads, queries, sources, width, generator=generator, dtype=torch.float64)
    alpha = torch.softmax(torch.randn(batch, queries, groups, generator=generator, dtype=torch.float64), dim=-1)
    membership = torch.softmax(torch.randn(batch, sources, groups, generator=generator, dtype=torch.float64), dim=-1)
    membership[..., 0] = 0.0  # Explicit empty group exercises the masked beta branch.
    membership = membership / membership.sum(dim=-1, keepdim=True)
    omega = torch.rand(batch, sources, generator=generator, dtype=torch.float64) + 0.1
    omega = omega / omega.sum(dim=-1, keepdim=True)
    mu = torch.einsum("be,bek->bk", omega, membership)
    rho = torch.einsum("bqk,bek->bqe", alpha, membership)
    parent_masses = omega[:, None, :] * rho
    parent_response, parent_valid = mass_weighted_response(scores, values, parent_masses)

    safe_mu = mu.clamp_min(torch.finfo(mu.dtype).tiny)
    beta = (omega[:, :, None] * membership / safe_mu[:, None, :]).permute(0, 2, 1)
    beta = torch.where(mu[:, :, None] > 0.0, beta, torch.zeros_like(beta))
    # Every group receives the original node set; only its mass rule differs.
    grouped_masses = alpha[:, :, :, None] * mu[:, None, :, None] * beta[:, None, :, :]
    grouped_scores = scores[:, :, :, None, :].expand(-1, -1, -1, groups, -1)
    grouped_values = values[:, :, :, None, :, :].expand(-1, -1, -1, groups, -1, -1)
    grouped_scores = grouped_scores.reshape(batch, heads, queries, groups * sources)
    grouped_values = grouped_values.reshape(batch, heads, queries, groups * sources, width)
    grouped_response, grouped_valid = mass_weighted_response(
        grouped_scores,
        grouped_values,
        grouped_masses.reshape(batch, queries, groups * sources),
    )
    torch.testing.assert_close(grouped_masses.reshape(batch, queries, groups, sources).sum(dim=2), parent_masses, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(grouped_response, parent_response, rtol=1.0e-12, atol=1.0e-12)
    assert torch.equal(grouped_valid, parent_valid)


def test_bilinear_affine_interpolation_returns_executed_original_token_corners() -> None:
    environment = ChannelThermalEnvironmentBuilder()(
        batch_size=1,
        num_env_tokens_x=3,
        num_env_tokens_y=2,
        domain_length_x=3.0,
        domain_length_y=2.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    layout = environment.sampler_layout
    assert isinstance(layout, RegularGridLayout)
    source = (environment.env_coords[..., 0] + 2.0 * environment.env_coords[..., 1]).unsqueeze(1)
    points = torch.tensor(
        [[[[0.75, 0.75], [1.25, 1.25]], [[2.25, 0.75], [1.75, 1.25]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    sampled, corner_ids, corner_weights, cell_ids = interpolate_regular_grid(source, points, layout)
    expected = points[..., 0:1] + 2.0 * points[..., 1:2]
    torch.testing.assert_close(sampled[:, 0], expected[..., 0], rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(corner_weights.sum(dim=-1), torch.ones_like(corner_weights[..., 0]))
    assert corner_ids.shape == (1, 2, 2, 4)
    assert cell_ids.shape == (1, 2, 2)
    sampled.sum().backward()
    torch.testing.assert_close(points.grad[..., 0], torch.full_like(points.grad[..., 0], 1.0))
    torch.testing.assert_close(points.grad[..., 1], torch.full_like(points.grad[..., 1], 2.0))


def test_packed_channel_and_token_permutation_preserves_interpolation_identity() -> None:
    x_centers = torch.tensor([0.5, 1.5])
    y_centers = torch.tensor([0.5, 1.5])
    # Token 0 is (row=1,col=0), token 1 is (row=0,col=0), ... .  This is a
    # deliberate permutation of the row-major adapter order.
    token_to_grid = torch.tensor([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=torch.long)
    layout = RegularGridLayout.from_axis_bounds(
        axis_bounds=((0.0, 2.0), (0.0, 2.0)),
        centre_axes=(x_centers, y_centers),
        token_to_grid=token_to_grid,
        weights=torch.ones(4),
    )
    source = torch.stack(
        [torch.arange(4, dtype=torch.float32) + 100.0 * channel for channel in range(4)],
        dim=0,
    ).unsqueeze(0)
    points = torch.tensor(
        [[[[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]]],
        dtype=torch.float32,
    )
    sampled, corner_ids, corner_weights, _ = interpolate_regular_grid(source, points, layout)
    # Physical row-major points correspond to original token IDs 1, 3, 0, 2.
    expected_ids = torch.tensor([1, 3, 0, 2])
    expected = source[0, :, expected_ids].unsqueeze(0).unsqueeze(2)
    torch.testing.assert_close(sampled, expected)
    selected_ids = corner_ids.gather(-1, corner_weights.argmax(dim=-1, keepdim=True)).squeeze(-1)
    assert torch.equal(selected_ids[0, 0], expected_ids)
    torch.testing.assert_close(
        corner_weights.max(dim=-1).values,
        torch.ones_like(corner_weights[..., 0]),
    )


def test_projected_kv_head_channel_pack_round_trip_preserves_token_order() -> None:
    environment = ChannelThermalEnvironmentBuilder()(
        batch_size=1,
        num_env_tokens_x=3,
        num_env_tokens_y=2,
        domain_length_x=3.0,
        domain_length_y=2.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    layout = environment.sampler_layout
    assert isinstance(layout, RegularGridLayout)
    heads, tokens, head_dim = 2, int(environment.env_coords.shape[1]), 3
    key = torch.arange(heads * tokens * head_dim, dtype=torch.float64).reshape(
        1, heads, tokens, head_dim
    )
    packed = key.permute(0, 1, 3, 2).reshape(1, heads * head_dim, tokens)
    points = environment.env_coords[:, None, :, :]
    sampled, _, _, _ = interpolate_regular_grid(packed, points, layout)
    unpacked = sampled.reshape(1, heads, head_dim, 1, tokens).permute(0, 1, 3, 4, 2)
    torch.testing.assert_close(unpacked[:, :, 0], key)


def test_predicted_read_uses_24_sites_and_updates_sampler() -> None:
    torch.manual_seed(1408001)
    batch, _ = _batch()
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload())).train()
    assert isinstance(core.backend, HypergraphQuadratureField)
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    read = core.read(prepared, batch.query_xy, receiver_chunk_size=5, return_routing_maps=True)
    aux = read.interaction_aux
    assert aux["group_control_environment_sample_coordinates"].shape == (2, 5, 6, 4, 2)
    torch.testing.assert_close(
        aux["group_control_environment_sample_beta"].sum(dim=-1),
        torch.ones(2, 5, 6, dtype=torch.float32),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        aux["group_control_environment_sample_lambda"].sum(dim=(2, 3)),
        aux["group_control_environment_overlap_mass_per_query"],
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    assert float(aux["group_control_environment_fine_rows"]) == 2 * 5 * 24
    assert float(aux["group_control_environment_interpolation_corner_loads"]) == 2 * 5 * 24 * 4
    expected_channels = 2 * 16 + 6 * 4
    expected_bytes = 2 * 5 * 24 * expected_channels * 4
    assert float(aux["group_control_environment_sampled_bank_bytes"]) == expected_bytes
    assert int(aux["group_control_environment_interpolation_corner_indices"].max()) < 12

    optimizer = torch.optim.Adam(core.parameters(), lr=1.0e-3)
    before = core.backend.sample_mlp[-1].weight.detach().clone()
    loss = read.context.square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert core.backend.sample_mlp[-1].weight.grad is not None
    assert torch.isfinite(core.backend.sample_mlp[-1].weight.grad).all()
    optimizer.step()
    assert not torch.equal(before, core.backend.sample_mlp[-1].weight.detach())


def test_out_of_domain_receiver_reference_keeps_sample_sites_inside_grid() -> None:
    batch, layout = _batch(query_count=3)
    batch.query_xy = torch.tensor(
        [
            [[-0.2, 2.0], [8.2, 2.0], [4.0, -0.1]],
            [[4.0, 4.1], [-1.0, -1.0], [9.0, 5.0]],
        ],
        dtype=torch.float32,
    )
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload())).train()
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    read = core.read(prepared, batch.query_xy, return_routing_maps=True)
    coordinates = read.interaction_aux["group_control_environment_sample_coordinates"]
    hull = layout.center_hull.to(coordinates)
    assert torch.isfinite(read.context).all()
    assert bool((coordinates >= hull[0]).all())
    assert bool((coordinates <= hull[1]).all())


def test_chunked_read_adds_work_counts_and_keeps_peak_sampled_bank() -> None:
    torch.manual_seed(1408003)
    batch, _ = _batch(query_count=13)
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload())).eval()
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    read = core.read(
        prepared,
        batch.query_xy,
        receiver_chunk_size=5,
        return_routing_maps=True,
    )
    aux = read.interaction_aux
    rows = 2 * 13 * 24
    assert float(aux["group_control_environment_sample_slots"]) == rows
    assert float(aux["group_control_environment_fine_rows"]) == rows
    assert float(aux["group_control_environment_geometry_rows_forward"]) == rows
    assert float(aux["group_control_environment_content_dot_rows_forward"]) == rows * 4
    assert float(aux["group_control_environment_interpolation_corner_loads"]) == rows * 4
    expected_peak_bytes = 2 * 5 * 24 * (2 * 16 + 6 * 4) * 4
    assert float(aux["group_control_environment_sampled_bank_bytes"]) == expected_peak_bytes


def test_sampled_qe_checkpoint_boundary_runs_real_backward() -> None:
    torch.manual_seed(1408002)
    batch, _ = _batch(query_count=3)
    core = InterfaceFieldCore(
        UnifiedForwardConfig.from_dict(_payload(activation_checkpointing=True))
    ).train()
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    read = core.read(prepared, batch.query_xy, receiver_chunk_size=3, return_routing_maps=False)
    rows = read.interaction_aux["group_control_environment_checkpoint_recomputations"]
    assert float(rows) == 2 * 3 * 24
    expected_bytes = 2 * 3 * 24 * (2 * 16 + 6 * 4) * 4
    assert float(read.interaction_aux["group_control_environment_sampled_bank_bytes"]) == expected_bytes
    loss = read.context.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert core.backend.reference_anchor_logits.grad is not None
    assert torch.isfinite(core.backend.reference_anchor_logits.grad).all()


def test_quadrature_zero_group_mass_removes_environment_output_bias_exactly() -> None:
    torch.manual_seed(1408003)
    batch, _ = _batch(query_count=2)
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload())).eval()
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    receivers = batch.query_xy
    receiver_features = core._receiver_features(prepared, receivers)
    route = core.backend._route(
        prepared.backend_state,
        encoded,
        receivers,
        receiver_features,
    )
    zero_route = GroupQueryRoute(
        query_control=route.query_control,
        assignment=torch.zeros_like(route.assignment),
        logits=route.logits,
        query_keys=route.query_keys,
    )
    context, mass, _ = core.backend._quadrature_read(
        prepared.backend_state,
        encoded,
        receivers,
        receiver_features,
        zero_route,
        return_maps=True,
    )
    assert torch.equal(mass, torch.zeros_like(mass))
    assert torch.equal(context, torch.zeros_like(context))
