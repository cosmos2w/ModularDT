from __future__ import annotations

import math

import pytest
import torch

from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
from honf_forward_core.interface_fields.hierarchical_regional import HierarchicalRegionalField
from honf_forward_core.interface_fields.regional_response import RegionalResponseField
from honf_forward_core.interface_fields.response_hierarchy import (
    EnvironmentHierarchy,
    build_rectangular_environment_hierarchy,
    prepare_hierarchy_geometry,
    reduce_preupdate_statistics,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _grid(nx: int = 4, ny: int = 3, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    x = (torch.arange(nx, dtype=dtype) + 0.5) / float(nx)
    y = (torch.arange(ny, dtype=dtype) + 0.5) / float(ny)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def _encoded(
    *,
    batch: int = 2,
    modules: int = 3,
    nx: int = 4,
    ny: int = 3,
    hidden: int = 12,
    seed: int = 19,
    dtype: torch.dtype = torch.float64,
    coordinates: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> EncodedInterfaceCase:
    generator = torch.Generator().manual_seed(seed)
    base_coords = _grid(nx, ny, dtype=dtype) if coordinates is None else coordinates.to(dtype=dtype)
    if base_coords.ndim == 2:
        env_coords = base_coords.unsqueeze(0).expand(batch, -1, -1).clone()
    else:
        env_coords = base_coords
        batch = int(env_coords.shape[0])
    environment = int(env_coords.shape[1])
    hierarchy = build_rectangular_environment_hierarchy(
        env_coords,
        grid_shape=(nx, ny),
        block_shape=(2, 2),
        env_weights=weights,
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    env_weights = (
        torch.rand(batch, environment, generator=generator, dtype=dtype) + 0.2
        if weights is None
        else (weights.unsqueeze(0).expand(batch, -1) if weights.ndim == 1 else weights).to(dtype=dtype)
    )
    return EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, hidden, generator=generator, dtype=dtype),
        env_tokens=torch.randn(batch, environment, hidden, generator=generator, dtype=dtype),
        global_token=torch.randn(batch, hidden, generator=generator, dtype=dtype),
        module_centers=torch.rand(batch, modules, 2, generator=generator, dtype=dtype),
        env_coords=env_coords,
        module_present=torch.ones(batch, modules, dtype=dtype),
        module_features=torch.randn(batch, modules, 4, generator=generator, dtype=dtype),
        env_features=None,
        env_weights=env_weights,
        coordinate_scale=torch.ones(1, 1, 2, dtype=dtype),
        env_hierarchy=hierarchy,
    )


def _warm_pair(hidden: int = 12) -> tuple[DensePairwiseField, HierarchicalRegionalField, RegionalResponseField]:
    dense = DensePairwiseField(hidden, 8, 3, 2).double()
    hierarchical = HierarchicalRegionalField(hidden, 8, 3, 2).double()
    regional = RegionalResponseField(hidden, 8, 3, 2).double()
    encoded = _encoded(hidden=hidden, batch=1)
    states = encoded.module_tokens.detach()
    receivers = torch.rand(1, 5, 2, dtype=torch.float64)
    features = torch.randn(1, 5, hidden, dtype=torch.float64)
    with torch.no_grad():
        dense_state = dense.prepare(encoded, states)
        dense.read(dense_state, encoded, receivers, features)
        hierarchical_state = hierarchical.prepare(encoded, states)
        hierarchical.read(hierarchical_state, encoded, receivers, features)
        region_ids = encoded.env_hierarchy.parent_index[encoded.env_hierarchy.leaf_nodes]
        regional_state = regional.prepare(encoded, states, region_ids=region_ids)
        regional.read(regional_state, encoded, receivers, features)
    hierarchical.load_state_dict(dense.state_dict(), strict=True)
    regional.load_state_dict(dense.state_dict(), strict=True)
    return dense, hierarchical, regional


def _dense_environment_oracle(
    model: HierarchicalRegionalField,
    state: dict[str, torch.Tensor],
    encoded: EncodedInterfaceCase,
    receivers: torch.Tensor,
    receiver_features: torch.Tensor,
    incidence: object,
) -> torch.Tensor:
    """Evaluate the same sparse rows through an intentionally dense tensor."""

    batch, query_count, _ = receivers.shape
    node_count = int(state["tree_states"].shape[1])
    env_query = model.env_query(receiver_features)
    projected_query = model.env_attention.project_query(env_query)
    scores = torch.einsum("bhqd,bhnd->bhqn", projected_query, state["tree_keys"])
    scores = scores / (float(model.env_attention.head_dim) ** 0.5)
    relative = (receivers[:, :, None, :] - state["tree_coords"][:, None, :, :]) / encoded.coordinate_scale
    geometry_bias = model._mlp(
        model.env_geometry_bias,
        model.relative_fourier(relative),
    ).permute(0, 3, 1, 2)
    eta = receivers.new_zeros(batch, query_count, node_count)
    eta.index_put_((incidence.batch, incidence.query, incidence.node), incidence.eta, accumulate=True)
    log_measure = torch.log(eta.to(torch.float64)) + torch.log(state["tree_mass"][:, None, :].to(torch.float64))
    logits = scores.to(torch.float64) + geometry_bias.to(torch.float64) + log_measure[:, None, :, :]
    attention = torch.softmax(logits, dim=-1).to(dtype=state["tree_values"].dtype)
    aggregated = torch.einsum("bhqn,bhnd->bhqd", attention, state["tree_values"])
    aggregated = aggregated.permute(0, 2, 1, 3).reshape(batch, query_count, model.hidden_dim)
    return model.env_attention.output(aggregated)


def test_rectangular_builder_has_expected_levels_and_to_contract() -> None:
    hierarchy = build_rectangular_environment_hierarchy(
        _grid(24, 8),
        grid_shape=(24, 8),
        block_shape=(2, 2),
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    assert hierarchy.level_sizes == (192, 48, 12, 3, 2, 1)
    assert hierarchy.num_nodes == 258
    assert hierarchy.num_levels == 6
    assert hierarchy.to("cpu") is hierarchy
    expanded = hierarchy.expand_batch(3)
    assert expanded.fine_to_leaf.shape == (3, 192)
    assert expanded.bounds_min.shape == (3, 258, 2)


def test_bottom_up_statistics_preserve_duplicates_permutation_and_partial_blocks() -> None:
    base = _grid(3, 2)
    values = torch.arange(12, dtype=torch.float64).reshape(1, 6, 2)
    response = values.square()
    weights = torch.tensor([[1.0, 2.0, 0.5, 3.0, 1.5, 2.5]], dtype=torch.float64)
    reference_hierarchy = build_rectangular_environment_hierarchy(
        base,
        grid_shape=(3, 2),
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    reference = reduce_preupdate_statistics(
        reference_hierarchy, values, response, weights, base.unsqueeze(0)
    )
    cached_geometry = prepare_hierarchy_geometry(
        reference_hierarchy, weights, base.unsqueeze(0)
    )
    cached_reference = reduce_preupdate_statistics(
        reference_hierarchy,
        values,
        response,
        weights,
        base.unsqueeze(0),
        geometry=cached_geometry,
    )
    torch.testing.assert_close(cached_reference.mass, reference.mass)
    torch.testing.assert_close(cached_reference.coordinates, reference.coordinates)
    torch.testing.assert_close(cached_reference.environment, reference.environment)
    torch.testing.assert_close(cached_reference.response, reference.response)

    duplicated = base.repeat_interleave(2, dim=0)
    duplicated_values = values.repeat_interleave(2, dim=1)
    duplicated_response = response.repeat_interleave(2, dim=1)
    duplicated_weights = (weights / 2.0).repeat_interleave(2, dim=1)
    permutation = torch.tensor([7, 0, 5, 2, 10, 3, 1, 8, 6, 4, 9, 11])
    duplicate_hierarchy = build_rectangular_environment_hierarchy(
        duplicated[permutation],
        grid_shape=(3, 2),
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    duplicate = reduce_preupdate_statistics(
        duplicate_hierarchy,
        duplicated_values[:, permutation],
        duplicated_response[:, permutation],
        duplicated_weights[:, permutation],
        duplicated[permutation].unsqueeze(0),
    )
    torch.testing.assert_close(reference.mass, duplicate.mass)
    torch.testing.assert_close(reference.coordinates, duplicate.coordinates)
    torch.testing.assert_close(reference.environment, duplicate.environment)
    torch.testing.assert_close(reference.response, duplicate.response)
    assert duplicate_hierarchy.level_sizes == (6, 2, 1)

    padded_coords = torch.tensor(
        [[[0.2, 0.2], [float("nan"), float("nan")]]], dtype=torch.float64
    )
    padded_weights = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    padded_hierarchy = build_rectangular_environment_hierarchy(
        padded_coords,
        grid_shape=(2, 1),
        env_weights=padded_weights,
    )
    assert padded_hierarchy.node_valid[:2].tolist() == [True, False]
    assert torch.isfinite(padded_hierarchy.bounds_min).all()


def test_tree_mass_accounting_is_exact_for_inside_outside_and_transition_queries() -> None:
    encoded = _encoded(batch=2, nx=8, ny=4)
    model = HierarchicalRegionalField(12, 8, 3, 2).double()
    state = model.prepare(encoded, encoded.module_tokens)
    receivers = torch.tensor(
        [
            [[0.5, 0.5], [1.4, 0.5], [0.5, 1.0], [0.1, 0.1]],
            [[0.5, 0.5], [1.3, 0.5], [0.5, 1.0], [0.9, 0.9]],
        ],
        dtype=torch.float64,
    )
    incidence = model._traverse(state, receivers)
    segment = incidence.batch * receivers.shape[1] + incidence.query
    mass = state["tree_mass"][incidence.batch, incidence.node]
    recovered = receivers.new_zeros(receivers.shape[0] * receivers.shape[1])
    recovered.index_add_(0, segment, incidence.eta * mass)
    torch.testing.assert_close(
        recovered.reshape(receivers.shape[0], receivers.shape[1]),
        encoded.env_weights.sum(dim=1)[:, None].expand_as(recovered.reshape(receivers.shape[0], receivers.shape[1])),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.all(incidence.eta > 0)


def test_fixed_leaf_and_first_parent_reads_match_dense_and_regional_forward_and_gradients() -> None:
    dense, hierarchical, regional = _warm_pair()
    encoded = _encoded(batch=1)
    receivers = torch.rand(1, 7, 2, dtype=torch.float64)
    features = torch.randn(1, 7, 12, dtype=torch.float64)
    states_dense = encoded.module_tokens.detach().clone().requires_grad_()
    states_hierarchical = encoded.module_tokens.detach().clone().requires_grad_()
    dense_state = dense.prepare(encoded, states_dense)
    hierarchical_state = hierarchical.prepare(encoded, states_hierarchical)
    dense_output, _ = dense.read(dense_state, encoded, receivers, features)
    leaf_output, _ = hierarchical.read(
        hierarchical_state, encoded, receivers, features, resolution_level="leaf"
    )
    torch.testing.assert_close(leaf_output, dense_output, rtol=2e-12, atol=2e-12)
    dense_targets = [states_dense, *dense.parameters()]
    hierarchical_targets = [states_hierarchical, *hierarchical.parameters()]
    dense_grads = torch.autograd.grad(dense_output.sum(), dense_targets, allow_unused=True)
    hierarchical_grads = torch.autograd.grad(leaf_output.sum(), hierarchical_targets, allow_unused=True)
    assert len(dense_grads) == len(hierarchical_grads)
    for dense_grad, hierarchical_grad in zip(dense_grads, hierarchical_grads, strict=True):
        if dense_grad is None or hierarchical_grad is None:
            assert dense_grad is None and hierarchical_grad is None
        else:
            torch.testing.assert_close(hierarchical_grad, dense_grad, rtol=2e-11, atol=2e-12)

    region_ids = encoded.env_hierarchy.parent_index[encoded.env_hierarchy.leaf_nodes]
    states_regional = encoded.module_tokens.detach().clone().requires_grad_()
    states_parent = encoded.module_tokens.detach().clone().requires_grad_()
    regional_state = regional.prepare(encoded, states_regional, region_ids=region_ids)
    parent_state = hierarchical.prepare(encoded, states_parent)
    regional_output, _ = regional.read(regional_state, encoded, receivers, features)
    parent_output, _ = hierarchical.read(
        parent_state, encoded, receivers, features, fixed_level=1
    )
    torch.testing.assert_close(parent_output, regional_output, rtol=2e-12, atol=2e-12)
    regional_grads = torch.autograd.grad(
        regional_output.sum(), [states_regional, *regional.parameters()], allow_unused=True
    )
    parent_grads = torch.autograd.grad(
        parent_output.sum(), [states_parent, *hierarchical.parameters()], allow_unused=True
    )
    assert len(regional_grads) == len(parent_grads)
    for regional_grad, parent_grad in zip(regional_grads, parent_grads, strict=True):
        if regional_grad is None or parent_grad is None:
            assert regional_grad is None and parent_grad is None
        else:
            torch.testing.assert_close(parent_grad, regional_grad, rtol=2e-11, atol=2e-12)


def test_sparse_segmented_read_is_chunk_consistent_and_gathers_before_geometry_mlp(monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = _encoded(batch=1, nx=8, ny=4)
    model = HierarchicalRegionalField(12, 8, 3, 2).double()
    state = model.prepare(encoded, encoded.module_tokens)
    receivers = torch.rand(1, 13, 2, dtype=torch.float64)
    features = torch.randn(1, 13, 12, dtype=torch.float64)
    calls: list[int] = []
    original = model.env_geometry_bias.forward

    def counted(values: torch.Tensor) -> torch.Tensor:
        calls.append(int(values.shape[0]))
        return original(values)

    monkeypatch.setattr(model.env_geometry_bias, "forward", counted)
    whole, aux = model.read(state, encoded, receivers, features, return_routing_maps=True)
    chunks = [
        model.read(state, encoded, receivers[:, start : start + 4], features[:, start : start + 4])[0]
        for start in range(0, receivers.shape[1], 4)
    ]
    torch.testing.assert_close(whole, torch.cat(chunks, dim=1), rtol=2e-12, atol=2e-12)
    assert calls[0] == int(aux["hierarchical_incidence_rows"].item())
    assert calls[0] < receivers.shape[1] * state["tree_states"].shape[1]
    assert "hierarchical_incidence_node" in aux
    assert aux["hierarchical_incidence_node"].numel() == calls[0]


def test_sparse_segmented_read_matches_dense_attention_oracle_and_global_mass_rescaling() -> None:
    encoded = _encoded(batch=1, nx=4, ny=3)
    model = HierarchicalRegionalField(12, 8, 3, 2).double()
    state = model.prepare(encoded, encoded.module_tokens)
    receivers = torch.rand(1, 6, 2, dtype=torch.float64)
    features = torch.randn(1, 6, 12, dtype=torch.float64)
    incidence = model._traverse(state, receivers)
    sparse, _ = model.read_environment(state, encoded, receivers, features)
    dense = _dense_environment_oracle(model, state, encoded, receivers, features, incidence)
    torch.testing.assert_close(sparse, dense, rtol=2e-11, atol=2e-12)

    # With constant K/V, multiplying every source mass by one global factor
    # adds one constant to each receiver's logit segment and must leave the
    # normalized environmental read unchanged.
    constant_state = dict(state)
    constant_state["tree_keys"] = torch.zeros_like(state["tree_keys"])
    constant_state["tree_values"] = torch.ones_like(state["tree_values"])
    scaled_state = dict(constant_state)
    scaled_state["tree_mass"] = state["tree_mass"] * 37.0
    constant_read, _ = model.read_environment(constant_state, encoded, receivers, features)
    scaled_read, _ = model.read_environment(scaled_state, encoded, receivers, features)
    torch.testing.assert_close(scaled_read, constant_read, rtol=2e-12, atol=2e-12)


def test_activation_checkpointing_matches_output_input_and_parameter_gradients() -> None:
    encoded = _encoded(batch=1, modules=2, hidden=8, dtype=torch.float64)
    uncheckpointed = HierarchicalRegionalField(
        8, 6, 2, 2, activation_checkpointing=False
    ).double()
    checkpointed = HierarchicalRegionalField(
        8, 6, 2, 2, activation_checkpointing=True
    ).double()
    warm_states = encoded.module_tokens.detach()
    receivers = torch.rand(1, 5, 2, dtype=torch.float64)
    features = torch.randn(1, 5, 8, dtype=torch.float64)
    with torch.no_grad():
        uncheckpointed.read(
            uncheckpointed.prepare(encoded, warm_states), encoded, receivers, features
        )
        checkpointed.read(
            checkpointed.prepare(encoded, warm_states), encoded, receivers, features
        )
    checkpointed.load_state_dict(uncheckpointed.state_dict(), strict=True)
    uncheckpointed.train()
    checkpointed.train()

    states_off = warm_states.clone().requires_grad_()
    states_on = warm_states.clone().requires_grad_()
    receivers_off = receivers.clone().requires_grad_()
    receivers_on = receivers.clone().requires_grad_()
    features_off = features.clone().requires_grad_()
    features_on = features.clone().requires_grad_()
    output_off, _ = uncheckpointed.read(
        uncheckpointed.prepare(encoded, states_off),
        encoded,
        receivers_off,
        features_off,
    )
    output_on, _ = checkpointed.read(
        checkpointed.prepare(encoded, states_on),
        encoded,
        receivers_on,
        features_on,
    )
    torch.testing.assert_close(output_on, output_off, rtol=2e-11, atol=2e-12)
    targets_off = [states_off, receivers_off, features_off, *uncheckpointed.parameters()]
    targets_on = [states_on, receivers_on, features_on, *checkpointed.parameters()]
    gradients_off = torch.autograd.grad(
        output_off.square().mean(), targets_off, allow_unused=True
    )
    gradients_on = torch.autograd.grad(
        output_on.square().mean(), targets_on, allow_unused=True
    )
    assert len(gradients_off) == len(gradients_on)
    for gradient_off, gradient_on in zip(gradients_off, gradients_on, strict=True):
        if gradient_off is None or gradient_on is None:
            assert gradient_off is None and gradient_on is None
        else:
            torch.testing.assert_close(gradient_on, gradient_off, rtol=2e-10, atol=2e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke requires a CUDA device")
def test_activation_checkpointing_tiny_cuda_backward() -> None:
    device = torch.device("cuda:0")
    encoded_cpu = _encoded(batch=1, modules=1, hidden=8, dtype=torch.float32)
    hierarchy = encoded_cpu.env_hierarchy.to(device)
    env_tokens = encoded_cpu.env_tokens.to(device)
    env_coords = encoded_cpu.env_coords.to(device)
    env_weights = encoded_cpu.env_weights.to(device)
    encoded = EncodedInterfaceCase(
        module_tokens=encoded_cpu.module_tokens.to(device),
        env_tokens=env_tokens,
        global_token=encoded_cpu.global_token.to(device),
        module_centers=encoded_cpu.module_centers.to(device),
        env_coords=env_coords,
        module_present=encoded_cpu.module_present.to(device),
        module_features=encoded_cpu.module_features.to(device),
        env_features=None,
        env_weights=env_weights,
        coordinate_scale=encoded_cpu.coordinate_scale.to(device),
        env_hierarchy=hierarchy,
        env_hierarchy_geometry=prepare_hierarchy_geometry(hierarchy, env_weights, env_coords),
    )
    model = HierarchicalRegionalField(
        8, 6, 2, 2, activation_checkpointing=True
    ).to(device).train()
    module_states = encoded.module_tokens.detach().clone().requires_grad_()
    receivers = torch.rand(1, 8, 2, device=device, requires_grad=True)
    receiver_features = torch.randn(1, 8, 8, device=device, requires_grad=True)
    state = model.prepare(encoded, module_states)
    output, _ = model.read(state, encoded, receivers, receiver_features)
    output.square().mean().backward()
    assert output.isfinite().all()
    assert module_states.grad is not None and module_states.grad.isfinite().all()
    assert receivers.grad is not None and receivers.grad.isfinite().all()
    assert receiver_features.grad is not None and receiver_features.grad.isfinite().all()


def test_environment_and_module_permutation_padding_and_live_refresh() -> None:
    encoded = _encoded(batch=1, nx=4, ny=3)
    model = HierarchicalRegionalField(12, 8, 3, 2).double().eval()
    receivers = torch.rand(1, 9, 2, dtype=torch.float64)
    features = torch.randn(1, 9, 12, dtype=torch.float64)
    reference_state = model.prepare(encoded, encoded.module_tokens)
    reference, _ = model.read(reference_state, encoded, receivers, features)

    env_permutation = torch.tensor([5, 0, 8, 2, 11, 1, 9, 3, 6, 10, 4, 7])
    env_encoded = EncodedInterfaceCase(
        module_tokens=encoded.module_tokens,
        env_tokens=encoded.env_tokens[:, env_permutation],
        global_token=encoded.global_token,
        module_centers=encoded.module_centers,
        env_coords=encoded.env_coords[:, env_permutation],
        module_present=encoded.module_present,
        module_features=encoded.module_features,
        env_features=None,
        env_weights=encoded.env_weights[:, env_permutation],
        coordinate_scale=encoded.coordinate_scale,
        env_hierarchy=build_rectangular_environment_hierarchy(
            encoded.env_coords[:, env_permutation],
            grid_shape=(4, 3),
            bounds=((0.0, 1.0), (0.0, 1.0)),
        ),
    )
    permuted, _ = model.read(model.prepare(env_encoded, encoded.module_tokens), env_encoded, receivers, features)
    torch.testing.assert_close(permuted, reference, rtol=2e-12, atol=2e-12)

    padding = 2
    padded_coords = torch.cat([encoded.env_coords, torch.zeros(1, padding, 2, dtype=torch.float64)], dim=1)
    padded_tokens = torch.cat([encoded.env_tokens, torch.zeros(1, padding, 12, dtype=torch.float64)], dim=1)
    padded_weights = torch.cat([encoded.env_weights, torch.zeros(1, padding, dtype=torch.float64)], dim=1)
    padded_hierarchy = build_rectangular_environment_hierarchy(
        padded_coords,
        grid_shape=(4, 3),
        env_weights=padded_weights,
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    padded_encoded = EncodedInterfaceCase(
        module_tokens=encoded.module_tokens,
        env_tokens=padded_tokens,
        global_token=encoded.global_token,
        module_centers=encoded.module_centers,
        env_coords=padded_coords,
        module_present=encoded.module_present,
        module_features=encoded.module_features,
        env_features=None,
        env_weights=padded_weights,
        coordinate_scale=encoded.coordinate_scale,
        env_hierarchy=padded_hierarchy,
    )
    padded, _ = model.read(model.prepare(padded_encoded, encoded.module_tokens), padded_encoded, receivers, features)
    torch.testing.assert_close(padded, reference, rtol=2e-12, atol=2e-12)

    refreshed = model.prepare(encoded, encoded.module_tokens + 0.25)
    assert refreshed["tree_keys"].requires_grad
    assert not torch.allclose(reference_state["tree_keys"], refreshed["tree_keys"])


def test_transition_coordinate_derivative_is_finite_and_has_live_path() -> None:
    encoded = _encoded(batch=1, nx=4, ny=3)
    model = HierarchicalRegionalField(12, 8, 3, 2).double()
    state = model.prepare(encoded, encoded.module_tokens)
    features = torch.randn(1, 1, 12, dtype=torch.float64)
    # Check both compact-support endpoints and an interior opening point.  A
    # fixed central-difference step keeps the comparison sensitive to a lost
    # geometry gradient instead of merely checking finiteness.
    step = 1.0e-5
    for normalized_distance in (1.0, 1.5, 2.0):
        receiver = torch.tensor(
            [[[0.5 + normalized_distance / math.sqrt(2.0), 0.5]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        value, _ = model.read(state, encoded, receiver, features)
        gradient = torch.autograd.grad(value.sum(), receiver)[0]
        before = receiver.detach()
        delta = torch.tensor([[[step, 0.0]]], dtype=torch.float64)
        plus = model.read(state, encoded, before + delta, features)[0]
        minus = model.read(state, encoded, before - delta, features)[0]
        finite_difference = (plus - minus) / (2.0 * step)
        assert torch.isfinite(value).all()
        assert torch.isfinite(gradient).all()
        assert torch.isfinite(finite_difference).all()
        torch.testing.assert_close(
            gradient[..., 0], finite_difference.sum(dim=-1), rtol=4e-3, atol=4e-5
        )


def test_tiny_three_dimensional_hierarchy_executes_generic_reduction() -> None:
    x = torch.tensor([0.25, 0.75], dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), dim=-1).reshape(-1, 3)
    hierarchy = build_rectangular_environment_hierarchy(
        grid,
        grid_shape=(2, 2, 2),
        block_shape=(2, 2, 2),
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    assert hierarchy.level_sizes == (8, 1)
    environment = torch.randn(1, 8, 5, dtype=torch.float64)
    response = torch.randn(1, 8, 5, dtype=torch.float64)
    stats = reduce_preupdate_statistics(
        hierarchy,
        environment,
        response,
        torch.ones(1, 8, dtype=torch.float64),
        grid.unsqueeze(0),
    )
    assert stats.mass.shape == (1, 9)
    torch.testing.assert_close(stats.mass[:, -1], torch.tensor([8.0], dtype=torch.float64))
