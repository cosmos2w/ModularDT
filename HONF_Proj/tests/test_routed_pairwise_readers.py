"""Focused delayed fine-reader and two-hop dispatch checks."""

from __future__ import annotations

import copy

import torch

from honf_forward_core.interface_fields.routed_pairwise import (
    RoutedPairwiseField,
    compile_positive_pairs,
)
from honf_forward_core.interface_fields.routing_index.sparse_projection import (
    build_typed_source_incidence,
)
from honf_forward_core.interface_fields.routing_index.types import PackedPairs
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _pairs(
    batch: list[int], receiver: list[int], source: list[int], prior: list[float], *, raw: int | None = None
) -> PackedPairs:
    values = torch.tensor(prior, dtype=torch.float64)
    count = len(prior)
    return PackedPairs(
        torch.tensor(batch, dtype=torch.long),
        torch.tensor(receiver, dtype=torch.long),
        torch.tensor(source, dtype=torch.long),
        values,
        count if raw is None else raw,
        count,
    )


def test_two_hop_join_coalesces_duplicate_hub_paths_and_keeps_prior_live() -> None:
    weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    membership = torch.tensor(
        [[[0.6, 0.4], [0.0, 1.0], [1.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    incidence = build_typed_source_incidence(weights, membership)
    density = torch.tensor([[[0.5, 0.5], [1.0, 0.0]]], dtype=torch.float64)
    compiled = compile_positive_pairs(density, incidence)

    assert compiled.raw_path_count == 6
    assert compiled.unique_pair_count == 5
    actual = {
            (int(b), int(q), int(i)): float(w.detach())
        for b, q, i, w in zip(
            compiled.batch_index, compiled.receiver_index, compiled.source_index, compiled.prior, strict=True
        )
    }
    expected = {
        (0, 0, 0): 0.1,
        (0, 0, 1): 0.15,
        (0, 0, 2): 0.25,
        (0, 1, 0): 0.12,
        (0, 1, 2): 0.5,
    }
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert abs(actual[key] - value) < 1.0e-12
    compiled.prior.sum().backward()
    assert membership.grad is not None
    assert torch.isfinite(membership.grad).all()


def _encoded(*, hidden: int = 8, modules: int = 3, environment: int = 5) -> EncodedInterfaceCase:
    generator = torch.Generator().manual_seed(33)
    return EncodedInterfaceCase(
        module_tokens=torch.randn(1, modules, hidden, generator=generator, dtype=torch.float64),
        env_tokens=torch.randn(1, environment, hidden, generator=generator, dtype=torch.float64),
        global_token=torch.randn(1, hidden, generator=generator, dtype=torch.float64),
        module_centers=torch.rand(1, modules, 2, generator=generator, dtype=torch.float64),
        env_coords=torch.rand(1, environment, 2, generator=generator, dtype=torch.float64),
        module_present=torch.ones(1, modules, dtype=torch.float64),
        module_features=torch.randn(1, modules, 2, generator=generator, dtype=torch.float64),
        env_features=None,
        env_weights=torch.tensor([[0.5, 1.0, 2.0, 0.25, 0.75]], dtype=torch.float64),
        coordinate_scale=torch.ones(1, 1, 2, dtype=torch.float64),
    )


def test_gathered_module_reader_matches_same_prior_dense_arithmetic_oracle() -> None:
    torch.manual_seed(4)
    encoded = _encoded()
    backend = RoutedPairwiseField(8, 6, 2, 2, routing_config={"fine_pair_chunk_size": 2}).double().eval()
    states = encoded.module_tokens + 0.2
    state = {"module_tokens": states}
    receivers = torch.rand(1, 2, 2, dtype=torch.float64)
    pairs = _pairs([0, 0, 0], [0, 0, 1], [0, 1, 2], [0.2, 0.3, 0.5])
    selected = backend.read_module_pairs(state, encoded, receivers, pairs)

    relative = (receivers[:, :, None, :] - encoded.module_centers[:, None, :, :]) / encoded.coordinate_scale.reshape(-1)
    source = states[:, None, :, :].expand(-1, receivers.shape[1], -1, -1)
    global_values = encoded.global_token[:, None, None, :].expand(-1, receivers.shape[1], states.shape[1], -1)
    messages = backend.query_module_message(
        torch.cat([source, backend.relative_fourier(relative), global_values], dim=-1)
    )
    dense_prior = messages.new_zeros(1, 2, 3)
    dense_prior[0, pairs.receiver_index, pairs.source_index] = pairs.prior
    dense = backend.query_module_output(
        (messages * dense_prior[..., None]).sum(dim=2)
        * (encoded.module_present.sum(dim=1) / (1.0 + encoded.module_present.sum(dim=1)))[:, None, None]
    )
    torch.testing.assert_close(selected, dense, rtol=1.0e-11, atol=1.0e-12)


def test_uniform_module_pairs_match_dense_reader_and_input_gradients() -> None:
    """A fully selected uniform source-measure route is exactly Dense's QM reader."""

    encoded = _encoded()
    backend = RoutedPairwiseField(8, 6, 2, 2, routing_config={"fine_pair_chunk_size": 2}).double().eval()
    module_states = encoded.module_tokens.clone().requires_grad_()
    receivers = torch.rand(1, 2, 2, dtype=torch.float64, requires_grad=True)
    receiver_features = torch.randn(1, 2, 8, dtype=torch.float64)
    batch_index = [0] * (receivers.shape[1] * encoded.module_centers.shape[1])
    receiver_index = [q for q in range(receivers.shape[1]) for _ in range(encoded.module_centers.shape[1])]
    source_index = list(range(encoded.module_centers.shape[1])) * receivers.shape[1]
    pairs = _pairs(
        batch_index,
        receiver_index,
        source_index,
        [1.0 / encoded.module_centers.shape[1]] * len(source_index),
    )
    routed = backend.read_module_pairs(
        {"module_tokens": module_states}, encoded, receivers, pairs
    )
    dense = backend.read_module(
        {"module_tokens": module_states}, encoded, receivers, receiver_features
    )
    torch.testing.assert_close(routed, dense, rtol=1.0e-11, atol=1.0e-12)

    routed.sum().backward()
    routed_state_grad = module_states.grad.detach().clone()
    routed_receiver_grad = receivers.grad.detach().clone()
    module_states.grad.zero_()
    receivers.grad.zero_()
    dense.sum().backward()
    torch.testing.assert_close(module_states.grad, routed_state_grad, rtol=1.0e-11, atol=1.0e-12)
    torch.testing.assert_close(receivers.grad, routed_receiver_grad, rtol=1.0e-11, atol=1.0e-12)


def test_uniform_environment_pairs_match_dense_reader_and_input_gradients() -> None:
    """A fully selected unit-prior route is exactly Dense's QE reader."""

    encoded = _encoded()
    encoded = EncodedInterfaceCase(
        **{**vars(encoded), "env_weights": torch.ones_like(encoded.env_weights)}
    )
    backend = RoutedPairwiseField(8, 6, 2, 2, routing_config={"fine_pair_chunk_size": 2}).double().eval()
    env_tokens = encoded.env_tokens.clone().requires_grad_()
    receivers = torch.rand(1, 2, 2, dtype=torch.float64, requires_grad=True)
    receiver_features = torch.randn(1, 2, 8, dtype=torch.float64, requires_grad=True)
    keys, values = backend.project_environment_sources(env_tokens)
    batch_index = [0] * (receivers.shape[1] * encoded.env_coords.shape[1])
    receiver_index = [q for q in range(receivers.shape[1]) for _ in range(encoded.env_coords.shape[1])]
    source_index = list(range(encoded.env_coords.shape[1])) * receivers.shape[1]
    pairs = _pairs(batch_index, receiver_index, source_index, [1.0] * len(source_index))
    routed = backend.read_environment_pairs(
        {"env_tokens": env_tokens, "env_keys": keys, "env_values": values},
        encoded,
        receivers,
        receiver_features,
        pairs,
    )
    dense, _ = backend.read_environment(
        {"env_tokens": env_tokens}, encoded, receivers, receiver_features
    )
    torch.testing.assert_close(routed, dense, rtol=1.0e-11, atol=1.0e-12)

    routed.sum().backward()
    routed_env_grad = env_tokens.grad.detach().clone()
    routed_receiver_grad = receivers.grad.detach().clone()
    routed_feature_grad = receiver_features.grad.detach().clone()
    env_tokens.grad.zero_()
    receivers.grad.zero_()
    receiver_features.grad.zero_()
    dense.sum().backward()
    torch.testing.assert_close(env_tokens.grad, routed_env_grad, rtol=1.0e-11, atol=1.0e-12)
    torch.testing.assert_close(receivers.grad, routed_receiver_grad, rtol=1.0e-11, atol=1.0e-12)
    torch.testing.assert_close(receiver_features.grad, routed_feature_grad, rtol=1.0e-11, atol=1.0e-12)


def test_activation_checkpointed_fine_tiles_match_eager_multitile_backward() -> None:
    """Checkpointing whole fine tiles preserves values and input derivatives."""

    encoded = _encoded()
    eager = RoutedPairwiseField(
        8, 6, 2, 2, routing_config={"fine_pair_chunk_size": 2}, activation_checkpointing=False
    ).double().train()
    # Initialize all lazy readers before copying, so both backends have
    # identical concrete module widths and parameters.
    warm_module = encoded.module_tokens.clone().requires_grad_()
    warm_env = encoded.env_tokens.clone().requires_grad_()
    warm_receivers = torch.rand(1, 2, 2, dtype=torch.float64)
    warm_features = torch.randn(1, 2, 8, dtype=torch.float64)
    all_module_pairs = _pairs(
        [0] * 6, [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [1.0] * 6
    )
    all_environment_pairs = _pairs(
        [0] * 10,
        [q for q in range(2) for _ in range(5)],
        list(range(5)) * 2,
        [1.0] * 10,
    )
    input_receivers = torch.rand(1, 2, 2, dtype=torch.float64)
    input_features = torch.randn(1, 2, 8, dtype=torch.float64)
    eager.read_module_pairs({"module_tokens": warm_module}, encoded, warm_receivers, all_module_pairs)
    eager.read_environment_pairs(
        {"env_tokens": warm_env}, encoded, warm_receivers, warm_features, all_environment_pairs
    )
    checkpointed = copy.deepcopy(eager)
    checkpointed.activation_checkpointing = True

    def run(model: RoutedPairwiseField):
        module_states = encoded.module_tokens.clone().requires_grad_()
        env_tokens = encoded.env_tokens.clone().requires_grad_()
        receivers = input_receivers.clone().requires_grad_()
        receiver_features = input_features.clone().requires_grad_()
        keys, values = model.project_environment_sources(env_tokens)
        module_context = model.read_module_pairs(
            {"module_tokens": module_states}, encoded, receivers, all_module_pairs
        )
        environment_context = model.read_environment_pairs(
            {"env_tokens": env_tokens, "env_keys": keys, "env_values": values},
            encoded,
            receivers,
            receiver_features,
            all_environment_pairs,
        )
        output = module_context + environment_context
        gradients = torch.autograd.grad(
            output.square().sum(), (module_states, env_tokens, receivers, receiver_features)
        )
        return output.detach(), tuple(value.detach() for value in gradients)

    eager_output, eager_gradients = run(eager)
    checkpointed_output, checkpointed_gradients = run(checkpointed)
    torch.testing.assert_close(checkpointed_output, eager_output, rtol=1.0e-11, atol=1.0e-12)
    for checkpointed_gradient, eager_gradient in zip(checkpointed_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(checkpointed_gradient, eager_gradient, rtol=1.0e-11, atol=1.0e-12)


def test_gathered_environment_reader_matches_same_prior_dense_oracle_and_is_tiled() -> None:
    torch.manual_seed(5)
    encoded = _encoded()
    backend = RoutedPairwiseField(8, 6, 2, 2, routing_config={"fine_pair_chunk_size": 2}).double().eval()
    state = {"env_tokens": encoded.env_tokens + 0.15}
    keys, values = backend.project_environment_sources(state["env_tokens"])
    state.update(env_keys=keys, env_values=values)
    receivers = torch.rand(1, 2, 2, dtype=torch.float64)
    # Source 2 is intentionally omitted.  The reader must never evaluate its
    # fine geometry/content rows, while the dense oracle below uses the same
    # positive sparse prior and mask.
    pairs = _pairs([0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 3], [0.2, 0.8, 0.6, 0.4])
    routed = backend.read_environment_pairs(state, encoded, receivers, backend.relative_fourier(receivers), pairs)

    # Build the dense arithmetic oracle directly.  Dense's public projected
    # reader accepts one source mass per source, whereas routed priors are
    # receiver/source specific after the two-hop join.
    prior = receivers.new_zeros(1, 2, encoded.env_coords.shape[1])
    prior[0, pairs.receiver_index, pairs.source_index] = pairs.prior
    receiver_features = backend.relative_fourier(receivers)
    query = backend.env_attention.project_query(backend.env_query(receiver_features))
    source_relative = (
        receivers[:, :, None, :] - encoded.env_coords[:, None, :, :]
    ) / encoded.coordinate_scale
    bias = backend.env_geometry_bias(
        backend.relative_fourier(source_relative)
    ).permute(0, 3, 1, 2)
    scores = torch.matmul(query, keys.transpose(-1, -2)) / (float(backend.env_attention.head_dim) ** 0.5)
    scores = scores + bias + torch.log(prior).clamp_min(torch.finfo(prior.dtype).min)[:, None]
    valid = prior[:, None] > 0.0
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    weights = weights * valid.to(weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    dense_values = torch.matmul(weights, values).transpose(1, 2).reshape(1, 2, -1)
    oracle = backend.env_attention.output(dense_values)
    torch.testing.assert_close(routed, oracle, rtol=1.0e-11, atol=1.0e-12)

    # Re-register with a mutable closure after the arithmetic comparison so
    # the assertion counts only the routed reader calls.
    counter = {"rows": 0}

    def count_rows(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        counter["rows"] += int(inputs[0].shape[0])

    handle = backend.env_geometry_bias.register_forward_hook(count_rows)
    backend.read_environment_pairs(state, encoded, receivers, backend.relative_fourier(receivers), pairs)
    handle.remove()
    assert counter["rows"] == pairs.unique_pair_count


def test_fine_reader_preserves_prior_gradient() -> None:
    encoded = _encoded()
    backend = RoutedPairwiseField(8, 6, 2, 2, routing_config={"fine_pair_chunk_size": 1}).double().eval()
    state = {"module_tokens": encoded.module_tokens}
    receivers = torch.rand(1, 1, 2, dtype=torch.float64)
    prior = torch.tensor([0.25, 0.75], dtype=torch.float64, requires_grad=True)
    pairs = PackedPairs(
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        prior,
        2,
        2,
    )
    value = backend.read_module_pairs(state, encoded, receivers, pairs).sum()
    value.backward()
    assert prior.grad is not None
    assert torch.isfinite(prior.grad).all()
    assert torch.count_nonzero(prior.grad) > 0
