"""Conditional plan `G` flow tests under `R,c`; `D/G_hat` are downstream."""

from __future__ import annotations

import torch

from honf_inverse_core.models.plan_flow import ConditionalPlanFlow
from honf_inverse_core.models.rectified_flow import flow_interpolation
from honf_inverse_core.models.request_encoder import RequestSetEncoder
from honf_inverse_core.training.losses import plan_training_losses
from tests.inverse_test_utils import request_batch


def test_plan_flow_shapes_losses_projection_and_request_sensitivity() -> None:
    torch.manual_seed(2)
    encoder = RequestSetEncoder(hidden_dim=32, layers=1, dropout=0.0).eval()
    request = request_batch(2)
    context = torch.zeros(2, 10)
    geometry = torch.zeros(2, 8)
    condition = encoder(request, context, geometry)
    model = ConditionalPlanFlow(
        num_edges=3, condition_dim=32, hidden_dim=48, layers=2, dropout=0.0, sampling_steps=3
    )
    target_plan = torch.rand(2, 3, 12)
    continuous = model.continuous_target(target_plan)
    state, velocity_target, time, noise = flow_interpolation(continuous)
    output = model(state, time, condition)
    endpoint = state + (1.0 - time[:, None, None]) * output.velocity
    losses = plan_training_losses(
        predicted_velocity=output.velocity,
        target_velocity=velocity_target,
        activity_logits=output.activity_logits,
        activity_target=torch.ones(2, 3),
        endpoint_estimate=endpoint,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    projected = model.project_plan(continuous)
    torch.testing.assert_close(projected[..., 5].sum(1), torch.ones(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(projected[..., 6].sum(1), torch.ones(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(projected[..., 10].sum(1), torch.ones(2), atol=1e-6, rtol=0)
    assert torch.equal(projected[..., 0], (projected[..., 7] > 0.05).float())
    logits = torch.tensor([[3.0, -3.0, -2.0], [-2.0, 4.0, -1.0]])
    gated = model.project_plan(continuous, logits)
    assert torch.equal(gated[..., 0].sum(dim=1), torch.ones(2))
    assert torch.equal(gated[..., 0], (gated[..., 7] > 0.05).float())

    shared_noise = noise[:1]
    first = model.sample(
        type(condition)(condition.global_embedding[:1], condition.token_embeddings[:1], condition.token_mask[:1]),
        steps=2,
        initial_noise=shared_noise,
    )
    changed_request = request_batch(1)
    changed_request["target_normalized"][:, 0] += 3.0
    changed_condition = encoder(changed_request, context[:1], geometry[:1])
    second = model.sample(changed_condition, steps=2, initial_noise=shared_noise)
    assert not torch.allclose(first.continuous_state, second.continuous_state)


def test_exchangeable_plan_velocity_is_permutation_equivariant_without_edge_embeddings() -> None:
    torch.manual_seed(269)
    model = ConditionalPlanFlow(
        num_edges=4,
        condition_dim=16,
        hidden_dim=32,
        layers=2,
        dropout=0.0,
        plan_token_mode="exchangeable_set",
        set_interaction_layers=2,
        set_attention_heads=4,
    ).eval()
    state = torch.randn(2, 4, model.state_dim)
    time = torch.tensor([0.2, 0.8])
    condition = torch.randn(2, 16)
    permutation = torch.tensor([2, 0, 3, 1])

    reference = model(state, time, condition)
    permuted = model(state[:, permutation], time, condition)

    torch.testing.assert_close(permuted.velocity, reference.velocity[:, permutation], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        permuted.activity_logits, reference.activity_logits[:, permutation], rtol=2e-6, atol=2e-6
    )
    assert not any("edge_embedding" in name for name, _ in model.named_parameters())


def test_exchangeable_plan_runtime_capacity_does_not_change_parameter_shapes() -> None:
    model = ConditionalPlanFlow(
        num_edges=4,
        condition_dim=16,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        plan_token_mode="exchangeable_set",
        set_attention_heads=4,
    ).eval()
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    model.set_edge_capacity(7)
    output = model(torch.randn(1, 7, model.state_dim), torch.zeros(1), torch.randn(1, 16))

    assert output.velocity.shape == (1, 7, model.state_dim)
    assert shapes == {name: tuple(value.shape) for name, value in model.state_dict().items()}
