"""Layout `D` flow conditioned on `G,R,c`; verification later yields `G_hat`."""

from __future__ import annotations

import torch

from honf_inverse_core.models.layout_flow import ConditionalLayoutFlow
from honf_inverse_core.models.request_encoder import RequestSetEncoder
from honf_inverse_core.training.losses import layout_plan_alignment_loss
from tests.inverse_test_utils import request_batch


def test_layout_flow_generates_bounded_diverse_count_constrained_layouts() -> None:
    torch.manual_seed(3)
    encoder = RequestSetEncoder(hidden_dim=32, layers=1, dropout=0.0).eval()
    condition = encoder(request_batch(1), torch.zeros(1, 10), torch.zeros(1, 8))
    model = ConditionalLayoutFlow(
        num_edges=3, max_modules=6, condition_dim=32, hidden_dim=48,
        layers=2, dropout=0.0, sampling_steps=3,
    ).eval()
    plans = torch.rand(2, 3, 12)
    plans[..., 0] = 1.0
    repeated = type(condition)(
        condition.global_embedding.repeat(2, 1),
        condition.token_embeddings.repeat(2, 1, 1),
        condition.token_mask.repeat(2, 1),
    )
    geometry = torch.zeros(2, 8)
    geometry[:, 0] = 2.0 / 6.0
    geometry[:, 1] = 4.0 / 6.0
    sample = model.sample(plans, repeated, geometry_constraints=geometry, steps=3)
    assert sample.layout.shape == (2, 6, 3)
    assert torch.all((sample.module_count >= 2) & (sample.module_count <= 4))
    assert torch.all(sample.layout[..., :2] >= 0.0) and torch.all(sample.layout[..., :2] <= 1.0)
    assert not torch.allclose(sample.layout[0], sample.layout[1])
    for row in range(2):
        count = int(sample.module_count[row])
        assert torch.all(sample.module_present[row, :count] == 1)
        assert torch.all(sample.module_present[row, count:] == 0)
        assert torch.all(sample.layout[row, count:] == 0)


def test_layout_condition_preserves_canonical_edge_order() -> None:
    torch.manual_seed(17)
    encoder = RequestSetEncoder(hidden_dim=16, layers=1, dropout=0.0).eval()
    condition = encoder(request_batch(1), torch.zeros(1, 10), torch.zeros(1, 8))
    model = ConditionalLayoutFlow(
        num_edges=3, max_modules=4, condition_dim=16, hidden_dim=24,
        layers=1, dropout=0.0, sampling_steps=2,
    ).eval()
    plan = torch.rand(1, 3, 12)
    plan[..., 0] = 1.0
    state = torch.zeros(1, 4, 3)
    time = torch.full((1,), 0.5)
    canonical = model(state, time, plan, condition).velocity
    permuted = model(state, time, plan[:, [2, 0, 1]], condition).velocity
    assert not torch.allclose(canonical, permuted)


def test_layout_plan_alignment_is_finite_and_differentiable() -> None:
    endpoint = torch.rand(2, 4, 3, requires_grad=True)
    present = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    plan = torch.rand(2, 3, 12)
    plan[..., 0] = 1.0
    plan[..., 11] = plan[..., 11] / plan[..., 11].sum(dim=1, keepdim=True)
    loss = layout_plan_alignment_loss(endpoint, present, plan)
    loss.backward()
    assert torch.isfinite(loss)
    assert endpoint.grad is not None and torch.isfinite(endpoint.grad).all()
