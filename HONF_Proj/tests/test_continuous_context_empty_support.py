"""Closed contextual exchange retains dense zero-gradient optimizer behavior."""
import copy

import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.content_hypergraph.staged_attention_v2 import (
    SpatialRouteAttentionV2, safe_masked_softmax,
)
from honf_forward_core.continuous_joint.preparation import contextual_value_update
from honf_forward_core.continuous_joint.primitives import positive_quadratic_gate


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_closed_context_matches_literal_dense_gradients_and_adamw(dtype):
    torch.manual_seed(19)
    packed = SpatialRouteAttentionV2(32, 4, 2).to(dtype)
    dense = copy.deepcopy(packed)
    base = [torch.randn(1, 4, 3, 8, dtype=dtype) for _ in range(3)]
    allowed = ~torch.eye(3, dtype=torch.bool)[None]
    mask = torch.ones(1, 3, dtype=torch.bool)
    measure = torch.ones(1, 3, dtype=dtype)
    bias = torch.zeros(1, 4, 3, 3, dtype=dtype)
    ids = torch.arange(3)[None]

    def policy(scores, eligible, *args, **kwargs):
        # This exact-zero gate has zero derivative at the closed boundary.
        return positive_quadratic_gate(scores.mean(1) - 100) * eligible

    records = []
    for layer, literal in ((packed, False), (dense, True)):
        q, k, v = [x.clone().requires_grad_() for x in base]
        parameters = [*layer.parameters(), q, k, v]
        optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-2)
        if literal:
            scores = q @ k.transpose(-2, -1) + bias
            weights = safe_masked_softmax(scores, allowed[:, None], dim=-1)
            unnormalized = weights * policy(scores, allowed)[:, None]
            mass = unnormalized.sum(-1, keepdim=True)
            denominator = mass + 1e-3 * (1 - mass).square()
            coefficients = unnormalized / denominator
            values = torch.einsum('bhrd,bhdk->bhrk', coefficients, v)
            values = values.transpose(1, 2).reshape(1, 3, 32)
            output = F.linear(values, layer.output.weight, None)
            output = output + layer.output.bias * (mass / denominator).squeeze(-1).mean(1)[..., None]
        else:
            output = contextual_value_update(layer, q, k, v, bias, measure,
                allowed, mask, policy, route_name='mm', stage_identity='cycle0/mm',
                destination_ids=ids, donor_ids=ids)[0]
        assert output.count_nonzero() == 0
        output.sum().backward()
        gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
        optimizer.step()
        records.append((gradients, copy.deepcopy(optimizer.state_dict()),
                        [p.detach().clone() for p in parameters]))

    for left, right in zip(records[0][0], records[1][0]):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    for key, left in records[0][1]['state'].items():
        for name, value in left.items():
            torch.testing.assert_close(value, records[1][1]['state'][key][name], rtol=0, atol=0)
    assert records[0][1]['state'].keys() == records[1][1]['state'].keys()
    for left, right in zip(records[0][2], records[1][2]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
