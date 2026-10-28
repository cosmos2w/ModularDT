"""Native snapshots retain live FP32 gates at convex-mixture roundoff."""
from dataclasses import replace

import pytest
import torch

from honf_forward_core.content_hypergraph.staged_attention_v2 import SpatialRouteAttentionV2
from honf_forward_core.interaction_reused_member_routing.core import AppliedInteractionIncidence


def _capture(capture):
    torch.manual_seed(17)
    layer = SpatialRouteAttentionV2(8, 4, 2)
    policy = AppliedInteractionIncidence(4)
    policy.set_relative_ee_gate('relative_quadratic', 1.49)
    with torch.no_grad():
        policy.routes['ee'].head_logits.copy_(torch.tensor([0.021946797, -0.038930506, 0.152425826, 0.114050463]))
    states = torch.randn(1, 1, 8, requires_grad=True)
    coord = torch.zeros(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)
    ids = torch.zeros(1, 1, dtype=torch.long)
    result = layer(states, states, coord, coord, torch.ones_like(coord), torch.ones_like(coord),
                   torch.ones(1, 1), torch.ones(1), mask, mask, torch.ones(1, 2),
                   route_name='ee', destination_ids=ids, donor_ids=ids, stage_identity='cycle0/ee',
                   capture_native_route_snapshot=capture, value_policy=policy)
    output = result[0]
    grads = torch.autograd.grad(output.square().sum(), (states, *layer.parameters(), *policy.parameters()), allow_unused=True)
    return result, grads


def test_actual_snapshot_accepts_roundoff_without_modifying_forward_or_gradients():
    captured, actual_gradients = _capture(True)
    uncaptured, expected_gradients = _capture(False)
    gate = captured[2].applied_incidence
    assert gate.item() == 1 + torch.finfo(gate.dtype).eps
    torch.testing.assert_close(captured[0], uncaptured[0], rtol=0, atol=0)
    for actual, expected in zip(actual_gradients, expected_gradients):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for bad in (1.01, -0.01, float('nan')):
        with pytest.raises(ValueError, match='Invalid applied contextual incidence'):
            replace(captured[2], applied_incidence=torch.full_like(gate, bad))
    with pytest.raises(ValueError, match='Invalid applied contextual incidence'):
        replace(captured[2], allowed=torch.zeros_like(captured[2].allowed))

    binary = replace(captured[2], applied_incidence=torch.ones_like(gate, dtype=torch.bool))
    assert binary.applied_incidence.dtype == torch.bool
