"""Focused value/control boundary audit for live and frozen contextual routes."""
import math

import torch
from torch import nn

from honf_forward_core.content_hypergraph.staged_attention_v2 import SpatialRouteAttentionV2
from honf_forward_core.continuous_joint.incidence import ContinuousIncidencePolicy
from honf_forward_core.continuous_joint.preparation import FrozenContextPlan, contextual_value_update


class _ReturnPlan(nn.Module):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def forward(self, *args, **kwargs):
        return self.plan


def _context_update(layer, query, key, value, allowed, policy, ids):
    batch, _, receivers, _ = query.shape
    geometry_bias = query.new_zeros(batch, layer.heads, receivers, key.shape[2])
    donor_measure = query.new_ones(batch, key.shape[2])
    destination_mask = torch.ones(batch, receivers, dtype=torch.bool, device=query.device)
    return contextual_value_update(
        layer,
        query,
        key,
        value,
        geometry_bias,
        donor_measure,
        allowed,
        destination_mask,
        policy,
        route_name="mm",
        stage_identity="cycle0/mm",
        destination_ids=ids,
        donor_ids=ids,
    )


def test_live_incidence_control_can_change_cross_component_value_flow():
    """A changed key can open a live edge; a full frozen plan blocks it."""
    dtype = torch.float64
    ids = torch.tensor([[13, 29, 47, 53]])
    allowed = (~torch.eye(4, dtype=torch.bool))[None]
    query = torch.zeros((1, 1, 4, 2), dtype=dtype)
    query[..., 0] = 1.0
    key_before = torch.zeros_like(query)
    key_before[0, 0, :, 0] = torch.tensor([0.0, -1.0, 0.0, 1.0], dtype=dtype)
    key_after = torch.zeros_like(query)
    key_after[0, 0, :, 0] = torch.tensor([2.0, -1.0, 0.0, 1.0], dtype=dtype)
    # Values stay fixed across both executions: only the score/control input changes.
    value = torch.zeros_like(query)
    value[0, 0, :, 0] = torch.tensor([8.0, 0.0, 100.0, 5.0], dtype=dtype)

    layer = SpatialRouteAttentionV2(2, 1, 1).to(dtype=dtype).eval()
    with torch.no_grad():
        layer.output.weight.zero_()
        layer.output.weight[0, 0] = 1.0
        layer.output.bias.zero_()

    # The complete frozen plan gives receiver 47 only its within-component 53 value.
    frozen_incidence = torch.zeros((1, 4, 4), dtype=dtype)
    frozen_coefficients = torch.zeros((1, 4, 4, 1), dtype=dtype)
    frozen_incidence[0, 2, 3] = 1.0
    frozen_coefficients[0, 2, 3, 0] = 1.0
    frozen_mass = frozen_coefficients.sum(2)
    frozen = _ReturnPlan(FrozenContextPlan(frozen_incidence, frozen_coefficients, frozen_mass))
    frozen_before = _context_update(layer, query, key_before, value, allowed, frozen, ids)[0]
    frozen_after = _context_update(layer, query, key_after, value, allowed, frozen, ids)[0]
    assert torch.equal(frozen_before[0, 2], frozen_after[0, 2])
    torch.testing.assert_close(frozen_before[0, 2], torch.tensor([5.0, 0.0], dtype=dtype), rtol=0, atol=0)

    # Exercise the production calibrated incidence policy. For receiver 47,
    # key perturbation moves the positive gate from within-component 53 to 13.
    live = ContinuousIncidencePolicy(heads=1).to(dtype=dtype).eval()
    with torch.no_grad():
        calibration = live.routes["mm"]
        calibration.raw_scale.fill_(math.log(0.2 / 0.8))
        calibration.bias.fill_(0.1)
    live_before = _context_update(layer, query, key_before, value, allowed, live, ids)
    live_after = _context_update(layer, query, key_after, value, allowed, live, ids)
    before_gate, after_gate = live_before[2][0, 2], live_after[2][0, 2]
    assert before_gate[0] == 0 and after_gate[0] > 0
    assert before_gate[3] > 0 and after_gate[3] == 0
    assert not torch.equal(live_before[0][0, 2], live_after[0][0, 2])
