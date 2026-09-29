"""Check native direct hard values and the isolated scorer shadow."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from honf_forward_core.interface_fields.native_direct_pair import hard_value_soft_direct_forward


@dataclass(frozen=True)
class _Encoded:
    module_present: torch.Tensor


@dataclass(frozen=True)
class _DirectPlan:
    weight: torch.Tensor


class _Backend(nn.Module):
    optional_native_policy = True
    cover_mode = "external"
    cover_executor = "dense_masked"
    activation_checkpointing = False


class _TinyPhysicalCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(forward_architecture="dense_pairwise_field")
        self.backend = _Backend()
        self.physical_weight = nn.Parameter(torch.tensor(2.0))

    def prepare(self, _encoded, states, *, fixed_cover_plans):
        return {"states": states, "plan": fixed_cover_plans[0]}

    def decode_queries(self, prepared, query_coords, _query_features, *, receiver_chunk_size):
        del receiver_chunk_size
        state = prepared["states"][:, :1, :1]
        return {
            "pred_field": self.physical_weight * (
                state + query_coords[:, :1, :1] + prepared["plan"].weight
            )
        }


def test_direct_shadow_uses_hard_values_without_soft_physical_gradient() -> None:
    core = _TinyPhysicalCore()
    state = torch.tensor([[[0.4]]], requires_grad=True)
    query = torch.tensor([[[0.7]]], requires_grad=True)
    score = nn.Parameter(torch.tensor(-0.3))
    hard = _DirectPlan(torch.tensor(0.0))
    soft = _DirectPlan(torch.sigmoid(score))
    result = hard_value_soft_direct_forward(
        core,
        _Encoded(torch.ones(1, 1)),
        state,
        (hard,),
        (soft,),
        query,
    )
    torch.testing.assert_close(result.prediction, result.hard_prediction, rtol=0, atol=0)
    result.prediction.sum().backward()
    torch.testing.assert_close(core.physical_weight.grad, torch.tensor(1.1))
    torch.testing.assert_close(state.grad, torch.full_like(state, 2.0))
    torch.testing.assert_close(query.grad, torch.full_like(query, 2.0))
    assert score.grad is not None and float(score.grad) > 0.0
