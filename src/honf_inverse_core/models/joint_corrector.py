"""Optional single bounded consistency correction for the hierarchy.

Planned compact mechanism ``G``, generated physical design ``D``, frozen-HONF
realization ``G_hat``, request residuals ``R``, and context embedding ``c``
produce small deltas. The module is callable once only by the public facade;
it is not an optimizer, repair loop, or recurrent correction process.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .plan_flow import ConditionalPlanFlow, PLAN_CONTINUOUS_INDICES
from .rectified_flow import ResidualMLPBlock


@dataclass(frozen=True)
class CorrectionOutput:
    corrected_plan: torch.Tensor
    corrected_layout: torch.Tensor
    delta_plan: torch.Tensor
    delta_layout: torch.Tensor
    magnitude: torch.Tensor


class JointConsistencyCorrector(nn.Module):
    """Predict one bounded residual over fixed plan and layout tokens."""

    def __init__(
        self,
        *,
        num_edges: int,
        max_modules: int,
        condition_dim: int = 128,
        max_request_tokens: int = 4,
        hidden_dim: int = 256,
        blocks: int = 2,
        dropout: float = 0.05,
        max_plan_delta: float = 0.05,
        max_layout_delta: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_edges = int(num_edges)
        self.max_modules = int(max_modules)
        self.max_plan_delta = float(max_plan_delta)
        self.max_layout_delta = float(max_layout_delta)
        input_dim = self.num_edges * 24 + self.max_modules * 4 + max_request_tokens + condition_dim
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(int(blocks))]
        )
        self.plan_head = nn.Linear(hidden_dim, self.num_edges * len(PLAN_CONTINUOUS_INDICES))
        self.layout_head = nn.Linear(hidden_dim, self.max_modules * 3)

    def forward(
        self,
        planned_plan: torch.Tensor,
        layout: torch.Tensor,
        module_present: torch.Tensor,
        realized_plan: torch.Tensor,
        request_residuals: torch.Tensor,
        condition_embedding: torch.Tensor,
    ) -> CorrectionOutput:
        batch = planned_plan.shape[0]
        if planned_plan.shape[1:] != (self.num_edges, 12) or realized_plan.shape != planned_plan.shape:
            raise ValueError("Corrector planned/realized plans must share fixed [B,K,12].")
        if layout.shape[1:] != (self.max_modules, 3) or module_present.shape != layout.shape[:2]:
            raise ValueError("Corrector layout/presence must use fixed [B,M,3]/[B,M].")
        features = torch.cat(
            [
                planned_plan.reshape(batch, -1),
                realized_plan.reshape(batch, -1),
                layout.reshape(batch, -1),
                module_present.float(),
                request_residuals.float(),
                condition_embedding.float(),
            ],
            dim=-1,
        )
        hidden = self.blocks(torch.nn.functional.silu(self.input(features)))
        delta_plan = torch.tanh(self.plan_head(hidden)).reshape(
            batch, self.num_edges, len(PLAN_CONTINUOUS_INDICES)
        ) * self.max_plan_delta
        delta_layout = torch.tanh(self.layout_head(hidden)).reshape(batch, self.max_modules, 3)
        delta_layout = delta_layout * self.max_layout_delta * module_present.unsqueeze(-1)
        continuous = planned_plan[..., list(PLAN_CONTINUOUS_INDICES)] + delta_plan
        activity_logits = torch.where(
            planned_plan[..., 0] > 0.5,
            planned_plan.new_full((), 10.0),
            planned_plan.new_full((), -10.0),
        )
        corrected_plan = ConditionalPlanFlow.project_plan(continuous, activity_logits)
        corrected_layout = layout + delta_layout
        corrected_layout = torch.cat(
            [corrected_layout[..., :2].clamp(0.0, 1.0), corrected_layout[..., 2:3].clamp(-5.0, 5.0)],
            dim=-1,
        ) * module_present.unsqueeze(-1)
        magnitude = torch.sqrt(
            delta_plan.square().mean(dim=(1, 2)) + delta_layout.square().mean(dim=(1, 2))
        )
        return CorrectionOutput(corrected_plan, corrected_layout, delta_plan, delta_layout, magnitude)


__all__ = ["CorrectionOutput", "JointConsistencyCorrector"]
