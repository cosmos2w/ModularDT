"""Conditional padded-slot rectified flow for modular physical layouts.

Compact plan ``G``, request ``R``, and context ``c`` condition generation of
physical design ``D`` as normalized centers/heat plus presence/count heads.
Frozen verification later yields realized ``G_hat``. Independent layout noise
allows several layouts to realize the same sampled mechanism without search.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .rectified_flow import ResidualMLPBlock, SinusoidalTimeEmbedding, integrate_rectified_flow
from .request_encoder import RequestEncoding


LAYOUT_STATE_DIM = 3


@dataclass(frozen=True)
class LayoutFlowOutput:
    velocity: torch.Tensor
    presence_logits: torch.Tensor
    count_logits: torch.Tensor


@dataclass(frozen=True)
class SampledLayout:
    layout: torch.Tensor
    module_present: torch.Tensor
    module_count: torch.Tensor
    continuous_state: torch.Tensor
    presence_logits: torch.Tensor
    count_logits: torch.Tensor


class ConditionalLayoutFlow(nn.Module):
    """Predict velocity over `[B,M,3]` center-x/center-y/heat layout states."""

    def __init__(
        self,
        *,
        num_edges: int,
        max_modules: int = 12,
        condition_dim: int = 128,
        hidden_dim: int = 256,
        layers: int = 4,
        dropout: float = 0.05,
        sampling_steps: int = 24,
    ) -> None:
        super().__init__()
        self.num_edges = int(num_edges)
        self.max_modules = int(max_modules)
        self.sampling_steps = int(sampling_steps)
        if self.num_edges <= 0 or self.max_modules <= 0:
            raise ValueError("Layout flow requires positive fixed K and M.")
        self.plan_projection = nn.Linear(12, hidden_dim)
        # The compact-plan ABI has a stable canonical edge order. Retain a
        # pooled content path for robustness, but also expose the ordered full
        # plan so q(D|G,R,c) can distinguish mechanisms whose edge sets have
        # similar averages but different source/region assignments.
        self.ordered_plan_projection = nn.Linear(self.num_edges * 12, hidden_dim)
        self.plan_blocks = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout), ResidualMLPBlock(hidden_dim, dropout)
        )
        self.request_projection = nn.Linear(condition_dim, hidden_dim)
        self.state_projection = nn.Linear(LAYOUT_STATE_DIM, hidden_dim)
        self.slot_embedding = nn.Embedding(self.max_modules, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(int(layers))]
        )
        self.velocity_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, LAYOUT_STATE_DIM))
        self.presence_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.max_modules + 1)
        )

    def encode_plan(self, compact_plan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if compact_plan.ndim != 3 or compact_plan.shape[1:] != (self.num_edges, 12):
            raise ValueError(f"compact_plan must have shape [B,{self.num_edges},12].")
        tokens = self.plan_blocks(self.plan_projection(compact_plan.float()))
        weights = compact_plan[..., 0:1].float()
        denominator = weights.sum(dim=1).clamp_min(1.0)
        pooled = (tokens * weights).sum(dim=1) / denominator
        pooled = pooled + self.ordered_plan_projection(compact_plan.float().reshape(compact_plan.shape[0], -1))
        return pooled, tokens

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        compact_plan: torch.Tensor,
        condition: RequestEncoding | torch.Tensor,
    ) -> LayoutFlowOutput:
        if state.ndim != 3 or state.shape[1:] != (self.max_modules, LAYOUT_STATE_DIM):
            raise ValueError(f"Layout state must have shape [B,{self.max_modules},3].")
        request_global = condition.global_embedding if isinstance(condition, RequestEncoding) else condition
        plan_global, _ = self.encode_plan(compact_plan)
        global_condition = plan_global + self.request_projection(request_global)
        slots = torch.arange(self.max_modules, device=state.device)
        hidden = self.state_projection(state)
        hidden = hidden + self.slot_embedding(slots).unsqueeze(0)
        hidden = hidden + self.time_embedding(time).unsqueeze(1)
        hidden = hidden + global_condition.unsqueeze(1)
        hidden = self.blocks(hidden)
        presence = self.presence_head(hidden).squeeze(-1)
        pooled_slots = hidden.mean(dim=1)
        count = self.count_head(torch.cat([pooled_slots, global_condition], dim=-1))
        return LayoutFlowOutput(self.velocity_head(hidden), presence, count)

    def project_layout(
        self,
        continuous: torch.Tensor,
        presence_logits: torch.Tensor,
        count_logits: torch.Tensor,
        geometry_constraints: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one bounded endpoint projection and canonical active-first sort."""

        if continuous.shape[1:] != (self.max_modules, 3):
            raise ValueError("continuous layout has the wrong fixed slot shape.")
        batch = continuous.shape[0]
        count = count_logits.argmax(dim=-1)
        if geometry_constraints is not None:
            minimum = torch.round(geometry_constraints[:, 0] * self.max_modules).long()
            maximum = torch.round(geometry_constraints[:, 1] * self.max_modules).long()
            count = torch.minimum(torch.maximum(count, minimum), maximum)
        count = count.clamp(0, self.max_modules)
        layout = continuous.clone()
        layout[..., :2] = layout[..., :2].clamp(0.0, 1.0)
        layout[..., 2] = layout[..., 2].clamp(-5.0, 5.0)
        present = torch.zeros_like(presence_logits)
        for row in range(batch):
            chosen = torch.topk(presence_logits[row], k=int(count[row]), largest=True).indices if int(count[row]) else []
            if len(chosen):
                present[row, chosen] = 1.0
        canonical_layout = torch.zeros_like(layout)
        canonical_present = torch.zeros_like(present)
        for row in range(batch):
            active = torch.nonzero(present[row] > 0.5, as_tuple=False).reshape(-1).tolist()
            active = sorted(
                active,
                key=lambda index: (
                    float(layout[row, index, 0]), float(layout[row, index, 1]),
                    float(layout[row, index, 2]), int(index),
                ),
            )
            if active:
                indices = torch.as_tensor(active, device=layout.device, dtype=torch.long)
                canonical_layout[row, : len(active)] = layout[row, indices]
                canonical_present[row, : len(active)] = 1.0
        return canonical_layout, canonical_present, count

    @torch.no_grad()
    def sample(
        self,
        compact_plan: torch.Tensor,
        condition: RequestEncoding | torch.Tensor,
        *,
        geometry_constraints: torch.Tensor | None = None,
        steps: int | None = None,
        method: str = "heun",
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> SampledLayout:
        batch = compact_plan.shape[0]
        initial = initial_noise
        if initial is None:
            initial = torch.randn(
                (batch, self.max_modules, LAYOUT_STATE_DIM),
                device=compact_plan.device,
                dtype=compact_plan.dtype,
                generator=generator,
            )
        elif initial.shape != (batch, self.max_modules, LAYOUT_STATE_DIM):
            raise ValueError("initial_noise has the wrong layout-flow shape.")

        def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return self.forward(state, time, compact_plan, condition).velocity

        endpoint = integrate_rectified_flow(
            velocity, initial, steps=self.sampling_steps if steps is None else int(steps), method=method
        )
        final = self.forward(
            endpoint,
            torch.ones(batch, device=endpoint.device, dtype=endpoint.dtype),
            compact_plan,
            condition,
        )
        layout, present, count = self.project_layout(
            endpoint, final.presence_logits, final.count_logits, geometry_constraints
        )
        return SampledLayout(layout, present, count, endpoint, final.presence_logits, final.count_logits)


__all__ = ["ConditionalLayoutFlow", "LAYOUT_STATE_DIM", "LayoutFlowOutput", "SampledLayout"]
