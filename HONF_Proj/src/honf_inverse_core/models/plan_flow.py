"""Conditional fixed-edge rectified flow for compact mechanisms.

Request ``R`` and context ``c`` condition Gaussian paths to compact plan ``G``.
Physical layout ``D`` is generated downstream, and frozen verification later
produces realized ``G_hat``. Independent noise exposes multiple plausible
mechanisms for the same request. Canonical edge positions are never permuted
during training; sampled plans are projected and canonically sorted once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .rectified_flow import ResidualMLPBlock, SinusoidalTimeEmbedding, integrate_rectified_flow
from .request_encoder import RequestEncoding


PLAN_FEATURE_DIM = 12
PLAN_CONTINUOUS_INDICES = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11)


@dataclass(frozen=True)
class PlanFlowOutput:
    velocity: torch.Tensor
    activity_logits: torch.Tensor


@dataclass(frozen=True)
class SampledPlan:
    compact_plan: torch.Tensor
    continuous_state: torch.Tensor
    activity_logits: torch.Tensor


class SetInteractionBlock(nn.Module):
    """Permutation-equivariant self-attention and shared token feed-forward block."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % int(heads) != 0:
            raise ValueError("Set-attention hidden_dim must be divisible by heads.")
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, int(heads), dropout=float(dropout), batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return tokens + self.feed_forward(self.output_norm(tokens))


class ConditionalPlanFlow(nn.Module):
    """Predict velocity over `[B,K,10]` independent compact-plan attributes."""

    def __init__(
        self,
        *,
        num_edges: int,
        condition_dim: int = 128,
        hidden_dim: int = 256,
        layers: int = 4,
        dropout: float = 0.05,
        sampling_steps: int = 24,
        plan_token_mode: str = "indexed",
        set_interaction_layers: int = 2,
        set_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if num_edges <= 0:
            raise ValueError("ConditionalPlanFlow requires positive num_edges.")
        self.num_edges = int(num_edges)
        self.state_dim = len(PLAN_CONTINUOUS_INDICES)
        self.sampling_steps = int(sampling_steps)
        self.plan_token_mode = str(plan_token_mode)
        if self.plan_token_mode not in {"indexed", "exchangeable_set"}:
            raise ValueError("plan_token_mode must be 'indexed' or 'exchangeable_set'.")
        if self.plan_token_mode == "indexed":
            self.edge_embedding = nn.Embedding(self.num_edges, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.input_projection = nn.Linear(self.state_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(int(layers))]
        )
        if self.plan_token_mode == "exchangeable_set":
            if int(set_interaction_layers) <= 0:
                raise ValueError("exchangeable_set requires positive set_interaction_layers.")
            self.set_interactions = nn.ModuleList(
                [
                    SetInteractionBlock(hidden_dim, int(set_attention_heads), dropout)
                    for _ in range(int(set_interaction_layers))
                ]
            )
        self.velocity_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.state_dim))
        self.activity_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def set_edge_capacity(self, capacity: int) -> None:
        """Change runtime set capacity without changing exchangeable parameters."""

        if int(capacity) <= 0:
            raise ValueError("Plan-flow edge capacity must be positive.")
        if self.plan_token_mode != "exchangeable_set" and int(capacity) != self.num_edges:
            raise ValueError("indexed plan flow has a fixed configured edge count.")
        self.num_edges = int(capacity)

    @staticmethod
    def continuous_target(compact_plan: torch.Tensor) -> torch.Tensor:
        if compact_plan.ndim != 3 or compact_plan.shape[-1] != PLAN_FEATURE_DIM:
            raise ValueError("compact_plan must have shape [B,K,12].")
        return compact_plan[..., list(PLAN_CONTINUOUS_INDICES)]

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: RequestEncoding | torch.Tensor,
    ) -> PlanFlowOutput:
        if state.ndim != 3 or state.shape[-1] != self.state_dim:
            raise ValueError(f"Plan flow state must have shape [B,K,{self.state_dim}].")
        if self.plan_token_mode == "indexed" and state.shape[1] != self.num_edges:
            raise ValueError(f"Plan flow state must have shape [B,{self.num_edges},{self.state_dim}].")
        if self.plan_token_mode == "exchangeable_set" and state.shape[1] <= 0:
            raise ValueError("Exchangeable plan flow requires at least one runtime token.")
        global_condition = condition.global_embedding if isinstance(condition, RequestEncoding) else condition
        hidden = self.input_projection(state)
        if self.plan_token_mode == "indexed":
            edge_ids = torch.arange(self.num_edges, device=state.device)
            hidden = hidden + self.edge_embedding(edge_ids).unsqueeze(0)
        hidden = hidden + self.time_embedding(time).unsqueeze(1)
        hidden = hidden + self.condition_projection(global_condition).unsqueeze(1)
        hidden = self.blocks(hidden)
        if self.plan_token_mode == "exchangeable_set":
            for interaction in self.set_interactions:
                hidden = interaction(hidden)
        return PlanFlowOutput(
            velocity=self.velocity_head(hidden),
            activity_logits=self.activity_head(hidden).squeeze(-1),
        )

    @staticmethod
    def project_plan(continuous: torch.Tensor, activity_logits: torch.Tensor | None = None) -> torch.Tensor:
        """Create schema-valid normalized `[B,K,12]`, deriving strength/activity."""

        if continuous.ndim != 3 or continuous.shape[-1] != len(PLAN_CONTINUOUS_INDICES):
            raise ValueError("continuous plan state must have shape [B,K,10].")
        _, edges, _ = continuous.shape
        coordinates = continuous[..., 0:4].clamp(0.0, 1.0)
        masses_positive = continuous[..., 4:6].clamp_min(1.0e-6)
        fractions_positive = continuous[..., 8:10].clamp_min(1.0e-6)
        if activity_logits is not None:
            if activity_logits.shape != continuous.shape[:2]:
                raise ValueError("activity_logits must have shape [B,K].")
            gate = activity_logits > 0.0
            # A compact mechanism must retain at least one mass-carrying edge.
            empty = ~gate.any(dim=1)
            if empty.any():
                fallback = activity_logits.argmax(dim=1)
                gate = gate.clone()
                gate[empty, fallback[empty]] = True
            gate_float = gate.to(continuous.dtype).unsqueeze(-1)
            gated_mass = masses_positive * gate_float
            proportions = gated_mass / gated_mass.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
            active_count = gate_float.sum(dim=1, keepdim=True)
            mass_floor = 0.055
            remainder = (1.0 - mass_floor * active_count).clamp_min(0.0)
            # Both independent mass simplexes use the same activity gate and a
            # floor just above the organizer threshold. Thus every gated edge
            # has sqrt(module_mass*environment_mass) > 0.05 after projection.
            masses = gate_float * mass_floor + proportions * remainder
            fractions_positive = fractions_positive * gate_float + 1.0e-8
        else:
            masses = masses_positive / masses_positive.sum(dim=1, keepdim=True)
        scales = continuous[..., 6:8].clamp(0.0, 1.0)
        fractions = fractions_positive / fractions_positive.sum(dim=1, keepdim=True)
        strength = torch.sqrt(masses[..., 0] * masses[..., 1] + 1.0e-6)
        active = (strength > 0.05).to(continuous.dtype)
        # Functional assembly is required here: this projection is also used
        # inside the trainable one-pass corrector, where in-place column edits
        # invalidate autograd version counters.
        plan = torch.stack(
            [
                active,
                coordinates[..., 0], coordinates[..., 1],
                coordinates[..., 2], coordinates[..., 3],
                masses[..., 0], masses[..., 1], strength,
                scales[..., 0], scales[..., 1],
                fractions[..., 0], fractions[..., 1],
            ],
            dim=-1,
        )
        # The head gates independent mass/fraction state, while serialized
        # activity remains the forward organizer's mass-derived definition.
        order_rows: list[torch.Tensor] = []
        for row in plan:
            keys = [
                (
                    0 if float(row[index, 0]) > 0.5 else 1,
                    float(row[index, 1]), float(row[index, 2]),
                    float(row[index, 3]), float(row[index, 4]),
                    -float(row[index, 7]), index,
                )
                for index in range(edges)
            ]
            order_rows.append(torch.as_tensor(sorted(range(edges), key=keys.__getitem__), device=row.device))
        order = torch.stack(order_rows)
        return torch.gather(plan, 1, order.unsqueeze(-1).expand(-1, -1, PLAN_FEATURE_DIM))

    @torch.no_grad()
    def sample(
        self,
        condition: RequestEncoding | torch.Tensor,
        *,
        batch_size: int | None = None,
        steps: int | None = None,
        method: str = "heun",
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> SampledPlan:
        global_condition = condition.global_embedding if isinstance(condition, RequestEncoding) else condition
        batch = int(global_condition.shape[0] if batch_size is None else batch_size)
        if global_condition.shape[0] != batch:
            raise ValueError("Plan sample condition batch does not match batch_size.")
        initial = initial_noise
        if initial is None:
            initial = torch.randn(
                (batch, self.num_edges, self.state_dim),
                device=global_condition.device,
                dtype=global_condition.dtype,
                generator=generator,
            )
        elif initial.shape != (batch, self.num_edges, self.state_dim):
            raise ValueError("initial_noise has the wrong plan-flow shape.")

        def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return self.forward(state, time, condition).velocity

        endpoint = integrate_rectified_flow(
            velocity, initial, steps=self.sampling_steps if steps is None else int(steps), method=method
        )
        final_time = torch.ones(batch, device=endpoint.device, dtype=endpoint.dtype)
        activity = self.forward(endpoint, final_time, condition).activity_logits
        return SampledPlan(self.project_plan(endpoint, activity), endpoint, activity)


__all__ = [
    "PLAN_CONTINUOUS_INDICES",
    "PLAN_FEATURE_DIM",
    "ConditionalPlanFlow",
    "PlanFlowOutput",
    "SampledPlan",
]
