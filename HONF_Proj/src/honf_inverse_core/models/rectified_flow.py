"""Shared conditional rectified-flow primitives for the inverse hierarchy.

Physical design ``D`` and compact plan ``G`` are continuous endpoint states;
context ``c`` and request ``R`` are conditions. Realized plan ``G_hat`` enters
only joint correction. Independent Gaussian paths make both hierarchy stages
one-to-many without iterative optimization.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Embed scalar rectified-flow time with fixed Fourier frequencies."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("Time embedding hidden_dim must be at least four.")
        self.hidden_dim = int(hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.reshape(-1, 1)
        half = self.hidden_dim // 2
        frequencies = torch.exp(
            torch.linspace(0.0, -math.log(10000.0), half, device=time.device, dtype=time.dtype)
        )
        embedding = torch.cat([torch.sin(time * frequencies), torch.cos(time * frequencies)], dim=-1)
        if embedding.shape[-1] < self.hidden_dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.hidden_dim - embedding.shape[-1]))
        return self.projection(embedding)


class ResidualMLPBlock(nn.Module):
    """Readable LayerNorm/SiLU residual block used by both flow fields."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


def flow_interpolation(
    target: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(x_t, velocity_target, time, noise)` for straight RF paths."""

    if noise is None:
        noise = torch.randn_like(target)
    if noise.shape != target.shape:
        raise ValueError("Rectified-flow noise and target shapes must match.")
    if time is None:
        time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
    if time.shape != (target.shape[0],):
        raise ValueError("Rectified-flow time must have shape [B].")
    broadcast = time.reshape(target.shape[0], *([1] * (target.ndim - 1)))
    state = (1.0 - broadcast) * noise + broadcast * target
    return state, target - noise, time, noise


VelocityFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def integrate_rectified_flow(
    velocity: VelocityFunction,
    initial: torch.Tensor,
    *,
    steps: int,
    method: str = "heun",
) -> torch.Tensor:
    """Integrate one fixed-step RF path with Euler or Heun; no optimization."""

    if int(steps) <= 0 or method not in {"euler", "heun"}:
        raise ValueError("RF integration requires positive steps and method=euler|heun.")
    state = initial
    dt = 1.0 / float(steps)
    batch = initial.shape[0]
    for index in range(int(steps)):
        t0 = torch.full((batch,), index * dt, device=state.device, dtype=state.dtype)
        first = velocity(state, t0)
        if method == "euler":
            state = state + dt * first
        else:
            predicted = state + dt * first
            t1 = torch.full((batch,), (index + 1) * dt, device=state.device, dtype=state.dtype)
            second = velocity(predicted, t1)
            state = state + 0.5 * dt * (first + second)
    return state


__all__ = [
    "ResidualMLPBlock",
    "SinusoidalTimeEmbedding",
    "flow_interpolation",
    "integrate_rectified_flow",
]
