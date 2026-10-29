"""Permutation-invariant request/context encoder for inverse design.

Request ``R`` is an unordered masked token set; context ``c`` and separate
geometry constraints condition it. The embedding drives sampled compact plan
``G`` and physical design ``D``. Realized plan ``G_hat`` is not consumed here.
The hierarchy remains one-to-many because embeddings condition independent
plan and layout noise rather than identifying a single answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .rectified_flow import ResidualMLPBlock


@dataclass(frozen=True)
class RequestEncoding:
    global_embedding: torch.Tensor
    token_embeddings: torch.Tensor
    token_mask: torch.Tensor


class RequestSetEncoder(nn.Module):
    """Encode `[B,L]` request tensors without depending on token order."""

    def __init__(
        self,
        *,
        num_request_types: int = 7,
        num_relations: int = 4,
        context_dim: int = 10,
        geometry_dim: int = 8,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        categorical_dim = max(8, hidden_dim // 8)
        self.type_embedding = nn.Embedding(num_request_types + 1, categorical_dim, padding_idx=0)
        self.relation_embedding = nn.Embedding(num_relations + 1, categorical_dim, padding_idx=0)
        continuous_dim = 13
        self.token_input = nn.Linear(2 * categorical_dim + continuous_dim, hidden_dim)
        self.token_blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(int(layers))]
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_dim * 2, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    @staticmethod
    def _float(request: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
        return request[name].float()

    def forward(
        self,
        request: Mapping[str, torch.Tensor],
        context: torch.Tensor,
        geometry_constraints: torch.Tensor,
        geometry_constraint_mask: torch.Tensor | None = None,
    ) -> RequestEncoding:
        mask = self._float(request, "active_mask")
        if mask.ndim != 2:
            raise ValueError("request.active_mask must have shape [B,L].")
        if context.ndim != 2 or geometry_constraints.ndim != 2:
            raise ValueError("context and geometry_constraints must have shape [B,F].")
        if geometry_constraint_mask is None:
            geometry_constraint_mask = torch.ones_like(geometry_constraints)
        type_ids = request["type_id"].long() + 1
        relation_ids = request["relation_id"].long() + 1
        type_ids = torch.where(mask > 0.5, type_ids, torch.zeros_like(type_ids))
        relation_ids = torch.where(mask > 0.5, relation_ids, torch.zeros_like(relation_ids))
        continuous = torch.cat(
            [
                self._float(request, "target_normalized").unsqueeze(-1),
                self._float(request, "target_mask").unsqueeze(-1),
                self._float(request, "tolerance_normalized").unsqueeze(-1),
                self._float(request, "range_normalized"),
                self._float(request, "range_mask").unsqueeze(-1),
                (self._float(request, "priority") / 3.0).unsqueeze(-1),
                torch.log1p(self._float(request, "weight")).unsqueeze(-1),
                self._float(request, "region"),
                self._float(request, "region_mask").unsqueeze(-1),
            ],
            dim=-1,
        )
        token = torch.cat(
            [self.type_embedding(type_ids), self.relation_embedding(relation_ids), continuous], dim=-1
        )
        token = self.token_blocks(self.token_input(token)) * mask.unsqueeze(-1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_mean = token.sum(dim=1) / denominator
        masked_for_max = token.masked_fill(mask.unsqueeze(-1) < 0.5, -torch.inf)
        pooled_max = masked_for_max.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        context_embedding = self.context_encoder(context.float())
        geometry_embedding = self.geometry_encoder(
            torch.cat([geometry_constraints.float(), geometry_constraint_mask.float()], dim=-1)
        )
        global_embedding = self.global_encoder(
            torch.cat([pooled_mean, pooled_max, context_embedding, geometry_embedding], dim=-1)
        )
        return RequestEncoding(global_embedding, token, mask)


__all__ = ["RequestEncoding", "RequestSetEncoder"]
