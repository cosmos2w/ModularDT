"""Unified hypergraph neural field baseline.

This model is the ablation baseline for progressive unification. It keeps only
the shared pathway: global/module encoding, environment tokens, hypergraph
organization, and a hypergraph-centric decoder. Local surrogates, port heads,
dynamic tokens, and duplicate-pruning logic are intentionally absent.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from unified_decoder import HypergraphFieldDecoder
from unified_organizer import HypergraphOrganizerCore
from unified_types import BatchData, UnifiedForwardConfig


class LazyMLP(nn.Module):
    """Lazy input MLP for adapter-provided feature dimensions."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnifiedHypergraphNeuralField(nn.Module):
    """Minimal unified forward model used for first-pass ablations."""

    def __init__(self, config: UnifiedForwardConfig):
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        self.global_encoder = LazyMLP(hidden_dim, float(config.dropout))
        self.module_feature_encoder = LazyMLP(hidden_dim, float(config.dropout))
        self.module_position_encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.env_encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.organizer = HypergraphOrganizerCore(config)
        self.decoder = HypergraphFieldDecoder(config)

    def forward(self, batch: BatchData) -> Dict[str, torch.Tensor]:
        cfg = self.config
        module_centers = batch.module_centers.float()
        module_present = batch.module_present.float()
        module_features = batch.module_features.float()
        global_context_raw = batch.global_context.float()
        query_xy = batch.query_xy.float()
        query_time = None if batch.query_time is None else batch.query_time.float()

        global_token = self.global_encoder(global_context_raw)
        module_pos = torch.stack(
            [
                module_centers[..., 0] / max(float(cfg.domain_length_x), 1e-6),
                module_centers[..., 1] / max(float(cfg.domain_length_y), 1e-6),
            ],
            dim=-1,
        )
        module_tokens = self.module_feature_encoder(module_features) + self.module_position_encoder(module_pos)
        module_tokens = module_tokens * module_present.unsqueeze(-1)

        env_coords = self._environment_coords(query_xy.device, query_xy.dtype)
        env_norm = torch.stack(
            [
                env_coords[..., 0] / max(float(cfg.domain_length_x), 1e-6),
                env_coords[..., 1] / max(float(cfg.domain_length_y), 1e-6),
            ],
            dim=-1,
        )
        env_tokens = self.env_encoder(env_norm).unsqueeze(0).expand(query_xy.shape[0], -1, -1)
        if cfg.use_global_context:
            env_tokens = env_tokens + global_token.unsqueeze(1)

        organizer_output = self.organizer(
            module_tokens=module_tokens,
            env_tokens=env_tokens,
            module_centers=module_centers,
            env_coords=env_coords,
            module_present=module_present,
            geometry_mode=cfg.geometry_mode,
        )
        decoder_output = self.decoder(
            query_xy=query_xy,
            query_time=query_time,
            organizer_output=organizer_output,
            global_context=global_token,
        )
        output: Dict[str, torch.Tensor] = {}
        output.update(organizer_output)
        output.update(decoder_output)
        return output

    def _environment_coords(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cfg = self.config
        nx = int(cfg.num_env_tokens_x)
        ny = int(cfg.num_env_tokens_y)
        xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) / max(float(nx), 1.0) * float(cfg.domain_length_x)
        ys = (torch.arange(ny, device=device, dtype=dtype) + 0.5) / max(float(ny), 1.0) * float(cfg.domain_length_y)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
