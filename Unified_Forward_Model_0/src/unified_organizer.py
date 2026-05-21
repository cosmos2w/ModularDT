"""Minimal hypergraph organizer for unified forward-model ablations.

The organizer creates soft module-hyperedge and environment-hyperedge
incidences, derives hyperedge state from token aggregates, and exposes simple
diagnostics. It deliberately avoids duplicate pruning, disabled-edge logic, and
case-specific organizer losses in this first sandbox pass.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from unified_types import UnifiedForwardConfig


EPS = 1e-6


def _masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    if mask is None:
        return torch.softmax(logits, dim=dim)
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    masked = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
    out = torch.softmax(masked, dim=dim) * mask
    return out / out.sum(dim=dim, keepdim=True).clamp_min(EPS)


def _as_batched_coords(coords: torch.Tensor, batch_size: int) -> torch.Tensor:
    if coords.ndim == 2:
        return coords.unsqueeze(0).expand(batch_size, -1, -1)
    return coords


def _relative_delta(src: torch.Tensor, dst: torch.Tensor, cfg: UnifiedForwardConfig) -> torch.Tensor:
    delta = dst - src
    if cfg.geometry_mode == "periodic":
        lengths = torch.tensor(
            [max(float(cfg.domain_length_x), EPS), max(float(cfg.domain_length_y), EPS)],
            device=delta.device,
            dtype=delta.dtype,
        )
        delta = torch.remainder(delta + 0.5 * lengths, lengths) - 0.5 * lengths
    return delta


def _weighted_coords(coords: torch.Tensor, weights: torch.Tensor, cfg: UnifiedForwardConfig) -> torch.Tensor:
    denom = weights.sum(dim=1).clamp_min(EPS).unsqueeze(-1)
    if cfg.geometry_mode != "periodic":
        return torch.einsum("bnk,bnd->bkd", weights, coords) / denom

    lengths = torch.tensor(
        [max(float(cfg.domain_length_x), EPS), max(float(cfg.domain_length_y), EPS)],
        device=coords.device,
        dtype=coords.dtype,
    )
    angles = 2.0 * math.pi * coords[:, None, :, :] / lengths
    weight_t = (weights / weights.sum(dim=1, keepdim=True).clamp_min(EPS)).transpose(1, 2).unsqueeze(-1)
    sin_sum = (weight_t * torch.sin(angles)).sum(dim=2)
    cos_sum = (weight_t * torch.cos(angles)).sum(dim=2)
    mean_angle = torch.atan2(sin_sum, cos_sum)
    return torch.remainder(mean_angle / (2.0 * math.pi) * lengths, lengths)


class HypergraphOrganizerCore(nn.Module):
    """Small learned organizer over module and environment tokens."""

    def __init__(self, config: UnifiedForwardConfig):
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        num_hyperedges = int(config.num_hyperedges)
        self.module_score = nn.Linear(hidden_dim, num_hyperedges)
        self.env_score = nn.Linear(hidden_dim, num_hyperedges)
        self.module_to_hyper = nn.Linear(hidden_dim, hidden_dim)
        self.env_to_hyper = nn.Linear(hidden_dim, hidden_dim)
        self.hyper_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.me_query = nn.Linear(hidden_dim, hidden_dim)
        self.me_key = nn.Linear(hidden_dim, hidden_dim)
        self.me_context_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        module_tokens: torch.Tensor,
        env_tokens: torch.Tensor,
        module_centers: torch.Tensor,
        env_coords: torch.Tensor,
        module_present: torch.Tensor,
        geometry_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.config
        if geometry_mode is not None and geometry_mode != cfg.geometry_mode:
            cfg = UnifiedForwardConfig.from_dict({**cfg.to_dict(), "geometry_mode": geometry_mode})

        batch_size, _, hidden_dim = module_tokens.shape
        env_coords_b = _as_batched_coords(env_coords.to(module_tokens.device, module_tokens.dtype), batch_size)
        module_present = module_present.to(device=module_tokens.device, dtype=module_tokens.dtype)

        if cfg.use_A_me_auxiliary:
            q = self.me_query(module_tokens)
            k = self.me_key(env_tokens)
            logits = torch.einsum("bmh,beh->bme", q, k) / math.sqrt(float(hidden_dim))
            A_me = torch.softmax(logits, dim=-1) * module_present.unsqueeze(-1)
            module_env_context = torch.einsum("bme,beh->bmh", A_me, env_tokens)
            module_tokens_for_hyper = module_tokens + 0.25 * self.me_context_proj(module_env_context)
            module_tokens_for_hyper = module_tokens_for_hyper * module_present.unsqueeze(-1)
        else:
            A_me = torch.zeros(
                module_tokens.shape[0],
                module_tokens.shape[1],
                env_tokens.shape[1],
                device=module_tokens.device,
                dtype=module_tokens.dtype,
            )
            module_env_context = torch.zeros_like(module_tokens)
            module_tokens_for_hyper = module_tokens

        module_logits = self.module_score(module_tokens_for_hyper)
        module_mask = module_present.unsqueeze(-1).expand_as(module_logits)
        A_mh = _masked_softmax(module_logits, module_mask, dim=-1)
        A_mh = A_mh * module_present.unsqueeze(-1)

        module_mass_raw = A_mh.sum(dim=1)
        hyper_module_mass = module_mass_raw / module_mass_raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
        source_weights = A_mh / A_mh.sum(dim=1, keepdim=True).clamp_min(EPS)
        hyper_source_coords = _weighted_coords(module_centers, source_weights, cfg)

        env_logits = self.env_score(env_tokens)
        delta = _relative_delta(hyper_source_coords[:, None, :, :], env_coords_b[:, :, None, :], cfg)
        dist = torch.sqrt(delta.square().sum(dim=-1) + EPS)
        scale = 0.25 * math.sqrt(float(cfg.domain_length_x) ** 2 + float(cfg.domain_length_y) ** 2)
        geometry_bias = -dist / max(scale, EPS)
        A_eh = torch.softmax(env_logits + geometry_bias, dim=-1)

        env_mass_raw = A_eh.sum(dim=1)
        hyper_env_mass = env_mass_raw / env_mass_raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
        region_weights = A_eh / A_eh.sum(dim=1, keepdim=True).clamp_min(EPS)
        hyper_region_coords = _weighted_coords(env_coords_b, region_weights, cfg)
        hyper_strength = torch.sqrt(hyper_module_mass * hyper_env_mass + EPS)

        module_summary = torch.einsum("bmk,bmh->bkh", A_mh, self.module_to_hyper(module_tokens_for_hyper))
        module_summary = module_summary / module_mass_raw.unsqueeze(-1).clamp_min(EPS)
        env_summary = torch.einsum("bek,beh->bkh", A_eh, self.env_to_hyper(env_tokens))
        env_summary = env_summary / env_mass_raw.unsqueeze(-1).clamp_min(EPS)
        hyper_state = self.hyper_mix(module_summary + env_summary)

        output: Dict[str, torch.Tensor] = {
            "A_mh": A_mh,
            "A_eh": A_eh,
            "hyper_state": hyper_state,
            "hyper_source_coords": hyper_source_coords,
            "hyper_region_coords": hyper_region_coords,
            "hyper_module_mass_raw": module_mass_raw,
            "hyper_env_mass_raw": env_mass_raw,
            "hyper_module_mass": hyper_module_mass,
            "hyper_env_mass": hyper_env_mass,
            "hyper_strength": hyper_strength,
            "module_tokens": module_tokens,
            "module_tokens_for_hyper": module_tokens_for_hyper,
            "env_tokens": env_tokens,
            "env_coords": env_coords_b,
            "module_centers": module_centers,
            "module_present": module_present,
            "A_me": A_me,
            "module_env_context": module_env_context,
        }

        return output
