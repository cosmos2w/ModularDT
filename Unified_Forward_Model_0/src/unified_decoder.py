"""Hypergraph-centric field decoder for unified forward ablations.

The decoder keeps the hypergraph path primary and introduces possible shortcut
paths only through explicit ablation modes. The direct module/environment path
is gated by a sigmoid/logit scalar initialized from the configuration.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from unified_types import UnifiedForwardConfig


EPS = 1e-6


class MLP(nn.Module):
    """Small feed-forward block used by the sandbox decoder."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HypergraphFieldDecoder(nn.Module):
    """Decode query fields from organized hyperedge state and ablated context."""

    def __init__(self, config: UnifiedForwardConfig):
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        field_dim = int(config.field_dim)

        self.query_encoder = MLP(5, hidden_dim, hidden_dim, float(config.dropout))
        self.query_to_hyper = nn.Linear(hidden_dim, hidden_dim)
        self.hyper_key = nn.Linear(hidden_dim, hidden_dim)
        self.hyper_value = nn.Linear(hidden_dim, hidden_dim)
        self.hyper_geometry_bias = nn.Linear(10, 1)
        self.direct_key = nn.Linear(hidden_dim, hidden_dim)
        self.direct_value = nn.Linear(hidden_dim, hidden_dim)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        self.near_proj = nn.Linear(hidden_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity()
        gate_init = min(max(float(config.direct_residual_gate_init), 1e-4), 1.0 - 1e-4)
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.direct_residual_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))

        self.pred_head = MLP(hidden_dim, hidden_dim, field_dim, float(config.dropout))
        if config.output_mean_residual_split:
            self.mean_head = MLP(hidden_dim, hidden_dim, field_dim, float(config.dropout))
            self.residual_head = MLP(hidden_dim, hidden_dim, field_dim, float(config.dropout))

    def forward(
        self,
        query_xy: torch.Tensor,
        query_time: Optional[torch.Tensor],
        organizer_output: Dict[str, torch.Tensor],
        global_context: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cfg = self.config
        query_features = self._query_features(query_xy, query_time)
        query_state = self.query_encoder(query_features)

        hyper_state = organizer_output["hyper_state"]
        hyper_logits = torch.einsum(
            "bqh,bkh->bqk",
            self.query_to_hyper(query_state),
            self.hyper_key(hyper_state),
        ) / math.sqrt(float(query_state.shape[-1]))
        if cfg.use_hyper_geometry_bias:
            geometry_bias = self.hyper_geometry_bias(self._hyper_geometry_features(query_xy, organizer_output)).squeeze(-1)
            hyper_logits = hyper_logits + float(cfg.hyper_geometry_bias_scale) * geometry_bias
        else:
            geometry_bias = torch.zeros_like(hyper_logits)
        hyper_attention = torch.softmax(hyper_logits, dim=-1)
        context = torch.einsum("bqk,bkh->bqh", hyper_attention, self.hyper_value(hyper_state))

        diagnostics: Dict[str, torch.Tensor | str] = {
            "hyper_attention_mean": hyper_attention.mean(dim=1),
            "hyper_geometry_bias_mean": geometry_bias.detach().mean(),
            "hyper_geometry_bias_std": geometry_bias.detach().std(unbiased=False),
            "decoder_mode": cfg.decoder_mode,
        }

        uses_global = self._uses_global()
        uses_direct = self._uses_direct()
        uses_near = self._uses_near_module()
        diagnostics["uses_global_context"] = torch.tensor(float(uses_global), device=query_xy.device, dtype=query_xy.dtype)
        diagnostics["uses_direct_context"] = torch.tensor(float(uses_direct), device=query_xy.device, dtype=query_xy.dtype)
        diagnostics["uses_near_module_context"] = torch.tensor(float(uses_near), device=query_xy.device, dtype=query_xy.dtype)

        if uses_global and global_context is not None:
            context = context + self.global_proj(global_context).unsqueeze(1)

        if uses_direct:
            direct_context, direct_attention = self._direct_context(query_state, organizer_output)
            gate = torch.sigmoid(self.direct_residual_logit)
            context = context + gate * direct_context
            diagnostics["direct_attention_mean"] = direct_attention.mean(dim=1)
            diagnostics["direct_residual_gate"] = gate.detach()
        else:
            diagnostics["direct_residual_gate"] = torch.zeros((), device=query_xy.device, dtype=query_xy.dtype)

        if uses_near:
            context = context + self.near_proj(self._near_module_context(query_xy, organizer_output))

        context = self.context_norm(context)
        output: Dict[str, torch.Tensor | str] = dict(diagnostics)
        if cfg.output_mean_residual_split:
            pred_mean = self.mean_head(context)
            pred_residual = self.residual_head(context)
            output["pred_mean"] = pred_mean
            output["pred_residual"] = pred_residual
            output["pred_field"] = pred_mean + pred_residual
        else:
            output["pred_field"] = self.pred_head(context)
        return output  # type: ignore[return-value]

    def _query_features(self, query_xy: torch.Tensor, query_time: Optional[torch.Tensor]) -> torch.Tensor:
        lx = max(float(self.config.domain_length_x), EPS)
        ly = max(float(self.config.domain_length_y), EPS)
        xy = torch.stack([query_xy[..., 0] / lx, query_xy[..., 1] / ly], dim=-1)
        if query_time is None:
            t = torch.zeros_like(query_xy[..., :1])
        else:
            t = query_time[..., :1]
        if self.config.query_time_mode == "phase":
            t_sin = torch.sin(2.0 * math.pi * t)
            t_cos = torch.cos(2.0 * math.pi * t)
        elif self.config.query_time_mode == "physical_time":
            t_sin = torch.sin(t)
            t_cos = torch.cos(t)
        else:
            t = torch.zeros_like(t)
            t_sin = torch.zeros_like(t)
            t_cos = torch.ones_like(t)
        return torch.cat([xy, t, t_sin, t_cos], dim=-1)

    def _hyper_geometry_features(
        self,
        query_xy: torch.Tensor,
        organizer_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        source = organizer_output["hyper_source_coords"]
        region = organizer_output["hyper_region_coords"]
        source_delta, source_downstream, source_lateral = self._relative_geometry(query_xy, source)
        region_delta, region_downstream, region_lateral = self._relative_geometry(query_xy, region)
        diag = math.sqrt(max(float(self.config.domain_length_x), EPS) ** 2 + max(float(self.config.domain_length_y), EPS) ** 2)
        source_dist = torch.sqrt(source_delta.square().sum(dim=-1, keepdim=True) + EPS) / max(diag, EPS)
        region_dist = torch.sqrt(region_delta.square().sum(dim=-1, keepdim=True) + EPS) / max(diag, EPS)
        lx = max(float(self.config.domain_length_x), EPS)
        ly = max(float(self.config.domain_length_y), EPS)
        return torch.cat(
            [
                source_delta[..., 0:1] / lx,
                source_delta[..., 1:2] / ly,
                region_delta[..., 0:1] / lx,
                region_delta[..., 1:2] / ly,
                source_dist,
                region_dist,
                source_downstream,
                region_downstream,
                source_lateral,
                region_lateral,
            ],
            dim=-1,
        )

    def _relative_geometry(
        self,
        query_xy: torch.Tensor,
        hyper_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lx = max(float(self.config.domain_length_x), EPS)
        ly = max(float(self.config.domain_length_y), EPS)
        delta = query_xy[:, :, None, :] - hyper_coords[:, None, :, :]
        if self.config.geometry_mode == "periodic":
            lengths = torch.tensor([lx, ly], device=query_xy.device, dtype=query_xy.dtype)
            raw_dx = delta[..., 0]
            delta = torch.remainder(delta + 0.5 * lengths, lengths) - 0.5 * lengths
            downstream = torch.remainder(raw_dx, lx).unsqueeze(-1) / lx
        else:
            downstream = torch.relu(delta[..., 0:1]) / lx
        lateral = delta[..., 1:2].abs() / ly
        return delta, downstream, lateral

    def _direct_context(
        self,
        query_state: torch.Tensor,
        organizer_output: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat([organizer_output["module_tokens"], organizer_output["env_tokens"]], dim=1)
        logits = torch.einsum("bqh,bnh->bqn", query_state, self.direct_key(tokens)) / math.sqrt(float(query_state.shape[-1]))
        module_present = organizer_output["module_present"]
        env_mask = torch.ones(
            module_present.shape[0],
            organizer_output["env_tokens"].shape[1],
            device=module_present.device,
            dtype=module_present.dtype,
        )
        mask = torch.cat([module_present, env_mask], dim=1).unsqueeze(1)
        logits = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1) * mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return torch.einsum("bqn,bnh->bqh", attention, self.direct_value(tokens)), attention

    def _near_module_context(
        self,
        query_xy: torch.Tensor,
        organizer_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        module_centers = organizer_output["module_centers"]
        module_tokens = organizer_output["module_tokens"]
        module_present = organizer_output["module_present"]
        delta = query_xy[:, :, None, :] - module_centers[:, None, :, :]
        if self.config.geometry_mode == "periodic":
            lengths = torch.tensor(
                [max(float(self.config.domain_length_x), EPS), max(float(self.config.domain_length_y), EPS)],
                device=query_xy.device,
                dtype=query_xy.dtype,
            )
            delta = torch.remainder(delta + 0.5 * lengths, lengths) - 0.5 * lengths
        dist2 = delta.square().sum(dim=-1)
        sigma2 = max(float(self.config.module_radius) ** 2, EPS)
        weights = torch.exp(-dist2 / (2.0 * sigma2)) * module_present[:, None, :]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return torch.einsum("bqm,bmh->bqh", weights, module_tokens)

    def _uses_global(self) -> bool:
        mode_uses = self.config.decoder_mode in {
            "hyper_plus_global",
            "hyper_plus_global_near",
            "hyper_plus_global_direct",
            "current_like",
        }
        return bool(self.config.use_global_context and mode_uses)

    def _uses_direct(self) -> bool:
        if self.config.decoder_mode == "hyper_only":
            return False
        mode_uses = self.config.decoder_mode in {
            "hyper_plus_direct_residual",
            "hyper_plus_global_direct",
            "hyper_plus_near_direct",
            "current_like",
        }
        return bool(self.config.use_direct_module_env_decoder or mode_uses)

    def _uses_near_module(self) -> bool:
        if self.config.decoder_mode == "hyper_only":
            return False
        mode_uses = self.config.decoder_mode in {
            "hyper_plus_near_module",
            "hyper_plus_global_near",
            "hyper_plus_near_direct",
            "current_like",
        }
        return bool(self.config.use_near_module_context or mode_uses)
