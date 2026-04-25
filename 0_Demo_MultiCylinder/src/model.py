from __future__ import annotations

"""
PyTorch model for hypergraph-organized dynamic wake reconstruction.

The PhiFlow benchmark uses periodic boundaries, so geometry stays periodic in
both x and y throughout the model.

1. Hypergraph organizer
   Produces module / environment / hyperedge states, soft incidences, and
   explicit source-centered plus wake-centered hyperedge geometry.
2. Behavior + dynamic token head
   Produces a compact smooth-context summary plus structured dynamic memory:
   `dynamic_global_token`, `dynamic_hyper_base`, and lightweight harmonic
   phase-conditioning coefficients.
3. Hierarchical decoder
   Uses separate spatial `(x, y)` and phase `tau` encoders, then predicts
   `pred_mean` and `pred_residual` through:
     - a light global mean branch
     - a dynamic residual branch with global read + wake-aware local refinement

The residual branch is the main carrier for vivid, phase-sensitive wake
structure. Final output heads read only updated query states.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Helper modules ---------------------------------


TWO_PI = 2.0 * math.pi


def periodic_delta_min_image(src: torch.Tensor, dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the minimum-image displacement from `src` to `dst` in normalized coords.

    The sign convention is explicit:
    - positive `dx` means `dst` lies downstream / to the right of `src`
    - positive `dy` means `dst` lies above `src`
    """

    delta = torch.remainder(dst - src + 0.5, 1.0) - 0.5
    return delta[..., 0], delta[..., 1]


def periodic_distance_min_image(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    dx, dy = periodic_delta_min_image(src, dst)
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


def directed_periodic_downstream_delta(src_x: torch.Tensor, dst_x: torch.Tensor) -> torch.Tensor:
    """Return downstream displacement from `src_x` to `dst_x` along +x in [0, 1)."""

    return torch.remainder(dst_x - src_x, 1.0)


def periodic_relative_features(
    src_xy: torch.Tensor,
    dst_xy: torch.Tensor,
    mode: str = "min_image",
) -> torch.Tensor:
    """Return explicit periodic relative features from `src_xy` to `dst_xy`.

    Features are:
      [dx, dy, distance, downstream, upstream]

    `mode="min_image"` uses minimum-image `dx` for proximity.
    `mode="directed_downstream"` uses directed periodic downstream distance in x
    while keeping y as minimum-image.
    """

    _, dy = periodic_delta_min_image(src_xy, dst_xy)
    if mode == "min_image":
        dx, _ = periodic_delta_min_image(src_xy, dst_xy)
        downstream = torch.clamp(dx, min=0.0)
        upstream = torch.clamp(-dx, min=0.0)
    elif mode == "directed_downstream":
        dx = directed_periodic_downstream_delta(src_xy[..., 0], dst_xy[..., 0])
        downstream = dx
        upstream = torch.zeros_like(dx)
    else:
        raise ValueError(f"Unsupported periodic_relative_features mode: {mode}")

    dist = torch.sqrt(dx.square() + dy.square() + 1e-8)
    return torch.stack([dx, dy, dist, downstream, upstream], dim=-1)


def periodic_weighted_mean(coords: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Circular weighted mean for periodic coords in [0, 1].

    Args:
        coords: [B, N, 2]
        weights: [B, N, K]
    Returns:
        [B, K, 2]
    """

    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError("coords must have shape [B, N, 2].")
    if weights.ndim != 3 or weights.shape[:2] != coords.shape[:2]:
        raise ValueError("weights must have shape [B, N, K] matching coords.")

    angles = TWO_PI * coords[:, None, :, :]
    weight_t = weights.transpose(1, 2).unsqueeze(-1)
    sin_sum = (weight_t * torch.sin(angles)).sum(dim=2)
    cos_sum = (weight_t * torch.cos(angles)).sum(dim=2)
    resultant_sq = sin_sum.square() + cos_sum.square()
    safe_sin = torch.where(resultant_sq > 1e-12, sin_sum, torch.zeros_like(sin_sum))
    safe_cos = torch.where(resultant_sq > 1e-12, cos_sum, torch.ones_like(cos_sum))
    mean_angle = torch.atan2(safe_sin, safe_cos)
    return torch.remainder(mean_angle / TWO_PI, 1.0)


def periodic_weighted_rms_spread(
    centers: torch.Tensor,
    coords: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted RMS spread of periodic coords around periodic centers.

    Args:
        centers: [B, K, 2]
        coords: [B, N, 2]
        weights: [B, N, K]
    Returns:
        [B, K, 1]
    """

    if centers.ndim != 3 or centers.shape[-1] != 2:
        raise ValueError("centers must have shape [B, K, 2].")
    dx, dy = periodic_delta_min_image(centers[:, :, None, :], coords[:, None, :, :])
    dist_sq = dx.square() + dy.square()
    weight_t = weights.transpose(1, 2)
    denom = weight_t.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.sqrt((weight_t * dist_sq).sum(dim=-1, keepdim=True) / denom + 1e-8)


def masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """Softmax with explicit zeroing for masked entries and fully masked rows."""

    if mask is None:
        return torch.softmax(logits, dim=dim)

    mask = mask.to(device=logits.device, dtype=logits.dtype)
    masked_logits = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights * mask
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)


class FourierEncoder(nn.Module):
    """Sin / cos positional encoding for low-dimensional continuous inputs."""

    def __init__(self, input_dim: int, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        freq_bands = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + (2 * self.input_dim * self.num_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = [x] if self.include_input else []
        for freq in self.freq_bands:
            pieces.append(torch.sin(2.0 * math.pi * freq * x))
            pieces.append(torch.cos(2.0 * math.pi * freq * x))
        return torch.cat(pieces, dim=-1)


class PhaseHarmonicEncoder(nn.Module):
    """Periodic harmonic features for canonical phase tau in [0, 1]."""

    def __init__(self, num_harmonics: int):
        super().__init__()
        self.num_harmonics = max(int(num_harmonics), 1)
        harmonics = torch.arange(1, self.num_harmonics + 1, dtype=torch.float32)
        self.register_buffer("harmonics", harmonics, persistent=False)

    @property
    def output_dim(self) -> int:
        return 2 * self.num_harmonics

    def phase_angles(self, tau: torch.Tensor) -> torch.Tensor:
        shape = [1] * (tau.ndim - 1) + [self.num_harmonics]
        harmonics = self.harmonics.view(*shape).to(device=tau.device, dtype=tau.dtype)
        return 2.0 * math.pi * tau * harmonics

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        angles = self.phase_angles(tau)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class MLP(nn.Module):
    """Simple configurable MLP block used across organizer and decoder heads."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 0) + [out_dim]
        act_layer = nn.GELU if activation == "gelu" else nn.SiLU

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act_layer())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelativeGeometryBias(nn.Module):
    """Affine query-to-token geometry bias for module/env attention only."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(5))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        dx: torch.Tensor,
        dy: torch.Tensor,
        dist: torch.Tensor,
        downstream: torch.Tensor,
        upstream: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.weight[0] * dx
            + self.weight[1] * dy
            + self.weight[2] * dist
            + self.weight[3] * downstream
            + self.weight[4] * upstream
            + self.bias
        )


class HyperWakeGeometryBias(nn.Module):
    """Affine wake-aware geometry bias over hyperedges."""

    def __init__(self):
        super().__init__()
        init = torch.tensor([-1.25, -1.00, 0.75, 0.25, 0.00, 0.50], dtype=torch.float32)
        self.weight = nn.Parameter(init)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        wake_dist_scaled: torch.Tensor,
        cross_scaled: torch.Tensor,
        along_scaled: torch.Tensor,
        downstream: torch.Tensor,
        extent_log: torch.Tensor,
        strength_log: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.weight[0] * wake_dist_scaled
            + self.weight[1] * cross_scaled
            + self.weight[2] * along_scaled
            + self.weight[3] * downstream
            + self.weight[4] * extent_log
            + self.weight[5] * strength_log
            + self.bias
        )


class MultiHeadCrossAttention(nn.Module):
    """Memory-safe multi-head cross-attention with optional query chunking."""

    def __init__(self, model_dim: int, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        inner_dim = self.num_heads * self.head_dim

        self.query_proj = nn.Linear(model_dim, inner_dim, bias=False)
        self.key_proj = nn.Linear(model_dim, inner_dim, bias=False)
        self.value_proj = nn.Linear(model_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scale = self.head_dim ** -0.5

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logits = torch.einsum("bhqd,bhnd->bhqn", query, key) * self.scale
        if attn_bias is not None:
            logits = logits + attn_bias[:, None, :, :]

        mask = None
        if context_mask is not None:
            mask = context_mask[:, None, None, :].to(dtype=logits.dtype, device=logits.device)
            logits = logits.masked_fill(mask <= 0, -1e9)

        weights = torch.softmax(logits, dim=-1)
        if mask is not None:
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = self.dropout(weights)
        return torch.einsum("bhqn,bhnd->bhqd", weights, value)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        query_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, num_queries, _ = query.shape
        num_context = context.shape[1]

        q = self.query_proj(query).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(context).view(batch_size, num_context, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(context).view(batch_size, num_context, self.num_heads, self.head_dim).transpose(1, 2)

        if query_chunk_size is None or num_queries <= query_chunk_size:
            attended = self._attend(q, k, v, attn_bias, context_mask)
        else:
            chunks = []
            for start in range(0, num_queries, query_chunk_size):
                end = min(start + query_chunk_size, num_queries)
                bias_chunk = attn_bias[:, start:end, :] if attn_bias is not None else None
                attended_chunk = self._attend(q[:, :, start:end, :], k, v, bias_chunk, context_mask)
                chunks.append(attended_chunk)
            attended = torch.cat(chunks, dim=2)

        attended = attended.transpose(1, 2).reshape(batch_size, num_queries, self.num_heads * self.head_dim)
        return self.out_proj(attended)


class CrossAttentionBlock(nn.Module):
    """Cross-attention + feedforward block for shared memory reads."""

    def __init__(self, model_dim: int, num_heads: int, head_dim: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(model_dim)
        self.attn = MultiHeadCrossAttention(model_dim, num_heads, head_dim, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = MLP(model_dim, model_dim * max(int(ffn_mult), 1), model_dim, num_layers=2, dropout=dropout)
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        query_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.attn_norm(query),
            context,
            attn_bias=attn_bias,
            context_mask=context_mask,
            query_chunk_size=query_chunk_size,
        )
        query = query + self.attn_dropout(attn_out)
        ffn_out = self.ffn(self.ffn_norm(query))
        return query + self.ffn_dropout(ffn_out)


class MultiHeadLocalCrossAttention(nn.Module):
    """Cross-attention where each query sees its own gathered local context."""

    def __init__(self, model_dim: int, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        inner_dim = self.num_heads * self.head_dim

        self.query_proj = nn.Linear(model_dim, inner_dim, bias=False)
        self.key_proj = nn.Linear(model_dim, inner_dim, bias=False)
        self.value_proj = nn.Linear(model_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_queries, _ = query.shape
        num_context = context.shape[2]

        q = self.query_proj(query).view(batch_size, num_queries, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key_proj(context).view(batch_size, num_queries, num_context, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = self.value_proj(context).view(batch_size, num_queries, num_context, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        logits = torch.einsum("bhqd,bhqkd->bhqk", q, k) * self.scale
        if attn_bias is not None:
            logits = logits + attn_bias[:, None, :, :]
        if context_mask is not None:
            mask = context_mask[:, None, :, :].to(dtype=logits.dtype, device=logits.device)
            logits = logits.masked_fill(mask <= 0, -1e9)
        else:
            mask = None

        weights = torch.softmax(logits, dim=-1)
        if mask is not None:
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = self.dropout(weights)
        attended = torch.einsum("bhqk,bhqkd->bhqd", weights, v)
        attended = attended.permute(0, 2, 1, 3).reshape(batch_size, num_queries, self.num_heads * self.head_dim)
        return self.out_proj(attended)


class LocalCrossAttentionBlock(nn.Module):
    """Cross-attention + feedforward block for explicit top-k local refinement."""

    def __init__(self, model_dim: int, num_heads: int, head_dim: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(model_dim)
        self.attn = MultiHeadLocalCrossAttention(model_dim, num_heads, head_dim, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = MLP(model_dim, model_dim * max(int(ffn_mult), 1), model_dim, num_layers=2, dropout=dropout)
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.attn_norm(query),
            context,
            attn_bias=attn_bias,
            context_mask=context_mask,
        )
        query = query + self.attn_dropout(attn_out)
        ffn_out = self.ffn(self.ffn_norm(query))
        return query + self.ffn_dropout(ffn_out)


# --------------------------- Configuration dataclass ---------------------------


@dataclass
class ModelConfig:
    """Configuration for the hypergraph-organized neural field model."""

    max_num_cylinders: int = 8
    domain_length_x: float = 24.0
    domain_length_y: float = 12.0
    re_scale: float = 200.0
    num_env_tokens_x: int = 24
    num_env_tokens_y: int = 8
    num_hyperedges: int = 6
    hidden_dim: int = 80
    behavior_dim: int = 80
    latent_dim: int = 80
    dynamic_token_dim: int = 80
    message_passing_steps: int = 3
    structure_fourier_frequencies: int = 1
    spatial_query_fourier_frequencies: int = 3
    phase_fourier_frequencies: int = 2
    decoder_hidden_dim: int = 160
    perceiver_num_layers_global: int = 1
    perceiver_num_layers_local: int = 2
    perceiver_num_heads: int = 4
    perceiver_head_dim: int = 20
    perceiver_ffn_mult: int = 2
    perceiver_dropout: float = 0.05
    perceiver_refine_topk_env: int = 24
    perceiver_refine_topk_mod: int = 5
    perceiver_query_chunk_size: int = 1024
    local_topk_mode: str = "wake_relevance"
    local_topk_distance_weight: float = 1.0
    local_topk_hyper_weight: float = 1.0
    local_topk_attention_bias_weight: float = 0.25
    local_topk_use_softmax_hyper_weights: bool = True
    local_topk_detach_scores: bool = True
    use_dynamic_hyper_tokens: bool = True
    use_hyper_phase_offsets: bool = True
    use_phase_conditioned_dynamic_tokens: bool = True
    dynamic_phase_harmonics: int = 3
    dynamic_phase_rank: int = 12
    dynamic_phase_mode: str = "low_rank_harmonic"
    phase_conditioning_dropout: float = 0.05
    use_wake_centered_hyper_geometry: bool = True
    future_module_feature_dim: int = 0
    future_global_feature_dim: int = 0
    dropout: float = 0.05
    use_layer_norm: bool = True
    decoder_type: str = "hierarchical_perceiver"

    @classmethod
    def from_dict(cls, payload: Dict) -> "ModelConfig":
        data = dict(payload)
        if "query_fourier_frequencies" in data and "spatial_query_fourier_frequencies" not in data:
            data["spatial_query_fourier_frequencies"] = data["query_fourier_frequencies"]
        if "phase_harmonics" in data and "phase_fourier_frequencies" not in data:
            data["phase_fourier_frequencies"] = data["phase_harmonics"]
        valid = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# --------------------------- Organizer / Encoder ------------------------------


class HypergraphOrganizer(nn.Module):
    """Structure-conditioned organizer that produces a soft organized state."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.max_num_cylinders = cfg.max_num_cylinders

        self.module_coord_encoder = FourierEncoder(2, cfg.structure_fourier_frequencies, include_input=True)
        self.env_coord_encoder = FourierEncoder(2, cfg.structure_fourier_frequencies, include_input=True)

        global_in_dim = 2 + cfg.future_global_feature_dim
        module_in_dim = self.module_coord_encoder.output_dim + global_in_dim + cfg.future_module_feature_dim
        env_in_dim = self.env_coord_encoder.output_dim + global_in_dim

        self.module_encoder = MLP(
            in_dim=module_in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.hidden_dim,
            num_layers=3,
            dropout=cfg.dropout,
            layer_norm=cfg.use_layer_norm,
        )
        self.env_encoder = MLP(
            in_dim=env_in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.hidden_dim,
            num_layers=3,
            dropout=cfg.dropout,
            layer_norm=cfg.use_layer_norm,
        )

        self.hyperedge_embeddings = nn.Parameter(torch.randn(cfg.num_hyperedges, cfg.hidden_dim) * 0.02)
        self.hyperedge_context = nn.Linear(global_in_dim, cfg.hidden_dim)

        rel_dim = 5
        self.me_score = MLP(2 * cfg.hidden_dim + rel_dim, cfg.hidden_dim, 1, num_layers=3, dropout=cfg.dropout)
        self.mh_score = MLP(2 * cfg.hidden_dim + rel_dim, cfg.hidden_dim, 1, num_layers=2, dropout=cfg.dropout)
        self.eh_score = MLP(2 * cfg.hidden_dim + rel_dim, cfg.hidden_dim, 1, num_layers=2, dropout=cfg.dropout)

        self.module_update = MLP(3 * cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=2, dropout=cfg.dropout)
        self.env_update = MLP(3 * cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=2, dropout=cfg.dropout)
        self.hyper_update = MLP(3 * cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=2, dropout=cfg.dropout)

        self.register_buffer("env_token_coords", self._make_env_token_grid(), persistent=False)

    def _make_env_token_grid(self) -> torch.Tensor:
        xs = (torch.arange(self.cfg.num_env_tokens_x, dtype=torch.float32) + 0.5) / float(self.cfg.num_env_tokens_x)
        ys = (torch.arange(self.cfg.num_env_tokens_y, dtype=torch.float32) + 0.5) / float(self.cfg.num_env_tokens_y)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    def _global_features(
        self,
        re_values: torch.Tensor,
        num_cylinders: torch.Tensor,
        extra_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        re_scaled = re_values / max(self.cfg.re_scale, 1e-6)
        nc_scaled = num_cylinders / max(float(self.cfg.max_num_cylinders), 1.0)
        feats = [re_scaled, nc_scaled]
        if extra_global is not None:
            feats.append(extra_global)
        return torch.cat(feats, dim=-1)

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.to(values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=dim).clamp_min(1e-6)
        return (values * mask).sum(dim=dim) / denom

    @staticmethod
    def masked_max(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.to(values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        very_neg = torch.full_like(values, torch.finfo(values.dtype).min)
        masked = torch.where(mask > 0, values, very_neg)
        max_vals = masked.max(dim=dim).values
        has_valid = mask.sum(dim=dim) > 0
        return torch.where(has_valid, max_vals, torch.zeros_like(max_vals))

    def forward(
        self,
        re_values: torch.Tensor,
        num_cylinders: torch.Tensor,
        centers: torch.Tensor,
        cyl_mask: torch.Tensor,
        extra_global: Optional[torch.Tensor] = None,
        extra_module: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = centers.device
        batch_size, n_max, _ = centers.shape
        global_feats = self._global_features(re_values, num_cylinders, extra_global=extra_global)

        module_coords_norm = centers.clone()
        module_coords_norm[..., 0] = module_coords_norm[..., 0] / max(self.cfg.domain_length_x, 1e-6)
        module_coords_norm[..., 1] = module_coords_norm[..., 1] / max(self.cfg.domain_length_y, 1e-6)
        module_coord_feat = self.module_coord_encoder(module_coords_norm)
        global_expand = global_feats[:, None, :].expand(batch_size, n_max, -1)

        module_inputs = [module_coord_feat, global_expand]
        if extra_module is not None:
            module_inputs.append(extra_module)
        module_state = self.module_encoder(torch.cat(module_inputs, dim=-1))
        module_state = module_state * cyl_mask.unsqueeze(-1)

        env_coords = self.env_token_coords.to(device=device).unsqueeze(0).expand(batch_size, -1, -1)
        env_coord_feat = self.env_coord_encoder(env_coords)
        env_global = global_feats[:, None, :].expand(batch_size, env_coords.shape[1], -1)
        env_state = self.env_encoder(torch.cat([env_coord_feat, env_global], dim=-1))

        hyper_state = self.hyperedge_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        hyper_state = hyper_state + self.hyperedge_context(global_feats).unsqueeze(1)

        rel_me = periodic_relative_features(module_coords_norm[:, :, None, :], env_coords[:, None, :, :])
        rel_mm = periodic_relative_features(module_coords_norm[:, :, None, :], module_coords_norm[:, None, :, :])

        eye = torch.eye(n_max, device=device, dtype=module_coords_norm.dtype)[None, :, :]
        pair_mask = cyl_mask[:, :, None] * cyl_mask[:, None, :] * (1.0 - eye)
        dist_mm = rel_mm[..., 2].clamp_min(1e-3)
        pair_w = pair_mask / dist_mm
        pair_w = pair_w / pair_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        module_geom_summary = torch.einsum("bnm,bnmf->bnf", pair_w, rel_mm)

        num_steps = max(int(self.cfg.message_passing_steps), 1)
        for _ in range(num_steps):
            n_env = env_state.shape[1]
            n_hyp = hyper_state.shape[1]

            module_expand = module_state[:, :, None, :].expand(-1, -1, n_env, -1)
            env_expand = env_state[:, None, :, :].expand(-1, n_max, -1, -1)
            me_logits = self.me_score(torch.cat([module_expand, env_expand, rel_me], dim=-1)).squeeze(-1)
            valid_pair_mask = cyl_mask[:, :, None].expand_as(me_logits)
            A_me = masked_softmax(me_logits, valid_pair_mask, dim=-1)

            env_from_mod = A_me.transpose(1, 2)
            env_from_mod = env_from_mod / env_from_mod.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            env_geom_summary = torch.einsum("bmn,bnf->bmf", env_from_mod, module_geom_summary)

            module_h = module_state[:, :, None, :].expand(-1, -1, n_hyp, -1)
            hyper_h = hyper_state[:, None, :, :].expand(-1, n_max, -1, -1)
            module_geom_h = module_geom_summary[:, :, None, :].expand(-1, -1, n_hyp, -1)
            mh_logits = self.mh_score(torch.cat([module_h, hyper_h, module_geom_h], dim=-1)).squeeze(-1)
            A_mh = masked_softmax(mh_logits, cyl_mask[:, :, None], dim=-1)

            env_h = env_state[:, :, None, :].expand(-1, -1, n_hyp, -1)
            hyper_e = hyper_state[:, None, :, :].expand(-1, n_env, -1, -1)
            env_geom_h = env_geom_summary[:, :, None, :].expand(-1, -1, n_hyp, -1)
            eh_logits = self.eh_score(torch.cat([env_h, hyper_e, env_geom_h], dim=-1)).squeeze(-1)
            A_eh = torch.softmax(eh_logits, dim=-1)

            env_to_module = torch.einsum("bnm,bmd->bnd", A_me, env_state)
            hyp_to_module = torch.einsum("bnk,bkd->bnd", A_mh, hyper_state)
            module_delta = self.module_update(torch.cat([module_state, env_to_module, hyp_to_module], dim=-1))
            module_state = (module_state + module_delta) * cyl_mask.unsqueeze(-1)

            mod_to_env = torch.einsum("bnm,bnd->bmd", A_me, module_state)
            hyp_to_env = torch.einsum("bmk,bkd->bmd", A_eh, hyper_state)
            env_delta = self.env_update(torch.cat([env_state, mod_to_env, hyp_to_env], dim=-1))
            env_state = env_state + env_delta

            mod_to_hyp = torch.einsum("bnk,bnd->bkd", A_mh, module_state)
            env_to_hyp = torch.einsum("bmk,bmd->bkd", A_eh, env_state)
            hyp_delta = self.hyper_update(torch.cat([hyper_state, mod_to_hyp, env_to_hyp], dim=-1))
            hyper_state = hyper_state + hyp_delta

        module_weights = A_mh * cyl_mask[:, :, None]
        module_weights = module_weights / module_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        hyper_source_coords = periodic_weighted_mean(module_coords_norm, module_weights)

        if self.cfg.use_wake_centered_hyper_geometry:
            env_weights = A_eh / A_eh.sum(dim=1, keepdim=True).clamp_min(1e-6)
            hyper_wake_coords = periodic_weighted_mean(env_coords, env_weights)
            hyper_wake_extent = periodic_weighted_rms_spread(hyper_wake_coords, env_coords, env_weights)
        else:
            hyper_wake_coords = hyper_source_coords
            hyper_wake_extent = periodic_weighted_rms_spread(hyper_source_coords, module_coords_norm, module_weights)

        wake_dx, wake_dy = periodic_delta_min_image(hyper_source_coords, hyper_wake_coords)
        hyper_wake_axis = torch.stack([wake_dx, wake_dy], dim=-1)
        axis_norm = torch.sqrt(hyper_wake_axis.square().sum(dim=-1, keepdim=True).clamp_min(1e-12))
        default_axis = torch.zeros_like(hyper_wake_axis)
        default_axis[..., 0] = 1.0
        hyper_wake_axis = torch.where(
            axis_norm > 1e-6,
            hyper_wake_axis / axis_norm,
            default_axis,
        )

        module_mass = (A_mh * cyl_mask[:, :, None]).sum(dim=1)
        module_mass = module_mass / cyl_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        env_mass = A_eh.mean(dim=1)
        hyper_strength = 0.5 * (module_mass + env_mass)

        return {
            "module_state": module_state,
            "env_state": env_state,
            "hyper_state": hyper_state,
            "A_me": A_me,
            "A_mh": A_mh,
            "A_eh": A_eh,
            "env_coords": env_coords,
            "module_coords_norm": module_coords_norm,
            "hyper_source_coords": hyper_source_coords,
            "hyper_wake_coords": hyper_wake_coords,
            "hyper_wake_axis": hyper_wake_axis,
            "hyper_wake_extent": hyper_wake_extent,
            "hyper_strength": hyper_strength.unsqueeze(-1),
            "global_features": global_feats,
            "cyl_mask": cyl_mask,
        }


# ----------------------------- Behavior head ----------------------------------


class BehaviorHead(nn.Module):
    """Maps organized structure into smooth context plus structured dynamic memory."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        global_dim = 2 + cfg.future_global_feature_dim
        pooled_dim = (6 * cfg.hidden_dim) + global_dim + 1

        self.behavior_mlp = MLP(
            in_dim=pooled_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.behavior_dim,
            num_layers=3,
            dropout=cfg.dropout,
            layer_norm=cfg.use_layer_norm,
        )
        self.mean_latent_head = nn.Linear(cfg.behavior_dim, cfg.latent_dim)
        self.freq_head = nn.Linear(cfg.behavior_dim, 1)

        dynamic_global_in_dim = cfg.behavior_dim + cfg.latent_dim + 1
        self.dynamic_global_head = MLP(
            dynamic_global_in_dim,
            cfg.hidden_dim,
            cfg.dynamic_token_dim,
            num_layers=2,
            dropout=cfg.dropout,
            layer_norm=cfg.use_layer_norm,
        )

        dynamic_hyper_in_dim = cfg.hidden_dim + cfg.behavior_dim + cfg.latent_dim + 1
        self.dynamic_hyper_head = MLP(
            dynamic_hyper_in_dim,
            cfg.hidden_dim,
            cfg.dynamic_token_dim,
            num_layers=2,
            dropout=cfg.dropout,
            layer_norm=cfg.use_layer_norm,
        )
        if cfg.use_phase_conditioned_dynamic_tokens:
            phase_out_dim = cfg.dynamic_phase_harmonics * cfg.dynamic_phase_rank
            self.hyper_phase_sin_head = nn.Linear(cfg.dynamic_token_dim, phase_out_dim)
            self.hyper_phase_cos_head = nn.Linear(cfg.dynamic_token_dim, phase_out_dim)
        else:
            self.hyper_phase_sin_head = None
            self.hyper_phase_cos_head = None

        if cfg.use_hyper_phase_offsets:
            self.hyper_phase_head = nn.Linear(cfg.dynamic_token_dim, cfg.dynamic_phase_harmonics)
        else:
            self.hyper_phase_head = None

    def forward(self, organized: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        module_state = organized["module_state"]
        env_state = organized["env_state"]
        hyper_state = organized["hyper_state"]
        cyl_mask = organized["cyl_mask"]
        global_features = organized["global_features"]
        hyper_strength = organized["hyper_strength"]

        module_mean = HypergraphOrganizer.masked_mean(module_state, cyl_mask, dim=1)
        module_max = HypergraphOrganizer.masked_max(module_state, cyl_mask, dim=1)
        env_mean = env_state.mean(dim=1)
        env_max = env_state.max(dim=1).values
        hyper_mean = hyper_state.mean(dim=1)
        hyper_max = hyper_state.max(dim=1).values
        hyper_strength_mean = hyper_strength.mean(dim=1)

        pooled = torch.cat(
            [
                module_mean,
                module_max,
                env_mean,
                env_max,
                hyper_mean,
                hyper_max,
                global_features,
                hyper_strength_mean,
            ],
            dim=-1,
        )
        behavior_latent = self.behavior_mlp(pooled)
        mean_latent = self.mean_latent_head(behavior_latent)
        freq_pred = F.softplus(self.freq_head(behavior_latent)) + 1e-6

        dynamic_global_token = self.dynamic_global_head(
            torch.cat([behavior_latent, mean_latent, freq_pred], dim=-1)
        ).unsqueeze(1)

        batch_size, num_hyper, _ = hyper_state.shape
        behavior_expand = behavior_latent[:, None, :].expand(batch_size, num_hyper, -1)
        mean_expand = mean_latent[:, None, :].expand(batch_size, num_hyper, -1)
        dynamic_hyper_inputs = torch.cat([hyper_state, behavior_expand, mean_expand, hyper_strength], dim=-1)
        dynamic_hyper_base = self.dynamic_hyper_head(dynamic_hyper_inputs)
        if self.cfg.use_dynamic_hyper_tokens:
            dynamic_hyper_base = dynamic_hyper_base + 0.25 * dynamic_global_token.expand(-1, num_hyper, -1)

        hyper_phase_offsets = None
        if self.hyper_phase_head is not None:
            hyper_phase_offsets = math.pi * torch.tanh(self.hyper_phase_head(dynamic_hyper_base))

        hyper_phase_sin_coeff = None
        hyper_phase_cos_coeff = None
        if self.hyper_phase_sin_head is not None and self.hyper_phase_cos_head is not None:
            coeff_shape = (batch_size, num_hyper, self.cfg.dynamic_phase_harmonics, self.cfg.dynamic_phase_rank)
            hyper_phase_sin_coeff = self.hyper_phase_sin_head(dynamic_hyper_base).view(*coeff_shape)
            hyper_phase_cos_coeff = self.hyper_phase_cos_head(dynamic_hyper_base).view(*coeff_shape)

        out = {
            "behavior_latent": behavior_latent,
            "mean_latent": mean_latent,
            # [B, 1, D_dyn] global residual-memory token shared across queries.
            "dynamic_global_token": dynamic_global_token,
            # [B, K_h, D_dyn] hyperedge-local dynamic memory read directly by residual decoding.
            "dynamic_hyper_base": dynamic_hyper_base,
            "dynamic_hyper_tokens": dynamic_hyper_base,
            "freq_pred": freq_pred,
        }
        if hyper_phase_offsets is not None:
            out["hyper_phase_offsets"] = hyper_phase_offsets
        if hyper_phase_sin_coeff is not None and hyper_phase_cos_coeff is not None:
            out["hyper_phase_sin_coeff"] = hyper_phase_sin_coeff
            out["hyper_phase_cos_coeff"] = hyper_phase_cos_coeff
        return out


class PhaseConditionedHyperContext(nn.Module):
    """Low-rank harmonic hyper-context without materializing [B, Q, K, D]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.dynamic_phase_mode != "low_rank_harmonic":
            raise ValueError(f"Unsupported dynamic_phase_mode={cfg.dynamic_phase_mode!r}")

        self.geometry_bias = HyperWakeGeometryBias()
        self.rank_to_model = nn.Linear(cfg.dynamic_phase_rank, cfg.dynamic_token_dim, bias=False)
        self.dropout = nn.Dropout(cfg.phase_conditioning_dropout) if cfg.phase_conditioning_dropout > 0.0 else nn.Identity()

        harmonics = torch.arange(1, cfg.dynamic_phase_harmonics + 1, dtype=torch.float32)
        self.register_buffer("harmonics", harmonics, persistent=False)

    def geometry_logits(
        self,
        query_xy_norm: torch.Tensor,
        organized: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        hyper_source_coords = organized["hyper_source_coords"]
        hyper_wake_coords = organized["hyper_wake_coords"]
        hyper_wake_axis = organized["hyper_wake_axis"]
        hyper_wake_extent = organized["hyper_wake_extent"].squeeze(-1).clamp_min(0.02)
        hyper_strength = organized["hyper_strength"].squeeze(-1)

        wake_dx, wake_dy = periodic_delta_min_image(
            hyper_wake_coords[:, None, :, :],
            query_xy_norm[:, :, None, :],
        )
        wake_dist = torch.sqrt(wake_dx.square() + wake_dy.square() + 1e-8)

        source_dx, source_dy = periodic_delta_min_image(
            hyper_source_coords[:, None, :, :],
            query_xy_norm[:, :, None, :],
        )
        downstream = directed_periodic_downstream_delta(
            hyper_source_coords[:, None, :, 0],
            query_xy_norm[:, :, None, 0],
        )

        axis_x = hyper_wake_axis[:, None, :, 0]
        axis_y = hyper_wake_axis[:, None, :, 1]
        along = source_dx * axis_x + source_dy * axis_y
        cross = torch.abs((-source_dx * axis_y) + (source_dy * axis_x))

        extent = hyper_wake_extent[:, None, :]
        wake_dist_scaled = wake_dist / extent
        cross_scaled = cross / (extent + 0.25 * downstream + 1e-3)
        along_scaled = along / (extent + 0.25 * downstream + 1e-3)
        extent_log = torch.log(extent)
        strength_log = torch.log1p(4.0 * hyper_strength[:, None, :])

        return self.geometry_bias(
            wake_dist_scaled,
            cross_scaled,
            along_scaled,
            downstream,
            extent_log,
            strength_log,
        )

    def forward(
        self,
        query_xy_norm: torch.Tensor,
        query_tau: torch.Tensor,
        behavior: Dict[str, torch.Tensor],
        organized: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dynamic_hyper_base = behavior["dynamic_hyper_base"]
        num_hyper = dynamic_hyper_base.shape[1]
        if num_hyper == 0:
            empty = torch.zeros(
                query_tau.shape[0],
                query_tau.shape[1],
                self.cfg.dynamic_token_dim,
                device=query_tau.device,
                dtype=query_tau.dtype,
            )
            return empty, torch.zeros(query_tau.shape[0], query_tau.shape[1], 0, device=query_tau.device, dtype=query_tau.dtype)

        logits = self.geometry_logits(query_xy_norm, organized)
        weights = torch.softmax(logits, dim=-1)

        base_ctx = torch.einsum("bqk,bkd->bqd", weights, dynamic_hyper_base)
        if not self.cfg.use_phase_conditioned_dynamic_tokens:
            return base_ctx, logits

        view_shape = [1] * (query_tau.ndim - 1) + [self.cfg.dynamic_phase_harmonics]
        harmonics = self.harmonics.view(*view_shape).to(device=query_tau.device, dtype=query_tau.dtype)
        angles = TWO_PI * query_tau * harmonics
        sin_basis = torch.sin(angles)
        cos_basis = torch.cos(angles)

        sin_coeff = behavior["hyper_phase_sin_coeff"]
        cos_coeff = behavior["hyper_phase_cos_coeff"]
        if "hyper_phase_offsets" in behavior:
            phase_offsets = behavior["hyper_phase_offsets"]
            cos_off = torch.cos(phase_offsets)[..., None]
            sin_off = torch.sin(phase_offsets)[..., None]
            base_sin_coeff = sin_coeff
            base_cos_coeff = cos_coeff
            sin_coeff = (base_sin_coeff * cos_off) + (base_cos_coeff * sin_off)
            cos_coeff = (base_cos_coeff * cos_off) - (base_sin_coeff * sin_off)

        harmonic_rank_ctx = (
            torch.einsum("bqk,bqh,bkhr->bqr", weights, sin_basis, sin_coeff)
            + torch.einsum("bqk,bqh,bkhr->bqr", weights, cos_basis, cos_coeff)
        )
        harmonic_ctx = self.rank_to_model(self.dropout(harmonic_rank_ctx))
        return base_ctx + harmonic_ctx, logits


# ----------------------------- Neural field decoder ----------------------------


class HierarchicalPerceiverDecoder(nn.Module):
    """Structured decoder with a light mean branch and a dynamic residual branch."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim

        self.spatial_query_encoder = FourierEncoder(2, cfg.spatial_query_fourier_frequencies, include_input=True)
        # Spatial and phase are encoded separately by design.
        self.phase_encoder = PhaseHarmonicEncoder(cfg.phase_fourier_frequencies)
        self.phase_hyper_context = PhaseConditionedHyperContext(cfg)

        self.structured_token_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.dynamic_token_proj = nn.Linear(cfg.dynamic_token_dim, cfg.hidden_dim)
        self.mean_global_token_proj = nn.Linear(
            cfg.behavior_dim + cfg.latent_dim + (2 + cfg.future_global_feature_dim),
            cfg.hidden_dim,
        )
        self.memory_norm = nn.LayerNorm(cfg.hidden_dim)
        self.memory_type_embeddings = nn.Parameter(torch.randn(6, cfg.hidden_dim) * 0.02)

        mean_query_in = self.spatial_query_encoder.output_dim + cfg.behavior_dim + cfg.latent_dim
        residual_query_in = (
            self.spatial_query_encoder.output_dim
            + self.phase_encoder.output_dim
            + (2 * cfg.dynamic_token_dim)
        )
        self.mean_query_proj = MLP(
            mean_query_in,
            cfg.hidden_dim,
            cfg.hidden_dim,
            num_layers=2,
            dropout=cfg.dropout,
            layer_norm=cfg.use_layer_norm,
        )
        self.residual_query_proj = MLP(
            residual_query_in,
            cfg.hidden_dim,
            cfg.hidden_dim,
            num_layers=2,
            dropout=cfg.dropout,
            layer_norm=cfg.use_layer_norm,
        )
        self.phase_film = nn.Linear(self.phase_encoder.output_dim, 2 * cfg.hidden_dim)
        self.phase_context_film = nn.Linear(cfg.dynamic_token_dim, 2 * cfg.hidden_dim)

        self.mean_global_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    cfg.hidden_dim,
                    cfg.perceiver_num_heads,
                    cfg.perceiver_head_dim,
                    cfg.perceiver_ffn_mult,
                    cfg.perceiver_dropout,
                )
                for _ in range(max(int(cfg.perceiver_num_layers_global), 1))
            ]
        )
        self.residual_global_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    cfg.hidden_dim,
                    cfg.perceiver_num_heads,
                    cfg.perceiver_head_dim,
                    cfg.perceiver_ffn_mult,
                    cfg.perceiver_dropout,
                )
                for _ in range(max(int(cfg.perceiver_num_layers_global), 1))
            ]
        )
        self.residual_local_blocks = nn.ModuleList(
            [
                LocalCrossAttentionBlock(
                    cfg.hidden_dim,
                    cfg.perceiver_num_heads,
                    cfg.perceiver_head_dim,
                    cfg.perceiver_ffn_mult,
                    cfg.perceiver_dropout,
                )
                for _ in range(max(int(cfg.perceiver_num_layers_local), 0))
            ]
        )

        self.module_relative_bias = RelativeGeometryBias()
        self.env_relative_bias = RelativeGeometryBias()

        self.mean_head_norm = nn.LayerNorm(cfg.hidden_dim)
        self.residual_head_norm = nn.LayerNorm(cfg.hidden_dim)
        self.mean_head = MLP(cfg.hidden_dim, cfg.decoder_hidden_dim, 4, num_layers=2, dropout=cfg.dropout)
        self.residual_head = MLP(cfg.hidden_dim, cfg.decoder_hidden_dim, 4, num_layers=2, dropout=cfg.dropout)

    def _normalize_query_xy(self, query_xy: torch.Tensor) -> torch.Tensor:
        xy = query_xy.clone()
        xy[..., 0] = xy[..., 0] / max(self.cfg.domain_length_x, 1e-6)
        xy[..., 1] = xy[..., 1] / max(self.cfg.domain_length_y, 1e-6)
        return xy

    @staticmethod
    def _relative_parts(query_xy_norm: torch.Tensor, token_xy_norm: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        dx, dy = periodic_delta_min_image(token_xy_norm[:, None, :, :], query_xy_norm[:, :, None, :])
        dist = torch.sqrt(dx.square() + dy.square() + 1e-8)
        downstream = torch.clamp(dx, min=0.0)
        upstream = torch.clamp(-dx, min=0.0)
        return dx, dy, dist, downstream, upstream

    def _build_relative_attention_bias_parts(
        self,
        organized: Dict[str, torch.Tensor],
        query_xy_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mod_parts = self._relative_parts(query_xy_norm, organized["module_coords_norm"])
        env_parts = self._relative_parts(query_xy_norm, organized["env_coords"])
        module_bias = self.module_relative_bias(*mod_parts)
        env_bias = self.env_relative_bias(*env_parts)
        return module_bias, env_bias

    def _concat_memory(self, pieces: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.memory_norm(torch.cat(list(pieces), dim=1))

    def _build_memory_tokens(
        self,
        organized: Dict[str, torch.Tensor],
        behavior: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        module_tokens = self.structured_token_proj(organized["module_state"]) + self.memory_type_embeddings[0]
        env_tokens = self.structured_token_proj(organized["env_state"]) + self.memory_type_embeddings[1]
        hyper_tokens = self.structured_token_proj(organized["hyper_state"]) + self.memory_type_embeddings[2]
        dynamic_hyper_tokens = self.dynamic_token_proj(behavior["dynamic_hyper_tokens"]) + self.memory_type_embeddings[3]
        dynamic_global_token = self.dynamic_token_proj(behavior["dynamic_global_token"]) + self.memory_type_embeddings[5]

        mean_global_inputs = torch.cat(
            [behavior["behavior_latent"], behavior["mean_latent"], organized["global_features"]],
            dim=-1,
        )
        mean_global_token = self.mean_global_token_proj(mean_global_inputs).unsqueeze(1) + self.memory_type_embeddings[4]

        batch_size = module_tokens.shape[0]
        module_mask = organized["cyl_mask"]
        env_mask = torch.ones(organized["env_state"].shape[:2], device=module_tokens.device, dtype=module_mask.dtype)
        hyper_mask = torch.ones(organized["hyper_state"].shape[:2], device=module_tokens.device, dtype=module_mask.dtype)
        mean_global_mask = torch.ones((batch_size, 1), device=module_tokens.device, dtype=module_mask.dtype)
        dynamic_global_mask = torch.ones((batch_size, 1), device=module_tokens.device, dtype=module_mask.dtype)

        return {
            "module_tokens": module_tokens,
            "env_tokens": env_tokens,
            "hyper_tokens": hyper_tokens,
            "dynamic_hyper_tokens": dynamic_hyper_tokens,
            "dynamic_global_token": dynamic_global_token,
            "mean_global_token": mean_global_token,
            "module_mask": module_mask,
            "env_mask": env_mask,
            "hyper_mask": hyper_mask,
            "mean_global_mask": mean_global_mask,
            "dynamic_global_mask": dynamic_global_mask,
        }

    def _zero_bias(self, query_xy_norm: torch.Tensor, count: int) -> torch.Tensor:
        return torch.zeros(
            query_xy_norm.shape[0],
            query_xy_norm.shape[1],
            count,
            device=query_xy_norm.device,
            dtype=query_xy_norm.dtype,
        )

    def _apply_phase_modulation(self, residual_query: torch.Tensor, phase_feat: torch.Tensor) -> torch.Tensor:
        scale, bias = self.phase_film(phase_feat).chunk(2, dim=-1)
        return residual_query * (1.0 + 0.1 * torch.tanh(scale)) + (0.1 * bias)

    def _apply_phase_context_modulation(self, residual_query: torch.Tensor, phase_context: torch.Tensor) -> torch.Tensor:
        scale, bias = self.phase_context_film(phase_context).chunk(2, dim=-1)
        return residual_query * (1.0 + 0.1 * torch.tanh(scale)) + (0.1 * bias)

    @staticmethod
    def _gather_token_subset(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        batch_index = torch.arange(batch_size, device=tokens.device)[:, None, None]
        return tokens[batch_index, indices]

    @staticmethod
    def _gather_bias_subset(bias: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch_size, num_queries, _ = indices.shape
        batch_index = torch.arange(batch_size, device=bias.device)[:, None, None]
        query_index = torch.arange(num_queries, device=bias.device)[None, :, None]
        return bias[batch_index, query_index, indices]

    def _build_local_context(
        self,
        query_xy_chunk_norm: torch.Tensor,
        organized: Dict[str, torch.Tensor],
        memory: Dict[str, torch.Tensor],
        module_bias_chunk: torch.Tensor,
        env_bias_chunk: torch.Tensor,
        hyper_wake_logits_chunk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, chunk_queries, _ = query_xy_chunk_norm.shape

        _, _, module_dist, _, _ = self._relative_parts(query_xy_chunk_norm, organized["module_coords_norm"])
        topk_mod = max(1, min(int(self.cfg.perceiver_refine_topk_mod), module_dist.shape[-1]))

        _, _, env_dist, _, _ = self._relative_parts(query_xy_chunk_norm, organized["env_coords"])
        topk_env = max(1, min(int(self.cfg.perceiver_refine_topk_env), env_dist.shape[-1]))

        module_valid = organized["cyl_mask"][:, None, :] > 0
        local_topk_mode = str(self.cfg.local_topk_mode).strip().lower()
        if local_topk_mode == "distance":
            module_rank = module_dist.masked_fill(~module_valid, float("inf"))
            env_rank = env_dist
            mod_indices = torch.topk(module_rank, k=topk_mod, dim=-1, largest=False).indices
            env_indices = torch.topk(env_rank, k=topk_env, dim=-1, largest=False).indices
        elif local_topk_mode == "wake_relevance":
            if self.cfg.local_topk_use_softmax_hyper_weights:
                hyper_weights = torch.softmax(hyper_wake_logits_chunk, dim=-1)
            else:
                hyper_weights = hyper_wake_logits_chunk
            module_hyper_relevance = torch.einsum("bqk,bnk->bqn", hyper_weights, organized["A_mh"])
            env_hyper_relevance = torch.einsum("bqk,bmk->bqm", hyper_weights, organized["A_eh"])

            distance_weight = float(self.cfg.local_topk_distance_weight)
            hyper_weight = float(self.cfg.local_topk_hyper_weight)
            attention_bias_weight = float(self.cfg.local_topk_attention_bias_weight)
            module_score = (
                -distance_weight * module_dist
                + hyper_weight * module_hyper_relevance
                + attention_bias_weight * module_bias_chunk
            )
            env_score = (
                -distance_weight * env_dist
                + hyper_weight * env_hyper_relevance
                + attention_bias_weight * env_bias_chunk
            )
            score_floor = torch.finfo(module_score.dtype).min
            module_score = module_score.masked_fill(~module_valid, score_floor)
            if self.cfg.local_topk_detach_scores:
                module_score = module_score.detach()
                env_score = env_score.detach()
            mod_indices = torch.topk(module_score, k=topk_mod, dim=-1, largest=True).indices
            env_indices = torch.topk(env_score, k=topk_env, dim=-1, largest=True).indices
        else:
            raise ValueError(
                f"Unsupported local_topk_mode={self.cfg.local_topk_mode!r}; "
                "expected 'wake_relevance' or 'distance'."
            )

        mod_tokens = self._gather_token_subset(memory["module_tokens"], mod_indices)
        mod_bias = self._gather_bias_subset(module_bias_chunk, mod_indices)
        mod_mask = self._gather_token_subset(memory["module_mask"].unsqueeze(-1), mod_indices).squeeze(-1)

        env_tokens = self._gather_token_subset(memory["env_tokens"], env_indices)
        env_bias = self._gather_bias_subset(env_bias_chunk, env_indices)
        env_mask = torch.ones(batch_size, chunk_queries, topk_env, device=query_xy_chunk_norm.device, dtype=query_xy_chunk_norm.dtype)

        # Stage-2 local refinement explicitly gathers top-k env/module tokens,
        # then appends all hyper-level dynamic memory without a dense Q x N_ctx x D expansion.
        hyper_tokens = memory["hyper_tokens"][:, None, :, :].expand(-1, chunk_queries, -1, -1)
        dynamic_hyper_tokens = memory["dynamic_hyper_tokens"][:, None, :, :].expand(-1, chunk_queries, -1, -1)
        dynamic_global_token = memory["dynamic_global_token"][:, None, :, :].expand(-1, chunk_queries, -1, -1)

        hyper_mask = torch.ones(hyper_tokens.shape[:3], device=query_xy_chunk_norm.device, dtype=query_xy_chunk_norm.dtype)
        dynamic_hyper_mask = torch.ones(dynamic_hyper_tokens.shape[:3], device=query_xy_chunk_norm.device, dtype=query_xy_chunk_norm.dtype)
        dynamic_global_mask = torch.ones(dynamic_global_token.shape[:3], device=query_xy_chunk_norm.device, dtype=query_xy_chunk_norm.dtype)

        local_context = torch.cat(
            [mod_tokens, env_tokens, hyper_tokens, dynamic_hyper_tokens, dynamic_global_token],
            dim=2,
        )
        local_bias = torch.cat(
            [
                mod_bias,
                env_bias,
                hyper_wake_logits_chunk,
                hyper_wake_logits_chunk,
                torch.zeros(batch_size, chunk_queries, 1, device=query_xy_chunk_norm.device, dtype=query_xy_chunk_norm.dtype),
            ],
            dim=-1,
        )
        local_mask = torch.cat([mod_mask, env_mask, hyper_mask, dynamic_hyper_mask, dynamic_global_mask], dim=-1)
        return local_context, local_bias, local_mask

    def _run_local_refinement(
        self,
        residual_query: torch.Tensor,
        query_xy_norm: torch.Tensor,
        organized: Dict[str, torch.Tensor],
        memory: Dict[str, torch.Tensor],
        module_bias: torch.Tensor,
        env_bias: torch.Tensor,
        hyper_wake_logits: torch.Tensor,
    ) -> torch.Tensor:
        if not self.residual_local_blocks:
            return residual_query

        query_chunk_size = max(1, int(self.cfg.perceiver_query_chunk_size))
        refined_chunks = []
        for start in range(0, residual_query.shape[1], query_chunk_size):
            end = min(start + query_chunk_size, residual_query.shape[1])
            query_chunk = residual_query[:, start:end, :]
            xy_chunk = query_xy_norm[:, start:end, :]
            module_bias_chunk = module_bias[:, start:end, :]
            env_bias_chunk = env_bias[:, start:end, :]
            hyper_wake_logits_chunk = hyper_wake_logits[:, start:end, :]
            local_context, local_bias, local_mask = self._build_local_context(
                xy_chunk,
                organized,
                memory,
                module_bias_chunk,
                env_bias_chunk,
                hyper_wake_logits_chunk,
            )
            for block in self.residual_local_blocks:
                query_chunk = block(
                    query_chunk,
                    local_context,
                    attn_bias=local_bias,
                    context_mask=local_mask,
                )
            refined_chunks.append(query_chunk)
        return torch.cat(refined_chunks, dim=1)

    def forward(
        self,
        organized: Dict[str, torch.Tensor],
        behavior: Dict[str, torch.Tensor],
        query_xy: torch.Tensor,
        query_tau: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_queries, _ = query_xy.shape
        query_xy_norm = self._normalize_query_xy(query_xy)
        spatial_feat = self.spatial_query_encoder(query_xy_norm)
        phase_feat = self.phase_encoder(query_tau)

        behavior_latent = behavior["behavior_latent"][:, None, :].expand(batch_size, num_queries, -1)
        mean_latent = behavior["mean_latent"][:, None, :].expand(batch_size, num_queries, -1)
        dynamic_global_token = behavior["dynamic_global_token"].expand(batch_size, num_queries, -1)
        phase_hyper_context, hyper_wake_logits = self.phase_hyper_context(query_xy_norm, query_tau, behavior, organized)

        mean_query = self.mean_query_proj(torch.cat([spatial_feat, behavior_latent, mean_latent], dim=-1))
        residual_query = self.residual_query_proj(
            torch.cat([spatial_feat, phase_feat, dynamic_global_token, phase_hyper_context], dim=-1)
        )
        residual_query = self._apply_phase_modulation(residual_query, phase_feat)
        residual_query = self._apply_phase_context_modulation(residual_query, phase_hyper_context)

        memory = self._build_memory_tokens(organized, behavior)
        module_bias, env_bias = self._build_relative_attention_bias_parts(organized, query_xy_norm)

        mean_memory = self._concat_memory(
            [
                memory["module_tokens"],
                memory["env_tokens"],
                memory["hyper_tokens"],
                memory["mean_global_token"],
            ]
        )
        mean_memory_mask = torch.cat(
            [
                memory["module_mask"],
                memory["env_mask"],
                memory["hyper_mask"],
                memory["mean_global_mask"],
            ],
            dim=1,
        )
        mean_attn_bias = torch.cat(
            [
                module_bias,
                env_bias,
                self._zero_bias(query_xy_norm, memory["hyper_tokens"].shape[1]),
                self._zero_bias(query_xy_norm, 1),
            ],
            dim=-1,
        )

        residual_memory = self._concat_memory(
            [
                memory["module_tokens"],
                memory["env_tokens"],
                memory["hyper_tokens"],
                memory["dynamic_hyper_tokens"],
                memory["dynamic_global_token"],
            ]
        )
        residual_memory_mask = torch.cat(
            [
                memory["module_mask"],
                memory["env_mask"],
                memory["hyper_mask"],
                memory["hyper_mask"],
                memory["dynamic_global_mask"],
            ],
            dim=1,
        )
        residual_attn_bias = torch.cat(
            [
                module_bias,
                env_bias,
                hyper_wake_logits,
                hyper_wake_logits,
                self._zero_bias(query_xy_norm, 1),
            ],
            dim=-1,
        )

        query_chunk_size = max(1, int(self.cfg.perceiver_query_chunk_size))
        for block in self.mean_global_blocks:
            mean_query = block(
                mean_query,
                mean_memory,
                attn_bias=mean_attn_bias,
                context_mask=mean_memory_mask,
                query_chunk_size=query_chunk_size,
            )
        for block in self.residual_global_blocks:
            # Stage 1: global read over all structured memory.
            residual_query = block(
                residual_query,
                residual_memory,
                attn_bias=residual_attn_bias,
                context_mask=residual_memory_mask,
                query_chunk_size=query_chunk_size,
            )

        # Stage 2: local refinement over gathered nearby env/module tokens.
        residual_query = self._run_local_refinement(
            residual_query,
            query_xy_norm,
            organized,
            memory,
            module_bias,
            env_bias,
            hyper_wake_logits,
        )

        pred_mean = self.mean_head(self.mean_head_norm(mean_query))
        pred_residual = self.residual_head(self.residual_head_norm(residual_query))
        pred_field = pred_mean + pred_residual

        return {
            "pred_mean": pred_mean,
            "pred_residual": pred_residual,
            "pred_field": pred_field,
        }


# ----------------------------- Full model wrapper ------------------------------


class HypergraphNeuralFieldModel(nn.Module):
    """End-to-end model wrapper used by training and evaluation scripts."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.organizer = HypergraphOrganizer(cfg)
        self.behavior_head = BehaviorHead(cfg)
        self.decoder = HierarchicalPerceiverDecoder(cfg)

    def forward(
        self,
        structure: Dict[str, torch.Tensor],
        query_xy: torch.Tensor,
        query_tau: torch.Tensor,
        return_aux: bool = True,
    ) -> Dict[str, torch.Tensor]:
        organized = self.organizer(
            re_values=structure["re_values"],
            num_cylinders=structure["num_cylinders"],
            centers=structure["centers"],
            cyl_mask=structure["cyl_mask"],
            extra_global=structure.get("extra_global"),
            extra_module=structure.get("extra_module"),
        )
        behavior = self.behavior_head(organized)
        decoded = self.decoder(organized, behavior, query_xy=query_xy, query_tau=query_tau)

        out = {
            "pred_field": decoded["pred_field"],
            "pred_mean": decoded["pred_mean"],
            "pred_residual": decoded["pred_residual"],
            "freq_pred": behavior["freq_pred"],
        }
        if return_aux:
            aux = {
                "behavior_latent": behavior["behavior_latent"],
                "mean_latent": behavior["mean_latent"],
                "dynamic_global_token": behavior["dynamic_global_token"],
                "dynamic_hyper_base": behavior["dynamic_hyper_base"],
                "dynamic_hyper_tokens": behavior["dynamic_hyper_tokens"],
                "A_me": organized["A_me"],
                "A_mh": organized["A_mh"],
                "A_eh": organized["A_eh"],
                "module_state": organized["module_state"],
                "env_state": organized["env_state"],
                "hyper_state": organized["hyper_state"],
                "env_coords": organized["env_coords"],
                "module_coords_norm": organized["module_coords_norm"],
                "hyper_source_coords": organized["hyper_source_coords"],
                "hyper_wake_coords": organized["hyper_wake_coords"],
                "hyper_wake_axis": organized["hyper_wake_axis"],
                "hyper_wake_extent": organized["hyper_wake_extent"],
                "hyper_strength": organized["hyper_strength"],
                "global_features": organized["global_features"],
                "cyl_mask": organized["cyl_mask"],
            }
            if "hyper_phase_offsets" in behavior:
                aux["hyper_phase_offsets"] = behavior["hyper_phase_offsets"]
            if "hyper_phase_sin_coeff" in behavior:
                aux["hyper_phase_sin_coeff"] = behavior["hyper_phase_sin_coeff"]
            if "hyper_phase_cos_coeff" in behavior:
                aux["hyper_phase_cos_coeff"] = behavior["hyper_phase_cos_coeff"]
            out.update(aux)
        return out

    def reconstruct_full_grid(
        self,
        structure: Dict[str, torch.Tensor],
        x_grid: torch.Tensor,
        y_grid: torch.Tensor,
        tau: torch.Tensor,
        query_batch_size: int = 16384,
    ) -> Dict[str, torch.Tensor]:
        if x_grid.ndim != 2 or y_grid.ndim != 2:
            raise ValueError("x_grid and y_grid must be rank-2 tensors [H, W].")

        device = x_grid.device
        height, width = x_grid.shape
        xy = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)[None, ...]
        tau = tau.reshape(1, 1).to(device=device, dtype=xy.dtype)
        tau_full = tau[:, None, :].expand(1, xy.shape[1], 1)

        field_chunks = []
        mean_chunks = []
        residual_chunks = []
        aux_cache = None
        for start in range(0, xy.shape[1], query_batch_size):
            end = min(start + query_batch_size, xy.shape[1])
            chunk_out = self.forward(
                structure,
                xy[:, start:end],
                tau_full[:, start:end],
                return_aux=(aux_cache is None),
            )
            field_chunks.append(chunk_out["pred_field"])
            mean_chunks.append(chunk_out["pred_mean"])
            residual_chunks.append(chunk_out["pred_residual"])
            if aux_cache is None:
                aux_cache = {
                    key: value
                    for key, value in chunk_out.items()
                    if key not in {"pred_field", "pred_mean", "pred_residual"}
                }

        result = {
            "pred_field": torch.cat(field_chunks, dim=1).reshape(1, height, width, 4),
            "pred_mean": torch.cat(mean_chunks, dim=1).reshape(1, height, width, 4),
            "pred_residual": torch.cat(residual_chunks, dim=1).reshape(1, height, width, 4),
        }
        if aux_cache is not None:
            result.update(aux_cache)
        return result


def build_model_from_config(model_cfg: Dict) -> HypergraphNeuralFieldModel:
    cfg = ModelConfig.from_dict(model_cfg)
    return HypergraphNeuralFieldModel(cfg)
