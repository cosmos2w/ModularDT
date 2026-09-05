"""Geometry-aware Set-Transformer/Perceiver-style field adaptation."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from honf_forward_core.nn import FourierFeatures, LazyMLP
from .common import BiasedMultiheadAttention, PreNormAttentionBlock
from .types import EncodedInterfaceCase


def _reference_coordinates(count: int, dimension: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return deterministic normalized lattice references, truncated to count."""

    side = max(1, int(math.ceil(float(count) ** (1.0 / float(dimension)))))
    axis = (torch.arange(side, device=device, dtype=dtype) + 0.5) / float(side)
    mesh = torch.meshgrid(*([axis] * dimension), indexing="ij")
    return torch.stack([value.reshape(-1) for value in mesh], dim=-1)[:count]


class GeometryLatentField(nn.Module):
    """Geometry-aware global latent backend; not an exact UPT/Transolver reproduction."""

    def __init__(
        self,
        hidden_dim: int,
        latent_count: int,
        latent_blocks: int,
        num_heads: int,
        fourier_frequencies: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_count = int(latent_count)
        self.num_heads = int(num_heads)
        self.relative_fourier = FourierFeatures(None, fourier_frequencies)
        self.seeds = nn.Parameter(torch.randn(latent_count, hidden_dim) / math.sqrt(float(hidden_dim)))
        self.module_attention = BiasedMultiheadAttention(hidden_dim, num_heads)
        self.env_attention = BiasedMultiheadAttention(hidden_dim, num_heads)
        self.module_bias = LazyMLP(hidden_dim, out_dim=num_heads, num_layers=2)
        self.env_bias = LazyMLP(hidden_dim, out_dim=num_heads, num_layers=2)
        self.blocks = nn.ModuleList(
            PreNormAttentionBlock(hidden_dim, num_heads) for _ in range(int(latent_blocks))
        )
        self.read_query = LazyMLP(hidden_dim, num_layers=2)
        self.read_bias = LazyMLP(hidden_dim, out_dim=num_heads, num_layers=2)
        self.read_attention = BiasedMultiheadAttention(hidden_dim, num_heads)

    def prepare(self, encoded: EncodedInterfaceCase, module_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = int(module_states.shape[0])
        dimension = int(encoded.module_centers.shape[-1])
        refs_norm = _reference_coordinates(self.latent_count, dimension, module_states.device, module_states.dtype)
        refs = refs_norm * encoded.coordinate_scale.reshape(1, dimension)
        refs_batch = refs.unsqueeze(0).expand(batch, -1, -1)
        seeds = self.seeds.unsqueeze(0).expand(batch, -1, -1)
        module_relative = (refs_batch[:, :, None, :] - encoded.module_centers[:, None, :, :]) / encoded.coordinate_scale
        module_bias = self.module_bias(self.relative_fourier(module_relative)).permute(0, 3, 1, 2)
        env_relative = (refs_batch[:, :, None, :] - encoded.env_coords[:, None, :, :]) / encoded.coordinate_scale
        env_bias = self.env_bias(self.relative_fourier(env_relative)).permute(0, 3, 1, 2)
        module_context, module_attention = self.module_attention(
            seeds, module_states, bias=module_bias, source_mask=encoded.module_present, return_attention=True
        )
        env_context, env_attention = self.env_attention(
            seeds,
            encoded.env_tokens,
            bias=env_bias,
            log_weights=torch.log(encoded.env_weights.clamp_min(torch.finfo(encoded.env_weights.dtype).tiny)),
            return_attention=True,
        )
        latents = seeds + module_context + env_context
        for block in self.blocks:
            latents = block(latents)
        return {
            "latents": latents,
            "reference_coords": refs_batch,
            "module_attention": module_attention,
            "environment_attention": env_attention,
        }

    def read(
        self,
        state: Dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        query = self.read_query(
            torch.cat(
                [receiver_features, encoded.global_token[:, None, :].expand(-1, receivers.shape[1], -1)],
                dim=-1,
            )
        )
        relative = (receivers[:, :, None, :] - state["reference_coords"][:, None, :, :]) / encoded.coordinate_scale
        bias = self.read_bias(self.relative_fourier(relative)).permute(0, 3, 1, 2)
        context, attention = self.read_attention(query, state["latents"], bias=bias, return_attention=True)
        return context, {
            "latent_query_attention": attention,
            "latent_count": context.new_full((context.shape[0],), float(self.latent_count)),
        }
