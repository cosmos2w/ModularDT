"""Contextualized dense pairwise neural-integral field adaptation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from honf_forward_core.nn import FourierFeatures, LazyMLP
from .common import BiasedMultiheadAttention
from .types import EncodedInterfaceCase


class DensePairwiseField(nn.Module):
    """Strong collective pairwise backend; not an exact paper reproduction."""

    def __init__(self, hidden_dim: int, message_hidden_dim: int, num_heads: int, fourier_frequencies: int, *, activation_checkpointing: bool = False):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.relative_fourier = FourierFeatures(None, fourier_frequencies)
        self.mm_message = LazyMLP(message_hidden_dim, out_dim=hidden_dim, num_layers=3)
        self.me_message = LazyMLP(message_hidden_dim, out_dim=hidden_dim, num_layers=3)
        self.em_message = LazyMLP(message_hidden_dim, out_dim=hidden_dim, num_layers=3)
        self.module_update = LazyMLP(hidden_dim, num_layers=3)
        self.env_update = LazyMLP(hidden_dim, num_layers=3)
        self.query_module_message = LazyMLP(message_hidden_dim, out_dim=hidden_dim, num_layers=3)
        self.query_module_output = nn.Linear(hidden_dim, hidden_dim)
        self.env_query = LazyMLP(hidden_dim, num_layers=2)
        self.env_attention = BiasedMultiheadAttention(hidden_dim, num_heads)
        self.env_geometry_bias = LazyMLP(message_hidden_dim, out_dim=num_heads, num_layers=2)

    def _mlp(self, module: nn.Module, values: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, values, use_reentrant=False)
        return module(values)

    def prepare(self, encoded: EncodedInterfaceCase, module_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        centers = encoded.module_centers
        env_coords = encoded.env_coords
        scale = encoded.coordinate_scale
        present = encoded.module_present
        batch, modules, _ = module_states.shape
        env_count = int(encoded.env_tokens.shape[1])
        active_count = present.sum(dim=1, keepdim=True).clamp_min(0.0)

        mm_rel = (centers[:, :, None, :] - centers[:, None, :, :]) / scale
        mm_features = self.relative_fourier(mm_rel)
        zi = module_states[:, :, None, :].expand(-1, -1, modules, -1)
        zl = module_states[:, None, :, :].expand(-1, modules, -1, -1)
        mm = self._mlp(self.mm_message, torch.cat([zi, zl, mm_features], dim=-1))
        eye = torch.eye(modules, device=present.device, dtype=torch.bool)[None]
        mm_mask = (present[:, :, None] > 0.5) & (present[:, None, :] > 0.5) & ~eye
        a_mm = (mm * mm_mask[..., None]).sum(dim=2) / (1.0 + active_count[..., None])

        me_rel = (centers[:, :, None, :] - env_coords[:, None, :, :]) / scale
        me_features = self.relative_fourier(me_rel)
        zi_env = module_states[:, :, None, :].expand(-1, -1, env_count, -1)
        ej_mod = encoded.env_tokens[:, None, :, :].expand(-1, modules, -1, -1)
        me = self._mlp(self.me_message, torch.cat([zi_env, ej_mod, me_features], dim=-1))
        weighted_me = me * encoded.env_weights[:, None, :, None]
        a_me = weighted_me.sum(dim=2) / encoded.env_weights.sum(dim=1)[:, None, None].clamp_min(1.0e-12)
        a_me = a_me * present[..., None]
        global_modules = encoded.global_token[:, None, :].expand(-1, modules, -1)
        contextual_modules = (
            module_states + self.module_update(torch.cat([module_states, a_mm, a_me, global_modules], dim=-1))
        ) * present[..., None]

        em_rel = -me_rel.transpose(1, 2)
        em_features = self.relative_fourier(em_rel)
        ej_sources = encoded.env_tokens[:, :, None, :].expand(-1, -1, modules, -1)
        zi_for_env = module_states[:, None, :, :].expand(-1, env_count, -1, -1)
        em = self._mlp(self.em_message, torch.cat([ej_sources, zi_for_env, em_features], dim=-1))
        a_em = (em * present[:, None, :, None]).sum(dim=2) / (1.0 + active_count[:, :, None])
        global_env = encoded.global_token[:, None, :].expand(-1, env_count, -1)
        contextual_env = encoded.env_tokens + self.env_update(torch.cat([encoded.env_tokens, a_em, global_env], dim=-1))
        return {"module_tokens": contextual_modules, "env_tokens": contextual_env}

    def read(
        self,
        state: Dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        modules = int(state["module_tokens"].shape[1])
        relative = (receivers[:, :, None, :] - encoded.module_centers[:, None, :, :]) / encoded.coordinate_scale
        relative_features = self.relative_fourier(relative)
        source = state["module_tokens"][:, None, :, :].expand(-1, receivers.shape[1], -1, -1)
        global_features = encoded.global_token[:, None, None, :].expand(-1, receivers.shape[1], modules, -1)
        module_messages = self._mlp(self.query_module_message, torch.cat([source, relative_features, global_features], dim=-1))
        module_context = self.query_module_output(
            (module_messages * encoded.module_present[:, None, :, None]).sum(dim=2)
            / (1.0 + encoded.module_present.sum(dim=1)[:, None, None])
        )

        env_relative = (receivers[:, :, None, :] - encoded.env_coords[:, None, :, :]) / encoded.coordinate_scale
        env_bias = self._mlp(self.env_geometry_bias, self.relative_fourier(env_relative)).permute(0, 3, 1, 2)
        env_query = self.env_query(receiver_features)
        env_context, env_attention = self.env_attention(
            env_query,
            state["env_tokens"],
            bias=env_bias,
            log_weights=torch.log(encoded.env_weights.clamp_min(torch.finfo(encoded.env_weights.dtype).tiny)),
            return_attention=True,
        )
        return module_context + env_context, {
            "dense_environment_attention": env_attention,
            "dense_module_context_norm": torch.linalg.vector_norm(module_context, dim=-1),
        }
