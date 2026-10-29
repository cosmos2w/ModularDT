"""CHANNELTHERMAL-SPECIFIC Stage-A local module thermal surrogate.

Port tokens are boundary/condition inputs: theta geometry plus outside
temperature and heat-transfer proxy. Interface targets are supervised outputs:
surface temperature and outward normal heat flux. The learned
``module_response_latent`` is the compact response token later consumed by the
global HONF wrapper. Inputs are one-module parameter vectors, per-angle port
tokens, and optional local disk query points. Outputs are internal temperature,
interface prediction, and module response latent tensors. This model is
ChannelThermal-specific and its architecture/state-dict names are preserved for
strict old-checkpoint compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from honf_runtime.compat import FourierEncoder, MLP, dataclass_from_dict, dataclass_to_dict


@dataclass
class LocalModuleConfig:
    """Architecture dimensions for the single-module Stage-A surrogate."""

    module_param_dim: int = 7
    port_token_dim: int = 5
    interface_target_dim: int = 2
    hidden_dim: int = 128
    latent_dim: int = 128
    num_port_latents: int = 8
    num_heads: int = 4
    num_layers: int = 2
    coord_fourier_frequencies: int = 4
    dropout: float = 0.05

    @classmethod
    def from_dict(cls, payload: Dict) -> "LocalModuleConfig":
        """Validate a dictionary and construct a local-surrogate configuration."""

        return dataclass_from_dict(cls, payload)

    def to_dict(self) -> Dict:
        """Serialize this configuration to plain Python values."""

        return dataclass_to_dict(self)


class CrossAttentionBlock(nn.Module):
    """Perceiver-style latent cross-attention followed by a small feed-forward block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        """Initialize CrossAttentionBlock and its required state."""

        super().__init__()
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, 4 * hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

    def forward(self, latents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Update latent queries ``[B,L,H]`` from port tokens ``[B,P,H]``."""

        attn_out, _ = self.attn(
            self.q_norm(latents),
            self.kv_norm(tokens),
            self.kv_norm(tokens),
            need_weights=False,
        )
        latents = latents + attn_out
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


class LocalModuleSurrogate(nn.Module):
    """Local neural-field surrogate for one heated circular module."""

    def __init__(self, config: LocalModuleConfig):
        """Initialize LocalModuleSurrogate and its required state."""

        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        latent = int(config.latent_dim)
        self.module_param_encoder = MLP(
            config.module_param_dim,
            hidden,
            hidden,
            num_layers=3,
            dropout=config.dropout,
            layer_norm=True,
        )
        self.port_token_encoder = MLP(
            config.port_token_dim,
            hidden,
            hidden,
            num_layers=3,
            dropout=config.dropout,
            layer_norm=True,
        )
        self.latent_queries = nn.Parameter(torch.randn(config.num_port_latents, hidden) * 0.02)
        self.latent_condition = nn.Linear(hidden, hidden)
        self.cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )
        self.module_latent_head = MLP(2 * hidden, hidden, latent, num_layers=2, dropout=config.dropout, layer_norm=True)
        self.coord_encoder = FourierEncoder(2, config.coord_fourier_frequencies, include_input=True)
        self.internal_decoder = MLP(
            self.coord_encoder.output_dim + latent,
            hidden,
            1,
            num_layers=4,
            dropout=config.dropout,
            layer_norm=True,
        )
        self.interface_decoder = MLP(
            hidden + latent,
            hidden,
            config.interface_target_dim,
            num_layers=3,
            dropout=config.dropout,
            layer_norm=True,
        )

    def forward(
        self,
        module_params: torch.Tensor,
        port_tokens: torch.Tensor,
        internal_query_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict internal temperature and interface response.

        Args:
            module_params: Known-before-solve module inputs with shape ``[B, P]``.
            port_tokens: Boundary/condition input tokens with shape ``[B, Ntheta, F]``.
            internal_query_points: Local ``(xi, eta)`` points with shape ``[B, Q, 2]``.

        Returns:
            Internal temperature ``[B,Q,1]``, interface temperature/normal flux
            ``[B,Ntheta,2]``, and one response latent ``[B,latent_dim]``.
        """
        module_params = module_params.float()
        port_tokens = port_tokens.float()
        batch_size = module_params.shape[0]

        param_state = self.module_param_encoder(module_params)
        port_state = self.port_token_encoder(port_tokens)
        latents = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        latents = latents + self.latent_condition(param_state).unsqueeze(1)
        for block in self.cross_blocks:
            latents = block(latents, port_state)

        pooled = latents.mean(dim=1)
        z_module = self.module_latent_head(torch.cat([pooled, param_state], dim=-1))

        if internal_query_points is not None:
            coord_features = self.coord_encoder(internal_query_points.float())
            z_query = z_module.unsqueeze(1).expand(-1, coord_features.shape[1], -1)
            internal_temperature = self.internal_decoder(torch.cat([coord_features, z_query], dim=-1))
        else:
            internal_temperature = torch.empty(
                batch_size,
                0,
                1,
                device=module_params.device,
                dtype=module_params.dtype,
            )

        z_port = z_module.unsqueeze(1).expand(-1, port_state.shape[1], -1)
        interface_pred = self.interface_decoder(torch.cat([port_state, z_port], dim=-1))
        return {
            "internal_temperature": internal_temperature,
            "interface_pred": interface_pred,
            "module_response_latent": z_module,
        }
