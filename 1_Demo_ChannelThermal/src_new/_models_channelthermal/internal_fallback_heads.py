"""CHANNELTHERMAL-SPECIFIC fallback internal/interface heads.

Inputs are module tokens from the global HONF wrapper, local disk query points,
module-present masks, and requested interface angular resolution. Outputs are
legacy-shaped internal temperature and interface temperature/flux tensors.
These heads are ChannelThermal-specific comparison/fallback heads and are not a
replacement for the pretrained local surrogate. They are disabled by default in
Phase 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from _helpers.model_utils import FourierEncoder, MLP


@dataclass
class FallbackHeadConfig:
    hidden_dim: int = 128
    internal_query_dim: int = 2
    interface_dim: int = 2
    fourier_frequencies: int = 4
    dropout: float = 0.05


class GlobalFallbackHeads(nn.Module):
    """Legacy-compatible global internal/interface fallback heads."""

    def __init__(self, module_token_dim: int, config: FallbackHeadConfig):
        super().__init__()
        self.config = config
        self.local_coord_encoder = FourierEncoder(
            int(config.internal_query_dim),
            int(config.fourier_frequencies),
            include_input=True,
        )
        self.interface_theta_encoder = FourierEncoder(3, 2, include_input=True)
        self.internal_head = MLP(
            int(module_token_dim) + self.local_coord_encoder.output_dim,
            int(config.hidden_dim),
            1,
            num_layers=3,
            dropout=float(config.dropout),
            layer_norm=True,
        )
        self.interface_head = MLP(
            int(module_token_dim) + self.interface_theta_encoder.output_dim,
            int(config.hidden_dim),
            int(config.interface_dim),
            num_layers=3,
            dropout=float(config.dropout),
            layer_norm=True,
        )

    def predict_internal(
        self,
        module_tokens: torch.Tensor,
        local_query_points: torch.Tensor | None,
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        if local_query_points is None:
            return module_tokens.new_empty(module_tokens.shape[0], module_tokens.shape[1], 0, 1)
        if local_query_points.ndim == 3:
            local_query_points = local_query_points[:, None, :, :].expand(-1, module_tokens.shape[1], -1, -1)
        batch, num_modules, num_points, _ = local_query_points.shape
        coord_features = self.local_coord_encoder(local_query_points)
        module_context = module_tokens[:, :, None, :].expand(-1, -1, num_points, -1)
        pred = self.internal_head(torch.cat([module_context, coord_features], dim=-1))
        return pred * module_present[:, :, None, None].to(dtype=pred.dtype)

    def predict_interface(
        self,
        module_tokens: torch.Tensor,
        *,
        ntheta: int,
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        theta_tokens = self.fixed_theta_tokens(int(ntheta), module_tokens.device, module_tokens.dtype)
        theta_features = self.interface_theta_encoder(theta_tokens).view(1, 1, int(ntheta), -1)
        theta_features = theta_features.expand(module_tokens.shape[0], module_tokens.shape[1], -1, -1)
        module_context = module_tokens[:, :, None, :].expand(-1, -1, int(ntheta), -1)
        pred = self.interface_head(torch.cat([module_context, theta_features], dim=-1))
        return pred * module_present[:, :, None, None].to(dtype=pred.dtype)

    @staticmethod
    def fixed_theta_tokens(ntheta: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        theta = torch.linspace(0.0, 2.0 * math.pi, max(int(ntheta), 1) + 1, device=device, dtype=dtype)[:-1]
        return torch.stack([theta, torch.cos(theta), torch.sin(theta)], dim=-1)
