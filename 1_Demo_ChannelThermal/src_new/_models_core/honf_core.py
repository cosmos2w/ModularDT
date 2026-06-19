"""CORE HONF neural field.

Inputs are a generic `BatchData` object with module centers/features, global
context, query coordinates, optional query time, and optional generic
environment coordinates/features. Outputs are field predictions and organizer
routing diagnostics. This module is reusable across domains and does not know
about ChannelThermal walls, inlet/outlet distances, or materials.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .honf_decoder import HypergraphFieldDecoder
from .honf_organizer import HypergraphOrganizerCore
from .honf_types import BatchData, UnifiedForwardConfig


class FourierFeatures(nn.Module):
    """Append powers-of-two sin/cos Fourier features to the input."""

    def __init__(self, num_frequencies: int):
        super().__init__()
        self.num_frequencies = max(0, int(num_frequencies))
        if self.num_frequencies > 0:
            frequencies = (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)) * torch.pi
        else:
            frequencies = torch.empty(0, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_frequencies <= 0:
            return x
        frequencies = self.frequencies.to(device=x.device, dtype=x.dtype).view(*([1] * (x.ndim - 1)), -1, 1)
        angles = x.unsqueeze(-2) * frequencies
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-2).flatten(start_dim=-2)
        return torch.cat([x, encoded], dim=-1)


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


class HONFNeuralField(nn.Module):
    """Reusable HONF field model used by domain-specific wrappers."""

    def __init__(self, config: UnifiedForwardConfig):
        super().__init__()

        self.config = config
        hidden_dim = int(config.hidden_dim)

        self.global_encoder          = LazyMLP(hidden_dim, float(config.dropout))
        self.module_feature_encoder  = LazyMLP(hidden_dim, float(config.dropout))
        self.position_fourier        = FourierFeatures(int(config.position_fourier_frequencies))
        self.module_position_encoder = LazyMLP(hidden_dim, float(config.dropout))
        self.env_encoder             = LazyMLP(hidden_dim, float(config.dropout))

        self.organizer               = HypergraphOrganizerCore(config)
        self.decoder                 = HypergraphFieldDecoder(config)

    def encode_and_organize(self, batch: BatchData) -> Dict[str, torch.Tensor]:
        """Encode generic inputs and build static HONF organizer state.

        This CORE HONF method intentionally does not decode query fields. Domain
        wrappers can fuse case-specific information into module tokens and then
        call `decode_queries()` exactly once for the final field.
        """
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
        if cfg.use_position_fourier_for_modules:
            module_pos_encoded = self.position_fourier(module_pos)
        else:
            module_pos_encoded = module_pos
        module_tokens = self.module_feature_encoder(module_features) + self.module_position_encoder(module_pos_encoded)
        module_tokens = module_tokens * module_present.unsqueeze(-1)

        env_coords = (
            self._environment_coords(query_xy.device, query_xy.dtype)
            if batch.env_coords is None
            else batch.env_coords.to(device=query_xy.device, dtype=query_xy.dtype)
        )
        env_coords_for_features = env_coords[0] if env_coords.ndim == 3 else env_coords
        env_norm = torch.stack(
            [
                env_coords_for_features[..., 0] / max(float(cfg.domain_length_x), 1e-6),
                env_coords_for_features[..., 1] / max(float(cfg.domain_length_y), 1e-6),
            ],
            dim=-1,
        )
        if cfg.use_position_fourier_for_env:
            env_pos_encoded = self.position_fourier(env_norm)
        else:
            env_pos_encoded = env_norm
        env_encoded_input = env_pos_encoded
        if batch.env_features is not None:
            env_features = batch.env_features.to(device=query_xy.device, dtype=query_xy.dtype)
            if env_features.ndim == 3:
                if env_encoded_input.ndim == 2:
                    env_encoded_input = env_encoded_input.unsqueeze(0).expand(env_features.shape[0], -1, -1)
                env_encoded_input = torch.cat([env_encoded_input, env_features], dim=-1)
            else:
                env_encoded_input = torch.cat([env_encoded_input, env_features], dim=-1)
        env_tokens = self.env_encoder(env_encoded_input)
        if env_tokens.ndim == 2:
            env_tokens = env_tokens.unsqueeze(0).expand(query_xy.shape[0], -1, -1)
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
        organizer_output["module_features_raw"] = module_features
        output: Dict[str, torch.Tensor] = {}
        output.update(organizer_output)
        output["global_token"] = global_token
        output["module_tokens"] = module_tokens
        output["env_tokens"] = env_tokens
        output["module_features_raw"] = module_features
        return output

    def decode_queries(
        self,
        query_xy: torch.Tensor,
        query_time: Optional[torch.Tensor],
        organizer_output: Dict[str, torch.Tensor],
        global_token: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Decode field values from already-organized HONF state."""

        return self.decoder(
            query_xy=query_xy,
            query_time=query_time,
            organizer_output=organizer_output,
            global_context=global_token,
            return_routing_maps=bool(return_routing_maps),
        )

    def forward(self, batch: BatchData) -> Dict[str, torch.Tensor]:
        encoded = self.encode_and_organize(batch)
        decoder_output = self.decode_queries(
            query_xy=batch.query_xy.float(),
            query_time=None if batch.query_time is None else batch.query_time.float(),
            organizer_output=encoded,
            global_token=encoded["global_token"],
        )
        output: Dict[str, torch.Tensor] = {}
        output.update(encoded)
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


UnifiedHypergraphNeuralField = HONFNeuralField
