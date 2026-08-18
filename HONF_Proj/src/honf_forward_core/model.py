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

from .decoder import HypergraphFieldDecoder
from .organizer import HypergraphOrganizerCore
from .config import BatchData, UnifiedForwardConfig
from .nn import FourierFeatures, LazyMLP


class HONFNeuralField(nn.Module):
    """Reusable HONF field model used by domain-specific wrappers."""

    def __init__(self, config: UnifiedForwardConfig):
        """Initialize HONFNeuralField and its required state."""

        super().__init__()

        self.config = config
        hidden_dim = int(config.hidden_dim)

        self.global_encoder          = LazyMLP(hidden_dim, dropout=float(config.dropout), include_zero_dropout=True)
        self.module_feature_encoder  = LazyMLP(hidden_dim, dropout=float(config.dropout), include_zero_dropout=True)
        self.position_fourier        = FourierFeatures(None, int(config.position_fourier_frequencies))
        self.module_position_encoder = LazyMLP(hidden_dim, dropout=float(config.dropout), include_zero_dropout=True)
        self.env_encoder             = LazyMLP(hidden_dim, dropout=float(config.dropout), include_zero_dropout=True)

        self.organizer               = HypergraphOrganizerCore(config)
        self.decoder                 = HypergraphFieldDecoder(config)

    def set_edge_capacity(self, capacity: int) -> None:
        """Set the runtime candidate-edge budget for exchangeable organization."""

        self.organizer.set_edge_capacity(capacity)

    def set_training_progress(self, *, epoch: int, total_epochs: Optional[int] = None) -> None:
        """Set organizer warmup progress once per training epoch."""

        self.organizer.set_training_progress(epoch=epoch, total_epochs=total_epochs)

    def encode_and_organize(self, batch: BatchData) -> Dict[str, torch.Tensor]:
        """Encode generic inputs and build static HONF organizer state.

        Input shapes use ``B`` cases, ``M`` module slots, ``E`` environment
        tokens, and feature widths ``Fm``/``Fg``: module centers ``[B,M,2]``,
        module features ``[B,M,Fm]``, presence mask ``[B,M]``, and global
        context ``[B,Fg]``. Optional environment coordinates/features are
        ``[E,2]`` or ``[B,E,2]`` and ``[B,E,Fe]``. The result contains encoded
        module/environment tokens ``[B,M,H]``/``[B,E,H]``, global token
        ``[B,H]``, incidences ``A_mh [B,M,K]`` and ``A_eh [B,E,K]``, and
        hyperedge state ``[B,K,H]``.

        No query field is decoded here. Domain wrappers may refine module
        tokens and call :meth:`decode_queries` without repeating case encoding.
        """
        cfg = self.config
        module_centers = batch.module_centers.float()
        module_present = batch.module_present.float()
        module_features = batch.module_features.float()
        global_context_raw = batch.global_context.float()
        query_xy = batch.query_xy.float()
        scale_x, scale_y = cfg.spatial_scale()

        global_token = self.global_encoder(global_context_raw)
        module_pos = torch.stack(
            [
                module_centers[..., 0] / max(scale_x, 1e-6),
                module_centers[..., 1] / max(scale_y, 1e-6),
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
        # Preserve per-case coordinates. Earlier code encoded only env_coords[0],
        # which was correct solely when every case shared one fixed domain.
        env_coords_for_features = env_coords
        env_norm = torch.stack(
            [
                env_coords_for_features[..., 0] / max(scale_x, 1e-6),
                env_coords_for_features[..., 1] / max(scale_y, 1e-6),
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
        if cfg.decoder_uses("global"):
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
        query_features: Optional[torch.Tensor] = None,
        *,
        return_routing_maps: bool = False,
        return_edge_fields: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Map queries ``[B,Q,2]`` and organized state to fields ``[B,Q,F]``."""

        return self.decoder(
            query_xy=query_xy,
            query_time=query_time,
            organizer_output=organizer_output,
            global_context=global_token,
            query_features=query_features,
            return_routing_maps=bool(return_routing_maps),
            return_edge_fields=bool(return_edge_fields),
        )

    def forward(self, batch: BatchData, *, return_edge_fields: bool = False) -> Dict[str, torch.Tensor]:
        """Encode, organize, and decode a complete :class:`BatchData` batch."""

        encoded = self.encode_and_organize(batch)
        decoder_output = self.decode_queries(
            query_xy=batch.query_xy.float(),
            query_time=None if batch.query_time is None else batch.query_time.float(),
            organizer_output=encoded,
            global_token=encoded["global_token"],
            query_features=None if batch.query_features is None else batch.query_features.float(),
            return_edge_fields=bool(return_edge_fields),
        )
        output: Dict[str, torch.Tensor] = {}
        output.update(encoded)
        output.update(decoder_output)
        return output

    def _environment_coords(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the default cell-centered rectangular grid as ``[E,2]``."""

        cfg = self.config
        nx = int(cfg.num_env_tokens_x)
        ny = int(cfg.num_env_tokens_y)
        scale_x, scale_y = cfg.spatial_scale()
        xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) / max(float(nx), 1.0) * scale_x
        ys = (torch.arange(ny, device=device, dtype=dtype) + 0.5) / max(float(ny), 1.0) * scale_y
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
