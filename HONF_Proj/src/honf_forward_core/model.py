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

    def selection_state(self) -> Dict[str, Optional[int]]:
        """Return the serialized organizer selection progress."""

        return self.organizer.selection_state()

    def encode_and_organize(
        self,
        batch: BatchData,
        *,
        organizer_selection_override: Optional[str] = None,
        return_residual_interaction_tensor: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode generic inputs and build static HONF organizer state.

        Input shapes use ``B`` cases, ``M`` module slots, ``E`` environment
        tokens, and feature widths ``Fm``/``Fg``: module centers ``[B,M,d]``,
        module features ``[B,M,Fm]``, presence mask ``[B,M]``, and global
        context ``[B,Fg]``. Optional environment coordinates/features are
        ``[E,d]`` or ``[B,E,d]`` and ``[B,E,Fe]``. The result contains encoded
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
        scales = query_xy.new_tensor(cfg.spatial_scale())
        if module_centers.ndim != 3 or module_centers.shape[-1] != cfg.spatial_dim:
            raise ValueError(
                f"module_centers must have shape [B,M,{cfg.spatial_dim}], got {tuple(module_centers.shape)}."
            )
        if query_xy.ndim != 3 or query_xy.shape[-1] != cfg.spatial_dim:
            raise ValueError(
                f"query_xy must have shape [B,Q,{cfg.spatial_dim}], got {tuple(query_xy.shape)}."
            )
        if module_centers.shape[0] != query_xy.shape[0]:
            raise ValueError("module_centers and query_xy must have the same batch dimension.")
        if not torch.isfinite(query_xy).all():
            raise ValueError("query_xy must be finite.")
        if not torch.isfinite(module_centers).all():
            # Inactive adapter padding is allowed to carry source NaNs, but
            # those values must never enter a Fourier/geometry operation.
            active = module_present[..., None] > 0.5
            if bool(torch.isfinite(module_centers).logical_or(~active).all()):
                module_centers = torch.where(active, module_centers, torch.zeros_like(module_centers))
            else:
                raise ValueError("active module_centers must be finite.")
        active = module_present[..., None] > 0.5
        module_centers = torch.where(active, module_centers, torch.zeros_like(module_centers))
        if not torch.isfinite(module_features).all():
            if bool(torch.isfinite(module_features).logical_or(~active).all()):
                module_features = torch.where(active, module_features, torch.zeros_like(module_features))
            else:
                raise ValueError("active module_features must be finite.")
        module_features = torch.where(active, module_features, torch.zeros_like(module_features))

        global_token = self.global_encoder(global_context_raw)
        module_pos = module_centers / scales.clamp_min(1e-6)
        if cfg.use_position_fourier_for_modules:
            module_pos_encoded = self.position_fourier(module_pos)
        else:
            module_pos_encoded = module_pos
        module_tokens = self.module_feature_encoder(module_features) + self.module_position_encoder(module_pos_encoded)
        module_tokens = module_tokens * module_present.unsqueeze(-1)

        if batch.env_coords is None:
            if cfg.spatial_dim != 2:
                raise ValueError(
                    "spatial_dim=3 requires adapter-supplied env_coords; the legacy rectangular default is 2-D."
                )
            env_coords = self._environment_coords(query_xy.device, query_xy.dtype)
        else:
            env_coords = batch.env_coords.to(device=query_xy.device, dtype=query_xy.dtype)
        if env_coords.ndim not in {2, 3} or env_coords.shape[-1] != cfg.spatial_dim:
            raise ValueError(
                f"env_coords must have shape [E,{cfg.spatial_dim}] or [B,E,{cfg.spatial_dim}]."
            )
        if env_coords.ndim == 3 and env_coords.shape[0] != query_xy.shape[0]:
            raise ValueError("Batched env_coords must match the batch dimension.")
        if not torch.isfinite(env_coords).all():
            raise ValueError("env_coords must be finite.")
        # Preserve per-case coordinates. Earlier code encoded only env_coords[0],
        # which was correct solely when every case shared one fixed domain.
        env_coords_for_features = env_coords
        env_norm = env_coords_for_features / scales.clamp_min(1e-6)
        if cfg.use_position_fourier_for_env:
            env_pos_encoded = self.position_fourier(env_norm)
        else:
            env_pos_encoded = env_norm
        env_encoded_input = env_pos_encoded
        if batch.env_features is not None:
            env_features = batch.env_features.to(device=query_xy.device, dtype=query_xy.dtype)
            if env_features.ndim == 2:
                if env_encoded_input.ndim == 3:
                    env_features = env_features.unsqueeze(0).expand(query_xy.shape[0], -1, -1)
            elif env_features.ndim == 3:
                if env_features.shape[0] != query_xy.shape[0]:
                    raise ValueError("Batched env_features must match the batch dimension.")
                if env_encoded_input.ndim == 2:
                    env_encoded_input = env_encoded_input.unsqueeze(0).expand(env_features.shape[0], -1, -1)
            else:
                raise ValueError("env_features must have shape [E,Fe] or [B,E,Fe].")
            if env_features.shape[-2] != env_encoded_input.shape[-2]:
                raise ValueError("env_features must align with env_coords along the environment axis.")
            if env_encoded_input.ndim != env_features.ndim:
                raise ValueError("env_features and env_coords must have compatible batch dimensions.")
            env_encoded_input = torch.cat([env_encoded_input, env_features], dim=-1)
        env_tokens = self.env_encoder(env_encoded_input)
        if env_tokens.ndim == 2:
            env_tokens = env_tokens.unsqueeze(0).expand(query_xy.shape[0], -1, -1)
        if cfg.decoder_uses("global"):
            env_tokens = env_tokens + global_token.unsqueeze(1)

        env_weights = self._environment_weights(
            batch.env_weights,
            batch_size=query_xy.shape[0],
            env_count=int(env_coords.shape[-2]),
            device=query_xy.device,
            dtype=query_xy.dtype,
        )

        organizer_output = self.organizer(
            module_tokens=module_tokens,
            env_tokens=env_tokens,
            module_centers=module_centers,
            env_coords=env_coords,
            module_present=module_present,
            env_weights=env_weights,
            geometry_mode=cfg.geometry_mode,
            selection_override=organizer_selection_override,
            global_token=global_token,
            return_residual_interaction_tensor=bool(return_residual_interaction_tensor),
        )
        organizer_output["module_features_raw"] = module_features
        output: Dict[str, torch.Tensor] = {}
        output.update(organizer_output)
        output["global_token"] = global_token
        output["module_tokens"] = module_tokens
        output["env_tokens"] = env_tokens
        output["module_features_raw"] = module_features
        if env_weights is not None:
            output["env_weights"] = env_weights
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
        """Map queries ``[B,Q,spatial_dim]`` to fields ``[B,Q,F]``."""

        return self.decoder(
            query_xy=query_xy,
            query_time=query_time,
            organizer_output=organizer_output,
            global_context=global_token,
            query_features=query_features,
            return_routing_maps=bool(return_routing_maps),
            return_edge_fields=bool(return_edge_fields),
        )

    def forward(
        self,
        batch: BatchData,
        *,
        return_edge_fields: bool = False,
        return_residual_interaction_tensor: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode, organize, and decode a complete :class:`BatchData` batch."""

        encoded = self.encode_and_organize(
            batch,
            return_residual_interaction_tensor=bool(return_residual_interaction_tensor),
        )
        decoder_output = self.decode_queries(
            query_xy=batch.query_xy.float(),
            query_time=None if batch.query_time is None else batch.query_time.float(),
            organizer_output=encoded,
            global_token=encoded["global_token"],
            query_features=None if batch.query_features is None else batch.query_features.float(),
            return_edge_fields=bool(return_edge_fields),
        )
        output: Dict[str, torch.Tensor] = {}
        output.update(
            {
                key: value
                for key, value in encoded.items()
                if not key.startswith("_runtime_")
            }
        )
        output.update(decoder_output)
        return output

    def _environment_coords(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the historical default cell-centered rectangular grid."""

        cfg = self.config
        if cfg.spatial_dim != 2:
            raise ValueError("The default environment grid is defined only for the historical 2-D path.")
        nx = int(cfg.num_env_tokens_x)
        ny = int(cfg.num_env_tokens_y)
        scale_x, scale_y = cfg.spatial_scale()
        xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) / max(float(nx), 1.0) * scale_x
        ys = (torch.arange(ny, device=device, dtype=dtype) + 0.5) / max(float(ny), 1.0) * scale_y
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    @staticmethod
    def _environment_weights(
        values: object,
        *,
        batch_size: int,
        env_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Validate adapter-owned environment masses without inventing defaults.

        ``None`` is kept as ``None`` so the fixed organizer can take its exact
        historical arithmetic branch.  A supplied one-dimensional vector is
        shared across cases; a two-dimensional tensor is case-specific.
        """

        if values is None:
            return None
        if not torch.is_tensor(values):
            raise ValueError("env_weights must be a tensor when supplied.")
        weights = values.to(device=device, dtype=dtype)
        if weights.ndim == 1:
            if int(weights.shape[0]) != env_count:
                raise ValueError("env_weights must align with env_coords as [E] or [B,E].")
            weights = weights.unsqueeze(0).expand(batch_size, -1)
        elif weights.ndim == 2:
            if tuple(weights.shape) != (batch_size, env_count):
                raise ValueError("env_weights must align with env_coords as [E] or [B,E].")
        else:
            raise ValueError("env_weights must have shape [E] or [B,E].")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
            raise ValueError("env_weights must contain finite strictly positive masses.")
        return weights
