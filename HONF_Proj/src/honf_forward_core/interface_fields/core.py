"""Encode/prepare/read facade for matched non-legacy interface fields."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.nn import FourierFeatures, LazyMLP
from .common import SharedInterfaceContext
from .dense_pairwise import DensePairwiseField
from .group_operator import SparseInterfaceHONF, SparseLayoutCache
from .latent_attention import GeometryLatentField
from .types import EncodedInterfaceCase, InterfaceRead, PreparedInterfaceField


class InterfaceFieldCore(nn.Module):
    """Small architecture factory with a common continuous-field interface."""

    def __init__(self, config: UnifiedForwardConfig):
        super().__init__()
        if config.interface_model is None:
            raise ValueError("InterfaceFieldCore requires interface_model settings.")
        self.config = config
        options = config.interface_model
        hidden = int(config.hidden_dim)
        heads = int(options.attention_heads)
        frequencies = int(options.relative_fourier_frequencies)
        self.global_encoder = LazyMLP(hidden, num_layers=2, include_zero_dropout=True)
        self.module_feature_encoder = LazyMLP(hidden, num_layers=2, include_zero_dropout=True)
        self.module_position_encoder = LazyMLP(hidden, num_layers=2, include_zero_dropout=True)
        self.env_encoder = LazyMLP(hidden, num_layers=2, include_zero_dropout=True)
        self.position_fourier = FourierFeatures(None, int(config.position_fourier_frequencies))
        self.receiver_fourier = FourierFeatures(None, int(config.query_fourier_frequencies))
        self.common = SharedInterfaceContext(
            hidden_dim=hidden,
            field_dim=int(config.field_dim),
            num_heads=heads,
            coarse_latent_count=int(options.coarse_latent_count),
            coarse_blocks=int(options.coarse_blocks),
            local_radius_factor=float(options.local_radius_factor),
            fourier_frequencies=frequencies,
        )
        if config.forward_architecture == "dense_pairwise_field":
            self.backend = DensePairwiseField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                activation_checkpointing=bool(options.activation_checkpointing),
            )
        elif config.forward_architecture == "geometry_latent_field":
            self.backend = GeometryLatentField(
                hidden,
                int(options.main_latent_count),
                int(options.main_latent_blocks),
                heads,
                frequencies,
            )
        elif config.forward_architecture == "sparse_interface_honf":
            if options.support_spacing_factor is None:
                raise ValueError("sparse_interface_honf requires support_spacing_factor.")
            self.backend = SparseInterfaceHONF(
                hidden,
                int(options.message_hidden_dim),
                frequencies,
                float(options.support_spacing_factor),
            )
        else:
            raise ValueError(f"Unsupported interface architecture: {config.forward_architecture!r}")
        self.receiver_chunk_size = int(options.receiver_chunk_size)

    def set_training_progress(self, *, epoch: int, total_epochs: Optional[int] = None) -> None:
        del epoch, total_epochs

    def selection_state(self) -> Dict[str, Optional[int]]:
        return {"epoch": None, "total_epochs": None}

    def _coordinate_scale(self, coordinates: torch.Tensor) -> torch.Tensor:
        dimension = int(coordinates.shape[-1])
        if self.config.coordinate_scale is not None:
            values = list(self.config.coordinate_scale)
            if len(values) != dimension:
                raise ValueError(f"coordinate_scale has {len(values)} values for {dimension}-D coordinates.")
            return coordinates.new_tensor(values).reshape(1, 1, dimension)
        if dimension == 2:
            return coordinates.new_tensor(
                [float(self.config.domain_length_x), float(self.config.domain_length_y)]
            ).reshape(1, 1, 2)
        return coordinates.new_ones(1, 1, dimension)

    def encode_case(self, batch: BatchData) -> EncodedInterfaceCase:
        module_centers = batch.module_centers.float()
        module_present = batch.module_present.float()
        module_features = batch.module_features.float()
        global_token = self.global_encoder(batch.global_context.float())
        scale = self._coordinate_scale(module_centers)
        module_pos = self.position_fourier(module_centers / scale)
        module_tokens = (
            self.module_feature_encoder(module_features) + self.module_position_encoder(module_pos)
        ) * module_present[..., None]
        if batch.env_coords is None:
            raise ValueError("New interface fields require adapter-supplied environment coordinates.")
        env_coords = batch.env_coords.to(device=module_centers.device, dtype=module_centers.dtype)
        if env_coords.ndim == 2:
            env_coords = env_coords.unsqueeze(0).expand(module_centers.shape[0], -1, -1)
        env_input = self.position_fourier(env_coords / scale)
        env_features = None
        if batch.env_features is not None:
            env_features = batch.env_features.to(device=module_centers.device, dtype=module_centers.dtype)
            if env_features.ndim == 2:
                env_features = env_features.unsqueeze(0).expand(module_centers.shape[0], -1, -1)
            env_input = torch.cat([env_input, env_features], dim=-1)
        # Deliberately no legacy env_tokens += global_token broadcast.
        env_tokens = self.env_encoder(env_input)
        domain_volume = torch.prod(scale.reshape(-1))
        env_weights = module_centers.new_full(
            (module_centers.shape[0], env_coords.shape[1]),
            1.0 / float(env_coords.shape[1]),
        ) * domain_volume
        return EncodedInterfaceCase(
            module_tokens=module_tokens,
            env_tokens=env_tokens,
            global_token=global_token,
            module_centers=module_centers,
            env_coords=env_coords,
            module_present=module_present,
            module_features=module_features,
            env_features=env_features,
            env_weights=env_weights,
            coordinate_scale=scale,
        )

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        layout_cache: Any = None,
    ) -> PreparedInterfaceField:
        if self.config.forward_architecture == "sparse_interface_honf":
            if not isinstance(layout_cache, SparseLayoutCache):
                raise ValueError("sparse_interface_honf requires a SparseLayoutCache built from module ports.")
            backend_state = self.backend.prepare(encoded, module_states, layout_cache)
        else:
            backend_state = self.backend.prepare(encoded, module_states)
        coarse_state = self.common.prepare_coarse(
            module_states, encoded.env_tokens, encoded.module_present, encoded.env_weights
        )
        aux: Dict[str, Any] = {
            "forward_architecture": self.config.forward_architecture,
            "coarse_latent_count": int(self.config.interface_model.coarse_latent_count),
            "main_latent_count": (
                int(self.config.interface_model.main_latent_count)
                if self.config.forward_architecture == "geometry_latent_field"
                else 0
            ),
        }
        if self.config.forward_architecture == "sparse_interface_honf":
            aux.update(self.backend.preparation_aux(backend_state))
        return PreparedInterfaceField(encoded, module_states, backend_state, coarse_state, aux)

    def build_layout(
        self,
        encoded: EncodedInterfaceCase,
        module_port_coordinates: torch.Tensor,
        *,
        port_quadrature_weights: torch.Tensor | None = None,
    ) -> SparseLayoutCache | None:
        """Build sparse case geometry once; dense/latent families need no cache."""

        if self.config.forward_architecture != "sparse_interface_honf":
            return None
        return self.backend.build_layout(
            encoded,
            module_port_coordinates,
            module_radius=float(self.config.module_radius),
            port_quadrature_weights=port_quadrature_weights,
        )

    def _receiver_features(self, prepared: PreparedInterfaceField, receivers: torch.Tensor) -> torch.Tensor:
        return self.receiver_fourier(receivers / prepared.encoded.coordinate_scale)

    def read(
        self,
        prepared: PreparedInterfaceField,
        receiver_coordinates: torch.Tensor,
    ) -> InterfaceRead:
        receivers = receiver_coordinates.float()
        contexts = []
        neighbour_counts = []
        main_norms = []
        coarse_norms = []
        local_norms = []
        backend_aux_chunks: list[Dict[str, torch.Tensor]] = []
        for start in range(0, int(receivers.shape[1]), self.receiver_chunk_size):
            chunk = receivers[:, start : start + self.receiver_chunk_size]
            receiver_features = self._receiver_features(prepared, chunk)
            main, backend_aux = self.backend.read(
                prepared.backend_state, prepared.encoded, chunk, receiver_features
            )
            coarse = self.common.read_coarse(
                receiver_features, prepared.encoded.global_token, prepared.coarse_state
            )
            local, counts = self.common.read_local(
                chunk,
                prepared.module_states,
                prepared.encoded.module_centers,
                prepared.encoded.module_features,
                prepared.encoded.module_present,
                prepared.encoded.coordinate_scale,
                float(self.config.module_radius),
            )
            contexts.append(main + coarse + local)
            neighbour_counts.append(counts)
            main_norms.append(torch.linalg.vector_norm(main, dim=-1))
            coarse_norms.append(torch.linalg.vector_norm(coarse, dim=-1))
            local_norms.append(torch.linalg.vector_norm(local, dim=-1))
            backend_aux_chunks.append(backend_aux)
        main_values = torch.cat(main_norms, dim=1)
        coarse_values = torch.cat(coarse_norms, dim=1)
        local_values = torch.cat(local_norms, dim=1)
        branch_total = (main_values + coarse_values + local_values).clamp_min(1.0e-12)
        aux: Dict[str, torch.Tensor] = {
            "local_neighbor_count": torch.cat(neighbour_counts, dim=1),
            "main_context_norm": main_values,
            "coarse_context_norm": coarse_values,
            "local_context_norm": local_values,
            "main_context_fraction": main_values / branch_total,
            "coarse_context_fraction": coarse_values / branch_total,
            "local_context_fraction": local_values / branch_total,
        }
        if backend_aux_chunks:
            for key in backend_aux_chunks[0]:
                values = [chunk[key] for chunk in backend_aux_chunks if key in chunk]
                if values and all(torch.is_tensor(value) and value.ndim >= 2 for value in values):
                    concat_dim = 2 if values[0].ndim == 4 else 1
                    aux[key] = torch.cat(values, dim=concat_dim)
        return InterfaceRead(torch.cat(contexts, dim=1), aux)

    def decode_queries(
        self,
        prepared: PreparedInterfaceField,
        query_xy: torch.Tensor,
        query_features: Optional[torch.Tensor] = None,
        *,
        return_routing_maps: bool = False,
        return_edge_fields: bool = False,
        return_interaction_aux: bool = False,
    ) -> Dict[str, Any]:
        if return_edge_fields:
            raise ValueError("Per-edge fields are not defined for interface-field baselines.")
        read = self.read(prepared, query_xy)
        receiver_features = self._receiver_features(prepared, query_xy.float())
        pred_field = self.common.predict_field(
            query_xy.float(),
            receiver_features,
            read.context,
            prepared.encoded.global_token,
            query_features,
        )
        result: Dict[str, torch.Tensor] = {"pred_field": pred_field}
        if return_routing_maps:
            result.update(read.interaction_aux)
        if return_interaction_aux:
            result["_interaction_aux"] = read.interaction_aux
        return result
