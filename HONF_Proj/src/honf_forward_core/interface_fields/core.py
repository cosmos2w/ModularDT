"""Encode/prepare/read facade for matched non-legacy interface fields."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.nn import FourierFeatures, LazyMLP
from .common import SharedInterfaceContext
from .dense_pairwise import DensePairwiseField
from .group_operator import SparseInterfaceHONF, SparseLayoutCache, packed_coarse_group_sources
from .hierarchical_regional import HierarchicalRegionalField
from .latent_attention import GeometryLatentField
from .regional_response import RegionalResponseField
from .response_hierarchy import prepare_hierarchy_geometry
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
            coarse_module_source=str(options.coarse_module_source),
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
                group_read_mode=str(options.group_read_mode),
            )
        elif config.forward_architecture == "regional_response_honf":
            self.backend = RegionalResponseField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                response_region_block_shape=tuple(options.response_region_block_shape),
                activation_checkpointing=bool(options.activation_checkpointing),
            )
        elif config.forward_architecture == "hierarchical_regional_honf":
            self.backend = HierarchicalRegionalField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                response_region_block_shape=tuple(options.response_region_block_shape),
                response_tree_opening_interval=tuple(options.response_tree_opening_interval),
                activation_checkpointing=bool(options.activation_checkpointing),
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
        env_region_ids = None
        if batch.env_region_ids is not None:
            env_region_ids = batch.env_region_ids.to(device=module_centers.device, dtype=torch.long)
            if env_region_ids.ndim == 1:
                env_region_ids = env_region_ids.unsqueeze(0).expand(module_centers.shape[0], -1)
            elif env_region_ids.ndim == 2 and env_region_ids.shape[0] == 1 and module_centers.shape[0] != 1:
                env_region_ids = env_region_ids.expand(module_centers.shape[0], -1)
            if env_region_ids.ndim != 2 or tuple(env_region_ids.shape) != tuple(env_coords.shape[:2]):
                raise ValueError("env_region_ids must align with env_coords as [B,E].")
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
        env_hierarchy = (
            batch.env_hierarchy.to(module_centers.device)
            if batch.env_hierarchy is not None else None
        )
        hierarchy_geometry = None
        if self.config.forward_architecture == "hierarchical_regional_honf":
            if env_hierarchy is None:
                raise ValueError("hierarchical_regional_honf requires adapter-supplied env_hierarchy.")
            hierarchy_geometry = prepare_hierarchy_geometry(env_hierarchy, env_weights, env_coords)
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
            env_region_ids=env_region_ids,
            env_hierarchy=env_hierarchy,
            env_hierarchy_geometry=hierarchy_geometry,
        )

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        layout_cache: Any = None,
        *,
        return_routing_maps: bool = False,
    ) -> PreparedInterfaceField:
        if self.config.forward_architecture == "sparse_interface_honf":
            if not isinstance(layout_cache, SparseLayoutCache):
                raise ValueError("sparse_interface_honf requires a SparseLayoutCache built from module ports.")
            backend_state = self.backend.prepare(encoded, module_states, layout_cache)
        elif self.config.forward_architecture == "regional_response_honf":
            backend_state = self.backend.prepare(
                encoded,
                module_states,
                region_ids=encoded.env_region_ids,
                return_routing_maps=bool(return_routing_maps),
            )
        else:
            backend_state = self.backend.prepare(
                encoded,
                module_states,
                return_routing_maps=bool(return_routing_maps),
            )
        coarse_kwargs = {}
        if self.config.interface_model.coarse_module_source == "group_states":
            groups = packed_coarse_group_sources(backend_state)
            coarse_kwargs = {
                "packed_group_states": groups.group_states,
                "packed_group_occupancy": groups.occupancy,
                "packed_group_valid": groups.valid,
            }
        coarse_state = self.common.prepare_coarse(
            module_states, encoded.env_tokens, encoded.module_present, encoded.env_weights,
            **coarse_kwargs,
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
        if self.config.forward_architecture in {"sparse_interface_honf", "hierarchical_regional_honf"}:
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
        *,
        receiver_chunk_size: Optional[int] = None,
        return_routing_maps: bool = False,
    ) -> InterfaceRead:
        """Read receivers with an optional evaluation-only chunk override.

        The configured chunk size remains the training/default execution
        policy.  ``receiver_chunk_size`` affects only this call and does not
        enter the model configuration or checkpoint state.
        """
        receivers = receiver_coordinates.float()
        chunk_size = self.receiver_chunk_size if receiver_chunk_size is None else int(receiver_chunk_size)
        if chunk_size <= 0:
            raise ValueError("receiver_chunk_size must be positive.")
        contexts = []
        neighbour_counts = []
        main_norms = []
        coarse_norms = []
        local_norms = []
        backend_aux_chunks: list[tuple[Dict[str, torch.Tensor], int]] = []
        for start in range(0, int(receivers.shape[1]), chunk_size):
            chunk = receivers[:, start : start + chunk_size]
            receiver_features = self._receiver_features(prepared, chunk)
            main, backend_aux = self.backend.read(
                prepared.backend_state,
                prepared.encoded,
                chunk,
                receiver_features,
                return_routing_maps=bool(return_routing_maps),
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
            backend_aux_chunks.append((backend_aux, int(chunk.shape[1])))
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
            keys = {key for chunk_aux, _ in backend_aux_chunks for key in chunk_aux}
            for key in keys:
                values_and_widths = [
                    (chunk_aux[key], width)
                    for chunk_aux, width in backend_aux_chunks
                    if key in chunk_aux
                ]
                values = [value for value, _ in values_and_widths]
                if not values or not all(torch.is_tensor(value) for value in values):
                    continue
                first = values[0]
                if key in {"hierarchical_traversal_rows", "hierarchical_incidence_rows"}:
                    aux[key] = torch.stack(values).sum()
                    continue
                if key.startswith("hierarchical_incidence_"):
                    # The optional tree maps are ragged incidence rows, not
                    # dense receiver/source arrays. Preserve every row and
                    # translate chunk-local query indices to this read call.
                    if key == "hierarchical_incidence_query":
                        offset = 0
                        shifted = []
                        for chunk_aux, width in backend_aux_chunks:
                            if key in chunk_aux:
                                shifted.append(chunk_aux[key] + offset)
                            offset += width
                        aux[key] = torch.cat(shifted, dim=0)
                    else:
                        aux[key] = torch.cat(values, dim=0)
                    continue
                # Backend summaries such as latent_count are batch-level and
                # repeated for each receiver chunk.  Keep one copy rather
                # than accidentally concatenating it across chunks.
                if first.ndim == 0 or (first.ndim == 1 and first.shape[0] == receivers.shape[0]):
                    aux[key] = first
                    continue
                # Query-local tensors use [B,Q,...] while attention maps use
                # [B,H,Q,S].  The chunk widths are retained explicitly so a
                # query count equal to the number of heads cannot confuse the
                # axis selection.
                if first.ndim >= 4 and all(
                    value.shape[2] == width for value, width in values_and_widths
                ):
                    aux[key] = torch.cat(values, dim=2)
                    continue
                if first.ndim >= 2 and all(
                    value.shape[1] == width for value, width in values_and_widths
                ):
                    aux[key] = torch.cat(values, dim=1)
                    continue
                aux[key] = first
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
        receiver_chunk_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        if return_edge_fields:
            raise ValueError("Per-edge fields are not defined for interface-field baselines.")
        read = self.read(
            prepared,
            query_xy,
            receiver_chunk_size=receiver_chunk_size,
            return_routing_maps=bool(return_routing_maps),
        )
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
