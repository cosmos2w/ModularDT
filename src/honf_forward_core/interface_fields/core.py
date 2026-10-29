"""Encode/prepare/read facade for matched non-legacy interface fields."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.nn import FourierFeatures, LazyMLP
from .common import SharedInterfaceContext
from .dense_pairwise import DensePairwiseField
from .group_operator import packed_coarse_group_sources
from .types import EncodedInterfaceCase, InterfaceRead, PreparedInterfaceField


def _merge_group_control_maps(chunks):
    """Merge Run-1406 diagnostics without averaging per-chunk work ratios."""

    merged = {}
    keys = {
        key
        for entry, _ in chunks
        for key in entry
        if key.startswith(("group_control_", "coalescence_"))
    }
    for key in keys:
        values_and_widths = [
            (entry[key], width)
            for entry, width in chunks
            if key in entry
        ]
        if not values_and_widths:
            continue
        values = [value for value, _ in values_and_widths]
        first = values[0]
        if key.startswith("coalescence_"):
            if key in {
                "coalescence_query_mass",
                "coalescence_query_density",
                "coalescence_query_logits",
                "coalescence_query_kq",
                "coalescence_virtual_constituent_support",
            }:
                merged[key] = torch.cat(values, dim=1)
            elif first.ndim == 0 and key.endswith(
                ("_pairs", "_rows", "_calls", "_work", "_forward", "_recompute")
            ):
                merged[key] = torch.stack(values).sum()
            else:
                # Prepared topology, multiplicity, banks, solver diagnostics,
                # and case summaries repeat for every receiver tile.
                merged[key] = first
            continue
        if key in {
            "group_control_adaptive_environment_mass",
            "group_control_adaptive_environment_centroids",
            "group_control_adaptive_environment_radius_sq",
            "group_control_adaptive_environment_group_states",
            "group_control_adaptive_environment_source_mass",
            "group_control_adaptive_environment_keys",
            "group_control_adaptive_source_count_per_group",
        }:
            # These are preparation/group-axis maps repeated for every
            # receiver chunk.  Explicit names avoid mistaking K=12 for a
            # query chunk width in the generic shape heuristics below.
            merged[key] = first
            continue
        if key in {
            "group_control_adaptive_query_count_per_group",
            "group_control_adaptive_fine_group_rows",
            "group_control_adaptive_fine_group_rows_logical",
            "group_control_adaptive_fine_group_rows_forward",
            "group_control_adaptive_fine_group_rows_padded",
            "group_control_adaptive_batched_fine_group_rows_forward",
        }:
            # Query/group execution maps are produced once per receiver tile.
            # Add the tiles rather than retaining the first one; their group
            # axis is not a source/preparation axis even when K equals a tile
            # width.
            merged[key] = torch.stack(values).sum(dim=0)
            continue
        if key.endswith(
            (
                "_incidence",
                "_membership",
                "_measure",
                "_mass",
                "_control",
                "_h",
                "_global",
                "_norm",
            )
        ) and not key.endswith("_per_query") and not key.startswith("group_control_query_"):
            # Source/group controls are repeated for every receiver chunk;
            # identify them by name before shape heuristics so M or E equal to
            # a chunk width cannot turn a source map into a query map.
            merged[key] = first
        elif first.ndim == 1 and key.endswith(
            ("_query", "_query_index", "_receiver", "_pair_query", "_pair_query_index")
        ):
            offset = 0
            shifted = []
            for entry, width in chunks:
                if key in entry:
                    shifted.append(entry[key] + offset)
                offset += width
            merged[key] = torch.cat(shifted, dim=0)
        elif first.ndim == 1 and key.endswith(("_pair_batch", "_pair_batch_index", "_pair_source", "_pair_source_index")):
            # Pair provenance is already global in batch/source coordinates;
            # only the query column above needs the receiver-chunk offset.
            merged[key] = torch.cat(values, dim=0)
        elif first.ndim >= 2 and all(
            value.ndim >= 2 and value.shape[1] == width
            for value, width in values_and_widths
        ):
            # Query-local maps and per-query numerators/denominators retain
            # full receiver indexing rather than combining unequal chunks.
            merged[key] = torch.cat(values, dim=1)
        elif (
            (key.endswith(("_numerator", "_denominator")) or "_pair_count" in key)
            and all(value.shape == first.shape for value in values)
        ):
            # Work/count numerators and denominators are additive across
            # receiver chunks; a ratio is formed only after this merge.  Keep
            # per-query arrays above this branch so equal-width chunks cannot
            # be mistaken for scalar ledgers.
            merged[key] = torch.stack(values).sum(dim=0)
        elif first.ndim == 0 and key.endswith(
            (
                "_sampled_bank_bytes",
                "_sampled_bank_elements",
                "_sampled_bank_key_value_elements",
            )
        ):
            # Sampled-bank storage is a per-tile live tensor footprint.  The
            # last receiver tile may be shorter, so retain the largest actual
            # allocation rather than summing mutually exclusive tile storage.
            merged[key] = torch.stack(values).max()
        elif first.ndim == 0 and key.endswith(
            (
                "_unique_pairs",
                "_logical_paths",
                "_rows",
                "_recomputations",
                "_pairs",
                "_paths",
                "_sample_slots",
                "_nonzero_sample_masses",
                "_fine_rows_forward",
                "_fine_rows_padded",
                "_rows_logical",
                "_rows_padded",
                "_padded_rows",
                "_geometry_rows_forward",
                "_content_dot_rows_forward",
                "_interpolation_corner_loads",
                "_unique_cells_touched",
                "_lower_cells_touched",
                "_fine_rows_recompute",
                "_block_calls",
                "_gemm_launches",
            )
        ):
            # The backend reports these execution counts once per receiver
            # chunk.  Preserve full-read totals instead of silently retaining
            # the first tile.
            merged[key] = torch.stack(values).sum()
        elif first.ndim == 0 and key in {
            "group_control_adaptive_fine_block_call_reduction",
            "group_control_adaptive_fine_gemm_launch_reduction",
            "group_control_adaptive_fine_batched_checkpoint_calls",
            "group_control_adaptive_batched_fine_rows_padded",
            "group_control_adaptive_batched_fine_rows_recompute",
        }:
            # Run-1503's batched executor emits explicit scalar-vs-batched
            # launch reductions and its common-padding tradeoff once per
            # receiver tile.  These are additive full-read ledgers, just like
            # the historical fine-row counters above.
            merged[key] = torch.stack(values).sum()
        else:
            # Source-only memberships, controls, masses, and summaries are
            # repeated for each receiver chunk and are retained once.
            merged[key] = first
    if {
        "group_control_adaptive_fine_rows",
        "group_control_adaptive_full_rectangle_rows",
    }.issubset(merged):
        # Ratios are derived only after all receiver tiles have been summed;
        # the last non-divisible tile must not be underweighted.
        merged["group_control_adaptive_fine_work_ratio"] = (
            merged["group_control_adaptive_fine_rows"]
            / merged["group_control_adaptive_full_rectangle_rows"].clamp_min(1.0)
        )
    return merged


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
        if config.forward_architecture != "dense_pairwise_field":
            raise ValueError(f"Unsupported interface architecture: {config.forward_architecture!r}")
        self.backend = DensePairwiseField(
            hidden, int(options.message_hidden_dim), heads, frequencies,
            activation_checkpointing=bool(options.activation_checkpointing),
        )
        self.receiver_chunk_size = int(options.receiver_chunk_size)

    def set_training_progress(self, *, epoch: int, total_epochs: Optional[int] = None) -> None:
        del epoch, total_epochs

    def selection_state(self) -> Dict[str, Optional[int]]:
        return {"epoch": None, "total_epochs": None}

    def _coordinate_scale(self, coordinates: torch.Tensor) -> torch.Tensor:
        dimension = int(coordinates.shape[-1])
        if dimension != int(self.config.spatial_dim):
            raise ValueError(
                f"coordinates have dimension {dimension}, but config.spatial_dim is {self.config.spatial_dim}."
            )
        if self.config.coordinate_scale is not None:
            values = list(self.config.coordinate_scale)
            if len(values) != dimension:
                raise ValueError(f"coordinate_scale has {len(values)} values for {dimension}-D coordinates.")
            return coordinates.new_tensor(values).reshape(1, 1, dimension)
        if dimension == 2:
            return coordinates.new_tensor(
                [float(self.config.domain_length_x), float(self.config.domain_length_y)]
            ).reshape(1, 1, 2)
        raise ValueError("spatial_dim=3 requires an explicit coordinate_scale.")

    def encode_case(self, batch: BatchData) -> EncodedInterfaceCase:
        module_centers = batch.module_centers.float()
        module_present = batch.module_present.float()
        module_features = batch.module_features.float()
        if module_centers.ndim != 3 or module_present.shape != module_centers.shape[:2]:
            raise ValueError("module_centers and module_present must have shapes [B,M,d] and [B,M].")
        if module_features.ndim != 3 or module_features.shape[:2] != module_centers.shape[:2]:
            raise ValueError("module_features must have shape [B,M,Fm] aligned with module_centers.")
        active = module_present[..., None] > 0.5
        if not torch.isfinite(module_centers).all():
            if bool(torch.isfinite(module_centers).logical_or(~active).all()):
                module_centers = torch.where(active, module_centers, torch.zeros_like(module_centers))
            else:
                raise ValueError("active module_centers must be finite.")
        module_centers = torch.where(active, module_centers, torch.zeros_like(module_centers))
        if not torch.isfinite(module_features).all():
            if bool(torch.isfinite(module_features).logical_or(~active).all()):
                module_features = torch.where(active, module_features, torch.zeros_like(module_features))
            else:
                raise ValueError("active module_features must be finite.")
        module_features = torch.where(active, module_features, torch.zeros_like(module_features))
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
        if env_coords.ndim != 3 or env_coords.shape[0] != module_centers.shape[0]:
            raise ValueError("env_coords must have shape [E,d] or [B,E,d] matching module_centers.")
        if env_coords.shape[-1] != module_centers.shape[-1]:
            raise ValueError("env_coords and module_centers must use the same spatial dimension.")
        if not torch.isfinite(env_coords).all():
            raise ValueError("env_coords must be finite.")
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
            elif env_features.ndim != 3 or env_features.shape[0] != module_centers.shape[0]:
                raise ValueError("env_features must have shape [E,Fe] or [B,E,Fe].")
            if env_features.shape[1] != env_coords.shape[1]:
                raise ValueError("env_features must align with env_coords along the environment axis.")
            env_input = torch.cat([env_input, env_features], dim=-1)
        # Deliberately no legacy env_tokens += global_token broadcast.
        env_tokens = self.env_encoder(env_input)
        if batch.env_weights is None:
            # Preserve the historical dense-backend fallback exactly when the
            # adapter does not provide a measure.
            domain_volume = torch.prod(scale.reshape(-1))
            env_weights = module_centers.new_full(
                (module_centers.shape[0], env_coords.shape[1]),
                1.0 / float(env_coords.shape[1]),
            ) * domain_volume
        else:
            env_weights = batch.env_weights.to(device=module_centers.device, dtype=module_centers.dtype)
            if env_weights.ndim == 1:
                if int(env_weights.shape[0]) != int(env_coords.shape[1]):
                    raise ValueError("env_weights must align with env_coords as [E] or [B,E].")
                env_weights = env_weights.unsqueeze(0).expand(module_centers.shape[0], -1)
            elif env_weights.ndim == 2:
                if tuple(env_weights.shape) != tuple(env_coords.shape[:2]):
                    raise ValueError("env_weights must align with env_coords as [E] or [B,E].")
            else:
                raise ValueError("env_weights must have shape [E] or [B,E].")
            if not bool(torch.isfinite(env_weights).all()) or bool((env_weights <= 0.0).any()):
                raise ValueError("env_weights must contain finite strictly positive masses.")
        env_hierarchy = (
            batch.env_hierarchy.to(module_centers.device)
            if batch.env_hierarchy is not None else None
        )
        hierarchy_geometry = None
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
            routing_geometry=batch.routing_geometry,
        )

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        layout_cache: Any = None,
        *,
        return_routing_maps: bool = False,
        fixed_cover_plans: Any = None,
        interaction_context: Any = None,
    ) -> PreparedInterfaceField:
        if fixed_cover_plans is not None:
            raise ValueError("Dense field does not support cover plans.")
        backend_state = self.backend.prepare(
            encoded, module_states, return_routing_maps=bool(return_routing_maps),
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
            "main_latent_count": 0,
        }
        return PreparedInterfaceField(encoded, module_states, backend_state, coarse_state, aux)

    def build_layout(
        self,
        encoded: EncodedInterfaceCase,
        module_port_coordinates: torch.Tensor,
        *,
        port_quadrature_weights: torch.Tensor | None = None,
    ) -> None:
        """Dense reads do not require a precomputed sparse layout."""
        return None

    def _receiver_features(self, prepared: PreparedInterfaceField, receivers: torch.Tensor) -> torch.Tensor:
        return self.receiver_fourier(receivers / prepared.encoded.coordinate_scale)

    def read(
        self,
        prepared: PreparedInterfaceField,
        receiver_coordinates: torch.Tensor,
        *,
        receiver_chunk_size: Optional[int] = None,
        return_routing_maps: bool = False,
        interaction_context: Any = None,
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
        interaction_context: Any = None,
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
