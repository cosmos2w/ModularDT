"""Encode/prepare/read facade for matched non-legacy interface fields."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from honf_forward_core.config import (
    ROUTING_TYPED_TEMPERATURE_NAMES,
    BatchData,
    UnifiedForwardConfig,
)
from honf_forward_core.nn import FourierFeatures, LazyMLP

from .common import SharedInterfaceContext
from .dense_pairwise import DensePairwiseField
from .group_operator import SparseInterfaceHONF, SparseLayoutCache, packed_coarse_group_sources
from .hierarchical_regional import HierarchicalRegionalField
from .latent_attention import GeometryLatentField
from .regional_response import RegionalResponseField
from .response_hierarchy import prepare_hierarchy_geometry
from .types import EncodedInterfaceCase, InterfaceRead, PreparedInterfaceField


def _merge_compiled_routing_maps(chunks):
    """Merge opt-in CSR diagnostics in global batch/receiver order."""
    merged = {}
    total_width = sum(width for _, width in chunks)
    for prefix in ("routing_module", "routing_environment"):
        if not all(prefix + "_partial_row_ptr" in entry for entry, _ in chunks):
            continue
        complete_indices, complete_priors = [], []
        partial_rows, partial_sources, partial_priors, count_chunks = [], [], [], []
        offset = 0
        for entry, width in chunks:
            local_complete = entry[prefix + "_complete_row_index"]
            complete_indices.append(torch.div(local_complete, width, rounding_mode="floor") * total_width
                                    + local_complete.remainder(width) + offset)
            complete_priors.append(entry[prefix + "_complete_prior"])
            ptr = entry[prefix + "_partial_row_ptr"]
            counts = ptr[1:] - ptr[:-1]
            count_chunks.append(counts.reshape(-1, width))
            local_rows = torch.repeat_interleave(torch.arange(counts.numel(), device=ptr.device), counts)
            partial_rows.append(torch.div(local_rows, width, rounding_mode="floor") * total_width
                                + local_rows.remainder(width) + offset)
            partial_sources.append(entry[prefix + "_partial_source"])
            partial_priors.append(entry[prefix + "_partial_prior"])
            offset += width
        complete = torch.cat(complete_indices)
        order = torch.argsort(complete, stable=True)
        merged[prefix + "_complete_row_index"] = complete[order]
        merged[prefix + "_complete_prior"] = torch.cat(complete_priors)[order]
        # This optional diagnostic ordering sorts unique rows, never hub paths.
        order = torch.argsort(torch.cat(partial_rows), stable=True)
        merged[prefix + "_partial_source"] = torch.cat(partial_sources)[order]
        merged[prefix + "_partial_prior"] = torch.cat(partial_priors)[order]
        counts = torch.cat(count_chunks, dim=1).reshape(-1)
        merged[prefix + "_partial_row_ptr"] = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return merged


def _merge_fixed_group_maps(chunks):
    """Merge Run-1405 query maps and ragged semantic triples by receiver order.

    Fixed-group diagnostics are deliberately explicit here because they do not
    all follow the historical dense-routing tensor layout.  In particular,
    dense environment attention is ``[B,K,H,Q,E]`` (the receiver axis is 3),
    while semantic triples are packed one-dimensional arrays whose query
    indices are local to each receiver chunk.
    """

    merged = {}
    keys = {
        key
        for entry, _ in chunks
        for key in entry
        if key.startswith("fixed_group_")
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
        if key.endswith("_triple_query"):
            offset = 0
            shifted = []
            for entry, width in chunks:
                if key in entry:
                    shifted.append(entry[key] + offset)
                offset += width
            merged[key] = torch.cat(shifted, dim=0)
        elif key.endswith(("_triple_batch", "_triple_group", "_triple_source")):
            merged[key] = torch.cat(values, dim=0)
        elif key.startswith("fixed_group_environment_attention_"):
            # Packed attention can optionally carry explicit provenance
            # columns alongside [triples, heads].  Query indices are local to
            # each receiver chunk; batch/group/source indices are not.
            suffix = key.removeprefix("fixed_group_environment_attention_")
            if suffix in {"query", "query_index"}:
                offset = 0
                shifted = []
                for entry, width in chunks:
                    if key in entry:
                        shifted.append(entry[key] + offset)
                    offset += width
                merged[key] = torch.cat(shifted, dim=0)
            elif suffix in {"batch", "batch_index", "group", "group_index", "source", "source_index"}:
                merged[key] = torch.cat(values, dim=0)
            else:
                merged[key] = first
        elif key == "fixed_group_environment_attention":
            if first.ndim >= 5 and all(
                value.ndim >= 5 and value.shape[3] == width
                for value, width in values_and_widths
            ):
                merged[key] = torch.cat(values, dim=3)
            elif first.ndim >= 2:
                # The backend may use a packed [triples, heads] diagnostic;
                # preserve every packed row across receiver chunks too.
                merged[key] = torch.cat(values, dim=0)
            else:  # pragma: no cover - diagnostics are tensor-valued.
                merged[key] = first
        elif first.ndim >= 2 and all(
            value.ndim >= 2 and value.shape[1] == width
            for value, width in values_and_widths
        ):
            # Query-local fixed-group maps: [B,Q,...].
            merged[key] = torch.cat(values, dim=1)
        else:
            # Incidence, centres, and batch-level summaries are repeated for
            # every receiver chunk; retain one copy.
            merged[key] = first
    return merged


def _merge_group_control_maps(chunks):
    """Merge Run-1406 diagnostics without averaging per-chunk work ratios."""

    merged = {}
    keys = {
        key
        for entry, _ in chunks
        for key in entry
        if key.startswith("group_control_")
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
        if config.forward_architecture in {
            "fixed_group_pairwise_honf",
            "group_control_pairwise_honf",
            "phase_shared_group_control_honf",
            "hypergraph_quadrature_honf",
            "budgeted_group_control_honf",
            "occupancy_adaptive_group_control_honf",
            "mass_competitive_group_control_honf",
            "sparse_incidence_group_control_honf",
            "adaptive_hyperedge_opening_honf",
        }:
            # Run 1405/1406 deliberately replace the historical coarse/local
            # context object with the three-term reader. Keep construction
            # conditional so every historical architecture retains the same
            # module tree and serialized parameter names.
            from .three_term_context import ThreeTermInterfaceContext

            self.common = ThreeTermInterfaceContext(
                hidden_dim=hidden,
                field_dim=int(config.field_dim),
                query_fourier_frequencies=int(config.query_fourier_frequencies),
            )
        else:
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
        # The science profile owns exactly four scalar route temperatures at
        # the reusable core boundary.  Historical profiles keep this
        # ParameterDict empty, so their state-dict structure and optimizer
        # inventory remain unchanged.  The routed backend receives the same
        # ParameterDict by reference and applies the values to source/query
        # projections without creating a second set of parameters.
        self.routing_log_temperatures = nn.ParameterDict()
        if (
            config.forward_architecture == "routed_pairwise_honf"
            and options.routing is not None
            and bool(options.routing.sparsification.learn_typed_temperatures)
        ):
            # ParameterDict.update sorts ordinary mappings in some supported
            # PyTorch versions.  Assign one key at a time so checkpoint and
            # optimizer inventories retain the plan's source-M, source-E,
            # query-M, query-E order as well as the exact four-key schema.
            for name in ROUTING_TYPED_TEMPERATURE_NAMES:
                self.routing_log_temperatures[name] = nn.Parameter(
                    torch.zeros((), dtype=torch.get_default_dtype())
                )
        if config.forward_architecture == "dense_pairwise_field":
            self.backend = DensePairwiseField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                activation_checkpointing=bool(options.activation_checkpointing),
            )
        elif config.forward_architecture == "routed_pairwise_honf":
            from .routed_pairwise import RoutedPairwiseField

            routed_kwargs = {
                "routing_config": options.routing,
                "activation_checkpointing": bool(options.activation_checkpointing),
            }
            # Keep the historical backend constructor path untouched until a
            # science profile actually asks for the new parameters.  The
            # exact executor accepts this optional object for the opt-in
            # profile; old checkpoints therefore remain loadable during the
            # transition and under older backend implementations.
            if len(self.routing_log_temperatures) > 0:
                # Pass a plain mapping of the already-registered Parameter
                # objects.  Registering the ParameterDict a second time under
                # the backend would duplicate state-dict paths and defeat the
                # strict four-key warm-start contract.
                routed_kwargs["typed_log_temperatures"] = dict(self.routing_log_temperatures)
            self.backend = RoutedPairwiseField(
                hidden, int(options.message_hidden_dim), heads, frequencies,
                **routed_kwargs,
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
        elif config.forward_architecture == "fixed_group_pairwise_honf":
            from .fixed_group_pairwise import FixedGroupPairwiseField

            self.backend = FixedGroupPairwiseField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                group_count=int(options.group_count),
                group_code_dim=int(options.group_code_dim),
                spatial_dim=int(config.spatial_dim),
                module_temperature=float(options.module_temperature),
                environment_temperature=float(options.environment_temperature),
                query_temperature=float(options.query_temperature),
                activation_checkpointing=bool(options.activation_checkpointing),
            )
        elif config.forward_architecture == "group_control_pairwise_honf":
            from .group_control_pairwise import GroupControlPairwiseField

            self.backend = GroupControlPairwiseField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                group_count=int(options.group_count),
                group_control_dim=int(options.group_control_dim),
                spatial_dim=int(config.spatial_dim),
                module_temperature=float(options.module_temperature),
                environment_temperature=float(options.environment_temperature),
                query_temperature=float(options.query_temperature),
                activation_checkpointing=bool(options.activation_checkpointing),
            )
        elif config.forward_architecture == "phase_shared_group_control_honf":
            from .phase_shared_group_control import PhaseSharedGroupControlPairwiseField

            self.backend = PhaseSharedGroupControlPairwiseField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                group_count=int(options.group_count),
                group_control_dim=int(options.group_control_dim),
                spatial_dim=int(config.spatial_dim),
                module_temperature=float(options.module_temperature),
                environment_temperature=float(options.environment_temperature),
                query_temperature=float(options.query_temperature),
                activation_checkpointing=bool(options.activation_checkpointing),
            )
        elif config.forward_architecture == "hypergraph_quadrature_honf":
            # Keep the opt-in import narrow so historical models do not
            # materialize or depend on the sampled-reader implementation.
            from .hypergraph_quadrature import HypergraphQuadratureField

            self.backend = HypergraphQuadratureField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                group_count=int(options.group_count),
                group_control_dim=int(options.group_control_dim),
                samples_per_group=int(options.samples_per_group),
                spatial_dim=int(config.spatial_dim),
                module_temperature=float(options.module_temperature),
                environment_temperature=float(options.environment_temperature),
                query_temperature=float(options.query_temperature),
                activation_checkpointing=bool(options.activation_checkpointing),
            )
        elif config.forward_architecture == "budgeted_group_control_honf":
            from .budgeted_group_control import BudgetedGroupControlPairwiseField

            budget = options.case_group_budget
            if budget is None or not budget.enabled:
                raise ValueError(
                    "budgeted_group_control_honf requires an enabled case_group_budget block."
                )
            self.backend = BudgetedGroupControlPairwiseField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                group_count=int(options.group_count),
                group_control_dim=int(options.group_control_dim),
                spatial_dim=int(config.spatial_dim),
                module_temperature=float(options.module_temperature),
                environment_temperature=float(options.environment_temperature),
                query_temperature=float(options.query_temperature),
                activation_checkpointing=bool(options.activation_checkpointing),
                gate_hidden_dim=int(budget.gate_hidden_dim),
                hard_concrete_temperature=float(budget.hard_concrete_temperature),
                stretch_lower=float(budget.stretch_lower),
                stretch_upper=float(budget.stretch_upper),
                initial_optional_open_probability=float(budget.initial_optional_open_probability),
                always_available_group=int(budget.always_available_group),
                execution_mode=str(budget.execution_mode),
                rescue_mode=bool(budget.rescue_mode),
                schedule=str(budget.schedule),
                routing_initial_scale=float(budget.routing_initial_scale),
                routing_full_epoch=int(budget.routing_full_epoch),
                compression_start_epoch=int(budget.compression_start_epoch),
                hardening_epoch=int(budget.hardening_epoch),
            )
        elif config.forward_architecture == "occupancy_adaptive_group_control_honf":
            from .occupancy_group_control import OccupancyAdaptiveGroupControlPairwiseField

            self.backend = OccupancyAdaptiveGroupControlPairwiseField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                group_count=int(options.group_count),
                group_control_dim=int(options.group_control_dim),
                spatial_dim=int(config.spatial_dim),
                module_temperature=float(options.module_temperature),
                environment_temperature=float(options.environment_temperature),
                query_temperature=float(options.query_temperature),
                activation_checkpointing=bool(options.activation_checkpointing),
            )
        elif config.forward_architecture == "mass_competitive_group_control_honf":
            from .mass_competitive_group_control import (
                MassCompetitiveGroupControlPairwiseField,
            )

            self.backend = MassCompetitiveGroupControlPairwiseField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                group_count=int(options.group_count),
                group_control_dim=int(options.group_control_dim),
                spatial_dim=int(config.spatial_dim),
                module_temperature=float(options.module_temperature),
                environment_temperature=float(options.environment_temperature),
                query_temperature=float(options.query_temperature),
                activation_checkpointing=bool(options.activation_checkpointing),
            )
        elif config.forward_architecture == "sparse_incidence_group_control_honf":
            from .sparse_incidence_group_control import (
                SparseIncidenceGroupControlPairwiseField,
            )

            self.backend = SparseIncidenceGroupControlPairwiseField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                group_count=int(options.group_count),
                group_control_dim=int(options.group_control_dim),
                spatial_dim=int(config.spatial_dim),
                module_temperature=float(options.module_temperature),
                environment_temperature=float(options.environment_temperature),
                query_temperature=float(options.query_temperature),
                activation_checkpointing=bool(options.activation_checkpointing),
                environment_refinement_normalizer=str(
                    options.environment_refinement_normalizer
                ),
            )
        elif config.forward_architecture == "adaptive_hyperedge_opening_honf":
            from .adaptive_hyperedge_opening import AdaptiveHyperedgeOpeningPairwiseField

            self.backend = AdaptiveHyperedgeOpeningPairwiseField(
                hidden,
                int(options.message_hidden_dim),
                heads,
                frequencies,
                group_count=int(options.group_count),
                group_control_dim=int(options.group_control_dim),
                spatial_dim=int(config.spatial_dim),
                module_temperature=float(options.module_temperature),
                environment_temperature=float(options.environment_temperature),
                query_temperature=float(options.query_temperature),
                activation_checkpointing=bool(options.activation_checkpointing),
                environment_refinement_normalizer=str(
                    options.environment_refinement_normalizer
                ),
            )
        else:
            raise ValueError(f"Unsupported interface architecture: {config.forward_architecture!r}")
        self.receiver_chunk_size = int(options.receiver_chunk_size)

    @property
    def typed_routing_temperatures_enabled(self) -> bool:
        """Whether this core carries the four learnable route temperatures."""

        return len(self.routing_log_temperatures) == len(ROUTING_TYPED_TEMPERATURE_NAMES)

    def set_training_progress(self, *, epoch: int, total_epochs: int | None = None) -> None:
        if self.config.forward_architecture == "budgeted_group_control_honf":
            setter = getattr(self.backend, "set_training_progress", None)
            if callable(setter):
                setter(epoch=epoch, total_epochs=total_epochs)

    def selection_state(self) -> dict[str, int | None]:
        if self.config.forward_architecture == "budgeted_group_control_honf":
            getter = getattr(self.backend, "selection_state", None)
            if callable(getter):
                return dict(getter())
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
            routing_geometry=batch.routing_geometry,
            sampler_layout=batch.sampler_layout,
        )

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        layout_cache: Any = None,
        *,
        return_routing_maps: bool = False,
        phase_shared_state: Any = None,
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
        elif self.config.forward_architecture in {
            "phase_shared_group_control_honf",
            "hypergraph_quadrature_honf",
            "budgeted_group_control_honf",
            "occupancy_adaptive_group_control_honf",
            "mass_competitive_group_control_honf",
        }:
            backend_state = self.backend.prepare(
                encoded,
                module_states,
                phase_shared_state=phase_shared_state,
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
        aux: dict[str, Any] = {
            "forward_architecture": self.config.forward_architecture,
            "coarse_latent_count": (
                0
                if self.config.forward_architecture in {
                    "fixed_group_pairwise_honf",
                    "group_control_pairwise_honf",
                    "phase_shared_group_control_honf",
                    "hypergraph_quadrature_honf",
                    "budgeted_group_control_honf",
                    "occupancy_adaptive_group_control_honf",
                    "mass_competitive_group_control_honf",
                    "sparse_incidence_group_control_honf",
                    "adaptive_hyperedge_opening_honf",
                }
                else int(self.config.interface_model.coarse_latent_count)
            ),
            "main_latent_count": (
                int(self.config.interface_model.main_latent_count)
                if self.config.forward_architecture == "geometry_latent_field"
                else 0
            ),
        }
        if self.config.forward_architecture in {
            "fixed_group_pairwise_honf",
            "group_control_pairwise_honf",
            "phase_shared_group_control_honf",
            "hypergraph_quadrature_honf",
            "budgeted_group_control_honf",
            "occupancy_adaptive_group_control_honf",
            "mass_competitive_group_control_honf",
            "sparse_incidence_group_control_honf",
            "adaptive_hyperedge_opening_honf",
        }:
            aux.update(
                self.backend.preparation_aux(
                    backend_state,
                    include_diagnostics=bool(return_routing_maps),
                )
            )
        elif self.config.forward_architecture in {
            "sparse_interface_honf",
            "hierarchical_regional_honf",
            "routed_pairwise_honf",
        }:
            aux.update(self.backend.preparation_aux(backend_state))
        return PreparedInterfaceField(
            encoded,
            module_states,
            backend_state,
            coarse_state,
            aux,
            (
                backend_state.get("phase_shared_group_control")
                if isinstance(backend_state, dict)
                else None
            )
            if self.config.forward_architecture != "budgeted_group_control_honf"
            else backend_state.get("case_group_budget"),
        )

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
        scale = prepared.encoded.coordinate_scale
        if scale.ndim == 1:
            scale = scale[None, None, :]
        elif scale.ndim == 2:
            scale = scale[:, None, :]
        elif scale.ndim == 3:
            if int(scale.shape[1]) != 1:
                raise ValueError("coordinate_scale must have shape [d], [B,d], or [B,1,d].")
        else:
            raise ValueError("coordinate_scale must have shape [d], [B,d], or [B,1,d].")
        return self.receiver_fourier(receivers / scale)

    def read(
        self,
        prepared: PreparedInterfaceField,
        receiver_coordinates: torch.Tensor,
        *,
        receiver_chunk_size: int | None = None,
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
        backend_aux_chunks: list[tuple[dict[str, torch.Tensor], int]] = []
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
        aux: dict[str, torch.Tensor] = {
            "local_neighbor_count": torch.cat(neighbour_counts, dim=1),
            "main_context_norm": main_values,
            "coarse_context_norm": coarse_values,
            "local_context_norm": local_values,
            "main_context_fraction": main_values / branch_total,
            "coarse_context_fraction": coarse_values / branch_total,
            "local_context_fraction": local_values / branch_total,
        }
        if backend_aux_chunks:
            compiled_maps = _merge_compiled_routing_maps(backend_aux_chunks)
            aux.update(compiled_maps)
            fixed_group_maps = _merge_fixed_group_maps(backend_aux_chunks)
            aux.update(fixed_group_maps)
            group_control_maps = _merge_group_control_maps(backend_aux_chunks)
            aux.update(group_control_maps)
            keys = (
                {key for chunk_aux, _ in backend_aux_chunks for key in chunk_aux}
                - set(compiled_maps)
                - set(fixed_group_maps)
                - set(group_control_maps)
            )
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
                if key.startswith("routing_") and key.endswith(("_raw_path_count", "_unique_pair_count")):
                    aux[key] = torch.stack(values).sum()
                    continue
                # Pair-cost components are live numerator/denominator
                # scalars.  Receiver chunking must combine them across the
                # full read before the case loss forms their ratio; retaining
                # only the first chunk would bias the objective toward the
                # first receiver tile.
                if key.endswith(("routing_paircost_numerator", "routing_paircost_denominator")):
                    if first.ndim == 2 and all(value.shape[1] == width for value, width in values_and_widths):
                        aux[key] = torch.cat(values, dim=1)
                    else:
                        aux[key] = torch.stack(values).sum()
                    continue
                if key.startswith(("routing_module_pair_", "routing_environment_pair_")):
                    if key.endswith("_receiver"):
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
                if key.startswith("routing_") and key.endswith("_duplicate_expansion"):
                    prefix = key.removesuffix("_duplicate_expansion")
                    raw = torch.stack([entry[prefix + "_raw_path_count"] for entry, _ in backend_aux_chunks]).sum()
                    unique = torch.stack([entry[prefix + "_unique_pair_count"] for entry, _ in backend_aux_chunks]).sum()
                    aux[key] = raw / unique.clamp_min(1)
                    continue
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
                if key.startswith("routing_") and first.ndim >= 2 and all(
                    value.shape[1] == width for value, width in values_and_widths
                ):
                    aux[key] = torch.cat(values, dim=1)
                    continue
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
        query_features: torch.Tensor | None = None,
        *,
        return_routing_maps: bool = False,
        return_edge_fields: bool = False,
        return_interaction_aux: bool = False,
        receiver_chunk_size: int | None = None,
    ) -> dict[str, Any]:
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
        result: dict[str, torch.Tensor] = {"pred_field": pred_field}
        if return_routing_maps:
            result.update(read.interaction_aux)
        if return_interaction_aux:
            result["_interaction_aux"] = read.interaction_aux
        return result
