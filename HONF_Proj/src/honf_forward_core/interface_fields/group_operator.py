"""Sparse nonlinear multi-entity group operator for interface-centred HONF."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from honf_forward_core.nn import MLP, FourierFeatures, LazyMLP

from .supports import SparseSupportLayout, build_sparse_support_layout, lookup_receivers
from .types import EncodedInterfaceCase

GROUP_READ_MODES = {
    "null_softmax",
    "geometry_envelope_attention",
}


@dataclass
class SparseEnvironmentCache:
    """Environment-only learned work reused across physical coupling passes."""

    pool: torch.Tensor
    learned_membership: torch.Tensor


@dataclass
class SparseLayoutCache:
    """One case-batch layout and its optional differentiable environment cache."""

    layout: SparseSupportLayout
    environment: SparseEnvironmentCache | None = None


@dataclass(frozen=True)
class SparseGroupState:
    """Refreshed group state paired with immutable sparse layout work."""

    cache: SparseLayoutCache
    group_state: torch.Tensor
    group_keys: torch.Tensor
    group_values: torch.Tensor
    module_learned_membership: torch.Tensor
    module_pool: torch.Tensor


def _zero_last_bias(module: LazyMLP) -> None:
    last = module.net[-1]
    if isinstance(last, nn.Linear) and last.bias is not None:
        nn.init.zeros_(last.bias)


class SparseInterfaceHONF(nn.Module):
    """Shared sparse group computation used for both ports and field queries."""

    def __init__(
        self,
        hidden_dim: int,
        message_hidden_dim: int,
        fourier_frequencies: int,
        support_spacing_factor: float,
        group_read_mode: str = "null_softmax",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.support_spacing_factor = float(support_spacing_factor)
        self.group_read_mode = str(group_read_mode)
        if self.group_read_mode not in GROUP_READ_MODES:
            allowed = ", ".join(sorted(GROUP_READ_MODES))
            raise ValueError(f"group_read_mode must be one of: {allowed}")
        self.relative_fourier = FourierFeatures(None, fourier_frequencies)
        self.group_position_fourier = FourierFeatures(None, fourier_frequencies)

        self.module_membership = LazyMLP(message_hidden_dim, out_dim=1, num_layers=2)
        self.environment_membership = LazyMLP(message_hidden_dim, out_dim=1, num_layers=2)
        _zero_last_bias(self.module_membership)
        _zero_last_bias(self.environment_membership)
        self.module_message = LazyMLP(message_hidden_dim, out_dim=hidden_dim, num_layers=3)
        self.environment_message = LazyMLP(message_hidden_dim, out_dim=hidden_dim, num_layers=3)

        self.group_input = LazyMLP(hidden_dim, num_layers=2)
        self.group_residual = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=3)
        self.group_norm = nn.LayerNorm(hidden_dim)
        self.group_key = nn.Linear(hidden_dim, hidden_dim)
        self.group_value = nn.Linear(hidden_dim, hidden_dim)

        self.receiver_query = LazyMLP(hidden_dim, num_layers=2)
        self.receiver_bias = LazyMLP(message_hidden_dim, out_dim=1, num_layers=2)

    def build_layout(
        self,
        encoded: EncodedInterfaceCase,
        module_port_coordinates: torch.Tensor,
        *,
        module_radius: float,
        port_quadrature_weights: torch.Tensor | None = None,
    ) -> SparseLayoutCache:
        spacing = float(module_radius) * self.support_spacing_factor
        layout = build_sparse_support_layout(
            module_port_coordinates,
            encoded.module_present,
            encoded.env_coords,
            encoded.env_weights,
            spacing=spacing,
            port_quadrature_weights=port_quadrature_weights,
        )
        return SparseLayoutCache(layout=layout)

    def _environment_cache(
        self,
        cache: SparseLayoutCache,
        encoded: EncodedInterfaceCase,
    ) -> SparseEnvironmentCache:
        if cache.environment is not None:
            return cache.environment
        layout = cache.layout
        source_index, group_index = layout.environment_group_indices
        batch_index = torch.div(source_index, layout.environment_count, rounding_mode="floor")
        local_index = torch.remainder(source_index, layout.environment_count)
        env_tokens = encoded.env_tokens[batch_index, local_index]
        relative = (
            encoded.env_coords[batch_index, local_index] - layout.centres[group_index]
        ) / layout.spacing
        relative_features = self.relative_fourier(relative)
        membership = torch.sigmoid(
            self.environment_membership(torch.cat([env_tokens, relative_features], dim=-1)).squeeze(-1)
        )
        messages = self.environment_message(torch.cat([env_tokens, relative_features], dim=-1))
        weighted = layout.environment_geometric_weight * membership
        pool = messages.new_zeros(layout.group_count, self.hidden_dim)
        pool.index_add_(0, group_index, weighted[:, None] * messages)
        pool = pool / (layout.spacing ** layout.spatial_dim)
        environment = SparseEnvironmentCache(pool=pool, learned_membership=membership)
        cache.environment = environment
        return environment

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        layout_cache: SparseLayoutCache,
    ) -> SparseGroupState:
        layout = layout_cache.layout
        environment = self._environment_cache(layout_cache, encoded)
        source_index, group_index = layout.module_group_indices
        batch_index = torch.div(source_index, layout.module_count, rounding_mode="floor")
        local_index = torch.remainder(source_index, layout.module_count)
        states = module_states[batch_index, local_index]
        module_features = encoded.module_features[batch_index, local_index]
        relative = (
            encoded.module_centers[batch_index, local_index] - layout.centres[group_index]
        ) / layout.spacing
        relative_features = self.relative_fourier(relative)
        membership = torch.sigmoid(
            self.module_membership(
                torch.cat([states, module_features, relative_features], dim=-1)
            ).squeeze(-1)
        )
        messages = self.module_message(
            torch.cat([states, module_features, relative_features], dim=-1)
        )
        weighted = layout.module_support_weight * membership
        module_pool = messages.new_zeros(layout.group_count, self.hidden_dim)
        module_pool.index_add_(0, group_index, weighted[:, None] * messages)
        module_pool = module_pool / (1.0 + layout.occupancy[:, None])

        group_batch = layout.group_batch
        coordinate_scale = encoded.coordinate_scale
        if coordinate_scale.ndim == 3:
            if coordinate_scale.shape[0] not in {1, layout.batch_size}:
                raise ValueError("encoded.coordinate_scale batch dimension does not match the support layout.")
            per_case_scale = coordinate_scale[:, 0, :]
        elif coordinate_scale.ndim == 2:
            per_case_scale = coordinate_scale
        elif coordinate_scale.ndim == 1:
            per_case_scale = coordinate_scale.unsqueeze(0)
        else:
            raise ValueError("encoded.coordinate_scale must have shape [d], [1|B,d], or [1|B,1,d].")
        if per_case_scale.shape[0] == 1:
            per_case_scale = per_case_scale.expand(layout.batch_size, -1)
        if per_case_scale.shape != (layout.batch_size, layout.spatial_dim):
            raise ValueError("encoded.coordinate_scale dimension does not match the support layout.")
        group_scale = per_case_scale.index_select(0, group_batch)
        position_features = self.group_position_fourier(
            layout.centres / group_scale
        )
        global_features = encoded.global_token[group_batch]
        group_inputs = torch.cat(
            [
                module_pool,
                environment.pool,
                torch.log1p(layout.occupancy)[:, None],
                layout.covered_volume_ratio[:, None],
                position_features,
                global_features,
            ],
            dim=-1,
        )
        base = self.group_input(group_inputs)
        group_state = base + self.group_residual(base)
        normalized = self.group_norm(group_state)
        return SparseGroupState(
            cache=layout_cache,
            group_state=group_state,
            group_keys=self.group_key(normalized),
            group_values=self.group_value(normalized),
            module_learned_membership=membership,
            module_pool=module_pool,
        )

    def read(
        self,
        state: SparseGroupState,
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Read group values at physical receivers.

        ``null_softmax`` retains the historical explicit-null arithmetic.  In
        ``geometry_envelope_attention`` the geometric availability is kept as
        an outer envelope and the positive incidences are normalized only
        against one another.  The optional routing-map switch avoids sorting
        and materializing dense local slots for ordinary inference while
        retaining the inexpensive per-receiver summaries.
        """
        layout = state.cache.layout
        lookup = lookup_receivers(layout, receivers)
        flat_count = int(receivers.shape[0] * receivers.shape[1])
        query = self.receiver_query(
            torch.cat(
                [
                    receiver_features,
                    encoded.global_token[:, None, :].expand(-1, receivers.shape[1], -1),
                ],
                dim=-1,
            )
        ).reshape(flat_count, self.hidden_dim)
        receiver_index, group_index = lookup.receiver_group_indices
        relative = (
            receivers.reshape(flat_count, layout.spatial_dim)[receiver_index]
            - layout.centres[group_index]
        ) / layout.spacing
        logits = (
            (query[receiver_index] * state.group_keys[group_index]).sum(dim=-1)
            / math.sqrt(float(self.hidden_dim))
            + self.receiver_bias(self.relative_fourier(relative)).squeeze(-1)
        )
        geometric = layout.occupancy_envelope[group_index] * lookup.geometric_weight

        # The historical path deliberately retains its explicit null entry.
        # Keep this arithmetic unchanged so old sparse checkpoints have the
        # same output and gradient semantics when the new setting is absent.
        if self.group_read_mode == "null_softmax":
            selected = torch.ones_like(geometric, dtype=torch.bool)
            selected_receiver = receiver_index
            selected_group = group_index
            selected_geometric = geometric
            selected_logits = logits

            maxima = logits.new_zeros(flat_count)
            maxima.scatter_reduce_(0, receiver_index, logits, reduce="amax", include_self=True)
            shifted = logits - maxima[receiver_index]
            unnormalized = geometric * torch.exp(shifted)
            mass = logits.new_zeros(flat_count)
            mass.index_add_(0, receiver_index, unnormalized)
            denominator = torch.exp(-maxima) + mass
            context = state.group_values.new_zeros(flat_count, self.hidden_dim)
            context.index_add_(
                0,
                receiver_index,
                unnormalized[:, None] * state.group_values[group_index],
            )
            context = context / denominator[:, None]

            effective_weight = unnormalized / denominator[receiver_index]
            tiny = torch.finfo(logits.dtype).tiny
            conditional_weight = unnormalized / mass[receiver_index].clamp_min(tiny)
            nonnull_mass = mass / denominator
            null_mass = 1.0 - nonnull_mass
            log_partition = maxima + torch.log(mass.clamp_min(tiny))
            # Empty receivers use the historical Z=0 convention rather than
            # exposing log(tiny) as a measured partition value.
            log_partition = torch.where(
                mass > 0.0,
                log_partition,
                torch.full_like(log_partition, -torch.inf),
            )
        else:
            # Only actual positive-weight incidences participate in the
            # conditional normalizer.  In particular, an empty receiver (or a
            # numerically vanishing envelope) is not represented by a learned
            # null logit or by a positive floor.
            selected = geometric > 0.0
            selected_receiver = receiver_index[selected]
            selected_group = group_index[selected]
            selected_geometric = geometric[selected]
            selected_logits = logits[selected]
            # Double precision scalar normalization preserves derivatives
            # through subnormal float32 support weights (the focused tiny-g
            # test exercises this). Context vectors remain in the model dtype.
            normalization_dtype = (
                torch.float64
                if selected_geometric.dtype in {torch.float16, torch.bfloat16, torch.float32}
                else selected_geometric.dtype
            )
            selected_log_weight = selected_logits.to(normalization_dtype) + torch.log(
                selected_geometric.to(normalization_dtype)
            )

            maxima = torch.full(
                (flat_count,), -torch.inf, device=logits.device, dtype=normalization_dtype
            )
            if selected_log_weight.numel():
                maxima.scatter_reduce_(
                    0,
                    selected_receiver,
                    selected_log_weight,
                    reduce="amax",
                    include_self=True,
                )
            shifted = selected_log_weight - maxima[selected_receiver]
            unnormalized = torch.exp(shifted)
            mass = torch.zeros(flat_count, device=logits.device, dtype=normalization_dtype)
            mass.index_add_(0, selected_receiver, unnormalized)
            availability = logits.new_zeros(flat_count)
            availability.index_add_(0, selected_receiver, selected_geometric)
            tiny = torch.finfo(normalization_dtype).tiny
            conditional_weight_normalized = unnormalized / mass[selected_receiver].clamp_min(tiny)
            effective_weight_normalized = (
                conditional_weight_normalized * availability[selected_receiver].to(normalization_dtype)
            )
            conditional_weight = conditional_weight_normalized.to(logits.dtype)
            effective_weight = effective_weight_normalized.to(logits.dtype)
            context = state.group_values.new_zeros(flat_count, self.hidden_dim)
            context.index_add_(
                0,
                selected_receiver,
                effective_weight[:, None] * state.group_values[selected_group],
            )
            nonnull_mass = availability
            null_mass = logits.new_zeros(flat_count)
            log_partition = (maxima + torch.log(mass.clamp_min(tiny))).to(logits.dtype)
            log_partition = torch.where(
                mass > 0.0,
                log_partition,
                torch.full_like(log_partition, -torch.inf),
            )

        if self.group_read_mode == "null_softmax":
            availability = logits.new_zeros(flat_count)
            availability.index_add_(0, selected_receiver, selected_geometric)
        dominant_weight = logits.new_zeros(flat_count)
        dominant_weight.scatter_reduce_(
            0,
            selected_receiver,
            effective_weight,
            reduce="amax",
            include_self=True,
        )

        aux: Dict[str, torch.Tensor] = {
            "group_read_degree": lookup.degree.reshape(receivers.shape[0], receivers.shape[1]),
            "group_read_weight_mass": nonnull_mass.reshape(receivers.shape[0], receivers.shape[1]),
            "group_read_max_weight": dominant_weight.reshape(receivers.shape[0], receivers.shape[1]),
            "group_read_geometric_availability": availability.reshape(
                receivers.shape[0], receivers.shape[1]
            ),
            "group_read_log_partition": log_partition.reshape(
                receivers.shape[0], receivers.shape[1]
            ),
            "group_read_null_weight_mass": null_mass.reshape(
                receivers.shape[0], receivers.shape[1]
            ),
        }

        if not return_routing_maps:
            return context.reshape(receivers.shape[0], receivers.shape[1], self.hidden_dim), aux

        if self.group_read_mode == "null_softmax":
            # Historical context/mass above retain their original arithmetic.
            # Diagnostic conditional attention must remain meaningful even
            # when that FP32 null read underflows. Normalize positive g in
            # log space instead of dividing by a clamped subnormal mass.
            positive = selected_geometric > 0.0
            diagnostic_receivers = selected_receiver[positive]
            diagnostic_log_weight = selected_logits[positive].double() + torch.log(
                selected_geometric[positive].double()
            )
            diagnostic_max = torch.full(
                (flat_count,), -torch.inf, device=logits.device, dtype=torch.float64
            )
            diagnostic_max.scatter_reduce_(
                0, diagnostic_receivers, diagnostic_log_weight, reduce="amax", include_self=True
            )
            diagnostic_exp = torch.exp(diagnostic_log_weight - diagnostic_max[diagnostic_receivers])
            diagnostic_mass = torch.zeros_like(diagnostic_max)
            diagnostic_mass.index_add_(0, diagnostic_receivers, diagnostic_exp)
            conditional_weight = torch.zeros_like(selected_geometric)
            conditional_weight[positive] = (
                diagnostic_exp / diagnostic_mass[diagnostic_receivers]
            ).to(logits.dtype)
            aux["group_read_log_partition"] = (
                diagnostic_max + torch.log(diagnostic_mass)
            ).to(logits.dtype).reshape(receivers.shape[0], receivers.shape[1])

        # The remaining summaries and per-slot arrays are intentionally
        # detailed exports.  They are requested by selected diagnostics and
        # evaluations, rather than materialized on every training read.
        conditional_context = state.group_values.new_zeros(flat_count, self.hidden_dim)
        conditional_context.index_add_(
            0,
            selected_receiver,
            conditional_weight[:, None] * state.group_values[selected_group],
        )
        conditional_value_norm = torch.linalg.vector_norm(conditional_context, dim=-1)
        summary_count = logits.new_zeros(flat_count)
        summary_count.index_add_(
            0, selected_receiver, torch.ones_like(selected_logits)
        )
        summary_safe_count = summary_count.clamp_min(1.0)

        def _summary_mean(values: torch.Tensor) -> torch.Tensor:
            total = logits.new_zeros(flat_count)
            total.index_add_(0, selected_receiver, values)
            return total / summary_safe_count

        # Center before squaring so a common logit offset near -60 (or lower)
        # does not erase the within-receiver spread through cancellation.
        diagnostic_dtype = (
            torch.float64 if selected_logits.dtype == torch.float32 else selected_logits.dtype
        )
        diagnostic_logits = selected_logits.to(diagnostic_dtype)
        diagnostic_mean = torch.zeros(
            flat_count, device=logits.device, dtype=diagnostic_dtype
        )
        diagnostic_mean.index_add_(0, selected_receiver, diagnostic_logits)
        diagnostic_mean = diagnostic_mean / summary_safe_count.to(diagnostic_dtype)
        centered_logits = diagnostic_logits - diagnostic_mean[selected_receiver]
        centered_second = torch.zeros(
            flat_count, device=logits.device, dtype=diagnostic_dtype
        )
        centered_second.index_add_(0, selected_receiver, centered_logits.square())
        logit_mean = diagnostic_mean.to(logits.dtype)
        logit_std = (
            centered_second / summary_safe_count.to(diagnostic_dtype)
        ).clamp_min(0.0).sqrt().to(logits.dtype)
        dot_values = (
            (query[selected_receiver] * state.group_keys[selected_group]).sum(dim=-1)
            / math.sqrt(float(self.hidden_dim))
            if selected_receiver.numel()
            else logits.new_empty((0,))
        )
        # ``bias`` is selected from the already-computed compatibility terms;
        # recomputing Fourier features here would duplicate work and could
        # make detailed statistics disagree at mixed precision boundaries.
        bias_values = selected_logits - dot_values
        key_norm_values = torch.linalg.vector_norm(state.group_keys[selected_group], dim=-1)
        value_norm_values = torch.linalg.vector_norm(state.group_values[selected_group], dim=-1)
        dot_mean = _summary_mean(dot_values)
        bias_mean = _summary_mean(bias_values)
        key_norm_mean = _summary_mean(key_norm_values)
        value_norm_mean = _summary_mean(value_norm_values)
        query_norm = torch.linalg.vector_norm(query, dim=-1)
        conditional_key_norm = logits.new_zeros(flat_count)
        conditional_value_norm_from_rows = logits.new_zeros(flat_count)
        conditional_key_norm.index_add_(
            0, selected_receiver, conditional_weight * key_norm_values
        )
        conditional_value_norm_from_rows.index_add_(
            0, selected_receiver, conditional_weight * value_norm_values
        )
        aux.update(
            {
                "group_read_logit_mean": logit_mean.reshape(receivers.shape[0], receivers.shape[1]),
                "group_read_logit_std": logit_std.reshape(receivers.shape[0], receivers.shape[1]),
                "group_read_dot_product_mean": dot_mean.reshape(
                    receivers.shape[0], receivers.shape[1]
                ),
                "group_read_bias_mean": bias_mean.reshape(
                    receivers.shape[0], receivers.shape[1]
                ),
                "group_read_query_norm": query_norm.reshape(receivers.shape[0], receivers.shape[1]),
                "group_read_key_norm_mean": key_norm_mean.reshape(
                    receivers.shape[0], receivers.shape[1]
                ),
                "group_read_value_norm_mean": value_norm_mean.reshape(
                    receivers.shape[0], receivers.shape[1]
                ),
                "group_read_conditional_key_norm": conditional_key_norm.reshape(
                    receivers.shape[0], receivers.shape[1]
                ),
                "group_read_conditional_value_norm": conditional_value_norm.reshape(
                    receivers.shape[0], receivers.shape[1]
                ),
                "group_read_conditional_value_norm_mean": conditional_value_norm_from_rows.reshape(
                    receivers.shape[0], receivers.shape[1]
                ),
            }
        )

        # Bounded local routing slots expose real support IDs without ever
        # allocating a dense receiver-by-K matrix.  lookup_receiver order is
        # normalized here so chunking/evaluation can concatenate by receiver.
        order = torch.argsort(selected_receiver, stable=True)
        ordered_receiver = selected_receiver[order]
        ordered_group = selected_group[order]
        ordered_geometric = lookup.geometric_weight[selected][order]
        ordered_effective_geometric = selected_geometric[order]
        ordered_weight = effective_weight[order]
        ordered_conditional = conditional_weight[order]
        ordered_logits = selected_logits[order]
        ordered_dot = dot_values[order]
        ordered_bias = bias_values[order]
        ordered_key_norm = key_norm_values[order]
        ordered_value_norm = value_norm_values[order]
        counts = torch.bincount(ordered_receiver, minlength=flat_count)
        starts = torch.cumsum(counts, dim=0) - counts
        slot = torch.arange(order.numel(), device=order.device) - torch.repeat_interleave(starts, counts)
        slot_count = 4 ** layout.spatial_dim
        group_slots = torch.full(
            (flat_count, slot_count), -1, device=group_index.device, dtype=group_index.dtype
        )
        geometric_slots = logits.new_zeros(flat_count, slot_count)
        effective_geometric_slots = logits.new_zeros(flat_count, slot_count)
        normalized_slots = logits.new_zeros(flat_count, slot_count)
        conditional_slots = logits.new_zeros(flat_count, slot_count)
        logit_slots = logits.new_zeros(flat_count, slot_count)
        dot_slots = logits.new_zeros(flat_count, slot_count)
        bias_slots = logits.new_zeros(flat_count, slot_count)
        key_norm_slots = logits.new_zeros(flat_count, slot_count)
        value_norm_slots = logits.new_zeros(flat_count, slot_count)
        group_slots[ordered_receiver, slot] = ordered_group
        geometric_slots[ordered_receiver, slot] = ordered_geometric
        effective_geometric_slots[ordered_receiver, slot] = ordered_effective_geometric
        normalized_slots[ordered_receiver, slot] = ordered_weight
        conditional_slots[ordered_receiver, slot] = ordered_conditional
        logit_slots[ordered_receiver, slot] = ordered_logits
        dot_slots[ordered_receiver, slot] = ordered_dot
        bias_slots[ordered_receiver, slot] = ordered_bias
        key_norm_slots[ordered_receiver, slot] = ordered_key_norm
        value_norm_slots[ordered_receiver, slot] = ordered_value_norm
        aux.update(
            {
                "group_read_group_index": group_slots.reshape(
                    receivers.shape[0], receivers.shape[1], slot_count
                ),
                "group_read_geometric_weight": geometric_slots.reshape(
                    receivers.shape[0], receivers.shape[1], slot_count
                ),
                "group_read_effective_geometric_weight": effective_geometric_slots.reshape(
                    receivers.shape[0], receivers.shape[1], slot_count
                ),
                "group_read_normalized_weight": normalized_slots.reshape(
                    receivers.shape[0], receivers.shape[1], slot_count
                ),
                "group_read_conditional_weight": conditional_slots.reshape(
                    receivers.shape[0], receivers.shape[1], slot_count
                ),
                "group_read_logit": logit_slots.reshape(
                    receivers.shape[0], receivers.shape[1], slot_count
                ),
                "group_read_dot_product": dot_slots.reshape(
                    receivers.shape[0], receivers.shape[1], slot_count
                ),
                "group_read_bias": bias_slots.reshape(
                    receivers.shape[0], receivers.shape[1], slot_count
                ),
                "group_read_key_norm": key_norm_slots.reshape(
                    receivers.shape[0], receivers.shape[1], slot_count
                ),
                "group_read_value_norm": value_norm_slots.reshape(
                    receivers.shape[0], receivers.shape[1], slot_count
                ),
            }
        )
        return context.reshape(receivers.shape[0], receivers.shape[1], self.hidden_dim), aux

    @staticmethod
    def preparation_aux(state: SparseGroupState) -> Dict[str, torch.Tensor | int | float]:
        layout = state.cache.layout
        group_count_per_case = torch.diff(layout.case_group_offsets)
        module_incidence_batch = layout.group_batch.index_select(
            0, layout.module_group_indices[1]
        )
        environment_incidence_batch = layout.group_batch.index_select(
            0, layout.environment_group_indices[1]
        )
        module_incidence_per_case = torch.bincount(
            module_incidence_batch, minlength=layout.batch_size
        )
        environment_incidence_per_case = torch.bincount(
            environment_incidence_batch, minlength=layout.batch_size
        )
        module_degree = layout.occupancy.new_zeros(layout.batch_size * layout.module_count)
        module_degree.index_add_(
            0,
            layout.module_group_indices[0],
            torch.ones_like(layout.module_support_weight),
        )
        group_module_degree = layout.occupancy.new_zeros(layout.group_count)
        group_module_degree.index_add_(
            0,
            layout.module_group_indices[1],
            torch.ones_like(layout.module_support_weight),
        )
        group_environment_degree = layout.occupancy.new_zeros(layout.group_count)
        group_environment_degree.index_add_(
            0,
            layout.environment_group_indices[1],
            torch.ones_like(layout.environment_geometric_weight),
        )
        return {
            "group_count_per_case": group_count_per_case,
            "module_group_incidence_count_per_case": module_incidence_per_case,
            "environment_group_incidence_count_per_case": environment_incidence_per_case,
            "group_count_batch_total": layout.group_count,
            "module_group_incidence_count_batch_total": int(layout.module_support_weight.numel()),
            "environment_group_incidence_count_batch_total": int(layout.environment_geometric_weight.numel()),
            "support_spacing": float(layout.spacing),
            "support_centres": layout.centres,
            "support_lattice_keys": layout.lattice_keys,
            "support_group_batch": layout.group_batch,
            "support_case_group_offsets": layout.case_group_offsets,
            "module_group_indices": layout.module_group_indices,
            "module_geometric_membership": layout.module_support_weight,
            "module_learned_membership": state.module_learned_membership,
            "environment_group_indices": layout.environment_group_indices,
            "environment_geometric_membership": layout.environment_geometric_weight,
            "environment_learned_membership": state.cache.environment.learned_membership,
            "group_occupancy": layout.occupancy,
            "group_occupancy_envelope": layout.occupancy_envelope,
            "group_covered_volume_ratio": layout.covered_volume_ratio,
            "module_support_degree": module_degree.reshape(layout.batch_size, layout.module_count),
            "group_module_degree": group_module_degree,
            "group_environment_degree": group_environment_degree,
            "group_state_norm": torch.linalg.vector_norm(state.group_state, dim=-1),
            "group_key_norm": torch.linalg.vector_norm(state.group_keys, dim=-1),
            "group_value_norm": torch.linalg.vector_norm(state.group_values, dim=-1),
        }
