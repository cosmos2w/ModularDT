"""Sparse nonlinear multi-entity group operator for interface-centred HONF."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from honf_forward_core.nn import FourierFeatures, LazyMLP, MLP
from .supports import SparseSupportLayout, build_sparse_support_layout, lookup_receivers
from .types import EncodedInterfaceCase


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
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.support_spacing_factor = float(support_spacing_factor)
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
        position_features = self.group_position_fourier(
            layout.centres / encoded.coordinate_scale.reshape(1, layout.spatial_dim)
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
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        # Stable local softmax with an explicit null entry.  The occupancy
        # envelope is multiplied outside the exponential and is never
        # normalized away when a sole support disappears.
        maxima = logits.new_zeros(flat_count)
        maxima.scatter_reduce_(0, receiver_index, logits, reduce="amax", include_self=True)
        shifted = logits - maxima[receiver_index]
        unnormalized = (
            layout.occupancy_envelope[group_index]
            * lookup.geometric_weight
            * torch.exp(shifted)
        )
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

        normalized_weight = unnormalized / denominator[receiver_index]
        dominant_weight = logits.new_zeros(flat_count)
        dominant_weight.scatter_reduce_(
            0, receiver_index, normalized_weight, reduce="amax", include_self=True
        )
        # Bounded local routing slots expose real support IDs without ever
        # allocating a dense receiver-by-K matrix.  lookup_receiver order is
        # normalized here so chunking/evaluation can concatenate by receiver.
        order = torch.argsort(receiver_index, stable=True)
        ordered_receiver = receiver_index[order]
        ordered_group = group_index[order]
        ordered_geometric = lookup.geometric_weight[order]
        ordered_weight = normalized_weight[order]
        counts = torch.bincount(ordered_receiver, minlength=flat_count)
        starts = torch.cumsum(counts, dim=0) - counts
        slot = torch.arange(order.numel(), device=order.device) - torch.repeat_interleave(starts, counts)
        slot_count = 4 ** layout.spatial_dim
        group_slots = torch.full(
            (flat_count, slot_count), -1, device=group_index.device, dtype=group_index.dtype
        )
        geometric_slots = logits.new_zeros(flat_count, slot_count)
        normalized_slots = logits.new_zeros(flat_count, slot_count)
        group_slots[ordered_receiver, slot] = ordered_group
        geometric_slots[ordered_receiver, slot] = ordered_geometric
        normalized_slots[ordered_receiver, slot] = ordered_weight
        return context.reshape(receivers.shape[0], receivers.shape[1], self.hidden_dim), {
            "group_read_degree": lookup.degree.reshape(receivers.shape[0], receivers.shape[1]),
            "group_read_weight_mass": (mass / denominator).reshape(receivers.shape[0], receivers.shape[1]),
            "group_read_max_weight": dominant_weight.reshape(receivers.shape[0], receivers.shape[1]),
            "group_read_group_index": group_slots.reshape(
                receivers.shape[0], receivers.shape[1], slot_count
            ),
            "group_read_geometric_weight": geometric_slots.reshape(
                receivers.shape[0], receivers.shape[1], slot_count
            ),
            "group_read_normalized_weight": normalized_slots.reshape(
                receivers.shape[0], receivers.shape[1], slot_count
            ),
        }

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
        }
