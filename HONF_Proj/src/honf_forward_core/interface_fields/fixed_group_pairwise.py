"""Run-1405 fixed-group pairwise interface-field backend.

This backend deliberately subclasses :class:`DensePairwiseField`.  Dense's
simultaneous MM/ME/EM preparation is the controlled source-state baseline;
only the receiver-side reader is replaced by the fixed-K group-conditioned
module/environment terms described in the Run-1405 plan.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from honf_forward_core.nn import LazyMLP

from .dense_pairwise import DensePairwiseField
from .fixed_group_router import FixedGroupQueryRoute, FixedGroupRouter, FixedGroupState
from .types import EncodedInterfaceCase


class FixedGroupPairwiseField(DensePairwiseField):
    """Dense fine preparation plus fixed-six-group conditioned reading."""

    def __init__(
        self,
        hidden_dim: int,
        message_hidden_dim: int,
        num_heads: int,
        fourier_frequencies: int,
        *,
        group_count: int = 6,
        group_code_dim: int = 32,
        spatial_dim: int = 2,
        module_temperature: float = 1.0,
        environment_temperature: float = 1.0,
        query_temperature: float = 1.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            hidden_dim,
            message_hidden_dim,
            num_heads,
            fourier_frequencies,
            activation_checkpointing=activation_checkpointing,
        )
        # Reuse Dense's MM/ME/EM preparation only.  Its receiver-side module
        # and environment readers are forbidden bypasses in Run 1405; remove
        # them from this opt-in module tree so they are neither trainable nor
        # left as uninitialized LazyModules during optimizer diagnostics.
        self.query_module_message = None
        self.query_module_output = None
        self.env_query = None
        self.env_attention = None
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for grouped environment attention.")
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_dim) // self.num_heads
        self.spatial_dim = int(spatial_dim)
        self.group_count = int(group_count)
        self.router = FixedGroupRouter(
            hidden_dim,
            group_count=group_count,
            group_code_dim=group_code_dim,
            fourier_frequencies=fourier_frequencies,
            spatial_dim=spatial_dim,
            module_temperature=module_temperature,
            environment_temperature=environment_temperature,
            query_temperature=query_temperature,
        )

        # Source codes are prepared once per physical state.  They are not
        # pooled into a field value; the fine source identity is retained for
        # the membership-weighted triadic module read and group-wise K/V read.
        self.module_source_code = LazyMLP(
            hidden_dim,
            out_dim=hidden_dim,
            num_layers=3,
        )
        self.environment_source_code = LazyMLP(
            hidden_dim,
            out_dim=hidden_dim,
            num_layers=3,
        )
        self.module_pair_message = LazyMLP(
            hidden_dim,
            out_dim=hidden_dim,
            num_layers=3,
        )

        # Group-conditioned environment attention uses the same head count as
        # Dense, but owns explicit K/V projections because every source has a
        # group index.  The source LayerNorm and output projection mirror the
        # existing BiasedMultiheadAttention contract.
        self.environment_source_norm = nn.LayerNorm(hidden_dim)
        self.environment_query_norm = nn.LayerNorm(hidden_dim)
        self.environment_query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.environment_key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.environment_value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.environment_output_projection = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def _batch_scale(encoded: EncodedInterfaceCase, batch_index: int) -> torch.Tensor:
        """Return one case's coordinate scale as a flat spatial vector."""

        scale = encoded.coordinate_scale
        if scale.ndim == 3 or scale.ndim == 2:
            selected = scale[batch_index if int(scale.shape[0]) > 1 else 0]
        elif scale.ndim == 1:
            selected = scale
        else:
            raise ValueError("encoded.coordinate_scale must have one, two, or three dimensions.")
        return selected.reshape(-1)

    @staticmethod
    def _batch_scales(encoded: EncodedInterfaceCase, batch_size: int) -> torch.Tensor:
        """Return coordinate scales as ``[B,d]`` without device transfers."""

        scale = encoded.coordinate_scale
        if scale.ndim == 3:
            scale = scale[:, 0, :]
        elif scale.ndim == 2:
            # Direct backend callers may provide either ``[B,d]`` or the
            # broadcastable ``[1,d]`` form.
            pass
        elif scale.ndim == 1:
            scale = scale[None, :]
        else:
            raise ValueError("encoded.coordinate_scale must have one, two, or three dimensions.")
        if int(scale.shape[0]) == 1:
            scale = scale.expand(batch_size, -1)
        if int(scale.shape[0]) != batch_size:
            raise ValueError("encoded.coordinate_scale batch dimension does not match the read batch.")
        return scale

    @staticmethod
    def _support_pairs(
        query_support: torch.Tensor,
        source_support: torch.Tensor,
        query_count: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build flattened same-batch Cartesian supports for one group.

        The returned query/source indices are local to their gathered support
        arrays.  ``query_global`` indexes a flattened ``[B,Q]`` output, and
        ``source_count_per_query`` is the semantic grouped row count used by
        the compute ledger.
        """

        query_indices = torch.nonzero(query_support, as_tuple=False)
        source_indices = torch.nonzero(source_support, as_tuple=False)
        query_counts = query_support.sum(dim=1, dtype=torch.long)
        source_counts = source_support.sum(dim=1, dtype=torch.long)
        source_count_per_query = torch.repeat_interleave(source_counts, query_counts)
        # ``nonzero`` is lexicographically ordered [batch, query, source],
        # which gives q-major Cartesian rows.  Rank maps translate those
        # global local-source coordinates to the compact gathered arrays.
        pair_indices = torch.nonzero(
            query_support[:, :, None] & source_support[:, None, :],
            as_tuple=False,
        )
        query_rank = query_support.to(torch.long).cumsum(dim=1) - 1
        source_rank = source_support.to(torch.long).cumsum(dim=1) - 1
        query_offsets = query_counts.cumsum(dim=0) - query_counts
        source_offsets = source_counts.cumsum(dim=0) - source_counts
        query_rank = query_rank + query_offsets[:, None]
        source_rank = source_rank + source_offsets[:, None]
        query_pair_index = query_rank[pair_indices[:, 0], pair_indices[:, 1]]
        source_pair_index = source_rank[pair_indices[:, 0], pair_indices[:, 2]]
        query_global = query_indices[:, 0] * int(query_count) + query_indices[:, 1]
        return (
            query_indices,
            source_indices,
            query_pair_index,
            source_pair_index,
            query_global,
            source_count_per_query,
        )

    @staticmethod
    def _canonical_support_order(
        source_indices: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Sort gathered sources by physical coordinates for permutation-stable sums."""

        values = coordinates[source_indices[:, 0], source_indices[:, 1]]
        order = torch.arange(source_indices.shape[0], device=source_indices.device)
        # Stable least-significant-to-most-significant lexicographic sort.
        for dimension in range(int(values.shape[-1]) - 1, -1, -1):
            order = order[torch.argsort(values[order, dimension], stable=True)]
        return order

    @staticmethod
    def _env_log_weights(
        weights: torch.Tensor,
        membership: torch.Tensor,
    ) -> torch.Tensor:
        """Return finite ``log(A^E_jk)`` with zero-support masking separate."""

        safe_membership = torch.where(
            membership > 0.0,
            membership,
            torch.ones_like(membership),
        )
        del weights
        return torch.log(safe_membership.permute(0, 2, 1))

    def _prepare_environment_kv(
        self,
        environment_source_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project ``[B,E,K,H]`` source codes to ``[B,K,heads,E,head_dim]``."""

        batch, environment_count, groups, hidden = environment_source_codes.shape
        if groups != self.group_count or hidden != self.hidden_dim:
            raise ValueError("Environment source-code shape does not match the fixed-group backend.")
        normalized = self.environment_source_norm(environment_source_codes)
        key = self.environment_key_projection(normalized).reshape(
            batch,
            environment_count,
            groups,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 3, 1, 4)
        value = self.environment_value_projection(normalized).reshape(
            batch,
            environment_count,
            groups,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 3, 1, 4)
        return key, value

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        """Run Dense MM/ME/EM preparation and cache all group-side work."""

        del return_routing_maps
        # This call is intentionally the inherited Dense implementation.  It
        # is the compatibility boundary for the Run-1405 source-state design.
        fine = self.prepare_fine_messages(encoded, module_states)
        global_env = encoded.global_token[:, None, :].expand(-1, fine["env_tokens"].shape[1], -1)
        contextual_env = fine["env_tokens"] + self.env_update(
            torch.cat([fine["env_tokens"], fine["environment_messages"], global_env], dim=-1)
        )

        group_state = self.router.prepare(
            encoded,
            fine["module_tokens"],
            contextual_env,
        )
        module_source_codes = self._build_module_source_codes(
            encoded,
            fine["module_tokens"],
            group_state,
        )
        environment_source_codes = self._build_environment_source_codes(
            encoded,
            contextual_env,
            group_state,
        )
        environment_keys, environment_values = self._prepare_environment_kv(
            environment_source_codes
        )
        preparation_aux = self._preparation_summary(group_state)
        return {
            "module_tokens": fine["module_tokens"],
            "env_tokens": contextual_env,
            "router_state": group_state,
            "module_source_codes": module_source_codes,
            "environment_source_codes": environment_source_codes,
            "environment_keys": environment_keys,
            "environment_values": environment_values,
            "fixed_group_preparation_aux": preparation_aux,
        }

    def _build_module_source_codes(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        group_state: FixedGroupState,
    ) -> torch.Tensor:
        """Build ``u^M_ik`` with explicit ``h_k`` conditioning."""

        scale = self._batch_scales(encoded, int(module_states.shape[0]))[:, None, None, :]
        module_relative = (
            encoded.module_centers[:, :, None, :]
            - group_state.module_centres[:, None, :, :]
        ) / scale
        module_count = int(module_states.shape[1])
        module_source_input = torch.cat(
            [
                module_states[:, :, None, :].expand(-1, -1, self.group_count, -1),
                group_state.group_state[:, None, :, :].expand(-1, module_count, -1, -1),
                encoded.global_token[:, None, None, :].expand(
                    -1, module_count, self.group_count, -1
                ),
                self.relative_fourier(module_relative),
            ],
            dim=-1,
        )
        return self._mlp(self.module_source_code, module_source_input)

    def _build_environment_source_codes(
        self,
        encoded: EncodedInterfaceCase,
        environment_states: torch.Tensor,
        group_state: FixedGroupState,
    ) -> torch.Tensor:
        """Build ``u^E_jk`` with explicit ``h_k`` conditioning."""

        scale = self._batch_scales(encoded, int(environment_states.shape[0]))[:, None, None, :]
        environment_relative = (
            encoded.env_coords[:, :, None, :]
            - group_state.environment_centres[:, None, :, :]
        ) / scale
        environment_count = int(environment_states.shape[1])
        environment_source_input = torch.cat(
            [
                environment_states[:, :, None, :].expand(-1, -1, self.group_count, -1),
                group_state.group_state[:, None, :, :].expand(-1, environment_count, -1, -1),
                encoded.global_token[:, None, None, :].expand(
                    -1, environment_count, self.group_count, -1
                ),
                self.relative_fourier(environment_relative),
            ],
            dim=-1,
        )
        return self._mlp(self.environment_source_code, environment_source_input)

    @staticmethod
    def _preparation_summary(group_state: FixedGroupState) -> dict[str, torch.Tensor]:
        """Small detached summaries safe to aggregate across receiver reads."""

        module_incidence_count = (group_state.module_membership > 0.0).sum(dim=1).sum(dim=-1)
        environment_incidence_count = (
            (group_state.environment_membership > 0.0).sum(dim=1).sum(dim=-1)
        )
        active_module_count = (group_state.module_membership.sum(dim=-1) > 0.0).sum(dim=-1)
        environment_count = group_state.environment_membership.shape[1]
        return {
            # Mean positive memberships per source, not total incidence rows.
            "fixed_group_s_m": (
                module_incidence_count / active_module_count.clamp_min(1).to(module_incidence_count.dtype)
            ).detach(),
            "fixed_group_s_e": (
                environment_incidence_count / max(int(environment_count), 1)
            ).detach(),
            "fixed_group_valid_count": group_state.valid.sum(dim=-1).detach(),
            "fixed_group_module_mass": group_state.module_mass.detach(),
            "fixed_group_environment_mass": group_state.environment_mass.detach(),
        }

    def preparation_aux(
        self,
        state: dict[str, Any],
        *,
        include_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return scalar summaries, or opt-in incidence/geometry diagnostics.

        Incidence tensors are retained in the prepared state because the
        reader needs them.  They are copied into the evidence payload only
        when explicitly requested so ordinary training reads do not duplicate
        large tensors.
        """

        group_state: FixedGroupState = state["router_state"]
        summary = dict(state.get("fixed_group_preparation_aux", {}))
        if include_diagnostics:
            summary.update(
                {
                    "fixed_group_module_incidence": group_state.module_membership,
                    "fixed_group_environment_incidence": group_state.environment_membership,
                    "fixed_group_module_centres": group_state.module_centres,
                    "fixed_group_environment_centres": group_state.environment_centres,
                    "fixed_group_h": group_state.group_state,
                    "fixed_group_valid": group_state.valid,
                }
            )
        return summary

    def _route(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
    ) -> FixedGroupQueryRoute:
        return self.router.route_queries(
            encoded,
            state["router_state"],
            receivers,
            receiver_features,
        )

    def _module_group_response(
        self,
        query_group: torch.Tensor,
        source_group: torch.Tensor,
        query_coordinates: torch.Tensor,
        source_coordinates: torch.Tensor,
        scales: torch.Tensor,
        query_batch: torch.Tensor,
        query_pair_index: torch.Tensor,
        source_pair_index: torch.Tensor,
        query_global: torch.Tensor,
        normalized_membership: torch.Tensor,
        alpha: torch.Tensor,
        output_count: int,
    ) -> torch.Tensor:
        """Evaluate one group's triadic module response into flat query order."""

        pair_input = torch.cat(
            [
                query_group[query_pair_index],
                source_group[source_pair_index],
                self.relative_fourier(
                    (
                        query_coordinates[query_pair_index]
                        - source_coordinates[source_pair_index]
                    )
                    / scales[query_batch[query_pair_index]]
                ),
            ],
            dim=-1,
        )
        psi = self.module_pair_message(pair_input)
        weighted = (
            psi
            * normalized_membership[source_pair_index, None]
            * alpha[query_pair_index, None]
        )
        return psi.new_zeros((int(output_count), self.hidden_dim)).index_add(
            0,
            query_global[query_pair_index],
            weighted,
        )

    def _read_module_grouped(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        route: FixedGroupQueryRoute,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate triadic ``psi_M(q,i,k)`` only on positive supports.

        Query and source supports are gathered per batch/group before the
        pair MLP is called.  This is important for the intended execution
        model: a dense ``[B,Q,M]`` call repeated six times would preserve the
        mathematics but erase the receiver-side work reduction.
        """

        group_state: FixedGroupState = state["router_state"]
        batch, query_count, _ = receivers.shape
        context_flat = receivers.new_zeros((batch * query_count, self.hidden_dim))
        semantic_rows_flat = receivers.new_zeros(batch * query_count)
        scales = self._batch_scales(encoded, batch)
        for group_index in range(self.group_count):
            (
                query_indices,
                source_indices,
                query_pair_index,
                source_pair_index,
                query_global,
                source_count_per_query,
            ) = self._support_pairs(
                route.assignment[:, :, group_index] > 0.0,
                group_state.module_membership[:, :, group_index] > 0.0,
                query_count,
            )
            source_order = self._canonical_support_order(source_indices, encoded.module_centers)
            source_inverse = torch.empty_like(source_order)
            source_inverse[source_order] = torch.arange(
                source_order.shape[0], device=source_order.device
            )
            source_indices = source_indices[source_order]
            source_pair_index = source_inverse[source_pair_index]
            query_batch = query_indices[:, 0]
            source_batch = source_indices[:, 0]
            query_local = query_indices[:, 1]
            source_local = source_indices[:, 1]
            query_group = route.descriptor[:, :, group_index, :][query_batch, query_local]
            source_group = state["module_source_codes"][:, :, group_index, :][source_batch, source_local]
            query_coordinates = receivers[query_batch, query_local]
            source_coordinates = encoded.module_centers[source_batch, source_local]
            membership = group_state.module_membership[:, :, group_index][source_batch, source_local]
            mass = group_state.module_mass[:, group_index][source_batch]
            normalized_membership = membership / mass.clamp_min(torch.finfo(membership.dtype).tiny)
            alpha = route.assignment[:, :, group_index][query_batch, query_local]
            group_args = (
                query_group,
                source_group,
                query_coordinates,
                source_coordinates,
                scales,
                query_batch,
                query_pair_index,
                source_pair_index,
                query_global,
                normalized_membership,
                alpha,
                batch * query_count,
            )
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                group_response = checkpoint(
                    self._module_group_response,
                    *group_args,
                    use_reentrant=False,
                )
            else:
                group_response = self._module_group_response(*group_args)
            context_flat = context_flat + group_response
            semantic_rows_flat = semantic_rows_flat.index_add(
                0,
                query_global,
                source_count_per_query.to(receivers.dtype),
            )
        context = context_flat.reshape(batch, query_count, self.hidden_dim)
        semantic_rows = semantic_rows_flat.reshape(batch, query_count)
        active_group_count = (route.assignment > 0.0).sum(dim=-1).to(receivers.dtype)
        return context, semantic_rows, active_group_count

    def read_module(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        route: FixedGroupQueryRoute | None = None,
    ) -> torch.Tensor:
        """Read only the group-conditioned module term."""

        if route is None:
            route = self._route(state, encoded, receivers, receiver_features)
        context, _, _ = self._read_module_grouped(state, encoded, receivers, route)
        return context

    def _environment_group_response(
        self,
        query_projection: torch.Tensor,
        source_key: torch.Tensor,
        source_value: torch.Tensor,
        query_coordinates: torch.Tensor,
        source_coordinates: torch.Tensor,
        scales: torch.Tensor,
        source_weights: torch.Tensor,
        source_membership: torch.Tensor,
        alpha: torch.Tensor,
        query_batch: torch.Tensor,
        query_pair_index: torch.Tensor,
        source_pair_index: torch.Tensor,
        query_global: torch.Tensor,
        output_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate one group's exact sparse environment attention.

        The method boundary lets the caller checkpoint the whole attention
        calculation.  Only the compact ``[active queries, H]`` response then
        survives the forward pass; score/softmax tensors are recomputed in
        backward instead of being retained for every P0/P1/P2 receiver chunk.
        """

        score_parts: list[torch.Tensor] = []
        pair_chunk_size = 131_072
        for pair_start in range(0, int(query_pair_index.numel()), pair_chunk_size):
            pair_end = min(pair_start + pair_chunk_size, int(query_pair_index.numel()))
            query_pair_chunk = query_pair_index[pair_start:pair_end]
            source_pair_chunk = source_pair_index[pair_start:pair_end]
            query_pair = query_projection[:, query_pair_chunk, :]
            key_pair = source_key[:, source_pair_chunk, :]
            dot = (query_pair * key_pair).sum(dim=-1) / (float(self.head_dim) ** 0.5)
            pair_relative = (
                query_coordinates[query_pair_chunk]
                - source_coordinates[source_pair_chunk]
            ) / scales[query_batch[query_pair_chunk]]
            geometry_bias = self.env_geometry_bias(
                self.relative_fourier(pair_relative)
            ).transpose(0, 1)
            score_parts.append(dot + geometry_bias)
        scores = torch.cat(score_parts, dim=1)
        source_weights_pair = source_weights[source_pair_index]
        source_membership_pair = source_membership[source_pair_index]
        scores = scores + torch.log(
            source_weights_pair.clamp_min(torch.finfo(source_weights_pair.dtype).tiny)
        )[None, :]
        scores = scores + torch.log(
            source_membership_pair.clamp_min(torch.finfo(source_membership_pair.dtype).tiny)
        )[None, :]

        query_segments = query_global[query_pair_index]
        segment_index = query_segments[None, :].expand(self.num_heads, -1)
        max_scores = torch.full(
            (self.num_heads, int(output_count)),
            -torch.inf,
            device=scores.device,
            dtype=scores.dtype,
        )
        max_scores.scatter_reduce_(1, segment_index, scores, reduce="amax", include_self=True)
        unnormalized = torch.exp(scores - max_scores[:, query_segments])
        normalizers = scores.new_zeros((self.num_heads, int(output_count)))
        normalizers.index_add_(1, query_segments, unnormalized)
        attention = unnormalized / normalizers[:, query_segments].clamp_min(
            torch.finfo(unnormalized.dtype).tiny
        )

        sparse_indices = torch.stack([query_segments, source_pair_index], dim=0)
        response_heads = []
        source_count = int(source_value.shape[1])
        for head_index in range(self.num_heads):
            attention_matrix = torch.sparse_coo_tensor(
                sparse_indices,
                attention[head_index],
                size=(int(output_count), source_count),
                device=attention.device,
                dtype=attention.dtype,
                is_coalesced=True,
            )
            response_heads.append(
                torch.sparse.mm(attention_matrix, source_value[head_index])
            )
        response_flat = torch.stack(response_heads, dim=0)
        response = response_flat[:, query_global, :].permute(1, 0, 2).reshape(
            query_coordinates.shape[0], self.hidden_dim
        )
        response = self.environment_output_projection(response)
        return response * alpha[:, None], attention

    def _read_environment_grouped(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        route: FixedGroupQueryRoute,
        *,
        return_attention: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Read grouped K/V only on positive q/group/source supports.

        The dense score tensor would be ``[B,K,heads,Q,E]``.  Instead each
        batch/group gathers active query and environment rows, performs the
        ordinary multi-head softmax over that group's source support, and
        scatters the routed response back to query order.  Optional attention
        diagnostics are returned as packed rows with explicit indices.
        """

        group_state: FixedGroupState = state["router_state"]
        batch, query_count, _ = receivers.shape
        context_flat = receivers.new_zeros((batch * query_count, self.hidden_dim))
        semantic_rows_flat = receivers.new_zeros(batch * query_count)
        attention_values: list[torch.Tensor] = []
        attention_batch: list[torch.Tensor] = []
        attention_query: list[torch.Tensor] = []
        attention_group: list[torch.Tensor] = []
        attention_source: list[torch.Tensor] = []
        keys = state["environment_keys"]
        values = state["environment_values"]
        scales = self._batch_scales(encoded, batch)
        for group_index in range(self.group_count):
            (
                query_indices,
                source_indices,
                query_pair_index,
                source_pair_index,
                query_global,
                source_count_per_query,
            ) = self._support_pairs(
                route.assignment[:, :, group_index] > 0.0,
                group_state.environment_membership[:, :, group_index] > 0.0,
                query_count,
            )
            if int(query_pair_index.numel()) == 0:
                continue
            source_order = self._canonical_support_order(source_indices, encoded.env_coords)
            source_inverse = torch.empty_like(source_order)
            source_inverse[source_order] = torch.arange(
                source_order.shape[0], device=source_order.device
            )
            source_indices = source_indices[source_order]
            source_pair_index = source_inverse[source_pair_index]
            query_batch = query_indices[:, 0]
            query_local = query_indices[:, 1]
            source_batch = source_indices[:, 0]
            source_local = source_indices[:, 1]
            query_descriptor = route.descriptor[:, :, group_index, :][query_batch, query_local]
            query_projection = self.environment_query_projection(
                self.environment_query_norm(query_descriptor)
            ).reshape(query_indices.shape[0], self.num_heads, self.head_dim).transpose(0, 1)
            source_key = keys[:, group_index, :, :, :][source_batch, :, source_local, :].permute(1, 0, 2)
            source_value = values[:, group_index, :, :, :][source_batch, :, source_local, :].permute(1, 0, 2)
            query_coordinates = receivers[query_batch, query_local]
            source_coordinates = encoded.env_coords[source_batch, source_local]
            alpha = route.assignment[:, :, group_index][query_batch, query_local]
            group_args = (
                query_projection,
                source_key,
                source_value,
                query_coordinates,
                source_coordinates,
                scales,
                encoded.env_weights[source_batch, source_local],
                group_state.environment_membership[source_batch, source_local, group_index],
                alpha,
                query_batch,
                query_pair_index,
                source_pair_index,
                query_global,
                batch * query_count,
            )
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                weighted_response = checkpoint(
                    lambda *values_: self._environment_group_response(*values_)[0],
                    *group_args,
                    use_reentrant=False,
                )
                attention = None
            else:
                weighted_response, attention = self._environment_group_response(*group_args)
            context_flat = context_flat.index_add(0, query_global, weighted_response)
            semantic_rows_flat = semantic_rows_flat.index_add(
                0,
                query_global,
                source_count_per_query.to(receivers.dtype),
            )
            if return_attention:
                if attention is None:  # pragma: no cover - maps are not requested in training.
                    raise RuntimeError("Fixed-group attention diagnostics are unavailable during checkpointed training.")
                attention_values.append(attention.transpose(0, 1))
                attention_batch.append(query_indices[:, 0][query_pair_index])
                attention_query.append(query_indices[:, 1][query_pair_index])
                attention_group.append(
                    torch.full(
                        (query_pair_index.shape[0],),
                        group_index,
                        device=receivers.device,
                        dtype=torch.long,
                    )
                )
                attention_source.append(source_indices[:, 1][source_pair_index])
        context = context_flat.reshape(batch, query_count, self.hidden_dim)
        semantic_rows = semantic_rows_flat.reshape(batch, query_count)
        active_group_count = (route.assignment > 0.0).sum(dim=-1).to(receivers.dtype)
        packed_attention: torch.Tensor | None = None
        attention_aux: dict[str, torch.Tensor] = {}
        if return_attention and attention_values:
            packed_attention = torch.cat(attention_values, dim=0)
            attention_aux = {
                "fixed_group_environment_attention_batch": torch.cat(attention_batch, dim=0),
                "fixed_group_environment_attention_query": torch.cat(attention_query, dim=0),
                "fixed_group_environment_attention_group": torch.cat(attention_group, dim=0),
                "fixed_group_environment_attention_source": torch.cat(attention_source, dim=0),
            }
        return context, packed_attention, semantic_rows, active_group_count, attention_aux

    def read_environment(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        route: FixedGroupQueryRoute | None = None,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read only the group-conditioned environmental term."""

        if route is None:
            route = self._route(state, encoded, receivers, receiver_features)
        context, attention, _, _, _ = self._read_environment_grouped(
            state,
            encoded,
            receivers,
            route,
            return_attention=bool(return_routing_maps),
        )
        return context, attention

    @staticmethod
    def _semantic_triples(
        assignment: torch.Tensor,
        membership: torch.Tensor,
        *,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        """Materialize semantic q-group-source triples on explicit request."""

        support = (assignment[..., None] > 0.0) & (
            membership.permute(0, 2, 1)[:, None, :, :] > 0.0
        )
        indices = torch.nonzero(support, as_tuple=False)
        # ``indices`` is [triples, batch/query/group/source].  Separate arrays
        # are easier for chunk aggregation and avoid a large padded tensor.
        return {
            f"{prefix}_triple_batch": indices[:, 0],
            f"{prefix}_triple_query": indices[:, 1],
            f"{prefix}_triple_group": indices[:, 2],
            f"{prefix}_triple_source": indices[:, 3],
        }

    def read(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Read the sum of exactly the grouped module and environment terms."""

        route = self._route(state, encoded, receivers, receiver_features)
        module_context, module_rows, module_group_count = self._read_module_grouped(
            state,
            encoded,
            receivers,
            route,
        )
        (
            environment_context,
            environment_attention,
            environment_rows,
            _environment_group_count,
            environment_attention_aux,
        ) = (
            self._read_environment_grouped(
                state,
                encoded,
                receivers,
                route,
                return_attention=bool(return_routing_maps),
            )
        )
        aux: dict[str, torch.Tensor] = {
            "fixed_group_module_context_norm": torch.linalg.vector_norm(module_context, dim=-1),
            "fixed_group_environment_context_norm": torch.linalg.vector_norm(environment_context, dim=-1),
            "fixed_group_s_q": module_group_count.mean(dim=1),
            "fixed_group_s_q_per_query": module_group_count,
            "fixed_group_p_m": module_rows,
            "fixed_group_p_e": environment_rows,
            "fixed_group_p_m_total": module_rows.sum(dim=1),
            "fixed_group_p_e_total": environment_rows.sum(dim=1),
        }
        active_modules = (encoded.module_present > 0.5).sum(dim=-1).to(receivers.dtype)
        environment_count = receivers.new_tensor(float(encoded.env_coords.shape[1]))
        aux["fixed_group_r_m"] = module_rows / active_modules[:, None].clamp_min(1.0)
        aux["fixed_group_r_e"] = environment_rows / environment_count.clamp_min(1.0)
        if return_routing_maps:
            group_state: FixedGroupState = state["router_state"]
            aux.update(
                {
                    "fixed_group_query_assignment": route.assignment,
                    # Canonical evidence name plus the explicit assignment
                    # spelling retained for callers that prefer tensor terms.
                    "fixed_group_query_routing": route.assignment,
                    "fixed_group_query_alpha": route.assignment,
                    "fixed_group_query_logits": route.logits,
                    "fixed_group_environment_attention": environment_attention,
                    "fixed_group_module_incidence": group_state.module_membership,
                    "fixed_group_environment_incidence": group_state.environment_membership,
                    "fixed_group_module_centres": group_state.module_centres,
                    "fixed_group_environment_centres": group_state.environment_centres,
                }
            )
            aux.update(environment_attention_aux)
            aux.update(
                self._semantic_triples(
                    route.assignment,
                    group_state.module_membership,
                    prefix="fixed_group_module",
                )
            )
            aux.update(
                self._semantic_triples(
                    route.assignment,
                    group_state.environment_membership,
                    prefix="fixed_group_environment",
                )
            )
        return module_context + environment_context, aux


__all__ = ["FixedGroupPairwiseField"]
