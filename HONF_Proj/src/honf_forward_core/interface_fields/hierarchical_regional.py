"""Hierarchical regional response backend for NStage2 Track A.

The backend keeps Dense's fine MM/ME/EM preparation and all existing learned
modules.  It changes only the environmental response representation and its
receiver read: fine pre-update sufficient statistics are reduced on an
adapter-owned tree, and a smooth geometric opening rule produces a sparse
receiver--node incidence list.  Geometry and attention work is evaluated on
that gathered list, never on a dense query-by-all-node tensor.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from .dense_pairwise import DensePairwiseField
from .response_hierarchy import (
    EnvironmentHierarchy,
    HierarchyGeometry,
    HierarchyStatistics,
    reduce_preupdate_statistics,
)
from .types import EncodedInterfaceCase


class _Incidence:
    """Flat sparse receiver--node rows for one read."""

    __slots__ = ("batch", "query", "node", "eta", "traversal_rows")

    def __init__(
        self,
        batch: torch.Tensor,
        query: torch.Tensor,
        node: torch.Tensor,
        eta: torch.Tensor,
        traversal_rows: int,
    ) -> None:
        self.batch = batch
        self.query = query
        self.node = node
        self.eta = eta
        self.traversal_rows = int(traversal_rows)

    @property
    def rows(self) -> int:
        return int(self.node.numel())


class HierarchicalRegionalField(DensePairwiseField):
    """Dense fine responses with a smooth sparse hierarchical environmental read."""

    def __init__(
        self,
        hidden_dim: int,
        message_hidden_dim: int,
        num_heads: int,
        fourier_frequencies: int,
        *,
        response_region_block_shape: tuple[int, ...] = (2, 2),
        response_tree_opening_interval: tuple[float, float] = (1.0, 2.0),
        activation_checkpointing: bool = False,
    ) -> None:
        # No learned modules are added here.  This keeps parameter ownership
        # and the fine MM/ME/EM arithmetic identical to Dense/Regional.
        super().__init__(
            hidden_dim,
            message_hidden_dim,
            num_heads,
            fourier_frequencies,
            activation_checkpointing=activation_checkpointing,
        )
        self.response_region_block_shape = tuple(int(item) for item in response_region_block_shape)
        interval = tuple(float(item) for item in response_tree_opening_interval)
        if len(interval) != 2 or not 0.0 < interval[0] < interval[1]:
            raise ValueError("response_tree_opening_interval must contain two increasing positive radii.")
        self.response_tree_opening_interval = interval

    @staticmethod
    def _hierarchy_for_encoded(
        encoded: EncodedInterfaceCase,
        hierarchy: EnvironmentHierarchy | None,
    ) -> EnvironmentHierarchy:
        candidate = hierarchy if hierarchy is not None else getattr(encoded, "env_hierarchy", None)
        if candidate is None:
            raise ValueError(
                "hierarchical_regional_honf requires adapter-supplied encoded.env_hierarchy."
            )
        if not isinstance(candidate, EnvironmentHierarchy):
            raise TypeError("encoded.env_hierarchy must be an EnvironmentHierarchy.")
        batch = int(encoded.env_tokens.shape[0])
        candidate = candidate.to(device=encoded.env_tokens.device).expand_batch(batch)
        if candidate.bounds_min.dtype != encoded.env_coords.dtype:
            candidate = replace(
                candidate,
                bounds_min=candidate.bounds_min.to(dtype=encoded.env_coords.dtype),
                bounds_max=candidate.bounds_max.to(dtype=encoded.env_coords.dtype),
            )
        if int(candidate.fine_to_leaf.shape[-1]) != int(encoded.env_tokens.shape[1]):
            raise ValueError("Environment hierarchy fine membership must align with encoded environment tokens.")
        return candidate

    def _prepare_tree(
        self,
        encoded: EncodedInterfaceCase,
        fine: dict[str, torch.Tensor],
        hierarchy: EnvironmentHierarchy,
        geometry: HierarchyGeometry | None = None,
    ) -> tuple[HierarchyStatistics, torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = reduce_preupdate_statistics(
            hierarchy,
            fine["env_tokens"],
            fine["environment_messages"],
            encoded.env_weights,
            encoded.env_coords,
            geometry=geometry,
        )
        global_nodes = encoded.global_token[:, None, :].expand(-1, hierarchy.num_nodes, -1)
        tree_state = stats.environment.new_zeros(stats.environment.shape)
        valid_rows = torch.nonzero(stats.valid, as_tuple=False)
        if valid_rows.numel():
            valid_update = self.env_update(
                torch.cat(
                    [
                        stats.environment[valid_rows[:, 0], valid_rows[:, 1]],
                        stats.response[valid_rows[:, 0], valid_rows[:, 1]],
                        global_nodes[valid_rows[:, 0], valid_rows[:, 1]],
                    ],
                    dim=-1,
                )
            )
            tree_state = tree_state.index_put(
                (valid_rows[:, 0], valid_rows[:, 1]),
                stats.environment[valid_rows[:, 0], valid_rows[:, 1]] + valid_update,
            )
        # The projections are made once for this freshly prepared graph and
        # remain live through all receiver chunks in the same physical pass.
        tree_key, tree_value = self.project_environment_sources(tree_state)
        return stats, tree_state, tree_key, tree_value

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        hierarchy: EnvironmentHierarchy | None = None,
        hierarchy_geometry: HierarchyGeometry | None = None,
        return_routing_maps: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Prepare fine messages, all live hierarchy states, and projections."""

        del return_routing_maps
        fine = self.prepare_fine_messages(encoded, module_states)
        hierarchy = self._hierarchy_for_encoded(encoded, hierarchy)
        if hierarchy_geometry is None:
            hierarchy_geometry = getattr(encoded, "env_hierarchy_geometry", None)
        if hierarchy_geometry is not None and not isinstance(hierarchy_geometry, HierarchyGeometry):
            raise TypeError("encoded.env_hierarchy_geometry must be a HierarchyGeometry.")
        stats, tree_state, tree_key, tree_value = self._prepare_tree(
            encoded, fine, hierarchy, hierarchy_geometry
        )
        bounds_min = hierarchy.bounds_min
        bounds_max = hierarchy.bounds_max
        levels = hierarchy.level
        offsets = hierarchy.level_offsets
        children = hierarchy.children
        node_valid = hierarchy.node_valid
        if node_valid is None:
            node_valid = torch.ones(
                (module_states.shape[0], hierarchy.num_nodes),
                device=module_states.device,
                dtype=torch.bool,
            )
        else:
            node_valid = node_valid.to(device=module_states.device) > 0.5
        state = {
            # Keep the inherited fine module source available to the direct QM
            # read and diagnostics.
            "module_tokens": fine["module_tokens"],
            "env_tokens": fine["env_tokens"],
            "environment_messages": fine["environment_messages"],
            "tree_states": tree_state,
            "tree_coords": stats.coordinates,
            "tree_mass": stats.mass,
            "tree_valid": stats.valid,
            "tree_bounds_min": bounds_min,
            "tree_bounds_max": bounds_max,
            "tree_node_valid": node_valid,
            "tree_levels": levels,
            "tree_level_offsets": offsets,
            "tree_children": children,
            "tree_keys": tree_key,
            "tree_values": tree_value,
            # Regional aliases make the existing diagnostics reusable while
            # the new names describe the actual multiresolution representation.
            "regional_response_states": tree_state,
            "regional_coords": stats.coordinates,
            "regional_weights": stats.mass,
            "regional_valid": stats.valid,
            "regional_keys": tree_key,
            "regional_values": tree_value,
            "regional_levels": levels,
        }
        return state

    @staticmethod
    def _opening_blend(
        receivers: torch.Tensor,
        bounds_min: torch.Tensor,
        bounds_max: torch.Tensor,
        near_radius: float,
        far_radius: float,
    ) -> torch.Tensor:
        center = 0.5 * (bounds_min + bounds_max)
        radius = 0.5 * torch.linalg.vector_norm(bounds_max - bounds_min, dim=-1)
        radius = radius.clamp_min(torch.finfo(bounds_min.dtype).eps)
        relative = receivers - center
        normalized_squared_distance = relative.square().sum(dim=-1) / radius.square()
        near_squared = float(near_radius) ** 2
        far_squared = float(far_radius) ** 2
        transition = ((normalized_squared_distance - near_squared) / (far_squared - near_squared)).clamp(0.0, 1.0)
        # Smoothstep is C1 at both compact-support endpoints.  This
        # factorisation remains nonnegative near t=1 in low precision, where
        # ``1 - 3*t**2 + 2*t**3`` can suffer cancellation.
        one_minus = 1.0 - transition
        return one_minus.square() * (1.0 + 2.0 * transition)

    @staticmethod
    def _empty_incidence(device: torch.device, dtype: torch.dtype) -> _Incidence:
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        return _Incidence(empty_long, empty_long, empty_long, torch.empty(0, device=device, dtype=dtype), 0)

    def _fixed_level_incidence(
        self,
        state: dict[str, torch.Tensor],
        batch: int,
        query_count: int,
        level: int,
    ) -> _Incidence:
        levels = state["tree_levels"]
        valid = state["tree_valid"]
        selected = torch.nonzero((levels[None, :] == int(level)) & valid, as_tuple=False)
        if selected.numel() == 0:
            return self._empty_incidence(valid.device, state["tree_mass"].dtype)
        node_batch = selected[:, 0].repeat_interleave(query_count)
        node = selected[:, 1].repeat_interleave(query_count)
        query = torch.arange(query_count, device=valid.device, dtype=torch.long).repeat(selected.shape[0])
        eta = state["tree_mass"].new_ones(node.shape[0])
        del batch
        # Keep detailed routing rows grouped by receiver, matching the mixed
        # tree walk and making fixed-level diagnostics chunk-stable as well.
        order = torch.argsort(node_batch * query_count + query, stable=True)
        return _Incidence(
            node_batch[order],
            query[order],
            node[order],
            eta[order],
            int(node.numel()),
        )

    def _traverse(
        self,
        state: dict[str, torch.Tensor],
        receivers: torch.Tensor,
        *,
        resolution_level: int | str | None = None,
    ) -> _Incidence:
        batch, query_count, _ = receivers.shape
        if resolution_level is not None:
            if isinstance(resolution_level, str):
                label = resolution_level.lower()
                if label in {"leaf", "fine", "level0", "0"}:
                    resolution_level = 0
                elif label in {"level1", "regional", "first_parent", "parent"}:
                    resolution_level = 1
                elif label in {"tree", "hierarchical", "mixed", "auto"}:
                    resolution_level = None
                else:
                    raise ValueError(f"Unknown hierarchical read resolution {resolution_level!r}.")
            if resolution_level is not None:
                if int(resolution_level) < 0 or int(resolution_level) >= int(state["tree_level_offsets"].numel()) - 1:
                    raise ValueError("resolution_level is outside the supplied hierarchy.")
                return self._fixed_level_incidence(state, batch, query_count, int(resolution_level))

        device = receivers.device
        mass = state["tree_mass"]
        valid = state["tree_valid"]
        level = state["tree_levels"]
        max_level = int(state["tree_level_offsets"].numel()) - 2
        root_start = int(state["tree_level_offsets"][-2].item())
        root_count = int(state["tree_level_offsets"][-1].item()) - root_start
        if root_count != 1:
            raise ValueError("EnvironmentHierarchy must have exactly one root node.")
        root = torch.full((batch, query_count), root_start, device=device, dtype=torch.long)
        root_batch = torch.arange(batch, device=device, dtype=torch.long)[:, None].expand(-1, query_count).reshape(-1)
        root_query = torch.arange(query_count, device=device, dtype=torch.long)[None, :].expand(batch, -1).reshape(-1)
        current_batch = root_batch
        current_query = root_query
        current_node = root.reshape(-1)
        current_carrier = receivers.new_ones(current_node.shape[0])
        incidence_batch: list[torch.Tensor] = []
        incidence_query: list[torch.Tensor] = []
        incidence_node: list[torch.Tensor] = []
        incidence_eta: list[torch.Tensor] = []
        traversal_rows = 0

        for _level_number in range(max_level, -1, -1):
            if current_node.numel() == 0:
                break
            current_valid = valid[current_batch, current_node] & (current_carrier > 0)
            if not bool(current_valid.any()):
                break
            current_batch = current_batch[current_valid]
            current_query = current_query[current_valid]
            current_node = current_node[current_valid]
            current_carrier = current_carrier[current_valid]
            traversal_rows += int(current_node.numel())
            node_level = level[current_node]
            # Levels are numbered from fine leaves (0) to the root.  We walk
            # downward, so every positive level is an internal node.
            internal = node_level > 0
            chi = current_carrier.new_zeros(current_node.shape)
            if bool(internal.any()):
                internal_index = torch.nonzero(internal, as_tuple=False).flatten()
                node_bounds_min = state["tree_bounds_min"][current_batch[internal_index], current_node[internal_index]]
                node_bounds_max = state["tree_bounds_max"][current_batch[internal_index], current_node[internal_index]]
                chi_internal = self._opening_blend(
                    receivers[current_batch[internal_index], current_query[internal_index]],
                    node_bounds_min,
                    node_bounds_max,
                    self.response_tree_opening_interval[0],
                    self.response_tree_opening_interval[1],
                )
                chi[internal_index] = chi_internal
            eta = torch.where(internal, current_carrier * (1.0 - chi), current_carrier)
            keep = eta > 0
            if bool(keep.any()):
                incidence_batch.append(current_batch[keep])
                incidence_query.append(current_query[keep])
                incidence_node.append(current_node[keep])
                incidence_eta.append(eta[keep])

            if not bool(internal.any()):
                current_batch = current_batch.new_empty(0)
                break
            internal_index = torch.nonzero(internal, as_tuple=False).flatten()
            child_index = state["tree_children"][current_node[internal_index]]
            child_carrier = (current_carrier[internal_index, None] * chi[internal_index, None]).expand_as(child_index)
            child_batch = current_batch[internal_index, None].expand_as(child_index)
            child_query = current_query[internal_index, None].expand_as(child_index)
            child_valid = child_index >= 0
            safe_child = child_index.clamp_min(0)
            child_valid = child_valid & state["tree_valid"][child_batch, safe_child]
            child_valid = child_valid & (child_carrier > 0)
            if bool(child_valid.any()):
                current_batch = child_batch[child_valid]
                current_query = child_query[child_valid]
                current_node = child_index[child_valid]
                current_carrier = child_carrier[child_valid]
            else:
                current_batch = current_batch.new_empty(0)
                current_query = current_query.new_empty(0)
                current_node = current_node.new_empty(0)
                current_carrier = current_carrier.new_empty(0, dtype=receivers.dtype)

        if not incidence_node:
            return _Incidence(
                root_batch.new_empty(0),
                root_query.new_empty(0),
                root.new_empty(0),
                receivers.new_empty(0),
                traversal_rows,
            )
        batch_index = torch.cat(incidence_batch)
        query_index = torch.cat(incidence_query)
        node_index = torch.cat(incidence_node)
        eta = torch.cat(incidence_eta)
        # Group rows by receiver so the optional detailed export is stable
        # across chunks.  Sorting only changes row order; all scalar eta
        # values remain live in the graph.
        order = torch.argsort(batch_index * query_count + query_index, stable=True)
        return _Incidence(
            batch_index[order],
            query_index[order],
            node_index[order],
            eta[order],
            traversal_rows,
        )

    @staticmethod
    def _coordinate_scale_rows(scale: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Select one live coordinate scale per gathered receiver row."""

        if scale.ndim == 1:
            return scale.reshape(1, -1).expand(batch_index.shape[0], -1)
        if scale.ndim == 2:
            if int(scale.shape[0]) == 1:
                return scale[0].reshape(1, -1).expand(batch_index.shape[0], -1)
            return scale[batch_index]
        if scale.ndim == 3:
            if int(scale.shape[0]) == 1:
                return scale[0, 0].reshape(1, -1).expand(batch_index.shape[0], -1)
            return scale[batch_index, 0]
        raise ValueError("coordinate_scale must have one, two, or three dimensions.")

    def _segmented_read(
        self,
        state: dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        incidence: _Incidence,
        *,
        return_routing_maps: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, query_count, _ = receivers.shape
        if incidence.rows == 0:
            return receivers.new_zeros(batch, query_count, self.hidden_dim), None

        env_query = self.env_query(receiver_features)
        projected_query = self.env_attention.project_query(env_query)
        projected_key = state["tree_keys"]
        projected_value = state["tree_values"]
        query_rows = projected_query[incidence.batch, :, incidence.query, :]
        key_rows = projected_key[incidence.batch, :, incidence.node, :]
        value_rows = projected_value[incidence.batch, :, incidence.node, :]
        scores = (query_rows * key_rows).sum(dim=-1) / (float(self.env_attention.head_dim) ** 0.5)
        # All geometry work starts from gathered source rows.  In particular,
        # there is no dense Q x N call into the Fourier or geometry MLP.
        source_coords = state["tree_coords"][incidence.batch, incidence.node]
        source_relative = (
            receivers[incidence.batch, incidence.query] - source_coords
        ) / self._coordinate_scale_rows(encoded.coordinate_scale, incidence.batch)
        geometry_bias = self._mlp(
            self.env_geometry_bias,
            self.relative_fourier(source_relative),
        )
        source_mass = state["tree_mass"][incidence.batch, incidence.node]
        # Keep the geometric measure in float64 for the log-space reduction.
        # This avoids turning a small but positive eta into an FP32 zero while
        # retaining the model dtype for the gathered value reduction.
        log_measure = torch.log(incidence.eta.to(torch.float64)) + torch.log(source_mass.to(torch.float64))
        logits = scores.to(torch.float64) + geometry_bias.to(torch.float64) + log_measure[:, None]
        segments = incidence.batch * query_count + incidence.query
        segment_count = batch * query_count
        segment_index = segments[:, None].expand(-1, self.env_attention.num_heads)
        max_logit = logits.new_full((segment_count, self.env_attention.num_heads), float("-inf"))
        max_logit.scatter_reduce_(0, segment_index, logits, reduce="amax", include_self=True)
        unnormalized = torch.exp(logits - max_logit[segments])
        denominator = logits.new_zeros(segment_count, self.env_attention.num_heads)
        denominator.scatter_add_(0, segment_index, unnormalized)
        normalized = unnormalized / denominator[segments].clamp_min(torch.finfo(logits.dtype).tiny)
        normalized_values = normalized.to(dtype=value_rows.dtype)
        weighted_value = normalized_values[..., None] * value_rows
        aggregated = value_rows.new_zeros(segment_count, self.env_attention.num_heads, self.env_attention.head_dim)
        value_index = segments[:, None, None].expand_as(weighted_value)
        aggregated.scatter_add_(0, value_index, weighted_value)
        aggregated = aggregated.reshape(batch, query_count, self.hidden_dim)
        result = self.env_attention.output(aggregated)
        present = denominator.sum(dim=1).reshape(batch, query_count) > 0
        result = result.masked_fill(~present[..., None], 0.0)
        return result, normalized if return_routing_maps else None

    def _resolve_read_level(self, resolution_level: int | str | None, read_level: int | str | None) -> int | str | None:
        if resolution_level is not None and read_level is not None:
            raise ValueError("Pass only one of resolution_level and read_level.")
        return resolution_level if resolution_level is not None else read_level

    def _environment_aux(
        self,
        state: dict[str, torch.Tensor],
        receivers: torch.Tensor,
        incidence: _Incidence,
        sparse_attention: torch.Tensor | None,
        *,
        return_routing_maps: bool,
    ) -> dict[str, torch.Tensor]:
        batch, query_count, _ = receivers.shape
        receiver_segments = incidence.batch * query_count + incidence.query
        selected_counts = receivers.new_zeros(batch * query_count)
        if incidence.rows:
            selected_counts.index_add_(0, receiver_segments, receivers.new_ones(incidence.rows))
        selected_counts = selected_counts.reshape(batch, query_count)
        aux: dict[str, torch.Tensor] = {
            "hierarchical_selected_count": selected_counts,
            "regional_selected_count": selected_counts,
            "hierarchical_traversal_rows": receivers.new_tensor(float(incidence.traversal_rows)),
            "hierarchical_incidence_rows": receivers.new_tensor(float(incidence.rows)),
        }
        if return_routing_maps:
            aux.update(
                {
                    "hierarchical_incidence_batch": incidence.batch,
                    "hierarchical_incidence_query": incidence.query,
                    "hierarchical_incidence_node": incidence.node,
                    "hierarchical_incidence_eta": incidence.eta,
                }
            )
            if sparse_attention is not None:
                aux["hierarchical_incidence_attention"] = sparse_attention
        return aux

    def read_environment(
        self,
        state: dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
        resolution_level: int | str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Read only the hierarchical environmental route.

        The physical/core read calls this canonical hook, which also makes
        diagnostics able to instrument or replace the environmental route
        without accidentally bypassing the actual forward path.
        """

        incidence = self._traverse(state, receivers, resolution_level=resolution_level)
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            # Keep traversal and its live carrier ``eta`` outside the
            # checkpoint: it is discrete/ragged metadata with a small graph.
            # Recompute the gathered query/key/value, geometry MLP, segmented
            # softmax, and output projection during backward.  The closure
            # deliberately calls the concrete hook so instrumentation and
            # subclasses still observe the real read implementation.
            def recompute_read(
                read_receivers: torch.Tensor,
                read_features: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor | None]:
                return self._segmented_read(
                    state,
                    encoded,
                    read_receivers,
                    read_features,
                    incidence,
                    return_routing_maps=bool(return_routing_maps),
                )

            context, sparse_attention = checkpoint(
                recompute_read,
                receivers,
                receiver_features,
                use_reentrant=False,
            )
        else:
            context, sparse_attention = self._segmented_read(
                state,
                encoded,
                receivers,
                receiver_features,
                incidence,
                return_routing_maps=bool(return_routing_maps),
            )
        aux = self._environment_aux(
            state,
            receivers,
            incidence,
            sparse_attention,
            return_routing_maps=bool(return_routing_maps),
        )
        aux["hierarchical_context_norm"] = torch.linalg.vector_norm(context, dim=-1)
        aux["regional_context_norm"] = aux["hierarchical_context_norm"]
        return context, aux

    def read(
        self,
        state: dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
        resolution_level: int | str | None = None,
        read_level: int | str | None = None,
        fixed_level: int | str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Read direct QM plus hierarchical environmental context.

        ``resolution_level=0``/``"leaf"`` gives the Dense fine limit and
        ``resolution_level=1``/``"level1"`` gives the fixed first-parent
        Regional limit.  The default uses the smooth mixed-resolution walk.
        """

        selected_level = self._resolve_read_level(resolution_level, read_level)
        if fixed_level is not None:
            if selected_level is not None:
                raise ValueError("Pass only one of fixed_level and resolution_level/read_level.")
            selected_level = fixed_level
        module_context = self.read_module(state, encoded, receivers, receiver_features)
        tree_context, environment_aux = self.read_environment(
            state,
            encoded,
            receivers,
            receiver_features,
            return_routing_maps=bool(return_routing_maps),
            resolution_level=selected_level,
        )
        response_count = state["tree_valid"].sum(dim=1)
        aux: dict[str, torch.Tensor] = {
            "dense_module_context_norm": torch.linalg.vector_norm(module_context, dim=-1),
            "hierarchical_context_norm": torch.linalg.vector_norm(tree_context, dim=-1),
            "regional_context_norm": torch.linalg.vector_norm(tree_context, dim=-1),
            "hierarchical_response_count": response_count,
            "regional_response_count": response_count,
        }
        aux.update(environment_aux)
        return module_context + tree_context, aux

    def read_regional(
        self,
        state: dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
        resolution_level: int | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read only the hierarchical environmental route."""
        context, aux = self.read_environment(
            state,
            encoded,
            receivers,
            receiver_features,
            return_routing_maps=bool(return_routing_maps),
            resolution_level=resolution_level,
        )
        return context, aux.get("hierarchical_incidence_attention")

    def read_fixed_level(
        self,
        state: dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        level: int,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Diagnostic helper for the Dense/Regional limiting resolutions."""

        return self.read_regional(
            state,
            encoded,
            receivers,
            receiver_features,
            return_routing_maps=bool(return_routing_maps),
            resolution_level=int(level),
        )

    def preparation_aux(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Small scalar summaries suitable for the core preparation record."""

        valid = state["tree_valid"]
        per_level = []
        levels = state["tree_levels"]
        for level in range(int(state["tree_level_offsets"].numel()) - 1):
            per_level.append((valid & (levels[None, :] == level)).sum(dim=1))
        return {
            "hierarchical_response_count": valid.sum(dim=1),
            "hierarchical_response_count_by_level": torch.stack(per_level, dim=1),
        }


__all__ = ["HierarchicalRegionalField"]
