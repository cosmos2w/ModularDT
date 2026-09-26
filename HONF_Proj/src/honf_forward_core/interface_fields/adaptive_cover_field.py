"""Opt-in physical-pair reader for a case-local interaction cover.

The inherited routed class supplies only its tested packed fine kernels. Its
fixed candidate router is never constructed or used. Full-access mode calls
the dense reader itself, making the parent comparison exact at the field
boundary. An externally supplied cover is an offline oracle diagnostic until
an input-only organizer has passed its own evidence gate.
"""

from __future__ import annotations

from typing import Any

import torch

from .adaptive_interaction_cover import (
    AdaptiveCoverPlan,
    CaseLocalReceiverTree,
    CoverPairLedger,
    ReceiverAnchorUniverse,
    compile_cover_pairs,
)
from .dense_pairwise import DensePairwiseField
from .routed_pairwise import RoutedPairwiseField
from .routing_index.types import PackedPairs
from .types import EncodedInterfaceCase


class AdaptiveCoverPairwiseField(RoutedPairwiseField):
    """Dense MM/ME/EM preparation and deduplicated QM/QE cover execution."""

    def __init__(
        self,
        hidden_dim: int,
        message_hidden_dim: int,
        num_heads: int,
        fourier_frequencies: int,
        *,
        activation_checkpointing: bool = False,
        max_nodes: int = 127,
        min_leaf_anchors: int = 4,
        overlap_fraction: float = 0.06,
        fine_pair_chunk_size: int = 16384,
    ) -> None:
        # Deliberately bypass RoutedPairwiseField.__init__: that constructor
        # registers the older fixed/module-hub router and its parameters.
        DensePairwiseField.__init__(
            self, hidden_dim, message_hidden_dim, num_heads, fourier_frequencies,
            activation_checkpointing=activation_checkpointing,
        )
        if max_nodes < 1 or min_leaf_anchors < 1 or fine_pair_chunk_size < 1:
            raise ValueError("cover capacity, leaf size, and pair tile must be positive")
        if not 0.0 < overlap_fraction < 1.0:
            raise ValueError("cover overlap must be in (0,1)")
        self.max_nodes = max_nodes
        self.min_leaf_anchors = min_leaf_anchors
        self.overlap_fraction = overlap_fraction
        self.fine_pair_chunk_size = fine_pair_chunk_size
        self.qe_backend = "torch"
        self.dense_environment_fast_path = False
        self.last_qe_backend = "torch"
        self.last_qe_backend_reason = "not_run"
        self.cover_mode = "full_access"

    def set_cover_mode(self, mode: str) -> None:
        if mode not in {"full_access", "external"}:
            raise ValueError("cover mode must be full_access or external")
        self.cover_mode = mode

    @staticmethod
    def _case_scale(encoded: EncodedInterfaceCase, case: int) -> torch.Tensor:
        scale = encoded.coordinate_scale
        if scale.ndim == 1:
            return scale
        if scale.ndim == 2:
            return scale[min(case, int(scale.shape[0]) - 1)]
        if scale.ndim == 3 and int(scale.shape[1]) == 1:
            return scale[min(case, int(scale.shape[0]) - 1), 0]
        raise ValueError("coordinate scale must provide one physical vector per case")

    def _default_universe(self, encoded: EncodedInterfaceCase, case: int) -> ReceiverAnchorUniverse:
        if encoded.receiver_anchor_coords is not None:
            assert encoded.receiver_anchor_weights is not None and encoded.receiver_anchor_roles is not None
            valid = encoded.receiver_anchor_weights[case] > 0
            return ReceiverAnchorUniverse(
                encoded.receiver_anchor_coords[case, valid],
                encoded.receiver_anchor_weights[case, valid],
                encoded.receiver_anchor_roles[case, valid],
                self._case_scale(encoded, case),
            )
        # The environmental quadrature is a reproducible geometry-only
        # receiver support. Module-attached locations ensure a case changes
        # when physical modules move. Case adapters may supply additional
        # functional locations with their own role codes and measures.
        valid = encoded.module_present[case] > 0.5
        coords = torch.cat((encoded.env_coords[case], encoded.module_centers[case, valid]), dim=0)
        weights = torch.cat((
            encoded.env_weights[case],
            encoded.env_weights.new_ones((int(valid.sum()),)),
        ))
        roles = torch.cat((
            torch.zeros(int(encoded.env_coords.shape[1]), device=coords.device, dtype=torch.long),
            torch.ones(int(valid.sum()), device=coords.device, dtype=torch.long),
        ))
        return ReceiverAnchorUniverse(coords, weights, roles, self._case_scale(encoded, case))

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        state = DensePairwiseField.prepare(
            self, encoded, module_states, return_routing_maps=return_routing_maps
        )
        trees = tuple(
            CaseLocalReceiverTree.build(
                self._default_universe(encoded, case),
                max_nodes=self.max_nodes,
                min_leaf_anchors=self.min_leaf_anchors,
                overlap_fraction=self.overlap_fraction,
            )
            for case in range(int(module_states.shape[0]))
        )
        state["cover_trees"] = trees
        state["cover_plans"] = tuple(
            AdaptiveCoverPlan.full_access(tree, encoded.module_present[case], int(encoded.env_coords.shape[1]))
            for case, tree in enumerate(trees)
        )
        batch, padded_modules = encoded.module_present.shape
        environment_count = int(encoded.env_coords.shape[1])
        active_modules = (encoded.module_present > 0.5).sum(dim=1)
        mm_executed = int(batch) * int(padded_modules) ** 2
        me_executed = int(batch) * int(padded_modules) * environment_count
        mm_valid = int((active_modules * (active_modules - 1)).sum())
        me_valid = int(active_modules.sum()) * environment_count
        # Dense's simultaneous MM, ME, and EM kernels run rectangular
        # tensors before masks are applied. Report both the actual evaluated
        # rows and the physically valid rows so query-side savings cannot be
        # mistaken for preparation-side sparsity.
        state["cover_preparation_ledger"] = {
            "cover_prepare_mm_executed_rows": mm_executed,
            "cover_prepare_me_executed_rows": me_executed,
            "cover_prepare_em_executed_rows": me_executed,
            "cover_prepare_mm_valid_rows": mm_valid,
            "cover_prepare_me_valid_rows": me_valid,
            "cover_prepare_em_valid_rows": me_valid,
            "cover_prepare_padded_or_self_rows": mm_executed + 2 * me_executed - mm_valid - 2 * me_valid,
        }
        return state

    def preparation_aux(
        self, state: dict[str, Any], *, include_diagnostics: bool = False
    ) -> dict[str, torch.Tensor]:
        del include_diagnostics
        reference = state["module_tokens"]
        return {
            name: torch.tensor(count, device=reference.device, dtype=torch.long)
            for name, count in state["cover_preparation_ledger"].items()
        }

    @staticmethod
    def with_external_plans(state: dict[str, Any], plans: tuple[AdaptiveCoverPlan, ...]) -> dict[str, Any]:
        if len(plans) != len(state["cover_trees"]):
            raise ValueError("one cover plan is required per case")
        for tree, plan in zip(state["cover_trees"], plans, strict=True):
            if plan.tree is not tree:
                raise ValueError("external plan must use the prepared case-local receiver tree")
        return {**state, "cover_plans": plans}

    @staticmethod
    def _combine_pairs(items: list[PackedPairs], case_count: int) -> PackedPairs:
        if len(items) != case_count:
            raise ValueError("one packed pair list is required per case")
        return PackedPairs(
            torch.cat([torch.full_like(item.batch_index, case) for case, item in enumerate(items)]),
            torch.cat([item.receiver_index for item in items]),
            torch.cat([item.source_index for item in items]),
            torch.cat([item.prior for item in items]),
            sum(item.raw_path_count for item in items),
            sum(item.unique_pair_count for item in items),
        )

    def _module_tile(
        self, sources: torch.Tensor, centers: torch.Tensor, receivers: torch.Tensor,
        global_token: torch.Tensor, scale: torch.Tensor, batch_index: torch.Tensor,
        receiver_index: torch.Tensor, source_index: torch.Tensor, prior: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_count = receivers.shape[:2]
        scale_rows = self._environment_scale_rows(scale, batch)[:, 0, :]
        relative = (
            receivers[batch_index, receiver_index] - centers[batch_index, source_index]
        ) / scale_rows[batch_index]
        messages = self.query_module_message(torch.cat((
            sources[batch_index, source_index], self.relative_fourier(relative), global_token[batch_index],
        ), dim=-1))
        reduced = sources.new_zeros((batch * query_count, sources.shape[-1]))
        reduced.index_add_(
            0, batch_index * query_count + receiver_index,
            messages * prior.to(messages.dtype)[:, None],
        )
        return reduced.reshape(batch, query_count, -1)

    def _environment_score_tile(
        self, query: torch.Tensor, key: torch.Tensor, receivers: torch.Tensor,
        coordinates: torch.Tensor, scale: torch.Tensor, batch_index: torch.Tensor,
        receiver_index: torch.Tensor, source_index: torch.Tensor,
    ) -> torch.Tensor:
        scale_rows = self._environment_scale_rows(scale, int(receivers.shape[0]))[:, 0, :]
        relative = (
            receivers[batch_index, receiver_index] - coordinates[batch_index, source_index]
        ) / scale_rows[batch_index]
        bias = self.env_geometry_bias(self.relative_fourier(relative))
        selected_query = query[batch_index, :, receiver_index, :]
        selected_key = key[batch_index, :, source_index, :]
        return (selected_query * selected_key).sum(-1) / (float(self.env_attention.head_dim) ** 0.5) + bias

    def read(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.cover_mode == "full_access":
            context, aux = DensePairwiseField.read(
                self, state, encoded, receivers, receiver_features,
                return_routing_maps=return_routing_maps,
            )
            query_count = int(receivers.shape[1])
            module_rows = int((encoded.module_present > 0.5).sum()) * query_count
            module_executed_rows = int(encoded.module_present.numel()) * query_count
            environment_rows = int(encoded.env_coords.shape[1]) * query_count * int(receivers.shape[0])
            aux.update({
                "cover_k_case": torch.ones(int(receivers.shape[0]), device=receivers.device, dtype=torch.long),
                "cover_transition_case": torch.zeros(int(receivers.shape[0]), device=receivers.device, dtype=torch.long),
                "cover_candidate_nodes": torch.tensor(
                    [len(tree.nodes) for tree in state["cover_trees"]], device=receivers.device
                ),
                "cover_capacity_saturated": torch.tensor(
                    [tree.capacity_saturated for tree in state["cover_trees"]], device=receivers.device
                ),
                "cover_qm_unique_rows": torch.tensor(module_rows, device=receivers.device),
                "cover_qe_unique_rows": torch.tensor(environment_rows, device=receivers.device),
                "cover_qm_executed_rows": torch.tensor(module_executed_rows, device=receivers.device),
                "cover_qe_executed_rows": torch.tensor(environment_rows, device=receivers.device),
                "cover_qm_padded_rows": torch.tensor(module_executed_rows - module_rows, device=receivers.device),
                "cover_qm_raw_paths": torch.tensor(module_rows, device=receivers.device),
                "cover_qe_raw_paths": torch.tensor(environment_rows, device=receivers.device),
                "cover_qm_rectangular_rows": torch.tensor(module_executed_rows, device=receivers.device),
                "cover_qe_rectangular_rows": torch.tensor(environment_rows, device=receivers.device),
                "cover_environment_fallback_queries": torch.tensor(0, device=receivers.device),
                "cover_query_degree_sum": torch.tensor(query_count * int(receivers.shape[0]), device=receivers.device),
                "cover_query_degree_max": torch.tensor(1, device=receivers.device),
                "cover_query_count": torch.tensor(query_count * int(receivers.shape[0]), device=receivers.device),
                "cover_executor_dense_fallback": torch.tensor(1, device=receivers.device),
            })
            return context, aux
        plans = state["cover_plans"]
        if len(plans) != int(receivers.shape[0]):
            raise ValueError("cover plans must align with the case batch")
        module_pairs: list[PackedPairs] = []
        environment_pairs: list[PackedPairs] = []
        ledgers: list[CoverPairLedger] = []
        k_case: list[int] = []
        transition_case: list[int] = []
        for case, plan in enumerate(plans):
            qm, qe, ledger = compile_cover_pairs(
                plan, receivers[case], module_present=encoded.module_present[case],
                environment_weights=encoded.env_weights[case],
            )
            module_pairs.append(qm)
            environment_pairs.append(qe)
            ledgers.append(ledger)
            access = plan.access(plan.tree.universe.coordinates)
            k_case.append(access.active_groups_on_anchors)
            transition_case.append(access.transition_nodes)
        qm = self._combine_pairs(module_pairs, len(plans))
        qe = self._combine_pairs(environment_pairs, len(plans))
        module_context = self.read_module_pairs(state, encoded, receivers, qm)
        environment_context = self.read_environment_pairs(
            state, encoded, receivers, receiver_features, qe
        )
        aux = {
            "cover_k_case": torch.tensor(k_case, device=receivers.device, dtype=torch.long),
            "cover_transition_case": torch.tensor(transition_case, device=receivers.device, dtype=torch.long),
            "cover_candidate_nodes": torch.tensor(
                [len(plan.tree.nodes) for plan in plans], device=receivers.device
            ),
            "cover_capacity_saturated": torch.tensor(
                [plan.tree.capacity_saturated for plan in plans], device=receivers.device
            ),
            "cover_qm_unique_rows": torch.tensor(sum(item.qm_unique_rows for item in ledgers), device=receivers.device),
            "cover_qe_unique_rows": torch.tensor(sum(item.qe_unique_rows for item in ledgers), device=receivers.device),
            "cover_qm_executed_rows": torch.tensor(qm.unique_pair_count, device=receivers.device),
            "cover_qe_executed_rows": torch.tensor(qe.unique_pair_count, device=receivers.device),
            "cover_qm_padded_rows": torch.tensor(0, device=receivers.device),
            "cover_qm_raw_paths": torch.tensor(sum(item.qm_raw_paths for item in ledgers), device=receivers.device),
            "cover_qe_raw_paths": torch.tensor(sum(item.qe_raw_paths for item in ledgers), device=receivers.device),
            "cover_qm_rectangular_rows": torch.tensor(
                sum(item.qm_rectangular_rows for item in ledgers), device=receivers.device
            ),
            "cover_qe_rectangular_rows": torch.tensor(
                sum(item.qe_rectangular_rows for item in ledgers), device=receivers.device
            ),
            "cover_environment_fallback_queries": torch.tensor(
                sum(item.environment_fallback_queries for item in ledgers), device=receivers.device
            ),
            "cover_query_degree_sum": torch.tensor(
                sum(item.query_degree_sum for item in ledgers), device=receivers.device
            ),
            "cover_query_degree_max": torch.tensor(
                max(item.query_degree_max for item in ledgers), device=receivers.device
            ),
            "cover_query_count": torch.tensor(
                sum(item.query_count for item in ledgers), device=receivers.device
            ),
            "cover_executor_dense_fallback": torch.tensor(0, device=receivers.device),
        }
        return module_context + environment_context, aux


__all__ = ["AdaptiveCoverPairwiseField"]
