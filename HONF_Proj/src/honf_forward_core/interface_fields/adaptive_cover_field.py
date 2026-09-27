"""Optional native execution policy backed by case-local interaction covers.

The inherited routed class supplies its tested packed QM/QE readers. Its
fixed candidate router is never constructed or used. The native Dense path
remains direct when no policy is attached, and full access calls the Dense
preparation and reader themselves. Partial plans mask the typed MM/ME/EM
preparation before source reduction and then use the packed QM/QE readers.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import torch

from .adaptive_interaction_cover import (
    AdaptiveCoverPlan,
    CaseLocalReceiverTree,
    CoverPairLedger,
    ReceiverAnchorUniverse,
    compile_cover_pairs,
    compile_cover_transport_pairs,
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
        optional_native_policy: bool = False,
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
        self.optional_native_policy = bool(optional_native_policy)
        self.cover_mode = "disabled" if self.optional_native_policy else "full_access"
        # Candidate searches commonly prepare many plans for one unchanged
        # case geometry. Keep only these immutable geometry indexes, with a
        # bounded per-backend lifetime; encoded states and plan tensors are
        # always recomputed by the ordinary prepare path.
        self.case_tree_cache_capacity = 64
        self._case_tree_cache: OrderedDict[bytes, CaseLocalReceiverTree] = OrderedDict()
        self._case_tree_cache_hits = 0
        self._case_tree_cache_misses = 0
        self._case_tree_cache_evictions = 0
        self._last_case_tree_cache_hits: tuple[bool, ...] = ()

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

    @staticmethod
    def _digest_tree_input(digest: Any, name: str, tensor: torch.Tensor | None) -> None:
        digest.update(name.encode("utf-8"))
        if tensor is None:
            digest.update(b"<none>")
            return
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tensor.device).encode("ascii"))
        value = tensor.detach().contiguous()
        if value.device.type != "cpu":
            value = value.cpu()
        # Hash raw bytes so dtypes such as bfloat16 are supported as well.
        digest.update(value.view(torch.uint8).numpy().tobytes())

    def _case_tree_cache_key(self, encoded: EncodedInterfaceCase, case: int) -> bytes:
        digest = hashlib.blake2b(digest_size=32, person=b"honf-tree-cache")
        digest.update(
            repr((self.max_nodes, self.min_leaf_anchors, self.overlap_fraction)).encode("ascii")
        )
        for name in (
            "receiver_anchor_coords",
            "receiver_anchor_weights",
            "receiver_anchor_roles",
            "module_centers",
            "module_present",
            "env_coords",
            "env_weights",
        ):
            value = getattr(encoded, name)
            self._digest_tree_input(digest, name, None if value is None else value[case])
        self._digest_tree_input(digest, "coordinate_scale", self._case_scale(encoded, case))
        return digest.digest()

    @staticmethod
    def _snapshot_universe(universe: ReceiverAnchorUniverse) -> ReceiverAnchorUniverse:
        """Keep detached geometry only; never retain encoder/model graphs."""

        return ReceiverAnchorUniverse(
            coordinates=universe.coordinates.detach().clone(),
            weights=universe.weights.detach().clone(),
            roles=universe.roles.detach().clone(),
            coordinate_scale=universe.coordinate_scale.detach().clone(),
        )

    @property
    def last_case_tree_cache_hits(self) -> tuple[bool, ...]:
        """Per-case hit flags for the most recent geometry-tree request."""

        return self._last_case_tree_cache_hits

    def case_tree_cache_info(self) -> dict[str, int]:
        return {
            "size": len(self._case_tree_cache),
            "capacity": int(self.case_tree_cache_capacity),
            "hits": self._case_tree_cache_hits,
            "misses": self._case_tree_cache_misses,
            "evictions": self._case_tree_cache_evictions,
        }

    def clear_case_tree_cache(self) -> None:
        """Drop cached geometry indexes and reset their session counters."""

        self._case_tree_cache.clear()
        self._case_tree_cache_hits = 0
        self._case_tree_cache_misses = 0
        self._case_tree_cache_evictions = 0
        self._last_case_tree_cache_hits = ()

    def build_case_trees(self, encoded: EncodedInterfaceCase) -> tuple[CaseLocalReceiverTree, ...]:
        """Reuse exact immutable geometry indexes across repeated prepares.

        The cache key hashes all physical design and receiver-measure inputs,
        plus tensor device/dtype and tree settings. A geometry change builds a
        new tree; model states, plans, and messages are never cached here.
        """

        if self.case_tree_cache_capacity < 0:
            raise ValueError("case tree cache capacity cannot be negative")
        trees: list[CaseLocalReceiverTree] = []
        hit_flags: list[bool] = []
        for case in range(int(encoded.module_present.shape[0])):
            tree_inputs = (
                encoded.receiver_anchor_coords,
                encoded.receiver_anchor_weights,
                encoded.receiver_anchor_roles,
                encoded.module_centers,
                encoded.module_present,
                encoded.env_coords,
                encoded.env_weights,
                self._case_scale(encoded, case),
            )
            if any(value is not None and value.requires_grad for value in tree_inputs):
                # Reusing a geometry snapshot would disconnect the current
                # input graph. Preserve the uncached behavior for
                # differentiable design tensors.
                self._case_tree_cache_misses += 1
                hit_flags.append(False)
                trees.append(CaseLocalReceiverTree.build(
                    self._default_universe(encoded, case),
                    max_nodes=self.max_nodes,
                    min_leaf_anchors=self.min_leaf_anchors,
                    overlap_fraction=self.overlap_fraction,
                ))
                continue
            key = self._case_tree_cache_key(encoded, case)
            tree = self._case_tree_cache.get(key)
            if tree is not None:
                self._case_tree_cache.move_to_end(key)
                self._case_tree_cache_hits += 1
                hit_flags.append(True)
                trees.append(tree)
                continue

            self._case_tree_cache_misses += 1
            hit_flags.append(False)
            universe = self._snapshot_universe(self._default_universe(encoded, case))
            tree = CaseLocalReceiverTree.build(
                universe,
                max_nodes=self.max_nodes,
                min_leaf_anchors=self.min_leaf_anchors,
                overlap_fraction=self.overlap_fraction,
            )
            if self.case_tree_cache_capacity > 0:
                self._case_tree_cache[key] = tree
                self._case_tree_cache.move_to_end(key)
                while len(self._case_tree_cache) > self.case_tree_cache_capacity:
                    self._case_tree_cache.popitem(last=False)
                    self._case_tree_cache_evictions += 1
            trees.append(tree)
        self._last_case_tree_cache_hits = tuple(hit_flags)
        return tuple(trees)

    @staticmethod
    def is_full_access_plan(plan: AdaptiveCoverPlan, module_present: torch.Tensor, environment_count: int) -> bool:
        """Recognize a root-only all-source plan without evaluating queries."""

        if len(plan.tree.nodes) == 0 or float(plan.split_gates[0].detach()) != 0.0:
            return False
        expected_module = (module_present > 0.5).to(plan.module_membership.dtype)
        expected_environment = plan.environment_membership.new_ones((environment_count,))
        return bool(
            torch.equal(plan.module_membership[0], expected_module)
            and torch.equal(plan.environment_membership[0], expected_environment)
        )

    @staticmethod
    def _validate_plans(
        encoded: EncodedInterfaceCase,
        trees: tuple[CaseLocalReceiverTree, ...],
        plans: tuple[AdaptiveCoverPlan, ...],
    ) -> None:
        batch = int(encoded.module_present.shape[0])
        if len(trees) != batch or len(plans) != batch:
            raise ValueError("native interaction policy must return one tree and plan per case")
        module_count = int(encoded.module_present.shape[1])
        environment_count = int(encoded.env_coords.shape[1])
        for case, (tree, plan) in enumerate(zip(trees, plans, strict=True)):
            if plan.tree is not tree:
                raise ValueError("native interaction plans must use the supplied case-local trees")
            if plan.module_membership.shape[1] != module_count:
                raise ValueError("native interaction module memberships do not match the padded module count")
            if plan.environment_membership.shape[1] != environment_count:
                raise ValueError("native interaction environment memberships do not match the encoded quadrature")
            if plan.split_gates.device != encoded.module_present.device:
                raise ValueError(f"native interaction plan for case {case} is on the wrong device")

    @staticmethod
    def _case_pairs(
        plans: tuple[AdaptiveCoverPlan, ...],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        *,
        source_role: str,
        receiver_valid: torch.Tensor | None = None,
        exclude_self: bool = False,
    ) -> PackedPairs:
        pairs = [
            compile_cover_transport_pairs(
                plan,
                receivers[case],
                source_role=source_role,
                module_present=encoded.module_present[case],
                environment_weights=encoded.env_weights[case],
                receiver_valid=None if receiver_valid is None else receiver_valid[case],
                exclude_self=exclude_self,
            )
            for case, plan in enumerate(plans)
        ]
        return AdaptiveCoverPairwiseField._combine_pairs(pairs, len(plans))

    def _prepare_message_pairs(
        self,
        network: torch.nn.Module,
        receiver_states: torch.Tensor,
        source_states: torch.Tensor,
        receiver_coords: torch.Tensor,
        source_coords: torch.Tensor,
        scale: torch.Tensor,
        pairs: PackedPairs,
        *,
        query_count: int,
    ) -> torch.Tensor:
        batch = int(receiver_states.shape[0])
        hidden = int(receiver_states.shape[-1])
        result = receiver_states.new_zeros((batch * query_count, hidden))
        scale_rows = self._environment_scale_rows(scale, batch)[:, 0, :]
        for start in range(0, pairs.unique_pair_count, self.fine_pair_chunk_size):
            end = min(start + self.fine_pair_chunk_size, pairs.unique_pair_count)
            batch_index = pairs.batch_index[start:end]
            receiver_index = pairs.receiver_index[start:end]
            source_index = pairs.source_index[start:end]
            relative = (
                receiver_coords[batch_index, receiver_index]
                - source_coords[batch_index, source_index]
            ) / scale_rows[batch_index]
            message_input = torch.cat((
                receiver_states[batch_index, receiver_index],
                source_states[batch_index, source_index],
                self.relative_fourier(relative),
            ), dim=-1)
            messages = self._mlp(network, message_input)
            row = batch_index * query_count + receiver_index
            result.index_add_(
                0,
                row,
                messages * pairs.prior[start:end].to(messages.dtype)[:, None],
            )
        return result.reshape(batch, query_count, hidden)

    @staticmethod
    def _preparation_ledger(
        encoded: EncodedInterfaceCase,
        *,
        mm_executed: int,
        me_executed: int,
        em_executed: int,
        dense_fallback: bool,
        tree_cache_hits: tuple[bool, ...] | None = None,
    ) -> dict[str, int]:
        batch, padded_modules = encoded.module_present.shape
        environment_count = int(encoded.env_coords.shape[1])
        active_modules = (encoded.module_present > 0.5).sum(dim=1)
        mm_rectangular = int(batch) * int(padded_modules) ** 2
        me_rectangular = int(batch) * int(padded_modules) * environment_count
        mm_valid = int((active_modules * (active_modules - 1)).sum())
        me_valid = int(active_modules.sum()) * environment_count
        ledger = {
            "cover_prepare_mm_executed_rows": int(mm_executed),
            "cover_prepare_me_executed_rows": int(me_executed),
            "cover_prepare_em_executed_rows": int(em_executed),
            "cover_prepare_mm_valid_rows": mm_valid,
            "cover_prepare_me_valid_rows": me_valid,
            "cover_prepare_em_valid_rows": me_valid,
            "cover_prepare_mm_rectangular_rows": mm_rectangular,
            "cover_prepare_me_rectangular_rows": me_rectangular,
            "cover_prepare_em_rectangular_rows": me_rectangular,
            "cover_prepare_padded_or_self_rows": mm_rectangular + 2 * me_rectangular - mm_valid - 2 * me_valid,
            "cover_prepare_dense_fallback": int(dense_fallback),
            "cover_prepare_coarse_bypass_active": 1,
            "cover_prepare_local_bypass_active": 1,
        }
        if tree_cache_hits is not None:
            if len(tree_cache_hits) != int(batch):
                raise ValueError("tree cache hit flags must align with the encoded batch")
            hits = sum(bool(value) for value in tree_cache_hits)
            ledger["cover_prepare_tree_cache_hits"] = int(hits)
            ledger["cover_prepare_tree_cache_misses"] = int(batch) - int(hits)
        return ledger

    def _prepare_cover_fine_messages(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        plans: tuple[AdaptiveCoverPlan, ...],
    ) -> tuple[dict[str, Any], dict[str, int]]:
        module_valid = encoded.module_present > 0.5
        module_pairs = self._case_pairs(
            plans, encoded, encoded.module_centers,
            source_role="module", receiver_valid=module_valid, exclude_self=True,
        )
        module_environment_pairs = self._case_pairs(
            plans, encoded, encoded.module_centers,
            source_role="environment", receiver_valid=module_valid,
        )
        environment_module_pairs = self._case_pairs(
            plans, encoded, encoded.env_coords, source_role="module",
        )
        mm = self._prepare_message_pairs(
            self.mm_message, module_states, module_states,
            encoded.module_centers, encoded.module_centers, encoded.coordinate_scale,
            module_pairs, query_count=int(module_states.shape[1]),
        )
        me = self._prepare_message_pairs(
            self.me_message, module_states, encoded.env_tokens,
            encoded.module_centers, encoded.env_coords, encoded.coordinate_scale,
            module_environment_pairs, query_count=int(module_states.shape[1]),
        )
        em = self._prepare_message_pairs(
            self.em_message, encoded.env_tokens, module_states,
            encoded.env_coords, encoded.module_centers, encoded.coordinate_scale,
            environment_module_pairs, query_count=int(encoded.env_tokens.shape[1]),
        )
        active_count = encoded.module_present.sum(dim=1, keepdim=True).clamp_min(0.0)
        a_mm = mm / (1.0 + active_count[..., None])
        a_me = me / encoded.env_weights.sum(dim=1)[:, None, None].clamp_min(1.0e-12)
        a_me = a_me * encoded.module_present[..., None]
        a_em = em / (1.0 + active_count[:, :, None])
        global_modules = encoded.global_token[:, None, :].expand(-1, module_states.shape[1], -1)
        contextual_modules = (
            module_states
            + self.module_update(torch.cat((module_states, a_mm, a_me, global_modules), dim=-1))
        ) * encoded.module_present[..., None]
        global_env = encoded.global_token[:, None, :].expand(-1, encoded.env_tokens.shape[1], -1)
        contextual_env = encoded.env_tokens + self.env_update(
            torch.cat((encoded.env_tokens, a_em, global_env), dim=-1)
        )
        state = {"module_tokens": contextual_modules, "env_tokens": contextual_env}
        ledger = self._preparation_ledger(
            encoded,
            mm_executed=module_pairs.unique_pair_count,
            me_executed=module_environment_pairs.unique_pair_count,
            em_executed=environment_module_pairs.unique_pair_count,
            dense_fallback=False,
        )
        return state, ledger

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
        cover_trees: tuple[CaseLocalReceiverTree, ...] | None = None,
        cover_plans: tuple[AdaptiveCoverPlan, ...] | None = None,
        cover_tree_cache_hits: tuple[bool, ...] | None = None,
    ) -> dict[str, Any]:
        if cover_trees is None and cover_plans is None and self.optional_native_policy:
            # This is the checkpoint-native no-policy path.  Preserve Dense's
            # original method call and avoid constructing any cover objects.
            return DensePairwiseField.prepare(
                self, encoded, module_states, return_routing_maps=return_routing_maps
            )
        if cover_trees is None:
            if cover_plans is not None:
                cover_trees = tuple(plan.tree for plan in cover_plans)
            else:
                cover_trees = self.build_case_trees(encoded)
                if cover_tree_cache_hits is None:
                    cover_tree_cache_hits = self.last_case_tree_cache_hits
        if cover_plans is None:
            cover_plans = tuple(
                AdaptiveCoverPlan.full_access(tree, encoded.module_present[case], int(encoded.env_coords.shape[1]))
                for case, tree in enumerate(cover_trees)
            )
        self._validate_plans(encoded, cover_trees, cover_plans)
        all_access = all(
            self.is_full_access_plan(plan, encoded.module_present[case], int(encoded.env_coords.shape[1]))
            for case, plan in enumerate(cover_plans)
        )
        if all_access:
            state = DensePairwiseField.prepare(
                self, encoded, module_states, return_routing_maps=return_routing_maps
            )
            batch, padded_modules = encoded.module_present.shape
            environment_count = int(encoded.env_coords.shape[1])
            ledger = self._preparation_ledger(
                encoded,
                mm_executed=int(batch) * int(padded_modules) ** 2,
                me_executed=int(batch) * int(padded_modules) * environment_count,
                em_executed=int(batch) * int(padded_modules) * environment_count,
                dense_fallback=True,
                tree_cache_hits=cover_tree_cache_hits,
            )
        else:
            state, ledger = self._prepare_cover_fine_messages(encoded, module_states, cover_plans)
            if cover_tree_cache_hits is not None:
                if len(cover_tree_cache_hits) != int(encoded.module_present.shape[0]):
                    raise ValueError("tree cache hit flags must align with the encoded batch")
                hits = sum(bool(value) for value in cover_tree_cache_hits)
                ledger["cover_prepare_tree_cache_hits"] = int(hits)
                ledger["cover_prepare_tree_cache_misses"] = int(encoded.module_present.shape[0]) - int(hits)
        state["cover_trees"] = cover_trees
        state["cover_plans"] = cover_plans
        state["cover_all_access"] = all_access
        state["cover_preparation_ledger"] = ledger
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
        plans = state.get("cover_plans")
        if (
            self.cover_mode == "full_access"
            or plans is None
            or (bool(state.get("cover_all_access", False)) and self.cover_mode != "external")
        ):
            context, aux = DensePairwiseField.read(
                self, state, encoded, receivers, receiver_features,
                return_routing_maps=return_routing_maps,
            )
            if plans is None:
                return context, aux
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
                "cover_path_diagnostics_available": torch.tensor(1, device=receivers.device),
                "cover_query_degree_sum": torch.tensor(query_count * int(receivers.shape[0]), device=receivers.device),
                "cover_query_degree_max": torch.tensor(1, device=receivers.device),
                "cover_query_count": torch.tensor(query_count * int(receivers.shape[0]), device=receivers.device),
                "cover_executor_dense_fallback": torch.tensor(1, device=receivers.device),
            })
            return context, aux
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
                include_path_diagnostics=not self.optional_native_policy,
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
            "cover_path_diagnostics_available": torch.tensor(
                int(all(item.path_diagnostics_included for item in ledgers)), device=receivers.device
            ),
        }
        if all(item.path_diagnostics_included for item in ledgers):
            aux["cover_qm_raw_paths"] = torch.tensor(
                sum(item.qm_raw_paths for item in ledgers), device=receivers.device
            )
            aux["cover_qe_raw_paths"] = torch.tensor(
                sum(item.qe_raw_paths for item in ledgers), device=receivers.device
            )
        return module_context + environment_context, aux


__all__ = ["AdaptiveCoverPairwiseField"]
