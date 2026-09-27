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
from dataclasses import replace
from typing import Any

import torch

from .adaptive_interaction_cover import (
    AdaptiveCoverPlan,
    CaseLocalReceiverTree,
    CompiledMechanismExecutionView,
    CoverPairLedger,
    InteractionContext,
    InteractionPermissionKey,
    MechanismPlan,
    ReceiverAnchorUniverse,
    compile_cover_pairs,
    compile_cover_transport_pairs,
    endpoint_smoothstep,
)
from .dense_pairwise import DensePairwiseField
from .routed_pairwise import RoutedPairwiseField
from .routing_index.types import PackedPairs
from .types import EncodedInterfaceCase


class _NativeForwardMaskedRoutingGradient(torch.autograd.Function):
    """Keep native values/AD and add only masked-path routing gradients."""

    @staticmethod
    def forward(ctx: Any, native: torch.Tensor, surrogate: torch.Tensor, *routing: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(surrogate, *routing)
        return native

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        surrogate, *routing = ctx.saved_tensors
        if not routing or not surrogate.requires_grad:
            return (gradient, None, *(None for _ in routing))
        with torch.enable_grad():
            routing_gradients = torch.autograd.grad(
                surrogate,
                tuple(routing),
                grad_outputs=gradient,
                retain_graph=True,
                allow_unused=True,
            )
        return (gradient, None, *routing_gradients)


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
        # Native policy experiments begin with a dense-masked scientific
        # reference. Historical adaptive-cover configurations keep their
        # packed reader unless explicitly switched by an evaluation.
        self.cover_executor = "dense_masked" if self.optional_native_policy else "packed"
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

    def set_cover_executor(self, executor: str) -> None:
        if executor not in {"dense_masked", "rectangular_subset", "packed"}:
            raise ValueError("cover executor must be dense_masked, rectangular_subset, or packed")
        self.cover_executor = executor

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

    def rebind_fixed_plans(
        self,
        encoded: EncodedInterfaceCase,
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
    ) -> tuple[MechanismPlan, ...]:
        """Reuse frozen combinatorial supports with current live geometry.

        Anchor count/order, role codes, split topology, module validity, and
        environmental catalogue size are checked. Coordinates and scale are
        rebound to this trial's encoded tensors so receiver access weights
        and all model states remain live to the current design.
        """

        batch = int(encoded.module_present.shape[0])
        if len(plans) != batch:
            raise ValueError("one fixed cover plan is required per encoded case")
        rebound: list[MechanismPlan] = []
        for case, raw_plan in enumerate(plans):
            plan = (
                raw_plan
                if isinstance(raw_plan, MechanismPlan)
                else MechanismPlan.from_legacy(raw_plan, encoded.module_present[case])
            )
            current_universe = self._default_universe(encoded, case)
            frozen_universe = plan.tree.universe
            if current_universe.coordinates.shape != frozen_universe.coordinates.shape:
                raise ValueError(f"fixed cover plan for case {case} has a different receiver-anchor shape")
            if current_universe.roles.shape != frozen_universe.roles.shape or not torch.equal(
                current_universe.roles, frozen_universe.roles
            ):
                raise ValueError(f"fixed cover plan for case {case} has a different receiver-role sequence")
            # Role zero is the stable environment-source role in every native
            # receiver catalogue.  Frozen plans contain source IDs, so a
            # same-count environmental permutation must fail before those IDs
            # can be rebound to different live feature rows. Module, port, and
            # other functional anchors may move while retaining their roles.
            environment_role = frozen_universe.roles.new_tensor(0)
            frozen_environment_coords = frozen_universe.coordinates[
                frozen_universe.roles == environment_role
            ]
            current_environment_coords = encoded.env_coords[case]
            if frozen_environment_coords.shape != current_environment_coords.shape:
                raise ValueError(
                    f"fixed cover plan for case {case} has a different environmental source support"
                )
            frozen_environment_coords = frozen_environment_coords.to(
                device=current_environment_coords.device,
                dtype=current_environment_coords.dtype,
            )
            if not torch.equal(frozen_environment_coords, current_environment_coords):
                raise ValueError(
                    f"fixed cover plan for case {case} has a different environmental source coordinate order"
                )
            if current_universe.coordinate_scale.shape != frozen_universe.coordinate_scale.shape:
                raise ValueError(f"fixed cover plan for case {case} uses a different coordinate dimension")
            if not torch.equal(current_universe.coordinate_scale, frozen_universe.coordinate_scale):
                raise ValueError(f"fixed cover plan for case {case} uses a different coordinate normalization")
            if not torch.equal(encoded.module_present[case] > 0.5, plan.module_present > 0.5):
                raise ValueError(f"fixed cover plan for case {case} has a different module-presence catalogue")
            if int(encoded.env_coords.shape[1]) != plan.environment_count:
                raise ValueError(f"fixed cover plan for case {case} has a different environment source count")
            current_tree = CaseLocalReceiverTree(
                current_universe,
                plan.tree.nodes,
                plan.tree.overlap_fraction,
                plan.tree.capacity_saturated,
            )
            rebound.append(replace(plan, tree=current_tree))
        return tuple(rebound)

    @staticmethod
    def is_full_access_plan(
        plan: AdaptiveCoverPlan | MechanismPlan,
        module_present: torch.Tensor,
        environment_count: int,
        *,
        phase: str | None = None,
        mechanisms: tuple[str, ...] = ("MM", "ME", "EM", "QM", "QE"),
    ) -> bool:
        """Recognize a root-only all-source plan without evaluating queries."""

        if isinstance(plan, MechanismPlan):
            return plan.is_full_access(phase=phase, mechanisms=mechanisms)  # type: ignore[arg-type]
        return MechanismPlan.from_legacy(plan, module_present).is_full_access(
            phase=phase,
            mechanisms=mechanisms,  # type: ignore[arg-type]
        )

    @staticmethod
    def _plan_has_differentiable_routing(
        plan: AdaptiveCoverPlan | MechanismPlan,
        *,
        phase: str | None,
        mechanisms: tuple[str, ...],
    ) -> bool:
        """Whether this call must retain gradients through explicit topology.

        An all-access hard-forward mask can still carry a straight-through
        gradient.  In that case the native Dense shortcut would make the
        permission values observationally dead, so use the differentiable
        dense-masked executor for the affected mechanisms.
        """

        if plan.split_gates.requires_grad:
            return True
        if isinstance(plan, MechanismPlan):
            return any(
                plan.permission_matrix(mechanism, phase=phase).requires_grad
                for mechanism in mechanisms
            )
        return (
            ("MM" in mechanisms or "EM" in mechanisms or "QM" in mechanisms)
            and plan.module_membership.requires_grad
        ) or (
            ("ME" in mechanisms or "QE" in mechanisms)
            and plan.environment_membership.requires_grad
        )

    @staticmethod
    def _differentiable_plan_inputs(
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        *,
        phase: str | None,
        mechanisms: tuple[str, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return effective live gate/permission tensors for the ST bridge."""

        values: list[torch.Tensor] = []
        seen: set[int] = set()

        def add(value: torch.Tensor) -> None:
            if value.requires_grad and id(value) not in seen:
                seen.add(id(value))
                values.append(value)

        for plan in plans:
            add(plan.split_gates)
            if isinstance(plan, MechanismPlan):
                for mechanism in mechanisms:
                    base_key = InteractionPermissionKey(mechanism)
                    phase_key = InteractionPermissionKey(mechanism, phase)
                    selected_key = (
                        phase_key
                        if phase is not None and phase_key in plan.permissions
                        else base_key
                    )
                    # Route gradients to the stored permission tensor itself.
                    # permission_matrix() applies validity masking and returns
                    # a fresh nonleaf tensor, which is not an input to the
                    # custom bridge even though it is used by the executor.
                    if selected_key in plan.permissions:
                        add(plan.permissions[selected_key])
            else:
                if "MM" in mechanisms or "EM" in mechanisms or "QM" in mechanisms:
                    add(plan.module_membership)
                if "ME" in mechanisms or "QE" in mechanisms:
                    add(plan.environment_membership)
        return tuple(values)

    @staticmethod
    def _bridge_preparation_ledger(
        native: dict[str, int], surrogate: dict[str, int]
    ) -> dict[str, int]:
        """Count both the exact native pass and its routing-gradient pass."""

        ledger = dict(native)
        native_total = surrogate_total = 0
        for mechanism in ("MM", "ME", "EM"):
            tag = mechanism.lower()
            prefix = f"cover_prepare_{tag}_"
            native_rows = int(native[f"{prefix}actual_rows"])
            surrogate_rows = int(surrogate[f"{prefix}actual_rows"])
            native_total += native_rows
            surrogate_total += surrogate_rows
            ledger[f"{prefix}gradient_bridge_native_rows"] = native_rows
            ledger[f"{prefix}gradient_bridge_surrogate_rows"] = surrogate_rows
            ledger[f"{prefix}gradient_bridge_duplicate_rows"] = min(
                native_rows, surrogate_rows
            )
            ledger[f"{prefix}executed_rows"] = native_rows + surrogate_rows
            ledger[f"{prefix}actual_rows"] = native_rows + surrogate_rows
            ledger[f"{prefix}rectangular_rows"] = int(native[f"{prefix}rectangular_rows"]) + int(
                surrogate[f"{prefix}rectangular_rows"]
            )
            ledger[f"{prefix}padded_rows"] = int(native[f"{prefix}padded_rows"]) + int(
                surrogate[f"{prefix}padded_rows"]
            )
            # Unique logical source/receiver pairs describe the plan, not the
            # number of times those pairs were evaluated by the bridge.
            ledger[f"{prefix}unique_pairs"] = int(surrogate[f"{prefix}unique_pairs"])
            ledger[f"{prefix}executor_dense_masked"] = 1
            ledger[f"{prefix}executor_native_dense"] = 1
            ledger[f"{prefix}gradient_bridge"] = 1
        ledger["cover_prepare_padded_or_self_rows"] = sum(
            int(ledger[f"cover_prepare_{tag}_padded_rows"])
            for tag in ("mm", "me", "em")
        )
        ledger["cover_prepare_executor_dense_masked"] = 1
        ledger["cover_prepare_executor_native_dense"] = 1
        ledger["cover_prepare_gradient_bridge"] = 1
        ledger["cover_prepare_gradient_bridge_native_rows"] = native_total
        ledger["cover_prepare_gradient_bridge_surrogate_rows"] = surrogate_total
        ledger["cover_prepare_gradient_bridge_extra_rows"] = surrogate_total
        return ledger

    @staticmethod
    def _bridge_query_aux(
        native: dict[str, torch.Tensor],
        surrogate: dict[str, torch.Tensor],
        *,
        native_module_rows: int,
        native_environment_rows: int,
    ) -> dict[str, torch.Tensor]:
        """Retain native diagnostics and charge both query execution paths."""

        aux = dict(native)
        aux.update({key: value for key, value in surrogate.items() if key.startswith("cover_")})
        device = next(iter(surrogate.values())).device
        native_total = torch.zeros((), device=device, dtype=torch.long)
        surrogate_total = torch.zeros((), device=device, dtype=torch.long)
        for mechanism, native_row_count in (
            ("QM", native_module_rows), ("QE", native_environment_rows)
        ):
            tag = mechanism.lower()
            native_rows = torch.tensor(native_row_count, device=device, dtype=torch.long)
            surrogate_rows = surrogate[f"cover_{tag}_actual_rows"]
            native_total = native_total + native_rows
            surrogate_total = surrogate_total + surrogate_rows
            aux[f"cover_{tag}_gradient_bridge_native_rows"] = native_rows
            aux[f"cover_{tag}_gradient_bridge_surrogate_rows"] = surrogate_rows
            aux[f"cover_{tag}_gradient_bridge_duplicate_rows"] = torch.minimum(
                native_rows, surrogate_rows
            )
            aux[f"cover_{tag}_actual_rows"] = native_rows + surrogate_rows
            aux[f"cover_{tag}_executed_rows"] = native_rows + surrogate_rows
            aux[f"cover_{tag}_rectangular_rows"] = (
                native_rows + surrogate[f"cover_{tag}_rectangular_rows"]
            )
            aux[f"cover_{tag}_padded_rows"] = (
                (native_rows - surrogate[f"cover_{tag}_unique_rows"]).clamp_min(0)
                + surrogate[f"cover_{tag}_padded_rows"]
            )
            aux[f"cover_{tag}_gradient_bridge"] = torch.ones((), device=device, dtype=torch.long)
            aux[f"cover_{tag}_executor_native_dense"] = torch.ones((), device=device, dtype=torch.long)
        aux["cover_executor_native_dense"] = torch.ones((), device=device, dtype=torch.long)
        aux["cover_gradient_bridge_surrogate"] = torch.ones((), device=device, dtype=torch.long)
        aux["cover_gradient_bridge_native_rows"] = native_total
        aux["cover_gradient_bridge_surrogate_rows"] = surrogate_total
        aux["cover_gradient_bridge_extra_rows"] = surrogate_total
        return aux

    @staticmethod
    def _validate_plans(
        encoded: EncodedInterfaceCase,
        trees: tuple[CaseLocalReceiverTree, ...],
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
    ) -> None:
        batch = int(encoded.module_present.shape[0])
        if len(trees) != batch or len(plans) != batch:
            raise ValueError("native interaction policy must return one tree and plan per case")
        module_count = int(encoded.module_present.shape[1])
        environment_count = int(encoded.env_coords.shape[1])
        for case, (tree, plan) in enumerate(zip(trees, plans, strict=True)):
            if plan.tree is not tree:
                raise ValueError("native interaction plans must use the supplied case-local trees")
            if isinstance(plan, MechanismPlan):
                if plan.module_present.shape != (module_count,):
                    raise ValueError("native typed plan module catalogue does not match the padded module count")
                if plan.environment_count != environment_count:
                    raise ValueError("native typed plan environment catalogue does not match the encoded quadrature")
                if not torch.equal(plan.module_present > 0.5, encoded.module_present[case] > 0.5):
                    raise ValueError("native typed plan module validity does not match the encoded case")
            else:
                if plan.module_membership.shape[1] != module_count:
                    raise ValueError("native interaction module memberships do not match the padded module count")
                if plan.environment_membership.shape[1] != environment_count:
                    raise ValueError("native interaction environment memberships do not match the encoded quadrature")
            if plan.split_gates.device != encoded.module_present.device:
                raise ValueError(f"native interaction plan for case {case} is on the wrong device")

    @staticmethod
    def _case_pairs(
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        *,
        source_role: str,
        mechanism: str,
        phase: str | None,
        receiver_valid: torch.Tensor | None = None,
        exclude_self: bool = False,
    ) -> PackedPairs:
        pairs = [
            compile_cover_transport_pairs(
                plan,
                receivers[case],
                source_role=source_role,
                mechanism=mechanism,
                phase=phase,
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
        mm_unique: int | None = None,
        me_unique: int | None = None,
        em_unique: int | None = None,
        executor: str = "packed",
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...] | None = None,
        phase: str | None = None,
        tree_cache_hits: tuple[bool, ...] | None = None,
    ) -> dict[str, int]:
        batch, padded_modules = encoded.module_present.shape
        environment_count = int(encoded.env_coords.shape[1])
        active_modules = (encoded.module_present > 0.5).sum(dim=1)
        mm_rectangular = int(batch) * int(padded_modules) ** 2
        me_rectangular = int(batch) * int(padded_modules) * environment_count
        mm_valid = int((active_modules * (active_modules - 1)).sum())
        me_valid = int(active_modules.sum()) * environment_count
        mm_unique = mm_valid if mm_unique is None else int(mm_unique)
        me_unique = me_valid if me_unique is None else int(me_unique)
        em_unique = me_valid if em_unique is None else int(em_unique)
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
            "cover_prepare_mm_actual_rows": int(mm_executed),
            "cover_prepare_me_actual_rows": int(me_executed),
            "cover_prepare_em_actual_rows": int(em_executed),
            "cover_prepare_mm_unique_pairs": mm_unique,
            "cover_prepare_me_unique_pairs": me_unique,
            "cover_prepare_em_unique_pairs": em_unique,
            "cover_prepare_mm_padded_rows": max(0, int(mm_executed) - mm_unique),
            "cover_prepare_me_padded_rows": max(0, int(me_executed) - me_unique),
            "cover_prepare_em_padded_rows": max(0, int(em_executed) - em_unique),
            "cover_prepare_padded_or_self_rows": max(
                0, int(mm_executed) + int(me_executed) + int(em_executed)
                - mm_unique - me_unique - em_unique
            ),
            "cover_prepare_dense_fallback": int(dense_fallback),
            "cover_prepare_executor_dense_masked": int(executor == "dense_masked"),
            "cover_prepare_executor_rectangular_subset": int(executor == "rectangular_subset"),
            "cover_prepare_executor_packed": int(executor == "packed"),
            "cover_prepare_executor_full_access": int(executor == "full_access"),
            "cover_prepare_coarse_bypass_active": 1,
            "cover_prepare_local_bypass_active": 1,
        }
        for mechanism in ("MM", "ME", "EM"):
            tag = mechanism.lower()
            ledger[f"cover_prepare_{tag}_executor_dense_masked"] = int(executor == "dense_masked")
            ledger[f"cover_prepare_{tag}_executor_rectangular_subset"] = int(executor == "rectangular_subset")
            ledger[f"cover_prepare_{tag}_executor_packed"] = int(executor == "packed")
            ledger[f"cover_prepare_{tag}_executor_full_access"] = int(executor == "full_access")
            ledger[f"cover_prepare_{tag}_dense_fallback"] = int(dense_fallback)
        if plans is not None:
            for mechanism in ("MM", "ME", "EM"):
                missing = full = 0
                for case, plan in enumerate(plans):
                    if isinstance(plan, MechanismPlan):
                        missing += int(plan.permission_status(mechanism, phase=phase) == "full_access_bypass_missing_key")
                        full += int(plan.is_full_access(phase=phase, mechanisms=(mechanism,)))
                    else:
                        full += int(
                            AdaptiveCoverPairwiseField.is_full_access_plan(
                                plan,
                                encoded.module_present[case],
                                int(plan.environment_membership.shape[1]),
                                mechanisms=(mechanism,),
                            )
                        )
                ledger[f"cover_prepare_{mechanism}_key_missing"] = missing
                ledger[f"cover_prepare_{mechanism}_full_access_bypass"] = full
        if tree_cache_hits is not None:
            if len(tree_cache_hits) != int(batch):
                raise ValueError("tree cache hit flags must align with the encoded batch")
            hits = sum(bool(value) for value in tree_cache_hits)
            ledger["cover_prepare_tree_cache_hits"] = int(hits)
            ledger["cover_prepare_tree_cache_misses"] = int(batch) - int(hits)
        return ledger

    def _prepare_cover_fine_messages_packed(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        *,
        phase: str | None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        module_valid = encoded.module_present > 0.5
        module_pairs = self._case_pairs(
            plans, encoded, encoded.module_centers,
            source_role="module", mechanism="MM", phase=phase,
            receiver_valid=module_valid, exclude_self=True,
        )
        module_environment_pairs = self._case_pairs(
            plans, encoded, encoded.module_centers,
            source_role="environment", mechanism="ME", phase=phase,
            receiver_valid=module_valid,
        )
        environment_module_pairs = self._case_pairs(
            plans, encoded, encoded.env_coords, source_role="module",
            mechanism="EM", phase=phase,
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
            mm_unique=module_pairs.unique_pair_count,
            me_unique=module_environment_pairs.unique_pair_count,
            em_unique=environment_module_pairs.unique_pair_count,
            executor="packed",
            plans=plans,
            phase=phase,
        )
        return state, ledger

    @staticmethod
    def _plan_access(
        plan: AdaptiveCoverPlan | MechanismPlan,
        mechanism: str,
        receivers: torch.Tensor,
        source_count: int,
        *,
        phase: str | None,
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(plan, MechanismPlan):
            return plan.access_for(
                mechanism,
                receivers,
                source_count,
                phase=phase,
                module_present=module_present if mechanism in {"MM", "EM", "QM"} else None,
            )
        access = plan.access(receivers)
        values = access.module_source if mechanism in {"MM", "EM", "QM"} else access.environment_source
        if mechanism in {"MM", "EM", "QM"}:
            values = values * (module_present > 0.5).to(values.dtype)[None, :]
        return values

    def _preparation_priors(
        self,
        encoded: EncodedInterfaceCase,
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        *,
        phase: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, modules = encoded.module_present.shape
        environment_count = int(encoded.env_coords.shape[1])
        module_valid = encoded.module_present > 0.5
        module_access: dict[str, list[torch.Tensor]] = {"MM": [], "EM": []}
        environment_access: list[torch.Tensor] = []
        for case, plan in enumerate(plans):
            module_access["MM"].append(self._plan_access(
                plan, "MM", encoded.module_centers[case], modules, phase=phase,
                module_present=encoded.module_present[case],
            ))
            module_access["EM"].append(self._plan_access(
                plan, "EM", encoded.env_coords[case], modules, phase=phase,
                module_present=encoded.module_present[case],
            ))
            environment_access.append(self._plan_access(
                plan, "ME", encoded.module_centers[case], environment_count, phase=phase,
                module_present=encoded.module_present[case],
            ))
        mm_prior = torch.stack(module_access["MM"]) * module_valid[:, :, None].to(encoded.module_centers.dtype)
        mm_prior = mm_prior * module_valid[:, None, :].to(mm_prior.dtype)
        mm_prior = mm_prior.masked_fill(
            torch.eye(modules, device=mm_prior.device, dtype=torch.bool)[None], 0.0
        )
        me_prior = torch.stack(environment_access) * encoded.env_weights[:, None, :]
        me_prior = me_prior * module_valid[:, :, None].to(me_prior.dtype)
        em_prior = torch.stack(module_access["EM"]) * module_valid[:, None, :].to(encoded.env_coords.dtype)
        return mm_prior, me_prior, em_prior

    def _dense_masked_message(
        self,
        network: torch.nn.Module,
        receiver_states: torch.Tensor,
        source_states: torch.Tensor,
        receiver_coords: torch.Tensor,
        source_coords: torch.Tensor,
        scale: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the full native rectangle, then apply live typed priors."""

        batch, receiver_count, _ = receiver_states.shape
        scale_rows = self._environment_scale_rows(scale, batch)[:, 0, :]
        values: list[torch.Tensor] = []
        receiver_tile = max(1, min(receiver_count, self.fine_pair_chunk_size // max(1, source_states.shape[1])))
        for start in range(0, receiver_count, receiver_tile):
            stop = min(start + receiver_tile, receiver_count)
            relative = (
                receiver_coords[:, start:stop, None, :] - source_coords[:, None, :, :]
            ) / scale_rows[:, None, None, :]
            receiver = receiver_states[:, start:stop, None, :].expand(-1, -1, source_states.shape[1], -1)
            source = source_states[:, None, :, :].expand(-1, stop - start, -1, -1)
            messages = self._mlp(network, torch.cat((receiver, source, self.relative_fourier(relative)), dim=-1))
            values.append((messages * prior[:, start:stop, :, None].to(messages.dtype)).sum(dim=2))
        return torch.cat(values, dim=1)

    def _rectangular_message(
        self,
        network: torch.nn.Module,
        receiver_states: torch.Tensor,
        source_states: torch.Tensor,
        receiver_coords: torch.Tensor,
        source_coords: torch.Tensor,
        scale_rows: torch.Tensor,
        prior: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Gather one source rectangle per compatible receiver support."""

        batch, receiver_count, hidden = receiver_states.shape
        outputs: list[torch.Tensor] = []
        actual_rows = 0
        for case in range(batch):
            weights = prior[case]
            grouped_rows = self._group_rows_by_support(weights)
            result = receiver_states.new_zeros((receiver_count, hidden))
            for support, row_values in grouped_rows.items():
                if not support:
                    continue
                row_ids = torch.as_tensor(row_values, device=receiver_states.device, dtype=torch.long)
                source_ids = torch.as_tensor(support, device=receiver_states.device, dtype=torch.long)
                selected_receiver_states = receiver_states[case].index_select(0, row_ids)
                selected_source_states = source_states[case].index_select(0, source_ids)
                selected_receiver_coords = receiver_coords[case].index_select(0, row_ids)
                selected_source_coords = source_coords[case].index_select(0, source_ids)
                relative = (
                    selected_receiver_coords[:, None, :] - selected_source_coords[None, :, :]
                ) / scale_rows[case]
                receiver = selected_receiver_states[:, None, :].expand(-1, len(support), -1)
                source = selected_source_states[None, :, :].expand(len(row_values), -1, -1)
                messages = self._mlp(network, torch.cat((receiver, source, self.relative_fourier(relative)), dim=-1))
                pair_prior = weights.index_select(0, row_ids).index_select(1, source_ids)
                result = result.index_copy(0, row_ids, (messages * pair_prior[..., None]).sum(dim=1))
                actual_rows += len(row_values) * len(support)
            outputs.append(result)
        return torch.stack(outputs), actual_rows

    @staticmethod
    def _group_rows_by_support(weights: torch.Tensor) -> dict[tuple[int, ...], list[int]]:
        """Group receiver rows after one bulk support-mask transfer.

        Repeated per-row ``nonzero(...).tolist()`` synchronizes the device
        once per receiver.  Export the Boolean topology once, hash compact
        byte signatures on the host, and keep every selected prior value live
        on the original device for the subsequent differentiable kernels.
        """

        if weights.ndim != 2:
            raise ValueError("receiver/source weights must have shape [Q,S]")
        support_masks = (weights.detach() > 0).to(dtype=torch.uint8).cpu().contiguous().numpy()
        grouped_rows: dict[bytes, list[int]] = {}
        for row, mask in enumerate(support_masks):
            grouped_rows.setdefault(mask.tobytes(), []).append(row)
        return {
            tuple(index for index, active in enumerate(signature) if active): rows
            for signature, rows in grouped_rows.items()
        }

    @staticmethod
    def _contextualize_prepared_messages(
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        mm: torch.Tensor,
        me: torch.Tensor,
        em: torch.Tensor,
        module_update: torch.nn.Module,
        env_update: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        active_count = encoded.module_present.sum(dim=1, keepdim=True).clamp_min(0.0)
        a_mm = mm / (1.0 + active_count[..., None])
        a_me = me / encoded.env_weights.sum(dim=1)[:, None, None].clamp_min(1.0e-12)
        a_me = a_me * encoded.module_present[..., None]
        a_em = em / (1.0 + active_count[:, :, None])
        global_modules = encoded.global_token[:, None, :].expand(-1, module_states.shape[1], -1)
        contextual_modules = (
            module_states + module_update(torch.cat((module_states, a_mm, a_me, global_modules), dim=-1))
        ) * encoded.module_present[..., None]
        global_env = encoded.global_token[:, None, :].expand(-1, encoded.env_tokens.shape[1], -1)
        contextual_env = encoded.env_tokens + env_update(
            torch.cat((encoded.env_tokens, a_em, global_env), dim=-1)
        )
        return {"module_tokens": contextual_modules, "env_tokens": contextual_env}

    def _prepare_cover_fine_messages_masked(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        *,
        phase: str | None,
        executor: str,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        mm_prior, me_prior, em_prior = self._preparation_priors(encoded, plans, phase=phase)
        if executor == "dense_masked":
            mm = self._dense_masked_message(
                self.mm_message, module_states, module_states, encoded.module_centers,
                encoded.module_centers, encoded.coordinate_scale, mm_prior,
            )
            me = self._dense_masked_message(
                self.me_message, module_states, encoded.env_tokens, encoded.module_centers,
                encoded.env_coords, encoded.coordinate_scale, me_prior,
            )
            em = self._dense_masked_message(
                self.em_message, encoded.env_tokens, module_states, encoded.env_coords,
                encoded.module_centers, encoded.coordinate_scale, em_prior,
            )
            actual = (
                int(encoded.module_present.shape[0] * encoded.module_present.shape[1] ** 2),
                int(encoded.module_present.shape[0] * encoded.module_present.shape[1] * encoded.env_coords.shape[1]),
                int(encoded.module_present.shape[0] * encoded.env_coords.shape[1] * encoded.module_present.shape[1]),
            )
        elif executor == "rectangular_subset":
            scale_rows = self._environment_scale_rows(
                encoded.coordinate_scale, int(encoded.module_present.shape[0])
            )[:, 0, :]
            mm, mm_actual = self._rectangular_message(
                self.mm_message, module_states, module_states, encoded.module_centers,
                encoded.module_centers, scale_rows, mm_prior,
            )
            me, me_actual = self._rectangular_message(
                self.me_message, module_states, encoded.env_tokens, encoded.module_centers,
                encoded.env_coords, scale_rows, me_prior,
            )
            em, em_actual = self._rectangular_message(
                self.em_message, encoded.env_tokens, module_states, encoded.env_coords,
                encoded.module_centers, scale_rows, em_prior,
            )
            actual = (mm_actual, me_actual, em_actual)
        else:
            raise ValueError(f"masked preparation does not support executor {executor!r}")
        state = self._contextualize_prepared_messages(
            encoded, module_states, mm, me, em, self.module_update, self.env_update
        )
        unique = (
            int((mm_prior > 0).sum()),
            int((me_prior > 0).sum()),
            int((em_prior > 0).sum()),
        )
        ledger = self._preparation_ledger(
            encoded,
            mm_executed=actual[0],
            me_executed=actual[1],
            em_executed=actual[2],
            # Dense-masked is the explicit scientific reference executor, not
            # a silent all-access bypass. Keep those accounting concepts
            # separate so the executor flag is the authoritative path.
            dense_fallback=False,
            mm_unique=unique[0],
            me_unique=unique[1],
            em_unique=unique[2],
            executor=executor,
            plans=plans,
            phase=phase,
        )
        return state, ledger

    def _prepare_cover_fine_messages(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        *,
        context: InteractionContext | None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        phase = None if context is None else context.phase
        executor = self.cover_executor
        if executor in {"dense_masked", "rectangular_subset"}:
            return self._prepare_cover_fine_messages_masked(
                encoded, module_states, plans, phase=phase, executor=executor
            )
        state, ledger = self._prepare_cover_fine_messages_packed(
            encoded, module_states, plans, phase=phase
        )
        ledger.update({
            "cover_prepare_executor_dense_masked": 0,
            "cover_prepare_executor_rectangular_subset": 0,
            "cover_prepare_executor_packed": 1,
        })
        return state, ledger

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
        cover_trees: tuple[CaseLocalReceiverTree, ...] | None = None,
        cover_plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...] | None = None,
        interaction_context: InteractionContext | None = None,
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
        phase = None if interaction_context is None else interaction_context.phase
        all_access = all(
            self.is_full_access_plan(
                plan,
                encoded.module_present[case],
                int(encoded.env_coords.shape[1]),
                phase=phase,
                mechanisms=("MM", "ME", "EM"),
            )
            for case, plan in enumerate(cover_plans)
        )
        differentiable_prepare = any(
            self._plan_has_differentiable_routing(
                plan, phase=phase, mechanisms=("MM", "ME", "EM")
            )
            for plan in cover_plans
        )
        if all_access and not differentiable_prepare:
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
                executor="full_access",
                plans=cover_plans,
                phase=phase,
                tree_cache_hits=cover_tree_cache_hits,
            )
        elif all_access and differentiable_prepare:
            # Use the incumbent Dense computation for exact forward values
            # and model/input derivatives. A second dense-masked pass supplies
            # only the declared topology surrogate derivative. Both work
            # scopes are reported in the preparation ledger below.
            native_state = DensePairwiseField.prepare(
                self, encoded, module_states, return_routing_maps=return_routing_maps
            )
            native_ledger = self._preparation_ledger(
                encoded,
                mm_executed=int(encoded.module_present.shape[0]) * int(encoded.module_present.shape[1]) ** 2,
                me_executed=int(encoded.module_present.shape[0])
                * int(encoded.module_present.shape[1]) * int(encoded.env_coords.shape[1]),
                em_executed=int(encoded.module_present.shape[0])
                * int(encoded.env_coords.shape[1]) * int(encoded.module_present.shape[1]),
                dense_fallback=True,
                executor="full_access",
                plans=cover_plans,
                phase=phase,
                tree_cache_hits=cover_tree_cache_hits,
            )
            surrogate_state, surrogate_ledger = self._prepare_cover_fine_messages_masked(
                encoded,
                module_states,
                cover_plans,
                phase=phase,
                executor="dense_masked",
            )
            prep_routing = self._differentiable_plan_inputs(
                cover_plans, phase=phase, mechanisms=("MM", "ME", "EM")
            )
            if prep_routing:
                for name in ("module_tokens", "env_tokens"):
                    native_state[name] = _NativeForwardMaskedRoutingGradient.apply(
                        native_state[name], surrogate_state[name], *prep_routing
                    )
            state = native_state
            state["cover_preparation_gradient_bridge"] = bool(prep_routing)
            ledger = self._bridge_preparation_ledger(native_ledger, surrogate_ledger)
        else:
            state, ledger = self._prepare_cover_fine_messages(
                encoded, module_states, cover_plans, context=interaction_context
            )
            if cover_tree_cache_hits is not None:
                if len(cover_tree_cache_hits) != int(encoded.module_present.shape[0]):
                    raise ValueError("tree cache hit flags must align with the encoded batch")
                hits = sum(bool(value) for value in cover_tree_cache_hits)
                ledger["cover_prepare_tree_cache_hits"] = int(hits)
                ledger["cover_prepare_tree_cache_misses"] = int(encoded.module_present.shape[0]) - int(hits)
        state["cover_trees"] = cover_trees
        state["cover_plans"] = cover_plans
        state["cover_all_access"] = all_access
        state["cover_prepare_all_access"] = all_access
        state["cover_interaction_context"] = interaction_context
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
    def with_external_plans(
        state: dict[str, Any], plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...]
    ) -> dict[str, Any]:
        if len(plans) != len(state["cover_trees"]):
            raise ValueError("one cover plan is required per case")
        for tree, plan in zip(state["cover_trees"], plans, strict=True):
            if plan.tree is not tree:
                raise ValueError("external plan must use the prepared case-local receiver tree")
        updated = {**state, "cover_plans": plans}
        if "cover_execution_views" in updated:
            # An externally substituted plan must never inherit stale
            # per-plan metadata from the original preparation.
            updated["cover_execution_views"] = None
        return updated

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

    def _query_accesses(
        self,
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        *,
        phase: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        module_access = torch.stack([
            self._plan_access(
                plan, "QM", receivers[case], int(encoded.module_present.shape[1]),
                phase=phase, module_present=encoded.module_present[case],
            )
            for case, plan in enumerate(plans)
        ])
        environment_access = torch.stack([
            self._plan_access(
                plan, "QE", receivers[case], int(encoded.env_coords.shape[1]),
                phase=phase, module_present=encoded.module_present[case],
            )
            for case, plan in enumerate(plans)
        ])
        return module_access, environment_access

    @staticmethod
    def _compiled_query_accesses(
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        views: tuple[CompiledMechanismExecutionView, ...],
        receivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """Evaluate one live receiver access matrix per case for both query routes."""

        if len(plans) != len(views) or len(plans) != int(receivers.shape[0]):
            raise ValueError("compiled cover views must align with the prepared cases")
        module_access = []
        environment_access = []
        receiver_access = []
        for case, (plan, view) in enumerate(zip(plans, views, strict=True)):
            if not view.matches(plan, phase=view.phase):
                raise ValueError("compiled cover execution view no longer matches its prepared plan")
            alpha = view.receiver_access(receivers[case], plan.split_gates)
            receiver_access.append(alpha)
            module_access.append(alpha @ view.permission_matrix("QM"))
            environment_access.append(alpha @ view.permission_matrix("QE"))
        return torch.stack(module_access), torch.stack(environment_access), tuple(receiver_access)

    @staticmethod
    def _environment_access_prior(
        environment_access: torch.Tensor,
        environment_weights: torch.Tensor,
        *,
        fallback_eligible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior = environment_access * environment_weights[:, None, :]
        if fallback_eligible is None:
            fallback_eligible = torch.ones(
                int(environment_access.shape[0]), device=environment_access.device, dtype=torch.bool
            )
        if fallback_eligible.shape != (int(environment_access.shape[0]),):
            raise ValueError("environment fallback eligibility must align with the case batch")
        safety_mass = environment_weights.sum(dim=1) * 1.0e-4
        environmental_mass = prior.sum(dim=-1)
        fallback = (environmental_mass < safety_mass[:, None]) & fallback_eligible[:, None]
        safety_weight = endpoint_smoothstep(
            (safety_mass[:, None] - environmental_mass) / safety_mass[:, None].clamp_min(
                torch.finfo(prior.dtype).tiny
            )
        ) * fallback_eligible[:, None].to(prior.dtype)
        prior = prior + safety_weight[:, :, None] * environment_weights[:, None, :]
        return prior, fallback

    @staticmethod
    def _legacy_environment_fallback_mask(
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Keep the historical safety fallback only for legacy tied plans."""

        return torch.tensor(
            [not isinstance(plan, MechanismPlan) for plan in plans],
            device=device,
            dtype=torch.bool,
        )

    def _read_module_rectangular(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        access: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        batch, query_count = receivers.shape[:2]
        scale_rows = self._environment_scale_rows(encoded.coordinate_scale, batch)[:, 0, :]
        outputs: list[torch.Tensor] = []
        actual_rows = 0
        for case in range(batch):
            weights = access[case] * (encoded.module_present[case] > 0.5).to(access.dtype)[None, :]
            grouped_rows = self._group_rows_by_support(weights)
            reduced = state["module_tokens"].new_zeros((query_count, self.hidden_dim))
            for support, row_values in grouped_rows.items():
                if not support:
                    continue
                row_ids = torch.as_tensor(row_values, device=receivers.device, dtype=torch.long)
                source_ids = torch.as_tensor(support, device=receivers.device, dtype=torch.long)
                source_states = state["module_tokens"][case].index_select(0, source_ids)
                source_coords = encoded.module_centers[case].index_select(0, source_ids)
                receiver_coords = receivers[case].index_select(0, row_ids)
                relative = (receiver_coords[:, None, :] - source_coords[None, :, :]) / scale_rows[case]
                source = source_states[None, :, :].expand(len(row_values), -1, -1)
                global_token = encoded.global_token[case][None, None, :].expand(len(row_values), len(support), -1)
                messages = self._mlp(self.query_module_message, torch.cat((
                    source, self.relative_fourier(relative), global_token,
                ), dim=-1))
                prior = weights.index_select(0, row_ids).index_select(1, source_ids)
                reduced = reduced.index_copy(0, row_ids, (messages * prior[..., None]).sum(dim=1))
                actual_rows += len(row_values) * len(support)
            outputs.append(reduced)
        reduced = torch.stack(outputs)
        active_count = encoded.module_present.sum(dim=1)[:, None, None]
        return self.query_module_output(reduced / (1.0 + active_count)), actual_rows

    def _read_environment_rectangular(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        prior: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        batch, query_count = receivers.shape[:2]
        key, value = self.project_environment_sources(state["env_tokens"])
        query = self.env_attention.project_query(self.env_query(receiver_features))
        scale_rows = self._environment_scale_rows(encoded.coordinate_scale, batch)[:, 0, :]
        output_rows: list[torch.Tensor] = []
        actual_rows = 0
        for case in range(batch):
            grouped_rows = self._group_rows_by_support(prior[case])
            case_output = value.new_zeros((query_count, self.hidden_dim))
            for support, row_values in grouped_rows.items():
                if not support:
                    continue
                row_ids = torch.as_tensor(row_values, device=receivers.device, dtype=torch.long)
                source_ids = torch.as_tensor(support, device=receivers.device, dtype=torch.long)
                receiver_coords = receivers[case].index_select(0, row_ids)
                source_coords = encoded.env_coords[case].index_select(0, source_ids)
                relative = (receiver_coords[:, None, :] - source_coords[None, :, :]) / scale_rows[case]
                bias = self._mlp(self.env_geometry_bias, self.relative_fourier(relative)).permute(2, 0, 1)
                query_rows = query[case, :, row_ids, :]
                key_rows = key[case, :, source_ids, :]
                value_rows = value[case, :, source_ids, :]
                scores = torch.matmul(query_rows, key_rows.transpose(-1, -2)) / (float(self.env_attention.head_dim) ** 0.5)
                scores = scores + bias
                prior_rows = prior[case].index_select(0, row_ids).index_select(1, source_ids)
                scores = scores.double() + prior_rows.double().log()[None, :, :]
                weights = torch.softmax(scores, dim=-1).to(value.dtype)
                attended = torch.matmul(weights, value_rows)
                context = attended.permute(1, 0, 2).reshape(len(row_values), self.hidden_dim)
                projected = self.env_attention.output(context[None])[0]
                case_output = case_output.index_copy(0, row_ids, projected)
                actual_rows += len(row_values) * len(support)
            output_rows.append(case_output)
        return torch.stack(output_rows), actual_rows

    def _cover_query_work_aux(
        self,
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        module_access: torch.Tensor,
        environment_prior: torch.Tensor,
        module_actual_rows: int | torch.Tensor,
        environment_actual_rows: int | torch.Tensor,
        module_unique_rows: int | torch.Tensor,
        environment_unique_rows: int | torch.Tensor,
        *,
        phase: str | None,
        executor: str,
        include_diagnostics: bool = True,
        compiled_views: tuple[CompiledMechanismExecutionView, ...] | None = None,
        receiver_group_access: tuple[torch.Tensor, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        device = encoded.module_present.device
        query_count = int(module_access.shape[1])

        def count_tensor(value: int | torch.Tensor) -> torch.Tensor:
            if torch.is_tensor(value):
                return value.to(device=device, dtype=torch.long).reshape(())
            return torch.tensor(int(value), device=device, dtype=torch.long)

        module_actual = count_tensor(module_actual_rows)
        environment_actual = count_tensor(environment_actual_rows)
        module_unique = count_tensor(module_unique_rows)
        environment_unique = count_tensor(environment_unique_rows)
        aux: dict[str, torch.Tensor] = {
            "cover_qm_unique_rows": module_unique,
            "cover_qe_unique_rows": environment_unique,
            "cover_qm_unique_source_receiver_pairs": module_unique,
            "cover_qe_unique_source_receiver_pairs": environment_unique,
            "cover_qm_actual_rows": module_actual,
            "cover_qe_actual_rows": environment_actual,
            "cover_qm_executed_rows": module_actual,
            "cover_qe_executed_rows": environment_actual,
            "cover_qm_padded_rows": (module_actual - module_unique).clamp_min(0),
            "cover_qe_padded_rows": (environment_actual - environment_unique).clamp_min(0),
            "cover_qm_rectangular_rows": module_actual,
            "cover_qe_rectangular_rows": environment_actual,
            "cover_environment_fallback_queries": torch.tensor(0, device=device, dtype=torch.long),
            "cover_path_diagnostics_available": torch.tensor(int(include_diagnostics), device=device, dtype=torch.long),
            "cover_query_count": torch.tensor(query_count * len(plans), device=device, dtype=torch.long),
            "cover_executor_dense_fallback": torch.tensor(int(executor == "full_access"), device=device, dtype=torch.long),
            "cover_executor_dense_masked": torch.tensor(int(executor == "dense_masked"), device=device, dtype=torch.long),
            "cover_executor_rectangular_subset": torch.tensor(int(executor == "rectangular_subset"), device=device, dtype=torch.long),
            "cover_executor_packed": torch.tensor(int(executor == "packed"), device=device, dtype=torch.long),
        }
        for mechanism, unique_pairs in (("QM", module_unique_rows), ("QE", environment_unique_rows)):
            tag = mechanism.lower()
            aux[f"cover_{tag}_executor_dense_masked"] = torch.tensor(
                int(executor == "dense_masked"), device=device, dtype=torch.long
            )
            aux[f"cover_{tag}_executor_rectangular_subset"] = torch.tensor(
                int(executor == "rectangular_subset"), device=device, dtype=torch.long
            )
            aux[f"cover_{tag}_executor_packed"] = torch.tensor(
                int(executor == "packed"), device=device, dtype=torch.long
            )
            aux[f"cover_{tag}_executor_full_access"] = torch.tensor(
                int(executor == "full_access"), device=device, dtype=torch.long
            )
            aux[f"cover_{tag}_dense_fallback"] = torch.tensor(
                int(executor == "full_access"), device=device, dtype=torch.long
            )
            aux[f"cover_{tag}_unique_source_receiver_pairs"] = count_tensor(unique_pairs)
        if not include_diagnostics:
            return aux

        raw_path_counts_qm: list[torch.Tensor] = []
        raw_path_counts_qe: list[torch.Tensor] = []
        query_degrees: list[torch.Tensor] = []
        if compiled_views is not None:
            if receiver_group_access is None or len(compiled_views) != len(plans):
                raise ValueError("compiled diagnostics require one receiver access matrix per case")
            for view, alpha in zip(compiled_views, receiver_group_access, strict=True):
                qm_membership = view.permission_matrix("QM")
                qe_membership = view.permission_matrix("QE")
                active_receiver_counts = (alpha > 0).sum(dim=0, dtype=torch.int64)
                qm_source_counts = (qm_membership > 0).sum(dim=1, dtype=torch.int64)
                qe_source_counts = (qe_membership > 0).sum(dim=1, dtype=torch.int64)
                raw_path_counts_qm.append((active_receiver_counts * qm_source_counts).sum())
                raw_path_counts_qe.append((active_receiver_counts * qe_source_counts).sum())
                active_source_nodes = (qm_membership > 0).any(dim=1) | (qe_membership > 0).any(dim=1)
                query_degrees.append(((alpha > 0) & active_source_nodes[None, :]).sum(dim=1))
            k_values = [
                max(view.summary("QM").source_bearing_active_nodes,
                    view.summary("QE").source_bearing_active_nodes)
                for view in compiled_views
            ]
            transitions = [view.transition_count for view in compiled_views]
            nodes = [view.candidate_node_count for view in compiled_views]
            saturated = [view.capacity_saturated for view in compiled_views]
            raw_paths_qm = torch.stack(raw_path_counts_qm).sum() if raw_path_counts_qm else torch.zeros((), device=device, dtype=torch.long)
            raw_paths_qe = torch.stack(raw_path_counts_qe).sum() if raw_path_counts_qe else torch.zeros((), device=device, dtype=torch.long)
            degree = torch.cat(query_degrees) if query_degrees else torch.zeros(0, device=device, dtype=torch.long)
        else:
            raw_paths_qm = torch.zeros((), device=device, dtype=torch.long)
            raw_paths_qe = torch.zeros((), device=device, dtype=torch.long)
            for case, plan in enumerate(plans):
                alpha = plan.tree.access(receivers[case], plan.split_gates)
                if isinstance(plan, MechanismPlan):
                    qm_membership = plan.permission_matrix("QM", phase=phase, module_present=encoded.module_present[case])
                    qe_membership = plan.permission_matrix("QE", phase=phase)
                else:
                    qm_membership = plan.module_membership
                    qe_membership = plan.environment_membership
                active_receiver_counts = (alpha > 0).sum(dim=0, dtype=torch.int64)
                qm_source_counts = (qm_membership > 0).sum(dim=1, dtype=torch.int64)
                qe_source_counts = (qe_membership > 0).sum(dim=1, dtype=torch.int64)
                raw_paths_qm = raw_paths_qm + (active_receiver_counts * qm_source_counts).sum()
                raw_paths_qe = raw_paths_qe + (active_receiver_counts * qe_source_counts).sum()
                active_source_nodes = (qm_membership > 0).any(dim=1) | (qe_membership > 0).any(dim=1)
                query_degrees.append(((alpha > 0) & active_source_nodes[None, :]).sum(dim=1))
            degree = torch.cat(query_degrees) if query_degrees else torch.zeros(0, device=device, dtype=torch.long)
            # The concrete rows and unions are authoritative. Frontier summaries
            # retain structural K diagnostics independently of execution.
            k_values = [
                max(
                    plan.frontier_summary("QM", phase).source_bearing_active_nodes,
                    plan.frontier_summary("QE", phase).source_bearing_active_nodes,
                ) if isinstance(plan, MechanismPlan) else plan.active_group_count()
                for plan in plans
            ]
            transitions = [
                sum(0.0 < float(plan.split_gates[index].detach()) < 1.0 for index, node in enumerate(plan.tree.nodes) if not node.is_leaf)
                for plan in plans
            ]
            nodes = [len(plan.tree.nodes) for plan in plans]
            saturated = [bool(plan.tree.capacity_saturated) for plan in plans]
        aux.update({
            "cover_k_case": torch.tensor(k_values, device=device, dtype=torch.long),
            "cover_transition_case": torch.tensor(transitions, device=device, dtype=torch.long),
            "cover_candidate_nodes": torch.tensor(nodes, device=device, dtype=torch.long),
            "cover_capacity_saturated": torch.tensor(saturated, device=device, dtype=torch.long),
            "cover_qm_raw_paths": count_tensor(raw_paths_qm),
            "cover_qe_raw_paths": count_tensor(raw_paths_qe),
            "cover_path_diagnostics_available": torch.ones((), device=device, dtype=torch.long),
            "cover_query_degree_sum": degree.sum().to(torch.long),
            "cover_query_degree_max": degree.max().to(torch.long) if degree.numel() else torch.zeros((), device=device, dtype=torch.long),
        })
        for mechanism, unique_pairs in (("QM", module_unique_rows), ("QE", environment_unique_rows)):
            tag = mechanism.lower()
            raw_frontier = 0
            source_bearing_frontier = 0
            nonredundant_packets = 0
            source_union = 0
            for case, plan in enumerate(plans):
                if compiled_views is not None:
                    summary = compiled_views[case].summary(mechanism)
                    raw_frontier += summary.raw_active_frontier_nodes
                    source_bearing_frontier += summary.source_bearing_active_nodes
                    nonredundant_packets += summary.nonredundant_packet_count
                    source_union += summary.source_union_count
                elif isinstance(plan, MechanismPlan):
                    summary = plan.frontier_summary(mechanism, phase)
                    raw_frontier += summary.raw_active_frontier_nodes
                    source_bearing_frontier += summary.source_bearing_active_nodes
                    nonredundant_packets += summary.nonredundant_packet_count
                    source_union += summary.source_union_count
                else:
                    summary_access = plan.tree.access(
                        plan.tree.universe.coordinates, plan.split_gates
                    )
                    membership = (
                        plan.module_membership
                        if mechanism == "QM" else plan.environment_membership
                    )
                    active_nodes = (summary_access > 0).any(dim=0)
                    bearing = active_nodes & (membership > 0).any(dim=1)
                    raw_frontier += int(active_nodes.sum())
                    source_bearing_frontier += int(bearing.sum())
                    bearing_cpu = bearing.detach().cpu().tolist()
                    membership_cpu = membership.detach().cpu().tolist()
                    nonredundant_packets += len({
                        tuple(float(value) for value in row)
                        for row, is_active in zip(membership_cpu, bearing_cpu)
                        if is_active
                    })
                    legacy_access = plan.access(plan.tree.universe.coordinates)
                    route_access = (
                        legacy_access.module_source
                        if mechanism == "QM" else legacy_access.environment_source
                    )
                    source_union += int((route_access > 0).any(dim=0).sum())
            aux[f"cover_{tag}_raw_active_frontier_nodes"] = torch.tensor(
                raw_frontier, device=device, dtype=torch.long
            )
            aux[f"cover_{tag}_source_bearing_active_frontier_nodes"] = torch.tensor(
                source_bearing_frontier, device=device, dtype=torch.long
            )
            aux[f"cover_{tag}_nonredundant_packets"] = torch.tensor(
                nonredundant_packets, device=device, dtype=torch.long
            )
            aux[f"cover_{tag}_source_union_count"] = torch.tensor(
                source_union, device=device, dtype=torch.long
            )
        for mechanism in ("QM", "QE"):
            key_missing = full_access = 0
            for case, plan in enumerate(plans):
                if compiled_views is not None:
                    key_missing += int(compiled_views[case].permission_key_missing(mechanism))
                    full_access += int(compiled_views[case].is_full_access(mechanism))
                elif isinstance(plan, MechanismPlan):
                    key_missing += int(plan.permission_status(mechanism, phase=phase) == "full_access_bypass_missing_key")
                    full_access += int(plan.is_full_access(phase=phase, mechanisms=(mechanism,)))
                else:
                    full_access += int(self.is_full_access_plan(
                        plan, encoded.module_present[case], int(encoded.env_coords.shape[1]), mechanisms=(mechanism,)
                    ))
            aux[f"cover_{mechanism.lower()}_key_missing"] = torch.tensor(key_missing, device=device, dtype=torch.long)
            aux[f"cover_{mechanism.lower()}_full_access_bypass"] = torch.tensor(full_access, device=device, dtype=torch.long)
        return aux

    @staticmethod
    def _compiled_full_access_query_aux(
        views: tuple[CompiledMechanismExecutionView, ...],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Report native full-access work from the prepared execution view.

        Full access has one exact all-source action per receiver after equal
        packet actions are compacted. Static frontier counts therefore need
        no query tree walk or permission tensor transfer in the receiver loop.
        Query-dependent path and degree counts are marked unavailable here;
        computing them would evaluate the receiver frontier solely for
        diagnostics.
        """

        batch, query_count = receivers.shape[:2]
        module_count = int(encoded.module_present.shape[1])
        environment_count = int(encoded.env_coords.shape[1])
        query_total = int(batch) * int(query_count)
        module_unique = sum(view.summary("QM").source_union_count for view in views) * int(query_count)
        environment_unique = sum(view.summary("QE").source_union_count for view in views) * int(query_count)
        module_actual = query_total * module_count
        environment_actual = query_total * environment_count
        device = receivers.device
        k_case = [
            max(
                view.summary("QM").source_bearing_active_nodes,
                view.summary("QE").source_bearing_active_nodes,
            )
            for view in views
        ]

        def scalar(value: int | bool) -> torch.Tensor:
            return torch.tensor(int(value), device=device, dtype=torch.long)

        aux: dict[str, torch.Tensor] = {
            "cover_k_case": torch.tensor(k_case, device=device, dtype=torch.long),
            "cover_transition_case": torch.tensor(
                [view.transition_count for view in views], device=device, dtype=torch.long
            ),
            "cover_candidate_nodes": torch.tensor(
                [view.candidate_node_count for view in views], device=device, dtype=torch.long
            ),
            "cover_capacity_saturated": torch.tensor(
                [view.capacity_saturated for view in views], device=device, dtype=torch.long
            ),
            "cover_qm_unique_rows": scalar(module_unique),
            "cover_qe_unique_rows": scalar(environment_unique),
            "cover_qm_unique_source_receiver_pairs": scalar(module_unique),
            "cover_qe_unique_source_receiver_pairs": scalar(environment_unique),
            "cover_qm_actual_rows": scalar(module_actual),
            "cover_qe_actual_rows": scalar(environment_actual),
            "cover_qm_executed_rows": scalar(module_actual),
            "cover_qe_executed_rows": scalar(environment_actual),
            "cover_qm_padded_rows": scalar(max(0, module_actual - module_unique)),
            "cover_qe_padded_rows": scalar(max(0, environment_actual - environment_unique)),
            "cover_qm_rectangular_rows": scalar(module_actual),
            "cover_qe_rectangular_rows": scalar(environment_actual),
            "cover_environment_fallback_queries": scalar(0),
            "cover_path_diagnostics_available": scalar(0),
            "cover_query_degree_sum": scalar(0),
            "cover_query_degree_max": scalar(0),
            "cover_query_count": scalar(query_total),
            "cover_qm_raw_paths": scalar(0),
            "cover_qe_raw_paths": scalar(0),
            "cover_executor_dense_fallback": scalar(1),
            "cover_executor_dense_masked": scalar(0),
            "cover_executor_rectangular_subset": scalar(0),
            "cover_executor_packed": scalar(0),
            "cover_executor_native_dense": scalar(1),
        }
        for mechanism, unique_rows in (("QM", module_unique), ("QE", environment_unique)):
            tag = mechanism.lower()
            summary = [view.summary(mechanism) for view in views]
            aux.update({
                f"cover_{tag}_executor_dense_masked": scalar(0),
                f"cover_{tag}_executor_rectangular_subset": scalar(0),
                f"cover_{tag}_executor_packed": scalar(0),
                f"cover_{tag}_executor_full_access": scalar(1),
                f"cover_{tag}_dense_fallback": scalar(1),
                f"cover_{tag}_unique_source_receiver_pairs": scalar(unique_rows),
                f"cover_{tag}_raw_active_frontier_nodes": scalar(
                    sum(item.raw_active_frontier_nodes for item in summary)
                ),
                f"cover_{tag}_source_bearing_active_frontier_nodes": scalar(
                    sum(item.source_bearing_active_nodes for item in summary)
                ),
                f"cover_{tag}_nonredundant_packets": scalar(
                    sum(item.nonredundant_packet_count for item in summary)
                ),
                f"cover_{tag}_source_union_count": scalar(
                    sum(item.source_union_count for item in summary)
                ),
                f"cover_{tag}_raw_paths": scalar(0),
                f"cover_{tag}_key_missing": scalar(
                    sum(view.permission_key_missing(mechanism) for view in views)
                ),
                f"cover_{tag}_full_access_bypass": scalar(
                    sum(view.is_full_access(mechanism) for view in views)
                ),
            })
        return aux

    def _read_cover_dense_masked(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        *,
        phase: str | None,
        return_routing_maps: bool,
        include_cover_diagnostics: bool,
        compiled_views: tuple[CompiledMechanismExecutionView, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, query_count = receivers.shape[:2]
        modules = int(encoded.module_present.shape[1])
        environment_count = int(encoded.env_coords.shape[1])
        receiver_group_access = None
        if compiled_views is not None:
            module_access, environment_access, receiver_group_access = self._compiled_query_accesses(
                plans, compiled_views, receivers
            )
        else:
            module_access, environment_access = self._query_accesses(
                plans, encoded, receivers, phase=phase
            )
        module_prior = module_access * (encoded.module_present > 0.5).to(module_access.dtype)[:, None, :]
        scale_rows = self._environment_scale_rows(encoded.coordinate_scale, batch)[:, 0, :]
        module_relative = (
            receivers[:, :, None, :] - encoded.module_centers[:, None, :, :]
        ) / scale_rows[:, None, None, :]
        module_sources = state["module_tokens"][:, None, :, :].expand(-1, query_count, -1, -1)
        module_globals = encoded.global_token[:, None, None, :].expand(-1, query_count, modules, -1)
        module_messages = self._mlp(self.query_module_message, torch.cat((
            module_sources, self.relative_fourier(module_relative), module_globals,
        ), dim=-1))
        active_count = encoded.module_present.sum(dim=1)[:, None, None]
        module_context = self.query_module_output(
            (module_messages * module_prior[..., None]).sum(dim=2) / (1.0 + active_count)
        )

        environment_prior, fallback = self._environment_access_prior(
            environment_access,
            encoded.env_weights,
            fallback_eligible=self._legacy_environment_fallback_mask(plans, device=receivers.device),
        )
        environment_relative = (
            receivers[:, :, None, :] - encoded.env_coords[:, None, :, :]
        ) / scale_rows[:, None, None, :]
        bias = self._mlp(self.env_geometry_bias, self.relative_fourier(environment_relative)).permute(0, 3, 1, 2)
        projected_query = self.env_attention.project_query(self.env_query(receiver_features))
        projected_key, projected_value = self.project_environment_sources(state["env_tokens"])
        scores = torch.matmul(projected_query, projected_key.transpose(-1, -2)) / (float(self.env_attention.head_dim) ** 0.5)
        scores = scores + bias
        support = environment_prior > 0
        empty_environment_rows = ~support.any(dim=-1)
        safe_prior = environment_prior.double().clamp_min(torch.finfo(torch.float64).tiny)
        log_prior = safe_prior.log().masked_fill(~support, -torch.inf)
        scores = scores.double() + log_prior[:, None, :, :]
        scores = torch.where(empty_environment_rows[:, None, :, None], torch.zeros_like(scores), scores)
        attention = torch.softmax(scores, dim=-1).to(projected_value.dtype)
        environment_context = self.env_attention.output(
            torch.matmul(attention, projected_value).transpose(1, 2).reshape(batch, query_count, self.hidden_dim)
        )
        environment_context = environment_context.masked_fill(empty_environment_rows[:, :, None], 0.0)
        aux = self._cover_query_work_aux(
            plans,
            encoded,
            receivers,
            module_access,
            environment_prior,
            batch * query_count * modules,
            batch * query_count * environment_count,
            (module_prior > 0).sum(),
            (environment_prior > 0).sum(),
            phase=phase,
            executor="dense_masked",
            include_diagnostics=include_cover_diagnostics,
            compiled_views=compiled_views,
            receiver_group_access=receiver_group_access,
        )
        aux["cover_environment_fallback_queries"] = fallback.sum().to(torch.long)
        if return_routing_maps:
            aux["dense_environment_attention"] = attention
        aux["dense_module_context_norm"] = torch.linalg.vector_norm(module_context, dim=-1)
        return module_context + environment_context, aux

    def _read_cover_rectangular_subset(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        plans: tuple[AdaptiveCoverPlan | MechanismPlan, ...],
        *,
        phase: str | None,
        return_routing_maps: bool,
        include_cover_diagnostics: bool,
        compiled_views: tuple[CompiledMechanismExecutionView, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        receiver_group_access = None
        if compiled_views is not None:
            module_access, environment_access, receiver_group_access = self._compiled_query_accesses(
                plans, compiled_views, receivers
            )
        else:
            module_access, environment_access = self._query_accesses(
                plans, encoded, receivers, phase=phase
            )
        module_prior = module_access * (encoded.module_present > 0.5).to(module_access.dtype)[:, None, :]
        environment_prior, fallback = self._environment_access_prior(
            environment_access,
            encoded.env_weights,
            fallback_eligible=self._legacy_environment_fallback_mask(plans, device=receivers.device),
        )
        module_context, module_actual = self._read_module_rectangular(
            state, encoded, receivers, module_access
        )
        environment_context, environment_actual = self._read_environment_rectangular(
            state, encoded, receivers, receiver_features, environment_prior
        )
        aux = self._cover_query_work_aux(
            plans,
            encoded,
            receivers,
            module_access,
            environment_prior,
            module_actual,
            environment_actual,
            (module_prior > 0).sum(),
            (environment_prior > 0).sum(),
            phase=phase,
            executor="rectangular_subset",
            include_diagnostics=include_cover_diagnostics,
            compiled_views=compiled_views,
            receiver_group_access=receiver_group_access,
        )
        aux["cover_environment_fallback_queries"] = fallback.sum().to(torch.long)
        if return_routing_maps:
            # The selected rectangle has already been reduced and normalized;
            # dense maps would duplicate the largest retained tensor.
            aux["cover_attention_map_omitted"] = torch.ones((), device=receivers.device, dtype=torch.long)
        aux["dense_module_context_norm"] = torch.linalg.vector_norm(module_context, dim=-1)
        return module_context + environment_context, aux

    def read(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
        interaction_context: InteractionContext | None = None,
        include_cover_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        plans = state.get("cover_plans")
        stored_context = state.get("cover_interaction_context")
        context = interaction_context if interaction_context is not None else stored_context
        phase = None if context is None else context.phase
        if plans is not None and len(plans) != int(receivers.shape[0]):
            raise ValueError("cover plans must align with the case batch")
        stored_views = state.get("cover_execution_views")
        compiled_views = None
        if (
            plans is not None
            and isinstance(stored_views, tuple)
            and len(stored_views) == len(plans)
            and all(
                isinstance(view, CompiledMechanismExecutionView)
                and view.matches(plan, phase=phase)
                for view, plan in zip(stored_views, plans, strict=True)
            )
        ):
            compiled_views = stored_views
        if compiled_views is not None:
            query_full_access = all(
                view.is_full_access("QM") and view.is_full_access("QE")
                for view in compiled_views
            )
            differentiable_query = any(
                view.has_differentiable_routing(("QM", "QE"))
                for view in compiled_views
            )
        else:
            query_full_access = plans is not None and all(
                self.is_full_access_plan(
                    plan,
                    encoded.module_present[case],
                    int(encoded.env_coords.shape[1]),
                    phase=phase,
                    mechanisms=("QM", "QE"),
                )
                for case, plan in enumerate(plans)
            )
            differentiable_query = plans is not None and any(
                self._plan_has_differentiable_routing(
                    plan, phase=phase, mechanisms=("QM", "QE")
                )
                for plan in plans
            )
        differentiable_full_access_query = query_full_access and differentiable_query
        if differentiable_full_access_query:
            # Keep exact native values and input/model derivatives. A detached
            # prepared state isolates this read surrogate from the separate
            # MM/ME/EM preparation bridge, so shared split gates receive one
            # gradient contribution from each declared mechanism path.
            native_context, native_aux = DensePairwiseField.read(
                self, state, encoded, receivers, receiver_features,
                return_routing_maps=return_routing_maps,
            )
            surrogate_state = dict(state)
            surrogate_state["module_tokens"] = state["module_tokens"].detach()
            surrogate_state["env_tokens"] = state["env_tokens"].detach()
            surrogate_context, surrogate_aux = self._read_cover_dense_masked(
                surrogate_state,
                encoded,
                receivers,
                receiver_features,
                plans,
                phase=phase,
                return_routing_maps=bool(return_routing_maps),
                include_cover_diagnostics=bool(include_cover_diagnostics),
                compiled_views=compiled_views,
            )
            query_routing = self._differentiable_plan_inputs(
                plans, phase=phase, mechanisms=("QM", "QE")
            )
            if query_routing:
                context = _NativeForwardMaskedRoutingGradient.apply(
                    native_context, surrogate_context, *query_routing
                )
            else:
                context = native_context
            batch_size, query_count = receivers.shape[:2]
            native_module_rows = (
                int(batch_size) * int(query_count) * int(encoded.module_present.shape[1])
            )
            native_environment_rows = (
                int(batch_size) * int(query_count) * int(encoded.env_coords.shape[1])
            )
            return context, self._bridge_query_aux(
                native_aux,
                surrogate_aux,
                native_module_rows=native_module_rows,
                native_environment_rows=native_environment_rows,
            )
        if differentiable_query:
            # A hard-forward partial plan changes the output by design. Route
            # it through the dense-masked reference rather than the global
            # all-access deployment bypass, including in full_access mode.
            return self._read_cover_dense_masked(
                state,
                encoded,
                receivers,
                receiver_features,
                plans,
                phase=phase,
                return_routing_maps=bool(return_routing_maps),
                include_cover_diagnostics=bool(include_cover_diagnostics),
                compiled_views=compiled_views,
            )
        if (
            (self.cover_mode == "full_access")
            or plans is None
            or (
                query_full_access
                and not differentiable_query
                and (self.cover_mode != "external" or compiled_views is not None)
            )
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
            if not include_cover_diagnostics:
                module_sources = module_rows
                aux.update({
                    "cover_qm_unique_rows": torch.tensor(module_sources, device=receivers.device),
                    "cover_qe_unique_rows": torch.tensor(environment_rows, device=receivers.device),
                    "cover_qm_unique_source_receiver_pairs": torch.tensor(module_sources, device=receivers.device),
                    "cover_qe_unique_source_receiver_pairs": torch.tensor(environment_rows, device=receivers.device),
                    "cover_qm_actual_rows": torch.tensor(module_executed_rows, device=receivers.device),
                    "cover_qe_actual_rows": torch.tensor(environment_rows, device=receivers.device),
                    "cover_qm_executed_rows": torch.tensor(module_executed_rows, device=receivers.device),
                    "cover_qe_executed_rows": torch.tensor(environment_rows, device=receivers.device),
                    "cover_qm_padded_rows": torch.tensor(module_executed_rows - module_sources, device=receivers.device),
                    "cover_qe_padded_rows": torch.zeros((), device=receivers.device, dtype=torch.long),
                    "cover_qm_rectangular_rows": torch.tensor(module_executed_rows, device=receivers.device),
                    "cover_qe_rectangular_rows": torch.tensor(environment_rows, device=receivers.device),
                    "cover_query_count": torch.tensor(query_count * int(receivers.shape[0]), device=receivers.device),
                    "cover_environment_fallback_queries": torch.zeros((), device=receivers.device, dtype=torch.long),
                    "cover_path_diagnostics_available": torch.zeros((), device=receivers.device, dtype=torch.long),
                    "cover_executor_dense_fallback": torch.ones((), device=receivers.device, dtype=torch.long),
                    "cover_executor_native_dense": torch.ones((), device=receivers.device, dtype=torch.long),
                })
                for mechanism in ("QM", "QE"):
                    tag = mechanism.lower()
                    aux[f"cover_{tag}_executor_dense_masked"] = torch.zeros((), device=receivers.device, dtype=torch.long)
                    aux[f"cover_{tag}_executor_rectangular_subset"] = torch.zeros((), device=receivers.device, dtype=torch.long)
                    aux[f"cover_{tag}_executor_packed"] = torch.zeros((), device=receivers.device, dtype=torch.long)
                    aux[f"cover_{tag}_executor_full_access"] = torch.ones((), device=receivers.device, dtype=torch.long)
                    aux[f"cover_{tag}_dense_fallback"] = torch.ones((), device=receivers.device, dtype=torch.long)
                    aux[f"cover_{tag}_full_access_bypass"] = torch.ones((), device=receivers.device, dtype=torch.long)
                    aux[f"cover_{tag}_key_missing"] = torch.tensor(
                        sum(
                            int(isinstance(plan, MechanismPlan) and plan.permission_status(
                                mechanism, phase=phase
                            ) == "full_access_bypass_missing_key")
                            for plan in plans
                        ),
                        device=receivers.device,
                        dtype=torch.long,
                    )
                return context, aux
            if compiled_views is not None and query_full_access:
                aux.update(self._compiled_full_access_query_aux(
                    compiled_views, encoded, receivers
                ))
                return context, aux
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
            receiver_group_access = None
            if compiled_views is not None:
                module_access, environment_access, receiver_group_access = self._compiled_query_accesses(
                    plans, compiled_views, receivers
                )
            else:
                module_access, environment_access = self._query_accesses(
                    plans, encoded, receivers, phase=phase
                )
            module_prior = module_access * (encoded.module_present > 0.5).to(module_access.dtype)[:, None, :]
            environment_prior, fallback = self._environment_access_prior(
                environment_access,
                encoded.env_weights,
                fallback_eligible=self._legacy_environment_fallback_mask(plans, device=receivers.device),
            )
            aux.update(self._cover_query_work_aux(
                plans,
                encoded,
                receivers,
                module_access,
                environment_prior,
                module_executed_rows,
                environment_rows,
                (module_prior > 0).sum(),
                (environment_prior > 0).sum(),
                phase=phase,
                executor="full_access",
                compiled_views=compiled_views,
                receiver_group_access=receiver_group_access,
            ))
            aux["cover_environment_fallback_queries"] = fallback.sum().to(torch.long)
            aux["cover_executor_dense_fallback"] = torch.ones((), device=receivers.device, dtype=torch.long)
            aux["cover_executor_native_dense"] = torch.ones((), device=receivers.device, dtype=torch.long)
            return context, aux
        if self.cover_executor == "dense_masked":
            return self._read_cover_dense_masked(
                state, encoded, receivers, receiver_features, plans,
                phase=phase, return_routing_maps=bool(return_routing_maps),
                include_cover_diagnostics=bool(include_cover_diagnostics),
                compiled_views=compiled_views,
            )
        if self.cover_executor == "rectangular_subset":
            return self._read_cover_rectangular_subset(
                state, encoded, receivers, receiver_features, plans,
                phase=phase, return_routing_maps=bool(return_routing_maps),
                include_cover_diagnostics=bool(include_cover_diagnostics),
                compiled_views=compiled_views,
            )
        module_pairs: list[PackedPairs] = []
        environment_pairs: list[PackedPairs] = []
        ledgers: list[CoverPairLedger] = []
        k_case: list[int] = []
        transition_case: list[int] = []
        for case, plan in enumerate(plans):
            qm, qe, ledger = compile_cover_pairs(
                plan, receivers[case], module_present=encoded.module_present[case],
                environment_weights=encoded.env_weights[case],
                phase=phase,
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
        typed_cases = torch.tensor(
            [isinstance(plan, MechanismPlan) for plan in plans],
            device=receivers.device,
            dtype=torch.bool,
        )
        if bool(typed_cases.any()):
            environment_rows = torch.zeros(
                (len(plans), int(receivers.shape[1])),
                device=receivers.device,
                dtype=torch.bool,
            )
            if qe.unique_pair_count:
                environment_rows[qe.batch_index, qe.receiver_index] = True
            environment_context = environment_context.masked_fill(
                (typed_cases[:, None] & ~environment_rows)[:, :, None], 0.0
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
            "cover_executor_dense_masked": torch.tensor(0, device=receivers.device),
            "cover_executor_rectangular_subset": torch.tensor(0, device=receivers.device),
            "cover_executor_packed": torch.tensor(1, device=receivers.device),
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
        module_access, environment_access = self._query_accesses(
            plans, encoded, receivers, phase=phase
        )
        module_prior = module_access * (encoded.module_present > 0.5).to(module_access.dtype)[:, None, :]
        environment_prior, fallback = self._environment_access_prior(
            environment_access,
            encoded.env_weights,
            fallback_eligible=self._legacy_environment_fallback_mask(plans, device=receivers.device),
        )
        aux.update(self._cover_query_work_aux(
            plans,
            encoded,
            receivers,
            module_access,
            environment_prior,
            qm.unique_pair_count,
            qe.unique_pair_count,
            int((module_prior > 0).sum()),
            int((environment_prior > 0).sum()),
            phase=phase,
            executor="packed",
            include_diagnostics=include_cover_diagnostics,
        ))
        aux["cover_environment_fallback_queries"] = fallback.sum().to(torch.long)
        return module_context + environment_context, aux


__all__ = ["AdaptiveCoverPairwiseField"]
