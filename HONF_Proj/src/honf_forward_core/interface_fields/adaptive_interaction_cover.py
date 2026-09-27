"""Case-local receiver cover and exact fine-pair work accounting.

This module owns the geometry-only candidate hierarchy and the live access
algebra. It does not infer physical effects from its support: oracle proposals
and an input-only amortizer must be judged against separate response evidence.
The hierarchy is built once per case and kept fixed over a local design trust
region; rebuilding it is a separately measured boundary event.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

import torch

from .routing_index.types import PackedPairs

InteractionMechanism = Literal["MM", "ME", "EM", "QM", "QE"]
INTERACTION_MECHANISMS: tuple[InteractionMechanism, ...] = ("MM", "ME", "EM", "QM", "QE")


@dataclass(frozen=True)
class InteractionContext:
    """Explicit physical phase and receiver role for a native policy call."""

    phase: str | None = None
    receiver_role: str | None = None

    def __post_init__(self) -> None:
        for name in ("phase", "receiver_role"):
            value = getattr(self, name)
            if value is not None:
                value = str(value).strip()
                if not value:
                    raise ValueError(f"interaction {name} cannot be empty")
                object.__setattr__(self, name, value)


@dataclass(frozen=True, order=True)
class InteractionPermissionKey:
    """A typed transport permission, optionally scoped to a physical phase."""

    mechanism: InteractionMechanism
    phase: str | None = None

    def __post_init__(self) -> None:
        mechanism = str(self.mechanism).upper()
        if mechanism not in INTERACTION_MECHANISMS:
            raise ValueError(f"unsupported interaction mechanism {self.mechanism!r}")
        phase = None if self.phase is None else str(self.phase).strip()
        if phase == "":
            raise ValueError("permission phase cannot be an empty string")
        object.__setattr__(self, "mechanism", mechanism)
        object.__setattr__(self, "phase", phase)

    @property
    def canonical_name(self) -> str:
        return self.mechanism if self.phase is None else f"{self.phase}:{self.mechanism}"

    @classmethod
    def coerce(cls, value: InteractionPermissionKey | str | tuple[str, str | None]) -> InteractionPermissionKey:
        if isinstance(value, cls):
            return value
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("permission tuple keys must be (mechanism, phase)")
            return cls(str(value[0]), value[1])
        text = str(value).strip()
        if ":" in text:
            phase, mechanism = text.split(":", 1)
            return cls(mechanism, phase)
        if "/" in text:
            phase, mechanism = text.split("/", 1)
            return cls(mechanism, phase)
        return cls(text)


@dataclass(frozen=True)
class CoverFrontierSummary:
    """Recursive receiver-frontier and unique-source work for one typed route."""

    mechanism: InteractionMechanism
    phase: str | None
    candidate_node_count: int
    raw_active_frontier_nodes: int
    source_bearing_active_nodes: int
    nonredundant_packet_count: int
    source_union_count: int
    unique_source_receiver_pairs: int

    def as_dict(self, *, prefix: str = "cover") -> dict[str, int | str]:
        tag = self.mechanism.lower()
        if self.phase is not None:
            tag = f"{self.phase.lower()}_{tag}"
        return {
            f"{prefix}_{tag}_candidate_nodes": self.candidate_node_count,
            f"{prefix}_{tag}_raw_active_frontier_nodes": self.raw_active_frontier_nodes,
            f"{prefix}_{tag}_source_bearing_active_nodes": self.source_bearing_active_nodes,
            f"{prefix}_{tag}_nonredundant_packets": self.nonredundant_packet_count,
            f"{prefix}_{tag}_source_union_count": self.source_union_count,
            f"{prefix}_{tag}_unique_source_receiver_pairs": self.unique_source_receiver_pairs,
        }


def endpoint_smoothstep(value: torch.Tensor) -> torch.Tensor:
    """Continuous exact-zero/one access gate with flat endpoint derivatives."""

    clipped = value.clamp(0.0, 1.0)
    return clipped.square() * (3.0 - 2.0 * clipped)


@dataclass(frozen=True)
class ReceiverAnchorUniverse:
    """One case's input-only physical receiver anchors and declared measure."""

    coordinates: torch.Tensor  # [A,d], in the case adapter's physical frame
    weights: torch.Tensor  # [A], positive receiver measure
    roles: torch.Tensor  # [A], case-owned integer role codes
    coordinate_scale: torch.Tensor  # [d], positive physical normalization

    def __post_init__(self) -> None:
        anchors = self.coordinates
        if anchors.ndim != 2 or anchors.shape[0] < 1:
            raise ValueError("receiver anchors must have shape [A,d] with A positive")
        if self.weights.shape != anchors.shape[:1] or self.roles.shape != anchors.shape[:1]:
            raise ValueError("receiver weights and roles must align with anchors")
        if self.coordinate_scale.shape != anchors.shape[1:]:
            raise ValueError("coordinate scale must have shape [d]")
        if (
            not bool(torch.isfinite(anchors).all())
            or not bool(torch.isfinite(self.weights).all())
            or not bool(torch.isfinite(self.coordinate_scale).all())
        ):
            raise ValueError("receiver anchors, weights, and coordinate scales must be finite")
        if bool((self.weights <= 0).any()) or bool((self.coordinate_scale <= 0).any()):
            raise ValueError("receiver weights and coordinate scales must be positive")
        if self.roles.dtype not in (torch.int32, torch.int64):
            raise TypeError("receiver roles must have integer dtype")


@dataclass(frozen=True)
class CandidateNode:
    anchor_indices: tuple[int, ...]
    left: int | None = None
    right: int | None = None
    split_axis: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


@dataclass(frozen=True)
class CaseLocalReceiverTree:
    """A bounded index over a fixed physical receiver-anchor universe."""

    universe: ReceiverAnchorUniverse
    nodes: tuple[CandidateNode, ...]
    overlap_fraction: float
    capacity_saturated: bool

    @classmethod
    def build(
        cls,
        universe: ReceiverAnchorUniverse,
        *,
        max_nodes: int = 127,
        min_leaf_anchors: int = 4,
        overlap_fraction: float = 0.06,
    ) -> CaseLocalReceiverTree:
        if max_nodes < 1 or min_leaf_anchors < 1:
            raise ValueError("tree capacity and minimum leaf size must be positive")
        if not 0.0 < overlap_fraction < 1.0:
            raise ValueError("receiver overlap fraction must be between zero and one")
        nodes: list[CandidateNode] = []
        saturated = False
        coordinates = (universe.coordinates / universe.coordinate_scale).detach().cpu()
        weights = universe.weights.detach().cpu()
        roles = universe.roles.detach().cpu()

        def refine(index: int, indices: tuple[int, ...]) -> None:
            nonlocal saturated
            if len(indices) <= min_leaf_anchors:
                return
            if len(nodes) + 2 > max_nodes:
                saturated = True
                return
            subset = coordinates[list(indices)]
            extent = subset.max(dim=0).values - subset.min(dim=0).values
            axis = int(torch.argmax(extent).item())
            if float(extent[axis]) <= 0.0:
                return
            # Geometry and role break ties. Physical source IDs and output
            # values are never consulted in candidate construction.
            ordered = sorted(
                indices,
                key=lambda item: (
                    float(coordinates[item, axis]),
                    *tuple(float(value) for value in coordinates[item]),
                    int(roles[item]),
                ),
            )
            half_mass = float(weights[list(ordered)].sum()) / 2.0
            accumulated = 0.0
            cut = 1
            for position, anchor_id in enumerate(ordered[:-1], start=1):
                accumulated += float(weights[anchor_id])
                cut = position
                if accumulated >= half_mass:
                    break
            left_ids, right_ids = tuple(ordered[:cut]), tuple(ordered[cut:])
            left, right = len(nodes), len(nodes) + 1
            nodes.extend((CandidateNode(left_ids), CandidateNode(right_ids)))
            nodes[index] = CandidateNode(indices, left, right, axis)
            refine(left, left_ids)
            refine(right, right_ids)

        root_ids = tuple(range(int(universe.coordinates.shape[0])))
        nodes.append(CandidateNode(root_ids))
        refine(0, root_ids)
        return cls(universe, tuple(nodes), overlap_fraction, saturated)

    def access(self, queries: torch.Tensor, split_gates: torch.Tensor) -> torch.Tensor:
        """Return nonnegative receiver access [Q,N] with row sum one.

        Child weights overlap continuously at the geometric split. At gate
        zero the children are absent; at gate one the parent contribution is
        absent. A mixed gate includes both in the exact active count.
        """

        if queries.ndim != 2 or queries.shape[1] != self.universe.coordinates.shape[1]:
            raise ValueError("queries must have shape [Q,d] in the anchor frame")
        if split_gates.shape != (len(self.nodes),):
            raise ValueError("split gates must have one value per candidate node")
        if not bool(torch.isfinite(split_gates).all()) or bool(((split_gates < 0) | (split_gates > 1)).any()):
            raise ValueError("split gates must be finite and in [0,1]")
        if queries.device != split_gates.device or queries.device != self.universe.coordinates.device:
            raise ValueError("queries, gates, and anchors must use one device")
        result: list[torch.Tensor | None] = [None] * len(self.nodes)

        def descend(index: int, incoming: torch.Tensor) -> None:
            node = self.nodes[index]
            if node.is_leaf:
                result[index] = incoming
                return
            assert node.left is not None and node.right is not None and node.split_axis is not None
            split_gate = split_gates[index]
            smooth_gate = endpoint_smoothstep(split_gate)
            if split_gate.requires_grad:
                # A hard-forward organizer gate needs a useful local
                # surrogate at 0/1. Preserve the exact smoothstep forward and
                # its derivative for interior gates, while using identity
                # backward only at exact endpoints. This is an explicitly
                # surrogate derivative through a discrete topology choice.
                endpoint = ((split_gate.detach() == 0.0) | (split_gate.detach() == 1.0))
                surrogate_gate = torch.where(endpoint, split_gate, smooth_gate)
                gate = surrogate_gate + (smooth_gate - surrogate_gate).detach()
            else:
                gate = smooth_gate
            result[index] = incoming * (1.0 - gate)
            left_ids = self.nodes[node.left].anchor_indices
            right_ids = self.nodes[node.right].anchor_indices
            axis = node.split_axis
            anchor_axis = self.universe.coordinates[:, axis]
            left_center = anchor_axis[list(left_ids)].mean()
            right_center = anchor_axis[list(right_ids)].mean()
            boundary = (left_center + right_center) / 2.0
            overlap = self.overlap_fraction * self.universe.coordinate_scale[axis]
            left_weight = endpoint_smoothstep((boundary + overlap / 2.0 - queries[:, axis]) / overlap)
            descend(node.left, incoming * gate * left_weight)
            descend(node.right, incoming * gate * (1.0 - left_weight))

        descend(0, queries.new_ones((int(queries.shape[0]),)))
        return torch.stack([value for value in result if value is not None], dim=1)


@dataclass(frozen=True)
class CoverAccess:
    receiver_group: torch.Tensor  # [Q,N]
    module_source: torch.Tensor  # [Q,M], after overlapping path deduplication
    environment_source: torch.Tensor  # [Q,E]
    active_groups_on_anchors: int
    transition_nodes: int
    capacity_saturated: bool


@dataclass(frozen=True)
class AdaptiveCoverPlan:
    """One frozen case plan; learned scores must be converted before evaluation."""

    tree: CaseLocalReceiverTree
    split_gates: torch.Tensor  # [N]
    module_membership: torch.Tensor  # [N,M], exact-zero source support
    environment_membership: torch.Tensor  # [N,E]

    def __post_init__(self) -> None:
        count = len(self.tree.nodes)
        if self.split_gates.shape != (count,):
            raise ValueError("split gates must match candidate-node capacity")
        for name, tensor in (
            ("module", self.module_membership),
            ("environment", self.environment_membership),
        ):
            if tensor.ndim != 2 or tensor.shape[0] != count:
                raise ValueError(f"{name} membership must have shape [N,S]")
            if not bool(torch.isfinite(tensor).all()) or bool(((tensor < 0) | (tensor > 1)).any()):
                raise ValueError(f"{name} membership must be finite and in [0,1]")
        if self.module_membership.device != self.split_gates.device or self.environment_membership.device != self.split_gates.device:
            raise ValueError("cover gates and memberships must use one device")

    @classmethod
    def full_access(
        cls, tree: CaseLocalReceiverTree, module_present: torch.Tensor, environment_count: int
    ) -> AdaptiveCoverPlan:
        anchors = tree.universe.coordinates
        count = len(tree.nodes)
        gates = anchors.new_zeros(count)
        if module_present.ndim != 1:
            raise ValueError("module validity must have shape [M]")
        module = (module_present > 0.5).to(anchors.dtype)[None, :].expand(count, -1).clone()
        environment = anchors.new_ones(count, environment_count)
        return cls(tree, gates, module, environment)

    def with_split(self, node: int, gate: float | torch.Tensor) -> AdaptiveCoverPlan:
        if self.tree.nodes[node].is_leaf:
            raise ValueError("a leaf has no split")
        replacement = torch.as_tensor(gate, device=self.split_gates.device, dtype=self.split_gates.dtype)
        gates = torch.cat((self.split_gates[:node], replacement.reshape(1), self.split_gates[node + 1 :]))
        return replace(self, split_gates=gates)

    def access(self, queries: torch.Tensor) -> CoverAccess:
        alpha = self.tree.access(queries, self.split_gates)
        module = alpha @ self.module_membership
        environment = alpha @ self.environment_membership
        transitions = sum(
            0.0 < float(self.split_gates[index].detach()) < 1.0
            for index, node in enumerate(self.tree.nodes) if not node.is_leaf
        )
        return CoverAccess(alpha, module, environment, self.active_group_count(), transitions, self.tree.capacity_saturated)

    def active_group_count(self) -> int:
        """Count source-bearing nodes reached on the declared anchor universe.

        The candidate hierarchy has fixed capacity; only these active nodes
        incur per-group organization/control work. This avoids allocating an
        anchor-by-environment access matrix just to count active groups.
        """

        on_anchors = self.tree.access(self.tree.universe.coordinates, self.split_gates)
        source_nonempty = (self.module_membership > 0).any(dim=1) | (
            self.environment_membership > 0
        ).any(dim=1)
        return int(((on_anchors > 0).any(dim=0) & source_nonempty).sum())


@dataclass(frozen=True)
class MechanismPlan:
    """Immutable typed permissions over a case-local receiver tree.

    ``permissions`` may contain any subset of the five initial transport
    mechanisms. A missing mechanism key means that mechanism is explicitly
    full-access at execution time; it is never interpreted as an empty mask.
    A phase-specific key takes precedence over a mechanism-wide key, which in
    turn takes precedence over the recorded full-access default.
    """

    tree: CaseLocalReceiverTree
    split_gates: torch.Tensor
    module_present: torch.Tensor
    environment_count: int
    permissions: Mapping[
        InteractionPermissionKey | str | tuple[str, str | None], torch.Tensor
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_count = len(self.tree.nodes)
        if self.split_gates.shape != (node_count,):
            raise ValueError("split gates must match candidate-node capacity")
        if not bool(torch.isfinite(self.split_gates).all()) or bool(
            ((self.split_gates < 0) | (self.split_gates > 1)).any()
        ):
            raise ValueError("split gates must be finite and in [0,1]")
        if self.module_present.ndim != 1:
            raise ValueError("module presence must have shape [M]")
        if self.environment_count < 1:
            raise ValueError("environment source count must be positive")
        if not bool(torch.isfinite(self.module_present).all()):
            raise ValueError("module presence must be finite")
        if self.split_gates.device != self.module_present.device:
            raise ValueError("split gates and source validity must use one device")
        if self.split_gates.device != self.tree.universe.coordinates.device:
            raise ValueError("cover tensors and receiver tree must use one device")
        normalized: dict[InteractionPermissionKey, torch.Tensor] = {}
        for raw_key, raw_value in self.permissions.items():
            key = InteractionPermissionKey.coerce(raw_key)
            if key in normalized:
                raise ValueError(f"duplicate normalized permission key {key.canonical_name!r}")
            source_count = self._source_count(key.mechanism)
            value = raw_value
            if value.shape != (node_count, source_count):
                raise ValueError(
                    f"permission {key.canonical_name} must have shape [{node_count},{source_count}]"
                )
            if value.device != self.split_gates.device:
                raise ValueError("permission tensors, split gates, and receiver tree must share a device")
            if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
                raise ValueError("permissions must be finite and in [0,1]")
            normalized[key] = value.clone()
        object.__setattr__(self, "split_gates", self.split_gates.clone())
        object.__setattr__(self, "module_present", self.module_present.clone())
        object.__setattr__(self, "permissions", MappingProxyType(normalized))

    def _source_count(self, mechanism: InteractionMechanism) -> int:
        return int(self.module_present.numel()) if mechanism in {"MM", "EM", "QM"} else int(self.environment_count)

    @classmethod
    def full_access(
        cls,
        tree: CaseLocalReceiverTree,
        module_present: torch.Tensor,
        environment_count: int,
        *,
        phase: str | None = None,
    ) -> MechanismPlan:
        """Build a fully explicit all-access plan for all five mechanisms."""

        gates = tree.universe.coordinates.new_zeros((len(tree.nodes),))
        module = (module_present > 0.5).to(gates.dtype)[None, :].expand(len(tree.nodes), -1).clone()
        environment = gates.new_ones((len(tree.nodes), int(environment_count)))
        permissions = {
            InteractionPermissionKey(mechanism, phase): (
                module if mechanism in {"MM", "EM", "QM"} else environment
            )
            for mechanism in INTERACTION_MECHANISMS
        }
        return cls(tree, gates, module_present, int(environment_count), permissions)

    @classmethod
    def from_legacy(cls, plan: AdaptiveCoverPlan, module_present: torch.Tensor | None = None) -> MechanismPlan:
        """Convert the historical tied module/environment mask explicitly."""

        if module_present is None:
            module_present = plan.module_membership[0]
        permissions = {
            mechanism: (
                plan.module_membership
                if mechanism in {"MM", "EM", "QM"}
                else plan.environment_membership
            )
            for mechanism in INTERACTION_MECHANISMS
        }
        return cls(
            plan.tree,
            plan.split_gates,
            module_present,
            int(plan.environment_membership.shape[1]),
            permissions,
        )

    @property
    def module_membership(self) -> torch.Tensor:
        """Compatibility view of the query-module permission."""

        return self.permission_matrix("QM")

    @property
    def environment_membership(self) -> torch.Tensor:
        """Compatibility view of the query-environment permission."""

        return self.permission_matrix("QE")

    @property
    def explicit_bypass_keys(self) -> tuple[str, ...]:
        """Mechanisms whose absent key compiles to the declared full default."""

        return tuple(
            mechanism
            for mechanism in INTERACTION_MECHANISMS
            if InteractionPermissionKey(mechanism) not in self.permissions
        )

    @property
    def explicit_phase_keys(self) -> tuple[str, ...]:
        return tuple(sorted(key.canonical_name for key in self.permissions if key.phase is not None))

    def permission_status(self, mechanism: str, *, phase: str | None = None) -> str:
        key = InteractionPermissionKey.coerce((mechanism, phase))
        if key in self.permissions:
            return "phase_permission" if phase is not None else "mechanism_permission"
        if phase is not None and InteractionPermissionKey(mechanism) in self.permissions:
            return "mechanism_permission_inherited"
        return "full_access_bypass_missing_key"

    def permission_matrix(
        self,
        mechanism: str,
        source_count: int | None = None,
        *,
        phase: str | None = None,
        module_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Resolve a typed mask and apply module padding validity."""

        base_key = InteractionPermissionKey(mechanism)
        selected_key = InteractionPermissionKey(mechanism, phase)
        key = selected_key if selected_key in self.permissions else base_key
        expected_count = self._source_count(base_key.mechanism)
        if source_count is not None and int(source_count) != expected_count:
            raise ValueError(f"{mechanism} source count must be {expected_count}")
        if key in self.permissions:
            value = self.permissions[key]
        elif base_key.mechanism in {"MM", "EM", "QM"}:
            value = self.split_gates.new_ones((len(self.tree.nodes), expected_count))
        else:
            value = self.split_gates.new_ones((len(self.tree.nodes), expected_count))
        if base_key.mechanism in {"MM", "EM", "QM"}:
            valid = self.module_present if module_present is None else module_present
            if valid.shape != (expected_count,):
                raise ValueError("module validity does not match the typed module source axis")
            value = value * (valid > 0.5).to(value.dtype)[None, :]
        return value

    def access_for(
        self,
        mechanism: str,
        receivers: torch.Tensor,
        source_count: int | None = None,
        *,
        phase: str | None = None,
        module_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one typed receiver/source access matrix ``[Q,S]``."""

        alpha = self.tree.access(receivers, self.split_gates)
        membership = self.permission_matrix(
            mechanism, source_count, phase=phase, module_present=module_present
        )
        return alpha @ membership

    def access(self, queries: torch.Tensor) -> CoverAccess:
        """Compatibility view with QM/QE sources for existing diagnostics."""

        alpha = self.tree.access(queries, self.split_gates)
        module = self.access_for("QM", queries)
        environment = self.access_for("QE", queries)
        transitions = sum(
            0.0 < float(self.split_gates[index].detach()) < 1.0
            for index, node in enumerate(self.tree.nodes)
            if not node.is_leaf
        )
        return CoverAccess(
            alpha,
            module,
            environment,
            self.active_group_count(),
            transitions,
            self.tree.capacity_saturated,
        )

    def with_split(self, node: int, gate: float | torch.Tensor) -> MechanismPlan:
        if self.tree.nodes[node].is_leaf:
            raise ValueError("a leaf has no split")
        replacement = torch.as_tensor(gate, device=self.split_gates.device, dtype=self.split_gates.dtype)
        gates = torch.cat((self.split_gates[:node], replacement.reshape(1), self.split_gates[node + 1 :]))
        return replace(self, split_gates=gates)

    def with_permission(
        self,
        mechanism: str,
        membership: torch.Tensor,
        *,
        phase: str | None = None,
    ) -> MechanismPlan:
        key = InteractionPermissionKey(mechanism, phase)
        permissions = dict(self.permissions)
        permissions[key] = membership
        return replace(self, permissions=permissions)

    def active_node_mask(self, receivers: torch.Tensor | None = None) -> torch.Tensor:
        if receivers is None:
            receivers = self.tree.universe.coordinates
        return (self.tree.access(receivers, self.split_gates) > 0).any(dim=0)

    def active_group_count(self) -> int:
        active = self.active_node_mask()
        source_bearing = torch.zeros_like(active)
        for mechanism in INTERACTION_MECHANISMS:
            source_bearing |= (self.permission_matrix(mechanism) > 0).any(dim=1)
        return int((active & source_bearing).sum())

    def is_full_access(
        self,
        *,
        phase: str | None = None,
        mechanisms: tuple[InteractionMechanism, ...] = INTERACTION_MECHANISMS,
    ) -> bool:
        active = self.active_node_mask()
        if not bool(active.any()):
            return False
        for mechanism in mechanisms:
            matrix = self.permission_matrix(mechanism, phase=phase)
            expected = (
                (self.module_present > 0.5).to(matrix.dtype)
                if mechanism in {"MM", "EM", "QM"}
                else matrix.new_ones(matrix.shape[1])
            )
            if not torch.equal(matrix[active], expected[None, :].expand(int(active.sum()), -1)):
                return False
        return True

    def frontier_summary(
        self,
        mechanism: str,
        phase: str | None = None,
        *,
        receivers: torch.Tensor | None = None,
    ) -> CoverFrontierSummary:
        """Count recursive frontier nodes, equivalent packets, and source work."""

        if receivers is None:
            receivers = self.tree.universe.coordinates
        alpha = self.tree.access(receivers, self.split_gates)
        active_nodes = (alpha > 0).any(dim=0)
        membership = self.permission_matrix(mechanism, phase=phase)
        has_sources = (membership > 0).any(dim=1)
        active_bearing = active_nodes & has_sources
        active_bearing_cpu = active_bearing.detach().cpu().tolist()
        membership_cpu = membership.detach().cpu().tolist()
        signatures = {
            tuple(float(value) for value in row)
            for row, is_active in zip(membership_cpu, active_bearing_cpu)
            if is_active
        }
        effective = alpha @ membership
        return CoverFrontierSummary(
            InteractionPermissionKey(mechanism, phase).mechanism,
            phase,
            len(self.tree.nodes),
            int(active_nodes.sum()),
            int(active_bearing.sum()),
            len(signatures),
            int((effective > 0).any(dim=0).sum()),
            int((effective > 0).sum()),
        )

    def canonical_hash(self) -> str:
        """Return a device-independent hash of geometry and every effective permission."""

        digest = hashlib.sha256()

        def add_tensor(name: str, tensor: torch.Tensor) -> None:
            value = tensor.detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())

        universe = self.tree.universe
        for name, tensor in (
            ("anchor_coordinates", universe.coordinates),
            ("anchor_weights", universe.weights),
            ("anchor_roles", universe.roles),
            ("coordinate_scale", universe.coordinate_scale),
            ("module_present", self.module_present),
            ("split_gates", self.split_gates),
        ):
            add_tensor(name, tensor)
        digest.update(json.dumps(
            {
                "nodes": [
                    (node.anchor_indices, node.left, node.right, node.split_axis)
                    for node in self.tree.nodes
                ],
                "overlap_fraction": self.tree.overlap_fraction,
                "capacity_saturated": self.tree.capacity_saturated,
                "environment_count": self.environment_count,
                "default_permission": "full_access_bypass",
                "explicit_permission_keys": sorted(
                    key.canonical_name for key in self.permissions
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        phases = sorted({key.phase for key in self.permissions if key.phase is not None})
        contexts = [(mechanism, None) for mechanism in INTERACTION_MECHANISMS]
        contexts.extend((mechanism, phase) for phase in phases for mechanism in INTERACTION_MECHANISMS)
        for mechanism, phase in contexts:
            key = InteractionPermissionKey(mechanism, phase)
            digest.update(key.canonical_name.encode("utf-8"))
            add_tensor(key.canonical_name, self.permission_matrix(mechanism, phase=phase))
        return digest.hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible, stable serialization of this plan."""

        universe = self.tree.universe
        return {
            "schema": "honf-mechanism-plan-v1",
            "default_permission": "full_access_bypass",
            "explicit_bypass_keys": list(self.explicit_bypass_keys),
            "tree": {
                "coordinates": universe.coordinates.detach().cpu().tolist(),
                "weights": universe.weights.detach().cpu().tolist(),
                "roles": universe.roles.detach().cpu().tolist(),
                "coordinate_scale": universe.coordinate_scale.detach().cpu().tolist(),
                "nodes": [
                    {
                        "anchor_indices": list(node.anchor_indices),
                        "left": node.left,
                        "right": node.right,
                        "split_axis": node.split_axis,
                    }
                    for node in self.tree.nodes
                ],
                "overlap_fraction": self.tree.overlap_fraction,
                "capacity_saturated": self.tree.capacity_saturated,
            },
            "split_gates": self.split_gates.detach().cpu().tolist(),
            "module_present": self.module_present.detach().cpu().tolist(),
            "environment_count": self.environment_count,
            "permissions": {
                key.canonical_name: value.detach().cpu().tolist()
                for key, value in sorted(self.permissions.items(), key=lambda item: item[0].canonical_name)
            },
            "canonical_hash": self.canonical_hash(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> MechanismPlan:
        if payload.get("schema") != "honf-mechanism-plan-v1":
            raise ValueError("unsupported mechanism plan serialization schema")
        tree_payload = payload.get("tree")
        if not isinstance(tree_payload, Mapping):
            raise TypeError("serialized mechanism plan has no tree record")
        target_device = torch.device(device)
        universe = ReceiverAnchorUniverse(
            coordinates=torch.as_tensor(tree_payload["coordinates"], device=target_device, dtype=dtype),
            weights=torch.as_tensor(tree_payload["weights"], device=target_device, dtype=dtype),
            roles=torch.as_tensor(tree_payload["roles"], device=target_device, dtype=torch.long),
            coordinate_scale=torch.as_tensor(tree_payload["coordinate_scale"], device=target_device, dtype=dtype),
        )
        nodes = tuple(
            CandidateNode(
                tuple(int(value) for value in node["anchor_indices"]),
                None if node["left"] is None else int(node["left"]),
                None if node["right"] is None else int(node["right"]),
                None if node["split_axis"] is None else int(node["split_axis"]),
            )
            for node in tree_payload["nodes"]
        )
        tree = CaseLocalReceiverTree(
            universe,
            nodes,
            float(tree_payload["overlap_fraction"]),
            bool(tree_payload["capacity_saturated"]),
        )
        permissions_payload = payload.get("permissions", {})
        if not isinstance(permissions_payload, Mapping):
            raise TypeError("serialized permissions must be a mapping")
        result = cls(
            tree,
            torch.as_tensor(payload["split_gates"], device=target_device, dtype=dtype),
            torch.as_tensor(payload["module_present"], device=target_device, dtype=dtype),
            int(payload["environment_count"]),
            {
                str(key): torch.as_tensor(value, device=target_device, dtype=dtype)
                for key, value in permissions_payload.items()
            },
        )
        declared_bypasses = payload.get("explicit_bypass_keys")
        if declared_bypasses is not None and tuple(sorted(str(key) for key in declared_bypasses)) != tuple(
            sorted(result.explicit_bypass_keys)
        ):
            raise ValueError("serialized full-access bypass ledger does not match its permissions")
        expected_hash = payload.get("canonical_hash")
        if expected_hash is not None and str(expected_hash) != result.canonical_hash():
            raise ValueError("serialized mechanism plan hash does not match its contents")
        return result


@dataclass(frozen=True)
class CoverPairLedger:
    qm_unique_rows: int
    qe_unique_rows: int
    qm_raw_paths: int
    qe_raw_paths: int
    qm_rectangular_rows: int
    qe_rectangular_rows: int
    environment_fallback_queries: int
    query_degree_sum: int
    query_degree_max: int
    query_count: int
    path_diagnostics_included: bool = True


def compile_cover_pairs(
    plan: AdaptiveCoverPlan | MechanismPlan,
    queries: torch.Tensor,
    *,
    module_present: torch.Tensor,
    environment_weights: torch.Tensor,
    phase: str | None = None,
    empty_environment: str = "fallback_full",
    include_path_diagnostics: bool = True,
) -> tuple[PackedPairs, PackedPairs, CoverPairLedger]:
    """Compile one case's positive union once, with live QM/QE priors.

    QM priors use ``rho/M`` so the inherited packed reader's
    ``M/(1+M)`` factor reproduces Dense at full access. QE priors keep
    quadrature separate from access until their product enters one global
    attention normalization. Historical tied plans retain a safety
    full-access fallback for no-support QE queries. Typed plans honor an
    explicit all-zero QE permission as a closed route.
    """

    if empty_environment != "fallback_full":
        raise ValueError("only explicit full-access empty-environment fallback is supported")
    access = plan.access(queries) if isinstance(plan, AdaptiveCoverPlan) else None
    if isinstance(plan, MechanismPlan):
        module_source = plan.access_for("QM", queries, phase=phase, module_present=module_present)
        environment_source = plan.access_for("QE", queries, phase=phase)
    else:
        assert access is not None
        module_source = access.module_source
        environment_source = access.environment_source
    if module_present.shape != (module_source.shape[1],):
        raise ValueError("module validity must have shape [M]")
    if environment_weights.numel() == 0:
        raise ValueError("the packed environmental reader requires nonempty positive quadrature")
    if environment_weights.shape != (environment_source.shape[1],):
        raise ValueError("environment weights must have shape [E]")
    if bool((environment_weights <= 0).any()) or not bool(torch.isfinite(environment_weights).all()):
        raise ValueError("environment weights must be finite and positive")
    module_valid = module_present > 0.5
    module_count = module_valid.sum().clamp_min(1)
    module_prior = module_source * module_valid.to(module_source.dtype) / module_count
    environment_prior = environment_source * environment_weights
    if isinstance(plan, AdaptiveCoverPlan):
        # Preserve the compatibility behavior of the historical tied plan.
        safety_mass = environment_weights.sum() * 1.0e-4
        environmental_mass = environment_prior.sum(dim=1)
        fallback = environmental_mass < safety_mass
        safety_weight = endpoint_smoothstep(
            (safety_mass - environmental_mass) / safety_mass
        )
        environment_prior = environment_prior + safety_weight[:, None] * environment_weights[None, :]
    else:
        fallback = torch.zeros(
            (int(queries.shape[0]),), device=queries.device, dtype=torch.bool
        )

    def pack(prior: torch.Tensor, raw_paths: int) -> PackedPairs:
        receiver, source = torch.nonzero(prior > 0, as_tuple=True)
        return PackedPairs(
            batch_index=torch.zeros_like(receiver),
            receiver_index=receiver,
            source_index=source,
            prior=prior[receiver, source],
            raw_path_count=max(raw_paths, int(receiver.numel())),
            unique_pair_count=int(receiver.numel()),
        )

    qm_paths = qe_paths = 0
    if include_path_diagnostics:
        # Contract node/source support in bounded source tiles. The former
        # Q x N x E Boolean product is much larger than the pair union. Each
        # contracted cell is an integer count <= tree capacity, so converting
        # the cells back to integers before summation is exact.
        alpha = plan.tree.access(queries, plan.split_gates)
        active_paths = (alpha > 0).to(torch.float32)

        def count_paths(membership: torch.Tensor, source_valid: torch.Tensor | None = None) -> int:
            support = membership > 0
            if source_valid is not None:
                support = support & source_valid[None, :]
            support = support.to(torch.float32)
            total = torch.zeros((), device=active_paths.device, dtype=torch.int64)
            tile_size = 64
            for start in range(0, int(support.shape[1]), tile_size):
                counts = active_paths @ support[:, start : start + tile_size]
                total = total + counts.to(torch.int64).sum()
            return int(total.item())

        if isinstance(plan, MechanismPlan):
            qm_membership = plan.permission_matrix("QM", phase=phase, module_present=module_valid)
            qe_membership = plan.permission_matrix("QE", phase=phase)
        else:
            qm_membership = plan.module_membership
            qe_membership = plan.environment_membership
        qm_paths = count_paths(qm_membership, module_valid)
        qe_paths = count_paths(qe_membership)
    qm = pack(module_prior, qm_paths)
    qe = pack(environment_prior, qe_paths)
    if isinstance(plan, MechanismPlan):
        source_nonempty = (plan.permission_matrix("QM", phase=phase, module_present=module_valid) > 0).any(dim=1) | (
            plan.permission_matrix("QE", phase=phase) > 0
        ).any(dim=1)
        receiver_group = plan.tree.access(queries, plan.split_gates)
    else:
        source_nonempty = (plan.module_membership > 0).any(dim=1) | (
            plan.environment_membership > 0
        ).any(dim=1)
        assert access is not None
        receiver_group = access.receiver_group
    query_degree = ((receiver_group > 0) & source_nonempty[None, :]).sum(dim=1)
    ledger = CoverPairLedger(
        qm.unique_pair_count,
        qe.unique_pair_count,
        qm.raw_path_count,
        qe.raw_path_count,
        # Dense's reference kernel evaluates padded module slots before its
        # presence mask is applied, so its executed rectangle uses M_pad.
        int(queries.shape[0]) * int(module_present.numel()),
        int(queries.shape[0]) * int(environment_weights.numel()),
        int(fallback.sum()),
        int(query_degree.sum()),
        int(query_degree.max()) if query_degree.numel() else 0,
        int(query_degree.numel()),
        bool(include_path_diagnostics),
    )
    return qm, qe, ledger


def compile_cover_transport_pairs(
    plan: AdaptiveCoverPlan | MechanismPlan,
    receivers: torch.Tensor,
    *,
    source_role: str,
    mechanism: str | None = None,
    phase: str | None = None,
    module_present: torch.Tensor,
    environment_weights: torch.Tensor,
    receiver_valid: torch.Tensor | None = None,
    exclude_self: bool = False,
) -> PackedPairs:
    """Compile one typed preparation transport into unique live pair rows.

    Unlike the query reader, this helper does not add an empty-support
    fallback: an empty preparation message is the zero message Dense would
    produce after its source mask. Source measure remains in each prior so
    the caller can preserve Dense's original denominator.
    """

    if source_role not in {"module", "environment"}:
        raise ValueError("source_role must be module or environment")
    if mechanism is None:
        mechanism = "MM" if source_role == "module" else "ME"
    if isinstance(plan, MechanismPlan):
        access_source = plan.access_for(
            mechanism,
            receivers,
            phase=phase,
            module_present=module_present if source_role == "module" else None,
        )
    else:
        access = plan.access(receivers)
        access_source = access.module_source if source_role == "module" else access.environment_source
    if source_role == "module":
        if module_present.shape != (access_source.shape[1],):
            raise ValueError("module validity must have shape [M]")
        prior = access_source * (module_present > 0.5).to(access_source.dtype)
    else:
        if environment_weights.shape != (access_source.shape[1],):
            raise ValueError("environment weights must have shape [E]")
        if not bool(torch.isfinite(environment_weights).all()) or bool((environment_weights < 0).any()):
            raise ValueError("environment weights must be finite and nonnegative")
        prior = access_source * environment_weights[None, :]
    if receiver_valid is not None:
        if receiver_valid.shape != (receivers.shape[0],):
            raise ValueError("receiver validity must have shape [Q]")
        prior = prior * (receiver_valid > 0.5).to(prior.dtype)[:, None]
    if exclude_self:
        if prior.shape[0] != prior.shape[1]:
            raise ValueError("self exclusion requires aligned receiver and source indices")
        prior = prior.masked_fill(torch.eye(prior.shape[0], device=prior.device, dtype=torch.bool), 0.0)
    receiver_index, source_index = torch.nonzero(prior > 0.0, as_tuple=True)
    unique = int(receiver_index.numel())
    return PackedPairs(
        batch_index=torch.zeros_like(receiver_index),
        receiver_index=receiver_index,
        source_index=source_index,
        prior=prior[receiver_index, source_index],
        raw_path_count=unique,
        unique_pair_count=unique,
    )


__all__ = [
    "INTERACTION_MECHANISMS",
    "AdaptiveCoverPlan",
    "CandidateNode",
    "CaseLocalReceiverTree",
    "CoverAccess",
    "CoverFrontierSummary",
    "CoverPairLedger",
    "InteractionContext",
    "InteractionMechanism",
    "InteractionPermissionKey",
    "MechanismPlan",
    "ReceiverAnchorUniverse",
    "compile_cover_pairs",
    "compile_cover_transport_pairs",
    "endpoint_smoothstep",
]
