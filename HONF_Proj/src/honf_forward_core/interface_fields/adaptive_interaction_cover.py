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
from typing import Any, Literal

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
        max_depth: int | None = None,
    ) -> CaseLocalReceiverTree:
        if max_nodes < 1 or min_leaf_anchors < 1:
            raise ValueError("tree capacity and minimum leaf size must be positive")
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0
        ):
            raise ValueError("max_depth must be a nonnegative integer or None")
        if not 0.0 < overlap_fraction < 1.0:
            raise ValueError("receiver overlap fraction must be between zero and one")
        nodes: list[CandidateNode] = []
        saturated = False
        coordinates = (universe.coordinates / universe.coordinate_scale).detach().cpu()
        weights = universe.weights.detach().cpu()
        roles = universe.roles.detach().cpu()
        # Sorting runs on the host. Convert its scalar keys once, preserving
        # their exact stored values and stable tie order. Prefix/cut sums and
        # extent/axis selection retain the original tensor arithmetic.
        coordinate_values = coordinates.tolist()
        weight_values = weights.tolist()
        role_values = roles.tolist()

        def refine(index: int, indices: tuple[int, ...], depth: int) -> None:
            nonlocal saturated
            if len(indices) <= min_leaf_anchors or (
                max_depth is not None and depth >= max_depth
            ):
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
                    coordinate_values[item][axis],
                    *coordinate_values[item],
                    role_values[item],
                ),
            )
            half_mass = float(weights[list(ordered)].sum()) / 2.0
            accumulated = 0.0
            cut = 1
            for position, anchor_id in enumerate(ordered[:-1], start=1):
                accumulated += weight_values[anchor_id]
                cut = position
                if accumulated >= half_mass:
                    break
            left_ids, right_ids = tuple(ordered[:cut]), tuple(ordered[cut:])
            left, right = len(nodes), len(nodes) + 1
            nodes.extend((CandidateNode(left_ids), CandidateNode(right_ids)))
            nodes[index] = CandidateNode(indices, left, right, axis)
            refine(left, left_ids, depth + 1)
            refine(right, right_ids, depth + 1)

        root_ids = tuple(range(int(universe.coordinates.shape[0])))
        nodes.append(CandidateNode(root_ids))
        refine(0, root_ids, 0)
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
class DirectPairAccess:
    """Exact independently scored physical receiver/source permissions."""

    receiver_coordinates: torch.Tensor  # [R,d], current native receiver order
    weights: torch.Tensor  # [R,S], hard support or differentiable soft shadow
    receiver_validity: torch.Tensor | None = None  # [R], optional native row validity

    def __post_init__(self) -> None:
        if self.receiver_coordinates.ndim != 2 or self.weights.ndim != 2:
            raise ValueError("direct pair access needs [R,d] coordinates and [R,S] weights")
        if self.receiver_coordinates.shape[0] != self.weights.shape[0]:
            raise ValueError("direct pair receiver coordinates and weights must align")
        if self.receiver_validity is not None and self.receiver_validity.shape != self.weights.shape[:1]:
            raise ValueError("direct pair receiver validity must align with receiver rows")
        if not bool(torch.isfinite(self.receiver_coordinates).all()) or not bool(torch.isfinite(self.weights).all()):
            raise ValueError("direct pair coordinates and weights must be finite")
        if bool(((self.weights < 0.0) | (self.weights > 1.0)).any()):
            raise ValueError("direct pair weights must lie in [0,1]")
        if self.receiver_coordinates.device != self.weights.device:
            raise ValueError("direct pair coordinates and weights must share one device")
        if self.receiver_validity is not None and self.receiver_validity.device != self.weights.device:
            raise ValueError("direct pair receiver validity and weights must share one device")
        if self.receiver_validity is not None and not bool(torch.isfinite(self.receiver_validity).all()):
            raise ValueError("direct pair receiver validity must be finite")


@dataclass(frozen=True)
class DirectPairPolicy:
    """Runtime-only direct scorer for panels that are known only at read time."""

    scorer: Any
    source_coordinates: torch.Tensor  # [S,d], current input-only catalogue
    source_features: torch.Tensor  # [S,F], detached current source encoding
    source_validity: torch.Tensor  # [S]
    hard_threshold: torch.Tensor  # scalar canonical threshold (may be +/-inf)
    soft_threshold: torch.Tensor  # scalar finite restoration threshold
    budget_fraction: float
    hard: bool
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not callable(self.scorer):
            raise TypeError("direct pair policy scorer must be callable")
        if self.source_coordinates.ndim != 2 or self.source_features.ndim != 2:
            raise ValueError("dynamic direct sources require [S,d] coordinates and [S,F] features")
        if self.source_coordinates.shape[0] != self.source_features.shape[0]:
            raise ValueError("dynamic direct source coordinates and features must align")
        if self.source_validity.shape != self.source_features.shape[:1]:
            raise ValueError("dynamic direct source validity must align with sources")
        if self.hard_threshold.numel() != 1 or self.soft_threshold.numel() != 1:
            raise ValueError("dynamic direct thresholds must be scalar")
        if not bool(torch.isfinite(self.source_coordinates).all()) or not bool(torch.isfinite(self.source_features).all()):
            raise ValueError("dynamic direct source descriptors must be finite")
        if not bool(torch.isfinite(self.source_validity).all()):
            raise ValueError("dynamic direct source validity must be finite")
        if not bool(torch.isfinite(self.soft_threshold).all()):
            raise ValueError("dynamic direct soft threshold must be finite")
        if not 0.0 <= float(self.budget_fraction) <= 1.0 or self.temperature <= 0.0:
            raise ValueError("dynamic direct budget and temperature are invalid")
        object.__setattr__(self, "source_coordinates", self.source_coordinates.detach().clone())
        object.__setattr__(self, "source_features", self.source_features.detach().clone())
        object.__setattr__(self, "source_validity", self.source_validity.detach().clone())
        object.__setattr__(self, "hard_threshold", self.hard_threshold.detach().clone())
        object.__setattr__(self, "soft_threshold", self.soft_threshold.detach().clone())

    def access_for(
        self,
        mechanism: str,
        receiver_coordinates: torch.Tensor,
        receiver_features: torch.Tensor | None,
        *,
        receiver_validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score one live native panel; only soft P mode carries score gradients."""

        if receiver_features is None:
            raise ValueError("dynamic direct P needs the current detached receiver features")
        if receiver_coordinates.ndim != 2 or receiver_features.ndim != 2:
            raise ValueError("dynamic direct receivers must be [R,d] coordinates and [R,F] features")
        if receiver_coordinates.shape[0] != receiver_features.shape[0]:
            raise ValueError("dynamic direct receiver coordinates and features must align")
        if receiver_coordinates.shape[1] != self.source_coordinates.shape[1]:
            raise ValueError("dynamic direct receivers and sources use different physical frames")
        source_coordinates = self.source_coordinates.to(
            device=receiver_coordinates.device, dtype=receiver_coordinates.dtype
        )
        source_features = self.source_features.to(
            device=receiver_features.device, dtype=receiver_features.dtype
        )
        source_validity = self.source_validity.to(device=receiver_coordinates.device) > 0.5
        eligible_rows = source_validity[None, :].expand(
            int(receiver_coordinates.shape[0]), -1
        )
        if receiver_validity is not None:
            if receiver_validity.shape != receiver_coordinates.shape[:1]:
                raise ValueError("dynamic direct receiver validity must align with live rows")
            eligible_rows = eligible_rows & (
                receiver_validity.to(device=receiver_coordinates.device) > 0.5
            )[:, None]
        if float(self.budget_fraction) == 1.0:
            # Matched full-access control/warm start: exact native all-source
            # values and no direct scorer gradient until sparse adaptation.
            return eligible_rows.to(receiver_features.dtype)
        # This scorer is an independent routing control. Inputs are frozen;
        # only scorer parameters receive gradients in soft P training.
        # Hard access is a detached threshold decision. Building a scorer
        # autograd graph here only stores and recomputes the large pair panel;
        # the soft policy below remains the trainable route shadow.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.hard):
            scores = self.scorer(
                receiver_features.detach(),
                source_features,
                receiver_coordinates.detach(),
                source_coordinates,
                mechanism=mechanism,
                budget_fraction=float(self.budget_fraction),
            )
        eligible = eligible_rows.to(device=scores.device)
        if self.hard:
            threshold = self.hard_threshold.to(device=scores.device, dtype=scores.dtype)
            return ((scores.detach() >= threshold) & eligible).to(scores.dtype)
        threshold = self.soft_threshold.to(device=scores.device, dtype=scores.dtype)
        return torch.sigmoid((scores - threshold) / float(self.temperature)) * eligible.to(scores.dtype)


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
    direct_access: Mapping[
        InteractionPermissionKey | str | tuple[str, str | None], DirectPairAccess
    ] = field(default_factory=dict)
    direct_policies: Mapping[
        InteractionPermissionKey | str | tuple[str, str | None], DirectPairPolicy
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
        direct: dict[InteractionPermissionKey, DirectPairAccess] = {}
        for raw_key, raw_value in self.direct_access.items():
            key = InteractionPermissionKey.coerce(raw_key)
            if key in direct:
                raise ValueError(f"duplicate normalized direct access key {key.canonical_name!r}")
            if not isinstance(raw_value, DirectPairAccess):
                raise TypeError("direct_access values must be DirectPairAccess records")
            source_count = self._source_count(key.mechanism)
            if raw_value.weights.shape[1] != source_count:
                raise ValueError(f"direct access {key.canonical_name} source order does not match the plan")
            if raw_value.weights.device != self.split_gates.device:
                raise ValueError("direct access, split gates, and receiver tree must share one device")
            if raw_value.receiver_coordinates.shape[1] != self.tree.universe.coordinates.shape[1]:
                raise ValueError("direct access coordinates use a different physical frame")
            direct[key] = DirectPairAccess(
                raw_value.receiver_coordinates.detach().clone(),
                raw_value.weights.clone(),
                None if raw_value.receiver_validity is None else raw_value.receiver_validity.detach().clone(),
            )
        policies: dict[InteractionPermissionKey, DirectPairPolicy] = {}
        for raw_key, raw_value in self.direct_policies.items():
            key = InteractionPermissionKey.coerce(raw_key)
            if key in policies:
                raise ValueError(f"duplicate normalized direct policy key {key.canonical_name!r}")
            if key in direct:
                raise ValueError(f"route {key.canonical_name} cannot have both fixed and dynamic direct access")
            if not isinstance(raw_value, DirectPairPolicy):
                raise TypeError("direct_policies values must be DirectPairPolicy records")
            if raw_value.source_coordinates.shape[1] != self.tree.universe.coordinates.shape[1]:
                raise ValueError("direct policy source coordinates use a different physical frame")
            if raw_value.source_features.device != self.split_gates.device:
                raise ValueError("direct policy sources, split gates, and tree must share a device")
            policies[key] = raw_value
        object.__setattr__(self, "split_gates", self.split_gates.clone())
        object.__setattr__(self, "module_present", self.module_present.clone())
        object.__setattr__(self, "permissions", MappingProxyType(normalized))
        object.__setattr__(self, "direct_access", MappingProxyType(direct))
        object.__setattr__(self, "direct_policies", MappingProxyType(policies))

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
            and InteractionPermissionKey(mechanism) not in self.direct_access
            and InteractionPermissionKey(mechanism) not in self.direct_policies
        )

    @property
    def explicit_phase_keys(self) -> tuple[str, ...]:
        return tuple(sorted(
            key.canonical_name
            for key in (*self.permissions.keys(), *self.direct_access.keys(), *self.direct_policies.keys())
            if key.phase is not None
        ))

    def direct_pair_access_for(
        self, mechanism: str, *, phase: str | None = None
    ) -> DirectPairAccess | None:
        """Resolve an exact physical-pair route, preserving its receiver panel."""

        base_key = InteractionPermissionKey(mechanism)
        selected_key = InteractionPermissionKey(mechanism, phase)
        if phase is not None and selected_key in self.direct_access:
            return self.direct_access[selected_key]
        return self.direct_access.get(base_key)

    def direct_pair_policy_for(
        self, mechanism: str, *, phase: str | None = None
    ) -> DirectPairPolicy | None:
        """Resolve the runtime-only scorer for a dynamic native receiver panel."""

        base_key = InteractionPermissionKey(mechanism)
        selected_key = InteractionPermissionKey(mechanism, phase)
        if phase is not None and selected_key in self.direct_policies:
            return self.direct_policies[selected_key]
        return self.direct_policies.get(base_key)

    def permission_status(self, mechanism: str, *, phase: str | None = None) -> str:
        key = InteractionPermissionKey.coerce((mechanism, phase))
        if key in self.direct_access:
            return "direct_pair_permission"
        if key in self.direct_policies:
            return "direct_pair_policy_runtime"
        if phase is not None and InteractionPermissionKey(mechanism) in self.direct_access:
            return "direct_pair_permission_inherited"
        if phase is not None and InteractionPermissionKey(mechanism) in self.direct_policies:
            return "direct_pair_policy_inherited"
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
        """Resolve a node/source mask and apply module padding validity.

        Exact direct-pair routes have no faithful node/source matrix. Callers
        must use ``access_for`` or ``direct_pair_access_for`` for those routes.
        """

        if (
            self.direct_pair_access_for(mechanism, phase=phase) is not None
            or self.direct_pair_policy_for(mechanism, phase=phase) is not None
        ):
            key = InteractionPermissionKey(mechanism, phase)
            raise ValueError(
                f"{key.canonical_name} is an exact direct-pair route; its access cannot be represented as node memberships"
            )
        return self.node_permission_matrix(
            mechanism,
            source_count,
            phase=phase,
            module_present=module_present,
        )

    def node_permission_matrix(
        self,
        mechanism: str,
        source_count: int | None = None,
        *,
        phase: str | None = None,
        module_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Resolve the stored node/source fallback, without direct overrides."""

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
        receiver_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one typed receiver/source access matrix ``[Q,S]``."""

        policy = self.direct_pair_policy_for(mechanism, phase=phase)
        if policy is not None:
            expected_count = self._source_count(InteractionPermissionKey(mechanism).mechanism)
            if source_count is not None and int(source_count) != expected_count:
                raise ValueError(f"{mechanism} source count must be {expected_count}")
            values = policy.access_for(
                mechanism,
                receivers,
                receiver_features,
                receiver_validity=None,
            )
            if InteractionPermissionKey(mechanism).mechanism in {"MM", "EM", "QM"}:
                valid = self.module_present if module_present is None else module_present
                if valid.shape != (expected_count,):
                    raise ValueError("module validity does not match the typed module source axis")
                values = values * (valid.to(device=values.device) > 0.5).to(values.dtype)[None, :]
            return values
        direct = self.direct_pair_access_for(mechanism, phase=phase)
        if direct is not None:
            base_key = InteractionPermissionKey(mechanism)
            expected_count = self._source_count(base_key.mechanism)
            if source_count is not None and int(source_count) != expected_count:
                raise ValueError(f"{mechanism} source count must be {expected_count}")
            coordinates = receivers.to(
                device=direct.receiver_coordinates.device,
                dtype=direct.receiver_coordinates.dtype,
            )
            if coordinates.shape != direct.receiver_coordinates.shape or not torch.equal(
                coordinates.detach(), direct.receiver_coordinates
            ):
                direct_key = InteractionPermissionKey(mechanism, phase)
                raise ValueError(
                    f"direct {direct_key.canonical_name} access was built for a different receiver panel/order"
                )
            values = direct.weights
            if base_key.mechanism in {"MM", "EM", "QM"}:
                valid = self.module_present if module_present is None else module_present
                if valid.shape != (expected_count,):
                    raise ValueError("module validity does not match the typed module source axis")
                values = values * (valid.to(device=values.device) > 0.5).to(values.dtype)[None, :]
            if direct.receiver_validity is not None:
                values = values * (direct.receiver_validity.to(device=values.device) > 0.5).to(
                    values.dtype
                )[:, None]
            return values

        alpha = self.tree.access(receivers, self.split_gates)
        membership = self.permission_matrix(
            mechanism, source_count, phase=phase, module_present=module_present
        )
        return alpha @ membership

    def access(self, queries: torch.Tensor) -> CoverAccess:
        """Compatibility view with QM/QE sources for existing diagnostics."""

        if (
            self.direct_pair_access_for("QM") is not None
            or self.direct_pair_access_for("QE") is not None
            or self.direct_pair_policy_for("QM") is not None
            or self.direct_pair_policy_for("QE") is not None
        ):
            raise ValueError(
                "direct query routes cannot be represented by CoverAccess; call access_for with the typed physical panel"
            )

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

    def with_direct_pair_access(
        self,
        mechanism: str,
        receiver_coordinates: torch.Tensor,
        weights: torch.Tensor,
        *,
        phase: str | None = None,
        receiver_validity: torch.Tensor | None = None,
    ) -> MechanismPlan:
        """Attach exact receiver/source access without a tree approximation.

        The receiver panel and its order are part of the frozen direct plan.
        Native execution rejects a different panel rather than silently
        rebinding the independently scored physical pairs.
        """

        key = InteractionPermissionKey(mechanism, phase)
        direct = dict(self.direct_access)
        direct[key] = DirectPairAccess(
            receiver_coordinates.detach(), weights, receiver_validity
        )
        return replace(self, direct_access=direct)

    def with_direct_pair_policy(
        self,
        mechanism: str,
        scorer: Any,
        *,
        source_coordinates: torch.Tensor,
        source_features: torch.Tensor,
        source_validity: torch.Tensor,
        hard_threshold: torch.Tensor,
        soft_threshold: torch.Tensor,
        budget_fraction: float,
        hard: bool,
        temperature: float = 1.0,
        phase: str | None = None,
    ) -> MechanismPlan:
        """Attach an independently scored route evaluated on live read panels.

        Source descriptors are frozen input-only tensors. The core revalidates
        them against the candidate's module or environmental catalogue at
        every fixed-plan bind. Query receiver features are supplied by the
        native read and detached inside the policy.
        """

        key = InteractionPermissionKey(mechanism, phase)
        policies = dict(self.direct_policies)
        policies[key] = DirectPairPolicy(
            scorer=scorer,
            source_coordinates=source_coordinates,
            source_features=source_features,
            source_validity=source_validity,
            hard_threshold=hard_threshold,
            soft_threshold=soft_threshold,
            budget_fraction=float(budget_fraction),
            hard=bool(hard),
            temperature=float(temperature),
        )
        return replace(self, direct_policies=policies)

    def active_node_mask(self, receivers: torch.Tensor | None = None) -> torch.Tensor:
        if receivers is None:
            receivers = self.tree.universe.coordinates
        return (self.tree.access(receivers, self.split_gates) > 0).any(dim=0)

    def active_group_count(self) -> int:
        active = self.active_node_mask()
        source_bearing = torch.zeros_like(active)
        for mechanism in INTERACTION_MECHANISMS:
            if (
                self.direct_pair_access_for(mechanism) is None
                and self.direct_pair_policy_for(mechanism) is None
            ):
                source_bearing |= (self.permission_matrix(mechanism) > 0).any(dim=1)
        return int((active & source_bearing).sum())

    def is_full_access(
        self,
        *,
        phase: str | None = None,
        mechanisms: tuple[InteractionMechanism, ...] = INTERACTION_MECHANISMS,
    ) -> bool:
        # A live receiver may reach a branch that has zero access on the
        # finite anchor sample. Check every branch allowed by the split gates
        # before declaring the plan safe for a native all-access shortcut.
        reachable = {0}
        frontier = (0,)
        gate_values = self.split_gates.detach().cpu().tolist()
        while frontier:
            children_next: list[int] = []
            for index in frontier:
                node = self.tree.nodes[index]
                if node.left is not None and node.right is not None and float(gate_values[index]) > 0.0:
                    children_next.extend((node.left, node.right))
                    reachable.update((node.left, node.right))
            frontier = tuple(children_next)
        reachable_indices = sorted(reachable)
        for mechanism in mechanisms:
            # A dynamic policy is evaluated for the live read panel and is
            # never eligible for a native full-access shortcut.
            if self.direct_pair_policy_for(mechanism, phase=phase) is not None:
                return False
            direct = self.direct_pair_access_for(mechanism, phase=phase)
            if direct is not None:
                expected = (
                    (self.module_present > 0.5).to(direct.weights.dtype)
                    if mechanism in {"MM", "EM", "QM"}
                    else direct.weights.new_ones((direct.weights.shape[1],))
                )
                expected_rows = expected[None, :].expand_as(direct.weights)
                if direct.receiver_validity is not None:
                    expected_rows = expected_rows * (
                        direct.receiver_validity.detach().to(expected_rows.dtype) > 0.5
                    ).to(expected_rows.dtype)[:, None]
                if not torch.equal(
                    direct.weights.detach(),
                    expected_rows,
                ):
                    return False
                continue
            matrix = self.permission_matrix(mechanism, phase=phase)
            expected = (
                (self.module_present > 0.5).to(matrix.dtype)
                if mechanism in {"MM", "EM", "QM"}
                else matrix.new_ones(matrix.shape[1])
            )
            if not torch.equal(
                matrix[reachable_indices],
                expected[None, :].expand(len(reachable_indices), -1),
            ):
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
        direct = self.direct_pair_access_for(mechanism, phase=phase)
        if direct is not None:
            live = direct.weights.detach() > 0.0
            return CoverFrontierSummary(
                InteractionPermissionKey(mechanism).mechanism,
                phase,
                len(self.tree.nodes),
                0,
                0,
                0,
                int(live.any(dim=0).sum()),
                int(live.sum()),
            )
        if self.direct_pair_policy_for(mechanism, phase=phase) is not None:
            return CoverFrontierSummary(
                InteractionPermissionKey(mechanism, phase).mechanism,
                phase,
                len(self.tree.nodes),
                0,
                0,
                0,
                0,
                0,
            )
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

        if self.direct_policies:
            raise ValueError("dynamic direct-pair policies are runtime-only and cannot be hashed for persistence")

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
            add_tensor(key.canonical_name, self.node_permission_matrix(mechanism, phase=phase))
        if self.direct_access:
            for key, direct in sorted(
                self.direct_access.items(), key=lambda item: item[0].canonical_name
            ):
                digest.update(f"direct:{key.canonical_name}".encode())
                add_tensor(f"direct_coordinates:{key.canonical_name}", direct.receiver_coordinates)
                add_tensor(f"direct_weights:{key.canonical_name}", direct.weights)
                if direct.receiver_validity is not None:
                    add_tensor(f"direct_receiver_validity:{key.canonical_name}", direct.receiver_validity)
        return digest.hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible, stable serialization of this plan."""

        if self.direct_policies:
            raise ValueError("dynamic direct-pair policies are runtime-only and cannot be serialized")

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
            "direct_access": {
                key.canonical_name: {
                    "receiver_coordinates": direct.receiver_coordinates.detach().cpu().tolist(),
                    "weights": direct.weights.detach().cpu().tolist(),
                    "receiver_validity": None
                    if direct.receiver_validity is None
                    else direct.receiver_validity.detach().cpu().tolist(),
                }
                for key, direct in sorted(
                    self.direct_access.items(), key=lambda item: item[0].canonical_name
                )
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
        direct_payload = payload.get("direct_access", {})
        if not isinstance(direct_payload, Mapping):
            raise TypeError("serialized direct access must be a mapping")
        result = cls(
            tree,
            torch.as_tensor(payload["split_gates"], device=target_device, dtype=dtype),
            torch.as_tensor(payload["module_present"], device=target_device, dtype=dtype),
            int(payload["environment_count"]),
            {
                str(key): torch.as_tensor(value, device=target_device, dtype=dtype)
                for key, value in permissions_payload.items()
            },
            {
                str(key): DirectPairAccess(
                    torch.as_tensor(value["receiver_coordinates"], device=target_device, dtype=dtype),
                    torch.as_tensor(value["weights"], device=target_device, dtype=dtype),
                    None
                    if value.get("receiver_validity") is None
                    else torch.as_tensor(value["receiver_validity"], device=target_device, dtype=dtype),
                )
                for key, value in direct_payload.items()
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


def _compiled_plan_tensor_versions(
    plan: AdaptiveCoverPlan | MechanismPlan,
) -> tuple[tuple[str, int, int], ...]:
    tensors: list[tuple[str, torch.Tensor]] = [("split_gates", plan.split_gates)]
    universe = plan.tree.universe
    tensors.extend((
        ("anchor_coordinates", universe.coordinates),
        ("coordinate_scale", universe.coordinate_scale),
    ))
    if isinstance(plan, MechanismPlan):
        tensors.append(("module_present", plan.module_present))
        tensors.extend(
            (f"permission:{key.canonical_name}", value)
            for key, value in sorted(plan.permissions.items(), key=lambda item: item[0].canonical_name)
        )
        for key, value in sorted(
            plan.direct_access.items(), key=lambda item: item[0].canonical_name
        ):
            tensors.extend((
                (f"direct_coordinates:{key.canonical_name}", value.receiver_coordinates),
                (f"direct_weights:{key.canonical_name}", value.weights),
            ))
            if value.receiver_validity is not None:
                tensors.append((f"direct_receiver_validity:{key.canonical_name}", value.receiver_validity))
        for key, policy in sorted(
            plan.direct_policies.items(), key=lambda item: item[0].canonical_name
        ):
            tensors.extend((
                (f"direct_policy_coordinates:{key.canonical_name}", policy.source_coordinates),
                (f"direct_policy_features:{key.canonical_name}", policy.source_features),
                (f"direct_policy_validity:{key.canonical_name}", policy.source_validity),
                (f"direct_policy_hard_threshold:{key.canonical_name}", policy.hard_threshold),
                (f"direct_policy_soft_threshold:{key.canonical_name}", policy.soft_threshold),
            ))
            scorer_parameters = getattr(policy.scorer, "named_parameters", lambda: ())()
            tensors.extend(
                (f"direct_policy_scorer:{key.canonical_name}:{name}", tensor)
                for name, tensor in scorer_parameters
            )
    else:
        tensors.extend((
            ("module_membership", plan.module_membership),
            ("environment_membership", plan.environment_membership),
        ))
    versions = []
    for name, tensor in tensors:
        try:
            version = int(tensor._version)
        except RuntimeError:  # Inference tensors do not expose a version counter.
            version = -1
        versions.append((name, id(tensor), version))
    return tuple(versions)


@dataclass(frozen=True, slots=True)
class CompiledMechanismExecutionView:
    """Private immutable routing metadata bound to one prepared plan and phase.

    Continuous access is still evaluated from live receiver coordinates. This
    view only caches structure and values that are static for the prepared
    plan: the active anchor frontier, typed source sets, exact action classes,
    and full-access/differentiability flags.
    """

    plan_identity: int
    phase: str | None
    tensor_versions: tuple[tuple[str, int, int], ...]
    active_node_indices: tuple[int, ...]
    parent_child_relations: tuple[tuple[int, int, int], ...]
    hard_execution_node_mask: tuple[bool, ...]
    split_axes: tuple[int | None, ...]
    children_by_node: tuple[tuple[int, int] | None, ...]
    split_boundaries: tuple[torch.Tensor | None, ...]
    overlap_widths: tuple[torch.Tensor | None, ...]
    coordinate_dim: int
    device: torch.device
    full_access_flags: tuple[tuple[str, bool], ...]
    differentiable_flags: tuple[tuple[str, bool], ...]
    key_missing_flags: tuple[tuple[str, bool], ...]
    permission_matrices: tuple[tuple[str, torch.Tensor], ...]
    source_indices_by_mechanism: tuple[
        tuple[str, tuple[tuple[int, ...], ...]], ...
    ]
    compact_action_classes: tuple[
        tuple[str, tuple[tuple[int, ...], ...]], ...
    ]
    frontier_summaries: tuple[tuple[str, CoverFrontierSummary], ...]
    candidate_node_count: int
    capacity_saturated: bool
    transition_count: int

    def matches(self, plan: AdaptiveCoverPlan | MechanismPlan, *, phase: str | None) -> bool:
        """Check that this runtime view still belongs to the supplied plan."""

        return (
            self.plan_identity == id(plan)
            and self.phase == phase
            and self.tensor_versions == _compiled_plan_tensor_versions(plan)
        )

    def receiver_access(self, queries: torch.Tensor, split_gates: torch.Tensor) -> torch.Tensor:
        """Evaluate live query access with prepared levels and split geometry.

        Hard split gates use the precompiled structurally reachable node set,
        which skips closed subtrees while retaining the original node order.
        Trainable gates retain the full graph, including endpoint surrogate
        derivatives through currently closed branches.
        """

        if queries.ndim != 2 or queries.shape[1] != self.coordinate_dim:
            raise ValueError("queries must have shape [Q,d] in the prepared anchor frame")
        if split_gates.shape != (len(self.split_axes),):
            raise ValueError("split gates must have one value per candidate-node capacity")
        if queries.device != split_gates.device:
            raise ValueError("queries and split gates must use one device")
        if queries.device != self.device:
            raise ValueError("queries, compiled geometry, and split gates must use one device")
        hard_gates = not split_gates.requires_grad
        result: list[torch.Tensor | None] = [None] * len(self.split_axes)

        def descend(index: int, incoming: torch.Tensor) -> None:
            if hard_gates and not self.hard_execution_node_mask[index]:
                return
            axis = self.split_axes[index]
            if axis is None:
                result[index] = incoming
                return
            boundary = self.split_boundaries[index]
            overlap = self.overlap_widths[index]
            assert boundary is not None and overlap is not None
            split_gate = split_gates[index]
            smooth_gate = endpoint_smoothstep(split_gate)
            if split_gate.requires_grad:
                endpoint = (split_gate.detach() == 0.0) | (split_gate.detach() == 1.0)
                surrogate_gate = torch.where(endpoint, split_gate, smooth_gate)
                gate = surrogate_gate + (smooth_gate - surrogate_gate).detach()
            else:
                gate = smooth_gate
            result[index] = incoming * (1.0 - gate)
            children = self.children_by_node[index]
            assert children is not None
            left, right = children
            if hard_gates and not self.hard_execution_node_mask[left]:
                return
            left_weight = endpoint_smoothstep(
                (boundary + overlap / 2.0 - queries[:, axis]) / overlap
            )
            descend(left, incoming * gate * left_weight)
            descend(right, incoming * gate * (1.0 - left_weight))

        if result:
            descend(0, queries.new_ones((int(queries.shape[0]),)))
        zero = queries.new_zeros((int(queries.shape[0]),))
        return torch.stack([zero if value is None else value for value in result], dim=1)

    def permission_matrix(self, mechanism: str) -> torch.Tensor:
        tag = InteractionPermissionKey(mechanism).mechanism
        return next(value for key, value in self.permission_matrices if key == tag)

    def is_full_access(self, mechanism: str) -> bool:
        tag = InteractionPermissionKey(mechanism).mechanism
        return next(value for key, value in self.full_access_flags if key == tag)

    def has_differentiable_routing(self, mechanisms: tuple[str, ...]) -> bool:
        requested = {InteractionPermissionKey(mechanism).mechanism for mechanism in mechanisms}
        return any(key in requested and value for key, value in self.differentiable_flags)

    def permission_key_missing(self, mechanism: str) -> bool:
        tag = InteractionPermissionKey(mechanism).mechanism
        return next(value for key, value in self.key_missing_flags if key == tag)

    def source_indices(self, mechanism: str) -> tuple[tuple[int, ...], ...]:
        tag = InteractionPermissionKey(mechanism).mechanism
        return next(value for key, value in self.source_indices_by_mechanism if key == tag)

    def action_classes(self, mechanism: str) -> tuple[tuple[int, ...], ...]:
        tag = InteractionPermissionKey(mechanism).mechanism
        return next(value for key, value in self.compact_action_classes if key == tag)

    def summary(self, mechanism: str) -> CoverFrontierSummary:
        tag = InteractionPermissionKey(mechanism).mechanism
        return next(value for key, value in self.frontier_summaries if key == tag)


def compile_mechanism_execution_view(
    plan: AdaptiveCoverPlan | MechanismPlan,
    *,
    module_present: torch.Tensor,
    environment_count: int,
    phase: str | None = None,
) -> CompiledMechanismExecutionView:
    """Compile static typed routing metadata once for one prepared plan.

    This is deliberately runtime-only; plan serialization and canonical
    identity remain unchanged. The anchor access matrix is evaluated once,
    then shared by all five mechanism summaries.
    """

    selected_phase = None if phase is None else str(phase).strip()
    if selected_phase == "":
        raise ValueError("execution view phase cannot be empty")
    if type(plan) not in {AdaptiveCoverPlan, MechanismPlan}:
        raise TypeError(
            "compiled execution views require an exact plan type; custom access_for overrides "
            "must use the reference executor"
        )
    typed = (
        plan
        if isinstance(plan, MechanismPlan)
        else MechanismPlan.from_legacy(plan, module_present)
    )
    if typed.module_present.shape != module_present.shape or not torch.equal(
        typed.module_present > 0.5, module_present > 0.5
    ):
        raise ValueError("execution view module source ordering does not match the prepared case")
    if typed.environment_count != int(environment_count):
        raise ValueError("execution view environment source ordering does not match the prepared case")
    if typed.split_gates.device != module_present.device:
        raise ValueError("execution view plan and prepared source catalogue must share a device")
    if typed.direct_access or typed.direct_policies:
        raise ValueError(
            "direct receiver/source plans cannot be represented by compiled node memberships; "
            "use the exact native reference executor"
        )

    # The anchor universe defines which plan nodes are structurally active.
    # It is intentionally detached: this metadata never participates in trial
    # coordinate or organizer gradients.
    with torch.no_grad():
        anchor_access = typed.tree.access(
            typed.tree.universe.coordinates.detach(), typed.split_gates.detach()
        )
        active_mask = (anchor_access > 0).any(dim=0)
        active_node_indices = tuple(
            int(index) for index in torch.nonzero(active_mask, as_tuple=False).flatten().cpu().tolist()
        )

    full_access_flags: list[tuple[str, bool]] = []
    differentiable_flags: list[tuple[str, bool]] = []
    key_missing_flags: list[tuple[str, bool]] = []
    permission_matrices: list[tuple[str, torch.Tensor]] = []
    source_indices_by_mechanism: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
    compact_action_classes: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
    frontier_summaries: list[tuple[str, CoverFrontierSummary]] = []
    split_axes = tuple(node.split_axis for node in typed.tree.nodes)
    children_by_node = tuple(
        None if node.left is None else (node.left, node.right)
        for node in typed.tree.nodes
    )
    split_boundaries: list[torch.Tensor | None] = [None] * len(typed.tree.nodes)
    overlap_widths: list[torch.Tensor | None] = [None] * len(typed.tree.nodes)
    for index, node in enumerate(typed.tree.nodes):
        if node.left is None:
            continue
        assert node.right is not None and node.split_axis is not None
        axis = node.split_axis
        anchor_axis = typed.tree.universe.coordinates[:, axis]
        left_center = anchor_axis[list(typed.tree.nodes[node.left].anchor_indices)].mean()
        right_center = anchor_axis[list(typed.tree.nodes[node.right].anchor_indices)].mean()
        split_boundaries[index] = (left_center + right_center) / 2.0
        overlap_widths[index] = (
            typed.tree.overlap_fraction * typed.tree.universe.coordinate_scale[axis]
        )
    # The hard executor may skip only branches whose incoming mass is
    # structurally zero for every receiver. This criterion depends on split
    # gates, not anchor-space activity, so it remains exact outside the anchor
    # envelope and keeps live split-boundary geometry derivatives intact.
    hard_reachable = {0}
    hard_gate_values = typed.split_gates.detach().cpu().tolist()
    frontier = (0,)
    while frontier:
        following: list[int] = []
        for index in frontier:
            children = children_by_node[index]
            if children is not None and float(hard_gate_values[index]) > 0.0:
                hard_reachable.update(children)
                following.extend(children)
        frontier = tuple(following)
    hard_execution_node_mask = tuple(
        index in hard_reachable for index in range(len(typed.tree.nodes))
    )
    hard_execution_node_indices = tuple(
        index for index, reachable in enumerate(hard_execution_node_mask) if reachable
    )
    hard_execution_index_tensor = torch.as_tensor(
        hard_execution_node_indices, device=module_present.device, dtype=torch.long
    )
    active_index_tensor = torch.as_tensor(
        active_node_indices, device=module_present.device, dtype=torch.long
    )
    parent_child_relations = tuple(
        (index, int(node.left), int(node.right))
        for index, node in enumerate(typed.tree.nodes)
        if node.left is not None and node.right is not None
    )

    for mechanism in INTERACTION_MECHANISMS:
        matrix = typed.permission_matrix(
            mechanism,
            phase=selected_phase,
            module_present=module_present,
        )
        permission_matrices.append((mechanism, matrix))
        detached_matrix = matrix.detach()
        if mechanism in {"MM", "EM", "QM"}:
            expected = (module_present > 0.5).to(detached_matrix.dtype)
        else:
            expected = detached_matrix.new_ones((detached_matrix.shape[1],))
        # A native all-access bypass must stay safe for live receiver queries
        # outside the anchor sample. Anchor-inactive nodes can still be reached
        # by a perturbed receiver, so validate every structurally reachable
        # hard node rather than only the anchor-active frontier.
        is_full = bool(hard_execution_node_indices) and torch.equal(
            detached_matrix.index_select(0, hard_execution_index_tensor),
            expected[None, :].expand(len(hard_execution_node_indices), -1),
        )
        full_access_flags.append((mechanism, is_full))
        differentiable_flags.append((
            mechanism,
            bool(typed.split_gates.requires_grad or matrix.requires_grad),
        ))
        key_missing_flags.append((
            mechanism,
            typed.permission_status(mechanism, phase=selected_phase)
            == "full_access_bypass_missing_key",
        ))

        with torch.no_grad():
            positive = detached_matrix > 0
            source_index_rows: list[tuple[int, ...]] = [() for _ in typed.tree.nodes]
            for row in active_node_indices:
                source_index_rows[row] = tuple(
                    int(value)
                    for value in torch.nonzero(positive[row], as_tuple=False).flatten().cpu().tolist()
                )
            source_indices = tuple(source_index_rows)
            active_anchor_support = anchor_access.index_select(
                1, active_index_tensor
            ) > 0
            active_source_support = positive.index_select(0, active_index_tensor)
            effective_support = (
                active_anchor_support.to(torch.float32)
                @ active_source_support.to(torch.float32)
            ) > 0
            source_union_count = int(effective_support.any(dim=0).sum().cpu())
            unique_pairs = int(effective_support.sum().cpu())
            active_bearing = active_mask & positive.any(dim=1)
            bearing_ids = tuple(
                int(index) for index in torch.nonzero(active_bearing, as_tuple=False).flatten().cpu().tolist()
            )
            bearing_rows = detached_matrix.index_select(
                0, torch.as_tensor(bearing_ids, device=module_present.device, dtype=torch.long)
            ).cpu().tolist()
            classes: dict[tuple[float, ...], list[int]] = {}
            for index, row in zip(bearing_ids, bearing_rows, strict=True):
                # Float tuples preserve the exact stored permission value;
                # no thresholding or approximate mask match is used.
                classes.setdefault(tuple(float(value) for value in row), []).append(index)
            action_groups = tuple(tuple(indices) for indices in classes.values())
            summary = CoverFrontierSummary(
                mechanism=mechanism,
                phase=selected_phase,
                candidate_node_count=len(typed.tree.nodes),
                raw_active_frontier_nodes=len(active_node_indices),
                source_bearing_active_nodes=len(bearing_ids),
                nonredundant_packet_count=len(action_groups),
                source_union_count=source_union_count,
                unique_source_receiver_pairs=unique_pairs,
            )

        source_indices_by_mechanism.append((mechanism, source_indices))
        compact_action_classes.append((mechanism, action_groups))
        frontier_summaries.append((mechanism, summary))

    transition_count = sum(
        0.0 < float(value) < 1.0
        for index, value in enumerate(typed.split_gates.detach().cpu().tolist())
        if typed.tree.nodes[index].left is not None
    )
    return CompiledMechanismExecutionView(
        plan_identity=id(plan),
        phase=selected_phase,
        tensor_versions=_compiled_plan_tensor_versions(plan),
        active_node_indices=active_node_indices,
        parent_child_relations=parent_child_relations,
        hard_execution_node_mask=hard_execution_node_mask,
        split_axes=split_axes,
        children_by_node=children_by_node,
        split_boundaries=tuple(split_boundaries),
        overlap_widths=tuple(overlap_widths),
        coordinate_dim=int(typed.tree.universe.coordinates.shape[1]),
        device=typed.tree.universe.coordinates.device,
        full_access_flags=tuple(full_access_flags),
        differentiable_flags=tuple(differentiable_flags),
        key_missing_flags=tuple(key_missing_flags),
        permission_matrices=tuple(permission_matrices),
        source_indices_by_mechanism=tuple(source_indices_by_mechanism),
        compact_action_classes=tuple(compact_action_classes),
        frontier_summaries=tuple(frontier_summaries),
        candidate_node_count=len(typed.tree.nodes),
        capacity_saturated=bool(typed.tree.capacity_saturated),
        transition_count=transition_count,
    )


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
    "CompiledMechanismExecutionView",
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
    "compile_mechanism_execution_view",
    "endpoint_smoothstep",
]
