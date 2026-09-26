"""Case-local receiver cover and exact fine-pair work accounting.

This module owns the geometry-only candidate hierarchy and the live access
algebra. It does not infer physical effects from its support: oracle proposals
and an input-only amortizer must be judged against separate response evidence.
The hierarchy is built once per case and kept fixed over a local design trust
region; rebuilding it is a separately measured boundary event.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .routing_index.types import PackedPairs


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
            gate = endpoint_smoothstep(split_gates[index])
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


def compile_cover_pairs(
    plan: AdaptiveCoverPlan,
    queries: torch.Tensor,
    *,
    module_present: torch.Tensor,
    environment_weights: torch.Tensor,
    empty_environment: str = "fallback_full",
) -> tuple[PackedPairs, PackedPairs, CoverPairLedger]:
    """Compile one case's positive union once, with live QM/QE priors.

    QM priors use ``rho/M`` so the inherited packed reader's
    ``M/(1+M)`` factor reproduces Dense at full access. QE priors keep
    quadrature separate from access until their product enters one global
    attention normalization. A no-support QE query explicitly falls back to
    full access and is counted; silent zero-output attention is forbidden.
    """

    if empty_environment != "fallback_full":
        raise ValueError("only explicit full-access empty-environment fallback is supported")
    access = plan.access(queries)
    if module_present.shape != (access.module_source.shape[1],):
        raise ValueError("module validity must have shape [M]")
    if environment_weights.numel() == 0:
        raise ValueError("the packed environmental reader requires nonempty positive quadrature")
    if environment_weights.shape != (access.environment_source.shape[1],):
        raise ValueError("environment weights must have shape [E]")
    if bool((environment_weights <= 0).any()) or not bool(torch.isfinite(environment_weights).all()):
        raise ValueError("environment weights must be finite and positive")
    module_valid = module_present > 0.5
    module_count = module_valid.sum().clamp_min(1)
    module_prior = access.module_source * module_valid.to(access.module_source.dtype) / module_count
    environment_prior = access.environment_source * environment_weights
    # The all-zero environment needs a defined attention denominator. Ramp
    # the fallback in *before* that endpoint so the field has a finite
    # one-sided limit while positive source support vanishes. Near-empty
    # rows pay for all E sources and are counted as fallback work.
    safety_mass = environment_weights.sum() * 1.0e-4
    environmental_mass = environment_prior.sum(dim=1)
    fallback = environmental_mass < safety_mass
    safety_weight = endpoint_smoothstep(
        (safety_mass - environmental_mass) / safety_mass
    )
    environment_prior = environment_prior + safety_weight[:, None] * environment_weights[None, :]

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

    active_paths = access.receiver_group > 0
    qm_paths = int(((active_paths[:, :, None]) & (plan.module_membership[None] > 0) & module_valid[None, None]).sum())
    qe_paths = int(((active_paths[:, :, None]) & (plan.environment_membership[None] > 0)).sum())
    qm = pack(module_prior, qm_paths)
    qe = pack(environment_prior, qe_paths)
    source_nonempty = (plan.module_membership > 0).any(dim=1) | (
        plan.environment_membership > 0
    ).any(dim=1)
    query_degree = ((access.receiver_group > 0) & source_nonempty[None, :]).sum(dim=1)
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
    )
    return qm, qe, ledger


__all__ = [
    "AdaptiveCoverPlan",
    "CandidateNode",
    "CaseLocalReceiverTree",
    "CoverAccess",
    "CoverPairLedger",
    "ReceiverAnchorUniverse",
    "compile_cover_pairs",
    "endpoint_smoothstep",
]
