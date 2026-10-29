"""Task-trained, phase-conditioned retention of proposal-function details.

The controller changes only access-function differences.  Physical source
incidence and source-resolved control moments are kept by the interface-field
backend, which may quotient them exactly when a deterministic subtree closes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from .functional_fusion_tree import pack_exact_closed_subtrees

Tensor = torch.Tensor


def _smoothstep(value: Tensor) -> Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def _ordered_binary_tree(node_membership: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Validate and canonically order a complete binary tree over K leaves.

    Returns bottom-up node membership, two child masks per node, child
    references (nonnegative for an internal node, ``-leaf-1`` for a leaf),
    and a node-by-node ancestor mask.  Canonical bottom-up ordering keeps the
    existing exact packer rule that the greatest closed node is the highest
    closed ancestor.
    """

    if not isinstance(node_membership, torch.Tensor) or node_membership.ndim != 2:
        raise ValueError("node_membership must be a rank-two tensor [N,K].")
    if node_membership.dtype != torch.bool:
        node_membership = node_membership.to(dtype=torch.bool)
    nodes, groups = map(int, node_membership.shape)
    if groups < 2 or nodes != groups - 1 or not bool(node_membership.any(dim=-1).all()):
        raise ValueError("a complete binary tree requires exactly K-1 nonempty nodes.")
    rows = [
        tuple(torch.nonzero(node_membership[index], as_tuple=False).flatten().tolist())
        for index in range(nodes)
    ]
    if len(set(rows)) != nodes:
        raise ValueError("tree nodes must have unique leaf subsets.")
    root = tuple(range(groups))
    if root not in rows:
        raise ValueError("tree must contain a root covering every proposal leaf.")
    row_sets = [set(row) for row in rows]
    for left in range(nodes):
        for right in range(left + 1, nodes):
            intersection = row_sets[left] & row_sets[right]
            if intersection and not (
                row_sets[left] <= row_sets[right]
                or row_sets[right] <= row_sets[left]
            ):
                raise ValueError("tree subsets must be laminar (nested or disjoint).")

    # Reorder independently of configuration row order, then rebuild indices.
    order = sorted(range(nodes), key=lambda index: (len(rows[index]), rows[index]))
    ordered_rows = [rows[index] for index in order]
    node_index_by_row = {row: index for index, row in enumerate(ordered_rows)}
    child_masks = torch.zeros((nodes, 2, groups), dtype=torch.bool)
    child_refs = torch.empty((nodes, 2), dtype=torch.long)
    for index, row in enumerate(ordered_rows):
        members = set(row)
        proper_nodes = [
            set(candidate)
            for candidate in ordered_rows
            if len(candidate) < len(row) and set(candidate) < members
        ]
        maximal_children = [
            candidate
            for candidate in proper_nodes
            if not any(candidate < other < members for other in proper_nodes)
        ]
        covered = set().union(*maximal_children) if maximal_children else set()
        child_sets = [*maximal_children, *({leaf} for leaf in members - covered)]
        child_sets.sort(key=lambda values: min(values))
        if len(child_sets) != 2 or set.union(*child_sets) != members:
            raise ValueError("each tree node must have exactly two disjoint children.")
        if child_sets[0] & child_sets[1]:
            raise ValueError("tree children must be disjoint.")
        for child_index, child in enumerate(child_sets):
            child_mask = torch.zeros(groups, dtype=torch.bool)
            child_mask[list(child)] = True
            child_masks[index, child_index] = child_mask
            child_row = tuple(sorted(child))
            child_refs[index, child_index] = node_index_by_row.get(
                child_row,
                -(child_row[0] + 1),
            )

    ordered_membership = torch.zeros((nodes, groups), dtype=torch.bool)
    for index, row in enumerate(ordered_rows):
        ordered_membership[index, list(row)] = True
    ancestor = torch.zeros((nodes, nodes), dtype=torch.bool)
    for contrast_index, contrast_row in enumerate(ordered_rows):
        contrast = set(contrast_row)
        for candidate_index, candidate_row in enumerate(ordered_rows):
            ancestor[candidate_index, contrast_index] = contrast <= set(candidate_row)
    return ordered_membership, child_masks, child_refs, ancestor


def build_haar_basis(node_membership: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build an orthonormal Haar basis and canonical tree metadata."""

    membership, child_masks, child_refs, ancestor = _ordered_binary_tree(
        node_membership.detach().to(device="cpu", dtype=torch.bool)
    )
    nodes, groups = map(int, membership.shape)
    basis = torch.zeros((groups, groups), dtype=torch.float64)
    basis[:, 0] = 1.0 / math.sqrt(groups)
    for node_index in range(nodes):
        left = child_masks[node_index, 0]
        right = child_masks[node_index, 1]
        n_left = int(left.sum())
        n_right = int(right.sum())
        total = n_left + n_right
        basis[left, node_index + 1] = math.sqrt(n_right / (n_left * total))
        basis[right, node_index + 1] = -math.sqrt(n_left / (n_right * total))
    if not torch.allclose(
        basis.transpose(0, 1) @ basis,
        torch.eye(groups, dtype=basis.dtype),
        atol=2.0e-14,
        rtol=2.0e-14,
    ):
        raise RuntimeError("constructed tree Haar basis is not orthonormal.")
    return basis, membership, child_masks, ancestor


def expected_class_count(
    node_probability: Tensor,
    active: Tensor,
    node_membership: Tensor,
    child_refs: Tensor,
    closure_eligible: Tensor,
) -> Tensor:
    """Expected exact quotient width under independent node closures."""

    if node_probability.ndim != 2:
        raise ValueError("node_probability must have shape [B,N].")
    batch, nodes = map(int, node_probability.shape)
    if active.ndim != 2 or int(active.shape[0]) != batch:
        raise ValueError("active must have shape [B,K].")
    if tuple(node_membership.shape) != (nodes, int(active.shape[1])):
        raise ValueError("node_membership must align with node_probability and active.")
    if tuple(child_refs.shape) != (nodes, 2):
        raise ValueError("child_refs must have shape [N,2].")
    if tuple(closure_eligible.shape) != (batch, nodes):
        raise ValueError("closure_eligible must have shape [B,N].")
    active = active.to(dtype=torch.bool)
    probability = torch.where(
        closure_eligible,
        node_probability,
        torch.ones_like(node_probability),
    )
    node_active = (active[:, None, :] & node_membership[None, :, :]).any(dim=-1)
    expected_by_node: list[Tensor | None] = [None] * nodes
    for node_index in range(nodes):  # membership rows are canonical bottom-up
        child_values: list[Tensor] = []
        for child in child_refs[node_index].tolist():
            if child >= 0:
                child_value = expected_by_node[child]
                if child_value is None:
                    raise RuntimeError("tree child order is not bottom-up.")
                child_values.append(child_value)
            else:
                child_values.append(active[:, -child - 1].to(node_probability.dtype))
        split = child_values[0] + child_values[1]
        expected_by_node[node_index] = (
            probability[:, node_index] * split
            + (1.0 - probability[:, node_index])
            * node_active[:, node_index].to(node_probability.dtype)
        )
    root_index = int(node_membership.sum(dim=-1).argmax().item())
    result = expected_by_node[root_index]
    if result is None:
        raise RuntimeError("tree root expected class count was not computed.")
    return result


@dataclass(frozen=True)
class FunctionalDetailPlan:
    """One prepared phase's live detail controls and exact closure metadata."""

    logits: Tensor
    probability: Tensor
    base_retention: Tensor
    retention: Tensor
    eligibility: Tensor
    transform: Tensor
    closed_nodes: Tensor
    actual_R: Tensor
    expected_R: Tensor
    expected_complexity: Tensor
    stochastic_mask: Tensor
    uniform: Tensor | None
    ramp: Tensor


class LearnedFunctionalDetailController(nn.Module):
    """Shared conditional hard-concrete control over fixed tree nodes."""

    lower = -0.1
    upper = 1.1
    beta = 2.0 / 3.0
    eligibility_low = 1.0e-6
    eligibility_high = 2.0e-6

    def __init__(
        self,
        node_membership: Tensor,
        descriptor_dim: int,
        *,
        hidden_dim: int = 32,
        initial_logit: float = 1.6,
    ) -> None:
        super().__init__()
        if int(descriptor_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("descriptor_dim and hidden_dim must be positive.")
        basis, membership, child_masks, ancestor = build_haar_basis(node_membership)
        _, _, child_refs, _ = _ordered_binary_tree(
            node_membership.detach().to(device="cpu", dtype=torch.bool)
        )
        self.descriptor_dim = int(descriptor_dim)
        self.group_count = int(membership.shape[1])
        self.node_count = int(membership.shape[0])
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.descriptor_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=1.0e-2)
        nn.init.constant_(self.network[-1].bias, float(initial_logit))
        self.register_buffer("node_membership", membership, persistent=True)
        self.register_buffer("child_masks", child_masks, persistent=True)
        self.register_buffer("child_refs", child_refs, persistent=True)
        self.register_buffer("ancestor_mask", ancestor, persistent=True)
        self.register_buffer("haar_basis", basis, persistent=True)

    def forward(self, descriptors: Tensor) -> Tensor:
        if descriptors.ndim != 3:
            raise ValueError("descriptors must have shape [B,N,F].")
        if int(descriptors.shape[1]) != self.node_count:
            raise ValueError("descriptor node count does not match the fixed tree.")
        if int(descriptors.shape[2]) != self.descriptor_dim:
            raise ValueError("descriptor width does not match descriptor_dim.")
        return self.network(descriptors).squeeze(-1)

    @staticmethod
    def _proposal_eligibility(proposal_mass: Tensor) -> Tensor:
        return _smoothstep(
            (proposal_mass - LearnedFunctionalDetailController.eligibility_low)
            / (
                LearnedFunctionalDetailController.eligibility_high
                - LearnedFunctionalDetailController.eligibility_low
            )
        )

    @classmethod
    def positive_probability(cls, logits: Tensor) -> Tensor:
        shift = -cls.beta * math.log(-cls.lower / cls.upper)
        return torch.sigmoid(logits + shift)

    @classmethod
    def deterministic_retention(cls, logits: Tensor) -> Tensor:
        z = cls.lower + (cls.upper - cls.lower) * torch.sigmoid(logits)
        z = z.clamp(0.0, 1.0)
        return z.square() * (3.0 - 2.0 * z)

    @classmethod
    def stochastic_retention(cls, logits: Tensor, uniform: Tensor) -> Tensor:
        if tuple(uniform.shape) != tuple(logits.shape):
            raise ValueError("uniform must have the same shape as logits [B,N].")
        if not uniform.is_floating_point():
            raise TypeError("uniform must be floating point.")
        eps = torch.finfo(logits.dtype).eps
        bounded = uniform.to(device=logits.device, dtype=logits.dtype).clamp(
            min=eps, max=1.0 - eps
        )
        logistic_noise = torch.log(bounded) - torch.log1p(-bounded)
        v = torch.sigmoid((logistic_noise + logits) / cls.beta)
        z = (cls.lower + (cls.upper - cls.lower) * v).clamp(0.0, 1.0)
        return z.square() * (3.0 - 2.0 * z)

    def transform_from_retention(self, retention: Tensor) -> Tensor:
        if tuple(retention.shape[1:]) != (self.node_count,):
            raise ValueError("retention must have shape [B,N].")
        membership = self.node_membership.to(device=retention.device)
        ancestors = self.ancestor_mask.to(device=retention.device)
        safe_factors = torch.where(
            ancestors[None, :, :],
            retention[:, :, None],
            torch.ones_like(retention[:, :, None]),
        )
        detail_scales = safe_factors.prod(dim=1)
        scales = torch.cat(
            [torch.ones_like(detail_scales[:, :1]), detail_scales], dim=-1
        )
        basis = self.haar_basis.to(device=retention.device, dtype=retention.dtype)
        transform = (basis[None, :, :] * scales[:, None, :]) @ basis.transpose(0, 1)
        # Preserve the parent Run-1502 route operation exactly when every
        # detail is fully retained.  This uses literal endpoint equality;
        # small positive retentions remain open and are never treated as a
        # closure by a numerical-rank test.
        all_open = (retention == 1.0).all(dim=-1)
        identity = torch.eye(
            self.group_count, device=retention.device, dtype=retention.dtype
        )
        return torch.where(all_open[:, None, None], identity, transform)

    def _actual_class_count(self, closed_nodes: Tensor, active: Tensor) -> Tensor:
        batch = int(active.shape[0])
        node_active = (
            active[:, None, :].to(dtype=torch.bool)
            & self.node_membership.to(device=active.device)[None, :, :]
        ).any(dim=-1)
        by_node: list[Tensor | None] = [None] * self.node_count
        for node_index in range(self.node_count):
            pieces: list[Tensor] = []
            for child in self.child_refs[node_index].tolist():
                if child >= 0:
                    value = by_node[child]
                    if value is None:
                        raise RuntimeError("tree child order is not bottom-up.")
                    pieces.append(value)
                else:
                    pieces.append(active[:, -child - 1].to(torch.long))
            open_count = pieces[0] + pieces[1]
            by_node[node_index] = torch.where(
                closed_nodes[:, node_index] & node_active[:, node_index],
                node_active[:, node_index].to(torch.long),
                open_count,
            )
        root_index = int(self.node_membership.sum(dim=-1).argmax().item())
        result = by_node[root_index]
        if result is None:
            raise RuntimeError("tree root class count was not computed.")
        return result

    def plan(
        self,
        descriptors: Tensor,
        active: Tensor,
        proposal_mass: Tensor,
        *,
        stochastic_mask: Tensor,
        uniform: Tensor | None = None,
        ramp: float | Tensor = 1.0,
    ) -> FunctionalDetailPlan:
        """Build a predictive plan and detached-input live complexity signal."""

        if active.ndim != 2 or tuple(active.shape[1:]) != (self.group_count,):
            raise ValueError("active must have shape [B,K] matching the tree.")
        if tuple(proposal_mass.shape) != tuple(active.shape):
            raise ValueError("proposal_mass must match active with shape [B,K].")
        if tuple(stochastic_mask.shape) != (int(active.shape[0]),):
            raise ValueError("stochastic_mask must have shape [B].")
        active_mask = active.to(dtype=torch.bool)
        stochastic_mask = stochastic_mask.to(device=active.device, dtype=torch.bool)
        logits = self.forward(descriptors)
        if uniform is None and bool(stochastic_mask.any()):
            uniform = torch.rand_like(logits)
        if uniform is None:
            base_retention = self.deterministic_retention(logits)
        else:
            if tuple(uniform.shape) != tuple(logits.shape):
                raise ValueError("uniform must have shape [B,N].")
            sampled = self.stochastic_retention(logits, uniform)
            deterministic = self.deterministic_retention(logits)
            base_retention = torch.where(stochastic_mask[:, None], sampled, deterministic)

        proposal_eligibility = self._proposal_eligibility(proposal_mass)
        member_mask = self.node_membership.to(device=active.device)
        eligibility_factors = torch.where(
            member_mask[None, :, :],
            proposal_eligibility[:, None, :],
            torch.ones_like(proposal_eligibility[:, None, :]),
        )
        all_active = ((~member_mask[None, :, :]) | active_mask[:, None, :]).all(dim=-1)
        eligibility = eligibility_factors.prod(dim=-1) * all_active.to(proposal_mass.dtype)
        ramp_tensor = torch.as_tensor(ramp, device=logits.device, dtype=logits.dtype)
        if ramp_tensor.ndim == 0:
            ramp_tensor = ramp_tensor.expand(logits.shape[0])
        if tuple(ramp_tensor.shape) != (int(logits.shape[0]),):
            raise ValueError("ramp must be scalar or have shape [B].")
        ramp_tensor = ramp_tensor.clamp(0.0, 1.0)
        retention = 1.0 - (
            ramp_tensor[:, None]
            * eligibility.to(dtype=logits.dtype)
            * (1.0 - base_retention)
        )
        transform = self.transform_from_retention(retention)
        closed_nodes = retention == 0.0
        actual_R = self._actual_class_count(closed_nodes, active_mask)

        # Complexity observes the same current phase descriptors, but stops
        # gradients into source features, masses, centers, and global control.
        complexity_logits = self.forward(descriptors.detach())
        positive_probability = self.positive_probability(complexity_logits)
        closure_eligible = all_active & (eligibility == 1.0)
        expected_R = expected_class_count(
            positive_probability,
            active_mask,
            member_mask,
            self.child_refs.to(device=active.device),
            closure_eligible,
        )
        occupied_count = active_mask.sum(dim=-1).to(expected_R.dtype)
        expected_complexity = torch.where(
            occupied_count > 1.0,
            (expected_R - 1.0) / (occupied_count - 1.0).clamp_min(1.0),
            torch.zeros_like(expected_R),
        )
        return FunctionalDetailPlan(
            logits=logits,
            probability=self.positive_probability(logits),
            base_retention=base_retention,
            retention=retention,
            eligibility=eligibility,
            transform=transform,
            closed_nodes=closed_nodes,
            actual_R=actual_R,
            expected_R=expected_R,
            expected_complexity=expected_complexity,
            stochastic_mask=stochastic_mask,
            uniform=uniform,
            ramp=ramp_tensor,
        )

    def pack(self, plan: FunctionalDetailPlan, active: Tensor):
        """Return exact integer classes from literal zero-retention nodes."""

        return pack_exact_closed_subtrees(
            plan.closed_nodes,
            self.node_membership.to(device=active.device),
            active.to(dtype=torch.bool),
        )


__all__ = [
    "FunctionalDetailPlan",
    "LearnedFunctionalDetailController",
    "build_haar_basis",
    "expected_class_count",
]
