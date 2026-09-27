"""Fixed-plan controls for packet refinement, quotienting, and source selection.

These helpers consume the native receiver tree, typed permission matrices,
query coordinates, and source axes already carried by a MechanismPlan. They
never alter source coordinates, source states, or quadrature measures.

The split helper preserves the exact forward access at initialization while
leaving child permission logits independently differentiable when the input
plan carries straight-through organizer gradients. The quotient and root
union helpers are evaluation controls over a fixed plan. Direct-pair scoring
is intentionally separate: callers must supply per-receiver/source scores
from a separately trained input-only scorer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from .adaptive_interaction_cover import (
    INTERACTION_MECHANISMS,
    InteractionPermissionKey,
    MechanismPlan,
)


@dataclass(frozen=True)
class CanonicalPacketQuotient:
    """Exact action classes for one fixed plan and a supplied receiver panel.

    Packet access is aggregated only when every resolved typed action is
    elementwise equal. The stored access and actions reproduce the original
    effective access for all five mechanisms and every explicit phase.
    This quotient is panel-specific and is not a global source-union control.
    """

    receiver_access: torch.Tensor  # [Q,P], sum of receiver access by class
    permission_actions: Mapping[str, torch.Tensor]  # [P,S] per typed context
    node_groups: tuple[tuple[int, ...], ...]

    @property
    def packet_count(self) -> int:
        return len(self.node_groups)

    def access_for(self, mechanism: str, *, phase: str | None = None) -> torch.Tensor:
        """Reconstruct one typed query/source access matrix from the quotient."""

        key = InteractionPermissionKey(mechanism, phase).canonical_name
        try:
            actions = self.permission_actions[key]
        except KeyError as exc:
            raise ValueError(f"unknown typed permission context {key!r}") from exc
        return self.receiver_access @ actions


@dataclass(frozen=True)
class DirectPairBudgetProjection:
    """Independent direct-pair support selected under an exact row budget."""

    selected_pairs: torch.Tensor  # [Q,S], boolean
    requested_work: int
    achieved_work: int
    feasible_capacity: int
    exact_match: bool

    @property
    def closest_feasible_work(self) -> int:
        """Return the feasible pair count nearest the requested work."""

        return self.achieved_work

    @property
    def work_shortfall(self) -> int:
        return max(0, self.requested_work - self.achieved_work)


def _permission_matrices(plan: MechanismPlan) -> dict[str, torch.Tensor]:
    phases = sorted({key.phase for key in plan.permissions if key.phase is not None})
    contexts = [(mechanism, None) for mechanism in INTERACTION_MECHANISMS]
    contexts.extend(
        (mechanism, phase)
        for phase in phases
        for mechanism in INTERACTION_MECHANISMS
    )
    return {
        InteractionPermissionKey(mechanism, phase).canonical_name:
            plan.permission_matrix(mechanism, phase=phase)
        for mechanism, phase in contexts
    }


def _validate_receivers(plan: MechanismPlan, receivers: torch.Tensor) -> None:
    if receivers.ndim != 2 or receivers.shape[1] != plan.tree.universe.coordinates.shape[1]:
        raise ValueError("receivers must have shape [Q,d] in the plan's native anchor frame")
    if receivers.shape[0] < 1:
        raise ValueError("at least one native receiver is required")
    if receivers.device != plan.split_gates.device:
        raise ValueError("receivers and plan must use one device")
    if not bool(torch.isfinite(receivers).all()):
        raise ValueError("receivers must be finite")


def _forward_value_with_input_gradient(value: torch.Tensor, target: float) -> torch.Tensor:
    """Set an exact forward scalar while retaining the input's local gradient."""

    return torch.full_like(value, target) + (value - value.detach())


def _tree_depths(plan: MechanismPlan) -> tuple[int, ...]:
    if not plan.tree.nodes:
        raise ValueError("receiver tree must contain a root")
    depths = [-1] * len(plan.tree.nodes)
    stack = [(0, 0)]
    while stack:
        index, depth = stack.pop()
        if index < 0 or index >= len(plan.tree.nodes) or depths[index] >= 0:
            raise ValueError("receiver tree has an invalid or repeated child index")
        depths[index] = depth
        node = plan.tree.nodes[index]
        if (node.left is None) != (node.right is None):
            raise ValueError("receiver tree nodes must have zero or two children")
        if node.left is not None and node.right is not None:
            stack.append((node.right, depth + 1))
            stack.append((node.left, depth + 1))
    if any(depth < 0 for depth in depths):
        raise ValueError("receiver tree contains nodes disconnected from its root")
    return tuple(depths)


def _function_preserving_split(
    plan: MechanismPlan,
    node_index: int,
    *,
    require_deeper: bool,
) -> MechanismPlan:
    """Open one active split with both children inheriting its action.

    The target must currently be closed, so its children and descendants carry
    no forward access. The split is opened with an exact unit forward gate;
    all lower split gates are closed for this initial two-child refinement.
    Each explicit typed permission row is copied to both children exactly in
    the forward pass. If the input rows carry straight-through gradients, each
    child retains its own gradient path so prediction evidence can separate
    their supports. This helper changes no physical/query/source data.
    """

    if not isinstance(node_index, int) or isinstance(node_index, bool):
        raise TypeError("node_index must be an integer")
    if node_index < 0 or node_index >= len(plan.tree.nodes):
        raise IndexError("node_index is outside the candidate tree")
    node = plan.tree.nodes[node_index]
    if node.is_leaf or node.left is None or node.right is None:
        raise ValueError("only an internal receiver node can be split")
    if require_deeper and _tree_depths(plan)[node_index] == 0:
        raise ValueError("the split control must refine below the root")
    if float(plan.split_gates[node_index].detach()) != 0.0:
        raise ValueError("the selected node must have a closed split gate")

    anchor_access = plan.tree.access(
        plan.tree.universe.coordinates,
        plan.split_gates.detach(),
    )
    if not bool((anchor_access[:, node_index] > 0).any()):
        raise ValueError("the selected node must be active on the native anchor universe")

    descendant_indices: set[int] = set()
    pending = [node_index]
    while pending:
        current = pending.pop()
        current_node = plan.tree.nodes[current]
        if current_node.left is not None and current_node.right is not None:
            pending.extend((current_node.left, current_node.right))
            if current != node_index:
                descendant_indices.add(current)

    gate_values = []
    for index, gate in enumerate(plan.split_gates):
        if index == node_index:
            gate_values.append(_forward_value_with_input_gradient(gate, 1.0))
        elif index in descendant_indices:
            gate_values.append(_forward_value_with_input_gradient(gate, 0.0))
        else:
            gate_values.append(gate)
    split_gates = torch.stack(gate_values)

    permissions: dict[InteractionPermissionKey, torch.Tensor] = {}
    for key, matrix in plan.permissions.items():
        rows = list(matrix.unbind(dim=0))
        parent_row = matrix[node_index].detach()
        for child_index in (node.left, node.right):
            # The child receives the parent's exact forward action and keeps
            # its own score gradient for subsequent train-only adaptation.
            rows[child_index] = parent_row + (
                matrix[child_index] - matrix[child_index].detach()
            )
        permissions[key] = torch.stack(rows)

    return MechanismPlan(
        plan.tree,
        split_gates,
        plan.module_present,
        plan.environment_count,
        permissions,
    )


def function_preserving_deeper_split(
    plan: MechanismPlan,
    node_index: int,
) -> MechanismPlan:
    """Open one active non-root split while preserving exact forward access."""

    return _function_preserving_split(plan, node_index, require_deeper=True)


def function_preserving_root_split(plan: MechanismPlan) -> MechanismPlan:
    """Function-preservingly open the root before a deeper refinement if needed."""

    return _function_preserving_split(plan, 0, require_deeper=False)


def canonical_packet_quotient(
    plan: MechanismPlan,
    receivers: torch.Tensor,
) -> CanonicalPacketQuotient:
    """Group exact typed actions and sum their access over a receiver panel.

    Equality is checked on the original scalar values, without rounding or
    approximate mask comparison. The result is a fixed-plan forward quotient;
    it intentionally does not rewrite autograd paths.
    """

    if plan.split_gates.requires_grad or plan.module_present.requires_grad or any(
        value.requires_grad for value in plan.permissions.values()
    ):
        raise ValueError("canonical packet quotient requires a detached evaluation plan")
    if receivers.requires_grad:
        raise ValueError("canonical packet quotient requires fixed native receivers")
    _validate_receivers(plan, receivers)

    alpha = plan.tree.access(receivers, plan.split_gates)
    active = (alpha > 0).any(dim=0).detach().cpu().tolist()
    matrices = _permission_matrices(plan)
    cpu_matrices = {
        key: matrix.detach().cpu()
        for key, matrix in matrices.items()
    }
    grouped_nodes: list[list[int]] = []
    for node_index, is_active in enumerate(active):
        if not is_active:
            continue
        if not any(bool((matrix[node_index] > 0).any()) for matrix in cpu_matrices.values()):
            continue
        for group in grouped_nodes:
            representative = group[0]
            if all(
                torch.equal(matrix[node_index], matrix[representative])
                for matrix in cpu_matrices.values()
            ):
                group.append(node_index)
                break
        else:
            grouped_nodes.append([node_index])

    node_groups = tuple(tuple(indices) for indices in grouped_nodes)
    if node_groups:
        receiver_access = torch.stack(
            [alpha[:, list(group)].sum(dim=1) for group in node_groups],
            dim=1,
        )
        permission_actions = {
            key: torch.stack([matrix[group[0]] for group in node_groups], dim=0)
            for key, matrix in matrices.items()
        }
    else:
        receiver_access = alpha.new_empty((int(receivers.shape[0]), 0))
        permission_actions = {
            key: matrix.new_empty((0, int(matrix.shape[1])))
            for key, matrix in matrices.items()
        }
    return CanonicalPacketQuotient(
        receiver_access=receiver_access,
        permission_actions=MappingProxyType(permission_actions),
        node_groups=node_groups,
    )


def _hard_frontier_indices(plan: MechanismPlan) -> tuple[int, ...]:
    """Return the structural frontier of a detached binary split tree."""

    frontier: list[int] = []
    pending = [0]
    while pending:
        node_index = pending.pop()
        node = plan.tree.nodes[node_index]
        if node.is_leaf or float(plan.split_gates[node_index]) == 0.0:
            frontier.append(node_index)
        else:
            if node.left is None or node.right is None:
                raise ValueError("an open internal split must have two children")
            pending.extend((node.right, node.left))
    return tuple(frontier)


def root_union_plan(plan: MechanismPlan) -> MechanismPlan:
    """Build a fixed root packet with each typed source support's frontier union.

    The union uses the entire structural frontier of the case tree, independent
    of any evaluation stencil. Thus a packet whose geometric access is zero
    on one finite panel can still contribute if its branch is open elsewhere.
    The input must be a detached hard plan; source support is unioned as exact
    binary membership. Missing permission keys remain missing and therefore
    retain the declared full-access default.
    """

    if plan.split_gates.requires_grad or any(
        value.requires_grad for value in plan.permissions.values()
    ):
        raise ValueError("root-union control requires a detached evaluation plan")
    if plan.module_present.requires_grad:
        raise ValueError("root-union control requires fixed module validity")
    if not bool(((plan.split_gates == 0) | (plan.split_gates == 1)).all()):
        raise ValueError("root-union control requires hard binary split gates")
    if any(
        not bool(((value == 0) | (value == 1)).all())
        for value in plan.permissions.values()
    ):
        raise ValueError("root-union control requires hard binary permissions")

    frontier = _hard_frontier_indices(plan)
    permissions: dict[InteractionPermissionKey, torch.Tensor] = {}
    for key, original in plan.permissions.items():
        resolved = plan.permission_matrix(key.mechanism, phase=key.phase)
        union = (
            (resolved[list(frontier)] > 0).any(dim=0).to(dtype=original.dtype)
            if frontier
            else torch.zeros_like(original[0])
        )
        root_rows = [torch.zeros_like(original[index]) for index in range(int(original.shape[0]))]
        root_rows[0] = union
        permissions[key] = torch.stack(root_rows)

    return MechanismPlan(
        plan.tree,
        torch.zeros_like(plan.split_gates),
        plan.module_present,
        plan.environment_count,
        permissions,
    )


def project_direct_pair_budget(
    direct_scores: torch.Tensor,
    requested_work: int,
    *,
    eligible_pairs: torch.Tensor | None = None,
) -> DirectPairBudgetProjection:
    """Select independently scored query/source pairs at the closest exact work.

    Each true output entry is one logical query/source row. Scores must come
    from an independently trained input-only direct scorer; packet-organizer
    logits are not implicitly accepted or interpreted here. Ties use stable
    row-major query/source order. If the requested work exceeds the eligible
    capacity, all eligible rows are selected and the shortfall is reported.
    The boolean support leaves original physical source measures untouched.
    """

    if direct_scores.ndim != 2 or min(direct_scores.shape) < 1:
        raise ValueError("direct scores must have shape [Q,S] with positive axes")
    if direct_scores.requires_grad:
        raise ValueError("direct-pair budget projection requires detached checkpoint scores")
    if isinstance(requested_work, bool) or not isinstance(requested_work, int):
        raise TypeError("requested_work must be an integer logical pair count")
    if requested_work < 0:
        raise ValueError("requested_work cannot be negative")
    if not bool(torch.isfinite(direct_scores).all()):
        raise ValueError("direct scores must be finite")

    if eligible_pairs is None:
        eligible = torch.ones_like(direct_scores, dtype=torch.bool)
    else:
        if eligible_pairs.shape != direct_scores.shape or eligible_pairs.dtype != torch.bool:
            raise ValueError("eligible_pairs must be a boolean mask matching direct_scores")
        if eligible_pairs.device != direct_scores.device:
            raise ValueError("eligible_pairs and direct_scores must use one device")
        eligible = eligible_pairs

    candidate_indices = torch.arange(
        direct_scores.numel(),
        device=direct_scores.device,
        dtype=torch.long,
    )[eligible.reshape(-1)]
    candidate_scores = direct_scores.reshape(-1)[candidate_indices]
    capacity = int(candidate_indices.numel())
    achieved = min(requested_work, capacity)
    selected = torch.zeros_like(eligible)
    if achieved:
        order = torch.argsort(candidate_scores, descending=True, stable=True)
        chosen = candidate_indices[order[:achieved]]
        selected.reshape(-1)[chosen] = True

    return DirectPairBudgetProjection(
        selected_pairs=selected,
        requested_work=requested_work,
        achieved_work=achieved,
        feasible_capacity=capacity,
        exact_match=achieved == requested_work,
    )


__all__ = [
    "CanonicalPacketQuotient",
    "DirectPairBudgetProjection",
    "canonical_packet_quotient",
    "function_preserving_deeper_split",
    "function_preserving_root_split",
    "project_direct_pair_budget",
    "root_union_plan",
]
