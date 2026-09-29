"""Bounded receiver cuts and tie-safe canonical pair budgets.

The receiver tree is only an index over fixed physical anchors. A packet's
work is the union of valid physical receiver/source pairs reached through its
active nodes, so overlapping packets never double count an interaction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_interaction_cover import CaseLocalReceiverTree
from .types import EncodedInterfaceCase


@dataclass(frozen=True)
class UniquePairBudgetProjection:
    """A score-threshold projection measured on the declared anchor panel."""

    membership: torch.Tensor  # [N,S] hard boolean support
    threshold: torch.Tensor  # scalar detached threshold; +inf means empty
    budget_fraction: float
    requested_work: float
    achieved_work: float
    full_access_work: float
    selected_unique_pairs: int
    full_unique_pairs: int
    sparse_success: bool
    tie_policy: str = "equal-score groups are retained or removed together"

    @property
    def work_fraction(self) -> float:
        if self.full_access_work <= 0.0:
            return 0.0
        return float(self.achieved_work / self.full_access_work)


@dataclass(frozen=True)
class DirectPairBudgetProjection:
    """Tie-atomic pair support for an independently scored direct control."""

    selected_pairs: torch.Tensor  # [Q,S] boolean
    threshold: torch.Tensor
    budget_fraction: float
    requested_work: float
    achieved_work: float
    full_access_work: float
    selected_unique_pairs: int
    full_unique_pairs: int
    sparse_success: bool


@dataclass(frozen=True)
class DirectPairThresholdWeights:
    """Hard values and a finite-threshold soft score shadow for a frozen P cap."""

    hard_weights: torch.Tensor
    soft_weights: torch.Tensor
    hard_threshold: torch.Tensor
    soft_threshold: torch.Tensor


@dataclass(frozen=True)
class CanonicalPairCatalog:
    """Typed input-only receiver/source universe for one physical route."""

    receiver_coordinates: torch.Tensor
    receiver_weights: torch.Tensor
    receiver_validity: torch.Tensor
    source_validity: torch.Tensor
    pair_validity: torch.Tensor


def canonical_pair_catalog(
    encoded: EncodedInterfaceCase,
    tree: CaseLocalReceiverTree,
    mechanism: str,
    *,
    case_index: int = 0,
) -> CanonicalPairCatalog:
    """Build native preparation or fixed query-anchor work for a typed route."""

    key = str(mechanism).upper()
    if key not in {"MM", "ME", "EM", "QM", "QE"}:
        raise ValueError(f"unsupported interaction mechanism {mechanism!r}")
    if case_index < 0 or case_index >= int(encoded.module_present.shape[0]):
        raise IndexError("case_index is outside the encoded batch")
    module_coords = encoded.module_centers[case_index]
    module_valid = encoded.module_present[case_index] > 0.5
    env_coords = encoded.env_coords[case_index]
    env_weights = encoded.env_weights[case_index]
    env_valid = env_weights > 0.0
    if key in {"MM", "ME"}:
        receiver_coordinates = module_coords
        receiver_weights = torch.ones_like(encoded.module_present[case_index])
        receiver_validity = module_valid
        source_validity = module_valid if key == "MM" else env_valid
    elif key == "EM":
        receiver_coordinates = env_coords
        receiver_weights = env_weights
        receiver_validity = env_valid
        source_validity = module_valid
    else:
        receiver_coordinates = tree.universe.coordinates
        receiver_weights = tree.universe.weights
        receiver_validity = torch.ones_like(tree.universe.weights, dtype=torch.bool)
        source_validity = module_valid if key == "QM" else env_valid
    pair_validity = receiver_validity[:, None] & source_validity[None, :]
    if key == "MM":
        if pair_validity.shape[0] != pair_validity.shape[1]:
            raise ValueError("MM canonical receiver and source axes must share native module order")
        pair_validity = pair_validity & ~torch.eye(
            pair_validity.shape[0], device=pair_validity.device, dtype=torch.bool
        )
    return CanonicalPairCatalog(
        receiver_coordinates=receiver_coordinates,
        receiver_weights=receiver_weights,
        receiver_validity=receiver_validity,
        source_validity=source_validity,
        pair_validity=pair_validity,
    )


@dataclass(frozen=True)
class FrontierUtilityPrediction:
    """Per-cut normalized distortion and canonical work-fraction predictions.

    ``role_distortion`` is dimensionless: the positive field-error excess over
    the same student's full-access prediction divided by a declared full-access
    or train-only per-role scale. Improvements over full access have zero
    distortion. ``predicted_work`` is canonical weighted unique-pair work
    divided by full-access work, in [0,1].
    """

    cuts: tuple[tuple[int, ...], ...]
    role_distortion: torch.Tensor  # [C,R], dimensionless nonnegative relative distortion
    predicted_work_fraction: torch.Tensor  # [C], predicted weighted unique-pair fraction

    @property
    def predicted_work(self) -> torch.Tensor:
        """Compatibility alias; values are dimensionless work fractions."""

        return self.predicted_work_fraction

    @property
    def utility(self) -> torch.Tensor:
        """Compatibility name: utility is the predicted work to minimize."""

        return self.predicted_work


@dataclass(frozen=True)
class FrontierSelection:
    """Deployment selection plus a separately labelled research fallback."""

    selected_frontier: tuple[int, ...] | None
    selected_index: int | None
    unsupported_at_budget: bool
    least_risk_frontier: tuple[int, ...]
    least_risk_index: int


def enumerate_frontier_cuts(
    tree: CaseLocalReceiverTree,
    *,
    max_depth: int = 3,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate complete cuts through at most ``max_depth`` tree edges.

    For a full depth-three binary tree this returns 26 cuts over 15 nodes.
    A real leaf ends its branch early; an internal node at the depth ceiling
    is treated as a leaf of this bounded candidate family.
    """

    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if not tree.nodes:
        raise ValueError("receiver tree must contain a root")

    def visit(index: int, depth: int) -> tuple[tuple[int, ...], ...]:
        if index < 0 or index >= len(tree.nodes):
            raise ValueError("receiver tree contains an out-of-range child")
        node = tree.nodes[index]
        if (node.left is None) != (node.right is None):
            raise ValueError("receiver tree nodes must have zero or two children")
        if node.left is None or depth >= max_depth:
            return ((index,),)
        left_cuts = visit(node.left, depth + 1)
        right_cuts = visit(node.right, depth + 1)
        refined = tuple(left + right for left, right in product(left_cuts, right_cuts))
        return ((index,), *refined)

    return visit(0, 0)


def frontier_paths(
    tree: CaseLocalReceiverTree,
    frontier: Sequence[int],
    *,
    max_depth: int = 3,
) -> tuple[str, ...]:
    """Identify a complete cut by root-to-node paths, independent of node IDs.

    Native tree node numbers may change when a trial moves a module. The
    abstract left/right refinement program is the comparable candidate.
    """

    canonical = tuple(int(index) for index in frontier)
    if canonical not in enumerate_frontier_cuts(tree, max_depth=max_depth):
        raise ValueError("frontier must be one complete bounded cut of its source tree")
    paths: dict[int, str] = {}

    def visit(index: int, path: str) -> None:
        paths[index] = path
        if len(path) >= max_depth:
            return
        node = tree.nodes[index]
        if node.left is not None:
            assert node.right is not None
            visit(node.left, path + "L")
            visit(node.right, path + "R")

    visit(0, "")
    return tuple(paths[index] for index in canonical)


def frontier_from_paths(
    tree: CaseLocalReceiverTree,
    paths: Sequence[str],
    *,
    max_depth: int = 3,
) -> tuple[int, ...]:
    """Apply the same bounded split pattern to another current-design tree.

    A missing branch is an unavailable candidate, never a request to change K
    or silently substitute full access. The caller must record that outcome.
    """

    mapped: list[int] = []
    for path in paths:
        if not isinstance(path, str) or len(path) > max_depth or set(path) - {"L", "R"}:
            raise ValueError("frontier path must use at most max_depth L/R steps")
        index = 0
        for direction in path:
            node = tree.nodes[index]
            child = node.left if direction == "L" else node.right
            if child is None:
                raise ValueError("frontier path is unavailable in the current tree")
            index = child
        mapped.append(index)
    canonical = tuple(mapped)
    if canonical not in enumerate_frontier_cuts(tree, max_depth=max_depth):
        raise ValueError("mapped frontier paths do not form a complete bounded cut")
    return canonical


def split_gates_for_frontier(
    tree: CaseLocalReceiverTree,
    frontier: Sequence[int],
    *,
    max_depth: int = 3,
) -> torch.Tensor:
    """Return exact binary split gates that realize one enumerated cut."""

    canonical = tuple(int(index) for index in frontier)
    if not canonical or len(set(canonical)) != len(canonical):
        raise ValueError("frontier must contain unique candidate node indices")
    if canonical not in enumerate_frontier_cuts(tree, max_depth=max_depth):
        raise ValueError("frontier is not a complete cut in the bounded candidate tree")
    cut = set(canonical)
    gates = tree.universe.coordinates.new_zeros((len(tree.nodes),))

    def open_to_cut(index: int, depth: int) -> None:
        if index in cut:
            return
        if depth >= max_depth:
            raise ValueError("frontier omitted a node at the bounded depth ceiling")
        node = tree.nodes[index]
        if node.left is None or node.right is None:
            raise ValueError("frontier does not cover a terminal receiver branch")
        gates[index] = 1.0
        open_to_cut(node.left, depth + 1)
        open_to_cut(node.right, depth + 1)

    open_to_cut(0, 0)
    return gates


def _pair_support(
    access: torch.Tensor,
    membership: torch.Tensor,
    eligible_pairs: torch.Tensor,
) -> torch.Tensor:
    support = torch.zeros_like(eligible_pairs)
    for node_index in range(int(access.shape[1])):
        receiver_rows = access[:, node_index] > 0.0
        if bool(receiver_rows.any()) and bool((membership[node_index] > 0.0).any()):
            support |= receiver_rows[:, None] & (membership[node_index] > 0.0)[None, :]
    return support & eligible_pairs


def _work(support: torch.Tensor, receiver_weights: torch.Tensor) -> float:
    # Work thresholds may sit near a fractional cap over very large QE
    # panels. Accumulate physical measures in float64 regardless of model
    # dtype; retain the separate integer pair count for exact accounting.
    return float((support.to(torch.float64) * receiver_weights[:, None].to(torch.float64)).sum().detach())


def _threshold_projection(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    row_weights: torch.Tensor,
    budget_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int, int]:
    """Find the lowest feasible exact-score threshold by monotone search."""

    if scores.ndim != 2 or min(scores.shape) < 1:
        raise ValueError("scores must have shape [rows,columns] with positive axes")
    if eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("eligible mask must be boolean and match scores")
    if eligible.device != scores.device or row_weights.shape != (scores.shape[0],):
        raise ValueError("eligibility and row weights must align with score rows")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    if not bool(torch.isfinite(row_weights).all()) or bool((row_weights < 0).any()):
        raise ValueError("row weights must be finite and nonnegative")
    if not 0.0 <= float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must be in [0,1]")

    detached = scores.detach()
    full_pairs = eligible
    full_work = _work(full_pairs, row_weights)
    requested_work = float(budget_fraction) * full_work
    full_unique = int(full_pairs.sum())
    if float(budget_fraction) == 1.0:
        return (
            full_pairs.clone(),
            detached.new_tensor(float("-inf")),
            requested_work,
            full_work,
            full_unique,
            full_unique,
        )
    if float(budget_fraction) == 0.0:
        empty = torch.zeros_like(eligible)
        return (
            empty,
            detached.new_tensor(float("inf")),
            0.0,
            0.0,
            0,
            full_unique,
        )
    candidate_values = detached[eligible]
    if candidate_values.numel() == 0:
        empty = torch.zeros_like(eligible)
        return (
            empty,
            detached.new_tensor(float("inf")),
            requested_work,
            0.0,
            0,
            0,
        )

    thresholds = torch.unique(candidate_values, sorted=True)
    # As the threshold rises, support/work can only decrease. Find the first
    # threshold that fits; >= treats every exact-score tie as one group.
    low, high = 0, int(thresholds.numel())
    while low < high:
        middle = (low + high) // 2
        mask = eligible & (detached >= thresholds[middle])
        achieved = _work(mask, row_weights)
        if achieved <= requested_work + 1.0e-12 * max(1.0, full_work):
            high = middle
        else:
            low = middle + 1
    if low == int(thresholds.numel()):
        selected = torch.zeros_like(eligible)
        threshold = detached.new_tensor(float("inf"))
    else:
        threshold = thresholds[low]
        selected = eligible & (detached >= threshold)
    achieved_work = _work(selected, row_weights)
    return (
        selected,
        threshold.detach(),
        requested_work,
        achieved_work,
        int(selected.sum()),
        full_unique,
    )


def project_unique_pair_budget(
    scores: torch.Tensor,
    tree: CaseLocalReceiverTree,
    split_gates: torch.Tensor,
    *,
    budget_fraction: float,
    source_validity: torch.Tensor | None = None,
    receiver_validity: torch.Tensor | None = None,
    pair_validity: torch.Tensor | None = None,
    receiver_weights: torch.Tensor | None = None,
    receiver_coordinates: torch.Tensor | None = None,
) -> UniquePairBudgetProjection:
    """Project node/source scores under weighted canonical physical-pair work.

    A receiver/source pair is counted once if any positively reached node in
    the overlapping cover grants it. Invalid sources, receivers and explicit
    pair exclusions (for example MM self pairs) do not consume the budget.
    Projection decisions use detached scores and remove exact-score ties as a
    whole group; gradients are supplied separately by a soft sigmoid shadow.
    """

    if scores.ndim != 2 or scores.shape[0] != len(tree.nodes):
        raise ValueError("node/source scores must have shape [tree nodes,sources]")
    if not 0.0 <= float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must be in [0,1]")
    if scores.device != split_gates.device or scores.device != tree.universe.coordinates.device:
        raise ValueError("scores, split gates and receiver tree must share a device")
    coordinates = tree.universe.coordinates if receiver_coordinates is None else receiver_coordinates
    if coordinates.ndim != 2 or coordinates.shape[1] != tree.universe.coordinates.shape[1]:
        raise ValueError("receiver_coordinates must use the plan's physical spatial frame")
    if coordinates.device != scores.device:
        raise ValueError("receiver coordinates and scores must share one device")
    detached_scores = scores.detach()
    receiver_count, source_count = int(coordinates.shape[0]), int(scores.shape[1])
    if source_validity is None:
        source_mask = torch.ones((source_count,), dtype=torch.bool, device=scores.device)
    else:
        if source_validity.shape != (source_count,):
            raise ValueError("source_validity must align with score columns")
        source_mask = source_validity.to(device=scores.device, dtype=torch.bool)
    if receiver_validity is None:
        receiver_mask = torch.ones((receiver_count,), dtype=torch.bool, device=scores.device)
    else:
        if receiver_validity.shape != (receiver_count,):
            raise ValueError("receiver_validity must align with physical receiver anchors")
        receiver_mask = receiver_validity.to(device=scores.device, dtype=torch.bool)
    if pair_validity is None:
        pair_mask = torch.ones((receiver_count, source_count), dtype=torch.bool, device=scores.device)
    else:
        if pair_validity.shape != (receiver_count, source_count):
            raise ValueError("pair_validity must have shape [anchors,sources]")
        pair_mask = pair_validity.to(device=scores.device, dtype=torch.bool)
    if receiver_weights is None:
        weights = (
            tree.universe.weights.to(device=scores.device, dtype=scores.dtype)
            if receiver_coordinates is None
            else scores.new_ones((receiver_count,))
        )
    else:
        if receiver_weights.shape != (receiver_count,):
            raise ValueError("receiver_weights must align with physical receiver anchors")
        weights = receiver_weights.to(device=scores.device, dtype=scores.dtype)
    eligible = receiver_mask[:, None] & source_mask[None, :] & pair_mask
    access = tree.access(coordinates.detach(), split_gates.detach()).detach()
    active_nodes = (access > 0.0).any(dim=0)
    eligible_nodes = active_nodes[:, None] & source_mask[None, :]
    full_work = _work(eligible, weights)
    requested_work = float(budget_fraction) * full_work

    if float(budget_fraction) == 1.0:
        membership = source_mask[None, :].expand_as(scores).clone()
        support = _pair_support(access, membership, eligible)
        return UniquePairBudgetProjection(
            membership=membership,
            threshold=detached_scores.new_tensor(float("-inf")),
            budget_fraction=1.0,
            requested_work=requested_work,
            achieved_work=_work(support, weights),
            full_access_work=full_work,
            selected_unique_pairs=int(support.sum()),
            full_unique_pairs=int(eligible.sum()),
            sparse_success=False,
        )
    if float(budget_fraction) == 0.0:
        membership = torch.zeros_like(scores, dtype=torch.bool)
        return UniquePairBudgetProjection(
            membership=membership,
            threshold=detached_scores.new_tensor(float("inf")),
            budget_fraction=0.0,
            requested_work=0.0,
            achieved_work=0.0,
            full_access_work=full_work,
            selected_unique_pairs=0,
            full_unique_pairs=int(eligible.sum()),
            sparse_success=False,
        )

    # The threshold is selected using canonical union work, not the number of
    # node/source assignments. Restrict candidate thresholds to reached nodes
    # so inaccessible deep logits cannot affect a live plan.
    candidate_values = detached_scores[eligible_nodes]
    if candidate_values.numel() == 0:
        membership = torch.zeros_like(scores, dtype=torch.bool)
        return UniquePairBudgetProjection(
            membership, detached_scores.new_tensor(float("inf")), float(budget_fraction),
            float(budget_fraction) * full_work, 0.0, full_work, 0, int(eligible.sum()),
            False,
        )

    # Repeatedly evaluating unique-pair work across every distinct score is
    # quadratic in the number of candidates. Binary-search the monotone
    # threshold over unique scores while preserving whole ties.
    thresholds = torch.unique(candidate_values, sorted=True)
    low, high = 0, int(thresholds.numel())
    best_membership = torch.zeros_like(scores, dtype=torch.bool)
    best_threshold = detached_scores.new_tensor(float("inf"))
    best_work = 0.0
    best_count = 0
    tolerance = 1.0e-12 * max(1.0, full_work)
    while low < high:
        middle = (low + high) // 2
        candidate = (detached_scores >= thresholds[middle]) & source_mask[None, :]
        support = _pair_support(access, candidate, eligible)
        work = _work(support, weights)
        if work <= requested_work + tolerance:
            best_membership = candidate
            best_threshold = thresholds[middle]
            best_work = work
            best_count = int(support.sum())
            high = middle
        else:
            low = middle + 1
    if low < int(thresholds.numel()):
        best_threshold = thresholds[low]
        best_membership = (detached_scores >= best_threshold) & source_mask[None, :]
        best_support = _pair_support(access, best_membership, eligible)
        best_work = _work(best_support, weights)
        best_count = int(best_support.sum())
    sparse_success = best_work > 0.0 and best_count > 0 and best_work < full_work - tolerance
    return UniquePairBudgetProjection(
        membership=best_membership,
        threshold=best_threshold.detach(),
        budget_fraction=float(budget_fraction),
        requested_work=requested_work,
        achieved_work=best_work,
        full_access_work=full_work,
        selected_unique_pairs=best_count,
        full_unique_pairs=int(eligible.sum()),
        sparse_success=sparse_success,
    )


def project_direct_pair_budget_by_fraction(
    scores: torch.Tensor,
    *,
    budget_fraction: float,
    eligible_pairs: torch.Tensor | None = None,
    receiver_weights: torch.Tensor | None = None,
) -> DirectPairBudgetProjection:
    """Tie-safe direct-pair control projection under the same work measure."""

    if scores.ndim != 2 or min(scores.shape) < 1:
        raise ValueError("direct scores must have shape [receivers,sources]")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("direct scores must be finite")
    if not 0.0 <= float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must be in [0,1]")
    if eligible_pairs is None:
        eligible = torch.ones_like(scores, dtype=torch.bool)
    else:
        if eligible_pairs.shape != scores.shape or eligible_pairs.dtype != torch.bool:
            raise ValueError("eligible_pairs must be boolean and match direct scores")
        if eligible_pairs.device != scores.device:
            raise ValueError("eligible_pairs and direct scores must share a device")
        eligible = eligible_pairs
    if receiver_weights is None:
        weights = scores.new_ones((scores.shape[0],))
    else:
        if receiver_weights.shape != (scores.shape[0],):
            raise ValueError("receiver_weights must align with direct score rows")
        weights = receiver_weights.to(device=scores.device, dtype=scores.dtype)
    selected, threshold, requested, achieved, count, full_count = _threshold_projection(
        scores, eligible, weights, float(budget_fraction)
    )
    full_work = _work(eligible, weights)
    tolerance = 1.0e-12 * max(1.0, full_work)
    return DirectPairBudgetProjection(
        selected_pairs=selected,
        threshold=threshold,
        budget_fraction=float(budget_fraction),
        requested_work=requested,
        achieved_work=achieved,
        full_access_work=full_work,
        selected_unique_pairs=count,
        full_unique_pairs=full_count,
        sparse_success=achieved > 0.0 and count > 0 and achieved < full_work - tolerance,
    )


def direct_pair_weights_at_threshold(
    scores: torch.Tensor,
    projection: DirectPairBudgetProjection,
    *,
    eligible_pairs: torch.Tensor,
    temperature: float = 1.0,
) -> DirectPairThresholdWeights:
    """Apply a canonical threshold to current native receiver/source rows.

    For empty or full tied projections, retain the +/- infinity structural
    sentinel in hard values and use a finite detached soft threshold so the
    scorer can still learn to restore support.
    """

    if scores.ndim != 2 or eligible_pairs.shape != scores.shape or eligible_pairs.dtype != torch.bool:
        raise ValueError("scores and direct eligibility must share [receivers,sources]")
    if eligible_pairs.device != scores.device:
        raise ValueError("direct eligibility and scores must share a device")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("direct scores must be finite")
    if temperature <= 0.0:
        raise ValueError("soft threshold temperature must be positive")
    threshold = projection.threshold.to(device=scores.device, dtype=scores.dtype).detach()
    if bool(torch.isposinf(threshold)):
        candidate = scores.detach()[eligible_pairs]
        soft_threshold = (
            candidate.max() + float(temperature) if candidate.numel() else scores.new_tensor(1.0)
        )
    elif bool(torch.isneginf(threshold)):
        candidate = scores.detach()[eligible_pairs]
        soft_threshold = (
            candidate.min() - float(temperature) if candidate.numel() else scores.new_tensor(-1.0)
        )
    else:
        soft_threshold = threshold
    hard = (scores.detach() >= threshold) & eligible_pairs
    soft = torch.sigmoid((scores - soft_threshold) / float(temperature)) * eligible_pairs.to(scores.dtype)
    return DirectPairThresholdWeights(
        hard_weights=hard.to(scores.dtype),
        soft_weights=soft,
        hard_threshold=threshold,
        soft_threshold=soft_threshold.detach(),
    )


class BudgetConditionedDirectPairScorer(nn.Module):
    """Independent typed receiver/source scorer for the direct-pair control.

    For the matched P control, callers should pass detached features from the
    physical model/encoder. The scorer parameters remain trainable, while the
    routing loss cannot update the physical encoder; hard native values still
    carry the intended model and design gradients.
    """

    def __init__(
        self,
        *,
        receiver_feature_dim: int,
        source_feature_dim: int,
        hidden_dim: int = 96,
        source_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        if min(receiver_feature_dim, source_feature_dim, source_chunk_size) < 1 or hidden_dim < 2:
            raise ValueError("direct scorer dimensions and chunk size must be positive")
        self.receiver_feature_dim = int(receiver_feature_dim)
        self.source_feature_dim = int(source_feature_dim)
        self.source_chunk_size = int(source_chunk_size)
        input_dim = self.receiver_feature_dim + self.source_feature_dim + 6
        mechanisms = ("MM", "ME", "EM", "QM", "QE")
        self.scorers = nn.ModuleDict({
            mechanism: nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for mechanism in mechanisms
        })

    @staticmethod
    def _spatial_summary(values: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            values.mean(dim=-1),
            values.abs().mean(dim=-1),
            values.square().mean(dim=-1).sqrt(),
            values.amax(dim=-1),
            values.amin(dim=-1),
        ), dim=-1)

    def forward(
        self,
        receiver_features: torch.Tensor,
        source_features: torch.Tensor,
        receiver_coordinates: torch.Tensor,
        source_coordinates: torch.Tensor,
        *,
        mechanism: str,
        budget_fraction: float | torch.Tensor,
    ) -> torch.Tensor:
        key = str(mechanism).upper()
        if key not in self.scorers:
            raise ValueError(f"unsupported direct-pair mechanism {mechanism!r}")
        if receiver_features.ndim != 2 or receiver_features.shape[1] != self.receiver_feature_dim:
            raise ValueError("receiver features do not match the configured width")
        if source_features.ndim != 2 or source_features.shape[1] != self.source_feature_dim:
            raise ValueError("source features do not match the configured width")
        if receiver_coordinates.ndim != 2 or source_coordinates.ndim != 2:
            raise ValueError("physical coordinates must have shape [count,dimension]")
        if receiver_coordinates.shape[1] != source_coordinates.shape[1]:
            raise ValueError("receiver and source coordinates must share a spatial dimension")
        if receiver_coordinates.shape[0] != receiver_features.shape[0] or source_coordinates.shape[0] != source_features.shape[0]:
            raise ValueError("coordinates and feature rows must align")
        reference = receiver_features
        budget = torch.as_tensor(budget_fraction, device=reference.device, dtype=reference.dtype)
        if budget.numel() != 1 or not bool(torch.isfinite(budget).all()) or bool(((budget < 0) | (budget > 1)).any()):
            raise ValueError("budget_fraction must be one finite value in [0,1]")
        source_features = source_features.to(device=reference.device, dtype=reference.dtype)
        receiver_coordinates = receiver_coordinates.to(device=reference.device, dtype=reference.dtype)
        source_coordinates = source_coordinates.to(device=reference.device, dtype=reference.dtype)
        chunks: list[torch.Tensor] = []
        for start in range(0, int(source_features.shape[0]), self.source_chunk_size):
            stop = min(int(source_features.shape[0]), start + self.source_chunk_size)
            relative = receiver_coordinates[:, None, :] - source_coordinates[None, start:stop, :]
            receiver_block = receiver_features[:, None, :].expand(-1, stop - start, -1)
            source_block = source_features[None, start:stop, :].expand(int(receiver_features.shape[0]), -1, -1)
            budget_block = budget.reshape(1, 1, 1).expand(int(receiver_features.shape[0]), stop - start, 1)
            pair_features = torch.cat((
                receiver_block,
                source_block,
                self._spatial_summary(relative),
                budget_block,
            ), dim=-1)
            chunks.append(self.scorers[key](pair_features).squeeze(-1))
        return torch.cat(chunks, dim=1)


class FrontierUtilityHead(nn.Module):
    """Input-only per-cut distortion/utility predictor for Stage C."""

    def __init__(self, *, hidden_dim: int, budget_dim: int, role_count: int) -> None:
        super().__init__()
        if min(hidden_dim, budget_dim, role_count) < 1:
            raise ValueError("frontier utility dimensions must be positive")
        self.role_count = int(role_count)
        self.budget_dim = int(budget_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + budget_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, role_count + 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        cuts: Sequence[Sequence[int]],
        budget_vector: torch.Tensor,
    ) -> FrontierUtilityPrediction:
        if node_embeddings.ndim != 2 or node_embeddings.shape[0] < 1:
            raise ValueError("node embeddings must have shape [N,H]")
        if budget_vector.shape != (self.budget_dim,):
            raise ValueError("budget vector does not match the utility head width")
        canonical_cuts = tuple(tuple(int(node) for node in cut) for cut in cuts)
        if not canonical_cuts:
            raise ValueError("at least one frontier cut is required")
        pooled = []
        for cut in canonical_cuts:
            if not cut or min(cut) < 0 or max(cut) >= int(node_embeddings.shape[0]):
                raise ValueError("frontier node index is outside the embedding table")
            pooled.append(node_embeddings[list(cut)].mean(dim=0))
        cut_embeddings = torch.stack(pooled)
        budgets = budget_vector.to(device=node_embeddings.device, dtype=node_embeddings.dtype)
        budget_rows = budgets[None, :].expand(len(canonical_cuts), -1)
        raw = self.network(torch.cat((cut_embeddings, budget_rows), dim=-1))
        return FrontierUtilityPrediction(
            cuts=canonical_cuts,
            role_distortion=F.softplus(raw[:, : self.role_count]),
            predicted_work_fraction=torch.sigmoid(raw[:, self.role_count]),
        )


def normalized_frontier_distortion_targets(
    candidate_role_errors: torch.Tensor,
    full_access_role_errors: torch.Tensor,
    *,
    role_scales: torch.Tensor | None = None,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Normalize Stage-B errors to dimensionless per-role distortion labels.

    Inputs are ``[cuts,roles]`` and ``[roles]``. By default each role is
    normalized by its matched same-student full-access error magnitude. For
    nearly exact roles, pass positive scales estimated on the training split
    only (such as training-fold target RMS).
    """

    if candidate_role_errors.ndim != 2 or full_access_role_errors.shape != candidate_role_errors.shape[1:]:
        raise ValueError("candidate and full-access role errors must align as [cuts,roles] and [roles]")
    if not bool(torch.isfinite(candidate_role_errors).all()) or not bool(torch.isfinite(full_access_role_errors).all()):
        raise ValueError("frontier target errors must be finite")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if role_scales is None:
        scales = full_access_role_errors.detach().abs().clamp_min(float(epsilon))
    else:
        if role_scales.shape != full_access_role_errors.shape:
            raise ValueError("role scales must provide one value per physical error role")
        scales = role_scales.to(
            device=candidate_role_errors.device, dtype=candidate_role_errors.dtype
        ).detach()
        if not bool(torch.isfinite(scales).all()) or bool((scales <= 0.0).any()):
            raise ValueError("role scales must be finite and positive")
    baseline = full_access_role_errors.to(
        device=candidate_role_errors.device, dtype=candidate_role_errors.dtype
    )
    return (candidate_role_errors - baseline[None, :]).clamp_min(0.0) / scales[None, :]


def normalized_frontier_work_targets(
    canonical_work: torch.Tensor,
    full_access_work: torch.Tensor | float,
    *,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """Return dimensionless weighted unique-pair work fractions per cut."""

    if canonical_work.ndim != 1:
        raise ValueError("canonical work must have one value per candidate cut")
    denominator = torch.as_tensor(
        full_access_work, device=canonical_work.device, dtype=torch.float64
    )
    if denominator.numel() != 1 or not bool(torch.isfinite(denominator).all()) or bool((denominator < 0.0).any()):
        raise ValueError("full-access canonical work must be one finite nonnegative value")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    numerator = canonical_work.to(torch.float64)
    if not bool(torch.isfinite(numerator).all()) or bool((numerator < 0.0).any()):
        raise ValueError("canonical work must be finite and nonnegative")
    if float(denominator) <= 0.0:
        return torch.zeros_like(canonical_work)
    return (numerator / denominator.clamp_min(float(epsilon))).to(canonical_work.dtype).clamp(0.0, 1.0)


def select_frontier_by_predictions(
    prediction: FrontierUtilityPrediction,
    *,
    role_tolerance: torch.Tensor,
    packet_counts: torch.Tensor | None = None,
) -> FrontierSelection:
    """Choose lowest predicted work among adequate cuts and expose least risk.

    ``packet_counts`` should be the measured nonredundant packet count for
    each hard candidate plan. If it is omitted, frontier size is used as an
    explicitly weaker deterministic tie-break.
    """

    if role_tolerance.shape != (prediction.role_distortion.shape[1],):
        raise ValueError("role tolerance must have one value per predicted role")
    tolerance = role_tolerance.to(
        device=prediction.role_distortion.device,
        dtype=prediction.role_distortion.dtype,
    )
    if not bool(torch.isfinite(tolerance).all()) or bool((tolerance < 0).any()):
        raise ValueError("role tolerances must be finite and nonnegative")
    adequate = (prediction.role_distortion <= tolerance[None, :]).all(dim=1)
    if packet_counts is None:
        packet_order = torch.as_tensor(
            [len(cut) for cut in prediction.cuts],
            device=prediction.predicted_work.device,
            dtype=torch.long,
        )
    else:
        if packet_counts.shape != (len(prediction.cuts),):
            raise ValueError("packet_counts must have one nonredundant count per cut")
        packet_order = packet_counts.to(device=prediction.predicted_work.device, dtype=torch.long)
        if bool((packet_order < 0).any()):
            raise ValueError("packet counts must be nonnegative")
    # Risk is the worst tolerance-normalized role error. Least risk remains
    # available for a budget-feasible research diagnostic when deployment is
    # unsupported; that candidate is never silently marked adequate.
    normalized = prediction.role_distortion / tolerance.clamp_min(
        torch.finfo(prediction.role_distortion.dtype).eps
    )[None, :]
    least_risk_index = int(normalized.amax(dim=1).argmin())
    if not bool(adequate.any()):
        return FrontierSelection(
            selected_frontier=None,
            selected_index=None,
            unsupported_at_budget=True,
            least_risk_frontier=prediction.cuts[least_risk_index],
            least_risk_index=least_risk_index,
        )
    candidate_indices = torch.nonzero(adequate, as_tuple=False).squeeze(1)
    # Lexicographic ordering is explicit: predicted physical work first,
    # then measured nonredundant packets, then canonical candidate order.
    selected = min(
        (int(index) for index in candidate_indices.tolist()),
        key=lambda index: (
            float(prediction.predicted_work[index].detach()),
            int(packet_order[index]),
            index,
        ),
    )
    return FrontierSelection(
        selected_frontier=prediction.cuts[selected],
        selected_index=selected,
        unsupported_at_budget=False,
        least_risk_frontier=prediction.cuts[least_risk_index],
        least_risk_index=least_risk_index,
    )


__all__ = [
    "BudgetConditionedDirectPairScorer",
    "CanonicalPairCatalog",
    "DirectPairBudgetProjection",
    "DirectPairThresholdWeights",
    "FrontierSelection",
    "FrontierUtilityHead",
    "FrontierUtilityPrediction",
    "UniquePairBudgetProjection",
    "canonical_pair_catalog",
    "direct_pair_weights_at_threshold",
    "enumerate_frontier_cuts",
    "frontier_from_paths",
    "frontier_paths",
    "normalized_frontier_distortion_targets",
    "normalized_frontier_work_targets",
    "project_direct_pair_budget_by_fraction",
    "project_unique_pair_budget",
    "select_frontier_by_predictions",
    "split_gates_for_frontier",
]
