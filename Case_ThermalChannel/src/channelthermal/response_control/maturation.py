"""Deterministic Thermal controlled-maturation exposure schedule.

The schedule separates primary response-family passes from the independent
historical-value replay clock. Four 0.90 sparse updates are followed by one
full-access replay. Sparse updates in an action pass use one requested packet
cut across a shuffled without-replacement pass over all eligible families.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from honf_forward_core.interface_fields.budgeted_frontier import (
    enumerate_frontier_cuts,
    frontier_from_paths,
    frontier_paths,
)

THERMAL_ACTION_PATHS: dict[str, tuple[str, ...]] = {
    "root": ("",),
    "two_packet": ("L", "R"),
    "four_packet": ("LL", "LR", "RL", "RR"),
}


@dataclass(frozen=True)
class MaturationUpdate:
    """The family, capacity, and action assigned to one optimizer update."""

    completed_updates_before: int
    relative_update: int
    relative_sparse_update: int
    family_index: int
    family_id: str
    primary_pass: int
    pass_position: int
    phase: str
    action: str
    requested_cut_paths: tuple[str, ...]
    full_access_replay: bool
    capacity_fraction: float
    query_seed: int


class ThermalMaturationSchedule:
    """Pure update-to-action mapping for the bounded 0.90 Thermal lane."""

    def __init__(
        self,
        family_ids: tuple[str, ...] | list[str],
        *,
        start_update: int = 200,
        seed: int = 7319,
        sparse_before_full: int = 4,
    ) -> None:
        families = tuple(str(value) for value in family_ids)
        if not families or len(families) != len(set(families)):
            raise ValueError("Thermal maturation needs unique eligible response-family IDs.")
        if start_update < 0 or seed < 0 or sparse_before_full <= 0:
            raise ValueError("Maturation start, seed, and sparse replay cadence must be nonnegative/positive.")
        self.family_ids = families
        self.start_update = int(start_update)
        self.seed = int(seed)
        self.sparse_before_full = int(sparse_before_full)

    def _family_order(self, primary_pass: int) -> tuple[int, ...]:
        order = list(range(len(self.family_ids)))
        random.Random(self.seed + primary_pass * 1_000_003).shuffle(order)
        return tuple(order)

    def plan(self, completed_updates_before: int) -> MaturationUpdate:
        """Return the deterministic update plan at an absolute update count."""

        absolute = int(completed_updates_before)
        if absolute < self.start_update:
            raise ValueError("The controlled schedule starts at its declared u200 base.")
        relative = absolute - self.start_update
        cadence = self.sparse_before_full + 1
        cadence_slot = relative % cadence
        cadence_block = relative // cadence
        full_replay = cadence_slot == self.sparse_before_full
        sparse_index = cadence_block * self.sparse_before_full + (
            self.sparse_before_full - 1 if full_replay else cadence_slot
        )
        primary_pass, pass_position = divmod(sparse_index, len(self.family_ids))
        family_index = self._family_order(primary_pass)[pass_position]
        if primary_pass < 2:
            phase = "two_packet_scaffold"
            action = "two_packet_scaffold"
            cut_paths = THERMAL_ACTION_PATHS["two_packet"]
        else:
            phase = "action_family"
            action = ("root", "two_packet", "four_packet")[(primary_pass - 2) % 3]
            cut_paths = THERMAL_ACTION_PATHS[action]
        return MaturationUpdate(
            completed_updates_before=absolute,
            relative_update=relative,
            relative_sparse_update=sparse_index,
            family_index=family_index,
            family_id=self.family_ids[family_index],
            primary_pass=primary_pass,
            pass_position=pass_position,
            phase=("scheduled_full_access_replay" if full_replay else phase),
            action=("full_access_replay" if full_replay else action),
            requested_cut_paths=cut_paths,
            full_access_replay=full_replay,
            capacity_fraction=(1.0 if full_replay else 0.90),
            query_seed=self.seed + relative * 104_729,
        )


def complementary_soft_shadow_variants(
    state_labels: tuple[str, ...] | list[str],
    *,
    relative_sparse_update: int,
    seed: int = 7319,
) -> tuple[str, ...]:
    """Choose a deterministic half of stencil variants, complemented next sparse step.

    The permutation is keyed by the pair of sparse updates, so full-access
    replays do not consume a half. Results are returned in native stencil
    order, independently of the shuffled order used to form the partition.
    """

    labels = tuple(str(label) for label in state_labels)
    if not labels or labels[0] != "baseline":
        raise ValueError("Soft-shadow selection requires baseline as the first state label.")
    variants = labels[1:]
    if not variants or len(variants) % 2 or len(set(labels)) != len(labels):
        raise ValueError("Soft-shadow selection requires unique variant labels in an even-sized panel.")
    if relative_sparse_update < 0 or seed < 0:
        raise ValueError("Soft-shadow sparse index and seed must be nonnegative.")

    shuffled = list(variants)
    pair_index = int(relative_sparse_update) // 2
    random.Random(int(seed) + pair_index * 1_000_003).shuffle(shuffled)
    split = len(shuffled) // 2
    selected = set(
        shuffled[:split]
        if int(relative_sparse_update) % 2 == 0
        else shuffled[split:]
    )
    return tuple(label for label in variants if label in selected)


def summarize_realized_cut_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_states: Sequence[str],
    completed_updates_before_attempt: int,
    attempted_optimizer_step_including_old_branch: int,
    full_access_replay: bool,
) -> dict[str, Any]:
    """Bind the successful optimizer update to its actual hard-plan cuts.

    Route-work rows are written for an optimizer attempt before ``optimizer.step``
    runs. This summary is only emitted by the completed-step callback, so a
    failed or orphan attempt cannot certify a completed update's cut.
    """

    update_before = int(completed_updates_before_attempt)
    attempted_step = int(attempted_optimizer_step_including_old_branch)
    if update_before < 0 or attempted_step < 1:
        raise ValueError("Realized-cut evidence needs a valid update and attempted-step identity.")
    if full_access_replay:
        if records:
            raise ValueError("Full-access replays must not claim sparse frontier records.")
        return {
            "status": "full_access_no_cut",
            "completed_updates_before_attempt": update_before,
            "attempted_optimizer_step_including_old_branch": attempted_step,
            "baseline": None,
            "by_state": {},
            "variant_cut_distribution": [],
        }

    if not records:
        raise ValueError("A sparse update is missing realized hard-plan cut records.")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in records:
        row = dict(source)
        if int(row.get("completed_updates_before_attempt", -1)) != update_before:
            raise ValueError("Realized-cut record belongs to a different completed-update cursor.")
        if int(row.get("attempted_optimizer_step_including_old_branch", -1)) != attempted_step:
            raise ValueError("Realized-cut record belongs to a different optimizer attempt.")
        state = str(row.get("state", ""))
        if not state:
            raise ValueError("Realized-cut record is missing its native state label.")
        frontier = tuple(int(value) for value in row.get("frontier", ()))
        paths = tuple(str(value) for value in row.get("frontier_paths", ()))
        raw_k = int(row.get("raw_frontier_k", -1))
        if not frontier or raw_k != len(frontier) or len(paths) != raw_k:
            raise ValueError("Realized frontier IDs, paths, and raw K disagree.")
        nonredundant = row.get("nonredundant_k")
        if nonredundant is not None:
            nonredundant = int(nonredundant)
            if not 1 <= nonredundant <= raw_k:
                raise ValueError("Realized nonredundant K is outside the realized frontier range.")
        grouped.setdefault(state, []).append(
            {
                "frontier": list(frontier),
                "frontier_paths": list(paths),
                "raw_frontier_k": raw_k,
                "nonredundant_k": nonredundant,
                "nonredundant_k_status": row.get("nonredundant_k_status"),
                "case_index": row.get("case_index"),
            }
        )

    expected = tuple(str(value) for value in expected_states)
    if not expected or expected[0] != "baseline" or len(set(expected)) != len(expected):
        raise ValueError("Expected native states must be unique and start with baseline.")
    if set(grouped) != set(expected):
        raise ValueError(
            "Sparse realized-cut states do not match the expected hard-call order: "
            f"expected {expected}, got {tuple(grouped)}."
        )
    if any(len(grouped[state]) != 1 for state in expected):
        raise ValueError("Each single-stencil native state must yield exactly one hard-plan cut.")
    by_state = {state: grouped[state][0] for state in expected}
    baseline = by_state["baseline"]
    variants = [
        {
            "state": state,
            "frontier_paths": by_state[state]["frontier_paths"],
            "raw_frontier_k": by_state[state]["raw_frontier_k"],
            "nonredundant_k": by_state[state]["nonredundant_k"],
            "nonredundant_k_status": by_state[state]["nonredundant_k_status"],
        }
        for state in expected[1:]
        if state != "historical_value_replay"
    ]
    return {
        "status": "realized_sparse_cut",
        "completed_updates_before_attempt": update_before,
        "attempted_optimizer_step_including_old_branch": attempted_step,
        "baseline": baseline,
        "by_state": by_state,
        "variant_cut_distribution": variants,
    }


def _node_for_path(tree: Any, path: str, *, max_depth: int) -> tuple[int, str]:
    node_index = 0
    resolved = ""
    if len(path) > max_depth or set(path) - {"L", "R"}:
        raise ValueError("A Thermal cut action must use valid bounded L/R paths.")
    for direction in path:
        node = tree.nodes[node_index]
        child = node.left if direction == "L" else node.right
        if child is None:
            return node_index, resolved
        node_index = int(child)
        resolved += direction
    return node_index, resolved


def available_frontier_for_paths(
    tree: Any,
    requested_paths: tuple[str, ...] | list[str],
    *,
    max_depth: int = 3,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Map an abstract action to its available cut when a branch ends early.

    A missing child resolves to the existing terminal ancestor. Duplicate
    ancestors are removed and paths are ordered by the tree's node order before
    validation, so the returned values always describe a complete native cut.
    """

    if not requested_paths:
        raise ValueError("A requested Thermal cut must contain at least one path.")
    resolved_by_node: dict[int, str] = {}
    for path in requested_paths:
        index, resolved = _node_for_path(tree, str(path), max_depth=max_depth)
        resolved_by_node[index] = resolved
    # Node IDs are assigned by the native tree builder, not by left-to-right
    # traversal. On asymmetric early-leaf trees a right parent can have a
    # smaller ID than its left grandchildren (e.g. IDs 3,4,2 for LL,LR,R).
    # The frontier APIs use lexicographic L/R path order, so sort resolved
    # paths by their structural order before mapping them back to node IDs.
    ordered = tuple(sorted(resolved_by_node.values()))
    frontier = frontier_from_paths(tree, ordered, max_depth=max_depth)
    if frontier not in enumerate_frontier_cuts(tree, max_depth=max_depth):
        raise ValueError("The available Thermal paths do not form a complete native cut.")
    if frontier_paths(tree, frontier, max_depth=max_depth) != ordered:
        raise ValueError("Resolved Thermal path order disagrees with the native cut.")
    return frontier, ordered


__all__ = [
    "THERMAL_ACTION_PATHS",
    "MaturationUpdate",
    "ThermalMaturationSchedule",
    "available_frontier_for_paths",
    "complementary_soft_shadow_variants",
    "summarize_realized_cut_records",
]
