"""Deterministic Thermal controlled-maturation exposure schedule.

The schedule separates primary response-family passes from the independent
historical-value replay clock. Four 0.90 sparse updates are followed by one
full-access replay. Sparse updates in an action pass use one requested packet
cut across a shuffled without-replacement pass over all eligible families.
"""

from __future__ import annotations

import random
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
    "MaturationUpdate",
    "THERMAL_ACTION_PATHS",
    "ThermalMaturationSchedule",
    "available_frontier_for_paths",
]
