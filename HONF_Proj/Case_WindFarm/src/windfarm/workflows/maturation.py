"""Deterministic exposure schedule for controlled WindFarm maturation.

The schedule gives both matched arms the same native rows and QE 0.95 / MM
0.90 sparse capacity. Grouped actions change only in complete without-replacement passes;
the direct arm records the same action blocks while retaining its direct-pair
access semantics. One same-case, fresh-query full-access replay follows each
four sparse updates and does not consume a new row from the primary-data clock.
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


WIND_ACTION_PATHS: dict[str, tuple[str, ...]] = {
    "root": ("",),
    "two_packet": ("L", "R"),
    "four_packet": ("LL", "LR", "RL", "RR"),
}


@dataclass(frozen=True)
class WindMaturationUpdate:
    completed_updates_before: int
    relative_update: int
    relative_sparse_update: int
    row_index: int
    row_id: int
    primary_pass: int
    pass_position: int
    phase: str
    action: str
    requested_cut_paths: tuple[str, ...]
    full_access_replay: bool
    replay_sparse_update: int | None
    capacity_vector: tuple[tuple[str, float], ...]
    query_seed: int


class WindMaturationSchedule:
    """Pure deterministic update-to-row/action mapping for matched G/P arms."""

    def __init__(
        self,
        row_ids: tuple[int, ...] | list[int],
        *,
        start_update: int = 100,
        seed: int = 2112,
        sparse_before_full: int = 4,
    ) -> None:
        rows = tuple(int(value) for value in row_ids)
        if not rows or len(rows) != len(set(rows)):
            raise ValueError("Wind maturation needs unique eligible training row IDs.")
        if start_update < 0 or seed < 0 or sparse_before_full <= 0:
            raise ValueError("Maturation start/seed must be nonnegative and replay cadence positive.")
        self.row_ids = rows
        self.start_update = int(start_update)
        self.seed = int(seed)
        self.sparse_before_full = int(sparse_before_full)

    def _row_order(self, primary_pass: int) -> tuple[int, ...]:
        order = list(range(len(self.row_ids)))
        random.Random(self.seed + int(primary_pass) * 1_000_003).shuffle(order)
        return tuple(order)

    def plan(self, completed_updates_before: int) -> WindMaturationUpdate:
        absolute = int(completed_updates_before)
        if absolute < self.start_update:
            raise ValueError("The controlled schedule starts at its declared u100 base.")
        relative = absolute - self.start_update
        cadence = self.sparse_before_full + 1
        cadence_slot = relative % cadence
        cadence_block = relative // cadence
        full_replay = cadence_slot == self.sparse_before_full
        sparse_index = cadence_block * self.sparse_before_full + (
            self.sparse_before_full - 1 if full_replay else cadence_slot
        )
        primary_pass, pass_position = divmod(sparse_index, len(self.row_ids))
        row_index = self._row_order(primary_pass)[pass_position]
        if primary_pass < 2:
            phase = "two_packet_scaffold"
            action = "two_packet_scaffold"
            paths = WIND_ACTION_PATHS["two_packet"]
        else:
            phase = "action_family"
            action = ("root", "two_packet", "four_packet")[(primary_pass - 2) % 3]
            paths = WIND_ACTION_PATHS[action]
        return WindMaturationUpdate(
            completed_updates_before=absolute,
            relative_update=relative,
            relative_sparse_update=sparse_index,
            row_index=row_index,
            row_id=self.row_ids[row_index],
            primary_pass=primary_pass,
            pass_position=pass_position,
            phase="scheduled_full_access_replay" if full_replay else phase,
            action="full_access_replay" if full_replay else action,
            requested_cut_paths=paths,
            full_access_replay=full_replay,
            replay_sparse_update=sparse_index if full_replay else None,
            capacity_vector=(
                (("QE", 1.0), ("MM", 1.0))
                if full_replay
                else (("QE", 0.95), ("MM", 0.90))
            ),
            query_seed=self.seed + relative * 104_729,
        )


def _resolve_node_path(tree: Any, path: str, *, max_depth: int) -> tuple[int, str]:
    node_index = 0
    resolved = ""
    if len(path) > max_depth or set(path) - {"L", "R"}:
        raise ValueError("A Wind action must use valid bounded L/R paths.")
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
    """Resolve abstract action paths to an available native complete cut."""

    if not requested_paths:
        raise ValueError("A requested Wind action must contain at least one path.")
    resolved_by_node: dict[int, str] = {}
    for path in requested_paths:
        index, resolved = _resolve_node_path(tree, str(path), max_depth=max_depth)
        resolved_by_node[index] = resolved
    # Native node indices are not guaranteed to follow left-to-right tree order.
    # Complete cuts are represented by lexicographically ordered L/R paths.
    ordered = tuple(sorted(resolved_by_node.values()))
    frontier = frontier_from_paths(tree, ordered, max_depth=max_depth)
    if frontier not in enumerate_frontier_cuts(tree, max_depth=max_depth):
        raise ValueError("Resolved Wind paths do not form a complete native cut.")
    if frontier_paths(tree, frontier, max_depth=max_depth) != ordered:
        raise ValueError("Resolved Wind paths disagree with the actual native cut.")
    return frontier, ordered


__all__ = [
    "WIND_ACTION_PATHS",
    "WindMaturationSchedule",
    "WindMaturationUpdate",
    "available_frontier_for_paths",
]
