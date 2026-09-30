"""Deterministic exposure schedule for controlled WindFarm maturation.

Both matched arms share native rows, action exposure, and a 4:1 sparse/full
replay cadence. The base recipe uses QE 0.95 / MM 0.90; the bounded remedy
epoch after u1850 uses QE 1.00 / MM 0.90 and repeats two complete shuffled
passes of each action family. Full replays reuse the previous sparse row with
a fresh query seed and do not consume a sparse row from either pass clock.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
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

BASE_RECIPE_STAGE = "qe095_mm090_v1"
QE_FULL_RECIPE_STAGE = "qe_full_mm090_v1"
WIND_SPARSE_RECIPE_CAPACITIES: dict[str, tuple[tuple[str, float], ...]] = {
    BASE_RECIPE_STAGE: (("QE", 0.95), ("MM", 0.90)),
    QE_FULL_RECIPE_STAGE: (("QE", 1.0), ("MM", 0.90)),
}
REMEDY_START_AFTER_UPDATE = 1850
REMEDY_ACTION_ORDER = ("four_packet", "root", "two_packet", "four_packet", "root", "two_packet")
REMEDY_PASSES_PER_ACTION = 2
REMEDY_RECIPE_AMENDMENT_ID = "wind_qe_full_mm090_v1"
BASE_LANE_GPU_BUDGET_SECONDS = 14 * 60 * 60
MAX_LANE_GPU_BUDGET_SECONDS = int(BASE_LANE_GPU_BUDGET_SECONDS * 1.2)


def sparse_capacity_for_recipe(recipe_stage: str) -> tuple[tuple[str, float], ...]:
    """Return a recipe's sparse route capacities in the stable QE/MM order."""

    try:
        return WIND_SPARSE_RECIPE_CAPACITIES[str(recipe_stage)]
    except KeyError as exc:
        raise ValueError(f"Unknown Wind sparse recipe stage: {recipe_stage!r}") from exc


def validated_lane_gpu_budget_seconds(manifest: Mapping[str, Any]) -> float:
    """Read the active Wind allocation, requiring a continuous record to grow it."""

    base = float(BASE_LANE_GPU_BUDGET_SECONDS)
    maximum = float(MAX_LANE_GPU_BUDGET_SECONDS)
    if (
        float(manifest.get("base_lane_gpu_budget_seconds", base)) != base
        or float(manifest.get("maximum_reallocated_lane_gpu_budget_seconds", maximum)) != maximum
    ):
        raise ValueError("Run2112 base or maximum Wind lane allocation metadata changed")
    active = float(manifest.get("lane_gpu_budget_seconds", base))
    if not math.isfinite(active) or active < base or active > maximum:
        raise ValueError(
            f"Run2112 Wind allocation must be between {base:g} and {maximum:g} seconds"
        )
    history = manifest.get("resource_reallocation_history", [])
    if not isinstance(history, list):
        raise TypeError("Run2112 resource_reallocation_history must be a list")
    expected_old = base
    for index, record in enumerate(history):
        if not isinstance(record, Mapping):
            raise TypeError(f"Run2112 resource reallocation record {index} must be a mapping")
        try:
            old = float(record["old_lane_gpu_budget_seconds"])
            new = float(record["new_lane_gpu_budget_seconds"])
            donor_seconds = float(record["donor_allocation_seconds"])
            pilot_forecast = float(record["measured_pilot_forecast_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Run2112 resource reallocation record {index} is incomplete") from exc
        if (
            not math.isfinite(old)
            or not math.isfinite(new)
            or not math.isfinite(donor_seconds)
            or not math.isfinite(pilot_forecast)
            or old != expected_old
            or new <= old
            or new > maximum
            or abs((new - old) - donor_seconds) > 1e-6
            or pilot_forecast <= 0.0
            or not str(record.get("donor_allocation", "")).strip()
            or not str(record.get("evidence_path", "")).strip()
            or not str(record.get("recorded_at_utc", "")).strip()
        ):
            raise ValueError(f"Run2112 resource reallocation record {index} is incomplete or discontinuous")
        expected_old = new
    if expected_old != active:
        raise ValueError(
            "Run2112 lane cap differs from the end of its append-only resource allocation history"
        )
    if active > base and not history:
        raise ValueError("Run2112 lane cap increase has no resource reallocation record")
    return active


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
    recipe_stage: str = BASE_RECIPE_STAGE
    sparse_capacity_vector: tuple[tuple[str, float], ...] = WIND_SPARSE_RECIPE_CAPACITIES[
        BASE_RECIPE_STAGE
    ]
    recipe_pass: int | None = None
    recipe_pass_position: int | None = None
    recipe_sparse_update: int | None = None


class WindMaturationSchedule:
    """Pure deterministic update-to-row/action mapping for matched G/P arms."""

    def __init__(
        self,
        row_ids: tuple[int, ...] | list[int],
        *,
        start_update: int = 100,
        seed: int = 2112,
        sparse_before_full: int = 4,
        remedy_start_after_update: int | None = REMEDY_START_AFTER_UPDATE,
        remedy_action_order: tuple[str, ...] = REMEDY_ACTION_ORDER,
        remedy_passes_per_action: int = REMEDY_PASSES_PER_ACTION,
    ) -> None:
        rows = tuple(int(value) for value in row_ids)
        if not rows or len(rows) != len(set(rows)):
            raise ValueError("Wind maturation needs unique eligible training row IDs.")
        if start_update < 0 or seed < 0 or sparse_before_full <= 0:
            raise ValueError("Maturation start/seed must be nonnegative and replay cadence positive.")
        if remedy_start_after_update is not None:
            remedy_start_after_update = int(remedy_start_after_update)
            if remedy_start_after_update < start_update:
                raise ValueError("The remedy epoch cannot begin before the base schedule.")
            if (remedy_start_after_update - start_update) % (sparse_before_full + 1) != 0:
                raise ValueError("The remedy epoch must begin immediately after a full replay boundary.")
        if (
            not remedy_action_order
            or any(action not in WIND_ACTION_PATHS for action in remedy_action_order)
            or remedy_passes_per_action <= 0
            or any(
                tuple(remedy_action_order).count(action) != int(remedy_passes_per_action)
                for action in WIND_ACTION_PATHS
            )
        ):
            raise ValueError("The remedy schedule needs valid actions and positive pass counts.")
        self.row_ids = rows
        self.start_update = int(start_update)
        self.seed = int(seed)
        self.sparse_before_full = int(sparse_before_full)
        self.remedy_start_after_update = remedy_start_after_update
        self.remedy_action_order = tuple(str(action) for action in remedy_action_order)
        self.remedy_passes_per_action = int(remedy_passes_per_action)

    def _row_order(self, primary_pass: int) -> tuple[int, ...]:
        order = list(range(len(self.row_ids)))
        random.Random(self.seed + int(primary_pass) * 1_000_003).shuffle(order)
        return tuple(order)

    def plan(self, completed_updates_before: int) -> WindMaturationUpdate:
        absolute = int(completed_updates_before)
        if absolute < self.start_update:
            raise ValueError("The controlled schedule starts at its declared u100 base.")
        relative = absolute - self.start_update
        if self.remedy_start_after_update is not None and absolute >= self.remedy_start_after_update:
            return self._plan_remedy(absolute=absolute, relative=relative)

        # Keep this base-stage mapping unchanged: saved u101-u1850 streams are
        # part of the matched historical evidence.
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
                else sparse_capacity_for_recipe(BASE_RECIPE_STAGE)
            ),
            query_seed=self.seed + relative * 104_729,
            recipe_stage=BASE_RECIPE_STAGE,
            sparse_capacity_vector=sparse_capacity_for_recipe(BASE_RECIPE_STAGE),
        )

    def _plan_remedy(self, *, absolute: int, relative: int) -> WindMaturationUpdate:
        assert self.remedy_start_after_update is not None
        remedy_relative = absolute - self.remedy_start_after_update
        cadence = self.sparse_before_full + 1
        cadence_slot = remedy_relative % cadence
        cadence_block = remedy_relative // cadence
        full_replay = cadence_slot == self.sparse_before_full
        remedy_sparse_update = cadence_block * self.sparse_before_full + (
            self.sparse_before_full - 1 if full_replay else cadence_slot
        )
        sparse_per_recipe = len(self.row_ids) * len(self.remedy_action_order)
        if remedy_sparse_update >= sparse_per_recipe:
            replay_count = (sparse_per_recipe - 1) // self.sparse_before_full
            replay_count += int(sparse_per_recipe % self.sparse_before_full == 0)
            final_relative_update = sparse_per_recipe + replay_count
            final_absolute_update = self.remedy_start_after_update + final_relative_update
            raise ValueError(
                f"The bounded Wind QE-full remedy schedule is complete at u{final_absolute_update}."
            )

        recipe_pass, pass_position = divmod(remedy_sparse_update, len(self.row_ids))
        if recipe_pass >= len(self.remedy_action_order):
            raise ValueError("The bounded Wind remedy action-family passes are complete.")
        action = self.remedy_action_order[recipe_pass]
        paths = WIND_ACTION_PATHS[action]

        # The base stage ends just before the next global sparse index. Keep
        # that counter monotone while restarting only the remedy exposure clock.
        boundary_relative = self.remedy_start_after_update - self.start_update
        cadence_block_at_boundary, cadence_slot_at_boundary = divmod(
            boundary_relative, cadence
        )
        if cadence_slot_at_boundary != 0:
            raise ValueError("The remedy transition is no longer at a sparse-clock boundary.")
        global_sparse_start = cadence_block_at_boundary * self.sparse_before_full
        sparse_update = global_sparse_start + remedy_sparse_update
        primary_pass_offset = (global_sparse_start + len(self.row_ids) - 1) // len(self.row_ids)
        primary_pass = primary_pass_offset + recipe_pass
        row_order = self._row_order(primary_pass)
        row_index = row_order[pass_position]
        recipe_capacity = sparse_capacity_for_recipe(QE_FULL_RECIPE_STAGE)
        full_capacity = (("QE", 1.0), ("MM", 1.0))

        return WindMaturationUpdate(
            completed_updates_before=absolute,
            relative_update=relative,
            relative_sparse_update=sparse_update,
            row_index=row_index,
            row_id=self.row_ids[row_index],
            primary_pass=primary_pass,
            pass_position=pass_position,
            phase="scheduled_full_access_replay" if full_replay else "action_family",
            action="full_access_replay" if full_replay else action,
            requested_cut_paths=paths,
            full_access_replay=full_replay,
            replay_sparse_update=sparse_update if full_replay else None,
            capacity_vector=full_capacity if full_replay else recipe_capacity,
            query_seed=self.seed + relative * 104_729,
            recipe_stage=QE_FULL_RECIPE_STAGE,
            sparse_capacity_vector=recipe_capacity,
            recipe_pass=recipe_pass,
            recipe_pass_position=pass_position,
            recipe_sparse_update=remedy_sparse_update,
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
    "BASE_LANE_GPU_BUDGET_SECONDS",
    "BASE_RECIPE_STAGE",
    "MAX_LANE_GPU_BUDGET_SECONDS",
    "QE_FULL_RECIPE_STAGE",
    "REMEDY_ACTION_ORDER",
    "REMEDY_PASSES_PER_ACTION",
    "REMEDY_RECIPE_AMENDMENT_ID",
    "REMEDY_START_AFTER_UPDATE",
    "WIND_ACTION_PATHS",
    "WIND_SPARSE_RECIPE_CAPACITIES",
    "WindMaturationSchedule",
    "WindMaturationUpdate",
    "available_frontier_for_paths",
    "sparse_capacity_for_recipe",
    "validated_lane_gpu_budget_seconds",
]
