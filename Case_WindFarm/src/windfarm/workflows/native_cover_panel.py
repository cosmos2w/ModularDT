"""Geometry-only WindFarm train panel and disjoint native probe selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry import support_weights
from ..study_spatial import downstream_envelope, native_coordinates

ORGANIZER_TRAIN_LAYOUT_INDICES = (4, 24, 70, 88, 115, 119, 174, 177)
ORGANIZER_DEVELOPMENT_LAYOUT_INDICES = (81, 103, 107, 196)
ORGANIZER_SPLIT_ID = "windfarm_run2103_e2475_receiver_local_8_train_4_development_v1"


@dataclass(frozen=True)
class TrainingLayout:
    layout_index: int
    rows: tuple[int, int, int]
    turbine_count: int
    feature_vector: tuple[float, float, float, float]


@dataclass(frozen=True)
class NativeProbeSet:
    row_index: int
    layout_index: int
    flat_indices: np.ndarray
    coordinates_D: np.ndarray
    target_mps: np.ndarray
    quadrature_weights_D3: np.ndarray
    roles: Mapping[str, np.ndarray]
    split: str


def freeze_organizer_layout_split(layouts: Sequence[TrainingLayout]) -> dict[str, Any]:
    """Return the preregistered geometry-only eight/four layout split.

    The exact twelve identities are those selected by the checkpoint-owned
    seed-42 training split and the compact turbine geometry. The split keeps
    every stored direction of a layout together and puts repeated turbine
    counts on both sides where available.
    """

    by_index = {int(item.layout_index): item for item in layouts}
    expected = set(ORGANIZER_TRAIN_LAYOUT_INDICES) | set(ORGANIZER_DEVELOPMENT_LAYOUT_INDICES)
    if len(by_index) != len(layouts) or set(by_index) != expected:
        raise ValueError(
            "receiver-local organizer split requires the frozen twelve layout identities "
            f"{sorted(expected)}, received {sorted(by_index)}"
        )
    for item in layouts:
        if len(item.rows) != 3 or len(set(item.rows)) != 3:
            raise ValueError(f"layout {item.layout_index} must keep its three direction rows together")

    training = [by_index[index] for index in ORGANIZER_TRAIN_LAYOUT_INDICES]
    development = [by_index[index] for index in ORGANIZER_DEVELOPMENT_LAYOUT_INDICES]
    train_counts = {int(item.turbine_count) for item in training}
    development_counts = {int(item.turbine_count) for item in development}
    shared_counts = sorted(train_counts & development_counts)
    if not {6, 29}.issubset(shared_counts):
        raise ValueError("frozen organizer split must retain same-M layout checks for M=6 and M=29")

    def layout_document(item: TrainingLayout) -> dict[str, Any]:
        return {
            "layout_index": int(item.layout_index),
            "rows_direction_order": [int(row) for row in item.rows],
            "turbine_count": int(item.turbine_count),
            "geometry_feature_vector": [float(value) for value in item.feature_vector],
        }

    content: dict[str, Any] = {
        "split_id": ORGANIZER_SPLIT_ID,
        "selection_basis": (
            "fixed identities from checkpoint-owned seed-42 training geometry; split frozen before "
            "new organizer outcome inspection; no field errors or pruning outcomes used"
        ),
        "direction_grouping": "all three stored directions remain in the same layout partition",
        "training_layouts": [layout_document(item) for item in training],
        "development_layouts": [layout_document(item) for item in development],
        "training_layout_indices": list(ORGANIZER_TRAIN_LAYOUT_INDICES),
        "development_layout_indices": list(ORGANIZER_DEVELOPMENT_LAYOUT_INDICES),
        "training_rows_direction_order": [
            int(row) for item in training for row in item.rows
        ],
        "development_rows_direction_order": [
            int(row) for item in development for row in item.rows
        ],
        "same_turbine_count_cross_split": {
            str(count): {
                "training_layouts": [
                    int(item.layout_index) for item in training if int(item.turbine_count) == count
                ],
                "development_layouts": [
                    int(item.layout_index) for item in development if int(item.turbine_count) == count
                ],
            }
            for count in shared_counts
        },
        "training_turbine_count_range": [
            min(int(item.turbine_count) for item in training),
            max(int(item.turbine_count) for item in training),
        ],
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**content, "split_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def write_organizer_split_lock(
    layouts: Sequence[TrainingLayout],
    *,
    checkpoint_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Persist the geometry-only split and its first freeze time/hash."""

    if len(checkpoint_sha256) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_sha256):
        raise ValueError("organizer split lock requires a lowercase SHA256 checkpoint identity")
    split = freeze_organizer_layout_split(layouts)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / "organizer_split_lock.json"
    if lock_path.is_file():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if (
            existing.get("split_sha256") != split["split_sha256"]
            or existing.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise ValueError("existing organizer split lock differs from the frozen geometry/checkpoint")
        return {
            **existing,
            "artifact_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            "path": str(lock_path),
        }

    lock = {
        "workflow": "windfarm_receiver_local_organizer_split_freeze",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "checkpoint_sha256": checkpoint_sha256,
        **split,
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return {**lock, "artifact_sha256": lock_sha256, "path": str(lock_path)}


def _layout_features(view: Any, row: int) -> tuple[int, tuple[float, float, float, float]]:
    counts = np.asarray(view.metadata["n_turbines"], dtype=np.int64)
    turbine_xy = np.asarray(view.metadata["turbine_xy_D"], dtype=np.float64)
    count = int(counts[row])
    active = turbine_xy[row, :count]
    span = np.ptp(active, axis=0) if count > 1 else np.zeros(2, dtype=np.float64)
    footprint = max(float(span[0] * span[1]), 1e-3)
    density = float(count / footprint)
    return count, (float(count), float(span[0]), float(span[1]), density)


def select_training_layouts(
    view: Any,
    training_rows: Sequence[int] | np.ndarray,
    *,
    count: int = 12,
) -> tuple[TrainingLayout, ...]:
    """Select whole training layouts by turbine count, density, and extent.

    The selection reads only compact geometry and group membership. It never
    opens field values or validation/test rows.
    """

    rows = np.asarray(training_rows, dtype=np.int64)
    if count < 1:
        raise ValueError("training panel size must be positive")
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("training rows must be a nonempty one-dimensional array")
    if np.unique(rows).size != rows.size or np.any(rows < 0) or np.any(rows >= view.n_cases):
        raise ValueError("training rows must be unique valid native row indices")
    layout_values = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    grouped: dict[int, list[int]] = {}
    allowed_rows = set(map(int, rows.tolist()))
    for row in rows.tolist():
        grouped.setdefault(int(layout_values[row]), []).append(int(row))
    candidates: list[TrainingLayout] = []
    for layout_index, group_rows in sorted(grouped.items()):
        all_rows = np.flatnonzero(layout_values == layout_index).astype(np.int64).tolist()
        if len(all_rows) != 3 or not set(all_rows).issubset(allowed_rows):
            continue
        ordered_rows = tuple(sorted(all_rows))
        count_value, feature = _layout_features(view, ordered_rows[0])
        candidates.append(TrainingLayout(layout_index, ordered_rows, count_value, feature))
    if len(candidates) < count:
        raise ValueError(f"need {count} complete training layouts, found {len(candidates)}")

    feature_array = np.asarray([item.feature_vector for item in candidates], dtype=np.float64)
    center = np.median(feature_array, axis=0)
    scale = np.std(feature_array, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (feature_array - center) / scale
    distance_to_center = np.sum(standardized**2, axis=1)
    first = int(np.argmin(distance_to_center))
    selected = [first]
    nearest = np.sum((standardized - standardized[first]) ** 2, axis=1)
    while len(selected) < count:
        nearest[selected] = -1.0
        next_index = int(np.argmax(nearest))
        selected.append(next_index)
        distance = np.sum((standardized - standardized[next_index]) ** 2, axis=1)
        nearest = np.minimum(nearest, distance)
    result = [candidates[index] for index in selected]
    result.sort(key=lambda item: item.layout_index)
    return tuple(result)


def _sample_flat_from_z(
    run: Any,
    z_indices: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    nx, ny = int(run.nx), int(run.ny)
    population = int(z_indices.size) * nx * ny
    if count > population:
        raise ValueError("probe quota exceeds available native cells in its z stratum")
    local = rng.choice(population, size=count, replace=False)
    iz = z_indices[local // (nx * ny)]
    rem = local % (nx * ny)
    iy = rem // nx
    ix = rem % nx
    return (ix + nx * (iy + ny * iz)).astype(np.int64)


def _nearby_flat(run: Any, coordinate_D: np.ndarray, diameter_m: float) -> int:
    x_m, y_m, z_m = (float(value) * diameter_m for value in coordinate_D)
    ix = int(np.argmin(np.abs(np.asarray(run.x_m) - x_m)))
    iy = int(np.argmin(np.abs(np.asarray(run.y_m) - y_m)))
    iz = int(np.argmin(np.abs(np.asarray(run.z_m) - z_m)))
    return ix + int(run.nx) * (iy + int(run.ny) * iz)


def _protected_probe_indices(case: Any) -> tuple[list[int], list[int]]:
    """Assign turbine and downstream diagnostic receivers across two probes."""

    search: list[int] = []
    verification: list[int] = []
    seen: set[int] = set()
    support_lower, support_upper = case.support.lower_D, case.support.upper_D
    for turbine, hub in enumerate(case.module_centers):
        candidates = [hub]
        for downstream_offset in (2.0, 5.0):
            point = np.asarray(hub, dtype=np.float64).copy()
            point[0] += downstream_offset
            if np.all(point >= support_lower) and np.all(point <= support_upper):
                candidates.append(point)
        for local_index, point in enumerate(candidates):
            flat = _nearby_flat(case.run, point, case.diameter_m)
            if flat in seen:
                continue
            seen.add(flat)
            (search if (turbine + local_index) % 2 == 0 else verification).append(flat)
    return search, verification


def make_disjoint_native_probes(
    case: Any,
    *,
    query_count: int = 1024,
    seed: int = 42,
) -> tuple[NativeProbeSet, NativeProbeSet]:
    """Create disjoint, hub-stratified exact-cell search and verify probes."""

    if query_count < 16:
        raise ValueError("native oracle probes need at least 16 queries per split")
    run = case.run
    rng = np.random.default_rng(np.random.SeedSequence((int(seed), int(case.index), 901)))
    search_special, verify_special = _protected_probe_indices(case)
    if len(search_special) >= query_count or len(verify_special) >= query_count:
        raise ValueError("protected WindFarm receiver points exceed the requested probe size")
    forbidden = set(search_special) | set(verify_special)
    z_coordinates_D = np.asarray(run.z_m, dtype=np.float64) / float(case.diameter_m)
    hub_height_D = float(case.hub_height_m) / float(case.diameter_m)
    hub_z = np.flatnonzero(np.abs(z_coordinates_D - hub_height_D) <= 0.5)
    other_z = np.flatnonzero(np.abs(z_coordinates_D - hub_height_D) > 0.5)
    if hub_z.size == 0 or other_z.size == 0:
        raise ValueError("native z grid lacks both hub-slab and outside-slab probe support")

    search_hub_present = sum(
        abs(float(run.z_m[(index // (run.nx * run.ny))]) / case.diameter_m - hub_height_D) <= 0.5
        for index in search_special
    )
    verify_hub_present = sum(
        abs(float(run.z_m[(index // (run.nx * run.ny))]) / case.diameter_m - hub_height_D) <= 0.5
        for index in verify_special
    )
    hub_quota = round(query_count * 0.25)
    search_hub_random = max(0, hub_quota - search_hub_present)
    verify_hub_random = max(0, hub_quota - verify_hub_present)
    search_other_random = query_count - len(search_special) - search_hub_random
    verify_other_random = query_count - len(verify_special) - verify_hub_random
    if search_other_random < 0 or verify_other_random < 0:
        raise ValueError("protected receiver points leave no room for the stratified probe quota")

    all_hub = _sample_flat_from_z(run, hub_z, search_hub_random + verify_hub_random, rng)
    all_other = _sample_flat_from_z(run, other_z, search_other_random + verify_other_random, rng)
    if forbidden.intersection(map(int, all_hub.tolist())) or forbidden.intersection(map(int, all_other.tolist())):
        # Random strata can hit a protected cell; deterministically redraw with
        # those cells removed through bounded rejection sampling below.
        def draw_excluding(z_indices: np.ndarray, count: int) -> np.ndarray:
            selected: list[int] = []
            selected_set = set(forbidden)
            while len(selected) < count:
                draw = _sample_flat_from_z(run, z_indices, min(max(count, 8), int(z_indices.size) * run.nx * run.ny), rng)
                for value in draw.tolist():
                    index = int(value)
                    if index not in selected_set:
                        selected.append(index)
                        selected_set.add(index)
                        if len(selected) == count:
                            break
            return np.asarray(selected, dtype=np.int64)

        all_hub = draw_excluding(hub_z, search_hub_random + verify_hub_random)
        all_other = draw_excluding(other_z, search_other_random + verify_other_random)

    search_indices = np.concatenate(
        (
            np.asarray(search_special, dtype=np.int64),
            all_hub[:search_hub_random],
            all_other[:search_other_random],
        )
    )
    verify_start_hub = search_hub_random
    verify_start_other = search_other_random
    verify_indices = np.concatenate(
        (
            np.asarray(verify_special, dtype=np.int64),
            all_hub[verify_start_hub:],
            all_other[verify_start_other:],
        )
    )
    if search_indices.size != query_count or verify_indices.size != query_count:
        raise RuntimeError("native probe construction produced the wrong query count")
    if np.intersect1d(search_indices, verify_indices).size:
        raise RuntimeError("search and verification probes share native field cells")

    axis_weights = support_weights(run.x_m, run.y_m, run.z_m)
    probes: list[NativeProbeSet] = []
    for split, indices in (("search", search_indices), ("verification", verify_indices)):
        coords = native_coordinates(run, indices, case.diameter_m)
        target = np.asarray(run.U[indices], dtype=np.float32).copy()
        ix = indices % run.nx
        iy = (indices // run.nx) % run.ny
        iz = indices // (run.nx * run.ny)
        weights = (
            np.asarray(axis_weights[0])[ix]
            * np.asarray(axis_weights[1])[iy]
            * np.asarray(axis_weights[2])[iz]
        ).astype(np.float64)
        hub_slab = np.abs(coords[:, 2] - hub_height_D) <= 0.5
        downstream = downstream_envelope(coords, np.asarray(case.module_centers, dtype=np.float64))
        relative_xy = coords[:, None, :2] - np.asarray(case.module_centers, dtype=np.float64)[None, :, :2]
        near_turbine = (np.sum(relative_xy**2, axis=-1).min(axis=1) <= 1.5**2) & hub_slab
        roles = {
            "volume": np.ones(query_count, dtype=bool),
            "hub_slab": hub_slab,
            "downstream_envelope": downstream,
            "background": ~downstream & ~hub_slab,
            "near_turbine": near_turbine,
        }
        missing = [name for name, mask in roles.items() if not bool(mask.any())]
        if missing:
            raise ValueError(f"{split} probe lacks protected receiver roles: {missing}")
        probes.append(
            NativeProbeSet(
                int(case.index),
                int(case.layout_index),
                indices,
                coords,
                target,
                weights,
                roles,
                split,
            )
        )
    return probes[0], probes[1]


__all__ = [
    "NativeProbeSet",
    "TrainingLayout",
    "freeze_organizer_layout_split",
    "make_disjoint_native_probes",
    "select_training_layouts",
    "write_organizer_split_lock",
]
