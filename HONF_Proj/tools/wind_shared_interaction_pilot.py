"""Bounded native WindFarm pilot for the shared nonlinear interaction core.

The tool freezes an input-only layout subset, fits its own train-only target
transform, samples only stored native OpenFOAM cells, and optionally trains one
fresh nonlinear candidate. Generated manifests, arrays, checkpoints, metrics,
and figures live under Case_WindFarm/diagnostics/generated/.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import inspect
import json
import linecache
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_IMPORT_STARTED = time.perf_counter()
_PROCESS_STARTED_PERF = _IMPORT_STARTED
_PROCESS_STARTED_UTC = datetime.now(timezone.utc).isoformat()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = PROJECT_ROOT / "Case_WindFarm"
for _path in (PROJECT_ROOT / "src", CASE_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from windfarm.data import WindFarmNativeView, case_batch
from windfarm.model import (
    SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE,
    build_windfarm_model,
)
from windfarm.normalization import (
    VelocityNormalizer,
    VerticalProfileBaseline,
    fit_velocity_statistics,
    read_normalization_json,
    write_normalization_json,
)
from windfarm.shared_interaction import (
    WindFarmSharedInteractionModel,
)
from windfarm.splits import GroupSplit, make_group_split
from windfarm.workflows.directed_packet_pair import ACTIVE_REUSE_CONFIG, _load_role_scales
from windfarm.workflows.evaluate_forward import load_checkpoint as load_retained_checkpoint
from windfarm.workflows.joint_forward import (
    ROLE_NAMES,
    NativeRoleCatalogueCache,
    sample_native_role_queries,
)

from honf_forward_core.interface_fields import interaction_core as shared_interaction_module
from honf_forward_core.interface_fields.interaction_core import (
    InteractionContextCore,
    NonlinearFieldReadout,
)
from honf_forward_core.interface_fields.source_response_operator import SourceResponseOperator
from honf_runtime.compat import load_trusted_checkpoint

IMPORT_SECONDS = time.perf_counter() - _IMPORT_STARTED
SCHEMA_VERSION = 1
SUBSET_ID = "wind_shared_fixed24_v1"
DEFAULT_SEED = 42
DEFAULT_UPDATES = 1000
MAX_UPDATES = 2000
PILOT_GPU_CAP_SECONDS = 90 * 60
DEFAULT_Q = 1024
ROLE_QUERY_COUNTS = {
    "volume": 205,
    "hub_slab": 205,
    "downstream_envelope": 205,
    "near_turbine": 205,
    "background": 204,
}
RETAINED_CHECKPOINT = (
    PROJECT_ROOT
    / "Trained_Results/WindFarm/HONF_Forward_Runs/"
    "Run_2103_20260913_135849_windfarm_dense_pairwise_b16_q8192/"
    "checkpoints/best_field.pt"
)
DATA_ROOT = CASE_ROOT / "Dataset/links/wind_farm"
DERIVED_ROOT = CASE_ROOT / "Dataset/derived/forward_velocity_v1"
DEFAULT_OUTPUT = CASE_ROOT / "diagnostics/generated/shared_interaction_20261007/wind_pilot"


@dataclass(frozen=True)
class FixedPilot:
    manifest: dict[str, Any]
    train_rows: np.ndarray
    validation_rows: np.ndarray


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_view() -> WindFarmNativeView:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Native WindFarm data link is unavailable: {DATA_ROOT}")
    return WindFarmNativeView(DATA_ROOT, allow_npz_metadata_fallback=True)


def _load_original_split(view: WindFarmNativeView) -> GroupSplit:
    groups = np.asarray(view.volume.array("layout_index"), dtype=np.int64)
    canonical = make_group_split(groups, seed=DEFAULT_SEED)
    path = DERIVED_ROOT / "split_indices.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"train", "validation", "test"}:
                raise ValueError("Original WindFarm split artifact has an unexpected partition inventory.")
            stored = {name: np.asarray(archive[name], dtype=np.int64) for name in archive.files}
        for name in ("train", "validation", "test"):
            if not np.array_equal(stored[name], getattr(canonical, name)):
                raise ValueError(f"Stored original WindFarm {name} split differs from the seed-42 layout split.")
    return canonical


def _layout_covariates(view: WindFarmNativeView, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    counts = np.asarray(view.metadata["n_turbines"], dtype=np.int64)
    directions = np.asarray(view.metadata["wd_deg"], dtype=np.float64)
    selected_groups = np.unique(groups[np.asarray(rows, dtype=np.int64)])
    group_count: dict[int, int] = {}
    group_directions: dict[int, tuple[float, ...]] = {}
    for group in selected_groups:
        indices = np.flatnonzero(groups == group)
        values = np.unique(counts[indices])
        if values.size != 1:
            raise ValueError(f"Input layout {group} has direction-varying turbine counts: {values.tolist()}")
        group_count[int(group)] = int(values[0])
        group_directions[int(group)] = tuple(sorted(float(value) for value in np.unique(directions[indices])))
    expected_directions = (270.0, 285.0, 300.0)
    for group, values in group_directions.items():
        if values != expected_directions:
            raise ValueError(f"Input layout {group} has incomplete direction coverage: {values}")
        indices = np.flatnonzero(groups == group)
        if indices.size != 3:
            raise ValueError(f"Input layout {group} has {indices.size} rows; expected all three directions.")
    return selected_groups, np.asarray([group_count[int(group)] for group in selected_groups], dtype=np.int64)


def _select_stratified_layouts(
    view: WindFarmNativeView,
    eligible_rows: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """Select layout IDs from input metadata only, spread over turbine counts."""

    groups = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    eligible_groups, group_counts = _layout_covariates(view, eligible_rows)
    if count < 1 or count > eligible_groups.size:
        raise ValueError(f"Cannot select {count} layouts from {eligible_groups.size} eligible groups.")
    count_by_group = {int(group): int(value) for group, value in zip(eligible_groups, group_counts)}
    by_m: dict[int, list[int]] = defaultdict(list)
    eligible_set = set(map(int, eligible_groups.tolist()))
    for group in eligible_groups:
        by_m[count_by_group[int(group)]].append(int(group))
    categories = sorted(by_m)
    if count < len(categories):
        category_positions = np.rint(np.linspace(0, len(categories) - 1, count)).astype(np.int64)
        chosen_categories = [categories[int(index)] for index in category_positions]
        allocation = {category: 0 for category in categories}
        for category in chosen_categories:
            allocation[category] += 1
        uncovered = sorted(set(categories) - set(chosen_categories))
    else:
        allocation = {category: 1 for category in categories}
        remainder = count - len(categories)
        total_layouts = sum(len(by_m[category]) for category in categories)
        exact = {category: remainder * len(by_m[category]) / total_layouts for category in categories}
        for category in categories:
            allocation[category] += math.floor(exact[category])
        left = count - sum(allocation.values())
        for category in sorted(categories, key=lambda value: (-(exact[value] - math.floor(exact[value])), value))[:left]:
            allocation[category] += 1
        for category in categories:
            allocation[category] = min(allocation[category], len(by_m[category]))
        # If a category was too small to satisfy its proportional share, fill
        # remaining slots from categories that still have unused layouts.
        while sum(allocation.values()) < count:
            available = [category for category in categories if allocation[category] < len(by_m[category])]
            if not available:
                raise RuntimeError("Input-only WindFarm stratification exhausted eligible layouts.")
            category = max(available, key=lambda value: (len(by_m[value]) - allocation[value], -value))
            allocation[category] += 1
        uncovered = []
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for category in categories:
        candidates = np.asarray(sorted(by_m[category]), dtype=np.int64)
        rng.shuffle(candidates)
        selected.extend(int(value) for value in candidates[: allocation[category]])
    if len(selected) != count or not set(selected).issubset(eligible_set):
        raise RuntimeError("Input-only WindFarm layout selection failed its exact-count contract.")
    selected.sort()
    rows = np.asarray(eligible_rows, dtype=np.int64)
    selected_rows = rows[np.isin(groups[rows], np.asarray(selected, dtype=np.int64))]
    selected_directions = np.unique(np.asarray(view.metadata["wd_deg"])[selected_rows]).astype(float).tolist()
    selected_m_counts = Counter(count_by_group[group] for group in selected)
    available_m = sorted(set(count_by_group.values()))
    selected_m = sorted(selected_m_counts)
    metadata = {
        "selection": "seeded input-only turbine-count rank stratification over layout_index",
        "selection_seed": int(seed),
        "layout_count": len(selected),
        "layout_indices": selected,
        "row_indices": selected_rows.astype(int).tolist(),
        "direction_categories": selected_directions,
        "direction_rows_per_layout": 3,
        "turbine_count_by_layout": {str(m): int(selected_m_counts[m]) for m in sorted(selected_m_counts)},
        "available_turbine_counts": available_m,
        "selected_turbine_counts": selected_m,
        "uncovered_turbine_counts": sorted(set(available_m) - set(selected_m)),
        "uncovered_turbine_count_strata_by_limited_selection": uncovered,
        "used_target_values_for_selection": False,
    }
    return selected, metadata


def _freeze_subset(view: WindFarmNativeView, split: GroupSplit, output_dir: Path, *, seed: int) -> FixedPilot:
    path = output_dir / "fixed_subset_manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("subset_id") != SUBSET_ID:
            raise ValueError("Existing WindFarm pilot manifest has an unexpected subset identity.")
        if int(manifest.get("seed", -1)) != int(seed):
            raise ValueError("Existing WindFarm pilot subset was frozen with a different selection seed.")
        train_rows = np.asarray(manifest["train_row_indices"], dtype=np.int64)
        validation_rows = np.asarray(manifest["validation_row_indices"], dtype=np.int64)
        groups = np.asarray(view.metadata["layout_index"], dtype=np.int64)
        if not set(groups[train_rows]).issubset(set(groups[split.train])):
            raise ValueError("Frozen WindFarm training layouts escaped the original TRAIN partition.")
        if not set(groups[validation_rows]).issubset(set(groups[split.validation])):
            raise ValueError("Frozen WindFarm validation layouts escaped the original validation partition.")
        if set(groups[train_rows]) & set(groups[validation_rows]):
            raise ValueError("Frozen WindFarm TRAIN and validation layouts overlap.")
        if train_rows.size != 72 or validation_rows.size != 24:
            raise ValueError("Frozen WindFarm pilot row counts must remain 72 TRAIN / 24 validation.")
        _layout_covariates(view, train_rows)
        _layout_covariates(view, validation_rows)
        if _stable_json_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"}) != manifest.get("manifest_sha256"):
            raise ValueError("Frozen WindFarm subset manifest hash does not verify.")
        return FixedPilot(manifest, train_rows, validation_rows)

    train_groups, train_meta = _select_stratified_layouts(view, split.train, count=24, seed=seed)
    validation_groups, validation_meta = _select_stratified_layouts(view, split.validation, count=8, seed=seed + 1)
    groups = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    train_rows = np.flatnonzero(np.isin(groups, np.asarray(train_groups, dtype=np.int64))).astype(np.int64)
    validation_rows = np.flatnonzero(np.isin(groups, np.asarray(validation_groups, dtype=np.int64))).astype(np.int64)
    if train_rows.size != 72 or validation_rows.size != 24:
        raise ValueError("WindFarm pilot must retain all three direction rows for each selected layout.")
    if set(groups[train_rows]) & set(groups[validation_rows]):
        raise ValueError("WindFarm pilot TRAIN and validation groups overlap.")
    source_split = {
        "seed": DEFAULT_SEED,
        "group_key": "layout_index",
        "original_train_layout_count": int(np.unique(groups[split.train]).size),
        "original_validation_layout_count": int(np.unique(groups[split.validation]).size),
        "original_test_layout_count": int(np.unique(groups[split.test]).size),
        "train_row_indices_sha256": hashlib.sha256(np.ascontiguousarray(split.train).tobytes()).hexdigest(),
        "validation_row_indices_sha256": hashlib.sha256(np.ascontiguousarray(split.validation).tobytes()).hexdigest(),
        "test_target_values_read": False,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "subset_id": SUBSET_ID,
        "source_volume_id": "wind_farm_volume_v1",
        "source_compact_id": "wind_farm_tensor_v1",
        "source_split": source_split,
        "train_layout_selection": train_meta,
        "validation_layout_selection": validation_meta,
        "train_row_indices": train_rows.astype(int).tolist(),
        "validation_row_indices": validation_rows.astype(int).tolist(),
        "train_layout_indices": train_groups,
        "validation_layout_indices": validation_groups,
        "train_row_count": int(train_rows.size),
        "validation_row_count": int(validation_rows.size),
        "direction_categories": [270.0, 285.0, 300.0],
        "field_components": ["Ux", "Uy", "Uz"],
        "field_units": "m/s",
        "used_targets_for_subset_selection": False,
        "seed": int(seed),
    }
    manifest["manifest_sha256"] = _stable_json_hash(manifest)
    _json_write(path, manifest)
    return FixedPilot(manifest, train_rows, validation_rows)


def _update_order(train_rows: np.ndarray, updates: int, seed: int) -> np.ndarray:
    rows = np.asarray(train_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    pieces: list[np.ndarray] = []
    remaining = int(updates)
    while remaining > 0:
        permutation = rows[rng.permutation(rows.size)]
        take = min(remaining, rows.size)
        pieces.append(permutation[:take])
        remaining -= take
    return np.concatenate(pieces).astype(np.int64, copy=False)


def _sample_panel(
    case: Any,
    *,
    seed_parts: tuple[int, ...],
    catalogue_cache: NativeRoleCatalogueCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    sampled = sample_native_role_queries(
        case, rng, ROLE_QUERY_COUNTS, catalogue_cache=catalogue_cache
    )
    if sampled.coordinates_D.shape[0] != DEFAULT_Q:
        raise RuntimeError("WindFarm role sampler did not return the fixed Q=1024 panel.")
    return (
        np.asarray(sampled.coordinates_D, dtype=np.float32),
        np.asarray(sampled.target_mps, dtype=np.float32),
        np.asarray(sampled.flat_indices, dtype=np.int64),
    )


def _panel_prefix_sha256(panel: dict[str, np.ndarray], count: int) -> str:
    digest = hashlib.sha256()
    for key in ("train_panel_rows", "train_panel_visits", "train_panel_coords_D", "train_panel_target_mps", "train_panel_flat_indices"):
        array = np.ascontiguousarray(panel[key][: int(count)])
        digest.update(key.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _write_panel_archive(path: Path, panel: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **panel)
    temporary.replace(path)
    _json_write(path.with_suffix(".json"), metadata)


def _prepare_panels(
    view: WindFarmNativeView,
    pilot: FixedPilot,
    output_dir: Path,
    *,
    updates: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    archive_path = output_dir / "role_sample_panels.npz"
    metadata_path = output_dir / "role_sample_panels.json"
    if archive_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            int(metadata.get("prepared_updates", -1)) >= int(updates)
            and metadata.get("subset_manifest_sha256") == pilot.manifest["manifest_sha256"]
            and metadata.get("role_query_counts") == ROLE_QUERY_COUNTS
        ):
            with np.load(archive_path, allow_pickle=False) as data:
                panel = {name: np.asarray(data[name]) for name in data.files}
            expected_prefix = metadata.get("prefix_sha256_by_update", {}).get(str(int(updates)))
            if expected_prefix != _panel_prefix_sha256(panel, updates):
                raise ValueError("Frozen WindFarm native role panel archive does not match its update-prefix hash.")
            return panel, metadata

    row_order = _update_order(pilot.train_rows, int(updates), seed)
    visit_counts: Counter[int] = Counter()
    positions_by_row: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for update_index, row_value in enumerate(row_order):
        row = int(row_value)
        visit = int(visit_counts[row])
        visit_counts[row] += 1
        positions_by_row[row].append((update_index, visit))
    train_coords: list[np.ndarray | None] = [None] * int(updates)
    train_targets: list[np.ndarray | None] = [None] * int(updates)
    train_indices: list[np.ndarray | None] = [None] * int(updates)
    cache_receipts: list[dict[str, Any]] = []
    prepared_count = 0
    for row in sorted(positions_by_row):
        case = view.run(row)
        cache = NativeRoleCatalogueCache(max_cached_bytes=2 * 1024**3)
        for update_index, visit in positions_by_row[row]:
            coordinates, target, flat = _sample_panel(
                case,
                seed_parts=(int(seed), row, visit, 3401),
                catalogue_cache=cache,
            )
            train_coords[update_index] = coordinates
            train_targets[update_index] = target
            train_indices[update_index] = flat
            prepared_count += 1
            if prepared_count % 24 == 0:
                print(f"[wind-pilot] prepared native role panels {prepared_count}/{updates}", flush=True)
        cache_receipts.append({
            "source_row": int(row),
            "catalogue_builds": int(cache.build_count),
            "catalogue_hits": int(cache.hit_count),
            "catalogue_misses": int(cache.miss_count),
            "catalogue_peak_cached_bytes": int(cache.peak_cached_bytes),
            "catalogue_oversize_bypasses": int(cache.oversize_bypass_count),
        })
        del cache, case
        gc.collect()

    validation_coords: list[np.ndarray] = []
    validation_targets: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []
    for index, row_value in enumerate(pilot.validation_rows):
        row = int(row_value)
        case = view.run(row)
        cache = NativeRoleCatalogueCache(max_cached_bytes=2 * 1024**3)
        coordinates, target, flat = _sample_panel(
            case,
            seed_parts=(int(seed), row, 0, 3441),
            catalogue_cache=cache,
        )
        validation_coords.append(coordinates)
        validation_targets.append(target)
        validation_indices.append(flat)
        if index % 4 == 3:
            print(f"[wind-pilot] prepared fixed validation panels {index + 1}/{len(pilot.validation_rows)}", flush=True)
        del cache, case
    gc.collect()

    visits_in_order = np.empty(int(updates), dtype=np.int16)
    seen_visits: Counter[int] = Counter()
    for index, row_value in enumerate(row_order):
        row = int(row_value)
        visits_in_order[index] = int(seen_visits[row])
        seen_visits[row] += 1
    panel = {
        "train_panel_rows": np.asarray(row_order, dtype=np.int64),
        "train_panel_visits": visits_in_order,
        "train_panel_coords_D": np.stack(train_coords),
        "train_panel_target_mps": np.stack(train_targets),
        "train_panel_flat_indices": np.stack(train_indices),
        "validation_rows": np.asarray(pilot.validation_rows, dtype=np.int64),
        "validation_coords_D": np.stack(validation_coords),
        "validation_target_mps": np.stack(validation_targets),
        "validation_flat_indices": np.stack(validation_indices),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "prepared_updates": int(updates),
        "q_per_update": DEFAULT_Q,
        "role_query_counts": ROLE_QUERY_COUNTS,
        "role_order": list(ROLE_NAMES),
        "query_source": "exact stored native cell centres and U.npy target values, sampled by maintained role masks",
        "training_rows_sha256": hashlib.sha256(np.ascontiguousarray(pilot.train_rows).tobytes()).hexdigest(),
        "validation_rows_sha256": hashlib.sha256(np.ascontiguousarray(pilot.validation_rows).tobytes()).hexdigest(),
        "subset_manifest_sha256": pilot.manifest["manifest_sha256"],
        "train_panel_prefix_sha256": _panel_prefix_sha256(panel, updates),
        "prefix_sha256_by_update": {
            str(boundary): _panel_prefix_sha256(panel, boundary)
            for boundary in (DEFAULT_UPDATES, MAX_UPDATES)
            if boundary <= int(updates)
        },
        "training_row_visits": {str(row): int(count) for row, count in sorted(visit_counts.items())},
        "native_catalogue_cache_by_training_row": cache_receipts,
        "native_catalogue_cache_by_validation_row": "one exact-geometry catalogue per selected validation row",
        "effective_passes_over_selected_rows": float(updates / pilot.train_rows.size),
        "validation_sampling": "fixed one-panel-per-row from input-only native role geometry; reused at every review",
        "test_target_values_read": False,
    }
    _write_panel_archive(archive_path, panel, metadata)
    return panel, metadata


def prepare_pilot(output_dir: Path, *, updates: int = DEFAULT_UPDATES, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    if updates not in (DEFAULT_UPDATES, MAX_UPDATES):
        raise ValueError("The WindFarm pilot supports only 1000 updates or a separately authorized 2000-update horizon.")
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    view = _load_view()
    split = _load_original_split(view)
    pilot = _freeze_subset(view, split, output_dir, seed=seed)

    normalization_path = output_dir / "train_only_normalization.json"
    normalization_binding_path = output_dir / "train_only_normalization.binding.json"
    training_rows_sha256 = hashlib.sha256(np.ascontiguousarray(pilot.train_rows).tobytes()).hexdigest()
    if normalization_path.is_file():
        normalizer, profile = read_normalization_json(normalization_path)
        if normalizer.source_rows != pilot.train_rows.size:
            raise ValueError("Existing WindFarm pilot normalization has the wrong TRAIN row count.")
        if normalizer.seed != DEFAULT_SEED or normalizer.sample_count_per_row != 8192:
            raise ValueError("Existing WindFarm pilot normalization does not match the frozen native fit protocol.")
        if not normalization_binding_path.is_file():
            raise ValueError("Existing WindFarm normalization has no frozen subset binding receipt.")
        binding = json.loads(normalization_binding_path.read_text(encoding="utf-8"))
        if binding != {
            "subset_manifest_sha256": pilot.manifest["manifest_sha256"],
            "training_row_indices_sha256": training_rows_sha256,
            "normalizer_sha256": _sha256(normalization_path),
        }:
            raise ValueError("Existing WindFarm normalization is not bound to this exact frozen TRAIN subset.")
    else:
        normalizer, profile = fit_velocity_statistics(view.volume, pilot.train_rows, samples_per_row=8192, seed=DEFAULT_SEED)
        write_normalization_json(normalization_path, normalizer, profile)
        _json_write(normalization_binding_path, {
            "subset_manifest_sha256": pilot.manifest["manifest_sha256"],
            "training_row_indices_sha256": training_rows_sha256,
            "normalizer_sha256": _sha256(normalization_path),
        })
    _, panel_metadata = _prepare_panels(view, pilot, output_dir, updates=updates, seed=seed)
    role_scales = _load_role_scales()
    receipt = {
        "status": "prepared_cpu_only",
        "subset_id": SUBSET_ID,
        "subset_manifest_sha256": pilot.manifest["manifest_sha256"],
        "panel_archive": str(output_dir / "role_sample_panels.npz"),
        "panel_metadata": panel_metadata,
        "train_only_normalization": normalizer.to_dict(),
        "train_only_normalization_binding": json.loads(normalization_binding_path.read_text(encoding="utf-8")),
        "train_only_background_profile": profile.to_dict() if profile is not None else None,
        "role_loss_scales_mps": role_scales,
        "role_loss_scale_source": {
            "path": str(ACTIVE_REUSE_CONFIG),
            "sha256": _sha256(ACTIVE_REUSE_CONFIG),
            "key": "forward.stage_a.role_loss_scales_mps",
        },
        "original_split": pilot.manifest["source_split"],
        "elapsed_cpu_preparation_seconds": time.perf_counter() - start,
        "module_import_seconds": IMPORT_SECONDS,
        "native_volume_memory_mapped": True,
        "new_solver_calls": 0,
        "test_target_values_read": False,
        "gpu_launched": False,
    }
    _json_write(output_dir / "preparation_receipt.json", receipt)
    return receipt


def _role_slices() -> dict[str, slice]:
    offset = 0
    result: dict[str, slice] = {}
    for role in ROLE_NAMES:
        count = int(ROLE_QUERY_COUNTS[role])
        result[role] = slice(offset, offset + count)
        offset += count
    if offset != DEFAULT_Q:
        raise RuntimeError(f"WindFarm role query counts sum to {offset}, expected Q={DEFAULT_Q}.")
    return result


def _role_errors(
    prediction_mps: np.ndarray,
    target_mps: np.ndarray,
    *,
    role_slices: dict[str, slice],
    role_scales: dict[str, float],
) -> dict[str, Any]:
    if prediction_mps.shape != target_mps.shape or prediction_mps.shape[-1] != 3:
        raise ValueError("WindFarm candidate/target predictions must align as [Q,3].")
    output: dict[str, Any] = {}
    objective_terms: list[float] = []
    for role in ROLE_NAMES:
        selection = role_slices[role]
        delta = np.asarray(prediction_mps[selection], dtype=np.float64) - np.asarray(target_mps[selection], dtype=np.float64)
        component_mse = float(np.mean(delta * delta))
        component_rmse = math.sqrt(max(component_mse, 0.0))
        vector_rmse = math.sqrt(max(float(np.mean(np.sum(delta * delta, axis=-1))), 0.0))
        objective_terms.append(component_mse / float(role_scales[role]) ** 2)
        output[role] = {
            "component_rmse_mps": component_rmse,
            "vector_rmse_mps": vector_rmse,
            "role_scale_mps": float(role_scales[role]),
            "query_count": int(selection.stop - selection.start),
        }
    output["equal_role_scaled_objective"] = float(np.mean(objective_terms))
    return output


def _summarize_rows(rows: list[dict[str, Any]], *, model_name: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty WindFarm validation panel.")
    result: dict[str, Any] = {
        "model": model_name,
        "layout_count": len({int(row["layout_index"]) for row in rows}),
        "row_count": len(rows),
        "equal_case_role_metrics": {},
        "equal_case_role_scaled_objective": float(np.mean([row["equal_role_scaled_objective"] for row in rows])),
        "rows": rows,
    }
    for role in ROLE_NAMES:
        values = np.asarray([row["roles"][role]["component_rmse_mps"] for row in rows], dtype=np.float64)
        vector = np.asarray([row["roles"][role]["vector_rmse_mps"] for row in rows], dtype=np.float64)
        result["equal_case_role_metrics"][role] = {
            "component_rmse_mps_mean": float(values.mean()),
            "component_rmse_mps_p95": float(np.quantile(values, 0.95)),
            "component_rmse_mps_worst": float(values.max()),
            "vector_rmse_mps_mean": float(vector.mean()),
            "vector_rmse_mps_p95": float(np.quantile(vector, 0.95)),
            "vector_rmse_mps_worst": float(vector.max()),
        }
    return result


def _row_record(case: Any, errors: dict[str, Any], *, model_name: str, updates: int) -> dict[str, Any]:
    return {
        "model": model_name,
        "updates": int(updates),
        "source_index": int(case.index),
        "case": str(case.case),
        "layout_index": int(case.layout_index),
        "n_turbines": int(case.n_turbines),
        "wind_direction_deg": float(case.wind_direction_deg),
        "equal_role_scaled_objective": float(errors["equal_role_scaled_objective"]),
        "roles": {role: dict(errors[role]) for role in ROLE_NAMES},
    }


def _save_metric_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = ["model", "updates", "source_index", "case", "layout_index", "n_turbines", "wind_direction_deg", "role", "component_rmse_mps", "vector_rmse_mps", "query_count"]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for row in summary["rows"]:
                for role in ROLE_NAMES:
                    metrics = row["roles"][role]
                    writer.writerow({
                        "model": summary["model"],
                        "updates": row["updates"],
                        "source_index": row["source_index"],
                        "case": row["case"],
                        "layout_index": row["layout_index"],
                        "n_turbines": row["n_turbines"],
                        "wind_direction_deg": row["wind_direction_deg"],
                        "role": role,
                        "component_rmse_mps": metrics["component_rmse_mps"],
                        "vector_rmse_mps": metrics["vector_rmse_mps"],
                        "query_count": metrics["query_count"],
                    })
    temporary.replace(path)


def _tensor_on_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _visible_physical_gpu(device: torch.device, physical_gpu: int | None) -> dict[str, Any]:
    if device.type != "cuda":
        if physical_gpu is not None:
            raise ValueError("--physical-gpu is valid only for a CUDA pilot.")
        return {"device": str(device), "gpu_associated": False}
    if physical_gpu is None:
        raise ValueError("A CUDA pilot requires --physical-gpu for the authorized device receipt.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the ModularDT environment.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    logical_index = 0 if device.index is None else int(device.index)
    if visible:
        items = [item.strip() for item in visible.split(",") if item.strip()]
        if all(item.isdigit() for item in items):
            if str(int(physical_gpu)) not in items:
                raise ValueError(f"Physical GPU {physical_gpu} is not present in CUDA_VISIBLE_DEVICES={visible!r}.")
            expected_logical = items.index(str(int(physical_gpu)))
            if logical_index != expected_logical:
                raise ValueError(f"cuda:{logical_index} maps to a different visible GPU; expected cuda:{expected_logical} for physical GPU {physical_gpu}.")
        elif len(items) == 1:
            # UUID-based visibility is externally audited by the launch receipt;
            # PyTorch's logical ordinal is then necessarily zero.
            if logical_index != 0:
                raise ValueError("A single UUID-visible physical GPU must be addressed as cuda:0.")
        else:
            raise ValueError("Cannot verify a multi-entry UUID CUDA_VISIBLE_DEVICES mapping from the pilot process.")
    elif logical_index != int(physical_gpu):
        raise ValueError("Without CUDA_VISIBLE_DEVICES, logical and authorized physical GPU indices must match.")
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "authorized_physical_index": int(physical_gpu),
        "cuda_visible_devices": visible,
        "device_name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "uuid": str(getattr(properties, "uuid", "unavailable")),
        "gpu_associated": True,
    }


def _set_seed(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _set_symlink_alias(alias: Path, target: Path) -> None:
    temporary = alias.with_name(f".{alias.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target.name)
    temporary.replace(alias)


def _plot_history(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    updates = [int(record["update_count"]) for record in records]
    train = [float(record["training_objective_mean"]) for record in records]
    validation = [float(record["candidate_validation"]["equal_case_role_scaled_objective"]) for record in records]
    fig, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    axis.plot(updates, train, marker="o", label="training role objective")
    axis.plot(updates, validation, marker="s", label="validation role objective")
    axis.set_xlabel("optimizer updates")
    axis.set_ylabel("equal-role scaled MSE")
    axis.set_title("WindFarm shared-core pilot")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=140)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    temporary.replace(path)


def _eval_candidate(
    model: WindFarmSharedInteractionModel,
    view: WindFarmNativeView,
    panel: dict[str, np.ndarray],
    pilot: FixedPilot,
    role_scales: dict[str, float],
    device: torch.device,
    *,
    updates: int,
) -> dict[str, Any]:
    model.eval()
    slices = _role_slices()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for panel_index, row_value in enumerate(panel["validation_rows"]):
            row = int(row_value)
            case = view.run(row)
            query = _tensor_on_device(panel["validation_coords_D"][panel_index][None], device)
            target = panel["validation_target_mps"][panel_index]
            prepared = model.prepare_case(case, device=device)
            prediction_std = model.predict_standardized(prepared, query, chunk_size=512)
            prediction_mps = model.denormalize_tensor(prediction_std)[0].detach().cpu().numpy()
            errors = _role_errors(prediction_mps, target, role_slices=slices, role_scales=role_scales)
            records.append(_row_record(case, errors, model_name="shared_nonlinear", updates=updates))
            del prepared, query, case
    model.train()
    return _summarize_rows(records, model_name="shared_nonlinear")


def _eval_controls(
    view: WindFarmNativeView,
    panel: dict[str, np.ndarray],
    pilot: FixedPilot,
    role_scales: dict[str, float],
    profile: VerticalProfileBaseline,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Retained Run 2103 field checkpoint is unavailable: {checkpoint_path}")
    slices = _role_slices()
    background_records: list[dict[str, Any]] = []
    retained_records: list[dict[str, Any]] = []
    retained_model = None
    for panel_index, row_value in enumerate(panel["validation_rows"]):
        row = int(row_value)
        case = view.run(row)
        coordinates = panel["validation_coords_D"][panel_index]
        target = panel["validation_target_mps"][panel_index]
        background_mps = profile.predict(coordinates[:, 2])
        background_errors = _role_errors(background_mps, target, role_slices=slices, role_scales=role_scales)
        background_records.append(_row_record(case, background_errors, model_name="train_profile_background", updates=0))
        if retained_model is None:
            bootstrap_batch = case_batch(case, coordinates)
            retained_model, checkpoint = load_retained_checkpoint(
                checkpoint_path,
                device=device,
                materialization_batch=bootstrap_batch,
            )
            if int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))) != 2475:
                raise ValueError("Retained Run 2103 checkpoint is not the manifest-selected epoch 2475.")
        batch = case_batch(case, coordinates).to(device)
        prepared = retained_model.prepare_case(batch)
        with torch.no_grad():
            prediction = retained_model.predict_physical(
                prepared,
                batch.query_xy,
                batch.query_features,
                receiver_chunk_size=512,
            )[0].detach().cpu().numpy()
        retained_errors = _role_errors(prediction, target, role_slices=slices, role_scales=role_scales)
        retained_records.append(_row_record(case, retained_errors, model_name="Run2103_e2475", updates=2475))
        del prepared, batch, case
    return (
        _summarize_rows(background_records, model_name="train_profile_background"),
        _summarize_rows(retained_records, model_name="Run2103_e2475"),
    )


def _load_pilot_artifacts(
    output_dir: Path,
    *,
    requested_updates: int,
) -> tuple[WindFarmNativeView, FixedPilot, VelocityNormalizer, VerticalProfileBaseline, dict[str, np.ndarray], dict[str, Any], dict[str, float]]:
    manifest_path = output_dir / "fixed_subset_manifest.json"
    normalization_path = output_dir / "train_only_normalization.json"
    panel_path = output_dir / "role_sample_panels.npz"
    panel_metadata_path = output_dir / "role_sample_panels.json"
    for path in (manifest_path, normalization_path, panel_path, panel_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared CPU pilot artifact {path}; run the prepare subcommand first.")
    view = _load_view()
    split = _load_original_split(view)
    pilot = _freeze_subset(view, split, output_dir, seed=DEFAULT_SEED)
    normalizer, profile = read_normalization_json(normalization_path)
    if profile is None:
        raise ValueError("WindFarm pilot requires the train-only vertical background profile.")
    binding_path = output_dir / "train_only_normalization.binding.json"
    if not binding_path.is_file():
        raise FileNotFoundError("WindFarm TRAIN-only normalization has no immutable subset binding receipt.")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    expected_binding = {
        "subset_manifest_sha256": pilot.manifest["manifest_sha256"],
        "training_row_indices_sha256": hashlib.sha256(np.ascontiguousarray(pilot.train_rows).tobytes()).hexdigest(),
        "normalizer_sha256": _sha256(normalization_path),
    }
    if binding != expected_binding:
        raise ValueError("WindFarm normalization binding does not match the selected TRAIN layouts.")
    panel_metadata = json.loads(panel_metadata_path.read_text(encoding="utf-8"))
    if int(panel_metadata.get("prepared_updates", -1)) < int(requested_updates):
        raise ValueError("Frozen native role panels do not cover the requested optimizer horizon.")
    if panel_metadata.get("subset_manifest_sha256") != pilot.manifest["manifest_sha256"]:
        raise ValueError("Frozen WindFarm panels are bound to a different input-only subset.")
    with np.load(panel_path, allow_pickle=False) as archive:
        panel = {name: np.asarray(archive[name]) for name in archive.files}
    prefix_hash = panel_metadata.get("prefix_sha256_by_update", {}).get(str(int(requested_updates)))
    if prefix_hash is not None and prefix_hash != _panel_prefix_sha256(panel, requested_updates):
        raise ValueError("WindFarm native role panel prefix hash does not verify.")
    if panel["train_panel_rows"].shape[0] < requested_updates:
        raise ValueError("WindFarm native role panels have fewer rows than the requested update horizon.")
    if not np.array_equal(panel["validation_rows"], pilot.validation_rows):
        raise ValueError("Frozen WindFarm validation panels do not match the selected validation rows.")
    role_scales = _load_role_scales()
    return view, pilot, normalizer, profile, panel, panel_metadata, role_scales


def _training_loss(
    model: WindFarmSharedInteractionModel,
    prepared: Any,
    coordinates_D: torch.Tensor,
    target_mps: torch.Tensor,
    role_scales: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    standardized = model.predict_standardized(prepared, coordinates_D, chunk_size=512)
    physical = model.denormalize_tensor(standardized)
    slices = _role_slices()
    terms = [
        (physical[:, selection] - target_mps[:, selection]).square().mean()
        / (float(role_scales[role]) ** 2)
        for role, selection in slices.items()
    ]
    loss = torch.stack(terms).mean()
    return loss, physical


def _case_visits(
    rows: np.ndarray,
    *,
    updates: int,
    layout_by_row: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(rows[: int(updates)], dtype=np.int64)
    row_counts = Counter(map(int, selected.tolist()))
    layout_counts = Counter(map(int, layout_by_row[selected].tolist()))
    return {
        "optimizer_updates": int(updates),
        "native_direction_case_visits": int(selected.size),
        "selected_direction_row_visits": int(selected.size),
        "unique_direction_rows_visited": len(row_counts),
        "effective_passes_over_72_direction_rows": float(updates / 72.0),
        "selected_layout_visits": int(sum(layout_counts.values())),
        "unique_layouts_visited": len(layout_counts),
        "mean_direction_case_visits_per_selected_layout": float(updates / max(len(layout_counts), 1)),
        "layout_visit_counts": {str(layout): int(count) for layout, count in sorted(layout_counts.items())},
        "direction_row_visit_counts": {str(row): int(count) for row, count in sorted(row_counts.items())},
    }


def _model_recipe() -> dict[str, Any]:
    return {
        "forward_architecture": SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE,
        "hidden": 32,
        "message": 32,
        "max_sources": 30,
    }


def _milestone_path(output_dir: Path, update_count: int) -> Path:
    return output_dir / "checkpoints" / f"updates_{int(update_count):05d}.pt"


def _save_training_checkpoint(
    output_dir: Path,
    *,
    update_count: int,
    model: WindFarmSharedInteractionModel,
    optimizer: torch.optim.Optimizer,
    optimizer_updates: list[float],
    history: list[dict[str, Any]],
    best_update: int,
    best_validation_objective: float,
    subset_manifest_sha256: str,
    panel_prefix_sha256: str,
    normalization_sha256: str,
    controls: dict[str, Any] | None,
) -> Path:
    path = _milestone_path(output_dir, update_count)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "wind_shared_fixed24_v1_source_resolved_nonlinear",
        "model_recipe": _model_recipe(),
        "model_config": model.core.config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_updates": [float(value) for value in optimizer_updates],
        "history": history,
        "update_count": int(update_count),
        "best_update": int(best_update),
        "best_validation_objective": float(best_validation_objective),
        "subset_manifest_sha256": subset_manifest_sha256,
        "panel_prefix_sha256": panel_prefix_sha256,
        "normalization_sha256": normalization_sha256,
        "controls": controls,
        "units": {"coordinates": "rotor diameters", "outputs": "m/s", "normalization": "TRAIN-only dimensionless U/U_ref"},
        "target_input_contract": "native U is used only as supervised targets; prepare_case receives geometry/context only",
    }
    _atomic_torch_save(path, payload)
    _set_symlink_alias(output_dir / "checkpoints" / "latest.pt", path)
    best_path = _milestone_path(output_dir, best_update)
    if best_path.is_file():
        _set_symlink_alias(output_dir / "checkpoints" / "best_field.pt", best_path)
    return path


def _load_resume(
    path: Path,
    *,
    model: WindFarmSharedInteractionModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    subset_manifest_sha256: str,
    normalization_sha256: str,
    panel: dict[str, np.ndarray],
) -> tuple[int, list[float], list[dict[str, Any]], int, float, dict[str, Any] | None]:
    checkpoint = load_trusted_checkpoint(path, map_location=device)
    checkpoint_update = int(checkpoint.get("update_count", -1))
    valid_bound_prefixes = {
        boundary: _panel_prefix_sha256(panel, boundary)
        for boundary in (DEFAULT_UPDATES, MAX_UPDATES)
        if boundary <= int(panel["train_panel_rows"].shape[0]) and boundary >= checkpoint_update
    }
    saved_prefix = str(checkpoint.get("panel_prefix_sha256", ""))
    if saved_prefix not in set(valid_bound_prefixes.values()):
        raise ValueError("WindFarm resume checkpoint panel prefix does not match any exact frozen continuation horizon.")
    expected = {
        "subset_manifest_sha256": subset_manifest_sha256,
        "normalization_sha256": normalization_sha256,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"WindFarm resume checkpoint has a different {key} binding.")
    if checkpoint.get("model_recipe") != _model_recipe():
        raise ValueError("WindFarm resume checkpoint belongs to another model recipe.")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return (
        int(checkpoint["update_count"]),
        list(map(float, checkpoint.get("optimizer_updates", []))),
        list(checkpoint.get("history", [])),
        int(checkpoint.get("best_update", checkpoint["update_count"])),
        float(checkpoint.get("best_validation_objective", float("inf"))),
        checkpoint.get("controls"),
    )


def _save_json_metric(output_dir: Path, update_count: int, payload: dict[str, Any]) -> None:
    _json_write(output_dir / "metrics" / f"validation_updates_{int(update_count):05d}.json", payload)


def _write_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    _json_write(output_dir / "metrics" / "training_history.json", {"records": history})
    _plot_history(output_dir / "figures" / "training_curve.png", history)


def _training_process_ledger(output_dir: Path, process_record: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "gpu_process_ledger.json"
    records = []
    if path.is_file():
        records = json.loads(path.read_text(encoding="utf-8")).get("processes", [])
    records.append(process_record)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "budget_cap_gpu_associated_seconds": PILOT_GPU_CAP_SECONDS,
        "processes": records,
        "cumulative_gpu_associated_seconds": float(sum(
            float(record["outer_process_elapsed_seconds"])
            for record in records if record.get("gpu_associated")
        )),
    }
    _json_write(path, payload)
    return payload


def _shared_code_receipt(runtime_trace: dict[str, Any] | None = None) -> dict[str, Any]:
    shared_path = PROJECT_ROOT / "src/honf_forward_core/interface_fields/interaction_core.py"
    operator_path = PROJECT_ROOT / "src/honf_forward_core/interface_fields/source_response_operator.py"
    adapter_path = CASE_ROOT / "src/windfarm/shared_interaction.py"
    shared_geometry = shared_interaction_module._geometry
    operator_geometry = inspect.getmodule(SourceResponseOperator).__dict__["_geometry"]
    shared_read = InteractionContextCore._read_features
    operator_read = SourceResponseOperator._read_features
    return {
        "executed_class": "honf_forward_core.interface_fields.interaction_core.NonlinearFieldReadout",
        "context_base_class": "honf_forward_core.interface_fields.interaction_core.InteractionContextCore",
        "wind_adapter_class": "windfarm.shared_interaction.WindFarmSharedInteractionModel",
        "declared_call_path": [
            "WindFarmSharedInteractionModel.prepare_case",
            "NonlinearFieldReadout.prepare",
            "InteractionContextCore.prepare_context",
            "interaction_core._geometry",
            "NonlinearFieldReadout.predict",
            "InteractionContextCore._read_features",
            "NonlinearFieldReadout.source_read",
            "physical_source_measure_reduction",
            "NonlinearFieldReadout.field_head",
        ],
        "runtime_trace": runtime_trace,
        "shared_function_identity_with_thermal_source_response": {
            "geometry_function_module": shared_geometry.__module__,
            "geometry_function_qualname": shared_geometry.__qualname__,
            "geometry_function_is_same_imported_object": bool(shared_geometry is operator_geometry),
            "read_feature_module": shared_read.__module__,
            "read_feature_qualname": shared_read.__qualname__,
            "read_feature_is_same_inherited_object": bool(shared_read is operator_read),
            "context_method_module": InteractionContextCore.prepare_context.__module__,
            "context_method_qualname": InteractionContextCore.prepare_context.__qualname__,
            "source_operator_prepare_context_dispatches_to_same_base_via_super": bool(
                InteractionContextCore in SourceResponseOperator.__mro__
                and "super" in SourceResponseOperator.prepare_context.__code__.co_names
                and "prepare_context" in SourceResponseOperator.prepare_context.__code__.co_names
            ),
        },
        "wind_shared_core_sha256": _sha256(shared_path),
        "thermal_source_response_sha256": _sha256(operator_path),
        "wind_adapter_sha256": _sha256(adapter_path),
        "architecture": "same shared source-resolved context and geometry/read primitives; separate Wind weights, dimensions, nonlinear head and normalization",
        "wind_source_width": 2,
        "wind_context_width": 11,
        "wind_environment_width": 7,
        "wind_query_width": 7,
        "wind_spatial_dim": 3,
        "wind_velocity_axis_contract": ["Ux along wind-aligned x", "Uy lateral", "Uz vertical"],
        "wind_velocity_units": "m/s",
        "wind_coordinate_units": "rotor diameters; D=80 m",
        "environment_representation": "8 geometry-only support-box quadrature atoms; D^3 normalized measures; no target velocities",
        "target_velocity_input": False,
        "new_solver_calls": 0,
    }


def _runtime_shared_trace(
    model: WindFarmSharedInteractionModel,
    view: WindFarmNativeView,
    pilot: FixedPilot,
    panel: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    row = int(pilot.validation_rows[0])
    case = view.run(row)
    query = _tensor_on_device(panel["validation_coords_D"][0, :64][None], device)
    core_path = Path(shared_interaction_module.__file__).resolve()
    adapter_path = Path(inspect.getfile(WindFarmSharedInteractionModel)).resolve()
    watched_names = {
        "prepare",
        "prepare_context",
        "_normalized_measure",
        "_geometry",
        "_read_features",
        "predict",
        "prepare_case",
    }
    call_counts: Counter[str] = Counter()
    actual_calls: list[dict[str, Any]] = []
    source_order: list[dict[str, Any]] = []
    hook_events: list[dict[str, Any]] = []
    sequence = 0

    def next_order(kind: str, detail: dict[str, Any]) -> None:
        nonlocal sequence
        sequence += 1
        source_order.append({"order": sequence, "event": kind, **detail})

    def profiler(frame, event, _arg):
        if event != "call" or frame.f_code.co_name not in watched_names:
            return
        path = Path(frame.f_code.co_filename).resolve()
        if path not in (core_path, adapter_path):
            return
        qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        key = f"{frame.f_globals.get('__name__', '<unknown>')}.{qualname}"
        call_counts[key] += 1
        actual_calls.append({
            "module": frame.f_globals.get("__name__", "<unknown>"),
            "qualname": qualname,
            "source_file": str(path),
            "first_line": int(frame.f_code.co_firstlineno),
        })
        next_order("python_call", {"qualified_name": key})

    predict_code = NonlinearFieldReadout.predict.__code__

    def line_tracer(frame, event, _arg):
        if frame.f_code is predict_code and event == "line":
            source = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
            seen = {item["event"] for item in source_order}
            if "self.source_read(" in source and "nonlinear_source_read_statement" not in seen:
                next_order("nonlinear_source_read_statement", {"line": int(frame.f_lineno), "source": source})
            elif "prepared.source_measure" in source and ".sum(2)" in source and "physical_source_measure_reduction_statement" not in seen:
                next_order("physical_source_measure_reduction_statement", {"line": int(frame.f_lineno), "source": source})
            elif "self.field_head(" in source and "nonlinear_field_head_statement" not in seen:
                next_order("nonlinear_field_head_statement", {"line": int(frame.f_lineno), "source": source})
        return line_tracer

    def module_hook(module_name: str):
        def hook(_module, inputs, output):
            event = {
                "module": module_name,
                "input_shapes": [list(value.shape) for value in inputs if torch.is_tensor(value)],
                "output_shape": list(output.shape) if torch.is_tensor(output) else None,
            }
            hook_events.append(event)
            next_order("module_forward_hook", event)
        return hook

    handles = [
        model.core.source_read.register_forward_hook(module_hook("NonlinearFieldReadout.source_read")),
        model.core.field_head.register_forward_hook(module_hook("NonlinearFieldReadout.field_head")),
    ]
    previous_profile = sys.getprofile()
    previous_trace = sys.gettrace()
    model.eval()
    try:
        sys.setprofile(profiler)
        sys.settrace(line_tracer)
        prepared = model.prepare_case(case, device=device)
        with torch.no_grad():
            prediction = model.predict_standardized(prepared, query, chunk_size=64)
        _synchronize(device)
    finally:
        sys.settrace(previous_trace)
        sys.setprofile(previous_profile)
        for handle in handles:
            handle.remove()
    source_line_order = {
        event: min(
            (int(item["order"]) for item in source_order if item["event"] == event),
            default=None,
        )
        for event in (
            "nonlinear_source_read_statement",
            "physical_source_measure_reduction_statement",
            "nonlinear_field_head_statement",
        )
    }
    result = {
        "execution_device": str(device),
        "source_row": row,
        "case": str(case.case),
        "layout_index": int(case.layout_index),
        "n_turbines_M": int(case.n_turbines),
        "query_count": int(query.shape[1]),
        "qualified_function_call_counts": dict(sorted(call_counts.items())),
        "actual_shared_module_function_entries": actual_calls,
        "executed_source_read_reduction_head_order": source_order,
        "nonlinear_module_forward_shapes": hook_events,
        "executed_line_order": source_line_order,
        "source_read_precedes_physical_measure_reduction": bool(
            source_line_order["nonlinear_source_read_statement"] is not None
            and source_line_order["physical_source_measure_reduction_statement"] is not None
            and source_line_order["nonlinear_source_read_statement"] < source_line_order["physical_source_measure_reduction_statement"]
        ),
        "field_head_follows_source_reduction": bool(
            source_line_order["physical_source_measure_reduction_statement"] is not None
            and source_line_order["nonlinear_field_head_statement"] is not None
            and source_line_order["physical_source_measure_reduction_statement"] < source_line_order["nonlinear_field_head_statement"]
        ),
        "prediction_shape": list(prediction.shape),
        "prediction_finite": bool(torch.isfinite(prediction).all()),
        "target_values_passed_to_scene_or_read": False,
    }
    del prepared, prediction, query, case
    return result


def _sample_role_coordinates_only(
    case: Any,
    *,
    query_count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if query_count % DEFAULT_Q:
        raise ValueError("Cost benchmark Q must be a positive multiple of the fixed role panel size 1024.")
    factor = int(query_count // DEFAULT_Q)
    counts = {role: int(ROLE_QUERY_COUNTS[role]) * factor for role in ROLE_NAMES}
    cache = NativeRoleCatalogueCache(max_cached_bytes=2 * 1024**3)
    catalogue = cache.get(case)
    rng = np.random.default_rng(int(seed))
    pieces = []
    for role in ROLE_NAMES:
        cdf = catalogue.role_cdf[role]
        positions = np.searchsorted(cdf, rng.random(counts[role]), side="right")
        positions = np.minimum(positions, cdf.size - 1)
        valid = catalogue.role_indices.get(role)
        selected = positions if valid is None else valid[positions]
        pieces.append(catalogue.coordinates_D[selected])
    coordinates = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
    if coordinates.shape != (query_count, 3):
        raise RuntimeError(f"Native cost-query sampler returned {coordinates.shape}, expected {(query_count, 3)}.")
    return coordinates, {
        "geometry_sha256": catalogue.geometry_sha256,
        "catalogue_build_count": cache.build_count,
        "catalogue_cached_bytes": catalogue.cached_nbytes,
        "role_query_counts": counts,
        "target_values_read": False,
    }


def _cost_benchmarks(
    model: WindFarmSharedInteractionModel,
    view: WindFarmNativeView,
    pilot: FixedPilot,
    device: torch.device,
) -> list[dict[str, Any]]:
    layout_counts = np.asarray(view.metadata["n_turbines"], dtype=np.int64)
    validation_rows = np.asarray(pilot.validation_rows, dtype=np.int64)
    low_row = min(validation_rows.tolist(), key=lambda row: (int(layout_counts[row]), int(row)))
    high_row = max(validation_rows.tolist(), key=lambda row: (int(layout_counts[row]), -int(row)))
    selected = []
    for label, row in (("low_M", low_row), ("high_M", high_row)):
        if any(item["source_row"] == int(row) for item in selected):
            continue
        selected.append({"label": label, "source_row": int(row)})
    output: list[dict[str, Any]] = []
    model.eval()
    for item in selected:
        row = int(item["source_row"])
        for query_count in (1024, 8192):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            total_started = time.perf_counter()
            case = view.run(row)
            geometry_started = time.perf_counter()
            coordinates_D, sampler = _sample_role_coordinates_only(
                case,
                query_count=query_count,
                seed=5500 + row + query_count,
            )
            sampling_seconds = time.perf_counter() - geometry_started
            query = _tensor_on_device(coordinates_D[None], device)
            context_started = time.perf_counter()
            prepared = model.prepare_case(case, device=device)
            _synchronize(device)
            context_seconds = time.perf_counter() - context_started
            read_started = time.perf_counter()
            with torch.no_grad():
                prediction = model.predict_standardized(prepared, query, chunk_size=512)
                _ = model.denormalize_tensor(prediction)
            _synchronize(device)
            read_seconds = time.perf_counter() - read_started
            total_seconds = time.perf_counter() - total_started
            output.append({
                "selection": item["label"],
                "source_row": row,
                "case": str(case.case),
                "layout_index": int(case.layout_index),
                "n_turbines_M": int(case.n_turbines),
                "wind_direction_deg": float(case.wind_direction_deg),
                "Q": int(query_count),
                "geometry_and_native_role_sampling_seconds": float(sampling_seconds),
                "context_preparation_seconds": float(context_seconds),
                "receiver_input_transfer_plus_fine_read_seconds": float(read_seconds),
                "complete_cold_seconds": float(total_seconds),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
                "native_role_geometry": sampler,
                "receiver_queries_are_exact_native_cell_centres": True,
                "target_velocities_read": False,
            })
            del prepared, query, case, coordinates_D
    model.train()
    return output


def _gradient_diagnostic(
    model: WindFarmSharedInteractionModel,
    view: WindFarmNativeView,
    pilot: FixedPilot,
    panel: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    row = int(pilot.validation_rows[0])
    case = view.run(row)
    centers = torch.as_tensor(case.module_centers[None].copy(), dtype=torch.float32, device=device).requires_grad_(True)
    receivers = _tensor_on_device(panel["validation_coords_D"][0, :64][None], device).requires_grad_(True)
    prepared = model.prepare_case(case, device=device, centers=centers)
    prediction = model.predict_physical_case(case, prepared, receivers, chunk_size=64)
    scalar = prediction.square().mean()
    center_gradient, receiver_gradient = torch.autograd.grad(scalar, (centers, receivers), allow_unused=False)
    tangent = torch.ones_like(centers) / math.sqrt(max(centers.numel(), 1))
    local = model.linearize_case(prepared, receivers.detach(), tangent, wrt="centers")
    result = {
        "source_row": row,
        "case": str(case.case),
        "layout_index": int(case.layout_index),
        "n_turbines_M": int(case.n_turbines),
        "query_count": int(receivers.shape[1]),
        "geometry_gradient_finite": bool(torch.isfinite(center_gradient).all()),
        "geometry_gradient_l2_norm": float(torch.linalg.vector_norm(center_gradient).detach().cpu()),
        "query_gradient_finite": bool(torch.isfinite(receiver_gradient).all()),
        "query_gradient_l2_norm": float(torch.linalg.vector_norm(receiver_gradient).detach().cpu()),
        "local_AD_linearization_finite": bool(torch.isfinite(local["jvp"]).all()),
        "local_AD_linearization_units": local["units"],
        "local_AD_linearization_semantics": local["semantics"],
        "target_values_read": False,
    }
    model.zero_grad(set_to_none=True)
    return result


def _representative_plane_rows(view: WindFarmNativeView, pilot: FixedPilot) -> list[tuple[str, int]]:
    rows = np.asarray(pilot.validation_rows, dtype=np.int64)
    turbine_counts = np.asarray(view.metadata["n_turbines"], dtype=np.int64)
    directions = np.asarray(view.metadata["wd_deg"], dtype=np.float64)
    candidates = [int(row) for row in rows if float(directions[int(row)]) == 270.0]
    if len(candidates) < 2:
        candidates = rows.astype(int).tolist()
    low = min(candidates, key=lambda row: (int(turbine_counts[row]), row))
    high = max(candidates, key=lambda row: (int(turbine_counts[row]), -row))
    if low == high:
        distinct = [row for row in candidates if row != low]
        if distinct:
            high = max(distinct, key=lambda row: (int(turbine_counts[row]), -row))
    return [("low_M", int(low)), ("high_M", int(high))] if low != high else [("representative", int(low))]


def _render_representative_planes(
    model: WindFarmSharedInteractionModel,
    view: WindFarmNativeView,
    pilot: FixedPilot,
    profile: VerticalProfileBaseline,
    device: torch.device,
    output_dir: Path,
    *,
    fit_updates: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    retained_model = None
    for label, row in _representative_plane_rows(view, pilot):
        case = view.run(row)
        iz = int(np.argmin(np.abs(np.asarray(case.z_m, dtype=np.float64) - float(case.hub_height_m))))
        z_m = float(case.z_m[iz])
        xx_m, yy_m = np.meshgrid(np.asarray(case.x_m), np.asarray(case.y_m), indexing="xy")
        zz_m = np.full_like(xx_m, z_m, dtype=np.float64)
        coordinates_D = np.stack((xx_m, yy_m, zz_m), axis=-1).astype(np.float32) / float(case.diameter_m)
        flat_query = coordinates_D.reshape(1, -1, 3)
        query = _tensor_on_device(flat_query, device)
        candidate_prepared = model.prepare_case(case, device=device)
        with torch.no_grad():
            candidate_std = model.predict_standardized(candidate_prepared, query, chunk_size=512)
            candidate = model.denormalize_tensor(candidate_std)[0].detach().cpu().numpy().reshape(*xx_m.shape, 3)
        native = np.asarray(case.run.U_structured[iz], dtype=np.float32)
        if native.shape != (*xx_m.shape, 3):
            raise ValueError(f"Native full plane has shape {native.shape}, expected {(*xx_m.shape, 3)}.")
        background = profile.predict(np.full(xx_m.shape, z_m / float(case.diameter_m), dtype=np.float32))

        if retained_model is None:
            bootstrap = case_batch(case, coordinates_D.reshape(-1, 3), include_receiver_anchors=False)
            retained_model, retained_checkpoint = load_retained_checkpoint(
                RETAINED_CHECKPOINT,
                device=device,
                materialization_batch=bootstrap,
            )
            if int(retained_checkpoint.get("epoch", retained_checkpoint.get("current_epoch", -1))) != 2475:
                raise ValueError("Retained WindFarm full-plane control is not selected e2475.")
            retained_model.eval()
        baseline_batch = case_batch(case, coordinates_D.reshape(-1, 3), include_receiver_anchors=False).to(device)
        baseline_prepared = retained_model.prepare_case(baseline_batch)
        with torch.no_grad():
            retained = retained_model.predict_physical(
                baseline_prepared,
                baseline_batch.query_xy,
                baseline_batch.query_features,
                receiver_chunk_size=512,
            )[0].detach().cpu().numpy().reshape(*xx_m.shape, 3)
        residual_candidate = candidate - native
        residual_retained = retained - native
        plane_path = output_dir / "arrays" / f"native_plane_{label}_row_{row}.npz"
        plane_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            plane_path,
            x_m=np.asarray(case.x_m),
            y_m=np.asarray(case.y_m),
            z_m=np.asarray([z_m]),
            native_U_mps=native,
            shared_nonlinear_U_mps=candidate,
            shared_nonlinear_residual_mps=residual_candidate,
            Run2103_e2475_U_mps=retained,
            Run2103_e2475_residual_mps=residual_retained,
            TRAIN24_height_only_background_U_mps=background,
        )
        field_arrays = [native[..., 0], candidate[..., 0], retained[..., 0], background[..., 0]]
        common_low = float(np.quantile(np.concatenate([value.reshape(-1) for value in field_arrays]), 0.01))
        common_high = float(np.quantile(np.concatenate([value.reshape(-1) for value in field_arrays]), 0.99))
        residual_scale = float(np.quantile(np.abs(np.concatenate((residual_candidate[..., 0].reshape(-1), residual_retained[..., 0].reshape(-1)))), 0.99))
        residual_scale = max(residual_scale, 1e-6)
        fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), constrained_layout=True)
        field_views = [native[..., 0], candidate[..., 0], residual_candidate[..., 0], retained[..., 0], residual_retained[..., 0], background[..., 0]]
        titles = [
            "Stored OpenFOAM Ux",
            f"Shared nonlinear Ux ({fit_updates} updates)",
            "Shared minus stored Ux",
            "Run2103 e2475 Ux",
            "Run2103 minus stored Ux",
            "TRAIN24 height-only profile Ux",
        ]
        for index, (axis, values, title) in enumerate(zip(axes.flat, field_views, titles)):
            is_residual = index in (2, 4)
            image_obj = axis.pcolormesh(
                xx_m,
                yy_m,
                values,
                shading="auto",
                cmap="coolwarm" if is_residual else "viridis",
                vmin=-residual_scale if is_residual else common_low,
                vmax=residual_scale if is_residual else common_high,
            )
            axis.set_title(title)
            axis.set_xlabel("wind-aligned x (m)")
            axis.set_ylabel("lateral y (m)")
            fig.colorbar(image_obj, ax=axis, shrink=0.78, label="Ux residual (m/s)" if is_residual else "Ux (m/s)")
        fig.suptitle(
            f"{case.case} | M={case.n_turbines} | wind direction={case.wind_direction_deg:.0f}° | z={z_m:.3f} m"
        )
        figure_path = output_dir / "figures" / f"native_plane_{label}_row_{row}.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=140)
        fig.savefig(figure_path.with_suffix(".pdf"))
        plt.close(fig)
        component = lambda value: np.sqrt(np.mean(np.square(np.asarray(value, dtype=np.float64)), axis=(0, 1))).tolist()
        record = {
            "selection": label,
            "source_row": int(row),
            "case": str(case.case),
            "layout_index": int(case.layout_index),
            "n_turbines_M": int(case.n_turbines),
            "wind_direction_deg": float(case.wind_direction_deg),
            "native_plane_z_m": z_m,
            "shape_yx": list(native.shape[:2]),
            "plane_scope": "complete native hub-height plane; stored OpenFOAM cell centres",
            f"shared_{fit_updates}_update_component_rmse_mps": component(residual_candidate),
            "Run2103_e2475_component_rmse_mps": component(residual_retained),
            "shared_minus_stored_Ux_display_scale_mps": [common_low, common_high],
            "signed_residual_Ux_display_limit_mps": residual_scale,
            "display_clipping": "field panels use joint 1st/99th percentiles; residual panels use joint 99th absolute percentile; metrics use unclipped arrays",
            "figure": str(figure_path),
            "arrays": str(plane_path),
            "native_target_read_for_evaluation_only": True,
        }
        output.append(record)
        del baseline_prepared, baseline_batch, candidate_prepared, query, case
        gc.collect()
    return output


def _final_diagnostics(
    model: WindFarmSharedInteractionModel,
    view: WindFarmNativeView,
    pilot: FixedPilot,
    normalizer: VelocityNormalizer,
    profile: VerticalProfileBaseline,
    panel: dict[str, np.ndarray],
    controls: dict[str, Any],
    role_scales: dict[str, float],
    device: torch.device,
    output_dir: Path,
    *,
    fit_updates: int,
) -> dict[str, Any]:
    runtime_trace = _runtime_shared_trace(model, view, pilot, panel, device)
    gradient = _gradient_diagnostic(model, view, pilot, panel, device)
    costs = _cost_benchmarks(model, view, pilot, device)
    planes = _render_representative_planes(
        model, view, pilot, profile, device, output_dir, fit_updates=fit_updates
    )
    result = {
        "fit_age_updates": int(fit_updates),
        "subset_id": SUBSET_ID,
        "train_layout_count": 24,
        "validation_layout_count": 8,
        "all_directions_per_selected_layout": [270.0, 285.0, 300.0],
        "case_visits": _case_visits(
            panel["train_panel_rows"],
            updates=fit_updates,
            layout_by_row=np.asarray(view.metadata["layout_index"], dtype=np.int64),
        ),
        "role_loss_scales_mps": role_scales,
        "normalization": normalizer.to_dict(),
        "background_control_interpretation": "TRAIN24-fitted height-only population profile including average wake effects; an input-defined diagnostic, not prescribed physical inlet truth",
        "controls": controls,
        "shared_runtime_trace": runtime_trace,
        "geometry_query_gradient_and_local_AD": gradient,
        "complete_native_cost_benchmarks": costs,
        "representative_full_planes": planes,
        "cost_and_plane_query_targets_used_only_for_measurement": True,
        "new_solver_calls": 0,
        "test_targets_read": False,
        "gpu_associated_training_cap_seconds": PILOT_GPU_CAP_SECONDS,
    }
    _json_write(output_dir / "final_diagnostics.json", result)
    _json_write(output_dir / "shared_code_receipt.json", _shared_code_receipt(runtime_trace))
    return result


def train_pilot(
    output_dir: Path,
    *,
    updates: int,
    stop_after_updates: int,
    device_arg: str,
    physical_gpu: int | None,
    resume: bool,
) -> dict[str, Any]:
    if updates not in (DEFAULT_UPDATES, MAX_UPDATES):
        raise ValueError("WindFarm training supports only the bounded 1000-update horizon or root-authorized 2000 extension.")
    if stop_after_updates < 1 or stop_after_updates > updates or stop_after_updates % 100:
        raise ValueError("WindFarm pilot stop points must be positive 100-update review boundaries within the horizon.")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_arg)
    gpu_receipt = _visible_physical_gpu(device, physical_gpu)
    _set_seed(DEFAULT_SEED, device)
    torch.backends.cudnn.benchmark = False
    view, pilot, normalizer, profile, panel, panel_metadata, role_scales = _load_pilot_artifacts(
        output_dir, requested_updates=updates
    )
    normalization_path = output_dir / "train_only_normalization.json"
    subset_hash = str(pilot.manifest["manifest_sha256"])
    panel_hash = str(panel_metadata["prefix_sha256_by_update"][str(int(updates))])
    normalization_hash = _sha256(normalization_path)
    model = build_windfarm_model(_model_recipe(), velocity_transform=normalizer)
    if not isinstance(model, WindFarmSharedInteractionModel):
        raise TypeError("Opt-in native Wind family factory did not return the shared nonlinear interaction adapter.")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-5)

    start_update = 0
    optimizer_updates: list[float] = []
    history: list[dict[str, Any]] = []
    best_update = 0
    best_validation_objective = float("inf")
    controls: dict[str, Any] | None = None
    previous_gpu_seconds = 0.0
    ledger_path = output_dir / "gpu_process_ledger.json"
    if ledger_path.is_file():
        previous_gpu_seconds = float(
            json.loads(ledger_path.read_text(encoding="utf-8")).get("cumulative_gpu_associated_seconds", 0.0)
        )
    latest_path = output_dir / "checkpoints" / "latest.pt"
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError("--resume was requested but there is no durable latest.pt checkpoint.")
        (
            start_update,
            optimizer_updates,
            history,
            best_update,
            best_validation_objective,
            controls,
        ) = _load_resume(
            latest_path,
            model=model,
            optimizer=optimizer,
            device=device,
            subset_manifest_sha256=subset_hash,
            normalization_sha256=normalization_hash,
            panel=panel,
        )
    elif latest_path.exists():
        raise FileExistsError("Pilot checkpoint already exists; use --resume to continue its exact bound state.")
    if start_update >= stop_after_updates:
        raise ValueError(f"Checkpoint is at update {start_update}, not before requested stop {stop_after_updates}.")

    train_rows = panel["train_panel_rows"]
    layout_by_row = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    train_coords = panel["train_panel_coords_D"]
    train_targets = panel["train_panel_target_mps"]
    _role_slices()
    interval_losses: list[float] = []
    candidate_validation_seconds: list[float] = []
    control_seconds = 0.0
    diagnostic_seconds = 0.0
    final_diagnostics: dict[str, Any] | None = None
    failure: str | None = None
    try:
        for update_index in range(start_update, stop_after_updates):
            update_started = time.perf_counter()
            row = int(train_rows[update_index])
            case = view.run(row)
            query = _tensor_on_device(train_coords[update_index][None], device)
            target = _tensor_on_device(train_targets[update_index][None], device)
            prepared = model.prepare_case(case, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, _prediction = _training_loss(model, prepared, query, target, role_scales)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite WindFarm training loss at update {update_index + 1}.")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(f"Nonfinite WindFarm gradient norm at update {update_index + 1}.")
            optimizer.step()
            _synchronize(device)
            optimizer_updates.append(time.perf_counter() - update_started)
            interval_losses.append(float(loss.detach().cpu()))
            if (update_index + 1) % 100 == 0:
                review_update = update_index + 1
                evaluation_started = time.perf_counter()
                candidate_validation = _eval_candidate(
                    model, view, panel, pilot, role_scales, device, updates=review_update
                )
                _synchronize(device)
                validation_seconds = time.perf_counter() - evaluation_started
                candidate_validation_seconds.append(validation_seconds)
                if controls is None:
                    control_started = time.perf_counter()
                    background, retained = _eval_controls(
                        view,
                        panel,
                        pilot,
                        role_scales,
                        profile,
                        RETAINED_CHECKPOINT,
                        device,
                    )
                    controls = {
                        "background_validation": background,
                        "retained_Run2103_e2475_validation": retained,
                        "retained_checkpoint_path": str(RETAINED_CHECKPOINT),
                        "retained_checkpoint_sha256": _sha256(RETAINED_CHECKPOINT),
                        "retained_epoch": 2475,
                        "same_validation_rows_and_role_panels": True,
                        "history_is_unequal": True,
                    }
                    control_seconds = time.perf_counter() - control_started
                    _save_metric_csv(output_dir / "metrics" / "control_validation_rows.csv", [background, retained])
                train_mean = float(np.mean(interval_losses))
                elapsed = time.perf_counter() - _PROCESS_STARTED_PERF
                record = {
                    "update_count": int(review_update),
                    "training_objective_mean": train_mean,
                    "training_objective_updates": len(interval_losses),
                    "candidate_validation": candidate_validation,
                    "candidate_validation_seconds": float(validation_seconds),
                    "control_evaluation_seconds_first_review": float(control_seconds if review_update == 100 else 0.0),
                    "optimizer_update_seconds_mean_last_100": float(np.mean(optimizer_updates[-100:])),
                    "optimizer_update_seconds_p95_last_100": float(np.quantile(optimizer_updates[-100:], 0.95)),
                    "outer_process_elapsed_seconds": float(elapsed),
                    "case_visits": _case_visits(train_rows, updates=review_update, layout_by_row=layout_by_row),
                }
                if len(optimizer_updates) >= 10 and review_update < updates:
                    first10_mean = float(np.mean(optimizer_updates[:10]))
                    observed_mean = float(np.mean(optimizer_updates[-100:]))
                    update_budget = max(first10_mean, observed_mean) * 1.30
                    eval_budget = max(candidate_validation_seconds, default=0.0) * math.ceil((updates - review_update) / 100)
                    reserve = 8 * 60.0
                    projected_total = previous_gpu_seconds + elapsed + (updates - review_update) * update_budget + eval_budget + reserve
                    record["first10_actual_update_seconds_mean"] = first10_mean
                    record["projected_total_gpu_associated_seconds_to_requested_horizon_including_reserve"] = projected_total
                    record["projected_budget_cap_seconds"] = PILOT_GPU_CAP_SECONDS
                    record["projection_has_30pct_update_margin_and_8min_diagnostics_reserve"] = True
                history.append(record)
                validation_objective = float(candidate_validation["equal_case_role_scaled_objective"])
                if validation_objective < best_validation_objective:
                    best_validation_objective = validation_objective
                    best_update = review_update
                _save_metric_csv(
                    output_dir / "metrics" / "candidate_validation_rows.csv",
                    [item["candidate_validation"] for item in history],
                )
                _save_json_metric(output_dir, review_update, {
                    "candidate": candidate_validation,
                    "controls": controls,
                    "case_visits": record["case_visits"],
                    "role_loss_scales_mps": role_scales,
                    "review_update": review_update,
                })
                _write_history(output_dir, history)
                checkpoint_path = _save_training_checkpoint(
                    output_dir,
                    update_count=review_update,
                    model=model,
                    optimizer=optimizer,
                    optimizer_updates=optimizer_updates,
                    history=history,
                    best_update=best_update,
                    best_validation_objective=best_validation_objective,
                    subset_manifest_sha256=subset_hash,
                    panel_prefix_sha256=panel_hash,
                    normalization_sha256=normalization_hash,
                    controls=controls,
                )
                interval_losses = []
                _json_write(output_dir / "metrics" / "training_history.json", {"records": history})
                _json_write(output_dir / "last_review.json", {
                    "update_count": review_update,
                    "candidate": candidate_validation,
                    "controls": controls,
                    "checkpoint": str(checkpoint_path),
                    "best_update": best_update,
                    "best_validation_objective": best_validation_objective,
                    "optimizer_update_seconds_first10_mean": float(np.mean(optimizer_updates[:10])) if optimizer_updates else None,
                    "optimizer_update_seconds_all_mean": float(np.mean(optimizer_updates)),
                    "candidate_validation_seconds": float(validation_seconds),
                    "control_evaluation_seconds": control_seconds,
                    "outer_process_elapsed_seconds": elapsed,
                    "projected_gpu_associated_seconds_to_requested_horizon": record.get("projected_total_gpu_associated_seconds_to_requested_horizon_including_reserve"),
                    "gpu": gpu_receipt,
                })
                print(
                    f"[wind-pilot] update={review_update} train_obj={train_mean:.6g} "
                    f"val_obj={validation_objective:.6g} best={best_update} "
                    f"elapsed={elapsed:.1f}s checkpoint={checkpoint_path}",
                    flush=True,
                )
                if review_update < stop_after_updates and projected_total > PILOT_GPU_CAP_SECONDS:
                    raise TimeoutError(
                        f"Measured Wind pilot forecast {projected_total:.1f}s exceeds its 90-minute GPU-associated cap."
                    )
        if interval_losses:
            raise RuntimeError("WindFarm pilot stopped between 100-update checkpoint boundaries.")
        if stop_after_updates == updates:
            if controls is None:
                raise RuntimeError("Final WindFarm pilot horizon has no matched validation controls.")
            diagnostic_started = time.perf_counter()
            model.eval()
            final_diagnostics = _final_diagnostics(
                model,
                view,
                pilot,
                normalizer,
                profile,
                panel,
                controls,
                role_scales,
                device,
                output_dir,
                fit_updates=stop_after_updates,
            )
            _synchronize(device)
            diagnostic_seconds = time.perf_counter() - diagnostic_started
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        ended_utc = datetime.now(timezone.utc).isoformat()
        elapsed_outer = float(time.perf_counter() - _PROCESS_STARTED_PERF)
        ledger = _training_process_ledger(output_dir, {
            "process_started_utc": _PROCESS_STARTED_UTC,
            "process_ended_utc": ended_utc,
            "outer_process_elapsed_seconds": elapsed_outer,
            "gpu_associated": bool(gpu_receipt["gpu_associated"]),
            "gpu": gpu_receipt,
            "requested_horizon_updates": int(updates),
            "stop_after_updates": int(stop_after_updates),
            "resume": bool(resume),
            "last_durable_update": len(optimizer_updates),
            "imports_seconds_included": float(IMPORT_SECONDS),
            "candidate_validation_seconds_included": float(sum(candidate_validation_seconds)),
            "first_control_evaluation_seconds_included": float(control_seconds),
            "final_physical_and_cost_diagnostics_seconds_included": float(diagnostic_seconds),
            "failure": failure,
        })
        runtime_trace = None if final_diagnostics is None else final_diagnostics.get("shared_runtime_trace")
        _json_write(output_dir / "shared_code_receipt.json", _shared_code_receipt(runtime_trace))
        _json_write(output_dir / "process_completion.json", {
            "process_started_utc": _PROCESS_STARTED_UTC,
            "process_ended_utc": ended_utc,
            "outer_process_elapsed_seconds": elapsed_outer,
            "module_import_seconds": float(IMPORT_SECONDS),
            "gpu_associated_cumulative_seconds": ledger["cumulative_gpu_associated_seconds"],
            "gpu_associated_cap_seconds": PILOT_GPU_CAP_SECONDS,
            "last_durable_update": len(optimizer_updates),
            "controls_saved": controls is not None,
            "status": "completed" if failure is None and len(optimizer_updates) >= stop_after_updates else "interrupted_or_failed",
        })
    return {
        "status": "review_boundary_completed",
        "updates": int(stop_after_updates),
        "checkpoint": str(latest_path),
        "cumulative_gpu_associated_seconds": float(ledger["cumulative_gpu_associated_seconds"]),
        "final_diagnostics": final_diagnostics,
        "gpu": gpu_receipt,
    }


def _record_top_level_failure(output_dir: Path, error: BaseException, *, device_arg: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "gpu_process_ledger.json"
    records: list[dict[str, Any]] = []
    if ledger_path.is_file():
        records = json.loads(ledger_path.read_text(encoding="utf-8")).get("processes", [])
    message = f"{type(error).__name__}: {error}"
    matching = next(
        (item for item in reversed(records) if item.get("process_started_utc") == _PROCESS_STARTED_UTC),
        None,
    )
    now_utc = datetime.now(timezone.utc).isoformat()
    elapsed = float(time.perf_counter() - _PROCESS_STARTED_PERF)
    if matching is None:
        records.append({
            "process_started_utc": _PROCESS_STARTED_UTC,
            "process_ended_utc": now_utc,
            "outer_process_elapsed_seconds": elapsed,
            "gpu_associated": str(device_arg).startswith("cuda"),
            "gpu_requested": str(device_arg),
            "imports_seconds_included": float(IMPORT_SECONDS),
            "failure": message,
            "last_durable_update": None,
        })
    else:
        matching["process_ended_utc"] = now_utc
        matching["failure"] = message
    cumulative = float(sum(
        float(item["outer_process_elapsed_seconds"])
        for item in records if item.get("gpu_associated")
    ))
    _json_write(ledger_path, {
        "schema_version": SCHEMA_VERSION,
        "budget_cap_gpu_associated_seconds": PILOT_GPU_CAP_SECONDS,
        "processes": records,
        "cumulative_gpu_associated_seconds": cumulative,
    })
    _json_write(output_dir / "process_completion.json", {
        "process_started_utc": _PROCESS_STARTED_UTC,
        "process_ended_utc": now_utc,
        "outer_process_elapsed_seconds": elapsed,
        "module_import_seconds": float(IMPORT_SECONDS),
        "gpu_associated_cumulative_seconds": cumulative,
        "gpu_associated_cap_seconds": PILOT_GPU_CAP_SECONDS,
        "status": "interrupted_or_failed",
        "failure": message,
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ignored generated-artifact directory for the frozen native pilot.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Freeze input-only subset, TRAIN normalization and native panels on CPU.")
    prepare_parser.add_argument("--updates", type=int, choices=(DEFAULT_UPDATES, MAX_UPDATES), default=DEFAULT_UPDATES)
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser = subparsers.add_parser("train", help="Run the bounded real native nonlinear Wind pilot.")
    train_parser.add_argument("--updates", type=int, choices=(DEFAULT_UPDATES, MAX_UPDATES), default=DEFAULT_UPDATES)
    train_parser.add_argument("--stop-after-updates", type=int, default=100)
    train_parser.add_argument("--device", type=str, default="cuda:0")
    train_parser.add_argument("--physical-gpu", type=int, default=None)
    train_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_pilot(args.output_dir, updates=args.updates, seed=args.seed)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 0
    try:
        result = train_pilot(
            args.output_dir,
            updates=args.updates,
            stop_after_updates=args.stop_after_updates,
            device_arg=args.device,
            physical_gpu=args.physical_gpu,
            resume=args.resume,
        )
    except BaseException as error:
        _record_top_level_failure(args.output_dir, error, device_arg=args.device)
        raise
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
