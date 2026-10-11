"""Wind TaskProvider for fresh joint regional-hypergraph field models.

The provider retains the fixed native five-role sampler, component-balanced
TRAIN calibration, rotor-diameter receiver frame, and profile-residual output
law. Only target arrays are attached after target-free scene construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_forward_core.interface_fields.interaction_core import DependencySpec, InteractionScene
from honf_runtime.unified_training import (
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    TaskBatch,
)
from honf_runtime.unified_training import (
    _sampling_dataset_id as _runtime_sampling_dataset_id,
)
from torch import nn

from ..data import WindFarmNativeView
from ..interaction_preserving import INTERACTION_PRESERVING_WIND_MODES, InteractionPreservingWindModel
from ..joint_regional import WindFarmJointRegionalModel
from ..normalization import (
    VelocityNormalizer,
    VerticalProfileBaseline,
    fit_velocity_statistics,
    read_normalization_json,
)
from ..splits import GroupSplit
from ..workflows.joint_forward import (
    DEFAULT_ROLE_CATALOGUE_CACHE_MAX_BYTES,
    ROLE_NAMES,
    NativeRoleCatalogueCache,
    sample_native_role_queries,
)
from ..workflows.native_role_cache import training_catalogue_cache_directory
from .unified_task import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DERIVED_ROOT,
    DEFAULT_PILOT_ROOT,
    _fit_train_role_scales,
    _indices_sha256,
    _load_original_split,
    _physical_case_direction_id,
    _read_fixed_rows,
    _stable_json_sha256,
)

JOINT_WIND_ENVIRONMENT_SHAPE = (4, 4, 4)
JOINT_WIND_ENVIRONMENT_COUNT = 64
JOINT_WIND_PROFILE_ID = "wind_shared_fixed24_v1"
JOINT_WIND_FORMAL_PROFILE_ID = "wind_formal_fulltrain_v1"
JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID = "wind_original420_train_fullVALID90_followup1000_v1"
FULL_TRAIN_FOLLOWUP_PROTOCOL = "interaction_preserving_full_train_followup1000_v1"
FULL_TRAIN_FOLLOWUP_DATASET = "wind_original420_train_fullVALID90_followup_v1"
FULL_TRAIN_FOLLOWUP_TRAIN_SHA256 = "a2170bb15349462160679632fc074d7f685d7e5d17d09aeeab1797a824c8a282"
FULL_TRAIN_FOLLOWUP_VALIDATION_SHA256 = "ed7295dd2650687c599c13e29a6bae3492508ec8346517e39f4cacf60d0eff72"
FULL_TRAIN_FOLLOWUP_TEST_SHA256 = "e11734b05e51691680fcae088ef07be8608951660eab9761d05dc29fe2fef998"
JOINT_WIND_ROLE_CALIBRATION_QUERY_COUNT = 1024
JOINT_WIND_ROLE_SCALE_RULE = (
    "TRAIN-only per-role component RMS of the residual from the fitted height profile, "
    "with the existing normalizer floor and 0.10 scalar-role floor"
)
JOINT_WIND_ROLE_SCALE_CACHE_SCHEMA = 1
JOINT_WIND_ROLE_SCALE_CALIBRATION_ID = "wind_joint_e64_native_role_scales_v1"
JOINT_WIND_BASELINE_FORMAL_SAMPLING_PROTOCOL = "baseline_formal_v1"
JOINT_WIND_BASELINE_TRAIN_STREAM = "wind_native_role_query"
JOINT_WIND_BASELINE_VALIDATION_STREAM = "wind_validation_role_query"
JOINT_WIND_BASELINE_CALIBRATION_STREAM = "wind_role_scale_calibration"
JOINT_WIND_BASELINE_TRAIN_ROW_SHA256_KEY = "train_rows_sha256"
DEFAULT_JOINT_WIND_ROLE_SCALE_CACHE_DIR = (
    Path(__file__).resolve().parents[3]
    / "diagnostics/generated/joint_regional_20261009/audit/train_role_scale_cache"
)
WIND_JOINT_DEPENDENCY = DependencySpec(
    dataset="WindFarm",
    output_law="nonlinear",
    configuration_inputs=(
        "turbine_geometry",
        "wind_direction",
        "native_domain_support",
        "environment_geometry_and_measure",
        "reference_inflow",
    ),
    applicable_controls=(),
    output_roles=("Ux", "Uy", "Uz"),
    units=("m/s", "m/s", "m/s"),
    edges=(
        ("turbine_geometry", "source_nodes_and_registered_edges"),
        ("environment_geometry_and_measure", "environment_nodes_and_registered_edges"),
        ("wind_direction", "shared_scene_context"),
        ("receiver_context", "regional_edge_read"),
        ("regional_edge_read", "nonlinear_velocity_residual"),
    ),
    prepared_nodes=("source_nodes", "environment_nodes", "regional_edges", "global_state"),
)

_ROLE_INDEX = {name: index for index, name in enumerate(ROLE_NAMES)}


def validate_full_train_followup_membership(
    train_rows: Sequence[int], validation_rows: Sequence[int], test_rows: Sequence[int], *,
    train_layout_count: int, validation_layout_count: int, test_layout_count: int,
) -> dict[str, Any]:
    """Validate the original 420/90/90 split from row/layout metadata only."""
    partitions = [np.asarray(rows, dtype=np.int64) for rows in (train_rows, validation_rows, test_rows)]
    expected = ((420, 140, FULL_TRAIN_FOLLOWUP_TRAIN_SHA256),
                (90, 30, FULL_TRAIN_FOLLOWUP_VALIDATION_SHA256),
                (90, 30, FULL_TRAIN_FOLLOWUP_TEST_SHA256))
    for name, rows, (row_count, layout_count, expected_sha), actual_layout_count in zip(
            ("TRAIN", "validation", "TEST"), partitions, expected,
            (train_layout_count, validation_layout_count, test_layout_count)):
        if rows.ndim != 1 or rows.size != row_count or not np.array_equal(rows, np.sort(rows)):
            raise ValueError(f"Wind full follow-up {name} rows must be sorted and contain exactly {row_count} rows.")
        if int(actual_layout_count) != layout_count:
            raise ValueError(f"Wind full follow-up {name} must preserve exactly {layout_count} layouts.")
        if _indices_sha256(rows) != expected_sha:
            raise ValueError(f"Wind full follow-up {name} membership differs from the sealed seed-42 split.")
    combined = np.concatenate(partitions)
    if combined.size != 600 or np.unique(combined).size != 600:
        raise ValueError("Wind full follow-up TRAIN/VALID/TEST row memberships must be disjoint and exhaustive.")
    if not np.array_equal(np.sort(combined), np.arange(600, dtype=np.int64)):
        raise ValueError("Wind full follow-up partitions must cover the original 600 canonical rows exactly.")
    return {
        "train_row_count": int(partitions[0].size),
        "train_row_indices_sha256": _indices_sha256(partitions[0]),
        "validation_row_count": int(partitions[1].size),
        "validation_row_indices_sha256": _indices_sha256(partitions[1]),
        "test_row_count": int(partitions[2].size),
        "test_row_indices_sha256": _indices_sha256(partitions[2]),
        "test_target_values_read": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_file_identity(path: Path) -> Mapping[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _wind_dataset_fingerprint(view: WindFarmNativeView) -> str:
    """Bind scale reuse to the exact read-only native volume and compact geometry."""

    volume_root = view.volume.volume_root
    required_arrays = (
        "U.npy",
        "run_cell_offsets.npy",
        "run_shape.npy",
        "run_x_offsets.npy",
        "run_y_offsets.npy",
        "run_z_offsets.npy",
        "x_cell_m.npy",
        "y_cell_m.npy",
        "z_cell_m.npy",
        "case.npy",
        "layout_index.npy",
        "wd_deg.npy",
        "completed.npy",
    )
    file_identity = {
        name: _source_file_identity(volume_root / name) for name in required_arrays
    }
    compact_geometry: dict[str, Any] = {}
    for name in sorted(view.metadata):
        value = np.ascontiguousarray(np.asarray(view.metadata[name]))
        if value.dtype.hasobject:
            raise ValueError(f"Wind dataset metadata {name!r} cannot have object dtype.")
        compact_geometry[name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
        }
    payload = {
        "schema_version": 1,
        "volume_root": str(volume_root.resolve()),
        "volume_arrays": file_identity,
        "compact_geometry": compact_geometry,
        "native_frame": "rotor_diameters",
        "velocity_units": "m/s",
    }
    return _stable_json_sha256(payload)


def _role_scale_cache_key(
    *,
    view: WindFarmNativeView,
    manifest: Mapping[str, Any],
    train_rows: np.ndarray,
    train_layouts: Sequence[int],
    profile_id: str,
    training_fingerprint: str,
    normalizer: Any,
    profile: Any,
    native_sampling_protocol: str | None = None,
) -> dict[str, Any]:
    rows = np.asarray(train_rows, dtype=np.int64)
    layout_by_row = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    calibration_layouts = tuple(int(value) for value in train_layouts[:4])
    calibration_rows = tuple(
        int(row) for row in rows if int(layout_by_row[int(row)]) in set(calibration_layouts)
    )
    if len(calibration_layouts) != 4 or len(calibration_rows) != 12:
        raise ValueError("Wind role-scale cache identity requires the first four selected TRAIN layouts and 12 rows.")
    payload = {
        "schema_version": JOINT_WIND_ROLE_SCALE_CACHE_SCHEMA,
        "calibration_id": JOINT_WIND_ROLE_SCALE_CALIBRATION_ID,
        "dataset_profile": str(profile_id),
        "dataset_fingerprint": _wind_dataset_fingerprint(view),
        "subset_manifest_sha256": str(manifest.get("manifest_sha256", training_fingerprint)),
        "training_fingerprint": str(training_fingerprint),
        "train_row_indices_sha256": _indices_sha256(rows),
        "train_row_indices": rows.astype(int).tolist(),
        "train_layout_indices": [int(value) for value in train_layouts],
        "calibration_layout_indices": list(calibration_layouts),
        "calibration_row_indices": list(calibration_rows),
        "normalizer_sha256": _stable_json_sha256(normalizer.to_dict()),
        "background_profile_sha256": _stable_json_sha256(profile.to_dict()),
        "input_view": {
            "scene_builder": "wind_joint_target_free_scene_v1",
            "environment_shape": list(JOINT_WIND_ENVIRONMENT_SHAPE),
            "environment_records": JOINT_WIND_ENVIRONMENT_COUNT,
            "environment_channels": 7,
            "environment_geometry_only": True,
            "environment_measure": "native_support_volume_D3_per_record",
            "receiver_frame": "rotor_diameters",
            "receiver_roles": list(ROLE_NAMES),
        },
        "calibration_queries": {
            "method": "sample_native_role_queries",
            "query_count_per_case": JOINT_WIND_ROLE_CALIBRATION_QUERY_COUNT,
            "role_counts_per_case": {name: int(value) for name, value in _role_counts_for_query_count(1024).items()},
            "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            "seed": 42,
            "stream": "wind_role_scale_calibration",
            "target_partition": "selected TRAIN calibration rows only",
            "target_values_from_windtest": False,
        },
    }
    if native_sampling_protocol is not None:
        if native_sampling_protocol != JOINT_WIND_BASELINE_FORMAL_SAMPLING_PROTOCOL:
            raise ValueError(f"Unsupported Wind native sampling protocol {native_sampling_protocol!r}.")
        payload["native_sampling_protocol"] = native_sampling_protocol
        payload["sampling_dataset_id"] = _sampling_dataset_id(
            profile_id, str(training_fingerprint), native_sampling_protocol
        )
        payload["calibration_queries"].update({
            "stream": JOINT_WIND_BASELINE_CALIBRATION_STREAM,
            "dataset_key_source": JOINT_WIND_BASELINE_TRAIN_ROW_SHA256_KEY,
            "role_key_coordinates": ["physical_case_direction_id", "role_name"],
        })
    return payload


@contextmanager
def _role_scale_cache_lock(cache_file: Path):
    """Serialize per-key cache fills across concurrently starting mode processes."""

    import fcntl

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_file.with_suffix(cache_file.suffix + ".lock")
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def _calibration_cache_payload(cache_key: Mapping[str, Any], calibration: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "schema_version": JOINT_WIND_ROLE_SCALE_CACHE_SCHEMA,
        "cache_key": dict(cache_key),
        "cache_key_sha256": _stable_json_sha256(cache_key),
        "calibration": dict(calibration),
        "calibration_sha256": _stable_json_sha256(calibration),
    }
    return {**core, "payload_sha256": _stable_json_sha256(core)}


def _read_role_scale_cache(cache_file: Path, cache_key: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if not cache_file.exists():
        return None
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    expected_fields = {"schema_version", "cache_key", "cache_key_sha256", "calibration", "calibration_sha256", "payload_sha256"}
    if set(payload) != expected_fields:
        raise ValueError(f"Wind role-scale cache has an unexpected field inventory: {cache_file}")
    core = {name: payload[name] for name in expected_fields if name != "payload_sha256"}
    if (
        int(payload["schema_version"]) != JOINT_WIND_ROLE_SCALE_CACHE_SCHEMA
        or payload["payload_sha256"] != _stable_json_sha256(core)
        or payload["cache_key_sha256"] != _stable_json_sha256(payload["cache_key"])
        or payload["calibration_sha256"] != _stable_json_sha256(payload["calibration"])
    ):
        raise ValueError(f"Wind role-scale cache checksum/schema validation failed: {cache_file}")
    if payload["cache_key"] != dict(cache_key):
        return None
    calibration = payload["calibration"]
    if (
        cache_key.get("calibration_id") != JOINT_WIND_ROLE_SCALE_CALIBRATION_ID
        or cache_key.get("calibration_queries", {}).get("target_values_from_windtest") is not False
        or cache_key.get("calibration_queries", {}).get("target_partition") != "selected TRAIN calibration rows only"
        or calibration.get("target_values_read") != "TRAIN calibration panel only"
        or calibration.get("wind_test_target_values_read", False) is not False
        or calibration.get("training_rows_sha256") != cache_key["train_row_indices_sha256"]
        or calibration.get("training_fingerprint") != cache_key["training_fingerprint"]
        or calibration.get("background_profile_sha256") != cache_key["background_profile_sha256"]
        or calibration.get("calibration_layout_indices") != cache_key["calibration_layout_indices"]
        or calibration.get("calibration_row_indices") != cache_key["calibration_row_indices"]
        or calibration.get("calibration_query_count_per_row") != JOINT_WIND_ROLE_CALIBRATION_QUERY_COUNT
        or calibration.get("target_values_read") != "TRAIN calibration panel only"
        or int(calibration.get("solver_attempts", -1)) != 0
    ):
        raise ValueError(f"Wind role-scale cache source/partition binding failed: {cache_file}")
    role_scales = calibration.get("component_role_scales_mps")
    if not isinstance(role_scales, Mapping) or set(role_scales) != set(ROLE_NAMES):
        raise ValueError(f"Wind role-scale cache omits one or more native roles: {cache_file}")
    for role in ROLE_NAMES:
        values = np.asarray(role_scales[role], dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"Wind role-scale cache contains invalid {role!r} component scales.")
    return calibration


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _transform_cache_payload(cache_key: Mapping[str, Any], normalizer: Any, profile: Any) -> dict[str, Any]:
    core = {
        "schema_version": JOINT_WIND_ROLE_SCALE_CACHE_SCHEMA,
        "cache_key": dict(cache_key),
        "cache_key_sha256": _stable_json_sha256(cache_key),
        "normalizer": normalizer.to_dict(),
        "background_profile": profile.to_dict(),
    }
    return {**core, "payload_sha256": _stable_json_sha256(core)}


def _read_transform_cache(
    cache_file: Path,
    cache_key: Mapping[str, Any],
) -> tuple[VelocityNormalizer, VerticalProfileBaseline] | None:
    if not cache_file.exists():
        return None
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version", "cache_key", "cache_key_sha256", "normalizer", "background_profile", "payload_sha256"
    }
    if set(payload) != expected_fields:
        raise ValueError(f"Wind TRAIN-transform cache has an unexpected field inventory: {cache_file}")
    core = {name: payload[name] for name in expected_fields if name != "payload_sha256"}
    if (
        int(payload["schema_version"]) != JOINT_WIND_ROLE_SCALE_CACHE_SCHEMA
        or payload["payload_sha256"] != _stable_json_sha256(core)
        or payload["cache_key_sha256"] != _stable_json_sha256(payload["cache_key"])
    ):
        raise ValueError(f"Wind TRAIN-transform cache checksum/schema validation failed: {cache_file}")
    if payload["cache_key"] != dict(cache_key):
        return None
    if (
        cache_key.get("dataset_profile") not in (JOINT_WIND_FORMAL_PROFILE_ID, JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID)
        or cache_key.get("wind_test_target_values_read") is not False
        or cache_key.get("target_partition_access") != "all selected original TRAIN rows only"
        or int(cache_key.get("solver_attempts", -1)) != 0
        or _indices_sha256(np.asarray(cache_key.get("train_row_indices", ()), dtype=np.int64))
        != cache_key.get("train_row_indices_sha256")
    ):
        raise ValueError(f"Wind TRAIN-transform cache has an invalid partition/source binding: {cache_file}")
    normalizer = VelocityNormalizer.from_dict(dict(payload["normalizer"]))
    profile = VerticalProfileBaseline.from_dict(dict(payload["background_profile"]))
    if (
        normalizer.source_rows != len(cache_key["train_row_indices"])
        or normalizer.sample_count_per_row != int(cache_key["samples_per_row"])
        or normalizer.seed != int(cache_key["seed"])
        or profile.bin_centers_D.size != int(cache_key["profile_bins"])
        or profile.values_mps.shape != (int(cache_key["profile_bins"]), 3)
    ):
        raise ValueError(f"Wind TRAIN-transform cache does not match its selected input-only fit recipe: {cache_file}")
    return normalizer, profile


def _fit_or_load_formal_transforms(
    *,
    view: WindFarmNativeView,
    train_rows: np.ndarray,
    train_layouts: Sequence[int],
    manifest: Mapping[str, Any],
    training_fingerprint: str,
    seed: int,
    cache_dir: Path,
    profile_id: str = JOINT_WIND_FORMAL_PROFILE_ID,
) -> tuple[VelocityNormalizer, VerticalProfileBaseline, Mapping[str, Any]]:
    rows = np.asarray(train_rows, dtype=np.int64)
    key = {
        "schema_version": JOINT_WIND_ROLE_SCALE_CACHE_SCHEMA,
        "transform_fit_id": ("wind_full_followup1000_original_train_equal_count_volume_fit_v1"
                             if profile_id == JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID else
                             "wind_full_original_train_equal_count_volume_fit_v1"),
        "dataset_profile": str(profile_id),
        "dataset_fingerprint": _wind_dataset_fingerprint(view),
        "subset_manifest_sha256": str(manifest["manifest_sha256"]),
        "training_fingerprint": str(training_fingerprint),
        "train_row_indices_sha256": _indices_sha256(rows),
        "train_row_indices": rows.astype(int).tolist(),
        "train_layout_indices": [int(value) for value in train_layouts],
        "samples_per_row": 8192,
        "profile_bins": 32,
        "seed": int(seed),
        "sampling": "native volume points; equal samples per selected original TRAIN row",
        "input_view": {
            "environment_shape": list(JOINT_WIND_ENVIRONMENT_SHAPE),
            "environment_records": JOINT_WIND_ENVIRONMENT_COUNT,
            "geometry_only": True,
            "receiver_frame": "rotor_diameters",
        },
        "target_partition_access": "all selected original TRAIN rows only",
        "wind_test_target_values_read": False,
        "solver_attempts": 0,
    }
    key_sha = _stable_json_sha256(key)
    cache_file = cache_dir / f"{profile_id}-transforms-{key_sha[:20]}.json"
    with _role_scale_cache_lock(cache_file):
        cached = _read_transform_cache(cache_file, key)
        if cached is None:
            normalizer, profile = fit_velocity_statistics(
                view.volume, rows, samples_per_row=int(key["samples_per_row"]), seed=int(seed),
                profile_bins=int(key["profile_bins"]),
            )
            payload = _transform_cache_payload(key, normalizer, profile)
            _atomic_write_json(cache_file, payload)
        else:
            normalizer, profile = cached
    binding = {
        "schema_version": JOINT_WIND_ROLE_SCALE_CACHE_SCHEMA,
        "transform_fit_id": key["transform_fit_id"],
        "cache_key_sha256": key_sha,
        "dataset_fingerprint": key["dataset_fingerprint"],
        "training_rows_sha256": key["train_row_indices_sha256"],
        "source_rows": len(rows),
        "samples_per_row": int(key["samples_per_row"]),
        "profile_bins": int(key["profile_bins"]),
        "target_partition_access": key["target_partition_access"],
        "wind_test_target_values_read": False,
        "solver_attempts": 0,
        "cache_file_sha256": _sha256(cache_file),
    }
    return normalizer, profile, binding


def _fit_or_load_role_scales(
    *,
    view: WindFarmNativeView,
    train_rows: np.ndarray,
    train_layouts: Sequence[int],
    profile: Any,
    normalizer: Any,
    manifest: Mapping[str, Any],
    training_fingerprint: str,
    catalogue_cache: NativeRoleCatalogueCache,
    cache_dir: Path,
    native_sampling_protocol: str | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    key = _role_scale_cache_key(
        view=view,
        manifest=manifest,
        train_rows=train_rows,
        train_layouts=train_layouts,
        profile_id=str(manifest.get("subset_id", JOINT_WIND_FORMAL_PROFILE_ID)),
        training_fingerprint=training_fingerprint,
        normalizer=normalizer,
        profile=profile,
        native_sampling_protocol=native_sampling_protocol,
    )
    key_sha = _stable_json_sha256(key)
    safe_profile = str(key["dataset_profile"]).replace("/", "_")
    cache_file = cache_dir / f"{safe_profile}-{key_sha[:20]}.json"
    with _role_scale_cache_lock(cache_file):
        cached = _read_role_scale_cache(cache_file, key)
        if cached is not None:
            return cached, {
                "schema_version": JOINT_WIND_ROLE_SCALE_CACHE_SCHEMA,
                "cache_key_sha256": key_sha,
                "dataset_fingerprint": key["dataset_fingerprint"],
                "calibration_id": JOINT_WIND_ROLE_SCALE_CALIBRATION_ID,
                "cache_file_sha256": _sha256(cache_file),
            }
        fitted_result = _fit_train_role_scales(
            view,
            train_rows,
            train_layouts,
            profile,
            normalizer,
            training_fingerprint=training_fingerprint,
            catalogue_cache=catalogue_cache,
        )
        volatile_fields = {
            "catalogue_build_or_lookup_seconds",
            "native_target_sampling_seconds",
            "catalogue_cache_after_calibration",
        }
        calibration = {name: value for name, value in fitted_result.items() if name not in volatile_fields}
        cache_payload = _calibration_cache_payload(key, calibration)
        _atomic_write_json(cache_file, cache_payload)
        return calibration, {
            "schema_version": JOINT_WIND_ROLE_SCALE_CACHE_SCHEMA,
            "cache_key_sha256": key_sha,
            "dataset_fingerprint": key["dataset_fingerprint"],
            "calibration_id": JOINT_WIND_ROLE_SCALE_CALIBRATION_ID,
            "cache_file_sha256": _sha256(cache_file),
        }


def _readonly_array(value: Any, dtype: Any = np.float32) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class WindJointSceneInputs:
    """Whitelisted Wind geometry/context; contains no physical targets."""

    row_index: int
    case_name: str
    layout_index: int
    wind_direction_deg: float
    module_centers: np.ndarray
    module_present: np.ndarray
    module_features: np.ndarray
    module_ids: np.ndarray
    global_context: np.ndarray
    support_lower_D: np.ndarray
    support_upper_D: np.ndarray
    support_extent_D: np.ndarray
    env_coords: np.ndarray
    env_features: np.ndarray
    env_weights: np.ndarray


@dataclass(frozen=True)
class WindJointReceivers:
    coordinates_D: np.ndarray
    role_ids: np.ndarray


@dataclass(frozen=True)
class WindJointTargets:
    velocity_mps: np.ndarray
    row_indices: tuple[int, ...]
    case_metadata: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class WindJointPredictions:
    main_mps: torch.Tensor
    role_ids: torch.Tensor
    auxiliary: Mapping[str, Any]
    execution_mode: str


def _scene_inputs(case: Any) -> WindJointSceneInputs:
    """Copy only fields built from compact geometry and prescribed context."""

    support = case.support
    return WindJointSceneInputs(
        row_index=int(case.index),
        case_name=str(case.case),
        layout_index=int(case.layout_index),
        wind_direction_deg=float(case.wind_direction_deg),
        module_centers=_readonly_array(case.module_centers),
        module_present=_readonly_array(case.module_present),
        module_features=_readonly_array(case.module_features),
        module_ids=_readonly_array(np.arange(np.asarray(case.module_centers).shape[0]), dtype=np.int64),
        global_context=_readonly_array(case.global_context),
        support_lower_D=_readonly_array(support.lower_D),
        support_upper_D=_readonly_array(support.upper_D),
        support_extent_D=_readonly_array(support.extent_D),
        env_coords=_readonly_array(case.env_coords),
        env_features=_readonly_array(case.env_features),
        env_weights=_readonly_array(case.env_weights),
    )


def _role_counts_for_query_count(query_count: int) -> dict[str, int]:
    """Scale the canonical five-role mix with deterministic largest remainders."""

    query_count = int(query_count)
    if query_count < len(ROLE_NAMES):
        raise ValueError("Wind joint queries must provide at least one draw for every native role.")
    canonical = {"volume": 205, "hub_slab": 205, "downstream_envelope": 205, "near_turbine": 205, "background": 204}
    total = sum(canonical.values())
    exact = {name: query_count * canonical[name] / total for name in ROLE_NAMES}
    counts = {name: math.floor(exact[name]) for name in ROLE_NAMES}
    remainder = query_count - sum(counts.values())
    order = sorted(ROLE_NAMES, key=lambda name: (-(exact[name] - counts[name]), ROLE_NAMES.index(name)))
    for name in order[:remainder]:
        counts[name] += 1
    if sum(counts.values()) != query_count or any(count <= 0 for count in counts.values()):
        raise RuntimeError("Wind joint role allocation failed its exact query-count contract.")
    return counts


def _role_ids(counts: Mapping[str, int]) -> np.ndarray:
    return np.concatenate([np.full(int(counts[name]), _ROLE_INDEX[name], dtype=np.int8) for name in ROLE_NAMES])


def _joint_scene_batch(
    inputs: Sequence[WindJointSceneInputs], device: torch.device, *, environment_count: int = JOINT_WIND_ENVIRONMENT_COUNT
) -> InteractionScene:
    if not inputs:
        raise ValueError("Wind joint scene batches cannot be empty.")
    batch = len(inputs)
    source_count = max(int(item.module_centers.shape[0]) for item in inputs)
    if source_count > 30:
        raise ValueError("Wind joint source inventory exceeds the 30 physical turbine slots.")
    environment_count = int(environment_count)
    if environment_count < 1:
        raise ValueError("Wind scene environment count must be positive.")
    if any(int(item.env_coords.shape[0]) != environment_count for item in inputs):
        raise ValueError(f"Wind joint scenes require exactly {environment_count} geometry-only environment records.")
    if any(item.env_features.shape != (environment_count, 7) for item in inputs):
        raise ValueError(f"Wind geometry-only environment features must have shape [{environment_count},7].")

    sources = torch.zeros((batch, source_count, 2), device=device, dtype=torch.float32)
    centers = torch.zeros((batch, source_count, 3), device=device, dtype=torch.float32)
    present = torch.zeros((batch, source_count), device=device, dtype=torch.float32)
    source_lengths = torch.ones_like(present)
    source_measures = torch.zeros_like(present)
    source_ids = torch.full((batch, source_count), -1, device=device, dtype=torch.long)
    contexts = torch.zeros((batch, 11), device=device, dtype=torch.float32)
    lengths = torch.zeros((batch, 3), device=device, dtype=torch.float32)
    environment_tokens = torch.zeros((batch, environment_count, 7), device=device, dtype=torch.float32)
    environment_coords = torch.zeros((batch, environment_count, 3), device=device, dtype=torch.float32)
    environment_measures = torch.zeros((batch, environment_count), device=device, dtype=torch.float32)

    for index, item in enumerate(inputs):
        count = int(item.module_centers.shape[0])
        if item.module_centers.shape != (count, 3) or item.module_features.shape != (count, 2):
            raise ValueError("Wind source geometry/features must have aligned [M,3] and [M,2] shapes.")
        if item.module_present.shape != (count,) or np.any((item.module_present < 0.0) | (item.module_present > 1.0)):
            raise ValueError("Wind source presence must align with native slots and lie in [0,1].")
        if item.module_ids.shape != (count,) or np.unique(item.module_ids).size != count or np.any(item.module_ids < 0):
            raise ValueError("Wind module IDs must be unique native physical source identities.")
        if item.global_context.shape != (11,) or item.support_extent_D.shape != (3,):
            raise ValueError("Wind joint context/support dimensions differ from the native adapter contract.")
        if item.env_coords.shape != (environment_count, 3):
            raise ValueError(f"Wind environment coordinates must have shape [{environment_count},3].")
        if item.env_weights.shape != (environment_count,) or np.any(item.env_weights < 0.0):
            raise ValueError("Wind environment D^3 quadrature measures must be nonnegative and align with the declared E.")
        centers[index, :count] = torch.tensor(item.module_centers, device=device)
        sources[index, :count] = torch.tensor(item.module_features, device=device)
        present[index, :count] = torch.tensor(item.module_present, device=device)
        source_lengths[index, :count] = 2.0 * sources[index, :count, 0]
        source_measures[index, :count] = present[index, :count]
        source_ids[index, :count] = torch.tensor(item.module_ids, device=device, dtype=torch.long)
        contexts[index] = torch.tensor(item.global_context, device=device)
        lengths[index] = torch.tensor(item.support_extent_D, device=device)
        environment_tokens[index] = torch.tensor(item.env_features, device=device)
        environment_coords[index] = torch.tensor(item.env_coords, device=device)
        environment_measures[index] = torch.tensor(item.env_weights, device=device)

    packed = (
        centers.flatten(), sources.flatten(), present.flatten(), contexts.flatten(), lengths.flatten(),
        environment_tokens.flatten(), environment_coords.flatten(), environment_measures.flatten(),
    )
    if not torch.isfinite(torch.cat(packed)).all():
        raise ValueError("Wind joint scene inputs contain nonfinite geometry, features or measures.")
    if bool((lengths <= 0).any()) or bool((source_lengths[present > 0] <= 0).any()):
        raise ValueError("Wind support/source lengths must be positive for active physical nodes.")
    if bool((environment_measures.sum(dim=1) <= 0).any()):
        raise ValueError("Wind E64 quadrature must represent positive physical domain volume.")
    return InteractionScene(
        sources=sources,
        context=contexts,
        centers=centers,
        present=present,
        lengths=lengths,
        source_lengths=source_lengths,
        dependency=WIND_JOINT_DEPENDENCY,
        source_measures=source_measures,
        environment_tokens=environment_tokens,
        environment_coords=environment_coords,
        environment_present=torch.ones_like(environment_measures),
        environment_measures=environment_measures,
        source_ids=source_ids,
    )


def _sampling_dataset_id(
    profile_id: str,
    fingerprint: str,
    native_sampling_protocol: str | None = None,
) -> str:
    # Match the engine's case_epoch_v1 key exactly: dataset plus the sealed
    # TRAIN membership fingerprint. The profile is already bound by that
    # manifest, and must not become a second, incompatible hashing recipe.
    del profile_id
    if native_sampling_protocol is not None:
        if native_sampling_protocol != JOINT_WIND_BASELINE_FORMAL_SAMPLING_PROTOCOL:
            raise ValueError(f"Unsupported Wind native sampling protocol {native_sampling_protocol!r}.")
        return _runtime_sampling_dataset_id({
            "dataset": "WindFarm",
            "native_sampling_identity": {
                "protocol": native_sampling_protocol,
                "dataset": "WindFarm",
                "train_membership_fingerprint": str(fingerprint),
            },
        })
    return _runtime_sampling_dataset_id(
        {"dataset": "WindFarm", "subset_manifest_sha256": str(fingerprint)}
    )


def _formal_sampling_fingerprint(
    protocol: str | None,
    *,
    manifest_fingerprint: str,
    train_rows: Sequence[int],
) -> str:
    """Resolve the per-case RNG namespace without changing development streams."""

    if protocol is None:
        return str(manifest_fingerprint)
    if protocol != JOINT_WIND_BASELINE_FORMAL_SAMPLING_PROTOCOL:
        raise ValueError(f"Unsupported Wind native sampling protocol {protocol!r}.")
    return _indices_sha256(np.asarray(train_rows, dtype=np.int64))


def _validate_formal_controls(
    *,
    formal_full: bool,
    total_epochs: int,
    optimizer_schedule: ScheduleSpec | None,
    weight_decay: float | None,
    validation_scope: str | None,
    native_sampling_protocol: str | None,
    allow_development_optimizer_controls: bool = False,
) -> None:
    """Keep legacy formal controls sealed while allowing P-family dev schedules."""

    if allow_development_optimizer_controls:
        if not formal_full and (validation_scope is not None or native_sampling_protocol is not None):
            raise ValueError("Wind formal data controls require formal_full.")
    else:
        controls = (optimizer_schedule, weight_decay, validation_scope, native_sampling_protocol)
        if any(value is not None for value in controls) and (not formal_full or int(total_epochs) != 5000):
            raise ValueError("Wind formal comparison controls are available only for formal_full 5000-epoch fits.")
    if optimizer_schedule is not None and not isinstance(optimizer_schedule, ScheduleSpec):
        raise TypeError("Wind optimizer_schedule must be a ScheduleSpec.")
    if optimizer_schedule is not None:
        expected_epochs = int(total_epochs) if allow_development_optimizer_controls else 5000
        if optimizer_schedule.total_epochs != expected_epochs:
            if allow_development_optimizer_controls:
                raise ValueError("Wind optimizer_schedule must match the declared training horizon.")
            raise ValueError("Wind formal optimizer_schedule must end at epoch 5000.")
    if (
        weight_decay is not None
        and (
            isinstance(weight_decay, bool)
            or not math.isfinite(float(weight_decay))
            or float(weight_decay) < 0.0
        )
    ):
        raise ValueError("Wind weight_decay must be finite and nonnegative.")
    if validation_scope not in (None, "fullVALID90"):
        raise ValueError("Wind validation_scope must be omitted or the explicit fullVALID90 scope.")
    if native_sampling_protocol not in (None, JOINT_WIND_BASELINE_FORMAL_SAMPLING_PROTOCOL):
        raise ValueError("Unsupported Wind native sampling protocol.")
    if (
        native_sampling_protocol == JOINT_WIND_BASELINE_FORMAL_SAMPLING_PROTOCOL
        and validation_scope != "fullVALID90"
    ):
        raise ValueError("Wind baseline_formal_v1 sampling must be paired with validation_scope='fullVALID90'.")


class WindJointRegionalTask:
    """Fresh joint Wind provider with component-balanced native supervision."""

    environment_token_shape = JOINT_WIND_ENVIRONMENT_SHAPE
    environment_count = JOINT_WIND_ENVIRONMENT_COUNT

    def __init__(
        self,
        view: WindFarmNativeView,
        *,
        model: WindFarmJointRegionalModel,
        train_rows: Sequence[int],
        validation_rows: Sequence[int],
        manifest: Mapping[str, Any],
        profile_id: str,
        training_fingerprint: str,
        role_component_scales: Mapping[str, Sequence[float]],
        role_scale_calibration: Mapping[str, Any],
        role_scale_cache_identity: Mapping[str, Any],
        role_scale_source_sha256: str,
        normalizer_source: Mapping[str, Any],
        seed: int,
        primary_queries: int,
        microbatch_size: int,
        effective_batch_size: int,
        total_epochs: int,
        device: torch.device | str,
        catalogue_cache: NativeRoleCatalogueCache,
        optimizer_schedule: ScheduleSpec | None = None,
        weight_decay: float | None = None,
        validation_scope: str | None = None,
        native_sampling_protocol: str | None = None,
        environment_token_shape: Sequence[int] = JOINT_WIND_ENVIRONMENT_SHAPE,
        full_train_followup1000: bool = False,
    ) -> None:
        self.view = view
        self.model = model
        self.environment_token_shape = tuple(int(value) for value in environment_token_shape)
        if len(self.environment_token_shape) != 3 or any(value < 1 for value in self.environment_token_shape):
            raise ValueError("Wind geometry token shape must contain three positive dimensions.")
        self.environment_count = int(np.prod(self.environment_token_shape))
        view_token_shape = getattr(view, "token_shape", None)
        if view_token_shape is not None and tuple(view_token_shape) != self.environment_token_shape:
            raise ValueError("Wind native view tokenization differs from the provider identity.")
        if model.mode in INTERACTION_PRESERVING_WIND_MODES and view_token_shape is None:
            raise ValueError("Wind P-family providers require the native view's explicit token_shape.")
        self.train_rows = np.asarray(train_rows, dtype=np.int64)
        self.validation_rows = np.asarray(validation_rows, dtype=np.int64)
        self.manifest = dict(manifest)
        self.profile_id = str(profile_id)
        self.training_fingerprint = str(training_fingerprint)
        self.role_component_scales = {
            name: np.asarray(role_component_scales[name], dtype=np.float64) for name in ROLE_NAMES
        }
        self.role_scale_calibration = dict(role_scale_calibration)
        self.role_scale_cache_identity = dict(role_scale_cache_identity)
        self.role_scale_source_sha256 = str(role_scale_source_sha256)
        self.normalizer_source = dict(normalizer_source)
        self.seed = int(seed)
        self.primary_queries = int(primary_queries)
        self.role_query_counts = _role_counts_for_query_count(self.primary_queries)
        self.device = torch.device(device)
        self.max_microbatch_cases = int(microbatch_size)
        self.effective_batch_size = int(effective_batch_size)
        self.total_epochs = int(total_epochs)
        self.catalogue_cache = catalogue_cache
        self.sampling_version = SamplingKey.CASE_EPOCH_VERSION
        manifest_fingerprint = str(self.manifest.get("manifest_sha256", self.training_fingerprint))
        self.full_train_followup1000 = bool(full_train_followup1000)
        profile_is_followup = self.profile_id == JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID
        if profile_is_followup != self.full_train_followup1000:
            raise ValueError("Wind full-followup profile and explicit protocol flag must agree.")
        if self.full_train_followup1000 and (self.total_epochs != 5000
                                             or model.mode not in INTERACTION_PRESERVING_WIND_MODES):
            raise ValueError("Wind full-TRAIN follow-up is a fresh P-family 5000-horizon identity.")
        if self.full_train_followup1000:
            layout_index = np.asarray(self.view.metadata["layout_index"], dtype=np.int64)
            validate_full_train_followup_membership(
                self.train_rows, self.validation_rows,
                self.manifest.get("test_row_indices", ()),
                train_layout_count=len(np.unique(layout_index[self.train_rows])),
                validation_layout_count=len(np.unique(layout_index[self.validation_rows])),
                test_layout_count=len(self.manifest.get("test_layout_indices", ())),
            )
        formal_full = (self.profile_id in (JOINT_WIND_FORMAL_PROFILE_ID,
                                           JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID)
                       and self.total_epochs == 5000)
        _validate_formal_controls(
            formal_full=formal_full,
            total_epochs=self.total_epochs,
            optimizer_schedule=optimizer_schedule,
            weight_decay=weight_decay,
            validation_scope=validation_scope,
            native_sampling_protocol=native_sampling_protocol,
            allow_development_optimizer_controls=model.mode in INTERACTION_PRESERVING_WIND_MODES,
        )
        self.optimizer_schedule = optimizer_schedule
        self.optimizer_weight_decay = None if weight_decay is None else float(weight_decay)
        self.validation_scope = validation_scope
        self.native_sampling_protocol = native_sampling_protocol
        self.sampling_fingerprint = _formal_sampling_fingerprint(
            native_sampling_protocol,
            manifest_fingerprint=manifest_fingerprint,
            train_rows=self.train_rows,
        )
        self.sampling_dataset_id = _sampling_dataset_id(
            self.profile_id, self.sampling_fingerprint, native_sampling_protocol
        )
        self._train_row_set = set(map(int, self.train_rows.tolist()))
        self._validation_cache: tuple[TaskBatch, ...] | None = None
        self.training_samples = 0
        self.validation_samples = 0
        if not self.train_rows.size or not self.validation_rows.size:
            raise ValueError("Wind joint provider requires nonempty TRAIN and development rows.")
        if not np.array_equal(self.train_rows, np.sort(self.train_rows)):
            raise ValueError("Wind joint TRAIN rows must be in canonical native order.")
        if not np.array_equal(self.validation_rows, np.sort(self.validation_rows)):
            raise ValueError("Wind joint development rows must be in canonical native order.")
        if self.primary_queries < len(ROLE_NAMES) or self.max_microbatch_cases <= 0:
            raise ValueError("Wind query/microbatch settings must be positive and cover all five roles.")
        if self.effective_batch_size != 24 or self.max_microbatch_cases > self.effective_batch_size:
            raise ValueError("Wind joint uses effective batch 24 and microbatches no larger than 24 cases.")
        if self.seed != 42:
            raise ValueError("Wind joint fixed-profile identity is sealed to initialization/sampling seed 42.")
        if self.total_epochs not in (2500, 5000):
            raise ValueError("Wind joint horizon must be 2500 development or 5000 formal/follow-up schedule epochs.")
        if self.validation_scope == "fullVALID90":
            layout_indices = np.asarray(self.view.metadata["layout_index"], dtype=np.int64)
            validation_layouts = np.unique(layout_indices[self.validation_rows])
            if self.validation_rows.size != 90 or validation_layouts.size != 30:
                raise ValueError("Wind fullVALID90 scope requires all 90 original validation rows across 30 layouts.")
        if set(self.role_component_scales) != set(ROLE_NAMES) or any(
            scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0.0)
            for scale in self.role_component_scales.values()
        ):
            raise ValueError("Wind joint scales must be positive finite three-component m/s vectors per role.")

    def identity_payload(self) -> Mapping[str, Any]:
        payload = {
            "dataset": "WindFarm",
            "family": getattr(self.model, "FAMILY", WindFarmJointRegionalModel.FAMILY),
            "mode": self.model.mode,
            "dataset_profile": self.profile_id,
            "profile_is_formal_fulltrain": self.profile_id == JOINT_WIND_FORMAL_PROFILE_ID,
            "subset_id": self.manifest.get("subset_id", self.profile_id),
            "subset_manifest_sha256": self.manifest.get("manifest_sha256", self.training_fingerprint),
            "train_row_indices_sha256": _indices_sha256(self.train_rows),
            "validation_row_indices_sha256": _indices_sha256(self.validation_rows),
            "train_rows": self.train_rows.astype(int).tolist(),
            "validation_rows": self.validation_rows.astype(int).tolist(),
            "train_layout_indices": sorted({int(self.view.metadata["layout_index"][row]) for row in self.train_rows}),
            "validation_layout_indices": sorted(
                {int(self.view.metadata["layout_index"][row]) for row in self.validation_rows}
            ),
            "directions_deg": [270.0, 285.0, 300.0],
            "environment_representation": {
                "token_shape": list(self.environment_token_shape),
                "record_count": self.environment_count,
                "measure": "geometry-only native domain-support quadrature in rotor_diameters^3",
                "field_sensors": False,
            },
            "normalizer_and_background_source": self.normalizer_source,
            "standalone_model_config": self.model.export_config(),
            "objective": {
                "name": "five_role_component_balanced_velocity_residual",
                "role_query_counts": dict(self.role_query_counts),
                "component_scales_mps": {
                    role: values.astype(float).tolist() for role, values in self.role_component_scales.items()
                },
                "scale_rule": JOINT_WIND_ROLE_SCALE_RULE,
                "scale_source_sha256": self.role_scale_source_sha256,
                "scale_calibration": self.role_scale_calibration,
                "scale_cache_binding": self.role_scale_cache_identity,
                "role_weight": 0.2,
                "per_role_formula": "mean_{q,c} ((Uhat-U)^2 / scale[role,c]^2)",
                "physical_output": "TRAIN height profile m/s + standardized nonlinear residual * u_ref_mps * safe_std",
                "target_partition_access": "TRAIN targets only; selected development targets only for metrics; WindTEST locked",
            },
            "training": {
                "initialization_seed": self.seed,
                "sampling_version": self.sampling_version,
                "sampling_dataset_id": self.sampling_dataset_id,
                "effective_batch_size": self.effective_batch_size,
                "microbatch_size": self.max_microbatch_cases,
                "primary_queries_per_case": self.primary_queries,
                "total_epochs": self.total_epochs,
                "optimizer": "AdamW",
                "peak_lr": 3.0e-4 if self.optimizer_schedule is None else self.optimizer_schedule.peak_lr,
                "warmup_epochs": 20 if self.optimizer_schedule is None else self.optimizer_schedule.warmup_epochs,
                "hold_through_epoch": 1000 if self.optimizer_schedule is None else self.optimizer_schedule.hold_through_epoch,
                "final_lr": 3.0e-6 if self.optimizer_schedule is None else self.optimizer_schedule.final_lr,
                "all_parameters_train_from_start": True,
            },
            "catalogue_cache_capacity_bytes": self.catalogue_cache.max_cached_bytes,
            "physical_reference": "stored native OpenFOAM velocity fields; no solver calls",
            "wind_test_target_values_read": False,
        }
        if self.full_train_followup1000:
            payload.update({
                "execution_protocol": FULL_TRAIN_FOLLOWUP_PROTOCOL,
                "dataset_protocol": FULL_TRAIN_FOLLOWUP_DATASET,
                "full_train_followup1000": True,
                "approved_stop_after": 1000,
                "test_row_indices_sha256": self.manifest.get("test_row_indices_sha256"),
                "test_row_count": self.manifest.get("test_row_count"),
                "test_layout_indices": self.manifest.get("test_layout_indices"),
                "wind_test_target_values_read": False,
                "normalization_training_membership_sha256": _indices_sha256(self.train_rows),
                "role_scale_training_membership_sha256": _indices_sha256(self.train_rows),
            })
        if self.native_sampling_protocol is not None:
            payload["native_sampling_identity"] = {
                "protocol": self.native_sampling_protocol,
                "dataset": "WindFarm",
                "train_membership_fingerprint": _indices_sha256(self.train_rows),
            }
        if self.validation_scope is not None:
            payload["validation_scope"] = {
                "name": self.validation_scope,
                "rows": int(self.validation_rows.size),
                "layouts": len({int(self.view.metadata["layout_index"][row]) for row in self.validation_rows}),
                "row_indices_sha256": _indices_sha256(self.validation_rows),
                "targets_from_windtest": False,
            }
        if self.native_sampling_protocol is not None:
            payload["native_sampling"] = {
                "protocol": self.native_sampling_protocol,
                "sampling_version": self.sampling_version,
                "seed": self.seed,
                "dataset_key_source": JOINT_WIND_BASELINE_TRAIN_ROW_SHA256_KEY,
                "train_rows_sha256": _indices_sha256(self.train_rows),
                "sampling_dataset_id": self.sampling_dataset_id,
                "training_stream": JOINT_WIND_BASELINE_TRAIN_STREAM,
                "validation_stream": JOINT_WIND_BASELINE_VALIDATION_STREAM,
                "role_calibration_stream": JOINT_WIND_BASELINE_CALIBRATION_STREAM,
                "per_case_role_key_coordinates": ["physical_case_direction_id", "role_name"],
            }
        if self.optimizer_schedule is not None or self.optimizer_weight_decay is not None:
            schedule = self.optimizer_schedule or ScheduleSpec(
                peak_lr=3.0e-4,
                warmup_start_lr=3.0e-5,
                warmup_epochs=20,
                hold_through_epoch=1000,
                total_epochs=self.total_epochs,
                final_lr=3.0e-6,
            )
            payload["training"]["optimizer_schedule"] = asdict(schedule)
            payload["training"]["optimizer_weight_decay"] = (
                1.0e-5 if self.optimizer_weight_decay is None else self.optimizer_weight_decay
            )
            payload["training"]["optimizer_betas"] = [0.9, 0.999]
            payload["training"]["optimizer_eps"] = 1.0e-8
        formal_controls_enabled = any((
            self.optimizer_schedule is not None,
            self.optimizer_weight_decay is not None,
            self.validation_scope is not None,
            self.native_sampling_protocol is not None,
        ))
        if self.profile_id in (JOINT_WIND_FORMAL_PROFILE_ID, JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID) \
                and formal_controls_enabled:
            payload["training"]["case_visits_per_epoch"] = int(self.train_rows.size)
            payload["training"]["horizon_updates_per_epoch"] = math.ceil(
                self.train_rows.size / self.effective_batch_size
            )
        return payload

    def preparation_summary(self) -> dict[str, Any]:
        """Return JSON-safe setup and native-cache accounting before optimization."""

        manifest_id = str(self.manifest.get("subset_id", self.profile_id))
        manifest_sha = str(self.manifest.get("manifest_sha256", self.training_fingerprint))
        train_rows = int(self.train_rows.size)
        validation_rows = int(self.validation_rows.size)
        train_layouts = {
            int(self.view.metadata["layout_index"][row]) for row in self.train_rows
        }
        validation_layouts = {
            int(self.view.metadata["layout_index"][row]) for row in self.validation_rows
        }
        training_cases_sampled = int(self.training_samples)
        validation_cases_sampled = int(self.validation_samples)
        return {
            "status": "prepared_only",
            "subset_id": manifest_id,
            "subset_manifest_sha256": manifest_sha,
            "dataset_profile": str(self.profile_id),
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "train_layout_count": len(train_layouts),
            "validation_layout_count": len(validation_layouts),
            "train_row_indices_sha256": _indices_sha256(self.train_rows),
            "validation_row_indices_sha256": _indices_sha256(self.validation_rows),
            "training_cases_sampled": training_cases_sampled,
            "validation_cases_sampled": validation_cases_sampled,
            "training_query_samples": training_cases_sampled * self.primary_queries,
            "validation_query_samples": validation_cases_sampled * self.primary_queries,
            "role_query_counts": {
                str(role): int(count) for role, count in self.role_query_counts.items()
            },
            "primary_queries_per_case": int(self.primary_queries),
            "effective_batch_size": int(self.effective_batch_size),
            "microbatch_size": int(self.max_microbatch_cases),
            "total_epochs": int(self.total_epochs),
            "sampling_version": str(self.sampling_version),
            "environment_representation": {
                "record_count": self.environment_count,
                "token_shape": list(self.environment_token_shape),
                "geometry_only": True,
                "field_sensors": False,
            },
            "normalizer_and_background_source": dict(self.normalizer_source),
            "role_scale_calibration": dict(self.role_scale_calibration),
            "role_scale_cache_binding": dict(self.role_scale_cache_identity),
            "role_scale_source_sha256": str(self.role_scale_source_sha256),
            "native_role_catalogue_cache": self.catalogue_cache.summary(),
            "optimizer_started": False,
            "solver_attempts": 0,
            "wind_test_target_values_read": False,
        }

    def loss_metadata(self) -> Mapping[str, Mapping[str, str]]:
        return {
            f"native_role/{role}": {
                "panel": "native_prediction",
                "label": f"{role} velocity",
                "unit": "m/s",
                "formula": "mean over sampled native points and Ux/Uy/Uz of squared residual divided by the TRAIN-fitted role/component scale squared",
                "weight": "0.2 (equal contribution from each of five physical roles)",
            }
            for role in ROLE_NAMES
        }

    def phase_metadata(self, phase: str = "joint") -> Mapping[str, Any]:
        if phase != "joint":
            raise ValueError("Wind joint regional provider supports only the ordinary joint learning phase.")
        return {"training_mode": "joint", "phase": "joint", "gate_schedule": None, "active_parameters": "all"}

    def epoch_cases(self, epoch: int, seed: int) -> Sequence[int]:
        del epoch
        if int(seed) != self.seed:
            raise ValueError("Wind joint epoch sampling seed differs from the sealed profile seed.")
        return tuple(map(int, self.train_rows.tolist()))

    def _sample_row(
        self, row: int, key: SamplingKey, *, validation: bool = False
    ) -> tuple[WindJointSceneInputs, Any]:
        case = self.view.run(int(row))
        physical_id = _physical_case_direction_id(case)
        if self.native_sampling_protocol == JOINT_WIND_BASELINE_FORMAL_SAMPLING_PROTOCOL:
            stream = JOINT_WIND_BASELINE_VALIDATION_STREAM if validation else JOINT_WIND_BASELINE_TRAIN_STREAM
            fallback_tag = 0x56414C31 if validation else 0x57494E44
            fallback_rng = key.numpy_rng(stream, int(row), fallback_tag)
        else:
            stream = "wind_joint_validation_role_query" if validation else "wind_joint_native_role_query"
            fallback_rng = key.numpy_rng(f"{stream}_fallback", physical_id)
        sample = sample_native_role_queries(
            case,
            fallback_rng,
            self.role_query_counts,
            catalogue_cache=self.catalogue_cache,
            rng_by_role={
                role: key.numpy_rng(stream, physical_id, role)
                for role in ROLE_NAMES
            },
        )
        if sample.coordinates_D.shape != (self.primary_queries, 3) or sample.target_mps.shape != (
            self.primary_queries, 3
        ):
            raise ValueError("Wind native role sampler changed the resolved joint query/target shape.")
        if not np.isfinite(sample.target_mps).all() or not np.isfinite(sample.coordinates_D).all():
            raise ValueError(f"Wind joint native row {row} contains nonfinite sampled values.")
        return _scene_inputs(case), sample

    def _make_batch(self, rows: Sequence[int], key: SamplingKey, *, training: bool) -> TaskBatch:
        if len(rows) == 0:
            raise ValueError("Wind joint batches cannot be empty.")
        if len(rows) > self.max_microbatch_cases:
            raise ValueError("Wind joint batch exceeds its declared case microbatch size.")
        row_ids = tuple(int(value) for value in rows)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Wind joint batches cannot repeat native rows.")
        if training and any(row not in self._train_row_set for row in row_ids):
            raise ValueError("Wind joint TRAIN batch contains a row outside the sealed TRAIN membership.")
        if not training and any(row not in set(map(int, self.validation_rows.tolist())) for row in row_ids):
            raise ValueError("Wind joint validation batch contains a row outside the selected development panel.")

        scene_rows: list[WindJointSceneInputs] = []
        receivers: list[np.ndarray] = []
        role_rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        metadata: list[Mapping[str, Any]] = []
        for row in row_ids:
            scene_inputs, sample = self._sample_row(row, key, validation=not training)
            scene_rows.append(scene_inputs)
            receivers.append(np.asarray(sample.coordinates_D, dtype=np.float32))
            role_ids = _role_ids(sample.role_sample_counts)
            if role_ids.shape != (self.primary_queries,):
                raise RuntimeError("Wind role assignments do not align with every native receiver.")
            role_rows.append(role_ids)
            targets.append(np.asarray(sample.target_mps, dtype=np.float32))
            metadata.append(
                {
                    "row_index": int(row),
                    "case": scene_inputs.case_name,
                    "layout_index": int(scene_inputs.layout_index),
                    "wind_direction_deg": float(scene_inputs.wind_direction_deg),
                    "n_turbines": int(np.count_nonzero(scene_inputs.module_present > 0.5)),
                }
            )
        if training:
            self.training_samples += len(row_ids)
        else:
            self.validation_samples += len(row_ids)
        receiver_inputs = WindJointReceivers(np.stack(receivers), np.stack(role_rows))
        target_inputs = WindJointTargets(np.stack(targets), row_ids, tuple(metadata))
        return TaskBatch(
            scene_inputs=tuple(scene_rows),
            receivers=receiver_inputs,
            targets=target_inputs,
            auxiliary={"role_names": ROLE_NAMES},
            case_keys=row_ids,
        )

    def make_batch(self, case_keys: Sequence[Any], key: SamplingKey) -> TaskBatch:
        if key.sampling_version != SamplingKey.CASE_EPOCH_VERSION or key.dataset_id != self.sampling_dataset_id:
            raise ValueError("Wind joint sampling requires the exact case_epoch_v1 dataset identity.")
        return self._make_batch(case_keys, key, training=True)

    def make_scene(self, scene_inputs: Sequence[WindJointSceneInputs]) -> InteractionScene:
        if any(not isinstance(item, WindJointSceneInputs) for item in scene_inputs):
            raise TypeError("Wind joint prediction accepts only target-free WindJointSceneInputs.")
        return _joint_scene_batch(scene_inputs, self.device, environment_count=self.environment_count)

    def _checked_role_ids(self, batch: TaskBatch) -> np.ndarray:
        coordinates = np.asarray(batch.receivers.coordinates_D, dtype=np.float32)
        role_ids = np.asarray(batch.receivers.role_ids, dtype=np.int64)
        if coordinates.ndim != 3 or coordinates.shape[0] != len(batch.scene_inputs) or coordinates.shape[-1] != 3:
            raise ValueError("Wind joint receivers must have shape [B,Q,3].")
        if role_ids.shape != coordinates.shape[:2] or np.any((role_ids < 0) | (role_ids >= len(ROLE_NAMES))):
            raise ValueError("Wind joint role IDs must align with every physical receiver.")
        expected = np.concatenate([
            np.full(self.role_query_counts[name], _ROLE_INDEX[name], dtype=np.int64) for name in ROLE_NAMES
        ])
        if any(not np.array_equal(row, expected) for row in role_ids):
            raise ValueError("Wind joint batches must preserve canonical role ordering and exact per-role query counts.")
        return role_ids

    def loss_denominators(
        self, batches: Sequence[TaskBatch], phase: str = "joint", arm: str = "J-H",
    ) -> Mapping[str, float]:
        del arm
        self.phase_metadata(phase)
        result = {f"native_role/{role}": 0.0 for role in ROLE_NAMES}
        for batch in batches:
            role_ids = self._checked_role_ids(batch)
            for role in ROLE_NAMES:
                count = int(np.count_nonzero(role_ids == _ROLE_INDEX[role]))
                if count <= 0:
                    raise ValueError(f"Wind joint microbatch has no sampled receivers for role {role!r}.")
                result[f"native_role/{role}"] += float(count * 3)
        return result

    def predict_native(
        self,
        model: nn.Module,
        scene: InteractionScene,
        receivers: WindJointReceivers,
        execution_mode: str = "joint",
        phase: str = "joint",
        epoch: int = 0,
        temperature: float | None = None,
    ) -> tuple[WindJointPredictions, Mapping[str, Any]]:
        del epoch, temperature
        self.phase_metadata(phase)
        if execution_mode not in {"joint", "full", "all", "all_fine", "J-H", "J-geometry", "J-direct",
                                  *INTERACTION_PRESERVING_WIND_MODES}:
            raise ValueError("Wind joint regional prediction does not support legacy gate/detail execution modes.")
        if not isinstance(model, WindFarmJointRegionalModel):
            raise TypeError("Wind joint provider requires a standalone WindFarmJointRegionalModel.")
        coordinates = torch.as_tensor(receivers.coordinates_D, device=scene.centers.device, dtype=torch.float32)
        role_ids = torch.as_tensor(receivers.role_ids, device=scene.centers.device, dtype=torch.long)
        if role_ids.shape != coordinates.shape[:2]:
            raise ValueError("Wind joint receiver roles are not aligned with the prediction queries.")
        prepared = model.prepare_scene(scene)
        predictions_mps = model.predict_physical(prepared, coordinates, chunk_size=model.receiver_tile)
        baseline = model.profile_at_receivers(coordinates)
        standardized = model.normalize_residual(predictions_mps, coordinates)
        auxiliary = {
            "baseline_mps": baseline,
            "standardized_residual": standardized,
            "prepared_context": prepared.core_context,
            "execution_mode": "joint",
            "phase": "joint",
        }
        result = WindJointPredictions(predictions_mps, role_ids, auxiliary, "joint")
        return result, auxiliary

    def _loss_terms(self, predictions: WindJointPredictions, targets: WindJointTargets) -> Mapping[str, LossTerm]:
        target = torch.as_tensor(targets.velocity_mps, device=predictions.main_mps.device, dtype=predictions.main_mps.dtype)
        if target.shape != predictions.main_mps.shape:
            raise ValueError("Wind joint target velocity shape differs from its native prediction.")
        error = predictions.main_mps - target
        result: dict[str, LossTerm] = {}
        for role in ROLE_NAMES:
            selected = predictions.role_ids == _ROLE_INDEX[role]
            if not bool(selected.any()):
                raise ValueError(f"Wind joint loss has no samples for role {role!r}.")
            scale = error.new_tensor(self.role_component_scales[role])
            numerator = (error[selected].square() / scale.square()).sum()
            denominator = float(int(selected.sum().item()) * 3)
            result[f"native_role/{role}"] = LossTerm(numerator, denominator, weight=0.2)
        return result

    def loss_terms(
        self,
        predictions: WindJointPredictions,
        targets: WindJointTargets,
        phase: str,
        auxiliary_state: Any,
    ) -> Mapping[str, LossTerm]:
        del auxiliary_state
        self.phase_metadata(phase)
        return self._loss_terms(predictions, targets)

    def validation_loss_terms(
        self,
        predictions: WindJointPredictions,
        targets: WindJointTargets,
        auxiliary_state: Any,
        *,
        batch: TaskBatch,
        arm: str,
    ) -> Mapping[str, LossTerm]:
        del auxiliary_state, batch, arm
        return self._loss_terms(predictions, targets)

    def validation_batches(self) -> Iterable[TaskBatch]:
        if self._validation_cache is None:
            batches = []
            for start in range(0, self.validation_rows.size, self.max_microbatch_cases):
                rows = self.validation_rows[start : start + self.max_microbatch_cases]
                key = SamplingKey(
                    self.seed,
                    0,
                    0,
                    start // self.max_microbatch_cases,
                    "validation",
                    "shared",
                    sampling_version=self.sampling_version,
                    dataset_id=self.sampling_dataset_id,
                )
                batch = self._make_batch(rows, key, training=False)
                batches.append(batch)
            self._validation_cache = tuple(batches)
        return iter(self._validation_cache)

    def validation_metrics(
        self,
        predictions: WindJointPredictions,
        targets: WindJointTargets,
        auxiliary_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del auxiliary_state
        predicted = predictions.main_mps.detach().cpu().numpy()
        actual = np.asarray(targets.velocity_mps, dtype=np.float64)
        role_ids = predictions.role_ids.detach().cpu().numpy()
        baseline = predictions.auxiliary["baseline_mps"].detach().cpu().numpy()
        records: list[dict[str, Any]] = []
        for batch_index, metadata in enumerate(targets.case_metadata):
            row: dict[str, Any] = dict(metadata)
            role_metrics: dict[str, Any] = {}
            objectives: list[float] = []
            baseline_objectives: list[float] = []
            for role in ROLE_NAMES:
                selected = role_ids[batch_index] == _ROLE_INDEX[role]
                if not selected.any():
                    raise ValueError(f"Wind development role {role!r} is empty for row {metadata['row_index']}.")
                delta = predicted[batch_index, selected] - actual[batch_index, selected]
                base_delta = baseline[batch_index, selected] - actual[batch_index, selected]
                component_mse = np.mean(delta * delta, axis=0)
                baseline_component_mse = np.mean(base_delta * base_delta, axis=0)
                scale = self.role_component_scales[role]
                objectives.append(float(np.mean(component_mse / np.square(scale))))
                baseline_objectives.append(float(np.mean(baseline_component_mse / np.square(scale))))
                role_metrics[role] = {
                    "component_rmse_mps": np.sqrt(component_mse).tolist(),
                    "component_rmse_mean_mps": float(np.sqrt(np.mean(component_mse))),
                    "vector_rmse_mps": float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1)))),
                    "component_scales_mps": scale.astype(float).tolist(),
                    "query_count": int(selected.sum()),
                }
            row["roles"] = role_metrics
            row["field_score"] = float(np.mean(objectives))
            row["height_profile_score"] = float(np.mean(baseline_objectives))
            records.append(row)
        return {"rows": records}

    def reduce_native_metrics(self, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        rows = [row for record in records for row in record["rows"]]
        expected_rows = set(map(int, self.validation_rows.tolist()))
        observed_rows = {int(row["row_index"]) for row in rows}
        if len(rows) != len(self.validation_rows) or observed_rows != expected_rows:
            raise ValueError("Wind joint metrics must contain the exact selected development row panel once.")
        summary: dict[str, Any] = {
            "field_score": float(np.mean([row["field_score"] for row in rows])),
            "height_profile_score": float(np.mean([row["height_profile_score"] for row in rows])),
            "row_count": len(rows),
            "layout_count": len({int(row["layout_index"]) for row in rows}),
            "rows": rows,
            "roles": {},
        }
        for role in ROLE_NAMES:
            values = np.asarray([row["roles"][role]["component_rmse_mps"] for row in rows], dtype=np.float64)
            vector = np.asarray([row["roles"][role]["vector_rmse_mps"] for row in rows], dtype=np.float64)
            summary["roles"][role] = {
                "component_rmse_mps_mean": values.mean(axis=0).tolist(),
                "component_rmse_mps_worst": values.max(axis=0).tolist(),
                "component_rmse_mps_p95": np.quantile(values, 0.95, axis=0).tolist(),
                "component_rmse_mean_mps": float(np.sqrt(np.mean(values * values))),
                "vector_rmse_mps_mean": float(vector.mean()),
                "vector_rmse_mps_worst": float(vector.max()),
            }
        return summary

    def work_counts(
        self, batch: TaskBatch, predictions: WindJointPredictions, auxiliary_state: Mapping[str, Any]
    ) -> Mapping[str, int | float]:
        del auxiliary_state
        batch_size, query_count = batch.receivers.coordinates_D.shape[:2]
        active_sources = sum(int(np.count_nonzero(item.module_present > 0.0)) for item in batch.scene_inputs)
        prepared = predictions.auxiliary.get("prepared_context")
        group_states = getattr(prepared, "group_states", None)
        edge_count = int(group_states.shape[1]) if torch.is_tensor(group_states) and group_states.ndim >= 3 else 0
        counts: dict[str, int | float] = {
            "native_receivers": int(batch_size * query_count),
            "active_physical_sources": int(active_sources),
            "physical_source_receiver_capacity": int(active_sources * query_count),
            "regional_edges_per_scene": int(edge_count),
            "receiver_edge_read_capacity": int(batch_size * query_count * edge_count),
            "environment_records": int(batch_size * self.environment_count),
            "complete_native_prediction": 1,
        }
        if self.model.mode not in INTERACTION_PRESERVING_WIND_MODES:
            return counts
        if prepared is None:
            raise ValueError("Wind P-family work accounting requires its prepared pair context.")
        core_context = prepared
        present = core_context.present
        source_capacity = int(present.shape[1])
        environment_capacity = int(core_context.environment_coords.shape[1])
        active_by_scene = (present > 0).sum(dim=1).to(torch.int64)
        batch_count = int(present.shape[0])
        active_pair_square = int((active_by_scene * active_by_scene).sum().item())
        active_pair_no_self = int((active_by_scene * (active_by_scene - 1)).sum().item())
        group_states = core_context.group_states
        groups = int(group_states.shape[1]) if torch.is_tensor(group_states) else 0
        source_membership = getattr(core_context, "source_membership", None)
        environment_membership = getattr(core_context, "environment_membership", None)
        pair_rounds = 2
        counts.update({
            "pair_rounds_per_context": pair_rounds,
            "pair_MM_executed_slots_including_self": pair_rounds * batch_count * source_capacity * source_capacity,
            "pair_MM_active_logical_pairs_excluding_self": pair_rounds * active_pair_no_self,
            "pair_MM_active_logical_pairs_including_self": pair_rounds * active_pair_square,
            "pair_ME_executed_slots": pair_rounds * batch_count * source_capacity * environment_capacity,
            "pair_ME_active_logical_pairs": pair_rounds * int((active_by_scene * environment_capacity).sum().item()),
            "pair_EM_executed_slots": pair_rounds * batch_count * environment_capacity * source_capacity,
            "pair_EM_active_logical_pairs": pair_rounds * int((active_by_scene * environment_capacity).sum().item()),
            "pair_padding_source_slots_per_scene": int(batch_count * source_capacity - active_by_scene.sum().item()),
            "source_conditioned_receiver_read_executed_pairs": int(batch_count * query_count * source_capacity),
            "source_conditioned_receiver_read_active_pairs": int(query_count * int(active_by_scene.sum().item())),
            "collective_groups_per_scene": groups,
            "collective_source_membership_score_slots": (0 if not groups else
                int(batch_count * groups * source_capacity)),
            "collective_environment_membership_score_slots": (0 if not groups else
                int(batch_count * groups * environment_capacity)),
            "collective_source_memberships_nonzero": (0 if not torch.is_tensor(source_membership) else
                int(torch.count_nonzero(source_membership > 0).item())),
            "collective_environment_memberships_nonzero": (0 if not torch.is_tensor(environment_membership) else
                int(torch.count_nonzero(environment_membership > 0).item())),
            "collective_receiver_access_score_slots": int(batch_count * query_count * groups),
            "full_access_fallback": 0,
            "sparse_executor_savings_measured": 0,
        })
        return counts

    def optimizer_groups(
        self, model: nn.Module, arm: str = "J-H", stage: str = "joint"
    ) -> Sequence[OptimizerGroupSpec]:
        del arm
        self.phase_metadata(stage)
        if model is not self.model:
            raise ValueError("Wind joint optimizer groups belong to this freshly built model/provider pair.")
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        all_names = tuple(name for name, _ in model.named_parameters())
        if not names or names != all_names:
            raise ValueError("Every Wind joint model parameter must be active in its one optimizer group.")
        schedule = ScheduleSpec(
            peak_lr=3.0e-4,
            warmup_start_lr=3.0e-5,
            warmup_epochs=20,
            hold_through_epoch=1000,
            total_epochs=self.total_epochs,
            final_lr=3.0e-6,
        ) if self.optimizer_schedule is None else self.optimizer_schedule
        weight_decay = 1.0e-5 if self.optimizer_weight_decay is None else self.optimizer_weight_decay
        return (OptimizerGroupSpec("wind_joint_regional", names, schedule, weight_decay=weight_decay),)


def _formal_split_manifest(split: GroupSplit, view: WindFarmNativeView) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    train_rows = np.asarray(split.train, dtype=np.int64)
    validation_rows = np.asarray(split.validation, dtype=np.int64)
    train_layouts = sorted({int(view.metadata["layout_index"][row]) for row in train_rows})
    validation_layouts = sorted({int(view.metadata["layout_index"][row]) for row in validation_rows})
    identity = {
        "subset_id": JOINT_WIND_FORMAL_PROFILE_ID,
        "source_split": split.metadata,
        "train_row_indices": train_rows.astype(int).tolist(),
        "validation_row_indices": validation_rows.astype(int).tolist(),
        "train_layout_indices": train_layouts,
        "validation_layout_indices": validation_layouts,
        "train_row_indices_sha256": _indices_sha256(train_rows),
        "validation_row_indices_sha256": _indices_sha256(validation_rows),
        "used_targets_for_membership": False,
        "wind_test_target_values_read": False,
    }
    identity["manifest_sha256"] = _stable_json_sha256(identity)
    return identity, train_rows, validation_rows


def _full_train_followup_split_manifest(
    split: GroupSplit, view: WindFarmNativeView,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Bind all three original seed-42 memberships without reading TEST targets."""
    manifest, train_rows, validation_rows = _formal_split_manifest(split, view)
    test_rows = np.asarray(split.test, dtype=np.int64)
    manifest = dict(manifest)
    manifest.update({
        "subset_id": JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID,
        "test_row_indices_sha256": _indices_sha256(test_rows),
        "test_row_indices": test_rows.astype(int).tolist(),
        "test_row_count": int(test_rows.size),
        "test_layout_indices": sorted({int(view.metadata["layout_index"][row]) for row in test_rows}),
        "wind_test_target_values_read": False,
        "full_train_followup1000": True,
        "dataset_protocol": FULL_TRAIN_FOLLOWUP_DATASET,
    })
    manifest["manifest_sha256"] = _stable_json_sha256(
        {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    )
    return manifest, train_rows, validation_rows, test_rows


def build_wind_joint_task(
    mode: str,
    *,
    device: torch.device | str = "cpu",
    microbatch_size: int = 4,
    effective_batch_size: int = 24,
    primary_queries: int = 4096,
    total_epochs: int = 2500,
    receiver_tile: int = 512,
    seed: int = 42,
    hidden: int = 128,
    message: int = 128,
    regional_anchors: int = 32,
    depth: int = 2,
    locality_prior_strength: float = 0.0,
    formal_full: bool = False,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    derived_root: str | Path = DEFAULT_DERIVED_ROOT,
    pilot_root: str | Path = DEFAULT_PILOT_ROOT,
    manifest_path: str | Path | None = None,
    normalization_path: str | Path | None = None,
    normalization_binding_path: str | Path | None = None,
    role_scale_cache_dir: str | Path = DEFAULT_JOINT_WIND_ROLE_SCALE_CACHE_DIR,
    catalogue_cache_bytes: int = DEFAULT_ROLE_CATALOGUE_CACHE_MAX_BYTES,
    optimizer_schedule: ScheduleSpec | None = None,
    weight_decay: float | None = None,
    validation_scope: str | None = None,
    native_sampling_protocol: str | None = None,
    environment_token_shape: Sequence[int] | None = None,
    collective_width: int = 64,
    max_sources: int = 30,
    full_train_followup1000: bool = False,
) -> tuple[WindFarmJointRegionalModel, WindJointRegionalTask]:
    """Build a fresh Wind model and provider from the sealed development or formal split.

    ``formal_full=True`` fits its normalization/profile from all original TRAIN
    rows and retains the original validation split; WindTEST targets are never
    sampled. It only prepares the data/model/provider and never starts training.
    """

    mode = str(mode)
    all_modes = {"J-H", "J-geometry", "J-direct", *INTERACTION_PRESERVING_WIND_MODES}
    if mode not in all_modes:
        raise ValueError(f"Wind joint mode must be one of {sorted(all_modes)}.")
    if int(seed) != 42:
        raise ValueError("Wind joint initialization and sampling are sealed to seed 42.")
    if int(effective_batch_size) != 24:
        raise ValueError("Wind joint comparison uses effective batch size 24.")
    if int(microbatch_size) <= 0 or int(microbatch_size) > 24:
        raise ValueError("Wind joint microbatch size must lie in [1,24].")
    if int(primary_queries) < len(ROLE_NAMES):
        raise ValueError("Wind joint primary query count must include every native role.")
    if type(full_train_followup1000) is not bool:
        raise TypeError("Wind full_train_followup1000 must be boolean.")
    if int(total_epochs) not in (2500, 5000):
        raise ValueError("Wind joint supports the 2500 development or 5000 formal/follow-up schedule horizon.")
    if full_train_followup1000:
        if formal_full or int(total_epochs) != 5000 or mode not in INTERACTION_PRESERVING_WIND_MODES:
            raise ValueError("Wind full follow-up requires non-formal P-family identity and 5000-epoch horizon.")
    elif bool(formal_full) != (int(total_epochs) == 5000):
        raise ValueError("The 5000-epoch horizon is reserved for the separate formal_full identity.")
    _validate_formal_controls(
        formal_full=bool(formal_full or full_train_followup1000),
        total_epochs=int(total_epochs),
        optimizer_schedule=optimizer_schedule,
        weight_decay=weight_decay,
        validation_scope=validation_scope,
        native_sampling_protocol=native_sampling_protocol,
        allow_development_optimizer_controls=mode in INTERACTION_PRESERVING_WIND_MODES,
    )
    locality_prior_strength = float(locality_prior_strength)
    if not np.isfinite(locality_prior_strength) or locality_prior_strength < 0.0:
        raise ValueError("Wind locality_prior_strength must be finite and nonnegative.")
    if mode in {"J-H", "J-geometry", "J-direct"} and locality_prior_strength > 0.0 and mode != "J-H":
        raise ValueError("Wind joint locality prior is supported only for J-H.")
    if mode == "P" and locality_prior_strength != 0.0:
        raise ValueError("Wind P has no collective locality prior.")
    if mode in ("P-G", "P-H") and locality_prior_strength != 1.0:
        raise ValueError("The recovery Wind P-G/P-H geometric locality prior is sealed to 1.0.")
    if mode in INTERACTION_PRESERVING_WIND_MODES and bool(formal_full) and not full_train_followup1000:
        raise ValueError("P-family full-TRAIN recipes remain unavailable until separately justified and authorized.")
    resolved_environment_shape = tuple(int(value) for value in (
        environment_token_shape if environment_token_shape is not None else
        ((2, 2, 2) if mode in INTERACTION_PRESERVING_WIND_MODES else JOINT_WIND_ENVIRONMENT_SHAPE)
    ))
    if len(resolved_environment_shape) != 3 or any(value < 1 for value in resolved_environment_shape):
        raise ValueError("Wind environment_token_shape must contain three positive dimensions.")
    if mode in INTERACTION_PRESERVING_WIND_MODES and resolved_environment_shape != (2, 2, 2):
        raise ValueError("The Wind P-family recipe is bound to the deployed E8 token shape (2,2,2).")
    device = torch.device(device)
    data_root = Path(data_root).expanduser().resolve()
    derived_root = Path(derived_root).expanduser().resolve()
    pilot_root = Path(pilot_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Native WindFarm data root is unavailable: {data_root}")
    default_manifest_path = pilot_root / "fixed_subset_manifest.json"
    default_normalization_path = pilot_root / "train_only_normalization.json"
    default_binding_path = pilot_root / "train_only_normalization.binding.json"
    manifest_path = Path(manifest_path or default_manifest_path).expanduser().resolve()
    normalization_path = Path(normalization_path or default_normalization_path).expanduser().resolve()
    normalization_binding_path = Path(normalization_binding_path or default_binding_path).expanduser().resolve()

    view = WindFarmNativeView(
        data_root,
        allow_npz_metadata_fallback=True,
        token_shape=resolved_environment_shape,
    )
    split = _load_original_split(view, derived_root)
    if formal_full or full_train_followup1000:
        if full_train_followup1000:
            manifest, train_rows, validation_rows, test_rows = _full_train_followup_split_manifest(split, view)
            validate_full_train_followup_membership(
                train_rows, validation_rows, test_rows,
                train_layout_count=len(manifest["train_layout_indices"]),
                validation_layout_count=len(manifest["validation_layout_indices"]),
                test_layout_count=len(manifest["test_layout_indices"]),
            )
            selected_profile_id = JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID
        else:
            manifest, train_rows, validation_rows = _formal_split_manifest(split, view)
            selected_profile_id = JOINT_WIND_FORMAL_PROFILE_ID
        if validation_scope == "fullVALID90":
            validation_layout_count = len({int(view.metadata["layout_index"][row]) for row in validation_rows})
            if validation_rows.size != 90 or validation_layout_count != 30:
                raise ValueError("Wind fullVALID90 scope requires the original 90 rows across 30 validation layouts.")
        train_layouts = manifest["train_layout_indices"]
        normalizer, profile, transform_cache_binding = _fit_or_load_formal_transforms(
            view=view,
            train_rows=train_rows,
            train_layouts=train_layouts,
            manifest=manifest,
            training_fingerprint=str(manifest["manifest_sha256"]),
            seed=seed,
            cache_dir=Path(role_scale_cache_dir).expanduser().resolve(),
            profile_id=selected_profile_id,
        )
        normalizer_source = {
            "kind": ("fresh_full_followup1000_original_train_fit" if full_train_followup1000
                     else "fresh_full_original_train_fit"),
            "training_rows_sha256": _indices_sha256(train_rows),
            "source_rows": int(train_rows.size),
            "samples_per_row": int(normalizer.sample_count_per_row),
            "seed": int(seed),
            "normalizer": normalizer.to_dict(),
            "background_profile": profile.to_dict(),
            "source_bound_cache": transform_cache_binding,
        }
        if full_train_followup1000:
            normalizer_source.update({
                "dataset_protocol": FULL_TRAIN_FOLLOWUP_DATASET,
                "normalization_fit_scope": "exact original420 TRAIN rows only",
                "unexposed_original_test_targets_read": False,
                "test_membership_sha256": manifest["test_row_indices_sha256"],
            })
        profile_id = selected_profile_id
    else:
        manifest, train_rows, validation_rows = _read_fixed_rows(view, split, manifest_path)
        normalizer, profile = read_normalization_json(normalization_path)
        if profile is None:
            raise ValueError("Wind joint development requires its already-fitted TRAIN height profile.")
        if normalizer.source_rows != 72 or normalizer.sample_count_per_row != 8192 or normalizer.seed != seed:
            raise ValueError("Wind fixed24 normalization differs from its 72-row TRAIN-only fitting contract.")
        binding = json.loads(normalization_binding_path.read_text(encoding="utf-8"))
        expected_binding = {
            "subset_manifest_sha256": manifest["manifest_sha256"],
            "training_row_indices_sha256": _indices_sha256(train_rows),
            "normalizer_sha256": _sha256(normalization_path),
        }
        if binding != expected_binding:
            raise ValueError("Wind fixed24 transforms are not bound to the exact selected TRAIN rows.")
        normalizer_source = {
            "kind": "verified_fixed24_train_fit",
            "normalization_path": str(normalization_path),
            "normalization_sha256": _sha256(normalization_path),
            "binding_path": str(normalization_binding_path),
            "binding": binding,
        }
        profile_id = JOINT_WIND_PROFILE_ID
        train_layouts = manifest["train_layout_indices"]

    if profile is None:
        raise ValueError("Wind joint model requires a TRAIN-fitted background profile.")
    training_fingerprint = str(manifest.get("manifest_sha256", _stable_json_sha256({
        "profile": profile_id,
        "train_row_indices_sha256": _indices_sha256(train_rows),
    })))
    role_scale_fingerprint = _formal_sampling_fingerprint(
        native_sampling_protocol,
        manifest_fingerprint=training_fingerprint,
        train_rows=train_rows,
    )
    catalogue_cache = NativeRoleCatalogueCache(
        persistent_dir=training_catalogue_cache_directory(),
        max_cached_bytes=int(catalogue_cache_bytes),
    )
    # The corrected role/component scales are an environment-independent
    # TRAIN-profile residual calibration. Reuse the existing sealed E64
    # cache namespace for P-family runs so the five physical role objectives
    # remain exactly comparable with the mature J controls; only the model
    # scene itself uses the deployed E8 representation.
    scale_view = view
    if mode in INTERACTION_PRESERVING_WIND_MODES:
        scale_view = WindFarmNativeView(
            data_root,
            allow_npz_metadata_fallback=True,
            token_shape=JOINT_WIND_ENVIRONMENT_SHAPE,
        )
    calibration, cache_identity = _fit_or_load_role_scales(
        view=scale_view,
        train_rows=train_rows,
        train_layouts=train_layouts,
        profile=profile,
        normalizer=normalizer,
        manifest=manifest,
        training_fingerprint=role_scale_fingerprint,
        catalogue_cache=catalogue_cache,
        cache_dir=Path(role_scale_cache_dir).expanduser().resolve(),
        native_sampling_protocol=native_sampling_protocol,
    )
    role_scales = {
        role: tuple(float(value) for value in calibration["component_role_scales_mps"][role])
        for role in ROLE_NAMES
    }
    scale_sha = _stable_json_sha256({"calibration": calibration, "cache_binding": cache_identity})
    if not scale_sha:
        raise RuntimeError("Wind joint TRAIN role-scale calibration did not produce a stable identity.")
    normalizer_source["profile_sha256"] = _stable_json_sha256(profile.to_dict())

    devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        devices = [torch.cuda.current_device() if device.index is None else int(device.index)]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        model_factory = (
            InteractionPreservingWindModel
            if mode in INTERACTION_PRESERVING_WIND_MODES
            else WindFarmJointRegionalModel
        )
        model_options = {
            "velocity_transform": normalizer,
            "background_profile": profile,
            "mode": mode,
            "hidden": hidden,
            "message": message,
            "regional_anchors": regional_anchors,
            "depth": depth,
            "receiver_tile": receiver_tile,
            "seed": seed,
        }
        if mode in INTERACTION_PRESERVING_WIND_MODES:
            model_options.update({
                "collective_width": collective_width,
                "max_sources": max_sources,
                "locality_prior_strength": (locality_prior_strength if mode != "P" else None),
            })
        else:
            model_options["locality_prior_strength"] = locality_prior_strength
        model = model_factory(**model_options).to(device)
    provider = WindJointRegionalTask(
        view,
        model=model,
        train_rows=train_rows,
        validation_rows=validation_rows,
        manifest=manifest,
        profile_id=profile_id,
        training_fingerprint=training_fingerprint,
        role_component_scales=role_scales,
        role_scale_calibration=calibration,
        role_scale_cache_identity=cache_identity,
        role_scale_source_sha256=scale_sha,
        normalizer_source=normalizer_source,
        seed=seed,
        primary_queries=primary_queries,
        microbatch_size=microbatch_size,
        effective_batch_size=effective_batch_size,
        total_epochs=total_epochs,
        device=device,
        catalogue_cache=catalogue_cache,
        optimizer_schedule=optimizer_schedule,
        weight_decay=weight_decay,
        validation_scope=validation_scope,
        native_sampling_protocol=native_sampling_protocol,
        environment_token_shape=resolved_environment_shape,
        full_train_followup1000=full_train_followup1000,
    )
    return model, provider


__all__ = [
    "DEFAULT_JOINT_WIND_ROLE_SCALE_CACHE_DIR",
    "JOINT_WIND_ENVIRONMENT_COUNT",
    "JOINT_WIND_ENVIRONMENT_SHAPE",
    "WindJointPredictions",
    "WindJointReceivers",
    "WindJointRegionalTask",
    "WindJointSceneInputs",
    "WindJointTargets",
    "build_wind_joint_task",
]
