"""Native Wind provider for the shared interaction-refinement trainer.

Scene inputs, receiver queries, and native velocity targets use separate
records. Native role catalogues are geometry-only and cached once; target
values are indexed only after a fresh epoch-keyed query sample is selected.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_forward_core.interface_fields.interaction_core import (
    DependencySpec,
    InteractionContextCore,
    InteractionScene,
)
from honf_runtime.unified_training import (
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    TaskBatch,
)
from torch import nn

from ..data import WindFarmNativeView
from ..model import REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE, build_windfarm_model
from ..normalization import (
    VelocityNormalizer,
    VerticalProfileBaseline,
    read_normalization_json,
)
from ..splits import GroupSplit, make_group_split
from ..workflows.directed_packet_pair import ACTIVE_REUSE_CONFIG
from ..workflows.joint_forward import (
    DEFAULT_ROLE_CATALOGUE_CACHE_MAX_BYTES,
    ROLE_NAMES,
    NativeRoleCatalogueCache,
    sample_native_role_queries,
)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "Dataset/links/wind_farm"
DEFAULT_DERIVED_ROOT = Path(__file__).resolve().parents[3] / "Dataset/derived/forward_velocity_v1"
DEFAULT_PILOT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "diagnostics/generated/shared_interaction_20261007/wind_pilot"
)
DEFAULT_ROLE_QUERY_COUNTS = {
    "volume": 205,
    "hub_slab": 205,
    "downstream_envelope": 205,
    "near_turbine": 205,
    "background": 204,
}
WIND_W0_RECIPE_ID = "wind_w0_scalar_q1024_v1"
WIND_W1_RECIPE_ID = "wind_w1_component_q1024_v1"
WIND_W2_RECIPE_ID = "wind_w2_component_q4096_v1"
WIND_W3_RECIPE_ID = "wind_w3_component_q1024_h128_v1"
WIND_RECIPE_SCHEMA_VERSION = 1
WIND_GATE_HARD_VERSION = "hard_v1"
WIND_GATE_COMPACT_C1_VERSION = "compact_c1_v1"
WIND_GATE_COMPACT_C1_TRANSITION = (0.35, 0.65)
WIND_ROLE_SCALE_CALIBRATION_QUERY_COUNTS = dict(DEFAULT_ROLE_QUERY_COUNTS)
WIND_COMPONENT_SCALE_MINIMUM_MPS = 1.0e-3
WIND_COMPONENT_SCALE_FRACTION_OF_SCALAR = 0.10
WIND_ALLOWED_QUERY_TILES = (512, 2048, 8192)
WIND_ALLOWED_CONTEXT_SHAPES = ((2, 2, 2), (8, 4, 2))
_ROLE_INDEX = {name: index for index, name in enumerate(ROLE_NAMES)}
_QUERY_STREAM = 0x57494E44
_MESSAGE_CALIBRATION_STREAM = 0x4D534753
_COST_CALIBRATION_STAGE = "expected_work_calibration"
_MESSAGE_CALIBRATION_LAYOUT_COUNT = 4
_MESSAGE_APPROXIMATION_COEFFICIENT = 0.1
_ROUTER_IMPORTANCE_COEFFICIENT = 0.01
_EXPECTED_WORK_TARGET_GRADIENT_SHARE = 0.05
_EXPECTED_WORK_COEFFICIENT_CAP = 0.1
_WIND_DEPENDENCY = DependencySpec(
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
        ("turbine_geometry", "source_context"),
        ("environment_geometry_and_measure", "source_context"),
        ("wind_direction", "global_context"),
        ("global_context", "velocity_field"),
    ),
    prepared_nodes=("source_context", "global_context"),
)


def _role_counts_for_query_count(query_count: int) -> dict[str, int]:
    """Scale the sealed five-role mixture with deterministic largest remainders."""

    query_count = int(query_count)
    if query_count < len(ROLE_NAMES):
        raise ValueError("Wind query count must provide at least one draw for every protected role.")
    base_total = sum(DEFAULT_ROLE_QUERY_COUNTS.values())
    exact = {
        name: query_count * int(DEFAULT_ROLE_QUERY_COUNTS[name]) / base_total
        for name in ROLE_NAMES
    }
    counts = {name: int(math.floor(exact[name])) for name in ROLE_NAMES}
    remainder = query_count - sum(counts.values())
    order = sorted(ROLE_NAMES, key=lambda name: (-(exact[name] - counts[name]), ROLE_NAMES.index(name)))
    for name in order[:remainder]:
        counts[name] += 1
    if sum(counts.values()) != query_count or any(count <= 0 for count in counts.values()):
        raise RuntimeError("Wind query role allocation failed its exact total contract.")
    return counts


def _builtin_wind_recipe(recipe_id: str) -> dict[str, Any]:
    """Return a versioned fresh-init screen recipe or the selected width follow-up."""

    definitions = {
        WIND_W0_RECIPE_ID: (1024, "scalar_role", "train_role_target_component_variance_v1", 64, 64),
        WIND_W1_RECIPE_ID: (1024, "component_role", "train_role_profile_residual_rms_v1", 64, 64),
        WIND_W2_RECIPE_ID: (4096, "component_role", "train_role_profile_residual_rms_v1", 64, 64),
        WIND_W3_RECIPE_ID: (1024, "component_role", "train_role_profile_residual_rms_v1", 128, 128),
    }
    if recipe_id not in definitions:
        raise ValueError(f"Unknown versioned Wind development recipe {recipe_id!r}.")
    query_count, objective, scale_rule, hidden, message = definitions[recipe_id]
    return {
        "schema_version": WIND_RECIPE_SCHEMA_VERSION,
        "recipe_id": recipe_id,
        "dataset_profile": "wind_shared_fixed24_v1",
        "objective": objective,
        "loss_scale_rule": scale_rule,
        "query_count": query_count,
        "role_query_counts": _role_counts_for_query_count(query_count),
        "calibration_query_count": 1024,
        "calibration_role_query_counts": dict(WIND_ROLE_SCALE_CALIBRATION_QUERY_COUNTS),
        "calibration_layout_count": _MESSAGE_CALIBRATION_LAYOUT_COUNT,
        "model": {
            "forward_architecture": REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE,
            "hidden": hidden,
            "message": message,
            "environment_token_shape": [2, 2, 2],
            "max_sources": 30,
            "base_width": 16,
            "router_hidden": 32,
        },
        "execution_backend": "selected",
        "query_tile_size": 512,
        "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
    }


def resolve_wind_recipe(
    recipe_id: str | None = None,
    *,
    recipe: Mapping[str, Any] | None = None,
    execution_backend: str | None = None,
    query_tile_size: int | None = None,
) -> dict[str, Any] | None:
    """Resolve a strict immutable recipe payload; ``None`` keeps the old profile."""

    if recipe is None and recipe_id is None:
        return None
    if recipe is not None:
        resolved = dict(recipe)
        saved_sha = resolved.pop("recipe_sha256", None)
        if saved_sha is not None and saved_sha != _stable_json_sha256(resolved):
            raise ValueError("Resolved Wind recipe SHA256 does not verify.")
        required = {
            "schema_version", "recipe_id", "dataset_profile", "objective", "loss_scale_rule",
            "query_count", "role_query_counts", "calibration_query_count", "calibration_role_query_counts",
            "calibration_layout_count", "model", "execution_backend", "query_tile_size", "sampling_version",
        }
        fields = set(resolved)
        gate_fields = {"gate_version", "gate_transition"}
        if fields not in (required, required | {"gate_version"}, required | gate_fields):
            raise ValueError("Resolved Wind recipe fields do not match the strict recipe schema.")
        if "gate_transition" in fields and "gate_version" not in fields:
            raise ValueError("A Wind gate transition requires an explicit gate version.")
        if recipe_id is not None and resolved["recipe_id"] != recipe_id:
            raise ValueError("Wind recipe ID does not match the supplied resolved recipe payload.")
    else:
        assert recipe_id is not None
        resolved = _builtin_wind_recipe(str(recipe_id))
    if int(resolved["schema_version"]) != WIND_RECIPE_SCHEMA_VERSION:
        raise ValueError("Unsupported Wind recipe schema version.")
    if resolved["dataset_profile"] != "wind_shared_fixed24_v1":
        raise ValueError("Versioned Wind development recipes must use wind_shared_fixed24_v1.")
    if resolved["objective"] not in {"scalar_role", "component_role"}:
        raise ValueError("Wind recipe objective must be scalar_role or component_role.")
    expected_scale_rule = {
        "scalar_role": "train_role_target_component_variance_v1",
        "component_role": "train_role_profile_residual_rms_v1",
    }[resolved["objective"]]
    if resolved["loss_scale_rule"] != expected_scale_rule:
        raise ValueError("Wind objective and TRAIN-only scale rule do not match the versioned contract.")
    query_count = int(resolved["query_count"])
    role_counts = {str(name): int(value) for name, value in dict(resolved["role_query_counts"]).items()}
    if set(role_counts) != set(ROLE_NAMES) or role_counts != _role_counts_for_query_count(query_count):
        raise ValueError("Wind recipe role counts must preserve the canonical mixture and sum exactly to Q.")
    if int(resolved["calibration_query_count"]) != 1024:
        raise ValueError("Wind scale calibration remains bound to its declared Q=1024 panel.")
    calibration_counts = {
        str(name): int(value) for name, value in dict(resolved["calibration_role_query_counts"]).items()
    }
    if calibration_counts != dict(WIND_ROLE_SCALE_CALIBRATION_QUERY_COUNTS):
        raise ValueError("Wind TRAIN scale calibration role counts are frozen to Q=1024.")
    if int(resolved["calibration_layout_count"]) != _MESSAGE_CALIBRATION_LAYOUT_COUNT:
        raise ValueError("Wind TRAIN scale calibration uses the fixed four-layout panel.")
    model = dict(resolved["model"])
    if set(model) != {
        "forward_architecture", "hidden", "message", "environment_token_shape", "max_sources", "base_width",
        "router_hidden",
    }:
        raise ValueError("Wind recipe model fields do not match the strict source-resolved model schema.")
    if model["forward_architecture"] != REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE:
        raise ValueError("Wind development recipes retain the shared refined nonlinear architecture.")
    if int(model["hidden"]) not in (64, 128) or int(model["message"]) not in (64, 128):
        raise ValueError("Wind recipe hidden and message widths must be 64 or 128.")
    shape = tuple(int(value) for value in model["environment_token_shape"])
    if shape not in WIND_ALLOWED_CONTEXT_SHAPES:
        raise ValueError("Wind environment context must use the declared E8 or E64 token shape.")
    for name in ("max_sources", "base_width", "router_hidden"):
        if int(model[name]) <= 0:
            raise ValueError(f"Wind recipe model field {name!r} must be positive.")
    backend = str(resolved["execution_backend"] if execution_backend is None else execution_backend)
    if backend not in {"selected", "dense_masked"}:
        raise ValueError("Wind execution backend must be selected or dense_masked.")
    tile = int(resolved["query_tile_size"] if query_tile_size is None else query_tile_size)
    if tile not in WIND_ALLOWED_QUERY_TILES:
        raise ValueError(f"Wind query tile size must be one of {WIND_ALLOWED_QUERY_TILES}.")
    if resolved["sampling_version"] != SamplingKey.CASE_EPOCH_VERSION:
        raise ValueError("New Wind recipes require packing-independent case_epoch_v1 sampling.")
    gate_version = str(resolved.get("gate_version", WIND_GATE_HARD_VERSION))
    if gate_version not in {WIND_GATE_HARD_VERSION, WIND_GATE_COMPACT_C1_VERSION}:
        raise ValueError("Wind gate version must be hard_v1 or compact_c1_v1.")
    if gate_version == WIND_GATE_COMPACT_C1_VERSION:
        if "gate_transition" not in resolved:
            raise ValueError("compact_c1_v1 requires its sealed gate_transition interval.")
        transition = tuple(float(value) for value in resolved["gate_transition"])
        if transition != WIND_GATE_COMPACT_C1_TRANSITION:
            raise ValueError("compact_c1_v1 is sealed to the TRAIN-diagnosed [0.35, 0.65] transition.")
        resolved["gate_transition"] = list(transition)
    elif "gate_transition" in resolved:
        raise ValueError("Only compact_c1_v1 may bind a gate_transition interval.")
    resolved["role_query_counts"] = role_counts
    resolved["calibration_role_query_counts"] = calibration_counts
    resolved["model"] = {**model, "environment_token_shape": list(shape)}
    resolved["execution_backend"] = backend
    resolved["query_tile_size"] = tile
    resolved["recipe_sha256"] = _stable_json_sha256(resolved)
    return resolved


@dataclass(frozen=True)
class WindSceneInputs:
    """Whitelisted geometry and prescribed context, with no native target."""

    row_index: int
    case_name: str
    layout_index: int
    wind_direction_deg: float
    module_centers: np.ndarray
    module_present: np.ndarray
    module_features: np.ndarray
    global_context: np.ndarray
    support_lower_D: np.ndarray
    support_upper_D: np.ndarray
    support_extent_D: np.ndarray
    env_coords: np.ndarray
    env_features: np.ndarray
    env_weights: np.ndarray


@dataclass(frozen=True)
class WindReceiverInputs:
    """Fresh physical query coordinates and fixed native-role membership."""

    coordinates_D: np.ndarray
    role_ids: np.ndarray


@dataclass(frozen=True)
class WindTargets:
    """Stored OpenFOAM velocities and metadata kept outside the scene path."""

    velocity_mps: np.ndarray
    row_indices: tuple[int, ...]
    case_metadata: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class WindPredictions:
    """Physical primary and full-detail predictions plus router auxiliaries."""

    main_mps: torch.Tensor
    full_mps: torch.Tensor | None
    role_ids: torch.Tensor
    auxiliary: Mapping[str, Any]
    execution_mode: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _indices_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


def _load_original_split(view: WindFarmNativeView, derived_root: Path) -> GroupSplit:
    groups = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    split = make_group_split(groups, seed=42)
    stored_path = derived_root / "split_indices.npz"
    if stored_path.is_file():
        with np.load(stored_path, allow_pickle=False) as archive:
            if set(archive.files) != {"train", "validation", "test"}:
                raise ValueError("Stored original WindFarm split has an unexpected partition inventory.")
            stored = {name: np.asarray(archive[name], dtype=np.int64) for name in archive.files}
        for name in ("train", "validation", "test"):
            if not np.array_equal(stored[name], getattr(split, name)):
                raise ValueError(f"Stored original WindFarm {name} split differs from the seed-42 group split.")
    return split


def _read_fixed_rows(
    view: WindFarmNativeView,
    split: GroupSplit,
    manifest_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("subset_id") != "wind_shared_fixed24_v1":
        raise ValueError("Wind refinement must reuse wind_shared_fixed24_v1.")
    saved_hash = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not isinstance(saved_hash, str) or _stable_json_sha256(payload) != saved_hash:
        raise ValueError("The fixed Wind subset manifest hash does not verify.")
    if manifest.get("used_targets_for_subset_selection") is not False:
        raise ValueError("Wind subset selection must remain input-only.")
    source_split = manifest.get("source_split")
    if not isinstance(source_split, Mapping):
        raise TypeError("Wind fixed subset source split must be a mapping.")
    if (
        int(source_split.get("seed", -1)) != 42
        or source_split.get("group_key") != "layout_index"
        or source_split.get("test_target_values_read") is not False
    ):
        raise ValueError("Wind subset manifest changed the frozen input-only seed-42 group split.")
    for partition in ("train", "validation"):
        saved_partition = source_split.get(f"{partition}_row_indices_sha256")
        actual_partition = split.metadata["partitions"][partition]["row_indices_sha256"]
        if saved_partition != actual_partition:
            raise ValueError(f"Wind fixed subset source {partition} split hash does not match the native rows.")
    if int(source_split.get("original_train_layout_count", -1)) != int(split.metadata["partitions"]["train"]["groups"]):
        raise ValueError("Wind fixed subset original TRAIN group count changed.")
    if int(source_split.get("original_validation_layout_count", -1)) != int(
        split.metadata["partitions"]["validation"]["groups"]
    ):
        raise ValueError("Wind fixed subset original validation group count changed.")
    if int(source_split.get("original_test_layout_count", -1)) != int(split.metadata["partitions"]["test"]["groups"]):
        raise ValueError("Wind fixed subset original TEST group count changed.")
    train_rows = np.asarray(manifest.get("train_row_indices"), dtype=np.int64)
    validation_rows = np.asarray(manifest.get("validation_row_indices"), dtype=np.int64)
    if train_rows.shape != (72,) or validation_rows.shape != (24,):
        raise ValueError("wind_shared_fixed24_v1 must contain 72 TRAIN and 24 development rows.")
    groups = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    directions = np.asarray(view.metadata["wd_deg"], dtype=np.float64)
    if np.any(train_rows < 0) or np.any(train_rows >= groups.size):
        raise ValueError("Fixed Wind TRAIN rows escape the native metadata inventory.")
    if np.any(validation_rows < 0) or np.any(validation_rows >= groups.size):
        raise ValueError("Fixed Wind validation rows escape the native metadata inventory.")
    if not set(groups[train_rows]).issubset(set(groups[split.train])):
        raise ValueError("Fixed Wind TRAIN layouts escaped the original TRAIN groups.")
    if not set(groups[validation_rows]).issubset(set(groups[split.validation])):
        raise ValueError("Fixed Wind development layouts escaped the original validation groups.")
    if set(groups[train_rows]) & set(groups[validation_rows]):
        raise ValueError("Fixed Wind TRAIN and development layouts overlap.")
    expected_directions = {270.0, 285.0, 300.0}
    for rows, expected_layouts, label in (
        (train_rows, 24, "TRAIN"),
        (validation_rows, 8, "development"),
    ):
        selected_groups = np.unique(groups[rows])
        if selected_groups.size != expected_layouts:
            raise ValueError(f"Fixed Wind {label} selection has {selected_groups.size} layouts, expected {expected_layouts}.")
        for layout in selected_groups:
            selected = rows[groups[rows] == layout]
            if selected.size != 3 or set(directions[selected]) != expected_directions:
                raise ValueError(f"Fixed Wind layout {layout} does not retain all three direction rows.")
    if manifest.get("direction_categories") != [270.0, 285.0, 300.0]:
        raise ValueError("Fixed Wind direction contract changed.")
    if int(manifest.get("train_row_count", -1)) != train_rows.size or int(
        manifest.get("validation_row_count", -1)
    ) != validation_rows.size:
        raise ValueError("Fixed Wind subset row counts disagree with the frozen row arrays.")
    for label, rows, layout_selection, expected_layouts in (
        ("TRAIN", train_rows, "train_layout_selection", 24),
        ("development", validation_rows, "validation_layout_selection", 8),
    ):
        selected = manifest.get(layout_selection)
        if not isinstance(selected, Mapping) or selected.get("used_target_values_for_selection") is not False:
            raise ValueError(f"Fixed Wind {label} layout selection must be explicitly target-free.")
        expected_rows = np.asarray(selected.get("row_indices"), dtype=np.int64)
        expected_group_ids = np.asarray(selected.get("layout_indices"), dtype=np.int64)
        top_level_group_ids = np.asarray(
            manifest.get("train_layout_indices" if label == "TRAIN" else "validation_layout_indices"),
            dtype=np.int64,
        )
        actual_group_ids = np.unique(groups[rows])
        if (
            not np.array_equal(expected_rows, rows)
            or not np.array_equal(expected_group_ids, actual_group_ids)
            or not np.array_equal(top_level_group_ids, actual_group_ids)
            or actual_group_ids.size != expected_layouts
        ):
            raise ValueError(f"Fixed Wind {label} nested layout/row inventory differs from its top-level binding.")
    return manifest, train_rows, validation_rows


def _copy_array(value: Any, dtype: np.dtype[Any] | type = np.float32) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    result.setflags(write=False)
    return result


def _scene_inputs(case: Any) -> WindSceneInputs:
    """Copy only native geometry/context fields; never read ``run.U``."""

    support = case.support
    return WindSceneInputs(
        row_index=int(case.index),
        case_name=str(case.case),
        layout_index=int(case.layout_index),
        wind_direction_deg=float(case.wind_direction_deg),
        module_centers=_copy_array(case.module_centers),
        module_present=_copy_array(case.module_present),
        module_features=_copy_array(case.module_features),
        global_context=_copy_array(case.global_context),
        support_lower_D=_copy_array(support.lower_D),
        support_upper_D=_copy_array(support.upper_D),
        support_extent_D=_copy_array(support.extent_D),
        env_coords=_copy_array(case.env_coords),
        env_features=_copy_array(case.env_features),
        env_weights=_copy_array(case.env_weights),
    )


def _role_ids(counts: Mapping[str, int]) -> np.ndarray:
    return np.concatenate(
        [np.full(int(counts[name]), _ROLE_INDEX[name], dtype=np.int8) for name in ROLE_NAMES]
    )


def _query_sampling_sha256(
    rows: Sequence[int], samples: Sequence[Any], key: SamplingKey
) -> str:
    """Bind each fresh geometry-only query selection to its arm-free stream key."""

    if len(rows) != len(samples):
        raise ValueError("Wind query receipt rows and sampled panels must have equal lengths.")
    digest = hashlib.sha256()
    if getattr(key, "sampling_version", SamplingKey.LEGACY_VERSION) == SamplingKey.CASE_EPOCH_VERSION:
        stream = {
            "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            "dataset_id": str(key.dataset_id),
            "seed": int(key.seed),
            "epoch": int(key.epoch),
        }
    else:
        # Preserve the historical receipt definition for sealed legacy runs.
        stream = {
            "seed": int(key.seed),
            "epoch": int(key.epoch),
            "update_index": int(key.update_index),
            "microbatch_index": int(key.microbatch_index),
            "stage": str(key.stage),
        }
    digest.update(json.dumps(stream, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for row, sample in zip(rows, samples):
        role_ids = _role_ids(sample.role_sample_counts)
        coordinates = np.ascontiguousarray(sample.coordinates_D, dtype=np.float32)
        flat_indices = np.ascontiguousarray(sample.flat_indices, dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or flat_indices.shape != (coordinates.shape[0],) or role_ids.shape != (coordinates.shape[0],):
            raise ValueError("Wind query receipt requires aligned native cell IDs, coordinates, and roles.")
        digest.update(int(row).to_bytes(8, "little", signed=False))
        if getattr(key, "sampling_version", SamplingKey.LEGACY_VERSION) == SamplingKey.CASE_EPOCH_VERSION:
            digest.update(json.dumps(sample.role_sample_counts, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(flat_indices.tobytes())
        digest.update(coordinates.tobytes())
        digest.update(np.ascontiguousarray(role_ids, dtype=np.int8).tobytes())
    return digest.hexdigest()


def _query_rng(key: SamplingKey, *coordinates: Any) -> np.random.Generator:
    """Use the engine's arm-independent stream when available."""

    numpy_rng = getattr(key, "numpy_rng", None)
    if callable(numpy_rng):
        return numpy_rng(*coordinates)
    seed_for = getattr(key, "seed_for", None)
    if callable(seed_for):
        return np.random.default_rng(seed_for(*coordinates))
    # Transitional compatibility for the engine revision before SamplingKey
    # gained named streams. The arm is deliberately excluded here as well.
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                int(key.seed),
                int(key.epoch),
                int(key.update_index),
                int(key.microbatch_index),
                *[int(value) for value in coordinates if isinstance(value, (int, np.integer))],
                _QUERY_STREAM,
            ]
        )
    )


def _role_counts_from_config(config: Mapping[str, Any]) -> dict[str, int]:
    counts = dict(DEFAULT_ROLE_QUERY_COUNTS)
    if "role_query_counts" in config and dict(config["role_query_counts"]) != counts:
        raise ValueError("Wind refinement uses the sealed Q=1024 five-role query allocation.")
    if sum(counts.values()) != 1024:
        raise RuntimeError("Wind role query counts no longer sum to Q=1024.")
    return counts


def _sampling_dataset_id(dataset: str, training_fingerprint: str) -> str:
    payload = json.dumps([str(dataset), str(training_fingerprint)], sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"{dataset}:{hashlib.sha256(payload).hexdigest()}"


def _physical_case_direction_id(case: Any) -> str:
    """Return a stable native row identity that includes its operating direction."""

    return f"{case.case}|layout={int(case.layout_index)}|wd={float(case.wind_direction_deg):.6f}"


def _role_scales(path: Path) -> tuple[dict[str, float], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    raw = config["forward"]["stage_a"]["role_loss_scales_mps"]
    scales = {str(name): float(value) for name, value in raw.items()}
    if set(scales) != set(ROLE_NAMES) or any(not math.isfinite(value) or value <= 0 for value in scales.values()):
        raise ValueError("Wind role loss scales are incomplete or invalid.")
    return scales, _sha256(path)


def _as_scene_batch(inputs: Sequence[WindSceneInputs], device: torch.device) -> InteractionScene:
    if not inputs:
        raise ValueError("Wind scene batches cannot be empty.")
    max_sources = max(int(item.module_centers.shape[0]) for item in inputs)
    if max_sources > 30:
        raise ValueError("Wind source capacity exceeds the sealed native H64 recipe.")
    batch = len(inputs)
    source_width = int(inputs[0].module_features.shape[-1])
    context_width = int(inputs[0].global_context.size)
    environment_width = int(inputs[0].env_features.shape[-1])
    environment_count = int(inputs[0].env_coords.shape[0])
    if any(int(item.env_coords.shape[0]) != environment_count for item in inputs):
        raise ValueError("Wind batches must retain one consistent environmental quadrature inventory.")
    if environment_count != 8:
        raise ValueError("The fresh Wind refinement recipe retains exactly eight native environment records.")
    if source_width != 2 or environment_width != 7 or context_width != 11:
        raise ValueError("Wind native scene feature widths differ from the sealed H64/message64 adapter contract.")
    centers = torch.zeros((batch, max_sources, 3), device=device, dtype=torch.float32)
    sources = torch.zeros((batch, max_sources, source_width), device=device, dtype=torch.float32)
    present = torch.zeros((batch, max_sources), device=device, dtype=torch.float32)
    source_lengths = torch.ones((batch, max_sources), device=device, dtype=torch.float32)
    source_measures = torch.zeros_like(present)
    source_ids = torch.full((batch, max_sources), -1, device=device, dtype=torch.long)
    contexts = torch.zeros((batch, context_width), device=device, dtype=torch.float32)
    lengths = torch.zeros((batch, 3), device=device, dtype=torch.float32)
    environment_tokens = torch.zeros((batch, environment_count, environment_width), device=device, dtype=torch.float32)
    environment_coords = torch.zeros((batch, environment_count, 3), device=device, dtype=torch.float32)
    environment_measures = torch.zeros((batch, environment_count), device=device, dtype=torch.float32)

    for index, item in enumerate(inputs):
        count = int(item.module_centers.shape[0])
        if item.module_centers.shape != (count, 3) or item.module_features.shape != (count, source_width):
            raise ValueError("Wind scene source arrays have inconsistent shapes.")
        if item.module_present.shape != (count,) or np.any((item.module_present < 0) | (item.module_present > 1)):
            raise ValueError("Wind scene source presence must align with physical source slots in [0,1].")
        if (
            item.global_context.shape != (context_width,)
            or item.support_lower_D.shape != (3,)
            or item.support_upper_D.shape != (3,)
            or item.support_extent_D.shape != (3,)
        ):
            raise ValueError("Wind scene context/support arrays have inconsistent shapes.")
        if item.env_coords.shape != (environment_count, 3) or item.env_features.shape != (
            environment_count,
            environment_width,
        ):
            raise ValueError("Wind environmental context arrays have inconsistent shapes.")
        if item.env_weights.shape != (environment_count,) or np.any(item.env_weights < 0):
            raise ValueError("Wind environmental measures must be nonnegative and align with all eight records.")
        centers[index, :count] = torch.tensor(item.module_centers, device=device)
        sources[index, :count] = torch.tensor(item.module_features, device=device)
        present[index, :count] = torch.tensor(item.module_present, device=device)
        source_lengths[index, :count] = 2.0 * sources[index, :count, 0]
        source_measures[index, :count] = present[index, :count]
        source_ids[index, :count] = torch.arange(count, device=device)
        contexts[index] = torch.tensor(item.global_context, device=device)
        lengths[index] = torch.tensor(item.support_extent_D, device=device)
        environment_tokens[index] = torch.tensor(item.env_features, device=device)
        environment_coords[index] = torch.tensor(item.env_coords, device=device)
        environment_measures[index] = torch.tensor(item.env_weights, device=device)

    if not bool(
        torch.isfinite(
            torch.cat(
                (
                    centers.flatten(),
                    sources.flatten(),
                    contexts.flatten(),
                    environment_tokens.flatten(),
                    environment_coords.flatten(),
                    environment_measures.flatten(),
                    lengths.flatten(),
                )
            ).all()
        )
    ):
        raise ValueError("Wind scene geometry/features contain nonfinite values.")
    if bool((lengths <= 0).any()) or bool((source_lengths[present > 0] <= 0).any()):
        raise ValueError("Wind scene lengths must be positive for active sources.")
    return InteractionScene(
        sources=sources,
        context=contexts,
        centers=centers,
        present=present,
        lengths=lengths,
        source_lengths=source_lengths,
        dependency=_WIND_DEPENDENCY,
        source_measures=source_measures,
        environment_tokens=environment_tokens,
        environment_coords=environment_coords,
        environment_present=torch.ones_like(environment_measures),
        environment_measures=environment_measures,
        source_ids=source_ids,
    )


def _receiver_features(scene: InteractionScene, receivers: torch.Tensor) -> torch.Tensor:
    scale = receivers.new_tensor([50.0, 38.0, 6.25])
    lower = scene.context[:, 5:8] * scale
    extent = scene.context[:, 8:11] * scale
    upper = lower + extent
    lower_distance = (receivers - lower[:, None]) / scale
    upper_distance = (upper[:, None] - receivers) / scale
    absolute_height = receivers[..., 2:3] / scale[2]
    return torch.cat((lower_distance, upper_distance, absolute_height), dim=-1)


class WindRefinementTask:
    """Source-resolved Wind adapter implementing the common TaskProvider API."""

    max_microbatch_cases = 4

    def __init__(
        self,
        view: WindFarmNativeView,
        *,
        train_rows: np.ndarray,
        validation_rows: np.ndarray,
        normalizer: VelocityNormalizer,
        background_profile: VerticalProfileBaseline,
        manifest: Mapping[str, Any],
        manifest_path: Path,
        normalization_path: Path,
        role_scales: Mapping[str, float],
        role_scale_sha256: str,
        seed: int = 42,
        device: torch.device | str = "cpu",
        role_query_counts: Mapping[str, int] = DEFAULT_ROLE_QUERY_COUNTS,
        catalogue_cache_bytes: int = DEFAULT_ROLE_CATALOGUE_CACHE_MAX_BYTES,
        message_width: int = 64,
        objective_weights: Mapping[str, float] | None = None,
        role_component_scales: Mapping[str, Sequence[float]] | None = None,
        role_scale_calibration: Mapping[str, Any] | None = None,
        role_scale_profile: Mapping[str, Any] | None = None,
        recipe: Mapping[str, Any] | None = None,
        sampling_dataset_fingerprint: str | None = None,
        catalogue_cache: NativeRoleCatalogueCache | None = None,
    ) -> None:
        self.view = view
        self.train_rows = np.asarray(train_rows, dtype=np.int64)
        self.validation_rows = np.asarray(validation_rows, dtype=np.int64)
        self.normalizer = normalizer
        self.background_profile = background_profile
        self.manifest = dict(manifest)
        self.manifest_path = manifest_path.resolve()
        self.normalization_path = normalization_path.resolve()
        self.role_scales = {name: float(role_scales[name]) for name in ROLE_NAMES}
        self.role_scale_sha256 = str(role_scale_sha256)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.role_query_counts = {name: int(role_query_counts[name]) for name in ROLE_NAMES}
        self.validation_role_query_counts = (
            dict(DEFAULT_ROLE_QUERY_COUNTS)
            if recipe is not None
            else dict(self.role_query_counts)
        )
        self.message_width = int(message_width)
        self.recipe = None if recipe is None else dict(recipe)
        self.gate_version = (
            WIND_GATE_HARD_VERSION if self.recipe is None else str(self.recipe.get("gate_version", WIND_GATE_HARD_VERSION))
        )
        self.gate_transition = (
            WIND_GATE_COMPACT_C1_TRANSITION
            if self.recipe is None or "gate_transition" not in self.recipe
            else tuple(float(value) for value in self.recipe["gate_transition"])
        )
        self.execution_backend = "selected" if self.recipe is None else str(self.recipe["execution_backend"])
        self.query_tile_size = 512 if self.recipe is None else int(self.recipe["query_tile_size"])
        self.sampling_version = (
            SamplingKey.LEGACY_VERSION if self.recipe is None else str(self.recipe["sampling_version"])
        )
        self.sampling_dataset_id = (
            ""
            if self.sampling_version == SamplingKey.LEGACY_VERSION
            else _sampling_dataset_id(
                "WindFarm",
                str(sampling_dataset_fingerprint or self.manifest.get("manifest_sha256", "")),
            )
        )
        self.role_scale_calibration = None if role_scale_calibration is None else dict(role_scale_calibration)
        self.role_scale_profile = None if role_scale_profile is None else dict(role_scale_profile)
        self.role_component_scales = {
            name: np.asarray(
                [self.role_scales[name]] * 3
                if role_component_scales is None
                else role_component_scales[name],
                dtype=np.float64,
            )
            for name in ROLE_NAMES
        }
        if catalogue_cache is not None and catalogue_cache.max_cached_bytes != int(catalogue_cache_bytes):
            raise ValueError("Injected Wind catalogue cache capacity differs from the provider binding.")
        self.catalogue_cache = catalogue_cache or NativeRoleCatalogueCache(max_cached_bytes=int(catalogue_cache_bytes))
        self.max_microbatch_cases = 24 if self.recipe is not None else 4
        requested_weights = dict(objective_weights or {})
        if requested_weights and requested_weights != {
            "base_loss": _MESSAGE_APPROXIMATION_COEFFICIENT,
            "router_importance_loss": _ROUTER_IMPORTANCE_COEFFICIENT,
        }:
            raise ValueError("Wind refinement auxiliary weights are fixed by the sealed common recipe.")
        self.objective_weights = {
            "base_loss": _MESSAGE_APPROXIMATION_COEFFICIENT,
            "router_importance_loss": _ROUTER_IMPORTANCE_COEFFICIENT,
            "expected_work": 0.0,
        }
        expected_message_width = 64 if self.recipe is None else int(self.recipe["model"]["message"])
        if self.message_width != expected_message_width:
            raise ValueError("Wind provider message width differs from its sealed model recipe.")
        if set(self.role_scales) != set(ROLE_NAMES) or set(self.role_query_counts) != set(ROLE_NAMES):
            raise ValueError("Wind provider role names do not match the fixed native role inventory.")
        self.query_count = sum(self.role_query_counts.values())
        if any(value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0.0)
               for value in self.role_component_scales.values()):
            raise ValueError("Wind component role scales must be positive finite physical m/s triples.")
        if self.recipe is None and self.query_count != 1024:
            raise ValueError("The sealed legacy Wind provider must retain exactly Q=1024 receiver samples per case.")
        if self.recipe is not None and self.role_query_counts != self.recipe["role_query_counts"]:
            raise ValueError("Wind provider role counts differ from the resolved recipe.")
        if self.recipe is not None and self.recipe["objective"] == "component_role" and role_component_scales is None:
            raise ValueError("Component-balanced Wind recipes require calibrated per-role component scales.")
        if self.recipe is not None and self.role_scale_calibration is None:
            raise ValueError("Versioned Wind recipes require a TRAIN-only role-scale calibration receipt.")
        if not np.array_equal(np.sort(self.train_rows), self.train_rows):
            raise ValueError("Wind provider TRAIN rows must be canonical source order.")
        if not np.array_equal(np.sort(self.validation_rows), self.validation_rows):
            raise ValueError("Wind provider development rows must be canonical source order.")
        self._validation_cache: tuple[TaskBatch, ...] | None = None
        train_layouts = tuple(int(value) for value in self.manifest["train_layout_indices"])
        self.message_calibration_layouts = train_layouts[:_MESSAGE_CALIBRATION_LAYOUT_COUNT]
        if len(self.message_calibration_layouts) != _MESSAGE_CALIBRATION_LAYOUT_COUNT:
            raise ValueError("Wind TRAIN calibration panel must contain exactly four frozen layouts.")
        layout_by_row = np.asarray(self.view.metadata["layout_index"], dtype=np.int64)
        self.message_calibration_rows = tuple(
            int(row)
            for row in self.train_rows
            if int(layout_by_row[int(row)]) in set(self.message_calibration_layouts)
        )
        if len(self.message_calibration_rows) != 3 * _MESSAGE_CALIBRATION_LAYOUT_COUNT:
            raise ValueError("Wind TRAIN message calibration panel must retain all three directions per layout.")
        self._message_scale: float | None = None
        self._expected_work_weight = 0.0
        self._expected_work_calibration: dict[str, Any] | None = None
        self._message_scale_calibration: dict[str, Any] | None = None
        self._train_row_set = set(map(int, self.train_rows.tolist()))
        self.training_samples = 0
        self.calibration_samples = 0
        self.validation_samples = 0

    def identity_payload(self) -> Mapping[str, Any]:
        if self._message_scale is None:
            raise RuntimeError("Calibrate the fixed TRAIN message scale before sealing Wind training identity.")
        payload = {
            "dataset": "WindFarm",
            "family": REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE,
            "subset_id": self.manifest["subset_id"],
            "subset_manifest_sha256": self.manifest["manifest_sha256"],
            "train_rows_sha256": _indices_sha256(self.train_rows),
            "validation_rows_sha256": _indices_sha256(self.validation_rows),
            "train_layout_indices": list(self.manifest["train_layout_indices"]),
            "validation_layout_indices": list(self.manifest["validation_layout_indices"]),
            "directions_deg": [270.0, 285.0, 300.0],
            "environment_representation": {
                "token_shape": list(self.view.token_shape),
                "record_count": int(np.prod(self.view.token_shape)),
                "measure": "geometry-only support-box quadrature in D^3",
            },
            "role_query_counts": dict(self.role_query_counts),
            "role_scales_mps": dict(self.role_scales),
            "role_scale_source_sha256": self.role_scale_sha256,
            "normalization_path": str(self.normalization_path),
            "normalization_sha256": _sha256(self.normalization_path),
            "background_choice": "train_only_height_profile",
            "background_profile": self.background_profile.to_dict(),
            "selected_turbine_counts": list(self.manifest["train_layout_selection"]["selected_turbine_counts"]),
            "uncovered_turbine_counts": list(self.manifest["train_layout_selection"]["uncovered_turbine_counts"]),
            "message_scale_calibration": {
                "method": "pre-fit fine source-message RMS",
                "layout_indices": list(self.message_calibration_layouts),
                "row_indices": list(self.message_calibration_rows),
                "query_count_per_row": 1024,
                "message_rms_latent": self._message_scale,
                "approximation_loss_weight": _MESSAGE_APPROXIMATION_COEFFICIENT,
                "scaled_weight_rule": "0.1 / fixed_train_message_rms**2",
            },
            "expected_work_calibration": {
                "method": "native TRAIN router-gradient share at the end of open phase",
                "epoch": 600,
                "layout_indices": list(self.message_calibration_layouts),
                "row_indices": list(self.message_calibration_rows),
                "target_router_gradient_share": _EXPECTED_WORK_TARGET_GRADIENT_SHARE,
                "coefficient_cap": _EXPECTED_WORK_COEFFICIENT_CAP,
                "calibration_stage": _COST_CALIBRATION_STAGE,
            },
            "catalogue_cache_capacity_bytes": self.catalogue_cache.max_cached_bytes,
            "horizon_updates_per_epoch": 3,
            "device": str(self.device),
            "physical_reference": "stored OpenFOAM native fields; no new solver calls",
        }
        if self.recipe is not None:
            payload.update({
                "resolved_recipe": dict(self.recipe),
                "resolved_recipe_sha256": self.recipe["recipe_sha256"],
                "objective": str(self.recipe["objective"]),
                "loss_scale_rule": str(self.recipe["loss_scale_rule"]),
                "role_component_scales_mps": {
                    name: self.role_component_scales[name].astype(float).tolist() for name in ROLE_NAMES
                },
                "role_scale_calibration": self.role_scale_calibration,
                "sampling_version": self.sampling_version,
                "sampling_dataset_id": self.sampling_dataset_id,
                "execution_backend": self.execution_backend,
                "query_tile_size": self.query_tile_size,
                "model": dict(self.recipe["model"]),
            })
        return payload

    def _sample_geometry_only_queries(self, row: int, key: SamplingKey) -> tuple[Any, np.ndarray]:
        """Sample the fixed role mixture without gathering any native values."""

        case = self.view.run(int(row))
        catalogue = self.catalogue_cache.get(case)
        rng = _query_rng(key, "wind_message_scale_calibration", int(row), _MESSAGE_CALIBRATION_STREAM)
        counts = (
            WIND_ROLE_SCALE_CALIBRATION_QUERY_COUNTS
            if self.recipe is not None
            else self.role_query_counts
        )
        pieces: list[np.ndarray] = []
        for role in ROLE_NAMES:
            cdf = catalogue.role_cdf[role]
            role_rng = (
                key.numpy_rng("wind_message_scale_calibration", _physical_case_direction_id(case), role)
                if self.sampling_version == SamplingKey.CASE_EPOCH_VERSION
                else rng
            )
            positions = np.searchsorted(cdf, role_rng.random(counts[role]), side="right")
            positions = np.minimum(positions, cdf.size - 1)
            valid = catalogue.role_indices.get(role)
            selected = positions if valid is None else valid[positions]
            pieces.append(catalogue.coordinates_D[selected])
        coordinates = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
        if coordinates.shape != (1024, 3) or not np.isfinite(coordinates).all():
            raise ValueError("Wind message-scale calibration query panel is invalid.")
        return case, coordinates

    def calibrate_train_message_scale(self, model: nn.Module) -> Mapping[str, Any]:
        """Bind one pre-fit message RMS using only fixed TRAIN geometry and queries."""

        if self._message_scale is not None:
            raise RuntimeError("Wind TRAIN message scale has already been calibrated for this task.")
        core = getattr(model, "core", None)
        refinement = getattr(core, "refinement", None)
        if core is None or refinement is None or not callable(getattr(core, "_read_features", None)):
            raise TypeError("Wind message calibration requires the source-resolved refined nonlinear core.")
        parameter = next(model.parameters(), None)
        if parameter is None:
            raise TypeError("Wind message calibration requires learned model parameters.")
        original_training = model.training
        model.eval()
        total_squared = 0.0
        total_components = 0
        query_digest = hashlib.sha256()
        try:
            with torch.no_grad():
                for index, row in enumerate(self.message_calibration_rows):
                    key = SamplingKey(
                        self.seed,
                        0,
                        0,
                        index,
                        "message_scale_calibration",
                        "shared",
                        sampling_version=self.sampling_version,
                        dataset_id=self.sampling_dataset_id,
                    )
                    case, coordinates = self._sample_geometry_only_queries(row, key)
                    query_digest.update(np.ascontiguousarray(coordinates).tobytes())
                    scene = self.make_scene((_scene_inputs(case),))
                    receivers = torch.as_tensor(
                        coordinates[None], device=parameter.device, dtype=torch.float32
                    )
                    receiver_features = _receiver_features(scene, receivers)
                    prepared = core.prepare(scene)
                    fine = core.source_read(
                        core._read_features(
                            prepared,
                            receivers,
                            prepared.source_states,
                            prepared.centers,
                            prepared.source_lengths,
                            receiver_features,
                        )
                    )
                    active = (prepared.present[:, None, :, None] > 0).expand_as(fine)
                    total_squared += float(fine.square().masked_select(active).double().sum().item())
                    total_components += int(active.sum().item())
        finally:
            model.train(original_training)
        if total_components <= 0 or not math.isfinite(total_squared):
            raise FloatingPointError("Wind TRAIN message-scale calibration produced no finite active messages.")
        scale = math.sqrt(total_squared / total_components)
        if not math.isfinite(scale) or scale < 1.0e-6:
            scale = 1.0e-6
        self._message_scale = float(scale)
        residual_scale = max(float(scale), 1.0e-6)
        if not torch.is_tensor(getattr(refinement, "residual_scale", None)):
            raise TypeError("Wind refined core omitted its residual-scale buffer.")
        refinement.residual_scale.fill_(residual_scale)
        receipt = {
            "method": "pre-fit fine source-message RMS",
            "layout_indices": list(self.message_calibration_layouts),
            "row_indices": list(self.message_calibration_rows),
            "query_count_per_row": 1024,
            "query_coordinates_sha256": query_digest.hexdigest(),
            "active_message_component_count": total_components,
            "message_rms_latent": float(scale),
            "minimum_scale": 1.0e-6,
            "target_values_read": False,
        }
        self._message_scale_calibration = receipt
        return dict(receipt)

    def training_state_dict(self) -> Mapping[str, Any]:
        return {
            "message_scale": self._message_scale,
            "message_scale_calibration": self._message_scale_calibration,
            "expected_work_weight": float(self._expected_work_weight),
            "expected_work_calibration": self._expected_work_calibration,
            "training_samples": int(self.training_samples),
            "calibration_samples": int(self.calibration_samples),
            "validation_samples": int(self.validation_samples),
        }

    def load_training_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        allow_prefit_calibration_replace: bool = False,
    ) -> None:
        """Restore checkpoint-bound provider state, normally requiring exact calibration.

        Evaluation may explicitly replace a pre-fit message RMS recalculated on
        another device. In that case every calibration-receipt field except the
        scalar RMS must match exactly; the trusted checkpoint value is then
        restored before its provider identity is checked. Training resume keeps
        the default strict equality behavior.
        """
        message_scale = state.get("message_scale")
        if message_scale is None or not math.isfinite(float(message_scale)) or float(message_scale) <= 0:
            raise ValueError("Saved Wind provider state omitted its positive TRAIN message scale.")
        if self._message_scale is not None and not math.isclose(
            self._message_scale, float(message_scale), rel_tol=0.0, abs_tol=0.0
        ):
            if not allow_prefit_calibration_replace:
                raise ValueError("Saved Wind provider message scale differs from the active fixed TRAIN calibration.")
            local_calibration = self._message_scale_calibration
            saved_calibration = state.get("message_scale_calibration")
            if not isinstance(local_calibration, Mapping) or not isinstance(saved_calibration, Mapping):
                raise ValueError("Replacing a device-local pre-fit calibration requires both calibration receipts.")
            local_receipt = dict(local_calibration)
            saved_receipt = dict(saved_calibration)
            local_receipt_scale = local_receipt.pop("message_rms_latent", None)
            saved_receipt_scale = saved_receipt.pop("message_rms_latent", None)
            if (
                local_receipt != saved_receipt
                or local_receipt_scale is None
                or saved_receipt_scale is None
                or not math.isclose(float(local_receipt_scale), self._message_scale, rel_tol=0.0, abs_tol=0.0)
                or not math.isclose(float(saved_receipt_scale), float(message_scale), rel_tol=0.0, abs_tol=0.0)
            ):
                raise ValueError("Saved Wind message scale is not the same pre-fit TRAIN calibration on this device.")
        self._message_scale = float(message_scale)
        message_calibration = state.get("message_scale_calibration")
        self._message_scale_calibration = (
            dict(message_calibration) if message_calibration is not None else None
        )
        expected_weight = float(state.get("expected_work_weight", 0.0))
        if not math.isfinite(expected_weight) or not 0 <= expected_weight <= _EXPECTED_WORK_COEFFICIENT_CAP:
            raise ValueError("Saved Wind expected-work coefficient violates the sealed finite cap.")
        self._expected_work_weight = expected_weight
        calibration = state.get("expected_work_calibration")
        self._expected_work_calibration = None if calibration is None else dict(calibration)
        self.training_samples = int(state.get("training_samples", self.training_samples))
        self.calibration_samples = int(state.get("calibration_samples", self.calibration_samples))
        self.validation_samples = int(state.get("validation_samples", self.validation_samples))
        if min(self.training_samples, self.calibration_samples, self.validation_samples) < 0:
            raise ValueError("Saved Wind sample counters cannot be negative.")

    def epoch_cases(self, epoch: int, seed: int) -> Sequence[int]:
        del epoch, seed
        return tuple(int(row) for row in self.train_rows)

    def _sample_row(self, row: int, key: SamplingKey) -> tuple[WindSceneInputs, np.ndarray, np.ndarray, Any]:
        case = self.view.run(int(row))
        rng = _query_rng(key, "wind_native_role_query", int(row), _QUERY_STREAM)
        rng_by_role = None
        if self.sampling_version == SamplingKey.CASE_EPOCH_VERSION:
            physical_id = _physical_case_direction_id(case)
            rng_by_role = {
                role: key.numpy_rng("wind_native_role_query", physical_id, role)
                for role in ROLE_NAMES
            }
        sampler_kwargs: dict[str, Any] = {"catalogue_cache": self.catalogue_cache}
        if rng_by_role is not None:
            sampler_kwargs["rng_by_role"] = rng_by_role
        sampled = sample_native_role_queries(case, rng, self.role_query_counts, **sampler_kwargs)
        if sampled.coordinates_D.shape != (self.query_count, 3) or sampled.target_mps.shape != (self.query_count, 3):
            raise ValueError("Wind role sampler changed the fixed query/target shape.")
        if not np.isfinite(sampled.target_mps).all():
            raise ValueError(f"Native Wind target contains nonfinite values for row {row}.")
        if key.stage == _COST_CALIBRATION_STAGE:
            self.calibration_samples += 1
        else:
            self.training_samples += 1
        scene_inputs = _scene_inputs(case)
        role_ids = _role_ids(sampled.role_sample_counts)
        if role_ids.shape != (self.query_count,):
            raise RuntimeError("Wind role assignments do not align with the receiver queries.")
        return scene_inputs, np.asarray(sampled.coordinates_D, dtype=np.float32), role_ids, sampled

    def make_batch(self, case_keys: Sequence[Any], key: SamplingKey) -> TaskBatch:
        if not case_keys:
            raise ValueError("Wind microbatches cannot be empty.")
        if len(case_keys) > self.max_microbatch_cases:
            raise ValueError(
                f"Wind microbatch size exceeds this task's limit of {self.max_microbatch_cases} native cases."
            )
        rows = tuple(int(value) for value in case_keys)
        if any(row not in self._train_row_set for row in rows):
            raise ValueError("Wind TRAIN batch contains a row outside the sealed 24-layout manifest.")
        if len(set(rows)) != len(rows):
            raise ValueError("Wind TRAIN microbatches cannot repeat a native row.")
        scene_rows: list[WindSceneInputs] = []
        receiver_rows: list[np.ndarray] = []
        role_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        metadata_rows: list[Mapping[str, Any]] = []
        sampled_rows: list[Any] = []
        for row in rows:
            scene_inputs, coordinates, role_ids, sample = self._sample_row(row, key)
            scene_rows.append(scene_inputs)
            receiver_rows.append(coordinates)
            role_rows.append(role_ids)
            target_rows.append(np.asarray(sample.target_mps, dtype=np.float32))
            sampled_rows.append(sample)
            metadata_rows.append(
                {
                    "row_index": int(row),
                    "case": scene_inputs.case_name,
                    "layout_index": int(scene_inputs.layout_index),
                    "wind_direction_deg": float(scene_inputs.wind_direction_deg),
                    "n_turbines": int(np.count_nonzero(scene_inputs.module_present > 0.5)),
                }
            )
        targets = WindTargets(
            velocity_mps=np.stack(target_rows),
            row_indices=rows,
            case_metadata=tuple(metadata_rows),
        )
        receivers = WindReceiverInputs(
            coordinates_D=np.stack(receiver_rows),
            role_ids=np.stack(role_rows),
        )
        return TaskBatch(
            scene_inputs=tuple(scene_rows),
            receivers=receivers,
            targets=targets,
            auxiliary={
                "role_names": ROLE_NAMES,
                "query_sampling_sha256": _query_sampling_sha256(rows, sampled_rows, key),
            },
            case_keys=rows,
        )

    def make_scene(self, scene_inputs: Sequence[WindSceneInputs]) -> InteractionScene:
        if any(not isinstance(item, WindSceneInputs) for item in scene_inputs):
            raise TypeError("Wind scene construction accepts only target-free WindSceneInputs.")
        return _as_scene_batch(scene_inputs, self.device)

    def _loss_denominators_one(
        self,
        batch: TaskBatch,
        *,
        include_router_importance: bool,
        include_expected_work: bool,
    ) -> dict[str, float]:
        receivers = np.asarray(batch.receivers.coordinates_D, dtype=np.float32)
        role_ids = np.asarray(batch.receivers.role_ids, dtype=np.int64)
        if receivers.ndim != 3 or receivers.shape[0] != len(batch.scene_inputs) or receivers.shape[-1] != 3:
            raise ValueError("Wind receiver batch must have shape [B,Q,3].")
        if role_ids.shape != receivers.shape[:2] or np.any((role_ids < 0) | (role_ids >= len(ROLE_NAMES))):
            raise ValueError("Wind role IDs must align one-to-one with physical receiver coordinates.")
        expected_role_ids = _role_ids(self.role_query_counts)
        if any(not np.array_equal(np.sort(row_ids), np.sort(expected_role_ids)) for row_ids in role_ids):
            raise ValueError("Wind receiver rows must retain the exact resolved five-role allocation.")
        denominators: dict[str, float] = {}
        for role in ROLE_NAMES:
            receiver_count = int(np.count_nonzero(role_ids == _ROLE_INDEX[role]))
            component_count = receiver_count * 3
            if component_count <= 0:
                raise ValueError(f"Wind microbatch has no valid receiver components for role {role!r}.")
            denominators[f"native_role/{role}"] = float(component_count)

        # Protected locality is a strict >0 test of the FP32 smooth near
        # weight. CPU/CUDA roundoff at its boundary can change one pair;
        # use the predictor's device and 512-receiver read chunks rather
        # than weakening the engine's denominator consistency check.
        scene = _as_scene_batch(batch.scene_inputs, self.device)
        query_tensor = torch.as_tensor(receivers, device=self.device, dtype=torch.float32)
        near = torch.cat([
            InteractionContextCore.near_weight(
                query_chunk, scene.centers, scene.source_lengths, scene.present,
            ) > 0
            for query_chunk in query_tensor.split(self.query_tile_size, dim=1)
        ], dim=1)
        active = scene.present[:, None, :] > 0
        active_pairs = int(active.expand_as(near).sum().item())
        protected_pairs = int((near & active).sum().item())
        unprotected_pairs = active_pairs - protected_pairs
        if min(active_pairs, protected_pairs) < 0 or active_pairs <= 0:
            raise ValueError("Wind microbatch has no valid source/receiver pairs.")
        if include_router_importance and unprotected_pairs <= 0:
            raise ValueError("Wind router importance requires at least one unprotected source/receiver pair.")
        denominators["base_loss"] = float(active_pairs * self.message_width)
        if include_router_importance:
            denominators["router_importance_loss"] = float(unprotected_pairs)
        if include_expected_work:
            denominators["expected_work"] = float(active_pairs)
        return denominators

    def loss_denominators(
        self,
        batches: Sequence[TaskBatch],
        phase: str,
        arm: str = "adaptive_detail",
    ) -> Mapping[str, float]:
        include_router_importance = arm == "adaptive_detail" and phase == "open"
        include_expected_work = arm == "adaptive_detail" and phase in ("soft", "hard")
        totals: dict[str, float] = {}
        for batch in batches:
            for name, value in self._loss_denominators_one(
                batch,
                include_router_importance=include_router_importance,
                include_expected_work=include_expected_work,
            ).items():
                totals[name] = totals.get(name, 0.0) + float(value)
        return totals

    def predict_native(
        self,
        model: nn.Module,
        scene: InteractionScene,
        receivers: WindReceiverInputs,
        execution_mode: str,
        phase: str,
        epoch: int = 0,
        temperature: float | None = None,
    ) -> tuple[WindPredictions, Mapping[str, Any]]:
        if not hasattr(model, "predict_refined_batch"):
            raise TypeError("Wind refined model must implement predict_refined_batch.")
        mode = {
            "warmup": "all_fine",
            "full_detail": "all_fine",
            "full": "all_fine",
            "adaptive_detail": "adaptive",
            "adaptive": "adaptive",
        }.get(execution_mode, execution_mode)
        effective_phase = phase if model.training else "hard"
        if temperature is None:
            temperature = self._temperature_for_epoch(epoch)
        coordinates = torch.as_tensor(receivers.coordinates_D, device=scene.centers.device, dtype=torch.float32)
        role_ids = torch.as_tensor(receivers.role_ids, device=scene.centers.device, dtype=torch.long)
        predictions, auxiliary = model.predict_refined_batch(
            scene,
            coordinates,
            execution_mode=mode,
            phase=effective_phase,
            training_signal=bool(model.training),
            temperature=float(temperature),
            execution_backend=self.execution_backend,
            chunk_size=self.query_tile_size,
            gate_version=self.gate_version,
            gate_transition=self.gate_transition,
        )
        if isinstance(predictions, Mapping):
            main_standardized = predictions["values"]
            full_standardized = predictions.get("full_values")
        else:
            main_standardized = predictions.values
            full_standardized = predictions.full_values
        baseline = model.profile_at_receivers(coordinates)
        output_scale = coordinates.new_tensor(self.normalizer.safe_std) * float(self.normalizer.u_ref_mps)
        main_mps = baseline + main_standardized * output_scale
        full_mps = None if full_standardized is None else baseline + full_standardized * output_scale
        if full_mps is None and mode == "all_fine":
            # All-fine inference is the retained fine readout itself. Adaptive
            # inference deliberately keeps this absent because it did not pay
            # for a second all-fine replay.
            full_mps = main_mps
        state = {
            **dict(auxiliary),
            "baseline_mps": baseline,
            "main_standardized": main_standardized,
            "full_standardized": full_standardized,
            "role_ids": role_ids,
            "execution_mode": mode,
            "phase": effective_phase,
            "temperature": float(temperature),
        }
        # Row identities are attached by the caller's target batch, not by
        # scene inputs, so they cannot enter the predictor.
        result = WindPredictions(
            main_mps=main_mps,
            full_mps=full_mps,
            role_ids=role_ids,
            auxiliary=state,
            execution_mode=mode,
        )
        return result, state

    def loss_terms(
        self,
        predictions: WindPredictions,
        targets: WindTargets,
        phase: str,
        auxiliary_state: Mapping[str, Any],
    ) -> Mapping[str, LossTerm]:
        target = torch.as_tensor(
            targets.velocity_mps,
            device=predictions.main_mps.device,
            dtype=predictions.main_mps.dtype,
        )
        role_ids = predictions.role_ids
        main_error = predictions.main_mps - target
        replay_weight = 0.0 if phase == "warmup" else 0.25
        if replay_weight and predictions.full_mps is None:
            raise ValueError("Wind training replay requires the already-paid all-fine prediction.")
        full_error = None if predictions.full_mps is None else predictions.full_mps - target
        result: dict[str, LossTerm] = {}
        main_weight = 1.0 - replay_weight
        for role in ROLE_NAMES:
            selected = role_ids == _ROLE_INDEX[role]
            scale = float(self.role_scales[role])
            component_scale = None
            if self.recipe is not None and self.recipe["objective"] == "component_role":
                component_scale = main_error.new_tensor(self.role_component_scales[role])
                numerator = main_weight * (main_error[selected].square() / component_scale.square()).sum()
            else:
                numerator = main_weight * (main_error[selected].square() / (scale * scale)).sum()
            if replay_weight:
                if full_error is None:
                    raise RuntimeError("Wind replay loss lost its all-fine residual.")
                if component_scale is not None:
                    numerator = numerator + replay_weight * (
                        full_error[selected].square() / component_scale.square()
                    ).sum()
                else:
                    numerator = numerator + replay_weight * (full_error[selected].square() / (scale * scale)).sum()
            denominator = float(int(selected.sum().item()) * 3)
            result[f"native_role/{role}"] = LossTerm(numerator, denominator, weight=0.2)

        auxiliary_names = {"base_loss": "base"}
        if predictions.execution_mode == "adaptive" and phase == "open":
            auxiliary_names["router_importance_loss"] = "router_importance"
        if predictions.execution_mode == "adaptive" and phase in ("soft", "hard"):
            auxiliary_names["expected_work"] = "expected_work"
        for name, auxiliary_name in auxiliary_names.items():
            numerator_key = f"{auxiliary_name}_numerator"
            denominator_key = f"{auxiliary_name}_denominator"
            numerator = auxiliary_state.get(numerator_key)
            denominator = auxiliary_state.get(denominator_key)
            if numerator is None or denominator is None:
                raise ValueError(f"Wind refinement readout omitted exact auxiliary sum/count for {name!r}.")
            weight = float(self.objective_weights.get(name, 0.0))
            if name == "base_loss":
                if self._message_scale is None:
                    raise RuntimeError("Wind base approximation loss requires its pre-fit TRAIN message scale.")
                weight = _MESSAGE_APPROXIMATION_COEFFICIENT / max(self._message_scale**2, 1.0e-12)
            if name == "expected_work":
                if self._expected_work_calibration is None:
                    raise RuntimeError("Wind expected-work loss cannot start before its TRAIN calibration.")
                weight = self._expected_work_weight
            if name == "router_importance_loss":
                weight = _ROUTER_IMPORTANCE_COEFFICIENT
            result[name] = LossTerm(numerator, denominator, weight=weight)
        return result

    def validation_loss_terms(
        self,
        predictions: WindPredictions,
        targets: WindTargets,
        auxiliary_state: Mapping[str, Any],
        *,
        batch: TaskBatch,
        arm: str,
    ) -> Mapping[str, LossTerm]:
        """Measure heldout hard-route terms without replaying omitted fine reads."""
        del auxiliary_state
        target = torch.as_tensor(
            targets.velocity_mps,
            device=predictions.main_mps.device,
            dtype=predictions.main_mps.dtype,
        )
        error = predictions.main_mps - target
        role_ids = predictions.role_ids
        result: dict[str, LossTerm] = {}
        for role in ROLE_NAMES:
            selected = role_ids == _ROLE_INDEX[role]
            scale = float(self.role_scales[role])
            numerator = (error[selected].square() / (scale * scale)).sum()
            denominator = float(int(selected.sum().item()) * 3)
            result[f"native_role/{role}"] = LossTerm(numerator, denominator, weight=0.2)

        if arm == "adaptive_detail" and self._expected_work_calibration is not None:
            auxiliary = predictions.auxiliary
            probability = auxiliary.get("probability")
            protected = auxiliary.get("protected")
            if torch.is_tensor(probability) and torch.is_tensor(protected):
                active_sources = torch.zeros(
                    probability.shape[0], probability.shape[2], dtype=torch.bool,
                    device=probability.device,
                )
                for index, scene in enumerate(batch.scene_inputs):
                    source_present = torch.as_tensor(
                        scene.module_present, dtype=torch.bool, device=probability.device)
                    count = min(int(source_present.numel()), int(probability.shape[2]))
                    active_sources[index, :count] = source_present[:count]
                active = active_sources[:, None, :].expand_as(probability)
                protected_active = protected.bool() & active
                eligible = active & ~protected_active
                numerator = (probability * eligible).sum() + protected_active.sum()
                result["expected_work"] = LossTerm(
                    numerator, active.sum(), float(self._expected_work_weight))
        return result

    def validation_batches(self) -> Iterable[TaskBatch]:
        if self._validation_cache is None:
            batches: list[TaskBatch] = []
            for start in range(0, self.validation_rows.size, 4):
                rows = self.validation_rows[start : start + 4]
                key = SamplingKey(
                    self.seed,
                    0,
                    0,
                    start // 4,
                    "validation",
                    "shared",
                    sampling_version=self.sampling_version,
                    dataset_id=self.sampling_dataset_id,
                )
                batches.append(self._make_validation_batch(rows, key))
            self._validation_cache = tuple(batches)
        return iter(self._validation_cache)

    def _make_validation_batch(self, rows: Sequence[int], key: SamplingKey) -> TaskBatch:
        scene_rows: list[WindSceneInputs] = []
        receiver_rows: list[np.ndarray] = []
        role_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        metadata_rows: list[Mapping[str, Any]] = []
        sampled_rows: list[Any] = []
        for row_value in rows:
            row = int(row_value)
            case = self.view.run(row)
            rng = _query_rng(key, "wind_validation_role_query", row, 0x56414C31)
            rng_by_role = None
            if self.sampling_version == SamplingKey.CASE_EPOCH_VERSION:
                physical_id = _physical_case_direction_id(case)
                rng_by_role = {
                    role: key.numpy_rng("wind_validation_role_query", physical_id, role)
                    for role in ROLE_NAMES
                }
            sampler_kwargs: dict[str, Any] = {"catalogue_cache": self.catalogue_cache}
            if rng_by_role is not None:
                sampler_kwargs["rng_by_role"] = rng_by_role
            sample = sample_native_role_queries(case, rng, self.validation_role_query_counts, **sampler_kwargs)
            self.validation_samples += 1
            scene_rows.append(_scene_inputs(case))
            receiver_rows.append(np.asarray(sample.coordinates_D, dtype=np.float32))
            role_rows.append(_role_ids(sample.role_sample_counts))
            target_rows.append(np.asarray(sample.target_mps, dtype=np.float32))
            sampled_rows.append(sample)
            metadata_rows.append(
                {
                    "row_index": row,
                    "case": str(case.case),
                    "layout_index": int(case.layout_index),
                    "wind_direction_deg": float(case.wind_direction_deg),
                    "n_turbines": int(case.n_turbines),
                }
            )
        rows_tuple = tuple(int(row) for row in rows)
        return TaskBatch(
            scene_inputs=tuple(scene_rows),
            receivers=WindReceiverInputs(np.stack(receiver_rows), np.stack(role_rows)),
            targets=WindTargets(np.stack(target_rows), rows_tuple, tuple(metadata_rows)),
            auxiliary={
                "role_names": ROLE_NAMES,
                "query_sampling_sha256": _query_sampling_sha256(rows_tuple, sampled_rows, key),
            },
            case_keys=rows_tuple,
        )

    def validation_metrics(
        self,
        predictions: WindPredictions,
        targets: WindTargets,
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
            objectives = []
            baseline_objectives = []
            for role in ROLE_NAMES:
                selected = role_ids[batch_index] == _ROLE_INDEX[role]
                delta = predicted[batch_index, selected] - actual[batch_index, selected]
                base_delta = baseline[batch_index, selected] - actual[batch_index, selected]
                if not selected.any():
                    raise ValueError(f"Wind validation role {role!r} is empty for native row {metadata['row_index']}.")
                component_mse = np.mean(delta * delta, axis=0)
                baseline_component_mse = np.mean(base_delta * base_delta, axis=0)
                scale = float(self.role_scales[role])
                objective = float(np.mean(component_mse) / (scale * scale))
                baseline_objective = float(np.mean(baseline_component_mse) / (scale * scale))
                objectives.append(objective)
                baseline_objectives.append(baseline_objective)
                role_metrics[role] = {
                    "component_rmse_mps": np.sqrt(component_mse).tolist(),
                    "component_rmse_mean_mps": float(np.sqrt(np.mean(component_mse))),
                    "vector_rmse_mps": float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1)))),
                    "role_scale_mps": scale,
                    "query_count": int(selected.sum()),
                }
            row["roles"] = role_metrics
            row["field_score"] = float(np.mean(objectives))
            row["height_profile_score"] = float(np.mean(baseline_objectives))
            records.append(row)
        return {"rows": records}

    def reduce_native_metrics(self, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        rows = [row for record in records for row in record["rows"]]
        if len(rows) != 24 or len({int(row["row_index"]) for row in rows}) != 24:
            raise ValueError("Wind validation must reduce the exact 24-row fixed development panel.")
        summary: dict[str, Any] = {
            "field_score": float(np.mean([row["field_score"] for row in rows])),
            "height_profile_score": float(np.mean([row["height_profile_score"] for row in rows])),
            "row_count": len(rows),
            "layout_count": len({int(row["layout_index"]) for row in rows}),
            "rows": rows,
            "roles": {},
        }
        for role in ROLE_NAMES:
            values = np.asarray(
                [row["roles"][role]["component_rmse_mps"] for row in rows], dtype=np.float64
            )
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
        self,
        batch: TaskBatch,
        predictions: WindPredictions,
        auxiliary_state: Mapping[str, Any],
    ) -> Mapping[str, int | float]:
        query_count = int(batch.receivers.coordinates_D.shape[1])
        max_sources = max(int(item.module_centers.shape[0]) for item in batch.scene_inputs)
        padded_capacity = len(batch.scene_inputs) * query_count * max_sources
        active_pairs = sum(int(np.count_nonzero(item.module_present > 0)) for item in batch.scene_inputs) * query_count
        auxiliary = predictions.auxiliary
        keep = auxiliary.get("keep")
        protected = auxiliary.get("protected")
        return {
            "cheap_rows": int(auxiliary_state.get("cheap_rows", 0)),
            "fine_rows": int(auxiliary_state.get("fine_rows", 0)),
            "near_rows": int(
                auxiliary_state.get(
                    "near_rows",
                    0 if protected is None else int(protected.sum().detach().item()),
                )
            ),
            "gate_rows": int(auxiliary_state.get("gate_rows", 0)),
            "active_pairs": int(auxiliary_state.get("active_pairs", active_pairs)),
            "padded_fine_capacity": int(auxiliary_state.get("padded_fine_capacity", padded_capacity)),
            "selected_detail_rows": int(
                auxiliary_state.get(
                    "selected_detail_rows",
                    active_pairs if keep is None and predictions.execution_mode == "all_fine"
                    else 0 if keep is None
                    else int(keep.sum().detach().item()),
                )
            ),
            "complete_fine_values": int(bool(auxiliary_state.get("complete_fine_values", False))),
        }

    def optimizer_groups(
        self,
        model: nn.Module,
        arm: str,
        stage: str,
    ) -> Sequence[OptimizerGroupSpec]:
        del arm, stage
        groups: dict[str, list[str]] = {"wind_fine_context": [], "wind_refinement": []}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = "wind_refinement" if ".refinement." in f".{name}." else "wind_fine_context"
            groups[group].append(name)
        schedule = ScheduleSpec(
            peak_lr=3.0e-4,
            warmup_start_lr=3.0e-4,
            warmup_epochs=0,
            hold_through_epoch=1000,
            total_epochs=2500,
            final_lr=3.0e-6,
        )
        return tuple(
            OptimizerGroupSpec(name=name, parameter_names=tuple(names), schedule=schedule)
            for name, names in groups.items()
            if names
        )

    def _expected_work_calibration_batches(self) -> tuple[TaskBatch, ...]:
        batches: list[TaskBatch] = []
        for microbatch_index, start in enumerate(range(0, len(self.message_calibration_rows), 4)):
            rows = self.message_calibration_rows[start : start + 4]
            key = SamplingKey(
                self.seed,
                600,
                0,
                microbatch_index,
                _COST_CALIBRATION_STAGE,
                "adaptive_detail",
            )
            batches.append(self.make_batch(rows, key))
        return tuple(batches)

    def calibrate_expected_work_weight(self, model: nn.Module) -> Mapping[str, Any]:
        """Calibrate cost pressure once from TRAIN router-gradient norms."""

        if self._message_scale is None:
            raise RuntimeError("Wind cost calibration requires the sealed pre-fit message scale.")
        if self._expected_work_calibration is not None:
            raise RuntimeError("Wind expected-work coefficient has already been calibrated.")
        core = getattr(model, "core", None)
        refinement = getattr(core, "refinement", None)
        router = getattr(refinement, "router", None)
        if core is None or refinement is None or not isinstance(router, nn.Module):
            raise TypeError("Wind cost calibration requires the source-resolved router parameters.")
        router_parameters = tuple(parameter for parameter in router.parameters() if parameter.requires_grad)
        if not router_parameters:
            raise ValueError("Wind router calibration found no trainable router parameters.")
        batches = self._expected_work_calibration_batches()
        denominators = self.loss_denominators(batches, "open", "adaptive_detail")
        expected_denominators = self.loss_denominators(batches, "soft", "adaptive_detail")
        denominators["expected_work"] = expected_denominators["expected_work"]
        native_gradients = [torch.zeros_like(parameter) for parameter in router_parameters]
        work_gradients = [torch.zeros_like(parameter) for parameter in router_parameters]
        query_digest = hashlib.sha256()
        original_training = model.training
        model.train(True)
        try:
            for batch_index, batch in enumerate(batches):
                query_digest.update(np.ascontiguousarray(batch.receivers.coordinates_D).tobytes())
                scene = self.make_scene(batch.scene_inputs)
                predictions, auxiliary = self.predict_native(
                    model,
                    scene,
                    batch.receivers,
                    execution_mode="adaptive_detail",
                    phase="open",
                    epoch=600,
                    temperature=1.0,
                )
                terms = self.loss_terms(predictions, batch.targets, "open", auxiliary)
                native_objective: torch.Tensor | None = None
                for name, term in terms.items():
                    if not name.startswith("native_role/"):
                        continue
                    contribution = term.numerator * (float(term.weight) / denominators[name])
                    native_objective = contribution if native_objective is None else native_objective + contribution
                if native_objective is None:
                    raise RuntimeError("Wind cost calibration panel produced no native reconstruction objective.")
                expected_numerator = auxiliary.get("expected_work_numerator")
                if not torch.is_tensor(expected_numerator):
                    raise ValueError("Wind readout omitted expected-work numerator during TRAIN calibration.")
                expected_objective = expected_numerator / denominators["expected_work"]
                native_local = torch.autograd.grad(
                    native_objective,
                    router_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                work_local = torch.autograd.grad(
                    expected_objective,
                    router_parameters,
                    retain_graph=False,
                    allow_unused=True,
                )
                for destination, source in zip(native_gradients, native_local):
                    if source is not None:
                        destination.add_(source.detach())
                for destination, source in zip(work_gradients, work_local):
                    if source is not None:
                        destination.add_(source.detach())
        finally:
            model.train(original_training)

        native_norm = math.sqrt(sum(float(gradient.double().square().sum().item()) for gradient in native_gradients))
        work_norm = math.sqrt(sum(float(gradient.double().square().sum().item()) for gradient in work_gradients))
        if not math.isfinite(native_norm) or not math.isfinite(work_norm):
            raise FloatingPointError("Wind TRAIN router-gradient calibration produced a nonfinite norm.")
        if native_norm <= 0 or work_norm <= 0:
            coefficient = 0.0
            reason = "zero router gradient; cost coefficient remains zero without division"
        else:
            raw_coefficient = _EXPECTED_WORK_TARGET_GRADIENT_SHARE * native_norm / work_norm
            coefficient = min(_EXPECTED_WORK_COEFFICIENT_CAP, raw_coefficient)
            reason = "one bounded TRAIN gradient-ratio calibration"
        achieved_share = 0.0 if native_norm <= 0 else coefficient * work_norm / native_norm
        self._expected_work_weight = float(coefficient)
        receipt = {
            "stage": _COST_CALIBRATION_STAGE,
            "absolute_epoch": 600,
            "training_arm": "adaptive_detail",
            "phase": "open",
            "temperature": 1.0,
            "layout_indices": list(self.message_calibration_layouts),
            "row_indices": list(self.message_calibration_rows),
            "query_coordinates_sha256": query_digest.hexdigest(),
            "router_parameter_count": sum(parameter.numel() for parameter in router_parameters),
            "router_native_gradient_norm": native_norm,
            "unit_expected_work_gradient_norm": work_norm,
            "target_gradient_share": _EXPECTED_WORK_TARGET_GRADIENT_SHARE,
            "coefficient_uncapped": (
                None if native_norm <= 0 or work_norm <= 0 else _EXPECTED_WORK_TARGET_GRADIENT_SHARE * native_norm / work_norm
            ),
            "coefficient_cap": _EXPECTED_WORK_COEFFICIENT_CAP,
            "coefficient": float(coefficient),
            "achieved_gradient_share": achieved_share,
            "reason": reason,
            "validation_values_read": False,
        }
        self._expected_work_calibration = receipt
        return dict(receipt)

    @staticmethod
    def _temperature_for_epoch(epoch: int) -> float:
        if epoch <= 600:
            return 1.0
        if epoch <= 800:
            return 1.0 - 0.9 * (epoch - 601) / 199.0
        return 0.1

    def on_phase_start(
        self,
        model: nn.Module,
        engine: Any = None,
        *,
        phase: str,
        epoch: int,
        arm: str,
        temperature: float | None = None,
    ) -> Mapping[str, Any]:
        """Run the sole cost calibration before the first soft-phase update."""

        del engine
        expected_temperature = self._temperature_for_epoch(int(epoch))
        temperature = expected_temperature if temperature is None else float(temperature)
        if not math.isfinite(temperature) or not math.isclose(
            temperature, expected_temperature, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("Wind stage callback temperature differs from the shared engine schedule.")
        receipt: dict[str, Any] = {
            "arm": str(arm),
            "epoch": int(epoch),
            "phase": str(phase),
            "temperature": float(temperature),
        }
        if (
            arm == "adaptive_detail"
            and int(epoch) == 601
            and phase == "soft"
            and self._expected_work_calibration is None
        ):
            receipt["expected_work_calibration"] = dict(self.calibrate_expected_work_weight(model))
        elif self._expected_work_calibration is not None:
            receipt["expected_work_weight"] = float(self._expected_work_weight)
        return receipt

    def preparation_summary(self) -> dict[str, Any]:
        return {
            "subset_id": self.manifest["subset_id"],
            "subset_manifest_sha256": self.manifest["manifest_sha256"],
            "train_rows": int(self.train_rows.size),
            "validation_rows": int(self.validation_rows.size),
            "train_native_query_samples": int(self.training_samples),
            "expected_work_train_calibration_query_samples": int(self.calibration_samples),
            "validation_native_query_samples": int(self.validation_samples),
            "role_query_counts": dict(self.role_query_counts),
            "validation_role_query_counts": dict(self.validation_role_query_counts),
            "query_count": int(self.query_count),
            "recipe_id": None if self.recipe is None else str(self.recipe["recipe_id"]),
            "resolved_recipe_sha256": None if self.recipe is None else str(self.recipe["recipe_sha256"]),
            "execution_backend": self.execution_backend,
            "query_tile_size": self.query_tile_size,
            "sampling_version": self.sampling_version,
            "role_scale_calibration": self.role_scale_calibration,
            "role_scale_profile": self.role_scale_profile,
            "static_native_catalogues": self.catalogue_cache.summary(),
            "test_values_read": False,
        }


def _fit_train_role_scales(
    view: WindFarmNativeView,
    train_rows: np.ndarray,
    train_layout_indices: Sequence[int],
    background_profile: VerticalProfileBaseline,
    normalizer: VelocityNormalizer,
    *,
    training_fingerprint: str,
    calibration_layout_count: int = _MESSAGE_CALIBRATION_LAYOUT_COUNT,
    catalogue_cache: NativeRoleCatalogueCache | None = None,
) -> dict[str, Any]:
    """Fit W0 and W1 scales from one deterministic, TRAIN-only native panel."""

    rows = np.asarray(train_rows, dtype=np.int64)
    layouts = tuple(int(value) for value in train_layout_indices[:calibration_layout_count])
    layout_by_row = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    layout_set = set(layouts)
    calibration_rows = tuple(int(row) for row in rows if int(layout_by_row[int(row)]) in layout_set)
    if len(layouts) != calibration_layout_count or len(calibration_rows) != 3 * calibration_layout_count:
        raise ValueError("Wind role-scale calibration must retain all three directions from four TRAIN layouts.")
    train_set = set(map(int, rows.tolist()))
    if any(row not in train_set for row in calibration_rows):
        raise ValueError("Wind role-scale calibration panel escaped the selected TRAIN rows.")
    cache = catalogue_cache or NativeRoleCatalogueCache(max_cached_bytes=DEFAULT_ROLE_CATALOGUE_CACHE_MAX_BYTES)
    sums = {name: np.zeros(3, dtype=np.float64) for name in ROLE_NAMES}
    squared_sums = {name: np.zeros(3, dtype=np.float64) for name in ROLE_NAMES}
    residual_squared_sums = {name: np.zeros(3, dtype=np.float64) for name in ROLE_NAMES}
    residual_sums = {name: np.zeros(3, dtype=np.float64) for name in ROLE_NAMES}
    sample_counts = {name: 0 for name in ROLE_NAMES}
    query_digest = hashlib.sha256()
    target_digest = hashlib.sha256()
    catalogue_build_seconds = 0.0
    native_target_sampling_seconds = 0.0
    dataset_id = _sampling_dataset_id("WindFarm", training_fingerprint)
    key = SamplingKey(
        42,
        0,
        0,
        0,
        "wind_role_scale_calibration",
        "shared",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        dataset_id=dataset_id,
    )
    for row in calibration_rows:
        case = view.run(row)
        catalogue_started = time.perf_counter()
        cache.get(case)
        catalogue_build_seconds += time.perf_counter() - catalogue_started
        physical_id = _physical_case_direction_id(case)
        rng_by_role = {
            role: key.numpy_rng("wind_role_scale_calibration", physical_id, role)
            for role in ROLE_NAMES
        }
        target_sampling_started = time.perf_counter()
        sample = sample_native_role_queries(
            case,
            key.numpy_rng("wind_role_scale_calibration_fallback", physical_id),
            WIND_ROLE_SCALE_CALIBRATION_QUERY_COUNTS,
            catalogue_cache=cache,
            rng_by_role=rng_by_role,
        )
        native_target_sampling_seconds += time.perf_counter() - target_sampling_started
        query_digest.update(row.to_bytes(8, "little", signed=False))
        query_digest.update(np.ascontiguousarray(sample.flat_indices, dtype=np.int64).tobytes())
        query_digest.update(np.ascontiguousarray(sample.coordinates_D, dtype=np.float32).tobytes())
        target_digest.update(row.to_bytes(8, "little", signed=False))
        target_digest.update(np.ascontiguousarray(sample.target_mps, dtype=np.float32).tobytes())
        background = np.asarray(background_profile.predict(sample.coordinates_D[:, 2]), dtype=np.float64)
        target_values = np.asarray(sample.target_mps, dtype=np.float64)
        residual = target_values - background
        for role in ROLE_NAMES:
            role_slice = sample.role_slices[role]
            target_role = target_values[role_slice]
            residual_role = residual[role_slice]
            if target_role.shape != (WIND_ROLE_SCALE_CALIBRATION_QUERY_COUNTS[role], 3):
                raise ValueError(f"Wind TRAIN scale panel has malformed native role {role!r} at row {row}.")
            if not np.isfinite(target_role).all() or not np.isfinite(residual_role).all():
                raise ValueError(f"Wind TRAIN scale panel contains nonfinite values for role {role!r}.")
            sums[role] += target_role.sum(axis=0, dtype=np.float64)
            squared_sums[role] += np.square(target_role).sum(axis=0, dtype=np.float64)
            residual_sums[role] += residual_role.sum(axis=0, dtype=np.float64)
            residual_squared_sums[role] += np.square(residual_role).sum(axis=0, dtype=np.float64)
            sample_counts[role] += int(target_role.shape[0])

    scalar_scales: dict[str, float] = {}
    component_scales: dict[str, list[float]] = {}
    statistics: dict[str, Any] = {}
    physical_component_floor = max(
        float(normalizer.u_ref_mps) * float(normalizer.std_floor),
        WIND_COMPONENT_SCALE_MINIMUM_MPS,
    )
    for role in ROLE_NAMES:
        count = sample_counts[role]
        expected_count = len(calibration_rows) * WIND_ROLE_SCALE_CALIBRATION_QUERY_COUNTS[role]
        if count != expected_count:
            raise RuntimeError(f"Wind TRAIN role scale {role!r} has an unexpected sample count.")
        target_mean = sums[role] / count
        target_variance = np.maximum(squared_sums[role] / count - np.square(target_mean), 0.0)
        scalar_scale = max(
            float(np.sqrt(target_variance.mean(dtype=np.float64))),
            WIND_COMPONENT_SCALE_MINIMUM_MPS,
        )
        residual_mean = residual_sums[role] / count
        residual_rms = np.sqrt(np.maximum(residual_squared_sums[role] / count, 0.0))
        component_scale = np.maximum.reduce((
            residual_rms,
            np.full(3, physical_component_floor, dtype=np.float64),
            np.full(3, WIND_COMPONENT_SCALE_FRACTION_OF_SCALAR * scalar_scale, dtype=np.float64),
        ))
        scalar_scales[role] = scalar_scale
        component_scales[role] = component_scale.astype(float).tolist()
        statistics[role] = {
            "target_mean_mps": target_mean.astype(float).tolist(),
            "target_std_mps": np.sqrt(target_variance).astype(float).tolist(),
            "profile_residual_mean_mps": residual_mean.astype(float).tolist(),
            "profile_residual_rms_mps": residual_rms.astype(float).tolist(),
            "profile_residual_scale_mps": component_scale.astype(float).tolist(),
            "sample_count": int(count),
        }
    return {
        "schema_version": 1,
        "method": "shared deterministic per-role native query panel; W0 target-component variance and W1 RMS residual from TRAIN height profile",
        "scope": "selected TRAIN rows only",
        "training_fingerprint": str(training_fingerprint),
        "training_rows_sha256": _indices_sha256(rows),
        "calibration_layout_indices": list(layouts),
        "calibration_row_indices": list(calibration_rows),
        "calibration_query_count_per_row": 1024,
        "role_query_counts_per_row": dict(WIND_ROLE_SCALE_CALIBRATION_QUERY_COUNTS),
        "role_sample_counts": sample_counts,
        "scalar_scale_rule": "max(sqrt(mean of three TRAIN target component variances), 1e-3 m/s)",
        "component_scale_rule": "max(profile-residual RMS including residual mean, u_ref_mps * normalizer.std_floor, 1e-3 m/s, 0.10 * scalar role scale)",
        "physical_component_floor_mps": physical_component_floor,
        "scalar_role_scales_mps": scalar_scales,
        "component_role_scales_mps": component_scales,
        "component_statistics": statistics,
        "query_sampling_sha256": query_digest.hexdigest(),
        "target_sample_sha256": target_digest.hexdigest(),
        "background_profile_sha256": _stable_json_sha256(background_profile.to_dict()),
        "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
        "catalogue_build_or_lookup_seconds": catalogue_build_seconds,
        "native_target_sampling_seconds": native_target_sampling_seconds,
        "catalogue_cache_after_calibration": cache.summary(),
        "target_values_read": "TRAIN calibration panel only",
        "solver_attempts": 0,
    }


def create_task(config: Mapping[str, Any]) -> tuple[nn.Module, WindRefinementTask, None]:
    """Construct a sealed legacy or versioned fixed24 Wind task."""

    config = dict(config)
    recipe = resolve_wind_recipe(
        None if config.get("recipe_id") is None else str(config["recipe_id"]),
        recipe=config.get("recipe"),
        execution_backend=config.get("execution_backend"),
        query_tile_size=config.get("query_tile_size"),
    )
    data_root = Path(config.get("data_root", DEFAULT_DATA_ROOT)).expanduser().resolve()
    derived_root = Path(config.get("derived_root", DEFAULT_DERIVED_ROOT)).expanduser().resolve()
    pilot_root = Path(config.get("pilot_root", DEFAULT_PILOT_ROOT)).expanduser().resolve()
    manifest_path = Path(config.get("manifest_path", pilot_root / "fixed_subset_manifest.json")).expanduser().resolve()
    normalization_path = Path(
        config.get("normalization_path", pilot_root / "train_only_normalization.json")
    ).expanduser().resolve()
    binding_path = Path(
        config.get("normalization_binding_path", pilot_root / "train_only_normalization.binding.json")
    ).expanduser().resolve()
    role_scale_path = Path(config.get("role_scale_path", ACTIVE_REUSE_CONFIG)).expanduser().resolve()
    background_choice = str(config.get("background_choice", "train_only_height_profile"))
    seed = int(config.get("seed", 42))
    if seed != 42:
        raise ValueError("Wind shared fixed24_v1 training is sealed to the existing seed-42 initialization and streams.")
    if background_choice != "train_only_height_profile":
        raise ValueError("The sealed Wind comparison uses the shared TRAIN-fitted height-profile background.")
    if not data_root.exists():
        raise FileNotFoundError(f"Native Wind data root is unavailable: {data_root}")
    model_recipe = {} if recipe is None else dict(recipe["model"])
    view = WindFarmNativeView(
        data_root,
        allow_npz_metadata_fallback=True,
        token_shape=(2, 2, 2) if recipe is None else tuple(model_recipe["environment_token_shape"]),
    )
    split = _load_original_split(view, derived_root)
    manifest, train_rows, validation_rows = _read_fixed_rows(view, split, manifest_path)
    normalizer, profile = read_normalization_json(normalization_path)
    if profile is None:
        raise ValueError("Wind shared training requires the already-fitted TRAIN height profile.")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    expected_binding = {
        "subset_manifest_sha256": manifest["manifest_sha256"],
        "training_row_indices_sha256": _indices_sha256(train_rows),
        "normalizer_sha256": _sha256(normalization_path),
    }
    if binding != expected_binding:
        raise ValueError("Wind normalization is not bound to the exact fixed24 TRAIN membership.")
    if normalizer.source_rows != 72 or normalizer.sample_count_per_row != 8192 or normalizer.seed != 42:
        raise ValueError("Wind normalization differs from the frozen 72-row TRAIN fitting protocol.")
    catalogue_cache_bytes = int(config.get("catalogue_cache_bytes", DEFAULT_ROLE_CATALOGUE_CACHE_MAX_BYTES))
    catalogue_cache = None if recipe is None else NativeRoleCatalogueCache(max_cached_bytes=catalogue_cache_bytes)
    role_component_scales = None
    role_scale_calibration = None
    role_scale_profile = None
    if recipe is None:
        role_scales, role_scale_hash = _role_scales(role_scale_path)
        roles = _role_counts_from_config(config)
        architecture = str(config.get("model", {}).get(
            "forward_architecture", REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE
        ))
        legacy_model = dict(config.get("model", {}))
        if architecture != REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE:
            raise ValueError("Wind common training requires the explicit refined nonlinear model recipe.")
        if int(legacy_model.get("hidden", 64)) != 64 or int(legacy_model.get("message", 64)) != 64:
            raise ValueError("The sealed legacy Wind development recipe uses H64/message64.")
        if int(legacy_model.get("environment_donors", 8)) != 8:
            raise ValueError("The sealed legacy Wind development recipe retains eight environment records.")
        model_payload = {
            "forward_architecture": architecture,
            "hidden": 64,
            "message": 64,
            "max_sources": int(legacy_model.get("max_sources", 30)),
            "base_width": int(legacy_model.get("base_width", 16)),
            "router_hidden": int(legacy_model.get("router_hidden", 32)),
        }
    else:
        if config.get("role_query_counts") is not None and dict(config["role_query_counts"]) != recipe["role_query_counts"]:
            raise ValueError("Wind explicit role counts do not match the sealed query recipe.")
        calibration_result = _fit_train_role_scales(
            view,
            train_rows,
            manifest["train_layout_indices"],
            profile,
            normalizer,
            training_fingerprint=manifest["manifest_sha256"],
            calibration_layout_count=int(recipe["calibration_layout_count"]),
            catalogue_cache=catalogue_cache,
        )
        profile_fields = {
            "catalogue_build_or_lookup_seconds",
            "native_target_sampling_seconds",
            "catalogue_cache_after_calibration",
        }
        role_scale_profile = {key: calibration_result[key] for key in profile_fields}
        calibration = {key: value for key, value in calibration_result.items() if key not in profile_fields}
        role_scales = {name: float(calibration["scalar_role_scales_mps"][name]) for name in ROLE_NAMES}
        role_component_scales = {
            name: tuple(float(value) for value in calibration["component_role_scales_mps"][name])
            for name in ROLE_NAMES
        }
        role_scale_calibration = calibration
        role_scale_hash = _stable_json_sha256(calibration)
        roles = dict(recipe["role_query_counts"])
        model_payload = {
            key: model_recipe[key]
            for key in ("forward_architecture", "hidden", "message", "max_sources", "base_width", "router_hidden")
        }
    device = torch.device(str(config.get("device", "cpu")))
    model = build_windfarm_model(model_payload, velocity_transform=normalizer, background_profile=profile)
    provider = WindRefinementTask(
        view,
        train_rows=train_rows,
        validation_rows=validation_rows,
        normalizer=normalizer,
        background_profile=profile,
        manifest=manifest,
        manifest_path=manifest_path,
        normalization_path=normalization_path,
        role_scales=role_scales,
        role_scale_sha256=role_scale_hash,
        seed=seed,
        device=device,
        role_query_counts=roles,
        catalogue_cache_bytes=catalogue_cache_bytes,
        objective_weights=config.get("objective_weights"),
        role_component_scales=role_component_scales,
        role_scale_calibration=role_scale_calibration,
        role_scale_profile=role_scale_profile,
        recipe=recipe,
        sampling_dataset_fingerprint=None if recipe is None else manifest["manifest_sha256"],
        message_width=int(model_payload["message"]),
        catalogue_cache=catalogue_cache,
    )
    model.to(device)
    provider.calibrate_train_message_scale(model)
    return model, provider, None


__all__ = [
    "DEFAULT_ROLE_QUERY_COUNTS",
    "WIND_W0_RECIPE_ID",
    "WIND_W1_RECIPE_ID",
    "WIND_W2_RECIPE_ID",
    "WIND_W3_RECIPE_ID",
    "WindPredictions",
    "WindReceiverInputs",
    "WindRefinementTask",
    "WindSceneInputs",
    "WindTargets",
    "_fit_train_role_scales",
    "_role_counts_for_query_count",
    "resolve_wind_recipe",
    "create_task",
]
