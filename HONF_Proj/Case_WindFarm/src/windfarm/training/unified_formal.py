"""Explicit full-data Wind provider for a future manual formal run.

The development fixed24 provider remains unchanged. This module binds the
original seed-42 layout-grouped split, fits fresh normalization only when
``prepare_recipe(..., metadata_only=False)`` is explicitly requested, and
reuses the shared TaskProvider and TrainingEngine contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..data import WindFarmNativeView
from ..model import REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE, build_windfarm_model
from ..normalization import (
    DEFAULT_PROFILE_BINS,
    DEFAULT_SAMPLES_PER_ROW,
    fit_velocity_statistics,
    read_normalization_json,
    write_normalization_json,
)
from ..splits import GroupSplit
from ..workflows.joint_forward import (
    DEFAULT_ROLE_CATALOGUE_CACHE_MAX_BYTES,
    ROLE_NAMES,
    NativeRoleCatalogueCache,
    sample_native_role_queries,
)
from .unified_task import (
    DEFAULT_ROLE_QUERY_COUNTS,
    WindRefinementTask,
    _indices_sha256,
    _load_original_split,
    _sha256,
    _stable_json_sha256,
)

FORMAL_RECIPE_ID = "windfarm_original_seed42_fulltrain_refinement_v1"
FORMAL_HORIZON = 5000
FORMAL_ROLE_CATALOGUE_CACHE_MAX_BYTES = 128 * 1024**3
FORMAL_HOLD_THROUGH_EPOCH = 2000
FORMAL_WARMUP_EPOCHS = 500
FORMAL_OPEN_THROUGH_EPOCH = 600
FORMAL_SOFT_THROUGH_EPOCH = 800
FORMAL_PEAK_LR = 3.0e-4
FORMAL_FINAL_LR = 3.0e-6
FORMAL_ROOT = (
    Path(__file__).resolve().parents[3]
    / "diagnostics/generated/unified_refinement_20261007/wind_formal"
)
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "Dataset/links/wind_farm"
DEFAULT_DERIVED_ROOT = Path(__file__).resolve().parents[3] / "Dataset/derived/forward_velocity_v1"
_ROLE_SCALE_STREAM = 0x46524D4C


def _stable_digest(payload: Mapping[str, Any]) -> str:
    return _stable_json_sha256(dict(payload))


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("recipe_sha256", None)
    result["recipe_sha256"] = _stable_digest(result)
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_child_path(parent: Path, child: Path, name: str) -> Path:
    resolved_parent = parent.expanduser().resolve()
    resolved_child = child.expanduser().resolve()
    if not resolved_child.is_relative_to(resolved_parent):
        raise ValueError(f"Formal Wind {name} output must remain inside its new recipe output directory.")
    return resolved_child


def _source_file_inventory(view: WindFarmNativeView) -> list[dict[str, Any]]:
    root = view.volume.volume_root
    names = (
        "U.npy",
        "case.npy",
        "layout_index.npy",
        "wd_deg.npy",
        "run_shape.npy",
        "run_cell_offsets.npy",
        "run_x_offsets.npy",
        "run_y_offsets.npy",
        "run_z_offsets.npy",
        "x_cell_m.npy",
        "y_cell_m.npy",
        "z_cell_m.npy",
    )
    result = []
    for name in names:
        path = root / name
        stat = path.stat()
        result.append({
            "name": name,
            "path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    return result


def _metadata_fingerprint(view: WindFarmNativeView) -> str:
    digest = hashlib.sha256()
    for name in ("case", "layout", "layout_index", "wd_deg", "n_turbines", "turbine_xy_D"):
        values = np.ascontiguousarray(np.asarray(view.metadata[name]))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str((values.shape, values.dtype.str)).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _validate_original_membership(view: WindFarmNativeView, split: GroupSplit) -> dict[str, Any]:
    groups = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    directions = np.asarray(view.metadata["wd_deg"], dtype=np.float64)
    if groups.shape != (600,) or set(np.unique(directions)) != {270.0, 285.0, 300.0}:
        raise ValueError("Formal Wind recipe requires the complete 600-row, three-direction native inventory.")
    expected = {270.0, 285.0, 300.0}
    for name, rows in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        layout_ids = np.unique(groups[rows])
        if rows.size != len(layout_ids) * 3:
            raise ValueError(f"Formal {name} rows do not retain all three directions for each layout.")
        for layout in layout_ids:
            if set(directions[rows[groups[rows] == layout]]) != expected:
                raise ValueError(f"Formal {name} layout {layout} has incomplete wind directions.")
    partitions = [set(map(int, rows.tolist())) for rows in (split.train, split.validation, split.test)]
    if partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2]:
        raise ValueError("Formal Wind split partitions overlap.")
    if set.union(*partitions) != set(range(600)):
        raise ValueError("Formal Wind split does not partition all 600 native rows.")
    if (split.train.size, split.validation.size, split.test.size) != (420, 90, 90):
        raise ValueError("The seed-42 original Wind split must contain 420/90/90 rows.")
    return {
        "split": split.metadata,
        "train_rows": split.train.astype(np.int64).tolist(),
        "validation_rows": split.validation.astype(np.int64).tolist(),
        "test_rows": split.test.astype(np.int64).tolist(),
        "train_rows_sha256": _indices_sha256(split.train),
        "validation_rows_sha256": _indices_sha256(split.validation),
        "test_rows_sha256": _indices_sha256(split.test),
        "train_layout_indices": np.unique(groups[split.train]).astype(int).tolist(),
        "validation_layout_indices": np.unique(groups[split.validation]).astype(int).tolist(),
        "test_layout_indices": np.unique(groups[split.test]).astype(int).tolist(),
    }


def _metadata_recipe(config: Mapping[str, Any]) -> tuple[dict[str, Any], WindFarmNativeView, GroupSplit]:
    config = dict(config)
    if int(config.get("seed", 42)) != 42:
        raise ValueError("The original formal Wind grouping is sealed to seed 42.")
    data_root = Path(config.get("data_root", DEFAULT_DATA_ROOT)).expanduser().resolve()
    derived_root = Path(config.get("derived_root", DEFAULT_DERIVED_ROOT)).expanduser().resolve()
    formal_root = Path(config.get("formal_root", config.get("output_dir", FORMAL_ROOT))).expanduser().resolve()
    normalization_path = _require_child_path(
        formal_root, Path(config.get("normalization_path", formal_root / "train_only_normalization.json")),
        "normalization",
    )
    binding_path = _require_child_path(
        formal_root, Path(config.get("normalization_binding_path", formal_root / "normalization.binding.json")),
        "normalization binding",
    )
    role_scale_path = _require_child_path(
        formal_root, Path(config.get("role_scale_path", formal_root / "train_only_role_scales.json")),
        "role scales",
    )
    view = WindFarmNativeView(data_root, allow_npz_metadata_fallback=True, token_shape=(2, 2, 2))
    split = _load_original_split(view, derived_root)
    membership = _validate_original_membership(view, split)
    turbine_counts = np.asarray(view.metadata["n_turbines"], dtype=np.int64)
    train_counts = sorted(set(map(int, turbine_counts[split.train].tolist())))
    all_counts = sorted(set(map(int, turbine_counts.tolist())))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "recipe_id": FORMAL_RECIPE_ID,
        "training_scope": "formal_full_original_train",
        "dataset": "WindFarm",
        "seed": 42,
        "data_root": str(data_root),
        "derived_root": str(derived_root),
        "source_metadata_sha256": _metadata_fingerprint(view),
        "source_file_inventory": _source_file_inventory(view),
        "source_file_binding": "resolved paths, exact byte sizes and nanosecond mtimes; target contents are not opened in metadata-only preparation",
        "partition_identity": membership,
        "normalization_path": str(normalization_path.expanduser().resolve()),
        "normalization_binding_path": str(binding_path.expanduser().resolve()),
        "role_scale_path": str(role_scale_path),
        "role_scale_sha256": None,
        "role_scale_calibration": {
            "method": "pooled componentwise standard deviation over deterministic Q=1024 five-role native samples",
            "scope": "all original TRAIN rows only",
            "train_row_count": int(split.train.size),
            "seed": 42,
            "role_query_counts": dict(DEFAULT_ROLE_QUERY_COUNTS),
            "scalar_scale": "sqrt(mean of the three per-component TRAIN variances)",
            "minimum_scale_mps": 1.0e-3,
            "status": "not_run_in_metadata_only_mode",
        },
        "role_query_counts": dict(DEFAULT_ROLE_QUERY_COUNTS),
        "model": {
            "forward_architecture": REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE,
            "hidden": 64,
            "message": 64,
            "max_sources": 30,
            "base_width": 16,
            "router_hidden": 32,
            "environment_token_shape": [2, 2, 2],
        },
        "background_choice": "train_only_height_profile",
        "normalization_fit": {
            "source_rows": int(split.train.size),
            "sample_count_per_row": DEFAULT_SAMPLES_PER_ROW,
            "seed": 42,
            "profile_bins": DEFAULT_PROFILE_BINS,
            "target_scope": "TRAIN rows only",
            "status": "not_run_in_metadata_only_mode",
        },
        "training_profile": {
            "microbatch_cases": 4,
            "effective_cases": 24,
            "case_visits_per_epoch": int(split.train.size),
            "optimizer_updates_per_epoch": int(np.ceil(split.train.size / 24)),
            "warmup_epochs": FORMAL_WARMUP_EPOCHS,
            "open_through_epoch": FORMAL_OPEN_THROUGH_EPOCH,
            "soft_through_epoch": FORMAL_SOFT_THROUGH_EPOCH,
            "total_epochs": FORMAL_HORIZON,
            "hold_through_epoch": FORMAL_HOLD_THROUGH_EPOCH,
            "peak_lr": FORMAL_PEAK_LR,
            "final_lr": FORMAL_FINAL_LR,
        },
        "turbine_count_coverage": {
            "train_counts_present": train_counts,
            "counts_missing_from_train": sorted(set(all_counts) - set(train_counts)),
        },
        "target_values_read": False,
        "solver_attempts": 0,
    }
    return _seal(payload), view, split


def prepare_recipe(config: Mapping[str, Any], *, metadata_only: bool = False) -> dict[str, Any]:
    """Bind original split metadata; optionally fit fresh TRAIN transforms.

    The metadata-only path reads geometry/split metadata only and is never
    training-ready. The full prepare path deliberately reads native targets
    from the 420 original TRAIN rows to fit a fresh normalizer and profile.
    """

    recipe, view, split = _metadata_recipe(config)
    formal_root = Path(config.get("formal_root", config.get("output_dir", FORMAL_ROOT))).expanduser().resolve()
    normalization_path = Path(recipe["normalization_path"])
    binding_path = Path(recipe["normalization_binding_path"])
    if metadata_only:
        metadata_path = formal_root / "metadata_recipe.json"
        if metadata_path.exists():
            raise FileExistsError("Metadata-only preparation needs a new formal recipe output directory.")
        result = {
            **recipe,
            "metadata_only": True,
            "ready_for_training": False,
            "status": "metadata_only_not_ready",
            "prepared_recipe_path": str(metadata_path),
        }
        result = _seal(result)
        _atomic_json(metadata_path, result)
        return result

    prepared_recipe_path = formal_root / "prepared_recipe.json"
    role_scale_path = Path(recipe["role_scale_path"])
    existing = [path for path in (normalization_path, binding_path, role_scale_path, prepared_recipe_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Full formal preparation requires a new output directory; existing files: {existing}.")

    normalizer, profile = fit_velocity_statistics(
        view.volume,
        split.train,
        samples_per_row=DEFAULT_SAMPLES_PER_ROW,
        seed=42,
        profile_bins=DEFAULT_PROFILE_BINS,
    )
    if normalizer.source_rows != 420 or normalizer.sample_count_per_row != DEFAULT_SAMPLES_PER_ROW or profile is None:
        raise RuntimeError("Fresh formal Wind transforms do not match the full original TRAIN fit contract.")
    write_normalization_json(normalization_path, normalizer, profile)
    normalization_sha256 = _sha256(normalization_path)
    role_scale_payload = _fit_formal_role_scales(view, split.train)
    _atomic_json(role_scale_path, role_scale_payload)
    role_scale_sha256 = _sha256(role_scale_path)
    binding = {
        "schema_version": 1,
        "recipe_id": FORMAL_RECIPE_ID,
        "source_metadata_sha256": recipe["source_metadata_sha256"],
        "train_rows_sha256": recipe["partition_identity"]["train_rows_sha256"],
        "normalization_sha256": normalization_sha256,
        "role_scale_sha256": role_scale_sha256,
        "normalization_source_rows": int(normalizer.source_rows),
        "normalization_samples_per_row": int(normalizer.sample_count_per_row),
        "normalization_seed": int(normalizer.seed),
        "profile_kind": "altitude_conditioned_training_profile",
        "target_values_read": "TRAIN only",
        "solver_attempts": 0,
    }
    binding["binding_sha256"] = _stable_digest(binding)
    _atomic_json(binding_path, binding)
    result = {
        **recipe,
        "metadata_only": False,
        "ready_for_training": True,
        "status": "prepared_ready",
        "normalization_sha256": normalization_sha256,
        "normalization_binding_sha256": binding["binding_sha256"],
        "role_scale_sha256": role_scale_sha256,
        "role_scales_mps": role_scale_payload["role_scales_mps"],
        "role_scale_calibration": {
            **dict(recipe["role_scale_calibration"]),
            "status": "fitted_from_all_420_original_TRAIN_rows",
            "role_sample_counts": role_scale_payload["role_sample_counts"],
            "query_sampling_sha256": role_scale_payload["query_sampling_sha256"],
            "target_sample_sha256": role_scale_payload["target_sample_sha256"],
        },
        "normalization_fit": {
            **dict(recipe["normalization_fit"]),
            "status": "fitted_from_all_420_original_TRAIN_rows",
            "normalizer": normalizer.to_dict(),
            "height_profile": profile.to_dict(),
        },
        "prepared_recipe_path": str(prepared_recipe_path),
    }
    result = _seal(result)
    _atomic_json(prepared_recipe_path, result)
    return result


def _read_recipe(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        recipe = dict(value)
    else:
        recipe = json.loads(Path(value).expanduser().read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise TypeError("Formal Wind recipe must be a JSON object.")
    saved = recipe.get("recipe_sha256")
    if not isinstance(saved, str) or _stable_digest({key: value for key, value in recipe.items() if key != "recipe_sha256"}) != saved:
        raise ValueError("Formal Wind recipe SHA256 does not verify.")
    if recipe.get("recipe_id") != FORMAL_RECIPE_ID or recipe.get("training_scope") != "formal_full_original_train":
        raise ValueError("Wind formal recipe namespace/scope differs from the sealed full-data identity.")
    return recipe


def _fit_formal_role_scales(view: WindFarmNativeView, train_rows: np.ndarray) -> dict[str, Any]:
    """Fit positive physical scales from deterministic native target samples on TRAIN only."""

    rows = np.asarray(train_rows, dtype=np.int64)
    if rows.shape != (420,) or np.unique(rows).size != rows.size:
        raise ValueError("Formal role-scale fitting requires the 420 unique original TRAIN rows.")
    counts = {name: int(DEFAULT_ROLE_QUERY_COUNTS[name]) for name in ROLE_NAMES}
    cache = NativeRoleCatalogueCache(max_cached_bytes=DEFAULT_ROLE_CATALOGUE_CACHE_MAX_BYTES)
    sums = {name: np.zeros(3, dtype=np.float64) for name in ROLE_NAMES}
    squared_sums = {name: np.zeros(3, dtype=np.float64) for name in ROLE_NAMES}
    sample_counts = {name: 0 for name in ROLE_NAMES}
    query_digest = hashlib.sha256()
    target_digest = hashlib.sha256()
    for row_value in rows:
        row = int(row_value)
        case = view.run(row)
        rng = np.random.default_rng(np.random.SeedSequence((42, row, _ROLE_SCALE_STREAM)))
        sample = sample_native_role_queries(case, rng, counts, catalogue_cache=cache)
        query_digest.update(row.to_bytes(8, "little", signed=False))
        query_digest.update(np.ascontiguousarray(sample.flat_indices, dtype=np.int64).tobytes())
        query_digest.update(np.ascontiguousarray(sample.coordinates_D, dtype=np.float32).tobytes())
        target_digest.update(row.to_bytes(8, "little", signed=False))
        target_digest.update(np.ascontiguousarray(sample.target_mps, dtype=np.float32).tobytes())
        for role in ROLE_NAMES:
            values = np.asarray(sample.target_mps[sample.role_slices[role]], dtype=np.float64)
            if values.shape != (counts[role], 3) or not np.isfinite(values).all():
                raise ValueError(f"Formal TRAIN role-scale sample for {role!r} is malformed at row {row}.")
            sums[role] += values.sum(axis=0, dtype=np.float64)
            squared_sums[role] += np.square(values).sum(axis=0, dtype=np.float64)
            sample_counts[role] += int(values.shape[0])
    scales: dict[str, float] = {}
    component_statistics: dict[str, dict[str, list[float]]] = {}
    for role in ROLE_NAMES:
        count = sample_counts[role]
        if count != 420 * counts[role]:
            raise RuntimeError(f"Formal TRAIN role scale {role!r} has an unexpected sample count.")
        mean = sums[role] / count
        variance = np.maximum(squared_sums[role] / count - np.square(mean), 0.0)
        raw_scale = float(np.sqrt(variance.mean(dtype=np.float64)))
        scales[role] = max(raw_scale, 1.0e-3)
        component_statistics[role] = {"mean_mps": mean.tolist(), "std_mps": np.sqrt(variance).tolist()}
    return {
        "schema_version": 1,
        "recipe_id": FORMAL_RECIPE_ID,
        "method": "pooled componentwise standard deviation over deterministic Q=1024 five-role native samples",
        "training_scope": "all 420 original TRAIN rows only",
        "training_rows_sha256": _indices_sha256(rows),
        "seed": 42,
        "role_query_counts_per_row": counts,
        "role_sample_counts": sample_counts,
        "scalar_scale_rule": "sqrt(mean of the three per-component TRAIN variances)",
        "minimum_scale_mps": 1.0e-3,
        "component_statistics": component_statistics,
        "role_scales_mps": scales,
        "query_sampling_sha256": query_digest.hexdigest(),
        "target_sample_sha256": target_digest.hexdigest(),
        "target_values_read": "TRAIN only",
        "solver_attempts": 0,
    }


def validate_recipe(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Read-only verification of the formal split and optional fitted transform."""

    recipe = _read_recipe(value)
    config = {
        "seed": 42,
        "data_root": recipe["data_root"],
        "derived_root": recipe["derived_root"],
        "formal_root": str(Path(recipe["prepared_recipe_path"]).parent),
        "normalization_path": recipe["normalization_path"],
        "normalization_binding_path": recipe["normalization_binding_path"],
        "role_scale_path": recipe["role_scale_path"],
    }
    current, _view, split = _metadata_recipe(config)
    for key in ("source_metadata_sha256", "partition_identity", "source_file_inventory", "role_scale_path"):
        if current[key] != recipe[key]:
            raise ValueError(f"Formal Wind recipe changed current {key}.")
    if recipe.get("ready_for_training") is not True:
        return {
            "status": "metadata_only_not_ready",
            "ready_for_training": False,
            "source_rows": {"train": int(split.train.size), "validation": int(split.validation.size),
                            "test_metadata_only": int(split.test.size)},
            "target_values_read": False,
            "normalization_fit_performed": False,
            "solver_attempts": 0,
            "recipe_sha256": recipe["recipe_sha256"],
        }
    normalization_path = Path(recipe["normalization_path"])
    binding_path = Path(recipe["normalization_binding_path"])
    if not normalization_path.is_file() or _sha256(normalization_path) != recipe.get("normalization_sha256"):
        raise ValueError("Formal Wind normalization file is missing or no longer matches its recipe binding.")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    saved_binding_hash = binding.pop("binding_sha256", None)
    if saved_binding_hash != recipe.get("normalization_binding_sha256") or _stable_digest(binding) != saved_binding_hash:
        raise ValueError("Formal Wind normalization binding file does not verify.")
    if (
        binding.get("recipe_id") != FORMAL_RECIPE_ID
        or binding.get("source_metadata_sha256") != recipe["source_metadata_sha256"]
        or binding.get("train_rows_sha256") != recipe["partition_identity"]["train_rows_sha256"]
        or binding.get("normalization_sha256") != recipe["normalization_sha256"]
        or binding.get("role_scale_sha256") != recipe.get("role_scale_sha256")
        or binding.get("normalization_source_rows") != 420
        or binding.get("normalization_samples_per_row") != DEFAULT_SAMPLES_PER_ROW
        or binding.get("normalization_seed") != 42
    ):
        raise ValueError("Formal Wind normalization binding differs from the full original TRAIN recipe.")
    role_scale_path = Path(recipe["role_scale_path"])
    if not role_scale_path.is_file() or _sha256(role_scale_path) != recipe.get("role_scale_sha256"):
        raise ValueError("Formal Wind role scales are missing or no longer match the fresh TRAIN-only fit.")
    role_scale_payload = json.loads(role_scale_path.read_text(encoding="utf-8"))
    expected_scale_counts = {name: 420 * int(DEFAULT_ROLE_QUERY_COUNTS[name]) for name in ROLE_NAMES}
    scales = role_scale_payload.get("role_scales_mps", {})
    if (
        role_scale_payload.get("recipe_id") != FORMAL_RECIPE_ID
        or role_scale_payload.get("training_rows_sha256") != recipe["partition_identity"]["train_rows_sha256"]
        or role_scale_payload.get("training_scope") != "all 420 original TRAIN rows only"
        or role_scale_payload.get("role_sample_counts") != expected_scale_counts
        or set(scales) != set(ROLE_NAMES)
        or not all(np.isfinite(float(value)) and float(value) > 0 for value in scales.values())
    ):
        raise ValueError("Formal Wind role-scale artifact does not match the full TRAIN-only recipe.")
    normalizer, profile = read_normalization_json(normalization_path)
    if normalizer.source_rows != 420 or normalizer.sample_count_per_row != DEFAULT_SAMPLES_PER_ROW or profile is None:
        raise ValueError("Formal Wind transform does not contain fresh 420-row TRAIN statistics and profile.")
    return {
        "status": "prepared_ready",
        "ready_for_training": True,
        "source_rows": {"train": int(split.train.size), "validation": int(split.validation.size),
                        "test_metadata_only": int(split.test.size)},
        "target_values_read": False,
        "normalization_fit_performed": False,
        "normalization_sha256": recipe["normalization_sha256"],
        "normalization_binding_sha256": saved_binding_hash,
        "role_scale_sha256": recipe["role_scale_sha256"],
        "recipe_sha256": recipe["recipe_sha256"],
        "solver_attempts": 0,
    }


class WindFormalRefinementTask(WindRefinementTask):
    """The common Wind provider with explicit full-data identity/schedule."""

    max_microbatch_cases = 24

    def __init__(
        self,
        *args: Any,
        formal_recipe: Mapping[str, Any],
        startup_benchmark: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.formal_recipe = dict(formal_recipe)
        self.startup_benchmark = bool(startup_benchmark)

    def identity_payload(self) -> Mapping[str, Any]:
        payload = dict(super().identity_payload())
        payload.update({
            "dataset_scope": "formal_full_original_train",
            "formal_recipe_id": FORMAL_RECIPE_ID,
            "formal_recipe_sha256": self.formal_recipe["recipe_sha256"],
            "source_metadata_sha256": self.formal_recipe["source_metadata_sha256"],
            "source_file_inventory": self.formal_recipe["source_file_inventory"],
            "partition_identity": self.formal_recipe["partition_identity"],
            "normalization_sha256": self.formal_recipe["normalization_sha256"],
            "normalization_binding_sha256": self.formal_recipe["normalization_binding_sha256"],
            "role_scale_sha256": self.formal_recipe["role_scale_sha256"],
            "startup_benchmark": self.startup_benchmark,
            "validation_scope": (
                "fixed24_input_only_startup_panel" if self.startup_benchmark
                else "all_90_original_validation_rows"
            ),
            "validation_rows_sha256": _indices_sha256(self.validation_rows),
            "formal_horizon": {
                "total_epochs": FORMAL_HORIZON,
                "hold_through_epoch": FORMAL_HOLD_THROUGH_EPOCH,
                "warmup_epochs": FORMAL_WARMUP_EPOCHS,
                "open_through_epoch": FORMAL_OPEN_THROUGH_EPOCH,
                "soft_through_epoch": FORMAL_SOFT_THROUGH_EPOCH,
            },
        })
        payload["subset_id"] = FORMAL_RECIPE_ID
        payload["subset_manifest_sha256"] = self.formal_recipe["recipe_sha256"]
        return payload

    def optimizer_groups(self, model: nn.Module, arm: str, stage: str):
        from dataclasses import replace

        from honf_runtime.unified_training import ScheduleSpec

        groups = super().optimizer_groups(model, arm, stage)
        schedule = ScheduleSpec(
            peak_lr=FORMAL_PEAK_LR,
            warmup_start_lr=FORMAL_PEAK_LR,
            warmup_epochs=0,
            hold_through_epoch=FORMAL_HOLD_THROUGH_EPOCH,
            total_epochs=FORMAL_HORIZON,
            final_lr=FORMAL_FINAL_LR,
        )
        return tuple(replace(group, schedule=schedule) for group in groups)

    def preparation_summary(self) -> dict[str, Any]:
        summary = dict(super().preparation_summary())
        summary.update({
            "dataset_scope": "formal_full_original_train",
            "formal_recipe_id": FORMAL_RECIPE_ID,
            "train_rows": 420,
            "validation_rows": int(self.validation_rows.size),
            "startup_benchmark": self.startup_benchmark,
            "test_rows_loaded_by_provider": 0,
            "total_epochs": FORMAL_HORIZON,
            "hold_through_epoch": FORMAL_HOLD_THROUGH_EPOCH,
        })
        return summary

    def reduce_native_metrics(self, records: Any) -> Mapping[str, Any]:
        rows = [row for record in records for row in record["rows"]]
        expected_rows = 24 if self.startup_benchmark else 90
        if len(rows) != expected_rows or len({int(row["row_index"]) for row in rows}) != expected_rows:
            raise ValueError(f"Formal Wind validation must reduce exactly {expected_rows} declared rows.")
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


def create_task(
    recipe: Mapping[str, Any] | str | Path,
    *,
    device: str | torch.device = "cpu",
    startup_benchmark: bool = False,
) -> tuple[nn.Module, WindFormalRefinementTask, None]:
    """Build a formal full-data provider only from a completed fresh prepare."""

    if isinstance(recipe, Mapping):
        recipe = dict(recipe)
        device = recipe.pop("device", device)
        startup_benchmark = bool(recipe.pop("startup_benchmark", startup_benchmark))
    sealed = _read_recipe(recipe)
    receipt = validate_recipe(sealed)
    if not receipt["ready_for_training"]:
        raise ValueError("Formal Wind task construction requires a completed full TRAIN-only normalization prepare.")
    view = WindFarmNativeView(
        Path(sealed["data_root"]), allow_npz_metadata_fallback=True, token_shape=(2, 2, 2)
    )
    split = _load_original_split(view, Path(sealed["derived_root"]))
    membership = _validate_original_membership(view, split)
    if membership != sealed["partition_identity"]:
        raise ValueError("Formal Wind full TRAIN/validation/test metadata changed after prepare.")
    normalizer, profile = read_normalization_json(sealed["normalization_path"])
    if profile is None:
        raise ValueError("Formal Wind prepared normalization omitted the TRAIN-fitted height profile.")
    role_scale_path = Path(sealed["role_scale_path"])
    role_scale_payload = json.loads(role_scale_path.read_text(encoding="utf-8"))
    role_scales = {name: float(role_scale_payload["role_scales_mps"][name]) for name in ROLE_NAMES}
    role_scale_sha256 = _sha256(role_scale_path)
    validation_rows = split.validation.astype(np.int64)
    if startup_benchmark:
        from .unified_task import DEFAULT_PILOT_ROOT, _read_fixed_rows

        _, _startup_train_rows, startup_validation_rows = _read_fixed_rows(
            view, split, DEFAULT_PILOT_ROOT / "fixed_subset_manifest.json"
        )
        if not set(map(int, startup_validation_rows.tolist())).issubset(set(map(int, split.validation.tolist()))):
            raise ValueError("Formal startup DEV24 rows must be a subset of the original full validation partition.")
        validation_rows = startup_validation_rows.astype(np.int64)
    groups = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    train_counts = sorted(set(map(int, np.asarray(view.metadata["n_turbines"])[split.train].tolist())))
    manifest_body = {
        "subset_id": FORMAL_RECIPE_ID,
        "train_layout_indices": np.unique(groups[split.train]).astype(int).tolist(),
        "validation_layout_indices": np.unique(groups[validation_rows]).astype(int).tolist(),
        "train_layout_selection": {
            "selected_turbine_counts": train_counts,
            "uncovered_turbine_counts": sorted(
                set(map(int, np.asarray(view.metadata["n_turbines"]).tolist())) - set(train_counts)
            ),
        },
    }
    manifest = {**manifest_body, "manifest_sha256": _stable_digest(manifest_body)}
    model = build_windfarm_model(
        {
            "forward_architecture": REFINED_SOURCE_RESOLVED_NONLINEAR_ARCHITECTURE,
            "hidden": 64,
            "message": 64,
            "max_sources": 30,
            "base_width": 16,
            "router_hidden": 32,
        },
        velocity_transform=normalizer,
        background_profile=profile,
    ).to(torch.device(device))
    provider = WindFormalRefinementTask(
        view,
        train_rows=split.train.astype(np.int64),
        validation_rows=validation_rows,
        normalizer=normalizer,
        background_profile=profile,
        manifest=manifest,
        manifest_path=Path(sealed["prepared_recipe_path"]),
        normalization_path=Path(sealed["normalization_path"]),
        role_scales=role_scales,
        role_scale_sha256=role_scale_sha256,
        seed=42,
        device=device,
        role_query_counts=DEFAULT_ROLE_QUERY_COUNTS,
        # Full TRAIN and first-validation geometry share this lazy cache.
        # Its capacity is sealed by the provider identity, not preallocated.
        catalogue_cache_bytes=FORMAL_ROLE_CATALOGUE_CACHE_MAX_BYTES,
        message_width=64,
        formal_recipe=sealed,
        startup_benchmark=startup_benchmark,
    )
    provider.calibrate_train_message_scale(model)
    return model, provider, None


__all__ = [
    "FORMAL_HORIZON",
    "FORMAL_RECIPE_ID",
    "WindFormalRefinementTask",
    "create_task",
    "prepare_recipe",
    "validate_recipe",
]
