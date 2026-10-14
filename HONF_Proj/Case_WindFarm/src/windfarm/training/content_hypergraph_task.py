"""Wind adapter for the shared content-hypergraph TrainingEngine contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_forward_core.content_hypergraph import ContentHypergraphCore, SceneBatch
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

from ..content_hypergraph import (
    WIND_CONTENT_COORDINATE_FRAME,
    WIND_CONTENT_ENVIRONMENT_COUNT,
    WIND_CONTENT_ENVIRONMENT_SHAPE,
    WindContentHypergraphAdapter,
    WindContentReceivers,
    WindContentSceneInputs,
)
from ..data import WindFarmNativeView
from ..normalization import read_normalization_json
from ..workflows.joint_forward import (
    ROLE_NAMES,
    NativeRoleCatalogueCache,
    _native_role_catalogue_recipe,
    sample_native_role_queries,
)
from ..workflows.native_role_cache import training_catalogue_cache_directory
from .joint_task import (
    DEFAULT_JOINT_WIND_ROLE_SCALE_CACHE_DIR,
    JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID,
    JOINT_WIND_PROFILE_ID,
    _fit_or_load_formal_transforms,
    _fit_or_load_role_scales,
    _full_train_followup_split_manifest,
    _indices_sha256,
    _physical_case_direction_id,
    _query_sampling_sha256,
    _role_counts_for_query_count,
    _role_ids,
    _sha256,
    _stable_json_sha256,
    validate_full_train_followup_membership,
)
from .unified_task import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DERIVED_ROOT,
    DEFAULT_PILOT_ROOT,
    _load_original_split,
    _read_fixed_rows,
)

WIND_CONTENT_PROVIDER_SCHEMA = "wind_content_hypergraph_provider_v1"
WIND_CONTENT_FAMILY = "honf_content_hypergraph_wind_v1"
WIND_CONTENT_ROLE_CACHE_RAM_BYTES = 2 * 1024**3
WIND_CONTENT_ROLE_CACHE_DEVELOPMENT_BYTES = 32 * 1024**3
WIND_CONTENT_ROLE_CACHE_FULL_BYTES = 128 * 1024**3
_WIND_CONTENT_CATALOGUE_CACHES: dict[tuple[str | None, int], NativeRoleCatalogueCache] = {}
_ROLE_INDEX = {name: index for index, name in enumerate(ROLE_NAMES)}


def _readonly_array(value: Any, dtype: Any = np.float32) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    result.setflags(write=False)
    return result


def _shared_read_only_catalogue_cache(max_ram_bytes: int = WIND_CONTENT_ROLE_CACHE_RAM_BYTES) -> NativeRoleCatalogueCache:
    if isinstance(max_ram_bytes, bool) or not isinstance(max_ram_bytes, int) or not 0 < max_ram_bytes <= 128 * 1024**3:
        raise ValueError("Wind catalogue cache RAM limit must be a positive integer no larger than 128 GiB.")
    persistent = training_catalogue_cache_directory()
    cache_path = None if persistent is None else str(Path(persistent).expanduser().resolve())
    key = (cache_path, max_ram_bytes)
    cache = _WIND_CONTENT_CATALOGUE_CACHES.get(key)
    if cache is None:
        cache = NativeRoleCatalogueCache(
            persistent_dir=cache_path,
            max_cached_bytes=max_ram_bytes,
            read_only_persistent=True,
        )
        _WIND_CONTENT_CATALOGUE_CACHES[key] = cache
    return cache


@dataclass(frozen=True)
class WindContentTargets:
    """Native velocity targets gathered only after the receiver draw is fixed."""

    velocity_mps: np.ndarray
    row_indices: tuple[int, ...]
    case_metadata: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        values = _readonly_array(self.velocity_mps)
        if values.ndim != 3 or values.shape[-1] != 3:
            raise ValueError("Wind targets must have shape [B,Q,3] in m/s.")
        if len(self.row_indices) != values.shape[0] or len(self.case_metadata) != values.shape[0]:
            raise ValueError("Wind target row metadata must match the batch dimension.")
        if not np.isfinite(values).all():
            raise ValueError("Wind target velocity contains nonfinite native values.")
        object.__setattr__(self, "velocity_mps", values)


@dataclass(frozen=True)
class WindContentPredictions:
    """Physical Wind prediction plus its adapter-owned residual components."""

    main_mps: torch.Tensor
    normalized_residual: torch.Tensor
    residual_mps: torch.Tensor
    profile_mps: torch.Tensor
    role_ids: torch.Tensor


class WindContentHypergraphTask:
    """Native Wind role objective, physical metrics, and shared-core adapter."""

    def __init__(
        self,
        view: WindFarmNativeView,
        *,
        model: ContentHypergraphCore,
        adapter: WindContentHypergraphAdapter,
        train_rows: Sequence[int],
        validation_rows: Sequence[int],
        manifest: Mapping[str, Any],
        role_component_scales: Mapping[str, Sequence[float]],
        role_scale_calibration: Mapping[str, Any],
        role_scale_cache_identity: Mapping[str, Any],
        normalizer_source: Mapping[str, Any],
        recipe: Mapping[str, Any],
        seed: int,
        primary_queries: int,
        microbatch_size: int,
        effective_batch_size: int,
        total_epochs: int,
        device: torch.device | str,
        catalogue_cache: NativeRoleCatalogueCache,
    ) -> None:
        self.view = view
        self.model = model
        self.adapter = adapter
        self.train_rows = _readonly_array(train_rows, dtype=np.int64)
        self.validation_rows = _readonly_array(validation_rows, dtype=np.int64)
        self.manifest = dict(manifest)
        self.profile_id = str(self.manifest["subset_id"])
        self.training_fingerprint = str(self.manifest["manifest_sha256"])
        self.role_component_scales = {
            role: _readonly_array(role_component_scales[role], dtype=np.float64) for role in ROLE_NAMES
        }
        self.role_scale_calibration = dict(role_scale_calibration)
        self.role_scale_cache_identity = dict(role_scale_cache_identity)
        self.normalizer_source = dict(normalizer_source)
        self.recipe = json.loads(json.dumps(dict(recipe), sort_keys=True, allow_nan=False))
        self.seed = int(seed)
        self.primary_queries = int(primary_queries)
        self.role_query_counts = _role_counts_for_query_count(self.primary_queries)
        self.device = torch.device(device)
        self.max_microbatch_cases = int(microbatch_size)
        self.effective_batch_size = int(effective_batch_size)
        self.total_epochs = int(total_epochs)
        self.catalogue_cache = catalogue_cache
        self.sampling_version = SamplingKey.CASE_EPOCH_VERSION
        self.sampling_dataset_id = _runtime_sampling_dataset_id(
            {
                "dataset": "WindFarm",
                "subset_manifest_sha256": self.manifest["manifest_sha256"],
            }
        )
        self._train_row_set = {int(value) for value in self.train_rows}
        self._validation_row_set = {int(value) for value in self.validation_rows}
        self._validation_cache: dict[int, tuple[TaskBatch, ...]] = {}
        self.training_samples = 0
        self.validation_samples = 0
        if self.train_rows.ndim != 1 or self.validation_rows.ndim != 1:
            raise ValueError("Wind role task row inventories must be one-dimensional.")
        if not self.train_rows.size or not self.validation_rows.size:
            raise ValueError("Wind content-hypergraph task needs nonempty TRAIN and exposed development rows.")
        if not np.array_equal(self.train_rows, np.sort(self.train_rows)):
            raise ValueError("Wind TRAIN rows must retain canonical native order.")
        if not np.array_equal(self.validation_rows, np.sort(self.validation_rows)):
            raise ValueError("Wind development rows must retain canonical native order.")
        if set(map(int, self.train_rows)) & set(map(int, self.validation_rows)):
            raise ValueError("Wind TRAIN and exposed development row membership must be disjoint.")
        if self.seed != 42:
            raise ValueError("Wind content-hypergraph initialization and sampling are fixed at seed 42.")
        if self.effective_batch_size != 24 or not 1 <= self.max_microbatch_cases <= 24:
            raise ValueError("Wind content-hypergraph uses effective batch 24 and microbatches in [1,24].")
        if self.total_epochs not in (2500, 5000):
            raise ValueError("Wind content-hypergraph requires a 2500 development or 5000 full-follow-up horizon.")
        if set(self.role_component_scales) != set(ROLE_NAMES) or any(
            value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0.0)
            for value in self.role_component_scales.values()
        ):
            raise ValueError("Wind role/component scales must be positive finite three-channel m/s values.")
        if tuple(self.view.token_shape) != WIND_CONTENT_ENVIRONMENT_SHAPE:
            raise ValueError("Wind shared core requires the native E64 geometry-only (4,4,4) representation.")
        expected_rows = 90 if self.profile_id == JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID else 24
        if self.validation_rows.size != expected_rows:
            raise ValueError(f"Wind profile {self.profile_id!r} requires {expected_rows} exposed validation rows.")

    def identity_payload(self) -> Mapping[str, Any]:
        model_config = self.model.get_config()
        adapter_state = self.adapter.state_payload()
        train_layouts = sorted({int(self.view.metadata["layout_index"][row]) for row in self.train_rows})
        validation_layouts = sorted({int(self.view.metadata["layout_index"][row]) for row in self.validation_rows})
        result = {
            "schema": WIND_CONTENT_PROVIDER_SCHEMA,
            "dataset": "WindFarm",
            "family": WIND_CONTENT_FAMILY,
            "architecture": str(model_config["architecture"]),
            "protocol": self.recipe["protocol"],
            "dataset_profile": self.profile_id,
            "subset_id": self.manifest.get("subset_id", self.profile_id),
            "subset_manifest_sha256": self.manifest["manifest_sha256"],
            "train_row_indices_sha256": _indices_sha256(self.train_rows),
            "validation_row_indices_sha256": _indices_sha256(self.validation_rows),
            "train_rows": self.train_rows.astype(int).tolist(),
            "validation_rows": self.validation_rows.astype(int).tolist(),
            "train_layout_indices": train_layouts,
            "validation_layout_indices": validation_layouts,
            "directions_deg": [270.0, 285.0, 300.0],
            "model_config": model_config,
            "adapter_state": adapter_state,
            "normalizer_and_background_source": self.normalizer_source,
            "normalizer_sha256": _stable_json_sha256(adapter_state["normalizer"]),
            "background_profile_sha256": _stable_json_sha256(adapter_state["background_profile"]),
            "catalogue_cache_ram_bytes": self.catalogue_cache.max_cached_bytes,
            "environment_representation": {
                "token_shape": list(WIND_CONTENT_ENVIRONMENT_SHAPE),
                "record_count": WIND_CONTENT_ENVIRONMENT_COUNT,
                "feature_width": 7,
                "geometry_only": True,
                "measure": "native centre-box quadrature mass in rotor_diameters^3",
                "field_sensors": False,
            },
            "coordinate_frame": {
                "name": WIND_CONTENT_COORDINATE_FRAME,
                "units": "rotor_diameters",
                "direction_source": "native row metadata and one-hot operating feature",
                "second_rotation": False,
            },
            "objective": {
                "name": "five_role_component_balanced_native_velocity",
                "role_names": list(ROLE_NAMES),
                "role_query_counts": dict(self.role_query_counts),
                "component_scales_mps": {
                    role: values.astype(float).tolist() for role, values in self.role_component_scales.items()
                },
                "scale_calibration": self.role_scale_calibration,
                "scale_cache_identity": self.role_scale_cache_identity,
                "weight_per_role": 0.2,
                "formula": "mean over q and xyz of squared physical m/s error divided by TRAIN role/component scale squared",
                "field_score": "mean across five TRAIN-normalized role/component MSEs, then mean across native rows",
                "height_profile_score": "same metric for the TRAIN-fitted height profile baseline",
                "target_partition_access": "TRAIN targets for optimization; selected exposed DEV targets for metrics; WindTEST locked",
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
                "optimizer_schedule": asdict(self._schedule()),
                "optimizer_weight_decay": self._weight_decay(),
                "optimizer_betas": [0.9, 0.999],
                "optimizer_eps": 1.0e-8,
                "all_parameters_train_from_start": True,
            },
            "native_role_catalogue_cache": self._catalogue_cache_identity(),
            "physical_reference": "stored native OpenFOAM velocity values; no solver calls",
            "wind_test_target_values_read": False,
            "solver_attempts": 0,
            "recipe": self.recipe,
        }
        if self.profile_id == JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID:
            result["test_row_indices_sha256"] = self.manifest["test_row_indices_sha256"]
            result["test_layout_indices"] = self.manifest["test_layout_indices"]
            result["unexposed_test_target_values_read"] = False
        # Keep checkpoint-bound pre-correction identities replayable. New
        # recipes explicitly declare the plan's bias/norm exclusion rule.
        if "optimizer_bias_norm_exclusion" in self.recipe:
            result["training"]["optimizer_bias_norm_exclusion"] = self.recipe["optimizer_bias_norm_exclusion"]
        return result

    def serialization_payload(self) -> Mapping[str, Any]:
        """Small self-contained model/adapter state for native inference loading."""
        identity = self.identity_payload()
        return {
            "schema": "wind_content_hypergraph_inference_v1",
            "model_config": identity["model_config"],
            "adapter_state": identity["adapter_state"],
            "role_names": list(ROLE_NAMES),
            "role_component_scales_mps": identity["objective"]["component_scales_mps"],
            "coordinate_frame": identity["coordinate_frame"],
            "environment_representation": identity["environment_representation"],
            "data_binding": {
                "dataset_profile": identity["dataset_profile"],
                "subset_manifest_sha256": identity["subset_manifest_sha256"],
                "train_row_indices_sha256": identity["train_row_indices_sha256"],
                "normalizer_source": identity["normalizer_and_background_source"],
                "role_scale_cache_identity": identity["objective"]["scale_cache_identity"],
            },
        }

    def _schedule(self) -> ScheduleSpec:
        raw = self.recipe.get("schedule")
        expected_hold = 1000 if self.recipe["protocol"] == "development" else 2000
        if raw is None:
            return ScheduleSpec(
                peak_lr=3.0e-4,
                warmup_start_lr=3.0e-5,
                warmup_epochs=20,
                hold_through_epoch=expected_hold,
                total_epochs=self.total_epochs,
                final_lr=3.0e-6,
            )
        if not isinstance(raw, Mapping):
            raise TypeError("Wind content-hypergraph schedule must be a mapping.")
        allowed = {"peak_lr", "warmup_start_lr", "warmup_epochs", "hold_through_epoch", "total_epochs", "final_lr"}
        if set(raw) != allowed:
            raise ValueError("Wind content-hypergraph schedule fields differ from the shared engine contract.")
        result = ScheduleSpec(**{key: raw[key] for key in allowed})
        if result.total_epochs != self.total_epochs:
            raise ValueError("Wind optimizer schedule must retain the recipe's immutable horizon.")
        if result.hold_through_epoch != expected_hold:
            raise ValueError(f"Wind {self.recipe['protocol']} schedule must hold through epoch {expected_hold}.")
        return result

    def _weight_decay(self) -> float:
        value = float(self.recipe.get("weight_decay", 1.0e-5))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Wind content-hypergraph weight decay must be finite and nonnegative.")
        return value

    def _catalogue_cache_identity(self) -> Mapping[str, Any]:
        persistent = self.catalogue_cache.persistent_dir
        recipe = _native_role_catalogue_recipe()
        recipe_sha = hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "cache_recipe_sha256": recipe_sha,
            "persistent_directory": None if persistent is None else str(persistent.resolve()),
            "policy": "read-only exact geometry reuse; bounded shared-RAM fallback; no disk writes",
            "max_ram_bytes": self.catalogue_cache.max_cached_bytes,
            "roles": list(ROLE_NAMES),
        }

    def preparation_summary(self) -> Mapping[str, Any]:
        return {
            "status": "prepared_only",
            "dataset_profile": self.profile_id,
            "subset_manifest_sha256": self.manifest["manifest_sha256"],
            "train_rows": int(self.train_rows.size),
            "validation_rows": int(self.validation_rows.size),
            "train_layout_count": len({int(self.view.metadata["layout_index"][row]) for row in self.train_rows}),
            "validation_layout_count": len(
                {int(self.view.metadata["layout_index"][row]) for row in self.validation_rows}
            ),
            "training_cases_sampled": int(self.training_samples),
            "validation_cases_sampled": int(self.validation_samples),
            "primary_queries_per_case": self.primary_queries,
            "role_query_counts": dict(self.role_query_counts),
            "environment_token_shape": list(WIND_CONTENT_ENVIRONMENT_SHAPE),
            "role_scales_mps": {
                role: values.astype(float).tolist() for role, values in self.role_component_scales.items()
            },
            "native_role_catalogue_cache": self.catalogue_cache.summary(),
            "optimizer_started": False,
            "solver_attempts": 0,
            "wind_test_target_values_read": False,
        }

    def epoch_cases(self, epoch: int, seed: int) -> Sequence[int]:
        del epoch
        if int(seed) != self.seed:
            raise ValueError("Wind content-hypergraph epoch seed differs from the sealed recipe.")
        return tuple(int(value) for value in self.train_rows)

    def _sample_row(
        self,
        row: int,
        key: SamplingKey,
        role_query_counts: Mapping[str, int],
        *,
        validation: bool,
    ) -> tuple[WindContentSceneInputs, Any, np.ndarray]:
        case = self.view.run(int(row))
        scene_inputs = self.adapter.scene_inputs(case)
        physical_id = _physical_case_direction_id(case)
        stream = "wind_joint_validation_role_query" if validation else "wind_joint_native_role_query"
        sample = sample_native_role_queries(
            case,
            key.numpy_rng(f"{stream}_fallback", physical_id),
            role_query_counts,
            catalogue_cache=self.catalogue_cache,
            rng_by_role={role: key.numpy_rng(stream, physical_id, role) for role in ROLE_NAMES},
        )
        expected_queries = sum(int(value) for value in role_query_counts.values())
        if sample.coordinates_D.shape != (expected_queries, 3) or sample.target_mps.shape != (expected_queries, 3):
            raise ValueError("Wind native role sampler changed its declared receiver/target shape.")
        if not np.isfinite(sample.coordinates_D).all() or not np.isfinite(sample.target_mps).all():
            raise ValueError(f"Wind native row {row} has nonfinite sampled geometry or target values.")
        features = self.adapter.query_features(sample.coordinates_D, case.support)
        return scene_inputs, sample, features

    def _make_batch(
        self,
        rows: Sequence[int],
        key: SamplingKey,
        *,
        validation: bool,
        query_count: int,
        stream_validation: bool | None = None,
        track_samples: bool = True,
    ) -> TaskBatch:
        if len(rows) == 0 or len(rows) > self.max_microbatch_cases:
            raise ValueError("Wind content-hypergraph microbatch is empty or exceeds its declared case limit.")
        row_ids = tuple(int(value) for value in rows)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Wind content-hypergraph batches cannot repeat native rows.")
        permitted = self._validation_row_set if validation else self._train_row_set
        if any(row not in permitted for row in row_ids):
            label = "development" if validation else "TRAIN"
            raise ValueError(f"Wind {label} batch contains a row outside its sealed membership.")
        expected_dataset_id = _runtime_sampling_dataset_id(self.identity_payload())
        if key.sampling_version != SamplingKey.CASE_EPOCH_VERSION or key.dataset_id != expected_dataset_id:
            raise ValueError("Wind sampling requires its exact case_epoch_v1 TRAIN-membership identity.")
        role_counts = _role_counts_for_query_count(int(query_count))
        scenes: list[WindContentSceneInputs] = []
        coordinate_rows: list[np.ndarray] = []
        feature_rows: list[np.ndarray] = []
        flat_rows: list[np.ndarray] = []
        role_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        local_batch_rows: list[np.ndarray] = []
        local_query_rows: list[np.ndarray] = []
        local_module_rows: list[np.ndarray] = []
        samples: list[Any] = []
        metadata: list[Mapping[str, Any]] = []
        for row in row_ids:
            scene_inputs, sample, features = self._sample_row(
                row,
                key,
                role_counts,
                validation=validation if stream_validation is None else stream_validation,
            )
            role_ids = _role_ids(role_counts)
            scenes.append(scene_inputs)
            coordinate_rows.append(np.asarray(sample.coordinates_D, dtype=np.float32))
            feature_rows.append(np.asarray(features, dtype=np.float32))
            flat_rows.append(np.asarray(sample.flat_indices, dtype=np.int64))
            role_rows.append(role_ids)
            target_rows.append(np.asarray(sample.target_mps, dtype=np.float32))
            local_query, local_module = self.adapter.packed_local_pair_indices(scene_inputs, sample.coordinates_D)
            local_batch_rows.append(np.full(local_query.size, len(scenes) - 1, dtype=np.int64))
            local_query_rows.append(local_query)
            local_module_rows.append(local_module)
            samples.append(sample)
            metadata.append(
                {
                    "row_index": row,
                    "case": scene_inputs.case_name,
                    "layout_index": scene_inputs.layout_index,
                    "wind_direction_deg": scene_inputs.wind_direction_deg,
                    "n_turbines": int(np.count_nonzero(scene_inputs.module_present > 0.5)),
                }
            )
        receivers = WindContentReceivers(
            np.stack(coordinate_rows),
            np.stack(feature_rows),
            np.stack(flat_rows),
            np.stack(role_rows),
            tuple(item.scene_key for item in scenes),
            np.concatenate(local_batch_rows),
            np.concatenate(local_query_rows),
            np.concatenate(local_module_rows),
        )
        targets = WindContentTargets(np.stack(target_rows), row_ids, tuple(metadata))
        if track_samples:
            if validation:
                self.validation_samples += len(row_ids)
            else:
                self.training_samples += len(row_ids)
        return TaskBatch(
            scene_inputs=tuple(scenes),
            receivers=receivers,
            targets=targets,
            auxiliary={
                "role_names": ROLE_NAMES,
                "role_query_counts": dict(role_counts),
                "query_sampling_sha256": _query_sampling_sha256(row_ids, samples, key),
            },
            case_keys=row_ids,
        )

    def make_batch(self, case_keys: Sequence[Any], key: SamplingKey) -> TaskBatch:
        return self._make_batch(
            case_keys,
            key,
            validation=False,
            query_count=self.primary_queries,
        )

    def make_scene(self, scene_inputs: Sequence[WindContentSceneInputs]) -> SceneBatch:
        return self.adapter.make_scene_batch(scene_inputs)

    def _checked_role_ids(self, receivers: WindContentReceivers, query_count: int) -> np.ndarray:
        role_ids = np.asarray(receivers.role_ids, dtype=np.int64)
        if role_ids.ndim != 2 or role_ids.shape[1] != query_count:
            raise ValueError("Wind role IDs must align with the requested fixed query panel.")
        expected = _role_ids(_role_counts_for_query_count(query_count)).astype(np.int64)
        if any(not np.array_equal(row, expected) for row in role_ids):
            raise ValueError("Wind role assignment changed its sealed five-role counts/order.")
        return role_ids

    def loss_denominators(
        self,
        batches: Sequence[TaskBatch],
        phase: str = "joint",
        arm: str = "A-D",
    ) -> Mapping[str, float]:
        del arm
        if phase != "joint":
            raise ValueError("Wind content-hypergraph training uses the shared joint phase.")
        result = {f"native_role/{role}": 0.0 for role in ROLE_NAMES}
        for batch in batches:
            query_count = int(batch.receivers.role_ids.shape[1])
            role_ids = self._checked_role_ids(batch.receivers, query_count)
            if batch.receivers.coordinates_D.shape[:2] != role_ids.shape:
                raise ValueError("Wind receiver coordinates and role IDs must have identical [B,Q] axes.")
            if not isinstance(batch.targets, WindContentTargets):
                raise TypeError("Wind role objective requires target arrays bound by WindContentTargets.")
            if batch.targets.velocity_mps.shape[:2] != role_ids.shape:
                raise ValueError("Wind target velocity and role IDs must have identical [B,Q] axes.")
            if tuple(batch.targets.row_indices) != tuple(int(value) for value in batch.case_keys):
                raise ValueError("Wind target row metadata differs from its sealed microbatch case keys.")
            for role in ROLE_NAMES:
                count = int(np.count_nonzero(role_ids == _ROLE_INDEX[role]))
                if count <= 0:
                    raise ValueError(f"Wind microbatch omitted protected role {role!r}.")
                result[f"native_role/{role}"] += float(3 * count)
        return result

    def predict_native(
        self,
        model: nn.Module,
        scene: SceneBatch,
        receivers: WindContentReceivers,
        execution_mode: str = "A-D",
        phase: str = "joint",
        epoch: int = 0,
        temperature: float | None = None,
    ) -> tuple[WindContentPredictions, Mapping[str, Any]]:
        del epoch, temperature
        if phase not in ("joint", "development", "validation"):
            raise ValueError("Wind content-hypergraph prediction uses one joint path for TRAIN and DEV.")
        if execution_mode not in ("A-D", "A-H6"):
            raise ValueError("Wind content-hypergraph only supports the A-D and A-H6 matched architectures.")
        if not isinstance(model, ContentHypergraphCore):
            raise TypeError("Wind provider requires the standalone shared ContentHypergraphCore model.")
        config = model.get_config()
        if config.get("architecture") != execution_mode:
            raise ValueError("Wind execution arm differs from the core's sealed architecture identity.")
        query = self.adapter.make_query_batch(scene, receivers)
        organization = model.prepare(scene)
        output = model.decode(organization, query)
        if not torch.is_tensor(output.nonlinear_field) or output.nonlinear_field.shape != query.coordinates.shape[
            :2
        ] + (3,):
            raise ValueError("Wind shared core must return a normalized residual field with shape [B,Q,3].")
        physical, profile, residual = self.adapter.physical_velocity(output.nonlinear_field, query.coordinates)
        role_ids = torch.tensor(receivers.role_ids, device=physical.device, dtype=torch.long)
        predictions = WindContentPredictions(physical, output.nonlinear_field, residual, profile, role_ids)
        auxiliary = {
            "organization": organization,
            "core_output": output,
            "profile_mps": profile,
            "residual_mps": residual,
            "prepared_hyperedge_count": organization.num_hyperedges,
            "execution_mode": execution_mode,
            "phase": phase,
        }
        return predictions, auxiliary

    def _loss_terms(self, predictions: WindContentPredictions, targets: WindContentTargets) -> Mapping[str, LossTerm]:
        target = torch.tensor(
            targets.velocity_mps,
            device=predictions.main_mps.device,
            dtype=predictions.main_mps.dtype,
        )
        if target.shape != predictions.main_mps.shape:
            raise ValueError("Wind native target velocity differs from the predicted [B,Q,3] field.")
        error = predictions.main_mps - target
        result: dict[str, LossTerm] = {}
        for role in ROLE_NAMES:
            selected = predictions.role_ids == _ROLE_INDEX[role]
            if not bool(selected.any()):
                raise ValueError(f"Wind role loss is missing protected role {role!r}.")
            scale = torch.as_tensor(
                np.array(self.role_component_scales[role], copy=True),
                device=error.device,
                dtype=error.dtype,
            )
            numerator = (error[selected].square() / scale.square()).sum()
            result[f"native_role/{role}"] = LossTerm(
                numerator,
                float(3 * int(selected.sum().item())),
                weight=0.2,
            )
        return result

    def loss_terms(
        self,
        predictions: WindContentPredictions,
        targets: WindContentTargets,
        phase: str,
        auxiliary_state: Any,
    ) -> Mapping[str, LossTerm]:
        del auxiliary_state
        if phase != "joint":
            raise ValueError("Wind content-hypergraph training uses the shared joint phase.")
        return self._loss_terms(predictions, targets)

    def validation_loss_terms(
        self,
        predictions: WindContentPredictions,
        targets: WindContentTargets,
        auxiliary_state: Any,
        *,
        batch: TaskBatch,
        arm: str,
    ) -> Mapping[str, LossTerm]:
        del auxiliary_state, batch, arm
        return self._loss_terms(predictions, targets)

    def _validation_batch(self, rows: Sequence[int], key: SamplingKey, query_count: int) -> TaskBatch:
        return self._make_batch(rows, key, validation=True, query_count=query_count)

    def validation_batches(self) -> Iterable[TaskBatch]:
        return self.validation_batches_for(self.primary_queries)

    def validation_batches_for(
        self,
        query_count: int,
        *,
        microbatch_size: int | None = None,
    ) -> Iterable[TaskBatch]:
        """Return deterministic DEV-only panels, including endpoint Q8192."""
        query_count = int(query_count)
        counts = _role_counts_for_query_count(query_count)
        chunk_size = self.max_microbatch_cases if microbatch_size is None else int(microbatch_size)
        if chunk_size < 1 or chunk_size > 24:
            raise ValueError("Wind validation microbatch size must lie in [1,24].")
        # Cache only the engine's default panel; explicit endpoint panels are
        # returned as fresh immutable native samples without any disk outputs.
        if query_count in self._validation_cache and microbatch_size is None:
            return iter(self._validation_cache[query_count])
        batches: list[TaskBatch] = []
        for batch_index, start in enumerate(range(0, self.validation_rows.size, chunk_size)):
            rows = self.validation_rows[start : start + chunk_size]
            key = SamplingKey(
                self.seed,
                0,
                0,
                batch_index,
                "validation",
                "shared",
                sampling_version=self.sampling_version,
                dataset_id=self.sampling_dataset_id,
            )
            batches.append(self._validation_batch(rows, key, query_count))
        if query_count == self.primary_queries:
            self._validation_cache[query_count] = tuple(batches)
        del counts
        return iter(batches)

    def native_evaluation_rows(self, split: str, case_keys: Sequence[Any] | None = None) -> tuple[int, ...]:
        """Choose a frozen native endpoint panel without consulting field errors."""
        if split not in ("validation", "train"):
            raise ValueError("Wind native endpoint split must be 'validation' or 'train'.")
        available = self.validation_rows if split == "validation" else self.train_rows
        allowed = {int(value) for value in available}
        if case_keys is not None:
            rows = tuple(int(value) for value in case_keys)
            if not rows or len(set(rows)) != len(rows) or any(row not in allowed for row in rows):
                raise ValueError(f"Wind endpoint {split} rows must be unique members of the sealed split.")
            return rows
        if split == "validation":
            return tuple(int(value) for value in self.validation_rows)
        counts = np.asarray(self.view.metadata["n_turbines"], dtype=np.int64)
        rows = np.asarray(self.train_rows, dtype=np.int64)
        low_count = int(counts[rows].min())
        high_count = int(counts[rows].max())
        low_rows = rows[counts[rows] == low_count]
        high_rows = rows[counts[rows] == high_count]
        selected = [int(low_rows[0])]
        high = int(high_rows[-1])
        if high != selected[0]:
            selected.append(high)
        elif rows.size > 1:
            selected.append(int(rows[-1] if rows[-1] != selected[0] else rows[0]))
        return tuple(selected)

    def native_evaluation_batches(
        self,
        split: str = "validation",
        case_keys: Sequence[Any] | None = None,
        *,
        microbatch_size: int | None = None,
    ) -> Iterable[TaskBatch]:
        """Stream native Q8192 endpoint panels over DEV or input-only TRAIN fixtures."""
        rows = self.native_evaluation_rows(split, case_keys)
        chunk_size = self.max_microbatch_cases if microbatch_size is None else int(microbatch_size)
        if chunk_size < 1 or chunk_size > 24:
            raise ValueError("Wind endpoint microbatch size must lie in [1,24].")
        is_validation = split == "validation"
        for batch_index, start in enumerate(range(0, len(rows), chunk_size)):
            key = SamplingKey(
                self.seed,
                0,
                0,
                batch_index,
                "native_endpoint",
                "shared",
                sampling_version=self.sampling_version,
                dataset_id=self.sampling_dataset_id,
            )
            yield self._make_batch(
                rows[start : start + chunk_size],
                key,
                validation=is_validation,
                stream_validation=True,
                query_count=8192,
                track_samples=False,
            )

    def validation_metrics(
        self,
        predictions: WindContentPredictions,
        targets: WindContentTargets,
        auxiliary_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del auxiliary_state
        predicted = predictions.main_mps.detach().cpu().numpy().astype(np.float64, copy=False)
        baseline = predictions.profile_mps.detach().cpu().numpy().astype(np.float64, copy=False)
        actual = np.asarray(targets.velocity_mps, dtype=np.float64)
        roles = predictions.role_ids.detach().cpu().numpy()
        records: list[dict[str, Any]] = []
        for batch_index, metadata in enumerate(targets.case_metadata):
            role_metrics: dict[str, Any] = {}
            objective_values: list[float] = []
            baseline_objectives: list[float] = []
            for role in ROLE_NAMES:
                selected = roles[batch_index] == _ROLE_INDEX[role]
                if not selected.any():
                    raise ValueError(f"Wind development metric omitted role {role!r} for row {metadata['row_index']}.")
                delta = predicted[batch_index, selected] - actual[batch_index, selected]
                baseline_delta = baseline[batch_index, selected] - actual[batch_index, selected]
                component_mse = np.mean(delta * delta, axis=0)
                baseline_component_mse = np.mean(baseline_delta * baseline_delta, axis=0)
                scales = np.asarray(self.role_component_scales[role], dtype=np.float64)
                objective_values.append(float(np.mean(component_mse / np.square(scales))))
                baseline_objectives.append(float(np.mean(baseline_component_mse / np.square(scales))))
                role_metrics[role] = {
                    "component_rmse_mps": np.sqrt(component_mse).tolist(),
                    "component_rmse_mean_mps": float(np.sqrt(np.mean(component_mse))),
                    "vector_rmse_mps": float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1)))),
                    "relative_l2": float(
                        np.linalg.norm(delta) / max(np.linalg.norm(actual[batch_index, selected]), 1.0e-12)
                    ),
                    "height_profile_relative_l2": float(
                        np.linalg.norm(baseline_delta) / max(np.linalg.norm(actual[batch_index, selected]), 1.0e-12)
                    ),
                    "component_scales_mps": scales.astype(float).tolist(),
                    "query_count": int(selected.sum()),
                }
            records.append(
                {
                    **dict(metadata),
                    "roles": role_metrics,
                    "field_score": float(np.mean(objective_values)),
                    "height_profile_score": float(np.mean(baseline_objectives)),
                }
            )
        return {"rows": records}

    def reduce_native_metrics(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        expected_rows: Sequence[int] | None = None,
    ) -> Mapping[str, Any]:
        rows = [dict(row) for record in records for row in record["rows"]]
        expected = (
            {int(value) for value in self.validation_rows}
            if expected_rows is None
            else {int(value) for value in expected_rows}
        )
        observed = [int(row["row_index"]) for row in rows]
        if len(observed) != len(expected) or set(observed) != expected or len(set(observed)) != len(observed):
            raise ValueError("Wind native metrics must contain the exact exposed development rows once.")
        result: dict[str, Any] = {
            "field_score": float(np.mean([row["field_score"] for row in rows])),
            "height_profile_score": float(np.mean([row["height_profile_score"] for row in rows])),
            "row_count": len(rows),
            "layout_count": len({int(row["layout_index"]) for row in rows}),
            "rows": rows,
            "roles": {},
        }
        for role in ROLE_NAMES:
            components = np.asarray(
                [row["roles"][role]["component_rmse_mps"] for row in rows],
                dtype=np.float64,
            )
            vectors = np.asarray([row["roles"][role]["vector_rmse_mps"] for row in rows], dtype=np.float64)
            result["roles"][role] = {
                "component_rmse_mps_mean": components.mean(axis=0).tolist(),
                "component_rmse_mps_worst": components.max(axis=0).tolist(),
                "component_rmse_mps_p95": np.quantile(components, 0.95, axis=0).tolist(),
                "component_rmse_mean_mps": float(np.sqrt(np.mean(components * components))),
                "vector_rmse_mps_mean": float(vectors.mean()),
                "vector_rmse_mps_worst": float(vectors.max()),
            }
        return result

    def reduce_native_evaluation_metrics(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        split: str,
        case_keys: Sequence[Any] | None = None,
    ) -> Mapping[str, Any]:
        """Reduce a native endpoint panel and verify its exact selected rows."""
        expected = self.native_evaluation_rows(split, case_keys)
        result = dict(self.reduce_native_metrics(records, expected_rows=expected))
        result["split"] = split
        result["query_count_per_row"] = 8192
        result["endpoint_rows"] = list(expected)
        result["endpoint_selection"] = (
            "all fixed development rows"
            if split == "validation"
            else "explicit input-only TRAIN rows"
            if case_keys is not None
            else "lowest and highest turbine-count TRAIN fixtures, deterministic native-row tie break"
        )
        return result

    def loss_metadata(self) -> Mapping[str, Mapping[str, str]]:
        return {
            f"native_role/{role}": {
                "panel": "native_prediction",
                "label": f"{role} velocity",
                "unit": "dimensionless normalized squared physical velocity error",
                "formula": "mean over native role queries and Ux/Uy/Uz of squared m/s error divided by TRAIN role/component scale squared",
                "weight": "0.2 per role, equal contribution across five native roles",
            }
            for role in ROLE_NAMES
        }

    def work_counts(
        self,
        batch: TaskBatch,
        predictions: WindContentPredictions,
        auxiliary_state: Mapping[str, Any],
    ) -> Mapping[str, int | float]:
        del predictions
        organization = auxiliary_state.get("organization")
        edge_count = int(getattr(organization, "num_hyperedges", 0))
        query_count = int(batch.receivers.coordinates_D.shape[1])
        active_sources = sum(int(np.count_nonzero(item.module_present > 0.5)) for item in batch.scene_inputs)
        batch_size = len(batch.scene_inputs)
        return {
            "native_receivers": int(batch_size * query_count),
            "active_physical_sources": int(active_sources),
            "physical_source_receiver_capacity": int(active_sources * query_count),
            "environment_records": int(batch_size * WIND_CONTENT_ENVIRONMENT_COUNT),
            "prepared_hyperedges": int(batch_size * edge_count),
            "hyperedge_receiver_access_slots": int(batch_size * query_count * edge_count),
            "complete_native_prediction": 1,
        }

    def optimizer_groups(
        self,
        model: nn.Module,
        arm: str = "A-D",
        stage: str = "joint",
    ) -> Sequence[OptimizerGroupSpec]:
        del stage
        if model is not self.model:
            raise ValueError("Wind optimizer groups must be requested for this provider's shared core.")
        if arm != model.get_config().get("architecture"):
            raise ValueError("Wind optimizer arm differs from the shared core architecture.")
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        all_names = tuple(name for name, _ in model.named_parameters())
        if not names or names != all_names:
            raise ValueError("Every Wind shared-core parameter must train jointly from initialization.")
        if self.recipe.get("optimizer_bias_norm_exclusion", False):
            no_decay = tuple(name for name in names if name.endswith(".bias") or "norm" in name.lower())
            excluded = set(no_decay)
            decay = tuple(name for name in names if name not in excluded)
            schedule = self._schedule()
            groups = []
            for group_name, parameter_names, weight_decay in (
                ("wind_content_decay", decay, self._weight_decay()),
                ("wind_content_no_decay", no_decay, 0.0),
            ):
                if parameter_names:
                    groups.append(
                        OptimizerGroupSpec(
                            group_name, parameter_names, schedule,
                            weight_decay=weight_decay, betas=(0.9, 0.999), eps=1.0e-8,
                        )
                    )
            return tuple(groups)
        # Absent/false is the original recorded optimizer contract. Do not
        # silently relabel or repartition its named state during native replay.
        return (
            OptimizerGroupSpec(
                "wind_content_hypergraph",
                names,
                self._schedule(),
                weight_decay=self._weight_decay(),
                betas=(0.9, 0.999),
                eps=1.0e-8,
            ),
        )

    def native_case_for_development_row(self, row: int) -> Any:
        """Expose one read-only native DEV case for Q8192 or hub-plane evaluation."""
        row = int(row)
        if row not in self._validation_row_set:
            raise ValueError("Native Wind evaluation access is restricted to the exposed development rows.")
        return self.view.run(row)

    def native_scene_for_development_row(self, row: int) -> tuple[WindContentSceneInputs, SceneBatch]:
        case = self.native_case_for_development_row(row)
        scene_inputs = self.adapter.scene_inputs(case)
        return scene_inputs, self.adapter.make_scene_batch((scene_inputs,))

    def native_velocity_for_development_row(self, row: int, flat_indices: Any) -> np.ndarray:
        """Read a compact copy of DEV velocity values at already selected native IDs."""
        case = self.native_case_for_development_row(row)
        indices = np.asarray(flat_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= int(case.run.cell_count)):
            raise ValueError("Wind DEV native velocity indices must be valid flat cell IDs.")
        result = np.asarray(case.run.U[indices], dtype=np.float32).copy()
        if result.shape != (indices.size, 3) or not np.isfinite(result).all():
            raise ValueError("Wind DEV native velocity read returned malformed values.")
        return result


def _validate_recipe(recipe: Mapping[str, Any]) -> tuple[dict[str, Any], str, int]:
    value = dict(recipe)
    if value.get("schema") != "content_hypergraph_recipe_v1" or value.get("dataset") != "wind":
        raise ValueError("Wind builder requires a content_hypergraph_recipe_v1 with dataset='wind'.")
    architecture = str(value.get("architecture"))
    if architecture not in ("A-D", "A-H6"):
        raise ValueError("Wind content-hypergraph supports only the matched A-D and A-H6 models.")
    protocol = str(value.get("protocol"))
    if protocol not in ("development", "full_train1000"):
        raise ValueError("Wind content-hypergraph requires development or full_train1000 protocol.")
    if "optimizer_bias_norm_exclusion" in value and type(value["optimizer_bias_norm_exclusion"]) is not bool:
        raise ValueError("Wind optimizer_bias_norm_exclusion must be an explicit boolean.")
    expected_epochs = 2500 if protocol == "development" else 5000
    catalogue_cache_ram_bytes = value.get("catalogue_cache_ram_bytes", WIND_CONTENT_ROLE_CACHE_RAM_BYTES)
    if (
        isinstance(catalogue_cache_ram_bytes, bool)
        or not isinstance(catalogue_cache_ram_bytes, int)
        or not 0 < catalogue_cache_ram_bytes <= WIND_CONTENT_ROLE_CACHE_FULL_BYTES
    ):
        raise ValueError("Wind catalogue_cache_ram_bytes must be a positive integer no larger than 128 GiB.")
    value["catalogue_cache_ram_bytes"] = catalogue_cache_ram_bytes
    if value.get("seed") != 42 or type(value.get("seed")) is not int:
        raise ValueError("Wind content-hypergraph recipes are sealed to seed 42.")
    if value.get("primary_queries") != 4096 or type(value.get("primary_queries")) is not int:
        raise ValueError("Wind content-hypergraph uses 4096 native queries per row.")
    if value.get("effective_batch_size") != 24 or type(value.get("effective_batch_size")) is not int:
        raise ValueError("Wind content-hypergraph uses effective batch size 24.")
    if value.get("total_epochs") != expected_epochs or type(value.get("total_epochs")) is not int:
        raise ValueError(f"Wind {protocol} schedule horizon must be {expected_epochs}.")
    if protocol == "full_train1000" and (architecture != "A-H6" or value.get("approved_stop_after") != 1000):
        raise ValueError("Wind full_train1000 requires A-H6 and approved_stop_after=1000.")
    if value.get("sampling_version", SamplingKey.CASE_EPOCH_VERSION) != SamplingKey.CASE_EPOCH_VERSION:
        raise ValueError("Wind content-hypergraph requires case_epoch_v1 sampling.")
    if value.get("precision", "fp32") != "fp32" or value.get("edge_access", "full") != "full":
        raise ValueError("Wind content-hypergraph development uses the full-access FP32 reference path.")
    microbatch = value.get("microbatch_size")
    if type(microbatch) is not int or not 1 <= microbatch <= 24:
        raise ValueError("Wind microbatch_size must be an integer in [1,24].")
    model_values = value.get("model", {})
    if not isinstance(model_values, Mapping):
        raise TypeError("Wind model overrides must be a mapping.")
    required = {
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 6,
        "local_hidden_dim": 64,
    }
    allowed = set(required) | {"dropout", "thermal_response_rank"}
    if set(model_values) - allowed:
        raise ValueError(f"Unsupported Wind content-hypergraph model overrides: {sorted(set(model_values) - allowed)}")
    for name, expected in required.items():
        if name in model_values and model_values[name] != expected:
            raise ValueError(f"Wind shared-core contract fixes model.{name}={expected}.")
    if "dropout" in model_values and float(model_values["dropout"]) != 0.0:
        raise ValueError("Matched Wind content-hypergraph cores use dropout=0.")
    # Thermal basis rank is intentionally disabled for this three-velocity task,
    # even if a dataset-neutral recipe template declares the Thermal default 4.
    if "thermal_response_rank" in model_values and int(model_values["thermal_response_rank"]) not in (0, 4):
        raise ValueError("Wind core thermal_response_rank resolves to zero and accepts only shared defaults 0 or 4.")
    value["model"] = dict(model_values)
    return value, architecture, microbatch


def build_wind_content_task(
    recipe: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[ContentHypergraphCore, WindContentHypergraphTask]:
    """Build a fresh Wind shared core/provider pair without legacy model construction."""
    value, architecture, microbatch = _validate_recipe(recipe)
    seed = int(value["seed"])
    protocol = value["protocol"]
    total_epochs = int(value["total_epochs"])
    device = torch.device(device)
    data_root = Path(value.get("data_root", DEFAULT_DATA_ROOT)).expanduser().resolve()
    derived_root = Path(value.get("derived_root", DEFAULT_DERIVED_ROOT)).expanduser().resolve()
    pilot_root = Path(value.get("pilot_root", DEFAULT_PILOT_ROOT)).expanduser().resolve()
    manifest_path = Path(value.get("manifest_path", pilot_root / "fixed_subset_manifest.json")).expanduser().resolve()
    normalization_path = (
        Path(
            value.get(
                "normalization_path",
                pilot_root / "train_only_normalization.json",
            )
        )
        .expanduser()
        .resolve()
    )
    binding_path = (
        Path(
            value.get(
                "normalization_binding_path",
                pilot_root / "train_only_normalization.binding.json",
            )
        )
        .expanduser()
        .resolve()
    )
    role_scale_cache_dir = (
        Path(
            value.get(
                "role_scale_cache_dir",
                DEFAULT_JOINT_WIND_ROLE_SCALE_CACHE_DIR,
            )
        )
        .expanduser()
        .resolve()
    )
    if not data_root.exists():
        raise FileNotFoundError(f"Native Wind data root is unavailable: {data_root}")

    view = WindFarmNativeView(
        data_root,
        allow_npz_metadata_fallback=True,
        token_shape=WIND_CONTENT_ENVIRONMENT_SHAPE,
    )
    split = _load_original_split(view, derived_root)
    if protocol == "development":
        manifest, train_rows, validation_rows = _read_fixed_rows(view, split, manifest_path)
        normalizer, profile = read_normalization_json(normalization_path)
        if profile is None:
            raise ValueError("Wind development requires the existing TRAIN-fitted height-profile transform.")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        expected_binding = {
            "subset_manifest_sha256": manifest["manifest_sha256"],
            "training_row_indices_sha256": _indices_sha256(train_rows),
            "normalizer_sha256": _sha256(normalization_path),
        }
        if binding != expected_binding:
            raise ValueError("Wind fixed24 transforms are not bound to the exact sealed TRAIN membership.")
        if normalizer.source_rows != 72 or normalizer.sample_count_per_row != 8192 or normalizer.seed != seed:
            raise ValueError("Wind fixed24 normalizer differs from the shared 72-row TRAIN fit.")
        profile_id = JOINT_WIND_PROFILE_ID
        train_layouts = manifest["train_layout_indices"]
        normalizer_source: dict[str, Any] = {
            "kind": "verified_fixed24_train_fit",
            "normalization_path": str(normalization_path),
            "normalization_sha256": _sha256(normalization_path),
            "binding_path": str(binding_path),
            "binding": binding,
            "normalizer": normalizer.to_dict(),
            "background_profile": profile.to_dict(),
        }
    else:
        manifest, train_rows, validation_rows, test_rows = _full_train_followup_split_manifest(split, view)
        validate_full_train_followup_membership(
            train_rows,
            validation_rows,
            test_rows,
            train_layout_count=len(manifest["train_layout_indices"]),
            validation_layout_count=len(manifest["validation_layout_indices"]),
            test_layout_count=len(manifest["test_layout_indices"]),
        )
        profile_id = JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID
        train_layouts = manifest["train_layout_indices"]
        normalizer, profile, transform_binding = _fit_or_load_formal_transforms(
            view=view,
            train_rows=np.asarray(train_rows, dtype=np.int64),
            train_layouts=train_layouts,
            manifest=manifest,
            training_fingerprint=str(manifest["manifest_sha256"]),
            seed=seed,
            cache_dir=role_scale_cache_dir,
            profile_id=profile_id,
            allow_cache_write=False,
        )
        normalizer_source = {
            "kind": "verified_full_train_followup_train_fit_cache",
            "training_rows_sha256": _indices_sha256(train_rows),
            "transform_cache_binding": dict(transform_binding),
            "normalizer": normalizer.to_dict(),
            "background_profile": profile.to_dict(),
            "wind_test_target_values_read": False,
        }

    # Reuse the exact E64 five-role scale cache. If it is absent or stale, fail
    # before any sampling or disk output; model arms must share this binding.
    catalogue_cache = _shared_read_only_catalogue_cache(int(value["catalogue_cache_ram_bytes"]))
    calibration, cache_identity = _fit_or_load_role_scales(
        view=view,
        train_rows=np.asarray(train_rows, dtype=np.int64),
        train_layouts=train_layouts,
        profile=profile,
        normalizer=normalizer,
        manifest=manifest,
        training_fingerprint=str(manifest["manifest_sha256"]),
        catalogue_cache=catalogue_cache,
        cache_dir=role_scale_cache_dir,
        allow_cache_write=False,
    )
    normalizer_source["profile_sha256"] = _stable_json_sha256(profile.to_dict())
    component_scales = {
        role: tuple(float(value) for value in calibration["component_role_scales_mps"][role]) for role in ROLE_NAMES
    }
    adapter = WindContentHypergraphAdapter(normalizer, profile, device=device)
    model_values = value["model"]
    model_config = {
        "module_input_dim": 2,
        "environment_input_dim": 7,
        "coordinate_dim": 3,
        "query_input_dim": 7,
        "operating_input_dim": 11,
        "output_dim": 3,
        "architecture": architecture,
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 6,
        "thermal_response_rank": 0,
        "local_hidden_dim": 64,
        "dropout": 0.0,
        "init_seed": seed,
    }
    if any(
        model_values.get(name, expected) != expected
        for name, expected in {
            "context_dim": 128,
            "num_heads": 4,
            "context_depth": 2,
            "num_hyperedges": 6,
            "local_hidden_dim": 64,
        }.items()
    ):
        raise ValueError("Wind core model configuration diverged from its matched scientific contract.")
    devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        devices = [torch.cuda.current_device() if device.index is None else int(device.index)]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        model = ContentHypergraphCore(**model_config).to(device)

    provider = WindContentHypergraphTask(
        view,
        model=model,
        adapter=adapter,
        train_rows=train_rows,
        validation_rows=validation_rows,
        manifest=manifest,
        role_component_scales=component_scales,
        role_scale_calibration=calibration,
        role_scale_cache_identity=cache_identity,
        normalizer_source=normalizer_source,
        recipe=value,
        seed=seed,
        primary_queries=int(value["primary_queries"]),
        microbatch_size=microbatch,
        effective_batch_size=int(value["effective_batch_size"]),
        total_epochs=total_epochs,
        device=device,
        catalogue_cache=catalogue_cache,
    )
    return model, provider


__all__ = [
    "WIND_CONTENT_FAMILY",
    "WIND_CONTENT_ROLE_CACHE_DEVELOPMENT_BYTES",
    "WIND_CONTENT_ROLE_CACHE_FULL_BYTES",
    "WIND_CONTENT_ROLE_CACHE_RAM_BYTES",
    "WindContentHypergraphTask",
    "WindContentPredictions",
    "WindContentTargets",
    "build_wind_content_task",
]
