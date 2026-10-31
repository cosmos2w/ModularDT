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
    WindContentHypergraphAdapterV3,
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
WIND_CONTENT_PROVIDER_SCHEMA_V2 = "wind_content_hypergraph_provider_v2"
WIND_CONTENT_FAMILY_V2 = "honf_content_hypergraph_wind_v2"
WIND_CONTENT_ROLE_CACHE_RAM_BYTES = 2 * 1024**3
WIND_CONTENT_ROLE_CACHE_DEVELOPMENT_BYTES = 32 * 1024**3
WIND_CONTENT_ROLE_CACHE_FULL_BYTES = 128 * 1024**3
WIND_SOURCE_READ_ROLE_CACHE_RAM_BYTES = 64 * 1024**3
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
        role_component_scales: Mapping[str, Sequence[float]] | None,
        role_scale_calibration: Mapping[str, Any] | None,
        role_scale_cache_identity: Mapping[str, Any] | None,
        normalizer_source: Mapping[str, Any],
        recipe: Mapping[str, Any],
        seed: int,
        primary_queries: int,
        microbatch_size: int,
        effective_batch_size: int,
        total_epochs: int,
        device: torch.device | str,
        catalogue_cache: NativeRoleCatalogueCache,
        allow_formal_protocol: bool = False,
    ) -> None:
        self.view = view
        self.model = model
        self.adapter = adapter
        self.train_rows = _readonly_array(train_rows, dtype=np.int64)
        self.validation_rows = _readonly_array(validation_rows, dtype=np.int64)
        self.manifest = dict(manifest)
        self.profile_id = str(self.manifest["subset_id"])
        self.training_fingerprint = str(self.manifest["manifest_sha256"])
        from honf_runtime.source_read_protocol import SOURCE_READ_PROTOCOLS

        source_read = recipe["protocol"] in SOURCE_READ_PROTOCOLS
        inherited_attention = recipe["protocol"] in (
            "inherited_attention_development_v1",
            "inherited_attention_full_train1000_v1",
        )
        if source_read or inherited_attention:
            if any(value is not None for value in (role_component_scales, role_scale_calibration, role_scale_cache_identity)):
                raise ValueError("Wind two-region native protocol must not depend on five-role calibration.")
            self.role_component_scales = None
            self.role_scale_calibration = None
            self.role_scale_cache_identity = None
        else:
            if role_component_scales is None or role_scale_calibration is None or role_scale_cache_identity is None:
                raise ValueError("Historical Wind five-role protocols require their exact TRAIN calibration and cache binding.")
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
        self.allow_formal_protocol = bool(allow_formal_protocol)
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
        if self.allow_formal_protocol:
            if self.seed != 42:
                raise ValueError("Formal-semantic Wind case/query sampling is fixed at sampling_seed=42.")
            if self.effective_batch_size != 16 or not 1 <= self.max_microbatch_cases <= 8:
                raise ValueError("Formal-semantic Wind requires effective batch 16 and microbatches in [1,8].")
            expected_horizon = (
                1000 if recipe["protocol"] == "inherited_attention_full_train1000_v1"
                else 2500
            )
            if recipe["protocol"] in (
                "inherited_attention_development_v1",
                "inherited_attention_full_train1000_v1",
            ) and self.total_epochs != expected_horizon:
                raise ValueError("Inherited-attention Wind horizon differs from its explicit recipe protocol.")
            if recipe["protocol"] not in (
                "inherited_attention_development_v1",
                "inherited_attention_full_train1000_v1",
            ) and self.total_epochs != 2500:
                raise ValueError("Formal Wind A2-D is fixed to its 2500-epoch horizon.")
        else:
            if self.seed != 42:
                raise ValueError("Wind content-hypergraph initialization and sampling are fixed at seed 42.")
            if self.effective_batch_size != 24 or not 1 <= self.max_microbatch_cases <= 24:
                raise ValueError("Wind content-hypergraph uses effective batch 24 and microbatches in [1,24].")
            if self.total_epochs not in (2500, 5000):
                raise ValueError("Wind content-hypergraph requires a 2500 development or 5000 full-follow-up horizon.")
        if self.role_component_scales is not None and (
            set(self.role_component_scales) != set(ROLE_NAMES)
            or any(
                value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0.0)
                for value in self.role_component_scales.values()
            )
        ):
            raise ValueError("Wind role/component scales must be positive finite three-channel m/s values.")
        if tuple(self.view.token_shape) != WIND_CONTENT_ENVIRONMENT_SHAPE:
            raise ValueError("Wind shared core requires the native E64 geometry-only (4,4,4) representation.")
        expected_rows = 90 if self.profile_id == JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID else 24
        if self.validation_rows.size != expected_rows:
            raise ValueError(f"Wind profile {self.profile_id!r} requires {expected_rows} exposed validation rows.")

    def identity_payload(self) -> Mapping[str, Any]:
        model_config = self.model.get_config()
        architecture = _wind_core_architecture(model_config)
        adapter_state = self.adapter.state_payload()
        train_layouts = sorted({int(self.view.metadata["layout_index"][row]) for row in self.train_rows})
        validation_layouts = sorted({int(self.view.metadata["layout_index"][row]) for row in self.validation_rows})
        result = {
            "schema": WIND_CONTENT_PROVIDER_SCHEMA,
            "dataset": "WindFarm",
            "family": WIND_CONTENT_FAMILY,
            "architecture": architecture,
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
            "objective": (
                {
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
                }
                if self.role_component_scales is not None
                else {
                    "name": "not_used_by_protocol",
                    "role_component_scales_mps": None,
                    "scale_calibration": None,
                    "scale_cache_identity": None,
                }
            ),
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
        objective = identity["objective"]
        return {
            "schema": "wind_content_hypergraph_inference_v1",
            "model_config": identity["model_config"],
            "adapter_state": identity["adapter_state"],
            "role_names": list(objective.get("role_names", ())),
            "role_component_scales_mps": objective.get("component_scales_mps"),
            "coordinate_frame": identity["coordinate_frame"],
            "environment_representation": identity["environment_representation"],
            "data_binding": {
                "dataset_profile": identity["dataset_profile"],
                "subset_manifest_sha256": identity["subset_manifest_sha256"],
                "train_row_indices_sha256": identity["train_row_indices_sha256"],
                "normalizer_source": identity["normalizer_and_background_source"],
                "role_scale_cache_identity": objective.get("scale_cache_identity"),
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
            "role_scales_mps": (
                None
                if self.role_component_scales is None
                else {role: values.astype(float).tolist() for role, values in self.role_component_scales.items()}
            ),
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
        if self.role_component_scales is None:
            raise RuntimeError("Five-role normalized training loss is unavailable in the Wind source-read protocol.")
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
        query_count = int(getattr(self, "validation_query_count", self.primary_queries)) if is_validation else self.primary_queries
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
                query_count=query_count,
                track_samples=False,
            )

    def validation_metrics(
        self,
        predictions: WindContentPredictions,
        targets: WindContentTargets,
        auxiliary_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del auxiliary_state
        if self.role_component_scales is None:
            raise RuntimeError("Five-role normalized validation metrics are unavailable in the Wind source-read protocol.")
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
        if arm != _wind_core_architecture(model.get_config()):
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


def _validate_recipe_v2(recipe: Mapping[str, Any]) -> tuple[dict[str, Any], str, int]:
    """Validate only the fresh Round2 Wind recipe schema and architecture IDs."""

    value = dict(recipe)
    if value.get("schema") != "content_hypergraph_recipe_v2" or value.get("dataset") != "wind":
        raise ValueError("Round2 Wind builder requires content_hypergraph_recipe_v2 with dataset='wind'.")
    architecture = str(value.get("architecture"))
    if architecture not in ("A2-D", "A2-H6"):
        raise ValueError("Round2 Wind content-hypergraph supports only A2-D and A2-H6.")
    protocol = str(value.get("protocol"))
    if protocol not in ("development", "full_train1000"):
        raise ValueError("Round2 Wind content-hypergraph requires development or full_train1000 protocol.")
    if value.get("optimizer_bias_norm_exclusion") is not True:
        raise ValueError("Every fresh Round2 Wind recipe must explicitly enable optimizer_bias_norm_exclusion.")
    expected_epochs = 2500 if protocol == "development" else 5000
    if type(value.get("total_epochs")) is not int or value["total_epochs"] != expected_epochs:
        raise ValueError(f"Round2 Wind {protocol} schedule horizon must be {expected_epochs}.")
    if protocol == "full_train1000" and (
        architecture != "A2-H6" or type(value.get("approved_stop_after")) is not int
        or value["approved_stop_after"] != 1000
    ):
        raise ValueError("Round2 Wind full_train1000 is an inert A2-H6 candidate stopped at epoch 1000.")
    if protocol == "development" and "approved_stop_after" in value:
        raise ValueError("Round2 Wind development recipes do not carry a full-data approved stop field.")
    if type(value.get("seed")) is not int or value["seed"] != 42:
        raise ValueError("Round2 Wind recipes are sealed to initialization and sampling seed 42.")
    if type(value.get("primary_queries")) is not int or value["primary_queries"] != 4096:
        raise ValueError("Round2 Wind uses exactly 4096 native receiver queries per case.")
    if type(value.get("effective_batch_size")) is not int or value["effective_batch_size"] != 24:
        raise ValueError("Round2 Wind uses effective case batch size 24.")
    microbatch = value.get("microbatch_size")
    if type(microbatch) is not int or not 1 <= microbatch <= 24:
        raise ValueError("Round2 Wind microbatch_size must be an integer in [1,24].")
    if value.get("precision") != "fp32" or value.get("edge_access") != "full":
        raise ValueError("Round2 Wind requires the full-access FP32 reference path.")
    if value.get("sampling_version", SamplingKey.CASE_EPOCH_VERSION) != SamplingKey.CASE_EPOCH_VERSION:
        raise ValueError("Round2 Wind requires unchanged deterministic case_epoch_v1 sampling.")
    loss_overrides = sorted(
        str(name) for name in value
        if "loss" in str(name).lower() or str(name).lower() == "objective"
    )
    if loss_overrides:
        raise ValueError(f"Round2 Wind native losses are provider-owned; recipe overrides are unsupported: {loss_overrides}.")
    top_level_core_overrides = sorted({
        "module_input_dim", "environment_input_dim", "coordinate_dim", "query_input_dim",
        "operating_input_dim", "output_dim", "context_dim", "num_heads", "context_depth",
        "num_hyperedges", "thermal_response_rank", "local_hidden_dim", "dropout", "init_seed",
    } & set(value))
    if top_level_core_overrides:
        raise ValueError(
            "Round2 Wind dimensions and initialization belong under the sealed model contract; "
            f"top-level overrides are unsupported: {top_level_core_overrides}."
        )
    expected_cache_bytes = (
        WIND_CONTENT_ROLE_CACHE_DEVELOPMENT_BYTES
        if protocol == "development"
        else WIND_CONTENT_ROLE_CACHE_FULL_BYTES
    )
    cache_bytes = value.get("catalogue_cache_ram_bytes")
    if type(cache_bytes) is not int or cache_bytes != expected_cache_bytes:
        raise ValueError(
            f"Round2 Wind {protocol} catalogue cache RAM must be explicitly set to {expected_cache_bytes} bytes."
        )
    model_values = value.get("model")
    if not isinstance(model_values, Mapping):
        raise TypeError("Round2 Wind model overrides must be a mapping.")
    required = {
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 6,
        "thermal_response_rank": 0,
        "local_hidden_dim": 128,
        "dropout": 0.0,
    }
    mode_key = "query_edge_geometry_mode"
    allowed = set(required) | {mode_key}
    if set(model_values) - allowed:
        raise ValueError(f"Unsupported Round2 Wind model overrides: {sorted(set(model_values) - allowed)}")
    if set(model_values) not in (set(required), set(required) | {mode_key}):
        raise ValueError("Round2 Wind model overrides must explicitly declare every frozen shared-core dimension.")
    if any(type(model_values[name]) is not type(expected) or model_values[name] != expected
           for name, expected in required.items()):
        raise ValueError("Round2 Wind model overrides differ from the frozen width-128, four-head core contract.")
    if mode_key in model_values:
        if type(model_values[mode_key]) is not str or model_values[mode_key] != "typed_w_centroids_v1":
            raise ValueError("Round2 Wind supports only query_edge_geometry_mode='typed_w_centroids_v1'.")
        if architecture != "A2-H6":
            raise ValueError("Round2 Wind typed centroid geometry is available only for A2-H6.")
    value["model"] = dict(model_values)
    return value, architecture, microbatch


def _validate_wind_v2_core_binding(recipe: Mapping[str, Any], model_config: Mapping[str, Any]) -> None:
    """Bind the optional Wind R1 recipe flag to the actual core identity."""
    if recipe.get("schema") != "content_hypergraph_recipe_v2" or recipe.get("dataset") != "wind":
        raise ValueError("Wind v2 core binding requires an explicit Round2 Wind recipe.")
    model_recipe = recipe.get("model")
    if not isinstance(model_recipe, Mapping):
        raise TypeError("Round2 Wind model recipe must be a mapping.")
    expected_mode = model_recipe.get("query_edge_geometry_mode")
    expected_config = {
        "schema": "content_hypergraph_core_v2",
        "module_input_dim": 2,
        "environment_input_dim": 7,
        "coordinate_dim": 3,
        "query_input_dim": 7,
        "operating_input_dim": 11,
        "output_dim": 3,
        "architecture": recipe.get("architecture"),
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 6 if recipe.get("architecture") == "A2-H6" else 0,
        "thermal_response_rank": 0,
        "local_hidden_dim": 128,
        "dropout": 0.0,
        "init_seed": recipe.get("seed"),
    }
    backend = model_recipe.get("reader_execution_backend", "legacy")
    conditioning = model_recipe.get("source_value_conditioning", "none")
    if backend not in ("legacy", "reduced_projection"):
        raise ValueError("Wind reader_execution_backend must be legacy or reduced_projection.")
    if conditioning not in ("none", "source_scale"):
        raise ValueError("Wind source_value_conditioning must be none or source_scale.")
    if recipe.get("protocol") not in ("source_read_development_v1", "source_read_full1000") and (
        "reader_execution_backend" in model_recipe or "source_value_conditioning" in model_recipe
    ):
        raise ValueError("Wind reader execution and value-conditioning options are source-read protocol fields.")
    # The v2 core keeps historical defaults out of get_config(). Explicit
    # legacy/none recipe values therefore bind to the same absent core fields.
    if backend != "legacy":
        expected_config["reader_execution_backend"] = backend
    if conditioning != "none":
        expected_config["source_value_conditioning"] = conditioning
    source_read = recipe.get("protocol") in (
        "source_read_development_v1", "source_read_full1000",
    )
    source_read_policy = model_recipe.get("source_read_policy", "full")
    if source_read and source_read_policy != "full":
        expected_config.update(
            source_read_policy=source_read_policy,
            source_read_executor=model_recipe.get("source_read_executor"),
            source_read_rank=model_recipe.get("source_read_rank"),
            source_read_scorer_width=model_recipe.get("source_read_scorer_width"),
        )
        if any(expected_config[name] is None for name in (
            "source_read_policy", "source_read_executor", "source_read_rank", "source_read_scorer_width",
        )):
            raise ValueError("Sparse Wind source-read core identity requires explicit policy/executor/rank/scorer fields.")
    if source_read and model_recipe.get("reader_query_chunk_size") not in (None, 8192):
        raise ValueError("This Wind source-read lineage uses whole-query execution; only its inert8192 recipe field can replay.")
    expected_keys = set(expected_config)
    if expected_mode is not None:
        expected_config["query_edge_geometry_mode"] = expected_mode
        expected_keys.add("query_edge_geometry_mode")
    if set(model_config) != expected_keys:
        raise ValueError("Round2 Wind core configuration contains missing or unsupported fields.")
    if any(
        type(model_config[name]) is not type(expected) or model_config[name] != expected
        for name, expected in expected_config.items()
    ):
        raise ValueError("Round2 Wind core dimensions, outputs, or seed differ from the sealed recipe contract.")
    if expected_mode is None:
        if "query_edge_geometry_mode" in model_config:
            raise ValueError("Default Round2 Wind recipes must retain the exact absent centroid-mode config key.")
    elif (
        recipe.get("architecture") != "A2-H6"
        or expected_mode != "typed_w_centroids_v1"
        or model_config.get("query_edge_geometry_mode") != expected_mode
    ):
        raise ValueError("Round2 Wind recipe and core typed centroid mode do not match the supported A2-H6 R1 contract.")


def _wind_core_architecture(model_config: Mapping[str, Any]) -> str:
    """Read architecture only from the explicit schema that owns it."""
    if model_config.get("schema") in (
        "native_retrieval_core_v1", "native_budgeted_dispatch_v2", "joint_native_hypergraph_v1", "continuous_joint_native_v1", "unified_continuous_joint_native_v1",
    ):
        base = model_config.get("base_model_config")
        if not isinstance(base, Mapping):
            raise ValueError("Native Wind core config is missing its base A2-D identity.")
        return str(base.get("architecture"))
    return str(model_config.get("architecture"))


def _validate_native_wind_core_binding(
    recipe: Mapping[str, Any], model_config: Mapping[str, Any],
) -> None:
    """Allow only the exact no-E NativeRetrievalCore over segmented D0."""
    expected_fields = {
        "schema", "base_model_config", "native_tau_mm", "native_tau_me",
        "native_tau_q", "native_backend", "environment_value_reader",
    }
    if (not isinstance(model_config, Mapping) or set(model_config) != expected_fields
            or model_config.get("schema") != "native_retrieval_core_v1"
            or model_config.get("environment_value_reader") is not False
            or model_config.get("native_backend") not in ("selected", "dense_masked", "compiled_selected")):
        raise ValueError("Wind native retrieval requires the exact no-E native core schema.")
    if (recipe.get("schema") != "content_hypergraph_recipe_v2"
            or recipe.get("dataset") != "wind"
            or recipe.get("protocol") != "source_read_development_v1"
            or recipe.get("source_reader_variant") != "D0"
            or recipe.get("architecture") != "A2-D"):
        raise ValueError("Wind native retrieval binds only to the segmented D0 development objective.")
    base = model_config.get("base_model_config")
    if not isinstance(base, Mapping):
        raise TypeError("Wind native retrieval base model config must be a mapping.")
    _validate_wind_v2_core_binding(recipe, base)


def _validate_budgeted_wind_core_binding(
    recipe: Mapping[str, Any], model_config: Mapping[str, Any],
) -> None:
    """Allow the separately registered Round8 core on the exact Wind D0 provider."""
    expected_fields = {
        "schema", "base_model_config", "native_backend", "routing_seed",
        "membership_bias", "lambda0", "environment_value_reader",
    }
    if (not isinstance(model_config, Mapping) or set(model_config) != expected_fields
            or model_config.get("schema") != "native_budgeted_dispatch_v2"
            or model_config.get("environment_value_reader") is not False
            or model_config.get("native_backend") not in (
                "dense_hard_reference", "flat_selected_reference", "grouped_native",
            )
            or type(model_config.get("routing_seed")) is not int
            or type(model_config.get("membership_bias")) not in (int, float)
            or type(model_config.get("lambda0")) not in (int, float)):
        raise ValueError("Wind Round8 requires the exact no-E budgeted native core schema.")
    if (recipe.get("schema") != "content_hypergraph_recipe_v2"
            or recipe.get("dataset") != "wind"
            or recipe.get("protocol") != "source_read_development_v1"
            or recipe.get("source_reader_variant") != "D0"
            or recipe.get("architecture") != "A2-D"):
        raise ValueError("Wind Round8 binds only to the segmented D0 development objective.")
    base = model_config.get("base_model_config")
    if not isinstance(base, Mapping):
        raise TypeError("Wind Round8 core config must embed its original A2-D base config.")
    _validate_wind_v2_core_binding(recipe, base)


def _validate_joint_wind_core_binding(
    recipe: Mapping[str, Any], model_config: Mapping[str, Any],
) -> None:
    """Bind the direct typed E capability to the unchanged segmented native task."""
    from honf_forward_core.joint_native.model import JointNativeCore

    base = JointNativeCore.validate_config(model_config)
    if (recipe.get("schema") != "content_hypergraph_recipe_v2"
            or recipe.get("dataset") != "wind"
            or recipe.get("protocol") != "source_read_development_v1"
            or recipe.get("source_reader_variant") != "D0"
            or recipe.get("architecture") != "A2-D"):
        raise ValueError("Wind joint-native retrieval binds only to the segmented D0 development objective.")
    _validate_wind_v2_core_binding(recipe, base)


def _apply_wind_source_read_core_options(
    model_config: dict[str, Any],
    model_values: Mapping[str, Any],
    *,
    schema_version: int,
    source_reader_variant: str | None,
) -> None:
    """Apply only explicit sparse-policy fields to new source-read cores."""
    if schema_version != 4 or source_reader_variant not in ("S-Utility", "S-Complete"):
        return
    for name in (
        "source_read_policy", "source_read_executor", "source_read_rank", "source_read_scorer_width",
    ):
        if name not in model_values:
            raise ValueError(f"Wind sparse source-read recipe must set model.{name} explicitly.")
        model_config[name] = model_values[name]
    # f40e18f's two dense controls execute the whole Wind request. Their
    # historical recipe-only8192 field was never applied to the constructor.
    # Children inherit that actual execution, including Q10240 monitoring;
    # do not silently introduce a new tile at the e500 attachment boundary.


def _validate_formal_wind_a2d_recipe_v1(
    recipe: Mapping[str, Any],
) -> tuple[dict[str, Any], str, int]:
    """Validate the separate historical-anchor full-TRAIN A2-D recipe."""
    value = dict(recipe)
    if value.get("schema") != "content_hypergraph_recipe_v2" or value.get("dataset") != "wind":
        raise ValueError("Formal Wind requires content_hypergraph_recipe_v2 with dataset='wind'.")
    if value.get("protocol") != "formal_d_full_train_v1" or value.get("architecture") != "A2-D":
        raise ValueError("Formal Wind builder accepts only protocol formal_d_full_train_v1 and architecture A2-D.")
    expected_fields = {
        "seed": 0,
        "sampling_seed": 42,
        "primary_queries": 8192,
        "volume_queries": 6144,
        "hub_band_queries": 2048,
        "validation_volume_queries": 8192,
        "validation_hub_band_queries": 2048,
        "effective_batch_size": 16,
        "microbatch_size": 8,
        "total_epochs": 2500,
        "approved_stop_after": 2500,
        "precision": "fp32",
        "edge_access": "full",
        "optimizer_bias_norm_exclusion": False,
        "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
        "catalogue_cache_ram_bytes": WIND_CONTENT_ROLE_CACHE_FULL_BYTES,
    }
    for name, expected in expected_fields.items():
        actual = value.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Formal Wind recipe requires {name}={expected!r}.")
    model = value.get("model")
    expected_model = {
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 6,
        "thermal_response_rank": 0,
        "local_hidden_dim": 128,
        "dropout": 0.0,
    }
    if not isinstance(model, Mapping) or dict(model) != expected_model:
        raise ValueError("Formal Wind A2-D fixes the shared E64, width-128 core configuration.")
    optimizer = value.get("optimizer")
    expected_optimizer = {
        "name": "AdamW",
        "lr": 3.0e-4,
        "weight_decay": 1.0e-5,
        "bias_and_norm_weight_decay": 1.0e-5,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
    }
    if not isinstance(optimizer, Mapping) or dict(optimizer) != expected_optimizer:
        raise ValueError("Formal Wind optimizer must be one constant-LR AdamW group with 1e-5 decay on all weights.")
    execution = value.get("execution")
    curve_every = execution.get("curve_every") if isinstance(execution, Mapping) else None
    # Fresh formal recipes use 10; 100 is retained only as the explicit
    # historical recipe value for replay. Resume policy is enforced by CLI.
    if type(curve_every) is not int or curve_every not in (10, 100):
        raise ValueError("Formal Wind curve_every must be 10 (default) or 100 (legacy replay).")
    expected_execution = {
        "monitor_every": 5,
        "latest_every": 5,
        "curve_every": curve_every,
        "checkpoint_epochs": [100, 500, 1000, 1500, 2000, 2500],
        "field_metric": "val_volume_mse",
    }
    execution = value.get("execution")
    if not isinstance(execution, Mapping) or dict(execution) != expected_execution:
        raise ValueError("Formal Wind execution monitoring/checkpointing differs from the approved field selector contract.")
    expected_objective = {
        "volume_sampling_fraction": 0.75,
        "volume_loss_weight": 0.75,
        "hub_band_loss_weight": 0.25,
        "validation_volume_loss_weight": 0.8,
        "validation_hub_band_loss_weight": 0.2,
        "channel_weights": [1.0, 1.0, 1.0],
        "sampler": "native_uniform_with_replacement_volume_and_hub_band_v1",
    }
    objective = value.get("objective")
    if not isinstance(objective, Mapping) or dict(objective) != expected_objective:
        raise ValueError("Formal Wind objective must retain the historical volume/hub-band sampling and weighting.")
    allowed = {
        *expected_fields,
        "schema", "dataset", "architecture", "protocol", "model", "optimizer", "execution", "objective",
        "data_root", "derived_root", "role_scale_cache_dir",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unsupported formal Wind recipe fields: {sorted(unknown)}")
    return value, "A2-D", int(value["microbatch_size"])


def _validate_source_read_wind_recipe_v1(
    recipe: Mapping[str, Any],
) -> tuple[dict[str, Any], str, int]:
    """Validate Wind's segmented or conditional-full source-read recipe."""
    from honf_runtime.source_read_protocol import (
        SOURCE_READ_DEVELOPMENT,
        SOURCE_READ_FULL,
        validate_source_read_recipe,
    )

    value = validate_source_read_recipe(recipe)
    if value.get("dataset") != "wind":
        raise ValueError("Wind source-read provider accepts only dataset='wind'.")
    if value.get("protocol") not in (SOURCE_READ_DEVELOPMENT, SOURCE_READ_FULL):
        raise ValueError("Wind source-read provider requires an explicit development or full1000 scope.")
    if value.get("sampling_version") != SamplingKey.CASE_EPOCH_VERSION:
        raise ValueError("Wind source-read sampling must use case_epoch_v1.")
    if type(value.get("catalogue_cache_ram_bytes")) is not int or value["catalogue_cache_ram_bytes"] != WIND_SOURCE_READ_ROLE_CACHE_RAM_BYTES:
        raise ValueError(
            f"Wind source-read catalogue RAM must be explicitly capped at {WIND_SOURCE_READ_ROLE_CACHE_RAM_BYTES} bytes."
        )
    if value.get("source_reader_variant") not in ("D0", "D-scale", "S-Utility", "S-Complete"):
        raise ValueError("Wind source-read recipes require an explicit supported source_reader_variant.")
    return value, "A2-D", int(value["microbatch_size"])


class WindContentHypergraphTaskV2(WindContentHypergraphTask):
    """Round2 provider with strict A2 dispatch and bounded core tracing."""

    def __init__(
        self,
        view: WindFarmNativeView,
        *,
        model: nn.Module,
        adapter: WindContentHypergraphAdapterV3,
        recipe: Mapping[str, Any],
        _allow_formal_protocol: bool = False,
        _allow_source_read_protocol: bool = False,
        _allow_native_retrieval: bool = False,
        **kwargs: Any,
    ) -> None:
        from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
        from honf_runtime.source_read_protocol import SOURCE_READ_PROTOCOLS

        if recipe.get("schema") != "content_hypergraph_recipe_v2":
            raise ValueError("Round2 Wind provider requires a content_hypergraph_recipe_v2.")
        is_historical_formal = recipe.get("protocol") == "formal_d_full_train_v1"
        is_source_read = recipe.get("protocol") in SOURCE_READ_PROTOCOLS
        expected_formal_flag = is_historical_formal or (_allow_source_read_protocol and is_source_read)
        if _allow_formal_protocol is not expected_formal_flag:
            raise ValueError("The formal Wind protocol is accepted only by WindFormalA2DTaskV1.")
        if _allow_source_read_protocol is not is_source_read:
            raise ValueError("Wind source-read scope is accepted only by WindSourceReadTaskV1.")
        if not isinstance(adapter, WindContentHypergraphAdapterV3):
            raise TypeError("Round2 Wind provider requires WindContentHypergraphAdapterV3.")
        if not isinstance(model, ContentHypergraphCoreV2):
            raise TypeError("Round2 Wind provider requires ContentHypergraphCoreV2.")
        config = model.get_config()
        if _allow_native_retrieval:
            _validate_native_wind_core_binding(recipe, config)
        else:
            _validate_wind_v2_core_binding(recipe, config)
        self._allow_native_retrieval = bool(_allow_native_retrieval)
        super().__init__(
            view,
            model=model,
            adapter=adapter,
            recipe=recipe,
            allow_formal_protocol=_allow_formal_protocol,
            **kwargs,
        )

    def identity_payload(self) -> Mapping[str, Any]:
        result = dict(super().identity_payload())
        result["schema"] = WIND_CONTENT_PROVIDER_SCHEMA_V2
        result["family"] = WIND_CONTENT_FAMILY_V2
        result["model_family"] = "honf_content_hypergraph"
        result["coordinate_frame"] = {
            **dict(result["coordinate_frame"]),
            "origin": "support.lower_D",
            "extent": "support.extent_D",
            "module_characteristic_length": "rotor_diameter_D = 2 * rotor_radius_D, per axis",
            "environment_characteristic_length": "support.extent_D / [4,4,4], per E64 cell and axis",
            "object_length_is_local_reader_radius": False,
        }
        result["environment_representation"] = {
            **dict(result["environment_representation"]),
            "characteristic_length": "support.extent_D / [4,4,4], per axis",
            "length_feature_is_physical_sample_count": False,
        }
        result["training"] = {
            **dict(result["training"]),
            "core_schema": "content_hypergraph_core_v2",
            "trace_policy": "trace_request is opt-in and bounded; ordinary prediction requests no trace",
        }
        return result

    def serialization_payload(self) -> Mapping[str, Any]:
        result = dict(super().serialization_payload())
        result["schema"] = "wind_content_hypergraph_inference_v2"
        result["coordinate_frame"] = dict(self.identity_payload()["coordinate_frame"])
        return result

    def predict_native(
        self,
        model: nn.Module,
        scene: SceneBatch,
        receivers: WindContentReceivers,
        execution_mode: str = "A2-D",
        phase: str = "joint",
        epoch: int = 0,
        temperature: float | None = None,
        trace_request: Any | None = None,
    ) -> tuple[WindContentPredictions, Mapping[str, Any]]:
        del epoch, temperature
        if phase not in ("joint", "development", "validation"):
            raise ValueError("Round2 Wind prediction uses one joint path for TRAIN and exposed DEV.")
        if execution_mode not in ("A2-D", "A2-H6"):
            raise ValueError("Round2 Wind only supports A2-D and A2-H6 matched architectures.")
        from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2

        if not isinstance(model, ContentHypergraphCoreV2):
            raise TypeError("Round2 Wind provider requires ContentHypergraphCoreV2; v1 cores are not accepted.")
        config = model.get_config()
        if config.get("schema") == "native_retrieval_core_v1":
            if not getattr(self, "_allow_native_retrieval", False):
                raise ValueError("Native retrieval inference requires its explicit runtime capability.")
            _validate_native_wind_core_binding(self.recipe, config)
        elif config.get("schema") == "native_budgeted_dispatch_v2":
            if not getattr(self, "_allow_budgeted_native", False):
                raise ValueError("Round8 native dispatch requires its explicit runtime capability.")
            _validate_budgeted_wind_core_binding(self.recipe, config)
        elif config.get("schema") in ("continuous_joint_native_v1","unified_continuous_joint_native_v1"):
            if not getattr(self,"_allow_continuous_joint",False):
                raise ValueError('Continuous Wind core requires its explicit provider binding.')
            from honf_forward_core.continuous_joint.model import ContinuousJointCore
            if config.get('schema')=='unified_continuous_joint_native_v1':
                from honf_forward_core.continuous_joint.unified import UnifiedJointCore as ContinuousJointCore
            ContinuousJointCore.validate_config(config)
            _validate_wind_v2_core_binding(self.recipe,config['base_model_config'])
        elif config.get("schema") == "joint_native_hypergraph_v1":
            if not getattr(self, "_allow_joint_native", False):
                raise ValueError("Joint native M/E prediction requires its explicit runtime capability.")
            _validate_joint_wind_core_binding(self.recipe, config)
        else:
            _validate_wind_v2_core_binding(self.recipe, config)
        if _wind_core_architecture(config) != execution_mode:
            raise ValueError("Round2 Wind execution arm differs from the core's sealed architecture identity.")
        query = self.adapter.make_query_batch(scene, receivers)
        organization = model.prepare(scene, trace_request=trace_request)
        output = model.decode(organization, query)
        if not torch.is_tensor(output.nonlinear_field) or output.nonlinear_field.shape != query.coordinates.shape[:2] + (3,):
            raise ValueError("Round2 Wind core must return a normalized nonlinear residual field [B,Q,3].")
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
            "trace_request": trace_request,
        }
        return predictions, auxiliary

    def optimizer_groups(
        self,
        model: nn.Module,
        arm: str = "A2-D",
        stage: str = "joint",
    ) -> Sequence[OptimizerGroupSpec]:
        return super().optimizer_groups(model, arm=arm, stage=stage)

    def work_counts(
        self,
        batch: TaskBatch,
        predictions: WindContentPredictions,
        auxiliary_state: Mapping[str, Any],
    ) -> Mapping[str, int]:
        counts = dict(super().work_counts(batch, predictions, auxiliary_state))
        for prefix, values in (
            ("core_prepare", auxiliary_state["organization"].work_counts),
            ("core_decode", auxiliary_state["core_output"].work_counts),
        ):
            if not isinstance(values, Mapping):
                raise RuntimeError("Round2 Wind prediction omitted complete shared-core work counts.")
            for name, value in values.items():
                if type(value) is not int or value < 0:
                    raise RuntimeError("Round2 Wind core work counts must be nonnegative integers.")
                counts[f"{prefix}_{name}"] = value
        return counts


class WindFormalA2DTaskV1(WindContentHypergraphTaskV2):
    """Historical-anchor full-TRAIN objective for a fresh Round2 A2-D fit."""

    LOSS_NAME = "formal_wind/native_velocity_mse"
    VOLUME_ROLE_ID = 0
    HUB_BAND_ROLE_ID = 1
    VOLUME_QUERIES = 6144
    HUB_BAND_QUERIES = 2048
    VALIDATION_VOLUME_QUERIES = 8192
    VALIDATION_HUB_BAND_QUERIES = 2048
    VALIDATION_QUERY_COUNT = VALIDATION_VOLUME_QUERIES + VALIDATION_HUB_BAND_QUERIES
    VOLUME_WEIGHT = 0.75
    HUB_BAND_WEIGHT = 0.25
    VALIDATION_VOLUME_WEIGHT = 0.8
    VALIDATION_HUB_BAND_WEIGHT = 0.2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, _allow_formal_protocol=True, **kwargs)
        self.initialization_seed = int(self.recipe["seed"])
        self.sampling_seed = int(self.recipe["sampling_seed"])
        if self.initialization_seed != 0 or self.sampling_seed != self.seed:
            raise ValueError("Formal Wind initialization and sampling seeds must remain separately bound.")
        self.role_query_counts = {"volume": self.VOLUME_QUERIES, "hub_band": self.HUB_BAND_QUERIES}
        self.validation_role_query_counts = {
            "volume": self.VALIDATION_VOLUME_QUERIES,
            "hub_band": self.VALIDATION_HUB_BAND_QUERIES,
        }
        self.validation_query_count = self.VALIDATION_QUERY_COUNT

    def _two_region_identity_payload(self, *, source_read: bool) -> Mapping[str, Any]:
        result = dict(super().identity_payload())
        if source_read:
            result["protocol"] = self.recipe["protocol"]
            result["source_read_scope"] = self.recipe["protocol"]
        else:
            result["formal_protocol"] = "formal_d_full_train_v1"
            result["test_row_indices_sha256"] = self.manifest["test_row_indices_sha256"]
        result["objective"] = {
            "name": (
                "source_read_formal_two_region_standardized_mse"
                if source_read
                else "historical_volume_hub_band_standardized_mse"
            ),
            "role_names": ["volume", "hub_band"],
            "role_query_counts": dict(self.role_query_counts),
            "validation_role_query_counts": dict(self.validation_role_query_counts),
            "volume_weight": self.VOLUME_WEIGHT,
            "hub_band_weight": self.HUB_BAND_WEIGHT,
            "validation_volume_weight": self.VALIDATION_VOLUME_WEIGHT,
            "validation_hub_band_weight": self.VALIDATION_HUB_BAND_WEIGHT,
            "channel_weights": [1.0, 1.0, 1.0],
            "formula": (
                "training: mean over cases of 0.75*standardized volume MSE + 0.25*standardized hub-band MSE; "
                "validation loss: 0.8*standardized volume MSE + 0.2*standardized hub-band MSE on fixed Q10240; "
                "field selection: val_volume_mse on the first 8192 fixed validation volume draws"
            ),
            "selection_metric": (
                "val_volume_mse: mean per-TRAIN-normalized native volume MSE over fixed VALID draws; "
                "validation draws are Q10240=8192 volume + 2048 hub-band"
            ),
            "role_component_scales_mps_are_training_objective_inputs": False,
            "target_partition_access": "TRAIN targets for optimization; fixed VALID targets for monitoring; TEST targets locked",
            "sampling": "native volume and hub-band draws with replacement; SeedSequence(sampling_seed, epoch, row, stream_id 101/102)",
        }
        training = dict(result["training"])
        training.update(
            {
                "initialization_seed": self.initialization_seed,
                "sampling_seed": self.sampling_seed,
                "effective_batch_size": 16,
                "microbatch_size": 8,
                "primary_queries_per_case": 8192,
                "validation_queries_per_case": self.VALIDATION_QUERY_COUNT,
                "total_epochs": self.total_epochs,
                "optimizer": "AdamW",
                "optimizer_schedule": asdict(self._schedule()),
                "optimizer_weight_decay": 1.0e-5,
                "bias_and_norm_weight_decay": 1.0e-5,
                "optimizer_bias_norm_exclusion": False,
                "optimizer_group_count": 1,
                "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            }
        )
        result["training"] = training
        result["native_role_catalogue_cache"]["roles"] = ["volume", "hub_band"]
        result["native_role_catalogue_cache"]["usage"] = (
            "not used by the two-region training objective"
            if source_read
            else "not used by the historical two-region objective"
        )
        full = self.profile_id == JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID
        fit_rows = (
            "all 420 original TRAIN rows only"
            if full
            else "the fixed 72-row TRAIN cohort only"
        )
        result["normalizer_and_background_source"] = {
            **dict(result["normalizer_and_background_source"]),
            "fit_rows": fit_rows,
            "fit_seed": self.sampling_seed,
            "test_target_values_read": False,
        }
        return result

    def identity_payload(self) -> Mapping[str, Any]:
        return self._two_region_identity_payload(source_read=False)

    def preparation_summary(self) -> Mapping[str, Any]:
        result = dict(super().preparation_summary())
        result.update(
            {
                "formal_protocol": "formal_d_full_train_v1",
                "initialization_seed": self.initialization_seed,
                "sampling_seed": self.sampling_seed,
                "role_query_counts": dict(self.role_query_counts),
                "validation_role_query_counts": dict(self.validation_role_query_counts),
                "validation_queries_per_case": self.VALIDATION_QUERY_COUNT,
                "optimizer_updates_per_epoch": math.ceil(len(self.train_rows) / self.effective_batch_size),
                "wind_test_target_values_read": False,
            }
        )
        return result

    def validation_batches(self) -> Iterable[TaskBatch]:
        """Expose the historical fixed Q10240 selector panel, separate from train Q8192."""
        return self.validation_batches_for(self.VALIDATION_QUERY_COUNT)

    def _formal_rng(self, row: int, epoch: int, stream_id: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence((self.sampling_seed, int(epoch), int(row), int(stream_id)))
        )

    def _formal_query_hash(
        self,
        rows: Sequence[int],
        volume_samples: Sequence[Any],
        band_samples: Sequence[Any],
        key: SamplingKey,
        *,
        validation: bool,
    ) -> str:
        if len(rows) != len(volume_samples) or len(rows) != len(band_samples):
            raise ValueError("Formal Wind query receipt rows and region samples differ in length.")
        epoch = 0 if validation else int(key.epoch)
        sampling_reference_scope = getattr(self, "sampling_reference_scope", None)
        reference_protocol = (
            sampling_reference_scope.get("protocol")
            if isinstance(sampling_reference_scope, Mapping) else None
        )
        protocol = reference_protocol or getattr(self, "recipe", {}).get("protocol", "formal_d_full_train_v1")
        payload = {
            "protocol": protocol,
            "sampling_seed": self.sampling_seed,
            "epoch": epoch,
            "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            "dataset_id": key.dataset_id,
            "region_query_counts": (
                self.validation_role_query_counts if validation else self.role_query_counts
            ),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        for row, volume, band in zip(rows, volume_samples, band_samples):
            digest.update(int(row).to_bytes(8, "little", signed=False))
            for role_id, sample in ((self.VOLUME_ROLE_ID, volume), (self.HUB_BAND_ROLE_ID, band)):
                digest.update(bytes((role_id,)))
                digest.update(np.ascontiguousarray(sample.flat_indices, dtype=np.int64).tobytes())
                digest.update(np.ascontiguousarray(sample.coords_D, dtype=np.float32).tobytes())
        return digest.hexdigest()

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
        del stream_validation
        expected_query_count = self.VALIDATION_QUERY_COUNT if validation else self.primary_queries
        if query_count != expected_query_count:
            if validation:
                raise ValueError("Formal Wind validation batches use exactly Q10240=8192 volume + 2048 hub-band.")
            raise ValueError("Formal Wind training batches use exactly Q8192=6144 volume + 2048 hub-band queries.")
        if len(rows) == 0 or len(rows) > self.max_microbatch_cases:
            raise ValueError("Formal Wind microbatch is empty or exceeds its declared case limit.")
        row_ids = tuple(int(value) for value in rows)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Formal Wind batches cannot repeat native rows.")
        permitted = self._validation_row_set if validation else self._train_row_set
        if any(row not in permitted for row in row_ids):
            raise ValueError("Formal Wind batch contains a row outside its sealed TRAIN/VALID membership.")
        expected_dataset_id = _runtime_sampling_dataset_id(self.identity_payload())
        if (
            key.sampling_version != SamplingKey.CASE_EPOCH_VERSION
            or key.seed != self.sampling_seed
            or key.dataset_id != expected_dataset_id
        ):
            raise ValueError("Formal Wind sampling key differs from its full-TRAIN case_epoch_v1 identity.")
        sample_epoch = 0 if validation else int(key.epoch)
        scenes: list[WindContentSceneInputs] = []
        coordinates: list[np.ndarray] = []
        features: list[np.ndarray] = []
        flat_indices: list[np.ndarray] = []
        role_ids: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        local_batches: list[np.ndarray] = []
        local_queries: list[np.ndarray] = []
        local_modules: list[np.ndarray] = []
        volume_samples: list[Any] = []
        band_samples: list[Any] = []
        metadata: list[Mapping[str, Any]] = []
        for batch_index, row in enumerate(row_ids):
            case = self.view.run(row)
            scene_inputs = self.adapter.scene_inputs(case)
            volume_queries = self.VALIDATION_VOLUME_QUERIES if validation else self.VOLUME_QUERIES
            hub_band_queries = self.VALIDATION_HUB_BAND_QUERIES if validation else self.HUB_BAND_QUERIES
            volume_sample = case.sample_queries(
                volume_queries,
                self._formal_rng(row, sample_epoch, 101),
                mode="volume",
            )
            band_sample = case.sample_queries(
                hub_band_queries,
                self._formal_rng(row, sample_epoch, 102),
                mode="hub_band",
            )
            query_coordinates = np.concatenate((volume_sample.coords_D, band_sample.coords_D), axis=0)
            query_target = np.concatenate((volume_sample.velocity_mps, band_sample.velocity_mps), axis=0)
            query_features = self.adapter.query_features(query_coordinates, case.support)
            volume_role_ids = np.full(volume_queries, self.VOLUME_ROLE_ID, dtype=np.int8)
            band_role_ids = np.full(hub_band_queries, self.HUB_BAND_ROLE_ID, dtype=np.int8)
            query_role_ids = np.concatenate((volume_role_ids, band_role_ids))
            local_query, local_module = self.adapter.packed_local_pair_indices(scene_inputs, query_coordinates)
            scenes.append(scene_inputs)
            coordinates.append(np.asarray(query_coordinates, dtype=np.float32))
            features.append(np.asarray(query_features, dtype=np.float32))
            flat_indices.append(
                np.concatenate((volume_sample.flat_indices, band_sample.flat_indices)).astype(np.int64, copy=False)
            )
            role_ids.append(query_role_ids)
            targets.append(np.asarray(query_target, dtype=np.float32))
            local_batches.append(np.full(local_query.size, batch_index, dtype=np.int64))
            local_queries.append(np.asarray(local_query, dtype=np.int64))
            local_modules.append(np.asarray(local_module, dtype=np.int64))
            volume_samples.append(volume_sample)
            band_samples.append(band_sample)
            metadata.append(
                {
                    "row_index": row,
                    "case": scene_inputs.case_name,
                    "layout_index": scene_inputs.layout_index,
                    "wind_direction_deg": scene_inputs.wind_direction_deg,
                    "n_turbines": int(np.count_nonzero(scene_inputs.module_present > 0.5)),
                    "volume_query_count": volume_queries,
                    "hub_band_query_count": hub_band_queries,
                }
            )
        receivers = WindContentReceivers(
            np.stack(coordinates),
            np.stack(features),
            np.stack(flat_indices),
            np.stack(role_ids),
            tuple(scene.scene_key for scene in scenes),
            np.concatenate(local_batches),
            np.concatenate(local_queries),
            np.concatenate(local_modules),
        )
        target_batch = WindContentTargets(np.stack(targets), row_ids, tuple(metadata))
        if track_samples:
            if validation:
                self.validation_samples += len(row_ids)
            else:
                self.training_samples += len(row_ids)
        return TaskBatch(
            scene_inputs=tuple(scenes),
            receivers=receivers,
            targets=target_batch,
            auxiliary={
                "role_names": ("volume", "hub_band"),
                "role_query_counts": dict(
                    self.validation_role_query_counts if validation else self.role_query_counts
                ),
                "query_sampling_sha256": self._formal_query_hash(
                    row_ids,
                    volume_samples,
                    band_samples,
                    key,
                    validation=validation,
                ),
            },
            case_keys=row_ids,
        )

    def loss_denominators(
        self,
        batches: Sequence[TaskBatch],
        phase: str = "joint",
        arm: str = "A2-D",
    ) -> Mapping[str, float]:
        del arm
        if phase != "joint":
            raise ValueError("Formal Wind A2-D uses one joint training objective.")
        denominator = 0
        for batch in batches:
            role_ids = np.asarray(batch.receivers.role_ids)
            if role_ids.shape != (len(batch.case_keys), 8192):
                raise ValueError("Formal Wind role IDs must retain the complete [B,8192] native panel.")
            expected = np.concatenate(
                (
                    np.zeros(self.VOLUME_QUERIES, dtype=np.int8),
                    np.ones(self.HUB_BAND_QUERIES, dtype=np.int8),
                )
            )
            if any(not np.array_equal(row, expected) for row in role_ids):
                raise ValueError("Formal Wind native volume/hub-band partition changed its sealed order or counts.")
            if not isinstance(batch.targets, WindContentTargets):
                raise TypeError("Formal Wind loss requires native WindContentTargets.")
            if tuple(batch.targets.row_indices) != tuple(int(value) for value in batch.case_keys):
                raise ValueError("Formal Wind target rows differ from their microbatch case keys.")
            denominator += len(batch.case_keys)
        return {self.LOSS_NAME: float(denominator)}

    def _formal_loss_terms(
        self,
        predictions: WindContentPredictions,
        targets: WindContentTargets,
        *,
        validation: bool = False,
    ) -> Mapping[str, LossTerm]:
        target = torch.tensor(
            targets.velocity_mps,
            device=predictions.main_mps.device,
            dtype=predictions.main_mps.dtype,
        )
        query_count = self.VALIDATION_QUERY_COUNT if validation else self.primary_queries
        volume_queries = self.VALIDATION_VOLUME_QUERIES if validation else self.VOLUME_QUERIES
        if target.shape != predictions.main_mps.shape or target.shape[1] != query_count:
            raise ValueError(f"Formal Wind target panel must align with [B,{query_count},3] predictions.")
        scale = torch.as_tensor(
            np.array(self.adapter.normalizer.safe_std, copy=True) * self.adapter.normalizer.u_ref_mps,
            device=predictions.main_mps.device,
            dtype=predictions.main_mps.dtype,
        )
        error = (predictions.main_mps - target) / scale
        squared = error.square()
        volume_per_case = squared[:, :volume_queries].mean(dim=(1, 2))
        band_per_case = squared[:, volume_queries:].mean(dim=(1, 2))
        volume_weight = self.VALIDATION_VOLUME_WEIGHT if validation else self.VOLUME_WEIGHT
        band_weight = self.VALIDATION_HUB_BAND_WEIGHT if validation else self.HUB_BAND_WEIGHT
        per_case = volume_weight * volume_per_case + band_weight * band_per_case
        return {self.LOSS_NAME: LossTerm(per_case.sum(), float(per_case.numel()), weight=1.0)}

    def loss_terms(
        self,
        predictions: WindContentPredictions,
        targets: WindContentTargets,
        phase: str,
        auxiliary_state: Any,
    ) -> Mapping[str, LossTerm]:
        del auxiliary_state
        if phase != "joint":
            raise ValueError("Formal Wind A2-D uses one joint training objective.")
        return self._formal_loss_terms(predictions, targets)

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
        return self._formal_loss_terms(predictions, targets, validation=True)

    def loss_metadata(self) -> Mapping[str, Mapping[str, str]]:
        return {
            self.LOSS_NAME: {
                "panel": "native_prediction",
                "label": "standardized Wind velocity",
                "unit": "dimensionless mean squared error",
                "formula": (
                    "TRAIN: equal-case 0.75 volume MSE + 0.25 hub-band MSE; "
                    "VALID loss: 0.8 volume MSE + 0.2 hub-band MSE; "
                    "field selector: volume MSE only; channels Ux/Uy/Uz equally weighted"
                ),
                "weight": "1.0",
            }
        }

    def _schedule(self) -> ScheduleSpec:
        optimizer = self.recipe["optimizer"]
        learning_rate = float(optimizer["lr"])
        return ScheduleSpec(
            peak_lr=learning_rate,
            warmup_start_lr=learning_rate,
            warmup_epochs=0,
            hold_through_epoch=2499,
            total_epochs=2500,
            final_lr=learning_rate,
        )

    def _weight_decay(self) -> float:
        return float(self.recipe["optimizer"]["weight_decay"])

    def optimizer_groups(
        self,
        model: nn.Module,
        arm: str = "A2-D",
        stage: str = "joint",
    ) -> Sequence[OptimizerGroupSpec]:
        del stage
        if model is not self.model or arm != "A2-D" or _wind_core_architecture(model.get_config()) != "A2-D":
            raise ValueError("Formal Wind optimizer group is bound only to this A2-D model.")
        optimizer = self.recipe["optimizer"]
        if (
            optimizer.get("name") != "AdamW"
            or float(optimizer["weight_decay"]) != 1.0e-5
            or float(optimizer["bias_and_norm_weight_decay"]) != 1.0e-5
            or self.recipe.get("optimizer_bias_norm_exclusion") is not False
        ):
            raise ValueError("Formal Wind requires one AdamW group with 1e-5 decay on every trainable parameter.")
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not names or names != tuple(name for name, _ in model.named_parameters()):
            raise ValueError("Every formal Wind core parameter must train jointly from initialization.")
        return (
            OptimizerGroupSpec(
                "wind_formal_adamw_all_parameters",
                names,
                self._schedule(),
                weight_decay=float(optimizer["weight_decay"]),
                betas=tuple(float(value) for value in optimizer["betas"]),
                eps=float(optimizer["eps"]),
            ),
        )

    def validation_metrics(
        self,
        predictions: WindContentPredictions,
        targets: WindContentTargets,
        auxiliary_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del auxiliary_state
        predicted = predictions.main_mps.detach().cpu().numpy().astype(np.float64, copy=False)
        actual = np.asarray(targets.velocity_mps, dtype=np.float64)
        expected_shape = (len(targets.row_indices), self.VALIDATION_QUERY_COUNT, 3)
        if predicted.shape != expected_shape or actual.shape != expected_shape:
            raise ValueError(
                "Formal Wind validation metrics require the exact Q10240=8192 volume + 2048 hub-band panel."
            )
        scale = np.asarray(self.adapter.normalizer.safe_std, dtype=np.float64) * float(
            self.adapter.normalizer.u_ref_mps
        )
        standardized_squared = np.square((predicted - actual) / scale.reshape(1, 1, 3))
        rows: list[dict[str, Any]] = []
        for index, metadata in enumerate(targets.case_metadata):
            volume = standardized_squared[index, : self.VALIDATION_VOLUME_QUERIES]
            band = standardized_squared[index, self.VALIDATION_VOLUME_QUERIES :]
            volume_mse = float(np.mean(volume, dtype=np.float64))
            band_mse = float(np.mean(band, dtype=np.float64))
            rows.append(
                {
                    **dict(metadata),
                    "val_volume_mse": volume_mse,
                    "val_band_mse": band_mse,
                    "val_weighted_mse": (
                        self.VALIDATION_VOLUME_WEIGHT * volume_mse
                        + self.VALIDATION_HUB_BAND_WEIGHT * band_mse
                    ),
                    "volume_vector_rmse_mps": float(
                        np.sqrt(np.mean(np.sum(np.square(predicted[index, : self.VALIDATION_VOLUME_QUERIES] - actual[index, : self.VALIDATION_VOLUME_QUERIES]), axis=-1)))
                    ),
                    "hub_band_vector_rmse_mps": float(
                        np.sqrt(np.mean(np.sum(np.square(predicted[index, self.VALIDATION_VOLUME_QUERIES :] - actual[index, self.VALIDATION_VOLUME_QUERIES :]), axis=-1)))
                    ),
                }
            )
        return {"rows": rows}

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
            raise ValueError("Formal Wind validation metrics must contain the exact VALID rows once.")
        return {
            "val_volume_mse": float(np.mean([row["val_volume_mse"] for row in rows], dtype=np.float64)),
            "val_band_mse": float(np.mean([row["val_band_mse"] for row in rows], dtype=np.float64)),
            "val_weighted_mse": float(np.mean([row["val_weighted_mse"] for row in rows], dtype=np.float64)),
            "validation_rows": len(rows),
            "rows": rows,
        }


class WindSourceReadTaskV1(WindFormalA2DTaskV1):
    """Round3 Wind scope using the successful formal two-region training path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        recipe = kwargs.get("recipe")
        if not isinstance(recipe, Mapping):
            raise TypeError("Wind source-read provider requires its bound recipe mapping.")
        _validate_source_read_wind_recipe_v1(recipe)
        # This scope shares formal-D's fixed sampler and batch constraints,
        # while keeping its own explicit protocol identity and stop policy.
        WindContentHypergraphTaskV2.__init__(
            self,
            *args,
            _allow_formal_protocol=True,
            _allow_source_read_protocol=True,
            **kwargs,
        )
        self.initialization_seed = int(self.recipe["seed"])
        self.sampling_seed = int(self.recipe["sampling_seed"])
        if self.initialization_seed != 0 or self.sampling_seed != self.seed:
            raise ValueError("Wind source-read initialization seed 0 and sampling seed 42 are separately bound.")
        self.role_query_counts = {"volume": self.VOLUME_QUERIES, "hub_band": self.HUB_BAND_QUERIES}
        self.validation_role_query_counts = {
            "volume": self.VALIDATION_VOLUME_QUERIES,
            "hub_band": self.VALIDATION_HUB_BAND_QUERIES,
        }
        self.validation_query_count = self.VALIDATION_QUERY_COUNT
        self.source_reader_variant = str(self.recipe["source_reader_variant"])

    def _set_source_read_training_epoch(self, model: nn.Module, epoch: int) -> None:
        if model is not self.model:
            raise ValueError("Wind source-read epoch dispatch requires its bound model.")
        setter = getattr(model, "set_source_read_training_epoch", None)
        if not callable(setter):
            raise RuntimeError("Wind source-read core is missing its epoch-bound selection transition.")
        setter(epoch)

    def predict_native(
        self,
        model: nn.Module,
        scene: SceneBatch,
        receivers: WindContentReceivers,
        execution_mode: str = "A2-D",
        phase: str = "joint",
        epoch: int = 0,
        temperature: float | None = None,
        trace_request: Any | None = None,
    ) -> tuple[WindContentPredictions, Mapping[str, Any]]:
        self._set_source_read_training_epoch(model, epoch)
        return super().predict_native(
            model, scene, receivers, execution_mode, phase,
            epoch=epoch, temperature=temperature, trace_request=trace_request,
        )

    def source_read_positive_train_envelope(self, train_case_keys: Sequence[Any]) -> float:
        """Wind has no heating control; retain the protocol's unused unit envelope."""
        keys = tuple(int(value) for value in train_case_keys)
        if keys != tuple(int(value) for value in self.train_rows):
            raise ValueError("Wind source-read envelope requires its exact ordered TRAIN rows.")
        return 1.0

    def _source_read_target_free_receivers(
        self,
        case_key: Any,
        key: SamplingKey,
    ) -> tuple[WindContentSceneInputs, WindContentReceivers]:
        """Draw the native role mixture from geometry catalogues without loading run.U."""
        row = int(case_key)
        if row not in self._train_row_set:
            raise ValueError("Wind source-read teacher probes require a sealed TRAIN row.")
        case = self.view.run(row)
        scene_inputs = self.adapter.scene_inputs(case)
        catalogue = self.catalogue_cache.get(case)
        physical_id = _physical_case_direction_id(case)
        stream = "wind_joint_native_role_query"
        role_counts = _role_counts_for_query_count(self.primary_queries)
        coordinates: list[np.ndarray] = []
        flat_indices: list[np.ndarray] = []
        for role in ROLE_NAMES:
            cdf = catalogue.role_cdf[role]
            positions = np.searchsorted(
                cdf,
                key.numpy_rng(stream, physical_id, role).random(role_counts[role]),
                side="right",
            )
            positions = np.minimum(positions, cdf.size - 1)
            valid = catalogue.role_indices.get(role)
            selected = positions if valid is None else valid[positions]
            coordinates.append(np.asarray(catalogue.coordinates_D[selected], dtype=np.float32))
            flat_indices.append(np.asarray(selected, dtype=np.int64))
        coords = np.concatenate(coordinates, axis=0)
        flat = np.concatenate(flat_indices, axis=0)
        features = self.adapter.query_features(coords, case.support)
        local_query, local_module = self.adapter.packed_local_pair_indices(scene_inputs, coords)
        receivers = WindContentReceivers(
            coordinates_D=coords[None],
            features=np.asarray(features, dtype=np.float32)[None],
            flat_indices=flat[None],
            role_ids=_role_ids(role_counts)[None],
            scene_keys=(scene_inputs.scene_key,),
            local_batch_indices=np.zeros(local_query.shape, dtype=np.int64),
            local_query_indices=local_query,
            local_module_indices=local_module,
        )
        return scene_inputs, receivers

    def source_read_teacher_trace(
        self,
        model: nn.Module,
        case_key: Any,
        key: SamplingKey,
        max_query_rows: int,
    ) -> Mapping[str, torch.Tensor | None]:
        """Build a bounded, target-free trace over the native Wind role mixture."""
        initial_key = key.epoch == 500 and key.stage == "source_read_auxiliary_initial"
        refresh_key = 501 <= key.epoch <= 2500 and key.stage == "source_read_auxiliary_refresh"
        if (model is not self.model or not (initial_key or refresh_key)
                or key.dataset_id != self.sampling_dataset_id):
            raise ValueError(
                "Wind source-read teacher keys must be an e500 initial fit or a bounded "
                "501..2500 current-model TRAIN refresh bound to this core and dataset."
            )
        from honf_runtime.source_read_auxiliary import select_evenly_spaced_teacher_rows

        scene_inputs, receivers = self._source_read_target_free_receivers(case_key, key)
        scene = self.adapter.make_scene_batch((scene_inputs,))
        self._set_source_read_training_epoch(model, key.epoch)
        organization = model.prepare(scene)
        query = self.adapter.make_query_batch(scene, receivers)
        query = select_evenly_spaced_teacher_rows(query, max_query_rows)
        trace_method = getattr(model, "source_read_teacher_trace_v1", None)
        if not callable(trace_method):
            raise RuntimeError("Wind A2-D core is missing the bounded Round3 teacher trace API.")
        trace = trace_method(organization, query)
        if not isinstance(trace, Mapping):
            raise ValueError("Wind teacher trace must be a mapping.")
        return trace

    def source_read_physical_calibration_loss(
        self,
        model: nn.Module,
        case_key: Any,
        key: SamplingKey,
    ) -> torch.Tensor:
        """Evaluate the unchanged all-full native TRAIN objective for one row."""
        if model is not self.model or key.epoch != 500 or key.dataset_id != self.sampling_dataset_id:
            raise ValueError("Wind physical calibration keys must bind dense e500 TRAIN.")
        from dataclasses import replace

        counters = {
            name: getattr(self, name)
            for name in ("physical_training_samples", "physical_training_queries", "training_samples", "training_queries")
            if hasattr(self, name)
        }
        try:
            probe_key = replace(key, update_index=0, microbatch_index=1)
            batch = self.make_batch((int(case_key),), probe_key)
            scene = self.make_scene(batch.scene_inputs)
            predictions, auxiliary = self.predict_native(
                model, scene, batch.receivers, "A2-D", "joint", epoch=500,
            )
            terms = self.loss_terms(predictions, batch.targets, "joint", auxiliary)
            objective: torch.Tensor | None = None
            for name, term in terms.items():
                denominator = float(term.denominator.detach().item()) if torch.is_tensor(term.denominator) else float(term.denominator)
                weight = float(term.weight)
                if denominator < 0 or not math.isfinite(denominator) or not math.isfinite(weight):
                    raise ValueError(f"Wind physical calibration term {name!r} has invalid scaling.")
                if denominator == 0:
                    if float(term.numerator.detach()) != 0.0:
                        raise ValueError(f"Wind physical calibration term {name!r} has no valid elements.")
                    continue
                piece = term.numerator.reshape(()) * (weight / denominator)
                objective = piece if objective is None else objective + piece
            if objective is None or not objective.requires_grad or not torch.isfinite(objective):
                raise ValueError("Wind physical calibration did not produce a differentiable native loss.")
            return objective
        finally:
            for name, value in counters.items():
                setattr(self, name, value)

    def identity_payload(self) -> Mapping[str, Any]:
        result = dict(self._two_region_identity_payload(source_read=True))
        protocol = str(self.recipe["protocol"])
        full = protocol == "source_read_full1000"
        result["source_reader_variant"] = self.source_reader_variant
        result["dense_host_variant"] = str(self.recipe.get("dense_host_variant", "D0"))
        result["source_read_horizon"] = {
            "declared_total_epochs": self.total_epochs,
            "approved_stop_after": int(self.recipe["approved_stop_after"]),
            "full_data_stop_is_hard": full,
        }
        training = dict(result["training"])
        training["approved_stop_after"] = int(self.recipe["approved_stop_after"])
        result["training"] = training
        if not full:
            train_selection = self.manifest.get("train_layout_selection", {})
            result["development_cohort_coverage"] = {
                "train_rows": int(self.train_rows.size),
                "train_layouts": len(result["train_layout_indices"]),
                "dev_rows": int(self.validation_rows.size),
                "dev_layouts": len(result["validation_layout_indices"]),
                "directions_per_layout": 3,
                "uncovered_train_module_count_strata": list(
                    train_selection.get("uncovered_turbine_count_strata_by_limited_selection", ())
                ),
                "membership_source": self.manifest.get("subset_id"),
                "selected_from_targets": False,
            }
        return result

    def serialization_payload(self) -> Mapping[str, Any]:
        result = dict(WindContentHypergraphTaskV2.serialization_payload(self))
        result["source_read_scope"] = self.recipe["protocol"]
        result["source_reader_variant"] = self.source_reader_variant
        result["dense_host_variant"] = self.recipe.get("dense_host_variant", "D0")
        result["data_binding"] = {
            **dict(result["data_binding"]),
            "source_read_scope": self.recipe["protocol"],
            "source_reader_variant": self.source_reader_variant,
        }
        return result

    def preparation_summary(self) -> Mapping[str, Any]:
        result = dict(WindContentHypergraphTask.preparation_summary(self))
        result.update(
            {
                "source_read_scope": self.recipe["protocol"],
                "source_reader_variant": self.source_reader_variant,
                "initialization_seed": self.initialization_seed,
                "sampling_seed": self.sampling_seed,
                "role_query_counts": dict(self.role_query_counts),
                "validation_role_query_counts": dict(self.validation_role_query_counts),
                "validation_queries_per_case": self.VALIDATION_QUERY_COUNT,
                "optimizer_updates_per_epoch": math.ceil(len(self.train_rows) / self.effective_batch_size),
                "approved_stop_after": int(self.recipe["approved_stop_after"]),
                "wind_test_target_values_read": False,
            }
        )
        if self.recipe["protocol"] == "source_read_development_v1":
            train_selection = self.manifest.get("train_layout_selection", {})
            result["uncovered_train_module_count_strata"] = list(
                train_selection.get("uncovered_turbine_count_strata_by_limited_selection", ())
            )
        return result


def build_wind_content_task(
    recipe: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[ContentHypergraphCore, WindContentHypergraphTask]:
    """Build a fresh v1 Wind shared core/provider pair for exact replay."""
    return _build_wind_content_task(recipe, device, schema_version=1)


def build_wind_content_task_v2(
    recipe: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, WindContentHypergraphTaskV2]:
    """Build a fresh Round2 Wind core/provider pair from its explicit v2 recipe."""
    return _build_wind_content_task(recipe, device, schema_version=2)


def build_wind_formal_content_task_v1(
    recipe: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, WindFormalA2DTaskV1]:
    """Build the separately validated full-TRAIN A2-D formal provider."""
    model, provider = _build_wind_content_task(recipe, device, schema_version=3)
    if not isinstance(provider, WindFormalA2DTaskV1):
        raise RuntimeError("Formal Wind builder returned the wrong provider class.")
    return model, provider


def build_wind_source_read_task_v1(
    recipe: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, WindSourceReadTaskV1]:
    """Build the explicit Round3 segmented or hard-stop full1000 Wind scope."""
    model, provider = _build_wind_content_task(recipe, device, schema_version=4)
    if not isinstance(provider, WindSourceReadTaskV1):
        raise RuntimeError("Wind source-read builder returned the wrong provider class.")
    return model, provider


def _build_wind_content_task(
    recipe: Mapping[str, Any],
    device: torch.device | str,
    *,
    schema_version: int,
    role_scale_loader=None,
    transform_loader=None,
) -> tuple[nn.Module, WindContentHypergraphTask]:
    if schema_version == 1:
        value, architecture, microbatch = _validate_recipe(recipe)
    elif schema_version == 2:
        value, architecture, microbatch = _validate_recipe_v2(recipe)
    elif schema_version == 3:
        value, architecture, microbatch = _validate_formal_wind_a2d_recipe_v1(recipe)
    elif schema_version == 4:
        value, architecture, microbatch = _validate_source_read_wind_recipe_v1(recipe)
    else:
        raise ValueError("Wind content-hypergraph supports only explicit v1, v2, formal, or source-read dispatch.")
    initialization_seed = int(value["seed"])
    seed = int(value["sampling_seed"]) if schema_version in (3, 4) else initialization_seed
    protocol = value["protocol"]
    from honf_runtime.source_read_protocol import SOURCE_READ_DEVELOPMENT, SOURCE_READ_FULL

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
    if protocol in ("development", SOURCE_READ_DEVELOPMENT):
        manifest, train_rows, validation_rows = _read_fixed_rows(view, split, manifest_path)
        if schema_version == 4:
            uncovered = list(
                manifest.get("train_layout_selection", {}).get(
                    "uncovered_turbine_count_strata_by_limited_selection", ()
                )
            )
            if (
                len(train_rows) != 72
                or len(validation_rows) != 24
                or len(manifest.get("train_layout_indices", ())) != 24
                or len(manifest.get("validation_layout_indices", ())) != 8
                or uncovered != [18]
            ):
                raise ValueError(
                    "Wind source-read development requires the exact fixed TRAIN72/DEV24 manifest; "
                    "its omitted TRAIN M18 stratum must remain disclosed."
                )
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
    elif protocol in ("formal_d_full_train_v1", "full_train1000", SOURCE_READ_FULL):
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
        load_transforms = _fit_or_load_formal_transforms if transform_loader is None else transform_loader
        normalizer, profile, transform_binding = load_transforms(
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
    else:
        raise ValueError(f"Wind content-hypergraph has no data provider for protocol {protocol!r}.")

    # Five-role calibration belongs only to the native role objective. The
    # formal two-region source-read path standardizes with the bound velocity
    # normalizer and must neither require nor serialize the unused role cache.
    catalogue_cache = _shared_read_only_catalogue_cache(int(value["catalogue_cache_ram_bytes"]))
    if schema_version == 4:
        calibration = None
        cache_identity = None
        component_scales = None
    else:
        loader = _fit_or_load_role_scales if role_scale_loader is None else role_scale_loader
        calibration, cache_identity = loader(
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
        component_scales = {
            role: tuple(float(value) for value in calibration["component_role_scales_mps"][role])
            for role in ROLE_NAMES
        }
    normalizer_source["profile_sha256"] = _stable_json_sha256(profile.to_dict())
    adapter_type = WindContentHypergraphAdapter if schema_version == 1 else WindContentHypergraphAdapterV3
    adapter = adapter_type(normalizer, profile, device=device)
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
        "local_hidden_dim": 64 if schema_version == 1 else 128,
        "dropout": 0.0,
        "init_seed": initialization_seed,
    }
    if "query_edge_geometry_mode" in model_values:
        model_config["query_edge_geometry_mode"] = model_values["query_edge_geometry_mode"]
    reader_backend = model_values.get("reader_execution_backend", "legacy")
    source_value_conditioning = model_values.get("source_value_conditioning", "none")
    if reader_backend != "legacy":
        model_config["reader_execution_backend"] = reader_backend
    if source_value_conditioning != "none":
        model_config["source_value_conditioning"] = source_value_conditioning
    _apply_wind_source_read_core_options(
        model_config, model_values, schema_version=schema_version,
        source_reader_variant=value.get("source_reader_variant"),
    )
    expected_model = {
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 6,
        "local_hidden_dim": 64 if schema_version == 1 else 128,
    }
    if any(model_values.get(name, expected) != expected for name, expected in expected_model.items()):
        raise ValueError("Wind core model configuration diverged from its matched scientific contract.")
    devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        devices = [torch.cuda.current_device() if device.index is None else int(device.index)]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(initialization_seed)
        if devices:
            torch.cuda.manual_seed_all(initialization_seed)
        if schema_version == 1:
            model = ContentHypergraphCore(**model_config).to(device)
        else:
            from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2

            model_config["schema"] = "content_hypergraph_core_v2"
            model = ContentHypergraphCoreV2.from_config(model_config).to(device)

    provider_type = (
        WindContentHypergraphTask
        if schema_version == 1
        else WindFormalA2DTaskV1
        if schema_version == 3
        else WindSourceReadTaskV1
        if schema_version == 4
        else WindContentHypergraphTaskV2
    )
    provider = provider_type(
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
    "WIND_CONTENT_FAMILY_V2",
    "WIND_CONTENT_PROVIDER_SCHEMA_V2",
    "WIND_CONTENT_ROLE_CACHE_DEVELOPMENT_BYTES",
    "WIND_CONTENT_ROLE_CACHE_FULL_BYTES",
    "WIND_CONTENT_ROLE_CACHE_RAM_BYTES",
    "WIND_SOURCE_READ_ROLE_CACHE_RAM_BYTES",
    "WindContentHypergraphTask",
    "WindContentHypergraphTaskV2",
    "WindFormalA2DTaskV1",
    "WindSourceReadTaskV1",
    "WindContentPredictions",
    "WindContentTargets",
    "build_wind_formal_content_task_v1",
    "build_wind_source_read_task_v1",
    "build_wind_content_task",
    "build_wind_content_task_v2",
    "_validate_recipe_v2",
    "_validate_source_read_wind_recipe_v1",
]
