"""Thermal binding for the shared A-D/A-H6 content-hypergraph engine."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from honf_runtime.unified_training import (
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    TaskBatch,
    _sampling_dataset_id,
)

from ..content_hypergraph import (
    THERMAL_CONTENT_COORDINATE_FRAME,
    THERMAL_CONTENT_ENVIRONMENT_SHAPE,
    THERMAL_CONTENT_SURFACE_POINTS,
    ThermalContentHypergraphAdapter,
)
from ..data.datasets import H5Normalizer, fit_global_normalizer
from ..data.development_split import development_case_ids, read_case_catalog
from .joint_task import (
    _load_data_binding,
    _read_selected_cases,
    validate_full_train_followup_membership,
)
from .unified_task import (
    DEFAULT_ATLAS_DIRECTORY,
    FIXED25_FINGERPRINT,
    NEAR_RESPONSE_IDS,
    RESPONSE_SURFACE_STRIDE,
    TRAIN_RESPONSE_IDS,
    ThermalResponseTargets,
    _move_tree,
    _read_response_families,
    _sample_primary,
    _structure_only,
)

THERMAL_CONTENT_PROVIDER_SCHEMA = "thermal_content_hypergraph_provider_v1"
THERMAL_CONTENT_PROVIDER_V2_SCHEMA = "thermal_content_hypergraph_provider_v2"
THERMAL_CONTENT_RECIPE_V1_SCHEMA = "content_hypergraph_recipe_v1"
THERMAL_CONTENT_RECIPE_V2_SCHEMA = "content_hypergraph_recipe_v2"
THERMAL_CONTENT_SAMPLING_VERSION = SamplingKey.CASE_EPOCH_VERSION
THERMAL_CONTENT_NATIVE_QUERY_COUNT = 8192
THERMAL_CONTENT_THETA_COUNT = 64
THERMAL_CONTENT_REPRESENTATIVE_IDS = tuple(NEAR_RESPONSE_IDS)
THERMAL_CONTENT_FULL_TRAIN_PROTOCOL = "thermal_content_hypergraph_original600_full1000_v1"
THERMAL_CONTENT_FULL_TRAIN_DATASET = (
    "thermal_content_hypergraph_original600_train_canonical89_validation_v1"
)
THERMAL_CONTENT_LOSS_WEIGHTS = {
    "flow/u": 1.0 / 8.0,
    "flow/v": 1.0 / 8.0,
    "flow/p": 1.0 / 8.0,
    "flow/omega": 1.0 / 8.0,
    "temperature/fluid": 1.0 / 6.0,
    "temperature/surface": 1.0 / 6.0,
    "temperature/material": 1.0 / 6.0,
    "q_proxy": 0.05,
    "response": 0.1,
}


def _endpoint_material_maximum_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> list[dict[str, float | int]]:
    """Compare each active module's predicted and reference endpoint maxima."""
    if prediction.shape != target.shape or target.shape != valid.shape or prediction.ndim != 2:
        raise ValueError("Thermal endpoint material arrays must align as [module, receiver].")
    rows = []
    for module in range(prediction.shape[0]):
        module_valid = valid[module] & np.isfinite(prediction[module]) & np.isfinite(target[module])
        if not np.any(module_valid):
            continue
        predicted_max = float(np.max(prediction[module, module_valid]))
        target_max = float(np.max(target[module, module_valid]))
        rows.append({
            "module_index": int(module),
            "predicted_max": predicted_max,
            "target_max": target_max,
            "error": predicted_max - target_max,
        })
    return rows


def _summarize_endpoint_material_maxima(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors = np.asarray([float(row["error"]) for row in rows], dtype=np.float64)
    absolute = np.abs(errors)
    if errors.size == 0:
        return {
            "module_count": 0,
            "signed_error_mean": None,
            "absolute_error_mean": None,
            "absolute_error_p90": None,
            "absolute_error_p95": None,
            "absolute_error_worst": None,
        }
    return {
        "module_count": int(errors.size),
        "signed_error_mean": float(np.mean(errors)),
        "absolute_error_mean": float(np.mean(absolute)),
        "absolute_error_p90": float(np.percentile(absolute, 90)),
        "absolute_error_p95": float(np.percentile(absolute, 95)),
        "absolute_error_worst": float(np.max(absolute)),
    }


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _ids_sha256(case_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for case_id in case_ids:
        digest.update(str(case_id).encode("utf-8") + b"\0")
    return digest.hexdigest()


def _batch_tensor(value: Any, device: torch.device, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    return tensor


@dataclass(frozen=True)
class ThermalContentSceneInputs:
    """Geometry and prescribed context only; heat and labels stay separate."""

    structure: Mapping[str, Any]
    response_structure: Mapping[str, Any] | None = None
    case_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ThermalContentReceivers:
    fluid_xy: torch.Tensor
    local_query_points: torch.Tensor
    heat: torch.Tensor
    response_fluid_xy: torch.Tensor | None = None
    response_local_query_points: torch.Tensor | None = None
    response_heat: torch.Tensor | None = None
    ntheta: int = THERMAL_CONTENT_SURFACE_POINTS
    response_ntheta: int = THERMAL_CONTENT_SURFACE_POINTS


@dataclass(frozen=True)
class ThermalContentTargets:
    case_ids: tuple[str, ...]
    case_indices: tuple[int, ...]
    epoch: int
    field_targets: torch.Tensor
    point_weights: torch.Tensor
    interface_target: torch.Tensor
    interface_valid_mask: torch.Tensor
    material_targets: torch.Tensor
    response: ThermalResponseTargets | None = None


@dataclass
class ThermalContentPredictions:
    native_prepared: Any
    native_main: Mapping[str, torch.Tensor]
    response_prepared: Any | None = None
    response_main: Mapping[str, torch.Tensor] | None = None
    work: Mapping[str, int] | None = None


class ThermalContentHypergraphTask:
    """Native Thermal supervision and data identity for the common engine."""

    def __init__(
        self,
        *,
        model: ThermalContentHypergraphAdapter,
        training_cases: Sequence[Mapping[str, Any]],
        validation_cases: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
        base_manifest_sha256: str,
        data_path: Path,
        source_binding: Mapping[str, Any],
        train_families: Sequence[Mapping[str, Any]],
        response_scales: Mapping[str, float],
        recipe: Mapping[str, Any],
        seed: int,
        primary_queries: int,
        microbatch_size: int,
        effective_batch_size: int,
        total_epochs: int,
        device: torch.device | str,
        protocol: str,
        case_split_by_id: Mapping[str, str],
        catalog_case_ids: Sequence[str],
        manifest_path: Path | None,
        atlas_directory: Path,
    ) -> None:
        self.model = model
        self.adapter = model
        self.training_cases = tuple(training_cases)
        self.validation_cases = tuple(validation_cases)
        self.manifest = copy.deepcopy(dict(manifest))
        self.base_manifest_sha256 = str(base_manifest_sha256)
        self.data_path = Path(data_path).resolve()
        self.source_binding = copy.deepcopy(dict(source_binding))
        self.train_families = tuple(train_families)
        self.response_scales = {str(key): float(value) for key, value in response_scales.items()}
        self.recipe = json.loads(json.dumps(_plain(dict(recipe)), sort_keys=True, allow_nan=False))
        self.seed = int(seed)
        self.primary_queries = int(primary_queries)
        self.microbatch_size = int(microbatch_size)
        self.effective_batch_size = int(effective_batch_size)
        self.total_epochs = int(total_epochs)
        self.device = torch.device(device)
        self.protocol = str(protocol)
        self.case_split_by_id = {str(key): str(value).lower() for key, value in case_split_by_id.items()}
        self.catalog_case_ids = tuple(map(str, catalog_case_ids))
        self.manifest_path = manifest_path
        self.atlas_directory = Path(atlas_directory).resolve()
        self.training_case_ids = tuple(str(case["case_id"]) for case in self.training_cases)
        self.validation_case_ids = tuple(str(case["case_id"]) for case in self.validation_cases)
        self.validation_scope = (
            "fixed25_v1_DEV22_exposed" if self.protocol in ("development", "inherited_attention_development_v1")
            else "canonical89_exposed_original_test_excluding_0273"
        )
        self.sampling_version = THERMAL_CONTENT_SAMPLING_VERSION
        self.sampling_dataset_id = _sampling_dataset_id({
            "dataset": "ThermalChannel",
            "subset_manifest_sha256": self.manifest["manifest_sha256"],
        })
        self.training_samples = 0
        self.validation_samples = 0
        self._validation_batches_cache: tuple[TaskBatch, ...] | None = None
        self._train_index_by_id = {case_id: index for index, case_id in enumerate(self.training_case_ids)}
        self._validation_index_by_id = {case_id: index for index, case_id in enumerate(self.validation_case_ids)}
        self._train_family_by_id = {str(item["family_id"]): item for item in self.train_families}
        if self.seed != 0 or self.primary_queries != 1024 or self.effective_batch_size != 48:
            raise ValueError("Thermal content-hypergraph is sealed to seed 0, Q=1024, effective batch 48.")
        if not 1 <= self.microbatch_size <= self.effective_batch_size:
            raise ValueError("Thermal microbatch size must be in [1,48].")
        expected = (
            (150, 22, 2500)
            if self.protocol in ("development", "inherited_attention_development_v1")
            else (600, 89, 1000)
            if self.protocol == "inherited_attention_full_train1000_v1"
            else (600, 89, 5000)
        )
        if (len(self.training_cases), len(self.validation_cases), self.total_epochs) != expected:
            raise ValueError("Thermal provider membership/horizon differs from the declared protocol.")
        if self.protocol not in (
            "development", "full_train1000", "formal_d_full_train_v1",
            "inherited_attention_development_v1", "inherited_attention_full_train1000_v1",
        ):
            raise ValueError("Unsupported Thermal content-hypergraph protocol.")
        if tuple(str(item["family_id"]) for item in self.train_families) != TRAIN_RESPONSE_IDS:
            raise ValueError("Response auxiliary must use the ordered original-TRAIN family inventory.")
        if set(self.response_scales) != {"fluid", "surface", "material"} or any(
            not math.isfinite(value) or value <= 0 for value in self.response_scales.values()
        ):
            raise ValueError("Thermal TRAIN response scales must be positive finite scalars.")
        if set(self.training_case_ids) & set(self.validation_case_ids):
            raise ValueError("Thermal training and exposed validation memberships overlap.")
        if any(len(case["query_xy"]) < self.primary_queries for case in self.training_cases):
            raise ValueError("A fixed Thermal training case has fewer than 1024 primary receivers.")
        if any(len(case["module_internal_query_points"]) < 32 for case in self.training_cases):
            raise ValueError("A fixed Thermal training case has fewer than 32 stored material receivers.")

    def identity_payload(self) -> Mapping[str, Any]:
        adapter_state = self.adapter.state_payload()
        model_config = self.model.get_config()
        source_stat = self.data_path.stat()
        response_sources = [
            {
                "family_id": str(item["family_id"]),
                "path": str(item["source"]),
                "sha256": str(item["source_sha256"]),
            }
            for item in self.train_families
        ]
        return {
            "schema": THERMAL_CONTENT_PROVIDER_SCHEMA,
            "dataset": "ThermalChannel",
            "task": "ThermalChannel",
            "architecture": self.model.architecture,
            "protocol": self.protocol,
            "dataset_profile": "fixed25_v1" if self.protocol == "development" else THERMAL_CONTENT_FULL_TRAIN_DATASET,
            "subset_manifest_sha256": str(self.manifest["manifest_sha256"]),
            "base_fixed25_manifest_sha256": self.base_manifest_sha256,
            "training_case_ids": list(self.training_case_ids),
            "training_case_ids_sha256": _ids_sha256(self.training_case_ids),
            "validation_case_ids": list(self.validation_case_ids),
            "validation_case_ids_sha256": _ids_sha256(self.validation_case_ids),
            "validation_scope": self.validation_scope,
            "response_family_sources": response_sources,
            "model_config": model_config,
            "adapter_state": adapter_state,
            "normalization_stats_sha256": _stable_sha256(adapter_state["normalization_stats"]),
            "source_binding": {
                **self.source_binding,
                "dataset_path": str(self.data_path),
                "dataset_size_bytes": int(source_stat.st_size),
                "dataset_mtime_ns": int(source_stat.st_mtime_ns),
                "base_manifest_path": None if self.manifest_path is None else str(self.manifest_path),
                "base_manifest_sha256": self.base_manifest_sha256,
                "normalization_fit_scope": (
                    "selected original TRAIN150 only" if self.protocol == "development"
                    else "exact original TRAIN600 only"
                ),
            },
            "environment_representation": {
                "token_shape_ny_nx": list(THERMAL_CONTENT_ENVIRONMENT_SHAPE),
                "count": 192,
                "geometry_only": True,
                "heating_in_scene": False,
                "temperature_or_flow_target_features": False,
            },
            "coordinate_frame": {
                "name": THERMAL_CONTENT_COORDINATE_FRAME,
                "units": "packed_dataset_native_xy",
                "native_grid_ny_nx": [self.adapter.ny, self.adapter.nx],
            },
            "objective": {
                "name": "equal_case_native_flow_temperature_and_flux_v1",
                "flow_weights": {role: THERMAL_CONTENT_LOSS_WEIGHTS[f"flow/{role}"]
                                 for role in ("u", "v", "p", "omega")},
                "temperature_weights": {role: THERMAL_CONTENT_LOSS_WEIGHTS[f"temperature/{role}"]
                                        for role in ("fluid", "surface", "material")},
                "q_proxy_weight": THERMAL_CONTENT_LOSS_WEIGHTS["q_proxy"],
                "response_weight": THERMAL_CONTENT_LOSS_WEIGHTS["response"],
                "surface_material_aggregation": "within-case, then equal mean over active modules",
                "response_family_ids": list(TRAIN_RESPONSE_IDS),
                "response_scales": self.response_scales,
                "response_cadence": "one original-TRAIN family on microbatch zero of the final optimizer macrobatch each epoch",
                "operator_residual_enabled": False,
                "exposed_validation_targets_read": True,
                "exposed_validation_target_scope": self.validation_scope,
                "unexposed_test_targets_read": False,
            },
            "training": {
                "initialization_seed": self.seed,
                "sampling_version": self.sampling_version,
                "sampling_dataset_id": self.sampling_dataset_id,
                "effective_batch_size": self.effective_batch_size,
                "microbatch_size": self.microbatch_size,
                "primary_queries_per_case": self.primary_queries,
                "surface_points_per_module": 16,
                "surface_stride": RESPONSE_SURFACE_STRIDE,
                "material_points_per_module": 32,
                "total_epochs": self.total_epochs,
                "optimizer": "AdamW",
                "optimizer_schedule": self._schedule_payload(),
                "weight_decay": 1.0e-4,
                "bias_and_norm_weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "gradient_clip": 1.0,
                "all_parameters_train_from_start": True,
            },
            "native_evaluation": {
                "fluid_points": THERMAL_CONTENT_NATIVE_QUERY_COUNT,
                "surface_theta_points_per_module": THERMAL_CONTENT_THETA_COUNT,
                "material_points": "all stored native points",
                "representative_case_ids": list(THERMAL_CONTENT_REPRESENTATIVE_IDS),
            },
            "historical_predictor_instantiated": False,
            "frozen_stage_a_or_dsep_instantiated": False,
            "fresh_matched_comparison": False,
            "solver_attempts": 0,
            "recipe": self.recipe,
        }

    def preparation_summary(self) -> Mapping[str, Any]:
        return {
            "status": "prepared_only",
            "protocol": self.protocol,
            "dataset_profile": "fixed25_v1" if self.protocol == "development" else THERMAL_CONTENT_FULL_TRAIN_DATASET,
            "manifest_sha256": self.manifest["manifest_sha256"],
            "training_case_count": len(self.training_cases),
            "validation_case_count": len(self.validation_cases),
            "training_case_ids_sha256": _ids_sha256(self.training_case_ids),
            "validation_case_ids_sha256": _ids_sha256(self.validation_case_ids),
            "normalization_scope": "TRAIN only",
            "response_family_ids": list(TRAIN_RESPONSE_IDS),
            "response_scales": dict(self.response_scales),
            "primary_queries": self.primary_queries,
            "effective_batch_size": self.effective_batch_size,
            "microbatch_size": self.microbatch_size,
            "environment_tokens": 192,
            "optimizer_updates": 0,
            "solver_attempts": 0,
        }

    def epoch_cases(self, epoch: int, seed: int) -> Sequence[str]:
        del epoch
        if int(seed) != self.seed:
            raise ValueError("Thermal epoch-order seed differs from the fixed recipe.")
        return self.training_case_ids

    def _response_sample(self, family: Mapping[str, Any], key: SamplingKey) -> dict[str, Any]:
        family_id = str(family["family_id"])
        valid_fluid = np.flatnonzero(np.asarray(family["fluid_valid"], dtype=bool))
        if valid_fluid.size < self.primary_queries or len(family["material_local"]) < 32:
            raise ValueError(f"Original-TRAIN response family {family_id} has insufficient fixed coverage.")
        fluid_rows = valid_fluid[key.native_indices(
            valid_fluid.size, self.primary_queries, family_id, "response_fluid", "response_addendum",
        )]
        material_rows = key.native_indices(
            len(family["material_local"]), 32, family_id, "response_material", "response_addendum",
        )
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)
        structure = {name: tensor(value)[None] for name, value in family["structure"].items()}
        source_ids = family["structure"].get("module_source_ids")
        if source_ids is not None:
            structure["module_source_ids"] = torch.as_tensor(source_ids, dtype=torch.long)[None]
        surface_target = family["deltas"]["interface"][:, 0].reshape(family["module_count"], -1)
        surface_mask = np.asarray(family["interface_valid"][:, 0], dtype=bool).reshape(
            family["module_count"], -1
        )
        material_target = family["deltas"]["solid_temperature"].reshape(family["module_count"], -1)
        return {
            "family_id": family_id,
            "structure": _move_tree(structure, self.device),
            "fluid_xy": tensor(family["fluid_xy"][fluid_rows])[None].to(self.device),
            "local": tensor(family["material_local"][material_rows])[None].to(self.device),
            "heat": tensor(family["heat_increment"])[None].to(self.device),
            "fluid_target": tensor(family["deltas"]["fluid_fields"][fluid_rows, 4])[None].to(self.device),
            "fluid_mask": torch.ones((1, len(fluid_rows)), device=self.device, dtype=torch.float32),
            "surface_target": tensor(surface_target[:, ::RESPONSE_SURFACE_STRIDE])[None].to(self.device),
            "surface_mask": tensor(surface_mask[:, ::RESPONSE_SURFACE_STRIDE])[None].to(self.device),
            "material_target": tensor(material_target[:, material_rows])[None].to(self.device),
            "material_mask": tensor(np.asarray(family["material_valid"], dtype=bool)[:, material_rows])[None].to(self.device),
        }

    def make_batch(self, case_keys: Sequence[Any], key: SamplingKey) -> TaskBatch:
        if key.sampling_version != self.sampling_version or key.dataset_id != self.sampling_dataset_id:
            raise ValueError("Thermal query draws require the exact case_epoch_v1 TRAIN membership key.")
        sampling_epoch_limit = int(getattr(self, "_round6_sampling_epoch_limit", self.total_epochs))
        if key.epoch < 1 or key.epoch > sampling_epoch_limit or not case_keys:
            raise ValueError("Thermal TRAIN batch has an invalid epoch or no cases.")
        if len(case_keys) > self.microbatch_size:
            raise ValueError("Thermal engine microbatch exceeds its declared case capacity.")
        indices = tuple(self._train_index_by_id[str(case_id)] for case_id in case_keys)
        if len(set(indices)) != len(indices):
            raise ValueError("A Thermal microbatch cannot repeat a case.")
        budget = {
            "fluid_queries": self.primary_queries,
            "material_queries_per_module": 32,
            "surface_stride": RESPONSE_SURFACE_STRIDE,
        }
        sampled, sample_hash = _sample_primary(
            self.training_cases,
            indices,
            epoch=key.epoch,
            key=key,
            training=True,
            budget=budget,
            device=self.device,
        )
        structure = _structure_only(sampled["structure"])
        case_ids = tuple(str(self.training_cases[index]["case_id"]) for index in indices)
        final_update_index = math.ceil(len(self.training_cases) / self.effective_batch_size) - 1
        response = None
        response_structure = None
        response_sample = None
        if key.update_index == final_update_index and key.microbatch_index == 0:
            family_id = TRAIN_RESPONSE_IDS[(key.epoch + key.update_index) % len(TRAIN_RESPONSE_IDS)]
            response_sample = self._response_sample(self._train_family_by_id[family_id], key)
            response_structure = response_sample["structure"]
            response = ThermalResponseTargets(
                family_id=family_id,
                fluid=response_sample["fluid_target"],
                surface=response_sample["surface_target"],
                surface_mask=response_sample["surface_mask"],
                material=response_sample["material_target"],
                material_mask=response_sample["material_mask"],
            )
        scene_inputs = ThermalContentSceneInputs(
            structure=structure,
            response_structure=response_structure,
            case_ids=case_ids,
        )
        receivers = ThermalContentReceivers(
            fluid_xy=sampled["query_xy"],
            local_query_points=sampled["module_internal_query_points"],
            heat=sampled["structure"]["heat_powers"],
            response_fluid_xy=None if response_sample is None else response_sample["fluid_xy"],
            response_local_query_points=None if response_sample is None else response_sample["local"],
            response_heat=None if response_sample is None else response_sample["heat"],
            ntheta=THERMAL_CONTENT_SURFACE_POINTS,
            response_ntheta=THERMAL_CONTENT_SURFACE_POINTS,
        )
        targets = ThermalContentTargets(
            case_ids=case_ids,
            case_indices=indices,
            epoch=key.epoch,
            field_targets=sampled["field_targets"],
            point_weights=sampled["point_weights"],
            interface_target=sampled["interface_target"],
            interface_valid_mask=sampled["interface_condition_valid_mask"],
            material_targets=sampled["module_internal_temperature_points"],
            response=response,
        )
        self.training_samples += len(indices)
        return TaskBatch(
            scene_inputs=scene_inputs,
            receivers=receivers,
            targets=targets,
            auxiliary={
                "query_sampling_sha256": sample_hash,
                "response_family_id": None if response is None else response.family_id,
            },
            case_keys=case_ids,
        )

    def make_scene(self, scene_inputs: ThermalContentSceneInputs) -> ThermalContentSceneInputs:
        if not isinstance(scene_inputs, ThermalContentSceneInputs):
            raise TypeError("Thermal content-hypergraph engine requires target-free scene inputs.")
        return scene_inputs

    def predict_native(
        self,
        model: nn.Module,
        scene: ThermalContentSceneInputs,
        receivers: ThermalContentReceivers,
        execution_mode: str,
        phase: str,
        epoch: int = 0,
        temperature: float | None = None,
    ) -> tuple[ThermalContentPredictions, Mapping[str, Any]]:
        del epoch, temperature
        if model is not self.model or execution_mode not in ("A-D", "A-H6"):
            raise ValueError("Thermal prediction requires the bound shared A-D/A-H6 model.")
        if model.architecture != execution_mode or phase not in ("joint", "validation", "development"):
            raise ValueError("Thermal prediction phase or architecture differs from the joint recipe.")
        prepared = model.prepare_native(
            scene.structure,
            receivers.fluid_xy,
            local_query_points=receivers.local_query_points,
            ntheta=receivers.ntheta,
            scene_key=scene.case_ids,
        )
        main = model.apply_native(prepared, receivers.heat)
        response_prepared = response_main = None
        if receivers.response_fluid_xy is not None:
            if (scene.response_structure is None or receivers.response_local_query_points is None
                    or receivers.response_heat is None):
                raise ValueError("Thermal TRAIN response addendum is missing geometry or physical heat controls.")
            response_prepared = model.prepare_native(
                scene.response_structure,
                receivers.response_fluid_xy,
                local_query_points=receivers.response_local_query_points,
                ntheta=receivers.response_ntheta,
                scene_key=("original_train_response",),
            )
            response_main = model.apply_increment(response_prepared, receivers.response_heat)
        work = {
            "native_fluid_queries": int(receivers.fluid_xy.shape[0] * receivers.fluid_xy.shape[1]),
            "native_union_rows": int(prepared.union.unique_counts.sum().item()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
            "response_native_union_rows": 0 if response_prepared is None else int(
                response_prepared.union.unique_counts.sum().item()
            ),
        }
        predictions = ThermalContentPredictions(prepared, main, response_prepared, response_main, work)
        return predictions, {"work": work}

    def loss_denominators(
        self, batches: Sequence[TaskBatch], phase: str = "joint", arm: str = "A-H6",
    ) -> Mapping[str, float]:
        del arm
        if phase != "joint":
            raise ValueError("Thermal content-hypergraph uses the shared joint phase.")
        totals = {name: 0.0 for name in THERMAL_CONTENT_LOSS_WEIGHTS if name != "response"}
        for batch in batches:
            count = len(batch.targets.case_ids)
            if count < 1:
                raise ValueError("Thermal content-hypergraph cannot plan an empty microbatch.")
            for name in totals:
                totals[name] += float(count)
            if batch.targets.response is not None:
                totals["response"] = totals.get("response", 0.0) + 1.0
        return totals

    def _role_losses(self, predictions: ThermalContentPredictions, targets: ThermalContentTargets) -> dict[str, torch.Tensor]:
        output = predictions.native_main
        prepared = predictions.native_prepared
        pred = output["pred_field"]
        target = targets.field_targets.to(pred)
        point_weight = targets.point_weights.to(pred) * prepared.stencils["fluid"].valid.to(pred)
        field_std = self.model.field_std.to(pred)
        result: dict[str, torch.Tensor] = {}
        for channel, role in enumerate(("u", "v", "p", "omega")):
            valid = point_weight * torch.isfinite(target[..., channel]).to(pred.dtype)
            error = (pred[..., channel] - torch.nan_to_num(target[..., channel])) / field_std[channel]
            result[f"flow/{role}"] = (error.square() * valid).sum(-1) / valid.sum(-1).clamp_min(1e-12)

        temp_valid = point_weight * torch.isfinite(target[..., 4]).to(pred.dtype)
        temp_error = (pred[..., 4] - torch.nan_to_num(target[..., 4])) / field_std[4]
        result["temperature/fluid"] = (temp_error.square() * temp_valid).sum(-1) / temp_valid.sum(-1).clamp_min(1e-12)

        present = prepared.source_present.to(pred.dtype)
        surface_shape = output["pred_interface"].shape[:-1]
        surface_valid = prepared.stencils["surface"].valid.reshape(surface_shape)
        outside_valid = prepared.stencils["outside"].valid.reshape(surface_shape)
        interface_target = targets.interface_target.to(pred)
        interface_observed = targets.interface_valid_mask.to(pred) > 0.5
        surface_mask = (present[..., None] > 0) & surface_valid & interface_observed & torch.isfinite(interface_target[..., 0])
        surface_scale = self.model.interface_target_std[0].to(pred).clamp_min(1e-12)
        surface_error = (output["pred_interface"][..., 0] - torch.nan_to_num(interface_target[..., 0])) / surface_scale
        surface_count = surface_mask.sum(-1)
        surface_per_module = (surface_error.square() * surface_mask).sum(-1) / surface_count.clamp_min(1)
        surface_module_live = (present > 0) & (surface_count > 0)
        result["temperature/surface"] = (surface_per_module * surface_module_live).sum(-1) / surface_module_live.sum(-1).clamp_min(1)

        material_target = targets.material_targets.to(pred)
        material_valid = (
            (present[..., None] > 0)
            & prepared.stencils["material"].valid.reshape(material_target.shape)
            & torch.isfinite(material_target)
        )
        material_scale = self.model.internal_temperature_std[0].to(pred).clamp_min(1e-12)
        material_error = (
            output["pred_internal_temperature"][..., 0] - torch.nan_to_num(material_target)
        ) / material_scale
        material_count = material_valid.sum(-1)
        material_per_module = (material_error.square() * material_valid).sum(-1) / material_count.clamp_min(1)
        material_module_live = (present > 0) & (material_count > 0)
        result["temperature/material"] = (material_per_module * material_module_live).sum(-1) / material_module_live.sum(-1).clamp_min(1)

        q_target = interface_target[..., 1]
        q_mask = (
            (present[..., None] > 0)
            & surface_valid
            & outside_valid
            & interface_observed
            & torch.isfinite(q_target)
        )
        q_scale = self.model.interface_target_std[1].to(pred).clamp_min(1e-12)
        q_error = (output["pred_interface"][..., 1] - torch.nan_to_num(q_target)) / q_scale
        q_count = q_mask.sum(-1)
        q_per_module = (q_error.square() * q_mask).sum(-1) / q_count.clamp_min(1)
        q_module_live = (present > 0) & (q_count > 0)
        result["q_proxy"] = (q_per_module * q_module_live).sum(-1) / q_module_live.sum(-1).clamp_min(1)
        return result

    def _response_loss(self, predictions: ThermalContentPredictions, targets: ThermalResponseTargets) -> torch.Tensor:
        if predictions.response_main is None or predictions.response_prepared is None:
            raise RuntimeError("Thermal response target was supplied without its affine native prediction.")
        output = predictions.response_main
        prepared = predictions.response_prepared
        device = output["fluid_temperature"].device
        fluid_target = targets.fluid.to(output["fluid_temperature"])
        fluid_mask = prepared.stencils["fluid"].valid.to(fluid_target.dtype) * torch.isfinite(fluid_target)
        fluid_scale = self.response_scales["fluid"]
        fluid_error = (output["fluid_temperature"][..., 0] - torch.nan_to_num(fluid_target)) / fluid_scale
        fluid_loss = (fluid_error.square() * fluid_mask).sum() / fluid_mask.sum().clamp_min(1)

        surface_target = targets.surface.to(output["pred_interface"])
        surface_valid = prepared.stencils["surface"].valid.reshape(surface_target.shape)
        surface_mask = (
            (prepared.source_present[..., None] > 0)
            & surface_valid
            & (targets.surface_mask.to(device=device) > 0.5)
            & torch.isfinite(surface_target)
        )
        surface_count = surface_mask.sum(-1)
        surface_error = (output["pred_interface"][..., 0] - torch.nan_to_num(surface_target)) / self.response_scales["surface"]
        surface_per_module = (surface_error.square() * surface_mask).sum(-1) / surface_count.clamp_min(1)
        surface_module_live = (prepared.source_present > 0) & (surface_count > 0)
        surface_loss = (surface_per_module * surface_module_live).sum() / surface_module_live.sum().clamp_min(1)

        material_target = targets.material.to(output["pred_internal_temperature"])
        material_valid = prepared.stencils["material"].valid.reshape(material_target.shape)
        material_mask = (
            (prepared.source_present[..., None] > 0)
            & material_valid
            & (targets.material_mask.to(device=device) > 0.5)
            & torch.isfinite(material_target)
        )
        material_count = material_mask.sum(-1)
        material_error = (
            output["pred_internal_temperature"][..., 0] - torch.nan_to_num(material_target)
        ) / self.response_scales["material"]
        material_per_module = (material_error.square() * material_mask).sum(-1) / material_count.clamp_min(1)
        material_module_live = (prepared.source_present > 0) & (material_count > 0)
        material_loss = (material_per_module * material_module_live).sum() / material_module_live.sum().clamp_min(1)
        return (fluid_loss + surface_loss + material_loss) / 3.0

    def loss_terms(
        self,
        predictions: ThermalContentPredictions,
        targets: ThermalContentTargets,
        phase: str,
        auxiliary_state: Any,
    ) -> Mapping[str, LossTerm]:
        if phase != "joint":
            raise ValueError("Thermal content-hypergraph loss is defined for the joint phase only.")
        per_case = self._role_losses(predictions, targets)
        result: dict[str, LossTerm] = {
            name: LossTerm(values.sum(), len(targets.case_ids), THERMAL_CONTENT_LOSS_WEIGHTS[name])
            for name, values in per_case.items()
        }
        result["q_proxy"] = LossTerm(
            per_case["q_proxy"].sum(), len(targets.case_ids), THERMAL_CONTENT_LOSS_WEIGHTS["q_proxy"]
        )
        if targets.response is not None:
            response = self._response_loss(predictions, targets.response)
            result["response"] = LossTerm(response, 1.0, THERMAL_CONTENT_LOSS_WEIGHTS["response"])
        if predictions.work is not None and auxiliary_state is not None:
            auxiliary_state["work"] = dict(predictions.work)
        return result

    def work_counts(
        self,
        batch: TaskBatch,
        predictions: ThermalContentPredictions,
        auxiliary_state: Any,
    ) -> Mapping[str, int | float]:
        """Expose already measured native work without another tensor reduction."""
        del batch
        if predictions.work is not None:
            return {str(name): value for name, value in predictions.work.items()}
        if isinstance(auxiliary_state, Mapping):
            work = auxiliary_state.get("work", {})
            if isinstance(work, Mapping):
                return {str(name): value for name, value in work.items()}
        return {}

    def validation_loss_terms(
        self,
        predictions: ThermalContentPredictions,
        targets: ThermalContentTargets,
        auxiliary_state: Any,
        *,
        batch: TaskBatch,
        arm: str,
    ) -> Mapping[str, LossTerm]:
        del auxiliary_state, batch, arm
        return self.loss_terms(predictions, targets, "joint", {})

    def validation_batches(self) -> Iterable[TaskBatch]:
        if self._validation_batches_cache is not None:
            return iter(self._validation_batches_cache)
        batches = []
        for batch_index, start in enumerate(range(0, len(self.validation_cases), self.microbatch_size)):
            case_indices = range(start, min(start + self.microbatch_size, len(self.validation_cases)))
            sampled, sample_hash = _sample_primary(
                self.validation_cases,
                case_indices,
                epoch=0,
                key=None,
                training=False,
                budget={
                    "fluid_queries": self.primary_queries,
                    "material_queries_per_module": 32,
                    "surface_stride": RESPONSE_SURFACE_STRIDE,
                },
                device=self.device,
            )
            indices = tuple(case_indices)
            case_ids = tuple(str(self.validation_cases[index]["case_id"]) for index in indices)
            batch = TaskBatch(
                scene_inputs=ThermalContentSceneInputs(
                    _structure_only(sampled["structure"]), case_ids=case_ids,
                ),
                receivers=ThermalContentReceivers(
                    sampled["query_xy"], sampled["module_internal_query_points"],
                    sampled["structure"]["heat_powers"],
                    ntheta=THERMAL_CONTENT_SURFACE_POINTS,
                    response_ntheta=THERMAL_CONTENT_SURFACE_POINTS,
                ),
                targets=ThermalContentTargets(
                    case_ids=case_ids,
                    case_indices=indices,
                    epoch=0,
                    field_targets=sampled["field_targets"],
                    point_weights=sampled["point_weights"],
                    interface_target=sampled["interface_target"],
                    interface_valid_mask=sampled["interface_condition_valid_mask"],
                    material_targets=sampled["module_internal_temperature_points"],
                ),
                auxiliary={"query_sampling_sha256": sample_hash, "validation_batch_index": batch_index},
                case_keys=case_ids,
            )
            batches.append(batch)
            self.validation_samples += len(case_ids)
        self._validation_batches_cache = tuple(batches)
        return iter(self._validation_batches_cache)

    def validation_metrics(
        self,
        predictions: ThermalContentPredictions,
        targets: ThermalContentTargets,
        auxiliary_state: Any,
    ) -> Mapping[str, Any]:
        del auxiliary_state
        output = predictions.native_main
        prepared = predictions.native_prepared
        predicted = output["pred_field"].detach().cpu().double().numpy()
        actual = targets.field_targets.detach().cpu().double().numpy()
        weights = targets.point_weights.detach().cpu().double().numpy()
        fluid_valid = prepared.stencils["fluid"].valid.detach().cpu().numpy()
        interface_target = targets.interface_target.detach().cpu().double().numpy()
        interface_valid_input = targets.interface_valid_mask.detach().cpu().numpy() > 0.5
        material_target = targets.material_targets.detach().cpu().double().numpy()
        present = prepared.source_present.detach().cpu().numpy() > 0.5
        surface_valid = prepared.stencils["surface"].valid.reshape(output["pred_interface"].shape[:-1]).detach().cpu().numpy()
        outside_valid = prepared.stencils["outside"].valid.reshape(output["pred_interface"].shape[:-1]).detach().cpu().numpy()
        material_valid_geometry = prepared.stencils["material"].valid.reshape(targets.material_targets.shape).detach().cpu().numpy()
        pred_interface = output["pred_interface"].detach().cpu().double().numpy()
        pred_material = output["pred_internal_temperature"][..., 0].detach().cpu().double().numpy()
        std = np.asarray(self.adapter.normalization_stats["field_std_by_channel"], dtype=np.float64)
        interface_std = np.asarray(self.adapter.normalization_stats["interface_targets_std"], dtype=np.float64)
        material_std = float(np.asarray(self.adapter.normalization_stats["internal_temperature_std"]).reshape(-1)[0])
        rows = []
        for index, case_id in enumerate(targets.case_ids):
            query_weight = weights[index] * fluid_valid[index]
            role_metrics: dict[str, Any] = {}
            flow_standardized = []
            flow_rmse = {}
            for channel, role in enumerate(("u", "v", "p", "omega")):
                valid = query_weight * np.isfinite(actual[index, :, channel])
                error = predicted[index, :, channel] - np.nan_to_num(actual[index, :, channel])
                denominator = max(float(valid.sum()), 1e-12)
                mse = float(np.sum(error * error * valid) / denominator)
                target_norm = float(np.sqrt(np.sum(np.square(np.nan_to_num(actual[index, :, channel])) * valid)))
                error_norm = float(np.sqrt(np.sum(error * error * valid)))
                role_metrics[role] = {
                    "rmse": math.sqrt(max(mse, 0.0)),
                    "relative_l2": error_norm / max(target_norm, 1e-12),
                    "valid_rows": int(np.count_nonzero(valid)),
                    "unit": "packed_dataset_native",
                }
                flow_rmse[role] = role_metrics[role]["rmse"]
                flow_standardized.append(mse / (std[channel] ** 2))
            flow_group = float(np.mean(flow_standardized))

            t_valid = query_weight * np.isfinite(actual[index, :, 4])
            t_error = predicted[index, :, 4] - np.nan_to_num(actual[index, :, 4])
            fluid_mse = float(np.sum(t_error * t_error * t_valid) / max(float(t_valid.sum()), 1e-12))
            role_metrics["temperature_fluid"] = {
                "rmse": math.sqrt(max(fluid_mse, 0.0)),
                "relative_l2": float(np.sqrt(np.sum(t_error * t_error * t_valid))) /
                    max(float(np.sqrt(np.sum(np.square(np.nan_to_num(actual[index, :, 4])) * t_valid))), 1e-12),
                "valid_rows": int(np.count_nonzero(t_valid)),
                "unit": "packed_dataset_native_temperature",
            }

            surface_target = interface_target[index, ..., 0]
            surface_mask = (
                present[index, :, None]
                & surface_valid[index]
                & interface_valid_input[index]
                & np.isfinite(surface_target)
            )
            surface_error = pred_interface[index, ..., 0] - np.nan_to_num(surface_target)
            surface_by_module = []
            surface_target_by_module = []
            for module in range(surface_mask.shape[0]):
                if surface_mask[module].any():
                    surface_by_module.append(float(np.mean(surface_error[module, surface_mask[module]] ** 2)))
                    surface_target_by_module.append(float(np.mean(surface_target[module, surface_mask[module]] ** 2)))
            surface_mse = float(np.mean(surface_by_module)) if surface_by_module else 0.0
            surface_reference_mse = float(np.mean(surface_target_by_module)) if surface_target_by_module else 0.0
            role_metrics["temperature_surface"] = {
                "rmse": math.sqrt(max(surface_mse, 0.0)),
                "relative_l2": math.sqrt(max(surface_mse, 0.0)) /
                    max(math.sqrt(max(surface_reference_mse, 0.0)), 1e-12),
                "valid_rows": int(surface_mask.sum()),
                "active_modules": len(surface_by_module),
                "unit": "packed_dataset_native_temperature",
            }

            material_mask = (
                present[index, :, None]
                & material_valid_geometry[index]
                & np.isfinite(material_target[index])
            )
            material_error = pred_material[index] - np.nan_to_num(material_target[index])
            material_by_module = []
            material_target_by_module = []
            for module in range(material_mask.shape[0]):
                if material_mask[module].any():
                    material_by_module.append(float(np.mean(material_error[module, material_mask[module]] ** 2)))
                    material_target_by_module.append(float(np.mean(material_target[index, module, material_mask[module]] ** 2)))
            material_mse = float(np.mean(material_by_module)) if material_by_module else 0.0
            material_reference_mse = float(np.mean(material_target_by_module)) if material_target_by_module else 0.0
            endpoint_maxima = _endpoint_material_maximum_rows(
                pred_material[index], material_target[index], material_mask
            )
            role_metrics["temperature_material"] = {
                "rmse": math.sqrt(max(material_mse, 0.0)),
                "relative_l2": math.sqrt(max(material_mse, 0.0)) /
                    max(math.sqrt(max(material_reference_mse, 0.0)), 1e-12),
                "valid_rows": int(material_mask.sum()),
                "active_modules": len(material_by_module),
                "unit": "packed_dataset_native_temperature",
            }
            role_metrics["temperature_material_endpoint_maxima"] = {
                **_summarize_endpoint_material_maxima(endpoint_maxima),
                "per_module": endpoint_maxima,
                "unit": "packed_dataset_native_temperature",
                "definition": "max(endpoint prediction) minus max(endpoint target), over aligned valid receivers",
            }

            q_target = interface_target[index, ..., 1]
            q_mask = (
                present[index, :, None]
                & surface_valid[index]
                & outside_valid[index]
                & interface_valid_input[index]
                & np.isfinite(q_target)
            )
            q_error = pred_interface[index, ..., 1] - np.nan_to_num(q_target)
            q_by_module = [float(np.mean(q_error[module, q_mask[module]] ** 2))
                           for module in range(q_mask.shape[0]) if q_mask[module].any()]
            q_mse = float(np.mean(q_by_module)) if q_by_module else 0.0
            thermal_group = float(np.mean((
                fluid_mse / std[4] ** 2,
                surface_mse / interface_std[0] ** 2,
                material_mse / material_std ** 2,
            )))
            rows.append({
                "case_id": str(case_id),
                "module_count": int(present[index].sum()),
                "field_score": 0.5 * flow_group + 0.5 * thermal_group,
                "flow_group_standardized_mse": flow_group,
                "thermal_group_standardized_mse": thermal_group,
                "flow_rmse": flow_rmse,
                "roles": role_metrics,
                "q_proxy_rmse": math.sqrt(max(q_mse, 0.0)),
                "q_proxy_relative_l2": math.sqrt(max(q_mse, 0.0)) /
                    max(float(np.sqrt(np.mean(np.square(np.nan_to_num(q_target[q_mask]))))) if q_mask.any() else 0.0, 1e-12),
                "q_proxy_valid_rows": int(q_mask.sum()),
                "temperature_unit": "packed_dataset_native_temperature",
                "flow_unit": "packed_dataset_native",
            })
        return {"case_rows": rows}

    def reduce_native_metrics(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        expected_case_ids: Sequence[str] | None = None,
    ) -> Mapping[str, Any]:
        rows = [dict(row) for record in records for row in record.get("case_rows", ())]
        expected_ids = tuple(self.validation_case_ids if expected_case_ids is None else map(str, expected_case_ids))
        observed = [str(row["case_id"]) for row in rows]
        if len(observed) != len(expected_ids) or set(observed) != set(expected_ids) or len(set(observed)) != len(observed):
            raise ValueError("Thermal native metric reduction requires each expected case exactly once.")
        endpoint_rows = [
            {
                "case_id": str(row["case_id"]),
                "module_count": int(row["module_count"]),
                **dict(module_row),
            }
            for row in rows
            for module_row in row["roles"]["temperature_material_endpoint_maxima"]["per_module"]
        ]
        endpoint_by_module_count = {}
        for module_count in (3, 5, 7, 10):
            selected_cases = [row for row in rows if int(row["module_count"]) == module_count]
            selected_modules = [
                item for item in endpoint_rows if item["module_count"] == module_count
            ]
            endpoint_by_module_count[f"M{module_count}"] = {
                "case_count": len(selected_cases),
                **_summarize_endpoint_material_maxima(selected_modules),
            }
        return {
            "scope": self.validation_scope if expected_case_ids is None else "explicit fixed native case panel",
            "case_count": len(rows),
            "case_ids": observed,
            "field_score": float(np.mean([row["field_score"] for row in rows])),
            "flow_group_standardized_mse": float(np.mean([row["flow_group_standardized_mse"] for row in rows])),
            "thermal_group_standardized_mse": float(np.mean([row["thermal_group_standardized_mse"] for row in rows])),
            "per_case": rows,
            "flow_rmse_mean": {
                role: float(np.mean([row["flow_rmse"][role] for row in rows]))
                for role in ("u", "v", "p", "omega")
            },
            "temperature_rmse_mean": {
                name: float(np.mean([row["roles"][name]["rmse"] for row in rows]))
                for name in ("temperature_fluid", "temperature_surface", "temperature_material")
            },
            "temperature_material_endpoint_maxima": {
                **_summarize_endpoint_material_maxima(endpoint_rows),
                "per_module": endpoint_rows,
                "by_module_count": endpoint_by_module_count,
                "definition": "Per active module: max(endpoint prediction) minus max(endpoint target), then error summaries over modules.",
                "unit": "packed_dataset_native_temperature",
            },
            "q_proxy_rmse_mean": float(np.mean([row["q_proxy_rmse"] for row in rows])),
        }

    def reduce_native_evaluation_metrics(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        split: str = "validation",
        case_keys: Sequence[str | int] | None = None,
    ) -> Mapping[str, Any]:
        """Reduce endpoint records while enforcing the requested fixed panel."""
        split = str(split).lower()
        if split == "validation" and case_keys is None:
            expected = self.validation_case_ids
        elif split == "train":
            if case_keys is None:
                raise ValueError("Native TRAIN metric reduction requires an explicit fixed case_keys panel.")
            expected = tuple(
                self.training_case_ids[int(key)] if isinstance(key, (int, np.integer)) else str(key)
                for key in case_keys
            )
        elif split in ("representative", "representative_detail"):
            if case_keys is None:
                expected = THERMAL_CONTENT_REPRESENTATIVE_IDS
            else:
                expected = tuple(str(key) for key in case_keys)
        elif split == "validation" and case_keys is not None:
            expected = tuple(
                self.validation_case_ids[int(key)] if isinstance(key, (int, np.integer)) else str(key)
                for key in case_keys
            )
        else:
            raise ValueError("Thermal native metric split must be validation, train, or representative_detail.")
        reduced = dict(self.reduce_native_metrics(records, expected_case_ids=expected))
        reduced["split"] = split
        return reduced

    def optimizer_groups(self, model: nn.Module, arm: str, stage: str) -> Sequence[OptimizerGroupSpec]:
        del stage
        if model is not self.model or arm != model.architecture:
            raise ValueError("Thermal optimizer request has the wrong model or A-D/A-H6 arm.")
        trainable = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not trainable or any(not name.startswith("core.") for name in trainable):
            raise ValueError("Thermal adapter must expose exactly one active shared content core.")
        no_decay = tuple(name for name in trainable if name.endswith(".bias") or "norm" in name.lower())
        decay = tuple(name for name in trainable if name not in set(no_decay))
        hold_through_epoch = 1000 if self.protocol == "development" else 2000
        schedule = ScheduleSpec(
            peak_lr=3.0e-4,
            warmup_start_lr=3.0e-5,
            warmup_epochs=20,
            hold_through_epoch=hold_through_epoch,
            total_epochs=self.total_epochs,
            final_lr=3.0e-6,
        )
        groups = []
        if decay:
            groups.append(OptimizerGroupSpec("joint_decay", decay, schedule, weight_decay=1.0e-4))
        if no_decay:
            groups.append(OptimizerGroupSpec("joint_no_decay", no_decay, schedule, weight_decay=0.0))
        return tuple(groups)

    def _schedule_payload(self) -> Mapping[str, Any]:
        hold_through_epoch = 1000 if self.protocol == "development" else 2000
        schedule = ScheduleSpec(
            peak_lr=3.0e-4, warmup_start_lr=3.0e-5, warmup_epochs=20,
            hold_through_epoch=hold_through_epoch, total_epochs=self.total_epochs, final_lr=3.0e-6,
        )
        return {
            "peak_lr": schedule.peak_lr,
            "warmup_start_lr": schedule.warmup_start_lr,
            "warmup_epochs": schedule.warmup_epochs,
            "hold_through_epoch": schedule.hold_through_epoch,
            "total_epochs": schedule.total_epochs,
            "final_lr": schedule.final_lr,
        }

    def _native_case_batch(
        self,
        cases: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        *,
        split: str,
        ntheta: int = THERMAL_CONTENT_THETA_COUNT,
    ) -> TaskBatch:
        from channelthermal.data.collation import ChannelThermalBatchCollator

        samples = []
        case_ids = []
        for index in indices:
            case = cases[int(index)]
            case_id = str(case["case_id"])
            x_grid = np.asarray(case["x_grid"], dtype=np.float32)
            y_grid = np.asarray(case["y_grid"], dtype=np.float32)
            if x_grid.ndim == 1 and y_grid.ndim == 1:
                xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")
            elif x_grid.ndim == 2 and y_grid.shape == x_grid.shape:
                xx, yy = x_grid, y_grid
            else:
                raise ValueError(f"Thermal native grid axes are malformed for {case_id}.")
            field = np.asarray(case["steady_field"], dtype=np.float32)
            if field.shape != (*xx.shape, 5) or field.shape[:2] != (self.adapter.ny, self.adapter.nx):
                raise ValueError(f"Thermal native reference grid for {case_id} is not the fixed 64x128, five-channel field.")
            module_mask = np.asarray(case.get("module_mask"))
            if module_mask.shape != field.shape[:2] or module_mask.dtype.kind not in "bui" or not np.isin(module_mask, (0, 1)).all():
                raise ValueError(f"Thermal native case {case_id} lacks its stored boolean geometry mask.")
            if case["interface_target"].shape[1] != THERMAL_CONTENT_THETA_COUNT:
                raise ValueError(f"Thermal native case {case_id} does not contain all 64 surface angles.")
            if case["module_internal_query_points"].shape[0] != case["module_internal_temperature_points"].shape[-1]:
                raise ValueError(f"Thermal native material points lost their stored target alignment in {case_id}.")
            fluid_xy = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1).astype(np.float32)
            if fluid_xy.shape != (THERMAL_CONTENT_NATIVE_QUERY_COUNT, 2):
                raise ValueError(
                    f"Thermal native grid for {case_id} does not reconstruct all "
                    f"{THERMAL_CONTENT_NATIVE_QUERY_COUNT} fluid receivers."
                )
            fluid_weights = (np.asarray(module_mask, dtype=bool) == 0).reshape(-1).astype(np.float32)
            samples.append({
                "structure": case["structure"],
                "query_xy": fluid_xy,
                "field_targets": field.reshape(-1, 5),
                "point_weights": fluid_weights,
                "interface_target": case["interface_target"],
                "interface_condition_valid_mask": case["interface_condition_valid_mask"],
                "module_internal_temperature_points": case["module_internal_temperature_points"],
                "module_internal_query_points": case["module_internal_query_points"],
            })
            case_ids.append(case_id)
        batch = _move_tree(ChannelThermalBatchCollator()(samples), self.device)
        structure = _structure_only(batch["structure"])
        return TaskBatch(
            scene_inputs=ThermalContentSceneInputs(structure, case_ids=tuple(case_ids)),
            receivers=ThermalContentReceivers(
                batch["query_xy"], batch["module_internal_query_points"], batch["structure"]["heat_powers"],
                ntheta=ntheta,
                response_ntheta=THERMAL_CONTENT_SURFACE_POINTS,
            ),
            targets=ThermalContentTargets(
                case_ids=tuple(case_ids),
                case_indices=tuple(int(index) for index in indices),
                epoch=0,
                field_targets=batch["field_targets"],
                point_weights=batch["point_weights"],
                interface_target=batch["interface_target"],
                interface_valid_mask=batch["interface_condition_valid_mask"],
                material_targets=batch["module_internal_temperature_points"],
            ),
            auxiliary={"evaluation_split": split, "native_query_count": THERMAL_CONTENT_NATIVE_QUERY_COUNT},
            case_keys=tuple(case_ids),
        )

    def _load_native_cases(self, case_ids: Sequence[str]) -> list[dict[str, Any]]:
        loaded: list[dict[str, Any]] = []
        groups: dict[str, list[str]] = {"train": [], "test": []}
        for case_id in map(str, case_ids):
            split = self.case_split_by_id.get(case_id)
            if split not in groups:
                raise ValueError(f"Thermal case {case_id} is outside original TRAIN/test partitions.")
            groups[split].append(case_id)
        normalizer = H5Normalizer({
            str(name): np.asarray(value, dtype=np.float32)
            for name, value in self.adapter.normalization_stats.items()
        })
        for split, ids in groups.items():
            if ids:
                loaded.extend(_read_selected_cases(
                    self.data_path, ids, split=split, normalizer=normalizer, include_grid=True,
                ))
        order = {str(case_id): index for index, case_id in enumerate(case_ids)}
        loaded.sort(key=lambda case: order[str(case["case_id"])])
        return loaded

    def native_evaluation_batches(
        self,
        split: str = "validation",
        case_keys: Sequence[str | int] | None = None,
        *,
        microbatch_size: int | None = None,
    ) -> Iterable[TaskBatch]:
        """Yield full 8192-fluid/64-angle/all-material native panels in memory."""
        split = str(split).lower()
        if split not in ("validation", "train"):
            raise ValueError("Thermal native evaluation split must be validation or train.")
        if case_keys is None:
            if split == "train":
                raise ValueError("Full native TRAIN evaluation requires an explicit fixed case_keys panel.")
            case_ids = self.validation_case_ids
        else:
            pool = self.training_case_ids if split == "train" else self.validation_case_ids
            resolved = []
            for key in case_keys:
                if isinstance(key, (int, np.integer)):
                    index = int(key)
                    if not 0 <= index < len(pool):
                        raise IndexError(f"Thermal {split} native evaluation index is out of bounds.")
                    resolved.append(pool[index])
                else:
                    case_id = str(key)
                    if case_id not in set(pool):
                        raise ValueError(f"Thermal native evaluation case {case_id} is outside {split} membership.")
                    resolved.append(case_id)
            if len(set(resolved)) != len(resolved):
                raise ValueError("Thermal native evaluation case panel contains duplicates.")
            case_ids = tuple(resolved)
        chunk_size = self.microbatch_size if microbatch_size is None else int(microbatch_size)
        if chunk_size < 1:
            raise ValueError("Thermal native evaluation microbatch size must be positive.")
        for start in range(0, len(case_ids), chunk_size):
            chunk_ids = case_ids[start:start + chunk_size]
            cases = self._load_native_cases(chunk_ids)
            yield self._native_case_batch(
                cases, tuple(range(len(cases))), split=split, ntheta=THERMAL_CONTENT_THETA_COUNT
            )

    def representative_samples(self) -> Iterable[TaskBatch]:
        """Return the explicitly detailed near-response panel as native batches."""
        ids = THERMAL_CONTENT_REPRESENTATIVE_IDS
        if any(case_id not in self.case_split_by_id for case_id in ids):
            raise ValueError("The fixed Thermal representative panel is missing from the original case catalog.")
        cases = self._load_native_cases(ids)
        for index, case in enumerate(cases):
            yield self._native_case_batch(
                [case], (index,), split="representative_detail", ntheta=THERMAL_CONTENT_THETA_COUNT
            )


def build_thermal_content_hypergraph_task(
    recipe: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> tuple[ThermalContentHypergraphAdapter, ThermalContentHypergraphTask, Mapping[str, Any]]:
    """Build the selected fixed25 development or full-TRAIN1000 provider."""
    if not isinstance(recipe, Mapping):
        raise TypeError("Thermal content-hypergraph recipe must be a mapping.")
    recipe_schema = recipe.get("schema", THERMAL_CONTENT_RECIPE_V1_SCHEMA)
    if recipe_schema == THERMAL_CONTENT_RECIPE_V2_SCHEMA:
        from .content_hypergraph_task_v2 import build_thermal_content_hypergraph_task_v2

        return build_thermal_content_hypergraph_task_v2(recipe, device=device)
    if recipe_schema != THERMAL_CONTENT_RECIPE_V1_SCHEMA:
        raise ValueError(f"Unsupported Thermal content-hypergraph recipe schema {recipe_schema!r}.")
    protocol = str(recipe.get("protocol", "development"))
    if protocol not in ("development", "full_train1000"):
        raise ValueError("Thermal recipe protocol must be development or full_train1000.")
    architecture = str(recipe.get("architecture", "A-H6")).upper()
    if architecture not in ("A-D", "A-H6"):
        raise ValueError("Thermal shared-core architecture must be A-D or A-H6.")
    if recipe.get("dataset") not in ("thermal", "ThermalChannel"):
        raise ValueError("Thermal builder requires dataset=thermal.")
    expected_epochs = 2500 if protocol == "development" else 5000
    if int(recipe.get("total_epochs", expected_epochs)) != expected_epochs:
        raise ValueError(f"Thermal {protocol} horizon is fixed at {expected_epochs} epochs.")
    if int(recipe.get("seed", 0)) != 0 or int(recipe.get("primary_queries", 1024)) != 1024:
        raise ValueError("Thermal content-hypergraph is sealed to initialization seed 0 and primary Q=1024.")
    if int(recipe.get("effective_batch_size", 48)) != 48:
        raise ValueError("Thermal content-hypergraph effective batch is fixed at 48.")
    if protocol == "full_train1000" and (architecture != "A-H6" or int(recipe.get("approved_stop_after", 0)) != 1000):
        raise ValueError("Thermal full_train1000 requires A-H6 and approved_stop_after=1000.")

    resolved_manifest, data_path, manifest_path = _load_data_binding(
        recipe.get("manifest"), recipe.get("data_path"),
    )
    base_manifest_sha256 = str(resolved_manifest["manifest_sha256"])
    catalog_source, catalog = read_case_catalog(data_path)
    case_split_by_id = {str(record["case_id"]): str(record["split"]).lower() for record in catalog}
    catalog_ids = tuple(case_split_by_id)
    if protocol == "development":
        train_ids = tuple(development_case_ids(resolved_manifest, "train"))
        validation_ids = tuple(development_case_ids(resolved_manifest, "test"))
        if len(train_ids) != 150 or len(validation_ids) != 22 or resolved_manifest["manifest_sha256"] != FIXED25_FINGERPRINT:
            raise ValueError("Thermal development requires the exact fixed25_v1 TRAIN150/DEV22 manifest.")
        effective_manifest = dict(resolved_manifest)
        followup_identity = None
    else:
        train_ids = tuple(record["case_id"] for record in catalog if record["split"] == "train")
        original_test_ids = tuple(record["case_id"] for record in catalog if record["split"] == "test")
        if len(original_test_ids) != 90 or original_test_ids.count("0273") != 1:
            raise ValueError("Thermal full follow-up requires original test90 with the historical duplicate 0273.")
        validation_ids = tuple(case_id for case_id in original_test_ids if case_id != "0273")
        followup_identity = validate_full_train_followup_membership(train_ids, validation_ids)
        followup_manifest = {
            "execution_protocol": THERMAL_CONTENT_FULL_TRAIN_PROTOCOL,
            "dataset_protocol": THERMAL_CONTENT_FULL_TRAIN_DATASET,
            "base_manifest_sha256": base_manifest_sha256,
            "training_case_ids": list(train_ids),
            "training_membership_sha256": followup_identity["training_membership_sha256"],
            "validation_case_ids": list(validation_ids),
            "validation_membership_sha256": followup_identity["validation_membership_sha256"],
            "excluded_validation_id": "0273",
            "excluded_validation_reason": "historical duplicate in original90 validation; preserve canonical89",
        }
        effective_manifest = {
            **resolved_manifest,
            "full_train_followup1000": followup_manifest,
            "manifest_sha256": _stable_sha256(followup_manifest),
        }

    normalizer = fit_global_normalizer(data_path, train_ids)
    training_cases = _read_selected_cases(
        data_path, train_ids, split="train", normalizer=normalizer, include_grid=False,
    )
    validation_cases = _read_selected_cases(
        data_path, validation_ids, split="test", normalizer=normalizer, include_grid=False,
    )
    atlas_directory = Path(recipe.get("atlas_directory", DEFAULT_ATLAS_DIRECTORY)).expanduser().resolve()
    train_families = _read_response_families(atlas_directory, TRAIN_RESPONSE_IDS)
    tools_directory = Path(__file__).resolve().parents[4] / "tools"
    import sys

    if str(tools_directory) not in sys.path:
        sys.path.insert(0, str(tools_directory))
    from thermal_source_response_fit import response_scales

    saved_response_scales = response_scales(train_families)
    model_overrides = dict(recipe.get("model", {}))
    allowed_model_fields = {
        "context_dim", "num_heads", "context_depth", "num_hyperedges",
        "thermal_response_rank", "local_hidden_dim", "dropout", "init_seed",
    }
    if set(model_overrides) - allowed_model_fields:
        raise ValueError("Thermal model override contains unsupported shared-core constructor fields.")
    core_config = {
        "module_input_dim": 8,
        "environment_input_dim": 8,
        "coordinate_dim": 2,
        "query_input_dim": 0,
        "operating_input_dim": 14,
        "output_dim": 3,
        "architecture": architecture,
        "init_seed": 0,
        **model_overrides,
    }
    model = ThermalContentHypergraphAdapter(
        normalizer.stats,
        core_config=core_config,
        nx=128,
        ny=64,
        environment_nx=24,
        environment_ny=8,
    ).to(device)
    source_stat = data_path.stat()
    source_binding = {
        "dataset_path": str(data_path),
        "dataset_id": "thermal_channel_global_v1",
        "dataset_size_bytes": int(source_stat.st_size),
        "dataset_mtime_ns": int(source_stat.st_mtime_ns),
        "catalog_metadata_sha256": catalog_source["metadata_sha256"],
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "base_manifest_sha256": base_manifest_sha256,
        "protocol": protocol,
        "normalization_source": "TRAIN-only H5Normalizer fit_global_normalizer",
        "response_family_sources": [
            {"family_id": str(item["family_id"]), "sha256": str(item["source_sha256"])}
            for item in train_families
        ],
        "full_followup_membership": followup_identity,
        "original_test_targets_outside_declared_panel_read": False,
    }
    provider = ThermalContentHypergraphTask(
        model=model,
        training_cases=training_cases,
        validation_cases=validation_cases,
        manifest=effective_manifest,
        base_manifest_sha256=base_manifest_sha256,
        data_path=data_path,
        source_binding=source_binding,
        train_families=train_families,
        response_scales=saved_response_scales,
        recipe=recipe,
        seed=int(recipe.get("seed", 0)),
        primary_queries=int(recipe.get("primary_queries", 1024)),
        microbatch_size=int(recipe.get("microbatch_size", 8)),
        effective_batch_size=int(recipe.get("effective_batch_size", 48)),
        total_epochs=expected_epochs,
        device=device,
        protocol=protocol,
        case_split_by_id=case_split_by_id,
        catalog_case_ids=catalog_ids,
        manifest_path=manifest_path,
        atlas_directory=atlas_directory,
    )
    preparation = provider.preparation_summary()
    return model, provider, preparation


__all__ = [
    "ThermalContentHypergraphTask",
    "ThermalContentPredictions",
    "ThermalContentReceivers",
    "ThermalContentSceneInputs",
    "ThermalContentTargets",
    "THERMAL_CONTENT_PROVIDER_SCHEMA",
    "THERMAL_CONTENT_PROVIDER_V2_SCHEMA",
    "THERMAL_CONTENT_RECIPE_V1_SCHEMA",
    "THERMAL_CONTENT_RECIPE_V2_SCHEMA",
    "build_thermal_content_hypergraph_task",
]
