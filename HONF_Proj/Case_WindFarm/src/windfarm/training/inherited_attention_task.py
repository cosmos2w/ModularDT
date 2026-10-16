"""Dataset-native IAG-C/IAG-Q provider binding for WindFarm."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_forward_core.content_hypergraph.contracts import SceneBatch
from honf_forward_core.inherited_attention.model import InheritedAttentionCoreV1
from honf_runtime.inherited_attention_protocol import (
    IAG_ARCHITECTURE_REPRESENTATION,
    IAG_C_ARCHITECTURE,
    IAG_FAMILY,
    IAG_FULL_PROTOCOL,
    IAG_Q_ARCHITECTURE,
    validate_inherited_attention_recipe,
)
from honf_runtime.unified_training import OptimizerGroupSpec, SamplingKey, ScheduleSpec, TaskBatch
from torch import nn

from ..content_hypergraph import (
    WIND_CONTENT_ENVIRONMENT_COUNT,
    WIND_CONTENT_ENVIRONMENT_SHAPE,
    WindContentReceivers,
    WindInheritedAttentionAdapterV1,
)
from ..data import WindFarmNativeView
from ..normalization import read_normalization_json
from .content_hypergraph_task import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DERIVED_ROOT,
    DEFAULT_JOINT_WIND_ROLE_SCALE_CACHE_DIR,
    DEFAULT_PILOT_ROOT,
    JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID,
    JOINT_WIND_PROFILE_ID,
    ROLE_NAMES,
    WIND_CONTENT_ROLE_CACHE_RAM_BYTES,
    WindContentHypergraphTask,
    WindContentPredictions,
    WindContentTargets,
    WindFormalA2DTaskV1,
    _fit_or_load_formal_transforms,
    _full_train_followup_split_manifest,
    _indices_sha256,
    _load_original_split,
    _read_fixed_rows,
    _runtime_sampling_dataset_id,
    _shared_read_only_catalogue_cache,
    _stable_json_sha256,
)
from .joint_task import validate_full_train_followup_membership
from .native_role_evaluation import (
    make_native_role_panel_batch,
    physical_role_metrics,
    summarize_physical_role_metrics,
)

IAG_WIND_PROVIDER_SCHEMA = "wind_inherited_attention_provider_v1"
IAG_WIND_FULL_PROFILE = "wind_original420_train_fullVALID90_followup1000_v1"
IAG_WIND_LOSS_NAME = "inherited_attention_wind/native_velocity_mse"
ROUND5_MOMENT_CORE_SCHEMA = "scene_prepared_moment_hyperedge_core_v1"
_WIND_REPRESENTATIONS = IAG_ARCHITECTURE_REPRESENTATION


def _core_from_config(config: Mapping[str, Any]) -> InheritedAttentionCoreV1:
    if config.get("schema") == ROUND5_MOMENT_CORE_SCHEMA:
        from honf_forward_core.inherited_attention.moment_model import MomentHyperedgeCoreV1

        return MomentHyperedgeCoreV1.from_config(dict(config))
    return InheritedAttentionCoreV1.from_config(dict(config))


def _wind_representation(recipe: Mapping[str, Any]) -> tuple[str, str]:
    architecture = recipe.get("architecture")
    representation = recipe.get("representation")
    if architecture not in _WIND_REPRESENTATIONS or representation != _WIND_REPRESENTATIONS[architecture]:
        raise ValueError("Wind inherited-attention recipe architecture and representation disagree.")
    return str(architecture), str(representation)


def _require_wind_core_architecture(model: nn.Module, architecture: str) -> None:
    config = model.get_config() if callable(getattr(model, "get_config", None)) else None
    if (not isinstance(config, Mapping)
            or config.get("architecture") != architecture
            or getattr(model, "architecture", None) != architecture):
        raise ValueError("Wind inherited-attention model architecture differs from its execution identity.")


def _provider_architecture(provider: Any, model: nn.Module) -> str:
    architecture = getattr(provider, "architecture", None)
    if architecture is None and getattr(provider, "model", None) is model:
        architecture = getattr(model, "architecture", None)
    if architecture not in _WIND_REPRESENTATIONS:
        raise ValueError("Wind inherited-attention provider has no supported bound architecture.")
    return architecture


class WindInheritedAttentionTaskV1(WindFormalA2DTaskV1):
    """Wind IAG provider retaining the formal-semantic two-region Wind law."""

    LOSS_NAME = IAG_WIND_LOSS_NAME

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        recipe = kwargs.get("recipe")
        if not isinstance(recipe, Mapping) or recipe.get("protocol") not in (
            "inherited_attention_development_v1", IAG_FULL_PROTOCOL,
        ):
            raise ValueError("Wind inherited-attention provider requires an explicit protocol recipe.")
        # Call the native provider directly. The formal superclass methods
        # below supply only the unchanged two-region sampling/loss/metrics;
        # its A2 constructor validator is deliberately not invoked.
        WindContentHypergraphTask.__init__(self, *args, allow_formal_protocol=True, **kwargs)
        if self.recipe.get("architecture") == IAG_Q_ARCHITECTURE:
            self.architecture, self.representation = _wind_representation(self.recipe)
            _require_wind_core_architecture(self.model, self.architecture)
        else:
            self.architecture, self.representation = IAG_C_ARCHITECTURE, "iag_c"
        self.initialization_seed = int(self.recipe["seed"])
        self.sampling_seed = int(self.recipe["sampling_seed"])
        if self.initialization_seed != 0 or self.sampling_seed != 42 or self.seed != 42:
            raise ValueError("Wind inherited-attention separates initialization seed 0 from sampling seed 42.")
        self.role_query_counts = {"volume": self.VOLUME_QUERIES, "hub_band": self.HUB_BAND_QUERIES}
        self.validation_role_query_counts = {
            "volume": self.VALIDATION_VOLUME_QUERIES,
            "hub_band": self.VALIDATION_HUB_BAND_QUERIES,
        }
        self.validation_query_count = self.VALIDATION_QUERY_COUNT
        train_membership_fingerprint = _indices_sha256(self.train_rows)
        if self.recipe["protocol"] == IAG_FULL_PROTOCOL:
            native_sampling_identity = {
                "protocol": "baseline_formal_v1",
                "dataset": "WindFarm",
                "train_membership_fingerprint": train_membership_fingerprint,
            }
            self.sampling_reference_scope = {
                "protocol": "baseline_formal_v1",
                "dataset_profile": IAG_WIND_FULL_PROFILE,
                "train_membership_fingerprint": train_membership_fingerprint,
                "purpose": "native case-epoch sampler namespace only; IAG protocol remains recipe-bound",
            }
            self.sampling_dataset_id = _runtime_sampling_dataset_id({
                "dataset": "WindFarm",
                "native_sampling_identity": native_sampling_identity,
            })
            self._inherited_native_sampling_identity = native_sampling_identity
        else:
            self.sampling_reference_scope = {
                "protocol": "source_read_development_v1",
                "dataset_profile": "fixed24_v1",
                "subset_manifest_sha256": str(self.manifest["manifest_sha256"]),
                "train_row_indices_sha256": train_membership_fingerprint,
                "validation_row_indices_sha256": _indices_sha256(self.validation_rows),
                "purpose": "Round3 query-hash namespace only; IAG protocol remains recipe-bound",
            }
            self._inherited_native_sampling_identity = None

    def _schedule(self) -> ScheduleSpec:
        optimizer = self.recipe["optimizer"]
        lr = float(optimizer["lr"])
        return ScheduleSpec(
            peak_lr=lr,
            warmup_start_lr=lr,
            warmup_epochs=0,
            hold_through_epoch=self.total_epochs - 1,
            total_epochs=self.total_epochs,
            final_lr=lr,
        )

    def _weight_decay(self) -> float:
        return float(self.recipe["optimizer"]["weight_decay"])

    def identity_payload(self) -> Mapping[str, Any]:
        if getattr(self, "architecture", IAG_C_ARCHITECTURE) == IAG_Q_ARCHITECTURE:
            if _provider_architecture(self, self.model) != self.architecture:
                raise ValueError("Wind IAG-Q provider architecture changed after construction.")
            _require_wind_core_architecture(self.model, self.architecture)
        payload = dict(WindContentHypergraphTask.identity_payload(self))
        full = self.recipe["protocol"] == IAG_FULL_PROTOCOL
        payload.update({
            "schema": IAG_WIND_PROVIDER_SCHEMA,
            "family": IAG_FAMILY,
            "model_family": IAG_FAMILY,
            "representation": getattr(self, "representation", "iag_c"),
            "architecture": getattr(self, "architecture", IAG_C_ARCHITECTURE),
            "dataset_profile": IAG_WIND_FULL_PROFILE if full else "fixed24_v1",
            "adapter_schema": self.adapter.SCHEMA,
            "coordinate_frame": {
                **dict(payload["coordinate_frame"]),
                "origin": "support.lower_D",
                "extent": "support.extent_D",
                "second_rotation": False,
            },
            "provider_inheritance": {
                "dataset_membership": "exact Round3 TRAIN72/DEV24 or original TRAIN420/VALID90 with TEST locked",
                "query_sampling": "formal volume/hub-band SeedSequence streams 101/102 unchanged",
                "physical_law": "TRAIN height profile plus U_ref*safe_std times the normalized residual",
                "receiver_cover": "geometry-only smoothstep cover from native E64 environment anchors",
            },
        })
        payload["sampling_reference_scope"] = copy.deepcopy(self.sampling_reference_scope)
        if self._inherited_native_sampling_identity is not None:
            payload["native_sampling_identity"] = copy.deepcopy(self._inherited_native_sampling_identity)
        payload["objective"] = {
            "name": "inherited_attention_wind_native_volume_hub_band_standardized_mse",
            "role_names": ["volume", "hub_band"],
            "role_query_counts": dict(self.role_query_counts),
            "validation_role_query_counts": dict(self.validation_role_query_counts),
            "volume_weight": self.VOLUME_WEIGHT,
            "hub_band_weight": self.HUB_BAND_WEIGHT,
            "validation_volume_weight": self.VALIDATION_VOLUME_WEIGHT,
            "validation_hub_band_weight": self.VALIDATION_HUB_BAND_WEIGHT,
            "channel_weights": [1.0, 1.0, 1.0],
            "formula": (
                "TRAIN equal-case 0.75*standardized volume MSE + 0.25*standardized hub-band MSE; "
                "VALID 0.8*volume + 0.2*hub-band on fixed Q10240; selection is val_volume_mse"
            ),
            "selection_metric": "val_volume_mse over the fixed 8192-point VALID volume panel",
            "role_component_scales_mps_are_training_objective_inputs": False,
            "target_partition_access": "TRAIN targets for optimization; fixed VALID targets for monitoring; Wind TEST locked",
            "sampling": "native volume and hub-band with replacement; SeedSequence(sampling_seed, epoch, row, stream_id 101/102)",
        }
        training = dict(payload["training"])
        training.update({
            "initialization_seed": self.initialization_seed,
            "sampling_seed": self.sampling_seed,
            "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            "effective_batch_size": 16,
            "microbatch_size": self.max_microbatch_cases,
            "primary_queries_per_case": 8192,
            "validation_queries_per_case": self.VALIDATION_QUERY_COUNT,
            "total_epochs": self.total_epochs,
            "optimizer": "AdamW",
            "optimizer_schedule": asdict(self._schedule()),
            "optimizer_weight_decay": 1.0e-5,
            "bias_and_norm_weight_decay": 1.0e-5,
            "optimizer_bias_norm_exclusion": False,
            "optimizer_group_count": 1,
            "normalization": self.recipe["normalization"],
            "absolute_epoch_transition": copy.deepcopy(self.recipe.get("sparse_transition")),
            "all_parameters_train_from_start": True,
        })
        payload["training"] = training
        payload["recipe"] = copy.deepcopy(self.recipe)
        if full:
            payload["test_row_indices_sha256"] = self.manifest["test_row_indices_sha256"]
            payload["test_layout_indices"] = list(self.manifest["test_layout_indices"])
            payload["unexposed_test_target_values_read"] = False
        return payload

    def serialization_payload(self) -> Mapping[str, Any]:
        identity = self.identity_payload()
        result = {
            "schema": "wind_inherited_attention_inference_v1",
            "model_config": identity["model_config"],
            "adapter_state": identity["adapter_state"],
            "coordinate_frame": identity["coordinate_frame"],
            "environment_representation": identity["environment_representation"],
            "data_binding": {
                "dataset_profile": identity["dataset_profile"],
                "subset_manifest_sha256": identity["subset_manifest_sha256"],
                "train_row_indices_sha256": identity["train_row_indices_sha256"],
                "normalizer_source": identity["normalizer_and_background_source"],
            },
        }
        if getattr(self, "architecture", IAG_C_ARCHITECTURE) == IAG_Q_ARCHITECTURE:
            result["architecture"] = identity["architecture"]
            result["representation"] = identity["representation"]
        return result

    def preparation_summary(self) -> Mapping[str, Any]:
        result = dict(WindContentHypergraphTask.preparation_summary(self))
        full = self.recipe["protocol"] == IAG_FULL_PROTOCOL
        result.update({
            "protocol": self.recipe["protocol"],
            "dataset_profile": IAG_WIND_FULL_PROFILE if full else "fixed24_v1",
            "provider_schema": IAG_WIND_PROVIDER_SCHEMA,
            "adapter_schema": self.adapter.SCHEMA,
            "normalization": self.recipe["normalization"],
            "initialization_seed": self.initialization_seed,
            "sampling_seed": self.sampling_seed,
            "validation_queries_per_case": self.VALIDATION_QUERY_COUNT,
            "approved_stop_after": self.total_epochs,
            "optimizer_updates_per_epoch": int(np.ceil(len(self.train_rows) / self.effective_batch_size)),
            "optimizer": "AdamW all parameters, constant 3e-4, weight decay 1e-5",
            "wind_test_target_values_read": False,
        })
        if getattr(self, "architecture", IAG_C_ARCHITECTURE) == IAG_Q_ARCHITECTURE:
            result["architecture"] = self.architecture
            result["representation"] = self.representation
        if not full:
            result["uncovered_train_module_count_strata"] = list(
                self.manifest.get("train_layout_selection", {}).get(
                    "uncovered_turbine_count_strata_by_limited_selection", ()
                )
            )
        return result

    def optimizer_groups(
        self,
        model: nn.Module,
        arm: str | None = None,
        stage: str = "joint",
    ) -> Sequence[OptimizerGroupSpec]:
        del stage
        if getattr(self, "architecture", IAG_C_ARCHITECTURE) == IAG_Q_ARCHITECTURE:
            bound_architecture = _provider_architecture(self, model)
            execution_mode = bound_architecture if arm is None else arm
            if model is not self.model or execution_mode != bound_architecture:
                raise ValueError("Wind IAG-Q optimizer is bound only to its recipe and core.")
            _require_wind_core_architecture(model, execution_mode)
        else:
            execution_mode = IAG_C_ARCHITECTURE if arm is None else arm
            if (model is not self.model or execution_mode != IAG_C_ARCHITECTURE
                    or model.get_config().get("architecture") != IAG_C_ARCHITECTURE):
                raise ValueError("Wind IAG-C optimizer is bound only to this core and IAG-C arm.")
        optimizer = self.recipe["optimizer"]
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not names or names != tuple(name for name, _ in model.named_parameters()):
            raise ValueError("Every Wind inherited-attention core parameter must train jointly from initialization.")
        return (
            OptimizerGroupSpec(
                "iag_adamw_all_parameters",
                names,
                self._schedule(),
                weight_decay=float(optimizer["weight_decay"]),
                betas=tuple(float(value) for value in optimizer["betas"]),
                eps=float(optimizer["eps"]),
            ),
        )

    def predict_native(
        self,
        model: nn.Module,
        scene: SceneBatch,
        receivers: WindContentReceivers,
        execution_mode: str | None = None,
        phase: str = "joint",
        epoch: int = 0,
        temperature: float | None = None,
        trace_request: Any | None = None,
    ) -> tuple[WindContentPredictions, Mapping[str, Any]]:
        del temperature, trace_request
        if getattr(self, "architecture", IAG_C_ARCHITECTURE) == IAG_Q_ARCHITECTURE:
            bound_architecture = _provider_architecture(self, model)
            execution_mode = bound_architecture if execution_mode is None else execution_mode
            if model is not self.model or execution_mode != bound_architecture:
                raise ValueError("Wind IAG-Q prediction requires its bound recipe architecture.")
            _require_wind_core_architecture(model, execution_mode)
        else:
            execution_mode = IAG_C_ARCHITECTURE if execution_mode is None else execution_mode
            if model is not self.model or execution_mode != IAG_C_ARCHITECTURE:
                raise ValueError("Wind IAG-C prediction requires its bound IAG-C core.")
        if phase not in ("joint", "development", "validation"):
            raise ValueError("Wind inherited-attention uses one native path for TRAIN and exposed validation.")
        setter = getattr(model, "set_inherited_attention_training_epoch", None)
        if not callable(setter):
            raise TypeError("Wind inherited-attention core lacks absolute-epoch normalization dispatch.")
        setter(int(epoch))
        if not isinstance(scene, SceneBatch):
            raise TypeError("Wind inherited-attention prediction requires the SceneBatch returned by provider.make_scene.")
        query = self.adapter.make_query_batch(scene, receivers)
        organization = model.prepare(scene)
        output = model.decode(organization, query)
        if not torch.is_tensor(output.nonlinear_field) or output.nonlinear_field.shape != query.coordinates.shape[:2] + (3,):
            raise ValueError("Wind inherited-attention must return a normalized residual field with shape [B,Q,3].")
        physical, profile, residual = self.adapter.physical_velocity(output.nonlinear_field, query.coordinates)
        role_ids = torch.as_tensor(receivers.role_ids, device=physical.device, dtype=torch.long)
        predictions = WindContentPredictions(physical, output.nonlinear_field, residual, profile, role_ids)
        return predictions, {
            "organization": organization,
            "core_output": output,
            "profile_mps": profile,
            "residual_mps": residual,
            "execution_mode": execution_mode,
            "phase": phase,
        }

    def work_counts(
        self,
        batch: TaskBatch,
        predictions: WindContentPredictions,
        auxiliary_state: Mapping[str, Any],
    ) -> Mapping[str, int]:
        del predictions
        counts: dict[str, int] = {
            "native_receivers": int(batch.receivers.coordinates_D.shape[0] * batch.receivers.coordinates_D.shape[1]),
            "active_physical_sources": sum(
                int(np.count_nonzero(item.module_present > 0.5)) for item in batch.scene_inputs
            ),
            "environment_records": int(len(batch.scene_inputs) * WIND_CONTENT_ENVIRONMENT_COUNT),
            "complete_native_prediction": 1,
        }
        for prefix, values in (
            ("core_prepare", auxiliary_state["organization"].work_counts),
            ("core_decode", auxiliary_state["core_output"].work_counts),
        ):
            if not isinstance(values, Mapping):
                raise TypeError("Wind inherited-attention prediction omitted measured work counters.")
            for name, value in values.items():
                if not isinstance(name, str) or type(value) is not int or value < 0:
                    raise RuntimeError("Wind inherited-attention core returned an invalid work counter.")
                counts[f"{prefix}_{name}"] = value
        return counts

    def native_role_evaluation_batches(
        self,
        split: str = "validation",
        case_keys: Sequence[Any] | None = None,
    ) -> Iterable[TaskBatch]:
        """Stream the fixed five-role Q8192 diagnostic panel independently of formal monitoring."""
        rows = self.native_evaluation_rows(split, case_keys)
        helper_split = "validation" if split == "validation" else "train-fixture"
        expected_role_ids = np.repeat(np.arange(5, dtype=np.int8), (1640, 1640, 1640, 1640, 1632))
        for batch_index, row in enumerate(rows):
            batch = make_native_role_panel_batch(
                self,
                row,
                split=helper_split,
                batch_index=batch_index,
            )
            role_ids = np.asarray(batch.receivers.role_ids)
            if role_ids.shape != (1, 8192) or not np.array_equal(role_ids[0], expected_role_ids):
                raise ValueError("Wind native role diagnostics require the exact five-role Q8192 board.")
            counts = batch.auxiliary.get("role_query_counts")
            if not isinstance(counts, Mapping) or tuple(int(counts[name]) for name in ROLE_NAMES) != (
                1640, 1640, 1640, 1640, 1632,
            ):
                raise ValueError("Wind native role panel changed its sealed 1640x4+1632 quotas.")
            yield batch

    def native_role_evaluation_metrics(
        self,
        predictions: WindContentPredictions,
        targets: WindContentTargets,
        sampling_auxiliary: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return unscaled physical per-role metrics and the exact query receipt."""
        if not isinstance(targets, WindContentTargets):
            raise TypeError("Wind native role metrics require bound native Wind targets.")
        predicted = predictions.main_mps.detach().cpu().numpy().astype(np.float64, copy=False)
        actual = np.asarray(targets.velocity_mps, dtype=np.float64)
        role_ids = predictions.role_ids.detach().cpu().numpy()
        expected_shape = (len(targets.row_indices), 8192, 3)
        if predicted.shape != expected_shape or actual.shape != expected_shape or role_ids.shape != expected_shape[:2]:
            raise ValueError("Wind native role metrics require aligned [B,8192,3] physical arrays.")
        query_hash = sampling_auxiliary.get("query_sampling_sha256")
        if (not isinstance(query_hash, str) or len(query_hash) != 64
                or any(character not in "0123456789abcdef" for character in query_hash)):
            raise ValueError("Wind native role metrics require the batch's lowercase query SHA256 receipt.")
        rows = []
        for index, metadata in enumerate(targets.case_metadata):
            row = physical_role_metrics(metadata, predicted[index], actual[index], role_ids[index])
            row["query_sampling_sha256"] = query_hash
            rows.append(row)
        return {"rows": rows}

    def reduce_native_role_evaluation_metrics(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        split: str,
        case_keys: Sequence[Any] | None = None,
    ) -> Mapping[str, Any]:
        """Validate a complete ordered five-role panel and summarize physical tails."""
        expected_rows = self.native_evaluation_rows(split, case_keys)
        rows = [dict(row) for record in records for row in record.get("rows", ())]
        observed_rows = [int(row["row_index"]) for row in rows]
        if observed_rows != list(expected_rows):
            raise ValueError("Wind native role board must contain the exact selected rows in sealed order.")
        query_hashes: dict[str, str] = {}
        expected_counts = (1640, 1640, 1640, 1640, 1632)
        for row in rows:
            query_hash = row.get("query_sampling_sha256")
            if (not isinstance(query_hash, str) or len(query_hash) != 64
                    or any(character not in "0123456789abcdef" for character in query_hash)):
                raise ValueError("Wind native role board contains an invalid query-ID hash.")
            counts = tuple(int(row["roles"][role]["query_count"]) for role in ROLE_NAMES)
            if counts != expected_counts or sum(counts) != 8192:
                raise ValueError("Wind native role board changed its fixed Q8192 quotas.")
            query_hashes[str(int(row["row_index"]))] = query_hash
        return {
            "schema": "wind_inherited_attention_native_role_metrics_v1",
            "split": split,
            "query_count_per_row": 8192,
            "role_query_counts": dict(zip(ROLE_NAMES, expected_counts)),
            "endpoint_rows": list(expected_rows),
            "row_query_sampling_sha256": query_hashes,
            **summarize_physical_role_metrics(rows),
        }


def build_wind_inherited_attention_task_v1(
    recipe: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[InheritedAttentionCoreV1, WindInheritedAttentionTaskV1]:
    """Build the declared Wind IAG architecture on the exact native data binding."""
    value = validate_inherited_attention_recipe(recipe)
    if value["dataset"] != "wind":
        raise ValueError("Wind inherited-attention builder accepts only dataset=wind.")
    architecture, _representation = _wind_representation(value)
    full = value["protocol"] == IAG_FULL_PROTOCOL
    initialization_seed = int(value["seed"])
    sampling_seed = int(value["sampling_seed"])
    selected_device = torch.device(device)
    data_root = Path(value.get("data_root", DEFAULT_DATA_ROOT)).expanduser().resolve()
    derived_root = Path(value.get("derived_root", DEFAULT_DERIVED_ROOT)).expanduser().resolve()
    pilot_root = Path(value.get("pilot_root", DEFAULT_PILOT_ROOT)).expanduser().resolve()
    manifest_path = Path(value.get("manifest_path", pilot_root / "fixed_subset_manifest.json")).expanduser().resolve()
    normalization_path = Path(value.get("normalization_path", pilot_root / "train_only_normalization.json")).expanduser().resolve()
    binding_path = Path(value.get("normalization_binding_path", pilot_root / "train_only_normalization.binding.json")).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Native Wind data root is unavailable: {data_root}")
    view = WindFarmNativeView(
        data_root,
        allow_npz_metadata_fallback=True,
        token_shape=WIND_CONTENT_ENVIRONMENT_SHAPE,
    )
    split = _load_original_split(view, derived_root)
    if full:
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
        normalizer, profile, transform_binding = _fit_or_load_formal_transforms(
            view=view,
            train_rows=np.asarray(train_rows, dtype=np.int64),
            train_layouts=manifest["train_layout_indices"],
            manifest=manifest,
            training_fingerprint=str(manifest["manifest_sha256"]),
            seed=sampling_seed,
            cache_dir=Path(value.get("transform_cache_dir", DEFAULT_JOINT_WIND_ROLE_SCALE_CACHE_DIR)).resolve(),
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
        manifest, train_rows, validation_rows = _read_fixed_rows(view, split, manifest_path)
        uncovered = list(manifest.get("train_layout_selection", {}).get(
            "uncovered_turbine_count_strata_by_limited_selection", (),
        ))
        if (
            len(train_rows) != 72
            or len(validation_rows) != 24
            or len(manifest.get("train_layout_indices", ())) != 24
            or len(manifest.get("validation_layout_indices", ())) != 8
            or uncovered != [18]
        ):
            raise ValueError("Wind development requires the exact fixed TRAIN72/DEV24 membership and M18 disclosure.")
        normalizer, profile = read_normalization_json(normalization_path)
        if profile is None:
            raise ValueError("Wind fixed24 inherited-attention requires the existing TRAIN-fitted height-profile transform.")
        transform_binding = json.loads(binding_path.read_text(encoding="utf-8"))
        expected_binding = {
            "subset_manifest_sha256": manifest["manifest_sha256"],
            "training_row_indices_sha256": _indices_sha256(train_rows),
            "normalizer_sha256": hashlib.sha256(normalization_path.read_bytes()).hexdigest(),
        }
        if transform_binding != expected_binding:
            raise ValueError("Wind fixed24 transforms are not bound to the exact TRAIN72 membership.")
        if normalizer.source_rows != 72 or normalizer.sample_count_per_row != 8192 or normalizer.seed != sampling_seed:
            raise ValueError("Wind fixed24 normalizer differs from the existing TRAIN72 fit.")
        profile_id = JOINT_WIND_PROFILE_ID
        normalizer_source = {
            "kind": "verified_fixed24_train_fit",
            "normalization_path": str(normalization_path),
            "normalization_sha256": expected_binding["normalizer_sha256"],
            "binding_path": str(binding_path),
            "binding": transform_binding,
            "normalizer": normalizer.to_dict(),
            "background_profile": profile.to_dict(),
        }
    if profile is None:
        raise ValueError("Wind physical conversion requires the native TRAIN height profile.")
    normalizer_source["profile_sha256"] = _stable_json_sha256(profile.to_dict())
    cache = _shared_read_only_catalogue_cache(WIND_CONTENT_ROLE_CACHE_RAM_BYTES)
    adapter = WindInheritedAttentionAdapterV1(normalizer, profile, device=selected_device)
    model_config = {
        "schema": "inherited_attention_core_v1",
        "architecture": architecture,
        **dict(value["model"]),
    }
    devices: list[int] = []
    if selected_device.type == "cuda" and torch.cuda.is_available():
        devices = [torch.cuda.current_device() if selected_device.index is None else int(selected_device.index)]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(initialization_seed)
        if devices:
            torch.cuda.manual_seed_all(initialization_seed)
        model = _core_from_config(model_config).to(selected_device)
    provider = WindInheritedAttentionTaskV1(
        view,
        model=model,
        adapter=adapter,
        train_rows=train_rows,
        validation_rows=validation_rows,
        manifest=manifest,
        role_component_scales=None,
        role_scale_calibration=None,
        role_scale_cache_identity=None,
        normalizer_source=normalizer_source,
        recipe=value,
        seed=sampling_seed,
        primary_queries=int(value["primary_queries"]),
        microbatch_size=int(value["microbatch_size"]),
        effective_batch_size=int(value["effective_batch_size"]),
        total_epochs=int(value["total_epochs"]),
        device=selected_device,
        catalogue_cache=cache,
    )
    return model, provider


__all__ = [
    "IAG_WIND_LOSS_NAME",
    "IAG_WIND_PROVIDER_SCHEMA",
    "WindInheritedAttentionTaskV1",
    "build_wind_inherited_attention_task_v1",
]
