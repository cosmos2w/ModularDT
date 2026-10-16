"""Dataset-native IAG-C adapter/provider binding for ThermalChannel."""

from __future__ import annotations

import copy
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from honf_forward_core.inherited_attention.model import InheritedAttentionCoreV1
from honf_runtime.inherited_attention_protocol import (
    IAG_FAMILY,
    IAG_FULL_PROTOCOL,
    validate_inherited_attention_recipe,
)
from honf_runtime.unified_training import OptimizerGroupSpec, ScheduleSpec

from ..content_hypergraph import (
    THERMAL_CONTENT_COORDINATE_FRAME_V2,
    THERMAL_CONTENT_GEOMETRY_CONTRACT_V2,
    THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2,
    ThermalInheritedAttentionAdapterV1,
)
from ..data.datasets import fit_global_normalizer
from ..data.development_split import development_case_ids, read_case_catalog
from .content_hypergraph_task import (
    THERMAL_CONTENT_FULL_TRAIN_DATASET,
    ThermalContentHypergraphTask,
    ThermalContentPredictions,
    ThermalContentReceivers,
    ThermalContentSceneInputs,
    _ids_sha256,
    _stable_sha256,
)
from .joint_task import (
    FULL_TRAIN_FOLLOWUP_DATASET,
    FULL_TRAIN_FOLLOWUP_PROTOCOL,
    _load_data_binding,
    _read_selected_cases,
    validate_full_train_followup_membership,
)
from .unified_task import (
    DEFAULT_ATLAS_DIRECTORY,
    FIXED25_FINGERPRINT,
    TRAIN_RESPONSE_IDS,
    _read_response_families,
)

IAG_THERMAL_PROVIDER_SCHEMA = "thermal_inherited_attention_provider_v1"
IAG_THERMAL_FULL_PROFILE = THERMAL_CONTENT_FULL_TRAIN_DATASET
ROUND5_MOMENT_CORE_SCHEMA = "scene_prepared_moment_hyperedge_core_v1"


def _core_from_config(config: Mapping[str, Any]) -> InheritedAttentionCoreV1:
    if config.get("schema") == ROUND5_MOMENT_CORE_SCHEMA:
        from honf_forward_core.inherited_attention.moment_model import MomentHyperedgeCoreV1

        return MomentHyperedgeCoreV1.from_config(dict(config))
    return InheritedAttentionCoreV1.from_config(dict(config))


class ThermalInheritedAttentionTaskV1(ThermalContentHypergraphTask):
    """IAG-C predictor on the unchanged Thermal native target/objective path."""

    def identity_payload(self) -> Mapping[str, Any]:
        payload = dict(super().identity_payload())
        full = self.protocol == IAG_FULL_PROTOCOL
        sampling_reference_scope = self.manifest.get(
            "inherited_attention_sampling_reference_scope"
        )
        payload.update({
            "schema": IAG_THERMAL_PROVIDER_SCHEMA,
            "family": IAG_FAMILY,
            "model_family": IAG_FAMILY,
            "representation": "iag_c",
            "architecture": "IAG-C",
            "dataset_profile": IAG_THERMAL_FULL_PROFILE if full else "fixed25_v1",
            "adapter_schema": self.adapter.SCHEMA,
            "coordinate_frame": {
                **dict(payload["coordinate_frame"]),
                "name": THERMAL_CONTENT_COORDINATE_FRAME_V2,
                "origin": "[0,0]",
                "extent": "[Lx,Ly]",
            },
            "iag_geometry_contract": copy.deepcopy(THERMAL_CONTENT_GEOMETRY_CONTRACT_V2),
            "iag_native_laws": copy.deepcopy(THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2),
            "provider_inheritance": {
                "dataset_membership": "exact Round3 TRAIN150/DEV22 or original TRAIN600/canonical VALID89",
                "query_sampling": "case_epoch_v1 unchanged",
                "response_addendum": "original TRAIN response families and scales unchanged",
                "native_objective": "Thermal flow, temperature, flux, and response terms unchanged",
                "receiver_cover": "geometry-only smoothstep cover from the native E192 anchors",
            },
        })
        if sampling_reference_scope is not None:
            # This pure metadata scope preserves the established Round3
            # case_epoch_v1 namespace. It does not select the IAG workflow,
            # objective, transforms, or model; those remain bound separately
            # by the IAG recipe and provider identity above.
            payload["sampling_reference_scope"] = copy.deepcopy(sampling_reference_scope)
        payload["source_binding"] = {
            **dict(payload["source_binding"]),
            "normalization_fit_scope": (
                "all original TRAIN600 only" if full else "fixed selected TRAIN150 only"
            ),
        }
        training = dict(payload["training"])
        learning_rate = float(self.recipe["optimizer"]["lr"])
        training.update({
            "optimizer": "AdamW",
            "optimizer_schedule": self._iag_schedule_payload(),
            "weight_decay": float(self.recipe["optimizer"]["weight_decay"]),
            "bias_and_norm_weight_decay": float(self.recipe["optimizer"]["bias_and_norm_weight_decay"]),
            "optimizer_group_count": 1,
            "optimizer_bias_norm_exclusion": False,
            "all_parameters_train_from_start": True,
            "normalization": self.recipe["normalization"],
            "absolute_epoch_transition": copy.deepcopy(self.recipe.get("sparse_transition")),
            "learning_rate": learning_rate,
        })
        payload["training"] = training
        payload["recipe"] = copy.deepcopy(self.recipe)
        return payload

    def preparation_summary(self) -> Mapping[str, Any]:
        full = self.protocol == IAG_FULL_PROTOCOL
        return {
            **dict(super().preparation_summary()),
            "status": "prepared_only",
            "protocol": self.protocol,
            "dataset_profile": IAG_THERMAL_FULL_PROFILE if full else "fixed25_v1",
            "provider_schema": IAG_THERMAL_PROVIDER_SCHEMA,
            "adapter_schema": self.adapter.SCHEMA,
            "normalization": self.recipe["normalization"],
            "normalization_scope": "TRAIN only",
            "approved_stop_after": self.total_epochs,
            "optimizer": "AdamW all parameters, constant 3e-4, weight decay 1e-5",
            "solver_attempts": 0,
        }

    def _iag_schedule(self) -> ScheduleSpec:
        lr = float(self.recipe["optimizer"]["lr"])
        return ScheduleSpec(
            peak_lr=lr,
            warmup_start_lr=lr,
            warmup_epochs=0,
            hold_through_epoch=self.total_epochs - 1,
            total_epochs=self.total_epochs,
            final_lr=lr,
        )

    def _iag_schedule_payload(self) -> Mapping[str, Any]:
        schedule = self._iag_schedule()
        return {
            "peak_lr": schedule.peak_lr,
            "warmup_start_lr": schedule.warmup_start_lr,
            "warmup_epochs": schedule.warmup_epochs,
            "hold_through_epoch": schedule.hold_through_epoch,
            "total_epochs": schedule.total_epochs,
            "final_lr": schedule.final_lr,
        }

    def optimizer_groups(self, model: nn.Module, arm: str, stage: str) -> Sequence[OptimizerGroupSpec]:
        del stage
        if model is not self.model or arm != "IAG-C" or model.architecture != "IAG-C":
            raise ValueError("Thermal IAG-C optimizer is bound only to its native adapter/core.")
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not names or names != tuple(name for name, _ in model.named_parameters()):
            raise ValueError("Every Thermal IAG-C core parameter must train jointly from initialization.")
        return (
            OptimizerGroupSpec(
                "iag_adamw_all_parameters",
                names,
                self._iag_schedule(),
                weight_decay=float(self.recipe["optimizer"]["weight_decay"]),
                betas=tuple(float(value) for value in self.recipe["optimizer"]["betas"]),
                eps=float(self.recipe["optimizer"]["eps"]),
            ),
        )

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
        del temperature
        if model is not self.model or execution_mode != "IAG-C":
            raise ValueError("Thermal IAG-C prediction requires its bound IAG-C adapter.")
        if phase not in ("joint", "validation", "development"):
            raise ValueError("Thermal IAG-C uses one native path for TRAIN and exposed validation.")
        setter = getattr(model.core, "set_inherited_attention_training_epoch", None)
        if not callable(setter):
            raise TypeError("Thermal IAG-C core lacks absolute-epoch normalization dispatch.")
        setter(int(epoch))
        prepared = model.prepare_native(
            scene.structure,
            receivers.fluid_xy,
            local_query_points=receivers.local_query_points,
            ntheta=receivers.ntheta,
            scene_key=scene.case_ids,
        )
        main, primary_work = model.apply_native_with_work_counts(prepared, receivers.heat)
        response_prepared = response_main = None
        response_work: Mapping[str, int] = {}
        if receivers.response_fluid_xy is not None:
            if (scene.response_structure is None or receivers.response_local_query_points is None
                    or receivers.response_heat is None):
                raise ValueError("Thermal IAG-C response addendum is missing bound native controls.")
            response_prepared = model.prepare_native(
                scene.response_structure,
                receivers.response_fluid_xy,
                local_query_points=receivers.response_local_query_points,
                ntheta=receivers.response_ntheta,
                scene_key=("original_train_response",),
            )
            response_main, response_work = model.apply_native_with_work_counts(
                response_prepared, receivers.response_heat, increment=True,
            )
        work: dict[str, int] = {
            "native_fluid_queries": int(receivers.fluid_xy.shape[0] * receivers.fluid_xy.shape[1]),
            "native_union_rows": int(prepared.union.unique_counts.sum().item()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
        }
        for prefix, values in (
            ("core_prepare_primary", prepared.organization.work_counts),
            ("core_decode_primary", primary_work),
            ("core_prepare_response", {} if response_prepared is None else response_prepared.organization.work_counts),
            ("core_decode_response_increment", response_work),
        ):
            if not isinstance(values, Mapping):
                raise TypeError("Thermal IAG-C core omitted measured work counters.")
            for name, value in values.items():
                if not isinstance(name, str) or type(value) is not int or value < 0:
                    raise RuntimeError("Thermal IAG-C core returned an invalid work counter.")
                work[f"{prefix}_{name}"] = value
        predictions = ThermalContentPredictions(prepared, main, response_prepared, response_main, work)
        return predictions, {"work": work}


def build_thermal_inherited_attention_task_v1(
    recipe: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> tuple[ThermalInheritedAttentionAdapterV1, ThermalInheritedAttentionTaskV1, Mapping[str, Any]]:
    """Build IAG-C with the exact Round3 data views and native response law."""
    value = validate_inherited_attention_recipe(recipe)
    if value["dataset"] != "thermal":
        raise ValueError("Thermal IAG-C builder accepts only dataset=thermal.")
    protocol = value["protocol"]
    full = protocol == IAG_FULL_PROTOCOL
    resolved_manifest, data_path, manifest_path = _load_data_binding(
        value.get("manifest"), value.get("data_path"),
    )
    base_manifest_sha256 = str(resolved_manifest["manifest_sha256"])
    catalog_source, catalog = read_case_catalog(data_path)
    case_split_by_id = {str(record["case_id"]): str(record["split"]).lower() for record in catalog}
    catalog_ids = tuple(case_split_by_id)
    if full:
        train_ids = tuple(record["case_id"] for record in catalog if record["split"] == "train")
        original_test_ids = tuple(record["case_id"] for record in catalog if record["split"] == "test")
        if len(original_test_ids) != 90 or original_test_ids.count("0273") != 1:
            raise ValueError("Thermal full-TRAIN IAG-C requires original TEST90 and canonical VALID89 excluding duplicate 0273.")
        validation_ids = tuple(case_id for case_id in original_test_ids if case_id != "0273")
        followup = validate_full_train_followup_membership(train_ids, validation_ids)
        established_followup_manifest = {
            "execution_protocol": FULL_TRAIN_FOLLOWUP_PROTOCOL,
            "dataset_protocol": FULL_TRAIN_FOLLOWUP_DATASET,
            "base_manifest_sha256": resolved_manifest["manifest_sha256"],
            "training_case_ids": list(train_ids),
            "training_membership_sha256": followup["training_membership_sha256"],
            "validation_case_ids": list(validation_ids),
            "validation_membership_sha256": followup["validation_membership_sha256"],
            "excluded_validation_id": "0273",
            "excluded_validation_reason": "historical duplicate in original90 validation; preserve canonical89",
        }
        binding = {
            "execution_protocol": protocol,
            "dataset_protocol": IAG_THERMAL_FULL_PROFILE,
            "base_manifest_sha256": base_manifest_sha256,
            "training_case_ids": list(train_ids),
            "training_membership_sha256": followup["training_membership_sha256"],
            "validation_case_ids": list(validation_ids),
            "validation_membership_sha256": followup["validation_membership_sha256"],
            "excluded_validation_id": "0273",
            "excluded_validation_reason": "historical duplicate in original90 validation; preserve canonical89",
        }
        effective_manifest = {
            **resolved_manifest,
            "full_train_followup1000": established_followup_manifest,
            "inherited_attention_full_train1000": binding,
            # Preserve the exact native full-TRAIN sampling namespace used by
            # the established Thermal full-follow-up provider. The IAG
            # protocol binding is carried separately above and cannot reseed
            # the same cases' case_epoch_v1 queries.
            "manifest_sha256": _stable_sha256(established_followup_manifest),
        }
    else:
        train_ids = tuple(development_case_ids(resolved_manifest, "train"))
        validation_ids = tuple(development_case_ids(resolved_manifest, "test"))
        if (len(train_ids), len(validation_ids)) != (150, 22) or base_manifest_sha256 != FIXED25_FINGERPRINT:
            raise ValueError("Thermal IAG-C development requires exact fixed25_v1 TRAIN150/DEV22 membership.")
        sampling_reference_scope = {
            "protocol": "source_read_development_v1",
            "dataset_protocol": "fixed25_v1",
            "base_manifest_sha256": base_manifest_sha256,
            "training_case_ids": list(train_ids),
            "training_membership_sha256": _ids_sha256(train_ids),
            "validation_case_ids": list(validation_ids),
            "validation_membership_sha256": _ids_sha256(validation_ids),
            "normalization_scope": "fixed25 selected TRAIN150 only",
            "future_horizon_epochs": int(value["total_epochs"]),
            "approved_stop_after": int(value["approved_stop_after"]),
        }
        effective_manifest = {
            **resolved_manifest,
            "inherited_attention_sampling_reference_scope": sampling_reference_scope,
            # case_epoch_v1 hashes this metadata namespace. Keep it identical
            # to Round3 while the recipe continues to identify IAG-C itself.
            "manifest_sha256": _stable_sha256(sampling_reference_scope),
        }
        followup = None
    normalizer = fit_global_normalizer(data_path, train_ids)
    training_cases = _read_selected_cases(data_path, train_ids, split="train", normalizer=normalizer, include_grid=False)
    validation_cases = _read_selected_cases(data_path, validation_ids, split="test", normalizer=normalizer, include_grid=False)
    atlas_directory = Path(value.get("atlas_directory", DEFAULT_ATLAS_DIRECTORY)).expanduser().resolve()
    train_families = _read_response_families(atlas_directory, TRAIN_RESPONSE_IDS)
    tools_directory = Path(__file__).resolve().parents[4] / "tools"
    if str(tools_directory) not in sys.path:
        sys.path.insert(0, str(tools_directory))
    from thermal_source_response_fit import response_scales

    saved_response_scales = response_scales(train_families)
    core_config = {
        "schema": "inherited_attention_core_v1",
        "architecture": "IAG-C",
        **dict(value["model"]),
    }
    devices: list[int] = []
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and torch.cuda.is_available():
        devices = [torch.cuda.current_device() if selected_device.index is None else int(selected_device.index)]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(value["seed"]))
        if devices:
            torch.cuda.manual_seed_all(int(value["seed"]))
        core = _core_from_config(core_config)
    model = ThermalInheritedAttentionAdapterV1(
        normalizer.stats,
        core=core,
        nx=128,
        ny=64,
        environment_nx=24,
        environment_ny=8,
    ).to(selected_device)
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
        "full_followup_membership": followup,
        "original_test_targets_outside_declared_panel_read": False,
    }
    provider = ThermalInheritedAttentionTaskV1(
        model=model,
        training_cases=training_cases,
        validation_cases=validation_cases,
        manifest=effective_manifest,
        base_manifest_sha256=base_manifest_sha256,
        data_path=data_path,
        source_binding=source_binding,
        train_families=train_families,
        response_scales=saved_response_scales,
        recipe=value,
        seed=int(value["seed"]),
        primary_queries=int(value["primary_queries"]),
        microbatch_size=int(value["microbatch_size"]),
        effective_batch_size=int(value["effective_batch_size"]),
        total_epochs=int(value["total_epochs"]),
        device=selected_device,
        protocol=protocol,
        case_split_by_id=case_split_by_id,
        catalog_case_ids=catalog_ids,
        manifest_path=manifest_path,
        atlas_directory=atlas_directory,
    )
    return model, provider, provider.preparation_summary()


__all__ = [
    "IAG_THERMAL_PROVIDER_SCHEMA",
    "ThermalInheritedAttentionTaskV1",
    "build_thermal_inherited_attention_task_v1",
]
