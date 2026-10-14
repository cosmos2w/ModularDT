"""Versioned Thermal v2 provider for the shared content-hypergraph engine."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..content_hypergraph import (
    THERMAL_CONTENT_COORDINATE_FRAME_V2,
    THERMAL_CONTENT_GEOMETRY_CONTRACT_V2,
    THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2,
    ThermalContentHypergraphAdapterV2,
)
from ..data.datasets import fit_global_normalizer
from ..data.development_split import development_case_ids, read_case_catalog
from .content_hypergraph_task import (
    THERMAL_CONTENT_FULL_TRAIN_DATASET,
    THERMAL_CONTENT_FULL_TRAIN_PROTOCOL,
    THERMAL_CONTENT_PROVIDER_V2_SCHEMA,
    THERMAL_CONTENT_RECIPE_V2_SCHEMA,
    ThermalContentPredictions,
    ThermalContentReceivers,
    ThermalContentSceneInputs,
    ThermalContentHypergraphTask,
    _stable_sha256,
)
from .joint_task import _load_data_binding, _read_selected_cases, validate_full_train_followup_membership
from .unified_task import (
    DEFAULT_ATLAS_DIRECTORY,
    FIXED25_FINGERPRINT,
    TRAIN_RESPONSE_IDS,
    _read_response_families,
)


_MODEL_FIELDS = {
    "module_input_dim": 8,
    "environment_input_dim": 8,
    "coordinate_dim": 2,
    "query_input_dim": 0,
    "operating_input_dim": 14,
    "output_dim": 3,
    "context_dim": 128,
    "num_heads": 4,
    "context_depth": 2,
    "num_hyperedges": 6,
    "thermal_response_rank": 4,
    "local_hidden_dim": 128,
    "dropout": 0.0,
    "init_seed": 0,
}
_OPTIMIZER_FIELDS = {"weight_decay": 1.0e-4, "bias_and_norm_weight_decay": 0.0}


class ThermalContentHypergraphTaskV2(ThermalContentHypergraphTask):
    """Reuse the sealed Thermal targets, sampling, and objective under v2 identity."""

    def identity_payload(self) -> Mapping[str, Any]:
        payload = dict(super().identity_payload())
        payload["schema"] = THERMAL_CONTENT_PROVIDER_V2_SCHEMA
        payload["coordinate_frame"] = {
            **dict(payload["coordinate_frame"]),
            "name": THERMAL_CONTENT_COORDINATE_FRAME_V2,
            "origin": "[0,0]",
            "extent": "[Lx,Ly]",
        }
        payload["thermal_v2_geometry"] = copy.deepcopy(THERMAL_CONTENT_GEOMETRY_CONTRACT_V2)
        payload["thermal_v2_native_laws"] = copy.deepcopy(THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2)
        payload["provider_inheritance"] = {
            "dataset_membership": "unchanged_from_thermal_content_hypergraph_provider_v1",
            "query_sampling": "case_epoch_v1_unchanged",
            "response_addendum": "original_train_families_and_scales_unchanged",
            "objective": "native_flow_temperature_flux_and_response_weights_unchanged",
        }
        return payload

    def preparation_summary(self) -> Mapping[str, Any]:
        return {
            **dict(super().preparation_summary()),
            "provider_schema": THERMAL_CONTENT_PROVIDER_V2_SCHEMA,
            "adapter_schema": "thermal_content_hypergraph_adapter_v2",
            "geometry_contract": copy.deepcopy(THERMAL_CONTENT_GEOMETRY_CONTRACT_V2),
            "optimizer_bias_and_norm_weight_decay": 0.0,
        }

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
        if model is not self.model or execution_mode not in ("A2-D", "A2-H6"):
            raise ValueError("Thermal v2 prediction requires the bound A2-D/A2-H6 model.")
        if model.architecture != execution_mode or phase not in ("joint", "validation", "development"):
            raise ValueError("Thermal v2 prediction phase or architecture differs from its recipe.")
        prepared = model.prepare_native(
            scene.structure,
            receivers.fluid_xy,
            local_query_points=receivers.local_query_points,
            ntheta=receivers.ntheta,
            scene_key=scene.case_ids,
        )
        main, primary_decode_work = model.apply_native_with_work_counts(prepared, receivers.heat)
        response_prepared = response_main = None
        response_decode_work: Mapping[str, int] = {}
        if receivers.response_fluid_xy is not None:
            if (
                scene.response_structure is None
                or receivers.response_local_query_points is None
                or receivers.response_heat is None
            ):
                raise ValueError("Thermal v2 TRAIN response addendum is missing geometry or heat controls.")
            response_prepared = model.prepare_native(
                scene.response_structure,
                receivers.response_fluid_xy,
                local_query_points=receivers.response_local_query_points,
                ntheta=receivers.response_ntheta,
                scene_key=("original_train_response",),
            )
            response_main, response_decode_work = model.apply_native_with_work_counts(
                response_prepared, receivers.response_heat, increment=True,
            )
        work = {
            "native_fluid_queries": int(receivers.fluid_xy.shape[0] * receivers.fluid_xy.shape[1]),
            "native_surface_receiver_slots": int(prepared.stencils["surface"].valid.numel()),
            "native_outside_receiver_slots": int(prepared.stencils["outside"].valid.numel()),
            "native_material_receiver_slots": int(prepared.stencils["material"].valid.numel()),
            "native_union_rows": int(prepared.union.unique_counts.sum().item()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
            "response_fluid_queries": 0 if receivers.response_fluid_xy is None else int(
                receivers.response_fluid_xy.shape[0] * receivers.response_fluid_xy.shape[1]
            ),
            "response_surface_receiver_slots": 0 if response_prepared is None else int(
                response_prepared.stencils["surface"].valid.numel()
            ),
            "response_outside_receiver_slots": 0 if response_prepared is None else int(
                response_prepared.stencils["outside"].valid.numel()
            ),
            "response_material_receiver_slots": 0 if response_prepared is None else int(
                response_prepared.stencils["material"].valid.numel()
            ),
            "response_native_union_rows": 0 if response_prepared is None else int(
                response_prepared.union.unique_counts.sum().item()
            ),
            "response_native_local_pairs": 0 if response_prepared is None else int(
                response_prepared.query_batch.local_weights.numel()
            ),
        }

        def add_core_counts(prefix: str, values: Mapping[str, int]) -> None:
            for name, value in values.items():
                if not isinstance(name, str) or type(value) is not int or value < 0:
                    raise RuntimeError("Thermal v2 core returned an invalid work-count mapping.")
                work[f"{prefix}_{name}"] = value

        primary_prepare_work = prepared.organization.work_counts
        if not isinstance(primary_prepare_work, Mapping):
            raise RuntimeError("Thermal v2 primary preparation omitted core work counters.")
        add_core_counts("core_prepare_primary", primary_prepare_work)
        add_core_counts("core_decode_primary", primary_decode_work)
        if response_prepared is not None:
            response_prepare_work = response_prepared.organization.work_counts
            if not isinstance(response_prepare_work, Mapping):
                raise RuntimeError("Thermal v2 response preparation omitted core work counters.")
            add_core_counts("core_prepare_response", response_prepare_work)
            add_core_counts("core_decode_response_increment", response_decode_work)
        predictions = ThermalContentPredictions(prepared, main, response_prepared, response_main, work)
        return predictions, {"work": work}

    def optimizer_groups(self, model: nn.Module, arm: str, stage: str):
        groups = tuple(super().optimizer_groups(model, arm, stage))
        named = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        no_decay_names = {name for group in groups if group.weight_decay == 0.0 for name in group.parameter_names}
        if no_decay_names != {
            name for name in named if name.endswith(".bias") or "norm" in name.lower()
        }:
            raise ValueError("Thermal v2 optimizer must explicitly exclude every bias and normalization parameter from decay.")
        if any(group.weight_decay not in (0.0, 1.0e-4) for group in groups):
            raise ValueError("Thermal v2 optimizer has an undeclared weight-decay value.")
        return groups


def build_thermal_content_hypergraph_task_v2(
    recipe: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> tuple[ThermalContentHypergraphAdapterV2, ThermalContentHypergraphTaskV2, Mapping[str, Any]]:
    """Build A2-D/A2-H6 using the original fixed25 data and response protocol."""
    if not isinstance(recipe, Mapping) or recipe.get("schema") != THERMAL_CONTENT_RECIPE_V2_SCHEMA:
        raise ValueError("Thermal v2 builder requires schema=content_hypergraph_recipe_v2.")
    protocol = str(recipe.get("protocol", ""))
    if protocol not in ("development", "full_train1000"):
        raise ValueError("Thermal v2 protocol must be development or full_train1000.")
    architecture = str(recipe.get("architecture", "")).upper()
    if architecture not in ("A2-D", "A2-H6"):
        raise ValueError("Thermal v2 architecture must be A2-D or A2-H6.")
    if recipe.get("dataset") not in ("thermal", "ThermalChannel"):
        raise ValueError("Thermal v2 builder requires dataset=thermal.")
    if recipe.get("model_family") != "honf_content_hypergraph":
        raise ValueError("Thermal v2 recipe must retain model_family=honf_content_hypergraph.")
    if recipe.get("adapter_schema") != "thermal_content_hypergraph_adapter_v2":
        raise ValueError("Thermal v2 recipe must select the versioned Thermal adapter explicitly.")
    if recipe.get("optimizer_bias_norm_exclusion") is not True:
        raise ValueError("Thermal v2 recipe must explicitly exclude bias and normalization weights from decay.")
    expected_epochs = 2500 if protocol == "development" else 5000
    if int(recipe.get("total_epochs", -1)) != expected_epochs:
        raise ValueError(f"Thermal v2 {protocol} horizon is fixed at {expected_epochs} epochs.")
    if (
        int(recipe.get("seed", -1)) != 0
        or int(recipe.get("primary_queries", -1)) != 1024
        or int(recipe.get("effective_batch_size", -1)) != 48
    ):
        raise ValueError("Thermal v2 is sealed to seed 0, Q=1024, and effective batch 48.")
    microbatch_size = recipe.get("microbatch_size")
    if type(microbatch_size) is not int or not 1 <= microbatch_size <= 48:
        raise ValueError("Thermal v2 microbatch packing must be an integer in [1, 48].")
    if recipe.get("precision") != "fp32" or recipe.get("edge_access") != "full":
        raise ValueError("Thermal v2 requires FP32 and full edge access.")
    optimizer = recipe.get("optimizer")
    if not isinstance(optimizer, Mapping) or dict(optimizer) != {
        "name": "AdamW",
        **_OPTIMIZER_FIELDS,
    }:
        raise ValueError("Thermal v2 requires explicit AdamW decay=1e-4 and bias/norm decay=0.")
    if protocol == "full_train1000" and (
        architecture != "A2-H6"
        or int(recipe.get("approved_stop_after", 0)) != 1000
    ):
        raise ValueError("Thermal v2 full_train1000 is A2-H6 with stop-after 1000.")

    model_fields = recipe.get("model")
    if not isinstance(model_fields, Mapping) or dict(model_fields) != _MODEL_FIELDS:
        raise ValueError("Thermal v2 model fields must match the sealed explicit constructor defaults.")
    core_config = {
        "schema": "content_hypergraph_core_v2",
        "architecture": architecture,
        **dict(model_fields),
    }

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
            raise ValueError("Thermal v2 development requires exact fixed25_v1 TRAIN150/DEV22 membership.")
        effective_manifest = dict(resolved_manifest)
        followup_identity = None
    else:
        train_ids = tuple(record["case_id"] for record in catalog if record["split"] == "train")
        original_test_ids = tuple(record["case_id"] for record in catalog if record["split"] == "test")
        if len(original_test_ids) != 90 or original_test_ids.count("0273") != 1:
            raise ValueError("Thermal v2 full follow-up requires original test90 with historical duplicate 0273.")
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
    model = ThermalContentHypergraphAdapterV2(
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
    provider = ThermalContentHypergraphTaskV2(
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
        seed=0,
        primary_queries=1024,
        microbatch_size=microbatch_size,
        effective_batch_size=48,
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
    "ThermalContentHypergraphTaskV2",
    "build_thermal_content_hypergraph_task_v2",
]
