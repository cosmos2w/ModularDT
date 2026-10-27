"""Maintained native Thermal provider for R13-6 faithful and typed readers."""
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import torch

from ..data.datasets import fit_global_normalizer
from ..data.development_split import read_case_catalog
from ..member_routing import (
    ThermalFaithfulMemberRoutingAdapter,
    ThermalTypedOrganizationMemberRoutingAdapter,
)
from .content_hypergraph_task import ThermalContentPredictions
from .joint_member_routing_task import (
    JointMemberRoutingHealth,
    ThermalJointMemberRoutingTask,
    ThermalJointPairPredictions,
)
from .member_routing_task import ThermalInteractionReusedMemberRoutingTask, _training_membership
from .joint_task import _load_data_binding, _read_selected_cases


CONFIG_SECTION_SCHEMA = "thermal_r13_6_member_routing_v1"
SOURCE_KIND = "scratch_r13_6_member_routing"
JOINT_BLOCKS = ("H_MM_applied", "H_EM_applied", "H_EE_applied", "H_ME_applied")
_VARIANTS = {
    ThermalFaithfulMemberRoutingAdapter.CORE_SCHEMA: {
        "adapter": ThermalFaithfulMemberRoutingAdapter,
        "arm": "R13-6-O-ME-FaithfulJoint95",
        "workflow": "interaction_reused_faithful_member_routing_training_v1",
        "provider_schema": "thermal_r13_6_faithful_member_routing_provider_v1",
        "training_state_schema": "thermal_r13_6_faithful_member_routing_training_state_v1",
        "candidate": "C",
    },
    ThermalTypedOrganizationMemberRoutingAdapter.CORE_SCHEMA: {
        "adapter": ThermalTypedOrganizationMemberRoutingAdapter,
        "arm": "R13-6-H-ME-TypedOrganizationJoint95",
        "workflow": "interaction_reused_typed_organization_member_routing_training_v1",
        "provider_schema": "thermal_r13_6_typed_organization_member_routing_provider_v1",
        "training_state_schema": "thermal_r13_6_typed_organization_member_routing_training_state_v1",
        "candidate": "H",
    },
}


class ThermalR136MemberRoutingTask(ThermalJointMemberRoutingTask):
    """Reuse R13-5's objective and typed health with the actual paired core call."""

    def __init__(self, **kwargs):
        # The R13-5 task methods are the maintained native objective, schedule,
        # validation and typed ME health implementation. Initialize their base
        # provider directly so R13-6 retains its own versioned identity.
        ThermalInteractionReusedMemberRoutingTask.__init__(self, **kwargs)
        variant = _VARIANTS.get(self.model.CORE_SCHEMA)
        if variant is None or self.model.architecture != variant["arm"]:
            raise ValueError("R13-6 task requires a matching faithful or typed-organization adapter.")
        self._r13_6_variant = variant
        self.candidate = str(variant["candidate"])
        self.arm = str(variant["arm"])
        self.workflow = str(variant["workflow"])
        self.provider_schema = str(variant["provider_schema"])
        self.training_state_schema = str(variant["training_state_schema"])
        initialization = self.recipe["initialization"]
        self.source_kind = str(initialization["source_kind"])
        self.source_parent_age = int(initialization.get("source_parent_age", 0))
        if (self.source_kind != SOURCE_KIND
                or self.source_parent_age != 0
                or initialization.get("source_checkpoint") is not None
                or initialization.get("pretrained_weights") is not False
                or initialization.get("source_arm") != "scratch_common_seed0"):
            raise ValueError("R13-6 requires a fresh seed-0 scratch lineage with no source checkpoint.")
        self.training_sampling_arm = str(
            self.recipe["training"].get("sampling_arm", "R13-3-Relative")
        )
        health_config = self.recipe.get("member_routing", {}).get("health")
        self._member_routing_health = None
        if isinstance(health_config, Mapping) and health_config.get("enabled") is True:
            self._member_routing_health = JointMemberRoutingHealth(self, health_config)

    def _paired_predictions(self, prepared, physical_heat):
        self.model.validate_prepared(prepared)
        if physical_heat is not None:
            given = torch.as_tensor(physical_heat, device=prepared.scene_batch.module_features.device)
            expected = prepared.structure["heat_powers"]
            if given.shape != expected.shape or not torch.equal(given.to(expected), expected):
                raise ValueError("Heat changed: rebuild the complete joint prepared context.")
        paired = self.model.core.decode_paired(prepared.organization, prepared.query_batch)
        if not isinstance(paired, tuple) or len(paired) != 3:
            raise TypeError("R13-6 paired decode must return full, restricted and shared work outputs.")
        full_output, restricted_output, shared_counts = paired

        def native(output):
            fields = getattr(output, "joint_fields", None)
            if not torch.is_tensor(fields) or fields.ndim != 3 or fields.shape[-1] != 4:
                raise ValueError("R13-6 paired decoder must return [batch, queries, 4] joint_fields.")
            return self.model._extract_native_roles(prepared, fields[..., 3:4], fields[..., :3])

        counts_by_mode = {
            "full": {str(name): int(value) for name, value in full_output.work_counts.items()},
            "restricted": {str(name): int(value) for name, value in restricted_output.work_counts.items()},
        }
        work = {
            "native_fluid_queries": int(prepared.fluid_xy.shape[0] * prepared.fluid_xy.shape[1]),
            "native_union_rows": int(prepared.union.unique_counts.sum()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
        }
        work.update({f"core_prepare_primary_{name}": int(value)
                     for name, value in prepared.organization.work_counts.items()})
        for mode, counts in counts_by_mode.items():
            work.update({f"core_decode_joint_{mode}_{name}": value for name, value in counts.items()})
        if not isinstance(shared_counts, Mapping):
            raise TypeError("R13-6 paired decode shared work must be a mapping.")
        work.update({f"core_decode_joint_shared_{name}": int(value)
                     for name, value in shared_counts.items()})
        return ThermalJointPairPredictions(
            ThermalContentPredictions(prepared, native(full_output), work=counts_by_mode["full"]),
            ThermalContentPredictions(prepared, native(restricted_output), work=counts_by_mode["restricted"]),
            work,
        )


def build_thermal_r13_6_member_routing_task(config, *, arm=None, device="cpu"):
    """Build a strict R13-6 adapter and the shared native TRAIN/VAL provider."""
    recipe = copy.deepcopy(dict(config))
    section = recipe.get("member_routing", {})
    model_config = section.get("model")
    selected_arm = str(arm or section.get("arm", ""))
    if section.get("schema") != CONFIG_SECTION_SCHEMA:
        raise ValueError(f"R13-6 requires {CONFIG_SECTION_SCHEMA}.")
    core_schema = model_config.get("schema") if isinstance(model_config, Mapping) else None
    variant = _VARIANTS.get(core_schema)
    if (variant is None or selected_arm != variant["arm"]
            or not isinstance(model_config, Mapping)
            or model_config.get("architecture") != selected_arm
            or model_config.get("address_mode") != "ME"):
        raise ValueError("R13-6 model config must match its strict ME candidate schema and arm.")

    manifest, data_path, manifest_path = _load_data_binding(
        recipe["dataset"].get("manifest_path"), recipe["dataset"].get("data_path"),
    )
    catalog_source, catalog = read_case_catalog(data_path)
    train_ids, validation_ids = _training_membership(recipe, catalog, manifest)
    normalizer = fit_global_normalizer(data_path, train_ids)
    training_cases = _read_selected_cases(
        data_path, train_ids, split="train", normalizer=normalizer, include_grid=False,
    )
    validation_cases = _read_selected_cases(
        data_path, validation_ids, split="test", normalizer=normalizer, include_grid=False,
    )
    heat_values = np.concatenate([
        np.asarray(case["structure"]["heat_powers"])[
            np.asarray(case["structure"]["module_present"]) > 0.5
        ] for case in training_cases
    ])
    heat_scale = max(float(np.sqrt(np.mean(heat_values.astype(np.float64) ** 2))), 1e-6)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(recipe["initialization"].get("seed", 0)))
        model = variant["adapter"](
            normalizer.stats, core_config=model_config, heat_scale=heat_scale,
        ).to(device)
    source_stat = data_path.stat()
    provider = ThermalR136MemberRoutingTask(
        model=model, training_cases=training_cases, validation_cases=validation_cases,
        manifest=manifest, data_path=data_path, recipe=recipe, device=device,
        manifest_path=manifest_path, catalog=catalog,
        source_binding={
            "dataset_path": str(data_path), "dataset_size_bytes": int(source_stat.st_size),
            "dataset_mtime_ns": int(source_stat.st_mtime_ns),
            "catalog_metadata_sha256": catalog_source["metadata_sha256"],
            "manifest_path": None if manifest_path is None else str(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "normalization_source": "TRAIN-only fit_global_normalizer",
            "dataset_copy_created": False, "persistent_cache_created": False,
        },
    )
    return model, provider


__all__ = [
    "CONFIG_SECTION_SCHEMA", "ThermalR136MemberRoutingTask",
    "build_thermal_r13_6_member_routing_task",
]
