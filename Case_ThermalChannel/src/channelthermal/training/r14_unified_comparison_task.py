"""Fresh Unified baseline with the R14 native-only Thermal sampling contract."""
from __future__ import annotations

import copy
from collections.abc import Mapping

import torch

from honf_runtime.unified_training import SamplingKey, TaskBatch
from honf_forward_core.continuous_joint.unified import UnifiedJointCore
from ..content_hypergraph import ThermalContentHypergraphAdapterV2
from ..data.datasets import fit_global_normalizer
from ..data.development_split import read_case_catalog
from .content_hypergraph_task import (
    ThermalContentPredictions, ThermalContentReceivers, ThermalContentSceneInputs, ThermalContentTargets,
)
from .contextual_joint_task import ThermalContextualJointTask
from .joint_task import _load_data_binding, _read_selected_cases
from .member_routing_task import ThermalInteractionReusedMemberRoutingTask, _training_membership
from .unified_task import _sample_primary, _structure_only

WORKFLOW = "thermal_r14_unified_comparison_v1"
PROVIDER_SCHEMA = "thermal_r14_unified_comparison_provider_v1"
TRAINING_STATE_SCHEMA = "thermal_r14_unified_comparison_training_state_v1"


class ThermalR14UnifiedComparisonTask(ThermalInteractionReusedMemberRoutingTask):
    """Use the native objective and case streams with the historical Unified model."""

    training_state_schema = TRAINING_STATE_SCHEMA
    joint_work_objective_enabled = False

    def __init__(self, **kwargs):
        recipe = copy.deepcopy(kwargs["recipe"])
        # Anchored health diagnostics expect anchored directories, not Unified.
        recipe.setdefault("member_routing", {}).pop("health", None)
        kwargs["recipe"] = recipe
        super().__init__(**kwargs)
        self.workflow = WORKFLOW
        self.provider_schema = PROVIDER_SCHEMA
        self.training_sampling_arm = "R13-3-Relative"

    def identity_payload(self):
        payload = ThermalContextualJointTask.identity_payload(self)
        payload.update({
            "schema": self.provider_schema, "workflow": self.workflow,
            "core_schema": self.model.core.schema,
            "training_sampling_arm": self.training_sampling_arm,
            "native_only_training": True,
            "response_training_family_ids": [], "response_supervision": None,
            "prescribed_heat_in_context": False,
            "objective": {
                "native_weights": copy.deepcopy(self.native_weights),
                "native_weights_fixed": True,
                "joint_fields": ["u", "v", "p", "temperature"],
                "affine_temperature_constraint": True,
                "heat_independent_flow_constraint": True,
                "response_training_families": [], "response_fit": False,
                "organizer_objective": False,
                "joint_work_objective": False,
            },
            "model_clock": "epoch is 3000 plus training epoch; macro clock receives within-epoch update index",
            "initialization": "fresh seed 0; configuration identity only; no reference weights",
            "recipe": copy.deepcopy(self.recipe),
        })
        return payload

    def make_batch(self, case_keys, key):
        if (key.sampling_version != SamplingKey.CASE_EPOCH_VERSION
                or key.sampling_version != self.sampling_version
                or key.dataset_id != self.sampling_dataset_id or key.epoch < 1):
            raise ValueError("Unified comparison requires the common case-epoch Thermal sampler identity.")
        self.model.core.set_training_clock(3000 + key.epoch, key.update_index)
        indices = tuple(self._train_index_by_id[str(v)] for v in case_keys)
        sampled, sampling = _sample_primary(
            self.training_cases, indices, epoch=key.epoch, key=key, training=True,
            budget={"fluid_queries": self.primary_queries, "material_queries_per_module": 32,
                    "surface_stride": 4}, device=self.device,
        )
        case_ids = tuple(str(v) for v in case_keys)
        self.training_samples += len(indices)
        return TaskBatch(
            scene_inputs=ThermalContentSceneInputs(_structure_only(sampled["structure"]), case_ids=case_ids),
            receivers=ThermalContentReceivers(sampled["query_xy"], sampled["module_internal_query_points"],
                                              sampled["structure"]["heat_powers"], ntheta=16),
            targets=ThermalContentTargets(case_ids, indices, key.epoch, sampled["field_targets"],
                sampled["point_weights"], sampled["interface_target"],
                sampled["interface_condition_valid_mask"], sampled["module_internal_temperature_points"]),
            auxiliary={"query_sampling_sha256": sampling}, case_keys=case_ids,
        )

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, temperature=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported Unified native prediction options: {sorted(kwargs)}")
        if model is not self.model or execution_mode != model.architecture:
            raise ValueError("Wrong Unified comparison model/arm.")
        model.core.joint_begin_microbatch()
        prepared = model.prepare_native(scene.structure, receivers.fluid_xy,
            local_query_points=receivers.local_query_points, ntheta=receivers.ntheta, scene_key=scene.case_ids)
        native, counts = model.apply_native_with_work_counts(prepared, receivers.heat)
        work = {"native_fluid_queries": int(receivers.fluid_xy.shape[0] * receivers.fluid_xy.shape[1]),
                "native_union_rows": int(prepared.union.unique_counts.sum()),
                "native_local_pairs": int(prepared.query_batch.local_weights.numel())}
        work.update({f"core_prepare_primary_{k}": int(v) for k, v in prepared.organization.work_counts.items()})
        work.update({f"core_decode_primary_{k}": int(v) for k, v in counts.items()})
        return ThermalContentPredictions(prepared, native, work=work), {"work": work}


def build_r14_unified_comparison_task(config: Mapping, device):
    """Return a fresh model and provider; the model mapping supplies identity, never weights."""
    recipe = copy.deepcopy(dict(config))
    core_config = recipe["member_routing"]["model"]
    if (core_config.get("schema") != UnifiedJointCore.schema
            or core_config.get("routing_mode") != "hypergraph"):
        raise ValueError("Unified comparison requires the literal unified hypergraph configuration.")
    UnifiedJointCore.validate_config(core_config)
    initialization = recipe["initialization"]
    if (initialization.get("seed") != 0 or initialization.get("pretrained_weights") is not False
            or initialization.get("source_checkpoint") is not None):
        raise ValueError("Unified comparison requires a fresh seed-0 model and no checkpoint weights.")
    manifest, data_path, manifest_path = _load_data_binding(
        recipe["dataset"].get("manifest_path"), recipe["dataset"].get("data_path"))
    catalog_source, catalog = read_case_catalog(data_path)
    train_ids, validation_ids = _training_membership(recipe, catalog, manifest)
    normalizer = fit_global_normalizer(data_path, train_ids)
    train = _read_selected_cases(data_path, train_ids, split="train", normalizer=normalizer, include_grid=False)
    validation = _read_selected_cases(data_path, validation_ids, split="test", normalizer=normalizer, include_grid=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = ThermalContentHypergraphAdapterV2(normalizer.stats, core_config=core_config).to(device)
    stat = data_path.stat()
    provider = ThermalR14UnifiedComparisonTask(
        model=model, training_cases=train, validation_cases=validation, manifest=manifest, data_path=data_path,
        recipe=recipe, device=device, manifest_path=manifest_path, catalog=catalog,
        source_binding={"dataset_path": str(data_path), "dataset_size_bytes": int(stat.st_size),
            "dataset_mtime_ns": int(stat.st_mtime_ns), "catalog_metadata_sha256": catalog_source["metadata_sha256"],
            "manifest_path": None if manifest_path is None else str(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"], "normalization_source": "TRAIN-only fit_global_normalizer",
            "dataset_copy_created": False, "persistent_cache_created": False},
    )
    return model, provider
