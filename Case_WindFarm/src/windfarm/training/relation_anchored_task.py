"""Native five-role Wind training bridge for the relation-anchored adapter."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from honf_forward_core.models import relation_anchored_config
from honf_runtime.unified_training import OptimizerGroupSpec, ScheduleSpec

from ..relation_anchored import WindRelationAnchoredAdapter
from .content_hypergraph_task import (
    WindContentHypergraphTaskV2, WindContentPredictions, _build_wind_content_task,
)
from .joint_task import (
    _fit_or_load_role_scales, _read_role_scale_cache, _role_scale_cache_key, _sha256,
    _fit_or_load_formal_transforms, _read_transform_cache, _wind_dataset_fingerprint,
    fit_velocity_statistics,
)
from .unified_task import _fit_train_role_scales


def _load_replayed_full_train_transforms(*, existing_cache, **kwargs):
    """Reuse a moved TRAIN420 cache after replaying every TRAIN fit draw."""
    try:
        return _fit_or_load_formal_transforms(**kwargs)
    except FileNotFoundError:
        pass
    path = Path(existing_cache).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = payload["cache_key"]
    saved = _read_transform_cache(path, key)
    expected = {
        "dataset_profile": kwargs["profile_id"],
        "subset_manifest_sha256": str(kwargs["manifest"]["manifest_sha256"]),
        "training_fingerprint": str(kwargs["training_fingerprint"]),
        "train_row_indices": np.asarray(kwargs["train_rows"], dtype=np.int64).tolist(),
        "train_layout_indices": list(kwargs["train_layouts"]),
        "samples_per_row": 8192, "profile_bins": 32, "seed": int(kwargs["seed"]),
    }
    if saved is None or any(key.get(name) != value for name, value in expected.items()):
        raise ValueError("Saved Wind transforms differ from the original full TRAIN recipe.")
    normalizer, profile = saved
    actual_normalizer, actual_profile = fit_velocity_statistics(
        kwargs["view"].volume, kwargs["train_rows"], samples_per_row=8192,
        seed=int(kwargs["seed"]), profile_bins=32)
    if (actual_normalizer.to_dict() != normalizer.to_dict()
            or actual_profile.to_dict() != profile.to_dict()):
        raise ValueError("Saved Wind transforms do not reproduce on all TRAIN420 fit draws.")
    return normalizer, profile, {
        "mode": "explicit_saved_transform_exact_full_train_replay",
        "cache_file": str(path), "cache_file_sha256": _sha256(path),
        "saved_dataset_fingerprint": key["dataset_fingerprint"],
        "dataset_fingerprint": _wind_dataset_fingerprint(kwargs["view"]),
        "training_rows_sha256": key["train_row_indices_sha256"],
        "source_rows": len(kwargs["train_rows"]), "samples_per_row": 8192,
        "wind_test_target_values_read": False, "solver_attempts": 0,
        "scope": "Exact sampled transform fit on every TRAIN row; not whole-dataset identity.",
    }


def _load_panel_verified_role_scales(*, existing_cache, **kwargs):
    """Reuse an explicit saved calibration only after replaying its TRAIN panel.

    Dataset filesystem identity remains the current identity; equal sampled
    panel evidence does not assert identity of the full underlying dataset.
    """
    try:
        return _fit_or_load_role_scales(**kwargs)
    except FileNotFoundError:
        pass
    path = Path(existing_cache).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    old_key = payload["cache_key"]
    saved = _read_role_scale_cache(path, old_key)
    current_key = _role_scale_cache_key(
        view=kwargs["view"], manifest=kwargs["manifest"], train_rows=kwargs["train_rows"],
        train_layouts=kwargs["train_layouts"], profile_id=str(kwargs["manifest"]["subset_id"]),
        training_fingerprint=kwargs["training_fingerprint"], normalizer=kwargs["normalizer"],
        profile=kwargs["profile"], native_sampling_protocol=kwargs.get("native_sampling_protocol"))
    old_other = {k: v for k, v in old_key.items() if k != "dataset_fingerprint"}
    current_other = {k: v for k, v in current_key.items() if k != "dataset_fingerprint"}
    if saved is None or old_other != current_other:
        raise ValueError("Saved Wind calibration differs from the current TRAIN recipe or bindings.")
    actual = _fit_train_role_scales(
        kwargs["view"], kwargs["train_rows"], kwargs["train_layouts"], kwargs["profile"], kwargs["normalizer"],
        training_fingerprint=kwargs["training_fingerprint"], catalogue_cache=kwargs["catalogue_cache"])
    volatile = {"catalogue_build_or_lookup_seconds", "native_target_sampling_seconds", "catalogue_cache_after_calibration"}
    actual = {name: value for name, value in actual.items() if name not in volatile}
    if actual != saved:
        raise ValueError("Saved Wind calibration does not reproduce on its current TRAIN panel.")
    return saved, {
        "mode": "explicit_saved_calibration_exact_train_panel_replay",
        "cache_file": str(path), "cache_file_sha256": _sha256(path),
        "saved_dataset_fingerprint": old_key["dataset_fingerprint"],
        "dataset_fingerprint": current_key["dataset_fingerprint"],
        "all_other_cache_key_fields_equal": True, "all_nonvolatile_calibration_fields_equal": True,
        "scope": "deterministic TRAIN calibration panel only; not whole-dataset identity",
    }


class WindRelationAnchoredTask(WindContentHypergraphTaskV2):
    """Reuse native sampling and five-role losses with masked ragged batches."""

    def identity_payload(self):
        result = dict(self._native_identity)
        result.update(schema="wind_relation_anchored_provider_v1", family="relation_anchored", model_family="relation_anchored",
                      architecture="relation_anchored", model_config=self.model.core.get_config(),
                      adapter_state=self.model.state_payload(), recipe=self.recipe)
        result["environment_representation"] = dict(result["environment_representation"],
            token_shape=list(self.model.environment_shape), record_count=int(np.prod(self.model.environment_shape)),
            characteristic_length="support.extent_D / environment_shape, per axis")
        result["coordinate_frame"] = dict(result["coordinate_frame"],
            environment_characteristic_length="support.extent_D / environment_shape, per axis")
        result["training"] = dict(result["training"], total_epochs=self.total_epochs,
            microbatch_size=self.max_microbatch_cases, optimizer_schedule=vars(self._schedule()),
            sampling_dataset_id=self.sampling_dataset_id, scene_execution="batched with masked module padding")
        result["uncovered_train_module_count_strata"] = list(
            self.manifest.get("train_layout_selection", {}).get("uncovered_turbine_count_strata_by_limited_selection", ()))
        return result

    def serialization_payload(self):
        return {"schema": "wind_relation_anchored_inference_v1", "adapter_state": self.model.state_payload(),
                "data_binding": self.identity_payload()["normalizer_and_background_source"],
                "objective": self.identity_payload()["objective"]}

    def _schedule(self):
        return ScheduleSpec(peak_lr=3e-4, warmup_start_lr=3e-4, warmup_epochs=0,
                            hold_through_epoch=self.total_epochs - 1,
                            total_epochs=self.total_epochs, final_lr=3e-4)

    def optimizer_groups(self, model, arm, stage):
        if model is not self.model or arm != "relation_anchored" or stage != "joint":
            raise ValueError("Wind relation-anchored optimizer requires its own joint model.")
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        excluded = tuple(name for name in names if name.endswith(".bias") or "norm" in name.lower())
        decay = tuple(name for name in names if name not in set(excluded))
        return tuple(OptimizerGroupSpec(label, group, self._schedule(), weight_decay=weight_decay,
                                       betas=(0.9, 0.999), eps=1e-8)
                     for label, group, weight_decay in (("wind_decay", decay, 1e-5),
                                                        ("wind_no_decay", excluded, 0.0)) if group)

    def make_scene(self, scene_inputs):
        return scene_inputs

    def preparation_summary(self):
        result = dict(super().preparation_summary())
        result["environment_token_shape"] = list(self.model.environment_shape)
        return result

    def predict_native(self, model, scene, receivers, execution_mode="relation_anchored", phase="joint",
                       epoch=0, temperature=None):
        del epoch, temperature
        if model is not self.model or execution_mode != "relation_anchored" or phase not in ("joint", "development", "validation"):
            raise ValueError("Invalid Wind relation-anchored model or execution mode.")
        cases = [SimpleNamespace(index=item.row_index, case=item.case_name, layout_index=item.layout_index,
                wind_direction_deg=item.wind_direction_deg, module_centers=item.module_centers_D,
                module_features=item.module_features, module_present=item.module_present,
                global_context=item.operating_features,
                support=SimpleNamespace(lower_D=item.support_lower_D, extent_D=item.support_extent_D)) for item in scene]
        native = model.make_scene_batch(cases, module_ids=[item.module_ids for item in scene])
        coords = torch.as_tensor(np.array(receivers.coordinates_D, copy=True), device=self.device,
                                 dtype=next(model.parameters()).dtype)
        ids = torch.as_tensor(np.array(receivers.flat_indices, copy=True), device=self.device)
        query = model.make_query(native, coords, query_ids=ids)
        organization = model.core.prepare(native)
        output = model.core.decode(organization, query)
        work = {f"{prefix}_{name}": int(count)
                for prefix, counts in (("core_prepare", organization.work_counts),
                                       ("core_decode", output.work_counts))
                for name, count in (counts or {}).items()}
        velocity, baseline, delta = model.transforms.physical_velocity(output.joint_fields, query.coordinates)
        predictions = WindContentPredictions(velocity, output.joint_fields, delta, baseline,
            torch.as_tensor(np.array(receivers.role_ids, copy=True), device=self.device))
        return predictions, {"work": work}

    def work_counts(self, batch, predictions, auxiliary_state):
        del predictions
        return {**auxiliary_state["work"], "native_receivers": int(np.prod(batch.receivers.coordinates_D.shape[:2])),
                "active_physical_sources": sum(int(np.count_nonzero(item.module_present)) for item in batch.scene_inputs),
                "module_padding_slots": len(batch.scene_inputs) * max(len(item.module_present) for item in batch.scene_inputs)
                    - sum(len(item.module_present) for item in batch.scene_inputs),
                "environment_records": len(batch.scene_inputs) * int(np.prod(self.model.environment_shape)),
                "complete_native_prediction": 1}

    def training_state_dict(self):
        return {"training_samples": self.training_samples, "validation_samples": self.validation_samples}

    def load_training_state_dict(self, state):
        self.training_samples = int(state.get("training_samples", 0))
        self.validation_samples = int(state.get("validation_samples", 0))


def build_wind_relation_anchored_task(recipe, device):
    """Load the existing subset or original full split and build a fresh model."""
    value = dict(recipe)
    full = value.get("protocol") == "full_train1000"
    required = {"schema": "wind_relation_anchored_r15_2_v1", "dataset": "wind", "architecture": "relation_anchored",
                "protocol": "full_train1000" if full else "development", "seed": 42,
                "primary_queries": 4096, "effective_batch_size": 24,
                "total_epochs": 1000 if full else 200, "precision": "fp32", "optimizer_bias_norm_exclusion": True}
    if any(value.get(k) != v for k, v in required.items()):
        raise ValueError("Wind R15-2 recipe differs from its declared subset/full workload.")
    if value.get("catalogue_cache_ram_bytes") != (128 if full else 32) * 1024**3:
        raise ValueError("Wind requires its declared subset/full catalogue RAM bound.")
    shape = tuple(value["environment_shape"])
    if shape not in ((4, 4, 4), (8, 4, 2), (8, 8, 2), (8, 4, 4), (8, 8, 4)):
        raise ValueError("Unsupported Wind environment shape.")
    old = dict(value, schema="content_hypergraph_recipe_v2", architecture="A2-H6" if full else "A2-D",
               total_epochs=5000 if full else 2500,
               edge_access="full",
               model={"context_dim": 128, "num_heads": 4, "context_depth": 2, "num_hyperedges": 6,
                      "thermal_response_rank": 0, "local_hidden_dim": 128, "dropout": 0.0})
    if full:
        old["approved_stop_after"] = 1000
    existing_cache = value["existing_role_scale_cache"]
    _, provider = _build_wind_content_task(old, "cpu", schema_version=2,
        role_scale_loader=lambda **kwargs: _load_panel_verified_role_scales(existing_cache=existing_cache, **kwargs),
        transform_loader=(lambda **kwargs: _load_replayed_full_train_transforms(
            existing_cache=value["existing_transform_cache"], **kwargs)) if full else None)
    native_identity = provider.identity_payload()
    base = ContentHypergraphCoreV2(module_input_dim=2, environment_input_dim=7, coordinate_dim=3,
        query_input_dim=7, operating_input_dim=11, output_dim=3, architecture="A2-D", context_dim=128,
        num_heads=4, context_depth=2, local_hidden_dim=128, dropout=0.0, init_seed=42).get_config()
    config = relation_anchored_config("relation_anchored", base, input_dataset="wind",
                                      query_tile_size=int(value["query_tile_size"]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = WindRelationAnchoredAdapter(provider.adapter.normalizer, provider.adapter.background_profile,
                                            core_config=config, environment_shape=shape).to(device)
    provider.__class__ = WindRelationAnchoredTask
    provider._native_identity = native_identity
    provider.model, provider.recipe, provider.total_epochs, provider.device = model, value, int(value["total_epochs"]), torch.device(device)
    return model, provider
