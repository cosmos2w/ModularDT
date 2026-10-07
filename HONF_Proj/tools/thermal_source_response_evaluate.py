#!/usr/bin/env python3
"""Evaluate source-response models and replay retained classics on saved data.

Classic bridge mode uses each checkpoint's strict native loader, TRAIN
normalization, predicted-port path and local surrogate. No mode launches a
reference solver or fits a model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter, time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "Case_ThermalChannel/src", ROOT / "tools"):
    sys.path.insert(0, str(source))

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
from channelthermal.data.development_split import development_case_ids, resolve_development_manifest
from channelthermal.source_response import FORMAL_RESPONSE_PROFILE_NAMES
from channelthermal.training.checkpoints import _file_sha256
from thermal_campaign_evaluate import aggregate_physical, physical_case_metrics
from thermal_development import validate_generated_output
from thermal_formal_profile import bind_formal_validation, bind_original_train
from thermal_response_refinement_evaluation import (
    finite_metrics,
    load_atlas_families,
    load_counted_families,
    native_functionals,
    stored_pool_decision,
    summarize_families,
    validate_heat_records,
)

from honf_runtime.compat import resolve_demo_path
from honf_runtime.run_store import atomic_write_json

FIXED4 = ("0277", "0291", "0294", "0687")
CHANNELS = ("u", "v", "p", "omega", "temperature")
FIXED25_MANIFEST_SHA256 = "933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044"
CLASSIC_IDENTITIES = {
    "Run1401": {
        "epoch": 4585,
        "sha256": "5be150bd6b4fc79599af62c767fba84490ba50edcc8cc8ce85026ae27a1846b3",
    },
    # This identity is the literal root epoch_5000_model.pt milestone; its
    # exact digest distinguishes it from the separate checkpoints/latest.pt.
    "Run1401_e5000_latest": {
        "epoch": 5000,
        "sha256": "cee978f0461db928b72c3b6b66cb0ef56647674f82d6ef2660a8380754a829d6",
    },
    "Run1804": {
        "epoch": 4738,
        "sha256": "71ed480ff0396813491c650dd11d887195174019b373fbd9a1fb25505142c066",
    },
    "Run1804_e5000_latest": {
        "epoch": 5000,
        "sha256": "9d0b83c562cecc2ffc52c3a08c993dfa8f47ae0e769f54ed6ffdd966047d63d9",
    },
    "Run1502": {
        "epoch": 4794,
        "sha256": "08d86a573c7f7d86463bde231eb9f84a745602fca2b8f97142f39540d33a85bb",
    },
    "Run1502_e5000_latest": {
        "epoch": 5000,
        "sha256": "20af85200796f639039d4853425fe91ee76e19e7caf5b5b68863badfe7690895",
    },
}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def numpy_value(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def classic_input_sample_for_heat(normalized_sample, raw_sample, heat, normalizer, *, normalize_inputs):
    """Replace only known physical heating, retaining the checkpoint's input transform."""
    sample = copy.deepcopy(normalized_sample)
    present = np.asarray(raw_sample["structure"]["module_present"], dtype=np.float32) > .5
    heat = np.asarray(heat, dtype=np.float32)
    if heat.shape != (int(present.sum()),) or not np.isfinite(heat).all():
        raise ValueError("Counted heating must provide one finite value per active physical source")
    slot_heat = np.zeros_like(np.asarray(raw_sample["structure"]["heat_powers"], dtype=np.float32))
    slot_heat[present] = heat
    sample["structure"]["heat_powers"] = normalizer.normalize_heat_power(slot_heat) if normalize_inputs else slot_heat
    local_params = np.array(sample["local_module_params"], dtype=np.float32, copy=True)
    if local_params.shape[0] != slot_heat.shape[0] or local_params.shape[-1] < 3:
        raise ValueError("Historical local-surrogate parameter catalogue does not match physical sources")
    local_params[:, 0] = slot_heat
    local_params[~present] = 0
    sample["local_module_params"] = local_params
    return sample


def load_native_cases(checkpoint, split="test"):
    config = checkpoint["train_config"]["dataset"]
    path = resolve_demo_path(config["packed_h5_path"])
    manifest = resolve_development_manifest(config, path)
    ids = development_case_ids(manifest, split)
    if len(ids) != (22 if split == "test" else 150):
        raise ValueError("Response evaluation requires fixed25_v1 150/22.")
    if manifest["manifest_sha256"] != "933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044":
        raise ValueError("Response evaluation cannot change the declared development membership.")
    normalizer = H5Normalizer(checkpoint["global_normalization_stats"])
    dataset = GlobalChannelThermalDataset(path, split=split, points_per_case=1,
        random_point_sampling=False, include_grid=True, normalizer=normalizer,
        normalize_inputs=False, normalize_targets=False, case_ids=ids)
    return dataset, manifest, path


def validate_classic_selection(classic_id, checkpoint, checkpoint_sha256):
    """Require the selected historic checkpoint identity recorded in the report."""
    if classic_id not in CLASSIC_IDENTITIES:
        raise ValueError(f"Unknown historical comparison identity: {classic_id!r}")
    expected = CLASSIC_IDENTITIES[classic_id]
    epoch = checkpoint.get("epoch", checkpoint.get("current_epoch"))
    if epoch != expected["epoch"]:
        raise ValueError(f"{classic_id} selected checkpoint must be e{expected['epoch']}, got {epoch!r}")
    if checkpoint_sha256 != expected["sha256"]:
        raise ValueError(f"{classic_id} selected checkpoint SHA-256 does not match the retained identity")
    return {"classic_id": classic_id, "epoch": expected["epoch"], "sha256": expected["sha256"]}


def load_classic_native_cases(checkpoint, manifest_path):
    """Load the shared DEV22 with the classic checkpoint's own input transform."""
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from thermal_development import evaluation_dataset_kwargs, resolve_evaluation_manifest

    from honf_runtime.compat import resolve_demo_path

    dataset_config = checkpoint.get("train_config", {}).get("dataset", {})
    if not dataset_config.get("packed_h5_path"):
        raise ValueError("Historical checkpoint has no packed dataset binding")
    path = resolve_demo_path(dataset_config["packed_h5_path"])
    manifest = resolve_evaluation_manifest(dataset_config, path,
        scope="development", manifest_path=Path(manifest_path))
    if manifest["manifest_sha256"] != FIXED25_MANIFEST_SHA256:
        raise ValueError("Historical bridge requires the retained fixed25_v1 membership")
    stats = checkpoint.get("global_normalization_stats", {})
    if not stats:
        raise ValueError("Historical bridge requires the checkpoint's native TRAIN normalization")
    normalizer = H5Normalizer({key: np.asarray(value, dtype=np.float32) for key, value in stats.items()})
    selection = evaluation_dataset_kwargs(manifest, "test")
    normalized = GlobalChannelThermalDataset(path, split="test", points_per_case=1,
        random_point_sampling=False, include_grid=True, normalizer=normalizer,
        normalize_inputs=bool(dataset_config.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_config.get("normalize_targets", False)), **selection)
    raw = GlobalChannelThermalDataset(path, split="test", points_per_case=1,
        random_point_sampling=False, include_grid=True, normalizer=normalizer, **selection)
    if normalized.selected_case_ids != raw.selected_case_ids:
        raise ValueError("Normalized and physical DEV22 case IDs differ")
    case_ids = tuple(str(value) for value in raw.selected_case_ids)
    if len(case_ids) != 22 or len(set(case_ids)) != 22:
        raise ValueError(f"Historical bridge expected 22 unique fixed25_v1 validation cases, got {len(case_ids)}")
    if not set(FIXED4).issubset(case_ids):
        raise ValueError("The fixed physical-field representative panel is absent from DEV22")
    return normalized, raw, manifest, path


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def formal_panel_case_ids(validation_binding, panel):
    """Return the sealed canonical89 or compatibility90 IDs in source order."""
    if panel not in ("canonical89", "original90"):
        raise ValueError(f"Unknown formal validation panel: {panel!r}")
    primary = validation_binding.get("primary_case_ids")
    compatibility = validation_binding.get("compatibility_case_ids")
    duplicate = validation_binding.get("excluded_training_duplicate_case_id")
    if (validation_binding.get("primary_scope") != "original_test_excluding_train_duplicate"
            or validation_binding.get("compatibility_scope") != "original_test_all_rows"
            or duplicate != "0273"
            or not isinstance(primary, list) or not isinstance(compatibility, list)
            or len(primary) != 89 or len(compatibility) != 90
            or len(set(primary)) != len(primary) or len(set(compatibility)) != len(compatibility)
            or duplicate in primary or duplicate not in compatibility
            or primary != [case_id for case_id in compatibility if case_id != duplicate]
            or validation_binding.get("primary_case_count") != 89
            or validation_binding.get("compatibility_case_count") != 90
            or validation_binding.get("primary_case_ids_sha256") != _canonical_sha256(primary)
            or validation_binding.get("compatibility_case_ids_sha256") != _canonical_sha256(compatibility)):
        raise ValueError("Formal checkpoint does not seal the expected canonical89/original90 panel memberships.")
    return list(primary if panel == "canonical89" else compatibility)


def validate_formal_evaluation_checkpoint(checkpoint, *, panel):
    """Rebind a trusted, final R-direct checkpoint to the current packed-H5 catalog."""
    if (checkpoint.get("formal_workflow_scope") != "formal_full_train_v1"
            or bool(checkpoint.get("startup_benchmark"))
            or int(checkpoint.get("epoch", 0)) != 5000):
        raise ValueError("Formal 89/90 evaluation requires a non-startup R-direct e5000 checkpoint.")
    recipe = checkpoint.get("fit_identity", {}).get("recipe", {})
    profile = recipe.get("profile", {})
    profile_data = profile.get("data", {})
    validation_profile = profile_data.get("formal_validation", {})
    if (recipe.get("identity") != "thermal_source_response_r_direct_formal5000_v1"
            or recipe.get("preferred_response_family") != "R-direct"
            or profile.get("profile_name") not in FORMAL_RESPONSE_PROFILE_NAMES
            or profile_data.get("expected_train_case_count") != 600
            or validation_profile.get("primary_scope") != "original_test_excluding_train_duplicate"
            or validation_profile.get("compatibility_scope") != "original_test_all_rows"
            or validation_profile.get("expected_primary_case_count") != 89
            or validation_profile.get("expected_compatibility_case_count") != 90
            or validation_profile.get("excluded_training_duplicate_case_id") != "0273"):
        raise ValueError("Checkpoint does not declare the maintained full-TRAIN R-direct evaluation profile.")
    data_binding = checkpoint.get("formal_dataset_binding", {})
    normalization_binding = checkpoint.get("formal_normalization_binding", {})
    validation_binding = checkpoint.get("formal_validation_binding", {})
    dataset_config = checkpoint.get("train_config", {}).get("dataset", {})
    configured_path = dataset_config.get("packed_h5_path")
    if not configured_path:
        raise ValueError("Formal checkpoint has no packed-H5 input binding.")
    path = resolve_demo_path(configured_path).resolve()
    if (data_binding.get("dataset_path") != str(path)
            or dataset_config.get("formal_dataset_binding") != data_binding
            or dataset_config.get("formal_normalization_binding") != normalization_binding
            or dataset_config.get("formal_validation_binding") != validation_binding
            or validation_binding.get("source_metadata_sha256") != data_binding.get("source_metadata_sha256")):
        raise ValueError("Formal checkpoint input and validation bindings are inconsistent.")
    actual_data_binding, _ = bind_original_train(
        path, expected_count=profile_data["expected_train_case_count"],
        dataset_id=data_binding.get("dataset_id"))
    actual_validation_binding, _, _ = bind_formal_validation(
        path, expected_primary_count=validation_profile["expected_primary_case_count"],
        expected_compatibility_count=validation_profile["expected_compatibility_case_count"],
        duplicate_case_id=validation_profile["excluded_training_duplicate_case_id"])
    if actual_data_binding != data_binding:
        raise ValueError("Current packed-H5 TRAIN inputs/metadata differ from the formal checkpoint binding.")
    if actual_validation_binding != validation_binding:
        raise ValueError("Current packed-H5 validation inputs/metadata differ from the formal 89/90 binding.")
    ids = formal_panel_case_ids(validation_binding, panel)
    return {
        "panel": panel,
        "case_ids": ids,
        "case_count": len(ids),
        "case_ids_sha256": _canonical_sha256(ids),
        "formal_dataset_binding_sha256": _canonical_sha256(data_binding),
        "formal_normalization_binding_sha256": _canonical_sha256(normalization_binding),
        "formal_validation_binding_sha256": _canonical_sha256(validation_binding),
        "source_metadata_sha256": actual_data_binding["source_metadata_sha256"],
        "dataset_path": str(path),
        "duplicate_case_id": validation_profile["excluded_training_duplicate_case_id"],
    }


def load_formal_native_cases(checkpoint, panel):
    """Load physical formal-panel cases with the R-direct full-TRAIN transform."""
    scope = validate_formal_evaluation_checkpoint(checkpoint, panel=panel)
    normalizer = H5Normalizer(checkpoint["global_normalization_stats"])
    dataset = GlobalChannelThermalDataset(scope["dataset_path"], split="test", points_per_case=1,
        random_point_sampling=False, include_grid=True, normalizer=normalizer,
        normalize_inputs=False, normalize_targets=False, case_ids=scope["case_ids"])
    if [str(value) for value in dataset.selected_case_ids] != scope["case_ids"]:
        raise ValueError("Formal physical reader case order differs from the sealed checkpoint panel.")
    return dataset, scope


def load_classic_formal_native_cases(checkpoint, formal_scope):
    """Read the same physical receivers with a retained classic's own transforms."""
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer

    from honf_runtime.compat import resolve_demo_path

    dataset_config = checkpoint.get("train_config", {}).get("dataset", {})
    configured_path = dataset_config.get("packed_h5_path")
    if not configured_path:
        raise ValueError("Historical checkpoint has no packed-H5 dataset binding.")
    path = resolve_demo_path(configured_path).resolve()
    if str(path) != formal_scope["dataset_path"]:
        raise ValueError("Historical checkpoint and formal reference do not use the same packed H5.")
    stats = checkpoint.get("global_normalization_stats", {})
    if not stats:
        raise ValueError("Historical checkpoint has no checkpoint-native TRAIN normalization.")
    for name, values in stats.items():
        if not np.isfinite(np.asarray(values)).all():
            raise ValueError(f"Historical checkpoint has non-finite normalization values in {name}.")
    normalizer = H5Normalizer({key: np.asarray(value, dtype=np.float32) for key, value in stats.items()})
    selection = {"case_ids": formal_scope["case_ids"]}
    normalized = GlobalChannelThermalDataset(path, split="test", points_per_case=1,
        random_point_sampling=False, include_grid=True, normalizer=normalizer,
        normalize_inputs=bool(dataset_config.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_config.get("normalize_targets", False)), **selection)
    raw = GlobalChannelThermalDataset(path, split="test", points_per_case=1,
        random_point_sampling=False, include_grid=True, normalizer=normalizer,
        normalize_inputs=False, normalize_targets=False, **selection)
    if normalized.selected_case_ids != raw.selected_case_ids:
        raise ValueError("Normalized and physical formal receiver IDs differ for the historical checkpoint.")
    if [str(value) for value in raw.selected_case_ids] != formal_scope["case_ids"]:
        raise ValueError("Historical physical reader case order differs from the formal receiver panel.")
    return normalized, raw, path


def _attach_formal_scope(summary, formal_scope):
    panel = formal_scope["panel"]
    summary.update({
        "dataset_scope": "formal full-TRAIN recipe validation: canonical89 primary" if panel == "canonical89"
            else "formal full-TRAIN recipe compatibility validation: original90 including the TRAIN duplicate",
        "formal_validation_panel": panel,
        "formal_validation_case_ids": formal_scope["case_ids"],
        "formal_validation_case_count": formal_scope["case_count"],
        "formal_validation_case_ids_sha256": formal_scope["case_ids_sha256"],
        "formal_dataset_binding_sha256": formal_scope["formal_dataset_binding_sha256"],
        "formal_normalization_binding_sha256": formal_scope["formal_normalization_binding_sha256"],
        "formal_validation_binding_sha256": formal_scope["formal_validation_binding_sha256"],
        "source_metadata_sha256": formal_scope["source_metadata_sha256"],
        "excluded_training_duplicate_case_id": formal_scope["duplicate_case_id"],
        "physical_reference_scope": "saved analytic-wake/shared-grid packed-H5 references; no new solver calls",
    })


def formal_physical_aggregates(rows, panel):
    """Keep the content duplicate out of primary errors for either model family."""
    if panel not in ("canonical89", "original90"):
        raise ValueError("Unknown formal physical aggregation panel.")
    primary = [row for row in rows if row["case_id"] != "0273"]
    result = {"primary_excluding_0273": aggregate_physical(primary)}
    if panel == "original90":
        result["compatibility_including_0273"] = aggregate_physical(rows)
    return result


def classic_field_evaluation(model, checkpoint, output, device, *, classic_id, manifest_path=None,
                             formal_scope=None):
    """Replay one retained classic through its native predicted-port/local path."""
    from channelthermal.evaluation.prepared import predict_case
    from channelthermal.evaluation.results import denormalize_predictions

    if formal_scope is None:
        normalized, raw, manifest, path = load_classic_native_cases(checkpoint, manifest_path)
        panel_summary = {
            "dataset_scope": "fixed25_v1 exposed development validation",
            "development_manifest_sha256": manifest["manifest_sha256"],
            "development_case_ids": [str(value) for value in raw.selected_case_ids],
            "scope": "22 repeatedly exposed fixed25_v1 validation cases; stored analytic-wake/shared-grid benchmark units",
        }
    else:
        normalized, raw, path = load_classic_formal_native_cases(checkpoint, formal_scope)
        panel_summary = {
            "formal_validation_panel": formal_scope["panel"],
            "formal_validation_case_ids": formal_scope["case_ids"],
            "formal_validation_case_count": formal_scope["case_count"],
            "formal_validation_case_ids_sha256": formal_scope["case_ids_sha256"],
            "formal_validation_binding_sha256": formal_scope["formal_validation_binding_sha256"],
            "source_metadata_sha256": formal_scope["source_metadata_sha256"],
            "excluded_training_duplicate_case_id": formal_scope["duplicate_case_id"],
            "dataset_scope": "formal full-TRAIN recipe validation",
            "scope": "same sealed original-test physical receivers as the formal R-direct reference; classic-native normalization and predicted-port/local-surrogate calls",
            "physical_reference_scope": "saved analytic-wake/shared-grid packed-H5 references; no new solver calls",
        }
    dataset_config = checkpoint.get("train_config", {}).get("dataset", {})
    summary = {
        "model": classic_id,
        "checkpoint_epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch"))),
        "dataset": str(path),
        "split": "test",
        "input_normalization": "checkpoint-native global TRAIN statistics and normalize_inputs setting",
        "target_normalization": "checkpoint-native global TRAIN statistics, denormalized before physical metrics",
        "port_mode": "native predicted-port trajectory with complete embedded or configured local surrogate",
        "not_applicable_roles": {},
        "rows": [],
        "solver_attempts": 0,
        "optimizer_updates": 0,
        "native_classic_calls": 0,
        **panel_summary,
    }
    if formal_scope is not None:
        _attach_formal_scope(summary, formal_scope)
    for index in range(len(raw)):
        normalized_sample, reference = normalized[index], raw[index]
        case_id = str(reference["case_id"])
        synchronize(device)
        started = perf_counter()
        with torch.no_grad():
            prediction = predict_case(model, normalized_sample, device,
                query_batch_size=1024, local_port_condition_mode="predicted", mixed_teacher_ratio=0)
        synchronize(device)
        elapsed = perf_counter() - started
        prediction = denormalize_predictions(prediction, normalized,
            bool(dataset_config.get("normalize_targets", False)))
        prediction = {key: numpy_value(value) for key, value in prediction.items()}
        metrics, masks = physical_case_metrics(reference, prediction, CHANNELS)
        module_count = int(numpy_value(reference["structure"]["module_present"]).sum())
        summary["rows"].append({"case_id": case_id, "intervention": "normal",
            "module_count": module_count, "metrics": metrics,
            "complete_native_predicted_port_local_call_seconds": elapsed})
        summary["native_classic_calls"] += 1
        if case_id in FIXED4:
            evidence = {
                "reference_field": numpy_value(reference["steady_field"]),
                "prediction_field": numpy_value(prediction["pred_field_grid"]),
                "residual_field": numpy_value(prediction["pred_field_grid"])
                    - numpy_value(reference["steady_field"])[..., :len(CHANNELS)],
                "fluid_mask": masks["fluid_mask"],
                "near_mask": masks["near_mask"],
                "far_mask": masks["far_mask"],
                "x_grid": numpy_value(reference["x_grid"]),
                "y_grid": numpy_value(reference["y_grid"]),
                "module_centers": numpy_value(reference["structure"]["module_centers"]),
                "module_present": numpy_value(reference["structure"]["module_present"]),
                "heat_powers": numpy_value(reference["structure"]["heat_powers"]),
                "material_params": numpy_value(reference["structure"]["material_params"]),
                "reference_internal_temperature": numpy_value(reference["module_internal_temperature_points"]),
                "prediction_internal_temperature": prediction["pred_internal_temperature"],
                "reference_interface": numpy_value(reference["interface_target"]),
                "prediction_interface": prediction["pred_interface"],
                "reference_ports": numpy_value(reference["teacher_port_tokens"]),
                "prediction_port_condition": prediction["pred_port_condition"],
                "prediction_port_condition_raw": prediction["pred_port_condition_raw"],
                **{f"{key}_mask": value for key, value in masks.items()},
            }
            np.savez_compressed(output / f"{case_id}_fields.npz", **evidence)
        print(json.dumps({"model": classic_id, "case_id": case_id,
            "fluid_temperature_rmse": metrics["fluid/temperature"]["rmse"],
            "seconds": elapsed}), flush=True)
    summary["equal_case_metrics"] = aggregate_physical(summary["rows"])
    if formal_scope is not None:
        summary.update(formal_physical_aggregates(summary["rows"], formal_scope["panel"]))
    summary["physical_eight_rows"] = {
        key: summary["equal_case_metrics"][key]
        for key in ("fluid/u", "fluid/v", "fluid/p", "fluid/omega", "fluid/temperature",
                    "surface_temperature", "material_temperature", "module_material_peak")
    }
    summary["module_count_strata"] = {
        f"M{count}": aggregate_physical([row for row in summary["rows"] if row["module_count"] == count])
        for count in sorted({row["module_count"] for row in summary["rows"]})
    }
    summary["timing_scope"] = (
        "One unsmoothed evaluation call per case after H5 sample retrieval; includes the full predicted-port/local-surrogate path and output copy, excludes checkpoint loading and dataset I/O."
    )
    return summary


def classic_counted_response_evaluation(model, checkpoint, output, device, *, classic_id,
                                        manifest_path, request_path, records_dir):
    """Replay only the retained counted heat states through a classic's native path.

    Geometry and receiver coordinates must join exactly to fixed25_v1. Heating is
    replaced by each saved physical input and normalized with this checkpoint's
    own TRAIN statistics. The evaluator does not substitute physical port targets
    into the predicted-port path and does not create a missing 0277 baseline.
    """
    from channelthermal.evaluation.prepared import predict_case
    from channelthermal.evaluation.results import denormalize_predictions

    normalized, raw, manifest, dataset_path = load_classic_native_cases(checkpoint, manifest_path)
    families, _ = load_counted_families(request_path, records_dir)
    if tuple(case_id for case_id, _ in families) != FIXED4:
        raise ValueError("Historical response replay requires the exact fixed4 counted family order")
    by_id = {str(raw[index]["case_id"]): index for index in range(len(raw))}
    if not set(FIXED4).issubset(by_id):
        raise ValueError("fixed25_v1 DEV22 does not contain every counted response layout")
    dataset_config = checkpoint.get("train_config", {}).get("dataset", {})
    normalize_inputs = bool(dataset_config.get("normalize_inputs", False))
    normalize_targets = bool(dataset_config.get("normalize_targets", False))
    summary = {
        "model": classic_id,
        "dataset": str(dataset_path),
        "split": "test",
        "dataset_scope": "fixed25_v1 exposed development validation; same retained counted physical receivers",
        "development_manifest_sha256": manifest["manifest_sha256"],
        "normalization": "checkpoint-native TRAIN input and target transforms; physical outputs denormalized before differencing",
        "port_mode": "native predicted-port trajectory with complete embedded or configured local surrogate",
        "physical_reference": "existing counted analytic-wake/shared-grid records; not CFD",
        "families": [], "primary_baseline_relative_state_count": 0,
        "secondary_0277_endpoint_state_count": 0,
        "solver_attempts": 0, "optimizer_updates": 0,
        "native_classic_calls": 0,
    }
    for case_id, records in families:
        validate_heat_records(records)
        if case_id not in FIXED4:
            raise ValueError(f"Unexpected counted response case {case_id}")
        index = by_id[case_id]
        norm_sample, raw_sample = normalized[index], raw[index]
        present = np.asarray(raw_sample["structure"]["module_present"]) > .5
        centers = np.asarray(raw_sample["structure"]["module_centers"], dtype=np.float64)[present]
        first = next(iter(records.values()))
        source_ids = first.design.active_module_ids
        record_centers = np.asarray([module.position_xy for module in first.design.modules], dtype=np.float64)
        if tuple(source_ids) != tuple(f"{case_id}:module:{slot}" for slot in range(len(source_ids))):
            raise ValueError(f"Counted source order does not match historical physical slot order for {case_id}")
        if centers.shape != record_centers.shape or not np.array_equal(centers, record_centers):
            raise ValueError(f"Historical DEV input geometry differs from counted receiver geometry for {case_id}")
        context = first.context.values
        material = np.asarray(raw_sample["structure"]["material_params"], dtype=np.float64)
        expected_material = np.asarray([context[key] for key in
            ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius")])
        if not np.array_equal(material, expected_material):
            raise ValueError(f"Historical DEV material context differs from counted records for {case_id}")
        fluid_role = first.output.roles["fluid_fields"]
        grid_xy = np.stack((np.asarray(raw_sample["x_grid"]).reshape(-1),
                            np.asarray(raw_sample["y_grid"]).reshape(-1)), -1)
        if not np.array_equal(np.asarray(fluid_role.query_features), grid_xy):
            raise ValueError(f"Historical DEV fluid grid differs from counted query ordering for {case_id}")
        m = len(source_ids)
        ntheta = first.output.roles["interface"].query_features.shape[0] // m
        expected_theta = np.tile(np.arange(ntheta, dtype=np.float64) * (2 * np.pi / ntheta), m)
        if not np.allclose(first.output.roles["interface"].query_features[:, 0], expected_theta, rtol=0, atol=5e-7):
            raise ValueError(f"Counted port-angle ordering differs from historical fixed-angle path for {case_id}")
        local_points = np.asarray(norm_sample["module_internal_query_points"])
        material_role = first.output.roles["solid_temperature"]
        receivers = np.asarray(material_role.receiver_module_ids)
        if local_points.ndim != 2 or receivers.shape != (material_role.query_features.shape[0],):
            raise ValueError(f"Historical DEV material query catalogue is not aligned for {case_id}")
        for slot, source_id in enumerate(source_ids):
            rows = np.flatnonzero(receivers == source_id)
            if rows.size != local_points.shape[0] or not np.allclose(material_role.query_features[rows], local_points, rtol=0, atol=5e-8):
                raise ValueError(f"Historical DEV material receivers differ for {case_id}/{source_id}")

        predictions, physical_references, absolute = {}, {}, []
        for label, record in records.items():
            physical_heat = np.asarray([module.heating for module in record.design.modules], dtype=np.float32)
            state_sample = classic_input_sample_for_heat(norm_sample, raw_sample, physical_heat,
                normalized.normalizer, normalize_inputs=normalize_inputs)
            synchronize(device)
            call_started = perf_counter()
            with torch.no_grad():
                prediction = predict_case(model, state_sample, device,
                    query_batch_size=1024, local_port_condition_mode="predicted", mixed_teacher_ratio=0)
            synchronize(device)
            elapsed = perf_counter() - call_started
            prediction = denormalize_predictions(prediction, normalized, normalize_targets)
            pred_field = np.asarray(prediction["pred_field_grid"])
            if pred_field.reshape(-1, pred_field.shape[-1]).shape != fluid_role.values.shape:
                raise ValueError(f"Historical predicted field shape differs from saved receiver role for {case_id}")
            predicted_roles = {
                "fluid_fields": pred_field.reshape(-1, pred_field.shape[-1]),
                "interface": np.asarray(prediction["pred_interface"])[:m].reshape(-1, 2),
                "solid_temperature": np.asarray(prediction["pred_internal_temperature"])[:m].reshape(-1, 1),
            }
            expected_rows = {name: role.values.shape for name, role in record.output.roles.items()}
            for role_name, values in predicted_roles.items():
                if values.shape != expected_rows[role_name]:
                    raise ValueError(f"Historical predicted {role_name} receiver rows differ for {case_id}: {values.shape} != {expected_rows[role_name]}")
                predictions[(label, role_name)] = values.astype(np.float64)
                physical_references[(label, role_name)] = np.asarray(record.output.roles[role_name].values, dtype=np.float64)
            summary["native_classic_calls"] += 1
            absolute.append({"state": label, "heat": physical_heat.tolist(), "seconds": elapsed})
            print(json.dumps({"model": classic_id, "case_id": case_id, "state": label,
                              "seconds": elapsed}), flush=True)
        baseline = "baseline" if "baseline" in records else "transfer_minus"
        finite = []
        base_record = records[baseline]
        for label, record in records.items():
            if label == baseline:
                continue
            row = {"state": label, "baseline_state": baseline,
                   "scope": "primary baseline-relative" if baseline == "baseline" else "secondary minus-to-plus span",
                   "delta_heat": (np.asarray([module.heating for module in record.design.modules], dtype=np.float64)
                                  - np.asarray([module.heating for module in base_record.design.modules], dtype=np.float64)).tolist(),
                   "roles": {}}
            for role_name, role in record.output.roles.items():
                base_prediction = predictions[(baseline, role_name)]
                state_prediction = predictions[(label, role_name)]
                response = finite_metrics(state_prediction, physical_references[(label, role_name)], role,
                    baseline_prediction=base_prediction, baseline_reference=physical_references[(baseline, role_name)])[0]
                row["roles"][role_name] = {
                    "absolute_state_error": finite_metrics(state_prediction, physical_references[(label, role_name)], role)[0],
                    "finite_response": response,
                }
            finite.append(row)
        family_dir = output / case_id
        family_dir.mkdir(exist_ok=False)
        arrays = {f"{label}/{role_name}/prediction": predictions[(label, role_name)]
                  for label in records for role_name in records[label].output.roles}
        for row in finite:
            label = row["state"]
            for role_name in records[label].output.roles:
                arrays[f"{label}/{role_name}/delta_prediction_FP64"] = predictions[(label, role_name)] - predictions[(baseline, role_name)]
                arrays[f"{label}/{role_name}/delta_reference_FP64"] = physical_references[(label, role_name)] - physical_references[(baseline, role_name)]
        for role_name, role in first.output.roles.items():
            arrays[f"{role_name}/query_features"] = role.query_features
            arrays[f"{role_name}/valid_mask"] = role.valid_mask
            arrays[f"{role_name}/quadrature_weights"] = role.quadrature_weights
        np.savez_compressed(family_dir / "evidence.npz", **arrays)
        family = {"case_id": case_id, "baseline_state": baseline,
                  "scope": "primary baseline-relative" if baseline == "baseline" else "secondary 0277 minus-to-plus only",
                  "source_ids": list(source_ids), "states": absolute, "finite": finite,
                  "arrays": str(family_dir / "evidence.npz")}
        atomic_write_json(family_dir / "summary.json", family)
        summary["families"].append(family)
        if case_id == "0277":
            summary["secondary_0277_endpoint_state_count"] += len(records)
        else:
            summary["primary_baseline_relative_state_count"] += len(records)
    summary["timing_scope"] = "Complete old native predicted-port/local-surrogate calls on each saved fixed receiver state; includes repeated field chunking, excludes checkpoint loading and H5/record I/O."
    return summary


def field_evaluation(model, checkpoint, output, device, *, detailed=True, formal_panel=None):
    if formal_panel is None:
        dataset, manifest, path = load_native_cases(checkpoint)
        summary = {"checkpoint_epoch": checkpoint["epoch"], "dataset": str(path),
            "split": "test", "dataset_scope": "development", "channel_order": list(CHANNELS),
            "development_manifest_binding": manifest, "port_mode": "native_shared_grid_extraction",
            "scope": "22 repeatedly exposed validation cases; analytic-wake/shared-grid benchmark units",
            "not_applicable_roles": {"initial_port/outside_temperature": "New response core has no initial-port refinement trajectory",
                                     "initial_port/h_effective": "New response core has no initial-port refinement trajectory"},
            "rows": [], "solver_attempts": 0, "optimizer_updates": 0,
            "old_thermal_wrapper_calls": 0, "native_candidate_calls": 0}
    else:
        dataset, formal_scope = load_formal_native_cases(checkpoint, formal_panel)
        path = Path(formal_scope["dataset_path"])
        summary = {"checkpoint_epoch": checkpoint["epoch"], "dataset": str(path),
            "split": "test", "channel_order": list(CHANNELS),
            "port_mode": "native_shared_grid_extraction",
            "not_applicable_roles": {"initial_port/outside_temperature": "New response core has no initial-port refinement trajectory",
                                     "initial_port/h_effective": "New response core has no initial-port refinement trajectory"},
            "rows": [], "solver_attempts": 0, "optimizer_updates": 0,
            "old_thermal_wrapper_calls": 0, "native_candidate_calls": 0}
        _attach_formal_scope(summary, formal_scope)
    for index in range(len(dataset)):
        sample = dataset[index]
        case_id = str(sample["case_id"])
        synchronize(device)
        started = perf_counter()
        with torch.no_grad():
            prediction = model.predict_native_sample(sample, device=device)
        synchronize(device)
        elapsed = perf_counter() - started
        prediction = {key: numpy_value(value) for key, value in prediction.items()}
        metrics, masks = physical_case_metrics(sample, prediction, CHANNELS)
        count = int(numpy_value(sample["structure"]["module_present"]).sum())
        row = {"case_id": case_id, "intervention": "normal", "module_count": count,
            "metrics": metrics, "complete_call_and_copy_seconds": elapsed}
        summary["rows"].append(row)
        summary["native_candidate_calls"] += 1
        if detailed and case_id in FIXED4:
            evidence = {"reference_field": numpy_value(sample["steady_field"]),
                "reference_interface": numpy_value(sample["interface_target"]),
                "reference_internal_temperature": numpy_value(sample["module_internal_temperature_points"]),
                "reference_port_condition": numpy_value(sample["teacher_port_tokens"]),
                "x_grid": numpy_value(sample["x_grid"]), "y_grid": numpy_value(sample["y_grid"]),
                "module_centers": numpy_value(sample["structure"]["module_centers"]),
                "module_present": numpy_value(sample["structure"]["module_present"]),
                "heat_powers": numpy_value(sample["structure"]["heat_powers"]),
                "material_params": numpy_value(sample["structure"]["material_params"]),
                "interface_condition": numpy_value(sample["interface_condition"]),
                "module_internal_query_points": numpy_value(sample["module_internal_query_points"]),
                **{f"prediction_{key}": value for key, value in prediction.items()}, **masks}
            np.savez_compressed(output / f"{case_id}_fields.npz", **evidence)
        print(json.dumps({"case_id": case_id, "fluid_temperature_rmse": metrics["fluid/temperature"]["rmse"],
                          "seconds": elapsed}), flush=True)
    if formal_panel is not None:
        summary.update(formal_physical_aggregates(summary["rows"], formal_panel))
    else:
        summary["primary_excluding_0273"] = aggregate_physical(summary["rows"])
    summary["physical_strata"] = {f"M{count}": aggregate_physical([r for r in summary["rows"] if r["module_count"] == count])
        for count in sorted({r["module_count"] for r in summary["rows"]})}
    return summary


def response_evaluation(model, families, output, device, *, cohort):
    """Compare cold outputs, prepared increments and the same saved records."""
    summary = {"scope": "Existing analytic-wake/shared-grid finite responses; no new reference",
        "cohort": cohort, "families": [], "solver_attempts": 0, "optimizer_updates": 0,
        "central_closure_definition": "(T_plus+T_minus)/2-T_baseline; multiply by two for the sum-minus-two-baseline convention",
        "old_thermal_wrapper_calls": 0, "cold_candidate_calls": 0, "prepared_increment_calls": 0}
    for case_id, records in families:
        validate_heat_records(records)
        first = next(iter(records.values()))
        directory = output / case_id
        directory.mkdir(exist_ok=True)
        predictions, arrays, absolute = {}, {}, []
        for label, record in records.items():
            with torch.no_grad():
                values = model.predict_record(record, device=device)
            predictions[label] = values
            summary["cold_candidate_calls"] += 1
            row = {"state": label, "roles": {},
                "heat": [module.heating for module in record.design.modules],
                "centres": [list(module.position_xy) for module in record.design.modules],
                "prediction_functionals": native_functionals(record, values),
                "reference_functionals": native_functionals(record, {name: role.values for name, role in record.output.roles.items()})}
            for name, role in record.output.roles.items():
                row["roles"][name], _, _ = finite_metrics(values[name], role.values, role)
                arrays[f"{label}/{name}/prediction"] = values[name]
                arrays[f"{label}/{name}/reference"] = role.values
            absolute.append(row)
        baseline = "baseline" if "baseline" in records else "transfer_minus"
        labels = [label for label in records if label != baseline]
        finite = []
        with torch.no_grad():
            prepared = model.prepare_record(records[baseline], device=device)
        if cohort == "counted":
            with torch.no_grad():
                response = prepared["thermal"].response
                operator = model.thermal.core.export_response_operator(response)
                organization = model.thermal.core.export_organization(response)
            kernel_arrays = {name: numpy_value(value) for name, value in operator.items()
                if isinstance(value, torch.Tensor)}
            kernel_arrays.update({f"organization/{name}": numpy_value(value)
                for name, value in organization.items() if isinstance(value, torch.Tensor)})
            kernel_arrays["physical_source_ids"] = np.asarray(prepared["record_source_ids"], dtype=str)
            nominal_heat = numpy_value(prepared["module_physical_heat"])
            kernel_arrays["physical_heat"] = nominal_heat
            kernel_arrays["actual_source_contributions"] = kernel_arrays["K"]*nominal_heat[:,None,:,None]
            context = response.context
            for name in ("centers", "present", "source_lengths", "environment_coords", "environment_present"):
                kernel_arrays[f"context/{name}"] = numpy_value(getattr(context, name))
            near = torch.zeros_like(operator["K"])
            b, q, source = response.near_indices
            near[b, q, source] = response.near_values / response.forcing_scale * response.near_weight[b, q, source, None]
            kernel_arrays["executed_near_kernel"] = numpy_value(near)
            kernel_arrays["executed_far_kernel"] = kernel_arrays["K"]-kernel_arrays["executed_near_kernel"]
            kernel_arrays["executed_near_contributions"] = kernel_arrays["executed_near_kernel"]*nominal_heat[:,None,:,None]
            kernel_arrays["executed_far_contributions"] = kernel_arrays["executed_far_kernel"]*nominal_heat[:,None,:,None]
            np.savez_compressed(directory / "operator.npz", **kernel_arrays)
            atomic_write_json(directory / "operator_metadata.json", {
                "mode": organization["mode"], "source_ids": list(prepared["record_source_ids"]),
                "receiver_rows": int(response.receivers.shape[1]), "near_work": organization["near_work"],
                "far_rows": organization["far_rows"], "kernel_units": "dataset temperature / physical heating amplitude",
                "environment_semantics": organization["environment_semantics"], "ancestry": organization["ancestry"],
                "source_kernels_are_estimates": True, "independently_solved_direction_rank": 1,
                "reference": "existing analytic-wake/shared-grid; not CFD or causal attribution"})
        for label in labels:
            base_heat = np.asarray([module.heating for module in records[baseline].design.modules], dtype=np.float64)
            trial_heat = np.asarray([module.heating for module in records[label].design.modules], dtype=np.float64)
            increment = torch.as_tensor(trial_heat-base_heat, device=device, dtype=torch.float32)[None]
            with torch.no_grad():
                prepared_values = model.apply_record_increment(prepared, increment)
            prepared_values = {name: numpy_value(value) for name, value in prepared_values.items()}
            summary["prepared_increment_calls"] += 1
            row = {"state": label, "baseline_state": baseline,
                "scope": "primary baseline-relative" if baseline == "baseline" else "secondary minus-to-plus span",
                "delta_heat": (trial_heat-base_heat).tolist(), "roles": {},
                "prepared_roles": {}, "cold_prepared_max_abs": {}, "module_peak_changes": {}}
            for name, role in records[label].output.roles.items():
                row["roles"][name], delta_pred, delta_ref = finite_metrics(predictions[label][name], role.values, role,
                    baseline_prediction=predictions[baseline][name], baseline_reference=records[baseline].output.roles[name].values)
                row["prepared_roles"][name], _, _ = finite_metrics(prepared_values[name], delta_ref, role)
                row["cold_prepared_max_abs"][name] = float(np.max(np.abs(delta_pred-prepared_values[name])))
                arrays[f"{label}/{name}/delta_prediction_FP64"] = delta_pred
                arrays[f"{label}/{name}/delta_reference_FP64"] = delta_ref
                arrays[f"{label}/{name}/prepared_increment"] = prepared_values[name]
            by_state = {row["state"]: row for row in absolute}
            base, trial = by_state[baseline], by_state[label]
            pred_drop = trial["prediction_functionals"]["pressure_drop_8pct"]-base["prediction_functionals"]["pressure_drop_8pct"]
            true_drop = trial["reference_functionals"]["pressure_drop_8pct"]-base["reference_functionals"]["pressure_drop_8pct"]
            row["pressure_drop_8pct_response"] = {"prediction": pred_drop, "reference": true_drop,
                "absolute_error": abs(pred_drop-true_drop), "unit": "dataset pressure units"}
            for index, module in enumerate(records[label].design.modules):
                mid = module.module_id
                pred = trial["prediction_functionals"]["module_peak_temperature"][mid]-base["prediction_functionals"]["module_peak_temperature"][mid]
                truth = trial["reference_functionals"]["module_peak_temperature"][mid]-base["reference_functionals"]["module_peak_temperature"][mid]
                row["module_peak_changes"][mid] = {"prediction": pred, "reference": truth,
                    "absolute_error": abs(pred-truth), "unchanged_own_heat": bool(trial_heat[index]==base_heat[index]),
                    "unit": "dataset temperature units"}
            finite.append(row)
        for name, role in first.output.roles.items():
            for key in ("query_features", "valid_mask", "quadrature_weights"):
                arrays[f"{name}/{key}"] = getattr(role, key)
            for key in ("channel_names", "channel_units", "query_ids", "receiver_module_ids"):
                arrays[f"{name}/{key}"] = np.asarray(getattr(role, key) or (), dtype=str)
        closure = {}
        if "baseline" in records and len(records)==3:
            ends = [label for label in records if label != "baseline"]
            for name, role in first.output.roles.items():
                predicted = (predictions[ends[0]][name].astype(np.float64)+predictions[ends[1]][name].astype(np.float64))/2-predictions["baseline"][name].astype(np.float64)
                reference = (records[ends[0]].output.roles[name].values.astype(np.float64)+records[ends[1]].output.roles[name].values.astype(np.float64))/2-records["baseline"].output.roles[name].values.astype(np.float64)
                closure[name], _, _ = finite_metrics(predicted, reference, role)
                arrays[f"{name}/central_closure_prediction"] = predicted
                arrays[f"{name}/central_closure_reference"] = reference
        np.savez_compressed(directory / "evidence.npz", **arrays)
        family = {"case_id": case_id, "physical_family_id": first.design.physical_family_id,
            "source_partition": first.design.split.value, "baseline_available": "baseline" in records,
            "absolute": absolute, "finite": finite, "central_closure": closure,
            "arrays": str(directory / "evidence.npz")}
        if cohort == "counted":
            reference = {r["state"]:r["reference_functionals"]["maximum_material_temperature"] for r in absolute}
            prediction = {r["state"]:r["prediction_functionals"]["maximum_material_temperature"] for r in absolute}
            warnings = [abs(float(r.provenance.get("runtime",{}).get("final_delta_inf",0))) for r in records.values()]
            family["stored_pool_decision"] = stored_pool_decision(reference, prediction, warning_scale=2*max(warnings))
        atomic_write_json(directory / "summary.json", family)
        summary["families"].append(family)
        print(json.dumps({"case_id":case_id,"states":len(records),"cohort":cohort}),flush=True)
    summary.update(summarize_families(summary["families"]))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scalars-only", action="store_true")
    parser.add_argument("--mode", choices=("fields", "responses", "classic-fields", "classic-responses"), default="fields")
    parser.add_argument("--classic-id", choices=tuple(CLASSIC_IDENTITIES),
        help="Exact retained classic identity required by a classic mode.")
    parser.add_argument("--development-manifest", type=Path,
        help="Exact shared fixed25_v1 manifest required by a classic mode.")
    parser.add_argument("--formal-panel", choices=("canonical89", "original90"),
        help="Opt in to the sealed full-TRAIN R-direct formal validation panel.")
    parser.add_argument("--formal-reference-checkpoint", type=Path,
        help="Trusted non-startup R-direct e5000 checkpoint that seals the physical formal panel for classic replay.")
    parser.add_argument("--cohort", choices=("fit", "development", "counted"), default="development")
    parser.add_argument("--atlas-dir", type=Path, default=ROOT / "diagnostics/generated/interactions/physical_response_atlas_20260926/families")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--records-dir", type=Path)
    args = parser.parse_args(argv)
    if args.formal_panel is not None and args.mode not in ("fields", "classic-fields"):
        parser.error("--formal-panel is supported only for fields and classic-fields modes.")
    if args.mode == "classic-fields":
        if args.classic_id is None:
            parser.error("classic-fields requires --classic-id")
        if args.formal_panel is None and args.development_manifest is None:
            parser.error("classic-fields requires --development-manifest unless --formal-panel is selected")
        if args.formal_panel is not None and args.formal_reference_checkpoint is None:
            parser.error("formal classic-fields requires --formal-reference-checkpoint")
        if args.formal_panel is not None and args.development_manifest is not None:
            parser.error("--development-manifest cannot be combined with --formal-panel")
    elif args.mode == "classic-responses":
        if args.classic_id is None or args.development_manifest is None:
            parser.error("classic-responses requires --classic-id and --development-manifest")
    elif args.classic_id is not None or args.development_manifest is not None:
        parser.error("--classic-id and --development-manifest are only valid in classic modes")
    if args.formal_reference_checkpoint is not None and not (
            args.mode == "classic-fields" and args.formal_panel is not None):
        parser.error("--formal-reference-checkpoint is only valid for formal classic-fields")
    if args.formal_panel is not None and args.mode == "fields" and args.development_manifest is not None:
        parser.error("--development-manifest does not apply to formal fields")
    started_unix, started = time(), perf_counter()
    torch.set_num_threads(1)
    output = validate_generated_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve measured evidence: choose a fresh output directory.")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    formal_scope = None
    if args.mode.startswith("classic-"):
        from channelthermal.evaluation.loading import load_model

        model, checkpoint = load_model(checkpoint_path, torch.device(args.device))
        identity = validate_classic_selection(args.classic_id, checkpoint,
            _file_sha256(checkpoint_path))
        if args.formal_panel is not None:
            from channelthermal.source_response import load_source_response_model

            reference_path = args.formal_reference_checkpoint.expanduser().resolve()
            reference_model, reference_checkpoint = load_source_response_model(reference_path, "cpu")
            formal_scope = validate_formal_evaluation_checkpoint(reference_checkpoint,
                panel=args.formal_panel)
            del reference_model
    else:
        from channelthermal.source_response import load_source_response_model

        model, checkpoint = load_source_response_model(checkpoint_path, args.device)
        if args.formal_panel is not None:
            formal_scope = validate_formal_evaluation_checkpoint(checkpoint, panel=args.formal_panel)
    model.eval().requires_grad_(False)
    before = model_digest(model)
    if args.mode == "classic-fields":
        summary = classic_field_evaluation(model, checkpoint, output, args.device,
            classic_id=args.classic_id, manifest_path=args.development_manifest,
            formal_scope=formal_scope)
        summary["selected_checkpoint_identity"] = identity
        if formal_scope is not None:
            summary["formal_reference_checkpoint"] = {
                "path": str(reference_path),
                "epoch": 5000,
                "sha256": _file_sha256(reference_path),
                "response_family": "R-direct",
            }
    elif args.mode == "classic-responses":
        if args.request is None or args.records_dir is None:
            parser.error("classic-responses requires its exact saved --request and --records-dir")
        summary = classic_counted_response_evaluation(model, checkpoint, output, args.device,
            classic_id=args.classic_id, manifest_path=args.development_manifest,
            request_path=args.request, records_dir=args.records_dir)
        summary["selected_checkpoint_identity"] = identity
    elif args.mode == "fields":
        summary = field_evaluation(model, checkpoint, output, args.device,
            detailed=not args.scalars_only, formal_panel=args.formal_panel)
    else:
        if args.cohort == "counted":
            if args.request is None or args.records_dir is None:
                parser.error("Counted replay requires its exact saved --request and --records-dir.")
            families, _ = load_counted_families(args.request, args.records_dir)
        else:
            ids = ("0001", "0318", "0333", "0348") if args.cohort == "fit" else ("0304", "0320", "0335", "0350")
            families = load_atlas_families([args.atlas_dir / f"train_{cid}_responses.npz" for cid in ids],
                ("heat_transfer_minus", "heat_transfer_plus"), args.cohort)
        summary = response_evaluation(model, families, output, args.device, cohort=args.cohort)
        summary["checkpoint_epoch"] = checkpoint["epoch"]
    summary.update(checkpoint=str(checkpoint_path),checkpoint_sha256=_file_sha256(checkpoint_path),
                   frozen_state_unchanged=model_digest(model)==before)
    if not summary["frozen_state_unchanged"]:
        raise RuntimeError("Evaluation changed the frozen response model.")
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "receipt.json", {"start_unix": started_unix, "end_unix": time(),
        "elapsed_seconds": perf_counter()-started, "pid": os.getpid(), "device": args.device,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "status": "completed",
        "optimizer_updates": 0, "solver_attempts": 0,
        "native_classic_calls": summary.get("native_classic_calls", 0)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
