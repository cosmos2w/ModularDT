#!/usr/bin/env python3
"""Evaluate retained Thermal references through their qualified public CPU path."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for source in (PROJECT / "src", PROJECT / "tools", PROJECT / "Case_ThermalChannel/src",
               PROJECT / "Case_WindFarm/src"):
    sys.path.insert(0, str(source))

import numpy as np
import torch
from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
from channelthermal.evaluation.loading import checkpoint_native_inputs_from_physical_sample, load_model
from channelthermal.evaluation.prepared import predict_case
from channelthermal.native_observations import native_cell_center_stencil
from content_hypergraph_evaluate import load_evaluation_components
from content_hypergraph_train import canonical_sha256

INPUTS = ("heat_powers", "interface_condition", "local_module_params", "teacher_port_tokens",
          "query_xy", "module_internal_query_points", "module_internal_mask",
          "module_present", "module_centers", "material_params")
OUTPUTS = ("pred_field_grid", "pred_interface", "pred_internal_temperature")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_metric(prediction: Any, target: Any, valid: Any) -> dict[str, Any]:
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("Reference and target native axes differ.")
    mask = np.broadcast_to(np.asarray(valid, dtype=bool), target.shape)
    if not mask.any() or not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Native reference metric requires finite, nonempty common support.")
    error, truth = (prediction - target)[mask], target[mask]
    norm = float(np.linalg.norm(truth))
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "relative_l2": float(np.linalg.norm(error) / norm) if norm > 0.0 else None,
            "reference_norm": norm, "valid_rows": int(mask.sum())}


def exact_array_comparison(left: Any, right: Any) -> dict[str, Any]:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("Public/helper reference input axes differ.")
    return {"shape": list(left.shape), "byte_equal": left.dtype == right.dtype and left.tobytes() == right.tobytes(),
            "maximum_absolute_difference": float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))}


def common_native_mask(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.stack((sample["x_grid"], sample["y_grid"]), axis=-1).astype(np.float32)
    if grid.shape != (64, 128, 2):
        raise ValueError("Thermal references require the original 128x64 physical grid.")
    structure = sample["structure"]
    lengths = torch.tensor([[float(np.asarray(structure[key]).reshape(-1)[0])
                             for key in ("domain_length_x", "domain_length_y")]], dtype=torch.float32)
    stencil = native_cell_center_stencil(torch.from_numpy(grid.reshape(1, -1, 2)), lengths, nx=128, ny=64)
    return stencil.valid.numpy().reshape(64, 128) & ~np.asarray(sample["module_mask"], dtype=bool), grid


def predict_reference(model: Any, checkpoint: dict[str, Any], sample: dict[str, Any]) -> dict[str, np.ndarray]:
    prediction = predict_case(model, copy.deepcopy(sample), torch.device("cpu"), query_batch_size=8192,
                              local_port_condition_mode="predicted", mixed_teacher_ratio=0.5)
    result = {key: np.asarray(prediction[key], dtype=np.float32) for key in OUTPUTS}
    config = checkpoint.get("global_normalization_config", checkpoint.get("train_config", {}).get("dataset", {}))
    if bool(config.get("normalize_targets", False)):
        normalizer = H5Normalizer(checkpoint["global_normalization_stats"])
        result = dict(zip(OUTPUTS, (
            np.asarray(normalizer.denormalize_fields(result[OUTPUTS[0]]), dtype=np.float32),
            np.asarray(normalizer.denormalize_interface_targets(result[OUTPUTS[1]]), dtype=np.float32),
            np.asarray(normalizer.denormalize_internal_temperature(result[OUTPUTS[2]]), dtype=np.float32),
        )))
    if result[OUTPUTS[0]].shape != (64, 128, 5):
        raise ValueError("Reference returned a non-native Thermal field.")
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Reference native prediction is nonfinite.")
    return result


def qualify_public_inputs(raw: dict[str, Any], public: dict[str, Any], model: Any,
                          checkpoint: dict[str, Any]) -> dict[str, Any]:
    helper = checkpoint_native_inputs_from_physical_sample(raw, checkpoint)
    inputs = {}
    for key in INPUTS:
        in_structure = key in ("heat_powers", "module_present", "module_centers", "material_params")
        inputs[key] = exact_array_comparison(helper["structure"][key] if in_structure else helper[key],
                                             public["structure"][key] if in_structure else public[key])
    predictions = [predict_reference(model, checkpoint, value) for value in (helper, public, public)]
    helper_diffs = {key: exact_array_comparison(predictions[0][key], predictions[1][key]) for key in OUTPUTS}
    repeat_diffs = {key: exact_array_comparison(predictions[1][key], predictions[2][key]) for key in OUTPUTS}
    passed = all(value["byte_equal"] for group in (inputs, helper_diffs, repeat_diffs) for value in group.values())
    return {"case_id": str(raw["case_id"]), "passed": passed, "inputs": inputs,
            "public_helper_predictions": helper_diffs, "repeated_public_predictions": repeat_diffs}


def reference_fields(prediction: dict[str, np.ndarray], sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fluid, grid = common_native_mask(sample)
    fields = {name: native_metric(prediction["pred_field_grid"][..., index], sample["steady_field"][..., index], fluid)
              for index, name in enumerate(("u", "v", "p", "omega", "temperature_fluid"))}
    present = np.asarray(sample["structure"]["module_present"]) > 0.5
    surface = np.asarray(sample["interface_condition_valid_mask"], dtype=bool) & present[:, None]
    interface, target_interface = prediction["pred_interface"], np.asarray(sample["interface_target"])
    if interface.shape != target_interface.shape or interface.shape[1:] != (64, 2):
        raise ValueError("Reference does not preserve all 64 native angular receivers per source.")
    fields["temperature_surface"] = native_metric(interface[..., 0], target_interface[..., 0], surface)
    fields["q_proxy"] = native_metric(interface[..., 1], target_interface[..., 1], surface)
    material = prediction["pred_internal_temperature"]
    if material.ndim == 3:
        material = material[..., 0]
    target_material = np.asarray(sample["module_internal_temperature_points"])
    material_valid = np.isfinite(target_material) & present[:, None]
    fields["temperature_material"] = native_metric(material, target_material, material_valid)
    pred_peak = np.where(material_valid, material, -np.inf).max(axis=1)[present]
    true_peak = np.where(material_valid, target_material, -np.inf).max(axis=1)[present]
    fields["module_peak_endpoint"] = native_metric(pred_peak, true_peak, np.ones_like(true_peak, dtype=bool))
    return fields, {"fluid_mask": fluid, "xy": grid, "interface_mask": surface,
                    "material_mask": material_valid, "material_prediction": material}


def panel_identity(provider: Any) -> dict[str, Any]:
    ids = tuple(map(str, provider.validation_case_ids))
    train = tuple(map(str, provider.training_case_ids))
    if provider.protocol == "development":
        if len(ids) != 22 or len(train) != 150:
            raise ValueError("Reference development panel must be TRAIN150/DEV22.")
    elif provider.protocol == "full_train1000":
        if len(ids) != 89 or len(train) != 600 or "0273" in ids:
            raise ValueError("Reference full follow-up requires TRAIN600/canonical89 excluding duplicate0273.")
    else:
        raise ValueError("Unrecognized Thermal reference panel protocol.")
    if set(ids) & set(train) or len(set(ids)) != len(ids):
        raise ValueError("Thermal reference membership is duplicated or crosses source partitions.")
    return {"protocol": provider.protocol, "validation_case_ids": list(ids),
            "validation_ids_sha256": canonical_sha256(ids), "train_count": len(train),
            "source_binding": provider.identity_payload()["source_binding"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-case", action="append", default=[])
    parser.add_argument("--qualification-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Reference output is immutable; choose a fresh directory.")
    torch.set_num_threads(4)
    started = time.perf_counter()
    bound_model, provider, _ = load_evaluation_components(args.panel_checkpoint, "cpu")
    if not hasattr(provider, "validation_case_ids"):
        raise ValueError("A Thermal content-family checkpoint must bind the reference panel.")
    panel = panel_identity(provider)
    del bound_model
    ids = tuple(panel["validation_case_ids"])
    if any(case not in ids for case in args.export_case):
        raise ValueError("Reference exports must belong to the bound validation panel.")
    probes = ("0291", "0687")
    if not set(probes).issubset(ids):
        raise ValueError("The native reference qualification probes are missing from the declared panel.")
    references = {}
    for value in args.reference_checkpoint:
        label, separator, filename = value.partition("=")
        if not separator or not label or label in references or Path(label).name != label:
            raise ValueError("Reference paths need unique safe LABEL=PATH arguments.")
        references[label] = Path(filename).resolve(strict=True)
    args.output.mkdir(parents=True)
    rows, models, qualification = [], {}, []
    with torch.no_grad():
        for label, path in references.items():
            model, checkpoint = load_model(path, torch.device("cpu"))
            model.eval()
            if "global_normalization_stats" not in checkpoint:
                raise ValueError("A native classic reference must retain its own global normalization state.")
            raw = GlobalChannelThermalDataset(provider.data_path, split="test", points_per_case=None,
                normalize_inputs=False, normalize_targets=False, random_point_sampling=False, seed=0,
                include_grid=True, normalizer=H5Normalizer({}), case_ids=ids)
            public = GlobalChannelThermalDataset(provider.data_path, split="test", points_per_case=None,
                normalize_inputs=True, normalize_targets=False, random_point_sampling=False, seed=0,
                include_grid=True, normalizer=H5Normalizer(checkpoint["global_normalization_stats"]), case_ids=ids)
            try:
                positions = {str(case): index for index, case in enumerate(raw.selected_case_ids)}
                for case in probes:
                    q = qualify_public_inputs(raw[positions[case]], public[positions[case]], model, checkpoint)
                    qualification.append({"model": label, **q})
                    if not q["passed"]:
                        (args.output / "qualification_failure.json").write_text(json.dumps(
                            {"checkpoint": str(path), "sha256": sha256(path), "qualification": qualification},
                            indent=2, allow_nan=False) + "\n")
                        raise ValueError("Public/helper inputs or repeated CPU predictions fail exact reference qualification.")
                if not args.qualification_only:
                    for case in ids:
                        sample, normalized = raw[positions[case]], public[positions[case]]
                        prediction = predict_reference(model, checkpoint, normalized)
                        fields, arrays = reference_fields(prediction, sample)
                        count = int(np.count_nonzero(sample["structure"]["module_present"] > 0.5))
                        rows.append({"model": label, "case_id": case, "module_count": count, "fields": fields,
                                     "common_fluid_mask_sha256": hashlib.sha256(arrays["fluid_mask"].tobytes()).hexdigest()})
                        if case in args.export_case:
                            np.savez_compressed(args.output / f"{label}_{case}.npz", **prediction, **arrays,
                                target=sample["steady_field"], interface_target=sample["interface_target"],
                                material_target=sample["module_internal_temperature_points"],
                                source_ids=sample["structure"].get("module_source_ids", np.arange(len(sample["structure"]["module_present"]))),
                                centers=sample["structure"]["module_centers"], present=sample["structure"]["module_present"])
            finally:
                raw.close()
                public.close()
            models[label] = {"checkpoint": str(path), "sha256": sha256(path), "stored_epoch": int(checkpoint["epoch"]),
                             "normalization_sha256": canonical_sha256({key: np.asarray(value).tolist()
                                 for key, value in checkpoint["global_normalization_stats"].items()})}
            del model
    if torch.cuda.is_initialized():
        raise RuntimeError("Native CPU reference evaluation unexpectedly initialized CUDA.")
    result = {"schema": "thermal_public_cpu_reference_board_v1", "status": "completed", "panel": panel,
              "panel_checkpoint": str(args.panel_checkpoint.resolve()), "panel_checkpoint_sha256": sha256(args.panel_checkpoint),
              "models": models, "qualification": qualification, "rows": rows, "device": "cpu",
              "qualification_only": args.qualification_only, "elapsed_seconds": time.perf_counter() - started,
              "old_GPU_reference_quarantine": "unchanged; this is a separately qualified CPU board",
              "new_solver_attempts": 0, "optimizer_updates": 0, "unexposed_targets_outside_bound_panel_read": False,
              "support": "native8192 fluid valid excluding solids; all64 surface; all stored material; q is native proxy"}
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "completed", "rows": len(rows), "models": models,
                      "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
