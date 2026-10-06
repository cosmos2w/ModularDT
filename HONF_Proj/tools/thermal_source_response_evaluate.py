#!/usr/bin/env python3
"""Measure an opt-in response model using existing native observations only.

This adapter reports unsupported initial-port history explicitly. It never
executes the old thermal wrapper, launches a reference solver, or fits a model.
"""
from __future__ import annotations

import argparse
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
from channelthermal.training.checkpoints import _file_sha256
from thermal_campaign_evaluate import aggregate_physical, physical_case_metrics
from thermal_development import validate_generated_output
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


def field_evaluation(model, checkpoint, output, device, *, detailed=True):
    dataset, manifest, path = load_native_cases(checkpoint)
    summary = {"checkpoint_epoch": checkpoint["epoch"], "dataset": str(path),
        "split": "test", "dataset_scope": "development", "channel_order": list(CHANNELS),
        "development_manifest_binding": manifest, "port_mode": "native_shared_grid_extraction",
        "scope": "22 repeatedly exposed validation cases; analytic-wake/shared-grid benchmark units",
        "not_applicable_roles": {"initial_port/outside_temperature": "New response core has no initial-port refinement trajectory",
                                 "initial_port/h_effective": "New response core has no initial-port refinement trajectory"},
        "rows": [], "solver_attempts": 0, "optimizer_updates": 0,
        "old_thermal_wrapper_calls": 0, "native_candidate_calls": 0}
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
    parser.add_argument("--mode", choices=("fields", "responses"), default="fields")
    parser.add_argument("--cohort", choices=("fit", "development", "counted"), default="development")
    parser.add_argument("--atlas-dir", type=Path, default=ROOT / "diagnostics/generated/interactions/physical_response_atlas_20260926/families")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--records-dir", type=Path)
    args = parser.parse_args(argv)
    started_unix, started = time(), perf_counter()
    torch.set_num_threads(1)
    from channelthermal.source_response import load_source_response_model
    output = validate_generated_output(args.output_dir)
    if (output / "summary.json").exists():
        raise FileExistsError("Preserve measured evidence: choose a fresh output directory.")
    output.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_source_response_model(args.checkpoint, args.device)
    model.eval().requires_grad_(False)
    before = model_digest(model)
    if args.mode == "fields":
        summary = field_evaluation(model, checkpoint, output, args.device, detailed=not args.scalars_only)
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
    summary.update(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=_file_sha256(args.checkpoint),
                   frozen_state_unchanged=model_digest(model)==before)
    if not summary["frozen_state_unchanged"]:
        raise RuntimeError("Evaluation changed the frozen response model.")
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "receipt.json", {"start_unix": started_unix, "end_unix": time(),
        "elapsed_seconds": perf_counter()-started, "pid": os.getpid(), "device": args.device,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "status": "completed",
        "optimizer_updates": 0, "solver_attempts": 0, "old_thermal_wrapper_calls": 0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
