#!/usr/bin/env python3
"""Frozen native-checkpoint finite responses from existing recorded stencils.

Decode each absolute state once, save its arrays, and compare field/material,
pressure and peak increments against both the model and zero change. No new
physical solve or inverse/forward fitting occurs in this evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def response_channel_metrics(prediction, reference, valid, weights, names, units):
    """Physical absolute errors and reference magnitudes; no unresolved ratios."""
    prediction, reference = np.asarray(prediction, dtype=float), np.asarray(reference, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    valid = np.broadcast_to(valid[:, None] if valid.ndim == 1 else valid, reference.shape)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if prediction.shape != reference.shape or reference.ndim != 2 or weights.shape != reference.shape[:1]:
        raise ValueError("Response arrays, masks and quadrature must align")
    if len(names) != reference.shape[1] or len(units) != len(names):
        raise ValueError("Response channels must retain their physical names/units")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Response quadrature must be finite and nonnegative")
    result = {}
    for index, (name, unit) in enumerate(zip(names, units)):
        mask = valid[:, index] & (weights > 0)
        row = {"unit": unit, "count": int(mask.sum()), "finite": True,
               "rmse": None, "mae": None, "reference_rms": None, "zero_change_rmse": None,
               "relative_accuracy": None, "response_floor": "unresolved; absolute evidence only"}
        if mask.any():
            predicted, target = prediction[mask, index], reference[mask, index]
            row["finite"] = bool(np.isfinite(predicted).all() and np.isfinite(target).all())
            if row["finite"]:
                weight = weights[mask] / weights[mask].sum()
                error = predicted - target
                row.update(rmse=float(np.sqrt(weight @ error**2)), mae=float(weight @ np.abs(error)),
                           reference_rms=float(np.sqrt(weight @ target**2)))
                row["zero_change_rmse"] = row["reference_rms"]
        result[str(name)] = row
    return result


def predict_stencil_with_fine_work(backend, operator, stencil, **kwargs):
    """Measure each actual absolute prediction; never repeat model inference."""
    from channelthermal.response_control.algebra import predict_stencil
    from thermal_campaign_benchmark import optional_fine_work

    labels = ["baseline", *stencil.variants]
    states = {}

    def measured_operator(*args, **inputs):
        label = labels[len(states)]
        with optional_fine_work(backend) as work:
            prediction = operator(*args, **inputs)
        states[label] = work
        return prediction

    predictions = predict_stencil(measured_operator, stencil, **kwargs)
    if list(states) != labels:
        raise RuntimeError("Fine work must retain every absolute stencil prediction")
    measured = all(work["measured"] for work in states.values())
    totals = None
    if measured:
        totals = {route: {key: sum(work["routes"][route][key] for work in states.values())
                          for key in ("padded_input_rows", "calls")}
                  for route in next(iter(states.values()))["routes"]}
    result = {"measured": measured, "routes": totals, "states": states,
              "scope": "All successful absolute native predictions in this stencil; five-route padded fine MLP input rows and calls",
              "excluded": "Eligible/unique pairs, attention, policy/coarse/local physics, backward and graph export"}
    if not measured:
        result["reason"] = "; ".join(sorted({work["reason"] for work in states.values() if not work["measured"]}))
    return predictions, result


def response_paths(checkpoint, manifest, stencil_paths, panel_config):
    """Development defaults use only the checkpoint's bounded training anchors."""
    from honf_runtime.paths import resolve_path

    if stencil_paths is None:
        if manifest is not None:
            values = checkpoint.get("train_config", {}).get("training", {}).get("campaign", {}).get("response_stencils", [])
        else:
            values = json.loads(Path(panel_config).read_text())["stencils"]
        stencil_paths = [resolve_path(value) for value in values]
    paths = [Path(value) for value in stencil_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Response evaluation requires unique available stencils; no silent full-atlas expansion")
    if manifest is not None and len(paths) > 4:
        raise ValueError("Development response evaluation is bounded to at most four selected-train anchors")
    return paths


def validate_response_membership(stencil, metadata, manifest):
    if manifest is None:
        return
    from channelthermal.data.development_split import development_case_ids

    if stencil.split.value != "train" or metadata.get("anchor_id") not in development_case_ids(manifest, "train"):
        raise ValueError("Development response family is outside selected training membership; use explicit formal-full scope")


def evaluate_responses(checkpoint_path, *, dataset_path, stencil_paths, output_dir, device="cpu", query_batch_size=1024,
                       evaluation_scope="auto", panel_config=None):
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model
    from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
    from channelthermal.response_control.contracts import DesignInput
    from channelthermal.response_control.native import DifferentiableThermalOperator
    from channelthermal.response_control.thermal import reduce_native_thermal_quantities
    from channelthermal.training.campaign_work import CampaignForwardWork
    from thermal_development import evaluation_dataset_kwargs, resolve_evaluation_manifest, validate_generated_output

    output_dir = validate_generated_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(Path(checkpoint_path), torch.device(device))
    model.eval().requires_grad_(False)
    settings = checkpoint.get("train_config", {}).get("dataset", {})
    manifest = resolve_evaluation_manifest(settings, dataset_path, scope=evaluation_scope)
    paths = response_paths(checkpoint, manifest, stencil_paths,
        panel_config or PROJECT_ROOT / "src/config_core/evaluation/thermal_campaign_response_panel.json")
    families = [(path, *load_response_atlas_stencil(path)) for path in paths]
    for _path, stencil, metadata in families:
        validate_response_membership(stencil, metadata, manifest)
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint.get("global_normalization_stats", {}).items()}
    if (manifest is not None or settings.get("development_manifest") or settings.get("development_subset")) and not stats:
        raise ValueError("Development responses require saved selected-training global normalization stats")
    dataset = GlobalChannelThermalDataset(dataset_path, split="train", points_per_case=1,
        random_point_sampling=False, include_grid=True, normalizer=H5Normalizer(stats) if stats else None,
        **evaluation_dataset_kwargs(manifest, "train"))
    operator = DifferentiableThermalOperator(model, dataset[0],
        dataset_config=checkpoint.get("train_config", {}).get("dataset", {}),
        normalization_stats=checkpoint.get("global_normalization_stats", {}), query_batch_size=query_batch_size)
    summary = {"checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_epoch": checkpoint.get("epoch"),
        "source": "existing independent physical-response atlas records; previously exposed cohorts",
        "limits": "analytic/shared-grid benchmark; q_normal proxy; no new solve; unresolved response floors",
        "forward_weights": "frozen", "dataset_scope": "development" if manifest is not None else "formal-full",
        "development_manifest_binding": manifest, "families": []}
    for path, stencil, _metadata in families:
        records = {"baseline": stencil.baseline, **stencil.variants}
        family_dir = output_dir / stencil.physical_family_id
        family_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model.core.backend, "set_execution_mode"):
            model.core.backend.set_plan_intervention("normal")
            model.core.backend.set_execution_mode("dense_masked_reference")
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        started = perf_counter()
        with torch.no_grad(), CampaignForwardWork(model.core) as work:
            predictions, fine_work = predict_stencil_with_fine_work(model.core.backend, operator, stencil, device=device)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        family = {"family_id": stencil.physical_family_id, "split": stencil.split.value,
            "stencil": str(Path(path).resolve()), "absolute_states": len(records),
            "complete_wrapper_seconds": perf_counter() - started, "forward_work": work.records,
            "complete_wrapper_seconds_scope": "Evidence timer includes forward ledgers and fine-MLP work hooks when supported; not uninstrumented benchmark latency",
            "fine_kernel_work": fine_work,
            "work_scope": "all absolute forward calls; no backward; default dense reference", "variants": []}
        quantities = {}
        arrays = {"labels": np.asarray(list(records))}
        for label, record in records.items():
            prediction = predictions.values[label]
            for role, value in prediction.role_values.items():
                arrays[f"{label}/{role}/prediction"] = value.cpu().numpy()
                arrays[f"{label}/{role}/reference"] = record.output.roles[role].values
                arrays[f"{label}/{role}/valid"] = record.output.roles[role].valid_mask
            arrays[f"{label}/heat"] = np.asarray([module.heating for module in record.design.modules])
            arrays[f"{label}/centres"] = np.asarray([module.position_xy for module in record.design.modules])
            quantities[label] = reduce_native_thermal_quantities(prediction,
                DesignInput.from_state(record.design, device=device), predictions.role_queries, record.context.values,
                module_ids=tuple(module.module_id for module in record.design.modules),
                solid_valid_mask=record.output.roles["solid_temperature"].valid_mask)
        for role, query in predictions.role_queries.items():
            arrays[f"{role}/queries"] = query.query_features.cpu().numpy()
            arrays[f"{role}/quadrature"] = stencil.baseline.output.roles[role].quadrature_weights
            arrays[f"{role}/channels"] = np.asarray(query.channel_names)
            arrays[f"{role}/units"] = np.asarray(query.channel_units)
        baseline = quantities["baseline"]
        for label, record in stencil.variants.items():
            row = {"label": label, "roles": {}}
            for role in predictions.role_queries:
                block = stencil.finite_change(label, role)
                delta = predictions.finite(label, role).cpu().numpy()
                arrays[f"{label}/{role}/delta_prediction"] = delta
                arrays[f"{label}/{role}/delta_reference"] = block.delta
                arrays[f"{label}/{role}/delta_valid"] = block.valid_mask
                row["roles"][role] = response_channel_metrics(delta, block.delta, block.valid_mask,
                    block.quadrature_weights, block.channel_names, block.channel_units)
            pressure = float(quantities[label].pressure_drop - baseline.pressure_drop)
            reference = float(record.output.quantities["pressure_drop"].value
                              - stencil.baseline.output.quantities["pressure_drop"].value)
            row["pressure_increment"] = {"prediction": pressure, "reference": reference,
                "absolute_error": abs(pressure-reference), "zero_change_error": abs(reference),
                "unit": quantities[label].pressure_drop_units}
            row["module_peak_increments"] = {}
            for module_id, peak in quantities[label].module_peak_temperature.items():
                delta = float(peak - baseline.module_peak_temperature[module_id])
                target = float(record.output.module_peak_temperature[module_id]
                               - stencil.baseline.output.module_peak_temperature[module_id])
                row["module_peak_increments"][module_id] = {"prediction": delta, "reference": target,
                    "absolute_error": abs(delta-target), "zero_change_error": abs(target),
                    "unit": quantities[label].temperature_units}
            family["variants"].append(row)
        np.savez_compressed(family_dir / "evidence.npz", **arrays)
        (family_dir / "summary.json").write_text(json.dumps(family, indent=2, allow_nan=False) + "\n")
        summary["families"].append(family)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(f"{stencil.physical_family_id}: saved {len(records)} absolute states and their finite responses", flush=True)
    return output_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stencils", nargs="+", type=Path)
    parser.add_argument("--panel-config", type=Path,
                        default=PROJECT_ROOT / "src/config_core/evaluation/thermal_campaign_response_panel.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--evaluation-scope", choices=("auto", "development", "formal-full"), default="auto")
    args = parser.parse_args(argv)
    if args.query_batch_size < 1:
        parser.error("query batch size must be positive")
    if args.stencils is not None and (not args.stencils or len(set(args.stencils)) != len(args.stencils)):
        parser.error("the response panel must contain unique stencils")
    return args


if __name__ == "__main__":
    args = parse_args()
    print(evaluate_responses(args.checkpoint, dataset_path=args.dataset, stencil_paths=args.stencils,
        output_dir=args.output_dir, device=args.device, query_batch_size=args.query_batch_size,
        evaluation_scope=args.evaluation_scope, panel_config=args.panel_config))
