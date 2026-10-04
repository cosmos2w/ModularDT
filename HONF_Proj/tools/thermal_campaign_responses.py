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


def audit_subset_response_pairs(dataset_path, manifest):
    """Input-only exact layout/context matching inside the frozen 150/22.

    This reads selected case inputs, not excluded populations or model errors.
    Exact geometry/context equality is conservative: a nearby layout does not
    create a physical response label. Heat alone is permitted to vary.
    """
    import h5py
    from channelthermal.data.development_split import development_case_ids

    families = {}
    selections = {partition: development_case_ids(manifest, partition) for partition in ("train", "test")}
    with h5py.File(dataset_path, "r") as handle:
        for partition, ids in selections.items():
            for case_id in ids:
                group = handle["cases"][case_id]
                config = json.loads(group["case_config_json"][()].decode())
                active = np.asarray(group["module_present"][...]) > .5
                positions = np.asarray(group["module_centers"][...])[active]
                heat = np.asarray(group["heat_powers"][...])[active]
                order = np.lexsort((positions[:, 1], positions[:, 0]))
                thermal = {key: value for key, value in config["thermal"].items() if key not in {"heat_power_min", "heat_power_max"}}
                context = {"domain": config["domain"], "flow": config["flow"], "thermal": thermal}
                key = (tuple(map(tuple, positions[order].tolist())), json.dumps(context, sort_keys=True))
                families.setdefault(key, []).append({"case_id": case_id, "partition": partition,
                    "heat": heat[order].tolist(), "converged": bool(group.attrs.get("converged", False))})
    matched = []
    for members in families.values():
        pairs = []
        for index, left in enumerate(members):
            for right in members[index+1:]:
                if left["heat"] != right["heat"] and left["converged"] and right["converged"]:
                    pairs.append({"left": left["case_id"], "right": right["case_id"],
                        "partitions": [left["partition"], right["partition"]]})
        if pairs:
            matched.append({"members": members, "different_heat_pairs": pairs})
    held = [family for family in matched if any("test" in pair["partitions"] for pair in family["different_heat_pairs"])]
    return {"dataset": str(Path(dataset_path).resolve()), "manifest_sha256": manifest["manifest_sha256"],
        "selected_counts": {partition: len(ids) for partition, ids in selections.items()},
        "selection": "exact coordinate/order-canonicalized geometry and prescribed solver context; different active heat",
        "distinct_layout_context_families": len(families), "matched_nonzero_response_families": matched,
        "identical_heat_layout_context_duplicates": [[{"case_id": item["case_id"], "partition": item["partition"]}
            for item in members] for members in families.values() if len(members) > 1 and len({tuple(item["heat"]) for item in members}) == 1],
        "held_response_family_count": len(held),
        "C_positive_response_transfer": "available stored matched families require aligned response evaluation" if held else "unavailable: no exact matched held-layout different-heat labels in fixed subset",
        "excluded_cases_read": 0, "solver_calls": 0}


def verify_stored_heat_null(control_directory):
    """Read-only null verification on historical stored controls, not training.

    Their excluded train0001 family is only a diagnostic of benchmark physics;
    it is never inserted into current training or selected-panel evaluation.
    """
    from channelthermal.interaction_evidence.storage import load_solve_record
    from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
    root = Path(control_directory)
    settings = json.loads((root / "frozen_inputs.json").read_text())
    stencil, _ = load_response_atlas_stencil(settings["source_atlas"])
    baseline = stencil.baseline
    rows = []
    for candidate in settings["candidates"]:
        record = load_solve_record(root / "records" / candidate["record_id"])
        if record.output is None or baseline.output is None:
            raise ValueError("Stored control is missing its completed physical output")
        left, right = baseline.output.roles["fluid_fields"], record.output.roles["fluid_fields"]
        if not np.array_equal(left.query_features, right.query_features) or left.channel_names != right.channel_names:
            raise ValueError("Stored heat-null fields do not align by receiver/channel")
        if record.context.values != baseline.context.values or any(a.position_xy != b.position_xy for a, b in zip(record.design.modules, baseline.design.modules)):
            raise ValueError("Stored heat-null control changes prescribed context or geometry")
        delta = right.values - left.values
        rows.append({"record_id": candidate["record_id"],
            "max_abs_change": {name: float(np.max(np.abs(delta[:, index]))) for index, name in enumerate(left.channel_names)},
            "heat_total_change": sum(item.heating for item in record.design.modules) - sum(item.heating for item in baseline.design.modules),
            "physical_reference": "stored analytic-wake/shared-grid benchmark; no new solve"})
    return {"source": str(root.resolve()), "scope": "read-only physics diagnosis on historical excluded0001 family; not selected-panel metrics or training",
        "controls": rows, "new_solver_calls": 0,
        "null_channels": ["u", "v", "p", "omega"], "temperature_zero_target": False}


def fixed_total_transfers(positions, heat, present, *, lower, upper, amplitudes=(.1, .2)):
    """Geometry-selected donor pairs, opposite feasible fixed-total changes."""
    heat = np.asarray(heat, dtype=np.float64)
    positions = np.asarray(positions)
    active = np.flatnonzero(np.asarray(present) > .5)
    if active.size < 2:
        return []
    order = active[np.lexsort((positions[active, 1], positions[active, 0]))]
    pairs = [(int(order[0]), int(order[-1]))]
    if order.size > 2:
        pairs.append((int(order[0]), int(order[1])))
    variants = []
    for pair_number, (first, second) in enumerate(pairs):
        bound = min(heat[first] - lower, upper - heat[first], heat[second] - lower, upper - heat[second])
        if not np.isfinite(bound) or bound <= 0:
            continue
        for fraction in amplitudes:
            if not 0 < fraction <= .5:
                raise ValueError("Response amplitudes must lie in (0,.5] of the feasible symmetric bound")
            for sign in (-1, 1):
                amount = float(sign * fraction * bound)
                value = heat.copy()
                value[first] += amount
                value[second] -= amount
                variants.append({"label": f"pair{pair_number}_a{fraction:g}_{'plus' if sign > 0 else 'minus'}",
                    "donors": [first, second], "fraction_of_feasible_bound": fraction,
                    "signed_heat_transfer": amount, "heat": value})
    return variants


def evaluate_development_heat_null(checkpoint_path, *, dataset_path, output_dir, device="cpu",
                                   query_count=256, amplitudes=(.1, .2), development_manifest=None):
    """All22 unseen-to-current-fit layout null responses, no fabricated labels.

    The known-zero target applies only to benchmark u/v/p/omega. Temperature,
    interface and material sensitivities are saved as model-only quantities.
    Detailed arrays are retained only for the same four input-selected cases.
    """
    import h5py
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.data.development_split import development_case_ids
    from channelthermal.evaluation.loading import load_model
    from thermal_campaign_evaluate import screen_indices
    from thermal_campaign_heat_inference import native_heat_predictor, atomic_json, atomic_npz
    from thermal_development import evaluation_dataset_kwargs, resolve_evaluation_manifest, validate_generated_output

    model, checkpoint = load_model(Path(checkpoint_path), torch.device(device))
    model.eval().requires_grad_(False)
    settings = checkpoint.get("train_config", {}).get("dataset", {})
    manifest = resolve_evaluation_manifest(settings, dataset_path, scope="development", manifest_path=development_manifest)
    if manifest is None:
        raise ValueError("Heat-null development evaluation requires the frozen development manifest")
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint.get("global_normalization_stats", {}).items()}
    if not stats:
        raise ValueError("Development heat-null metrics require saved train-only normalization")
    dataset = GlobalChannelThermalDataset(dataset_path, split="test", points_per_case=1,
        random_point_sampling=False, include_grid=True, normalizer=H5Normalizer(stats),
        **evaluation_dataset_kwargs(manifest, "test"))
    if len(dataset) != 22:
        raise ValueError("Fixed25_v1 heat-null statistics require exactly22 validation cases")
    output_dir = validate_generated_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lower, upper = np.inf, -np.inf
    with h5py.File(dataset_path, "r") as handle:
        for case_id in development_case_ids(manifest, "train"):
            group = handle["cases"][case_id]
            values = np.asarray(group["heat_powers"][...])[np.asarray(group["module_present"][...]) > .5]
            if values.size:
                lower, upper = min(lower, float(values.min())), max(upper, float(values.max()))
    field_names = list(model.config.channelthermal.field_names)
    scales = np.asarray(stats["field_std_by_channel"]).reshape(-1)
    representatives = set(screen_indices(dataset, 4))
    summary = {"checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_epoch": checkpoint.get("epoch"),
        "development_manifest_sha256": manifest["manifest_sha256"], "partition": "fixed22 exposed development validation",
        "query_count": query_count, "amplitudes": amplitudes, "train_heat_range": [lower, upper],
        "physical_reference": "benchmark analytic flow is independent of heat at fixed geometry/context; stored controls verified separately",
        "C_positive_response_transfer": "not measured here; nonzero thermal sensitivities are model-only",
        "pairs_audit": audit_subset_response_pairs(dataset_path, manifest), "cases": []}
    for index in range(len(dataset)):
        sample = dataset[index]
        structure = sample["structure"]
        coordinates = np.stack((np.asarray(sample["x_grid"]).ravel(), np.asarray(sample["y_grid"]).ravel()), -1)
        eligible = np.ones(len(coordinates), dtype=bool)
        radius = float(np.asarray(structure["material_params"])[5])
        for center in np.asarray(structure["module_centers"])[np.asarray(structure["module_present"]) > .5]:
            eligible &= np.linalg.norm(coordinates-center, axis=-1) > radius
        fluid = np.flatnonzero(eligible)
        rows = fluid[np.linspace(0, len(fluid)-1, min(query_count, len(fluid)), dtype=int)]
        lx = float(np.asarray(structure["domain_length_x"]).reshape(-1)[0])
        inlet = np.flatnonzero(eligible & (coordinates[:, 0] <= .08 * lx))
        outlet = np.flatnonzero(eligible & (coordinates[:, 0] >= .92 * lx))
        pressure_rows = np.concatenate((inlet[np.linspace(0, len(inlet)-1, 32, dtype=int)],
                                        outlet[np.linspace(0, len(outlet)-1, 32, dtype=int)]))
        points = coordinates[np.concatenate((rows, pressure_rows))].astype(np.float32)
        predictor, _ = native_heat_predictor(model, checkpoint, sample, points, np.arange(len(rows)), np.arange(len(rows)))
        heat = np.asarray(structure["heat_powers"])
        variants = fixed_total_transfers(structure["module_centers"], heat, structure["module_present"],
            lower=lower, upper=upper, amplitudes=amplitudes)
        case = {"case_id": str(sample["case_id"]), "module_count": int((np.asarray(structure["module_present"]) > .5).sum()),
            "eligible": bool(variants), "wrapper_forward_calls": 0, "variants": [],
            "null_reference_channels": ["u", "v", "p", "omega"], "thermal_reference": "no newly evaluated label"}
        arrays = {"query_xy": points, "baseline_heat": heat, "fluid_grid_rows": rows, "pressure_grid_rows": pressure_rows,
            "channel_names": np.asarray(field_names), "field_scales_train": scales}
        started = perf_counter()
        with torch.no_grad():
            baseline = predictor(torch.as_tensor(heat, device=device, dtype=torch.float32))
            case["wrapper_forward_calls"] += 1
            arrays["baseline_fields"] = baseline["fields"].cpu().numpy()
            arrays["baseline_interface"] = baseline["interface"].cpu().numpy()
            arrays["baseline_solid"] = baseline["solid"].cpu().numpy()
            for variant in variants:
                prediction = predictor(torch.as_tensor(variant["heat"], device=device, dtype=torch.float32))
                case["wrapper_forward_calls"] += 1
                delta = (prediction["fields"] - baseline["fields"]).cpu().numpy()
                rms = np.sqrt(np.mean(delta[:len(rows)]**2, axis=0))
                row = {key: value for key, value in variant.items() if key != "heat"}
                row["channels"] = {name: {"rms_change_native": float(rms[channel]),
                    "native_unit": {"u": "dataset velocity units", "v": "dataset velocity units",
                        "p": "dataset pressure units", "omega": "dataset vorticity units",
                        "temperature": "dataset temperature units"}.get(name, "checkpoint native units"),
                    "rms_change_train_scaled": float(rms[channel] / max(float(scales[channel]), 1e-8)),
                    "reference_increment": 0. if name in {"u", "v", "p", "omega"} else None,
                    "reference_basis": "analytic benchmark known heat-null" if name in {"u", "v", "p", "omega"} else "model-only thermal sensitivity"}
                    for channel, name in enumerate(field_names)}
                pressure_index = field_names.index("p")
                row["pressure_functional_increment"] = float(delta[len(rows):len(rows)+32, pressure_index].mean() - delta[len(rows)+32:, pressure_index].mean())
                row["pressure_functional"] = "32 deterministic fluid rows per maintained inlet/outlet section (x<=.08Lx and x>=.92Lx); section-mean difference"
                row["surface_temperature_rms_change_model_only"] = float((prediction["interface"][:, 0] - baseline["interface"][:, 0]).square().mean().sqrt())
                row["material_temperature_rms_change_model_only"] = float((prediction["solid"] - baseline["solid"]).square().mean().sqrt())
                row["heat_total_error"] = float(np.sum(variant["heat"]) - np.sum(heat))
                case["variants"].append(row)
                if index in representatives:
                    label = variant["label"]
                    arrays[f"{label}/heat"] = variant["heat"]
                    arrays[f"{label}/delta_fields"] = delta
                    arrays[f"{label}/delta_interface"] = (prediction["interface"] - baseline["interface"]).cpu().numpy()
                    arrays[f"{label}/delta_solid"] = (prediction["solid"] - baseline["solid"]).cpu().numpy()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        case["elapsed_seconds"] = perf_counter() - started
        if index in representatives:
            case_dir = output_dir / case["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            atomic_npz(case_dir / "evidence.npz", **arrays)
            case["detail_arrays"] = str(case_dir / "evidence.npz")
        summary["cases"].append(case)
        # First average variants within case, then give each eligible case
        # equal weight. Ineligible cases are visible, never zero successes.
        summary["eligible_case_count"] = sum(bool(item["eligible"]) for item in summary["cases"])
        summary["equal_case_channels"] = {}
        for name in field_names:
            means = [np.mean([row["channels"][name]["rms_change_native"] for row in item["variants"]])
                     for item in summary["cases"] if item["eligible"]]
            summary["equal_case_channels"][name] = {"mean_rms_change_native": float(np.mean(means)) if means else None,
                "p90_case_rms_change_native": float(np.quantile(means, .9)) if means else None,
                "max_case_rms_change_native": float(np.max(means)) if means else None,
                "scope": "known benchmark null" if name in {"u", "v", "p", "omega"} else "model-only nonzero thermal sensitivity"}
        atomic_json(output_dir / "summary.json", summary)
        print(f"{case['case_id']}: saved {len(variants)} bounded fixed-total transfers", flush=True)
    return output_dir


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
    parser.add_argument("--heat-null-development", action="store_true")
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--null-query-count", type=int, default=256)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[.1, .2])
    args = parser.parse_args(argv)
    if args.query_batch_size < 1:
        parser.error("query batch size must be positive")
    if args.stencils is not None and (not args.stencils or len(set(args.stencils)) != len(args.stencils)):
        parser.error("the response panel must contain unique stencils")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.heat_null_development:
        if args.evaluation_scope == "formal-full" or not 1 <= args.null_query_count <= 1024:
            raise ValueError("Bounded heat-null evaluation requires development scope and Q<=1024")
        print(evaluate_development_heat_null(args.checkpoint, dataset_path=args.dataset,
            output_dir=args.output_dir, device=args.device, query_count=args.null_query_count,
            amplitudes=tuple(args.amplitudes), development_manifest=args.development_manifest))
    else:
        print(evaluate_responses(args.checkpoint, dataset_path=args.dataset, stencil_paths=args.stencils,
            output_dir=args.output_dir, device=args.device, query_batch_size=args.query_batch_size,
            evaluation_scope=args.evaluation_scope, panel_config=args.panel_config))
