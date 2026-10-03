#!/usr/bin/env python3
"""Evaluate saved native Thermal checkpoints once and retain physical evidence.

The screen panel is fixed by input-only module count/context strata. Longer
milestones evaluate the complete exposed development split, with both the
89-case sensitivity and 90-case compatibility aggregate. Numerical arrays are
saved locally for figure reuse; this workflow never launches a solver.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

INTERVENTIONS = ("normal", "full_access", "root_union", "control_identity",
                 "geometry_control", "effective_rewire", "source_group_exchange", "fixed_frontier", "fixed_summary",
                 "control_identity_fixed_access", "full_access_fixed_controls", "geometry_reference_actions")
REFERENCE_ACTIONS = {"control_identity_fixed_access": "control_identity",
                     "full_access_fixed_controls": "full_access_fixed_controls",
                     "geometry_reference_actions": "geometry_reference_actions"}


def physical_errors(prediction, reference, mask=None):
    """Retain sufficient statistics so pooled and equal-case scores differ."""
    prediction, reference = np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape:
        raise ValueError(f"Prediction/reference shapes differ: {prediction.shape} vs {reference.shape}")
    if mask is not None:
        prediction, reference = prediction[np.asarray(mask, dtype=bool)], reference[np.asarray(mask, dtype=bool)]
    difference = (prediction - reference).reshape(-1)
    if not difference.size:
        return {"count": 0, "finite": True, "rmse": None, "mae": None,
                "max_abs": None, "squared_error_sum": 0., "absolute_error_sum": 0.}
    if not np.isfinite(difference).all():
        return {"count": int(difference.size), "finite": False, "rmse": None, "mae": None,
                "max_abs": None, "squared_error_sum": None, "absolute_error_sum": None}
    squared, absolute = float(difference @ difference), float(np.abs(difference).sum())
    return {"count": int(difference.size), "finite": True, "rmse": float(np.sqrt(squared / difference.size)),
            "mae": absolute / difference.size, "max_abs": float(np.abs(difference).max()),
            "squared_error_sum": squared, "absolute_error_sum": absolute}


def physical_case_metrics(sample, predictions, channel_order):
    from channelthermal.evaluation_tools.plots import module_and_fluid_masks, module_radius_from_sample
    field = np.asarray(predictions["pred_field_grid"])
    reference = np.asarray(sample["steady_field"])[..., :field.shape[-1]]
    _, fluid = module_and_fluid_masks(sample, field)
    present = np.asarray(sample["structure"]["module_present"]) > .5
    centres = np.asarray(sample["structure"]["module_centers"])[present]
    radius = module_radius_from_sample(sample)
    distance = np.full(fluid.shape, np.inf)
    for centre in centres:
        distance = np.minimum(distance, np.hypot(sample["x_grid"] - centre[0], sample["y_grid"] - centre[1]))
    near, far = fluid & (distance <= 2 * radius), fluid & (distance > 2 * radius)
    metrics = {}
    for index, name in enumerate(channel_order[:field.shape[-1]]):
        for region, mask in (("fluid", fluid), ("near", near), ("far", far)):
            metrics[f"{region}/{name}"] = physical_errors(field[..., index], reference[..., index], mask)
    interface = np.asarray(predictions["pred_interface"])
    if interface.size:
        for index, name in enumerate(("surface_temperature", "q_normal_proxy")):
            metrics[name] = physical_errors(interface[..., index], np.asarray(sample["interface_target"])[..., index], present)
    internal = np.asarray(predictions["pred_internal_temperature"])
    target_internal = np.asarray(sample["module_internal_temperature_points"])
    if internal.size:
        if internal.ndim == target_internal.ndim + 1 and internal.shape[-1] == 1:
            internal = internal[..., 0]
        metrics["material_temperature"] = physical_errors(internal, target_internal, present)
        metrics["module_material_peak"] = physical_errors(internal.max(-1), target_internal.max(-1), present)
    ports = np.asarray(sample["teacher_port_tokens"])
    for stage, key in (("initial", "pred_port_condition_raw"), ("final", "pred_port_condition")):
        predicted = np.asarray(predictions.get(key, []))
        if predicted.size:
            for index, name in ((3, "outside_temperature"), (4, "h_effective")):
                metrics[f"{stage}_port/{name}"] = physical_errors(predicted[..., index], ports[..., index], present)
    if "p" in channel_order:
        index = channel_order.index("p")
        x = np.asarray(sample["x_grid"])
        inlet, outlet = fluid & np.isclose(x, x.min()), fluid & np.isclose(x, x.max())
        if inlet.any() and outlet.any():
            delta = np.asarray([field[..., index][inlet].mean() - field[..., index][outlet].mean()])
            target_delta = np.asarray([reference[..., index][inlet].mean() - reference[..., index][outlet].mean()])
            metrics["inlet_outlet_pressure_difference"] = physical_errors(delta, target_delta)
    return metrics, {"fluid_mask": fluid, "near_mask": near, "far_mask": far}


def aggregate_physical(rows):
    """Never combine different physical channels into a scalar score."""
    output = {}
    for key in sorted({key for row in rows for key in row["metrics"]}):
        values = [row["metrics"][key] for row in rows if key in row["metrics"]]
        valid = [value for value in values if value["finite"] and value["count"] > 0]
        count = sum(value["count"] for value in valid)
        errors = np.asarray([value["rmse"] for value in valid])
        output[key] = {"cases": len(valid), "nonfinite_cases": sum(not value["finite"] for value in values),
            "empty_cases": sum(value["count"] == 0 for value in values),
            "equal_case_rmse_mean": float(errors.mean()) if errors.size else None,
            "equal_case_mae_mean": float(np.mean([value["mae"] for value in valid])) if valid else None,
            "case_rmse_p90": float(np.quantile(errors, .9)) if errors.size else None,
            "case_rmse_max": float(errors.max()) if errors.size else None,
            "pooled_rmse": float(np.sqrt(sum(value["squared_error_sum"] for value in valid) / count)) if count else None,
            "pooled_mae": sum(value["absolute_error_sum"] for value in valid) / count if count else None,
            "pooled_count": count}
    return output


def screen_indices(dataset, count=18):
    """Round-robin M strata and span supported input context; no field targets."""
    strata = {}
    for index, case_id in enumerate(dataset.selected_case_ids):
        if str(case_id).lstrip("0") == "273":
            continue
        group = dataset.h5["cases"][case_id]
        modules = int((np.asarray(group["module_present"]) > .5).sum())
        material = group.get("material_parameters")
        context = (float(material.attrs.get("u_in", 0)) if material is not None else 0,
                   float(np.asarray(group["heat_powers"]).sum()), str(case_id))
        strata.setdefault(modules, []).append((context, index))
    for modules, entries in strata.items():
        entries.sort()
        # Endpoints first, then progressively cover each context-sorted stratum.
        order = [0] if len(entries) == 1 else [0, len(entries) - 1]
        gaps = [(0, len(entries) - 1)]
        while gaps:
            left, right = max(gaps, key=lambda interval: (interval[1] - interval[0], -interval[0]))
            gaps.remove((left, right))
            if right - left > 1:
                middle = (left + right) // 2
                order.append(middle)
                gaps.extend(((left, middle), (middle, right)))
        strata[modules] = [entries[position][1] for position in order]
    selected = []
    while len(selected) < count and any(strata.values()):
        for modules in sorted(strata):
            if strata[modules] and len(selected) < count:
                selected.append(strata[modules].pop(0))
    return selected


def fixed_screen_indices(dataset, panel_path, count):
    """Replay the input-only panel used by the first completed screen."""
    specification = json.loads(Path(panel_path).read_text())
    if specification["split"] != dataset.split:
        raise ValueError("Fixed campaign panel belongs to a different partition")
    case_ids = specification["case_ids"]
    if len(set(case_ids)) != len(case_ids) or any(str(value).lstrip("0") == "273" for value in case_ids):
        raise ValueError("Fixed screen IDs repeat or contain the known development duplicate")
    lookup = {str(case_id): index for index, case_id in enumerate(dataset.selected_case_ids)}
    if any(case_id not in lookup for case_id in case_ids):
        raise ValueError("Fixed campaign panel is missing from this dataset")
    if count > len(case_ids):
        raise ValueError("Requested fixed screen exceeds its predeclared size")
    return [lookup[case_id] for case_id in case_ids[:count]]


def evaluation_indices(dataset, args):
    """Bound reference-only controls at every stage; retain mature full screens."""
    reference_only = all(name in REFERENCE_ACTIONS for name in args.interventions)
    if args.stage == 100 or reference_only:
        return (fixed_screen_indices(dataset, args.panel_config, args.panel_size)
                if args.panel_config is not None else screen_indices(dataset, args.panel_size))
    return list(range(len(dataset)))


def intervention_effectiveness(anchor_changes, native_comparison=None):
    """Prefer actual native streams; separately label the P2 anchor diagnostic."""
    anchor_effective = any(anchor_changes.values())
    if native_comparison is None:
        return {"effective_anchor_pair_intervention": anchor_effective,
                "effective_pair_intervention": anchor_effective,
                "effective_pair_intervention_scope": "P2 anchor support only; native streams unmeasured"}
    return {"effective_anchor_pair_intervention": anchor_effective,
            "effective_pair_intervention": any(row["changed_pairs"] > 0 for row in native_comparison.values()),
            "effective_weight_intervention": any(row["max_absolute_weight_change"] > 0 for row in native_comparison.values()),
            "effective_pair_intervention_scope": "Recorded actual native P0/P1/P2 receiver streams; anchors reported separately"}


@contextmanager
def intervention(model, name):
    """Same physical weights; preserve degree/weight and report actual pair work."""
    backend = model.core.backend
    if not hasattr(backend, "organizer"):
        if name != "normal":
            raise ValueError("Organizer intervention requires a typed candidate checkpoint")
        yield
        return
    old_mode = backend.plan_intervention
    aliases = {"geometry_control": "geometry", "effective_rewire": "rewire",
               "source_group_exchange": "exchange", "fixed_frontier": "fixed_structure",
               "control_identity_fixed_access": "control_identity", "full_access_fixed_controls": "normal",
               "geometry_reference_actions": "normal"}
    backend.set_plan_intervention(aliases.get(name, name))
    try:
        yield
    finally:
        backend.set_plan_intervention(old_mode)


def _arrays(prefix, value, output):
    if torch.is_tensor(value):
        output[prefix] = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        output[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            _arrays(f"{prefix}/{key}", child, output)


def calibrate_training_summary(model, checkpoint, dataset_path, *, count=600):
    """Collect P0/P1/P2 controls from training physical inputs, without labels."""
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from thermal_campaign_heat_inference import native_heat_predictor, sensor_panel

    from honf_forward_core.interface_fields.training_population_summary import TrainingPopulationSummaryAccumulator
    dataset = GlobalChannelThermalDataset(dataset_path, split="train", points_per_case=1,
        random_point_sampling=False, include_grid=True)
    organizer = model.core.backend.organizer
    accumulator = TrainingPopulationSummaryAccumulator()
    original = organizer.prepare
    selected = list(range(len(dataset))) if count >= len(dataset) else screen_indices(dataset, count)

    def collect(*args, **kwargs):
        state = original(*args, **kwargs)
        accumulator.add(state, partition=dataset.split)
        return state

    organizer.prepare = collect
    try:
        with torch.no_grad():
            for index in selected:
                sample = dataset[index]
                sensors, _, _, observed_rows, held_rows = sensor_panel(sample)
                predictor, _ = native_heat_predictor(model, checkpoint, sample, sensors, observed_rows, held_rows)
                heat = torch.as_tensor(sample["structure"]["heat_powers"], device=next(model.parameters()).device)
                predictor(heat)
    finally:
        organizer.prepare = original
    return accumulator.finish(), [str(dataset.selected_case_ids[index]) for index in selected]


def evaluate(args):
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model
    from channelthermal.evaluation.prepared import predict_case
    from channelthermal.evaluation.results import denormalize_predictions

    from honf_runtime.compat import resolve_demo_path
    checkpoint_path = args.checkpoint.expanduser().resolve()
    device = torch.device(args.device)
    model, checkpoint = load_model(checkpoint_path, device)
    executor = getattr(model.core.backend, "set_execution_mode", None)
    if callable(executor):
        executor(args.executor, receiver_chunk_size=args.executor_receiver_chunk)
    elif args.executor != "dense_masked_reference":
        raise ValueError("The rectangular subset executor is supported by typed campaign candidates only")
    dataset_config = checkpoint.get("train_config", {}).get("dataset", {})
    dataset_path = resolve_demo_path(args.dataset or dataset_config["packed_h5_path"])
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint.get("global_normalization_stats", {}).items()}
    normalized = GlobalChannelThermalDataset(dataset_path, split=args.split, points_per_case=1,
        random_point_sampling=False, include_grid=True, normalizer=H5Normalizer(stats) if stats else None,
        normalize_inputs=bool(dataset_config.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_config.get("normalize_targets", False)))
    raw = GlobalChannelThermalDataset(dataset_path, split=args.split, points_per_case=1,
        random_point_sampling=False, include_grid=True)
    if normalized.selected_case_ids != raw.selected_case_ids:
        raise ValueError("Normalized and physical dataset indices differ")
    indices = evaluation_indices(raw, args)
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(PROJECT_ROOT) and not any(output.is_relative_to(PROJECT_ROOT / root) for root in ("diagnostics", "Trained_Results")):
        raise ValueError("Generated evaluation evidence must live in ignored diagnostics or Trained_Results")
    output.mkdir(parents=True, exist_ok=True)
    reference_access_dir = getattr(args, "reference_access_dir", None)
    if any(name in REFERENCE_ACTIONS for name in args.interventions):
        reference_access_dir = reference_access_dir.expanduser().resolve()
        reference_summary = json.loads((reference_access_dir / "summary.json").read_text())
        if (Path(reference_summary["checkpoint"]).resolve() != checkpoint_path
                or reference_summary["checkpoint_epoch"] != checkpoint["epoch"]
                or reference_summary["split"] != args.split
                or Path(reference_summary["dataset"]).resolve() != Path(dataset_path).resolve()):
            raise ValueError("Reference access checkpoint/dataset/partition identity mismatch")
    if "fixed_summary" in args.interventions:
        if not hasattr(model.core.backend, "organizer"):
            raise ValueError("Universal fixed-summary intervention requires a typed candidate")
        summary, calibration_ids = calibrate_training_summary(model, checkpoint, dataset_path,
            count=args.fixed_summary_train_cases)
        model.core.backend.training_population_summary = summary
        controls = {f"P{phase}/{tau}": value.cpu().numpy() for phase, values in summary.controls.items()
                    for tau, value in values.items()}
        np.savez_compressed(output / "training_summary_controls.npz", **controls)
        (output / "training_summary_calibration.json").write_text(json.dumps({
            "partition": "train", "case_ids": calibration_ids, "phase_case_counts": summary.case_counts,
            "source_bearing_case_counts": summary.source_bearing_case_counts,
            "operator": "universal root, uniform valid-source membership, fixed typed controls; fine values/geometry remain live",
            "work_matching": "not work matched; universal full-access work reported"}, indent=2) + "\n")
    rows = []
    channel_order = list(raw.channel_order)
    if args.panel_config is not None:
        fixed_size = len(json.loads(args.panel_config.read_text())["case_ids"])
        graph_indices = fixed_screen_indices(raw, args.panel_config, min(args.graph_panel_size, fixed_size))
        if args.graph_panel_size > fixed_size:
            graph_indices = list(dict.fromkeys(
                [*graph_indices, *screen_indices(raw, len(raw))]))[:args.graph_panel_size]
    else:
        graph_indices = screen_indices(raw, min(args.graph_panel_size, len(raw)))
    graph_panel = set(graph_indices)
    for index in indices:
        sample, reference = normalized[index], raw[index]
        case_id = str(reference["case_id"])
        for name in (args.interventions if index in graph_panel else ("normal",)):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            phase_capture = None
            if getattr(args, "capture_phase_graphs", False):
                from honf_forward_core.evaluation.typed_work_evidence import TypedWorkEvidenceRecorder
                phase_capture = TypedWorkEvidenceRecorder(model.core.backend)
            reference_capture = None
            if name in REFERENCE_ACTIONS:
                from honf_forward_core.evaluation.reference_access import FixedReferenceAccessReplay
                reference_capture = FixedReferenceAccessReplay(model.core.backend,
                    reference_access_dir / case_id / "normal" / "phase_graphs.npz", mode=REFERENCE_ACTIONS[name])
            with intervention(model, name):
                with reference_capture if reference_capture is not None else nullcontext(), \
                     phase_capture if phase_capture is not None else nullcontext():
                    prediction = predict_case(model, sample, device, query_batch_size=args.query_batch_size,
                        local_port_condition_mode="predicted", mixed_teacher_ratio=0,
                        return_routing_maps=True, return_prepared_state=index in graph_panel)
                graph = None
                if index in graph_panel and hasattr(model.core.backend, "organizer"):
                    graph = model.export_typed_hypergraph(prediction["_prepared_state"])
                    prepared = prediction["_prepared_state"].prepared
                    plan = prepared.backend_state["hypergraph_plan"]
                    graph_accesses = {}
                    for tau in plan.memberships:
                        kind = "M" if tau in ("MM", "ME") else "E" if tau == "EM" else None
                        receivers = plan.source_coords[kind] if kind else (prepared.encoded.receiver_anchor_coords
                            if prepared.encoded.receiver_anchor_coords is not None else prepared.encoded.env_coords)
                        graph_accesses[tau] = graph["receiver_access"](receivers, tau)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = perf_counter() - start
            prediction = denormalize_predictions(prediction, normalized, bool(dataset_config.get("normalize_targets", False)))
            metrics, masks = physical_case_metrics(reference, prediction, channel_order)
            directory = output / case_id / name
            directory.mkdir(parents=True, exist_ok=True)
            arrays = {**masks, "x_grid": reference["x_grid"], "y_grid": reference["y_grid"],
                "reference_field": reference["steady_field"], "prediction_field": prediction["pred_field_grid"],
                "residual_field": prediction["pred_field_grid"] - reference["steady_field"][..., :len(channel_order)],
                "reference_internal": reference["module_internal_temperature_points"],
                "reference_interface": reference["interface_target"], "reference_ports": reference["teacher_port_tokens"]}
            _arrays("input", reference["structure"], arrays)
            for key in ("pred_internal_temperature", "pred_interface", "pred_port_condition", "pred_port_condition_raw"):
                arrays[key] = np.asarray(prediction[key])
            if graph is not None:
                _arrays("graph", graph, arrays)
                prepared = prediction["_prepared_state"].prepared
                plan = prepared.backend_state["hypergraph_plan"]
                _arrays("graph/typed_admission", plan.strategy_data.get("typed_admission", {}), arrays)
                _arrays("graph/typed_centres", plan.strategy_data.get("typed_centres", {}), arrays)
                pair_masks = {}
                for tau, access in graph_accesses.items():
                    _arrays(f"graph/{tau}_anchor_weight", access.weight, arrays)
                    _arrays(f"graph/{tau}_anchor_access", access.edge_access, arrays)
                    pair_masks[tau] = access.support.detach().cpu().numpy()
                _arrays("graph/anchor_support", pair_masks, arrays)
            np.savez_compressed(directory / "evidence.npz", **arrays)
            work = {key: float(np.asarray(value)) for key, value in prediction.get("interaction_aux", {}).items()
                    if "hypergraph" in key and np.asarray(value).ndim == 0 and np.issubdtype(np.asarray(value).dtype, np.number)}
            row = {"case_id": case_id, "module_count": int((reference["structure"]["module_present"] > .5).sum()),
                "intervention": name, "metrics": metrics, "complete_wrapper_seconds": seconds,
                "work": work,
                "work_scope": "Legacy prediction interaction_aux aggregated across external field query chunks; not summed complete-wrapper executor work",
                "arrays": str(directory / "evidence.npz"), "graph_phase": 2 if graph is not None else None}
            if phase_capture is not None:
                phase_path = directory / "phase_graphs.npz"
                np.savez_compressed(phase_path, **phase_capture.arrays)
                row["phase_graph_arrays"] = str(phase_path)
                row["phase_graph_scope"] = "P0/P1/P2 post-intervention source plans and actual native prepare/read receiver streams; no extra fine reads"
                row["complete_wrapper_seconds_scope"] = "Includes phase-recording copies and anchor reconstruction; not uninstrumented speed evidence"
                if name != "normal":
                    from honf_forward_core.evaluation.typed_work_evidence import compare_native_access
                    comparison_directory = reference_access_dir if reference_capture is not None else output
                    with np.load(comparison_directory / case_id / "normal" / "phase_graphs.npz", allow_pickle=False) as baseline:
                        row["actual_native_pair_intervention"] = compare_native_access(baseline, phase_capture.arrays)
                    row.update(intervention_effectiveness({}, row["actual_native_pair_intervention"]))
            if reference_capture is not None:
                row["reference_permission_source"] = str(reference_access_dir / case_id / "normal" / "phase_graphs.npz")
                row["phase_plan_scope"] = "Current conditional organizer plans recomputed from live physical state; authoritative native access permissions replay saved normal plans"
                row["reference_permission_scope"] = {
                    "control_identity_fixed_access": "All P0/P1/P2 native access density/weight/support/edge_access/near fixed; controls and gain/score biases disabled; physical source values/ports/local physics remain live",
                    "full_access_fixed_controls": "All P0/P1/P2 eligible density/weight/support full and uniform; normal source controls and projection biases retained; physical source values/ports/local physics remain live",
                    "geometry_reference_actions": "All P0/P1/P2 normal joint permission/control tuples reassigned toward physical geometry at fixed receiver/source binary degrees and row tuple multisets; normal projection biases and physical source values/ports/local physics remain live",
                }[name]
                row["reference_group_access_scope"] = "Saved edge_access is reference group provenance; authoritative returned source permissions may be reassigned independently of shared groups"
                row["source_resolved_action_statistics"] = reference_capture.action_statistics
                row["reference_call_count"] = len(reference_capture.seen)
                row["reference_density_scope"] = "Missing historical density reconstructed from saved normal membership/access/measures/validity/near with normalized weight/support validation"
                row["reference_density_max_weight_reconstruction_error"] = reference_capture.density_reconstruction_max_weight_error
            if name != "normal" and graph is not None:
                normal_path = output / case_id / "normal" / "evidence.npz"
                changed = {}
                if normal_path.exists():
                    with np.load(normal_path, allow_pickle=False) as baseline:
                        for tau, support in pair_masks.items():
                            key = f"graph/anchor_support/{tau}"
                            changed[tau] = int(np.count_nonzero(support != baseline[key]))
                row["changed_anchor_pairs"] = changed
                row.update(intervention_effectiveness(changed, row.get("actual_native_pair_intervention")))
                if normal_path.exists():
                    baseline_row = json.loads((normal_path.parent / "metrics.json").read_text())
                    row["work_delta_from_normal"] = {key: value - baseline_row["work"][key]
                        for key, value in work.items() if key in baseline_row["work"]}
                    pairs = {key: delta for key, delta in row["work_delta_from_normal"].items() if key.endswith("unique_pairs")}
                    row["unique_pair_work_matched"] = bool(pairs) and all(delta == 0 for delta in pairs.values())
                    with np.load(normal_path, allow_pickle=False) as baseline:
                        row["source_group_incidence_comparison"] = {}
                        for tau in plan.memberships:
                            key = f"graph/source_membership/{tau}"
                            before, after = baseline[key], arrays[key]
                            kind = "M" if tau in ("MM", "EM", "QM") else "E"
                            mu = arrays[f"graph/source_measures/{kind}"][:, None]
                            admitted_key = f"graph/typed_admission/{tau}"
                            before_admission = baseline[admitted_key] if admitted_key in baseline else baseline["graph/group_admission"]
                            after_admission = arrays.get(admitted_key, arrays["graph/group_admission"])
                            row["source_group_incidence_comparison"][tau] = {
                                "normal_positive_incidence": int(np.count_nonzero(before)),
                                "intervention_positive_incidence": int(np.count_nonzero(after)),
                                "normal_admitted_positive_incidence": int(np.count_nonzero(before * (before_admission > 0)[..., None])),
                                "intervention_admitted_positive_incidence": int(np.count_nonzero(after * (after_admission > 0)[..., None])),
                                "membership_multiset_equal": bool(np.allclose(np.sort(before, axis=-1), np.sort(after, axis=-1))),
                                "all_group_membership_multiset_equal": bool(np.allclose(np.sort(before.reshape(before.shape[0], -1), axis=-1),
                                                                                         np.sort(after.reshape(after.shape[0], -1), axis=-1))),
                                "normal_group_mass": (before * mu).sum(-1).tolist(),
                                "intervention_group_mass": (after * mu).sum(-1).tolist()}
            rows.append(row)
            (directory / "metrics.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            write_summary(output, rows, checkpoint_path, checkpoint, args, dataset_path, indices, channel_order)
            print(f"{case_id} {name}: saved complete field and physical metrics ({seconds:.2f}s)", flush=True)
    if args.inverse_cases:
        from thermal_campaign_heat_inference import evaluate_heat
        evaluate_heat(checkpoint_path, dataset_path=dataset_path, output_dir=output / "heat_inference",
            device=args.device, cases=args.inverse_cases, starts=args.inverse_starts, steps=args.inverse_steps)
    return output


def write_summary(output, rows, checkpoint_path, checkpoint, args, dataset_path, indices, channel_order):
    normal = [row for row in rows if row["intervention"] == "normal"]
    primary = [row for row in normal if row["case_id"].lstrip("0") != "273"]
    payload = {"checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint.get("epoch"),
        "requested_stage": args.stage, "dataset": str(dataset_path), "split": args.split,
        "evidence_partition": "previously exposed development", "channel_order": channel_order,
        "channel_units": "benchmark physical scales; units must be read from generator metadata before dimensional claims",
        "reference_limit": "stored analytic-wake/shared-grid benchmark; q_normal is a proxy; no new physical solves",
        "fine_executor": args.executor,
        "cpu_threads": torch.get_num_threads(),
        "executor_scope": "fine message/geometry rows and attention cells; policy moments/control projections remain dense",
        "port_mode": "predicted", "near_definition": "fluid within two native module radii",
        "planned_case_indices": indices, "completed_normal_cases": len(normal),
        "fixed_screen_configuration": str(args.panel_config) if args.panel_config is not None else None,
        "primary_excluding_0273": aggregate_physical(primary), "compatibility_including_0273": aggregate_physical(normal),
        "unavailable_interventions": ["independently_retrained_fixed_K"],
        "rows": rows}
    (output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--stage", type=int, choices=(100, 500, 1000), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-size", type=int, default=18)
    parser.add_argument("--panel-config", type=Path,
                        default=PROJECT_ROOT / "src/config_core/evaluation/thermal_campaign_e100_panel.json")
    parser.add_argument("--graph-panel-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--executor", choices=("dense_masked_reference", "rectangular_subset"),
                        default="dense_masked_reference")
    parser.add_argument("--executor-receiver-chunk", type=int, default=128)
    parser.add_argument("--interventions", nargs="+", choices=INTERVENTIONS, default=["normal"])
    parser.add_argument("--inverse-cases", type=int, default=0)
    parser.add_argument("--inverse-starts", type=int, default=3)
    parser.add_argument("--inverse-steps", type=int, default=30)
    parser.add_argument("--fixed-summary-train-cases", type=int, default=600)
    parser.add_argument("--capture-phase-graphs", action="store_true",
                        help="Save source plans and actual physical prepare/read access at every phase")
    parser.add_argument("--reference-access-dir", type=Path,
                        help="Saved normal evaluation directory for fixed-access identity isolation")
    args = parser.parse_args(argv)
    if min(args.panel_size, args.graph_panel_size, args.query_batch_size,
           args.fixed_summary_train_cases, args.executor_receiver_chunk) < 1:
        parser.error("panel and query sizes must be positive")
    reference_only = all(name in REFERENCE_ACTIONS for name in args.interventions)
    if (args.interventions[0] != "normal" and not reference_only) or len(set(args.interventions)) != len(args.interventions):
        parser.error("interventions must start with normal and must not repeat")
    if any(name in REFERENCE_ACTIONS for name in args.interventions) and (args.reference_access_dir is None or not args.capture_phase_graphs):
        parser.error("reference actions require --reference-access-dir and --capture-phase-graphs")
    if reference_only and (args.panel_size > args.graph_panel_size or args.inverse_cases):
        parser.error("reference-only utility requires all cases in the graph panel and no inverse run")
    if args.inverse_cases < 0 or min(args.inverse_starts, args.inverse_steps) <= 0:
        parser.error("inverse case count must be nonnegative and starts/steps positive")
    return args


if __name__ == "__main__":
    print(evaluate(parse_args()))
