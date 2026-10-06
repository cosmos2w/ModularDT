#!/usr/bin/env python3
"""Bounded native response reads and saved-evidence summaries, with no solver.

The command reads existing atlas families or counted fixed-panel records. It
never fits a model, creates a design, or fills in a missing physical baseline.
All deltas are formed after widening predictions and references to float64.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

CORE_FIELDS = ("fluid/temperature", "surface_temperature", "material_temperature", "fluid/u", "fluid/p")
FIXED4 = ("0277", "0291", "0294", "0687")
PRIMARY = ("0291", "0294", "0687")
ROLES = ("fluid_fields", "interface", "solid_temperature")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def finite_metrics(prediction, reference, role, *, baseline_prediction=None, baseline_reference=None):
    """Native quadrature metrics; unknown physical sign floors stay unknown."""
    from thermal_campaign_responses import response_channel_metrics

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    if (baseline_prediction is None) != (baseline_reference is None):
        raise ValueError("Finite changes require both model and physical baselines")
    if baseline_prediction is not None:
        pred = pred - np.asarray(baseline_prediction, dtype=np.float64)
        truth = truth - np.asarray(baseline_reference, dtype=np.float64)
    result = response_channel_metrics(pred, truth, role.valid_mask, role.quadrature_weights,
                                      role.channel_names, role.channel_units)
    for channel, name in enumerate(role.channel_names):
        valid = role.valid_mask[:, channel] & (role.quadrature_weights > 0)
        if not valid.any():
            continue
        weight = role.quadrature_weights[valid].astype(np.float64)
        weight /= weight.sum()
        result[name].update(reference_signed_mean=float(weight @ truth[valid, channel]),
                            prediction_signed_mean=float(weight @ pred[valid, channel]),
                            prediction_rms=float(np.sqrt(weight @ pred[valid, channel] ** 2)),
                            max_abs_error=float(np.max(np.abs(pred[valid, channel] - truth[valid, channel]))),
                            sign_floor_status="No certified physical sign or grid-error floor")
    return result, pred, truth


def native_functionals(record, values):
    """True material-grid maxima and full native 8%-band pressure functional."""
    fluid = record.output.roles["fluid_fields"]
    pressure_channel = fluid.channel_names.index("p")
    valid = fluid.valid_mask[:, pressure_channel]
    x = fluid.query_features[:, 0]
    length = record.context.values["domain_length_x"]
    inlet, outlet = valid & (x <= .08 * length), valid & (x >= .92 * length)
    if not inlet.any() or not outlet.any():
        raise ValueError("Pressure evaluation requires both full native section supports")
    pressure = np.asarray(values["fluid_fields"], dtype=np.float64)[:, pressure_channel]
    material = record.output.roles["solid_temperature"]
    receivers = np.asarray(material.receiver_module_ids)
    temperature = np.asarray(values["solid_temperature"], dtype=np.float64)[:, 0]
    peaks = {}
    for module_id in record.design.active_module_ids:
        mask = (receivers == module_id) & material.valid_mask[:, 0]
        if not mask.any() or not np.isfinite(temperature[mask]).all():
            raise ValueError(f"Missing/nonfinite native material grid for {module_id}")
        peaks[module_id] = float(temperature[mask].max())
    return {"pressure_drop_8pct": float(pressure[inlet].mean() - pressure[outlet].mean()),
            "module_peak_temperature": peaks, "maximum_material_temperature": max(peaks.values())}


def validate_heat_records(records):
    """Every heat-only endpoint must share the exact physical receiver join."""
    first = next(iter(records.values()))
    for record in records.values():
        if record.design.active_module_ids != first.design.active_module_ids or dict(record.context.values) != dict(first.context.values):
            raise ValueError("Response endpoints have different module identities or context")
        if [module.position_xy for module in record.design.modules] != [module.position_xy for module in first.design.modules]:
            raise ValueError("Heat response endpoints change geometry")
        if not np.isclose(sum(module.heating for module in record.design.modules),
                          sum(module.heating for module in first.design.modules), rtol=0, atol=2e-6):
            raise ValueError("Heat response endpoints do not preserve total heating")
        for name, left in first.output.roles.items():
            right = record.output.roles[name]
            for key in ("query_features", "valid_mask", "quadrature_weights"):
                if not np.array_equal(getattr(left, key), getattr(right, key)):
                    raise ValueError(f"Heat response receiver {key} differs for {name}")
            for key in ("query_ids", "receiver_module_ids", "channel_names", "channel_units"):
                if getattr(left, key) != getattr(right, key):
                    raise ValueError(f"Heat response receiver {key} differs for {name}")


def prepare_counted_replay(families):
    """Predeclare fixed pools/objective and numerical warnings before predictions."""
    result = {"objective": "lowest true maximum material temperature on native module grids",
              "pool_policy": "exact existing baseline/minus/plus; 0277 two-point secondary only",
              "physical_reference_attempts_cumulative": 326, "physical_reference_allowance": 326,
              "new_solver_attempts": 0, "new_model_calls": 0, "families": []}
    for case_id, records in families:
        validate_heat_records(records)
        maxima = {}
        warnings = []
        for label, record in records.items():
            values = {name: role.values for name, role in record.output.roles.items()}
            maxima[label] = native_functionals(record, values)["maximum_material_temperature"]
            warning = record.provenance.get("runtime", {}).get("final_delta_inf")
            if warning is not None:
                warnings.append(abs(float(warning)))
        warning_scale = 2 * max(warnings) if warnings else 0.0
        result["families"].append({"case_id": case_id, "candidate_states": sorted(records),
            "scope": "primary stored-pool decision" if case_id in PRIMARY else "secondary two-point replay",
            "reference_maximum_material_temperature": maxima, "warning_scale": warning_scale,
            "warning_scale_status": "Twice maximum final iterate delta; descriptive warning, not certified grid error",
            "source": "existing counted analytic-wake/shared-grid records", "native_receiver_join_verified": True})
    return result


def stored_pool_decision(reference, prediction, *, warning_scale=0.0):
    """Rank one identical finite pool; leave small reference gaps unresolved."""
    if set(reference) != set(prediction) or len(reference) < 2:
        raise ValueError("Stored candidate pools must be identical and contain at least two states")
    if warning_scale < 0 or not np.isfinite(warning_scale):
        raise ValueError("Decision warning scale must be finite and nonnegative")
    if not all(np.isfinite(value) for value in (*reference.values(), *prediction.values())):
        raise ValueError("Stored decisions require finite native material maxima")
    reference_order = sorted(reference, key=lambda name: (reference[name], name))
    predicted_order = sorted(prediction, key=lambda name: (prediction[name], name))
    best = reference_order[0]
    gap = float(reference[reference_order[1]] - reference[best])
    tied = [name for name in reference_order if reference[name] - reference[best] <= warning_scale]
    resolved = len(tied) == 1
    chosen = predicted_order[0]
    return {"candidate_states": sorted(reference), "prediction_ranking": predicted_order,
            "reference_ranking": reference_order, "predicted_best_state": chosen,
            "measured_best_state": best if resolved else None, "unresolved_reference_best_states": [] if resolved else tied,
            "reference_ranking_gap": gap, "realized_benchmark_regret": float(reference[chosen] - reference[best]),
            "choice_correct": chosen == best if resolved else None, "warning_scale": warning_scale,
            "warning_scale_status": "Descriptive final-iterate warning only; not certified grid accuracy",
            "objective": "lowest true maximum material temperature on native module grids",
            "scope": "finite already-solved candidate replay; no inverse search or generator"}


def audit_field_comparability(reference, candidate):
    """Reject cohort/role/mask denominators that cannot support a field dashboard."""
    keys = ("dataset", "split", "channel_order", "dataset_scope", "development_manifest_binding", "port_mode")
    if any(reference.get(key) != candidate.get(key) for key in keys):
        raise ValueError("Field dashboard dataset, normalization membership, channels or port policy differ")
    left = {row["case_id"]: row for row in reference["rows"] if row["intervention"] == "normal"}
    right = {row["case_id"]: row for row in candidate["rows"] if row["intervention"] == "normal"}
    if set(left) != set(right) or len(left) != 22:
        raise ValueError("Field dashboard must use the same exact 22 development cases")
    for case_id, a in left.items():
        b = right[case_id]
        if a["module_count"] != b["module_count"]:
            raise ValueError("Field dashboard module strata differ")
        if {name: row["count"] for name, row in a["metrics"].items()} != {name: row["count"] for name, row in b["metrics"].items()}:
            raise ValueError("Field dashboard native support denominators differ")
    return {"exact_cohort_case_ids": sorted(left), "normal_cases": len(left), "role_counts_identical": True,
            "manifest_fingerprint": reference["development_manifest_binding"]["manifest_sha256"]}


def field_dashboard(summaries):
    reference = next(iter(summaries.values()))
    result = {"scope": "22 repeatedly exposed fixed25_v1 development cases; equal-case native RMSE", "models": {}}
    for name, summary in summaries.items():
        audit = audit_field_comparability(reference, summary)
        result["models"][name] = {"checkpoint": summary["checkpoint"], "checkpoint_epoch": summary["checkpoint_epoch"],
                                  "cohort_audit": audit, "all_roles": summary["primary_excluding_0273"],
                                  "core_fields": {key: summary["primary_excluding_0273"][key] for key in CORE_FIELDS},
                                  "physical_strata": summary["physical_strata"]}
    return result


def aggregate_response_rows(rows):
    """Count layouts separately from opposite directions and macro-average RMSE."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["role"], row["channel"])].append(row)
    result = []
    for (role, channel), values in sorted(groups.items()):
        identity = [(row["case_id"], row["state"]) for row in values]
        if len(set(identity)) != len(identity):
            raise ValueError("Duplicate family-direction response entry")
        result.append({"role": role, "channel": channel, "layout_count": len({row["case_id"] for row in values}),
                       "finite_direction_count": len(values), "macro_mean_state_rmse": float(np.mean([row["rmse"] for row in values])),
                       "equal_state_pooled_rmse": float(np.sqrt(np.mean([row["rmse"] ** 2 for row in values]))),
                       "macro_zero_change_rmse": float(np.mean([row["reference_rms"] for row in values])),
                       "unit": values[0]["unit"]})
    return result


def summarize_families(families):
    """Small native response dashboard, with secondary evidence kept separate."""
    primary, secondary, nonlocal_peaks, pressure = [], [], [], []
    decisions = []
    for family in families:
        case_id = family["case_id"]
        for finite in family["finite"]:
            destination = primary if finite["baseline_state"] == "baseline" else secondary
            for role, channels in finite["roles"].items():
                for channel, metrics in channels.items():
                    destination.append({"case_id": case_id, "state": finite["state"],
                                        "role": role, "channel": channel, **metrics})
            if destination is primary:
                for module_id, value in finite.get("module_peak_changes", {}).items():
                    if value["unchanged_own_heat"]:
                        nonlocal_peaks.append({"case_id": case_id, "state": finite["state"],
                                               "module_id": module_id, **value})
                if "pressure_drop_8pct_response" in finite:
                    pressure.append({"case_id": case_id, "state": finite["state"], **finite["pressure_drop_8pct_response"]})
        if "stored_pool_decision" in family:
            decisions.append({"case_id": case_id, "primary": family["baseline_available"], **family["stored_pool_decision"]})
    return {"primary_response_aggregate": aggregate_response_rows(primary), "primary_response_rows": primary,
            "secondary_span_aggregate": aggregate_response_rows(secondary), "secondary_span_rows": secondary,
            "unchanged_own_heat_module_rows": nonlocal_peaks, "pressure_drop_8pct_response_rows": pressure,
            "stored_pool_decisions": decisions}


def parent_field_gate(parent, candidate):
    """Predeclared per-field mean and tail warnings; no aggregate-MSE hiding."""
    audit_field_comparability(parent, candidate)
    result = {}
    for field in CORE_FIELDS:
        old = parent["primary_excluding_0273"][field]
        new = candidate["primary_excluding_0273"][field]
        mean_change = 100 * (new["equal_case_rmse_mean"] / old["equal_case_rmse_mean"] - 1)
        tail_change = 100 * (new["case_rmse_p90"] / old["case_rmse_p90"] - 1)
        result[field] = {"mean_change_percent": mean_change, "p90_change_percent": tail_change,
                         "mean_degradation_flag": mean_change > 5, "p90_degradation_flag": tail_change > 10}
    return result


def load_counted_families(request_path, records_dir):
    """Read exact stored request outcomes, preserving absent 0277 baseline."""
    from channelthermal.interaction_evidence.reference_adapter import operating_context_from_config
    from channelthermal.interaction_evidence.types import (
        DesignState,
        MeasuredQuantity,
        ModuleState,
        PhysicalSolveOutput,
        RoleOutput,
        SolveRecord,
    )

    request = json.loads(Path(request_path).read_text())
    families = []
    for case in request["cases"]:
        case_id = case["case_id"]
        if case_id not in FIXED4:
            raise ValueError("Counted response evaluation is restricted to the declared fixed4")
        records = {}
        for state in case["states"]:
            label = state["state_id"]
            path = Path(records_dir) / f"receiver_interaction_{case_id}_{label}.json"
            if not path.exists():
                if (case_id, label) != ("0277", "baseline"):
                    raise FileNotFoundError(path)
                ledger_path = Path(records_dir).parent / "reference_attempts.jsonl"
                events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
                terminal = [event for event in events if event.get("event") == "terminal"
                            and event.get("attempt_id") == path.stem]
                if len(terminal) != 1 or terminal[0].get("status") != "failed_no_record":
                    raise ValueError("Missing 0277 baseline lacks its preserved failed-attempt receipt")
                continue
            metadata = json.loads(path.read_text())
            if (metadata["status"] != "converged" or metadata["exact_request_input"] != state
                    or metadata["case_id"] != case_id or metadata["state_id"] != label
                    or metadata["source"] != "reference_solver"):
                raise ValueError("Stored counted record does not match the exact converged request")
            config = case["baseline_generator_input_config"]
            design = DesignState(case_id, f"receiver_interaction_fixed4:{case_id}", "development", tuple(
                ModuleState(f"{case_id}:module:{slot}", tuple(xy), float(heat))
                for slot, xy, heat in zip(case["active_source_slots"], config["layout"]["centers"], state["active_heat_powers"])))
            with np.load(metadata["output_arrays"], allow_pickle=False) as data:
                if tuple(data["active_module_ids"].tolist()) != design.active_module_ids:
                    raise ValueError("Counted physical module IDs differ from exact request")
                roles = {}
                for role, kind in zip(ROLES, ("eulerian", "interface_material_angle", "solid_material_normalized_xy")):
                    receiver = tuple(data[f"{role}/receiver_module_ids"].tolist()) if role != "fluid_fields" else None
                    roles[role] = RoleOutput(role, data[f"{role}/query_features"], data[f"{role}/values"],
                        tuple(data[f"{role}/channel_names"].tolist()), tuple(data[f"{role}/channel_units"].tolist()),
                        data[f"{role}/valid_mask"], data[f"{role}/quadrature_weights"], tuple(data[f"{role}/query_ids"].tolist()), receiver, kind)
                output = PhysicalSolveOutput(roles, {key: MeasuredQuantity(**value) for key, value in metadata["quantities"].items()},
                    design.active_module_ids, dict(zip(design.active_module_ids, data["module_peak_temperature"].astype(float).tolist())),
                    {"reference": "analytic-wake/shared-grid; not CFD"})
            records[label] = SolveRecord(metadata.get("record_id", path.stem),
                design, operating_context_from_config(config), metadata["source"], metadata["status"],
                metadata["solver_elapsed_seconds"], metadata["provenance"], output)
        families.append((case_id, records))
    if tuple(case_id for case_id, _ in families) != FIXED4 or sum(len(records) for _, records in families) != 11:
        raise ValueError("Fixed4 evaluation requires exactly the existing eleven states")
    return families, request["source_dataset"]


@contextlib.contextmanager
def joint_intervention(model, mode):
    if mode == "normal":
        yield
        return
    module = model.core.backend.tensor_query_interaction
    if module.mode != "joint" or mode != "zero_joint":
        raise ValueError("Only the final joint model's I-only intervention is supported")
    original = module.intervention
    module.intervention = "zero_joint"
    try:
        yield
    finally:
        module.intervention = original


def evaluate_families(checkpoint_path, families, dataset_path, output_dir, *, device="cpu", intervention="normal"):
    """Frozen complete-wrapper reads; model sees designs/context/queries only."""
    import h5py
    import torch
    from channelthermal.evaluation.loading import load_model
    from channelthermal.response_control.contracts import DesignInput, context_inputs, role_queries_from_record
    from channelthermal.response_control.native import DifferentiableThermalOperator
    from thermal_campaign_benchmark import optional_fine_work
    from thermal_campaign_heat_inference import snapshot_forward_state, verify_frozen_forward
    from thermal_development import resolve_evaluation_manifest, validate_generated_output

    output_dir = validate_generated_output(output_dir)
    if (output_dir / "summary.json").exists():
        raise ValueError("Preserve previous evidence: use a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(Path(checkpoint_path), torch.device(device))
    model.eval().requires_grad_(False)
    snapshot = snapshot_forward_state(model)
    manifest = resolve_evaluation_manifest(checkpoint["train_config"]["dataset"], dataset_path, scope="development")
    stats = checkpoint.get("global_normalization_stats")
    if not manifest or not stats:
        raise ValueError("Refinement evaluation requires bound development identity and saved normalizers")
    summary = {"checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_epoch": checkpoint["epoch"],
               "intervention": intervention, "manifest_fingerprint": manifest["manifest_sha256"],
               "solver_attempts": 0, "inverse_search_calls": 0, "optimizer_updates": 0, "wrapper_calls": 0,
               "scope": "stored analytic/shared-grid responses, exposed development evidence; no CFD claim", "families": []}
    state = checkpoint.get("campaign_training_state", {})
    summary["lineage"] = {key: state[key] for key in ("interface_fit_attachment", "forward_refinement_attachment") if key in state}
    with h5py.File(dataset_path, "r") as packed:
        for case_id, records in families:
            validate_heat_records(records)
            first = next(iter(records.values()))
            queries = role_queries_from_record(first, device=device)
            context = context_inputs(first.context)
            capacity = packed["cases"][case_id]["module_centers"].shape[0]
            material = np.asarray([context[key] for key in ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius")], dtype=np.float32)
            template = {"structure": {"module_centers": np.zeros((capacity, 2), dtype=np.float32), "material_params": material}}
            operator = DifferentiableThermalOperator(model, template, dataset_config=checkpoint["train_config"]["dataset"],
                normalization_stats=stats, query_batch_size=1024, organizer_shadow=False)
            predictions, arrays, functionals, fine_work = {}, {}, {}, {}
            family = {"case_id": case_id, "physical_family_id": first.design.physical_family_id, "absolute": [], "finite": [],
                      "baseline_available": "baseline" in records, "source_partition": first.design.split.value}
            family_started = perf_counter()
            for label, record in records.items():
                # Query IDs/coordinates and typed receiver joins are checked by
                # the response record contracts, never replaced by solved inputs.
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize()
                wrapper_started = perf_counter()
                with torch.no_grad(), joint_intervention(model, intervention), optional_fine_work(model.core.backend) as work:
                    output = operator(DesignInput.from_state(record.design, device=device), context, queries)
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize()
                summary["wrapper_calls"] += 1
                predictions[label] = {role: value.detach().cpu().numpy() for role, value in output.role_values.items()}
                fine_work[label] = work
                functionals[label] = native_functionals(record, predictions[label])
                reference_values = {role: value.values for role, value in record.output.roles.items()}
                reference_functionals = native_functionals(record, reference_values)
                row = {"state": label, "roles": {}, "prediction_functionals": functionals[label],
                       "reference_functionals": reference_functionals, "heat": [module.heating for module in record.design.modules],
                       "centres": [list(module.position_xy) for module in record.design.modules],
                       "complete_wrapper_seconds": perf_counter() - wrapper_started,
                       "timer_scope": "complete native call plus instrumented fine-work hooks and CPU output copy; not bare latency"}
                for role, value in record.output.roles.items():
                    row["roles"][role], _, _ = finite_metrics(predictions[label][role], value.values, value)
                    arrays[f"{label}/{role}/prediction"] = predictions[label][role]
                    arrays[f"{label}/{role}/reference"] = value.values
                family["absolute"].append(row)
            pairs = [("baseline", label) for label in records if label != "baseline"] if "baseline" in records else [("transfer_minus", "transfer_plus")]
            for baseline, label in pairs:
                row = {"state": label, "baseline_state": baseline, "scope": "primary baseline-relative" if baseline == "baseline" else "secondary minus-to-plus span", "roles": {}}
                for role, value in records[label].output.roles.items():
                    row["roles"][role], delta_pred, delta_ref = finite_metrics(predictions[label][role], value.values, value,
                        baseline_prediction=predictions[baseline][role], baseline_reference=records[baseline].output.roles[role].values)
                    arrays[f"{label}/{role}/delta_prediction_FP64"] = delta_pred
                    arrays[f"{label}/{role}/delta_reference_FP64"] = delta_ref
                absolute_by_state = {value["state"]: value for value in family["absolute"]}
                base = absolute_by_state[baseline]
                trial = absolute_by_state[label]
                predicted_drop = trial["prediction_functionals"]["pressure_drop_8pct"] - base["prediction_functionals"]["pressure_drop_8pct"]
                reference_drop = trial["reference_functionals"]["pressure_drop_8pct"] - base["reference_functionals"]["pressure_drop_8pct"]
                row["pressure_drop_8pct_response"] = {"prediction": predicted_drop, "reference": reference_drop,
                    "absolute_error": abs(predicted_drop - reference_drop), "unit": "dataset pressure units"}
                row["module_peak_changes"] = {}
                baseline_heat = {module.module_id: module.heating for module in records[baseline].design.modules}
                for module in records[label].design.modules:
                    module_id = module.module_id
                    predicted_change = trial["prediction_functionals"]["module_peak_temperature"][module_id] - base["prediction_functionals"]["module_peak_temperature"][module_id]
                    reference_change = trial["reference_functionals"]["module_peak_temperature"][module_id] - base["reference_functionals"]["module_peak_temperature"][module_id]
                    row["module_peak_changes"][module_id] = {"prediction": predicted_change, "reference": reference_change,
                        "absolute_error": abs(predicted_change - reference_change), "unchanged_own_heat": module.heating == baseline_heat[module_id],
                        "unit": "dataset temperature units"}
                family["finite"].append(row)
            for role, value in first.output.roles.items():
                for key in ("query_features", "valid_mask", "quadrature_weights"):
                    arrays[f"{role}/{key}"] = getattr(value, key)
                for key in ("channel_names", "channel_units", "query_ids", "receiver_module_ids"):
                    arrays[f"{role}/{key}"] = np.asarray(getattr(value, key) or (), dtype=str)
            family_dir = output_dir / case_id
            family_dir.mkdir(exist_ok=True)
            np.savez_compressed(family_dir / "evidence.npz", **arrays)
            reference_maxima = {row["state"]: row["reference_functionals"]["maximum_material_temperature"] for row in family["absolute"]}
            prediction_maxima = {row["state"]: row["prediction_functionals"]["maximum_material_temperature"] for row in family["absolute"]}
            warnings = [abs(float(record.provenance.get("runtime", {}).get("final_delta_inf", 0))) for record in records.values()]
            family.update(arrays=str(family_dir / "evidence.npz"), fine_kernel_work=fine_work,
                          family_elapsed_seconds=perf_counter() - family_started)
            if case_id in FIXED4:
                family["stored_pool_decision"] = stored_pool_decision(reference_maxima, prediction_maxima, warning_scale=2 * max(warnings))
            write_json(family_dir / "summary.json", family)
            summary["families"].append(family)
            write_json(output_dir / "summary.json", summary)
    summary["frozen_state"] = verify_frozen_forward(model, snapshot)
    if not summary["frozen_state"]["passed"]:
        raise RuntimeError("Frozen evaluation changed model values")
    summary.update(summarize_families(summary["families"]))
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dashboard = subparsers.add_parser("dashboard", help="CPU aggregation of existing full22 field summaries")
    dashboard.add_argument("--summary", action="append", required=True, help="NAME=/absolute/summary.json")
    dashboard.add_argument("--output", type=Path, required=True)
    counted = subparsers.add_parser("counted", help="Read the existing eleven fixed4 states with one checkpoint")
    counted.add_argument("--request", type=Path, required=True)
    counted.add_argument("--records-dir", type=Path, required=True)
    counted.add_argument("--checkpoint", type=Path, required=True)
    counted.add_argument("--output-dir", type=Path, required=True)
    counted.add_argument("--device", default="cpu")
    counted.add_argument("--intervention", choices=("normal", "zero_joint"), default="normal")
    preparation = subparsers.add_parser("prepare-counted", help="CPU receiver/input/pool audit; no model reads")
    preparation.add_argument("--request", type=Path, required=True)
    preparation.add_argument("--records-dir", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    atlas = subparsers.add_parser("atlas", help="Read declared heat-only endpoints of existing response-development families")
    atlas.add_argument("--stencil", type=Path, action="append", required=True)
    atlas.add_argument("--variant", action="append", required=True)
    atlas.add_argument("--checkpoint", type=Path, required=True)
    atlas.add_argument("--dataset", type=Path, required=True)
    atlas.add_argument("--output-dir", type=Path, required=True)
    atlas.add_argument("--device", default="cpu")
    atlas.add_argument("--intervention", choices=("normal", "zero_joint"), default="normal")
    args = parser.parse_args(argv)
    if args.command == "dashboard":
        from thermal_development import validate_generated_output
        summaries = {}
        for item in args.summary:
            name, path = item.split("=", 1)
            if name in summaries:
                raise ValueError("Dashboard model labels must be unique")
            summaries[name] = json.loads(Path(path).read_text())
        write_json(validate_generated_output(args.output), field_dashboard(summaries))
    elif args.command == "atlas":
        from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
        families = []
        for path in args.stencil:
            stencil, _ = load_response_atlas_stencil(path)
            if (stencil.baseline.design.anchor_id not in {"0304", "0320", "0335", "0350"}
                    or stencil.split.value != "train"):
                raise ValueError("Response-development evaluation is restricted to four declared withheld families")
            if len(args.variant) != 2 or len(set(args.variant)) != 2:
                raise ValueError("Select exactly two opposite stored heat directions")
            records = {"baseline": stencil.baseline, **{label: stencil.variants[label] for label in args.variant}}
            validate_heat_records(records)
            families.append((stencil.baseline.design.anchor_id, records))
        if len({case_id for case_id, _ in families}) != len(families):
            raise ValueError("Response-development families must not repeat")
        evaluate_families(args.checkpoint, families, args.dataset, args.output_dir, device=args.device, intervention=args.intervention)
    else:
        families, dataset_path = load_counted_families(args.request, args.records_dir)
        if args.command == "prepare-counted":
            from thermal_development import validate_generated_output
            write_json(validate_generated_output(args.output), prepare_counted_replay(families))
        else:
            evaluate_families(args.checkpoint, families, dataset_path, args.output_dir, device=args.device, intervention=args.intervention)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
