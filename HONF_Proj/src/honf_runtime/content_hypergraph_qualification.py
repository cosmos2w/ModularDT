"""Measured Round2 follow-up criteria, independent of the historical v1 gate.

Inputs are retained native evaluation records, never training selectors or a
per-channel oracle. A qualification binds immutable evidence and the exact
fresh recipe; it is not permission to execute the prospective 5000 horizon.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA = "content_hypergraph_full1000_qualification_v2"
FIRST_DIRECT = {
    "thermal": "30da0a857a1b23f649250f02aaa9d65eab1d314ba958fa058b584ff3b36a5bae",
    "wind": "79fea32bcf618928b1a54ae519f6f9919fb0aa0a200a93051d06b447886e53cd",
}
FIRST_ORGANIZED = {
    "thermal": "0d4eb5eeb120c8be630dc439e7fc39e2c625e8fad3a28b53a644836ae032e117",
    "wind": "6c4ff6ecff90150c04f6382cbf1c23609fc10559e071dbd1ebd874492b3984cc",
}
WIND_DEV_PANEL_SHA256 = "594681b0f270eb82ed271adf5d962c00be0b1885f9e99b157662c5b2c3f028ed"
WIND_ROLE_QUERY_COUNTS = {
    "volume": 1640,
    "hub_slab": 1640,
    "downstream_envelope": 1640,
    "near_turbine": 1640,
    "background": 1632,
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def recipe_architecture(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Only partition, execution stop, schedule horizon and RAM capacity differ."""
    value = dict(recipe)
    for key in ("protocol", "total_epochs", "approved_stop_after", "catalogue_cache_ram_bytes"):
        value.pop(key, None)
    return value


def physical_output_law(provider_identity: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    """Physical equations only; selected TRAIN normalization values stay separate."""
    if dataset == "thermal":
        return dict(provider_identity["thermal_v2_native_laws"])
    state = provider_identity["adapter_state"]
    return {name: state[name] for name in ("physical_output", "residual_transform",
                                         "wind_direction_handling", "local_radius_D")}


def evidence_json(descriptor: Mapping[str, Any], *, require_completed: bool = True) -> dict[str, Any]:
    path = Path(descriptor["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor.get("sha256"):
        raise ValueError(f"Qualification evidence changed: {path}.")
    value = json.loads(raw)
    if not isinstance(value, dict) or (require_completed and value.get("status") != "completed"):
        raise ValueError(f"Qualification evidence is incomplete: {path}.")
    return value


def verify_source_files(descriptors: Any) -> None:
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("Measured qualification requires inspectable source evidence files.")
    for descriptor in descriptors:
        digest = hashlib.sha256()
        with Path(descriptor["path"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != descriptor.get("sha256"):
            raise ValueError("A measured qualification source file changed.")


def native_record(value: Mapping[str, Any], dataset: str, *, label: str | None = None) -> dict[str, Any]:
    """Normalize the existing native evaluator and same-panel Wind board formats."""
    if dataset == "wind" and "endpoint_models" in value:
        if label is None:
            raise ValueError("A native Wind board requires one explicit coherent model label.")
        model = value["endpoint_models"][label]
        checkpoint = model["checkpoint"]
        return {"checkpoint_sha256": checkpoint["sha256"], "epoch": checkpoint["completed_epoch"],
                "architecture": checkpoint["architecture"], "rows": model["row_metrics"],
                "field_score": model["field_score_mean"],
                "panel_sha256": model["endpoint_panel_sha256"]}
    if value.get("panel") != "native" or value.get("split") != "validation":
        raise ValueError("Qualification requires the unchanged exposed native validation panel.")
    metrics = value["metrics"]
    return {"checkpoint_sha256": value["checkpoint_sha256"], "epoch": value["completed_epoch"],
            "architecture": value["architecture"], "rows": metrics["per_case" if dataset == "thermal" else "rows"],
            "field_score": metrics["field_score"], "panel_sha256": value.get("endpoint_panel_sha256")}


def summarize_native(record: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    rows = record["rows"]
    expected = 22 if dataset == "thermal" else 24
    if len(rows) != expected:
        raise ValueError(f"Round2 native DEV requires all {expected} cases/rows.")
    identifiers = [str(row["case_id" if dataset == "thermal" else "row_index"]) for row in rows]
    if len(set(identifiers)) != expected:
        raise ValueError("Native DEV rows must occur exactly once.")
    if dataset == "thermal":
        names = ("u", "v", "p", "omega", "temperature_fluid", "temperature_surface", "temperature_material")
        roles = {name: float(np.mean([row["roles"][name]["rmse"] for row in rows])) for name in names}
        roles["q_proxy"] = float(np.mean([row["q_proxy_rmse"] for row in rows]))
        tails = {"temperature_fluid": float(np.quantile(
            [row["roles"]["temperature_fluid"]["rmse"] for row in rows], .90))}
        components = {}
    else:
        names = ("volume", "hub_slab", "downstream_envelope", "near_turbine", "background")
        roles = {name: float(np.mean([row["roles"][name]["vector_rmse_mps"] for row in rows])) for name in names}
        component_rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
        for row in rows:
            for name in names:
                values = row["roles"][name]["component_rmse_mps"]
                try:
                    vector = np.asarray(values, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Wind component RMSE vectors must contain exactly three finite values.") from exc
                if vector.shape != (3,) or not np.all(np.isfinite(vector)) or np.any(vector < 0):
                    raise ValueError("Wind component RMSE vectors must contain exactly three finite nonnegative values.")
                component_rows[name].append(vector)
                count = row["roles"][name].get("query_count")
                if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
                    raise ValueError("Wind qualification requires integer per-role native query counts.")
                if int(count) != WIND_ROLE_QUERY_COUNTS[name]:
                    raise ValueError("Wind qualification requires the exact per-role native query quotas.")
        components = {name: np.mean(component_rows[name], axis=0).tolist() for name in names}
        tails = {name: float(np.quantile([row["roles"][name]["vector_rmse_mps"] for row in rows], .90))
                 for name in ("near_turbine", "downstream_envelope")}
    values = [record["field_score"], *roles.values(), *tails.values(),
              *(x for vector in components.values() for x in vector)]
    if any(not math.isfinite(float(x)) or float(x) < 0 for x in values):
        raise ValueError("Native field metrics must be finite and nonnegative.")
    return {"ids": sorted(identifiers), "field_score": float(record["field_score"]), "roles": roles,
            "components": components, "priority_p90": tails,
            "p90_definition": "quantile of case/row native RMSE; not a pointwise residual quantile"}


def field_decision(dataset: str, organized: Mapping[str, Any], direct: Mapping[str, Any],
                   first_direct: Mapping[str, Any], *, floors: Mapping[str, float]) -> dict[str, Any]:
    """Apply the fixed Section10 criteria and retain every miss separately."""
    summaries = [summarize_native(x, dataset) for x in (organized, direct, first_direct)]
    candidate, matched, baseline = summaries
    if not (candidate["ids"] == matched["ids"] == baseline["ids"]):
        raise ValueError("Qualification panels have different native memberships.")
    if dataset == "wind":
        panels = (organized.get("panel_sha256"), direct.get("panel_sha256"), first_direct.get("panel_sha256"))
        if any(panel != WIND_DEV_PANEL_SHA256 for panel in panels):
            raise ValueError("Wind qualification requires the sealed same-panel DEV Q8192 evidence.")
    if any(x["epoch"] != 2500 for x in (organized, direct, first_direct)):
        raise ValueError("Round2 qualification requires literal2500 for every development arm.")
    if (organized["architecture"], direct["architecture"], first_direct["architecture"]) != ("A2-H6", "A2-D", "A-D"):
        raise ValueError("Qualification must compare the declared new pair with coherent first-round A-D.")
    if first_direct["checkpoint_sha256"] != FIRST_DIRECT[dataset]:
        raise ValueError("First-round A-D baseline is not the reviewed coherent e2500 checkpoint.")

    def ratio(a: float, b: float, name: str) -> float:
        floor = float(floors.get(name, 0.0))
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError(f"Missing positive inherited TRAIN-scale floor for {name}.")
        return float(a / max(b, floor))

    ratios = {"broad_vs_first": ratio(candidate["field_score"], baseline["field_score"], "field_score")}
    checks = {"broad_no_worse_than_first": ratios["broad_vs_first"] <= 1.0}
    priority = {"temperature_fluid"} if dataset == "thermal" else {"near_turbine", "downstream_envelope"}
    if dataset == "thermal":
        ratios["priority_vs_first"] = ratio(candidate["roles"]["temperature_fluid"], baseline["roles"]["temperature_fluid"], "temperature_fluid")
        ratios["priority_vs_matched"] = ratio(candidate["roles"]["temperature_fluid"], matched["roles"]["temperature_fluid"], "temperature_fluid")
        ratios["broad_vs_matched"] = ratio(candidate["field_score"], matched["field_score"], "field_score")
        checks.update(priority_ten_percent_gain=ratios["priority_vs_first"] <= .90,
                      priority_vs_matched=ratios["priority_vs_matched"] <= 1.15,
                      broad_vs_matched=ratios["broad_vs_matched"] <= 1.15)
    else:
        six = {
            f"{role}/{axis}": ratio(candidate["components"][role][index],
                                     baseline["components"][role][index], f"{role}/{axis}")
            for role in sorted(priority)
            for index, axis in enumerate(("Ux", "Uy", "Uz"))
        }
        if len(six) != 6:
            raise ValueError("Wind priority comparison requires all six near/downstream component ratios.")
        ratios.update(six)
        ratios["six_component_geometric_mean"] = float(np.prod(tuple(six.values())) ** (1 / 6))
        checks["priority_ten_percent_gain"] = ratios["six_component_geometric_mean"] <= .90
        for role in sorted(priority):
            ratios[f"{role}_vs_matched"] = ratio(candidate["roles"][role], matched["roles"][role], role)
            checks[f"{role}_vs_matched"] = ratios[f"{role}_vs_matched"] <= 1.15
    for role in sorted(set(candidate["roles"]) - priority):
        ratios[f"nonpriority/{role}"] = ratio(candidate["roles"][role], baseline["roles"][role], role)
        checks[f"nonpriority/{role}"] = ratios[f"nonpriority/{role}"] <= 1.25
    for role in sorted(priority):
        ratios[f"p90/{role}"] = ratio(candidate["priority_p90"][role], baseline["priority_p90"][role], role)
        checks[f"p90/{role}"] = ratios[f"p90/{role}"] <= 1.25
    return {"qualified": all(checks.values()), "checks": checks, "ratios": ratios,
            "summaries": dict(zip(("organized", "matched_direct", "first_direct"), summaries))}


def validate_round2_qualification(path: Path, recipe: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA or value.get("dataset") != recipe["dataset"]:
        raise ValueError("Missing measured Round2 full-TRAIN qualification.")
    if value.get("full_recipe_sha256") != canonical_sha256(recipe):
        raise ValueError("Full recipe differs from its measured qualification.")
    development = value["development_recipe"]
    if (development.get("protocol") != "development" or development.get("total_epochs") != 2500
            or recipe_architecture(development) != recipe_architecture(recipe)):
        raise ValueError("A full follow-up cannot silently change the qualified candidate.")
    if value.get("approved_stop_after") != 1000 or value.get("prospective_schedule_epochs") != 5000:
        raise ValueError("Round2 follow-up is limited to the explicit 1000 stop.")
    evidence = value["evidence"]
    records = {}
    for name in ("organized", "matched_direct", "first_direct"):
        artifact = evidence_json(evidence[name])
        records[name] = native_record(artifact, recipe["dataset"], label=evidence[name].get("model_label"))
    decision = field_decision(recipe["dataset"], records["organized"], records["matched_direct"],
                              records["first_direct"], floors=value["inherited_train_floors"])
    if not decision["qualified"]:
        raise ValueError(f"Round2 native improvement criteria failed: {decision['checks']}.")
    measured = {}
    for name in ("correctness", "content", "cost", "budget"):
        artifact = evidence_json(evidence[name])
        if artifact.get("architecture_config_sha256") != value.get("architecture_config_sha256"):
            raise ValueError(f"{name} evidence belongs to a different architecture/configuration.")
        if artifact.get("development_checkpoint_sha256") != records["organized"]["checkpoint_sha256"]:
            raise ValueError(f"{name} evidence belongs to a different development checkpoint.")
        if artifact.get("physical_output_law_sha256") != value.get("physical_output_law_sha256"):
            raise ValueError(f"{name} evidence belongs to a different physical output law.")
        if artifact.get("qualified") is not True or not artifact.get("measurements"):
            raise ValueError(f"Missing measured {name} qualification.")
        measured[name] = artifact["measurements"]
        verify_source_files(artifact["measurements"].get("source_evidence"))
    correctness = measured["correctness"]
    required_checks = {"attention_reference", "masked_empty_rows", "staged_exchange", "measure_refinement",
                       "native_observations", "fresh_head_gradients", "resume_identity"}
    if recipe["dataset"] == "thermal":
        required_checks.update({"source_coefficients_W_coupling", "affine_and_zero_offset", "heat_adjoint"})
    else:
        required_checks.update({"nonlinear_velocity_law", "native_frame_and_profile"})
    if (not required_checks <= correctness.get("checks", {}).keys()
            or not all(correctness["checks"][key] is True for key in required_checks)
            or correctness.get("failed_checks") != 0 or int(correctness.get("cpu_test_pass_count", 0)) < 1):
        raise ValueError("Numerical/native qualification is incomplete or contains failed checks.")
    inspection = evidence_json(measured["content"]["inspection"])
    if (inspection.get("architecture") != "A2-H6" or inspection.get("completed_epoch") != 2500
            or inspection.get("checkpoint_sha256") != records["organized"]["checkpoint_sha256"]
            or len(inspection.get("cases", [])) < 2):
        raise ValueError("Content-use evidence is not the mature candidate on its fixed scene panel.")
    for case in inspection["cases"]:
        if case.get("edge_selection", {}).get("edge") != 0:
            raise ValueError("Round2 content removal must use input-declared edge0.")
        conditions = case["conditions"]
        removed = conditions["input_declared_edge_removed"]
        local = conditions["local_branch_removed"]
        if (removed.get("edge_access_frozen_without_renormalization") is not True
                or local.get("edge_access_preserved") is not True
                or not math.isfinite(float(removed["output_delta_l2"])) or float(removed["output_delta_l2"]) <= 0
                or not math.isfinite(float(local["output_delta_l2"])) or float(local["output_delta_l2"]) <= 0):
            raise ValueError("Actual edge-content/local output use has not been established.")
    cost = measured["cost"]
    if not 0 < float(cost["current_same_device_same_output_high_m_ratio"]) <= 2:
        raise ValueError("Round2 complete high-M cost exceeds its engineering ceiling.")
    baseline_seconds = float(cost["baseline_complete_high_m_median_seconds"])
    candidate_seconds = float(cost["candidate_complete_high_m_median_seconds"])
    if (not math.isfinite(baseline_seconds) or baseline_seconds <= 0
            or not math.isfinite(candidate_seconds) or candidate_seconds <= 0
            or not math.isclose(candidate_seconds / baseline_seconds,
                                float(cost["current_same_device_same_output_high_m_ratio"]), rel_tol=1e-10)
            or cost.get("baseline_checkpoint_sha256") != FIRST_ORGANIZED[recipe["dataset"]]
            or not cost.get("same_device_uuid") or not cost.get("same_output_request_sha256")):
        raise ValueError("Complete cost does not bind a current same-device/output baseline replay.")
    points = cost["fixed_batch_backward"]
    for arm in ("A2-D", "A2-H6"):
        observed = {(int(x["batch_size"]), int(x["requested_q_per_scene"])) for x in points
                    if x["architecture"] == arm and x["status"] == "completed"}
        if not {(8, 1024), (8, 4096), (8, 8192)} <= observed:
            raise ValueError("Fixed-B8 native backward scaling is incomplete or infeasible.")
    budget = measured["budget"]
    for key in ("forecast_gpu_associated_seconds", "remaining_gpu_associated_seconds", "forecast_elapsed_seconds",
                "closeout_reserve_seconds", "remaining_elapsed_seconds"):
        if not math.isfinite(float(budget[key])) or float(budget[key]) < 0:
            raise ValueError("The full-run resource forecast must be finite and nonnegative.")
    if (float(budget["forecast_gpu_associated_seconds"]) > float(budget["remaining_gpu_associated_seconds"])
            or float(budget["forecast_elapsed_seconds"]) + float(budget["closeout_reserve_seconds"])
            > float(budget["remaining_elapsed_seconds"]) or float(budget["closeout_reserve_seconds"]) < 7200):
        raise ValueError("The complete full-run forecast does not fit the resource reserve.")
    if not value.get("partition_binding") or not value.get("architecture_config_sha256") or not value.get("physical_output_law_sha256"):
        raise ValueError("Missing configuration, native-law or full-partition binding.")
    identity = evidence_json(value["development_identity_evidence"], require_completed=False)
    if (identity.get("recipe") != development
            or canonical_sha256(identity["provider_identity"]["model_config"]) != value["architecture_config_sha256"]
            or not identity.get("loaded_source", {}).get("files")):
        raise ValueError("Development architecture/source identity does not bind its measured candidate.")
    checkpoint = value["development_checkpoint_evidence"]
    if checkpoint.get("sha256") != records["organized"]["checkpoint_sha256"]:
        raise ValueError("Development checkpoint differs from the native field evidence.")
    verify_source_files([checkpoint])
    from .compat import load_trusted_checkpoint
    payload = load_trusted_checkpoint(checkpoint["path"], map_location="cpu")
    if (payload.get("epoch") != 2500 or payload.get("arm") != "A2-H6"
            or canonical_sha256(payload.get("experiment_identity")) != canonical_sha256(identity)):
        raise ValueError("Development checkpoint and its identity evidence are inconsistent.")
    value["development_loaded_source"] = identity["loaded_source"]
    value["verified_field_decision"] = decision
    return value
