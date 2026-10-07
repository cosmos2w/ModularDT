#!/usr/bin/env python3
"""Build a bounded receiver-local explanatory view from the saved R-direct model.

This is a post-fit view of the full learned source-resolved kernel. It does not
change the authoritative forward path, fit a selector, or invoke a solver.
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
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from channelthermal.source_response import load_source_response_model
from thermal_development import validate_generated_output
from thermal_source_response_evaluate import load_native_cases

CASES = ("0277", "0291", "0294", "0687")
TRAIN_HEAT_MINIMUM = 0.5036706328392029
TRAIN_HEAT_MAXIMUM = 1.999153733253479
TRAIN_TEMPERATURE_RESPONSE_RMS = 0.44082161399072206
BUDGET_FRACTIONS = (0.01, 0.02)


def bounded_radii(heat, present, train_minimum=TRAIN_HEAT_MINIMUM,
                  train_maximum=TRAIN_HEAT_MAXIMUM):
    """Existing input-only balanced forcing box, in physical heat units."""
    heat = np.asarray(heat, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    if heat.ndim != 1 or present.shape != heat.shape or not present.any():
        raise ValueError("Heat and presence must be aligned one-dimensional source catalogs")
    if not np.isfinite(heat).all() or np.any(heat[present] < train_minimum) or np.any(heat[present] > train_maximum):
        raise ValueError("Physical heat lies outside the existing TRAIN-bounded response box")
    mean_heat = float(heat[present].mean())
    cap = 0.20 * mean_heat
    return np.where(present, np.maximum(0.0, np.minimum.reduce((heat - train_minimum,
        train_maximum - heat, np.full_like(heat, cap)))), 0.0)


def make_patch_catalog(xy, fluid_mask, *, domain_x=12.0, domain_y=6.0, nx=4, ny=4):
    """Partition the native grid into fixed equal-area patches and normalize W_P."""
    xy = np.asarray(xy, dtype=np.float64)
    fluid = np.asarray(fluid_mask, dtype=bool)
    if xy.ndim != 2 or xy.shape[1] != 2 or fluid.shape != (xy.shape[0],):
        raise ValueError("Patch catalog requires aligned [Q,2] coordinates and [Q] fluid mask")
    if nx < 1 or ny < 1 or domain_x <= 0 or domain_y <= 0:
        raise ValueError("Patch dimensions and physical domain extents must be positive")
    ix = np.minimum(np.floor(xy[:, 0] / domain_x * nx).astype(int), nx - 1)
    iy = np.minimum(np.floor(xy[:, 1] / domain_y * ny).astype(int), ny - 1)
    labels = iy * nx + ix
    weights = np.zeros(fluid.shape, dtype=np.float64)
    names, centroids, counts = [], [], []
    for py in range(ny):
        for px in range(nx):
            patch = py * nx + px
            rows = (labels == patch) & fluid
            count = int(rows.sum())
            if count == 0:
                raise ValueError(f"Native receiver patch {py},{px} has no valid fluid rows")
            weights[rows] = 1.0 / count
            names.append(f"py{py}_px{px}")
            centroids.append([(px + 0.5) * domain_x / nx, (py + 0.5) * domain_y / ny])
            counts.append(count)
    if not np.all(np.bincount(labels, minlength=nx * ny) > 0):
        raise ValueError("Every native receiver row must belong to one patch")
    return {"labels": labels, "weights": weights, "names": names,
            "centroids": np.asarray(centroids, dtype=np.float64), "fluid_counts": counts,
            "nx": int(nx), "ny": int(ny)}


def response_scores(kernel, patch_weights, radii):
    """q_i(P) = r_i ||W_P^(1/2) K_(P,i)||_2."""
    kernel = np.asarray(kernel, dtype=np.float64)
    weights = np.asarray(patch_weights, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if kernel.ndim != 2 or weights.shape != (kernel.shape[0],) or radii.shape != (kernel.shape[1],):
        raise ValueError("Kernel, normalized patch weights and forcing radii are misaligned")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Patch quadrature weights must be nonnegative and sum to one")
    return radii * np.sqrt(np.einsum("q,qm,qm->m", weights, kernel, kernel, optimize=True))


def retained_kernel_precision_audit(kernel_fp64, kernel_retained_fp32):
    """Report the expected roundoff gap against the retained FP32 operator."""
    precise = np.asarray(kernel_fp64, dtype=np.float64)
    retained = np.asarray(kernel_retained_fp32, dtype=np.float64)
    if precise.shape != retained.shape or not np.isfinite(precise).all() or not np.isfinite(retained).all():
        raise ValueError("Native FP64 and retained FP32 source kernels must be finite and aligned")
    difference = precise - retained
    max_abs = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(np.square(difference))))
    # The retained counted operator was exported in FP32. A few-micro-unit
    # discrepancy is expected when re-exporting the same checkpoint in FP64;
    # this envelope was measured over all four fixed DEV22 receiver cases.
    if max_abs > 1e-5 or rms > 1e-6:
        raise ValueError(f"Native FP64 kernel differs materially from retained FP32 operator: max={max_abs}, rms={rms}")
    return {"max_abs": max_abs, "rms": rms}


def learned_cover(scores, budget, source_ids):
    """Return the smallest deterministic donor set whose omitted q sum fits."""
    scores = np.asarray(scores, dtype=np.float64)
    ids = tuple(str(value) for value in source_ids)
    if scores.ndim != 1 or scores.size != len(ids) or not np.isfinite(scores).all():
        raise ValueError("Cover scores must be finite and match the physical donor IDs")
    if budget < 0 or not np.isfinite(budget):
        raise ValueError("Cover budget must be finite and nonnegative")
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    keep = len(ids)
    for count in range(len(ids) + 1):
        omitted = order[count:]
        if float(scores[omitted].sum()) <= budget:
            keep = count
            break
    selected = order[:keep]
    omitted_bound = float(scores[[i for i in range(len(ids)) if i not in selected]].sum())
    return {"selected_indices": selected, "selected_source_ids": [ids[i] for i in selected],
            "retained_count": len(selected), "omitted_triangle_bound": omitted_bound,
            "minimal_for_budget": omitted_bound <= budget and
                (keep == 0 or float(scores[order[keep - 1:]].sum()) > budget)}


def geometry_orders(centers, centroid, source_ids, u_in):
    """Input-only nearest and nearest-upstream controls with physical-ID ties."""
    centers = np.asarray(centers, dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64)
    ids = tuple(str(value) for value in source_ids)
    if centers.shape != (len(ids), 2) or centroid.shape != (2,) or u_in == 0 or not np.isfinite(u_in):
        raise ValueError("Geometry controls require aligned centers and nonzero input flow")
    distance = np.linalg.norm(centers - centroid[None], axis=1)
    nearest = sorted(range(len(ids)), key=lambda i: (distance[i], ids[i]))
    upstream_mask = np.sign(u_in) * (centers[:, 0] - centroid[0]) <= 0
    upstream = sorted((i for i in nearest if upstream_mask[i]), key=lambda i: (distance[i], ids[i]))
    downstream = [i for i in nearest if i not in set(upstream)]
    return {"nearest": nearest, "upstream": upstream + downstream}


def weighted_rms(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape[0] != weights.shape[0] or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Weighted RMS requires aligned values and normalized weights")
    return float(np.sqrt(np.einsum("q,q...->", weights, np.square(values), optimize=True)))


def make_response_row(kernel, delta_heat, reference_delta, selected, patch_rows, patch_weights,
                      scores, *, model_cold_delta=None, bound_applies=True):
    """Measure response omission and physical error after a selector is fixed."""
    kernel = np.asarray(kernel, dtype=np.float64)
    delta_heat = np.asarray(delta_heat, dtype=np.float64)
    reference_delta = np.asarray(reference_delta, dtype=np.float64)
    selected = np.asarray(selected, dtype=int)
    rows = np.asarray(patch_rows, dtype=int)
    weights = np.asarray(patch_weights, dtype=np.float64)
    if delta_heat.shape != (kernel.shape[1],) or reference_delta.shape != (kernel.shape[0],):
        raise ValueError("Saved finite response must align to learned sources and fluid receivers")
    keep_mask = np.zeros(kernel.shape[1], dtype=bool)
    keep_mask[selected] = True
    full = kernel[rows] @ delta_heat
    approximate = kernel[rows][:, keep_mask] @ delta_heat[keep_mask] if selected.size else np.zeros(rows.size)
    omitted = full - approximate
    bound = float(np.asarray(scores, dtype=np.float64)[~keep_mask].sum())
    distortion = weighted_rms(omitted, weights)
    tolerance = 1e-10 * max(1.0, bound)
    if bound_applies and distortion > bound + tolerance:
        raise AssertionError(f"Measured local omission {distortion} exceeds triangle bound {bound}")
    return {"measured_model_omission_rms": distortion,
            "omitted_triangle_bound": bound if bound_applies else None,
            "input_score_omitted_triangle_value": bound,
            "bound_applies_to_saved_delta": bool(bound_applies),
            "bound_holds_for_saved_delta": True if bound_applies else None,
            "full_model_physical_response_rmse": weighted_rms(full - reference_delta[rows], weights),
            "covered_model_physical_response_rmse": weighted_rms(approximate - reference_delta[rows], weights),
            "approximate_response_rms": weighted_rms(approximate, weights),
            "reference_response_rms": weighted_rms(reference_delta[rows], weights),
            "cold_endpoint_delta_comparison_max_abs": None if model_cold_delta is None else
                float(np.max(np.abs(full - np.asarray(model_cold_delta, dtype=np.float64)[rows])))}


def _load_signed_responses(case_dir, source_ids):
    summary = json.loads((case_dir / "summary.json").read_text())
    evidence_path = case_dir / "evidence.npz"
    responses = []
    with np.load(evidence_path, allow_pickle=False) as arrays:
        for row in summary["finite"]:
            state = row["state"]
            delta_heat = np.asarray(row["delta_heat"], dtype=np.float64)
            if delta_heat.shape != (len(source_ids),):
                raise ValueError(f"Saved counted change for {case_dir.name}/{state} is not source aligned")
            key = f"{state}/fluid_fields/"
            predicted = np.asarray(arrays[key + "delta_prediction_FP64"], dtype=np.float64)[:, 4]
            reference = np.asarray(arrays[key + "delta_reference_FP64"], dtype=np.float64)[:, 4]
            responses.append({"state": state, "baseline_state": row["baseline_state"],
                "scope": row["scope"], "delta_heat": delta_heat,
                "balanced_fixed_total": bool(abs(float(delta_heat.sum())) <= 2e-6),
                "cold_prediction_delta": predicted, "reference_delta": reference})
    return summary, responses


def _native_prepared_kernel(model, sample, case_id, device):
    thermal = model.thermal
    structure = {}
    raw_structure = sample["structure"]
    present = np.asarray(raw_structure["module_present"]) > .5
    ids = tuple(f"{case_id}:module:{i}" for i in range(len(present)))
    for key in ("module_centers", "module_present", "material_params", "re", "u_in",
                "domain_length_x", "domain_length_y"):
        structure[key] = torch.as_tensor(raw_structure[key], device=device, dtype=torch.float32)[None]
    structure["module_source_ids"] = ids
    xy = np.stack((np.asarray(sample["x_grid"]).reshape(-1), np.asarray(sample["y_grid"]).reshape(-1)), axis=-1)
    fluid_xy = torch.as_tensor(xy, device=device, dtype=torch.float32)[None]
    local = torch.as_tensor(sample["module_internal_query_points"], device=device, dtype=torch.float32)
    if local.ndim == 2:
        local = local[None]
    ntheta = int(np.asarray(sample["interface_condition"]).shape[-2])
    with torch.no_grad():
        prepared = thermal.prepare_native(structure, fluid_xy, local_query_points=local, ntheta=ntheta)
        kernels = thermal.export_native_kernels(prepared, accumulation_dtype=torch.float64)
    active_index = torch.as_tensor(np.flatnonzero(present), device=device, dtype=torch.long)
    kernel = kernels["fluid"][0, :, :, 0].index_select(1, active_index).detach().cpu().numpy().astype(np.float64, copy=False)
    return xy.astype(np.float64), kernel, np.asarray(raw_structure["module_centers"], dtype=np.float64)[present], ids, present


def verify_training_domain(checkpoint, domain, domain_path):
    """Re-derive selected-TRAIN heat limits and verify the retained TRAIN-only seal."""
    expected_scope = "150primaryTRAIN active amplitudes + baseline/positive saved inputs of fourdeclared TRAINaddendum families"
    if domain.get("input_scope") != expected_scope:
        raise ValueError("TRAIN domain source scope differs from the retained bounded calibration")
    if domain.get("calibration_uses_development_or_audit_outcomes") is not False or domain.get("physical_error_does_not_select_modes") is not True:
        raise ValueError("TRAIN domain seal must exclude development/audit outcomes and physical-error-selected modes")
    atlas_root = ROOT / "diagnostics/generated/interactions/physical_response_atlas_20260926/families"
    checked = []
    for source in domain.get("sources", []):
        family_id = str(source.get("family_id"))
        source_path = atlas_root / f"train_{family_id}_responses.npz"
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != source.get("sha256"):
            raise ValueError(f"Retained TRAIN addendum source hash changed for {family_id}")
        checked.append({"family_id": family_id, "path": str(source_path), "sha256": digest})
    if len(checked) != 4 or {item["family_id"] for item in checked} != {"0001", "0318", "0333", "0348"}:
        raise ValueError("TRAIN domain requires the exact four retained response addenda")
    train, manifest, dataset_path = load_native_cases(checkpoint, split="train")
    if len(train) != 150 or manifest["manifest_sha256"] != checkpoint["train_config"]["dataset"]["development_manifest_sha256"]:
        raise ValueError("TRAIN heat re-derivation requires the checkpoint's exact selected 150-case partition")
    values = []
    try:
        for index in range(len(train)):
            sample = train[index]
            present = np.asarray(sample["structure"]["module_present"]) > 0.5
            values.extend(np.asarray(sample["structure"]["heat_powers"], dtype=np.float64)[present].tolist())
    finally:
        train.close()
    minimum, maximum = min(values), max(values)
    if minimum != float(domain["train_heat_minimum"]) or maximum != float(domain["train_heat_maximum"]):
        raise ValueError("Retained TRAIN heat limits do not match a fresh metadata-only selected-TRAIN derivation")
    return {"receipt_path": str(domain_path), "input_scope": expected_scope,
        "selected_train_case_count": len(train), "selected_train_active_heat_value_count": len(values),
        "selected_train_heat_minimum_rederived": minimum, "selected_train_heat_maximum_rederived": maximum,
        "selected_train_manifest_sha256": manifest["manifest_sha256"], "dataset": str(dataset_path),
        "response_rms_from_same_train_only_seal": float(domain["sealed_train_temperature_response_rms"]),
        "addendum_sources_sha256_verified": checked,
        "response_rms_provenance": "retained TRAIN-only calibration seal; not re-estimated from counted or DEV response outcomes"}


def verify_counted_identity(counted_root, checkpoint_path, checkpoint_sha256, checkpoint_epoch):
    """Require the exact selected direct checkpoint and frozen counted records."""
    summary_path = Path(counted_root) / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Counted physical responses do not name the selected R-direct checkpoint SHA")
    if int(summary.get("checkpoint_epoch", -1)) != int(checkpoint_epoch):
        raise ValueError("Counted physical responses do not name the selected R-direct checkpoint epoch")
    if Path(summary.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("Counted physical response path identity differs from the selected checkpoint")
    if summary.get("frozen_state_unchanged") is not True or int(summary.get("solver_attempts", -1)) != 0 or int(summary.get("optimizer_updates", -1)) != 0:
        raise ValueError("Counted response receipt must confirm frozen checkpoint and zero solves/updates")
    family_ids = {str(row.get("case_id")) for row in summary.get("families", [])}
    if family_ids != set(CASES):
        raise ValueError("Counted physical response family set differs from the fixed four-case local view")
    return {"path": str(summary_path.resolve()),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "scope": summary.get("scope"), "cohort": summary.get("cohort"),
        "checkpoint": summary.get("checkpoint"), "checkpoint_epoch": summary.get("checkpoint_epoch"),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "frozen_state_unchanged": summary.get("frozen_state_unchanged"),
        "solver_attempts": summary.get("solver_attempts"),
        "optimizer_updates": summary.get("optimizer_updates"),
        "family_ids": sorted(family_ids)}


def run_local_view(checkpoint_path, counted_root, domain_path, output_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    counted_root = Path(counted_root).expanduser().resolve()
    domain_path = Path(domain_path).expanduser().resolve()
    output = validate_generated_output(output_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve measured arrays: choose a new empty local-view output directory")
    output.mkdir(parents=True, exist_ok=True)
    domain = json.loads(domain_path.read_text())
    if domain.get("train_heat_minimum") != TRAIN_HEAT_MINIMUM or domain.get("train_heat_maximum") != TRAIN_HEAT_MAXIMUM:
        raise ValueError("Local view must use the exact existing sealed TRAIN heat bounds")
    if domain.get("sealed_train_temperature_response_rms") != TRAIN_TEMPERATURE_RESPONSE_RMS:
        raise ValueError("Local view must use the exact existing sealed TRAIN temperature-response scale")
    torch.set_num_threads(1)
    start_unix, start = time(), perf_counter()
    model, checkpoint = load_source_response_model(checkpoint_path, device)
    model.eval().requires_grad_(False)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    counted_identity = verify_counted_identity(counted_root, checkpoint_path, checkpoint_sha256,
        int(checkpoint["epoch"]))
    train_domain_audit = verify_training_domain(checkpoint, domain, domain_path)
    dataset, manifest, dataset_path = load_native_cases(checkpoint, split="test")
    summary = {"scope": "post-fit R-direct receiver-local explanatory view; authoritative source-resolved forward path unchanged",
        "model": "R-direct", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint["epoch"]), "dataset": str(dataset_path),
        "counted_reference_audit": counted_identity,
        "development_manifest_sha256": manifest["manifest_sha256"],
        "receivers": "native 128x64 fluid temperature grid; equal-area physical 4x4 patches; fluid-only normalized quadrature per patch",
        "kernel_units": "dataset temperature units / physical heating units",
        "source_semantics": "physical source IDs preserved; per-source kernels remain learned estimates",
        "organization_semantics": "post-fit cover only; no sparse executor, speedup or causal attribution claim",
        "patches_per_case": 16, "patch_matrices": 64,
        "train_forcing_box": {"minimum": TRAIN_HEAT_MINIMUM, "maximum": TRAIN_HEAT_MAXIMUM,
            "radius_rule": "r_i=max(0,min(h_i-h_min,h_max-h_i,0.20 mean active heat)); balanced increments"},
        "train_temperature_response_rms": TRAIN_TEMPERATURE_RESPONSE_RMS,
        "train_domain_audit": train_domain_audit,
        "budget_fractions": list(BUDGET_FRACTIONS), "cases": []}
    for case_id in CASES:
        index = tuple(str(value) for value in dataset.selected_case_ids).index(case_id)
        sample = dataset[index]
        xy, kernel, centers, padded_ids, present = _native_prepared_kernel(model, sample, case_id, torch.device(device))
        source_ids = [padded_ids[i] for i in np.flatnonzero(present)]
        case_dir = counted_root / case_id
        old_operator = np.load(case_dir / "operator.npz", allow_pickle=False)
        old_ids = tuple(str(x) for x in old_operator["physical_source_ids"].reshape(-1))
        if tuple(source_ids) != old_ids:
            raise ValueError(f"Native full-kernel export source order differs from retained counted operator for {case_id}")
        kernel_old = np.asarray(old_operator["K"], dtype=np.float64)[0, :, :, 0]
        precision_audit = retained_kernel_precision_audit(kernel, kernel_old)
        heat = np.asarray(old_operator["physical_heat"], dtype=np.float64).reshape(-1)
        if heat.shape != (len(source_ids),):
            raise ValueError(f"Retained counted heat does not align with active sources for {case_id}")
        radii = bounded_radii(heat, np.ones_like(heat, dtype=bool))
        stored_summary, responses = _load_signed_responses(case_dir, source_ids)
        if bool(stored_summary["baseline_available"]) != (case_id != "0277"):
            raise ValueError(f"Counted baseline availability changed for {case_id}")
        baseline_state = responses[0]["baseline_state"] if responses else None
        absolute_state = next((row for row in stored_summary["absolute"] if row["state"] == baseline_state), None)
        if absolute_state is None or not np.array_equal(np.asarray(absolute_state["heat"], dtype=np.float64), heat):
            raise ValueError(f"Retained current-contribution heat does not match the exact saved baseline state for {case_id}")
        for response in responses:
            delta = response["delta_heat"]
            response["inside_declared_balanced_box"] = bool(response["balanced_fixed_total"] and
                delta.shape == radii.shape and np.all(np.abs(delta) <= radii + 2e-6))
        with np.load(case_dir / "evidence.npz", allow_pickle=False) as arrays:
            fluid_valid = np.asarray(arrays["fluid_fields/valid_mask"], dtype=bool)
            fluid_mask = fluid_valid[:, 4] if fluid_valid.ndim == 2 else fluid_valid.reshape(-1)
        catalog = make_patch_catalog(xy, fluid_mask,
            domain_x=float(np.asarray(sample["structure"]["domain_length_x"]).reshape(-1)[0]),
            domain_y=float(np.asarray(sample["structure"]["domain_length_y"]).reshape(-1)[0]))
        patch_scores = np.zeros((16, len(source_ids)), dtype=np.float64)
        patch_rows = []
        all_patch_rows = []
        for patch_index, patch_name in enumerate(catalog["names"]):
            rows = np.flatnonzero((catalog["labels"] == patch_index) & fluid_mask)
            weights = catalog["weights"][rows]
            all_patch_rows.append(rows)
            scores = response_scores(kernel[rows], weights, radii)
            patch_scores[patch_index] = scores
            patch_rows.append({"name": patch_name, "receiver_count": int(rows.size),
                "centroid_xy": catalog["centroids"][patch_index].tolist(), "scores": scores})
        npz_arrays = {"K_fluid_FP64": kernel, "current_signed_contributions_FP64": kernel * heat[None, :],
            "source_ids": np.asarray(source_ids, dtype=str), "source_centers": centers,
            "physical_heat": heat, "forcing_radii": radii, "fluid_xy": xy,
            "patch_labels": np.asarray([catalog["names"][label] for label in catalog["labels"]], dtype=str),
            "patch_weights": catalog["weights"], "patch_q_score": patch_scores,
            "fluid_valid_mask": fluid_mask.astype(np.uint8)}
        case_summary = {"case_id": case_id, "source_ids": source_ids,
            "source_heat": heat.tolist(), "source_radii": radii.tolist(),
            "source_centers": centers.tolist(),
            "counted_scope": counted_identity["scope"],
            "baseline_available": bool(stored_summary["baseline_available"]),
            "response_domain_checks": [{"state": response["state"], "baseline_state": response["baseline_state"],
                "scope": response["scope"], "delta_heat_sum": float(response["delta_heat"].sum()),
                "balanced_fixed_total": response["balanced_fixed_total"],
                "inside_declared_balanced_box": response["inside_declared_balanced_box"]}
                for response in responses],
            "finite_response_count": len(responses), "patches": []}
        for patch_index, patch in enumerate(patch_rows):
            rows = all_patch_rows[patch_index]
            weights = catalog["weights"][rows]
            geometry = geometry_orders(centers, catalog["centroids"][patch_index], source_ids,
                float(np.asarray(sample["structure"]["u_in"]).reshape(-1)[0]))
            for fraction in BUDGET_FRACTIONS:
                absolute_budget = fraction * TRAIN_TEMPERATURE_RESPONSE_RMS
                selected_cover = learned_cover(patch["scores"], absolute_budget, source_ids)
                retained = selected_cover["retained_count"]
                controls = {
                    "learned_q": selected_cover["selected_indices"],
                    "nearest_geometry": geometry["nearest"][:retained],
                    "upstream_geometry": geometry["upstream"][:retained],
                }
                method_results = {}
                for method, selected in controls.items():
                    cover_scores = patch["scores"]
                    omitted_ids = [source_ids[i] for i in range(len(source_ids)) if i not in set(selected)]
                    replay_rows = []
                    for response in responses:
                        result = make_response_row(kernel, response["delta_heat"], response["reference_delta"],
                            selected, rows, weights, cover_scores,
                            model_cold_delta=response["cold_prediction_delta"],
                            bound_applies=response["inside_declared_balanced_box"])
                        replay_rows.append({"state": response["state"],
                            "baseline_state": response["baseline_state"], "scope": response["scope"], **result})
                        if method == "learned_q":
                            npz_arrays[f"{case_id}/{patch['name']}/{fraction:g}/{response['state']}/full_kernel_delta"] = kernel[rows] @ response["delta_heat"]
                            npz_arrays[f"{case_id}/{patch['name']}/{fraction:g}/{response['state']}/reference_delta"] = response["reference_delta"][rows]
                    method_results[method] = {"selected_source_ids": [source_ids[i] for i in selected],
                        "omitted_source_ids": omitted_ids, "retained_K_local": len(selected),
                        "input_score_omitted_triangle_bound": float(cover_scores[[i for i in range(len(source_ids)) if i not in set(selected)]].sum()),
                        "finite_replays": replay_rows}
                case_summary["patches"].append({"patch": patch["name"], "receiver_count": patch["receiver_count"],
                    "centroid_xy": patch["centroid_xy"], "budget_fraction": fraction,
                    "absolute_temperature_budget": absolute_budget,
                    "learned_q_scores_by_source": {source_ids[i]: float(patch["scores"][i]) for i in range(len(source_ids))},
                    "learned_cover_minimal_for_budget": selected_cover["minimal_for_budget"],
                    "geometry_controls_match_learned_K_local": all(len(control) == retained for control in controls.values()),
                    "methods": method_results})
        npz_arrays["finite_response_states"] = np.asarray([row["state"] for row in responses], dtype=str)
        for response in responses:
            npz_arrays[f"finite/{response['state']}/delta_heat"] = response["delta_heat"]
            npz_arrays[f"finite/{response['state']}/cold_prediction_delta"] = response["cold_prediction_delta"]
            npz_arrays[f"finite/{response['state']}/reference_delta"] = response["reference_delta"]
        case_output = output / case_id
        case_output.mkdir(exist_ok=False)
        np.savez_compressed(case_output / "kernels.npz", **npz_arrays)
        case_summary["arrays"] = str(case_output / "kernels.npz")
        case_summary["patch_matrix_count"] = len(patch_rows)
        case_summary["patch_budget_cover_count"] = len(case_summary["patches"])
        case_summary["native_kernel_vs_retained_FP32_difference"] = precision_audit
        case_summary["kernel_vs_saved_cold_delta_max_abs_by_state"] = {
            response["state"]: float(np.max(np.abs(kernel @ response["delta_heat"] - response["cold_prediction_delta"])))
            for response in responses}
        atomic_json(case_output / "summary.json", case_summary)
        summary["cases"].append(case_summary)
        print(json.dumps({"case_id": case_id, "sources": len(source_ids), "patch_matrices": len(case_summary["patches"]),
            "finite_responses": len(responses), "native_kernel_vs_retained_FP32_difference": precision_audit}), flush=True)
    summary["timing_scope"] = "CPU/GPU-independent measured native FP64 kernel export and deterministic post-fit cover calculation; no solver, optimizer, fit or inverse calls."
    summary["elapsed_seconds"] = perf_counter() - start
    summary["solver_attempts"] = 0
    summary["optimizer_updates"] = 0
    summary["new_inverse_designs"] = 0
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "receipt.json", {"start_unix": start_unix, "end_unix": time(),
        "elapsed_seconds": perf_counter() - start, "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "status": "completed", "solver_attempts": 0, "optimizer_updates": 0, "new_inverse_designs": 0})
    return summary


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--counted-root", type=Path, required=True)
    parser.add_argument("--compression-domain", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    run_local_view(args.checkpoint, args.counted_root, args.compression_domain, args.output_dir, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
