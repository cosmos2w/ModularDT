"""Supervised, input-only fit and disjoint evaluation for G-cover plans.

This workflow consumes only coherent plans that passed the fixed teacher gate
on both search and disjoint verification probes. It stops before optimization
when the bounded evidence has no variable, cost-competitive hard-cover target.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_forward_core.interface_fields.adaptive_interaction_cover import AdaptiveCoverPlan
from honf_forward_core.interface_fields.input_cover_organizer import (
    InputOnlyCoverOrganizer,
    masked_binary_logit_loss,
)
from torch.nn.utils import clip_grad_norm_

from ..geometry import support_weights
from ..normalization import VelocityNormalizer
from ..study_spatial import WeightedVelocityErrors, downstream_envelope, native_coordinates
from . import native_cover_organizer as base
from .evaluate_forward import load_checkpoint
from .native_cover_panel import make_disjoint_native_probes

FIT_UPDATES = 100
FIT_SEED = 2103
MIN_COMPETITIVE_PARTIAL_LAYOUTS = 2


def _load_fit_artifacts(oracle_dir: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    directory = Path(oracle_dir).expanduser().resolve()
    report_path = directory / "oracle_benchmark.json"
    manifest_path = directory / "panel_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("oracle directory must contain oracle_benchmark.json and panel_manifest.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
        raise ValueError("oracle report and frozen panel manifest checkpoint hashes differ")
    documents: dict[int, dict[str, Any]] = {}
    for row in report.get("rows", []):
        row_index = int(row["row_index"])
        plan_path = directory / f"selected_plan_row_{row_index}.json"
        if plan_path.is_file():
            documents[row_index] = json.loads(plan_path.read_text(encoding="utf-8"))
    return directory, report, manifest, documents


def _teacher_verified_partial(row: dict[str, Any], plan_document: dict[str, Any] | None) -> bool:
    """Require one coherent partial plan with direct search and disjoint gates."""

    winner = row.get("teacher_oracle_search_winner")
    if not isinstance(winner, dict) or plan_document is None:
        return False
    proposal = winner.get("proposal")
    if not isinstance(proposal, dict) or proposal.get("kind") == "root_full_access":
        return False
    verification = winner.get("disjoint_verification", {})
    verification_gate = verification.get("teacher_preservation", {}).get("gate_0p10")
    if not bool(winner.get("teacher_search_gate_passed")):
        return False
    if not bool(verification_gate) or not bool(winner.get("teacher_adequate_for_primary_labels")):
        return False
    if int(winner.get("actual_hard_support", {}).get("hard_cover_k", 0)) < 1:
        return False
    supervision = plan_document.get("primary_supervision")
    if not isinstance(supervision, dict):
        return False
    try:
        split = np.asarray(supervision["split_targets"], dtype=np.float64)
        module = np.asarray(supervision["module_targets"], dtype=np.float64)
        environment = np.asarray(supervision["environment_targets"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    winner_arrays = (
        winner.get("split_gates"),
        winner.get("module_membership"),
        winner.get("environment_membership"),
    )
    if any(value is None for value in winner_arrays):
        return False
    try:
        exact_winner_arrays = tuple(np.asarray(value, dtype=np.float64) for value in winner_arrays)
    except (TypeError, ValueError):
        return False
    if not (
        np.array_equal(split, exact_winner_arrays[0])
        and np.array_equal(module, exact_winner_arrays[1])
        and np.array_equal(environment, exact_winner_arrays[2])
    ):
        return False
    return bool(np.any(split > 0.5) or np.any(module < 0.5) or np.any(environment < 0.5))


def _assess_fit_target(
    oracle_report: dict[str, Any],
    panel_manifest: dict[str, Any],
    row_documents: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Decide whether the frozen, directly observed labels justify organizer fitting."""

    rows = list(oracle_report.get("rows", []))
    dense_baselines = {
        int(item["row_index"]): float(item["dense_prepare_ms"]) + float(item["dense_search_decode_ms"])
        for item in oracle_report.get("baseline_timings", [])
        if "row_index" in item and item.get("dense_prepare_ms") is not None
        and item.get("dense_search_decode_ms") is not None
    }
    selected_partial_rows: list[dict[str, Any]] = []
    train_search_accepted_partial = 0
    search_only_accepted_partial = 0
    accepted_partial_faster_than_dense = 0
    label_signatures: set[str] = set()
    for row in rows:
        row_index = int(row["row_index"])
        row_document = row_documents.get(row_index)
        winner = row.get("teacher_oracle_search_winner", {})
        selected_verified = _teacher_verified_partial(row, row_document)
        if selected_verified:
            support = winner.get("actual_hard_support", {})
            k = int(support.get("hard_cover_k", 0))
            actual_ms = float(winner.get("actual_synchronized_complete_ms", float("inf")))
            dense_ms = dense_baselines.get(row_index)
            selected_partial_rows.append({
                "row_index": row_index,
                "layout_index": int(row.get("layout_index", -1)),
                "hard_cover_k": k,
                "partial_complete_ms": actual_ms,
                "policy_none_dense_complete_ms": dense_ms,
                "faster_than_policy_none_dense": bool(
                    dense_ms is not None
                    and math.isfinite(actual_ms)
                    and math.isfinite(dense_ms)
                    and 0.0 < actual_ms < dense_ms
                ),
            })
            supervision = row_document["primary_supervision"]
            label_signatures.add(json.dumps({
                "split_targets": supervision["split_targets"],
                "module_targets": supervision["module_targets"],
                "environment_targets": supervision["environment_targets"],
            }, sort_keys=True, separators=(",", ":")))
        selected_observation_consumed = False
        for trial in row.get("candidate_observations", []):
            proposal = trial.get("proposal", {})
            if bool(trial.get("accepted_by_oracle")) and proposal.get("kind") != "root_full_access":
                train_search_accepted_partial += 1
                selected_observation = bool(
                    selected_verified
                    and not selected_observation_consumed
                    and proposal == winner.get("proposal")
                    and float(trial.get("actual_synchronized_complete_ms", float("inf")))
                    == float(winner.get("actual_synchronized_complete_ms", float("inf")))
                )
                if selected_observation:
                    selected_observation_consumed = True
                    continue
                search_only_accepted_partial += 1
                dense_ms = dense_baselines.get(row_index)
                if (
                    dense_ms is not None
                    and math.isfinite(dense_ms)
                    and 0.0 < float(trial.get("actual_synchronized_complete_ms", float("inf"))) < dense_ms
                ):
                    accepted_partial_faster_than_dense += 1

    qualifying = [item for item in selected_partial_rows if item["faster_than_policy_none_dense"]]
    qualifying_layouts = sorted({item["layout_index"] for item in qualifying})
    active_layouts = int(oracle_report.get("active_training_layouts", 0))
    panel_layout_count = int(panel_manifest.get("layout_count", 0))
    expected_active_rows = sum(
        len(item.get("training_rows_direction_order", []))
        for item in panel_manifest.get("layouts", [])
        if item.get("activated_in_this_run")
    )
    expected_active_row_ids = {
        int(row_index)
        for item in panel_manifest.get("layouts", [])
        if item.get("activated_in_this_run")
        for row_index in item.get("training_rows_direction_order", [])
    }
    actual_rows = len(rows)
    actual_row_ids = {int(row["row_index"]) for row in rows}
    reasons: list[str] = []
    if panel_layout_count != 12 or len(panel_manifest.get("layouts", [])) != 12:
        reasons.append("frozen checkpoint-owned panel does not contain all 12 layout identities")
    if actual_rows != expected_active_rows or actual_row_ids != expected_active_row_ids:
        reasons.append("oracle row identities do not match the activated frozen-panel direction rows")
    if active_layouts < MIN_COMPETITIVE_PARTIAL_LAYOUTS:
        reasons.append(
            f"oracle evidence covers fewer than {MIN_COMPETITIVE_PARTIAL_LAYOUTS} independent layouts"
        )
    if not selected_partial_rows:
        reasons.append("no coherent partial plan is directly recorded with both search and disjoint teacher gates")
    if not qualifying:
        reasons.append("no directly observed partial plan beats policy=None Dense on complete measured work")
    if len(qualifying_layouts) < MIN_COMPETITIVE_PARTIAL_LAYOUTS:
        reasons.append(
            f"cost-competitive partial labels cover fewer than {MIN_COMPETITIVE_PARTIAL_LAYOUTS} independent layouts"
        )
    if len(label_signatures) < 2:
        reasons.append("coherent verified partial labels do not vary across cases")
    available = not reasons
    return {
        "status": "fit_eligible" if available else "target_unavailable",
        "evidence_scope": {
            "frozen_layout_identity_count": panel_layout_count,
            "activated_training_layouts": active_layouts,
            "observed_direction_rows": actual_rows,
            "expected_rows_for_activated_layouts": expected_active_rows,
            "query_count_per_search_and_disjoint_verification": oracle_report.get(
                "query_count_per_disjoint_search_and_verification_probe"
            ),
            "scope_note": (
                "Oracle labels and cost evidence are limited to activated train layouts; this does not claim a population result."
            ),
        },
        "partial_evidence": {
            "directly_search_and_disjoint_verified_partial_rows": selected_partial_rows,
            "train_search_accepted_partial_observations": train_search_accepted_partial,
            "search_only_accepted_partial_observations_not_used_as_joint_labels": search_only_accepted_partial,
            "search_acceptance_semantics": (
                "These proposals passed the train-search oracle. They are not called teacher-inadequate; the available artifact lacks a coherent selected-plan label and a disjoint verification record for them."
            ),
            "search_only_accepted_candidates_faster_than_policy_none_dense": accepted_partial_faster_than_dense,
            "distinct_verified_primary_label_signatures": len(label_signatures),
            "measured_faster_partial_layouts": qualifying_layouts,
        },
        "dense_cost_comparator": {
            "definition": "policy=None native Dense prepare_case plus decode synchronized timings from the same search probes",
            "per_row_complete_ms": dense_baselines,
        },
        "optimizer": {
            "updates": 0 if not available else FIT_UPDATES,
            "class_balance": "equal mean loss for observed positive and negative decisions when both classes occur",
            "status": "skipped" if not available else "eligible_for_bounded_100_update_review",
        },
        "reasons": reasons,
        "controls": {
            "fixed_cover": "not_evaluated_target_unavailable" if not available else "evaluate_on_disjoint_queries",
            "collapsed_pair_control": "not_evaluated_target_unavailable" if not available else "evaluate_on_disjoint_queries",
        },
        "interpretation": (
            "The available bounded evidence does not establish a deployable adaptive cover target."
            if not available
            else "A bounded supervised organizer fit is justified for review; no deployment claim follows from fitting alone."
        ),
    }


def write_fit_target_assessment(
    *,
    checkpoint_path: str | Path,
    oracle_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the CPU-only target-availability decision from frozen oracle artifacts."""

    directory, report, manifest, documents = _load_fit_artifacts(oracle_dir)
    requested_checkpoint = Path(checkpoint_path).expanduser().resolve()
    requested_checkpoint_hash = base._checkpoint_sha256(requested_checkpoint)
    if requested_checkpoint_hash != report.get("checkpoint_sha256"):
        raise ValueError("requested fit checkpoint bytes differ from the frozen oracle checkpoint")
    assessment = _assess_fit_target(report, manifest, documents)
    assessment.update({
        "workflow": "native_cover_input_only_organizer_target_assessment",
        "oracle_directory": str(directory),
        "requested_checkpoint": str(requested_checkpoint),
        "checkpoint_sha256": requested_checkpoint_hash,
        "teacher_checkpoint_id": f"Run2103:e2475:{requested_checkpoint_hash[:16]}",
        "oracle_evidence_mode": "teacher_preservation",
        "teacher_distortion_limit": report.get("oracle_limits", {}).get(
            "maximum_normalized_vector_rmse_by_role", base.TEACHER_DISTORTION_LIMIT
        ),
        "physical_reference_sufficiency_claimed": False,
    })
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "fit_target_assessment.json"
    result_path.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assessment["report_path"] = str(result_path)
    return assessment


def _label_tensors(document: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    labels = document["primary_supervision"]
    return {
        "split_targets": torch.as_tensor(labels["split_targets"], device=device, dtype=torch.float32),
        "split_observed": torch.as_tensor(labels["split_observed"], device=device, dtype=torch.bool),
        "module_targets": torch.as_tensor(labels["module_targets"], device=device, dtype=torch.float32),
        "module_observed": torch.as_tensor(labels["module_observed"], device=device, dtype=torch.bool),
        "environment_targets": torch.as_tensor(labels["environment_targets"], device=device, dtype=torch.float32),
        "environment_observed": torch.as_tensor(labels["environment_observed"], device=device, dtype=torch.bool),
    }


def _fit_row_ids_to_use(active_rows: list[int], assessment: dict[str, Any]) -> list[int]:
    """Keep only rows with one directly selected, disjoint-verified partial label."""

    eligible = {
        int(item["row_index"])
        for item in assessment["partial_evidence"]["directly_search_and_disjoint_verified_partial_rows"]
    }
    selected = [int(row) for row in active_rows if int(row) in eligible]
    if not selected:
        raise ValueError("fit-eligible assessment contains no directly verified training rows")
    return selected


def _plan_from_labels(record: Any, labels: dict[str, torch.Tensor]) -> AdaptiveCoverPlan:
    return AdaptiveCoverPlan(
        record.trees[0],
        labels["split_targets"],
        labels["module_targets"],
        labels["environment_targets"],
    )


def _organizer_state(record: Any) -> dict[str, torch.Tensor]:
    dense = record.full_prepared.dense_prepared
    return {
        "module_states": dense.module_states.detach(),
        "environment_states": record.encoded.env_tokens.detach(),
        "global_state": record.encoded.global_token.detach(),
    }


def _training_loss(organizer: InputOnlyCoverOrganizer, record: Any, labels: dict[str, torch.Tensor]) -> torch.Tensor:
    scores = organizer.score_cases(record.encoded, _organizer_state(record), record.trees)[0]
    return torch.stack((
        masked_binary_logit_loss(
            scores.split_logits, labels["split_targets"], labels["split_observed"], balance_classes=True
        ),
        masked_binary_logit_loss(
            scores.module_logits, labels["module_targets"], labels["module_observed"], balance_classes=True
        ),
        masked_binary_logit_loss(
            scores.environment_logits,
            labels["environment_targets"],
            labels["environment_observed"],
            balance_classes=True,
        ),
    )).mean()


def _native_teacher_role_errors(
    model: Any,
    candidate_prepared: Any,
    teacher_prepared: Any,
    case: Any,
    normalizer: VelocityNormalizer,
    *,
    device: torch.device,
    chunk_size: int = 8192,
) -> dict[str, Any]:
    """Compare fitted and G-full physical predictions over complete native cells."""

    run = case.run
    wx, wy, wz = support_weights(run.x_m, run.y_m, run.z_m)
    wx = np.asarray(wx, dtype=np.float64) / float(case.diameter_m)
    wy = np.asarray(wy, dtype=np.float64) / float(case.diameter_m)
    wz = np.asarray(wz, dtype=np.float64) / float(case.diameter_m)
    accumulators = {
        name: WeightedVelocityErrors()
        for name in ("volume", "hub_slab", "downstream_envelope", "background", "near_turbine")
    }
    hubs = np.asarray(case.module_centers, dtype=np.float64)
    hub_height = float(case.hub_height_m) / float(case.diameter_m)
    with torch.inference_mode():
        for start in range(0, int(run.cell_count), chunk_size):
            stop = min(start + chunk_size, int(run.cell_count))
            flat = np.arange(start, stop, dtype=np.int64)
            coords_D = native_coordinates(run, flat, float(case.diameter_m))
            geometry = case.geometry_for_queries(coords_D)
            query = torch.as_tensor(geometry["query_xy"], device=device).unsqueeze(0)
            features = torch.as_tensor(geometry["query_features"], device=device).unsqueeze(0)
            candidate = model.decode(candidate_prepared, query, query_features=features)["pred_field"][0]
            teacher = model.decode(teacher_prepared, query, query_features=features)["pred_field"][0]
            mean = candidate.new_tensor(normalizer.mean)
            scale = candidate.new_tensor(normalizer.safe_std)
            candidate_physical = ((candidate * scale + mean) * float(normalizer.u_ref_mps)).cpu().numpy()
            teacher_physical = ((teacher * scale + mean) * float(normalizer.u_ref_mps)).cpu().numpy()
            ix = flat % int(run.nx)
            iy = (flat // int(run.nx)) % int(run.ny)
            iz = flat // (int(run.nx) * int(run.ny))
            weights = wx[ix] * wy[iy] * wz[iz]
            hub_slab = np.abs(coords_D[:, 2] - hub_height) <= 0.5
            wake = downstream_envelope(coords_D, hubs)
            relative_xy = coords_D[:, None, :2] - hubs[None, :, :2]
            near = (np.sum(relative_xy**2, axis=-1).min(axis=1) <= 1.5**2) & hub_slab
            masks = {
                "volume": np.ones(len(flat), dtype=bool),
                "hub_slab": hub_slab,
                "downstream_envelope": wake,
                "background": ~wake & ~hub_slab,
                "near_turbine": near,
            }
            for name, mask in masks.items():
                if bool(mask.any()):
                    accumulators[name].add(candidate_physical[mask], teacher_physical[mask], weights[mask])
    physical_std = normalizer.safe_std * float(normalizer.u_ref_mps)
    return {
        role: accumulator.result(physical_std, float(normalizer.u_ref_mps))
        for role, accumulator in accumulators.items()
    }


def run_organizer_fit(
    *,
    checkpoint_path: str | Path,
    volume_path: str | Path,
    compact_path: str | Path,
    oracle_dir: str | Path,
    device: str | torch.device,
    output_dir: str | Path,
    query_count: int = 1024,
    seed: int = FIT_SEED,
    updates: int = FIT_UPDATES,
    native_validation_layouts: int = 2,
) -> dict[str, Any]:
    """Fit exactly one bounded organizer when the oracle target is available."""

    if updates != FIT_UPDATES:
        raise ValueError(f"the review run is frozen at exactly {FIT_UPDATES} optimizer updates")
    if query_count != 1024:
        raise ValueError("organizer review uses the frozen disjoint Q1024 search and verification probes")
    if not 1 <= native_validation_layouts <= 2:
        raise ValueError("native validation is bounded to one or two geometry-stratified layouts")
    destination = Path(output_dir).expanduser().resolve()
    if "generated" not in destination.parts:
        raise ValueError("organizer fit outputs must stay under an ignored diagnostics/generated directory")
    assessment = write_fit_target_assessment(
        checkpoint_path=checkpoint_path, oracle_dir=oracle_dir, output_dir=destination
    )
    if assessment["status"] != "fit_eligible":
        return assessment

    oracle_directory, oracle_report, oracle_manifest, plan_documents = _load_fit_artifacts(oracle_dir)
    (
        checkpoint,
        payload,
        checkpoint_hash,
        view,
        _split,
        layouts,
        _manifest_destination,
        frozen_manifest,
    ) = base._freeze_training_panel(
        checkpoint_path=checkpoint_path,
        volume_path=volume_path,
        compact_path=compact_path,
        output_dir=destination,
        layout_count=12,
        active_layouts=int(oracle_report.get("active_training_layouts", 0)),
        anchor_measure_variant=str(oracle_manifest.get("anchor_measure_variant", "raw")),
    )
    if checkpoint_hash != oracle_report.get("checkpoint_sha256"):
        raise ValueError("fit checkpoint differs from the oracle artifact checkpoint")
    if frozen_manifest.get("layouts") != oracle_manifest.get("layouts"):
        raise ValueError("fit panel geometry identities differ from the oracle frozen panel")
    active_rows = [row for layout in layouts for row in layout.rows]
    active_rows = _fit_row_ids_to_use(active_rows, assessment)
    normalizer = VelocityNormalizer.from_dict(dict(payload["normalization"]))
    target_device = torch.device(device)
    first_case = view.run(active_rows[0])
    first_search, _ = make_disjoint_native_probes(first_case, query_count=query_count, seed=seed)
    materialization_batch = base._model_batch(
        base._probe_batch(first_case, first_search, normalizer), target_device
    )
    model, _loaded = load_checkpoint(checkpoint, device=target_device, materialization_batch=materialization_batch)
    if model.architecture != "dense_pairwise_field":
        raise ValueError("input-only fitting must keep the exact Dense Run2103 teacher frozen")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    teacher_checkpoint_id = f"Run2103:e2475:{checkpoint_hash[:16]}"

    records: list[Any] = []
    label_by_row: dict[int, dict[str, torch.Tensor]] = {}
    for row in active_rows:
        case = view.run(row)
        record, _timing = base._build_panel_case(
            model,
            case,
            query_count=query_count,
            seed=seed,
            teacher_checkpoint_id=teacher_checkpoint_id,
            device=target_device,
            anchor_measure_variant=str(oracle_manifest.get("anchor_measure_variant", "raw")),
        )
        document = plan_documents.get(int(row))
        if document is None:
            raise ValueError(f"missing directly verified partial plan label for row {row}")
        labels = _label_tensors(document, target_device)
        record.oracle_plan = _plan_from_labels(record, labels)
        records.append(record)
        label_by_row[int(row)] = labels

    first = records[0]
    encoded = first.encoded
    torch.manual_seed(seed)
    organizer = InputOnlyCoverOrganizer(
        state_dim=int(encoded.module_tokens.shape[-1]),
        module_feature_dim=int(encoded.module_features.shape[-1]),
        environment_feature_dim=0 if encoded.env_features is None else int(encoded.env_features.shape[-1]),
        hidden_dim=32,
        role_count=max(8, int(first.trees[0].universe.roles.max().item()) + 1),
    ).to(target_device)
    optimizer = torch.optim.AdamW(organizer.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(records)) for _ in range((updates + len(records) - 1) // len(records))])
    losses: list[float] = []
    fit_started = time.perf_counter()
    organizer.train()
    for update, index in enumerate(order[:updates], start=1):
        record = records[int(index)]
        optimizer.zero_grad(set_to_none=True)
        loss = _training_loss(organizer, record, label_by_row[int(record.case.index)])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite organizer supervised loss at update {update}")
        loss.backward()
        clip_grad_norm_(organizer.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    organizer.eval()
    fit_elapsed = time.perf_counter() - fit_started
    state_path = destination / "input_only_organizer_state.pt"
    torch.save({
        "state_dict": organizer.state_dict(),
        "checkpoint_sha256": checkpoint_hash,
        "updates": updates,
        "seed": seed,
        "architecture": "InputOnlyCoverOrganizer",
        "teacher_checkpoint_id": teacher_checkpoint_id,
    }, state_path)

    module_mask, environment_mask, fixed_summary = base._population_fixed_masks(records)
    per_row: list[dict[str, Any]] = []
    fitted_prepared_by_layout: dict[int, Any] = {}
    for record in records:
        row_index = int(record.case.index)
        dense_report, _dense_prepared, _dense_plan = base._evaluate_probe_variant(
            model,
            record,
            name="policy-none-Dense",
            probe=record.verification,
            batch=record.verification_batch,
            policy=None,
            reference_prediction=record.full_verification_prediction,
            teacher_checkpoint_id=teacher_checkpoint_id,
            device=target_device,
        )
        fitted_report, fitted_prepared, fitted_plan = base._evaluate_probe_variant(
            model,
            record,
            name="G-input-only-organizer-hard",
            probe=record.verification,
            batch=record.verification_batch,
            policy=organizer,
            reference_prediction=record.full_verification_prediction,
            teacher_checkpoint_id=teacher_checkpoint_id,
            device=target_device,
        )
        fixed_modules = torch.as_tensor(
            module_mask[: int(record.oracle_plan.module_membership.shape[1])],
            device=record.oracle_plan.module_membership.device,
            dtype=record.oracle_plan.module_membership.dtype,
        )
        fixed_environment = torch.as_tensor(
            environment_mask,
            device=record.oracle_plan.environment_membership.device,
            dtype=record.oracle_plan.environment_membership.dtype,
        )
        fixed_plan = base._root_mask_plan(record.full_plan, fixed_modules, fixed_environment)
        fixed_report, _fixed_prepared, _fixed_plan = base._evaluate_probe_variant(
            model,
            record,
            name="training-population-fixed-cover-control",
            probe=record.verification,
            batch=record.verification_batch,
            policy=base._FixedPlanPolicy((fixed_plan,)),
            reference_prediction=record.full_verification_prediction,
            teacher_checkpoint_id=teacher_checkpoint_id,
            device=target_device,
        )
        pair_plan = base._per_case_pair_control(record)
        pair_report, _pair_prepared, _pair_plan = base._evaluate_probe_variant(
            model,
            record,
            name="per-case-collapsed-pair-control",
            probe=record.verification,
            batch=record.verification_batch,
            policy=base._FixedPlanPolicy((pair_plan,)),
            reference_prediction=record.full_verification_prediction,
            teacher_checkpoint_id=teacher_checkpoint_id,
            device=target_device,
        )
        per_row.append({
            "row_index": row_index,
            "layout_index": int(record.case.layout_index),
            "direction_deg": float(record.case.wind_direction_deg),
            "probe_label": "disjoint Q1024 train probe; teacher and reference errors reported separately",
            "dense": dense_report,
            "fitted_hard_organizer": fitted_report,
            "fixed_cover_control": fixed_report,
            "collapsed_pair_control": pair_report,
            "fitted_hard_plan": {
                "split_gates": fitted_plan.split_gates.detach().cpu().tolist(),
                "module_membership": fitted_plan.module_membership.detach().cpu().tolist(),
                "environment_membership": fitted_plan.environment_membership.detach().cpu().tolist(),
                "hard_k": int(fitted_plan.active_group_count()),
            },
        })
        if int(record.case.layout_index) not in fitted_prepared_by_layout:
            fitted_prepared_by_layout[int(record.case.layout_index)] = fitted_prepared

    validation_layouts = list(dict.fromkeys(int(record.case.layout_index) for record in records))[
        :native_validation_layouts
    ]
    native_validation: list[dict[str, Any]] = []
    for layout_index in validation_layouts:
        row_index = next(row for row in active_rows if int(view.run(row).layout_index) == layout_index)
        record = next(item for item in records if int(item.case.index) == int(row_index))
        fitted_prepared = fitted_prepared_by_layout[layout_index]
        fitted_metrics, fitted_ms = base._native_grid_metrics(
            model, fitted_prepared, record.case, normalizer, device=target_device
        )
        teacher_delta = _native_teacher_role_errors(
            model,
            fitted_prepared,
            record.full_prepared,
            record.case,
            normalizer,
            device=target_device,
        )
        teacher_reference_metrics, teacher_ms = base._native_grid_metrics(
            model, record.full_prepared, record.case, normalizer, device=target_device
        )
        native_validation.append({
            "row_index": int(row_index),
            "layout_index": layout_index,
            "native_cell_count": int(record.case.run.cell_count),
            "interpretation": "complete native-grid physical-reference and direct G-fit minus G-full diagnostics",
            "fitted_physical_reference_metrics": fitted_metrics,
            "fitted_native_decode_ms": float(fitted_ms),
            "G_full_physical_reference_metrics": teacher_reference_metrics,
            "G_full_native_decode_ms": float(teacher_ms),
            "direct_G_fit_minus_G_full_role_errors": teacher_delta,
        })

    result = {
        **assessment,
        "workflow": "native_cover_input_only_organizer_fit",
        "oracle_directory": str(oracle_directory),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "device": str(target_device),
        "teacher_checkpoint_id": teacher_checkpoint_id,
        "frozen_model": {"architecture": "dense_pairwise_field", "eval_mode": bool(not model.training)},
        "training": {
            "updates": updates,
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "loss": "pre-activation BCE; observed positive and negative classes equally weighted when both occur",
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_mean": float(np.mean(losses)) if losses else None,
            "elapsed_seconds": fit_elapsed,
            "state_path": str(state_path),
        },
        "fixed_cover_control_definition": fixed_summary,
        "disjoint_verification_rows": per_row,
        "native_grid_validation": native_validation,
        "interpretation": "A fitted deterministic hard policy is reported only if its actual K, source support, teacher distortion, physical-reference errors, and complete measured work pass review; fitting alone does not establish speedup or physical sufficiency.",
    }
    (destination / "organizer_fit_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


__all__ = ["run_organizer_fit", "write_fit_target_assessment"]
