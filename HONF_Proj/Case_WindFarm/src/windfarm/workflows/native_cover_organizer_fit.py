"""Supervised, input-only fit and disjoint evaluation for G-cover plans.

This workflow consumes only coherent plans that passed the fixed teacher gate
on both search and disjoint verification probes. Learning eligibility is
reported independently from reference sufficiency and deployment cost.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    INTERACTION_MECHANISMS,
    AdaptiveCoverPlan,
    InteractionContext,
    MechanismPlan,
)
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
from .native_cover_panel import (
    TrainingLayout,
    freeze_organizer_layout_split,
    make_disjoint_native_probes,
)

FIT_UPDATES = 100
FIT_SEED = 2103
MIN_COMPETITIVE_PARTIAL_LAYOUTS = 2
TYPED_SUPERVISED_UPDATE_CAP = 150
TYPED_PREDICTIVE_UPDATE_CAP = 500
TYPED_FORMAL_UPDATE_CAP = 1400  # 1,500 cumulative, including the historical 100-step G0 diagnostic.
TYPED_G0_PRIOR_UPDATES = 100
TYPED_WORK_PENALTY_WEIGHT = 0.01
TYPED_SUPERVISED_FEEDBACK_WEIGHT = 0.25
TYPED_REFERENCE_FEEDBACK_WEIGHT = 0.25
TYPED_ALL_ACCESS_PERMISSION_LOGIT = 2.0
TYPED_ALL_ACCESS_SPLIT_LOGIT = -2.0
TYPED_MIN_PROTECTED_PREDICTION_GRADIENT_L2 = 1.0e-8
TYPED_ORGANIZER_QUADRATURE_MEASURE_SCHEMA = "exact-coordinate-feature-summed-atom-mean-v1"


def _frozen_organizer_split_from_manifest(panel_manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate and recover the geometry-only split from a frozen panel."""

    try:
        layouts = tuple(
            TrainingLayout(
                layout_index=int(item["layout_index"]),
                rows=tuple(map(int, item["training_rows_direction_order"])),
                turbine_count=int(item["turbine_count"]),
                feature_vector=tuple(map(float, item["geometry_feature_vector"])),
            )
            for item in panel_manifest["layouts"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen panel manifest lacks geometry-only organizer split inputs") from exc
    split = freeze_organizer_layout_split(layouts)
    recorded = panel_manifest.get("organizer_split_frozen_before_new_outcomes")
    if recorded is not None and recorded.get("split_sha256") != split["split_sha256"]:
        raise ValueError("panel manifest organizer split does not match the frozen geometry identities")
    return split


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
    """Report learning, reference, and deployment eligibility independently."""

    rows = list(oracle_report.get("rows", []))
    split = _frozen_organizer_split_from_manifest(panel_manifest)
    training_layouts = set(map(int, split["training_layout_indices"]))
    development_layouts = set(map(int, split["development_layout_indices"]))
    training_rows = set(map(int, split["training_rows_direction_order"]))
    row_to_layout = {
        int(row_index): int(layout["layout_index"])
        for layout in panel_manifest.get("layouts", [])
        for row_index in layout.get("training_rows_direction_order", [])
    }
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
    data_reasons: list[str] = []
    if panel_layout_count != 12 or len(panel_manifest.get("layouts", [])) != 12:
        data_reasons.append("frozen checkpoint-owned panel does not contain all 12 layout identities")
    if actual_rows != expected_active_rows or actual_row_ids != expected_active_row_ids:
        data_reasons.append("oracle row identities do not match the activated frozen-panel direction rows")
    if actual_rows != len(actual_row_ids):
        data_reasons.append("oracle report contains duplicate native direction rows")
    if not actual_row_ids.issubset(training_rows):
        data_reasons.append("oracle labels include a layout outside the frozen organizer training split")
    if any(
        row_to_layout.get(int(row["row_index"])) != int(row.get("layout_index", -1))
        for row in rows
    ):
        data_reasons.append("oracle row-to-layout identities disagree with the geometry-only panel manifest")
    if not selected_partial_rows:
        data_reasons.append("no coherent partial plan is directly recorded with both search and disjoint teacher gates")

    learning_eligible = not data_reasons
    speed_eligible = len(qualifying_layouts) >= MIN_COMPETITIVE_PARTIAL_LAYOUTS
    deployment_reasons: list[str] = []
    if not qualifying:
        deployment_reasons.append("no directly measured partial plan beats policy=None Dense on complete measured work")
    if len(qualifying_layouts) < MIN_COMPETITIVE_PARTIAL_LAYOUTS:
        deployment_reasons.append(
            f"cost-competitive partial labels cover fewer than {MIN_COMPETITIVE_PARTIAL_LAYOUTS} independent layouts"
        )
    deployment_reasons.append("partial-plan physical-reference sufficiency has not been established")
    return {
        "status": "learning_eligible" if learning_eligible else "learning_ineligible",
        "learning_eligible": learning_eligible,
        "reference_sufficient": "unknown",
        "measured_speed_eligible": speed_eligible,
        "deployment_eligible": False,
        "evidence_scope": {
            "frozen_layout_identity_count": panel_layout_count,
            "activated_training_layouts": active_layouts,
            "observed_direction_rows": actual_rows,
            "expected_rows_for_activated_layouts": expected_active_rows,
            "query_count_per_search_and_disjoint_verification": oracle_report.get(
                "query_count_per_disjoint_search_and_verification_probe"
            ),
            "organizer_training_layout_indices": sorted(training_layouts),
            "organizer_development_layout_indices": sorted(development_layouts),
            "organizer_split_sha256": split["split_sha256"],
            "organizer_split_lock_present_in_oracle_manifest": (
                panel_manifest.get("organizer_split_frozen_before_new_outcomes") is not None
            ),
            "scope_note": (
                "Oracle labels are used only when their complete layout belongs to the frozen organizer training split. "
                "The four development layouts remain outside fitting."
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
            "learning_labels_may_be_slow": True,
        },
        "dense_cost_comparator": {
            "definition": "policy=None native Dense prepare_case plus decode synchronized timings from the same search probes",
            "per_row_complete_ms": dense_baselines,
        },
        "deployment_reasons": deployment_reasons,
        "optimizer": {
            "updates": FIT_UPDATES if learning_eligible else 0,
            "class_balance": "equal mean loss for observed positive and negative decisions when both classes occur",
            "status": "skipped" if not learning_eligible else "eligible_for_bounded_100_update_review",
        },
        "reasons": data_reasons,
        "controls": {
            "fixed_cover": "not_evaluated_learning_ineligible" if not learning_eligible else "evaluate_on_disjoint_queries",
            "collapsed_pair_control": "not_evaluated_learning_ineligible" if not learning_eligible else "evaluate_on_disjoint_queries",
        },
        "interpretation": (
            "No coherent, provenance-valid partial label is available for the bounded learning diagnostic."
            if not learning_eligible
            else "Coherent teacher-preservation labels are learning-eligible even when their current executor is slow; "
            "physical-reference sufficiency is unknown and deployment eligibility is false."
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


def _typed_organizer_state(record: Any) -> dict[str, torch.Tensor]:
    """Return pre-interaction input encodings for formal typed plan formation.

    The typed organizer must not depend on the all-access dense preparation,
    because that would require running the very pairwise interaction path that
    its permissions are intended to select.  Module, environment, and global
    tokens here are produced by ``encode_case`` from the current case inputs.
    """

    encoded = record.encoded
    return {
        "module_states": encoded.module_tokens.detach(),
        "environment_states": encoded.env_tokens.detach(),
        "global_state": encoded.global_token.detach(),
    }


def _typed_organizer_input_provenance() -> dict[str, Any]:
    return {
        "environment_source_measure_feature_schema": TYPED_ORGANIZER_QUADRATURE_MEASURE_SCHEMA,
        "quadrature_atom_equivalence": (
            "formal typed scoring aggregates positive quadrature weights for exact coordinate+adapter-feature "
            "duplicate atoms and assigns their total measure to every duplicate slot; normalization is over unique "
            "supports, so splitting one atom preserves all support features"
        ),
        "native_quadrature_execution": (
            "the native executor retains the original environment slots and applies each original slot weight"
        ),
        "quadrature_support_invariance_scope": (
            "the support-feature guarantee is tested with a fixed physical receiver tree; if source duplication also "
            "changes the adapter's receiver-anchor catalog or tree topology, invariance of newly learned split gates "
            "is not established by this change"
        ),
        "plan_formation_state_inputs": {
            "module_state": "encoded.module_tokens",
            "environment_state": "encoded.env_tokens",
            "global_state": "encoded.global_token",
        },
        "dense_prepared_module_states_used_for_plan_formation": False,
        "input_encoding_scope": {
            "module_tokens": "module feature and position encoders applied to current module inputs",
            "environment_tokens": "environment position/features encoder on current input support",
            "global_token": "adapter-provided input global_context encoded by global_encoder",
            "receiver_tree": "current case geometry and declared receiver anchors",
        },
        "target_or_teacher_outputs_used_for_plan_formation": False,
        "native_dense_preparation_role": (
            "retained for teacher targets, parity, and reference evaluation; not an organizer input"
        ),
        "deployment_timing_boundary": (
            "saved-plan native replay timings exclude input encoding, receiver-tree construction, organizer scoring, "
            "and hard-plan materialization unless a separate formation-plus-call timing is recorded"
        ),
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


def _masked_hard_label_metrics(
    plan: AdaptiveCoverPlan,
    labels: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Score deterministic hard decisions only where a coherent label was observed."""

    predictions = {
        "split": plan.split_gates >= 0.5,
        "module": plan.module_membership >= 0.5,
        "environment": plan.environment_membership >= 0.5,
    }
    targets = {
        "split": labels["split_targets"] >= 0.5,
        "module": labels["module_targets"] >= 0.5,
        "environment": labels["environment_targets"] >= 0.5,
    }
    observed = {
        "split": labels["split_observed"],
        "module": labels["module_observed"],
        "environment": labels["environment_observed"],
    }
    result: dict[str, Any] = {}
    for name in ("split", "module", "environment"):
        mask = observed[name].to(dtype=torch.bool)
        predicted = predictions[name][mask]
        target = targets[name][mask]
        positive = target
        negative = ~target
        result[name] = {
            "observed_count": int(mask.sum().detach().cpu()),
            "observed_positive_count": int(positive.sum().detach().cpu()),
            "observed_negative_count": int(negative.sum().detach().cpu()),
            "predicted_positive_count": int(predicted.sum().detach().cpu()),
            "masked_accuracy": (
                float((predicted == target).float().mean().detach().cpu()) if predicted.numel() else None
            ),
            "positive_accuracy": (
                float((predicted[positive] == target[positive]).float().mean().detach().cpu())
                if bool(positive.any()) else None
            ),
            "negative_accuracy": (
                float((predicted[negative] == target[negative]).float().mean().detach().cpu())
                if bool(negative.any()) else None
            ),
        }
    return result


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _atomic_json_write(payload: dict[str, Any], destination: Path) -> None:
    """Write review evidence atomically so a timeout cannot leave half JSON."""

    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _typed_fit_artifacts(
    search_dir: str | Path,
    *,
    checkpoint_sha256: str,
    layouts: tuple[Any, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    """Load only complete, directly verified G2 plans from the frozen train split."""

    directory = Path(search_dir).expanduser().resolve()
    report_path = directory / "typed_search_report.json"
    manifest_path = directory / "panel_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("typed fit requires a completed G2 report and frozen panel manifest")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError("typed fit cannot consume an incomplete receiver-local search")
    if report.get("workflow") != "windfarm_receiver_local_incremental_typed_search":
        raise ValueError("typed fit requires the formal typed G2 search provenance")
    if report.get("checkpoint_sha256") != checkpoint_sha256 or manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("typed G2 labels and field checkpoint bytes differ")
    frozen = base.freeze_organizer_layout_split(layouts)
    if report.get("frozen_split", {}).get("split_sha256") != frozen["split_sha256"]:
        raise ValueError("typed G2 labels do not match the geometry-frozen organizer split")
    expected_rows = list(map(int, frozen["training_rows_direction_order"]))
    reported_rows = list(map(int, report.get("training_rows", [])))
    if reported_rows != expected_rows:
        raise ValueError("typed G2 rows do not exactly match the frozen eight-layout train order")
    if report.get("development_rows_used_for_search_or_fit") != []:
        raise ValueError("typed development layouts must remain outside search and fitting")
    if set(map(int, report.get("development_layout_indices", []))) != set(
        map(int, frozen["development_layout_indices"])
    ):
        raise ValueError("typed G2 development-layout provenance differs from the frozen split")
    anchor_variant = str(manifest.get("anchor_measure_variant", ""))
    if anchor_variant not in {"raw", "role_balanced"}:
        raise ValueError("typed G2 panel manifest lacks a recognized anchor-measure variant")
    if report.get("anchor_measure_variant", anchor_variant) != anchor_variant:
        raise ValueError("typed G2 report and panel manifest use different anchor-measure variants")
    documents: dict[int, dict[str, Any]] = {}
    for row in expected_rows:
        path = directory / f"selected_typed_plan_row_{row:04d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing complete typed supervision artifact for train row {row}")
        document = json.loads(path.read_text(encoding="utf-8"))
        verification = document.get("search_selection", {}).get("selected_disjoint_teacher_verification", {})
        if not bool(verification.get("passed")):
            raise ValueError(f"typed training row {row} lacks a passing disjoint teacher gate")
        plan_payload = document.get("plan")
        supervision = document.get("supervision")
        if not isinstance(plan_payload, dict) or plan_payload.get("schema") != "honf-mechanism-plan-v1":
            raise ValueError(f"typed training row {row} lacks a complete typed plan serialization")
        if not isinstance(supervision, dict) or supervision.get("schema") != "honf_typed_receiver_local_supervision_v1":
            raise ValueError(f"typed training row {row} lacks recursive typed supervision")
        if int(document.get("row_index", -1)) != row:
            raise ValueError(f"typed training row artifact identity mismatch for row {row}")
        plan = MechanismPlan.from_dict(plan_payload)
        document_plan_hash = str(document.get("plan_hash", ""))
        if not document_plan_hash or plan.canonical_hash() != document_plan_hash:
            raise ValueError(f"typed training row {row} full plan hash does not match its serialization")
        _validate_typed_plan_document_supervision(plan, supervision, row)
        documents[row] = document
    _validate_typed_report_plan_bindings(report, directory, expected_rows, documents)
    _validate_typed_merged_document_bytes(report, directory, expected_rows)
    _validate_corrected_merged_typed_search_provenance(report, documents)
    return report, manifest, documents


def _validate_typed_plan_document_supervision(
    plan: MechanismPlan,
    supervision: dict[str, Any],
    row: int,
) -> None:
    """Require saved recursive labels to be exactly derived from the bound plan."""

    expected = base._typed_plan_supervision(plan)
    if supervision != expected:
        raise ValueError(
            f"typed training row {row} supervision differs from deterministic labels derived from its plan"
        )


def _validate_typed_report_plan_bindings(
    report: dict[str, Any],
    directory: Path,
    expected_rows: list[int],
    documents: Mapping[int, dict[str, Any]],
) -> None:
    """Bind each loaded train label to the merged report's exact path and hash."""

    report_rows = report.get("rows")
    if not isinstance(report_rows, list) or [int(item.get("row_index", -1)) for item in report_rows] != expected_rows:
        raise ValueError("typed G2 report rows do not exactly match the frozen train direction order")
    report_rows_by_id = {int(item["row_index"]): item for item in report_rows}
    for row in expected_rows:
        document = documents[row]
        row_summary = report_rows_by_id[row]
        expected_path = (directory / f"selected_typed_plan_row_{row:04d}.json").resolve()
        reported_path = Path(str(row_summary.get("selected_plan_path", ""))).expanduser().resolve()
        if reported_path != expected_path:
            raise ValueError(f"typed G2 report selected plan path differs for train row {row}")
        if row_summary.get("selected_plan_hash") != document.get("plan_hash"):
            raise ValueError(f"typed G2 report selected plan hash differs for train row {row}")


def _validate_typed_merged_document_bytes(
    report: dict[str, Any], directory: Path, expected_rows: list[int]
) -> None:
    """Require merged selected documents to be byte-identical to their source shard."""

    source_directories = report.get("merged_stage_directories")
    if not isinstance(source_directories, list) or not source_directories:
        raise ValueError("typed G2 merged report lacks source stage directories for document integrity")
    reports_by_directory: dict[Path, dict[str, Any]] = {}
    for value in source_directories:
        source_directory = Path(str(value)).expanduser().resolve()
        source_report_path = source_directory / "typed_search_report.json"
        if not source_report_path.is_file():
            raise FileNotFoundError(f"typed G2 source stage report is missing: {source_report_path}")
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        if source_report.get("status") != "stage_complete":
            raise ValueError(f"typed G2 source stage is not complete: {source_directory}")
        reports_by_directory[source_directory] = source_report
    for row in expected_rows:
        matching = [
            source_directory
            for source_directory, source_report in reports_by_directory.items()
            if row in set(map(int, source_report.get("training_rows", [])))
        ]
        if len(matching) != 1:
            raise ValueError(f"typed G2 train row {row} does not bind to exactly one source shard")
        merged_path = directory / f"selected_typed_plan_row_{row:04d}.json"
        source_path = matching[0] / f"selected_typed_plan_row_{row:04d}.json"
        if not merged_path.is_file() or not source_path.is_file():
            raise FileNotFoundError(f"typed G2 train row {row} merged/source plan document is missing")
        if base._checkpoint_sha256(merged_path) != base._checkpoint_sha256(source_path):
            raise ValueError(f"typed G2 train row {row} merged plan bytes differ from its source shard")


def _validate_corrected_merged_typed_search_provenance(
    report: dict[str, Any], documents: dict[int, dict[str, Any]]
) -> None:
    """Reject G5 labels from unmerged or pre-correction G2 evidence."""

    merged_stages = report.get("merged_stage_directories")
    if not isinstance(merged_stages, list) or len(merged_stages) < 2 or not all(
        isinstance(value, str) and value for value in merged_stages
    ):
        raise ValueError("typed fit requires labels merged from corrected G2 stages")
    for row, document in documents.items():
        selected_work = document.get("search_selection", {}).get("selected_search_work", {})
        pair_semantics = str(selected_work.get("pair_work_semantics", ""))
        score_semantics = str(selected_work.get("logical_work_score_semantics", ""))
        if (
            not pair_semantics.startswith("exact valid pair rows from native core compilers:")
            or "MM excludes self and padded modules" not in pair_semantics
            or "ME uses present module receivers and positive environment weights" not in pair_semantics
            or "EM uses all environment receiver slots and present module sources" not in pair_semantics
            or "QM/QE use Q query receivers and valid sources" not in pair_semantics
            or not score_semantics.startswith(
                "heuristic score equal to exact valid native pair rows plus 0.05 times the count of "
            )
            or "not executed work or latency" not in score_semantics
        ):
            raise ValueError(
                f"typed training row {row} lacks corrected exact-native-work G2 provenance"
            )


def _typed_label_tensors(document: dict[str, Any], device: torch.device) -> dict[str, Any]:
    supervision = document["supervision"]
    return {
        "split_targets": torch.as_tensor(supervision["split_targets"], device=device, dtype=torch.float32),
        "split_observed": torch.as_tensor(supervision["split_observed"], device=device, dtype=torch.bool),
        "mechanism_targets": {
            mechanism: torch.as_tensor(
                supervision["mechanism_targets"][mechanism], device=device, dtype=torch.float32
            )
            for mechanism in INTERACTION_MECHANISMS
        },
        "mechanism_observed": {
            mechanism: torch.as_tensor(
                supervision["mechanism_observed"][mechanism], device=device, dtype=torch.bool
            )
            for mechanism in INTERACTION_MECHANISMS
        },
    }


def _typed_plan_from_document(record: Any, document: dict[str, Any], device: torch.device) -> MechanismPlan:
    saved = MechanismPlan.from_dict(document["plan"], device=device)
    current = record.trees[0]
    saved_universe = saved.tree.universe
    current_universe = current.universe
    for name in ("coordinates", "weights", "roles", "coordinate_scale"):
        if not torch.equal(getattr(saved_universe, name), getattr(current_universe, name)):
            raise ValueError(f"typed G2 plan tree {name} differs from the recreated training input")
    if saved.tree.nodes != current.nodes:
        raise ValueError("typed G2 plan receiver tree differs from the recreated training input")
    if saved.canonical_hash() != document.get("plan_hash"):
        raise ValueError("typed G2 plan serialization hash does not match its selected artifact")
    return replace(saved, tree=current)


def _typed_scores(organizer: InputOnlyCoverOrganizer, record: Any) -> Any:
    return organizer.score_cases(record.encoded, _typed_organizer_state(record), record.trees)[0]


def _typed_supervised_loss(
    organizer: InputOnlyCoverOrganizer,
    record: Any,
    labels: dict[str, Any],
) -> torch.Tensor:
    scores = _typed_scores(organizer, record)
    return _typed_supervised_loss_from_scores(scores, labels)


def _typed_supervised_loss_from_scores(scores: Any, labels: dict[str, Any]) -> torch.Tensor:
    terms = [masked_binary_logit_loss(
        scores.split_logits,
        labels["split_targets"],
        labels["split_observed"],
        balance_classes=True,
    )]
    if set(scores.mechanism_logits) != set(INTERACTION_MECHANISMS):
        raise ValueError("typed scorer must return independent logits for all five mechanisms")
    for mechanism in INTERACTION_MECHANISMS:
        terms.append(masked_binary_logit_loss(
            scores.mechanism_logits[mechanism],
            labels["mechanism_targets"][mechanism],
            labels["mechanism_observed"][mechanism],
            balance_classes=True,
        ))
    return torch.stack(terms).mean()


def _typed_hard_plan(organizer: InputOnlyCoverOrganizer, record: Any) -> MechanismPlan:
    scores = _typed_scores(organizer, record)
    return organizer.plans_from_scores(
        (scores,), record.encoded, record.trees, hard=True
    )[0]


def _initialize_typed_organizer_all_access(organizer: InputOnlyCoverOrganizer) -> dict[str, Any]:
    """Set hard inference to no split and full typed permissions before training."""

    with torch.no_grad():
        organizer.split_head.weight.zero_()
        organizer.split_head.bias.fill_(TYPED_ALL_ACCESS_SPLIT_LOGIT)
        for mechanism in INTERACTION_MECHANISMS:
            final = organizer.pair_scorers[mechanism][-1]
            final.weight.zero_()
            final.bias.fill_(TYPED_ALL_ACCESS_PERMISSION_LOGIT)
    probability = float(torch.sigmoid(torch.tensor(TYPED_ALL_ACCESS_PERMISSION_LOGIT)))
    return {
        "method": "zero final head weights; split bias=-2 and each typed permission head bias=+2",
        "split_initial_logit": TYPED_ALL_ACCESS_SPLIT_LOGIT,
        "permission_initial_logit": TYPED_ALL_ACCESS_PERMISSION_LOGIT,
        "permission_sigmoid_derivative_at_initial_logit": probability * (1.0 - probability),
        "hard_threshold": 0.5,
        "split_hard_state": "all closed",
        "permission_hard_state": "all active module/environment slots open independently for MM/ME/EM/QM/QE",
        "straight_through_estimator": "hard + sigmoid(logits) - sigmoid(logits).detach()",
    }


def _typed_direct_forward(
    model: Any,
    batch: Any,
    plan: MechanismPlan,
    *,
    device: torch.device,
    return_aux: bool = True,
) -> tuple[Any, Any, InteractionContext, dict[str, Any]]:
    prepared, encoded, context = base._prepare_typed_plan(model, batch, plan, device=device)
    query, features = base._batch_query_tensors(batch)
    output = model.core.decode_queries(
        prepared,
        query,
        query_features=features,
        return_interaction_aux=return_aux,
        interaction_context=context,
    )
    return prepared, encoded, context, output


def _typed_role_mean_square(
    prediction: torch.Tensor,
    target: torch.Tensor,
    roles: dict[str, Any],
) -> torch.Tensor:
    errors = []
    for mask_value in roles.values():
        mask = torch.as_tensor(mask_value, device=prediction.device, dtype=torch.bool)
        if bool(mask.any()):
            errors.append(torch.mean((prediction[0, mask] - target[0, mask]) ** 2))
    if not errors:
        raise ValueError("predictive feedback requires at least one protected receiver role")
    return torch.stack(errors).mean()


def _typed_valid_source_mask(encoded: Any, mechanism: str) -> torch.Tensor:
    if mechanism in {"MM", "EM", "QM"}:
        if encoded.module_present.shape[0] != 1:
            raise ValueError("typed native work requires a single encoded case")
        return encoded.module_present[0] > 0.5
    if mechanism in {"ME", "QE"}:
        if encoded.env_coords.shape[0] != 1:
            raise ValueError("typed native work requires a single encoded case")
        if encoded.env_weights is None:
            return torch.ones(
                encoded.env_coords.shape[1], device=encoded.env_coords.device, dtype=torch.bool
            )
        return encoded.env_weights[0] > 0.0
    raise ValueError(f"unknown typed interaction mechanism {mechanism!r}")


def _typed_native_access_matrix(
    plan: MechanismPlan,
    encoded: Any,
    query_receivers: torch.Tensor,
    mechanism: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    receiver_axes = base._typed_mechanism_receivers(encoded, query_receivers)
    source_valid = _typed_valid_source_mask(encoded, mechanism)
    access = plan.access_for(
        mechanism,
        receiver_axes[mechanism],
        source_count=int(source_valid.numel()),
        module_present=encoded.module_present[0]
        if mechanism in {"MM", "EM", "QM"}
        else None,
    )
    if access.shape[1] != int(source_valid.numel()):
        raise ValueError(f"{mechanism} access does not match its native source axis")
    source_indices = torch.nonzero(source_valid, as_tuple=False).flatten()
    access = access[:, source_valid]
    if mechanism == "MM":
        module_valid_count = int(source_valid.sum().detach().cpu())
        if access.shape != (module_valid_count, module_valid_count):
            raise ValueError("MM access does not match its valid native module axes")
        access = access * (~torch.eye(
            module_valid_count, device=access.device, dtype=torch.bool
        )).to(access.dtype)
    return access, source_indices


def _typed_native_source_indices(
    plan: MechanismPlan,
    encoded: Any,
    query_receivers: torch.Tensor,
    mechanism: str,
) -> torch.Tensor:
    access, source_indices = _typed_native_access_matrix(
        plan, encoded, query_receivers, mechanism
    )
    return source_indices[(access > 0.5).any(dim=0)]


def _typed_differentiable_work(
    plan: MechanismPlan,
    encoded: Any,
    query_receivers: torch.Tensor,
) -> torch.Tensor:
    """Normalize soft permission mass on each mechanism's exact native pair axis."""

    numerator = query_receivers.new_zeros(())
    denominator = 0
    module_present = encoded.module_present[0] > 0.5
    for mechanism in INTERACTION_MECHANISMS:
        access, _source_indices = _typed_native_access_matrix(
            plan, encoded, query_receivers, mechanism
        )
        numerator = numerator + access.sum()
        valid_pair_count = int(access.numel())
        if mechanism == "MM":
            valid_pair_count -= int(module_present.sum().detach().cpu())
        denominator += valid_pair_count
    return numerator / max(denominator, 1)


def _typed_hard_label_metrics(plan: MechanismPlan, labels: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "split": _masked_binary_metrics(
            plan.split_gates, labels["split_targets"], labels["split_observed"]
        )
    }
    for mechanism in INTERACTION_MECHANISMS:
        result[mechanism] = _masked_binary_metrics(
            plan.permission_matrix(mechanism),
            labels["mechanism_targets"][mechanism],
            labels["mechanism_observed"][mechanism],
        )
    return result


@dataclass(frozen=True)
class _TypedDirectPairPlan(MechanismPlan):
    """Evaluation-only exact pair mask using the ordinary native mechanisms.

    The core's grouped rectangular-subset executor derives access from tree
    membership rows, so this plan must run through the dense-masked reference
    executor. Its ``access_for`` returns the direct receiver/source matrix to
    the same MM/ME/EM/QM/QE native compilers and message heads.
    """

    direct_pair_access: Mapping[str, torch.Tensor] = field(default_factory=dict)
    direct_receiver_coordinates: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        MechanismPlan.__post_init__(self)
        access: dict[str, torch.Tensor] = {}
        coordinates: dict[str, torch.Tensor] = {}
        if set(self.direct_pair_access) != set(INTERACTION_MECHANISMS):
            raise ValueError("direct-pair control requires a matrix for all five typed mechanisms")
        if set(self.direct_receiver_coordinates) != set(INTERACTION_MECHANISMS):
            raise ValueError("direct-pair control requires receiver coordinates for all five mechanisms")
        for mechanism in INTERACTION_MECHANISMS:
            matrix = self.direct_pair_access[mechanism]
            receivers = self.direct_receiver_coordinates[mechanism]
            if matrix.ndim != 2 or matrix.shape[1] != self._source_count(mechanism):
                raise ValueError(f"direct {mechanism} mask must have [native receivers, native sources] axes")
            if receivers.ndim != 2 or receivers.shape[0] != matrix.shape[0]:
                raise ValueError(f"direct {mechanism} receiver coordinates do not match its pair mask")
            if matrix.device != self.split_gates.device or receivers.device != self.split_gates.device:
                raise ValueError("direct pair masks, receiver coordinates, and cover plan must share a device")
            if not bool(torch.isfinite(matrix).all()) or bool(((matrix < 0) | (matrix > 1)).any()):
                raise ValueError("direct pair masks must be finite and in [0,1]")
            access[mechanism] = matrix.detach().clone()
            coordinates[mechanism] = receivers.detach().clone()
        object.__setattr__(self, "direct_pair_access", MappingProxyType(access))
        object.__setattr__(self, "direct_receiver_coordinates", MappingProxyType(coordinates))

    def access_for(
        self,
        mechanism: str,
        receivers: torch.Tensor,
        source_count: int | None = None,
        *,
        phase: str | None = None,
        module_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mechanism not in self.direct_pair_access:
            raise ValueError(f"direct pair mask has no mechanism {mechanism!r}")
        if phase is not None:
            raise ValueError("the Q1024 direct-pair control is phase-independent")
        matrix = self.direct_pair_access[mechanism]
        expected = self.direct_receiver_coordinates[mechanism].to(
            device=receivers.device, dtype=receivers.dtype
        )
        if receivers.ndim != 2 or expected.ndim != 2 or receivers.shape[1:] != expected.shape[1:]:
            raise ValueError(f"direct {mechanism} receiver coordinates have incompatible dimensions")
        exact_matches = (receivers[:, None, :] == expected[None, :, :]).all(dim=-1)
        match_counts = exact_matches.sum(dim=1)
        if bool((match_counts != 1).any()):
            raise ValueError(
                f"direct {mechanism} receiver coordinates are absent or ambiguous after control construction"
            )
        receiver_ids = exact_matches.to(torch.int64).argmax(dim=1)
        matrix = matrix.index_select(0, receiver_ids)
        if source_count is not None and int(source_count) != self._source_count(mechanism):
            raise ValueError(f"direct {mechanism} source catalogue changed after control construction")
        matrix = matrix.to(device=receivers.device, dtype=receivers.dtype)
        if mechanism in {"MM", "EM", "QM"} and module_present is not None:
            if module_present.shape != (matrix.shape[1],):
                raise ValueError(f"direct {mechanism} module validity does not match its source catalogue")
            matrix = matrix * (module_present > 0.5).to(matrix.dtype)[None, :]
        return matrix

    def is_full_access(self, **_kwargs: Any) -> bool:
        # A sparse direct-pair matrix is never eligible for Dense bypass, even
        # when its source union happens to contain every source at least once.
        return False

    def canonical_hash(self) -> str:
        digest = hashlib.sha256(super().canonical_hash().encode("ascii"))
        for mechanism in INTERACTION_MECHANISMS:
            for name, tensor in (
                (f"{mechanism}:pairs", self.direct_pair_access[mechanism]),
                (f"{mechanism}:receivers", self.direct_receiver_coordinates[mechanism]),
            ):
                value = tensor.detach().contiguous().cpu()
                digest.update(name.encode("ascii"))
                digest.update(str(tuple(value.shape)).encode("ascii"))
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()


TYPED_G6_VARIANTS = (
    "all_access",
    "verified_oracle_hard",
    "learned_input_only",
    "population_fixed_support",
    "ungrouped_direct_pair_matched_budget",
    "collapsed_root_source_union",
)
TYPED_ORGANIZER_INPUT_STATE_SCHEMA = "encoded-input-token-state-v1"


def _typed_root_permissions_plan(
    record: Any,
    support_by_mechanism: Mapping[str, torch.Tensor],
) -> MechanismPlan:
    """Make a no-split native plan from five source supports."""

    tree = record.trees[0]
    encoded = record.encoded
    node_count = len(tree.nodes)
    gates = encoded.module_present.new_zeros((node_count,))
    permissions: dict[str, torch.Tensor] = {}
    for mechanism in INTERACTION_MECHANISMS:
        source_count = (
            int(encoded.module_present.shape[1])
            if mechanism in {"MM", "EM", "QM"}
            else int(encoded.env_coords.shape[1])
        )
        support = torch.as_tensor(
            support_by_mechanism[mechanism],
            device=encoded.module_present.device,
            dtype=encoded.module_present.dtype,
        ).reshape(-1)
        if support.shape != (source_count,):
            raise ValueError(f"{mechanism} fixed source support must have {source_count} entries")
        if mechanism in {"MM", "EM", "QM"}:
            support = support * (encoded.module_present[0] > 0.5).to(support.dtype)
        permission = gates.new_zeros((node_count, source_count))
        permission[0] = support
        permissions[mechanism] = permission
    return MechanismPlan(
        tree,
        gates,
        encoded.module_present[0],
        int(encoded.env_coords.shape[1]),
        permissions,
    )


def _typed_population_fixed_support(
    training_records: list[Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Derive fixed source supports from the frozen train rows' G2 plans only."""

    if not training_records:
        raise ValueError("population-fixed typed support requires train rows")
    counts: dict[str, list[int]] = {mechanism: [] for mechanism in INTERACTION_MECHANISMS}
    frequencies: dict[str, np.ndarray] = {}
    denominators: dict[str, np.ndarray] = {}
    for mechanism in INTERACTION_MECHANISMS:
        source_count = max(
            int(_typed_valid_source_mask(record.encoded, mechanism).numel())
            for record in training_records
        )
        frequencies[mechanism] = np.zeros(source_count, dtype=np.float64)
        denominators[mechanism] = np.zeros(source_count, dtype=np.float64)
    for record in training_records:
        plan = getattr(record, "oracle_plan", None)
        if not isinstance(plan, MechanismPlan):
            raise TypeError("population-fixed support may use only verified typed train plans")
        queries = record.search_batch.query_xy[0]
        for mechanism in INTERACTION_MECHANISMS:
            valid = _typed_valid_source_mask(record.encoded, mechanism).detach().cpu().numpy()
            selected = np.zeros(len(valid), dtype=bool)
            source_ids = _typed_native_source_indices(plan, record.encoded, queries, mechanism)
            selected[source_ids.detach().cpu().numpy()] = True
            frequencies[mechanism][: len(selected)] += selected
            denominators[mechanism][: len(valid)] += valid
            counts[mechanism].append(int(selected.sum()))
    masks: dict[str, torch.Tensor] = {}
    report: dict[str, Any] = {
        "source": "frequency-ranked exact native G2 source support from the frozen 24 training rows only",
        "support_indexing": "training-maximum module/environment slot indices, cropped to each case's native source axis",
        "permutation_invariant_physical_support": False,
        "limitation": (
            "This is an artificial slot-indexed control. Equal slot indices across layouts are not asserted to identify "
            "the same physical source, and reordering module or environment source arrays changes which physical "
            "sources the unchanged support indices select."
        ),
        "training_row_indices": [int(record.case.index) for record in training_records],
        "development_rows_used": [],
        "source_support_by_mechanism": {},
        "frequency_by_source": {},
        "target_source_count_by_mechanism": {},
        "selection_rule": "select the rounded mean per-case source-union count; rank by train-row frequency, then stable source index",
    }
    for mechanism in INTERACTION_MECHANISMS:
        valid_frequency = frequencies[mechanism] / np.maximum(denominators[mechanism], 1.0)
        available = np.flatnonzero(denominators[mechanism] > 0)
        target_count = min(
            len(available),
            round(float(np.mean(counts[mechanism]))) if counts[mechanism] else 0,
        )
        ranked = sorted(available.tolist(), key=lambda source: (-valid_frequency[source], source))
        selected = np.zeros(len(valid_frequency), dtype=bool)
        selected[ranked[:target_count]] = True
        masks[mechanism] = torch.as_tensor(
            selected, device=training_records[0].encoded.module_present.device,
            dtype=training_records[0].encoded.module_present.dtype,
        )
        report["source_support_by_mechanism"][mechanism] = np.flatnonzero(selected).tolist()
        report["frequency_by_source"][mechanism] = valid_frequency.tolist()
        report["target_source_count_by_mechanism"][mechanism] = target_count
    return masks, report


def _typed_collapsed_root_plan(record: Any, oracle_plan: MechanismPlan) -> MechanismPlan:
    """Collapse a train-only verified plan to its observed native source union."""

    queries = record.search_batch.query_xy[0]
    supports = {
        mechanism: torch.zeros(
            oracle_plan._source_count(mechanism),
            device=oracle_plan.split_gates.device,
            dtype=oracle_plan.split_gates.dtype,
        )
        for mechanism in INTERACTION_MECHANISMS
    }
    for mechanism in INTERACTION_MECHANISMS:
        source_ids = _typed_native_source_indices(oracle_plan, record.encoded, queries, mechanism)
        supports[mechanism][source_ids] = 1.0
    return _typed_root_permissions_plan(record, supports)


def _typed_direct_pair_plan(
    record: Any,
    learned_plan: MechanismPlan,
    query_receivers: torch.Tensor,
) -> tuple[_TypedDirectPairPlan, dict[str, Any]]:
    """Rank individual valid native pairs by normalized geometric distance.

    The total selected count equals the learned plan's exact logical pair
    count across MM/ME/EM/QM/QE. Geometry and stable indices break ties; no
    teacher output, reference target, G2 plan, or dev label enters ranking.
    """

    encoded = record.encoded
    module_coords = encoded.module_centers[0]
    module_valid = encoded.module_present[0] > 0.5
    environment_coords = encoded.env_coords[0]
    environment_valid = encoded.env_weights[0] > 0.0
    receiver_coords = {
        "MM": module_coords,
        "ME": module_coords,
        "EM": environment_coords,
        "QM": query_receivers,
        "QE": query_receivers,
    }
    source_coords = {
        "MM": module_coords,
        "ME": environment_coords,
        "EM": module_coords,
        "QM": module_coords,
        "QE": environment_coords,
    }
    pair_validity = {
        "MM": module_valid[:, None] & module_valid[None, :]
        & ~torch.eye(len(module_valid), device=module_valid.device, dtype=torch.bool),
        "ME": module_valid[:, None] & environment_valid[None, :],
        "EM": torch.ones((len(environment_coords),), device=module_coords.device, dtype=torch.bool)[:, None]
        & module_valid[None, :],
        "QM": torch.ones((len(query_receivers),), device=module_coords.device, dtype=torch.bool)[:, None]
        & module_valid[None, :],
        "QE": torch.ones((len(query_receivers),), device=module_coords.device, dtype=torch.bool)[:, None]
        & environment_valid[None, :],
    }
    learned_summary = base._typed_work_summary(
        learned_plan, encoded, query_receivers, include_root_child_support=False
    )
    budget = int(learned_summary["total_unique_source_receiver_pairs_across_mechanisms"])
    scale = encoded.coordinate_scale
    if scale.ndim == 3:
        scale = scale[0, 0]
    elif scale.ndim == 2:
        scale = scale[0]
    scale_np = scale.detach().cpu().double().numpy()
    ranked_mechanism: list[np.ndarray] = []
    ranked_receiver: list[np.ndarray] = []
    ranked_source: list[np.ndarray] = []
    ranked_distance: list[np.ndarray] = []
    for mechanism_id, mechanism in enumerate(INTERACTION_MECHANISMS):
        receivers = receiver_coords[mechanism].detach().cpu().double().numpy()
        sources = source_coords[mechanism].detach().cpu().double().numpy()
        valid = pair_validity[mechanism].detach().cpu().numpy()
        receiver_ids, source_ids = np.nonzero(valid)
        distance = (((receivers[receiver_ids] - sources[source_ids]) / scale_np) ** 2).sum(axis=1)
        ranked_mechanism.append(np.full(len(receiver_ids), mechanism_id, dtype=np.int8))
        ranked_receiver.append(receiver_ids.astype(np.int64, copy=False))
        ranked_source.append(source_ids.astype(np.int64, copy=False))
        ranked_distance.append(distance)
    all_mechanisms = np.concatenate(ranked_mechanism)
    all_receivers = np.concatenate(ranked_receiver)
    all_sources = np.concatenate(ranked_source)
    all_distances = np.concatenate(ranked_distance)
    candidate_count = len(all_distances)
    if budget < 0 or budget > candidate_count:
        raise ValueError("learned logical pair budget exceeds the valid direct-pair catalogue")
    order = np.lexsort((all_sources, all_receivers, all_mechanisms, all_distances))
    selected = order[:budget]
    matrices: dict[str, torch.Tensor] = {}
    permission_supports: dict[str, torch.Tensor] = {}
    coordinate_rows: dict[str, torch.Tensor] = {}
    for mechanism_id, mechanism in enumerate(INTERACTION_MECHANISMS):
        shape = (len(receiver_coords[mechanism]), len(source_coords[mechanism]))
        matrix = encoded.module_present.new_zeros(shape)
        chosen = selected[all_mechanisms[selected] == mechanism_id]
        if len(chosen):
            receiver_index = torch.as_tensor(
                all_receivers[chosen], device=matrix.device, dtype=torch.long
            )
            source_index = torch.as_tensor(
                all_sources[chosen], device=matrix.device, dtype=torch.long
            )
            matrix[receiver_index, source_index] = 1.0
        matrices[mechanism] = matrix
        permission_supports[mechanism] = (matrix > 0).any(dim=0).to(matrix.dtype)
        coordinate_rows[mechanism] = receiver_coords[mechanism]
    root_plan = _typed_root_permissions_plan(record, permission_supports)
    direct_plan = _TypedDirectPairPlan(
        root_plan.tree,
        root_plan.split_gates,
        root_plan.module_present,
        root_plan.environment_count,
        root_plan.permissions,
        matrices,
        coordinate_rows,
    )
    exact_summary = base._typed_work_summary(
        direct_plan, encoded, query_receivers, include_root_child_support=False
    )
    exact_count = int(exact_summary["total_unique_source_receiver_pairs_across_mechanisms"])
    if exact_count != budget:
        raise RuntimeError(f"direct native pair compilers produced {exact_count} rows for budget {budget}")
    exact_summary["control_type"] = "ungrouped_direct_receiver_source_pair_mask"
    exact_summary["tree_packet_counters_applicable"] = False
    exact_summary["logical_work_score"] = exact_count
    exact_summary["logical_work_score_semantics"] = (
        "exact valid direct native pair rows only; tree packet structure is not applicable to this control"
    )
    report = {
        "ranking_rule": "ascending squared Euclidean distance after coordinate_scale normalization; tie-break mechanism order MM,ME,EM,QM,QE, then receiver and source indices",
        "ranking_inputs": "module/environment/query coordinates and validity masks only",
        "teacher_oracle_or_reference_values_used_for_ranking": False,
        "learned_exact_pair_budget": budget,
        "valid_direct_pair_catalogue_size": candidate_count,
        "selected_pair_count_by_mechanism": {
            mechanism: int(matrices[mechanism].sum().detach().cpu())
            for mechanism in INTERACTION_MECHANISMS
        },
        "selected_pair_count_exact_native_compiler": exact_count,
        "plan_hash": direct_plan.canonical_hash(),
        "work": exact_summary,
        "executor_requirement": "dense_masked reference; grouped rectangular_subset bypasses direct receiver/source masks",
        "measured_speed_or_deployment_claim": False,
    }
    return direct_plan, report


def _typed_evaluate_g6_variant(
    model: Any,
    record: Any,
    plan: MechanismPlan,
    *,
    name: str,
    teacher_checkpoint_id: str,
    normalizer: VelocityNormalizer,
    device: torch.device,
    provenance: Mapping[str, Any],
    direct_pair_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure one hard native control on this row's disjoint Q1024 probe."""

    backend = model.core.backend
    had_executor_attribute = hasattr(backend, "cover_executor")
    previous_executor = getattr(backend, "cover_executor", None)
    direct_pair = isinstance(plan, _TypedDirectPairPlan)
    if direct_pair:
        backend.set_cover_executor("dense_masked")
    try:
        with torch.inference_mode():
            prepared, prediction, output, prepare_ms, decode_ms = base._typed_replay_plan(
                model,
                record.verification_batch,
                plan,
                device=device,
                include_aux=True,
            )
    finally:
        if direct_pair:
            if had_executor_attribute:
                if previous_executor is None:
                    raise RuntimeError("native cover executor unexpectedly had a null prior value")
                backend.set_cover_executor(str(previous_executor))
            else:
                delattr(backend, "cover_executor")

    observation, role_metrics = base._probe_observation(
        prediction[0],
        record.full_verification_prediction[0],
        record.verification,
        normalizer,
        teacher_checkpoint_id=teacher_checkpoint_id,
        training_evidence_id=(
            f"run2103_typed_G6_{name}_{record.case.index}_verification_Q1024"
        ),
    )
    teacher_passed, teacher_reasons = base.teacher_preservation_gate(
        observation,
        {
            role: base.TeacherDistortionLimit(base.TYPED_PROTECTED_ROLE_TEACHER_LIMIT)
            for role in record.verification.roles
        },
    )
    query_receivers = record.verification_batch.query_xy[0]
    work = base._typed_work_summary(
        plan,
        record.encoded,
        query_receivers,
        include_root_child_support=not direct_pair,
    )
    if direct_pair:
        exact_pairs = int(work["total_unique_source_receiver_pairs_across_mechanisms"])
        work["control_type"] = "ungrouped_direct_receiver_source_pair_mask"
        work["tree_packet_counters_applicable"] = False
        work["logical_work_score"] = exact_pairs
        work["logical_work_score_semantics"] = (
            "exact valid direct native pair rows only; tree packet structure is not applicable to this control"
        )
    source_sets = {
        mechanism: _typed_native_source_indices(
            plan, record.encoded, query_receivers, mechanism
        ).detach().cpu().tolist()
        for mechanism in INTERACTION_MECHANISMS
    }
    reference_rmse = _typed_reference_rmse_by_role(observation)
    normalized_teacher = {
        role: None if error.teacher is None else float(error.teacher)
        for role, error in observation.roles.items()
    }
    preparation_ledger = prepared.backend_state.get("cover_preparation_ledger", {})
    query_ledger = output.get("_interaction_aux", {})
    prep_keys = {
        key: int(value)
        for key, value in preparation_ledger.items()
        if key.endswith(("_actual_rows", "_unique_pairs", "_padded_rows", "_rectangular_rows"))
        or key in {
            "cover_executor_dense_masked",
            "cover_executor_rectangular_subset",
            "cover_executor_packed",
            "cover_executor_full_access",
        }
    }
    query_keys = {
        key: int(value.detach().cpu()) if torch.is_tensor(value) else int(value)
        for key, value in query_ledger.items()
        if key.startswith(("cover_qm_", "cover_qe_", "cover_executor_"))
        and (key.endswith((
                 "_actual_rows", "_unique_source_receiver_pairs", "_padded_rows", "_rectangular_rows",
                 "_executor_dense_masked", "_executor_rectangular_subset", "_executor_packed", "_executor_full_access",
             ))
             or key == "cover_executor_dense_masked")
    }
    if direct_pair:
        # The core's ``padded_rows`` field on the dense-masked reference is
        # the dense rectangle minus selected pairs. It therefore includes
        # intentionally omitted pairs and is not padding overhead.
        prep_keys = {
            key.replace("_padded_rows", "_rectangle_minus_selected_mask_rows"): value
            for key, value in prep_keys.items()
        }
        query_keys = {
            key.replace("_padded_rows", "_rectangle_minus_selected_mask_rows"): value
            for key, value in query_keys.items()
        }
    result: dict[str, Any] = {
        "status": "complete",
        "variant": name,
        "plan_schema": (
            "windfarm_direct_pair_mask_v1"
            if direct_pair
            else "honf-mechanism-plan-v1"
        ),
        "plan_hash": plan.canonical_hash(),
        "source_sets_by_mechanism": source_sets,
        "logical_work": work,
        "teacher_preservation": {
            "passed": bool(teacher_passed),
            "reasons": base._teacher_gate_reason_list(teacher_reasons),
            "normalized_error_by_protected_role": normalized_teacher,
            "limit": base.TYPED_PROTECTED_ROLE_TEACHER_LIMIT,
            "frontier": base._typed_teacher_gate_frontier(
                observation, record.verification.roles
            ),
        },
        "native_reference_rmse_mps_by_protected_role": reference_rmse,
        "reference_measurement_scope": "disjoint Q1024 exact native-cell gathers; not a native-grid sufficiency decision",
        "role_metrics": role_metrics,
        "actual_synchronized_prepare_ms": float(prepare_ms),
        "actual_synchronized_decode_ms": float(decode_ms),
        "actual_synchronized_complete_ms": float(prepare_ms + decode_ms),
        "native_executor_work": {
            "preparation": prep_keys,
            "query": query_keys,
            "direct_pair_dense_masked_reference": bool(direct_pair),
        },
        "measured_speed_eligible": False,
        "deployment_eligible": False,
        "optimizer_calls_in_this_stage": 0,
        "new_physical_solves": 0,
        **dict(provenance),
    }
    if direct_pair_metadata is not None:
        result["direct_pair_control"] = direct_pair_metadata
    if name == "all_access":
        max_delta = base._max_abs(prediction, record.full_verification_prediction)
        result["dense_teacher_parity"] = {
            "max_abs_mps": max_delta,
            "exact_equal": bool(torch.equal(prediction, record.full_verification_prediction)),
            "within_frozen_tolerance": bool(torch.allclose(
                prediction,
                record.full_verification_prediction,
                atol=base.COLD_WARM_ABS_TOLERANCE_MPS,
                rtol=base.COLD_WARM_REL_TOLERANCE,
            )),
        }
    return result


def _typed_reference_rmse_by_role(observation: Any) -> dict[str, float]:
    """Retain measured Q1024 reference errors even when sufficiency is unresolved."""

    return {
        role: float(error.reference)
        for role, error in observation.roles.items()
        if error.reference is not None and math.isfinite(float(error.reference))
    }


def _typed_g6_comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate measured variant gates/work without promoting speed claims."""

    summary: dict[str, Any] = {}
    for variant in TYPED_G6_VARIANTS:
        summary[variant] = {}
        for partition in ("training", "development"):
            entries = [
                item.get("variants", {}).get(variant, {})
                for item in rows
                if item.get("partition") == partition
            ]
            measured = [item for item in entries if item.get("status") == "complete"]
            unavailable = [item for item in entries if item.get("status") == "unavailable"]
            pairs = [
                int(item["logical_work"]["total_unique_source_receiver_pairs_across_mechanisms"])
                for item in measured
                if isinstance(item.get("logical_work"), dict)
            ]
            role_values: dict[str, list[float]] = {}
            for item in measured:
                for role, value in item.get("native_reference_rmse_mps_by_protected_role", {}).items():
                    role_values.setdefault(role, []).append(float(value))
            complete_times = [float(item["actual_synchronized_complete_ms"]) for item in measured]
            summary[variant][partition] = {
                "measured_rows": len(measured),
                "unavailable_rows": len(unavailable),
                "teacher_gate_pass_count": sum(
                    bool(item.get("teacher_preservation", {}).get("passed")) for item in measured
                ),
                "teacher_gate_fail_count": sum(
                    not bool(item.get("teacher_preservation", {}).get("passed")) for item in measured
                ),
                "exact_native_pair_count_sum": sum(pairs),
                "exact_native_pair_count_min": min(pairs) if pairs else None,
                "exact_native_pair_count_max": max(pairs) if pairs else None,
                "max_q1024_reference_rmse_mps_by_role": {
                    role: max(values) for role, values in sorted(role_values.items())
                },
                "single_call_timing_median_ms_descriptive_only": (
                    float(np.median(complete_times)) if complete_times else None
                ),
            }
    return {
        "variants": summary,
        "development_control_rule": "verified oracle and collapsed-root source-union are unavailable for development rows; no per-case development G2 labels are searched or loaded",
        "fixed_support_rule": (
            "slot-indexed artificial support derived only from frozen training-row G2 source unions and then applied "
            "unchanged to train and development rows; source-slot reordering changes physical source identity"
        ),
        "direct_pair_rule": "geometry-ranked independent native pairs exactly match each row's learned hard-plan logical pair total",
        "speed_interpretation": "one synchronized call per row/variant is descriptive only; no speed or deployment eligibility is assigned",
    }


def _typed_same_module_count_holdout_summary(frozen_split: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only the preregistered cross-layout same-M development checks."""

    training = list(frozen_split.get("training_layouts", []))
    development = list(frozen_split.get("development_layouts", []))
    train_by_count: dict[int, list[dict[str, Any]]] = {}
    dev_by_count: dict[int, list[dict[str, Any]]] = {}
    for item in training:
        train_by_count.setdefault(int(item["turbine_count"]), []).append(item)
    for item in development:
        dev_by_count.setdefault(int(item["turbine_count"]), []).append(item)
    comparisons: dict[str, Any] = {}
    for count in sorted(set(train_by_count) & set(dev_by_count)):
        comparisons[str(count)] = {
            "module_count": count,
            "training_layouts": [
                {
                    "layout_index": int(item["layout_index"]),
                    "rows_direction_order": list(map(int, item["rows_direction_order"])),
                }
                for item in train_by_count[count]
            ],
            "development_layouts": [
                {
                    "layout_index": int(item["layout_index"]),
                    "rows_direction_order": list(map(int, item["rows_direction_order"])),
                }
                for item in dev_by_count[count]
            ],
            "interpretation": "held-out layout comparison at a module count seen in training",
        }
    repeated_training_counts = {
        str(count): [int(item["layout_index"]) for item in layouts]
        for count, layouts in train_by_count.items()
        if len(layouts) > 1
    }
    return {
        "same_module_count_cross_layout_holdouts": comparisons,
        "same_module_count_counts": sorted(map(int, comparisons)),
        "training_layouts_with_repeated_module_counts": repeated_training_counts,
        "training_same_module_count_layout_pairs": [],
        "within_training_same_module_count_adaptation_comparison_available": False,
        "interpretation": (
            "Compare held-out development layouts to train layouts at M=6 and M=29. Training layouts have no repeated M, "
            "so this split cannot estimate search adaptation across different layouts at the same M within training."
        ),
    }


def _typed_initial_prediction_gradient_probe(
    organizer: InputOnlyCoverOrganizer,
    model: Any,
    record: Any,
    *,
    device: torch.device,
    forward_completed: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Require useful native-output gradients at the all-access hard initialization."""

    with torch.enable_grad():
        scores = _typed_scores(organizer, record)
        plan = organizer.plans_from_scores(
            (scores,), record.encoded, record.trees,
            hard=True, straight_through_hard=True,
        )[0]
        if not plan.is_full_access() or bool((plan.split_gates != 0).any()):
            raise RuntimeError("initial typed predictive-gradient probe must use all-access hard masks")
        _prepared, _encoded, _context, output = _typed_direct_forward(
            model,
            record.search_batch,
            plan,
            device=device,
            return_aux=False,
        )
        if forward_completed is not None:
            forward_completed()
        prediction = output["pred_field"]
        teacher_loss = _typed_role_mean_square(
            prediction, record.full_search_standardized, record.search.roles
        )
        reference_loss = _typed_role_mean_square(
            prediction, record.search_batch.target_field, record.search.roles
        )
        protected_loss = teacher_loss + TYPED_REFERENCE_FEEDBACK_WEIGHT * reference_loss
        pair_parameters = {
            mechanism: tuple(organizer.pair_scorers[mechanism].parameters())
            for mechanism in INTERACTION_MECHANISMS
        }
        flat_parameters = tuple(
            parameter
            for mechanism in INTERACTION_MECHANISMS
            for parameter in pair_parameters[mechanism]
        )
        gradients = torch.autograd.grad(
            protected_loss, flat_parameters, allow_unused=True
        )
        by_mechanism: dict[str, float] = {}
        cursor = 0
        for mechanism in INTERACTION_MECHANISMS:
            count = len(pair_parameters[mechanism])
            values = gradients[cursor : cursor + count]
            by_mechanism[mechanism] = (
                float(torch.sqrt(sum(
                    value.detach().square().sum() for value in values if value is not None
                )).cpu())
                if any(value is not None for value in values)
                else 0.0
            )
            cursor += count
        total_gradient = sum(by_mechanism.values())
        if not math.isfinite(total_gradient) or total_gradient < TYPED_MIN_PROTECTED_PREDICTION_GRADIENT_L2:
            raise RuntimeError(
                "protected native output loss has no usable gradient through all-access typed permission heads: "
                f"L2={total_gradient:.8g}"
            )
    return {
        "row_index": int(record.case.index),
        "hard_plan_hash": plan.canonical_hash(),
        "hard_full_access": True,
        "split_count": int((plan.split_gates >= 0.5).sum().detach().cpu()),
        "protected_teacher_loss": float(teacher_loss.detach().cpu()),
        "native_reference_loss": float(reference_loss.detach().cpu()),
        "protected_prediction_gradient_l2_by_mechanism": by_mechanism,
        "protected_prediction_gradient_l2_sum_by_mechanism": total_gradient,
        "minimum_usable_gradient_l2": TYPED_MIN_PROTECTED_PREDICTION_GRADIENT_L2,
        "gradient_measured_before_optimizer_update": True,
        "straight_through_hard_forward": True,
        "complete_native_forward_calls": 1,
    }


def _attach_typed_reference_status(
    row_results: list[dict[str, Any]],
    native_grid_results: list[dict[str, Any]],
) -> None:
    """Copy only measured, declared full-grid role-guard outcomes into row reports."""

    by_row = {int(item["row_index"]): item for item in native_grid_results}
    if len(by_row) != len(native_grid_results):
        raise ValueError("native full-grid reference results contain duplicate training rows")
    for row in row_results:
        row_index = int(row["row_index"])
        evidence = by_row.get(row_index)
        if evidence is None:
            row["reference_sufficient"] = "unknown"
            row["reference_evidence"] = {
                "status": "unknown",
                "reason": "no declared full-grid reference guard was executed for this row",
            }
            continue
        guard = evidence["native_metrics"]["train_only_physical_reference_guard"]
        status = str(guard.get("status", "unknown"))
        if status not in {"passed", "failed", "unknown"}:
            raise ValueError(f"unexpected native reference guard status {status!r}")
        row["reference_sufficient"] = status
        row["reference_evidence"] = {
            "status": status,
            "source": "frozen train/development layout native grid",
            "layout_index": int(evidence["layout_index"]),
            "partition": evidence.get("partition"),
            "guard": guard,
        }


def _typed_select_full_grid_rows(
    frozen_split: Mapping[str, Any],
    q1024_rows_by_id: Mapping[int, dict[str, Any]],
    *,
    native_validation_layouts: int,
    include_development_layout: bool,
) -> tuple[list[int], dict[str, Any]]:
    """Select fixed train rows and optionally one predeclared held-out layout row."""

    if not 1 <= native_validation_layouts <= 2:
        raise ValueError("typed full-grid evaluation is capped at two total layouts")
    development_slots = int(include_development_layout)
    training_slots = native_validation_layouts - development_slots
    if training_slots < 0:
        raise ValueError("full-grid layout cap cannot fit the requested development row")

    selected: list[int] = []
    selected_metadata: list[dict[str, Any]] = []

    def require_q1024_pass(row: int, partition: str, layout_index: int) -> dict[str, Any]:
        evidence = q1024_rows_by_id.get(row)
        if evidence is None:
            raise ValueError(f"typed full-grid row {row} has no matching Q1024 artifact")
        if (
            evidence.get("status") != "complete"
            or evidence.get("partition") != partition
            or int(evidence.get("layout_index", -1)) != layout_index
            or not bool(evidence.get("teacher_preservation", {}).get("passed"))
        ):
            raise ValueError(
                f"typed full-grid {partition} row {row} lacks a passing Q1024 protected teacher gate"
            )
        if partition == "development":
            if evidence.get("development_g2_labels_loaded") is not False:
                raise ValueError("development full-grid selection must not load per-case G2 labels")
            if evidence.get("independent_holdout") is not True:
                raise ValueError("development full-grid selection requires an independent held-out Q1024 row")
        return evidence

    development_layout = None
    if include_development_layout:
        development_layouts = {
            int(item["layout_index"]): item
            for item in frozen_split.get("development_layouts", [])
        }
        # The same-M M=6 cross-layout comparison was fixed before outcomes;
        # layout 103 is therefore the first choice, with M=29 layout 81 as
        # the schema-compatible fallback for alternate frozen panels.
        for preferred in (103, 81):
            if preferred in development_layouts:
                development_layout = development_layouts[preferred]
                break
        if development_layout is None:
            raise ValueError("frozen split contains neither predeclared dev layout 103 nor 81")

    training_layouts = list(frozen_split.get("training_layouts", []))
    if development_layout is not None:
        dev_turbine_count = int(development_layout["turbine_count"])
        training_layouts = [
            item for item in training_layouts
            if int(item["turbine_count"]) == dev_turbine_count
        ]
        if training_slots and not training_layouts:
            raise ValueError("frozen split has no same-M training layout for the selected dev reference")
    for layout_doc in training_layouts[:training_slots]:
        layout = int(layout_doc["layout_index"])
        row = int(layout_doc["rows_direction_order"][0])
        evidence = require_q1024_pass(row, "training", layout)
        selected.append(row)
        selected_metadata.append({
            "row_index": row,
            "layout_index": layout,
            "partition": "training",
            "q1024_teacher_gate_passed": True,
            "source_q1024_row_artifact": evidence.get("artifact_path"),
            "development_g2_labels_loaded": False,
        })

    if include_development_layout:
        assert development_layout is not None
        layout = int(development_layout["layout_index"])
        row = int(development_layout["rows_direction_order"][0])
        evidence = require_q1024_pass(row, "development", layout)
        selected.append(row)
        selected_metadata.append({
            "row_index": row,
            "layout_index": layout,
            "partition": "development",
            "q1024_teacher_gate_passed": True,
            "source_q1024_row_artifact": evidence.get("artifact_path"),
            "development_g2_labels_loaded": False,
            "development_outputs_used_for_row_selection": False,
            "selection_rule": "predeclared same-M held-out layout 103 (or layout 81 fallback), first frozen direction",
        })
    if len(selected) != native_validation_layouts:
        raise RuntimeError("typed full-grid row selection does not match its total layout cap")
    return selected, {
        "native_validation_layouts_cap": native_validation_layouts,
        "development_layout_included": bool(include_development_layout),
        "development_rows_used_for_search_or_fit": [],
        "development_g2_labels_loaded": False,
        "selected_layout_rows": selected_metadata,
    }


def _masked_binary_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, Any]:
    selected = observed.to(dtype=torch.bool)
    predicted = prediction[selected] >= 0.5
    truth = target[selected] >= 0.5
    positive, negative = truth, ~truth
    return {
        "observed_count": int(selected.sum().detach().cpu()),
        "observed_positive_count": int(positive.sum().detach().cpu()),
        "observed_negative_count": int(negative.sum().detach().cpu()),
        "predicted_positive_count": int(predicted.sum().detach().cpu()),
        "masked_accuracy": float((predicted == truth).float().mean().detach().cpu()) if truth.numel() else None,
        "positive_accuracy": (
            float((predicted[positive] == truth[positive]).float().mean().detach().cpu())
            if bool(positive.any()) else None
        ),
        "negative_accuracy": (
            float((predicted[negative] == truth[negative]).float().mean().detach().cpu())
            if bool(negative.any()) else None
        ),
    }


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
    if not bool(assessment["learning_eligible"]):
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
    attempted_optimizer_calls = 0
    completed_optimizer_calls = 0
    checkpoint_paths: list[str] = []
    update_ledger_path = destination / "optimizer_update_ledger.jsonl"
    fit_started = time.perf_counter()
    organizer.train()
    with update_ledger_path.open("w", encoding="utf-8") as update_ledger:
        for update, index in enumerate(order[:updates], start=1):
            record = records[int(index)]
            optimizer.zero_grad(set_to_none=True)
            loss = _training_loss(organizer, record, label_by_row[int(record.case.index)])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite organizer supervised loss at update {update}")
            loss.backward()
            clip_grad_norm_(organizer.parameters(), max_norm=1.0)
            attempted_optimizer_calls += 1
            update_ledger.write(json.dumps({
                "update": update,
                "event": "optimizer_step_attempted",
                "row_index": int(record.case.index),
                "loss": float(loss.detach().cpu()),
            }, sort_keys=True) + "\n")
            update_ledger.flush()
            optimizer.step()
            completed_optimizer_calls += 1
            losses.append(float(loss.detach().cpu()))
            update_ledger.write(json.dumps({
                "update": update,
                "event": "optimizer_step_completed",
            }, sort_keys=True) + "\n")
            update_ledger.flush()
            if update in {1, 50, updates}:
                checkpoint_path = destination / f"organizer_checkpoint_update_{update:04d}.pt"
                _atomic_torch_save({
                    "state_dict": organizer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "checkpoint_sha256": checkpoint_hash,
                    "source_checkpoint": str(checkpoint),
                    "teacher_checkpoint_id": teacher_checkpoint_id,
                    "updates_attempted": attempted_optimizer_calls,
                    "updates_completed": completed_optimizer_calls,
                    "seed": seed,
                    "active_scope": "input_only_cover_organizer",
                    "active_terms": ["observed_split_bce", "observed_module_bce", "observed_environment_bce"],
                    "loss_scales": {"split": 1.0, "module": 1.0, "environment": 1.0},
                    "plan_schema": "legacy_adaptive_cover_root_membership_v1",
                    "organizer_split_sha256": assessment["evidence_scope"]["organizer_split_sha256"],
                    "sampler_order": order[:updates].tolist(),
                    "sampler_cursor": update,
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_cpu_rng_state": torch.get_rng_state(),
                    "torch_cuda_rng_state": (
                        torch.cuda.get_rng_state(target_device) if target_device.type == "cuda" else None
                    ),
                }, checkpoint_path)
                checkpoint_paths.append(str(checkpoint_path))
    organizer.eval()
    fit_elapsed = time.perf_counter() - fit_started
    state_path = destination / "input_only_organizer_state.pt"
    _atomic_torch_save({
        "state_dict": organizer.state_dict(),
        "checkpoint_sha256": checkpoint_hash,
        "updates_attempted": attempted_optimizer_calls,
        "updates_completed": completed_optimizer_calls,
        "seed": seed,
        "architecture": "InputOnlyCoverOrganizer",
        "teacher_checkpoint_id": teacher_checkpoint_id,
        "organizer_split_sha256": assessment["evidence_scope"]["organizer_split_sha256"],
        "plan_schema": "legacy_adaptive_cover_root_membership_v1",
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
        verified_oracle_report, _verified_oracle_prepared, _verified_oracle_plan = base._evaluate_probe_variant(
            model,
            record,
            name="per-case-verified-oracle-hard",
            probe=record.verification,
            batch=record.verification_batch,
            policy=base._FixedPlanPolicy((record.oracle_plan,)),
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
            "verified_oracle_hard": verified_oracle_report,
            "fitted_hard_organizer": fitted_report,
            "masked_label_metrics": _masked_hard_label_metrics(
                fitted_plan,
                label_by_row[row_index],
            ),
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
            "attempted_optimizer_calls": attempted_optimizer_calls,
            "completed_optimizer_calls": completed_optimizer_calls,
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "loss": "pre-activation BCE; observed positive and negative classes equally weighted when both occur",
            "active_terms": ["observed_split_bce", "observed_module_bce", "observed_environment_bce"],
            "plan_schema": "legacy_adaptive_cover_root_membership_v1",
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_mean": float(np.mean(losses)) if losses else None,
            "elapsed_seconds": fit_elapsed,
            "update_ledger_path": str(update_ledger_path),
            "atomic_checkpoint_paths": checkpoint_paths,
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


def run_typed_organizer_fit(
    *,
    checkpoint_path: str | Path,
    volume_path: str | Path,
    compact_path: str | Path,
    typed_search_dir: str | Path,
    device: str | torch.device,
    output_dir: str | Path,
    query_count: int = 1024,
    seed: int = FIT_SEED,
    supervised_updates: int = TYPED_SUPERVISED_UPDATE_CAP,
    predictive_updates: int = TYPED_PREDICTIVE_UPDATE_CAP,
    native_validation_layouts: int = 2,
) -> dict[str, Any]:
    """Fit the formal typed organizer with hard-forward native predictive feedback."""

    workflow_started = time.perf_counter()
    if query_count != 1024:
        raise ValueError("formal typed fitting is locked to disjoint Q1024 train probes")
    if not 0 <= supervised_updates <= TYPED_SUPERVISED_UPDATE_CAP:
        raise ValueError(f"supervised warm-up is capped at {TYPED_SUPERVISED_UPDATE_CAP} updates")
    if not 0 <= predictive_updates <= TYPED_PREDICTIVE_UPDATE_CAP:
        raise ValueError(f"predictive training is capped at {TYPED_PREDICTIVE_UPDATE_CAP} updates")
    total_updates = int(supervised_updates + predictive_updates)
    if total_updates < 1 or total_updates > TYPED_FORMAL_UPDATE_CAP:
        raise ValueError(f"formal organizer updates must be between 1 and {TYPED_FORMAL_UPDATE_CAP}")
    if predictive_updates and not supervised_updates:
        raise ValueError("predictive feedback requires at least one supervised warm-up update")
    if not 0 <= native_validation_layouts <= 2:
        raise ValueError("typed full-grid evaluation may be skipped or capped at two frozen train layouts")

    destination = Path(output_dir).expanduser().resolve()
    if "generated" not in destination.parts:
        raise ValueError("typed fit artifacts must stay under ignored diagnostics/generated")
    destination.mkdir(parents=True, exist_ok=True)
    progress_path = destination / "typed_fit_progress.json"
    if progress_path.exists() or (destination / "typed_organizer_report.json").exists():
        raise FileExistsError("typed fit output already exists; select a fresh ignored run directory")

    prior_manifest_path = (
        Path(__file__).resolve().parents[4]
        / "Case_WindFarm/diagnostics/generated/native_cover_organizer/run2103_e2475"
        / "six_label_hard_diagnostic_gpu2/diagnostic_timeout_manifest.json"
    )
    if not prior_manifest_path.is_file():
        raise FileNotFoundError("the historical G0 optimizer ledger is required for cumulative update accounting")
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    prior_updates = int(prior_manifest.get("optimizer", {}).get("completed_calls_from_ledger", -1))
    if prior_updates != TYPED_G0_PRIOR_UPDATES:
        raise ValueError("formal typed fit requires the attested 100-update G0 diagnostic history")

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    checkpoint_hash = base._checkpoint_sha256(checkpoint)
    search_dir = Path(typed_search_dir).expanduser().resolve()
    search_manifest_path = search_dir / "panel_manifest.json"
    if not search_manifest_path.is_file():
        raise FileNotFoundError("typed fit requires the G2 frozen panel manifest")
    search_manifest = json.loads(search_manifest_path.read_text(encoding="utf-8"))
    anchor_variant = str(search_manifest.get("anchor_measure_variant", ""))
    if anchor_variant not in {"raw", "role_balanced"}:
        raise ValueError("typed G2 panel manifest lacks a recognized anchor-measure variant")
    (
        checkpoint,
        payload,
        checkpoint_hash,
        view,
        _split,
        layouts,
        _manifest_path,
        panel_manifest,
    ) = base._freeze_training_panel(
        checkpoint_path=checkpoint,
        volume_path=volume_path,
        compact_path=compact_path,
        output_dir=destination,
        layout_count=12,
        active_layouts=base.TYPED_TRAIN_LAYOUT_COUNT,
        anchor_measure_variant=anchor_variant,
    )
    search_report, search_manifest, documents = _typed_fit_artifacts(
        typed_search_dir,
        checkpoint_sha256=checkpoint_hash,
        layouts=layouts,
    )
    frozen_split = base.freeze_organizer_layout_split(layouts)
    if panel_manifest.get("organizer_split_frozen_before_new_outcomes", {}).get("split_sha256") != frozen_split[
        "split_sha256"
    ]:
        raise ValueError("formal typed fit panel differs from the locked geometry-only split")
    split_lock_path = Path(str(search_report.get("split_lock_path", ""))).expanduser().resolve()
    if not split_lock_path.is_file():
        raise FileNotFoundError("typed G2 split-lock artifact is missing")
    split_lock_hash = base._checkpoint_sha256(split_lock_path)
    if split_lock_hash != search_report.get("split_lock_sha256"):
        raise ValueError("typed G2 split-lock bytes do not match the search report")

    active_layouts = base._active_organizer_training_layouts(layouts, base.TYPED_TRAIN_LAYOUT_COUNT)
    active_rows = [int(row) for layout in active_layouts for row in layout.rows]
    expected_rows = list(map(int, frozen_split["training_rows_direction_order"]))
    if active_rows != expected_rows or active_rows != list(map(int, search_report["training_rows"])):
        raise ValueError("formal typed fit must use exactly the frozen 24 training directions")
    development_rows = set(map(int, frozen_split["development_rows_direction_order"]))
    if development_rows.intersection(active_rows):
        raise RuntimeError("development direction entered formal typed fitting")

    normalizer = VelocityNormalizer.from_dict(dict(payload["normalization"]))
    target_device = torch.device(device)
    first_case = view.run(active_rows[0])
    first_search, _ = make_disjoint_native_probes(first_case, query_count=query_count, seed=seed)
    materialization_batch = base._model_batch(
        base._probe_batch(first_case, first_search, normalizer), target_device
    )
    model, _loaded = load_checkpoint(
        checkpoint, device=target_device, materialization_batch=materialization_batch
    )
    if model.architecture != "dense_pairwise_field":
        raise ValueError("typed organizer must preserve the intact frozen Run2103 Dense teacher")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    teacher_checkpoint_id = f"Run2103:e2475:{checkpoint_hash[:16]}"

    records: list[Any] = []
    labels_by_row: dict[int, dict[str, Any]] = {}
    target_plans: dict[int, MechanismPlan] = {}
    panel_timings: list[dict[str, Any]] = []
    for row in active_rows:
        record, timing = base._build_panel_case(
            model,
            view.run(row),
            query_count=query_count,
            seed=seed,
            teacher_checkpoint_id=teacher_checkpoint_id,
            device=target_device,
            anchor_measure_variant=anchor_variant,
        )
        document = documents[int(row)]
        labels = _typed_label_tensors(document, target_device)
        plan = _typed_plan_from_document(record, document, target_device)
        record.oracle_plan = plan
        records.append(record)
        labels_by_row[int(row)] = labels
        target_plans[int(row)] = plan
        panel_timings.append(timing)

    first = records[0]
    encoded = first.encoded
    torch.manual_seed(seed)
    organizer = InputOnlyCoverOrganizer(
        state_dim=int(encoded.module_tokens.shape[-1]),
        module_feature_dim=int(encoded.module_features.shape[-1]),
        environment_feature_dim=0 if encoded.env_features is None else int(encoded.env_features.shape[-1]),
        hidden_dim=32,
        role_count=max(8, int(first.trees[0].universe.roles.max().item()) + 1),
        quadrature_invariant_source_measure=True,
    ).to(target_device)
    initialization = _initialize_typed_organizer_all_access(organizer)
    optimizer = torch.optim.AdamW(organizer.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    rng = np.random.default_rng(seed)
    supervised_order = np.concatenate([
        rng.permutation(len(records))
        for _ in range((supervised_updates + len(records) - 1) // len(records))
    ])[:supervised_updates].astype(int).tolist()
    predictive_order = np.concatenate([
        rng.permutation(len(records))
        for _ in range((predictive_updates + len(records) - 1) // len(records))
    ])[:predictive_updates].astype(int).tolist()

    organizer.eval()
    parity_rows: list[dict[str, Any]] = []
    for record in records:
        hard_plan = _typed_hard_plan(organizer, record)
        if not hard_plan.is_full_access() or bool((hard_plan.split_gates != 0).any()):
            raise RuntimeError("typed organizer all-access hard initialization is not full and unsplit")
        with torch.inference_mode():
            _prepared, prediction, _output, prepare_ms, decode_ms = base._typed_replay_plan(
                model,
                record.search_batch,
                hard_plan,
                device=target_device,
            )
        max_delta = base._max_abs(prediction, record.full_search_prediction)
        passed = bool(torch.allclose(
            prediction,
            record.full_search_prediction,
            atol=base.COLD_WARM_ABS_TOLERANCE_MPS,
            rtol=base.COLD_WARM_REL_TOLERANCE,
        ))
        parity_rows.append({
            "row_index": int(record.case.index),
            "layout_index": int(record.case.layout_index),
            "plan_hash": hard_plan.canonical_hash(),
            "full_access_all_five_mechanisms": bool(hard_plan.is_full_access()),
            "split_count": int((hard_plan.split_gates >= 0.5).sum()),
            "max_abs_mps_vs_policy_none_dense": max_delta,
            "passed": passed,
            "prepare_ms": float(prepare_ms),
            "decode_ms": float(decode_ms),
        })
        if not passed:
            raise RuntimeError(
                f"hard all-access typed organizer failed native parity on row {record.case.index}: {max_delta:.8g} m/s"
            )
    parity_path = destination / "typed_all_access_parity.json"
    complete_forward_count = 1 + 4 * len(records) + len(parity_rows)
    progress = {
        "workflow": "windfarm_typed_organizer_supervised_predictive_fit",
        "status": "running",
        "checkpoint_sha256": checkpoint_hash,
        "typed_search_report_sha256": base._checkpoint_sha256(
            Path(typed_search_dir).expanduser().resolve() / "typed_search_report.json"
        ),
        "frozen_split_sha256": frozen_split["split_sha256"],
        "split_lock_sha256": split_lock_hash,
        "training_rows": active_rows,
        "development_rows_used_for_search_or_fit": [],
        "supervised_update_cap": supervised_updates,
        "predictive_update_cap": predictive_updates,
        "prior_G0_optimizer_updates": prior_updates,
        "total_complete_native_forward_calls": complete_forward_count,
        "all_access_parity_complete_forward_calls": len(parity_rows),
        "initial_gradient_probe_complete_forward_calls": 0,
        "completed_updates": 0,
        "attempted_updates": 0,
    }
    _atomic_json_write(progress, progress_path)
    initial_gradient_probe: dict[str, Any] | None = None
    if predictive_updates:
        initial_probe_path = destination / "typed_all_access_gradient_probe.json"
        initial_probe_document: dict[str, Any] = {
            "status": "running",
            "checkpoint_sha256": checkpoint_hash,
            "training_row_index": int(first.case.index),
            "hard_forward_state": "all five mechanisms active, no split",
            "optimizer_updates_before_probe": 0,
        }
        _atomic_json_write(initial_probe_document, initial_probe_path)

        def count_initial_gradient_forward() -> None:
            nonlocal complete_forward_count
            complete_forward_count += 1
            progress["initial_gradient_probe_complete_forward_calls"] = 1
            progress["total_complete_native_forward_calls"] = complete_forward_count
            progress["current_stage"] = "all_access_gradient_probe"
            progress["current_row_index"] = int(first.case.index)
            _atomic_json_write(progress, progress_path)

        try:
            initial_gradient_probe = _typed_initial_prediction_gradient_probe(
                organizer,
                model,
                first,
                device=target_device,
                forward_completed=count_initial_gradient_forward,
            )
        except Exception as exc:
            initial_probe_document.update({
                "status": "failed",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "complete_native_forward_calls": int(
                    progress["initial_gradient_probe_complete_forward_calls"]
                ),
            })
            progress.update({
                "status": "failed",
                "failure_stage": "all_access_gradient_probe",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "total_complete_native_forward_calls": complete_forward_count,
                "optimizer_updates": 0,
                "new_physical_solves": 0,
            })
            _atomic_json_write(initial_probe_document, initial_probe_path)
            _atomic_json_write(progress, progress_path)
            raise
        initial_probe_document.update({"status": "complete", **initial_gradient_probe})
        _atomic_json_write(initial_probe_document, initial_probe_path)
    _atomic_json_write({
        "checkpoint_sha256": checkpoint_hash,
        "typed_search_report": str(Path(typed_search_dir).expanduser().resolve() / "typed_search_report.json"),
        "plan_schema": "honf-mechanism-plan-v1",
        "initialization": initialization,
        "training_started_after_parity": True,
        "complete_native_forward_calls": len(parity_rows)
        + int(initial_gradient_probe is not None),
        "initial_all_access_gradient_probe": initial_gradient_probe,
        "rows": parity_rows,
        "all_passed": all(item["passed"] for item in parity_rows),
    }, parity_path)

    update_ledger_path = destination / "typed_optimizer_update_ledger.jsonl"
    if update_ledger_path.exists():
        raise FileExistsError("typed optimizer update ledger already exists")
    checkpoint_paths: list[str] = []
    attempted_updates = 0
    completed_updates = 0
    supervised_losses: list[float] = []
    predictive_losses: list[dict[str, float]] = []
    predictive_gradient_evidence: dict[str, Any] | None = None
    hard_mask_checkpoints: list[dict[str, Any]] = []
    predictive_native_forwards = 0
    optimizer_training_started = time.perf_counter()

    def snapshot_hard_masks(update: int, stage: str) -> None:
        organizer.eval()
        row_summaries = []
        with torch.no_grad():
            for record in records:
                plan = _typed_hard_plan(organizer, record)
                row_summaries.append({
                    "row_index": int(record.case.index),
                    "layout_index": int(record.case.layout_index),
                    "plan_hash": plan.canonical_hash(),
                    "source_support_by_mechanism": {
                        mechanism: {
                            "source_count": int((plan.permission_matrix(mechanism) > 0.5).any(dim=0).sum()),
                            "active_node_count": int((plan.permission_matrix(mechanism) > 0.5).any(dim=1).sum()),
                        }
                        for mechanism in INTERACTION_MECHANISMS
                    },
                })
        hard_mask_checkpoints.append({
            "completed_update": update,
            "stage": stage,
            "rows": row_summaries,
        })

    def save_checkpoint(update: int, stage: str) -> None:
        path = destination / f"typed_organizer_checkpoint_update_{update:04d}.pt"
        _atomic_torch_save({
            "state_dict": organizer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "checkpoint_sha256": checkpoint_hash,
            "source_checkpoint": str(checkpoint),
            "teacher_checkpoint_id": teacher_checkpoint_id,
            "typed_search_report": str(Path(typed_search_dir).expanduser().resolve() / "typed_search_report.json"),
            "typed_search_report_sha256": base._checkpoint_sha256(
                Path(typed_search_dir).expanduser().resolve() / "typed_search_report.json"
            ),
            "completed_updates": update,
            "updates_attempted": attempted_updates,
            "optimizer_state_scope": "input_only_typed_cover_organizer",
            "active_stage": stage,
            "plan_schema": "honf-mechanism-plan-v1",
            "typed_organizer_input_state_schema": TYPED_ORGANIZER_INPUT_STATE_SCHEMA,
            "frozen_split_sha256": frozen_split["split_sha256"],
            "split_lock_sha256": split_lock_hash,
            "prior_G0_optimizer_updates": prior_updates,
            "cumulative_organizer_optimizer_updates": prior_updates + update,
            "supervised_sampler_order": supervised_order,
            "predictive_sampler_order": predictive_order,
            "numpy_rng_state": rng.bit_generator.state,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": (
                torch.cuda.get_rng_state(target_device) if target_device.type == "cuda" else None
            ),
        }, path)
        checkpoint_paths.append(str(path))

    progress.update({
        "status": "running",
        "total_complete_native_forward_calls": complete_forward_count,
        "initial_gradient_probe_complete_forward_calls": int(initial_gradient_probe is not None),
        "parity_artifact": str(parity_path),
        "completed_updates": 0,
        "attempted_updates": 0,
    })
    _atomic_json_write(progress, progress_path)

    with update_ledger_path.open("w", encoding="utf-8") as ledger:
        for stage, order in (("supervised", supervised_order), ("predictive", predictive_order)):
            for record_index in order:
                record = records[int(record_index)]
                row_index = int(record.case.index)
                labels = labels_by_row[row_index]
                local_stage_update = len(supervised_losses) + 1 if stage == "supervised" else len(predictive_losses) + 1
                update = completed_updates + 1
                organizer.train()
                optimizer.zero_grad(set_to_none=True)
                if stage == "supervised":
                    loss = _typed_supervised_loss(organizer, record, labels)
                    pieces = {"supervised_loss": float(loss.detach().cpu())}
                else:
                    scores = _typed_scores(organizer, record)
                    plan = organizer.plans_from_scores(
                        (scores,),
                        record.encoded,
                        record.trees,
                        hard=True,
                        straight_through_hard=True,
                    )[0]
                    if not bool(((plan.split_gates == 0) | (plan.split_gates == 1)).all()):
                        raise RuntimeError("predictive stage plan forward must use exact binary split decisions")
                    for mechanism in INTERACTION_MECHANISMS:
                        membership = plan.permission_matrix(mechanism)
                        if not bool(((membership == 0) | (membership == 1)).all()):
                            raise RuntimeError(
                                f"predictive stage {mechanism} forward mask is not exactly binary"
                            )
                    _prepared, _encoded, _context, output = _typed_direct_forward(
                        model,
                        record.search_batch,
                        plan,
                        device=target_device,
                    )
                    predictive_native_forwards += 1
                    complete_forward_count += 1
                    progress.update({
                        "total_complete_native_forward_calls": complete_forward_count,
                        "predictive_native_forward_calls": predictive_native_forwards,
                        "current_stage": "predictive_hard_forward",
                        "current_row_index": row_index,
                    })
                    _atomic_json_write(progress, progress_path)
                    prediction = output["pred_field"]
                    teacher_loss = _typed_role_mean_square(
                        prediction,
                        record.full_search_standardized,
                        record.search.roles,
                    )
                    reference_loss = _typed_role_mean_square(
                        prediction,
                        record.search_batch.target_field,
                        record.search.roles,
                    )
                    supervised_loss = _typed_supervised_loss_from_scores(scores, labels)
                    query = record.search_batch.query_xy[0]
                    work_loss = _typed_differentiable_work(plan, record.encoded, query)
                    protected_loss = teacher_loss + TYPED_REFERENCE_FEEDBACK_WEIGHT * reference_loss
                    loss = (
                        protected_loss
                        + TYPED_SUPERVISED_FEEDBACK_WEIGHT * supervised_loss
                        + TYPED_WORK_PENALTY_WEIGHT * work_loss
                    )
                    pieces = {
                        "protected_teacher_loss": float(teacher_loss.detach().cpu()),
                        "native_reference_loss": float(reference_loss.detach().cpu()),
                        "coherent_supervision_loss": float(supervised_loss.detach().cpu()),
                        "logical_work_fraction": float(work_loss.detach().cpu()),
                        "predictive_total_loss": float(loss.detach().cpu()),
                        "hard_plan_hash": plan.canonical_hash(),
                    }
                    if predictive_gradient_evidence is None:
                        pair_parameters = {
                            mechanism: tuple(organizer.pair_scorers[mechanism].parameters())
                            for mechanism in INTERACTION_MECHANISMS
                        }
                        prediction_gradients = torch.autograd.grad(
                            protected_loss,
                            tuple(parameter for values in pair_parameters.values() for parameter in values),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        work_gradients = torch.autograd.grad(
                            work_loss,
                            tuple(parameter for values in pair_parameters.values() for parameter in values),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        prediction_norms: dict[str, float] = {}
                        work_norms: dict[str, float] = {}
                        cursor = 0
                        for mechanism in INTERACTION_MECHANISMS:
                            count = len(pair_parameters[mechanism])
                            values = prediction_gradients[cursor : cursor + count]
                            work_values = work_gradients[cursor : cursor + count]
                            prediction_norms[mechanism] = float(torch.sqrt(sum(
                                value.detach().square().sum()
                                for value in values if value is not None
                            )).cpu()) if any(value is not None for value in values) else 0.0
                            work_norms[mechanism] = float(torch.sqrt(sum(
                                value.detach().square().sum()
                                for value in work_values if value is not None
                            )).cpu()) if any(value is not None for value in work_values) else 0.0
                            cursor += count
                        protected_gradient_norm = sum(prediction_norms.values())
                        if protected_gradient_norm < TYPED_MIN_PROTECTED_PREDICTION_GRADIENT_L2:
                            raise RuntimeError(
                                "protected native output loss has no usable gradient through typed permission heads: "
                                f"L2={protected_gradient_norm:.8g}"
                            )
                        predictive_gradient_evidence = {
                            "row_index": row_index,
                            "hard_plan_hash": plan.canonical_hash(),
                            "surrogate_gradient": "hard + sigmoid(logits) - sigmoid(logits).detach(); exact hard forward, approximate topology gradient",
                            "protected_prediction_loss_pair_scorer_gradient_l2_by_mechanism": prediction_norms,
                            "protected_prediction_gradient_l2_sum_by_mechanism": protected_gradient_norm,
                            "protected_prediction_gradient_l2_minimum_usable": (
                                TYPED_MIN_PROTECTED_PREDICTION_GRADIENT_L2
                            ),
                            "logical_work_loss_pair_scorer_gradient_l2_by_mechanism": work_norms,
                            "protected_prediction_gradient_nonzero": True,
                            "protected_prediction_gradient_usable": True,
                            "logical_work_gradient_nonzero": sum(work_norms.values()) > 0.0,
                            "gradient_measured_before_optimizer_update": True,
                        }
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite typed {stage} loss at update {update}")
                loss.backward()
                grad_norm = clip_grad_norm_(organizer.parameters(), max_norm=1.0)
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError(f"nonfinite typed organizer gradient at update {update}")
                attempted_updates += 1
                ledger.write(json.dumps({
                    "update": update,
                    "stage": stage,
                    "stage_update": local_stage_update,
                    "event": "optimizer_step_attempted",
                    "row_index": row_index,
                    "loss": float(loss.detach().cpu()),
                    "loss_terms": pieces,
                    "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                }, sort_keys=True, allow_nan=False) + "\n")
                ledger.flush()
                optimizer.step()
                completed_updates += 1
                if stage == "supervised":
                    supervised_losses.append(float(loss.detach().cpu()))
                else:
                    predictive_losses.append({key: float(value) for key, value in pieces.items() if isinstance(value, (float, int))})
                ledger.write(json.dumps({
                    "update": update,
                    "stage": stage,
                    "event": "optimizer_step_completed",
                }, sort_keys=True) + "\n")
                ledger.flush()
                organizer.eval()
                if completed_updates == 1 or completed_updates % 50 == 0 or completed_updates in {
                    supervised_updates, total_updates
                }:
                    snapshot_hard_masks(completed_updates, stage)
                    save_checkpoint(completed_updates, stage)
                _atomic_json_write({
                    "workflow": "windfarm_typed_organizer_supervised_predictive_fit",
                    "status": "running",
                    "checkpoint_sha256": checkpoint_hash,
                    "typed_search_report_sha256": base._checkpoint_sha256(
                        Path(typed_search_dir).expanduser().resolve() / "typed_search_report.json"
                    ),
                    "frozen_split_sha256": frozen_split["split_sha256"],
                    "split_lock_sha256": split_lock_hash,
                    "training_rows": active_rows,
                    "development_rows_used_for_search_or_fit": [],
                    "supervised_updates_cap": supervised_updates,
                    "predictive_updates_cap": predictive_updates,
                    "completed_updates": completed_updates,
                    "attempted_updates": attempted_updates,
                    "complete_native_forward_calls": complete_forward_count,
                    "predictive_native_forward_calls": predictive_native_forwards,
                    "latest_stage": stage,
                    "latest_row_index": row_index,
                }, progress_path)

    optimizer_training_wall_seconds = float(time.perf_counter() - optimizer_training_started)
    if predictive_updates and predictive_gradient_evidence is None:
        raise RuntimeError("predictive updates completed without a measured typed-path gradient check")
    state_path = destination / "typed_input_only_organizer_state.pt"
    _atomic_torch_save({
        "state_dict": organizer.state_dict(),
        "checkpoint_sha256": checkpoint_hash,
        "typed_search_report_sha256": base._checkpoint_sha256(
            Path(typed_search_dir).expanduser().resolve() / "typed_search_report.json"
        ),
        "updates_attempted": attempted_updates,
        "updates_completed": completed_updates,
        "prior_G0_optimizer_updates": prior_updates,
        "cumulative_organizer_optimizer_updates": prior_updates + completed_updates,
        "seed": seed,
        "architecture": "InputOnlyCoverOrganizer",
        "typed_organizer_input_state_schema": TYPED_ORGANIZER_INPUT_STATE_SCHEMA,
        "typed_organizer_quadrature_measure_schema": TYPED_ORGANIZER_QUADRATURE_MEASURE_SCHEMA,
        "teacher_checkpoint_id": teacher_checkpoint_id,
        "organizer_split_sha256": frozen_split["split_sha256"],
        "plan_schema": "honf-mechanism-plan-v1",
        "straight_through_hard": bool(predictive_updates),
    }, state_path)

    postfit_evaluation_started = time.perf_counter()
    organizer.eval()
    row_results: list[dict[str, Any]] = []
    final_plans: dict[int, MechanismPlan] = {}
    for record in records:
        row_index = int(record.case.index)
        learned_plan = _typed_hard_plan(organizer, record)
        final_plans[row_index] = learned_plan
        with torch.inference_mode():
            _prepared, learned_prediction, _output, prepare_ms, decode_ms = base._typed_replay_plan(
                model,
                record.verification_batch,
                learned_plan,
                device=target_device,
            )
            _oracle_prepared, oracle_prediction, _oracle_output, oracle_prepare_ms, oracle_decode_ms = base._typed_replay_plan(
                model,
                record.verification_batch,
                target_plans[row_index],
                device=target_device,
            )
        complete_forward_count += 2
        learned_obs, learned_metrics = base._probe_observation(
            learned_prediction[0],
            record.full_verification_prediction[0],
            record.verification,
            normalizer,
            teacher_checkpoint_id=teacher_checkpoint_id,
            training_evidence_id=f"run2103_typed_fit_row_{row_index}_verification_s{seed}_Q{query_count}",
        )
        learned_passed, learned_reasons = base.teacher_preservation_gate(
            learned_obs,
            {
                name: base.TeacherDistortionLimit(base.TYPED_PROTECTED_ROLE_TEACHER_LIMIT)
                for name in record.verification.roles
            },
        )
        oracle_obs, oracle_metrics = base._probe_observation(
            oracle_prediction[0],
            record.full_verification_prediction[0],
            record.verification,
            normalizer,
            teacher_checkpoint_id=teacher_checkpoint_id,
            training_evidence_id=f"run2103_typed_G2_row_{row_index}_verification_s{seed}_Q{query_count}",
        )
        oracle_passed, oracle_reasons = base.teacher_preservation_gate(
            oracle_obs,
            {
                name: base.TeacherDistortionLimit(base.TYPED_PROTECTED_ROLE_TEACHER_LIMIT)
                for name in record.verification.roles
            },
        )
        full_plan = MechanismPlan.full_access(
            record.trees[0], record.encoded.module_present[0], int(record.encoded.env_coords.shape[1])
        )
        mask_changes = {}
        for mechanism in INTERACTION_MECHANISMS:
            predicted = learned_plan.permission_matrix(mechanism) >= 0.5
            full = full_plan.permission_matrix(mechanism) >= 0.5
            mask_changes[mechanism] = {
                "changed_entry_count_vs_all_access": int((predicted != full).sum().detach().cpu()),
                "hard_source_support": int(_typed_native_source_indices(
                    learned_plan,
                    record.encoded,
                    record.verification_batch.query_xy[0],
                    mechanism,
                ).numel()),
                "hard_active_receiver_nodes": int((predicted.any(dim=1)).sum().detach().cpu()),
                "frontier": learned_plan.frontier_summary(
                    mechanism,
                    receivers=base._typed_mechanism_receivers(
                        record.encoded, record.verification_batch.query_xy[0]
                    )[mechanism],
                ).as_dict(prefix="cover"),
            }
        result_row = {
            "row_index": row_index,
            "layout_index": int(record.case.layout_index),
            "direction_deg": float(record.case.wind_direction_deg),
            "learned_plan_hash": learned_plan.canonical_hash(),
            "learned_plan": learned_plan.to_dict(),
            "learned_plan_work": base._typed_work_summary(
                learned_plan,
                record.encoded,
                record.verification_batch.query_xy[0],
            ),
            "hard_mask_changes_vs_all_access": mask_changes,
            "hard_label_metrics": _typed_hard_label_metrics(learned_plan, labels_by_row[row_index]),
            "protected_role_teacher_preservation": {
                "passed": bool(learned_passed),
                "limit_per_role": base.TYPED_PROTECTED_ROLE_TEACHER_LIMIT,
                "reasons": base._teacher_gate_reason_list(learned_reasons),
                "normalized_distortion_by_role": {
                    role: (None if error.teacher is None else float(error.teacher))
                    for role, error in learned_obs.roles.items()
                },
                "reference_vector_rmse_mps_by_role": {
                    role: error.reference for role, error in learned_obs.roles.items()
                },
                "role_metrics": learned_metrics,
                "actual_synchronized_prepare_ms": float(prepare_ms),
                "actual_synchronized_decode_ms": float(decode_ms),
                "actual_synchronized_complete_ms": float(prepare_ms + decode_ms),
            },
            "verified_G2_control": {
                "passed": bool(oracle_passed),
                "reasons": base._teacher_gate_reason_list(oracle_reasons),
                "normalized_distortion_by_role": {
                    role: (None if error.teacher is None else float(error.teacher))
                    for role, error in oracle_obs.roles.items()
                },
                "reference_vector_rmse_mps_by_role": {
                    role: error.reference for role, error in oracle_obs.roles.items()
                },
                "role_metrics": oracle_metrics,
                "actual_synchronized_prepare_ms": float(oracle_prepare_ms),
                "actual_synchronized_decode_ms": float(oracle_decode_ms),
                "actual_synchronized_complete_ms": float(oracle_prepare_ms + oracle_decode_ms),
                "source_plan_hash": target_plans[row_index].canonical_hash(),
            },
            "learning_eligible": True,
            "reference_sufficient": "unknown",
            "deployment_eligible": False,
        }
        row_results.append(result_row)
        _atomic_json_write(result_row, destination / f"typed_fit_row_{row_index:04d}.json")

    q1024_evaluation_wall_seconds = float(time.perf_counter() - postfit_evaluation_started)
    native_grid_started = time.perf_counter()
    unique_layouts = list(dict.fromkeys(int(record.case.layout_index) for record in records))
    native_grid_results: list[dict[str, Any]] = []
    for layout_index in unique_layouts[:native_validation_layouts]:
        record = next(item for item in records if int(item.case.layout_index) == layout_index)
        row_index = int(record.case.index)
        plan = final_plans[row_index]
        model.set_native_interaction_policy(None)
        with torch.inference_mode():
            candidate_prepared, _candidate_encoded, candidate_context = base._prepare_typed_plan(
                model, record.search_batch, plan, device=target_device
            )
        metrics = _native_candidate_reference_grid(
            model,
            candidate_prepared,
            record.full_prepared,
            record.case,
            normalizer,
            device=target_device,
            candidate_is_core_prepared=True,
            candidate_interaction_context=candidate_context,
        )
        chunks = math.ceil(int(record.case.run.cell_count) / 8192)
        grid_result = {
            "row_index": row_index,
            "layout_index": layout_index,
            "direction_deg": float(record.case.wind_direction_deg),
            "native_cell_count": int(record.case.run.cell_count),
            "native_decode_call_count": 2 * chunks,
            "learned_hard_plan_hash": plan.canonical_hash(),
            "native_metrics": metrics,
        }
        native_grid_results.append(grid_result)
        _atomic_json_write(grid_result, destination / f"typed_fit_native_layout_{layout_index:04d}.json")

    _attach_typed_reference_status(row_results, native_grid_results)
    for row_result in row_results:
        _atomic_json_write(
            row_result,
            destination / f"typed_fit_row_{int(row_result['row_index']):04d}.json",
        )
    native_grid_evaluation_wall_seconds = float(time.perf_counter() - native_grid_started)
    statuses = [item["native_metrics"]["train_only_physical_reference_guard"]["status"] for item in native_grid_results]
    reference_status = (
        "unknown"
        if native_validation_layouts == 0
        or len(native_grid_results) < native_validation_layouts
        or any(value == "unknown" for value in statuses)
        else "passed" if all(value == "passed" for value in statuses)
        else "failed"
    )
    fit_elapsed = float(time.perf_counter() - workflow_started)
    result = {
        "status": "complete",
        "workflow": "windfarm_typed_organizer_supervised_predictive_fit",
        "plan_schema": "honf-mechanism-plan-v1",
        "typed_organizer_input_provenance": _typed_organizer_input_provenance(),
        "legacy_G0_provenance_remains_separate": "legacy_adaptive_cover_root_membership_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "teacher_checkpoint_id": teacher_checkpoint_id,
        "typed_search_directory": str(Path(typed_search_dir).expanduser().resolve()),
        "anchor_measure_variant": anchor_variant,
        "typed_search_report_sha256": base._checkpoint_sha256(
            Path(typed_search_dir).expanduser().resolve() / "typed_search_report.json"
        ),
        "typed_split_lock_path": str(split_lock_path),
        "typed_split_lock_sha256": split_lock_hash,
        "frozen_split_sha256": frozen_split["split_sha256"],
        "training_layout_indices": list(frozen_split["training_layout_indices"]),
        "development_layout_indices": list(frozen_split["development_layout_indices"]),
        "training_rows": active_rows,
        "development_rows_used_for_search_or_fit": [],
        "training": {
            "supervised_updates": len(supervised_losses),
            "predictive_updates": len(predictive_losses),
            "attempted_optimizer_calls": attempted_updates,
            "completed_optimizer_calls": completed_updates,
            "prior_G0_optimizer_updates": prior_updates,
            "cumulative_organizer_optimizer_updates": prior_updates + completed_updates,
            "cumulative_cap_including_G0": TYPED_G0_PRIOR_UPDATES + TYPED_FORMAL_UPDATE_CAP,
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "supervised_loss": "class-balanced masked BCE for recursive split and all five typed permission mechanisms",
            "predictive_loss": {
                "protected_teacher_role_mean_square_weight": 1.0,
                "native_reference_role_mean_square_weight": TYPED_REFERENCE_FEEDBACK_WEIGHT,
                "coherent_supervision_weight": TYPED_SUPERVISED_FEEDBACK_WEIGHT,
                "differentiable_logical_work_weight": TYPED_WORK_PENALTY_WEIGHT,
                "logical_work_definition": "mean query/source permission mass across MM/ME/EM/QM/QE, divided by all-access rectangular pairs",
            },
            "hard_initialization": initialization,
            "all_access_native_parity_artifact": str(parity_path),
            "all_access_parity_complete_native_forwards": len(parity_rows),
            "all_access_parity_passed": all(item["passed"] for item in parity_rows),
            "straight_through_gradient_evidence": predictive_gradient_evidence,
            "supervised_loss_first": supervised_losses[0] if supervised_losses else None,
            "supervised_loss_last": supervised_losses[-1] if supervised_losses else None,
            "predictive_loss_first": predictive_losses[0] if predictive_losses else None,
            "predictive_loss_last": predictive_losses[-1] if predictive_losses else None,
            "hard_mask_checkpoints": hard_mask_checkpoints,
            "checkpoint_paths": checkpoint_paths,
            "state_path": str(state_path),
            "elapsed_seconds": fit_elapsed,
            "stage_wall_seconds": {
                "optimizer_training": optimizer_training_wall_seconds,
                "q1024_postfit_evaluation": q1024_evaluation_wall_seconds,
                "native_grid_evaluation": native_grid_evaluation_wall_seconds,
            },
        },
        "forward_accounting": {
            "materialization_complete_forward_calls": 1,
            "panel_native_reference_complete_forward_calls": 4 * len(records),
            "all_access_parity_complete_forward_calls": len(parity_rows),
            "predictive_update_complete_forward_calls": predictive_native_forwards,
            "learned_and_G2_verification_complete_forward_calls": 2 * len(records),
            "native_grid_decode_calls": sum(item["native_decode_call_count"] for item in native_grid_results),
            "total_complete_native_forward_calls_including_grid_chunks": complete_forward_count
            + sum(item["native_decode_call_count"] for item in native_grid_results),
            "optimizer_updates_are_not_forward_calls": True,
        },
        "disjoint_verification_rows": row_results,
        "native_grid_validation": native_grid_results,
        "learning_eligible": True,
        "reference_sufficient": reference_status,
        "measured_speed_eligible": False,
        "deployment_eligible": False,
        "deployment_reason": "production complete-workload synchronized timing and development-layout generalization remain unmeasured",
        "new_physical_solves": 0,
    }
    progress.update({
        "status": "complete",
        "completed_updates": completed_updates,
        "attempted_updates": attempted_updates,
        "complete_native_forward_calls": result["forward_accounting"]["total_complete_native_forward_calls_including_grid_chunks"],
        "summary_path": str(destination / "typed_organizer_report.json"),
        "elapsed_seconds": fit_elapsed,
    })
    _atomic_json_write(progress, progress_path)
    _atomic_json_write(result, destination / "typed_organizer_report.json")
    return result


def _load_saved_organizer_context(
    *,
    checkpoint_path: str | Path,
    volume_path: str | Path,
    compact_path: str | Path,
    oracle_dir: str | Path,
    organizer_checkpoint_path: str | Path,
    device: str | torch.device,
    output_dir: str | Path,
    query_count: int,
    seed: int,
    row_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Rebuild the frozen train inputs and load a saved legacy u100 state."""

    destination = Path(output_dir).expanduser().resolve()
    if "generated" not in destination.parts:
        raise ValueError("checkpoint-only evaluation outputs must stay under diagnostics/generated")
    destination.mkdir(parents=True, exist_ok=True)
    assessment = write_fit_target_assessment(
        checkpoint_path=checkpoint_path, oracle_dir=oracle_dir, output_dir=destination
    )
    if not bool(assessment["learning_eligible"]):
        raise ValueError("saved-state evaluation requires the provenance-valid G0 learning labels")
    oracle_directory, oracle_report, oracle_manifest, plan_documents = _load_fit_artifacts(oracle_dir)
    active_count = int(oracle_report.get("active_training_layouts", 0))
    if not 1 <= active_count <= 8:
        raise ValueError("G0 saved-state evaluation is restricted to the frozen eight-layout train split")
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
        active_layouts=active_count,
        anchor_measure_variant=str(oracle_manifest.get("anchor_measure_variant", "raw")),
    )
    if checkpoint_hash != oracle_report.get("checkpoint_sha256"):
        raise ValueError("saved-state evaluation checkpoint differs from the frozen oracle checkpoint")
    if frozen_manifest.get("organizer_split_frozen_before_new_outcomes", {}).get("split_sha256") != (
        assessment["evidence_scope"]["organizer_split_sha256"]
    ):
        raise ValueError("saved-state evaluation geometry split differs from the pre-outcome lock")
    if frozen_manifest.get("layouts") != oracle_manifest.get("layouts"):
        raise ValueError("saved-state evaluation panel differs from the frozen oracle panel")

    active_layouts = base._active_organizer_training_layouts(layouts, active_count)
    active_rows = [int(row) for layout in active_layouts for row in layout.rows]
    if row_ids is None:
        rows_to_use = _fit_row_ids_to_use(active_rows, assessment)
    else:
        rows_to_use = [int(row) for row in row_ids]
        if not set(rows_to_use).issubset(set(active_rows)):
            raise ValueError("checkpoint-only evaluation requested a row outside the frozen train layouts")
        label_rows = set(plan_documents)
        if not set(rows_to_use).issubset(label_rows):
            raise ValueError("checkpoint-only evaluation requested a row without a verified G0 label")
    split = _frozen_organizer_split_from_manifest(frozen_manifest)
    train_layout_ids = set(map(int, split["training_layout_indices"]))
    dev_layout_ids = set(map(int, split["development_layout_indices"]))
    for row in rows_to_use:
        layout_index = int(view.run(row).layout_index)
        if layout_index not in train_layout_ids or layout_index in dev_layout_ids:
            raise ValueError(f"development layout {layout_index} cannot enter G0 fit/evaluation")

    normalizer = VelocityNormalizer.from_dict(dict(payload["normalization"]))
    target_device = torch.device(device)
    first_case = view.run(rows_to_use[0])
    first_search, _first_verification = make_disjoint_native_probes(
        first_case, query_count=query_count, seed=seed
    )
    materialization_batch = base._model_batch(
        base._probe_batch(first_case, first_search, normalizer), target_device
    )
    model, _loaded = load_checkpoint(
        checkpoint, device=target_device, materialization_batch=materialization_batch
    )
    if model.architecture != "dense_pairwise_field":
        raise ValueError("saved legacy G0 state requires the intact Dense Run2103 teacher")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    teacher_checkpoint_id = f"Run2103:e2475:{checkpoint_hash[:16]}"

    records: list[Any] = []
    labels_by_row: dict[int, dict[str, torch.Tensor]] = {}
    build_timings: list[dict[str, Any]] = []
    for row in rows_to_use:
        case = view.run(row)
        record, timing = base._build_panel_case(
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
            raise ValueError(f"missing verified legacy supervision document for train row {row}")
        labels_by_row[int(row)] = _label_tensors(document, target_device)
        record.oracle_plan = _plan_from_labels(record, labels_by_row[int(row)])
        records.append(record)
        build_timings.append(timing)

    first = records[0]
    encoded = first.encoded
    organizer = InputOnlyCoverOrganizer(
        state_dim=int(encoded.module_tokens.shape[-1]),
        module_feature_dim=int(encoded.module_features.shape[-1]),
        environment_feature_dim=0 if encoded.env_features is None else int(encoded.env_features.shape[-1]),
        hidden_dim=32,
        role_count=max(8, int(first.trees[0].universe.roles.max().item()) + 1),
    ).to(target_device)
    state_path = Path(organizer_checkpoint_path).expanduser().resolve()
    state_document = torch.load(state_path, map_location=target_device, weights_only=True)
    if not isinstance(state_document, dict) or "state_dict" not in state_document:
        raise ValueError("saved organizer checkpoint has no state_dict")
    if state_document.get("plan_schema") != "legacy_adaptive_cover_root_membership_v1":
        raise ValueError("G0 evaluation cannot consume typed formal organizer provenance")
    if state_document.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("saved organizer state belongs to a different field checkpoint")
    expected_split = assessment["evidence_scope"]["organizer_split_sha256"]
    if state_document.get("organizer_split_sha256") != expected_split:
        raise ValueError("saved organizer state belongs to a different frozen geometry split")
    attempted = int(state_document.get("updates_attempted", -1))
    completed = int(state_document.get("updates_completed", -1))
    if attempted != 100 or completed != 100:
        raise ValueError("G0 checkpoint-only evaluation requires exactly 100 attempted and completed updates")
    organizer.load_state_dict(state_document["state_dict"], strict=True)
    organizer.eval()
    module_mask, environment_mask, fixed_summary = base._population_fixed_masks(records)
    return {
        "assessment": assessment,
        "oracle_directory": oracle_directory,
        "oracle_report": oracle_report,
        "oracle_manifest": oracle_manifest,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "teacher_checkpoint_id": teacher_checkpoint_id,
        "view": view,
        "normalizer": normalizer,
        "device": target_device,
        "model": model,
        "records": records,
        "labels_by_row": labels_by_row,
        "build_timings": build_timings,
        "organizer": organizer,
        "organizer_state_path": state_path,
        "organizer_state_sha256": base._checkpoint_sha256(state_path),
        "organizer_updates_attempted": attempted,
        "organizer_updates_completed": completed,
        "split_sha256": expected_split,
        "module_mask": module_mask,
        "environment_mask": environment_mask,
        "fixed_summary": fixed_summary,
        "active_train_layout_indices": [int(item.layout_index) for item in active_layouts],
    }


def _saved_state_provenance(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_sha256": context["checkpoint_sha256"],
        "teacher_checkpoint_id": context["teacher_checkpoint_id"],
        "organizer_state_path": str(context["organizer_state_path"]),
        "organizer_state_sha256": context["organizer_state_sha256"],
        "organizer_updates_attempted": context["organizer_updates_attempted"],
        "organizer_updates_completed": context["organizer_updates_completed"],
        "organizer_plan_schema": "legacy_adaptive_cover_root_membership_v1",
        "organizer_split_sha256": context["split_sha256"],
    }


def _hard_plan_from_document(record: Any, document: dict[str, Any], device: torch.device) -> AdaptiveCoverPlan:
    raw = document["fitted_hard_plan"]
    return AdaptiveCoverPlan(
        record.trees[0],
        torch.as_tensor(raw["split_gates"], device=device, dtype=torch.float32),
        torch.as_tensor(raw["module_membership"], device=device, dtype=torch.float32),
        torch.as_tensor(raw["environment_membership"], device=device, dtype=torch.float32),
    )


def _evaluate_saved_disjoint_rows(
    context: dict[str, Any],
    *,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "disjoint_evaluation_progress.json"
    provenance = _saved_state_provenance(context)
    variants = (
        "policy_none_dense",
        "learned_hard",
        "verified_oracle_hard",
        "population_fixed_support",
        "collapsed_pair_control",
    )
    existing = list(output_dir.glob("row_*_variant_*.json")) + list(output_dir.glob("row_*_complete.json"))
    if existing and not resume:
        raise FileExistsError("evaluation output already contains row artifacts; pass --resume-evaluation")
    progress: dict[str, Any] = {
        "stage": "disjoint_q1024",
        "status": "running",
        **provenance,
        "query_count_per_disjoint_probe": 1024,
        "variants_per_row": list(variants),
        "completed_variant_artifacts": [],
        "completed_row_artifacts": [],
        "complete_variant_forward_calls": 0,
        "replayed_variant_forward_calls": 0,
        "reference_sufficient": "unknown",
        "deployment_eligible": False,
    }
    if resume and progress_path.is_file():
        prior = json.loads(progress_path.read_text(encoding="utf-8"))
        for key in ("checkpoint_sha256", "organizer_state_sha256", "organizer_split_sha256"):
            if prior.get(key) != provenance[key]:
                raise ValueError(f"cannot resume disjoint evaluation with changed {key}")
        progress.update({
            "completed_variant_artifacts": list(prior.get("completed_variant_artifacts", [])),
            "completed_row_artifacts": list(prior.get("completed_row_artifacts", [])),
            "complete_variant_forward_calls": int(prior.get("complete_variant_forward_calls", 0)),
            "replayed_variant_forward_calls": int(prior.get("replayed_variant_forward_calls", 0)),
        })
    _atomic_json_write(progress, progress_path)
    completed_keys = set(progress["completed_variant_artifacts"])
    row_results: list[dict[str, Any]] = []
    for record in context["records"]:
        row_index = int(record.case.index)
        row_path = output_dir / f"row_{row_index:04d}_complete.json"
        if resume and row_path.is_file():
            row_doc = json.loads(row_path.read_text(encoding="utf-8"))
            if row_doc.get("provenance") != provenance:
                raise ValueError(f"row {row_index} saved result provenance mismatch")
            row_results.append(row_doc)
            progress["replayed_variant_forward_calls"] += len(variants)
            continue
        labels = context["labels_by_row"][row_index]
        row_variant_reports: dict[str, dict[str, Any]] = {}
        fitted_plan: AdaptiveCoverPlan | None = None
        for variant in variants:
            variant_path = output_dir / f"row_{row_index:04d}_variant_{variant}.json"
            if resume and variant_path.is_file():
                variant_doc = json.loads(variant_path.read_text(encoding="utf-8"))
                if variant_doc.get("provenance") != provenance:
                    raise ValueError(f"row {row_index} variant {variant} provenance mismatch")
                row_variant_reports[variant] = variant_doc["result"]
                if variant == "learned_hard":
                    fitted_plan = _hard_plan_from_document(record, variant_doc["result"], context["device"])
                progress["replayed_variant_forward_calls"] += 1
                continue
            if variant == "policy_none_dense":
                report, _prepared, _plan = base._evaluate_probe_variant(
                    context["model"], record,
                    name="policy-none-Dense",
                    probe=record.verification,
                    batch=record.verification_batch,
                    policy=None,
                    reference_prediction=record.full_verification_prediction,
                    teacher_checkpoint_id=context["teacher_checkpoint_id"],
                    device=context["device"],
                )
            elif variant == "learned_hard":
                report, _prepared, fitted_plan = base._evaluate_probe_variant(
                    context["model"], record,
                    name="G-input-only-organizer-hard",
                    probe=record.verification,
                    batch=record.verification_batch,
                    policy=context["organizer"],
                    reference_prediction=record.full_verification_prediction,
                    teacher_checkpoint_id=context["teacher_checkpoint_id"],
                    device=context["device"],
                )
                report["masked_label_metrics"] = _masked_hard_label_metrics(
                    fitted_plan, labels
                )
                report["fitted_hard_plan"] = {
                    "split_gates": fitted_plan.split_gates.detach().cpu().tolist(),
                    "module_membership": fitted_plan.module_membership.detach().cpu().tolist(),
                    "environment_membership": fitted_plan.environment_membership.detach().cpu().tolist(),
                    "hard_k": int(fitted_plan.active_group_count()),
                }
            elif variant == "verified_oracle_hard":
                report, _prepared, _plan = base._evaluate_probe_variant(
                    context["model"], record,
                    name="per-case-verified-oracle-hard",
                    probe=record.verification,
                    batch=record.verification_batch,
                    policy=base._FixedPlanPolicy((record.oracle_plan,)),
                    reference_prediction=record.full_verification_prediction,
                    teacher_checkpoint_id=context["teacher_checkpoint_id"],
                    device=context["device"],
                )
            elif variant == "population_fixed_support":
                fixed_modules = torch.as_tensor(
                    context["module_mask"][: int(record.oracle_plan.module_membership.shape[1])],
                    device=record.oracle_plan.module_membership.device,
                    dtype=record.oracle_plan.module_membership.dtype,
                )
                fixed_environment = torch.as_tensor(
                    context["environment_mask"],
                    device=record.oracle_plan.environment_membership.device,
                    dtype=record.oracle_plan.environment_membership.dtype,
                )
                fixed_plan = base._root_mask_plan(record.full_plan, fixed_modules, fixed_environment)
                report, _prepared, _plan = base._evaluate_probe_variant(
                    context["model"], record,
                    name="training-population-fixed-cover-control",
                    probe=record.verification,
                    batch=record.verification_batch,
                    policy=base._FixedPlanPolicy((fixed_plan,)),
                    reference_prediction=record.full_verification_prediction,
                    teacher_checkpoint_id=context["teacher_checkpoint_id"],
                    device=context["device"],
                )
            else:
                pair_plan = base._per_case_pair_control(record)
                report, _prepared, _plan = base._evaluate_probe_variant(
                    context["model"], record,
                    name="per-case-collapsed-pair-control",
                    probe=record.verification,
                    batch=record.verification_batch,
                    policy=base._FixedPlanPolicy((pair_plan,)),
                    reference_prediction=record.full_verification_prediction,
                    teacher_checkpoint_id=context["teacher_checkpoint_id"],
                    device=context["device"],
                )
            row_variant_reports[variant] = report
            result_payload: dict[str, Any] = {"provenance": provenance, "result": report}
            _atomic_json_write(result_payload, variant_path)
            completed_keys.add(str(variant_path))
            progress["completed_variant_artifacts"] = sorted(completed_keys)
            progress["complete_variant_forward_calls"] += 1
            _atomic_json_write(progress, progress_path)
        if fitted_plan is None:
            fitted_plan = _hard_plan_from_document(
                record,
                json.loads((output_dir / f"row_{row_index:04d}_variant_learned_hard.json").read_text())["result"],
                context["device"],
            )
        row_doc = {
            "provenance": provenance,
            "row_index": row_index,
            "layout_index": int(record.case.layout_index),
            "direction_deg": float(record.case.wind_direction_deg),
            "probe_label": "disjoint Q1024 train probe; physical native reference and checkpoint teacher are separate",
            "variants": row_variant_reports,
            "fitted_hard_plan": {
                "split_gates": fitted_plan.split_gates.detach().cpu().tolist(),
                "module_membership": fitted_plan.module_membership.detach().cpu().tolist(),
                "environment_membership": fitted_plan.environment_membership.detach().cpu().tolist(),
                "hard_k": int(fitted_plan.active_group_count()),
            },
        }
        _atomic_json_write(row_doc, row_path)
        row_results.append(row_doc)
        progress["completed_row_artifacts"] = sorted(
            set(progress["completed_row_artifacts"]) | {str(row_path)}
        )
        _atomic_json_write(progress, progress_path)

    learned_by_layout: dict[int, list[dict[str, Any]]] = {}
    for item in row_results:
        learned_by_layout.setdefault(int(item["layout_index"]), []).append(
            item["variants"]["learned_hard"]
        )
    faster_layouts = [
        layout for layout, items in learned_by_layout.items()
        if all(
            float(item["measured_complete_ms"])
            < float(next(row["variants"]["policy_none_dense"]["measured_complete_ms"]
                         for row in row_results if int(row["row_index"]) == int(item["row_index"])))
            for item in items
        )
    ]
    result = {
        "status": "disjoint_q1024_complete",
        "stage": "disjoint_q1024",
        "workflow": "native_cover_legacy_g0_checkpoint_only_evaluation",
        **provenance,
        "learning_eligible": True,
        "reference_sufficient": "unknown",
        "measured_speed_eligible": len(faster_layouts) >= MIN_COMPETITIVE_PARTIAL_LAYOUTS,
        "deployment_eligible": False,
        "deployment_reason": "Q1024 disjoint evidence does not replace full native-grid and production-workload review",
        "frozen_train_layout_indices": context["active_train_layout_indices"],
        "evaluated_rows": [int(item["row_index"]) for item in row_results],
        "optimizer_calls_in_this_stage": 0,
        "optimizer_calls_in_source_state": 100,
        "forward_call_accounting": {
            "completed_variant_prepare_decode_calls": int(progress["complete_variant_forward_calls"]),
            "replayed_variant_artifacts": int(progress["replayed_variant_forward_calls"]),
            "panel_build_per_row": {
                "prepare_case_calls": 2,
                "decode_calls": 4,
                "scope": "one dense and one explicit all-access prepare per row, each decoded on disjoint search and verification Q1024 probes",
            },
            "complete_hard_plan_and_control_variants_per_row": 5,
        },
        "population_fixed_support_definition": context["fixed_summary"],
        "faster_layouts_at_q1024": faster_layouts,
        "row_results": [str(output_dir / f"row_{int(item['row_index']):04d}_complete.json") for item in row_results],
        "full_grid_next_step": "consider only if learned hard plans preserve every disjoint protected role and the isolated budget is reassigned",
    }
    progress.update({
        "status": "complete",
        "summary_path": str(output_dir / "disjoint_evaluation_report.json"),
        "reference_sufficient": "unknown",
        "deployment_eligible": False,
    })
    _atomic_json_write(progress, progress_path)
    _atomic_json_write(result, output_dir / "disjoint_evaluation_report.json")
    return result


def _native_candidate_reference_grid(
    model: Any,
    candidate_prepared: Any,
    reference_prepared: Any,
    case: Any,
    normalizer: VelocityNormalizer,
    *,
    device: torch.device,
    chunk_size: int = 8192,
    candidate_is_core_prepared: bool = False,
    candidate_interaction_context: InteractionContext | None = None,
) -> dict[str, Any]:
    """Stream learned and G-full predictions together over stored native cells."""

    run = case.run
    wx, wy, wz = support_weights(run.x_m, run.y_m, run.z_m)
    wx = np.asarray(wx, dtype=np.float64) / float(case.diameter_m)
    wy = np.asarray(wy, dtype=np.float64) / float(case.diameter_m)
    wz = np.asarray(wz, dtype=np.float64) / float(case.diameter_m)
    roles = ("volume", "hub_slab", "downstream_envelope", "background", "near_turbine")
    candidate_errors = {name: WeightedVelocityErrors() for name in roles}
    reference_errors = {name: WeightedVelocityErrors() for name in roles}
    direct_errors = {name: WeightedVelocityErrors() for name in roles}
    hubs = np.asarray(case.module_centers, dtype=np.float64)
    hub_height = float(case.hub_height_m) / float(case.diameter_m)
    background = base._native_background_profile(case)
    residual_error_candidate = residual_energy_candidate = 0.0
    residual_error_reference = residual_energy_reference = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, int(run.cell_count), chunk_size):
            stop = min(start + chunk_size, int(run.cell_count))
            flat = np.arange(start, stop, dtype=np.int64)
            coords_D = native_coordinates(run, flat, float(case.diameter_m))
            geometry = case.geometry_for_queries(coords_D)
            query = torch.as_tensor(geometry["query_xy"], device=device).unsqueeze(0)
            features = torch.as_tensor(geometry["query_features"], device=device).unsqueeze(0)
            if candidate_is_core_prepared:
                candidate = model.core.decode_queries(
                    candidate_prepared,
                    query,
                    query_features=features,
                    interaction_context=candidate_interaction_context,
                )["pred_field"][0]
            else:
                candidate = model.decode(candidate_prepared, query, query_features=features)["pred_field"][0]
            reference = model.decode(reference_prepared, query, query_features=features)["pred_field"][0]
            mean = candidate.new_tensor(normalizer.mean)
            scale = candidate.new_tensor(normalizer.safe_std)
            candidate_mps = ((candidate * scale + mean) * float(normalizer.u_ref_mps)).cpu().numpy()
            reference_mps = ((reference * scale + mean) * float(normalizer.u_ref_mps)).cpu().numpy()
            truth = np.asarray(run.U[start:stop], dtype=np.float64).copy()
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
                    candidate_errors[name].add(candidate_mps[mask], truth[mask], weights[mask])
                    reference_errors[name].add(reference_mps[mask], truth[mask], weights[mask])
                    direct_errors[name].add(candidate_mps[mask], reference_mps[mask], weights[mask])
            if bool(wake.any()):
                candidate_residual = candidate_mps[wake, 0] - background[iz[wake]]
                reference_residual = reference_mps[wake, 0] - background[iz[wake]]
                truth_residual = truth[wake, 0] - background[iz[wake]]
                residual_error_candidate += float(np.sum(weights[wake] * (candidate_residual - truth_residual) ** 2))
                residual_energy_candidate += float(np.sum(weights[wake] * truth_residual**2))
                residual_error_reference += float(np.sum(weights[wake] * (reference_residual - truth_residual) ** 2))
                residual_energy_reference += float(np.sum(weights[wake] * truth_residual**2))
    base._sync(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    physical_std = normalizer.safe_std * float(normalizer.u_ref_mps)
    def vector_rmse(item: dict[str, Any]) -> float:
        return float(
            np.sqrt(np.sum(np.asarray(item["squared_error_integral"], dtype=np.float64))
                    / float(item["quadrature_volume_D3"]) / 3.0)
        )
    candidate_metrics = {name: item.result(physical_std, float(normalizer.u_ref_mps)) for name, item in candidate_errors.items()}
    reference_metrics = {name: item.result(physical_std, float(normalizer.u_ref_mps)) for name, item in reference_errors.items()}
    direct_metrics = {name: item.result(physical_std, float(normalizer.u_ref_mps)) for name, item in direct_errors.items()}
    parity_allowance_mps = 1.0e-5
    guard_by_role: dict[str, Any] = {}
    for role in roles:
        candidate_value = candidate_metrics[role]
        reference_value = reference_metrics[role]
        if not candidate_value.get("available") or not reference_value.get("available"):
            guard_by_role[role] = {"measured": False, "passed": None}
            continue
        candidate_rmse = vector_rmse(candidate_value)
        reference_rmse = vector_rmse(reference_value)
        guard_by_role[role] = {
            "measured": True,
            "candidate_equal_layout_vector_rmse_mps": candidate_rmse,
            "G_full_equal_layout_vector_rmse_mps": reference_rmse,
            "maximum_allowed_additional_error_fraction": 0.10,
            "numerical_parity_allowance_mps": parity_allowance_mps,
            "passed": bool(candidate_rmse <= 1.10 * reference_rmse + parity_allowance_mps),
        }
    all_measured = all(bool(item.get("measured")) for item in guard_by_role.values())
    reference_passed = all_measured and all(bool(item.get("passed")) for item in guard_by_role.values())
    return {
        "native_cell_count": int(run.cell_count),
        "chunk_size": int(chunk_size),
        "native_decode_passes": 2,
        "complete_native_grid_elapsed_ms": float(elapsed_ms),
        "candidate_physical_reference_metrics": candidate_metrics,
        "G_full_physical_reference_metrics": reference_metrics,
        "direct_candidate_minus_G_full_role_errors": direct_metrics,
        "wake_ux_residual_relative_l2": {
            "candidate": float(np.sqrt(residual_error_candidate / residual_energy_candidate))
            if residual_energy_candidate > 0 else None,
            "G_full": float(np.sqrt(residual_error_reference / residual_energy_reference))
            if residual_energy_reference > 0 else None,
        },
        "train_only_physical_reference_guard": {
            "status": "passed" if reference_passed else "failed" if all_measured else "unknown",
            "protected_roles": guard_by_role,
            "interpretation": "candidate versus incumbent G-full stored native-reference error, equal-layout and per protected role",
        },
    }


def _evaluate_saved_full_grid(
    context: dict[str, Any],
    *,
    output_dir: Path,
    disjoint_results_dir: str | Path,
    native_validation_layouts: int,
) -> dict[str, Any]:
    if not 1 <= native_validation_layouts <= 2:
        raise ValueError("G0 full native validation is capped at two frozen training layouts")
    source_dir = Path(disjoint_results_dir).expanduser().resolve()
    source_report_path = source_dir / "disjoint_evaluation_report.json"
    if not source_report_path.is_file():
        raise FileNotFoundError("full-grid evaluation requires the completed checkpoint-only Q1024 disjoint report")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    provenance = _saved_state_provenance(context)
    for key in ("checkpoint_sha256", "organizer_state_sha256", "organizer_split_sha256"):
        if source_report.get(key) != provenance[key]:
            raise ValueError(f"full-grid pass Q1024 provenance mismatch for {key}")
    source_row_ids = set(map(int, source_report.get("evaluated_rows", [])))
    layouts: list[int] = []
    for record in context["records"]:
        layout = int(record.case.layout_index)
        if layout not in layouts:
            layouts.append(layout)
    selected_layouts = layouts[:native_validation_layouts]
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "full_grid_evaluation_progress.json"
    progress = {
        "stage": "full_grid_native_reference",
        "status": "running",
        **provenance,
        "source_disjoint_report": str(source_report_path),
        "native_validation_layouts_cap": native_validation_layouts,
        "completed_layout_artifacts": [],
        "native_decode_passes_per_layout": 2,
        "optimizer_calls_in_this_stage": 0,
        "deployment_eligible": False,
    }
    _atomic_json_write(progress, progress_path)
    layout_results: list[dict[str, Any]] = []
    for layout in selected_layouts:
        record = next(item for item in context["records"] if int(item.case.layout_index) == layout)
        row_index = int(record.case.index)
        if row_index not in source_row_ids:
            raise ValueError(f"Q1024 report has no hard-plan row for selected layout {layout}")
        row_path = source_dir / f"row_{row_index:04d}_complete.json"
        if not row_path.is_file():
            raise FileNotFoundError(f"missing atomic Q1024 row result for full-grid row {row_index}")
        row_doc = json.loads(row_path.read_text(encoding="utf-8"))
        if row_doc.get("provenance") != provenance:
            raise ValueError(f"Q1024 row {row_index} state/checkpoint provenance mismatch")
        plan = _hard_plan_from_document(record, row_doc, context["device"])
        context["model"].set_native_interaction_policy(base._FixedPlanPolicy((plan,)))
        with torch.inference_mode():
            candidate_prepared, prepare_ms = base._timed(
                lambda current_record=record: context["model"].prepare_case(
                    current_record.verification_batch
                ), context["device"]
            )
        context["model"].set_native_interaction_policy(None)
        metrics = _native_candidate_reference_grid(
            context["model"],
            candidate_prepared,
            record.full_prepared,
            record.case,
            context["normalizer"],
            device=context["device"],
        )
        layout_result = {
            "row_index": row_index,
            "layout_index": layout,
            "direction_deg": float(record.case.wind_direction_deg),
            "fitted_hard_plan": row_doc["fitted_hard_plan"],
            "hard_mask_sha256": hashlib.sha256(json.dumps(
                row_doc["fitted_hard_plan"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            "prepare_ms": float(prepare_ms),
            "native_metrics": metrics,
        }
        layout_path = output_dir / f"layout_{layout:04d}_native_reference.json"
        _atomic_json_write({"provenance": provenance, "result": layout_result}, layout_path)
        layout_results.append(layout_result)
        progress["completed_layout_artifacts"] = [
            *progress["completed_layout_artifacts"], str(layout_path)
        ]
        _atomic_json_write(progress, progress_path)
    statuses = [item["native_metrics"]["train_only_physical_reference_guard"]["status"] for item in layout_results]
    reference_status = (
        "unknown" if any(value == "unknown" for value in statuses)
        else "passed" if all(value == "passed" for value in statuses)
        else "failed"
    )
    result = {
        "status": "complete",
        "stage": "full_grid_native_reference",
        "workflow": "native_cover_legacy_g0_checkpoint_only_evaluation",
        **provenance,
        "learning_eligible": True,
        "reference_sufficient": reference_status,
        "deployment_eligible": False,
        "deployment_reason": "isolated complete-workload deployment speed remains unmeasured",
        "optimizer_calls_in_this_stage": 0,
        "full_grid_native_decode_passes": 2 * len(layout_results),
        "native_layout_results": layout_results,
        "source_disjoint_report": str(source_report_path),
    }
    progress.update({
        "status": "complete",
        "summary_path": str(output_dir / "full_grid_evaluation_report.json"),
        "reference_sufficient": reference_status,
    })
    _atomic_json_write(progress, progress_path)
    _atomic_json_write(result, output_dir / "full_grid_evaluation_report.json")
    return result


def run_saved_organizer_evaluation(
    *,
    checkpoint_path: str | Path,
    volume_path: str | Path,
    compact_path: str | Path,
    oracle_dir: str | Path,
    organizer_checkpoint_path: str | Path,
    device: str | torch.device,
    output_dir: str | Path,
    stage: str = "disjoint",
    disjoint_results_dir: str | Path | None = None,
    query_count: int = 1024,
    seed: int = FIT_SEED,
    native_validation_layouts: int = 2,
    resume: bool = False,
) -> dict[str, Any]:
    """Evaluate u100 without optimizer steps; persist each complete result atomically."""

    if query_count != 1024:
        raise ValueError("checkpoint-only G0 uses the frozen disjoint Q1024 probes")
    if stage not in {"disjoint", "full-grid"}:
        raise ValueError("checkpoint-only stage must be 'disjoint' or 'full-grid'")
    if stage == "full-grid" and disjoint_results_dir is None:
        raise ValueError("full-grid evaluation requires --disjoint-results-dir from the Q1024 stage")
    destination = Path(output_dir).expanduser().resolve()
    if "generated" not in destination.parts:
        raise ValueError("checkpoint-only evaluation outputs must stay under diagnostics/generated")
    rows: list[int] | None = None
    if stage == "full-grid":
        if not 1 <= native_validation_layouts <= 2:
            raise ValueError("G0 full native validation is capped at two frozen training layouts")
        # Full-grid review is intentionally limited to the same first one or
        # two labeled training layouts used by the frozen G0 diagnostic.
        _directory, oracle_report, _manifest, documents = _load_fit_artifacts(oracle_dir)
        selected_rows: list[int] = []
        selected_layouts: set[int] = set()
        for item in oracle_report.get("rows", []):
            layout_index = int(item["layout_index"])
            if layout_index in selected_layouts:
                continue
            selected_rows.append(int(item["row_index"]))
            selected_layouts.add(layout_index)
            if len(selected_rows) == native_validation_layouts:
                break
        rows = selected_rows
        if len(documents) < 1:
            raise ValueError("full-grid review has no verified legacy G0 plan documents")
    context = _load_saved_organizer_context(
        checkpoint_path=checkpoint_path,
        volume_path=volume_path,
        compact_path=compact_path,
        oracle_dir=oracle_dir,
        organizer_checkpoint_path=organizer_checkpoint_path,
        device=device,
        output_dir=destination,
        query_count=query_count,
        seed=seed,
        row_ids=rows,
    )
    if stage == "disjoint":
        return _evaluate_saved_disjoint_rows(
            context,
            output_dir=destination,
            resume=resume,
        )
    assert disjoint_results_dir is not None
    return _evaluate_saved_full_grid(
        context,
        output_dir=destination,
        disjoint_results_dir=disjoint_results_dir,
        native_validation_layouts=native_validation_layouts,
    )


def run_typed_checkpoint_only_evaluation(
    *,
    checkpoint_path: str | Path,
    volume_path: str | Path,
    compact_path: str | Path,
    typed_search_dir: str | Path,
    organizer_state_path: str | Path,
    device: str | torch.device,
    output_dir: str | Path,
    stage: str = "disjoint",
    disjoint_results_dir: str | Path | None = None,
    query_count: int = 1024,
    seed: int = FIT_SEED,
    native_validation_layouts: int = 2,
    include_development_native_grid: bool = False,
) -> dict[str, Any]:
    """Evaluate a saved typed organizer without optimizer calls.

    Disjoint Q1024 mode evaluates every frozen training and development
    direction. Full-grid mode is separate and requires the matching completed
    Q1024 report with a passing protected teacher gate for each selected train
    layout.
    """

    if query_count != 1024:
        raise ValueError("typed checkpoint evaluation is frozen to disjoint Q1024 probes")
    if stage not in {"disjoint", "full-grid"}:
        raise ValueError("typed checkpoint evaluation stage must be 'disjoint' or 'full-grid'")
    if stage == "full-grid" and disjoint_results_dir is None:
        raise ValueError("typed full-grid evaluation requires a completed typed Q1024 report")
    if not 1 <= native_validation_layouts <= 2:
        raise ValueError("typed full-grid evaluation is capped at two total frozen layouts")
    if include_development_native_grid and stage != "full-grid":
        raise ValueError("development native-grid inclusion is available only in typed full-grid mode")

    destination = Path(output_dir).expanduser().resolve()
    if "generated" not in destination.parts:
        raise ValueError("typed checkpoint evaluation outputs must stay under diagnostics/generated")
    destination.mkdir(parents=True, exist_ok=True)
    progress_path = destination / "typed_checkpoint_evaluation_progress.json"
    report_path = destination / (
        "typed_checkpoint_disjoint_evaluation_report.json"
        if stage == "disjoint"
        else "typed_checkpoint_full_grid_evaluation_report.json"
    )
    if progress_path.exists() or report_path.exists():
        raise FileExistsError("typed checkpoint evaluation output already exists; choose a fresh directory")

    search_dir = Path(typed_search_dir).expanduser().resolve()
    panel_manifest_path = search_dir / "panel_manifest.json"
    search_report_path = search_dir / "typed_search_report.json"
    if not panel_manifest_path.is_file() or not search_report_path.is_file():
        raise FileNotFoundError("typed checkpoint evaluation requires complete G2 artifacts")
    search_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    anchor_variant = str(search_manifest.get("anchor_measure_variant", ""))
    if anchor_variant not in {"raw", "role_balanced"}:
        raise ValueError("typed search panel manifest has no recognized anchor measure")

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    checkpoint_hash = base._checkpoint_sha256(checkpoint)
    typed_search_report_path = search_dir / "typed_search_report.json"
    typed_search_report_hash = base._checkpoint_sha256(typed_search_report_path)
    state_path = Path(organizer_state_path).expanduser().resolve()
    organizer_state_hash = base._checkpoint_sha256(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not isinstance(state.get("state_dict"), dict):
        raise TypeError("typed checkpoint lacks an organizer state_dict")
    if state.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("typed organizer state belongs to different Run2103 checkpoint bytes")
    if state.get("typed_search_report_sha256") != typed_search_report_hash:
        raise ValueError("typed organizer state belongs to different G2 search evidence")
    if state.get("plan_schema") != "honf-mechanism-plan-v1":
        raise ValueError("typed organizer checkpoint has an unsupported plan schema")
    if state.get("typed_organizer_input_state_schema") != TYPED_ORGANIZER_INPUT_STATE_SCHEMA:
        raise ValueError("typed organizer checkpoint has an unsupported input-state schema")
    if state.get("typed_organizer_quadrature_measure_schema") != TYPED_ORGANIZER_QUADRATURE_MEASURE_SCHEMA:
        raise ValueError("typed organizer checkpoint has an unsupported quadrature measure schema")

    (
        checkpoint,
        payload,
        checkpoint_hash,
        view,
        _split,
        layouts,
        _manifest_path,
        current_panel_manifest,
    ) = base._freeze_training_panel(
        checkpoint_path=checkpoint,
        volume_path=volume_path,
        compact_path=compact_path,
        output_dir=destination,
        layout_count=12,
        active_layouts=base.TYPED_TRAIN_LAYOUT_COUNT,
        anchor_measure_variant=anchor_variant,
    )
    frozen_split = base.freeze_organizer_layout_split(layouts)
    search_report, search_manifest, _documents = _typed_fit_artifacts(
        search_dir,
        checkpoint_sha256=checkpoint_hash,
        layouts=layouts,
    )
    if frozen_split["split_sha256"] != state.get("organizer_split_sha256"):
        raise ValueError("typed organizer checkpoint differs from the frozen geometry-only split")
    if current_panel_manifest.get("organizer_split_frozen_before_new_outcomes", {}).get(
        "split_sha256"
    ) != frozen_split["split_sha256"]:
        raise ValueError("typed evaluation panel changed the pre-outcome geometry split")
    if search_report.get("frozen_split", {}).get("split_sha256") != frozen_split["split_sha256"]:
        raise ValueError("typed G2 report differs from the recreated frozen split")
    split_lock_path = Path(str(search_report.get("split_lock_path", ""))).expanduser().resolve()
    if not split_lock_path.is_file() or base._checkpoint_sha256(split_lock_path) != search_report.get(
        "split_lock_sha256"
    ):
        raise ValueError("typed G2 split lock is missing or has changed")

    training_rows = list(map(int, frozen_split["training_rows_direction_order"]))
    development_rows = list(map(int, frozen_split["development_rows_direction_order"]))
    if set(training_rows) & set(development_rows) or len(training_rows) != 24 or len(development_rows) != 12:
        raise RuntimeError("typed checkpoint evaluation requires the frozen 24/12 grouped direction split")
    q1024_rows_by_id: dict[int, dict[str, Any]] = {}
    if stage == "full-grid":
        assert disjoint_results_dir is not None
        source_report_path = (
            Path(disjoint_results_dir).expanduser().resolve()
            / "typed_checkpoint_disjoint_evaluation_report.json"
        )
        if not source_report_path.is_file():
            raise FileNotFoundError("typed full-grid mode requires the completed Q1024 checkpoint report")
        q1024_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        expected_provenance = {
            "checkpoint_sha256": checkpoint_hash,
            "organizer_state_sha256": organizer_state_hash,
            "typed_search_report_sha256": typed_search_report_hash,
            "organizer_split_sha256": frozen_split["split_sha256"],
        }
        for key, value in expected_provenance.items():
            if q1024_report.get(key) != value:
                raise ValueError(f"typed Q1024 report provenance mismatch for {key}")
        if q1024_report.get("status") != "complete":
            raise ValueError("typed full-grid mode requires complete checkpoint-only Q1024 evidence")
        if q1024_report.get("development_labels_used") is not False:
            raise ValueError("typed full-grid mode requires Q1024 evidence with no development labels")
        if list(map(int, q1024_report.get("training_rows", []))) != training_rows:
            raise ValueError("typed Q1024 report differs from the frozen training direction rows")
        if list(map(int, q1024_report.get("development_rows", []))) != development_rows:
            raise ValueError("typed Q1024 report differs from the frozen development direction rows")
        q1024_directory = Path(disjoint_results_dir).expanduser().resolve()
        for artifact in q1024_report.get("row_results", []):
            artifact_path = Path(str(artifact)).expanduser().resolve()
            if artifact_path.parent != q1024_directory:
                raise ValueError("typed Q1024 row artifacts must remain in the bound disjoint output directory")
            document = json.loads(artifact_path.read_text(encoding="utf-8"))
            row = int(document.get("row_index", -1))
            if artifact_path.name != f"typed_checkpoint_eval_row_{row:04d}.json":
                raise ValueError(f"typed Q1024 artifact name does not match row {row}")
            if document.get("artifact_path") != str(artifact_path):
                raise ValueError(f"typed Q1024 row {row} artifact path is not self-bound")
            for key, value in expected_provenance.items():
                if document.get(key) != value:
                    raise ValueError(f"typed Q1024 row {row} provenance mismatch for {key}")
            expected_partition = "training" if row in set(training_rows) else "development"
            if document.get("partition") != expected_partition:
                raise ValueError(f"typed Q1024 row {row} partition differs from the frozen split")
            if expected_partition == "development" and document.get("development_g2_labels_loaded") is not False:
                raise ValueError(f"typed Q1024 development row {row} loaded a G2 label")
            if row in q1024_rows_by_id:
                raise ValueError(f"typed Q1024 report contains duplicate row artifact {row}")
            q1024_rows_by_id[row] = document
        if set(q1024_rows_by_id) != set(training_rows + development_rows):
            raise ValueError("typed Q1024 row artifacts are not the complete frozen 24/12 direction panel")
        selected_rows, full_grid_selection = _typed_select_full_grid_rows(
            frozen_split,
            q1024_rows_by_id,
            native_validation_layouts=native_validation_layouts,
            include_development_layout=bool(include_development_native_grid),
        )
    else:
        selected_rows = training_rows + development_rows
        full_grid_selection = None

    normalizer = VelocityNormalizer.from_dict(dict(payload["normalization"]))
    target_device = torch.device(device)
    first_case = view.run(selected_rows[0])
    first_search, _first_verification = make_disjoint_native_probes(
        first_case, query_count=query_count, seed=seed
    )
    materialization_batch = base._model_batch(
        base._probe_batch(first_case, first_search, normalizer), target_device
    )
    model, _loaded = load_checkpoint(
        checkpoint, device=target_device, materialization_batch=materialization_batch
    )
    if model.architecture != "dense_pairwise_field":
        raise ValueError("typed checkpoint evaluation requires the intact frozen Run2103 Dense teacher")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    teacher_checkpoint_id = f"Run2103:e2475:{checkpoint_hash[:16]}"

    progress: dict[str, Any] = {
        "workflow": "windfarm_typed_organizer_checkpoint_only_evaluation",
        "status": "running",
        "stage": stage,
        "typed_organizer_input_provenance": _typed_organizer_input_provenance(),
        "checkpoint_sha256": checkpoint_hash,
        "organizer_state_path": str(state_path),
        "organizer_state_sha256": organizer_state_hash,
        "typed_search_report_sha256": typed_search_report_hash,
        "organizer_split_sha256": frozen_split["split_sha256"],
        "training_rows": training_rows,
        "development_rows": development_rows,
        "evaluation_rows": selected_rows,
        "development_labels_used": False,
        "optimizer_calls_in_this_stage": 0,
        "new_physical_solves": 0,
        "total_variant_forward_calls": 0,
        "total_complete_native_forward_calls": 1,
        "total_prepare_case_calls": 1,
        "total_decode_calls": 1,
        "completed_row_artifacts": [],
        "native_validation_layouts_cap": native_validation_layouts if stage == "full-grid" else 0,
        "full_grid_selection": full_grid_selection,
    }
    _atomic_json_write(progress, progress_path)

    training_row_set = set(training_rows)
    records: list[Any] = []
    for row in selected_rows:
        record, _timing = base._build_panel_case(
            model,
            view.run(row),
            query_count=query_count,
            seed=seed,
            teacher_checkpoint_id=teacher_checkpoint_id,
            device=target_device,
            anchor_measure_variant=anchor_variant,
        )
        records.append(record)
        progress["total_complete_native_forward_calls"] += 4
        progress["total_prepare_case_calls"] += 2
        progress["total_decode_calls"] += 4
        progress["completed_panel_rows"] = len(records)
        progress["current_row_index"] = int(row)
        _atomic_json_write(progress, progress_path)

    training_records: list[Any] = []
    fixed_support_masks: dict[str, torch.Tensor] | None = None
    fixed_support_report: dict[str, Any] | None = None
    if stage == "disjoint":
        for record in records:
            row = int(record.case.index)
            if row not in training_row_set:
                continue
            document = _documents.get(row)
            if document is None:
                raise ValueError(f"typed G6 training row {row} lacks its G2 supervision artifact")
            record.oracle_plan = _typed_plan_from_document(record, document, target_device)
            training_records.append(record)
        if [int(item.case.index) for item in training_records] != training_rows:
            raise RuntimeError("typed G6 fixed support must be built from all 24 train rows in frozen order")
        fixed_support_masks, fixed_support_report = _typed_population_fixed_support(training_records)
        progress["fixed_support_definition"] = fixed_support_report
        progress["development_g2_labels_loaded"] = False
        _atomic_json_write(progress, progress_path)

    first = records[0]
    encoded = first.encoded
    organizer = InputOnlyCoverOrganizer(
        state_dim=int(encoded.module_tokens.shape[-1]),
        module_feature_dim=int(encoded.module_features.shape[-1]),
        environment_feature_dim=0 if encoded.env_features is None else int(encoded.env_features.shape[-1]),
        hidden_dim=32,
        role_count=max(8, int(first.trees[0].universe.roles.max().item()) + 1),
        quadrature_invariant_source_measure=True,
    ).to(target_device)
    incompatible = organizer.load_state_dict(state["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("typed organizer checkpoint state was not loaded strictly")
    organizer.eval()

    if stage == "disjoint":
        completed_rows: list[dict[str, Any]] = []
        for record in records:
            row = int(record.case.index)
            role = "training" if row in training_row_set else "development"
            learned_plan = _typed_hard_plan(organizer, record)
            if not learned_plan.canonical_hash():
                raise RuntimeError("typed hard-plan canonical identity is empty")
            all_access_plan = MechanismPlan.full_access(
                record.trees[0],
                record.encoded.module_present[0],
                int(record.encoded.env_coords.shape[1]),
            )
            assert fixed_support_masks is not None and fixed_support_report is not None
            fixed_plan = _typed_root_permissions_plan(
                record,
                {
                    mechanism: support[: int(_typed_valid_source_mask(record.encoded, mechanism).numel())]
                    for mechanism, support in fixed_support_masks.items()
                },
            )
            direct_plan, direct_metadata = _typed_direct_pair_plan(
                record,
                learned_plan,
                record.verification_batch.query_xy[0],
            )
            variant_plans: dict[str, tuple[MechanismPlan, dict[str, Any], dict[str, Any] | None]] = {
                "all_access": (
                    all_access_plan,
                    {
                        "information_scope": "explicit native all-access reference",
                        "inherits_oracle_information": False,
                        "per_case_development_outputs_used_for_selection": False,
                    },
                    None,
                ),
                "learned_input_only": (
                    learned_plan,
                    {
                        "information_scope": "saved input-only organizer applied to case inputs",
                        "per_case_g2_plan_used_at_evaluation": False,
                        "development_outputs_used_for_selection": False,
                    },
                    None,
                ),
                "population_fixed_support": (
                    fixed_plan,
                    {
                        "information_scope": (
                            "one slot-indexed artificial source support derived from frozen train-row G2 labels; "
                            "slot IDs do not define permutation-invariant physical identities"
                        ),
                        "support_indexing": "padded module/environment source slot indices",
                        "permutation_invariant_physical_support": False,
                        "inherits_oracle_information": True,
                        "development_rows_used_for_control_construction": False,
                    },
                    None,
                ),
                "ungrouped_direct_pair_matched_budget": (
                    direct_plan,
                    {
                        "information_scope": "learned exact pair budget plus physical geometry only",
                        "per_case_g2_plan_used_for_selection": False,
                        "development_outputs_used_for_selection": False,
                    },
                    direct_metadata,
                ),
            }
            oracle_plan: MechanismPlan | None = None
            if role == "training":
                oracle_plan = getattr(record, "oracle_plan", None)
                if not isinstance(oracle_plan, MechanismPlan):
                    raise ValueError(f"typed train row {row} has no per-case G2 plan")
                variant_plans["verified_oracle_hard"] = (
                    oracle_plan,
                    {
                        "information_scope": "per-case G2 train label selected and disjoint-verified before G5",
                        "inherits_oracle_information": True,
                        "g2_plan_hash": oracle_plan.canonical_hash(),
                        "g2_selected_disjoint_teacher_gate_passed": True,
                        "reuses_g2_disjoint_verification_probe": True,
                        "independent_reverification": False,
                        "evaluation_interpretation": "G2-label-reuse diagnostic on the exact already-used Q1024 verification probe",
                        "deployment_policy": False,
                    },
                    None,
                )
                collapsed_plan = _typed_collapsed_root_plan(record, oracle_plan)
                variant_plans["collapsed_root_source_union"] = (
                    collapsed_plan,
                    {
                        "information_scope": "per-case G2 plan's exact train-search native source union, collapsed to root",
                        "inherits_oracle_information": True,
                        "source_union_basis": "valid physical pairs on the G2 Q1024 search receivers and native MM/ME/EM axes",
                        "g2_plan_hash": oracle_plan.canonical_hash(),
                        "reuses_g2_disjoint_verification_probe": True,
                        "independent_reverification": False,
                        "evaluation_interpretation": "G2-label-reuse diagnostic on the exact already-used Q1024 verification probe",
                        "deployment_policy": False,
                    },
                    None,
                )
            evaluation_scope = (
                "in-sample training-row Q1024 diagnostic; probe reused from G2 disjoint verification"
                if role == "training"
                else "held-out development-layout Q1024 diagnostic; no G2 labels or search"
            )
            variant_plans = {
                variant: (
                    control_plan,
                    {
                        **variant_provenance,
                        "evaluation_scope": evaluation_scope,
                        "in_sample_organizer_diagnostic": role == "training",
                        "reuses_g2_disjoint_verification_probe": role == "training",
                        "independent_holdout": role == "development",
                    },
                    direct_info,
                )
                for variant, (control_plan, variant_provenance, direct_info) in variant_plans.items()
            }
            row_doc: dict[str, Any] = {
                "status": "running",
                "stage": "disjoint_q1024",
                "row_index": row,
                "layout_index": int(record.case.layout_index),
                "wind_direction_deg": float(record.case.wind_direction_deg),
                "partition": role,
                "probe_split": (
                    "same_g2_disjoint_verification_probe_q1024_reused_for_comparison"
                    if role == "training"
                    else "held_out_development_layout_q1024_probe"
                ),
                "evaluation_scope": evaluation_scope,
                "reuses_g2_disjoint_verification_probe": role == "training",
                "independent_holdout": role == "development",
                "checkpoint_sha256": checkpoint_hash,
                "organizer_state_path": str(state_path),
                "organizer_state_sha256": organizer_state_hash,
                "typed_search_report_sha256": typed_search_report_hash,
                "organizer_split_sha256": frozen_split["split_sha256"],
                "plan_schema": "honf-mechanism-plan-v1",
                "typed_organizer_input_provenance": _typed_organizer_input_provenance(),
                "plan_formation_timing_scope": (
                    "the per-variant replay time below excludes model input encoding, receiver-tree construction, "
                    "typed organizer scoring, and hard-plan materialization"
                ),
                "learned_hard_plan_hash": learned_plan.canonical_hash(),
                "optimizer_calls_in_this_stage": 0,
                "new_physical_solves": 0,
                "variants": {},
            }
            if role == "training":
                labels = _typed_label_tensors(_documents[row], target_device)
                row_doc["g2_source_plan_hash"] = oracle_plan.canonical_hash() if oracle_plan else None
                row_doc["supervised_hard_label_metrics"] = _typed_hard_label_metrics(learned_plan, labels)
            else:
                row_doc["development_g2_labels_loaded"] = False
            row_path = destination / f"typed_checkpoint_eval_row_{row:04d}.json"
            row_doc["artifact_path"] = str(row_path)
            _atomic_json_write(row_doc, row_path)
            if str(row_path) not in progress["completed_row_artifacts"]:
                progress["completed_row_artifacts"].append(str(row_path))
            for variant in TYPED_G6_VARIANTS:
                if variant not in variant_plans:
                    row_doc["variants"][variant] = {
                        "status": "unavailable",
                        "reason": "development rows have no per-case G2 labels; no oracle search is run on development",
                        "development_g2_labels_loaded": False,
                    }
                    _atomic_json_write(row_doc, row_path)
                    progress["completed_variant_artifacts"] = [
                        *progress.get("completed_variant_artifacts", []), f"{row}:{variant}:unavailable"
                    ]
                    progress["current_row_index"] = row
                    _atomic_json_write(progress, progress_path)
                    continue
                control_plan, variant_provenance, direct_info = variant_plans[variant]
                result_doc = _typed_evaluate_g6_variant(
                    model,
                    record,
                    control_plan,
                    name=variant,
                    teacher_checkpoint_id=teacher_checkpoint_id,
                    normalizer=normalizer,
                    device=target_device,
                    provenance=variant_provenance,
                    direct_pair_metadata=direct_info,
                )
                row_doc["variants"][variant] = result_doc
                progress["total_variant_forward_calls"] += 1
                progress["total_complete_native_forward_calls"] += 1
                progress["total_prepare_case_calls"] += 1
                progress["total_decode_calls"] += 1
                progress["completed_variant_artifacts"] = [
                    *progress.get("completed_variant_artifacts", []), f"{row}:{variant}:complete"
                ]
                progress["current_row_index"] = row
                _atomic_json_write(row_doc, row_path)
                _atomic_json_write(progress, progress_path)

            learned_result = row_doc["variants"]["learned_input_only"]
            row_doc.update({
                "status": "complete",
                "plan_schema": "honf-mechanism-plan-v1",
                "learned_hard_plan_hash": learned_plan.canonical_hash(),
                "source_sets_by_mechanism": learned_result["source_sets_by_mechanism"],
                "logical_work": learned_result["logical_work"],
                "teacher_preservation": learned_result["teacher_preservation"],
                "native_reference_rmse_mps_by_protected_role": learned_result[
                    "native_reference_rmse_mps_by_protected_role"
                ],
                "role_metrics": learned_result["role_metrics"],
                "actual_synchronized_prepare_ms": learned_result["actual_synchronized_prepare_ms"],
                "actual_synchronized_decode_ms": learned_result["actual_synchronized_decode_ms"],
                "actual_synchronized_complete_ms": learned_result["actual_synchronized_complete_ms"],
                "g6_variant_names": list(TYPED_G6_VARIANTS),
                "development_g2_labels_used": False,
            })
            _atomic_json_write(row_doc, row_path)
            completed_rows.append(row_doc)
            progress["current_row_index"] = row
            _atomic_json_write(progress, progress_path)

        training_results = [item for item in completed_rows if item["partition"] == "training"]
        development_results = [item for item in completed_rows if item["partition"] == "development"]
        g6_summary = _typed_g6_comparison_summary(completed_rows)
        result = {
            "status": "complete",
            "workflow": "windfarm_typed_organizer_checkpoint_only_evaluation",
            "stage": "disjoint_q1024",
            "typed_organizer_input_provenance": _typed_organizer_input_provenance(),
            "plan_formation_timing_scope": (
                "variant timings measure only native replay of already-formed plans; no deployment-speed claim "
                "includes input encoding, receiver-tree construction, organizer scoring, or hard-plan materialization"
            ),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "organizer_state_path": str(state_path),
            "organizer_state_sha256": organizer_state_hash,
            "typed_search_report_sha256": typed_search_report_hash,
            "organizer_split_sha256": frozen_split["split_sha256"],
            "plan_schema": "honf-mechanism-plan-v1",
            "training_layout_indices": list(frozen_split["training_layout_indices"]),
            "development_layout_indices": list(frozen_split["development_layout_indices"]),
            "training_rows": training_rows,
            "development_rows": development_rows,
            "development_labels_used": False,
            "development_labels_used_for_search_or_fit": [],
            "development_labels_used_for_evaluation": False,
            "development_g2_labels_used_for_controls": False,
            "development_native_reference_used_for_fidelity_metrics": True,
            "optimizer_calls_in_this_stage": 0,
            "new_physical_solves": 0,
            "training_gate_pass_count": sum(bool(item["teacher_preservation"]["passed"]) for item in training_results),
            "training_gate_row_count": len(training_results),
            "development_gate_pass_count": sum(bool(item["teacher_preservation"]["passed"]) for item in development_results),
            "development_gate_row_count": len(development_results),
            "training_gate_all_passed": bool(training_results) and all(
                bool(item["teacher_preservation"]["passed"]) for item in training_results
            ),
            "development_gate_all_passed": bool(development_results) and all(
                bool(item["teacher_preservation"]["passed"]) for item in development_results
            ),
            "reference_sufficient": "unknown",
            "measured_speed_eligible": False,
            "deployment_eligible": False,
            "optimizer_update_count_from_source_state": int(state.get("updates_completed", -1)),
            "g6_control_variants": list(TYPED_G6_VARIANTS),
            "g6_control_comparison": g6_summary,
            "same_module_count_holdout_comparison": _typed_same_module_count_holdout_summary(
                frozen_split
            ),
            "population_fixed_support_definition": fixed_support_report,
            "direct_pair_control_definition": g6_summary["direct_pair_rule"],
            "per_case_oracle_and_collapsed_root_scope": "training rows only; development variants are explicitly unavailable",
            "training_comparison_scope": (
                "all training-row variants reuse the same Q1024 probe used for G2 disjoint verification and are "
                "in-sample organizer diagnostics; verified_oracle_hard and collapsed_root_source_union additionally "
                "reuse G2 labels, so their scores are not independent reverification. Development variants use "
                "held-out layouts with no G2 labels or search"
            ),
            "forward_call_accounting": {
                "materialization_complete_forward_calls": 1,
                "panel_reference_complete_forward_calls": 4 * len(records),
                "complete_variant_forward_calls": int(progress["total_variant_forward_calls"]),
                "complete_variant_forward_calls_by_name": {
                    variant: sum(
                        1
                        for row_doc in completed_rows
                        if row_doc.get("variants", {}).get(variant, {}).get("status") == "complete"
                    )
                    for variant in TYPED_G6_VARIANTS
                },
                "total_complete_native_forward_calls": int(progress["total_complete_native_forward_calls"]),
                "total_prepare_case_calls": int(progress["total_prepare_case_calls"]),
                "total_decode_calls": int(progress["total_decode_calls"]),
                "optimizer_updates_are_not_forward_calls": True,
            },
            "row_results": progress["completed_row_artifacts"],
        }
        progress.update({
            "status": "complete",
            "summary_path": str(report_path),
            "training_gate_all_passed": result["training_gate_all_passed"],
            "development_gate_all_passed": result["development_gate_all_passed"],
        })
        _atomic_json_write(progress, progress_path)
        _atomic_json_write(result, report_path)
        return result

    # The separate native-grid mode is admitted only by the matching Q1024
    # report's hard teacher gate. An optional development row is predeclared
    # by geometry and consumes one of the same two-layout total cap.
    grid_results: list[dict[str, Any]] = []
    for record in records:
        row = int(record.case.index)
        plan = _typed_hard_plan(organizer, record)
        evidence = q1024_rows_by_id[row]
        if evidence.get("learned_hard_plan_hash") != plan.canonical_hash():
            raise ValueError(f"typed full-grid plan hash differs from the saved Q1024 row {row}")
        partition = str(evidence.get("partition"))
        model.set_native_interaction_policy(None)
        with torch.inference_mode():
            candidate_prepared, _encoded, context = base._prepare_typed_plan(
                model, record.search_batch, plan, device=target_device
            )
        progress["total_prepare_case_calls"] += 1
        metrics = _native_candidate_reference_grid(
            model,
            candidate_prepared,
            record.full_prepared,
            record.case,
            normalizer,
            device=target_device,
            candidate_is_core_prepared=True,
            candidate_interaction_context=context,
        )
        chunks = math.ceil(int(record.case.run.cell_count) / 8192)
        complete_decodes = 2 * chunks
        progress["total_complete_native_forward_calls"] += complete_decodes
        progress["total_decode_calls"] += complete_decodes
        grid_doc = {
            "status": "complete",
            "row_index": row,
            "layout_index": int(record.case.layout_index),
            "partition": partition,
            "wind_direction_deg": float(record.case.wind_direction_deg),
            "checkpoint_sha256": checkpoint_hash,
            "organizer_state_sha256": organizer_state_hash,
            "typed_search_report_sha256": typed_search_report_hash,
            "organizer_split_sha256": frozen_split["split_sha256"],
            "learned_hard_plan_hash": plan.canonical_hash(),
            "source_q1024_row_artifact": str(
                evidence.get("artifact_path", "")
            ),
            "q1024_teacher_gate_passed": bool(
                evidence.get("teacher_preservation", {}).get("passed")
            ),
            "development_g2_labels_loaded": False,
            "development_outputs_used_for_plan_selection": False if partition == "development" else None,
            "optimizer_calls_in_this_stage": 0,
            "new_physical_solves": 0,
            "native_decode_call_count": complete_decodes,
            "native_metrics": metrics,
        }
        grid_path = destination / f"typed_checkpoint_native_grid_row_{row:04d}.json"
        _atomic_json_write(grid_doc, grid_path)
        grid_results.append(grid_doc)
        progress["completed_row_artifacts"] = [
            *progress["completed_row_artifacts"], str(grid_path)
        ]
        progress["current_row_index"] = row
        _atomic_json_write(progress, progress_path)

    statuses = [
        item["native_metrics"]["train_only_physical_reference_guard"]["status"]
        for item in grid_results
    ]
    reference_status = (
        "unknown" if any(status == "unknown" for status in statuses)
        else "passed" if all(status == "passed" for status in statuses)
        else "failed"
    )
    result = {
        "status": "complete",
        "workflow": "windfarm_typed_organizer_checkpoint_only_evaluation",
        "stage": "full_grid_native_reference",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "organizer_state_path": str(state_path),
        "organizer_state_sha256": organizer_state_hash,
        "typed_search_report_sha256": typed_search_report_hash,
        "organizer_split_sha256": frozen_split["split_sha256"],
        "source_q1024_report": str(
            Path(disjoint_results_dir).expanduser().resolve()
            / "typed_checkpoint_disjoint_evaluation_report.json"
        ),
        "optimizer_calls_in_this_stage": 0,
        "optimizer_update_count_from_source_state": int(state.get("updates_completed", -1)),
        "new_physical_solves": 0,
        "native_validation_layouts_cap": native_validation_layouts,
        "full_grid_selection": full_grid_selection,
        "development_native_reference_included": any(
            item.get("partition") == "development" for item in grid_results
        ),
        "development_g2_labels_loaded": False,
        "development_outputs_used_for_plan_selection": False,
        "reference_sufficient": reference_status,
        "measured_speed_eligible": False,
        "deployment_eligible": False,
        "native_grid_results": grid_results,
        "forward_call_accounting": {
            "materialization_complete_forward_calls": 1,
            "panel_reference_complete_forward_calls": 4 * len(records),
            "native_grid_complete_forward_calls": sum(item["native_decode_call_count"] for item in grid_results),
            "total_complete_native_forward_calls": int(progress["total_complete_native_forward_calls"]),
            "total_prepare_case_calls": int(progress["total_prepare_case_calls"]),
            "total_decode_calls": int(progress["total_decode_calls"]),
            "optimizer_updates_are_not_forward_calls": True,
        },
    }
    progress.update({
        "status": "complete",
        "summary_path": str(report_path),
        "reference_sufficient": reference_status,
    })
    _atomic_json_write(progress, progress_path)
    _atomic_json_write(result, report_path)
    return result


__all__ = [
    "run_organizer_fit",
    "run_saved_organizer_evaluation",
    "run_typed_checkpoint_only_evaluation",
    "write_fit_target_assessment",
]
