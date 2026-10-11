"""Execution contract for a fresh full-population P-family follow-up.

This module is deliberately side-effect free. It validates identity and
checkpoint metadata only; it does not load datasets, fit transforms or
calibration, construct a model, or advance an optimizer.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

PROTOCOL = "interaction_preserving_full_train_followup1000_v1"
SCHEDULE_HORIZON = 5000
APPROVED_STOP = 1000
APPROVED_REVIEW_STOPS = (100, 500, 1000)
APPROVED_RESUME_AGES = (100, 500)
PROTOCOL_GO_FIELD = "root_protocol_go"
FULL_SCHEDULE = {
    "peak_lr": 3.0e-4,
    "warmup_start_lr": 3.0e-5,
    "warmup_epochs": 20,
    "hold_through_epoch": 1000,
    "final_lr": 3.0e-6,
}
MODES = ("P", "P-G", "P-H")
THERMAL_FLOW_READOUT_LAW = "native_curl_cell_centred_v1"
TRAIN_RESPONSE_FAMILIES = ("0001", "0318", "0333", "0348")
RUN_IDS = {
    "thermal": {"P": "T4201", "P-G": "T4202", "P-H": "T4203"},
    "wind": {"P": "W2501", "P-G": "W2502", "P-H": "W2503"},
}
DATA = {
    "thermal": {
        "dataset_protocol": "thermal_original600_train_canonical89_validation_followup_v1",
        "train_count": 600,
        "train_sha256": "3e255541359ec6551863bbcd122591b782d075b4477145c63d4d5f4f7adb3b25",
        "validation_count": 89,
        "validation_sha256": "1c33b4cc5ebddb1720a6ba6adcd9ad88beb988302008c58623cf55b2fdea41ab",
        "excluded_validation_id": "0273",
        "excluded_validation_reason": "historical duplicate in original90 validation; preserve canonical89",
        "source_metadata_sha256": "1bba5ab5c0535fabab2f2434de33eccb67184ef3fd7d7881eb01f5088e15e211",
        "source_h5_sha256": "4224093c22a67af4adfecc8b21d53548e4263ec2254c230dc83c89526b36da05",
        "normalization_scope": "exact original600 TRAIN only",
        "validation_scope": "canonical89 exposed validation from original H5 split=test; not independent/unexposed TEST",
    },
    "wind": {
        "dataset_protocol": "wind_original420_train_fullVALID90_followup_v1",
        "train_count": 420,
        "train_sha256": "a2170bb15349462160679632fc074d7f685d7e5d17d09aeeab1797a824c8a282",
        "validation_count": 90,
        "validation_sha256": "ed7295dd2650687c599c13e29a6bae3492508ec8346517e39f4cacf60d0eff72",
        "test_count": 90,
        "test_sha256": "e11734b05e51691680fcae088ef07be8608951660eab9761d05dc29fe2fef998",
        "train_layout_count": 140,
        "validation_layout_count": 30,
        "test_layout_count": 30,
        "split_seed": 42,
        "normalization_scope": "exact original420 TRAIN rows only",
        "validation_scope": "fullVALID90 exposed validation; not TEST",
        "test_target_policy": "locked; no TEST target materialization or evaluation",
    },
}


def _case_ids_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for case_id in ids:
        digest.update(str(case_id).encode("utf-8") + b"\0")
    return digest.hexdigest()


def _indices_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(int(index).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def canonical_thermal_validation_ids(original_test_ids: Sequence[str]) -> tuple[str, ...]:
    """Reproduce the already-established canonical89 membership rule."""
    ids = tuple(map(str, original_test_ids))
    if len(ids) != 90 or len(set(ids)) != 90 or ids.count("0273") != 1:
        raise ValueError("Thermal original validation must be 90 unique IDs with one historical duplicate 0273")
    canonical = tuple(case_id for case_id in ids if case_id != "0273")
    if len(canonical) != 89:
        raise ValueError("Thermal canonical validation must contain exactly 89 IDs")
    return canonical


def validate_membership(task: str, train_ids: Sequence[Any], validation_ids: Sequence[Any],
                        test_ids: Sequence[Any] = ()) -> dict[str, Any]:
    """Validate caller-supplied metadata against the sealed full split.

    For Thermal IDs are canonical case strings. For Wind IDs are canonical
    native row indices. Callers supply already-derived memberships; this
    helper never opens a dataset or examines target values.
    """
    if task not in DATA:
        raise ValueError("task must be thermal or wind")
    spec = DATA[task]
    train = tuple(train_ids)
    valid = tuple(validation_ids)
    test = tuple(test_ids)
    if not train or not valid or len(set(train)) != len(train) or len(set(valid)) != len(valid):
        raise ValueError("TRAIN/validation memberships must be nonempty and unique")
    if set(train).intersection(valid) or set(train).intersection(test) or set(valid).intersection(test):
        raise ValueError("Full follow-up memberships must be disjoint")
    hasher = _case_ids_sha256 if task == "thermal" else _indices_sha256
    if len(train) != spec["train_count"] or hasher(train) != spec["train_sha256"]:
        raise ValueError("TRAIN membership differs from the sealed full-population protocol")
    if len(valid) != spec["validation_count"] or hasher(valid) != spec["validation_sha256"]:
        raise ValueError("validation membership differs from the sealed full-population protocol")
    if task == "thermal":
        if test:
            raise ValueError("Thermal source TEST is already exposed only as canonical89 validation; do not add TEST IDs")
        if spec["excluded_validation_id"] in valid:
            raise ValueError("Thermal canonical89 must exclude historical duplicate 0273")
    else:
        if len(test) != spec["test_count"] or hasher(test) != spec["test_sha256"]:
            raise ValueError("Wind locked TEST membership identity must be preserved without reading targets")
    return {
        "task": task,
        "dataset_protocol": spec["dataset_protocol"],
        "training_count": len(train),
        "training_membership_sha256": hasher(train),
        "validation_count": len(valid),
        "validation_membership_sha256": hasher(valid),
        "normalization_scope": spec["normalization_scope"],
        "test_target_policy": spec.get("test_target_policy", "not independently unexposed"),
    }


def validate_recipe(recipe: Mapping[str, Any], *, command: str, stop_after: int | None = None,
                    resume_checkpoint_payload: Mapping[str, Any] | None = None) -> None:
    """Validate the new bounded execution identity without changing legacy rules."""
    task, mode = recipe.get("task"), recipe.get("mode")
    if task not in DATA or mode not in MODES:
        raise ValueError("full-data follow-up is limited to Thermal/Wind P-family recipes")
    if recipe.get("execution_protocol") != PROTOCOL:
        raise ValueError("explicit full-data follow-up protocol is required")
    if recipe.get("formal_full") is not False:
        raise ValueError("follow-up identity must remain distinct from legacy manual formal_full")
    if recipe.get("run_id") != RUN_IDS[task][mode]:
        raise ValueError("candidate run ID is not the reserved full-data follow-up ID")
    if recipe.get("dataset_protocol") != DATA[task]["dataset_protocol"]:
        raise ValueError("dataset protocol differs from the separate full-data follow-up population")
    if recipe.get("initialization") != "fresh_all_trainable" or recipe.get("parent_checkpoint") is not None:
        raise ValueError("follow-up must use fresh initialization and no learned parent")
    if recipe.get("total_epochs") != SCHEDULE_HORIZON:
        raise ValueError("schedule horizon must remain 5000; do not shorten cosine decay to the approved stop")
    if recipe.get("approved_stop_after") != APPROVED_STOP:
        raise ValueError("the currently approved execution boundary is exactly epoch 1000")
    if recipe.get("optimizer_schedule") != FULL_SCHEDULE:
        raise ValueError("the full-run schedule must remain explicit and bound to its 5000-epoch horizon")
    if recipe.get("launch_policy") != "root_protocol_go" or recipe.get("manual_full_followup") is not True:
        raise ValueError("full-data follow-up must use the separate root protocol gate")
    if type(recipe.get(PROTOCOL_GO_FIELD)) is not bool:
        raise ValueError("root_protocol_go must be an explicit boolean in the candidate recipe")
    expected_prior = 0.0 if mode == "P" else 1.0
    if recipe.get("locality_prior_strength") != expected_prior:
        raise ValueError("P-family locality identity differs from the segmented controls")
    if command in ("calibrate-thermal", "start", "resume") and recipe.get(PROTOCOL_GO_FIELD) is not True:
        raise ValueError("this full-population action requires the root protocol GO field")
    if command == "start" and stop_after != 100:
        raise ValueError("a fresh full-population run must stop at epoch 100 for its first review")
    if command == "resume":
        validate_resume_checkpoint(recipe, resume_checkpoint_payload, stop_after=stop_after)
    if command == "branch":
        raise ValueError("full-population follow-up branches are forbidden; resume only its own latest checkpoint")
    if command not in ("prepare", "dry-run", "start", "resume", "calibrate-thermal"):
        raise ValueError("unsupported command for the candidate follow-up identity")
    if command == "calibrate-thermal" and (task != "thermal" or mode != "P"):
        raise ValueError("calibrate-thermal is reserved for the fresh full-TRAIN Thermal P recipe")
    if task == "thermal":
        if recipe.get("flow_readout_law") != THERMAL_FLOW_READOUT_LAW:
            raise ValueError("Thermal follow-up must retain the selected native-curl readout law")
        descriptor = recipe.get("calibration_receipt")
        if not isinstance(descriptor, Mapping) or descriptor.get("dataset_protocol") != DATA[task]["dataset_protocol"]:
            raise ValueError("Thermal recipe must require a receipt calibrated on this full TRAIN population")
        if not descriptor.get("required_payload_sha256"):
            raise ValueError("Thermal full-population calibration receipt must be digest-bound")
        if recipe.get("auxiliary_calibration_fallback") is not False:
            raise ValueError("quarter-data or implicit calibration fallback is forbidden")
        if command == "calibrate-thermal":
            if recipe.get("auxiliary_calibration") is not None:
                raise ValueError("full-population calibration requires an unresolved fresh-P receipt")
        elif command in ("start", "resume"):
            receipt = recipe.get("auxiliary_calibration")
            if not isinstance(receipt, Mapping):
                raise ValueError("training requires the bound full-TRAIN P calibration receipt")
            validate_calibration_receipt(receipt, task="thermal",
                                         expected_training_sha256=DATA[task]["train_sha256"])
    elif recipe.get("flow_readout_law") is not None:
        raise ValueError("Wind follow-up must retain its existing nonlinear velocity readout")


def validate_resume_checkpoint(recipe: Mapping[str, Any], payload: Mapping[str, Any] | None,
                               *, stop_after: int | None) -> None:
    """Allow only this new identity's staged latest-checkpoint continuation."""
    if not isinstance(payload, Mapping):
        raise TypeError("resume requires the current run's readable latest checkpoint metadata")
    source_epoch = payload.get("epoch")
    identity = payload.get("experiment_identity")
    source_recipe = identity.get("recipe") if isinstance(identity, Mapping) else None
    if type(source_epoch) is not int or source_epoch not in APPROVED_RESUME_AGES:
        raise ValueError("follow-up resumes are permitted only from the latest epoch 100 or 500 checkpoint")
    expected_next_stop = {100: 500, 500: 1000}[source_epoch]
    if stop_after != expected_next_stop:
        raise ValueError("resume must advance exactly 100-to-500 or 500-to-1000 for the next review")
    if not isinstance(source_recipe, Mapping):
        raise TypeError("resume checkpoint does not contain a sealed follow-up recipe")
    identity_fields = ("execution_protocol", "task", "mode", "run_id", "dataset_protocol",
                       "flow_readout_law", "approved_stop_after", "total_epochs", "optimizer_schedule",
                       "population_binding", "calibration_receipt")
    if any(source_recipe.get(name) != recipe.get(name) for name in identity_fields):
        raise ValueError("resume checkpoint belongs to another run, population, schedule, or calibration receipt")
    if source_recipe.get(PROTOCOL_GO_FIELD) is not True or source_recipe.get("parent_checkpoint") is not None:
        raise ValueError("resume source is not a fresh, root-approved follow-up lineage")


def assert_same_segmented_mechanism(candidate: Mapping[str, Any],
                                    segmented: Mapping[str, Any]) -> None:
    """Allow population/schedule identity to change, but freeze model/objective law."""
    mechanism_fields = (
        "task", "mode", "seed", "hidden", "message", "regional_anchors",
        "collective_width", "depth", "receiver_tile", "primary_queries", "microbatch_cases",
        "effective_cases", "max_sources", "environment_token_shape",
        "locality_prior_strength", "flow_readout_law", "model_contract",
        "engine_schedule",
        "optimizer_name", "optimizer_betas", "optimizer_eps", "gradient_clip_norm",
        "weight_decay",
    )
    drift = [name for name in mechanism_fields if candidate.get(name) != segmented.get(name)]
    left = json.loads(json.dumps(candidate.get("objective_contract"), sort_keys=True))
    right = json.loads(json.dumps(segmented.get("objective_contract"), sort_keys=True))
    if not isinstance(left, dict) or not isinstance(right, dict):
        drift.append("objective_contract")
    else:
        for obj in (left, right):
            obj.pop("primary_cohort", None)
            obj.pop("calibration_scope", None)
            addendum = obj.get("response_addendum")
            if isinstance(addendum, dict):
                # The response-family IDs/queries/weights stay fixed. Its
                # overlap with the expanded primary cohort necessarily changes.
                addendum.pop("primary_cohort_overlap", None)
        if left != right:
            drift.append("objective_contract")
    if candidate.get("optimizer_schedule") != FULL_SCHEDULE:
        drift.append("optimizer_schedule")
    if candidate.get("checkpoint_epochs") != list(range(100, 1001, 100)):
        drift.append("checkpoint_epochs")
    if drift:
        raise ValueError(f"follow-up changed a segmented mechanism/objective invariant: {sorted(set(drift))}")


def validate_calibration_receipt(receipt: Mapping[str, Any], *, task: str,
                                 expected_training_sha256: str) -> None:
    """Require full-population TRAIN-only provenance for a fresh Thermal P receipt."""
    if task != "thermal":
        raise ValueError("only Thermal has the TRAIN-only auxiliary calibration receipt")
    source_binding = receipt.get("source_binding")
    native_catalog = (source_binding.get("native_geometry_mask_catalog")
                      if isinstance(source_binding, Mapping) else None)
    response_catalog = (source_binding.get("response_geometry_mask_catalog")
                        if isinstance(source_binding, Mapping) else None)
    native_training_ids = (native_catalog.get("training_case_ids")
                           if isinstance(native_catalog, Mapping) else None)
    native_validation_ids = (native_catalog.get("validation_case_ids")
                             if isinstance(native_catalog, Mapping) else None)
    response_family_ids = (response_catalog.get("family_order")
                           if isinstance(response_catalog, Mapping) else None)
    if (receipt.get("calibration_mode") != "P"
            or receipt.get("dataset_protocol") != DATA[task]["dataset_protocol"]
            or receipt.get("flow_readout_law") != THERMAL_FLOW_READOUT_LAW
            or receipt.get("training_membership_sha256") != expected_training_sha256
            or receipt.get("primary_partition") != "original600 TRAIN only"
            or receipt.get("validation_values_read") is not False
            or receipt.get("validation_cases_materialized") != 0
            or receipt.get("optimizer_steps") != 0
            or receipt.get("development_response_families_loaded") is not False
            or not receipt.get("initial_model_state_sha256")
            or not receipt.get("normalization_stats_sha256")
            or receipt.get("primary_query_count") != 1024
            or receipt.get("operator_rows_per_case") != 128
            or receipt.get("response_surface_stride") != 4
            or receipt.get("response_material_queries_per_module") != 32
            or receipt.get("target_added_gradient_ratio") != 0.5
            or receipt.get("q_proxy_coefficient") != 0.05
            or receipt.get("response_family_ids") != list(TRAIN_RESPONSE_FAMILIES)
            or receipt.get("response_family_partition") != "original TRAIN response addendum; all four fixed families"
            or not isinstance(native_catalog, Mapping)
            or not native_catalog.get("catalog_sha256")
            or not isinstance(native_training_ids, list)
            or len(native_training_ids) != 600
            or _case_ids_sha256(native_training_ids) != expected_training_sha256
            or not isinstance(native_validation_ids, list)
            or len(native_validation_ids) != 89
            or _case_ids_sha256(native_validation_ids) != DATA[task]["validation_sha256"]
            or not isinstance(response_catalog, Mapping)
            or not response_catalog.get("catalog_sha256")):
        raise ValueError("Thermal calibration receipt is not a complete full-TRAIN-only fresh-P artifact")
    if response_family_ids != list(TRAIN_RESPONSE_FAMILIES):
        raise ValueError("Thermal calibration response-mask catalog differs from the four fixed TRAIN families")
