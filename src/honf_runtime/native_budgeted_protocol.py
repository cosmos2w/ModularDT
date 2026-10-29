"""Strict Round8 recipe and research-admission contracts.

Round8 deliberately has its own family and launch receipt.  The Round7
``native_retrieval_v1`` cost gate remains unchanged and is never consulted by
this protocol.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .source_read_protocol import validate_source_read_recipe

NATIVE_BUDGETED_FAMILY = "native_budgeted_dispatch_v2"
NATIVE_BUDGETED_WORKFLOW = "native_budgeted_dispatch_joint_v2"
NATIVE_BUDGETED_PROTOCOL = "learned_budgeted_native_hypergraph_v1"
NATIVE_BUDGETED_RECIPE_SCHEMA = "native_budgeted_dispatch_recipe_v2"
NATIVE_BUDGETED_CHILD_SCHEMA = "native_budgeted_dispatch_child_v2"
NATIVE_BUDGETED_ADMISSION_SCHEMA = "round8_research_admission_v1"
NATIVE_BUDGETED_BACKENDS = frozenset((
    "dense_hard_reference", "flat_selected_reference", "grouped_native",
))
NATIVE_BUDGETED_TREATMENTS = frozenset(("D-control", "G-budgeted-M"))

_RECIPE_FIELDS = frozenset({
    "schema", "family", "model_family", "workflow", "protocol", "dataset",
    "architecture", "treatment", "source_checkpoint", "source_checkpoint_sha256",
    "source_epoch", "source_recipe_sha256", "source_provider_identity_sha256",
    "base_recipe", "base_recipe_sha256", "research_admission", "native_backend",
    "environment_value_reader", "routing_initialization", "budget_objective",
    "budget_schedule", "rng_policy", "execution",
})
_BUDGET_SCHEDULE = {
    "schema": "budgeted_native_absolute_age_schedule_v1",
    "branch_boundary_epoch": 500,
    "near_open_epochs": [501, 700],
    "near_open_retention": [0.98, 0.85],
    "learning_epochs": [701, 1500],
    "learning_retention": [0.85, 0.60],
    "exact_work_marginal_temperature_through_e700": 1.0,
    "post700_work_relaxation_temperature": 0.5,
    "relaxation_temperature": [1.0, 0.5],
    "relaxation_anneal_epochs": [501, 700],
    "dense_router_every_macro_steps": 4,
    "noise_anneal_through_epoch": 1500,
    "deterministic_epochs": [1501, 2500],
    "deterministic_retention": 0.60,
    "dual_update_every_macro_steps": 10,
    "exploration": "none",
    "absolute_age": True,
}
_RNG_POLICY = {
    "schema": "budgeted_native_counter_rng_v1",
    "seed_coordinate": "original_D0_A2-D_seed",
    "sampler_namespace": "original_D0_sampling_dataset_id",
    "sampling_version": "case_epoch_v1",
    "fit_epoch_is_absolute": True,
    "sampling_arm": "A2-D",
    "query_id_is_rng_label_only": True,
    "membership_draw": "one_scene_sample_reused_across_query_chunks_and_auxiliaries",
    "membership_noise": "logistic_from_counter_uniform",
    "dispatch_noise": "gumbel_from_counter_uniform",
    "counter_seed_tuple": ["routing_seed", "absolute_epoch", "update_index", "case_key", "route_type"],
    "counter_seed_digest": "sha256_repr_tuple_first_8_bytes_little_endian_mod_2147483647",
    "counter_integer_prime": 2147483647,
    "counter_integer_multipliers": [48271, 69621, 16807],
    "counter_integer_xor_shifts": [13, 11],
    "counter_uniform_clamp": [1e-7, 0.9999999],
    "native_ids_and_query_ids": "labels_for_correspondence_and_rng_only_never_scores",
    "global_rng_consumption": False,
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_path(value: Any, field: str) -> str:
    if (not isinstance(value, str) or not Path(value).is_absolute()
            or str(Path(value).resolve()) != value):
        raise ValueError(f"{field} must be an absolute canonical path.")
    return value


def _source_contract(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    from .compat import load_trusted_checkpoint

    source_path = Path(path).expanduser().resolve(strict=True)
    payload = load_trusted_checkpoint(source_path, map_location="cpu")
    identity = payload.get("experiment_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Round8 source checkpoint has no sealed experiment identity.")
    recipe, provider = identity.get("recipe"), identity.get("provider_identity")
    if (payload.get("workflow") != "content_hypergraph_joint_v2"
            or identity.get("workflow") != "content_hypergraph_joint_v2"
            or identity.get("model_family") != "honf_content_hypergraph"
            or not isinstance(recipe, Mapping) or not isinstance(provider, Mapping)
            or payload.get("arm") != "A2-D" or payload.get("epoch") != 500):
        raise ValueError("Round8 must branch from the exact segmented D0 A2-D e500 checkpoint.")
    base_recipe = validate_source_read_recipe(recipe)
    if (base_recipe.get("protocol") != "source_read_development_v1"
            or base_recipe.get("source_reader_variant") != "D0"
            or base_recipe.get("architecture") != "A2-D"
            or base_recipe.get("dataset") not in ("thermal", "wind")):
        raise ValueError("Round8 source must retain the same dataset's segmented D0 development recipe.")
    if (not isinstance(payload.get("model_state_dict"), Mapping)
            or not isinstance(payload.get("optimizer_state_by_name"), Mapping)
            or not payload.get("optimizer_group_names")
            or not payload.get("optimizer_schedule_contract")):
        raise ValueError("Round8 source checkpoint is missing model or named AdamW continuation state.")
    return payload, copy.deepcopy(dict(identity)), file_sha256(source_path)


def validate_round8_research_admission(
    value: Mapping[str, Any], *, source_checkpoint: str | Path | None = None,
    source_checkpoint_sha256: str | None = None, dataset: str | None = None,
    native_backend: str | None = None, verify_files: bool = False,
) -> dict[str, Any]:
    """Validate correctness/resource authorization without a speed requirement."""
    if not isinstance(value, Mapping):
        raise TypeError("Round8 research admission must be a JSON object.")
    receipt = copy.deepcopy(dict(value))
    expected = {
        "schema", "status", "dataset", "source_checkpoint", "source_checkpoint_sha256",
        "native_backend", "evidence", "resource_reservation", "startup_speed",
    }
    if set(receipt) != expected or receipt.get("schema") != NATIVE_BUDGETED_ADMISSION_SCHEMA:
        raise ValueError("Round8 admission receipt has an unsupported schema or field inventory.")
    if receipt.get("status") != "admitted" or receipt.get("dataset") not in ("thermal", "wind"):
        raise ValueError("Round8 scientific fit requires an explicitly admitted research receipt.")
    _canonical_path(receipt.get("source_checkpoint"), "admission.source_checkpoint")
    for key in ("source_checkpoint_sha256",):
        digest = receipt.get(key)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"Admission {key} must be a lowercase SHA256 digest.")
    if receipt.get("native_backend") not in NATIVE_BUDGETED_BACKENDS:
        raise ValueError("Round8 admission must name one registered binary native backend.")
    evidence = receipt.get("evidence")
    required_evidence = {
        "all_open_baseline_recovery", "dense_binary_reference", "useful_router_gradient",
        "target_heat_isolation", "real_optimizer_memory", "reserved_resources",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required_evidence:
        raise ValueError("Round8 admission must contain the exact correctness, gradient and resource evidence set.")
    for name in required_evidence:
        row = evidence[name]
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            raise ValueError(f"Round8 research admission lacks a passing {name} check.")
    memory = evidence["real_optimizer_memory"]
    if (memory.get("device") != "cuda" or type(memory.get("peak_allocated_bytes")) is not int
            or memory["peak_allocated_bytes"] <= 0 or type(memory.get("optimizer_updates")) is not int
            or memory["optimizer_updates"] < 1
            or memory.get("gpu_uuid") != "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39"):
        raise ValueError("Round8 admission requires a measured real-objective optimizer-memory update.")
    reservation = receipt.get("resource_reservation")
    if not isinstance(reservation, Mapping) or set(reservation) != {"path", "sha256"}:
        raise ValueError("Round8 admission must pin its resource-reservation receipt.")
    reservation_path = _canonical_path(reservation.get("path"), "resource_reservation.path")
    digest = reservation.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Round8 resource-reservation SHA256 is malformed.")
    reserved = evidence["reserved_resources"]
    reserved_fields = {
        "passed", "resource_reservation_path", "resource_reservation_sha256", "gpu_uuid", "gpu2_only",
        "minimum_free_gpu_bytes", "minimum_free_fraction", "estimated_associated_seconds",
        "estimated_elapsed_seconds", "estimated_persistent_bytes", "estimated_temporary_bytes",
    }
    if set(reserved) != reserved_fields:
        raise ValueError("Round8 reserved-resources evidence has an unsupported field inventory.")
    if (reserved.get("resource_reservation_path") != reservation_path
            or reserved.get("resource_reservation_sha256") != digest
            or reserved.get("gpu_uuid") != "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39"
            or reserved.get("gpu2_only") is not True
            or type(reserved.get("minimum_free_gpu_bytes")) is not int
            or reserved["minimum_free_gpu_bytes"] != 8 * 1024**3
            or type(reserved.get("minimum_free_fraction")) not in (int, float)
            or not math.isclose(float(reserved["minimum_free_fraction"]), 0.20, rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError("Round8 resource evidence differs from the pinned reservation or physical GPU2 floor.")
    for field in ("estimated_associated_seconds", "estimated_elapsed_seconds",
                  "estimated_persistent_bytes", "estimated_temporary_bytes"):
        if type(reserved.get(field)) is not int or reserved[field] <= 0:
            raise ValueError(f"Round8 resource evidence requires a positive {field} estimate.")
    if reserved["estimated_associated_seconds"] > 6_000 or reserved["estimated_elapsed_seconds"] > 6_000:
        raise ValueError("Round8 full-arm forecast exceeds its conservative 6,000-second envelope.")
    if reserved["estimated_persistent_bytes"] > 3 * 1024**3 or reserved["estimated_temporary_bytes"] > 2 * 1024**3:
        raise ValueError("Round8 per-arm storage forecast exceeds its reserved 3/2 GiB bounds.")
    if verify_files:
        if not Path(reservation_path).is_file() or file_sha256(reservation_path) != digest:
            raise ValueError("Round8 reserved-resource evidence points at a changed resource receipt.")
    calibration = evidence["useful_router_gradient"].get("calibration")
    if (not isinstance(calibration, Mapping)
            or set(calibration) != {"membership_bias", "lambda0", "path", "sha256"}):
        raise ValueError("Round8 router-gradient evidence must pin calibration values and their TRAIN-only birth receipt.")
    calibration_path = _canonical_path(calibration.get("path"), "calibration.path")
    calibration_digest = calibration.get("sha256")
    if (not isinstance(calibration_digest, str) or len(calibration_digest) != 64
            or any(character not in "0123456789abcdef" for character in calibration_digest)):
        raise ValueError("Round8 input-only calibration receipt must have a lowercase SHA256 pin.")
    for name in ("membership_bias", "lambda0"):
        number = calibration.get(name)
        if type(number) not in (int, float) or not math.isfinite(float(number)):
            raise ValueError(f"Round8 calibrated {name} must be finite numeric in its admission evidence.")
    if verify_files:
        if not Path(calibration_path).is_file() or file_sha256(calibration_path) != calibration_digest:
            raise ValueError("Round8 input-only calibration receipt changed after sealing.")
        calibration_receipt = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        if not isinstance(calibration_receipt, Mapping):
            raise TypeError("Round8 birth-calibration receipt must be a JSON object.")
        calibration_row = (calibration_receipt.get("datasets", {}).get(receipt["dataset"])
                           if isinstance(calibration_receipt, Mapping) else None)
        batches = calibration_row.get("input_only_calibration_batches") if isinstance(calibration_row, Mapping) else None
        if (calibration_receipt.get("schema") != "round8_birth_native_qualification_v1"
                or calibration_receipt.get("status") != "completed"
                or calibration_receipt.get("TEST_access") is not False
                or calibration_receipt.get("reused_sealed_input_calibration") is not True
                or not isinstance(calibration_row, Mapping)
                or calibration_row.get("source_sha256") != receipt["source_checkpoint_sha256"]
                or calibration_row.get("calibration_targets_used") is not False
                or not isinstance(batches, list) or not 1 <= len(batches) <= 8
                or type(calibration_row.get("membership_bias")) not in (int, float)
                or type(calibration_row.get("lambda0")) not in (int, float)
                or not math.isfinite(float(calibration_row["membership_bias"]))
                or not math.isfinite(float(calibration_row["lambda0"]))
                or float(calibration_row["lambda0"]) <= 0.0
                or float(calibration["membership_bias"]) != float(calibration_row["membership_bias"])
                or float(calibration["lambda0"]) != float(calibration_row["lambda0"])):
            raise ValueError("Round8 calibration receipt is not the exact input-only TRAIN birth/gradient record.")
    startup_speed = receipt.get("startup_speed")
    if (not isinstance(startup_speed, Mapping)
            or startup_speed.get("status") not in ("faster", "not_faster", "not_measured")):
        raise ValueError("Round8 startup-speed evidence must be explicitly recorded but does not gate admission.")
    if source_checkpoint is not None and receipt["source_checkpoint"] != str(Path(source_checkpoint).resolve()):
        raise ValueError("Round8 admission names a different D0 e500 source checkpoint.")
    if source_checkpoint_sha256 is not None and receipt["source_checkpoint_sha256"] != source_checkpoint_sha256:
        raise ValueError("Round8 admission source checkpoint digest differs from the pinned D0 parent.")
    if dataset is not None and receipt["dataset"] != dataset:
        raise ValueError("Round8 admission dataset differs from the D0 source.")
    if native_backend is not None and receipt["native_backend"] != native_backend:
        raise ValueError("Round8 admission backend differs from the G8 recipe.")
    if verify_files:
        if not Path(reservation_path).is_file() or file_sha256(reservation_path) != digest:
            raise ValueError("Round8 admission resource reservation changed after sealing.")
        source = Path(receipt["source_checkpoint"])
        if not source.is_file() or file_sha256(source) != receipt["source_checkpoint_sha256"]:
            raise ValueError("Round8 admission D0 source checkpoint changed after sealing.")
    if canonical_sha256(receipt) != canonical_sha256(dict(value)):
        raise ValueError("Round8 admission contains noncanonical values.")
    return receipt


def _admission_contract(path: str | Path, *, source: Path, digest: str,
                        dataset: str, backend: str) -> dict[str, Any]:
    admission_path = Path(path).expanduser().resolve(strict=True)
    value = json.loads(admission_path.read_text(encoding="utf-8"))
    receipt = validate_round8_research_admission(
        value, source_checkpoint=source, source_checkpoint_sha256=digest,
        dataset=dataset, native_backend=backend, verify_files=True,
    )
    return {"path": str(admission_path), "sha256": file_sha256(admission_path),
            "receipt_sha256": canonical_sha256(receipt)}


def make_native_budgeted_recipe(
    source_checkpoint: str | Path,
    *, treatment: str, research_admission: str | Path, native_backend: str,
    membership_bias: float = 0.0, lambda0: float = 0.0, routing_seed: int = 0,
) -> dict[str, Any]:
    """Seal one D8/G8 continuation to its exact D0 source and new admission."""
    source, identity, digest = _source_contract(source_checkpoint)
    source_path = Path(source_checkpoint).expanduser().resolve(strict=True)
    base_recipe = validate_source_read_recipe(identity["recipe"])
    if treatment not in NATIVE_BUDGETED_TREATMENTS:
        raise ValueError("Round8 treatment must be D-control or G-budgeted-M.")
    if native_backend not in NATIVE_BUDGETED_BACKENDS:
        raise ValueError("Round8 backend must be a declared dense, flat or grouped binary executor.")
    if treatment == "D-control":
        if membership_bias != 0.0 or lambda0 != 0.0:
            raise ValueError("D-control is the unchanged A2-D function and has no router calibration or penalty.")
        admission_path = Path(research_admission).expanduser().resolve(strict=True)
        admission_value = json.loads(admission_path.read_text(encoding="utf-8"))
        admission = validate_round8_research_admission(
            admission_value, source_checkpoint=source_path, source_checkpoint_sha256=digest,
            dataset=base_recipe["dataset"], native_backend=native_backend, verify_files=True,
        )
        admission_binding = {"path": str(admission_path), "sha256": file_sha256(admission_path),
                             "receipt_sha256": canonical_sha256(admission)}
    else:
        if (not math.isfinite(float(membership_bias)) or float(membership_bias) < -32.0
                or float(membership_bias) > 32.0):
            raise ValueError("G8 membership bias must be a finite input-only TRAIN calibration value in [-32,32].")
        if not math.isfinite(float(lambda0)) or float(lambda0) <= 0.0:
            raise ValueError("G8 requires a positive TRAIN-calibrated work coefficient lambda0.")
        if type(routing_seed) is not int or routing_seed < 0:
            raise ValueError("G8 routing seed must be a nonnegative integer.")
        admission_binding = _admission_contract(
            research_admission, source=source_path, digest=digest,
            dataset=base_recipe["dataset"], backend=native_backend,
        )
    recipe = {
        "schema": NATIVE_BUDGETED_RECIPE_SCHEMA,
        "family": NATIVE_BUDGETED_FAMILY,
        "model_family": NATIVE_BUDGETED_FAMILY,
        "workflow": NATIVE_BUDGETED_WORKFLOW,
        "protocol": NATIVE_BUDGETED_PROTOCOL,
        "dataset": base_recipe["dataset"],
        "architecture": "A2-D",
        "treatment": treatment,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": digest,
        "source_epoch": 500,
        "source_recipe_sha256": str(identity["recipe_sha256"]),
        "source_provider_identity_sha256": canonical_sha256(identity["provider_identity"]),
        "base_recipe": base_recipe,
        "base_recipe_sha256": canonical_sha256(base_recipe),
        "research_admission": admission_binding,
        "native_backend": native_backend,
        "environment_value_reader": False,
        "routing_initialization": {
            "membership_bias": float(membership_bias),
            "membership_scale": 1.0,
            "dispatch_scale": 1.0,
            "lambda0": float(lambda0),
            "zeta": float(lambda0),
            "lambda_max": float(lambda0) * 10.0,
            "eta_lambda": float(lambda0),
            "routing_seed": int(routing_seed),
            "head_mixture_initialization": "uniform",
            "membership_birth_target_nonself": 0.98,
            "calibration_source": "up_to_eight_fixed_input_only_TRAIN_batches",
            "new_optimizer_group": "budgeted_native_router",
            "new_optimizer_weight_decay": 0.0,
            "new_optimizer_lr_schedule": "exact_source_schedule",
        },
        "budget_objective": {
            "schema": "native_equal_case_work_budget_v1",
            "quantity": "expected_unique_nonlocal_native_value_rows",
            "protection": "input_defined_direct_local_support_plus_at_least_one_valid_donor",
            "aggregation": "equal_case_means_then_scene_strata",
            "penalty": "lambda*relu(C-rho)+zeta/2*relu(C-rho)^2",
            "response_request_is_a_separate_case": True,
            "controller_observation": "expected_stochastic_work_through_e700_then_deterministic_hard_work",
        },
        "budget_schedule": copy.deepcopy(_BUDGET_SCHEDULE),
        "rng_policy": copy.deepcopy(_RNG_POLICY),
        "execution": copy.deepcopy(base_recipe["execution"]),
    }
    validate_native_budgeted_recipe(recipe)
    assert source["epoch"] == 500
    return recipe


def validate_native_budgeted_recipe(
    value: Mapping[str, Any], *, verify_source_file: bool = False,
    verify_admission: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECIPE_FIELDS:
        raise ValueError("Round8 recipe must contain the exact registered field inventory.")
    recipe = copy.deepcopy(dict(value))
    for key, expected in (
        ("schema", NATIVE_BUDGETED_RECIPE_SCHEMA), ("family", NATIVE_BUDGETED_FAMILY),
        ("model_family", NATIVE_BUDGETED_FAMILY), ("workflow", NATIVE_BUDGETED_WORKFLOW),
        ("protocol", NATIVE_BUDGETED_PROTOCOL), ("architecture", "A2-D"),
    ):
        if recipe.get(key) != expected:
            raise ValueError(f"Round8 recipe requires {key}={expected!r}.")
    if recipe.get("dataset") not in ("thermal", "wind") or recipe.get("treatment") not in NATIVE_BUDGETED_TREATMENTS:
        raise ValueError("Round8 dataset or treatment is not registered.")
    source_path = _canonical_path(recipe.get("source_checkpoint"), "source_checkpoint")
    for name in ("source_checkpoint_sha256", "source_recipe_sha256", "source_provider_identity_sha256", "base_recipe_sha256"):
        digest = recipe.get(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    if recipe.get("source_epoch") != 500 or type(recipe.get("source_epoch")) is not int:
        raise ValueError("Round8 parent must be exact absolute e500.")
    base = validate_source_read_recipe(recipe.get("base_recipe", {}))
    if (base.get("dataset") != recipe["dataset"] or base.get("source_reader_variant") != "D0"
            or base.get("protocol") != "source_read_development_v1" or base.get("architecture") != "A2-D"
            or canonical_sha256(base) != recipe["base_recipe_sha256"]):
        raise ValueError("Round8 base recipe differs from the exact same-dataset segmented D0 source.")
    if recipe.get("environment_value_reader") is not False:
        raise ValueError("Round8 primary family has no direct E value consumer.")
    if recipe.get("native_backend") not in NATIVE_BUDGETED_BACKENDS:
        raise ValueError("Round8 native backend is not registered.")
    if recipe.get("budget_schedule") != _BUDGET_SCHEDULE or recipe.get("rng_policy") != _RNG_POLICY:
        raise ValueError("Round8 absolute-age budget or original A2-D sampling policy changed.")
    if recipe.get("execution") != base.get("execution"):
        raise ValueError("Round8 must retain the original D0 checkpoint, validation and response cadence.")
    admission = recipe.get("research_admission")
    if not isinstance(admission, Mapping) or set(admission) != {"path", "sha256", "receipt_sha256"}:
        raise ValueError("Round8 recipe must pin a distinct research admission receipt.")
    _canonical_path(admission.get("path"), "research_admission.path")
    for key in ("sha256", "receipt_sha256"):
        if not isinstance(admission.get(key), str) or len(admission[key]) != 64:
            raise ValueError(f"Round8 admission {key} is malformed.")
    init = recipe.get("routing_initialization")
    if not isinstance(init, Mapping) or set(init) != {
        "membership_bias", "membership_scale", "dispatch_scale", "lambda0", "zeta", "lambda_max",
        "eta_lambda", "routing_seed", "head_mixture_initialization", "membership_birth_target_nonself",
        "calibration_source", "new_optimizer_group", "new_optimizer_weight_decay", "new_optimizer_lr_schedule",
    }:
        raise ValueError("Round8 route initialization must contain the exact calibration and optimizer contract.")
    objective = recipe.get("budget_objective")
    if not isinstance(objective, Mapping) or set(objective) != {
        "schema", "quantity", "protection", "aggregation", "penalty",
        "response_request_is_a_separate_case", "controller_observation",
    } or objective != {
        "schema": "native_equal_case_work_budget_v1",
        "quantity": "expected_unique_nonlocal_native_value_rows",
        "protection": "input_defined_direct_local_support_plus_at_least_one_valid_donor",
        "aggregation": "equal_case_means_then_scene_strata",
        "penalty": "lambda*relu(C-rho)+zeta/2*relu(C-rho)^2",
        "response_request_is_a_separate_case": True,
        "controller_observation": "expected_stochastic_work_through_e700_then_deterministic_hard_work",
    }:
        raise ValueError("Round8 work objective semantics changed from the declared unique-row constraint.")
    for key in ("membership_bias", "membership_scale", "dispatch_scale", "lambda0", "zeta", "lambda_max", "eta_lambda", "membership_birth_target_nonself", "new_optimizer_weight_decay"):
        number = init.get(key)
        if type(number) not in (int, float) or not math.isfinite(float(number)):
            raise ValueError(f"Round8 initialization {key} must be finite numeric.")
    if recipe["treatment"] == "D-control":
        if (any(float(init[key]) != 0.0 for key in ("membership_bias", "lambda0", "zeta", "lambda_max", "eta_lambda"))
                or float(init["membership_scale"]) != 1.0 or float(init["dispatch_scale"]) != 1.0):
            raise ValueError("D-control has no learned route, router penalty or native executor substitution.")
    else:
        lambda0 = float(init["lambda0"])
        if (lambda0 <= 0.0 or float(init["zeta"]) != lambda0
                or float(init["eta_lambda"]) != lambda0 or float(init["lambda_max"]) != 10.0 * lambda0
                or float(init["membership_scale"]) != 1.0 or float(init["dispatch_scale"]) != 1.0
                or int(init["routing_seed"]) < 0 or type(init["routing_seed"]) is not int
                or float(init["membership_birth_target_nonself"]) != 0.98
                or init["head_mixture_initialization"] != "uniform"
                or init["new_optimizer_group"] != "budgeted_native_router"
                or float(init["new_optimizer_weight_decay"]) != 0.0
                or init["new_optimizer_lr_schedule"] != "exact_source_schedule"):
            raise ValueError("G8 route calibration, dual constants or fresh optimizer group differ from the sealed policy.")
    if canonical_sha256(recipe) != canonical_sha256(dict(value)):
        raise ValueError("Round8 recipe contains noncanonical fields.")
    if verify_source_file or verify_admission:
        if file_sha256(source_path) != recipe["source_checkpoint_sha256"]:
            raise ValueError("Round8 D0 source checkpoint changed after recipe creation.")
        _payload, identity, _digest = _source_contract(source_path)
        if (identity.get("recipe_sha256") != recipe["source_recipe_sha256"]
                or canonical_sha256(identity.get("provider_identity")) != recipe["source_provider_identity_sha256"]
                or canonical_sha256(identity.get("recipe")) != recipe["base_recipe_sha256"]):
            raise ValueError("Round8 D0 source recipe/provider identity changed after sealing.")
    if verify_admission:
        admission_path = Path(admission["path"])
        if file_sha256(admission_path) != admission["sha256"]:
            raise ValueError("Round8 research admission receipt changed after recipe creation.")
        value = json.loads(admission_path.read_text(encoding="utf-8"))
        checked = validate_round8_research_admission(
            value, source_checkpoint=source_path, source_checkpoint_sha256=recipe["source_checkpoint_sha256"],
            dataset=recipe["dataset"], native_backend=recipe["native_backend"], verify_files=True,
        )
        if canonical_sha256(checked) != admission["receipt_sha256"]:
            raise ValueError("Round8 research admission content no longer matches the recipe binding.")
        if recipe["treatment"] == "G-budgeted-M":
            calibrated = checked["evidence"]["useful_router_gradient"]["calibration"]
            if (float(init["membership_bias"]) != float(calibrated["membership_bias"])
                    or float(init["lambda0"]) != float(calibrated["lambda0"])):
                raise ValueError("G8 recipe bias/lambda differ from the pinned input-only TRAIN calibration receipt.")
    return recipe


def native_budgeted_child_declaration(recipe: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_native_budgeted_recipe(recipe)
    return {
        "schema": NATIVE_BUDGETED_CHILD_SCHEMA,
        "treatment": value["treatment"],
        "source_checkpoint": value["source_checkpoint"],
        "source_checkpoint_sha256": value["source_checkpoint_sha256"],
        "source_epoch": 500,
        "source_recipe_sha256": value["source_recipe_sha256"],
        "source_provider_identity_sha256": value["source_provider_identity_sha256"],
        "child_recipe_sha256": canonical_sha256(value),
        "optimizer_policy": "exact_named_adamw_source_groups_plus_fresh_named_route_group_for_G8",
        "sampling_policy": "original_D0_A2-D_case_epoch_v1_absolute_age_and_query_namespace",
        "budget_schedule": copy.deepcopy(value["budget_schedule"]),
        "research_admission": copy.deepcopy(value["research_admission"]),
    }


__all__ = [
    "NATIVE_BUDGETED_ADMISSION_SCHEMA", "NATIVE_BUDGETED_BACKENDS", "NATIVE_BUDGETED_CHILD_SCHEMA",
    "NATIVE_BUDGETED_FAMILY", "NATIVE_BUDGETED_PROTOCOL", "NATIVE_BUDGETED_RECIPE_SCHEMA",
    "NATIVE_BUDGETED_TREATMENTS", "NATIVE_BUDGETED_WORKFLOW", "canonical_sha256", "file_sha256",
    "make_native_budgeted_recipe", "native_budgeted_child_declaration",
    "validate_native_budgeted_recipe", "validate_round8_research_admission",
]
