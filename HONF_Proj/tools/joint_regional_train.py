#!/usr/bin/env python
"""Fresh joint field development and explicitly manual full-TRAIN recipes.

All optimization uses the maintained TrainingEngine. Preparation never takes
an optimizer step; dry-run uses a disposable CPU model. Formal optimization
requires the explicit --manual-formal-launch switch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from full_train_followup_contract import (
    PROTOCOL as FULL_TRAIN_FOLLOWUP_PROTOCOL,
)
from full_train_followup_contract import (
    validate_recipe as validate_full_train_followup_recipe,
)

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "Case_ThermalChannel/src", ROOT / "Case_WindFarm/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

MODES = ("J-direct", "J-geometry", "J-H", "P", "P-G", "P-H")
RECIPE_KEYS = {
    "schema_version", "task", "mode", "run_id", "seed", "hidden", "message", "regional_anchors",
    "depth", "receiver_tile", "primary_queries", "microbatch_cases", "effective_cases",
    "formal_full", "total_epochs", "initialization", "launch_policy", "dataset_protocol",
    "response_coefficient", "operator_coefficient",
    "auxiliary_calibration",
    "calibration_receipt",
    "locality_prior_strength", "collective_width", "max_sources", "environment_token_shape",
    "model_contract", "objective_contract", "engine_schedule", "optimizer_name",
    "optimizer_betas", "optimizer_eps", "gradient_clip_norm",
    "validation_scope", "optimizer_schedule", "weight_decay", "native_sampling_protocol",
    "checkpoint_epochs", "write_initial_artifacts", "flow_readout_law",
    "execution_protocol", "full_train_followup1000", "approved_stop_after",
    "root_protocol_go", "manual_full_followup", "parent_checkpoint",
    "auxiliary_calibration_fallback", "population_binding", "mechanism_reference",
    "wind_test_target_values_read",
}

RECOVERY_OPTIMIZER_SCHEDULE = {
    "peak_lr": 3.0e-4,
    "warmup_start_lr": 3.0e-5,
    "warmup_epochs": 20,
    "hold_through_epoch": 1000,
    "final_lr": 3.0e-6,
}
RECOVERY_ENGINE_SCHEDULE = {
    "training_mode": "joint",
    "sampling_version": "case_epoch_v1",
    "warmup_epochs": 20,
    "open_through_epoch": 20,
    "soft_through_epoch": 20,
    "monitor_every": 100,
    "latest_every": 100,
    "curve_every": 100,
}
RECOVERY_CHECKPOINT_EPOCHS = list(range(100, 2501, 100))
RECOVERY_SOURCE_FILES = (
    "HONF_Proj/tools/joint_regional_train.py",
    "HONF_Proj/src/honf_runtime/unified_training.py",
    "HONF_Proj/src/honf_runtime/reproducibility.py",
    "HONF_Proj/src/honf_runtime/compat.py",
    "HONF_Proj/src/honf_runtime/run_layout.py",
    "HONF_Proj/src/honf_forward_core/interface_fields/interaction_core.py",
    "HONF_Proj/src/honf_forward_core/interface_fields/interaction_preserving_joint.py",
    "HONF_Proj/src/honf_forward_core/interface_fields/source_response_operator.py",
    "HONF_Proj/Case_ThermalChannel/src/channelthermal/joint_regional.py",
    "HONF_Proj/Case_ThermalChannel/src/channelthermal/source_response.py",
    "HONF_Proj/Case_ThermalChannel/src/channelthermal/training/joint_task.py",
    "HONF_Proj/Case_ThermalChannel/src/channelthermal/training/unified_task.py",
    "HONF_Proj/Case_ThermalChannel/src/channelthermal/source_response_residual.py",
    "HONF_Proj/Case_ThermalChannel/src/channelthermal/data/datasets.py",
    "HONF_Proj/Case_ThermalChannel/src/channelthermal/data/development_split.py",
    "HONF_Proj/tools/thermal_source_response_fit.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/interaction_preserving.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/joint_regional.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/geometry.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/study_spatial.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/io.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/workflows/native_role_cache.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/normalization.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/training/joint_task.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/training/unified_task.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/data.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/splits.py",
    "HONF_Proj/Case_WindFarm/src/windfarm/workflows/joint_forward.py",
)


NATIVE_CURL_THERMAL_LAW = "native_curl_cell_centred_v1"
NATIVE_CURL_FLOW_SOURCE = (
    "HONF_Proj/Case_ThermalChannel/src/channelthermal/flow_curl.py"
)


def _recovery_source_files(flow_readout_law: str | None = None, *,
                           full_train_followup: bool = False) -> tuple[str, ...]:
    files = RECOVERY_SOURCE_FILES
    if flow_readout_law is not None:
        if flow_readout_law != NATIVE_CURL_THERMAL_LAW:
            raise ValueError("Unsupported optional Thermal flow readout law.")
        files += (NATIVE_CURL_FLOW_SOURCE,)
    if full_train_followup:
        files += ("HONF_Proj/tools/full_train_followup_contract.py",)
    return files


def _recovery_model_contract(task: str, mode: str,
                             flow_readout_law: str | None = None) -> dict[str, Any]:
    if task == "thermal":
        contract = {
            "input_frame": "packed Thermal native x-y coordinates; source centres/radii normalized by physical domain lengths",
            "native_shared_grid_shape": [128, 64],
            "source": {
                "feature_width": 8,
                "feature_inventory": ["center_x/Lx", "center_y/Ly", "radius/Lx", "radius/Ly", "solid_alpha_x100", "fluid_alpha_x100", "solid_k", "fluid_k"],
                "context_width": 14,
                "context_inventory": ["Re/100", "u_in", "nu*100", "solid_alpha*100", "fluid_alpha*100", "solid_k", "fluid_k", "radius", "Lx/12", "Ly/6", "module_count/12", "module_count/(Lx*Ly)", "zero_inlet_temperature", "zero_wall_temperature"],
                "measure": "pi * physical_radius^2 * source_presence",
                "identity": "original module slot IDs; ordered and unique among active sources",
                "max_sources": 12,
            },
            "environment": {
                "feature_width": 8,
                "shape": [24, 8],
                "count": 192,
                "features": ["x/Lx", "y/Ly", "nearest_active_source_distance/radius - 1", "sum(exp(-distance_squared/radius_squared) * source_presence)", "x/Lx", "y/Ly", "1-x/Lx", "1-y/Ly"],
                "measure": "Lx * Ly / 192 physical area per geometry-only token",
                "target_free": True,
            },
            "receivers": {"frame": "packed native x-y physical coordinates", "feature_width": 0, "tile": 512},
            "readouts": {
                "flow_head": {"law": "nonlinear source-conditioned field read", "outputs": 4, "order": ["u", "v", "p", "omega"]},
                "temperature_head": {"law": "source-resolved affine heat response", "outputs": 1, "zero_offset": True},
                "source_read_networks": {"flow": "field_source_read", "temperature": "affine_source_read", "distinct": True},
                "forcing_scale": 1.0,
            },
        }
        if flow_readout_law is not None:
            if flow_readout_law != NATIVE_CURL_THERMAL_LAW:
                raise ValueError("Unsupported optional Thermal flow readout law.")
            from channelthermal.flow_curl import native_curl_readout_contract
            contract["readouts"]["flow_head"] = {
                "law": "nonlinear source-conditioned field read",
                "outputs": 3,
                "order": ["u", "v", "p"],
                "initialization_reference_field_outputs": 4,
            }
            contract["readouts"]["derived_omega"] = native_curl_readout_contract()
            contract["receivers"]["native_solid_mask"] = (
                "saved boolean [B,ny,nx] geometry-only receiver metadata; excluded from learned context"
            )
    else:
        if flow_readout_law is not None:
            raise ValueError("A native Thermal curl readout is invalid for Wind.")
        contract = {
            "input_frame": "Wind native rotor-diameter coordinates; rotor locations, support and receivers remain in D",
            "source": {
                "feature_width": 2,
                "feature_inventory": ["rotor_radius_D", "hub_height_D"],
                "context_width": 11,
                "context_inventory": ["wind_direction_one_hot_270", "wind_direction_one_hot_285", "wind_direction_one_hot_300", "active_turbines/30", "U_ref/U_REF", "support_lower_x_D/50", "support_lower_y_D/38", "support_lower_z_D/6.25", "support_extent_x_D/50", "support_extent_y_D/38", "support_extent_z_D/6.25"],
                "measure": "source presence; source length is rotor diameter in D",
                "identity": "original turbine slot IDs; explicit and unique among active sources",
                "max_sources": 30,
            },
            "environment": {
                "feature_width": 7,
                "shape": [2, 2, 2],
                "count": 8,
                "features": ["(x-support_lower_x)/50D", "(y-support_lower_y)/38D", "(z-support_lower_z)/6.25D", "(support_upper_x-x)/50D", "(support_upper_y-y)/38D", "(support_upper_z-z)/6.25D", "absolute_z/6.25D"],
                "measure": "native domain-support quadrature volume in rotor_diameters^3 per geometry-only token",
                "target_free": True,
            },
            "receivers": {
                "frame": "rotor_diameters",
                "feature_width": 7,
                "feature_scale_D": [50.0, 38.0, 6.25],
                "feature_quantities": ["receiver_minus_domain_origin/scale_D", "domain_upper_minus_receiver/scale_D", "absolute_height/6.25_D"],
                "tile": 512,
            },
            "readouts": {
                "field_head": {"law": "nonlinear source-conditioned standardized velocity residual", "outputs": 3, "order": ["Ux", "Uy", "Uz"]},
                "affine_outputs": 0,
                "physical_output": "TRAIN height profile m/s + residual * u_ref_mps * safe_std",
            },
        }
    if mode == "P":
        contract["collective"] = {"enabled": False, "placement": "none"}
    else:
        contract["collective"] = {
            "enabled": True,
            "placement": "one typed block between the two pair-message rounds",
            "physical_source_edges": "one edge centered on each active source; Thermal scale is physical source radius; Wind scale is rotor diameter in D",
            "regional_anchors": "K deterministic Halton anchors from radical-inverse bases (2,3) in 2D or (2,3,5) in 3D, indices 1..K mapped over domain_origin + unit * lengths",
            "regional_anchor_scale": "per-axis domain lengths / K^(1/spatial_dimension)",
            "membership": "source and geometry-token measure-aware softmax; physical measure enters donor mass exactly once",
            "mode_incidence_and_access": "P-G uses fixed geometric incidence/access; P-H adds learned score residuals over the same lambda=1 geometric prior, with zero-initialized residual scores",
            "return_rule": "normalized transpose of typed donor incidence for context sharing",
            "initialization": "zero node/environment/receiver output projections recover P exactly at initialization",
        }
    return contract


def _recovery_objective_contract(task: str) -> dict[str, Any]:
    if task == "thermal":
        return {
            "primary_queries_per_case": 1024,
            "primary_cohort": "fixed25_v1: all 150 selected TRAIN cases; exposed DEV22 metrics only",
            "flow_group": {"weight": 0.5, "roles": ["u", "v", "p", "omega"], "per_role_weight": 0.125, "normalization": "selected TRAIN global field std per channel"},
            "thermal_group": {"weight": 0.5, "roles": ["fluid_temperature", "surface_temperature", "material_temperature"], "per_role_weight": 1.0 / 6.0, "normalization": "selected TRAIN physical role scales"},
            "q_proxy": {"weight": 0.05, "inside_balanced_group": False},
            "native_operator": {"rows_per_case": 128, "coefficient": "fresh-P TRAIN gradient-ratio receipt"},
            "response_addendum": {
                "families": ["0001", "0318", "0333", "0348"],
                "partition": "original TRAIN only",
                "fluid_queries_per_family": 1024,
                "material_queries_per_module": 32,
                "surface_stride": 4,
                "coefficient": "fresh-P TRAIN gradient-ratio receipt",
                "primary_cohort_overlap": ["0348"],
            },
            "calibration_scope": "fresh P init; fixed25 TRAIN only; no DEV case or DEV response labels materialized; zero optimizer steps",
        }
    return {
        "primary_queries_per_case": 4096,
        "primary_cohort": "wind_shared_fixed24_v1: matched selected TRAIN/validation rows and fixed query stream",
        "role_query_counts": {"background": 816, "downstream_envelope": 820, "hub_slab": 820, "near_turbine": 820, "volume": 820},
        "objective": "equal-weight five-role component-balanced velocity residual",
        "role_weight": 0.2,
        "component_scales": "sealed corrected TRAIN-profile residual RMS by role and velocity component; same E64 calibration cache as J controls",
        "role_vector_rmse": "sqrt(sum of three component MSE); no divide by three",
        "windtest": "locked; no WindTEST targets read and zero solver attempts",
    }


def read_recipe(path: str | Path) -> dict[str, Any]:
    recipe = json.loads(Path(path).read_text())
    return validate_recipe(recipe)


def validate_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(recipe, dict) or set(recipe) - RECIPE_KEYS:
        raise ValueError("Joint recipe contains unsupported fields.")
    if recipe.get("schema_version") != 1 or recipe.get("task") not in ("thermal", "wind"):
        raise ValueError("Joint recipe schema/task is invalid.")
    if recipe.get("mode") not in MODES or recipe.get("initialization") != "fresh_all_trainable":
        raise ValueError("Joint recipes require a declared mode and fresh trainable weights.")
    flow_readout_law = recipe.get("flow_readout_law")
    if flow_readout_law is not None:
        if (recipe["task"] != "thermal" or recipe["mode"] not in ("P", "P-G", "P-H")
                or recipe["formal_full"]):
            raise ValueError("The native curl law is limited to bounded fresh Thermal P-family recipes.")
        if flow_readout_law != NATIVE_CURL_THERMAL_LAW:
            raise ValueError("Unsupported optional Thermal flow readout law.")
    if recipe.get("formal_full") and recipe.get("launch_policy") != "manual_only":
        raise ValueError("Formal recipes must be manual-only.")
    for name in ("hidden", "message", "depth", "receiver_tile", "primary_queries",
                 "microbatch_cases", "effective_cases", "total_epochs"):
        if type(recipe.get(name)) is not int or recipe[name] < 1:
            raise ValueError(f"Recipe {name} must be a positive integer.")
    anchors = recipe.get("regional_anchors")
    if type(anchors) is not int or anchors < (0 if recipe.get("mode") == "P" else 1):
        raise ValueError("Recipe regional_anchors must be zero only for P and positive for graph modes.")
    effective = 48 if recipe["task"] == "thermal" else 24
    if recipe["effective_cases"] != effective or recipe["microbatch_cases"] > effective:
        raise ValueError("Joint recipes preserve the dataset's effective batch.")
    full_followup = recipe.get("execution_protocol") == FULL_TRAIN_FOLLOWUP_PROTOCOL
    if recipe["total_epochs"] != (5000 if recipe.get("formal_full") or full_followup else 2500):
        raise ValueError("Joint recipe horizon must match the development/formal identity.")
    if type(recipe.get("seed")) is not int or type(recipe.get("formal_full")) is not bool:
        raise ValueError("Joint seed/formal identity has invalid types.")
    expected = recipe["dataset_protocol"] if full_followup else (
        ("original600_train" if recipe["task"] == "thermal" else "original420_train")
        if recipe["formal_full"] else
        ("fixed25_v1" if recipe["task"] == "thermal" else "wind_shared_fixed24_v1")
    )
    if recipe.get("dataset_protocol") != expected:
        raise ValueError("Joint recipe dataset protocol differs from its population identity.")
    for name in ("response_coefficient", "operator_coefficient"):
        if name in recipe and (not isinstance(recipe[name], (float, int)) or
                               not math.isfinite(recipe[name]) or recipe[name] < 0):
            raise ValueError(f"Recipe {name} must be finite and nonnegative.")
    recovery = recipe["mode"] in ("P", "P-G", "P-H")
    prior = recipe.get("locality_prior_strength", 0.0)
    if (type(prior) not in (float, int) or not math.isfinite(prior) or prior < 0
            or (prior and recipe["mode"] not in ("J-H", "P-G", "P-H"))):
        raise ValueError("Locality prior must be finite, nonnegative and used only by a declared graph arm.")
    if recipe["mode"] == "P" and prior != 0:
        raise ValueError("P is the direct source-conditioned path and has no locality prior.")
    if recipe["mode"] in ("P-G", "P-H") and prior != 1.0:
        raise ValueError("P-G and P-H are sealed to the shared geometric locality prior lambda=1.")
    formal_options = {"validation_scope", "optimizer_schedule", "weight_decay",
                      "native_sampling_protocol", "checkpoint_epochs", "write_initial_artifacts"}
    if (not recipe["formal_full"] and not full_followup and formal_options.intersection(recipe)
            and not (recovery and not (formal_options - {"optimizer_schedule", "weight_decay", "checkpoint_epochs"}).intersection(recipe))):
        raise ValueError("Formal comparison controls require the separate fullTRAIN identity.")
    if "validation_scope" in recipe and recipe["validation_scope"] != (
            "canonical89" if recipe["task"] == "thermal" else "fullVALID90"):
        raise ValueError("Formal validation scope differs from the dataset's primary panel.")
    if "native_sampling_protocol" in recipe and recipe["native_sampling_protocol"] != "baseline_formal_v1":
        raise ValueError("Unsupported formal native sampling protocol.")
    if "optimizer_schedule" in recipe:
        from honf_runtime.unified_training import ScheduleSpec
        schedule = recipe["optimizer_schedule"]
        keys = {"peak_lr", "warmup_start_lr", "warmup_epochs", "hold_through_epoch", "final_lr"}
        if not isinstance(schedule, dict) or set(schedule) != keys:
            raise ValueError("Formal optimizer schedule must declare every rate and clock boundary.")
        for name in ("peak_lr", "warmup_start_lr", "final_lr"):
            if type(schedule[name]) not in (int, float):
                raise ValueError("Formal learning rates must be numeric, not boolean.")
        for name in ("warmup_epochs", "hold_through_epoch"):
            if type(schedule[name]) is not int:
                raise ValueError("Formal schedule clock boundaries must be integers.")
        ScheduleSpec(total_epochs=recipe["total_epochs"], **schedule)
    if "weight_decay" in recipe and (type(recipe["weight_decay"]) not in (int, float)
            or not math.isfinite(recipe["weight_decay"]) or recipe["weight_decay"] < 0):
        raise ValueError("Formal weight decay must be finite and nonnegative.")
    if "checkpoint_epochs" in recipe:
        epochs = recipe["checkpoint_epochs"]
        required_final_checkpoint = recipe.get("approved_stop_after") if full_followup else recipe["total_epochs"]
        if (not isinstance(epochs, list)
                or any(type(epoch) is not int or not 1 <= epoch <= recipe["total_epochs"] for epoch in epochs)
                or sorted(set(epochs)) != epochs or required_final_checkpoint not in epochs):
            raise ValueError("Formal checkpoint epochs must be sorted unique and include the horizon.")
    if "write_initial_artifacts" in recipe and type(recipe["write_initial_artifacts"]) is not bool:
        raise ValueError("Initial artifact control must be boolean.")
    if full_followup:
        if recipe.get("full_train_followup1000") is not True:
            raise ValueError("Full-data follow-up must declare its distinct bounded protocol flag.")
        if recipe.get("formal_full") is not False:
            raise ValueError("Full-data follow-up cannot reuse the legacy formal_full identity.")
        _validate_full_followup_mechanism(recipe)
        validate_full_train_followup_recipe(recipe, command="prepare")
    elif recovery:
        _validate_recovery_recipe(recipe)
    return recipe


def _validate_recovery_recipe(recipe: dict[str, Any]) -> None:
    task, mode = recipe["task"], recipe["mode"]
    if task == "thermal" and recipe.get("flow_readout_law") == NATIVE_CURL_THERMAL_LAW:
        expected_id = {"P": "T4111", "P-G": "T4112", "P-H": "T4113"}[mode]
    else:
        expected_id = ({"P": "T4101", "P-G": "T4102", "P-H": "T4103"} if task == "thermal"
                       else {"P": "W2401", "P-G": "W2402", "P-H": "W2403"})[mode]
    expected_seed = 0 if task == "thermal" else 42
    expected_anchors = 0 if mode == "P" else (16 if task == "thermal" else 32)
    expected_shape = [24, 8] if task == "thermal" else [2, 2, 2]
    expected_sources = 12 if task == "thermal" else 30
    expected_queries = 1024 if task == "thermal" else 4096
    expected_effective = 48 if task == "thermal" else 24
    required = {
        "run_id", "collective_width", "max_sources", "environment_token_shape", "model_contract",
        "objective_contract", "engine_schedule", "optimizer_name", "optimizer_betas", "optimizer_eps",
        "gradient_clip_norm", "optimizer_schedule", "weight_decay", "checkpoint_epochs",
    }
    if recipe["task"] == "thermal":
        required.add("calibration_receipt")
    if not required.issubset(recipe):
        raise ValueError(f"Recovery recipe is missing sealed fields: {sorted(required - set(recipe))}.")
    exact = {
        "run_id": expected_id,
        "seed": expected_seed,
        "hidden": 128,
        "message": 128,
        "regional_anchors": expected_anchors,
        "collective_width": 64,
        "depth": 2,
        "receiver_tile": 512,
        "primary_queries": expected_queries,
        "effective_cases": expected_effective,
        "formal_full": False,
        "total_epochs": 2500,
        "initialization": "fresh_all_trainable",
        "launch_policy": "bounded_development",
        "dataset_protocol": "fixed25_v1" if task == "thermal" else "wind_shared_fixed24_v1",
        "max_sources": expected_sources,
        "environment_token_shape": expected_shape,
        "model_contract": _recovery_model_contract(
            task, mode, recipe.get("flow_readout_law")),
        "objective_contract": _recovery_objective_contract(task),
        "engine_schedule": RECOVERY_ENGINE_SCHEDULE,
        "optimizer_name": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1.0e-8,
        "gradient_clip_norm": 1.0,
        "optimizer_schedule": RECOVERY_OPTIMIZER_SCHEDULE,
        "weight_decay": 1.0e-4 if task == "thermal" else 1.0e-5,
        "checkpoint_epochs": RECOVERY_CHECKPOINT_EPOCHS,
    }
    for name, value in exact.items():
        if recipe.get(name) != value:
            raise ValueError(f"Recovery recipe {name} differs from the sealed common P/P-G/P-H contract.")
    if recipe["microbatch_cases"] not in (8, 16):
        raise ValueError("Recovery recipes use one of the two measured microbatch candidates: 8 or 16.")
    if recipe["total_epochs"] != 2500 or recipe["checkpoint_epochs"] != RECOVERY_CHECKPOINT_EPOCHS:
        raise ValueError("Recovery recipe must retain the shared 2500-epoch review/checkpoint boundaries.")
    if task == "thermal":
        descriptor = recipe["calibration_receipt"]
        native_curl = recipe.get("flow_readout_law") == NATIVE_CURL_THERMAL_LAW
        expected_path = (
            "diagnostics/generated/interaction_recovery_20261010/thermal_native_curl_fresh_P_calibration.json"
            if native_curl else
            "diagnostics/generated/interaction_recovery_20261010/thermal_fresh_P_calibration.json"
        )
        basic_descriptor = (isinstance(descriptor, dict)
                            and descriptor.get("receipt_path") == expected_path
                            and isinstance(descriptor.get("required_payload_sha256"), str))
        if not basic_descriptor:
            raise ValueError("Thermal recovery recipe must bind the maintained fresh-P TRAIN receipt location.")
        digest = descriptor["required_payload_sha256"]
        expected_sentinel = ("sealed_after_fresh_P_native_curl_calibration" if native_curl
                             else "sealed_after_fresh_P_calibration")
        if digest == expected_sentinel:
            invalid_auxiliary_calibration = (
                recipe.get("auxiliary_calibration") is not None if native_curl else
                "auxiliary_calibration" in recipe
            )
            if (set(descriptor) != {"receipt_path", "required_payload_sha256"}
                    or invalid_auxiliary_calibration):
                raise ValueError("Unresolved Thermal calibration descriptor has unexpected receipt fields.")
        else:
            valid_digest = len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
            file_digest = descriptor.get("file_sha256")
            valid_file_digest = (isinstance(file_digest, str) and len(file_digest) == 64
                                 and all(char in "0123456789abcdef" for char in file_digest))
            receipt = recipe.get("auxiliary_calibration")
            if (set(descriptor) != {"receipt_path", "required_payload_sha256", "file_sha256"}
                    or not valid_digest or not valid_file_digest or not isinstance(receipt, dict)
                    or receipt.get("payload_sha256") != digest
                    or _canonical_payload_sha256(receipt) != digest
                    or recipe.get("response_coefficient") != receipt.get("response_coefficient")
                    or recipe.get("operator_coefficient") != receipt.get("operator_coefficient")):
                raise ValueError("Resolved Thermal recipe is not sealed to its external calibration receipt.")


def _validate_full_followup_mechanism(recipe: dict[str, Any]) -> None:
    """Validate the unchanged model/objective by mapping population labels back to the sealed quarter recipe."""
    segmented = copy.deepcopy(recipe)
    task = recipe["task"]
    expected_objective = _recovery_objective_contract(task)
    objective = segmented.get("objective_contract")
    if not isinstance(objective, dict):
        raise TypeError("Full follow-up recipe must retain the sealed native objective contract.")
    for label in ("primary_cohort", "calibration_scope"):
        if label in expected_objective:
            objective[label] = expected_objective[label]
    if "response_addendum" in expected_objective:
        addendum = objective.get("response_addendum")
        if not isinstance(addendum, dict):
            raise ValueError("Thermal full follow-up must retain its four-family response addendum.")
        addendum["primary_cohort_overlap"] = expected_objective["response_addendum"]["primary_cohort_overlap"]
    segmented["objective_contract"] = objective
    segmented["dataset_protocol"] = "fixed25_v1" if task == "thermal" else "wind_shared_fixed24_v1"
    segmented["total_epochs"] = 2500
    segmented["launch_policy"] = "bounded_development"
    segmented["optimizer_schedule"] = RECOVERY_OPTIMIZER_SCHEDULE
    segmented["checkpoint_epochs"] = RECOVERY_CHECKPOINT_EPOCHS
    segmented["run_id"] = (
        {"P": "T4111", "P-G": "T4112", "P-H": "T4113"}[recipe["mode"]]
        if task == "thermal" and recipe.get("flow_readout_law") == NATIVE_CURL_THERMAL_LAW else
        {"P": "W2401", "P-G": "W2402", "P-H": "W2403"}[recipe["mode"]]
    )
    if task == "thermal":
        segmented["calibration_receipt"] = {
            "receipt_path": "diagnostics/generated/interaction_recovery_20261010/thermal_native_curl_fresh_P_calibration.json",
            "required_payload_sha256": "sealed_after_fresh_P_native_curl_calibration",
        }
        segmented.pop("auxiliary_calibration", None)
        segmented.pop("response_coefficient", None)
        segmented.pop("operator_coefficient", None)
    segmented.pop("execution_protocol", None)
    segmented.pop("full_train_followup1000", None)
    segmented.pop("approved_stop_after", None)
    segmented.pop("root_protocol_go", None)
    segmented.pop("manual_full_followup", None)
    segmented.pop("parent_checkpoint", None)
    segmented.pop("auxiliary_calibration_fallback", None)
    segmented.pop("population_binding", None)
    segmented.pop("mechanism_reference", None)
    segmented.pop("wind_test_target_values_read", None)
    _validate_recovery_recipe(segmented)


def resolved_recipe(args: argparse.Namespace) -> dict[str, Any]:
    if args.recipe_json:
        recipe = read_recipe(args.recipe_json)
        if recipe["mode"] in ("P", "P-G", "P-H") and recipe["task"] == "thermal" \
                and args.command not in ("calibrate-thermal", "validate-followup"):
            recipe = _bind_calibration_receipt(recipe)
        return recipe
    task = args.task
    if task is None:
        raise ValueError("Provide --task or --recipe-json.")
    recipe = {
        "schema_version": 1, "task": task, "mode": args.mode,
        "seed": args.seed if args.seed is not None else (0 if task == "thermal" else 42),
        "hidden": args.hidden, "message": args.message,
        "regional_anchors": args.regional_anchors or (16 if task == "thermal" else 32),
        "depth": args.depth, "receiver_tile": args.receiver_tile,
        "primary_queries": args.primary_queries or (1024 if task == "thermal" else 4096),
        "microbatch_cases": args.microbatch_cases or (4 if task == "thermal" else 1),
        "effective_cases": 48 if task == "thermal" else 24,
        "formal_full": False, "total_epochs": 2500,
        "initialization": "fresh_all_trainable", "launch_policy": "bounded_development",
        "dataset_protocol": "fixed25_v1" if task == "thermal" else "wind_shared_fixed24_v1",
    }
    if args.locality_prior_strength:
        recipe['locality_prior_strength'] = args.locality_prior_strength
    return validate_recipe(recipe)


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode()).hexdigest()


def _bind_calibration_receipt(recipe: dict[str, Any]) -> dict[str, Any]:
    descriptor = recipe["calibration_receipt"]
    path = (ROOT / descriptor["receipt_path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Thermal P-family requires the fresh-P TRAIN calibration receipt at {path}; "
            "run calibrate-thermal with the P recipe first."
        )
    receipt = json.loads(path.read_text())
    actual_digest = _canonical_payload_sha256(receipt)
    if receipt.get("payload_sha256") != actual_digest:
        raise ValueError("Thermal calibration receipt payload digest is missing or invalid.")
    flow_readout_law = recipe.get("flow_readout_law")
    source_options = {"require_clean": True}
    if flow_readout_law is not None:
        source_options["flow_readout_law"] = flow_readout_law
    if recipe.get("execution_protocol") == FULL_TRAIN_FOLLOWUP_PROTOCOL:
        source_options["full_train_followup"] = True
    active_source_identity = training_source_identity(**source_options)
    if receipt.get("training_source_identity") != active_source_identity:
        raise ValueError("Thermal calibration receipt was produced by a different committed training source revision.")
    if (recipe.get("flow_readout_law") is not None
            and receipt.get("flow_readout_law") != recipe.get("flow_readout_law")):
        raise ValueError("Thermal calibration receipt binds a different flow readout law.")
    expected_sentinel = (
        "sealed_after_fresh_P_native_curl_calibration"
        if flow_readout_law == NATIVE_CURL_THERMAL_LAW else
        "sealed_after_fresh_P_calibration"
    )
    if (descriptor["required_payload_sha256"] != expected_sentinel
            and descriptor["required_payload_sha256"] != actual_digest):
        raise ValueError("Thermal recipe calibration digest differs from the external TRAIN receipt.")
    resolved = copy.deepcopy(recipe)
    bound_descriptor = {
        "receipt_path": descriptor["receipt_path"],
        "required_payload_sha256": actual_digest,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if recipe.get("execution_protocol") == FULL_TRAIN_FOLLOWUP_PROTOCOL:
        bound_descriptor["dataset_protocol"] = descriptor["dataset_protocol"]
    resolved["calibration_receipt"] = bound_descriptor
    resolved["auxiliary_calibration"] = receipt
    resolved["response_coefficient"] = float(receipt["response_coefficient"])
    resolved["operator_coefficient"] = float(receipt["operator_coefficient"])
    return validate_recipe(resolved)


def _check_fresh_calibration_destinations(recipe: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Refuse to replace either half of a previously sealed calibration audit."""
    receipt_path = (ROOT / recipe["calibration_receipt"]["receipt_path"]).resolve()
    invocation_path = Path(output_dir).expanduser().resolve() / "calibration_invocation.json"
    existing = [path for path in (receipt_path, invocation_path) if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Fresh-P calibration outputs are immutable and already exist: {joined}; reuse the sealed receipt."
        )
    return receipt_path, invocation_path


def training_source_identity(*, require_clean: bool,
                             flow_readout_law: str | None = None,
                             full_train_followup: bool = False) -> dict[str, Any]:
    repo_root = ROOT.parent
    source_files = _recovery_source_files(flow_readout_law, full_train_followup=full_train_followup)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *source_files],
        cwd=repo_root, check=True, text=True, capture_output=True,
    ).stdout.splitlines()
    if require_clean and dirty:
        raise RuntimeError(
            "Recovery execution requires committed, clean training sources before calibration/fitting; "
            f"dirty source paths: {dirty}"
        )
    hashes = {}
    for relative in source_files:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Recovery source identity is missing {relative}.")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *source_files],
        cwd=repo_root, check=True, text=True, capture_output=True,
    ).stdout.strip()
    body = {"source_git_commit": source_commit, "source_sha256": hashes}
    body["source_set_sha256"] = hashlib.sha256(json.dumps(
        hashes, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return body


def repository_head_observed() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=True,
                          text=True, capture_output=True).stdout.strip()


def seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build(recipe: dict[str, Any], *, device: str, calibration_only: bool = False,
          require_clean_source: bool = False):
    from honf_runtime.unified_training import EngineConfig, ScheduleSpec, SelectionPolicy, TrainingEngine
    seed_all(recipe["seed"])
    if recipe["task"] == "thermal":
        from channelthermal.training.joint_task import build_thermal_joint_task as factory
    else:
        from windfarm.training.joint_task import build_wind_joint_task as factory
    task_options = {name: recipe[name] for name in ("response_coefficient", "operator_coefficient", "auxiliary_calibration",
                                                 "locality_prior_strength", "validation_scope", "weight_decay",
                                                 "native_sampling_protocol", "flow_readout_law",
                                                 "full_train_followup1000")
                    if name in recipe}
    if "optimizer_schedule" in recipe:
        task_options["optimizer_schedule"] = ScheduleSpec(total_epochs=recipe["total_epochs"],
                                                          **recipe["optimizer_schedule"])
    if calibration_only:
        task_options.pop("auxiliary_calibration", None)
        task_options["calibration_only"] = True
    if "collective_width" in recipe:
        task_options["collective_width"] = recipe["collective_width"]
        task_options["max_sources"] = recipe["max_sources"]
    if recipe["task"] == "wind" and "environment_token_shape" in recipe:
        task_options["environment_token_shape"] = recipe["environment_token_shape"]
    model, provider = factory(
        mode=recipe["mode"], device=device, seed=recipe["seed"], hidden=recipe["hidden"],
        message=recipe["message"], regional_anchors=recipe["regional_anchors"], depth=recipe["depth"],
        receiver_tile=recipe["receiver_tile"], primary_queries=recipe["primary_queries"],
        microbatch_size=recipe["microbatch_cases"], effective_batch_size=recipe["effective_cases"],
        formal_full=recipe["formal_full"], total_epochs=recipe["total_epochs"],
        **task_options,
    )
    declared_engine = recipe.get("engine_schedule", {})
    config = EngineConfig(
        seed=recipe["seed"], microbatch_cases=recipe["microbatch_cases"],
        effective_cases=recipe["effective_cases"], total_epochs=recipe["total_epochs"],
        training_mode=declared_engine.get("training_mode", "joint"),
        sampling_version=declared_engine.get("sampling_version", "case_epoch_v1"),
        monitor_every=declared_engine.get("monitor_every", 100),
        warmup_epochs=declared_engine.get("warmup_epochs", 20),
        open_through_epoch=declared_engine.get("open_through_epoch", 20),
        soft_through_epoch=declared_engine.get("soft_through_epoch", 20),
        latest_every=declared_engine.get("latest_every", 100),
        curve_every=declared_engine.get("curve_every", 100),
        gradient_clip=recipe.get("gradient_clip_norm", 1.0),
        checkpoint_epochs=(tuple(recipe["checkpoint_epochs"]) if "checkpoint_epochs" in recipe else None),
        write_initial_artifacts=recipe.get("write_initial_artifacts", False),
    )
    engine = TrainingEngine(config, device=device, selection=SelectionPolicy(field_metric="field_score"))
    flow_readout_law = recipe.get("flow_readout_law")
    source_options = {"require_clean": require_clean_source}
    if flow_readout_law is not None:
        source_options["flow_readout_law"] = flow_readout_law
    if recipe.get("execution_protocol") == FULL_TRAIN_FOLLOWUP_PROTOCOL:
        source_options["full_train_followup"] = True
    source_identity = training_source_identity(**source_options)
    identity = {
        "workflow": "joint_regional_fields_v1", "recipe": recipe,
        "training_source_identity": source_identity,
        "initial_model_state_sha256": model_digest(model),
        "initialization": "fresh_all_trainable", "external_learned_field_files": [],
        "solver_attempts": 0, "WindTEST_targets": "locked",
    }
    return model, provider, engine, identity


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_json_once(path: Path, payload: Any) -> None:
    """Publish an immutable JSON artifact without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summary(model: Any, provider: Any, engine: Any, identity: dict[str, Any]) -> dict[str, Any]:
    from dataclasses import asdict
    groups = provider.optimizer_groups(model, identity["recipe"]["mode"], "joint")
    covered = [name for group in groups for name in group.parameter_names]
    active = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if len(covered) != len(set(covered)) or set(covered) != active:
        raise ValueError("Declared joint optimizer groups omit or duplicate an active parameter.")
    return {
        "identity": identity, "provider_identity": provider.identity_payload(),
        "invocation_provenance": {"repository_head_observed": repository_head_observed()},
        "preparation": provider.preparation_summary(),
        "engine_config": asdict(engine.config), "selection_policy": asdict(engine.selection),
        "active_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "optimizer_parameter_coverage": "all active parameters exactly once",
        "optimizer_schedule_contract": [asdict(group) for group in groups],
        "optimizer_started": False, "formal_training_started": False,
    }


def validate_manual_recipe(model: Any, provider: Any, engine: Any, identity: dict[str, Any]) -> dict[str, Any]:
    """Inert fullTRAIN identity/head validation; no optimizer is constructed."""
    import torch

    from honf_runtime.unified_training import SamplingKey, _sampling_dataset_id
    result = summary(model, provider, engine, identity)
    cases = tuple(provider.epoch_cases(1, identity["recipe"]["seed"]))
    if not cases:
        raise ValueError("Manual recipe has no TRAIN cases.")
    dataset_id = _sampling_dataset_id(provider.identity_payload())
    key = SamplingKey(identity["recipe"]["seed"], 1, 0, 0, "joint", identity["recipe"]["mode"],
                      sampling_version="case_epoch_v1", dataset_id=dataset_id)
    checks = []
    for case in dict.fromkeys((cases[0], cases[-1])):
        batch = provider.make_batch((case,), key)
        model.eval()
        with torch.no_grad():
            predictions, auxiliary = provider.predict_native(
                model, provider.make_scene(batch.scene_inputs), batch.receivers,
                identity["recipe"]["mode"], "joint", epoch=1, temperature=1.0)
            terms = provider.validation_loss_terms(predictions, batch.targets, auxiliary,
                                                   batch=batch, arm=identity["recipe"]["mode"])
            if not terms or not all(bool(torch.isfinite(term.numerator)) for term in terms.values()):
                raise FloatingPointError("Manual recipe native physical head check is nonfinite.")
        checks.append({"case_key": case, "finite_physical_head_terms": sorted(terms)})
    result["native_forward_checks"] = checks
    result["optimizer_instantiated"] = False
    result["optimizer_updates"] = 0
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "dry-run", "validate-followup", "calibrate-thermal", "start", "resume", "branch"))
    parser.add_argument("--task", choices=("thermal", "wind"))
    parser.add_argument("--mode", choices=MODES, default="J-H")
    parser.add_argument("--recipe-json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--message", type=int, default=128)
    parser.add_argument("--regional-anchors", type=int)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--receiver-tile", type=int, default=512)
    parser.add_argument("--primary-queries", type=int)
    parser.add_argument("--microbatch-cases", type=int)
    parser.add_argument('--locality-prior-strength', type=float, default=0.0)
    parser.add_argument('--branch-from-checkpoint')
    parser.add_argument('--revision-declaration')
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--manual-formal-launch", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    recipe = resolved_recipe(args)
    full_followup = recipe.get("execution_protocol") == FULL_TRAIN_FOLLOWUP_PROTOCOL
    if args.command == "validate-followup":
        if not full_followup:
            parser.error("validate-followup accepts only the explicit full-population follow-up protocol.")
        validate_full_train_followup_recipe(recipe, command="prepare")
        print(json.dumps({
            "status": "metadata_only_followup_recipe_valid",
            "run_id": recipe["run_id"],
            "execution_protocol": recipe["execution_protocol"],
            "dataset_protocol": recipe["dataset_protocol"],
            "optimizer_horizon": recipe["total_epochs"],
            "approved_stop_after": recipe["approved_stop_after"],
            "root_protocol_go": recipe["root_protocol_go"],
            "dataset_access": "none",
            "optimizer_updates": 0,
        }, indent=2, sort_keys=True), flush=True)
        return 0
    if full_followup:
        resume_payload = None
        if args.command == "resume":
            if not args.output_dir:
                parser.error("Full-population exact resume requires the original --output-dir.")
            from honf_runtime.compat import load_trusted_checkpoint
            from honf_runtime.run_layout import resolve_checkpoint
            latest = resolve_checkpoint(Path(args.output_dir).expanduser().resolve(), "latest")
            resume_payload = load_trusted_checkpoint(latest, map_location="cpu")
        validate_full_train_followup_recipe(
            recipe, command=args.command, stop_after=args.stop_after,
            resume_checkpoint_payload=resume_payload,
        )
    if args.command == "calibrate-thermal":
        if (recipe["task"] != "thermal" or recipe["mode"] != "P" or recipe["formal_full"]
                or recipe.get("auxiliary_calibration") is not None):
            parser.error("calibrate-thermal requires an unresolved fresh Thermal P recipe only.")
        if not args.output_dir:
            parser.error("calibrate-thermal requires --output-dir for its auditable invocation record.")
        calibration_receipt_path, calibration_invocation_path = _check_fresh_calibration_destinations(
            recipe, args.output_dir
        )
    recovery_execution = recipe["mode"] in ("P", "P-G", "P-H") and args.command in (
        "calibrate-thermal", "start", "resume", "branch")
    if args.command in ("start", "resume", "branch"):
        if not args.output_dir or args.stop_after is None:
            parser.error("Optimization requires --output-dir and --stop-after.")
        if recipe["formal_full"] and not full_followup and not args.manual_formal_launch:
            parser.error("Formal optimization requires explicit --manual-formal-launch; preparation is inert.")
        if not 1 <= args.stop_after <= recipe["total_epochs"]:
            parser.error("Requested stop exceeds the declared horizon.")
    if args.command == 'branch':
        if (recipe['formal_full'] or recipe['mode'] != 'J-H'
                or not args.branch_from_checkpoint or not args.revision_declaration):
            parser.error('A core revision requires a development J-H recipe, parent checkpoint and sealed declaration.')
    elif args.branch_from_checkpoint or args.revision_declaration:
        parser.error('Core revision parent/declaration options are valid only for branch.')
    if args.command in ("prepare", "dry-run"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        args.device = "cpu"
    import torch
    torch.set_num_threads(args.threads)
    started = time.time()
    model, provider, engine, identity = build(
        recipe, device=args.device,
        calibration_only=(args.command == "calibrate-thermal"),
        require_clean_source=recovery_execution,
    )
    setup_seconds = time.time() - started
    if args.command == "calibrate-thermal":
        receipt = dict(provider.calibrate_auxiliary_coefficients())
        receipt["training_source_identity"] = identity["training_source_identity"]
        receipt["payload_sha256"] = _canonical_payload_sha256(receipt)
        receipt_path = calibration_receipt_path
        write_json_once(receipt_path, receipt)
        payload = {
            "status": "train_only_calibration_complete_no_optimizer_updates",
            "receipt_path": str(receipt_path),
            "receipt_file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "receipt_payload_sha256": receipt["payload_sha256"],
            "response_coefficient": receipt["response_coefficient"],
            "operator_coefficient": receipt["operator_coefficient"],
            "primary_partition": receipt["primary_partition"],
            "response_family_partition": receipt["response_family_partition"],
            "calibration_case_ids": receipt["calibration_case_ids"],
            "validation_values_read": receipt["validation_values_read"],
            "validation_cases_materialized": receipt["validation_cases_materialized"],
            "development_response_families_loaded": receipt["development_response_families_loaded"],
            "optimizer_steps": receipt["optimizer_steps"],
            "cold_setup_seconds": setup_seconds,
            "repository_head_observed": repository_head_observed(),
        }
        output = Path(args.output_dir).expanduser().resolve()
        if output / "calibration_invocation.json" != calibration_invocation_path:
            raise RuntimeError("Calibration invocation path changed after the immutable-output check.")
        write_json_once(calibration_invocation_path, payload | {
            "training_source_identity": identity["training_source_identity"],
            "receipt": receipt,
        })
    elif args.command == "prepare":
        payload = {"status": "prepared_only", "cold_setup_seconds": setup_seconds,
                   **summary(model, provider, engine, identity)}
        if args.output_dir:
            write_json(Path(args.output_dir) / "preparation.json", payload)
    elif args.command == "dry-run":
        if full_followup:
            # Full-population follow-up validation never advances an optimizer.
            payload = {
                "status": "full_train_followup_inert_validation",
                "cold_setup_seconds": setup_seconds,
                "optimizer_updates": 0,
                "preflight_update_executed": False,
                "formal_training_started": False,
                "checkpoint_written": False,
            }
        elif recipe["formal_full"]:
            # FullTRAIN preparation is deliberately inert. No formal optimizer
            # is instantiated, even for a disposable update.
            payload = {"status": "manual_formal_recipe_validated_inert",
                       **validate_manual_recipe(model, provider, engine, identity)}
        else:
            payload = {"status": "disposable_cpu_joint_update", "cold_setup_seconds": setup_seconds,
                       "update": engine.preflight_one_update(model, provider, arm=recipe["mode"]),
                       "formal_training_started": False, "checkpoint_written": False}
    else:
        from honf_runtime.run_layout import resolve_checkpoint
        output = Path(args.output_dir).resolve()
        if args.command in ('start', 'branch'):
            write_json(output / "resolved_recipe.json", {
                "recipe": recipe,
                "training_source_identity": identity["training_source_identity"],
                "repository_head_observed": repository_head_observed(),
            })
            write_json(output / "preparation.json", {"cold_setup_seconds": setup_seconds,
                       **summary(model, provider, engine, identity)})
        if torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        payload = engine.fit(
            model, provider, output, identity=identity, arm=recipe["mode"],
            stop_after=args.stop_after,
            resume_checkpoint=resolve_checkpoint(output, "latest") if args.command == "resume" else None,
            branch_from_checkpoint=args.branch_from_checkpoint if args.command == 'branch' else None,
            joint_core_revision=json.loads(Path(args.revision_declaration).read_text())
                if args.command == 'branch' else None,
        )
        payload["cold_setup_seconds"] = setup_seconds
        payload["outer_elapsed_seconds"] = time.time() - started
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize()
            payload["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            payload["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(output / f"invocation_{args.command}_e{args.stop_after:04d}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
