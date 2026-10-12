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
import uuid
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

MODES = ("J-direct", "J-geometry", "J-H", "P", "P-G", "P-H", "P-R", "P-R2")
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
    "fresh_matched_comparison", "comparison_protocol", "field_selection", "native_validation_policy",
    "conditional_confirmation",
}
FRESH_MATCHED_COMPARISON_PROTOCOL = "fresh_segmented_multiview_2500_v1"
FRESH_MATCHED_NATIVE_VALIDATION_POLICY = "consolidated_route_flags_v1"
FRESH_MATCHED_MODES = ("P", "P-R", "P-R2")
FRESH_MATCHED_REVIEW_EPOCHS = (100, 500, 1000, 1500, 2500)
FRESH_MATCHED_SELECTION_EPOCHS = tuple(range(100, 2501, 100))

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
FRESH_MATCHED_SOURCE_FILES = (
    "HONF_Proj/src/honf_forward_core/interface_fields/native_interaction_routing.py",
    "HONF_Proj/src/config_core/forward/joint_regional/fresh_multiview_thermal_v1.json",
    "HONF_Proj/src/config_core/forward/joint_regional/fresh_multiview_wind_v1.json",
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
        legacy_curl_mode = recipe["mode"] in ("P", "P-G", "P-H")
        fresh_matched_curl_mode = (
            recipe.get("fresh_matched_comparison") is True
            and recipe["mode"] in FRESH_MATCHED_MODES
        )
        if (recipe["task"] != "thermal"
                or not (legacy_curl_mode or fresh_matched_curl_mode)
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
    if type(anchors) is not int or anchors < (0 if recipe.get("mode") in ("P", "P-R", "P-R2") else 1):
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
    fresh_matched = recipe.get("fresh_matched_comparison", False)
    if type(fresh_matched) is not bool:
        raise ValueError("Fresh matched comparison flag must be a boolean.")
    if "conditional_confirmation" in recipe and not fresh_matched:
        raise ValueError("Conditional confirmation belongs only to the separate fresh matched protocol.")
    if recipe["mode"] in ("P-R", "P-R2") and not fresh_matched:
        raise ValueError("Fresh P-R/P-R2 recipes require the explicit matched-initialization protocol.")
    recovery = recipe["mode"] in ("P", "P-G", "P-H") and not fresh_matched
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
            and not fresh_matched
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
    elif fresh_matched:
        _validate_fresh_matched_recipe(recipe)
    elif recovery:
        _validate_recovery_recipe(recipe)
    return recipe


def _fresh_matched_model_contract(task: str, mode: str) -> dict[str, Any]:
    flow_readout_law = NATIVE_CURL_THERMAL_LAW if task == "thermal" else None
    contract = _recovery_model_contract(task, "P", flow_readout_law)
    if mode == "P":
        contract["native_routing"] = {"enabled": False, "mode": "P"}
    elif mode == "P-R":
        contract["native_routing"] = {
            "schema": "native_interaction_routing_v1", "mode": "P-R",
            "policy": "learned_density", "views": 1,
            "score_hidden": 32, "score_bound": 2.0,
        }
    elif mode == "P-R2":
        contract["native_routing"] = {
            "schema": "native_interaction_routing_v2", "mode": "P-R2",
            "policy": "learned_density", "views": 2, "feature_blocks": [64, 64],
            "score_hidden": 32, "score_bound": 2.0,
        }
    else:
        raise ValueError("Fresh matched model contract requires P, P-R or P-R2.")
    return contract


def _validate_fresh_matched_recipe(recipe: dict[str, Any]) -> None:
    from honf_runtime.unified_training import fresh_matched_recipe_seed

    task, mode = recipe["task"], recipe["mode"]
    if (recipe.get("comparison_protocol") != FRESH_MATCHED_COMPARISON_PROTOCOL
            or mode not in FRESH_MATCHED_MODES or recipe.get("formal_full") is not False
            or recipe.get("full_train_followup1000") is True):
        raise ValueError("Fresh matched recipes must use the segmented 2500-epoch P/R1/R2 protocol.")
    run_id = recipe.get("run_id")
    prefix = "TMV1-" if task == "thermal" else "WMV1-"
    if (not isinstance(run_id, str) or not run_id.startswith(prefix)
            or len(run_id) != len(prefix) + 32
            or any(character not in "0123456789abcdef" for character in run_id[len(prefix):])):
        raise ValueError("Fresh matched recipe run_id must come from the new UUID allocator.")
    expected = {
        "seed": fresh_matched_recipe_seed(recipe, verify_decision=True),
        "hidden": 128,
        "message": 128,
        "regional_anchors": 0,
        "collective_width": 64,
        "depth": 2,
        "receiver_tile": 512,
        "primary_queries": 1024 if task == "thermal" else 4096,
        "microbatch_cases": 8,
        "effective_cases": 48 if task == "thermal" else 24,
        "total_epochs": 2500,
        "initialization": "fresh_all_trainable",
        "launch_policy": "bounded_development",
        "dataset_protocol": "fixed25_v1" if task == "thermal" else "wind_shared_fixed24_v1",
        "formal_full": False,
        "environment_token_shape": [24, 8] if task == "thermal" else [2, 2, 2],
        "max_sources": 12 if task == "thermal" else 30,
        "model_contract": _fresh_matched_model_contract(task, mode),
        "objective_contract": _recovery_objective_contract(task),
        "engine_schedule": RECOVERY_ENGINE_SCHEDULE,
        "optimizer_name": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1.0e-8,
        "optimizer_schedule": RECOVERY_OPTIMIZER_SCHEDULE,
        "weight_decay": 1.0e-4 if task == "thermal" else 1.0e-5,
        "gradient_clip_norm": 1.0,
        "checkpoint_epochs": RECOVERY_CHECKPOINT_EPOCHS,
        "write_initial_artifacts": False,
        "field_selection": {
            "metric": "field_score", "direction": "minimize",
            "eligible_epochs": list(FRESH_MATCHED_SELECTION_EPOCHS),
            "same_selector_for_all_arms": True,
        },
        "native_validation_policy": FRESH_MATCHED_NATIVE_VALIDATION_POLICY,
    }
    if task == "thermal":
        expected["flow_readout_law"] = NATIVE_CURL_THERMAL_LAW
    required = set(expected)
    if task == "thermal":
        required.add("calibration_receipt")
    missing = required - set(recipe)
    if missing:
        raise ValueError(f"Fresh matched recipe is missing sealed fields: {sorted(missing)}.")
    for name, value in expected.items():
        if recipe.get(name) != value:
            raise ValueError(f"Fresh matched recipe {name} differs from its fixed segmented protocol.")
    routing = recipe["model_contract"].get("native_routing")
    if mode == "P-R":
        if (not isinstance(routing, dict)
                or set(routing) != {"schema", "mode", "policy", "views", "score_hidden", "score_bound"}
                or type(routing.get("views")) is not int
                or type(routing.get("score_hidden")) is not int
                or type(routing.get("score_bound")) is not float):
            raise ValueError("Fresh R1 routing metadata must use the strict v1 scalar schema types.")
    elif mode == "P-R2" and (
            not isinstance(routing, dict)
            or set(routing) != {"schema", "mode", "policy", "views", "feature_blocks",
                                "score_hidden", "score_bound"}
            or type(routing.get("views")) is not int
            or type(routing.get("score_hidden")) is not int
            or type(routing.get("score_bound")) is not float
            or type(routing.get("feature_blocks")) is not list
            or any(type(value) is not int for value in routing["feature_blocks"])):
        raise ValueError("Fresh R2 routing metadata must use the strict v2 two-view schema types.")
    if "locality_prior_strength" in recipe and recipe["locality_prior_strength"] != 0:
        raise ValueError("Fresh P/R1/R2 recipes retain the zero-locality-prior source-preserving core.")
    if task == "thermal":
        descriptor = recipe["calibration_receipt"]
        if (not isinstance(descriptor, dict)
                or set(descriptor) not in ({"receipt_path", "required_payload_sha256"},
                                           {"receipt_path", "required_payload_sha256", "file_sha256"})
                or not isinstance(descriptor["receipt_path"], str)
                or not descriptor["receipt_path"]):
            raise ValueError("Fresh matched Thermal recipes require one shared TRAIN-only calibration receipt.")
        digest = descriptor["required_payload_sha256"]
        valid_digest = (isinstance(digest, str) and len(digest) == 64
                        and all(character in "0123456789abcdef" for character in digest))
        if digest not in ("sealed_after_fresh_P_native_curl_calibration",) and not valid_digest:
            raise ValueError("Thermal fresh comparison receipt needs the allocator sentinel or a sealed digest.")
        if "file_sha256" in descriptor and (
                not isinstance(descriptor["file_sha256"], str)
                or len(descriptor["file_sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in descriptor["file_sha256"])):
            raise ValueError("Thermal fresh comparison calibration file hash is malformed.")
        calibration = recipe.get("auxiliary_calibration")
        if digest == "sealed_after_fresh_P_native_curl_calibration" and calibration is not None:
            raise ValueError("Unresolved Thermal fresh P receipt cannot carry pre-calibrated objective weights.")


def fresh_matched_config(task: str) -> tuple[dict[str, Any], Path, str]:
    if task not in ("thermal", "wind"):
        raise ValueError("Fresh matched comparison task must be thermal or wind.")
    config_path = ROOT / "src/config_core/forward/joint_regional" / f"fresh_multiview_{task}_v1.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if (config.get("schema") != "fresh_segmented_multiview_comparison_config_v1"
            or config.get("task") != task or config.get("arms") != list(FRESH_MATCHED_MODES)
            or config.get("total_epochs") != 2500
            or config.get("review_epochs") != list(FRESH_MATCHED_REVIEW_EPOCHS)
            or config.get("checkpoint_epochs") != list(RECOVERY_CHECKPOINT_EPOCHS)
            or config.get("field_selection", {}).get("eligible_epochs")
            != list(FRESH_MATCHED_SELECTION_EPOCHS)
            or config.get("native_validation_policy") != FRESH_MATCHED_NATIVE_VALIDATION_POLICY
            or (task == "thermal" and config.get("flow_readout_law") != NATIVE_CURL_THERMAL_LAW)
            or (task == "wind" and "flow_readout_law" in config)):
        raise ValueError("Maintained fresh matched comparison config differs from its sealed protocol.")
    return config, config_path, hashlib.sha256(config_bytes).hexdigest()


def _fresh_matched_recipe(config: dict[str, Any], task: str, mode: str,
                          run_id: str, calibration_path: Path | None) -> dict[str, Any]:
    optimizer = config["optimizer"]
    recipe = {
        "schema_version": 1,
        "task": task,
        "mode": mode,
        "run_id": run_id,
        "seed": config["seed"],
        "hidden": config["hidden"],
        "message": config["message"],
        "regional_anchors": config["regional_anchors"],
        "depth": config["depth"],
        "receiver_tile": config["receiver_tile"],
        "primary_queries": config["primary_queries"],
        "microbatch_cases": config["microbatch_cases"],
        "effective_cases": config["effective_cases"],
        "formal_full": False,
        "total_epochs": config["total_epochs"],
        "initialization": "fresh_all_trainable",
        "launch_policy": "bounded_development",
        "dataset_protocol": config["dataset_protocol"],
        "collective_width": config["collective_width"],
        "max_sources": config["max_sources"],
        "environment_token_shape": config["environment_token_shape"],
        "model_contract": _fresh_matched_model_contract(task, mode),
        "objective_contract": _recovery_objective_contract(task),
        "engine_schedule": config["engine_schedule"],
        "optimizer_name": optimizer["name"],
        "optimizer_betas": optimizer["betas"],
        "optimizer_eps": optimizer["eps"],
        "gradient_clip_norm": optimizer["gradient_clip_norm"],
        "optimizer_schedule": optimizer["schedule"],
        "weight_decay": optimizer["weight_decay"],
        "checkpoint_epochs": config["checkpoint_epochs"],
        "write_initial_artifacts": config["write_initial_artifacts"],
        "fresh_matched_comparison": True,
        "comparison_protocol": FRESH_MATCHED_COMPARISON_PROTOCOL,
        "field_selection": config["field_selection"],
        "native_validation_policy": config["native_validation_policy"],
    }
    if task == "thermal":
        if calibration_path is None:
            raise ValueError("Thermal fresh comparison allocation needs its campaign-owned calibration path.")
        try:
            receipt_path = calibration_path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            receipt_path = str(calibration_path.resolve())
        recipe["calibration_receipt"] = {
            "receipt_path": receipt_path,
            "required_payload_sha256": "sealed_after_fresh_P_native_curl_calibration",
        }
        recipe["flow_readout_law"] = config["flow_readout_law"]
    return validate_recipe(recipe)


def allocate_fresh_matched_comparison(task: str, output_dir: str | Path) -> dict[str, Any]:
    """Create immutable recipes with new collision-checked IDs; create no model/run state."""
    config, config_path, config_sha256 = fresh_matched_config(task)
    campaign_root = Path(output_dir).expanduser().resolve()
    campaign_root.mkdir(parents=True, exist_ok=False)
    calibration_path = (campaign_root / "thermal_train_calibration.json") if task == "thermal" else None
    arms: dict[str, Any] = {}
    allocated_ids: set[str] = set()
    for mode in FRESH_MATCHED_MODES:
        prefix = "TMV1-" if task == "thermal" else "WMV1-"
        while True:
            run_id = prefix + uuid.uuid4().hex
            run_dir = campaign_root / "runs" / run_id
            if run_id not in allocated_ids and not run_dir.exists():
                break
        allocated_ids.add(run_id)
        recipe = _fresh_matched_recipe(config, task, mode, run_id, calibration_path)
        recipe_path = campaign_root / "recipes" / f"{mode.replace('-', '_').lower()}.json"
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_once(recipe_path, recipe)
        arms[mode] = {
            "run_id": run_id,
            "recipe_path": str(recipe_path),
            "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
            "output_dir": str(run_dir),
        }
    manifest = {
        "schema": FRESH_MATCHED_COMPARISON_PROTOCOL,
        "task": task,
        "dataset_protocol": config["dataset_protocol"],
        "seed": config["seed"],
        "source_config_path": str(config_path),
        "source_config_sha256": config_sha256,
        "campaign_root": str(campaign_root),
        "review_epochs": config["review_epochs"],
        "checkpoint_epochs": config["checkpoint_epochs"],
        "field_selection": config["field_selection"],
        "thermal_calibration_path": None if calibration_path is None else str(calibration_path),
        "calibration_output_dir": (None if task != "thermal"
                                   else str(campaign_root / "calibration-preparation")),
        "arms": arms,
        "optimizer_updates": 0,
        "optimizer_state_created": False,
    }
    write_json_once(campaign_root / "comparison_manifest.json", manifest)
    return manifest


def _read_fresh_comparison_manifest(
    path: str | Path, *, require_fresh_run_dirs: bool = False,
) -> tuple[dict[str, Any], Path]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != FRESH_MATCHED_COMPARISON_PROTOCOL
            or manifest.get("task") not in ("thermal", "wind")
            or set(manifest.get("arms", {})) != set(FRESH_MATCHED_MODES)):
        raise ValueError("Fresh matched comparison manifest has an unsupported schema or arm inventory.")
    config, config_path, config_sha256 = fresh_matched_config(manifest["task"])
    if (manifest.get("source_config_path") != str(config_path)
            or manifest.get("source_config_sha256") != config_sha256
            or manifest.get("dataset_protocol") != config["dataset_protocol"]
            or manifest.get("review_epochs") != config["review_epochs"]
            or manifest.get("checkpoint_epochs") != config["checkpoint_epochs"]
            or manifest.get("field_selection") != config["field_selection"]):
        raise ValueError("Fresh matched campaign manifest differs from its maintained config.")
    campaign_root = Path(manifest["campaign_root"]).resolve()
    if campaign_root != manifest_path.parent:
        raise ValueError("Fresh matched manifest path differs from its campaign root.")
    ids = []
    for mode in FRESH_MATCHED_MODES:
        arm = manifest["arms"][mode]
        recipe_path = Path(arm["recipe_path"]).resolve()
        if (recipe_path.parent != campaign_root / "recipes"
                or not recipe_path.is_file()
                or hashlib.sha256(recipe_path.read_bytes()).hexdigest() != arm["recipe_sha256"]):
            raise ValueError(f"Fresh matched {mode} recipe is missing or changed after allocation.")
        recipe = read_recipe(recipe_path)
        if recipe["mode"] != mode or recipe["run_id"] != arm["run_id"]:
            raise ValueError(f"Fresh matched {mode} recipe differs from its allocated run identity.")
        output_dir = Path(arm["output_dir"]).resolve()
        if output_dir != campaign_root / "runs" / arm["run_id"]:
            raise ValueError(f"Fresh matched {mode} output path is not owned by its allocated run ID.")
        if require_fresh_run_dirs and output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Fresh matched allocated output is already nonempty: {output_dir}")
        ids.append(arm["run_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Fresh matched arm run IDs must be unique.")
    return manifest, manifest_path


def _native_score_trunk_digest(model: Any) -> str:
    digest = hashlib.sha256()
    tensors = [
        (name, value) for name, value in model.state_dict().items()
        if name.startswith("core.native_routing.score_mlps.") and not name.endswith(".2.weight")
    ]
    if not tensors:
        raise ValueError("Native learned-density model is missing its score trunk inventory.")
    for name, tensor in sorted(tensors):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
        digest.update(tensor.detach().to(device="cpu").contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _provider_shared_comparison_binding(provider_identity: dict[str, Any]) -> dict[str, Any]:
    if provider_identity.get("task") == "ThermalChannel":
        names = ("manifest_fingerprint", "training_case_ids_sha256", "validation_case_ids_sha256",
                 "normalization_stats_sha256", "response_scales", "native_query_budget",
                 "loss_weights", "auxiliary_calibration")
        return {name: provider_identity.get(name) for name in names}
    if provider_identity.get("dataset") == "WindFarm":
        objective = provider_identity.get("objective", {})
        training = provider_identity.get("training", {})
        return {
            "subset_manifest_sha256": provider_identity.get("subset_manifest_sha256"),
            "train_row_indices_sha256": provider_identity.get("train_row_indices_sha256"),
            "validation_row_indices_sha256": provider_identity.get("validation_row_indices_sha256"),
            "normalizer_and_background_source": provider_identity.get("normalizer_and_background_source"),
            "component_scales_mps": objective.get("component_scales_mps"),
            "scale_rule": objective.get("scale_rule"),
            "scale_source_sha256": objective.get("scale_source_sha256"),
            "scale_calibration": objective.get("scale_calibration"),
            "scale_cache_binding": objective.get("scale_cache_binding"),
            "role_query_counts": objective.get("role_query_counts"),
            "primary_queries_per_case": training.get("primary_queries_per_case"),
        }
    raise ValueError("Fresh comparison provider identity has an unsupported dataset.")


def prepare_fresh_matched_comparison(manifest_path: str | Path) -> dict[str, Any]:
    """Build all arms on CPU, verify common starts/scales, and seal preparation only."""
    manifest, manifest_path = _read_fresh_comparison_manifest(manifest_path, require_fresh_run_dirs=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)
    prepared: dict[str, Any] = {}
    identities: dict[str, dict[str, Any]] = {}
    providers: dict[str, dict[str, Any]] = {}
    initializations: dict[str, dict[str, Any]] = {}
    models: dict[str, Any] = {}
    for mode in FRESH_MATCHED_MODES:
        arm = manifest["arms"][mode]
        recipe_path = Path(arm["recipe_path"])
        recipe = read_recipe(recipe_path)
        if recipe["task"] == "thermal":
            recipe = _bind_calibration_receipt(recipe)
        model, provider, engine, identity = build(
            recipe, device="cpu", require_clean_source=True, include_dev_targets=False)
        if recipe["task"] == "thermal":
            if (getattr(provider, "dev_targets_materialized", None) is not False
                    or tuple(getattr(provider, "validation_cases", ()))):
                raise RuntimeError("Fresh Thermal preparation must keep all DEV targets unmaterialized.")
        elif getattr(provider, "_validation_cache", None) is not None:
            raise RuntimeError("Fresh Wind preparation must leave its lazy DEV validation cache empty.")
        model_summary = summary(model, provider, engine, identity)
        providers[mode] = dict(provider.identity_payload())
        identities[mode] = identity
        initializations[mode] = identity["fresh_initialization"]
        models[mode] = model
        prepared[mode] = {
            "run_id": arm["run_id"],
            "recipe_sha256": arm["recipe_sha256"],
            "initial_model_state_sha256": identity["initial_model_state_sha256"],
            "common_initialization_sha256": initializations[mode]["common_initialization_sha256"],
            "provider_scale_binding_sha256": initializations[mode]["provider_scale_binding_sha256"],
            "sampler_namespace": initializations[mode]["sampler_namespace"],
            "optimizer_updates": 0,
            "optimizer_started": False,
            "active_parameter_count": model_summary["active_parameter_count"],
            "optimizer_schedule_contract": model_summary["optimizer_schedule_contract"],
        }
    common_digests = {item["common_initialization_sha256"] for item in prepared.values()}
    scale_digests = {item["provider_scale_binding_sha256"] for item in prepared.values()}
    provider_bindings = {_canonical_json(_provider_shared_comparison_binding(identity))
                         for identity in providers.values()}
    sampler_namespaces = {_canonical_json(item["sampler_namespace"]) for item in prepared.values()}
    if len(common_digests) != 1:
        raise ValueError("P, R1 and R2 common initialized value tensors differ.")
    if len(scale_digests) != 1 or len(provider_bindings) != 1:
        raise ValueError("P, R1 and R2 do not share exact TRAIN normalization/calibration/objective bindings.")
    if len(sampler_namespaces) != 1:
        raise ValueError("P, R1 and R2 do not share the exact dataset/seed/query sampler namespace.")
    r1_trunk = _native_score_trunk_digest(models["P-R"])
    r2_trunk = _native_score_trunk_digest(models["P-R2"])
    if r1_trunk != r2_trunk:
        raise ValueError("Fresh R1/R2 typed score trunks do not share their private initialization seed.")
    for mode in ("P-R", "P-R2"):
        projection_names = [name for name, tensor in models[mode].state_dict().items()
                            if name.startswith("core.native_routing.score_mlps.") and name.endswith(".2.weight")]
        if not projection_names or any(bool(torch.count_nonzero(models[mode].state_dict()[name]))
                                       for name in projection_names):
            raise ValueError(f"Fresh {mode} score projections must initialize to exact zero.")
    preparation_root = manifest_path.parent / "preparation"
    preparation_root.mkdir(parents=True, exist_ok=True)
    for mode in FRESH_MATCHED_MODES:
        write_json_once(preparation_root / f"{mode.replace('-', '_').lower()}.json", {
            "status": "prepared_only",
            "identity": identities[mode],
            "provider_identity": providers[mode],
            "optimizer_updates": 0,
            "optimizer_started": False,
            "formal_training_started": False,
        })
    seal = {
        "schema": "fresh_matched_initialization_seal_v1",
        "comparison_manifest": str(manifest_path),
        "comparison_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "task": manifest["task"],
        "dataset_protocol": manifest["dataset_protocol"],
        "source_config_sha256": manifest["source_config_sha256"],
        "arms": prepared,
        "common_initialization_sha256": next(iter(common_digests)),
        "provider_scale_binding_sha256": next(iter(scale_digests)),
        "shared_provider_binding_sha256": hashlib.sha256(next(iter(provider_bindings)).encode()).hexdigest(),
        "sampler_namespace_sha256": hashlib.sha256(next(iter(sampler_namespaces)).encode()).hexdigest(),
        "r1_r2_score_trunk_sha256": r1_trunk,
        "optimizer_updates": 0,
        "optimizer_started": False,
        "formal_training_started": False,
        "training_launch_authorized": False,
    }
    write_json_once(manifest_path.parent / "comparison_seal.json", seal)
    return seal


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"), allow_nan=False)


def _fresh_audit_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_fresh_matched_streams(manifest_path: str | Path) -> dict[str, Any]:
    manifest, manifest_path = _read_fresh_comparison_manifest(manifest_path)
    seal_path = manifest_path.parent / "comparison_seal.json"
    if not seal_path.is_file():
        raise FileNotFoundError("Run prepare-matched before reviewing training stream audits.")
    seal = json.loads(seal_path.read_text())
    if seal.get("schema") != "fresh_matched_initialization_seal_v1":
        raise ValueError("Fresh matched preparation seal is missing or malformed.")
    if (seal.get("comparison_manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            or seal.get("task") != manifest["task"]
            or seal.get("dataset_protocol") != manifest["dataset_protocol"]
            or set(seal.get("arms", {})) != set(FRESH_MATCHED_MODES)):
        raise ValueError("Fresh matched preparation seal does not bind the current campaign manifest.")
    receipts: dict[str, Any] = {}
    history_hashes: dict[str, str] = {}
    identity_bindings: dict[str, Any] = {}
    fields = (
        "case_order_sha256", "case_visits", "optimizer_updates", "microbatches",
        "query_sampling_sha256", "query_sampling_by_update_sha256", "observed_mask_sha256",
        "mask_sha256_by_update", "observed_denominators_sha256",
        "observed_denominators_by_update", "observed_denominator_totals",
        "effective_cases", "microbatch_cases", "sampling_dataset_id", "sampler_namespace_sha256",
    )
    for mode in FRESH_MATCHED_MODES:
        arm = manifest["arms"][mode]
        output = Path(arm["output_dir"])
        identity_path = output / "experiment_identity.json"
        history_path = output / "history.json"
        if not identity_path.is_file() or not history_path.is_file():
            raise FileNotFoundError(f"Fresh matched {mode} lacks its saved identity or history.")
        identity = json.loads(identity_path.read_text())
        rows = json.loads(history_path.read_text())
        row = next((item for item in rows if int(item.get("epoch", -1)) == 1), None)
        audit = row.get("observed_stream_audit") if isinstance(row, dict) else None
        marker = identity.get("fresh_initialization")
        if (not isinstance(marker, dict) or not isinstance(audit, dict)
                or audit.get("schema") != "observed_fresh_training_stream_v1"
                or identity.get("recipe", {}).get("mode") != mode
                or identity.get("recipe", {}).get("fresh_matched_comparison") is not True
                or marker.get("active_mode") != mode):
            raise ValueError(f"Fresh matched {mode} epoch-1 observed stream receipt is absent.")
        preparation_arm = seal["arms"][mode]
        if (marker.get("common_initialization_sha256") != seal.get("common_initialization_sha256")
                or marker.get("provider_scale_binding_sha256") != seal.get("provider_scale_binding_sha256")
                or marker.get("sampler_namespace") != preparation_arm.get("sampler_namespace")
                or preparation_arm.get("run_id") != arm["run_id"]):
            raise ValueError(f"Fresh matched {mode} training identity differs from its prepared seal.")
        if not isinstance(row, dict):
            raise TypeError(f"Fresh matched {mode} history has no epoch-1 row.")
        update_count = row.get("optimizer_updates")
        case_visits = row.get("case_visits")
        microbatches = row.get("microbatches")
        if (type(update_count) is not int or update_count < 1
                or type(case_visits) is not int or case_visits < 1
                or type(microbatches) is not int or microbatches < update_count
                or type(audit.get("optimizer_updates")) is not int
                or type(audit.get("case_visits")) is not int
                or type(audit.get("microbatches")) is not int
                or type(audit.get("effective_cases")) is not int
                or type(audit.get("microbatch_cases")) is not int
                or not isinstance(audit.get("sampling_dataset_id"), str)
                or audit.get("optimizer_updates") != update_count
                or audit.get("case_visits") != case_visits
                or audit.get("microbatches") != microbatches
                or audit.get("effective_cases") != identity.get("engine_config", {}).get("effective_cases")
                or audit.get("microbatch_cases") != identity.get("engine_config", {}).get("microbatch_cases")
                or audit.get("sampling_dataset_id") != identity.get("sampling_dataset_id")
                or audit.get("sampler_namespace_sha256") != _fresh_audit_sha256(marker.get("sampler_namespace"))):
            raise ValueError(f"Fresh matched {mode} observed stream counters differ from the saved engine identity.")
        for name in ("case_order_sha256", "query_sampling_sha256", "observed_mask_sha256",
                     "observed_denominators_sha256", "sampler_namespace_sha256"):
            if not _is_sha256(audit.get(name)):
                raise ValueError(f"Fresh matched {mode} observed stream digest {name} is malformed.")
        if (row.get("case_order_sha256") != audit["case_order_sha256"]
                or row.get("query_sampling_sha256") != audit["query_sampling_sha256"]):
            raise ValueError(f"Fresh matched {mode} history summary differs from its observed streams.")
        query_updates = audit.get("query_sampling_by_update_sha256")
        mask_updates = audit.get("mask_sha256_by_update")
        denominator_updates = audit.get("observed_denominators_by_update")
        if not all(isinstance(value, list) and len(value) == update_count for value in (
                query_updates, mask_updates, denominator_updates)):
            raise ValueError(f"Fresh matched {mode} per-update stream evidence has the wrong length.")
        expected_indices = list(range(update_count))
        if (not all(isinstance(item, dict) and type(item.get("update_index")) is int
                    for item in query_updates + mask_updates + denominator_updates)
                or [item.get("update_index") for item in query_updates if isinstance(item, dict)] != expected_indices
                or [item.get("update_index") for item in mask_updates if isinstance(item, dict)] != expected_indices
                or [item.get("update_index") for item in denominator_updates if isinstance(item, dict)]
                != expected_indices
                or not all(_is_sha256(item.get("sha256")) for item in query_updates if isinstance(item, dict))
                or not all(_is_sha256(item.get("sha256")) for item in mask_updates if isinstance(item, dict))):
            raise ValueError(f"Fresh matched {mode} per-update stream digests are malformed or reordered.")
        digest_sequence = hashlib.sha256(json.dumps([item["sha256"] for item in query_updates], separators=(",", ":"),
                                             allow_nan=False).encode("utf-8")).hexdigest()
        if digest_sequence != audit["query_sampling_sha256"]:
            raise ValueError(f"Fresh matched {mode} query stream aggregate differs from its update hashes.")
        if _fresh_audit_sha256(mask_updates) != audit["observed_mask_sha256"]:
            raise ValueError(f"Fresh matched {mode} mask stream aggregate differs from its update hashes.")
        if _fresh_audit_sha256(denominator_updates) != audit["observed_denominators_sha256"]:
            raise ValueError(f"Fresh matched {mode} denominator aggregate differs from its update records.")
        totals: dict[str, float] = {}
        for item in denominator_updates:
            if not isinstance(item, dict) or not isinstance(item.get("denominators"), dict):
                raise TypeError(f"Fresh matched {mode} denominator update record is malformed.")
            for name, value in item["denominators"].items():
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Fresh matched {mode} denominator {name} is not finite and nonnegative.")
                totals[name] = totals.get(name, 0.0) + float(value)
        if audit.get("observed_denominator_totals") != totals:
            raise ValueError(f"Fresh matched {mode} denominator totals differ from the observed update records.")
        receipts[mode] = {name: audit[name] for name in fields}
        history_hashes[mode] = hashlib.sha256(history_path.read_bytes()).hexdigest()
        identity_bindings[mode] = {
            "common_initialization_sha256": marker["common_initialization_sha256"],
            "provider_scale_binding_sha256": marker["provider_scale_binding_sha256"],
            "sampler_namespace": marker["sampler_namespace"],
        }
    reference = receipts["P"]
    differences = {
        mode: [name for name in fields if receipts[mode][name] != reference[name]]
        for mode in ("P-R", "P-R2")
    }
    if any(differences.values()):
        raise ValueError(f"Observed epoch-1 case/query/mask/denominator streams differ across arms: {differences}.")
    receipt = {
        "schema": "fresh_matched_observed_stream_comparison_v1",
        "comparison_manifest": str(manifest_path),
        "comparison_seal_sha256": hashlib.sha256(seal_path.read_bytes()).hexdigest(),
        "task": manifest["task"],
        "dataset_protocol": manifest["dataset_protocol"],
        "arms": {
            mode: {
                "run_id": manifest["arms"][mode]["run_id"],
                "history_sha256": history_hashes[mode],
                "observed_stream_audit": receipts[mode],
                "identity_binding": identity_bindings[mode],
            }
            for mode in FRESH_MATCHED_MODES
        },
        "matched": True,
        "optimizer_updates_observed": reference["optimizer_updates"],
        "case_visits_observed": reference["case_visits"],
    }
    write_json_once(manifest_path.parent / "observed_stream_comparison.json", receipt)
    return receipt


def forecast_fresh_matched_run(output_dir: str | Path, *, horizon: int = 2500) -> dict[str, Any]:
    """Forecast full epochs from observed train and full-DEV validation durations."""
    import numpy as np

    output = Path(output_dir).expanduser().resolve()
    identity_path, history_path = output / "experiment_identity.json", output / "history.json"
    if not identity_path.is_file() or not history_path.is_file():
        raise FileNotFoundError("A fresh-run forecast requires a saved identity and completed epoch history.")
    identity = json.loads(identity_path.read_text())
    marker = identity.get("fresh_initialization")
    if not isinstance(marker, dict):
        raise TypeError("Forecast target is not an explicitly fresh matched run.")
    recipe = identity.get("recipe", {})
    if (recipe.get("fresh_matched_comparison") is not True or recipe.get("total_epochs") != horizon
            or recipe.get("comparison_protocol") != FRESH_MATCHED_COMPARISON_PROTOCOL):
        raise ValueError("Forecast horizon or recipe differs from the fresh segmented comparison.")
    rows = json.loads(history_path.read_text())
    if not rows:
        raise ValueError("Fresh matched run has no completed epoch rows to forecast from.")
    training_rows = [float(row["train_seconds"]) for row in rows if row.get("train_seconds") is not None]
    validation_rows = [float(row["validation_seconds"]) for row in rows
                       if row.get("validation_seconds") is not None]
    if not training_rows or not validation_rows:
        return {
            "status": "incomplete_forecast_inputs",
            "reason": "Need measured TRAIN and full-DEV validation durations from at least one review epoch.",
            "completed_epoch": max(int(row["epoch"]) for row in rows),
            "optimizer_updates": sum(int(row["optimizer_updates"]) for row in rows),
            "case_visits": sum(int(row["case_visits"]) for row in rows),
            "forecast_seconds": None,
        }
    recent_training = training_rows[-min(len(training_rows), 20):]
    recent_validation = validation_rows[-min(len(validation_rows), 5):]
    training_epoch_seconds = float(np.median(recent_training))
    validation_review_seconds = float(np.median(recent_validation))
    completed_epoch = max(int(row["epoch"]) for row in rows)
    remaining_epochs = max(0, horizon - completed_epoch)
    review_count = sum(1 for epoch in FRESH_MATCHED_SELECTION_EPOCHS
                       if completed_epoch < epoch <= horizon)
    last = rows[-1]
    best_row = min((row for row in rows if row.get("validation", {}).get("field_score") is not None),
                   key=lambda row: float(row["validation"]["field_score"]), default=None)
    provider = identity.get("provider_identity", {})
    if provider.get("task") == "ThermalChannel":
        query_budget = provider.get("native_query_budget")
    else:
        objective = provider.get("objective", {})
        query_budget = {
            "primary_queries_per_case": provider.get("training", {}).get("primary_queries_per_case"),
            "role_query_counts": objective.get("role_query_counts"),
        }
    return {
        "status": "measured_epoch_and_validation_forecast",
        "run_id": recipe["run_id"],
        "active_mode": marker["active_mode"],
        "completed_epoch": completed_epoch,
        "remaining_epochs": remaining_epochs,
        "review_epochs_remaining": review_count,
        "median_training_seconds_per_epoch_recent": training_epoch_seconds,
        "median_full_dev_validation_seconds_recent": validation_review_seconds,
        "forecast_full_review_epoch_seconds": training_epoch_seconds + validation_review_seconds,
        "forecast_remaining_train_seconds": remaining_epochs * training_epoch_seconds,
        "forecast_remaining_validation_seconds": review_count * validation_review_seconds,
        "forecast_remaining_seconds_including_validation": (
            remaining_epochs * training_epoch_seconds + review_count * validation_review_seconds),
        "observed_case_visits_per_epoch": int(last["case_visits"]),
        "observed_optimizer_updates_per_epoch": int(last["optimizer_updates"]),
        "observed_case_visits_total": sum(int(row["case_visits"]) for row in rows),
        "observed_optimizer_updates_total": sum(int(row["optimizer_updates"]) for row in rows),
        "observed_work_counts_last_epoch": last.get("work_counts", {}),
        "observed_train_objective_last_epoch": last.get("train_losses", {}),
        "observed_validation_field_score_last_review": (
            None if not last.get("validation") else last["validation"].get("field_score")),
        "current_best_field_selected_age": None if best_row is None else int(best_row["epoch"]),
        "primary_and_auxiliary_query_budget": query_budget,
        "forecast_is_extrapolation_from_measured_epochs": True,
        "optimizer_updates_in_this_command": 0,
    }


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
        if (recipe["task"] == "thermal"
                and (recipe.get("fresh_matched_comparison") is True
                     or recipe["mode"] in ("P", "P-G", "P-H"))
                and args.command not in ("calibrate-thermal", "validate-followup")):
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
    if recipe.get("fresh_matched_comparison") is True:
        source_options["fresh_matched"] = True
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
                             full_train_followup: bool = False,
                             fresh_matched: bool = False) -> dict[str, Any]:
    repo_root = ROOT.parent
    source_files = _recovery_source_files(flow_readout_law, full_train_followup=full_train_followup)
    if fresh_matched:
        source_files = tuple(dict.fromkeys((*source_files, *FRESH_MATCHED_SOURCE_FILES)))
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
          require_clean_source: bool = False, include_dev_targets: bool = True):
    from honf_runtime.unified_training import EngineConfig, ScheduleSpec, SelectionPolicy, TrainingEngine
    if type(include_dev_targets) is not bool:
        raise TypeError("Fresh runner include_dev_targets must be boolean.")
    seed_all(recipe["seed"])
    if recipe["task"] == "thermal":
        from channelthermal.training.joint_task import build_thermal_joint_task as factory
    else:
        from windfarm.training.joint_task import build_wind_joint_task as factory
    task_options = {name: recipe[name] for name in ("response_coefficient", "operator_coefficient", "auxiliary_calibration",
                                                 "locality_prior_strength", "validation_scope", "weight_decay",
                                                 "native_sampling_protocol", "flow_readout_law",
                                                 "full_train_followup1000", "fresh_matched_comparison",
                                                 "conditional_confirmation")
                    if name in recipe}
    if recipe["task"] == "thermal":
        task_options["include_dev_targets"] = include_dev_targets
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
    fresh_matched = recipe.get("fresh_matched_comparison") is True
    if fresh_matched:
        if recipe.get("native_validation_policy") != FRESH_MATCHED_NATIVE_VALIDATION_POLICY:
            raise ValueError("Fresh matched recipe lacks its explicit consolidated native-validation policy.")
        core = getattr(model, "core", None)
        setter = getattr(core, "set_consolidated_native_validation", None)
        if not callable(setter):
            raise TypeError("Fresh matched native model does not expose consolidated validation controls.")
        setter(True)
        if getattr(core, "consolidated_native_validation", None) is not True:
            raise RuntimeError("Fresh matched consolidated native-validation policy was not enabled.")
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
    if recipe.get("fresh_matched_comparison") is True:
        source_options["fresh_matched"] = True
    source_identity = training_source_identity(**source_options)
    identity = {
        "workflow": "joint_regional_fields_v1", "recipe": recipe,
        "training_source_identity": source_identity,
        "initial_model_state_sha256": model_digest(model),
        "initialization": "fresh_all_trainable", "external_learned_field_files": [],
        "solver_attempts": 0, "WindTEST_targets": "locked",
    }
    if fresh_matched:
        identity["native_validation_policy"] = FRESH_MATCHED_NATIVE_VALIDATION_POLICY
    if fresh_matched and not calibration_only:
        from honf_runtime.unified_training import fresh_matched_initialization_receipt

        identity["workflow"] = FRESH_MATCHED_COMPARISON_PROTOCOL
        identity["fresh_initialization"] = fresh_matched_initialization_receipt(
            model, provider.identity_payload(), active_mode=recipe["mode"],
            seed=recipe["seed"], sampling_version=config.sampling_version,
            deterministic_case_order=config.deterministic_case_order)
        identity["initial_model_state_sha256"] = identity["fresh_initialization"]["initial_model_state_sha256"]
        identity["initialization"] = "fresh_common_trunk_private_router_rng"
        identity["inherited_parameter_names"] = []
        identity["optimizer_state_start"] = "fresh_zero_moments"
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
    parser.add_argument("command", choices=(
        "prepare", "dry-run", "validate-followup", "calibrate-thermal", "start", "resume", "branch",
        "allocate-matched", "prepare-matched", "validate-matched-streams", "forecast",
    ))
    parser.add_argument("--task", choices=("thermal", "wind"))
    parser.add_argument("--mode", choices=MODES, default="J-H")
    parser.add_argument("--recipe-json")
    parser.add_argument("--comparison-manifest")
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
    parser.add_argument("--forecast-horizon", type=int, default=2500)
    parser.add_argument("--manual-formal-launch", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "allocate-matched":
        if args.task is None or args.output_dir is None:
            parser.error("allocate-matched requires --task and a new --output-dir.")
        try:
            result = allocate_fresh_matched_comparison(args.task, args.output_dir)
        except FileExistsError as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "prepare-matched":
        if args.comparison_manifest is None:
            parser.error("prepare-matched requires --comparison-manifest.")
        print(json.dumps(prepare_fresh_matched_comparison(args.comparison_manifest),
                         indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "validate-matched-streams":
        if args.comparison_manifest is None:
            parser.error("validate-matched-streams requires --comparison-manifest.")
        print(json.dumps(validate_fresh_matched_streams(args.comparison_manifest),
                         indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "forecast":
        if args.output_dir is None:
            parser.error("forecast requires --output-dir for a prepared/running fresh history.")
        print(json.dumps(forecast_fresh_matched_run(args.output_dir, horizon=args.forecast_horizon),
                         indent=2, sort_keys=True), flush=True)
        return 0
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
    recovery_execution = (recipe["mode"] in ("P", "P-G", "P-H")
                          or recipe.get("fresh_matched_comparison") is True) and args.command in (
        "calibrate-thermal", "start", "resume", "branch")
    if args.command in ("start", "resume", "branch"):
        if not args.output_dir or args.stop_after is None:
            parser.error("Optimization requires --output-dir and --stop-after.")
        if recipe["formal_full"] and not full_followup and not args.manual_formal_launch:
            parser.error("Formal optimization requires explicit --manual-formal-launch; preparation is inert.")
        if not 1 <= args.stop_after <= recipe["total_epochs"]:
            parser.error("Requested stop exceeds the declared horizon.")
        if recipe.get("fresh_matched_comparison") is True:
            if args.command == "branch":
                parser.error("Fresh matched P/R1/R2 histories cannot branch from a checkpoint.")
            if args.stop_after not in FRESH_MATCHED_SELECTION_EPOCHS:
                parser.error("Fresh matched runs stop only at identical 100-epoch review/selection ages.")
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
            if recipe.get("fresh_matched_comparison") is True:
                if output.name != recipe["run_id"]:
                    parser.error("Fresh matched output directory basename must equal its allocated UUID run_id.")
                if output.exists() and any(output.iterdir()):
                    parser.error("Fresh matched start refuses to overwrite a nonempty allocated run directory.")
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
