#!/usr/bin/env python3
"""Fit the sealed paired source-response recipe from native TRAIN observations.

No generator, reference adapter, inverse search, solver or thermal teacher is
called. Stored TRAIN velocity appears only in the optional residual stencil.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter, time

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "Case_ThermalChannel/src", ROOT / "tools"):
    sys.path.insert(0, str(source))

import numpy as np
import torch
from channelthermal.data.collation import ChannelThermalBatchCollator
from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
from channelthermal.data.development_split import development_case_ids, resolve_development_manifest
from channelthermal.source_response import SOURCE_RESPONSE_CAPABILITY, SOURCE_RESPONSE_ID, ThermalSourceResponse
from channelthermal.source_response_residual import DiscreteThermalBalance, module_grid_ids
from channelthermal.training.checkpoints import _file_sha256, atomic_save_checkpoint_payload
from channelthermal.training.stop_request import acknowledge_stop, stop_requested
from thermal_development import validate_generated_output
from thermal_formal_profile import (
    FORMAL_TRAIN_SCOPE,
    bind_formal_validation,
    bind_original_train,
    ensure_formal_resume_identity,
    fit_formal_normalizer,
    formal_train_config,
    initialize_formal_output,
    stats_sha256,
    validate_formal_bindings,
)

from honf_runtime.compat import load_trusted_checkpoint, recursive_to_device, set_seed
from honf_runtime.run_store import atomic_write_json

TRAIN_FAMILIES = ("0001", "0318", "0333", "0348")
MANIFEST_FINGERPRINT = "933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044"
COLLATE = ChannelThermalBatchCollator()
SCHEDULE = {"hold_epochs": 1000, "total_epochs": 2500, "initial_lr": 3e-4, "final_lr": 3e-6}
BUDGET = {"fluid_queries": 1024, "material_queries_per_module": 32, "surface_queries_per_module": 16,
          "microbatch_cases": 8, "effective_cases": 48, "operator_rows_per_case": 128}


def thermal_learning_rate(epoch, schedule=SCHEDULE):
    hold_epochs = int(schedule["hold_epochs"])
    total_epochs = int(schedule["total_epochs"])
    initial_lr = float(schedule["initial_lr"])
    final_lr = float(schedule["final_lr"])
    if epoch <= hold_epochs:
        return initial_lr
    decay_epochs = max(total_epochs - hold_epochs, 1)
    progress = min(max((int(epoch) - hold_epochs) / decay_epochs, 0.0), 1.0)
    return final_lr + .5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * progress))


def read_primary(parent, split, normalization_stats=None):
    config = copy.deepcopy(parent["train_config"]["dataset"])
    manifest = resolve_development_manifest(config, config["packed_h5_path"])
    if manifest["manifest_sha256"] != MANIFEST_FINGERPRINT:
        raise ValueError("The new family requires literal fixed25_v1 membership, not another150/22 cohort.")
    ids = development_case_ids(manifest, split)
    if len(ids) != (150 if split == "train" else 22):
        raise ValueError("Source response uses the unchanged fixed25_v1 150/22 membership.")
    dataset = GlobalChannelThermalDataset(config["packed_h5_path"], split=split, points_per_case=None,
        normalize_inputs=False, normalize_targets=False, include_grid=split == "train",
        random_point_sampling=False, seed=0, case_ids=ids,
        normalizer=H5Normalizer(parent["global_normalization_stats"] if normalization_stats is None else normalization_stats))
    cases = [dataset[index] for index in range(len(dataset))]
    for case in cases:
        present = case["structure"]["module_present"] > .5
        if not np.array_equal(np.flatnonzero(present), np.arange(int(present.sum()))):
            raise ValueError("This packed residual binding requires audited contiguous original module slots.")
        case["structure"]["module_source_ids"] = np.arange(len(present), dtype=np.int64)
    return cases, manifest


def read_scoped_cases(dataset_config, split, case_ids, normalization_stats, include_grid=False):
    """Load metadata-bound cases under one declared full-TRAIN transform."""
    dataset = GlobalChannelThermalDataset(dataset_config["packed_h5_path"], split=split, points_per_case=None,
        normalize_inputs=False, normalize_targets=False, include_grid=include_grid, random_point_sampling=False,
        seed=0, case_ids=case_ids, normalizer=H5Normalizer(normalization_stats))
    if dataset.selected_case_ids != list(case_ids):
        raise ValueError("Formal loader did not preserve the declared source-split case order.")
    cases = [dataset[index] for index in range(len(dataset))]
    for case in cases:
        present = case["structure"]["module_present"] > .5
        if not np.array_equal(np.flatnonzero(present), np.arange(int(present.sum()))):
            raise ValueError("The packed residual binding requires audited contiguous original module slots.")
        case["structure"]["module_source_ids"] = np.arange(len(present), dtype=np.int64)
    return cases


def validate_formal_profile(profile):
    """Validate the explicit, separately identified formal recipe values."""
    required = {"schema_version", "profile_name", "workflow_scope", "preferred_response_family", "mode",
        "seed", "architecture", "data", "budget", "schedule", "optimizer", "loss", "checkpointing"}
    if set(profile) != required:
        raise ValueError("Formal source-response profile fields differ from the maintained schema.")
    if profile["schema_version"] != 1 or profile["workflow_scope"] != FORMAL_TRAIN_SCOPE:
        raise ValueError("Formal source-response profile requires the explicit full-TRAIN workflow identity.")
    if profile["preferred_response_family"] != "R-direct" or profile["mode"] != "direct":
        raise ValueError("The formal recommended recipe is the R-direct response family.")
    if set(profile["architecture"]) != {"hidden", "message", "background_mode"}:
        raise ValueError("Formal R-direct architecture must retain the reviewed source-response dimensions.")
    data = profile["data"]
    if set(data) != {"training_split", "normalization_source", "expected_train_case_count",
            "startup_validation", "formal_validation"}:
        raise ValueError("Formal source-response data profile fields differ from the maintained schema.")
    if data["training_split"] != "train" or data["normalization_source"] != "all_original_train_only":
        raise ValueError("Formal source-response data must use every original TRAIN case and a TRAIN-only normalizer.")
    if set(data["startup_validation"]) != {"scope", "manifest_sha256", "expected_case_count", "maximum_new_epochs"}:
        raise ValueError("Formal startup validation must declare an exact bounded DEV panel.")
    if (data["startup_validation"]["scope"] != "fixed25_v1_DEV22_exposed"
            or int(data["startup_validation"]["expected_case_count"]) != 22
            or int(data["startup_validation"]["maximum_new_epochs"]) != 3):
        raise ValueError("Formal startup may use only the fixed25_v1 DEV22 panel and up to three new epochs.")
    if set(data["formal_validation"]) != {"primary_scope", "compatibility_scope",
            "excluded_training_duplicate_case_id", "expected_primary_case_count", "expected_compatibility_case_count"}:
        raise ValueError("Formal canonical89/original90 validation fields differ from the maintained schema.")
    if (data["formal_validation"]["primary_scope"] != "original_test_excluding_train_duplicate"
            or data["formal_validation"]["compatibility_scope"] != "original_test_all_rows"
            or int(data["formal_validation"]["expected_primary_case_count"]) != 89
            or int(data["formal_validation"]["expected_compatibility_case_count"]) != 90):
        raise ValueError("Formal monitoring must declare canonical89 primary and original90 compatibility panels.")
    budget = profile["budget"]
    if set(budget) != {"fluid_queries", "material_queries_per_module", "surface_stride", "microbatch_cases",
            "effective_cases", "operator_rows_per_case"}:
        raise ValueError("Formal source-response budget fields differ from the maintained schema.")
    if (int(budget["fluid_queries"]) < 1 or int(budget["material_queries_per_module"]) < 1
            or int(budget["surface_stride"]) < 1 or int(budget["microbatch_cases"]) < 1
            or int(budget["effective_cases"]) < int(budget["microbatch_cases"])
            or int(budget["operator_rows_per_case"]) < 4
            or int(budget["operator_rows_per_case"]) % 4):
        raise ValueError("Formal source-response query and batch budgets must be positive and internally consistent.")
    schedule = profile["schedule"]
    if set(schedule) != {"horizon_epochs", "hold_fraction", "initial_lr", "final_lr"}:
        raise ValueError("Formal source-response schedule fields differ from the maintained schema.")
    if (int(schedule["horizon_epochs"]) < 1 or not 0 <= float(schedule["hold_fraction"]) < 1
            or not 0 < float(schedule["final_lr"]) <= float(schedule["initial_lr"])):
        raise ValueError("Formal schedule must have a positive final learning rate and a valid horizon.")
    if set(profile["optimizer"]) != {"name", "weight_decay", "gradient_clip"} or profile["optimizer"]["name"] != "AdamW":
        raise ValueError("Formal source-response optimizer must be the declared AdamW profile.")
    if set(profile["loss"]) != {"q_proxy_coefficient", "use_measured_response", "use_qualified_operator"}:
        raise ValueError("Formal source-response objective fields differ from the maintained schema.")
    if set(profile["checkpointing"]) != {"monitoring_interval_epochs", "save_latest", "save_best_field", "milestone_epochs"}:
        raise ValueError("Formal source-response checkpoint fields differ from the maintained schema.")
    if (int(profile["checkpointing"]["monitoring_interval_epochs"]) != 100
            or profile["checkpointing"]["save_latest"] is not True
            or profile["checkpointing"]["save_best_field"] is not True):
        raise ValueError("Formal source-response monitoring must save latest/best state every 100 epochs.")
    horizon = int(schedule["horizon_epochs"])
    expected_milestones = list(range(100, horizon + 1, 100))
    if [int(value) for value in profile["checkpointing"]["milestone_epochs"]] != expected_milestones:
        raise ValueError("Formal source-response milestones must be every 100 epochs through the endpoint.")
    return profile


def sample_primary(cases, indices, epoch, device, training=True, budget=BUDGET):
    samples = []
    for index in indices:
        original = cases[int(index)]
        seed = (0 if training else 1000) + int(index) * 104729 + (epoch * 1000003 if training else 0)
        rng = np.random.default_rng(seed)
        fluid = rng.choice(len(original["query_xy"]), int(budget["fluid_queries"]), replace=False)
        material = rng.choice(len(original["module_internal_query_points"]),
            int(budget["material_queries_per_module"]), replace=False)
        sample = {"structure": original["structure"], "query_xy": original["query_xy"][fluid],
            "field_targets": original["field_targets"][fluid], "point_weights": original["point_weights"][fluid],
            "interface_target": original["interface_target"][:, ::int(budget.get("surface_stride", 4))],
            "interface_condition_valid_mask": original["interface_condition_valid_mask"][:, ::int(budget.get("surface_stride", 4))],
            "module_internal_temperature_points": original["module_internal_temperature_points"][:, material],
            "module_internal_query_points": original["module_internal_query_points"][material]}
        samples.append(sample)
    return recursive_to_device(COLLATE(samples), device)


def scalar_scale(stats, key, channel=None):
    if key == "interface_targets_std" and key not in stats:
        key = "interface_target_std"
    value = np.asarray(stats[key]).reshape(-1)
    return max(float(value[0 if channel is None else channel]), 1e-6)


def reconstruction_terms(model, batch, stats, q_proxy_coefficient=.05):
    prepared = model.prepare_native(batch["structure"], batch["query_xy"],
        local_query_points=batch["module_internal_query_points"], ntheta=16)
    prediction = model.apply_native(prepared, batch["structure"]["heat_powers"])
    weight = batch["point_weights"]
    fluid = ((prediction["fluid_temperature"][..., 0] - batch["field_targets"][..., 4]) /
             scalar_scale(stats, "field_std_by_channel", 4)).square()
    fluid = (fluid * weight).sum(1) / weight.sum(1).clamp_min(1e-12)
    present = batch["structure"]["module_present"]
    native_surface_valid = prepared.stencils["surface"].valid.reshape(prediction["pred_interface"].shape[:-1])
    native_outside_valid = prepared.stencils["outside"].valid.reshape(prediction["pred_interface"].shape[:-1])
    surface_mask = present[..., None] * native_surface_valid * torch.isfinite(batch["interface_target"][..., 0])
    surface_error = ((prediction["pred_interface"][..., 0] - batch["interface_target"][..., 0]) /
        scalar_scale(stats, "interface_targets_std", 0))
    valid_module = present * (surface_mask.sum(-1) > 0)
    surface_per_module = (surface_error.square() * surface_mask).sum(-1) / surface_mask.sum(-1).clamp_min(1)
    surface = (surface_per_module * valid_module).sum(-1) / valid_module.sum(-1).clamp_min(1)
    material_error = (prediction["pred_internal_temperature"][..., 0] -
        batch["module_internal_temperature_points"]) / scalar_scale(stats, "internal_temperature_std")
    material = ((material_error.square() * present[..., None]).sum((1, 2)) /
        (present.sum(1) * material_error.shape[-1]).clamp_min(1))
    q_error = ((prediction["pred_interface"][..., 1] - batch["interface_target"][..., 1]) /
        scalar_scale(stats, "interface_targets_std", 1))
    q_mask = present[..., None] * native_surface_valid * native_outside_valid * torch.isfinite(batch["interface_target"][..., 1])
    q_modules = present * (q_mask.sum(-1) > 0)
    q_per_module = (q_error.square() * q_mask).sum(-1) / q_mask.sum(-1).clamp_min(1)
    q = (q_per_module * q_modules).sum(-1) / q_modules.sum(-1).clamp_min(1)
    thermal = (fluid + surface + material) / 3
    return {"reconstruction": thermal + float(q_proxy_coefficient) * q, "thermal_selector": thermal,
            "fluid": fluid, "surface": surface, "material": material, "q_proxy": q}, prepared, prediction


def read_response_families(atlas_directory):
    families = []
    for family_id in TRAIN_FAMILIES:
        path = Path(atlas_directory) / f"train_{family_id}_responses.npz"
        with np.load(path, allow_pickle=False) as handle:
            atlas = {name: handle[name].copy() for name in handle.files}
        meta = json.loads(str(atlas["family_metadata_json"]))
        expected_family = "duplicate_family:0001+0273" if family_id == "0001" else f"stored_family:{family_id}"
        if (meta["source_dataset_split"] != "train" or meta["anchor_id"] != family_id
                or meta["physical_family_id"] != expected_family):
            raise ValueError("Only declared original-TRAIN response families are eligible.")
        labels = atlas["all_labels"].tolist()
        base, plus = labels.index("baseline"), labels.index("heat_transfer_plus")
        if not np.array_equal(atlas["module_centers_xy"][base], atlas["module_centers_xy"][plus]):
            raise ValueError("Response supervision must hold layout fixed.")
        case = meta["case_config"]
        centers = atlas["module_centers_xy"][base].astype(np.float32)
        count = len(centers)
        module_ids = [f"{family_id}:module:{slot}" for slot in range(count)]
        if atlas["active_module_ids"].tolist() != module_ids:
            raise ValueError("Response source IDs differ from their original module slots.")
        for role in ("interface", "solid_temperature"):
            role_ids = atlas[f"{role}_receiver_module_ids"]
            if len(role_ids) % count or not all(np.all(block == module_id) for block, module_id in
                    zip(np.split(role_ids, count), module_ids)):
                raise ValueError("Native role blocks must retain their explicit physical module IDs.")
        material = np.asarray([case["runtime"]["nu"], case["thermal"]["solid_alpha"],
            case["thermal"]["fluid_alpha"], case["thermal"]["solid_k"], case["thermal"]["fluid_k"],
            case["domain"]["module_radius"]], dtype=np.float32)
        structure = {"module_centers": centers, "module_present": np.ones(count, dtype=np.float32),
            "module_source_ids": np.arange(count, dtype=np.int64),
            "material_params": material, "re": np.asarray([case["flow"]["re"]], dtype=np.float32),
            "u_in": np.asarray([case["flow"]["u_in"]], dtype=np.float32),
            "domain_length_x": np.asarray([case["domain"]["lx"]], dtype=np.float32),
            "domain_length_y": np.asarray([case["domain"]["ly"]], dtype=np.float32)}
        deltas = {role: atlas[f"{role}_values"][plus].astype(np.float64) -
            atlas[f"{role}_values"][base].astype(np.float64) for role in ("fluid_fields", "interface", "solid_temperature")}
        local_count = len(atlas["solid_temperature_query_features"]) // count
        # Atlas contract declares solid_material_normalized_xy. These are
        # already normalized local coordinates, not physical global x/y.
        local = atlas["solid_temperature_query_features"][:local_count, :2]
        if not all(np.array_equal(local, atlas["solid_temperature_query_features"][slot * local_count:(slot + 1) * local_count, :2]) for slot in range(count)):
            raise ValueError("Material response coordinates must preserve the exact per-module local join.")
        families.append({"family_id": family_id, "source": str(path), "source_sha256": _file_sha256(path), "structure": structure,
            "heat_increment": (atlas["heating"][plus].astype(np.float64) - atlas["heating"][base].astype(np.float64)).astype(np.float32),
            "fluid_xy": atlas["fluid_fields_query_features"][:, :2].astype(np.float32),
            "fluid_valid": atlas["heat_common_fluid_mask"].astype(bool),
            "material_local": local.astype(np.float32), "deltas": deltas, "module_count": count,
            "interface_valid": atlas["heat_common_interface_mask"].astype(bool),
            "material_valid": atlas["heat_common_solid_mask"].reshape(count, local_count).astype(bool)})
    return families


def response_scales(families):
    return {name: max(float(np.sqrt(np.mean(values))), 1e-6) for name, values in {
        "fluid": [np.mean(f["deltas"]["fluid_fields"][f["fluid_valid"], 4] ** 2) for f in families],
        "surface": [np.mean(f["deltas"]["interface"][:, 0][f["interface_valid"][:, 0]] ** 2) for f in families],
        "material": [np.mean(f["deltas"]["solid_temperature"].reshape(f["module_count"], -1)[f["material_valid"]] ** 2) for f in families],
    }.items()}


def response_loss(model, family, epoch, device, scales, budget=BUDGET):
    index = TRAIN_FAMILIES.index(family["family_id"])
    rng = np.random.default_rng(index * 104729 + epoch * 1000003)
    valid_fluid = np.flatnonzero(family["fluid_valid"])
    fluid_count = int(budget["fluid_queries"])
    material_count = int(budget["material_queries_per_module"])
    fluid_ids = rng.choice(valid_fluid, fluid_count, replace=False)
    material_ids = rng.choice(len(family["material_local"]), material_count, replace=False)
    structure = {name: torch.as_tensor(value, device=device)[None] for name, value in family["structure"].items()}
    fluid = torch.as_tensor(family["fluid_xy"][fluid_ids], device=device)[None]
    local = torch.as_tensor(family["material_local"][material_ids], device=device)[None]
    prepared = model.prepare_native(structure, fluid, local_query_points=local, ntheta=16)
    delta = model.apply_native(prepared, torch.as_tensor(family["heat_increment"], device=device)[None], increment=True)
    tensor = lambda value: torch.as_tensor(value, device=device, dtype=torch.float32)
    fluid_target = tensor(family["deltas"]["fluid_fields"][fluid_ids, 4])
    surface_stride = int(budget.get("surface_stride", 4))
    surface_target = tensor(family["deltas"]["interface"][:, 0].reshape(family["module_count"], -1)[:, ::surface_stride])
    material_target = tensor(family["deltas"]["solid_temperature"].reshape(family["module_count"], -1)[:, material_ids])
    surface_mask = tensor(family["interface_valid"][:, 0].reshape(family["module_count"], -1)[:, ::surface_stride])
    material_mask = tensor(family["material_valid"][:, material_ids])
    terms = [((delta["fluid_temperature"][0, :, 0] - fluid_target) / scales["fluid"]).square().mean(),
        (((delta["pred_interface"][0, ..., 0] - surface_target) / scales["surface"]).square() * surface_mask).sum() / surface_mask.sum().clamp_min(1),
        (((delta["pred_internal_temperature"][0, ..., 0] - material_target) / scales["material"]).square() * material_mask).sum() / material_mask.sum().clamp_min(1)]
    return sum(terms) / 3, {"response_fluid_rows": int(fluid_ids.size),
        "response_surface_rows": int(surface_target.numel()), "response_material_rows": int(material_target.numel()),
        "response_neural_receiver_rows": int(prepared.grid_indices.numel())}


def build_balances(cases):
    balances = []
    for case in cases:
        structure = case["structure"]
        present = structure["module_present"]
        slot = module_grid_ids(torch.as_tensor(case["x_grid"]), torch.as_tensor(case["y_grid"]),
            torch.as_tensor(structure["module_centers"])[None], torch.as_tensor(present)[None],
            torch.as_tensor(structure["material_params"][5:6]))
        lengths = np.array([float(structure["domain_length_x"][0]), float(structure["domain_length_y"][0])], dtype=np.float32)
        velocity = torch.as_tensor(case["steady_field"])
        balances.append(DiscreteThermalBalance.from_training_fields(velocity[None, ..., 0], velocity[None, ..., 1],
            slot, torch.as_tensor(structure["material_params"])[None], torch.as_tensor(lengths)[None],
            torch.as_tensor(present)[None]))
    return balances


def gradient_norm(loss, model):
    gradient = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    return float(torch.sqrt(sum(value.square().sum() for value in gradient if value is not None)))


def construct(mode):
    set_seed(0)
    model = ThermalSourceResponse({"mode": mode, "hidden": 64, "message": 64, "background_mode": True})
    model.core_config = copy.deepcopy(model.core.config)
    return model


def construct_profiled(mode, architecture, seed):
    set_seed(int(seed))
    model = ThermalSourceResponse({"mode": mode, **dict(architecture)})
    model.core_config = copy.deepcopy(model.core.config)
    return model


def operator_loss(model, prepared, balances, indices, epoch, device, stats, rows_per_case=128):
    from channelthermal.source_response_residual import sampled_kernel_residual
    coefficient = torch.cat([balances[int(i)].coefficients for i in indices]).to(device)
    slots = torch.cat([balances[int(i)].source_slots for i in indices]).to(device)
    present = prepared.source_present
    balance = DiscreteThermalBalance(coefficient, slots, present)
    rows = []
    for index in indices:
        source = balances[int(index)].source_slots[0].numpy()
        _ny, _nx = source.shape
        boundary = np.zeros_like(source, dtype=bool)
        boundary[[0, -1], :] = True; boundary[:, [0, -1]] = True
        near_interface = np.zeros_like(source, dtype=bool)
        near_interface[:, 1:] |= (source[:, 1:] >= 0) != (source[:, :-1] >= 0)
        near_interface[1:] |= (source[1:] >= 0) != (source[:-1] >= 0)
        rng = np.random.default_rng(int(index) * 104729 + epoch * 1000003 + 17)
        categories = [boundary, (source >= 0) & ~boundary, near_interface & ~boundary, (source < 0) & ~boundary]
        chosen = []
        rows_per_category = int(rows_per_case) // len(categories)
        if rows_per_category * len(categories) != int(rows_per_case):
            raise ValueError("Operator rows per case must divide evenly across the four fixed stencil strata.")
        for category in categories:
            valid = np.flatnonzero(category)
            chosen.extend(rng.choice(valid, rows_per_category, replace=len(valid) < rows_per_category).tolist())
        rows.append(chosen)
    residual, receipt = sampled_kernel_residual(model.core, prepared.context, balance,
        torch.as_tensor(rows, device=device), prepared.context.lengths)
    residual = residual / scalar_scale(stats, "field_std_by_channel", 4)
    per_case = (residual.square() * present[:, None]).sum((1, 2)) / (present.sum(-1) * int(rows_per_case)).clamp_min(1)
    actual_rows = int(len(indices) * int(rows_per_case))
    return per_case.mean(), {**receipt, "operator_rows": actual_rows,
                  "operator_source_columns": int(prepared.source_present.sum().item()) * int(rows_per_case)}


def calibrate(models, cases, families, stats, balances, use_operator, budget=BUDGET,
              q_proxy_coefficient=.05):
    scales = response_scales(families)
    counts = [int(c["structure"]["module_present"].sum()) for c in cases]
    selected = [next(i for i, count in enumerate(counts) if count == wanted) for wanted in (1, 3, 10, 12)]
    measured = []
    for mode, model in models.items():
        reconstruction_gradients, q_gradients, operator_gradients, response_gradients = [], [], [], []
        for index in selected:
            batch = sample_primary(cases, [index], 1, torch.device("cpu"), budget=budget)
            terms, prepared, _ = reconstruction_terms(model, batch, stats, q_proxy_coefficient)
            reconstruction_gradients.append(gradient_norm(terms["thermal_selector"].mean(), model))
            q_terms, _, _ = reconstruction_terms(model, batch, stats, q_proxy_coefficient)
            q_gradients.append(gradient_norm(float(q_proxy_coefficient) * q_terms["q_proxy"].mean(), model))
            if use_operator:
                terms, prepared, _ = reconstruction_terms(model, batch, stats)
                loss, _ = operator_loss(model, prepared, balances, [index], 1, torch.device("cpu"), stats,
                    rows_per_case=int(budget["operator_rows_per_case"]))
                operator_gradients.append(gradient_norm(loss, model))
        for family in families:
            loss, _ = response_loss(model, family, 1, torch.device("cpu"), scales, budget)
            response_gradients.append(gradient_norm(loss, model))
        measured.append({"mode": mode, "reconstruction_gradient_norms": reconstruction_gradients,
            "q_auxiliary_gradient_norms": q_gradients, "response_gradient_norms": response_gradients,
            "operator_gradient_norms": operator_gradients})
    response_ratios = [.5 * np.mean(row["reconstruction_gradient_norms"]) /
        max(float(np.mean(row["response_gradient_norms"])), 1e-12) for row in measured]
    operator_ratios = [.5 * np.mean(row["reconstruction_gradient_norms"]) /
        max(float(np.mean(row["operator_gradient_norms"])), 1e-12) for row in measured] if use_operator else []
    return {"response_coefficient": min(1., *response_ratios),
        "operator_coefficient": min(1., *operator_ratios) if use_operator else 0.,
        "finite_coefficient_cap": 1., "target_added_gradient_ratio": .5, "gradient_measurements": measured,
        "calibration_case_ids": [cases[index]["case_id"] for index in selected], "response_scales": scales,
        "operator_constraint": "qualified" if use_operator else "disabled_with_reason"}


@torch.no_grad()
def validate(model, cases, stats, device, budget=BUDGET, q_proxy_coefficient=.05):
    rows = []
    for start in range(0, len(cases), int(budget["microbatch_cases"])):
        batch = sample_primary(cases, range(start, min(start + int(budget["microbatch_cases"]), len(cases))),
            0, device, False, budget)
        terms, _, _ = reconstruction_terms(model, batch, stats, q_proxy_coefficient)
        for offset in range(len(terms["thermal_selector"])):
            rows.append({"case_id": cases[start + offset]["case_id"],
                **{name: float(value[offset]) for name, value in terms.items()}})
    return float(np.mean([row["thermal_selector"] for row in rows])), rows


@torch.no_grad()
def validate_composition(model, flow_model, cases, stats, device, budget, q_proxy_coefficient):
    """Measure the thermal selector and common five-field score on one panel."""
    per_case_mse = []
    rows = []
    microbatch = int(budget["microbatch_cases"])
    means = torch.as_tensor(stats["field_mean_by_channel"][:4], device=device, dtype=torch.float32)
    scales = torch.as_tensor(stats["field_std_by_channel"], device=device, dtype=torch.float32)
    for start in range(0, len(cases), microbatch):
        indices = range(start, min(start + microbatch, len(cases)))
        batch = sample_primary(cases, indices, 0, device, False, budget)
        thermal_terms, _, thermal_prediction = reconstruction_terms(model, batch, stats, q_proxy_coefficient)
        flow_normalized = flow_model(batch["structure"], batch["query_xy"])
        flow_physical = flow_normalized * scales[:4] + means
        prediction = torch.cat((flow_physical, thermal_prediction["fluid_temperature"]), dim=-1)
        target = batch["field_targets"]
        weight = batch["point_weights"]
        channel_mse = (((prediction - target).square() * weight[..., None]).sum(1)
            / weight.sum(1).clamp_min(1e-12)[:, None])
        standardized = channel_mse / scales.square()
        per_case_mse.extend(channel_mse.cpu().numpy())
        for offset, values in enumerate(channel_mse.cpu().numpy()):
            rows.append({"case_id": cases[start + offset]["case_id"],
                "thermal_selector": float(thermal_terms["thermal_selector"][offset]),
                "physical_field_mse_by_channel": values.tolist(),
                "standardized_five_field_mse": float(standardized[offset].mean())})
    case_mse = np.asarray(per_case_mse, dtype=np.float64)
    summary = {
        "thermal_selector": float(np.mean([row["thermal_selector"] for row in rows])),
        "standardized_five_field_mse": float(np.mean([row["standardized_five_field_mse"] for row in rows])),
        "physical_rmse_by_channel": np.sqrt(case_mse.mean(axis=0)).tolist(),
        "channel_order": ["u", "v", "p", "omega", "temperature"],
        "case_count": len(cases),
        "rows": rows,
    }
    return summary


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([row["epoch"] for row in history], [row["train_thermal_selector"] for row in history], label="TRAIN thermal")
    review = [row for row in history if "validation_thermal_selector" in row]
    axes[0].plot([row["epoch"] for row in review], [row["validation_thermal_selector"] for row in review], "o-", label="declared validation panel")
    axes[0].set(xlabel="New thermal epochs", ylabel="Standardized three-role MSE", yscale="log"); axes[0].legend()
    axes[1].plot([row["epoch"] for row in history], [row["learning_rate"] for row in history])
    axes[1].set(xlabel="New thermal epochs", ylabel="Actual learning rate")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def formal_checkpoint_payload(model, optimizer, flow, train_config, recipe, epoch, best, history,
                              aggregate_process_seconds):
    return {
        "checkpoint_schema_version": 1, "case_id": "ThermalChannel", "model_family": "honf_forward",
        "workflow": "forward", "stage": SOURCE_RESPONSE_ID, "source_response_identity": SOURCE_RESPONSE_ID,
        "case_capability": SOURCE_RESPONSE_CAPABILITY,
        "channel_order": ["u", "v", "p", "omega", "temperature"],
        "source_response_config": {"core": model.core_config, "adapter": model.adapter_config()},
        "thermal_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "schedule_state": {"completed_epoch": epoch, "declaration": recipe["schedule"]},
        "fit_identity": {"recipe": recipe, "mode": "direct"}, "epoch": epoch, "current_epoch": epoch,
        "best_metric": best, "history": history, "aggregate_process_seconds": aggregate_process_seconds,
        "formal_workflow_scope": FORMAL_TRAIN_SCOPE, "startup_benchmark": recipe["startup_benchmark"],
        "flow_checkpoint": recipe["flow_checkpoint"], "flow_checkpoint_sha256": recipe["flow_checkpoint_sha256"],
        "formal_dataset_binding": recipe["formal_dataset_binding"],
        "formal_normalization_binding": recipe["formal_normalization_binding"],
        "formal_validation_binding": recipe["formal_validation_binding"],
        "global_normalization_stats": flow["global_normalization_stats"],
        "global_normalization_config": recipe["formal_normalization_binding"],
        "train_config": train_config,
    }


def summarize_composition_rows(rows, case_ids, scope):
    wanted = set(case_ids)
    selected = [row for row in rows if row["case_id"] in wanted]
    if len(selected) != len(wanted):
        raise ValueError("Composed model validation lost declared source case IDs.")
    physical_mse = np.asarray([row["physical_field_mse_by_channel"] for row in selected], dtype=np.float64)
    return {"scope": scope, "case_count": len(selected),
        "thermal_selector": float(np.mean([row["thermal_selector"] for row in selected])),
        "standardized_five_field_mse": float(np.mean([row["standardized_five_field_mse"] for row in selected])),
        "physical_rmse_by_channel": np.sqrt(physical_mse.mean(axis=0)).tolist(),
        "channel_order": ["u", "v", "p", "omega", "temperature"]}


def run_formal(args, start, started):
    setup_start = perf_counter()
    profile_path = Path(args.profile_file).resolve()
    profile = validate_formal_profile(json.loads(profile_path.read_text()))
    if not args.startup_benchmark and args.run_identity and args.run_identity.startswith("startup_"):
        raise ValueError("Formal full-horizon identities cannot use a disposable startup run name.")
    if args.startup_benchmark and not (args.run_identity or "").startswith("startup_thermal_"):
        raise ValueError("Disposable thermal startup identities must start with startup_thermal_.")
    if args.mode not in (None, "direct"):
        raise ValueError("The maintained full-TRAIN recipe permits R-direct only.")
    output = initialize_formal_output(validate_generated_output(args.output), prepare_only=args.prepare_only)
    parent_path = Path(args.parent).resolve()
    flow_path = Path(args.flow_checkpoint).resolve()
    parent = load_trusted_checkpoint(parent_path, map_location="cpu")
    flow = load_trusted_checkpoint(flow_path, map_location="cpu")
    parent_dataset = parent["train_config"]["dataset"]
    dataset_path = parent_dataset["packed_h5_path"]
    dataset_binding, train_ids = bind_original_train(dataset_path,
        expected_count=profile["data"]["expected_train_case_count"], dataset_id=parent_dataset.get("dataset_id"))
    formal_validation_binding, primary_validation_ids, compatibility_validation_ids = bind_formal_validation(
        dataset_path,
        expected_primary_count=profile["data"]["formal_validation"]["expected_primary_case_count"],
        expected_compatibility_count=profile["data"]["formal_validation"]["expected_compatibility_case_count"],
        duplicate_case_id=profile["data"]["formal_validation"]["excluded_training_duplicate_case_id"])
    normalizer, normalization_binding = fit_formal_normalizer(dataset_path, dataset_binding)
    stats = normalizer.stats
    if flow.get("dependency_policy") != "D-sep" or flow.get("dependency_identity") != "thermal_dependency_flow_v1":
        raise ValueError("Formal R-direct requires a freshly trained D-sep flow component.")
    if flow.get("case_capability") != "channelthermal_analytic_wake_prescribed_flow_v1":
        raise ValueError("Formal flow partner does not declare the audited heat-independent D-sep capability.")
    if flow.get("formal_workflow_scope") != FORMAL_TRAIN_SCOPE:
        raise ValueError("Formal R-direct rejects a development or historical D-sep flow checkpoint.")
    validate_formal_bindings(flow, dataset_binding, normalization_binding, formal_validation_binding, stats)
    if bool(flow.get("startup_benchmark")) != bool(args.startup_benchmark):
        raise ValueError("Flow and thermal components must share the startup-versus-full formal identity scope.")
    if args.startup_benchmark:
        if int(flow.get("epoch", 0)) > int(profile["data"]["startup_validation"]["maximum_new_epochs"]):
            raise ValueError("Disposable thermal startup rejects a flow component older than three new epochs.")
    elif int(flow.get("epoch", 0)) != int(profile["schedule"]["horizon_epochs"]):
        raise ValueError("Full formal thermal fitting requires the exact e5000 full-TRAIN flow endpoint.")
    flow_dataset_path = flow["train_config"]["dataset"].get("packed_h5_path")
    if (not flow_dataset_path or Path(flow_dataset_path).resolve() != Path(dataset_path).resolve()):
        raise ValueError("Formal R-direct and D-sep components refer to different packed datasets.")
    expected_flow_train_config = formal_train_config(parent["train_config"], dataset_binding,
        normalization_binding, formal_validation_binding)
    if flow["train_config"].get("dataset") != expected_flow_train_config["dataset"]:
        raise ValueError("Formal D-sep train_config does not match the parent-derived full-TRAIN data and normalization configuration.")
    if not 1 <= int(flow.get("epoch", 0)) <= int(profile["schedule"]["horizon_epochs"]):
        raise ValueError("Formal D-sep component age is outside the declared full-data schedule.")

    startup_panel = profile["data"]["startup_validation"]
    dev_manifest = parent_dataset.get("development_subset")
    if (not isinstance(dev_manifest, dict) or dev_manifest.get("manifest_sha256") != startup_panel["manifest_sha256"]
            or len(dev_manifest.get("partitions", {}).get("test", {}).get("case_ids", []))
            != int(startup_panel["expected_case_count"])):
        raise ValueError("Formal startup validation reference must bind the declared fixed25_v1 DEV22 panel.")
    startup_validation_ids = list(dev_manifest["partitions"]["test"]["case_ids"])
    dataset_load_start = perf_counter()
    cases = read_scoped_cases(parent_dataset, "train", train_ids, stats, include_grid=True)
    if args.startup_benchmark:
        validation_scope = startup_panel["scope"]
        validation, val_manifest = read_primary(parent, "test", stats)
        validation_case_ids = [case["case_id"] for case in validation]
        compatibility_cases = None
    else:
        validation_scope = profile["data"]["formal_validation"]["primary_scope"]
        compatibility_cases = read_scoped_cases(parent_dataset, "test", compatibility_validation_ids, stats)
        compatibility_set = set(compatibility_validation_ids)
        if [case["case_id"] for case in compatibility_cases] != compatibility_validation_ids:
            raise ValueError("Formal validation loader changed original90 compatibility membership order.")
        primary_set = set(primary_validation_ids)
        validation = [case for case in compatibility_cases if case["case_id"] in primary_set]
        validation_case_ids = [case["case_id"] for case in validation]
        val_manifest = {"manifest_sha256": formal_validation_binding["primary_case_ids_sha256"]}
        if len(validation) != len(primary_validation_ids) or compatibility_set != primary_set | {
                profile["data"]["formal_validation"]["excluded_training_duplicate_case_id"]}:
            raise ValueError("Formal canonical89/original90 panels do not match their bound identities.")
    dataset_load_seconds = perf_counter() - dataset_load_start
    if [case["case_id"] for case in cases] != train_ids:
        raise ValueError("Formal source-response cases do not match the source H5 TRAIN metadata order.")
    families = read_response_families(args.atlas_directory)
    decision = json.loads(Path(args.operator_decision).read_text())
    if decision.get("operator_constraint") not in ("qualified", "disabled_with_reason") or not decision.get("reason"):
        raise ValueError("Formal startup requires a declared qualified or disabled TRAIN-only operator decision.")
    use_operator = decision["operator_constraint"] == "qualified"
    if bool(profile["loss"]["use_qualified_operator"]) != use_operator:
        raise ValueError("Formal profile operator objective differs from the qualified operator decision.")
    if not bool(profile["loss"]["use_measured_response"]):
        raise ValueError("Formal R-direct must retain the established measured-response supervision.")
    balances = build_balances(cases) if use_operator else []
    budget = dict(profile["budget"])
    schedule = {"hold_epochs": round(int(profile["schedule"]["horizon_epochs"])
        * float(profile["schedule"]["hold_fraction"])),
        "total_epochs": int(profile["schedule"]["horizon_epochs"]),
        "initial_lr": float(profile["schedule"]["initial_lr"]),
        "final_lr": float(profile["schedule"]["final_lr"])}
    if schedule["hold_epochs"] >= schedule["total_epochs"]:
        raise ValueError("Formal thermal hold must end before the declared horizon.")
    flow_dataset_binding = dict(flow["formal_dataset_binding"])
    if flow_dataset_binding != dataset_binding:
        raise ValueError("Formal source-response and flow full-TRAIN data bindings differ.")

    train_config = flow["train_config"]
    profile_value = json.loads(json.dumps(profile))
    run_identity = args.run_identity or "formal5000"
    recipe = {
        "identity": "thermal_source_response_r_direct_formal5000_v1",
        "workflow_scope": FORMAL_TRAIN_SCOPE,
        "run_identity": run_identity,
        "startup_benchmark": bool(args.startup_benchmark),
        "preferred_response_family": "R-direct",
        "mode": "direct",
        "profile": profile_value,
        "profile_path": str(profile_path),
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "seed": int(profile["seed"]), "horizon": schedule["total_epochs"],
        "budget": budget, "schedule": schedule,
        "weight_decay": float(profile["optimizer"]["weight_decay"]),
        "gradient_clip": float(profile["optimizer"]["gradient_clip"]),
        "validation_reference_checkpoint": str(parent_path),
        "validation_reference_checkpoint_sha256": _file_sha256(parent_path),
        "validation_manifest_sha256": val_manifest["manifest_sha256"],
        "validation_scope": validation_scope,
        "validation_case_ids": validation_case_ids,
        "compatibility_validation_case_ids": [] if compatibility_cases is None else compatibility_validation_ids,
        "formal_validation_binding": formal_validation_binding,
        "startup_validation_case_ids": startup_validation_ids,
        "flow_checkpoint": str(flow_path), "flow_checkpoint_sha256": _file_sha256(flow_path),
        "flow_component_age": int(flow["epoch"]),
        "formal_dataset_binding": flow_dataset_binding,
        "formal_normalization_binding": normalization_binding,
        "normalization_stats_sha256": stats_sha256(stats),
        "operator_decision": decision,
        "response_fit_families": list(TRAIN_FAMILIES),
        "response_sources": [{"family_id": family["family_id"], "path": family["source"],
            "sha256": family["source_sha256"]} for family in families],
        "primary_loss": "mean standardized fluid/surface/material T MSE + profile q-proxy coefficient + TRAIN response + qualified operator residual",
        "selector": f"minimum {validation_scope} mean standardized fluid/surface/material reconstruction at 100-epoch cadence",
        "new_solver_attempts": 0,
    }

    if args.prepare_only:
        set_seed(int(profile["seed"]))
        model = construct_profiled("direct", profile["architecture"], int(profile["seed"]))
        calibration = calibrate({"direct": model}, cases, families, stats, balances, use_operator, budget,
            float(profile["loss"]["q_proxy_coefficient"]))
        recipe["calibration"] = calibration
        recipe["core_configs"] = {"direct": model.core_config}
        recipe["adapter_config"] = model.adapter_config()
        recipe["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        recipe["calibration_case_ids"] = calibration["calibration_case_ids"]
        atomic_write_json(output / "formal_recipe.json", recipe)
        optimizer = torch.optim.AdamW(model.parameters(), lr=schedule["initial_lr"],
            weight_decay=recipe["weight_decay"])
        payload = formal_checkpoint_payload(model, optimizer, flow, train_config, recipe, 0, float("inf"), [], 0.)
        atomic_save_checkpoint_payload(output / "latest_model.pt", payload)
        atomic_write_json(output / "formal_profile_binding.json", {"profile": profile_value,
            "dataset_binding": flow_dataset_binding, "normalization_binding": normalization_binding,
            "normalization_stats_sha256": stats_sha256(stats), "case_count": len(cases),
            "validation_case_ids": recipe["validation_case_ids"],
            "compatibility_validation_case_ids": recipe["compatibility_validation_case_ids"],
            "validation_scope": validation_scope, "startup_benchmark": bool(args.startup_benchmark),
            "flow_checkpoint": str(flow_path), "flow_epoch": int(flow["epoch"]), "run_identity": run_identity})
        print(json.dumps({"status": "prepared", "profile": profile["profile_name"],
            "run_identity": run_identity, "training_cases": len(cases), "validation_cases": len(validation),
            "output": str(output)}, sort_keys=True), flush=True)
        return 0

    if args.recipe is None or args.resume is None:
        raise ValueError("Formal fitting requires its sealed --recipe and explicit --resume checkpoint.")
    recipe_path = Path(args.recipe).resolve()
    saved_recipe = json.loads(recipe_path.read_text())
    if saved_recipe != recipe:
        # Calibration is part of the sealed recipe and is only available after
        # preparation; compare all current source/profile bindings separately.
        for key, value in recipe.items():
            if saved_recipe.get(key) != value:
                raise ValueError("Sealed formal recipe profile/source identity changed.")
        recipe = saved_recipe
    saved = load_trusted_checkpoint(args.resume, map_location="cpu")
    if saved.get("fit_identity") != {"recipe": recipe, "mode": "direct"}:
        raise ValueError("Strict formal same-arm resume rejects changed profile, data, flow, or schedule identity.")
    ensure_formal_resume_identity(output, "formal_recipe.json", recipe)
    begin, best, history = int(saved["epoch"]), saved["best_metric"], list(saved["history"])
    stop_after = int(args.stop_after if args.stop_after is not None else
        (profile["data"]["startup_validation"]["maximum_new_epochs"] if args.startup_benchmark else recipe["horizon"]))
    if args.startup_benchmark and stop_after > int(profile["data"]["startup_validation"]["maximum_new_epochs"]):
        raise ValueError("Disposable thermal startup may not exceed three absolute epochs.")
    if not begin < stop_after <= int(recipe["horizon"]):
        raise ValueError("Declared formal stop must follow saved age and stay within the sealed horizon.")
    device = torch.device(args.device)
    load_start = perf_counter()
    model = ThermalSourceResponse(recipe["core_configs"]["direct"], **recipe["adapter_config"]).to(device)
    model.load_state_dict(saved["thermal_state_dict"], strict=True)
    from channelthermal.dependency_flow import ThermalFlowReader
    flow_model = ThermalFlowReader("D-sep", flow["flow_reader_config"]).to(device)
    flow_model.load_state_dict(flow["flow_state_dict"], strict=True)
    flow_model.eval()
    for parameter in flow_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["schedule"]["initial_lr"],
        weight_decay=recipe["weight_decay"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_seconds = perf_counter() - load_start
    receipt = {"pid": os.getpid(), "started_unix": started, "device": str(device),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"), "begin_epoch": begin,
        "status": "running", "load_seconds": load_seconds, "dataset_load_seconds": dataset_load_seconds,
        "profile_data_setup_seconds": perf_counter() - setup_start,
        "validation_scope": validation_scope, "startup_benchmark": bool(args.startup_benchmark),
        "case_visits": 0, "optimizer_updates": 0, "training_seconds": 0.,
        "validation_seconds": 0., "save_seconds": 0.}
    atomic_write_json(output / "active_process.json", receipt)
    calibration = recipe["calibration"]
    train_count = len(cases)
    effective_cases = int(budget["effective_cases"])
    microbatch_cases = int(budget["microbatch_cases"])
    for epoch in range(begin + 1, stop_after + 1):
        epoch_start = perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        learning_rate = thermal_learning_rate(epoch, recipe["schedule"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        order = np.random.default_rng(epoch).permutation(train_count)
        totals = {name: 0. for name in ("reconstruction", "thermal_selector", "fluid", "surface", "material", "q_proxy")}
        response_total, operator_total = 0., 0.
        neural_rows = response_rows = operator_rows = operator_columns = 0
        response_neural_rows = operator_neural_rows = operator_unique_rows = 0
        fluid_queries = material_queries = surface_queries = 0
        gradient_norms = []
        update_count = case_visits = 0
        model.train()
        for effective_start in range(0, train_count, effective_cases):
            effective = order[effective_start:effective_start + effective_cases]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(effective), microbatch_cases):
                indices = effective[micro_start:micro_start + microbatch_cases]
                batch = sample_primary(cases, indices, epoch, device, budget=budget)
                terms, prepared, prediction = reconstruction_terms(model, batch, stats,
                    float(profile["loss"]["q_proxy_coefficient"]))
                loss = terms["reconstruction"].sum() / len(effective)
                if use_operator:
                    residual, work = operator_loss(model, prepared, balances, indices, epoch, device, stats,
                        rows_per_case=int(budget["operator_rows_per_case"]))
                    loss = loss + calibration["operator_coefficient"] * residual * len(indices) / len(effective)
                    operator_total += float(residual.detach()) * len(indices)
                    operator_rows += work["operator_rows"]
                    operator_columns += work["operator_source_columns"]
                    operator_neural_rows += int(work["neural_stencil_receiver_rows"])
                    operator_unique_rows += int(work["unique_stencil_receiver_rows"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite formal native reconstruction/operator loss.")
                loss.backward()
                for name, value in terms.items():
                    totals[name] += float(value.detach().sum())
                neural_rows += int(prediction["native_neural_receiver_rows"])
                fluid_queries += int(batch["query_xy"].shape[0] * batch["query_xy"].shape[1])
                present = batch["structure"]["module_present"] > .5
                material_queries += int(present.sum().item()) * int(batch["module_internal_query_points"].shape[-2])
                surface_queries += int((present[..., None] * prepared.stencils["surface"].valid.reshape(
                    prediction["pred_interface"].shape[:-1])).sum().item())
            family = families[(epoch + update_count) % len(families)]
            response, work = response_loss(model, family, epoch, device,
                calibration["response_scales"], budget)
            (calibration["response_coefficient"] * response).backward()
            response_total += float(response.detach())
            response_rows += sum(int(value) for name, value in work.items() if name != "response_neural_receiver_rows")
            response_neural_rows += int(work["response_neural_receiver_rows"])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"])
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError("Nonfinite or zero actual formal R-direct parameter gradient.")
            gradient_norms.append(float(norm))
            optimizer.step()
            update_count += 1
            case_visits += len(effective)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory = 0
        train_seconds = perf_counter() - epoch_start
        row = {"epoch": epoch,
            **{f"train_{name}": value / case_visits for name, value in totals.items()},
            "response_loss": response_total / update_count,
            "operator_loss": operator_total / case_visits if use_operator else 0.,
            "learning_rate": learning_rate, "case_visits": case_visits, "optimizer_updates": update_count,
            "fluid_reconstruction_queries": fluid_queries, "material_reconstruction_queries": material_queries,
            "surface_reconstruction_queries": surface_queries, "response_target_rows": response_rows,
            "operator_rows": operator_rows, "operator_source_column_rows": operator_columns,
            "neural_reconstruction_rows": neural_rows, "neural_response_rows": response_neural_rows,
            "neural_operator_rows": operator_neural_rows, "unique_operator_receiver_rows": operator_unique_rows,
            "neural_total_rows": neural_rows + response_neural_rows + operator_neural_rows,
            "neural_work_receipt_schema": 3, "gradient_norm_mean": float(np.mean(gradient_norms)),
            "train_seconds": train_seconds, "peak_allocated_memory_bytes": peak_memory}
        requested_stop = stop_requested(output)
        review = epoch % int(profile["checkpointing"]["monitoring_interval_epochs"]) == 0 \
            or epoch == stop_after or requested_stop
        validation_seconds = 0.
        save_start = None
        if review:
            model.eval()
            validation_start = perf_counter()
            validation_model = compatibility_cases if compatibility_cases is not None else validation
            composition = validate_composition(model, flow_model, validation_model, stats, device, budget,
                float(profile["loss"]["q_proxy_coefficient"]))
            validation_rows = composition["rows"]
            if compatibility_cases is not None:
                primary_summary = summarize_composition_rows(validation_rows, primary_validation_ids,
                    profile["data"]["formal_validation"]["primary_scope"])
                compatibility_summary = summarize_composition_rows(validation_rows, compatibility_validation_ids,
                    profile["data"]["formal_validation"]["compatibility_scope"])
            else:
                primary_summary = summarize_composition_rows(validation_rows, validation_case_ids, validation_scope)
                compatibility_summary = None
            score = primary_summary["thermal_selector"]
            validation_seconds = perf_counter() - validation_start
            row.update(validation_thermal_selector=score,
                validation_five_field_standardized_mse=primary_summary["standardized_five_field_mse"],
                validation_physical_rmse_by_channel=primary_summary["physical_rmse_by_channel"],
                validation_scope=validation_scope, validation_seconds=validation_seconds)
            if compatibility_summary is not None:
                row["compatibility_five_field_standardized_mse"] = compatibility_summary["standardized_five_field_mse"]
                row["compatibility_physical_rmse_by_channel"] = compatibility_summary["physical_rmse_by_channel"]
            save_start = perf_counter()
            atomic_write_json(output / f"validation_epoch_{epoch:04d}.json",
                {"primary": primary_summary, "compatibility": compatibility_summary,
                 "rows": validation_rows, "flow_component_age": int(flow["epoch"]),
                 "normalization_stats_sha256": stats_sha256(stats)})
        history.append(row)
        save_seconds = 0.
        if review:
            improved = score < best and epoch % int(profile["checkpointing"]["monitoring_interval_epochs"]) == 0
            if epoch % int(profile["checkpointing"]["monitoring_interval_epochs"]) == 0:
                best = min(best, score)
            payload = formal_checkpoint_payload(model, optimizer, flow, train_config, recipe, epoch, best, history,
                saved.get("aggregate_process_seconds", 0.) + perf_counter() - start)
            atomic_save_checkpoint_payload(output / "latest_model.pt", payload)
            milestone_epochs = {int(value) for value in profile["checkpointing"]["milestone_epochs"]}
            if epoch in milestone_epochs:
                atomic_save_checkpoint_payload(output / f"epoch_{epoch:04d}_model.pt", payload)
            if improved and bool(profile["checkpointing"]["save_best_field"]):
                atomic_save_checkpoint_payload(output / "best_by_field_mse_model.pt", payload)
            atomic_write_json(output / "history.json", history)
            plot_history(history, output / "thermal_learning.pdf")
            save_seconds = perf_counter() - save_start
            row["save_seconds"] = save_seconds
            history[-1]["save_seconds"] = save_seconds
        row["validation_seconds"] = validation_seconds
        row.setdefault("save_seconds", save_seconds)
        receipt["case_visits"] += case_visits
        receipt["optimizer_updates"] += update_count
        receipt["training_seconds"] += train_seconds
        receipt["validation_seconds"] += validation_seconds
        receipt["save_seconds"] += save_seconds
        receipt["completed_epoch"] = epoch
        receipt["peak_allocated_memory_bytes"] = max(int(receipt.get("peak_allocated_memory_bytes", 0)), peak_memory)
        atomic_write_json(output / "history.json", history)
        atomic_write_json(output / "active_process.json", receipt)
        print(json.dumps(row, sort_keys=True), flush=True)
        if requested_stop:
            acknowledge_stop(output, epoch=epoch)
            break
    receipt.update(status="stopped_resumable" if requested_stop else "completed", ended_unix=time(),
        process_seconds=perf_counter() - start)
    atomic_write_json(output / "active_process.json", receipt)
    sessions_path = output / "resource_sessions.json"
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
    sessions.append(receipt)
    atomic_write_json(sessions_path, sessions)
    atomic_write_json(output / "fit_summary.json", {**receipt,
        "best_metric": best, "new_case_visits": receipt["case_visits"],
        "new_optimizer_updates": receipt["optimizer_updates"],
        "new_primary_fluid_queries": sum(int(row["fluid_reconstruction_queries"]) for row in history if row["epoch"] > begin),
        "training_seconds": receipt["training_seconds"], "validation_seconds": receipt["validation_seconds"],
        "save_seconds": receipt["save_seconds"], "load_seconds": load_seconds,
        "peak_allocated_memory_bytes": receipt.get("peak_allocated_memory_bytes", 0),
        "flow_component_age": int(flow["epoch"]), "flow_checkpoint_sha256": recipe["flow_checkpoint_sha256"],
        "formal_dataset_binding_sha256": flow["formal_dataset_binding"]["source_metadata_sha256"],
        "normalization_stats_sha256": normalization_binding["stats_sha256"]})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, help="Read-only dataset/DEV22 reference checkpoint; not a thermal initializer.")
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-file", help="Explicit maintained full-TRAIN profile; omitting it replays fixed25_v1 development.")
    parser.add_argument("--run-identity", help="Unique manual startup or formal output identity.")
    parser.add_argument("--startup-benchmark", action="store_true",
        help="Disposable full-TRAIN startup; caps new thermal epochs at three and uses DEV22 only.")
    parser.add_argument("--atlas-directory", default=str(ROOT / "diagnostics/generated/interactions/physical_response_atlas_20260926/families"))
    parser.add_argument("--operator-decision", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mode", choices=("direct", "group"))
    parser.add_argument("--recipe")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    start, started = perf_counter(), time()
    torch.set_num_threads(1)
    set_seed(0)
    if args.profile_file:
        return run_formal(args, start, started)
    if args.startup_benchmark:
        raise ValueError("--startup-benchmark requires an explicit maintained --profile-file.")
    if args.stop_after is None:
        args.stop_after = 100
    output = validate_generated_output(args.output); output.mkdir(parents=True, exist_ok=True)
    parent_path, flow_path = Path(args.parent).resolve(), Path(args.flow_checkpoint).resolve()
    parent = load_trusted_checkpoint(parent_path, map_location="cpu")
    flow = load_trusted_checkpoint(flow_path, map_location="cpu")
    if flow.get("dependency_policy") != "D-sep" or flow.get("epoch") != 2500:
        raise ValueError("The new pair uses the same exact completed D-sep2500 endpoint.")
    cases, manifest = read_primary(parent, "train")
    validation, val_manifest = read_primary(parent, "test")
    if manifest != val_manifest:
        raise ValueError("Primary/validation manifest mismatch.")
    families = read_response_families(args.atlas_directory)
    decision = json.loads(Path(args.operator_decision).read_text())
    if decision.get("operator_constraint") not in ("qualified", "disabled_with_reason"):
        raise ValueError("Optional operator must be explicitly qualified or disabled_with_reason before sealing.")
    if not decision.get("reason"):
        raise ValueError("Operator decision requires its qualification or disabling reason.")
    use_operator = decision["operator_constraint"] == "qualified"
    balances = build_balances(cases) if use_operator else []
    if args.prepare_only:
        models = {mode: construct(mode) for mode in ("direct", "group")}
        common = [name for name in models["direct"].state_dict() if name in models["group"].state_dict()
                  and name.split(".")[1] not in ("far_head",)]
        equal = all(torch.equal(models["direct"].state_dict()[name], models["group"].state_dict()[name]) for name in common)
        if not equal:
            raise ValueError("Paired common tensors must initialize identically.")
        calibration = calibrate(models, cases, families, parent["global_normalization_stats"], balances, use_operator)
        recipe = {"identity": "thermal_source_response_pair2500_v1", "seed": 0, "horizon": 2500,
            "budget": BUDGET, "schedule": SCHEDULE, "weight_decay": 1e-5, "gradient_clip": 1.,
            "normalization_parent": str(parent_path), "normalization_parent_sha256": _file_sha256(parent_path),
            "flow_checkpoint": str(flow_path), "flow_checkpoint_sha256": _file_sha256(flow_path),
            "manifest_sha256": manifest["manifest_sha256"], "training_case_ids": [c["case_id"] for c in cases],
            "validation_case_ids": [c["case_id"] for c in validation], "response_fit_families": list(TRAIN_FAMILIES),
            "response_atlas_directory": str(Path(args.atlas_directory).resolve()), "operator_decision": decision,
            "response_sources": [{"family_id": f["family_id"], "path": f["source"], "sha256": f["source_sha256"]} for f in families],
            "calibration": calibration, "primary_loss": "mean standardized fluid/surface/material T MSE + .05 standardized q_proxy MSE",
            "selector": "minimum all22 mean standardized fluid/surface/material reconstruction at100cadence",
            "core_configs": {mode: models[mode].core_config for mode in models},
            "adapter_config": models["direct"].adapter_config(),
            "parameters": {mode: sum(p.numel() for p in models[mode].parameters()) for mode in models},
            "common_initial_tensors_exact_equal": equal, "operator_rows_training_only": True,
            "new_solver_attempts": 0, "prepare_seconds": perf_counter() - start}
        atomic_write_json(output / "paired_recipe.json", recipe)
        for mode, model in models.items():
            arm = output / f"R-{mode}"; arm.mkdir(exist_ok=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
            payload = checkpoint_payload(model, optimizer, parent, recipe, mode, 0, float("inf"), [], 0.)
            atomic_save_checkpoint_payload(arm / "latest_model.pt", payload)
        print(json.dumps(recipe), flush=True)
        return 0
    if args.mode is None or args.recipe is None or args.resume is None:
        raise ValueError("Fitting requires one sealed recipe/mode and its explicit latest checkpoint.")
    recipe = json.loads(Path(args.recipe).read_text())
    if (recipe["normalization_parent_sha256"] != _file_sha256(parent_path)
            or recipe["flow_checkpoint_sha256"] != _file_sha256(flow_path)
            or recipe["manifest_sha256"] != manifest["manifest_sha256"]
            or recipe["response_sources"] != [{"family_id": f["family_id"], "path": f["source"], "sha256": f["source_sha256"]} for f in families]
            or recipe["operator_decision"] != decision):
        raise ValueError("Sealed recipe source/flow/manifest/operator identity changed.")
    saved = load_trusted_checkpoint(args.resume, map_location="cpu")
    if saved["fit_identity"] != {"recipe": recipe, "mode": args.mode}:
        raise ValueError("Strict same-arm resume rejects changed sealed recipe.")
    device = torch.device(args.device)
    model = ThermalSourceResponse(recipe["core_configs"][args.mode], **recipe["adapter_config"]).to(device)
    model.load_state_dict(saved["thermal_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    begin, best, history = saved["epoch"], saved["best_metric"], saved["history"]
    if not begin < args.stop_after <= recipe["horizon"]:
        raise ValueError("Declared stop must follow saved age and remain within sealed horizon.")
    receipt = {"pid": os.getpid(), "started_unix": started, "device": str(device),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"), "begin_epoch": begin, "status": "running"}
    atomic_write_json(output / "active_process.json", receipt)
    calibration = recipe["calibration"]
    for epoch in range(begin + 1, args.stop_after + 1):
        epoch_start = perf_counter()
        for group in optimizer.param_groups: group["lr"] = thermal_learning_rate(epoch)
        order = np.random.default_rng(epoch).permutation(150)
        totals = {name: 0. for name in ("reconstruction", "thermal_selector", "fluid", "surface", "material", "q_proxy")}
        response_total, operator_total, neural_rows, response_rows, operator_rows, operator_columns = 0., 0., 0, 0, 0, 0
        response_neural_rows, operator_neural_rows, operator_unique_rows = 0, 0, 0
        gradient_norms = []
        model.train()
        for update, effective_start in enumerate(range(0, 150, 48)):
            effective = order[effective_start:effective_start + 48]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(effective), 8):
                indices = effective[micro_start:micro_start + 8]
                batch = sample_primary(cases, indices, epoch, device)
                terms, prepared, prediction = reconstruction_terms(model, batch, parent["global_normalization_stats"])
                loss = terms["reconstruction"].sum() / len(effective)
                if use_operator:
                    residual, work = operator_loss(model, prepared, balances, indices, epoch, device, parent["global_normalization_stats"])
                    loss = loss + calibration["operator_coefficient"] * residual * len(indices) / len(effective)
                    operator_total += float(residual.detach()) * len(indices)
                    operator_rows += work["operator_rows"]; operator_columns += work["operator_source_columns"]
                    operator_neural_rows += work["neural_stencil_receiver_rows"]
                    operator_unique_rows += work["unique_stencil_receiver_rows"]
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite native reconstruction/operator loss")
                loss.backward()
                for name, value in terms.items(): totals[name] += float(value.detach().sum())
                neural_rows += int(prediction["native_neural_receiver_rows"])
            family = families[(epoch + update) % len(families)]
            response, work = response_loss(model, family, epoch, device, calibration["response_scales"])
            (calibration["response_coefficient"] * response).backward()
            response_total += float(response.detach())
            response_rows += sum(value for name, value in work.items() if name != "response_neural_receiver_rows")
            response_neural_rows += work["response_neural_receiver_rows"]
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm) or norm <= 0: raise FloatingPointError("Nonfinite/zero actual parameter gradient")
            gradient_norms.append(float(norm)); optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize(device)
        row = {"epoch": epoch, **{f"train_{name}": value / 150 for name, value in totals.items()},
            "response_loss": response_total / 4, "operator_loss": operator_total / 150,
            "learning_rate": thermal_learning_rate(epoch), "case_visits": 150, "optimizer_updates": 4,
            "fluid_reconstruction_queries": 153600, "material_reconstruction_queries": int(sum(c["structure"]["module_present"].sum() for c in cases)) * 32,
            "surface_reconstruction_queries": int(sum(c["structure"]["module_present"].sum() for c in cases)) * 16,
            "response_target_rows": response_rows, "operator_rows": operator_rows,
            "operator_source_column_rows": operator_columns, "neural_reconstruction_rows": neural_rows,
            "neural_response_rows": response_neural_rows, "neural_operator_rows": operator_neural_rows,
            "unique_operator_receiver_rows": operator_unique_rows,
            "neural_total_rows": neural_rows + response_neural_rows + operator_neural_rows,
            "neural_work_receipt_schema": 2,
            "gradient_norm_mean": float(np.mean(gradient_norms)), "train_seconds": perf_counter() - epoch_start}
        requested_stop = stop_requested(output)
        review = epoch % 100 == 0 or epoch == args.stop_after or requested_stop
        if review:
            model.eval(); validation_start = perf_counter()
            score, rows = validate(model, validation, parent["global_normalization_stats"], device)
            row.update(validation_thermal_selector=score, validation_seconds=perf_counter() - validation_start)
            atomic_write_json(output / f"validation_epoch_{epoch:04d}.json", {"score": score, "rows": rows})
        history.append(row); print(json.dumps(row), flush=True)
        if epoch == 10:
            atomic_write_json(output / "first10_forecast.json", {"epochs": 10,
                "complete_training_seconds": sum(r["train_seconds"] for r in history if r["epoch"] <= 10),
                "mean_complete_epoch_seconds": float(np.mean([r["train_seconds"] for r in history if r["epoch"] <= 10])),
                "projected2500_training_seconds": 2500 * float(np.mean([r["train_seconds"] for r in history if r["epoch"] <= 10])),
                "scheduled_objectives_included": ["native reconstruction", "TRAIN measured positive response", "qualified discrete kernel residual" if use_operator else "operator disabled"],
                "timing_scope": "data transfer/context/native neural reads/role extraction/backward/ordinary updates; separate validation/save/loading"})
        if review:
            improved = score < best and epoch % 100 == 0
            best = min(best, score) if epoch % 100 == 0 else best
            payload = checkpoint_payload(model, optimizer, parent, recipe, args.mode, epoch, best, history,
                saved["aggregate_process_seconds"] + perf_counter() - start)
            atomic_save_checkpoint_payload(output / "latest_model.pt", payload)
            if epoch % 100 == 0: atomic_save_checkpoint_payload(output / f"epoch_{epoch:04d}_model.pt", payload)
            if improved: atomic_save_checkpoint_payload(output / "best_by_field_mse_model.pt", payload)
            atomic_write_json(output / "history.json", history)
            plot_history(history, output / "thermal_learning.pdf")
        if requested_stop:
            acknowledge_stop(output, epoch=epoch); break
    receipt.update(status="completed" if not requested_stop else "stopped_resumable", ended_unix=time(),
        process_seconds=perf_counter() - start, completed_epoch=epoch)
    atomic_write_json(output / "active_process.json", receipt)
    sessions_path = output / "resource_sessions.json"
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
    sessions.append(receipt); atomic_write_json(sessions_path, sessions)
    atomic_write_json(output / "fit_summary.json", {**receipt, "best_metric": best,
        "new_case_visits": (epoch - begin) * 150, "new_optimizer_updates": (epoch - begin) * 4,
        "new_primary_fluid_queries": (epoch - begin) * 153600})
    return 0


def checkpoint_payload(model, optimizer, parent, recipe, mode, epoch, best, history, process_seconds):
    return {"checkpoint_schema_version": 1, "case_id": "ThermalChannel", "model_family": "honf_forward",
        "workflow": "forward", "stage": SOURCE_RESPONSE_ID, "source_response_identity": SOURCE_RESPONSE_ID,
        "case_capability": SOURCE_RESPONSE_CAPABILITY, "channel_order": ["u", "v", "p", "omega", "temperature"],
        "source_response_config": {"core": model.core_config, "adapter": model.adapter_config()},
        "thermal_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "schedule_state": {"completed_epoch": epoch, "declaration": SCHEDULE},
        "fit_identity": {"recipe": recipe, "mode": mode}, "epoch": epoch, "current_epoch": epoch,
        "best_metric": best, "history": history, "aggregate_process_seconds": process_seconds,
        "flow_checkpoint": recipe["flow_checkpoint"], "flow_checkpoint_sha256": recipe["flow_checkpoint_sha256"],
        "global_normalization_stats": parent["global_normalization_stats"],
        "global_normalization_config": parent["global_normalization_config"], "train_config": parent["train_config"]}


if __name__ == "__main__":
    raise SystemExit(main())
