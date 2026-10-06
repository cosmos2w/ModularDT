#!/usr/bin/env python3
"""Fit the sealed paired source-response recipe from native TRAIN observations.

No generator, reference adapter, inverse search, solver or thermal teacher is
called. Stored TRAIN velocity appears only in the optional residual stencil.
"""
from __future__ import annotations

import argparse
import copy
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

from honf_runtime.compat import load_trusted_checkpoint, recursive_to_device, set_seed
from honf_runtime.run_store import atomic_write_json

TRAIN_FAMILIES = ("0001", "0318", "0333", "0348")
MANIFEST_FINGERPRINT = "933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044"
COLLATE = ChannelThermalBatchCollator()
SCHEDULE = {"hold_epochs": 1000, "total_epochs": 2500, "initial_lr": 3e-4, "final_lr": 3e-6}
BUDGET = {"fluid_queries": 1024, "material_queries_per_module": 32, "surface_queries_per_module": 16,
          "microbatch_cases": 8, "effective_cases": 48, "operator_rows_per_case": 128}


def thermal_learning_rate(epoch):
    if epoch <= 1000:
        return 3e-4
    return 3e-6 + .5 * (3e-4 - 3e-6) * (1 + math.cos(math.pi * min(epoch - 1000, 1500) / 1500))


def read_primary(parent, split):
    config = copy.deepcopy(parent["train_config"]["dataset"])
    manifest = resolve_development_manifest(config, config["packed_h5_path"])
    if manifest["manifest_sha256"] != MANIFEST_FINGERPRINT:
        raise ValueError("The new family requires literal fixed25_v1 membership, not another150/22 cohort.")
    ids = development_case_ids(manifest, split)
    if len(ids) != (150 if split == "train" else 22):
        raise ValueError("Source response uses the unchanged fixed25_v1 150/22 membership.")
    dataset = GlobalChannelThermalDataset(config["packed_h5_path"], split=split, points_per_case=None,
        normalize_inputs=False, normalize_targets=False, include_grid=split == "train",
        random_point_sampling=False, seed=0, case_ids=ids, normalizer=H5Normalizer(parent["global_normalization_stats"]))
    cases = [dataset[index] for index in range(len(dataset))]
    for case in cases:
        present = case["structure"]["module_present"] > .5
        if not np.array_equal(np.flatnonzero(present), np.arange(int(present.sum()))):
            raise ValueError("This packed residual binding requires audited contiguous original module slots.")
        case["structure"]["module_source_ids"] = np.arange(len(present), dtype=np.int64)
    return cases, manifest


def sample_primary(cases, indices, epoch, device, training=True):
    samples = []
    for index in indices:
        original = cases[int(index)]
        seed = (0 if training else 1000) + int(index) * 104729 + (epoch * 1000003 if training else 0)
        rng = np.random.default_rng(seed)
        fluid = rng.choice(len(original["query_xy"]), 1024, replace=False)
        material = rng.choice(len(original["module_internal_query_points"]), 32, replace=False)
        sample = {"structure": original["structure"], "query_xy": original["query_xy"][fluid],
            "field_targets": original["field_targets"][fluid], "point_weights": original["point_weights"][fluid],
            "interface_target": original["interface_target"][:, ::4],
            "interface_condition_valid_mask": original["interface_condition_valid_mask"][:, ::4],
            "module_internal_temperature_points": original["module_internal_temperature_points"][:, material],
            "module_internal_query_points": original["module_internal_query_points"][material]}
        samples.append(sample)
    return recursive_to_device(COLLATE(samples), device)


def scalar_scale(stats, key, channel=None):
    if key == "interface_targets_std" and key not in stats:
        key = "interface_target_std"
    value = np.asarray(stats[key]).reshape(-1)
    return max(float(value[0 if channel is None else channel]), 1e-6)


def reconstruction_terms(model, batch, stats):
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
    return {"reconstruction": thermal + .05 * q, "thermal_selector": thermal,
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


def response_loss(model, family, epoch, device, scales):
    index = TRAIN_FAMILIES.index(family["family_id"])
    rng = np.random.default_rng(index * 104729 + epoch * 1000003)
    valid_fluid = np.flatnonzero(family["fluid_valid"])
    fluid_ids = rng.choice(valid_fluid, 1024, replace=False)
    material_ids = rng.choice(len(family["material_local"]), 32, replace=False)
    structure = {name: torch.as_tensor(value, device=device)[None] for name, value in family["structure"].items()}
    fluid = torch.as_tensor(family["fluid_xy"][fluid_ids], device=device)[None]
    local = torch.as_tensor(family["material_local"][material_ids], device=device)[None]
    prepared = model.prepare_native(structure, fluid, local_query_points=local, ntheta=16)
    delta = model.apply_native(prepared, torch.as_tensor(family["heat_increment"], device=device)[None], increment=True)
    tensor = lambda value: torch.as_tensor(value, device=device, dtype=torch.float32)
    fluid_target = tensor(family["deltas"]["fluid_fields"][fluid_ids, 4])
    surface_target = tensor(family["deltas"]["interface"][:, 0].reshape(family["module_count"], -1)[:, ::4])
    material_target = tensor(family["deltas"]["solid_temperature"].reshape(family["module_count"], -1)[:, material_ids])
    surface_mask = tensor(family["interface_valid"][:, 0].reshape(family["module_count"], -1)[:, ::4])
    material_mask = tensor(family["material_valid"][:, material_ids])
    terms = [((delta["fluid_temperature"][0, :, 0] - fluid_target) / scales["fluid"]).square().mean(),
        (((delta["pred_interface"][0, ..., 0] - surface_target) / scales["surface"]).square() * surface_mask).sum() / surface_mask.sum().clamp_min(1),
        (((delta["pred_internal_temperature"][0, ..., 0] - material_target) / scales["material"]).square() * material_mask).sum() / material_mask.sum().clamp_min(1)]
    return sum(terms) / 3, {"response_fluid_rows": 1024, "response_surface_rows": 16 * family["module_count"],
        "response_material_rows": 32 * family["module_count"], "response_neural_receiver_rows": int(prepared.grid_indices.numel())}


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


def operator_loss(model, prepared, balances, indices, epoch, device, stats):
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
        for category in categories:
            valid = np.flatnonzero(category)
            chosen.extend(rng.choice(valid, 32, replace=len(valid) < 32).tolist())
        rows.append(chosen)
    residual, receipt = sampled_kernel_residual(model.core, prepared.context, balance,
        torch.as_tensor(rows, device=device), prepared.context.lengths)
    residual = residual / scalar_scale(stats, "field_std_by_channel", 4)
    per_case = (residual.square() * present[:, None]).sum((1, 2)) / (present.sum(-1) * 128).clamp_min(1)
    return per_case.mean(), {**receipt, "operator_rows": len(indices) * 128,
                  "operator_source_columns": int(prepared.source_present.sum().item()) * 128}


def calibrate(models, cases, families, stats, balances, use_operator):
    scales = response_scales(families)
    counts = [int(c["structure"]["module_present"].sum()) for c in cases]
    selected = [next(i for i, count in enumerate(counts) if count == wanted) for wanted in (1, 3, 10, 12)]
    measured = []
    for mode, model in models.items():
        reconstruction_gradients, q_gradients, operator_gradients, response_gradients = [], [], [], []
        for index in selected:
            batch = sample_primary(cases, [index], 1, torch.device("cpu"))
            terms, prepared, _ = reconstruction_terms(model, batch, stats)
            reconstruction_gradients.append(gradient_norm(terms["thermal_selector"].mean(), model))
            terms, prepared, _ = reconstruction_terms(model, batch, stats)
            q_gradients.append(gradient_norm(.05 * terms["q_proxy"].mean(), model))
            if use_operator:
                terms, prepared, _ = reconstruction_terms(model, batch, stats)
                loss, _ = operator_loss(model, prepared, balances, [index], 1, torch.device("cpu"), stats)
                operator_gradients.append(gradient_norm(loss, model))
        for family in families:
            loss, _ = response_loss(model, family, 1, torch.device("cpu"), scales)
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
def validate(model, cases, stats, device):
    rows = []
    for start in range(0, len(cases), 8):
        batch = sample_primary(cases, range(start, min(start + 8, len(cases))), 0, device, False)
        terms, _, _ = reconstruction_terms(model, batch, stats)
        for offset in range(len(terms["thermal_selector"])):
            rows.append({"case_id": cases[start + offset]["case_id"],
                **{name: float(value[offset]) for name, value in terms.items()}})
    return float(np.mean([row["thermal_selector"] for row in rows])), rows


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([row["epoch"] for row in history], [row["train_thermal_selector"] for row in history], label="TRAIN thermal")
    review = [row for row in history if "validation_thermal_selector" in row]
    axes[0].plot([row["epoch"] for row in review], [row["validation_thermal_selector"] for row in review], "o-", label="22 exposed DEV")
    axes[0].set(xlabel="New thermal epochs", ylabel="Standardized three-role MSE", yscale="log"); axes[0].legend()
    axes[1].plot([row["epoch"] for row in history], [row["learning_rate"] for row in history])
    axes[1].set(xlabel="New thermal epochs", ylabel="Actual learning rate")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, help="Read-only normalization/primary lineage checkpoint.")
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atlas-directory", default=str(ROOT / "diagnostics/generated/interactions/physical_response_atlas_20260926/families"))
    parser.add_argument("--operator-decision", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mode", choices=("direct", "group"))
    parser.add_argument("--recipe")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    start, started = perf_counter(), time()
    torch.set_num_threads(1)
    set_seed(0)
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
