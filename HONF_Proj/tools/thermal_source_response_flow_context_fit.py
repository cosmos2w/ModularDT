"""Continue one matched fixed25_v1 Thermal pair with optional predicted u/v context.

The R-geom and R-flowctx arms share the retained R-direct e2500 weights,
AdamW moments, D-sep e2500 flow partner, data identities, objectives and
schedule. The only trainable difference is a zero-initialized 3-to-hidden
projection of frozen-flow predicted standardized u/v plus a fluid-validity bit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter, time

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "Case_ThermalChannel/src", ROOT / "tools"):
    if str(source) not in os.sys.path:
        os.sys.path.insert(0, str(source))

import numpy as np
import thermal_source_response_fit as fit
import torch
from channelthermal.dependency_flow import ThermalFlowReader
from channelthermal.source_response import (
    SOURCE_RESPONSE_CAPABILITY,
    SOURCE_RESPONSE_ID,
    SourceResponseThermalModel,
    ThermalSourceResponse,
)
from channelthermal.training.checkpoints import _file_sha256, atomic_save_checkpoint_payload
from channelthermal.training.stop_request import acknowledge_stop, stop_requested
from thermal_development import validate_generated_output

from honf_runtime.compat import load_trusted_checkpoint, set_seed
from honf_runtime.run_store import atomic_write_json

SCHEDULE = {
    "warmup_epochs": 20,
    "warmup_start_lr": 3.0e-6,
    "peak_lr": 5.0e-5,
    "hold_through_epoch": 1000,
    "total_new_epochs": 2500,
    "final_lr": 3.0e-6,
}
ARMS = {
    "R-geom": "ThermalFlowContext_Run4001_Rgeom_fixed25_v1",
    "R-flowctx": "ThermalFlowContext_Run4002_Rflowctx_fixed25_v1",
}
FLOW_FEATURE_SCHEMA = "predicted_physical_uv_train_standardized_plus_fluid_valid_v1"
Q_PROXY_COEFFICIENT = 0.05
DETAIL_CASE_IDS = ("0277", "0291", "0294", "0687")
DETAIL_EPOCHS = frozenset((100, 500, 1000, 2500))
SCENE_KEYS = (
    "module_centers", "module_present", "material_params", "re", "u_in",
    "domain_length_x", "domain_length_y", "t_in", "t_wall",
)


def _sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _scene_signature(structure):
    digest = hashlib.sha256()
    for key in SCENE_KEYS:
        if key not in structure:
            digest.update(key.encode("utf-8") + b"\0<absent>\0")
            continue
        value = structure[key]
        if torch.is_tensor(value):
            if value.requires_grad:
                raise ValueError("Cached predicted-flow context is restricted to fixed-input scenes without geometry gradients.")
            array = value.detach().cpu().contiguous().numpy()
        else:
            array = np.ascontiguousarray(value)
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _feature_structure(case, device):
    """Whitelist only geometry/prescribed context for frozen D-sep reads."""
    raw = case["structure"]
    structure = {}
    for key in SCENE_KEYS:
        if key not in raw:
            continue
        tensor = torch.as_tensor(raw[key], device=device, dtype=torch.float32)
        structure[key] = tensor.unsqueeze(0)
    return structure


def _fit_feature_cache(parent, flow_path, train_cases, validation_cases, families, device, output):
    flow_hash = _file_sha256(flow_path)
    adapter = parent["source_response_config"]["adapter"]
    environment_count = int(adapter["environment_nx"]) * int(adapter["environment_ny"])
    cache_path = output / "environment_flow_features.pt"
    metadata_path = output / "environment_flow_features.json"
    expected_meta = {
        "schema_version": 1,
        "feature_schema": FLOW_FEATURE_SCHEMA,
        "flow_checkpoint": str(flow_path),
        "flow_checkpoint_sha256": flow_hash,
        "development_manifest_sha256": fit.MANIFEST_FINGERPRINT,
        "training_case_ids": [case["case_id"] for case in train_cases],
        "validation_case_ids": [case["case_id"] for case in validation_cases],
        "response_family_ids": [family["family_id"] for family in families],
    }
    if cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        if saved.get("metadata") != expected_meta:
            raise ValueError("Cached flow features are bound to another checkpoint, split, or response-family set.")
        expected_keys = {
            *(f"train:{case['case_id']}" for case in train_cases),
            *(f"validation:{case['case_id']}" for case in validation_cases),
            *(f"response:{family['family_id']}" for family in families),
        }
        if set(saved.get("features", {})) != expected_keys:
            raise ValueError("Cached flow feature membership differs from the sealed training inputs.")
        for key, record in saved["features"].items():
            case = _feature_case(key, train_cases, validation_cases, families)
            if record.get("scene_sha256") != _scene_signature(case["structure"]):
                raise ValueError(f"Cached predicted-flow feature scene changed for {key}.")
            if tuple(record["value"].shape) != (environment_count, 3) or not torch.isfinite(record["value"]).all():
                raise ValueError(f"Cached predicted-flow feature tensor is malformed for {key}.")
        if not metadata_path.exists():
            atomic_write_json(metadata_path, {**expected_meta, "cache_load_seconds": 0.0,
                "cache_created_by": "previous matched-pair process"})
        return saved["features"], 0.0, str(cache_path)

    flow_saved = load_trusted_checkpoint(flow_path, map_location="cpu")
    if flow_saved.get("dependency_policy") != "D-sep" or int(flow_saved.get("epoch", 0)) != 2500:
        raise ValueError("Flow feature preparation requires the exact retained development D-sep2500 parent.")
    flow = ThermalFlowReader("D-sep", flow_saved["flow_reader_config"])
    flow.load_state_dict(flow_saved["flow_state_dict"], strict=True)
    flow = flow.to(device).eval().requires_grad_(False)
    thermal = ThermalSourceResponse(parent["source_response_config"]["core"],
        **{**parent["source_response_config"]["adapter"], "environment_flow_context": True}).to(device).eval()
    provider = SourceResponseThermalModel(thermal, flow, parent["global_normalization_stats"]).eval()
    cases = {
        **{f"train:{case['case_id']}": case for case in train_cases},
        **{f"validation:{case['case_id']}": case for case in validation_cases},
        **{f"response:{family['family_id']}": family for family in families},
    }
    started = perf_counter()
    features = {}
    solid_invalid_counts = {}
    with torch.no_grad():
        for key, case in cases.items():
            structure = _feature_structure(case, device)
            value = provider.predicted_environment_flow_features(structure)[0].detach().cpu().contiguous()
            if value.shape != (environment_count, 3) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Frozen D-sep produced invalid context features for {key}.")
            features[key] = {"scene_sha256": _scene_signature(case["structure"]), "value": value}
            solid_invalid_counts[key] = int((value[:, 2] == 0).sum())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    payload = {"metadata": expected_meta, "features": features}
    fd, temporary = tempfile.mkstemp(prefix=".environment_flow_features.", suffix=".pt", dir=output)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    atomic_write_json(metadata_path, {**expected_meta, "feature_cache_seconds": elapsed,
        "scene_count": len(features), "invalid_solid_donor_counts": solid_invalid_counts,
        "source": "frozen predicted D-sep flow queried at native Thermal environment coordinates",
        "cache_scope": "fixed-geometry training/dev/declared TRAIN response inputs only"})
    return features, elapsed, str(cache_path)


def _feature_case(key, train_cases, validation_cases, families):
    scope, case_id = key.split(":", 1)
    candidates = {"train": train_cases, "validation": validation_cases, "response": families}[scope]
    field = "family_id" if scope == "response" else "case_id"
    return next(case for case in candidates if case[field] == case_id)


def _inherited_objective_contract(parent_recipe):
    """Copy the retained TRAIN calibration without silently dropping scales."""
    expected_loss = "mean standardized fluid/surface/material T MSE + .05 standardized q_proxy MSE"
    if parent_recipe.get("primary_loss") != expected_loss:
        raise ValueError("The matched pair only inherits the qualified R-direct reconstruction objective.")
    calibration = parent_recipe["calibration"]
    response_scales = calibration.get("response_scales")
    if set(response_scales or {}) != {"fluid", "surface", "material"} or not all(
        np.isfinite(float(value)) and float(value) > 0 for value in response_scales.values()
    ):
        raise ValueError("Retained TRAIN response scales are missing or invalid.")
    operator_decision = copy.deepcopy(parent_recipe["operator_decision"])
    if operator_decision.get("operator_constraint") != calibration.get("operator_constraint"):
        raise ValueError("Inherited operator decision and coefficient calibration disagree.")
    return {
        "q_proxy_coefficient": Q_PROXY_COEFFICIENT,
        "response_coefficient": float(calibration["response_coefficient"]),
        "response_scales": {key: float(response_scales[key]) for key in ("fluid", "surface", "material")},
        "operator_coefficient": float(calibration["operator_coefficient"]),
        "operator_decision": operator_decision,
    }


def _optimizer_for(model, parent_optimizer_state, parent_model, *, flow_context):
    parent_group = parent_optimizer_state["param_groups"][0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(parent_group["lr"]),
        betas=tuple(parent_group["betas"]), eps=float(parent_group["eps"]),
        weight_decay=float(parent_group["weight_decay"]), amsgrad=bool(parent_group.get("amsgrad", False)))
    if not flow_context:
        optimizer.load_state_dict(copy.deepcopy(parent_optimizer_state))
        _assert_parent_moments(optimizer, model, parent_optimizer_state, parent_model)
        return optimizer

    old_names = [name for name, _parameter in parent_model.named_parameters()]
    child_names = [name for name, _parameter in model.named_parameters()]
    if child_names[:len(old_names)] != old_names or len(child_names) != len(old_names) + 1:
        raise ValueError("The flow projection must append exactly one parameter after unchanged parent parameters.")
    old_ids = parent_optimizer_state["param_groups"][0]["params"]
    child_ids = optimizer.state_dict()["param_groups"][0]["params"]
    if len(old_ids) != len(old_names) or len(child_ids) != len(child_names):
        raise ValueError("Parent AdamW parameter inventory cannot be mapped by stable named-parameter order.")
    extended = copy.deepcopy(parent_optimizer_state)
    extended["state"] = {
        child_ids[index]: copy.deepcopy(parent_optimizer_state["state"][old_ids[index]])
        for index in range(len(old_ids)) if old_ids[index] in parent_optimizer_state["state"]
    }
    extended["param_groups"][0]["params"] = child_ids
    optimizer.load_state_dict(extended)
    _assert_parent_moments(optimizer, model, parent_optimizer_state, parent_model)
    if model.environment_flow_projection.weight in optimizer.state:
        raise ValueError("New flow projection must start with empty AdamW moments.")
    return optimizer


def _assert_parent_moments(optimizer, child_model, parent_optimizer_state, parent_model):
    parent_named = list(parent_model.named_parameters())
    child_named = list(child_model.named_parameters())
    parent_ids = parent_optimizer_state["param_groups"][0]["params"]
    parent_names = [name for name, _parameter in parent_named]
    child_parent_names = [name for name, _parameter in child_named[:len(parent_named)]]
    if len(parent_named) != len(parent_ids) or child_parent_names != parent_names:
        raise ValueError("Parent AdamW state no longer aligns with unchanged named parameters.")
    child_parameters = dict(child_named)
    for index, (name, _old_parameter) in enumerate(parent_named):
        new_parameter = child_parameters[name]
        old_state = parent_optimizer_state["state"].get(parent_ids[index], {})
        new_state = optimizer.state.get(new_parameter, {})
        if set(old_state) != set(new_state):
            raise ValueError(f"AdamW state fields changed for retained parameter {name}.")
        for key, old_value in old_state.items():
            new_value = new_state[key]
            if torch.is_tensor(old_value):
                if not torch.equal(old_value.detach().cpu(), new_value.detach().cpu()):
                    raise ValueError(f"AdamW moment {key} changed for retained parameter {name}.")
            elif old_value != new_value:
                raise ValueError(f"AdamW scalar state {key} changed for retained parameter {name}.")


def _arm_model(parent, arm, device):
    adapter = copy.deepcopy(parent["source_response_config"]["adapter"])
    is_context = arm == "R-flowctx"
    if is_context:
        adapter["environment_flow_context"] = True
    model = ThermalSourceResponse(parent["source_response_config"]["core"], **adapter).to(device)
    parent_state = parent["thermal_state_dict"]
    target_state = model.state_dict()
    if not set(parent_state).issubset(target_state):
        raise ValueError("New child removed one or more retained R-direct parameters.")
    target_state.update(parent_state)
    model.load_state_dict(target_state, strict=True)
    if is_context and torch.count_nonzero(model.environment_flow_projection.weight):
        raise ValueError("Predicted-flow projection must recover the parent exactly at update zero.")
    return model


def _arm_recipe(pair_recipe, parent, arm):
    adapter = copy.deepcopy(parent["source_response_config"]["adapter"])
    if arm == "R-flowctx":
        adapter["environment_flow_context"] = True
    return {**pair_recipe, "run_identity": ARMS[arm], "arm": arm,
        "adapter_config": adapter, "parameters": pair_recipe["parameters"][arm]}


def _payload(model, optimizer, parent, pair_recipe, arm_recipe, arm, epoch, best, history, process_seconds):
    return {
        "checkpoint_schema_version": 1,
        "case_id": "ThermalChannel",
        "model_family": "honf_forward",
        "workflow": "forward",
        "stage": SOURCE_RESPONSE_ID,
        "source_response_identity": SOURCE_RESPONSE_ID,
        "case_capability": SOURCE_RESPONSE_CAPABILITY,
        "channel_order": ["u", "v", "p", "omega", "temperature"],
        "source_response_config": {"core": model.core_config, "adapter": model.adapter_config()},
        "thermal_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "schedule_state": {"completed_epoch": epoch, "declaration": SCHEDULE},
        "fit_identity": {"recipe": arm_recipe, "mode": "direct", "arm": arm},
        "epoch": epoch,
        "current_epoch": epoch,
        "parent_epoch": int(parent["epoch"]),
        "continuation_parent_checkpoint": pair_recipe["thermal_parent_checkpoint"],
        "continuation_parent_sha256": pair_recipe["thermal_parent_sha256"],
        "best_metric": best,
        "history": history,
        "aggregate_process_seconds": process_seconds,
        "flow_checkpoint": pair_recipe["flow_checkpoint"],
        "flow_checkpoint_sha256": pair_recipe["flow_checkpoint_sha256"],
        "global_normalization_stats": parent["global_normalization_stats"],
        "global_normalization_config": parent.get("global_normalization_config"),
        "train_config": parent["train_config"],
    }


def _make_pair_recipe(parent_path, flow_path, parent, flow, train_cases, validation_cases):
    parent_recipe = parent["fit_identity"]["recipe"]
    flow_recipe = flow["fit_identity"]
    inherited_objectives = _inherited_objective_contract(parent_recipe)
    if int(parent.get("epoch", 0)) != 2500 or parent_recipe.get("identity") != "thermal_source_response_pair2500_v1":
        raise ValueError("R-geom/R-flowctx continuation requires the literal retained R-direct2500 parent.")
    if parent["fit_identity"].get("mode") != "direct" or parent["source_response_config"]["core"].get("mode") != "direct":
        raise ValueError("Only the retained R-direct source-response parent is eligible.")
    if int(flow.get("epoch", 0)) != 2500 or flow.get("dependency_policy") != "D-sep":
        raise ValueError("The paired frozen flow partner must be the completed development D-sep2500 checkpoint.")
    if flow_recipe.get("development_manifest_sha256") != fit.MANIFEST_FINGERPRINT:
        raise ValueError("D-sep parent is not bound to fixed25_v1.")
    if parent_recipe.get("manifest_sha256") != fit.MANIFEST_FINGERPRINT:
        raise ValueError("R-direct parent is not bound to fixed25_v1.")
    if parent_recipe.get("flow_checkpoint_sha256") != _file_sha256(flow_path):
        raise ValueError("R-direct parent and D-sep checkpoint identities do not match.")
    if _file_sha256(Path(parent_recipe["flow_checkpoint"])) != _file_sha256(flow_path):
        raise ValueError("R-direct's recorded D-sep endpoint differs from the requested partner.")
    if parent["train_config"]["dataset"] != flow["train_config"]["dataset"]:
        raise ValueError("R-direct and D-sep training data bindings differ.")
    stats_left, stats_right = parent["global_normalization_stats"], flow["global_normalization_stats"]
    if set(stats_left) != set(stats_right) or any(not np.array_equal(stats_left[k], stats_right[k]) for k in stats_left):
        raise ValueError("R-direct and D-sep normalization metadata differs.")
    if len(train_cases) != 150 or len(validation_cases) != 22:
        raise ValueError("The new continuation requires exactly fixed25_v1 150 TRAIN / 22 DEV cases.")
    old_optimizer = parent["optimizer_state_dict"]
    if len(old_optimizer.get("param_groups", [])) != 1:
        raise ValueError("Retained R-direct parent must have one AdamW parameter group.")
    return {
        "schema_version": 1,
        "identity": "thermal_source_response_flow_context_pair2500_new_v1",
        "seed": 0,
        "thermal_parent_checkpoint": str(parent_path),
        "thermal_parent_sha256": _file_sha256(parent_path),
        "thermal_parent_run_identity": parent_recipe["identity"],
        "thermal_parent_epoch": 2500,
        "flow_checkpoint": str(flow_path),
        "flow_checkpoint_sha256": _file_sha256(flow_path),
        "flow_parent_policy": "D-sep",
        "flow_parent_epoch": 2500,
        "development_manifest_sha256": fit.MANIFEST_FINGERPRINT,
        "training_case_ids": [case["case_id"] for case in train_cases],
        "validation_case_ids": [case["case_id"] for case in validation_cases],
        "normalization_source": "identical retained fixed25_v1 TRAIN-only parent statistics",
        "normalization_stats_sha256": _sha256_json({k: np.asarray(v).tolist()
            for k, v in sorted(parent["global_normalization_stats"].items())}),
        "budget": copy.deepcopy(parent_recipe["budget"]),
        "schedule": copy.deepcopy(SCHEDULE),
        "optimizer": {"name": "AdamW", "weight_decay": float(old_optimizer["param_groups"][0]["weight_decay"]),
            "betas": list(old_optimizer["param_groups"][0]["betas"]),
            "eps": float(old_optimizer["param_groups"][0]["eps"]), "gradient_clip": 1.0,
            "parent_moments": "copied by unchanged named parameter order; new projection state empty"},
        "core_configs": {"direct": copy.deepcopy(parent["source_response_config"]["core"])},
        "base_adapter_config": copy.deepcopy(parent["source_response_config"]["adapter"]),
        "environment_grid_shape": [int(parent["source_response_config"]["adapter"]["environment_ny"]),
            int(parent["source_response_config"]["adapter"]["environment_nx"])],
        "environment_donor_count": int(parent["source_response_config"]["adapter"]["environment_nx"])
            * int(parent["source_response_config"]["adapter"]["environment_ny"]),
        "parameters": {},
        "arms": ARMS,
        "environment_flow_feature_schema": FLOW_FEATURE_SCHEMA,
        "environment_flow_source": "frozen predicted D-sep physical u/v at existing Thermal environment donors",
        "environment_flow_scale": "same TRAIN-only field means/stds stored in both parent checkpoints",
        "solid_donor_rule": "retain donor and measure; zero standardized u/v and set valid=0 inside any active disk",
        "target_flow_input": False,
        "heat_dependency": "D-sep predictor and thermal coefficient context do not read current heat",
        "training_feature_cache": "fixed-input only; scene SHA256 checked; not used for inference or geometry VJPs",
        "objectives": {
            "reconstruction": "retained standardized fluid/surface/material thermal selector plus 0.05 standardized q proxy",
            "response": "the four retained TRAIN response families with inherited coefficient/scales",
            "operator": "inherited qualified TRAIN-only discrete residual decision and coefficient",
            **inherited_objectives,
        },
        "monitoring": {"interval_epochs": 100, "all_validation_cases": 22,
            "selector": "minimum mean standardized fluid/surface/material reconstruction across all fixed DEV22 cases",
            "milestones_new_epochs": list(range(100, 2501, 100))},
        "per_epoch_work": {"training_case_visits": 150, "optimizer_updates": 4,
            "fluid_queries": 153600, "response_families": 4, "operator_rows_per_case": 128},
        "maximum_new_solver_attempts": 0,
        "maximum_new_formal_or_inverse_runs": 0,
    }


def prepare_pair(args):
    output = validate_generated_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    parent_path, flow_path = Path(args.parent).resolve(), Path(args.flow_checkpoint).resolve()
    parent = load_trusted_checkpoint(parent_path, map_location="cpu")
    flow = load_trusted_checkpoint(flow_path, map_location="cpu")
    train_cases, train_manifest = fit.read_primary(parent, "train")
    validation_cases, val_manifest = fit.read_primary(parent, "test")
    if train_manifest["manifest_sha256"] != val_manifest["manifest_sha256"]:
        raise ValueError("Primary and validation memberships differ.")
    recipe = _make_pair_recipe(parent_path, flow_path, parent, flow, train_cases, validation_cases)
    model_probe = {}
    parent_model = ThermalSourceResponse(parent["source_response_config"]["core"],
        **parent["source_response_config"]["adapter"])
    for arm, _run_identity in ARMS.items():
        model = _arm_model(parent, arm, torch.device("cpu"))
        model_probe[arm] = sum(parameter.numel() for parameter in model.parameters())
    recipe["parameters"] = model_probe
    atlas_directory = Path(args.atlas_directory).expanduser().resolve()
    family_paths = fit.read_response_families(atlas_directory)
    recipe["response_sources"] = [{"family_id": family["family_id"], "path": family["source"],
        "sha256": family["source_sha256"]} for family in family_paths]
    expected_recipe_path = output / "paired_recipe.json"
    if expected_recipe_path.exists():
        existing = json.loads(expected_recipe_path.read_text())
        if existing != recipe:
            raise ValueError("Output already contains a different frozen pair recipe.")
    else:
        atomic_write_json(expected_recipe_path, recipe)
    arm_receipts = {}
    for arm, _run_identity in ARMS.items():
        model = _arm_model(parent, arm, torch.device("cpu"))
        optimizer = _optimizer_for(model, parent["optimizer_state_dict"], parent_model,
            flow_context=(arm == "R-flowctx"))
        arm_recipe = _arm_recipe(recipe, parent, arm)
        arm_dir = output / arm
        arm_dir.mkdir(exist_ok=True)
        payload = _payload(model, optimizer, parent, recipe, arm_recipe, arm, 0, float("inf"), [], 0.0)
        latest = arm_dir / "latest_model.pt"
        if latest.exists():
            old = load_trusted_checkpoint(latest, map_location="cpu")
            if old.get("fit_identity") != payload["fit_identity"]:
                raise ValueError(f"Existing {arm} initialization belongs to another child identity.")
        else:
            atomic_save_checkpoint_payload(latest, payload)
        arm_receipts[arm] = {
            "run_identity": _run_identity,
            "initial_checkpoint": str(latest),
            "parameters": model_probe[arm],
            "parent_named_parameter_count": len(list(parent_model.named_parameters())),
            "optimizer_state_entries": len(optimizer.state),
            "parent_adamw_moment_audit": "exact old step/exp_avg/exp_avg_sq copied by stable named-parameter order",
            "new_projection_adamw_state": "empty" if arm == "R-flowctx" else "not_applicable",
            "new_projection_zero": (arm != "R-flowctx" or not bool(torch.count_nonzero(model.environment_flow_projection.weight))),
        }
    receipt = {"pair_recipe": str(expected_recipe_path), "thermal_parent_sha256": recipe["thermal_parent_sha256"],
        "flow_parent_sha256": recipe["flow_checkpoint_sha256"], "manifest_sha256": recipe["development_manifest_sha256"],
        "new_solver_attempts": 0, "GPU_launched": False, "arms": arm_receipts,
        "training_case_visits_per_epoch": 150, "updates_per_epoch": 4,
        "epoch100_commands": _stage100_commands(args, output)}
    atomic_write_json(output / "thermal_pair_preparation_receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


def _stage100_commands(args, output):
    script = Path(__file__).resolve()
    common = (f"PYTHONPATH=src:Case_ThermalChannel/src:tools CUDA_VISIBLE_DEVICES=<GPU> "
        f"conda run --no-capture-output -n ModularDT python {script} --parent {Path(args.parent).resolve()} "
        f"--flow-checkpoint {Path(args.flow_checkpoint).resolve()} --output {output} --atlas-directory {Path(args.atlas_directory).resolve()} ")
    return {
        "R-geom": common.replace("<GPU>", "0") + "--arm R-geom --resume " + str(output / "R-geom/latest_model.pt") + " --stop-after 100 --device cuda:0",
        "R-flowctx": common.replace("<GPU>", "0") + "--arm R-flowctx --resume " + str(output / "R-flowctx/latest_model.pt") + " --stop-after 100 --device cuda:0",
    }


def _validate_child_recipe(saved, pair_recipe, arm):
    expected = _arm_recipe(pair_recipe, load_trusted_checkpoint(pair_recipe["thermal_parent_checkpoint"],
        map_location="cpu"), arm)
    if saved.get("fit_identity") != {"recipe": expected, "mode": "direct", "arm": arm}:
        raise ValueError("Strict child resume rejects a changed pair, arm, or parent identity.")


def _cache_tensor(feature_records, scope, cases, device):
    values = []
    for case in cases:
        key = f"{scope}:{case['case_id']}"
        record = feature_records[key]
        if record["scene_sha256"] != _scene_signature(case["structure"]):
            raise ValueError(f"Fixed-input predicted-flow cache is stale for {key}.")
        values.append(record["value"])
    return torch.stack(values).to(device)


def _prepare_detail_export(parent, flow_path, model, device, validation_cases):
    """Use the public live-flow composition on the fixed detailed DEV panel."""
    validation_ids = {case["case_id"] for case in validation_cases}
    if not set(DETAIL_CASE_IDS).issubset(validation_ids):
        raise ValueError("Detailed field panel must remain within the sealed fixed25_v1 DEV22 cohort.")
    cases = fit.read_scoped_cases(parent["train_config"]["dataset"], "test", DETAIL_CASE_IDS,
        parent["global_normalization_stats"], include_grid=True)
    flow_saved = load_trusted_checkpoint(flow_path, map_location="cpu")
    flow = ThermalFlowReader("D-sep", flow_saved["flow_reader_config"])
    flow.load_state_dict(flow_saved["flow_state_dict"], strict=True)
    flow = flow.to(device).eval().requires_grad_(False)
    composed = SourceResponseThermalModel(model, flow, parent["global_normalization_stats"]).to(device).eval()
    return composed, cases


def _export_detail_fields(composed, cases, arm_dir, epoch, arm, checkpoint_path, pair_recipe, device):
    started = perf_counter()
    arrays = {"case_ids": np.asarray(DETAIL_CASE_IDS), "field_names": np.asarray(["u", "v", "p", "omega", "temperature"])}
    rows = []
    for index, case in enumerate(cases):
        prediction = composed.predict_native_sample(case, device=device)
        target = np.asarray(case["steady_field"], dtype=np.float32)
        predicted = np.asarray(prediction["pred_field_grid"], dtype=np.float32)
        module_mask = np.asarray(case.get("module_mask", np.zeros(target.shape[:2])), dtype=bool)
        if target.shape != predicted.shape or target.shape[-1] != 5 or module_mask.shape != target.shape[:2]:
            raise ValueError(f"Detailed physical field arrays have incompatible shapes for {case['case_id']}.")
        valid = (~module_mask)[..., None] & np.isfinite(target) & np.isfinite(predicted)
        metrics = {}
        for channel, name in enumerate(("u", "v", "p", "omega", "temperature")):
            channel_valid = valid[..., channel]
            residual = predicted[..., channel] - target[..., channel]
            selected_residual = residual[channel_valid].astype(np.float64)
            selected_target = target[..., channel][channel_valid].astype(np.float64)
            l2_target = float(np.linalg.norm(selected_target))
            metrics[name] = {
                "fluid_rmse_native_units": float(np.sqrt(np.mean(selected_residual**2))),
                "fluid_relative_l2": float(np.linalg.norm(selected_residual) / l2_target) if l2_target > 0 else None,
                "valid_fluid_grid_points": int(channel_valid.sum()),
            }
        rows.append({"case_id": case["case_id"], "per_field": metrics,
            "module_count": int(np.sum(np.asarray(case["structure"]["module_present"]) > 0.5))})
        arrays[f"x_grid_{index}"] = np.asarray(case["x_grid"], dtype=np.float32)
        arrays[f"y_grid_{index}"] = np.asarray(case["y_grid"], dtype=np.float32)
        arrays[f"target_field_{index}"] = target
        arrays[f"predicted_field_{index}"] = predicted
        arrays[f"residual_field_{index}"] = predicted - target
        arrays[f"module_mask_{index}"] = module_mask.astype(np.uint8)
        arrays[f"predicted_interface_{index}"] = np.asarray(prediction["pred_interface"], dtype=np.float32)
        arrays[f"predicted_material_temperature_{index}"] = np.asarray(
            prediction["pred_internal_temperature"], dtype=np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    field_path = arm_dir / f"physical_fields_epoch_{epoch:04d}.npz"
    np.savez_compressed(field_path, **arrays)
    atomic_write_json(arm_dir / f"physical_field_metrics_epoch_{epoch:04d}.json", {
        "scope": "fixed25_v1 DEV22 detailed representative panel; four cases, separate from all-22 aggregate selector",
        "arm": arm, "run_identity": ARMS[arm], "completed_new_epoch": epoch,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _file_sha256(checkpoint_path),
        "thermal_parent_sha256": pair_recipe["thermal_parent_sha256"],
        "flow_parent_sha256": pair_recipe["flow_checkpoint_sha256"],
        "development_manifest_sha256": pair_recipe["development_manifest_sha256"],
        "field_order": ["u", "v", "p", "omega", "temperature"],
        "units": "native packed dataset units; each channel reported separately",
        "mask": "module interiors excluded; fluid-grid metrics only",
        "provider": "public SourceResponseThermalModel; frozen D-sep flow queried live, not target flow or feature cache",
        "array_archive": str(field_path), "export_seconds": elapsed, "cases": rows,
    })
    return elapsed


@torch.no_grad()
def _validate(model, cases, stats, device, feature_tensor, q_proxy_coefficient):
    rows = []
    for start in range(0, len(cases), fit.BUDGET["microbatch_cases"]):
        stop = min(start + fit.BUDGET["microbatch_cases"], len(cases))
        batch = fit.sample_primary(cases, range(start, stop), 0, device, False, fit.BUDGET)
        features = None if feature_tensor is None else feature_tensor[start:stop]
        terms, _prepared, _prediction = fit.reconstruction_terms(model, batch, stats,
            q_proxy_coefficient=q_proxy_coefficient, environment_flow_features=features)
        for offset in range(stop - start):
            rows.append({"case_id": cases[start + offset]["case_id"],
                **{name: float(value[offset]) for name, value in terms.items()}})
    if len(rows) != 22:
        raise ValueError("Every 100-epoch selector must include the fixed DEV22 panel.")
    return float(np.mean([row["thermal_selector"] for row in rows])), rows


def train_arm(args):
    outer_start = perf_counter()
    started_unix = time()
    output = validate_generated_output(args.output)
    recipe_path = output / "paired_recipe.json"
    if not recipe_path.exists():
        raise ValueError("Run --prepare-only to seal the matched recipe before fitting either arm.")
    pair_recipe = json.loads(recipe_path.read_text())
    if args.arm not in ARMS or pair_recipe.get("arms") != ARMS:
        raise ValueError("Requested arm is not part of the sealed single pair.")
    parent_path, flow_path = Path(args.parent).resolve(), Path(args.flow_checkpoint).resolve()
    if str(parent_path) != pair_recipe["thermal_parent_checkpoint"] or str(flow_path) != pair_recipe["flow_checkpoint"]:
        raise ValueError("Requested checkpoint paths differ from the sealed parent identities.")
    parent = load_trusted_checkpoint(parent_path, map_location="cpu")
    if _file_sha256(parent_path) != pair_recipe["thermal_parent_sha256"] or _file_sha256(flow_path) != pair_recipe["flow_checkpoint_sha256"]:
        raise ValueError("A frozen parent checkpoint changed after recipe sealing.")
    train_cases, manifest = fit.read_primary(parent, "train")
    validation_cases, val_manifest = fit.read_primary(parent, "test")
    if manifest["manifest_sha256"] != pair_recipe["development_manifest_sha256"] or val_manifest != manifest:
        raise ValueError("Fixed25_v1 membership changed after pair sealing.")
    atlas_directory = Path(args.atlas_directory).expanduser().resolve()
    families = fit.read_response_families(atlas_directory)
    if [{"family_id": family["family_id"], "path": family["source"], "sha256": family["source_sha256"]}
            for family in families] != pair_recipe.get("response_sources"):
        raise ValueError("TRAIN response family source identity changed after pair sealing.")
    operator_decision = pair_recipe["objectives"]["operator_decision"]
    use_operator = operator_decision["operator_constraint"] == "qualified"
    balances = fit.build_balances(train_cases) if use_operator else []
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = _arm_model(parent, args.arm, device)
    optimizer = _optimizer_for(model, parent["optimizer_state_dict"],
        ThermalSourceResponse(parent["source_response_config"]["core"],
            **parent["source_response_config"]["adapter"]), flow_context=(args.arm == "R-flowctx"))
    arm_recipe = _arm_recipe(pair_recipe, parent, args.arm)
    arm_dir = output / args.arm
    resume_path = Path(args.resume).resolve()
    saved = load_trusted_checkpoint(resume_path, map_location="cpu")
    _validate_child_recipe(saved, pair_recipe, args.arm)
    model.load_state_dict(saved["thermal_state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    begin, best, history = int(saved["epoch"]), float(saved["best_metric"]), list(saved["history"])
    if args.stop_after is None or not begin < args.stop_after <= int(SCHEDULE["total_new_epochs"]):
        raise ValueError("--stop-after must advance this child without exceeding the frozen 2500-new schedule.")

    feature_cache_seconds = 0.0
    feature_cache_path = None
    train_flow_features = val_flow_features = None
    response_flow_features = {}
    if args.arm == "R-flowctx":
        feature_records, feature_cache_seconds, feature_cache_path = _fit_feature_cache(
            parent, flow_path, train_cases, validation_cases, families, device, output)
        train_flow_features = _cache_tensor(feature_records, "train", train_cases, device)
        val_flow_features = _cache_tensor(feature_records, "validation", validation_cases, device)
        for family in families:
            key = f"response:{family['family_id']}"
            record = feature_records[key]
            if record["scene_sha256"] != _scene_signature(family["structure"]):
                raise ValueError(f"Fixed-input predicted-flow cache is stale for {key}.")
            response_flow_features[family["family_id"]] = record["value"].to(device)[None]
    detail_composed, detail_cases = _prepare_detail_export(parent, flow_path, model, device, validation_cases)

    active = {"pid": os.getpid(), "started_unix": started_unix, "device": str(device),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"), "run_identity": ARMS[args.arm],
        "arm": args.arm, "begin_new_epoch": begin, "target_new_epoch": args.stop_after, "status": "running",
        "thermal_parent_sha256": pair_recipe["thermal_parent_sha256"],
        "flow_parent_sha256": pair_recipe["flow_checkpoint_sha256"],
        "feature_cache_seconds": feature_cache_seconds, "feature_cache_path": feature_cache_path}
    atomic_write_json(arm_dir / "active_process.json", active)
    calibration = pair_recipe["objectives"]
    training_seconds = []
    start_epoch = perf_counter()
    for epoch in range(begin + 1, args.stop_after + 1):
        epoch_start = perf_counter()
        lr = learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        order = np.random.default_rng(epoch).permutation(len(train_cases))
        totals = {name: 0.0 for name in ("reconstruction", "thermal_selector", "fluid", "surface", "material", "q_proxy")}
        response_total, response_rows, response_neural_rows, operator_total = 0.0, 0, 0, 0.0
        operator_rows = operator_columns = 0
        gradient_norms = []
        model.train()
        for update, effective_start in enumerate(range(0, 150, 48)):
            effective = order[effective_start:effective_start + 48]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(effective), 8):
                indices = effective[micro_start:micro_start + 8]
                batch = fit.sample_primary(train_cases, indices, epoch, device, True, fit.BUDGET)
                features = None
                if train_flow_features is not None:
                    if any(torch.is_tensor(value) and value.requires_grad for value in batch["structure"].values()):
                        raise ValueError("Training feature cache cannot be used with differentiable scene inputs.")
                    features = train_flow_features[torch.as_tensor(indices, device=device)]
                terms, prepared, _prediction = fit.reconstruction_terms(model, batch,
                    parent["global_normalization_stats"],
                    q_proxy_coefficient=float(calibration["q_proxy_coefficient"]),
                    environment_flow_features=features)
                loss = terms["reconstruction"].sum() / len(effective)
                if use_operator:
                    residual, work = fit.operator_loss(model, prepared, balances, indices, epoch, device,
                        parent["global_normalization_stats"], rows_per_case=int(fit.BUDGET["operator_rows_per_case"]))
                    loss = loss + float(calibration["operator_coefficient"]) * residual * len(indices) / len(effective)
                    operator_total += float(residual.detach()) * len(indices)
                    operator_rows += int(work["operator_rows"])
                    operator_columns += int(work["operator_source_columns"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite Thermal reconstruction/operator objective.")
                loss.backward()
                for name, value in terms.items():
                    totals[name] += float(value.detach().sum())
            family = families[(epoch + update) % len(families)]
            family_features = response_flow_features.get(family["family_id"])
            response, work = fit.response_loss(model, family, epoch, device,
                calibration["response_scales"], fit.BUDGET, environment_flow_features=family_features)
            (float(calibration["response_coefficient"]) * response).backward()
            response_total += float(response.detach())
            response_rows += sum(int(value) for name, value in work.items() if name != "response_neural_receiver_rows")
            response_neural_rows += int(work["response_neural_receiver_rows"])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError("Nonfinite or zero actual Thermal parameter gradient.")
            gradient_norms.append(float(norm))
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = perf_counter() - epoch_start
        training_seconds.append(epoch_seconds)
        row = {"epoch": epoch, **{f"train_{name}": value / 150 for name, value in totals.items()},
            "response_loss": response_total / 4, "operator_loss": operator_total / 150,
            "learning_rate": lr, "case_visits": 150, "optimizer_updates": 4,
            "fluid_reconstruction_queries": 153600, "material_reconstruction_queries": int(sum(
                case["structure"]["module_present"].sum() for case in train_cases)) * 32,
            "surface_reconstruction_queries": int(sum(case["structure"]["module_present"].sum()
                for case in train_cases)) * 16,
            "response_target_rows": response_rows,
            "response_neural_receiver_rows": response_neural_rows,
            "operator_rows": operator_rows,
            "operator_source_column_rows": operator_columns,
            "gradient_norm_mean": float(np.mean(gradient_norms)), "train_seconds": epoch_seconds,
            "environment_flow_cache": feature_cache_path if args.arm == "R-flowctx" else None}
        requested_stop = stop_requested(arm_dir)
        review = epoch % 100 == 0 or epoch == args.stop_after or requested_stop
        if review:
            model.eval()
            validation_start = perf_counter()
            score, validation_rows = _validate(model, validation_cases,
                parent["global_normalization_stats"], device, val_flow_features,
                float(calibration["q_proxy_coefficient"]))
            row.update(validation_thermal_selector=score,
                validation_seconds=perf_counter() - validation_start)
            atomic_write_json(arm_dir / f"validation_epoch_{epoch:04d}.json",
                {"score": score, "rows": validation_rows,
                 "scope": "fixed25_v1 DEV22 exposed validation", "arm": args.arm,
                 "completed_new_epoch": epoch})
        history.append(row)
        print(json.dumps(row), flush=True)
        if epoch == 10:
            mean_epoch = float(np.mean(training_seconds))
            first10 = history[-10:]
            atomic_write_json(arm_dir / "first10_forecast.json", {
                "arm": args.arm, "epochs_measured": 10,
                "training_seconds_first10": float(sum(training_seconds)),
                "mean_training_seconds_per_epoch": mean_epoch,
                "projected_2500_new_epoch_training_seconds": 2500 * mean_epoch,
                "feature_cache_seconds_before_training": feature_cache_seconds,
                "mean_first10_objectives": {key: float(np.mean([row[key] for row in first10])) for key in (
                    "train_reconstruction", "train_thermal_selector", "train_fluid", "train_surface",
                    "train_material", "train_q_proxy", "response_loss", "operator_loss")},
                "objective_counts": {"native_reconstruction": 150, "optimizer_updates": 4,
                    "measured_response_families_per_epoch": 4,
                    "operator_rows_per_epoch": 150 * int(fit.BUDGET["operator_rows_per_case"]) if use_operator else 0},
                "forecast_scope": "measured epochs include reconstruction, all four response-family losses and the qualified operator residual when enabled",
                "forecast_excludes": ["100-epoch DEV22 evaluation", "detailed fixed-four physical field export", "checkpoint and PDF writes", "outer imports and data load"],
                "case_visits_first10": 1500, "updates_first10": 40,
                "response_target_rows_first10": int(sum(row["response_target_rows"] for row in first10)),
                "response_neural_receiver_rows_first10": int(sum(row["response_neural_receiver_rows"] for row in first10)),
                "operator_rows_first10": int(sum(row["operator_rows"] for row in first10)),
                "new_case_visits_to_2500": 375000, "new_updates_to_2500": 10000,
            })
        if review:
            improved = score < best and epoch % 100 == 0
            if epoch % 100 == 0:
                best = min(best, score)
            payload = _payload(model, optimizer, parent, pair_recipe, arm_recipe, args.arm,
                epoch, best, history, float(saved.get("aggregate_process_seconds", 0.0)) + perf_counter() - start_epoch)
            atomic_save_checkpoint_payload(arm_dir / "latest_model.pt", payload)
            if epoch % 100 == 0:
                atomic_save_checkpoint_payload(arm_dir / f"epoch_{epoch:04d}_model.pt", payload)
            if improved:
                atomic_save_checkpoint_payload(arm_dir / "best_by_field_mse_model.pt", payload)
            atomic_write_json(arm_dir / "history.json", history)
            fit.plot_history(history, arm_dir / "thermal_flow_context_learning.pdf")
            if epoch in DETAIL_EPOCHS:
                _export_detail_fields(detail_composed, detail_cases, arm_dir, epoch, args.arm,
                    arm_dir / f"epoch_{epoch:04d}_model.pt", pair_recipe, device)
        if requested_stop:
            acknowledge_stop(arm_dir, epoch=epoch)
            break

    ended = time()
    process_seconds = perf_counter() - outer_start
    active.update(status="completed" if not stop_requested(arm_dir) else "stopped_resumable",
        ended_unix=ended, process_seconds=process_seconds, completed_new_epoch=epoch,
        new_training_seconds=float(sum(training_seconds)))
    atomic_write_json(arm_dir / "active_process.json", active)
    sessions_path = arm_dir / "resource_sessions.json"
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
    sessions.append(active)
    atomic_write_json(sessions_path, sessions)
    atomic_write_json(arm_dir / "fit_summary.json", {**active,
        "best_metric": best, "new_case_visits": (epoch - begin) * 150,
        "new_optimizer_updates": (epoch - begin) * 4,
        "new_primary_fluid_queries": (epoch - begin) * 153600,
        "new_response_families": 4 * (epoch - begin),
        "new_response_target_rows": int(sum(row["response_target_rows"] for row in history if row["epoch"] > begin)),
        "new_response_neural_receiver_rows": int(sum(row["response_neural_receiver_rows"] for row in history if row["epoch"] > begin)),
        "new_operator_rows": int(sum(row["operator_rows"] for row in history if row["epoch"] > begin)),
        "new_operator_source_column_rows": int(sum(row["operator_source_column_rows"] for row in history if row["epoch"] > begin)),
        "objective": "inherited R-direct reconstruction + four measured TRAIN response families + inherited qualified TRAIN-only operator residual"})
    return 0


def learning_rate(epoch):
    epoch = int(epoch)
    warmup = int(SCHEDULE["warmup_epochs"])
    if epoch <= warmup:
        fraction = epoch / warmup
        return float(SCHEDULE["warmup_start_lr"] + fraction * (SCHEDULE["peak_lr"] - SCHEDULE["warmup_start_lr"]))
    if epoch <= int(SCHEDULE["hold_through_epoch"]):
        return float(SCHEDULE["peak_lr"])
    decay = int(SCHEDULE["total_new_epochs"]) - int(SCHEDULE["hold_through_epoch"])
    fraction = min(max((epoch - int(SCHEDULE["hold_through_epoch"])) / decay, 0.0), 1.0)
    return float(SCHEDULE["final_lr"] + 0.5 * (SCHEDULE["peak_lr"] - SCHEDULE["final_lr"])
        * (1.0 + np.cos(np.pi * fraction)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atlas-directory", default=str(ROOT / "diagnostics/generated/interactions/physical_response_atlas_20260926/families"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    set_seed(0)
    if args.prepare_only:
        if args.arm or args.resume or args.stop_after is not None:
            raise ValueError("Pair sealing cannot be combined with an arm run.")
        return prepare_pair(args)
    if args.arm is None or args.resume is None or args.stop_after is None:
        raise ValueError("Fitting requires one sealed arm, an explicit resume checkpoint and a declared stop.")
    return train_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
