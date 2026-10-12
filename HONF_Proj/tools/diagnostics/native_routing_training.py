"""Build lineage-bound P-R/P-C provider contexts without training or refitting.

The helper accepts only explicit source paths and a CPU-loaded e1000 parent
checkpoint. It constructs a fresh native-routing architecture child and the
original fixed TRAIN provider; continuation state attachment remains the
TrainingEngine architecture-child contract's responsibility.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA = "native_routing_saved_context_v1"
_SOURCE = {
    "thermal": {
        "run_id": "T4111",
        "epoch": 1000,
        "checkpoint_sha256": "917771696ec6558047cf8b025e8733b13edb519461b6c013026a07f9e1783c36",
        "identity_sha256": "2167f4964d14ff041cf7a720a002cb6bfa6a7bbc55a01fc16baeec6c2a7c0c30",
        "recipe_sha256": "224901c43d9fb4185b9c00272bbbe98e0a8d2db9ed7a953b9d214b6c999832ff",
        "manifest_sha256": "933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044",
        "normalization_sha256": "9162379696b687d45f3b0ac5942c01ee787021d5b0361986e6b0572f5bcb1bc1",
        "train_count": 150,
        "dev_count": 22,
        "group_name": "joint",
        "parameter_count": 60,
        "weight_decay": 1.0e-4,
        "seed": 0,
        "effective_cases": 48,
        "microbatch_cases": 8,
        "queries": 1024,
        "flow_readout_law": "native_curl_cell_centred_v1",
        "family": "thermal_native_curl_interaction_routing_v1",
    },
    "wind": {
        "run_id": "W2401",
        "epoch": 1000,
        "checkpoint_sha256": "881a9fb84bc6d48475680d17c94b13657fad413a90b2779fcb720d0f6fc42f11",
        "identity_sha256": "fe7f559dae068f8b0433bdb20673dba455ee52e6a917b16e8d126e785b9f0a91",
        "recipe_sha256": "130dc4ff66e2db23d2266bb845367aadf1faf9728fd28b2955993ec54acb838d",
        "manifest_sha256": "b223e174d7262682a7e1ac1977cb38f721993e14c0340b2dc0f5fb5f95bfa715",
        "normalization_sha256": "6cd6da0f9c14cd623949019ce2c62831311659fc8617dd02c476074c3b55d060",
        "profile_sha256": "21e293f57f1a5eba22b14f4b5f92ec78acc405153c3e8032c4d809239048d28e",
        "train_row_indices_sha256": "f0e12ef8350ab11757c7893ce6c1ab439ce13950c9d85da5f684dcc5c42e69c1",
        "validation_row_indices_sha256": "eec4ba473ebb7de046bc683fa9cfae1be79a6d196974c73b8ddca7033b720520",
        "train_count": 72,
        "dev_count": 24,
        "group_name": "wind_joint_regional",
        "parameter_count": 52,
        "weight_decay": 1.0e-5,
        "seed": 42,
        "effective_cases": 24,
        "microbatch_cases": 8,
        "queries": 4096,
        "family": "wind_native_interaction_routing_v1",
    },
}
_SHARED_SCHEDULE = {
    "peak_lr": 3.0e-4,
    "warmup_start_lr": 3.0e-5,
    "warmup_epochs": 20,
    "hold_through_epoch": 1000,
    "total_epochs": 2500,
    "final_lr": 3.0e-6,
}
_ENGINE_REQUIRED = {
    "total_epochs": 2500,
    "warmup_epochs": 20,
    "open_through_epoch": 20,
    "soft_through_epoch": 20,
    "monitor_every": 100,
    "gradient_clip": 1.0,
    "deterministic_case_order": True,
    "latest_every": 100,
    "curve_every": 100,
    "sampling_version": "case_epoch_v1",
    "training_mode": "joint",
}


def _canonical(value: Any) -> Any:
    """Convert scientific identity values into stable JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False, default=str).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_file(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"source_paths.{name} must be an explicit nonempty path.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Explicit source file {name!r} is unavailable: {path}")
    return path


def _resolved_directory(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"source_paths.{name} must be an explicit nonempty path.")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Explicit source directory {name!r} is unavailable: {path}")
    return path


def _path_record(path: Path, *, content_hash: bool) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "path_sha256": hashlib.sha256(str(resolved).encode("utf-8")).hexdigest(),
        "kind": "file" if resolved.is_file() else "directory",
    }
    if resolved.is_file():
        stat = resolved.stat()
        result.update({"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        if content_hash:
            result["sha256"] = _sha256_file(resolved)
    else:
        stat = resolved.stat()
        result.update({"mtime_ns": int(stat.st_mtime_ns), "directory_fingerprint_sha256": None})
    return result


def _source_paths(factory_config: Mapping[str, Any], domain: str) -> tuple[dict[str, Any], dict[str, Path]]:
    raw = factory_config.get("source_paths")
    if not isinstance(raw, Mapping):
        raise TypeError("factory_config.source_paths must be a mapping of explicit source paths.")
    common = {"checkpoint", "identity", "resolved_recipe", "source_snapshot_directory"}
    domain_keys = ({"dataset_path", "manifest_path", "atlas_directory"} if domain == "thermal" else {
        "data_root", "derived_root", "pilot_root", "manifest_path", "normalization_path",
        "normalization_binding_path", "role_scale_cache_dir", "catalogue_cache_dir",
    })
    expected = common | domain_keys
    if set(raw) != expected:
        raise ValueError(f"source_paths keys must be exactly {sorted(expected)}; received {sorted(raw)}")
    files = {
        name: _resolved_file(raw[name], name)
        for name in ("checkpoint", "identity", "resolved_recipe", "manifest_path")
    }
    files["source_snapshot_directory"] = _resolved_directory(raw["source_snapshot_directory"], "source_snapshot_directory")
    if domain == "thermal":
        files["dataset_path"] = _resolved_file(raw["dataset_path"], "dataset_path")
        files["atlas_directory"] = _resolved_directory(raw["atlas_directory"], "atlas_directory")
    else:
        for name in ("data_root", "derived_root", "pilot_root", "role_scale_cache_dir", "catalogue_cache_dir"):
            files[name] = _resolved_directory(raw[name], name)
        files["normalization_path"] = _resolved_file(raw["normalization_path"], "normalization_path")
        files["normalization_binding_path"] = _resolved_file(raw["normalization_binding_path"], "normalization_binding_path")
    return dict(raw), files


def _validate_snapshot_and_parent(
    domain: str, payload: Mapping[str, Any], paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    expected = _SOURCE[domain]
    checkpoint_sha = _sha256_file(paths["checkpoint"])
    identity_sha = _sha256_file(paths["identity"])
    recipe_sha = _sha256_file(paths["resolved_recipe"])
    if (checkpoint_sha != expected["checkpoint_sha256"]
            or identity_sha != expected["identity_sha256"]
            or recipe_sha != expected["recipe_sha256"]):
        raise ValueError(f"{expected['run_id']} source checkpoint, identity or resolved recipe hash changed.")
    snapshot = paths["source_snapshot_directory"]
    snapshot_names = {
        "checkpoint": "epoch_1000_model.pt",
        "identity": "experiment_identity.json",
        "resolved_recipe": "resolved_recipe.json",
    }
    snapshot_hashes: dict[str, str] = {}
    for key, filename in snapshot_names.items():
        candidate = snapshot / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Source snapshot is missing {candidate}")
        digest = _sha256_file(candidate)
        bound = {"checkpoint": checkpoint_sha, "identity": identity_sha, "resolved_recipe": recipe_sha}[key]
        if digest != bound:
            raise ValueError(f"{expected['run_id']} {key} snapshot differs from its explicit source path.")
        snapshot_hashes[filename] = digest

    identity = json.loads(paths["identity"].read_text(encoding="utf-8"))
    resolved = json.loads(paths["resolved_recipe"].read_text(encoding="utf-8"))
    if _canonical(payload.get("experiment_identity")) != _canonical(identity):
        raise ValueError("Passed checkpoint payload identity differs from the exact saved identity JSON.")
    if _canonical(resolved.get("recipe")) != _canonical(identity.get("recipe")):
        raise ValueError("Resolved recipe wrapper and checkpoint identity recipe differ.")
    if payload.get("arm") != "P" or payload.get("epoch") != expected["epoch"] or payload.get("current_epoch") != expected["epoch"]:
        raise ValueError(f"The source must be the exact {expected['run_id']}/P/e{expected['epoch']} endpoint.")
    if payload.get("workflow") != "joint_regional_fields_v1":
        raise ValueError("The parent checkpoint workflow is not the segmented joint-field P path.")
    if identity.get("sampling_dataset_id") is None:
        raise ValueError("Parent identity omits its case-epoch sampling namespace.")
    sampler = payload.get("sampler_state")
    if (not isinstance(sampler, Mapping) or sampler.get("completed_epoch") != 1000
            or sampler.get("next_epoch") != 1001 or sampler.get("next_case_cursor") != 0
            or sampler.get("case_visits_this_epoch") != expected["train_count"]):
        raise ValueError("Parent sampler is not the exact absolute e1000 continuation boundary.")
    if identity.get("engine_config", {}).get("sampling_version") != "case_epoch_v1":
        raise ValueError("Parent did not use the sealed case_epoch_v1 sampler namespace.")
    provider = identity.get("provider_identity")
    recipe = identity.get("recipe")
    if not isinstance(provider, Mapping) or not isinstance(recipe, Mapping):
        raise TypeError("Parent checkpoint identity lacks provider or resolved recipe mappings.")
    if provider.get("mode", provider.get("arm")) != "P":
        raise ValueError("Parent provider identity is not literal P.")
    if domain == "thermal":
        if (provider.get("task") != "ThermalChannel" or provider.get("model_family") != "thermal_interaction_preserving_native_curl_v1"
                or provider.get("flow_readout_law") != expected["flow_readout_law"]
                or provider.get("manifest_fingerprint") != expected["manifest_sha256"]
                or provider.get("training_case_count") != expected["train_count"]
                or provider.get("validation_case_count") != expected["dev_count"]
                or provider.get("normalization_stats_sha256") != expected["normalization_sha256"]
                or provider.get("response_family_ids") != ["0001", "0318", "0333", "0348"]):
            raise ValueError("T4111 provider identity changed its family, manifest, population, normalization or TRAIN response set.")
        calibration = provider.get("auxiliary_calibration")
        saved_calibration = payload.get("provider_training_state", {}).get("auxiliary_calibration")
        if (not isinstance(calibration, Mapping) or not isinstance(saved_calibration, Mapping)
                or _canonical(calibration) != _canonical(saved_calibration)
                or calibration.get("optimizer_steps") != 0
                or calibration.get("primary_partition") != "fixed25_v1 TRAIN only"
                or calibration.get("development_response_families_loaded") is not False):
            raise ValueError("T4111 calibration is not its exact checkpoint-owned TRAIN-only zero-step receipt.")
        if provider.get("training_case_ids") is None or provider.get("validation_case_ids") is None:
            raise ValueError("T4111 provider identity must retain exact fixed TRAIN and exposed DEV IDs.")
        if len(provider["training_case_ids"]) != expected["train_count"] or len(provider["validation_case_ids"]) != expected["dev_count"]:
            raise ValueError("T4111 saved TRAIN/DEV case ID populations changed.")
        if "0291" not in provider["validation_case_ids"]:
            raise ValueError("The saved exposed Thermal DEV membership no longer includes replay case 0291.")
        if "0291" in calibration.get("response_family_ids", []):
            raise ValueError("Thermal replay case 0291 cannot be relabelled as a TRAIN response family.")
    else:
        objective = provider.get("objective", {})
        calibration = objective.get("scale_calibration") if isinstance(objective, Mapping) else None
        if (provider.get("dataset") != "WindFarm" or provider.get("family") != "wind_interaction_preserving_joint_v1"
                or provider.get("dataset_profile") != "wind_shared_fixed24_v1"
                or provider.get("subset_manifest_sha256") != expected["manifest_sha256"]
                or provider.get("profile_is_formal_fulltrain") is not False
                or provider.get("mode") != "P"
                or provider.get("training", {}).get("sampling_version") != "case_epoch_v1"
                or len(provider.get("train_rows", ())) != expected["train_count"]
                or len(provider.get("validation_rows", ())) != expected["dev_count"]
                or provider.get("train_row_indices_sha256") != expected["train_row_indices_sha256"]
                or provider.get("validation_row_indices_sha256") != expected["validation_row_indices_sha256"]):
            raise ValueError("W2401 provider identity changed its fixed24 profile, mode or TRAIN/DEV membership.")
        model_config = provider.get("standalone_model_config")
        if not isinstance(model_config, Mapping) or model_config.get("core", {}).get("mode") != "P":
            raise ValueError("W2401 saved model config is not its exact direct-P model config.")
        if (not isinstance(calibration, Mapping) or calibration.get("target_values_read") != "TRAIN calibration panel only"
                or calibration.get("wind_test_target_values_read", False) is not False
                or calibration.get("training_fingerprint") != expected["manifest_sha256"]
                or calibration.get("training_rows_sha256") != provider.get("train_row_indices_sha256")
                or calibration.get("scope") != "selected TRAIN rows only"
                or calibration.get("sampling_version") != "case_epoch_v1"
                or set(calibration.get("component_role_scales_mps", {})) != {
                    "background", "downstream_envelope", "hub_slab", "near_turbine", "volume"
                }
                or int(calibration.get("solver_attempts", -1)) != 0):
            raise ValueError("W2401 component-role calibration is not the source checkpoint's exact TRAIN-only receipt.")
        if payload.get("provider_training_state") != {}:
            raise ValueError("Unexpected Wind mutable provider state; role calibration must come from W2401 identity.")

    engine = identity.get("engine_config")
    if not isinstance(engine, Mapping):
        raise TypeError("Parent identity lacks an engine config.")
    for key, value in _ENGINE_REQUIRED.items():
        if engine.get(key) != value:
            raise ValueError(f"Parent engine setting {key!r} changed from the segmented continuation contract.")
    if (engine.get("seed") != expected["seed"]
            or engine.get("effective_cases") != expected["effective_cases"]
            or engine.get("microbatch_cases") != expected["microbatch_cases"]):
        raise ValueError("Parent case batch sizes, initialization seed or absolute schedule changed.")
    groups = identity.get("optimizer_schedule_contract")
    state_groups = payload.get("optimizer_schedule_contract")
    named = payload.get("optimizer_state_by_name")
    if (not isinstance(groups, list) or len(groups) != 1 or _canonical(groups) != _canonical(state_groups)
            or groups[0].get("name") != expected["group_name"]
            or len(groups[0].get("parameter_names", ())) != expected["parameter_count"]
            or groups[0].get("weight_decay") != expected["weight_decay"]
            or _canonical(groups[0].get("schedule")) != _canonical(_SHARED_SCHEDULE)
            or groups[0].get("betas") != [0.9, 0.999] or groups[0].get("eps") != 1.0e-8):
        raise ValueError("Parent named optimizer group, decay, LR schedule or AdamW constants changed.")
    if (not isinstance(named, Mapping) or set(named.get("state_by_name", {})) != set(payload.get("model_state_dict", {}))
            or [group.get("name") for group in named.get("groups", [])] != [expected["group_name"]]
            or named["groups"][0].get("parameter_names") != groups[0]["parameter_names"]):
        raise ValueError("Parent named optimizer moments do not cover the exact source P tensor/group inventory.")
    if len(payload.get("model_state_dict", {})) != expected["parameter_count"]:
        raise ValueError("Parent model tensor inventory changed from the source P contract.")
    snapshots = {
        "checkpoint_sha256": checkpoint_sha,
        "identity_sha256": identity_sha,
        "resolved_recipe_sha256": recipe_sha,
        "snapshot_file_sha256": snapshot_hashes,
    }
    return identity, _canonical(resolved["recipe"]), snapshots


def _engine_for_parent(identity: Mapping[str, Any], device: str):
    from honf_runtime.unified_training import EngineConfig, SelectionPolicy, TrainingEngine

    fields = dict(identity["engine_config"])
    fields["monitor_epochs"] = tuple(fields.get("monitor_epochs", ()))
    if fields.get("checkpoint_epochs") is not None:
        fields["checkpoint_epochs"] = tuple(fields["checkpoint_epochs"])
    config = EngineConfig(**fields)
    selection = SelectionPolicy(**dict(identity["selection_policy"]))
    return TrainingEngine(config, device=device, selection=selection)


def _expected_training_schedule(identity: Mapping[str, Any]):
    from honf_runtime.unified_training import ScheduleSpec

    groups = identity["optimizer_schedule_contract"]
    return ScheduleSpec(**dict(groups[0]["schedule"])), float(groups[0]["weight_decay"])


def _verify_child_group_contract(model: torch.nn.Module, provider: Any,
                                 parent_identity: Mapping[str, Any], target_mode: str) -> dict[str, Any]:
    source_group = parent_identity["optimizer_schedule_contract"][0]
    groups = tuple(provider.optimizer_groups(model, arm=target_mode, stage="joint"))
    if len(groups) != 1:
        raise ValueError("Native routing child must preserve the source's single named AdamW group.")
    group = groups[0]
    names = tuple(name for name, _ in model.named_parameters())
    if (not names or len(set(names)) != len(names)
            or any(not parameter.requires_grad for _, parameter in model.named_parameters())
            or tuple(group.parameter_names) != names
            or group.name != source_group["name"]
            or float(group.weight_decay) != float(source_group["weight_decay"])
            or tuple(group.betas) != tuple(source_group["betas"])
            or float(group.eps) != float(source_group["eps"])
            or _canonical(asdict(group.schedule)) != _canonical(source_group["schedule"])):
        raise ValueError("Native child groups do not give every old/new parameter the contemporaneous parent LR/decay.")
    old_names = tuple(source_group["parameter_names"])
    added = sorted(set(names) - set(old_names))
    if (set(old_names) - set(names) or not added
            or any(not name.startswith("core.native_routing.") for name in added)):
        raise ValueError("Native child parameter inventory differs outside the private routing module.")
    return {
        "group_name": group.name,
        "parent_parameter_count": len(old_names),
        "child_parameter_count": len(names),
        "added_parameter_names": added,
        "all_parameters_trainable": True,
        "schedule": _canonical(asdict(group.schedule)),
        "weight_decay": float(group.weight_decay),
        "betas": list(group.betas),
        "eps": float(group.eps),
    }


def _runtime_source_hashes(project_root: Path) -> dict[str, str]:
    relative = (
        "src/honf_runtime/unified_training.py",
        "src/honf_runtime/architecture_child.py",
        "src/honf_forward_core/interface_fields/interaction_core.py",
        "src/honf_forward_core/interface_fields/interaction_preserving_joint.py",
        "src/honf_forward_core/interface_fields/native_interaction_routing.py",
        "Case_ThermalChannel/src/channelthermal/joint_regional.py",
        "Case_ThermalChannel/src/channelthermal/training/joint_task.py",
        "Case_ThermalChannel/src/channelthermal/training/unified_task.py",
        "Case_WindFarm/src/windfarm/interaction_preserving.py",
        "Case_WindFarm/src/windfarm/joint_regional.py",
        "Case_WindFarm/src/windfarm/training/joint_task.py",
        "Case_WindFarm/src/windfarm/training/unified_task.py",
        "Case_WindFarm/src/windfarm/data.py",
        "Case_WindFarm/src/windfarm/io.py",
        "Case_WindFarm/src/windfarm/normalization.py",
        "Case_WindFarm/src/windfarm/workflows/native_role_cache.py",
    )
    result = {}
    for name in relative:
        path = project_root / name
        if path.is_file():
            result[name] = _sha256_file(path)
    if len(result) != len(relative):
        raise FileNotFoundError("The current provider/runtime source inventory is incomplete.")
    return result


def _metadata_digest(metadata: Mapping[str, Any]) -> str:
    names = ("case", "layout", "layout_index", "wd_deg", "n_turbines", "turbine_xy_D", "U_ref", "D_m", "hub_height_m")
    digest = hashlib.sha256()
    for name in names:
        value = np.ascontiguousarray(np.asarray(metadata[name]))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str((value.shape, value.dtype.str)).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_saved_context(
    factory_config: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    *,
    include_dev_targets: bool = False,
):
    """Build ``(model, provider, engine, provenance)`` from one exact P/e1000 parent.

    ``factory_config`` must contain exactly ``domain``, ``target_mode``,
    ``device`` and ``source_paths``. The optional keyword controls Thermal DEV
    target materialization for a sealed fit that needs held-out monitoring; its
    default keeps qualification and preflight construction TRAIN-only. Wind
    retains lazy DEV loading through ``validation_batches``. No model fitting,
    normalization fitting, calibration, full-population read, solver call or
    optimizer step is performed here.
    """
    required = {"domain", "target_mode", "device", "source_paths"}
    if not isinstance(factory_config, Mapping) or set(factory_config) != required:
        raise ValueError(f"factory_config keys must be exactly {sorted(required)}")
    if type(include_dev_targets) is not bool:
        raise TypeError("include_dev_targets must be boolean.")
    domain = str(factory_config["domain"])
    target_mode = str(factory_config["target_mode"])
    device = str(factory_config["device"])
    if domain not in _SOURCE or target_mode not in ("P-R", "P-C"):
        raise ValueError("Native routing contexts require domain thermal/wind and target_mode P-R/P-C.")
    if not isinstance(checkpoint_payload, Mapping):
        raise TypeError("checkpoint_payload must be the loaded source P/e1000 mapping.")
    raw_paths, paths = _source_paths(factory_config, domain)
    parent_identity, resolved_recipe, source_hashes = _validate_snapshot_and_parent(
        domain, checkpoint_payload, paths,
    )
    parent_provider = parent_identity["provider_identity"]
    parent_recipe = parent_identity["recipe"]
    project_root = Path(__file__).resolve().parents[2]
    if project_root.name != "HONF_Proj":
        raise RuntimeError("Could not resolve the HONF_Proj source root from this helper location.")
    for candidate in (project_root / "src", project_root / "Case_ThermalChannel/src", project_root / "Case_WindFarm/src"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    schedule, weight_decay = _expected_training_schedule(parent_identity)

    if domain == "thermal":
        from channelthermal.joint_regional import InteractionPreservingThermalAdapter
        from channelthermal.training.joint_task import (
            NATIVE_CURL_READOUT_LAW,
            build_thermal_joint_task,
        )

        expected = _SOURCE[domain]
        source_binding = parent_provider["source_binding"]
        if (paths["dataset_path"] != Path(source_binding["dataset_path"]).expanduser().resolve()
                or paths["manifest_path"] != Path(source_binding["development_manifest_path"]).expanduser().resolve()):
            raise ValueError("Thermal dataset and fixed25 manifest must match the checkpoint's exact source binding.")
        recipe = parent_recipe
        config = parent_provider["model_config"]
        adapter = parent_provider["adapter_config"]
        model = InteractionPreservingThermalAdapter(
            mode=target_mode,
            normalization_stats=parent_provider["normalization_stats"],
            hidden=int(config["hidden"]),
            message=int(config["message"]),
            regional_anchors=0,
            depth=int(config["depth"]),
            receiver_tile=int(config["receiver_tile"]),
            nx=int(config["nx"]),
            ny=int(config["ny"]),
            environment_nx=int(adapter["environment_nx"]),
            environment_ny=int(adapter["environment_ny"]),
            forcing_scale=float(adapter["forcing_scale"]),
            collective_width=int(config["collective_width"]),
            locality_prior_strength=None,
            seed=int(config["seed"]),
            max_sources=int(config["max_sources"]),
            flow_readout_law=NATIVE_CURL_READOUT_LAW,
            factorized_affine_execution=bool(config.get("factorized_affine_execution", False)),
        )
        calibration = copy.deepcopy(dict(parent_provider["auxiliary_calibration"]))
        model, provider = build_thermal_joint_task(
            mode=target_mode,
            device=device,
            hidden=int(recipe["hidden"]),
            message=int(recipe["message"]),
            microbatch_size=int(recipe["microbatch_cases"]),
            effective_batch_size=int(recipe["effective_cases"]),
            primary_queries=int(recipe["primary_queries"]),
            manifest=paths["manifest_path"],
            formal_full=False,
            regional_anchors=0,
            locality_prior_strength=0.0,
            receiver_tile=int(recipe["receiver_tile"]),
            seed=int(recipe["seed"]),
            total_epochs=int(recipe["total_epochs"]),
            operator_rows_per_case=int(calibration["operator_rows_per_case"]),
            response_coefficient=float(calibration["response_coefficient"]),
            operator_coefficient=float(calibration["operator_coefficient"]),
            data_path=paths["dataset_path"],
            atlas_directory=paths["atlas_directory"],
            depth=int(recipe["depth"]),
            auxiliary_calibration=calibration,
            saved_normalization_stats=parent_provider["normalization_stats"],
            saved_response_scales=parent_provider["response_scales"],
            optimizer_schedule=schedule,
            weight_decay=weight_decay,
            native_sampling_protocol=None,
            collective_width=int(recipe["collective_width"]),
            max_sources=int(recipe["max_sources"]),
            flow_readout_law=NATIVE_CURL_READOUT_LAW,
            include_dev_targets=include_dev_targets,
            model_override=model,
        )
        child_provider_identity = _canonical(provider.identity_payload())
        child_config = model.model_config()
        expected_config = copy.deepcopy(dict(config))
        expected_config.update({
            "family": expected["family"],
            "mode": target_mode,
            "native_routing": copy.deepcopy(model.core.config["native_routing"]),
        })
        if _canonical(child_config) != _canonical(expected_config):
            raise ValueError("Thermal native adapter exported a noncanonical child config.")
        train_ids = list(provider._train_ids)
        dev_ids = list(provider._validation_ids)
        if child_provider_identity.get("validation_case_ids") != dev_ids:
            raise ValueError("Thermal provider did not retain the exact frozen DEV IDs in its identity.")
        if include_dev_targets:
            if (not provider.dev_targets_materialized or len(provider.validation_cases) != len(dev_ids)
                    or [str(case["case_id"]) for case in provider.validation_cases] != dev_ids):
                raise ValueError("Thermal fit context did not materialize exactly the sealed exposed DEV22 targets.")
        elif provider.dev_targets_materialized or provider.validation_cases:
            raise ValueError("Thermal TRAIN-only context unexpectedly materialized exposed DEV targets.")
        membership_fingerprint = child_provider_identity["manifest_fingerprint"]
        normalization_binding = {
            "kind": "checkpoint_owned_T4111_TRAIN_only",
            "normalization_stats_sha256": child_provider_identity["normalization_stats_sha256"],
            "scope": child_provider_identity["normalization_scope"],
            "no_refit": True,
        }
        calibration_binding = {
            "source": "T4111 checkpoint provider_training_state and provider_identity",
            "receipt_sha256": calibration["payload_sha256"],
            "optimizer_steps": calibration["optimizer_steps"],
            "train_response_family_ids": list(calibration["response_family_ids"]),
            "development_response_families_loaded": calibration["development_response_families_loaded"],
        }
        split_access = {
            "train_target_cases_materialized": len(provider.training_cases),
            "exposed_dev_ids_retained": dev_ids,
            "exposed_dev_target_cases_materialized": len(provider.validation_cases),
            "exposed_dev_targets_materialized": bool(provider.dev_targets_materialized),
            "exposed_dev_target_access_policy": (
                "materialized_at_context_construction_for_sealed_monitoring"
                if include_dev_targets else "not_materialized"
            ),
            "unexposed_original_test_targets_materialized": False,
            "train_response_families_materialized": list(calibration["response_family_ids"]),
        }
        adapter_config = model.adapter_config()
        export_config = child_config
        data_sources = {
            "dataset_path": _path_record(paths["dataset_path"], content_hash=False),
            "manifest_path": _path_record(paths["manifest_path"], content_hash=True),
            "atlas_directory": _path_record(paths["atlas_directory"], content_hash=False),
        }
        response_hashes = {
            family: _sha256_file(paths["atlas_directory"] / f"train_{family}_responses.npz")
            for family in calibration["response_family_ids"]
        }
        if response_hashes != calibration["response_family_source_sha256"]:
            raise ValueError("Thermal source-P response atlas files differ from the saved TRAIN calibration hashes.")
        data_sources["train_response_family_sha256"] = {
            f"train_{family}_responses.npz": digest for family, digest in response_hashes.items()
        }
        extra_data_identity = {"thermal_unexposed_test_targets_read": False}
    else:
        from windfarm.interaction_preserving import InteractionPreservingWindModel
        from windfarm.training.joint_task import build_wind_joint_task

        expected = _SOURCE[domain]
        source_transform = parent_provider["normalizer_and_background_source"]
        if (paths["normalization_path"] != Path(source_transform["normalization_path"]).expanduser().resolve()
                or paths["normalization_binding_path"] != Path(source_transform["binding_path"]).expanduser().resolve()
                or paths["manifest_path"] != (paths["pilot_root"] / "fixed_subset_manifest.json").resolve()):
            raise ValueError("Wind normalization, binding and subset manifest must be the source checkpoint's exact files.")
        if not (paths["derived_root"] / "split_indices.npz").is_file():
            raise FileNotFoundError("Wind source split_indices.npz is missing from the explicit derived root.")
        parent_config = copy.deepcopy(dict(parent_provider["standalone_model_config"]))
        child_config = copy.deepcopy(parent_config)
        child_config["family"] = expected["family"]
        child_config["core"]["mode"] = target_mode
        routing = {
            "schema": "native_interaction_routing_v1",
            "mode": target_mode,
            "policy": "learned_density" if target_mode == "P-R" else "destination_gain",
            "initialization_seed": int(parent_config["seed"]),
        }
        if target_mode == "P-R":
            routing.update({"score_hidden": 32, "score_bound": 2.0})
        else:
            routing["control_hidden"] = 34
        child_config["core"]["native_routing"] = routing
        model = InteractionPreservingWindModel.from_config(child_config)
        objective = parent_provider["objective"]
        model, provider = build_wind_joint_task(
            mode=target_mode,
            device=device,
            microbatch_size=int(parent_recipe["microbatch_cases"]),
            effective_batch_size=int(parent_recipe["effective_cases"]),
            primary_queries=int(parent_recipe["primary_queries"]),
            total_epochs=int(parent_recipe["total_epochs"]),
            receiver_tile=int(parent_recipe["receiver_tile"]),
            seed=int(parent_recipe["seed"]),
            hidden=int(parent_recipe["hidden"]),
            message=int(parent_recipe["message"]),
            regional_anchors=0,
            depth=int(parent_recipe["depth"]),
            locality_prior_strength=0.0,
            formal_full=False,
            data_root=paths["data_root"],
            derived_root=paths["derived_root"],
            pilot_root=paths["pilot_root"],
            manifest_path=paths["manifest_path"],
            normalization_path=paths["normalization_path"],
            normalization_binding_path=paths["normalization_binding_path"],
            role_scale_cache_dir=paths["role_scale_cache_dir"],
            catalogue_cache_dir=paths["catalogue_cache_dir"],
            optimizer_schedule=schedule,
            weight_decay=weight_decay,
            environment_token_shape=(2, 2, 2),
            collective_width=int(parent_recipe["collective_width"]),
            max_sources=int(parent_recipe["max_sources"]),
            saved_role_scale_calibration=copy.deepcopy(objective["scale_calibration"]),
            saved_role_scale_cache_binding=copy.deepcopy(objective["scale_cache_binding"]),
            saved_role_scale_source_sha256=str(objective["scale_source_sha256"]),
            model_override=model,
        )
        child_provider_identity = _canonical(provider.identity_payload())
        export_config = model.export_config()
        if _canonical(export_config) != _canonical(child_config):
            raise ValueError("Wind native adapter did not round-trip the canonical P-R/P-C export config.")
        if (provider.profile_id != "wind_shared_fixed24_v1" or len(provider.train_rows) != expected["train_count"]
                or len(provider.validation_rows) != expected["dev_count"]):
            raise ValueError("Wind provider did not retain the exact fixed24 TRAIN72/DEV24 membership.")
        train_ids = provider.train_rows.astype(int).tolist()
        dev_ids = provider.validation_rows.astype(int).tolist()
        if provider._validation_cache is not None:
            raise ValueError("Wind context unexpectedly materialized validation target batches at construction.")
        membership_fingerprint = child_provider_identity["subset_manifest_sha256"]
        normalization_binding = {
            "kind": "verified_fixed24_TRAIN_only_source_binding",
            "normalization_sha256": child_provider_identity["normalizer_and_background_source"]["normalization_sha256"],
            "profile_sha256": child_provider_identity["normalizer_and_background_source"]["profile_sha256"],
            "binding": child_provider_identity["normalizer_and_background_source"]["binding"],
            "no_refit": True,
        }
        calibration = objective["scale_calibration"]
        calibration_binding = {
            "source": "W2401 checkpoint experiment_identity.provider_identity.objective",
            "calibration_sha256": _json_sha256(calibration),
            "cache_binding": copy.deepcopy(objective["scale_cache_binding"]),
            "source_sha256": objective["scale_source_sha256"],
            "target_values_read": calibration["target_values_read"],
            "calibration_train_rows_sha256": calibration["training_rows_sha256"],
            "test_target_values_read": calibration.get("wind_test_target_values_read", False),
        }
        split_access = {
            "train_metadata_cases_materialized": len(train_ids),
            "exposed_dev_ids_retained": dev_ids,
            "train_target_values_materialized": False,
            "exposed_dev_target_values_materialized": False,
            "exposed_dev_target_access_policy": "lazy_only_when_validation_batches_is_called",
            "wind_test_target_values_read": False,
            "solver_attempts": 0,
        }
        adapter_config = export_config
        data_sources = {
            name: _path_record(paths[name], content_hash=False)
            for name in ("data_root", "derived_root", "pilot_root", "role_scale_cache_dir", "catalogue_cache_dir")
        }
        data_sources.update({
            "manifest_path": _path_record(paths["manifest_path"], content_hash=True),
            "normalization_path": _path_record(paths["normalization_path"], content_hash=True),
            "normalization_binding_path": _path_record(paths["normalization_binding_path"], content_hash=True),
            "split_indices_sha256": _sha256_file(paths["derived_root"] / "split_indices.npz"),
            "compact_geometry_metadata_sha256": _metadata_digest(provider.view.metadata),
        })
        extra_data_identity = {
            "dataset_profile": provider.profile_id,
            "train_rows_sha256": child_provider_identity["train_row_indices_sha256"],
            "validation_rows_sha256": child_provider_identity["validation_row_indices_sha256"],
            "train_layout_indices": child_provider_identity["train_layout_indices"],
            "validation_layout_indices": child_provider_identity["validation_layout_indices"],
        }

    child_sampling_id = None
    if domain == "thermal":
        from honf_runtime.unified_training import _sampling_dataset_id
        child_sampling_id = _sampling_dataset_id(child_provider_identity)
    else:
        child_sampling_id = provider.sampling_dataset_id
    if child_sampling_id != parent_identity["sampling_dataset_id"]:
        raise ValueError("Child provider changed the source P case_epoch_v1 dataset namespace.")
    group_contract = _verify_child_group_contract(model, provider, parent_identity, target_mode)
    engine = _engine_for_parent(parent_identity, device)
    sampling_input = {
        "seed": engine.config.seed,
        "sampling_version": engine.config.sampling_version,
        "sampling_dataset_id": child_sampling_id,
        "source_sampler_state": _canonical(checkpoint_payload["sampler_state"]),
    }
    source_snapshot_files = {
        name: _path_record(paths["source_snapshot_directory"] / name, content_hash=True)
        for name in ("epoch_1000_model.pt", "experiment_identity.json", "resolved_recipe.json")
    }
    source_file_hashes = _runtime_source_hashes(project_root)
    source_file_hashes.update({
        "tools/diagnostics/native_routing_training.py": _sha256_file(Path(__file__).resolve()),
    })
    path_records = {}
    for name in raw_paths:
        path = paths[name]
        content_hash = path.is_file() and name not in ("checkpoint", "dataset_path")
        path_records[name] = _path_record(path, content_hash=content_hash)
    path_records["checkpoint"]["sha256"] = source_hashes["checkpoint_sha256"]
    path_records["identity"]["sha256"] = source_hashes["identity_sha256"]
    path_records["resolved_recipe"]["sha256"] = source_hashes["resolved_recipe_sha256"]
    provenance = {
        "schema": SCHEMA,
        "domain": domain,
        "source_run_id": _SOURCE[domain]["run_id"],
        "source_epoch": _SOURCE[domain]["epoch"],
        "source_arm": "P",
        "source_checkpoint_sha256": source_hashes["checkpoint_sha256"],
        "source_experiment_identity_sha256": source_hashes["identity_sha256"],
        "source_resolved_recipe_sha256": source_hashes["resolved_recipe_sha256"],
        "source_snapshot_sha256": source_hashes["snapshot_file_sha256"],
        "target_mode": target_mode,
        "target_family": _SOURCE[domain]["family"],
        "adapter_config": _canonical(adapter_config),
        "export_config": _canonical(export_config),
        "source_parent_recipe": _canonical(resolved_recipe),
        "source_parent_provider_identity_sha256": _json_sha256(parent_provider),
        "child_provider_identity_sha256": _json_sha256(child_provider_identity),
        "train_ids_or_rows": train_ids,
        "train_membership_fingerprint": membership_fingerprint,
        "dev_ids_or_rows": dev_ids,
        "dev_membership_fingerprint": (
            child_provider_identity.get("validation_case_ids_sha256")
            if domain == "thermal" else child_provider_identity.get("validation_row_indices_sha256")
        ),
        "dev_targets_materialized_at_context_construction": bool(
            split_access.get("exposed_dev_targets_materialized", False)
            or split_access.get("exposed_dev_target_values_materialized", False)
        ),
        "normalization_binding": normalization_binding,
        "calibration_binding": calibration_binding,
        "sampling_dataset_id": child_sampling_id,
        "sampling_contract": sampling_input,
        "optimizer_contract": group_contract,
        "source_epoch_boundary": {
            "completed_epoch": 1000,
            "next_absolute_epoch": 1001,
            "next_case_cursor": 0,
            "source_optimizer_named_state_entries": len(checkpoint_payload["optimizer_state_by_name"]["state_by_name"]),
            "source_optimizer_updates_per_epoch": int(checkpoint_payload["sampler_state"]["optimizer_updates_this_epoch"]),
            "source_epoch_replay_executed_by_factory": False,
        },
        "split_access": split_access,
        "source_paths": path_records,
        "source_snapshot_files": source_snapshot_files,
        "source_file_sha256": source_file_hashes,
        "data_identity": extra_data_identity,
        "factory_config_sha256": _json_sha256(factory_config),
        "factory_source_sha256": _sha256_file(Path(__file__).resolve()),
        "construction_effects": {
            "normalization_refit": False,
            "calibration_refit": False,
            "optimizer_steps": 0,
            "solver_calls": 0,
            "full_population_read": False,
            "validation_batches_called": False,
        },
    }
    return model, provider, engine, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON containing domain, target_mode, device and explicit source_paths.")
    parser.add_argument("--provenance-out", type=Path,
                        help="Optional JSON output for context provenance; no training or evaluation is run.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_checkpoint = Path(config["source_paths"]["checkpoint"]).expanduser().resolve()
    checkpoint_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    _model, _provider, _engine, provenance = build_saved_context(config, checkpoint_payload)
    output = json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.provenance_out is None:
        sys.stdout.write(output)
    else:
        destination = args.provenance_out.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
