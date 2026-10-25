"""Maintained-engine runner for the paired Thermal contextual-frontier screen.

The Thermal task owns data, native physics, and the J-D/J-H objective. This
module only binds its task factory to the existing research TrainingEngine,
checkpoint layout, storage accounting, and exact resume path.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from . import research_training as research
from .run_layout import RunLayout, resolve_checkpoint
from .unified_training import (
    EngineConfig,
    SamplingKey,
    SelectionPolicy,
)


CONTEXTUAL_FRONTIER_CONFIG_SCHEMA = "thermal_contextual_frontier_v1"
SUPPORTED_ARMS = ("J-D", "J-H")
INTERACTION_GROUPS_CONFIG_SCHEMA = "thermal_interaction_groups_v1"
INTERACTION_GROUPS_ARMS = ("G-EE", "G-F", "G-H", "G-EE-H", "F13-0", "F13-1")
SPATIAL_INTERACTION_GROUP_CORE_SCHEMA = "spatial_interaction_group_fields_v1"
EXECUTION_BUDGET_SCHEMA = "honf_contextual_execution_budget_v1"


def _is_spatial_interaction_group_section(section: Mapping[str, Any]) -> bool:
    model = section.get("model", {})
    return (section.get("core_schema") == SPATIAL_INTERACTION_GROUP_CORE_SCHEMA
            or (isinstance(model, Mapping) and model.get("schema") == SPATIAL_INTERACTION_GROUP_CORE_SCHEMA))


def _model_section(config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    if "interaction_groups" in config:
        return "interaction_groups", config["interaction_groups"]
    return "contextual_frontier", config["contextual_frontier"]


def _arm(config: Mapping[str, Any]) -> str:
    return str(_model_section(config)[1]["arm"])


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read contextual-frontier JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def load_contextual_frontier_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load the ordinary research-run config plus the versioned new model arm."""
    config_path = Path(path).expanduser().resolve(strict=True)
    config = _read_json(config_path)
    if config.get("schema") != research.RESEARCH_SCHEMA:
        raise ValueError(f"Contextual-frontier config must declare schema={research.RESEARCH_SCHEMA}.")
    section_name = "interaction_groups" if "interaction_groups" in config else "contextual_frontier"
    required = {"name", "dataset", "initialization", "training", "optimizer", "execution", "storage",
                section_name, "objective"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Contextual-frontier config is missing {sorted(missing)}.")

    for section in ("dataset", "initialization", "training", "optimizer", "execution", "storage", "objective"):
        if not isinstance(config[section], Mapping):
            raise TypeError(f"{section} must be a JSON object.")
    frontier = config[section_name]
    expected_schema = INTERACTION_GROUPS_CONFIG_SCHEMA if section_name == "interaction_groups" else CONTEXTUAL_FRONTIER_CONFIG_SCHEMA
    if not isinstance(frontier, Mapping) or frontier.get("schema") != expected_schema:
        raise ValueError(f"{section_name}.schema must be {expected_schema}.")
    arm = frontier.get("arm")
    supported_arms = INTERACTION_GROUPS_ARMS if section_name == "interaction_groups" else SUPPORTED_ARMS
    if arm not in supported_arms:
        raise ValueError(f"{section_name} arm must be one of {supported_arms}.")
    spatial_groups = section_name == "interaction_groups" and _is_spatial_interaction_group_section(frontier)
    if spatial_groups:
        model = frontier.get("model")
        if (arm not in ("F13-0", "F13-1") or not isinstance(model, Mapping)
                or model.get("schema") != SPATIAL_INTERACTION_GROUP_CORE_SCHEMA
                or model.get("architecture") != arm):
            raise ValueError("Spatial interaction-group config must bind its core schema and F13 arm.")
        initialization = config["initialization"]
        training = config["training"]
        source_kind = initialization.get("source_kind")
        if source_kind == "balanced_singleton_recovery_epoch100":
            continuation = config.get("continuation")
            continuation_requested = continuation is not None
            valid_initialization = (
                initialization.get("source_arm") == "G-EE"
                and initialization.get("source_parent_age") == 100
                and initialization.get("normalization_mode") == "inherit_balanced_singleton_train150_stats"
            )
            valid_training = (
                training.get("membership") == "fixed25_train150_dev22"
                and training.get("local_horizon") == (500 if continuation_requested else 100)
                and training.get("first_segment_epochs") == (500 if continuation_requested else 100)
            )
            valid_continuation = (
                isinstance(continuation, Mapping)
                and continuation.get("mode") == "full_state"
                and continuation.get("source_epoch") == 100
                and isinstance(continuation.get("source_checkpoint"), str)
                and Path(continuation["source_checkpoint"]).expanduser().is_absolute()
                and training.get("optimizer_schedule_horizon") == 100
            ) if continuation_requested else True
            if not (valid_initialization and valid_training and valid_continuation):
                raise ValueError("F13 development must adapt the literal singleton epoch100 under inherited TRAIN150 normalization.")
        elif source_kind == "same_model_development_epoch100":
            if (training.get("membership") != "full_train600"
                    or training.get("local_horizon") != 1000
                    or training.get("first_segment_epochs") != 200
                    or initialization.get("source_arm") != arm
                    or initialization.get("source_parent_age") != 100
                    or initialization.get("normalization_mode") != "inherit_matching_f13_train150_stats"):
                raise ValueError("Full-TRAIN F13 handoff must start from its matching epoch100 development endpoint.")
        elif source_kind == "same_model_development_epoch500":
            if (training.get("membership") != "full_train600"
                    or training.get("local_horizon") != 1000
                    or training.get("first_segment_epochs") != 500
                    or initialization.get("source_arm") != arm
                    or initialization.get("source_parent_age") != 500
                    or initialization.get("normalization_mode") != "inherit_matching_f13_train150_stats"):
                raise ValueError("Full-TRAIN F13 training must start fresh from its matching development epoch500 weights and TRAIN150 normalization.")
        else:
            raise ValueError("Spatial interaction-group initialization must use a supported versioned Round13 source.")
    elif section_name == "interaction_groups" and arm in ("F13-0", "F13-1"):
        raise ValueError("F13 arms require the spatial interaction-group core schema.")

    dataset = config["dataset"]
    if not isinstance(dataset, Mapping) or dataset.get("name") != "thermal":
        raise ValueError("Contextual-frontier training supports the Thermal dataset only.")
    for key in ("data_path", "manifest_path", "base_recipe_path"):
        if not isinstance(dataset.get(key), str) or not dataset[key]:
            raise ValueError(f"dataset.{key} must be an explicit path.")
    base_recipe = Path(dataset["base_recipe_path"]).expanduser()
    if not base_recipe.is_absolute():
        base_recipe = config_path.parent / base_recipe
    dataset["base_recipe_path"] = str(base_recipe.resolve(strict=True))
    dataset["data_path"] = str(Path(dataset["data_path"]).expanduser().resolve(strict=False))
    dataset["manifest_path"] = str(Path(dataset["manifest_path"]).expanduser().resolve(strict=False))
    if dataset.get("atlas_directory"):
        dataset["atlas_directory"] = str(
            Path(dataset["atlas_directory"]).expanduser().resolve(strict=False)
        )

    training = config["training"]
    for key in ("local_horizon", "first_segment_epochs", "effective_batch_size", "microbatch_cases",
                "primary_queries", "monitor_every", "latest_every", "curve_every",
                "wall_clock_budget_seconds", "clean_save_reserve_seconds"):
        if type(training.get(key)) is not int or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer.")
    if training["clean_save_reserve_seconds"] >= training["wall_clock_budget_seconds"]:
        raise ValueError("The clean-save reserve must be smaller than the per-arm wall-clock budget.")
    if training["first_segment_epochs"] > training["local_horizon"]:
        raise ValueError("The initial segment cannot exceed the configured absolute horizon.")
    if training["microbatch_cases"] > training["effective_batch_size"]:
        raise ValueError("Training microbatch cannot exceed the effective batch size.")
    checkpoints = training.get("checkpoint_epochs")
    if (not isinstance(checkpoints, list)
            or any(type(epoch) is not int or not 1 <= epoch <= training["local_horizon"] for epoch in checkpoints)
            or checkpoints != sorted(set(checkpoints))):
        raise ValueError("training.checkpoint_epochs must be sorted, unique in-horizon epochs.")
    if not isinstance(frontier.get("model"), Mapping):
        raise ValueError(f"{section_name}.model must be a model configuration mapping.")
    if not isinstance(config["objective"].get("native_weights"), Mapping):
        raise ValueError("objective.native_weights must be declared by the Thermal task config.")
    if section_name == "interaction_groups" and spatial_groups:
        objective = config["objective"]
        if (objective.get("response_microbatch_every_primary_updates") != 4
                or float(objective.get("response_shared_gradient_cap_fraction", -1)) != 0.1
                or objective.get("response_coefficient_source") != "literal_initialization_checkpoint"
                or objective.get("response_coefficient_calibration") != "none"
                or objective.get("response_cap_shared_parameter_rule") != (
                    "module/environment/operating encoders, MM/ME stages excluding EE, "
                    "group reader, fusion trunk, and field head"
                )
                or objective.get("native_weights_fixed") is not True
                or objective.get("response_training_families") != ["0348"]
                or objective.get("response_scale_exploration") is not False
                or objective.get("paired_utility_every_primary_updates") != 4
                or float(objective.get("paired_utility_huber_delta", -1)) != 1.0
                or float(objective.get("paired_utility_huber_coefficient", -1)) != 1.0
                or objective.get("paired_utility_physical_reduction") != "average_over_actual_macro_case_count"
                or float(objective.get("scale_exploration_fraction", -1)) != 0.25
                or objective.get("scale_exploration_depth_distribution") != "uniform_available_depth"
                or objective.get("scale_exploration_shared_keys_across_arms") is not True
                or float(objective.get("additional_work_pressure", -1)) != 0.0):
            raise ValueError("Spatial interaction-group objective does not match the paired-utility training contract.")
    elif section_name == "interaction_groups":
        objective = config["objective"]
        if (objective.get("response_microbatch_every_primary_updates") != 4
                or float(objective.get("response_shared_gradient_cap_fraction", -1)) != 0.1
                or objective.get("response_calibration_epoch") != 1
                or objective.get("native_weights_fixed") is not True
                or objective.get("native_weights_frozen_through_epoch") != 100):
            raise ValueError("Interaction-group development requires epoch-1 response calibration, a 10% cap, and fixed native weights through epoch 100 and beyond.")

    storage = config["storage"]
    if not isinstance(storage.get("output_root"), str) or not storage["output_root"].startswith("/data2/"):
        raise ValueError("Contextual-frontier output_root must be an explicit path on /data2.")
    for key in ("max_output_bytes", "max_peak_temporary_bytes", "minimum_free_headroom_bytes"):
        if type(storage.get(key)) is not int or storage[key] < 0:
            raise ValueError(f"storage.{key} must be a nonnegative byte count.")
    if storage["max_output_bytes"] == 0:
        raise ValueError("A contextual-frontier run must declare a positive output budget.")
    if storage["max_peak_temporary_bytes"] == 0:
        raise ValueError("A contextual-frontier run must declare a positive temporary-space budget.")
    preferred = config["execution"].get("preferred_devices")
    if preferred is not None and (
            not isinstance(preferred, list) or any(not isinstance(item, str) for item in preferred)):
        raise ValueError("execution.preferred_devices must be a list of device names when present.")
    clip = config["optimizer"].get("gradient_clip")
    if type(clip) not in (int, float) or not math.isfinite(float(clip)) or float(clip) <= 0:
        raise ValueError("optimizer.gradient_clip must be positive.")
    return config, config_path


def _ensure_workspace_storage_bindings(project: Path | None = None) -> dict[str, str]:
    """Repair the declared DATA2 workspace links before any run output is written."""
    data2 = Path("/data2")
    if not data2.is_dir() or not os.path.ismount(data2):
        raise RuntimeError("/data2 is not mounted; refusing to create contextual-frontier outputs.")
    data2_device = data2.stat().st_dev
    root = project or Path(__file__).resolve().parents[2]
    relocated = data2 / "wanglz/ModularDT/relocated_workspace/HONF_Proj"
    bindings = (
        (root / "Trained_Results", relocated / "Trained_Results", True),
        (root / "diagnostics", relocated / "diagnostics", True),
        (root / "Case_WindFarm/diagnostics/generated/native_role_catalogues_v1",
         relocated / "Case_WindFarm/diagnostics/generated/native_role_catalogues_v1", False),
    )
    checked: dict[str, str] = {}
    for link, target, create_target in bindings:
        target = target.resolve(strict=False)
        if not target.is_relative_to(data2.resolve()):
            raise RuntimeError(f"Declared DATA2 target escaped /data2: {target}.")
        if target.exists() and target.stat().st_dev != data2_device:
            raise RuntimeError(f"Declared target is not on the mounted /data2 filesystem: {target}.")
        if not target.exists():
            if not create_target:
                raise RuntimeError(f"Required shared DATA2 cache is missing: {target}.")
            target.mkdir(parents=True, exist_ok=True)
        resolved_target = target.resolve(strict=True)
        if resolved_target.stat().st_dev != data2_device:
            raise RuntimeError(f"Resolved DATA2 target is on the wrong filesystem: {resolved_target}.")

        valid_link = False
        if link.is_symlink():
            try:
                valid_link = link.resolve(strict=True) == resolved_target
            except FileNotFoundError:
                valid_link = False
        elif link.exists():
            raise RuntimeError(f"Expected a workspace symlink at {link}, found a non-symlink entry.")
        if not valid_link:
            if link.is_symlink():
                link.unlink()
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(resolved_target, target_is_directory=True)
        if link.resolve(strict=True) != resolved_target or link.resolve(strict=True).stat().st_dev != data2_device:
            raise RuntimeError(f"Workspace DATA2 binding verification failed for {link}.")
        checked[str(link)] = str(resolved_target)
    return checked


def _task_factory() -> Callable[..., tuple[torch.nn.Module, Any]]:
    from channelthermal.training.contextual_joint_task import build_thermal_contextual_frontier_task

    return build_thermal_contextual_frontier_task


def _interaction_group_task_factory(config: Mapping[str, Any]) -> Callable[..., tuple[torch.nn.Module, Any]]:
    section = _model_section(config)[1]
    if _is_spatial_interaction_group_section(section):
        from channelthermal.training.spatial_interaction_group_task import (
            build_thermal_spatial_interaction_group_task,
        )

        return build_thermal_spatial_interaction_group_task
    from channelthermal.training.interaction_group_task import build_thermal_interaction_group_task

    return build_thermal_interaction_group_task


def _build_task(config: Mapping[str, Any], arm: str, device: torch.device,
                task_factory: Callable[..., tuple[torch.nn.Module, Any]] | None = None):
    factory = task_factory or (_interaction_group_task_factory(config)
                               if "interaction_groups" in config else _task_factory())
    model, provider = factory(config, arm=arm, device=device)
    if model is None or provider is None:
        raise TypeError("Contextual-frontier task factory must return a model and provider.")
    if getattr(model, "architecture", None) != arm:
        raise ValueError(f"Contextual-frontier factory returned a model for {getattr(model, 'architecture', None)!r}, expected {arm}.")
    if getattr(provider, "model", None) is not model:
        raise ValueError("Contextual-frontier provider must be bound to the returned model.")
    return model, provider


def _engine_config(config: Mapping[str, Any]) -> EngineConfig:
    training = config["training"]
    return EngineConfig(
        seed=int(training["seed"]),
        microbatch_cases=int(training["microbatch_cases"]),
        effective_cases=int(training["effective_batch_size"]),
        total_epochs=int(training["local_horizon"]),
        monitor_every=int(training["monitor_every"]),
        gradient_clip=float(config["optimizer"]["gradient_clip"]),
        deterministic_case_order=True,
        checkpoint_epochs=tuple(map(int, training["checkpoint_epochs"])),
        latest_every=int(training["latest_every"]),
        curve_every=int(training["curve_every"]),
        progress_enabled=bool(training["progress_enabled"]),
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        training_mode="joint",
    )


def _run_identity(config: Mapping[str, Any], provider: Any) -> dict[str, Any]:
    section_name, section = _model_section(config)
    frontier = copy.deepcopy(dict(section))
    interaction_groups = section_name == "interaction_groups"
    spatial_groups = interaction_groups and _is_spatial_interaction_group_section(section)
    return {
        "schema": ("honf_spatial_interaction_groups_training_v1" if spatial_groups else
                   "honf_interaction_groups_training_v1" if interaction_groups else
                   "honf_contextual_frontier_training_v1"),
        "workflow": ("spatial_interaction_groups_joint_training_v1" if spatial_groups else
                     "interaction_groups_joint_training_v1" if interaction_groups else
                     "contextual_frontier_joint_training_v1"),
        "model_family": ("spatial_interaction_group_fields_joint_field_v1" if spatial_groups else
                         "interaction_groups_joint_field_v1" if interaction_groups else
                         "contextual_organization_joint_field_v1"),
        "dataset": "thermal",
        "config_name": config["name"],
        "training_membership": config["training"]["membership"],
        "normalization_mode": config["initialization"]["normalization_mode"],
        "initialization": copy.deepcopy(dict(config["initialization"])),
        "optimizer": copy.deepcopy(dict(config["optimizer"])),
        "recipe": {
            section_name: frontier,
            "native_weights": copy.deepcopy(dict(config["objective"]["native_weights"])),
            "absolute_horizon": int(config["training"]["local_horizon"]),
            "primary_queries": int(config["training"]["primary_queries"]),
            "effective_batch_size": int(config["training"]["effective_batch_size"]),
            "microbatch_cases": int(config["training"]["microbatch_cases"]),
            "seed": int(config["training"]["seed"]),
            "wall_clock_budget_seconds": int(config["training"]["wall_clock_budget_seconds"]),
            "clean_save_reserve_seconds": int(config["training"]["clean_save_reserve_seconds"]),
        },
    }


def _model_state_bytes(model: torch.nn.Module) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size())
               for tensor in model.state_dict().values() if torch.is_tensor(tensor))


def _storage_estimate(model: torch.nn.Module, *, completed_epoch: int, stop_after: int,
                      config: Mapping[str, Any], fresh_namespace: bool = False) -> tuple[int, int]:
    """Conservatively include model, named Adam moments, aliases, and atomic-save space."""
    state_bytes = max(1, _model_state_bytes(model))
    payload_bytes = 4 * state_bytes + 8 * 1024**2
    milestones = sum(
        1 for epoch in config["training"]["checkpoint_epochs"]
        if completed_epoch < int(epoch) <= stop_after
    )
    terminal = int(stop_after not in config["training"]["checkpoint_epochs"])
    if completed_epoch == 0 or fresh_namespace:
        alias_count = 2  # latest and best-field are distinct retained files
    else:
        alias_count = 0  # both aliases already exist and are atomically replaced
    new_payload_count = milestones + terminal + alias_count
    history_and_plots = 64 * 1024**2
    estimated_new = new_payload_count * payload_bytes + history_and_plots
    peak_temporary = payload_bytes + 32 * 1024**2
    return estimated_new, peak_temporary


def _prepare_storage(config: Mapping[str, Any], output: Path, *, estimated_bytes: int,
                     peak_temporary_bytes: int, bindings: Mapping[str, str] | None = None) -> dict[str, Any]:
    snapshot = research.verify_storage(
        config, output=output, estimated_bytes=estimated_bytes,
        peak_temporary_bytes=peak_temporary_bytes,
    )
    if snapshot["free_bytes_before"] - estimated_bytes - peak_temporary_bytes < snapshot[
            "minimum_free_headroom_bytes"]:
        raise RuntimeError("Contextual-frontier output plus atomic-save space would breach DATA2 headroom.")
    snapshot["workspace_bindings"] = dict(bindings or {})
    return snapshot


def _write_run_manifest(output: Path, config: Mapping[str, Any], config_path: Path,
                        arm: str, storage: Mapping[str, Any], *, resumed: bool,
                        execution_budget: Mapping[str, Any] | None = None) -> None:
    layout = RunLayout(output)
    layout.ensure()
    section_name, section = _model_section(config)
    spatial_groups = section_name == "interaction_groups" and _is_spatial_interaction_group_section(section)
    manifest_schema = ("honf_spatial_interaction_groups_run_manifest_v1" if spatial_groups else
                       "honf_interaction_groups_run_manifest_v1" if section_name == "interaction_groups" else
                       "honf_contextual_frontier_run_manifest_v1")
    if not resumed:
        research._atomic_json(layout.write_path("resolved_config.json"), dict(config))
        research._atomic_json(layout.write_path("run_manifest.json"), {
            "schema": manifest_schema,
            "artifact_layout_version": 1,
            "config": dict(config),
            "config_path": str(config_path),
            "arm": arm,
            "created_unix": time.time(),
            "storage": dict(storage),
        })
    else:
        manifest_path = layout.read_path("run_manifest.json")
        manifest = _read_json(manifest_path)
        expected_schema = manifest_schema
        if (manifest.get("schema") != expected_schema
                or manifest.get("artifact_layout_version") != 1
                or manifest.get("arm") != arm
                or manifest.get("config") != dict(config)):
            raise ValueError("Contextual-frontier run manifest differs from its saved arm/config.")
        manifest["last_resume_storage"] = dict(storage)
        manifest["last_resume_unix"] = time.time()
        if execution_budget is not None:
            manifest["execution_budget"] = copy.deepcopy(dict(execution_budget))
        research._atomic_json(manifest_path, manifest)


def _provider_elapsed_seconds(payload: Mapping[str, Any]) -> float:
    provider_state = payload.get("provider_training_state")
    if not isinstance(provider_state, Mapping):
        raise ValueError("Execution-budget continuation requires saved provider training state.")
    elapsed = provider_state.get("training_elapsed_seconds")
    if (type(elapsed) not in (int, float) or not math.isfinite(float(elapsed))
            or float(elapsed) < 0):
        raise ValueError("Saved provider training elapsed seconds must be finite and nonnegative.")
    return float(elapsed)


def _supports_cumulative_budget_override(config: Mapping[str, Any]) -> bool:
    try:
        section_name, section = _model_section(config)
        initialization = config["initialization"]
        training = config["training"]
        arm = section["arm"]
        return (
            section_name == "interaction_groups"
            and _is_spatial_interaction_group_section(section)
            and arm in ("F13-0", "F13-1")
            and initialization.get("source_kind") == "same_model_development_epoch500"
            and initialization.get("source_arm") == arm
            and initialization.get("source_parent_age") == 500
            and initialization.get("normalization_mode") == "inherit_matching_f13_train150_stats"
            and training.get("membership") == "full_train600"
            and training.get("local_horizon") == 1000
        )
    except (KeyError, TypeError):
        return False


def _resume_execution_budget(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    latest_path: Path,
    latest_payload: Mapping[str, Any],
    requested_seconds: int | None,
) -> tuple[int, dict[str, Any] | None]:
    """Resolve an execution-only cumulative provider-time allocation for one exact resume."""
    stored = manifest.get("execution_budget")
    if requested_seconds is None and stored is None:
        return int(config["training"]["wall_clock_budget_seconds"]), None
    if not _supports_cumulative_budget_override(config):
        raise ValueError(
            "Cumulative wall-clock overrides are limited to F13 full-TRAIN600 runs sourced from matching development epoch500."
        )
    arm = _arm(config)
    epoch = int(latest_payload.get("epoch", -1))
    identity = latest_payload.get("experiment_identity")
    if (epoch < 500 or latest_payload.get("arm") != arm
            or not isinstance(identity, Mapping)
            or identity.get("workflow") != "spatial_interaction_groups_joint_training_v1"
            or identity.get("training_membership") != "full_train600"):
        raise ValueError("Cumulative wall-clock overrides require the matching F13 full-TRAIN checkpoint at epoch500 or later.")
    training = config["training"]
    recipe_seconds = int(training["wall_clock_budget_seconds"])
    reserve_seconds = int(training["clean_save_reserve_seconds"])
    if stored is not None:
        if (not isinstance(stored, Mapping)
                or stored.get("schema") != EXECUTION_BUDGET_SCHEMA
                or stored.get("scope") != "absolute_cumulative_provider_elapsed_seconds"
                or stored.get("recipe_budget_seconds") != recipe_seconds
                or stored.get("clean_save_reserve_seconds") != reserve_seconds
                or type(stored.get("absolute_cumulative_seconds")) is not int
                or stored["absolute_cumulative_seconds"] <= reserve_seconds):
            raise ValueError("Saved contextual execution-budget metadata is malformed or differs from its run recipe.")
    if requested_seconds is not None:
        if type(requested_seconds) is not int or requested_seconds <= reserve_seconds:
            raise ValueError("Cumulative provider-time budget must be a positive integer greater than the configured clean-save reserve.")
        selected_seconds = requested_seconds
        declaration = "explicit_cli_value"
    elif stored is not None:
        selected_seconds = int(stored["absolute_cumulative_seconds"])
        declaration = "reused_manifest_value"
    else:
        raise RuntimeError("Execution budget resolution reached an invalid state.")

    source = {
        "checkpoint": str(latest_path.resolve(strict=True)),
        "epoch": epoch,
        "source_saved_provider_elapsed_seconds": _provider_elapsed_seconds(latest_payload),
    }
    result = copy.deepcopy(dict(stored)) if isinstance(stored, Mapping) else {
        "schema": EXECUTION_BUDGET_SCHEMA,
        "scope": "absolute_cumulative_provider_elapsed_seconds",
        "recipe_budget_seconds": recipe_seconds,
        "clean_save_reserve_seconds": reserve_seconds,
        "initial_source": source,
    }
    result.update({
        "absolute_cumulative_seconds": selected_seconds,
        "last_resume_source": source,
        "last_declaration": declaration,
        "declared_unix": time.time(),
        "declaration_status": "declared_before_fit",
    })
    return selected_seconds, result


def _link_parent_field_selector(output: Path, source_checkpoint: Path,
                                source_payload: Mapping[str, Any]) -> Path:
    """Preserve the parent best-field checkpoint by reference in a fresh continuation namespace."""
    source_run = source_checkpoint.resolve(strict=True).parents[1]
    source_best = resolve_checkpoint(source_run, "best_field").resolve(strict=True)
    from .compat import load_trusted_checkpoint

    best_payload = load_trusted_checkpoint(source_best, map_location="cpu")
    source_identity = source_payload.get("experiment_identity")
    if (best_payload.get("arm") != source_payload.get("arm")
            or int(best_payload.get("epoch", -1)) > int(source_payload.get("epoch", -1))
            or best_payload.get("experiment_identity") != source_identity):
        raise ValueError("Parent best-field selector does not belong to the exact full-state source lineage.")
    layout = RunLayout(output)
    destination = layout.write_path("best_by_field_mse_model.pt")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Fresh continuation selector path already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source_best, destination.parent))
    if destination.resolve(strict=True) != source_best:
        destination.unlink()
        raise RuntimeError("Parent best-field selector reference failed verification.")
    return destination


def _fit(model: torch.nn.Module, provider: Any, config: Mapping[str, Any], output: Path, *,
         device: torch.device, stop_after: int, resume_checkpoint: Path | None = None,
         external_state_continuation: bool = False,
         wall_clock_budget_seconds: int | None = None) -> dict[str, Any]:
    from .research_training import ResearchTrainingEngine

    arm = _arm(config)
    engine_config = _engine_config(config)
    engine = ResearchTrainingEngine(
        engine_config, device=device, parent_age=0,
        selection=SelectionPolicy(field_metric="field_score"),
    )
    engine.bind_execution_runtime(
        output / "execution_runtime.json",
        nominal_microbatch_cases=int(config["training"]["microbatch_cases"]),
    )
    budget_binder = getattr(provider, "bind_wall_clock_budget", None)
    if wall_clock_budget_seconds is not None and not callable(budget_binder):
        raise ValueError("The selected provider cannot enforce an explicit cumulative wall-clock budget.")
    if callable(budget_binder):
        budget_binder(
            output_dir=output,
            budget_seconds=int(
                config["training"]["wall_clock_budget_seconds"]
                if wall_clock_budget_seconds is None else wall_clock_budget_seconds
            ),
            saver_reserve_seconds=int(config["training"]["clean_save_reserve_seconds"]),
        )
    previous_handlers = research._install_clean_stop_handlers(output)
    try:
        continuation_options = ({
            "external_state_continuation": True,
            "allow_horizon_extension": True,
        } if external_state_continuation else {})
        return engine.fit(
            model, provider, output,
            identity=_run_identity(config, provider),
            arm=arm,
            stop_after=int(stop_after),
            resume_checkpoint=resume_checkpoint,
            **continuation_options,
        )
    finally:
        research._restore_handlers(previous_handlers)


def train_contextual_frontier(config_path: str | Path, *, device: str, output: str | Path,
                              epochs: int | None = None, devices: Sequence[str] | None = None,
                              task_factory: Callable[..., tuple[torch.nn.Module, Any]] | None = None,
                              verify_bindings: bool = True) -> dict[str, Any]:
    config, resolved_config_path = load_contextual_frontier_config(config_path)
    training = config["training"]
    stop_after = int(training["first_segment_epochs"] if epochs is None else epochs)
    if not 1 <= stop_after <= int(training["local_horizon"]):
        raise ValueError("Requested training stop must lie within the configured absolute horizon.")
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"New contextual-frontier runs require an empty output directory: {output_path}")
    root = Path(config["storage"]["output_root"]).expanduser().resolve(strict=False)
    if not output_path.is_relative_to(root):
        raise ValueError(f"Run output must be inside configured DATA2 output root {root}.")
    binding_info = _ensure_workspace_storage_bindings() if verify_bindings else {}
    research.verify_storage(config, output=output_path, estimated_bytes=0, peak_temporary_bytes=0)
    configured_devices = config["execution"].get("preferred_devices")
    permitted = research._permitted_order(device, devices or configured_devices)
    selected_device, device_info = research.resolve_device(permitted[0], permitted_devices=permitted)
    arm = _arm(config)
    model, provider = _build_task(config, arm, selected_device, task_factory=task_factory)
    estimated_bytes, peak_temporary = _storage_estimate(
        model, completed_epoch=0, stop_after=stop_after, config=config,
    )
    storage = _prepare_storage(config, output_path, estimated_bytes=estimated_bytes,
                                peak_temporary_bytes=peak_temporary, bindings=binding_info)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(output_path, config, resolved_config_path, arm, storage, resumed=False)
    print(json.dumps({
        "event": "contextual_frontier_run_prepared",
        "output": str(output_path),
        "arm": arm,
        "selected_device": device_info,
        "training_membership": training["membership"],
        "case_count": len(getattr(provider, "training_case_ids", ())),
        "validation_case_count": len(getattr(provider, "validation_case_ids", ())),
        "effective_batch": training["effective_batch_size"],
        "microbatch": training["microbatch_cases"],
        "primary_queries": training["primary_queries"],
        "initial_stop_after": stop_after,
        "storage": storage,
    }, sort_keys=True, default=str, allow_nan=False), flush=True)
    return _fit(model, provider, config, output_path, device=selected_device, stop_after=stop_after)


def continue_contextual_frontier(config_path: str | Path, *, device: str, output: str | Path,
                                 epochs: int | None = None, devices: Sequence[str] | None = None,
                                 task_factory: Callable[..., tuple[torch.nn.Module, Any]] | None = None,
                                 verify_bindings: bool = True) -> dict[str, Any]:
    """Continue one F13 development checkpoint in a fresh run namespace, preserving full state."""
    config, resolved_config_path = load_contextual_frontier_config(config_path)
    continuation = config.get("continuation")
    if (not isinstance(continuation, Mapping) or continuation.get("mode") != "full_state"
            or continuation.get("source_epoch") != 100):
        raise ValueError("Contextual full-state continuation requires its declared F13 epoch100 source.")
    source_path = Path(continuation["source_checkpoint"]).expanduser().resolve(strict=True)
    from .compat import load_trusted_checkpoint

    source_payload = load_trusted_checkpoint(source_path, map_location="cpu")
    arm = _arm(config)
    if (int(source_payload.get("epoch", -1)) != int(continuation["source_epoch"])
            or source_payload.get("arm") != arm):
        raise ValueError("Full-state continuation source age or arm differs from its ordinary recipe.")
    identity = source_payload.get("experiment_identity")
    provider_identity = identity.get("provider_identity") if isinstance(identity, Mapping) else None
    if (not isinstance(identity, Mapping)
            or identity.get("workflow") != "spatial_interaction_groups_joint_training_v1"
            or identity.get("training_membership") != "fixed25_train150_dev22"
            or not isinstance(provider_identity, Mapping)
            or provider_identity.get("workflow") != "spatial_interaction_groups_joint_training_v1"):
        raise ValueError("Full-state continuation source is not the matching F13 development lineage.")
    training = config["training"]
    stop_after = int(training["first_segment_epochs"] if epochs is None else epochs)
    if not int(source_payload["epoch"]) < stop_after <= int(training["local_horizon"]):
        raise ValueError("Continuation stop must advance the source epoch and remain inside the declared horizon.")
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"Fresh full-state continuations require an empty output namespace: {output_path}")
    root = Path(config["storage"]["output_root"]).expanduser().resolve(strict=False)
    if not output_path.is_relative_to(root):
        raise ValueError(f"Continuation output must be inside configured DATA2 output root {root}.")
    binding_info = _ensure_workspace_storage_bindings() if verify_bindings else {}
    research.verify_storage(config, output=output_path, estimated_bytes=0, peak_temporary_bytes=0)
    configured_devices = config["execution"].get("preferred_devices")
    permitted = research._permitted_order(device, devices or configured_devices)
    selected_device, device_info = research.resolve_device(permitted[0], permitted_devices=permitted)
    model, provider = _build_task(config, arm, selected_device, task_factory=task_factory)
    estimated_bytes, peak_temporary = _storage_estimate(
        model, completed_epoch=int(source_payload["epoch"]), stop_after=stop_after,
        config=config, fresh_namespace=True,
    )
    storage = _prepare_storage(config, output_path, estimated_bytes=estimated_bytes,
                                peak_temporary_bytes=peak_temporary, bindings=binding_info)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(output_path, config, resolved_config_path, arm, storage, resumed=False)
    _link_parent_field_selector(output_path, source_path, source_payload)
    from .unified_training import _render_loss_curves

    _render_loss_curves(
        source_payload.get("history", ()),
        RunLayout(output_path).write_path("loss_curves.png").parent,
        "field_score",
        loss_metadata=(provider.loss_metadata() if callable(getattr(provider, "loss_metadata", None)) else None),
    )
    print(json.dumps({
        "event": "contextual_frontier_full_state_continuation_prepared",
        "source_checkpoint": str(source_path),
        "source_epoch": int(source_payload["epoch"]),
        "output": str(output_path),
        "arm": arm,
        "selected_device": device_info,
        "training_membership": training["membership"],
        "case_count": len(getattr(provider, "training_case_ids", ())),
        "validation_case_count": len(getattr(provider, "validation_case_ids", ())),
        "source_history_rows": len(source_payload.get("history", ())),
        "stop_after_epoch": stop_after,
        "storage": storage,
    }, sort_keys=True, default=str, allow_nan=False), flush=True)
    return _fit(model, provider, config, output_path, device=selected_device,
                stop_after=stop_after, resume_checkpoint=source_path,
                external_state_continuation=True)


def resume_contextual_frontier(run_dir: str | Path, *, device: str, stop_after_epoch: int,
                               devices: Sequence[str] | None = None,
                               task_factory: Callable[..., tuple[torch.nn.Module, Any]] | None = None,
                               wall_clock_budget_seconds: int | None = None,
                               verify_bindings: bool = True) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve(strict=True)
    layout = RunLayout(run)
    manifest_path = layout.read_path("run_manifest.json")
    manifest = _read_json(manifest_path)
    if (manifest.get("schema") not in {"honf_contextual_frontier_run_manifest_v1",
                                        "honf_interaction_groups_run_manifest_v1",
                                        "honf_spatial_interaction_groups_run_manifest_v1"}
            or manifest.get("artifact_layout_version") != 1):
        raise ValueError("Run is missing the maintained contextual-frontier manifest.")
    config = copy.deepcopy(manifest.get("config"))
    if not isinstance(config, dict):
        raise ValueError("Contextual-frontier run manifest has no saved configuration.")
    try:
        arm = _arm(config)
    except (KeyError, TypeError):
        arm = ""
    if manifest.get("arm") != arm:
        raise ValueError("Saved run arm differs from its versioned config.")
    # Revalidate the stored config with a temporary on-disk-free structural pass.
    _validate_loaded_contextual_config(config)
    latest_path = resolve_checkpoint(run, "latest")
    from .compat import load_trusted_checkpoint

    latest_payload = load_trusted_checkpoint(latest_path, map_location="cpu")
    current_epoch = int(latest_payload.get("epoch", -1))
    if current_epoch < 1 or stop_after_epoch <= current_epoch:
        raise ValueError("Resume stop must advance the latest completed epoch.")
    if stop_after_epoch > int(config["training"]["local_horizon"]):
        raise ValueError(f"Resume stop exceeds the configured absolute {config['training']['local_horizon']}-epoch horizon.")
    selected_budget, execution_budget = _resume_execution_budget(
        config, manifest, latest_path, latest_payload, wall_clock_budget_seconds,
    )

    binding_info = _ensure_workspace_storage_bindings() if verify_bindings else {}
    research.verify_storage(config, output=run, estimated_bytes=0, peak_temporary_bytes=0)
    configured_devices = config["execution"].get("preferred_devices")
    permitted = research._permitted_order(device, devices or configured_devices)
    selected_device, device_info = research.resolve_device(permitted[0], permitted_devices=permitted)
    model, provider = _build_task(config, arm, selected_device, task_factory=task_factory)
    estimated_bytes, peak_temporary = _storage_estimate(
        model, completed_epoch=current_epoch, stop_after=stop_after_epoch, config=config,
    )
    storage = _prepare_storage(config, run, estimated_bytes=estimated_bytes,
                                peak_temporary_bytes=peak_temporary, bindings=binding_info)
    _write_run_manifest(run, config, Path(str(manifest.get("config_path", ""))),
                        arm, storage, resumed=True, execution_budget=execution_budget)
    print(json.dumps({
        "event": "contextual_frontier_resume_prepared",
        "run": str(run),
        "arm": arm,
        "selected_device": device_info,
        "completed_epoch": current_epoch,
        "stop_after_epoch": int(stop_after_epoch),
        "wall_clock_budget_seconds": selected_budget,
        "wall_clock_budget_scope": "absolute cumulative provider elapsed seconds",
        "storage": storage,
    }, sort_keys=True, default=str, allow_nan=False), flush=True)
    return _fit(model, provider, config, run, device=selected_device,
                stop_after=stop_after_epoch, resume_checkpoint=latest_path,
                wall_clock_budget_seconds=selected_budget)


def _validate_loaded_contextual_config(config: Mapping[str, Any]) -> None:
    """Validate a manifest-held config without resolving its external input files."""
    section_name = "interaction_groups" if "interaction_groups" in config else "contextual_frontier"
    frontier = config.get(section_name, {})
    expected_schema = INTERACTION_GROUPS_CONFIG_SCHEMA if section_name == "interaction_groups" else CONTEXTUAL_FRONTIER_CONFIG_SCHEMA
    supported_arms = INTERACTION_GROUPS_ARMS if section_name == "interaction_groups" else SUPPORTED_ARMS
    spatial_groups = section_name == "interaction_groups" and isinstance(frontier, Mapping) and _is_spatial_interaction_group_section(frontier)
    if (config.get("schema") != research.RESEARCH_SCHEMA
            or not isinstance(frontier, Mapping)
            or frontier.get("schema") != expected_schema
            or frontier.get("arm") not in supported_arms):
        raise ValueError("Saved contextual model config is malformed or uses an unsupported arm.")
    if spatial_groups:
        model = frontier.get("model")
        arm = frontier.get("arm")
        if (frontier.get("arm") not in ("F13-0", "F13-1") or not isinstance(model, Mapping)
                or model.get("schema") != SPATIAL_INTERACTION_GROUP_CORE_SCHEMA
                or model.get("architecture") != arm):
            raise ValueError("Saved spatial interaction-group config has a malformed core binding.")
        initialization = config.get("initialization", {})
        training = config.get("training", {})
        source_kind = initialization.get("source_kind") if isinstance(initialization, Mapping) else None
        if source_kind == "balanced_singleton_recovery_epoch100":
            continuation = config.get("continuation")
            matured = continuation is not None
            expected_binding = (
                "fixed25_train150_dev22", 500 if matured else 100, 500 if matured else 100, "G-EE",
                "inherit_balanced_singleton_train150_stats",
            )
            if matured and (
                    not isinstance(continuation, Mapping)
                    or continuation.get("mode") != "full_state"
                    or continuation.get("source_epoch") != 100
                    or not isinstance(continuation.get("source_checkpoint"), str)
                    or not Path(continuation["source_checkpoint"]).is_absolute()
                    or training.get("optimizer_schedule_horizon") != 100):
                raise ValueError("Saved spatial development continuation does not bind its literal epoch100 full-state source.")
        elif source_kind == "same_model_development_epoch100":
            expected_binding = (
                "full_train600", 1000, 200, arm,
                "inherit_matching_f13_train150_stats",
            )
        elif source_kind == "same_model_development_epoch500":
            expected_binding = (
                "full_train600", 1000, 500, arm,
                "inherit_matching_f13_train150_stats",
            )
        else:
            raise ValueError("Saved spatial config declares an unsupported initialization source.")
        actual_binding = (
            training.get("membership") if isinstance(training, Mapping) else None,
            training.get("local_horizon") if isinstance(training, Mapping) else None,
            training.get("first_segment_epochs") if isinstance(training, Mapping) else None,
            initialization.get("source_arm") if isinstance(initialization, Mapping) else None,
            initialization.get("normalization_mode") if isinstance(initialization, Mapping) else None,
        )
        expected_parent_age = 500 if source_kind == "same_model_development_epoch500" else 100
        if (actual_binding != expected_binding
                or initialization.get("source_parent_age") != expected_parent_age):
            raise ValueError("Saved spatial config changed its versioned source, horizon, or normalization binding.")
        objective = config.get("objective", {})
        if (not isinstance(objective, Mapping)
                or objective.get("response_microbatch_every_primary_updates") != 4
                or float(objective.get("response_shared_gradient_cap_fraction", -1)) != 0.1
                or objective.get("response_coefficient_source") != "literal_initialization_checkpoint"
                or objective.get("response_coefficient_calibration") != "none"
                or objective.get("response_cap_shared_parameter_rule") != (
                    "module/environment/operating encoders, MM/ME stages excluding EE, "
                    "group reader, fusion trunk, and field head"
                )
                or objective.get("response_training_families") != ["0348"]
                or objective.get("response_scale_exploration") is not False
                or objective.get("native_weights_fixed") is not True
                or objective.get("paired_utility_every_primary_updates") != 4
                or objective.get("paired_utility_physical_reduction") != "average_over_actual_macro_case_count"
                or float(objective.get("paired_utility_huber_delta", -1)) != 1.0
                or float(objective.get("paired_utility_huber_coefficient", -1)) != 1.0
                or float(objective.get("scale_exploration_fraction", -1)) != 0.25
                or objective.get("scale_exploration_depth_distribution") != "uniform_available_depth"
                or objective.get("scale_exploration_shared_keys_across_arms") is not True
                or float(objective.get("additional_work_pressure", -1)) != 0.0):
            raise ValueError("Saved spatial interaction-group objective contract is malformed.")
    if section_name == "interaction_groups" and not spatial_groups:
        objective = config.get("objective", {})
        if (not isinstance(objective, Mapping)
                or objective.get("response_microbatch_every_primary_updates") != 4
                or objective.get("response_shared_gradient_cap_fraction") != 0.1
                or objective.get("response_calibration_epoch") != 1
                or objective.get("native_weights_fixed") is not True
                or objective.get("native_weights_frozen_through_epoch") != 100):
            raise ValueError("Saved interaction-group objective contract is malformed.")
    if not isinstance(config.get("dataset"), Mapping) or config["dataset"].get("name") != "thermal":
        raise ValueError("Saved contextual-frontier run does not use the Thermal task.")
    training = config.get("training", {})
    if type(training.get("local_horizon")) is not int or training["local_horizon"] < 1:
        raise ValueError("Saved contextual-frontier run has an invalid absolute horizon.")
    storage = config.get("storage", {})
    if (type(storage.get("max_output_bytes")) is not int or storage["max_output_bytes"] <= 0
            or type(storage.get("max_peak_temporary_bytes")) is not int
            or storage["max_peak_temporary_bytes"] < 0
            or type(storage.get("minimum_free_headroom_bytes")) is not int
            or storage["minimum_free_headroom_bytes"] < 0):
        raise ValueError("Saved contextual-frontier storage budget is invalid.")


__all__ = [
    "CONTEXTUAL_FRONTIER_CONFIG_SCHEMA",
    "continue_contextual_frontier",
    "load_contextual_frontier_config",
    "resume_contextual_frontier",
    "train_contextual_frontier",
]
