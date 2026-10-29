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


R13_8_SUBSET_CHECKPOINT_EPOCHS = [100, 200, 300, 400, 500]
R13_8_E_ANCHORED_CONFIG_SCHEMA = "thermal_r13_8_e_anchored_member_routing_v1"
R13_8_E_ANCHORED_CORE_SCHEMA = "interaction_reused_e_anchored_member_routing_v1"
R13_8_A_ARM = "R13-8-A-EAnchor"
R13_8_L_ARM = "R13-8-L-LocalECover"
R13_8_E_ANCHORED_ARMS = (R13_8_A_ARM, R13_8_L_ARM)
R13_8_E_ANCHORED_WORKFLOW = "interaction_reused_e_anchored_member_routing_training_v1"
R13_8_E_ANCHORED_PROVIDER_SCHEMA = "thermal_r13_8_e_anchored_member_routing_provider_v1"
R13_8_E_ANCHORED_MANIFEST_SCHEMA = "honf_r13_8_e_anchored_member_routing_run_manifest_v1"
R13_8_E_ANCHORED_SOURCE_KIND = "scratch_r13_8_e_anchored_member_routing"


def _member_routing_seed(training, model):
    return 0


def _anchored_schedule_and_model(training, initialization, model, arm):
    actual = (training.get("local_horizon"), training.get("first_segment_epochs"),
              training.get("optimizer_schedule_horizon"), training.get("checkpoint_epochs"),
              initialization.get("normalization_mode"))
    membership = training.get("membership")
    if membership == "fixed25_train150_dev22":
        schedules = ((100, 100, 100, [100], "fixed25_train150_stats"),
                     (500, 500, 500, R13_8_SUBSET_CHECKPOINT_EPOCHS, "fixed25_train150_stats"),
                     (1000, 100, 1000, list(range(100, 1001, 100)), "fixed25_train150_stats"),
                     (1000, 200, 1000, list(range(100, 1001, 100)), "fixed25_train150_stats"))
        if arm == R13_8_L_ARM:
            schedules += ((20, 20, 20, [20], "fixed25_train150_stats"),)
    elif membership == "full_train600" and initialization.get("source_kind") == R13_8_E_ANCHORED_SOURCE_KIND:
        schedules = ((100, 100, 500, [100], "scratch_train600_stats"),
                     (500, 100, 500, R13_8_SUBSET_CHECKPOINT_EPOCHS, "scratch_train600_stats"),
                     (1000, 100, 1000, list(range(100, 1001, 100)), "scratch_train600_stats"),
                     (1000, 200, 1000, list(range(100, 1001, 100)), "scratch_train600_stats"),
                     (5000, 200, 5000, list(range(100, 5001, 100)), "scratch_train600_stats"))
    else:
        schedules = ()
    locators = {R13_8_A_ARM: "e_anchor", R13_8_L_ARM: "local_e_cover"}
    locator = locators.get(arm)
    valid_model = (model.get("architecture") == arm and locator is not None
        and model.get("locator_mode") == locator and model.get("retained_mass") == 0.95
        and (model.get("score_calibration") == "inherited"
             or (training.get("local_horizon") == 1000 and model.get("score_calibration") == "bounded_tanh"))
        and model.get("access_measure", "inventory_unit_total") in ("inventory_unit_total", "mean_one", "active_domain")
        and model.get("ee_incidence_aggregation", "pre_gate") in ("pre_gate", "post_gate"))
    valid_model = valid_model and model.get("mediated_mode") in (None, "none")
    return actual in schedules and training.get("curve_every") == 10, valid_model
def _model_section(config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    return "member_routing", config["member_routing"]


def _arm(config: Mapping[str, Any]) -> str:
    return str(config["member_routing"]["arm"])


def _member_routing_core_schema(section: Mapping[str, Any]) -> str | None:
    model = section.get("model")
    return str(model.get("schema")) if isinstance(model, Mapping) else None


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
    section_name = ("member_routing" if "member_routing" in config else
                    "interaction_groups" if "interaction_groups" in config else "contextual_frontier")
    required = {"name", "dataset", "initialization", "training", "optimizer", "execution", "storage",
                section_name, "objective"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Contextual-frontier config is missing {sorted(missing)}.")

    for section in ("dataset", "initialization", "training", "optimizer", "execution", "storage", "objective"):
        if not isinstance(config[section], Mapping):
            raise TypeError(f"{section} must be a JSON object.")
    frontier = config[section_name]
    if (section_name != "member_routing" or not isinstance(frontier, Mapping)
            or frontier.get("schema") != R13_8_E_ANCHORED_CONFIG_SCHEMA
            or _member_routing_core_schema(frontier) != R13_8_E_ANCHORED_CORE_SCHEMA
            or frontier.get("arm") not in R13_8_E_ANCHORED_ARMS):
        raise ValueError("Only anchored member-routing configurations are supported.")
    arm = frontier.get("arm")
    if section_name == "member_routing":
        model = frontier.get("model")
        base = model.get("base_model_config") if isinstance(model, Mapping) else None
        training = config["training"]
        initialization = config["initialization"]
        membership = training.get("membership")
        core_schema = _member_routing_core_schema(frontier)
        config_schema = frontier.get("schema")
        valid_schedule, valid_model = _anchored_schedule_and_model(training, initialization, model, arm)
        optimizer = config["optimizer"]
        objective = config["objective"]
        valid_source = (
            initialization.get("source_kind") == R13_8_E_ANCHORED_SOURCE_KIND
            and initialization.get("source_checkpoint") is None
            and initialization.get("pretrained_weights") is False
            and initialization.get("source_parent_age") == 0
            and initialization.get("source_arm") == f"scratch_common_seed{_member_routing_seed(training, model)}"
        )
        if (not isinstance(model, Mapping) or not isinstance(base, Mapping)
                or not valid_model
                or model.get("input_dataset") != "thermal"
                or model.get("input_capability") != "joint_prescribed_records_v1"
                or base.get("schema") != "content_hypergraph_core_v2"
                or base.get("architecture") != "A2-D"
                or base.get("module_input_dim") != 9
                or base.get("environment_input_dim") != 6
                or base.get("operating_input_dim") != 12
                or base.get("coordinate_dim") != 2
                or base.get("output_dim") != 4
                or not valid_source
                or initialization.get("seed") != training.get("seed")
                or not valid_schedule
                or training.get("seed") != _member_routing_seed(training, model)
                or training.get("primary_queries") != 1024
                or training.get("effective_batch_size") != 48
                or training.get("microbatch_cases") != 24
                or optimizer.get("kind") != "fresh_adamw_from_config"
                or optimizer.get("schedule") != "constant_config_lr"
                or float(optimizer.get("lr", -1)) != 3e-4
                or float(optimizer.get("weight_decay", -1)) != 1e-5
                or tuple(optimizer.get("betas", ())) != (0.9, 0.999)
                or float(optimizer.get("eps", -1)) != 1e-8
                or objective.get("native_weights_fixed") is not True
                or objective.get("response_training_families") != []
                or objective.get("response_fit") is not False
                or objective.get("organizer_objective") is not False):
            raise ValueError("Member-routing config differs from its scratch/native-only training contract.")
        if (
                objective.get("paired_full_restricted") is not True
                or float(objective.get("guide_kl_weight", -1)) != 0.0
                or model.get("input_dataset") != "thermal"
                or model.get("input_capability") != "joint_prescribed_records_v1"
                or training.get("seed") != _member_routing_seed(training, model)
                or initialization.get("seed") != _member_routing_seed(training, model)):
            raise ValueError("Paired member-routing objective, input contract, and seed must retain declared values.")
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
    if "write_initial_artifacts" in training and type(training["write_initial_artifacts"]) is not bool:
        raise ValueError("training.write_initial_artifacts must be boolean when supplied.")
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


def _member_routing_task_factory(config: Mapping[str, Any] | None = None) -> Callable[..., tuple[torch.nn.Module, Any]]:
    if (config is None or "member_routing" not in config
            or _member_routing_core_schema(config["member_routing"]) != R13_8_E_ANCHORED_CORE_SCHEMA
            or config["member_routing"].get("arm") not in R13_8_E_ANCHORED_ARMS):
        raise ValueError("Only anchored member-routing tasks are supported.")
    from channelthermal.training.r13_8_anchored_task import build_thermal_r13_8_anchored_member_routing_task
    return build_thermal_r13_8_anchored_member_routing_task


def _build_task(config: Mapping[str, Any], arm: str, device: torch.device,
                task_factory: Callable[..., tuple[torch.nn.Module, Any]] | None = None):
    factory = task_factory or _member_routing_task_factory(config)
    model, provider = factory(config, arm=arm, device=device)
    if model is None or provider is None:
        raise TypeError("Contextual-frontier task factory must return a model and provider.")
    if getattr(model, "architecture", None) != arm:
        raise ValueError(f"Contextual-frontier factory returned a model for {getattr(model, 'architecture', None)!r}, expected {arm}.")
    if getattr(provider, "model", None) is not model:
        raise ValueError("Contextual-frontier provider must be bound to the returned model.")
    return model, provider


def _engine_config(config: Mapping[str, Any], *, total_epochs: int | None = None,
                   checkpoint_epochs: Sequence[int] | None = None) -> EngineConfig:
    training = config["training"]
    return EngineConfig(
        seed=int(training["seed"]),
        microbatch_cases=int(training["microbatch_cases"]),
        effective_cases=int(training["effective_batch_size"]),
        total_epochs=int(training["local_horizon"] if total_epochs is None else total_epochs),
        monitor_every=int(training["monitor_every"]),
        gradient_clip=float(config["optimizer"]["gradient_clip"]),
        deterministic_case_order=True,
        checkpoint_epochs=tuple(map(int, training["checkpoint_epochs"] if checkpoint_epochs is None
                                   else checkpoint_epochs)),
        latest_every=int(training["latest_every"]),
        curve_every=int(training["curve_every"]),
        progress_enabled=bool(training["progress_enabled"]),
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        training_mode="joint",
        write_initial_artifacts=bool(training.get("write_initial_artifacts", False)),
    )


def _run_identity(config: Mapping[str, Any], provider: Any) -> dict[str, Any]:
    section_name, section = _model_section(config)
    frontier = copy.deepcopy(dict(section))
    return {
        "schema": "honf_r13_8_e_anchored_member_routing_training_v1",
        "workflow": R13_8_E_ANCHORED_WORKFLOW,
        "model_family": "e_anchored_joint_field_member_routing_v1",
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
                      config: Mapping[str, Any], fresh_namespace: bool = False,
                      checkpoint_epochs: Sequence[int] | None = None) -> tuple[int, int]:
    """Conservatively include model, named Adam moments, aliases, and atomic-save space."""
    state_bytes = max(1, _model_state_bytes(model))
    payload_bytes = 4 * state_bytes + 8 * 1024**2
    declared_checkpoints = (config["training"]["checkpoint_epochs"] if checkpoint_epochs is None
                           else checkpoint_epochs)
    milestones = sum(
        1 for epoch in declared_checkpoints
        if completed_epoch < int(epoch) <= stop_after
    )
    terminal = int(stop_after not in declared_checkpoints)
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
    manifest_schema = R13_8_E_ANCHORED_MANIFEST_SCHEMA
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


def _resume_execution_budget(
    config: Mapping[str, Any], manifest: Mapping[str, Any], latest_path: Path,
    latest_payload: Mapping[str, Any], requested_seconds: int | None,
) -> tuple[int, None]:
    if requested_seconds is not None or manifest.get("execution_budget") is not None:
        raise ValueError("Anchored runs do not support cumulative wall-clock overrides.")
    return int(config["training"]["wall_clock_budget_seconds"]), None


def _fit(model: torch.nn.Module, provider: Any, config: Mapping[str, Any], output: Path, *,
         device: torch.device, stop_after: int, resume_checkpoint: Path | None = None,
         wall_clock_budget_seconds: int | None = None,
         engine_horizon: int | None = None,
         engine_checkpoint_epochs: Sequence[int] | None = None,
         allow_horizon_extension: bool = False,
         compact_history: bool | None = None) -> dict[str, Any]:
    from .research_training import ResearchTrainingEngine

    arm = _arm(config)
    engine_config = _engine_config(config, total_epochs=engine_horizon,
                                   checkpoint_epochs=engine_checkpoint_epochs)
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
        continuation_options = {}
        if allow_horizon_extension:
            continuation_options["allow_horizon_extension"] = True
        if compact_history is not None:
            continuation_options["compact_history"] = compact_history
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
                              verify_bindings: bool = True, compact_history: bool = True) -> dict[str, Any]:
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
    return _fit(model, provider, config, output_path, device=selected_device, stop_after=stop_after,
                compact_history=compact_history)


def resume_contextual_frontier(run_dir: str | Path, *, device: str, stop_after_epoch: int,
                               devices: Sequence[str] | None = None,
                               task_factory: Callable[..., tuple[torch.nn.Module, Any]] | None = None,
                               wall_clock_budget_seconds: int | None = None,
                               extend_horizon: bool = False,
                               verify_bindings: bool = True,
                               compact_history: bool | None = None) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve(strict=True)
    layout = RunLayout(run)
    manifest_path = layout.read_path("run_manifest.json")
    manifest = _read_json(manifest_path)
    if (manifest.get("schema") != R13_8_E_ANCHORED_MANIFEST_SCHEMA
            or manifest.get("artifact_layout_version") != 1):
        raise ValueError("Run is missing the anchored member-routing manifest.")
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
    if compact_history is None:
        history_path = layout.read_path("history.json")
        saved_history = _read_json(history_path) if history_path.exists() else []
        compact_history = bool(saved_history and saved_history[0].get("history_metadata_version") == 1)
    latest_path = resolve_checkpoint(run, "latest")
    from .compat import load_trusted_checkpoint

    latest_payload = load_trusted_checkpoint(latest_path, map_location="cpu")
    current_epoch = int(latest_payload.get("epoch", -1))
    if current_epoch < 1 or stop_after_epoch <= current_epoch:
        raise ValueError("Resume stop must advance the latest completed epoch.")
    configured_horizon = int(config["training"]["local_horizon"])
    member_routing = "member_routing" in config
    horizon_extension = stop_after_epoch > configured_horizon
    if horizon_extension and not (
            extend_horizon and member_routing
            and config["training"].get("membership") == "full_train600"
            and current_epoch >= configured_horizon):
        raise ValueError(
            f"Resume stop exceeds the configured absolute {configured_horizon}-epoch horizon; "
            "the maintained member-routing runner requires --extend-horizon and a completed configured endpoint."
        )
    if extend_horizon and not member_routing:
        raise ValueError("Explicit horizon extension is supported only for maintained member-routing runs.")
    engine_checkpoint_epochs = tuple(map(int, config["training"]["checkpoint_epochs"]))
    if horizon_extension:
        engine_checkpoint_epochs = tuple(sorted({
            *engine_checkpoint_epochs,
            *range(100, stop_after_epoch + 1, 100),
        }))
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
        checkpoint_epochs=engine_checkpoint_epochs,
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
        "explicit_horizon_extension": bool(horizon_extension),
        "wall_clock_budget_seconds": selected_budget,
        "wall_clock_budget_scope": "absolute cumulative provider elapsed seconds",
        "storage": storage,
    }, sort_keys=True, default=str, allow_nan=False), flush=True)
    return _fit(model, provider, config, run, device=selected_device,
                stop_after=stop_after_epoch, resume_checkpoint=latest_path,
                wall_clock_budget_seconds=selected_budget,
                engine_horizon=(stop_after_epoch if horizon_extension else None),
                engine_checkpoint_epochs=engine_checkpoint_epochs,
                allow_horizon_extension=horizon_extension, compact_history=compact_history)


def _validate_loaded_contextual_config(config: Mapping[str, Any]) -> None:
    """Validate a manifest-held config without resolving its external input files."""
    section_name = ("member_routing" if "member_routing" in config else
                    "interaction_groups" if "interaction_groups" in config else "contextual_frontier")
    frontier = config.get(section_name, {})
    if (section_name != "member_routing" or not isinstance(frontier, Mapping)
            or frontier.get("schema") != R13_8_E_ANCHORED_CONFIG_SCHEMA
            or _member_routing_core_schema(frontier) != R13_8_E_ANCHORED_CORE_SCHEMA
            or frontier.get("arm") not in R13_8_E_ANCHORED_ARMS):
        raise ValueError("Saved config is not a supported anchored member-routing model.")
    if config.get("schema") != research.RESEARCH_SCHEMA:
        raise ValueError("Saved anchored model config has an invalid research schema.")
    if section_name == "member_routing":
        model = frontier.get("model")
        arm = frontier.get("arm")
        base = model.get("base_model_config") if isinstance(model, Mapping) else None
        training = config.get("training", {})
        initialization = config.get("initialization", {})
        membership = training.get("membership") if isinstance(training, Mapping) else None
        actual = (training.get("local_horizon"), training.get("first_segment_epochs"),
                  training.get("optimizer_schedule_horizon"),
                  training.get("checkpoint_epochs"), initialization.get("normalization_mode"))
        optimizer = config.get("optimizer", {})
        objective = config.get("objective", {})
        core_schema = _member_routing_core_schema(frontier)
        valid_schedule, valid_model = _anchored_schedule_and_model(training, initialization, model, arm)
        valid_source = (
            initialization.get("source_kind") == R13_8_E_ANCHORED_SOURCE_KIND
            and initialization.get("source_checkpoint") is None
            and initialization.get("pretrained_weights") is False
            and initialization.get("source_parent_age") == 0
            and initialization.get("source_arm") == f"scratch_common_seed{_member_routing_seed(training, model)}"
        )
        if (not isinstance(model, Mapping) or not isinstance(base, Mapping)
                or not valid_model
                or model.get("input_dataset") != "thermal"
                or model.get("input_capability") != "joint_prescribed_records_v1"
                or base.get("schema") != "content_hypergraph_core_v2"
                or base.get("architecture") != "A2-D"
                or (base.get("module_input_dim"), base.get("environment_input_dim"),
                    base.get("operating_input_dim"), base.get("coordinate_dim"),
                    base.get("output_dim")) != (9, 6, 12, 2, 4)
                or not valid_schedule
                or not valid_source
                or training.get("seed") != _member_routing_seed(training, model) or training.get("primary_queries") != 1024
                or initialization.get("seed") != _member_routing_seed(training, model)
                or optimizer.get("schedule") != "constant_config_lr"
                or float(optimizer.get("lr", -1)) != 3e-4
                or objective.get("response_training_families") != []
                or objective.get("response_fit") is not False
                or objective.get("organizer_objective") is not False):
            raise ValueError("Saved member-routing config violates its bounded scratch contract.")
        if (
                objective.get("paired_full_restricted") is not True
                or float(objective.get("guide_kl_weight", -1)) != 0.0
                or objective.get("native_weights_fixed") is not True):
            raise ValueError("Saved paired member-routing config changed its native objective contract.")
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
    "load_contextual_frontier_config",
    "resume_contextual_frontier",
    "train_contextual_frontier",
]
