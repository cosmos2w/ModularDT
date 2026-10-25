"""Small Thermal-first research runner built on the maintained HONF engine.

This module deliberately keeps experiment setup ordinary: a selected source
checkpoint, the existing Thermal provider, one local schedule, and the common
``TrainingEngine``. It does not import or modify Round9/10 campaign services.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import signal
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .run_layout import RunLayout, resolve_checkpoint
from .unified_training import (
    EngineConfig,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    SelectionPolicy,
    TrainingEngine,
    _atomic_json,
    _observed_mask_digest,
    _sampling_dataset_id,
    capture_rng_state,
    restore_rng_state,
)
from channelthermal.training.content_hypergraph_task_v2 import ThermalContentHypergraphTaskV2


RESEARCH_SCHEMA = "honf_thermal_research_run_v1"
TASK_SCHEMA = "honf_thermal_research_provider_v1"
_CONFIG_NAME = "research_run.json"
_EVAL_DIR = "evaluations/research"


def _is_round12_scratch(config: Mapping[str, Any]) -> bool:
    return config["initialization"].get("source_kind") == "scratch_compact_config"


def _is_softonly(config: Mapping[str, Any]) -> bool:
    return config.get("round12", {}).get("schema") == "thermal_compact_softonly_v2"


def _initialization_path(config: Mapping[str, Any]) -> Path:
    value = (config["dataset"]["base_recipe_path"] if _is_round12_scratch(config)
             else config["initialization"]["source_checkpoint"])
    return Path(value).expanduser().resolve(strict=True)


def _parent_age(config: Mapping[str, Any]) -> int:
    return 0 if config["initialization"]["normalization_mode"] in (
        "fresh_full_train600_stats", "classic_train600_stats", "scratch_train600_stats",
    ) else int(
        config["initialization"]["source_parent_age"],
    )


def _fresh_model_schedule_epoch(config: Mapping[str, Any], local_epoch: int) -> int:
    """Map fresh local epochs onto the native continuous controller coordinate."""
    schedule = config["initialization"].get("fresh_model_schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("Fresh full-TRAIN normalization requires an explicit native model schedule.")
    start = int(schedule["coordinate_start_epoch"])
    span = int(schedule["coordinate_span_epochs"])
    ramp = int(schedule["ramp_epochs"])
    if local_epoch < 1 or span < 1 or ramp < 1:
        raise ValueError("Fresh model schedule needs positive local epochs, span, and ramp.")
    return start + round(int(local_epoch) * span / ramp)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


_EXECUTION_OPTION_DEFAULTS = {
    "backend": "dense_weighted",
    "query_tile_size": 512,
    "checkpoint": "tile",
    "context_backend": "segment",
}


def _validated_execution_options(options: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(options))
    if set(result) != set(_EXECUTION_OPTION_DEFAULTS):
        raise ValueError("Execution options must declare backend, query tile, checkpoint, and context backend.")
    if result["backend"] not in ("segment", "dense_weighted", "source_order", "packed", "dense"):
        raise ValueError("Unsupported runtime execution backend.")
    if type(result["query_tile_size"]) is not int or result["query_tile_size"] < 1:
        raise ValueError("Runtime query tile size must be a positive integer.")
    if result["checkpoint"] not in ("off", "values", "tile"):
        raise ValueError("Runtime checkpoint mode must be off, values, or tile.")
    if result["context_backend"] not in ("segment", "dense_weighted"):
        raise ValueError("Unsupported runtime context backend.")
    return result


def _restore_checkpoint_identity_metadata(provider, payload: dict[str, Any]) -> dict[str, Any]:
    """Keep execution settings out of the saved model/provider identity fields."""
    for field, payload_key in (
        ("_research_identity_adapter_state", "adapter_state"),
        ("_research_identity_model_config", "model_config"),
    ):
        saved = getattr(provider, field, None)
        if saved is not None:
            payload[payload_key] = copy.deepcopy(saved)
    return payload


def _resolve_resume_stop(*, current_epoch: int, local_horizon: int,
                         additional_epochs: int | None, stop_after_epoch: int | None) -> int:
    if (additional_epochs is None) == (stop_after_epoch is None):
        raise ValueError("Specify exactly one of additional_epochs or stop_after_epoch.")
    if additional_epochs is not None:
        if type(additional_epochs) is not int or additional_epochs < 1:
            raise ValueError("--additional-epochs must be a positive integer.")
        stop_after = int(current_epoch) + additional_epochs
    else:
        if type(stop_after_epoch) is not int or stop_after_epoch <= current_epoch:
            raise ValueError("--stop-after-epoch must be greater than the current checkpoint epoch.")
        stop_after = stop_after_epoch
    if stop_after > local_horizon:
        raise ValueError(f"Resume would exceed local horizon {local_horizon}.")
    return stop_after


def research_membership_fingerprint(*, dataset_identity: Mapping[str, Any], training_case_ids: Sequence[str],
                                    validation_case_ids: Sequence[str]) -> str:
    """Fingerprint only dataset identity and actual TRAIN/DEV memberships."""
    payload = {
        "schema": TASK_SCHEMA,
        "dataset_identity": copy.deepcopy(dict(dataset_identity)),
        "training_case_ids": list(map(str, training_case_ids)),
        "validation_case_ids": list(map(str, validation_case_ids)),
    }
    return _canonical_sha256(payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def load_research_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve(strict=True)
    config = read_json(config_path)
    if config.get("schema") != RESEARCH_SCHEMA:
        raise ValueError(f"Research config must declare schema={RESEARCH_SCHEMA}.")
    required = {"name", "dataset", "initialization", "training", "optimizer", "execution", "storage"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Research config is missing {sorted(missing)}.")
    if config["dataset"].get("name") != "thermal":
        raise ValueError("This P0 runner currently supports the Thermal provider only.")
    base_recipe = Path(config["dataset"]["base_recipe_path"]).expanduser()
    if not base_recipe.is_absolute():
        base_recipe = config_path.parent / base_recipe
    config["dataset"]["base_recipe_path"] = str(base_recipe.resolve(strict=True))
    config["dataset"]["data_path"] = str(Path(config["dataset"]["data_path"]).expanduser().resolve(strict=True))
    config["dataset"]["manifest_path"] = str(Path(config["dataset"]["manifest_path"]).expanduser().resolve(strict=True))
    if config["dataset"].get("atlas_directory"):
        config["dataset"]["atlas_directory"] = str(Path(config["dataset"]["atlas_directory"]).expanduser().resolve(strict=True))
    _validate_research_config(config)
    return config, config_path


def _validate_research_config(config: Mapping[str, Any]) -> None:
    training = config["training"]
    optimizer = config["optimizer"]
    execution = config["execution"]
    if training.get("membership") not in (
        "fixed25_train150_dev22", "original_train600_fixed25_dev22", "classic_train600_exposed_val89",
        "parent_train600_exposed_val89", "scratch_train600_exposed_val89",
    ):
        raise ValueError("Thermal membership must be fixed25 TRAIN150/DEV22, original TRAIN600/DEV22, or an inherited full TRAIN600/VAL89 panel.")
    for field in ("local_horizon", "first_segment_epochs", "effective_batch_size", "microbatch_cases",
                  "primary_queries", "monitor_every", "latest_every", "curve_every"):
        if type(training.get(field)) is not int or training[field] < 1:
            raise ValueError(f"training.{field} must be a positive integer.")
    if training["effective_batch_size"] != 48 or training["primary_queries"] != 1024:
        raise ValueError("The formal Thermal protocol uses effective batch 48 and Q=1024.")
    if training["microbatch_cases"] > training["effective_batch_size"]:
        raise ValueError("Training microbatch cannot exceed effective batch size.")
    if training["first_segment_epochs"] > training["local_horizon"]:
        raise ValueError("First segment cannot exceed the local horizon.")
    normalization_mode = config["initialization"].get("normalization_mode")
    allowed_optimizer_contracts = {
        ("fresh_adamw_from_source_groups", "constant_source_terminal_lr"),
        ("fresh_adamw_from_classic_formal_contract", "constant_classic_formal_lr"),
        ("fresh_adamw_from_config", "constant_config_lr"),
    }
    if (optimizer.get("kind"), optimizer.get("schedule")) not in allowed_optimizer_contracts:
        raise ValueError("Research starts require a supported fresh AdamW optimizer contract.")
    if normalization_mode in ("fresh_full_train600_stats", "classic_train600_stats", "scratch_train600_stats"):
        schedule = config["initialization"].get("fresh_model_schedule")
        if not isinstance(schedule, Mapping) or any(
                type(schedule.get(key)) is not int or schedule[key] < 1 for key in
                ("coordinate_start_epoch", "coordinate_span_epochs", "ramp_epochs")):
            raise ValueError("Fresh normalization needs an explicit positive native model-clock ramp.")
        strengths = config["initialization"].get("fresh_controller_lambda0")
        arm = config["initialization"].get("fresh_routing_mode")
        expected_strengths = ({"M", "E"} if arm == "dense" else {"M", "E", "context"})
        if arm is None:
            expected_strengths = {"M", "E", "context"}
        if (not isinstance(strengths, Mapping) or set(strengths) != expected_strengths
                or any(not isinstance(value, (int, float)) or not math.isfinite(float(value))
                       or float(value) <= 0.0 for value in strengths.values())):
            raise ValueError("Fresh normalization needs positive calibrated strengths for its active M/E/context duals.")
        if not config["initialization"].get("fresh_controller_lambda0_source"):
            raise ValueError("Fresh controller strengths must record their calibration source.")
    options = execution.get("options", {})
    if options.get("backend") not in ("segment", "dense_weighted", "source_order", "packed", "dense"):
        raise ValueError("Unsupported execution backend in research config.")
    if options.get("checkpoint") not in ("off", "values", "tile"):
        raise ValueError("Unsupported activation-checkpoint option in research config.")
    if options.get("context_backend") not in ("segment", "dense_weighted"):
        raise ValueError("Unsupported context backend in research config.")
    if type(options.get("query_tile_size")) is not int or options["query_tile_size"] < 1:
        raise ValueError("execution.options.query_tile_size must be positive.")
    storage = config["storage"]
    if not isinstance(storage.get("output_root"), str) or not storage["output_root"].startswith("/data2/"):
        raise ValueError("Research output_root must be an explicit path on /data2.")
    for key in ("max_output_bytes", "max_peak_temporary_bytes", "minimum_free_headroom_bytes"):
        if type(storage.get(key)) is not int or storage[key] < 0:
            raise ValueError(f"storage.{key} must be a nonnegative integer byte count.")
    if config["initialization"].get("normalization_mode") not in (
        "inherit_source_train150", "fresh_full_train600_stats", "classic_train600_stats", "scratch_train600_stats",
        "inherit_parent_train_stats",
    ):
        raise ValueError("Select source TRAIN150 normalization, fresh full-TRAIN600 stats, saved classic TRAIN600 stats, or inherited parent TRAIN600 stats.")
    if _is_round12_scratch(config):
        init, compact = config["initialization"], config.get("round12", {})
        if (normalization_mode != "scratch_train600_stats"
                or training.get("membership") != "scratch_train600_exposed_val89"
                or init.get("source_parent_age") != 0
                or any(init.get(key) is not None for key in
                       ("source_checkpoint", "source_run_dir", "source_identity_path", "source_sha256"))
                or init.get("fresh_routing_mode") != "hypergraph"
                or init.get("fresh_model_schedule") != {"coordinate_start_epoch": 3000,
                    "coordinate_span_epochs": 1000, "ramp_epochs": 1000}
                or (optimizer.get("kind"), optimizer.get("schedule")) !=
                    ("fresh_adamw_from_config", "constant_config_lr")):
            raise ValueError("Scratch compact runs forbid checkpoint/parent bindings and require fresh TRAIN600/VAL89 initialization.")
        if _is_softonly(config):
            from .thermal_soft_organization import validate_softonly_config
            validate_softonly_config(config)
        elif (compact.get("schema") != "thermal_compact_round12_v1"
                or (compact.get("role"), compact.get("compact_arm")) != ("R1", "merge")
                or compact.get("tie_loss_gradient") != "stop_outer_probability_v1"
                or compact.get("shared_prefix_epochs") != 20
                or compact.get("cut_after_local_epoch") != 40
                or compact.get("structural_loss_scale") != 0.02
                or compact.get("calibration_max_microbatches") != 2
                or init.get("clock_mapping") != {"sampler_parent_epoch": 0,
                    "controller_parent_coordinate": 3000, "parent_update_index": 0, "parent_macro_clock": 0}):
            raise ValueError("Scratch compact runs require the amended Round12 20/40 phases and fresh clocks.")
        checkpoints = training.get("checkpoint_epochs")
        horizon = training["local_horizon"]
        if (horizon != 1000 or training["first_segment_epochs"] != horizon
                or checkpoints != (list(range(100, horizon + 1, 100)) if _is_softonly(config)
                                   else [20, 40, *range(100, horizon + 1, 100)])
                or training.get("checkpoint_every") != 100
                or training["seed"] != 0 or training["microbatch_cases"] != 24):
            raise ValueError("Scratch compact1000 requires the frozen full1000 schedule and 20/40/100-step milestones.")
    elif normalization_mode == "scratch_train600_stats":
        raise ValueError("Scratch normalization requires scratch_compact_config initialization.")
    if normalization_mode == "classic_train600_stats":
        init = config["initialization"]
        if (training.get("membership") != "classic_train600_exposed_val89"
                or init.get("source_kind") != "classic_identity_metadata_only"
                or init.get("fresh_routing_mode") not in ("dense", "hypergraph", "direct")
                or not init.get("source_identity_path")):
            raise ValueError("Classic fresh profiles require the saved classic identity, exact TRAIN600/VAL89, and a declared routing mode.")
        if config["optimizer"].get("kind") != "fresh_adamw_from_classic_formal_contract":
            raise ValueError("Classic fresh profiles must use the saved classic formal AdamW contract.")
        milestone_epochs = training.get("checkpoint_epochs")
        if (not isinstance(milestone_epochs, list) or not milestone_epochs
                or any(type(epoch) is not int or epoch < 1 or epoch > training["local_horizon"]
                       for epoch in milestone_epochs)
                or milestone_epochs != sorted(set(milestone_epochs))):
            raise ValueError("Classic profiles require sorted unique in-horizon checkpoint_epochs.")
    if normalization_mode == "inherit_parent_train_stats":
        init = config["initialization"]
        round12 = config.get("round12")
        clock = init.get("clock_mapping")
        if (init.get("source_kind") != "research_run_checkpoint_adaptation"
                or init.get("source_arm") != "U-H"
                or init.get("source_parent_age") != 1000
                or training.get("membership") != "parent_train600_exposed_val89"
                or not isinstance(round12, Mapping)
                or round12.get("schema") != "thermal_compact_round12_v1"
                or round12.get("role") not in ("common", "R0", "R1")
                or round12.get("compact_arm") not in ("common", "singleton", "merge")
                or not isinstance(clock, Mapping)
                or (clock.get("sampler_parent_epoch"), clock.get("controller_parent_coordinate"),
                    clock.get("parent_update_index"), clock.get("parent_macro_clock")) != (1000, 4000, 12, 13000)):
            raise ValueError("Round12 adaptation must bind the literal U-H e1000 parent, full TRAIN600/VAL89, and separate saved sampler/controller/update clocks.")
        if (round12.get("role"), round12.get("compact_arm")) not in (
                ("common", "common"), ("R0", "singleton"), ("R1", "merge")):
            raise ValueError("Round12 role and compact arm must be common/common, R0/singleton, or R1/merge.")
        checkpoints = training.get("checkpoint_epochs")
        if (training.get("local_horizon", 0) < 100 or training.get("first_segment_epochs") != 20
                or not isinstance(checkpoints, list) or checkpoints[:3] != [20, 40, 100]
                or checkpoints != sorted(set(checkpoints))
                or any(type(epoch) is not int or epoch < 1 or epoch > training["local_horizon"]
                       for epoch in checkpoints)
                or training.get("checkpoint_every") != 10):
            raise ValueError("Round12 defaults to a 100-epoch endpoint, common prefix 20, and checkpoints at 20/40/100 with latest every 10; longer explicit horizons remain valid.")
        if (training.get("effective_batch_size") != 48 or training.get("microbatch_cases") != 24
                or training.get("primary_queries") != 1024 or training.get("seed") != 0):
            raise ValueError("Round12 must retain the parent seed, effective batch 48, microbatch 24, and Q=1024.")
        if config["optimizer"].get("kind") != "fresh_adamw_from_source_groups":
            raise ValueError("Round12 source adaptation must construct fresh AdamW moments over the compact parameter names.")
        if round12.get("cut_after_local_epoch") != 40 or round12.get("structural_loss_scale") != 0.02:
            raise ValueError("Round12 hard cut and structural-loss calibration boundary differ from the declared protocol.")
        if round12.get("calibration_max_microbatches") != 2:
            raise ValueError("Round12 structural-loss calibration is bounded to two TRAIN microbatches.")


def verify_storage(config: Mapping[str, Any], *, output: str | Path | None = None,
                   estimated_bytes: int = 0, peak_temporary_bytes: int = 0) -> dict[str, Any]:
    """Verify configurable /data2 output, total budget and free-space headroom."""
    data2 = Path("/data2")
    if not data2.is_dir() or not os.path.ismount(data2):
        raise RuntimeError("/data2 is not mounted; refusing to write research artifacts.")
    storage = config["storage"]
    root = Path(storage["output_root"]).expanduser().resolve(strict=False)
    data2_stat = data2.stat()
    parent = root
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if parent.stat().st_dev != data2_stat.st_dev:
        raise RuntimeError("Configured research root does not resolve to the mounted /data2 filesystem.")
    if output is not None:
        resolved_output = Path(output).expanduser().resolve(strict=False)
        try:
            resolved_output.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Run output must be inside {root}.") from error
    usage = os.statvfs(data2)
    free_bytes = int(usage.f_bavail * usage.f_frsize)
    headroom = int(storage["minimum_free_headroom_bytes"])
    if free_bytes - int(estimated_bytes) < headroom:
        raise RuntimeError("The requested run would breach configured /data2 free-space headroom.")
    current_bytes = _tree_bytes(root) if root.exists() else 0
    related_bytes = 0
    for raw in storage.get("related_output_roots", ()):
        related = Path(raw).expanduser().resolve(strict=True)
        if not related.is_relative_to(data2.resolve()) or related.stat().st_dev != data2_stat.st_dev:
            raise RuntimeError("Related research evidence must remain on the mounted /data2 filesystem.")
        if related == root or related.is_relative_to(root) or root.is_relative_to(related):
            raise ValueError("Related evidence roots must not overlap the scientific run root.")
        related_bytes += _tree_bytes(related)
    current_bytes += related_bytes
    output_budget = int(storage["max_output_bytes"])
    if current_bytes + int(estimated_bytes) > output_budget:
        raise RuntimeError(
            f"Research output budget exceeded: {current_bytes} existing + {estimated_bytes} estimated bytes "
            f"> {output_budget}."
        )
    peak_limit = int(storage["max_peak_temporary_bytes"])
    if int(peak_temporary_bytes) > peak_limit:
        raise RuntimeError("Estimated peak temporary space exceeds the configured maximum.")
    return {
        "data2_mount": str(data2.resolve()),
        "output_root": str(root),
        "filesystem_device": int(data2_stat.st_dev),
        "free_bytes_before": free_bytes,
        "output_bytes_before": current_bytes,
        "related_evidence_bytes_before": related_bytes,
        "estimated_new_bytes": int(estimated_bytes),
        "output_budget_bytes": output_budget,
        "estimated_peak_temporary_bytes": int(peak_temporary_bytes),
        "maximum_peak_temporary_bytes": peak_limit,
        "minimum_free_headroom_bytes": headroom,
    }


def _tree_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [name for name in names if not (Path(directory) / name).is_symlink()]
        for name in files:
            item = Path(directory) / name
            if not item.is_symlink():
                try:
                    total += item.stat().st_size
                except FileNotFoundError:
                    continue
    return total


def _estimate_run_bytes(config: Mapping[str, Any], *, epochs: int, source_size: int, diagnostic: bool) -> int:
    training = config["training"]
    interval = int(training["checkpoint_every"])
    configured_epochs = training.get("checkpoint_epochs")
    milestones = (sum(1 for epoch in configured_epochs if int(epoch) <= epochs)
                  if isinstance(configured_epochs, list) else math.ceil(epochs / interval))
    milestones = max(1, milestones)
    # The engine retains each declared milestone, best and latest aliases.
    model_payloads = (milestones + 2) * int(source_size * 1.4)
    plots_and_history = 256 * 1024**2 if not diagnostic else 128 * 1024**2
    return model_payloads + plots_and_history


def resolve_device(device: str, *, permitted_devices: Sequence[str] | None = None) -> tuple[torch.device, dict[str, Any]]:
    requested = torch.device(device)
    if requested.type == "cpu":
        if permitted_devices is not None and "cpu" not in permitted_devices:
            raise ValueError("CPU was not included in the explicitly permitted device list.")
        return requested, {"requested": "cpu", "selected": "cpu", "physical_mapping": "cpu"}
    if requested.type != "cuda":
        raise ValueError("Research runner supports CPU or CUDA devices.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process; use --device cpu or a CUDA-visible process.")
    index = 0 if requested.index is None else requested.index
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise ValueError(f"Requested visible CUDA ordinal {index} is outside this process's {count} devices.")
    normalized = f"cuda:{index}"
    if permitted_devices is not None and normalized not in permitted_devices:
        raise ValueError("The selected device is not in the explicitly permitted --devices list.")
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_tokens = [] if raw_visible in (None, "") else [item.strip() for item in raw_visible.split(",")]
    mapping = visible_tokens[index] if index < len(visible_tokens) else f"physical ordinal {index} (identity mapping)"
    properties = torch.cuda.get_device_properties(index)
    memory = None
    try:
        free, total = torch.cuda.mem_get_info(index)
        memory = {"free_bytes_advisory": int(free), "total_bytes": int(total)}
    except Exception:
        pass
    return torch.device(normalized), {
        "requested": device,
        "selected": normalized,
        "visible_ordinal": index,
        "physical_mapping": mapping,
        "cuda_visible_devices": raw_visible,
        "device_name": str(properties.name),
        "memory_snapshot_advisory_only": memory,
    }


def _permitted_order(device: str, devices: Sequence[str] | None) -> list[str]:
    if devices is None:
        return [device]
    ordered = [str(item).strip() for item in devices if str(item).strip()]
    if not ordered or len(ordered) != len(set(ordered)):
        raise ValueError("Explicit permitted devices must be a nonempty unique list.")
    normalized = "cuda:0" if device == "cuda" else device
    if normalized not in ordered:
        raise ValueError("--device must also be named in --devices when alternates are supplied.")
    return [normalized, *(item for item in ordered if item != normalized)]


def _install_clean_stop_handlers(output: Path):
    previous: dict[int, Any] = {}
    request = {
        "schema": "ordinary_clean_stop_request_v1",
        "request_id": f"sigint-{os.getpid()}-{uuid.uuid4().hex}",
        "requested_unix": time.time(),
        "requested_by": "research_runner_signal_handler",
    }

    def handler(signum, frame):
        del frame
        request["signal"] = int(signum)
        _atomic_json(output / "CLEAN_STOP_REQUEST.json", request)
        print(json.dumps({"event": "clean_stop_requested", **request}), flush=True)

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)
    return previous


def _restore_handlers(previous: Mapping[int, Any]) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def _cpu_state_snapshot(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        try:
            result = type(value)()
        except TypeError:
            result = {}
        for key, item in value.items():
            result[key] = _cpu_state_snapshot(item)
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, tuple):
        return tuple(_cpu_state_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_cpu_state_snapshot(item) for item in value]
    return copy.deepcopy(value)


def _round12_model_parameter_names(model) -> set[str]:
    """Map the compact core's frozen policy names onto adapter parameter names."""
    core = getattr(model, "core", model)
    declared = getattr(core, "compact_frozen_parameter_names", ())
    if callable(declared):
        declared = declared()
    if not declared:
        return set()
    names = dict(model.named_parameters())
    result = set()
    for core_name in map(str, declared):
        if core_name in names:
            result.add(core_name)
            continue
        matches = {name for name in names if name.endswith("." + core_name)}
        if len(matches) != 1:
            raise ValueError(f"Compact frozen parameter {core_name!r} does not map uniquely into the Thermal model.")
        result.update(matches)
    return result


def _round12_adaptation_source_components(config: Mapping[str, Any], device: torch.device):
    """Load the literal U-H parent through its owned run and migrate its core."""
    init = config["initialization"]
    source_path = Path(init["source_checkpoint"]).expanduser().resolve(strict=True)
    source_run = Path(init["source_run_dir"]).expanduser().resolve(strict=True)
    payload, selected_path = _read_run_checkpoint(source_run, source_path)
    if selected_path != source_path:
        raise ValueError("Round12 source resolution changed the literal e1000 checkpoint path.")
    actual_sha = sha256_file(source_path)
    if init.get("source_sha256") and init["source_sha256"] != actual_sha:
        raise ValueError("Round12 U-H parent checkpoint bytes differ from its configured SHA256.")
    identity = payload.get("experiment_identity", {})
    provider_identity = identity.get("provider_identity", {})
    recipe = identity.get("recipe", {})
    if (identity.get("family") != TASK_SCHEMA
            or identity.get("model_family") != "unified_continuous_joint_native_v1"
            or identity.get("dataset") != "thermal"
            or int(payload.get("epoch", -1)) != 1000
            or int(init.get("source_parent_age", -1)) != 1000
            or recipe.get("config_name") != "thermal_u_h_full5000_v1"):
        raise ValueError("Round12 requires the self-contained Thermal U-H e1000 research checkpoint.")
    if (len(provider_identity.get("training_case_ids", ())) != 600
            or len(provider_identity.get("validation_case_ids", ())) != 89
            or provider_identity.get("validation_scope") != (
                "canonical89_exposed_original_test_excluding_0273; unexposed remainder excluded"
            )):
        raise ValueError("Round12 parent must bind its exact full TRAIN600 and exposed canonical VAL89 memberships.")
    expected_sampling = identity.get("sampling_dataset_id")
    if (not isinstance(expected_sampling, str)
            or _sampling_dataset_id(provider_identity) != expected_sampling):
        raise ValueError("Round12 parent sampling namespace does not match its sealed TRAIN membership.")
    adapter_state = provider_identity.get("adapter_state")
    if not isinstance(adapter_state, Mapping):
        raise ValueError("Round12 parent checkpoint lacks self-contained Thermal adapter state.")
    expected_normalizer = config["round12"].get("expected_parent_normalizer_key")
    if adapter_state.get("normalizer_key") != expected_normalizer:
        raise ValueError("Round12 parent normalizer differs from the configured inherited TRAIN600 statistics.")
    if expected_sampling != config["round12"].get("expected_parent_sampling_dataset_id"):
        raise ValueError("Round12 parent sampling namespace differs from the configured U-H lineage.")
    # This safe adapter reconstruction owns the parent's learned core state and
    # keeps the normalizer, grid, and physical extraction boundary intact.
    parent_model = _model_from_run_payload(payload, device)
    parent_core = getattr(parent_model, "core", None)
    if parent_core is None or getattr(parent_core, "routing_mode", None) != "hypergraph":
        raise ValueError("Round12 source is not the learned U-H hypergraph model.")
    for name, expected in (("joint_epoch", 4000), ("joint_update_index", 12), ("joint_macro_clock", 13000)):
        actual = getattr(parent_core, name, None)
        if isinstance(actual, torch.Tensor):
            actual = int(actual.detach().cpu())
        if actual is None or int(actual) != expected:
            raise ValueError(f"Round12 U-H parent {name} clock is {actual!r}, expected {expected}.")

    from honf_forward_core.continuous_joint.unified import CompactUnifiedJointCore

    compact_core = CompactUnifiedJointCore.from_unified_parent(parent_core)
    parent_model.core = compact_core
    compact_core.set_execution_options(**config["execution"]["options"])
    coefficient_setter = getattr(compact_core, "set_compact_loss_coefficients", None)
    if not callable(coefficient_setter):
        raise TypeError("Round12 compact core is missing its persistent loss-coefficient setter.")
    coefficient_setter(tie=0.0, capacity=0.0)
    parent_model.train(True)

    source_contract = identity.get("optimizer_schedule_contract", payload.get("optimizer_schedule_contract"))
    if not isinstance(source_contract, Sequence) or len(source_contract) != 1:
        raise ValueError("Round12 U-H parent must expose its one-group formal_joint AdamW schedule.")
    source_group = copy.deepcopy(dict(source_contract[0]))
    schedule = source_group.get("schedule", {})
    if (source_group.get("name") != "formal_joint"
            or abs(float(schedule.get("final_lr", -1.0)) - 3.0e-4) > 1.0e-12
            or abs(float(source_group.get("weight_decay", -1.0)) - 1.0e-5) > 1.0e-12):
        raise ValueError("Round12 U-H source optimizer differs from formal_joint AdamW at 3e-4/1e-5.")
    active_names = [name for name, parameter in parent_model.named_parameters() if parameter.requires_grad]
    if not active_names:
        raise ValueError("Round12 compact migration produced no trainable parameters.")
    source_group["parameter_names"] = active_names
    optimizer_contract = [source_group]
    source_payload = copy.deepcopy(payload)
    source_payload["round12_parent_sampling_dataset_id"] = expected_sampling
    source_payload["round12_parent_normalizer_key"] = expected_normalizer
    source_payload["compact_parameter_migration"] = copy.deepcopy(
        getattr(compact_core, "compact_parameter_migration", None)
    )
    source_payload["round12_optimizer_schedule_contract"] = optimizer_contract
    return parent_model, source_payload, source_path, actual_sha


def _scratch_compact_components(config: Mapping[str, Any], device: torch.device):
    """Construct every weight from config; fit normalization on TRAIN600 only."""
    from channelthermal.training.content_hypergraph_task_v2 import build_thermal_content_hypergraph_task_v2
    from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
    from honf_forward_core.continuous_joint.unified import UnifiedJointCore, CompactUnifiedJointCore

    recipe = read_json(config["dataset"]["base_recipe_path"])
    recipe.update(data_path=config["dataset"]["data_path"], manifest=config["dataset"]["manifest_path"])
    if config["dataset"].get("atlas_directory"):
        recipe["atlas_directory"] = config["dataset"]["atlas_directory"]
    _, base, _ = build_thermal_content_hypergraph_task_v2(recipe, device="cpu")
    if len(base.training_case_ids) != 600 or len(base.validation_case_ids) != 89:
        raise ValueError("Scratch formal factory must provide TRAIN600 and canonical VAL89.")
    provider_identity = copy.deepcopy(dict(base.identity_payload()))
    state = base.model.state_payload()
    base_config = base.model.core.get_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config["training"]["seed"]))
        fresh = UnifiedJointCore(base_config, input_dataset="thermal", routing_mode="hypergraph")
        # Copy only newly generated random tensors inside this initialization;
        # no saved model or checkpoint enters this path.
        core = (CompactUnifiedJointCore.from_unified_parent(fresh, compact_mode="soft_only")
                if _is_softonly(config) else CompactUnifiedJointCore.from_unified_parent(fresh))
    core.set_execution_options(**config["execution"]["options"])
    for name, value in config["initialization"]["fresh_controller_lambda0"].items():
        core.joint_budget.duals[name].set_calibrated_lambda0(float(value))
    if int(core.joint_epoch) != 3000 or int(core.joint_macro_clock) != 0:
        raise RuntimeError("Scratch core must start with fresh work-controller clocks.")
    core.set_compact_phase(0, arm="soft_only" if _is_softonly(config) else "merge")
    adapter = ThermalContentHypergraphAdapterV2(
        state["normalization_stats"], core=core,
        nx=state["native_grid_nx_ny"][0], ny=state["native_grid_nx_ny"][1],
        environment_nx=state["environment_grid_nx_ny"][0],
        environment_ny=state["environment_grid_nx_ny"][1],
    ).to(device)
    recipe_optimizer = recipe["optimizer"]
    lr = float(recipe_optimizer["lr"])
    schedule = asdict(ScheduleSpec(peak_lr=lr, warmup_start_lr=lr, warmup_epochs=0,
        hold_through_epoch=1000, total_epochs=1001, final_lr=lr))
    contract = [{"name": "formal_joint", "parameter_names": [
        n for n, p in adapter.named_parameters() if p.requires_grad], "schedule": schedule,
        "weight_decay": float(recipe_optimizer["weight_decay"]),
        "betas": list(recipe_optimizer["betas"]), "eps": float(recipe_optimizer["eps"])}]
    payload = {"epoch": 0, "experiment_identity": {"provider_identity": provider_identity,
        "optimizer_schedule_contract": contract}, "compact_parameter_migration":
        copy.deepcopy(core.compact_parameter_migration), "pretrained_weights_loaded": False}
    path = _initialization_path(config)
    return adapter, payload, path, sha256_file(path)


def _source_components(config: Mapping[str, Any], device: torch.device):
    init = config["initialization"]
    if _is_round12_scratch(config):
        return _scratch_compact_components(config, device)
    source_path = Path(init["source_checkpoint"]).expanduser().resolve(strict=True)
    expected_sha = init.get("source_sha256")
    actual_sha = sha256_file(source_path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("Source initialization checkpoint bytes differ from the optional preset SHA256.")
    if init.get("source_kind") == "research_run_checkpoint_adaptation":
        return _round12_adaptation_source_components(config, device)
    from .compat import load_trusted_checkpoint
    if init.get("source_kind") == "classic_identity_metadata_only":
        # The classic weight file is bound as provenance, but it is never
        # deserialized. Architecture, fitted TRAIN600 normalization, exact
        # memberships and the AdamW schedule come from its small saved identity.
        identity_path = Path(init["source_identity_path"]).expanduser().resolve(strict=True)
        identity = read_json(identity_path)
        recipe = identity.get("recipe", {})
        provider_identity = identity.get("provider_identity", {})
        source_epoch = int(recipe.get("total_epochs", -1))
        if (identity.get("workflow") != "content_hypergraph_joint_v2"
                or recipe.get("protocol") != "formal_d_full_train_v1"
                or recipe.get("architecture") != "A2-D"
                or int(recipe.get("seed", -1)) != 0
                or source_epoch != int(init.get("source_parent_age", -2))
                or len(provider_identity.get("training_case_ids", ())) != 600
                or len(provider_identity.get("validation_case_ids", ())) != 89
                or provider_identity.get("validation_scope") != "canonical89_exposed_original_test_excluding_0273"):
            raise ValueError("Classic identity does not declare the authorized A2-D TRAIN600/exposed VAL89 contract.")
        adapter_state = provider_identity.get("adapter_state", {})
        adapter_model_config = provider_identity.get("model_config", {})
        base_model_config = adapter_state.get("core")
        if (not isinstance(base_model_config, Mapping)
                or base_model_config.get("schema") != "content_hypergraph_core_v2"
                or base_model_config.get("architecture") != "A2-D"
                or adapter_model_config.get("core") != base_model_config):
            raise ValueError("Classic identity lacks the saved actual A2-D base architecture config.")
        if provider_identity.get("normalization_stats_sha256") != "7c8f7691455b01c42a7caa626257393c5e40d4d2482ab85847bf1f278e9e754c":
            raise ValueError("Classic TRAIN600 normalizer statistics differ from the authorized saved identity.")
        source_payload = {
            "epoch": source_epoch,
            "experiment_identity": {
                "recipe": recipe,
                "provider_identity": provider_identity,
                "optimizer_schedule_contract": identity.get("optimizer_schedule_contract"),
            },
            "classic_identity_path": str(identity_path),
            "classic_sampling_dataset_id": identity.get("sampling_dataset_id"),
            "classic_checkpoint_sha256": actual_sha,
        }
        source_contract = identity.get("optimizer_schedule_contract")
        if not isinstance(source_contract, Sequence) or len(source_contract) != 1:
            raise ValueError("Classic identity lacks the one-group formal AdamW schedule contract.")
        classic_group = source_contract[0]
        if (classic_group.get("name") != "formal_joint"
                or abs(float(classic_group.get("schedule", {}).get("final_lr", -1.0)) - 3e-4) > 1e-12
                or abs(float(classic_group.get("weight_decay", -1.0)) - 1e-5) > 1e-12):
            raise ValueError("Classic formal optimizer contract differs from the declared 3e-4/1e-5 AdamW setup.")
        from honf_forward_core.continuous_joint.unified import UnifiedJointCore
        from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2

        fresh_base_config = copy.deepcopy(dict(base_model_config))
        fresh_base_config["init_seed"] = int(config["training"].get("seed", 0))
        routing_mode = str(init["fresh_routing_mode"])
        execution_options = config["execution"]["options"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config["training"].get("seed", 0)))
            core = UnifiedJointCore(
                fresh_base_config,
                input_dataset="thermal",
                routing_mode=routing_mode,
                native_backend=execution_options["backend"],
                checkpoint_pipeline=execution_options["checkpoint"] != "off",
                query_tile_size=int(execution_options["query_tile_size"]),
                entmax_temperature=1.0,
            )
        core.set_execution_options(**execution_options)
        strengths = init["fresh_controller_lambda0"]
        duals = getattr(getattr(core, "joint_budget", None), "duals", {})
        if set(duals) != set(strengths):
            raise ValueError(
                f"Fresh {routing_mode} core exposes duals {sorted(duals)}, "
                f"but config declares {sorted(strengths)}."
            )
        for name, value in strengths.items():
            dual = duals[name]
            setter = getattr(dual, "set_calibrated_lambda0", None)
            if not callable(setter):
                raise TypeError(f"Fresh {name} controller lacks set_calibrated_lambda0().")
            setter(float(value))
        stats = adapter_state.get("normalization_stats")
        if not isinstance(stats, Mapping):
            raise ValueError("Classic identity lacks saved TRAIN600 normalization statistics.")
        adapter = ThermalContentHypergraphAdapterV2(
            stats,
            core=core,
            nx=int(adapter_state["native_grid_nx_ny"][0]),
            ny=int(adapter_state["native_grid_nx_ny"][1]),
            environment_nx=int(adapter_state["environment_grid_nx_ny"][0]),
            environment_ny=int(adapter_state["environment_grid_nx_ny"][1]),
        ).to(device)
        if adapter.normalizer_key != adapter_state.get("normalizer_key"):
            raise ValueError("Fresh model did not bind the exact saved classic TRAIN600 normalizer.")
        adapter.train(True)
        return adapter, source_payload, source_path, actual_sha

    source_payload = load_trusted_checkpoint(source_path, map_location="cpu")
    identity = source_payload.get("experiment_identity", {})
    recipe = identity.get("recipe", {})
    if (identity.get("family") != "unified_continuous_joint_native_v1"
            or identity.get("workflow") != "continuous_joint_train_v1"
            or recipe.get("dataset") != "thermal"
            or recipe.get("arm") != init.get("source_arm")
            or int(source_payload.get("epoch", -1)) != int(init.get("source_parent_age", -2))):
        raise ValueError("Source is not the declared completed Thermal U-triplet model export.")
    if source_payload["experiment_identity"]["provider_identity"].get("source_binding", {}).get(
            "dataset_path") != config["dataset"].get("data_path"):
        raise ValueError("Preset dataset path differs from the source checkpoint's packed dataset binding.")
    from .continuous_joint_runtime import load_continuous_joint_checkpoint
    source_model, _, validated = load_continuous_joint_checkpoint(source_path, device=str(device))
    if int(validated.get("epoch", -1)) != int(init["source_parent_age"]):
        raise ValueError("Validated source loader returned a different parent age.")
    core = getattr(source_model, "core", source_model)
    set_options = getattr(core, "set_execution_options", None)
    if not callable(set_options):
        raise TypeError("Loaded continuous core lacks the maintained execution-options API.")
    set_options(**config["execution"]["options"])
    source_model.train(True)
    return source_model, source_payload, source_path, actual_sha


def _read_run_checkpoint(run_dir: Path, selector: str = "latest") -> tuple[dict[str, Any], Path]:
    from .compat import load_trusted_checkpoint
    path = resolve_checkpoint(run_dir, selector).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("Selected checkpoint is outside the declared research run.")
    if not path.is_file():
        raise FileNotFoundError(f"Run has no {selector} checkpoint: {path}")
    payload = load_trusted_checkpoint(path, map_location="cpu")
    saved_identity = read_json(RunLayout(run_dir).read_path("experiment_identity.json"))
    if _canonical_sha256(payload.get("experiment_identity")) != _canonical_sha256(saved_identity):
        from .unified_training import _resume_identity_amendment
        receipt = RunLayout(run_dir).read_path("configs/horizon_extension.json")
        if not receipt.exists() or read_json(receipt).get("execution_horizon") != saved_identity.get("engine_config", {}).get("total_epochs"):
            raise ValueError("Selected checkpoint identity differs from the declared research run.")
        _resume_identity_amendment(payload.get("experiment_identity", {}), saved_identity,
                                   allow_horizon_extension=True)
    return payload, path


def _model_from_run_payload(payload: Mapping[str, Any], device: torch.device):
    identity = payload.get("experiment_identity", {})
    provider_identity = identity.get("provider_identity", {})
    adapter_state = provider_identity.get("adapter_state")
    if not isinstance(adapter_state, Mapping):
        raise ValueError("Research checkpoint is missing its self-contained Thermal adapter state.")
    from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
    model = ThermalContentHypergraphAdapterV2.from_state_payload(
        adapter_state, device=str(device), state_dict=payload["model_state_dict"],
    )
    core = getattr(model, "core", model)
    compact_adaptation = provider_identity.get("recipe", {}).get("compact_adaptation")
    saved_migration = (compact_adaptation.get("parameter_migration")
                       if isinstance(compact_adaptation, Mapping) else None)
    if saved_migration is not None:
        if not isinstance(saved_migration, Mapping):
            raise ValueError("Compact checkpoint migration metadata must be a named mapping.")
        actual_core_names = dict(core.named_parameters())
        actual_new_names = {name for name in actual_core_names if name.startswith("compact_retrieval.")}
        declared_new_names = tuple(map(str, saved_migration.get("new_names", ())))
        if (not declared_new_names or len(set(declared_new_names)) != len(declared_new_names)
                or set(declared_new_names) != actual_new_names):
            raise ValueError("Saved compact new-parameter names differ from the reconstructed retrieval module.")
        declared_dropped = tuple(map(str, saved_migration.get("dropped_names", ())))
        if (len(set(declared_dropped)) != len(declared_dropped)
                or not set(declared_dropped) <= set(actual_core_names)):
            raise ValueError("Saved compact optimizer-exclusion names differ from the reconstructed core.")
        for key in ("copied_names", "optimizer_excluded_names", "state_dropped_names",
                    "cloned_route_and_reader_names"):
            values = tuple(map(str, saved_migration.get(key, ())))
            if len(values) != len(set(values)):
                raise ValueError(f"Saved compact migration field {key!r} contains duplicate names.")
        if not set(map(str, saved_migration.get("optimizer_excluded_names", ()))) <= set(declared_dropped):
            raise ValueError("Saved compact optimizer exclusions are not covered by the dropped-name contract.")
        core.compact_parameter_migration = copy.deepcopy(dict(saved_migration))
    return model


class ResearchThermalProvider(ThermalContentHypergraphTaskV2):
    """Task identity, optimizer, and full-TRAIN membership wrapper.

    Runtime batching/losses/validation are inherited directly from the actual
    Thermal v2 provider built by ``build_thermal_content_hypergraph_task_v2``.
    """

    def __init__(self, base_provider, *, model, config, train_ids, validation_ids,
                 training_cases, validation_cases, source_payload, optimizer_contract,
                 diagnostic=False, identity_microbatch=None, local_horizon=None,
                 classic_provider_identity=None):
        self.__dict__ = base_provider.__dict__
        self.model = self.adapter = model
        self.device = next(model.parameters()).device
        self.training_cases = tuple(training_cases)
        self.validation_cases = tuple(validation_cases)
        self.training_case_ids = tuple(map(str, train_ids))
        self.validation_case_ids = tuple(map(str, validation_ids))
        self._train_index_by_id = {case_id: index for index, case_id in enumerate(self.training_case_ids)}
        self._validation_index_by_id = {case_id: index for index, case_id in enumerate(self.validation_case_ids)}
        self._validation_batches_cache = None
        # Objective and batch semantics are inherited from the maintained v2
        # provider; the classic full-data profile uses its canonical formal
        # TRAIN600/VAL89 membership and sampling namespace.
        self.protocol = "formal_d_full_train_v1" if classic_provider_identity is not None else "development"
        if classic_provider_identity is not None:
            self.validation_scope = "canonical89_exposed_original_test_excluding_0273"
        self.total_epochs = int(local_horizon or config["training"]["local_horizon"])
        self._round6_sampling_epoch_limit = _parent_age(config) + self.total_epochs
        self.microbatch_size = int(config["training"]["microbatch_cases"])
        self._identity_microbatch_size = int(identity_microbatch or self.microbatch_size)
        self.effective_batch_size = int(config["training"]["effective_batch_size"])
        self.primary_queries = int(config["training"]["primary_queries"])
        self.seed = int(config["training"].get("seed", 0))
        self._research_config = copy.deepcopy(dict(config))
        self._research_source_payload = source_payload
        self._research_optimizer_contract = copy.deepcopy(list(optimizer_contract))
        self._research_local_horizon = self.total_epochs
        self._research_case_limit = len(train_ids) if diagnostic else None
        self._research_membership = str(config["training"]["membership"])
        self._research_normalization_mode = str(config["initialization"]["normalization_mode"])
        self._classic_provider_identity = (None if classic_provider_identity is None else
                                           copy.deepcopy(dict(classic_provider_identity)))
        self._round12 = copy.deepcopy(dict(config.get("round12", {}))) if (
            config["initialization"]["normalization_mode"] in ("inherit_parent_train_stats", "scratch_train600_stats")
        ) else None
        self._round12_parent_macro_clock = int(
            config["initialization"].get("clock_mapping", {}).get("parent_macro_clock", 0)
        )
        self._round12_local_epoch = 0
        self.sampling_dataset_id = _sampling_dataset_id(self.identity_payload())
        if _is_softonly(config):
            from .thermal_soft_organization import SoftOrganizationRuntime
            self.soft_organization = SoftOrganizationRuntime(self)

    def before_optimizer_step(self, model, optimizer, losses, work, *, epoch, update_index):
        runtime = getattr(self, "soft_organization", None)
        if runtime is not None:
            runtime.before_step(model, optimizer, losses, work, epoch=epoch, update_index=update_index)

    def completed_training_epoch(self, model, engine, epoch, output, row):
        runtime = getattr(self, "soft_organization", None)
        if runtime is not None:
            runtime.completed_epoch(model, engine, epoch, Path(output), row)

    def training_state_dict(self):
        base_reader = getattr(super(ResearchThermalProvider, self), "training_state_dict", None)
        base_state = base_reader() if callable(base_reader) else {}
        if self._round12 is None:
            return copy.deepcopy(base_state) if isinstance(base_state, Mapping) else {}
        core = getattr(self.model, "core", self.model)
        compact_export = getattr(core, "export_compact_state", None)
        if not callable(compact_export):
            raise TypeError("Round12 checkpointing requires the compact core's explicit state export.")
        current_macro = getattr(core, "joint_macro_clock", None)
        if isinstance(current_macro, torch.Tensor):
            current_macro = int(current_macro.detach().cpu())
        if current_macro is None:
            raise ValueError("Round12 compact core omitted its native joint macro clock.")
        current_update = getattr(core, "joint_update_index", None)
        if isinstance(current_update, torch.Tensor):
            current_update = int(current_update.detach().cpu())
        current_epoch = getattr(core, "joint_epoch", None)
        if isinstance(current_epoch, torch.Tensor):
            current_epoch = int(current_epoch.detach().cpu())
        result = {
            "schema": "thermal_compact_round12_provider_state_v1",
            "base": copy.deepcopy(dict(base_state)) if isinstance(base_state, Mapping) else {},
            "compact_state": copy.deepcopy(compact_export()),
            "clocks": {
                **copy.deepcopy(dict(self._research_config["initialization"]["clock_mapping"])),
                "local_epoch": int(compact_export().get("local_epoch", 0)),
                "controller_epoch": int(current_epoch),
                "macro_update_index": int(current_update),
                "joint_macro_clock": int(current_macro),
                "optimizer_updates_since_parent": int(current_macro) - self._round12_parent_macro_clock,
            },
        }
        runtime = getattr(self, "soft_organization", None)
        if runtime is not None:
            result["soft_organization"] = runtime.state_dict()
        return result

    def load_training_state_dict(self, state):
        if not isinstance(state, Mapping) or state.get("schema") != "thermal_compact_round12_provider_state_v1":
            loader = getattr(super(ResearchThermalProvider, self), "load_training_state_dict", None)
            if callable(loader):
                loader(state)
            elif state not in ({}, None):
                raise ValueError("Round12 provider received an unsupported checkpoint state wrapper.")
            return
        base_loader = getattr(super(ResearchThermalProvider, self), "load_training_state_dict", None)
        if callable(base_loader):
            base_loader(state.get("base", {}))
        core = getattr(self.model, "core", self.model)
        compact_loader = getattr(core, "load_compact_state", None)
        if not callable(compact_loader):
            raise TypeError("Round12 resume requires the compact core's explicit state loader.")
        compact_loader(state["compact_state"])
        runtime = getattr(self, "soft_organization", None)
        if runtime is not None:
            runtime.load_state_dict(state["soft_organization"])
        clocks = state.get("clocks")
        expected = self._research_config["initialization"]["clock_mapping"]
        if not isinstance(clocks, Mapping) or any(clocks.get(key) != expected[key] for key in (
                "sampler_parent_epoch", "controller_parent_coordinate", "parent_update_index",
                "parent_macro_clock")):
            raise ValueError("Round12 provider checkpoint clocks differ from this run's sealed parent mapping.")
        actual_macro = getattr(core, "joint_macro_clock", None)
        if isinstance(actual_macro, torch.Tensor):
            actual_macro = int(actual_macro.detach().cpu())
        if actual_macro != clocks.get("joint_macro_clock"):
            raise ValueError("Round12 compact state and saved macro-update clock disagree.")
        if int(clocks.get("optimizer_updates_since_parent", -1)) != int(actual_macro) - self._round12_parent_macro_clock:
            raise ValueError("Round12 cumulative optimizer-update count differs from the inherited core clock.")

    def identity_payload(self) -> Mapping[str, Any]:
        from channelthermal.training.content_hypergraph_task import _ids_sha256
        payload = copy.deepcopy(dict(super(ResearchThermalProvider, self).identity_payload()))
        source_binding = payload.get("source_binding", {})
        dataset_identity = {
            key: source_binding[key]
            for key in ("dataset_id", "dataset_size_bytes", "dataset_mtime_ns",
                        "catalog_metadata_sha256", "base_manifest_sha256")
            if key in source_binding
        }
        research_membership_sha = research_membership_fingerprint(
            dataset_identity=dataset_identity,
            training_case_ids=self.training_case_ids,
            validation_case_ids=self.validation_case_ids,
        )
        classic_identity = self._classic_provider_identity
        membership_sha = (classic_identity["subset_manifest_sha256"] if classic_identity is not None
                          else research_membership_sha)
        if not isinstance(membership_sha, str) or len(membership_sha) != 64:
            raise ValueError("Thermal sampling identity requires a sealed membership key.")
        payload.pop("native_sampling_identity", None)
        payload.update({
            "schema": TASK_SCHEMA,
            "protocol": self.protocol,
            "dataset_profile": ("diagnostic_train_subset_dev22" if self._research_case_limit else
                self._research_membership),
            "subset_manifest_sha256": membership_sha,
            "training_case_ids": list(self.training_case_ids),
            "training_case_ids_sha256": _ids_sha256(self.training_case_ids),
            "validation_case_ids": list(self.validation_case_ids),
            "validation_case_ids_sha256": _ids_sha256(self.validation_case_ids),
            "normalization_stats_sha256": (
                classic_identity.get("normalization_stats_sha256") if classic_identity is not None else
                self.model.state_payload()["normalizer_key"]
            ),
            "case_limit_diagnostic": self._research_case_limit,
            "validation_scope": (
                "canonical89_exposed_original_test_excluding_0273; unexposed remainder excluded"
                if classic_identity is not None else
                "fixed25_v1_DEV22_exposed_original_test_subset; locked remainder excluded"
            ),
            "source_binding": {
                **dict(payload.get("source_binding", {})),
                "protocol": TASK_SCHEMA,
                "dataset_profile": self._research_membership,
                "normalization_fit_scope": (
                    "freshly fitted original TRAIN600 only; no pretrained model" if _is_round12_scratch(self._research_config) else
                    "inherited U-H full TRAIN600 statistics; no refit" if self._round12 is not None else
                    "original TRAIN600 only; saved classic statistics reused" if classic_identity is not None else
                    "original TRAIN600 only" if self._research_normalization_mode == "fresh_full_train600_stats" else
                    "inherited source TRAIN150; unchanged"
                ),
                "normalization_mode": self._research_normalization_mode,
                **({"original_test_targets_outside_DEV22_read": False} if classic_identity is None else {}),
                "original_test_targets_outside_declared_panel_read": False,
                "declared_exposed_validation_count": len(self.validation_case_ids),
                "locked_test_targets_read": False,
                "research_membership_sha256": research_membership_sha,
                "source_classic_subset_manifest_sha256": membership_sha if classic_identity is not None else None,
                "source_checkpoint_sha256": self._research_config["initialization"].get("source_sha256")
                    or self._research_source_payload.get("classic_checkpoint_sha256")
                    or (classic_identity or {}).get("source_binding", {}).get("source_checkpoint_sha256"),
            },
            "objective": {
                **dict(payload.get("objective", {})),
                "exposed_validation_target_scope": self.validation_scope,
                "unexposed_test_targets_read": False,
                "locked_test_targets_read": False,
                "train_dev_are_disjoint": not bool(set(self.training_case_ids) & set(self.validation_case_ids)),
            },
            "training": {
                **dict(payload.get("training", {})),
                "initialization_seed": self.seed,
                "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
                "effective_batch_size": self.effective_batch_size,
                "microbatch_size": self._identity_microbatch_size,
                "primary_queries_per_case": self.primary_queries,
                "total_epochs": self.total_epochs,
                "optimizer": ("fresh AdamW over random compact weights"
                              if _is_round12_scratch(self._research_config) else
                              "fresh AdamW source-terminal groups over the compact U-H parameter migration"
                              if self._round12 is not None else
                              "fresh AdamW formal_joint group over all active U-core parameters"
                              if classic_identity is not None else
                              "AdamW with fresh moments and source-compatible named groups"),
                "optimizer_schedule": {
                    "kind": self._research_config["optimizer"]["schedule"],
                    "local_horizon_epochs": self.total_epochs,
                    "parent_age": _parent_age(self._research_config),
                    "fresh_model_schedule": copy.deepcopy(
                        self._research_config["initialization"].get("fresh_model_schedule")
                    ),
                    "fresh_controller_lambda0": copy.deepcopy(
                        self._research_config["initialization"].get("fresh_controller_lambda0")
                    ),
                    "fresh_controller_lambda0_source": self._research_config[
                        "initialization"
                    ].get("fresh_controller_lambda0_source"),
                },
            },
            "recipe": {
                "schema": RESEARCH_SCHEMA,
                "config_name": self._research_config["name"],
                "source_arm": self._research_config["initialization"]["source_arm"],
                "source_checkpoint": self._research_config["initialization"].get("source_checkpoint"),
                "source_checkpoint_sha256": self._research_config["initialization"].get("source_sha256")
                    or self._research_source_payload.get("classic_checkpoint_sha256")
                    or (classic_identity or {}).get("source_binding", {}).get("source_checkpoint_sha256"),
                "local_horizon": self.total_epochs,
                "training_membership": self._research_membership,
                "normalization_mode": self._research_normalization_mode,
                "fresh_controller_lambda0": copy.deepcopy(
                    self._research_config["initialization"].get("fresh_controller_lambda0")
                ),
                "fresh_controller_lambda0_source": self._research_config[
                    "initialization"
                ].get("fresh_controller_lambda0_source"),
                "exposed_validation_membership": (
                    "classic canonical89 exposed original test panel" if classic_identity is not None else
                    "fixed25_v1 DEV22"
                ),
                "locked_test_targets_read": False,
                "diagnostic_case_limit": self._research_case_limit,
                "execution_options": copy.deepcopy(self._research_config["execution"]["options"]),
                "optimizer_moments": "reset on source initialization; retained only on exact in-run resume",
                "compact_adaptation": (None if self._round12 is None else {
                    "schema": self._round12["schema"],
                    "role": self._round12["role"],
                    "compact_arm": self._round12["compact_arm"],
                    "parameter_migration": copy.deepcopy(getattr(
                        getattr(self.model, "core", self.model), "compact_parameter_migration", None,
                    )),
                    "clock_mapping": copy.deepcopy(self._research_config["initialization"]["clock_mapping"]),
                    "hard_cut_after_local_epoch": self._round12["cut_after_local_epoch"],
                }),
            },
        })
        # Pre-Round12 identities omitted inactive compact metadata entirely.
        if self._round12 is None:
            payload["recipe"].pop("compact_adaptation", None)
        _restore_checkpoint_identity_metadata(self, payload)
        self.sampling_dataset_id = _sampling_dataset_id(payload)
        return payload

    def optimizer_groups(self, model, arm, stage):
        del stage
        if model is not self.model or arm != model.architecture:
            raise ValueError("Research optimizer requested for the wrong bound model.")
        named_parameters = dict(model.named_parameters())
        if self._round12 is None:
            named = {name for name, parameter in named_parameters.items() if parameter.requires_grad}
            allowed_frozen: set[str] = set()
        else:
            # The Round12 hard cut stops policy gradients but keeps the same
            # named AdamW groups and moments, so an exact resume has one stable
            # optimizer contract on both sides of the transition.
            named = set(named_parameters)
            allowed_frozen = _round12_model_parameter_names(model)
        groups = []
        covered: set[str] = set()
        for source in self._research_optimizer_contract:
            names = tuple(source["parameter_names"])
            if not names or not set(names) <= named or covered.intersection(names):
                raise ValueError(f"Source optimizer group {source.get('name')} no longer binds the active model.")
            covered.update(names)
            lr = float(source["schedule"]["final_lr"])
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError("Source group's terminal learning rate must be positive and finite.")
            schedule = (ScheduleSpec(**dict(source["schedule"]))
                        if self._research_normalization_mode == "classic_train600_stats" else ScheduleSpec(
                            peak_lr=lr,
                            warmup_start_lr=lr,
                            warmup_epochs=0,
                            hold_through_epoch=self.total_epochs,
                            total_epochs=self.total_epochs + 1,
                            final_lr=lr,
                        ))
            groups.append(OptimizerGroupSpec(
                str(source["name"]), names, schedule,
                weight_decay=float(source["weight_decay"]),
                betas=tuple(map(float, source.get("betas", (0.9, 0.999)))),
                eps=float(source.get("eps", 1.0e-8)),
            ))
        trainable = {name for name, parameter in named_parameters.items() if parameter.requires_grad}
        if self._round12 is None:
            if covered != trainable:
                raise ValueError(f"Source optimizer groups omit current trainable parameters: {sorted(trainable-covered)}")
        elif covered != trainable | (covered & allowed_frozen):
            raise ValueError(
                "Round12 optimizer contract must include every trainable parameter and may retain only "
                "the declared frozen compatibility policy parameters."
            )
        elif not allowed_frozen <= covered:
            raise ValueError("Round12 optimizer contract lost a frozen compatibility policy parameter.")
        return tuple(groups)

    def _schedule_payload(self):
        return {
            "kind": self._research_config["optimizer"]["schedule"],
            "local_horizon_epochs": self.total_epochs,
            "fresh_model_schedule": copy.deepcopy(
                self._research_config["initialization"].get("fresh_model_schedule")
            ),
            "source_group_terminal_lrs": [float(row["schedule"]["final_lr"])
                                           for row in self._research_optimizer_contract],
        }

    def preparation_summary(self):
        return {
            "status": "prepared_only",
            "protocol": self.protocol,
            "training_case_count": len(self.training_case_ids),
            "validation_case_count": len(self.validation_case_ids),
            "training_case_ids_sha256": _canonical_sha256(list(self.training_case_ids)),
            "validation_case_ids_sha256": _canonical_sha256(list(self.validation_case_ids)),
            "normalization_scope": self._research_normalization_mode,
            "optimizer_updates": 0,
            "training_membership": self._research_membership if not self._research_case_limit else
                f"diagnostic subset of {self._research_membership} ({self._research_case_limit})",
            "validation_scope": ("canonical89 exposed original test panel; unexposed remainder excluded"
                                 if self._classic_provider_identity is not None else
                                 "fixed25_v1 DEV22; locked remainder excluded"),
        }


class ResearchTrainingEngine(TrainingEngine):
    """Use local history epochs with separate sampler and native model clocks."""

    supports_successful_mask_digest = True

    def __init__(self, config: EngineConfig, *, device, parent_age: int, selection=None,
                 fresh_model_schedule: Mapping[str, Any] | None = None,
                 execution_options: Mapping[str, Any] | None = None,
                 sampler_parent_epoch: int | None = None,
                 controller_parent_coordinate: int | None = None,
                 compact_arm: str | None = None,
                 compact_cut_epoch: int = 40,
                 round12_clock_mapping: Mapping[str, Any] | None = None):
        super().__init__(config, device=device, selection=selection)
        self.parent_age = int(parent_age)
        self.fresh_model_schedule = None if fresh_model_schedule is None else dict(fresh_model_schedule)
        self.sampler_parent_epoch = int(
            (0 if self.fresh_model_schedule is not None else self.parent_age)
            if sampler_parent_epoch is None else sampler_parent_epoch
        )
        self.controller_parent_coordinate = int(
            self.parent_age if controller_parent_coordinate is None else controller_parent_coordinate
        )
        self.compact_arm = compact_arm
        self.compact_cut_epoch = int(compact_cut_epoch)
        self.round12_clock_mapping = copy.deepcopy(dict(round12_clock_mapping or {}))
        if compact_arm is not None and compact_arm not in ("common", "singleton", "merge", "soft_only"):
            raise ValueError("Compact arm must be common, singleton, or merge.")
        if self.compact_cut_epoch < 1:
            raise ValueError("Compact hard-cut epoch must be positive.")
        self.nominal_microbatch_cases = int(config.microbatch_cases)
        self.runtime_microbatch_cases = self.nominal_microbatch_cases
        self.nominal_execution_options = _validated_execution_options(
            _EXECUTION_OPTION_DEFAULTS if execution_options is None else execution_options,
        )
        self.runtime_execution_options = copy.deepcopy(self.nominal_execution_options)
        self.local_oom_retry_count = 0
        self._execution_runtime_path: Path | None = None
        self._execution_runtime_dirty = False
        self._last_update_execution_metadata: dict[str, Any] | None = None

    def take_last_update_execution_metadata(self) -> dict[str, Any] | None:
        """Return the completed packing receipt once, without retaining its batches."""
        metadata = self._last_update_execution_metadata
        self._last_update_execution_metadata = None
        return metadata

    def _make_optimizer(self, model, provider, arm, initial_epoch, optimizer_seed, *, group_specs=None):
        if self.compact_arm is None:
            return super()._make_optimizer(
                model, provider, arm, initial_epoch, optimizer_seed, group_specs=group_specs,
            )
        frozen_names = _round12_model_parameter_names(model)
        if not frozen_names:
            return super()._make_optimizer(
                model, provider, arm, initial_epoch, optimizer_seed, group_specs=group_specs,
            )
        named = dict(model.named_parameters())
        prior_flags = {name: named[name].requires_grad for name in frozen_names}
        try:
            for name in frozen_names:
                named[name].requires_grad_(True)
            return super()._make_optimizer(
                model, provider, arm, initial_epoch, optimizer_seed, group_specs=group_specs,
            )
        finally:
            for name, flag in prior_flags.items():
                named[name].requires_grad_(flag)
                if not flag:
                    named[name].grad = None

    def bind_execution_runtime(self, path: str | Path, *, nominal_microbatch_cases: int,
                               default_execution_options: Mapping[str, Any] | None = None) -> None:
        """Load run-local packing state without changing the sealed engine identity."""
        nominal = int(nominal_microbatch_cases)
        if nominal < 1 or nominal > self.config.effective_cases:
            raise ValueError("Nominal research microbatch must fit inside the effective batch.")
        self.nominal_microbatch_cases = nominal
        self.runtime_microbatch_cases = nominal
        self.runtime_execution_options = _validated_execution_options(
            self.nominal_execution_options if default_execution_options is None else default_execution_options,
        )
        self.local_oom_retry_count = 0
        self._execution_runtime_dirty = False
        self._execution_runtime_path = Path(path)
        if not self._execution_runtime_path.exists():
            return
        payload = read_json(self._execution_runtime_path)
        if payload.get("schema") != "honf_research_execution_runtime_v1":
            raise ValueError("Research execution runtime state has an unsupported schema.")
        if payload.get("nominal_microbatch_cases") != nominal:
            raise ValueError("Saved runtime packing differs from the run's nominal microbatch identity.")
        runtime = payload.get("runtime_microbatch_cases")
        retries = payload.get("local_oom_retry_count")
        if (type(runtime) is not int or not 1 <= runtime <= self.config.effective_cases
                or type(retries) is not int or retries < 0):
            raise ValueError("Saved research runtime packing or retry count is malformed.")
        self.runtime_microbatch_cases = runtime
        self.local_oom_retry_count = retries
        saved_options = payload.get("runtime_execution_options")
        if saved_options is not None:
            if not isinstance(saved_options, Mapping):
                raise ValueError("Saved runtime execution options are malformed.")
            self.runtime_execution_options = _validated_execution_options(saved_options)

    def set_runtime_overrides(self, *, provider, microbatch_cases: int | None = None,
                              execution_options: Mapping[str, Any] | None = None) -> None:
        """Apply execution-only choices while preserving all nominal identity fields."""
        changed = False
        if microbatch_cases is not None:
            if type(microbatch_cases) is not int or not 1 <= microbatch_cases <= self.config.effective_cases:
                raise ValueError("Runtime microbatch must fit inside the effective batch.")
            if microbatch_cases != self.runtime_microbatch_cases:
                self.runtime_microbatch_cases = microbatch_cases
                changed = True
        if execution_options is not None:
            resolved = _validated_execution_options(execution_options)
            if resolved != self.runtime_execution_options:
                self.runtime_execution_options = resolved
                changed = True
        # The inherited Thermal provider uses this only as a batch-size guard.
        # Its sealed research identity reads _identity_microbatch_size instead.
        provider.microbatch_size = self.runtime_microbatch_cases
        self._execution_runtime_dirty = self._execution_runtime_dirty or changed

    def _write_execution_runtime(self, *, last_oom: Mapping[str, Any]) -> None:
        if self._execution_runtime_path is None:
            return
        _atomic_json(self._execution_runtime_path, {
            "schema": "honf_research_execution_runtime_v1",
            "nominal_microbatch_cases": self.nominal_microbatch_cases,
            "runtime_microbatch_cases": self.runtime_microbatch_cases,
            "runtime_execution_options": copy.deepcopy(self.runtime_execution_options),
            "local_oom_retry_count": self.local_oom_retry_count,
            "last_oom": dict(last_oom),
            "updated_unix": time.time(),
        })

    def prepare_update_microbatches(self, provider, *, arm: str, phase: str, epoch: int,
                                   update_index: int, update_cases: Sequence[Any],
                                   sampling_dataset_id: str):
        """Use recoverable packing while preserving the exact macro case order."""
        provider.microbatch_size = self.runtime_microbatch_cases
        if self.runtime_microbatch_cases == self.config.microbatch_cases:
            return super().prepare_update_microbatches(
                provider, arm=arm, phase=phase, epoch=epoch, update_index=update_index,
                update_cases=update_cases, sampling_dataset_id=sampling_dataset_id,
            )
        if self.config.sampling_version != SamplingKey.CASE_EPOCH_VERSION:
            raise ValueError("Research runtime repacking requires case_epoch_v1 query keys.")
        batches = []
        for micro_index, start in enumerate(range(0, len(update_cases), self.runtime_microbatch_cases)):
            case_chunk = update_cases[start:start + self.runtime_microbatch_cases]
            key = SamplingKey(
                self.config.seed, self.sampler_epoch(epoch), update_index, micro_index, phase, arm,
                sampling_version=self.config.sampling_version,
                dataset_id=sampling_dataset_id,
            )
            batch = provider.make_batch(case_chunk, key)
            if tuple(batch.case_keys) and list(batch.case_keys) != list(case_chunk):
                raise ValueError("Provider changed the ordered case keys inside a retry microbatch.")
            if not tuple(batch.case_keys):
                from .unified_training import TaskBatch
                batch = TaskBatch(batch.scene_inputs, batch.receivers, batch.targets, batch.auxiliary,
                                  tuple(case_chunk))
            batches.append(batch)
        return batches

    def sampler_epoch(self, fit_epoch: int) -> int:
        if type(fit_epoch) is not int or fit_epoch < 1:
            raise ValueError("Local research epoch must be positive before parent-age mapping.")
        return fit_epoch if self.fresh_model_schedule is not None else self.sampler_parent_epoch + fit_epoch

    def sampler_epoch_contract(self):
        if self.compact_arm is not None:
            return {
                "schema": "research_round12_clock_mapping_v1",
                "source_parent_age": self.parent_age,
                "sampler_parent_epoch": self.sampler_parent_epoch,
                "controller_parent_coordinate": self.controller_parent_coordinate,
                "fit_epoch_to_sampler_epoch": "sampler_parent_epoch + local_experiment_epoch",
                "fit_epoch_to_model_clock": "controller_parent_coordinate + local_experiment_epoch",
                "fit_epoch_to_optimizer_schedule": "local_experiment_epoch; constant source terminal learning rate",
                "parent_joint_update_index": int(self._round12_clock("parent_update_index")),
                "parent_joint_macro_clock": int(self._round12_clock("parent_macro_clock")),
                "macro_update_index_mapping": "zero_based_within_local_epoch; native_macro_clock_continues_from_parent",
                "local_checkpoint_epoch_remains_local": True,
            }
        return {
            "schema": "research_parent_age_mapping_v1",
            "parent_age": self.parent_age,
            "fit_epoch_to_sampler_epoch": ("local_experiment_epoch" if self.fresh_model_schedule is not None
                                           else "parent_age + local_experiment_epoch"),
            "fit_epoch_to_model_clock": (
                "coordinate_start_epoch + round(local_experiment_epoch * coordinate_span_epochs / ramp_epochs)"
                if self.fresh_model_schedule is not None else "parent_age + local_experiment_epoch"
            ),
            "fresh_model_schedule": copy.deepcopy(self.fresh_model_schedule),
            "local_checkpoint_epoch_remains_local": True,
        }

    def _round12_clock(self, field: str) -> int:
        value = getattr(self, "round12_clock_mapping", {}).get(field)
        if type(value) is not int:
            return 0
        return value

    def _run_update(self, *args, epoch: int, **kwargs):
        self._last_update_execution_metadata = None
        capture_successful_mask_digest = kwargs.pop("capture_successful_mask_digest", False)
        if type(capture_successful_mask_digest) is not bool:
            raise TypeError("Successful-batch mask digest control must be boolean.")
        # The optimizer schedule is local and constant. Warm starts continue
        # their native controller clock; fresh full-TRAIN starts run the
        # controller's near-open work ramp on its native absolute coordinate.
        model_epoch = (self.controller_parent_coordinate + int(epoch) if self.fresh_model_schedule is None
                       else _fresh_model_schedule_epoch(
                           {"initialization": {"fresh_model_schedule": self.fresh_model_schedule}}, int(epoch),
                       ))
        if len(args) < 4:
            raise TypeError("Research optimizer updates require model, provider, optimizer, and microbatches.")
        model, provider, optimizer, initial_microbatches = args[:4]
        phase = kwargs.get("phase")
        arm = kwargs.get("arm")
        update_index = kwargs.get("update_index")
        if phase is None or arm is None or type(update_index) is not int:
            raise TypeError("Research optimizer updates require phase, arm, and macro update index.")
        native_core = getattr(model, "core", model)
        soft_runtime = getattr(provider, "soft_organization", None)
        if soft_runtime is not None:
            soft_runtime.before_update(self, model, provider, optimizer, int(epoch), update_index)
        elif _is_round12_scratch(getattr(provider, "_research_config", {"initialization": {}})):
            if self._execution_runtime_path is None:
                raise RuntimeError("Scratch phase transitions require the owned run directory.")
            _scratch_round12_boundary(self._execution_runtime_path.parent, model, provider,
                                      local_epoch=int(epoch), update_index=update_index, device=self.device)
        if self.compact_arm is not None:
            phase_setter = getattr(native_core, "set_compact_phase", None)
            if not callable(phase_setter):
                raise TypeError("Round12 training requires the compact core's local phase API.")
            if (self.compact_arm == "merge" and int(epoch) == self.compact_cut_epoch + 1
                    and not bool(getattr(native_core, "compact_cut_active").detach().cpu())):
                # Normal operation uses `honf_research activate-cut` between
                # the ordinary e40 before-state and the after-state. This
                # fallback makes an explicit continuation from an uncut e40
                # checkpoint apply the declared transition before update 41.
                activate_cut = getattr(native_core, "activate_hard_cut", None)
                if not callable(activate_cut):
                    raise TypeError("Round12 R1 requires the compact core's persistent hard-cut transition.")
                activate_cut()
            # The core requires the hard cut to be activated while its saved
            # local phase is still 40. In particular, an uncut e40 resume must
            # not first advance the persistent phase to 41.
            phase_setter(int(epoch), arm=self.compact_arm)
            if hasattr(provider, "_round12_local_epoch"):
                provider._round12_local_epoch = int(epoch)
        if self._execution_runtime_dirty:
            self._write_execution_runtime(last_oom={
                "status": "execution_only_resume_override",
                "runtime_microbatch_cases": self.runtime_microbatch_cases,
            })
            self._execution_runtime_dirty = False
        original_cases = tuple(case for batch in initial_microbatches for case in batch.case_keys)
        if not original_cases:
            raise ValueError("Research optimizer macro must retain its ordered case keys for exact retry.")
        provider_state_reader = getattr(provider, "training_state_dict", None)
        provider_state_loader = getattr(provider, "load_training_state_dict", None)
        model_state = _cpu_state_snapshot(model.state_dict())
        optimizer_state = _cpu_state_snapshot(optimizer.state_dict())
        rng_state = _cpu_state_snapshot(capture_rng_state())
        provider_state = (_cpu_state_snapshot(provider_state_reader())
                          if callable(provider_state_reader) else None)
        microbatches = initial_microbatches
        attempt = 0
        failed_attempt_microbatches_constructed = 0
        while True:
            try:
                update_result = super()._run_update(
                    model, provider, optimizer, microbatches, *args[4:], epoch=model_epoch, **kwargs,
                )
                if (self.compact_arm == "merge"
                        and 21 <= int(epoch) <= self.compact_cut_epoch):
                    if (not isinstance(update_result, tuple) or not update_result
                            or not isinstance(update_result[0], Mapping)):
                        raise TypeError("Round12 soft-merge updates must return the maintained loss mapping.")
                    missing_structural = {"compact_tie", "compact_capacity"} - set(update_result[0])
                    if missing_structural:
                        raise RuntimeError(
                            "Round12 soft-merge additive losses were not returned for training history: "
                            f"{sorted(missing_structural)}"
                        )
                self._last_update_execution_metadata = {
                    "successful_microbatch_count": len(microbatches),
                    "oom_retry_attempts": attempt,
                    "failed_attempt_microbatches_constructed": failed_attempt_microbatches_constructed,
                    "successful_mask_sha256": (
                        _observed_mask_digest(microbatches) if capture_successful_mask_digest else None
                    ),
                }
                return update_result
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if not _is_cuda_oom(error):
                    raise
                failed_attempt_microbatches_constructed += len(microbatches)
                error_type = type(error).__name__
                error_text = str(error)
                error.__traceback__ = None
                del error
                self._restore_update_boundary(
                    model, optimizer, provider, model_state, optimizer_state, rng_state,
                    provider_state, provider_state_loader,
                )
                current_pack = int(self.runtime_microbatch_cases)
                if (attempt >= 2 or current_pack <= 1
                        or self.config.sampling_version != SamplingKey.CASE_EPOCH_VERSION):
                    self._write_execution_runtime(last_oom={
                        "local_epoch": int(epoch), "update_index": update_index,
                        "attempt": attempt + 1, "error_type": error_type, "error": error_text,
                        "status": "local_retry_exhausted",
                    })
                    raise ResearchLocalOOMExhausted(
                        f"Research optimizer update CUDA OOM after {attempt} local retries at "
                        f"epoch {epoch}, update {update_index}; completed updates were retained."
                    ) from None

                next_pack = max(1, current_pack // 2)
                if next_pack == current_pack:
                    self._write_execution_runtime(last_oom={
                        "local_epoch": int(epoch), "update_index": update_index,
                        "attempt": attempt + 1, "error_type": error_type, "error": error_text,
                        "status": "no_smaller_case_packing",
                    })
                    raise ResearchLocalOOMExhausted(
                        f"Research optimizer update CUDA OOM at case microbatch {current_pack}; "
                        "no smaller packing is available and completed updates were retained."
                    ) from None
                self.runtime_microbatch_cases = next_pack
                provider.microbatch_size = next_pack
                self.local_oom_retry_count += 1
                attempt += 1
                self._write_execution_runtime(last_oom={
                    "local_epoch": int(epoch), "update_index": update_index,
                    "attempt": attempt, "error_type": error_type, "error": error_text,
                    "previous_microbatch_cases": current_pack,
                    "runtime_microbatch_cases": next_pack,
                    "status": "repacked_same_optimizer_macro",
                    "ordered_case_keys": list(map(str, original_cases)),
                })
                dataset_id = getattr(provider, "sampling_dataset_id", None)
                if not isinstance(dataset_id, str) or not dataset_id:
                    raise RuntimeError("Research retry requires the provider's sealed sampling dataset ID.")
                retry_microbatches = self.prepare_update_microbatches(
                    provider, arm=arm, phase=phase, epoch=int(epoch), update_index=update_index,
                    update_cases=original_cases, sampling_dataset_id=dataset_id,
                )
                replayed_cases = tuple(case for batch in retry_microbatches for case in batch.case_keys)
                if replayed_cases != original_cases:
                    raise RuntimeError("OOM retry changed the ordered cases in the optimizer macro.")
                # Batch construction is target-only and keyed, but reset RNG a
                # second time so model-side stochastic layers see the exact
                # pre-update stream on every attempt.
                restore_rng_state(rng_state)
                microbatches = retry_microbatches

    @staticmethod
    def _restore_update_boundary(model, optimizer, provider, model_state, optimizer_state,
                                 rng_state, provider_state, provider_state_loader):
        import gc
        native_core = getattr(model, "core", model)
        for name in ("_joint_entries", "_context_entries", "_budgeted_entries",
                     "_budgeted_observations", "_compact_loss_entries"):
            if isinstance(getattr(native_core, name, None), list):
                setattr(native_core, name, [])
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
        model.load_state_dict(model_state, strict=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        optimizer.load_state_dict(optimizer_state)
        if provider_state is not None and callable(provider_state_loader):
            provider_state_loader(provider_state)
        restore_rng_state(rng_state)


class ResearchLocalOOMExhausted(RuntimeError):
    """A macro-local retry limit was reached without rewinding prior updates."""


def _write_execution_runtime_under_lock(engine: ResearchTrainingEngine, output: Path,
                                        *, last_oom: Mapping[str, Any]) -> None:
    """Persist outer checkpoint-fallback packing while owning the run writer lock."""
    import fcntl
    engine._execution_runtime_path = output / "execution_runtime.json"
    with (output / ".training.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Cannot update research runtime packing while another writer owns the run.") from error
        try:
            _atomic_json(engine._execution_runtime_path, {
                "schema": "honf_research_execution_runtime_v1",
                "nominal_microbatch_cases": engine.nominal_microbatch_cases,
                "runtime_microbatch_cases": engine.runtime_microbatch_cases,
                "runtime_execution_options": copy.deepcopy(engine.runtime_execution_options),
                "local_oom_retry_count": engine.local_oom_retry_count,
                "last_oom": dict(last_oom),
                "updated_unix": time.time(),
            })
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _build_research_provider(config: Mapping[str, Any], model, checkpoint_payload,
                             optimizer_contract: Sequence[Mapping[str, Any]], *,
                             case_ids: Sequence[str] | None, local_horizon: int,
                             diagnostic: bool, case_limit: int | None = None,
                             identity_microbatch: int | None = None,
                             fit_fresh_normalizer: bool = False):
    """Build the ordinary provider through the maintained v2 Thermal task."""
    from channelthermal.data.datasets import H5Normalizer, fit_global_normalizer
    from channelthermal.data.development_split import development_case_ids
    from channelthermal.training.content_hypergraph_task import _read_selected_cases
    from channelthermal.training.content_hypergraph_task_v2 import build_thermal_content_hypergraph_task_v2

    dataset = config["dataset"]
    recipe = read_json(dataset["base_recipe_path"])
    recipe["data_path"] = dataset["data_path"]
    recipe["manifest"] = dataset["manifest_path"]
    if dataset.get("atlas_directory"):
        recipe["atlas_directory"] = dataset["atlas_directory"]
    else:
        recipe.pop("atlas_directory", None)
    supplied_stats = None
    if config["initialization"]["normalization_mode"] in ("inherit_parent_train_stats", "scratch_train600_stats"):
        adapter_state = checkpoint_payload["experiment_identity"]["provider_identity"]["adapter_state"]
        supplied_stats = adapter_state.get("normalization_stats")
        if not isinstance(supplied_stats, Mapping):
            raise ValueError("Round12 source checkpoint is missing inherited TRAIN600 normalization statistics.")
    if supplied_stats is None:
        _, base, _ = build_thermal_content_hypergraph_task_v2(recipe, device="cpu")
    else:
        # The factory accepts saved parent statistics directly and must not fit
        # a replacement normalizer from the full source dataset.
        _, base, _ = build_thermal_content_hypergraph_task_v2(
            recipe, device="cpu", normalization_stats=supplied_stats,
        )
    source_train600 = tuple(case_id for case_id in base.catalog_case_ids
                            if base.case_split_by_id[case_id] == "train")
    fixed_train150 = tuple(base.training_case_ids)
    dev_ids = tuple(development_case_ids(base.manifest, "test"))
    classic_provider_identity = None
    full_train_parent_mode = config["initialization"]["normalization_mode"] in (
        "classic_train600_stats", "inherit_parent_train_stats", "scratch_train600_stats",
    )
    if full_train_parent_mode:
        classic_provider_identity = checkpoint_payload["experiment_identity"]["provider_identity"]
        classic_train = tuple(map(str, classic_provider_identity.get("training_case_ids", ())))
        classic_validation = tuple(map(str, classic_provider_identity.get("validation_case_ids", ())))
        if (len(source_train600) != 600 or len(classic_train) != 600 or len(classic_validation) != 89
                or set(classic_train) != set(source_train600)
                or set(classic_train) & set(classic_validation)
                or any(base.case_split_by_id.get(key) != "train" for key in classic_train)
                or any(base.case_split_by_id.get(key) != "test" for key in classic_validation)):
            raise ValueError("Classic identity does not bind exact disjoint original TRAIN600 and exposed VAL89 IDs.")
        train_ids = classic_train
        dev_ids = classic_validation
    else:
        if len(source_train600) != 600 or len(fixed_train150) != 150 or len(dev_ids) != 22:
            raise ValueError("Thermal membership must resolve to original TRAIN600 and fixed25 TRAIN150/DEV22.")
        if set(source_train600) & set(dev_ids) or any(base.case_split_by_id.get(key) != "test" for key in dev_ids):
            raise ValueError("TRAIN and exposed DEV22 membership is not bound to disjoint source partitions.")
        train_ids = (fixed_train150 if config["training"]["membership"] == "fixed25_train150_dev22"
                     else source_train600)
    expected_full_membership = (
        "scratch_train600_exposed_val89" if _is_round12_scratch(config) else
        "classic_train600_exposed_val89"
        if config["initialization"]["normalization_mode"] == "classic_train600_stats"
        else "parent_train600_exposed_val89"
    )
    if full_train_parent_mode and config["training"].get("membership") != expected_full_membership:
        raise ValueError("Inherited full-data identity may only be used for its declared TRAIN600/VAL89 profile.")
    if case_limit is not None:
        if case_ids is not None or not diagnostic or type(case_limit) is not int or not 1 <= case_limit < len(train_ids):
            raise ValueError("Diagnostic case-limit must select a nonempty strict subset of the configured TRAIN membership.")
        case_ids = tuple(sorted(train_ids)[:case_limit])
    if case_ids is not None:
        selected = tuple(map(str, case_ids))
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(train_ids):
            raise ValueError("Diagnostic case IDs must be a unique subset of the configured training membership.")
        train_ids = selected
    if fit_fresh_normalizer:
        if config["training"]["membership"] != "original_train600_fixed25_dev22" or diagnostic:
            raise ValueError("Fresh normalization is available only for a complete original TRAIN600 fit.")
        normalizer = fit_global_normalizer(dataset["data_path"], source_train600)
        from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
        old = model
        fresh_core = _fresh_core_from_template(
            old.core,
            seed=int(config["training"].get("seed", 0)),
            execution_options=config["execution"]["options"],
            controller_lambda0=config["initialization"]["fresh_controller_lambda0"],
        )
        model = ThermalContentHypergraphAdapterV2(
            normalizer.stats,
            core=fresh_core,
            nx=old.nx,
            ny=old.ny,
            environment_nx=old.environment_nx,
            environment_ny=old.environment_ny,
        ).to(next(old.parameters()).device)
    else:
        adapter_state = checkpoint_payload["experiment_identity"]["provider_identity"]["adapter_state"]
        normalizer = H5Normalizer({key: np.asarray(value, dtype=np.float32)
                                   for key, value in adapter_state["normalization_stats"].items()})

    # The v2 factory has already loaded the full formal TRAIN600/VAL89 panels.
    # Reuse those arrays for Round12 when their identities and inherited stats
    # match exactly, avoiding a second full in-memory dataset materialization.
    reused_factory_cases = False
    if supplied_stats is not None:
        factory_normalizer_key = base.model.state_payload().get("normalizer_key")
        factory_train_ids = tuple(map(str, getattr(base, "training_case_ids", ())))
        factory_validation_ids = tuple(map(str, getattr(base, "validation_case_ids", ())))
        if (factory_normalizer_key == config["round12"].get("expected_parent_normalizer_key", model.normalizer_key)
                and set(factory_train_ids) == set(train_ids)
                and set(factory_validation_ids) == set(dev_ids)
                and len(base.training_cases) == len(factory_train_ids)
                and len(base.validation_cases) == len(factory_validation_ids)):
            train_by_id = dict(zip(factory_train_ids, base.training_cases))
            validation_by_id = dict(zip(factory_validation_ids, base.validation_cases))
            training_cases = tuple(train_by_id[key] for key in train_ids)
            validation_cases = tuple(validation_by_id[key] for key in dev_ids)
            reused_factory_cases = True
    if not reused_factory_cases:
        # The base v2 task may hold a different profile's examples. Drop them
        # before loading exactly the selected IDs through its established reader.
        base.training_cases = ()
        base.validation_cases = ()
        import gc
        gc.collect()
        training_cases = _read_selected_cases(
            dataset["data_path"], train_ids, split="train", normalizer=normalizer, include_grid=False,
        )
        validation_cases = _read_selected_cases(
            dataset["data_path"], dev_ids, split="test", normalizer=normalizer, include_grid=False,
        )
    else:
        base.training_cases = ()
        base.validation_cases = ()
    base.model = base.adapter = model
    base.device = next(model.parameters()).device
    provider = ResearchThermalProvider(
        base,
        model=model,
        config=config,
        train_ids=train_ids,
        validation_ids=dev_ids,
        training_cases=training_cases,
        validation_cases=validation_cases,
        source_payload=checkpoint_payload,
        optimizer_contract=optimizer_contract,
        diagnostic=diagnostic,
        identity_microbatch=identity_microbatch,
        local_horizon=local_horizon,
        classic_provider_identity=classic_provider_identity,
    )
    provider.sampling_dataset_id = _sampling_dataset_id(provider.identity_payload())
    if full_train_parent_mode:
        expected_sampling_id = checkpoint_payload.get("classic_sampling_dataset_id")
        if expected_sampling_id is None:
            expected_sampling_id = checkpoint_payload.get("round12_parent_sampling_dataset_id")
        if expected_sampling_id is None:
            expected_sampling_id = checkpoint_payload.get("experiment_identity", {}).get("sampling_dataset_id")
        if expected_sampling_id and provider.sampling_dataset_id != expected_sampling_id:
            raise ValueError("Inherited full-data provider changed the source TRAIN sampling namespace.")
    return model, provider


def _fresh_core_from_template(source_core, *, seed: int, execution_options: Mapping[str, Any],
                              controller_lambda0: Mapping[str, float]):
    """Create architecture-matched weights and fresh controller state from config alone."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        core = type(source_core).from_config(source_core.get_config())
    core.set_execution_options(**dict(execution_options))
    joint_budget = getattr(core, "joint_budget", None)
    duals = getattr(joint_budget, "duals", None)
    if duals is None or set(duals) != set(controller_lambda0):
        raise ValueError("Fresh model must expose exactly the calibrated M/E/context budget duals.")
    for name, value in controller_lambda0.items():
        dual = duals[name]
        setter = getattr(dual, "set_calibrated_lambda0", None)
        if not callable(setter):
            raise TypeError(f"Fresh {name} controller lacks set_calibrated_lambda0().")
        setter(float(value))
        # A from-config core is newly initialized, so these fields should be
        # empty. Explicitly clear them to make the fresh-state contract robust
        # to architecture config defaults that may seed controller buffers.
        for field in ("macro_step", "window_step_count", "dual_update_count"):
            counter = getattr(dual, field, None)
            if isinstance(counter, torch.Tensor):
                counter.zero_()
        residual_sum = getattr(dual, "window_residual_sum", None)
        if isinstance(residual_sum, torch.Tensor):
            residual_sum.zero_()
    set_clock = getattr(core, "set_training_clock", None)
    if callable(set_clock):
        # e3000 is the native continuous schedule's open boundary, not parent fit age.
        set_clock(3000, 0)
    if hasattr(core, "joint_budget"):
        schedule = core.joint_budget.schedule()
        if schedule.get("absolute_epoch") != 3000 or schedule.get("sparse_age") != 0:
            raise RuntimeError("Fresh native work controller did not initialize at its open boundary.")
    return core


def _execution_options_for_run(run_dir: Path, config: Mapping[str, Any],
                               overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    options = _validated_execution_options(config["execution"]["options"])
    runtime_path = run_dir / "execution_runtime.json"
    if runtime_path.is_file():
        payload = read_json(runtime_path)
        if payload.get("schema") != "honf_research_execution_runtime_v1":
            raise ValueError("Research execution runtime state has an unsupported schema.")
        saved = payload.get("runtime_execution_options")
        if saved is not None:
            options = _validated_execution_options(saved)
    if overrides:
        options.update(dict(overrides))
    return _validated_execution_options(options)


def _runtime_microbatch_from_file(run_dir: Path, nominal: int, effective: int) -> int:
    runtime_path = run_dir / "execution_runtime.json"
    if not runtime_path.is_file():
        return int(nominal)
    payload = read_json(runtime_path)
    if payload.get("schema") != "honf_research_execution_runtime_v1":
        raise ValueError("Research execution runtime state has an unsupported schema.")
    runtime = payload.get("runtime_microbatch_cases", nominal)
    if (payload.get("nominal_microbatch_cases") != nominal or type(runtime) is not int
            or not 1 <= runtime <= effective):
        raise ValueError("Saved runtime packing does not match this run's nominal/effective batch identity.")
    return runtime


def _apply_runtime_execution_options(model, provider, execution_options: Mapping[str, Any]) -> str:
    """Change runtime core options while proving provider identity remains sealed."""
    options = _validated_execution_options(execution_options)
    before = _canonical_sha256(provider.identity_payload())
    core = getattr(model, "core", model)
    setter = getattr(core, "set_execution_options", None)
    if not callable(setter):
        raise TypeError("Loaded continuous core lacks set_execution_options().")
    setter(**options)
    after = _canonical_sha256(provider.identity_payload())
    if before != after:
        raise RuntimeError("Execution-only runtime options changed the sealed provider identity.")
    return after


def _build_provider_for_resume(config, checkpoint_payload, optimizer_contract,
                               case_ids, local_horizon, diagnostic, device,
                               runtime_execution_options: Mapping[str, Any] | None = None):
    """Rebuild from this run's own checkpoint and stored group contract only."""
    model = _model_from_run_payload(checkpoint_payload, device)
    core = getattr(model, "core", model)
    core.set_execution_options(**config["execution"]["options"])
    model, provider = _build_research_provider(
        config,
        model,
        checkpoint_payload,
        optimizer_contract,
        case_ids=case_ids,
        local_horizon=local_horizon,
        diagnostic=diagnostic,
        fit_fresh_normalizer=False,
    )
    provider_identity = checkpoint_payload.get("experiment_identity", {}).get("provider_identity", {})
    adapter_state = provider_identity.get("adapter_state")
    if not isinstance(adapter_state, Mapping):
        raise ValueError("Research checkpoint is missing its original adapter identity state.")
    provider._research_identity_adapter_state = copy.deepcopy(dict(adapter_state))
    model_config = provider_identity.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("Research checkpoint is missing its original model identity config.")
    provider._research_identity_model_config = copy.deepcopy(dict(model_config))
    if runtime_execution_options is not None:
        _apply_runtime_execution_options(model, provider, runtime_execution_options)
    return model, provider


def _engine_for(config, *, device: torch.device, parent_age: int, microbatch: int | None = None,
                diagnostic: bool = False) -> ResearchTrainingEngine:
    training = config["training"]
    horizon = int(training["local_horizon"])
    batch = int(training["effective_batch_size"])
    microbatch_value = int(microbatch or training["microbatch_cases"])
    monitors = (1,) if diagnostic else tuple(range(int(training["monitor_every"]), horizon + 1,
                                                     int(training["monitor_every"])))
    declared_checkpoints = training.get("checkpoint_epochs")
    checkpoints = ((1,) if diagnostic else tuple(int(epoch) for epoch in declared_checkpoints)
                   if isinstance(declared_checkpoints, list) else
                   tuple(range(int(training["checkpoint_every"]), horizon + 1,
                               int(training["checkpoint_every"]))))
    engine_config = EngineConfig(
        seed=int(training.get("seed", 0)),
        microbatch_cases=microbatch_value,
        effective_cases=batch,
        total_epochs=horizon,
        monitor_every=int(training["monitor_every"]),
        monitor_epochs=monitors,
        checkpoint_epochs=checkpoints,
        latest_every=int(training["latest_every"]),
        curve_every=int(training["curve_every"]),
        gradient_clip=1.0,
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        training_mode="joint",
        progress_enabled=bool(training.get("progress_enabled", True)),
    )
    return ResearchTrainingEngine(
        engine_config,
        device=device,
        parent_age=parent_age,
        selection=SelectionPolicy(field_metric="field_score"),
        fresh_model_schedule=(config["initialization"].get("fresh_model_schedule")
                              if config["initialization"]["normalization_mode"] in (
                                  "fresh_full_train600_stats", "classic_train600_stats", "scratch_train600_stats",
                              )
                              else None),
        execution_options=config["execution"]["options"],
        sampler_parent_epoch=config["initialization"].get("clock_mapping", {}).get(
            "sampler_parent_epoch", parent_age,
        ),
        controller_parent_coordinate=config["initialization"].get("clock_mapping", {}).get(
            "controller_parent_coordinate", parent_age,
        ),
        compact_arm=config.get("round12", {}).get("compact_arm"),
        compact_cut_epoch=(40 if _is_softonly(config) else
                           int(config.get("round12", {}).get("cut_after_local_epoch", 40))),
        round12_clock_mapping=config["initialization"].get("clock_mapping"),
    )


def _write_run_manifest(output: Path, *, config: Mapping[str, Any], config_path: Path,
                        source_path: Path, source_sha: str, optimizer_contract,
                        case_ids: Sequence[str] | None, storage_snapshot: Mapping[str, Any],
                        diagnostic: bool, batch: int, microbatch: int,
                        identity_microbatch: int) -> None:
    value = {
        "schema": RESEARCH_SCHEMA,
        "config": copy.deepcopy(dict(config)),
        "config_path": str(config_path),
        "source_checkpoint": None if _is_round12_scratch(config) else str(source_path),
        "initialization_recipe": str(source_path) if _is_round12_scratch(config) else None,
        "source_checkpoint_sha256": None if _is_round12_scratch(config) else source_sha,
        "initialization_recipe_sha256": source_sha if _is_round12_scratch(config) else None,
        "source_alias": config["initialization"].get("source_alias", ""),
        "source_epoch": int(config["initialization"]["source_parent_age"]),
        "optimizer_moments": "fresh on source initialization; exact-run checkpoint on resume",
        "optimizer_schedule_contract": copy.deepcopy(list(optimizer_contract)),
        "run_case_ids": None if case_ids is None else list(map(str, case_ids)),
        "diagnostic_only": bool(diagnostic),
        "diagnostic_notice": "case-limited plumbing qualification; not scientific fit evidence" if diagnostic else None,
        "effective_batch_size": int(batch),
        "microbatch_cases": int(microbatch),
        "identity_microbatch_cases": int(identity_microbatch),
        "storage_snapshot_before_run": dict(storage_snapshot),
        "normalization_mode": config["initialization"]["normalization_mode"],
        "training_membership": config["training"]["membership"],
        "dataset_access": {
            "training": config["training"]["membership"],
            "validation": ("canonical89 exposed original test panel only"
                           if config["training"]["membership"] in (
                               "classic_train600_exposed_val89", "parent_train600_exposed_val89", "scratch_train600_exposed_val89",
                           ) else "fixed25_v1 DEV22 only"),
            "locked_test_targets_read": False,
            "unexposed_original_test_targets_read": False,
            "dataset_copy_created": False,
            "persistent_dataset_cache_created": False,
        },
        "created_unix": time.time(),
    }
    _atomic_json(output / _CONFIG_NAME, value)


def _load_run_manifest(run_dir: str | Path) -> tuple[dict[str, Any], Path]:
    root = Path(run_dir).expanduser().resolve(strict=True)
    value = read_json(root / _CONFIG_NAME)
    if value.get("schema") != RESEARCH_SCHEMA:
        raise ValueError(f"Not a {RESEARCH_SCHEMA} run directory: {root}")
    config = value.get("config")
    if not isinstance(config, dict) or config.get("schema") != RESEARCH_SCHEMA:
        raise ValueError("Research run manifest is missing its self-contained config snapshot.")
    _validate_research_config(config)
    return value, Path(str(value.get("config_path", "research_run.json")))


def _fit_with_retries(engine: ResearchTrainingEngine, model, provider, output: Path, *,
                      identity: Mapping[str, Any], stop_after: int, arm: str,
                      resume_checkpoint: Path | None = None, allow_microbatch_change: bool = False,
                      runtime_microbatch_cases: int | None = None,
                      runtime_execution_options: Mapping[str, Any] | None = None,
                      permitted_devices: Sequence[str] | None = None,
                      allow_horizon_extension: bool = False):
    retries = 0
    active_engine = engine
    current_resume = resume_checkpoint
    original_output = output
    pending_runtime_microbatch_override = runtime_microbatch_cases
    initial_model_state = _cpu_state_snapshot(model.state_dict())
    initial_rng_state = _cpu_state_snapshot(capture_rng_state())
    provider_state_reader = getattr(provider, "training_state_dict", None)
    initial_provider_state = _cpu_state_snapshot(provider_state_reader()) if callable(provider_state_reader) else None
    while True:
        if isinstance(active_engine, ResearchTrainingEngine):
            active_engine.bind_execution_runtime(
                output / "execution_runtime.json",
                nominal_microbatch_cases=int(getattr(
                    provider, "_identity_microbatch_size", active_engine.config.microbatch_cases,
                )),
                default_execution_options=getattr(
                    active_engine, "nominal_execution_options", active_engine.runtime_execution_options,
                ),
            )
            active_engine.set_runtime_overrides(
                provider=provider,
                microbatch_cases=pending_runtime_microbatch_override,
                execution_options=runtime_execution_options,
            )
            # Apply a CLI pack once. If an outer retry persists a smaller pack,
            # the reloaded runtime state must take precedence on the next loop.
            pending_runtime_microbatch_override = None
        previous_handlers = _install_clean_stop_handlers(output)
        try:
            result = active_engine.fit(
                model, provider, output,
                identity=identity,
                arm=arm,
                stop_after=int(stop_after),
                resume_checkpoint=current_resume,
                allow_microbatch_change=allow_microbatch_change,
                allow_horizon_extension=allow_horizon_extension,
            )
            if _is_softonly(provider._research_config):
                payload, _ = _read_run_checkpoint(output, "latest")
                if (int(payload["epoch"]) == int(provider._research_config["training"]["local_horizon"])
                        == active_engine.config.total_epochs):
                    from .thermal_soft_reviews import run_final_validation
                    runtime = provider.soft_organization
                    remaining = max(0.0, 14400.0 - runtime.state["review_elapsed_seconds"])
                    result["softonly_terminal_validation"] = run_final_validation(
                        output, str(active_engine.device), budget_seconds=remaining)
            return result
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if isinstance(error, ResearchLocalOOMExhausted):
                raise
            if not _is_cuda_oom(error):
                raise
            if retries >= 2:
                _write_oom_receipt(output, retries, error, status="retry_limit_reached")
                raise RuntimeError("CUDA OOM retry limit (two retries) was exhausted.") from error
            retries += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            layout = RunLayout(output)
            current_latest = layout.read_path("latest_model.pt")
            next_device = active_engine.device
            nominal_microbatch = int(getattr(
                active_engine, "nominal_microbatch_cases", active_engine.config.microbatch_cases,
            ))
            current_runtime_microbatch = int(getattr(
                active_engine, "runtime_microbatch_cases", active_engine.config.microbatch_cases,
            ))
            current_runtime_execution_options = copy.deepcopy(getattr(
                active_engine, "runtime_execution_options", _EXECUTION_OPTION_DEFAULTS,
            ))
            next_microbatch = max(1, current_runtime_microbatch // 2)
            if permitted_devices:
                for candidate in permitted_devices:
                    try:
                        candidate_device, _ = resolve_device(candidate, permitted_devices=permitted_devices)
                    except (ValueError, RuntimeError):
                        continue
                    if candidate_device != active_engine.device:
                        next_device = candidate_device
                        break

            if current_latest.is_file():
                old_microbatch = nominal_microbatch
                old_runtime_microbatch = current_runtime_microbatch
                old_device = active_engine.device
                if next_microbatch == old_runtime_microbatch and next_device == active_engine.device:
                    _write_oom_receipt(output, retries, error, status="no_smaller_microbatch_or_permitted_device",
                                       microbatch=old_runtime_microbatch, device=str(active_engine.device))
                    raise RuntimeError("CUDA OOM at microbatch one with no permitted alternate device.") from error
                active_engine = _engine_for(
                    provider._research_config,
                    device=next_device,
                    parent_age=active_engine.parent_age,
                    microbatch=nominal_microbatch,
                    diagnostic=provider._research_case_limit is not None,
                )
                _extend_execution_horizon(active_engine, engine.config.total_epochs, provider)
                active_engine.bind_execution_runtime(
                    output / "execution_runtime.json", nominal_microbatch_cases=nominal_microbatch,
                )
                active_engine.runtime_microbatch_cases = next_microbatch
                active_engine.runtime_execution_options = current_runtime_execution_options
                model.to(next_device)
                provider.device = next_device
                current_resume = current_latest
                allow_microbatch_change = False
                _write_execution_runtime_under_lock(active_engine, output, last_oom={
                    "status": "epoch_checkpoint_retry",
                    "retry": retries,
                    "previous_runtime_microbatch_cases": old_runtime_microbatch,
                    "runtime_microbatch_cases": next_microbatch,
                    "nominal_microbatch_cases": nominal_microbatch,
                    "previous_device": str(old_device),
                    "device": str(next_device),
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                _write_oom_receipt(
                    output, retries, error,
                    status="resuming_last_complete_epoch",
                    nominal_microbatch=old_microbatch,
                    previous_runtime_microbatch=old_runtime_microbatch,
                    runtime_microbatch=next_microbatch,
                    previous_device=str(old_device),
                    device=str(next_device),
                )
                continue

            # The first complete epoch has not been checkpointed. Preserve the
            # failed attempt, then restore the exact initialization before retry.
            failed = original_output.with_name(f"{original_output.name}.oom_attempt_{retries:02d}")
            _move_failed_attempt(output, failed)
            model.load_state_dict(initial_model_state, strict=True)
            restore_rng_state(initial_rng_state)
            provider_state_loader = getattr(provider, "load_training_state_dict", None)
            if initial_provider_state is not None and callable(provider_state_loader):
                provider_state_loader(initial_provider_state)
            model.to(next_device)
            provider.device = next_device
            next_microbatch = min(next_microbatch, nominal_microbatch, active_engine.config.effective_cases)
            previous_runtime_microbatch = current_runtime_microbatch
            previous_retry_count = int(getattr(active_engine, "local_oom_retry_count", 0))
            active_engine = _engine_for(
                provider._research_config,
                device=next_device,
                parent_age=active_engine.parent_age,
                microbatch=nominal_microbatch,
                diagnostic=provider._research_case_limit is not None,
            )
            active_engine.nominal_microbatch_cases = nominal_microbatch
            active_engine.runtime_microbatch_cases = next_microbatch
            active_engine.runtime_execution_options = current_runtime_execution_options
            active_engine.local_oom_retry_count = previous_retry_count
            output.mkdir(parents=True, exist_ok=False)
            failed_manifest = failed / _CONFIG_NAME
            if failed_manifest.is_file():
                manifest = read_json(failed_manifest)
                _atomic_json(output / _CONFIG_NAME, manifest)
            _write_execution_runtime_under_lock(active_engine, output, last_oom={
                "status": "restarted_from_initialization_after_oom_outside_update",
                "retry": retries,
                "previous_runtime_microbatch_cases": previous_runtime_microbatch,
                "runtime_microbatch_cases": next_microbatch,
                "nominal_microbatch_cases": nominal_microbatch,
                "device": str(next_device),
                "error_type": type(error).__name__,
                "error": str(error),
            })
            current_resume = None
            allow_microbatch_change = False
            _write_oom_receipt(
                failed, retries, error,
                status="restarted_from_initialization_after_first_epoch_oom",
                microbatch=next_microbatch,
                device=str(next_device),
            )
        finally:
            _restore_handlers(previous_handlers)

def _is_cuda_oom(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "cuda out of memory" in str(error).lower()


def _write_oom_receipt(output: Path, retry: int, error: BaseException, *, status: str, **extra):
    _atomic_json(output / f"oom_retry_{retry:02d}.json", {
        "schema": "ordinary_cuda_oom_retry_v1",
        "status": status,
        "retry": int(retry),
        "error_type": type(error).__name__,
        "error": str(error),
        "time_unix": time.time(),
        "restore_boundary": "last complete epoch checkpoint; deterministic case-epoch sampling replays an incomplete epoch",
        **extra,
    })


def _move_failed_attempt(source: Path, destination: Path):
    import fcntl
    if destination.exists():
        raise FileExistsError(destination)
    lock_path = source / ".training.lock"
    with lock_path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Cannot relocate OOM attempt while another writer holds its run lock.") from error
        os.replace(source, destination)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _experiment_identity(config: Mapping[str, Any], provider: ResearchThermalProvider,
                         source_path: str | Path, source_sha: str, *, diagnostic: bool) -> dict[str, Any]:
    provider_identity = dict(provider.identity_payload())
    identity = {
        "schema": RESEARCH_SCHEMA,
        "family": TASK_SCHEMA,
        "workflow": "ordinary_honf_research_train_v1",
        "model_family": "unified_continuous_joint_native_v1",
        "dataset": "thermal",
        "config_name": config["name"],
        "config_sha256": _canonical_sha256(config),
        "training_membership": config["training"]["membership"],
        "normalization_mode": config["initialization"]["normalization_mode"],
        "source_checkpoint": None if _is_round12_scratch(config) else str(source_path),
        "initialization_recipe": str(source_path) if _is_round12_scratch(config) else None,
        "source_checkpoint_sha256": None if _is_round12_scratch(config) else source_sha,
        "initialization_recipe_sha256": source_sha if _is_round12_scratch(config) else None,
        "source_parent_age": int(config["initialization"]["source_parent_age"]),
        "training_parent_age": _parent_age(config),
        "source_model_initialization": (
            "fresh_random_compact_core; no checkpoint read; no pretrained weights or optimizer state"
            if _is_round12_scratch(config) else
            "fresh_random_core_from_declared_architecture; source weights not loaded"
            if _parent_age(config) == 0 and int(config["initialization"]["source_parent_age"]) > 0
            else "model_weights_only; optimizer moments reset"
        ),
        "classic_reference_identity_path": config["initialization"].get("source_identity_path"),
        "classic_checkpoint_weights_deserialized": (
            False if config["initialization"].get("source_kind") == "classic_identity_metadata_only" else None
        ),
        "fresh_architecture_config": (
            "classic saved actual A2-D core config; init_seed=0; UnifiedJointCore constructed from config"
            if config["initialization"].get("source_kind") == "classic_identity_metadata_only" else None
        ),
        "fresh_controller_initialization": (
            {
                "kind": "calibrated_lambda0_hyperparameters_with_reset_dual_state",
                "lambda0": copy.deepcopy(config["initialization"]["fresh_controller_lambda0"]),
                "calibration_source": config["initialization"]["fresh_controller_lambda0_source"],
                "optimizer_moments_and_mature_dual_state_loaded": False,
            }
            if config["initialization"]["normalization_mode"] in (
                "fresh_full_train600_stats", "classic_train600_stats", "scratch_train600_stats",
            )
            else None
        ),
        "recipe": dict(provider_identity["recipe"]),
        "execution_options": copy.deepcopy(config["execution"]["options"]),
        "diagnostic_only": diagnostic,
        "diagnostic_case_limit": len(provider.training_case_ids) if diagnostic else None,
    }
    if not config.get("round12"):
        # Preserve the original U schema under the strict resume check.
        identity.pop("initialization_recipe")
        identity.pop("initialization_recipe_sha256")
    return identity


def train(config_path: str | Path, *, device: str, output: str | Path,
          epochs: int | None = None, devices: Sequence[str] | None = None,
          case_limit: int | None = None, case_ids: Sequence[str] | None = None,
          diagnostic_only: bool = False) -> dict[str, Any]:
    config, resolved_config_path = load_research_config(config_path)
    training = config["training"]
    stop_after = int(epochs or training["first_segment_epochs"])
    if not 1 <= stop_after <= int(training["local_horizon"]):
        raise ValueError("Requested local segment must lie inside the configured local horizon.")
    if case_limit is not None and case_ids is not None:
        raise ValueError("Choose one diagnostic membership form: --case-limit or --case-id.")
    if diagnostic_only != (case_limit is not None or case_ids is not None):
        raise ValueError("A case-limited fit requires --diagnostic-only, and diagnostics require an explicit subset.")
    if case_ids is not None:
        case_ids = tuple(map(str, case_ids))
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("Diagnostic --case-id values must be a nonempty unique list.")
    if (config["initialization"]["normalization_mode"] in ("fresh_full_train600_stats", "classic_train600_stats", "scratch_train600_stats")
            and diagnostic_only):
        raise ValueError("Fresh full-TRAIN profiles are available only for a full-membership fit.")

    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists() and (_is_round12_scratch(config) or not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"New research fits require an empty output directory: {output_path}")
    source_path = _initialization_path(config)
    source_size = (64 * 1024**2 if _is_round12_scratch(config) else source_path.stat().st_size)
    estimate = _estimate_run_bytes(config, epochs=stop_after, source_size=source_size,
                                   diagnostic=diagnostic_only)
    peak_estimate = 2 * source_size + 256 * 1024**2
    storage_snapshot = verify_storage(config, output=output_path, estimated_bytes=estimate,
                                      peak_temporary_bytes=peak_estimate)
    Path(storage_snapshot["output_root"]).mkdir(parents=True, exist_ok=True)
    permitted = _permitted_order(device, devices)
    selected_device, device_info = resolve_device(permitted[0], permitted_devices=permitted)
    model, source_payload, source_path, source_sha = _source_components(config, selected_device)
    optimizer_contract = (
        source_payload.get("round12_optimizer_schedule_contract")
        if config["initialization"]["normalization_mode"] == "inherit_parent_train_stats"
        else source_payload["experiment_identity"].get(
            "optimizer_schedule_contract", source_payload.get("optimizer_schedule_contract"),
        )
    )
    if not isinstance(optimizer_contract, Sequence) or not optimizer_contract:
        raise ValueError("Source checkpoint lacks its named optimizer group schedule contract.")
    if config["initialization"]["normalization_mode"] in ("classic_train600_stats", "scratch_train600_stats"):
        classic_group = copy.deepcopy(optimizer_contract[0])
        classic_group["parameter_names"] = [name for name, parameter in model.named_parameters()
                                             if parameter.requires_grad]
        optimizer_contract = [classic_group]

    runtime_config = copy.deepcopy(config)
    if case_limit is not None:
        runtime_config["training"]["effective_batch_size"] = min(
            int(training["effective_batch_size"]), case_limit,
        )
    elif case_ids is not None:
        runtime_config["training"]["effective_batch_size"] = min(
            int(training["effective_batch_size"]), len(case_ids),
        )
    runtime_config["training"]["microbatch_cases"] = min(
        int(training["microbatch_cases"]), int(runtime_config["training"]["effective_batch_size"]),
    )
    model, provider = _build_research_provider(
        runtime_config,
        model,
        source_payload,
        optimizer_contract,
        case_ids=case_ids,
        case_limit=case_limit,
        local_horizon=int(training["local_horizon"]),
        diagnostic=diagnostic_only,
        identity_microbatch=int(runtime_config["training"]["microbatch_cases"]),
        fit_fresh_normalizer=(
            config["initialization"]["normalization_mode"] == "fresh_full_train600_stats"
        ),
    )
    output_path.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(
        output_path,
        config=config,
        config_path=resolved_config_path,
        source_path=source_path,
        source_sha=source_sha,
        optimizer_contract=optimizer_contract,
        case_ids=provider.training_case_ids if diagnostic_only else None,
        storage_snapshot=storage_snapshot,
        diagnostic=diagnostic_only,
        batch=provider.effective_batch_size,
        microbatch=provider.microbatch_size,
        identity_microbatch=provider._identity_microbatch_size,
    )
    engine = _engine_for(
        runtime_config,
        device=selected_device,
        parent_age=_parent_age(config),
        microbatch=provider.microbatch_size,
        diagnostic=diagnostic_only,
    )
    identity = _experiment_identity(config, provider, source_path, source_sha, diagnostic=diagnostic_only)
    print(json.dumps({
        "event": "research_run_prepared",
        "output": str(output_path),
        "source_checkpoint": None if _is_round12_scratch(config) else str(source_path),
        "initialization_recipe": str(source_path) if _is_round12_scratch(config) else None,
        "source_sha256": source_sha,
        "source_epoch": config["initialization"]["source_parent_age"],
        "selected_device": device_info,
        "training_membership": config["training"]["membership"],
        "train_cases": len(provider.training_case_ids),
        "validation_cases": len(provider.validation_case_ids),
        "effective_batch": provider.effective_batch_size,
        "microbatch": provider.microbatch_size,
        "local_stop_after": stop_after,
        "optimizer_updates_per_epoch": math.ceil(
            len(provider.training_case_ids) / provider.effective_batch_size,
        ),
        "storage": storage_snapshot,
    }, sort_keys=True), flush=True)
    return _fit_with_retries(
        engine,
        model,
        provider,
        output_path,
        identity=identity,
        stop_after=stop_after,
        arm=model.architecture,
        permitted_devices=permitted,
    )

def _extend_execution_horizon(engine: ResearchTrainingEngine, horizon: int, provider=None) -> None:
    """Extend execution/review clocks while leaving provider and LR schedules sealed."""
    if provider is not None and hasattr(provider, "_round6_sampling_epoch_limit"):
        provider._round6_sampling_epoch_limit = _parent_age(provider._research_config) + horizon
    if horizon <= engine.config.total_epochs:
        return
    if horizon % 100:
        raise ValueError("Research horizon extensions must end on a 100-epoch milestone.")
    checkpoints = tuple(range(100, horizon + 1, 100))
    engine.config = replace(engine.config, total_epochs=horizon,
        monitor_epochs=tuple(range(engine.config.monitor_every, horizon + 1, engine.config.monitor_every)),
        checkpoint_epochs=checkpoints)


def resume(run_dir: str | Path, *, device: str, additional_epochs: int | None = None,
           stop_after_epoch: int | None = None, devices: Sequence[str] | None = None,
           microbatch_cases: int | None = None, query_tile_size: int | None = None,
           checkpoint: str | None = None, backend: str | None = None,
           extend_horizon: bool = False) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve(strict=True)
    run_manifest, _ = _load_run_manifest(run)
    config = copy.deepcopy(run_manifest["config"])
    permitted = _permitted_order(device, devices)
    selected_device, device_info = resolve_device(permitted[0], permitted_devices=permitted)
    storage_snapshot = verify_storage(config, output=run, estimated_bytes=0)
    payload, latest_path = _read_run_checkpoint(run, "latest")
    local_horizon = int(config["training"]["local_horizon"])
    current_epoch = int(payload["epoch"])
    execution_horizon = int(payload["experiment_identity"]["engine_config"]["total_epochs"])
    requested_stop = (current_epoch + additional_epochs if type(additional_epochs) is int else stop_after_epoch)
    if extend_horizon:
        if (not _is_softonly(config) or type(requested_stop) is not int
                or requested_stop % 100 or requested_stop <= execution_horizon or current_epoch != execution_horizon):
            raise ValueError("Explicit horizon extension requires a terminal soft-only checkpoint and a larger stop.")
        execution_horizon = requested_stop
    stop_after = _resolve_resume_stop(
        current_epoch=current_epoch,
        local_horizon=execution_horizon,
        additional_epochs=additional_epochs,
        stop_after_epoch=stop_after_epoch,
    )

    option_overrides = {}
    if backend is not None:
        option_overrides["backend"] = backend
    if query_tile_size is not None:
        option_overrides["query_tile_size"] = query_tile_size
    if checkpoint is not None:
        option_overrides["checkpoint"] = checkpoint
    runtime_execution_options = _execution_options_for_run(run, config, option_overrides)

    runtime_config = copy.deepcopy(config)
    runtime_config["training"]["effective_batch_size"] = int(
        run_manifest.get("effective_batch_size", config["training"]["effective_batch_size"]),
    )
    runtime_config["training"]["microbatch_cases"] = int(
        run_manifest.get("microbatch_cases", config["training"]["microbatch_cases"]),
    )
    if (microbatch_cases is not None
            and (type(microbatch_cases) is not int
                 or not 1 <= microbatch_cases <= runtime_config["training"]["effective_batch_size"])):
        raise ValueError("--microbatch-cases must fit inside the run's effective batch.")
    diagnostic = bool(run_manifest.get("diagnostic_only"))
    optimizer_contract = run_manifest.get("optimizer_schedule_contract")
    if not isinstance(optimizer_contract, list) or not optimizer_contract:
        raise ValueError("The run manifest is missing its self-contained optimizer group contract.")
    model, provider = _build_provider_for_resume(
        runtime_config,
        payload,
        optimizer_contract,
        run_manifest.get("run_case_ids"),
        local_horizon,
        diagnostic,
        selected_device,
        runtime_execution_options=runtime_execution_options,
    )
    provider._identity_microbatch_size = int(
        run_manifest.get("identity_microbatch_cases", config["training"]["microbatch_cases"]),
    )
    engine = _engine_for(
        runtime_config,
        device=selected_device,
        parent_age=_parent_age(config),
        microbatch=runtime_config["training"]["microbatch_cases"],
        diagnostic=diagnostic,
    )
    _extend_execution_horizon(engine, execution_horizon, provider)
    identity = _experiment_identity(
        config,
        provider,
        run_manifest.get("initialization_recipe") or run_manifest["source_checkpoint"],
        run_manifest.get("initialization_recipe_sha256") or run_manifest["source_checkpoint_sha256"],
        diagnostic=diagnostic,
    )
    if extend_horizon:
        from .unified_training import _resume_identity_amendment
        sealed, _ = _round12_sealed_identity(config, model, provider, engine,
            run_manifest["initialization_recipe"], run_manifest["initialization_recipe_sha256"])
        amendment = _resume_identity_amendment(payload["experiment_identity"], sealed,
                                              allow_horizon_extension=True)
        # Preserve the original terminal state and manifest; fit records the amendment
        # and restores all training state through its ordinary strict resume path.
        import fcntl
        with (run / ".training.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            milestone, milestone_path = _read_run_checkpoint(run, f"epoch_{current_epoch:04d}_model.pt")
            if int(milestone["epoch"]) != current_epoch:
                raise ValueError("The original terminal milestone must be retained before extension.")
            _atomic_json(RunLayout(run).write_path("horizon_extension.json", category="configs"), {
                **amendment, "source_checkpoint": str(latest_path), "source_epoch": current_epoch,
                "source_checkpoint_sha256": sha256_file(latest_path),
                "retained_terminal_milestone": str(milestone_path),
                "retained_terminal_milestone_sha256": sha256_file(milestone_path),
                "original_config": config, "storage": verify_storage(config, output=run,
                    estimated_bytes=2 * 1024**3, peak_temporary_bytes=512 * 1024**2),
                "created_unix": time.time(),
            })
    print(json.dumps({
        "event": "research_resume_prepared",
        "run": str(run),
        "latest_checkpoint": str(latest_path),
        "from_epoch": current_epoch,
        "stop_after": stop_after,
        "nominal_microbatch_cases": provider._identity_microbatch_size,
        "runtime_microbatch_cases": (microbatch_cases if microbatch_cases is not None else
            _runtime_microbatch_from_file(run, provider._identity_microbatch_size,
                                          provider.effective_batch_size)),
        "runtime_execution_options": runtime_execution_options,
        "device": device_info,
        "training_membership": config["training"]["membership"],
        "train_cases": len(provider.training_case_ids),
        "storage": storage_snapshot,
    }, sort_keys=True), flush=True)
    return _fit_with_retries(
        engine,
        model,
        provider,
        run,
        identity=identity,
        stop_after=stop_after,
        arm=model.architecture,
        resume_checkpoint=latest_path,
        runtime_microbatch_cases=microbatch_cases,
        runtime_execution_options=(runtime_execution_options if option_overrides else None),
        permitted_devices=permitted,
        allow_horizon_extension=extend_horizon,
    )


def activate_round12_cut(run_dir: str | Path) -> dict[str, Any]:
    """Persist the R1 hard cut between its saved before/after local-40 states."""
    import fcntl

    run = Path(run_dir).expanduser().resolve(strict=True)
    run_manifest, _ = _load_run_manifest(run)
    config = run_manifest["config"]
    round12 = config.get("round12", {})
    if (config["initialization"].get("normalization_mode") != "inherit_parent_train_stats"
            or round12.get("role") != "R1" or round12.get("compact_arm") != "merge"):
        raise ValueError("Hard-cut activation applies only to a Round12 R1 merge run.")
    if round12.get("cut_after_local_epoch") != 40:
        raise ValueError("Run does not declare the Round12 local-40 cut boundary.")
    latest_path = resolve_checkpoint(run, "latest").resolve()
    if not latest_path.is_relative_to(run):
        raise ValueError("Latest checkpoint is outside the declared research run.")
    before_path = RunLayout(run).write_path("epoch_0040_model.pt")
    after_path = RunLayout(run).write_path("epoch_0040_after_cut_model.pt")
    if after_path.exists():
        raise FileExistsError(f"Round12 after-cut checkpoint already exists: {after_path}")

    from .unified_training import _atomic_torch_save

    size = latest_path.stat().st_size
    storage = verify_storage(config, output=run, estimated_bytes=size,
                             peak_temporary_bytes=2 * size)
    lock_path = run / ".training.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Cannot activate the cut while another writer owns the R1 run.") from error
        try:
            payload, selected_path = _read_run_checkpoint(run, "latest")
            if selected_path != latest_path:
                raise RuntimeError("R1 latest checkpoint changed while the cut command waited for its lock.")
            if int(payload.get("epoch", -1)) != 40:
                raise ValueError("Round12 R1 cut activation requires the ordinary local-40 checkpoint.")
            before_path = before_path.resolve(strict=True)
            if not before_path.is_relative_to(run) or not before_path.is_file():
                raise FileNotFoundError("The ordinary pre-cut epoch-40 milestone is missing.")
            before_payload, _ = _read_run_checkpoint(run, "epoch_0040_model.pt")
            if int(before_payload.get("epoch", -1)) != 40:
                raise ValueError("Saved pre-cut milestone is not local epoch 40.")
            provider_state = payload.get("provider_training_state")
            if (not isinstance(provider_state, Mapping)
                    or provider_state.get("schema") != "thermal_compact_round12_provider_state_v1"):
                raise ValueError("R1 e40 checkpoint lacks its explicit compact provider state.")
            compact_state = provider_state.get("compact_state")
            if (not isinstance(compact_state, Mapping)
                    or compact_state.get("schema") != "compact_unified_groups_state_v1"
                    or int(compact_state.get("local_epoch", -1)) != 40
                    or compact_state.get("arm") != "merge"
                    or bool(compact_state.get("cut_active"))):
                raise ValueError("R1 e40 latest state is not the expected pre-cut merge state.")
            model = _model_from_run_payload(payload, torch.device("cpu"))
            core = getattr(model, "core", model)
            setter = getattr(core, "set_compact_phase", None)
            activate = getattr(core, "activate_hard_cut", None)
            exporter = getattr(core, "export_compact_state", None)
            if not callable(setter) or not callable(activate) or not callable(exporter):
                raise TypeError("R1 compact core is missing its persisted hard-cut API.")
            setter(40, arm="merge")
            activate()
            changed = copy.deepcopy(dict(payload))
            changed["model_state_dict"] = _cpu_state_snapshot(model.state_dict())
            changed["provider_training_state"] = copy.deepcopy(dict(provider_state))
            changed["provider_training_state"]["compact_state"] = copy.deepcopy(exporter())
            active_state = changed["provider_training_state"]["compact_state"]
            if not active_state.get("cut_active") or active_state.get("last_topology") is not None:
                raise RuntimeError("Hard-cut export must mark the cut active and clear pre-cut topology.")
            _atomic_torch_save(after_path, changed)
            _atomic_torch_save(RunLayout(run).write_path("latest_model.pt"), changed)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {
        "status": "round12_hard_cut_activated",
        "run": str(run),
        "before_checkpoint": str(before_path),
        "after_checkpoint": str(after_path),
        "latest_checkpoint": str(resolve_checkpoint(run, "latest")),
        "epoch": 40,
        "optimizer_group_contract_preserved": True,
        "storage": storage,
    }


def _scratch_round12_boundary(run: Path, model, provider, *, local_epoch: int,
                              update_index: int, device: torch.device) -> None:
    """Automatic algorithm boundaries within one new scratch fit, never a fork."""
    if update_index != 0 or local_epoch not in (21, 41):
        return
    config = provider._research_config
    core = model.core
    if local_epoch == 21:
        calibration_path = run / "round12_calibration.json"
        if calibration_path.exists():
            record = read_json(calibration_path)
            if record.get("source_common_local_epoch") != 20:
                raise ValueError("Scratch calibration must belong to this run's warmup20.")
            coefficients = {name: record["terms"][name]["coefficient"] for name in ("tie", "capacity")}
        else:
            payload, _ = _read_run_checkpoint(run, "latest")
            if int(payload.get("epoch", -1)) != 20:
                raise ValueError("Scratch calibration needs this new run's local20 checkpoint.")
            calibrated = _round12_calibrate_structural_losses(run, payload, config, device)
            record, coefficients = calibrated["record"], calibrated["coefficients"]
            _atomic_json(calibration_path, record)
        core.set_compact_loss_coefficients(**coefficients)
    elif not bool(core.compact_cut_active):
        # TrainingEngine already owns the writer lock. Preserve the ordinary
        # e40 checkpoint and its RNG/Adam/history/clocks; change cut state only.
        from .unified_training import _atomic_torch_save
        payload, _ = _read_run_checkpoint(run, "latest")
        if int(payload.get("epoch", -1)) != 40 or int(core.compact_local_epoch) != 40:
            raise ValueError("Scratch hard cut needs the completed pre-cut local40 state.")
        before = RunLayout(run).read_path("epoch_0040_model.pt")
        if not before.is_file():
            raise FileNotFoundError("Scratch hard cut requires the retained pre-cut milestone.")
        after = RunLayout(run).write_path("epoch_0040_after_cut_model.pt")
        if after.exists():
            raise FileExistsError("Scratch after-cut checkpoint already exists; refusing to overwrite it.")
        core.activate_hard_cut()
        changed = copy.deepcopy(dict(payload))
        changed["model_state_dict"] = _cpu_state_snapshot(model.state_dict())
        changed["provider_training_state"]["compact_state"] = copy.deepcopy(core.export_compact_state())
        _atomic_torch_save(after, changed)
        _atomic_torch_save(RunLayout(run).write_path("latest_model.pt"), changed)


def _round12_new_parameter_names(model) -> tuple[str, ...]:
    core = getattr(model, "core", model)
    migration = getattr(core, "compact_parameter_migration", None)
    if not isinstance(migration, Mapping):
        raise TypeError("Round12 compact core omitted its named parameter migration contract.")
    declared = migration.get("new_names")
    if not isinstance(declared, Sequence) or not declared:
        raise ValueError("Round12 compact migration must list its newly introduced parameter names.")
    model_names = tuple(dict(model.named_parameters()))
    resolved = []
    for core_name in map(str, declared):
        if core_name in model_names:
            resolved.append(core_name)
            continue
        matches = [name for name in model_names if name.endswith("." + core_name)]
        if len(matches) != 1:
            raise ValueError(f"New compact parameter {core_name!r} does not map uniquely into the Thermal model.")
        resolved.append(matches[0])
    if len(set(resolved)) != len(resolved):
        raise ValueError("Compact migration maps multiple new core names onto one adapter parameter.")
    return tuple(sorted(resolved))


def _gradient_norm_by_name(gradients: Mapping[str, torch.Tensor]) -> float:
    values = [value.detach().double().norm() for value in gradients.values() if value.numel()]
    if not values:
        return 0.0
    norm = float(torch.stack(values).norm().cpu())
    return norm if math.isfinite(norm) else float("nan")


def _round12_term_gradient_calibration(native_gradients, term_gradients, scale):
    participating = set(term_gradients)
    native_norm = _gradient_norm_by_name({
        name: gradient for name, gradient in native_gradients.items() if name in participating
    })
    term_norm = _gradient_norm_by_name(term_gradients)
    coefficient = (float(scale) * native_norm / term_norm
                   if math.isfinite(native_norm) and native_norm > 0
                   and math.isfinite(term_norm) and term_norm > 0 else 0.0)
    if not math.isfinite(coefficient) or coefficient < 0:
        coefficient = 0.0
    return native_norm, term_norm, coefficient


def _round12_merge_native_work_gradients(task_gradients, parameter_names, parameters):
    """Combine task and ordinary joint-work VJPs before per-term calibration."""
    combined = {name: gradient.detach().clone() for name, gradient in task_gradients.items()}
    for name, parameter in zip(parameter_names, parameters):
        gradient = parameter.grad
        if gradient is None:
            continue
        if name in combined:
            combined[name].add_(gradient.detach())
        else:
            combined[name] = gradient.detach().clone()
    return combined


def _round12_calibrate_structural_losses(common_run: Path, common_payload: Mapping[str, Any],
                                         config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Calibrate raw structural terms on the next TRAIN macro, without stepping."""
    caller_rng = _cpu_state_snapshot(capture_rng_state())
    model = provider = engine = None
    try:
        horizon = int(config["training"]["local_horizon"])
        optimizer_contract = _load_run_manifest(common_run)[0]["optimizer_schedule_contract"]
        model, provider = _build_provider_for_resume(
            config,
            common_payload,
            optimizer_contract,
            None,
            horizon,
            False,
            device,
        )
        provider_state = common_payload.get("provider_training_state", {})
        provider.load_training_state_dict(provider_state)
        core = getattr(model, "core", model)
        core.set_compact_phase(40, arm="merge")
        core.set_compact_loss_coefficients(tie=0.0, capacity=0.0)
        if int(common_payload.get("epoch", -1)) != 20:
            raise ValueError("Round12 structural calibration must branch from the completed common local-20 checkpoint.")
        local_epoch = 21
        arm = model.architecture
        sampler_epoch = int(config["initialization"]["clock_mapping"]["sampler_parent_epoch"]) + local_epoch
        ordered_cases = tuple(provider.epoch_cases(sampler_epoch, int(config["training"]["seed"])))
        if len(ordered_cases) != 600:
            raise ValueError("Round12 calibration requires the complete inherited TRAIN600 cohort.")
        update_cases = ordered_cases[:int(config["training"]["effective_batch_size"])]
        engine = _engine_for(
            config,
            device=device,
            parent_age=_parent_age(config),
            microbatch=int(config["training"]["microbatch_cases"]),
        )
        batches = engine.prepare_update_microbatches(
            provider,
            arm=arm,
            phase="joint",
            epoch=local_epoch,
            update_index=0,
            update_cases=update_cases,
            sampling_dataset_id=provider.sampling_dataset_id,
        )
        maximum = int(config["round12"]["calibration_max_microbatches"])
        if not 1 <= len(batches) <= maximum:
            raise ValueError("Round12 calibration microbatch count is outside the configured small bound.")
        # The first calibration microbatches exactly match the next common-prefix
        # continuation's TRAIN case order and maintained case_epoch_v1 keys.
        restore_rng_state(common_payload["rng_state"])
        model.train(True)
        native_clock = int(config["initialization"]["clock_mapping"]["controller_parent_coordinate"]) + local_epoch
        named_parameters = dict(model.named_parameters())
        new_names = _round12_new_parameter_names(model)
        parameters = [named_parameters[name] for name in new_names]
        if any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("A compact new parameter is unexpectedly frozen during soft calibration.")
        # A disposable optimizer supplies the exact native macro hooks; it is
        # never stepped and its empty moments are discarded with this model.
        optimizer, _ = engine._make_optimizer(model, provider, arm, local_epoch, None)
        optimizer.zero_grad(set_to_none=True)
        from .joint_native_objective import JointMacroObjective
        objective_api = JointMacroObjective(getattr(model, "core", model), optimizer, native_clock, 0)
        planned = {
            name: float(value)
            for name, value in provider.loss_denominators(batches, "joint", arm).items()
        }
        if not planned or any(not math.isfinite(value) or value <= 0 for value in planned.values()):
            raise ValueError("Round12 calibration requires the exact positive native TRAIN macro denominators.")
        scene_count = sum(len(set(map(str, batch.case_keys))) for batch in batches)
        if scene_count != len(update_cases):
            raise ValueError("Round12 calibration batches changed or duplicated the next TRAIN macro's primary cases.")
        native_gradients: dict[str, torch.Tensor] = {}
        term_gradients: dict[str, dict[str, torch.Tensor]] = {"tie": {}, "capacity": {}}
        observed: dict[str, float] = {name: 0.0 for name in planned}
        case_keys = []
        alpha = float(getattr(core, "compact_alpha", -1.0))
        if alpha != 1.0:
            raise ValueError(f"Round12 calibration must use tying alpha=1; core reported {alpha}.")
        for batch in batches:
            batch_auxiliary = batch.auxiliary if isinstance(batch.auxiliary, Mapping) else {}
            case_keys.extend(map(str, batch.case_keys))
            scene = provider.make_scene(batch.scene_inputs)
            objective_api.begin_microbatch()
            predictions, auxiliary_state = provider.predict_native(
                model, scene, batch.receivers, arm, "joint", epoch=native_clock,
                temperature=engine.config.temperature_for_epoch(native_clock),
            )
            expected_cases = len(batch.case_keys) + int(bool(batch_auxiliary.get("response_family_id")))
            objective_api.observe(expected_cases)
            loss_terms = provider.loss_terms(predictions, batch.targets, "joint", auxiliary_state)
            native_loss = None
            for name, term in loss_terms.items():
                if name not in planned:
                    raise ValueError(f"Round12 calibration native loss {name!r} is missing from the denominator prepass.")
                denominator = float(term.denominator)
                if denominator < 0 or not math.isfinite(denominator):
                    raise ValueError(f"Round12 calibration native denominator for {name!r} is invalid.")
                observed[name] += denominator
                if denominator == 0:
                    if float(term.numerator.detach()) != 0.0:
                        raise ValueError(f"Round12 calibration term {name!r} is nonzero with no valid targets.")
                    continue
                weighted = term.numerator * (float(term.weight) / planned[name])
                native_loss = weighted if native_loss is None else native_loss + weighted
            if native_loss is None:
                raise ValueError("Round12 calibration microbatch produced no ordinary native objective.")
            native_current = torch.autograd.grad(
                native_loss, parameters, retain_graph=True, allow_unused=True,
            )
            for name, gradient in zip(new_names, native_current):
                if gradient is not None:
                    native_gradients[name] = (gradient.detach().clone() if name not in native_gradients
                                              else native_gradients[name] + gradient.detach())
            prepared = getattr(predictions, "native_prepared", None)
            organization = getattr(prepared, "organization", None)
            if organization is None:
                raise ValueError("Round12 calibration requires the native prepared organization from predictions.")
            components = core.compact_structural_loss_components(organization)
            if not isinstance(components, Mapping) or set(components) != {"tie", "capacity"}:
                raise ValueError("Compact structural API must return exactly tie and capacity tensors.")
            structural_losses = {}
            for name in ("tie", "capacity"):
                value = components[name]
                if (not torch.is_tensor(value) or value.ndim != 1
                        or value.shape[0] != len(batch.case_keys) or not torch.isfinite(value.detach()).all()):
                    raise ValueError(f"Round12 raw {name} structural loss must be one finite value per primary scene.")
                structural_losses[name] = value.sum() * (alpha / scene_count)
            for index, name in enumerate(("tie", "capacity")):
                current = torch.autograd.grad(
                    structural_losses[name], parameters,
                    retain_graph=index == 0,
                    allow_unused=True,
                )
                for parameter_name, gradient in zip(new_names, current):
                    if gradient is not None:
                        term_gradients[name][parameter_name] = (
                            gradient.detach().clone() if parameter_name not in term_gradients[name]
                            else term_gradients[name][parameter_name] + gradient.detach()
                        )
        for name, expected in planned.items():
            if not math.isclose(observed[name], expected, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(f"Round12 calibration observed {observed[name]} targets for {name}, expected {expected}.")
        # Match the maintained update's ordinary native gradient: weighted
        # task/auxiliary/response loss plus the existing completed-macro
        # M/E/context work-controller VJPs, before gradient clipping. The
        # structural channels are zero-weighted for this native reference.
        objective_api.apply()
        native_gradients = _round12_merge_native_work_gradients(
            native_gradients, new_names, parameters,
        )
        coefficient_values = {}
        details = {}
        for name in ("tie", "capacity"):
            native_norm, term_norm, coefficient = _round12_term_gradient_calibration(
                native_gradients, term_gradients[name], config["round12"]["structural_loss_scale"],
            )
            coefficient_values[name] = coefficient
            details[name] = {"native_gradient_l2_norm_on_participating_new_parameters": native_norm,
                             "raw_gradient_l2_norm": term_norm, "coefficient": coefficient,
                             "scaled_gradient_l2_norm": coefficient * term_norm}
        result = {
            "schema": "thermal_compact_structural_loss_calibration_v1",
            "source_common_run": str(common_run),
            "source_checkpoint": str(_read_run_checkpoint(common_run, "latest")[1]),
            "source_common_local_epoch": 20,
            "sampling_partition": "TRAIN only; exposed validation and locked test were not read",
            "calibration_local_epoch": local_epoch,
            "calibration_sampler_epoch": sampler_epoch,
            "calibration_controller_epoch": native_clock,
            "optimizer_update_index": 0,
            "case_keys": case_keys,
            "microbatch_count": len(batches),
            "unique_primary_scene_count": scene_count,
            "alpha": alpha,
            "new_parameter_names": list(new_names),
            "terms": details,
            "formula": "0.02 * native_gradient_l2_norm / raw_structural_gradient_l2_norm; zero when either norm is zero or nonfinite",
            "native_gradient_convention": (
                "ordinary weighted TRAIN task/auxiliary/response gradient plus active existing M/E/context "
                "macro work-controller gradient, before clipping; each norm restricted to that structural "
                "term's participating compact parameters; new structural terms excluded"
            ),
        }
        return {"record": result, "coefficients": coefficient_values}
    finally:
        # The reconstructed model/provider are disposable. Restore only
        # process RNG streams and leave the common checkpoint untouched.
        restore_rng_state(caller_rng)
        if model is not None:
            del model
        if provider is not None:
            del provider
        if engine is not None:
            del engine
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _round12_sealed_identity(config, model, provider, engine, source_path, source_sha):
    from .unified_training import _engine_config_payload, _optimizer_spec_payload

    identity = _experiment_identity(config, provider, source_path, source_sha, diagnostic=False)
    specs = tuple(provider.optimizer_groups(model, model.architecture, "joint"))
    provider_identity = dict(provider.identity_payload())
    identity.update({
        "provider_identity": provider_identity,
        "engine_config": _engine_config_payload(engine.config),
        "selection_policy": asdict(engine.selection),
        "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in specs],
    })
    sampler_contract = engine.sampler_epoch_contract()
    if sampler_contract is not None:
        identity["sampler_epoch_mapping"] = dict(sampler_contract)
    if engine.config.sampling_version == SamplingKey.CASE_EPOCH_VERSION:
        identity["sampling_dataset_id"] = _sampling_dataset_id(provider_identity)
    return identity, specs


def _round12_write_child_start(run_output: Path, *, config, config_path: Path,
                               child_identity: Mapping[str, Any], source_path: Path,
                               source_sha: str, source_best_payload: Mapping[str, Any],
                               child_payload: Mapping[str, Any], optimizer_contract,
                               storage_snapshot: Mapping[str, Any], calibration_record: Mapping[str, Any],
                               parent_lineage: Mapping[str, Any], diagnostic: bool = False) -> None:
    import fcntl

    run_output = Path(run_output).expanduser().resolve(strict=False)
    run_output.mkdir(parents=True, exist_ok=True)
    lock_path = run_output / ".training.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Cannot initialize a Round12 child while another writer owns its run directory.") from error
        try:
            if any(path.name != ".training.lock" for path in run_output.iterdir()):
                raise FileExistsError(f"Round12 child run directory is no longer empty: {run_output}")
            _round12_write_child_start_locked(
                run_output,
                config=config,
                config_path=config_path,
                child_identity=child_identity,
                source_path=source_path,
                source_sha=source_sha,
                source_best_payload=source_best_payload,
                child_payload=child_payload,
                optimizer_contract=optimizer_contract,
                storage_snapshot=storage_snapshot,
                calibration_record=calibration_record,
                parent_lineage=parent_lineage,
                diagnostic=diagnostic,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _round12_write_child_start_locked(run_output: Path, *, config, config_path: Path,
                                      child_identity: Mapping[str, Any], source_path: Path,
                                      source_sha: str, source_best_payload: Mapping[str, Any],
                                      child_payload: Mapping[str, Any], optimizer_contract,
                                      storage_snapshot: Mapping[str, Any], calibration_record: Mapping[str, Any],
                                      parent_lineage: Mapping[str, Any], diagnostic: bool = False) -> None:
    from .unified_training import _atomic_torch_save

    run_output.mkdir(parents=True, exist_ok=True)
    layout = RunLayout(run_output)
    _atomic_json(layout.path("artifact_layout.json"), {
        "artifact_layout_version": 1,
        "run_dir": str(run_output),
    })
    _atomic_json(layout.write_path("experiment_identity.json"), child_identity)
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_file():
        resolved_config_path = layout.write_path(
            f"continuation_config_{int(config['training']['local_horizon'])}.json", category="configs",
        )
        _atomic_json(resolved_config_path, config)
    _write_run_manifest(
        run_output,
        config=config,
        config_path=resolved_config_path,
        source_path=source_path,
        source_sha=source_sha,
        optimizer_contract=optimizer_contract,
        case_ids=None,
        storage_snapshot=storage_snapshot,
        diagnostic=diagnostic,
        batch=int(config["training"]["effective_batch_size"]),
        microbatch=int(config["training"]["microbatch_cases"]),
        identity_microbatch=int(config["training"]["microbatch_cases"]),
    )
    best_payload = copy.deepcopy(dict(source_best_payload))
    best_payload["experiment_identity"] = copy.deepcopy(dict(child_identity))
    best_payload["branch_parent"] = copy.deepcopy(dict(parent_lineage))
    payload = copy.deepcopy(dict(child_payload))
    payload["branch_parent"] = copy.deepcopy(dict(parent_lineage))
    _atomic_json(layout.write_path("history.json"), list(payload.get("history", ())))
    _atomic_json(run_output / "round12_calibration.json", calibration_record)
    _atomic_torch_save(layout.write_path(f"epoch_{int(payload['epoch']):04d}_model.pt"), payload)
    _atomic_torch_save(layout.write_path("latest_model.pt"), payload)
    _atomic_torch_save(layout.write_path("best_by_field_mse_model.pt"), best_payload)
    from .unified_training import environment_snapshot, source_state_snapshot
    _atomic_json(layout.write_path("software.json"), environment_snapshot())
    _atomic_json(layout.write_path("source_state.json"), source_state_snapshot(Path(__file__).resolve().parents[2]))
    source_runtime = Path(str(parent_lineage["run_dir"])) / "execution_runtime.json"
    if source_runtime.is_file():
        import shutil
        shutil.copy2(source_runtime, run_output / "execution_runtime.json")
    else:
        _atomic_json(run_output / "execution_runtime.json", {
            "schema": "honf_research_execution_runtime_v1",
            "nominal_microbatch_cases": int(config["training"]["microbatch_cases"]),
            "runtime_microbatch_cases": int(config["training"]["microbatch_cases"]),
            "runtime_execution_options": copy.deepcopy(config["execution"]["options"]),
            "local_oom_retry_count": 0,
            "last_oom": {"status": "none"},
            "updated_unix": time.time(),
        })


def _validate_round12_common_fork(common_manifest, r0_config, r1_config):
    common = common_manifest["config"]
    if (common.get("round12", {}).get("role") != "common"
            or common.get("round12", {}).get("compact_arm") != "common"):
        raise ValueError("Round12 child initialization requires the ordinary common role run.")
    for config, role, arm in ((r0_config, "R0", "singleton"), (r1_config, "R1", "merge")):
        if (config.get("round12", {}).get("role") != role
                or config.get("round12", {}).get("compact_arm") != arm
                or config["initialization"] != common["initialization"]
                or config["dataset"] != common["dataset"]
                or config["optimizer"] != common["optimizer"]
                or config["training"] != common["training"]
                or config["execution"] != common["execution"]
                or config["storage"] != common["storage"]):
            raise ValueError(f"Round12 {role} child configuration diverges from the sealed common-prefix protocol.")
        for key in ("schema", "expected_parent_normalizer_key", "expected_parent_sampling_dataset_id",
                    "shared_prefix_epochs", "cut_after_local_epoch", "structural_loss_scale",
                    "calibration_max_microbatches", "hard_cut_rule"):
            if config["round12"].get(key) != common["round12"].get(key):
                raise ValueError(f"Round12 {role} child changes common {key}.")


def _validate_round12_common_checkpoint(common_payload: Mapping[str, Any]) -> None:
    """Require the adapted compact U-H common20 state before creating branches."""
    if common_payload.get("arm") != "A2-D":
        raise ValueError("Round12 common20 checkpoint is not the expected adapted U-H compact model.")
    compact_state = common_payload.get("provider_training_state", {}).get("compact_state", {})
    if (compact_state.get("schema") != "compact_unified_groups_state_v1"
            or int(compact_state.get("local_epoch", -1)) != 20
            or compact_state.get("arm") != "common"
            or bool(compact_state.get("cut_active"))):
        raise ValueError("Round12 common checkpoint is not the uncut shared K=N local-20 state.")


def fork_round12_common(common_run: str | Path, *, r0_config_path: str | Path,
                        r0_output: str | Path, r1_config_path: str | Path,
                        r1_output: str | Path, device: str) -> dict[str, Any]:
    """Fork calibrated R0/R1 runs from common20 with named optimizer/RNG state intact."""
    common_path = Path(common_run).expanduser().resolve(strict=True)
    common_manifest, _ = _load_run_manifest(common_path)
    common_config = common_manifest["config"]
    r0_config, r0_path = load_research_config(r0_config_path)
    r1_config, r1_path = load_research_config(r1_config_path)
    _validate_round12_common_fork(common_manifest, r0_config, r1_config)
    common_payload, common_checkpoint = _read_run_checkpoint(common_path, "latest")
    common_best_payload, common_best_checkpoint = _read_run_checkpoint(common_path, "best_by_field_mse")
    if int(common_payload.get("epoch", -1)) != 20:
        raise ValueError("Round12 branch fork requires the common run stopped at local epoch 20.")
    if not 1 <= int(common_best_payload.get("epoch", -1)) <= 20:
        raise ValueError("Round12 common best-field selector must refer to the completed shared prefix.")
    history = common_payload.get("history")
    if (not isinstance(history, Sequence) or len(history) != 20
            or any(int(row.get("epoch", -1)) != index or int(row.get("case_visits", -1)) != 600
                   or int(row.get("optimizer_updates", -1)) != 13 for index, row in enumerate(history, 1))):
        raise ValueError("Round12 common20 must retain all 20 complete TRAIN600 epochs and 260 optimizer updates.")
    _validate_round12_common_checkpoint(common_payload)
    r0_output_path = Path(r0_output).expanduser().resolve(strict=False)
    r1_output_path = Path(r1_output).expanduser().resolve(strict=False)
    if r0_output_path == r1_output_path:
        raise ValueError("Round12 R0 and R1 require distinct output run identities.")
    for output in (r0_output_path, r1_output_path):
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"Round12 fork outputs must be empty new directories: {output}")
    if Path(common_manifest["source_checkpoint"]).resolve() != Path(common_config["initialization"]["source_checkpoint"]).resolve():
        raise ValueError("Round12 common run manifest changed its literal U-H e1000 source binding.")
    source_path = Path(common_manifest["source_checkpoint"]).resolve(strict=True)
    source_sha = str(common_manifest["source_checkpoint_sha256"])
    if sha256_file(source_path) != source_sha:
        raise ValueError("Round12 source e1000 checkpoint changed after the common run was sealed.")
    device_value, device_info = resolve_device(device)
    checkpoint_size = int(common_checkpoint.stat().st_size)
    estimated = sum(_estimate_run_bytes(cfg, epochs=int(cfg["training"]["local_horizon"]),
                                        source_size=checkpoint_size, diagnostic=False)
                    for cfg in (r0_config, r1_config))
    common_storage = verify_storage(
        common_config, output=r0_output_path, estimated_bytes=estimated,
        peak_temporary_bytes=2 * checkpoint_size + 256 * 1024**2,
    )
    verify_storage(common_config, output=r1_output_path, estimated_bytes=0)
    calibrated = _round12_calibrate_structural_losses(common_path, common_payload, common_config, device_value)
    calibration_record = calibrated["record"]
    calibration_record["fork_device"] = device_info
    child_results = []
    for config, config_path, output, role, arm in (
        (r0_config, r0_path, r0_output_path, "R0", "singleton"),
        (r1_config, r1_path, r1_output_path, "R1", "merge"),
    ):
        model, provider = _build_provider_for_resume(
            config,
            common_payload,
            common_manifest["optimizer_schedule_contract"],
            None,
            int(config["training"]["local_horizon"]),
            False,
            device_value,
        )
        provider.load_training_state_dict(common_payload.get("provider_training_state", {}))
        core = getattr(model, "core", model)
        core.set_compact_phase(20, arm=arm)
        core.set_compact_loss_coefficients(
            tie=calibrated["coefficients"]["tie"],
            capacity=calibrated["coefficients"]["capacity"],
        )
        provider._round12_local_epoch = 20
        engine = _engine_for(
            config,
            device=device_value,
            parent_age=_parent_age(config),
            microbatch=int(config["training"]["microbatch_cases"]),
        )
        identity, specs = _round12_sealed_identity(
            config, model, provider, engine, source_path, source_sha,
        )
        from .unified_training import _optimizer_spec_payload
        optimizer_contract = [_optimizer_spec_payload(spec) for spec in specs]
        if (optimizer_contract != common_payload.get("optimizer_schedule_contract")
                or common_payload.get("optimizer_group_names") != [spec.name for spec in specs]):
            raise ValueError("Round12 branch fork would change the common named optimizer groups or schedule.")
        if identity.get("sampling_dataset_id") != common_payload.get("experiment_identity", {}).get("sampling_dataset_id"):
            raise ValueError("Round12 branch fork changed the exact parent TRAIN sampling namespace.")
        child_payload = copy.deepcopy(dict(common_payload))
        child_payload["experiment_identity"] = copy.deepcopy(identity)
        child_payload["model_state_dict"] = _cpu_state_snapshot(model.state_dict())
        child_payload["provider_training_state"] = _cpu_state_snapshot(provider.training_state_dict())
        if child_payload["provider_training_state"]["compact_state"].get("arm") != arm:
            raise RuntimeError("Round12 child fork failed to persist its compact arm before the first update.")
        parent_lineage = {
            "kind": "round12_common_prefix_fork",
            "run_dir": str(common_path),
            "checkpoint": str(common_checkpoint),
            "epoch": 20,
            "role": role,
            "shared_prefix_optimizer_updates": sum(int(row["optimizer_updates"]) for row in history),
        }
        _round12_write_child_start(
            output,
            config=config,
            config_path=config_path,
            child_identity=identity,
            source_path=source_path,
            source_sha=source_sha,
            source_best_payload=common_best_payload,
            child_payload=child_payload,
            optimizer_contract=optimizer_contract,
            storage_snapshot=common_storage,
            calibration_record=calibration_record,
            parent_lineage=parent_lineage,
        )
        child_results.append({"role": role, "arm": arm, "output": str(output),
                             "epoch": 20, "optimizer_updates_preserved": 260})
        del model, provider, engine
        import gc
        gc.collect()
        if device_value.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "status": "round12_children_forked_from_common20",
        "common_run": str(common_path),
        "common_checkpoint": str(common_checkpoint),
        "calibration": calibration_record,
        "children": child_results,
        "storage": common_storage,
    }


def fork_round12_terminal(run_dir: str | Path, *, output: str | Path,
                          local_horizon: int, device: str) -> dict[str, Any]:
    """Create a new exact-state continuation identity at a user-chosen horizon."""
    source_run = Path(run_dir).expanduser().resolve(strict=True)
    source_manifest, _ = _load_run_manifest(source_run)
    parent_config = source_manifest["config"]
    role = parent_config.get("round12", {}).get("role")
    arm = parent_config.get("round12", {}).get("compact_arm")
    if (parent_config["initialization"].get("normalization_mode") != "inherit_parent_train_stats"
            or (role, arm) not in (("R0", "singleton"), ("R1", "merge"))):
        raise ValueError("Round12 horizon continuation is available only from terminal R0 or R1 runs.")
    if type(local_horizon) is not int or local_horizon <= int(parent_config["training"]["local_horizon"]):
        raise ValueError("An explicit Round12 continuation horizon must exceed its terminal run horizon.")
    latest_payload, latest_path = _read_run_checkpoint(source_run, "latest")
    current_epoch = int(latest_payload.get("epoch", -1))
    if current_epoch != int(parent_config["training"]["local_horizon"]):
        raise ValueError("Round12 continuation requires a latest checkpoint at the sealed terminal epoch.")
    compact_state = latest_payload.get("provider_training_state", {}).get("compact_state", {})
    if (compact_state.get("schema") != "compact_unified_groups_state_v1"
            or int(compact_state.get("local_epoch", -1)) != current_epoch
            or compact_state.get("arm") != arm):
        raise ValueError("Round12 terminal checkpoint's compact phase differs from its run identity.")
    if role == "R1" and not bool(compact_state.get("cut_active")):
        raise ValueError("Terminal R1 continuation must preserve its active post-e40 hard cut.")
    if role == "R0" and bool(compact_state.get("cut_active")):
        raise ValueError("Terminal R0 singleton control cannot carry the R1 hard-cut state.")
    history = latest_payload.get("history")
    if (not isinstance(history, Sequence) or len(history) != current_epoch
            or int(history[-1].get("epoch", -1)) != current_epoch):
        raise ValueError("Round12 continuation checkpoint lacks its full local training history.")
    source_best_payload, source_best_path = _read_run_checkpoint(source_run, "best_by_field_mse")
    if int(source_best_payload.get("epoch", -1)) > current_epoch:
        raise ValueError("Round12 best-field selector points beyond the terminal checkpoint.")
    calibration_path = source_run / "round12_calibration.json"
    calibration_record = read_json(calibration_path)
    if calibration_record.get("schema") != "thermal_compact_structural_loss_calibration_v1":
        raise ValueError("Round12 terminal run lacks its original common-prefix calibration record.")

    child_config = copy.deepcopy(parent_config)
    old_name = str(child_config["name"])
    child_config["name"] = f"{old_name}_continue_{local_horizon}"
    child_config["training"]["local_horizon"] = local_horizon
    child_config["training"]["checkpoint_epochs"] = sorted(set(
        list(map(int, child_config["training"]["checkpoint_epochs"])) + [local_horizon]
    ))
    _validate_research_config(child_config)
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path == source_run or output_path.is_relative_to(source_run):
        raise ValueError("Round12 continuation requires a distinct output run identity.")
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"Round12 continuation output must be a new or empty directory: {output_path}")
    device_value, device_info = resolve_device(device)
    checkpoint_size = int(latest_path.stat().st_size)
    estimate = _estimate_run_bytes(
        child_config, epochs=local_horizon, source_size=checkpoint_size, diagnostic=False,
    )
    storage_snapshot = verify_storage(
        child_config, output=output_path, estimated_bytes=estimate,
        peak_temporary_bytes=2 * checkpoint_size + 256 * 1024**2,
    )
    source_path = Path(source_manifest["source_checkpoint"]).resolve(strict=True)
    source_sha = str(source_manifest["source_checkpoint_sha256"])
    if sha256_file(source_path) != source_sha:
        raise ValueError("Round12 source U-H e1000 checkpoint changed since terminal training.")
    model, provider = _build_provider_for_resume(
        child_config,
        latest_payload,
        source_manifest["optimizer_schedule_contract"],
        source_manifest.get("run_case_ids"),
        local_horizon,
        bool(source_manifest.get("diagnostic_only")),
        device_value,
    )
    provider.load_training_state_dict(latest_payload.get("provider_training_state", {}))
    core = getattr(model, "core", model)
    core.set_compact_phase(current_epoch, arm=arm)
    provider._round12_local_epoch = current_epoch
    engine = _engine_for(
        child_config,
        device=device_value,
        parent_age=_parent_age(child_config),
        microbatch=int(child_config["training"]["microbatch_cases"]),
    )
    identity, specs = _round12_sealed_identity(
        child_config, model, provider, engine, source_path, source_sha,
    )
    from .unified_training import _optimizer_spec_payload
    optimizer_contract = [_optimizer_spec_payload(spec) for spec in specs]
    if latest_payload.get("optimizer_group_names") != [spec.name for spec in specs]:
        raise ValueError("Round12 continuation changed named optimizer group order.")
    old_optimizer_groups = latest_payload.get("optimizer_state_by_name", {}).get("groups", ())
    current_groups = [
        {"name": spec.name, "parameter_names": list(spec.parameter_names)} for spec in specs
    ]
    if old_optimizer_groups != current_groups:
        raise ValueError("Round12 continuation changed named optimizer membership or order.")
    child_payload = copy.deepcopy(dict(latest_payload))
    child_payload["experiment_identity"] = copy.deepcopy(identity)
    child_payload["model_state_dict"] = _cpu_state_snapshot(model.state_dict())
    child_payload["provider_training_state"] = _cpu_state_snapshot(provider.training_state_dict())
    child_payload["optimizer_schedule_contract"] = copy.deepcopy(optimizer_contract)
    best_payload = copy.deepcopy(dict(source_best_payload))
    best_payload["experiment_identity"] = copy.deepcopy(identity)
    best_payload["optimizer_schedule_contract"] = copy.deepcopy(optimizer_contract)
    parent_lineage = {
        "kind": "round12_explicit_horizon_continuation",
        "run_dir": str(source_run),
        "checkpoint": str(latest_path),
        "epoch": current_epoch,
        "role": role,
        "new_local_horizon": local_horizon,
        "native_clocks": copy.deepcopy(latest_payload["provider_training_state"].get("clocks", {})),
    }
    _round12_write_child_start(
        output_path,
        config=child_config,
        config_path=output_path / f"continuation_{local_horizon}.json",
        child_identity=identity,
        source_path=source_path,
        source_sha=source_sha,
        source_best_payload=best_payload,
        child_payload=child_payload,
        optimizer_contract=optimizer_contract,
        storage_snapshot=storage_snapshot,
        calibration_record=calibration_record,
        parent_lineage=parent_lineage,
    )
    return {
        "status": "round12_terminal_fork_created",
        "parent_run": str(source_run),
        "parent_epoch": current_epoch,
        "role": role,
        "output": str(output_path),
        "new_local_horizon": local_horizon,
        "resume_from_local_epoch": current_epoch,
        "next_sampler_epoch": int(child_config["initialization"]["clock_mapping"]["sampler_parent_epoch"])
            + current_epoch + 1,
        "next_controller_epoch": int(child_config["initialization"]["clock_mapping"]["controller_parent_coordinate"])
            + current_epoch + 1,
        "storage": storage_snapshot,
    }


def evaluate(run_dir: str | Path, *, device: str = "cpu", checkpoint: str = "latest") -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve(strict=True)
    run_manifest, _ = _load_run_manifest(run)
    config = copy.deepcopy(run_manifest["config"])
    resolved_device, device_info = resolve_device(device)
    storage = verify_storage(config, output=run, estimated_bytes=16 * 1024**2,
                             peak_temporary_bytes=16 * 1024**2)
    payload, checkpoint_path = _read_run_checkpoint(run, checkpoint)
    if payload.get("experiment_identity", {}).get("family") != TASK_SCHEMA:
        raise ValueError("Selected checkpoint does not belong to this research runner.")
    local_horizon = int(config["training"]["local_horizon"])
    runtime_config = copy.deepcopy(config)
    runtime_config["training"]["effective_batch_size"] = int(
        run_manifest.get("effective_batch_size", config["training"]["effective_batch_size"]),
    )
    runtime_config["training"]["microbatch_cases"] = int(
        run_manifest.get("microbatch_cases", config["training"]["microbatch_cases"]),
    )
    runtime_execution_options = _execution_options_for_run(run, config)
    optimizer_contract = run_manifest.get("optimizer_schedule_contract")
    model, provider = _build_provider_for_resume(
        runtime_config,
        payload,
        optimizer_contract,
        run_manifest.get("run_case_ids"),
        local_horizon,
        bool(run_manifest.get("diagnostic_only")),
        resolved_device,
        runtime_execution_options=runtime_execution_options,
    )
    provider._identity_microbatch_size = int(
        run_manifest.get("identity_microbatch_cases", config["training"]["microbatch_cases"]),
    )
    engine = _engine_for(
        runtime_config,
        device=resolved_device,
        parent_age=_parent_age(config),
        microbatch=runtime_config["training"]["microbatch_cases"],
        diagnostic=bool(run_manifest.get("diagnostic_only")),
    )
    engine.bind_execution_runtime(
        run / "execution_runtime.json",
        nominal_microbatch_cases=provider._identity_microbatch_size,
        default_execution_options=config["execution"]["options"],
    )
    engine.set_runtime_overrides(provider=provider)
    epoch = int(payload["epoch"])
    model.eval()
    metrics = engine._evaluate(model, provider, model.architecture, "joint", epoch)
    result = {
        "schema": "honf_thermal_research_evaluation_v1",
        "run": str(run),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "parent_training_age": (_parent_age(config) if config["initialization"].get("fresh_model_schedule")
                                else _parent_age(config) + epoch),
        "local_training_epoch": epoch,
        "native_controller_epoch": (
            _fresh_model_schedule_epoch(config, epoch)
            if config["initialization"].get("fresh_model_schedule") else _parent_age(config) + epoch
        ),
        "dataset": "thermal",
        "training_membership": config["training"]["membership"],
        "partition": ("classic_canonical89_exposed_original_test"
                      if config["training"]["membership"] == "classic_train600_exposed_val89"
                      else "fixed25_v1_DEV22_exposed"),
        "locked_test_targets_read": False,
        "unexposed_original_test_targets_read": False,
        "device": device_info,
        "metrics": metrics,
        "created_unix": time.time(),
        "storage_snapshot": storage,
    }
    destination = run / _EVAL_DIR / f"validation_epoch_{epoch:04d}_{int(time.time())}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, result)
    result["output_path"] = str(destination)
    return result

def assess(run_dir: str | Path, *, dataset: str = "thermal") -> dict[str, Any]:
    if dataset != "thermal":
        raise ValueError("P0 assessment supports --dataset thermal only.")
    run = Path(run_dir).expanduser().resolve(strict=True)
    value, _ = _load_run_manifest(run)
    layout = RunLayout(run)
    history_path = layout.read_path("history.json")
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
    if not isinstance(history, list):
        raise TypeError("Run history must be a JSON list.")
    checkpoints = {}
    for selector in ("latest", "best_by_field_mse"):
        try:
            from .compat import load_trusted_checkpoint
            checkpoint_path = resolve_checkpoint(run, selector)
            payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
            checkpoints[selector] = {"path": str(checkpoint_path), "epoch": int(payload["epoch"]),
                                     "field_selector": payload.get("best_field_metric")}
        except FileNotFoundError:
            checkpoints[selector] = None
    validations = []
    for row in history:
        if isinstance(row, Mapping) and isinstance(row.get("validation"), Mapping):
            metrics = row["validation"]
            validations.append({"epoch": int(row["epoch"]),
                                "field_score": metrics.get("field_score"),
                                "flow_rmse": metrics.get("flow_rmse"),
                                "temperature_rmse": metrics.get("temperature_rmse"),
                                "surface_temperature_rmse": metrics.get("surface_temperature_rmse"),
                                "material_temperature_rmse": metrics.get("material_temperature_rmse")})
    best = min((row for row in validations if isinstance(row.get("field_score"), (int, float))),
               key=lambda row: float(row["field_score"]), default=None)
    summary = {
        "schema": "honf_thermal_research_assessment_v1",
        "run": str(run),
        "dataset": dataset,
        "diagnostic_only": bool(value.get("diagnostic_only")),
        "fit_status": "no completed epochs" if not history else "completed or cleanly stopped",
        "completed_epochs": len(history),
        "latest_epoch": None if not history else int(history[-1]["epoch"]),
        "case_visits": sum(int(row.get("case_visits", 0)) for row in history if isinstance(row, Mapping)),
        "optimizer_updates": sum(int(row.get("optimizer_updates", 0)) for row in history if isinstance(row, Mapping)),
        "total_train_seconds": sum(float(row.get("train_seconds", 0.0)) for row in history if isinstance(row, Mapping)),
        "latest_checkpoint": checkpoints.get("latest"),
        "best_field_checkpoint": checkpoints.get("best_by_field_mse"),
        "best_exposed_dev_review": best,
        "validation_scope": (
            "classic canonical89 exposed original test panel only; remainder excluded"
            if value.get("training_membership") == "classic_train600_exposed_val89"
            else "fixed25_v1 DEV22 only; locked TEST excluded"
        ),
        "locked_test_targets_read": False,
        "validation_reviews": validations,
        "diagnostic_notice": value.get("diagnostic_notice"),
        "source_checkpoint_sha256": value.get("source_checkpoint_sha256"),
    }
    return summary
