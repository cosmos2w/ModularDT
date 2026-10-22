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
    _sampling_dataset_id,
    capture_rng_state,
    restore_rng_state,
)
from channelthermal.training.content_hypergraph_task_v2 import ThermalContentHypergraphTaskV2


RESEARCH_SCHEMA = "honf_thermal_research_run_v1"
TASK_SCHEMA = "honf_thermal_research_provider_v1"
_CONFIG_NAME = "research_run.json"
_EVAL_DIR = "evaluations/research"


def _parent_age(config: Mapping[str, Any]) -> int:
    return 0 if config["initialization"]["normalization_mode"] == "fresh_full_train600_stats" else int(
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
    if training.get("membership") not in ("fixed25_train150_dev22", "original_train600_fixed25_dev22"):
        raise ValueError("Thermal membership must be fixed25 TRAIN150/DEV22 or original TRAIN600/fixed25 DEV22.")
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
    if (optimizer.get("kind") != "fresh_adamw_from_source_groups"
            or optimizer.get("schedule") != "constant_source_terminal_lr"):
        raise ValueError("Research starts reset optimizer moments and use each source group's terminal learning rate.")
    if normalization_mode == "fresh_full_train600_stats":
        schedule = config["initialization"].get("fresh_model_schedule")
        if not isinstance(schedule, Mapping) or any(
                type(schedule.get(key)) is not int or schedule[key] < 1 for key in
                ("coordinate_start_epoch", "coordinate_span_epochs", "ramp_epochs")):
            raise ValueError("Fresh normalization needs an explicit positive native model-clock ramp.")
        strengths = config["initialization"].get("fresh_controller_lambda0")
        if (not isinstance(strengths, Mapping) or set(strengths) != {"M", "E", "context"}
                or any(not isinstance(value, (int, float)) or not math.isfinite(float(value))
                       or float(value) <= 0.0 for value in strengths.values())):
            raise ValueError("Fresh normalization needs positive calibrated M/E/context lambda0 strengths.")
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
        "inherit_source_train150", "fresh_full_train600_stats",
    ):
        raise ValueError("Select source TRAIN150 normalization or explicit fresh full-TRAIN600 normalization.")


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
    milestones = max(1, math.ceil(epochs / interval))
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


def _source_components(config: Mapping[str, Any], device: torch.device):
    init = config["initialization"]
    source_path = Path(init["source_checkpoint"]).expanduser().resolve(strict=True)
    expected_sha = init.get("source_sha256")
    actual_sha = sha256_file(source_path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("Source initialization checkpoint bytes differ from the optional preset SHA256.")
    from .compat import load_trusted_checkpoint
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
    path = resolve_checkpoint(run_dir, selector)
    if not path.is_file():
        raise FileNotFoundError(f"Run has no {selector} checkpoint: {path}")
    return load_trusted_checkpoint(path, map_location="cpu"), path


def _model_from_run_payload(payload: Mapping[str, Any], device: torch.device):
    identity = payload.get("experiment_identity", {})
    provider_identity = identity.get("provider_identity", {})
    adapter_state = provider_identity.get("adapter_state")
    if not isinstance(adapter_state, Mapping):
        raise ValueError("Research checkpoint is missing its self-contained Thermal adapter state.")
    from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
    return ThermalContentHypergraphAdapterV2.from_state_payload(
        adapter_state, device=str(device), state_dict=payload["model_state_dict"],
    )


class ResearchThermalProvider(ThermalContentHypergraphTaskV2):
    """Task identity, optimizer, and full-TRAIN membership wrapper.

    Runtime batching/losses/validation are inherited directly from the actual
    Thermal v2 provider built by ``build_thermal_content_hypergraph_task_v2``.
    """

    def __init__(self, base_provider, *, model, config, train_ids, validation_ids,
                 training_cases, validation_cases, source_payload, optimizer_contract,
                 diagnostic=False, identity_microbatch=None, local_horizon=None):
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
        # Inherited v2 methods preserve the base development objective and DEV22 semantics.
        self.protocol = "development"
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
        self.sampling_dataset_id = _sampling_dataset_id(self.identity_payload())

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
        membership_sha = research_membership_fingerprint(
            dataset_identity=dataset_identity,
            training_case_ids=self.training_case_ids,
            validation_case_ids=self.validation_case_ids,
        )
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
            "normalization_stats_sha256": self.model.state_payload()["normalizer_key"],
            "case_limit_diagnostic": self._research_case_limit,
            "validation_scope": "fixed25_v1_DEV22_exposed_original_test_subset; locked remainder excluded",
            "source_binding": {
                **dict(payload.get("source_binding", {})),
                "protocol": TASK_SCHEMA,
                "dataset_profile": self._research_membership,
                "normalization_fit_scope": (
                    "original TRAIN600 only" if self._research_normalization_mode == "fresh_full_train600_stats"
                    else "inherited source TRAIN150; unchanged"
                ),
                "normalization_mode": self._research_normalization_mode,
                "original_test_targets_outside_DEV22_read": False,
                "locked_test_targets_read": False,
                "research_membership_sha256": membership_sha,
                "source_checkpoint_sha256": self._research_config["initialization"].get("source_sha256"),
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
                "optimizer": "AdamW with fresh moments and source-compatible named groups",
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
                "source_checkpoint": self._research_config["initialization"]["source_checkpoint"],
                "source_checkpoint_sha256": self._research_config["initialization"].get("source_sha256"),
                "local_horizon": self.total_epochs,
                "training_membership": self._research_membership,
                "normalization_mode": self._research_normalization_mode,
                "fresh_controller_lambda0": copy.deepcopy(
                    self._research_config["initialization"].get("fresh_controller_lambda0")
                ),
                "fresh_controller_lambda0_source": self._research_config[
                    "initialization"
                ].get("fresh_controller_lambda0_source"),
                "exposed_validation_membership": "fixed25_v1 DEV22",
                "locked_test_targets_read": False,
                "diagnostic_case_limit": self._research_case_limit,
                "execution_options": copy.deepcopy(self._research_config["execution"]["options"]),
                "optimizer_moments": "reset on source initialization; retained only on exact in-run resume",
            },
        })
        self.sampling_dataset_id = _sampling_dataset_id(payload)
        return payload

    def optimizer_groups(self, model, arm, stage):
        del stage
        if model is not self.model or arm != model.architecture:
            raise ValueError("Research optimizer requested for the wrong bound model.")
        named = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
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
            schedule = ScheduleSpec(
                peak_lr=lr,
                warmup_start_lr=lr,
                warmup_epochs=0,
                hold_through_epoch=self.total_epochs,
                total_epochs=self.total_epochs + 1,
                final_lr=lr,
            )
            groups.append(OptimizerGroupSpec(
                str(source["name"]), names, schedule,
                weight_decay=float(source["weight_decay"]),
                betas=tuple(map(float, source.get("betas", (0.9, 0.999)))),
                eps=float(source.get("eps", 1.0e-8)),
            ))
        if covered != named:
            raise ValueError(f"Source optimizer groups omit current trainable parameters: {sorted(named-covered)}")
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
            "validation_scope": "fixed25_v1 DEV22; locked remainder excluded",
        }


class ResearchTrainingEngine(TrainingEngine):
    """Use local history epochs with separate sampler and native model clocks."""

    def __init__(self, config: EngineConfig, *, device, parent_age: int, selection=None,
                 fresh_model_schedule: Mapping[str, Any] | None = None):
        super().__init__(config, device=device, selection=selection)
        self.parent_age = int(parent_age)
        self.fresh_model_schedule = None if fresh_model_schedule is None else dict(fresh_model_schedule)
        self.nominal_microbatch_cases = int(config.microbatch_cases)
        self.runtime_microbatch_cases = self.nominal_microbatch_cases
        self.local_oom_retry_count = 0
        self._execution_runtime_path: Path | None = None

    def bind_execution_runtime(self, path: str | Path, *, nominal_microbatch_cases: int) -> None:
        """Load run-local packing state without changing the sealed engine identity."""
        nominal = int(nominal_microbatch_cases)
        if nominal < 1 or nominal > self.config.effective_cases:
            raise ValueError("Nominal research microbatch must fit inside the effective batch.")
        self.nominal_microbatch_cases = nominal
        self.runtime_microbatch_cases = nominal
        self.local_oom_retry_count = 0
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
        if (type(runtime) is not int or not 1 <= runtime <= nominal
                or type(retries) is not int or retries < 0):
            raise ValueError("Saved research runtime packing or retry count is malformed.")
        self.runtime_microbatch_cases = runtime
        self.local_oom_retry_count = retries

    def _write_execution_runtime(self, *, last_oom: Mapping[str, Any]) -> None:
        if self._execution_runtime_path is None:
            return
        _atomic_json(self._execution_runtime_path, {
            "schema": "honf_research_execution_runtime_v1",
            "nominal_microbatch_cases": self.nominal_microbatch_cases,
            "runtime_microbatch_cases": self.runtime_microbatch_cases,
            "local_oom_retry_count": self.local_oom_retry_count,
            "last_oom": dict(last_oom),
            "updated_unix": time.time(),
        })
    def prepare_update_microbatches(self, provider, *, arm: str, phase: str, epoch: int,
                                   update_index: int, update_cases: Sequence[Any],
                                   sampling_dataset_id: str):
        """Use recoverable packing while preserving the exact macro case order."""
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
        return fit_epoch if self.fresh_model_schedule is not None else self.parent_age + fit_epoch

    def sampler_epoch_contract(self):
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

    def _run_update(self, *args, epoch: int, **kwargs):
        # The optimizer schedule is local and constant. Warm starts continue
        # their native controller clock; fresh full-TRAIN starts run the
        # controller's near-open work ramp on its native absolute coordinate.
        model_epoch = (self.parent_age + int(epoch) if self.fresh_model_schedule is None
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
        while True:
            try:
                return super()._run_update(
                    model, provider, optimizer, microbatches, *args[4:], epoch=model_epoch, **kwargs,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if not _is_cuda_oom(error):
                    raise
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
        for name in ("_joint_entries", "_context_entries", "_budgeted_entries", "_budgeted_observations"):
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
    _, base, _ = build_thermal_content_hypergraph_task_v2(recipe, device="cpu")
    source_train600 = tuple(case_id for case_id in base.catalog_case_ids
                            if base.case_split_by_id[case_id] == "train")
    fixed_train150 = tuple(base.training_case_ids)
    dev_ids = tuple(development_case_ids(base.manifest, "test"))
    if len(source_train600) != 600 or len(fixed_train150) != 150 or len(dev_ids) != 22:
        raise ValueError("Thermal membership must resolve to original TRAIN600 and fixed25 TRAIN150/DEV22.")
    if set(source_train600) & set(dev_ids) or any(base.case_split_by_id.get(key) != "test" for key in dev_ids):
        raise ValueError("TRAIN and exposed DEV22 membership is not bound to disjoint source partitions.")
    train_ids = (fixed_train150 if config["training"]["membership"] == "fixed25_train150_dev22"
                 else source_train600)
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

    # The base v2 task may hold TRAIN150 and DEV22 examples. Drop them before
    # loading exactly the selected IDs through the same established reader.
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
    )
    provider.sampling_dataset_id = _sampling_dataset_id(provider.identity_payload())
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


def _build_provider_for_resume(config, checkpoint_payload, optimizer_contract,
                               case_ids, local_horizon, diagnostic, device):
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
    return model, provider


def _engine_for(config, *, device: torch.device, parent_age: int, microbatch: int | None = None,
                diagnostic: bool = False) -> ResearchTrainingEngine:
    training = config["training"]
    horizon = int(training["local_horizon"])
    batch = int(training["effective_batch_size"])
    microbatch_value = int(microbatch or training["microbatch_cases"])
    monitors = (1,) if diagnostic else tuple(range(int(training["monitor_every"]), horizon + 1,
                                                     int(training["monitor_every"])))
    checkpoints = (1,) if diagnostic else tuple(range(int(training["checkpoint_every"]), horizon + 1,
                                                       int(training["checkpoint_every"])))
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
                              if config["initialization"]["normalization_mode"] == "fresh_full_train600_stats"
                              else None),
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
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
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
            "validation": "fixed25_v1 DEV22 only",
            "locked_test_targets_read": False,
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
                      permitted_devices: Sequence[str] | None = None):
    retries = 0
    active_engine = engine
    current_resume = resume_checkpoint
    original_output = output
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
            )
        previous_handlers = _install_clean_stop_handlers(output)
        try:
            return active_engine.fit(
                model, provider, output,
                identity=identity,
                arm=arm,
                stop_after=int(stop_after),
                resume_checkpoint=current_resume,
                allow_microbatch_change=allow_microbatch_change,
            )
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
                active_engine.bind_execution_runtime(
                    output / "execution_runtime.json", nominal_microbatch_cases=nominal_microbatch,
                )
                active_engine.runtime_microbatch_cases = next_microbatch
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
    return {
        "schema": RESEARCH_SCHEMA,
        "family": TASK_SCHEMA,
        "workflow": "ordinary_honf_research_train_v1",
        "model_family": "unified_continuous_joint_native_v1",
        "dataset": "thermal",
        "config_name": config["name"],
        "config_sha256": _canonical_sha256(config),
        "training_membership": config["training"]["membership"],
        "normalization_mode": config["initialization"]["normalization_mode"],
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "source_parent_age": int(config["initialization"]["source_parent_age"]),
        "training_parent_age": _parent_age(config),
        "source_model_initialization": (
            "fresh_random_core_from_declared_architecture; source weights not loaded"
            if _parent_age(config) == 0 and int(config["initialization"]["source_parent_age"]) > 0
            else "model_weights_only; optimizer moments reset"
        ),
        "fresh_controller_initialization": (
            {
                "kind": "calibrated_lambda0_hyperparameters_with_reset_dual_state",
                "lambda0": copy.deepcopy(config["initialization"]["fresh_controller_lambda0"]),
                "calibration_source": config["initialization"]["fresh_controller_lambda0_source"],
                "optimizer_moments_and_mature_dual_state_loaded": False,
            }
            if config["initialization"]["normalization_mode"] == "fresh_full_train600_stats"
            else None
        ),
        "recipe": dict(provider_identity["recipe"]),
        "execution_options": copy.deepcopy(config["execution"]["options"]),
        "diagnostic_only": diagnostic,
        "diagnostic_case_limit": len(provider.training_case_ids) if diagnostic else None,
    }


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
    if (config["initialization"]["normalization_mode"] == "fresh_full_train600_stats"
            and diagnostic_only):
        raise ValueError("Fresh full-TRAIN normalization is available only for a full-membership fit.")

    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"New research fits require an empty output directory: {output_path}")
    source_path = Path(config["initialization"]["source_checkpoint"]).expanduser().resolve(strict=True)
    source_size = source_path.stat().st_size
    estimate = _estimate_run_bytes(config, epochs=stop_after, source_size=source_size,
                                   diagnostic=diagnostic_only)
    peak_estimate = 2 * source_size + 256 * 1024**2
    storage_snapshot = verify_storage(config, output=output_path, estimated_bytes=estimate,
                                      peak_temporary_bytes=peak_estimate)
    Path(storage_snapshot["output_root"]).mkdir(parents=True, exist_ok=True)
    permitted = _permitted_order(device, devices)
    selected_device, device_info = resolve_device(permitted[0], permitted_devices=permitted)
    model, source_payload, source_path, source_sha = _source_components(config, selected_device)
    optimizer_contract = source_payload["experiment_identity"].get(
        "optimizer_schedule_contract", source_payload.get("optimizer_schedule_contract"),
    )
    if not isinstance(optimizer_contract, Sequence) or not optimizer_contract:
        raise ValueError("Source checkpoint lacks its named optimizer group schedule contract.")

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
        "source_checkpoint": str(source_path),
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

def resume(run_dir: str | Path, *, device: str, additional_epochs: int,
           devices: Sequence[str] | None = None) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve(strict=True)
    run_manifest, _ = _load_run_manifest(run)
    config = copy.deepcopy(run_manifest["config"])
    if type(additional_epochs) is not int or additional_epochs < 1:
        raise ValueError("--additional-epochs must be a positive integer.")
    permitted = _permitted_order(device, devices)
    selected_device, device_info = resolve_device(permitted[0], permitted_devices=permitted)
    storage_snapshot = verify_storage(config, output=run, estimated_bytes=0)
    payload, latest_path = _read_run_checkpoint(run, "latest")
    local_horizon = int(config["training"]["local_horizon"])
    current_epoch = int(payload["epoch"])
    stop_after = current_epoch + additional_epochs
    if stop_after > local_horizon:
        raise ValueError(f"Resume would exceed local horizon {local_horizon}.")

    runtime_config = copy.deepcopy(config)
    runtime_config["training"]["effective_batch_size"] = int(
        run_manifest.get("effective_batch_size", config["training"]["effective_batch_size"]),
    )
    runtime_config["training"]["microbatch_cases"] = int(
        run_manifest.get("microbatch_cases", config["training"]["microbatch_cases"]),
    )
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
    identity = _experiment_identity(
        config,
        provider,
        run_manifest["source_checkpoint"],
        run_manifest["source_checkpoint_sha256"],
        diagnostic=diagnostic,
    )
    print(json.dumps({
        "event": "research_resume_prepared",
        "run": str(run),
        "latest_checkpoint": str(latest_path),
        "from_epoch": current_epoch,
        "stop_after": stop_after,
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
        permitted_devices=permitted,
    )


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
    optimizer_contract = run_manifest.get("optimizer_schedule_contract")
    model, provider = _build_provider_for_resume(
        runtime_config,
        payload,
        optimizer_contract,
        run_manifest.get("run_case_ids"),
        local_horizon,
        bool(run_manifest.get("diagnostic_only")),
        resolved_device,
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
    epoch = int(payload["epoch"])
    model.eval()
    metrics = engine._evaluate(model, provider, model.architecture, "joint", epoch)
    result = {
        "schema": "honf_thermal_research_evaluation_v1",
        "run": str(run),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "parent_training_age": _parent_age(config) + epoch,
        "dataset": "thermal",
        "training_membership": config["training"]["membership"],
        "partition": "fixed25_v1_DEV22_exposed",
        "locked_test_targets_read": False,
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
        "validation_scope": "fixed25_v1 DEV22 only; locked TEST excluded",
        "locked_test_targets_read": False,
        "validation_reviews": validations,
        "diagnostic_notice": value.get("diagnostic_notice"),
        "source_checkpoint_sha256": value.get("source_checkpoint_sha256"),
    }
    return summary
