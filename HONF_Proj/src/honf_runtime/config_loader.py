"""Strict loading and deterministic composition of core and case profiles."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from honf_forward_core.config import UnifiedForwardConfig

from .paths import PROJECT_ROOT, resolve_path


CORE_TOP_LEVEL_KEYS = {
    "schema_version",
    "profile_name",
    "workflow",
    "model_family",
    "case",
    "model",
    "training",
    "checkpointing",
    "run",
    "_note",
}
CASE_TOP_LEVEL_KEYS = {
    "schema_version",
    "profile_name",
    "case_id",
    "plugin",
    "dataset",
    "model",
    "loss",
    "evaluation",
    "local_modules",
    "_note",
}
CORE_CASE_KEYS = {"id", "config", "dataset_id", "local_module_id"}
CORE_TRAINING_KEYS = {
    "seed",
    "device",
    "epochs",
    "learning_rate",
    "organizer_learning_rate",
    "weight_decay",
    "amp",
    "gradient_clip_norm",
    "plot_every_epochs",
    "port_curriculum",
    "max_train_batches_per_epoch",
    "max_val_batches",
    "init_checkpoint_path",
}
PORT_CURRICULUM_KEYS = {
    "schedule",
    "mode",
    "mixed_teacher_ratio",
    "teacher_epochs",
    "predicted_after_epoch",
    "mixed_teacher_ratio_start",
    "mixed_teacher_ratio_end",
}
CHECKPOINTING_KEYS = {
    "save_best",
    "save_best_field_mse",
    "save_best_temperature_mse",
    "save_best_predicted",
    "save_latest",
    "save_latest_every_epochs",
    "save_epoch_milestones",
}
RUN_KEYS = {"id", "name", "output_root"}


@dataclass(frozen=True)
class ConfigBundle:
    """Source profiles plus the immutable effective compatibility payload."""

    core_source: Path
    case_source: Path
    core: dict[str, Any]
    case: dict[str, Any]
    effective: dict[str, Any]
    config_hash: str
    overrides: dict[str, Any]
    experiment_source: Path | None = None
    experiment: dict[str, Any] = field(default_factory=dict)
    core_source_payload: dict[str, Any] = field(default_factory=dict)
    case_source_payload: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration root must be an object: {path}")
    return payload


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = sorted(key for key in payload if key not in allowed and not str(key).startswith("_"))
    if unknown:
        raise ValueError(f"Unknown {label} configuration sections: {unknown}")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object.")
    return value


def _validate_core_sections(core: Mapping[str, Any]) -> None:
    """Validate maintained nested core namespaces before case/resource writes."""

    case = _mapping(core.get("case"), label="core.case")
    training = _mapping(core.get("training"), label="core.training")
    checkpointing = _mapping(core.get("checkpointing"), label="core.checkpointing")
    run = _mapping(core.get("run"), label="core.run")
    model = _mapping(core.get("model"), label="core.model")
    _reject_unknown(case, CORE_CASE_KEYS, label="core.case")
    _reject_unknown(training, CORE_TRAINING_KEYS, label="core.training")
    _reject_unknown(checkpointing, CHECKPOINTING_KEYS, label="core.checkpointing")
    _reject_unknown(run, RUN_KEYS, label="core.run")
    workflow = str(core.get("workflow", ""))
    if workflow not in {"forward", "local_module"}:
        raise ValueError("Core workflow must be 'forward' or 'local_module'.")
    if workflow == "forward":
        _reject_unknown(model, {"core_honf"}, label="core.model")
        core_honf = _mapping(model.get("core_honf"), label="core.model.core_honf")
        allowed = set(UnifiedForwardConfig.__dataclass_fields__)  # type: ignore[attr-defined]
        _reject_unknown(core_honf, allowed, label="core.model.core_honf")
    elif model:
        raise ValueError("A local_module core profile must leave core.model empty; the case owns its architecture.")
    curriculum = training.get("port_curriculum")
    if curriculum is not None:
        _reject_unknown(
            _mapping(curriculum, label="core.training.port_curriculum"),
            PORT_CURRICULUM_KEYS,
            label="core.training.port_curriculum",
        )
    missing = [key for key in ("id", "config", "dataset_id") if not case.get(key)]
    if missing:
        raise ValueError(f"core.case is missing required values: {missing}")
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("core.training.epochs must be positive.")
    if float(training.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("core.training.learning_rate must be positive.")
    organizer_learning_rate = training.get("organizer_learning_rate")
    if organizer_learning_rate is not None and float(organizer_learning_rate) <= 0.0:
        raise ValueError("core.training.organizer_learning_rate must be null or positive.")
    save_latest_every_epochs = checkpointing.get("save_latest_every_epochs", 1)
    if (
        isinstance(save_latest_every_epochs, bool)
        or not isinstance(save_latest_every_epochs, int)
        or save_latest_every_epochs <= 0
    ):
        raise ValueError("core.checkpointing.save_latest_every_epochs must be a positive integer.")
    milestones = checkpointing.get("save_epoch_milestones", [])
    if (
        not isinstance(milestones, list)
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in milestones)
        or len(set(milestones)) != len(milestones)
    ):
        raise ValueError(
            "core.checkpointing.save_epoch_milestones must be a list of unique positive integers."
        )


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _merge_existing(base: dict[str, Any], patch: Mapping[str, Any], *, label: str) -> None:
    """Apply an overlay only to keys already declared by its source profile."""

    for key, value in patch.items():
        if str(key).startswith("_"):
            continue
        if key not in base:
            raise ValueError(f"Experiment overlay cannot introduce {label}.{key}; add it to the source profile first.")
        current = base[key]
        if isinstance(value, Mapping):
            if not isinstance(current, dict):
                raise TypeError(f"Experiment overlay {label}.{key} expects a scalar, not an object.")
            _merge_existing(current, value, label=f"{label}.{key}")
        else:
            base[key] = copy.deepcopy(value)


def _normalize_run_id(value: Any) -> str:
    raw = str(value if value is not None else "0001").strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be numeric, for example '0001'; got {raw!r}.")
    return f"{int(raw):04d}"


def load_config_bundle(
    core_config: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    overrides: Mapping[str, Any] | None = None,
    experiment_overlay: str | Path | None = None,
) -> ConfigBundle:
    """Load one core profile and its referenced case profile.

    The returned ``effective`` mapping intentionally matches the shape used by
    the proven ChannelThermal workflows.  Core and case source mappings remain
    separate and are saved alongside it for provenance.
    """

    root = Path(project_root).expanduser().resolve()
    core_path = resolve_path(core_config, project_root=root)
    if not core_path.exists():
        raise FileNotFoundError(f"Core configuration not found: {core_path}")
    core = _read_json(core_path)
    _reject_unknown(core, CORE_TOP_LEVEL_KEYS, label="core")
    if int(core.get("schema_version", 0)) != 1:
        raise ValueError("The current runtime supports core schema_version=1.")
    _validate_core_sections(core)

    case_ref = core.get("case")
    if not isinstance(case_ref, dict):
        raise ValueError("Core configuration must contain a 'case' object.")
    case_path_value = case_ref.get("config")
    if not case_path_value:
        raise ValueError("Core configuration must set case.config.")
    case_path = resolve_path(case_path_value, source_dir=core_path.parent, project_root=root)
    if not case_path.exists():
        raise FileNotFoundError(f"Case configuration not found: {case_path}")
    case = _read_json(case_path)
    _reject_unknown(case, CASE_TOP_LEVEL_KEYS, label="case")
    if int(case.get("schema_version", 0)) != 1:
        raise ValueError("The current runtime supports case schema_version=1.")
    configured_case_id = str(case_ref.get("id", "")).strip()
    if configured_case_id and configured_case_id != str(case.get("case_id", "")):
        raise ValueError(
            f"Core case.id={configured_case_id!r} does not match case config "
            f"case_id={case.get('case_id')!r}."
        )

    core_resolved = copy.deepcopy(core)
    case_resolved = copy.deepcopy(case)
    experiment_path = None
    experiment: dict[str, Any] = {}
    if experiment_overlay is not None:
        experiment_path = resolve_path(experiment_overlay, project_root=root)
        if not experiment_path.is_file():
            raise FileNotFoundError(f"Experiment overlay not found: {experiment_path}")
        experiment = _read_json(experiment_path)
        _reject_unknown(experiment, {"schema_version", "core", "case", "_note"}, label="experiment overlay")
        if int(experiment.get("schema_version", 0)) != 1:
            raise ValueError("The current runtime supports experiment overlay schema_version=1.")
        core_patch = _mapping(experiment.get("core", {}), label="experiment.core")
        case_patch = _mapping(experiment.get("case", {}), label="experiment.case")
        _reject_unknown(core_patch, {"model", "training", "checkpointing"}, label="experiment.core")
        _reject_unknown(case_patch, {"dataset", "model", "loss", "evaluation", "local_modules"}, label="experiment.case")
        _merge_existing(core_resolved, core_patch, label="core")
        _merge_existing(case_resolved, case_patch, label="case")
        _validate_core_sections(core_resolved)

    applied = {key: value for key, value in dict(overrides or {}).items() if value is not None}
    if "device" in applied:
        core_resolved.setdefault("training", {})["device"] = applied["device"]
    if "epochs" in applied:
        core_resolved.setdefault("training", {})["epochs"] = int(applied["epochs"])
    if "run_name" in applied:
        core_resolved.setdefault("run", {})["name"] = str(applied["run_name"])
    if "run_id" in applied:
        core_resolved.setdefault("run", {})["id"] = _normalize_run_id(applied["run_id"])

    run_cfg = core_resolved.setdefault("run", {})
    run_id = _normalize_run_id(run_cfg.get("id", "0001"))
    run_cfg["id"] = run_id
    effective: dict[str, Any] = {
        "schema_version": 1,
        "Run_ID": run_id,
        "workflow": str(core_resolved.get("workflow", "forward")),
        "model_family": str(core_resolved.get("model_family", "honf_forward")),
        "model": {
            **copy.deepcopy(core_resolved.get("model", {})),
            **copy.deepcopy(case_resolved.get("model", {})),
        },
        "dataset": copy.deepcopy(case_resolved.get("dataset", {})),
        "loss": copy.deepcopy(case_resolved.get("loss", {})),
        "training": copy.deepcopy(core_resolved.get("training", {})),
        "checkpointing": copy.deepcopy(core_resolved.get("checkpointing", {})),
        "paths": {
            "saved_model_dir": str(run_cfg.get("output_root", "project://Trained_Results")),
        },
        "case": {
            "id": str(case_resolved.get("case_id", "")),
            "plugin": str(case_resolved.get("plugin", "")),
            "profile_name": str(case_resolved.get("profile_name", "")),
            "config_path": str(case_path),
            "selection": {
                key: copy.deepcopy(value)
                for key, value in case_ref.items()
                if key not in {"id", "config"}
            },
        },
        "run": copy.deepcopy(run_cfg),
        "evaluation": copy.deepcopy(case_resolved.get("evaluation", {})),
        "local_modules": copy.deepcopy(case_resolved.get("local_modules", {})),
    }
    combined_for_hash = {
        "core": core_resolved,
        "case": case_resolved,
        "experiment": experiment,
        "effective": effective,
    }
    return ConfigBundle(
        core_source=core_path,
        case_source=case_path,
        core=core_resolved,
        case=case_resolved,
        effective=effective,
        config_hash=_canonical_hash(combined_for_hash),
        overrides=applied,
        experiment_source=experiment_path,
        experiment=experiment,
        core_source_payload=core,
        case_source_payload=case,
    )
