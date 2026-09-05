"""Train the coupled ChannelThermal hypergraph operator neural field.

This case-owned implementation is dispatched by the project-level ``train.py``
after core and case configurations have been composed and confirmed.

A run writes resolved settings, CSV metrics, selected checkpoints, diagnostic
loss curves, and a compact summary.

The tensor path is documented in ``README.md``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from channelthermal.data.collation import ChannelThermalBatchCollator, ModuleCountBucketBatchSampler
from channelthermal.data.datasets import GlobalChannelThermalDataset
from honf_runtime.compat import (
    autocast_context,
    count_parameters,
    current_timestamp,
    ensure_dir,
    make_grad_scaler,
    load_trusted_checkpoint,
    read_json,
    recursive_to_device,
    resolve_demo_path,
    select_device,
    set_seed,
    strip_module_prefix,
    write_json,
)
from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_forward_core.training.diagnostics import HONF_DIAGNOSTIC_KEYS, compute_honf_diagnostics, organizer_regularization_loss
from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.training_tools.losses import channelthermal_field_mse
from channelthermal.training.checkpoints import (
    _file_sha256,
    _partial_initialize_model,
    _restore_rng_state,
    _validate_initialization_checkpoint,
    _validate_resume_checkpoint,
    save_checkpoint,
)
from channelthermal.training.epoch import (
    GRADIENT_DIAGNOSTIC_KEYS,
    INTERFACE_DIAGNOSTIC_KEYS,
    effective_local_loss_weights,
    effective_port_condition_settings,
    effective_port_global_weight,
    interface_loss,
    internal_loss,
    make_model_inputs,
    organizer_regularization,
    pack_scalar_metrics,
    port_condition_loss,
    port_cyclic_smoothness_loss,
    port_global_consistency_loss,
    predicted_consistency_weight_for_epoch,
    run_epoch,
)
from channelthermal.training.optimizer import (
    _optimizer_group_record,
    _optimizer_inventory_digest,
    _print_optimizer_group_inventory,
    _validate_optimizer_resume_compatibility,
    build_forward_optimizer,
    refresh_optimizer_group_inventory,
)
from channelthermal.training.reporting import (
    best_metrics_payload,
    repair_metrics_csv_for_append,
    save_global_loss_plots,
    write_metrics_row,
)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line options for this workflow."""

    parser = argparse.ArgumentParser(description="Train the standalone global-field ChannelThermal HONF model.")
    parser.add_argument("--config", type=str, required=True, help="Resolved legacy-format JSON; prefer the project train.py.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--Run_ID", dest="run_id", type=str, default=None)
    checkpoint_mode = parser.add_mutually_exclusive_group()
    checkpoint_mode.add_argument("--resume-checkpoint", type=str, default=None, help="Resume training from a saved HONF-CL checkpoint.")
    checkpoint_mode.add_argument(
        "--initialize-checkpoint",
        type=str,
        default=None,
        help="Initialize a new run from compatible name-and-shape matched checkpoint weights.",
    )
    return parser.parse_args()


def normalize_run_id(value: Any, fallback: str = "0001") -> str:
    """Normalize run id."""

    raw = str(value or fallback).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be a numeric serial such as '0001'; got {raw!r}.")
    return f"{int(raw):04d}"


def reuses_primary_validation_for_predicted_mode(local_port_condition_mode: str) -> bool:
    """Return whether the primary validation pass already uses predicted ports."""

    return str(local_port_condition_mode).lower() == "predicted"


def should_save_latest_checkpoint(
    epoch: int,
    total_epochs: int,
    checkpoint_config: Dict[str, Any],
) -> bool:
    """Apply the optional latest-checkpoint cadence while always saving the final epoch."""

    if not bool(checkpoint_config.get("save_latest", True)):
        return False
    cadence = max(int(checkpoint_config.get("save_latest_every_epochs", 1)), 1)
    return int(epoch) % cadence == 0 or int(epoch) == int(total_epochs)


def should_save_milestone_checkpoint(
    epoch: int,
    checkpoint_config: Dict[str, Any],
) -> bool:
    """Return whether this exact epoch is an explicitly retained milestone."""

    return int(epoch) in {
        int(value) for value in checkpoint_config.get("save_epoch_milestones", [])
    }


def resolve_run_id(args_value: Any, cfg: Dict[str, Any], training_cfg: Dict[str, Any]) -> str:
    """Resolve Run_ID with CLI override first, then template settings.

    Some templates carry both top-level `Run_ID` and `training.Run_ID`.
    `training.Run_ID` is the operational training setting, so it wins when no
    `--Run_ID` override is provided. The resolved value is mirrored back into
    both places before saving `config_resolved.json`.
    """

    top_level = cfg.get("Run_ID")
    training_value = training_cfg.get("Run_ID")
    if args_value is not None:
        return normalize_run_id(args_value, "0001")
    if top_level is not None and training_value is not None:
        normalized_top = normalize_run_id(top_level, "0001")
        normalized_training = normalize_run_id(training_value, "0001")
        if normalized_top != normalized_training:
            print(
                "[warning] Conflicting Run_ID values in config: "
                f"top-level Run_ID={normalized_top}, training.Run_ID={normalized_training}; "
                "using training.Run_ID. Pass --Run_ID to override both."
            )
        return normalized_training
    return normalize_run_id(training_value if training_value is not None else top_level, "0001")


def sanitize_run_suffix(value: Any) -> str:
    """Perform the sanitize run suffix operation used by this module."""

    raw = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw).strip("_")


def resolve_config_path(path_like: str) -> Path:
    """Resolve config path."""

    path = Path(path_like).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return resolve_demo_path(path)


def _auto_int(value: Any, fallback: int) -> int:
    """Perform the auto int operation used by this module."""

    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return int(fallback)
    return int(value)


def _auto_float(value: Any, fallback: float) -> float:
    """Perform the auto float operation used by this module."""

    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return float(fallback)
    return float(value)


def _first_domain_and_radius(dataset: GlobalChannelThermalDataset) -> tuple[float, float, float]:
    """Perform the first domain and radius operation used by this module."""

    if len(dataset) == 0:
        return 12.0, 4.0, 0.45
    sample = dataset[0]
    structure = sample["structure"]
    lx = float(structure["domain_length_x"][0])
    ly = float(structure["domain_length_y"][0])
    material = structure["material_params"]
    radius = float(material[5]) if material.shape[0] > 5 and float(material[5]) > 0.0 else 0.45
    return lx, ly, radius


LOCAL_COUPLING_KEYS = {
    "use_local_surrogate",
    "local_surrogate_checkpoint_path",
    "freeze_local_surrogate",
    "local_surrogate_latent_dim",
    "local_module_params_from_used_ports",
    "default_num_interface_points",
}

PHYSICAL_CORRECTION_KEYS = {
    "local_surrogate_flux_mode",
    "local_surrogate_flux_blend_alpha",
    "interaction_refinement_steps",
    "port_global_consistency_radius_offset",
    "port_global_consistency_num_points",
}

CHANNELTHERMAL_KEYS = {
    "field_names",
    "material_param_dim",
    "heat_scale",
    "global_feature_schema",
    "legacy_active_fraction_reference_slots",
    "internal_prediction_mode",
    "fallback_internal_query_dim",
    "fallback_interface_dim",
    "fallback_hidden_dim",
    "fallback_fourier_frequencies",
}


def _merge_authoritative(
    channel_payload: Dict[str, Any],
    section_payload: Dict[str, Any],
    keys: set[str],
    *,
    section_name: str,
) -> None:
    """Perform the merge authoritative operation used by this module."""

    for key in keys:
        if key not in section_payload:
            continue
        if key in channel_payload and channel_payload[key] != section_payload[key]:
            print(
                f"[warning] Conflicting model.channelthermal.{key}={channel_payload[key]!r}; "
                f"using authoritative model.{section_name}.{key}={section_payload[key]!r}."
            )
        channel_payload[key] = section_payload[key]


def build_model_config(payload: Dict[str, Any], dataset: GlobalChannelThermalDataset) -> ChannelThermalHONFConfig:
    """Build model config."""

    model_payload = dict(payload.get("model", {}))
    core_payload = dict(model_payload.get("core_honf", {}))
    channel_payload = dict(model_payload.get("channelthermal", {}))
    local_payload = dict(model_payload.get("local_coupling", {}))
    physical_payload = dict(model_payload.get("physical_correction", {}))
    if "enable_fallback_heads" in channel_payload:
        print("[warning] model.channelthermal.enable_fallback_heads is ignored; use internal_prediction_mode instead.")
        channel_payload.pop("enable_fallback_heads", None)
    _merge_authoritative(channel_payload, local_payload, LOCAL_COUPLING_KEYS, section_name="local_coupling")
    _merge_authoritative(channel_payload, physical_payload, PHYSICAL_CORRECTION_KEYS, section_name="physical_correction")
    lx, ly, radius = _first_domain_and_radius(dataset)
    core_payload["field_dim"] = _auto_int(core_payload.get("field_dim"), dataset.field_dim)
    core_payload["domain_length_x"] = _auto_float(core_payload.get("domain_length_x"), lx)
    core_payload["domain_length_y"] = _auto_float(core_payload.get("domain_length_y"), ly)
    core_payload["module_radius"] = _auto_float(core_payload.get("module_radius"), radius)
    channel_payload["material_param_dim"] = _auto_int(channel_payload.get("material_param_dim"), dataset.material_param_dim)
    channel_payload["default_num_interface_points"] = _auto_int(
        channel_payload.get("default_num_interface_points"),
        dataset.n_interface_points or 64,
    )
    return ChannelThermalHONFConfig.from_dict({"core_honf": core_payload, "channelthermal": channel_payload})


def resolved_config_payload(
    cfg: Dict[str, Any],
    model_config: ChannelThermalHONFConfig,
    dataset_cfg: Dict[str, Any],
    local_checkpoint_provenance: Optional[str],
) -> Dict[str, Any]:
    """Write a concrete config with no auto-valued model fields."""

    resolved = dict(cfg)
    channel = model_config.channelthermal.to_dict()
    resolved["model"] = {
        "core_honf": model_config.core_honf.to_dict(),
        "channelthermal": {key: channel[key] for key in CHANNELTHERMAL_KEYS if key in channel},
        "local_coupling": {key: channel[key] for key in LOCAL_COUPLING_KEYS if key in channel},
        "physical_correction": {key: channel[key] for key in PHYSICAL_CORRECTION_KEYS if key in channel},
        "effective_model_config": model_config.to_dict(),
    }
    resolved["dataset"] = dict(dataset_cfg)
    resolved["dataset"]["normalize_inputs"] = bool(dataset_cfg.get("normalize_inputs", False))
    resolved["dataset"]["normalize_targets"] = bool(dataset_cfg.get("normalize_targets", False))
    resolved["local_checkpoint_provenance"] = local_checkpoint_provenance
    return resolved


def resolve_auto_internal_mode(model_config: ChannelThermalHONFConfig, model: ChannelThermalHONFModel) -> None:
    """Resolve auto internal mode."""

    if str(model_config.channelthermal.internal_prediction_mode) != "auto":
        return
    model_config.channelthermal.internal_prediction_mode = (
        "local_surrogate" if model.local_surrogate_attached else "global_head"
    )


def run_from_config(
    config: Dict[str, Any],
    args: argparse.Namespace,
    *,
    run_dir_override: Optional[Path] = None,
) -> int:
    """Run coupled training from an already composed configuration.

    The generic project entry point performs source-profile validation,
    resource resolution, launch confirmation, and run reservation before this
    case-specific workflow is entered.  A deep copy keeps automatic resolution
    and compatibility normalization local to this run.
    """

    cfg = copy.deepcopy(config)
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    checkpoint_cfg = cfg.get("checkpointing", {})
    ignored_organizer_keys = [
        key
        for key in loss_cfg
        if key.startswith("organizer_") and key not in {"organizer_regularization"}
        and float(loss_cfg.get(key, 0.0) or 0.0) != 0.0
    ]
    if ignored_organizer_keys:
        print(
            "[warning] Deprecated/legacy organizer losses are ignored; use "
            f"loss.organizer_regularization instead: {ignored_organizer_keys}"
        )
    set_seed(int(training_cfg.get("seed", 42)))
    device = select_device(args.device or training_cfg.get("device"))

    train_dataset = GlobalChannelThermalDataset(
        dataset_cfg.get("packed_h5_path", "./Case_ThermalChannel/Dataset/links/thermal_channel_global_v1.h5"),
        split=dataset_cfg.get("train_split", "train"),
        points_per_case=dataset_cfg.get("points_per_case", 4096),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=bool(dataset_cfg.get("random_point_sampling", True)),
        seed=int(training_cfg.get("seed", 42)),
        require_converged=bool(dataset_cfg.get("require_converged", False)),
    )
    val_dataset = GlobalChannelThermalDataset(
        dataset_cfg.get("packed_h5_path", "./Case_ThermalChannel/Dataset/links/thermal_channel_global_v1.h5"),
        split=dataset_cfg.get("val_split", "test"),
        points_per_case=dataset_cfg.get("val_points_per_case", dataset_cfg.get("points_per_case", 4096)),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        seed=int(training_cfg.get("seed", 42)) + 1000,
        require_converged=bool(dataset_cfg.get("require_converged", False)),
        normalizer=train_dataset.normalizer,
    )
    if len(val_dataset) == 0:
        if bool(dataset_cfg.get("allow_train_as_validation", False)):
            print("[warning] validation split is empty; explicitly reusing training data.")
            val_dataset = train_dataset
        else:
            raise RuntimeError(
                "Validation split is empty. Select a non-empty split or set "
                "case.dataset.allow_train_as_validation=true explicitly."
            )
    model_config = build_model_config(cfg, train_dataset)
    model = ChannelThermalHONFModel(model_config).to(device)
    model.set_global_target_normalization(train_dataset.normalizer.stats, normalize_targets=bool(dataset_cfg.get("normalize_targets", False)))
    resolve_auto_internal_mode(model_config, model)
    cfg = resolved_config_payload(
        cfg,
        model_config,
        dataset_cfg,
        model.local_coupling.local_surrogate_checkpoint_path or model_config.channelthermal.local_surrogate_checkpoint_path,
    )

    batch_size = int(dataset_cfg.get("batch_size", training_cfg.get("batch_size", 4)))
    val_batch_size = int(dataset_cfg.get("val_batch_size", batch_size))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    max_modules_per_batch = dataset_cfg.get("max_modules_per_batch")
    collator = ChannelThermalBatchCollator(
        dynamic_module_padding=bool(dataset_cfg.get("dynamic_module_padding", True)),
        max_modules_per_batch=None if max_modules_per_batch is None else int(max_modules_per_batch),
    )
    if bool(dataset_cfg.get("bucket_by_module_count", True)):
        train_batch_sampler = ModuleCountBucketBatchSampler(
            train_dataset.selected_module_counts,
            batch_size=batch_size,
            bucket_size_multiplier=int(dataset_cfg.get("module_count_bucket_size_multiplier", 8)),
            seed=int(training_cfg.get("seed", 42)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    optimizer, optimizer_group_inventory = build_forward_optimizer(model, training_cfg)
    scaler = make_grad_scaler(device, bool(training_cfg.get("amp", False)))
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 200))
    max_train_batches = args.max_train_batches if args.max_train_batches is not None else training_cfg.get("max_train_batches_per_epoch")
    max_val_batches = args.max_val_batches if args.max_val_batches is not None else training_cfg.get("max_val_batches")
    resume_checkpoint = resolve_demo_path(args.resume_checkpoint) if args.resume_checkpoint else None
    initialize_arg = getattr(args, "initialize_checkpoint", None)
    initialize_checkpoint = resolve_demo_path(initialize_arg) if initialize_arg else None
    if resume_checkpoint is not None and initialize_checkpoint is not None:
        raise ValueError("--resume-checkpoint and --initialize-checkpoint are mutually exclusive.")

    paths_cfg = cfg.get("paths", {})
    saved_root = ensure_dir(resolve_demo_path(paths_cfg.get("saved_model_dir", "./Trained_Results/ThermalChannel/HONF_Forward_Runs")))
    run_id = resolve_run_id(args.run_id, cfg, training_cfg)
    cfg["Run_ID"] = run_id
    cfg.setdefault("training", {})["Run_ID"] = run_id
    if run_dir_override is not None and resume_checkpoint is None:
        run_dir = ensure_dir(Path(run_dir_override).resolve())
    elif resume_checkpoint is None:
        suffix = sanitize_run_suffix(args.run_name or training_cfg.get("run_name"))
        stamp = current_timestamp()
        run_name = f"Run_{run_id}_{stamp}_{suffix}" if suffix else f"Run_{run_id}_{stamp}"
        run_dir = ensure_dir(saved_root / run_name)
    else:
        run_dir = ensure_dir(resume_checkpoint.parent)
    write_json(run_dir / "config_resolved.json", cfg)
    write_json(run_dir / "optimizer_group_inventory.json", optimizer_group_inventory)
    optimizer_inventory_announced = all(
        bool(group["scalar_count_complete"])
        for group in optimizer_group_inventory["groups"]
    )
    if optimizer_inventory_announced:
        _print_optimizer_group_inventory(optimizer_group_inventory)
    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "epoch",
        "loss_total",
        "loss_field",
        "loss_internal_temperature",
        "loss_interface",
        "loss_port_condition",
        "loss_port_smoothness",
        "loss_port_global_consistency",
        "loss_predicted_consistency",
        "loss_predicted_consistency_internal",
        "loss_predicted_consistency_interface",
        "loss_organizer",
        "effective_port_global_consistency_weight",
        "effective_predicted_consistency_weight",
        "effective_internal_temperature_weight",
        "effective_interface_weight",
        "field_mse",
        "temperature_mse",
        *INTERFACE_DIAGNOSTIC_KEYS,
        *HONF_DIAGNOSTIC_KEYS,
        *GRADIENT_DIAGNOSTIC_KEYS,
        "train_wall_seconds",
        "val_wall_seconds",
        "peak_cuda_memory_mb",
        "val_loss_total",
        "val_loss_field",
        "val_loss_internal_temperature",
        "val_loss_interface",
        "val_loss_port_condition",
        "val_loss_port_smoothness",
        "val_loss_port_global_consistency",
        "val_loss_predicted_consistency",
        "val_loss_predicted_consistency_internal",
        "val_loss_predicted_consistency_interface",
        "val_loss_organizer",
        "val_effective_port_global_consistency_weight",
        "val_effective_predicted_consistency_weight",
        "val_effective_internal_temperature_weight",
        "val_effective_interface_weight",
        "val_field_mse",
        "val_temperature_mse",
        *[f"val_{key}" for key in INTERFACE_DIAGNOSTIC_KEYS],
        *[f"val_{key}" for key in HONF_DIAGNOSTIC_KEYS],
        "val_predicted_loss_total",
        "val_predicted_field_mse",
        "val_predicted_temperature_mse",
    ]

    print(f"[setup] device={device}, train_cases={len(train_dataset)}, val_cases={len(val_dataset)}, params={count_parameters(model):,}")
    best_total = math.inf
    best_field = math.inf
    best_temperature = math.inf
    best_predicted = math.inf
    start_epoch = 1
    if initialize_checkpoint is not None:
        checkpoint = load_trusted_checkpoint(initialize_checkpoint, map_location=device)
        source_config = _validate_initialization_checkpoint(
            checkpoint,
            model=model,
            dataset=train_dataset,
            dataset_config=dataset_cfg,
        )
        # Materialize every lazy adapter/core parameter from the current data
        # schema before exact shape comparison. This is a no-grad inference
        # pass and does not restore or advance any checkpoint training state.
        materialization_batch = recursive_to_device(next(iter(train_loader)), device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(
                **make_model_inputs(
                    materialization_batch,
                    local_port_condition_mode="predicted",
                    mixed_teacher_ratio=0.0,
                    return_predicted_port_outputs=False,
                    return_port_global_consistency=False,
                )
            )
        model.train(was_training)
        initialization_inventory = _partial_initialize_model(
            model,
            checkpoint,
            source_config=source_config,
        )
        optimizer_group_inventory = refresh_optimizer_group_inventory(
            model,
            optimizer_group_inventory,
        )
        write_json(run_dir / "optimizer_group_inventory.json", optimizer_group_inventory)
        if not optimizer_inventory_announced:
            _print_optimizer_group_inventory(optimizer_group_inventory)
            optimizer_inventory_announced = True
        initialization_inventory.update(
            {
                "checkpoint_path": str(initialize_checkpoint),
                "checkpoint_sha256": _file_sha256(initialize_checkpoint),
                "source_field_assembly_mode": source_config.core_honf.field_assembly_mode,
                "target_field_assembly_mode": model_config.core_honf.field_assembly_mode,
                "source_organizer_mode": source_config.core_honf.organizer_mode,
                "target_organizer_mode": model_config.core_honf.organizer_mode,
            }
        )
        write_json(run_dir / "initialization_inventory.json", initialization_inventory)
        cfg["initialization"] = {
            "checkpoint_path": str(initialize_checkpoint),
            "checkpoint_sha256": initialization_inventory["checkpoint_sha256"],
            "inventory_path": str(run_dir / "initialization_inventory.json"),
        }
        write_json(run_dir / "config_resolved.json", cfg)
        print(
            f"[initialize] loaded {len(initialization_inventory['loaded'])} parameters from "
            f"{initialize_checkpoint}; skipped={len(initialization_inventory['skipped'])}, "
            f"missing={len(initialization_inventory['missing'])}, "
            f"unexpected={len(initialization_inventory['unexpected'])}"
        )
    if resume_checkpoint is not None:
        repair_metrics_csv_for_append(metrics_path)
        checkpoint = load_trusted_checkpoint(resume_checkpoint, map_location=device)
        _validate_resume_checkpoint(
            checkpoint,
            model=model,
            model_config=model_config,
            dataset=train_dataset,
            dataset_config=dataset_cfg,
        )
        _validate_optimizer_resume_compatibility(checkpoint, optimizer_group_inventory)
        model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
        optimizer_group_inventory = refresh_optimizer_group_inventory(
            model,
            optimizer_group_inventory,
        )
        write_json(run_dir / "optimizer_group_inventory.json", optimizer_group_inventory)
        if not optimizer_inventory_announced:
            _print_optimizer_group_inventory(optimizer_group_inventory)
            optimizer_inventory_announced = True
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state and scaler is not None:
            scaler.load_state_dict(scaler_state)
        checkpoint_epoch = int(checkpoint.get("epoch", checkpoint.get("current_epoch", 0)) or 0)
        best_payload = dict(checkpoint.get("best_metrics") or {})
        best_total = float(best_payload.get("best_val_loss_total", checkpoint.get("best_metric", math.inf)))
        best_field = float(best_payload.get("best_val_field_mse", math.inf))
        best_temperature = float(best_payload.get("best_val_temperature_mse", math.inf))
        best_predicted = float(best_payload.get("best_val_predicted_loss_total", math.inf))
        start_epoch = checkpoint_epoch + 1
        _restore_rng_state(checkpoint)
        print(f"[resume] loaded {resume_checkpoint}; continuing at epoch {start_epoch} / {epochs}")

    total_train_seconds = 0.0
    total_val_seconds = 0.0
    peak_cuda_memory_mb = 0.0
    for epoch in range(start_epoch, epochs + 1):
        model.set_training_progress(epoch=epoch, total_epochs=epochs)
        train_dataset.set_epoch(epoch)
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        effective_mode, effective_ratio = effective_port_condition_settings(epoch, training_cfg)
        eff_internal, eff_interface = effective_local_loss_weights(loss_cfg, effective_mode, effective_ratio)
        pred_consistency_weight = predicted_consistency_weight_for_epoch(epoch, loss_cfg)
        gradient_clip_norm = float(training_cfg.get("gradient_clip_norm", 0.0) or 0.0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            loss_cfg,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(training_cfg.get("amp", False)),
            max_batches=None if max_train_batches is None else int(max_train_batches),
            local_port_condition_mode=effective_mode,
            mixed_teacher_ratio=effective_ratio,
            effective_internal_temperature_weight=eff_internal,
            effective_interface_weight=eff_interface,
            predicted_consistency_weight=pred_consistency_weight,
            gradient_clip_norm=gradient_clip_norm,
            record_gradient_diagnostics=(epoch in {1, 2, 5, 10, 20} or epoch % 50 == 0),
        )
        train_wall_seconds = time.perf_counter() - train_started
        total_train_seconds += train_wall_seconds
        optimizer_group_inventory = refresh_optimizer_group_inventory(
            model,
            optimizer_group_inventory,
        )
        write_json(run_dir / "optimizer_group_inventory.json", optimizer_group_inventory)
        if not optimizer_inventory_announced:
            _print_optimizer_group_inventory(optimizer_group_inventory)
            optimizer_inventory_announced = True
        val_started = time.perf_counter()
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            loss_cfg,
            optimizer=None,
            scaler=None,
            amp=bool(training_cfg.get("amp", False)),
            max_batches=None if max_val_batches is None else int(max_val_batches),
            local_port_condition_mode=effective_mode,
            mixed_teacher_ratio=effective_ratio,
            effective_internal_temperature_weight=eff_internal,
            effective_interface_weight=eff_interface,
            predicted_consistency_weight=pred_consistency_weight,
            gradient_clip_norm=gradient_clip_norm,
        )
        if reuses_primary_validation_for_predicted_mode(effective_mode):
            predicted_val_metrics = val_metrics
        else:
            predicted_val_metrics = run_epoch(
                model,
                val_loader,
                device,
                loss_cfg,
                optimizer=None,
                scaler=None,
                amp=bool(training_cfg.get("amp", False)),
                max_batches=None if max_val_batches is None else int(max_val_batches),
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.0,
                effective_internal_temperature_weight=float(loss_cfg.get("internal_temperature_weight", 1.0)),
                effective_interface_weight=float(loss_cfg.get("interface_weight", 0.2)),
                predicted_consistency_weight=0.0,
                gradient_clip_norm=gradient_clip_norm,
            )
        val_wall_seconds = time.perf_counter() - val_started
        total_val_seconds += val_wall_seconds
        epoch_peak_memory_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )
        peak_cuda_memory_mb = max(peak_cuda_memory_mb, epoch_peak_memory_mb)
        row = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "val_predicted_loss_total": predicted_val_metrics.get("loss_total", math.nan),
            "val_predicted_field_mse": predicted_val_metrics.get("field_mse", math.nan),
            "val_predicted_temperature_mse": predicted_val_metrics.get("temperature_mse", math.nan),
            "train_wall_seconds": train_wall_seconds,
            "val_wall_seconds": val_wall_seconds,
            "peak_cuda_memory_mb": epoch_peak_memory_mb,
        }
        write_metrics_row(metrics_path, fieldnames, row)
        total_metric = float(row["val_loss_total"])
        field_metric = float(row["val_field_mse"])
        temp_metric = float(row["val_temperature_mse"])
        if math.isfinite(total_metric) and total_metric < best_total:
            best_total = total_metric
            if bool(checkpoint_cfg.get("save_best", True)):
                save_checkpoint(run_dir / "best_model.pt", model=model, model_config=model_config, train_config=cfg, dataset=train_dataset, epoch=epoch, best_metric=best_total, optimizer=optimizer, scaler=scaler, best_metrics=best_metrics_payload(row, best_total, best_field, best_temperature, best_predicted), optimizer_group_inventory=optimizer_group_inventory)
        if math.isfinite(field_metric) and field_metric < best_field:
            best_field = field_metric
            if bool(checkpoint_cfg.get("save_best_field_mse", True)):
                save_checkpoint(run_dir / "best_by_field_mse_model.pt", model=model, model_config=model_config, train_config=cfg, dataset=train_dataset, epoch=epoch, best_metric=best_field, optimizer=optimizer, scaler=scaler, best_metrics=best_metrics_payload(row, best_total, best_field, best_temperature, best_predicted), optimizer_group_inventory=optimizer_group_inventory)
        if math.isfinite(temp_metric) and temp_metric < best_temperature:
            best_temperature = temp_metric
            if bool(checkpoint_cfg.get("save_best_temperature_mse", True)):
                save_checkpoint(run_dir / "best_by_temperature_mse_model.pt", model=model, model_config=model_config, train_config=cfg, dataset=train_dataset, epoch=epoch, best_metric=best_temperature, optimizer=optimizer, scaler=scaler, best_metrics=best_metrics_payload(row, best_total, best_field, best_temperature, best_predicted), optimizer_group_inventory=optimizer_group_inventory)
        predicted_metric = float(row["val_predicted_loss_total"])
        if math.isfinite(predicted_metric) and predicted_metric < best_predicted:
            best_predicted = predicted_metric
            if bool(checkpoint_cfg.get("save_best_predicted", True)):
                save_checkpoint(run_dir / "best_predicted_model.pt", model=model, model_config=model_config, train_config=cfg, dataset=train_dataset, epoch=epoch, best_metric=best_predicted, optimizer=optimizer, scaler=scaler, best_metrics=best_metrics_payload(row, best_total, best_field, best_temperature, best_predicted), optimizer_group_inventory=optimizer_group_inventory)
        if should_save_latest_checkpoint(epoch, epochs, checkpoint_cfg):
            save_checkpoint(run_dir / "latest_model.pt", model=model, model_config=model_config, train_config=cfg, dataset=train_dataset, epoch=epoch, best_metric=best_total, optimizer=optimizer, scaler=scaler, best_metrics=best_metrics_payload(row, best_total, best_field, best_temperature, best_predicted), optimizer_group_inventory=optimizer_group_inventory)
        if should_save_milestone_checkpoint(epoch, checkpoint_cfg):
            save_checkpoint(run_dir / f"epoch_{epoch:04d}_model.pt", model=model, model_config=model_config, train_config=cfg, dataset=train_dataset, epoch=epoch, best_metric=best_total, optimizer=optimizer, scaler=scaler, best_metrics=best_metrics_payload(row, best_total, best_field, best_temperature, best_predicted), optimizer_group_inventory=optimizer_group_inventory)
        plot_every = max(int(training_cfg.get("plot_every_epochs", 10)), 1)
        if epoch % plot_every == 0 or epoch == epochs:
            save_global_loss_plots(metrics_path, run_dir)
        print(
            f"[epoch {epoch:04d}] loss={row['loss_total']:.4e} field={row['field_mse']:.4e} "
            f"internal={row['loss_internal_temperature']:.4e} interface={row['loss_interface']:.4e} "
            f"port={row['loss_port_condition']:.4e} port_global={row['loss_port_global_consistency']:.4e} "
            f"pred_cons={row['loss_predicted_consistency']:.4e} mode={effective_mode} ratio={effective_ratio:.3f} "
            f"val={row['val_loss_total']:.4e} val_field={row['val_field_mse']:.4e} "
            f"val_temp={row['val_temperature_mse']:.4e} val_pred={row['val_predicted_loss_total']:.4e}"
        )

    write_json(
        run_dir / "summary.json",
        {
            "stage": "channelthermal_prompt3_honf_physical_coupling",
            "run_dir": str(run_dir),
            "best_val_loss_total": best_total,
            "best_val_field_mse": best_field,
            "best_val_temperature_mse": best_temperature,
            "best_val_predicted_loss_total": best_predicted,
            "epochs": epochs,
            "train_cases": len(train_dataset),
            "val_cases": len(val_dataset),
            "model_config": model_config.to_dict(),
            "optimizer_group_inventory": optimizer_group_inventory,
            "trainable_parameter_count": count_parameters(model),
            "actual_train_wall_seconds": total_train_seconds,
            "actual_validation_wall_seconds": total_val_seconds,
            "actual_total_epoch_wall_seconds": total_train_seconds + total_val_seconds,
            "peak_cuda_memory_mb": peak_cuda_memory_mb,
        },
    )
    print(f"[done] saved global HONF-CL physical-coupling run: {run_dir}")
    return 0


def main() -> int:
    """Run the legacy direct CLI using one combined JSON configuration."""

    args = parse_args()
    cfg = read_json(resolve_config_path(args.config))
    return run_from_config(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
