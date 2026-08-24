"""Strict forward checkpoint save, resume, and partial initialization mechanics."""

from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.model import ChannelThermalHONFModel
from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.compat import strip_module_prefix


def save_checkpoint(
    path: Path,
    *,
    model: ChannelThermalHONFModel,
    model_config: ChannelThermalHONFConfig,
    train_config: Dict[str, Any],
    dataset: GlobalChannelThermalDataset,
    epoch: int,
    best_metric: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Any = None,
    best_metrics: Optional[Dict[str, float]] = None,
    optimizer_group_inventory: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically save model, optimizer, schema, and normalization metadata."""

    local = model.local_coupling
    def buffer_payload(name: str):
        """Copy one non-empty normalization buffer into checkpoint-safe NumPy data."""

        value = getattr(local, name)
        return value.detach().cpu().numpy().copy() if torch.is_tensor(value) and value.numel() > 0 else None
    local_stats = {
        key: value
        for key, value in {
            "module_params_mean": buffer_payload("local_module_params_mean"),
            "module_params_std": buffer_payload("local_module_params_std"),
            "port_tokens_mean": buffer_payload("local_port_tokens_mean"),
            "port_tokens_std": buffer_payload("local_port_tokens_std"),
            "internal_temperature_mean": buffer_payload("local_internal_temperature_mean"),
            "internal_temperature_std": buffer_payload("local_internal_temperature_std"),
            "interface_targets_mean": buffer_payload("local_interface_targets_mean"),
            "interface_targets_std": buffer_payload("local_interface_targets_std"),
        }.items()
        if value is not None
    }
    local_model_config = None
    if local.local_surrogate is not None:
        local_model_config = local.local_surrogate.config.to_dict()
    config_payload = model_config.to_dict()
    if config_payload.get("channelthermal", {}).get("internal_prediction_mode") == "auto":
        config_payload["channelthermal"]["internal_prediction_mode"] = (
            "local_surrogate" if model.local_surrogate_attached else "global_head"
        )
    payload = {
        "checkpoint_schema_version": 1,
        "case_id": "ThermalChannel",
        "model_family": "honf_forward",
        "workflow": "forward",
        "stage": "channelthermal_honf_cl_physical_coupling",
            "epoch": int(epoch),
            "current_epoch": int(epoch),
            "best_metric": float(best_metric),
            "best_metrics": dict(best_metrics or {}),
            "model_config": config_payload,
            "model_state_dict": model.state_dict(),
            "selection_state": model.selection_state(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "optimizer_group_inventory": copy.deepcopy(optimizer_group_inventory),
            "scaler_state_dict": None if scaler is None else scaler.state_dict(),
            "train_config": train_config,
            "channel_order": list(dataset.channel_order),
            "field_dim": int(dataset.field_dim),
            "interface_condition_feature_names": list(dataset.interface_condition_feature_names),
            "interface_target_names": list(dataset.interface_target_names),
            "dataset_id": train_config.get("dataset", {}).get("dataset_id"),
            "dataset_schema": train_config.get("dataset", {}).get("dataset_schema"),
            "dataset_fingerprint": train_config.get("dataset", {}).get("dataset_fingerprint"),
            "dataset_metadata": {
                "max_num_modules": int(dataset.max_num_modules),
                "selected_module_count_min": min(dataset.selected_module_counts, default=0),
                "selected_module_count_max": max(dataset.selected_module_counts, default=0),
            },
            "feature_schemas": {
                "channel_order": list(dataset.channel_order),
                "interface_condition_feature_names": list(dataset.interface_condition_feature_names),
                "interface_target_names": list(dataset.interface_target_names),
                "module_feature_names": list(model.input_adapter.feature_names),
                "global_context_names": list(model.input_adapter.global_context_names),
            },
            "global_normalization_config": {
                "normalize_inputs": bool(train_config.get("dataset", {}).get("normalize_inputs", False)),
                "normalize_targets": bool(train_config.get("dataset", {}).get("normalize_targets", False)),
            },
            "global_normalization_stats": {name: value.copy() for name, value in dataset.normalizer.stats.items()},
            "local_surrogate_checkpoint_path": model_config.channelthermal.local_surrogate_checkpoint_path,
            "local_checkpoint_provenance": local.local_surrogate_checkpoint_path or model_config.channelthermal.local_surrogate_checkpoint_path,
            "local_model_config": local_model_config,
            "local_surrogate_frozen": bool(local.local_surrogate_frozen),
            "local_normalization_config": {
                "normalize_inputs": bool(local.local_surrogate_normalize_inputs),
                "normalize_targets": bool(local.local_surrogate_normalize_targets),
            },
            "local_normalization_stats": local_stats,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _restore_rng_state(checkpoint: Dict[str, Any]) -> None:
    """Restore optional random streams from a current forward checkpoint."""

    state = dict(checkpoint.get("rng_state") or {})
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _validate_resume_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    model: ChannelThermalHONFModel,
    model_config: ChannelThermalHONFConfig,
    dataset: GlobalChannelThermalDataset,
    dataset_config: Dict[str, Any],
) -> None:
    """Reject architecture, schema, or normalization drift before resume."""

    validate_checkpoint_identity(
        checkpoint,
        case_id="ThermalChannel",
        model_family="honf_forward",
        workflow="forward",
    )
    current_config = model_config.to_dict()
    if current_config["channelthermal"].get("internal_prediction_mode") == "auto":
        current_config["channelthermal"]["internal_prediction_mode"] = (
            "local_surrogate" if model.local_surrogate_attached else "global_head"
        )
    saved_config = dict(checkpoint.get("model_config") or {})
    normalized_saved_config = ChannelThermalHONFConfig.from_dict(saved_config).to_dict() if saved_config else {}
    if normalized_saved_config and normalized_saved_config != current_config:
        raise ValueError("Resume checkpoint forward architecture/configuration does not match this launch.")
    expected_schemas = {
        "channel_order": list(dataset.channel_order),
        "interface_condition_feature_names": list(dataset.interface_condition_feature_names),
        "interface_target_names": list(dataset.interface_target_names),
    }
    for key, expected in expected_schemas.items():
        saved = checkpoint.get(key)
        if saved is not None and list(saved) != expected:
            raise ValueError(f"Resume checkpoint schema mismatch for {key!r}.")
    saved_fingerprint = checkpoint.get("dataset_fingerprint")
    expected_fingerprint = dataset_config.get("dataset_fingerprint")
    if saved_fingerprint and expected_fingerprint and saved_fingerprint != expected_fingerprint:
        raise ValueError("Resume checkpoint dataset fingerprint does not match the selected dataset.")
    saved_stats = dict(checkpoint.get("global_normalization_stats") or {})
    for name, saved_value in saved_stats.items():
        current = dataset.normalizer.stats.get(name)
        if current is None or not np.allclose(
            np.asarray(saved_value), np.asarray(current), rtol=1.0e-7, atol=1.0e-7
        ):
            raise ValueError(f"Resume checkpoint normalization mismatch for {name!r}.")


def _file_sha256(path: Path) -> str:
    """Return the immutable digest recorded for initialization provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_initialization_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    model: ChannelThermalHONFModel,
    dataset: GlobalChannelThermalDataset,
    dataset_config: Dict[str, Any],
) -> ChannelThermalHONFConfig:
    """Validate provenance required for safe partial parameter initialization."""

    validate_checkpoint_identity(
        checkpoint,
        case_id="ThermalChannel",
        model_family="honf_forward",
        workflow="forward",
    )
    source_config_payload = dict(checkpoint.get("model_config") or {})
    if not source_config_payload:
        raise ValueError("Initialization checkpoint is missing model_config provenance.")
    source_config = ChannelThermalHONFConfig.from_dict(source_config_payload)

    expected_schemas = {
        "channel_order": list(dataset.channel_order),
        "interface_condition_feature_names": list(dataset.interface_condition_feature_names),
        "interface_target_names": list(dataset.interface_target_names),
    }
    for key, expected in expected_schemas.items():
        saved = checkpoint.get(key)
        if saved is None or list(saved) != expected:
            raise ValueError(f"Initialization checkpoint field schema mismatch for {key!r}.")
    saved_field_dim = checkpoint.get("field_dim", source_config.core_honf.field_dim)
    if int(saved_field_dim) != int(dataset.field_dim):
        raise ValueError("Initialization checkpoint field_dim does not match the selected field schema.")

    for key in ("dataset_id", "dataset_schema", "dataset_fingerprint"):
        expected = dataset_config.get(key)
        saved = checkpoint.get(key)
        if expected is not None and (saved is None or saved != expected):
            raise ValueError(f"Initialization checkpoint dataset identity mismatch for {key!r}.")

    expected_global_config = {
        "normalize_inputs": bool(dataset_config.get("normalize_inputs", False)),
        "normalize_targets": bool(dataset_config.get("normalize_targets", False)),
    }
    saved_global_config = checkpoint.get("global_normalization_config")
    if not isinstance(saved_global_config, dict) or {
        key: bool(saved_global_config.get(key, False)) for key in expected_global_config
    } != expected_global_config:
        raise ValueError("Initialization checkpoint global normalization configuration mismatch.")
    saved_stats = checkpoint.get("global_normalization_stats")
    if not isinstance(saved_stats, dict) or set(saved_stats) != set(dataset.normalizer.stats):
        raise ValueError("Initialization checkpoint global normalization statistics inventory mismatch.")
    for name, current in dataset.normalizer.stats.items():
        if not np.allclose(
            np.asarray(saved_stats[name]), np.asarray(current), rtol=1.0e-7, atol=1.0e-7
        ):
            raise ValueError(f"Initialization checkpoint normalization mismatch for {name!r}.")

    expected_local_config = {
        "normalize_inputs": bool(model.local_coupling.local_surrogate_normalize_inputs),
        "normalize_targets": bool(model.local_coupling.local_surrogate_normalize_targets),
    }
    saved_local_config = checkpoint.get("local_normalization_config")
    if model.local_surrogate_attached and (
        not isinstance(saved_local_config, dict)
        or {key: bool(saved_local_config.get(key, False)) for key in expected_local_config}
        != expected_local_config
    ):
        raise ValueError("Initialization checkpoint local normalization configuration mismatch.")
    if model.local_surrogate_attached:
        local_buffer_names = {
            "module_params_mean": "local_module_params_mean",
            "module_params_std": "local_module_params_std",
            "port_tokens_mean": "local_port_tokens_mean",
            "port_tokens_std": "local_port_tokens_std",
            "internal_temperature_mean": "local_internal_temperature_mean",
            "internal_temperature_std": "local_internal_temperature_std",
            "interface_targets_mean": "local_interface_targets_mean",
            "interface_targets_std": "local_interface_targets_std",
        }
        expected_local_stats = {
            public_name: getattr(model.local_coupling, buffer_name).detach().cpu().numpy()
            for public_name, buffer_name in local_buffer_names.items()
            if getattr(model.local_coupling, buffer_name).numel() > 0
        }
        saved_local_stats = checkpoint.get("local_normalization_stats")
        if not isinstance(saved_local_stats, dict) or set(saved_local_stats) != set(expected_local_stats):
            raise ValueError("Initialization checkpoint local normalization statistics inventory mismatch.")
        for name, current in expected_local_stats.items():
            if not np.allclose(
                np.asarray(saved_local_stats[name]), current, rtol=1.0e-7, atol=1.0e-7
            ):
                raise ValueError(f"Initialization checkpoint local normalization mismatch for {name!r}.")
    return source_config


def _partial_initialize_model(
    model: ChannelThermalHONFModel,
    checkpoint: Dict[str, Any],
    *,
    source_config: ChannelThermalHONFConfig,
) -> Dict[str, Any]:
    """Load only compatible parameters and return a complete key inventory."""

    source_state = strip_module_prefix(checkpoint["model_state_dict"])
    target_parameters = dict(model.named_parameters())
    target_state = model.state_dict()
    cross_assembly = (
        source_config.core_honf.field_assembly_mode == "context_fusion"
        and model.config.core_honf.field_assembly_mode == "edge_additive"
    )
    fixed_to_exchangeable = (
        source_config.core_honf.organizer_mode == "fixed_projection"
        and model.config.core_honf.organizer_mode == "exchangeable_slots"
    )
    physical_output_prefixes = (
        "core.decoder.pred_head.",
        "core.decoder.mean_head.",
        "core.decoder.residual_head.",
        "fallback_heads.internal_head.",
        "fallback_heads.interface_head.",
        "local_coupling.port_refinement_head.",
    )
    runtime_state_suffixes = (
        "._selection_epoch_state",
        "._selection_total_epochs_state",
    )
    loaded: list[str] = []
    skipped: list[Dict[str, Any]] = []
    unexpected: list[str] = []
    loadable: Dict[str, torch.Tensor] = {}

    for name, source_value in source_state.items():
        if name.endswith(runtime_state_suffixes):
            skipped.append({"key": name, "reason": "runtime_selection_state"})
            continue
        if fixed_to_exchangeable and name.startswith("core.organizer."):
            skipped.append({"key": name, "reason": "fixed_projection_organizer"})
            continue
        if cross_assembly and name.startswith(physical_output_prefixes):
            skipped.append({"key": name, "reason": "context_to_additive_output_head"})
            continue
        target_value = target_parameters.get(name)
        if target_value is None:
            if name in target_state:
                skipped.append({"key": name, "reason": "non_parameter_state"})
            else:
                unexpected.append(name)
            continue
        try:
            target_shape = tuple(target_value.shape)
        except (RuntimeError, ValueError):
            target_shape = None
        source_shape = tuple(source_value.shape)
        if target_shape is None:
            skipped.append({"key": name, "reason": "uninitialized_target_shape"})
            continue
        if target_shape is not None and source_shape != target_shape:
            skipped.append(
                {
                    "key": name,
                    "reason": "shape_mismatch",
                    "source_shape": list(source_shape),
                    "target_shape": list(target_shape),
                }
            )
            continue
        loadable[name] = source_value
        loaded.append(name)

    model.load_state_dict(loadable, strict=False)
    missing = sorted(name for name in target_parameters if name not in loadable)
    return {
        "loaded": sorted(loaded),
        "skipped": sorted(skipped, key=lambda item: str(item["key"])),
        "missing": missing,
        "unexpected": sorted(unexpected),
        "source_parameter_or_state_count": len(source_state),
        "target_parameter_count": len(target_parameters),
        "source_organizer_mode": source_config.core_honf.organizer_mode,
        "target_organizer_mode": model.config.core_honf.organizer_mode,
    }

