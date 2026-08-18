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
import math
import random
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
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Resume training from a saved HONF-CL checkpoint.")
    return parser.parse_args()


def normalize_run_id(value: Any, fallback: str = "0001") -> str:
    """Normalize run id."""

    raw = str(value or fallback).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be a numeric serial such as '0001'; got {raw!r}.")
    return f"{int(raw):04d}"


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


def organizer_regularization(output: Dict[str, Any], loss_cfg: Dict[str, Any]) -> torch.Tensor:
    """Perform the organizer regularization operation used by this module."""

    return organizer_regularization_loss(output, loss_cfg.get("organizer_regularization", {}))


def internal_loss(output: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
    """Perform the internal loss operation used by this module."""

    pred = output["pred_internal_temperature"]
    if pred.numel() == 0 or pred.shape[-2] == 0:
        return output["pred_field"].new_zeros(())
    target = batch["module_internal_temperature_points"].float().unsqueeze(-1)
    mask = batch["structure"]["module_present"].float()[:, :, None, None]
    return ((pred - target).square() * mask).sum() / (mask.sum() * pred.new_tensor(float(pred.shape[-2]))).clamp_min(1.0e-6)


def interface_loss(output: Dict[str, Any], batch: Dict[str, Any], loss_cfg: Dict[str, Any]) -> torch.Tensor:
    """Perform the interface loss operation used by this module."""

    pred = output["pred_interface"]
    if pred.numel() == 0 or pred.shape[-2] == 0:
        return output["pred_field"].new_zeros(())
    target = batch["interface_target"].float()
    weights = pred.new_ones(pred.shape[-1])
    if loss_cfg.get("interface_target_weights") is not None:
        custom = torch.as_tensor(loss_cfg["interface_target_weights"], device=pred.device, dtype=pred.dtype)
        weights[: min(custom.numel(), pred.shape[-1])] = custom[: pred.shape[-1]]
    mask = batch["structure"]["module_present"].float()[:, :, None, None]
    loss_type = str(loss_cfg.get("interface_loss_type", "mse")).lower()
    if loss_type == "smooth_l1":
        per_value = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none") * weights
    elif loss_type == "mse":
        per_value = (pred - target).square() * weights
    else:
        raise ValueError(f"interface_loss_type must be 'mse' or 'smooth_l1', got {loss_type!r}.")
    return (per_value * mask).sum() / (mask.sum() * pred.new_tensor(float(pred.shape[-2] * pred.shape[-1]))).clamp_min(1.0e-6)


def port_condition_loss(output: Dict[str, Any], batch: Dict[str, Any], loss_cfg: Dict[str, Any]) -> torch.Tensor:
    """Perform the port condition loss operation used by this module."""

    pred = output["pred_port_condition"]
    target = batch.get("teacher_port_tokens")
    if target is None or pred.numel() == 0 or pred.shape[-2] == 0:
        return output["pred_field"].new_zeros(())
    pred_values = pred[..., 3:5]
    target_values = target.float()[..., 3:5]
    module_mask = batch["structure"]["module_present"].float()[:, :, None, None]
    valid_h = batch.get("interface_condition_valid_mask")
    t_scale = max(float(loss_cfg.get("port_temperature_scale", 10.0)), 1.0e-6)
    loss_t = ((pred_values[..., 0:1] / t_scale - target_values[..., 0:1] / t_scale).square() * module_mask).sum()
    loss_t = loss_t / (module_mask.sum() * pred.new_tensor(float(pred.shape[-2]))).clamp_min(1.0e-6)
    h_mask = module_mask if valid_h is None else module_mask * valid_h.float().unsqueeze(-1)
    pred_h = torch.log1p(pred_values[..., 1:2].clamp_min(0.0))
    target_h = torch.log1p(target_values[..., 1:2].clamp_min(0.0))
    h_loss_type = str(loss_cfg.get("port_h_loss_type", "mse")).lower()
    if h_loss_type == "smooth_l1":
        h_error = torch.nn.functional.smooth_l1_loss(pred_h, target_h, reduction="none")
    elif h_loss_type == "mse":
        h_error = (pred_h - target_h).square()
    else:
        raise ValueError(f"port_h_loss_type must be 'mse' or 'smooth_l1', got {h_loss_type!r}.")
    loss_h = (h_error * h_mask).sum() / h_mask.sum().clamp_min(1.0e-6)
    return float(loss_cfg.get("port_temperature_weight", 1.0)) * loss_t + float(loss_cfg.get("port_h_weight", 1.0)) * loss_h


def port_cyclic_smoothness_loss(output: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
    """Perform the port cyclic smoothness loss operation used by this module."""

    pred = output["pred_port_condition"]
    if pred.numel() == 0 or pred.shape[-2] <= 1:
        return output["pred_field"].new_zeros(())
    values = pred[..., 3:5]
    signal = torch.cat([values[..., 0:1], torch.log1p(values[..., 1:2].clamp_min(0.0))], dim=-1)
    diff = signal - torch.roll(signal, shifts=-1, dims=-2)
    mask = batch["structure"]["module_present"].float()[:, :, None, None]
    return (diff.abs() * mask).sum() / (mask.sum() * pred.new_tensor(float(pred.shape[-2] * 2))).clamp_min(1.0e-6)


def port_global_consistency_loss(output: Dict[str, Any]) -> torch.Tensor:
    """Perform the port global consistency loss operation used by this module."""

    if "pred_port_global_temperature" not in output:
        return output["pred_field"].new_zeros(())
    pred = output["pred_port_global_temperature"][..., None]
    target = output["pred_port_global_temperature_target"][..., None]
    mask = output.get("pred_port_global_consistency_mask")
    if mask is None:
        return (pred - target).square().mean()
    mask = mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
    return ((pred - target).square() * mask).sum() / mask.sum().clamp_min(1.0e-6)


def effective_port_global_weight(loss_cfg: Dict[str, Any], mode: str, mixed_teacher_ratio: float) -> float:
    """Perform the effective port global weight operation used by this module."""

    mode = str(mode).lower()
    if mode == "teacher":
        return float(loss_cfg.get("port_global_consistency_teacher_weight", 0.0))
    if mode == "mixed":
        return float(loss_cfg.get("port_global_consistency_weight", 0.0)) * max(0.0, 1.0 - float(mixed_teacher_ratio))
    return float(loss_cfg.get("port_global_consistency_weight", 0.0))


def predicted_consistency_weight_for_epoch(epoch: int, loss_cfg: Dict[str, Any]) -> float:
    """Perform the predicted consistency weight for epoch operation used by this module."""

    base = float(loss_cfg.get("predicted_consistency_weight", 0.0))
    warmup = max(int(loss_cfg.get("predicted_consistency_warmup_epochs", 1)), 1)
    return base * min(max(float(epoch) / float(warmup), 0.0), 1.0)


def effective_port_condition_settings(epoch: int, training_cfg: Dict[str, Any]) -> tuple[str, float]:
    """Perform the effective port condition settings operation used by this module."""

    curriculum = training_cfg.get("port_curriculum", {}) if isinstance(training_cfg.get("port_curriculum"), dict) else {}
    schedule = str(curriculum.get("schedule", training_cfg.get("port_condition_schedule", "none"))).lower()
    base_mode = str(curriculum.get("mode", training_cfg.get("local_port_condition_mode", "predicted"))).lower()
    base_ratio = float(curriculum.get("mixed_teacher_ratio", training_cfg.get("mixed_teacher_ratio", 0.5)))
    if schedule == "none":
        return base_mode, base_ratio
    if schedule != "teacher_to_predicted":
        raise ValueError(f"Unsupported port curriculum schedule={schedule!r}.")
    teacher_epochs = int(curriculum.get("teacher_epochs", training_cfg.get("teacher_epochs", 50)))
    predicted_after = int(curriculum.get("predicted_after_epoch", training_cfg.get("predicted_after_epoch", teacher_epochs + 100)))
    ratio_start = float(curriculum.get("mixed_teacher_ratio_start", training_cfg.get("mixed_teacher_ratio_start", 1.0)))
    ratio_end = float(curriculum.get("mixed_teacher_ratio_end", training_cfg.get("mixed_teacher_ratio_end", 0.0)))
    if int(epoch) <= teacher_epochs:
        return "teacher", 1.0
    if int(epoch) <= predicted_after:
        span = max(predicted_after - teacher_epochs, 1)
        progress = (int(epoch) - teacher_epochs) / span
        ratio = ratio_start + (ratio_end - ratio_start) * progress
        return "mixed", float(min(max(ratio, 0.0), 1.0))
    return "predicted", 0.0


def effective_local_loss_weights(loss_cfg: Dict[str, Any], mode: str, mixed_teacher_ratio: float) -> tuple[float, float]:
    """Perform the effective local loss weights operation used by this module."""

    base_internal = float(loss_cfg.get("internal_temperature_weight", 1.0))
    base_interface = float(loss_cfg.get("interface_weight", 0.2))
    mode = str(mode).lower()
    if mode == "teacher":
        scale = 0.0
    elif mode == "mixed":
        scale = 1.0 - float(mixed_teacher_ratio)
    else:
        scale = 1.0
    scale = min(max(scale, 0.0), 1.0)
    return base_internal * scale, base_interface * scale


def make_model_inputs(
    batch: Dict[str, Any],
    *,
    local_port_condition_mode: str,
    mixed_teacher_ratio: float,
    return_predicted_port_outputs: bool = False,
    return_port_global_consistency: bool = False,
) -> Dict[str, Any]:
    """Create model inputs."""

    return {
        "structure": batch["structure"],
        "query_xy": batch["query_xy"],
        "interface_condition": batch.get("interface_condition"),
        "local_module_params": batch.get("local_module_params"),
        "teacher_port_tokens": batch.get("teacher_port_tokens"),
        "local_query_points": batch.get("module_internal_query_points"),
        "local_port_condition_mode": local_port_condition_mode,
        "mixed_teacher_ratio": mixed_teacher_ratio,
        "return_predicted_port_outputs": return_predicted_port_outputs,
        "return_port_global_consistency": return_port_global_consistency,
    }


def run_epoch(
    model: ChannelThermalHONFModel,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: Dict[str, Any],
    *,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Any,
    amp: bool,
    max_batches: Optional[int],
    local_port_condition_mode: str,
    mixed_teacher_ratio: float,
    effective_internal_temperature_weight: float,
    effective_interface_weight: float,
    predicted_consistency_weight: float,
    gradient_clip_norm: float = 0.0,
) -> Dict[str, float]:
    """Run one train/validation epoch and return averaged loss/diagnostic scalars."""

    training = optimizer is not None
    model.train(training)
    sums: Dict[str, float] = {}
    count = 0
    iterator = tqdm(loader, leave=False, desc="train" if training else "val")
    for batch_idx, batch in enumerate(iterator, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        batch = recursive_to_device(batch, device)
        target = batch["field_targets"].float()
        point_weights = batch.get("point_weights")
        with torch.set_grad_enabled(training):
            with autocast_context(device, amp):
                port_global_weight = effective_port_global_weight(loss_cfg, local_port_condition_mode, mixed_teacher_ratio)
                output = model(
                    **make_model_inputs(
                        batch,
                        local_port_condition_mode=local_port_condition_mode,
                        mixed_teacher_ratio=mixed_teacher_ratio,
                        return_predicted_port_outputs=bool(predicted_consistency_weight > 0.0),
                        return_port_global_consistency=bool(port_global_weight != 0.0),
                    )
                )
                loss_field = channelthermal_field_mse(
                    output["pred_field"],
                    target,
                    loss_cfg,
                    field_names=model.config.channelthermal.field_names,
                    point_weights=point_weights,
                )
                zero = output["pred_field"].new_zeros(())
                loss_internal = internal_loss(output, batch) if effective_internal_temperature_weight != 0.0 else zero
                loss_interface = interface_loss(output, batch, loss_cfg) if effective_interface_weight != 0.0 else zero
                port_supervised_weight = float(loss_cfg.get("port_supervised_weight", loss_cfg.get("port_condition_weight", 0.0)))
                port_smoothness_weight = float(loss_cfg.get("port_smoothness_weight", 0.0))
                loss_port = port_condition_loss(output, batch, loss_cfg) if port_supervised_weight != 0.0 else zero
                loss_port_smoothness = port_cyclic_smoothness_loss(output, batch) if port_smoothness_weight != 0.0 else zero
                loss_port_global = port_global_consistency_loss(output) if port_global_weight != 0.0 else zero
                if "predicted_port_internal_temperature" in output and "predicted_port_interface" in output:
                    pred_cons_internal = internal_loss(
                        {"pred_internal_temperature": output["predicted_port_internal_temperature"], "pred_field": output["pred_field"]},
                        batch,
                    )
                    pred_cons_interface = interface_loss(
                        {"pred_interface": output["predicted_port_interface"], "pred_field": output["pred_field"]},
                        batch,
                        loss_cfg,
                    )
                    loss_predicted_consistency = pred_cons_internal + pred_cons_interface
                else:
                    pred_cons_internal = output["pred_field"].new_zeros(())
                    pred_cons_interface = output["pred_field"].new_zeros(())
                    loss_predicted_consistency = output["pred_field"].new_zeros(())
                loss_org = organizer_regularization(output, loss_cfg)
                loss = (
                    float(loss_cfg.get("field_mse_weight", 1.0)) * loss_field
                    + float(effective_internal_temperature_weight) * loss_internal
                    + float(effective_interface_weight) * loss_interface
                    + port_supervised_weight * loss_port
                    + port_smoothness_weight * loss_port_smoothness
                    + float(port_global_weight) * loss_port_global
                    + float(predicted_consistency_weight) * loss_predicted_consistency
                    + loss_org
                )
        if training:
            optimizer.zero_grad(set_to_none=True)
            clip_norm = float(gradient_clip_norm or 0.0)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if clip_norm > 0.0:
                    # AMP gradients must be unscaled before clipping; otherwise
                    # the threshold applies to scaled values and is meaningless.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
        with torch.no_grad():
            pred = output["pred_field"].detach()
            mse = torch.mean((pred - target) ** 2)
            temp_mse = torch.mean((pred[..., 4] - target[..., 4]) ** 2) if pred.shape[-1] >= 5 else mse
            reg_cfg = loss_cfg.get("organizer_regularization", {}) if isinstance(loss_cfg.get("organizer_regularization"), dict) else {}
            honf_diag = compute_honf_diagnostics(
                output,
                edge_strength_threshold=float(reg_cfg.get("edge_strength_threshold", 0.05)),
                edge_strength_temperature=float(reg_cfg.get("edge_strength_temperature", 0.05)),
            )
            metrics = {
                "loss_total": float(loss.detach().cpu()),
                "loss_field": float(loss_field.detach().cpu()),
                "loss_internal_temperature": float(loss_internal.detach().cpu()),
                "loss_interface": float(loss_interface.detach().cpu()),
                "loss_port_condition": float(loss_port.detach().cpu()),
                "loss_port_smoothness": float(loss_port_smoothness.detach().cpu()),
                "loss_port_global_consistency": float(loss_port_global.detach().cpu()),
                "loss_predicted_consistency": float(loss_predicted_consistency.detach().cpu()),
                "loss_predicted_consistency_internal": float(pred_cons_internal.detach().cpu()),
                "loss_predicted_consistency_interface": float(pred_cons_interface.detach().cpu()),
                "loss_organizer": float(loss_org.detach().cpu()),
                "effective_port_global_consistency_weight": float(port_global_weight),
                "effective_predicted_consistency_weight": float(predicted_consistency_weight),
                "effective_internal_temperature_weight": float(effective_internal_temperature_weight),
                "effective_interface_weight": float(effective_interface_weight),
                "field_mse": float(mse.detach().cpu()),
                "temperature_mse": float(temp_mse.detach().cpu()),
            }
            metrics.update(honf_diag)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        iterator.set_postfix(loss=f"{metrics['loss_total']:.3e}", field=f"{metrics['field_mse']:.3e}")
    if count == 0:
        return {
            key: math.nan
            for key in (
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
                *HONF_DIAGNOSTIC_KEYS,
            )
        }
    return {key: value / count for key, value in sums.items()}


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


def write_metrics_row(path: Path, fieldnames: Iterable[str], row: Dict[str, Any]) -> None:
    """Write metrics row."""

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def repair_metrics_csv_for_append(path: Path) -> None:
    """Repair metrics csv for append."""

    if not path.exists():
        return
    raw = path.read_bytes()
    repaired = raw.replace(b"\x00", b"")
    if repaired and not repaired.endswith(b"\n"):
        repaired += b"\n"
    if repaired != raw:
        path.write_bytes(repaired)


def best_metrics_payload(row: Dict[str, Any], best_total: float, best_field: float, best_temperature: float, best_predicted: float) -> Dict[str, float]:
    """Perform the best metrics payload operation used by this module."""

    payload = {
        "best_val_loss_total": float(best_total),
        "best_val_field_mse": float(best_field),
        "best_val_temperature_mse": float(best_temperature),
        "best_val_predicted_loss_total": float(best_predicted),
    }
    for key in HONF_DIAGNOSTIC_KEYS:
        val_key = f"val_{key}"
        if val_key in row:
            payload[val_key] = float(row[val_key])
    return payload


def _read_metric_history(metrics_path: Path) -> Dict[str, list[float]]:
    """Read numeric metric columns from `metrics.csv` for compact plotting."""

    if not metrics_path.exists():
        return {}
    columns: Dict[str, list[float]] = {}
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(parsed):
                    columns.setdefault(key, []).append(parsed)
    # Read-only aliases keep plots usable for historical metrics.csv files.
    aliases = {
        "selected_edge_count": "active_edge_count",
        "functional_edge_count": "active_edge_count",
        "soft_functional_edge_count": "soft_active_edge_count",
    }
    for new_name, old_name in aliases.items():
        for prefix in ("", "val_"):
            new_key = prefix + new_name
            old_key = prefix + old_name
            if new_key not in columns and old_key in columns:
                columns[new_key] = list(columns[old_key])
    return columns


def _plot_metric_group(
    ax: Any,
    history: Dict[str, list[float]],
    keys: tuple[str, ...],
    *,
    title: str,
    ylabel: str = "value",
    log_scale: bool = True,
    y_min_zero: bool = False,
    reference_lines: tuple[tuple[float, str], ...] = (),
) -> None:
    """Perform the plot metric group operation used by this module."""

    epochs = history.get("epoch", [])
    for key in keys:
        values = history.get(key)
        if not values:
            continue
        label = key
        for prefix in ("val_", "loss_"):
            label = label.removeprefix(prefix)
        if key.startswith("val_"):
            label = f"val {label}"
        ax.plot(epochs[: len(values)], values, label=label)
    if epochs:
        for index, (reference_y, reference_label) in enumerate(reference_lines):
            ax.axhline(
                float(reference_y),
                color="black" if index == 0 else "#666666",
                linestyle="--" if index == 0 else ":",
                linewidth=1.0,
                alpha=0.60,
                label=reference_label,
            )
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    if y_min_zero:
        _, top = ax.get_ylim()
        reference_max = max((value for value, _ in reference_lines), default=0.0)
        ax.set_ylim(bottom=0.0, top=max(float(top), float(reference_max) * 1.10, 1.0))
    ax.grid(True, alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=8)


def _resolved_active_edge_references(run_dir: Path) -> tuple[tuple[float, str], ...]:
    """Return configured candidate/selection references for activity plots."""

    config_path = run_dir / "config_resolved.json"
    if not config_path.exists():
        return ()
    try:
        payload = read_json(config_path)
        core = payload.get("model", {}).get("core_honf", {})
        if core.get("organizer_mode", "fixed_projection") == "exchangeable_slots":
            references = []
            for key in ("edge_capacity", "initial_active_edges"):
                value = core.get(key)
                if value is not None:
                    references.append((float(value), f"{key}={int(value)}"))
            return tuple(references)
        value = core.get("num_hyperedges")
        return () if value is None else ((float(value), f"num_hyperedges={int(value)}"),)
    except (OSError, TypeError, ValueError):
        return ()


def save_global_loss_plots(metrics_path: Path, run_dir: Path) -> None:
    """Save readable grouped plots instead of one overcrowded metric figure."""

    history = _read_metric_history(metrics_path)
    if not history or not history.get("epoch"):
        return
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-honf_cl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Keep the main PNG concise: it is for human scanning during training.
    # Detailed scalar diagnostics remain in metrics.csv and focused plots below.
    diagnostics_dir = run_dir / "diagnostic_plots"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    active_edge_references = _resolved_active_edge_references(run_dir)
    stale_root_plots = [
        "loss_curves.png",
        "loss_total_curve.png",
        "loss_field_curve.png",
        "loss_local_coupling_curve.png",
        "loss_port_condition_curve.png",
        "honf_entropy_activity_curve.png",
        "honf_context_curve.png",
    ]
    for filename in stale_root_plots:
        stale_path = run_dir / filename
        if stale_path.exists():
            stale_path.unlink()

    panels = [
        ("Total", ("loss_total", "val_loss_total", "val_predicted_loss_total"), "loss", True, False),
        ("Global Field", ("loss_field", "val_loss_field", "field_mse", "val_field_mse"), "loss / mse", True, False),
        (
            "Local/Internal Coupling",
            ("loss_internal_temperature", "val_loss_internal_temperature", "loss_interface", "val_loss_interface"),
            "loss",
            True,
            False,
        ),
        (
            "Port and Consistency",
            ("loss_port_condition", "val_loss_port_condition", "loss_port_global_consistency", "val_loss_port_global_consistency"),
            "loss",
            True,
            False,
        ),
        ("Temperature", ("temperature_mse", "val_temperature_mse"), "mse", True, False),
        ("Selected and Functional H-Edge Activity", ("selected_edge_count", "val_selected_edge_count", "functional_edge_count", "val_functional_edge_count", "soft_functional_edge_count", "val_soft_functional_edge_count"), "edges", False, True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.4), constrained_layout=True)
    for ax, (title, keys, ylabel, log_scale, y_min_zero) in zip(axes.reshape(-1), panels):
        _plot_metric_group(
            ax,
            history,
            keys,
            title=title,
            ylabel=ylabel,
            log_scale=log_scale,
            y_min_zero=y_min_zero,
            reference_lines=active_edge_references if y_min_zero else (),
        )
    fig.suptitle("HONF-CL Training Overview", fontsize=13)
    fig.savefig(str(run_dir / "loss_curve.png"), dpi=160)
    plt.close(fig)

    focused = {
        "loss_total_curve.png": ("Total Loss", ("loss_total", "val_loss_total", "val_predicted_loss_total"), "loss", True, False),
        "loss_field_curve.png": ("Field Loss and MSE", ("loss_field", "val_loss_field", "field_mse", "val_field_mse"), "loss / mse", True, False),
        "loss_local_coupling_curve.png": (
            "Internal Temperature and Interface Loss",
            ("loss_internal_temperature", "val_loss_internal_temperature", "loss_interface", "val_loss_interface"),
            "loss",
            True,
            False,
        ),
        "loss_port_condition_curve.png": (
            "Port and Consistency Loss",
            (
                "loss_port_condition",
                "val_loss_port_condition",
                "loss_port_global_consistency",
                "val_loss_port_global_consistency",
                "loss_predicted_consistency",
                "val_loss_predicted_consistency",
            ),
            "loss",
            True,
            False,
        ),
        "honf_entropy_activity_curve.png": (
            "HONF Entropy and Selected H-Edge Count",
            (
                "A_mh_entropy",
                "val_A_mh_entropy",
                "A_eh_entropy",
                "val_A_eh_entropy",
                "selected_edge_count",
                "val_selected_edge_count",
                "functional_edge_count",
                "val_functional_edge_count",
            ),
            "value",
            False,
            True,
        ),
        "honf_context_curve.png": (
            "HONF Context Norms and Pairwise Gate",
            (
                "pairwise_kernel_gate",
                "val_pairwise_kernel_gate",
                "pairwise_context_norm",
                "val_pairwise_context_norm",
                "total_hyper_context_norm",
                "val_total_hyper_context_norm",
                "nonhyper_context_norm",
                "val_nonhyper_context_norm",
            ),
            "value",
            True,
            False,
        ),
    }
    for filename, (title, keys, ylabel, log_scale, y_min_zero) in focused.items():
        fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
        _plot_metric_group(
            ax,
            history,
            keys,
            title=title,
            ylabel=ylabel,
            log_scale=log_scale,
            y_min_zero=y_min_zero,
            reference_lines=active_edge_references if y_min_zero else (),
        )
        fig.savefig(str(diagnostics_dir / filename), dpi=160)
        plt.close(fig)


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

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(training_cfg.get("learning_rate", 2.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-5)),
    )
    scaler = make_grad_scaler(device, bool(training_cfg.get("amp", False)))
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 200))
    max_train_batches = args.max_train_batches if args.max_train_batches is not None else training_cfg.get("max_train_batches_per_epoch")
    max_val_batches = args.max_val_batches if args.max_val_batches is not None else training_cfg.get("max_val_batches")
    resume_checkpoint = resolve_demo_path(args.resume_checkpoint) if args.resume_checkpoint else None

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
        *HONF_DIAGNOSTIC_KEYS,
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
        model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
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

    for epoch in range(start_epoch, epochs + 1):
        model.set_training_progress(epoch=epoch, total_epochs=epochs)
        train_dataset.set_epoch(epoch)
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        effective_mode, effective_ratio = effective_port_condition_settings(epoch, training_cfg)
        eff_internal, eff_interface = effective_local_loss_weights(loss_cfg, effective_mode, effective_ratio)
        pred_consistency_weight = predicted_consistency_weight_for_epoch(epoch, loss_cfg)
        gradient_clip_norm = float(training_cfg.get("gradient_clip_norm", 0.0) or 0.0)
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
        )
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
        if effective_mode == "predicted" and float(effective_ratio) == 0.0:
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
        row = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "val_predicted_loss_total": predicted_val_metrics.get("loss_total", math.nan),
            "val_predicted_field_mse": predicted_val_metrics.get("field_mse", math.nan),
            "val_predicted_temperature_mse": predicted_val_metrics.get("temperature_mse", math.nan),
        }
        write_metrics_row(metrics_path, fieldnames, row)
        total_metric = float(row["val_loss_total"])
        field_metric = float(row["val_field_mse"])
        temp_metric = float(row["val_temperature_mse"])
        if math.isfinite(total_metric) and total_metric < best_total:
            best_total = total_metric
            if bool(checkpoint_cfg.get("save_best", True)):
                save_checkpoint(run_dir / "best_model.pt", model=model, model_config=model_config, train_config=cfg, dataset=train_dataset, epoch=epoch, best_metric=best_total, optimizer=optimizer, scaler=scaler, best_metrics=best_metrics_payload(row, best_total, best_field, best_temperature, best_predicted))
        if math.isfinite(field_metric) and field_metric < best_field:
            best_field = field_metric
            if bool(checkpoint_cfg.get("save_best_field_mse", True)):
                save_checkpoint(run_dir / "best_by_field_mse_model.pt", model=model, model_config=model_config, train_config=cfg, dataset=train_dataset, epoch=epoch, best_metric=best_field, optimizer=optimizer, scaler=scaler, best_metrics=best_metrics_payload(row, best_total, best_field, best_temperature, best_predicted))
        if math.isfinite(temp_metric) and temp_metric < best_temperature:
            best_temperature = temp_metric
            if bool(checkpoint_cfg.get("save_best_temperature_mse", True)):
                save_checkpoint(run_dir / "best_by_temperature_mse_model.pt", model=model, model_config=model_config, train_config=cfg, dataset=train_dataset, epoch=epoch, best_metric=best_temperature, optimizer=optimizer, scaler=scaler, best_metrics=best_metrics_payload(row, best_total, best_field, best_temperature, best_predicted))
        predicted_metric = float(row["val_predicted_loss_total"])
        if math.isfinite(predicted_metric) and predicted_metric < best_predicted:
            best_predicted = predicted_metric
            if bool(checkpoint_cfg.get("save_best_predicted", True)):
                save_checkpoint(run_dir / "best_predicted_model.pt", model=model, model_config=model_config, train_config=cfg, dataset=train_dataset, epoch=epoch, best_metric=best_predicted, optimizer=optimizer, scaler=scaler, best_metrics=best_metrics_payload(row, best_total, best_field, best_temperature, best_predicted))
        if bool(checkpoint_cfg.get("save_latest", True)):
            save_checkpoint(run_dir / "latest_model.pt", model=model, model_config=model_config, train_config=cfg, dataset=train_dataset, epoch=epoch, best_metric=best_total, optimizer=optimizer, scaler=scaler, best_metrics=best_metrics_payload(row, best_total, best_field, best_temperature, best_predicted))
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
