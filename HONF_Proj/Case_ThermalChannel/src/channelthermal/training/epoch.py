"""Forward epoch model invocation, physical losses, and metric aggregation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from channelthermal.model import ChannelThermalHONFModel
from channelthermal.training_tools.losses import channelthermal_field_mse
from honf_forward_core.training.diagnostics import (
    HONF_DIAGNOSTIC_KEYS,
    compute_honf_diagnostics,
    organizer_regularization_loss,
)
from honf_runtime.compat import autocast_context, recursive_to_device


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
            metrics = pack_scalar_metrics(
                {
                    "loss_total": loss,
                    "loss_field": loss_field,
                    "loss_internal_temperature": loss_internal,
                    "loss_interface": loss_interface,
                    "loss_port_condition": loss_port,
                    "loss_port_smoothness": loss_port_smoothness,
                    "loss_port_global_consistency": loss_port_global,
                    "loss_predicted_consistency": loss_predicted_consistency,
                    "loss_predicted_consistency_internal": pred_cons_internal,
                    "loss_predicted_consistency_interface": pred_cons_interface,
                    "loss_organizer": loss_org,
                    "field_mse": mse,
                    "temperature_mse": temp_mse,
                }
            )
            metrics.update(
                {
                    "effective_port_global_consistency_weight": float(port_global_weight),
                    "effective_predicted_consistency_weight": float(predicted_consistency_weight),
                    "effective_internal_temperature_weight": float(effective_internal_temperature_weight),
                    "effective_interface_weight": float(effective_interface_weight),
                }
            )
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

