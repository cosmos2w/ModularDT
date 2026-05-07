from __future__ import annotations

"""Train the Stage B global Channel Thermal neural-field model."""

import argparse
import csv
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from channelthermal_datasets import GlobalChannelThermalDataset
from channelthermal_model_utils import (
    autocast_context,
    count_parameters,
    current_timestamp,
    ensure_dir,
    make_grad_scaler,
    masked_mse,
    read_json,
    recursive_to_device,
    resolve_demo_path,
    select_device,
    set_seed,
    write_json,
)
from model import (
    GlobalChannelThermalModel,
    GlobalChannelThermalModelConfig,
    build_thermal_module_env_prior,
    load_local_surrogate_from_checkpoint,
)


DEFAULT_CONFIG_PATH = "./Configs/train_global_config_template.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Demo 1 global Channel Thermal model.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="JSON config file path.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override, like cpu or cuda:0")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Stop each epoch after this many train batches.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Stop validation after this many batches.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional descriptive suffix after the numeric Run_ID.")
    parser.add_argument("--Run_ID", dest="run_id", type=str, default=None, help="Numeric run serial, e.g. 0001.")
    return parser.parse_args()


def resolve_config_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return resolve_demo_path(path)


def _auto_int(value: Any, fallback: int) -> int:
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return int(fallback)
    return int(value)


def _auto_float(value: Any, fallback: float) -> float:
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return float(fallback)
    return float(value)


def _first_domain_and_radius(dataset: GlobalChannelThermalDataset) -> tuple[float, float, float]:
    if len(dataset) == 0:
        return 12.0, 4.0, 0.45
    sample = dataset[0]
    structure = sample["structure"]
    lx = float(structure["domain_length_x"][0])
    ly = float(structure["domain_length_y"][0])
    material = structure["material_params"]
    radius = float(material[5]) if material.shape[0] > 5 and float(material[5]) > 0.0 else 0.45
    return lx, ly, radius


def build_model_config(payload: Dict[str, Any], dataset: GlobalChannelThermalDataset) -> GlobalChannelThermalModelConfig:
    model_cfg = dict(payload.get("model", {}))
    lx, ly, radius = _first_domain_and_radius(dataset)
    model_cfg["field_dim"] = _auto_int(model_cfg.get("field_dim"), dataset.field_dim)
    model_cfg["max_num_modules"] = _auto_int(model_cfg.get("max_num_modules"), dataset.max_num_modules)
    model_cfg["domain_length_x"] = _auto_float(model_cfg.get("domain_length_x"), lx)
    model_cfg["domain_length_y"] = _auto_float(model_cfg.get("domain_length_y"), ly)
    model_cfg["module_radius"] = _auto_float(model_cfg.get("module_radius"), radius)
    model_cfg["default_num_interface_points"] = _auto_int(
        model_cfg.get("default_num_interface_points"), dataset.n_interface_points or 64
    )
    model_cfg["material_param_dim"] = _auto_int(model_cfg.get("material_param_dim"), dataset.material_param_dim)
    return GlobalChannelThermalModelConfig.from_dict(model_cfg)


def normalize_run_id(value: Any, fallback: str = "0001") -> str:
    raw = str(value or fallback).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be a numeric serial such as '0001'; got {raw!r}.")
    return f"{int(raw):04d}"


def sanitize_run_suffix(value: Any) -> str:
    raw = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw).strip("_")


def resolve_run_id(args: argparse.Namespace, cfg: Dict[str, Any], fallback: str) -> str:
    training_cfg = cfg.get("training", {})
    return normalize_run_id(
        args.run_id
        or cfg.get("Run_ID")
        or cfg.get("run_id")
        or training_cfg.get("Run_ID")
        or training_cfg.get("run_id")
        or fallback,
        fallback,
    )


def field_loss(pred: torch.Tensor, target: torch.Tensor, loss_cfg: Dict[str, Any]) -> torch.Tensor:
    weights = torch.ones(pred.shape[-1], device=pred.device, dtype=pred.dtype)
    temperature_weight = float(loss_cfg.get("temperature_weight", 1.0))
    if pred.shape[-1] >= 5:
        weights[4] = temperature_weight
    channel_weights = loss_cfg.get("field_channel_weights")
    if channel_weights is not None:
        custom = torch.as_tensor(channel_weights, device=pred.device, dtype=pred.dtype)
        weights[: custom.numel()] = custom[: pred.shape[-1]]
    return ((pred - target).square() * weights).mean()


def port_condition_loss(
    pred_port: torch.Tensor,
    target_port: torch.Tensor,
    module_present: torch.Tensor,
    loss_cfg: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scaled optimization loss and raw physical MSE for port tokens.

    The port head predicts physical ``T_env`` and ``h`` values. Their raw MSE can
    be hundreds at initialization, so the optimization loss uses stable physical
    scales while the raw MSE is kept as a diagnostic.
    """
    pred_values = pred_port[..., 3:5]
    target_values = target_port[..., 3:5]
    raw_mse = masked_mse(pred_values, target_values, module_present[:, :, None])
    scales = pred_values.new_tensor(
        [
            max(float(loss_cfg.get("port_temperature_scale", 10.0)), 1.0e-6),
            max(float(loss_cfg.get("port_h_scale", 10.0)), 1.0e-6),
        ]
    )
    scaled_mse = masked_mse(pred_values / scales, target_values / scales, module_present[:, :, None])
    return scaled_mse, raw_mse


def interface_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    module_present: torch.Tensor,
    outputs: Dict[str, Any],
    loss_cfg: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return interface loss plus per-target diagnostics.

    A frozen Stage-A local surrogate may have been trained on a synthetic
    q_normal distribution that does not match the global dataset. In that case
    local_surrogate_interface_target_weights can disable q_normal supervision
    while keeping T_surface supervision.
    """
    source = str(outputs.get("interface_source", "global_head"))
    weights_cfg = loss_cfg.get("interface_target_weights", None)
    if source == "local_surrogate" and loss_cfg.get("local_surrogate_interface_target_weights", None) is not None:
        weights_cfg = loss_cfg.get("local_surrogate_interface_target_weights")

    mse_by_target = []
    for idx in range(pred.shape[-1]):
        mse_by_target.append(masked_mse(pred[..., idx : idx + 1], target[..., idx : idx + 1], module_present[:, :, None]))
    if not mse_by_target:
        zero = pred.new_tensor(0.0)
        return zero, zero, zero

    if weights_cfg is None:
        loss = masked_mse(pred, target, module_present[:, :, None])
    else:
        weights = torch.as_tensor(weights_cfg, device=pred.device, dtype=pred.dtype)
        if weights.numel() < pred.shape[-1]:
            weights = F.pad(weights, (0, pred.shape[-1] - weights.numel()), value=1.0)
        weights = weights[: pred.shape[-1]].clamp_min(0.0)
        weighted = torch.stack(mse_by_target) * weights
        loss = weighted.sum() / weights.sum().clamp_min(1.0e-6)

    t_surface_mse = mse_by_target[0]
    q_normal_mse = mse_by_target[1] if len(mse_by_target) > 1 else pred.new_tensor(0.0)
    return loss, t_surface_mse, q_normal_mse


def _normalize_rows(values: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return values / values.sum(dim=-1, keepdim=True).clamp_min(eps)


def _thermal_prior_params(loss_cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "sigma_distance": float(loss_cfg.get("organizer_me_sigma_distance", 1.50)),
        "sigma_downstream": float(loss_cfg.get("organizer_me_sigma_downstream", 2.00)),
        "sigma_lateral": float(loss_cfg.get("organizer_me_sigma_lateral", 0.90)),
        "wall_weight": float(loss_cfg.get("organizer_me_wall_weight", 0.15)),
        "heat_power_weight": float(loss_cfg.get("organizer_me_heat_power_weight", 0.15)),
    }


def build_thermal_prior_from_batch(outputs: Dict[str, Any], batch: Dict[str, Any], loss_cfg: Dict[str, Any]) -> torch.Tensor:
    """Build the weak channel module->environment prior used for organizer losses."""
    structure = batch["structure"]
    material = structure.get("material_params")
    module_radius = 0.45
    if torch.is_tensor(material) and material.numel() > 5:
        module_radius = float(material.reshape(material.shape[0], -1)[0, 5].detach().cpu())
        if module_radius <= 0.0:
            module_radius = 0.45
    domain_length_x = float(structure.get("domain_length_x", outputs["module_centers"].new_tensor([[12.0]])).reshape(-1)[0].detach().cpu())
    domain_length_y = float(structure.get("domain_length_y", outputs["module_centers"].new_tensor([[4.0]])).reshape(-1)[0].detach().cpu())
    return build_thermal_module_env_prior(
        outputs["module_centers"],
        outputs["env_coords"],
        outputs["module_present"],
        outputs["heat_powers"],
        domain_length_x,
        domain_length_y,
        module_radius,
        **_thermal_prior_params(loss_cfg),
        eps=float(loss_cfg.get("organizer_eps", 1.0e-6)),
    )


def compute_hyperedge_masses_channelthermal(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw and normalized module/environment mass per hyperedge."""
    A_mh = outputs["A_mh"]
    A_eh = outputs["A_eh"]
    module_present = batch["structure"]["module_present"].to(device=A_mh.device, dtype=A_mh.dtype)
    module_mass_raw = (A_mh * module_present[:, :, None]).sum(dim=1)
    module_mass_raw = module_mass_raw / module_present.sum(dim=1, keepdim=True).clamp_min(eps)
    env_mass_raw = A_eh.mean(dim=1)
    module_mass = module_mass_raw / module_mass_raw.sum(dim=-1, keepdim=True).clamp_min(eps)
    env_mass = env_mass_raw / env_mass_raw.sum(dim=-1, keepdim=True).clamp_min(eps)
    return module_mass_raw, env_mass_raw, module_mass, env_mass


def build_eh_factor_target_channelthermal(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    target_source: str = "prior_me",
    *,
    loss_cfg: Optional[Dict[str, Any]] = None,
    detach_mh: bool = True,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Target A_eh[e,k] from module->env influence factored through A_mh."""
    loss_cfg = loss_cfg or {}
    source = str(target_source).strip().lower()
    if source == "prior_me":
        A_me_for_target = build_thermal_prior_from_batch(outputs, batch, loss_cfg)
    elif source == "learned_me":
        A_me_for_target = outputs["A_me"].detach()
    else:
        raise ValueError("organizer_eh_factor_target_source must be 'prior_me' or 'learned_me'.")
    A_mh = outputs["A_mh"].detach() if detach_mh else outputs["A_mh"]
    raw = torch.einsum("bnm,bnk->bmk", A_me_for_target, A_mh)
    denom = raw.sum(dim=-1, keepdim=True)
    _, _, module_mass, _ = compute_hyperedge_masses_channelthermal(outputs, batch, eps=eps)
    fallback = module_mass[:, None, :].expand_as(raw)
    target = torch.where(denom > eps, raw / denom.clamp_min(eps), fallback)
    return _normalize_rows(target.clamp_min(eps), eps=eps)


def _module_module_affinity_prior_channelthermal(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], eps: float = 1.0e-6) -> torch.Tensor:
    centers = outputs["module_centers"]
    present = batch["structure"]["module_present"].to(device=centers.device, dtype=centers.dtype)
    dx = centers[:, None, :, 0] - centers[:, :, None, 0]
    dy = centers[:, None, :, 1] - centers[:, :, None, 1]
    dist = torch.sqrt(dx.square() + dy.square() + eps)
    same_lateral = torch.exp(-(dy.abs() / 0.9).square())
    close = torch.exp(-dist / 1.5)
    downstream_pair = torch.exp(-torch.relu(dx) / 2.0) * same_lateral
    affinity = 0.55 * close + 0.45 * downstream_pair
    n = centers.shape[1]
    eye = torch.eye(n, device=centers.device, dtype=centers.dtype)[None, :, :]
    return affinity * present[:, :, None] * present[:, None, :] * (1.0 - eye)


def organizer_mass_diagnostics_channelthermal(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    eps: float = 1.0e-6,
) -> Dict[str, torch.Tensor]:
    """Collapse-oriented diagnostics for the channel organizer."""
    _, _, module_mass, env_mass = compute_hyperedge_masses_channelthermal(outputs, batch, eps=eps)
    env_entropy = -(env_mass.clamp_min(eps) * env_mass.clamp_min(eps).log()).sum(dim=-1)
    module_entropy = -(module_mass.clamp_min(eps) * module_mass.clamp_min(eps).log()).sum(dim=-1)
    A_mh = outputs["A_mh"]
    A_eh = outputs["A_eh"]
    present = batch["structure"]["module_present"].to(device=A_mh.device, dtype=A_mh.dtype)
    valid_module_max = A_mh.max(dim=-1).values * present
    return {
        "org_env_mass_max": env_mass.max(dim=-1).values.mean(),
        "org_module_mass_max": module_mass.max(dim=-1).values.mean(),
        "org_mass_l1": (env_mass - module_mass).abs().mean(),
        "org_env_effective_hyperedges": torch.exp(env_entropy).mean(),
        "org_module_effective_hyperedges": torch.exp(module_entropy).mean(),
        "org_max_A_eh_mass": A_eh.max(dim=-1).values.mean(),
        "org_max_A_mh_mass": valid_module_max.sum() / present.sum().clamp_min(1.0),
    }


def compute_organizer_losses(outputs: Dict[str, Any], batch: Dict[str, Any], loss_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Weak direct supervision for physically meaningful channel hyperedges."""
    aux = outputs.get("organizer_aux", {})
    if not isinstance(aux, dict) or not {"A_me", "A_mh", "A_eh"}.issubset(aux):
        zero = outputs["pred_field"].new_tensor(0.0)
        return {
            "loss_org_me": zero,
            "loss_org_eh_factor": zero,
            "loss_org_mass_align": zero,
            "loss_org_mm": zero,
            "loss_org_direct": zero,
            "org_env_mass_max": zero,
            "org_module_mass_max": zero,
            "org_mass_l1": zero,
            "org_env_effective_hyperedges": zero,
            "org_module_effective_hyperedges": zero,
            "org_max_A_eh_mass": zero,
            "org_max_A_mh_mass": zero,
        }

    eps = float(loss_cfg.get("organizer_eps", 1.0e-6))
    present = batch["structure"]["module_present"].to(device=aux["A_me"].device, dtype=aux["A_me"].dtype)

    # A_me remains learned, but it is weakly guided toward a nonperiodic
    # channel-thermal prior: near/downstream/wall/heat-power environment mass.
    prior_me = build_thermal_prior_from_batch(aux, batch, loss_cfg).detach()
    kl_me = F.kl_div(aux["A_me"].clamp_min(eps).log(), prior_me, reduction="none").sum(dim=-1)
    loss_org_me = (kl_me * present).sum() / present.sum().clamp_min(1.0)

    # Direct A_eh factor target: env token e should prefer hyperedge k when
    # modules influencing e are assigned to k by A_mh.
    target_eh = build_eh_factor_target_channelthermal(
        aux,
        batch,
        target_source=str(loss_cfg.get("organizer_eh_factor_target_source", "prior_me")),
        loss_cfg=loss_cfg,
        detach_mh=bool(loss_cfg.get("organizer_eh_factor_detach_mh", True)),
        eps=eps,
    ).detach()
    loss_org_eh_factor = F.kl_div(aux["A_eh"].clamp_min(eps).log(), target_eh, reduction="none").sum(dim=-1).mean()

    _, _, module_mass, env_mass = compute_hyperedge_masses_channelthermal(aux, batch, eps=eps)
    module_target = module_mass.detach() if bool(loss_cfg.get("organizer_mass_align_detach_module", True)) else module_mass
    if bool(loss_cfg.get("organizer_mass_align_log_space", True)):
        loss_org_mass_align = F.smooth_l1_loss(torch.log(env_mass + eps), torch.log(module_target + eps))
    else:
        loss_org_mass_align = F.smooth_l1_loss(env_mass, module_target)

    if float(loss_cfg.get("organizer_mm_weight", 0.0)) > 0.0:
        pred_mm = torch.matmul(aux["A_mh"], aux["A_mh"].transpose(1, 2))
        prior_mm = _module_module_affinity_prior_channelthermal(aux, batch, eps=eps)
        n = pred_mm.shape[1]
        eye = torch.eye(n, device=pred_mm.device, dtype=pred_mm.dtype)[None, :, :]
        valid = present[:, :, None] * present[:, None, :] * (1.0 - eye)
        loss_org_mm = (((pred_mm - prior_mm) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)
    else:
        loss_org_mm = aux["A_mh"].new_tensor(0.0)

    loss_org_direct = (
        float(loss_cfg.get("organizer_me_weight", 0.01)) * loss_org_me
        + float(loss_cfg.get("organizer_eh_factor_weight", 0.05)) * loss_org_eh_factor
        + float(loss_cfg.get("organizer_mass_align_weight", 0.02)) * loss_org_mass_align
        + float(loss_cfg.get("organizer_mm_weight", 0.0)) * loss_org_mm
    )
    out = {
        "loss_org_me": loss_org_me,
        "loss_org_eh_factor": loss_org_eh_factor,
        "loss_org_mass_align": loss_org_mass_align,
        "loss_org_mm": loss_org_mm,
        "loss_org_direct": loss_org_direct,
    }
    out.update(organizer_mass_diagnostics_channelthermal(aux, batch, eps=eps))
    return out


def effective_port_condition_settings(epoch: int, training_cfg: Dict[str, Any]) -> tuple[str, float]:
    """Resolve the Stage-B local-port curriculum for this epoch.

    Stage B can first learn with teacher boundary tokens, then gradually move
    the local surrogate toward the model-predicted port conditions that are
    available during autonomous design inference.
    """
    schedule = str(training_cfg.get("port_condition_schedule", "none")).lower()
    base_mode = str(training_cfg.get("local_port_condition_mode", "teacher")).lower()
    base_ratio = float(training_cfg.get("mixed_teacher_ratio", 0.5))
    if schedule == "none":
        return base_mode, base_ratio
    if schedule != "teacher_to_predicted":
        raise ValueError(f"Unsupported port_condition_schedule={schedule!r}.")

    teacher_epochs = int(training_cfg.get("teacher_epochs", 100))
    mixed_epochs = int(training_cfg.get("mixed_epochs", 200))
    predicted_after_epoch_cfg = training_cfg.get("predicted_after_epoch", None)
    predicted_after_epoch = int(predicted_after_epoch_cfg) if predicted_after_epoch_cfg is not None else teacher_epochs + mixed_epochs
    ratio_start = float(training_cfg.get("mixed_teacher_ratio_start", 1.0))
    ratio_end = float(training_cfg.get("mixed_teacher_ratio_end", 0.0))
    if int(epoch) <= teacher_epochs:
        return "teacher", 1.0
    if int(epoch) <= predicted_after_epoch:
        span = max(predicted_after_epoch - teacher_epochs, 1)
        progress = (int(epoch) - teacher_epochs) / span
        ratio = ratio_start + (ratio_end - ratio_start) * progress
        return "mixed", float(min(max(ratio, 0.0), 1.0))
    return "predicted", 0.0


def validate_local_surrogate_training_path(
    model_config: GlobalChannelThermalModelConfig,
    training_cfg: Dict[str, Any],
    loss_cfg: Dict[str, Any],
) -> None:
    """Catch frozen teacher-forced local losses that cannot improve."""
    if not model_config.use_local_surrogate or not model_config.freeze_local_surrogate:
        return
    schedule = str(training_cfg.get("port_condition_schedule", "none")).lower()
    mode = str(training_cfg.get("local_port_condition_mode", "teacher")).lower()
    local_loss_weight = float(loss_cfg.get("internal_temperature_weight", 1.0)) + float(loss_cfg.get("interface_weight", 0.2))
    if schedule == "none" and mode == "teacher" and local_loss_weight > 0.0:
        raise ValueError(
            "The local surrogate is frozen and local_port_condition_mode='teacher' with "
            "port_condition_schedule='none'. In this configuration pred_internal_temperature "
            "and pred_interface are fixed outputs of the local checkpoint, so the internal "
            "temperature and interface losses cannot decrease. Set "
            "training.port_condition_schedule='teacher_to_predicted' so those losses can train "
            "the global port head after the teacher phase, or set local_port_condition_mode='predicted', "
            "or set internal_temperature_weight/interface_weight to 0 if you only want diagnostic local losses."
        )


def compute_losses(outputs: Dict[str, Any], batch: Dict[str, Any], loss_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    structure = batch["structure"]
    module_present = structure["module_present"]
    losses: Dict[str, torch.Tensor] = {}
    losses["loss_field"] = field_loss(outputs["pred_field"], batch["field_targets"], loss_cfg)

    if outputs.get("pred_internal_temperature") is not None and outputs["pred_internal_temperature"].numel() > 0:
        target_internal = batch["module_internal_temperature_points"].unsqueeze(-1)
        losses["loss_internal_temperature"] = masked_mse(
            outputs["pred_internal_temperature"],
            target_internal,
            module_present[:, :, None],
        )
    else:
        losses["loss_internal_temperature"] = outputs["pred_field"].new_tensor(0.0)

    if outputs.get("pred_interface") is not None and outputs["pred_interface"].numel() > 0:
        (
            losses["loss_interface"],
            losses["diag_interface_T_surface_mse"],
            losses["diag_interface_q_normal_mse"],
        ) = interface_loss(outputs["pred_interface"], batch["interface_target"], module_present, outputs, loss_cfg)
    else:
        losses["loss_interface"] = outputs["pred_field"].new_tensor(0.0)
        losses["diag_interface_T_surface_mse"] = outputs["pred_field"].new_tensor(0.0)
        losses["diag_interface_q_normal_mse"] = outputs["pred_field"].new_tensor(0.0)

    pred_port = outputs["pred_port_condition"]
    target_port = batch["teacher_port_tokens"]
    losses["loss_port_condition"], losses["diag_port_condition_physical_mse"] = port_condition_loss(
        pred_port,
        target_port,
        module_present,
        loss_cfg,
    )

    aux = outputs.get("organizer_aux", {})
    if isinstance(aux, dict) and "hyper_strength" in aux:
        losses["diag_organizer_strength_mean"] = aux["hyper_strength"].mean()
    else:
        losses["diag_organizer_strength_mean"] = pred_port.new_tensor(0.0)
    losses.update(compute_organizer_losses(outputs, batch, loss_cfg))

    total = (
        float(loss_cfg.get("field_mse_weight", 1.0)) * losses["loss_field"]
        + float(loss_cfg.get("internal_temperature_weight", 1.0)) * losses["loss_internal_temperature"]
        + float(loss_cfg.get("interface_weight", 0.2)) * losses["loss_interface"]
        + float(loss_cfg.get("port_condition_weight", 0.1)) * losses["loss_port_condition"]
        + losses["loss_org_direct"]
    )
    losses["loss_total"] = total
    return losses


def _read_loss_history(history_path: Path) -> Dict[str, list[float]]:
    """Read numeric columns from the global loss CSV."""
    if not history_path.exists():
        return {}
    columns: Dict[str, list[float]] = {}
    with history_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                columns.setdefault(key, []).append(parsed)
    return columns


def _plot_loss_group(
    ax: Any,
    history: Dict[str, list[float]],
    keys: tuple[str, ...],
    *,
    title: str,
    ylabel: str = "loss",
) -> None:
    epochs = history.get("epoch", [])
    for key in keys:
        values = history.get(key)
        if not values:
            continue
        label = key.removeprefix("val_").removeprefix("loss_")
        if key.startswith("val_"):
            label = f"val {label}"
        ax.plot(epochs[: len(values)], values, label=label)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def save_global_loss_plots(history_path: Path, run_dir: Path) -> None:
    """Save readable global loss plots instead of one overcrowded figure."""
    history = _read_loss_history(history_path)
    if not history or not history.get("epoch"):
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Total", ("loss_total", "val_loss_total")),
        ("Global Field", ("loss_field", "val_loss_field")),
        ("Module/Internal Coupling", ("loss_internal_temperature", "val_loss_internal_temperature", "loss_interface", "val_loss_interface")),
        ("Port Condition", ("loss_port_condition", "val_loss_port_condition")),
        ("Organizer", ("loss_org_me", "val_loss_org_me", "loss_org_eh_factor", "val_loss_org_eh_factor", "loss_org_mass_align", "val_loss_org_mass_align")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.2), constrained_layout=True)
    for ax, (title, keys) in zip(axes.reshape(-1), panels):
        _plot_loss_group(ax, history, keys, title=title)
    for ax in axes.reshape(-1)[len(panels) :]:
        ax.axis("off")
    fig.suptitle("Global Channel Thermal Losses", fontsize=13)
    fig.savefig(run_dir / "loss_curve.png", dpi=160)
    plt.close(fig)

    focused = {
        "loss_total_curve.png": ("Total Loss", ("loss_total", "val_loss_total")),
        "loss_field_curve.png": ("Field Loss", ("loss_field", "val_loss_field")),
        "loss_internal_interface_curve.png": (
            "Internal Temperature and Interface Loss",
            ("loss_internal_temperature", "val_loss_internal_temperature", "loss_interface", "val_loss_interface"),
        ),
        "loss_port_condition_curve.png": ("Port Condition Loss", ("loss_port_condition", "val_loss_port_condition")),
    }
    for filename, (title, keys) in focused.items():
        fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        _plot_loss_group(ax, history, keys, title=title)
        fig.savefig(run_dir / filename, dpi=160)
        plt.close(fig)


def run_epoch(
    model: GlobalChannelThermalModel,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler=None,
    amp: bool = False,
    max_batches: Optional[int] = None,
    local_port_condition_mode: str = "teacher",
    mixed_teacher_ratio: float = 0.5,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if model.local_surrogate is not None and model.config.freeze_local_surrogate:
        model.local_surrogate.eval()
    sums: Dict[str, float] = {}
    count = 0
    iterator = tqdm(loader, desc="train" if training else "val", unit="batch", dynamic_ncols=True, leave=False)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = recursive_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context(device, amp):
            outputs = model(
                batch["structure"],
                batch["query_xy"],
                interface_condition=batch["interface_condition"],
                local_module_params=batch["local_module_params"],
                teacher_port_tokens=batch["teacher_port_tokens"],
                local_query_points=batch["module_internal_query_points"],
                local_port_condition_mode=local_port_condition_mode,
                mixed_teacher_ratio=float(mixed_teacher_ratio),
            )
            losses = compute_losses(outputs, batch, loss_cfg)
        if training:
            clip_norm = float(training_cfg.get("gradient_clip_norm", 1.0))
            if scaler is not None and scaler.is_enabled():
                scaler.scale(losses["loss_total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
        batch_size = int(batch["query_xy"].shape[0])
        count += batch_size
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        iterator.set_postfix(
            loss=f"{float(losses['loss_total'].detach().cpu()):.3e}",
            org_l1=f"{float(losses.get('org_mass_l1', losses['loss_total']).detach().cpu()):.2e}",
        )
    if count == 0:
        return {
            key: float("nan")
            for key in [
                "loss_total",
                "loss_field",
                "loss_internal_temperature",
                "loss_interface",
                "diag_interface_T_surface_mse",
                "diag_interface_q_normal_mse",
                "loss_port_condition",
                "diag_port_condition_physical_mse",
                "diag_organizer_strength_mean",
                "loss_org_me",
                "loss_org_eh_factor",
                "loss_org_mass_align",
                "loss_org_mm",
                "loss_org_direct",
                "org_env_mass_max",
                "org_module_mass_max",
                "org_mass_l1",
                "org_env_effective_hyperedges",
                "org_module_effective_hyperedges",
                "org_max_A_eh_mass",
                "org_max_A_mh_mass",
            ]
        }
    return {key: value / count for key, value in sums.items()}


def save_checkpoint(
    path: Path,
    *,
    model: GlobalChannelThermalModel,
    model_config: GlobalChannelThermalModelConfig,
    train_config: Dict[str, Any],
    dataset: GlobalChannelThermalDataset,
    epoch: int,
    best_metric: float,
) -> None:
    torch.save(
        {
            "stage": "global_channelthermal_stage_b",
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "model_config": model_config.to_dict(),
            "model_state_dict": model.state_dict(),
            "train_config": train_config,
            "channel_order": list(dataset.channel_order),
            "field_dim": int(dataset.field_dim),
            "interface_condition_feature_names": list(dataset.interface_condition_feature_names),
            "interface_target_names": list(dataset.interface_target_names),
            "global_normalization_config": {
                "normalize_inputs": bool(train_config.get("dataset", {}).get("normalize_inputs", False)),
                "normalize_targets": bool(train_config.get("dataset", {}).get("normalize_targets", False)),
            },
            "global_normalization_stats": {name: value.copy() for name, value in dataset.normalizer.stats.items()},
        },
        path,
    )


def local_surrogate_normalization_config(checkpoint: Dict[str, Any]) -> Dict[str, bool]:
    config = checkpoint.get("local_normalization_config")
    if isinstance(config, dict):
        return {
            "normalize_inputs": bool(config.get("normalize_inputs", False)),
            "normalize_targets": bool(config.get("normalize_targets", False)),
        }
    dataset_cfg = checkpoint.get("train_config", {}).get("dataset", {})
    return {
        "normalize_inputs": bool(dataset_cfg.get("normalize_inputs", False)),
        "normalize_targets": bool(dataset_cfg.get("normalize_targets", False)),
    }


def require_local_normalization_stats(checkpoint: Dict[str, Any], normalization_config: Dict[str, bool]) -> Dict[str, Any]:
    stats = checkpoint.get("local_normalization_stats", {})
    if not isinstance(stats, dict):
        stats = {}
    if bool(normalization_config.get("normalize_inputs", False)):
        missing = [key for key in ("module_params_mean", "module_params_std", "port_tokens_mean", "port_tokens_std") if key not in stats]
        if missing:
            raise ValueError(f"Local surrogate checkpoint was trained with normalized inputs but is missing stats: {missing}")
    if bool(normalization_config.get("normalize_targets", False)):
        missing = [
            key
            for key in (
                "internal_temperature_mean",
                "internal_temperature_std",
                "interface_targets_mean",
                "interface_targets_std",
            )
            if key not in stats
        ]
        if missing:
            raise ValueError(f"Local surrogate checkpoint was trained with normalized targets but is missing stats: {missing}")
    return stats


def attach_local_surrogate_if_needed(
    model: GlobalChannelThermalModel,
    model_config: GlobalChannelThermalModelConfig,
    cfg: Dict[str, Any],
    device: torch.device,
) -> None:
    if not model_config.use_local_surrogate:
        return
    checkpoint_path = cfg.get("model", {}).get("local_surrogate_checkpoint_path")
    if not checkpoint_path:
        raise ValueError(
            "model.use_local_surrogate=true requires model.local_surrogate_checkpoint_path. "
            "Set use_local_surrogate=false for the global-only baseline."
        )
    local_model, local_checkpoint = load_local_surrogate_from_checkpoint(resolve_demo_path(checkpoint_path), map_location=device)
    normalization_config = local_surrogate_normalization_config(local_checkpoint)
    normalization_stats = require_local_normalization_stats(local_checkpoint, normalization_config)
    local_model.to(device)
    model.set_local_surrogate(
        local_model,
        freeze=bool(model_config.freeze_local_surrogate),
        normalization_config=normalization_config,
        normalization_stats=normalization_stats,
    )


def print_dataset_quality(name: str, dataset: GlobalChannelThermalDataset) -> None:
    """Print convergence metadata for a loaded global dataset split."""
    print(
        f"[dataset:{name}] target_mode={getattr(dataset, 'target_mode', 'unknown')} "
        f"require_converged_attr={getattr(dataset, 'require_converged', False)} "
        f"converged_cases={getattr(dataset, 'num_selected_converged', 0)} "
        f"unconverged_cases={getattr(dataset, 'num_selected_unconverged', 0)}"
    )


def enforce_dataset_convergence(name: str, dataset: GlobalChannelThermalDataset, require_converged: bool) -> None:
    """Raise clearly if config asks for converged cases but the split contains unconverged cases."""
    if not require_converged:
        return
    unconverged = [
        case_id
        for case_id, flag in zip(getattr(dataset, "selected_case_ids", []), getattr(dataset, "selected_converged_flags", []))
        if not flag
    ]
    if unconverged:
        preview = ", ".join(str(case_id) for case_id in unconverged[:10])
        raise ValueError(
            f"dataset.require_converged=true but {name} split contains {len(unconverged)} unconverged cases. "
            f"Examples: {preview}"
        )


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    cfg = read_json(config_path)
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    if float(loss_cfg.get("organizer_strength_weight", 0.0)) != 0.0:
        warnings.warn(
            "organizer_strength_weight is ignored; hyperedge strength is logged as "
            "diag_organizer_strength_mean. Positive organizer_strength_weight used "
            "to penalize strong hyperedges and should usually stay 0.0.",
            stacklevel=2,
        )
    set_seed(int(training_cfg.get("seed", 42)))
    device = select_device(args.device or training_cfg.get("device"))

    train_dataset = GlobalChannelThermalDataset(
        dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"),
        split=dataset_cfg.get("train_split", "train"),
        points_per_case=dataset_cfg.get("points_per_case", 4096),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=bool(dataset_cfg.get("random_point_sampling", True)),
        seed=int(training_cfg.get("seed", 42)),
    )
    val_dataset = GlobalChannelThermalDataset(
        dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"),
        split=dataset_cfg.get("val_split", "test"),
        points_per_case=dataset_cfg.get("val_points_per_case", dataset_cfg.get("points_per_case", 4096)),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        seed=int(training_cfg.get("seed", 42)) + 1000,
    )
    if len(val_dataset) == 0:
        val_dataset = train_dataset
    require_converged = bool(dataset_cfg.get("require_converged", False))
    print_dataset_quality("train", train_dataset)
    print_dataset_quality("val", val_dataset)
    enforce_dataset_convergence("train", train_dataset, require_converged)

    model_config = build_model_config(cfg, train_dataset)
    validate_local_surrogate_training_path(model_config, training_cfg, loss_cfg)
    model = GlobalChannelThermalModel(model_config).to(device)
    model.set_global_target_normalization(
        train_dataset.normalizer.stats,
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
    )
    attach_local_surrogate_if_needed(model, model_config, cfg, device)
    print(f"[setup] device={device}, train_cases={len(train_dataset)}, val_cases={len(val_dataset)}")
    print(
        "[setup] port schedule: "
        f"schedule={training_cfg.get('port_condition_schedule', 'none')}, "
        f"base_mode={training_cfg.get('local_port_condition_mode', 'teacher')}, "
        f"teacher_epochs={int(training_cfg.get('teacher_epochs', 100))}, "
        f"mixed_epochs={int(training_cfg.get('mixed_epochs', 200))}, "
        f"predicted_after_epoch={training_cfg.get('predicted_after_epoch', None)}"
    )
    print(
        "[setup] normalization: "
        f"inputs={bool(dataset_cfg.get('normalize_inputs', False))}, "
        f"targets={bool(dataset_cfg.get('normalize_targets', False))}"
    )
    if train_dataset.normalizer.has("internal_temperature_mean", "internal_temperature_std"):
        internal_mean = train_dataset.normalizer.stats["internal_temperature_mean"].reshape(-1)[0]
        internal_std = train_dataset.normalizer.stats["internal_temperature_std"].reshape(-1)[0]
        interface_mean = train_dataset.normalizer.stats.get("interface_target_mean")
        interface_std = train_dataset.normalizer.stats.get("interface_target_std")
        if interface_mean is None:
            interface_mean = train_dataset.normalizer.stats.get("interface_targets_mean")
        if interface_std is None:
            interface_std = train_dataset.normalizer.stats.get("interface_targets_std")
        print(f"[setup] internal T mean/std={float(internal_mean):.6g}/{float(internal_std):.6g}")
        if interface_mean is not None and interface_std is not None:
            print(
                "[setup] interface target mean/std="
                f"{interface_mean.astype(float).tolist()}/{interface_std.astype(float).tolist()}"
            )
    print(f"[setup] model parameters={count_parameters(model):,}")

    batch_size = int(dataset_cfg.get("batch_size", training_cfg.get("batch_size", 4)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(dataset_cfg.get("val_batch_size", batch_size)),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(training_cfg.get("learning_rate", 2.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-5)),
    )
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 200))
    scaler = make_grad_scaler(device, bool(training_cfg.get("amp", False)))
    saved_root = ensure_dir(resolve_demo_path(cfg.get("paths", {}).get("saved_model_dir", "./Saved_Model")))
    run_id = resolve_run_id(args, cfg, "0001")
    cfg["Run_ID"] = run_id
    run_suffix = sanitize_run_suffix(args.run_name or training_cfg.get("run_name"))
    run_name = f"Run_{run_id}_{run_suffix}_{current_timestamp()}" if run_suffix else f"Run_{run_id}_{current_timestamp()}"
    run_dir = ensure_dir(saved_root / run_name)
    write_json(run_dir / "resolved_train_config.json", cfg)
    history_path = run_dir / "loss_history.csv"
    fieldnames = [
        "epoch",
        "loss_total",
        "loss_field",
        "loss_internal_temperature",
        "loss_interface",
        "diag_interface_T_surface_mse",
        "diag_interface_q_normal_mse",
        "loss_port_condition",
        "diag_port_condition_physical_mse",
        "diag_organizer_strength_mean",
        "loss_org_me",
        "loss_org_eh_factor",
        "loss_org_mass_align",
        "loss_org_mm",
        "loss_org_direct",
        "org_env_mass_max",
        "org_module_mass_max",
        "org_mass_l1",
        "org_env_effective_hyperedges",
        "org_module_effective_hyperedges",
        "org_max_A_eh_mass",
        "org_max_A_mh_mass",
        "effective_local_port_condition_mode",
        "effective_mixed_teacher_ratio",
        "val_loss_total",
        "val_loss_field",
        "val_loss_internal_temperature",
        "val_loss_interface",
        "val_diag_interface_T_surface_mse",
        "val_diag_interface_q_normal_mse",
        "val_loss_port_condition",
        "val_diag_port_condition_physical_mse",
        "val_diag_organizer_strength_mean",
        "val_loss_org_me",
        "val_loss_org_eh_factor",
        "val_loss_org_mass_align",
        "val_loss_org_mm",
        "val_loss_org_direct",
        "val_org_env_mass_max",
        "val_org_module_mass_max",
        "val_org_mass_l1",
        "val_org_env_effective_hyperedges",
        "val_org_module_effective_hyperedges",
        "val_org_max_A_eh_mass",
        "val_org_max_A_mh_mass",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    max_train_batches = args.max_train_batches
    if max_train_batches is None and training_cfg.get("max_train_batches_per_epoch") is not None:
        max_train_batches = int(training_cfg["max_train_batches_per_epoch"])
    max_val_batches = args.max_val_batches
    if max_val_batches is None and training_cfg.get("max_val_batches") is not None:
        max_val_batches = int(training_cfg["max_val_batches"])

    best_metric = math.inf
    for epoch in range(1, epochs + 1):
        effective_mode, effective_ratio = effective_port_condition_settings(epoch, training_cfg)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            loss_cfg,
            training_cfg,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(training_cfg.get("amp", False)),
            max_batches=max_train_batches,
            local_port_condition_mode=effective_mode,
            mixed_teacher_ratio=effective_ratio,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            loss_cfg,
            training_cfg,
            optimizer=None,
            scaler=None,
            amp=bool(training_cfg.get("amp", False)),
            max_batches=max_val_batches,
            local_port_condition_mode=effective_mode,
            mixed_teacher_ratio=effective_ratio,
        )
        row = {
            "epoch": epoch,
            "loss_total": train_metrics.get("loss_total", math.nan),
            "loss_field": train_metrics.get("loss_field", math.nan),
            "loss_internal_temperature": train_metrics.get("loss_internal_temperature", math.nan),
            "loss_interface": train_metrics.get("loss_interface", math.nan),
            "diag_interface_T_surface_mse": train_metrics.get("diag_interface_T_surface_mse", math.nan),
            "diag_interface_q_normal_mse": train_metrics.get("diag_interface_q_normal_mse", math.nan),
            "loss_port_condition": train_metrics.get("loss_port_condition", math.nan),
            "diag_port_condition_physical_mse": train_metrics.get("diag_port_condition_physical_mse", math.nan),
            "diag_organizer_strength_mean": train_metrics.get("diag_organizer_strength_mean", math.nan),
            "loss_org_me": train_metrics.get("loss_org_me", math.nan),
            "loss_org_eh_factor": train_metrics.get("loss_org_eh_factor", math.nan),
            "loss_org_mass_align": train_metrics.get("loss_org_mass_align", math.nan),
            "loss_org_mm": train_metrics.get("loss_org_mm", math.nan),
            "loss_org_direct": train_metrics.get("loss_org_direct", math.nan),
            "org_env_mass_max": train_metrics.get("org_env_mass_max", math.nan),
            "org_module_mass_max": train_metrics.get("org_module_mass_max", math.nan),
            "org_mass_l1": train_metrics.get("org_mass_l1", math.nan),
            "org_env_effective_hyperedges": train_metrics.get("org_env_effective_hyperedges", math.nan),
            "org_module_effective_hyperedges": train_metrics.get("org_module_effective_hyperedges", math.nan),
            "org_max_A_eh_mass": train_metrics.get("org_max_A_eh_mass", math.nan),
            "org_max_A_mh_mass": train_metrics.get("org_max_A_mh_mass", math.nan),
            "effective_local_port_condition_mode": effective_mode,
            "effective_mixed_teacher_ratio": effective_ratio,
            "val_loss_total": val_metrics.get("loss_total", math.nan),
            "val_loss_field": val_metrics.get("loss_field", math.nan),
            "val_loss_internal_temperature": val_metrics.get("loss_internal_temperature", math.nan),
            "val_loss_interface": val_metrics.get("loss_interface", math.nan),
            "val_diag_interface_T_surface_mse": val_metrics.get("diag_interface_T_surface_mse", math.nan),
            "val_diag_interface_q_normal_mse": val_metrics.get("diag_interface_q_normal_mse", math.nan),
            "val_loss_port_condition": val_metrics.get("loss_port_condition", math.nan),
            "val_diag_port_condition_physical_mse": val_metrics.get("diag_port_condition_physical_mse", math.nan),
            "val_diag_organizer_strength_mean": val_metrics.get("diag_organizer_strength_mean", math.nan),
            "val_loss_org_me": val_metrics.get("loss_org_me", math.nan),
            "val_loss_org_eh_factor": val_metrics.get("loss_org_eh_factor", math.nan),
            "val_loss_org_mass_align": val_metrics.get("loss_org_mass_align", math.nan),
            "val_loss_org_mm": val_metrics.get("loss_org_mm", math.nan),
            "val_loss_org_direct": val_metrics.get("loss_org_direct", math.nan),
            "val_org_env_mass_max": val_metrics.get("org_env_mass_max", math.nan),
            "val_org_module_mass_max": val_metrics.get("org_module_mass_max", math.nan),
            "val_org_mass_l1": val_metrics.get("org_mass_l1", math.nan),
            "val_org_env_effective_hyperedges": val_metrics.get("org_env_effective_hyperedges", math.nan),
            "val_org_module_effective_hyperedges": val_metrics.get("org_module_effective_hyperedges", math.nan),
            "val_org_max_A_eh_mass": val_metrics.get("org_max_A_eh_mass", math.nan),
            "val_org_max_A_mh_mass": val_metrics.get("org_max_A_mh_mass", math.nan),
        }
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        metric = float(row["val_loss_total"])
        if math.isfinite(metric) and metric < best_metric:
            best_metric = metric
            save_checkpoint(
                run_dir / "best_model.pt",
                model=model,
                model_config=model_config,
                train_config=cfg,
                dataset=train_dataset,
                epoch=epoch,
                best_metric=best_metric,
            )
            print(f'\nModel improving! Saving new best model...')
        save_checkpoint(
            run_dir / "latest_model.pt",
            model=model,
            model_config=model_config,
            train_config=cfg,
            dataset=train_dataset,
            epoch=epoch,
            best_metric=best_metric,
        )
        save_global_loss_plots(history_path, run_dir)
        print(
            f"[epoch {epoch:04d}] loss={row['loss_total']:.4e} field={row['loss_field']:.4e} "
            f"internal={row['loss_internal_temperature']:.4e} interface={row['loss_interface']:.4e} "
            f"iface_T={row['diag_interface_T_surface_mse']:.4e} iface_q={row['diag_interface_q_normal_mse']:.4e} "
            f"port={row['loss_port_condition']:.4e} port_phys={row['diag_port_condition_physical_mse']:.4e} "
            f"org_me={row['loss_org_me']:.3e} org_eh={row['loss_org_eh_factor']:.3e} "
            f"org_mass_l1={row['org_mass_l1']:.3e} "
            f"mode={effective_mode} ratio={effective_ratio:.3f} "
            f"val={row['val_loss_total']:.4e}"
        )

    print(f"[done] saved global model run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
