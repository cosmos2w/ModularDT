"""Forward epoch model invocation, physical losses, and metric aggregation."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from channelthermal.model import ChannelThermalHONFModel
from channelthermal.training_tools.losses import (
    case_group_budget_loss,
    channelthermal_field_mse,
    induced_pair_cost_loss,
)
from honf_forward_core.config import ROUTING_TYPED_TEMPERATURE_NAMES
from honf_forward_core.training.diagnostics import (
    HONF_DIAGNOSTIC_KEYS,
    compute_honf_diagnostics,
    organizer_regularization_loss,
)
from honf_runtime.compat import autocast_context, recursive_to_device


TASK_TRAINED_FUNCTIONAL_COALESCENCE = "task_trained_functional_coalescence_honf"

INTERFACE_DIAGNOSTIC_KEYS = (
    "interaction_local_neighbor_count_mean",
    "interaction_main_context_norm_mean",
    "interaction_direct_module_context_norm_mean",
    "interaction_regional_context_norm_mean",
    "interaction_coarse_context_norm_mean",
    "interaction_local_context_norm_mean",
    "interaction_main_context_fraction_mean",
    "interaction_coarse_context_fraction_mean",
    "interaction_local_context_fraction_mean",
    "interaction_total_latent_count",
    "interaction_group_count",
    "interaction_module_group_incidence_count",
    "interaction_environment_group_incidence_count",
    "interaction_group_read_degree_mean",
    "interaction_group_read_weight_mass_mean",
    "interaction_group_read_geometric_availability_mean",
    "interaction_group_module_degree_mean",
    "interaction_group_module_degree_max",
    "interaction_group_environment_degree_mean",
    "interaction_group_occupancy_mean",
    "interaction_group_occupancy_envelope_mean",
    "interaction_group_covered_volume_ratio_mean",
    "interaction_regional_response_count_mean",
)

GRADIENT_DIAGNOSTIC_GROUPS = (
    "encoder", "backend", "head", "local_coupling",
    "group_prepare", "group_receiver", "coarse", "local",
    # Regional-response detail rows are additive observations.  The
    # historical aggregate groups above remain unchanged so old runs retain
    # their established interpretation.
    "regional_prepare", "regional_receiver", "direct_module",
    "coarse_group_source", "coarse_env_source",
)
GRADIENT_DIAGNOSTIC_KEYS = (
    "preclip_gradient_norm",
    "gradient_clip_scale",
    "parameter_update_norm",
    *[f"preclip_gradient_norm_{group}" for group in GRADIENT_DIAGNOSTIC_GROUPS],
    *[f"parameter_update_norm_{group}" for group in GRADIENT_DIAGNOSTIC_GROUPS],
)


def _diagnostic_parameter_group(name: str) -> str:
    if name.startswith("core.global_encoder.") or name.startswith("core.module_") or name.startswith("core.env_encoder."):
        return "encoder"
    if name.startswith("core.backend.") or name.startswith("core.common.coarse") or name.startswith("core.common.local"):
        return "backend"
    if name.startswith("core.common.field_head.") or name.startswith("local_coupling.port_head."):
        return "head"
    return "local_coupling"


def _diagnostic_detail_group(name: str) -> str | None:
    """Split sparse learning from its bypasses; retain historical backend totals."""
    if name.startswith("core.common.coarse"):
        return "coarse"
    if name.startswith("core.common.local"):
        return "local"
    if name.startswith((
        "core.backend.query_module_message.",
        "core.backend.query_module_output.",
    )):
        return "direct_module"
    # Dense and regional backends intentionally share the typed EM/update
    # names.  These rows therefore describe environmental response
    # preparation for Dense as well; the report labels them as such rather
    # than treating the labels as proof of the regional candidate's grouping.
    if name.startswith((
        "core.backend.em_message.",
        "core.backend.env_update.",
    )):
        return "regional_prepare"
    if name.startswith((
        "core.backend.env_query.",
        "core.backend.env_geometry_bias.",
        "core.backend.env_attention.",
    )):
        return "regional_receiver"
    if name.startswith(("core.backend.receiver_query.", "core.backend.receiver_bias.", "core.backend.group_key.")):
        return "group_receiver"
    if name.startswith(tuple(f"core.backend.{part}." for part in (
        "module_membership", "environment_membership", "module_message",
        "environment_message", "group_input", "group_residual", "group_norm", "group_value",
    ))):
        return "group_prepare"
    return None


def _fp64_group_norm(named_values: list[tuple[str, torch.Tensor]]) -> tuple[float, Dict[str, float]]:
    total = 0.0
    by_group = {group: 0.0 for group in GRADIENT_DIAGNOSTIC_GROUPS}
    for name, value in named_values:
        squared = float(value.detach().double().square().sum().cpu())
        total += squared
        by_group[_diagnostic_parameter_group(name)] += squared
        detail = _diagnostic_detail_group(name)
        if detail is not None:
            by_group[detail] += squared
        # Source-level detail supplements, rather than replaces, the existing
        # coarse/backend totals. These are ordinary gradient/update logs.
        if name.startswith("core.common.coarse_group_attention."):
            by_group["coarse_group_source"] += squared
        elif name.startswith("core.common.coarse_env_attention."):
            by_group["coarse_env_source"] += squared
    return math.sqrt(total), {group: math.sqrt(value) for group, value in by_group.items()}


def pack_scalar_metrics(tensor_metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Transfer a batch of scalar metrics to the CPU in one synchronization."""

    names = tuple(tensor_metrics)
    values = torch.stack(
        tuple(tensor_metrics[name].detach().reshape(()) for name in names)
    ).cpu().tolist()
    return {name: float(value) for name, value in zip(names, values)}


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


def _routing_sparsification_settings(model: ChannelThermalHONFModel) -> Any:
    """Return the optional routed science settings without widening legacy APIs."""

    core_config = getattr(getattr(model, "config", None), "core_honf", None)
    interface_config = getattr(core_config, "interface_model", None)
    routing_config = getattr(interface_config, "routing", None)
    return getattr(routing_config, "sparsification", None)


def _case_group_budget_settings(
    model: ChannelThermalHONFModel,
    loss_cfg: Dict[str, Any],
) -> tuple[bool, float]:
    """Resolve the one opt-in expected optional-group objective."""

    core_config = getattr(getattr(model, "config", None), "core_honf", None)
    architecture = str(getattr(core_config, "forward_architecture", ""))
    budget_enabled = architecture == "budgeted_group_control_honf"
    if budget_enabled and "case_group_budget_weight" not in loss_cfg:
        raise ValueError(
            "budgeted_group_control_honf requires numeric loss.case_group_budget_weight; "
            "calibration may use an explicit zero value."
        )
    weight = float(loss_cfg.get("case_group_budget_weight", 0.0))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("case_group_budget_weight must be finite and nonnegative.")
    if weight != 0.0 and not budget_enabled:
        raise ValueError(
            "case_group_budget_weight is only valid for budgeted_group_control_honf."
        )
    return budget_enabled, weight


def _case_group_budget_schedule(
    model: ChannelThermalHONFModel,
    loss_cfg: Dict[str, Any],
) -> str:
    """Return the schedule encoded in the model config.

    The schedule is part of the model architecture contract so checkpoint
    evaluation and resume reconstruct the same objective.  ``loss_cfg`` is
    intentionally not a fallback: historical budgeted checkpoints have the
    static default, while rescue checkpoints carry the explicit v2 block in
    ``CaseGroupBudgetConfig``.
    """

    del loss_cfg
    configured = getattr(model, "budgeted_schedule_mode", "static")
    value = str(configured).strip().lower()
    if value not in {"static", "dense_to_sparse_v2"}:
        raise ValueError(
            "case_group_budget.schedule must be 'static' or 'dense_to_sparse_v2'."
        )
    return value


def assemble_channelthermal_loss_terms(
    output: dict[str, Any],
    batch: dict[str, Any],
    model: ChannelThermalHONFModel,
    loss_cfg: dict[str, Any],
    *,
    local_port_condition_mode: str,
    mixed_teacher_ratio: float,
    effective_internal_temperature_weight: float,
    effective_interface_weight: float,
    predicted_consistency_weight: float,
) -> dict[str, torch.Tensor]:
    """Assemble the physical objective and the optional induced-pair term.

    Keeping this calculation separate from :func:`run_epoch` gives the
    calibration/replay tools one authoritative loss assembly point.  The
    returned ``loss_physical`` is the unchanged supervised objective,
    ``loss_paircost`` is the dimensionless Eq. (18) surrogate, and ``loss`` is
    their sum with one configured fixed coefficient.
    """

    target = batch["field_targets"].float()
    point_weights = batch.get("point_weights")
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
    port_global_weight = effective_port_global_weight(
        loss_cfg, local_port_condition_mode, mixed_teacher_ratio
    )
    port_supervised_weight = float(
        loss_cfg.get("port_supervised_weight", loss_cfg.get("port_condition_weight", 0.0))
    )
    port_smoothness_weight = float(loss_cfg.get("port_smoothness_weight", 0.0))
    loss_port = port_condition_loss(output, batch, loss_cfg) if port_supervised_weight != 0.0 else zero
    loss_port_smoothness = (
        port_cyclic_smoothness_loss(output, batch) if port_smoothness_weight != 0.0 else zero
    )
    loss_port_global = port_global_consistency_loss(output) if port_global_weight != 0.0 else zero
    if "predicted_port_internal_temperature" in output and "predicted_port_interface" in output:
        pred_cons_internal = internal_loss(
            {
                "pred_internal_temperature": output["predicted_port_internal_temperature"],
                "pred_field": output["pred_field"],
            },
            batch,
        )
        pred_cons_interface = interface_loss(
            {"pred_interface": output["predicted_port_interface"], "pred_field": output["pred_field"]},
            batch,
            loss_cfg,
        )
        loss_predicted_consistency = pred_cons_internal + pred_cons_interface
    else:
        pred_cons_internal = zero
        pred_cons_interface = zero
        loss_predicted_consistency = zero
    loss_org = organizer_regularization(output, loss_cfg)

    budget_enabled, budget_weight = _case_group_budget_settings(model, loss_cfg)
    budget_schedule = _case_group_budget_schedule(model, loss_cfg)
    # Keep the live per-case excess scalar raw.  The ThermalChannel objective
    # applies the calibrated coefficient and its deterministic c(t) ramp once.
    effective_budget_weight = output["pred_field"].new_tensor(budget_weight)
    continuation = output.get("case_group_budget_continuation")
    routing_strength = output.get("case_group_budget_routing_strength")
    if budget_schedule == "dense_to_sparse_v2":
        if not torch.is_tensor(continuation) or continuation.numel() == 0:
            raise RuntimeError(
                "dense_to_sparse_v2 requires the live P0 continuation scalar in model output."
            )
        effective_budget_weight = effective_budget_weight * continuation.mean().to(
            device=effective_budget_weight.device,
            dtype=effective_budget_weight.dtype,
        )
    loss_group_budget = case_group_budget_loss(
        output,
        enabled=budget_enabled,
        # Keep the live path present during the one-time zero-weight gradient
        # calibration. A missing scalar must never silently disable the mode.
        require_live=budget_enabled,
    )

    sparsification = _routing_sparsification_settings(model)
    paircost_enabled = bool(getattr(sparsification, "enabled", False))
    paircost_weight = float(getattr(sparsification, "cost_weight", 0.0))
    if paircost_weight != 0.0 and not paircost_enabled:
        raise ValueError("A nonzero induced pair-cost coefficient requires routing.sparsification.enabled.")
    loss_paircost = induced_pair_cost_loss(
        output,
        enabled=paircost_enabled,
        # The zero-weight pre-calibration profile still needs the canonical
        # live components so the parent can measure their router gradients;
        # silently accepting a detached reporting scalar would invalidate that
        # calibration.
        require_components=paircost_enabled,
    )
    loss_physical = (
        float(loss_cfg.get("field_mse_weight", 1.0)) * loss_field
        + float(effective_internal_temperature_weight) * loss_internal
        + float(effective_interface_weight) * loss_interface
        + port_supervised_weight * loss_port
        + port_smoothness_weight * loss_port_smoothness
        + float(port_global_weight) * loss_port_global
        + float(predicted_consistency_weight) * loss_predicted_consistency
        + loss_org
    )
    loss = loss_physical + paircost_weight * loss_paircost + effective_budget_weight * loss_group_budget
    detail_terms: dict[str, torch.Tensor] = {}
    architecture = str(model.config.core_honf.forward_architecture)
    if architecture == TASK_TRAINED_FUNCTIONAL_COALESCENCE:
        detail_complexity = output.get("functional_detail_expected_complexity")
        detail_expected_r = output.get("functional_detail_expected_R")
        detail_ramp = output.get("functional_detail_ramp_weight")
        if not torch.is_tensor(detail_complexity) or not torch.is_tensor(detail_expected_r):
            raise RuntimeError(
                "task_trained_functional_coalescence_honf must expose live expected complexity and R."
            )
        if not torch.is_tensor(detail_ramp) or detail_ramp.numel() != 1:
            raise RuntimeError(
                "task_trained_functional_coalescence_honf must expose one scalar ramp weight."
            )
        configured_detail_weight = float(loss_cfg.get("functional_detail_complexity_weight", 0.0))
        if not math.isfinite(configured_detail_weight) or configured_detail_weight < 0.0:
            raise ValueError("functional_detail_complexity_weight must be finite and nonnegative.")
        raw_detail_weight = output["pred_field"].new_tensor(configured_detail_weight)
        effective_detail_weight = raw_detail_weight * detail_ramp.to(
            device=raw_detail_weight.device,
            dtype=raw_detail_weight.dtype,
        )
        detail_complexity = detail_complexity.mean()
        detail_expected_r = detail_expected_r.mean()
        # Structural pressure is a training objective. Validation loss remains
        # the existing physical task objective used for checkpoint selection.
        if model.training:
            loss = loss + effective_detail_weight * detail_complexity
        detail_terms = {
            "loss_functional_detail_complexity": detail_complexity,
            "functional_detail_complexity_weight": effective_detail_weight,
            "functional_detail_configured_complexity_weight": raw_detail_weight,
            "functional_detail_ramp_weight": detail_ramp.reshape(()),
            "functional_detail_expected_R": detail_expected_r,
            "functional_detail_stochastic_fraction": output.get(
                "functional_detail_stochastic_fraction",
                output["pred_field"].new_zeros(()),
            ),
        }
        for phase_name in ("p0", "p1", "p2"):
            for suffix in ("expected_complexity", "expected_R", "actual_R"):
                metric_key = f"functional_detail_{phase_name}_{suffix}"
                value = output.get(metric_key)
                if torch.is_tensor(value):
                    detail_terms[metric_key] = value.float().mean()
                else:
                    detail_terms[metric_key] = output["pred_field"].new_full((), math.nan)
    return {
        "loss": loss,
        "loss_physical": loss_physical,
        "loss_group_budget": loss_group_budget,
        "case_group_budget_weight": effective_budget_weight,
        "case_group_budget_final_weight": output["pred_field"].new_tensor(budget_weight),
        "case_group_budget_schedule": budget_schedule,
        "case_group_budget_routing_strength": (
            routing_strength.mean()
            if torch.is_tensor(routing_strength) and routing_strength.numel()
            else output["pred_field"].new_ones(())
        ),
        "case_group_budget_continuation": (
            continuation.mean()
            if torch.is_tensor(continuation) and continuation.numel()
            else output["pred_field"].new_ones(())
        ),
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
        "loss_paircost": loss_paircost,
        "paircost_weight": output["pred_field"].new_tensor(paircost_weight),
        "port_global_weight": output["pred_field"].new_tensor(port_global_weight),
        **detail_terms,
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
    record_gradient_diagnostics: bool = False,
) -> Dict[str, float]:
    """Run one train/validation epoch and return averaged loss/diagnostic scalars."""

    training = optimizer is not None
    model.train(training)
    sums: Dict[str, float] = {}
    one_shot_metrics: Dict[str, float] = {}
    count = 0
    sparsification = _routing_sparsification_settings(model)
    paircost_enabled = bool(getattr(sparsification, "enabled", False))
    budget_enabled, _budget_weight = _case_group_budget_settings(model, loss_cfg)
    iterator = tqdm(loader, leave=False, desc="train" if training else "val")
    for batch_idx, batch in enumerate(iterator, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        batch = recursive_to_device(batch, device)
        target = batch["field_targets"].float()
        with torch.set_grad_enabled(training):
            with autocast_context(device, amp):
                port_global_weight = effective_port_global_weight(
                    loss_cfg, local_port_condition_mode, mixed_teacher_ratio
                )
                output = model(
                    **make_model_inputs(
                        batch,
                        local_port_condition_mode=local_port_condition_mode,
                        mixed_teacher_ratio=mixed_teacher_ratio,
                        return_predicted_port_outputs=bool(predicted_consistency_weight > 0.0),
                        return_port_global_consistency=bool(port_global_weight != 0.0),
                    )
                )
                loss_terms = assemble_channelthermal_loss_terms(
                    output,
                    batch,
                    model,
                    loss_cfg,
                    local_port_condition_mode=local_port_condition_mode,
                    mixed_teacher_ratio=mixed_teacher_ratio,
                    effective_internal_temperature_weight=effective_internal_temperature_weight,
                    effective_interface_weight=effective_interface_weight,
                    predicted_consistency_weight=predicted_consistency_weight,
                )
                loss = loss_terms["loss"]
                loss_field = loss_terms["loss_field"]
                loss_internal = loss_terms["loss_internal_temperature"]
                loss_interface = loss_terms["loss_interface"]
                loss_port = loss_terms["loss_port_condition"]
                loss_port_smoothness = loss_terms["loss_port_smoothness"]
                loss_port_global = loss_terms["loss_port_global_consistency"]
                loss_predicted_consistency = loss_terms["loss_predicted_consistency"]
                pred_cons_internal = loss_terms["loss_predicted_consistency_internal"]
                pred_cons_interface = loss_terms["loss_predicted_consistency_interface"]
                loss_org = loss_terms["loss_organizer"]
                loss_paircost = loss_terms["loss_paircost"]
                loss_group_budget = loss_terms["loss_group_budget"]
        if training:
            optimizer.zero_grad(set_to_none=True)
            clip_norm = float(gradient_clip_norm or 0.0)
            capture_update = bool(record_gradient_diagnostics and batch_idx == 1)
            routing_model = getattr(getattr(model.config, "core_honf", None), "forward_architecture", "") == "routed_pairwise_honf"
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if clip_norm > 0.0 or capture_update:
                    # AMP gradients must be unscaled before clipping; otherwise
                    # the threshold applies to scaled values and is meaningless.
                    scaler.unscale_(optimizer)
                named_gradients = [
                    (name, parameter.grad)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ] if capture_update else []
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                    if capture_update and parameter.requires_grad
                }
                if capture_update:
                    total_grad, grouped_grad = _fp64_group_norm(named_gradients)
                    if routing_model:
                        router_grad, _ = _fp64_group_norm([
                            (name, value) for name, value in named_gradients
                            if name.startswith("core.backend.router.")
                        ])
                if clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                named_gradients = [
                    (name, parameter.grad)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ] if capture_update else []
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                    if capture_update and parameter.requires_grad
                }
                if capture_update:
                    total_grad, grouped_grad = _fp64_group_norm(named_gradients)
                    if routing_model:
                        router_grad, _ = _fp64_group_norm([
                            (name, value) for name, value in named_gradients
                            if name.startswith("core.backend.router.")
                        ])
                if clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            if capture_update:
                named_updates = [
                    (name, parameter.detach() - before[name])
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and name in before
                ]
                total_update, grouped_update = _fp64_group_norm(named_updates)
                if routing_model:
                    router_update, _ = _fp64_group_norm([
                        (name, value) for name, value in named_updates
                        if name.startswith("core.backend.router.")
                    ])
                one_shot_metrics = {
                    "preclip_gradient_norm": total_grad,
                    "gradient_clip_scale": min(1.0, clip_norm / max(total_grad, 1.0e-300)) if clip_norm > 0.0 else 1.0,
                    "parameter_update_norm": total_update,
                    **{f"preclip_gradient_norm_{key}": value for key, value in grouped_grad.items()},
                    **{f"parameter_update_norm_{key}": value for key, value in grouped_update.items()},
                }
                if routing_model:
                    one_shot_metrics.update(routing_preclip_gradient_norm=router_grad,
                                            routing_parameter_update_norm=router_update)
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
            interaction_aux = output.get("interaction_aux")
            metric_tensors = {
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
            if paircost_enabled or budget_enabled:
                metric_tensors["loss_physical"] = loss_terms["loss_physical"]
            if budget_enabled:
                metric_tensors["loss_group_budget"] = loss_group_budget
                metric_tensors["case_group_budget_weight"] = loss_terms["case_group_budget_weight"]
                metric_tensors["case_group_budget_final_weight"] = loss_terms[
                    "case_group_budget_final_weight"
                ]
                metric_tensors["case_group_budget_routing_strength"] = loss_terms[
                    "case_group_budget_routing_strength"
                ]
                metric_tensors["case_group_budget_continuation"] = loss_terms[
                    "case_group_budget_continuation"
                ]
                expected_count = output.get("case_group_budget_expected_optional_count")
                metric_tensors["case_group_budget_expected_optional_count"] = (
                    expected_count.mean()
                    if torch.is_tensor(expected_count) and expected_count.numel()
                    else pred.new_full((), math.nan)
                )
            if paircost_enabled:
                metric_tensors["loss_paircost"] = loss_paircost
                # Keep the four learned temperatures in the ordinary metrics
                # row for the opt-in science profile.  The names deliberately
                # avoid the routing_* prefix so the existing routing-summary
                # sidecar continues to receive only backend diagnostics.
                for parameter_name in ROUTING_TYPED_TEMPERATURE_NAMES:
                    relation = parameter_name.removeprefix("log_temperature_")
                    value = (
                        interaction_aux.get("routing_temperature_" + relation)
                        if isinstance(interaction_aux, dict)
                        else None
                    )
                    metric_tensors["temperature_" + relation] = (
                        value if torch.is_tensor(value) else pred.new_full((), math.nan)
                    )
            if "loss_functional_detail_complexity" in loss_terms:
                detail_metric_keys = (
                    "loss_functional_detail_complexity",
                    "functional_detail_complexity_weight",
                    "functional_detail_configured_complexity_weight",
                    "functional_detail_ramp_weight",
                    "functional_detail_expected_R",
                    "functional_detail_stochastic_fraction",
                    *(
                        f"functional_detail_{phase}_{suffix}"
                        for phase in ("p0", "p1", "p2")
                        for suffix in ("expected_complexity", "expected_R", "actual_R")
                    ),
                )
                for key in detail_metric_keys:
                    metric_tensors[key] = loss_terms[key]
            metrics = pack_scalar_metrics(metric_tensors)
            metrics.update(
                {
                    "effective_port_global_consistency_weight": float(port_global_weight),
                    "effective_predicted_consistency_weight": float(predicted_consistency_weight),
                    "effective_internal_temperature_weight": float(effective_internal_temperature_weight),
                    "effective_interface_weight": float(effective_interface_weight),
                }
            )
            if isinstance(interaction_aux, dict):
                if getattr(getattr(model.config, "core_honf", None), "forward_architecture", "") == "routed_pairwise_honf":
                    for key, value in interaction_aux.items():
                        if key.startswith(("routing_", "initial_port_routing_", "provisional_routing_")) and torch.is_tensor(value):
                            # Only compact backend summaries are present in ordinary training.
                            metrics["routing_summary_" + key] = (
                                float(value.detach().float().mean().cpu()) if value.numel() else math.nan
                            )
                mapping = {
                    "interaction_local_neighbor_count_mean": "local_neighbor_count",
                    "interaction_main_context_norm_mean": "main_context_norm",
                    "interaction_direct_module_context_norm_mean": "dense_module_context_norm",
                    "interaction_regional_context_norm_mean": "regional_context_norm",
                    "interaction_coarse_context_norm_mean": "coarse_context_norm",
                    "interaction_local_context_norm_mean": "local_context_norm",
                    "interaction_main_context_fraction_mean": "main_context_fraction",
                    "interaction_coarse_context_fraction_mean": "coarse_context_fraction",
                    "interaction_local_context_fraction_mean": "local_context_fraction",
                    "interaction_group_read_degree_mean": "group_read_degree",
                    "interaction_group_read_weight_mass_mean": "group_read_weight_mass",
                    "interaction_group_read_geometric_availability_mean": "group_read_geometric_availability",
                    "interaction_group_module_degree_mean": "group_module_degree",
                    "interaction_group_environment_degree_mean": "group_environment_degree",
                    "interaction_group_occupancy_mean": "group_occupancy",
                    "interaction_group_occupancy_envelope_mean": "group_occupancy_envelope",
                    "interaction_group_covered_volume_ratio_mean": "group_covered_volume_ratio",
                    "interaction_regional_response_count_mean": "regional_response_count",
                }
                for metric_name, aux_name in mapping.items():
                    value = interaction_aux.get(aux_name)
                    metrics[metric_name] = float(value.detach().float().mean().cpu()) if torch.is_tensor(value) else math.nan
                metrics["interaction_total_latent_count"] = float(
                    interaction_aux.get("coarse_latent_count", 0) + interaction_aux.get("main_latent_count", 0)
                )
                for key in (
                    "group_count",
                    "module_group_incidence_count",
                    "environment_group_incidence_count",
                ):
                    value = interaction_aux.get(f"{key}_per_case")
                    metrics[f"interaction_{key}"] = (
                        float(value.detach().float().mean().cpu())
                        if torch.is_tensor(value) and value.numel()
                        else math.nan
                    )
                group_module_degree = interaction_aux.get("group_module_degree")
                metrics["interaction_group_module_degree_max"] = (
                    float(group_module_degree.detach().float().max().cpu())
                    if torch.is_tensor(group_module_degree) and group_module_degree.numel()
                    else math.nan
                )
            else:
                metrics.update({key: math.nan for key in INTERFACE_DIAGNOSTIC_KEYS})
            # Keep the metrics row schema stable when an older/fake model
            # omits optional organizer diagnostics.  In particular, the
            # case-adaptive residual metrics are direct organizer values and
            # should not be reconstructed here from legacy edge strength.
            architecture = getattr(getattr(model.config, "core_honf", None), "forward_architecture", "legacy_honf")
            if architecture == "legacy_honf":
                metrics.update(honf_diag)
                for key in HONF_DIAGNOSTIC_KEYS:
                    metrics.setdefault(key, float(honf_diag.get(key, 0.0)))
            else:
                # Topology quantities are not applicable to these baselines;
                # NaN is intentional and avoids fabricated K=1/zero metrics.
                metrics.update({key: math.nan for key in HONF_DIAGNOSTIC_KEYS})
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        iterator.set_postfix(loss=f"{metrics['loss_total']:.3e}", field=f"{metrics['field_mse']:.3e}")
    if count == 0:
        empty_metrics = {
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
        if paircost_enabled:
            empty_metrics["loss_physical"] = math.nan
            empty_metrics["loss_paircost"] = math.nan
            empty_metrics.update(
                {
                    "temperature_" + name.removeprefix("log_temperature_"): math.nan
                    for name in ROUTING_TYPED_TEMPERATURE_NAMES
                }
            )
        if budget_enabled:
            empty_metrics.update(
                {
                    "loss_physical": math.nan,
                    "loss_group_budget": math.nan,
                    "case_group_budget_weight": math.nan,
                    "case_group_budget_final_weight": math.nan,
                    "case_group_budget_routing_strength": math.nan,
                    "case_group_budget_continuation": math.nan,
                    "case_group_budget_expected_optional_count": math.nan,
                }
            )
        return empty_metrics
    averaged = {key: value / count for key, value in sums.items()}
    if training:
        averaged.update({key: math.nan for key in GRADIENT_DIAGNOSTIC_KEYS})
        averaged.update(one_shot_metrics)
        if getattr(getattr(model.config, "core_honf", None), "forward_architecture", "") == "routed_pairwise_honf":
            averaged.setdefault("routing_preclip_gradient_norm", math.nan)
            averaged.setdefault("routing_parameter_update_norm", math.nan)
    return averaged
