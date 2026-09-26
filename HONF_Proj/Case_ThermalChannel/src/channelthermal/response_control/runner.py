"""Argumentized ThermalChannel full-access response-control fitting runner.

The runner converts a selected native checkpoint into the explicit
``three_term_full_access_honf`` refit target, samples typed train stencils,
and supports a one-update correctness preflight or paired staged fits. All
one-time state and reports are written to a caller-supplied ignored directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter

from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.evaluation.loading import load_model
from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.interaction_evidence.response_dataset import ResponseStencil
from channelthermal.interaction_evidence.types import EvidenceSplit, MeasuredQuantity
from channelthermal.model import ChannelThermalHONFModel
from honf_forward_core.config import UnifiedForwardConfig
from honf_forward_core.interface_fields.checkpoint_warm_start import (
    warm_start_three_term_full_access,
)
from honf_forward_core.interface_fields.core import InterfaceFieldCore

from .algebra import MixedResponseSpec
from .contracts import DesignInput, RoleQuery, context_inputs, role_queries_from_stencil
from .derivative_check import check_pressure_peak_ad_fd
from .evaluation import evaluate_stencil
from .losses import ThermalLossScales
from .native import DifferentiableThermalOperator
from .paired import run_paired_staged_fits, write_paired_training_curves
from .resume_provenance import validate_paired_resume_provenance
from .sampling import ReceiverSamplingConfig, SamplingSummary, sample_training_panel
from .thermal import pressure_section_masks
from .training import (
    StagedTrainingConfig,
    TrainingStage,
    load_staged_training_config,
    restore_checkpoint_payload,
    run_staged_fit,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.tmp-", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot serialize {type(value).__name__} to the run manifest.")


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.tmp-", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_dataset_path(checkpoint: Mapping[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    train_config = checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {}) if isinstance(train_config, Mapping) else {}
    configured = dataset_config.get("packed_h5_path")
    if not configured:
        raise ValueError("Checkpoint has no packed_h5_path; pass --dataset explicitly.")
    return Path(str(configured)).expanduser().resolve()


def _make_refit_model(
    source_model: ChannelThermalHONFModel,
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[ChannelThermalHONFModel, dict[str, Any]]:
    model = copy.deepcopy(source_model)
    model_config = copy.deepcopy(source_model.config)
    core_payload = model_config.core_honf.to_dict()
    core_payload["forward_architecture"] = "three_term_full_access_honf"
    interface = dict(core_payload.get("interface_model") or {})
    # The clean full-access architecture fixes these normalizers. This is a
    # documented refit of the inherited organizer controls, not an identity
    # conversion of Run1508 outputs.
    interface["source_normalizer"] = "entmax15"
    interface["query_normalizer"] = "entmax15"
    interface["environment_refinement_normalizer"] = "entmax15"
    for name in ("module_temperature", "environment_temperature", "query_temperature"):
        interface[name] = 1.0
    core_payload["interface_model"] = interface
    core_config = UnifiedForwardConfig.from_dict(core_payload)
    model_config.core_honf = core_config
    model.config = model_config
    model.core = InterfaceFieldCore(core_config).to(device)
    normalization_config = checkpoint.get(
        "global_normalization_config",
        checkpoint.get("train_config", {}).get("dataset", {}),
    )
    model.set_global_target_normalization(
        checkpoint.get("global_normalization_stats", {}),
        normalize_targets=bool(normalization_config.get("normalize_targets", False)),
    )
    return model.to(device), {
        "forward_architecture": core_config.forward_architecture,
        "source_normalizer": interface["source_normalizer"],
        "query_normalizer": interface["query_normalizer"],
        "environment_refinement_normalizer": interface["environment_refinement_normalizer"],
        "source_prediction_mode": source_model.config.channelthermal.internal_prediction_mode,
        "local_module_params_from_used_ports": bool(
            source_model.config.channelthermal.local_module_params_from_used_ports
        ),
        "prediction_identity_claim": False,
    }


def _materialize_and_warm_start(
    source_model: ChannelThermalHONFModel,
    target_model: ChannelThermalHONFModel,
    operator: DifferentiableThermalOperator,
    stencil: ResponseStencil,
    *,
    device: torch.device,
) -> dict[str, Any]:
    queries = role_queries_from_stencil(stencil, device=device)
    design = DesignInput.from_state(stencil.baseline.design, device=device)
    with torch.no_grad():
        operator(design, context_inputs(stencil.baseline.context), queries)
    source_state = source_model.core.state_dict()
    target_state = target_model.core.state_dict()
    initialized, inventory = warm_start_three_term_full_access(
        source_state,
        target_state,
        source_architecture=str(source_model.config.core_honf.forward_architecture),
    )
    target_model.core.load_state_dict(initialized, strict=True)
    return {
        "copied_tensor_count": len(inventory["copied"]),
        "initialized_tensor_count": len(inventory["initialized"]),
        "discarded_source_tensor_count": len(inventory["discarded_source"]),
        "source_architecture": inventory["source_architecture"],
        "prediction_identity_claim": inventory["prediction_identity_claim"],
        "copied_tensors": list(inventory["copied"]),
        "initialized_tensors": list(inventory["initialized"]),
        "discarded_source_tensors": list(inventory["discarded_source"]),
    }


def _parameter_inventory(model: torch.nn.Module) -> dict[str, Any]:
    uninitialized = [
        name for name, parameter in model.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ]
    if uninitialized:
        raise RuntimeError(f"Model has lazy parameters after role materialization: {uninitialized}.")
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    local = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("local_coupling.local_surrogate.")
    ]
    local_trainable = [name for name, parameter in local if parameter.requires_grad]
    return {
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameter_names": [name for name, _ in trainable],
        "core_trainable_parameter_count": sum(
            parameter.numel() for name, parameter in trainable if name.startswith("core.")
        ),
        "core_trainable_parameter_names": [
            name for name, _ in trainable if name.startswith("core.")
        ],
        "local_surrogate_attached": bool(model.local_coupling.has_local_surrogate),
        "local_surrogate_parameter_count": sum(parameter.numel() for _, parameter in local),
        "local_surrogate_trainable_parameter_names": local_trainable,
        "local_surrogate_frozen": bool(local) and not local_trainable,
        "local_surrogate_frozen_by_loader": bool(model.local_coupling.local_surrogate_frozen),
        "local_surrogate_checkpoint_provenance": model.local_coupling.local_surrogate_checkpoint_path,
        "local_surrogate_configured_checkpoint": model.config.channelthermal.local_surrogate_checkpoint_path,
        "optimizer_includes_all_trainable_parameters": True,
        "lazy_parameters_remaining": uninitialized,
    }


def _verify_optimizer_inventory(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, int | bool]:
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if actual != expected:
        raise RuntimeError(
            f"Optimizer parameter mismatch: {len(expected - actual)} trainable parameters omitted, "
            f"{len(actual - expected)} nontrainable/duplicate parameters included."
        )
    return {
        "trainable_parameter_tensors": len(expected),
        "optimizer_parameter_tensors": len(actual),
        "all_trainable_parameters_in_optimizer": True,
    }


def _make_input_template(dataset: GlobalChannelThermalDataset) -> dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError("Checkpoint packed dataset has no train cases for module-slot capacity.")
    sample = dataset[0]
    structure = sample.get("structure")
    if not isinstance(structure, Mapping):
        raise TypeError("Packed dataset sample has no structure mapping.")
    # Only these input-schema arrays reach the native callback. Targets and
    # masks from the sample are deliberately dropped at this boundary.
    return {
        "structure": {
            "module_centers": np.array(structure["module_centers"], copy=True),
            "material_params": np.array(structure["material_params"], copy=True),
        }
    }


def _rms_from_arrays(values: Sequence[np.ndarray], masks: Sequence[np.ndarray], channel: int) -> float:
    squares: list[np.ndarray] = []
    for array, mask in zip(values, masks):
        target = np.asarray(array, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        if valid.ndim == 1:
            valid = np.broadcast_to(valid[:, None], target.shape)
        selected = target[valid[:, channel], channel]
        if selected.size:
            squares.append(np.square(selected))
    if not squares:
        return 1.0
    return max(float(np.sqrt(np.mean(np.concatenate(squares)))), 1.0e-6)


def derive_training_scales(
    stencils: Sequence[ResponseStencil], *, smooth_peak_beta: float = 1.0
) -> ThermalLossScales:
    """Freeze physical loss scales from train stencils only."""

    if not stencils or any(stencil.split is not EvidenceSplit.TRAIN for stencil in stencils):
        raise ValueError("Loss-scale derivation requires only nonempty train stencils.")
    roles = tuple(stencils[0].baseline.output.roles)  # type: ignore[union-attr]
    value_scales: dict[str, tuple[float, ...]] = {}
    finite_scales: dict[str, tuple[float, ...]] = {}
    mixed_scales: dict[str, tuple[float, ...]] = {}
    for role_name in roles:
        role_examples = [
            record.output.roles[role_name]
            for stencil in stencils
            for record in stencil.records
            if record.output is not None
        ]
        channels = len(role_examples[0].channel_names)
        value_scales[role_name] = tuple(
            _rms_from_arrays(
                [role.values for role in role_examples],
                [role.valid_mask for role in role_examples],
                channel,
            )
            for channel in range(channels)
        )
        finite_blocks = [
            stencil.finite_change(label, role_name)
            for stencil in stencils
            for label in stencil.variants
        ]
        finite_scales[role_name] = tuple(
            _rms_from_arrays(
                [block.delta for block in finite_blocks],
                [block.valid_mask for block in finite_blocks],
                channel,
            )
            for channel in range(channels)
        )
        # There is no mixed noise floor in the current atlas. These scales
        # define units only; the loss builder leaves those labels unknown.
        mixed_scales[role_name] = finite_scales[role_name]

    baseline_pressure = [
        float(stencil.baseline.output.quantities["pressure_drop"].value)  # type: ignore[union-attr]
        for stencil in stencils
    ]
    pressure_deltas = [
        float(stencil.variants[label].output.quantities["pressure_drop"].value)  # type: ignore[union-attr]
        - float(stencil.baseline.output.quantities["pressure_drop"].value)  # type: ignore[union-attr]
        for stencil in stencils
        for label in stencil.variants
    ]
    pressure_value = max(float(np.sqrt(np.mean(np.square(baseline_pressure)))), 1.0e-6)
    pressure_response = max(float(np.sqrt(np.mean(np.square(pressure_deltas)))), pressure_value * 1.0e-4, 1.0e-6)
    peak_values = [
        float(value)
        for stencil in stencils
        for record in stencil.records
        for value in (record.output.module_peak_temperature.values() if record.output else ())
    ]
    solid_scale = max(float(np.sqrt(np.mean(np.square(peak_values)))), 1.0e-6)
    scales = ThermalLossScales(
        value=value_scales,
        finite=finite_scales,
        mixed=mixed_scales,
        pressure_value=pressure_value,
        pressure_response=pressure_response,
        pressure_limit=float(np.median(baseline_pressure) * 1.05),
        pressure_boundary=max(pressure_value * 0.05, 1.0e-6),
        solid_temperature=solid_scale,
        smooth_peak_beta=float(smooth_peak_beta),
        near_limit_band=max(pressure_value * 0.05, 1.0e-6),
    )
    return scales.with_train_reference_limits(stencils)


def _load_frozen_loss_scales(
    path: Path,
    training_stencils: Sequence[ResponseStencil],
    *,
    smooth_peak_beta: float,
) -> tuple[ThermalLossScales, dict[str, Any]]:
    """Load and validate a train-only calibration frozen by an earlier recipe."""

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    frozen = payload.get("frozen_scales", payload)
    if not isinstance(frozen, Mapping):
        raise TypeError("Frozen scale JSON must contain an object named frozen_scales.")
    required = {
        "value",
        "finite",
        "mixed",
        "pressure_value",
        "pressure_response",
        "pressure_boundary",
        "solid_temperature",
        "smooth_peak_beta",
        "near_limit_band",
        "pressure_limit_by_family",
    }
    missing = required - set(frozen)
    if missing:
        raise ValueError(f"Frozen loss scales are missing fields: {sorted(missing)}")
    if not training_stencils or any(stencil.split is not EvidenceSplit.TRAIN for stencil in training_stencils):
        raise ValueError("Frozen scale reconstruction requires only train stencils.")
    train_baseline_pressures: dict[str, float] = {}
    for stencil in training_stencils:
        output = stencil.baseline.output
        if output is None:
            raise ValueError("Frozen pressure limits require train baseline outputs.")
        pressure = output.quantities["pressure_drop"]
        if not pressure.resolved:
            raise ValueError("Frozen pressure limits require resolved train baseline pressures.")
        train_baseline_pressures[stencil.physical_family_id] = float(pressure.value)
    derived_scalar_limit = float(np.median(list(train_baseline_pressures.values())) * 1.05)
    if "pressure_limit" in frozen and not np.isclose(
        float(frozen["pressure_limit"]), derived_scalar_limit, rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError("Frozen scalar pressure limit does not match the train baseline median.")
    scales = ThermalLossScales(
        value=frozen["value"],
        finite=frozen["finite"],
        mixed=frozen["mixed"],
        pressure_value=float(frozen["pressure_value"]),
        pressure_response=float(frozen["pressure_response"]),
        pressure_limit=derived_scalar_limit,
        pressure_boundary=float(frozen["pressure_boundary"]),
        solid_temperature=float(frozen["solid_temperature"]),
        smooth_peak_beta=float(frozen["smooth_peak_beta"]),
        near_limit_band=float(frozen["near_limit_band"]),
        near_limit_multiplier=float(frozen.get("near_limit_multiplier", 2.0)),
        pressure_limit_by_family=frozen["pressure_limit_by_family"],
    )
    family_ids = {stencil.physical_family_id for stencil in training_stencils}
    if set(scales.pressure_limit_by_family or {}) != family_ids:
        raise ValueError("Frozen per-family pressure limits must exactly match the training panel.")
    expected_family_limits = {
        family_id: pressure * 1.05 for family_id, pressure in train_baseline_pressures.items()
    }
    for family_id, expected_limit in expected_family_limits.items():
        if not np.isclose(
            scales.pressure_limit_by_family[family_id], expected_limit, rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError(
                f"Frozen pressure limit for {family_id!r} does not match 1.05x its train baseline."
            )
    if not np.isclose(scales.smooth_peak_beta, smooth_peak_beta, rtol=0.0, atol=1.0e-12):
        raise ValueError("Frozen smooth_peak_beta differs from the requested training functional.")
    return scales, {
        "path": str(source),
        "sha256": _checkpoint_digest(source),
        "scope": "frozen_train_only_recipe_calibration",
        "family_ids": sorted(family_ids),
    }


def _load_frozen_response_weights(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Load a previously calibrated paired response multiplier set."""

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_weights = payload.get("calibrated_response_weights")
    if not isinstance(raw_weights, Mapping):
        raise TypeError("Frozen weight JSON must contain calibrated_response_weights.")
    weights = {str(name): float(value) for name, value in raw_weights.items()}
    required = {"value", "finite", "decision", "constraint"}
    if not required.issubset(weights):
        raise ValueError(f"Frozen response weights are missing: {sorted(required - set(weights))}")
    if any(not np.isfinite(value) or value <= 0.0 for value in weights.values()):
        raise ValueError("Frozen response weights must be positive and finite.")
    if not np.isclose(weights["value"], 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("The frozen response calibration must keep the value multiplier at one.")
    return weights, {
        "path": str(source),
        "sha256": _checkpoint_digest(source),
        "scope": "frozen_train_only_recipe_calibration",
    }


def _validate_review_gate(
    review_cap: int,
    config: StagedTrainingConfig,
    training_stencil_count: int,
) -> int:
    """Return the actual update cap and reject review gates beyond it."""

    if training_stencil_count <= 0:
        raise ValueError("At least one training stencil is required to compute the update cap.")
    if review_cap not in config.review_updates:
        raise ValueError(f"Review gate {review_cap} is not configured: {config.review_updates}.")
    effective_cap = min(
        config.max_optimizer_updates,
        config.total_optimizer_update_ceiling,
        config.max_epochs * training_stencil_count,
    )
    if review_cap > effective_cap:
        raise ValueError(
            f"Requested review gate {review_cap} is unreachable: {training_stencil_count} train "
            f"stencils x max_epochs={config.max_epochs} limit each arm to {effective_cap} updates."
        )
    return effective_cap


def _mixed_specs(stencil: ResponseStencil) -> tuple[MixedResponseSpec, ...]:
    variants = set(stencil.variants)
    corners = (("mm", "i_minus", "j_minus"), ("mp", "i_minus", "j_plus"),
               ("pm", "i_plus", "j_minus"), ("pp", "i_plus", "j_plus"))
    roles = tuple(stencil.baseline.output.roles)  # type: ignore[union-attr]
    return tuple(
        MixedResponseSpec(role, joint, first, second, f"{role}_{joint}")
        for joint, first, second in corners
        if {joint, first, second}.issubset(variants)
        for role in roles
    )


def _config_for_one_update(max_wall_seconds: float) -> StagedTrainingConfig:
    return StagedTrainingConfig(
        arm="B_response",
        max_optimizer_updates=1,
        max_epochs=1,
        total_optimizer_update_ceiling=20000,
        checkpoint_every_updates=1,
        review_updates=(),
        random_seed=2317,
        stages=(TrainingStage("preflight_value", 0, 1, ("value",)),),
        max_wall_seconds=max_wall_seconds,
    )


def _roundtrip_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: StagedTrainingConfig,
    payload: Mapping[str, Any],
    label: str,
) -> Path:
    path = output_dir / f"response_control_{label}.pt"
    _atomic_torch_save(path, payload)
    restored_payload = torch.load(path, map_location="cpu", weights_only=False)
    restore_checkpoint_payload(model, optimizer, restored_payload, config=config)
    return path


def _all_close_predictions(left: Any, right: Any, *, atol: float = 1.0e-6) -> bool:
    if set(left.role_values) != set(right.role_values):
        return False
    values_match = all(
        torch.allclose(left.role_values[name], right.role_values[name], atol=atol, rtol=1.0e-6)
        for name in left.role_values
    )
    if not values_match:
        return False
    if left.receiver_world_xy is None or right.receiver_world_xy is None:
        return left.receiver_world_xy is right.receiver_world_xy
    return all(
        torch.allclose(left.receiver_world_xy[name], right.receiver_world_xy[name], atol=atol, rtol=1.0e-6)
        for name in left.receiver_world_xy
    )


def _prediction_difference_summary(
    left: Any,
    right: Any,
    role_queries: Mapping[str, RoleQuery],
) -> dict[str, Any]:
    def summarize(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
        difference = (a - b).detach().abs()
        scale = torch.maximum(a.detach().abs(), b.detach().abs()).clamp_min(1.0e-12)
        return {
            "max_absolute": float(difference.max().cpu()),
            "max_relative": float((difference / scale).max().cpu()),
            "allclose_at_2e-5_atol_1e-6_rtol": bool(
                torch.allclose(a, b, atol=2.0e-5, rtol=1.0e-6)
            ),
            "allclose_at_3e-6_atol_3e-5_rtol": bool(
                torch.allclose(a, b, atol=3.0e-6, rtol=3.0e-5)
            ),
        }

    result: dict[str, Any] = {
        "role_values": {
            name: summarize(left.role_values[name], right.role_values[name])
            for name in left.role_values
        },
        "role_channels": {},
    }
    for name, query in role_queries.items():
        a = left.role_values[name]
        b = right.role_values[name]
        channel_rows = []
        for index, (channel_name, unit) in enumerate(zip(query.channel_names, query.channel_units)):
            channel_a = a[:, index]
            channel_b = b[:, index]
            item = summarize(channel_a, channel_b)
            item.update(
                {
                    "channel": channel_name,
                    "unit": unit,
                    "reference_max_absolute": float(channel_a.detach().abs().max().cpu()),
                    "candidate_max_absolute": float(channel_b.detach().abs().max().cpu()),
                }
            )
            channel_scale = max(
                item["reference_max_absolute"], item["candidate_max_absolute"]
            )
            tolerance = 3.0e-6 + 3.0e-6 * channel_scale
            item.update(
                {
                    "permutation_absolute_tolerance": 3.0e-6,
                    "permutation_relative_channel_scale_tolerance": 3.0e-6,
                    "permutation_allowed_max_absolute": tolerance,
                    "permutation_scale_gate_passed": item["max_absolute"] <= tolerance,
                }
            )
            channel_rows.append(item)
        result["role_channels"][name] = channel_rows
    result["permutation_world_coordinates_passed"] = True
    if left.receiver_world_xy is not None and right.receiver_world_xy is not None:
        result["receiver_world_xy"] = {
            name: summarize(left.receiver_world_xy[name], right.receiver_world_xy[name])
            for name in left.receiver_world_xy
        }
        result["permutation_world_coordinates_passed"] = all(
            torch.allclose(
                left.receiver_world_xy[name],
                right.receiver_world_xy[name],
                atol=1.0e-6,
                rtol=1.0e-6,
            )
            for name in left.receiver_world_xy
        )
    result["permutation_equivalence_criterion"] = (
        "per channel max_abs_delta <= 3e-6 + 3e-6*max_abs_channel; "
        "receiver_world_xy atol=1e-6, rtol=1e-6"
    )
    result["permutation_gate_passed"] = bool(
        result["permutation_world_coordinates_passed"]
        and all(
            item["permutation_scale_gate_passed"]
            for rows in result["role_channels"].values()
            for item in rows
        )
    )
    return result


def _poison_targets(stencil: ResponseStencil) -> ResponseStencil:
    record = stencil.baseline
    if record.output is None:
        raise ValueError("Target-poison regression requires a solved baseline.")
    poisoned_roles = {
        name: replace(
            role,
            values=np.full_like(role.values, 9.87654321e5),
            valid_mask=np.zeros_like(role.valid_mask, dtype=bool),
            noise_floor=None,
        )
        for name, role in record.output.roles.items()
    }
    quantities = dict(record.output.quantities)
    pressure = quantities["pressure_drop"]
    quantities["pressure_drop"] = MeasuredQuantity(
        -9.87654321e8, pressure.units, resolved=True, metadata={"poisoned": True}
    )
    poisoned_output = replace(
        record.output,
        roles=poisoned_roles,
        quantities=quantities,
        module_peak_temperature={
            module_id: -9.87654321e7 for module_id in record.output.active_module_ids
        },
    )
    poisoned_record = replace(record, output=poisoned_output)
    return ResponseStencil(poisoned_record, dict(stencil.variants))


def _permuted_role_queries(
    queries: Mapping[str, RoleQuery], permutation: torch.Tensor
) -> dict[str, RoleQuery]:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    result = {}
    for name, query in queries.items():
        slots = query.receiver_slots
        if slots is not None:
            slot_tensor = torch.as_tensor(slots, dtype=torch.long, device=permutation.device)
            slots = tuple(int(value) for value in inverse[slot_tensor].detach().cpu().tolist())
        result[name] = RoleQuery(
            role=query.role,
            query_features=query.query_features,
            channel_names=query.channel_names,
            channel_units=query.channel_units,
            receiver_slots=slots,
            coordinate_kind=query.coordinate_kind,
        )
    return result


def _pressure_counts(stencil: ResponseStencil, device: torch.device) -> dict[str, int]:
    queries = role_queries_from_stencil(stencil, device=device)
    design = DesignInput.from_state(stencil.baseline.design, device=device)
    inlet, outlet, _ = pressure_section_masks(
        queries["fluid_fields"], design, context_inputs(stencil.baseline.context)
    )
    return {
        "inlet_queries_retained": int(inlet.sum().detach().cpu()),
        "outlet_queries_retained": int(outlet.sum().detach().cpu()),
    }


def _sampling_summary_mapping(summary: SamplingSummary) -> dict[str, Any]:
    """Convert immutable sampling metadata to plain JSON-ready containers."""

    return {
        "physical_family_id": summary.physical_family_id,
        "split": summary.split,
        "original_counts": dict(summary.original_counts),
        "sampled_counts": dict(summary.sampled_counts),
        "protected_counts": dict(summary.protected_counts),
        "solid_peak_query_coverage": dict(summary.solid_peak_query_coverage),
        "inverse_probability_weighting": summary.inverse_probability_weighting,
    }


def _select_geometry_x_probe(stencil: ResponseStencil) -> tuple[int, float]:
    """Choose an active module and a centered x step inside the domain."""

    modules = stencil.baseline.design.modules
    domain_x = float(stencil.baseline.context.values["domain_length_x"])
    module_radius = float(stencil.baseline.context.values["module_radius"])
    candidates = [
        (index, float(module.position_xy[0]))
        for index, module in enumerate(modules)
        if module.active
    ]
    if not candidates:
        raise ValueError("Geometry AD/FD requires at least one active module.")
    clearances = [
        min(position_x - module_radius, domain_x - module_radius - position_x)
        for _, position_x in candidates
    ]
    best_index = int(np.argmax(clearances))
    clearance = float(clearances[best_index])
    if clearance <= 0.0:
        raise ValueError("No active module center has positive x clearance for the geometry AD/FD check.")
    return candidates[best_index][0], min(1.0e-3, clearance * 0.25)


def run_one_update_preflight(
    *,
    checkpoint_path: Path,
    stencil_path: Path,
    dataset_path: Path | None,
    output_dir: Path,
    device: torch.device,
    sampling: ReceiverSamplingConfig,
    max_wall_seconds: float,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    smooth_peak_beta: float = 1.0,
    query_batch_size: int = 2048,
) -> dict[str, Any]:
    """Run exactly one successful optimizer update from one train family."""

    started = time.monotonic()
    source_model, checkpoint = load_model(checkpoint_path, device)
    source_model.eval()
    stencil, _ = load_response_atlas_stencil(stencil_path)
    if stencil.split is not EvidenceSplit.TRAIN:
        raise ValueError("The one-update preflight requires one EvidenceSplit.TRAIN stencil.")
    sampled = sample_training_panel((stencil,), config=sampling)[0]
    sampled_stencil = sampled.stencil
    scales = derive_training_scales((sampled_stencil,), smooth_peak_beta=smooth_peak_beta)
    dataset_root = _resolve_dataset_path(checkpoint, str(dataset_path) if dataset_path else None)
    train_config = checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {})
    raw_dataset = GlobalChannelThermalDataset(
        dataset_root,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    template = _make_input_template(raw_dataset)
    target_model, refit_config = _make_refit_model(source_model, checkpoint, device=device)
    operator = DifferentiableThermalOperator(
        target_model,
        template,
        dataset_config=dataset_config,
        normalization_stats=checkpoint.get("global_normalization_stats", {}),
        query_batch_size=query_batch_size,
    )
    transfer = _materialize_and_warm_start(
        source_model, target_model, operator, sampled_stencil, device=device
    )
    target_model.eval()

    regression_started = time.perf_counter()
    derivative_started = time.perf_counter()
    derivative = check_pressure_peak_ad_fd(
        operator,
        sampled_stencil,
        variable="heating",
        module_slot=0,
        step=max(abs(float(sampled_stencil.baseline.design.modules[0].heating)) * 1.0e-2, 1.0e-5),
        relative_tolerance=0.35,
        absolute_tolerance=1.0e-4,
        device=device,
    )
    if not derivative["passed"]:
        raise RuntimeError(f"Whole-wrapper pressure/peak AD-FD check failed: {derivative}")
    derivative_seconds = time.perf_counter() - derivative_started

    geometry_slot, geometry_step = _select_geometry_x_probe(sampled_stencil)
    geometry_started = time.perf_counter()
    geometry_derivative = check_pressure_peak_ad_fd(
        operator,
        sampled_stencil,
        variable="position_x",
        module_slot=geometry_slot,
        step=geometry_step,
        relative_tolerance=0.35,
        absolute_tolerance=1.0e-4,
        device=device,
    )
    if not geometry_derivative["passed"]:
        raise RuntimeError(
            f"Whole-wrapper position pressure/peak AD-FD check failed: {geometry_derivative}"
        )
    geometry_derivative_seconds = time.perf_counter() - geometry_started

    # Target leakage regression: poisoned atlas labels and poisoned irrelevant
    # packed outputs do not enter the target-free native callback.
    original_queries = role_queries_from_stencil(sampled_stencil, device=device)
    design = DesignInput.from_state(sampled_stencil.baseline.design, device=device)
    context = context_inputs(sampled_stencil.baseline.context)
    with torch.no_grad():
        first = operator(design, context, original_queries)
        poison = {
            "structure": {
                "module_centers": np.full_like(template["structure"]["module_centers"], 777.0),
                "material_params": np.full_like(template["structure"]["material_params"], -123.0),
            },
            "field_targets": np.full((7, 3), -9.0, dtype=np.float32),
            "interface_condition": np.full((5, 8), -8.0e4, dtype=np.float32),
            "teacher_port_tokens": np.full((5, 5), 3.0e6, dtype=np.float32),
            "interface_target": np.full((5, 2), 1.0e4, dtype=np.float32),
            "module_internal_temperature_points": np.full((9, 1), 9.0e5, dtype=np.float32),
            "module_internal_mask": np.zeros((9,), dtype=np.uint8),
            "interface_condition_valid_mask": np.zeros((5,), dtype=np.uint8),
        }
        poisoned_operator = DifferentiableThermalOperator(
            target_model,
            poison,
            dataset_config=dataset_config,
            normalization_stats=checkpoint.get("global_normalization_stats", {}),
            query_batch_size=query_batch_size,
        )
        poisoned = poisoned_operator(design, context, original_queries)
    if not _all_close_predictions(first, poisoned):
        raise AssertionError("Native operator changed when target/irrelevant sample arrays were poisoned.")
    poisoned_stencil = _poison_targets(sampled_stencil)
    poisoned_queries = role_queries_from_stencil(poisoned_stencil, device=device)
    with torch.no_grad():
        poisoned_labels = operator(
            design, context_inputs(poisoned_stencil.baseline.context), poisoned_queries
        )
    if not _all_close_predictions(first, poisoned_labels):
        raise AssertionError(
            "Native prediction consumed atlas role values, masks, scalar targets, or peak labels."
        )

    # Re changes both an explicit input and its family-specific material
    # vector; a static packed template must not overwrite either value.
    changed_context = dict(context)
    changed_context["re"] = float(changed_context["re"]) * 1.07
    changed_context["nu"] = float(changed_context["nu"]) / 1.07
    original_inputs = operator._context_structure(context)
    changed_inputs = operator._context_structure(changed_context)
    if torch.equal(original_inputs["re"], changed_inputs["re"]) or torch.equal(
        original_inputs["material_params"], changed_inputs["material_params"]
    ):
        raise AssertionError("Per-stencil Reynolds/material context did not reach the native model inputs.")
    with torch.no_grad():
        changed_re_prediction = operator(design, changed_context, original_queries)
    if _all_close_predictions(first, changed_re_prediction, atol=1.0e-8):
        raise AssertionError("Changing per-stencil Re/nu did not change the native absolute prediction.")

    module_count = int(design.module_positions.shape[0])
    permutation = torch.arange(module_count - 1, -1, -1, device=device)
    permuted_design = DesignInput(
        design.module_positions.index_select(0, permutation),
        design.module_heating.index_select(0, permutation),
        design.module_present.index_select(0, permutation),
    )
    permuted_queries = _permuted_role_queries(original_queries, permutation)
    with torch.no_grad():
        permuted_prediction = operator(permuted_design, context, permuted_queries)
    permutation_diagnostic = _prediction_difference_summary(
        first, permuted_prediction, original_queries
    )
    if not permutation_diagnostic["permutation_gate_passed"]:
        raise AssertionError(
            "Native outputs changed under module-row permutation with receiver slots remapped: "
            + json.dumps(permutation_diagnostic, sort_keys=True)
        )

    spectator_changed = False
    if module_count >= 3:
        spectator_positions = design.module_positions.detach().clone()
        spectator_positions[-1, 0] += 0.03
        spectator_design = DesignInput(
            spectator_positions, design.module_heating, design.module_present
        )
        with torch.no_grad():
            spectator_prediction = operator(spectator_design, context, original_queries)
        spectator_changed = any(
            not torch.allclose(
                first.role_values[name],
                spectator_prediction.role_values[name],
                atol=1.0e-8,
                rtol=1.0e-7,
            )
            for name in first.role_values
        )
        if not spectator_changed:
            raise AssertionError("Moving an unchanged spectator did not change the recomputed absolute prediction.")
    regression_seconds = time.perf_counter() - regression_started
    parameter_inventory = _parameter_inventory(target_model)

    target_model.train()
    model_config = checkpoint.get("train_config", {}).get("training", {})
    optimizer = torch.optim.AdamW(
        (parameter for parameter in target_model.parameters() if parameter.requires_grad),
        lr=float(learning_rate if learning_rate is not None else model_config.get("learning_rate", 3.0e-4)),
        weight_decay=float(weight_decay if weight_decay is not None else model_config.get("weight_decay", 1.0e-5)),
    )
    optimizer_inventory = _verify_optimizer_inventory(target_model, optimizer)
    config = _config_for_one_update(max_wall_seconds)
    checkpoint_paths: list[str] = []

    def save_and_roundtrip(payload: Mapping[str, Any], label: str) -> None:
        checkpoint_file = _roundtrip_checkpoint(output_dir, target_model, optimizer, config, payload, label)
        checkpoint_paths.append(str(checkpoint_file))

    preflight_peak_allocated = 0
    preflight_peak_reserved = 0
    free_bytes_before_fit = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        preflight_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        preflight_peak_reserved = int(torch.cuda.max_memory_reserved(device))
        free_bytes_before_fit, _ = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
    fit_started = time.perf_counter()
    fit = run_staged_fit(
        operator,
        target_model,
        optimizer,
        (sampled_stencil,),
        scales=scales,
        loss_weights={"value": 1.0},
        config=config,
        device=device,
        on_checkpoint=save_and_roundtrip,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - fit_started
    if fit.actual_optimizer_updates != 1 or fit.attempted_optimizer_steps != 1:
        raise RuntimeError(
            f"Expected exactly one successful update, got {fit.actual_optimizer_updates} completed "
            f"and {fit.attempted_optimizer_steps} attempted."
        )
    if len(checkpoint_paths) < 2:
        raise RuntimeError("Expected both zero-update and post-update checkpoint round trips.")
    return {
        "status": "passed",
        "mode": "one_update_preflight",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _checkpoint_digest(checkpoint_path),
        "selected_epoch": checkpoint.get("selection_state", {}).get("epoch", checkpoint.get("epoch")),
        "atlas_npz": str(stencil_path.resolve()),
        "atlas_family_id": stencil.physical_family_id,
        "atlas_split": stencil.split.value,
        "atlas_context": dict(sampled_stencil.baseline.context.values),
        "atlas_record_count": len(sampled_stencil.records),
        "refit_config": refit_config,
        "warm_start": transfer,
        "parameter_inventory": parameter_inventory,
        "optimizer_inventory": optimizer_inventory,
        "sampling": _sampling_summary_mapping(sampled.summary),
        "role_query_counts": {
            name: query.query_features.shape[0]
            for name, query in original_queries.items()
        },
        "pressure_sections": _pressure_counts(sampled_stencil, device),
        "whole_wrapper_ad_fd": derivative,
        "whole_wrapper_ad_fd_wall_seconds": derivative_seconds,
        "whole_wrapper_geometry_ad_fd": geometry_derivative,
        "whole_wrapper_geometry_ad_fd_wall_seconds": geometry_derivative_seconds,
        "wrapper_regression_wall_seconds": regression_seconds,
        "target_poison_invariance": True,
        "atlas_target_poison_invariance": True,
        "per_stencil_reynolds_material_route": True,
        "reynolds_change_prediction_changed": True,
        "module_row_permutation_invariance": True,
        "module_row_permutation_diagnostic": permutation_diagnostic,
        "spectator_intervention_changed_prediction": spectator_changed,
        "optimizer_updates": fit.actual_optimizer_updates,
        "attempted_optimizer_steps": fit.attempted_optimizer_steps,
        "losses": [asdict(step) for step in fit.history],
        "checkpoint_roundtrip_paths": checkpoint_paths,
        "fit_wall_seconds": fit_seconds,
        "total_wall_seconds": float(time.monotonic() - started),
        "cuda": _cuda_evidence(
            device,
            free_bytes_before_fit=free_bytes_before_fit,
            preflight_peak_allocated_bytes=preflight_peak_allocated,
            preflight_peak_reserved_bytes=preflight_peak_reserved,
        ),
    }


def run_paired_fit(
    *,
    checkpoint_path: Path,
    stencil_paths: Sequence[Path],
    development_paths: Sequence[Path],
    dataset_path: Path | None,
    output_dir: Path,
    device: torch.device,
    sampling: ReceiverSamplingConfig,
    max_wall_seconds: float,
    review_cap: int,
    review_continuations: Sequence[int],
    smooth_peak_beta: float,
    query_batch_size: int,
    resume_value_path: Path | None = None,
    resume_response_path: Path | None = None,
    frozen_loss_scales_path: Path | None = None,
    frozen_response_weights_path: Path | None = None,
    resume_fit_manifest_path: Path | None = None,
    resume_replay_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run matched value/response arms through one explicit review gate."""

    total_started = time.monotonic()
    if not stencil_paths:
        raise ValueError("Paired fitting needs at least one train stencil.")
    if (resume_value_path is None) != (resume_response_path is None):
        raise ValueError("Resume requires both arm checkpoints from the same paired gate.")
    resuming = resume_value_path is not None
    if resuming:
        if frozen_loss_scales_path is None or frozen_response_weights_path is not None:
            raise ValueError(
                "A paired resume requires frozen loss scales but takes response weights from its checkpoints."
            )
        if resume_fit_manifest_path is None or resume_replay_manifest_path is None:
            raise ValueError("A paired resume requires both source fit and read-only replay manifests.")
    else:
        if (frozen_loss_scales_path is None) != (frozen_response_weights_path is None):
            raise ValueError(
                "A fresh fit requires both frozen scale and response-weight JSON files, or neither."
            )
        if resume_fit_manifest_path is not None or resume_replay_manifest_path is not None:
            raise ValueError("Resume manifests can be supplied only with a paired checkpoint resume.")
    source_model, checkpoint = load_model(checkpoint_path, device)
    source_model.eval()
    loaded_stencils = [load_response_atlas_stencil(path) for path in stencil_paths]
    raw_stencils = [stencil for stencil, _ in loaded_stencils]
    if any(stencil.split is not EvidenceSplit.TRAIN for stencil in raw_stencils):
        raise ValueError("Paired fitting accepts EvidenceSplit.TRAIN stencils only.")
    config = replace(
        load_staged_training_config(arm="B_response"),
        max_wall_seconds=max_wall_seconds,
    )
    effective_update_cap = _validate_review_gate(
        review_cap, config, len(raw_stencils)
    )
    sampled_panel = sample_training_panel(raw_stencils, config=sampling)
    training_stencils = tuple(item.stencil for item in sampled_panel)
    scales_provenance: dict[str, Any] | None = None
    response_weight_provenance: dict[str, Any] | None = None
    fixed_response_weights: dict[str, float] | None = None
    if frozen_loss_scales_path is None:
        scales = derive_training_scales(training_stencils, smooth_peak_beta=smooth_peak_beta)
    else:
        scales, scales_provenance = _load_frozen_loss_scales(
            frozen_loss_scales_path,
            raw_stencils,
            smooth_peak_beta=smooth_peak_beta,
        )
        if not resuming:
            fixed_response_weights, response_weight_provenance = _load_frozen_response_weights(
                frozen_response_weights_path  # type: ignore[arg-type]
            )
    loss_scales_snapshot = {
        "value": dict(scales.value),
        "finite": dict(scales.finite),
        "mixed": dict(scales.mixed),
        "pressure_value": scales.pressure_value,
        "pressure_response": scales.pressure_response,
        "pressure_limit": scales.pressure_limit,
        "pressure_boundary": scales.pressure_boundary,
        "solid_temperature": scales.solid_temperature,
        "smooth_peak_beta": scales.smooth_peak_beta,
        "near_limit_band": scales.near_limit_band,
        "near_limit_multiplier": scales.near_limit_multiplier,
        "pressure_limit_by_family": dict(scales.pressure_limit_by_family or {}),
    }
    resume_payloads = None
    if resuming:
        resume_payloads = {
            "B_value": torch.load(resume_value_path, map_location="cpu", weights_only=False),
            "B_response": torch.load(resume_response_path, map_location="cpu", weights_only=False),
        }
    dataset_root = _resolve_dataset_path(checkpoint, str(dataset_path) if dataset_path else None)
    train_config = checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {})
    raw_dataset = GlobalChannelThermalDataset(
        dataset_root,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    template = _make_input_template(raw_dataset)
    target_model, refit_config = _make_refit_model(source_model, checkpoint, device=device)
    resume_provenance = None
    if resuming:
        resume_provenance = validate_paired_resume_provenance(
            fit_manifest_path=resume_fit_manifest_path,  # type: ignore[arg-type]
            replay_manifest_path=resume_replay_manifest_path,  # type: ignore[arg-type]
            source_checkpoint_path=checkpoint_path,
            train_atlas_paths=stencil_paths,
            development_atlas_paths=development_paths,
            train_family_ids=[stencil.physical_family_id for stencil in raw_stencils],
            train_sampling=[_sampling_summary_mapping(item.summary) for item in sampled_panel],
            frozen_scales_path=frozen_loss_scales_path,  # type: ignore[arg-type]
            loss_scales=loss_scales_snapshot,
            refit_config=refit_config,
            resume_checkpoint_paths={
                "B_value": resume_value_path,  # type: ignore[dict-item]
                "B_response": resume_response_path,  # type: ignore[dict-item]
            },
            resume_payloads=resume_payloads,  # type: ignore[arg-type]
            training_config=config,
            required_update=100,
        )
        response_weight_provenance = {
            "source": "verified_B_response_u100_checkpoint",
            "fit_manifest_sha256": resume_provenance["fit_manifest"]["sha256"],
        }
    operator = DifferentiableThermalOperator(
        target_model,
        template,
        dataset_config=dataset_config,
        normalization_stats=checkpoint.get("global_normalization_stats", {}),
        query_batch_size=query_batch_size,
    )
    transfer = _materialize_and_warm_start(
        source_model, target_model, operator, training_stencils[0], device=device
    )
    parameter_inventory = _parameter_inventory(target_model)
    target_model.eval()
    derivative_started = time.perf_counter()
    derivative = check_pressure_peak_ad_fd(
        operator,
        training_stencils[0],
        variable="heating",
        module_slot=0,
        step=max(abs(float(training_stencils[0].baseline.design.modules[0].heating)) * 1.0e-2, 1.0e-5),
        relative_tolerance=0.35,
        absolute_tolerance=1.0e-4,
        device=device,
    )
    if not derivative["passed"]:
        raise RuntimeError(f"Whole-wrapper pressure/peak AD-FD check failed: {derivative}")
    derivative_wall_seconds = time.perf_counter() - derivative_started

    optimizer_config = checkpoint.get("train_config", {}).get("training", {})
    lr = float(optimizer_config.get("learning_rate", 3.0e-4))
    decay = float(optimizer_config.get("weight_decay", 1.0e-5))

    def optimizer_factory(model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=lr,
            weight_decay=decay,
        )

    latest_payloads: dict[str, Mapping[str, Any]] = {}
    checkpoint_paths: dict[str, list[str]] = {"B_value": [], "B_response": []}

    def save_checkpoint(arm: str, payload: Mapping[str, Any], label: str) -> None:
        arm_config = StagedTrainingConfig.from_mapping(payload["training_config"])
        safe_label = "".join(character if character.isalnum() else "_" for character in label)
        update = int(payload["actual_optimizer_updates"])
        path = output_dir / f"response_control_{arm}_{safe_label}_u{update:05d}.pt"
        persisted_payload = dict(payload)
        persisted_payload["response_control_calibration_provenance"] = {
            "loss_scales": loss_scales_snapshot,
            "loss_scales_source": scales_provenance,
            "response_weight_source": response_weight_provenance,
            "resume_provenance": resume_provenance,
        }
        _atomic_torch_save(path, persisted_payload)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        restore_checkpoint_payload(target_model, active_optimizers[arm], loaded, config=arm_config)
        latest_payloads[arm] = loaded
        checkpoint_paths[arm].append(str(path))

    active_optimizers: dict[str, torch.optim.Optimizer] = {}

    def optimizer_factory_tracked(model: torch.nn.Module) -> torch.optim.Optimizer:
        optimizer = optimizer_factory(model)
        _verify_optimizer_inventory(model, optimizer)
        arm = "B_value" if len(active_optimizers) == 0 else "B_response"
        active_optimizers[arm] = optimizer
        return optimizer

    all_mixed_specs: list[MixedResponseSpec] = []
    seen_specs: set[tuple[str, str, str, str]] = set()
    for stencil in training_stencils:
        for spec in _mixed_specs(stencil):
            key = (spec.role, spec.joint_variant, spec.first_variant, spec.second_variant)
            if key not in seen_specs:
                seen_specs.add(key)
                all_mixed_specs.append(spec)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    fit_started = time.perf_counter()
    paired = run_paired_staged_fits(
        target_model,
        lambda model: DifferentiableThermalOperator(
            model,
            template,
            dataset_config=dataset_config,
            normalization_stats=checkpoint.get("global_normalization_stats", {}),
            query_batch_size=query_batch_size,
        ),
        optimizer_factory_tracked,
        training_stencils,
        scales=scales,
        mixed_specs=tuple(all_mixed_specs),
        config=config,
        device=device,
        on_checkpoint=save_checkpoint,
        stop_at_update=review_cap,
        review_continuations=review_continuations,
        resume_payloads=resume_payloads,
        fixed_response_weights=fixed_response_weights,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        train_cuda_peak = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        torch.cuda.reset_peak_memory_stats(device)
    else:
        train_cuda_peak = None
    fit_seconds = time.perf_counter() - fit_started
    curve_path = write_paired_training_curves(output_dir / "paired_training_curves.csv", paired)
    reached_updates = {
        arm: result.final_update for arm, result in paired.arms.items()
    }
    if any(update != review_cap for update in reached_updates.values()):
        raise TimeoutError(
            f"Paired arms did not reach requested review gate {review_cap}: {reached_updates}. "
            "Any numbered checkpoints and partial loss curve were preserved."
        )

    development_results: dict[str, list[dict[str, Any]]] = {"B_value": [], "B_response": []}
    development_started = time.perf_counter()
    for arm in ("B_value", "B_response"):
        if not latest_payloads.get(arm):
            continue
        target_model.load_state_dict(latest_payloads[arm]["model"], strict=True)
        target_model.eval()
        for path in development_paths:
            dev_stencil, _ = load_response_atlas_stencil(path)
            if dev_stencil.split is EvidenceSplit.TRAIN:
                raise ValueError("Development evaluation paths must not carry the train split label.")
            dev_limit = float(dev_stencil.baseline.output.quantities["pressure_drop"].value) * 1.05  # type: ignore[union-attr]
            metrics = evaluate_stencil(
                operator,
                dev_stencil,
                pressure_limit={dev_stencil.physical_family_id: dev_limit},
                mixed_specs=_mixed_specs(dev_stencil),
                smooth_peak_beta=smooth_peak_beta,
                device=device,
            )
            development_results[arm].append(
                {
                    "family_id": dev_stencil.physical_family_id,
                    "split": dev_stencil.split.value,
                    "source": dev_stencil.source.value,
                    "pressure_limit_from_original_start": dev_limit,
                    "metrics": metrics,
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        development_cuda_peak = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    else:
        development_cuda_peak = None
    development_wall_seconds = time.perf_counter() - development_started
    return {
        "status": "passed",
        "mode": "paired_staged_fit",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _checkpoint_digest(checkpoint_path),
        "train_atlas_paths": [str(path.resolve()) for path in stencil_paths],
        "train_family_ids": [stencil.physical_family_id for stencil in training_stencils],
        "train_contexts": [dict(stencil.baseline.context.values) for stencil in training_stencils],
        "train_sampling": [_sampling_summary_mapping(item.summary) for item in sampled_panel],
        "development_paths": [str(path.resolve()) for path in development_paths],
        "refit_config": refit_config,
        "warm_start": transfer,
        "parameter_inventory": parameter_inventory,
        "whole_wrapper_ad_fd": derivative,
        "whole_wrapper_ad_fd_wall_seconds": derivative_wall_seconds,
        "gradient_calibration": {
            "scope": (
                "resume_checkpoint"
                if resume_payloads is not None
                else "frozen_recipe_calibration"
                if fixed_response_weights is not None
                else "first_fixed_train_stencil_only"
            ),
            "physical_family_id": training_stencils[0].physical_family_id,
            "split": training_stencils[0].split.value,
            "weights_source": response_weight_provenance,
            "weights": dict(paired.calibrated_response_weights),
        },
        "loss_scales_source": scales_provenance,
        "loss_scales": loss_scales_snapshot,
        "resume_provenance": resume_provenance,
        "review_decisions": dict(paired.review_decisions),
        "review_cap": review_cap,
        "effective_update_cap": effective_update_cap,
        "review_continuations": list(review_continuations),
        "arms": {
            arm: {
                "initial_update": result.initial_update,
                "actual_optimizer_updates": result.actual_optimizer_updates,
                "attempted_optimizer_steps": result.attempted_optimizer_steps,
                "final_update": result.final_update,
                "total_attempted_optimizer_steps": result.total_attempted_optimizer_steps,
                "stopped_at_review": result.stopped_at_review,
                "stopped_for_wall_time": result.stopped_for_wall_time,
                "wall_seconds": result.wall_seconds,
            }
            for arm, result in paired.arms.items()
        },
        "calibrated_response_weights": dict(paired.calibrated_response_weights),
        "curve_path": str(curve_path),
        "checkpoint_paths": checkpoint_paths,
        "development_evaluation": development_results,
        "development_evaluation_wall_seconds": development_wall_seconds,
        "checkpoint_selection_rule": (
            "No automatic selection. Review matched B_value/B_response at this gate; "
            "feasibility and critical responses first, then decision and role-separated receiver quality."
        ),
        "fit_wall_seconds": fit_seconds,
        "total_wall_seconds": float(time.monotonic() - total_started),
        "cuda": _cuda_evidence(device),
        "cuda_phase_peaks": {
            "paired_fit": train_cuda_peak,
            "development_replay": development_cuda_peak,
        },
    }


def _cuda_evidence(
    device: torch.device,
    *,
    free_bytes_before_fit: int | None = None,
    preflight_peak_allocated_bytes: int | None = None,
    preflight_peak_reserved_bytes: int | None = None,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"available": False}
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "available": True,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "unavailable")),
        "free_bytes_before_fit": (
            int(free_bytes_before_fit) if free_bytes_before_fit is not None else None
        ),
        "free_bytes_after_run": int(free_bytes),
        "total_bytes": int(total_bytes),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "preflight_peak_allocated_bytes": preflight_peak_allocated_bytes,
        "preflight_peak_reserved_bytes": preflight_peak_reserved_bytes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "paired"), default="preflight")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-stencil", required=True, type=Path, action="append")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--max-fluid-queries", type=int, default=3072)
    parser.add_argument("--solid-queries-per-module", type=int, default=128)
    parser.add_argument("--hot-solid-points-per-module", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=2317)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--smooth-peak-beta", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--review-cap", type=int, choices=(100, 300, 1000, 2000), default=100)
    parser.add_argument("--continue-after", type=int, choices=(100, 300, 1000), action="append", default=[])
    parser.add_argument("--development-stencil", type=Path, action="append", default=[])
    parser.add_argument("--resume-b-value", type=Path, default=None)
    parser.add_argument("--resume-b-response", type=Path, default=None)
    parser.add_argument("--resume-fit-manifest-json", type=Path, default=None)
    parser.add_argument("--resume-replay-manifest-json", type=Path, default=None)
    parser.add_argument("--frozen-loss-scales-json", type=Path, default=None)
    parser.add_argument("--frozen-response-weights-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    path_parts = output_dir.parts
    if not any(
        path_parts[index] == "diagnostics" and index + 1 < len(path_parts)
        and path_parts[index + 1] == "generated"
        for index in range(len(path_parts) - 1)
    ):
        raise ValueError("Run outputs must be placed under an ignored diagnostics/generated directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "response_control_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "mode": args.mode,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "train_stencils": [str(path.expanduser().resolve()) for path in args.train_stencil],
        "dataset": None if args.dataset is None else str(args.dataset.expanduser().resolve()),
        "device": args.device,
        "max_wall_seconds": args.max_wall_seconds,
        "frozen_loss_scales_json": (
            None
            if args.frozen_loss_scales_json is None
            else str(args.frozen_loss_scales_json.expanduser().resolve())
        ),
        "frozen_response_weights_json": (
            None
            if args.frozen_response_weights_json is None
            else str(args.frozen_response_weights_json.expanduser().resolve())
        ),
        "resume_fit_manifest_json": (
            None
            if args.resume_fit_manifest_json is None
            else str(args.resume_fit_manifest_json.expanduser().resolve())
        ),
        "resume_replay_manifest_json": (
            None
            if args.resume_replay_manifest_json is None
            else str(args.resume_replay_manifest_json.expanduser().resolve())
        ),
        "started_unix_seconds": time.time(),
    }
    _atomic_json(manifest_path, manifest)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def timeout_handler(signum: int, _frame: Any) -> None:
        raise TimeoutError(f"Received signal {signum} before the bounded run completed.")

    signal.signal(signal.SIGTERM, timeout_handler)
    try:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        for stencil_path in args.train_stencil + args.development_stencil:
            if not stencil_path.is_file():
                raise FileNotFoundError(stencil_path)
        if args.max_wall_seconds <= 0.0:
            raise ValueError("max_wall_seconds must be positive.")
        device = torch.device(args.device)
        if device.type == "cuda" and (not torch.cuda.is_available() or device.index not in {None, 0}):
            raise RuntimeError("Use CUDA_VISIBLE_DEVICES=2 with logical --device cuda:0 for the allocated GPU.")
        sampling = ReceiverSamplingConfig(
            max_fluid_queries=args.max_fluid_queries,
            solid_queries_per_module=args.solid_queries_per_module,
            hot_solid_points_per_module=args.hot_solid_points_per_module,
            random_seed=args.random_seed,
        )
        if args.mode == "preflight":
            if any(
                value is not None
                for value in (
                    args.frozen_loss_scales_json,
                    args.frozen_response_weights_json,
                    args.resume_b_value,
                    args.resume_b_response,
                    args.resume_fit_manifest_json,
                    args.resume_replay_manifest_json,
                )
            ):
                raise ValueError("Frozen calibration and resume overrides apply only to paired-fit mode.")
            if len(args.train_stencil) != 1:
                raise ValueError("The first launch is restricted to exactly one train stencil and one update.")
            result = run_one_update_preflight(
                checkpoint_path=args.checkpoint.expanduser().resolve(),
                stencil_path=args.train_stencil[0].expanduser().resolve(),
                dataset_path=args.dataset,
                output_dir=output_dir,
                device=device,
                sampling=sampling,
                max_wall_seconds=args.max_wall_seconds,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                smooth_peak_beta=args.smooth_peak_beta,
                query_batch_size=args.query_batch_size,
            )
        else:
            resuming = args.resume_b_value is not None or args.resume_b_response is not None
            if (args.resume_b_value is None) != (args.resume_b_response is None):
                raise ValueError("Resume requires both paired arm checkpoints.")
            if resuming:
                if args.frozen_loss_scales_json is None or args.frozen_response_weights_json is not None:
                    raise ValueError(
                        "A resume requires --frozen-loss-scales-json and takes response weights from the checkpoints."
                    )
                if args.resume_fit_manifest_json is None or args.resume_replay_manifest_json is None:
                    raise ValueError("A resume requires both source provenance manifest paths.")
            else:
                if (args.frozen_loss_scales_json is None) != (args.frozen_response_weights_json is None):
                    raise ValueError("Supply both frozen scale and response-weight JSON files, or neither.")
                if args.resume_fit_manifest_json is not None or args.resume_replay_manifest_json is not None:
                    raise ValueError("Resume manifests require both --resume-b-value and --resume-b-response.")
            if args.resume_b_value is not None and not args.resume_b_value.is_file():
                raise FileNotFoundError(args.resume_b_value)
            if args.resume_b_response is not None and not args.resume_b_response.is_file():
                raise FileNotFoundError(args.resume_b_response)
            if args.frozen_loss_scales_json is not None and not args.frozen_loss_scales_json.is_file():
                raise FileNotFoundError(args.frozen_loss_scales_json)
            if args.frozen_response_weights_json is not None and not args.frozen_response_weights_json.is_file():
                raise FileNotFoundError(args.frozen_response_weights_json)
            if args.resume_fit_manifest_json is not None and not args.resume_fit_manifest_json.is_file():
                raise FileNotFoundError(args.resume_fit_manifest_json)
            if args.resume_replay_manifest_json is not None and not args.resume_replay_manifest_json.is_file():
                raise FileNotFoundError(args.resume_replay_manifest_json)
            result = run_paired_fit(
                checkpoint_path=args.checkpoint.expanduser().resolve(),
                stencil_paths=[path.expanduser().resolve() for path in args.train_stencil],
                development_paths=[path.expanduser().resolve() for path in args.development_stencil],
                dataset_path=args.dataset,
                output_dir=output_dir,
                device=device,
                sampling=sampling,
                max_wall_seconds=args.max_wall_seconds,
                review_cap=args.review_cap,
                review_continuations=args.continue_after,
                smooth_peak_beta=args.smooth_peak_beta,
                query_batch_size=args.query_batch_size,
                resume_value_path=args.resume_b_value,
                resume_response_path=args.resume_b_response,
                frozen_loss_scales_path=args.frozen_loss_scales_json,
                frozen_response_weights_path=args.frozen_response_weights_json,
                resume_fit_manifest_path=args.resume_fit_manifest_json,
                resume_replay_manifest_path=args.resume_replay_manifest_json,
            )
        manifest.update(result)
        manifest["finished_unix_seconds"] = time.time()
        _atomic_json(manifest_path, manifest)
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "total_wall_seconds": time.time() - float(manifest["started_unix_seconds"]),
                "finished_unix_seconds": time.time(),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI
    raise SystemExit(main())
