"""Prelaunch audit for the Run-1409 dense-to-sparse continuation.

This diagnostic intentionally performs a small real-data training replay rather
than creating a managed run.  It consumes exactly two B=48, Q=1024 batches
from the resolved ThermalChannel train loader at epoch 1 and performs exactly
one optimizer step (on the second batch).  The first two forwards request the
maintained routing maps and prepared state so the P0 gate plan, phase reuse,
routing entropy, physical prototype gradients, and the finite update can be
checked together.

The audit is for the v2 ``dense_to_sparse_v2`` schedule.  At epoch 1 its
continuation is c=0: every effective gate must remain positive and the route
logits use the prescribed 0.1 scale.  The raw hard-concrete realization is
still recorded separately.  A physical-loss gradient through the gate network
is expected to be zero/unconnected at c=0; prototype gradients are required to
be finite, nonzero for every one of the 12 rows, and non-identical.

No checkpoint or managed run directory is written.  The command is intended
for one free GPU after the caller has selected the v2 profile and overlay::

    PYTHONPATH=src:Case_ThermalChannel/src \\
      python tools/diagnostics/run_run1409_v2_prelaunch_audit.py \\
      --profile src/config_core/forward/budgeted_group_control_honf_context.json \\
      --overlay src/config_core/forward/experiments/run1409_case_group_budget_calibrated.json \\
      --device cuda:0 \\
      --output /abs/path/Run_1409_.../evaluations/prelaunch_epoch1_audit.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = PROJECT_ROOT / "src/config_core/forward/budgeted_group_control_honf_context.json"
DEFAULT_OVERLAY = (
    PROJECT_ROOT / "src/config_core/forward/experiments/run1409_case_group_budget_calibrated.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "diagnostics/generated/run1409_dense_to_sparse_v2/prelaunch_epoch1_audit.json"
EXPECTED_ARCHITECTURE = "budgeted_group_control_honf"
EXPECTED_SCHEDULE = "dense_to_sparse_v2"
EXPECTED_GROUP_COUNT = 12
EXPECTED_CONTROL_DIM = 16
EXPECTED_BATCH_SIZE = 48
EXPECTED_QUERY_COUNT = 1024
EXPECTED_BATCHES = 2


def _jsonable(value: Any) -> Any:
    """Convert tensors and bounded arrays to JSON without retaining devices."""

    if hasattr(value, "detach") and hasattr(value, "numel"):
        value = value.detach().cpu()
        if int(value.numel()) == 1:
            return _jsonable(value.item())
        if int(value.numel()) <= 256:
            return value.tolist()
        flat = value.float().reshape(-1)
        finite = flat[flat.isfinite()]
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "min": None if not finite.numel() else float(finite.min()),
            "max": None if not finite.numel() else float(finite.max()),
            "mean": None if not finite.numel() else float(finite.mean()),
        }
    if isinstance(value, np.ndarray):
        if value.size <= 256:
            return value.tolist()
        finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else value
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.size),
            "min": None if finite.size == 0 else float(np.min(finite)),
            "max": None if finite.size == 0 else float(np.max(finite)),
            "mean": None if finite.size == 0 else float(np.mean(finite)),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return np.asarray(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _scalar(value: Any) -> float | None:
    array = _array(value)
    if array is None or array.size != 1:
        return None
    try:
        result = float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _attr(value: Any, names: Sequence[str]) -> Any:
    if value is None:
        return None
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _budget_plan(output: Mapping[str, Any]) -> Any:
    """Find the live CaseGroupBudget carried by PreparedState."""

    prepared_state = output.get("prepared_state")
    prepared = getattr(prepared_state, "prepared", None)
    candidates = (
        getattr(prepared_state, "phase_shared_state", None),
        getattr(prepared, "phase_shared_state", None),
        output.get("case_group_budget"),
        output.get("phase_shared_state"),
    )
    for candidate in candidates:
        if candidate is not None and (
            _attr(candidate, ("z", "effective_z")) is not None
            or _attr(candidate, ("raw_z", "z_raw")) is not None
        ):
            return candidate
    return None


def _find_output_or_aux(output: Mapping[str, Any], names: Sequence[str]) -> Any:
    """Look through the maintained output and interaction diagnostic channel."""

    aux = output.get("interaction_aux")
    for name in names:
        if name in output:
            return output[name]
        if isinstance(aux, Mapping) and name in aux:
            return aux[name]
    return None


def _budget_summary(output: Mapping[str, Any]) -> dict[str, Any]:
    plan = _budget_plan(output)
    raw_z = _attr(plan, ("raw_z", "z_raw", "sampled_z"))
    effective_z = _attr(plan, ("z", "effective_z", "z_eff"))
    raw_support = _attr(plan, ("raw_support", "support_raw"))
    effective_support = _attr(plan, ("support", "effective_support", "executed_support"))
    probability = _attr(plan, ("positive_probability", "probability", "gate_probability"))
    expected_count = _attr(plan, ("expected_group_count", "expected_count"))
    expected_optional = _attr(plan, ("expected_optional_count",))
    packed_ids = _attr(plan, ("packed_ids", "prototype_ids", "compact_ids"))
    packed_valid = _attr(plan, ("packed_valid", "compact_valid"))
    fallback = _attr(plan, ("fallback_used", "fallback", "used_fallback"))
    kappa = _attr(plan, ("kappa",))
    eta = _attr(plan, ("eta",))
    optional_logits = _attr(plan, ("optional_logits", "gate_logits"))
    continuation = _attr(plan, ("continuation", "continuation_c", "sparsification_c"))
    route_scale = _attr(plan, ("route_logit_scale", "routing_strength", "route_strength"))

    raw_z_np = _array(raw_z)
    effective_z_np = _array(effective_z)
    raw_support_np = _array(raw_support)
    effective_support_np = _array(effective_support)
    probability_np = _array(probability)
    packed_ids_np = _array(packed_ids)
    packed_valid_np = _array(packed_valid)
    fallback_np = _array(fallback)
    if raw_z_np is not None and raw_z_np.ndim == 1:
        raw_z_np = raw_z_np[None, :]
    if effective_z_np is not None and effective_z_np.ndim == 1:
        effective_z_np = effective_z_np[None, :]
    if raw_support_np is None and raw_z_np is not None:
        raw_support_np = raw_z_np > 0.0
    if effective_support_np is None and effective_z_np is not None:
        effective_support_np = effective_z_np > 0.0
    if raw_support_np is not None and raw_support_np.ndim == 1:
        raw_support_np = raw_support_np[None, :]
    if effective_support_np is not None and effective_support_np.ndim == 1:
        effective_support_np = effective_support_np[None, :]

    expected_np = _array(expected_count)
    if expected_np is None and probability_np is not None:
        expected_np = probability_np.sum(axis=-1)
    optional_np = _array(expected_optional)
    if expected_np is None and optional_np is not None:
        expected_np = optional_np + 1.0
    if expected_np is not None:
        expected_np = expected_np.reshape(-1)
    raw_counts = None if raw_support_np is None else raw_support_np.astype(bool).sum(axis=-1).astype(int)
    effective_counts = (
        None if effective_support_np is None else effective_support_np.astype(bool).sum(axis=-1).astype(int)
    )
    if packed_ids_np is not None:
        packed_ids_np = packed_ids_np.astype(int)
        if packed_ids_np.ndim == 1:
            packed_ids_np = packed_ids_np[None, :]
    if packed_valid_np is not None and packed_valid_np.ndim == 1:
        packed_valid_np = packed_valid_np[None, :]
    if fallback_np is not None:
        fallback_np = fallback_np.astype(bool).reshape(-1)

    return {
        "status": "complete" if plan is not None else "unavailable",
        "Kmax": EXPECTED_GROUP_COUNT,
        "K_raw": None if raw_counts is None else raw_counts.astype(int).tolist(),
        "K_executed": None if effective_counts is None else effective_counts.astype(int).tolist(),
        "K_expected": None if expected_np is None else expected_np.tolist(),
        "K_expected_optional": None if optional_np is None else optional_np.reshape(-1).tolist(),
        "Klive": None if effective_counts is None else effective_counts.astype(int).tolist(),
        "Kpack": None if packed_ids_np is None else int(packed_ids_np.shape[1]),
        "fallback_used": None if fallback_np is None else fallback_np.tolist(),
        "prototype_ids": None if packed_ids_np is None else packed_ids_np.tolist(),
        "packed_valid": None if packed_valid_np is None else packed_valid_np.tolist(),
        "effective_gate_values": _jsonable(effective_z),
        "raw_gate_values": _jsonable(raw_z),
        "positive_probability": _jsonable(probability),
        "eta": _jsonable(eta),
        "kappa": _jsonable(kappa),
        "optional_logits": _jsonable(optional_logits),
        "continuation_c": _scalar(continuation),
        "routing_strength": _scalar(route_scale),
        "all_effective_gates_positive": bool(
            effective_z_np is not None and np.all(effective_z_np > 0.0)
        ),
        "all_effective_gate_count": (
            None if effective_z_np is None else int(effective_z_np.shape[-1])
        ),
        "symmetric_plan": bool(_attr(plan, ("symmetric",)) or False) if plan is not None else None,
        "deterministic_plan": bool(_attr(plan, ("deterministic",)) or False) if plan is not None else None,
    }


def _entropy_row_summary(value: Any) -> dict[str, Any]:
    array = _array(value)
    if array is None:
        return {"status": "unavailable", "reason": "routing map missing"}
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim < 3:
        return {"status": "unavailable", "reason": f"unexpected map shape {list(array.shape)}"}
    array = array.reshape(-1, array.shape[-1]).astype(np.float64, copy=False)
    array = np.maximum(array, 0.0)
    mass = array.sum(axis=-1)
    valid = mass > 1.0e-12
    if not np.any(valid):
        return {"status": "unavailable", "reason": "all routing rows are empty"}
    probabilities = array[valid] / mass[valid, None]
    groups = int(probabilities.shape[-1])
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1.0e-30))).sum(axis=-1)
    normalized = entropy / math.log(max(groups, 2))
    uniform = np.full((1, groups), 1.0 / float(groups))
    max_deviation = np.max(np.abs(probabilities - uniform), axis=-1)
    max_probability = np.max(probabilities, axis=-1)
    return {
        "status": "complete",
        "rows": int(probabilities.shape[0]),
        "groups": groups,
        "entropy_mean": float(np.mean(entropy)),
        "entropy_min": float(np.min(entropy)),
        "entropy_max": float(np.max(entropy)),
        "normalized_entropy_mean": float(np.mean(normalized)),
        "normalized_entropy_min": float(np.min(normalized)),
        "uniform_max_abs_deviation_mean": float(np.mean(max_deviation)),
        "max_probability_mean": float(np.mean(max_probability)),
        "near_uniform_fraction": float(
            np.mean((normalized >= 0.90) & (max_deviation <= 0.10))
        ),
        "near_uniform_threshold": {
            "normalized_entropy_at_least": 0.90,
            "max_abs_deviation_at_most": 0.10,
        },
    }


def _routing_summary(output: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "query": ("group_control_query_routing", "group_control_query_alpha"),
        "module": ("group_control_module_incidence", "group_control_module_membership"),
        "environment": (
            "group_control_environment_incidence",
            "group_control_environment_membership",
        ),
    }
    return {
        name: _entropy_row_summary(_find_output_or_aux(output, keys))
        for name, keys in aliases.items()
    }


def _phase_refresh_summary(output: Mapping[str, Any]) -> dict[str, Any]:
    aux = output.get("interaction_aux")
    if not isinstance(aux, Mapping):
        return {"status": "unavailable", "reason": "interaction_aux missing"}
    values: dict[str, Any] = {}
    for key, value in aux.items():
        text = str(key)
        if "case_group_budget_gate_reused" in text or "case_group_budget_fine_source_refresh" in text:
            values[text] = _jsonable(value)
    return {
        "status": "complete" if values else "unavailable",
        "values": values,
        "required_interpretation": "P0 gate plan is shared; memberships, collective controls, and fine physical values refresh at P1/P2",
    }


def _parameter_by_suffix(model: Any, suffix: str, shape: tuple[int, ...] | None = None) -> tuple[str, Any] | None:
    for name, parameter in model.named_parameters():
        if str(name).endswith(suffix) and (shape is None or tuple(parameter.shape) == shape):
            return str(name), parameter
    return None


def _gradient_vector_summary(gradients: Sequence[Any]) -> dict[str, Any]:
    import torch

    present = [gradient.detach().float() for gradient in gradients if gradient is not None]
    if not present:
        return {
            "status": "unconnected",
            "finite": True,
            "nonzero": False,
            "norm": 0.0,
            "parameter_count_with_gradient": 0,
        }
    vector = torch.cat([item.reshape(-1) for item in present])
    finite = bool(torch.isfinite(vector).all())
    norm = float(vector.double().norm()) if vector.numel() else 0.0
    return {
        "status": "complete",
        "finite": finite,
        "nonzero": bool(torch.count_nonzero(vector)),
        "norm": norm if math.isfinite(norm) else None,
        "parameter_count_with_gradient": len(present),
    }


def _prototype_gradient_summary(gradient: Any) -> dict[str, Any]:
    import torch

    if gradient is None:
        return {
            "status": "unavailable",
            "reason": "prototype parameter is not connected to physical loss",
        }
    values = gradient.detach().float()
    if values.ndim != 2:
        return {"status": "unavailable", "reason": f"unexpected gradient shape {list(values.shape)}"}
    finite = bool(torch.isfinite(values).all())
    row_norms = values.double().norm(dim=-1)
    nonzero = row_norms > 1.0e-12
    if values.shape[0] > 1:
        differences = values[:, None, :] - values[None, :, :]
        pairwise = differences.abs().amax(dim=-1)
        pairwise_max = float(pairwise.max())
        pairwise_min_nonzero = float(pairwise[pairwise > 0.0].min()) if bool((pairwise > 0.0).any()) else 0.0
    else:
        pairwise_max = 0.0
        pairwise_min_nonzero = 0.0
    return {
        "status": "complete",
        "shape": list(values.shape),
        "finite": finite,
        "row_norms": [float(value) for value in row_norms],
        "all_rows_nonzero": bool(nonzero.all()),
        "all_rows_identical": bool(pairwise_max <= 1.0e-12),
        "pairwise_max_abs_difference": pairwise_max,
        "pairwise_min_nonzero_abs_difference": pairwise_min_nonzero,
        "required_rows": EXPECTED_GROUP_COUNT,
        "required_width": EXPECTED_CONTROL_DIM,
    }


def _physical_gradient_summary(model: Any, loss: Any, *, continuation: float | None) -> dict[str, Any]:
    import torch

    prototype = _parameter_by_suffix(model, ".group_codes", (EXPECTED_GROUP_COUNT, EXPECTED_CONTROL_DIM))
    gate_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "case_gate" in str(name) and parameter.requires_grad
    ]
    targets = ([] if prototype is None else [prototype[1]]) + gate_parameters
    if not targets:
        return {
            "status": "unavailable",
            "reason": "no group_codes or case_gate parameters found",
        }
    gradients = torch.autograd.grad(
        loss,
        targets,
        retain_graph=True,
        allow_unused=True,
    )
    prototype_gradient = gradients[0] if prototype is not None else None
    gate_gradients = gradients[1:] if prototype is not None else gradients
    gate_summary = _gradient_vector_summary(gate_gradients)
    continuation_value = None if continuation is None else float(continuation)
    expected_gate_zero = continuation_value is not None and abs(continuation_value) <= 1.0e-12
    if expected_gate_zero:
        gate_status = "expected_zero_at_c0"
    elif gate_summary["status"] == "unconnected":
        gate_status = "unconnected"
    else:
        gate_status = "observed"
    gate_summary = dict(gate_summary)
    gate_summary["status"] = gate_status
    gate_summary["expected_zero_at_c0"] = expected_gate_zero
    return {
        "status": "complete",
        "prototype_parameter": None if prototype is None else prototype[0],
        "prototype": _prototype_gradient_summary(prototype_gradient),
        "gate_network": gate_summary,
    }


def _scalar_losses(loss_terms: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in loss_terms.items():
        if key == "case_group_budget_schedule":
            result[key] = str(value)
            continue
        scalar = _scalar(value)
        if scalar is not None:
            result[key] = scalar
    return result


def _optimizer_update_summary(model: Any, before: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    norms: list[torch.Tensor] = []
    finite = True
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        update = parameter.detach().cpu().float() - before[name]
        finite = finite and bool(torch.isfinite(update).all())
        norms.append(update.reshape(-1))
    vector = torch.cat(norms) if norms else torch.empty(0)
    norm = float(vector.double().norm()) if vector.numel() else 0.0
    prototype = _parameter_by_suffix(model, ".group_codes", (EXPECTED_GROUP_COUNT, EXPECTED_CONTROL_DIM))
    prototype_update = None
    if prototype is not None and prototype[0] in before:
        update = prototype[1].detach().cpu().float() - before[prototype[0]]
        prototype_update = {
            "row_norms": [float(value) for value in update.double().norm(dim=-1)],
            "finite": bool(torch.isfinite(update).all()),
            "nonzero": bool(torch.count_nonzero(update)),
        }
    return {
        "status": "complete",
        "finite": bool(finite),
        "nonzero": bool(torch.count_nonzero(vector)) if vector.numel() else False,
        "norm": norm if math.isfinite(norm) else None,
        "parameter_count": len(before),
        "prototype": prototype_update,
    }


def _build_model_and_loader(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Construct the maintained model/data/optimizer objects for the audit."""

    import torch

    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.model import ChannelThermalHONFModel
    from channelthermal.training.optimizer import build_forward_optimizer
    from channelthermal.workflows.train_forward import build_model_config, resolve_auto_internal_mode
    from honf_runtime.case_protocol import WorkflowRequest
    from honf_runtime.compat import set_seed
    from honf_runtime.config_loader import load_config_bundle
    from honf_runtime.registry import load_case_plugin
    from run_run1409_real_update import _build_loader

    device = torch.device(args.device)
    profile = Path(args.profile).expanduser().resolve()
    overlay = Path(args.overlay).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    bundle = load_config_bundle(str(profile), experiment_overlay=str(overlay))
    plugin = load_case_plugin(str(bundle.case["plugin"]))
    plugin.validate_config(bundle)
    request = WorkflowRequest(workflow="forward", device=str(args.device), epochs=int(args.epochs))
    cfg = plugin._forward_config(bundle, request, output.parent)
    architecture = str(cfg["model"]["core_honf"]["forward_architecture"])
    if architecture != EXPECTED_ARCHITECTURE:
        raise ValueError(f"resolved architecture is {architecture!r}, expected {EXPECTED_ARCHITECTURE!r}")
    loss_cfg = dict(cfg["loss"])
    budget_cfg = dict(
        cfg["model"]["core_honf"]["interface_model"]["case_group_budget"]
    )
    schedule = str(budget_cfg.get("schedule", ""))
    if schedule != EXPECTED_SCHEDULE:
        raise ValueError(
            f"resolved case_group_budget.schedule is {schedule!r}, "
            f"expected {EXPECTED_SCHEDULE!r}"
        )
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    if bool(training_cfg.get("amp", False)):
        raise ValueError("the prelaunch gradient audit requires amp=false")
    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)
    dataset = GlobalChannelThermalDataset(
        dataset_cfg["packed_h5_path"],
        split=dataset_cfg.get("train_split", "train"),
        points_per_case=int(dataset_cfg.get("points_per_case", 4096)),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=bool(dataset_cfg.get("random_point_sampling", True)),
        seed=seed,
        require_converged=bool(dataset_cfg.get("require_converged", False)),
    )
    dataset.set_epoch(1)
    model_config = build_model_config(cfg, dataset)
    model = ChannelThermalHONFModel(model_config).to(device)
    model.set_global_target_normalization(
        dataset.normalizer.stats,
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
    )
    resolve_auto_internal_mode(model_config, model)
    local_path = model.local_coupling.local_surrogate_checkpoint_path
    if not model.local_surrogate_attached or not local_path or not Path(local_path).is_file():
        raise RuntimeError("the maintained local-surrogate checkpoint was not attached")
    configure = getattr(model, "configure_budgeted_schedule", None)
    if callable(configure):
        configure(schedule)
    set_progress = getattr(model, "set_training_progress", None)
    if callable(set_progress):
        set_progress(epoch=1, total_epochs=int(args.epochs))
    loader = _build_loader(cfg, dataset, device)
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(1)
    optimizer, optimizer_inventory = build_forward_optimizer(model, training_cfg)
    return (
        model,
        loader,
        optimizer,
        cfg,
        loss_cfg,
        training_cfg,
        dataset,
        optimizer_inventory,
        (profile, overlay, seed, local_path),
    )


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from channelthermal.training.epoch import (
        assemble_channelthermal_loss_terms,
        effective_local_loss_weights,
        effective_port_condition_settings,
        effective_port_global_weight,
        make_model_inputs,
        predicted_consistency_weight_for_epoch,
    )
    from honf_runtime.compat import recursive_to_device

    if int(args.batch_count) != EXPECTED_BATCHES:
        raise ValueError("the prelaunch protocol is fixed at exactly two real batches")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index != 0:
        raise ValueError("this audit must use one free GPU with --device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)

    (
        model,
        loader,
        optimizer,
        cfg,
        loss_cfg,
        training_cfg,
        dataset,
        optimizer_inventory,
        provenance,
    ) = _build_model_and_loader(args)
    profile, overlay, seed, local_path = provenance
    mode, ratio = effective_port_condition_settings(1, training_cfg)
    if str(mode).lower() != "predicted":
        raise ValueError(f"resolved epoch-1 port mode is {mode!r}, expected predicted")
    internal_weight, interface_weight = effective_local_loss_weights(loss_cfg, mode, ratio)
    predicted_weight = predicted_consistency_weight_for_epoch(1, loss_cfg)
    port_global_weight = effective_port_global_weight(loss_cfg, mode, ratio)
    clip_norm = float(training_cfg.get("gradient_clip_norm", 0.0) or 0.0)

    model.train(True)
    batch_rows: list[dict[str, Any]] = []
    optimizer_before: dict[str, Any] | None = None
    optimizer_update: dict[str, Any] | None = None
    optimizer_steps = 0
    iterator = iter(loader)
    for batch_index in range(EXPECTED_BATCHES):
        raw_batch = next(iterator)
        batch = recursive_to_device(raw_batch, device)
        batch_size = int(batch["query_xy"].shape[0])
        query_count = int(batch["query_xy"].shape[1])
        if batch_size != EXPECTED_BATCH_SIZE or query_count != EXPECTED_QUERY_COUNT:
            raise RuntimeError(
                f"expected real B{EXPECTED_BATCH_SIZE}/Q{EXPECTED_QUERY_COUNT} batch, "
                f"got B{batch_size}/Q{query_count} at batch {batch_index + 1}"
            )
        inputs = make_model_inputs(
            batch,
            local_port_condition_mode=str(mode),
            mixed_teacher_ratio=float(ratio),
            return_predicted_port_outputs=bool(predicted_weight > 0.0),
            return_port_global_consistency=bool(port_global_weight != 0.0),
        )
        inputs.update(return_routing_maps=True, return_prepared_state=True)
        output = model(**inputs)
        loss_terms = assemble_channelthermal_loss_terms(
            output,
            batch,
            model,
            loss_cfg,
            local_port_condition_mode=str(mode),
            mixed_teacher_ratio=float(ratio),
            effective_internal_temperature_weight=float(internal_weight),
            effective_interface_weight=float(interface_weight),
            predicted_consistency_weight=float(predicted_weight),
        )
        budget = _budget_summary(output)
        continuation = budget.get("continuation_c")
        physical_gradients = _physical_gradient_summary(
            model,
            loss_terms["loss_physical"],
            continuation=continuation,
        )
        batch_row = {
            "batch_index": batch_index + 1,
            "batch_size": batch_size,
            "query_count": query_count,
            "case_ids": [str(value) for value in batch.get("case_id", [])],
            "module_counts": _jsonable(batch.get("module_count")),
            "losses": _scalar_losses(loss_terms),
            "budget": budget,
            "routing_entropy": _routing_summary(output),
            "phase_refresh": _phase_refresh_summary(output),
            "physical_gradients": physical_gradients,
        }
        if batch_index == 1:
            optimizer.zero_grad(set_to_none=True)
            optimizer_before = {
                name: parameter.detach().cpu().float().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            loss_terms["loss"].backward()
            gradients = [
                parameter.grad.detach().float()
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            flat = torch.cat([gradient.reshape(-1) for gradient in gradients]) if gradients else torch.empty(0)
            preclip_finite = bool(torch.isfinite(flat).all()) if flat.numel() else True
            preclip_norm = float(flat.double().norm()) if flat.numel() else 0.0
            if clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            optimizer_steps += 1
            optimizer_update = _optimizer_update_summary(model, optimizer_before)
            optimizer_update.update(
                {
                    "preclip_gradient_finite": preclip_finite,
                    "preclip_gradient_norm": preclip_norm,
                    "gradient_clip_norm": clip_norm,
                    "step_count": optimizer_steps,
                }
            )
            batch_row["optimizer_update"] = optimizer_update
        batch_rows.append(batch_row)
        del loss_terms, output, batch, raw_batch
        gc.collect()
    torch.cuda.synchronize(device)
    if optimizer_steps != 1:
        raise RuntimeError(f"expected one optimizer step, observed {optimizer_steps}")
    entropy_checks = []
    for row in batch_rows:
        routing = row["routing_entropy"]
        entropy_checks.append(
            all(
                routing[name].get("status") == "complete"
                and routing[name].get("normalized_entropy_mean", 0.0) >= 0.90
                for name in ("query", "module", "environment")
            )
        )
    prototype_checks = []
    gate_checks = []
    for row in batch_rows:
        prototype = row["physical_gradients"].get("prototype", {})
        gate = row["physical_gradients"].get("gate_network", {})
        prototype_checks.append(
            prototype.get("status") == "complete"
            and prototype.get("shape") == [EXPECTED_GROUP_COUNT, EXPECTED_CONTROL_DIM]
            and prototype.get("finite") is True
            and prototype.get("all_rows_nonzero") is True
            and prototype.get("all_rows_identical") is False
        )
        gate_checks.append(
            gate.get("status") == "expected_zero_at_c0"
            and gate.get("finite") is True
            and gate.get("nonzero") is False
        )
    effective_gate_checks = [
        row["budget"].get("all_effective_gates_positive") is True
        and row["budget"].get("all_effective_gate_count") == EXPECTED_GROUP_COUNT
        for row in batch_rows
    ]
    update_check = bool(
        optimizer_update
        and optimizer_update.get("finite") is True
        and optimizer_update.get("nonzero") is True
        and optimizer_update.get("preclip_gradient_finite") is True
        and optimizer_update.get("step_count") == 1
    )
    protocol_checks = {
        "two_real_batches": len(batch_rows) == EXPECTED_BATCHES,
        "effective_gates_all_positive": all(effective_gate_checks),
        "routing_entropy_near_uniform": all(entropy_checks),
        "prototype_physical_gradients_finite_nonzero_nonidentical": all(prototype_checks),
        "gate_network_physical_gradient_zero_at_c0": all(gate_checks),
        "one_finite_optimizer_update": update_check,
    }
    result = {
        "schema_version": 1,
        "task": "run1409_dense_to_sparse_v2_prelaunch_epoch1_audit",
        "status": "complete",
        "architecture": EXPECTED_ARCHITECTURE,
        "schedule": EXPECTED_SCHEDULE,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "profile": str(profile),
        "experiment_overlay": str(overlay),
        "seed": int(seed),
        "epoch": 1,
        "total_epochs": int(args.epochs),
        "continuation_expected": 0.0,
        "routing_strength_expected": 0.1,
        "port_condition_mode": str(mode),
        "mixed_teacher_ratio": float(ratio),
        "case_group_budget_weight": float(loss_cfg.get("case_group_budget_weight", 0.0)),
        "local_surrogate": {
            "attached": bool(model.local_surrogate_attached),
            "checkpoint_path": str(local_path),
            "frozen": bool(model.local_coupling.local_surrogate_frozen),
        },
        "dataset": {
            "path": str(dataset.path),
            "split": str(dataset.split),
            "epoch": int(dataset.epoch),
            "random_point_sampling": bool(dataset.random_point_sampling),
        },
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "steps": optimizer_steps,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
        },
        "protocol": {
            "real_batches": EXPECTED_BATCHES,
            "required_shape": {"batch_size": EXPECTED_BATCH_SIZE, "query_count": EXPECTED_QUERY_COUNT},
            "optimizer_steps": 1,
            "routing_maps": True,
            "prepared_state": True,
            "gate_plan": "one P0 plan reused through P1/P2",
            "refresh_contract": "memberships, collective controls, and fine physical values refresh at P1/P2",
            "gate_physical_gradient": "zero or unconnected at c=0 is expected",
        },
        "protocol_checks": protocol_checks,
        "protocol_pass": bool(all(protocol_checks.values())),
        "batches": batch_rows,
        "optimizer_update": optimizer_update,
        "checkpoint_written": False,
        "managed_run_allocated": False,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status={result['status']} batches={len(batch_rows)} optimizer_steps={optimizer_steps}")
    print(f"evidence={output}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--overlay", default=str(DEFAULT_OVERLAY))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-count", type=int, default=EXPECTED_BATCHES)
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "tools" / "diagnostics"))
    run_audit(parse_args())
