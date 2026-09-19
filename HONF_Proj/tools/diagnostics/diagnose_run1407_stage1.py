#!/usr/bin/env python3
# ruff: noqa: I001
"""Bounded Run-1407 Stage-I diagnosis on the Run-1406 epoch-500 model.

This tool is deliberately an evaluation-only instrument.  It does not edit a
checkpoint or model class.  The optional counterfactual replaces only the
runtime group-control state after the P0 preparation; the later fine module
first-affine and environment K/V tensors are taken from the corresponding
phase preparation.  Thus it is a diagnostic equation, not a prediction of a
trained Run 1407.

The command is scoped to four cases and records compact JSON/NPZ evidence in
the managed diagnostics tree.  Set ``CUDA_VISIBLE_DEVICES=1`` at the command
line when using the requested physical GPU 1; the script never selects or
touches another device.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ("0273", "0653", "0680", "0298")
EPS = 1.0e-12


def _bootstrap_imports() -> None:
    for path in (PROJECT / "src", PROJECT / "Case_ThermalChannel" / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


_bootstrap_imports()

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
from channelthermal.evaluation.loading import load_model, make_batch
from channelthermal.evaluation.prepared import predict_case, select_sample
from channelthermal.evaluation.results import denormalize_predictions
from channelthermal.evaluation_tools.plots import module_and_fluid_masks
from channelthermal.workflows.compare_models import reconstruction_metrics
from honf_forward_core.interface_fields.types import PreparedInterfaceField


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=dtype)
    if result.ndim > 0 and result.shape[0] == 1:
        result = result[0]
    return result


def _finite(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    return result[np.isfinite(result)]


def _stats(value: Any) -> dict[str, float]:
    values = _finite(value)
    if values.size == 0:
        return {"count": 0.0}
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _phase_value(aux: Mapping[str, Any], phase: str, name: str) -> np.ndarray | None:
    prefix = {"P0": "initial_port_", "P1": "provisional_", "P2": ""}[phase]
    value = aux.get(prefix + name)
    if value is None:
        return None
    return _array(value)


def _relative_l2(delta: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(delta, dtype=np.float64).reshape(-1)))
    denominator = max(float(np.linalg.norm(np.asarray(reference, dtype=np.float64).reshape(-1))), EPS)
    return numerator / denominator


def _phase_changes(aux: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    phases = ("P0", "P1", "P2")
    for name in ("group_control_module_incidence", "group_control_environment_incidence", "group_control_h"):
        arrays = {phase: _phase_value(aux, phase, name) for phase in phases}
        if any(value is None for value in arrays.values()):
            result[name] = {"status": "missing"}
            continue
        transitions: dict[str, Any] = {}
        for target in ("P1", "P2"):
            source = arrays["P0"]
            current = arrays[target]
            assert source is not None and current is not None
            if source.shape != current.shape:
                transitions[f"P0_to_{target}"] = {
                    "status": "shape_mismatch",
                    "P0_shape": list(source.shape),
                    "target_shape": list(current.shape),
                }
                continue
            source_support = source > 0.0
            current_support = current > 0.0
            transitions[f"P0_to_{target}"] = {
                "mean_abs": float(np.mean(np.abs(current - source))),
                "max_abs": float(np.max(np.abs(current - source))) if source.size else 0.0,
                "relative_l2": _relative_l2(current - source, source),
                "support_flip_fraction": float(np.mean(source_support != current_support)) if source.size else 0.0,
                "support_equal_fraction": float(np.mean(source_support == current_support)) if source.size else 1.0,
                "source_count": int(source.shape[0]) if source.ndim else 0,
                "shape": list(source.shape),
            }
        result[name] = {"status": "ok", "transitions": transitions}
    return result


def _cosine_summary(keys: np.ndarray) -> dict[str, float]:
    keys = np.asarray(keys, dtype=np.float64)
    norms = np.linalg.norm(keys, axis=-1)
    normalized = keys / np.maximum(norms[..., None], EPS)
    cosine = normalized @ normalized.T
    off_diag = cosine[~np.eye(cosine.shape[0], dtype=bool)]
    return {
        "mean_off_diagonal": float(np.mean(off_diag)) if off_diag.size else 1.0,
        "min_off_diagonal": float(np.min(off_diag)) if off_diag.size else 1.0,
        "max_off_diagonal": float(np.max(off_diag)) if off_diag.size else 1.0,
    }


def _logit_spread(logits: np.ndarray, valid_rows: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    if valid_rows is not None:
        values = values[np.asarray(valid_rows, dtype=bool)]
    row_std = np.std(values, axis=-1) if values.size else np.zeros(0)
    row_range = np.ptp(values, axis=-1) if values.size else np.zeros(0)
    if values.size:
        sorted_values = np.sort(values, axis=-1)
        top_gap = sorted_values[:, -1] - sorted_values[:, -2]
    else:
        top_gap = np.zeros(0)
    return {
        "row_std": _stats(row_std),
        "row_range": _stats(row_range),
        "top_minus_second": _stats(top_gap),
        "global": _stats(values),
        "shape": list(values.shape),
    }


def _source_logits(model: Any, source_control: np.ndarray) -> np.ndarray:
    router = model.core.backend.router
    codes = router.group_codes.detach().float().cpu().numpy()
    return np.einsum("sd,kd->sk", np.asarray(source_control, dtype=np.float64), codes) / math.sqrt(float(codes.shape[-1]))


def _routing_metrics(model: Any, aux: Mapping[str, Any]) -> dict[str, Any]:
    module_a = _phase_value(aux, "P2", "group_control_module_incidence")
    env_a = _phase_value(aux, "P2", "group_control_environment_incidence")
    h = _phase_value(aux, "P2", "group_control_h")
    module_control = _phase_value(aux, "P2", "group_control_module_control")
    env_control = _phase_value(aux, "P2", "group_control_environment_control")
    alpha = _phase_value(aux, "P2", "group_control_query_routing")
    logits = _phase_value(aux, "P2", "group_control_query_logits")
    query_norm = _phase_value(aux, "P2", "group_control_query_control_norm")
    module_measure = _phase_value(aux, "P2", "group_control_module_measure")
    env_measure = _phase_value(aux, "P2", "group_control_environment_measure")
    module_mass = _phase_value(aux, "P2", "group_control_module_mass")
    env_mass = _phase_value(aux, "P2", "group_control_environment_mass")
    required = {
        "module_incidence": module_a,
        "environment_incidence": env_a,
        "group_control": h,
        "module_control": module_control,
        "environment_control": env_control,
        "query_routing": alpha,
        "query_logits": logits,
        "query_norm": query_norm,
        "module_measure": module_measure,
        "environment_measure": env_measure,
        "module_mass": module_mass,
        "environment_mass": env_mass,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        return {"status": "missing", "missing": missing}
    assert module_a is not None and env_a is not None and h is not None
    assert module_control is not None and env_control is not None
    assert alpha is not None and logits is not None and query_norm is not None
    assert module_measure is not None and env_measure is not None
    assert module_mass is not None and env_mass is not None
    module_valid = module_measure > 0.0
    env_valid = env_measure > 0.0
    module_logits = _source_logits(model, module_control)
    env_logits = _source_logits(model, env_control)
    router = model.core.backend.router
    with torch.no_grad():
        h_tensor = torch.as_tensor(h, device=next(model.parameters()).device, dtype=torch.float32)
        key_tensor = router.query_group_projection(h_tensor).detach().float().cpu().numpy()
    key_norm = np.linalg.norm(key_tensor, axis=-1)
    result: dict[str, Any] = {
        "status": "ok",
        "source_logit_spread": {
            "module": _logit_spread(module_logits, module_valid),
            "environment": _logit_spread(env_logits, env_valid),
        },
        "query_logit_spread": _logit_spread(logits),
        "query_vector_norm": _stats(query_norm),
        "old_query_key_norm": _stats(key_norm),
        "old_query_key_norm_by_group": key_norm.tolist(),
        "old_query_key_pairwise_cosine": _cosine_summary(key_tensor[0] if key_tensor.ndim == 3 else key_tensor),
        "exact_supports": {
            "module_source_degree": _stats((module_a[module_valid] > 0.0).sum(axis=-1)),
            "environment_source_degree": _stats((env_a[env_valid] > 0.0).sum(axis=-1)),
            "query_group_degree": _stats((alpha > 0.0).sum(axis=-1)),
            "module_source_singleton_fraction": float(np.mean((module_a[module_valid] > 0.0).sum(axis=-1) == 1)) if np.any(module_valid) else 0.0,
            "environment_source_singleton_fraction": float(np.mean((env_a[env_valid] > 0.0).sum(axis=-1) == 1)) if np.any(env_valid) else 0.0,
            "query_singleton_fraction": float(np.mean((alpha > 0.0).sum(axis=-1) == 1)) if alpha.size else 0.0,
            "query_exact_zero_fraction": float(np.mean(alpha == 0.0)) if alpha.size else 0.0,
        },
        "source_mass": {
            "module": module_mass.tolist(),
            "environment": env_mass.tolist(),
            "module_empty_groups": np.flatnonzero(module_mass <= 0.0).astype(int).tolist(),
            "environment_empty_groups": np.flatnonzero(env_mass <= 0.0).astype(int).tolist(),
        },
    }
    for name, incidence, mass in (
        ("module", module_a, module_mass),
        ("environment", env_a, env_mass),
    ):
        empty = np.asarray(mass) <= 0.0
        probability_on_empty = alpha[:, empty].sum(axis=-1) if np.any(empty) else np.zeros(alpha.shape[0])
        result["query_probability_on_source_empty_groups_" + name] = {
            "mean": float(np.mean(probability_on_empty)) if probability_on_empty.size else 0.0,
            "p95": float(np.quantile(probability_on_empty, 0.95)) if probability_on_empty.size else 0.0,
            "max": float(np.max(probability_on_empty)) if probability_on_empty.size else 0.0,
            "fraction_queries_positive": float(np.mean(probability_on_empty > 0.0)) if probability_on_empty.size else 0.0,
            "empty_group_count": int(np.sum(empty)),
        }
    return result


def _prediction_from_output(output: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    x_grid = np.asarray(sample["x_grid"])
    field = output["pred_field"].detach().cpu().numpy()[0].reshape(*x_grid.shape, -1)
    return {
        "pred_field_grid": field.astype(np.float32),
        "pred_internal_temperature": output["pred_internal_temperature"].detach().cpu().numpy()[0].astype(np.float32),
        "pred_interface": output["pred_interface"].detach().cpu().numpy()[0].astype(np.float32),
        "pred_port_condition": output["pred_port_condition"].detach().cpu().numpy()[0].astype(np.float32),
        "pred_port_condition_raw": output.get("pred_port_condition_raw", output["pred_port_condition"]).detach().cpu().numpy()[0].astype(np.float32),
        "interface_flux_mode": output.get("interface_source", "unknown"),
    }


def _refresh_with_shared_controller(
    core: Any,
    encoded: Any,
    module_states: torch.Tensor,
    p0_backend: Mapping[str, Any],
    *,
    return_routing_maps: bool,
) -> dict[str, Any]:
    """Refresh Dense fine tensors while retaining the live P0 controller.

    This mirrors only the compact Run-1406 preparation boundary.  It keeps
    ``prepare_fine_messages`` and both fine source projections live for the
    current module state, then applies the P0 memberships/control banks to
    those refreshed values.  The code is intentionally local to the
    diagnostic tool so this explanatory counterfactual cannot become a model
    default or a checkpoint-loaded runtime cache.
    """

    backend = core.backend
    dense_encoded = replace(
        encoded,
        coordinate_scale=backend._dense_coordinate_scale(encoded, int(module_states.shape[0])),
    )
    fine = backend.prepare_fine_messages(dense_encoded, module_states)
    global_env = encoded.global_token[:, None, :].expand(-1, fine["env_tokens"].shape[1], -1)
    contextual_env = fine["env_tokens"] + backend.env_update(
        torch.cat([fine["env_tokens"], fine["environment_messages"], global_env], dim=-1)
    )
    controls = p0_backend["group_control_state"]
    module_affine = backend._prepare_module_affine(encoded, fine["module_tokens"])
    environment_key, environment_value, source_group_control, head_control, head_source_control = (
        backend._prepare_environment_bank(contextual_env, controls)
    )
    # The controller-dependent banks are deliberately taken from P0.  The
    # contextual module/environment values and K/V above remain phase-live.
    return {
        "module_tokens": fine["module_tokens"],
        "env_tokens": contextual_env,
        "group_control_state": controls,
        "module_first_affine": module_affine,
        "module_control_bank": p0_backend["module_control_bank"],
        "module_control_bank_membership": controls.module_membership,
        "module_control_bank_group_control": controls.group_control,
        "environment_keys": environment_key,
        "environment_values": environment_value,
        "environment_source_group_control": source_group_control,
        "environment_head_control": head_control,
        "environment_head_source_control": head_source_control,
        "group_control_preparation_aux": p0_backend["group_control_preparation_aux"],
    }


def _direct_forward(
    model: Any,
    sample: Mapping[str, Any],
    device: torch.device,
    *,
    reuse_p0_index: bool,
    collect_timing: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query = np.stack([sample["x_grid"].reshape(-1), sample["y_grid"].reshape(-1)], axis=-1).astype(np.float32)
    batch = make_batch(sample, query, device)
    records: dict[str, list[float]] = {"p0_port": [], "p1_refinement": [], "p2_field": [], "p2_port_global_consistency": []}
    original_prepare = model.core.prepare
    shared_p0: list[PreparedInterfaceField] = []

    def wrapped_prepare(
        core_self: Any,
        encoded: Any,
        module_states: torch.Tensor,
        layout_cache: Any = None,
        *,
        return_routing_maps: bool = False,
    ) -> PreparedInterfaceField:
        role = str(getattr(core_self, "_interface_read_role", "unknown"))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        if reuse_p0_index and role in {"p1_refinement", "p2_field"} and shared_p0:
            backend_state = _refresh_with_shared_controller(
                core_self,
                encoded,
                module_states,
                shared_p0[0].backend_state,
                return_routing_maps=return_routing_maps,
            )
            coarse_state = core_self.common.prepare_coarse(
                module_states,
                encoded.env_tokens,
                encoded.module_present,
                encoded.env_weights,
            )
            aux = {
                "forward_architecture": core_self.config.forward_architecture,
                "coarse_latent_count": 0,
                "main_latent_count": 0,
            }
            aux.update(
                core_self.backend.preparation_aux(
                    backend_state,
                    include_diagnostics=bool(return_routing_maps),
                )
            )
            prepared = PreparedInterfaceField(
                encoded,
                module_states,
                backend_state,
                coarse_state,
                aux,
            )
        else:
            prepared = original_prepare(
                encoded,
                module_states,
                layout_cache=layout_cache,
                return_routing_maps=return_routing_maps,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if collect_timing and role in records:
            records[role].append(float(time.perf_counter() - started))
        if not reuse_p0_index:
            return prepared
        if role == "p0_port":
            if not shared_p0:
                shared_p0.append(prepared)
            return prepared
        return prepared

    # Install the wrapper for both paths so standard preparation timings and
    # the counterfactual are measured through the same boundary.
    model.core.prepare = types.MethodType(wrapped_prepare, model.core)
    try:
        with torch.no_grad():
            output = model(
                batch["structure"],
                batch["query_xy"],
                interface_condition=batch.get("interface_condition"),
                local_module_params=batch.get("local_module_params"),
                teacher_port_tokens=batch.get("teacher_port_tokens"),
                local_query_points=batch.get("module_internal_query_points"),
                local_port_condition_mode="predicted",
                return_routing_maps=False,
                return_prepared_state=False,
            )
    finally:
        model.core.prepare = original_prepare
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return _prediction_from_output(output, sample), {
        "prepare_seconds": {key: values for key, values in records.items()},
        "full_forward_seconds": None,
    }


def _time_forward(
    model: Any,
    sample: Mapping[str, Any],
    device: torch.device,
    *,
    reuse_p0_index: bool,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    _, timing = _direct_forward(
        model,
        sample,
        device,
        reuse_p0_index=reuse_p0_index,
        collect_timing=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timing["full_forward_seconds"] = float(time.perf_counter() - started)
    return timing


def _metric_row(
    predictions: Mapping[str, Any],
    raw_sample: Mapping[str, Any],
    dataset: GlobalChannelThermalDataset,
    *,
    normalize_targets: bool,
) -> dict[str, Any]:
    physical = denormalize_predictions(dict(predictions), dataset, normalize_targets)
    row, _ = reconstruction_metrics(
        base_row={"case_id": str(raw_sample["case_id"])},
        predictions=physical,
        raw_sample=raw_sample,
        dataset=dataset,
        checkpoint_targets_normalized=False,
        channel_order=dataset.channel_order,
    )
    return row


def _metric_attribution(
    predictions: Mapping[str, Any],
    raw_sample: Mapping[str, Any],
    dataset: GlobalChannelThermalDataset,
    *,
    normalize_targets: bool,
) -> dict[str, Any]:
    physical = denormalize_predictions(dict(predictions), dataset, normalize_targets)
    pred = np.asarray(physical["pred_field_grid"], dtype=np.float64)
    target = np.asarray(raw_sample["steady_field"], dtype=np.float64)[..., : pred.shape[-1]]
    pred_norm = dataset.normalizer.normalize_fields(pred)
    target_norm = dataset.normalizer.normalize_fields(target)
    module_mask, fluid_mask = module_and_fluid_masks(raw_sample, pred)

    def components(mask: np.ndarray | None) -> dict[str, float]:
        selected_pred = pred_norm if mask is None else pred_norm[mask, :]
        selected_target = target_norm if mask is None else target_norm[mask, :]
        diff = selected_pred.reshape(-1) - selected_target.reshape(-1)
        truth = selected_target.reshape(-1)
        return {
            "sse": float(np.dot(diff, diff)),
            "target_sse": float(np.dot(truth, truth)),
            "num_values": float(diff.size),
            "relative_l2": float(np.linalg.norm(diff) / max(float(np.linalg.norm(truth)), EPS)),
        }

    all_parts = components(None)
    module_parts = components(module_mask)
    fluid_parts = components(fluid_mask)
    per_channel: dict[str, Any] = {}
    for index, name in enumerate(dataset.channel_order[: pred.shape[-1]]):
        per_channel[str(name)] = {}
        # Use scalar channel arrays to match compare_models' field_<channel>_all
        # and field_<channel>_fluid definitions.
        for label, mask in (("all", None), ("module", module_mask), ("fluid", fluid_mask)):
            a = pred_norm[..., index] if mask is None else pred_norm[..., index][mask]
            b = target_norm[..., index] if mask is None else target_norm[..., index][mask]
            delta = a.reshape(-1) - b.reshape(-1)
            truth = b.reshape(-1)
            per_channel[str(name)][label] = {
                "sse": float(np.dot(delta, delta)),
                "target_sse": float(np.dot(truth, truth)),
                "num_values": float(delta.size),
                "relative_l2": float(np.linalg.norm(delta) / max(float(np.linalg.norm(truth)), EPS)),
            }
    return {
        "definition": {
            "all_domain": "compare_models.reconstruction_metrics global_field_all: normalized field grid over every grid cell and channel",
            "fluid_domain": "compare_models.reconstruction_metrics global_field_fluid: same normalized field tensor masked by module_and_fluid_masks fluid complement",
            "internal_temperature": "separate physical local-module tensor metric; it is not included in global_field_all SSE",
            "surface_temperature": "separate physical interface_target[...,0] metric",
            "normal_heat_flux": "separate physical interface_target[...,1] metric",
            "channel_order": list(dataset.channel_order[: pred.shape[-1]]),
        },
        "mask_counts": {
            "all_grid_cells": int(pred.shape[0] * pred.shape[1]),
            "module_grid_cells": int(np.count_nonzero(module_mask)),
            "fluid_grid_cells": int(np.count_nonzero(fluid_mask)),
            "field_channels": int(pred.shape[-1]),
        },
        "all": all_parts,
        "module_region_field": module_parts,
        "fluid_region_field": fluid_parts,
        "decomposition_residual": {
            "sse": float(all_parts["sse"] - module_parts["sse"] - fluid_parts["sse"]),
            "target_sse": float(all_parts["target_sse"] - module_parts["target_sse"] - fluid_parts["target_sse"]),
        },
        "per_channel": per_channel,
        "interpretation": "The all-domain field aggregate decomposes into solid/module-grid and fluid-grid field SSEs, but the separately reported internal-temperature metric cannot be treated as the causal source of an all-domain advantage without a further field/target attribution. This audit does not establish CFD truth.",
    }


def _case_diagnostic(
    model: Any,
    sample: Mapping[str, Any],
    raw_sample: Mapping[str, Any],
    dataset: GlobalChannelThermalDataset,
    device: torch.device,
    *,
    query_batch_size: int,
    normalize_targets: bool,
    output_dir: Path,
) -> dict[str, Any]:
    case_id = str(raw_sample["case_id"])
    predictions = predict_case(
        model,
        dict(sample),
        device,
        query_batch_size=int(query_batch_size),
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.5,
        return_routing_maps=True,
        return_prepared_state=False,
    )
    aux = predictions["interaction_aux"]
    routing = _routing_metrics(model, aux)
    changes = _phase_changes(aux)
    standard_metrics = _metric_row(predictions, raw_sample, dataset, normalize_targets=normalize_targets)
    attribution = _metric_attribution(predictions, raw_sample, dataset, normalize_targets=normalize_targets)
    # Timings are separate calls with maps disabled; the maps call above is
    # untimed evidence and must not be mistaken for a normal executor timing.
    standard_timing = _time_forward(model, sample, device, reuse_p0_index=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cf_started = time.perf_counter()
    cf_predictions, cf_timing = _direct_forward(
        model,
        sample,
        device,
        reuse_p0_index=True,
        collect_timing=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cf_timing["full_forward_seconds"] = float(time.perf_counter() - cf_started)
    cf_metrics = _metric_row(cf_predictions, raw_sample, dataset, normalize_targets=normalize_targets)

    metric_delta = {}
    for key, value in standard_metrics.items():
        if not key.endswith(("_norm_l2", "_physical_relative_l2", "_physical_mae", "_physical_rmse")):
            continue
        candidate = cf_metrics.get(key)
        if isinstance(value, (int, float)) and isinstance(candidate, (int, float)):
            metric_delta[key] = float(candidate - value)
    arrays = {}
    for name in (
        "group_control_module_incidence",
        "group_control_environment_incidence",
        "group_control_h",
        "group_control_query_routing",
        "group_control_query_logits",
        "group_control_module_control",
        "group_control_environment_control",
        "group_control_module_mass",
        "group_control_environment_mass",
    ):
        value = aux.get(name)
        if value is not None:
            arrays[name] = np.asarray(value)
    standard_physical = denormalize_predictions(dict(predictions), dataset, normalize_targets)
    counterfactual_physical = denormalize_predictions(dict(cf_predictions), dataset, normalize_targets)
    arrays.update(
        {
            "standard_pred_field": np.asarray(standard_physical["pred_field_grid"], dtype=np.float32),
            "counterfactual_pred_field": np.asarray(counterfactual_physical["pred_field_grid"], dtype=np.float32),
            "target_field": np.asarray(raw_sample["steady_field"], dtype=np.float32),
            "module_mask": np.asarray(module_and_fluid_masks(raw_sample)[0], dtype=np.uint8),
            "fluid_mask": np.asarray(module_and_fluid_masks(raw_sample)[1], dtype=np.uint8),
        }
    )
    np.savez_compressed(output_dir / f"case_{case_id}_stage1_arrays.npz", **arrays)
    return {
        "case_id": case_id,
        "protocol": {
            "checkpoint_phase": "Run 1406 exact epoch 500",
            "query_count": int(np.asarray(sample["x_grid"]).size),
            "query_batch_size": int(query_batch_size),
            "maps_call_untimed": True,
            "local_port_condition_mode": "predicted",
            "counterfactual": "frozen 1406 P0 controller with later fine module affine and environment K/V refreshed",
        },
        "routing": routing,
        "phase_changes": changes,
        "standard_metrics": standard_metrics,
        "frozen_p0_reuse_metrics": cf_metrics,
        "frozen_p0_reuse_metric_delta_vs_standard": metric_delta,
        "standard_timing": standard_timing,
        "frozen_p0_reuse_timing": cf_timing,
        "metric_attribution_audit": attribution,
        "artifact_arrays": str(output_dir / f"case_{case_id}_stage1_arrays.npz"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--case-id", action="append", dest="case_ids", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(device)
    model, checkpoint = load_model(args.checkpoint.resolve(), device)
    model.eval()
    train_config = checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {})
    stats = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in checkpoint.get("global_normalization_stats", {}).items()
    }
    normalizer = H5Normalizer(stats) if stats else None
    normalize_inputs = bool(dataset_config.get("normalize_inputs", False))
    normalize_targets = bool(dataset_config.get("normalize_targets", False))
    dataset = GlobalChannelThermalDataset(
        str(args.dataset),
        split=str(dataset_config.get("val_split", "test")),
        points_per_case=1,
        normalize_inputs=normalize_inputs,
        normalize_targets=normalize_targets,
        random_point_sampling=False,
        include_grid=True,
        normalizer=normalizer,
    )
    raw_dataset = GlobalChannelThermalDataset(
        str(args.dataset),
        split=str(dataset_config.get("val_split", "test")),
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=True,
    )
    case_ids = tuple(args.case_ids) if args.case_ids else DEFAULT_CASES
    rows = []
    for case_id in case_ids:
        sample = select_sample(dataset, str(case_id), 0)
        raw_sample = select_sample(raw_dataset, str(case_id), 0)
        rows.append(
            _case_diagnostic(
                model,
                sample,
                raw_sample,
                dataset,
                device,
                query_batch_size=int(args.query_batch_size),
                normalize_targets=normalize_targets,
                output_dir=output_dir,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest = {
        "schema_version": 1,
        "task": "run1407_stage1_bounded_diagnosis",
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "device_argument": str(args.device),
        "visible_cuda_device_note": "Caller must set CUDA_VISIBLE_DEVICES=1 so cuda:0 maps to physical GPU 1; this tool does not inspect or use GPU 0.",
        "cases": [str(value) for value in case_ids],
        "results": rows,
        "limitations": [
            "The frozen P0 controller is an explanatory inference counterfactual, not a trained 1407 accuracy estimate.",
            "Timing calls are single-case measurements with maps disabled; they are not the matched 90-case benchmark protocol.",
            "Learned routes are interaction diagnostics, not physical causality or CFD validation.",
        ],
    }
    _write_json(output_dir / "stage1_diagnosis.json", manifest)
    _write_json(
        output_dir / "commands.json",
        {
            "tool": str(Path(__file__).resolve()),
            "command_contract": "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src python tools/diagnostics/diagnose_run1407_stage1.py --checkpoint ... --dataset ... --output-dir ... --device cuda:0 --query-batch-size 8192",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
