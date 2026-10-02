#!/usr/bin/env python3
"""Measure the paired Thermal value-weight candidates on fixed native evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "diagnostics/generated/thermal_maturation_20260929/controlled_run"
SOURCE_CHECKPOINT = RUN_ROOT / "checkpoints/G_u1300_training_checkpoint.pt"
SOURCE_MANIFEST = RUN_ROOT / "run_manifest_u1300_closed_snapshot.json"
OUTPUT_ROOT = ROOT / "diagnostics/generated/directed_stable_20261002/thermal/value_recovery_pair"
ATLAS = ROOT / "diagnostics/generated/interactions/physical_response_atlas_20260926/families"
GPU2_UUID = "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39"
PROCESS_STATE: dict[str, Any] = {}
ADDITIONAL_FAMILIES = ("0325", "0340")
ALL_CALIBRATION_FAMILIES = ("0310", "0325", "0340", "0355")
CHANNELS = {
    "fluid_fields": ("u", "v", "p", "omega", "temperature"),
    "interface": ("T_surface", "q_normal"),
    "solid_temperature": ("temperature",),
}

sys.path.insert(0, str(CASE_ROOT / "scripts"))
import run_controlled_maturation as driver
from channelthermal.evaluation.loading import load_model
from channelthermal.interaction_evidence.reference_adapter import load_stored_reference_case
from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.response_control.contracts import (
    DesignInput,
    context_inputs,
    role_queries_from_record,
    role_queries_from_stencil,
)
from channelthermal.response_control.historical import select_value_recovery_historical_cases
from channelthermal.response_control.native import DifferentiableThermalOperator
from channelthermal.response_control.thermal import reduce_native_thermal_quantities

from honf_runtime.compat import load_trusted_checkpoint


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def finite_metrics(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Weighted field errors plus complete validity and finite-value counts."""

    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.asarray(mask, dtype=bool).reshape(-1) & np.isfinite(truth) & np.isfinite(weight) & (weight > 0.0)
    finite_pred = valid & np.isfinite(pred)
    selected = int(valid.sum())
    finite_count = int(finite_pred.sum())
    result: dict[str, Any] = {
        "valid_entry_count": selected,
        "finite_prediction_entry_count": finite_count,
        "nonfinite_prediction_entry_count": selected - finite_count,
        "finite_fraction": float(finite_count / selected) if selected else None,
    }
    if not finite_count:
        result.update(
            {
                "rmse_native_units": None,
                "target_rms_native_units": None,
                "signal_relative_rmse_unitless": None,
                "mean_absolute_error_native_units": None,
                "max_absolute_error_native_units": None,
                "p95_absolute_error_native_units": None,
                "mean_signed_error_native_units": None,
            }
        )
        return result
    selected_weights = weight[finite_pred]
    normalized = selected_weights / selected_weights.sum(dtype=np.float64)
    error = pred[finite_pred] - truth[finite_pred]
    absolute = np.abs(error)
    target_rms = float(np.sqrt(np.sum(normalized * np.square(truth[finite_pred]))))
    rmse = float(np.sqrt(np.sum(normalized * np.square(error))))
    result.update(
        {
            "rmse_native_units": rmse,
            "target_rms_native_units": target_rms,
            "signal_relative_rmse_unitless": rmse / target_rms if target_rms > np.finfo(np.float64).eps else None,
            "mean_absolute_error_native_units": float(np.sum(normalized * absolute)),
            "max_absolute_error_native_units": float(np.max(absolute)),
            "p95_absolute_error_native_units": float(np.quantile(absolute, 0.95)),
            "mean_signed_error_native_units": float(np.sum(normalized * error)),
        }
    )
    return result


def _quantity_value(record: Any, name: str) -> float:
    quantity = record.output.quantities.get(name)
    if quantity is None:
        raise ValueError(f"Stored reference case {record.design.case_id} has no {name!r} quantity.")
    return float(quantity.value)


def reference_module_peaks(record: Any) -> dict[str, float]:
    role = record.output.roles["solid_temperature"]
    col = role.channel_names.index("temperature") if "temperature" in role.channel_names else 0
    if role.receiver_module_ids is None:
        raise ValueError("Stored material outputs do not identify receiver module IDs.")
    output: dict[str, float] = {}
    active = {module.module_id for module in record.design.active_modules}
    for module_id in record.output.active_module_ids:
        if module_id not in active:
            raise ValueError(f"Reference active module {module_id!r} is absent from its design.")
        rows = np.asarray([current == module_id for current in role.receiver_module_ids], dtype=bool)
        rows &= np.asarray(role.valid_mask[:, col], dtype=bool)
        values = np.asarray(role.values[:, col], dtype=np.float64)
        if not rows.any() or not np.isfinite(values[rows]).all():
            raise ValueError(f"Reference material peak is unavailable for module {module_id!r}.")
        output[module_id] = float(np.max(values[rows]))
    return output


class CountedOperator:
    """Record each complete native wrapper attempt by case, state and purpose."""

    def __init__(self, operator: Any, device: torch.device, output: Path, counters: dict[str, Any]) -> None:
        self.operator = operator
        self.device = device
        self.output = output
        self.counters = counters
        self.context: dict[str, Any] = {}

    def set_context(self, **values: Any) -> None:
        self.context = dict(values)

    def __call__(self, design: Any, context: Any, queries: Any) -> Any:
        purpose = str(self.context.get("purpose", "native_model_evaluation"))
        model = str(self.context.get("model", "unknown"))
        key = f"{purpose}:{model}"
        counter = self.counters.setdefault(key, {"attempted": 0, "succeeded": 0, "failed": 0})
        counter["attempted"] += 1
        query_counts = {role: int(query.query_features.shape[0]) for role, query in queries.items()}
        units = {role: list(query.channel_units) for role, query in queries.items()}
        tensors: list[torch.Tensor] = [design.module_positions, design.module_heating, design.module_present]
        values = context if isinstance(context, Mapping) else getattr(context, "values", {})
        if isinstance(values, Mapping):
            tensors.extend(value for value in values.values() if isinstance(value, torch.Tensor))
        tensors.extend(query.query_features for query in queries.values())
        event = {
            **self.context,
            "purpose": purpose,
            "model": model,
            "attempt_index_for_model_and_purpose": counter["attempted"],
            "receiver_counts_by_role": query_counts,
            "receiver_count_total": int(sum(query_counts.values())),
            "receiver_units_by_role": units,
            "input_devices": sorted({str(tensor.device) for tensor in tensors}),
            "started_at_utc": utc_now(),
        }
        started = time.monotonic()
        try:
            if not tensors or any(tensor.device != self.device for tensor in tensors):
                raise RuntimeError(f"Native wrapper inputs do not all reside on {self.device}.")
            result = self.operator(design, context, queries)
            if any(value.device != self.device for value in result.role_values.values()):
                raise RuntimeError(f"Native wrapper outputs do not all reside on {self.device}.")
            counter["succeeded"] += 1
            event["status"] = "success"
            event["output_states"] = [self.context.get("state", "unspecified")]
            event["output_shapes_by_role"] = {role: list(value.shape) for role, value in result.role_values.items()}
            return result
        except Exception as exc:
            counter["failed"] += 1
            event["status"] = "failure"
            event["error_type"] = type(exc).__name__
            event["error"] = str(exc)
            raise
        finally:
            event["elapsed_wrapper_wall_seconds"] = time.monotonic() - started
            event["finished_at_utc"] = utc_now()
            append_jsonl(self.output / "native_calls.jsonl", event)


def _make_calibration_case(family: str) -> dict[str, Any]:
    path = ATLAS / f"calibration_{family}_responses.npz"
    stencil, metadata = load_response_atlas_stencil(path)
    if stencil.physical_family_id != f"stored_family:{family}" or stencil.split.value != "calibration":
        raise ValueError(f"Calibration atlas identity or partition mismatch for {family}.")
    if "i_plus" not in stencil.variants:
        raise ValueError(f"Calibration family {family} lacks its paired i_plus state.")
    baseline = stencil.baseline
    variant = stencil.variants["i_plus"]
    for role in baseline.output.roles:
        # Validate the full fixed query universe, including IDs, material
        # receiver ownership, features, channel schema, and quadrature weights.
        stencil.finite_change("i_plus", role)
    return {
        "case_id": f"stored_family:{family}",
        "family_id": f"stored_family:{family}",
        "split": "calibration",
        "kind": "calibration_response_atlas",
        "metadata": metadata,
        "stencil": stencil,
        "baseline": baseline,
        "i_plus": variant,
    }


def _make_historical_case(dataset_path: Path, selection: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(selection["case_id"])
    record = load_stored_reference_case(dataset_path, case_id)
    if record.design.split.value != "train" or record.output is None:
        raise ValueError(f"Selected historical case {case_id} is not a stored train reference.")
    return {
        "case_id": case_id,
        "family_id": record.design.physical_family_id,
        "split": "train",
        "kind": "historical_stored_reference",
        "metadata": dict(selection),
        # Historical train rows provide a baseline record only. They have no
        # perturbed solve and therefore must not be wrapped as a response stencil.
        "stencil": None,
        "baseline": record,
        "i_plus": None,
    }


def _input_design_summary(case: Mapping[str, Any]) -> dict[str, Any]:
    records = {"baseline": case["baseline"]}
    if case.get("i_plus") is not None:
        records["i_plus"] = case["i_plus"]
    result: dict[str, Any] = {
        "module_position_units": "dataset coordinate units",
        "module_heating_units": "dataset heating units",
        "context_native_values": dict(case["baseline"].context.values),
        "states": {},
    }
    for state, record in records.items():
        result["states"][state] = {
            "module_ids": [module.module_id for module in record.design.modules],
            "module_positions_xy_native": [list(module.position_xy) for module in record.design.modules],
            "module_heating_native": [float(module.heating) for module in record.design.modules],
            "module_active": [bool(module.active) for module in record.design.modules],
            "active_module_ids": list(record.output.active_module_ids),
        }
    if "i_plus" in records:
        baseline_modules = records["baseline"].design.modules
        plus_modules = records["i_plus"].design.modules
        if [module.module_id for module in baseline_modules] != [module.module_id for module in plus_modules]:
            raise ValueError(f"Paired design module IDs differ for {case['case_id']}.")
        position_delta = np.asarray([module.position_xy for module in plus_modules], dtype=np.float64) - np.asarray(
            [module.position_xy for module in baseline_modules], dtype=np.float64
        )
        heating_delta = np.asarray([module.heating for module in plus_modules], dtype=np.float64) - np.asarray(
            [module.heating for module in baseline_modules], dtype=np.float64
        )
        moved = [
            module.module_id
            for index, module in enumerate(plus_modules)
            if np.any(np.abs(position_delta[index]) > 1.0e-8) or abs(heating_delta[index]) > 1.0e-8
        ]
        result["paired_design_delta_i_plus_minus_baseline"] = {
            "module_ids": [module.module_id for module in baseline_modules],
            "module_position_delta_xy_native": position_delta.tolist(),
            "module_heating_delta_native": heating_delta.tolist(),
            "changed_module_ids": moved,
        }
    return result


def _to_numpy(prediction: Any) -> dict[str, np.ndarray]:
    return {role: value.detach().cpu().numpy().copy() for role, value in prediction.role_values.items()}


def _metric_block(
    model_predictions: Mapping[str, Mapping[str, np.ndarray]],
    case: Mapping[str, Any],
    *,
    states: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_record = case["baseline"]
    for role_name, channels in CHANNELS.items():
        baseline_role = baseline_record.output.roles[role_name]
        for channel in channels:
            channel_index = baseline_role.channel_names.index(channel)
            base_target = np.asarray(baseline_role.values)
            base_mask = np.asarray(baseline_role.valid_mask[:, channel_index], dtype=bool)
            base_weights = np.asarray(baseline_role.quadrature_weights, dtype=np.float64)
            for model_name, predictions in model_predictions.items():
                base_pred = predictions["baseline"][role_name]
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "split": case["split"],
                        "model": model_name,
                        "role": role_name,
                        "channel": channel,
                        "units": baseline_role.channel_units[channel_index],
                        "quantity": "absolute_baseline",
                        "metrics": finite_metrics(
                            base_pred[:, channel_index], base_target[:, channel_index], base_weights, base_mask
                        ),
                    }
                )
                if "i_plus" not in states:
                    continue
                variant = case["i_plus"]
                variant_role = variant.output.roles[role_name]
                variant_col = variant_role.channel_names.index(channel)
                next_pred = predictions["i_plus"][role_name]
                next_target = np.asarray(variant_role.values)
                next_mask = np.asarray(variant_role.valid_mask[:, variant_col], dtype=bool)
                next_weights = np.asarray(variant_role.quadrature_weights, dtype=np.float64)
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "split": case["split"],
                        "model": model_name,
                        "role": role_name,
                        "channel": channel,
                        "units": variant_role.channel_units[variant_col],
                        "quantity": "absolute_i_plus",
                        "metrics": finite_metrics(
                            next_pred[:, variant_col], next_target[:, variant_col], next_weights, next_mask
                        ),
                    }
                )
                finite_pred = next_pred[:, variant_col] - base_pred[:, channel_index]
                finite_target = next_target[:, variant_col] - base_target[:, channel_index]
                finite_mask = base_mask & next_mask
                response_metrics = finite_metrics(
                    finite_pred, finite_target, np.minimum(base_weights, next_weights), finite_mask
                )
                resolution = case["stencil"].finite_change("i_plus", role_name)
                noise_floor = resolution.noise_floor
                if noise_floor is None:
                    response_metrics.update(
                        {
                            "noise_floor_rms_native_units": None,
                            "response_resolved_entry_count": None,
                            "response_unresolved_entry_count": None,
                            "response_resolution_status": "unknown_no_noise_floor",
                        }
                    )
                else:
                    measure = np.minimum(base_weights, next_weights)
                    selected = finite_mask & (measure > 0.0)
                    floor_values = np.asarray(noise_floor, dtype=np.float64)[:, variant_col]
                    resolved = selected & (np.abs(finite_target) > floor_values)
                    selected_weights = measure[selected]
                    if selected_weights.size:
                        selected_weights = selected_weights / selected_weights.sum(dtype=np.float64)
                    response_metrics.update(
                        {
                            "noise_floor_rms_native_units": (
                                float(np.sqrt(np.sum(selected_weights * np.square(floor_values[selected]))))
                                if selected_weights.size
                                else None
                            ),
                            "response_resolved_entry_count": int(resolved.sum()),
                            "response_unresolved_entry_count": int(selected.sum() - resolved.sum()),
                            "response_resolution_status": "resolved_some" if resolved.any() else "unresolved_vs_floor",
                        }
                    )
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "split": case["split"],
                        "model": model_name,
                        "role": role_name,
                        "channel": channel,
                        "units": variant_role.channel_units[variant_col],
                        "quantity": "finite_i_plus",
                        "metrics": response_metrics,
                    }
                )
    return rows


def _scalar_block(
    case: Mapping[str, Any],
    model_predictions: Mapping[str, Mapping[str, np.ndarray]],
    model_scalars: Mapping[str, Mapping[str, Any]],
    *,
    states: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    records = {"baseline": case["baseline"]}
    if "i_plus" in states:
        records["i_plus"] = case["i_plus"]
    for model_name in model_predictions:
        for state, record in records.items():
            target_pressure = _quantity_value(record, "pressure_drop")
            predicted_pressure = model_scalars[model_name][state]["pressure_drop"]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "model": model_name,
                    "state": state,
                    "functional": "pressure_drop",
                    "units": model_scalars[model_name][state]["pressure_units"],
                    "target": target_pressure,
                    "prediction": predicted_pressure,
                    "signed_error": predicted_pressure - target_pressure,
                    "absolute_error": abs(predicted_pressure - target_pressure),
                }
            )
            target_peaks = reference_module_peaks(record)
            predicted_peaks = model_scalars[model_name][state]["module_peak_temperature"]
            if set(target_peaks) != set(predicted_peaks):
                raise ValueError(f"{model_name}/{case['case_id']}/{state}: reference and model peak module IDs differ.")
            for module_id, target in target_peaks.items():
                prediction = predicted_peaks[module_id]
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "model": model_name,
                        "state": state,
                        "functional": "module_peak_temperature",
                        "module_id": module_id,
                        "units": model_scalars[model_name][state]["temperature_units"],
                        "target": target,
                        "prediction": prediction,
                        "signed_error": prediction - target,
                        "absolute_error": abs(prediction - target),
                    }
                )
        if "i_plus" in states:
            target_base = _quantity_value(case["baseline"], "pressure_drop")
            target_next = _quantity_value(case["i_plus"], "pressure_drop")
            pred_base = model_scalars[model_name]["baseline"]["pressure_drop"]
            pred_next = model_scalars[model_name]["i_plus"]["pressure_drop"]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "model": model_name,
                    "state": "finite_i_plus",
                    "functional": "pressure_drop_response",
                    "units": model_scalars[model_name]["baseline"]["pressure_units"],
                    "target": target_next - target_base,
                    "prediction": pred_next - pred_base,
                    "signed_error": (pred_next - pred_base) - (target_next - target_base),
                    "absolute_error": abs((pred_next - pred_base) - (target_next - target_base)),
                }
            )
            target_base_peaks = reference_module_peaks(case["baseline"])
            target_next_peaks = reference_module_peaks(case["i_plus"])
            pred_base_peaks = model_scalars[model_name]["baseline"]["module_peak_temperature"]
            pred_next_peaks = model_scalars[model_name]["i_plus"]["module_peak_temperature"]
            for module_id in target_base_peaks:
                target_delta = target_next_peaks[module_id] - target_base_peaks[module_id]
                pred_delta = pred_next_peaks[module_id] - pred_base_peaks[module_id]
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "model": model_name,
                        "state": "finite_i_plus",
                        "functional": "module_peak_temperature_response",
                        "module_id": module_id,
                        "units": model_scalars[model_name]["baseline"]["temperature_units"],
                        "target": target_delta,
                        "prediction": pred_delta,
                        "signed_error": pred_delta - target_delta,
                        "absolute_error": abs(pred_delta - target_delta),
                    }
                )
    return rows


def _base_arrays(case: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    records = {"baseline": case["baseline"]}
    if case.get("i_plus") is not None:
        records["i_plus"] = case["i_plus"]
    baseline_modules = case["baseline"].design.modules
    arrays["input__module_ids"] = np.asarray([module.module_id for module in baseline_modules], dtype=str)
    arrays["input__module_positions_units"] = np.asarray(["dataset coordinate units"], dtype=str)
    arrays["input__module_heating_units"] = np.asarray(["dataset heating units"], dtype=str)
    for role_name in CHANNELS:
        role = case["baseline"].output.roles[role_name]
        arrays[f"{role_name}__query_features"] = np.asarray(role.query_features)
        arrays[f"{role_name}__query_ids"] = np.asarray(role.query_ids, dtype=str)
        arrays[f"{role_name}__channel_names"] = np.asarray(role.channel_names, dtype=str)
        arrays[f"{role_name}__channel_units"] = np.asarray(role.channel_units, dtype=str)
        arrays[f"{role_name}__valid_mask_baseline"] = np.asarray(role.valid_mask, dtype=bool)
        arrays[f"{role_name}__quadrature_weights_baseline"] = np.asarray(role.quadrature_weights)
        if role.receiver_module_ids is not None:
            arrays[f"{role_name}__receiver_module_ids"] = np.asarray(role.receiver_module_ids, dtype=str)
        else:
            arrays[f"{role_name}__receiver_module_ids"] = np.full(role.values.shape[0], "", dtype=str)
        arrays[f"{role_name}__target_baseline"] = np.asarray(role.values)
        arrays[f"{role_name}__noise_floor_baseline"] = (
            np.asarray(role.noise_floor) if role.noise_floor is not None else np.full_like(role.values, np.nan)
        )
        if "i_plus" in records:
            next_role = records["i_plus"].output.roles[role_name]
            if role.query_ids != next_role.query_ids or role.receiver_module_ids != next_role.receiver_module_ids:
                raise ValueError(f"Baseline/i_plus receiver identities diverge for {case['case_id']}/{role_name}.")
            if not np.array_equal(role.query_features, next_role.query_features):
                raise ValueError(
                    f"Baseline/i_plus material/query coordinates diverge for {case['case_id']}/{role_name}."
                )
            if not np.array_equal(role.quadrature_weights, next_role.quadrature_weights):
                raise ValueError(f"Baseline/i_plus receiver weights diverge for {case['case_id']}/{role_name}.")
            arrays[f"{role_name}__query_features_i_plus"] = np.asarray(next_role.query_features)
            arrays[f"{role_name}__query_ids_i_plus"] = np.asarray(next_role.query_ids, dtype=str)
            if next_role.receiver_module_ids is not None:
                arrays[f"{role_name}__receiver_module_ids_i_plus"] = np.asarray(
                    next_role.receiver_module_ids, dtype=str
                )
            else:
                arrays[f"{role_name}__receiver_module_ids_i_plus"] = np.full(next_role.values.shape[0], "", dtype=str)
            arrays[f"{role_name}__valid_mask_i_plus"] = np.asarray(next_role.valid_mask, dtype=bool)
            arrays[f"{role_name}__quadrature_weights_i_plus"] = np.asarray(next_role.quadrature_weights)
            arrays[f"{role_name}__target_i_plus"] = np.asarray(next_role.values)
            arrays[f"{role_name}__target_finite_i_plus"] = np.asarray(next_role.values) - np.asarray(role.values)
            arrays[f"{role_name}__noise_floor_i_plus"] = (
                np.asarray(next_role.noise_floor)
                if next_role.noise_floor is not None
                else np.full_like(next_role.values, np.nan)
            )
            finite_block = case["stencil"].finite_change("i_plus", role_name)
            arrays[f"{role_name}__noise_floor_finite_i_plus"] = (
                np.asarray(finite_block.noise_floor)
                if finite_block.noise_floor is not None
                else np.full_like(role.values, np.nan)
            )
    for state, record in records.items():
        arrays[f"input__module_positions_{state}_native_xy"] = np.asarray(
            [module.position_xy for module in record.design.modules], dtype=np.float64
        )
        arrays[f"input__module_heating_{state}_native"] = np.asarray(
            [module.heating for module in record.design.modules], dtype=np.float64
        )
        arrays[f"input__module_active_{state}"] = np.asarray(
            [module.active for module in record.design.modules], dtype=bool
        )
    if "i_plus" in records:
        arrays["input__module_position_delta_i_plus_minus_baseline_native_xy"] = (
            arrays["input__module_positions_i_plus_native_xy"] - arrays["input__module_positions_baseline_native_xy"]
        )
        arrays["input__module_heating_delta_i_plus_minus_baseline_native"] = (
            arrays["input__module_heating_i_plus_native"] - arrays["input__module_heating_baseline_native"]
        )
    return arrays


def _array_statistics(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    measure = np.asarray(weights, dtype=np.float64).reshape(-1)
    selected = np.asarray(mask, dtype=bool).reshape(-1) & np.isfinite(data) & np.isfinite(measure) & (measure > 0.0)
    if not selected.any():
        return {"valid_entry_count": 0, "finite_entry_count": 0}
    normalized = measure[selected] / measure[selected].sum(dtype=np.float64)
    selected_values = data[selected]
    return {
        "valid_entry_count": int(np.asarray(mask, dtype=bool).sum()),
        "finite_entry_count": int(selected.sum()),
        "weighted_mean_native_units": float(np.sum(normalized * selected_values)),
        "weighted_rms_native_units": float(np.sqrt(np.sum(normalized * np.square(selected_values)))),
        "minimum_native_units": float(np.min(selected_values)),
        "maximum_native_units": float(np.max(selected_values)),
    }


def _key_profile_array_stats(
    case: Mapping[str, Any],
    predictions_by_model: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, Any]:
    fields = {"fluid_fields": "temperature", "interface": "q_normal"}
    result: dict[str, Any] = {}
    for role_name, channel in fields.items():
        baseline_role = case["baseline"].output.roles[role_name]
        column = baseline_role.channel_names.index(channel)
        target_base = np.asarray(baseline_role.values[:, column])
        base_mask = np.asarray(baseline_role.valid_mask[:, column], dtype=bool)
        base_weights = np.asarray(baseline_role.quadrature_weights, dtype=np.float64)
        result[f"{role_name}/{channel}"] = {"units": baseline_role.channel_units[column], "models": {}}
        for model_name, states in predictions_by_model.items():
            model_entry: dict[str, Any] = {
                "reference_baseline": _array_statistics(target_base, base_weights, base_mask),
                "prediction_baseline": _array_statistics(
                    states["baseline"][role_name][:, column], base_weights, base_mask
                ),
            }
            if case.get("i_plus") is not None:
                next_role = case["i_plus"].output.roles[role_name]
                next_column = next_role.channel_names.index(channel)
                target_next = np.asarray(next_role.values[:, next_column])
                next_mask = np.asarray(next_role.valid_mask[:, next_column], dtype=bool)
                next_weights = np.asarray(next_role.quadrature_weights, dtype=np.float64)
                common = base_mask & next_mask
                finite_weights = np.minimum(base_weights, next_weights)
                finite_target = target_next - target_base
                finite_prediction = (
                    states["i_plus"][role_name][:, next_column] - states["baseline"][role_name][:, column]
                )
                model_entry.update(
                    {
                        "reference_i_plus": _array_statistics(target_next, next_weights, next_mask),
                        "prediction_i_plus": _array_statistics(
                            states["i_plus"][role_name][:, next_column], next_weights, next_mask
                        ),
                        "reference_finite_i_plus": _array_statistics(finite_target, finite_weights, common),
                        "prediction_finite_i_plus": _array_statistics(finite_prediction, finite_weights, common),
                    }
                )
            result[f"{role_name}/{channel}"]["models"][model_name] = model_entry
    return result


def _run_model_state(
    operator: CountedOperator,
    model_name: str,
    case: Mapping[str, Any],
    state: str,
    record: Any,
    queries: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    operator.set_context(
        purpose="native_model_evaluation",
        model=model_name,
        case_id=case["case_id"],
        family_id=case["family_id"],
        split=case["split"],
        state=state,
        active_module_ids=list(record.output.active_module_ids),
        module_count=len(record.output.active_module_ids),
        query_batch_size=2048,
    )
    design = DesignInput.from_state(record.design, device=device)
    context = context_inputs(record.context)
    with torch.no_grad():
        prediction = operator(design, context, queries)
        scalars = reduce_native_thermal_quantities(
            prediction,
            design,
            queries,
            context,
            module_ids=tuple(module.module_id for module in record.design.modules),
            solid_valid_mask=record.output.roles["solid_temperature"].valid_mask,
        )
    return _to_numpy(prediction), {
        "pressure_drop": float(scalars.pressure_drop.detach().cpu()),
        "pressure_units": scalars.pressure_drop_units,
        "module_peak_temperature": {
            key: float(value.detach().cpu()) for key, value in scalars.module_peak_temperature.items()
        },
        "temperature_units": scalars.temperature_units,
    }


def _load_candidate_checkpoint(arm: str, expected_update: int) -> tuple[Path, dict[str, Any]]:
    receipt = OUTPUT_ROOT / arm / "checkpoints/latest.json"
    latest = json.loads(receipt.read_text(encoding="utf-8"))
    if int(latest["update"]) != expected_update:
        raise ValueError(f"{arm} latest checkpoint update is {latest['update']}, expected {expected_update}.")
    path = Path(latest["path"]).expanduser().resolve()
    payload = load_trusted_checkpoint(path, map_location="cpu")
    if int(payload.get("actual_optimizer_updates", -1)) != expected_update:
        raise ValueError(f"{arm} checkpoint payload does not contain exact update {expected_update}.")
    if float(payload.get("calibrated_loss_weights", {}).get("value", -1.0)) != (4.0 if arm == "value4" else 8.0):
        raise ValueError(f"{arm} checkpoint value weight is inconsistent with its arm.")
    return path, payload


def _summary_tables(
    out: Path,
    field_rows: list[dict[str, Any]],
    scalar_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> None:
    import csv

    fields = [
        "case_id",
        "split",
        "model",
        "role",
        "channel",
        "units",
        "quantity",
        "rmse_native_units",
        "target_rms_native_units",
        "signal_relative_rmse_unitless",
        "mean_absolute_error_native_units",
        "max_absolute_error_native_units",
        "p95_absolute_error_native_units",
        "mean_signed_error_native_units",
        "finite_fraction",
        "valid_entry_count",
        "finite_prediction_entry_count",
        "nonfinite_prediction_entry_count",
        "noise_floor_rms_native_units",
        "response_resolved_entry_count",
        "response_unresolved_entry_count",
        "response_resolution_status",
    ]
    with (out / "field_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in field_rows:
            writer.writerow({**{key: row.get(key) for key in fields[:7]}, **row["metrics"]})
    scalar_fields = [
        "case_id",
        "model",
        "state",
        "functional",
        "module_id",
        "units",
        "target",
        "prediction",
        "signed_error",
        "absolute_error",
    ]
    with (out / "scalar_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in scalar_fields} for row in scalar_rows)
    selected = [
        row
        for row in field_rows
        if row["role"] in {"fluid_fields", "interface", "solid_temperature"}
        and row["quantity"] in {"absolute_baseline", "finite_i_plus"}
    ]
    grouped_fields: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in selected:
        grouped_fields.setdefault((row["model"], row["quantity"], row["role"], row["channel"]), []).append(row)
    note = [
        "# Thermal value-weight recovery numeric note",
        "",
        "This evaluation uses stored response-atlas outputs and packed train references. It makes no physical solver calls.",
        "Field rows summarize paired family-level comparisons over every evaluated case for that quantity; absolute rows include the historical cases, while finite i_plus rows cover calibration families. Errors use native units and quadrature weights. Response-resolution counts use saved receiver-specific noise floors only when present; absent floors are marked unknown.",
        "Module peaks are maxima over the same queried material receivers used for each stored target. The attached noise floors and response limits come from existing solver evidence; they are not new CFD measurements or independent physical validation.",
        "",
        "| Model | Quantity | Field/channel | Mean relative RMSE | Worst relative RMSE | Worst case | Worst max absolute error | Finite/valid |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    for (model, quantity, role, channel), rows in sorted(grouped_fields.items()):
        valid_ratios = [
            (row["metrics"]["signal_relative_rmse_unitless"], row)
            for row in rows
            if row["metrics"]["signal_relative_rmse_unitless"] is not None
        ]
        worst_ratio, worst_row = max(
            valid_ratios, default=(None, rows[0]), key=lambda item: item[0] if item[0] is not None else -1.0
        )
        total_finite = sum(int(row["metrics"]["finite_prediction_entry_count"]) for row in rows)
        total_valid = sum(int(row["metrics"]["valid_entry_count"]) for row in rows)
        maximum_absolute = max(
            (
                float(row["metrics"]["max_absolute_error_native_units"])
                for row in rows
                if row["metrics"]["max_absolute_error_native_units"] is not None
            ),
            default=float("nan"),
        )
        mean_ratio = float(np.mean([ratio for ratio, _ in valid_ratios])) if valid_ratios else None
        note.append(
            f"| {model} | {quantity} | {role}/{channel} | {mean_ratio if mean_ratio is not None else 'n/a'} | "
            f"{worst_ratio if worst_ratio is not None else 'n/a'} | {worst_row['case_id']} | {maximum_absolute} | "
            f"{total_finite}/{total_valid} |"
        )
    scalar_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in scalar_rows:
        if row["functional"] not in {
            "pressure_drop",
            "pressure_drop_response",
            "module_peak_temperature",
            "module_peak_temperature_response",
        }:
            continue
        scalar_groups.setdefault((row["model"], row["functional"], row["state"]), []).append(row)
    note.extend(
        [
            "",
            "## Pressure and module-peak errors",
            "",
            "| Model | Functional | State | Mean absolute error | Worst absolute error | Worst case/module |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for (model, functional, state), rows in sorted(scalar_groups.items()):
        worst = max(rows, key=lambda row: float(row["absolute_error"]))
        label = worst["case_id"] + (f"/{worst['module_id']}" if worst.get("module_id") else "")
        note.append(
            f"| {model} | {functional} | {state} | {np.mean([row['absolute_error'] for row in rows])} | "
            f"{worst['absolute_error']} | {label} |"
        )
    resolution_statuses: dict[str, int] = {}
    for row in field_rows:
        if row["quantity"] != "finite_i_plus":
            continue
        status = str(row["metrics"].get("response_resolution_status", "unspecified"))
        resolution_statuses[status] = resolution_statuses.get(status, 0) + 1
    note.append("")
    note.append(
        "Per-receiver response-resolution status rows: "
        + ", ".join(f"{status}={count}" for status, count in sorted(resolution_statuses.items()))
        + "."
    )
    calibration_cases = [case for case in cases if case["kind"] == "calibration_response_atlas"]
    if calibration_cases:
        note.extend(
            [
                "",
                "## Stored response-solve limits",
                "",
                "These limits are copied from the saved atlas solver metadata. They describe the stored numerical solves and do not estimate a new per-receiver error floor.",
                "",
                "| Family | i_plus native perturbation scales | Convergence tolerance | Final delta infinity | Final relative L2 | Final step |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for case in calibration_cases:
            meta = case["metadata"]
            plus = meta.get("records", {}).get("i_plus", {})
            runtime = plus.get("runtime", {})
            note.append(
                f"| {case['family_id']} | {plus.get('physical_step_scales', {})} | "
                f"{runtime.get('convergence_tol')} | {runtime.get('final_delta_inf')} | "
                f"{runtime.get('final_delta_l2_rel')} | {runtime.get('final_step')} |"
            )
    note.extend(
        [
            "",
            "The complete machine-readable field and scalar tables are `field_metrics.csv`, `scalar_metrics.csv`, and `metrics.json`. Per-case JSON records include input module coordinates/heats and weighted reference/candidate profile statistics for fluid temperature and interface q_normal. Per-case `.npz` files retain receiver coordinates/IDs, target arrays, masks, quadrature weights, raw predictions, and finite i_plus profiles.",
            "",
        ]
    )
    (out / "numeric_note.md").write_text("\n".join(note), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("additional", "final"), required=True)
    parser.add_argument("--expected-update", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    args = parser.parse_args()
    expected = 1420 if args.stage == "additional" else 1540
    if args.expected_update != expected:
        raise ValueError(f"Stage {args.stage} requires exact paired update {expected}.")
    out = args.output_root.expanduser().resolve() / f"evaluation_{args.stage}_u{expected}"
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    started_at = utc_now()
    PROCESS_STATE.update(
        {"output_dir": out, "started_at_utc": started_at, "started_monotonic": start, "stage": args.stage}
    )
    write_json(
        out / "process_start.json",
        {
            "stage": args.stage,
            "expected_update": expected,
            "started_at_utc": started_at,
            "physical_gpu_uuid_required": GPU2_UUID,
            "requested_torch_device": "cuda:2 ordinary physical visibility",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise RuntimeError("Evaluation requires ordinary CUDA visibility with CUDA_VISIBLE_DEVICES unset.")
    gpu_uuid, gpu_name = driver._physical_gpu2_identity()
    if gpu_uuid != GPU2_UUID:
        raise RuntimeError(f"Assigned physical GPU2 UUID mismatch: {gpu_uuid}.")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 2:
        raise RuntimeError("Explicit cuda:2 is unavailable under ordinary visibility.")
    device = torch.device("cuda:2")
    torch.cuda.set_device(device)
    if torch.cuda.current_device() != 2:
        raise RuntimeError("Evaluation did not select explicit physical cuda:2.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    reference_model, reference_payload = load_model(driver.DEFAULT_REFERENCE, device)
    if int(reference_payload.get("epoch", reference_payload.get("current_epoch", -1))) != 4738:
        raise ValueError("Operational Run1804 reference must be the retained e4738 checkpoint.")
    if any(parameter.device != device for parameter in reference_model.parameters()):
        raise RuntimeError("Run1804 reference model contains parameters outside explicit cuda:2.")
    reference_model.eval()
    dataset_path = driver._resolve_dataset_path(reference_payload, None)
    dataset = driver.GlobalChannelThermalDataset(
        dataset_path,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    template = driver._make_input_template(dataset)
    historical_selection = select_value_recovery_historical_cases(
        dataset, requested=16, focus_module_counts=(3, 5, 7, 10)
    )

    family_ids = ADDITIONAL_FAMILIES if args.stage == "additional" else ALL_CALIBRATION_FAMILIES
    cases = [_make_calibration_case(family) for family in family_ids]
    if args.stage == "final":
        cases.extend(_make_historical_case(Path(dataset_path), row) for row in historical_selection)
    write_json(
        out / "cohort.json",
        {
            "stage": args.stage,
            "expected_update": expected,
            "calibration_families": list(family_ids),
            "historical_train_cases": list(historical_selection),
            "historical_case_count_evaluated": 16 if args.stage == "final" else 0,
            "metadata_strata_covered": sorted({int(row["module_count"]) for row in historical_selection}),
            "solver_calls": 0,
        },
    )

    # Capture the retained input-only route encoding for the frozen G controls.
    sample = cases[0]["stencil"]
    setup_native = DifferentiableThermalOperator(
        reference_model,
        template,
        dataset_config=reference_payload["train_config"]["dataset"],
        normalization_stats=reference_payload["global_normalization_stats"],
        query_batch_size=args.query_batch_size,
        capture_packet_inputs=True,
    )
    counters: dict[str, Any] = {}
    setup_operator = CountedOperator(setup_native, device, out, counters)
    setup_queries = role_queries_from_stencil(sample, device=device)
    setup_record = sample.baseline
    setup_operator.set_context(
        purpose="setup_capture",
        model="Run1804",
        case_id=sample.physical_family_id,
        family_id=setup_record.design.physical_family_id,
        split=setup_record.design.split.value,
        state="baseline",
        active_module_ids=list(setup_record.output.active_module_ids),
        module_count=len(setup_record.output.active_module_ids),
        query_batch_size=args.query_batch_size,
    )
    with torch.no_grad():
        setup_operator(
            DesignInput.from_state(setup_record.design, device=device),
            context_inputs(setup_record.context),
            setup_queries,
        )
    if (
        not isinstance(setup_native.last_packet_inputs, Mapping)
        or setup_native.last_packet_inputs.get("encoded") is None
    ):
        raise RuntimeError("Run1804 setup wrapper did not expose retained input-only route encoding.")
    encoded = setup_native.last_packet_inputs["encoded"]
    del setup_operator, setup_native
    torch.cuda.empty_cache()

    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    source_payload = load_trusted_checkpoint(SOURCE_CHECKPOINT, map_location="cpu")
    if int(source_payload.get("actual_optimizer_updates", -1)) != 1300:
        raise ValueError("G-u1300 control is not the exact retained source checkpoint.")
    model_specs: list[tuple[str, Path | None, dict[str, Any] | None, int]] = [
        ("Run1804", None, None, 4738),
        ("G_u1300", SOURCE_CHECKPOINT, source_payload, 1300),
    ]
    for arm in ("value4", "value8"):
        path, payload = _load_candidate_checkpoint(arm, expected)
        model_specs.append((arm, path, payload, expected))

    arrays_by_case = {case["case_id"]: _base_arrays(case) for case in cases}
    predictions_by_case: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {
        case["case_id"]: {} for case in cases
    }
    scalars_by_case: dict[str, dict[str, dict[str, Any]]] = {case["case_id"]: {} for case in cases}
    for model_name, checkpoint, payload, update in model_specs:
        route_model = None
        bundle = None
        if model_name == "Run1804":
            model = reference_model
        else:
            if payload is None or checkpoint is None:
                raise RuntimeError(f"Missing checkpoint payload for {model_name}.")
            model, route_model, bundle = driver._setup_bundle(
                "G",
                payload,
                reference_checkpoint=driver.DEFAULT_REFERENCE,
                encoded=encoded,
                extra_route=str(manifest["extra_route"]),
                device=device,
            )
        if any(parameter.device != device for parameter in model.parameters()):
            raise RuntimeError(f"{model_name} contains model parameters outside explicit cuda:2.")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        native = CountedOperator(
            DifferentiableThermalOperator(
                model,
                template,
                dataset_config=reference_payload["train_config"]["dataset"],
                normalization_stats=reference_payload["global_normalization_stats"],
                query_batch_size=args.query_batch_size,
            ),
            device,
            out,
            counters,
        )
        for case in cases:
            predictions: dict[str, dict[str, np.ndarray]] = {}
            scalar_states: dict[str, Any] = {}
            queries = role_queries_from_record(case["baseline"], device=device)
            state_records = {"baseline": case["baseline"]}
            if case["i_plus"] is not None:
                state_records["i_plus"] = case["i_plus"]
            for state, record in state_records.items():
                role_prediction, scalar = _run_model_state(native, model_name, case, state, record, queries, device)
                predictions[state] = role_prediction
                scalar_states[state] = scalar
                for role, value in role_prediction.items():
                    arrays_by_case[case["case_id"]][f"{role}__{model_name}__{state}"] = value
            predictions_by_case[case["case_id"]][model_name] = predictions
            scalars_by_case[case["case_id"]][model_name] = scalar_states
        del native
        if model_name != "Run1804":
            del model, route_model, bundle
        torch.cuda.empty_cache()

    field_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        case_predictions = predictions_by_case[case["case_id"]]
        states = ("baseline", "i_plus") if case["i_plus"] is not None else ("baseline",)
        field_rows.extend(_metric_block(case_predictions, case, states=states))
        case_scalars = _scalar_block(case, case_predictions, scalars_by_case[case["case_id"]], states=states)
        scalar_rows.extend(case_scalars)
        filename = f"{case['case_id'].replace(':', '_')}__receiver_predictions.npz"
        with (out / filename).open("wb") as stream:
            np.savez_compressed(stream, **arrays_by_case[case["case_id"]])
        case_metrics = {
            "case_id": case["case_id"],
            "family_id": case["family_id"],
            "split": case["split"],
            "kind": case["kind"],
            "metadata": case["metadata"],
            "input_design": _input_design_summary(case),
            "states": list(states),
            "field_metrics": [row for row in field_rows if row["case_id"] == case["case_id"]],
            "scalar_metrics": case_scalars,
            "key_profile_array_stats": _key_profile_array_stats(case, case_predictions),
            "receiver_array_file": filename,
        }
        write_json(out / f"{case['case_id'].replace(':', '_')}__metrics.json", case_metrics)
        case_summaries.append({"case_id": case["case_id"], "states": list(states), "receiver_array_file": filename})

    _summary_tables(out, field_rows, scalar_rows, cases)
    summary = {
        "stage": args.stage,
        "expected_update": expected,
        "started_at_utc": started_at,
        "stopped_at_utc": utc_now(),
        "elapsed_process_wall_seconds": time.monotonic() - start,
        "physical_gpu_uuid": gpu_uuid,
        "physical_gpu_name": gpu_name,
        "torch_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cases": case_summaries,
        "models": [
            {
                "model": name,
                "checkpoint": str(path) if path else str(driver.DEFAULT_REFERENCE),
                "update_or_epoch": update,
            }
            for name, path, _, update in model_specs
        ],
        "native_wrapper_call_counters_by_purpose_model": counters,
        "native_wrapper_call_attempts": sum(int(value["attempted"]) for value in counters.values()),
        "native_wrapper_call_successes": sum(int(value["succeeded"]) for value in counters.values()),
        "native_wrapper_call_failures": sum(int(value["failed"]) for value in counters.values()),
        "solver_calls": 0,
        "optimizer_calls": 0,
        "field_metric_row_count": len(field_rows),
        "scalar_metric_row_count": len(scalar_rows),
        "receiver_arrays": "Per-case NPZ retains raw target and model role arrays, query features/IDs, receiver module IDs, masks, quadrature weights, and absolute/finite i_plus profiles when present.",
    }
    write_json(out / "metrics.json", {"field_metrics": field_rows, "scalar_metrics": scalar_rows})
    write_json(out / "summary.json", summary)
    del reference_model, dataset
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if PROCESS_STATE:
            append_jsonl(
                Path(PROCESS_STATE["output_dir"]) / "process_failures.jsonl",
                {
                    "stage": PROCESS_STATE.get("stage"),
                    "started_at_utc": PROCESS_STATE.get("started_at_utc"),
                    "failed_at_utc": utc_now(),
                    "elapsed_process_wall_seconds": time.monotonic() - float(PROCESS_STATE["started_monotonic"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "native_wrapper_attempts": 0,
                    "solver_calls": 0,
                    "optimizer_calls": 0,
                },
            )
        raise
