"""Held-stencil evaluation with role, pressure, and solid-peak metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from channelthermal.interaction_evidence.response_dataset import ResponseBlock, ResponseStencil
from channelthermal.interaction_evidence.types import SolveRecord

from .algebra import MixedResponseSpec, predict_stencil
from .contracts import AbsoluteOperator, AbsolutePrediction, DesignInput, context_inputs, role_queries_from_record
from .losses import _design_like, _pressure_prediction
from .thermal import (
    NEAR_INTERFACE_DISTANCE,
    module_peak_temperatures_from_role,
    reduce_native_thermal_quantities,
    smooth_module_peak,
)


def _near_interface_mask(role_query: Any, design: Any, context: Mapping[str, Any]) -> np.ndarray:
    """Select Eulerian rows within 0.25 of an active module surface."""

    if role_query.coordinate_kind != "eulerian":
        raise ValueError("Near-interface fluid evaluation requires Eulerian receiver coordinates.")
    if role_query.query_features.shape[1] < 2:
        raise ValueError("Eulerian fluid queries need at least x/y coordinates.")
    try:
        radius = float(context["module_radius"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Near-interface fluid evaluation requires module_radius context.") from exc
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("module_radius must be positive and finite.")
    centers = [module.position_xy for module in design.modules if module.active]
    if not centers:
        raise ValueError("Near-interface fluid evaluation needs an active module.")
    query_xy = role_query.query_features[:, :2]
    center_xy = torch.as_tensor(centers, dtype=query_xy.dtype, device=query_xy.device)
    surface_distance = torch.linalg.vector_norm(
        query_xy[:, None, :] - center_xy[None, :, :], dim=-1
    ).min(dim=1).values - radius
    mask = (surface_distance >= 0.0) & (surface_distance <= NEAR_INTERFACE_DISTANCE)
    return mask.detach().cpu().numpy().astype(bool, copy=False)


@dataclass(frozen=True)
class ChannelMetric:
    family_id: str
    evidence_source: str
    kind: str
    role: str
    label: str
    channel: str
    units: str
    observed_count: int
    resolved_sign_count: int
    weighted_mae: float | None
    weighted_rmse: float | None
    reference_rms: float | None
    noise_floor_rms: float | None
    signal_relative_rmse: float | None
    noise_aware_relative_rmse: float | None
    resolved_sign_accuracy: float | None
    physical_resolution_status: str = "not_applicable"


@dataclass(frozen=True)
class PressureMetric:
    family_id: str
    evidence_source: str
    kind: str
    label: str
    units: str
    reference_value: float
    predicted_value: float
    absolute_error: float
    pressure_limit: float
    reference_feasible: bool | None
    predicted_feasible: bool | None
    false_feasible: bool | None


@dataclass(frozen=True)
class SolidPeakSummaryMetric:
    family_id: str
    evidence_source: str
    label: str
    reduction: str
    units: str
    module_count: int
    reference_value: float
    predicted_value: float
    absolute_error: float
    smooth_beta: float | None


@dataclass(frozen=True)
class SolidPeakMetric:
    family_id: str
    evidence_source: str
    label: str
    module_slot: int
    units: str
    reference_peak: float
    predicted_peak: float
    absolute_error: float


def _metric_rows(
    predicted: torch.Tensor,
    *,
    family_id: str,
    evidence_source: str,
    kind: str,
    role: str,
    label: str,
    channel_names: Sequence[str],
    channel_units: Sequence[str],
    reference: np.ndarray,
    observed_mask: np.ndarray,
    weights: np.ndarray,
    noise_floor: np.ndarray | None,
) -> list[ChannelMetric]:
    pred = predicted.detach().to(dtype=torch.float64).cpu().numpy()
    truth = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    if mask.ndim == 1:
        mask = np.broadcast_to(mask[:, None], truth.shape)
    measure = np.asarray(weights, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape or mask.shape != truth.shape or measure.shape != (truth.shape[0],):
        raise ValueError("Evaluation arrays are not aligned to one role block.")
    floor = None if noise_floor is None else np.asarray(noise_floor, dtype=np.float64)
    if floor is not None and floor.ndim == 1:
        floor = np.broadcast_to(floor[:, None], truth.shape)
    rows: list[ChannelMetric] = []
    for channel, (name, unit) in enumerate(zip(channel_names, channel_units)):
        valid = mask[:, channel] & (measure > 0.0)
        count = int(valid.sum())
        if not count:
            status = "unknown_no_noise_floor" if kind == "mixed" and floor is None else "unresolved"
            rows.append(ChannelMetric(
                family_id, evidence_source, kind, role, label, str(name), str(unit),
                0, 0, None, None, None, None, None, None, None, status,
            ))
            continue
        if not np.isfinite(pred[valid, channel]).all() or not np.isfinite(truth[valid, channel]).all():
            raise ValueError(f"Non-finite observed values in {kind}/{role}/{name}.")
        weight = measure[valid] / measure[valid].sum()
        error = pred[valid, channel] - truth[valid, channel]
        mae = float(np.sum(weight * np.abs(error)))
        rmse = float(np.sqrt(np.sum(weight * np.square(error))))
        signal_rms = float(np.sqrt(np.sum(weight * np.square(truth[valid, channel]))))
        signal_relative = rmse / signal_rms if signal_rms > np.finfo(np.float64).eps else None
        floor_rms = None
        noise_relative = None
        resolved_count = 0
        sign_accuracy = None
        resolution_status = "not_applicable"
        if floor is not None:
            floor_rms = float(np.sqrt(np.sum(weight * np.square(floor[valid, channel]))))
            denominator = max(signal_rms, floor_rms, np.finfo(np.float64).eps)
            noise_relative = rmse / denominator
            resolved = np.abs(truth[valid, channel]) > floor[valid, channel]
            resolved_count = int(resolved.sum())
            if kind == "mixed":
                resolution_status = "resolved_some" if resolved_count else "unresolved_vs_floor"
            if resolved_count:
                sign_accuracy = float(np.mean(
                    np.sign(pred[valid, channel][resolved]) == np.sign(truth[valid, channel][resolved])
                ))
        elif kind == "mixed":
            resolution_status = "unknown_no_noise_floor"
        rows.append(ChannelMetric(
            family_id, evidence_source, kind, role, label, str(name), str(unit), count,
            resolved_count, mae, rmse, signal_rms, floor_rms, signal_relative,
            noise_relative, sign_accuracy,
            resolution_status,
        ))
    return rows


def evaluate_stencil(
    operator: AbsoluteOperator,
    stencil: ResponseStencil,
    *,
    pressure_limit: float | Mapping[str, float],
    mixed_specs: Sequence[MixedResponseSpec] = (),
    smooth_peak_beta: float = 1.0,
    device: torch.device | str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate absolute states and responses from a callback-backed operator.

    The caller chooses calibration/development versus final-review data. This
    function does not select checkpoints or mutate the callback's model.
    """

    if not np.isfinite(smooth_peak_beta) or smooth_peak_beta <= 0.0:
        raise ValueError("smooth_peak_beta must be positive and finite.")
    predictions = predict_stencil(operator, stencil, device=device)
    family = stencil.physical_family_id
    if isinstance(pressure_limit, Mapping):
        if family not in pressure_limit:
            raise KeyError(f"No frozen pressure limit was supplied for family {family!r}.")
        family_pressure_limit = float(pressure_limit[family])
    else:
        family_pressure_limit = float(pressure_limit)
    if not np.isfinite(family_pressure_limit):
        raise ValueError("The frozen original pressure limit must be finite.")
    source = stencil.source.value
    rows: dict[str, list[dict[str, Any]]] = {
        "absolute_roles": [],
        "near_interface_fluid": [],
        "finite_roles": [],
        "mixed_roles": [],
        "pressure": [],
        "solid_peaks": [],
        "solid_peak_summary": [],
    }
    records = [("baseline", stencil.baseline), *stencil.variants.items()]
    for label, record in records:
        output = record.output
        if output is None:
            raise ValueError("Evaluation records must contain physical target outputs.")
        absolute = predictions.values[label]
        for role_name, target_role in output.roles.items():
            metric = _metric_rows(
                absolute.role_values[role_name],
                family_id=family,
                evidence_source=source,
                kind="absolute",
                role=role_name,
                label=label,
                channel_names=target_role.channel_names,
                channel_units=target_role.channel_units,
                reference=target_role.values,
                observed_mask=target_role.valid_mask,
                weights=target_role.quadrature_weights,
                noise_floor=target_role.noise_floor,
            )
            rows["absolute_roles"].extend(asdict(item) for item in metric)
            if role_name == "fluid_fields":
                near = _near_interface_mask(
                    predictions.role_queries[role_name], record.design, record.context.values
                )
                observed = np.asarray(target_role.valid_mask, dtype=bool)
                near_observed = observed & (near[:, None] if observed.ndim == 2 else near)
                near_metric = _metric_rows(
                    absolute.role_values[role_name],
                    family_id=family,
                    evidence_source=source,
                    kind="absolute",
                    role="fluid_fields_near_interface",
                    label=label,
                    channel_names=target_role.channel_names,
                    channel_units=target_role.channel_units,
                    reference=target_role.values,
                    observed_mask=near_observed,
                    weights=target_role.quadrature_weights,
                    noise_floor=target_role.noise_floor,
                )
                rows["near_interface_fluid"].extend(asdict(item) for item in near_metric)

        peak_by_slot = module_peak_temperatures_from_role(
            absolute.role_values["solid_temperature"],
            predictions.role_queries["solid_temperature"],
            _design_like(record, absolute.role_values["solid_temperature"]),
        )
        slot_by_id = {module.module_id: index for index, module in enumerate(record.design.modules)}
        peak_quantity = output.quantities.get("internal_temperature_max")
        peak_units = peak_quantity.units if peak_quantity is not None else output.units_metadata.get("temperature", "unknown")
        target_peaks = {slot_by_id[module_id]: float(value) for module_id, value in output.module_peak_temperature.items()}
        predicted_values = [float(value.detach().cpu()) for _, value in sorted(peak_by_slot.items())]
        target_values = [value for _, value in sorted(target_peaks.items())]
        if not target_values:
            raise ValueError("Solid peak evaluation requires active module labels.")
        true_reference_peak = max(target_values)
        true_predicted_peak = max(predicted_values)
        rows["solid_peak_summary"].append(asdict(SolidPeakSummaryMetric(
            family, source, label, "true_module_max", peak_units, len(target_values),
            true_reference_peak, true_predicted_peak,
            abs(true_predicted_peak - true_reference_peak), None,
        )))
        smooth_reference_peak = float(smooth_module_peak(
            {slot: torch.tensor(value, dtype=torch.float64) for slot, value in target_peaks.items()}, smooth_peak_beta
        ).item())
        smooth_predicted_peak = float(smooth_module_peak(peak_by_slot, smooth_peak_beta).detach().cpu())
        rows["solid_peak_summary"].append(asdict(SolidPeakSummaryMetric(
            family, source, label, "smooth_module_max", peak_units, len(target_values),
            smooth_reference_peak, smooth_predicted_peak,
            abs(smooth_predicted_peak - smooth_reference_peak), smooth_peak_beta,
        )))
        for module_id, reference_peak in output.module_peak_temperature.items():
            slot = slot_by_id[module_id]
            if slot not in peak_by_slot:
                raise ValueError(f"Prediction omitted active module peak at slot {slot}.")
            rows["solid_peaks"].append(asdict(SolidPeakMetric(
                family, source, label, slot, peak_units, float(reference_peak),
                float(peak_by_slot[slot].detach().cpu()),
                abs(float(peak_by_slot[slot].detach().cpu()) - float(reference_peak)),
            )))

        pressure_quantity = output.quantities["pressure_drop"]
        if pressure_quantity.resolved:
            value = float(_pressure_prediction(label, predictions, stencil).detach().cpu())
            reference = float(pressure_quantity.value)
            rows["pressure"].append(asdict(PressureMetric(
                family, source, "absolute", label, pressure_quantity.units,
                reference, value, abs(value - reference), family_pressure_limit,
                reference <= family_pressure_limit, value <= family_pressure_limit,
                bool(value <= family_pressure_limit and reference > family_pressure_limit),
            )))

    for label in stencil.variants:
        for role_name in stencil.baseline.output.roles:  # type: ignore[union-attr]
            block = stencil.finite_change(label, role_name)
            metric = _metric_rows(
                predictions.finite(label, role_name),
                family_id=family,
                evidence_source=source,
                kind="finite",
                role=role_name,
                label=label,
                channel_names=block.channel_names,
                channel_units=block.channel_units,
                reference=block.delta,
                observed_mask=block.valid_mask,
                weights=block.quadrature_weights,
                noise_floor=block.noise_floor,
            )
            rows["finite_roles"].extend(asdict(item) for item in metric)

        base_quantity = stencil.baseline.output.quantities["pressure_drop"]  # type: ignore[union-attr]
        trial_quantity = stencil.variants[label].output.quantities["pressure_drop"]  # type: ignore[union-attr]
        if base_quantity.resolved and trial_quantity.resolved:
            predicted_delta = float((
                _pressure_prediction(label, predictions, stencil)
                - _pressure_prediction("baseline", predictions, stencil)
            ).detach().cpu())
            true_delta = float(trial_quantity.value - base_quantity.value)
            rows["pressure"].append(asdict(PressureMetric(
                family, source, "finite", label, trial_quantity.units,
                true_delta, predicted_delta, abs(predicted_delta - true_delta),
                family_pressure_limit, None, None, None,
            )))

    for spec in mixed_specs:
        if not {spec.joint_variant, spec.first_variant, spec.second_variant}.issubset(stencil.variants):
            continue
        block: ResponseBlock = stencil.anchored_interaction(
            role=spec.role,
            joint_variant=spec.joint_variant,
            first_variant=spec.first_variant,
            second_variant=spec.second_variant,
            label=spec.label,
        )
        metric = _metric_rows(
            predictions.mixed(spec),
            family_id=family,
            evidence_source=source,
            kind="mixed",
            role=spec.role,
            label=spec.label,
            channel_names=block.channel_names,
            channel_units=block.channel_units,
            reference=block.delta,
            observed_mask=block.valid_mask,
            weights=block.quadrature_weights,
            noise_floor=block.noise_floor,
        )
        rows["mixed_roles"].extend(asdict(item) for item in metric)
    return rows


def evaluate_absolute_record(
    operator: AbsoluteOperator,
    record: SolveRecord,
    *,
    pressure_limit: float,
    device: torch.device | str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate one broad historical value case through target-free role inputs."""

    if record.output is None:
        raise ValueError("Historical value replay requires a stored physical record with outputs.")
    if not np.isfinite(pressure_limit):
        raise ValueError("The frozen original pressure limit must be finite.")
    queries = role_queries_from_record(record, device=device)
    design = DesignInput.from_state(record.design, device=device)
    prediction = operator(design, context_inputs(record.context), queries)
    if not isinstance(prediction, AbsolutePrediction):
        raise TypeError("Absolute operator callbacks must return AbsolutePrediction.")
    if set(prediction.role_values) != set(queries):
        raise ValueError("Historical replay predictions must match their role queries exactly.")
    family = record.design.physical_family_id
    source = record.source.value
    rows: dict[str, list[dict[str, Any]]] = {
        "absolute_roles": [],
        "near_interface_fluid": [],
        "pressure": [],
        "solid_peaks": [],
        "solid_peak_summary": [],
    }
    for role_name, target_role in record.output.roles.items():
        absolute_values = prediction.role_values[role_name]
        expected = (target_role.values.shape[0], len(target_role.channel_names))
        if tuple(absolute_values.shape) != expected:
            raise ValueError(
                f"Historical role {role_name!r} has prediction shape {tuple(absolute_values.shape)}, "
                f"expected {expected}."
            )
        metrics = _metric_rows(
            absolute_values,
            family_id=family,
            evidence_source=source,
            kind="absolute",
            role=role_name,
            label="historical_value",
            channel_names=target_role.channel_names,
            channel_units=target_role.channel_units,
            reference=target_role.values,
            observed_mask=target_role.valid_mask,
            weights=target_role.quadrature_weights,
            noise_floor=target_role.noise_floor,
        )
        rows["absolute_roles"].extend(asdict(metric) for metric in metrics)
        if role_name == "fluid_fields":
            near = _near_interface_mask(queries[role_name], record.design, record.context.values)
            observed = np.asarray(target_role.valid_mask, dtype=bool)
            near_observed = observed & (near[:, None] if observed.ndim == 2 else near)
            near_metrics = _metric_rows(
                absolute_values,
                family_id=family,
                evidence_source=source,
                kind="absolute",
                role="fluid_fields_near_interface",
                label="historical_value",
                channel_names=target_role.channel_names,
                channel_units=target_role.channel_units,
                reference=target_role.values,
                observed_mask=near_observed,
                weights=target_role.quadrature_weights,
                noise_floor=target_role.noise_floor,
            )
            rows["near_interface_fluid"].extend(asdict(metric) for metric in near_metrics)

    native_quantities = reduce_native_thermal_quantities(
        prediction,
        design,
        queries,
        record.context.values,
        module_ids=tuple(module.module_id for module in record.design.modules),
        solid_valid_mask=record.output.roles["solid_temperature"].valid_mask,
    )
    slot_by_id = {module.module_id: index for index, module in enumerate(record.design.modules)}
    target_peaks = {
        slot_by_id[module_id]: float(value)
        for module_id, value in record.output.module_peak_temperature.items()
    }
    solid_role = record.output.roles["solid_temperature"]
    temperature_channel = (
        solid_role.channel_names.index("temperature")
        if "temperature" in solid_role.channel_names else 0
    )
    peak_units = solid_role.channel_units[temperature_channel]
    reference_peak = max(target_peaks.values())
    predicted_peak = max(
        float(value.detach().cpu())
        for value in native_quantities.module_peak_temperature.values()
    )
    rows["solid_peak_summary"].append(asdict(SolidPeakSummaryMetric(
        family, source, "historical_value", "true_module_max", peak_units,
        len(target_peaks), reference_peak, predicted_peak, abs(predicted_peak - reference_peak), None,
    )))
    for module_id, reference_value in record.output.module_peak_temperature.items():
        slot = slot_by_id[module_id]
        predicted_value = float(
            native_quantities.module_peak_temperature[module_id].detach().cpu()
        )
        rows["solid_peaks"].append(asdict(SolidPeakMetric(
            family, source, "historical_value", slot, peak_units,
            float(reference_value), predicted_value, abs(predicted_value - float(reference_value)),
        )))

    pressure_quantity = record.output.quantities["pressure_drop"]
    if pressure_quantity.resolved:
        predicted_pressure = float(native_quantities.pressure_drop.detach().cpu())
        reference_pressure = float(pressure_quantity.value)
        rows["pressure"].append(asdict(PressureMetric(
            family, source, "absolute", "historical_value", pressure_quantity.units,
            reference_pressure, predicted_pressure, abs(predicted_pressure - reference_pressure),
            float(pressure_limit), reference_pressure <= pressure_limit,
            predicted_pressure <= pressure_limit,
            bool(predicted_pressure <= pressure_limit and reference_pressure > pressure_limit),
        )))
    return rows


__all__ = [
    "NEAR_INTERFACE_DISTANCE",
    "ChannelMetric",
    "PressureMetric",
    "SolidPeakMetric",
    "SolidPeakSummaryMetric",
    "evaluate_absolute_record",
    "evaluate_stencil",
]
