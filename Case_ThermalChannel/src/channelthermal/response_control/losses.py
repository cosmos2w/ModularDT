"""Role-separated value, finite-response, decision, and constraint losses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import torch
import torch.nn.functional as F

from channelthermal.interaction_evidence.response_dataset import ResponseStencil
from channelthermal.interaction_evidence.types import SolveRecord

from .algebra import MixedResponseSpec, StencilPredictions
from .contracts import (
    AbsoluteOperator,
    DesignInput,
    RoleQuery,
    context_inputs,
    role_queries_from_record,
)
from .thermal import (
    module_peak_temperatures_from_role,
    pressure_drop_from_field,
    smooth_module_peak,
)


def fixed_heat_material_peak_coverage(record: SolveRecord) -> Mapping[str, Mapping[str, object]]:
    """Verify stored per-ID peaks against every valid material receiver row."""

    if record.output is None:
        raise ValueError("Material peak coverage requires a solved physical record.")
    role = record.output.roles["solid_temperature"]
    if role.receiver_module_ids is None:
        raise ValueError("Material peak coverage needs an explicit receiver module ID per row.")
    try:
        temperature_channel = role.channel_names.index("temperature")
    except ValueError as exc:
        raise ValueError("Material peak coverage needs the temperature channel.") from exc
    valid = np.asarray(role.valid_mask, dtype=bool)
    if valid.ndim == 1:
        valid = np.broadcast_to(valid[:, None], role.values.shape)
    receiver_ids = np.asarray(role.receiver_module_ids, dtype=object)
    values = np.asarray(role.values)[:, temperature_channel]
    temperature_valid = valid[:, temperature_channel]
    if set(record.output.module_peak_temperature) != set(record.output.active_module_ids):
        raise ValueError("Material peak labels do not cover the active physical module IDs.")
    if set(receiver_ids) - set(record.output.active_module_ids):
        raise ValueError("Material queries refer to a module outside the active peak targets.")

    coverage: dict[str, Mapping[str, object]] = {}
    for module_id, target in record.output.module_peak_temperature.items():
        selected = (receiver_ids == module_id) & temperature_valid
        if not np.any(selected):
            raise ValueError(f"Material peak target {module_id!r} has no valid receiver rows.")
        measured_max = float(np.max(values[selected]))
        if measured_max != float(target):
            raise ValueError(
                f"Stored per-ID peak for {module_id!r} does not match the valid solid receiver maximum."
            )
        peak_rows = np.flatnonzero(selected & (values == measured_max))
        coverage[module_id] = MappingProxyType({
            "receiver_count": int(np.count_nonzero(receiver_ids == module_id)),
            "valid_receiver_count": int(np.count_nonzero(selected)),
            "peak_receiver_ids": tuple(role.query_ids[int(row)] for row in peak_rows),
            "target_peak": measured_max,
            "recomputed_peak": measured_max,
            "exact_target_match": True,
        })
    return MappingProxyType(coverage)


@dataclass(frozen=True)
class FixedHeatNullControl:
    """One verified train control that reallocates heat at fixed geometry."""

    control_id: str
    baseline: SolveRecord
    control: SolveRecord
    sampled_role_indices: Mapping[str, Sequence[int]] | None = None
    sample_weight_factors: Mapping[str, Sequence[float]] | None = None

    def __post_init__(self) -> None:
        if not self.control_id:
            raise ValueError("A fixed-heat control needs a stable record ID.")
        baseline = self.baseline
        control = self.control
        for record in (baseline, control):
            if record.output is None or record.status.value != "converged":
                raise ValueError("Fixed-heat controls need converged records with physical outputs.")
            if record.design.split.value != "train":
                raise ValueError("Fixed-heat null controls are restricted to train records.")
        if baseline.source is not control.source:
            raise ValueError("A fixed-heat control cannot mix evidence sources.")
        if baseline.design.anchor_id != control.design.anchor_id:
            raise ValueError("A fixed-heat control must retain its baseline anchor.")
        if baseline.design.physical_family_id != control.design.physical_family_id:
            raise ValueError("A fixed-heat control must retain its baseline physical family.")
        if baseline.design.active_module_ids != control.design.active_module_ids:
            raise ValueError("A fixed-heat control must preserve active module identities.")
        if tuple(module.module_id for module in baseline.design.modules) != tuple(
            module.module_id for module in control.design.modules
        ) or tuple(module.active for module in baseline.design.modules) != tuple(
            module.active for module in control.design.modules
        ):
            raise ValueError("A fixed-heat control must preserve ordered design slots.")
        if dict(baseline.context.values) != dict(control.context.values):
            raise ValueError("A fixed-heat control must retain its baseline operating context.")
        base_positions = np.asarray([module.position_xy for module in baseline.design.modules])
        control_positions = np.asarray([module.position_xy for module in control.design.modules])
        if base_positions.shape != control_positions.shape or not np.array_equal(base_positions, control_positions):
            raise ValueError("A fixed-heat null control requires bitwise fixed module geometry.")
        base_heating = np.asarray([module.heating for module in baseline.design.modules], dtype=np.float64)
        control_heating = np.asarray([module.heating for module in control.design.modules], dtype=np.float64)
        if np.array_equal(base_heating, control_heating) or not np.isclose(
            base_heating.sum(), control_heating.sum(), rtol=0.0, atol=1.0e-10
        ):
            raise ValueError("A fixed-heat control must change allocation while preserving total heat.")
        if set(baseline.output.roles) != set(control.output.roles):
            raise ValueError("Fixed-heat control role sets differ from their baseline.")
        for role_name in baseline.output.roles:
            base_role = baseline.output.roles[role_name]
            control_role = control.output.roles[role_name]
            if (
                base_role.channel_names != control_role.channel_names
                or base_role.channel_units != control_role.channel_units
                or base_role.coordinate_kind != control_role.coordinate_kind
                or base_role.query_ids != control_role.query_ids
                or base_role.query_features.shape != control_role.query_features.shape
                or not np.allclose(base_role.query_features, control_role.query_features, rtol=0.0, atol=1.0e-7)
            ):
                raise ValueError(f"Fixed-heat {role_name} role is not coordinate-aligned to its baseline.")
        fixed_heat_material_peak_coverage(baseline)
        fixed_heat_material_peak_coverage(control)
        solid_base = baseline.output.roles["solid_temperature"]
        solid_control = control.output.roles["solid_temperature"]
        if (
            solid_base.receiver_module_ids != solid_control.receiver_module_ids
            or not np.array_equal(solid_base.valid_mask, solid_control.valid_mask)
        ):
            raise ValueError("Fixed-geometry heat controls must retain material receiver validity by ID.")
        fluid_base = baseline.output.roles["fluid_fields"]
        fluid_control = control.output.roles["fluid_fields"]
        common = np.asarray(fluid_base.valid_mask, dtype=bool) & np.asarray(fluid_control.valid_mask, dtype=bool)
        if common.ndim == 1:
            common = np.broadcast_to(common[:, None], fluid_base.values.shape)
        for channel_name in ("u", "v", "p"):
            channel = fluid_base.channel_names.index(channel_name)
            difference = fluid_control.values[:, channel] - fluid_base.values[:, channel]
            if np.any(difference[common[:, channel]] != 0.0):
                raise ValueError(
                    f"Verified fixed-geometry heat control {self.control_id!r} has nonzero {channel_name} target change."
                )
        base_pressure = baseline.output.quantities["pressure_drop"]
        control_pressure = control.output.quantities["pressure_drop"]
        if not base_pressure.resolved or not control_pressure.resolved or float(base_pressure.value) != float(control_pressure.value):
            raise ValueError("Fixed-heat controls require an exact resolved pressure-drop null target.")
        preserved_thermal_signal = False
        for role_name, channel_names in {
            "fluid_fields": ("temperature",),
            "interface": ("T_surface", "q_normal"),
            "solid_temperature": ("temperature",),
        }.items():
            base_role = baseline.output.roles[role_name]
            control_role = control.output.roles[role_name]
            common_role = np.asarray(base_role.valid_mask, dtype=bool) & np.asarray(
                control_role.valid_mask, dtype=bool
            )
            if common_role.ndim == 1:
                common_role = np.broadcast_to(common_role[:, None], base_role.values.shape)
            for channel_name in channel_names:
                channel = base_role.channel_names.index(channel_name)
                delta = control_role.values[:, channel] - base_role.values[:, channel]
                if np.any(delta[common_role[:, channel]] != 0.0):
                    preserved_thermal_signal = True
        if not preserved_thermal_signal:
            raise ValueError(f"Fixed-heat control {self.control_id!r} lost its measured thermal response.")
        if self.sampled_role_indices is not None:
            indices = {str(role): tuple(int(index) for index in values) for role, values in self.sampled_role_indices.items()}
            if set(indices) != set(baseline.output.roles):
                raise ValueError("Sampled fixed-heat receiver indices must cover every physical role.")
            factors = self.sample_weight_factors or {}
            if set(factors) != set(indices):
                raise ValueError("Fixed-heat sample factors must cover every sampled role.")
            normalized_factors = {
                str(role): tuple(float(value) for value in values)
                for role, values in factors.items()
            }
            for role, role_indices in indices.items():
                factor = normalized_factors[role]
                if len(role_indices) != len(factor) or any(value < 0.0 or not np.isfinite(value) for value in factor):
                    raise ValueError("Fixed-heat sample factors must be finite and align to sampled indices.")
                if any(index < 0 or index >= baseline.output.roles[role].query_features.shape[0] for index in role_indices):
                    raise ValueError("Fixed-heat sample index falls outside its stored role.")
            object.__setattr__(self, "sampled_role_indices", MappingProxyType(indices))
            object.__setattr__(self, "sample_weight_factors", MappingProxyType(normalized_factors))

    @property
    def family_id(self) -> str:
        return self.baseline.design.physical_family_id

    def sampled_for_panel(self, sampled_baseline: SolveRecord) -> FixedHeatNullControl:
        """Align this stored control to the family's deterministic sampled queries."""

        if sampled_baseline.output is None or sampled_baseline.design.anchor_id != self.baseline.design.anchor_id:
            raise ValueError("Sampled control adapter must use this train family's sampled baseline.")
        indices_by_role: dict[str, tuple[int, ...]] = {}
        factors_by_role: dict[str, tuple[float, ...]] = {}
        for role_name, full_base in self.baseline.output.roles.items():
            sampled_role = sampled_baseline.output.roles[role_name]
            control_role = self.control.output.roles[role_name]  # type: ignore[union-attr]
            if sampled_role.channel_names != full_base.channel_names or sampled_role.query_ids is None:
                raise ValueError(f"Sampled train role {role_name!r} has a different physical schema.")
            full_lookup = {query_id: index for index, query_id in enumerate(full_base.query_ids)}
            control_lookup = {query_id: index for index, query_id in enumerate(control_role.query_ids)}
            requested_ids = (
                control_role.query_ids
                if role_name == "solid_temperature"
                else sampled_role.query_ids
            )
            if role_name == "solid_temperature" and (
                tuple(requested_ids) != tuple(full_base.query_ids)
                or tuple(requested_ids) != tuple(control_role.query_ids)
            ):
                raise ValueError("Fixed-heat peak supervision requires the full maintained material receiver universe.")
            try:
                base_indices = tuple(full_lookup[query_id] for query_id in requested_ids)
                control_indices = tuple(control_lookup[query_id] for query_id in requested_ids)
            except KeyError as exc:
                raise ValueError(f"Sampled {role_name} receiver is missing from a fixed-heat record.") from exc
            if base_indices != control_indices:
                raise ValueError("Fixed-heat sample adapter found mismatched receiver ordering.")
            expected_features = (
                full_base.query_features[np.asarray(base_indices)]
                if role_name == "solid_temperature"
                else sampled_role.query_features
            )
            if not np.allclose(expected_features, control_role.query_features[np.asarray(control_indices)], rtol=0.0, atol=1.0e-7):
                raise ValueError("Sampled fixed-heat coordinates do not match the response panel.")
            if role_name == "solid_temperature":
                factor = np.ones((len(base_indices),), dtype=np.float64)
            else:
                base_weight = np.asarray(full_base.quadrature_weights)[np.asarray(base_indices)]
                sampled_weight = np.asarray(sampled_role.quadrature_weights)
                factor = np.divide(
                    sampled_weight,
                    base_weight,
                    out=np.ones_like(sampled_weight, dtype=np.float64),
                    where=base_weight > 0.0,
                )
            indices_by_role[role_name] = control_indices
            factors_by_role[role_name] = tuple(float(value) for value in factor)
        return replace(
            self,
            sampled_role_indices=indices_by_role,
            sample_weight_factors=factors_by_role,
        )


def _scale_vector(
    scales: Mapping[str, float | Sequence[float]],
    role: str,
    channels: int,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    if role not in scales:
        raise KeyError(f"No fixed training normalizer was configured for role {role!r}.")
    values = torch.as_tensor(scales[role], dtype=like.dtype, device=like.device).reshape(-1)
    if values.numel() == 1:
        values = values.expand(channels)
    if values.shape != (channels,) or not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError(f"Training normalizer for role {role!r} must be positive and have {channels} channels.")
    return values


def weighted_masked_mse(
    prediction: torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    valid_mask: np.ndarray | torch.Tensor,
    quadrature_weights: np.ndarray | torch.Tensor,
    scales: float | Sequence[float] | torch.Tensor,
) -> torch.Tensor | None:
    """Return equal-channel weighted MSE in fixed physical training scales.

    ``None`` means the block has no observed positive-weight support. Missing
    and unresolved receiver values are never converted to zero targets.
    """

    if prediction.ndim != 2:
        raise ValueError("Role predictions must have shape [N,C].")
    def tensor_input(value: np.ndarray | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype, device=prediction.device)
        # Typed evidence owns read-only NumPy arrays. Copy them instead of
        # creating Torch tensors over immutable backing memory.
        return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=prediction.device)

    truth = tensor_input(target, prediction.dtype)
    mask = tensor_input(valid_mask, torch.bool)
    weights = tensor_input(quadrature_weights, prediction.dtype).reshape(-1)
    if truth.shape != prediction.shape or mask.shape not in {(prediction.shape[0],), prediction.shape}:
        raise ValueError("Prediction, target, and mask shapes do not align.")
    if mask.ndim == 1:
        mask = mask[:, None].expand_as(prediction)
    if weights.shape != (prediction.shape[0],) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Quadrature weights must be finite, nonnegative, and length N.")
    mask = mask & (weights[:, None] > 0)
    if not bool(mask.any()):
        return None
    if not bool(torch.isfinite(prediction[mask]).all()) or not bool(torch.isfinite(truth[mask]).all()):
        raise ValueError("Observed prediction and target values must be finite.")
    scale = torch.as_tensor(scales, dtype=prediction.dtype, device=prediction.device).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand(prediction.shape[1])
    if scale.shape != (prediction.shape[1],) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("Loss scales must be positive and scalar or channel-shaped.")
    safe_truth = torch.where(mask, truth, torch.zeros_like(truth))
    safe_prediction = torch.where(mask, prediction, torch.zeros_like(prediction))
    residual = (safe_prediction - safe_truth) / scale[None, :]
    weighted_mask = mask.to(prediction.dtype) * weights[:, None]
    denominator = weighted_mask.sum(dim=0)
    by_channel = (residual.square() * weighted_mask).sum(dim=0) / denominator.clamp_min(torch.finfo(prediction.dtype).tiny)
    supported = denominator > 0
    return by_channel[supported].mean()


def historical_absolute_value_loss(
    operator: AbsoluteOperator,
    record: SolveRecord,
    *,
    scales: ThermalLossScales,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Supervise one broad train case through target-free native role queries."""

    if record.output is None or record.design.split.value != "train":
        raise ValueError("Historical value loss requires a solved train record.")
    design = DesignInput.from_state(record.design, device=device)
    queries = role_queries_from_record(record, device=device)
    prediction = operator(design, context_inputs(record.context), queries)
    role_losses: list[torch.Tensor] = []
    for name, target in record.output.roles.items():
        predicted = prediction.role_values[name]
        role_loss = weighted_masked_mse(
            predicted,
            target.values,
            valid_mask=target.valid_mask,
            quadrature_weights=target.quadrature_weights,
            scales=_scale_vector(scales.value, name, len(target.channel_names), like=predicted),
        )
        if role_loss is not None:
            role_losses.append(role_loss)
    if not role_losses:
        raise ValueError("Historical train record has no observed value support.")
    return torch.stack(role_losses).mean()


def _subset_role_query(query: RoleQuery, indices: Sequence[int]) -> RoleQuery:
    index = torch.as_tensor(indices, dtype=torch.long, device=query.query_features.device)
    receiver_slots = (
        None
        if query.receiver_slots is None
        else tuple(query.receiver_slots[int(row)] for row in indices)
    )
    return replace(
        query,
        query_features=query.query_features.index_select(0, index),
        receiver_slots=receiver_slots,
    )


def fixed_heat_control_loss_terms(
    operator: AbsoluteOperator,
    control: FixedHeatNullControl,
    *,
    scales: ThermalLossScales,
    device: torch.device | str | None = None,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    """Return exact flow-null and observed nonzero thermal heat-transfer losses.

    The stored control's query IDs define the comparison rows. Atlas-to-control
    quadrature differences are preserved by multiplying each stored control
    weight by the family's deterministic inverse-inclusion sample factor.
    """

    baseline = control.baseline
    trial = control.control
    if baseline.output is None or trial.output is None:
        raise ValueError("Fixed-heat loss requires two converged physical records.")
    full_queries = role_queries_from_record(trial, device=device)
    role_indices = control.sampled_role_indices or {
        name: tuple(range(query.query_features.shape[0]))
        for name, query in full_queries.items()
    }
    sample_factors = control.sample_weight_factors or {
        name: tuple(1.0 for _ in indices) for name, indices in role_indices.items()
    }
    queries = {
        name: _subset_role_query(query, role_indices[name])
        for name, query in full_queries.items()
    }
    baseline_design = DesignInput.from_state(baseline.design, device=device)
    trial_design = DesignInput.from_state(trial.design, device=device)
    context = context_inputs(trial.context)
    baseline_prediction = operator(baseline_design, context, queries)
    trial_prediction = operator(trial_design, context, queries)

    fluid_query = queries["fluid_fields"]
    base_fluid = baseline.output.roles["fluid_fields"]
    trial_fluid = trial.output.roles["fluid_fields"]
    fluid_indices = np.asarray(role_indices["fluid_fields"], dtype=np.int64)
    fluid_factors = np.asarray(sample_factors["fluid_fields"], dtype=np.float64)
    fluid_weights = np.asarray(trial_fluid.quadrature_weights)[fluid_indices] * fluid_factors
    common_fluid = (
        np.asarray(base_fluid.valid_mask, dtype=bool)[fluid_indices]
        & np.asarray(trial_fluid.valid_mask, dtype=bool)[fluid_indices]
    )
    if common_fluid.ndim == 1:
        common_fluid = np.broadcast_to(common_fluid[:, None], base_fluid.values[fluid_indices].shape)
    channel_indices = [fluid_query.channel_names.index(name) for name in ("u", "v", "p")]
    target_null = np.asarray(trial_fluid.values)[fluid_indices][:, channel_indices] - np.asarray(
        base_fluid.values
    )[fluid_indices][:, channel_indices]
    if np.any(target_null[common_fluid[:, channel_indices]] != 0.0):
        raise ValueError(f"Fixed-heat control {control.control_id!r} no longer has exact u/v/p null labels.")
    fluid_scale = _scale_vector(
        scales.finite,
        "fluid_fields",
        len(fluid_query.channel_names),
        like=trial_prediction.role_values["fluid_fields"],
    )[channel_indices]
    field_null = weighted_masked_mse(
        trial_prediction.role_values["fluid_fields"][:, channel_indices]
        - baseline_prediction.role_values["fluid_fields"][:, channel_indices],
        target_null,
        valid_mask=common_fluid[:, channel_indices],
        quadrature_weights=fluid_weights,
        scales=fluid_scale,
    )
    if field_null is None:
        raise ValueError("Fixed-heat flow-null control has no sampled fluid support.")
    base_pressure = pressure_drop_from_field(
        baseline_prediction.role_values["fluid_fields"],
        fluid_query,
        baseline_design,
        baseline.context.values,
    )
    trial_pressure = pressure_drop_from_field(
        trial_prediction.role_values["fluid_fields"],
        fluid_query,
        trial_design,
        trial.context.values,
    )
    pressure_target_delta = float(trial.output.quantities["pressure_drop"].value) - float(
        baseline.output.quantities["pressure_drop"].value
    )
    if pressure_target_delta != 0.0:
        raise ValueError("The fixed-heat pressure-drop null target must be exactly zero.")
    pressure_null = (trial_pressure - base_pressure) / float(
        scales.pressure_response
    )
    null_loss = torch.stack((field_null, pressure_null.square())).mean()

    thermal_role_losses: list[torch.Tensor] = []
    thermal_diagnostics: dict[str, torch.Tensor] = {
        "fixed_heat_null/velocity_pressure_fields": field_null.detach(),
        "fixed_heat_null/pressure_drop": pressure_null.square().detach(),
    }
    thermal_channels = {
        "fluid_fields": ("temperature",),
        "interface": ("T_surface", "q_normal"),
        "solid_temperature": ("temperature",),
    }
    for role_name, channel_names in thermal_channels.items():
        base_role = baseline.output.roles[role_name]
        trial_role = trial.output.roles[role_name]
        indices = np.asarray(role_indices[role_name], dtype=np.int64)
        factors = np.asarray(sample_factors[role_name], dtype=np.float64)
        weight = np.asarray(trial_role.quadrature_weights)[indices] * factors
        common = (
            np.asarray(base_role.valid_mask, dtype=bool)[indices]
            & np.asarray(trial_role.valid_mask, dtype=bool)[indices]
        )
        if common.ndim == 1:
            common = np.broadcast_to(common[:, None], base_role.values[indices].shape)
        model_base = baseline_prediction.role_values[role_name]
        model_trial = trial_prediction.role_values[role_name]
        names = queries[role_name].channel_names
        selected = [names.index(name) for name in channel_names]
        target_delta = (
            np.asarray(trial_role.values)[indices][:, selected]
            - np.asarray(base_role.values)[indices][:, selected]
        )
        channel_scales = _scale_vector(
            scales.finite, role_name, len(names), like=model_trial
        )[selected]
        role_loss = weighted_masked_mse(
            model_trial[:, selected] - model_base[:, selected],
            target_delta,
            valid_mask=common[:, selected],
            quadrature_weights=weight,
            scales=channel_scales,
        )
        if role_loss is None:
            raise ValueError(f"Fixed-heat thermal control has no observed {role_name} support.")
        thermal_role_losses.append(role_loss)
        thermal_diagnostics[f"fixed_heat_thermal/{role_name}"] = role_loss.detach()

    solid_query = queries["solid_temperature"]
    base_solid = baseline_prediction.role_values["solid_temperature"]
    trial_solid = trial_prediction.role_values["solid_temperature"]
    solid_indices = np.asarray(role_indices["solid_temperature"], dtype=np.int64)
    solid_role = trial.output.roles["solid_temperature"]
    selected_solid_ids = tuple(solid_role.query_ids[int(index)] for index in solid_indices)
    if selected_solid_ids != tuple(solid_role.query_ids):
        raise ValueError("Fixed-heat peak supervision must score the full material receiver universe in stored order.")
    solid_channel = solid_query.channel_names.index("temperature")
    base_solid_valid = np.asarray(
        baseline.output.roles["solid_temperature"].valid_mask, dtype=bool
    )[solid_indices]
    trial_solid_valid = np.asarray(solid_role.valid_mask, dtype=bool)[solid_indices]
    if base_solid_valid.ndim == 1:
        base_solid_valid = np.broadcast_to(
            base_solid_valid[:, None], baseline.output.roles["solid_temperature"].values[solid_indices].shape
        )
    if trial_solid_valid.ndim == 1:
        trial_solid_valid = np.broadcast_to(
            trial_solid_valid[:, None], solid_role.values[solid_indices].shape
        )
    slots = torch.as_tensor(solid_query.receiver_slots, dtype=torch.long, device=base_solid.device)
    base_valid_temperature = torch.as_tensor(
        base_solid_valid[:, solid_channel], dtype=torch.bool, device=base_solid.device
    )
    trial_valid_temperature = torch.as_tensor(
        trial_solid_valid[:, solid_channel], dtype=torch.bool, device=trial_solid.device
    )
    active_slots = torch.nonzero(baseline_design.module_present.bool(), as_tuple=False).reshape(-1)
    predicted_base_peaks: dict[int, torch.Tensor] = {}
    predicted_trial_peaks: dict[int, torch.Tensor] = {}
    for slot_tensor in active_slots:
        slot = int(slot_tensor.item())
        base_rows = (slots == slot) & base_valid_temperature
        trial_rows = (slots == slot) & trial_valid_temperature
        if not bool(base_rows.any()) or not bool(trial_rows.any()):
            raise ValueError(f"Fixed-heat control has no valid material rows for module slot {slot}.")
        predicted_base_peaks[slot] = base_solid[base_rows, solid_channel].max()
        predicted_trial_peaks[slot] = trial_solid[trial_rows, solid_channel].max()
    base_module_ids = [module.module_id for module in baseline.design.modules]
    base_target_by_id = dict(baseline.output.module_peak_temperature)
    trial_target_by_id = dict(trial.output.module_peak_temperature)
    common_slots = [
        slot for slot, module_id in enumerate(base_module_ids)
        if slot in predicted_base_peaks
        and module_id in base_target_by_id
        and module_id in trial_target_by_id
    ]
    if not common_slots:
        raise ValueError("Fixed-heat control has no common per-module peak targets.")
    predicted_peak_delta = torch.stack([
        predicted_trial_peaks[slot] - predicted_base_peaks[slot]
        for slot in common_slots
    ])
    target_peak_delta = predicted_peak_delta.new_tensor([
        trial_target_by_id[base_module_ids[slot]] - base_target_by_id[base_module_ids[slot]]
        for slot in common_slots
    ])
    normalized_peak_error = (predicted_peak_delta - target_peak_delta) / float(scales.solid_temperature)
    per_id_peak_loss = normalized_peak_error.square().mean()
    selected_base_peaks = torch.stack([predicted_base_peaks[slot] for slot in common_slots])
    selected_trial_peaks = torch.stack([predicted_trial_peaks[slot] for slot in common_slots])
    beta = float(scales.smooth_peak_beta)
    smooth_predicted_delta = (
        smooth_module_peak(dict(enumerate(selected_trial_peaks)), beta)
        - smooth_module_peak(dict(enumerate(selected_base_peaks)), beta)
    )
    target_base_peaks = target_peak_delta.new_tensor([
        base_target_by_id[base_module_ids[slot]] for slot in common_slots
    ])
    target_trial_peaks = target_peak_delta.new_tensor([
        trial_target_by_id[base_module_ids[slot]] for slot in common_slots
    ])
    smooth_target_delta = (
        smooth_module_peak(dict(enumerate(target_trial_peaks)), beta)
        - smooth_module_peak(dict(enumerate(target_base_peaks)), beta)
    )
    smooth_peak_loss = ((smooth_predicted_delta - smooth_target_delta) / float(scales.solid_temperature)).square()
    hard_predicted_delta = selected_trial_peaks.max() - selected_base_peaks.max()
    hard_target_delta = target_trial_peaks.max() - target_base_peaks.max()
    hard_peak_loss = ((hard_predicted_delta - hard_target_delta) / float(scales.solid_temperature)).square()
    peak_loss = per_id_peak_loss + smooth_peak_loss + hard_peak_loss
    thermal_role_losses.append(peak_loss)
    thermal_diagnostics["fixed_heat_thermal/peak_per_id"] = per_id_peak_loss.detach()
    thermal_diagnostics["fixed_heat_thermal/peak_smooth_global"] = smooth_peak_loss.detach()
    thermal_diagnostics["fixed_heat_thermal/peak_hard_global"] = hard_peak_loss.detach()
    for index, slot in enumerate(common_slots):
        thermal_diagnostics[
            f"fixed_heat_thermal/peak/{base_module_ids[slot]}"
        ] = normalized_peak_error[index].square().detach()

    thermal_loss = torch.stack(thermal_role_losses).mean()
    return (
        MappingProxyType({"fixed_heat_null": null_loss, "fixed_heat_thermal": thermal_loss}),
        MappingProxyType(thermal_diagnostics),
    )


def _mean_or_zero(values: list[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else like.new_zeros(())


def _design_like(record: SolveRecord, like: torch.Tensor) -> DesignInput:
    return DesignInput.from_state(record.design, dtype=like.dtype, device=like.device)


def _pressure_prediction(
    record_label: str,
    predictions: StencilPredictions,
    stencil: ResponseStencil,
) -> torch.Tensor:
    record = stencil.baseline if record_label == "baseline" else stencil.variants[record_label]
    absolute = predictions.values[record_label]
    return pressure_drop_from_field(
        absolute.role_values["fluid_fields"],
        predictions.role_queries["fluid_fields"],
        _design_like(record, absolute.role_values["fluid_fields"]),
        record.context.values,
    )


@dataclass(frozen=True)
class ThermalLossScales:
    """Fixed, training-derived physical scales and the original pressure limit."""

    value: Mapping[str, float | Sequence[float]]
    finite: Mapping[str, float | Sequence[float]]
    mixed: Mapping[str, float | Sequence[float]]
    pressure_value: float
    pressure_response: float
    pressure_limit: float
    pressure_boundary: float
    solid_temperature: float
    smooth_peak_beta: float
    near_limit_band: float
    near_limit_multiplier: float = 2.0
    pressure_limit_by_family: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        for name in ("value", "finite", "mixed"):
            mapping = {
                str(role): tuple(float(item) for item in np.asarray(scale).reshape(-1))
                for role, scale in getattr(self, name).items()
            }
            if any(not values or not np.isfinite(values).all() or np.any(np.asarray(values) <= 0) for values in mapping.values()):
                raise ValueError(f"{name} scales must be positive and finite.")
            object.__setattr__(self, name, MappingProxyType(mapping))
        scalar_values = (
            self.pressure_value,
            self.pressure_response,
            self.pressure_boundary,
            self.solid_temperature,
            self.smooth_peak_beta,
            self.near_limit_band,
            self.near_limit_multiplier,
        )
        if not np.isfinite(scalar_values).all() or any(value <= 0.0 for value in scalar_values):
            raise ValueError("Physical loss scales, peak beta, and near-limit weights must be positive and finite.")
        if not np.isfinite(self.pressure_limit):
            raise ValueError("The original pressure limit must be finite.")
        if self.pressure_limit_by_family is not None:
            limits = {str(key): float(value) for key, value in self.pressure_limit_by_family.items()}
            if not limits or not np.isfinite(list(limits.values())).all():
                raise ValueError("Per-family pressure limits must be nonempty and finite.")
            object.__setattr__(self, "pressure_limit_by_family", MappingProxyType(limits))

    def with_train_reference_limits(
        self,
        training_stencils: Sequence[ResponseStencil],
        *,
        limit_factor: float = 1.05,
    ) -> ThermalLossScales:
        """Freeze each train family's constraint from its original baseline."""

        if not np.isfinite(limit_factor) or limit_factor <= 0.0:
            raise ValueError("limit_factor must be positive and finite.")
        limits: dict[str, float] = {}
        for stencil in training_stencils:
            if stencil.split.value != "train" or stencil.baseline.output is None:
                raise ValueError("Family limits may be derived from train baseline records only.")
            pressure = stencil.baseline.output.quantities["pressure_drop"]
            if not pressure.resolved:
                raise ValueError(f"Train baseline pressure is unresolved for {stencil.physical_family_id!r}.")
            limits[stencil.physical_family_id] = float(pressure.value) * float(limit_factor)
        if not limits:
            raise ValueError("At least one train family is required to freeze pressure limits.")
        return replace(self, pressure_limit_by_family=limits)


@dataclass(frozen=True)
class StencilLossTerms:
    """Dimensionless loss terms before the calibrated outer multipliers."""

    terms: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", MappingProxyType(dict(self.terms)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def total(self, weights: Mapping[str, float]) -> torch.Tensor:
        if not weights:
            raise ValueError("At least one active loss weight is required.")
        missing = set(weights) - set(self.terms)
        if missing:
            raise KeyError(f"Requested loss terms were not computed: {sorted(missing)}.")
        reference = next(iter(self.terms.values()))
        active = []
        for name, weight in weights.items():
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(f"Loss weight {name!r} must be finite and nonnegative.")
            if weight:
                active.append(self.terms[name] * float(weight))
        if not active:
            raise ValueError("At least one active loss weight must be positive.")
        return torch.stack(active).sum() if active else reference.new_zeros(())


def compute_stencil_loss_terms(
    predictions: StencilPredictions,
    stencil: ResponseStencil,
    *,
    scales: ThermalLossScales,
    mixed_specs: Sequence[MixedResponseSpec] = (),
    enabled_terms: Sequence[str] | None = None,
    include_feasibility_bce: bool = True,
    supplemental_terms: Mapping[str, torch.Tensor] | None = None,
) -> StencilLossTerms:
    """Compute role, four-state, solid-peak, and pressure losses for a stencil."""

    if set(predictions.values) != {"baseline", *stencil.variants}:
        raise ValueError("Predictions must cover exactly the physical stencil states.")
    legacy_terms = {"value", "finite", "mixed", "decision", "constraint"}
    response_terms = {
        "finite_peak",
        "pressure_value",
        "pressure_response",
        "fixed_heat_null",
        "fixed_heat_thermal",
    }
    enabled = legacy_terms if enabled_terms is None else set(enabled_terms)
    unknown = enabled - legacy_terms - response_terms
    if unknown:
        raise ValueError(f"Unknown response-loss terms: {sorted(unknown)}.")
    value_losses: list[torch.Tensor] = []
    finite_losses: list[torch.Tensor] = []
    finite_peak_losses: list[torch.Tensor] = []
    mixed_losses: list[torch.Tensor] = []
    decision_losses: list[torch.Tensor] = []
    pressure_value_losses: list[torch.Tensor] = []
    pressure_delta_losses: list[torch.Tensor] = []
    feasibility_losses: list[torch.Tensor] = []
    feasibility_candidates: list[tuple[str, torch.Tensor, bool]] = []
    diagnostics: dict[str, torch.Tensor] = {}
    records: list[tuple[str, SolveRecord]] = [("baseline", stencil.baseline)]
    records.extend(stencil.variants.items())

    for label, record in records:
        if record.output is None:
            raise ValueError("Every training stencil state must have reference outputs.")
        absolute = predictions.values[label]
        if "value" in enabled:
            for role_name, target_role in record.output.roles.items():
                pred = absolute.role_values[role_name]
                scale = _scale_vector(scales.value, role_name, len(target_role.channel_names), like=pred)
                loss = weighted_masked_mse(
                    pred,
                    target_role.values,
                    valid_mask=target_role.valid_mask,
                    quadrature_weights=target_role.quadrature_weights,
                    scales=scale,
                )
                if loss is not None:
                    value_losses.append(loss)
                    diagnostics[f"value/{label}/{role_name}"] = loss.detach()

        if "decision" in enabled:
            solid = absolute.role_values["solid_temperature"]
            peaks = module_peak_temperatures_from_role(
                solid,
                predictions.role_queries["solid_temperature"],
                _design_like(record, solid),
            )
            target_by_slot = {
                slot: solid.new_tensor(value) for slot, value in _target_peak_slots(record).items()
            }
            common_slots = sorted(set(peaks) & set(target_by_slot))
            if not common_slots:
                raise ValueError("Predicted and reference solid-module peak sets do not overlap.")
            predicted_vector = torch.stack([peaks[slot] for slot in common_slots])
            target_vector = torch.stack([target_by_slot[slot] for slot in common_slots])
            normalized_peak = (predicted_vector - target_vector) / float(scales.solid_temperature)
            smooth_pred = _smooth_peak_tensor(predicted_vector, scales.smooth_peak_beta)
            smooth_target = _smooth_peak_tensor(target_vector, scales.smooth_peak_beta)
            true_peak_error = (predicted_vector.max() - target_vector.max()) / float(scales.solid_temperature)
            decision = normalized_peak.square().mean() + (
                (smooth_pred - smooth_target) / float(scales.solid_temperature)
            ).square() + true_peak_error.square()
            decision_losses.append(decision)
            diagnostics[f"decision/{label}/module_peak"] = normalized_peak.square().mean().detach()
            diagnostics[f"decision/{label}/smooth_peak"] = (
                (smooth_pred - smooth_target) / float(scales.solid_temperature)
            ).square().detach()
            diagnostics[f"decision/{label}/true_peak"] = true_peak_error.square().detach()

        if "constraint" in enabled or "pressure_value" in enabled:
            quantity = record.output.quantities["pressure_drop"]
            if quantity.resolved:
                predicted_drop = _pressure_prediction(label, predictions, stencil)
                target_drop = predicted_drop.new_tensor(float(quantity.value))
                normalized_error = (predicted_drop - target_drop) / float(scales.pressure_value)
                if scales.pressure_limit_by_family is None:
                    family_limit = float(scales.pressure_limit)
                else:
                    try:
                        family_limit = float(scales.pressure_limit_by_family[stencil.physical_family_id])
                    except KeyError as exc:
                        raise KeyError(
                            f"No frozen training pressure limit for family {stencil.physical_family_id!r}."
                        ) from exc
                distance = abs(float(quantity.value) - family_limit)
                near_weight = (
                    float(scales.near_limit_multiplier)
                    if distance <= float(scales.near_limit_band)
                    else 1.0
                )
                pressure_value_losses.append(near_weight * normalized_error.square())
                logit = (family_limit - predicted_drop) / float(scales.pressure_boundary)
                feasibility_candidates.append(
                    (label, logit, bool(quantity.value <= family_limit))
                )
                diagnostics[f"constraint/{label}/pressure_drop"] = normalized_error.square().detach()

    for variant in stencil.variants if enabled & {"finite", "finite_peak", "constraint", "pressure_response"} else ():
        record = stencil.variants[variant]
        if record.output is None or stencil.baseline.output is None:
            continue
        for role_name in record.output.roles if "finite" in enabled else ():
            block = stencil.finite_change(variant, role_name)
            pred = predictions.finite(variant, role_name)
            scale = _scale_vector(scales.finite, role_name, len(block.channel_names), like=pred)
            loss = weighted_masked_mse(
                pred,
                block.delta,
                valid_mask=block.valid_mask,
                quadrature_weights=block.quadrature_weights,
                scales=scale,
            )
            if loss is not None:
                finite_losses.append(loss)
                diagnostics[f"finite/{variant}/{role_name}"] = loss.detach()

        base_quantity = stencil.baseline.output.quantities["pressure_drop"]
        trial_quantity = record.output.quantities["pressure_drop"]
        if ("constraint" in enabled or "pressure_response" in enabled) and base_quantity.resolved and trial_quantity.resolved:
            base_pressure = _pressure_prediction("baseline", predictions, stencil)
            trial_pressure = _pressure_prediction(variant, predictions, stencil)
            target_delta = float(trial_quantity.value) - float(base_quantity.value)
            response = (trial_pressure - base_pressure - target_delta) / float(scales.pressure_response)
            pressure_delta_losses.append(response.square())
            diagnostics[f"constraint/{variant}/pressure_response"] = response.square().detach()

        if "finite_peak" in enabled:
            baseline_prediction = predictions.values["baseline"]
            baseline_record = stencil.baseline
            baseline_solid = baseline_prediction.role_values["solid_temperature"]
            trial_solid = predictions.values[variant].role_values["solid_temperature"]
            baseline_peaks = module_peak_temperatures_from_role(
                baseline_solid,
                predictions.role_queries["solid_temperature"],
                _design_like(baseline_record, baseline_solid),
            )
            trial_peaks = module_peak_temperatures_from_role(
                trial_solid,
                predictions.role_queries["solid_temperature"],
                _design_like(record, trial_solid),
            )
            baseline_targets = _target_peak_slots(baseline_record)
            trial_targets = _target_peak_slots(record)
            common_slots = sorted(
                set(baseline_peaks)
                & set(trial_peaks)
                & set(baseline_targets)
                & set(trial_targets)
            )
            if not common_slots:
                raise ValueError("Finite per-module peak supervision has no resolved common module IDs.")
            predicted_delta = torch.stack(
                [trial_peaks[slot] - baseline_peaks[slot] for slot in common_slots]
            )
            target_delta = predicted_delta.new_tensor(
                [trial_targets[slot] - baseline_targets[slot] for slot in common_slots]
            )
            normalized_error = (predicted_delta - target_delta) / float(scales.solid_temperature)
            per_module_loss = normalized_error.square().mean()
            smooth_predicted_delta = (
                _smooth_peak_tensor(torch.stack([trial_peaks[slot] for slot in common_slots]), scales.smooth_peak_beta)
                - _smooth_peak_tensor(torch.stack([baseline_peaks[slot] for slot in common_slots]), scales.smooth_peak_beta)
            )
            smooth_target_delta = (
                _smooth_peak_tensor(
                    target_delta + target_delta.new_tensor(
                        [baseline_targets[slot] for slot in common_slots]
                    ),
                    scales.smooth_peak_beta,
                )
                - _smooth_peak_tensor(
                    target_delta.new_tensor([baseline_targets[slot] for slot in common_slots]),
                    scales.smooth_peak_beta,
                )
            )
            true_predicted_delta = torch.stack([trial_peaks[slot] for slot in common_slots]).max() - torch.stack(
                [baseline_peaks[slot] for slot in common_slots]
            ).max()
            true_target_delta = max(trial_targets[slot] for slot in common_slots) - max(
                baseline_targets[slot] for slot in common_slots
            )
            smooth_error = (smooth_predicted_delta - smooth_target_delta) / float(scales.solid_temperature)
            true_error = (true_predicted_delta - float(true_target_delta)) / float(scales.solid_temperature)
            finite_peak_losses.append(per_module_loss + smooth_error.square() + true_error.square())
            diagnostics[f"finite_peak/{variant}/per_id"] = per_module_loss.detach()
            diagnostics[f"finite_peak/{variant}/smooth_global"] = smooth_error.square().detach()
            diagnostics[f"finite_peak/{variant}/true_global"] = true_error.square().detach()
            module_by_slot = {index: module for index, module in enumerate(record.design.modules)}
            for index, slot in enumerate(common_slots):
                module_id = module_by_slot[slot].module_id
                diagnostics[f"finite_peak/{variant}/{module_id}"] = normalized_error[index].square().detach()

    for spec in mixed_specs if "mixed" in enabled else ():
        if not {spec.joint_variant, spec.first_variant, spec.second_variant}.issubset(stencil.variants):
            # Missing corners mean this physical mixed response is unknown,
            # not a zero label and not an error for the rest of the batch.
            continue
        if spec.role not in predictions.role_queries:
            raise KeyError(f"Mixed response role {spec.role!r} is absent from the stencil.")
        block = stencil.anchored_interaction(
            role=spec.role,
            joint_variant=spec.joint_variant,
            first_variant=spec.first_variant,
            second_variant=spec.second_variant,
            label=spec.label,
        )
        # Mixed labels without an empirical per-role numerical floor cannot
        # establish physical resolution. Keep them unknown instead of
        # training on potentially solver-noise-scale values.
        resolved_mask = block.resolved_sign_mask
        if resolved_mask is None or not np.any(resolved_mask):
            continue
        pred = predictions.mixed(spec)
        scale = _scale_vector(scales.mixed, spec.role, len(block.channel_names), like=pred)
        loss = weighted_masked_mse(
            pred,
            block.delta,
            valid_mask=resolved_mask,
            quadrature_weights=block.quadrature_weights,
            scales=scale,
        )
        if loss is not None:
            mixed_losses.append(loss)
            diagnostics[f"mixed/{spec.label}/{spec.role}"] = loss.detach()

    feasibility_targets = [target for _, _, target in feasibility_candidates]
    if include_feasibility_bce and "constraint" in enabled and len(set(feasibility_targets)) > 1:
        for label, logit, target in feasibility_candidates:
            feasible = logit.new_tensor(float(target))
            loss = F.binary_cross_entropy_with_logits(logit, feasible)
            feasibility_losses.append(loss)
            diagnostics[f"constraint/{label}/feasibility_bce"] = loss.detach()
        diagnostics["constraint/feasibility_bce_skipped_single_class"] = predictions.values[
            "baseline"
        ].role_values["fluid_fields"].new_tensor(0.0)
    elif feasibility_targets:
        # A one-class stencil has no data-supported feasibility boundary.
        # Keep its continuous pressure value/response supervision, but do not
        # train a classifier toward an unsupported all-one/all-zero boundary.
        diagnostics["constraint/feasibility_bce_skipped_single_class"] = predictions.values[
            "baseline"
        ].role_values["fluid_fields"].new_tensor(1.0)

    like = predictions.values["baseline"].role_values["fluid_fields"]
    terms: dict[str, torch.Tensor] = {}
    for name, values in (
        ("value", value_losses),
        ("finite", finite_losses),
        ("finite_peak", finite_peak_losses),
        ("mixed", mixed_losses),
        ("decision", decision_losses),
    ):
        if name in enabled and values:
            terms[name] = torch.stack(values).mean()
    constraint_groups = {
        "pressure_value": pressure_value_losses,
        "pressure_response": pressure_delta_losses,
        "feasibility": feasibility_losses,
    }
    observed_constraint_components = []
    for name, values in constraint_groups.items():
        component = _mean_or_zero(values, like)
        diagnostics[f"constraint/{name}_mean"] = component.detach()
        if values:
            observed_constraint_components.append(component)
    if "pressure_value" in enabled and pressure_value_losses:
        terms["pressure_value"] = torch.stack(pressure_value_losses).mean()
    if "pressure_response" in enabled and pressure_delta_losses:
        terms["pressure_response"] = torch.stack(pressure_delta_losses).mean()
    if "constraint" in enabled and observed_constraint_components:
        terms["constraint"] = torch.stack(observed_constraint_components).mean()
    for name, value in (supplemental_terms or {}).items():
        if name not in {"fixed_heat_null", "fixed_heat_thermal"}:
            raise ValueError(f"Unknown supplemental response term {name!r}.")
        if name not in enabled:
            continue
        if not isinstance(value, torch.Tensor) or value.numel() != 1 or not bool(torch.isfinite(value)):
            raise ValueError(f"Supplemental response term {name!r} must be a finite scalar tensor.")
        terms[name] = value.reshape(())
    return StencilLossTerms(terms=terms, diagnostics=diagnostics)


def _target_peak_slots(record: SolveRecord) -> dict[int, float]:
    if record.output is None:
        raise ValueError("Peak supervision requires a physical output record.")
    slot_by_id = {module.module_id: index for index, module in enumerate(record.design.modules)}
    try:
        return {slot_by_id[key]: float(value) for key, value in record.output.module_peak_temperature.items()}
    except KeyError as exc:
        raise ValueError("Reference peak names a module outside the design state.") from exc


def _smooth_peak_tensor(values: torch.Tensor, beta: float) -> torch.Tensor:
    maximum = values.max()
    return maximum + (torch.logsumexp(beta * (values - maximum), dim=0) - values.new_tensor(float(values.numel())).log()) / beta


__all__ = [
    "FixedHeatNullControl",
    "StencilLossTerms",
    "ThermalLossScales",
    "compute_stencil_loss_terms",
    "fixed_heat_control_loss_terms",
    "fixed_heat_material_peak_coverage",
    "weighted_masked_mse",
]
