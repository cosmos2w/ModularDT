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
from .contracts import DesignInput
from .thermal import module_peak_temperatures_from_role, pressure_drop_from_field


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
) -> StencilLossTerms:
    """Compute role, four-state, solid-peak, and pressure losses for a stencil."""

    if set(predictions.values) != {"baseline", *stencil.variants}:
        raise ValueError("Predictions must cover exactly the physical stencil states.")
    enabled = {"value", "finite", "mixed", "decision", "constraint"} if enabled_terms is None else set(enabled_terms)
    unknown = enabled - {"value", "finite", "mixed", "decision", "constraint"}
    if unknown:
        raise ValueError(f"Unknown response-loss terms: {sorted(unknown)}.")
    value_losses: list[torch.Tensor] = []
    finite_losses: list[torch.Tensor] = []
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

        if "constraint" in enabled:
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

    for variant in stencil.variants if "finite" in enabled or "constraint" in enabled else ():
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
        if "constraint" in enabled and base_quantity.resolved and trial_quantity.resolved:
            base_pressure = _pressure_prediction("baseline", predictions, stencil)
            trial_pressure = _pressure_prediction(variant, predictions, stencil)
            target_delta = float(trial_quantity.value) - float(base_quantity.value)
            response = (trial_pressure - base_pressure - target_delta) / float(scales.pressure_response)
            pressure_delta_losses.append(response.square())
            diagnostics[f"constraint/{variant}/pressure_response"] = response.square().detach()

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
    if len(set(feasibility_targets)) > 1:
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
    if "constraint" in enabled and observed_constraint_components:
        terms["constraint"] = torch.stack(observed_constraint_components).mean()
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
    "StencilLossTerms",
    "ThermalLossScales",
    "compute_stencil_loss_terms",
    "weighted_masked_mse",
]
