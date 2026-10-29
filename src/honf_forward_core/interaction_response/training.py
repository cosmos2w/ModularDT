"""Masked, role-balanced objectives for finite response supervision."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from .factor_operator import AnchoredResponseFactorOperator
from .types import (
    VALID_EVIDENCE_SOURCES,
    BaselineResponseCache,
    ModuleId,
    ResponseFactor,
    ResponseQueries,
)


@dataclass(frozen=True)
class ResponseTrainingExample:
    """One response target with observation and significance masks separated."""

    record_id: str
    anchor_id: str
    physical_family_id: str
    split: str
    source: str
    response_kind: str
    delta_by_module_id: Mapping[ModuleId, torch.Tensor]
    targets: Mapping[str, torch.Tensor]
    observed_masks: Mapping[str, torch.Tensor]
    loss_masks: Mapping[str, torch.Tensor]
    sign_resolved_masks: Mapping[str, torch.Tensor]
    quadrature_weights: Mapping[str, torch.Tensor]
    mask_basis: str = "stencil_common_geometry"

    def __post_init__(self) -> None:
        if not self.record_id or not self.anchor_id or not self.physical_family_id:
            raise ValueError("Training examples require record, anchor, and family IDs.")
        source = str(getattr(self.source, "value", self.source))
        if source not in VALID_EVIDENCE_SOURCES:
            raise ValueError(f"Unknown training evidence source {source!r}.")
        if self.response_kind not in {"finite_change", "anchored_mixed_response", "centered_mixed_derivative"}:
            raise ValueError(f"Unknown response label kind {self.response_kind!r}.")
        if not self.targets or set(self.targets) != set(self.observed_masks) or set(self.targets) != set(self.loss_masks):
            raise ValueError("Targets, observed masks, and loss masks must define the same nonempty roles.")
        if set(self.sign_resolved_masks) - set(self.targets):
            raise ValueError("Sign-resolved masks cannot refer to an absent output role.")
        if set(self.quadrature_weights) - set(self.targets):
            raise ValueError("Quadrature weights cannot refer to an absent output role.")
        for role, target in self.targets.items():
            if target.ndim != 3:
                raise ValueError(f"Training targets for {role!r} must have shape [B,Q,C].")
            observed = self.observed_masks[role]
            loss = self.loss_masks[role]
            if observed.shape != target.shape or loss.shape != target.shape:
                raise ValueError(f"Observation and finite-loss masks for {role!r} must match [B,Q,C].")
            if observed.dtype is not torch.bool or loss.dtype is not torch.bool:
                raise ValueError(f"Observation and finite-loss masks for {role!r} must be boolean.")
            if bool((loss & ~observed).any()):
                raise ValueError(f"Finite-loss mask for {role!r} cannot include an unobserved response entry.")
            if role in self.sign_resolved_masks:
                sign_mask = self.sign_resolved_masks[role]
                if sign_mask.shape != target.shape or sign_mask.dtype is not torch.bool:
                    raise ValueError(f"Sign-resolution mask for {role!r} must be boolean and match [B,Q,C].")
                if bool((sign_mask & ~observed).any()):
                    raise ValueError(f"Sign-resolution mask for {role!r} cannot include an unobserved entry.")
            weights = self.quadrature_weights.get(role)
            if weights is not None and weights.shape != target.shape[:2]:
                raise ValueError(f"Quadrature weights for {role!r} must have shape [B,Q].")
        if not self.mask_basis:
            raise ValueError("mask_basis must state how finite-value supervision was selected.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "split", str(getattr(self.split, "value", self.split)))
        object.__setattr__(self, "delta_by_module_id", MappingProxyType(dict(self.delta_by_module_id)))
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))
        object.__setattr__(self, "observed_masks", MappingProxyType(dict(self.observed_masks)))
        object.__setattr__(self, "loss_masks", MappingProxyType(dict(self.loss_masks)))
        object.__setattr__(self, "sign_resolved_masks", MappingProxyType(dict(self.sign_resolved_masks)))
        object.__setattr__(self, "quadrature_weights", MappingProxyType(dict(self.quadrature_weights)))


def stencil_examples_to_torch(
    examples: Sequence[Any],
    *,
    module_ids: Sequence[ModuleId],
    input_coordinate_scales: Sequence[float],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    response_kind: str = "finite_change",
    loss_mask_overrides_by_record_role: Mapping[tuple[str, str], np.ndarray] | None = None,
    sign_resolution_floors_by_role: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    mask_basis: str = "stencil_common_geometry",
) -> tuple[ResponseTrainingExample, ...]:
    """Convert source-labeled stencil examples without dropping masks or IDs.

    This adapter is intentionally structural: it consumes the public
    ``ResponseExample``/``ResponseBlock`` fields from the evidence package but
    keeps the case-neutral HONF core free of a ThermalChannel dependency.
    Coordinate scales are training-input conventions (for example domain
    lengths and heat scale), never fitted from response targets.
    """

    scales = np.asarray(input_coordinate_scales, dtype=np.float64).reshape(-1)
    if scales.size == 0 or not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("Input coordinate scales must be finite, positive, and nonempty.")
    ids = tuple(module_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("module_ids must contain unique active physical identities.")
    result = []
    loss_overrides = loss_mask_overrides_by_record_role or {}
    sign_floors = sign_resolution_floors_by_role or {}
    for example in examples:
        raw_perturbations = example.perturbation_by_module
        missing = [module_id for module_id in ids if module_id not in raw_perturbations]
        if missing:
            raise KeyError(f"Stencil example omits explicit perturbations for active modules: {missing!r}.")
        deltas: dict[ModuleId, torch.Tensor] = {}
        for module_id in ids:
            raw_delta = np.asarray(raw_perturbations[module_id], dtype=np.float64).reshape(-1)
            if raw_delta.shape != scales.shape or not np.isfinite(raw_delta).all():
                raise ValueError(f"Perturbation for {module_id!r} must be finite with {scales.size} coordinates.")
            deltas[module_id] = torch.as_tensor(
                (raw_delta / scales)[None], dtype=dtype, device=device
            )

        targets: dict[str, torch.Tensor] = {}
        observed_masks: dict[str, torch.Tensor] = {}
        loss_masks: dict[str, torch.Tensor] = {}
        sign_masks: dict[str, torch.Tensor] = {}
        quadrature: dict[str, torch.Tensor] = {}
        for role, block in example.roles.items():
            target = np.array(block.delta, copy=True)
            mask = np.array(block.valid_mask, dtype=bool, copy=True)
            weights = np.array(block.quadrature_weights, dtype=np.float64, copy=True).reshape(-1)
            if target.ndim != 2 or mask.shape != target.shape or weights.shape != (target.shape[0],):
                raise ValueError(f"Response block {role!r} has inconsistent [Q,C] target/mask/weight shapes.")
            if np.any(~np.isfinite(target[mask])):
                raise ValueError(f"Resolved stencil target entries for {role!r} must be finite.")
            if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError(f"Quadrature weights for {role!r} must be finite and nonnegative.")
            loss_mask = loss_overrides.get((str(example.record_id), role))
            if loss_mask is None:
                loss_mask = mask.copy()
            else:
                loss_mask = np.array(loss_mask, dtype=bool, copy=True)
                if loss_mask.shape != mask.shape:
                    raise ValueError(f"Loss-mask override for {(example.record_id, role)!r} must match [Q,C].")
                if np.any(loss_mask & ~mask):
                    raise ValueError("Loss-mask overrides may only restrict the observed stencil mask.")
            if role in sign_floors:
                floors = np.asarray(sign_floors[role], dtype=np.float64).reshape(-1)
                if floors.shape != (target.shape[1],) or not np.isfinite(floors).all() or np.any(floors < 0.0):
                    raise ValueError(f"Sign-resolution floors for {role!r} must be finite nonnegative [C].")
                sign_masks[role] = torch.as_tensor(
                    (mask & (np.abs(target) > floors[None]))[None], dtype=torch.bool, device=device
                )
            targets[role] = torch.as_tensor(target[None], dtype=dtype, device=device)
            observed_masks[role] = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
            loss_masks[role] = torch.as_tensor(loss_mask[None], dtype=torch.bool, device=device)
            quadrature[role] = torch.as_tensor(weights[None], dtype=dtype, device=device)

        result.append(
            ResponseTrainingExample(
                record_id=str(example.record_id),
                anchor_id=str(example.anchor_id),
                physical_family_id=str(example.physical_family_id),
                split=example.split,
                source=example.source,
                response_kind=response_kind,
                delta_by_module_id=deltas,
                targets=targets,
                observed_masks=observed_masks,
                loss_masks=loss_masks,
                sign_resolved_masks=sign_masks,
                quadrature_weights=quadrature,
                mask_basis=mask_basis,
            )
        )
    return tuple(result)


def response_factor_training_step(
    operator: AnchoredResponseFactorOperator,
    optimizer: torch.optim.Optimizer,
    cache: BaselineResponseCache,
    factors: Sequence[ResponseFactor],
    queries: ResponseQueries,
    example: ResponseTrainingExample,
    *,
    training_scales: Mapping[str, float | Sequence[float] | torch.Tensor],
    role_weights: Mapping[str, float] | None = None,
    query_chunk_size: int | None = None,
    factor_weights: Mapping[str, float | torch.Tensor] | None = None,
) -> dict[str, float | str]:
    """Run one recorded, mask-aware optimizer step through the factor graph."""

    if set(example.targets) - set(queries):
        raise KeyError("Training targets include a role absent from the fixed response queries.")
    for role, target in example.targets.items():
        if target.shape[0] != cache.batch_size or target.shape[1] != queries[role].features.shape[1]:
            raise ValueError(f"Training target role {role!r} does not align with cache/query batch dimensions.")
    optimizer.zero_grad(set_to_none=True)
    predictions = operator.predict_response(
        cache,
        factors,
        example.delta_by_module_id,
        queries,
        factor_weights=factor_weights,
        query_chunk_size=query_chunk_size,
    )
    if not isinstance(predictions, dict):
        raise TypeError("The training step requires summed role predictions.")
    loss, components = masked_role_response_loss(
        predictions,
        example.targets,
        example.loss_masks,
        training_scales=training_scales,
        role_weights=role_weights,
        quadrature_weights=example.quadrature_weights,
        return_components=True,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Response training loss is nonfinite.")
    loss.backward()
    gradients = [parameter.grad for parameter in operator.parameters() if parameter.grad is not None]
    if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Response optimizer gradients are absent or nonfinite.")
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "response_kind": example.response_kind,
        **{f"loss/{role}": float(value.detach().item()) for role, value in components.items()},
    }


def masked_role_response_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    loss_masks: Mapping[str, torch.Tensor],
    *,
    training_scales: Mapping[str, float | Sequence[float] | torch.Tensor],
    role_weights: Mapping[str, float] | None = None,
    quadrature_weights: Mapping[str, torch.Tensor] | None = None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a masked finite-response loss without treating unknown as zero.

    Each named physical output role is first averaged over its own resolved,
    quadrature-weighted query/channel entries and normalized by an externally
    fitted training-only scale. The role means are then averaged so a dense
    fluid grid cannot overwhelm a smaller interface or solid response block.
    The masks passed here describe finite-value entries selected for this loss.
    They can reflect common geometric observation support without asserting
    that every small response is above a numerical significance floor. Keep
    sign/support resolution masks separate. Call once per response label
    family and combine with explicit weights when both are trained.
    """

    if set(predictions) != set(targets) or set(predictions) != set(loss_masks):
        raise ValueError("Predictions, targets, and finite-loss masks must have the same output roles.")
    if set(predictions) - set(training_scales):
        raise ValueError("Every response role needs a training-only normalization scale.")
    if not predictions:
        raise ValueError("At least one resolved response role is required.")

    components: dict[str, torch.Tensor] = {}
    weights_by_role = role_weights or {}
    quadrature_by_role = quadrature_weights or {}
    for role, prediction in predictions.items():
        target = targets[role]
        mask = loss_masks[role].to(device=prediction.device, dtype=torch.bool)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction and target shapes differ for role {role!r}.")
        if mask.shape == prediction.shape[:-1]:
            mask = mask.unsqueeze(-1).expand_as(prediction)
        elif mask.shape != prediction.shape:
            raise ValueError(f"Loss mask for role {role!r} must have shape [B,Q] or match [B,Q,C].")
        if not bool(mask.any()):
            continue
        target_tensor = target.to(device=prediction.device, dtype=prediction.dtype)
        selected_target = target_tensor[mask]
        selected_prediction = prediction[mask]
        if not bool(torch.isfinite(selected_target).all()):
            raise ValueError(f"Observed response targets for role {role!r} must be finite under the loss mask.")
        if not bool(torch.isfinite(selected_prediction).all()):
            raise ValueError(f"Predicted responses for resolved role {role!r} must be finite.")

        scale_spec = training_scales[role]
        scale = torch.as_tensor(scale_spec, device=prediction.device, dtype=prediction.dtype)
        if scale.ndim == 0:
            scale = scale.reshape(1)
        if scale.ndim != 1 or scale.numel() not in (1, prediction.shape[-1]):
            raise ValueError(f"Training scale for role {role!r} must be scalar or have one value per channel.")
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any()):
            raise ValueError(f"Training scales for role {role!r} must be finite and positive.")
        safe_target = torch.where(mask, target_tensor, prediction.detach())
        normalized_squared = ((prediction - safe_target) / scale).square()

        if role in quadrature_by_role:
            spatial_weight = quadrature_by_role[role].to(device=prediction.device, dtype=prediction.dtype)
            if spatial_weight.shape != prediction.shape[:-1]:
                raise ValueError(f"Quadrature weights for role {role!r} must have shape [B,Q].")
            if bool((spatial_weight < 0.0).any()) or not bool(torch.isfinite(spatial_weight).all()):
                raise ValueError(f"Quadrature weights for role {role!r} must be finite and nonnegative.")
            entry_weight = spatial_weight.unsqueeze(-1).expand_as(prediction) * mask
        else:
            entry_weight = mask.to(dtype=prediction.dtype)
        denominator = entry_weight.sum()
        if not bool(denominator > 0.0):
            continue
        role_mean = (normalized_squared * entry_weight).sum() / denominator
        role_weight = float(weights_by_role.get(role, 1.0))
        if not math.isfinite(role_weight) or role_weight <= 0.0:
            raise ValueError(f"Role weight for {role!r} must be finite and positive.")
        components[role] = role_mean

    if not components:
        raise ValueError("No resolved response entries with positive quadrature weight were supplied.")
    total_role_weight = sum(float(weights_by_role.get(role, 1.0)) for role in components)
    weighted_components = [components[role] * float(weights_by_role.get(role, 1.0)) for role in components]
    loss = torch.stack(weighted_components).sum() / total_role_weight
    if return_components:
        return loss, components
    return loss
