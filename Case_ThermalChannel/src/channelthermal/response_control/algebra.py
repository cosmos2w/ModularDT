"""Finite and mixed responses derived from one absolute field operator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from channelthermal.interaction_evidence.response_dataset import ResponseStencil
from channelthermal.interaction_evidence.types import SolveRecord

from .contracts import (
    AbsoluteOperator,
    AbsolutePrediction,
    DesignInput,
    RoleQuery,
    context_inputs,
    role_queries_from_stencil,
)


@dataclass(frozen=True)
class MixedResponseSpec:
    """Four physical corners defining one anchored mixed response."""

    role: str
    joint_variant: str
    first_variant: str
    second_variant: str
    label: str = "anchored_mixed_response"

    def __post_init__(self) -> None:
        if not self.role or not all(
            (self.joint_variant, self.first_variant, self.second_variant)
        ):
            raise ValueError("A mixed-response specification needs a role and three corner labels.")
        if len({self.joint_variant, self.first_variant, self.second_variant}) != 3:
            raise ValueError("The joint and two single-move corners must be distinct.")


@dataclass(frozen=True)
class StencilPredictions:
    """Absolute predictions and target-free requests keyed by stencil label."""

    values: Mapping[str, AbsolutePrediction]
    role_queries: Mapping[str, RoleQuery]

    def __post_init__(self) -> None:
        values = dict(self.values)
        queries = dict(self.role_queries)
        if "baseline" not in values:
            raise ValueError("Stencil predictions must include the absolute baseline.")
        if not queries:
            raise ValueError("Stencil predictions must retain their receiver requests.")
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "role_queries", MappingProxyType(queries))

    def finite(self, variant: str, role: str) -> torch.Tensor:
        """Return F(d_variant)-F(d_base), with autograd through both states."""

        try:
            return self.values[variant].role_values[role] - self.values["baseline"].role_values[role]
        except KeyError as exc:
            raise KeyError(f"Missing prediction for variant/role {variant!r}/{role!r}.") from exc

    def mixed(self, spec: MixedResponseSpec) -> torch.Tensor:
        """Return the explicit four-state inclusion-exclusion prediction."""

        try:
            base = self.values["baseline"].role_values[spec.role]
            joint = self.values[spec.joint_variant].role_values[spec.role]
            first = self.values[spec.first_variant].role_values[spec.role]
            second = self.values[spec.second_variant].role_values[spec.role]
        except KeyError as exc:
            raise KeyError(f"Missing absolute prediction needed for mixed response {spec.label!r}.") from exc
        return joint - first - second + base


def _invoke(
    operator: AbsoluteOperator,
    record: SolveRecord,
    role_queries: Mapping[str, RoleQuery],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    design_requires_grad: bool,
) -> AbsolutePrediction:
    design = DesignInput.from_state(
        record.design, dtype=dtype, device=device, requires_grad=design_requires_grad
    )
    # The callback receives the frozen operating context, never the record or
    # any target-side output/provenance metadata.
    prediction = operator(design, context_inputs(record.context), role_queries)
    if not isinstance(prediction, AbsolutePrediction):
        raise TypeError("Absolute operator callbacks must return AbsolutePrediction.")
    if set(prediction.role_values) != set(role_queries):
        raise ValueError("Absolute prediction roles must exactly match the requested roles.")
    for name, request in role_queries.items():
        value = prediction.role_values[name]
        expected = (request.query_features.shape[0], len(request.channel_names))
        if tuple(value.shape) != expected:
            raise ValueError(
                f"Prediction role {name!r} has shape {tuple(value.shape)}, expected {expected}."
            )
    return prediction


def predict_stencil(
    operator: AbsoluteOperator,
    stencil: ResponseStencil,
    *,
    role_queries: Mapping[str, RoleQuery] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    design_requires_grad: bool = False,
    query_requires_grad: bool = False,
) -> StencilPredictions:
    """Evaluate the same absolute operator at every stencil design.

    The physical target fields are never inputs. Each state is decoded using
    the same fixed receiver universe from the baseline role schema. Deployable
    finite responses are then differences of these absolute predictions.
    """

    queries = role_queries or role_queries_from_stencil(
        stencil, dtype=dtype, device=device, query_requires_grad=query_requires_grad
    )
    predicted: dict[str, AbsolutePrediction] = {}
    predicted["baseline"] = _invoke(
        operator,
        stencil.baseline,
        queries,
        dtype=dtype,
        device=device,
        design_requires_grad=design_requires_grad,
    )
    for label, record in stencil.variants.items():
        predicted[label] = _invoke(
            operator,
            record,
            queries,
            dtype=dtype,
            device=device,
            design_requires_grad=design_requires_grad,
        )
    return StencilPredictions(predicted, queries)


__all__ = ["MixedResponseSpec", "StencilPredictions", "predict_stencil"]
