"""Target-free bridge between response evidence and an absolute operator.

The physical evidence records live in ``interaction_evidence``. This module
strips their labels, split identifiers, and validity masks before constructing
the callback input used by a forward model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import torch

from channelthermal.interaction_evidence.response_dataset import ResponseStencil
from channelthermal.interaction_evidence.types import DesignState, OperatingContext, RoleOutput


def _tensor_array(value: Any, *, dtype: torch.dtype, device: torch.device | str | None) -> torch.Tensor:
    # Evidence arrays are intentionally read-only; copy so Torch never exposes
    # a writable tensor view over their immutable backing storage.
    return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=device)


@dataclass(frozen=True)
class DesignInput:
    """Numeric, differentiable design inputs without evidence-family metadata."""

    module_positions: torch.Tensor
    module_heating: torch.Tensor
    module_present: torch.Tensor

    def __post_init__(self) -> None:
        positions = self.module_positions
        heating = self.module_heating
        present = self.module_present
        if positions.ndim != 2 or positions.shape[-1] != 2:
            raise ValueError("module_positions must have shape [M,2].")
        if heating.shape != (positions.shape[0],) or present.shape != heating.shape:
            raise ValueError("module heating/presence must have shape [M].")
        if not positions.is_floating_point() or not heating.is_floating_point():
            raise TypeError("module positions and heating must be floating point tensors.")
        if not torch.isfinite(positions).all() or not torch.isfinite(heating).all():
            raise ValueError("Design inputs must be finite.")

    @classmethod
    def from_state(
        cls,
        design: DesignState,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        requires_grad: bool = False,
    ) -> DesignInput:
        """Copy only physical design values; omit anchor, family, and split IDs."""

        positions = _tensor_array(
            [module.position_xy for module in design.modules], dtype=dtype, device=device
        )
        heating = _tensor_array([module.heating for module in design.modules], dtype=dtype, device=device)
        present = _tensor_array([module.active for module in design.modules], dtype=torch.bool, device=device)
        if requires_grad:
            positions.requires_grad_(True)
            heating.requires_grad_(True)
        return cls(positions, heating, present)


@dataclass(frozen=True)
class RoleQuery:
    """Target-free receiver geometry and fixed output schema for one role.

    ``receiver_slots`` maps material-coordinate queries to rows in the current
    design tensor. It carries no learned identity. Query IDs, target values,
    validity masks, quadrature weights, and noise floors are intentionally
    absent.
    """

    role: str
    query_features: torch.Tensor
    channel_names: tuple[str, ...]
    channel_units: tuple[str, ...]
    receiver_slots: tuple[int, ...] | None
    coordinate_kind: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("A receiver role must be named.")
        if self.query_features.ndim != 2 or not self.query_features.is_floating_point():
            raise ValueError("query_features must be a floating point [N,D] tensor.")
        if len(self.channel_names) == 0 or len(self.channel_names) != len(self.channel_units):
            raise ValueError("Role channel names and units must be nonempty and aligned.")
        if self.receiver_slots is not None:
            if len(self.receiver_slots) != self.query_features.shape[0]:
                raise ValueError("receiver_slots must have one entry per receiver query.")
            if any(slot < 0 for slot in self.receiver_slots):
                raise ValueError("receiver slots must be nonnegative.")


@dataclass(frozen=True)
class AbsolutePrediction:
    """Physical-unit outputs of the absolute field operator at one design."""

    role_values: Mapping[str, torch.Tensor]
    receiver_world_xy: Mapping[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        values = dict(self.role_values)
        world = dict(self.receiver_world_xy or {})
        if not values:
            raise ValueError("An absolute prediction must contain at least one role.")
        for role, tensor in values.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Prediction for role {role!r} must be a torch.Tensor.")
            if tensor.ndim != 2:
                raise ValueError(f"Prediction for role {role!r} must have shape [N,C].")
        if set(world) - set(values):
            raise ValueError("Receiver world coordinates may name only predicted roles.")
        for role, tensor in world.items():
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape != (values[role].shape[0], 2):
                raise ValueError(f"World coordinates for role {role!r} must have shape [N,2].")
        object.__setattr__(self, "role_values", MappingProxyType(values))
        object.__setattr__(self, "receiver_world_xy", MappingProxyType(world))


class AbsoluteOperator(Protocol):
    """Callable adapter for any absolute ThermalChannel field architecture."""

    def __call__(
        self,
        design: DesignInput,
        context: Mapping[str, Any],
        role_queries: Mapping[str, RoleQuery],
    ) -> AbsolutePrediction: ...


def _role_query(
    role: RoleOutput,
    design: DesignState,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    query_requires_grad: bool,
) -> RoleQuery:
    slots = None
    if role.receiver_module_ids is not None:
        index_by_id = {module.module_id: index for index, module in enumerate(design.modules)}
        try:
            slots = tuple(index_by_id[module_id] for module_id in role.receiver_module_ids)
        except KeyError as exc:
            raise ValueError(
                f"Role {role.role!r} names a receiver module absent from its design."
            ) from exc
    features = _tensor_array(role.query_features, dtype=dtype, device=device)
    if query_requires_grad:
        features.requires_grad_(True)
    return RoleQuery(
        role=role.role,
        query_features=features,
        channel_names=role.channel_names,
        channel_units=role.channel_units,
        receiver_slots=slots,
        coordinate_kind=role.coordinate_kind,
    )


def role_queries_from_stencil(
    stencil: ResponseStencil,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    query_requires_grad: bool = False,
) -> Mapping[str, RoleQuery]:
    """Build model inputs from the baseline role geometry, never its labels."""

    output = stencil.baseline.output
    if output is None:  # guarded by ResponseStencil, retained as a clear API error
        raise ValueError("A response stencil baseline must contain physical output.")
    queries = {
        name: _role_query(
            role,
            stencil.baseline.design,
            dtype=dtype,
            device=device,
            query_requires_grad=query_requires_grad,
        )
        for name, role in output.roles.items()
    }
    return MappingProxyType(queries)


def context_inputs(context: OperatingContext) -> Mapping[str, Any]:
    """Expose only the declared operating context values to the model adapter."""

    return MappingProxyType(dict(context.values))


def role_receiver_world_xy(
    query: RoleQuery,
    design: DesignInput,
    *,
    module_radius: float,
) -> torch.Tensor:
    """Recompute trial world positions while preserving material labels.

    Eulerian coordinates remain fixed. Interface angles use their declared
    unit normals, and normalized solid-material coordinates use the declared
    local ``[-1,1]`` frame. Material receiver slots select the live trial
    module centers, so a material point moves with its own module without
    changing the local coordinates passed to the native model heads.
    """

    if not np.isfinite(module_radius) or module_radius <= 0.0:
        raise ValueError("module_radius must be positive and finite.")
    if query.coordinate_kind == "eulerian":
        if query.query_features.shape[1] < 2:
            raise ValueError("Eulerian queries need x and y coordinates.")
        return query.query_features[:, :2]
    if query.receiver_slots is None:
        raise ValueError(f"Material role {query.role!r} needs design receiver slots.")
    slots = torch.as_tensor(query.receiver_slots, dtype=torch.long, device=design.module_positions.device)
    if slots.numel() and int(slots.max()) >= int(design.module_positions.shape[0]):
        raise ValueError(f"Role {query.role!r} contains a receiver slot outside the live design.")
    centers = design.module_positions.index_select(0, slots)
    if query.coordinate_kind == "interface_material_angle":
        if query.query_features.shape[1] < 3:
            raise ValueError("Interface angle queries need theta and unit-normal x/y channels.")
        local = query.query_features[:, 1:3]
    elif query.coordinate_kind == "solid_material_normalized_xy":
        if query.query_features.shape[1] < 2:
            raise ValueError("Solid material queries need normalized local x/y coordinates.")
        local = query.query_features[:, :2]
    else:
        raise ValueError(f"Unsupported receiver coordinate kind {query.coordinate_kind!r}.")
    return centers + float(module_radius) * local.to(dtype=centers.dtype, device=centers.device)


__all__ = [
    "AbsoluteOperator",
    "AbsolutePrediction",
    "DesignInput",
    "RoleQuery",
    "context_inputs",
    "role_queries_from_stencil",
    "role_receiver_world_xy",
]
