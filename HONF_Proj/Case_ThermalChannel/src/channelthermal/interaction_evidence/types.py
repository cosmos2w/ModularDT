"""Typed records for anchored, source-aware physical response evidence.

The records deliberately keep physical solver output, stored dataset targets,
learned teachers, and analytic checks distinct. Arrays are copied and marked
read-only at construction so one workflow cannot silently mutate another
workflow's evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np


class EvidenceSource(str, Enum):
    REFERENCE_SOLVER = "reference_solver"
    STORED_REFERENCE = "stored_reference"
    SURROGATE_TEACHER = "surrogate_teacher"
    ANALYTIC_SYNTHETIC = "analytic_synthetic"


class EvidenceSplit(str, Enum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    FINAL_REVIEW = "final_review"
    DEVELOPMENT = "development"


class SolveStatus(str, Enum):
    CONVERGED = "converged"
    UNCONVERGED = "unconverged"
    FAILED = "failed"


def _readonly_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        return _readonly_array(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ModuleState:
    """One explicit module identity and its current physical design values."""

    module_id: str
    position_xy: tuple[float, float]
    heating: float
    active: bool = True

    def __post_init__(self) -> None:
        if not self.module_id:
            raise ValueError("module_id must be a nonempty explicit label.")
        if len(self.position_xy) != 2 or not np.isfinite(self.position_xy).all():
            raise ValueError("position_xy must contain two finite coordinates.")
        if not np.isfinite(self.heating):
            raise ValueError("heating must be finite.")


@dataclass(frozen=True)
class DesignState:
    """A design and the family split that owns every related perturbation."""

    anchor_id: str
    physical_family_id: str
    split: EvidenceSplit | str
    modules: tuple[ModuleState, ...]

    def __post_init__(self) -> None:
        split = EvidenceSplit(self.split)
        modules = tuple(self.modules)
        active_ids = [module.module_id for module in modules if module.active]
        if not self.anchor_id or not self.physical_family_id:
            raise ValueError("anchor_id and physical_family_id must be explicit.")
        if not modules or not active_ids:
            raise ValueError("A design must contain at least one active module.")
        if len(active_ids) != len(set(active_ids)):
            raise ValueError("Active module IDs must be unique within a family.")
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "modules", modules)

    @property
    def active_modules(self) -> tuple[ModuleState, ...]:
        return tuple(module for module in self.modules if module.active)

    @property
    def active_module_ids(self) -> tuple[str, ...]:
        return tuple(module.module_id for module in self.active_modules)


@dataclass(frozen=True)
class OperatingContext:
    """Frozen solver context; nested values use the source solver's config names."""

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze(self.values))


@dataclass(frozen=True)
class RoleOutput:
    """Receiver samples for one output role with channel-level validity masks."""

    role: str
    query_features: np.ndarray
    values: np.ndarray
    channel_names: tuple[str, ...]
    channel_units: tuple[str, ...]
    valid_mask: np.ndarray
    quadrature_weights: np.ndarray
    query_ids: tuple[str, ...]
    receiver_module_ids: tuple[str, ...] | None = None
    coordinate_kind: str = "fixed_query"
    noise_floor: np.ndarray | None = None

    @property
    def observed_mask(self) -> np.ndarray:
        """Geometric/source observation support, independent of noise resolution."""

        return self.valid_mask

    def __post_init__(self) -> None:
        query = _readonly_array(self.query_features, dtype=np.float64)
        values = _readonly_array(self.values, dtype=np.float64)
        names = tuple(str(name) for name in self.channel_names)
        units = tuple(str(unit) for unit in self.channel_units)
        query_ids = tuple(str(value) for value in self.query_ids)
        if query.ndim != 2 or values.ndim != 2:
            raise ValueError("query_features and values must have shape [N,D] and [N,C].")
        if values.shape[0] != query.shape[0] or values.shape[1] != len(names) or len(names) != len(units):
            raise ValueError("Role feature/value/channel shapes are inconsistent.")
        if len(query_ids) != query.shape[0] or len(set(query_ids)) != len(query_ids):
            raise ValueError("query_ids must uniquely label every receiver sample.")
        mask = np.asarray(self.valid_mask, dtype=bool)
        if mask.ndim == 1:
            mask = np.broadcast_to(mask[:, None], values.shape)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must be [N] or [N,C].")
        if np.any(~np.isfinite(values[mask])):
            raise ValueError("Every observed role value must be finite.")
        if np.any(~np.isfinite(query)):
            raise ValueError("Role query features must be finite.")
        weights = _readonly_array(self.quadrature_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (query.shape[0],) or not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("quadrature_weights must be finite, nonnegative, and [N].")
        receiver_ids = self.receiver_module_ids
        if receiver_ids is not None:
            receiver_ids = tuple(str(value) for value in receiver_ids)
            if len(receiver_ids) != query.shape[0]:
                raise ValueError("receiver_module_ids must contain one label per query.")
        floor = None
        if self.noise_floor is not None:
            floor = _readonly_array(self.noise_floor, dtype=np.float64)
            if floor.ndim == 1:
                floor = np.broadcast_to(floor[:, None], values.shape).copy()
            if floor.shape != values.shape or not np.isfinite(floor).all() or np.any(floor < 0.0):
                raise ValueError("noise_floor must be finite, nonnegative, and [N,C].")
            floor.setflags(write=False)
        object.__setattr__(self, "query_features", query)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "channel_units", units)
        object.__setattr__(self, "valid_mask", _readonly_array(mask, dtype=bool))
        object.__setattr__(self, "quadrature_weights", weights)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "receiver_module_ids", receiver_ids)
        object.__setattr__(self, "noise_floor", floor)


@dataclass(frozen=True)
class MeasuredQuantity:
    value: float
    units: str
    resolved: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.resolved and not np.isfinite(self.value):
            raise ValueError("A resolved quantity must be finite.")
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class PhysicalSolveOutput:
    """Common-grid and material-coordinate outputs consumed by response models."""

    roles: Mapping[str, RoleOutput]
    quantities: Mapping[str, MeasuredQuantity]
    active_module_ids: tuple[str, ...]
    module_peak_temperature: Mapping[str, float]
    units_metadata: Mapping[str, str]
    case_dir: str | None = None

    def __post_init__(self) -> None:
        roles = dict(self.roles)
        required = {"fluid_fields", "interface", "solid_temperature"}
        if not required.issubset(roles):
            raise ValueError(f"Physical output is missing roles: {sorted(required - set(roles))}.")
        if any(name != role.role for name, role in roles.items()):
            raise ValueError("Role map keys must match each RoleOutput.role.")
        active = tuple(str(value) for value in self.active_module_ids)
        if len(active) != len(set(active)):
            raise ValueError("active_module_ids must be unique.")
        peaks = {str(key): float(value) for key, value in self.module_peak_temperature.items()}
        if set(peaks) != set(active) or not np.isfinite(list(peaks.values())).all():
            raise ValueError("One finite internal-temperature peak is required for each active module.")
        if "pressure_drop" not in self.quantities:
            raise ValueError("Physical output must include a pressure_drop quantity.")
        object.__setattr__(self, "roles", MappingProxyType(roles))
        object.__setattr__(self, "quantities", MappingProxyType(dict(self.quantities)))
        object.__setattr__(self, "active_module_ids", active)
        object.__setattr__(self, "module_peak_temperature", MappingProxyType(peaks))
        object.__setattr__(self, "units_metadata", _freeze(self.units_metadata))

    @property
    def pressure_drop(self) -> float:
        return float(self.quantities["pressure_drop"].value)

    def as_mapping(self) -> dict[str, Any]:
        """Return the stable bridge keys used by the inverse callback."""

        fluid = self.roles["fluid_fields"]
        interface = self.roles["interface"]
        solid = self.roles["solid_temperature"]
        angles = interface.query_features[:, 0]
        return {
            "fluid_fields": fluid.values,
            "grid_xy": fluid.query_features,
            "grid_valid_mask": np.all(fluid.valid_mask, axis=1),
            "channel_order": fluid.channel_names,
            "interface": interface.values,
            "interface_angles": angles,
            "interface_module_ids": interface.receiver_module_ids,
            "interface_valid_mask": interface.valid_mask,
            "solid_temperature": solid.values,
            "solid_local_xy": solid.query_features,
            "solid_module_ids": solid.receiver_module_ids,
            "solid_valid_mask": solid.valid_mask,
            "active_module_ids": self.active_module_ids,
            "pressure_drop": self.pressure_drop,
            "module_peak_temperature": self.module_peak_temperature,
            "units_metadata": self.units_metadata,
            "case_dir": self.case_dir,
        }


@dataclass(frozen=True)
class SolveRecord:
    """One source-labeled solve attempt, successful or failed."""

    record_id: str
    design: DesignState
    context: OperatingContext
    source: EvidenceSource | str
    status: SolveStatus | str
    elapsed_seconds: float
    provenance: Mapping[str, Any]
    output: PhysicalSolveOutput | None = None
    perturbation_by_module: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    physical_step_scales: Mapping[str, float] = field(default_factory=dict)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        source = EvidenceSource(self.source)
        status = SolveStatus(self.status)
        if not self.record_id:
            raise ValueError("record_id must be explicit.")
        if not np.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and nonnegative.")
        if status is SolveStatus.FAILED and self.output is not None:
            raise ValueError("A failed attempt cannot claim a solved output.")
        if status is not SolveStatus.FAILED and self.output is None:
            raise ValueError("A completed solver attempt requires an output record.")
        perturbations = {
            str(key): tuple(float(item) for item in value)
            for key, value in self.perturbation_by_module.items()
        }
        if any(not np.isfinite(value).all() for value in perturbations.values()):
            raise ValueError("Perturbation deltas must be finite.")
        scales = {str(key): float(value) for key, value in self.physical_step_scales.items()}
        if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
            raise ValueError("Physical step scales must be positive and finite.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        object.__setattr__(self, "perturbation_by_module", MappingProxyType(perturbations))
        object.__setattr__(self, "physical_step_scales", MappingProxyType(scales))


# The plan's descriptive name for a source-labeled response record.
InteractionEvidenceRecord = SolveRecord


__all__ = [
    "DesignState",
    "EvidenceSource",
    "EvidenceSplit",
    "InteractionEvidenceRecord",
    "MeasuredQuantity",
    "ModuleState",
    "OperatingContext",
    "PhysicalSolveOutput",
    "RoleOutput",
    "SolveRecord",
    "SolveStatus",
]
