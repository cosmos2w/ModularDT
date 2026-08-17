"""Case-neutral data contracts for the hierarchical HONF inverse workflow.

Physical design ``D`` is padded module geometry/presence/heat, context ``c`` is
the operating/material condition, request ``R`` is a masked functional set plus
separate geometry constraints, compact plan ``G`` is the intended canonical
mechanism, and realized plan ``G_hat`` is re-extracted from the frozen HONF.
The contracts here carry those objects without importing a physical case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def finite_array(value: Any, *, name: str, dtype: np.dtype[Any] = np.dtype(np.float32)) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def jsonable(value: Any) -> Any:
    """Convert contract values to strict JSON primitives without hiding nonfinite data."""

    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Cannot serialize a non-finite floating-point value.")
    return value


@dataclass(frozen=True)
class PhysicalDesign:
    """One padded modular design in physical units."""

    module_centers: np.ndarray
    module_present: np.ndarray
    heat_powers: np.ndarray
    module_family_id: str = "thermal_disk"

    def __post_init__(self) -> None:
        centers = finite_array(self.module_centers, name="module_centers")
        present = finite_array(self.module_present, name="module_present").reshape(-1)
        heat = finite_array(self.heat_powers, name="heat_powers").reshape(-1)
        if centers.ndim != 2 or centers.shape[-1] != 2:
            raise ValueError("module_centers must have shape [M,2].")
        if centers.shape[0] != present.shape[0] or heat.shape != present.shape:
            raise ValueError("Design center/presence/heat slot counts must match.")
        if not np.isin(present, [0.0, 1.0]).all():
            raise ValueError("module_present must be binary.")
        if not str(self.module_family_id).strip():
            raise ValueError("module_family_id must be non-empty.")
        object.__setattr__(self, "module_centers", centers.astype(np.float32))
        object.__setattr__(self, "module_present", present.astype(np.float32))
        object.__setattr__(self, "heat_powers", heat.astype(np.float32))

    @property
    def max_modules(self) -> int:
        return int(self.module_present.shape[0])

    @property
    def module_count(self) -> int:
        return int(np.sum(self.module_present > 0.5))

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_centers": self.module_centers.tolist(),
            "module_present": self.module_present.astype(np.uint8).tolist(),
            "heat_powers": self.heat_powers.tolist(),
            "module_count": self.module_count,
            "module_family_id": self.module_family_id,
        }


@dataclass(frozen=True)
class NamedContext:
    """One named operating-context vector."""

    feature_names: tuple[str, ...]
    vector: np.ndarray
    schema_name: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.feature_names)
        vector = finite_array(self.vector, name="context vector").reshape(-1)
        if not names or len(set(names)) != len(names) or len(names) != vector.shape[0]:
            raise ValueError("Context feature names must be unique and match vector length.")
        if not str(self.schema_name).strip() or int(self.schema_version) <= 0:
            raise ValueError("Context schema name/version are required.")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "vector", vector.astype(np.float32))

    def as_mapping(self) -> dict[str, float]:
        return {name: float(self.vector[index]) for index, name in enumerate(self.feature_names)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            **self.as_mapping(),
        }


@dataclass(frozen=True)
class CompactPlan:
    """One fixed-edge compact plan in raw and normalized schema coordinates."""

    raw: np.ndarray
    normalized: np.ndarray
    feature_names: tuple[str, ...]
    schema_name: str
    schema_version: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = finite_array(self.raw, name="compact plan raw")
        normalized = finite_array(self.normalized, name="compact plan normalized")
        names = tuple(str(name) for name in self.feature_names)
        if raw.ndim != 2 or raw.shape != normalized.shape or raw.shape[1] != len(names):
            raise ValueError("Compact plan raw/normalized arrays must share [K,F] shape and feature names.")
        if int(self.schema_version) <= 0 or not str(self.schema_name).strip():
            raise ValueError("Compact plan schema name/version are required.")
        object.__setattr__(self, "raw", raw.astype(np.float32))
        object.__setattr__(self, "normalized", normalized.astype(np.float32))
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_edges(self) -> int:
        return int(self.raw.shape[0])

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            "feature_names": list(self.feature_names),
            "num_edges": self.num_edges,
            "metadata": jsonable(self.metadata),
        }
        if include_arrays:
            payload.update(raw=self.raw.tolist(), normalized=self.normalized.tolist())
        return payload


@dataclass(frozen=True)
class FunctionalValue:
    """One exact physical functional result with selection audit data."""

    request_type: str
    value: float
    valid: bool
    selected_count: int
    units: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.valid) and not np.isfinite(float(self.value)):
            raise ValueError("A valid functional value must be finite.")
        if int(self.selected_count) < 0:
            raise ValueError("Functional selected_count must be nonnegative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_type": self.request_type,
            "value": float(self.value) if np.isfinite(float(self.value)) else None,
            "valid": bool(self.valid),
            "selected_count": int(self.selected_count),
            "units": self.units,
            "metadata": jsonable(self.metadata),
        }


@dataclass(frozen=True)
class VerificationResult:
    """Case-neutral frozen-forward result for one physical design."""

    design: PhysicalDesign
    context: NamedContext
    compact_plan: CompactPlan
    full_plan: Mapping[str, Any]
    functionals: Mapping[str, FunctionalValue]
    geometry: Mapping[str, Any]
    checkpoint_provenance: Mapping[str, Any]
    outputs: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_dense_arrays: bool = False) -> dict[str, Any]:
        return {
            "design": self.design.to_dict(),
            "context": self.context.to_dict(),
            "compact_plan": self.compact_plan.to_dict(include_arrays=include_dense_arrays),
            "full_plan": jsonable(self.full_plan) if include_dense_arrays else {"keys": sorted(self.full_plan)},
            "functionals": {name: value.to_dict() for name, value in self.functionals.items()},
            "geometry": jsonable(self.geometry),
            "checkpoint_provenance": jsonable(self.checkpoint_provenance),
            "outputs": jsonable(self.outputs) if include_dense_arrays else {"keys": sorted(self.outputs)},
        }
