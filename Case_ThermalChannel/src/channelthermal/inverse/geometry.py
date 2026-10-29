"""ThermalChannel physical-design canonicalization and exact geometry checks.

Physical design ``D`` is padded centers/presence/heat in a rectangular channel.
Context ``c`` provides radius and domain lengths. Request ``R`` carries the
separate constraints evaluated here. Compact plans ``G`` and ``G_hat`` do not
replace physical geometry validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from honf_inverse_core.contracts import NamedContext, PhysicalDesign
from honf_inverse_core.normalization import ScalarStats
from honf_inverse_core.request_schema import GeometryConstraints


GEOMETRY_CONSTRAINT_NAMES = (
    "module_count_min",
    "module_count_max",
    "minimum_center_distance",
    "wall_clearance",
    "inlet_clearance",
    "outlet_clearance",
    "total_heat_low",
    "total_heat_high",
)
GEOMETRY_ACTUAL_NAMES = (
    "module_count",
    "minimum_pair_distance",
    "wall_clearance",
    "inlet_clearance",
    "outlet_clearance",
    "total_heat",
)


@dataclass(frozen=True)
class CanonicalDesignView:
    """Canonical active-first ``D`` plus reversible slot permutations."""

    design: PhysicalDesign
    canonical_to_source: np.ndarray
    source_to_canonical: np.ndarray

    def __post_init__(self) -> None:
        c2s = np.asarray(self.canonical_to_source, dtype=np.int64).reshape(-1)
        s2c = np.asarray(self.source_to_canonical, dtype=np.int64).reshape(-1)
        expected = list(range(self.design.max_modules))
        if sorted(c2s.tolist()) != expected or sorted(s2c.tolist()) != expected:
            raise ValueError("Canonical/source maps must each be a full slot permutation.")
        if not np.array_equal(s2c[c2s], np.arange(self.design.max_modules)):
            raise ValueError("Canonical/source maps must be mutual inverses.")
        object.__setattr__(self, "canonical_to_source", c2s)
        object.__setattr__(self, "source_to_canonical", s2c)


@dataclass(frozen=True)
class GeometryEvaluation:
    """Exact realized constraint values, margins, violations, and masks."""

    constraint_raw: np.ndarray
    constraint_mask: np.ndarray
    actual_raw: np.ndarray
    margin_raw: np.ndarray
    violation_raw: np.ndarray
    valid: bool
    pair_distance_defined: bool

    def __post_init__(self) -> None:
        constraint = np.asarray(self.constraint_raw, dtype=np.float32).reshape(-1)
        mask = np.asarray(self.constraint_mask, dtype=np.uint8).reshape(-1)
        actual = np.asarray(self.actual_raw, dtype=np.float32).reshape(-1)
        margin = np.asarray(self.margin_raw, dtype=np.float32).reshape(-1)
        violation = np.asarray(self.violation_raw, dtype=np.float32).reshape(-1)
        if constraint.shape != (8,) or mask.shape != (8,) or margin.shape != (8,) or violation.shape != (8,):
            raise ValueError("Geometry constraint/mask/margin/violation arrays must have length 8.")
        if actual.shape != (6,):
            raise ValueError("Geometry actual array must have length 6.")
        if not np.isfinite(np.concatenate([constraint, actual, margin, violation])).all():
            raise ValueError("Geometry evaluation contains non-finite values.")
        object.__setattr__(self, "constraint_raw", constraint)
        object.__setattr__(self, "constraint_mask", mask)
        object.__setattr__(self, "actual_raw", actual)
        object.__setattr__(self, "margin_raw", margin)
        object.__setattr__(self, "violation_raw", violation)

    @property
    def total_violation(self) -> float:
        return float(np.sum(self.violation_raw * self.constraint_mask))

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_names": list(GEOMETRY_CONSTRAINT_NAMES),
            "constraint_raw": self.constraint_raw.tolist(),
            "constraint_mask": self.constraint_mask.tolist(),
            "actual_names": list(GEOMETRY_ACTUAL_NAMES),
            "actual_raw": self.actual_raw.tolist(),
            "margin_raw": self.margin_raw.tolist(),
            "violation_raw": self.violation_raw.tolist(),
            "total_violation": self.total_violation,
            "valid": bool(self.valid),
            "pair_distance_defined": bool(self.pair_distance_defined),
        }


def design_from_forward_structure(structure: Mapping[str, Any]) -> PhysicalDesign:
    """Read the maintained raw forward structure into ``D``."""

    return PhysicalDesign(
        module_centers=np.asarray(structure["module_centers"], dtype=np.float32),
        module_present=np.asarray(structure["module_present"], dtype=np.float32),
        heat_powers=np.asarray(structure["heat_powers"], dtype=np.float32),
    )


def canonicalize_design(design: PhysicalDesign, context: NamedContext) -> CanonicalDesignView:
    """Sort active modules lexicographically, pack them first, and zero padding."""

    values = context.as_mapping()
    lx = values["domain_length_x"]
    ly = values["domain_length_y"]
    present = design.module_present > 0.5
    order = sorted(
        range(design.max_modules),
        key=lambda index: (
            0 if present[index] else 1,
            float(design.module_centers[index, 0] / lx) if present[index] else 0.0,
            float(design.module_centers[index, 1] / ly) if present[index] else 0.0,
            float(design.heat_powers[index]) if present[index] else 0.0,
            int(index),
        ),
    )
    canonical_to_source = np.asarray(order, dtype=np.int64)
    source_to_canonical = np.empty_like(canonical_to_source)
    source_to_canonical[canonical_to_source] = np.arange(design.max_modules, dtype=np.int64)
    canonical_present = design.module_present[canonical_to_source].copy()
    canonical_centers = design.module_centers[canonical_to_source].copy()
    canonical_heat = design.heat_powers[canonical_to_source].copy()
    inactive = canonical_present < 0.5
    canonical_centers[inactive] = 0.0
    canonical_heat[inactive] = 0.0
    return CanonicalDesignView(
        design=PhysicalDesign(
            module_centers=canonical_centers,
            module_present=canonical_present,
            heat_powers=canonical_heat,
            module_family_id=design.module_family_id,
        ),
        canonical_to_source=canonical_to_source,
        source_to_canonical=source_to_canonical,
    )


def decode_generated_design(
    normalized_layout: Any,
    module_present: Any,
    context: NamedContext,
    constraints: GeometryConstraints,
    *,
    heat_mean: float,
    heat_std: float,
) -> PhysicalDesign:
    """Apply one analytic constraint-aware decode from flow state to physical `D`.

    This is part of the endpoint parameterization, not iterative repair. Direct
    normalized centers are first clipped to the edge-valid rectangle. If that
    layout already satisfies pair distance it passes through exactly, matching
    the training target representation. Only an invalid endpoint takes one
    analytic ordered-x fallback after reserving the requested minimum gap.
    Optional total heat is scaled once into its allowed interval.
    """

    layout = np.asarray(normalized_layout, dtype=np.float64)
    present = np.asarray(module_present, dtype=np.float64).reshape(-1) > 0.5
    if layout.ndim != 2 or layout.shape != (present.shape[0], 3):
        raise ValueError("Generated normalized layout must have shape [M,3].")
    validate_geometry_constraints(constraints, context, max_modules=present.shape[0])
    values = context.as_mapping()
    radius = values["module_radius"]
    lx = values["domain_length_x"]
    ly = values["domain_length_y"]
    x_low = radius + constraints.inlet_clearance
    x_high = lx - radius - constraints.outlet_clearance
    y_low = radius + constraints.wall_clearance
    y_high = ly - radius - constraints.wall_clearance
    if x_low > x_high or y_low > y_high:
        raise ValueError("Geometry constraints leave no edge-valid center rectangle.")
    active = np.flatnonzero(present)
    centers = np.zeros((present.shape[0], 2), dtype=np.float64)
    if active.size:
        raw = np.clip(layout[active, :2], 0.0, 1.0)
        minimum = constraints.minimum_center_distance
        direct = np.column_stack(
            [
                np.clip(raw[:, 0] * lx, x_low, x_high),
                np.clip(raw[:, 1] * ly, y_low, y_high),
            ]
        )
        if active.size < 2:
            direct_valid = True
        else:
            delta = direct[:, None, :] - direct[None, :, :]
            distance = np.sqrt(np.sum(delta**2, axis=-1))
            distance[np.diag_indices_from(distance)] = np.inf
            direct_valid = bool(np.min(distance) + 1.0e-7 >= minimum)
        if direct_valid:
            centers[active] = direct
        else:
            order = np.lexsort((direct[:, 1], direct[:, 0]))
            ordered_slots = active[order]
            ordered_direct = direct[order]
            residual_width = (x_high - x_low) - max(active.size - 1, 0) * minimum
            if residual_width < -1.0e-7:
                raise ValueError(
                    "Schema-v1 analytic layout decode cannot fit the requested count/minimum distance along x."
                )
            if x_high > x_low:
                ordered_coordinate = (ordered_direct[:, 0] - x_low) / (x_high - x_low)
            else:
                ordered_coordinate = np.zeros(active.size, dtype=np.float64)
            centers[ordered_slots, 0] = (
                x_low
                + np.arange(active.size, dtype=np.float64) * minimum
                + max(residual_width, 0.0) * ordered_coordinate
            )
            centers[ordered_slots, 1] = ordered_direct[:, 1]
    heat = np.maximum(layout[:, 2] * float(heat_std) + float(heat_mean), 0.0) * present
    if constraints.total_heat_range is not None and active.size:
        low, high = constraints.total_heat_range
        current = float(np.sum(heat))
        desired = float(np.clip(current, low, high))
        if current <= 1.0e-12:
            desired = max(float(low), 0.0)
            heat[active] = desired / float(active.size)
        else:
            heat *= desired / current
    return PhysicalDesign(
        module_centers=centers.astype(np.float32),
        module_present=present.astype(np.float32),
        heat_powers=heat.astype(np.float32),
    )


def validate_geometry_constraints(
    constraints: GeometryConstraints,
    context: NamedContext,
    *,
    max_modules: int,
) -> None:
    """Reject count/clearance requests impossible in the current rectangular domain."""

    values = context.as_mapping()
    radius = values["module_radius"]
    lx = values["domain_length_x"]
    ly = values["domain_length_y"]
    if constraints.module_count_max > int(max_modules):
        raise ValueError(f"module_count_max exceeds max_modules={max_modules}.")
    usable_x = lx - 2.0 * radius - constraints.inlet_clearance - constraints.outlet_clearance
    usable_y = ly - 2.0 * (radius + constraints.wall_clearance)
    if constraints.module_count_min > 0 and (usable_x < 0.0 or usable_y < 0.0):
        raise ValueError("Requested edge clearances leave no feasible module-center region.")
    max_pair = float(np.hypot(max(usable_x, 0.0), max(usable_y, 0.0)))
    if constraints.module_count_min >= 2 and constraints.minimum_center_distance > max_pair + 1.0e-7:
        raise ValueError("minimum_center_distance exceeds the usable-domain diagonal.")


def geometry_constraint_tensor(constraints: GeometryConstraints) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed eight-value raw constraint vector and optional mask."""

    heat = constraints.total_heat_range
    values = np.asarray(
        [
            constraints.module_count_min,
            constraints.module_count_max,
            constraints.minimum_center_distance,
            constraints.wall_clearance,
            constraints.inlet_clearance,
            constraints.outlet_clearance,
            0.0 if heat is None else heat[0],
            0.0 if heat is None else heat[1],
        ],
        dtype=np.float32,
    )
    mask = np.asarray([1, 1, 1, 1, 1, 1, int(heat is not None), int(heat is not None)], dtype=np.uint8)
    return values, mask


def normalize_geometry_constraints(
    constraints: GeometryConstraints,
    context: NamedContext,
    *,
    max_modules: int,
    total_heat_stats: ScalarStats,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize the eight fixed constraint values without changing masks."""

    raw, mask = geometry_constraint_tensor(constraints)
    values = context.as_mapping()
    normalized = raw.copy()
    normalized[0:2] /= float(max_modules)
    normalized[2] /= float(np.hypot(values["domain_length_x"], values["domain_length_y"]))
    normalized[3] /= values["domain_length_y"]
    normalized[4:6] /= values["domain_length_x"]
    if constraints.total_heat_range is not None:
        normalized[6:8] = total_heat_stats.normalize(raw[6:8])
    return normalized.astype(np.float32), mask


def evaluate_geometry(
    design: PhysicalDesign,
    context: NamedContext,
    constraints: GeometryConstraints,
) -> GeometryEvaluation:
    """Evaluate exact disk-edge geometry and heat constraints for one ``D``."""

    validate_geometry_constraints(constraints, context, max_modules=design.max_modules)
    values = context.as_mapping()
    lx = values["domain_length_x"]
    ly = values["domain_length_y"]
    radius = values["module_radius"]
    indices = np.flatnonzero(design.module_present > 0.5)
    centers = design.module_centers[indices]
    heat = design.heat_powers[indices]
    count = int(indices.size)
    domain_diagonal = float(np.hypot(lx, ly))
    pair_defined = count >= 2
    if pair_defined:
        delta = centers[:, None, :] - centers[None, :, :]
        distances = np.linalg.norm(delta, axis=-1)
        distances[np.eye(count, dtype=bool)] = np.inf
        minimum_pair = float(np.min(distances))
    else:
        minimum_pair = domain_diagonal
    if count:
        wall = float(np.min(np.minimum(centers[:, 1] - radius, ly - radius - centers[:, 1])))
        inlet = float(np.min(centers[:, 0] - radius))
        outlet = float(np.min(lx - radius - centers[:, 0]))
    else:
        wall = float(max(0.5 * ly - radius, 0.0))
        inlet = float(max(lx - radius, 0.0))
        outlet = inlet
    total_heat = float(np.sum(heat))
    constraint_raw, constraint_mask = geometry_constraint_tensor(constraints)
    actual = np.asarray([count, minimum_pair, wall, inlet, outlet, total_heat], dtype=np.float32)
    margins = np.asarray(
        [
            count - constraints.module_count_min,
            constraints.module_count_max - count,
            minimum_pair - constraints.minimum_center_distance,
            wall - constraints.wall_clearance,
            inlet - constraints.inlet_clearance,
            outlet - constraints.outlet_clearance,
            0.0 if constraints.total_heat_range is None else total_heat - constraints.total_heat_range[0],
            0.0 if constraints.total_heat_range is None else constraints.total_heat_range[1] - total_heat,
        ],
        dtype=np.float32,
    )
    violations = np.maximum(-margins, 0.0) * constraint_mask
    valid = bool(np.all(margins[constraint_mask > 0] >= -1.0e-6))
    return GeometryEvaluation(
        constraint_raw=constraint_raw,
        constraint_mask=constraint_mask,
        actual_raw=actual,
        margin_raw=margins,
        violation_raw=violations,
        valid=valid,
        pair_distance_defined=pair_defined,
    )


__all__ = [
    "CanonicalDesignView",
    "GEOMETRY_ACTUAL_NAMES",
    "GEOMETRY_CONSTRAINT_NAMES",
    "GeometryEvaluation",
    "canonicalize_design",
    "design_from_forward_structure",
    "decode_generated_design",
    "evaluate_geometry",
    "geometry_constraint_tensor",
    "normalize_geometry_constraints",
    "validate_geometry_constraints",
]
