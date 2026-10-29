"""ThermalChannel compact mechanism-plan schema derived from canonical HONF.

Physical design ``D`` supplies physical heat and module presence, context ``c``
supplies domain scales, request ``R`` is not used during extraction, compact
plan ``G`` is the intended fixed-edge representation, and ``G_hat`` is the same
representation extracted after frozen forward verification.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from honf_forward_core.evaluation.hypergraph_plan import validate_hypergraph_plan
from honf_inverse_core.contracts import CompactPlan, NamedContext, PhysicalDesign


COMPACT_PLAN_SCHEMA_NAME = "thermalchannel_compact_mechanism_plan"
COMPACT_PLAN_SCHEMA_VERSION = 1
CANONICAL_FULL_PLAN_SCHEMA_VERSION = 2
STRENGTH_EPS = 1.0e-6
ACTIVE_STRENGTH_THRESHOLD = 0.05

COMPACT_PLAN_FEATURE_NAMES = (
    "active",
    "source_x",
    "source_y",
    "region_x",
    "region_y",
    "module_mass",
    "environment_mass",
    "strength",
    "environment_region_scale_x",
    "environment_region_scale_y",
    "edge_heat_fraction",
    "edge_module_source_fraction",
)

INDEPENDENT_CONTINUOUS_FEATURE_INDICES = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11)


def _as_array(plan: Mapping[str, Any], name: str, *, ndim: int | None = None) -> np.ndarray:
    if name not in plan:
        raise ValueError(f"Canonical full plan is missing {name!r}.")
    value = np.asarray(plan[name], dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"Canonical full plan {name!r} contains non-finite values.")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"Canonical full plan {name!r} must have ndim={ndim}; got {value.shape}.")
    return value


def _canonical_permutation(features: np.ndarray) -> np.ndarray:
    active = features[:, 0] > 0.5
    order = sorted(
        range(features.shape[0]),
        key=lambda index: (
            0 if active[index] else 1,
            float(features[index, 1]),
            float(features[index, 2]),
            float(features[index, 3]),
            float(features[index, 4]),
            -float(features[index, 7]),
        ),
    )
    return np.asarray(order, dtype=np.int64)


def normalize_compact_plan(raw: Any, context: NamedContext) -> np.ndarray:
    """Apply the analytic compact-plan normalization."""

    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(COMPACT_PLAN_FEATURE_NAMES):
        raise ValueError("Compact plan must have shape [K,12].")
    if not np.isfinite(values).all():
        raise ValueError("Compact plan contains non-finite values.")
    domain = context.as_mapping()
    normalized = values.copy()
    normalized[:, [1, 3, 8]] /= domain["domain_length_x"]
    normalized[:, [2, 4, 9]] /= domain["domain_length_y"]
    return normalized.astype(np.float32)


def denormalize_compact_plan(normalized: Any, context: NamedContext) -> np.ndarray:
    """Invert the analytic compact-plan normalization."""

    values = np.asarray(normalized, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(COMPACT_PLAN_FEATURE_NAMES):
        raise ValueError("Normalized compact plan must have shape [K,12].")
    if not np.isfinite(values).all():
        raise ValueError("Normalized compact plan contains non-finite values.")
    domain = context.as_mapping()
    raw = values.copy()
    raw[:, [1, 3, 8]] *= domain["domain_length_x"]
    raw[:, [2, 4, 9]] *= domain["domain_length_y"]
    return raw.astype(np.float32)


def validate_compact_plan(raw: Any, context: NamedContext, *, atol: float = 2.0e-4) -> None:
    """Validate compact schema values, derived fields, simplex sums, and order."""

    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(COMPACT_PLAN_FEATURE_NAMES) or values.shape[0] <= 0:
        raise ValueError("Compact plan must have non-empty shape [K,12].")
    if not np.isfinite(values).all():
        raise ValueError("Compact plan contains non-finite values.")
    active = values[:, 0]
    if not np.isin(active, [0.0, 1.0]).all():
        raise ValueError("Compact plan active indicators must be binary.")
    expected_active = (values[:, 7] > ACTIVE_STRENGTH_THRESHOLD).astype(np.float64)
    if not np.array_equal(active, expected_active):
        raise ValueError("Compact plan active indicators disagree with strength > 0.05.")
    domain = context.as_mapping()
    for column in (1, 3):
        if np.any(values[:, column] < -atol) or np.any(values[:, column] > domain["domain_length_x"] + atol):
            raise ValueError(f"Compact plan x-coordinate column {column} is outside the domain.")
    for column in (2, 4):
        if np.any(values[:, column] < -atol) or np.any(values[:, column] > domain["domain_length_y"] + atol):
            raise ValueError(f"Compact plan y-coordinate column {column} is outside the domain.")
    if np.any(values[:, 8:10] < -atol):
        raise ValueError("Compact plan environment-region scales must be nonnegative.")
    for column in (5, 6, 7, 10, 11):
        if np.any(values[:, column] < -atol) or np.any(values[:, column] > 1.0 + atol):
            raise ValueError(f"Compact plan unit feature column {column} is outside [0,1].")
    for column in (5, 6, 10, 11):
        if not np.isclose(np.sum(values[:, column]), 1.0, atol=atol):
            raise ValueError(f"Compact plan simplex feature column {column} does not sum to one.")
    expected_strength = np.sqrt(values[:, 5] * values[:, 6] + STRENGTH_EPS)
    if not np.allclose(values[:, 7], expected_strength, atol=atol, rtol=atol):
        raise ValueError("Compact plan strength disagrees with the forward mass-derived formula.")
    if not np.array_equal(_canonical_permutation(values), np.arange(values.shape[0])):
        raise ValueError("Compact plan edges are not in canonical order.")


def extract_compact_plan(
    full_plan: Mapping[str, Any],
    design: PhysicalDesign,
    context: NamedContext,
) -> CompactPlan:
    """Derive compact schema v1 from the current canonical full plan."""

    plan = {key: np.asarray(value) for key, value in full_plan.items()}
    validate_hypergraph_plan(plan)
    if int(np.asarray(plan["schema_version"])) != CANONICAL_FULL_PLAN_SCHEMA_VERSION:
        raise ValueError("Canonical full-plan schema version mismatch.")
    A_mh = _as_array(plan, "A_mh", ndim=2)
    A_eh = _as_array(plan, "A_eh", ndim=2)
    source = _as_array(plan, "hyper_source_coords", ndim=2)
    region = _as_array(plan, "hyper_region_coords", ndim=2)
    module_mass = _as_array(plan, "hyper_module_mass").reshape(-1)
    environment_mass = _as_array(plan, "hyper_env_mass").reshape(-1)
    strength = _as_array(plan, "hyper_strength").reshape(-1)
    active = _as_array(plan, "active_hyperedge_mask").reshape(-1)
    env_coords = _as_array(plan, "env_coords", ndim=2)
    num_edges = strength.shape[0]
    if A_mh.shape != (design.max_modules, num_edges) or A_eh.shape[1] != num_edges:
        raise ValueError("Canonical assignments do not align with design slots/hyperedges.")
    if source.shape != (num_edges, 2) or region.shape != (num_edges, 2) or env_coords.shape != (A_eh.shape[0], 2):
        raise ValueError("Canonical coordinate shapes do not align with hyperedges/environment assignments.")
    plan_present = _as_array(plan, "module_present").reshape(-1) > 0.5
    design_present = design.module_present > 0.5
    if not np.array_equal(plan_present, design_present):
        raise ValueError("Canonical full-plan module_present does not match physical design D.")

    column_weights = A_eh / np.maximum(np.sum(A_eh, axis=0, keepdims=True), 1.0e-12)
    centered = env_coords[:, None, :] - region[None, :, :]
    variance = np.sum(column_weights[..., None] * centered**2, axis=0)
    region_scale = np.sqrt(np.maximum(variance, 0.0))

    abs_heat = np.abs(design.heat_powers) * design.module_present
    zero_total_heat = bool(float(np.sum(abs_heat)) <= 1.0e-12)
    if zero_total_heat:
        heat_mass = np.sum(A_mh * design.module_present[:, None], axis=0)
    else:
        heat_mass = np.sum(A_mh * abs_heat[:, None], axis=0)
    heat_fraction = heat_mass / max(float(np.sum(heat_mass)), 1.0e-12)

    active_indices = np.flatnonzero(design_present)
    hard_fraction = np.zeros(num_edges, dtype=np.float64)
    if active_indices.size:
        owners = np.argmax(A_mh[active_indices], axis=1)
        hard_fraction = np.bincount(owners, minlength=num_edges).astype(np.float64) / float(active_indices.size)

    raw = np.column_stack(
        [
            active,
            source[:, 0],
            source[:, 1],
            region[:, 0],
            region[:, 1],
            module_mass,
            environment_mass,
            strength,
            region_scale[:, 0],
            region_scale[:, 1],
            heat_fraction,
            hard_fraction,
        ]
    ).astype(np.float32)
    validate_compact_plan(raw, context)
    normalized = normalize_compact_plan(raw, context)
    round_trip = denormalize_compact_plan(normalized, context)
    if not np.allclose(round_trip, raw, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("Compact-plan analytic normalization failed its round-trip check.")
    return CompactPlan(
        raw=raw,
        normalized=normalized,
        feature_names=COMPACT_PLAN_FEATURE_NAMES,
        schema_name=COMPACT_PLAN_SCHEMA_NAME,
        schema_version=COMPACT_PLAN_SCHEMA_VERSION,
        metadata={
            "canonical_full_plan_schema_version": CANONICAL_FULL_PLAN_SCHEMA_VERSION,
            "zero_total_heat": zero_total_heat,
            "num_edges": int(num_edges),
            "independent_continuous_feature_indices": list(INDEPENDENT_CONTINUOUS_FEATURE_INDICES),
        },
    )


__all__ = [
    "ACTIVE_STRENGTH_THRESHOLD",
    "CANONICAL_FULL_PLAN_SCHEMA_VERSION",
    "COMPACT_PLAN_FEATURE_NAMES",
    "COMPACT_PLAN_SCHEMA_NAME",
    "COMPACT_PLAN_SCHEMA_VERSION",
    "INDEPENDENT_CONTINUOUS_FEATURE_INDICES",
    "STRENGTH_EPS",
    "denormalize_compact_plan",
    "extract_compact_plan",
    "normalize_compact_plan",
    "validate_compact_plan",
]
