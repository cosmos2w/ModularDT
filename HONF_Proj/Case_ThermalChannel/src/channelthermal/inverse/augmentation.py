"""Deterministic feasible request augmentation for the inverse dataset.

Physical design ``D`` and context ``c`` determine exact realized values.
Request ``R`` is sampled around those values, compact plan ``G`` remains the
case target, and realized plan ``G_hat`` is unaffected by augmentation. Every
random stream is keyed by case and variant, not traversal order.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Mapping

import numpy as np

from honf_inverse_core.contracts import FunctionalValue, NamedContext, PhysicalDesign
from honf_inverse_core.normalization import ScalarStats
from honf_inverse_core.request_schema import GeometryConstraints, StructuredRequest

from .geometry import evaluate_geometry
from .request import make_request_codec
from .vocabulary import NONREGIONAL_REQUEST_TYPES, REGIONAL_REQUEST_TYPES, REQUEST_TYPES


RELATION_PROBABILITIES = {
    "upper_bound": 0.45,
    "target_range": 0.35,
    "lower_bound": 0.10,
    "minimize": 0.10,
}


def stable_variant_seed(global_seed: int, case_id: str, variant_index: int) -> int:
    """Return a stable unsigned 64-bit seed for one case/request variant."""

    payload = f"{int(global_seed)}\0{case_id}\0{int(variant_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False)


@dataclass(frozen=True)
class TokenRecipe:
    request_type: str
    relation: str
    realized_value: float
    region: tuple[float, float, float, float] | None
    slack_low_factor: float
    slack_high_factor: float
    tolerance_factor: float
    priority: int


@dataclass(frozen=True)
class RequestRecipe:
    variant_index: int
    variant_seed: int
    tokens: tuple[TokenRecipe, ...]
    geometry_constraints: GeometryConstraints


RegionalEvaluator = Callable[[str, tuple[float, float, float, float]], FunctionalValue]


def _sample_region(rng: np.random.Generator) -> tuple[float, float, float, float]:
    width = float(rng.uniform(0.15, 0.50))
    height = float(rng.uniform(0.20, 0.75))
    x0 = float(rng.uniform(0.0, 1.0 - width))
    y0 = float(rng.uniform(0.0, 1.0 - height))
    return (x0, y0, x0 + width, y0 + height)


def _functional_names(
    rng: np.random.Generator,
    count: int,
    *,
    forced_regional_name: str | None = None,
) -> tuple[str, ...]:
    """Sample distinct names while enforcing at most one regional token."""

    include_regional = bool(forced_regional_name is not None or rng.random() < 0.45)
    regional_count = int(include_regional and count > 0)
    nonregional_count = count - regional_count
    nonregional = rng.choice(NONREGIONAL_REQUEST_TYPES, size=nonregional_count, replace=False).tolist()
    if forced_regional_name is not None:
        regional = [forced_regional_name]
    else:
        regional = (
            rng.choice(REGIONAL_REQUEST_TYPES, size=regional_count, replace=False).tolist()
            if regional_count
            else []
        )
    names = nonregional + regional
    rng.shuffle(names)
    return tuple(str(value) for value in names)


def _feasible_geometry(
    design: PhysicalDesign,
    context: NamedContext,
    rng: np.random.Generator,
    *,
    total_heat_probability: float,
) -> GeometryConstraints:
    unconstrained = GeometryConstraints(0, design.max_modules, 0.0, 0.0, 0.0, 0.0, None)
    actual = evaluate_geometry(design, context, unconstrained).actual_raw
    count = design.module_count
    total_heat = float(actual[5])
    heat_range = None
    if rng.random() < total_heat_probability:
        scale = max(abs(total_heat) * float(rng.uniform(0.05, 0.20)), 1.0e-4)
        heat_range = (total_heat - scale, total_heat + scale)
    return GeometryConstraints(
        module_count_min=max(0, count - int(rng.integers(0, 2))),
        module_count_max=min(design.max_modules, count + int(rng.integers(0, 2))),
        minimum_center_distance=(
            max(0.0, float(actual[1])) * float(rng.uniform(0.40, 0.85)) if count >= 2 else 0.0
        ),
        wall_clearance=max(0.0, float(actual[2])) * float(rng.uniform(0.40, 0.85)),
        inlet_clearance=max(0.0, float(actual[3])) * float(rng.uniform(0.40, 0.85)),
        outlet_clearance=max(0.0, float(actual[4])) * float(rng.uniform(0.40, 0.85)),
        total_heat_range=heat_range,
    )


def generate_request_recipe(
    *,
    global_seed: int,
    case_id: str,
    variant_index: int,
    nonregional_values: Mapping[str, FunctionalValue],
    regional_evaluator: RegionalEvaluator,
    design: PhysicalDesign,
    context: NamedContext,
    min_active_tokens: int = 2,
    max_active_tokens: int = 4,
    total_heat_probability: float = 0.5,
) -> RequestRecipe:
    """Generate one deterministic pre-normalization request recipe."""

    if not (2 <= min_active_tokens <= max_active_tokens <= 4):
        raise ValueError("Schema v1 augmentation requires 2 <= min <= max <= 4 active tokens.")
    seed = stable_variant_seed(global_seed, case_id, variant_index)
    rng = np.random.default_rng(seed)
    count = int(rng.integers(min_active_tokens, max_active_tokens + 1))
    forced_regional = (
        REGIONAL_REQUEST_TYPES[(variant_index // 4) % len(REGIONAL_REQUEST_TYPES)]
        if variant_index % 4 == 0
        else None
    )
    names = _functional_names(rng, count, forced_regional_name=forced_regional)
    relation_names = tuple(RELATION_PROBABILITIES)
    relation_p = np.asarray(tuple(RELATION_PROBABILITIES.values()), dtype=np.float64)
    recipes: list[TokenRecipe] = []
    for token_index, name in enumerate(names):
        region = None
        if name in REGIONAL_REQUEST_TYPES:
            value = None
            for _ in range(64):
                candidate = _sample_region(rng)
                try:
                    result = regional_evaluator(name, candidate)
                except ValueError:
                    continue
                if result.valid and result.selected_count >= 16:
                    region = candidate
                    value = float(result.value)
                    break
            if value is None or region is None:
                raise ValueError(f"Could not sample a nonempty region for case {case_id!r}.")
        else:
            result = nonregional_values[name]
            if not result.valid:
                raise ValueError(f"Nonregional functional {name!r} is invalid for case {case_id!r}.")
            value = float(result.value)
        sampled_relation = str(rng.choice(relation_names, p=relation_p))
        if token_index == 0:
            sampled_relation = relation_names[variant_index % len(relation_names)]
        recipes.append(
            TokenRecipe(
                request_type=name,
                relation=sampled_relation,
                realized_value=value,
                region=region,
                slack_low_factor=float(rng.uniform(0.5, 2.0)),
                slack_high_factor=float(rng.uniform(0.5, 2.0)),
                tolerance_factor=float(rng.uniform(0.0, 0.25)),
                priority=int(rng.integers(1, 4)),
            )
        )
    geometry = _feasible_geometry(
        design,
        context,
        rng,
        total_heat_probability=float(total_heat_probability),
    )
    return RequestRecipe(variant_index, seed, tuple(recipes), geometry)


def materialize_request(
    recipe: RequestRecipe,
    normalization: Mapping[str, ScalarStats],
) -> StructuredRequest:
    """Convert a recipe to a strict normalized request-schema-v1 object."""

    tokens: list[dict[str, object]] = []
    for recipe_token in recipe.tokens:
        stats = normalization[recipe_token.request_type]
        value = float(recipe_token.realized_value)
        scale = max(0.05 * float(stats.std), 0.01 * abs(value), 1.0e-6)
        low_slack = scale * recipe_token.slack_low_factor
        high_slack = scale * recipe_token.slack_high_factor
        relation = recipe_token.relation
        target: float | None
        target_range: list[float] | None
        tolerance: float
        if relation == "upper_bound":
            target, target_range = value + high_slack, None
            tolerance = recipe_token.tolerance_factor * high_slack
        elif relation == "lower_bound":
            target, target_range = value - low_slack, None
            tolerance = recipe_token.tolerance_factor * low_slack
        elif relation == "target_range":
            target_range = [value - low_slack, value + high_slack]
            target = 0.5 * (target_range[0] + target_range[1])
            tolerance = recipe_token.tolerance_factor * min(low_slack, high_slack)
        elif relation == "minimize":
            target, target_range, tolerance = None, None, 0.0
        else:  # pragma: no cover - recipe construction owns the relation set.
            raise ValueError(f"Unsupported recipe relation: {relation}")
        tokens.append(
            {
                "request_type": recipe_token.request_type,
                "relation": relation,
                "target": target,
                "target_range": target_range,
                "tolerance": tolerance,
                "priority": recipe_token.priority,
                "region": recipe_token.region,
                "active": True,
            }
        )
    payload = {
        "schema_name": "thermalchannel_inverse_request",
        "schema_version": 1,
        "tokens": tokens,
        "geometry_constraints": recipe.geometry_constraints.to_dict(),
    }
    return make_request_codec(normalization).parse(payload)


def realized_values_for_request(recipe: RequestRecipe) -> np.ndarray:
    """Return schema-padded realized functional values for audit storage."""

    result = np.zeros((4,), dtype=np.float32)
    for index, token in enumerate(recipe.tokens):
        result[index] = token.realized_value
    return result


__all__ = [
    "RELATION_PROBABILITIES",
    "RequestRecipe",
    "TokenRecipe",
    "generate_request_recipe",
    "materialize_request",
    "realized_values_for_request",
    "stable_variant_seed",
]
