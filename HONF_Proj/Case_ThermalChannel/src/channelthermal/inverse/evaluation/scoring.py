"""Exact request and `G`/`G_hat` scoring for verified candidates.

Physical design ``D`` has already been checked under context ``c``. Request
``R`` is scored term by term in train-normalized units; planned compact ``G``
and realized ``G_hat`` use the same schema. Priority is reported, while weight
is the sole multiplier.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from honf_inverse_core.contracts import FunctionalValue
from honf_inverse_core.normalization import ScalarStats
from honf_inverse_core.models.matching import match_tokens
from honf_inverse_core.request_schema import StructuredRequest

from channelthermal.inverse.compact_plan import INDEPENDENT_CONTINUOUS_FEATURE_INDICES


def evaluate_request_satisfaction(
    request: StructuredRequest,
    functionals: Mapping[str, FunctionalValue],
    normalization: Mapping[str, ScalarStats],
) -> tuple[list[dict[str, object]], float, bool, np.ndarray]:
    terms: list[dict[str, object]] = []
    weighted_total = 0.0
    weight_total = 0.0
    residual_vector = np.zeros((4,), dtype=np.float32)
    all_satisfied = True
    for slot, token in enumerate(token for token in request.tokens if token.active):
        functional = functionals[token.request_type]
        stats = normalization[token.request_type]
        value_normalized = float(stats.normalize(functional.value))
        tolerance = float(stats.normalize_width(token.tolerance))
        if token.relation == "upper_bound":
            threshold = float(stats.normalize(token.target)) + tolerance
            signed_residual = max(value_normalized - threshold, 0.0)
        elif token.relation == "lower_bound":
            threshold = float(stats.normalize(token.target)) - tolerance
            signed_residual = -max(threshold - value_normalized, 0.0)
        elif token.relation == "target_range":
            low, high = stats.normalize(token.target_range)
            below = max(float(low) - tolerance - value_normalized, 0.0)
            above = max(value_normalized - float(high) - tolerance, 0.0)
            signed_residual = above - below
        else:
            # Schema-v1 minimize has no arbitrary target. Zero violation means
            # at-or-better than the inverse-train mean; raw value remains visible.
            signed_residual = max(value_normalized, 0.0)
        residual = abs(signed_residual)
        satisfied = residual <= 1.0e-8
        all_satisfied = all_satisfied and satisfied
        weighted_total += token.weight * residual
        weight_total += token.weight
        residual_vector[slot] = signed_residual
        terms.append(
            {
                "request_type": token.request_type,
                "relation": token.relation,
                "value_raw": float(functional.value),
                "value_normalized": value_normalized,
                "violation_normalized": float(residual),
                "satisfied": satisfied,
                "priority": token.priority,
                "weight": token.weight,
                "region": token.region,
            }
        )
    return terms, weighted_total / max(weight_total, 1.0e-8), all_satisfied, residual_vector


def compact_plan_distance(
    planned: np.ndarray,
    realized: np.ndarray,
    *,
    matching_mode: str = "canonical",
) -> float:
    planned = np.asarray(planned, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    if planned.shape != realized.shape or planned.ndim != 2 or planned.shape[1] != 12:
        raise ValueError("Plan distance requires two aligned [K,12] normalized plans.")
    if matching_mode != "canonical":
        aligned = match_tokens(
            torch.as_tensor(planned[None], dtype=torch.float32),
            torch.as_tensor(realized[None], dtype=torch.float32),
            method=matching_mode,
        )
        realized = aligned[0].detach().cpu().numpy().astype(np.float64)
    continuous = np.mean(np.square(planned[:, INDEPENDENT_CONTINUOUS_FEATURE_INDICES] - realized[:, INDEPENDENT_CONTINUOUS_FEATURE_INDICES]))
    activity = np.mean(np.abs(planned[:, 0] - realized[:, 0]))
    return float(np.sqrt(continuous) + 0.25 * activity)


__all__ = ["compact_plan_distance", "evaluate_request_satisfaction"]
