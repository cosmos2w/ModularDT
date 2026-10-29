"""Noise-aware comparisons for physically indexed response evidence.

Callers compare each output role and channel separately.  A missing or
numerically unresolved label is excluded by its mask; it is never replaced
with a zero-valued target.  This module does not assign an evidence source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _on_response_grid(value: np.ndarray | float, shape: tuple[int, ...]) -> np.ndarray:
    """Accept a receiver grid with or without the final output channel."""

    array = np.asarray(value)
    if array.shape == shape[:-1]:
        array = array[..., None]
    return np.broadcast_to(array, shape)


@dataclass(frozen=True)
class ResponseComparison:
    """One output block at a stated physical unit and sampling measure."""

    output_role: str
    units: str
    observed_count: int
    resolved_sign_count: int
    weighted_mae: float | None
    weighted_rmse: float | None
    noise_aware_relative_rms: float | None
    resolved_sign_accuracy: float | None


def compare_response_block(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    output_role: str,
    units: str,
    observed_mask: np.ndarray,
    noise_floor: float | np.ndarray,
    weights: np.ndarray | None = None,
) -> ResponseComparison:
    """Compare a measured block without turning unresolved signs into errors.

    ``weights`` are physical quadrature/sampling weights, not a proxy for
    extra independent observations.  ``observed_mask`` is the common valid
    receiver mask for the stencil.  It must exclude missing labels.
    """

    truth = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("Reference and prediction must share a nonempty shape.")
    mask = _on_response_grid(np.asarray(observed_mask, dtype=bool), truth.shape)
    floor = _on_response_grid(np.asarray(noise_floor, dtype=np.float64), truth.shape)
    measure = (
        np.ones_like(truth)
        if weights is None
        else _on_response_grid(np.asarray(weights, dtype=np.float64), truth.shape)
    )
    if not np.isfinite(floor).all() or np.any(floor < 0.0):
        raise ValueError("Noise floors must be finite and nonnegative.")
    if not np.isfinite(measure).all() or np.any(measure < 0.0):
        raise ValueError("Weights must be finite and nonnegative.")
    if not np.isfinite(truth[mask]).all() or not np.isfinite(pred[mask]).all():
        raise ValueError("Observed response values must be finite.")
    valid = mask & (measure > 0.0)
    count = int(np.count_nonzero(valid))
    if count == 0:
        return ResponseComparison(output_role, units, 0, 0, None, None, None, None)

    w = measure[valid]
    w = w / w.sum()
    y = truth[valid]
    yhat = pred[valid]
    uncertainty = floor[valid]
    residual = yhat - y
    mae = float(np.sum(w * np.abs(residual)))
    rmse = float(np.sqrt(np.sum(w * np.square(residual))))
    reference_rms = float(np.sqrt(np.sum(w * np.square(y))))
    floor_rms = float(np.sqrt(np.sum(w * np.square(uncertainty))))
    relative = rmse / max(reference_rms, floor_rms, np.finfo(np.float64).eps)
    resolved = np.abs(y) > uncertainty
    sign_count = int(np.count_nonzero(resolved))
    sign_accuracy = (
        float(np.sum(w[resolved] * (np.sign(yhat[resolved]) == np.sign(y[resolved]))) / np.sum(w[resolved]))
        if sign_count
        else None
    )
    return ResponseComparison(output_role, units, count, sign_count, mae, rmse, relative, sign_accuracy)


@dataclass(frozen=True)
class CandidateRankingComparison:
    """Ranking and finite-pool regret for lower-is-better objectives."""

    candidate_count: int
    spearman: float | None
    selected_index: int
    reference_best_index: int
    selected_reference_value: float
    reference_pool_best_value: float
    finite_pool_regret: float
    improvement_sign_accuracy: float | None
    resolved_improvement_count: int


def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        stop = position + 1
        while stop < values.size and values[order[stop]] == values[order[position]]:
            stop += 1
        ranks[order[position:stop]] = (position + stop - 1) / 2.0
        position = stop
    return ranks


def compare_candidate_ranking(
    predicted_objectives: np.ndarray,
    reference_objectives: np.ndarray,
    *,
    baseline_reference_objective: float,
    baseline_predicted_objective: float,
    improvement_floor: float,
) -> CandidateRankingComparison:
    """Score one common finite pool evaluated by both model and oracle.

    A reference oracle may be physical or a frozen teacher; provenance must
    be reported by the caller.  Pool regret makes no global-optimum claim.
    """

    pred = np.asarray(predicted_objectives, dtype=np.float64).reshape(-1)
    truth = np.asarray(reference_objectives, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("Candidate pools must have the same nonzero size.")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Candidate objectives must be finite.")
    if not np.isfinite([baseline_reference_objective, baseline_predicted_objective, improvement_floor]).all():
        raise ValueError("Baseline objectives and improvement floor must be finite.")
    if improvement_floor < 0.0:
        raise ValueError("Improvement floor must be nonnegative.")
    if pred.size > 1:
        predicted_ranks = _average_tie_ranks(pred)
        reference_ranks = _average_tie_ranks(truth)
        pred_spread = float(np.std(predicted_ranks))
        truth_spread = float(np.std(reference_ranks))
        spearman = (
            float(np.corrcoef(predicted_ranks, reference_ranks)[0, 1])
            if pred_spread > 0.0 and truth_spread > 0.0
            else None
        )
    else:
        spearman = None
    chosen = int(np.argmin(pred))
    best = int(np.argmin(truth))
    actual_improvement = baseline_reference_objective - truth
    predicted_improvement = baseline_predicted_objective - pred
    resolved = np.abs(actual_improvement) > improvement_floor
    sign_count = int(np.count_nonzero(resolved))
    sign_accuracy = (
        float(np.mean(np.sign(predicted_improvement[resolved]) == np.sign(actual_improvement[resolved])))
        if sign_count
        else None
    )
    return CandidateRankingComparison(
        candidate_count=int(pred.size),
        spearman=spearman,
        selected_index=chosen,
        reference_best_index=best,
        selected_reference_value=float(truth[chosen]),
        reference_pool_best_value=float(truth[best]),
        finite_pool_regret=float(truth[chosen] - truth[best]),
        improvement_sign_accuracy=sign_accuracy,
        resolved_improvement_count=sign_count,
    )
