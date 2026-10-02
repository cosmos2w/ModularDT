"""Exact weighted squared-error accounting for aligned prediction vectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SquaredErrorAttribution:
    """Weighted squared-error terms for baseline, adapted, and accessed outputs.

    The stored squared errors use a normalized, nonnegative measure over the
    selected entries.  They are weighted mean squared errors when the supplied
    weights are quadrature weights.  Cross terms and squared perturbation terms
    are retained separately so signed reductions are visible.
    """

    selected_entry_count: int
    normalized_weight_sum: float
    baseline_squared_error: float
    adapted_full_squared_error: float
    accessed_squared_error: float
    adaptation_cross_term: float
    adaptation_squared_term: float
    adaptation_delta: float
    access_cross_term: float
    access_squared_term: float
    access_delta: float
    total_delta: float
    adaptation_closure_error: float
    access_closure_error: float
    total_closure_error: float

    @property
    def baseline_rmse(self) -> float:
        """Return the weighted root mean squared error for the baseline."""

        return float(np.sqrt(self.baseline_squared_error))

    @property
    def adapted_full_rmse(self) -> float:
        """Return the weighted root mean squared error after adaptation."""

        return float(np.sqrt(self.adapted_full_squared_error))

    @property
    def accessed_rmse(self) -> float:
        """Return the weighted root mean squared error after access restriction."""

        return float(np.sqrt(self.accessed_squared_error))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping, including derived weighted RMSEs."""

        result = asdict(self)
        result.update(
            {
                "baseline_rmse": self.baseline_rmse,
                "adapted_full_rmse": self.adapted_full_rmse,
                "accessed_rmse": self.accessed_rmse,
            }
        )
        return result


def weighted_squared_error_attribution(
    baseline_error: np.ndarray,
    adaptation_delta: np.ndarray,
    access_delta: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> SquaredErrorAttribution:
    """Attribute exact output-space squared-error changes on aligned arrays.

    ``baseline_error`` is ``F_B(full) - y``; ``adaptation_delta`` is
    ``F_G(full) - F_B(full)``; and ``access_delta`` is ``F_G(action) -
    F_G(full)``.  All three arrays must have the same shape and units.

    ``weights`` may be omitted for an equal-entry measure, have shape ``[N]``
    for one physical query weight per first-axis entry, or broadcast to the
    full array shape for an explicit component-weight convention.  The selected
    nonnegative weights are normalized to sum to one.  ``mask`` may have shape
    ``[N]`` or the full array shape.  Nonfinite entries outside the mask are
    ignored; nonfinite selected values or weights are rejected.

    This function accounts for output differences.  It does not identify the
    cause of a training change, and its deltas are not additive RMSE changes.
    """

    e_base, delta_adapt, delta_access = _aligned_arrays(
        baseline_error, adaptation_delta, access_delta
    )
    shape = e_base.shape
    if not shape or shape[0] == 0:
        raise ValueError("At least one aligned output entry is required.")

    if weights is None:
        weight_array = np.ones(shape, dtype=np.float64)
    else:
        weight_array = _broadcast_query_array(weights, shape, "weights", dtype=np.float64)
    if mask is None:
        valid = np.ones(shape, dtype=bool)
    else:
        valid = _broadcast_query_array(mask, shape, "mask", dtype=bool).copy()

    if np.any(~np.isfinite(weight_array[valid])):
        raise ValueError("Selected weights must be finite.")
    if np.any(weight_array[valid] < 0.0):
        raise ValueError("Selected weights must be nonnegative.")
    valid &= weight_array > 0.0
    if not np.any(valid):
        raise ValueError("The common mask and positive weights select no entries.")
    for name, array in (
        ("baseline_error", e_base),
        ("adaptation_delta", delta_adapt),
        ("access_delta", delta_access),
    ):
        if np.any(~np.isfinite(array[valid])):
            raise ValueError(f"Selected {name} entries must be finite.")

    selected_weights = weight_array[valid]
    weight_total = float(selected_weights.sum(dtype=np.float64))
    if not np.isfinite(weight_total) or weight_total <= 0.0:
        raise ValueError("Selected weights must have a finite positive sum.")
    normalized_weights = selected_weights / weight_total
    e_base = e_base[valid]
    delta_adapt = delta_adapt[valid]
    delta_access = delta_access[valid]
    e_adapt = e_base + delta_adapt
    e_access = e_adapt + delta_access

    baseline_sse = _weighted_square(e_base, normalized_weights)
    adapted_sse = _weighted_square(e_adapt, normalized_weights)
    accessed_sse = _weighted_square(e_access, normalized_weights)
    adapt_cross = 2.0 * _weighted_dot(e_base, delta_adapt, normalized_weights)
    adapt_square = _weighted_square(delta_adapt, normalized_weights)
    adapt_delta = adapt_cross + adapt_square
    access_cross = 2.0 * _weighted_dot(e_adapt, delta_access, normalized_weights)
    access_square = _weighted_square(delta_access, normalized_weights)
    access_error_delta = access_cross + access_square
    total_delta = adapt_delta + access_error_delta

    return SquaredErrorAttribution(
        selected_entry_count=int(valid.sum()),
        normalized_weight_sum=float(normalized_weights.sum(dtype=np.float64)),
        baseline_squared_error=baseline_sse,
        adapted_full_squared_error=adapted_sse,
        accessed_squared_error=accessed_sse,
        adaptation_cross_term=adapt_cross,
        adaptation_squared_term=adapt_square,
        adaptation_delta=adapt_delta,
        access_cross_term=access_cross,
        access_squared_term=access_square,
        access_delta=access_error_delta,
        total_delta=total_delta,
        adaptation_closure_error=(adapted_sse - baseline_sse) - adapt_delta,
        access_closure_error=(accessed_sse - adapted_sse) - access_error_delta,
        total_closure_error=(accessed_sse - baseline_sse) - total_delta,
    )


def _aligned_arrays(
    baseline_error: np.ndarray,
    adaptation_delta: np.ndarray,
    access_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (baseline_error, adaptation_delta, access_delta)
    )
    if arrays[0].shape != arrays[1].shape or arrays[0].shape != arrays[2].shape:
        raise ValueError("Baseline, adaptation, and access arrays must be aligned.")
    if arrays[0].ndim == 0:
        raise ValueError("Output vectors must have at least one dimension.")
    return arrays


def _broadcast_query_array(
    value: np.ndarray,
    shape: tuple[int, ...],
    name: str,
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 1 and array.shape == (shape[0],):
        array = array.reshape((shape[0],) + (1,) * (len(shape) - 1))
    try:
        return np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError(f"{name} must be [N] or broadcastable to {shape}.") from exc


def _weighted_dot(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * left * right, dtype=np.float64))


def _weighted_square(value: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * np.square(value), dtype=np.float64))


__all__ = ["SquaredErrorAttribution", "weighted_squared_error_attribution"]
