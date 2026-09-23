"""Bounded Run-1503 adaptive hyperedge-opening diagnostics.

The frozen mode in this module is intentionally evaluation-only.  It reads the
Run-1502 epoch-500 organization (or a future Run-1503 prediction payload),
reports group geometry and group-major work, and leaves accuracy and training
decisions to the caller.  In particular, support counts are not executor work:
actual rows are copied only from an explicit phase ledger.

The pure helpers are useful before the Run-1503 core exists.  They accept a
small synthetic payload in tests and accept both the current Run-1502 aliases
and the planned ``adaptive_hyperedge_*`` auxiliary keys once the new reader is
available.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import inspect
import json
import math
import resource
import sys
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep the Run-1502 panel fixed.  The first two are the established
# representative anchors; the remaining four provide a small, stable
# module-count/geometry panel without turning this diagnostic into a population
# evaluation.
REPRESENTATIVE_CASE_IDS = ("0273", "0653")
MODULE_PANEL_CASE_IDS = ("0277", "0291", "0680", "0281")
DEFAULT_CASE_IDS = tuple(dict.fromkeys(REPRESENTATIVE_CASE_IDS + MODULE_PANEL_CASE_IDS))
CASE_SELECTION_REASONS = {
    "0273": "representative anchor: three active modules",
    "0653": "representative anchor: five active modules",
    "0277": "module-count panel: three active modules/intermediate spacing",
    "0291": "module-count panel: five active modules/crowded interior",
    "0680": "module-count panel: ten active modules/crowded near-wall",
    "0281": "module-count panel: three active modules/separated near-wall",
}

DEFAULT_QUERY_COUNT = 1024
DEFAULT_QUERY_BATCH_SIZE = 1024
DEFAULT_RECEIVER_CHUNK_SIZE = 2048
DEFAULT_WARMUPS = 2
DEFAULT_REPETITIONS = 5
DEFAULT_ASSIGNMENT_TOLERANCE = 0.0
DEFAULT_MASS_EPSILON = 1.0e-12
DEFAULT_MEASURE_EPSILON = 1.0e-12
DEFAULT_GIANT_MASS_FRACTION = 0.5
DEFAULT_GIANT_SOURCE_FRACTION = 0.5

# The diagnostic looks up these aliases in order.  The first spelling in each
# tuple is the explicit Run-1503 contract; the remaining spellings are the
# current Run-1502 prepared-state/routing names and a few established generic
# names used by prior evidence tools.
AUX_ALIASES: dict[str, tuple[str, ...]] = {
    "environment_membership": (
        "adaptive_hyperedge_environment_membership",
        "adaptive_hyperedge_environment_incidence",
        "sparse_incidence_environment_incidence",
        "sparse_incidence_environment_membership",
        "group_control_environment_incidence",
        "group_control_environment_membership",
        "environment_membership",
        "A_e",
        "A_E",
    ),
    "environment_measure": (
        "adaptive_hyperedge_environment_measure",
        "adaptive_hyperedge_source_measure",
        "sparse_incidence_environment_measure",
        "group_control_environment_measure",
        "environment_measure",
        "omega_e",
        "env_weights",
    ),
    "environment_coordinates": (
        "adaptive_hyperedge_environment_coordinates",
        "adaptive_hyperedge_environment_coords",
        "sparse_incidence_environment_coordinates",
        "sparse_incidence_environment_coords",
        "environment_coordinates",
        "environment_coords",
        "env_coords",
    ),
    "query_assignment": (
        "adaptive_hyperedge_query_sparse_assignment",
        "adaptive_hyperedge_fine_open_assignment",
        "adaptive_hyperedge_query_assignment",
        "group_control_adaptive_environment_alpha",
        "sparse_incidence_query_routing",
        "group_control_query_routing",
        "query_assignment",
        "alpha_qk",
    ),
    "query_logits": (
        "adaptive_hyperedge_query_logits",
        "sparse_incidence_query_logits",
        "group_control_query_logits",
        "query_logits",
        "ell_qk",
    ),
    "coarse_assignment": (
        "adaptive_hyperedge_query_coarse_assignment",
        "adaptive_hyperedge_coarse_assignment",
        "adaptive_hyperedge_group_mixture",
        "group_control_adaptive_environment_p",
        "coarse_group_assignment",
        "coarse_group_routing",
        "p_qk",
    ),
    "opening_blend": (
        "adaptive_hyperedge_opening_blend",
        "adaptive_hyperedge_opening",
        "group_control_adaptive_opening_blend",
        "fine_opening_blend",
        "opening_blend",
        "o_qk",
    ),
    "fine_open_mask": (
        "adaptive_hyperedge_fine_open_mask",
        "adaptive_hyperedge_open_mask",
        "fine_open_mask",
        "opened_group_mask",
    ),
    "coarse_available": (
        "adaptive_hyperedge_coarse_available",
        "adaptive_hyperedge_coarse_global_available",
        "coarse_group_available",
        "coarse_global_available",
    ),
    "phase_ledger": (
        "adaptive_hyperedge_phase_ledger",
        "adaptive_hyperedge_executor_phase_ledger",
        "group_control_phase_ledger",
        "phase_ledger",
    ),
    "group_mass": (
        "adaptive_hyperedge_group_mass",
        "adaptive_hyperedge_environment_mass",
        "group_control_adaptive_environment_mass",
        "sparse_incidence_environment_mass",
        "group_control_environment_mass",
        "environment_mass",
    ),
    "group_centroid": (
        "adaptive_hyperedge_group_centroid",
        "adaptive_hyperedge_environment_centroid",
        "group_control_adaptive_environment_centroids",
        "sparse_incidence_environment_centres",
        "sparse_incidence_environment_centers",
        "group_control_environment_centres",
        "group_control_environment_centers",
        "environment_centres",
        "environment_centers",
    ),
    "group_radius": (
        "adaptive_hyperedge_group_radius",
        "adaptive_hyperedge_environment_radius",
        "group_control_adaptive_environment_radius_sq",
        "environment_group_radius",
    ),
    "fine_group_rows": (
        "adaptive_hyperedge_fine_group_rows",
        "group_control_adaptive_fine_group_rows",
        "fine_group_rows",
    ),
    "fine_rows": (
        "adaptive_hyperedge_fine_rows",
        "group_control_adaptive_fine_rows",
        "group_control_environment_fine_rows_forward",
        "group_control_environment_fine_rows",
        "fine_rows",
    ),
    "fine_rows_per_query": (
        "adaptive_hyperedge_fine_rows_per_query",
        "group_control_adaptive_fine_rows_per_query",
        "fine_rows_per_query",
    ),
    "coarse_rows": (
        "adaptive_hyperedge_coarse_rows",
        "group_control_adaptive_coarse_rows",
        "coarse_rows",
    ),
    "full_rectangle_rows": (
        "adaptive_hyperedge_full_rectangle_rows",
        "group_control_adaptive_full_rectangle_rows",
        "full_rectangle_rows",
    ),
    "fine_work_ratio": (
        "adaptive_hyperedge_fine_work_ratio",
        "group_control_adaptive_fine_work_ratio",
        "fine_work_ratio",
    ),
    "padded_rows": (
        "adaptive_hyperedge_fine_rows_padded",
        "group_control_environment_fine_rows_padded",
        "group_control_environment_padded_rows",
        "padded_rows",
    ),
    "recomputed_rows": (
        "adaptive_hyperedge_fine_rows_recompute",
        "group_control_environment_fine_rows_recompute",
        "group_control_environment_checkpoint_recomputations",
        "recomputed_rows",
    ),
    "support_pairs": (
        "adaptive_hyperedge_support_pairs",
        "group_control_environment_unique_pairs",
        "group_control_environment_support_rows",
        "support_pairs",
    ),
    "coarse_contribution": (
        "adaptive_hyperedge_coarse_contribution",
        "group_control_adaptive_coarse_contribution",
        "coarse_contribution",
    ),
    "fine_contribution": (
        "adaptive_hyperedge_fine_contribution",
        "group_control_adaptive_fine_contribution",
        "fine_contribution",
    ),
}

_CONTAINER_KEYS = (
    "adaptive_hyperedge_aux",
    "adaptive_hyperedge_debug",
    "adaptive_hyperedge_diagnostics",
    "interaction_aux",
    "routing_maps",
    "routing_aux",
    "organizer_aux",
    "base_organizer_aux",
    "prepared_state",
    "_prepared_state",
    "prepared",
    "encoded",
    "backend_state",
    "group_control_state",
    "controls",
    "aux",
    "diagnostics",
)


class Run1503DiagnosticError(RuntimeError):
    """Raised when a payload cannot satisfy the structural diagnostic ABI."""


def _jsonable(value: Any) -> Any:
    """Convert small diagnostic values to JSON without dumping large arrays."""

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size <= 256:
            return value.tolist()
        finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else np.asarray([])
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.size),
            "min": None if finite.size == 0 else float(np.min(finite)),
            "max": None if finite.size == 0 else float(np.max(finite)),
            "mean": None if finite.size == 0 else float(np.mean(finite)),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _array(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise Run1503DiagnosticError(f"cannot convert {name!r} to NumPy") from exc
    if result.dtype == object:
        raise Run1503DiagnosticError(f"{name!r} has object dtype")
    return result


def _iter_containers(payload: Any, *, _seen: set[int] | None = None, _depth: int = 0):
    """Yield shallow mappings/objects without recursively walking model state."""

    seen = set() if _seen is None else _seen
    if payload is None or _depth > 8:
        return
    identity = id(payload)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(payload, Mapping):
        yield payload
        for key in _CONTAINER_KEYS:
            if key in payload:
                yield from _iter_containers(payload[key], _seen=seen, _depth=_depth + 1)
        return
    if isinstance(payload, (str, bytes, int, float, bool, np.ndarray)):
        return
    attrs = getattr(payload, "__dict__", None)
    if isinstance(attrs, Mapping):
        yield attrs
        for key in _CONTAINER_KEYS:
            if key in attrs:
                yield from _iter_containers(attrs[key], _seen=seen, _depth=_depth + 1)
    for key in _CONTAINER_KEYS:
        try:
            value = getattr(payload, key)
        except (AttributeError, RuntimeError, TypeError):  # pragma: no cover - proxy objects
            continue
        if value is not None:
            yield from _iter_containers(value, _seen=seen, _depth=_depth + 1)


def _lookup(payload: Any, name: str, *, required: bool = False) -> Any:
    aliases = AUX_ALIASES.get(name, (name,))
    for container in _iter_containers(payload):
        for alias in aliases:
            if alias in container and container[alias] is not None:
                return container[alias]
    if required:
        raise Run1503DiagnosticError(
            f"missing {name}; accepted keys: {', '.join(aliases)}"
        )
    return None


def _case_array(value: Any, *, name: str, ndim: int, case_index: int = 0) -> np.ndarray:
    array = _array(value, name=name)
    if array.ndim == ndim + 1:
        if not 0 <= int(case_index) < int(array.shape[0]):
            raise Run1503DiagnosticError(
                f"{name} case index {case_index} is outside batch size {array.shape[0]}"
            )
        array = array[int(case_index)]
    if array.ndim != ndim:
        raise Run1503DiagnosticError(
            f"{name} must have rank {ndim} or batched rank {ndim + 1}, got shape {array.shape}"
        )
    return np.asarray(array)


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {key: 0.0 for key in ("mean", "median", "p95", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _finite_nonnegative(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(array).all():
        raise Run1503DiagnosticError(f"{name} contains non-finite values")
    if np.any(array < 0.0):
        raise Run1503DiagnosticError(f"{name} contains negative values")


def group_geometry(
    environment_membership: Any,
    environment_measure: Any,
    environment_coordinates: Any,
    *,
    mass_epsilon: float = DEFAULT_MASS_EPSILON,
    measure_epsilon: float = DEFAULT_MEASURE_EPSILON,
    assignment_tolerance: float = DEFAULT_ASSIGNMENT_TOLERANCE,
    giant_mass_fraction: float = DEFAULT_GIANT_MASS_FRACTION,
    giant_source_fraction: float = DEFAULT_GIANT_SOURCE_FRACTION,
) -> dict[str, Any]:
    """Compute mass-weighted group geometry and empty/giant diagnostics.

    Inputs are one-case arrays ``A[E,K]``, ``nu[E]`` and ``x[E,D]``.  Source
    weights are normalized internally if they are finite and positive but do
    not already sum to one; the original sum is retained in the result.
    """

    assignment = _case_array(environment_membership, name="environment_membership", ndim=2)
    measure = _case_array(environment_measure, name="environment_measure", ndim=1).astype(np.float64, copy=False)
    coordinates = _case_array(environment_coordinates, name="environment_coordinates", ndim=2).astype(np.float64, copy=False)
    if assignment.shape[0] != measure.shape[0] or coordinates.shape[0] != measure.shape[0]:
        raise Run1503DiagnosticError(
            "environment membership, measure, and coordinates must share source count"
        )
    if not np.isfinite(float(mass_epsilon)) or float(mass_epsilon) < 0.0:
        raise ValueError("mass_epsilon must be finite and non-negative")
    if not 0.0 < float(giant_mass_fraction) <= 1.0:
        raise ValueError("giant_mass_fraction must be in (0, 1]")
    if not 0.0 < float(giant_source_fraction) <= 1.0:
        raise ValueError("giant_source_fraction must be in (0, 1]")
    _finite_nonnegative(assignment.astype(np.float64, copy=False), name="environment_membership")
    _finite_nonnegative(measure, name="environment_measure")
    if not np.isfinite(coordinates).all():
        raise Run1503DiagnosticError("environment_coordinates contains non-finite values")
    measure_sum = float(np.sum(measure))
    if measure_sum <= float(measure_epsilon):
        raise Run1503DiagnosticError("environment_measure has no positive total mass")
    normalized_measure = measure / measure_sum
    row_sums = np.sum(assignment, axis=-1)
    active_rows = normalized_measure > float(measure_epsilon)
    row_error = np.abs(row_sums[active_rows] - 1.0) if np.any(active_rows) else np.zeros(0)
    weighted = normalized_measure[:, None] * assignment
    mass = np.sum(weighted, axis=0)
    safe_mass = np.maximum(mass, float(mass_epsilon))
    centroids = np.sum(weighted[..., None] * coordinates[:, None, :], axis=0) / safe_mass[:, None]
    centroids = np.where((mass > float(mass_epsilon))[:, None], centroids, 0.0)
    displacement = coordinates[:, None, :] - centroids[None, :, :]
    radius_squared = np.sum(weighted * np.sum(displacement * displacement, axis=-1), axis=0) / safe_mass
    radius = np.sqrt(np.maximum(radius_squared, 0.0))
    source_active = active_rows[:, None] & (assignment > float(assignment_tolerance))
    source_counts = np.sum(source_active, axis=0).astype(np.int64)
    active_source_count = int(np.sum(active_rows))
    source_fraction = source_counts / max(active_source_count, 1)
    empty_groups = np.flatnonzero(mass <= float(mass_epsilon)).astype(int).tolist()
    giant_groups = np.flatnonzero(
        (mass >= float(giant_mass_fraction)) | (source_fraction >= float(giant_source_fraction))
    ).astype(int).tolist()
    return {
        "group_count": int(assignment.shape[1]),
        "source_count": int(assignment.shape[0]),
        "active_source_count": active_source_count,
        "measure_input_sum": measure_sum,
        "measure_normalized": bool(abs(measure_sum - 1.0) <= 1.0e-6),
        "mass": mass.astype(np.float64),
        "centroid": centroids.astype(np.float64),
        "radius": radius.astype(np.float64),
        "radius_squared": radius_squared.astype(np.float64),
        "source_count_per_group": source_counts,
        "source_fraction_per_group": source_fraction.astype(np.float64),
        "empty_groups": empty_groups,
        "giant_groups": giant_groups,
        "row_normalization_max_abs_error": float(np.max(row_error)) if row_error.size else 0.0,
        "row_normalization_mean_abs_error": float(np.mean(row_error)) if row_error.size else 0.0,
        "mass_sum": float(np.sum(mass)),
        "nonnegative": bool(np.all(mass >= -float(mass_epsilon))),
        "thresholds": {
            "mass_epsilon": float(mass_epsilon),
            "measure_epsilon": float(measure_epsilon),
            "assignment_tolerance": float(assignment_tolerance),
            "giant_mass_fraction": float(giant_mass_fraction),
            "giant_source_fraction": float(giant_source_fraction),
        },
    }


def _canonical_optional_assignment(
    value: Any,
    *,
    name: str,
    query_count: int,
    group_count: int,
    case_index: int,
) -> np.ndarray | None:
    if value is None:
        return None
    array = _case_array(value, name=name, ndim=2, case_index=case_index).astype(np.float64, copy=False)
    if tuple(array.shape) != (int(query_count), int(group_count)):
        raise Run1503DiagnosticError(
            f"{name} shape {array.shape} does not match [Q,K]=[{query_count},{group_count}]"
        )
    _finite_nonnegative(array, name=name)
    return array


def _ledger_record(payload: Any, *, case_index: int) -> dict[str, Any] | None:
    value = _lookup(payload, "phase_ledger")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return {"status": "unavailable", "reason": "phase ledger is not a mapping"}
    # Preserve the full explicit ledger but annotate whether it was batched.
    if case_index in value and isinstance(value[case_index], Mapping):
        value = value[case_index]
    return dict(value)


def _counter(record: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    for alias in aliases:
        if alias not in record or record[alias] is None:
            continue
        try:
            number = float(np.asarray(record[alias]).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(number):
            return number
    return None


def _ledger_scope(ledger: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize explicit P2/environment counters without inferring rows."""

    if ledger is None:
        return {"status": "unavailable", "reason": "no explicit phase ledger"}
    phase = ledger.get("P2", ledger.get("p2", ledger))
    if not isinstance(phase, Mapping):
        return {"status": "unavailable", "reason": "P2 ledger is not a mapping"}
    environment = phase.get("environment", phase.get("env", phase))
    if not isinstance(environment, Mapping):
        return {"status": "unavailable", "reason": "environment ledger is not a mapping"}
    aliases = {
        "support_pairs": ("support_pairs", "support_pair_count", "unique_pair_count", "logical_support_pairs"),
        "executed_rows": ("executed_rows", "actual_fine_rows", "fine_rows_forward", "actual_rows"),
        "padded_rows": ("padded_rows", "fine_rows_padded", "actual_padded_rows"),
        "recomputed_rows": ("recomputed_rows", "fine_rows_recompute", "checkpoint_recomputations"),
        "coarse_rows": ("coarse_rows", "coarse_group_rows", "coarse_execution_rows"),
        "fine_group_rows": ("fine_group_rows", "group_fine_rows", "fine_rows"),
    }
    normalized = {key: _counter(environment, names) for key, names in aliases.items()}
    required = ("support_pairs", "executed_rows", "padded_rows", "recomputed_rows")
    normalized["status"] = "ok" if all(normalized[key] is not None for key in required) else "partial"
    normalized["source"] = "explicit_normal_executor_phase_ledger"
    return normalized


def opening_diagnostics(
    query_assignment: Any,
    environment_membership: Any,
    environment_measure: Any,
    *,
    coarse_assignment: Any | None = None,
    opening_blend: Any | None = None,
    fine_open_mask: Any | None = None,
    coarse_available: Any | None = None,
    fine_group_rows: Any | None = None,
    fine_rows_per_query: Any | None = None,
    fine_rows: Any | None = None,
    coarse_rows: Any | None = None,
    full_rectangle_rows: Any | None = None,
    fine_work_ratio: Any | None = None,
    receiver_chunk_size: int = DEFAULT_RECEIVER_CHUNK_SIZE,
    assignment_tolerance: float = DEFAULT_ASSIGNMENT_TOLERANCE,
    measure_epsilon: float = DEFAULT_MEASURE_EPSILON,
    case_index: int = 0,
) -> dict[str, Any]:
    """Compute Q_k/E_k, fine block area, coarse access, and local openings."""

    query = _case_array(query_assignment, name="query_assignment", ndim=2).astype(np.float64, copy=False)
    environment = _case_array(environment_membership, name="environment_membership", ndim=2).astype(np.float64, copy=False)
    measure = _case_array(environment_measure, name="environment_measure", ndim=1).astype(np.float64, copy=False)
    if environment.shape[0] != measure.shape[0] or query.shape[1] != environment.shape[1]:
        raise Run1503DiagnosticError("query/environment assignments must share K and environment measure must share E")
    _finite_nonnegative(query, name="query_assignment")
    _finite_nonnegative(environment, name="environment_membership")
    _finite_nonnegative(measure, name="environment_measure")
    group_count = int(query.shape[1])
    query_count = int(query.shape[0])
    source_count = int(environment.shape[0])
    query_positive = query > float(assignment_tolerance)
    active_sources = measure > float(measure_epsilon)
    environment_positive = environment > float(assignment_tolerance)
    qk = np.sum(query_positive, axis=0).astype(np.int64)
    ek = np.sum(environment_positive & active_sources[:, None], axis=0).astype(np.int64)
    block_area = qk.astype(np.float64) * ek.astype(np.float64)
    full_area = float(query_count * source_count)
    active_area = float(query_count * int(np.sum(active_sources)))
    fine_block_area = float(np.sum(block_area))
    logical_paths = int(np.sum(query_positive[:, None, :] & environment_positive[None, :, :]))
    unique_pairs = int(np.sum(np.any(query_positive[:, None, :] & environment_positive[None, :, :], axis=-1)))
    coarse = _canonical_optional_assignment(
        coarse_assignment,
        name="coarse_assignment",
        query_count=query_count,
        group_count=group_count,
        case_index=case_index,
    )
    coarse_rows = None if coarse is None else int(query_count * group_count)
    if coarse is None:
        coarse_access = {
            "status": "unavailable",
            "reason": "future Run-1503 core should emit p_qk/coarse assignment",
        }
    else:
        coarse_positive = coarse > float(assignment_tolerance)
        coarse_access = {
            "status": "observed_assignment",
            "row_normalization_max_abs_error": float(
                np.max(np.abs(np.sum(coarse, axis=-1) - 1.0))
            )
            if query_count
            else 0.0,
            "available_group_count_per_query": np.sum(coarse_positive, axis=-1).astype(np.int64),
            "available_fraction": float(np.mean(np.any(coarse_positive & (ek > 0)[None, :], axis=-1)))
            if query_count
            else 0.0,
        }
    blend = _canonical_optional_assignment(
        opening_blend,
        name="opening_blend",
        query_count=query_count,
        group_count=group_count,
        case_index=case_index,
    )
    if blend is not None and (np.any(blend < 0.0) or np.any(blend > 1.0 + 1.0e-6)):
        raise Run1503DiagnosticError("opening_blend must lie in [0,1]")
    explicit_mask = None
    if fine_open_mask is not None:
        explicit_mask = _case_array(fine_open_mask, name="fine_open_mask", ndim=2).astype(bool, copy=False)
        if tuple(explicit_mask.shape) != (query_count, group_count):
            raise Run1503DiagnosticError("fine_open_mask shape does not match [Q,K]")
    if explicit_mask is not None:
        open_mask = explicit_mask
        opening_status = "explicit_mask"
    elif blend is not None:
        open_mask = blend > float(assignment_tolerance)
        opening_status = "explicit_blend_support"
    else:
        # Current Run-1502 has no separate opening blend.  Its sparse query
        # support is a useful structural proxy, but must not be relabelled as
        # Run-1503 coarse/fine execution evidence.
        open_mask = query_positive
        opening_status = "inferred_from_sparse_query_support"
    opening_counts = np.sum(open_mask, axis=-1).astype(np.int64)
    open_group_counts = np.sum(open_mask, axis=0).astype(np.int64)
    opened_block_area = float(np.sum(open_group_counts.astype(np.float64) * ek.astype(np.float64)))
    if coarse_available is None:
        coarse_available_summary: dict[str, Any] = {
            "status": coarse_access["status"] if coarse is not None else "unavailable",
            "explicit": False,
            "reason": "no coarse availability execution flag was emitted",
        }
    else:
        raw_available = _array(coarse_available, name="coarse_available")
        if raw_available.ndim >= 1 and raw_available.shape[0] == 1:
            raw_available = raw_available[0]
        coarse_available_summary = {
            "status": "explicit",
            "explicit": True,
            "fraction": float(np.mean(raw_available.astype(bool))),
            "shape": list(raw_available.shape),
        }
    observed_execution: dict[str, Any] = {"status": "unavailable"}
    if any(value is not None for value in (fine_group_rows, fine_rows_per_query, fine_rows, coarse_rows, full_rectangle_rows, fine_work_ratio)):
        observed_execution = {"status": "observed_flat_executor_aux"}
        for name, value in (
            ("fine_group_rows", fine_group_rows),
            ("fine_rows_per_query", fine_rows_per_query),
            ("fine_rows", fine_rows),
            ("coarse_rows", coarse_rows),
            ("full_rectangle_rows", full_rectangle_rows),
            ("fine_work_ratio", fine_work_ratio),
        ):
            if value is not None:
                observed_execution[name] = _jsonable(_array(value, name=name))
    chunk = int(receiver_chunk_size)
    if chunk <= 0:
        raise ValueError("receiver_chunk_size must be positive")
    return {
        "query_count": query_count,
        "environment_count": source_count,
        "active_environment_count": int(np.sum(active_sources)),
        "group_count": group_count,
        "Q_k": qk,
        "E_k": ek,
        "fine_block_area_per_group": block_area,
        "fine_block_area": fine_block_area,
        "full_QE_area": full_area,
        "active_QE_area": active_area,
        "fine_block_ratio": float(fine_block_area / full_area) if full_area else 0.0,
        "fine_block_ratio_active_denominator": float(fine_block_area / active_area) if active_area else 0.0,
        "logical_paths": logical_paths,
        "unique_query_source_pairs": unique_pairs,
        "empty_query_count": int(np.sum(opening_counts == 0)),
        "active_opened_group_count_per_query": opening_counts,
        "opening_group_query_count": open_group_counts,
        "opened_fine_block_area": opened_block_area,
        "opening_status": opening_status,
        "opening_fraction_of_queries": float(np.mean(opening_counts > 0)) if query_count else 0.0,
        "opening_count_summary": _summary(opening_counts),
        "coarse_rows": coarse_rows,
        "coarse_global_availability": coarse_available_summary,
        "coarse_assignment_present": coarse is not None,
        "coarse_assignment_row_normalization_max_abs_error": (
            coarse_access.get("row_normalization_max_abs_error") if coarse is not None else None
        ),
        "observed_execution": observed_execution,
        "chunk_support": {
            "receiver_chunk_size": chunk,
            "query_chunk_count": math.ceil(query_count / chunk) if query_count else 0,
            "timing_matched_to_receiver_chunk": True,
        },
        "thresholds": {
            "assignment_tolerance": float(assignment_tolerance),
            "measure_epsilon": float(measure_epsilon),
        },
    }


def opening_locality_diagnostics(
    query_coordinates: Any,
    query_assignment: Any,
    group_centroids: Any,
    group_radii: Any,
    *,
    assignment_tolerance: float = DEFAULT_ASSIGNMENT_TOLERANCE,
    radius_epsilon: float = DEFAULT_MASS_EPSILON,
) -> dict[str, Any]:
    """Measure whether opened groups are geometrically local to each query."""

    query_xy = _case_array(query_coordinates, name="query_coordinates", ndim=2).astype(
        np.float64, copy=False
    )
    assignment = _case_array(query_assignment, name="query_assignment", ndim=2).astype(
        np.float64, copy=False
    )
    centroids = _case_array(group_centroids, name="group_centroids", ndim=2).astype(
        np.float64, copy=False
    )
    radii = _case_array(group_radii, name="group_radii", ndim=1).astype(
        np.float64, copy=False
    )
    if query_xy.shape[0] != assignment.shape[0]:
        raise Run1503DiagnosticError("query coordinates and assignment must share Q")
    if centroids.shape[0] != assignment.shape[1] or radii.shape[0] != assignment.shape[1]:
        raise Run1503DiagnosticError("group centroids/radii and assignment must share K")
    if query_xy.shape[1] != centroids.shape[1]:
        raise Run1503DiagnosticError("query coordinates and group centroids must share dimension")
    if not np.isfinite(query_xy).all() or not np.isfinite(centroids).all():
        raise Run1503DiagnosticError("locality coordinates contain non-finite values")
    _finite_nonnegative(assignment, name="query_assignment")
    _finite_nonnegative(radii, name="group_radii")
    distances = np.linalg.norm(query_xy[:, None, :] - centroids[None, :, :], axis=-1)
    normalized_distances = distances / np.maximum(radii[None, :], float(radius_epsilon))
    opened = assignment > float(assignment_tolerance)
    closed = ~opened
    nearest = np.argmin(distances, axis=-1)
    query_indices = np.arange(query_xy.shape[0])
    opened_distances = distances[opened]
    closed_distances = distances[closed]
    opened_normalized = normalized_distances[opened]
    closed_normalized = normalized_distances[closed]
    assignment_mass = float(np.sum(assignment))
    return {
        "opened_distance_summary": _summary(opened_distances),
        "closed_distance_summary": _summary(closed_distances),
        "opened_radius_normalized_distance_summary": _summary(opened_normalized),
        "closed_radius_normalized_distance_summary": _summary(closed_normalized),
        "opened_to_closed_mean_distance_ratio": (
            None
            if not opened_distances.size
            or not closed_distances.size
            or float(np.mean(closed_distances)) <= 0.0
            else float(np.mean(opened_distances) / np.mean(closed_distances))
        ),
        "assignment_weighted_mean_distance": (
            None
            if assignment_mass <= 0.0
            else float(np.sum(assignment * distances) / assignment_mass)
        ),
        "nearest_centroid_open_fraction": (
            float(np.mean(opened[query_indices, nearest])) if query_xy.shape[0] else 0.0
        ),
        "all_queries_have_open_group": bool(np.all(np.any(opened, axis=-1))),
        "interpretation": "Learned organization evidence only: locality compares query positions with learned environmental-group centroids; it is not a claim of physical causality.",
    }


def collect_case_diagnostic(
    payload: Any,
    *,
    case_id: str | None = None,
    case_index: int = 0,
    module_count: int | None = None,
    receiver_chunk_size: int = DEFAULT_RECEIVER_CHUNK_SIZE,
    mass_epsilon: float = DEFAULT_MASS_EPSILON,
    measure_epsilon: float = DEFAULT_MEASURE_EPSILON,
    assignment_tolerance: float = DEFAULT_ASSIGNMENT_TOLERANCE,
    giant_mass_fraction: float = DEFAULT_GIANT_MASS_FRACTION,
    giant_source_fraction: float = DEFAULT_GIANT_SOURCE_FRACTION,
) -> dict[str, Any]:
    """Extract and summarize one frozen case or future Run-1503 payload."""

    environment_membership = _lookup(payload, "environment_membership", required=True)
    environment_measure = _lookup(payload, "environment_measure", required=True)
    environment_coordinates = _lookup(payload, "environment_coordinates", required=True)
    query_assignment = _lookup(payload, "query_assignment", required=True)
    geometry = group_geometry(
        environment_membership,
        environment_measure,
        environment_coordinates,
        mass_epsilon=mass_epsilon,
        measure_epsilon=measure_epsilon,
        assignment_tolerance=assignment_tolerance,
        giant_mass_fraction=giant_mass_fraction,
        giant_source_fraction=giant_source_fraction,
    )
    query = _case_array(query_assignment, name="query_assignment", ndim=2, case_index=case_index)
    environment = _case_array(environment_membership, name="environment_membership", ndim=2, case_index=case_index)
    measure = _case_array(environment_measure, name="environment_measure", ndim=1, case_index=case_index)
    coarse_value = _lookup(payload, "coarse_assignment")
    blend_value = _lookup(payload, "opening_blend")
    mask_value = _lookup(payload, "fine_open_mask")
    coarse_available_value = _lookup(payload, "coarse_available")
    fine_group_rows_value = _lookup(payload, "fine_group_rows")
    fine_rows_per_query_value = _lookup(payload, "fine_rows_per_query")
    fine_rows_value = _lookup(payload, "fine_rows")
    coarse_rows_value = _lookup(payload, "coarse_rows")
    full_rectangle_rows_value = _lookup(payload, "full_rectangle_rows")
    fine_work_ratio_value = _lookup(payload, "fine_work_ratio")
    openings = opening_diagnostics(
        query,
        environment,
        measure,
        coarse_assignment=coarse_value,
        opening_blend=blend_value,
        fine_open_mask=mask_value,
        coarse_available=coarse_available_value,
        fine_group_rows=fine_group_rows_value,
        fine_rows_per_query=fine_rows_per_query_value,
        fine_rows=fine_rows_value,
        coarse_rows=coarse_rows_value,
        full_rectangle_rows=full_rectangle_rows_value,
        fine_work_ratio=fine_work_ratio_value,
        receiver_chunk_size=receiver_chunk_size,
        assignment_tolerance=assignment_tolerance,
        measure_epsilon=measure_epsilon,
        case_index=case_index,
    )
    ledger_value = _ledger_record(payload, case_index=case_index)
    if ledger_value is not None:
        ledger = _ledger_scope(ledger_value)
    else:
        flat = {
            "support_pairs": _lookup(payload, "support_pairs"),
            "executed_rows": _lookup(payload, "fine_rows"),
            "padded_rows": _lookup(payload, "padded_rows"),
            "recomputed_rows": _lookup(payload, "recomputed_rows"),
            "coarse_rows": coarse_rows_value,
            "fine_group_rows": fine_group_rows_value,
        }
        if any(value is not None for value in flat.values()):
            ledger = {
                key: (
                    None
                    if value is None
                    else _jsonable(_array(value, name=key))
                )
                for key, value in flat.items()
            }
            required = ("support_pairs", "executed_rows", "padded_rows", "recomputed_rows")
            ledger["status"] = "ok" if all(ledger[key] is not None for key in required) else "partial"
            ledger["source"] = "flat_normal_executor_aux"
        else:
            ledger = _ledger_scope(None)
    branch_values = {
        "coarse_contribution": _lookup(payload, "coarse_contribution"),
        "fine_contribution": _lookup(payload, "fine_contribution"),
    }
    branch_finite = {
        key: (
            None
            if value is None
            else bool(np.isfinite(_array(value, name=key)).all())
        )
        for key, value in branch_values.items()
    }
    result: dict[str, Any] = {
        "case_id": None if case_id is None else str(case_id),
        "selection_reason": CASE_SELECTION_REASONS.get(str(case_id), "explicit case") if case_id is not None else None,
        "module_count": None if module_count is None else int(module_count),
        "geometry": geometry,
        "opening": openings,
        "executor_ledger": ledger,
        "branch_finite": branch_finite,
        "aux_contract": {
            "coarse_assignment": "observed" if coarse_value is not None else "unavailable",
            "opening_blend": "observed" if blend_value is not None else "unavailable_or_inferred",
            "coarse_available": "observed" if coarse_available_value is not None else "unavailable",
            "actual_rows": "explicit ledger only",
        },
        "interpretation": (
            "Group incidence and sparse query support are learned organizational evidence, not physical causality. "
            "Fine block area is a planned group-major rectangle count; it is not measured runtime work unless an explicit executor ledger is present."
        ),
    }
    return result


def expected_aux_contract() -> dict[str, Any]:
    """Return the exact future-core ABI needed for complete Run-1503 evidence."""

    return {
        "required": {
            "adaptive_hyperedge_environment_membership": "[B,E,K] nonnegative; each active source row sums to one",
            "adaptive_hyperedge_environment_measure": "[B,E] positive normalized source measure nu_j",
            "adaptive_hyperedge_environment_coordinates": "[B,E,2] physical source coordinates",
            "adaptive_hyperedge_query_sparse_assignment": "[B,Q,K] alpha_qk from the shared query/group logits",
        },
        "recommended": {
            "adaptive_hyperedge_query_logits": "[B,Q,K] one shared ell_qk bank used by p and alpha",
            "adaptive_hyperedge_query_coarse_assignment": "[B,Q,K] p_qk masked softmax(logits + log(mu))",
            "adaptive_hyperedge_opening_blend": "[B,Q,K] clamp(alpha/(p+eps), 0, 1)",
            "adaptive_hyperedge_fine_open_mask": "[B,Q,K] bool; alpha-positive groups only",
            "adaptive_hyperedge_coarse_available": "[B,Q,K] or [B,Q] bool; availability, not merely p>0",
            "adaptive_hyperedge_phase_ledger": "mapping with P2/environment support_pairs, executed_rows, padded_rows, recomputed_rows",
            "group_control_adaptive_fine_group_rows": "[B,K] actual group-major fine rectangle rows",
            "group_control_adaptive_fine_rows_per_query": "[B,Q] summed fine rows attributable to each receiver",
            "group_control_adaptive_fine_rows": "[B] explicit summed fine rows",
            "group_control_adaptive_coarse_rows": "[B] explicit coarse QxK rows",
            "group_control_adaptive_full_rectangle_rows": "[B] explicit dense QE denominator",
            "group_control_adaptive_fine_work_ratio": "[B] explicit fine rows/full rectangle rows",
        },
        "semantics": {
            "group_mass": "sum_j nu_j A^E_jk; diagnostic may derive it",
            "group_centroid": "mass-weighted physical centroid; diagnostic may derive it",
            "group_radius": "mass-weighted RMS radius; diagnostic may derive it",
            "Q_k": "count_q(alpha_qk > tolerance)",
            "E_k": "count_j(A^E_jk > tolerance and nu_j > measure_epsilon)",
            "fine_block_ratio": "sum_k Q_k E_k / (Q E), with active-QE ratio reported separately",
            "actual_rows": "never infer from support or block area; use explicit phase ledger",
        },
    }


def _rss_bytes() -> int | None:
    try:
        # Linux reports ru_maxrss in KiB; this workspace is Ubuntu.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (AttributeError, OSError):  # pragma: no cover - non-POSIX fallback
        return None


def _sync_device(device: Any) -> None:
    if getattr(device, "type", str(device)) != "cuda":
        return
    import torch

    torch.cuda.synchronize(device)


def _invoke_scope(function: Callable[..., Any], receiver_chunk_size: int) -> Any:
    """Invoke the benchmark callback with one matched receiver chunk."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(receiver_chunk_size)
    parameters = list(signature.parameters.values())
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    accepts_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
    return function(receiver_chunk_size) if positional or accepts_varargs else function()


def _measure_timing_scope(
    name: str,
    function: Callable[..., Any],
    *,
    device: Any,
    receiver_chunk_size: int,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    """Measure one application scope with synchronized wall/event/memory fields."""

    if int(warmups) < 0 or int(repetitions) <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    if int(receiver_chunk_size) <= 0:
        raise ValueError("receiver_chunk_size must be positive")
    is_cuda = getattr(device, "type", str(device)) == "cuda"
    torch = None
    if is_cuda:
        import torch as _torch

        torch = _torch
    with contextlib.suppress(Exception):
        for _ in range(int(warmups)):
            output = _invoke_scope(function, int(receiver_chunk_size))
            del output
        _sync_device(device)
    samples: list[dict[str, Any]] = []
    tracing_started = tracemalloc.is_tracing()
    if not tracing_started:
        tracemalloc.start()
    for repetition in range(int(repetitions)):
        if is_cuda and torch is not None:
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
        else:
            baseline_allocated = None
            baseline_reserved = None
            event_start = event_end = None
        tracemalloc.reset_peak()
        current_before, _ = tracemalloc.get_traced_memory()
        rss_before = _rss_bytes()
        _sync_device(device)
        started = time.perf_counter()
        output: Any = None
        error: str | None = None
        if event_start is not None:
            event_start.record()
        try:
            output = _invoke_scope(function, int(receiver_chunk_size))
        except Exception as exc:  # noqa: BLE001 - preserve per-scope evidence
            error = f"{type(exc).__name__}: {exc}"
        if event_end is not None:
            event_end.record()
        _sync_device(device)
        elapsed = time.perf_counter() - started
        current_after, peak_python = tracemalloc.get_traced_memory()
        rss_after = _rss_bytes()
        cuda_ms = None
        if event_start is not None and event_end is not None and error is None:
            cuda_ms = float(event_start.elapsed_time(event_end)) / 1000.0
        if is_cuda and torch is not None:
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = None
            peak_reserved = None
        samples.append(
            {
                "repetition": int(repetition + 1),
                "status": "error" if error is not None else "complete",
                "error": error,
                "elapsed_seconds": float(elapsed),
                "cuda_event_elapsed_seconds": cuda_ms,
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "incremental_peak_allocated_bytes": (
                    None if peak_allocated is None or baseline_allocated is None else peak_allocated - baseline_allocated
                ),
                "incremental_peak_reserved_bytes": (
                    None if peak_reserved is None or baseline_reserved is None else peak_reserved - baseline_reserved
                ),
                "python_tracemalloc_baseline_bytes": int(current_before),
                "python_tracemalloc_current_bytes": int(current_after),
                "python_tracemalloc_peak_bytes": int(peak_python),
                "process_rss_before_bytes": rss_before,
                "process_rss_after_bytes": rss_after,
            }
        )
        del output
        if error is not None:
            break
    if not tracing_started:
        tracemalloc.stop()
    complete = [row for row in samples if row["status"] == "complete"]
    elapsed_values = [float(row["elapsed_seconds"]) for row in complete]
    event_values = [float(row["cuda_event_elapsed_seconds"]) for row in complete if row["cuda_event_elapsed_seconds"] is not None]
    return {
        "scope": str(name),
        "status": "complete" if len(complete) == int(repetitions) else "incomplete",
        "device": str(device),
        "receiver_chunk_size": int(receiver_chunk_size),
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "samples": samples,
        "median_elapsed_seconds": None if not elapsed_values else float(np.median(elapsed_values)),
        "min_elapsed_seconds": None if not elapsed_values else float(np.min(elapsed_values)),
        "max_elapsed_seconds": None if not elapsed_values else float(np.max(elapsed_values)),
        "median_cuda_event_elapsed_seconds": None if not event_values else float(np.median(event_values)),
        "timed_maps": False,
        "timed_profiler": False,
        "memory_scope": "CUDA allocated/reserved when CUDA; Python tracemalloc and process RSS fields on CPU",
    }


def benchmark_application_scopes(
    scopes: Mapping[str, Callable[..., Any] | None],
    *,
    receiver_chunk_size: int = DEFAULT_RECEIVER_CHUNK_SIZE,
    device: str = "cpu",
    warmups: int = DEFAULT_WARMUPS,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Benchmark named application scopes under one matched chunk protocol.

    Callbacks receive ``receiver_chunk_size`` when they declare a positional
    argument.  A missing callback is reported as unavailable, never replaced
    by another phase.
    """

    if str(device) != "cpu":
        raise ValueError("Run1503 diagnostics are CPU-only in this task; pass --device cpu")
    try:
        import torch

        resolved_device = torch.device("cpu")
    except ImportError:  # pragma: no cover - project runtime requires torch
        resolved_device = "cpu"
    result: dict[str, Any] = {
        "status": "complete",
        "device": "cpu",
        "receiver_chunk_size": int(receiver_chunk_size),
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "timed_maps": False,
        "timed_profiler": False,
        "timing_role": "structural CPU timing only; not the formal GPU execution benchmark",
        "formal_gpu_benchmark": False,
        "scope_order": list(scopes),
        "scopes": {},
        "chunk_policy": "all available scopes receive the same evaluation-only receiver chunk",
    }
    for name, function in scopes.items():
        if function is None:
            result["scopes"][str(name)] = {
                "scope": str(name),
                "status": "unavailable",
                "reason": "scope callback was not supplied; no phase substituted",
                "receiver_chunk_size": int(receiver_chunk_size),
            }
            result["status"] = "partial"
            continue
        result["scopes"][str(name)] = _measure_timing_scope(
            str(name),
            function,
            device=resolved_device,
            receiver_chunk_size=int(receiver_chunk_size),
            warmups=int(warmups),
            repetitions=int(repetitions),
        )
        if result["scopes"][str(name)]["status"] != "complete":
            result["status"] = "partial"
    return result


def _parse_checkpoint(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Run1502 epoch-500 checkpoint does not exist: {path}")
    return path


def _case_ids(args: argparse.Namespace) -> tuple[str, ...]:
    values = tuple(str(value) for value in (args.case_id or DEFAULT_CASE_IDS))
    if not values or len(set(values)) != len(values):
        raise ValueError("case IDs must be non-empty and unique")
    return values


def _protocol(args: argparse.Namespace, checkpoint: Path) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_role": "frozen Run-1502 epoch-500 structural input; no weights updated",
        "case_ids": list(_case_ids(args)),
        "representative_case_ids": list(REPRESENTATIVE_CASE_IDS),
        "module_panel_case_ids": list(MODULE_PANEL_CASE_IDS),
        "query_count": int(args.query_count),
        "query_batch_size": int(args.query_batch_size),
        "receiver_chunk_size": int(args.receiver_chunk_size),
        "inference_warmups": int(args.warmups),
        "inference_repetitions": int(args.repetitions),
        "timed_scopes": ["full_physical_forward", "prepared_p2_decode", "application_evaluator"],
        "timed_maps": False,
        "timed_profiler": False,
        "device": "cpu",
        "timing_role": "structural CPU timing only; not the formal GPU execution benchmark",
        "training_launched": False,
        "checkpoint_written": False,
        "actual_row_policy": "explicit executor phase ledger only; support/block area is not relabelled as executed rows",
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build a CPU-safe plan without importing checkpoint/model runtime code."""

    checkpoint = _parse_checkpoint(args.checkpoint)
    if int(args.query_count) <= 0 or int(args.query_batch_size) <= 0 or int(args.receiver_chunk_size) <= 0:
        raise ValueError("query and receiver chunk sizes must be positive")
    if int(args.warmups) < 0 or int(args.repetitions) <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    if str(args.device) != "cpu":
        raise ValueError("Run1503 diagnostics are CPU-only in this task; pass --device cpu")
    output = Path(args.output).expanduser().resolve()
    return {
        "schema_version": 1,
        "task": "run1503_adaptive_hyperedge_opening_frozen_diagnostic",
        "status": "plan_only",
        "checkpoint_policy": {
            "path": str(checkpoint),
            "selection": "explicit_cli_checkpoint",
            "required_role": "Run-1502 exact epoch-500 model; diagnostic does not change it",
            "accepted_frozen_architecture": "sparse_incidence_group_control_honf",
            "future_target_architecture": "adaptive_hyperedge_opening_honf",
            "missing_core_policy": "report coarse/opening fields unavailable until target architecture emits the contract",
        },
        "protocol": _protocol(args, checkpoint),
        "expected_aux_contract": expected_aux_contract(),
        "planned_outputs": {
            "json": str(output),
            "csv": str(output.with_suffix(".csv")),
        },
        "limitations": [
            "Frozen organization is structural evidence, not a Run-1503 retrained-accuracy forecast.",
            "Learned groups and routes are not physical causality.",
            "Fine block area is not measured runtime work without an explicit executor ledger.",
            "No GPU, optimizer, training launch, or checkpoint write is performed by the CPU diagnostic.",
            "CPU timing scopes are structural checks only and must not be reported as the formal GPU benchmark.",
        ],
    }


def _write_case_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "module_count",
        "group_count",
        "environment_count",
        "mass_sum",
        "empty_group_count",
        "giant_group_count",
        "query_count",
        "fine_block_area",
        "fine_block_ratio",
        "opened_fine_block_area",
        "opening_status",
        "coarse_assignment_present",
        "coarse_global_status",
        "executor_ledger_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            geometry = row.get("geometry", {})
            opening = row.get("opening", {})
            coarse = opening.get("coarse_global_availability", {})
            ledger = row.get("executor_ledger", {})
            writer.writerow(
                {
                    "case_id": row.get("case_id"),
                    "module_count": row.get("module_count"),
                    "group_count": geometry.get("group_count"),
                    "environment_count": geometry.get("source_count"),
                    "mass_sum": geometry.get("mass_sum"),
                    "empty_group_count": len(geometry.get("empty_groups", [])),
                    "giant_group_count": len(geometry.get("giant_groups", [])),
                    "query_count": opening.get("query_count"),
                    "fine_block_area": opening.get("fine_block_area"),
                    "fine_block_ratio": opening.get("fine_block_ratio"),
                    "opened_fine_block_area": opening.get("opened_fine_block_area"),
                    "opening_status": opening.get("opening_status"),
                    "coarse_assignment_present": opening.get("coarse_assignment_present"),
                    "coarse_global_status": coarse.get("status"),
                    "executor_ledger_status": ledger.get("status"),
                }
            )


def _runtime_payload_for_prediction(prediction: Mapping[str, Any]) -> Mapping[str, Any]:
    """Expose current Run-1502 prepared controls beside routing maps."""

    payload: dict[str, Any] = dict(prediction)
    prepared_case = prediction.get("_prepared_state")
    prepared = getattr(prepared_case, "prepared", None)
    if prepared is not None:
        encoded = getattr(prepared, "encoded", None)
        backend_state = getattr(prepared, "backend_state", None)
        controls = backend_state.get("group_control_state") if isinstance(backend_state, Mapping) else None
        if controls is not None:
            payload.setdefault("group_control_state", controls)
            payload.setdefault("environment_membership", getattr(controls, "environment_membership", None))
            payload.setdefault("environment_measure", getattr(controls, "environment_measure", None))
        if encoded is not None:
            payload.setdefault("encoded", encoded)
            payload.setdefault("environment_coordinates", getattr(encoded, "env_coords", None))
    return payload


def _require_exact_epoch500(checkpoint: Mapping[str, Any]) -> int:
    """Reject a checkpoint whose payload is not unambiguously epoch 500."""

    selection_state = checkpoint.get("selection_state")
    epoch_values = []
    for value in (
        checkpoint.get("epoch"),
        checkpoint.get("current_epoch"),
        selection_state.get("epoch") if isinstance(selection_state, Mapping) else None,
    ):
        if value is None:
            continue
        try:
            epoch_values.append(int(value))
        except (TypeError, ValueError) as exc:
            raise Run1503DiagnosticError(
                f"Run1502 checkpoint epoch is not an integer: {value!r}"
            ) from exc
    if not epoch_values or any(value != 500 for value in epoch_values):
        raise Run1503DiagnosticError(
            "Run1503 frozen diagnostic requires a checkpoint payload at exact epoch 500; "
            f"observed epoch fields {epoch_values!r}"
        )
    return 500


def _load_frozen_target_model(checkpoint_path: Path, device: Any) -> tuple[Any, Mapping[str, Any], dict[str, Any]]:
    """Strict-load Run1502 weights into the parameter-compatible Run1503 core.

    The source checkpoint is never modified.  Only the in-memory
    ``forward_architecture`` selector changes; the Run1502 sparsemax setting,
    optimizer metadata, normalization, and state dictionary are retained.
    """

    from channelthermal.config import ChannelThermalHONFConfig
    from channelthermal.local_surrogate.model import LocalModuleConfig, LocalModuleSurrogate
    from channelthermal.model import ChannelThermalHONFModel

    from honf_runtime.checkpoints import validate_checkpoint_identity
    from honf_runtime.compat import load_trusted_checkpoint, strip_module_prefix

    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    validate_checkpoint_identity(
        checkpoint,
        case_id="ThermalChannel",
        model_family="honf_forward",
        workflow="forward",
    )
    checkpoint_epoch = _require_exact_epoch500(checkpoint)
    original_payload = checkpoint.get("model_config", {})
    payload = copy.deepcopy(original_payload)
    core_payload = payload.get("core_honf", payload.get("core", {}))
    if not isinstance(core_payload, Mapping):
        raise Run1503DiagnosticError("checkpoint model_config has no core_honf mapping")
    source_architecture = str(core_payload.get("forward_architecture", ""))
    if source_architecture != "sparse_incidence_group_control_honf":
        raise Run1503DiagnosticError(
            "frozen Run1503 construction requires a Run-1502 sparse-incidence checkpoint; "
            f"observed {source_architecture!r}"
        )
    interface_payload = core_payload.get("interface_model", {})
    if not isinstance(interface_payload, Mapping):
        raise Run1503DiagnosticError("checkpoint core_honf.interface_model is not a mapping")
    normalizer = str(interface_payload.get("environment_refinement_normalizer", ""))
    if normalizer != "sparsemax":
        raise Run1503DiagnosticError(
            "Run1503 frozen construction changes architecture only; checkpoint final environment "
            f"normalizer must already be sparsemax, got {normalizer!r}"
        )
    core_payload["forward_architecture"] = "adaptive_hyperedge_opening_honf"
    if "core_honf" in payload:
        payload["core_honf"] = core_payload
    else:
        payload["core"] = core_payload
    config = ChannelThermalHONFConfig.from_dict(payload)
    model = ChannelThermalHONFModel(config, attach_local_from_checkpoint=False)
    if bool(config.channelthermal.use_local_surrogate):
        local_payload = checkpoint.get("local_model_config")
        if isinstance(local_payload, Mapping):
            local_model = LocalModuleSurrogate(LocalModuleConfig.from_dict(dict(local_payload)))
            model.local_coupling.set_local_surrogate(
                local_model,
                freeze=bool(checkpoint.get("local_surrogate_frozen", config.channelthermal.freeze_local_surrogate)),
                normalization_config=checkpoint.get("local_normalization_config", {}),
                normalization_stats=checkpoint.get("local_normalization_stats", {}),
            )
            model.local_coupling.local_surrogate_checkpoint_path = checkpoint.get(
                "local_checkpoint_provenance", checkpoint.get("local_surrogate_checkpoint_path")
            )
        elif config.channelthermal.local_surrogate_checkpoint_path:
            model.local_coupling.attach_from_checkpoint(
                config.channelthermal.local_surrogate_checkpoint_path,
                freeze=bool(config.channelthermal.freeze_local_surrogate),
                map_location="cpu",
            )
    model = model.to(device)
    global_norm_cfg = checkpoint.get(
        "global_normalization_config",
        checkpoint.get("train_config", {}).get("dataset", {}),
    )
    model.set_global_target_normalization(
        checkpoint.get("global_normalization_stats", {}),
        normalize_targets=bool(global_norm_cfg.get("normalize_targets", False)),
    )
    state = strip_module_prefix(checkpoint["model_state_dict"])
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise Run1503DiagnosticError(
            "Run1503 frozen target is not parameter-compatible with Run1502 checkpoint; "
            f"strict load failed: {exc}"
        ) from exc
    selection_state = checkpoint.get("selection_state")
    configured_epochs = checkpoint.get("train_config", {}).get("training", {}).get("epochs")
    if isinstance(selection_state, Mapping) and selection_state.get("epoch") is not None:
        total_epochs = selection_state.get("total_epochs", configured_epochs)
        model.set_training_progress(
            epoch=int(selection_state["epoch"]),
            total_epochs=None if total_epochs is None else int(total_epochs),
        )
    else:
        model.set_training_progress(
            epoch=checkpoint_epoch,
            total_epochs=None if configured_epochs is None else int(configured_epochs),
        )
    model.eval()
    return model, checkpoint, {
        "source_architecture": source_architecture,
        "target_architecture": "adaptive_hyperedge_opening_honf",
        "architecture_changed_only": True,
        "environment_refinement_normalizer": normalizer,
        "strict_state_dict_load": True,
        "checkpoint_epoch": checkpoint_epoch,
        "state_dict_key_count": len(state),
    }


def _runtime_timing_scopes(
    *,
    model: Any,
    selected: Mapping[str, Any],
    query_xy: np.ndarray,
    device: Any,
    query_batch_size: int,
    receiver_chunk_size: int,
    warmups: int = DEFAULT_WARMUPS,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Construct matched-chunk CPU application scopes for one selected case."""

    import torch
    from channelthermal.evaluation.loading import make_batch
    from channelthermal.evaluation.prepared import predict_case
    from run_stage3_interface_study import _runtime_receiver_chunk_size

    forward_kwargs: dict[str, Any] | None = None
    prepared_holder: dict[str, Any] = {}

    def _prepare_one(_chunk: int) -> Any:
        nonlocal forward_kwargs
        with _runtime_receiver_chunk_size(model, int(receiver_chunk_size)):
            batch = make_batch(dict(selected), query_xy[:1], device)
            forward_kwargs = {
                "interface_condition": batch.get("interface_condition"),
                "local_module_params": batch.get("local_module_params"),
                "teacher_port_tokens": batch.get("teacher_port_tokens"),
                "local_query_points": batch.get("module_internal_query_points"),
                "local_port_condition_mode": "predicted",
                "mixed_teacher_ratio": 0.0,
            }
            output = model(
                batch["structure"],
                batch["query_xy"],
                return_routing_maps=False,
                return_prepared_state=True,
                **forward_kwargs,
            )
            prepared_holder["prepared"] = output["prepared_state"]
            return output

    def _full(_chunk: int) -> Any:
        with _runtime_receiver_chunk_size(model, int(receiver_chunk_size)):
            return predict_case(
                model,
                dict(selected),
                device,
                query_batch_size=int(query_batch_size),
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.0,
                return_routing_maps=False,
                return_prepared_state=False,
            )

    def _decode(_chunk: int) -> Any:
        if "prepared" not in prepared_holder:
            _prepare_one(int(receiver_chunk_size))
        query = torch.from_numpy(np.asarray(query_xy, dtype=np.float32)).unsqueeze(0).to(device)
        with _runtime_receiver_chunk_size(model, int(receiver_chunk_size)):
            return model.decode_prepared(
                prepared_holder["prepared"],
                query,
                return_routing_maps=False,
                receiver_chunk_size=int(receiver_chunk_size),
            )

    # The application evaluator is deliberately not substituted with a field
    # forward.  Its scope remains visible and unavailable until a caller wires
    # the case-level physical evaluator.
    return benchmark_application_scopes(
        {
            "full_physical_forward": _full,
            "prepared_p2_decode": _decode,
            "application_evaluator": None,
        },
        receiver_chunk_size=int(receiver_chunk_size),
        device="cpu",
        warmups=int(warmups),
        repetitions=int(repetitions),
    )


def run_frozen(args: argparse.Namespace) -> dict[str, Any]:
    """Run the bounded CPU-only frozen traversal when project runtime is available."""

    if str(args.device) != "cpu":
        raise ValueError("run_frozen is CPU-only; pass --device cpu")
    checkpoint_path = _parse_checkpoint(args.checkpoint)
    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import torch
        from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
        from channelthermal.evaluation.prepared import predict_case
        from run_run1409_occupancy_population import _dataset_for_checkpoint, _query_sample
    except ImportError as exc:  # pragma: no cover - exercised only without case runtime
        result = build_plan(args)
        result["status"] = "blocked_missing_runtime"
        result["reason"] = f"case runtime import failed: {type(exc).__name__}: {exc}"
        _write_json(Path(args.output), result)
        return result
    device = torch.device("cpu")
    try:
        model, checkpoint, construction = _load_frozen_target_model(checkpoint_path, device)
    except (Run1503DiagnosticError, KeyError, RuntimeError, ValueError, OSError) as exc:
        result = build_plan(args)
        result["status"] = "blocked_target_construction"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        _write_json(Path(args.output), result)
        return result
    architecture = str(construction["target_architecture"])
    dataset, dataset_path, _ = _dataset_for_checkpoint(
        checkpoint,
        dataset_path=args.dataset,
        split=args.split,
        GlobalChannelThermalDataset=GlobalChannelThermalDataset,
        H5Normalizer=H5Normalizer,
    )
    available = {str(value): index for index, value in enumerate(dataset.selected_case_ids)}
    module_counts = {
        str(case_id): int(dataset.selected_module_counts[index])
        for index, case_id in enumerate(dataset.selected_case_ids)
    }
    case_ids = _case_ids(args)
    missing = [case_id for case_id in case_ids if case_id not in available]
    if missing:
        raise KeyError(f"case IDs are absent from split {args.split!r}: {missing}")
    rows: list[dict[str, Any]] = []
    try:
        for case_id in case_ids:
            index = available[case_id]
            sample = dataset[index]
            selected, query_xy = _query_sample(sample, int(args.query_count))
            prediction = predict_case(
                model,
                selected,
                device,
                query_batch_size=int(args.query_batch_size),
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.0,
                return_routing_maps=True,
                return_prepared_state=True,
            )
            payload = _runtime_payload_for_prediction(prediction)
            row = collect_case_diagnostic(
                payload,
                case_id=case_id,
                module_count=module_counts.get(case_id),
                receiver_chunk_size=int(args.receiver_chunk_size),
            )
            row["opening"]["spatial_locality"] = opening_locality_diagnostics(
                query_xy,
                _lookup(payload, "query_assignment", required=True),
                row["geometry"]["centroid"],
                row["geometry"]["radius"],
            )
            row["query_selection"] = str(selected.get("query_selection", "unknown"))
            row["query_count"] = int(query_xy.shape[0])
            prediction_field = prediction.get("pred_field_grid")
            row["prediction_finite"] = (
                None
                if prediction_field is None
                else bool(np.isfinite(np.asarray(prediction_field)).all())
            )
            row["timing"] = _runtime_timing_scopes(
                model=model,
                selected=selected,
                query_xy=query_xy,
                device=device,
                query_batch_size=int(args.query_batch_size),
                receiver_chunk_size=int(args.receiver_chunk_size),
                warmups=int(args.warmups),
                repetitions=int(args.repetitions),
            )
            rows.append(row)
            del prediction, selected, sample
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        del model, dataset
        gc.collect()
    output = Path(args.output).expanduser().resolve()
    result = {
        "schema_version": 1,
        "task": "run1503_adaptive_hyperedge_opening_frozen_diagnostic",
        "status": "complete",
        "architecture": architecture,
        "frozen_construction": construction,
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "split": str(args.split),
        "protocol": _protocol(args, checkpoint_path),
        "cases": rows,
        "expected_aux_contract": expected_aux_contract(),
        "timing": {
            "scope": "per-case; see cases[*].timing",
            "receiver_chunk_match": int(args.receiver_chunk_size),
            "role": "structural CPU timing only; not the formal GPU execution benchmark",
            "formal_gpu_benchmark": False,
            "timed_maps": False,
            "timed_profiler": False,
        },
        "interpretation": (
            "Run-1502 epoch-500 weights were strictly loaded into the parameter-compatible "
            "Run-1503 reader with architecture selector changed in memory only. This is a "
            "frozen operator/organization diagnostic, not a retrained Run-1503 accuracy forecast."
        ),
    }
    _write_json(output, result)
    _write_case_csv(output.with_suffix(".csv"), rows)
    return result


def _smoke_gradient_probe(
    scalar: Any,
    named_parameters: Sequence[tuple[str, Any]],
    *,
    name: str,
) -> dict[str, Any]:
    """Take one retained-graph gradient probe for a named physical path."""

    import torch

    selected = [
        (parameter_name, parameter)
        for parameter_name, parameter in named_parameters
        if parameter.requires_grad
    ]
    result: dict[str, Any] = {
        "path": str(name),
        "matched_parameter_count": len(selected),
        "parameter_names": [parameter_name for parameter_name, _ in selected],
        "finite": False,
        "nonzero": False,
        "gradient_norm": None,
    }
    if not torch.is_tensor(scalar):
        result.update(status="missing", reason="path tensor was not returned")
        return result
    result["scalar_mean"] = float(scalar.detach().mean().cpu())
    if not bool(torch.isfinite(scalar.detach()).all()):
        result.update(status="failed", reason="path tensor is non-finite")
        return result
    if not scalar.requires_grad:
        result.update(status="failed", reason="path tensor is detached")
        return result
    if not selected:
        result.update(status="failed", reason="no path parameters matched")
        return result
    branch_loss = scalar.square().mean()
    gradients = torch.autograd.grad(
        branch_loss,
        [parameter for _, parameter in selected],
        retain_graph=True,
        allow_unused=True,
    )
    squared = 0.0
    finite = True
    for gradient in gradients:
        if gradient is None:
            continue
        finite = finite and bool(torch.isfinite(gradient.detach()).all())
        squared += float(gradient.detach().double().square().sum().cpu())
    norm = math.sqrt(squared)
    result.update(
        status="ok" if finite and norm > 0.0 else "failed",
        finite=bool(finite),
        nonzero=bool(norm > 0.0),
        gradient_norm=float(norm),
        probe_scalar="mean_square",
    )
    return result


def _run_predicted_port_optimizer_step(
    maintained: Any,
    model: Any,
    optimizer: Any,
    dataset: Any,
    case_ids: Sequence[str],
    device: Any,
    *,
    args: argparse.Namespace,
    train_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one physical predicted-port step and retain branch gradients."""

    import torch
    from channelthermal.data.collation import ChannelThermalBatchCollator
    from channelthermal.training.epoch import (
        _fp64_group_norm,
        assemble_channelthermal_loss_terms,
        make_model_inputs,
    )
    from torch.utils.data import DataLoader, Subset

    from honf_runtime.compat import recursive_to_device

    indices_by_case = {
        str(case_id): index for index, case_id in enumerate(dataset.selected_case_ids)
    }
    indices = [indices_by_case[str(case_id)] for case_id in case_ids]
    if not indices:
        raise ValueError("predicted-port smoke requires at least one case")
    dataset_subset = Subset(dataset, indices)
    dataset_cfg = train_config.get("dataset", {})
    if not isinstance(dataset_cfg, Mapping):
        dataset_cfg = {}
    collator = ChannelThermalBatchCollator(
        dynamic_module_padding=bool(dataset_cfg.get("dynamic_module_padding", True)),
        max_modules_per_batch=dataset_cfg.get("max_modules_per_batch"),
    )
    loader = DataLoader(
        dataset_subset,
        batch_size=min(int(args.batch_size), len(indices)),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    batch = recursive_to_device(next(iter(loader)), device)
    loss_cfg = train_config.get("loss")
    if not isinstance(loss_cfg, Mapping) or not loss_cfg:
        raise RuntimeError(
            "predicted-port smoke requires the full resolved forward config with a non-empty loss mapping"
        )
    if "predicted_consistency_weight" not in loss_cfg:
        raise RuntimeError(
            "resolved forward loss config is missing predicted_consistency_weight; refusing a fallback objective"
        )
    configured_predicted_weight = float(loss_cfg["predicted_consistency_weight"])
    if not math.isfinite(configured_predicted_weight) or configured_predicted_weight <= 0.0:
        raise RuntimeError(
            "predicted-port smoke requires a finite positive predicted_consistency_weight"
        )
    training_cfg = train_config.get("training", {})
    if not isinstance(training_cfg, Mapping):
        training_cfg = {}
    # This smoke is explicitly predicted-port, independent of any curriculum
    # epoch.  It still uses the maintained physical loss assembly below.
    local_port_condition_mode = "predicted"
    mixed_teacher_ratio = 0.0
    internal_weight, interface_weight = maintained.effective_local_loss_weights(
        dict(loss_cfg), local_port_condition_mode, mixed_teacher_ratio
    )
    predicted_weight = maintained.predicted_consistency_weight_for_epoch(
        1, dict(loss_cfg)
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    model_inputs = make_model_inputs(
        batch,
        local_port_condition_mode=local_port_condition_mode,
        mixed_teacher_ratio=mixed_teacher_ratio,
        return_predicted_port_outputs=bool(predicted_weight > 0.0),
        return_port_global_consistency=False,
    )
    # Routing maps are needed only to retain the new operator's live branch
    # tensors.  They are not timed or serialized as benchmark outputs.
    model_inputs["return_routing_maps"] = True
    model_inputs["return_prepared_state"] = True
    output = model(**model_inputs)
    loss_terms = assemble_channelthermal_loss_terms(
        output,
        batch,
        model,
        dict(loss_cfg),
        local_port_condition_mode=local_port_condition_mode,
        mixed_teacher_ratio=mixed_teacher_ratio,
        effective_internal_temperature_weight=internal_weight,
        effective_interface_weight=interface_weight,
        predicted_consistency_weight=predicted_weight,
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
    ]
    backend = "core.backend."
    path_prefixes = {
        "coarse_path": (
            f"{backend}environment_value_control.",
            f"{backend}env_attention.output.",
        ),
        "fine_path": (
            f"{backend}environment_score_control.",
            f"{backend}env_geometry_bias.",
            f"{backend}env_attention.output.",
        ),
        "module_reader": (
            f"{backend}module_control_gain.",
            f"{backend}query_module_message.",
            f"{backend}module_message.",
            f"{backend}module_membership.",
            f"{backend}router.module_",
        ),
    }
    interaction_aux = output.get("interaction_aux", {})
    if not isinstance(interaction_aux, Mapping):
        interaction_aux = {}
    prepared_case = output.get("prepared_state")
    prepared = getattr(prepared_case, "prepared", None)
    if prepared is None:
        raise RuntimeError("predicted-port smoke did not retain an interface prepared state")
    receivers = batch["query_xy"].float()
    receiver_features = model.core._receiver_features(prepared, receivers)
    backend = model.core.backend
    route = backend._route(
        prepared.backend_state,
        prepared.encoded,
        receivers,
        receiver_features,
    )
    _backend_context, live_backend_aux = backend._read_environment(
        prepared.backend_state,
        prepared.encoded,
        receivers,
        receiver_features,
        route,
        include_diagnostics=True,
    )
    path_tensors = {
        # InterfaceFieldCore intentionally detaches public diagnostics.  The
        # backend probe above is the bounded live graph, not a synthetic
        # reconstruction from detached serialized maps.
        "coarse_path": live_backend_aux.get("group_control_adaptive_coarse_contribution"),
        "fine_path": live_backend_aux.get("group_control_adaptive_fine_contribution"),
    }
    branch_gradients = {}
    for path_name, prefixes in path_prefixes.items():
        selected = [
            (parameter_name, parameter)
            for parameter_name, parameter in named_parameters
            if parameter_name.startswith(prefixes)
        ]
        scalar = path_tensors.get(path_name) if path_name != "module_reader" else loss_terms["loss"]
        branch_gradients[path_name] = _smoke_gradient_probe(
            scalar,
            selected,
            name=path_name,
        )
    before = {
        name: parameter.detach().clone()
        for name, parameter in named_parameters
        if parameter.requires_grad
    }
    loss = loss_terms["loss"]
    if not torch.is_tensor(loss) or not bool(torch.isfinite(loss.detach()).all()):
        raise RuntimeError("predicted-port physical loss is non-finite")
    loss.backward()
    named_gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    total_gradient, grouped_gradient = _fp64_group_norm(named_gradients)
    clip_norm = float(training_cfg.get("gradient_clip_norm", training_cfg.get("grad_clip_norm", 0.0)))
    if clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    named_updates = [
        (name, parameter.detach() - before[name])
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in before
    ]
    total_update, grouped_update = _fp64_group_norm(named_updates)
    max_delta = max(
        (float(delta.detach().abs().max().cpu()) for _, delta in named_updates),
        default=0.0,
    )
    parameters_finite = all(
        bool(torch.isfinite(parameter.detach()).all())
        for _, parameter in model.named_parameters()
        if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
    )
    optimizer_state_values = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    ]
    optimizer_state_finite = bool(optimizer_state_values) and all(
        bool(torch.isfinite(value.detach()).all()) for value in optimizer_state_values
    )
    metrics = {
        "loss_total": float(loss.detach().cpu()),
        "preclip_gradient_norm": float(total_gradient),
        "parameter_update_norm": float(total_update),
        **{f"preclip_gradient_norm_{key}": value for key, value in grouped_gradient.items()},
        **{f"parameter_update_norm_{key}": value for key, value in grouped_update.items()},
    }
    return {
        "case_ids": [str(case_id) for case_id in case_ids],
        "module_counts": [
            int(dataset.selected_module_counts[indices_by_case[str(case_id)]])
            for case_id in case_ids
        ],
        "batch_size": int(min(int(args.batch_size), len(indices))),
        "points_per_case": int(args.points_per_case),
        "metrics": metrics,
        "canonical_loss_config": {
            "key_count": len(loss_cfg),
            "predicted_consistency_weight_configured": configured_predicted_weight,
            "predicted_consistency_weight_used": float(predicted_weight),
        },
        "branch_gradients": branch_gradients,
        "max_parameter_delta": float(max_delta),
        "parameters_finite": bool(parameters_finite),
        "optimizer_state_finite": bool(optimizer_state_finite),
        "optimizer_state_tensor_count": len(optimizer_state_values),
        "optimizer_update_applied": bool(max_delta > 0.0),
        "adaptive_aux_keys_present": sorted(
            key
            for key in interaction_aux
            if str(key).startswith("group_control_adaptive_")
        ),
    }


def _run_predicted_port_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Build the candidate profile and execute its two disposable CPU steps."""

    import run_regional_response_study as maintained
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.model import ChannelThermalHONFModel
    from channelthermal.training.optimizer import build_forward_optimizer
    from channelthermal.workflows.train_forward import (
        build_model_config,
        resolve_auto_internal_mode,
        set_seed,
    )

    from honf_runtime.case_protocol import WorkflowRequest
    from honf_runtime.config_loader import load_config_bundle
    from honf_runtime.registry import load_case_plugin

    device = maintained.select_device("cpu")
    bundle = load_config_bundle(str(args.profile))
    plugin = load_case_plugin(str(bundle.case["plugin"]))
    config = plugin._forward_config(
        bundle,
        WorkflowRequest(workflow="forward", device="cpu", epochs=500),
        Path(args.output).expanduser().resolve().parent,
    )
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    set_seed(int(train_cfg["seed"]))
    dataset = GlobalChannelThermalDataset(
        dataset_cfg["packed_h5_path"],
        split=dataset_cfg["train_split"],
        points_per_case=int(args.points_per_case),
        normalize_inputs=bool(dataset_cfg["normalize_inputs"]),
        normalize_targets=bool(dataset_cfg["normalize_targets"]),
        random_point_sampling=False,
        seed=int(train_cfg["seed"]),
        require_converged=bool(dataset_cfg.get("require_converged", False)),
    )
    model = None
    try:
        model_config = build_model_config(config, dataset)
        model = ChannelThermalHONFModel(model_config).to(device)
        model.set_global_target_normalization(
            dataset.normalizer.stats,
            normalize_targets=bool(dataset_cfg["normalize_targets"]),
        )
        resolve_auto_internal_mode(model_config, model)
        optimizer, _ = build_forward_optimizer(model, train_cfg)
        case_ids_small = tuple(
            str(value)
            for value in (
                args.small_case_id
                or maintained._smoke_case_ids(dataset, count=int(args.case_count), large=False)
            )
        )
        case_ids_large = tuple(
            str(value)
            for value in (
                args.large_case_id
                or maintained._smoke_case_ids(dataset, count=int(args.case_count), large=True)
            )
        )
        available = {str(value) for value in dataset.selected_case_ids}
        missing = sorted(
            set(case_ids_small + case_ids_large).difference(available)
        )
        if missing:
            raise KeyError(f"smoke case IDs are absent from the train split: {missing}")
        steps = {
            "small_module_batch": _run_predicted_port_optimizer_step(
                maintained,
                model,
                optimizer,
                dataset,
                case_ids_small,
                device,
                args=args,
                train_config=config,
            ),
            "large_module_batch": _run_predicted_port_optimizer_step(
                maintained,
                model,
                optimizer,
                dataset,
                case_ids_large,
                device,
                args=args,
                train_config=config,
            ),
        }
        return {
            "schema_version": 1,
            "task": "smoke",
            "stage": "physical_predicted_port_optimizer_steps",
            "profile": str(args.profile),
            "architecture": "adaptive_hyperedge_opening_honf",
            "fresh_model": True,
            "steps": steps,
            "managed_run_reserved": False,
            "checkpoint_saved": False,
        }
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        if model is not None:
            del model
        gc.collect()


def _smoke_gradient_evidence(step: Mapping[str, Any]) -> dict[str, Any]:
    """Validate finite real optimizer evidence returned by the maintained smoke."""

    metrics = step.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return {"status": "missing", "reason": "smoke step has no metrics mapping"}
    required = (
        "loss_total",
        "preclip_gradient_norm",
        "parameter_update_norm",
    )
    detail_groups = (
        "backend",
        "group_prepare",
        "group_receiver",
        "coarse",
        "head",
        "local_coupling",
    )
    finite: dict[str, bool] = {}
    for key in required:
        try:
            finite[key] = bool(math.isfinite(float(metrics[key])))
        except (KeyError, TypeError, ValueError):
            finite[key] = False
    for group in detail_groups:
        for prefix in ("preclip_gradient_norm", "parameter_update_norm"):
            key = f"{prefix}_{group}"
            if key in metrics:
                try:
                    finite[key] = bool(math.isfinite(float(metrics[key])))
                except (TypeError, ValueError):
                    finite[key] = False
    positive_detail_updates = {
        key: float(metrics[key])
        for key in finite
        if key.startswith("parameter_update_norm_")
        and finite[key]
        and float(metrics[key]) > 0.0
    }
    state_checks = {
        "parameters_finite": bool(step.get("parameters_finite", False)),
        "optimizer_state_finite": bool(step.get("optimizer_state_finite", False)),
        "optimizer_update_applied": bool(step.get("optimizer_update_applied", False)),
    }
    canonical_loss = step.get("canonical_loss_config", {})
    canonical_loss_ok = False
    if isinstance(canonical_loss, Mapping):
        try:
            canonical_loss_ok = (
                int(canonical_loss.get("key_count", 0)) > 0
                and math.isfinite(float(canonical_loss["predicted_consistency_weight_configured"]))
                and float(canonical_loss["predicted_consistency_weight_configured"]) > 0.0
                and math.isfinite(float(canonical_loss["predicted_consistency_weight_used"]))
                and float(canonical_loss["predicted_consistency_weight_used"]) > 0.0
            )
        except (KeyError, TypeError, ValueError):
            canonical_loss_ok = False
    return {
        "status": (
            "ok"
            if all(finite.values())
            and all(state_checks.values())
            and positive_detail_updates
            and canonical_loss_ok
            else "failed"
        ),
        "required_finite": finite,
        "state_checks": state_checks,
        "canonical_loss_config": canonical_loss,
        "canonical_loss_config_ok": canonical_loss_ok,
        "positive_detail_updates": positive_detail_updates,
        "loss": float(metrics["loss_total"]) if finite.get("loss_total") else None,
        "total_gradient_norm": (
            float(metrics["preclip_gradient_norm"])
            if finite.get("preclip_gradient_norm")
            else None
        ),
        "total_parameter_update_norm": (
            float(metrics["parameter_update_norm"])
            if finite.get("parameter_update_norm")
            else None
        ),
    }


def run_predicted_port_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run one real CPU AdamW step for minimum and maximum module-count cases.

    The maintained physical-batch helper owns dataset collation and loss
    assembly.  This wrapper supplies the Run1503 profile and turns its scalar
    gradient/update rows into an explicit acceptance record.  The model and
    optimizer are disposable; no checkpoint or managed run is created.
    """

    if str(args.device) != "cpu":
        raise ValueError("Run1503 predicted-port smoke is CPU-only in this task; pass --device cpu")
    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        raw = _run_predicted_port_runtime(args)
    except Exception as exc:  # noqa: BLE001 - preserve bounded smoke failure evidence
        result = {
            "schema_version": 1,
            "task": "run1503_predicted_port_optimizer_smoke",
            "status": "blocked_or_failed",
            "profile": str(args.profile),
            "device": "cpu",
            "reason": f"{type(exc).__name__}: {exc}",
            "training_launched": False,
            "checkpoint_saved": False,
        }
        _write_json(Path(args.output), result)
        return result
    steps = raw.get("steps", {}) if isinstance(raw, Mapping) else {}
    evidence: dict[str, Any] = {}
    module_count_values: list[int] = []
    for name, step in steps.items():
        if not isinstance(step, Mapping):
            evidence[str(name)] = {"status": "missing"}
            continue
        evidence[str(name)] = _smoke_gradient_evidence(step)
        module_count_values.extend(int(value) for value in step.get("module_counts", []))
    expected_steps = {"small_module_batch", "large_module_batch"}
    present_steps = {str(name) for name in steps}
    finite_updates = all(row.get("status") == "ok" for row in evidence.values())
    path_evidence = {
        str(name): step.get("branch_gradients", {})
        for name, step in steps.items()
        if isinstance(step, Mapping)
    }
    path_names = ("coarse_path", "fine_path", "module_reader")
    path_updates = all(
        isinstance(paths, Mapping)
        and all(paths.get(path, {}).get("status") == "ok" for path in path_names)
        for paths in path_evidence.values()
    )
    has_min_max = (
        bool(module_count_values)
        and min(module_count_values) < max(module_count_values)
        and expected_steps.issubset(present_steps)
    )
    result = {
        "schema_version": 1,
        "task": "run1503_predicted_port_optimizer_smoke",
        "status": "complete" if finite_updates and has_min_max and path_updates else "failed",
        "architecture": "adaptive_hyperedge_opening_honf",
        "profile": str(args.profile),
        "device": "cpu",
        "port_condition": "predicted",
        "steps": steps,
        "gradient_update_evidence": evidence,
        "adaptive_path_gradient_evidence": path_evidence,
        "module_count_policy": "one real optimizer step for deterministic minimum and maximum active-module cases",
        "module_count_range_observed": (
            None if not module_count_values else [min(module_count_values), max(module_count_values)]
        ),
        "training_launched": False,
        "checkpoint_saved": False,
        "managed_run_reserved": False,
        "limitations": [
            "Disposable physical optimizer smoke only; no accuracy claim.",
            "Finite loss/gradient/update evidence does not establish Run1503 learning health at epoch 50.",
        ],
    }
    _write_json(Path(args.output), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="explicit Run-1502 epoch-500 checkpoint")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    parser.add_argument("--receiver-chunk-size", type=int, default=DEFAULT_RECEIVER_CHUNK_SIZE)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plan-only", action="store_true", help="write a CPU-safe plan without loading runtime")
    parser.add_argument("--smoke", action="store_true", help="run the disposable CPU predicted-port optimizer smoke")
    parser.add_argument(
        "--profile",
        default="project://src/config_core/forward/adaptive_hyperedge_opening_honf_context.json",
        help="profile used by --smoke",
    )
    parser.add_argument("--points-per-case", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--case-count", type=int, default=1)
    parser.add_argument("--small-case-id", action="append", default=None)
    parser.add_argument("--large-case-id", action="append", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        result = run_predicted_port_smoke(args)
    else:
        result = build_plan(args) if args.plan_only else run_frozen(args)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0 if result.get("status") in {"plan_only", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUX_ALIASES",
    "CASE_SELECTION_REASONS",
    "DEFAULT_CASE_IDS",
    "MODULE_PANEL_CASE_IDS",
    "REPRESENTATIVE_CASE_IDS",
    "Run1503DiagnosticError",
    "benchmark_application_scopes",
    "build_parser",
    "build_plan",
    "collect_case_diagnostic",
    "expected_aux_contract",
    "group_geometry",
    "main",
    "opening_diagnostics",
    "opening_locality_diagnostics",
    "run_frozen",
    "run_predicted_port_smoke",
]
