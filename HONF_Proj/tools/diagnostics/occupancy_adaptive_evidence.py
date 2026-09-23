"""CPU-side evidence extraction for occupancy-adaptive Run 1409.

The live occupancy backend owns the runtime ``OccupancyGroupPlan``.  This
module only reads detached output/debug payloads and writes evidence under an
explicit caller supplied directory.  It accepts the core names used by the
new router as well as the ordinary ``group_control_*`` names used by the
Run-1406 reader, which makes the report tool usable while the wrapper API is
being integrated.

Typical offline use after a map pass is::

    PYTHONPATH=src \
      python tools/diagnostics/occupancy_adaptive_evidence.py \
      --payload /tmp/run1409_maps.json \
      --output-dir /path/to/Run_1409/evaluations/occupancy_adaptive

The script never chooses a checkpoint from a run number and never writes a
checkpoint.  ``--plan-only`` performs no model, dataset, or CUDA import.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CASE_COUNT = 90
DEFAULT_KMAX = 12
DEFAULT_QUERY_COUNT = 8192
DEFAULT_CASE_IDS = ("0273", "0653")


class OccupancyEvidenceError(ValueError):
    """Raised when a map payload cannot be audited without inference."""


_ALIASES: dict[str, tuple[str, ...]] = {
    "module_assignment": (
        "module_assignment",
        "occupancy_group_module_membership",
        "occupancy_adaptive_module_membership",
        "occupancy_module_membership",
        "occupancy_group_A_m",
        "group_control_module_incidence",
        "group_control_module_membership",
        "module_membership",
        "A_m",
        "A_mh",
    ),
    "environment_assignment": (
        "environment_assignment",
        "occupancy_group_environment_membership",
        "occupancy_adaptive_environment_membership",
        "occupancy_environment_membership",
        "occupancy_group_A_e",
        "group_control_environment_incidence",
        "group_control_environment_membership",
        "environment_membership",
        "A_e",
        "A_eh",
    ),
    "query_assignment": (
        "query_assignment",
        "occupancy_group_query_routing",
        "occupancy_adaptive_query_routing",
        "occupancy_query_routing",
        "group_control_query_routing",
        "group_control_query_alpha",
        "query_routing",
        "alpha_qk",
    ),
    "proposal_module": (
        "occupancy_group_proposal_module_membership",
        "occupancy_adaptive_proposal_module_membership",
        "proposal_module_membership",
        "proposal_A_m",
    ),
    "proposal_environment": (
        "occupancy_group_proposal_environment_membership",
        "occupancy_adaptive_proposal_environment_membership",
        "proposal_environment_membership",
        "proposal_A_e",
    ),
    "module_measure": (
        "occupancy_group_module_measure",
        "occupancy_adaptive_module_measure",
        "group_control_module_measure",
        "module_measure",
        "omega_m",
    ),
    "environment_measure": (
        "occupancy_group_environment_measure",
        "occupancy_adaptive_environment_measure",
        "group_control_environment_measure",
        "environment_measure",
        "omega_e",
        "env_weights",
    ),
    "module_coords": ("module_coords", "module_centers", "module_centres"),
    "environment_coords": ("env_coords", "environment_coords", "environment_centers", "environment_centres"),
    "query_coords": ("query_xy", "query_coords", "receiver_coords", "receivers"),
    "query_grid_indices": ("query_grid_indices", "original_query_indices"),
    "query_source_count": ("query_source_count", "original_query_count", "source_query_count"),
    "query_selection": ("query_selection", "query_grid_scope"),
    "query_domain_bounds": ("query_domain_bounds", "original_query_domain_bounds"),
    "module_present": ("module_present", "active_modules", "module_valid"),
    "prototype_ids": (
        "occupancy_group_prototype_ids",
        "occupancy_adaptive_prototype_ids",
        "prototype_ids",
    ),
    "packed_prototype_ids": (
        "occupancy_group_packed_prototype_ids",
        "occupancy_adaptive_packed_prototype_ids",
        "packed_prototype_ids",
        "packed_ids",
    ),
    "packed_valid": (
        "occupancy_group_packed_valid",
        "occupancy_adaptive_packed_valid",
        "packed_valid",
    ),
    "active_mask": (
        "occupancy_group_active_mask",
        "occupancy_adaptive_active_mask",
        "active_mask",
        "occupancy_support",
    ),
    "kplan": ("occupancy_group_k_plan", "occupancy_adaptive_k_plan", "K_plan", "k_plan"),
    "kappa": ("occupancy_group_kappa", "occupancy_adaptive_kappa", "kappa"),
    "module_mass": (
        "occupancy_group_module_mass",
        "occupancy_adaptive_module_mass",
        "group_control_module_mass",
        "module_mass",
    ),
    "environment_mass": (
        "occupancy_group_environment_mass",
        "occupancy_adaptive_environment_mass",
        "group_control_environment_mass",
        "environment_mass",
    ),
    "module_centres": (
        "occupancy_group_module_centres",
        "occupancy_group_module_centers",
        "occupancy_adaptive_module_centres",
        "occupancy_adaptive_module_centers",
        "group_control_module_centres",
        "group_control_module_centers",
        "module_group_centres",
        "module_group_centers",
    ),
    "environment_centres": (
        "occupancy_group_environment_centres",
        "occupancy_group_environment_centers",
        "occupancy_adaptive_environment_centres",
        "occupancy_adaptive_environment_centers",
        "group_control_environment_centres",
        "group_control_environment_centers",
        "environment_group_centres",
        "environment_group_centers",
    ),
    "joint_centres": (
        "occupancy_group_joint_centres",
        "occupancy_group_joint_centers",
        "occupancy_adaptive_joint_centres",
        "occupancy_adaptive_joint_centers",
        "joint_centres",
        "joint_centers",
    ),
    "occupancy_group_phase_ledger": (
        "occupancy_group_phase_ledger",
        "occupancy_adaptive_phase_ledger",
        "group_control_phase_ledger",
        "phase_ledger",
    ),
}


def _array(value: Any, *, name: str, allow_none: bool = False) -> np.ndarray | None:
    if value is None:
        if allow_none:
            return None
        raise OccupancyEvidenceError(f"missing array {name!r}")
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise OccupancyEvidenceError(f"cannot convert {name!r} to NumPy") from exc
    if result.dtype == object:
        raise OccupancyEvidenceError(f"array {name!r} has object dtype")
    return result


def _jsonable(value: Any) -> Any:
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
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _containers(payload: Any) -> list[Mapping[str, Any]]:
    """Collect shallow debug containers without traversing large tensors."""

    containers: list[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        for key in (
            "occupancy_group_debug",
            "occupancy_adaptive_debug",
            "interaction_aux",
            "routing_aux",
            "group_control_debug",
            "prepared_state",
            "backend_state",
            "phase_shared_state",
        ):
            value = payload.get(key)
            if isinstance(value, Mapping):
                containers.extend(_containers(value))
            elif is_dataclass(value):
                containers.extend(_containers(asdict(value)))
                containers.append({key: value})
        containers.append(payload)
    elif is_dataclass(payload):
        containers.append(asdict(payload))
    else:
        for key in (
            "interaction_aux",
            "occupancy_group_debug",
            "occupancy_adaptive_debug",
            "backend_state",
            "phase_shared_state",
        ):
            value = getattr(payload, key, None)
            if isinstance(value, Mapping):
                containers.extend(_containers(value))
            elif is_dataclass(value):
                containers.extend(_containers(asdict(value)))
            elif value is not None:
                containers.append({key: value})
    return containers


def _lookup(payload: Any, name: str, *, required: bool = False) -> Any:
    aliases = _ALIASES.get(name, (name,))
    for container in _containers(payload):
        for alias in aliases:
            if alias in container:
                return container[alias]
        for alias in aliases:
            value = container.get(alias)
            if value is not None:
                return value
    if required:
        raise OccupancyEvidenceError(f"missing {name}; accepted keys: {', '.join(aliases)}")
    return None


def _plan_object(payload: Any) -> Any:
    for container in _containers(payload):
        for key in ("occupancy_group_plan", "occupancy_adaptive_plan", "plan"):
            value = container.get(key)
            if value is not None and not isinstance(value, (str, bytes, int, float, np.ndarray)):
                return value
    return None


def _plan_attr(payload: Any, name: str) -> Any:
    plan = _plan_object(payload)
    if plan is None:
        return None
    if isinstance(plan, Mapping):
        return plan.get(name)
    return getattr(plan, name, None)


def _lookup_with_plan(payload: Any, name: str) -> Any:
    value = _lookup(payload, name)
    if value is not None:
        return value
    plan_names = {
        "prototype_ids": ("prototype_ids",),
        "packed_prototype_ids": ("packed_prototype_ids",),
        "packed_valid": ("packed_valid",),
        "active_mask": ("active_mask",),
        "kplan": ("k_plan",),
        "kappa": ("kappa",),
        "module_mass": ("module_mass",),
        "environment_mass": ("environment_mass",),
        "joint_centres": ("centres",),
    }
    for attr in plan_names.get(name, (name,)):
        value = _plan_attr(payload, attr)
        if value is not None:
            return value
    return None


def _squeeze_batch(value: np.ndarray | None, *, case_index: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] > 1:
        if not 0 <= int(case_index) < int(array.shape[0]):
            raise OccupancyEvidenceError(f"{name} case index is outside batch size {array.shape[0]}")
        return array[int(case_index)]
    if array.ndim >= 1 and array.shape[0] == 1:
        return array[0]
    return array


def _ensure_source_assignment(value: Any, *, name: str, case_index: int, source_count: int | None = None) -> np.ndarray:
    array = _squeeze_batch(_array(value, name=name), case_index=case_index, name=name)
    if array is None or array.ndim != 2:
        raise OccupancyEvidenceError(f"{name} must have shape [source,K], got {None if array is None else array.shape}")
    if source_count is not None and int(array.shape[0]) != int(source_count):
        raise OccupancyEvidenceError(f"{name} source count does not match geometry")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise OccupancyEvidenceError(f"{name} must be finite and nonnegative")
    return result


def _ensure_query_assignment(value: Any, *, name: str, case_index: int, query_count: int) -> np.ndarray:
    array = _squeeze_batch(_array(value, name=name), case_index=case_index, name=name)
    if array is None or array.ndim != 2:
        raise OccupancyEvidenceError(f"{name} must have shape [query,K]")
    if array.shape[0] == query_count:
        result = array
    elif array.shape[1] == query_count:
        result = array.T
    else:
        raise OccupancyEvidenceError(f"{name} does not match query count {query_count}: {array.shape}")
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise OccupancyEvidenceError(f"{name} must be finite and nonnegative")
    return result


def _geometry(payload: Any, name: str, *, case_index: int, dimension: int | None = None) -> np.ndarray:
    value = _lookup(payload, name, required=True)
    array = _squeeze_batch(_array(value, name=name), case_index=case_index, name=name)
    if array is None or array.ndim != 2 or (dimension is not None and int(array.shape[1]) != int(dimension)):
        raise OccupancyEvidenceError(f"{name} must have shape [N,d], got {None if array is None else array.shape}")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all() or not result.shape[0]:
        raise OccupancyEvidenceError(f"{name} must be finite and nonempty")
    return result


def _vector(payload: Any, name: str, *, case_index: int, length: int, default: np.ndarray | None = None) -> np.ndarray:
    value = _lookup(payload, name)
    if value is None:
        if default is None:
            return np.ones(length, dtype=np.float64)
        return np.asarray(default, dtype=np.float64)
    array = _squeeze_batch(_array(value, name=name), case_index=case_index, name=name)
    if array is None:
        return np.ones(length, dtype=np.float64)
    result = np.asarray(array, dtype=np.float64).reshape(-1)
    if result.size != length:
        raise OccupancyEvidenceError(f"{name} length {result.size} does not match {length}")
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise OccupancyEvidenceError(f"{name} must be finite and nonnegative")
    total = float(result.sum())
    if total <= 0.0:
        raise OccupancyEvidenceError(f"{name} must have positive total mass")
    return result / total


def _normalise_rows(values: np.ndarray, *, valid: np.ndarray | None = None, name: str) -> dict[str, Any]:
    row_sums = values.sum(axis=-1)
    if valid is None:
        valid = np.ones(values.shape[0], dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool).reshape(-1)
    if valid.size != values.shape[0]:
        raise OccupancyEvidenceError(f"{name} valid mask does not align with rows")
    active_error = np.abs(row_sums[valid] - 1.0) if np.any(valid) else np.zeros(0)
    inactive_zero = bool(np.all(row_sums[~valid] == 0.0))
    return {
        "active_rows_normalized": bool(np.all(active_error <= 2.0e-5)),
        "inactive_rows_zero": inactive_zero,
        "max_active_row_error": float(np.max(active_error)) if active_error.size else 0.0,
        "positive_degree_mean": float(np.mean((values[valid] > 0.0).sum(axis=-1))) if np.any(valid) else 0.0,
        "empty_columns": np.flatnonzero(~np.any(values[valid] > 0.0, axis=0)).astype(int).tolist() if np.any(valid) else list(range(values.shape[1])),
    }


def _effective(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, None)
    return 1.0 / np.maximum(np.sum(values * values, axis=-1), np.finfo(np.float64).tiny)


def _entropy(values: np.ndarray, kmax: int) -> np.ndarray:
    values = np.clip(values, 0.0, None)
    positive = values > 0.0
    log_values = np.zeros_like(values)
    log_values[positive] = np.log(values[positive])
    raw = -np.sum(values * log_values, axis=-1)
    return raw / max(math.log(max(kmax, 2)), 1.0)


def _centres(coords: np.ndarray, measure: np.ndarray, assignment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mass = np.einsum("s,sk->k", measure, assignment)
    numerator = np.einsum("s,sk,sd->kd", measure, assignment, coords)
    centre = np.zeros((assignment.shape[1], coords.shape[1]), dtype=np.float64)
    valid = mass > 0.0
    centre[valid] = numerator[valid] / mass[valid, None]
    return mass, centre


def _centres_from_payload(payload: Any, name: str, *, case_index: int, fallback: np.ndarray) -> np.ndarray:
    value = _lookup_with_plan(payload, name)
    if value is None:
        return fallback
    raw = _array(value, name=name)
    # The runtime plan stores centres as [source-type,B,K,d], with source
    # types ordered module, environment, joint. Handle that layout before the
    # ordinary batch squeeze so B=1 cannot be mistaken for source type.
    if name == "joint_centres" and raw is not None and raw.ndim == 4 and raw.shape[0] == 3:
        array = raw[2, int(case_index)]
    else:
        array = _squeeze_batch(raw, case_index=case_index, name=name)
    if array is None:
        return fallback
    array = np.asarray(array, dtype=np.float64)
    # The runtime plan stores centres as [3,B,K,d].
    if name == "joint_centres" and array.ndim == 3 and array.shape[0] == 3:
        array = array[2]
    if array.shape != fallback.shape:
        raise OccupancyEvidenceError(f"{name} shape {array.shape} does not match {fallback.shape}")
    if not np.isfinite(array).all():
        raise OccupancyEvidenceError(f"{name} contains non-finite values")
    return array


def _pair_support(query: np.ndarray, source: np.ndarray, valid: np.ndarray | None) -> tuple[int, int, int, float, float, np.ndarray]:
    q_positive = query > 0.0
    s_positive = source > 0.0
    if valid is not None:
        s_positive &= np.asarray(valid, dtype=bool)[:, None]
    logical = q_positive.astype(np.int64) @ s_positive.astype(np.int64).T
    support = logical > 0
    unique = int(support.sum())
    paths = int(logical.sum())
    active_sources = int(s_positive.any(axis=1).sum())
    dense = int(query.shape[0] * active_sources)
    ratio = float(unique / dense) if dense else None
    multiplicity = float(paths / unique) if unique else None
    return paths, unique, dense, ratio, multiplicity, logical


def _shuffled_radius_baseline(
    coordinates: np.ndarray,
    measure: np.ndarray,
    assignment: np.ndarray,
    *,
    valid: np.ndarray | None,
    repetitions: int = 64,
    seed: int = 1409,
) -> float:
    """Shuffle coordinates against fixed source masses and memberships.

    The source rows, assignment columns, and measure weights stay fixed, so
    group mass is preserved exactly. Only spatial compactness is randomized.
    """

    valid_rows = np.ones(len(coordinates), dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    indices = np.flatnonzero(valid_rows)
    if len(indices) < 2:
        return 0.0
    mass = np.einsum("s,sk->k", measure, assignment)
    occupied = mass > 0.0
    if not np.any(occupied):
        return 0.0
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(repetitions)):
        permuted = coordinates.copy()
        permuted_indices = indices[rng.permutation(len(indices))]
        permuted[indices] = coordinates[permuted_indices]
        centres = np.einsum("s,sk,sd->kd", measure, assignment, permuted) / np.maximum(mass[:, None], np.finfo(np.float64).tiny)
        squared = np.sum((permuted[:, None, :] - centres[None, :, :]) ** 2, axis=-1)
        radius = np.sqrt(np.maximum(np.sum(measure[:, None] * assignment * squared, axis=0) / np.maximum(mass, np.finfo(np.float64).tiny), 0.0))
        samples.append(float(np.mean(radius[occupied])))
    return float(np.mean(samples))


def _phase_ledger(payload: Any) -> dict[str, Any]:
    """Read explicit P0/P1/P2 ledgers and keep absent records unavailable."""

    aliases = {
        "logical_paths": ("logical_path_count", "logical_paths", "logical_support_rows"),
        "unique_pairs": ("unique_pair_count", "unique_pairs", "support_pairs"),
        "actual_rows": ("actual_rows", "actual_fine_call_count", "fine_rows", "fine_rows_forward", "executed_rows"),
        "padded_rows": ("padded_rows", "padded_fine_rows", "fine_rows_padded"),
        "geometry_rows": (
            "geometry_rows",
            "geometry_rows_forward",
            "environment_geometry_network_rows",
            "environment_geometry_network_rows_forward",
        ),
        "content_rows": (
            "content_rows",
            "content_dot_rows",
            "content_dot_rows_forward",
            "environment_content_rows",
            "environment_content_dot_rows_forward",
        ),
    }

    def phase_value(container: Mapping[str, Any], names: Sequence[str]) -> Any:
        """Read a value, summing explicit receiver-chunk records when present."""

        def direct(mapping: Mapping[str, Any]) -> Any:
            for name in names:
                if name in mapping:
                    return mapping[name]
            return None

        chunk_values: list[float] = []
        for chunk_key in ("chunks", "receiver_chunks", "chunk_records", "records"):
            chunks = container.get(chunk_key)
            if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
                continue
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    continue
                value = direct(chunk)
                try:
                    numeric = float(np.asarray(value).reshape(-1)[0])
                except (TypeError, ValueError, IndexError):
                    continue
                if math.isfinite(numeric):
                    chunk_values.append(numeric)
            if chunk_values:
                return sum(chunk_values)
        return direct(container)

    phases = {}
    containers = _containers(payload)
    explicit = _lookup(payload, "occupancy_group_phase_ledger")
    if isinstance(explicit, Mapping):
        containers.insert(0, explicit)
    for phase in ("P0", "P1", "P2", "P2_consistency"):
        found = None
        for container in containers:
            for key in (phase, phase.lower(), f"occupancy_group_{phase.lower()}_ledger", f"occupancy_adaptive_{phase.lower()}_ledger"):
                if isinstance(container.get(key), Mapping):
                    found = container[key]
                    break
            if found is not None:
                break
        if found is None:
            phases[phase] = {"status": "unavailable", "reason": "explicit phase ledger missing"}
            continue
        row: dict[str, Any] = {"status": "available"}
        for source in ("module", "environment", "M", "E"):
            nested = found.get(source)
            if not isinstance(nested, Mapping):
                continue
            source_row: dict[str, Any] = {}
            for metric, names in aliases.items():
                value = phase_value(nested, names)
                source_row[metric] = None if value is None else _jsonable(value)
            row[source] = source_row
        for metric, names in aliases.items():
            value = phase_value(found, names)
            if value is None:
                source_values = []
                for source in ("module", "environment", "M", "E"):
                    nested = found.get(source)
                    if isinstance(nested, Mapping):
                        source_value = phase_value(nested, names)
                        if source_value is not None:
                            try:
                                source_values.append(float(np.asarray(source_value).reshape(-1)[0]))
                            except (TypeError, ValueError, IndexError):
                                pass
                if source_values:
                    value = sum(source_values)
            row[metric] = None if value is None else _jsonable(value)
        phases[phase] = row
    return phases


def canonicalize_case(payload: Any, *, case_index: int = 0, query_count: int | None = None, kmax: int | None = None) -> dict[str, Any]:
    """Extract one case and compute all structural Run-1409 metrics."""

    module_coords = _geometry(payload, "module_coords", case_index=case_index)
    environment_coords = _geometry(payload, "environment_coords", case_index=case_index)
    query_value = _lookup(payload, "query_coords", required=True)
    query_array = _squeeze_batch(_array(query_value, name="query_coords"), case_index=case_index, name="query_coords")
    if query_array is None or query_array.ndim != 2:
        raise OccupancyEvidenceError("query_coords must have shape [Q,d]")
    query_array = np.asarray(query_array, dtype=np.float64)
    if query_count is not None and int(query_array.shape[0]) != int(query_count):
        raise OccupancyEvidenceError(f"query count {query_array.shape[0]} does not match requested {query_count}")
    source_count_value = _lookup(payload, "query_source_count")
    query_source_count = None
    if source_count_value is not None:
        try:
            query_source_count = int(np.asarray(source_count_value).reshape(-1)[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise OccupancyEvidenceError("query_source_count must be an integer") from exc
        if query_source_count < int(query_array.shape[0]):
            raise OccupancyEvidenceError("query_source_count cannot be smaller than recorded query count")
    selection_value = _lookup(payload, "query_selection")
    query_selection = None if selection_value is None else str(selection_value)
    domain_value = _lookup(payload, "query_domain_bounds")
    query_domain_bounds = None
    if domain_value is not None:
        try:
            query_domain_bounds = np.asarray(domain_value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise OccupancyEvidenceError("query_domain_bounds must be numeric") from exc
        if query_domain_bounds.size != 4 or not np.isfinite(query_domain_bounds).all():
            raise OccupancyEvidenceError("query_domain_bounds must contain four finite values")
        if query_domain_bounds[1] < query_domain_bounds[0] or query_domain_bounds[3] < query_domain_bounds[2]:
            raise OccupancyEvidenceError("query_domain_bounds must be ordered xmin,xmax,ymin,ymax")
    query_indices_value = _lookup(payload, "query_grid_indices")
    query_indices = None
    if query_indices_value is not None:
        query_indices = np.asarray(query_indices_value).reshape(-1).astype(np.int64)
        if query_indices.size != int(query_array.shape[0]):
            raise OccupancyEvidenceError("query_grid_indices does not align with recorded query count")
    module = _ensure_source_assignment(
        _lookup(payload, "module_assignment", required=True),
        name="module_assignment",
        case_index=case_index,
        source_count=len(module_coords),
    )
    environment = _ensure_source_assignment(
        _lookup(payload, "environment_assignment", required=True),
        name="environment_assignment",
        case_index=case_index,
        source_count=len(environment_coords),
    )
    query = _ensure_query_assignment(
        _lookup(payload, "query_assignment", required=True),
        name="query_assignment",
        case_index=case_index,
        query_count=len(query_array),
    )
    inferred_kmax = int(module.shape[1])
    if environment.shape[1] != inferred_kmax or query.shape[1] != inferred_kmax:
        raise OccupancyEvidenceError("module/environment/query assignments do not share Kmax")
    if kmax is not None and inferred_kmax != int(kmax):
        raise OccupancyEvidenceError(f"payload Kmax={inferred_kmax}, requested Kmax={kmax}")
    kmax = inferred_kmax
    present_value = _lookup(payload, "module_present")
    present = np.ones(len(module_coords), dtype=bool) if present_value is None else _squeeze_batch(_array(present_value, name="module_present"), case_index=case_index, name="module_present").reshape(-1) > 0.5
    if present.size != len(module_coords):
        raise OccupancyEvidenceError("module_present does not align with module coordinates")
    module_measure = _vector(payload, "module_measure", case_index=case_index, length=len(module_coords), default=present.astype(np.float64))
    environment_measure = _vector(payload, "environment_measure", case_index=case_index, length=len(environment_coords))
    proposal_module_value = _lookup(payload, "proposal_module")
    proposal_environment_value = _lookup(payload, "proposal_environment")
    proposal_assignments_available = (
        proposal_module_value is not None and proposal_environment_value is not None
    )
    proposal_module = module if proposal_module_value is None else _ensure_source_assignment(proposal_module_value, name="proposal_module", case_index=case_index, source_count=len(module_coords))
    proposal_environment = environment if proposal_environment_value is None else _ensure_source_assignment(proposal_environment_value, name="proposal_environment", case_index=case_index, source_count=len(environment_coords))
    module_audit = _normalise_rows(module, valid=present, name="module_assignment")
    environment_audit = _normalise_rows(environment, name="environment_assignment")
    proposal_module_audit = _normalise_rows(proposal_module, valid=present, name="proposal_module")
    proposal_environment_audit = _normalise_rows(proposal_environment, name="proposal_environment")
    module_mass, module_fallback_centres = _centres(module_coords, module_measure, module)
    environment_mass, environment_fallback_centres = _centres(environment_coords, environment_measure, environment)
    proposal_module_mass, _ = _centres(module_coords, module_measure, proposal_module)
    proposal_environment_mass, _ = _centres(environment_coords, environment_measure, proposal_environment)
    final_active = (module_mass + environment_mass) > 0.0
    proposal_active = (proposal_module_mass + proposal_environment_mass) > 0.0
    proposal_ids_reused = (
        bool(np.array_equal(proposal_active, final_active))
        if proposal_assignments_available
        else None
    )
    module_refresh_delta = float(np.max(np.abs(module - proposal_module))) if module.size else 0.0
    environment_refresh_delta = float(np.max(np.abs(environment - proposal_environment))) if environment.size else 0.0
    ids_value = _lookup_with_plan(payload, "prototype_ids")
    ids = np.flatnonzero(final_active).astype(np.int64) if ids_value is None else _squeeze_batch(_array(ids_value, name="prototype_ids"), case_index=case_index, name="prototype_ids").reshape(-1).astype(np.int64)
    valid_value = _lookup_with_plan(payload, "packed_valid")
    active_value = _lookup_with_plan(payload, "active_mask")
    active_mask = final_active if active_value is None else _squeeze_batch(_array(active_value, name="active_mask"), case_index=case_index, name="active_mask").reshape(-1).astype(bool)
    if valid_value is None:
        packed_valid = active_mask.copy() if active_value is not None else (ids >= 0)
    else:
        packed_valid = _squeeze_batch(_array(valid_value, name="packed_valid"), case_index=case_index, name="packed_valid").reshape(-1).astype(bool)
    if active_mask.size != kmax:
        raise OccupancyEvidenceError("active_mask does not match Kmax")
    if ids.size != kmax:
        if ids.size < kmax:
            ids = np.pad(ids, (0, kmax - ids.size), constant_values=-1)
        else:
            raise OccupancyEvidenceError("prototype_ids width exceeds Kmax")
    packed_ids_value = _lookup_with_plan(payload, "packed_prototype_ids")
    if packed_ids_value is None:
        if packed_valid.size == kmax:
            packed_ids = ids[active_mask]
        else:
            candidate_ids = ids[ids >= 0]
            if candidate_ids.size < packed_valid.size:
                raise OccupancyEvidenceError("cannot infer packed prototype IDs from the full ID plan")
            packed_ids = candidate_ids[: packed_valid.size]
    else:
        packed_ids = _squeeze_batch(
            _array(packed_ids_value, name="packed_prototype_ids"),
            case_index=case_index,
            name="packed_prototype_ids",
        ).reshape(-1).astype(np.int64)
    if packed_valid.size != packed_ids.size:
        raise OccupancyEvidenceError("packed_valid does not align with packed_prototype_ids")
    module_centres = _centres_from_payload(payload, "module_centres", case_index=case_index, fallback=module_fallback_centres)
    environment_centres = _centres_from_payload(payload, "environment_centres", case_index=case_index, fallback=environment_fallback_centres)
    total_mass = module_mass + environment_mass
    joint_fallback = np.divide(
        module_mass[:, None] * module_fallback_centres + environment_mass[:, None] * environment_fallback_centres,
        np.maximum(total_mass[:, None], np.finfo(np.float64).tiny),
    )
    joint_centres = _centres_from_payload(payload, "joint_centres", case_index=case_index, fallback=joint_fallback)
    pi = 0.5 * (module_mass + environment_mass)
    kappa_value = _lookup_with_plan(payload, "kappa")
    kappa = float(np.asarray(_squeeze_batch(_array(kappa_value, name="kappa"), case_index=case_index, name="kappa")).reshape(-1)[0]) if kappa_value is not None else float(1.0 / np.maximum(np.sum(pi * pi), np.finfo(np.float64).tiny))
    kplan_value = _lookup_with_plan(payload, "kplan")
    kplan = int(np.asarray(_squeeze_batch(_array(kplan_value, name="kplan"), case_index=case_index, name="kplan")).reshape(-1)[0]) if kplan_value is not None else int(final_active.sum())
    module_paths, module_unique, module_dense, module_ratio, module_multiplicity, module_logical = _pair_support(query, module, present)
    environment_paths, environment_unique, environment_dense, environment_ratio, environment_multiplicity, environment_logical = _pair_support(query, environment, None)
    radii_m = np.sqrt(np.maximum(np.sum(module_measure[:, None] * module * np.sum((module_coords[:, None, :] - module_centres[None, :, :]) ** 2, axis=-1), axis=0) / np.maximum(module_mass, np.finfo(np.float64).tiny), 0.0))
    radii_e = np.sqrt(np.maximum(np.sum(environment_measure[:, None] * environment * np.sum((environment_coords[:, None, :] - environment_centres[None, :, :]) ** 2, axis=-1), axis=0) / np.maximum(environment_mass, np.finfo(np.float64).tiny), 0.0))
    shuffled_radius_m = _shuffled_radius_baseline(
        module_coords, module_measure, module, valid=present, seed=1409 + int(case_index)
    )
    shuffled_radius_e = _shuffled_radius_baseline(
        environment_coords, environment_measure, environment, valid=None, seed=2409 + int(case_index)
    )
    occupied = final_active
    if int(occupied.sum()) > 1:
        differences = joint_centres[occupied, None, :] - joint_centres[None, occupied, :]
        distances = np.sqrt(np.sum(differences * differences, axis=-1))
        centre_separation = float(distances[np.triu_indices(int(occupied.sum()), 1)].mean())
    else:
        centre_separation = 0.0
    query_positive = query > 0.0
    query_source_empty = int(np.count_nonzero(query_positive[:, ~((module_mass + environment_mass) > 0.0)]))
    phase_ledger = _phase_ledger(payload)
    p2 = phase_ledger.get("P2", {})
    source_rows = {}
    for source, prefix in (("module", "module"), ("environment", "environment")):
        phase_source = p2.get(source, {}) if isinstance(p2.get(source), Mapping) else p2
        for metric in ("actual_rows", "padded_rows", "geometry_rows", "content_rows"):
            value = phase_source.get(metric) if isinstance(phase_source, Mapping) else None
            if value is None and isinstance(p2, Mapping):
                value = p2.get(f"{prefix}_{metric}")
            source_rows[f"p2_{source}_{metric}"] = value
    row = {
        "kmax": kmax,
        "kplan": kplan,
        "active_module_count": int(present.sum()),
        "M_active": int(present.sum()),
        "M_padded": int(len(module_coords)),
        "active_module_source_ids": np.flatnonzero(present).astype(np.int64).tolist(),
        "query_source_count": query_source_count,
        "query_selection": query_selection,
        "query_domain_bounds": None if query_domain_bounds is None else query_domain_bounds.tolist(),
        "kplan_minus_active_module_count": int(kplan - int(present.sum())),
        "kplan_equals_active_module_count": bool(kplan == int(present.sum())),
        "prototype_ids": ids.tolist(),
        "packed_prototype_ids": packed_ids.tolist(),
        "packed_valid": packed_valid.tolist(),
        "active_mask": active_mask.tolist(),
        "module_mass": module_mass.tolist(),
        "environment_mass": environment_mass.tolist(),
        "kappa": kappa,
        "module_source_degree_mean": float(np.mean((module[present] > 0.0).sum(axis=-1))) if np.any(present) else 0.0,
        "environment_source_degree_mean": float(np.mean((environment > 0.0).sum(axis=-1))),
        "query_degree_mean": float(np.mean(query_positive.sum(axis=-1))),
        "module_effective_groups_mean": float(np.mean(_effective(module[present]))) if np.any(present) else 0.0,
        "environment_effective_groups_mean": float(np.mean(_effective(environment))),
        "query_effective_groups_mean": float(np.mean(_effective(query))),
        "module_assignment_numerical_rank": int(
            np.linalg.matrix_rank(module[present])
        ) if np.any(present) else 0,
        "environment_assignment_numerical_rank": int(
            np.linalg.matrix_rank(environment)
        ),
        "query_assignment_numerical_rank": int(np.linalg.matrix_rank(query)),
        "joint_source_assignment_numerical_rank": int(
            np.linalg.matrix_rank(np.concatenate((module[present], environment), axis=0))
        ),
        "module_entropy_mean": float(np.mean(_entropy(module[present], kmax))) if np.any(present) else 0.0,
        "environment_entropy_mean": float(np.mean(_entropy(environment, kmax))),
        "query_entropy_mean": float(np.mean(_entropy(query, kmax))),
        "module_empty_columns": module_audit["empty_columns"],
        "environment_empty_columns": environment_audit["empty_columns"],
        "proposal_module_empty_columns": proposal_module_audit["empty_columns"],
        "proposal_environment_empty_columns": proposal_environment_audit["empty_columns"],
        "proposal_assignments_available": proposal_assignments_available,
        "proposal_active_mask": proposal_active.tolist(),
        "proposal_vs_refined_module_max_abs": module_refresh_delta if proposal_assignments_available else None,
        "proposal_vs_refined_environment_max_abs": environment_refresh_delta if proposal_assignments_available else None,
        "p1_p2_assignment_refresh_observed": (
            bool(module_refresh_delta > 1.0e-8 or environment_refresh_delta > 1.0e-8)
            if proposal_assignments_available
            else None
        ),
        "module_radius_mean": float(np.mean(radii_m[occupied])) if np.any(occupied) else 0.0,
        "environment_radius_mean": float(np.mean(radii_e[occupied])) if np.any(occupied) else 0.0,
        "module_radius_shuffled_mass_preserving_mean": shuffled_radius_m,
        "environment_radius_shuffled_mass_preserving_mean": shuffled_radius_e,
        "joint_centre_separation_mean": centre_separation,
        "query_to_joint_centre_mean": float(np.mean(np.min(np.sqrt(np.sum((query_array[:, None, :] - joint_centres[None, :, :]) ** 2, axis=-1))[:, occupied], axis=1))) if np.any(occupied) else 0.0,
        "query_source_empty_positive_entries": query_source_empty,
        "module_logical_paths": module_paths,
        "environment_logical_paths": environment_paths,
        "module_unique_pairs": module_unique,
        "environment_unique_pairs": environment_unique,
        "module_RM_support": module_ratio,
        "environment_RE_support": environment_ratio,
        "module_multiplicity": module_multiplicity,
        "environment_multiplicity": environment_multiplicity,
        "module_dense_valid_pairs": module_dense,
        "environment_dense_valid_pairs": environment_dense,
        "proposal_normalized": bool(proposal_module_audit["active_rows_normalized"] and proposal_environment_audit["active_rows_normalized"]),
        "refined_normalized": bool(module_audit["active_rows_normalized"] and environment_audit["active_rows_normalized"]),
        "inactive_module_rows_zero": bool(module_audit["inactive_rows_zero"]),
        "proposal_ids_reused": proposal_ids_reused,
        "phase_ledger": phase_ledger,
        **source_rows,
    }
    return {
        "case": row,
        "maps": {
            "module_coords": module_coords,
            "module_present": present,
            "module_source_ids": np.arange(len(module_coords), dtype=np.int64),
            "environment_coords": environment_coords,
            "query_coords": query_array,
            "query_grid_indices": query_indices if query_indices is not None else np.arange(len(query_array), dtype=np.int64),
            "query_source_count": np.asarray(
                int(query_source_count) if query_source_count is not None else len(query_array),
                dtype=np.int64,
            ),
            "query_domain_bounds": (
                np.asarray(query_domain_bounds, dtype=np.float64)
                if query_domain_bounds is not None
                else np.asarray(
                    [
                        query_array[:, 0].min(),
                        query_array[:, 0].max(),
                        query_array[:, 1].min(),
                        query_array[:, 1].max(),
                    ],
                    dtype=np.float64,
                )
            ),
            "module_assignment": module,
            "environment_assignment": environment,
            "query_assignment": query,
            "module_centres": module_centres,
            "environment_centres": environment_centres,
            "joint_centres": joint_centres,
            "module_logical": module_logical,
            "environment_logical": environment_logical,
        },
    }


def summarize_population(rows: Sequence[Mapping[str, Any]], *, expected_cases: int = EXPECTED_CASE_COUNT) -> dict[str, Any]:
    """Aggregate exact per-case rows without assigning group identities."""

    if expected_cases > 0 and len(rows) != expected_cases:
        raise OccupancyEvidenceError(f"expected {expected_cases} cases, found {len(rows)}")
    if not rows:
        raise OccupancyEvidenceError("population is empty")
    numeric_fields = (
        "kplan", "active_module_count", "M_active", "M_padded", "kplan_minus_active_module_count", "kappa", "module_source_degree_mean", "environment_source_degree_mean", "query_degree_mean",
        "module_effective_groups_mean", "environment_effective_groups_mean", "query_effective_groups_mean",
        "module_assignment_numerical_rank", "environment_assignment_numerical_rank",
        "query_assignment_numerical_rank", "joint_source_assignment_numerical_rank",
        "module_entropy_mean", "environment_entropy_mean", "query_entropy_mean", "module_radius_mean",
        "environment_radius_mean", "module_radius_shuffled_mass_preserving_mean", "environment_radius_shuffled_mass_preserving_mean", "joint_centre_separation_mean", "query_to_joint_centre_mean", "module_RM_support",
        "environment_RE_support", "module_multiplicity", "environment_multiplicity", "module_logical_paths",
        "environment_logical_paths", "module_unique_pairs", "environment_unique_pairs", "p2_module_actual_rows",
        "p2_environment_actual_rows", "p2_module_padded_rows", "p2_environment_padded_rows",
        "p2_module_geometry_rows", "p2_environment_geometry_rows", "p2_module_content_rows", "p2_environment_content_rows",
    )
    summary: dict[str, Any] = {"case_count": len(rows), "kplan_histogram": {}}
    kplans = [int(row["kplan"]) for row in rows]
    for value in kplans:
        summary["kplan_histogram"][str(value)] = summary["kplan_histogram"].get(str(value), 0) + 1
    summary["kplan_histogram"] = dict(sorted(summary["kplan_histogram"].items(), key=lambda item: int(item[0])))
    summary["kplan_mean"] = float(np.mean(kplans))
    summary["kplan_median"] = float(np.median(kplans))
    summary["kplan_min"] = int(np.min(kplans))
    summary["kplan_max"] = int(np.max(kplans))
    frequencies = np.zeros(DEFAULT_KMAX, dtype=np.int64)
    for row in rows:
        # ``prototype_ids`` is the registered-width original-ID plan and
        # ``active_mask`` is its matching mask.  Packed IDs have their own
        # compact ``packed_valid`` mask; never zip a full ID vector with a
        # packed mask because nonconsecutive IDs would be counted against the
        # wrong prototype columns.
        full_ids = row.get("prototype_ids", [])
        full_valid = row.get("active_mask", [])
        if len(full_ids) == len(full_valid):
            pairs = zip(full_ids, full_valid, strict=False)
        else:
            pairs = zip(row.get("packed_prototype_ids", []), row.get("packed_valid", []), strict=False)
        for prototype_id, valid in pairs:
            if valid and 0 <= int(prototype_id) < len(frequencies):
                frequencies[int(prototype_id)] += 1
    summary["prototype_occupancy_frequency"] = frequencies.tolist()
    for field in numeric_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None and math.isfinite(float(row[field]))]
        if values:
            summary[field] = {"mean": float(np.mean(values)), "median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}
        else:
            summary[field] = None
    summary["continuation_gate"] = {
        "healthy_nontrivial_hypergraph": bool(summary["kplan_min"] > 1 and summary["kplan_max"] < DEFAULT_KMAX),
        "case_dependent_kplan": bool(summary["kplan_min"] != summary["kplan_max"]),
        "support_is_less_than_dense": bool(
            summary["module_RM_support"] is not None
            and summary["environment_RE_support"] is not None
            and float(summary["module_RM_support"]["mean"]) < 1.0
            and float(summary["environment_RE_support"]["mean"]) < 1.0
        ),
        "actual_rows_available": bool(any(row.get("p2_module_actual_rows") is not None or row.get("p2_environment_actual_rows") is not None for row in rows)),
        "decision": "review_accuracy_and_executor_evidence_before_continuing",
    }
    return summary


def save_case_arrays(path: Path, maps: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{str(key): np.asarray(value) for key, value in maps.items()})


def _dominant_group(values: np.ndarray) -> np.ndarray:
    positive = values > 0.0
    scores = np.where(positive, values, -np.inf)
    return np.where(positive.any(axis=-1), np.argmax(scores, axis=-1), -1)


def render_case_board(record: Mapping[str, Any], path: Path) -> None:
    """Render one compact evidence board from a canonical case record."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    maps = record["maps"]
    row = record["case"]
    module_coords = np.asarray(maps["module_coords"])
    environment_coords = np.asarray(maps["environment_coords"])
    query_coords = np.asarray(maps["query_coords"])
    module = np.asarray(maps["module_assignment"])
    environment = np.asarray(maps["environment_assignment"])
    query = np.asarray(maps["query_assignment"])
    joint_centres = np.asarray(maps["joint_centres"])
    active = np.asarray(row["active_mask"], dtype=bool)
    module_present = np.asarray(
        maps.get("module_present", np.any(module > 0.0, axis=1)), dtype=bool
    ).reshape(-1)
    if module_present.size != module_coords.shape[0]:
        raise OccupancyEvidenceError("module_present does not align with rendered module sources")
    module_coords = module_coords[module_present]
    module = module[module_present]
    kmax = int(row["kmax"])
    cmap = plt.get_cmap("tab20", max(kmax, 1))
    module_group = _dominant_group(module)
    environment_group = _dominant_group(environment)
    query_group = _dominant_group(query)
    query_degree = (query > 0.0).sum(axis=-1)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axis = axes[0, 0]
    axis.scatter(module_coords[:, 0], module_coords[:, 1], c=[cmap(int(index)) if index >= 0 else "#bdbdbd" for index in module_group], marker="s", s=32, edgecolors="black", linewidths=0.3)
    axis.scatter(joint_centres[active, 0], joint_centres[active, 1], marker="+", c=np.flatnonzero(active), cmap=cmap, s=80)
    axis.set_title("Module sources and joint centres")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis = axes[0, 1]
    axis.scatter(environment_coords[:, 0], environment_coords[:, 1], c=[cmap(int(index)) if index >= 0 else "#bdbdbd" for index in environment_group], s=8, alpha=0.72)
    axis.scatter(joint_centres[active, 0], joint_centres[active, 1], marker="+", c=np.flatnonzero(active), cmap=cmap, s=80)
    axis.set_title("Environment sources and joint centres")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis = axes[0, 2]
    axis.scatter(query_coords[:, 0], query_coords[:, 1], c=[cmap(int(index)) if index >= 0 else "#bdbdbd" for index in query_group], s=4, alpha=0.6)
    axis.set_title("Query dominant group")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    axis = axes[1, 0]
    x = np.arange(kmax)
    width = 0.38
    axis.bar(x - width / 2.0, np.asarray(row["module_mass"]), width, label="module mass", color="#0072B2")
    axis.bar(x + width / 2.0, np.asarray(row["environment_mass"]), width, label="environment mass", color="#009E73")
    axis.set_xlabel("original prototype ID")
    axis.set_ylabel("mass")
    axis.set_title("P0 mass and occupancy")
    axis.legend(frameon=False, fontsize=8)
    axis = axes[1, 1]
    image = axis.scatter(query_coords[:, 0], query_coords[:, 1], c=query_degree, s=4, cmap="magma")
    figure.colorbar(image, ax=axis, label="positive query degree")
    axis.set_title("Query support degree")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    axis = axes[1, 2]
    if len(query_coords):
        selected = int(np.argmax(query_degree))
        query_point = query_coords[selected]
        axis.scatter(query_point[0], query_point[1], marker="*", s=130, color="#d55e00", label=f"q={selected}")
        for group in np.flatnonzero(active & (query[selected] > 0.0)):
            axis.plot([query_point[0], joint_centres[group, 0]], [query_point[1], joint_centres[group, 1]], color=cmap(int(group)), linewidth=1.0, alpha=0.75)
            axis.scatter(joint_centres[group, 0], joint_centres[group, 1], marker="+", color=cmap(int(group)), s=85)
            for coords, assignments, color in ((module_coords, module, "#0072B2"), (environment_coords, environment, "#009E73")):
                selected_sources = np.flatnonzero(assignments[:, group] > 0.0)
                if len(selected_sources):
                    source = selected_sources[0]
                    axis.plot([joint_centres[group, 0], coords[source, 0]], [joint_centres[group, 1], coords[source, 1]], color=color, linewidth=0.5, alpha=0.35)
        axis.legend(frameon=False, fontsize=8)
    axis.set_title("Query → group → source support")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    module_rows = row.get("p2_module_actual_rows")
    environment_rows = row.get("p2_environment_actual_rows")
    module_rows_text = "n/a" if module_rows is None else str(int(float(module_rows)))
    environment_rows_text = (
        "n/a" if environment_rows is None else str(int(float(environment_rows)))
    )
    figure.suptitle(
        f"Occupancy adaptive Run 1409 · Kmax={row['kmax']} · Kplan={row['kplan']} · kappa={float(row['kappa']):.3f} · "
        f"RM/RE={float(row['module_RM_support']):.3f}/{float(row['environment_RE_support']):.3f} · "
        f"P2 actual rows M/E={module_rows_text}/{environment_rows_text}\n"
        "Learned routing organization; group IDs are permutation ambiguous and are not physical causality claims",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def render_kplan_histogram(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    values = [int(row["kplan"]) for row in rows]
    unique, counts = np.unique(values, return_counts=True)
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = axis.bar(unique, counts, color="#0072B2")
    axis.set_xlabel("Kplan")
    axis.set_ylabel("number of cases")
    axis.set_title("Run 1409 occupancy plan across the explicit 90-case population")
    for bar, count in zip(bars, counts, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2.0, float(count), str(int(count)), ha="center", va="bottom", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def write_population_evidence(
    cases: Sequence[tuple[str, Any]],
    output_dir: str | Path,
    *,
    expected_cases: int = EXPECTED_CASE_COUNT,
    query_count: int | None = None,
    kmax: int = DEFAULT_KMAX,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Write managed evaluation files for an explicit set of map payloads."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    array_dir = output / "arrays"
    figure_dir = output / "figures"
    rows: list[dict[str, Any]] = []
    records: list[tuple[str, dict[str, Any]]] = []
    array_paths: dict[str, str] = {}
    for case_id, payload in cases:
        record = canonicalize_case(payload, query_count=query_count, kmax=kmax)
        rows.append({"case_id": str(case_id), **record["case"]})
        records.append((str(case_id), record))
        array_path = array_dir / f"{case_id}.npz"
        save_case_arrays(array_path, record["maps"])
        array_paths[str(case_id)] = str(array_path)
    summary = summarize_population(rows, expected_cases=expected_cases)
    render_kplan_histogram(rows, figure_dir / "kplan_histogram.png")
    for case_id, record in records:
        render_case_board(record, figure_dir / f"occupancy_board__{case_id}.png")
    payload = {
        "schema_version": 1,
        "task": "run1409_occupancy_adaptive_population_evidence",
        "status": "complete",
        "candidate": {
            "architecture": "occupancy_adaptive_group_control_honf",
            "checkpoint": None if checkpoint is None else str(Path(checkpoint).expanduser().resolve()),
            "checkpoint_selection": "explicit_path_only",
            "Kmax": int(kmax),
            "no_gate_network": True,
            "no_capacity_loss": True,
        },
        "population": summary,
        "cases": rows,
        "arrays": array_paths,
        "figures": {
            "kplan_histogram": str(figure_dir / "kplan_histogram.png"),
            "case_boards": [str(figure_dir / f"occupancy_board__{case_id}.png") for case_id, _record in records],
        },
        "missing_evidence": [
            "No map is a physical causality claim; learned group labels are permutation ambiguous.",
            "This population artifact does not embed matched Run-1404/1406/1804 accuracy and cost tables; those require a separate explicit checkpoint evaluation.",
        ]
        + (
            []
            if bool(summary["continuation_gate"]["actual_rows_available"])
            else [
                "Actual execution rows remain unavailable because the backend phase ledger is absent."
            ]
        ),
    }
    _write_json(output / "evidence.json", payload)
    _write_json(output / "population_summary.json", summary)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output / "population_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})
    return payload


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Describe the explicit evidence protocol without touching runtime state."""

    output = Path(args.output_dir).expanduser().resolve()
    checkpoint = None if args.checkpoint is None else Path(args.checkpoint).expanduser().resolve()
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    return {
        "schema_version": 1,
        "task": "run1409_occupancy_adaptive_evidence",
        "status": "plan_only",
        "candidate": {
            "architecture": "occupancy_adaptive_group_control_honf",
            "Kmax": int(args.kmax),
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "checkpoint_selection": "explicit_path_only",
        },
        "protocol": {
            "expected_cases": int(args.expected_cases),
            "query_count": None if args.query_count is None else int(args.query_count),
            "two_predicted_port_batches": True,
            "one_optimizer_update_if_cpu_read_only_path_is_available": True,
            "population": [
                "Kplan histogram and original prototype IDs",
                "source/query positive degree, effective groups, entropy",
                "within-group spatial RMS radii and joint-centre separation",
                "query maps and query-to-centre distances",
                "logical support RM/RE, multiplicity, unique pairs, actual rows",
                "matched Run-1404/1406/1804 metrics from explicit checkpoint paths",
            ],
        },
        "planned_outputs": {
            "output_dir": str(output),
            "evidence_json": str(output / "evidence.json"),
            "population_summary": str(output / "population_summary.json"),
            "case_csv": str(output / "population_cases.csv"),
            "array_dir": str(output / "arrays"),
        },
        "continuation_gates": {
            "continue_only_if": [
                "P0 proposal and refined rows normalize; exact empty columns are audited",
                "Kplan is nonzero and not universal one-group/all-group collapse",
                "P0 IDs are reused while P1/P2 source values and assignments refresh",
                "query assignment masks source-empty groups",
                "full/packed parity and source permutation checks pass",
                "actual rows are reported separately from logical support",
                "accuracy is compared at matched explicit checkpoints",
            ],
            "stop_or_hold_if": [
                "phase ledger is missing for actual execution claims",
                "all cases share one Kplan or all groups remain active without an interpretation",
                "support decreases but measured rows/latency do not improve",
                "the candidate cannot be compared to 1404/1406/1804 under matched data",
            ],
        },
    }


def _load_payload(path: Path) -> Any:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _parse_case_payload(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--case-payload must use CASE=PATH")
    case_id, raw_path = spec.split("=", 1)
    if not case_id.strip() or not raw_path.strip():
        raise ValueError("--case-payload must include a case ID and path")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return case_id.strip(), path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit candidate checkpoint; never inferred from a run number")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-payload", action="append", default=[], metavar="CASE=PATH", help="Offline map payload; repeat for each case")
    parser.add_argument("--expected-cases", type=int, default=EXPECTED_CASE_COUNT)
    parser.add_argument("--query-count", type=int, default=None)
    parser.add_argument("--kmax", type=int, default=DEFAULT_KMAX)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.kmax) <= 0 or int(args.expected_cases) < 0:
        raise ValueError("kmax must be positive and expected-cases must be nonnegative")
    if args.plan_only:
        payload = build_plan(args)
        _write_json(Path(args.output_dir) / "plan.json", payload)
        print(json.dumps({"status": "plan_only", "output": str(Path(args.output_dir).resolve() / "plan.json")}, indent=2))
        return 0
    if not args.case_payload:
        raise ValueError("physical invocation requires explicit --case-payload inputs; use --plan-only otherwise")
    cases = [(case_id, _load_payload(path)) for case_id, path in (_parse_case_payload(value) for value in args.case_payload)]
    payload = write_population_evidence(
        cases,
        args.output_dir,
        expected_cases=int(args.expected_cases),
        query_count=args.query_count,
        kmax=int(args.kmax),
        checkpoint=args.checkpoint,
    )
    print(json.dumps({"status": payload["status"], "output_dir": str(Path(args.output_dir).resolve()), "cases": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OccupancyEvidenceError",
    "build_plan",
    "canonicalize_case",
    "main",
    "summarize_population",
    "write_population_evidence",
]
