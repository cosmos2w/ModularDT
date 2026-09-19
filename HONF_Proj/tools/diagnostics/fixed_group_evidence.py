"""Shared CPU-side evidence helpers for the fixed-group Run-1405 tools.

The fixed-group backend is expected to expose its compact debug tensors through
``interaction_aux`` (the ordinary model output) or an equivalent mapping on the
prepared state.  This module deliberately contains no model imports and no
CUDA calls, so map extraction, semantic counting, and board tests can run on a
CPU-only machine.

Canonical debug keys
--------------------

The preferred names are:

``fixed_group_module_incidence``
    ``[B,M,K]`` non-negative module membership ``A_m``.
``fixed_group_environment_incidence``
    ``[B,E,K]`` non-negative environment membership ``A_e``.
``fixed_group_module_centres`` / ``fixed_group_environment_centres``
    ``[B,K,d]`` group centres ``r_m`` and ``r_e``.
``fixed_group_query_routing``
    ``[B,Q,K]`` query-to-group routing ``alpha``.

The extractor accepts a small set of spelling aliases to ease integration, but
it never reconstructs these arrays from dense pair counts.  Thus ``P_M`` and
``P_E`` are computed from the actual positive semantic supports rather than a
compiler approximation.  ``s_Q`` is the mean number of positive groups per
query; ``s_M`` and ``s_E`` are the mean number of positive groups per active
module and environment source, respectively.  Companion source fan-out
statistics are retained separately.  All routes are learned interaction
routes, not physical-causality claims.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

GROUP_COUNT = 6


class FixedGroupEvidenceError(ValueError):
    """Raised when a fixed-group debug payload violates the evidence contract."""


@dataclass(frozen=True)
class FixedGroupArrays:
    """One case of fixed-group geometry and learned routing arrays.

    Batch dimensions are removed because the board and the per-case evidence
    artifact each describe one physical case.  The source dimensions retain
    padded rows only when ``module_present`` marks them inactive; inactive
    module rows are excluded from semantic counts.
    """

    module_coords: np.ndarray
    module_present: np.ndarray
    env_coords: np.ndarray
    env_weights: np.ndarray
    module_incidence: np.ndarray
    environment_incidence: np.ndarray
    module_group_centres: np.ndarray
    environment_group_centres: np.ndarray
    query_xy: np.ndarray
    query_routing: np.ndarray

    @property
    def group_count(self) -> int:
        return int(self.query_routing.shape[-1])

    @property
    def active_module_mask(self) -> np.ndarray:
        return np.asarray(self.module_present, dtype=float).reshape(-1) > 0.5


@dataclass(frozen=True)
class FixedGroupMetrics:
    """Exact semantic counts and descriptive support statistics."""

    values: dict[str, Any]
    selected_query_index: int
    selected_module_triples: np.ndarray
    selected_environment_triples: np.ndarray
    query_group_source_counts: np.ndarray


_ALIASES: dict[str, tuple[str, ...]] = {
    "module_incidence": (
        "fixed_group_module_incidence",
        "fixed_group_module_membership",
        "module_group_incidence",
        "module_group_membership",
        "group_module_incidence",
        "A_m",
        "A_M",
        "module_incidence",
    ),
    "environment_incidence": (
        "fixed_group_environment_incidence",
        "fixed_group_environment_membership",
        "environment_group_incidence",
        "environment_group_membership",
        "group_environment_incidence",
        "A_e",
        "A_E",
        "environment_incidence",
    ),
    "module_group_centres": (
        "fixed_group_module_centres",
        "fixed_group_module_centers",
        "module_group_centres",
        "module_group_centers",
        "group_module_centres",
        "group_module_centers",
        "r_m",
        "r_M",
    ),
    "environment_group_centres": (
        "fixed_group_environment_centres",
        "fixed_group_environment_centers",
        "environment_group_centres",
        "environment_group_centers",
        "group_environment_centres",
        "group_environment_centers",
        "r_e",
        "r_E",
    ),
    "query_routing": (
        "fixed_group_query_routing",
        "fixed_group_query_alpha",
        "query_group_routing",
        "query_group_alpha",
        "query_group_probability",
        "query_group_probabilities",
        "group_query_alpha",
        "alpha_qk",
        "alpha",
        "query_routing",
    ),
}


def _as_array(value: Any, *, name: str) -> np.ndarray:
    if value is None:
        raise FixedGroupEvidenceError(f"missing fixed-group debug array {name!r}")
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise FixedGroupEvidenceError(f"cannot convert debug array {name!r} to NumPy") from exc
    if array.dtype == object:
        raise FixedGroupEvidenceError(f"debug array {name!r} has object dtype")
    return array


def _find(mapping: Mapping[str, Any], name: str, *, required: bool = True) -> Any:
    for key in _ALIASES.get(name, (name,)):
        if key in mapping:
            return mapping[key]
    if required:
        choices = ", ".join(_ALIASES.get(name, (name,)))
        raise FixedGroupEvidenceError(f"missing {name}; expected one of: {choices}")
    return None


def _without_batch(value: Any, *, name: str) -> np.ndarray:
    array = _as_array(value, name=name)
    if array.ndim >= 1 and array.shape[0] == 1:
        return array[0]
    return array


def _orient_membership(value: Any, *, source_count: int, group_count: int, name: str) -> np.ndarray:
    array = _without_batch(value, name=name)
    if array.ndim != 2:
        raise FixedGroupEvidenceError(
            f"{name} must be [source,K] or [K,source] after batch squeeze; got {array.shape}"
        )
    if array.shape == (source_count, group_count):
        result = array
    elif array.shape == (group_count, source_count):
        result = array.T
    else:
        raise FixedGroupEvidenceError(
            f"{name} does not match source_count={source_count}, group_count={group_count}: {array.shape}"
        )
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < -1.0e-10):
        raise FixedGroupEvidenceError(f"{name} must be finite and non-negative")
    # Preserve exact zeros while removing only numerical negative noise.
    return np.where(result < 0.0, 0.0, result)


def _orient_centres(value: Any, *, group_count: int, dimension: int, name: str) -> np.ndarray:
    array = _without_batch(value, name=name)
    if array.ndim != 2 or array.shape != (group_count, dimension):
        raise FixedGroupEvidenceError(
            f"{name} must have shape [{group_count},{dimension}] after batch squeeze; got {array.shape}"
        )
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FixedGroupEvidenceError(f"{name} must be finite")
    return result


def _orient_query(value: Any, *, query_count: int, group_count: int, name: str) -> np.ndarray:
    array = _without_batch(value, name=name)
    if array.ndim != 2:
        raise FixedGroupEvidenceError(
            f"{name} must be [Q,K] or [K,Q] after batch squeeze; got {array.shape}"
        )
    if array.shape == (query_count, group_count):
        result = array
    elif array.shape == (group_count, query_count):
        result = array.T
    else:
        raise FixedGroupEvidenceError(
            f"{name} does not match query_count={query_count}, group_count={group_count}: {array.shape}"
        )
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < -1.0e-10):
        raise FixedGroupEvidenceError(f"{name} must be finite and non-negative")
    return np.where(result < 0.0, 0.0, result)


def _geometry(mapping: Mapping[str, Any], key: str, *, source_name: str, dimension: int) -> np.ndarray:
    value = mapping.get(key)
    if value is None:
        aliases = {
            "module_coords": ("module_coords", "module_centres", "module_centers"),
            "env_coords": ("env_coords", "environment_coords", "environment_centres", "environment_centers"),
            "query_xy": ("query_xy", "query_coords", "receiver_coords", "receivers"),
        }
        for alias in aliases.get(key, (key,)):
            if alias in mapping:
                value = mapping[alias]
                break
    if value is None:
        raise FixedGroupEvidenceError(f"missing geometry array {key!r}")
    array = _without_batch(value, name=key)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise FixedGroupEvidenceError(f"{key} must have shape [N,{dimension}], got {array.shape}")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FixedGroupEvidenceError(f"{key} must be finite")
    if not len(result) and source_name:
        raise FixedGroupEvidenceError(f"{key} cannot be empty for {source_name}")
    return result


def canonicalize_fixed_group_arrays(
    payload: Mapping[str, Any],
    *,
    expected_group_count: int = GROUP_COUNT,
) -> FixedGroupArrays:
    """Validate and canonicalize one fixed-group debug payload.

    ``payload`` may contain tensors from a model output, NumPy arrays loaded
    from an NPZ map, or a mixture of both.  Only one batch row is accepted;
    this makes accidental cross-case aggregation impossible.
    """

    module_coords = _geometry(payload, "module_coords", source_name="module", dimension=2)
    env_coords = _geometry(payload, "env_coords", source_name="environment", dimension=2)
    query_xy = _geometry(payload, "query_xy", source_name="query", dimension=2)
    module_present_value = payload.get("module_present")
    if module_present_value is None:
        module_present = np.ones((len(module_coords),), dtype=np.float64)
    else:
        module_present = _without_batch(module_present_value, name="module_present").reshape(-1).astype(np.float64)
        if len(module_present) != len(module_coords):
            raise FixedGroupEvidenceError("module_present must align with module_coords")
    env_weights_value = payload.get("env_weights", payload.get("environment_weights"))
    if env_weights_value is None:
        env_weights = np.ones((len(env_coords),), dtype=np.float64)
    else:
        env_weights = _without_batch(env_weights_value, name="env_weights").reshape(-1).astype(np.float64)
        if len(env_weights) != len(env_coords):
            raise FixedGroupEvidenceError("env_weights must align with env_coords")
        if not np.isfinite(env_weights).all() or np.any(env_weights <= 0.0):
            raise FixedGroupEvidenceError("env_weights must be finite and strictly positive")
    query_value = _find(payload, "query_routing")
    query_array = _without_batch(query_value, name="query_routing")
    if query_array.ndim != 2:
        raise FixedGroupEvidenceError(f"query_routing must be rank-2 after batch squeeze, got {query_array.shape}")
    inferred_group_count = int(query_array.shape[-1])
    if query_array.shape[0] != len(query_xy) and query_array.shape[1] == len(query_xy):
        inferred_group_count = int(query_array.shape[0])
    if expected_group_count and inferred_group_count != int(expected_group_count):
        raise FixedGroupEvidenceError(
            f"fixed-group query routing has K={inferred_group_count}; expected K={expected_group_count}"
        )
    group_count = int(expected_group_count or inferred_group_count)
    module_incidence = _orient_membership(
        _find(payload, "module_incidence"),
        source_count=len(module_coords),
        group_count=group_count,
        name="module_incidence",
    )
    environment_incidence = _orient_membership(
        _find(payload, "environment_incidence"),
        source_count=len(env_coords),
        group_count=group_count,
        name="environment_incidence",
    )
    query_routing = _orient_query(
        query_value,
        query_count=len(query_xy),
        group_count=group_count,
        name="query_routing",
    )
    module_centres_value = _find(payload, "module_group_centres")
    environment_centres_value = _find(payload, "environment_group_centres", required=False)
    if environment_centres_value is None:
        # A single centre tensor is accepted as a compatibility fallback, but
        # the canonical Run-1405 API should expose both r_m and r_e.
        environment_centres_value = module_centres_value
    module_group_centres = _orient_centres(
        module_centres_value,
        group_count=group_count,
        dimension=module_coords.shape[1],
        name="module_group_centres",
    )
    environment_group_centres = _orient_centres(
        environment_centres_value,
        group_count=group_count,
        dimension=env_coords.shape[1],
        name="environment_group_centres",
    )
    return FixedGroupArrays(
        module_coords=module_coords,
        module_present=module_present,
        env_coords=env_coords,
        env_weights=env_weights,
        module_incidence=module_incidence,
        environment_incidence=environment_incidence,
        module_group_centres=module_group_centres,
        environment_group_centres=environment_group_centres,
        query_xy=query_xy,
        query_routing=query_routing,
    )


def _triples_for_query(
    arrays: FixedGroupArrays,
    query_index: int,
    *,
    source_kind: str,
) -> np.ndarray:
    if source_kind == "module":
        source_support = arrays.module_incidence > 0.0
        source_support &= arrays.active_module_mask[:, None]
    elif source_kind == "environment":
        source_support = arrays.environment_incidence > 0.0
    else:
        raise ValueError(f"unknown source_kind={source_kind!r}")
    groups = np.flatnonzero(arrays.query_routing[query_index] > 0.0)
    rows: list[tuple[int, int, int]] = []
    for group in groups:
        for source in np.flatnonzero(source_support[:, group]):
            rows.append((int(query_index), int(group), int(source)))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def semantic_metrics(
    arrays: FixedGroupArrays,
    *,
    selected_query_index: int | None = None,
    prepared_decode_median_ms: float | None = None,
) -> FixedGroupMetrics:
    """Compute exact grouped semantic counts from positive support masks."""

    query_support = arrays.query_routing > 0.0
    module_support = (arrays.module_incidence > 0.0) & arrays.active_module_mask[:, None]
    environment_support = arrays.environment_incidence > 0.0
    module_counts = module_support.sum(axis=0).astype(np.int64)
    environment_counts = environment_support.sum(axis=0).astype(np.int64)
    q_groups = query_support.sum(axis=1).astype(np.int64)
    module_group_counts_per_source = module_support.sum(axis=1).astype(np.int64)
    environment_group_counts_per_source = environment_support.sum(axis=1).astype(np.int64)
    query_group_rows = np.argwhere(query_support)
    if len(query_group_rows):
        selected_module_counts = module_counts[query_group_rows[:, 1]]
        selected_environment_counts = environment_counts[query_group_rows[:, 1]]
    else:
        selected_module_counts = np.zeros((0,), dtype=np.int64)
        selected_environment_counts = np.zeros((0,), dtype=np.int64)
    p_m = int(selected_module_counts.sum())
    p_e = int(selected_environment_counts.sum())
    q_count = int(arrays.query_routing.shape[0])
    m_active = int(arrays.active_module_mask.sum())
    e_active = int(arrays.env_coords.shape[0])
    denominator_m = q_count * m_active
    denominator_e = q_count * e_active
    if selected_query_index is None:
        active_centres = arrays.module_coords[arrays.active_module_mask]
        if len(active_centres):
            distance = np.linalg.norm(
                arrays.query_xy[:, None, :] - active_centres[None, :, :], axis=-1
            )
            selected_query_index = int(np.argmax(np.min(distance, axis=1)))
        else:
            selected_query_index = max(q_count - 1, 0)
    selected_query_index = int(selected_query_index)
    if not 0 <= selected_query_index < q_count:
        raise FixedGroupEvidenceError(
            f"selected_query_index={selected_query_index} outside Q={q_count}"
        )
    query_group_source_counts = np.stack(
        (
            np.broadcast_to(module_counts[None, :], (q_count, arrays.group_count)),
            np.broadcast_to(environment_counts[None, :], (q_count, arrays.group_count)),
        ),
        axis=-1,
    )
    query_group_source_counts = query_group_source_counts * query_support[..., None]
    values: dict[str, Any] = {
        "K": int(arrays.group_count),
        "Q": q_count,
        "M_active": m_active,
        "M_padded": len(arrays.module_coords),
        "E_active": e_active,
        "P_M": p_m,
        "P_E": p_e,
        "R_M": None if denominator_m == 0 else float(p_m / denominator_m),
        "R_E": None if denominator_e == 0 else float(p_e / denominator_e),
        "sQ": float(np.mean(q_groups)) if q_count else 0.0,
        "sQ_min": int(q_groups.min()) if q_count else 0,
        "sQ_max": int(q_groups.max()) if q_count else 0,
        "sM": (
            float(np.mean(module_group_counts_per_source[arrays.active_module_mask]))
            if m_active
            else 0.0
        ),
        "sE": float(np.mean(environment_group_counts_per_source)) if e_active else 0.0,
        "mean_module_sources_per_query_group": (
            float(np.mean(selected_module_counts)) if len(selected_module_counts) else 0.0
        ),
        "mean_environment_sources_per_query_group": (
            float(np.mean(selected_environment_counts)) if len(selected_environment_counts) else 0.0
        ),
        "group_module_source_counts": module_counts.astype(int).tolist(),
        "group_environment_source_counts": environment_counts.astype(int).tolist(),
        "selected_query_index": selected_query_index,
        "selected_query_xy": arrays.query_xy[selected_query_index].astype(float).tolist(),
        "route_semantics": "positive learned interaction support; not physical causality",
    }
    if prepared_decode_median_ms is not None:
        value = float(prepared_decode_median_ms)
        values["prepared_decode_median_ms"] = value if math.isfinite(value) else None
    return FixedGroupMetrics(
        values=values,
        selected_query_index=selected_query_index,
        selected_module_triples=_triples_for_query(arrays, selected_query_index, source_kind="module"),
        selected_environment_triples=_triples_for_query(arrays, selected_query_index, source_kind="environment"),
        query_group_source_counts=query_group_source_counts.astype(np.int64),
    )


def _mapping_containers(outputs: Any) -> list[Mapping[str, Any]]:
    """Collect model-output/prepared-state mappings without importing models."""

    containers: list[Mapping[str, Any]] = []
    if isinstance(outputs, Mapping):
        for key in ("fixed_group_debug", "interaction_aux", "routing_aux", "debug_aux"):
            value = outputs.get(key)
            if isinstance(value, Mapping):
                containers.append(value)
        containers.append(outputs)
        prepared = outputs.get("prepared_state")
    else:
        prepared = outputs
    prepared_inner = getattr(prepared, "prepared", prepared)
    for candidate in (
        getattr(prepared_inner, "interaction_aux", None),
        getattr(prepared_inner, "fixed_group_debug", None),
        getattr(getattr(prepared_inner, "backend_state", None), "diagnostics", None),
    ):
        if isinstance(candidate, Mapping):
            containers.append(candidate)
    return containers


def collect_debug_payload(
    outputs: Mapping[str, Any],
    *,
    query_xy: Any,
) -> dict[str, Any]:
    """Build a canonical extraction payload from a model debug forward.

    The prepared state's encoded geometry is preferred over duplicating input
    geometry in the model output.  A descriptive error names every missing
    canonical tensor when the backend has not implemented the debug API yet.
    """

    merged: dict[str, Any] = {}
    for container in _mapping_containers(outputs):
        merged.update(container)
    prepared = outputs.get("prepared_state")
    prepared_inner = getattr(prepared, "prepared", prepared)
    encoded = getattr(prepared_inner, "encoded", None)
    geometry_sources = {
        "module_coords": getattr(encoded, "module_centers", None),
        "module_present": getattr(encoded, "module_present", None),
        "env_coords": getattr(encoded, "env_coords", None),
        "env_weights": getattr(encoded, "env_weights", None),
    }
    for key, value in geometry_sources.items():
        if value is not None:
            merged.setdefault(key, value)
    merged["query_xy"] = query_xy
    return merged


def save_evidence_npz(
    path: str | Path,
    arrays: FixedGroupArrays,
    metrics: FixedGroupMetrics,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write one compact board input with metrics and selected actual triples."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "module_coords": arrays.module_coords,
        "module_present": arrays.module_present,
        "env_coords": arrays.env_coords,
        "env_weights": arrays.env_weights,
        "module_incidence": arrays.module_incidence,
        "environment_incidence": arrays.environment_incidence,
        "module_group_centres": arrays.module_group_centres,
        "environment_group_centres": arrays.environment_group_centres,
        "query_xy": arrays.query_xy,
        "query_routing": arrays.query_routing,
        "selected_module_triples": metrics.selected_module_triples,
        "selected_environment_triples": metrics.selected_environment_triples,
        "query_group_source_counts": metrics.query_group_source_counts,
        "metrics_json": np.asarray(json.dumps(metrics.values, sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(dict(metadata or {}), sort_keys=True)),
    }
    np.savez_compressed(destination, **payload)
    return destination


def load_evidence_npz(path: str | Path) -> tuple[FixedGroupArrays, FixedGroupMetrics, dict[str, Any]]:
    """Load and validate one saved evidence NPZ."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    arrays = canonicalize_fixed_group_arrays(payload)
    raw_metrics = payload.get("metrics_json")
    values: dict[str, Any]
    if raw_metrics is None:
        values = semantic_metrics(arrays).values
    else:
        text = str(np.asarray(raw_metrics).reshape(-1)[0])
        values = json.loads(text)
    computed = semantic_metrics(
        arrays,
        selected_query_index=int(values.get("selected_query_index", 0)),
        prepared_decode_median_ms=values.get("prepared_decode_median_ms"),
    )
    raw_metadata = payload.get("metadata_json")
    metadata = {} if raw_metadata is None else json.loads(str(np.asarray(raw_metadata).reshape(-1)[0]))
    return arrays, FixedGroupMetrics(values, computed.selected_query_index, computed.selected_module_triples, computed.selected_environment_triples, computed.query_group_source_counts), metadata


def jsonable(value: Any) -> Any:
    """Convert NumPy values for manifests/tests without changing semantics."""

    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "GROUP_COUNT",
    "FixedGroupArrays",
    "FixedGroupEvidenceError",
    "FixedGroupMetrics",
    "canonicalize_fixed_group_arrays",
    "collect_debug_payload",
    "jsonable",
    "load_evidence_npz",
    "save_evidence_npz",
    "semantic_metrics",
]
