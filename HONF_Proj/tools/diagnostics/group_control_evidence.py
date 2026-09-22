"""CPU-side evidence helpers for the Run-1406 group-control operator.

Run 1406 has two different notions of work.  A query/source pair may have
several logical ``q -> group -> source`` paths, while the low-dimensional
operator should execute one fine ``q -> source`` function after collapsing
those paths into ``rho`` and ``n``.  This module keeps those quantities
separate and never treats a logical path count as a fine-call count.

The backend's opt-in debug payload is expected under ``group_control_*`` keys
in ``interaction_aux`` (or ``group_control_debug``):

``group_control_module_incidence`` / ``group_control_environment_incidence``
    ``[B,M,K]`` and ``[B,E,K]`` non-negative memberships.
``group_control_query_routing``
    ``[B,Q,K]`` non-negative query routing.
``group_control_module_overlap`` / ``group_control_environment_overlap``
    optional ``[B,Q,M]`` and ``[B,Q,E]`` scalar overlaps.  They are derived
    from the memberships when omitted.
``group_control_*_control_moment``
    optional ``[B,Q,N,D]`` mass-weighted moments for selected-pair display.
``group_control_phase_ledger``
    optional detached P0/P1/P2 records containing actual rows/calls and
    checkpoint recomputations.  Missing phase ledgers are reported as
    unavailable; this module does not infer P0/P1 calls from a P2 map.

The Thermal wrapper may instead expose explicit raw phase keys in the same
auxiliary mapping: ``initial_port_`` (P0), ``provisional_`` (P1), unprefixed
Run-1406 keys (P2), and ``port_global_`` (P2 consistency).  The adapter below
maps those prefixes to the ledger schema without synthesizing missing phases.

The helper has no model or CUDA dependency and is suitable for board tests on
a CPU-only host.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

GROUP_COUNT = 6
CONTROL_DIM = 16
PHASES = ("P0", "P1", "P2", "P2_consistency")
SOURCE_TYPES = ("module", "environment")


class GroupControlEvidenceError(ValueError):
    """Raised when an opt-in group-control debug payload is malformed."""


@dataclass(frozen=True)
class GroupControlArrays:
    """One case of group-control maps after removing a singleton batch axis."""

    module_coords: np.ndarray
    module_present: np.ndarray
    env_coords: np.ndarray
    env_weights: np.ndarray
    module_incidence: np.ndarray
    environment_incidence: np.ndarray
    query_xy: np.ndarray
    query_routing: np.ndarray
    module_overlap: np.ndarray
    environment_overlap: np.ndarray
    module_group_centres: np.ndarray
    environment_group_centres: np.ndarray
    group_control: np.ndarray | None = None
    module_control_moment: np.ndarray | None = None
    environment_control_moment: np.ndarray | None = None

    @property
    def group_count(self) -> int:
        return int(self.query_routing.shape[-1])

    @property
    def active_module_mask(self) -> np.ndarray:
        return np.asarray(self.module_present, dtype=np.float64).reshape(-1) > 0.5


@dataclass(frozen=True)
class GroupControlMetrics:
    """Selected-query triples and scalar summaries used by the board."""

    values: dict[str, Any]
    selected_query_index: int
    selected_module_logical_triples: np.ndarray
    selected_environment_logical_triples: np.ndarray
    selected_module_unique_pairs: np.ndarray
    selected_environment_unique_pairs: np.ndarray
    selected_module_moments: np.ndarray | None
    selected_environment_moments: np.ndarray | None


_ALIASES: dict[str, tuple[str, ...]] = {
    "module_incidence": (
        "group_control_module_incidence",
        "group_control_module_membership",
        "module_group_incidence",
        "module_incidence",
        "A_m",
    ),
    "environment_incidence": (
        "group_control_environment_incidence",
        "group_control_environment_membership",
        "environment_group_incidence",
        "environment_incidence",
        "A_e",
    ),
    "query_routing": (
        "group_control_query_routing",
        "group_control_query_alpha",
        "query_group_routing",
        "query_routing",
        "alpha_qk",
    ),
    "module_overlap": (
        "group_control_module_overlap",
        "group_control_module_rho",
        "module_query_overlap",
        "module_overlap",
        "rho_module",
        "rho_M",
    ),
    "environment_overlap": (
        "group_control_environment_overlap",
        "group_control_environment_rho",
        "environment_query_overlap",
        "environment_overlap",
        "rho_environment",
        "rho_E",
    ),
    "module_control_moment": (
        "group_control_module_control_moment",
        "group_control_module_moment",
        "module_control_moment",
        "n_M",
    ),
    "environment_control_moment": (
        "group_control_environment_control_moment",
        "group_control_environment_moment",
        "environment_control_moment",
        "n_E",
    ),
    "module_group_centres": (
        "group_control_module_centres",
        "group_control_module_centers",
        "module_group_centres",
        "module_group_centers",
    ),
    "environment_group_centres": (
        "group_control_environment_centres",
        "group_control_environment_centers",
        "environment_group_centres",
        "environment_group_centers",
    ),
    "group_control": (
        "group_control_h",
        "group_control_group_control",
        "group_control",
    ),
}


def _as_array(value: Any, *, name: str) -> np.ndarray:
    if value is None:
        raise GroupControlEvidenceError(f"missing group-control array {name!r}")
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise GroupControlEvidenceError(f"cannot convert {name!r} to NumPy") from exc
    if array.dtype == object:
        raise GroupControlEvidenceError(f"group-control array {name!r} has object dtype")
    return array


def _without_batch(value: Any, *, name: str) -> np.ndarray:
    array = _as_array(value, name=name)
    if array.ndim >= 1 and array.shape[0] == 1:
        return array[0]
    return array


def _find(mapping: Mapping[str, Any], name: str, *, required: bool = True) -> Any:
    for key in _ALIASES.get(name, (name,)):
        if key in mapping:
            return mapping[key]
    if required:
        expected = ", ".join(_ALIASES.get(name, (name,)))
        raise GroupControlEvidenceError(f"missing {name}; expected one of: {expected}")
    return None


def _geometry(mapping: Mapping[str, Any], key: str, dimension: int = 2) -> np.ndarray:
    aliases = {
        "module_coords": ("module_coords", "module_centres", "module_centers"),
        "env_coords": ("env_coords", "environment_coords", "environment_centres", "environment_centers"),
        "query_xy": ("query_xy", "query_coords", "receiver_coords", "receivers"),
    }
    value = next((mapping[name] for name in aliases[key] if name in mapping), None)
    if value is None:
        raise GroupControlEvidenceError(f"missing geometry array {key!r}")
    array = _without_batch(value, name=key)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise GroupControlEvidenceError(f"{key} must have shape [N,{dimension}], got {array.shape}")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all() or not len(result):
        raise GroupControlEvidenceError(f"{key} must be finite and non-empty")
    return result


def _orient_membership(value: Any, *, source_count: int, group_count: int, name: str) -> np.ndarray:
    array = _without_batch(value, name=name)
    if array.ndim != 2:
        raise GroupControlEvidenceError(f"{name} must be rank-2 after batch squeeze; got {array.shape}")
    if array.shape == (source_count, group_count):
        result = array
    elif array.shape == (group_count, source_count):
        result = array.T
    else:
        raise GroupControlEvidenceError(
            f"{name} shape {array.shape} does not match source_count={source_count}, K={group_count}"
        )
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < -1.0e-10):
        raise GroupControlEvidenceError(f"{name} must be finite and non-negative")
    return np.where(result < 0.0, 0.0, result)


def _orient_query(value: Any, *, query_count: int, group_count: int) -> np.ndarray:
    array = _without_batch(value, name="query_routing")
    if array.ndim != 2:
        raise GroupControlEvidenceError(f"query_routing must be rank-2; got {array.shape}")
    if array.shape == (query_count, group_count):
        result = array
    elif array.shape == (group_count, query_count):
        result = array.T
    else:
        raise GroupControlEvidenceError(
            f"query_routing shape {array.shape} does not match Q={query_count}, K={group_count}"
        )
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < -1.0e-10):
        raise GroupControlEvidenceError("query_routing must be finite and non-negative")
    return np.where(result < 0.0, 0.0, result)


def _orient_overlap(value: Any, *, query_count: int, source_count: int, name: str) -> np.ndarray:
    array = _without_batch(value, name=name)
    if array.ndim != 2:
        raise GroupControlEvidenceError(f"{name} must be rank-2; got {array.shape}")
    if array.shape == (query_count, source_count):
        result = array
    elif array.shape == (source_count, query_count):
        result = array.T
    else:
        raise GroupControlEvidenceError(
            f"{name} shape {array.shape} does not match Q={query_count}, N={source_count}"
        )
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < -1.0e-10):
        raise GroupControlEvidenceError(f"{name} must be finite and non-negative")
    return np.where(result < 0.0, 0.0, result)


def _orient_moment(value: Any, *, query_count: int, source_count: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = _without_batch(value, name=name)
    if array.ndim != 3:
        raise GroupControlEvidenceError(f"{name} must be rank-3 [Q,N,D]; got {array.shape}")
    if array.shape[0:2] == (query_count, source_count):
        result = array
    elif array.shape[0:2] == (source_count, query_count):
        result = array.transpose(1, 0, 2)
    else:
        raise GroupControlEvidenceError(
            f"{name} shape {array.shape} does not match Q={query_count}, N={source_count}"
        )
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise GroupControlEvidenceError(f"{name} must be finite")
    if result.shape[-1] != CONTROL_DIM:
        raise GroupControlEvidenceError(
            f"{name} must use the Run-1406 control dimension D={CONTROL_DIM}; got D={result.shape[-1]}"
        )
    return result


def _orient_group_control(value: Any, *, group_count: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    raw = _as_array(value, name=name)
    if raw.ndim == 2 and raw.shape in ((group_count, CONTROL_DIM), (CONTROL_DIM, group_count)):
        array = raw
    else:
        array = _without_batch(raw, name=name)
    if array.ndim != 2:
        raise GroupControlEvidenceError(f"{name} must be rank-2 [K,D]; got {array.shape}")
    if array.shape[0] == group_count:
        result = array
    elif array.shape[1] == group_count:
        result = array.T
    else:
        raise GroupControlEvidenceError(f"{name} shape {array.shape} does not match K={group_count}")
    result = np.asarray(result, dtype=np.float64)
    if result.shape[-1] != CONTROL_DIM or not np.isfinite(result).all():
        raise GroupControlEvidenceError(f"{name} must be finite with D={CONTROL_DIM}")
    return result


def _weighted_centres(coords: np.ndarray, weights: np.ndarray, incidence: np.ndarray) -> np.ndarray:
    mass = np.einsum("s,sk->k", weights, incidence)
    centres = np.full((incidence.shape[1], coords.shape[1]), np.nan, dtype=np.float64)
    valid = mass > 0.0
    if np.any(valid):
        centres[valid] = np.einsum("s,sk,sd->kd", weights, incidence, coords)[valid] / mass[valid, None]
    return centres


def _centres(mapping: Mapping[str, Any], name: str, *, fallback: np.ndarray) -> np.ndarray:
    value = _find(mapping, name, required=False)
    if value is None:
        return fallback
    raw = _as_array(value, name=name)
    array = raw if raw.shape == fallback.shape else _without_batch(raw, name=name)
    if array.ndim != 2 or array.shape != fallback.shape:
        raise GroupControlEvidenceError(f"{name} must have shape {fallback.shape}; got {array.shape}")
    result = np.asarray(array, dtype=np.float64)
    # Empty groups are represented by NaN centres; infinities are never valid.
    if np.any(np.isinf(result)):
        raise GroupControlEvidenceError(f"{name} has invalid infinite values")
    return result


def _mapping_containers(outputs: Any) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = []
    if isinstance(outputs, Mapping):
        for key in (
            "group_control_debug",
            "interaction_aux",
            "routing_aux",
            "provisional_read_aux",
            "debug_aux",
        ):
            value = outputs.get(key)
            if isinstance(value, Mapping):
                containers.append(value)
        containers.append(outputs)
        prepared = outputs.get("prepared_state")
    else:
        prepared = outputs
    prepared_inner = getattr(prepared, "prepared", prepared)
    if isinstance(prepared_inner, Mapping):
        for key in ("group_control_debug", "interaction_aux", "encoded"):
            value = prepared_inner.get(key)
            if isinstance(value, Mapping):
                containers.append(value)
    for candidate in (
        getattr(prepared_inner, "group_control_debug", None),
        getattr(prepared_inner, "interaction_aux", None),
        getattr(getattr(prepared_inner, "backend_state", None), "diagnostics", None),
    ):
        if isinstance(candidate, Mapping):
            containers.append(candidate)
    return containers


def collect_group_control_payload(outputs: Any, *, query_xy: Any) -> dict[str, Any]:
    """Merge the opt-in debug containers and attach query/encoded geometry."""

    merged: dict[str, Any] = {}
    for container in _mapping_containers(outputs):
        merged.update(container)
    prepared = outputs.get("prepared_state") if isinstance(outputs, Mapping) else outputs
    prepared_inner = getattr(prepared, "prepared", prepared)
    encoded = getattr(prepared_inner, "encoded", None)
    if isinstance(prepared_inner, Mapping):
        encoded = prepared_inner.get("encoded", encoded)
    if encoded is not None:
        for key, aliases in {
            "module_coords": ("module_centers", "module_centres"),
            "module_present": ("module_present",),
            "env_coords": ("env_coords",),
            "env_weights": ("env_weights",),
        }.items():
            for alias in aliases:
                value = getattr(encoded, alias, None)
                if value is None and isinstance(encoded, Mapping):
                    value = encoded.get(alias)
                if value is not None:
                    merged.setdefault(key, value)
                    break
    merged["query_xy"] = query_xy
    return merged


_RAW_PHASE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("P0", "initial_port_"),
    ("P1", "provisional_"),
    ("P1", "provisional_read_"),
    ("P2_consistency", "port_global_"),
    ("P2", ""),
)
_RAW_PHASE_FIELDS: dict[str, tuple[str, ...]] = {
    "logical_path_count": ("logical_paths", "logical_path_count", "raw_path_count"),
    "unique_pair_count": ("unique_pairs", "unique_pair_count"),
    "actual_fine_call_count": (
        "actual_fine_call_count",
        "fine_rows",
        "fine_rows_forward",
        "fine_forward_rows",
    ),
    "module_mlp_rows": (
        "module_mlp_rows",
        "actual_module_mlp_rows",
        "module_fine_rows",
        "fine_rows",
        "fine_rows_forward",
    ),
    "environment_geometry_network_rows": (
        "geometry_network_rows",
        "geometry_rows",
        "geometry_rows_forward",
    ),
    "environment_content_rows": (
        "content_rows",
        "content_dot_rows",
        "content_dot_rows_forward",
    ),
    "scalar_control_rows": ("scalar_control_rows",),
    "source_projection_rows": (
        "source_projection_rows",
        "source_projection_count",
    ),
    "forward_call_count": ("forward_call_count", "fine_forward_call_count", "forward_calls"),
    "checkpoint_recompute_count": (
        "checkpoint_recompute_count",
        "checkpoint_recomputations",
        "fine_rows_recompute",
        "recompute_count",
    ),
    "valid_pair_denominator": (
        "valid_pair_denominator",
        "valid_pairs",
        "dense_valid_pair_count",
        "valid_pair_count",
    ),
    "padded_pair_denominator": (
        "padded_pair_denominator",
        "fine_rows_padded",
        "padded_rows",
    ),
}


def build_group_control_phase_records(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt explicit Thermal raw phase prefixes to the evidence ledger.

    This function only maps fields that are explicitly present.  In
    particular, an unprefixed P2 field never creates a P0 or P1 record.  The
    caller can therefore distinguish a real provisional/initial-port read
    from a P2-only debug payload.
    """

    containers = _mapping_containers(payload)
    if not containers:
        containers = [payload]
    merged: dict[str, Any] = {}
    for container in containers:
        merged.update(container)
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for phase, prefix in _RAW_PHASE_PREFIXES:
        phase_values: dict[str, Any] = {}
        for key, value in merged.items():
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            bare = key[len(prefix) :]
            if bare.startswith("group_control_"):
                phase_values[bare] = value
        if not phase_values:
            continue
        phase_record: dict[str, dict[str, Any]] = {}
        for source_kind in SOURCE_TYPES:
            source_record: dict[str, Any] = {}
            source_prefix = f"group_control_{source_kind}_"
            for field, suffixes in _RAW_PHASE_FIELDS.items():
                for suffix in suffixes:
                    key = source_prefix + suffix
                    if key in phase_values:
                        source_record[field] = phase_values[key]
                        break
            if source_record:
                phase_record[source_kind] = source_record
        if phase_record:
            records.setdefault(phase, {}).update(phase_record)
    return records


def canonicalize_group_control_arrays(
    payload: Mapping[str, Any],
    *,
    expected_group_count: int = GROUP_COUNT,
) -> tuple[GroupControlArrays, Mapping[str, Any] | None]:
    """Validate maps and derive scalar overlaps without enumerating triples."""

    module_coords = _geometry(payload, "module_coords")
    env_coords = _geometry(payload, "env_coords")
    query_xy = _geometry(payload, "query_xy")
    module_present_value = payload.get("module_present")
    module_present = (
        np.ones(len(module_coords), dtype=np.float64)
        if module_present_value is None
        else _without_batch(module_present_value, name="module_present").reshape(-1).astype(np.float64)
    )
    if len(module_present) != len(module_coords):
        raise GroupControlEvidenceError("module_present must align with module_coords")
    env_weights_value = payload.get("env_weights", payload.get("environment_weights"))
    env_weights = (
        np.ones(len(env_coords), dtype=np.float64)
        if env_weights_value is None
        else _without_batch(env_weights_value, name="env_weights").reshape(-1).astype(np.float64)
    )
    if len(env_weights) != len(env_coords) or not np.isfinite(env_weights).all() or np.any(env_weights <= 0.0):
        raise GroupControlEvidenceError("env_weights must be finite and strictly positive")
    query_value = _find(payload, "query_routing")
    raw_query = _without_batch(query_value, name="query_routing")
    if raw_query.ndim != 2:
        raise GroupControlEvidenceError("query_routing must be rank-2")
    inferred_groups = raw_query.shape[-1] if raw_query.shape[0] == len(query_xy) else raw_query.shape[0]
    if expected_group_count and int(inferred_groups) != int(expected_group_count):
        raise GroupControlEvidenceError(f"expected K={expected_group_count}, got K={inferred_groups}")
    group_count = int(expected_group_count or inferred_groups)
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
    query_routing = _orient_query(query_value, query_count=len(query_xy), group_count=group_count)
    module_overlap_value = _find(payload, "module_overlap", required=False)
    environment_overlap_value = _find(payload, "environment_overlap", required=False)
    module_overlap = (
        np.einsum("qk,sk->qs", query_routing, module_incidence)
        if module_overlap_value is None
        else _orient_overlap(module_overlap_value, query_count=len(query_xy), source_count=len(module_coords), name="module_overlap")
    )
    environment_overlap = (
        np.einsum("qk,sk->qs", query_routing, environment_incidence)
        if environment_overlap_value is None
        else _orient_overlap(environment_overlap_value, query_count=len(query_xy), source_count=len(env_coords), name="environment_overlap")
    )
    module_fallback = _weighted_centres(module_coords, module_present, module_incidence)
    environment_fallback = _weighted_centres(env_coords, env_weights, environment_incidence)
    module_centres = _centres(payload, "module_group_centres", fallback=module_fallback)
    environment_centres = _centres(payload, "environment_group_centres", fallback=environment_fallback)
    group_control = _orient_group_control(
        _find(payload, "group_control", required=False),
        group_count=group_count,
        name="group_control_h",
    )
    module_moment = _orient_moment(
        _find(payload, "module_control_moment", required=False),
        query_count=len(query_xy),
        source_count=len(module_coords),
        name="module_control_moment",
    )
    environment_moment = _orient_moment(
        _find(payload, "environment_control_moment", required=False),
        query_count=len(query_xy),
        source_count=len(env_coords),
        name="environment_control_moment",
    )
    if module_moment is None and group_control is not None:
        module_moment = np.einsum(
            "qk,sk,kd->qsd",
            query_routing,
            module_incidence,
            group_control,
        )
    if environment_moment is None and group_control is not None:
        environment_moment = np.einsum(
            "qk,sk,kd->qsd",
            query_routing,
            environment_incidence,
            group_control,
        )
    phase_ledger = next(
        (payload[key] for key in ("group_control_phase_ledger", "group_control_ledger", "phase_ledger") if isinstance(payload.get(key), Mapping)),
        None,
    )
    raw_phase_ledger = build_group_control_phase_records(payload)
    if phase_ledger is None:
        phase_ledger = raw_phase_ledger or None
    elif raw_phase_ledger:
        # Prefer an explicit nested record, but fill only genuinely missing
        # phases/source records from the Thermal prefix adapter.
        merged_phase_ledger: dict[str, Any] = {str(key): value for key, value in phase_ledger.items()}
        for phase, phase_value in raw_phase_ledger.items():
            existing = merged_phase_ledger.get(phase)
            if not isinstance(existing, Mapping):
                merged_phase_ledger[phase] = phase_value
                continue
            merged_sources = dict(existing)
            for source_kind, source_value in phase_value.items():
                merged_sources.setdefault(source_kind, source_value)
            merged_phase_ledger[phase] = merged_sources
        phase_ledger = merged_phase_ledger
    arrays = GroupControlArrays(
        module_coords=module_coords,
        module_present=module_present,
        env_coords=env_coords,
        env_weights=env_weights,
        module_incidence=module_incidence,
        environment_incidence=environment_incidence,
        query_xy=query_xy,
        query_routing=query_routing,
        module_overlap=module_overlap,
        environment_overlap=environment_overlap,
        module_group_centres=module_centres,
        environment_group_centres=environment_centres,
        group_control=group_control,
        module_control_moment=module_moment,
        environment_control_moment=environment_moment,
    )
    return arrays, phase_ledger


def _selected_query(arrays: GroupControlArrays) -> int:
    active = arrays.active_module_mask
    if not np.any(active):
        return max(len(arrays.query_xy) - 1, 0)
    distances = np.linalg.norm(
        arrays.query_xy[:, None, :] - arrays.module_coords[active][None, :, :], axis=-1
    )
    return int(np.argmax(np.min(distances, axis=1)))


def _logical_triples(arrays: GroupControlArrays, query_index: int, source_kind: str) -> np.ndarray:
    if source_kind == "module":
        incidence = arrays.module_incidence
        active = arrays.active_module_mask
    else:
        incidence = arrays.environment_incidence
        active = np.ones(len(arrays.env_coords), dtype=bool)
    groups = np.flatnonzero(arrays.query_routing[query_index] > 0.0)
    rows: list[tuple[int, int, int]] = []
    for group in groups:
        for source in np.flatnonzero((incidence[:, group] > 0.0) & active):
            rows.append((int(query_index), int(group), int(source)))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def _unique_pairs(arrays: GroupControlArrays, query_index: int, source_kind: str) -> np.ndarray:
    overlap = arrays.module_overlap if source_kind == "module" else arrays.environment_overlap
    active = arrays.active_module_mask if source_kind == "module" else np.ones(len(arrays.env_coords), dtype=bool)
    rows = np.flatnonzero((overlap[query_index] > 0.0) & active)
    return np.asarray(
        [(int(query_index), int(source), float(overlap[query_index, source])) for source in rows],
        dtype=np.float64,
    ).reshape(-1, 3)


def _pair_summary(arrays: GroupControlArrays, source_kind: str) -> dict[str, Any]:
    if source_kind == "module":
        incidence = arrays.module_incidence
        overlap = arrays.module_overlap
        active = arrays.active_module_mask
    else:
        incidence = arrays.environment_incidence
        overlap = arrays.environment_overlap
        active = np.ones(len(arrays.env_coords), dtype=bool)
    logical_by_pair = np.einsum(
        "qk,sk->qs",
        (arrays.query_routing > 0.0).astype(np.float64),
        (incidence > 0.0).astype(np.float64),
    )
    logical_by_pair[:, ~active] = 0
    logical = int(logical_by_pair.sum())
    unique = int(np.count_nonzero((overlap > 0.0) & active[None, :]))
    dense = int(arrays.query_routing.shape[0] * int(active.sum()))
    multiplicities = logical_by_pair[logical_by_pair > 0.0]
    histogram = {
        str(value): int(np.count_nonzero(multiplicities == value))
        for value in range(1, arrays.group_count + 1)
    }
    return {
        "logical_path_count": logical,
        "unique_pair_count": unique,
        "multiplicity": float(logical / max(unique, 1)),
        "unique_over_dense_valid": None if dense == 0 else float(unique / dense),
        "dense_valid_pair_count": dense,
        "active_source_count": int(active.sum()),
        "pair_multiplicity_min": None if not len(multiplicities) else float(np.min(multiplicities)),
        "pair_multiplicity_mean": None if not len(multiplicities) else float(np.mean(multiplicities)),
        "pair_multiplicity_max": None if not len(multiplicities) else float(np.max(multiplicities)),
        "pair_multiplicity_histogram": histogram,
        "ledger_source": "derived_from_positive_membership_and_overlap",
    }


def _phase_source(entry: Mapping[str, Any], source_kind: str) -> Mapping[str, Any]:
    source = entry.get(source_kind)
    if isinstance(source, Mapping):
        return source
    return entry


def _numeric_value(value: Any) -> float | None:
    """Read one scalar, accepting ``{value: ...}`` ledger cells."""

    if isinstance(value, Mapping):
        value = value.get("value", value.get("numerator"))
    return _finite_number(value)


def _phase_value(
    entry: Mapping[str, Any], source_kind: str, names: Sequence[str]
) -> tuple[float | None, dict[str, Any]]:
    """Read a phase field and sum receiver-chunk records when present.

    The backend may emit one source record per receiver chunk under
    ``chunks``/``receiver_chunks``.  Counts and numerators are additive; the
    comparison tool must never average chunk ratios.  The returned metadata
    makes that aggregation inspectable in the board/report.
    """

    source = _phase_source(entry, source_kind)
    prefix = "module_" if source_kind == "module" else "environment_"

    def direct(mapping: Mapping[str, Any]) -> Any:
        for name in names:
            if name in mapping:
                return mapping[name]
        for name in names:
            key = f"{prefix}{name}"
            if key in mapping:
                return mapping[key]
        return None

    chunk_values: list[float] = []
    for chunk_key in ("chunks", "receiver_chunks", "chunk_records", "records"):
        chunks = source.get(chunk_key)
        if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
            continue
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            value = direct(chunk)
            numeric = _numeric_value(value)
            if numeric is not None:
                chunk_values.append(numeric)
        if chunk_values:
            return sum(chunk_values), {
                "policy": "sum_over_receiver_chunks",
                "chunk_count": len(chunk_values),
            }

    value = direct(source)
    numeric = _numeric_value(value)
    metadata: dict[str, Any] = {"policy": "direct_backend_value", "chunk_count": 0}
    if isinstance(value, Mapping):
        denominator = _numeric_value(value.get("denominator"))
        if denominator is not None:
            metadata["denominator"] = denominator
    return numeric, metadata


def _finite_number(value: Any) -> float | None:
    try:
        if hasattr(value, "detach") and callable(value.detach):
            value = value.detach().cpu().numpy()
        number = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return number if math.isfinite(number) else None


def _merge_phase_record(derived: Mapping[str, Any], entry: Mapping[str, Any] | None, source_kind: str) -> dict[str, Any]:
    result = dict(derived)
    if entry is None:
        result.update(
            {
                "status": "unavailable",
                "actual_fine_call_count": None,
                "module_mlp_rows": None,
                "environment_geometry_network_rows": None,
                "environment_content_rows": None,
                "scalar_control_rows": None,
                "source_projection_rows": None,
                "forward_call_count": None,
                "checkpoint_recompute_count": None,
                "ledger_source": "derived_support_only; phase debug ledger missing",
            }
        )
        return result
    result["status"] = "ok"
    aliases = {
        "backend_logical_path_count": (
            "logical_path_count",
            "logical_paths",
            "logical_numerator",
        ),
        "backend_unique_pair_count": (
            "unique_pair_count",
            "unique_pairs",
            "unique_numerator",
        ),
        "actual_fine_call_count": ("actual_fine_call_count", "fine_call_count", "executed_pair_count", "pair_call_count", "call_count"),
        "module_mlp_rows": (
            "module_mlp_rows",
            "actual_module_mlp_rows",
            "module_fine_rows",
            "fine_rows",
            "fine_rows_forward",
        ),
        "environment_geometry_network_rows": ("environment_geometry_network_rows", "env_geometry_rows", "geometry_network_rows"),
        "environment_content_rows": ("environment_content_rows", "env_content_rows", "content_dot_rows"),
        "scalar_control_rows": ("scalar_control_rows", "control_scalar_rows"),
        "source_projection_rows": ("source_projection_rows", "source_projection_count"),
        "forward_call_count": ("forward_call_count", "fine_forward_call_count", "forward_calls"),
        "checkpoint_recompute_count": ("checkpoint_recompute_count", "recompute_count", "checkpoint_recomputations", "fine_rows_recompute"),
    }
    aggregation: dict[str, Any] = {}
    for result_key, names in aliases.items():
        value, metadata = _phase_value(entry, source_kind, names)
        result[result_key] = value
        aggregation[result_key] = metadata
        if result_key.endswith(("count", "rows")) and result[result_key] is not None:
            result[result_key] = int(result[result_key])
    result["receiver_chunk_aggregation"] = aggregation
    denominator_aliases = {
        "valid_pair_denominator": ("valid_pair_denominator", "valid_pairs", "dense_valid_pair_count", "valid_pair_count"),
        "padded_pair_denominator": ("padded_pair_denominator", "padded_pairs", "padded_pair_count"),
    }
    for result_key, names in denominator_aliases.items():
        value, metadata = _phase_value(entry, source_kind, names)
        result[result_key] = None if value is None else int(value)
        aggregation[result_key] = metadata
    for semantic_key, backend_key in (
        ("logical_path_count", "backend_logical_path_count"),
        ("unique_pair_count", "backend_unique_pair_count"),
    ):
        backend_value = result.get(backend_key)
        if backend_value is not None:
            result[f"derived_{semantic_key}"] = result.get(semantic_key)
            result[semantic_key] = int(backend_value)
    if result.get("unique_pair_count", 0) > 0:
        result["multiplicity"] = float(result.get("logical_path_count", 0) / result["unique_pair_count"])
    result["ledger_source"] = "backend_group_control_phase_ledger"
    return result


def phase_ledger(arrays: GroupControlArrays, phase_records: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return P0/P1/P2 ledgers, preserving unavailable phase records."""

    records = phase_records or {}
    result: dict[str, Any] = {}
    for phase in PHASES:
        raw_entry = next(
            (records[key] for key in (phase, phase.lower()) if isinstance(records.get(key), Mapping)),
            None,
        )
        derived = {
            "phase": phase,
            "module": _pair_summary(arrays, "module"),
            "environment": _pair_summary(arrays, "environment"),
        }
        if raw_entry is None and phase == "P2":
            raw_entry = records if any(key in records for key in ("module", "environment", "module_logical_path_count")) else None
        for source_kind in SOURCE_TYPES:
            derived[source_kind] = _merge_phase_record(derived[source_kind], raw_entry, source_kind)
        result[phase] = derived
    return result


def semantic_metrics(
    arrays: GroupControlArrays,
    *,
    selected_query_index: int | None = None,
    prepared_decode_median_ms: float | None = None,
) -> GroupControlMetrics:
    """Compute selected logical triples, unique pairs, moments, and summaries."""

    selected = _selected_query(arrays) if selected_query_index is None else int(selected_query_index)
    if not 0 <= selected < len(arrays.query_xy):
        raise GroupControlEvidenceError(f"selected_query_index={selected} outside Q={len(arrays.query_xy)}")
    module_logical = _logical_triples(arrays, selected, "module")
    environment_logical = _logical_triples(arrays, selected, "environment")
    module_unique = _unique_pairs(arrays, selected, "module")
    environment_unique = _unique_pairs(arrays, selected, "environment")
    module_moment = None if arrays.module_control_moment is None else arrays.module_control_moment[selected, module_unique[:, 1].astype(int)]
    environment_moment = None if arrays.environment_control_moment is None else arrays.environment_control_moment[selected, environment_unique[:, 1].astype(int)]
    module_summary = _pair_summary(arrays, "module")
    environment_summary = _pair_summary(arrays, "environment")
    moment_width = module_moment if module_moment is not None else environment_moment
    values: dict[str, Any] = {
        "K": arrays.group_count,
        "D": None if moment_width is None else int(moment_width.shape[-1]),
        "Q": int(arrays.query_routing.shape[0]),
        "M_active": arrays.active_module_mask.sum().item(),
        "M_padded": len(arrays.module_coords),
        "E_active": len(arrays.env_coords),
        "selected_query_index": selected,
        "selected_query_xy": arrays.query_xy[selected].tolist(),
        "module_logical_path_count": module_summary["logical_path_count"],
        "environment_logical_path_count": environment_summary["logical_path_count"],
        "module_unique_pair_count": module_summary["unique_pair_count"],
        "environment_unique_pair_count": environment_summary["unique_pair_count"],
        "module_multiplicity": module_summary["multiplicity"],
        "environment_multiplicity": environment_summary["multiplicity"],
        "module_unique_over_dense_valid": module_summary["unique_over_dense_valid"],
        "environment_unique_over_dense_valid": environment_summary["unique_over_dense_valid"],
        "module_pair_multiplicity_histogram": module_summary["pair_multiplicity_histogram"],
        "environment_pair_multiplicity_histogram": environment_summary["pair_multiplicity_histogram"],
        "route_semantics": "logical learned group paths; unique q->source execution candidates; not physical causality",
    }
    if prepared_decode_median_ms is not None and math.isfinite(float(prepared_decode_median_ms)):
        values["prepared_decode_median_ms"] = float(prepared_decode_median_ms)
    return GroupControlMetrics(
        values=values,
        selected_query_index=selected,
        selected_module_logical_triples=module_logical,
        selected_environment_logical_triples=environment_logical,
        selected_module_unique_pairs=module_unique,
        selected_environment_unique_pairs=environment_unique,
        selected_module_moments=module_moment,
        selected_environment_moments=environment_moment,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_group_control_npz(
    path: str | Path,
    arrays: GroupControlArrays,
    metrics: GroupControlMetrics,
    ledger: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "module_coords": arrays.module_coords,
        "module_present": arrays.module_present,
        "env_coords": arrays.env_coords,
        "env_weights": arrays.env_weights,
        "module_incidence": arrays.module_incidence,
        "environment_incidence": arrays.environment_incidence,
        "query_xy": arrays.query_xy,
        "query_routing": arrays.query_routing,
        "module_overlap": arrays.module_overlap,
        "environment_overlap": arrays.environment_overlap,
        "module_group_centres": arrays.module_group_centres,
        "environment_group_centres": arrays.environment_group_centres,
        "selected_module_logical_triples": metrics.selected_module_logical_triples,
        "selected_environment_logical_triples": metrics.selected_environment_logical_triples,
        "selected_module_unique_pairs": metrics.selected_module_unique_pairs,
        "selected_environment_unique_pairs": metrics.selected_environment_unique_pairs,
        "metrics_json": np.asarray(json.dumps(_jsonable(metrics.values), sort_keys=True)),
        "ledger_json": np.asarray(json.dumps(_jsonable(ledger), sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(_jsonable(dict(metadata or {})), sort_keys=True)),
    }
    if arrays.group_control is not None:
        payload["group_control_h"] = arrays.group_control
    if arrays.module_control_moment is not None:
        payload["module_control_moment"] = arrays.module_control_moment
    if arrays.environment_control_moment is not None:
        payload["environment_control_moment"] = arrays.environment_control_moment
    if metrics.selected_module_moments is not None:
        payload["selected_module_moments"] = metrics.selected_module_moments
    if metrics.selected_environment_moments is not None:
        payload["selected_environment_moments"] = metrics.selected_environment_moments
    np.savez_compressed(destination, **payload)
    return destination


def load_group_control_npz(path: str | Path) -> tuple[GroupControlArrays, GroupControlMetrics, dict[str, Any], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    # Historical maps retain the six-group contract. Budgeted maps carry an
    # explicit gate plan and may store either full or compact group columns.
    expected_group_count = 0 if "case_group_budget_gate_values" in payload else GROUP_COUNT
    arrays, _ = canonicalize_group_control_arrays(payload, expected_group_count=expected_group_count)
    values = json.loads(str(np.asarray(payload["metrics_json"]).reshape(-1)[0]))
    selected = int(values.get("selected_query_index", 0))
    metrics = semantic_metrics(arrays, selected_query_index=selected, prepared_decode_median_ms=values.get("prepared_decode_median_ms"))
    ledger = json.loads(str(np.asarray(payload["ledger_json"]).reshape(-1)[0]))
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).reshape(-1)[0]))
    return arrays, metrics, ledger, metadata


__all__ = [
    "CONTROL_DIM",
    "GROUP_COUNT",
    "PHASES",
    "GroupControlArrays",
    "GroupControlEvidenceError",
    "GroupControlMetrics",
    "build_group_control_phase_records",
    "canonicalize_group_control_arrays",
    "collect_group_control_payload",
    "load_group_control_npz",
    "phase_ledger",
    "save_group_control_npz",
    "semantic_metrics",
]
