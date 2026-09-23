"""Read one explicit Run-1503 candidate checkpoint and write organization evidence.

This adapter is deliberately evaluation-only.  It loads the checkpoint named
by ``--checkpoint`` on CPU, evaluates the fixed 1024-query panel (or an
explicit 90-case population), and adapts the live prepared controls to the
existing Run-1503 and sparse-incidence evidence helpers.  It never chooses a
checkpoint from a run number, starts training, inspects GPU processes, or
relabels support/block area as measured executor work.

Typical bounded panel use::

    PYTHONPATH=src:Case_ThermalChannel/src:tools/diagnostics \
      python tools/diagnostics/run_run1503_organization_evidence.py \
      --checkpoint /abs/path/to/Run1503/epoch_0050_model.pt \
      --output-dir /abs/path/to/evaluations/run1503_organization \
      --device cpu --query-count 1024

Add ``--population`` to evaluate the complete 90-case split.  A repeated
``--case-id`` remains the explicit override for a bounded subset.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
EXPECTED_CASE_COUNT = 90
DEFAULT_KMAX = 12
DEFAULT_QUERY_COUNT = 1024
DEFAULT_QUERY_BATCH_SIZE = 1024
DEFAULT_CASE_IDS = ("0273", "0653")
EXPECTED_ARCHITECTURE = "adaptive_hyperedge_opening_honf"

# The imports are pure CPU contracts.  Runtime model/dataset imports remain
# lazy in ``_runtime_imports`` so focused tests can exercise all extraction
# and rendering logic without importing the case package.
if str(DIAGNOSTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS_ROOT))

import sparse_incidence_evidence as sparse_evidence
from run_run1503_diagnostics import collect_case_diagnostic


class Run1503OrganizationEvidenceError(ValueError):
    """Raised when candidate organization evidence cannot be audited."""


def _jsonable(value: Any) -> Any:
    """Convert diagnostic values to JSON without losing small arrays."""

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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _batched_measure(value: Any, *, name: str) -> np.ndarray:
    """Keep a prepared source measure on an explicit ``[B, M]`` axis.

    The shared evidence canonicalizer treats a one-dimensional vector as a
    batched value and therefore selects its first scalar when ``M > 1``.
    Prepared Run1503 controls are already ``[B, M]``; the only accepted
    compatibility conversion is an unbatched ``[M]`` vector to ``[1, M]``.
    """

    array = _to_numpy(value)
    if array.ndim == 1:
        array = array[None, ...]
    if array.ndim != 2:
        raise Run1503OrganizationEvidenceError(
            f"{name} must have shape [B,M] (or unbatched [M]), got {array.shape}"
        )
    return array


def _case_measure(value: Any, *, name: str, length: int) -> np.ndarray:
    array = _batched_measure(value, name=name)
    if int(array.shape[1]) != int(length):
        raise Run1503OrganizationEvidenceError(
            f"{name} padded source axis {array.shape[1]} does not match {length}"
        )
    return np.asarray(array[0], dtype=np.float64)


def _member(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _first(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        values = _to_numpy(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size == 0 or not math.isfinite(float(values[0])):
        return None
    return float(values[0])


def _finite_values(value: Any) -> np.ndarray:
    try:
        array = _to_numpy(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.zeros(0, dtype=np.float64)
    return array[np.isfinite(array)]


def _summary(values: Any) -> dict[str, float] | None:
    finite = _finite_values(values)
    if finite.size == 0:
        return None
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _distribution(values: Any) -> dict[str, Any] | None:
    finite = _finite_values(values)
    if finite.size == 0:
        return None
    return {
        "count": int(finite.size),
        "summary": _summary(finite),
        "positive_fraction": float(np.mean(finite > 0.0)),
        "zero_count": int(np.count_nonzero(finite == 0.0)),
    }


def _matrix(value: Any, *, query_count: int, group_count: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = _to_numpy(value, dtype=np.float64)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise Run1503OrganizationEvidenceError(f"{name} must be [Q,K], got {array.shape}")
    if tuple(array.shape) == (group_count, query_count):
        array = array.T
    if tuple(array.shape) != (query_count, group_count):
        raise Run1503OrganizationEvidenceError(
            f"{name} shape {array.shape} does not match [Q,K]=[{query_count},{group_count}]"
        )
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise Run1503OrganizationEvidenceError(f"{name} must be finite and nonnegative")
    return array


def _model_architecture(model: Any) -> str:
    config = getattr(model, "config", None)
    core = _member(config, "core_honf")
    architecture = _member(core, "forward_architecture")
    if architecture is None and isinstance(config, Mapping):
        core_payload = config.get("core_honf", config.get("core", {}))
        architecture = _member(core_payload, "forward_architecture")
    return "" if architecture is None else str(architecture)


def _prepared_payload(
    sample: Mapping[str, Any], prediction: Mapping[str, Any], query_xy: np.ndarray
) -> dict[str, Any]:
    """Adapt one live Run1503 prediction to both evidence contracts."""

    payload: dict[str, Any] = {}
    interaction = prediction.get("interaction_aux", {})
    routing = prediction.get("routing_maps", {})
    if isinstance(interaction, Mapping):
        payload.update(interaction)
    if isinstance(routing, Mapping):
        payload.update(routing)

    prepared_case = prediction.get("_prepared_state")
    prepared = _member(prepared_case, "prepared")
    encoded = _member(prepared, "encoded")
    backend_state = _member(prepared, "backend_state")
    controls = _member(backend_state, "group_control_state")
    if controls is None:
        controls = _member(prepared, "group_control_state")

    # Keep the canonical names alongside the maintained backend names.  This
    # makes the source of each organization map visible and lets the two
    # existing evidence modules consume the same detached payload.
    control_aliases = {
        "module_membership": "module_assignment",
        "environment_membership": "environment_assignment",
        "module_measure": "module_measure",
        "environment_measure": "environment_measure",
        "module_mass": "module_mass",
        "environment_mass": "environment_mass",
        "phase_occupied": "active_mask",
        "pi": "pi",
        "kappa": "kappa",
        "module_centres": "module_centres",
        "environment_centres": "environment_centres",
        "joint_centres": "joint_centres",
    }
    for source, target in control_aliases.items():
        value = _member(controls, source)
        if value is not None:
            if source in ("module_measure", "environment_measure"):
                value = _batched_measure(value, name=source)
            payload[target] = value
            if source == "module_membership":
                payload["module_membership"] = value
            elif source == "environment_membership":
                payload["environment_membership"] = value
            elif source == "module_measure":
                # ``occupancy_adaptive_evidence`` checks this maintained alias
                # before ``module_measure``.  Keep it on the same explicit
                # batch/padded-source axis as the authoritative control.
                payload["group_control_module_measure"] = value
            elif source == "environment_measure":
                payload["group_control_environment_measure"] = value
            elif source == "phase_occupied":
                # sparse-incidence compatibility data may expose an
                # unbatched ``sparse_incidence_phase_occupied`` vector.  The
                # prepared active mask has the registered ``[B,K]`` axis and
                # must win the shared alias lookup.
                payload["occupancy_group_active_mask"] = value
            elif source in ("module_centres", "environment_centres", "joint_centres"):
                centre_aliases = {
                    "module_centres": (
                        "occupancy_group_module_centres",
                        "occupancy_adaptive_module_centres",
                        "group_control_module_centres",
                    ),
                    "environment_centres": (
                        "occupancy_group_environment_centres",
                        "occupancy_adaptive_environment_centres",
                        "group_control_environment_centres",
                    ),
                    "joint_centres": (
                        "occupancy_group_joint_centres",
                        "occupancy_adaptive_joint_centres",
                        "group_control_joint_centres",
                    ),
                }
                for alias in centre_aliases[source]:
                    payload[alias] = value

    if payload.get("environment_measure") is None:
        payload["environment_measure"] = _member(encoded, "env_weights")
    environment_coordinates = _member(encoded, "env_coords")
    if environment_coordinates is not None:
        payload["environment_coords"] = environment_coordinates

    structure = sample.get("structure", {})
    if isinstance(structure, Mapping):
        if "module_centers" in structure:
            payload["module_coords"] = np.asarray(structure["module_centers"])
        if "module_present" in structure:
            payload["module_present"] = np.asarray(structure["module_present"])

    query_array = np.asarray(query_xy, dtype=np.float32)
    payload["query_xy"] = query_array[None, ...]
    for key in ("query_grid_indices", "query_source_count", "query_selection", "query_domain_bounds"):
        if key in sample:
            payload[key] = sample[key]
    payload.setdefault("query_source_count", int(query_array.shape[0]))
    payload.setdefault("query_selection", "recorded_query_grid")
    payload.setdefault(
        "query_domain_bounds",
        np.asarray(
            [query_array[:, 0].min(), query_array[:, 0].max(), query_array[:, 1].min(), query_array[:, 1].max()],
            dtype=np.float32,
        ),
    )

    query_assignment = _first(
        payload,
        (
            "query_assignment",
            "adaptive_hyperedge_query_sparse_assignment",
            "adaptive_hyperedge_fine_open_assignment",
            "group_control_adaptive_environment_alpha",
            "sparse_incidence_query_routing",
            "group_control_query_routing",
        ),
    )
    if query_assignment is not None:
        payload["query_assignment"] = query_assignment
        payload.setdefault("sparse_incidence_query_routing", query_assignment)

    coarse_assignment = _first(
        payload,
        (
            "coarse_assignment",
            "adaptive_hyperedge_query_coarse_assignment",
            "adaptive_hyperedge_coarse_assignment",
            "group_control_adaptive_environment_p",
        ),
    )
    if coarse_assignment is not None:
        payload["coarse_assignment"] = coarse_assignment

    opening_blend = _first(
        payload,
        (
            "opening_blend",
            "adaptive_hyperedge_opening_blend",
            "group_control_adaptive_opening_blend",
        ),
    )
    if opening_blend is not None:
        payload["opening_blend"] = opening_blend

    fine_mask = _first(
        payload,
        ("fine_open_mask", "adaptive_hyperedge_fine_open_mask", "adaptive_hyperedge_open_mask"),
    )
    if fine_mask is not None:
        payload["fine_open_mask"] = fine_mask

    coarse_available = _first(
        payload,
        ("coarse_available", "adaptive_hyperedge_coarse_available", "adaptive_hyperedge_coarse_global_available"),
    )
    if coarse_available is not None:
        payload["coarse_available"] = coarse_available

    # Run1503 emits flat, explicit executor counters.  Expose them through a
    # normal P2/environment ledger for sparse-incidence row accounting too;
    # no support-derived rows are synthesized here.
    if not isinstance(payload.get("phase_ledger"), Mapping):
        ledger_names = {
            "support_pairs": ("support_pairs", "group_control_environment_unique_pairs"),
            "executed_rows": ("fine_rows", "group_control_adaptive_fine_rows"),
            "padded_rows": ("padded_rows", "group_control_adaptive_fine_rows_padded"),
            "recomputed_rows": (
                "recomputed_rows",
                "group_control_environment_checkpoint_recomputations",
                "group_control_environment_fine_rows_recompute",
            ),
            "coarse_rows": ("coarse_rows", "group_control_adaptive_coarse_rows"),
        }
        environment_ledger: dict[str, Any] = {}
        for target, names in ledger_names.items():
            value = _first(payload, names)
            if value is not None:
                scalar_value = _scalar(value)
                environment_ledger[target] = value if scalar_value is None else scalar_value
        if environment_ledger:
            payload["phase_ledger"] = {"P2": {"environment": environment_ledger}}

    return payload


def _prediction_payload(
    sample: Mapping[str, Any], prediction: Mapping[str, Any], query_xy: np.ndarray
) -> dict[str, Any]:
    """Compatibility alias matching the existing population adapters."""

    return _prepared_payload(sample, prediction, query_xy)


def _distribution_from_payload(payload: Mapping[str, Any], names: Sequence[str], *, query_count: int, group_count: int, name: str) -> np.ndarray | None:
    value = _first(payload, names)
    return _matrix(value, query_count=query_count, group_count=group_count, name=name)


def canonicalize_candidate_case(
    payload: Mapping[str, Any],
    *,
    case_id: str,
    module_count: int | None = None,
    query_count: int | None = None,
    kmax: int = DEFAULT_KMAX,
) -> dict[str, Any]:
    """Create one inspectable sparse/Run1503 organization record."""

    if int(kmax) != DEFAULT_KMAX:
        raise Run1503OrganizationEvidenceError(
            f"Run1503 adaptive hyperedge organization requires registered K={DEFAULT_KMAX}, got {kmax}"
        )
    # Accept the canonical sparse-evidence spellings as well as the
    # Run1503 diagnostic spellings.  The adapter normally supplies both, but
    # keeping this boundary explicit makes the CPU contract useful for saved
    # map payloads and focused tests too.
    payload = dict(payload)
    if payload.get("environment_membership") is None and payload.get("environment_assignment") is not None:
        payload["environment_membership"] = payload["environment_assignment"]
    if payload.get("module_membership") is None and payload.get("module_assignment") is not None:
        payload["module_membership"] = payload["module_assignment"]
    if payload.get("env_coords") is None and payload.get("environment_coords") is not None:
        payload["env_coords"] = payload["environment_coords"]

    def ensure_case_axis(key: str, rank: int) -> None:
        value = payload.get(key)
        if value is None:
            return
        array = _to_numpy(value)
        if array.ndim == rank:
            payload[key] = array[None, ...]

    for key in (
        "module_coords",
        "environment_coords",
        "env_coords",
        "query_xy",
    ):
        ensure_case_axis(key, 2)
    for key in (
        "module_assignment",
        "module_membership",
        "environment_assignment",
        "environment_membership",
        "query_assignment",
        "coarse_assignment",
        "opening_blend",
        "fine_open_mask",
    ):
        ensure_case_axis(key, 2)
    for key in (
        "module_measure",
        "environment_measure",
        "active_mask",
        "pi",
        "kappa",
        "module_present",
    ):
        ensure_case_axis(key, 1)
    # Prepared controls are authoritative for source measures.  The live
    # interaction payload also carries maintained aliases such as
    # ``group_control_module_measure``; those can be unbatched ``[M]`` while
    # the prepared control is ``[B,M]``.  Synchronize all accepted measure
    # aliases after selecting the explicit batch/padded representation so the
    # shared canonicalizer cannot mistake the padded source axis for a batch.
    for canonical, aliases in (
        (
            "module_measure",
            (
                "occupancy_group_module_measure",
                "occupancy_adaptive_module_measure",
                "group_control_module_measure",
            ),
        ),
        (
            "environment_measure",
            (
                "occupancy_group_environment_measure",
                "occupancy_adaptive_environment_measure",
                "group_control_environment_measure",
            ),
        ),
    ):
        value = payload.get(canonical)
        if value is None:
            for alias in aliases:
                value = payload.get(alias)
                if value is not None:
                    break
        if value is not None:
            value = _batched_measure(value, name=canonical)
            payload[canonical] = value
            for alias in aliases:
                payload[alias] = value
    query_array = _to_numpy(payload.get("query_xy"), dtype=np.float64)
    while query_array.ndim > 2 and query_array.shape[0] == 1:
        query_array = query_array[0]
    if query_array.ndim != 2:
        raise Run1503OrganizationEvidenceError("query_xy must have shape [Q,2]")
    measured_queries = int(query_array.shape[0])
    if query_count is not None and measured_queries != int(query_count):
        raise Run1503OrganizationEvidenceError(
            f"query count {measured_queries} does not match requested {query_count}"
        )

    query_assignment = _first(
        payload,
        ("query_assignment", "adaptive_hyperedge_query_sparse_assignment", "group_control_adaptive_environment_alpha"),
    )
    query_matrix = _matrix(
        query_assignment,
        query_count=measured_queries,
        group_count=kmax,
        name="query_assignment",
    )
    if query_matrix is None:
        raise Run1503OrganizationEvidenceError("Run1503 payload has no alpha/query assignment")
    payload["query_assignment"] = query_assignment
    ensure_case_axis("query_assignment", 2)

    coarse_assignment = _first(
        payload,
        (
            "coarse_assignment",
            "adaptive_hyperedge_query_coarse_assignment",
            "adaptive_hyperedge_coarse_assignment",
            "group_control_adaptive_environment_p",
        ),
    )
    if coarse_assignment is not None:
        payload["coarse_assignment"] = coarse_assignment
        ensure_case_axis("coarse_assignment", 2)
    opening_blend = _first(
        payload,
        ("opening_blend", "adaptive_hyperedge_opening_blend", "group_control_adaptive_opening_blend"),
    )
    if opening_blend is not None:
        payload["opening_blend"] = opening_blend
        ensure_case_axis("opening_blend", 2)

    diagnostic = collect_case_diagnostic(
        payload,
        case_id=str(case_id),
        module_count=module_count,
    )
    sparse_record = sparse_evidence.canonicalize_case(
        payload,
        query_count=measured_queries,
        kmax=kmax,
    )
    maps = dict(sparse_record["maps"])
    maps["query_degree"] = (query_matrix > 0.0).sum(axis=-1).astype(np.int64)

    module_assignment = np.asarray(maps["module_assignment"], dtype=np.float64)
    environment_assignment = np.asarray(maps["environment_assignment"], dtype=np.float64)
    module_present = np.asarray(maps.get("module_present", np.ones(module_assignment.shape[0])), dtype=bool)
    environment_measure = _case_measure(
        payload.get("environment_measure"),
        name="environment_measure",
        length=environment_assignment.shape[0],
    )
    module_measure_value = payload.get("module_measure")
    module_measure = (
        np.ones(module_assignment.shape[0], dtype=np.float64)
        if module_measure_value is None
        else _case_measure(
            module_measure_value,
            name="module_measure",
            length=module_assignment.shape[0],
        )
    )
    if environment_measure.size != environment_assignment.shape[0]:
        raise Run1503OrganizationEvidenceError("environment_measure does not align with environment sources")
    if module_measure.size != module_assignment.shape[0]:
        raise Run1503OrganizationEvidenceError("module_measure does not align with module sources")

    module_degree = (module_assignment > 0.0).sum(axis=-1).astype(np.int64)
    environment_degree = (environment_assignment > 0.0).sum(axis=-1).astype(np.int64)
    empty_module_sources = (~module_present) | (module_degree == 0) | (module_measure <= 1.0e-12)
    empty_environment_sources = (environment_degree == 0) | (environment_measure <= 1.0e-12)
    maps.update(
        {
            "module_measure": module_measure,
            "environment_measure": environment_measure,
            "module_source_degree": module_degree,
            "environment_source_degree": environment_degree,
        }
    )

    group_count = int(module_assignment.shape[1])
    p_matrix = _distribution_from_payload(
        payload,
        ("coarse_assignment", "adaptive_hyperedge_query_coarse_assignment", "group_control_adaptive_environment_p"),
        query_count=measured_queries,
        group_count=group_count,
        name="p_assignment",
    )
    alpha_matrix = query_matrix
    opening_matrix = _distribution_from_payload(
        payload,
        ("opening_blend", "adaptive_hyperedge_opening_blend", "group_control_adaptive_opening_blend"),
        query_count=measured_queries,
        group_count=group_count,
        name="opening_blend",
    )
    if p_matrix is not None:
        maps["p_assignment"] = p_matrix
    maps["alpha_assignment"] = alpha_matrix
    if opening_matrix is not None:
        maps["opening_blend"] = opening_matrix

    geometry = diagnostic["geometry"]
    opening = diagnostic["opening"]
    active_mass = np.asarray(geometry["mass"], dtype=np.float64).reshape(-1)
    row = {
        "case_id": str(case_id),
        "module_count": None if module_count is None else int(module_count),
        **sparse_record["case"],
        "run1503_diagnostic": diagnostic,
        "active_mass_values": active_mass.tolist(),
        "active_group_count": int(np.count_nonzero(active_mass > 1.0e-12)),
        "active_mass_total": float(np.sum(active_mass)),
        "active_mass_distribution": _distribution(active_mass),
        "source_count": int(environment_assignment.shape[0]),
        "active_source_count": int(np.count_nonzero(~empty_environment_sources)),
        "module_source_count": int(module_assignment.shape[0]),
        "empty_group_count": len(geometry.get("empty_groups", [])),
        "empty_query_count": int(opening.get("empty_query_count", 0)),
        "empty_source_count": int(np.count_nonzero(empty_environment_sources)),
        "empty_environment_source_count": int(np.count_nonzero(empty_environment_sources)),
        "empty_module_source_count": int(np.count_nonzero(empty_module_sources)),
        "module_source_degree_distribution": _distribution(module_degree),
        "environment_source_degree_distribution": _distribution(environment_degree),
        "p_distribution": _distribution(p_matrix),
        "alpha_distribution": _distribution(alpha_matrix),
        "opening_distribution": _distribution(opening_matrix),
        "fine_work_ratio": _scalar(opening.get("observed_execution", {}).get("fine_work_ratio")),
        "fine_work_ratio_source": (
            "explicit_run1503_aux"
            if opening.get("observed_execution", {}).get("fine_work_ratio") is not None
            else "unavailable"
        ),
        "query_count": measured_queries,
        "query_degree_distribution": _distribution(maps["query_degree"]),
        "empty_count_semantics": (
            "empty_source_count counts environmental source rows with no positive group assignment or nonpositive measure; "
            "empty_module_source_count separately counts padded/inactive module rows."
        ),
    }
    return {"case": row, "maps": maps}


def _count_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float | int]:
    values = np.asarray([int(row.get(key, 0)) for row in rows], dtype=np.float64)
    return {
        "total": int(np.sum(values)),
        "mean": float(np.mean(values)) if values.size else 0.0,
        "min": int(np.min(values)) if values.size else 0,
        "max": int(np.max(values)) if values.size else 0,
        "cases_with_any": int(np.count_nonzero(values > 0.0)),
    }


def summarize_candidate_population(
    rows: Sequence[Mapping[str, Any]], *, expected_cases: int = 0
) -> dict[str, Any]:
    """Summarize the fixed panel or explicit population without relabeling it."""

    if not rows:
        raise Run1503OrganizationEvidenceError("candidate population is empty")
    if int(expected_cases) > 0 and len(rows) != int(expected_cases):
        raise Run1503OrganizationEvidenceError(
            f"expected {int(expected_cases)} cases, found {len(rows)}"
        )
    # The sparse-incidence summarizer already owns Kq/support/rank definitions.
    base = sparse_evidence.summarize_population(rows, expected_cases=len(rows))
    summary: dict[str, Any] = dict(base)
    summary.update(
        {
            "query_count_per_case": sorted({int(row["query_count"]) for row in rows}),
            "query_source_count_per_case": sorted(
                {int(row["query_source_count"]) for row in rows if row.get("query_source_count") is not None}
            ),
            "empty_group_count": _count_summary(rows, "empty_group_count"),
            "empty_query_count": _count_summary(rows, "empty_query_count"),
            "empty_source_count": _count_summary(rows, "empty_source_count"),
            "empty_environment_source_count": _count_summary(rows, "empty_environment_source_count"),
            "empty_module_source_count": _count_summary(rows, "empty_module_source_count"),
            "active_group_count": _summary([row.get("active_group_count") for row in rows]),
            "active_mass_total": _summary([row.get("active_mass_total") for row in rows]),
            "fine_work_ratio": _summary(
                [row["fine_work_ratio"] for row in rows if row.get("fine_work_ratio") is not None]
            ),
            "p_distribution_case_means": _summary(
                [row["p_distribution"]["summary"]["mean"] for row in rows if row.get("p_distribution")]
            ),
            "alpha_distribution_case_means": _summary(
                [row["alpha_distribution"]["summary"]["mean"] for row in rows if row.get("alpha_distribution")]
            ),
            "opening_distribution_case_means": _summary(
                [row["opening_distribution"]["summary"]["mean"] for row in rows if row.get("opening_distribution")]
            ),
            "organization_scope": "fixed representative panel" if len(rows) == 2 else "explicit selected population",
            "interpretation": (
                "Run1503 p, alpha, opening, source degrees, masses, and Kq describe learned organization. "
                "Group labels are permutation-ambiguous and do not establish physical causality. "
                "Fine work ratio is retained only when emitted by explicit Run1503 executor auxiliary data."
            ),
        }
    )
    return summary


def _figure_setup(path: Path) -> Any | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - runtime environment normally has matplotlib
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return plt


def _case_ticks(labels: Sequence[str], *, max_ticks: int = 12) -> tuple[np.ndarray, list[str]]:
    """Thin population case labels while retaining the first and last case."""

    if len(labels) <= max_ticks:
        positions = np.arange(len(labels), dtype=np.int64)
    else:
        step = max(1, math.ceil((len(labels) - 1) / (max_ticks - 1)))
        positions = np.arange(0, len(labels), step, dtype=np.int64)
        if int(positions[-1]) != len(labels) - 1:
            positions = np.r_[positions, len(labels) - 1]
    return positions, [str(labels[index]) for index in positions]


def render_source_degree_figure(rows: Sequence[Mapping[str, Any]], array_dir: Path, path: Path) -> None:
    plt = _figure_setup(path)
    if plt is None:
        return
    module: list[np.ndarray] = []
    environment: list[np.ndarray] = []
    labels: list[str] = []
    means_m: list[float] = []
    means_e: list[float] = []
    for row in rows:
        array_path = array_dir / f"{row['case_id']}.npz"
        if not array_path.is_file():
            continue
        with np.load(array_path) as data:
            module.append(np.asarray(data["module_source_degree"], dtype=np.float64))
            environment.append(np.asarray(data["environment_source_degree"], dtype=np.float64))
        labels.append(str(row["case_id"]))
        means_m.append(float(np.mean(module[-1])))
        means_e.append(float(np.mean(environment[-1])))
    if not module:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    max_degree = max(int(np.max(values)) for values in module + environment)
    bins = np.arange(max_degree + 2) - 0.5
    axes[0].hist(
        np.concatenate(module),
        bins=bins,
        alpha=0.72,
        label="module source degree",
        color="#0072B2",
    )
    axes[0].hist(
        np.concatenate(environment),
        bins=bins,
        alpha=0.62,
        label="environment source degree",
        color="#009E73",
    )
    axes[0].set_xlabel("positive source-to-group degree")
    axes[0].set_ylabel("source rows")
    axes[0].set_title("Run 1503 source degree")
    axes[0].legend(frameon=False, fontsize=8)
    x = np.arange(len(labels))
    axes[1].plot(x, means_m, "s-", label="module", color="#0072B2")
    axes[1].plot(x, means_e, "o-", label="environment", color="#009E73")
    tick_positions, tick_labels = _case_ticks(labels)
    axes[1].set_xticks(tick_positions, tick_labels, rotation=55)
    axes[1].set_xlabel("case ID")
    axes[1].set_ylabel("mean positive degree")
    axes[1].set_title("Per-case source degree")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def render_active_mass_figure(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _figure_setup(path)
    if plt is None:
        return
    labels = [str(row["case_id"]) for row in rows]
    masses = np.asarray([row.get("active_mass_values", []) for row in rows], dtype=np.float64)
    active = np.asarray([int(row.get("active_group_count", 0)) for row in rows], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    image = axes[0].imshow(masses, aspect="auto", interpolation="nearest", cmap="viridis")
    figure.colorbar(image, ax=axes[0], label="active group mass")
    tick_positions, tick_labels = _case_ticks(labels)
    axes[0].set_yticks(tick_positions, tick_labels)
    axes[0].set_xlabel("learned group ID")
    axes[0].set_ylabel("case ID")
    axes[0].set_title("Run 1503 active group mass")
    axes[1].bar(np.arange(len(labels)), active, color="#E69F00")
    axes[1].set_xticks(tick_positions, tick_labels, rotation=55)
    axes[1].set_xlabel("case ID")
    axes[1].set_ylabel("groups with mass > 1e-12")
    axes[1].set_title("Active group count")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def render_p_alpha_opening_figure(rows: Sequence[Mapping[str, Any]], array_dir: Path, path: Path) -> None:
    plt = _figure_setup(path)
    if plt is None:
        return
    arrays: dict[str, list[np.ndarray]] = {"p": [], "alpha": [], "opening": []}
    for row in rows:
        array_path = array_dir / f"{row['case_id']}.npz"
        if not array_path.is_file():
            continue
        with np.load(array_path) as data:
            for label, key in (("p", "p_assignment"), ("alpha", "alpha_assignment"), ("opening", "opening_blend")):
                if key in data:
                    arrays[label].append(np.asarray(data[key], dtype=np.float64).reshape(-1))
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    colors = {"p": "#0072B2", "alpha": "#D55E00", "opening": "#009E73"}
    titles = {"p": "coarse p", "alpha": "fine alpha", "opening": "opening blend"}
    for axis, label in zip(axes, ("p", "alpha", "opening"), strict=True):
        values = np.concatenate(arrays[label]) if arrays[label] else np.zeros(0, dtype=np.float64)
        if values.size:
            axis.hist(values, bins=24, color=colors[label], alpha=0.82)
        else:
            axis.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=axis.transAxes)
        axis.set_xlabel(titles[label])
        axis.set_ylabel("entries")
        axis.set_title(f"{titles[label]} distribution")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "Run 1503 p / alpha / opening distributions\n"
        "alpha and opening are learned organization evidence; they are not physical causality",
        fontsize=12,
    )
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def render_fine_work_ratio_figure(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _figure_setup(path)
    if plt is None:
        return
    labels = [str(row["case_id"]) for row in rows]
    values = np.asarray(
        [np.nan if row.get("fine_work_ratio") is None else float(row["fine_work_ratio"]) for row in rows],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    colors = ["#56B4E9" if np.isfinite(value) else "#BDBDBD" for value in values]
    axis.bar(np.arange(len(labels)), np.nan_to_num(values, nan=0.0), color=colors)
    axis.axhline(1.0, linestyle="--", color="#555555", linewidth=1.0, label="dense rectangle = 1")
    tick_positions, tick_labels = _case_ticks(labels)
    axis.set_xticks(tick_positions, tick_labels, rotation=55)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("case ID")
    axis.set_ylabel("explicit fine rows / full rectangle rows")
    axis.set_title("Run 1503 fine work ratio\nmissing bars mean no explicit executor auxiliary was emitted")
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def render_empty_counts_figure(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _figure_setup(path)
    if plt is None:
        return
    labels = [str(row["case_id"]) for row in rows]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    width = 0.24
    for offset, key, label, color in (
        (-width, "empty_group_count", "groups", "#CC79A7"),
        (0.0, "empty_query_count", "queries", "#E69F00"),
        (width, "empty_source_count", "environment sources", "#009E73"),
    ):
        axis.bar(x + offset, [int(row.get(key, 0)) for row in rows], width, label=label, color=color)
    tick_positions, tick_labels = _case_ticks(labels)
    axis.set_xticks(tick_positions, tick_labels, rotation=55)
    axis.set_xlabel("case ID")
    axis.set_ylabel("explicit empty count")
    axis.set_title("Run 1503 empty groups / queries / sources")
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def render_candidate_figures(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, *, run_label: str = "Run 1503"
) -> dict[str, str]:
    """Reuse sparse-incidence boards and add candidate-specific distributions."""

    figure_dir = output_dir / "figures"
    array_dir = output_dir / "arrays"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    histogram_path = figure_dir / "kq_histogram.png"
    sparse_evidence.render_kplan_histogram(rows, histogram_path, run_label=run_label)
    if histogram_path.is_file():
        paths["kq_histogram"] = str(histogram_path)
    paths.update(sparse_evidence.render_population_figures(rows, figure_dir, run_label=run_label))

    extra = (
        ("source_degree", render_source_degree_figure),
        ("active_mass", render_active_mass_figure),
        ("p_alpha_opening_distributions", render_p_alpha_opening_figure),
        ("fine_work_ratio", render_fine_work_ratio_figure),
        ("empty_counts", render_empty_counts_figure),
    )
    for key, renderer in extra:
        path = figure_dir / f"{key}.png"
        if key in {"source_degree", "p_alpha_opening_distributions"}:
            renderer(rows, array_dir, path)
        else:
            renderer(rows, path)
        if path.is_file():
            paths[key] = str(path)
    return paths


CSV_FIELDS = (
    "case_id",
    "module_count",
    "query_count",
    "query_source_count",
    "registered_capacity",
    "group_count",
    "source_count",
    "active_source_count",
    "active_group_count",
    "active_mass_total",
    "empty_group_count",
    "empty_query_count",
    "empty_source_count",
    "empty_environment_source_count",
    "empty_module_source_count",
    "query_degree_mean",
    "module_source_degree_mean",
    "environment_source_degree_mean",
    "fine_work_ratio",
    "fine_work_ratio_source",
    "opening_status",
    "coarse_assignment_present",
    "executor_ledger_status",
    "p_distribution",
    "alpha_distribution",
    "opening_distribution",
)


def write_case_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            geometry = row.get("run1503_diagnostic", {}).get("geometry", {})
            opening = row.get("run1503_diagnostic", {}).get("opening", {})
            ledger = row.get("run1503_diagnostic", {}).get("executor_ledger", {})
            output: dict[str, Any] = {}
            for field in CSV_FIELDS:
                value = row.get(field)
                if field == "registered_capacity":
                    value = row.get("registered_capacity", row.get("kmax"))
                elif field == "group_count":
                    value = geometry.get("group_count", row.get("kmax"))
                elif field == "opening_status":
                    value = opening.get("opening_status")
                elif field == "coarse_assignment_present":
                    value = opening.get("coarse_assignment_present")
                elif field == "executor_ledger_status":
                    value = ledger.get("status")
                elif field.endswith("_distribution"):
                    value = json.dumps(_jsonable(row.get(field)), sort_keys=True)
                output[field] = _jsonable(value)
            writer.writerow(output)


def _runtime_imports() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model
    from channelthermal.evaluation.prepared import predict_case
    from run_run1409_occupancy_population import _dataset_for_checkpoint, _query_sample

    return (
        torch,
        GlobalChannelThermalDataset,
        H5Normalizer,
        load_model,
        predict_case,
        _dataset_for_checkpoint,
        _query_sample,
        sparse_evidence,
    )


def _case_selection(args: argparse.Namespace, available: Sequence[str]) -> tuple[list[str], int]:
    available_set = {str(value) for value in available}
    if args.case_id:
        case_ids = [str(value) for value in args.case_id]
    elif args.population:
        case_ids = [str(value) for value in available]
    else:
        case_ids = list(DEFAULT_CASE_IDS)
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise Run1503OrganizationEvidenceError("case IDs must be non-empty and unique")
    missing = [case_id for case_id in case_ids if case_id not in available_set]
    if missing:
        raise KeyError(f"case IDs are absent from split {args.split!r}: {missing}")
    expected = int(args.expected_cases)
    if expected < 0:
        raise ValueError("--expected-cases must be non-negative")
    if expected == 0:
        expected = EXPECTED_CASE_COUNT if args.population and not args.case_id else len(case_ids)
    if expected > 0 and len(case_ids) != expected:
        raise Run1503OrganizationEvidenceError(
            f"expected {expected} explicit cases, selected {len(case_ids)}; pass --expected-cases 0 for a bounded subset"
        )
    return case_ids, expected


def run_population(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the explicit candidate checkpoint on CPU and write evidence."""

    if str(args.device) != "cpu":
        raise ValueError("Run1503 organization evidence is CPU-only; pass --device cpu")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"explicit candidate checkpoint does not exist: {checkpoint_path}")
    if int(args.query_count) <= 0 or int(args.query_batch_size) <= 0:
        raise ValueError("--query-count and --query-batch-size must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        torch,
        GlobalChannelThermalDataset,
        H5Normalizer,
        load_model,
        predict_case,
        dataset_for_checkpoint,
        query_sample,
        _evidence,
    ) = _runtime_imports()
    device = torch.device("cpu")
    model, checkpoint = load_model(checkpoint_path, device)
    architecture = _model_architecture(model)
    if architecture != EXPECTED_ARCHITECTURE:
        raise Run1503OrganizationEvidenceError(
            f"explicit candidate architecture is {architecture!r}, expected {EXPECTED_ARCHITECTURE!r}"
        )
    dataset, dataset_path, dataset_config = dataset_for_checkpoint(
        checkpoint,
        dataset_path=args.dataset,
        split=args.split,
        GlobalChannelThermalDataset=GlobalChannelThermalDataset,
        H5Normalizer=H5Normalizer,
    )
    available = [str(value) for value in dataset.selected_case_ids]
    case_ids, expected_cases = _case_selection(args, available)
    index_by_case = {case_id: index for index, case_id in enumerate(available)}
    module_counts = {
        str(case_id): int(dataset.selected_module_counts[index])
        for index, case_id in enumerate(available)
        if hasattr(dataset, "selected_module_counts")
    }

    array_dir = output_dir / "arrays"
    rows: list[dict[str, Any]] = []
    arrays: dict[str, str] = {}
    try:
        for case_id in case_ids:
            sample = dataset[index_by_case[case_id]]
            selected, query_xy = query_sample(sample, int(args.query_count))
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
            payload = _prepared_payload(selected, prediction, query_xy)
            record = canonicalize_candidate_case(
                payload,
                case_id=case_id,
                module_count=module_counts.get(case_id),
                query_count=int(query_xy.shape[0]),
                kmax=int(args.kmax),
            )
            row = record["case"]
            row["prediction_finite"] = (
                None
                if prediction.get("pred_field_grid") is None
                else bool(np.isfinite(np.asarray(prediction["pred_field_grid"])).all())
            )
            rows.append(row)
            array_path = array_dir / f"{case_id}.npz"
            sparse_evidence.save_case_arrays(array_path, record["maps"])
            arrays[case_id] = str(array_path)
            del prediction, payload, record, selected, sample
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        del model, dataset
        gc.collect()

    summary = summarize_candidate_population(rows, expected_cases=expected_cases)
    figures = render_candidate_figures(rows, output_dir)
    csv_path = output_dir / "population_cases.csv"
    write_case_csv(csv_path, rows)
    output = {
        "schema_version": 1,
        "task": "run1503_candidate_organization_evidence",
        "status": "complete",
        "output_dir": str(output_dir),
        "candidate": {
            "architecture": architecture,
            "checkpoint": str(checkpoint_path),
            "checkpoint_selection": "explicit_path_only",
            "dataset": str(dataset_path),
            "dataset_config_split": dataset_config.get("test_split", dataset_config.get("train_split")),
            "split": str(args.split),
            "device": "cpu",
            "training_launched": False,
            "checkpoint_written": False,
            "gpu_process_inspection": False,
        },
        "protocol": {
            "case_ids": case_ids,
            "representative_case_ids": list(DEFAULT_CASE_IDS),
            "population_requested": bool(args.population),
            "query_count_requested": int(args.query_count),
            "query_batch_size": int(args.query_batch_size),
            "kmax": int(args.kmax),
            "local_port_condition_mode": "predicted",
            "mixed_teacher_ratio": 0.0,
            "device": "cpu",
            "actual_row_policy": "explicit Run1503 executor auxiliary or phase ledger only",
        },
        "population": summary,
        "cases": rows,
        "arrays": arrays,
        "csv": str(csv_path),
        "figures": figures,
        "limitations": [
            "Group labels are permutation-ambiguous learned organization labels, not physical causality.",
            "Kq, p, alpha, opening, source degree, and active mass are structural organization evidence.",
            "Fine work ratio is unavailable unless the candidate emits explicit fine/full executor counters.",
            "No Run1501/Run1502 checkpoint is selected, read, or substituted by this adapter.",
        ],
    }
    _write_json(output_dir / "evidence.json", output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Explicit Run1503 candidate checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default=None, help="Optional explicit HDF5 dataset path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=[], help="Bounded explicit case subset; repeat this option")
    parser.add_argument("--population", action="store_true", help="Evaluate all cases in the selected split (expected 90)")
    parser.add_argument("--expected-cases", type=int, default=0, help="Expected selected case count; 0 derives panel/population policy")
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    parser.add_argument("--kmax", type=int, default=DEFAULT_KMAX)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    output = run_population(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "status": output["status"],
                "output_dir": output["output_dir"],
                "case_count": output["population"]["case_count"],
                "kq_histogram": output["population"]["kq_histogram"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CASE_IDS",
    "DEFAULT_KMAX",
    "DEFAULT_QUERY_COUNT",
    "EXPECTED_ARCHITECTURE",
    "EXPECTED_CASE_COUNT",
    "Run1503OrganizationEvidenceError",
    "build_parser",
    "canonicalize_candidate_case",
    "main",
    "render_candidate_figures",
    "run_population",
    "summarize_candidate_population",
    "write_case_csv",
]
