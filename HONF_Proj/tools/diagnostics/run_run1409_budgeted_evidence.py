"""Bounded evidence driver for the opt-in Run-1409 budgeted reader.

The driver measures one explicitly supplied candidate checkpoint at exact
review epoch 50, 150, or 500.  It keeps
the following quantities in separate fields:

* registered capacity (``Kmax``), packed width, sampled live groups, and
  query/group support;
* logical q -> group -> source paths, unique q -> source support, and actual
  fine rows reported by the backend ledger; and
* full-width versus compact-column execution, stochastic gate output, and the
  deterministic hard-concrete estimate.

Timed calls do not request routing maps.  The map/ledger call is one separate
untimed forward per case and mode.  No run directory is allocated and no
checkpoint is modified.  The default output is intended for a managed
evaluation directory supplied by the caller, for example::

    PYTHONPATH=src:Case_ThermalChannel/src \
      python tools/diagnostics/run_run1409_budgeted_evidence.py --plan-only \
      --checkpoint /abs/path/Run_1409_.../epoch_0050_model.pt \
      --output /abs/path/Run_1409_.../evaluations/budgeted_router/evidence.json

Physical evidence requires an explicit ``--checkpoint`` and ``--dataset`` is
optional when the checkpoint records the packed HDF5 path.  The script uses
the maintained Stage-3 loader and ThermalChannel adapter; it does not infer a
checkpoint from a run number.  This is a diagnostic/benchmark artifact, not a
training command or a claim of optimal physical rank.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import inspect
import json
import math
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE = "budgeted_group_control_honf"
GROUP_COUNT = 12
CONTROL_DIM = 16
REVIEW_EPOCHS = (50, 150, 500)
DEFAULT_CASE_IDS = ("0273", "0653")
DEFAULT_QUERY_COUNT = 8192
DEFAULT_RECEIVER_CHUNK_SIZE = 2048
DEFAULT_WARMUPS = 2
DEFAULT_REPETITIONS = 5
DEFAULT_GATE_REPETITIONS = 3


def _finite(value: Any) -> float | None:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1)[0].item()
        value = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError, RuntimeError):
        return None
    return value if math.isfinite(value) else None


def _jsonable(value: Any) -> Any:
    """Convert small tensors completely and large tensors to bounded stats."""

    if hasattr(value, "detach") and hasattr(value, "numel"):
        value = value.detach().cpu()
        if int(value.numel()) == 1:
            return _jsonable(value.item())
        if int(value.numel()) <= 256:
            return value.tolist()
        flat = value.reshape(-1).float()
        return {
            "tensor": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": _finite(flat.min()),
            "max": _finite(flat.max()),
            "mean": _finite(flat.mean()),
            "numel": int(value.numel()),
        }
    if isinstance(value, np.ndarray):
        if value.size <= 256:
            return value.tolist()
        flat = value.reshape(-1).astype(np.float64, copy=False)
        finite = flat[np.isfinite(flat)]
        return {
            "array": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": None if finite.size == 0 else float(np.min(finite)),
            "max": None if finite.size == 0 else float(np.max(finite)),
            "mean": None if finite.size == 0 else float(np.mean(finite)),
            "numel": int(value.size),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_checkpoint(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return path


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.query_count) <= 0 or int(args.receiver_chunk_size) <= 0:
        raise ValueError("query-count and receiver-chunk-size must be positive")
    if int(args.warmups) < 0 or int(args.repetitions) <= 0:
        raise ValueError("warmups must be nonnegative and repetitions must be positive")
    if int(args.gate_repetitions) <= 0:
        raise ValueError("gate-repetitions must be positive")
    modes = _modes(args)
    if not modes or any(value not in {"full_width", "compact"} for value in modes):
        raise ValueError("mode must contain only full_width and/or compact")
    if len(set(modes)) != len(modes):
        raise ValueError("mode values must be unique")
    cases = tuple(str(value) for value in (args.case_id or DEFAULT_CASE_IDS))
    if not cases:
        raise ValueError("at least one case-id is required")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Validate paths and emit a no-runtime plan."""

    _validate_args(args)
    checkpoint = _parse_checkpoint(args.checkpoint)
    output = Path(args.output).expanduser().resolve()
    array_root = Path(args.array_dir or output.parent / "arrays").expanduser().resolve()
    return {
        "schema_version": 1,
        "task": "run1409_budgeted_group_control_evidence",
        "status": "plan_only",
        "candidate": {
            "architecture": ARCHITECTURE,
            "checkpoint": str(checkpoint),
            "selection_policy": "explicit_cli_checkpoint",
            "required_group_capacity": GROUP_COUNT,
            "required_control_dimension": CONTROL_DIM,
        },
        "protocol": {
            "case_ids": [str(value) for value in (args.case_id or DEFAULT_CASE_IDS)],
            "query_count": int(args.query_count),
            "receiver_chunk_size": int(args.receiver_chunk_size),
            "warmups": int(args.warmups),
            "repetitions": int(args.repetitions),
            "gate_repetitions": int(args.gate_repetitions),
            "modes": list(_modes(args)),
            "timed_maps": False,
            "map_pass": "one untimed map/ledger forward per case and execution mode",
            "deterministic_policy": "model.eval() uses the deployed deterministic hard-concrete estimate; stochastic draws use a separate gate-only override",
        },
        "planned_outputs": {
            "evidence_json": str(output),
            "array_directory": str(array_root),
        },
        "ledger_contract": {
            "capacity": "registered Kmax=12",
            "packed_width": "shared compact-column width from the P0 gate plan",
            "live_count": "positive hard-concrete availability slots, including ordinary group 0",
            "logical_support": "positive q -> group -> source paths",
            "unique_pairs": "positive q -> source pairs after group contraction",
            "actual_fine_rows": "backend-reported rows executed by the selected reader",
            "rank": "routing/group-state numerical rank only; no physical-rank claim",
        },
        "limitations": [
            "No run allocation, training, checkpoint write, or CUDA initialization occurs in plan-only mode.",
            "A missing phase ledger or deterministic gate hook is retained as unavailable.",
            "Logical support and actual execution are never reconstructed from one another.",
            "Selected fine reads are reported only when the backend exposes an explicit executor policy.",
        ],
    }


def _runtime_imports() -> tuple[Any, Any, Any, Any, Any]:
    """Import maintained runtime modules only for a physical invocation."""

    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    import run_dynamic_sparse_routing_study as dynamic
    import run_run1405_epoch50_comparison as run1405
    import run_stage3_interface_study as stage3
    from channelthermal.evaluation.loading import make_batch
    from channelthermal.evaluation.prepared import select_sample

    return torch, dynamic, run1405, stage3, (make_batch, select_sample)


def _sync(device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _measure(
    function: Callable[[], Any],
    device: Any,
    *,
    warmups: int,
    repetitions: int,
    inference: bool = True,
) -> dict[str, Any]:
    """Measure synchronized elapsed time and incremental CUDA peaks."""

    import torch

    context = torch.inference_mode if inference else contextlib.nullcontext
    with context():
        for _ in range(int(warmups)):
            result = function()
            del result
        _sync(device)
    samples: list[dict[str, Any]] = []
    for index in range(int(repetitions)):
        if getattr(device, "type", None) == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
        else:
            baseline_allocated = baseline_reserved = None
        _sync(device)
        started = time.perf_counter()
        result = None
        error = None
        try:
            with context():
                result = function()
        except Exception as exc:  # noqa: BLE001 - preserve evidence failures.
            error = f"{type(exc).__name__}: {exc}"
        _sync(device)
        elapsed = time.perf_counter() - started
        if getattr(device, "type", None) == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = peak_reserved = None
        samples.append(
            {
                "repetition": index + 1,
                "elapsed_seconds": float(elapsed),
                "status": "error" if error else "complete",
                "error": error,
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "incremental_peak_allocated_bytes": (
                    None
                    if peak_allocated is None or baseline_allocated is None
                    else peak_allocated - baseline_allocated
                ),
                "incremental_peak_reserved_bytes": (
                    None
                    if peak_reserved is None or baseline_reserved is None
                    else peak_reserved - baseline_reserved
                ),
            }
        )
        del result
        if error:
            break
    complete = [row for row in samples if row["status"] == "complete"]
    elapsed = [float(row["elapsed_seconds"]) for row in complete]
    return {
        "status": "complete" if len(complete) == int(repetitions) else "incomplete",
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "samples": samples,
        "median_seconds": None if not elapsed else float(np.median(elapsed)),
        "min_seconds": None if not elapsed else float(np.min(elapsed)),
        "max_seconds": None if not elapsed else float(np.max(elapsed)),
        "spread_seconds": None if not elapsed else float(np.max(elapsed) - np.min(elapsed)),
        "peak_allocated_bytes": max(
            (row["peak_allocated_bytes"] for row in samples if row["peak_allocated_bytes"] is not None),
            default=None,
        ),
        "peak_reserved_bytes": max(
            (row["peak_reserved_bytes"] for row in samples if row["peak_reserved_bytes"] is not None),
            default=None,
        ),
    }


def _aux_mapping(output: Any) -> Mapping[str, Any]:
    """Find interaction diagnostics in either wrapper or prepared output."""

    if isinstance(output, Mapping):
        for key in ("interaction_aux", "_interaction_aux"):
            value = output.get(key)
            if isinstance(value, Mapping):
                return value
        prepared = output.get("prepared_state")
    else:
        prepared = output
    inner = getattr(prepared, "prepared", prepared)
    value = getattr(inner, "interaction_aux", None)
    if isinstance(value, Mapping):
        return value
    if isinstance(inner, Mapping):
        for key in ("interaction_aux", "_interaction_aux"):
            value = inner.get(key)
            if isinstance(value, Mapping):
                return value
    return {}


def _all_containers(output: Any) -> list[Any]:
    """Return shallow output/prepared/backend containers for gate lookup."""

    values: list[Any] = []
    queue: list[Any] = [output]
    seen: set[int] = set()
    for _ in range(8):
        if not queue:
            break
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        values.append(current)
        if isinstance(current, Mapping):
            for key in ("prepared_state", "prepared", "backend_state", "case_group_budget", "interaction_aux"):
                if key in current and current[key] is not None:
                    queue.append(current[key])
        else:
            for key in ("prepared", "backend_state", "case_group_budget", "phase_shared_state", "interaction_aux"):
                child = getattr(current, key, None)
                if child is not None:
                    queue.append(child)
    return values


def _lookup(output: Any, names: Sequence[str]) -> Any:
    for container in _all_containers(output):
        if isinstance(container, Mapping):
            for name in names:
                if name in container:
                    return container[name]
        else:
            for name in names:
                value = getattr(container, name, None)
                if value is not None:
                    return value
    return None


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
    except (TypeError, ValueError, RuntimeError):
        return None
    if value.dtype == object:
        return None
    return value


def _tensor_summary(value: Any, *, keep_values: bool = True) -> dict[str, Any] | None:
    array = _array(value)
    if array is None:
        return None
    finite = np.asarray(array, dtype=np.float64)[np.isfinite(array)]
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "numel": int(array.size),
        "min": None if finite.size == 0 else float(np.min(finite)),
        "max": None if finite.size == 0 else float(np.max(finite)),
        "mean": None if finite.size == 0 else float(np.mean(finite)),
    }
    if keep_values and array.size <= 256:
        result["values"] = array.tolist()
    return result


def _scalar(value: Any) -> float | None:
    array = _array(value)
    if array is None or array.size != 1:
        return None
    try:
        item = float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    return item if math.isfinite(item) else None


def _phase_key(prefix: str, source: str, suffixes: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"{prefix}{root}{source}_{suffix}"
        for root in ("", "group_control_", "case_group_budget_")
        for suffix in suffixes
    )


def _find_aux(aux: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in aux:
            return aux[key]
    return None


def _modes(args: argparse.Namespace) -> tuple[str, ...]:
    values = args.mode or ("full_width", "compact")
    return tuple(str(value) for value in values)


def _phase_ledger(aux: Mapping[str, Any], backend: Any | None) -> dict[str, Any]:
    """Collect phase/source ledgers without deriving execution from support."""

    prefixes = {
        "P0": ("initial_port_", "p0_"),
        "P1": ("provisional_", "p1_"),
        "P2": ("", "p2_"),
        "P2_consistency": ("port_global_", "p2_consistency_"),
    }
    aliases = {
        "logical_path_count": ("logical_paths", "logical_path_count"),
        "unique_pair_count": ("unique_pairs", "unique_pair_count"),
        "logical_support_rows": ("support_rows",),
        "actual_fine_call_count": ("fine_rows_forward", "fine_forward_rows", "fine_rows"),
        "padded_fine_rows": ("fine_rows_padded", "padded_rows"),
        "valid_pair_denominator": ("valid_pair_denominator",),
        "padded_pair_denominator": ("padded_pair_denominator",),
        "geometry_rows": ("geometry_rows_forward", "geometry_rows"),
        "content_dot_rows": ("content_dot_rows_forward", "content_dot_rows"),
        "scalar_control_rows": ("scalar_control_rows",),
        "source_projection_rows": ("source_projection_rows",),
        "checkpoint_recompute_count": ("checkpoint_recomputations", "fine_rows_recompute"),
        "executor_selected": ("executor_selected",),
        "complete_support": ("complete_support",),
    }
    phases: dict[str, Any] = {}
    for phase, phase_prefixes in prefixes.items():
        phase_row: dict[str, Any] = {
            "phase": phase,
            "status": "unavailable",
            "sources": {},
            "policy": getattr(backend, "executor_policy", None),
            "rectangular_ledger_rows": getattr(backend, "ledger_rectangular_rows", None),
        }
        any_value = False
        for source in ("module", "environment"):
            source_row: dict[str, Any] = {}
            for metric, suffixes in aliases.items():
                value = None
                for prefix in phase_prefixes:
                    value = _find_aux(aux, _phase_key(prefix, source, suffixes))
                    if value is not None:
                        break
                scalar = _scalar(value)
                if scalar is not None:
                    source_row[metric] = scalar
                    any_value = True
            if source_row:
                phase_row["sources"][source] = source_row
        if any_value:
            phase_row["status"] = "explicit_backend_ledger"
        phases[phase] = phase_row
    return phases


def _weighted_operator_rank(
    aux: Mapping[str, Any],
    *,
    source: str,
    relative_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Measure the weighted low-dimensional routing operator without Q×S.

    For ``alpha=[B,Q,K]`` and source memberships ``A=[B,S,K]`` this computes
    the singular values of ``diag(sqrt(wq)) alpha A^T
    diag(sqrt(omega))`` through the thin-QR core ``R_alpha R_A^T``.  It never
    forms the full query/source matrix.  The result is a learned routing
    operator rank and is explicitly not a physical-rank guarantee.
    """

    alpha = _array(_find_aux(aux, ("group_control_query_routing", "group_control_query_alpha")))
    membership = _array(
        _find_aux(
            aux,
            (f"group_control_{source}_incidence", f"group_control_{source}_membership"),
        )
    )
    measure = _array(_find_aux(aux, (f"group_control_{source}_measure", f"{source}_measure")))
    if alpha is None or membership is None:
        return {
            "status": "unavailable",
            "reason": "query routing or source incidence map missing",
            "interpretation": "weighted learned routing rank; not physical operator rank",
        }
    if alpha.ndim == 2:
        alpha = alpha[None, ...]
    if membership.ndim == 2:
        membership = membership[None, ...]
    if alpha.ndim != 3 or membership.ndim != 3 or alpha.shape[0] != membership.shape[0]:
        return {
            "status": "unavailable",
            "reason": f"unexpected alpha/A shapes: {alpha.shape}/{membership.shape}",
            "interpretation": "weighted learned routing rank; not physical operator rank",
        }
    batch, query_count, capacity = alpha.shape
    if membership.shape[-1] != capacity:
        return {
            "status": "unavailable",
            "reason": f"group axis mismatch: alpha={alpha.shape}, A={membership.shape}",
            "interpretation": "weighted learned routing rank; not physical operator rank",
        }
    if measure is None:
        return {
            "status": "unavailable",
            "reason": "backend source measure missing; a weighted rank cannot be inferred",
            "interpretation": "weighted learned routing rank; not physical operator rank",
        }
    source_weights = np.asarray(measure, dtype=np.float64)
    if source_weights.ndim == 1:
        source_weights = source_weights[None, ...]
    if source_weights.shape != membership.shape[:2]:
        return {
            "status": "unavailable",
            "reason": f"source measure shape {source_weights.shape} does not match A={membership.shape[:2]}",
            "interpretation": "weighted learned routing rank; not physical operator rank",
        }
    source_weight_policy = "backend source measure"
    ranks: list[int] = []
    effective: list[float] = []
    spectra: list[list[float]] = []
    for index in range(batch):
        # Query probes carry equal measure in the fixed-grid evaluator.
        query_weights = np.full(query_count, 1.0 / float(query_count), dtype=np.float64)
        source_weights_case = np.maximum(source_weights[index], 0.0)
        U = np.sqrt(query_weights)[:, None] * np.asarray(alpha[index], dtype=np.float64)
        V = np.sqrt(source_weights_case)[:, None] * np.asarray(membership[index], dtype=np.float64)
        # Reduced QR produces at most K columns/rows.  The product below is
        # <= K×K, independent of the full Q×S physical rectangle.
        _q_u, r_u = np.linalg.qr(U, mode="reduced")
        _q_v, r_v = np.linalg.qr(V, mode="reduced")
        core = r_u @ r_v.T
        singular = np.linalg.svd(core, compute_uv=False)
        scale = float(np.max(singular)) if singular.size else 0.0
        tolerance = float(relative_tolerance) * scale
        rank = int(np.count_nonzero(singular > tolerance)) if scale > 0.0 else 0
        # Use the same relative tolerance for entropy support as for rank;
        # singular values below it are numerical noise for this audit.
        positive = singular[singular > tolerance]
        if positive.size:
            probabilities = positive / float(np.sum(positive))
            entropy_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        else:
            entropy_rank = 0.0
        ranks.append(rank)
        effective.append(entropy_rank)
        spectra.append([float(value) for value in singular])
    return {
        "status": "complete",
        "source": source,
        "alpha_shape": list(alpha.shape),
        "incidence_shape": list(membership.shape),
        "capacity_bound": int(capacity),
        "rank_min": int(min(ranks)),
        "rank_max": int(max(ranks)),
        "rank_mean": float(np.mean(ranks)),
        "entropy_effective_rank_min": float(min(effective)),
        "entropy_effective_rank_max": float(max(effective)),
        "entropy_effective_rank_mean": float(np.mean(effective)),
        "singular_values": spectra,
        "relative_tolerance": float(relative_tolerance),
        "entropy_support_rule": "singular values strictly above relative_tolerance * max_singular_value",
        "query_weight_policy": "uniform over fixed evaluator probes",
        "source_weight_policy": source_weight_policy,
        "interpretation": "weighted learned q-to-source routing rank; not physical operator rank or optimal rank",
    }


def _rank_summary(aux: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "interpretation": "weighted learned routing rank; not physical operator rank",
        "module": _weighted_operator_rank(aux, source="module"),
        "environment": _weighted_operator_rank(aux, source="environment"),
    }


def _budget_summary(output: Any, aux: Mapping[str, Any]) -> dict[str, Any]:
    names = {
        "sampled_live_count": (
            "case_group_budget_sampled_live_count",
            "sampled_live_count",
            "live_count",
        ),
        "packed_width": (
            "case_group_budget_packed_width",
            "packed_width",
        ),
        "kappa": ("case_group_budget_kappa", "kappa"),
        "gate_reused": ("case_group_budget_gate_reused", "gate_reused"),
        "fine_source_refresh": (
            "case_group_budget_fine_source_refresh",
            "fine_source_refresh",
        ),
        "expected_optional_count": (
            "case_group_budget_expected_optional_count",
            "expected_optional_count",
            "expected_optional_group_count",
        ),
        "expected_count": (
            "case_group_budget_expected_count",
            "expected_count",
            "expected_live_count",
            "expected_group_count",
        ),
        "raw_z": (
            "case_group_budget_raw_z",
            "raw_z",
            "z_raw",
            "sampled_z",
        ),
        "effective_z": (
            "case_group_budget_effective_z",
            "effective_z",
            "z_eff",
            "z",
        ),
        "raw_support": (
            "case_group_budget_raw_support",
            "raw_support",
            "support_raw",
        ),
        "executed_support": (
            "case_group_budget_executed_support",
            "effective_support",
            "case_group_budget_support",
            "support",
        ),
        "positive_probability": (
            "case_group_budget_positive_probability",
            "case_group_budget_gate_positive_probability",
            "positive_probability",
            "gate_probability",
        ),
        "fallback_used": (
            "case_group_budget_fallback_used",
            "case_group_budget_fallback",
            "fallback_used",
            "fallback",
        ),
        "prototype_ids": (
            "case_group_budget_packed_prototype_ids",
            "case_group_budget_prototype_ids",
            "packed_ids",
            "prototype_ids",
        ),
        "continuation_c": (
            "case_group_budget_continuation",
            "continuation_c",
            "sparsification_c",
        ),
        "routing_strength": (
            "case_group_budget_routing_strength",
            "routing_strength",
            "route_strength",
        ),
        "z": ("case_group_budget_z", "effective_z", "z_eff", "z"),
        "support": (
            "case_group_budget_support",
            "executed_support",
            "effective_support",
            "support",
        ),
        "eta": ("case_group_budget_eta", "eta"),
    }
    result: dict[str, Any] = {"capacity": GROUP_COUNT}
    for label, aliases in names.items():
        value = _lookup(output, aliases)
        if value is None:
            value = _find_aux(aux, aliases)
        summary = _tensor_summary(value)
        if summary is not None:
            result[label] = summary
    support_value = _lookup(output, names["support"])
    if support_value is None:
        support_value = _find_aux(aux, names["support"])
    support = _array(support_value)
    if support is not None:
        result["support_positive_counts"] = np.sum(support > 0, axis=-1).astype(int).tolist()
    for label, aliases in (
        ("K_raw", names["raw_support"]),
        ("K_executed", names["executed_support"]),
    ):
        value = _lookup(output, aliases)
        if value is None:
            value = _find_aux(aux, aliases)
        array = _array(value)
        if array is not None:
            if array.ndim == 1:
                array = array[None, ...]
            result[label] = np.sum(array > 0, axis=-1).astype(int).tolist()
    live_value = _lookup(output, names["sampled_live_count"])
    if live_value is None:
        live_value = _find_aux(aux, names["sampled_live_count"])
    live_summary = _tensor_summary(live_value)
    if live_summary is not None:
        result["Klive"] = live_summary
    expected_value = _lookup(output, names["expected_count"])
    if expected_value is None:
        expected_value = _find_aux(aux, names["expected_count"])
    if expected_value is None:
        optional_value = _lookup(output, names["expected_optional_count"])
        if optional_value is None:
            optional_value = _find_aux(aux, names["expected_optional_count"])
        optional_scalar = _scalar(optional_value)
        if optional_scalar is not None:
            result["K_expected"] = 1.0 + optional_scalar
    else:
        expected_summary = _tensor_summary(expected_value)
        if expected_summary is not None:
            result["K_expected"] = expected_summary
    ids_value = _lookup(output, names["prototype_ids"])
    if ids_value is None:
        ids_value = _find_aux(aux, names["prototype_ids"])
    ids_array = _array(ids_value)
    if ids_array is not None and ids_array.size <= 256:
        result["prototype_ids"] = ids_array.tolist()
    # Source nonempty counts describe logical group support, not physical rank.
    for label, aliases in (
        ("module_nonempty_group_count", ("group_control_module_incidence",)),
        ("environment_nonempty_group_count", ("group_control_environment_incidence",)),
    ):
        array = _array(_find_aux(aux, aliases))
        if array is not None and array.ndim >= 2:
            result[label] = np.sum(np.any(array > 0.0, axis=-2), axis=-1).astype(int).tolist()
    result.update(_rank_summary(aux))
    return result


def _prediction_difference(lhs: Any, rhs: Any) -> dict[str, float] | None:
    if lhs is None or rhs is None:
        return None
    try:
        left = lhs.detach().float()
        right = rhs.detach().float()
        delta = (left - right).abs()
        denominator = right.abs().mean().clamp_min(1.0e-12)
        return {
            "absolute_mean": float(delta.mean().cpu()),
            "absolute_max": float(delta.max().cpu()),
            "relative_to_reference_mean": float((delta.mean() / denominator).cpu()),
        }
    except (AttributeError, RuntimeError, ValueError):
        return None


def _detach_cpu(value: Any) -> Any:
    """Drop a diagnostic tensor's device storage after its scalar use."""

    if hasattr(value, "detach"):
        return value.detach().cpu()
    return value


def _rng_state(torch: Any, device: Any) -> tuple[Any, list[Any] | None]:
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if getattr(device, "type", None) == "cuda" else None
    return cpu, cuda


def _restore_rng(torch: Any, device: Any, state: tuple[Any, list[Any] | None]) -> None:
    cpu, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


@contextlib.contextmanager
def _execution_mode(model: Any, mode: str) -> Iterator[None]:
    backend = getattr(getattr(model, "core", None), "backend", None)
    if backend is None:
        raise RuntimeError("model has no interface backend")
    setter = getattr(backend, "set_execution_mode", None)
    previous = getattr(backend, "execution_mode", None)
    if callable(setter):
        setter(str(mode))
    elif previous is not None:
        setattr(backend, "execution_mode", str(mode))
    else:
        raise RuntimeError("budgeted backend does not expose set_execution_mode/execution_mode")
    try:
        yield
    finally:
        if previous is not None:
            if callable(setter):
                setter(str(previous))
            else:
                setattr(backend, "execution_mode", previous)


@contextlib.contextmanager
def _stochastic_gate_mode(model: Any) -> Iterator[dict[str, Any]]:
    """Patch only gate sampling while preserving the deployed eval model."""

    backend = getattr(getattr(model, "core", None), "backend", None)
    router = getattr(backend, "router", None)
    if router is None or not callable(getattr(router, "prepare", None)):
        yield {"status": "unavailable", "reason": "budgeted backend router.prepare hook is absent"}
        return
    setter = getattr(backend, "set_gate_sampling_mode", None)
    if callable(setter):
        previous_mode = getattr(backend, "gate_sampling_mode", None)
        setter("stochastic")
        try:
            yield {"status": "patched", "hook": "backend.set_gate_sampling_mode('stochastic')"}
        finally:
            setter(previous_mode if previous_mode is not None else "deployed")
        return
    if hasattr(backend, "gate_sampling_mode"):
        previous_mode = getattr(backend, "gate_sampling_mode")
        setattr(backend, "gate_sampling_mode", "stochastic")
        try:
            yield {"status": "patched", "hook": "backend.gate_sampling_mode='stochastic'"}
        finally:
            setattr(backend, "gate_sampling_mode", previous_mode)
        return
    original_prepare = router.prepare
    original_instance = router.__dict__.get("prepare", None) if hasattr(router, "__dict__") else None
    patched_names: list[str] = []
    previous_values: dict[str, Any] = {}
    for name in ("deterministic_gates", "use_deterministic_gates", "gate_sampling_mode"):
        if hasattr(backend, name):
            previous_values[name] = getattr(backend, name)
            setattr(backend, name, True if name != "gate_sampling_mode" else "deterministic")
            patched_names.append(name)

    def prepare_with_deterministic(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("budget") is None:
            kwargs["deterministic_gates"] = False
        return original_prepare(*args, **kwargs)

    setattr(router, "prepare", prepare_with_deterministic)
    try:
        yield {
            "status": "patched",
            "hook": "router.prepare(deterministic_gates=False) on P0; shared budget remains in later phases",
            "backend_flags": patched_names,
        }
    finally:
        if original_instance is None:
            try:
                delattr(router, "prepare")
            except AttributeError:
                pass
        else:
            setattr(router, "prepare", original_instance)
        for name, value in previous_values.items():
            setattr(backend, name, value)


def _set_mode_once(model: Any, mode: str) -> None:
    backend = model.core.backend
    setter = getattr(backend, "set_execution_mode", None)
    if callable(setter):
        setter(str(mode))
    elif hasattr(backend, "execution_mode"):
        backend.execution_mode = str(mode)
    else:
        raise RuntimeError("budgeted backend does not expose execution mode")


def _case_forward(
    model: Any,
    batch: Mapping[str, Any],
    query: Any,
    kwargs: Mapping[str, Any],
    dynamic: Any,
    receiver_chunk: int,
    *,
    return_prepared_state: bool = False,
    return_routing_maps: bool = False,
) -> Any:
    with dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
        return model(
            batch["structure"],
            query,
            return_prepared_state=bool(return_prepared_state),
            return_routing_maps=bool(return_routing_maps),
            **kwargs,
        )


def _save_debug_arrays(
    path: Path,
    aux: Mapping[str, Any],
    *,
    output: Any,
    query_xy: Any,
    phase_ledger: Mapping[str, Any],
    budget: Mapping[str, Any],
    mode: str,
    case_id: str,
    prepared_decode_median_ms: float | None,
) -> dict[str, Any]:
    """Save the canonical maintained interaction-board NPZ plus gate IDs."""

    arrays: dict[str, np.ndarray] = {}
    for key, value in aux.items():
        text = str(key)
        if not any(token in text for token in ("group_control", "case_group_budget")):
            continue
        array = _array(value)
        if array is None or array.size > 20_000_000:
            continue
        safe = "".join(character if character.isalnum() or character == "_" else "_" for character in text)
        arrays[safe] = array
    # Reuse the maintained Run-1406 board file contract.  The candidate adds
    # gate arrays and their original prototype IDs as optional NPZ members;
    # historical boards continue to load because those members are ignored.
    try:
        import group_control_evidence as group_control

        prepared_state = output.get("prepared_state") if isinstance(output, Mapping) else None
        prepared = getattr(prepared_state, "prepared", None)
        encoded = getattr(prepared, "encoded", None)
        module_coords = _array(getattr(encoded, "module_centers", None))
        module_present = _array(getattr(encoded, "module_present", None))
        env_coords = _array(getattr(encoded, "env_coords", None))
        env_weights = _array(getattr(encoded, "env_weights", None))
        module_incidence = _array(_find_aux(aux, ("group_control_module_incidence",)))
        environment_incidence = _array(_find_aux(aux, ("group_control_environment_incidence",)))
        query_routing = _array(_find_aux(aux, ("group_control_query_routing", "group_control_query_alpha")))
        if any(value is None for value in (module_coords, module_present, env_coords, env_weights, module_incidence, environment_incidence, query_routing)):
            return {"status": "unavailable", "reason": "canonical board geometry or group maps were not returned"}

        def squeeze(value: np.ndarray) -> np.ndarray:
            return value[0] if value.ndim > 0 and value.shape[0] == 1 else value

        module_coords = squeeze(module_coords)
        module_present = squeeze(module_present)
        env_coords = squeeze(env_coords)
        env_weights = squeeze(env_weights)
        module_incidence = squeeze(module_incidence)
        environment_incidence = squeeze(environment_incidence)
        query_routing = squeeze(query_routing)
        query_array = _array(query_xy)
        if query_array is None:
            raise ValueError("query coordinates are unavailable")
        query_array = squeeze(query_array)
        module_overlap = _array(_find_aux(aux, ("group_control_module_overlap",)))
        environment_overlap = _array(_find_aux(aux, ("group_control_environment_overlap",)))
        module_overlap = squeeze(module_overlap) if module_overlap is not None else np.einsum("qk,sk->qs", query_routing, module_incidence)
        environment_overlap = squeeze(environment_overlap) if environment_overlap is not None else np.einsum("qk,sk->qs", query_routing, environment_incidence)
        module_centres = _array(_find_aux(aux, ("group_control_module_centres", "group_control_module_centers")))
        environment_centres = _array(_find_aux(aux, ("group_control_environment_centres", "group_control_environment_centers")))
        module_centres = squeeze(module_centres) if module_centres is not None else np.full((GROUP_COUNT, 2), np.nan)
        environment_centres = squeeze(environment_centres) if environment_centres is not None else np.full((GROUP_COUNT, 2), np.nan)
        group_state = _array(_find_aux(aux, ("group_control_h", "group_control_group_control")))
        group_state = squeeze(group_state) if group_state is not None else None
        board_arrays = group_control.GroupControlArrays(
            module_coords=np.asarray(module_coords, dtype=np.float64),
            module_present=np.asarray(module_present, dtype=np.float64),
            env_coords=np.asarray(env_coords, dtype=np.float64),
            env_weights=np.asarray(env_weights, dtype=np.float64),
            module_incidence=np.asarray(module_incidence, dtype=np.float64),
            environment_incidence=np.asarray(environment_incidence, dtype=np.float64),
            query_xy=np.asarray(query_array, dtype=np.float64),
            query_routing=np.asarray(query_routing, dtype=np.float64),
            module_overlap=np.asarray(module_overlap, dtype=np.float64),
            environment_overlap=np.asarray(environment_overlap, dtype=np.float64),
            module_group_centres=np.asarray(module_centres, dtype=np.float64),
            environment_group_centres=np.asarray(environment_centres, dtype=np.float64),
            group_control=None if group_state is None else np.asarray(group_state, dtype=np.float64),
        )
        decode_ms = None if prepared_decode_median_ms is None else float(prepared_decode_median_ms)
        selected = min(max(int(query_routing.shape[0] // 2), 0), int(query_routing.shape[0] - 1))
        board_metrics = group_control.semantic_metrics(
            board_arrays,
            selected_query_index=selected,
            prepared_decode_median_ms=decode_ms,
        )
        board_ledger = {
            phase: {
                source: dict(value.get("sources", {}).get(source, {}))
                for source in ("module", "environment")
            }
            for phase, value in phase_ledger.items()
            if isinstance(value, Mapping)
        }
        for phase in board_ledger:
            for source in ("module", "environment"):
                row = board_ledger[phase][source]
                logical = row.get("logical_path_count")
                unique = row.get("unique_pair_count")
                if logical is not None and unique not in (None, 0):
                    row["multiplicity"] = float(logical) / float(unique)
        metadata = {
            "label": f"Run1409_{mode}",
            "case_id": str(case_id),
            "architecture": ARCHITECTURE,
            "mode": str(mode),
            "group_capacity": GROUP_COUNT,
            "control_dimension": CONTROL_DIM,
            "executor_policy": getattr(getattr(getattr(output, "core", None), "backend", None), "executor_policy", None),
            "gate_original_ids": list(range(GROUP_COUNT)),
            "gate_reference": "gate values are stored with original prototype IDs; packed columns retain this map",
            "budget_summary": budget,
            "phase_ledger_policy": "logical support, unique pairs, and actual fine rows remain separate",
        }
        for metadata_key, gate_key in (
            ("gate_values", "case_group_budget_gate_values"),
            ("gate_support", "case_group_budget_gate_support"),
            ("gate_positive_probability", "case_group_budget_gate_positive_probability"),
            ("gate_reference_eta", "case_group_budget_gate_reference_eta"),
        ):
            gate_value = _lookup(output, (gate_key,))
            if gate_value is None:
                gate_value = _find_aux(aux, (gate_key,))
            gate_array = _array(gate_value)
            if gate_array is not None and gate_array.size <= 256:
                metadata[metadata_key] = gate_array.tolist()
        group_control.save_group_control_npz(
            path,
            board_arrays,
            board_metrics,
            board_ledger,
            metadata=metadata,
        )
        # Add candidate-only gate arrays after the maintained helper writes its
        # canonical payload.  This keeps the existing board loader contract.
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        gate_names = (
            "case_group_budget_gate_values",
            "case_group_budget_gate_support",
            "case_group_budget_packed_prototype_ids",
            "case_group_budget_packed_valid",
            "case_group_budget_gate_positive_probability",
            "case_group_budget_gate_reference_eta",
        )
        for name in gate_names:
            value = _lookup(output, (name,))
            if value is None:
                value = _find_aux(aux, (name,))
            array = _array(value)
            if array is not None:
                payload[name] = array
        np.savez_compressed(path, **payload)
        return {
            "status": "complete",
            "path": str(path),
            "board_contract": "group_control_evidence.save_group_control_npz",
            "arrays": {key: list(value.shape) for key, value in payload.items() if isinstance(value, np.ndarray)},
            "gate_original_ids": list(range(GROUP_COUNT)),
        }
    except (ImportError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def _case_measurement(
    *,
    model: Any,
    checkpoint: Mapping[str, Any],
    dataset: Any,
    case_id: str,
    args: argparse.Namespace,
    mode: str,
    torch: Any,
    dynamic: Any,
    run1405: Any,
    make_batch: Callable[..., Any],
    select_sample: Callable[..., Any],
    device: Any,
    array_root: Path,
) -> dict[str, Any]:
    sample = select_sample(dataset, str(case_id), 0)
    query_np = stage3_query_points(sample, int(args.query_count))
    batch = make_batch(dict(sample), query_np, device)
    query = batch["query_xy"]
    forward_kwargs = run1405._phase_forward_kwargs(batch)
    chunk = int(args.receiver_chunk_size)

    def full_forward() -> Any:
        return _case_forward(model, batch, query, forward_kwargs, dynamic, chunk)

    def prepare_one() -> Any:
        return _case_forward(
            model,
            batch,
            query[:, :1],
            forward_kwargs,
            dynamic,
            chunk,
            return_prepared_state=True,
        )

    phases = {
        "full_forward": _measure(
            full_forward,
            device,
            warmups=int(args.warmups),
            repetitions=int(args.repetitions),
        ),
        "preparation_plus_one_query": _measure(
            prepare_one,
            device,
            warmups=int(args.warmups),
            repetitions=int(args.repetitions),
        ),
    }
    with torch.inference_mode():
        prepared_output = prepare_one()
    prepared = prepared_output.get("prepared_state") if isinstance(prepared_output, Mapping) else None
    if prepared is None:
        phases["prepared_p2_decode"] = {"status": "unavailable", "reason": "prepared_state missing"}
    else:
        def prepared_decode() -> Any:
            with dynamic._runtime_receiver_chunk_size(model, chunk):
                return model.decode_prepared(
                    prepared,
                    query,
                    return_routing_maps=False,
                    receiver_chunk_size=chunk,
                )

        phases["prepared_p2_decode"] = _measure(
            prepared_decode,
            device,
            warmups=int(args.warmups),
            repetitions=int(args.repetitions),
        )
    del prepared_output, prepared

    with torch.inference_mode():
        debug_output = _case_forward(
            model,
            batch,
            query,
            forward_kwargs,
            dynamic,
            chunk,
            return_prepared_state=True,
            return_routing_maps=True,
        )
    aux = _aux_mapping(debug_output)
    budget = _budget_summary(debug_output, aux)
    ledger = _phase_ledger(aux, getattr(model.core, "backend", None))
    rank = _rank_summary(aux)
    array_path = array_root / f"{mode}__{case_id}.npz"
    arrays = _save_debug_arrays(
        array_path,
        aux,
        output=debug_output,
        query_xy=query,
        phase_ledger=ledger,
        budget=budget,
        mode=mode,
        case_id=str(case_id),
        prepared_decode_median_ms=(
            None
            if phases["prepared_p2_decode"].get("median_seconds") is None
            else float(phases["prepared_p2_decode"]["median_seconds"]) * 1000.0
        ),
    )
    query_count = int(query.shape[1])
    # ``aux`` and ``debug_output`` contain query/source maps on the GPU.  The
    # compact summaries and canonical NPZ have already been written above;
    # do not retain those live tensors in the row list across mode/case
    # measurements.  This keeps later peak-memory samples comparable.
    del debug_output, aux, batch, query
    gc.collect()
    return {
        "case_id": str(case_id),
        "mode": str(mode),
        "checkpoint_epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "query_count": query_count,
        "receiver_chunk_size": chunk,
        "phases": phases,
        "budget": budget,
        "rank_summary": rank,
        "phase_ledger": ledger,
        "arrays": arrays,
        "executor": {
            "policy": getattr(model.core.backend, "executor_policy", None),
            "rectangular_ledger_rows": getattr(model.core.backend, "ledger_rectangular_rows", None),
            "diagnostic_executor_independent": getattr(model.core.backend, "diagnostic_executor_independent", None),
        },
    }


def stage3_query_points(sample: Mapping[str, Any], count: int) -> np.ndarray:
    """Deterministic fixed probes matching the maintained stage-3 evaluator."""

    x = np.asarray(sample["x_grid"], dtype=np.float32).reshape(-1)
    y = np.asarray(sample["y_grid"], dtype=np.float32).reshape(-1)
    points = np.stack((x, y), axis=-1)
    if count <= 0:
        raise ValueError("query count must be positive")
    if len(points) <= count:
        return points
    indices = np.linspace(0, len(points) - 1, int(count), dtype=np.int64)
    return points[indices]


def _gate_audit(
    *,
    model: Any,
    dataset: Any,
    case_id: str,
    args: argparse.Namespace,
    mode: str,
    torch: Any,
    dynamic: Any,
    run1405: Any,
    make_batch: Callable[..., Any],
    select_sample: Callable[..., Any],
    device: Any,
) -> dict[str, Any]:
    sample = select_sample(dataset, str(case_id), 0)
    query_np = stage3_query_points(sample, min(int(args.query_count), 1024))
    batch = make_batch(dict(sample), query_np, device)
    query = batch["query_xy"]
    forward_kwargs = run1405._phase_forward_kwargs(batch)
    stochastic: list[dict[str, Any]] = []
    stochastic_fields: list[Any] = []
    was_training = bool(model.training)
    try:
        # Keep the deployed eval model fixed. The context changes only the
        # P0 gate sampler; it does not toggle dropout, BatchNorm, or any
        # physical reader state.
        model.eval()
        with _execution_mode(model, mode), _stochastic_gate_mode(model) as stochastic_hook, torch.inference_mode():
            if stochastic_hook.get("status") == "unavailable":
                stochastic.append({"status": "unavailable", "hook": stochastic_hook})
            else:
                for index in range(int(args.gate_repetitions)):
                    output = _case_forward(
                        model,
                        batch,
                        query,
                        forward_kwargs,
                        dynamic,
                        int(args.receiver_chunk_size),
                        return_prepared_state=True,
                        return_routing_maps=True,
                    )
                    aux = _aux_mapping(output)
                    stochastic.append(
                        {
                            "repeat": index + 1,
                            "budget": _budget_summary(output, aux),
                            "phase_ledger": _phase_ledger(aux, getattr(model.core, "backend", None)),
                        }
                    )
                    if isinstance(output, Mapping):
                        stochastic_fields.append(_detach_cpu(output.get("pred_field")))
                    del output, aux
        stochastic_disagreement = []
        if stochastic_fields:
            reference = stochastic_fields[0]
            for field in stochastic_fields[1:]:
                stochastic_disagreement.append(_prediction_difference(field, reference))

        deterministic: dict[str, Any]
        # Eval mode is the deployed deterministic estimate in Run-1409. Keep
        # this call unpatched so the audit cannot silently replace production
        # routing with a test-specific estimator.
        with _execution_mode(model, mode), torch.inference_mode():
            output = _case_forward(
                model,
                batch,
                query,
                forward_kwargs,
                dynamic,
                int(args.receiver_chunk_size),
                return_prepared_state=True,
                return_routing_maps=True,
            )
            aux = _aux_mapping(output)
            deterministic = {
                "status": "complete",
                "policy": "model.eval() deployed deterministic hard-concrete estimate",
                "stochastic_hook": stochastic_hook,
                "budget": _budget_summary(output, aux),
                "phase_ledger": _phase_ledger(aux, getattr(model.core, "backend", None)),
                "prediction": _detach_cpu(output.get("pred_field"))
                if isinstance(output, Mapping)
                else None,
            }
            del output, aux
        if deterministic.get("status") == "complete":
            deterministic_disagreement = [
                _prediction_difference(field, deterministic.get("prediction"))
                for field in stochastic_fields
            ]
            deterministic["stochastic_to_deterministic_prediction_disagreement"] = deterministic_disagreement
            deterministic.pop("prediction", None)
        return {
            "case_id": str(case_id),
            "mode": str(mode),
            "stochastic_repeats": stochastic,
            "stochastic_prediction_disagreement_vs_first": stochastic_disagreement,
            "deterministic": deterministic,
            "interpretation": "gate-plan variation and frozen-model output disagreement are reported; neither establishes physical causality",
        }
    finally:
        model.train(was_training)


def _compare_modes(full: Mapping[str, Any], compact: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "complete", "case_id": full.get("case_id")}
    result["prediction_disagreement"] = _prediction_difference(full.get("prediction"), compact.get("prediction"))
    result["budget"] = {
        "full_width": full.get("budget"),
        "compact": compact.get("budget"),
    }
    result["phase_ledger"] = {
        "full_width": full.get("phase_ledger"),
        "compact": compact.get("phase_ledger"),
    }
    result["interpretation"] = "same operator under matched RNG state; any residual is an implementation discrepancy, not a physical-rank estimate"
    return result


def _matched_mode_pair(
    *,
    model: Any,
    dataset: Any,
    case_id: str,
    args: argparse.Namespace,
    torch: Any,
    dynamic: Any,
    run1405: Any,
    make_batch: Callable[..., Any],
    select_sample: Callable[..., Any],
    device: Any,
) -> dict[str, Any]:
    """Run full/compact map passes from the same eval RNG/state."""

    sample = select_sample(dataset, str(case_id), 0)
    query_np = stage3_query_points(sample, int(args.query_count))
    batch = make_batch(dict(sample), query_np, device)
    query = batch["query_xy"]
    forward_kwargs = run1405._phase_forward_kwargs(batch)
    model.eval()
    initial_rng = _rng_state(torch, device)
    with _execution_mode(model, "full_width"), torch.inference_mode():
        full = _case_forward(
            model,
            batch,
            query,
            forward_kwargs,
            dynamic,
            int(args.receiver_chunk_size),
            return_prepared_state=True,
            return_routing_maps=True,
        )
    _restore_rng(torch, device, initial_rng)
    with _execution_mode(model, "compact"), torch.inference_mode():
        compact = _case_forward(
            model,
            batch,
            query,
            forward_kwargs,
            dynamic,
            int(args.receiver_chunk_size),
            return_prepared_state=True,
            return_routing_maps=True,
        )
    full_aux = _aux_mapping(full)
    compact_aux = _aux_mapping(compact)
    return {
        "status": "complete",
        "case_id": str(case_id),
        "gate_policy": "same eval deterministic gate estimate and restored RNG state",
        "prediction_disagreement": _prediction_difference(full.get("pred_field"), compact.get("pred_field")),
        "budget": {
            "full_width": _budget_summary(full, full_aux),
            "compact": _budget_summary(compact, compact_aux),
        },
        "phase_ledger": {
            "full_width": _phase_ledger(full_aux, getattr(model.core, "backend", None)),
            "compact": _phase_ledger(compact_aux, getattr(model.core, "backend", None)),
        },
        "interpretation": "same deterministic gate plan and operator; residual prediction difference is an implementation discrepancy, not a physical-rank estimate",
    }


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    checkpoint_path = _parse_checkpoint(args.checkpoint)
    torch, dynamic, run1405, stage3, helpers = _runtime_imports()
    make_batch, select_sample = helpers
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device={args.device}")
        torch.cuda.set_device(device)
    spec = stage3.CheckpointSpec(label="1409", path=checkpoint_path)
    model, checkpoint = stage3._load_model_spec(spec, device)
    checkpoint_epoch = int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1)))
    if checkpoint_epoch not in REVIEW_EPOCHS:
        raise ValueError(
            f"Run-1409 v2 evidence accepts exact review checkpoints {REVIEW_EPOCHS}; "
            f"got epoch={checkpoint_epoch}"
        )
    architecture = str(model.config.core_honf.forward_architecture)
    if architecture != ARCHITECTURE:
        raise ValueError(f"checkpoint architecture is {architecture!r}; expected {ARCHITECTURE!r}")
    interface = model.config.core_honf.interface_model
    if int(interface.group_count) != GROUP_COUNT or int(interface.group_control_dim) != CONTROL_DIM:
        raise ValueError(
            f"candidate config must use Kmax={GROUP_COUNT}, D={CONTROL_DIM}; "
            f"got K={interface.group_count}, D={interface.group_control_dim}"
        )
    dataset, dataset_path = stage3._load_dataset(
        checkpoint,
        argparse.Namespace(dataset=args.dataset, split=args.split),
    )
    output = Path(args.output).expanduser().resolve()
    array_root = Path(args.array_dir or output.parent / "arrays").expanduser().resolve()
    modes = _modes(args)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    mode_rows: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for mode in modes:
            _set_mode_once(model, mode)
            for case_id in (args.case_id or DEFAULT_CASE_IDS):
                # Keep allocator history from a prior mode/case out of the
                # next headline measurement.  Model weights remain live;
                # only released diagnostic/query buffers are returned.
                gc.collect()
                if getattr(device, "type", None) == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                row = _case_measurement(
                    model=model,
                    checkpoint=checkpoint,
                    dataset=dataset,
                    case_id=str(case_id),
                    args=args,
                    mode=mode,
                    torch=torch,
                    dynamic=dynamic,
                    run1405=run1405,
                    make_batch=make_batch,
                    select_sample=select_sample,
                    device=device,
                    array_root=array_root,
                )
                mode_rows[(str(case_id), mode)] = row
                rows.append(row)
                audits.append(
                    _gate_audit(
                        model=model,
                        dataset=dataset,
                        case_id=str(case_id),
                        args=args,
                        mode=mode,
                        torch=torch,
                        dynamic=dynamic,
                        run1405=run1405,
                        make_batch=make_batch,
                        select_sample=select_sample,
                        device=device,
                    )
                )
        mode_comparisons = []
        if "full_width" in modes and "compact" in modes:
            for case_id in (args.case_id or DEFAULT_CASE_IDS):
                full = mode_rows[(str(case_id), "full_width")]
                compact = mode_rows[(str(case_id), "compact")]
                mode_comparisons.append(
                    _matched_mode_pair(
                        model=model,
                        dataset=dataset,
                        case_id=str(case_id),
                        args=args,
                        torch=torch,
                        dynamic=dynamic,
                        run1405=run1405,
                        make_batch=make_batch,
                        select_sample=select_sample,
                        device=device,
                    )
                )
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        gc.collect()
    # Remove live tensors before JSON conversion while retaining compact audit data.
    for row in rows:
        row.pop("prediction", None)
        row.pop("interaction_aux", None)
    return {
        "schema_version": 1,
        "task": "run1409_budgeted_group_control_evidence",
        "status": "complete",
        "candidate": {
            "architecture": architecture,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "dataset": str(dataset_path),
            "group_capacity": GROUP_COUNT,
            "control_dimension": CONTROL_DIM,
            "selection_policy": "explicit_cli_checkpoint",
        },
        "protocol": {
            "case_ids": [str(value) for value in (args.case_id or DEFAULT_CASE_IDS)],
            "query_count": int(args.query_count),
            "receiver_chunk_size": int(args.receiver_chunk_size),
            "warmups": int(args.warmups),
            "repetitions": int(args.repetitions),
            "gate_repetitions": int(args.gate_repetitions),
            "modes": list(modes),
            "timed_maps": False,
            "map_pass": "one untimed map/ledger forward per case and mode",
            "device": str(device),
        },
        "rows": rows,
        "gate_audits": audits,
        "mode_comparisons": mode_comparisons,
        "limitations": [
            "The rank summaries describe learned routing/group-state arrays and do not estimate optimal physical rank.",
            "Logical support, unique pairs, and actual fine rows are retained as separate ledger fields.",
            "A missing stochastic gate hook is evidence of an unavailable stochastic audit path, not a deterministic result.",
            "This artifact does not compare accuracy against Run 1404/1406/1804; use the matched evaluator for that table.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--array-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--receiver-chunk-size", type=int, default=DEFAULT_RECEIVER_CHUNK_SIZE)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--gate-repetitions", type=int, default=DEFAULT_GATE_REPETITIONS)
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="execution mode; repeat for both (default: full_width and compact)",
    )
    parser.add_argument("--plan-only", action="store_true", help="validate explicit paths/protocol without loading CUDA/model")
    return parser


def _print_status(payload: Mapping[str, Any], output: Path) -> None:
    """Print a bounded completion line; the full evidence stays in JSON."""

    protocol = payload.get("protocol")
    if isinstance(protocol, Mapping):
        cases = protocol.get("case_ids", [])
    else:
        cases = []
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "task": payload.get("task"),
                "output": str(output.expanduser().resolve()),
                "cases": list(cases) if isinstance(cases, (list, tuple)) else [],
                "rows": len(payload.get("rows", [])) if isinstance(payload.get("rows"), list) else 0,
                "gate_audits": len(payload.get("gate_audits", []))
                if isinstance(payload.get("gate_audits"), list)
                else 0,
                "mode_comparisons": len(payload.get("mode_comparisons", []))
                if isinstance(payload.get("mode_comparisons"), list)
                else 0,
            },
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    if args.plan_only:
        _write_json(args.output, plan)
        _print_status(plan, args.output)
        return 0
    payload = run_measurement(args)
    _write_json(args.output, payload)
    _print_status(payload, args.output)
    return 0 if payload.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE",
    "GROUP_COUNT",
    "CONTROL_DIM",
    "build_parser",
    "build_plan",
    "main",
    "run_measurement",
]
