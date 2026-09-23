"""Benchmark the exact Run-1503 adaptive hyperedge-opening executor.

This is an evaluation-only runner for explicitly labelled checkpoints whose
architecture is exactly ``adaptive_hyperedge_opening_honf``.  The three
timed scopes use routing maps off and one matched evaluation receiver chunk:

* ``full_physical_forward``: the complete physical model forward;
* ``prepared_p2_decode``: a prepared case decoded at the requested receivers;
* ``application_evaluator``: the maintained full-grid application evaluator,
  including its CPU serialization scope.

Fine/coarse/source-row ledgers are collected by a separate untimed maps-on
probe with the same query count and receiver chunk.  A failed large-chunk
scope (for example CUDA OOM at 2048 receivers) is retained as unavailable;
the runner never silently substitutes a different chunk or execution mode.
Checkpoint loading is strict through the maintained evaluator loader.  This
module does not launch training, write checkpoints, or alter model defaults.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from run_run1501_selected_executor_benchmark import (
    _application_evaluator as _run1501_application_evaluator,
)
from run_run1501_selected_executor_benchmark import (
    _decode_prepared as _run1501_decode_prepared,
)
from run_run1501_selected_executor_benchmark import (
    _forward as _run1501_forward,
)
from run_run1501_selected_executor_benchmark import (
    _imports as _run1501_imports,
)
from run_run1501_selected_executor_benchmark import (
    _prediction as _run1501_prediction,
)
from run_run1501_selected_executor_benchmark import (
    _prepare as _run1501_prepare,
)
from run_run1501_selected_executor_benchmark import (
    _query_points as _run1501_query_points,
)

EXPECTED_ARCHITECTURE = "adaptive_hyperedge_opening_honf"
TIMED_SCOPES = (
    "full_physical_forward",
    "prepared_p2_decode",
    "application_evaluator",
)


def _jsonable(value: Any) -> Any:
    """Convert scalar/tensor values used by the evidence payload to JSON."""

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu()
        if int(value.numel()) == 1:
            return _jsonable(value.item())
        if int(value.numel()) <= 256:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _jsonable(value.item())
        if int(value.size) <= 256:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _imports() -> tuple[Any, ...]:
    """Import the maintained case/runtime modules only when a live run starts."""
    # Keep the maintained Run-1501 loader/forward import surface as the
    # compatibility path; this file only specializes identity, ledgers, and
    # memory accounting for Run-1503.
    return _run1501_imports()


def _query_points(sample: Mapping[str, Any], count: int) -> np.ndarray:
    """Select deterministic points from the physical case grid."""
    return _run1501_query_points(sample, count)


def _require_checkpoint_identity(
    model: Any,
    checkpoint: Mapping[str, Any],
    *,
    expected_epoch: int | None = None,
) -> dict[str, Any]:
    """Enforce the Run-1503 architecture and optional exact gate epoch."""

    core = getattr(getattr(model, "config", None), "core_honf", None)
    architecture = str(getattr(core, "forward_architecture", ""))
    if architecture != EXPECTED_ARCHITECTURE:
        raise ValueError(
            "Run-1503 benchmark requires exact architecture "
            f"{EXPECTED_ARCHITECTURE!r}, got {architecture!r}"
        )
    model_config = checkpoint.get("model_config", {})
    config_core = model_config.get("core_honf", {}) if isinstance(model_config, Mapping) else {}
    config_architecture = config_core.get("forward_architecture") if isinstance(config_core, Mapping) else None
    if config_architecture is not None and str(config_architecture) != EXPECTED_ARCHITECTURE:
        raise ValueError(
            "checkpoint metadata architecture does not match Run-1503: "
            f"{config_architecture!r}"
        )
    epoch_value = checkpoint.get("epoch", checkpoint.get("current_epoch", -1))
    try:
        epoch = int(epoch_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint epoch is not an integer: {epoch_value!r}") from exc
    selection_state = checkpoint.get("selection_state")
    if isinstance(selection_state, Mapping) and selection_state.get("epoch") is not None:
        try:
            selection_epoch = int(selection_state["epoch"])
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint selection_state.epoch is not an integer") from exc
        if selection_epoch != epoch:
            raise ValueError(
                "checkpoint epoch metadata disagree: "
                f"top-level={epoch}, selection_state={selection_epoch}"
            )
    if expected_epoch is not None:
        if int(expected_epoch) not in {150, 500}:
            raise ValueError("Run-1503 gate epoch must be exactly 150 or 500")
        if epoch != int(expected_epoch):
            raise ValueError(
                f"Run-1503 gate requires epoch {int(expected_epoch)}, checkpoint has epoch {epoch}"
            )
    return {
        "architecture": architecture,
        "epoch": epoch,
        "strict_model_state_load": True,
        "expected_gate_epoch": None if expected_epoch is None else int(expected_epoch),
    }


def _forward(
    model: Any,
    batch: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    *,
    return_routing_maps: bool = False,
    return_prepared_state: bool = False,
    query: Any | None = None,
) -> Any:
    return _run1501_forward(
        model,
        batch,
        kwargs,
        return_routing_maps=return_routing_maps,
        return_prepared_state=return_prepared_state,
        query=query,
    )


def _prepare(model: Any, batch: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Any:
    """Build the final physical prepared state once outside decode timing."""
    return _run1501_prepare(model, batch, kwargs)


def _decode_prepared(model: Any, prepared: Any, query: Any, receiver_chunk: int) -> Any:
    return _run1501_decode_prepared(model, prepared, query, receiver_chunk)


def _prediction(output: Mapping[str, Any]) -> Any:
    return _run1501_prediction(output)


def _measure(
    torch: Any,
    device: Any,
    function: Any,
    *,
    receiver_chunk_size: int,
    warmups: int,
    repetitions: int,
    name: str,
) -> dict[str, Any]:
    """Measure synchronized wall/CUDA time and absolute/delta peak memory."""

    if int(warmups) < 0 or int(repetitions) <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    if int(receiver_chunk_size) <= 0:
        raise ValueError("receiver chunk size must be positive")
    is_cuda = getattr(device, "type", str(device)) == "cuda"
    try:
        with torch.inference_mode():
            for _ in range(int(warmups)):
                output = function()
                del output
            if is_cuda:
                torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001 - preserve chunk/OOM boundaries
        if is_cuda:
            torch.cuda.empty_cache()
        return {
            "scope": str(name),
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc!s}",
            "receiver_chunk_size": int(receiver_chunk_size),
            "warmups": int(warmups),
            "repetitions": int(repetitions),
            "timed_maps": False,
        }

    samples: list[dict[str, Any]] = []
    for repetition in range(1, int(repetitions) + 1):
        baseline_allocated = baseline_reserved = None
        peak_allocated = peak_reserved = None
        cuda_event_ms = None
        error = None
        output: Any = None
        if is_cuda:
            torch.cuda.synchronize(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
        else:
            start_event = stop_event = None
        wall_started = time.perf_counter()
        try:
            if start_event is not None:
                start_event.record()
            output = function()
            if stop_event is not None:
                stop_event.record()
                stop_event.synchronize()
            wall_ms = 1000.0 * (time.perf_counter() - wall_started)
            if start_event is not None and stop_event is not None:
                cuda_event_ms = float(start_event.elapsed_time(stop_event))
            if is_cuda:
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
        except Exception as exc:  # noqa: BLE001 - retain unavailable large chunks
            wall_ms = 1000.0 * (time.perf_counter() - wall_started)
            error = f"{type(exc).__name__}: {exc!s}"
            if is_cuda:
                torch.cuda.synchronize(device)
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
        finally:
            del output
            if is_cuda:
                if error is not None:
                    torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
        samples.append(
            {
                "repetition": int(repetition),
                "status": "unavailable" if error is not None else "complete",
                "error": error,
                "wall_ms": float(wall_ms),
                "cuda_event_ms": cuda_event_ms,
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "incremental_peak_allocated_bytes": (
                    None
                    if peak_allocated is None or baseline_allocated is None
                    else int(peak_allocated - baseline_allocated)
                ),
                "incremental_peak_reserved_bytes": (
                    None
                    if peak_reserved is None or baseline_reserved is None
                    else int(peak_reserved - baseline_reserved)
                ),
            }
        )
        if error is not None:
            break

    complete = [sample for sample in samples if sample["status"] == "complete"]
    wall_values = np.asarray([sample["wall_ms"] for sample in complete], dtype=np.float64)
    cuda_values = np.asarray(
        [sample["cuda_event_ms"] for sample in complete if sample["cuda_event_ms"] is not None],
        dtype=np.float64,
    )

    def _max_field(field: str) -> int | None:
        values = [sample[field] for sample in complete if sample[field] is not None]
        return None if not values else int(max(values))

    peak_total_allocated = _max_field("peak_allocated_bytes")
    peak_total_reserved = _max_field("peak_reserved_bytes")
    peak_incremental_allocated = _max_field("incremental_peak_allocated_bytes")
    peak_incremental_reserved = _max_field("incremental_peak_reserved_bytes")
    return {
        "scope": str(name),
        "status": "complete" if len(complete) == int(repetitions) else "unavailable",
        "error": None if len(complete) == len(samples) else samples[-1]["error"],
        "receiver_chunk_size": int(receiver_chunk_size),
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "samples": samples,
        "median_wall_ms": None if not len(wall_values) else float(np.median(wall_values)),
        "p95_wall_ms": None if not len(wall_values) else float(np.percentile(wall_values, 95)),
        "median_cuda_event_ms": None if not len(cuda_values) else float(np.median(cuda_values)),
        "peak_total_allocated_bytes": peak_total_allocated,
        "peak_total_reserved_bytes": peak_total_reserved,
        "total_peak_allocated_bytes": peak_total_allocated,
        "total_peak_reserved_bytes": peak_total_reserved,
        "peak_incremental_allocated_bytes": peak_incremental_allocated,
        "peak_incremental_reserved_bytes": peak_incremental_reserved,
        "timed_maps": False,
    }


def _ledger_value(value: Any, *, max_elements: int = 256) -> Any:
    """Keep small K/E ledgers exact and summarize larger Q tensors."""

    if value is None:
        return None
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    if int(array.size) <= int(max_elements):
        return _jsonable(array)
    numeric = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sum": float(np.sum(numeric)),
        "mean": float(np.mean(numeric)),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
    }


def _first_aux(aux: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in aux and aux[name] is not None:
            return aux[name]
    return None


def _adaptive_ledger(output: Mapping[str, Any], *, receiver_chunk_size: int) -> dict[str, Any]:
    """Extract adaptive fine/coarse/source rows from one maps-on probe."""

    aux = output.get("interaction_aux", {})
    if not isinstance(aux, Mapping):
        return {
            "status": "unavailable",
            "reason": "maps-on output did not contain interaction_aux",
            "receiver_chunk_size": int(receiver_chunk_size),
        }
    fine_group_forward = _first_aux(
        aux,
        "group_control_adaptive_fine_group_rows_forward",
        "group_control_adaptive_fine_group_rows",
    )
    fine_group_logical = _first_aux(aux, "group_control_adaptive_fine_group_rows_logical")
    fine_group_padded = _first_aux(aux, "group_control_adaptive_fine_group_rows_padded")
    fine_total = _first_aux(
        aux,
        "group_control_adaptive_fine_rows_forward",
        "group_control_adaptive_fine_rows",
    )
    source_mass = _first_aux(
        aux,
        "group_control_adaptive_environment_source_mass",
        "adaptive_hyperedge_environment_source_mass",
    )
    source_incidence = _first_aux(
        aux,
        "group_control_environment_incidence",
        "sparse_incidence_environment_incidence",
    )
    source_measure = _first_aux(
        aux,
        "group_control_environment_measure",
        "sparse_incidence_environment_measure",
    )
    source_rows_per_group = None
    if source_mass is not None:
        source_array = np.asarray(source_mass.detach().cpu() if hasattr(source_mass, "detach") else source_mass)
        source_array = source_array[0] if source_array.ndim == 3 and source_array.shape[0] == 1 else source_array
        if source_array.ndim == 2:
            source_rows_per_group = (source_array > 0.0).sum(axis=0)
    if source_rows_per_group is None and source_incidence is not None:
        incidence_array = np.asarray(
            source_incidence.detach().cpu() if hasattr(source_incidence, "detach") else source_incidence
        )
        incidence_array = incidence_array[0] if incidence_array.ndim == 3 and incidence_array.shape[0] == 1 else incidence_array
        if incidence_array.ndim == 2:
            source_rows_per_group = (incidence_array > 0.0).sum(axis=0)
    query_count = int(output["pred_field"].shape[1]) if hasattr(output.get("pred_field"), "shape") else None
    if query_count is None:
        rows_per_query = _first_aux(aux, "group_control_adaptive_fine_rows_per_query")
        query_count = None if rows_per_query is None else int(np.asarray(rows_per_query).shape[-1])
    required = (fine_group_forward, fine_total, source_rows_per_group)
    coarse_total = _first_aux(aux, "group_control_adaptive_coarse_rows")
    return {
        "status": "complete" if all(value is not None for value in required) else "partial",
        "receiver_chunk_size": int(receiver_chunk_size),
        "query_count": query_count,
        "query_chunk_count": None
        if query_count is None
        else math.ceil(query_count / int(receiver_chunk_size)),
        "fine_rows_forward": _ledger_value(fine_total),
        "fine_rows_logical": _ledger_value(
            _first_aux(aux, "group_control_adaptive_fine_rows_logical")
        ),
        "fine_rows_padded": _ledger_value(
            _first_aux(
                aux,
                "group_control_adaptive_fine_rows_padded",
                "group_control_environment_fine_rows_padded",
            )
        ),
        "coarse_rows": _ledger_value(coarse_total),
        "source_rows_per_group": _ledger_value(source_rows_per_group),
        "fine": {
            "group_rows_forward": _ledger_value(fine_group_forward),
            "group_rows_logical": _ledger_value(fine_group_logical),
            "group_rows_padded": _ledger_value(fine_group_padded),
            "rows_per_query": _ledger_value(
                _first_aux(aux, "group_control_adaptive_fine_rows_per_query")
            ),
            "rows_logical": _ledger_value(
                _first_aux(aux, "group_control_adaptive_fine_rows_logical")
            ),
            "rows_forward": _ledger_value(fine_total),
            "rows_padded": _ledger_value(
                _first_aux(
                    aux,
                    "group_control_adaptive_fine_rows_padded",
                    "group_control_environment_fine_rows_padded",
                )
            ),
            "work_ratio": _ledger_value(
                _first_aux(aux, "group_control_adaptive_fine_work_ratio")
            ),
        },
        "coarse": {
            "rows": _ledger_value(
                _first_aux(aux, "group_control_adaptive_coarse_rows")
            ),
            "full_rectangle_rows": _ledger_value(
                _first_aux(aux, "group_control_adaptive_full_rectangle_rows")
            ),
            "available": _ledger_value(
                _first_aux(
                    aux,
                    "adaptive_hyperedge_coarse_available",
                    "group_control_adaptive_coarse_available",
                )
            ),
        },
        "source": {
            "rows_per_group": _ledger_value(source_rows_per_group),
            "source_mass": _ledger_value(source_mass),
            "incidence": _ledger_value(source_incidence),
            "measure": _ledger_value(source_measure),
            "source_projection_rows": _ledger_value(
                _first_aux(
                    aux,
                    "group_control_adaptive_environment_source_projection_rows",
                )
            ),
            "coarse_source_projection_rows": _ledger_value(
                _first_aux(
                    aux,
                    "group_control_adaptive_coarse_source_projection_rows",
                )
            ),
        },
        "support": {
            "unique_pairs": _ledger_value(
                _first_aux(
                    aux,
                    "group_control_environment_unique_pairs",
                    "group_control_adaptive_support_pairs",
                )
            ),
            "logical_paths": _ledger_value(
                _first_aux(aux, "group_control_environment_logical_paths")
            ),
            "support_rows": _ledger_value(
                _first_aux(aux, "group_control_environment_support_rows")
            ),
        },
        "maps_probe_timed": False,
    }


def _application_evaluator(
    predict_case: Any,
    model: Any,
    sample: Mapping[str, Any],
    device: Any,
    receiver_chunk: int,
) -> Any:
    """Run the maintained evaluator with the same temporary receiver chunk."""
    return _run1501_application_evaluator(
        predict_case, model, sample, device, receiver_chunk
    )


def _case_row(
    *,
    torch: Any,
    stage3: Any,
    predict_case: Any,
    model: Any,
    sample: Mapping[str, Any],
    batch: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    query: Any,
    device: Any,
    receiver_chunk: int,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    with torch.inference_mode(), stage3._runtime_receiver_chunk_size(model, receiver_chunk):
        prepared = _prepare(model, batch, kwargs)
        timings = {
            "full_physical_forward": _measure(
                torch,
                device,
                lambda: _forward(model, batch, kwargs),
                receiver_chunk_size=receiver_chunk,
                warmups=warmups,
                repetitions=repetitions,
                name="full_physical_forward",
            ),
            "prepared_p2_decode": _measure(
                torch,
                device,
                lambda: _decode_prepared(model, prepared, query, receiver_chunk),
                receiver_chunk_size=receiver_chunk,
                warmups=warmups,
                repetitions=repetitions,
                name="prepared_p2_decode",
            ),
            "application_evaluator": _measure(
                torch,
                device,
                lambda: _application_evaluator(
                    predict_case, model, sample, device, receiver_chunk
                ),
                receiver_chunk_size=receiver_chunk,
                warmups=warmups,
                repetitions=repetitions,
                name="application_evaluator",
            ),
        }

        try:
            full_output = _forward(model, batch, kwargs)
            full_prediction = _prediction(full_output)
            del full_output
            prepared_output = _decode_prepared(model, prepared, query, receiver_chunk)
            prepared_prediction = _prediction(prepared_output)
            del prepared_output
            parity = {
                "prepared_vs_full_max_abs": float(
                    (prepared_prediction - full_prediction).abs().max()
                ),
                "prepared_vs_full_mean_abs": float(
                    (prepared_prediction - full_prediction).abs().mean()
                ),
            }
        except Exception as exc:  # noqa: BLE001 - retain large-chunk boundaries
            parity = {
                "prepared_vs_full_max_abs": None,
                "prepared_vs_full_mean_abs": None,
                "parity_status": "unavailable",
                "parity_error": f"{type(exc).__name__}: {exc!s}",
            }
            full_prediction = None
            prepared_prediction = None

        # This is deliberately untimed and maps-on.  It emits the adaptive
        # execution ledgers while leaving every timed scope maps-off.
        try:
            ledger_output = _forward(
                model,
                batch,
                kwargs,
                return_routing_maps=True,
            )
            ledger = _adaptive_ledger(ledger_output, receiver_chunk_size=receiver_chunk)
            del ledger_output
        except Exception as exc:  # noqa: BLE001 - maps-on may exceed memory at 2048
            ledger = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc!s}",
                "receiver_chunk_size": int(receiver_chunk),
                "maps_probe_timed": False,
            }
        application_query_count = int(np.asarray(sample["x_grid"]).size)
        prepared_shape = list(query.shape[:2]) + [int(model.config.field_dim)]
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.empty_cache()
    return {
        "timings": timings,
        "ledger": ledger,
        "ledger_probe_maps": True,
        "ledger_probe_timed": False,
        "prepared_decode_output_device": str(query.device),
        "prepared_decode_output_shape": prepared_shape,
        "application_query_count": application_query_count,
        **parity,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch, run1405, stage3, make_batch, predict_case, select_sample = _imports()
    specs = stage3.parse_checkpoint_specs(args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    receiver_chunk = int(args.receiver_chunk_size)
    if receiver_chunk <= 0:
        raise ValueError("receiver chunk size must be positive")
    case_ids = [str(case_id) for case_id in (args.case_id or ["0273", "0653"])]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be non-empty and unique")
    rows: list[dict[str, Any]] = []
    checkpoint_records: dict[str, Any] = {}
    for spec in specs:
        model, checkpoint_payload = stage3._load_model_spec(spec, device)
        identity = _require_checkpoint_identity(
            model,
            checkpoint_payload,
            expected_epoch=args.expected_epoch,
        )
        checkpoint_records[spec.label] = {
            "path": str(spec.path),
            **identity,
        }
        dataset, _dataset_path = stage3._load_dataset(
            checkpoint_payload,
            argparse.Namespace(dataset=args.dataset, split=args.split),
        )
        try:
            for case_id in case_ids:
                sample = select_sample(dataset, case_id, 0)
                query_np = _query_points(sample, int(args.query_count))
                batch = make_batch(dict(sample), query_np, device)
                kwargs = run1405._phase_forward_kwargs(batch)
                row = _case_row(
                    torch=torch,
                    stage3=stage3,
                    predict_case=predict_case,
                    model=model,
                    sample=sample,
                    batch=batch,
                    kwargs=kwargs,
                    query=batch["query_xy"],
                    device=device,
                    receiver_chunk=receiver_chunk,
                    warmups=int(args.warmups),
                    repetitions=int(args.repetitions),
                )
                row.update(
                    {
                        "checkpoint_label": spec.label,
                        "checkpoint": str(spec.path),
                        "checkpoint_epoch": identity["epoch"],
                        "architecture": identity["architecture"],
                        "case_id": case_id,
                        "query_count": int(args.query_count),
                        "receiver_chunk_size": receiver_chunk,
                        "device": str(device),
                        "dtype": str(batch["query_xy"].dtype),
                        "timed_maps": False,
                    }
                )
                rows.append(row)
        finally:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    row_statuses = [
        all(scope["status"] == "complete" for scope in row["timings"].values())
        for row in rows
    ]
    return {
        "schema_version": 1,
        "task": "run1503_adaptive_hyperedge_executor_scopes",
        "status": "complete" if all(row_statuses) else "partial",
        "device": str(device),
        "checkpoints": checkpoint_records,
        "case_ids": case_ids,
        "query_count": int(args.query_count),
        "receiver_chunk_size": receiver_chunk,
        "warmups": int(args.warmups),
        "repetitions": int(args.repetitions),
        "timed_scopes": list(TIMED_SCOPES),
        "timed_maps": False,
        "synchronized_timing": {
            "wall_clock": "time.perf_counter around each call",
            "cuda_event": "torch.cuda.Event start/stop with stop-event synchronization when CUDA",
        },
        "memory_contract": {
            "total_peak": "peak_allocated_bytes and peak_reserved_bytes",
            "incremental_peak": "peak minus live baseline at scope entry",
        },
        "rows": rows,
        "notes": [
            "Checkpoint loading uses the maintained strict state-dict loader.",
            "The maps-on adaptive ledger probe is separate from all timed maps-off scopes.",
            "The requested receiver chunk is preserved; an unavailable 2048 scope is not silently rerun at another chunk.",
            "No Run-1501/Run-1502 checkpoint is inspected or monitored, and no training is launched.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeat for each explicitly labelled adaptive Run-1503 checkpoint",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--query-count", type=int, default=8192)
    parser.add_argument(
        "--receiver-chunk-size",
        type=int,
        default=2048,
        help="one matched evaluation-only receiver chunk; 2048 is the default gate probe",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--expected-epoch", type=int, choices=(150, 500), default=None)
    parser.add_argument("--device", default="cuda:1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.case_id:
        args.case_id = ["0273", "0653"]
    payload = run(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ARCHITECTURE",
    "TIMED_SCOPES",
    "_adaptive_ledger",
    "_measure",
    "_require_checkpoint_identity",
    "build_parser",
    "run",
]
