"""Benchmark the exact Run-1501 support-block executor against rectangular fallback.

This is a narrow, read-only benchmark for the two maintained executor modes.
The timed scopes deliberately keep routing maps off:

* ``full_gpu_forward``: complete model forward, with the result retained on GPU;
* ``prepared_p2_decode``: prepared P1 state is built outside timing, then the
  requested Q receivers are decoded on GPU; and
* ``application_evaluator``: the application evaluator's full-grid prepared
  path, including its CPU transfer scope.

An untimed maps-on probe records the execution ledger.  It is kept separate so
diagnostic maps cannot change the timed operator path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_MODES = ("rectangular", "support_blocks")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu()
        if int(value.numel()) == 1:
            return _jsonable(value.item())
        return value.tolist() if int(value.numel()) <= 256 else {"shape": list(value.shape)}
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
    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_run1405_epoch50_comparison as run1405
    import run_stage3_interface_study as stage3
    import torch
    from channelthermal.evaluation.loading import make_batch
    from channelthermal.evaluation.prepared import predict_case, select_sample

    return torch, run1405, stage3, make_batch, predict_case, select_sample


def _query_points(sample: Mapping[str, Any], count: int) -> np.ndarray:
    x = np.asarray(sample["x_grid"], dtype=np.float32).reshape(-1)
    y = np.asarray(sample["y_grid"], dtype=np.float32).reshape(-1)
    points = np.stack([x, y], axis=-1)
    if count <= 0 or count > len(points):
        raise ValueError(f"query count must be in [1,{len(points)}]")
    if count == len(points):
        return points
    indices = np.rint(np.linspace(0, len(points) - 1, count)).astype(np.int64)
    return points[indices]


def _configure(backend: Any, mode: str) -> None:
    """Select one explicitly named executor mode and no hidden hybrid path."""

    if mode == "rectangular":
        backend.executor_policy = "rectangular_reference"
        backend.diagnostic_executor_independent = True
        backend.ledger_rectangular_rows = True
    elif mode == "support_blocks":
        backend.executor_policy = "support_blocks"
        backend.diagnostic_executor_independent = False
        backend.ledger_rectangular_rows = False
    else:
        raise ValueError(f"unknown executor mode {mode!r}")


def _forward(
    model: Any,
    batch: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    *,
    return_routing_maps: bool = False,
    return_prepared_state: bool = False,
    query: Any | None = None,
) -> Any:
    return model(
        batch["structure"],
        batch["query_xy"] if query is None else query,
        return_prepared_state=bool(return_prepared_state),
        return_routing_maps=bool(return_routing_maps),
        **kwargs,
    )


def _prepare(
    model: Any,
    batch: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> Any:
    """Build P1 once; the returned state is intentionally mode-specific."""

    output = _forward(
        model,
        batch,
        kwargs,
        return_prepared_state=True,
        return_routing_maps=False,
        query=batch["query_xy"][:, :1],
    )
    try:
        return output["prepared_state"]
    finally:
        del output


def _decode_prepared(model: Any, prepared: Any, query: Any, receiver_chunk: int) -> Any:
    return model.decode_prepared(
        prepared,
        query,
        return_routing_maps=False,
        return_edge_fields=False,
        receiver_chunk_size=int(receiver_chunk),
    )


def _measure(
    torch: Any,
    device: Any,
    function: Any,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    """Measure synchronized wall time, CUDA event time, and allocation peaks."""

    if int(warmups) < 0 or int(repetitions) <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    samples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for _ in range(int(warmups)):
            output = function()
            del output
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for repetition in range(1, int(repetitions) + 1):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                baseline_allocated = int(torch.cuda.memory_allocated(device))
                baseline_reserved = int(torch.cuda.memory_reserved(device))
                torch.cuda.reset_peak_memory_stats(device)
                start_event = torch.cuda.Event(enable_timing=True)
                stop_event = torch.cuda.Event(enable_timing=True)
                wall_started = time.perf_counter()
                start_event.record()
                output = function()
                stop_event.record()
                stop_event.synchronize()
                wall_ms = 1000.0 * (time.perf_counter() - wall_started)
                cuda_ms = float(start_event.elapsed_time(stop_event))
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
            else:
                baseline_allocated = baseline_reserved = None
                started = time.perf_counter()
                output = function()
                wall_ms = 1000.0 * (time.perf_counter() - started)
                cuda_ms = None
                peak_allocated = peak_reserved = None
            del output
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append(
                {
                    "repetition": repetition,
                    "status": "complete",
                    "error": None,
                    "wall_ms": wall_ms,
                    "cuda_event_ms": cuda_ms,
                    "baseline_allocated_bytes": baseline_allocated,
                    "baseline_reserved_bytes": baseline_reserved,
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

    wall_values = np.asarray([sample["wall_ms"] for sample in samples], dtype=np.float64)
    cuda_values = np.asarray(
        [sample["cuda_event_ms"] for sample in samples if sample["cuda_event_ms"] is not None],
        dtype=np.float64,
    )
    allocated_values = [
        value for value in (sample["incremental_peak_allocated_bytes"] for sample in samples) if value is not None
    ]
    reserved_values = [
        value for value in (sample["incremental_peak_reserved_bytes"] for sample in samples) if value is not None
    ]
    return {
        "status": "complete",
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "samples": samples,
        "median_wall_ms": float(np.median(wall_values)),
        "p95_wall_ms": float(np.percentile(wall_values, 95)),
        "median_cuda_event_ms": None if not len(cuda_values) else float(np.median(cuda_values)),
        "peak_incremental_allocated_bytes": None if not allocated_values else int(max(allocated_values)),
        "peak_incremental_reserved_bytes": None if not reserved_values else int(max(reserved_values)),
    }


def _ledger(output: Mapping[str, Any]) -> dict[str, Any]:
    """Extract scalar final-P2 executor ledgers from a maps-on probe."""

    aux = output.get("interaction_aux", {})
    if not isinstance(aux, Mapping):
        aux = {}
    names = (
        "module_logical_paths",
        "module_unique_pairs",
        "module_support_rows",
        "module_fine_rows_forward",
        "module_fine_rows_padded",
        "module_executor_selected",
        "environment_logical_paths",
        "environment_unique_pairs",
        "environment_support_rows",
        "environment_fine_rows_forward",
        "environment_fine_rows_padded",
        "environment_geometry_rows_forward",
        "environment_content_dot_rows_forward",
        "environment_executor_selected",
    )
    result: dict[str, Any] = {}
    for name in names:
        prefix, field = name.split("_", 1)
        key = f"group_control_{prefix}_{field}"
        result[key] = aux.get(key)
    return result


def _prediction(output: Mapping[str, Any]) -> Any:
    field = output["pred_field"]
    return field.detach().float().cpu()


def _application_evaluator(
    predict_case: Any,
    model: Any,
    sample: Mapping[str, Any],
    device: Any,
    receiver_chunk: int,
) -> Any:
    """Run the application evaluator, including its CPU serialization scope."""

    return predict_case(
        model,
        dict(sample),
        device,
        query_batch_size=int(receiver_chunk),
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.0,
        return_routing_maps=False,
        return_topology_signature=False,
        return_prepared_state=False,
    )


def _mode_row(
    *,
    torch: Any,
    stage3: Any,
    predict_case: Any,
    model: Any,
    backend: Any,
    mode: str,
    sample: Mapping[str, Any],
    batch: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    query: Any,
    device: Any,
    receiver_chunk: int,
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, Any], Any]:
    _configure(backend, mode)
    with torch.inference_mode(), stage3._runtime_receiver_chunk_size(model, receiver_chunk):
        prepared = _prepare(model, batch, kwargs)
        full_timing = _measure(
            torch,
            device,
            lambda: _forward(
                model,
                batch,
                kwargs,
                return_routing_maps=False,
                return_prepared_state=False,
            ),
            warmups,
            repetitions,
        )
        prepared_timing = _measure(
            torch,
            device,
            lambda: _decode_prepared(model, prepared, query, receiver_chunk),
            warmups,
            repetitions,
        )
        evaluator_timing = _measure(
            torch,
            device,
            lambda: _application_evaluator(predict_case, model, sample, device, receiver_chunk),
            warmups,
            repetitions,
        )

        parity_output = _forward(
            model,
            batch,
            kwargs,
            return_routing_maps=False,
            return_prepared_state=False,
        )
        parity_prediction = _prediction(parity_output)
        del parity_output
        ledger_output = _forward(
            model,
            batch,
            kwargs,
            return_routing_maps=True,
            return_prepared_state=False,
        )
        ledger = _ledger(ledger_output)
        del ledger_output
        prepared_output = _decode_prepared(model, prepared, query, receiver_chunk)
        prepared_prediction = _prediction(prepared_output)
        del prepared_output
        output_device = str(query.device)
        prepared_shape = list(query.shape[:2]) + [int(model.config.field_dim)]
        application_query_count = int(np.asarray(sample["x_grid"]).size)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        {
            "full_gpu_forward": full_timing,
            "prepared_p2_decode": prepared_timing,
            "application_evaluator": evaluator_timing,
            "full_forward_ledger": ledger,
            "ledger_probe_maps": True,
            "ledger_probe_timed": False,
            "decode_output_device": output_device,
            "prepared_decode_output_shape": prepared_shape,
            "application_query_count": application_query_count,
            "prepared_vs_full_max_abs": float((prepared_prediction - parity_prediction).abs().max()),
            "prepared_vs_full_mean_abs": float((prepared_prediction - parity_prediction).abs().mean()),
        },
        parity_prediction,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch, run1405, stage3, make_batch, predict_case, select_sample = _imports()
    specs = stage3.parse_checkpoint_specs(args.checkpoint)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index != 1:
        raise ValueError("this Run-1501 benchmark is restricted to cuda:1")
    torch.cuda.set_device(device)
    receiver_chunk = int(args.receiver_chunk_size)
    if receiver_chunk <= 0:
        raise ValueError("receiver chunk size must be positive")
    rows: list[dict[str, Any]] = []
    for spec in specs:
        model, checkpoint_payload = stage3._load_model_spec(spec, device)
        architecture = str(model.config.core_honf.forward_architecture)
        if architecture != "sparse_incidence_group_control_honf":
            raise RuntimeError(f"checkpoint architecture is {architecture!r}")
        dataset, _dataset_path = stage3._load_dataset(
            checkpoint_payload,
            argparse.Namespace(dataset=args.dataset, split=args.split),
        )
        backend = model.core.backend
        original = (
            backend.executor_policy,
            backend.diagnostic_executor_independent,
            backend.ledger_rectangular_rows,
        )
        try:
            for case_id in args.case_id:
                sample = select_sample(dataset, str(case_id), 0)
                query_np = _query_points(sample, int(args.query_count))
                batch = make_batch(dict(sample), query_np, device)
                query = batch["query_xy"]
                kwargs = run1405._phase_forward_kwargs(batch)
                mode_rows: dict[str, dict[str, Any]] = {}
                predictions: dict[str, Any] = {}
                for mode in EXECUTOR_MODES:
                    mode_row, prediction = _mode_row(
                        torch=torch,
                        stage3=stage3,
                        predict_case=predict_case,
                        model=model,
                        backend=backend,
                        mode=mode,
                        sample=sample,
                        batch=batch,
                        kwargs=kwargs,
                        query=query,
                        device=device,
                        receiver_chunk=receiver_chunk,
                        warmups=int(args.warmups),
                        repetitions=int(args.repetitions),
                    )
                    mode_rows[mode] = mode_row
                    predictions[mode] = prediction
                delta = (predictions["support_blocks"] - predictions["rectangular"]).abs()
                rectangular = mode_rows["rectangular"]
                support = mode_rows["support_blocks"]
                rows.append(
                    {
                        "checkpoint_label": spec.label,
                        "checkpoint": str(spec.path),
                        "checkpoint_epoch": int(
                            checkpoint_payload.get("epoch", checkpoint_payload.get("current_epoch", -1))
                        ),
                        "case_id": str(case_id),
                        "query_count": int(args.query_count),
                        "receiver_chunk_size": receiver_chunk,
                        "device": str(device),
                        "dtype": str(query.dtype),
                        "maps": False,
                        "modes": mode_rows,
                        "prediction_max_abs_difference": float(delta.max()),
                        "prediction_mean_abs_difference": float(delta.mean()),
                        "support_blocks_over_rectangular": {
                            scope: float(support[scope]["median_wall_ms"] / rectangular[scope]["median_wall_ms"])
                            for scope in (
                                "full_gpu_forward",
                                "prepared_p2_decode",
                                "application_evaluator",
                            )
                        },
                        "support_blocks_over_rectangular_cuda_event": {
                            scope: (
                                None
                                if support[scope]["median_cuda_event_ms"] is None
                                or rectangular[scope]["median_cuda_event_ms"] is None
                                else float(
                                    support[scope]["median_cuda_event_ms"] / rectangular[scope]["median_cuda_event_ms"]
                                )
                            )
                            for scope in (
                                "full_gpu_forward",
                                "prepared_p2_decode",
                                "application_evaluator",
                            )
                        },
                    }
                )
        finally:
            (
                backend.executor_policy,
                backend.diagnostic_executor_independent,
                backend.ledger_rectangular_rows,
            ) = original
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    max_delta = max(float(row["prediction_max_abs_difference"]) for row in rows)
    return {
        "schema_version": 2,
        "task": "run1501_exact_support_blocks_scopes",
        "status": "complete",
        "device": str(device),
        "anchors": [str(case_id) for case_id in args.case_id],
        "checkpoints": {spec.label: str(spec.path) for spec in specs},
        "query_count": int(args.query_count),
        "receiver_chunk_size": receiver_chunk,
        "warmups": int(args.warmups),
        "repetitions": int(args.repetitions),
        "timed_maps": False,
        "synchronized_timing": {
            "wall_clock": "time.perf_counter around each call with CUDA stop-event synchronization",
            "cuda_event": "torch.cuda.Event start/stop with stop-event synchronization",
        },
        "rows": rows,
        "max_prediction_abs_difference": max_delta,
        "operator_preserved_within_5e-5": bool(max_delta <= 5.0e-5),
        "notes": [
            "Run 1501 active maturation was not inspected or monitored.",
            "Rectangular is the historical reference; support_blocks uses observed query signatures and unique source unions.",
            "Full forward and prepared decode outputs remain on GPU during timed calls; application evaluator includes its CPU transfer scope.",
            "Ledgers come from separate untimed maps-on probes and are not included in timing medians.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeat for each trusted checkpoint, using explicit LABEL=PATH syntax",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--query-count", type=int, default=8192)
    parser.add_argument("--receiver-chunk-size", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--device", default="cuda:1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.case_id:
        args.case_id = ["0273", "0653"]
    payload = run(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "run"]
