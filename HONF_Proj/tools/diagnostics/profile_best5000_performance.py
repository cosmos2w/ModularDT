"""Profile best-before-5000 HONF checkpoints on one physical CUDA device.

This diagnostic is deliberately disposable: it loads the explicit ``best_field``
checkpoints, runs evaluation and real predicted-port optimizer steps, and writes
JSON/CSV/PNG evidence without allocating a managed run or writing model state.
The four architectures are measured in one process, one model at a time, with
``CUDA_VISIBLE_DEVICES`` selected by the caller (physical GPU 1 is the intended
protocol for the 2026-09-20 comparison).

The memory trace uses the maintained loss assembly and optimizer path for
repeated M12 updates.  It records allocator state before/after forward,
backward, optimizer, and zero-grad boundaries and summarizes compact routing
support/degree tensors when the normal output exposes them.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_ROOT = PROJECT_ROOT / "tools" / "diagnostics"
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel" / "src", DIAGNOSTIC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


DEFAULT_RUN_ROOT = PROJECT_ROOT / "Trained_Results" / "ThermalChannel" / "HONF_Forward_Runs"
DEFAULT_OUTPUT = PROJECT_ROOT / "diagnostics" / "generated" / "run1404_1406_1407_1804_best5000_performance_20260920"
DEFAULT_CASES = ("0273", "0653")
DEFAULT_LABELS = ("1404", "1406", "1407", "1804")
ARCHITECTURE_NAMES = {
    "1404": "routing_only_pairwise",
    "1406": "group_control_pairwise_honf",
    "1407": "phase_shared_group_control_honf",
    "1804": "dense_pairwise_field",
}
CHECKPOINT_DEFAULTS = {
    "1404": DEFAULT_RUN_ROOT / "Run_1404_20260916_092508_routing_only_pairwise" / "checkpoints" / "best_field.pt",
    "1406": DEFAULT_RUN_ROOT / "Run_1406_20260919_132837_low_dimensional_group_control_executor_optimized_rerun" / "checkpoints" / "best_field.pt",
    "1407": DEFAULT_RUN_ROOT / "Run_1407_20260919_174751_phase_shared_prototype_group_control" / "checkpoints" / "best_field.pt",
    "1804": DEFAULT_RUN_ROOT / "Run_1804_20260905_081349_dense_pairwise_field_adaptation" / "checkpoints" / "best_field.pt",
}


def _jsonable(value: Any) -> Any:
    """Convert tensors/NumPy values and non-finite scalars to JSON values."""

    try:
        import torch
    except ImportError:  # pragma: no cover - runtime diagnostic always has torch
        torch = None
    if torch is not None and torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    values = {
        "1404": args.checkpoint_1404 or CHECKPOINT_DEFAULTS["1404"],
        "1406": args.checkpoint_1406 or CHECKPOINT_DEFAULTS["1406"],
        "1407": args.checkpoint_1407 or CHECKPOINT_DEFAULTS["1407"],
        "1804": args.checkpoint_1804 or CHECKPOINT_DEFAULTS["1804"],
    }
    result: dict[str, Path] = {}
    for label, raw in values.items():
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Run {label} checkpoint does not exist: {path}")
        result[label] = path
    return result


def _sync(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(torch: Any, device: Any) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "active_bytes": None,
            "inactive_split_bytes": None,
            "alloc_retries": None,
            "oom_count": None,
        }
    stats = torch.cuda.memory_stats(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "active_bytes": int(stats.get("active_bytes.all.current", 0)),
        "inactive_split_bytes": int(stats.get("inactive_split_bytes.all.current", 0)),
        "alloc_retries": int(stats.get("num_alloc_retries", 0)),
        "oom_count": int(stats.get("num_ooms", 0)),
    }


def _measure_call(torch: Any, device: Any, fn: Callable[[], Any], *, warmups: int, repetitions: int) -> dict[str, Any]:
    """Measure synchronized wall time and absolute/incremental allocator peaks."""

    with torch.inference_mode():
        for _ in range(int(warmups)):
            value = fn()
            del value
    _sync(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baseline = _memory_snapshot(torch, device)
    samples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for _ in range(int(repetitions)):
            _sync(torch, device)
            started = time.perf_counter()
            value = fn()
            _sync(torch, device)
            elapsed = time.perf_counter() - started
            peak = _memory_snapshot(torch, device)
            samples.append(
                {
                    "elapsed_seconds": float(elapsed),
                    "baseline_allocated_bytes": baseline["allocated_bytes"],
                    "baseline_reserved_bytes": baseline["reserved_bytes"],
                    "peak_allocated_bytes": peak["peak_allocated_bytes"],
                    "peak_reserved_bytes": peak["peak_reserved_bytes"],
                    "incremental_peak_allocated_bytes": (
                        None
                        if peak["peak_allocated_bytes"] is None or baseline["allocated_bytes"] is None
                        else peak["peak_allocated_bytes"] - baseline["allocated_bytes"]
                    ),
                    "incremental_peak_reserved_bytes": (
                        None
                        if peak["peak_reserved_bytes"] is None or baseline["reserved_bytes"] is None
                        else peak["peak_reserved_bytes"] - baseline["reserved_bytes"]
                    ),
                    "post_call_allocated_bytes": peak["allocated_bytes"],
                    "post_call_reserved_bytes": peak["reserved_bytes"],
                }
            )
            del value
    values = [row["elapsed_seconds"] for row in samples]
    result = {
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "samples": samples,
        "median_seconds": float(statistics.median(values)),
        "mean_seconds": float(statistics.mean(values)),
        "min_seconds": float(min(values)),
        "max_seconds": float(max(values)),
        "spread_seconds": float(max(values) - min(values)),
    }
    for key in (
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "incremental_peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
    ):
        observed = [row[key] for row in samples if row[key] is not None]
        result[f"median_{key}"] = None if not observed else float(statistics.median(observed))
    return result


class EventRecorder:
    """Low-overhead CUDA-event aggregation for one detailed forward pass."""

    def __init__(self, torch: Any, device: Any):
        self.torch = torch
        self.device = device
        self.events: dict[str, list[tuple[Any, Any]]] = defaultdict(list)

    def call(self, label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self.device.type != "cuda":
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                self.events[label].append((elapsed, None))
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return fn(*args, **kwargs)
        finally:
            end.record()
            self.events[label].append((start, end))

    def resolve(self) -> dict[str, dict[str, Any]]:
        _sync(self.torch, self.device)
        result: dict[str, dict[str, Any]] = {}
        for label, pairs in self.events.items():
            values: list[float] = []
            for start, end in pairs:
                if end is None:
                    values.append(float(start) * 1000.0)
                else:
                    values.append(float(start.elapsed_time(end)))
            result[label] = {
                "calls": len(values),
                "total_ms": float(sum(values)),
                "mean_ms": float(statistics.mean(values)),
                "samples_ms": values,
            }
        return result


class PatchStack:
    """Temporarily replace bound instance methods for event timing."""

    def __init__(self):
        self._saved: list[tuple[Any, str, Any]] = []

    def method(self, obj: Any, name: str, recorder: EventRecorder, label_fn: Callable[[], str] | str) -> None:
        original = getattr(obj, name)
        self._saved.append((obj, name, original))

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            label = label_fn() if callable(label_fn) else str(label_fn)
            return recorder.call(label, original, *args, **kwargs)

        setattr(obj, name, wrapped)

    def restore(self) -> None:
        for obj, name, original in reversed(self._saved):
            setattr(obj, name, original)
        self._saved.clear()


def _current_role(model: Any) -> str:
    return str(getattr(getattr(model, "core", None), "_interface_read_role", "unmarked"))


def _install_detailed_wrappers(model: Any, recorder: EventRecorder) -> PatchStack:
    """Instrument semantic boundaries without changing timed calls."""

    patches = PatchStack()
    architecture = str(getattr(getattr(model.config, "core_honf", None), "forward_architecture", ""))
    if architecture == "legacy_honf":
        patches.method(model.core, "encode_and_organize", recorder, "encode_organize")
        patches.method(model.core, "decode_queries", recorder, "legacy_decode")
        patches.method(model.core.decoder, "forward", recorder, "legacy_decoder")
        if getattr(model.core.decoder, "pairwise_kernel", None) is not None:
            patches.method(model.core.decoder.pairwise_kernel, "forward", recorder, "legacy_pairwise")
        return patches

    patches.method(model.core, "encode_case", recorder, "encode")
    patches.method(model.core, "prepare", recorder, lambda: f"prepare_{_current_role(model)}")
    patches.method(model.core, "read", recorder, lambda: f"read_{_current_role(model)}")
    patches.method(model.core, "decode_queries", recorder, lambda: f"decode_{_current_role(model)}")
    backend = model.core.backend
    for name, prefix in (
        ("prepare", "backend_prepare"),
        ("read", "backend_read"),
        ("_route", "route"),
        ("_read_module", "QM"),
        ("_read_environment", "QE"),
    ):
        if hasattr(backend, name):
            patches.method(backend, name, recorder, lambda prefix=prefix: f"{prefix}_{_current_role(model)}")
    for name, label in (
        ("call_local_surrogate", "local_surrogate"),
        ("fuse_module_state", "local_fusion"),
        ("assemble_interface", "local_assemble"),
        ("local_response_summary", "local_response_summary"),
    ):
        if hasattr(model.local_coupling, name):
            patches.method(model.local_coupling, name, recorder, label)
    for module, label in (
        (getattr(model.local_coupling, "port_head", None), "port_head"),
        (getattr(model.local_coupling, "port_refinement_head", None), "port_refinement_head"),
    ):
        if module is not None and hasattr(module, "forward"):
            patches.method(module, "forward", recorder, label)
    return patches


def _summary_stats(value: Any, torch: Any) -> dict[str, float] | None:
    if not torch.is_tensor(value) or value.numel() == 0:
        return None
    tensor = value.detach().float()
    if tensor.numel() > 2_000_000:
        # The trace is intended to be compact; avoid copying large diagnostic maps.
        return {"numel": float(tensor.numel())}
    return {
        "numel": float(tensor.numel()),
        "mean": float(tensor.mean().cpu()),
        "min": float(tensor.amin().cpu()),
        "max": float(tensor.amax().cpu()),
        "nonzero_fraction": float((tensor != 0).float().mean().cpu()),
    }


def _routing_summary(output: Any, torch: Any) -> dict[str, Any]:
    """Extract compact route/support evidence from a normal (untimed) output."""

    if not isinstance(output, Mapping):
        return {}
    sources: list[tuple[str, Mapping[str, Any]]] = []
    for root in ("interaction_aux", "routing_aux", "provisional_interaction_aux"):
        value = output.get(root)
        if isinstance(value, Mapping):
            sources.append((root, value))
    selected: dict[str, Any] = {}
    tokens = ("route", "support", "degree", "pair", "row", "sparse", "incidence", "path", "group", "active", "execut")
    for root, source in sources:
        for key, value in source.items():
            name = f"{root}.{key}"
            if not any(token in str(key).lower() for token in tokens):
                continue
            summary = _summary_stats(value, torch)
            if summary is not None:
                selected[name] = summary
            elif isinstance(value, str):
                selected[name] = value
    return selected


def _load_runtime() -> tuple[Any, Any, Any, Any, Any]:
    from run_run1405_epoch50_comparison import _load_runtime_modules

    return _load_runtime_modules()


def _load_checkpoints(paths: Mapping[str, Path]) -> dict[str, Mapping[str, Any]]:
    from honf_runtime.compat import load_trusted_checkpoint

    return {label: load_trusted_checkpoint(path, map_location="cpu") for label, path in paths.items()}


def _model_parameters(model: Any) -> tuple[int, int]:
    parameters = list(model.parameters())
    return sum(int(p.numel()) for p in parameters), sum(int(p.numel()) for p in parameters if p.requires_grad)


def _prepare_case(torch: Any, stage3: Any, select_sample: Any, make_batch: Any, model: Any, dataset: Any, case_id: str, args: argparse.Namespace, device: Any) -> tuple[dict[str, Any], dict[str, Any], Any]:
    sample = select_sample(dataset, str(case_id), 0)
    query = stage3._query_points(sample, int(args.query_count))
    batch = make_batch(dict(sample), query, device)
    kwargs = {
        "interface_condition": batch.get("interface_condition"),
        "local_module_params": batch.get("local_module_params"),
        "teacher_port_tokens": batch.get("teacher_port_tokens"),
        "local_query_points": batch.get("module_internal_query_points"),
        "local_port_condition_mode": "predicted",
        "mixed_teacher_ratio": 0.0,
    }
    return batch, kwargs, sample


def _benchmark_inference(label: str, model: Any, checkpoint: Mapping[str, Any], dataset: Any, args: argparse.Namespace, torch: Any, dynamic: Any, stage3: Any, select_sample: Any, make_batch: Any, device: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    detailed_rows: dict[str, Any] = {}
    model.eval()
    total_parameters, trainable_parameters = _model_parameters(model)
    for case_id in args.case_id or DEFAULT_CASES:
        case_batch, forward_kwargs, sample = _prepare_case(torch, stage3, select_sample, make_batch, model, dataset, str(case_id), args, device)

        def full(_batch: Mapping[str, Any] = case_batch, _kwargs: Mapping[str, Any] = forward_kwargs) -> Any:
            return model(_batch["structure"], _batch["query_xy"], return_routing_maps=False, **_kwargs)

        def prep(_batch: Mapping[str, Any] = case_batch, _kwargs: Mapping[str, Any] = forward_kwargs) -> Any:
            return model(_batch["structure"], _batch["query_xy"][:, :1], return_prepared_state=True, return_routing_maps=False, **_kwargs)

        with dynamic._runtime_receiver_chunk_size(model, int(args.receiver_chunk_size)):
            full_measure = _measure_call(torch, device, full, warmups=args.inference_warmups, repetitions=args.inference_repetitions)
            prep_measure = _measure_call(torch, device, prep, warmups=args.inference_warmups, repetitions=args.inference_repetitions)
            with torch.inference_mode():
                prepared_output = prep()
            prepared = prepared_output["prepared_state"]

            def decode(_prepared: Any = prepared, _batch: Mapping[str, Any] = case_batch) -> Any:
                return model.decode_prepared(_prepared, _batch["query_xy"], return_routing_maps=False, receiver_chunk_size=int(args.receiver_chunk_size))

            decode_measure = _measure_call(torch, device, decode, warmups=args.inference_warmups, repetitions=args.inference_repetitions)
            del prepared_output, prepared

            recorder = EventRecorder(torch, device)
            patches = _install_detailed_wrappers(model, recorder)
            try:
                with torch.inference_mode():
                    detailed_output = full()
                _sync(torch, device)
                route_summary = _routing_summary(detailed_output, torch)
                del detailed_output
                detailed_rows[str(case_id)] = {
                    "events": recorder.resolve(),
                    "route_summary": route_summary,
                }
            finally:
                patches.restore()
        rows.extend(
            {
                "label": label,
                "architecture": ARCHITECTURE_NAMES[label],
                "case_id": str(case_id),
                "checkpoint_epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
                "query_count": int(case_batch["query_xy"].shape[1]),
                "receiver_chunk_size": int(args.receiver_chunk_size),
                "phase": phase,
                "median_ms": float(measurement["median_seconds"] * 1000.0),
                "mean_ms": float(measurement["mean_seconds"] * 1000.0),
                "spread_ms": float(measurement["spread_seconds"] * 1000.0),
                "median_peak_allocated_mib": None if measurement.get("median_peak_allocated_bytes") is None else float(measurement["median_peak_allocated_bytes"] / 2**20),
                "median_peak_reserved_mib": None if measurement.get("median_peak_reserved_bytes") is None else float(measurement["median_peak_reserved_bytes"] / 2**20),
                "median_incremental_peak_allocated_mib": None if measurement.get("median_incremental_peak_allocated_bytes") is None else float(measurement["median_incremental_peak_allocated_bytes"] / 2**20),
                "median_incremental_peak_reserved_mib": None if measurement.get("median_incremental_peak_reserved_bytes") is None else float(measurement["median_incremental_peak_reserved_bytes"] / 2**20),
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
            }
            for phase, measurement in {
                "full_forward": full_measure,
                "preparation_plus_one_query": prep_measure,
                "prepared_decode": decode_measure,
            }.items()
        )
        del case_batch, forward_kwargs, sample
        gc.collect()
    return rows, detailed_rows


def _flatten_route_stats(stats: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in stats.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            for child, child_value in _flatten_route_stats(value, name).items():
                flat[child] = child_value
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            flat[name] = float(value)
    return flat


def _trace_snapshot(torch: Any, device: Any, *, label: str, step: int, event: str, bucket: str | None = None, route_stats: Mapping[str, Any] | None = None) -> dict[str, Any]:
    _sync(torch, device)
    values = _memory_snapshot(torch, device)
    row: dict[str, Any] = {"label": label, "bucket": bucket, "step": int(step), "event": event, **values}
    if route_stats:
        for key, value in _flatten_route_stats(route_stats).items():
            row[f"route.{key}"] = value
    return row


def _install_trace_hooks(model: Any, optimizer: Any, torch: Any, device: Any, label: str, step_state: dict[str, Any], rows: list[dict[str, Any]]) -> list[tuple[Any, str, Any]]:
    saved: list[tuple[Any, str, Any]] = []
    original_forward = model.forward
    saved.append((model, "forward", original_forward))

    def forward_wrapped(*args: Any, **kwargs: Any) -> Any:
        step = int(step_state["step"])
        rows.append(_trace_snapshot(torch, device, label=label, bucket=step_state.get("bucket"), step=step, event="before_forward"))
        output = original_forward(*args, **kwargs)
        route = _routing_summary(output, torch)
        step_state["route_stats"] = route
        rows.append(_trace_snapshot(torch, device, label=label, bucket=step_state.get("bucket"), step=step, event="after_forward", route_stats=route))
        return output

    setattr(model, "forward", forward_wrapped)  # noqa: B010 - restore below in the same disposable process

    original_backward = torch.Tensor.backward
    saved.append((torch.Tensor, "backward", original_backward))

    def backward_wrapped(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        step = int(step_state["step"])
        rows.append(_trace_snapshot(torch, device, label=label, bucket=step_state.get("bucket"), step=step, event="before_backward", route_stats=step_state.get("route_stats")))
        result = original_backward(tensor, *args, **kwargs)
        rows.append(_trace_snapshot(torch, device, label=label, bucket=step_state.get("bucket"), step=step, event="after_backward", route_stats=step_state.get("route_stats")))
        return result

    torch.Tensor.backward = backward_wrapped

    original_step = optimizer.step
    saved.append((optimizer, "step", original_step))

    def optimizer_step_wrapped(*args: Any, **kwargs: Any) -> Any:
        step = int(step_state["step"])
        rows.append(_trace_snapshot(torch, device, label=label, bucket=step_state.get("bucket"), step=step, event="before_optimizer", route_stats=step_state.get("route_stats")))
        result = original_step(*args, **kwargs)
        rows.append(_trace_snapshot(torch, device, label=label, bucket=step_state.get("bucket"), step=step, event="after_optimizer", route_stats=step_state.get("route_stats")))
        return result

    optimizer.step = optimizer_step_wrapped
    original_zero = optimizer.zero_grad
    saved.append((optimizer, "zero_grad", original_zero))

    def zero_wrapped(*args: Any, **kwargs: Any) -> Any:
        step = int(step_state["step"])
        rows.append(_trace_snapshot(torch, device, label=label, bucket=step_state.get("bucket"), step=step, event="before_zero_grad", route_stats=step_state.get("route_stats")))
        result = original_zero(*args, **kwargs)
        rows.append(_trace_snapshot(torch, device, label=label, bucket=step_state.get("bucket"), step=step, event="after_zero_grad", route_stats=step_state.get("route_stats")))
        return result

    optimizer.zero_grad = zero_wrapped
    return saved


def _restore_trace_hooks(torch: Any, saved: Sequence[tuple[Any, str, Any]]) -> None:
    for obj, name, original in reversed(list(saved)):
        setattr(obj, name, original)


def _training_trace_for_model(label: str, model: Any, checkpoint: Mapping[str, Any], dataset: Any, bucket: Any, args: argparse.Namespace, torch: Any, train_bench: Any, device: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = train_bench._build_loader(dataset, checkpoint, bucket, batch_size=int(args.train_batch_size))
    schedule = train_bench._training_schedule(checkpoint)
    optimizer, inventory = train_bench.build_forward_optimizer(model, schedule[0])
    model.train()
    trace_rows: list[dict[str, Any]] = []
    state = {"step": 0, "route_stats": {}}
    state["bucket"] = bucket.label
    saved = _install_trace_hooks(model, optimizer, torch, device, label, state, trace_rows)
    timings: list[float] = []
    try:
        for _ in range(int(args.trace_warmups)):
            state["step"] = -1
            _sync(torch, device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_bench._run_step(model, loader, device, checkpoint, optimizer=optimizer, training_schedule=schedule)
            _sync(torch, device)
        for index in range(int(args.trace_steps)):
            state["step"] = index
            _sync(torch, device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            train_bench._run_step(model, loader, device, checkpoint, optimizer=optimizer, training_schedule=schedule)
            _sync(torch, device)
            timings.append(float(time.perf_counter() - started))
            trace_rows.append(_trace_snapshot(torch, device, label=label, bucket=bucket.label, step=index, event="after_step", route_stats=state.get("route_stats")))
    finally:
        _restore_trace_hooks(torch, saved)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, loader
        gc.collect()
    return {
        "label": label,
        "bucket": {"label": bucket.label, "module_count": int(bucket.module_count), "batch_size": int(args.train_batch_size), "query_points": int(args.train_query_count)},
        "checkpoint_epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "trace_steps": int(args.trace_steps),
        "warmup_steps": int(args.trace_warmups),
        "median_step_ms": float(statistics.median(timings) * 1000.0),
        "mean_step_ms": float(statistics.mean(timings) * 1000.0),
        "min_step_ms": float(min(timings) * 1000.0),
        "max_step_ms": float(max(timings) * 1000.0),
        "optimizer_inventory": inventory,
        "schedule": {"mode": schedule[2], "teacher_ratio": schedule[3], "internal_weight": schedule[4], "interface_weight": schedule[5], "predicted_weight": schedule[6]},
    }, trace_rows


def _trace_summary(rows: Sequence[Mapping[str, Any]], label: str, bucket: str | None = None) -> dict[str, Any]:
    selected = [row for row in rows if row.get("label") == label and (bucket is None or row.get("bucket") == bucket) and int(row.get("step", -1)) >= 0]
    summary: dict[str, Any] = {"label": label, "bucket": bucket, "steps": sorted({int(row["step"]) for row in selected})}
    for event in ("before_forward", "after_forward", "after_backward", "before_optimizer", "after_optimizer", "after_zero_grad", "after_step"):
        values = [row.get("allocated_bytes") for row in selected if row.get("event") == event and row.get("allocated_bytes") is not None]
        reserved = [row.get("reserved_bytes") for row in selected if row.get("event") == event and row.get("reserved_bytes") is not None]
        if values:
            summary[event] = {
                "allocated_min_mib": min(values) / 2**20,
                "allocated_max_mib": max(values) / 2**20,
                "allocated_mean_mib": statistics.mean(values) / 2**20,
                "reserved_min_mib": min(reserved) / 2**20 if reserved else None,
                "reserved_max_mib": max(reserved) / 2**20 if reserved else None,
                "reserved_mean_mib": statistics.mean(reserved) / 2**20 if reserved else None,
                "unique_allocated_values": len(set(values)),
                "unique_reserved_values": len(set(reserved)),
            }
            peak_allocated = [row.get("peak_allocated_bytes") for row in selected if row.get("event") == event and row.get("peak_allocated_bytes") is not None]
            peak_reserved = [row.get("peak_reserved_bytes") for row in selected if row.get("event") == event and row.get("peak_reserved_bytes") is not None]
            summary[event].update(
                {
                    "peak_allocated_max_mib": max(peak_allocated) / 2**20 if peak_allocated else None,
                    "peak_reserved_max_mib": max(peak_reserved) / 2**20 if peak_reserved else None,
                }
            )
    after_steps = [row for row in selected if row.get("event") == "after_step"]
    if after_steps:
        allocated = [row.get("allocated_bytes") for row in after_steps if row.get("allocated_bytes") is not None]
        reserved = [row.get("reserved_bytes") for row in after_steps if row.get("reserved_bytes") is not None]
        summary["plateaus"] = {
            "after_step_allocated_change_count": sum(a != b for a, b in pairwise(allocated)),
            "after_step_reserved_change_count": sum(a != b for a, b in pairwise(reserved)),
            "after_step_allocated_unique": len(set(allocated)),
            "after_step_reserved_unique": len(set(reserved)),
        }
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _render_plots(output: Path, inference_rows: Sequence[Mapping[str, Any]], training_rows: Sequence[Mapping[str, Any]], phase_rows: Sequence[Mapping[str, Any]], trace_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(DEFAULT_LABELS)
    colors = {"1404": "#4C78A8", "1406": "#F2A541", "1407": "#E45756", "1804": "#303030"}
    paths: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    phases = ["full_forward", "preparation_plus_one_query", "prepared_decode"]
    x = np.arange(len(labels))
    width = 0.24
    for index, phase in enumerate(phases):
        values = []
        for label in labels:
            rows = [r for r in inference_rows if r.get("label") == label and r.get("phase") == phase]
            values.append(float(statistics.mean([r["median_ms"] for r in rows])) if rows else np.nan)
        axes[0].bar(x + (index - 1) * width, values, width, label=phase.replace("_", " "), color=[colors[l] for l in labels], alpha=0.45 + 0.2 * index, edgecolor="white")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("milliseconds (lower is better)")
    axes[0].set_title("Inference cost across best-field checkpoints")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].legend(fontsize=8)
    train_phases = ["M1", "M12"]
    for index, bucket in enumerate(train_phases):
        values = [next((float(r["median_step_ms"]) for r in training_rows if r.get("label") == label and r.get("bucket", {}).get("label") == bucket), np.nan) for label in labels]
        axes[1].bar(x + (index - 0.5) * width, values, width, label=bucket, color=[colors[l] for l in labels], alpha=0.65 + 0.15 * index, edgecolor="white")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("milliseconds per optimizer update")
    axes[1].set_title("Disposable predicted-port training step")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[1].legend(fontsize=8)
    path = output / "performance_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    # Detail bars are intentionally labelled as inclusive/nested events in the JSON;
    # they are diagnostic attribution, not additive wall-time accounting.
    detail_names = ["encode", "prepare_p0_port", "prepare_p1_refinement", "prepare_p2_field", "read_p0_port", "read_p1_refinement", "read_p2_field", "read_p2_port_global_consistency", "local_surrogate", "local_fusion", "QM_p2_field", "QE_p2_field"]
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    xpos = np.arange(len(detail_names))
    width = 0.19
    for idx, label in enumerate(labels):
        case_rows = [r for r in phase_rows if r.get("label") == label]
        values = []
        for name in detail_names:
            values.append(float(statistics.mean([r["total_ms"] for r in case_rows if r.get("event") == name])) if any(r.get("event") == name for r in case_rows) else np.nan)
        ax.bar(xpos + (idx - 1.5) * width, values, width, label=label, color=colors[label], alpha=0.9)
    ax.set_xticks(xpos, [name.replace("_", "\n") for name in detail_names], rotation=0, fontsize=8)
    ax.set_ylabel("CUDA-event milliseconds; nested regions not additive")
    ax.set_title("Semantic forward attribution where architecture exposes the boundary")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(title="run")
    path = output / "phase_breakdown.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for label in labels:
        selected = [r for r in trace_rows if r.get("label") == label and r.get("bucket") == "M12" and r.get("event") == "after_step"]
        selected.sort(key=lambda r: int(r.get("step", 0)))
        if not selected:
            continue
        steps = [r["step"] for r in selected]
        axes[0].plot(steps, [r["allocated_bytes"] / 2**30 for r in selected], label=label, color=colors[label], linewidth=1.8)
        axes[1].plot(steps, [r["reserved_bytes"] / 2**30 for r in selected], label=label, color=colors[label], linewidth=1.8)
    axes[0].set_ylabel("allocated GiB")
    axes[1].set_ylabel("reserved GiB")
    axes[1].set_xlabel("M12 update step (after optimizer/step cleanup)")
    axes[0].set_title("Allocator trajectory over consecutive real M12 predicted-port updates")
    for ax in axes:
        ax.grid(color="#dddddd", linewidth=0.7)
        ax.legend(ncol=4, fontsize=8)
    path = output / "memory_dynamics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in DEFAULT_LABELS:
        parser.add_argument(f"--checkpoint-{label}", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0", help="visible CUDA device; use CUDA_VISIBLE_DEVICES=1 for physical GPU 1")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-count", type=int, default=8192)
    parser.add_argument("--receiver-chunk-size", type=int, default=2048)
    parser.add_argument("--inference-warmups", type=int, default=2)
    parser.add_argument("--inference-repetitions", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=48)
    parser.add_argument("--train-query-count", type=int, default=1024)
    parser.add_argument("--trace-warmups", type=int, default=1)
    parser.add_argument("--trace-steps", type=int, default=64)
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--trace-only", action="store_true", help="skip inference and run only the disposable training trace")
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.query_count <= 0 or args.receiver_chunk_size <= 0 or args.train_batch_size <= 0 or args.train_query_count <= 0:
        raise ValueError("query, chunk, batch, and training query sizes must be positive")
    if args.inference_warmups < 0 or args.inference_repetitions <= 0 or args.trace_warmups < 0 or args.trace_steps <= 0:
        raise ValueError("warmups must be nonnegative and repetitions/trace steps positive")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = _checkpoint_paths(args)
    torch, train_bench, dynamic, stage3, loaders = _load_runtime()
    make_batch, select_sample = loaders
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for {device}; caller should set CUDA_VISIBLE_DEVICES")
    checkpoints = _load_checkpoints(paths)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "task": "best_before_5000_performance_and_memory_profile",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "checkpoint_policy": "explicit checkpoints/best_field.pt; checkpoint epoch is the best validation-field epoch before terminal 5000",
        "protocol": {
            "labels": list(DEFAULT_LABELS),
            "case_ids": list(args.case_id or DEFAULT_CASES),
            "query_count": int(args.query_count),
            "receiver_chunk_size": int(args.receiver_chunk_size),
            "inference_warmups": int(args.inference_warmups),
            "inference_repetitions": int(args.inference_repetitions),
            "training_batch_size": int(args.train_batch_size),
            "training_query_count": int(args.train_query_count),
            "training_trace_warmups": int(args.trace_warmups),
            "training_trace_steps": int(args.trace_steps),
            "training_port_condition": "predicted",
            "optimizer_state_policy": "fresh disposable optimizer; no checkpoint or model writeback",
        },
        "checkpoints": {
            label: {
                "path": str(path),
                "epoch": int(checkpoints[label].get("epoch", checkpoints[label].get("current_epoch", -1))),
                "best_metric": checkpoints[label].get("best_metric"),
                "best_metrics": checkpoints[label].get("best_metrics", {}),
                "architecture": ARCHITECTURE_NAMES[label],
            }
            for label, path in paths.items()
        },
        "limitations": [
            "Inference phase rows are synchronized unprofiled calls; detailed CUDA-event regions are a separate untimed diagnostic pass and are nested, not additive.",
            "Run 1404 is the legacy routing-only architecture and does not expose P0/P1/P2 interface-field preparation/read boundaries; its legacy encode/decode/pairwise regions are reported instead.",
            "Training trace updates parameters only in disposable memory and never saves them; repeated M12 batches use the maintained real predicted-port loss path.",
            "Allocator values are empirical for the selected GPU, driver, PyTorch build, batch, query, and receiver chunk; they do not establish hardware-independent cost.",
            "Routing/support tensors are learned interaction diagnostics, not physical causality; compact output summaries cannot replace full untimed topology maps.",
        ],
    }
    inference_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    training_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    datasets: dict[str, Any] = {}
    try:
        if not args.trace_only:
            for label in DEFAULT_LABELS:
                model, _ = stage3._load_model_spec(stage3.CheckpointSpec(label=label, path=paths[label]), device)
                dataset, _ = stage3._load_dataset(checkpoints[label], SimpleNamespace(dataset=None, split="test"))
                datasets[label] = dataset
                rows, detail = _benchmark_inference(label, model, checkpoints[label], dataset, args, torch, dynamic, stage3, select_sample, make_batch, device)
                inference_rows.extend(rows)
                detailed[label] = detail
                for case_id, case_detail in detail.items():
                    for event, event_row in case_detail.get("events", {}).items():
                        phase_rows.append({"label": label, "case_id": case_id, "event": event, **event_row})
                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None
                gc.collect()

        if not args.skip_trace:
            # The exact M1/M12 buckets are shared across all four datasets.
            train_datasets: dict[str, Any] = {}
            for label in DEFAULT_LABELS:
                train_datasets[label], _ = stage3._load_dataset(
                    checkpoints[label],
                    SimpleNamespace(dataset=None, split="train"),
                    points_per_case_override=int(args.train_query_count),
                    random_point_sampling_override=False,
                )
                train_datasets[label].include_grid = False
            from run_run1405_epoch50_comparison import _make_exact_training_buckets

            buckets = _make_exact_training_buckets([train_datasets[label] for label in DEFAULT_LABELS], batch_size=int(args.train_batch_size))
            for label in DEFAULT_LABELS:
                for bucket in buckets:
                    model, _ = stage3._load_model_spec(stage3.CheckpointSpec(label=label, path=paths[label]), device)
                    result, rows = _training_trace_for_model(label, model, checkpoints[label], train_datasets[label], bucket, args, torch, train_bench, device)
                    training_rows.append(result)
                    trace_rows.extend(rows)
                    del model
                    torch.cuda.empty_cache() if device.type == "cuda" else None
                    gc.collect()
            for dataset in train_datasets.values():
                close = getattr(dataset, "close", None)
                if callable(close):
                    close()
    finally:
        for dataset in datasets.values():
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
    metadata["inference_rows"] = inference_rows
    metadata["phase_breakdown_rows"] = phase_rows
    metadata["training_rows"] = training_rows
    metadata["memory_trace_summary"] = [
        _trace_summary(trace_rows, label, bucket)
        for label in DEFAULT_LABELS
        for bucket in ("M1", "M12")
    ]
    metadata["detailed_forward"] = detailed
    metadata["artifacts"] = {
        "summary_json": str(output / "summary.json"),
        "inference_csv": str(output / "inference_timings.csv"),
        "training_csv": str(output / "training_timings.csv"),
        "phase_csv": str(output / "phase_breakdown.csv"),
        "memory_csv": str(output / "memory_trace.csv"),
    }
    _write_json(output / "summary.json", metadata)
    _write_csv(output / "inference_timings.csv", inference_rows)
    _write_csv(output / "training_timings.csv", training_rows)
    _write_csv(output / "phase_breakdown.csv", phase_rows)
    _write_csv(output / "memory_trace.csv", trace_rows)
    if not args.skip_plots:
        metadata["plots"] = _render_plots(output, inference_rows, training_rows, phase_rows, trace_rows)
        _write_json(output / "summary.json", metadata)
    print(json.dumps(_jsonable({"status": "complete", "output": str(output), "artifacts": metadata["artifacts"], "plots": metadata.get("plots", [])}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
