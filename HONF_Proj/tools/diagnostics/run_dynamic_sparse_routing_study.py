#!/usr/bin/env python3
"""Bounded diagnostics for the Goal-1 module-hub routing candidate.

The driver is deliberately an evaluation and measurement tool.  It reuses
the regional/stage-3 loaders, the ordinary physical forward path, and the
existing comparison workflow.  In particular, it does not reserve a managed
run, write a checkpoint, or start a training process.  The ``smoke`` task is
the exception to evaluation-only execution in the narrow sense that it runs
the existing one-batch AdamW helper on a fresh, disposable model.

Run from ``HONF_Proj/``.  Checkpoints use explicit ``LABEL=PATH`` syntax::

    python tools/diagnostics/run_dynamic_sparse_routing_study.py profile \
        --checkpoint exact=/path/to/epoch_0500_model.pt \
        --case-id 0273 --trace --output diagnostics/profile.json

The output schema keeps routing summaries compact.  Detailed arrays are
written only by ``ledger --save-maps`` for the requested anchor cases.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import inspect
import json
import math
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    PROJECT_ROOT / "tools",
    PROJECT_ROOT / "tools" / "diagnostics",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "Case_ThermalChannel" / "src",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# These imports intentionally point at the maintained study APIs.  Keeping
# model loading and canonical errors in those modules prevents this driver
# from growing a second checkpoint or physical-target implementation.
from run_regional_response_study import (  # type: ignore[import-not-found]
    _canonical_ground_truth_errors,
    _error_deltas,
    _forward_batch,
    _load_dataset,
    _load_model_spec,
    _load_raw_sample,
    _measure_phase,
    _query_points,
    _relative_difference,
    _run_variant,
    _runtime_receiver_chunk_size,
    _state_keys,
    _synthetic_query_tensor,
    _synthetic_structure,
    common_read_zero,
    parse_checkpoint_specs,
    regular_synthetic_environment,
    select_device,
)
from run_stage3_interface_study import (  # type: ignore[import-not-found]
    _active_centers,
    _forward_tensor_batch,
    _scaled_domain,
    make_batch,
    temporary_domain,
)
from run_stage3_interface_study import (
    synthetic_environment as stage3_synthetic_environment,
)

ANCHOR_CASE_IDS = ("0273", "0653", "0283", "0298", "0302")
DEFAULT_PROFILE = "project://src/config_core/forward/routing_module_hubs_context.json"
DEFAULT_LARGE_SHAPE = (128, 3072, 262144)
DEFAULT_QUERY_BATCH_SIZE = 32768
DEFAULT_RECEIVER_CHUNK_SIZE = 2048
ROUTING_WEIGHT_SEMANTICS = (
    "learned routing weights; not physical influence or field-value substitutes"
)


class DiagnosticUnavailable(RuntimeError):
    """A requested diagnostic hook is absent from the current backend."""


def _json_value(value: Any) -> Any:
    """Convert scalar diagnostic values without serializing large tensors."""

    if torch.is_tensor(value):
        if value.numel() <= 256:
            if value.numel() == 1:
                return _json_value(value.detach().cpu().item())
            return value.detach().cpu().tolist()
        return {"tensor": _tensor_summary(value)}
    if isinstance(value, np.ndarray):
        if value.size <= 256:
            return value.tolist()
        return {"array": _array_summary(value)}
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _json_value(vars(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _map_value_to_cpu_array(value: Any) -> Any:
    """Detach tensor map values before NumPy inspection/export."""

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one compact diagnostic artifact beneath the requested path."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array) if np.issubdtype(array.dtype, np.floating) else np.ones(array.shape, dtype=bool)
    selected = array[finite]
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_count": int(finite.sum()),
        "count": int(array.size),
    }
    if selected.size and np.issubdtype(selected.dtype, np.number):
        result.update(
            {
                "min": float(np.min(selected)),
                "max": float(np.max(selected)),
                "mean": float(np.mean(selected)),
                "l2": float(np.linalg.norm(selected.reshape(-1))),
            }
        )
    return result


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "requires_grad": bool(value.requires_grad),
        "count": int(detached.numel()),
    }
    if detached.numel() == 0:
        result.update({"finite_count": 0, "nonzero_count": 0})
        return result
    if torch.is_floating_point(detached) or torch.is_complex(detached):
        finite = torch.isfinite(detached)
        selected = detached[finite]
        result["finite_count"] = int(finite.sum().cpu())
        result["nonzero_count"] = int((detached != 0).sum().cpu())
        if selected.numel():
            real = selected.real if torch.is_complex(selected) else selected
            result.update(
                {
                    "min": float(real.min().cpu()),
                    "max": float(real.max().cpu()),
                    "mean": float(real.mean().cpu()),
                    "l2": float(torch.linalg.vector_norm(real).cpu()),
                }
            )
    else:
        result["finite_count"] = int(detached.numel())
        result["nonzero_count"] = int((detached != 0).sum().cpu())
    return result


def _is_routing_key(key: Any) -> bool:
    text = str(key).lower()
    return any(
        token in text
        for token in (
            "routing",
            "route",
            "candidate",
            "hub",
            "incidence",
            "occupancy",
            "pair",
            "sparsemax",
            "membership",
            "support",
            "prior",
            "projection",
        )
    )


def _summarize_routing_mapping(value: Any, *, depth: int = 0) -> Any:
    """Summarize route-only mappings and preserve scalar metadata."""

    if torch.is_tensor(value):
        return _tensor_summary(value)
    if isinstance(value, np.ndarray):
        return _array_summary(value)
    if dataclasses.is_dataclass(value):
        if depth >= 3:
            return {"type": type(value).__name__}
        return _summarize_routing_mapping(vars(value), depth=depth + 1)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _is_routing_key(key) or not isinstance(item, (torch.Tensor, np.ndarray, Mapping)):
                result[str(key)] = _summarize_routing_mapping(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if depth >= 3:
            return {"type": type(value).__name__, "length": len(value)}
        return [_summarize_routing_mapping(item, depth=depth + 1) for item in value[:32]]
    return _json_value(value)


def _routing_containers(outputs: Mapping[str, Any]) -> dict[str, Any]:
    containers: dict[str, Any] = {}
    for name in ("routing_aux", "interaction_aux", "provisional_read_aux"):
        value = outputs.get(name)
        if isinstance(value, Mapping):
            containers[name] = value
    prepared = outputs.get("prepared_state")
    prepared_inner = getattr(prepared, "prepared", prepared)
    for name in ("interaction_aux", "diagnostics"):
        value = getattr(prepared_inner, name, None)
        if isinstance(value, Mapping):
            containers[f"prepared_{name}"] = value
    backend_state = getattr(prepared_inner, "backend_state", None)
    diagnostics = getattr(backend_state, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        containers["backend_diagnostics"] = diagnostics
    return containers


def summarize_routing_output(outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact routing summaries from a model output/prepared state."""

    return {
        "containers": {
            name: _summarize_routing_mapping(value)
            for name, value in _routing_containers(outputs).items()
        },
        "available": bool(_routing_containers(outputs)),
    }


def _routing_scalar_values(outputs: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for container_name, container in _routing_containers(outputs).items():
        if not isinstance(container, Mapping):
            continue
        for key, value in container.items():
            if not _is_routing_key(key):
                continue
            if torch.is_tensor(value) and value.numel() == 1:
                scalar = float(value.detach().cpu())
            elif isinstance(value, (int, float, np.integer, np.floating)):
                scalar = float(value)
            else:
                continue
            if math.isfinite(scalar):
                values[f"{container_name}.{key}"] = scalar
    return values


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _profile_args(args: argparse.Namespace) -> argparse.Namespace:
    """Adapt this driver's common options to the maintained stage-3 loader."""

    return SimpleNamespace(
        dataset=getattr(args, "dataset", None),
        split=getattr(args, "split", "test"),
        query_count=getattr(args, "query_count", 8192),
    )


def _build_fresh_profile_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any], Path]:
    """Construct a fresh profile model using the ordinary workflow config path."""

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

    bundle = load_config_bundle(str(args.profile))
    plugin = load_case_plugin(str(bundle.case["plugin"]))
    cfg = plugin._forward_config(
        bundle,
        WorkflowRequest(workflow="forward", device=str(device), epochs=500),
        Path(args.output).expanduser().resolve().parent,
    )
    train_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]
    set_seed(int(train_cfg["seed"]))
    dataset_path = Path(dataset_cfg["packed_h5_path"]).expanduser().resolve()
    dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=str(dataset_cfg["train_split"]),
        points_per_case=int(getattr(args, "points_per_case", 1)),
        normalize_inputs=bool(dataset_cfg["normalize_inputs"]),
        normalize_targets=bool(dataset_cfg["normalize_targets"]),
        random_point_sampling=False,
        seed=int(train_cfg["seed"]),
        require_converged=bool(dataset_cfg.get("require_converged", False)),
    )
    model_config = build_model_config(cfg, dataset)
    model = ChannelThermalHONFModel(model_config).to(device)
    model.set_global_target_normalization(
        dataset.normalizer.stats,
        normalize_targets=bool(dataset_cfg["normalize_targets"]),
    )
    resolve_auto_internal_mode(model_config, model)
    # Build and immediately discard the optimizer so lazy parameter modules
    # follow the same construction path as the real smoke check.  No state is
    # retained or written by this diagnostic.
    optimizer, _ = build_forward_optimizer(model, train_cfg)
    del optimizer
    model.eval()
    checkpoint = {
        "train_config": cfg,
        "epoch": 0,
        "global_normalization_stats": {
            key: np.asarray(value).tolist()
            for key, value in dataset.normalizer.stats.items()
        },
    }
    return model, dataset, checkpoint, dataset_path


def _load_profile_models(
    args: argparse.Namespace,
    device: torch.device,
    *,
    require_checkpoint: bool = False,
) -> list[tuple[str, Any, Any, dict[str, Any], Path]]:
    """Load explicit checkpoints or one fresh profile model."""

    raw_specs = list(getattr(args, "checkpoint", []) or [])
    if raw_specs:
        records = []
        for spec in parse_checkpoint_specs(raw_specs):
            model, checkpoint = _load_model_spec(spec, device)
            # Keep the provenance in the in-memory record so the profile
            # manifest is self-contained.  This is diagnostic metadata only;
            # the trusted checkpoint object is never written back.
            checkpoint["_path"] = str(spec.path)
            dataset, dataset_path = _load_dataset(checkpoint, _profile_args(args))
            records.append((spec.label, model, dataset, checkpoint, dataset_path))
        return records
    if require_checkpoint:
        raise ValueError("This diagnostic requires at least one --checkpoint LABEL=PATH.")
    model, train_dataset, checkpoint, dataset_path = _build_fresh_profile_model(args, device)
    checkpoint["_path"] = "fresh_profile_unsaved"
    # The model must be built from the profile's training statistics, while
    # fixed real anchors are read from the requested evaluation split.
    close = getattr(train_dataset, "close", None)
    if callable(close):
        close()
    dataset, dataset_path = _load_dataset(checkpoint, _profile_args(args))
    return [("fresh_profile", model, dataset, checkpoint, dataset_path)]


def _profile_forward_kwargs(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "interface_condition": batch.get("interface_condition"),
        "local_module_params": batch.get("local_module_params"),
        "teacher_port_tokens": batch.get("teacher_port_tokens"),
        "local_query_points": batch.get("module_internal_query_points"),
        "local_port_condition_mode": "predicted",
        "mixed_teacher_ratio": 0.0,
    }


def _active_only_dense_fine_reference(
    model: Any,
    batch: Mapping[str, Any],
    prepare_one: Callable[[], Mapping[str, Any]],
    device: torch.device,
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    """Time Dense's inherited QM reader with padded and active-only modules.

    The two measurements use one final routed prepared state and differ only
    in the module source width passed to the inherited dense fine reader.  The
    result is a padding reference, not an end-to-end architecture comparison;
    environmental sources and all common physical paths remain outside this
    narrow kernel measurement.
    """

    prepared_output: Mapping[str, Any] | None = None
    try:
        with torch.inference_mode():
            prepared_output = prepare_one()
        prepared_wrapper = prepared_output["prepared_state"]
        prepared = getattr(prepared_wrapper, "prepared", prepared_wrapper)
        encoded = getattr(prepared, "encoded", None)
        state = getattr(prepared, "backend_state", None)
        backend = getattr(getattr(model, "core", None), "backend", None)
        if encoded is None or not isinstance(state, Mapping) or backend is None:
            raise DiagnosticUnavailable("prepared routed state is unavailable for active-only reference")
        module_present = encoded.module_present > 0.5
        active_index = torch.nonzero(module_present[0], as_tuple=False).reshape(-1)
        if active_index.numel() == 0:
            raise DiagnosticUnavailable("active-only reference has no active physical modules")
        receivers = batch["query_xy"].float()
        receiver_features = model.core._receiver_features(prepared, receivers)
        module_state = state.get("module_tokens")
        if not torch.is_tensor(module_state):
            raise DiagnosticUnavailable("prepared module source states are unavailable")
        # The first real profile batch is a single case.  Keep a general
        # batched index when all cases share the same active slots; otherwise
        # report the reference as unavailable rather than silently mixing masks.
        if module_present.shape[0] > 1 and not torch.equal(
            module_present, module_present[:1].expand_as(module_present)
        ):
            raise DiagnosticUnavailable("active-only reference requires one shared active-slot mask")
        compact_encoded = dataclasses.replace(
            encoded,
            module_centers=encoded.module_centers.index_select(1, active_index),
            module_present=encoded.module_present.index_select(1, active_index),
            module_features=encoded.module_features.index_select(1, active_index),
        )
        compact_state = dict(state)
        compact_state["module_tokens"] = module_state.index_select(1, active_index)

        def padded_read() -> Any:
            return backend.read_module(state, encoded, receivers, receiver_features)

        def active_read() -> Any:
            return backend.read_module(compact_state, compact_encoded, receivers, receiver_features)

        # Validate the scientific premise of this timing reference before
        # timing either path: removing inactive padded slots must preserve
        # Dense's inherited QM context exactly.  Keep this comparison
        # inference-only and release both outputs before the memory-sensitive
        # measurements below.
        with torch.inference_mode():
            padded_value = padded_read()
            active_value = active_read()
        if not torch.is_tensor(padded_value) or not torch.is_tensor(active_value):
            raise DiagnosticUnavailable("Dense QM reference returned a non-tensor context")
        difference: torch.Tensor | None = None
        scale: torch.Tensor | None = None
        if tuple(padded_value.shape) != tuple(active_value.shape):
            agreement = {
                "allclose": False,
                "reason": "padded and active-only context shapes differ",
                "padded_shape": list(padded_value.shape),
                "active_only_shape": list(active_value.shape),
            }
        else:
            difference = (padded_value - active_value).abs()
            scale = active_value.abs().clamp_min(1.0e-12)
            agreement = {
                "allclose": bool(torch.allclose(
                    padded_value,
                    active_value,
                    atol=1.0e-6,
                    rtol=1.0e-5,
                )),
                "atol": 1.0e-6,
                "rtol": 1.0e-5,
                "max_abs": float(difference.max().detach().cpu()) if difference.numel() else 0.0,
                "max_relative": float((difference / scale).max().detach().cpu()) if difference.numel() else 0.0,
                "shape": list(padded_value.shape),
                "finite": bool(torch.isfinite(padded_value).all() and torch.isfinite(active_value).all()),
            }
        del padded_value, active_value
        if difference is not None:
            del difference, scale
        if not agreement["allclose"]:
            return {
                "status": "failed",
                "scope": "inherited_dense_QM_reader_only",
                "active_module_count": int(active_index.numel()),
                "padded_module_width": int(encoded.module_centers.shape[1]),
                "M_active": int(active_index.numel()),
                "Mpack": int(encoded.module_centers.shape[1]),
                "agreement": agreement,
                "reason": "active-only Dense QM context does not agree with padded Dense context",
            }
        padded = _measure_phase(
            padded_read, device, warmups=int(warmups), repetitions=int(repetitions)
        )
        active = _measure_phase(
            active_read, device, warmups=int(warmups), repetitions=int(repetitions)
        )
        return {
            "status": "ok",
            "scope": "inherited_dense_QM_reader_only",
            "active_module_count": int(active_index.numel()),
            "padded_module_width": int(encoded.module_centers.shape[1]),
            "M_active": int(active_index.numel()),
            "Mpack": int(encoded.module_centers.shape[1]),
            "agreement": agreement,
            "padded": padded,
            "active_only": active,
            "padding_speedup_ratio": (
                float(padded["median_ms"] / active["median_ms"])
                if float(active.get("median_ms", 0.0)) > 0.0
                else None
            ),
            "interpretation": "The active-only row removes padded module slots for the inherited dense fine reader; it does not measure routed end-to-end speed or learned physical sparsity.",
        }
    except DiagnosticUnavailable as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - diagnostics must preserve the failed phase
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        if prepared_output is not None:
            del prepared_output


def _receiver_chunk_contract(
    model: Any,
    *,
    requested_chunk_size: int,
    query_count: int,
    query_batch_size: int,
) -> dict[str, Any]:
    """Describe the chunking that the measured calls actually exercise.

    The channel wrapper accepts ``receiver_chunk_size`` for the matched
    interface-field families.  Its legacy ``decode_prepared`` branch does not
    forward that argument, so recording the requested value alone would
    falsely claim that Legacy was chunked.  Full-forward calls are also kept
    as the established all-query call; only prepared decoding uses the outer
    query batch loop.
    """

    architecture = str(model.config.core_honf.forward_architecture)
    core = getattr(model, "core", None)
    has_runtime_chunk = hasattr(core, "receiver_chunk_size")
    matched_interface = architecture != "legacy_honf" and bool(has_runtime_chunk)
    configured_chunk = getattr(core, "receiver_chunk_size", None) if matched_interface else None
    try:
        configured_chunk = None if configured_chunk is None else int(configured_chunk)
    except (TypeError, ValueError):
        configured_chunk = None
    query_count = max(0, int(query_count))
    query_batch_size = max(1, int(query_batch_size))
    return {
        "architecture": architecture,
        "requested_receiver_chunk_size": int(requested_chunk_size),
        "configured_receiver_chunk_size": (
            configured_chunk
        ),
        "effective_receiver_chunk_size": (
            int(requested_chunk_size) if matched_interface else None
        ),
        "receiver_chunk_override_applied": bool(matched_interface),
        "receiver_chunk_semantics": (
            "temporary core receiver_chunk_size override is used by matched "
            "interface-field reads"
            if matched_interface
            else (
                "legacy decode_prepared does not consume receiver_chunk_size; "
                "no effective receiver chunk is claimed"
                if architecture == "legacy_honf"
                else "model exposes no runtime receiver chunk; no effective receiver chunk is claimed"
            )
        ),
        "full_forward_outer_query_batch_size": int(query_count),
        "full_forward_outer_query_calls": 1,
        "full_forward_query_chunking": "single_all_query_call",
        "prepared_decode_outer_query_batch_size": min(query_count, query_batch_size),
        "prepared_decode_outer_query_calls": (
            math.ceil(query_count / query_batch_size) if query_count else 0
        ),
        "prepared_decode_query_chunking": "outer_query_batch_loop",
        "full_physical_outer_batched_preparation_query_count": 1 if query_count else 0,
        "full_physical_outer_batched_decode_query_batch_size": min(query_count, query_batch_size),
        "full_physical_outer_batched_decode_query_calls": (
            math.ceil(query_count / query_batch_size) if query_count else 0
        ),
        "full_physical_outer_batched_includes_preparation": True,
        "legacy_full_forward_all_query_warning": bool(architecture == "legacy_honf"),
    }


def _memory_phase_ledger(phases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Expose live/peak/incremental allocation fields for every phase."""

    fields = (
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "incremental_peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
    )
    return {
        str(name): {field: phase.get(field) for field in fields}
        for name, phase in phases.items()
        if isinstance(phase, Mapping)
    }


def _measure_profile_phase(
    function: Callable[[], Any],
    device: torch.device,
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    """Keep one failed profile scope inspectable and continue to later scopes."""

    try:
        return _measure_phase(
            function,
            device,
            warmups=int(warmups),
            repetitions=int(repetitions),
        )
    except RuntimeError as exc:
        out_of_memory = "out of memory" in str(exc).lower()
        if out_of_memory and device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "out_of_memory": bool(out_of_memory),
            "baseline_allocated_bytes": None,
            "baseline_reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "incremental_peak_allocated_bytes": None,
            "incremental_peak_reserved_bytes": None,
        }
    except Exception as exc:  # noqa: BLE001 - preserve an actual failed profile scope
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "out_of_memory": False,
            "baseline_allocated_bytes": None,
            "baseline_reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "incremental_peak_allocated_bytes": None,
            "incremental_peak_reserved_bytes": None,
        }


def _synchronized(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profiler_trace(
    function: Callable[[], Any],
    path: Path,
    label: str,
    device: torch.device,
) -> dict[str, Any]:
    """Capture one short trace, returning an explicit unavailable status."""

    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            with torch.profiler.record_function(label), torch.inference_mode():
                value = function()
            del value
        profiler.export_chrome_trace(str(path))
        key_averages = profiler.key_averages()
        def event_row(event: Any) -> dict[str, Any]:
            # ``device_time_total`` is the CUDA total when a CUDA activity is
            # present and is zero for the CPU-only trace.  Keep the accessor
            # names compatible with the PyTorch profiler versions used by the
            # existing workflows.
            return {
                "name": str(event.key),
                "self_cpu_time_total_us": float(event.self_cpu_time_total),
                "cpu_time_total_us": float(event.cpu_time_total),
                "self_device_time_total_us": float(getattr(event, "self_device_time_total", 0.0)),
                "device_time_total_us": float(getattr(event, "device_time_total", 0.0)),
                "count": int(event.count),
            }

        event_rows = [event_row(event) for event in key_averages]
        top_events = sorted(
            event_rows,
            key=lambda row: row["self_cpu_time_total_us"],
            reverse=True,
        )[:20]
        routing_events = [
            row
            for row in event_rows
            if "routing" in row["name"].lower() or _is_routing_key(row["name"])
        ]
        routing_events.sort(key=lambda row: row["self_cpu_time_total_us"], reverse=True)
        routing_totals = {
            "scope_count": len(routing_events),
            "event_count": int(sum(row["count"] for row in routing_events)),
            "self_cpu_time_total_us": float(sum(row["self_cpu_time_total_us"] for row in routing_events)),
            "cpu_time_total_us": float(sum(row["cpu_time_total_us"] for row in routing_events)),
            "self_device_time_total_us": float(sum(row["self_device_time_total_us"] for row in routing_events)),
            "device_time_total_us": float(sum(row["device_time_total_us"] for row in routing_events)),
        }
        return {
            "status": "ok",
            "path": str(path),
            "top_events": top_events,
            "routing_events": routing_events,
            "routing_totals": routing_totals,
        }
    except (ImportError, AttributeError) as exc:  # profiler availability is environment-dependent
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - diagnostics must preserve the failed trace
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def _prepared_environment_count(prepared_output: Any) -> int | None:
    """Read the realized environment width from either model wrapper.

    Matched interface fields expose ``prepared.encoded.env_coords``.  The
    historical Legacy wrapper intentionally has no encoded object, but its
    prepared organizer retains the adapter-built ``env_coords`` tensor.  The
    profile must report that observed width rather than echoing a requested
    synthetic shape.
    """

    candidates: list[Any] = [prepared_output]
    nested = getattr(prepared_output, "prepared", None)
    if nested is not None:
        candidates.append(nested)
    index = 0
    while index < len(candidates):
        candidate = candidates[index]
        index += 1
        encoded = getattr(candidate, "encoded", None)
        if encoded is not None:
            candidates.append(encoded)
        organizer = getattr(candidate, "organizer", None)
        if organizer is not None:
            candidates.append(organizer)
        if isinstance(candidate, Mapping):
            for key in ("prepared_state", "organizer", "encoded"):
                if key in candidate:
                    candidates.append(candidate[key])
    for candidate in candidates:
        env_coords = getattr(candidate, "env_coords", None)
        if env_coords is None and isinstance(candidate, Mapping):
            env_coords = candidate.get("env_coords")
        if torch.is_tensor(env_coords) and env_coords.ndim >= 2:
            return int(env_coords.shape[-2])
    return None


def _synthetic_environment_context(model: Any, env_count: int, lx: float, ly: float):
    """Choose an existing builder context while preserving actual E.

    The regional helper deliberately restricts its three prescribed even
    grids.  Legacy's bounded shape check also covers a tiny E=12 workload, so
    use the maintained stage-3 arbitrary-grid builder only for Legacy.  Both
    contexts restore the original builder on exit.
    """

    if str(model.config.core_honf.forward_architecture) == "legacy_honf":
        return stage3_synthetic_environment(model, env_count, lx, ly)
    return regular_synthetic_environment(model, env_count, lx, ly)


def _profile_real_workload(
    model: Any,
    dataset: Any,
    case_id: str,
    device: torch.device,
    args: argparse.Namespace,
    *,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    sample = __import__("channelthermal.evaluation.prepared", fromlist=["select_sample"]).select_sample(
        dataset, str(case_id), 0
    )
    query_np = _query_points(sample, int(args.query_count))
    batch = make_batch(dict(sample), query_np, device)
    query = batch["query_xy"]
    forward_kwargs = _profile_forward_kwargs(batch)
    architecture = str(model.config.core_honf.forward_architecture)
    receiver_chunk = int(args.receiver_chunk_size)
    query_batch = max(1, int(args.query_batch_size))

    def full_forward() -> Any:
        with _runtime_receiver_chunk_size(model, receiver_chunk):
            return model(batch["structure"], query, return_routing_maps=False, **forward_kwargs)

    def prepare_one() -> Any:
        with _runtime_receiver_chunk_size(model, receiver_chunk):
            return model(
                batch["structure"],
                query[:, :1],
                return_prepared_state=True,
                return_routing_maps=False,
                **forward_kwargs,
            )

    def prepared_decode(prepared_state: Any) -> Any:
        values = []
        with _runtime_receiver_chunk_size(model, receiver_chunk):
            for start in range(0, int(query.shape[1]), query_batch):
                values.append(
                    model.decode_prepared(
                        prepared_state,
                        query[:, start : start + query_batch],
                        return_routing_maps=False,
                        receiver_chunk_size=receiver_chunk,
                    )["pred_field"]
                )
        return torch.cat(values, dim=1)

    def full_physical_outer_batched() -> Any:
        """Time preparation plus the prescribed outer query-batch decode."""

        prepared_output: Mapping[str, Any] | None = None
        prepared_state: Any = None
        try:
            prepared_output = prepare_one()
            prepared_state = prepared_output["prepared_state"]
            return prepared_decode(prepared_state)
        finally:
            if prepared_state is not None:
                del prepared_state
            if prepared_output is not None:
                del prepared_output

    phases = {
        "physical_preparation_plus_one_query": _measure_profile_phase(
            prepare_one,
            device,
            warmups=int(args.warmups),
            repetitions=int(args.repetitions),
        ),
        "full_forward": _measure_profile_phase(
            full_forward,
            device,
            warmups=int(args.warmups),
            repetitions=int(args.repetitions),
        ),
    }
    # Prepare once outside the decode repetitions.  The state is private to
    # this phase and is released before the optional profiler pass.
    with torch.inference_mode():
        prepared_output = prepare_one()
    prepared_state = prepared_output["prepared_state"]
    realized_env_count = _prepared_environment_count(prepared_state)
    phases["prepared_decode"] = _measure_profile_phase(
        lambda prepared_state=prepared_state: prepared_decode(prepared_state),
        device,
        warmups=int(args.warmups),
        repetitions=int(args.repetitions),
    )
    # Do not let a retained prepared graph/state affect the independent
    # encoding/layout or full-forward baselines below.  The decode phase is
    # the only measurement that owns this prepared state.
    del prepared_state, prepared_output
    # Measure the combined outer-batched scope only after the independent
    # prepared-decode state has been released.  Its callback owns and releases
    # a fresh preparation, so live/peak memory belongs to this phase.
    phases["full_physical_outer_batched"] = _measure_profile_phase(
        full_physical_outer_batched,
        device,
        warmups=int(args.warmups),
        repetitions=int(args.repetitions),
    )
    # The preparation helper is kept separate from full physical preparation;
    # it records the adapter/layout phase using the actual case batch where the
    # backend can provide it.  The maintained helper is a matched-family
    # encoder; Legacy has no corresponding phase and is reported explicitly.
    if architecture == "legacy_honf":
        phases["encoding_plus_layout_construction"] = {
            "status": "not_applicable",
            "reason": "legacy_honf uses the legacy organizer wrapper; no matched-family adapter/layout phase is defined",
        }
    else:
        try:
            from profile_stage2_sparse_inference import _new_family_encoding_and_layout

            phases["encoding_plus_layout_construction"] = _measure_phase(
                lambda: _new_family_encoding_and_layout(model, batch),
                device,
                warmups=int(args.warmups),
                repetitions=int(args.repetitions),
            )
        except (ImportError, AttributeError) as exc:
            phases["encoding_plus_layout_construction"] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        except Exception as exc:  # noqa: BLE001 - retain an explicit failed adapter phase
            phases["encoding_plus_layout_construction"] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    try:
        with torch.inference_mode():
            observed = full_forward()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device.type == "cuda":
            torch.cuda.empty_cache()
        routing_observation = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        routing_scalars = {}
    except Exception as exc:  # noqa: BLE001 - preserve an actual observation failure
        routing_observation = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        routing_scalars = {}
    else:
        routing_observation = summarize_routing_output(observed)
        routing_scalars = _routing_scalar_values(observed)
        del observed
    # Retain only compact summaries before any independent reference timing.
    module_mask = batch["structure"]["module_present"] > 0.5
    active_module_count = int(module_mask.sum().detach().cpu())
    padded_module_width = int(module_mask.shape[-1])
    chunk_contract = _receiver_chunk_contract(
        model,
        requested_chunk_size=receiver_chunk,
        query_count=int(query.shape[1]),
        query_batch_size=query_batch,
    )
    dense_work = {
        "Q": int(query.shape[1]),
        "M_active": active_module_count,
        "Mpack": padded_module_width,
        "E_realized": realized_env_count,
        "dense_active_module_pairs": int(query.shape[1]) * active_module_count,
        "dense_padded_module_pairs": int(query.shape[1]) * padded_module_width,
        "dense_realized_environment_pairs": (
            None
            if realized_env_count is None
            else int(query.shape[1]) * realized_env_count
        ),
        "source_width_semantics": "M_active is physical present modules; Mpack is the encoded padded module width",
    }
    active_only_reference = None
    if bool(getattr(args, "active_only_reference", True)):
        active_only_reference = _active_only_dense_fine_reference(
            model,
            batch,
            prepare_one,
            device,
            warmups=int(args.reference_warmups),
            repetitions=int(args.reference_repetitions),
        )
    # The profiler performs its own forward after the active-only reference.
    trace = None
    if trace_path is not None:
        trace = _profiler_trace(full_forward, trace_path, f"dynamic_sparse_routing.real.{case_id}.full_forward", device)
    return {
        "name": f"real:{case_id}",
        "status": "ok",
        "case_id": str(case_id),
        "query_count": int(query.shape[1]),
        "realized_shape": {
            "M": active_module_count,
            "E": realized_env_count,
            "Q": int(query.shape[1]),
        },
        "receiver_chunk_size": receiver_chunk,
        "query_batch_size": query_batch,
        "receiver_chunk_contract": chunk_contract,
        "module_counts": {
            "active": active_module_count,
            "padded_width": padded_module_width,
            "M_active": active_module_count,
            "Mpack": padded_module_width,
        },
        "dense_work": dense_work,
        "phases": phases,
        "memory_ledger": _memory_phase_ledger(phases),
        "phase_state_lifetime": {
            "full_forward": "one direct all-query call; no prepared state retained",
            "prepared_decode": "one prepared state retained only for this phase and released before the independent outer-batched phase",
            "full_physical_outer_batched": "fresh preparation and outer query-batch decode owned by the measured callback and released before it returns",
            "profiler": "runs after all standalone prepared state is released",
        },
        "active_only_dense_fine_kernel_reference": active_only_reference,
        "routing_observation": routing_observation,
        "routing_scalars": routing_scalars,
        "trace": trace,
    }


def _profile_synthetic_workload(
    model: Any,
    shape: tuple[int, int, int],
    device: torch.device,
    args: argparse.Namespace,
    *,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    module_count, env_count, requested_query_count = (int(value) for value in shape)
    query_count = int(args.large_query_count or requested_query_count)
    # Keep the exact inherited execution geometry used by the maintained
    # stage-3/regional scaling workflow.
    lx, ly = _scaled_domain(module_count)
    structure = _synthetic_structure(model, module_count, lx, ly, device)
    query = _synthetic_query_tensor(query_count, lx, ly, device)
    receiver_chunk = int(args.receiver_chunk_size)
    query_batch = max(1, int(args.query_batch_size))

    def full_forward() -> Any:
        with _runtime_receiver_chunk_size(model, receiver_chunk):
            return model(
                structure,
                query,
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.0,
                return_routing_maps=False,
            )

    def prepare_one() -> Any:
        with _runtime_receiver_chunk_size(model, receiver_chunk):
            return model(
                structure,
                query[:, :1],
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.0,
                return_prepared_state=True,
                return_routing_maps=False,
            )

    realized_env_count: int | None = None
    try:
        # Unlike the real case adapter, this synthetic workload has no
        # serialized environment builder.  Override its domain and grid only
        # for this execution, restoring both through the maintained contexts.
        with temporary_domain(model, lx, ly), _synthetic_environment_context(
            model, env_count, lx, ly
        ):
            # Keep the independent preparation/full baselines free from any
            # retained prepared state.  A private state is created only for
            # the prepared-decode phase and released immediately afterwards.
            phases = {
                "physical_preparation_plus_one_query": _measure_profile_phase(
                    prepare_one,
                    device,
                    warmups=int(args.large_warmups),
                    repetitions=int(args.large_repetitions),
                ),
                "full_forward": _measure_profile_phase(
                    full_forward,
                    device,
                    warmups=int(args.large_warmups),
                    repetitions=int(args.large_repetitions),
                ),
            }
            with torch.inference_mode():
                prepared_output = prepare_one()
            prepared_state = prepared_output["prepared_state"]

            def prepared_decode(prepared_value: Any) -> Any:
                values = []
                with _runtime_receiver_chunk_size(model, receiver_chunk):
                    for start in range(0, query_count, query_batch):
                        values.append(
                            model.decode_prepared(
                                prepared_value,
                                query[:, start : start + query_batch],
                                return_routing_maps=False,
                                receiver_chunk_size=receiver_chunk,
                            )["pred_field"]
                        )
                return torch.cat(values, dim=1)

            def full_physical_outer_batched() -> Any:
                """Time preparation plus the prescribed outer query-batch decode."""

                prepared_output_value: Mapping[str, Any] | None = None
                prepared_state_value: Any = None
                try:
                    prepared_output_value = prepare_one()
                    prepared_state_value = prepared_output_value["prepared_state"]
                    return prepared_decode(prepared_state_value)
                finally:
                    if prepared_state_value is not None:
                        del prepared_state_value
                    if prepared_output_value is not None:
                        del prepared_output_value

            phases["prepared_decode"] = _measure_profile_phase(
                lambda prepared_value=prepared_state: prepared_decode(prepared_value),
                device,
                warmups=int(args.large_warmups),
                repetitions=int(args.large_repetitions),
            )
            prepared_inner = getattr(prepared_state, "prepared", prepared_state)
            encoded = getattr(prepared_inner, "encoded", None)
            realized_env_count = _prepared_environment_count(prepared_state)
            module_mask = structure["module_present"] > 0.5
            active_module_count = int(module_mask.sum().detach().cpu())
            padded_module_width = int(module_mask.shape[-1])
            chunk_contract = _receiver_chunk_contract(
                model,
                requested_chunk_size=receiver_chunk,
                query_count=query_count,
                query_batch_size=query_batch,
            )
            dense_work = {
                "Q": query_count,
                "M_active": active_module_count,
                "Mpack": padded_module_width,
                "dense_active_module_pairs": query_count * active_module_count,
                "dense_padded_module_pairs": query_count * padded_module_width,
                "E_realized": realized_env_count,
                "dense_realized_environment_pairs": (
                    None
                    if realized_env_count is None
                    else query_count * realized_env_count
                ),
                "source_width_semantics": "M_active is physical present modules; Mpack is the encoded padded module width",
            }
            del prepared_state, prepared_output, prepared_inner, encoded
            # As in the real path, measure this combined scope only after all
            # state retained by the standalone prepared-decode phase has been
            # released.  The callback owns its own preparation lifetime.
            phases["full_physical_outer_batched"] = _measure_profile_phase(
                full_physical_outer_batched,
                device,
                warmups=int(args.large_warmups),
                repetitions=int(args.large_repetitions),
            )
            try:
                with torch.inference_mode():
                    observed = full_forward()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device.type == "cuda":
                    torch.cuda.empty_cache()
                routing_observation = {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                routing_scalars = {}
            except Exception as exc:  # noqa: BLE001 - preserve an actual observation failure
                routing_observation = {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                routing_scalars = {}
            else:
                routing_observation = summarize_routing_output(observed)
                routing_scalars = _routing_scalar_values(observed)
                del observed
            # The profiler performs its own forward; retain only compact
            # summaries from this observation.
            trace = None
            if trace_path is not None:
                trace = _profiler_trace(
                    full_forward,
                    trace_path,
                    "dynamic_sparse_routing.synthetic.large.full_forward",
                    device,
                )
            result = {
                "name": f"synthetic:{module_count},{env_count},{requested_query_count}",
                "status": "ok",
                "requested_shape": {"M": module_count, "E": env_count, "Q": requested_query_count},
                "realized_shape": {"M": module_count, "E": realized_env_count, "Q": query_count},
                "receiver_chunk_size": receiver_chunk,
                "query_batch_size": query_batch,
                "receiver_chunk_contract": chunk_contract,
                "dense_work": dense_work,
                "phases": phases,
                "memory_ledger": _memory_phase_ledger(phases),
                "phase_state_lifetime": {
                    "full_forward": "one direct all-query call; no prepared state retained",
                    "prepared_decode": "one prepared state retained only for this phase and released before the independent outer-batched phase",
                    "full_physical_outer_batched": "fresh preparation and outer query-batch decode owned by the measured callback and released before it returns",
                    "profiler": "runs after all standalone prepared state is released",
                },
                "routing_observation": routing_observation,
                "routing_scalars": routing_scalars,
                "trace": trace,
            }
            return result
    except Exception as exc:  # noqa: BLE001 - preserve per-model profile failure
        return {
            "name": f"synthetic:{module_count},{env_count},{requested_query_count}",
            "status": "failed",
            "requested_shape": {"M": module_count, "E": env_count, "Q": requested_query_count},
            "realized_shape": {"M": module_count, "E": realized_env_count, "Q": query_count},
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run the two existing disposable real physical AdamW checks."""

    from run_regional_response_study import run_smoke as regional_run_smoke

    regional_args = SimpleNamespace(
        output=Path(args.output),
        device=str(args.device),
        dataset=getattr(args, "dataset", None),
        split=str(args.split),
        query_count=8192,
        profile=str(args.profile),
        points_per_case=int(args.points_per_case),
        batch_size=int(args.batch_size),
        case_count=int(args.case_count),
        batch_kind=str(args.batch_kind),
        small_case_id=args.small_case_id,
        large_case_id=args.large_case_id,
    )
    payload = regional_run_smoke(regional_args)
    steps = payload.get("steps", {})
    return {
        "schema_version": 1,
        "task": "dynamic_sparse_routing_smoke",
        "profile": str(args.profile),
        "device": str(args.device),
        "checks": {
            "small_M": steps.get("small_module_batch"),
            "largest_ordinary_M": steps.get("large_module_batch"),
        },
        "requested": {
            "points_per_case": int(args.points_per_case),
            "batch_size": int(args.batch_size),
            "small_case_count": int(args.case_count),
            "large_case_count": int(args.case_count),
            "predicted_port_loop": True,
            "frozen_stage_a": True,
            "adamw": True,
            "fp64_gradient_reductions": True,
        },
        "source": "run_regional_response_study.run_smoke",
        "managed_run_reserved": False,
        "checkpoint_saved": False,
        "steps": steps,
        "limitations": payload.get("limitations", []),
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Measure synchronized phases and optionally capture real/large traces."""

    device = select_device(args.device)
    records = _load_profile_models(args, device)
    trace_dir = Path(args.trace_dir).expanduser().resolve() if args.trace_dir else Path(args.output).expanduser().resolve().parent / "traces"
    model_rows: list[dict[str, Any]] = []
    try:
        for label, model, dataset, checkpoint, dataset_path in records:
            real_rows: list[dict[str, Any]] = []
            case_ids = [str(value) for value in (args.case_id or ANCHOR_CASE_IDS[:2])]
            for case_id in case_ids:
                trace_path = trace_dir / f"{label}_real_{case_id}.json" if args.trace else None
                try:
                    real_rows.append(
                        _profile_real_workload(
                            model,
                            dataset,
                            case_id,
                            device,
                            args,
                            trace_path=trace_path,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - preserve per-anchor profile failure
                    real_rows.append(
                        {
                            "name": f"real:{case_id}",
                            "status": "failed",
                            "case_id": case_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            synthetic_rows: list[dict[str, Any]] = []
            for shape_values in (args.large_shape or []):
                shape = tuple(int(value) for value in shape_values)
                trace_path = (
                    trace_dir / f"{label}_synthetic_{shape[0]}_{shape[1]}_{shape[2]}.json"
                    if args.trace
                    else None
                )
                synthetic_rows.append(
                    _profile_synthetic_workload(
                        model,
                        shape,
                        device,
                        args,
                        trace_path=trace_path,
                    )
                )
            model_rows.append(
                {
                    "label": label,
                    "checkpoint": {
                        "path": str(checkpoint.get("_path", "")),
                        "epoch": checkpoint.get("epoch", "fresh"),
                    },
                    "dataset": str(dataset_path),
                    "architecture": str(model.config.core_honf.forward_architecture),
                    "real_anchors": real_rows,
                    "large_shapes": synthetic_rows,
                }
            )
    finally:
        for _label, model, dataset, _checkpoint, _dataset_path in records:
            del model
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
    return {
        "schema_version": 1,
        "task": "dynamic_sparse_routing_profile",
        "device": str(device),
        "protocol": {
            "real_warmups": int(args.warmups),
            "real_repetitions": int(args.repetitions),
            "large_warmups": int(args.large_warmups),
            "large_repetitions": int(args.large_repetitions),
            "receiver_chunk_size": int(args.receiver_chunk_size),
            "query_batch_size": int(args.query_batch_size),
            "full_forward_query_call_contract": "one outer call containing all Q query points",
            "prepared_decode_query_call_contract": "outer query batches use query_batch_size; receiver chunk semantics are reported per model row",
            "full_physical_outer_batched_contract": "separate preparation plus outer query-batch decode scope; direct all-Q full_forward is retained as its own measured scope",
            "legacy_chunking_contract": "legacy_honf decode_prepared ignores receiver_chunk_size; profile rows report effective_receiver_chunk_size=null",
            "dense_work_contract": "profile rows expose M_active, Mpack, Q, and active/padded pair references",
            "memory_contract": "each phase exposes baseline, peak, and incremental allocated/reserved byte fields",
            "detailed_routing_maps": False,
        },
        "models": model_rows,
        "interpretation": {
            "phase_medians": "Unprofiled synchronized phase medians; separately measured phases overlap and need not sum.",
            "large_shapes": "Actual generated layouts and model forwards when requested; execution evidence only until a reference field exists.",
            "trace": "Profiler traces attribute route discovery, joins, deduplication, fine readers, and reductions; latency conclusions use unprofiled rows.",
        },
    }


def _checkpoint_bindings(args: argparse.Namespace) -> list[tuple[str, str, Path, int]]:
    """Resolve endpoint labels and validate exact/best checkpoint policy."""

    from honf_runtime.compat import load_trusted_checkpoint

    required_normalization_keys = {
        "field_mean_by_channel",
        "field_std_by_channel",
        "heat_power_mean",
        "heat_power_std",
        "interface_condition_mean",
        "interface_condition_std",
        "interface_target_mean",
        "interface_target_std",
        "internal_temperature_mean",
        "internal_temperature_std",
        "sampled_point_mean_by_channel",
        "sampled_point_std_by_channel",
    }
    values: list[tuple[str, str]] = []
    values.extend(("exact500", str(value)) for value in (args.exact_checkpoint or []))
    values.extend(("saved_best", str(value)) for value in (args.best_checkpoint or []))
    values.extend(("unspecified", str(value)) for value in (args.checkpoint or []))
    if not values:
        raise ValueError("endpoint requires --checkpoint, --exact-checkpoint, or --best-checkpoint.")
    bindings: list[tuple[str, str, Path, int]] = []
    labels: set[str] = set()
    for default_policy, raw in values:
        if "=" not in raw:
            raise ValueError(f"Endpoint checkpoint must use LABEL=PATH syntax, got {raw!r}.")
        label, path_text = raw.split("=", 1)
        label = label.strip()
        lowered = label.lower()
        path = Path(path_text.strip()).expanduser().resolve()
        if not label or not path.is_file():
            raise FileNotFoundError(f"Endpoint checkpoint {label!r} is missing: {path}")
        policy = default_policy
        if policy == "unspecified":
            policy = "saved_best" if "best" in lowered else "exact500"
        checkpoint = load_trusted_checkpoint(path, map_location="cpu")
        raw_epoch = checkpoint.get("epoch", checkpoint.get("current_epoch"))
        try:
            actual_epoch = int(raw_epoch)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Endpoint checkpoint has no integer epoch: {path}") from exc
        if not 1 <= actual_epoch <= 500:
            raise ValueError(
                f"Endpoint checkpoint epoch {actual_epoch} is outside the authorized 1..500 range: {path}"
            )
        if policy == "exact500" and actual_epoch != 500:
            raise ValueError(
                f"Exact endpoint checkpoint must contain epoch 500, got epoch {actual_epoch}: {path}"
            )
        if policy == "saved_best":
            # The ordinary filename is authoritative.  A caller may use a
            # different existing filename only with an explicit label stating
            # that it is the validation-field selection, preserving the
            # exact-versus-best distinction in the endpoint artifact.
            normalized_label = lowered.replace("_", "").replace("-", "")
            if path.name != "best_by_field_mse_model.pt" and "actualfieldbest" not in normalized_label:
                raise ValueError(
                    "Saved-best endpoint must use best_by_field_mse_model.pt "
                    "or an explicit actualfieldbest label"
                )
        model_config = checkpoint.get("model_config", {})
        core_config = model_config.get("core_honf", {}) if isinstance(model_config, Mapping) else {}
        interface_config = core_config.get("interface_model", {}) if isinstance(core_config, Mapping) else {}
        receiver_chunk = interface_config.get("receiver_chunk_size") if isinstance(interface_config, Mapping) else None
        if receiver_chunk is None or int(receiver_chunk) != 128:
            raise ValueError(
                f"Endpoint checkpoint must retain configured receiver_chunk_size=128, got {receiver_chunk!r}: {path}"
            )
        normalization = checkpoint.get("global_normalization_stats", {})
        if not isinstance(normalization, Mapping) or not required_normalization_keys.issubset(normalization):
            missing = sorted(required_normalization_keys - set(normalization) if isinstance(normalization, Mapping) else required_normalization_keys)
            raise ValueError(f"Endpoint checkpoint normalization stats missing {missing}: {path}")
        output_label = f"{policy}:{label}"
        if output_label in labels:
            raise ValueError(f"Duplicate endpoint checkpoint label: {output_label}")
        labels.add(output_label)
        bindings.append((output_label, policy, path, actual_epoch))
    return bindings


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def run_endpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Call the existing complete-grid evaluator for exact and saved-best rows."""

    from channelthermal.workflows import compare_models

    bindings = _checkpoint_bindings(args)
    evaluation_dir = (
        Path(args.evaluation_dir).expanduser().resolve()
        if args.evaluation_dir
        else Path(args.output).expanduser().resolve().parent / "comparison"
    )
    argv: list[str] = []
    for label, _policy, path, _epoch in bindings:
        argv.extend(("--checkpoint-path", str(path), "--label", label))
    argv.extend(
        (
            "--saved-root",
            str(evaluation_dir),
            "--split",
            str(args.split),
            "--case-ratio",
            "1.0",
            "--seed",
            "0",
            "--query-batch-size",
            str(args.query_batch_size),
            "--local-port-condition-mode",
            "predicted",
            "--output-dir",
            str(evaluation_dir),
        )
    )
    if args.device:
        argv.extend(("--device", str(args.device)))
    if args.dataset:
        argv.extend(("--dataset", str(args.dataset)))
    if not args.skip_figures:
        pass
    else:
        argv.append("--skip-figures")
    # Only the five established anchors receive debug arrays.  The evaluator
    # still computes the complete 90-case grid for every policy row.
    requested_anchors = tuple(args.anchor_case_id or ANCHOR_CASE_IDS)
    if len(requested_anchors) != len(ANCHOR_CASE_IDS) or set(requested_anchors) != set(ANCHOR_CASE_IDS):
        raise ValueError(
            "endpoint debug anchors are fixed to 0273, 0653, 0283, 0298, and 0302"
        )
    argv.append("--save-debug-npz")
    for case_id in ANCHOR_CASE_IDS:
        argv.extend(("--debug-case-id", str(case_id), "--anchor-case-id", str(case_id)))
    compare_models.main(argv)

    table_path = evaluation_dir / "tables" / "per_case_metrics.csv"
    rows = _read_csv(table_path)
    counts: dict[str, int] = {}
    case_sets: dict[str, set[str]] = {}
    for row in rows:
        label = str(row.get("model_label", ""))
        counts[label] = counts.get(label, 0) + 1
        case_sets.setdefault(label, set()).add(str(row.get("case_id", "")))
    policies = [
        {
            "label": label,
            "policy": policy,
            "checkpoint": str(path),
            "actual_epoch": int(epoch),
            "observed_case_count": int(counts.get(label, 0)),
            "observed_unique_case_count": len(case_sets.get(label, set())),
            "expected_case_count": 90,
            "complete": bool(
                counts.get(label, 0) == 90
                and len(case_sets.get(label, set())) == 90
            ),
        }
        for label, policy, path, epoch in bindings
    ]
    observed_sets = [case_sets.get(row["label"], set()) for row in policies]
    case_sets_match = bool(observed_sets) and all(case_set == observed_sets[0] for case_set in observed_sets[1:])
    complete = bool(all(bool(row["complete"]) for row in policies) and case_sets_match)
    run_config_path = evaluation_dir / "logs" / "run_config.json"
    if run_config_path.is_file():
        # compare_models accepts explicit paths but its historical metadata
        # field defaults to the generic selector name.  Make the endpoint
        # artifact state the actual explicit-path policy used here.
        try:
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            run_config["checkpoint_selector"] = "explicit_path"
            run_config["checkpoint_selection_policy"] = "exact_and_saved_best_separate"
            run_config["explicit_checkpoint_policies"] = {
                label: {"policy": policy, "actual_epoch": int(epoch), "checkpoint": str(path)}
                for label, policy, path, epoch in bindings
            }
            run_config_path.write_text(
                json.dumps(run_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not update explicit endpoint metadata: {run_config_path}") from exc
    return {
        "schema_version": 1,
        "task": "dynamic_sparse_routing_endpoint",
        "status": "ok" if complete else "failed",
        "failure_count": 0 if complete else sum(not bool(row["complete"]) for row in policies),
        "split": str(args.split),
        "device": str(args.device),
        "policies": policies,
        "evaluation": {
            "workflow": "channelthermal.workflows.compare_models",
            "output_dir": str(evaluation_dir),
            "per_case_metrics": str(table_path),
            "predicted_ports": True,
            "receiver_chunk_size": 128,
            "query_batch_size": int(args.query_batch_size),
            "anchors": [str(value) for value in ANCHOR_CASE_IDS],
            "fixed_anchor_case_ids": list(ANCHOR_CASE_IDS),
            "case_sets_match": case_sets_match,
            "exact_and_saved_best_separate": True,
        },
        "limitations": [
            "The current test split is the established development holdout; it is not an untouched CFD test set.",
            "A saved-best policy is reported at its checkpoint epoch and is never substituted for exact epoch 500.",
        ],
    }


def _zero_context_result(value: Any) -> Any:
    """Zero the context payload while preserving auxiliary shape metadata."""

    if torch.is_tensor(value):
        return torch.zeros_like(value)
    if isinstance(value, tuple):
        if not value:
            return value
        return (_zero_context_result(value[0]), *value[1:])
    if isinstance(value, list):
        return [_zero_context_result(item) for item in value]
    if isinstance(value, Mapping):
        result = dict(value)
        for key in ("context", "value", "response", "module_context", "environment_context", "qm_context", "qe_context"):
            if key in result and torch.is_tensor(result[key]):
                result[key] = torch.zeros_like(result[key])
                break
        return result
    return value


def _routed_backend_zero(
    model: Any,
    *,
    phase: str,
    component: str,
) -> contextlib.AbstractContextManager[Any]:
    """Patch an explicit routed QM/QE backend hook when one exists.

    A generic combined ``read`` hook is intentionally not treated as a QM or
    QE hook: doing so would report the wrong intervention.  This is why an
    unsupported backend returns a visible ``unavailable`` row.
    """

    backend = getattr(model.core, "backend", None)
    if backend is None:
        raise DiagnosticUnavailable("model has no interface backend")
    aliases = {
        "qm": (
            "read_qm",
            "read_module_pairs",
            "read_module_messages",
            "read_fine_module",
            "read_module_context",
        ),
        "qe": (
            "read_qe",
            "read_environment_pairs",
            "read_environment_messages",
            "read_fine_environment",
            "read_environment_context",
        ),
    }
    names = aliases.get(str(component))
    if names is None:
        # Uniform routing is a backend-owned intervention because it must
        # replace both typed priors without changing source states.
        for name in ("routing_intervention", "routing_override", "intervention"):
            method = getattr(backend, name, None)
            if not callable(method):
                continue
            try:
                candidate = method(mode=str(component), phase=str(phase))
            except TypeError:
                candidate = method(str(component), str(phase))
            if hasattr(candidate, "__enter__") and hasattr(candidate, "__exit__"):
                return candidate
        raise DiagnosticUnavailable(
            f"backend exposes no explicit uniform-routing intervention for {component!r}"
        )
    method_name = next((name for name in names if callable(getattr(backend, name, None))), None)
    if method_name is None:
        raise DiagnosticUnavailable(
            f"backend exposes no explicit {component.upper()} fine-read hook; combined read was not patched"
        )

    @contextlib.contextmanager
    def manager() -> Iterator[dict[str, Any]]:
        original = getattr(backend, method_name)
        core = model.core
        previous_role = getattr(core, "_interface_read_role", None)

        def zero(*call_args: Any, **call_kwargs: Any) -> Any:
            # Physical coupling tags each backend read.  Restrict the patch
            # to the requested phase so P0/P1/P2 are recomputed independently.
            if getattr(core, "_interface_read_role", None) != str(phase):
                return original(*call_args, **call_kwargs)
            result = original(*call_args, **call_kwargs)
            return _zero_context_result(result)

        setattr(backend, method_name, zero)
        try:
            yield {
                "status": "ok",
                "phase": str(phase),
                "component": str(component),
                "implementation": method_name,
                "role_scope": str(phase),
            }
        finally:
            setattr(backend, method_name, original)
            if previous_role is None:
                try:
                    delattr(core, "_interface_read_role")
                except AttributeError:
                    pass
            else:
                core._interface_read_role = previous_role

    return manager()


@contextlib.contextmanager
def _routed_uniform(model: Any, *, phase: str) -> Iterator[dict[str, Any]]:
    """Replace both query densities by source-measure support at one phase.

    Setting ``d[q,k]=1`` on every valid hub makes the compiled prior reduce to
    ``Pi[q,i]=omega[i]`` because each ordinary source membership row sums to
    one.  The source states, fine readers, and physical loop remain intact.
    """

    from honf_forward_core.interface_fields.routing_index.types import MeasureQueryProjection

    backend = getattr(model.core, "backend", None)
    original = getattr(backend, "_query_density", None)
    if not callable(original):
        raise DiagnosticUnavailable("backend has no _query_density hook for uniform typed-prior intervention")
    core = model.core

    def uniform_query_density(*call_args: Any, **call_kwargs: Any) -> Any:
        result = original(*call_args, **call_kwargs)
        if getattr(core, "_interface_read_role", None) != str(phase):
            return result
        if not isinstance(result, tuple) or len(result) != 2:
            raise DiagnosticUnavailable("backend _query_density returned an unsupported value")
        logits, projection = result
        if not hasattr(projection, "support"):
            raise DiagnosticUnavailable("backend _query_density returned no typed projection")
        incidence = call_kwargs.get("incidence")
        if incidence is None and len(call_args) >= 5:
            incidence = call_args[4]
        candidates = call_kwargs.get("candidates")
        if candidates is None and len(call_args) >= 4:
            candidates = call_args[3]
        hub_measure = getattr(incidence, "hub_measure", None)
        candidate_valid = getattr(candidates, "valid", None)
        if not torch.is_tensor(hub_measure) or not torch.is_tensor(candidate_valid):
            raise DiagnosticUnavailable("uniform intervention could not recover typed source measures")
        valid = candidate_valid[:, None, :].expand_as(logits) & (hub_measure[:, None, :] > 0.0)
        density = torch.where(valid, torch.ones_like(logits), torch.zeros_like(logits))
        measure = hub_measure[:, None, :].expand_as(logits)
        probability = torch.where(valid, measure, torch.zeros_like(measure))
        threshold = logits.new_zeros(logits.shape[:-1])
        return logits.new_zeros(logits.shape), MeasureQueryProjection(
            density=density,
            probability=probability,
            threshold=threshold,
            support=valid,
        )

    backend._query_density = uniform_query_density
    try:
        yield {
            "status": "ok",
            "phase": str(phase),
            "implementation": "_query_density: density=1 on every valid hub",
            "effective_prior": "source_measure",
        }
    finally:
        backend._query_density = original


def _routed_intervention_context(model: Any, mode: str) -> contextlib.AbstractContextManager[Any]:
    mode = str(mode)
    if mode == "uniform_p2":
        return _routed_uniform(model, phase="p2_field")
    if mode == "uniform_p0":
        return _routed_uniform(model, phase="p0_port")
    if mode == "uniform_p1":
        return _routed_uniform(model, phase="p1_refinement")
    if mode == "qm_zero_p2":
        return _routed_backend_zero(model, phase="p2_field", component="qm")
    if mode == "qe_zero_p2":
        return _routed_backend_zero(model, phase="p2_field", component="qe")
    if mode == "coarse_zero_p2":
        return common_read_zero(model, role="p2_field", component="coarse")
    raise ValueError(f"unknown routed intervention mode={mode!r}")


def run_interventions(args: argparse.Namespace) -> dict[str, Any]:
    """Run explicit P0/P1/P2 routing interventions on fixed anchors."""

    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 1:
        raise ValueError("interventions requires one explicitly labelled checkpoint")
    device = select_device(args.device)
    spec = specs[0]
    model, checkpoint = _load_model_spec(spec, device)
    architecture = str(model.config.core_honf.forward_architecture)
    if architecture != "routed_pairwise_honf":
        raise ValueError(f"interventions is for routed_pairwise_honf, got {architecture!r}")
    dataset, dataset_path = _load_dataset(checkpoint, _profile_args(args))
    case_ids = [str(value) for value in (args.case_id or ANCHOR_CASE_IDS)]
    raw_samples = {case_id: _load_raw_sample(dataset_path, args.split, case_id) for case_id in case_ids}
    modes = tuple(args.mode or ("uniform_p2", "qm_zero_p2", "qe_zero_p2", "uniform_p0", "uniform_p1", "coarse_zero_p2"))
    rows: list[dict[str, Any]] = []
    state_before = _state_keys(model)
    try:
        for case_id in case_ids:
            sample = __import__("channelthermal.evaluation.prepared", fromlist=["select_sample"]).select_sample(dataset, case_id, 0)
            query_np = _query_points(sample, int(args.query_count))
            base = _run_variant(model, sample, query_np, device)
            base_errors = _canonical_ground_truth_errors(base, sample, raw_samples[case_id], dataset, model, checkpoint)
            row: dict[str, Any] = {
                "case_id": case_id,
                "query_count": len(query_np),
                "ground_truth_errors": {"normal": base_errors},
                "interventions": {},
            }
            for mode in modes:
                try:
                    context = _routed_intervention_context(model, mode)
                    variant = _run_variant(model, sample, query_np, device, context)
                    errors = _canonical_ground_truth_errors(variant, sample, raw_samples[case_id], dataset, model, checkpoint)
                    row["ground_truth_errors"][mode] = errors
                    row["interventions"][mode] = {
                        "status": "ok",
                        "prediction_difference": _relative_difference(variant.get("pred_field"), base.get("pred_field")),
                        "prediction_differences": {
                            key: _relative_difference(variant.get(key), base.get(key))
                            for key in ("pred_field", "pred_interface", "pred_internal_temperature", "pred_port_condition")
                        },
                        "error_deltas": _error_deltas(base_errors, errors),
                    }
                except DiagnosticUnavailable as exc:
                    row["interventions"][mode] = {
                        "status": "unavailable",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                except Exception as exc:  # noqa: BLE001 - preserve per-anchor intervention failure
                    row["interventions"][mode] = {
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
            rows.append(row)
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    return {
        "schema_version": 1,
        "task": "dynamic_sparse_routing_interventions",
        "stage": f"endpoint{int(checkpoint.get('epoch', -1))}",
        "checkpoint": {
            "label": spec.label,
            "path": str(spec.path),
            "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        },
        "architecture": architecture,
        "dataset": str(dataset_path),
        "case_ids": case_ids,
        "results": rows,
        "state_dict_structure_unchanged": state_before == _state_keys(model),
        "interpretation": {
            "uniform": "source-measure intervention at the selected phase, if the backend exposes an explicit typed-prior override",
            "qm_qe": "zero only the named fine branch; a combined read hook is never treated as a branch hook",
            "error_delta": "intervened minus normal canonical error; positive means worse",
        },
    }


def _find_named_value(root: Any, names: Sequence[str], *, depth: int = 0) -> Any:
    if root is None or depth > 4:
        return None
    wanted = {str(name).lower() for name in names}
    if isinstance(root, Mapping):
        for key, value in root.items():
            if str(key).lower() in wanted:
                return value
        for value in root.values():
            found = _find_named_value(value, names, depth=depth + 1)
            if found is not None:
                return found
        return None
    if dataclasses.is_dataclass(root):
        return _find_named_value(vars(root), names, depth=depth + 1)
    for name in names:
        if hasattr(root, name):
            return getattr(root, name)
    if hasattr(root, "__dict__"):
        return _find_named_value(vars(root), names, depth=depth + 1)
    return None


def _integer_count(value: Any) -> int | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return int(value.detach().cpu())
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _route_value(outputs: Mapping[str, Any], key: str) -> Any:
    """Read one exact final or phase-prefixed route value from model output."""

    # The ThermalChannel wrapper keeps the final read and the P0/P1 read
    # summaries in interaction_aux.  A small direct-output fallback makes the
    # helper usable with a bare InterfaceFieldCore result in focused tests.
    for container_name in ("interaction_aux", "routing_aux", "provisional_read_aux"):
        container = outputs.get(container_name)
        if isinstance(container, Mapping) and key in container:
            return container[key]
    return outputs.get(key)


def _query_receiver_count(value: torch.Tensor) -> int | None:
    """Count receiver axes in a route map, including reshaped P0 ports."""

    if not torch.is_tensor(value) or value.ndim < 3:
        return None
    count = 1
    for width in value.shape[1:-1]:
        count *= int(width)
    return int(count)


def _scalar_number(value: Any) -> float | None:
    """Convert a scalar tensor/number to a finite float when available."""

    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _distribution_summary(value: torch.Tensor | np.ndarray | Sequence[float]) -> dict[str, Any]:
    """Summarize an occupancy/support distribution without retaining its map."""

    if torch.is_tensor(value):
        array = value.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    else:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": int(array.size), "finite_count": 0}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "zero_fraction": float(np.mean(finite == 0.0)),
        "singleton_fraction": float(np.mean(finite == 1.0)),
    }


def _route_pair_mask(
    outputs: Mapping[str, Any],
    prefix: str,
    *,
    batch_count: int,
    receiver_count: int,
    source_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Materialize only a bounded selected-pair mask for the omitted-source audit."""

    aux = outputs.get("interaction_aux")
    if not isinstance(aux, Mapping):
        raise DiagnosticUnavailable("routed output has no interaction_aux pair maps")
    batch = aux.get(f"routing_{prefix}_pair_batch")
    receiver = aux.get(f"routing_{prefix}_pair_receiver")
    source = aux.get(f"routing_{prefix}_pair_source")
    if not all(torch.is_tensor(value) for value in (batch, receiver, source)):
        raise DiagnosticUnavailable(f"routed output has no final {prefix.upper()} pair maps")
    batch = batch.to(device=device, dtype=torch.long).reshape(-1)
    receiver = receiver.to(device=device, dtype=torch.long).reshape(-1)
    source = source.to(device=device, dtype=torch.long).reshape(-1)
    if not (batch.numel() == receiver.numel() == source.numel()):
        raise RuntimeError(f"final {prefix.upper()} pair maps have mismatched lengths")
    if batch.numel() and (
        int(batch.min()) < 0
        or int(batch.max()) >= int(batch_count)
        or int(receiver.min()) < 0
        or int(receiver.max()) >= int(receiver_count)
        or int(source.min()) < 0
        or int(source.max()) >= int(source_count)
    ):
        raise RuntimeError(f"final {prefix.upper()} pair map contains an out-of-range index")
    selected = torch.zeros(
        (int(batch_count), int(receiver_count), int(source_count)),
        device=device,
        dtype=torch.bool,
    )
    if batch.numel():
        selected[batch, receiver, source] = True
    return selected


def _missed_source_audit(
    model: Any,
    outputs: Mapping[str, Any],
    sample: Mapping[str, Any],
    raw_sample: Mapping[str, Any],
    dataset: Any,
    query_np: np.ndarray,
    device: torch.device,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare normal sparse reads with P2 source-measure/full-support reads.

    The dense attention and omitted module norms are evaluated from the same
    prepared source states as the normal routed output.  This keeps the audit
    focused on omitted source work and avoids treating a descriptor as a
    field-value substitute.
    """

    prepared_wrapper = outputs.get("prepared_state")
    prepared = getattr(prepared_wrapper, "prepared", prepared_wrapper)
    if prepared is None:
        raise DiagnosticUnavailable("prepared state is required for omitted-source audit")
    encoded = getattr(prepared, "encoded", None)
    state = getattr(prepared, "backend_state", None)
    backend = getattr(getattr(model, "core", None), "backend", None)
    if encoded is None or not isinstance(state, Mapping) or backend is None:
        raise DiagnosticUnavailable("prepared routed backend state is unavailable")
    batch = make_batch(dict(sample), query_np, device)
    receivers = batch["query_xy"].float()
    batch_count, receiver_count = int(receivers.shape[0]), int(receivers.shape[1])
    module_count = int(encoded.module_centers.shape[1])
    environment_count = int(encoded.env_coords.shape[1])
    module_selected = _route_pair_mask(
        outputs,
        "module",
        batch_count=batch_count,
        receiver_count=receiver_count,
        source_count=module_count,
        device=device,
    )
    environment_selected = _route_pair_mask(
        outputs,
        "environment",
        batch_count=batch_count,
        receiver_count=receiver_count,
        source_count=environment_count,
        device=device,
    )
    receiver_features = model.core._receiver_features(prepared, receivers)
    with torch.no_grad():
        # DensePairwiseField's inherited reader is used only as an evaluation
        # oracle.  It reads the routed prepared environmental values with the
        # original geometry and quadrature-weighted attention.
        _dense_environment_context, dense_attention = backend.read_environment(
            state,
            encoded,
            receivers,
            receiver_features,
            return_routing_maps=True,
        )
        if dense_attention is None:
            raise DiagnosticUnavailable("Dense environmental attention map is unavailable")
        dense_probability = dense_attention.mean(dim=1)
        environment_omitted = ~environment_selected
        environment_omitted_mass = dense_probability.masked_fill(~environment_omitted, 0.0).sum(dim=-1)

        module_tokens = state.get("module_tokens")
        if not torch.is_tensor(module_tokens):
            raise DiagnosticUnavailable("prepared module source states are unavailable")
        relative = (
            receivers[:, :, None, :] - encoded.module_centers[:, None, :, :]
        ) / encoded.coordinate_scale
        source = module_tokens[:, None, :, :].expand(-1, receiver_count, -1, -1)
        relative_features = backend.relative_fourier(relative)
        global_features = encoded.global_token[:, None, None, :].expand(
            -1, receiver_count, module_count, -1
        )
        module_messages = backend._mlp(
            backend.query_module_message,
            torch.cat([source, relative_features, global_features], dim=-1),
        )
        module_norm = torch.linalg.vector_norm(module_messages, dim=-1)
        active_modules = encoded.module_present[:, None, :] > 0.5
        module_omitted = active_modules & ~module_selected
        module_omitted_norm = module_norm.masked_select(module_omitted)

        uniform_output = _run_variant(
            model,
            sample,
            query_np,
            device,
            _routed_intervention_context(model, "uniform_p2"),
        )
        normal_errors = _canonical_ground_truth_errors(
            outputs, sample, raw_sample, dataset, model, checkpoint
        )
        uniform_errors = _canonical_ground_truth_errors(
            uniform_output, sample, raw_sample, dataset, model, checkpoint
        )

    def _stats(value: torch.Tensor) -> dict[str, Any]:
        return _distribution_summary(value)

    return {
        "status": "ok",
        "query_count": receiver_count,
        "normal_vs_uniform_p2": {
            "prediction_difference": _relative_difference(
                uniform_output.get("pred_field"), outputs.get("pred_field")
            ),
            "error_delta": _error_deltas(normal_errors, uniform_errors),
            "uniform_effective_prior": "source_measure on every valid hub; all valid fine sources are dispatched",
        },
        "module": {
            "active_source_count": int(active_modules.sum().cpu()),
            "selected_source_count": int(module_selected.sum().cpu()),
            "omitted_pair_count": int(module_omitted.sum().cpu()),
            "omitted_message_norm": _stats(module_omitted_norm),
            "message_norm_definition": "raw inherited QM message norm before source reduction and output projection; vector cancellation and nonlinear heads prevent physical-importance interpretation",
        },
        "environment": {
            "source_count": environment_count,
            "selected_source_count": int(environment_selected.sum().cpu()),
            "omitted_pair_count": int(environment_omitted.sum().cpu()),
            "dense_attention_omitted_probability_mass": _stats(environment_omitted_mass),
            "dense_attention_probability_definition": "mean over inherited Dense attention heads, using the same prepared environmental states, original geometry bias, and quadrature log weights",
        },
    }


def _routing_index_summary(
    state: Mapping[str, Any],
    encoded: Any,
    *,
    phase: str,
    retain_support: bool = False,
    retain_maps: bool = False,
    max_map_values: int = 2_000_000,
) -> dict[str, Any]:
    """Summarize one prepared routing index, including true ``D_k``.

    ``membership`` is shaped ``[B,N,K]``.  Summing over ``K`` describes the
    number of hubs incident to each source (source fanout); the prescribed
    occupancy is the opposite reduction over ``N``:
    ``D_k = sum_i 1[A[i,k] > 0]``.  Keep both names explicit so a source
    fanout cannot be mistaken for hub occupancy.  Only compact summaries are
    retained unless a bounded turnover measurement explicitly requests the
    boolean support masks.
    """

    routing_index = state.get("routing_index") if isinstance(state, Mapping) else None
    if routing_index is None:
        raise DiagnosticUnavailable("prepared state has no routing_index")
    candidates = getattr(routing_index, "candidates", None)
    candidate_valid = getattr(candidates, "valid", None)
    candidate_mask = (
        candidate_valid.to(dtype=torch.bool)
        if torch.is_tensor(candidate_valid)
        else None
    )
    candidate_summary: dict[str, Any] = {}
    if candidates is not None:
        coords = getattr(candidates, "coords", None)
        valid = getattr(candidates, "valid", None)
        if torch.is_tensor(coords):
            candidate_summary["padded_width"] = int(coords.shape[1])
        if torch.is_tensor(valid):
            active_count = valid.to(dtype=torch.int64).sum(dim=-1)
            candidate_summary["active_count"] = _distribution_summary(active_count)
            # Keep the per-batch denominator available to phase query-union
            # fractions.  It is at most the small evaluation batch dimension,
            # while the full incidence/query maps remain discarded.
            candidate_summary["active_count_per_batch"] = active_count.detach().cpu().tolist()

    encoded_module_present = getattr(encoded, "module_present", None)
    encoded_env_weights = getattr(encoded, "env_weights", None)
    source_validity = {
        "module": (
            encoded_module_present.to(dtype=torch.bool)
            if torch.is_tensor(encoded_module_present)
            else None
        ),
        "environment": (
            encoded_env_weights > 0.0
            if torch.is_tensor(encoded_env_weights)
            else None
        ),
    }
    incidence_by_type = {
        "module": getattr(routing_index, "module_incidence", None),
        "environment": getattr(routing_index, "environment_incidence", None),
    }
    incidence_rows: dict[str, Any] = {}
    support_masks: dict[str, torch.Tensor] = {}
    map_arrays: dict[str, torch.Tensor] = {}
    map_omissions: list[dict[str, Any]] = []

    def retain_map(name: str, value: Any) -> None:
        """Retain one bounded CPU map for a selected-anchor NPZ export."""

        if not retain_maps or not torch.is_tensor(value):
            return
        if int(value.numel()) > int(max_map_values):
            map_omissions.append(
                {
                    "name": str(name),
                    "shape": list(value.shape),
                    "count": int(value.numel()),
                    "max_map_values": int(max_map_values),
                }
            )
            return
        map_arrays[str(name)] = value.detach().cpu()

    if candidates is not None:
        for candidate_name in (
            "coords",
            "descriptors",
            "propensity",
            "valid",
            "candidate_origin",
        ):
            candidate_value = getattr(candidates, candidate_name, None)
            if torch.is_tensor(candidate_value):
                retain_map(f"hub_{candidate_name}", candidate_value)
    for source_type, incidence in incidence_by_type.items():
        if incidence is None:
            continue
        membership = getattr(incidence, "membership", None)
        hub_measure = getattr(incidence, "hub_measure", None)
        if not torch.is_tensor(membership) or membership.ndim != 3:
            continue
        positive = membership > 0.0
        if candidate_mask is not None and tuple(candidate_mask.shape) == tuple(membership.shape[:1] + membership.shape[2:]):
            positive = positive & candidate_mask[:, None, :]
            hub_valid = candidate_mask
        else:
            hub_valid = torch.ones(
                membership.shape[:1] + membership.shape[2:],
                device=membership.device,
                dtype=torch.bool,
            )
        source_valid = source_validity.get(source_type)
        if source_valid is None or tuple(source_valid.shape) != tuple(membership.shape[:2]):
            source_valid = torch.ones(
                membership.shape[:2], device=membership.device, dtype=torch.bool
            )
        valid_pairs = source_valid.unsqueeze(-1) & hub_valid.unsqueeze(1)
        positive = positive & valid_pairs
        # A hub with zero induced source measure is not an occupied routing
        # hub, even if padded arithmetic left a positive membership entry.
        if torch.is_tensor(hub_measure) and tuple(hub_measure.shape) == tuple(hub_valid.shape):
            occupied_hub = hub_valid & (hub_measure > 0.0) & positive.any(dim=1)
        else:
            occupied_hub = hub_valid & positive.any(dim=1)
        source_fanout = positive.sum(dim=-1).to(dtype=torch.float32)
        d_k = positive.sum(dim=1).to(dtype=torch.float32)
        fanout_values = source_fanout.masked_select(source_valid)
        d_k_values = d_k.masked_select(hub_valid)
        occupied_d_k_values = d_k.masked_select(occupied_hub)
        fanout_summary = _distribution_summary(fanout_values)
        valid_pair_values = valid_pairs.reshape(-1)
        valid_positive_values = positive.reshape(-1)
        membership_zero_fraction = (
            float((~valid_positive_values[valid_pair_values]).to(dtype=torch.float32).mean().detach().cpu())
            if bool(valid_pair_values.any())
            else None
        )
        incidence_rows[source_type] = {
            "source_count": int(membership.shape[1]),
            "hub_count": int(membership.shape[2]),
            "incidence_nnz": int(positive.sum().detach().cpu()),
            "source_fanout_occupancy": fanout_summary,
            "source_support_occupancy": fanout_summary,
            "hub_source_occupancy_Dk": _distribution_summary(d_k_values),
            "Dk_occupancy": _distribution_summary(d_k_values),
            "occupied_hub_source_occupancy_Dk": _distribution_summary(occupied_d_k_values),
            "occupied_hub_count": _distribution_summary(occupied_hub.sum(dim=-1)),
            "active_source_count": _distribution_summary(source_valid.sum(dim=-1)),
            "source_active_hub_count": _distribution_summary(occupied_hub.sum(dim=-1)),
            "source_zero_fraction": fanout_summary.get("zero_fraction"),
            "source_singleton_fraction": fanout_summary.get("singleton_fraction"),
            "membership_zero_fraction": membership_zero_fraction,
            "valid_hub_count": _distribution_summary(hub_valid.sum(dim=-1)),
        }
        retain_map(f"{source_type}_source_A", membership)
        retain_map(f"{source_type}_source_weights", getattr(incidence, "source_weights", None))
        retain_map(f"{source_type}_hub_measure", hub_measure)
        if retain_support:
            # The turnover CLI uses only small fixed-query endpoint forwards;
            # move the bounded masks to CPU so its model can be released.
            support_masks[source_type] = positive.detach().cpu()
    result: dict[str, Any] = {
        "phase": str(phase),
        "candidate": candidate_summary,
        "incidence": incidence_rows,
    }
    if retain_support:
        result["_support_masks"] = support_masks
    if retain_maps:
        result["_routing_maps"] = map_arrays
        if map_omissions:
            result["routing_map_omissions"] = map_omissions
    return result


@contextlib.contextmanager
def _capture_routing_preparations(
    model: Any,
    *,
    retain_support: bool = False,
    retain_maps: bool = False,
    max_map_values: int = 2_000_000,
) -> Iterator[dict[str, list[dict[str, Any]]]]:
    """Capture compact P0/P1/P2 preparation summaries through the backend hook."""

    backend = getattr(getattr(model, "core", None), "backend", None)
    original = getattr(backend, "prepare", None)
    if not callable(original):
        raise DiagnosticUnavailable("routed backend has no prepare hook")
    records: dict[str, list[dict[str, Any]]] = {}
    core = model.core

    def wrapped(*call_args: Any, **call_kwargs: Any) -> Any:
        state = original(*call_args, **call_kwargs)
        encoded = call_args[0] if call_args else call_kwargs.get("encoded")
        role = str(getattr(core, "_interface_read_role", "unscoped"))
        if encoded is not None and isinstance(state, Mapping):
            summary = _routing_index_summary(
                state,
                encoded,
                phase=role,
                retain_support=retain_support,
                retain_maps=retain_maps,
                max_map_values=max_map_values,
            )
            records.setdefault(role, []).append(summary)
        return state

    backend.prepare = wrapped
    try:
        yield records
    finally:
        backend.prepare = original


def _ledger_row(
    outputs: Mapping[str, Any],
    sample: Mapping[str, Any],
    case_id: str,
    query_count: int,
) -> dict[str, Any]:
    prepared = outputs.get("prepared_state")
    inner = getattr(prepared, "prepared", prepared)
    encoded = getattr(inner, "encoded", None)
    backend_state = getattr(inner, "backend_state", None)
    module_present = np.asarray(sample["structure"]["module_present"], dtype=float).reshape(-1)
    active_modules = int((module_present > 0.5).sum())
    padded_modules = int(module_present.size)
    env_count = None
    if encoded is not None and torch.is_tensor(getattr(encoded, "env_coords", None)):
        env_count = int(encoded.env_coords.shape[1])
    routing_index = _find_named_value(backend_state, ("routing_index",))
    candidate = getattr(routing_index, "candidates", None)
    if candidate is None:
        candidate = _find_named_value(backend_state, ("candidates", "routing_candidates"))
    candidate_count = None
    candidate_padded_width = None
    candidate_active = None
    if candidate is not None:
        coords = getattr(candidate, "coords", None)
        valid = getattr(candidate, "valid", None)
        if torch.is_tensor(coords):
            candidate_padded_width = int(coords.shape[1])
        if torch.is_tensor(valid):
            active_values = valid.to(dtype=torch.float32).sum(dim=-1)
            candidate_active = _distribution_summary(active_values)
            candidate_count = (
                int(active_values.reshape(-1)[0].detach().cpu())
                if active_values.numel() == 1
                else candidate_active
            )

    try:
        final_index_summary = _routing_index_summary(
            backend_state,
            encoded,
            phase="p2_field",
        )
    except DiagnosticUnavailable:
        final_index_summary = {"phase": "p2_field", "candidate": {}, "incidence": {}}
    source_rows = final_index_summary.get("incidence", {})
    candidate_summary = final_index_summary.get("candidate", {})

    def phase_metrics(phase: str, prefix: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for metric in (
            "raw_path_count",
            "unique_pair_count",
            "duplicate_expansion",
            "fine_pair_count",
            "query_hub_support",
        ):
            value = _route_value(outputs, f"{prefix}_{metric}")
            scalar = _scalar_number(value)
            if scalar is not None:
                metrics[metric] = scalar
            elif torch.is_tensor(value):
                metrics[metric] = _distribution_summary(value)
        density = _route_value(outputs, f"{prefix}_query_density")
        if torch.is_tensor(density) and density.ndim >= 3:
            support = density > 0.0
            # The union is reduced over receiver rows; it is intentionally
            # separate from per-receiver fanout and captures whether the
            # phase ever used a hub anywhere in its actual receiver set.
            receiver_count = _query_receiver_count(density)
            support_flat = support.reshape(int(support.shape[0]), -1, int(support.shape[-1]))
            union_count = support_flat.any(dim=1).sum(dim=-1)
            metrics["executed_receiver_count"] = receiver_count
            metrics["query_hub_union"] = _distribution_summary(union_count)
            padded_fraction = _distribution_summary(
                union_count.to(dtype=torch.float32) / max(int(density.shape[-1]), 1)
            )
            metrics["query_hub_union_fraction"] = padded_fraction
            metrics["query_hub_union_fraction_padded"] = padded_fraction
            metrics["query_hub_union_fraction_denominator"] = "padded candidate width K_padded"
            active_counts = candidate_summary.get("active_count_per_batch")
            if isinstance(active_counts, Sequence) and not isinstance(active_counts, (str, bytes)):
                active_denominator = torch.as_tensor(
                    active_counts, device=union_count.device, dtype=torch.float32
                ).reshape(-1)
                if int(active_denominator.numel()) == int(union_count.numel()):
                    metrics["query_hub_union_fraction_activeK"] = _distribution_summary(
                        union_count.to(dtype=torch.float32)
                        / active_denominator.clamp_min(1.0)
                    )
                    metrics["query_hub_union_activeK_denominator"] = _distribution_summary(
                        active_denominator
                    )
                    metrics["query_hub_union_fraction_activeK_denominator"] = (
                        "valid candidate hubs K_active per batch"
                    )
        else:
            metrics["executed_receiver_count"] = None
        return metrics

    final_module = phase_metrics("p2", "routing_module")
    final_environment = phase_metrics("p2", "routing_environment")
    raw_path = _scalar_number(_route_value(outputs, "routing_module_raw_path_count"))
    unique_pair = _scalar_number(_route_value(outputs, "routing_module_unique_pair_count"))
    duplicate_expansion = _scalar_number(_route_value(outputs, "routing_module_duplicate_expansion"))
    module_source_row = source_rows.get("module", {})
    environment_source_row = source_rows.get("environment", {})

    def summary_mean(value: Any) -> float | None:
        return float(value["mean"]) if isinstance(value, Mapping) and value.get("mean") is not None else None

    dense_module_pairs = int(query_count) * active_modules
    dense_environment_pairs = None if env_count is None else int(query_count) * env_count
    ntheta = None
    predicted_ports = outputs.get("pred_port_condition")
    if torch.is_tensor(predicted_ports) and predicted_ports.ndim >= 3:
        ntheta = int(predicted_ports.shape[2])
    p0_receiver_count = phase_metrics("p0", "initial_port_routing_module").get(
        "executed_receiver_count"
    )
    p0_active_receiver_count = None if ntheta is None else int(active_modules * ntheta)
    row: dict[str, Any] = {
        "case_id": str(case_id),
        "query_count": int(query_count),
        "active_module_source_count": active_modules,
        "padded_module_source_count": padded_modules,
        "M_active": active_modules,
        "Mpack": padded_modules,
        "environment_source_count": env_count,
        "candidate_hub_count": candidate_count,
        "candidate_padded_width": candidate_padded_width,
        "candidate_active_hub_count": candidate_active,
        "source_incidence_nnz": module_source_row.get("incidence_nnz"),
        "environment_incidence_nnz": environment_source_row.get("incidence_nnz"),
        "module_source_active_hub_count": module_source_row.get("source_active_hub_count"),
        "environment_source_active_hub_count": environment_source_row.get("source_active_hub_count"),
        "module_source_fanout_occupancy": module_source_row.get("source_fanout_occupancy"),
        "environment_source_fanout_occupancy": environment_source_row.get("source_fanout_occupancy"),
        "module_source_occupancy": module_source_row.get("source_fanout_occupancy"),
        "environment_source_occupancy": environment_source_row.get("source_fanout_occupancy"),
        "module_source_Dk_occupancy": module_source_row.get("hub_source_occupancy_Dk"),
        "environment_source_Dk_occupancy": environment_source_row.get("hub_source_occupancy_Dk"),
        "module_source_hub_occupancy_Dk": module_source_row.get("hub_source_occupancy_Dk"),
        "environment_source_hub_occupancy_Dk": environment_source_row.get("hub_source_occupancy_Dk"),
        "candidate_active_hub_count_mean": summary_mean(candidate_active),
        "module_source_active_hub_count_mean": summary_mean(
            module_source_row.get("source_active_hub_count")
        ),
        "environment_source_active_hub_count_mean": summary_mean(
            environment_source_row.get("source_active_hub_count")
        ),
        "module_source_occupancy_mean": summary_mean(
            module_source_row.get("source_fanout_occupancy")
        ),
        "environment_source_occupancy_mean": summary_mean(
            environment_source_row.get("source_fanout_occupancy")
        ),
        "module_source_fanout_mean": summary_mean(
            module_source_row.get("source_fanout_occupancy")
        ),
        "environment_source_fanout_mean": summary_mean(
            environment_source_row.get("source_fanout_occupancy")
        ),
        "module_source_Dk_mean": summary_mean(
            module_source_row.get("hub_source_occupancy_Dk")
        ),
        "environment_source_Dk_mean": summary_mean(
            environment_source_row.get("hub_source_occupancy_Dk")
        ),
        "p0_executed_receiver_count": p0_receiver_count,
        "p0_active_physical_port_receiver_count": p0_active_receiver_count,
        "raw_two_hop_path_count": raw_path,
        "unique_receiver_source_pair_count": unique_pair,
        "duplicate_expansion": duplicate_expansion,
        "dense_active_module_pair_reference": dense_module_pairs,
        "dense_environment_pair_reference": dense_environment_pairs,
        "routing_aux_scalars": _routing_scalar_values(outputs),
        "phase_metrics": {
            "p0_port_module": phase_metrics("p0", "initial_port_routing_module"),
            "p0_port_environment": phase_metrics("p0", "initial_port_routing_environment"),
            "p1_refinement_module": phase_metrics("p1", "provisional_routing_module"),
            "p1_refinement_environment": phase_metrics("p1", "provisional_routing_environment"),
            "p2_field_module": final_module,
            "p2_field_environment": final_environment,
        },
    }
    for phase_name, metrics in row["phase_metrics"].items():
        if not isinstance(metrics, Mapping):
            continue
        for metric_name in ("raw_path_count", "unique_pair_count", "duplicate_expansion"):
            metric_value = metrics.get(metric_name)
            if isinstance(metric_value, (int, float, np.integer, np.floating)):
                row[f"{phase_name}_{metric_name}"] = float(metric_value)
        support = metrics.get("query_hub_support")
        if isinstance(support, Mapping):
            row[f"{phase_name}_query_hub_support_mean"] = summary_mean(support)
    row["p2_module_query_hub_support_mean"] = summary_mean(final_module.get("query_hub_support"))
    row["p2_environment_query_hub_support_mean"] = summary_mean(
        final_environment.get("query_hub_support")
    )
    row["p2_module_query_active_hub_count"] = final_module.get("query_hub_union")
    row["p2_environment_query_active_hub_count"] = final_environment.get("query_hub_union")
    row["module_source_active_hub_count"] = module_source_row.get("occupied_hub_count")
    row["environment_source_active_hub_count"] = environment_source_row.get("occupied_hub_count")
    row["final_routing_index"] = final_index_summary
    # These pair references use the receiver count actually executed by each
    # phase.  The final field count is kept separately for the usual Q=32
    # ledger row, while P0 reports padded physical-port work explicitly.
    for phase_name, metrics in row["phase_metrics"].items():
        if not isinstance(metrics, Mapping):
            continue
        receiver_count = metrics.get("executed_receiver_count")
        if isinstance(receiver_count, (int, float, np.integer, np.floating)):
            row[f"{phase_name}_dense_active_module_pair_reference"] = int(receiver_count) * active_modules
            if env_count is not None:
                row[f"{phase_name}_dense_environment_pair_reference"] = int(receiver_count) * env_count
    if candidate_summary:
        row["candidate_summary"] = candidate_summary
    return row


def _selected_routing_map_arrays(
    model: Any,
    outputs: Mapping[str, Any],
    sample: Mapping[str, Any],
    query_np: np.ndarray,
    preparation_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_map_values: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect the bounded geometry/route slices used by the five-anchor figures.

    The normal ledger is intentionally scalar and compact.  ``--save-maps``
    adds only source geometry/incidence maps for the captured P0/P1/P2
    preparations and one selected port plus one far P2 query.  In particular,
    it does not serialize a query-by-source map for every receiver.
    """

    maps: dict[str, np.ndarray] = {}
    omissions: list[dict[str, Any]] = []

    def add(name: str, value: Any) -> None:
        if value is None:
            return
        if torch.is_tensor(value):
            if int(value.numel()) > int(max_map_values):
                omissions.append(
                    {
                        "name": str(name),
                        "shape": list(value.shape),
                        "count": int(value.numel()),
                        "max_map_values": int(max_map_values),
                    }
                )
                return
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
            if int(array.size) > int(max_map_values):
                omissions.append(
                    {
                        "name": str(name),
                        "shape": list(array.shape),
                        "count": int(array.size),
                        "max_map_values": int(max_map_values),
                    }
                )
                return
        maps[str(name)] = np.asarray(array)

    def without_single_batch(value: Any) -> Any:
        value = _map_value_to_cpu_array(value)
        array = np.asarray(value) if isinstance(value, np.ndarray) else None
        if array is not None and array.ndim > 0 and int(array.shape[0]) == 1:
            return array[0]
        return value

    prepared_wrapper = outputs.get("prepared_state")
    prepared = getattr(prepared_wrapper, "prepared", prepared_wrapper)
    encoded = getattr(prepared, "encoded", None)
    geometry_arrays: dict[str, Any] = {}
    if encoded is not None:
        for name, attribute in (
            ("module_centers", "module_centers"),
            ("module_present", "module_present"),
            ("env_coords", "env_coords"),
            ("env_weights", "env_weights"),
        ):
            value = without_single_batch(getattr(encoded, attribute, None))
            geometry_arrays[name] = value
            add(name, value)

    # Store renderer metadata from the loaded routed model and the actual
    # arrays above.  In particular, do not infer a future strategy name from
    # this diagnostic exporter: the strategy is read from the model config.
    interface_config = getattr(getattr(model.config, "core_honf", None), "interface_model", None)
    routing_config = getattr(interface_config, "routing", None)
    routing_strategy = getattr(routing_config, "strategy", None)
    source_kinds = []
    for source_name, geometry_name in (("module", "module_centers"), ("environment", "env_coords")):
        geometry = geometry_arrays.get(geometry_name)
        if geometry is not None and np.asarray(geometry).size:
            source_kinds.append(source_name)
    coordinate_dims = []
    for geometry_name in ("module_centers", "env_coords"):
        geometry = geometry_arrays.get(geometry_name)
        if geometry is not None:
            array = np.asarray(geometry)
            if array.ndim >= 2:
                coordinate_dims.append(int(array.shape[-1]))
    if routing_strategy is not None:
        add("routing_strategy", str(routing_strategy))
    if source_kinds:
        add("source_kind", "+".join(source_kinds))
    if coordinate_dims and len(set(coordinate_dims)) == 1:
        add("coordinate_dim", coordinate_dims[0])
    add("route_weight_semantics", ROUTING_WEIGHT_SEMANTICS)

    # Each captured preparation includes the same candidate metadata and the
    # source-to-hub A/mu for its own phase.  Prefixing by phase keeps P0/P1/P2
    # refreshes inspectable without claiming that a final index describes all
    # physical reads.
    for phase, summaries in preparation_records.items():
        if not summaries:
            continue
        summary = summaries[-1]
        routing_maps = summary.get("_routing_maps")
        if not isinstance(routing_maps, Mapping):
            summary_omissions = summary.get("routing_map_omissions")
            if isinstance(summary_omissions, Sequence) and not isinstance(summary_omissions, (str, bytes)):
                omissions.extend(dict(item) for item in summary_omissions if isinstance(item, Mapping))
            continue
        for name, value in routing_maps.items():
            add(f"{phase}__{name}", without_single_batch(value))
        summary_omissions = summary.get("routing_map_omissions")
        if isinstance(summary_omissions, Sequence) and not isinstance(summary_omissions, (str, bytes)):
            omissions.extend(dict(item) for item in summary_omissions if isinstance(item, Mapping))

    interaction = outputs.get("interaction_aux", {})
    if not isinstance(interaction, Mapping):
        interaction = {}

    def tensor_value(key: str) -> torch.Tensor | None:
        value = interaction.get(key)
        return value if torch.is_tensor(value) else None

    module_present = np.asarray(
        sample.get("structure", {}).get("module_present", []), dtype=np.float64
    ).reshape(-1)
    active_module_indices = np.flatnonzero(module_present > 0.5)
    module_index = int(active_module_indices[0]) if active_module_indices.size else 0
    pred_ports = outputs.get("pred_port_condition")
    ntheta = (
        int(pred_ports.shape[2])
        if torch.is_tensor(pred_ports) and pred_ports.ndim >= 3
        else None
    )
    if ntheta is None:
        probe = tensor_value("initial_port_routing_module_query_probability")
        if probe is not None and probe.ndim >= 4:
            ntheta = int(probe.shape[2])
    ntheta = max(int(ntheta or 1), 1)
    theta_index = 0
    port_receiver_index = module_index * ntheta + theta_index

    query_array = np.asarray(query_np, dtype=np.float64).reshape(-1, 2)
    centers = np.asarray(
        sample.get("structure", {}).get("module_centers", []), dtype=np.float64
    ).reshape(-1, 2)
    if active_module_indices.size:
        active_centers = centers[active_module_indices]
        distances = np.linalg.norm(query_array[:, None, :] - active_centers[None, :, :], axis=-1)
        far_query_index = int(np.argmax(np.min(distances, axis=1)))
    else:
        far_query_index = max(int(query_array.shape[0]) - 1, 0)

    selection: dict[str, Any] = {
        "port_module_index": module_index,
        "port_theta_index": theta_index,
        "port_flat_receiver_index": port_receiver_index,
        "far_query_index": far_query_index,
        "query_count": int(query_array.shape[0]),
        "map_scope": "selected P0 port and far P2 query; source A/geometry for P0/P1/P2",
    }
    add("selected_port_indices", np.asarray([module_index, theta_index, port_receiver_index], dtype=np.int64))
    if query_array.size:
        add("selected_far_query_xy", query_array[far_query_index])
    if centers.size and module_index < centers.shape[0]:
        radius = float(getattr(getattr(model.config, "core_honf", None), "module_radius", 0.0))
        try:
            parameter = next(model.parameters())
            fixed = model.local_coupling.port_head.fixed_theta_tokens(
                ntheta, device=parameter.device, dtype=parameter.dtype
            )
            normal = fixed[theta_index, 1:3].detach().cpu().numpy().astype(np.float64)
            add("selected_port_xy", centers[module_index] + radius * normal)
        except (AttributeError, StopIteration, TypeError, RuntimeError):
            # The route map remains useful without an adapter-specific port
            # coordinate helper; the selected module/theta indices are saved.
            pass

    def selected_query_value(
        prefix: str,
        *,
        module: int,
        theta: int,
        flat_receiver: int,
        query_index: int,
    ) -> torch.Tensor | None:
        value = tensor_value(prefix)
        if value is None:
            return None
        if value.ndim >= 4:
            return value[0, module, theta]
        if value.ndim == 3:
            return value[0, flat_receiver if prefix.startswith("initial_port_") else query_index]
        if value.ndim == 2:
            return value[flat_receiver if prefix.startswith("initial_port_") else query_index]
        return None

    def selected_pairs(
        source_type: str,
        pair_prefix: str,
        *,
        receiver_index: int,
        label: str,
        source_count: int | None,
    ) -> None:
        batch = tensor_value(f"{pair_prefix}pair_batch")
        receiver = tensor_value(f"{pair_prefix}pair_receiver")
        source = tensor_value(f"{pair_prefix}pair_source")
        prior = tensor_value(f"{pair_prefix}pair_prior")
        if not all(torch.is_tensor(value) for value in (batch, receiver, source, prior)):
            return
        batch_np = batch.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
        receiver_np = receiver.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
        source_np = source.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
        prior_np = prior.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
        if not (batch_np.size == receiver_np.size == source_np.size == prior_np.size):
            return
        selected = (batch_np == 0) & (receiver_np == int(receiver_index))
        pair_ids = np.stack((batch_np[selected], receiver_np[selected], source_np[selected]), axis=1)
        pair_prior = prior_np[selected]
        add(f"{label}_{source_type}_pair_ids", pair_ids)
        add(f"{label}_{source_type}_pair_Pi", pair_prior)
        count = int(source_count or (int(source_np.max()) + 1 if source_np.size else 0))
        effective = np.zeros((max(count, 0),), dtype=np.float64)
        valid = (source_np[selected] >= 0) & (source_np[selected] < effective.size)
        if np.any(valid):
            np.add.at(effective, source_np[selected][valid], pair_prior[valid])
        add(f"{label}_{source_type}_effective_Pi", effective)

    for source_type in ("module", "environment"):
        for phase_label, prefix, receiver_index, query_index in (
            ("p0_port", "initial_port_routing_", port_receiver_index, 0),
            ("p2_far", "routing_", far_query_index, far_query_index),
        ):
            query_prefix = f"{prefix}{source_type}_"
            alpha = selected_query_value(
                f"{query_prefix}query_probability",
                module=module_index,
                theta=theta_index,
                flat_receiver=port_receiver_index,
                query_index=query_index,
            )
            density = selected_query_value(
                f"{query_prefix}query_density",
                module=module_index,
                theta=theta_index,
                flat_receiver=port_receiver_index,
                query_index=query_index,
            )
            if alpha is not None:
                add(f"{phase_label}_{source_type}_query_hub_alpha", alpha)
                add(
                    f"{phase_label}_{source_type}_query_hub_support",
                    (alpha > 0.0).to(dtype=torch.bool),
                )
            if density is not None:
                add(f"{phase_label}_{source_type}_query_hub_density", density)
            source_count = None
            if encoded is not None:
                source_value = getattr(
                    encoded,
                    "module_centers" if source_type == "module" else "env_coords",
                    None,
                )
                if torch.is_tensor(source_value):
                    source_count = int(source_value.shape[1])
            selected_pairs(
                source_type,
                query_prefix,
                receiver_index=receiver_index,
                label=phase_label,
                source_count=source_count,
            )

    if omissions:
        selection["omitted_maps"] = omissions
    return maps, selection


def run_ledger(args: argparse.Namespace) -> dict[str, Any]:
    """Collect source/occupancy/path counts and bounded omitted-source metadata."""

    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 1:
        raise ValueError("ledger requires one explicitly labelled checkpoint")
    device = select_device(args.device)
    spec = specs[0]
    model, checkpoint = _load_model_spec(spec, device)
    if str(model.config.core_honf.forward_architecture) != "routed_pairwise_honf":
        raise ValueError("ledger requires routed_pairwise_honf")
    dataset, dataset_path = _load_dataset(checkpoint, _profile_args(args))
    case_ids = [str(value) for value in (args.case_id or ANCHOR_CASE_IDS)]
    rows: list[dict[str, Any]] = []
    map_paths: list[str] = []
    maps_dir = Path(args.maps_dir).expanduser().resolve() if args.maps_dir else Path(args.output).expanduser().resolve().parent / "routing_maps"
    try:
        for case_id in case_ids:
            sample = __import__("channelthermal.evaluation.prepared", fromlist=["select_sample"]).select_sample(dataset, case_id, 0)
            query_np = _query_points(sample, int(args.query_count))
            retain_selected_maps = bool(args.save_maps and case_id in ANCHOR_CASE_IDS)
            with _capture_routing_preparations(
                model,
                retain_maps=retain_selected_maps,
                max_map_values=int(args.max_map_values),
            ) as preparation_records:
                outputs = _forward_batch(
                    model,
                    sample,
                    query_np,
                    device,
                    return_prepared_state=True,
                    return_routing_maps=True,
                    return_organizer_passes=True,
                )
            row = _ledger_row(outputs, sample, case_id, len(query_np))
            row["routing_observation"] = summarize_routing_output(outputs)
            # Private map tensors are exported to the selected-anchor NPZ and
            # omitted from the JSON summary so the ledger stays compact.
            phase_preparation_summary: dict[str, list[dict[str, Any]]] = {}
            for phase, summaries in preparation_records.items():
                phase_preparation_summary[phase] = []
                for summary in summaries:
                    compact = dict(summary)
                    compact.pop("_routing_maps", None)
                    phase_preparation_summary[phase].append(compact)
            row["phase_preparations"] = phase_preparation_summary
            if case_id in ANCHOR_CASE_IDS[:2] and bool(getattr(args, "missed_source_audit", True)):
                try:
                    raw_sample = _load_raw_sample(dataset_path, args.split, case_id)
                    row["missed_source_audit"] = _missed_source_audit(
                        model,
                        outputs,
                        sample,
                        raw_sample,
                        dataset,
                        query_np,
                        device,
                        checkpoint,
                    )
                except DiagnosticUnavailable as exc:
                    row["missed_source_audit"] = {
                        "status": "unavailable",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                except Exception as exc:  # noqa: BLE001 - preserve per-anchor ledger failure
                    row["missed_source_audit"] = {
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
            elif case_id in ANCHOR_CASE_IDS[:2]:
                row["missed_source_audit"] = {
                    "status": "not_requested",
                    "reason": "omitted-source audit disabled by --no-missed-source-audit",
                }
            else:
                row["missed_source_audit"] = {
                    "status": "not_requested",
                    "reason": "the prescribed omitted-source audit is bounded to anchors 0273 and 0653",
                }
            if retain_selected_maps:
                maps, map_selection = _selected_routing_map_arrays(
                    model,
                    outputs,
                    sample,
                    query_np,
                    preparation_records,
                    max_map_values=int(args.max_map_values),
                )
                row["routing_map_selection"] = map_selection
                if maps:
                    maps_dir.mkdir(parents=True, exist_ok=True)
                    map_path = maps_dir / f"{spec.label}__{case_id}.npz"
                    np.savez_compressed(map_path, **maps)
                    row["routing_maps"] = str(map_path)
                    map_paths.append(str(map_path))
            elif args.save_maps:
                row["routing_map_selection"] = {
                    "status": "not_requested",
                    "reason": "detailed route maps are bounded to the five fixed anchors",
                }
            rows.append(row)
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    ledger_csv = Path(args.csv).expanduser().resolve() if args.csv else Path(args.output).expanduser().resolve().with_name("routing_ledger.csv")
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, (Mapping, list))}
        flat_rows.append(flat)
    _write_csv(ledger_csv, flat_rows)
    return {
        "schema_version": 1,
        "task": "dynamic_sparse_routing_ledger",
        "checkpoint": {"label": spec.label, "path": str(spec.path), "epoch": int(checkpoint.get("epoch", -1))},
        "architecture": str(model.config.core_honf.forward_architecture),
        "dataset": str(dataset_path),
        "case_ids": case_ids,
        "query_count": int(args.query_count),
        "missed_source_audit": {
            "requested": bool(getattr(args, "missed_source_audit", True)),
            "scope": "anchors 0273 and 0653",
            "disabled_reason": (
                None
                if bool(getattr(args, "missed_source_audit", True))
                else "omitted-source audit disabled by --no-missed-source-audit"
            ),
        },
        "rows": rows,
        "csv": str(ledger_csv),
        "routing_maps": map_paths,
        "ledger_definitions": {
            "source_incidence_nnz": "positive source-to-hub memberships before query join",
            "raw_two_hop_path_count": "positive query-hub/source-hub paths before receiver-source deduplication",
            "unique_receiver_source_pair_count": "positive receiver-source pairs after stable integer grouping and live prior scatter-add",
            "duplicate_expansion": "raw path count divided by unique pair count",
            "source_fanout_occupancy": "number of positive hubs incident to each valid source row, including zero and singleton rows",
            "source_active_hub_count": "number of occupied hubs after valid-source/candidate masking; retained separately from source fanout",
            "hub_source_occupancy_Dk": "D_k over every valid candidate hub, including zero-occupancy valid hubs; padded candidates are excluded",
            "occupied_hub_source_occupancy_Dk": "D_k restricted to valid occupied hubs, reported alongside the all-valid-hub distribution",
            "membership_zero_fraction": "fraction of zero memberships over valid source rows crossed with valid candidate hubs",
            "query_hub_support": "number of positive query-hub densities in each receiver row; phase rows retain P0/P1/P2 values",
            "query_hub_union_fraction": "union fraction over padded candidate width K_padded; query_hub_union_fraction_activeK uses valid candidate hubs per batch",
            "candidate_hub_count": "active valid routing candidates K; candidate_padded_width is the candidate tensor width",
            "M_active": "number of physical present module source rows",
            "Mpack": "padded module source width used by the encoded dense reference",
            "route_weights": "learned routing priors, not physical influence or field-value substitutes",
        },
    }


def _query_support_mask(outputs: Mapping[str, Any], prefix: str) -> torch.Tensor | None:
    """Return a compact ``[B,Q,K]`` positive query support mask."""

    density = _route_value(outputs, f"{prefix}_query_density")
    if not torch.is_tensor(density) or density.ndim < 3:
        return None
    return (density > 0.0).detach().cpu()


def _support_jaccard(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    """Compare two fixed-shape boolean supports without retaining either map."""

    first = first.to(dtype=torch.bool)
    second = second.to(dtype=torch.bool)
    if tuple(first.shape) != tuple(second.shape):
        return {
            "status": "unavailable",
            "reason": f"support shapes differ: {tuple(first.shape)} vs {tuple(second.shape)}",
        }
    intersection = (first & second).sum().item()
    union = (first | second).sum().item()
    return {
        "status": "ok",
        "intersection_count": int(intersection),
        "union_count": int(union),
        "jaccard": 1.0 if union == 0 else float(intersection / union),
        "changed_fraction": 0.0 if union == 0 else float(1.0 - intersection / union),
    }


def run_turnover(args: argparse.Namespace) -> dict[str, Any]:
    """Measure support turnover across explicitly supplied checkpoint epochs."""

    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) < 2:
        raise ValueError("turnover requires at least two ordered LABEL=PATH checkpoints")
    device = select_device(args.device)
    requested_case_ids = tuple(str(value) for value in (args.case_id or ("0273",)))
    if len(requested_case_ids) != 1:
        raise ValueError(
            "turnover accepts exactly one --case-id per process; run separate processes for additional anchors"
        )
    case_id = requested_case_ids[0]
    records: list[dict[str, Any]] = []
    for spec in specs:
        model, checkpoint = _load_model_spec(spec, device)
        if str(model.config.core_honf.forward_architecture) != "routed_pairwise_honf":
            raise ValueError(
                f"turnover requires routed_pairwise_honf, got {model.config.core_honf.forward_architecture!r}"
            )
        dataset, dataset_path = _load_dataset(checkpoint, _profile_args(args))
        try:
            sample = __import__("channelthermal.evaluation.prepared", fromlist=["select_sample"]).select_sample(
                dataset, case_id, 0
            )
            query_np = _query_points(sample, int(args.query_count))
            with _capture_routing_preparations(model, retain_support=True) as preparation_records:
                outputs = _forward_batch(
                    model,
                    sample,
                    query_np,
                    device,
                    return_prepared_state=False,
                    return_routing_maps=True,
                    return_organizer_passes=True,
                )
            phase_source_support: dict[str, dict[str, torch.Tensor]] = {}
            for phase, summaries in preparation_records.items():
                if not summaries:
                    continue
                latest = summaries[-1]
                support = latest.get("_support_masks")
                if isinstance(support, Mapping):
                    phase_source_support[phase] = {
                        str(source_type): value
                        for source_type, value in support.items()
                        if torch.is_tensor(value)
                    }
            phase_query_support: dict[str, dict[str, torch.Tensor]] = {}
            for phase, prefix in (
                ("p0_port", "initial_port_routing_module"),
                ("p1_refinement", "provisional_routing_module"),
                ("p2_field", "routing_module"),
            ):
                module_support = _query_support_mask(outputs, prefix)
                environment_support = _query_support_mask(
                    outputs, prefix.replace("module", "environment")
                )
                if module_support is not None or environment_support is not None:
                    phase_query_support[phase] = {}
                    if module_support is not None:
                        phase_query_support[phase]["module"] = module_support
                    if environment_support is not None:
                        phase_query_support[phase]["environment"] = environment_support
            phase_summaries: dict[str, Any] = {}
            for phase in sorted(set(phase_source_support) | set(phase_query_support)):
                source_summary: dict[str, Any] = {}
                for source_type, support in phase_source_support.get(phase, {}).items():
                    source_summary[source_type] = {
                        "shape": list(support.shape),
                        "positive_count": int(support.sum().item()),
                    }
                query_summary: dict[str, Any] = {}
                for source_type, support in phase_query_support.get(phase, {}).items():
                    support_flat = support.reshape(
                        int(support.shape[0]), -1, int(support.shape[-1])
                    )
                    query_summary[source_type] = {
                        "shape": list(support.shape),
                        "positive_count": int(support.sum().item()),
                        "hub_union_count": int(support_flat.any(dim=1).sum().item()),
                    }
                phase_summaries[phase] = {
                    "source_support": source_summary,
                    "query_support": query_summary,
                }
            records.append(
                {
                    "label": spec.label,
                    "checkpoint": str(spec.path),
                    "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
                    "dataset": str(dataset_path),
                    "phase_source_support": phase_source_support,
                    "phase_query_support": phase_query_support,
                    "support_summaries": phase_summaries,
                }
            )
            del outputs
        finally:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
            del model

    comparisons: list[dict[str, Any]] = []
    for first, second in pairwise(records):
        phase_rows: dict[str, Any] = {}
        phases = sorted(
            set(first["phase_source_support"])
            | set(second["phase_source_support"])
            | set(first["phase_query_support"])
            | set(second["phase_query_support"])
        )
        for phase in phases:
            source_rows: dict[str, Any] = {}
            query_rows: dict[str, Any] = {}
            for source_type in ("module", "environment"):
                first_source = first["phase_source_support"].get(phase, {}).get(source_type)
                second_source = second["phase_source_support"].get(phase, {}).get(source_type)
                if torch.is_tensor(first_source) and torch.is_tensor(second_source):
                    source_rows[source_type] = _support_jaccard(first_source, second_source)
                first_query = first["phase_query_support"].get(phase, {}).get(source_type)
                second_query = second["phase_query_support"].get(phase, {}).get(source_type)
                if torch.is_tensor(first_query) and torch.is_tensor(second_query):
                    query_rows[source_type] = _support_jaccard(first_query, second_query)
            phase_rows[phase] = {
                "source_support_jaccard": source_rows,
                "query_support_jaccard": query_rows,
            }
        comparisons.append(
            {
                "from": {"label": first["label"], "epoch": first["epoch"]},
                "to": {"label": second["label"], "epoch": second["epoch"]},
                "phases": phase_rows,
            }
        )
    for record in records:
        record.pop("phase_source_support", None)
        record.pop("phase_query_support", None)
    return {
        "schema_version": 1,
        "task": "dynamic_sparse_routing_turnover",
        "case_id": case_id,
        "query_count": int(args.query_count),
        "ordered_checkpoints": records,
        "adjacent_comparisons": comparisons,
        "interpretation": {
            "source_support": "positive ordinary source-to-hub incidence support, compared at fixed source slots",
            "query_support": "positive source-measure sparsemax query-to-hub support, compared at fixed query points",
            "jaccard": "intersection divided by union; changed_fraction is one minus Jaccard",
            "scope": "evaluation-only support turnover; no snapshot, freeze, or monitoring state is created",
        },
    }


def _toy_routing_math_checks() -> dict[str, Any]:
    """Run float64 sparsemax/projection/prior derivative and invariance checks."""

    try:
        from honf_forward_core.interface_fields.routing_index.pair_join import compile_two_hop_pairs
        from honf_forward_core.interface_fields.routing_index.sparse_projection import (
            build_typed_source_incidence,
            ordinary_source_sparsemax,
            source_measure_sparsemax,
        )
    except (ImportError, AttributeError) as exc:
        # The driver is also used while the shared backend is being assembled.
        # Preserve a machine-readable result rather than importing a stale
        # partially written package and claiming that the check ran.
        return {
            "status": "unavailable",
            "reason": f"routing package import incomplete: {type(exc).__name__}: {exc}",
        }

    dtype = torch.float64
    source_logits = torch.tensor([[[2.0, 0.0, -1.0], [0.2, 1.6, -0.5], [1.0, 1.0, -1.0]]], dtype=dtype, requires_grad=True)
    source_weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=dtype, requires_grad=True)
    membership = ordinary_source_sparsemax(source_logits)
    incidence = build_typed_source_incidence(source_weights, membership)
    query_logits = torch.tensor([[[1.4, 0.1, -0.5], [0.0, 1.1, 0.2]]], dtype=dtype, requires_grad=True)
    projection = source_measure_sparsemax(query_logits, incidence.hub_measure)
    pairs = compile_two_hop_pairs(projection.density, incidence)
    objective = pairs.prior.square().sum() + projection.probability.square().sum()
    grad_source_logits, grad_source_weights, grad_query_logits = torch.autograd.grad(
        objective,
        (source_logits, source_weights, query_logits),
        allow_unused=True,
    )
    hub_measure = incidence.hub_measure.detach()
    shifted_query = source_measure_sparsemax(query_logits.detach() + 3.75, hub_measure)
    shifted_source = ordinary_source_sparsemax(source_logits.detach() + 3.75)
    dense_prior = source_weights[:, None, :] * torch.einsum(
        "bqk,bnk->bqn", projection.density, membership
    )
    packed_prior = dense_prior[
        pairs.batch_index,
        pairs.receiver_index,
        pairs.source_index,
    ]
    prior_oracle_error = float(
        (packed_prior.detach() - pairs.prior.detach()).abs().max().cpu()
    ) if pairs.prior.numel() else 0.0
    split_weights = torch.cat(
        (source_weights[:, :1] * 0.5, source_weights[:, :1] * 0.5, source_weights[:, 1:]),
        dim=1,
    )
    split_membership = torch.cat(
        (membership[:, :1], membership[:, :1], membership[:, 1:]),
        dim=1,
    )
    split_incidence = build_typed_source_incidence(split_weights, split_membership)
    split_projection = source_measure_sparsemax(query_logits.detach(), split_incidence.hub_measure.detach())
    split_dense_prior = split_weights[:, None, :] * torch.einsum(
        "bqk,bnk->bqn", split_projection.density, split_membership
    )
    split_aggregate = torch.cat(
        (split_dense_prior[..., :2].sum(dim=-1, keepdim=True), split_dense_prior[..., 2:]),
        dim=-1,
    )
    split_error = float((split_aggregate.detach() - dense_prior.detach()).abs().max().cpu())
    return {
        "dtype": str(dtype),
        "ordinary_row_mass": membership.sum(dim=-1).detach().cpu().tolist(),
        "query_probability_mass": projection.probability.sum(dim=-1).detach().cpu().tolist(),
        "support": projection.support.detach().cpu().tolist(),
        "raw_path_count": pairs.raw_path_count,
        "unique_pair_count": pairs.unique_pair_count,
        "duplicate_expansion": pairs.duplicate_expansion,
        "prior_dense_oracle_max_abs": prior_oracle_error,
        "prior_matches_dense_oracle": bool(prior_oracle_error <= 1.0e-12),
        "split_source_quadrature_max_abs": split_error,
        "split_source_quadrature_invariant": bool(split_error <= 1.0e-12),
        "source_logit_gradient_finite": bool(
            grad_source_logits is not None and torch.isfinite(grad_source_logits).all()
        ),
        "source_measure_gradient_finite": bool(
            grad_source_weights is not None and torch.isfinite(grad_source_weights).all()
        ),
        "query_logit_gradient_finite": bool(
            grad_query_logits is not None and torch.isfinite(grad_query_logits).all()
        ),
        "gradients_finite": bool(
            grad_source_logits is not None
            and grad_source_weights is not None
            and grad_query_logits is not None
            and torch.isfinite(grad_source_logits).all()
            and torch.isfinite(grad_source_weights).all()
            and torch.isfinite(grad_query_logits).all()
        ),
        "common_shift_invariant": bool(
            torch.allclose(membership.detach(), shifted_source, atol=1e-12, rtol=1e-12)
            and torch.allclose(projection.probability.detach(), shifted_query.probability, atol=1e-12, rtol=1e-12)
        ),
        "source_measure_query_threshold": projection.threshold.detach().cpu().tolist(),
        "prior_requires_grad": bool(pairs.prior.requires_grad),
    }


def _model_gradient_probe(
    model: Any,
    dataset: Any,
    case_id: str,
    device: torch.device,
    *,
    query_count: int,
    step: float,
) -> dict[str, Any]:
    sample = __import__("channelthermal.evaluation.prepared", fromlist=["select_sample"]).select_sample(dataset, case_id, 0)
    query_np = _query_points(sample, query_count)
    query = torch.from_numpy(query_np).unsqueeze(0).to(device)
    centers_np, active = _active_centers(sample)
    centers = torch.from_numpy(centers_np.astype(np.float32)).unsqueeze(0).to(device).requires_grad_(True)
    outputs = _forward_tensor_batch(model, sample, query, device, centers=centers, return_prepared_state=False)
    scalar = outputs["pred_field"].square().mean()
    gradient = torch.autograd.grad(scalar, centers, allow_unused=True)[0]
    if gradient is None:
        return {"case_id": case_id, "status": "unavailable", "reason": "center gradient is disconnected"}
    index = int(np.flatnonzero(active)[0])
    direction = torch.zeros_like(centers)
    direction[0, index, 0] = float(step)
    with torch.no_grad():
        plus = _forward_tensor_batch(model, sample, query, device, centers=centers.detach() + direction, return_prepared_state=False)["pred_field"].square().mean()
        minus = _forward_tensor_batch(model, sample, query, device, centers=centers.detach() - direction, return_prepared_state=False)["pred_field"].square().mean()
    ad = float(gradient[0, index, 0].detach().cpu())
    fd = float(((plus - minus) / (2.0 * float(step))).detach().cpu())
    return {
        "case_id": case_id,
        "status": "ok",
        "active_module_index": index,
        "step": float(step),
        "ad": ad,
        "fd": fd,
        "absolute_error": abs(ad - fd),
        "relative_error": abs(ad - fd) / max(abs(ad), abs(fd), 1.0e-12),
        "gradient_finite": bool(torch.isfinite(gradient).all()),
    }


def run_gradients(args: argparse.Namespace) -> dict[str, Any]:
    """Run fixed-source math checks plus bounded model position AD/FD probes."""

    try:
        from routing_numerical_checks import run_routing_numerical_checks

        numerical_output = (
            Path(args.numerics_output).expanduser().resolve()
            if args.numerics_output
            else Path(args.output).expanduser().resolve().with_name("routing_numerical_checks.json")
        )
        math_checks = run_routing_numerical_checks(output=numerical_output)
    except (ImportError, AttributeError) as exc:
        # Keep the main driver usable during an incremental checkout while
        # making the missing prescribed helper explicit in the artifact.
        math_checks = {
            "status": "unavailable",
            "reason": f"routing numerical helper unavailable: {type(exc).__name__}: {exc}",
        }
    if not args.checkpoint:
        return {
            "schema_version": 1,
            "task": "dynamic_sparse_routing_gradients",
            "numerical": math_checks,
            "math": math_checks,
            "model": {"status": "not_requested", "reason": "pass --checkpoint for physical position probes"},
        }
    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 1:
        raise ValueError("gradients accepts one checkpoint")
    device = select_device(args.device)
    spec = specs[0]
    model, checkpoint = _load_model_spec(spec, device)
    dataset, dataset_path = _load_dataset(checkpoint, _profile_args(args))
    rows: list[dict[str, Any]] = []
    physical_derivatives: dict[str, Any]
    try:
        for case_id in (args.case_id or ("0273", "0298")):
            try:
                rows.append(
                    _model_gradient_probe(
                        model,
                        dataset,
                        str(case_id),
                        device,
                        query_count=int(args.query_count),
                        step=float(args.fd_step),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - preserve per-case probe failure
                rows.append({"case_id": str(case_id), "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        try:
            from routing_derivative_checks import run_anchor_derivative_checks

            physical_derivatives = run_anchor_derivative_checks(
                model,
                checkpoint,
                dataset,
                device,
                case_ids=tuple(str(value) for value in (args.case_id or ("0273", "0298"))),
                query_count=int(args.query_count),
            )
        except ImportError as exc:
            physical_derivatives = {
                "status": "unavailable",
                "reason": f"routing derivative helper unavailable: {type(exc).__name__}: {exc}",
            }
        except Exception as exc:  # noqa: BLE001 - preserve physical derivative failure
            physical_derivatives = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    return {
        "schema_version": 1,
        "task": "dynamic_sparse_routing_gradients",
        "checkpoint": {"label": spec.label, "path": str(spec.path), "epoch": int(checkpoint.get("epoch", -1))},
        "dataset": str(dataset_path),
        "numerical": math_checks,
        "math": math_checks,
        "position_ad_fd": rows,
        "physical_derivatives": physical_derivatives,
        "interpretation": {
            "model_ad_fd": "model-versus-itself derivative self-consistency only; it is not a physical gradient certificate",
            "support_boundaries": "integer support changes are reported by the route ledger and are not smoothed with a straight-through estimator",
        },
    }


def run_figures(args: argparse.Namespace) -> dict[str, Any]:
    """Delegate all routing figures to the shared renderer implementation."""

    from render_routing_diagnostics import render_routing_diagnostics

    ledger_path = Path(args.ledger).expanduser().resolve()
    figure_dir = Path(args.figure_dir).expanduser().resolve()
    endpoint_links = None
    if args.endpoint_links is not None:
        endpoint_links = json.loads(
            Path(args.endpoint_links).expanduser().resolve().read_text(encoding="utf-8")
        )
        if not isinstance(endpoint_links, Mapping):
            raise ValueError("--endpoint-links JSON must contain a case-to-link object")

    renderer_parameters = inspect.signature(render_routing_diagnostics).parameters
    accepts_turnover = "turnover" in renderer_parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in renderer_parameters.values()
    )
    turnover = [Path(value).expanduser().resolve() for value in (args.turnover or [])]
    if turnover and not accepts_turnover:
        return {
            "schema_version": 1,
            "task": "dynamic_sparse_routing_figures",
            "status": "unavailable",
            "reason": "shared renderer does not yet expose its turnover argument",
            "ledger": str(ledger_path),
            "figure_dir": str(figure_dir),
            "turnover": [str(value) for value in turnover],
        }
    renderer_kwargs: dict[str, Any] = {
        "output_dir": figure_dir,
        "endpoint_links": endpoint_links,
        "expected_case_ids": tuple(str(value) for value in (args.case_id or ANCHOR_CASE_IDS)),
        "strict": bool(args.strict),
    }
    if turnover:
        renderer_kwargs["turnover"] = turnover
    manifest = render_routing_diagnostics(ledger_path, **renderer_kwargs)
    return {
        "schema_version": 1,
        "task": "dynamic_sparse_routing_figures",
        "renderer": "tools/diagnostics/render_routing_diagnostics.py",
        "ledger": str(ledger_path),
        "figure_dir": str(figure_dir),
        "turnover": [str(value) for value in turnover],
        "manifest": manifest,
        "interpretation": "Route weights and occupancy are learned routing diagnostics; they are not field values or physical influence maps.",
    }


def _add_common(parser: argparse.ArgumentParser, *, default_device: str = "cpu") -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-count", type=int, default=8192)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)

    smoke = subparsers.add_parser("smoke", help="two disposable real physical AdamW checks")
    _add_common(smoke)
    smoke.add_argument("--profile", default=DEFAULT_PROFILE)
    smoke.add_argument("--points-per-case", type=int, default=1024)
    smoke.add_argument("--batch-size", type=int, default=48)
    smoke.add_argument("--case-count", type=int, default=48)
    smoke.add_argument("--batch-kind", choices=("both", "small", "large"), default="both")
    smoke.add_argument("--small-case-id", action="append", default=None)
    smoke.add_argument("--large-case-id", action="append", default=None)
    smoke.set_defaults(handler=run_smoke)

    profile = subparsers.add_parser("profile", help="unprofiled phase timing and optional short profiler traces")
    _add_common(profile)
    profile.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    profile.add_argument("--profile", default=DEFAULT_PROFILE)
    profile.add_argument("--points-per-case", type=int, default=1)
    profile.add_argument("--warmups", type=int, default=2)
    profile.add_argument("--repetitions", type=int, default=5)
    profile.add_argument(
        "--reference-warmups",
        type=int,
        default=1,
        help="warmups for the active-only inherited Dense QM reference",
    )
    profile.add_argument(
        "--reference-repetitions",
        type=int,
        default=3,
        help="repetitions for the active-only inherited Dense QM reference",
    )
    profile.add_argument(
        "--active-only-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the padded-versus-active-only inherited Dense QM reference",
    )
    profile.add_argument("--large-warmups", type=int, default=1)
    profile.add_argument("--large-repetitions", type=int, default=3)
    profile.add_argument("--receiver-chunk-size", type=int, default=DEFAULT_RECEIVER_CHUNK_SIZE)
    profile.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    profile.add_argument(
        "--large-shape",
        type=int,
        nargs=3,
        action="append",
        default=None,
        metavar=("M", "E", "Q"),
        help="optional actual synthetic shape; pass 128 3072 262144 for the prescribed large workload",
    )
    profile.add_argument("--large-query-count", type=int, default=None)
    profile.add_argument("--trace", action="store_true", help="capture one Chrome trace per real anchor and large workload")
    profile.add_argument("--trace-dir", type=Path, default=None)
    profile.set_defaults(handler=run_profile)

    endpoint = subparsers.add_parser("endpoint", help="complete 90-case exact and saved-best endpoint via existing evaluation")
    _add_common(endpoint, default_device="cuda:0")
    endpoint.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    endpoint.add_argument("--exact-checkpoint", action="append", default=[], metavar="LABEL=PATH")
    endpoint.add_argument("--best-checkpoint", action="append", default=[], metavar="LABEL=PATH")
    endpoint.add_argument("--evaluation-dir", type=Path, default=None)
    endpoint.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    endpoint.add_argument("--anchor-case-id", action="append", default=None)
    endpoint.add_argument("--skip-figures", action="store_true")
    endpoint.set_defaults(handler=run_endpoint)

    interventions = subparsers.add_parser("interventions", help="fixed-anchor P0/P1/P2 routing interventions")
    _add_common(interventions)
    interventions.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    interventions.add_argument("--mode", action="append", choices=("uniform_p2", "qm_zero_p2", "qe_zero_p2", "uniform_p0", "uniform_p1", "coarse_zero_p2"), default=None)
    interventions.set_defaults(handler=run_interventions)

    ledger = subparsers.add_parser("ledger", help="sparse source/occupancy/path work ledger")
    _add_common(ledger)
    ledger.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    ledger.add_argument("--maps-dir", type=Path, default=None)
    ledger.add_argument("--csv", type=Path, default=None)
    ledger.add_argument("--save-maps", action="store_true")
    ledger.add_argument("--max-map-values", type=int, default=2_000_000)
    ledger.add_argument(
        "--missed-source-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the bounded omitted-source audit on anchors 0273 and 0653",
    )
    ledger.set_defaults(handler=run_ledger, query_count=32)

    turnover = subparsers.add_parser(
        "turnover",
        help="fixed-anchor source/query support Jaccard across ordered checkpoints",
    )
    _add_common(turnover)
    turnover.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    turnover.set_defaults(handler=run_turnover, query_count=32)

    gradients = subparsers.add_parser("gradients", help="sparsemax, two-hop prior, and model AD/FD probes")
    _add_common(gradients)
    gradients.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    gradients.add_argument("--fd-step", type=float, default=1.0e-4)
    gradients.add_argument(
        "--numerics-output",
        type=Path,
        default=None,
        help="optional path for the standalone CPU routing numerical-check artifact",
    )
    gradients.set_defaults(handler=run_gradients, query_count=32)

    figures = subparsers.add_parser("figures", help="compact routing ledger figures")
    figures.add_argument("--ledger", type=Path, required=True)
    figures.add_argument("--figure-dir", type=Path, required=True)
    figures.add_argument("--endpoint-links", type=Path, default=None)
    figures.add_argument("--turnover", action="append", type=Path, default=None, metavar="JSON")
    figures.add_argument("--case-id", action="append", default=None)
    figures.add_argument("--strict", action="store_true")
    figures.add_argument("--output", type=Path, required=True)
    figures.set_defaults(handler=run_figures)
    return parser


def _status_counts(value: Any) -> tuple[int, int]:
    """Count genuine failures separately from unavailable optional hooks."""

    failed = 0
    unavailable = 0
    if isinstance(value, Mapping):
        status = str(value.get("status", ""))
        if status in {"failed", "error", "out_of_memory"}:
            failed += 1
        elif status in {"unavailable", "unsupported", "not_requested", "not_applicable"}:
            unavailable += 1
        for item in value.values():
            nested_failed, nested_unavailable = _status_counts(item)
            failed += nested_failed
            unavailable += nested_unavailable
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested_failed, nested_unavailable = _status_counts(item)
            failed += nested_failed
            unavailable += nested_unavailable
    return failed, unavailable


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = args.handler(args)
    failed, unavailable = _status_counts(payload)
    payload["status"] = "failed" if failed else "ok"
    payload["failure_count"] = int(failed)
    payload["unavailable_count"] = int(unavailable)
    payload["output"] = str(Path(args.output).expanduser().resolve())
    write_json(Path(args.output), payload)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
