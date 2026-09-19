"""Run-1406 same-checkpoint executor parity and unprofiled evidence.

This tool is deliberately separate from the Dense-1804 development comparison.
It loads one explicitly supplied epoch-50 Run-1406 checkpoint through the
maintained trusted/strict evaluation loader, then runs two isolated workers:

* ``reference``: a temporary archive of the pre-optimization commit;
* ``optimized``: the current working source tree.

The workers use the same cases and batches.  Their tensor artifacts are
compared in the parent process, so importing the reference and optimized
executor into one Python interpreter is never attempted.  Timed calls request
``return_routing_maps=False`` and do not request a detailed execution ledger;
the parity call also keeps maps disabled while exposing the P0/P1/P2 and
port-global output contracts.

Normal invocation performs no training update, no managed job launch, and no
profiler trace.  The one predicted-port backward pass is a real loss backward
on a deterministic data-loader batch; its optimizer is intentionally absent
so the supplied checkpoint remains unchanged.

Plan-only validation is safe on a CPU-only host::

    python tools/diagnostics/run_run1406_executor_evidence.py \
        --checkpoint /path/to/Run1406/epoch_0050_model.pt \
        --output diagnostics/generated/run1406_executor/evidence.json \
        --plan-only

The physical command is emitted by the plan and report.  It requires the
caller to select a CUDA device and supply the existing epoch-50 checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_CASE_IDS = ("0273", "0653")
DEFAULT_EPOCH = 50
DEFAULT_QUERY_COUNT = 8192
DEFAULT_RECEIVER_CHUNK_SIZE = 2048
DEFAULT_WARMUPS = 2
DEFAULT_REPETITIONS = 5
DEFAULT_TRAIN_BATCH_SIZE = 48
DEFAULT_TRAIN_QUERY_COUNT = 1024
DEFAULT_BACKWARD_MODULE_COUNT = 12

FIELD_OUTPUT_KEYS = (
    "pred_field",
    "pred_internal_temperature",
    "pred_interface",
    "pred_port_condition",
    "pred_port_condition_raw",
    "local_port_condition_used",
    "pred_port_global_temperature",
    "pred_port_global_temperature_target",
    "pred_port_global_consistency_mask",
    "predicted_port_internal_temperature",
    "predicted_port_interface",
)

# These are execution counters/ledger fields, not physical predictions.  They
# must never be part of a timed parity/benchmark read and need not be tensor
# compared when the optimized reader intentionally omits them.
LEDGER_MARKERS = (
    "logical_paths",
    "unique_pairs",
    "pair_count",
    "fine_rows",
    "forward_rows",
    "padded",
    "denominator",
    "recomput",
    "complete_support",
    "partial_support",
    "scalar_control_rows",
    "source_projection_rows",
    "geometry_rows",
    "content_dot_rows",
    "qm_",
    "kv_projection",
)


def _finite(value: Any) -> float | None:
    try:
        scalar = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return scalar if math.isfinite(scalar) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
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
        raise FileNotFoundError(f"Run-1406 checkpoint does not exist: {path}")
    return path


def _validate_protocol(args: argparse.Namespace) -> None:
    cases = tuple(str(value) for value in (args.case_id or DEFAULT_CASE_IDS))
    if set(cases) != set(DEFAULT_CASE_IDS) or len(cases) != len(DEFAULT_CASE_IDS):
        raise ValueError("the executor evidence protocol requires exactly cases 0273 and 0653")
    expected = {
        "query_count": DEFAULT_QUERY_COUNT,
        "receiver_chunk_size": DEFAULT_RECEIVER_CHUNK_SIZE,
        "warmups": DEFAULT_WARMUPS,
        "repetitions": DEFAULT_REPETITIONS,
        "train_batch_size": DEFAULT_TRAIN_BATCH_SIZE,
        "train_query_count": DEFAULT_TRAIN_QUERY_COUNT,
        "backward_module_count": DEFAULT_BACKWARD_MODULE_COUNT,
    }
    for name, value in expected.items():
        if int(getattr(args, name)) != value:
            raise ValueError(f"controlled protocol requires {name}={value}, got {getattr(args, name)}")


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_epoch": DEFAULT_EPOCH,
        "case_ids": [str(value) for value in (args.case_id or DEFAULT_CASE_IDS)],
        "query_count": int(args.query_count),
        "receiver_chunk_size": int(args.receiver_chunk_size),
        "inference_warmups": int(args.warmups),
        "inference_repetitions": int(args.repetitions),
        "timed_maps": False,
        "timed_detailed_ledgers": False,
        "timed_profiler": False,
        "timed_port_global_consistency": False,
        "backward": {
            "batch_size": int(args.train_batch_size),
            "query_count": int(args.train_query_count),
            "module_count": int(args.backward_module_count),
            "port_condition_mode": "predicted",
            "optimizer_update": False,
        },
        "checkpoint_policy": "one explicit checkpoint; trusted load; strict model_state_dict reconstruction",
        "executor_order": ["reference_pre_optimization", "optimized_working_tree"],
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _parse_checkpoint(args.checkpoint)
    _validate_protocol(args)
    output = Path(args.output).expanduser().resolve()
    return {
        "schema_version": 1,
        "task": "run1406_executor_parity_and_unprofiled_evidence",
        "status": "plan_only",
        "checkpoint_policy": {
            "path": str(checkpoint),
            "selection_policy": "explicit_cli_checkpoint",
            "required_epoch": DEFAULT_EPOCH,
            "strict_load": True,
            "trusted_loader": "channelthermal.evaluation.loading.load_model -> load_trusted_checkpoint",
        },
        "executor_policy": {
            "reference_commit": str(args.reference_commit),
            "reference_source": (
                str(Path(args.reference_source).expanduser().resolve())
                if args.reference_source
                else "temporary git archive of --reference-commit"
            ),
            "optimized_source": str(PROJECT_ROOT),
            "comparison_isolation": "one subprocess per source tree; no mixed module cache",
            "baseline_artifact_policy": "temporary source archive only; no tracked snapshot",
        },
        "protocol": _protocol(args),
        "parity_contract": {
            "cases": list(DEFAULT_CASE_IDS),
            "phase_outputs": ["P0 predicted port context", "P1 refinement", "P2 field", "P2 port-global consistency"],
            "tensors": list(FIELD_OUTPUT_KEYS),
            "field_relative_tolerance": 2.0e-6,
            "derived_output_relative_tolerance": 2.0e-5,
            "derived_output_absolute_tolerance": 2.0e-6,
            "gradient_relative_tolerance": 1.0e-4,
            "near_zero_gradient_absolute_tolerance": 1.0e-6,
            "semantic_note": "phase diagnostics are execution evidence; learned routes are not physical causality",
        },
        "planned_outputs": {
            "evidence_json": str(output),
            "report": str(Path(args.report).expanduser().resolve() if args.report else output.with_suffix(".md")),
        },
        "limitations": [
            "No profiler trace is taken by this tool.",
            "The benchmark has no detailed ledger/map request in timed reads.",
            "Parity is a same-checkpoint executor check, not a retraining or CFD validation result.",
        ],
    }


def _configure_source_root(source_root: Path) -> None:
    """Make a worker import only project modules from ``source_root``."""

    source_root = source_root.expanduser().resolve()
    paths = (
        source_root / "src",
        source_root / "Case_ThermalChannel" / "src",
        source_root / "tools" / "diagnostics",
        source_root / "tools",
    )
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _sync(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _measure_phase(torch: Any, function: Any, device: Any, *, warmups: int, repetitions: int) -> dict[str, Any]:
    """Measure only the requested unprofiled function at synchronized boundaries."""

    with torch.inference_mode():
        for _ in range(int(warmups)):
            result = function()
            del result
        _sync(torch, device)
    samples: list[dict[str, Any]] = []
    for index in range(int(repetitions)):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
        else:
            baseline_allocated = None
            baseline_reserved = None
        _sync(torch, device)
        started = time.perf_counter()
        error: str | None = None
        result: Any = None
        try:
            with torch.inference_mode():
                result = function()
        except Exception as exc:  # noqa: BLE001 - preserve bounded evidence
            error = f"{type(exc).__name__}: {exc}"
        _sync(torch, device)
        elapsed = time.perf_counter() - started
        peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
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
                    None if peak_allocated is None or baseline_allocated is None else peak_allocated - baseline_allocated
                ),
                "incremental_peak_reserved_bytes": (
                    None if peak_reserved is None or baseline_reserved is None else peak_reserved - baseline_reserved
                ),
            }
        )
        del result
        if error:
            break
    elapsed = [row["elapsed_seconds"] for row in samples if row["status"] == "complete"]
    return {
        "status": "complete" if len(elapsed) == int(repetitions) else "incomplete",
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "samples": samples,
        "median_seconds": None if not elapsed else float(np.median(elapsed)),
        "min_seconds": None if not elapsed else float(np.min(elapsed)),
        "max_seconds": None if not elapsed else float(np.max(elapsed)),
        "peak_allocated_bytes": max(
            (row["peak_allocated_bytes"] for row in samples if row["peak_allocated_bytes"] is not None),
            default=None,
        ),
        "peak_reserved_bytes": max(
            (row["peak_reserved_bytes"] for row in samples if row["peak_reserved_bytes"] is not None),
            default=None,
        ),
        "timed_maps": False,
        "timed_detailed_ledgers": False,
        "timed_profiler": False,
    }


def _tensor_summary(value: Any) -> dict[str, Any]:
    import torch

    if not torch.is_tensor(value):
        return {"kind": type(value).__name__}
    tensor = value.detach()
    finite = bool(torch.isfinite(tensor).all()) if tensor.is_floating_point() else True
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
        "norm": float(torch.linalg.vector_norm(tensor.float()).cpu()) if tensor.numel() else 0.0,
        "min": float(tensor.float().min().cpu()) if tensor.numel() and tensor.is_floating_point() else None,
        "max": float(tensor.float().max().cpu()) if tensor.numel() and tensor.is_floating_point() else None,
    }


def _is_ledger_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(marker in lowered for marker in LEDGER_MARKERS)


def _collect_worker_outputs(output: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Collect physical outputs plus non-ledger P0/P1/P2 tensors."""

    import torch

    physical: dict[str, torch.Tensor] = {}
    for key in FIELD_OUTPUT_KEYS:
        value = output.get(key)
        if torch.is_tensor(value):
            physical[key] = value.detach().cpu()
    provisional: dict[str, torch.Tensor] = {}
    provisional_aux = output.get("provisional_read_aux", {})
    if isinstance(provisional_aux, Mapping):
        for key, value in provisional_aux.items():
            if (str(key) == "pred_field" or str(key).startswith("pred_")) and torch.is_tensor(value):
                provisional[str(key)] = value.detach().cpu()
    phase_tensors: dict[str, torch.Tensor] = {}
    phase_summary: dict[str, Any] = {}
    interaction_aux = output.get("interaction_aux", {})
    if isinstance(interaction_aux, Mapping):
        for key, value in interaction_aux.items():
            name = str(key)
            phase_summary[name] = _tensor_summary(value)
            if not _is_ledger_key(name) and torch.is_tensor(value) and value.numel() <= 2_000_000:
                phase_tensors[name] = value.detach().cpu()
    return {"physical": physical, "provisional": provisional}, phase_tensors, phase_summary


def _phase_request(batch: Mapping[str, Any], query: Any, *, consistency: bool, prepared: bool = False) -> dict[str, Any]:
    """Create a controlled model request; maps/ledgers are explicitly off."""

    return {
        "structure": batch["structure"],
        "query_xy": query,
        "interface_condition": batch.get("interface_condition"),
        "local_module_params": batch.get("local_module_params"),
        "teacher_port_tokens": batch.get("teacher_port_tokens"),
        "local_query_points": batch.get("module_internal_query_points"),
        "local_port_condition_mode": "predicted",
        "mixed_teacher_ratio": 0.0,
        "return_predicted_port_outputs": True,
        "return_port_global_consistency": bool(consistency),
        "return_prepared_state": bool(prepared),
        "return_routing_maps": False,
        "return_organizer_passes": bool(consistency),
    }


def _extract_prepared(output: Any) -> Any:
    if isinstance(output, Mapping):
        return output["prepared_state"]
    return output.prepared_state


def _load_runtime(source_root: Path) -> tuple[Any, Any, Any, Any, Any, Any]:
    _configure_source_root(source_root)
    import benchmark_routing_optimization as train_bench
    import run_dynamic_sparse_routing_study as dynamic
    import run_run1405_epoch50_comparison as run1405
    import run_stage3_interface_study as stage3
    import torch
    from channelthermal.evaluation.loading import make_batch
    from channelthermal.evaluation.prepared import select_sample

    return torch, train_bench, dynamic, run1405, stage3, (make_batch, select_sample)


def _epoch_of(checkpoint: Mapping[str, Any]) -> int:
    try:
        return int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1)))
    except (TypeError, ValueError):
        return -1


def _worker(args: argparse.Namespace) -> int:
    """Run one source-isolated worker and save compact tensors for comparison."""

    source_root = Path(args.source_root).expanduser().resolve()
    torch, train_bench, dynamic, run1405, stage3, loaders = _load_runtime(source_root)
    make_batch, select_sample = loaders
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device={args.device}")
    checkpoint_path = _parse_checkpoint(args.checkpoint)
    model, checkpoint = stage3._load_model_spec(
        stage3.CheckpointSpec(label="1406", path=checkpoint_path), device
    )
    epoch = _epoch_of(checkpoint)
    if epoch != DEFAULT_EPOCH:
        raise ValueError(f"Run-1406 checkpoint epoch={epoch}; strict evidence requires exact epoch 50")
    state_keys = tuple(model.state_dict().keys())
    load_record = {
        "status": "pass",
        "strict": True,
        "trusted_loader": "channelthermal.evaluation.loading.load_model",
        "checkpoint_epoch": epoch,
        "model_state_key_count": len(state_keys),
    }
    dataset, _ = stage3._load_dataset(
        checkpoint,
        SimpleNamespace(dataset=args.dataset, split=args.eval_split),
    )
    model.eval()
    inference: dict[str, Any] = {}
    saved_physical: dict[str, Any] = {}
    saved_phase: dict[str, Any] = {}
    phase_summaries: dict[str, Any] = {}
    try:
        for case_id in DEFAULT_CASE_IDS:
            sample = select_sample(dataset, str(case_id), 0)
            query_np = dynamic._query_points(sample, DEFAULT_QUERY_COUNT)
            if len(query_np) != DEFAULT_QUERY_COUNT:
                raise ValueError(f"case {case_id} produced Q={len(query_np)}; expected {DEFAULT_QUERY_COUNT}")
            batch = make_batch(dict(sample), query_np, device)
            query = batch["query_xy"]

            # Untimed parity request: maps and detailed ledgers remain off,
            # while P0/P1/P2 and the port-global consistency probe are exposed.
            with torch.inference_mode(), dynamic._runtime_receiver_chunk_size(model, DEFAULT_RECEIVER_CHUNK_SIZE):
                parity_output = model(**_phase_request(batch, query, consistency=True))
            collected, phase_tensors, phase_summary = _collect_worker_outputs(parity_output)
            saved_physical[case_id] = collected
            saved_phase[case_id] = phase_tensors
            phase_summaries[case_id] = phase_summary

            prepared_holder: dict[str, Any] = {}

            def full_forward(*, _batch: Mapping[str, Any] = batch, _query: Any = query) -> Any:
                with dynamic._runtime_receiver_chunk_size(model, DEFAULT_RECEIVER_CHUNK_SIZE):
                    return model(**_phase_request(_batch, _query, consistency=False))

            def prepare_one(*, _batch: Mapping[str, Any] = batch, _query: Any = query) -> Any:
                with dynamic._runtime_receiver_chunk_size(model, DEFAULT_RECEIVER_CHUNK_SIZE):
                    return model(**_phase_request(_batch, _query[:, :1], consistency=False, prepared=True))

            def prepared_decode(*, _holder: dict[str, Any] = prepared_holder, _query: Any = query) -> Any:
                with dynamic._runtime_receiver_chunk_size(model, DEFAULT_RECEIVER_CHUNK_SIZE):
                    return model.decode_prepared(
                        _holder["prepared"],
                        _query,
                        return_routing_maps=False,
                        receiver_chunk_size=DEFAULT_RECEIVER_CHUNK_SIZE,
                    )

            phases = {
                "full_physical_forward": _measure_phase(
                    torch,
                    full_forward,
                    device,
                    warmups=DEFAULT_WARMUPS,
                    repetitions=DEFAULT_REPETITIONS,
                ),
                "physical_preparation_plus_one_query": _measure_phase(
                    torch,
                    prepare_one,
                    device,
                    warmups=DEFAULT_WARMUPS,
                    repetitions=DEFAULT_REPETITIONS,
                ),
            }
            with torch.inference_mode():
                prepared_output = prepare_one()
            prepared_holder["prepared"] = _extract_prepared(prepared_output)
            phases["prepared_p2_decode"] = _measure_phase(
                torch,
                prepared_decode,
                device,
                warmups=DEFAULT_WARMUPS,
                repetitions=DEFAULT_REPETITIONS,
            )
            prepared_holder.clear()
            del prepared_output
            inference[case_id] = {
                "case_id": case_id,
                "query_count": int(query.shape[1]),
                "receiver_chunk_size": DEFAULT_RECEIVER_CHUNK_SIZE,
                "phases": phases,
                "timed_maps": False,
                "timed_detailed_ledgers": False,
                "timed_profiler": False,
                "port_condition_mode": "predicted",
            }
            del parity_output, batch
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()

    # One real, deterministic, predicted-port backward batch.  The loader and
    # canonical loss assembly are shared with the maintained training path;
    # there is intentionally no optimizer step.
    train_dataset, _ = stage3._load_dataset(
        checkpoint,
        SimpleNamespace(dataset=args.dataset, split=args.train_split),
        points_per_case_override=DEFAULT_TRAIN_QUERY_COUNT,
        random_point_sampling_override=False,
    )
    train_dataset.include_grid = False
    try:
        buckets = run1405._make_exact_training_buckets(
            [train_dataset], batch_size=DEFAULT_TRAIN_BATCH_SIZE
        )
        bucket = next(
            (item for item in buckets if int(item.module_count) == DEFAULT_BACKWARD_MODULE_COUNT),
            None,
        )
        if bucket is None:
            raise ValueError("no real training bucket with M=12 for the required backward evidence")
        loader = train_bench._build_loader(
            train_dataset,
            checkpoint,
            bucket,
            batch_size=DEFAULT_TRAIN_BATCH_SIZE,
        )
        batch = next(iter(loader))
        batch = train_bench.recursive_to_device(batch, device) if hasattr(train_bench, "recursive_to_device") else _recursive_to_device(batch, device)
        training_config, loss_config, mode, ratio, internal_weight, interface_weight, predicted_weight = run1405._matched_train_schedule(checkpoint)
        del training_config
        if str(mode).lower() != "predicted" or float(ratio) != 0.0:
            raise ValueError(f"matched epoch-50 schedule is not predicted mode: mode={mode!r}, ratio={ratio}")
        from channelthermal.training import epoch as epoch_training

        port_global_weight = epoch_training.effective_port_global_weight(loss_config, "predicted", 0.0)
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(
            **epoch_training.make_model_inputs(
                batch,
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.0,
                return_predicted_port_outputs=bool(predicted_weight > 0.0),
                return_port_global_consistency=bool(port_global_weight != 0.0),
            )
        )
        terms = epoch_training.assemble_channelthermal_loss_terms(
            output,
            batch,
            model,
            loss_config,
            local_port_condition_mode="predicted",
            mixed_teacher_ratio=0.0,
            effective_internal_temperature_weight=float(internal_weight),
            effective_interface_weight=float(interface_weight),
            predicted_consistency_weight=float(predicted_weight),
        )
        terms["loss"].backward()
        gradients = {
            name: parameter.grad.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        none_gradients = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None
        ]
        backward = {
            "status": "pass",
            "batch_size": int(batch["query_xy"].shape[0]),
            "query_count": int(batch["query_xy"].shape[1]),
            "module_count": int(bucket.module_count),
            "case_ids": list(bucket.case_ids),
            "port_condition_mode": "predicted",
            "optimizer_update": False,
            "loss_terms": {
                key: float(value.detach().cpu())
                for key, value in terms.items()
                if torch.is_tensor(value) and value.numel() == 1
            },
            "gradient_count": len(gradients),
            "none_gradient_count": len(none_gradients),
            "none_gradients": none_gradients,
            "gradients": gradients,
        }
    finally:
        close = getattr(train_dataset, "close", None)
        if callable(close):
            close()
    raw_path = Path(args.raw_output).expanduser().resolve()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "label": str(args.worker_label),
            "source_root": str(source_root),
            "load": load_record,
            "inference": inference,
            "physical": saved_physical,
            "phase_tensors": saved_phase,
            "phase_summaries": phase_summaries,
            "backward": backward,
        },
        raw_path,
    )
    print(json.dumps({"status": "complete", "label": args.worker_label, "raw_output": str(raw_path)}))
    return 0


def _recursive_to_device(value: Any, device: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _recursive_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_recursive_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_recursive_to_device(item, device) for item in value)
    return value


def _safe_extract_archive(archive_bytes: bytes, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != destination.resolve() and destination.resolve() not in target.parents:
                raise RuntimeError(f"reference archive contains unsafe member: {member.name}")
        archive.extractall(destination)
    source = destination / "HONF_Proj"
    if not source.is_dir():
        raise RuntimeError(f"reference commit archive did not contain HONF_Proj/: {source}")
    return source


def _reference_source(args: argparse.Namespace, temporary_root: Path) -> tuple[Path, str]:
    if args.reference_source:
        source = Path(args.reference_source).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"reference source directory does not exist: {source}")
        return source, "explicit_reference_source"
    command = ["git", "archive", "--format=tar", str(args.reference_commit), "HONF_Proj"]
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return _safe_extract_archive(result.stdout, temporary_root / "reference_archive"), "git_archive"


def _run_worker(
    *,
    source_root: Path,
    label: str,
    args: argparse.Namespace,
    raw_output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--source-root",
        str(source_root),
        "--worker-label",
        label,
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(raw_output.with_suffix(".json")),
        "--raw-output",
        str(raw_output),
        "--device",
        str(args.device),
    ]
    if args.dataset is not None:
        command.extend(["--dataset", str(args.dataset)])
    command.extend(["--eval-split", str(args.eval_split), "--train-split", str(args.train_split)])
    env = dict(os.environ)
    source_paths = [
        source_root / "src",
        source_root / "Case_ThermalChannel" / "src",
        source_root / "tools" / "diagnostics",
        source_root / "tools",
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(path) for path in source_paths] + ([existing] if existing else []))
    completed = subprocess.run(
        command,
        cwd=source_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} executor worker failed with exit={completed.returncode}:\n"
            f"stdout={completed.stdout[-4000:]}\n stderr={completed.stderr[-4000:]}"
        )
    return {
        "label": label,
        "source_root": str(source_root),
        "worker_stdout": completed.stdout[-4000:],
        "worker_stderr": completed.stderr[-4000:],
        "raw_output": str(raw_output),
    }


def _load_raw(path: Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _difference(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    left = reference.detach().double()
    right = candidate.detach().double()
    if left.shape != right.shape:
        return {
            "status": "fail",
            "shape_reference": list(left.shape),
            "shape_candidate": list(right.shape),
            "relative_norm": None,
            "absolute_norm": None,
            "max_absolute": None,
            "reference_norm": None,
            "finite": False,
        }
    delta = right - left
    difference = torch.linalg.vector_norm(delta).item() if delta.numel() else 0.0
    reference_norm = torch.linalg.vector_norm(left).item() if left.numel() else 0.0
    return {
        "status": "pending",
        "relative_norm": difference / max(reference_norm, 1.0e-30),
        "absolute_norm": difference,
        "max_absolute": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "reference_norm": reference_norm,
        "finite": bool(torch.isfinite(right).all()),
        "shape": list(left.shape),
    }


def _difference_status(row: Mapping[str, Any], *, relative_tolerance: float, absolute_tolerance: float, near_zero_norm: float = 1.0e-5) -> str:
    if not row.get("finite", False) or row.get("relative_norm") is None:
        return "fail"
    if float(row["reference_norm"]) < near_zero_norm:
        return "pass" if float(row["max_absolute"]) <= absolute_tolerance else "fail"
    return "pass" if float(row["relative_norm"]) <= relative_tolerance else "fail"


def _compare_tensor_maps(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    relative_tolerance_by_key: Mapping[str, float] | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    missing_reference = sorted(set(candidate) - set(reference))
    missing_candidate = sorted(set(reference) - set(candidate))
    for key in sorted(set(reference) & set(candidate)):
        row = _difference(reference[key], candidate[key])
        key_relative_tolerance = float(
            (relative_tolerance_by_key or {}).get(str(key), relative_tolerance)
        )
        row["status"] = _difference_status(
            row,
            relative_tolerance=key_relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        row["relative_tolerance"] = key_relative_tolerance
        rows[str(key)] = row
    return {
        "status": "pass" if not missing_reference and not missing_candidate and (rows or allow_empty) and all(row["status"] == "pass" for row in rows.values()) else "fail",
        "tolerance": {
            "relative": relative_tolerance,
            "relative_by_key": dict(relative_tolerance_by_key or {}),
            "near_zero_absolute": absolute_tolerance,
        },
        "missing_from_reference": missing_reference,
        "missing_from_candidate": missing_candidate,
        "tensors": rows,
    }


def _compare_evidence(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    parity: dict[str, Any] = {"cases": {}, "backward": {}}
    for case_id in DEFAULT_CASE_IDS:
        reference_case = reference["physical"][case_id]
        candidate_case = candidate["physical"][case_id]
        parity["cases"][case_id] = {
            "physical_outputs": _compare_tensor_maps(
                reference_case["physical"],
                candidate_case["physical"],
                relative_tolerance=2.0e-5,
                absolute_tolerance=2.0e-6,
                relative_tolerance_by_key={"pred_field": 2.0e-6},
            ),
            "provisional_outputs": _compare_tensor_maps(
                reference_case["provisional"],
                candidate_case["provisional"],
                relative_tolerance=2.0e-5,
                absolute_tolerance=2.0e-6,
                allow_empty=True,
            ),
            "phase_outputs": _compare_tensor_maps(
                reference["phase_tensors"][case_id],
                candidate["phase_tensors"][case_id],
                relative_tolerance=2.0e-5,
                absolute_tolerance=2.0e-6,
            ),
            "phase_summary_reference": reference["phase_summaries"].get(case_id, {}),
            "phase_summary_candidate": candidate["phase_summaries"].get(case_id, {}),
        }
    reference_backward = reference["backward"]
    candidate_backward = candidate["backward"]
    reference_grads = reference_backward["gradients"]
    candidate_grads = candidate_backward["gradients"]
    gradient_compare = _compare_tensor_maps(
        reference_grads,
        candidate_grads,
        relative_tolerance=1.0e-4,
        absolute_tolerance=1.0e-6,
    )
    loss_reference = {
        key: torch.as_tensor(value, dtype=torch.float64)
        for key, value in reference_backward.get("loss_terms", {}).items()
    }
    loss_candidate = {
        key: torch.as_tensor(value, dtype=torch.float64)
        for key, value in candidate_backward.get("loss_terms", {}).items()
    }
    loss_compare = _compare_tensor_maps(
        loss_reference,
        loss_candidate,
        relative_tolerance=2.0e-6,
        absolute_tolerance=1.0e-6,
    )
    disconnected = sorted(
        set(reference_backward.get("none_gradients", [])) ^ set(candidate_backward.get("none_gradients", []))
    )
    parity["backward"] = {
        "status": "pass" if gradient_compare["status"] == "pass" and loss_compare["status"] == "pass" and not disconnected else "fail",
        "reference_batch": {key: value for key, value in reference_backward.items() if key not in {"gradients", "loss_terms"}},
        "candidate_batch": {key: value for key, value in candidate_backward.items() if key not in {"gradients", "loss_terms"}},
        "loss_terms": loss_compare,
        "gradients": gradient_compare,
        "disconnected_gradient_changes": disconnected,
    }
    case_statuses = [
        value["physical_outputs"]["status"] == "pass"
        and value["provisional_outputs"]["status"] == "pass"
        and value["phase_outputs"]["status"] == "pass"
        for value in parity["cases"].values()
    ]
    parity["status"] = "pass" if all(case_statuses) and parity["backward"]["status"] == "pass" else "fail"
    parity["contract"] = {
        "field_relative_tolerance": 2.0e-6,
        "derived_output_relative_tolerance": 2.0e-5,
        "derived_output_absolute_tolerance": 2.0e-6,
        "gradient_relative_tolerance": 1.0e-4,
        "near_zero_absolute_tolerance": 1.0e-6,
        "same_checkpoint": True,
        "detailed_ledgers_in_timed_reads": False,
    }
    return parity


def _ratio(candidate: Any, reference: Any) -> float | None:
    candidate_value = _finite(candidate)
    reference_value = _finite(reference)
    if candidate_value is None or reference_value is None or reference_value <= 0.0:
        return None
    return candidate_value / reference_value


def _benchmark_summary(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for case_id in DEFAULT_CASE_IDS:
        reference_phases = reference["inference"][case_id]["phases"]
        candidate_phases = candidate["inference"][case_id]["phases"]
        rows[case_id] = {
            phase: {
                "reference": reference_phases[phase],
                "optimized": candidate_phases[phase],
                "optimized_over_reference_median": _ratio(
                    candidate_phases[phase].get("median_seconds"),
                    reference_phases[phase].get("median_seconds"),
                ),
            }
            for phase in reference_phases
        }
    return {
        "cases": rows,
        "protocol": {
            "warmups": DEFAULT_WARMUPS,
            "repetitions": DEFAULT_REPETITIONS,
            "query_count": DEFAULT_QUERY_COUNT,
            "receiver_chunk_size": DEFAULT_RECEIVER_CHUNK_SIZE,
            "maps": False,
            "detailed_ledgers": False,
            "profiler": False,
        },
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    protocol = payload.get("protocol", {})
    checkpoint = payload.get("checkpoint_policy", {})
    parity = payload.get("parity", {})
    benchmark = payload.get("benchmark", {})
    lines = [
        "# Run 1406 Executor Parity and Unprofiled Evidence",
        "",
        "This artifact compares one explicit epoch-50 Run-1406 checkpoint through an isolated pre-optimization reference executor and the optimized working tree. The mathematical model and checkpoint state are held fixed.",
        "",
        "## Checkpoint and protocol",
        "",
        f"- Checkpoint: `{checkpoint.get('path', '')}`; required epoch `{checkpoint.get('required_epoch', '')}`; strict load `{checkpoint.get('strict_load', '')}`.",
        f"- Cases `{', '.join(protocol.get('case_ids', []))}`; Q={protocol.get('query_count')}; receiver chunk={protocol.get('receiver_chunk_size')}; {protocol.get('inference_warmups')} warmups/{protocol.get('inference_repetitions')} synchronized repetitions.",
        "- Timed reads set `return_routing_maps=False`; detailed execution ledgers and profiler are disabled in timed calls.",
        f"- Backward evidence: real B={protocol.get('backward', {}).get('batch_size')} / Q={protocol.get('backward', {}).get('query_count')} / M={protocol.get('backward', {}).get('module_count')} batch, predicted-port mode, no optimizer update.",
        "",
        "## Strict load",
        "",
    ]
    for label, worker in payload.get("workers", {}).items():
        load = worker.get("load", {}) if isinstance(worker, Mapping) else {}
        lines.append(f"- `{label}`: status={load.get('status')}; strict={load.get('strict')}; epoch={load.get('checkpoint_epoch')}; state keys={load.get('model_state_key_count')}.")
    lines.extend(["", "## Old/new parity", ""])
    lines.append(f"Overall parity status: **{parity.get('status', 'unavailable')}**.")
    lines.append("")
    for case_id, row in parity.get("cases", {}).items():
        lines.append(
            f"- Case `{case_id}`: physical={row.get('physical_outputs', {}).get('status')}; P1/provisional={row.get('provisional_outputs', {}).get('status')}; phase diagnostics={row.get('phase_outputs', {}).get('status')}."
        )
    lines.append(
        f"- Predicted-port backward batch: status={parity.get('backward', {}).get('status')}; gradients={parity.get('backward', {}).get('gradients', {}).get('status')}; loss terms={parity.get('backward', {}).get('loss_terms', {}).get('status')}."
    )
    lines.extend(["", "## Unprofiled benchmark", "", "| Case | Phase | Reference median (ms) | Optimized median (ms) | Optimized/reference |", "|---|---|---:|---:|---:|"])
    for case_id, case in benchmark.get("cases", {}).items():
        for phase, row in case.items():
            reference_ms = _finite(row.get("reference", {}).get("median_seconds"))
            optimized_ms = _finite(row.get("optimized", {}).get("median_seconds"))
            ratio = row.get("optimized_over_reference_median")
            lines.append(
                f"| {case_id} | {phase} | {'NA' if reference_ms is None else f'{1000 * reference_ms:.3f}'} | {'NA' if optimized_ms is None else f'{1000 * optimized_ms:.3f}'} | {'NA' if ratio is None else f'{ratio:.4f}'} |"
            )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This is same-checkpoint executor evidence only; it does not establish a retrained epoch-50 result or physical/CFD truth.",
            "- Phase counters and ledgers are execution bookkeeping, not physical causality. The parity gate is applied to physical outputs, provisional outputs, and first derivatives.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    _validate_protocol(args)
    checkpoint = _parse_checkpoint(args.checkpoint)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run1406_executor_evidence_", dir=str(output.parent)) as temporary:
        temporary_root = Path(temporary)
        reference_source, reference_policy = _reference_source(args, temporary_root)
        optimized_source = PROJECT_ROOT
        reference_raw = temporary_root / "reference.pt"
        optimized_raw = temporary_root / "optimized.pt"
        worker_specs = {
            "reference": (reference_source, reference_raw),
            "optimized": (optimized_source, optimized_raw),
        }
        worker_logs: dict[str, Any] = {}
        for label, (source, raw_path) in worker_specs.items():
            worker_logs[label] = _run_worker(
                source_root=source,
                label=label,
                args=args,
                raw_output=raw_path,
            )
        reference = _load_raw(reference_raw)
        optimized = _load_raw(optimized_raw)
        parity = _compare_evidence(reference, optimized)
        benchmark = _benchmark_summary(reference, optimized)
        workers = {
            label: {
                **worker_logs[label],
                "load": raw.get("load", {}),
                "backward": {
                    key: value
                    for key, value in raw.get("backward", {}).items()
                    if key not in {"gradients"}
                },
            }
            for label, raw in (("reference", reference), ("optimized", optimized))
        }
    payload = {
        "schema_version": 1,
        "task": "run1406_executor_parity_and_unprofiled_evidence",
        "status": "complete" if parity.get("status") == "pass" else "parity_failed",
        "checkpoint_policy": {
            "path": str(checkpoint),
            "selection_policy": "explicit_cli_checkpoint",
            "required_epoch": DEFAULT_EPOCH,
            "strict_load": True,
            "trusted_loader": "channelthermal.evaluation.loading.load_model",
        },
        "executor_policy": {
            "reference_commit": str(args.reference_commit),
            "reference_source_policy": reference_policy,
            "optimized_source": str(PROJECT_ROOT),
            "comparison_isolation": "separate subprocess per source tree",
        },
        "protocol": _protocol(args),
        "workers": workers,
        "parity": parity,
        "benchmark": benchmark,
        "limitations": [
            "No profiler trace or managed training job was launched.",
            "Detailed execution ledgers/maps are disabled in timed reads.",
            "The backward evidence performs loss.backward() but no optimizer update.",
        ],
    }
    _write_json(output, payload)
    report = Path(args.report).expanduser().resolve() if args.report else output.with_suffix(".md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--receiver-chunk-size", type=int, default=DEFAULT_RECEIVER_CHUNK_SIZE)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--train-query-count", type=int, default=DEFAULT_TRAIN_QUERY_COUNT)
    parser.add_argument("--backward-module-count", type=int, default=DEFAULT_BACKWARD_MODULE_COUNT)
    parser.add_argument("--reference-commit", default="HEAD")
    parser.add_argument("--reference-source", type=Path, default=None)
    parser.add_argument("--plan-only", action="store_true")
    # Worker-only flags are intentionally hidden from the endpoint command.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-label", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--raw-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if args.source_root is None or args.worker_label is None or args.raw_output is None:
            raise ValueError("worker mode requires --source-root, --worker-label, and --raw-output")
        return _worker(args)
    plan = build_plan(args)
    if args.plan_only:
        _write_json(Path(args.output), plan)
        print(json.dumps(_jsonable(plan), indent=2, sort_keys=True))
        return 0
    payload = run_measurement(args)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0 if payload.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CASE_IDS",
    "DEFAULT_EPOCH",
    "DEFAULT_QUERY_COUNT",
    "DEFAULT_RECEIVER_CHUNK_SIZE",
    "build_parser",
    "build_plan",
    "main",
    "run_measurement",
]
