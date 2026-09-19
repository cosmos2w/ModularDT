"""Controlled Run-1405 epoch-50 evidence entry point.

This is a bounded comparison tool, not a trainer and not a managed-run
allocator.  It reuses the maintained Stage-3 checkpoint/dataset loaders and
the canonical ``run_epoch`` loss path only when the caller explicitly omits
``--plan-only`` on a selected device.  The default protocol is deliberately
the Run-1405 plan:

* explicit checkpoints for 1405, 1804, and 1401;
* cases 0273 and 0653, ``Q=8192``, receiver chunk 2048;
* two untimed warmups and five synchronized repetitions for full physical
  forward and prepared P2 decode;
* fresh-optimizer one-batch steps at ``B=48``, ``Q=1024`` for exact M1/M12
  buckets, with one warmup and three measured updates for 1405, 1804, and
  the explicitly supplied 1401 context checkpoint;
* allocated/reserved CUDA memory and exact fixed-group semantic ``P_M/P_E``,
  ``R_M/R_E``, ``sQ/sM/sE`` when the Run-1405 debug API is available.
* an explicit Run-1405 ``metrics.csv`` health gate covering epochs 1--50,
  finite losses/diagnostics, positive major-group updates, and deterministic
  validation-window checks.

Timed calls keep routing-map materialization disabled.  One separate untimed
debug forward requests the fixed-group arrays and writes one NPZ per case for
the CPU-only interaction board.  The expected API is documented in
``fixed_group_evidence.py`` and is recorded in the output manifest.  Missing
debug arrays are reported as unavailable rather than reconstructed from dense
pair counts.

The 1401 checkpoint is intentionally an explicit CLI input even though this
workspace has no matched epoch-50 1401 checkpoint.  Its exact supplied epoch
is recorded as context only; the fidelity gate is strictly Run-1405 versus
Run-1804 at epoch 50.

Plan-only validation (safe on a CPU-only host)::

    python tools/diagnostics/run_run1405_epoch50_comparison.py \
        --checkpoint-1405 /path/epoch_0050_model.pt \
        --checkpoint-1804 /path/dense_epoch_0050_model.pt \
        --checkpoint-1401 /path/explicit_1401_context.pt \
        --metrics-1405 /path/run1405/metrics.csv \
        --output diagnostics/generated/run1405_epoch50/comparison.json \
        --plan-only

Physical execution is intentionally a separate user-authorized step and is
never launched by this module's tests.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from fixed_group_evidence import (
    FixedGroupEvidenceError,
    canonicalize_fixed_group_arrays,
    collect_debug_payload,
    jsonable,
    save_evidence_npz,
    semantic_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_IDS = ("0273", "0653")
DEFAULT_EPOCH = 50
DEFAULT_QUERY_COUNT = 8192
DEFAULT_RECEIVER_CHUNK_SIZE = 2048
DEFAULT_INFERENCE_WARMUPS = 2
DEFAULT_INFERENCE_REPETITIONS = 5
DEFAULT_TRAIN_BATCH_SIZE = 48
DEFAULT_TRAIN_QUERY_COUNT = 1024
DEFAULT_TRAIN_WARMUPS = 1
DEFAULT_TRAIN_REPETITIONS = 3
COMPARISON_LABELS = ("1405", "1804", "1401")
TRAINING_LABELS = COMPARISON_LABELS
TRAIN_HEALTH_CORE_COLUMNS = (
    "loss_total",
    "loss_field",
    "field_mse",
    "val_loss_total",
    "val_loss_field",
    "val_field_mse",
)
TRAIN_HEALTH_MAJOR_GROUPS = ("encoder", "backend", "head", "local_coupling")
TRAIN_HEALTH_DIAGNOSTIC_COLUMNS = (
    "preclip_gradient_norm",
    "parameter_update_norm",
    *(
        f"{prefix}_{group}"
        for prefix in ("preclip_gradient_norm", "parameter_update_norm")
        for group in TRAIN_HEALTH_MAJOR_GROUPS
    ),
)
TRAIN_HEALTH_FIRST_WINDOW = (1, 10)
TRAIN_HEALTH_LAST_WINDOW = (41, DEFAULT_EPOCH)
TRAIN_HEALTH_CATASTROPHIC_FACTOR = 2.0


def _parse_path(raw: str, *, name: str) -> Path:
    value = Path(str(raw)).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"{name} checkpoint does not exist: {value}")
    return value


def _finite(value: Any) -> float | None:
    try:
        scalar = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return scalar if math.isfinite(scalar) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sync(device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _measure_phase(
    function: Callable[[], Any],
    device: Any,
    *,
    warmups: int,
    repetitions: int,
    inference: bool = True,
) -> dict[str, Any]:
    """Measure one synchronized phase with per-repetition memory fields."""

    import torch

    if int(warmups) < 0 or int(repetitions) <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    execution_context = torch.inference_mode if inference else nullcontext
    with execution_context():
        for _ in range(int(warmups)):
            result = function()
            del result
        _sync(device)
    samples: list[dict[str, Any]] = []
    for repetition in range(int(repetitions)):
        if getattr(device, "type", None) == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
        else:
            baseline_allocated = None
            baseline_reserved = None
        _sync(device)
        started = time.perf_counter()
        error: str | None = None
        result: Any = None
        try:
            with execution_context():
                result = function()
        except Exception as exc:  # noqa: BLE001 - preserve per-repetition evidence
            error = f"{type(exc).__name__}: {exc}"
        _sync(device)
        elapsed = time.perf_counter() - started
        if getattr(device, "type", None) == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = None
            peak_reserved = None
        samples.append(
            {
                "repetition": int(repetition + 1),
                "elapsed_seconds": float(elapsed),
                "status": "error" if error is not None else "complete",
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
        if error is not None:
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
            (int(row["peak_allocated_bytes"]) for row in samples if row["peak_allocated_bytes"] is not None),
            default=None,
        ),
        "peak_reserved_bytes": max(
            (int(row["peak_reserved_bytes"]) for row in samples if row["peak_reserved_bytes"] is not None),
            default=None,
        ),
    }


def _checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve the three explicit checkpoint CLI inputs; never infer 1401."""

    return {
        "1405": _parse_path(args.checkpoint_1405, name="Run 1405"),
        "1804": _parse_path(args.checkpoint_1804, name="Run 1804"),
        "1401": _parse_path(args.checkpoint_1401, name="Run 1401 context"),
    }


def _checkpoint_policy(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        label: {
            "label": label,
            "path": str(path),
            "selection_policy": "explicit_cli_checkpoint",
            "required_epoch": DEFAULT_EPOCH if label in {"1405", "1804"} else None,
            "epoch_role": (
                "matched_epoch_50_candidate"
                if label == "1405"
                else "matched_epoch_50_fidelity_reference"
                if label == "1804"
                else "explicit_context_only; no matched epoch_50 assumption"
            ),
        }
        for label, path in paths.items()
    }


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_epoch": DEFAULT_EPOCH,
        "case_ids": [str(value) for value in (args.case_id or DEFAULT_CASE_IDS)],
        "query_count": int(args.query_count),
        "receiver_chunk_size": int(args.receiver_chunk_size),
        "inference_warmups": int(args.warmups),
        "inference_repetitions": int(args.repetitions),
        "routing_maps_in_timed_phases": False,
        "debug_map_pass": "one untimed full-Q forward after timing",
        "training_labels": list(TRAINING_LABELS),
        "training_batch_size": int(args.train_batch_size),
        "training_query_count": int(args.train_query_count),
        "training_buckets": ["M1", "M12"],
        "training_warmups": int(args.train_warmups),
        "training_repetitions": int(args.train_repetitions),
        "training_optimizer": "fresh optimizer per model/bucket; matched Run-1804 hyperparameters",
        "training_port_condition": "predicted",
        "training_health_metrics_csv": (
            None if args.metrics_1405 is None else str(Path(args.metrics_1405).expanduser().resolve())
        ),
        "training_health_target_epoch": DEFAULT_EPOCH,
        "training_health_required_epoch_span": [1, DEFAULT_EPOCH],
        "training_health_major_groups": list(TRAIN_HEALTH_MAJOR_GROUPS),
        "training_health_catastrophic_rule": (
            "at most one last-10 val_field_mse value above 2x the last-10 median"
        ),
        "quickcheck_run": False,
        "managed_training": False,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Validate only CLI/file contracts; this function never imports CUDA/model code."""

    paths = _checkpoint_paths(args)
    if args.metrics_1405 is None:
        raise ValueError("--metrics-1405 is required for the epoch-50 training-health gate")
    cases = tuple(str(value) for value in (args.case_id or DEFAULT_CASE_IDS))
    if not {"0273", "0653"}.issubset(cases):
        raise ValueError("the Run-1405 protocol requires cases 0273 and 0653")
    for name, value in (
        ("query_count", args.query_count),
        ("receiver_chunk_size", args.receiver_chunk_size),
        ("train_batch_size", args.train_batch_size),
        ("train_query_count", args.train_query_count),
        ("train_repetitions", args.train_repetitions),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.warmups) < 0 or int(args.train_warmups) < 0:
        raise ValueError("warmup counts must be non-negative")
    if int(args.repetitions) <= 0:
        raise ValueError("inference repetitions must be positive")
    return {
        "schema_version": 1,
        "task": "run1405_epoch50_comparison",
        "status": "plan_only",
        "checkpoint_policy": _checkpoint_policy(paths),
        "protocol": _protocol(args),
        "fidelity_gate": {
            "definition": "Run-1405 epoch-50 validation field MSE <= 2x matched Run-1804 epoch-50 validation field MSE",
            "reference_label": "1804",
            "context_label": "1401",
            "status": "pending_measurement",
        },
        "training_health": {
            "metrics_csv": str(Path(args.metrics_1405).expanduser().resolve()),
            "target_epoch": DEFAULT_EPOCH,
            "required_epoch_span": [1, DEFAULT_EPOCH],
            "status": "pending_measurement",
            "required_checks": [
                "complete unique rows through epoch 50",
                "finite core train/validation losses and field MSE",
                "finite aggregate and major-group gradient/update diagnostics",
                "positive finite encoder/backend/head/local_coupling gradient/update evidence",
                "last-10 val_field_mse median < first-10 median",
                "at most one last-10 val_field_mse value above 2x last-10 median",
            ],
        },
        "debug_api_contract": _debug_api_contract(),
        "limitations": [
            "Plan-only mode validates paths and protocol only; it does not load checkpoints, datasets, or CUDA.",
            "Run 1401 is explicit context only because this workspace has no matched epoch-50 checkpoint.",
            "No quickcheck run, training launch, or managed run allocation is performed by this tool.",
        ],
    }


def _debug_api_contract() -> dict[str, Any]:
    return {
        "container": "model output interaction_aux (preferred); prepared_state.prepared.interaction_aux is accepted",
        "arrays": {
            "fixed_group_module_incidence": "[B,M,K] nonnegative A_m",
            "fixed_group_environment_incidence": "[B,E,K] nonnegative A_e",
            "fixed_group_module_centres": "[B,K,2] r_m",
            "fixed_group_environment_centres": "[B,K,2] r_e",
            "fixed_group_query_routing": "[B,Q,K] nonnegative alpha_qk",
        },
        "geometry": "prepared_state.prepared.encoded.{module_centers,module_present,env_coords,env_weights}",
        "group_count": 6,
        "semantics": "positive support is the actual q->group->source interaction relation; route weights are not physical causality",
    }


def _load_runtime_modules() -> tuple[Any, Any, Any, Any, Any]:
    """Import maintained loaders only for the explicit physical run."""

    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import benchmark_routing_optimization as train_bench
    import run_dynamic_sparse_routing_study as dynamic
    import run_stage3_interface_study as stage3
    import torch
    from channelthermal.evaluation.loading import make_batch
    from channelthermal.evaluation.prepared import select_sample

    return torch, train_bench, dynamic, stage3, (make_batch, select_sample)


def _epoch_of(checkpoint: Mapping[str, Any]) -> int:
    value = checkpoint.get("epoch", checkpoint.get("current_epoch", -1))
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _require_matched_epoch(checkpoint: Mapping[str, Any], label: str) -> None:
    epoch = _epoch_of(checkpoint)
    if epoch != DEFAULT_EPOCH:
        raise ValueError(f"Run {label} checkpoint epoch={epoch}; controlled comparison requires exact epoch 50")


def _phase_forward_kwargs(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "interface_condition": batch.get("interface_condition"),
        "local_module_params": batch.get("local_module_params"),
        "teacher_port_tokens": batch.get("teacher_port_tokens"),
        "local_query_points": batch.get("module_internal_query_points"),
        "local_port_condition_mode": "predicted",
        "mixed_teacher_ratio": 0.0,
    }


def _inference_case(
    *,
    label: str,
    model: Any,
    checkpoint: Mapping[str, Any],
    dataset: Any,
    case_id: str,
    args: argparse.Namespace,
    map_dir: Path,
    torch: Any,
    dynamic: Any,
    select_sample: Callable[..., Any],
    make_batch: Callable[..., Any],
) -> tuple[dict[str, Any], Path | None]:
    sample = select_sample(dataset, str(case_id), 0)
    query_np = dynamic._query_points(sample, int(args.query_count))
    if len(query_np) != int(args.query_count):
        raise ValueError(
            f"case {case_id} produced Q={len(query_np)} grid probes; expected exactly {args.query_count}"
        )
    batch = make_batch(dict(sample), query_np, torch.device(args.device))
    query = batch["query_xy"]
    forward_kwargs = _phase_forward_kwargs(batch)
    receiver_chunk = int(args.receiver_chunk_size)

    def full_forward() -> Any:
        with dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
            return model(batch["structure"], query, return_routing_maps=False, **forward_kwargs)

    def prepare_one() -> Any:
        with dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
            return model(
                batch["structure"],
                query[:, :1],
                return_prepared_state=True,
                return_routing_maps=False,
                **forward_kwargs,
            )

    prepared_state_holder: dict[str, Any] = {}

    def prepared_decode() -> Any:
        prepared = prepared_state_holder["prepared"]
        with dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
            return model.decode_prepared(
                prepared,
                query,
                return_routing_maps=False,
                receiver_chunk_size=receiver_chunk,
            )

    device = torch.device(args.device)
    phases = {
        "full_physical_forward": _measure_phase(
            full_forward,
            device,
            warmups=int(args.warmups),
            repetitions=int(args.repetitions),
        ),
        "physical_preparation_plus_one_query": _measure_phase(
            prepare_one,
            device,
            warmups=int(args.warmups),
            repetitions=int(args.repetitions),
        ),
    }
    with torch.inference_mode():
        prepared_output = prepare_one()
    prepared_state_holder["prepared"] = prepared_output["prepared_state"]
    phases["prepared_p2_decode"] = _measure_phase(
        prepared_decode,
        device,
        warmups=int(args.warmups),
        repetitions=int(args.repetitions),
    )
    prepared = prepared_state_holder.pop("prepared")
    del prepared, prepared_output
    gc.collect()

    map_path: Path | None = None
    semantic: dict[str, Any] = {"status": "unavailable"}
    debug_error: str | None = None
    try:
        with torch.inference_mode(), dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
            debug_output = model(
                batch["structure"],
                query,
                return_prepared_state=True,
                return_routing_maps=True,
                **forward_kwargs,
            )
        payload = collect_debug_payload(debug_output, query_xy=query)
        arrays = canonicalize_fixed_group_arrays(payload)
        decode_seconds = phases["prepared_p2_decode"].get("median_seconds")
        debug_metrics = semantic_metrics(
            arrays,
            prepared_decode_median_ms=None if decode_seconds is None else float(decode_seconds) * 1000.0,
        )
        map_dir.mkdir(parents=True, exist_ok=True)
        map_path = save_evidence_npz(
            map_dir / f"{label}__{case_id}.npz",
            arrays,
            debug_metrics,
            metadata={
                "label": label,
                "case_id": str(case_id),
                "checkpoint_epoch": _epoch_of(checkpoint),
                "prepared_decode_median_ms": None if decode_seconds is None else float(decode_seconds) * 1000.0,
                "debug_api": _debug_api_contract(),
            },
        )
        semantic = {"status": "ok", **debug_metrics.values}
        del debug_output
    except (FixedGroupEvidenceError, KeyError, RuntimeError, ValueError, TypeError) as exc:
        debug_error = f"{type(exc).__name__}: {exc}"
        semantic = {"status": "unavailable", "reason": debug_error}
    return (
        {
            "label": label,
            "case_id": str(case_id),
            "query_count": int(query.shape[1]),
            "receiver_chunk_size": receiver_chunk,
            "checkpoint_epoch": _epoch_of(checkpoint),
            "phases": phases,
            "semantic": semantic,
            "debug_api_error": debug_error,
            "route_semantics": "learned interaction routes, not physical causality",
        },
        map_path,
    )


def _common_case_ids(datasets: Sequence[Any]) -> list[str]:
    common = {str(value) for value in datasets[0].selected_case_ids}
    for dataset in datasets[1:]:
        common.intersection_update(str(value) for value in dataset.selected_case_ids)
    return sorted(common)


def _case_count(dataset: Any, case_id: str) -> int:
    mapping = {str(value): index for index, value in enumerate(dataset.selected_case_ids)}
    index = mapping.get(str(case_id))
    if index is None:
        raise KeyError(f"case {case_id!r} absent from {dataset.split!r} split")
    return int(dataset.selected_module_counts[index])


def _make_exact_training_buckets(
    datasets: Sequence[Any],
    *,
    batch_size: int,
    requested_counts: Sequence[int] = (1, 12),
) -> list[Any]:
    """Build deterministic B=48 buckets for exactly M1 and M12."""

    import benchmark_routing_optimization as train_bench

    common = _common_case_ids(datasets)
    if not common:
        raise ValueError("training checkpoints have no common case IDs")
    counts = {case_id: _case_count(datasets[0], case_id) for case_id in common}
    for dataset in datasets[1:]:
        for case_id in common:
            other = _case_count(dataset, case_id)
            if other != counts[case_id]:
                raise ValueError(f"matched checkpoints disagree on M for case={case_id}")
    buckets = []
    for count in requested_counts:
        candidates = sorted(case_id for case_id, value in counts.items() if value == int(count))
        if not candidates:
            raise ValueError(f"no common real train case with exact M={count}")
        case_ids = tuple(candidates[index % len(candidates)] for index in range(int(batch_size)))
        buckets.append(train_bench.Bucket(f"M{count}", int(count), case_ids))
    return buckets


def _matched_train_schedule(checkpoint: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, float, float, float, float]:
    """Use Run-1804's loss/optimizer hyperparameters and force predicted ports."""

    import benchmark_routing_optimization as train_bench
    from channelthermal.training.epoch import effective_local_loss_weights, predicted_consistency_weight_for_epoch

    train_config, loss_config, _mode, _ratio, _internal, _interface, _predicted = train_bench._training_schedule(checkpoint)
    internal, interface = effective_local_loss_weights(loss_config, "predicted", 0.0)
    predicted = predicted_consistency_weight_for_epoch(DEFAULT_EPOCH, loss_config)
    return train_config, loss_config, "predicted", 0.0, float(internal), float(interface), float(predicted)


def _measure_training_bucket(
    *,
    label: str,
    model: Any,
    checkpoint: Mapping[str, Any],
    dataset: Any,
    bucket: Any,
    device: Any,
    args: argparse.Namespace,
    shared_training_config: Mapping[str, Any],
    shared_schedule: tuple[dict[str, Any], dict[str, Any], str, float, float, float, float],
    train_bench: Any,
) -> dict[str, Any]:
    loader = train_bench._build_loader(dataset, checkpoint, bucket, batch_size=int(args.train_batch_size))
    shape = train_bench._validate_batch_shape(loader, device, int(args.train_query_count))
    optimizer, inventory = train_bench.build_forward_optimizer(model, dict(shared_training_config))
    model.train()

    def step() -> Any:
        return train_bench._run_step(
            model,
            loader,
            device,
            checkpoint,
            optimizer=optimizer,
            training_schedule=shared_schedule,
        )

    measurement = _measure_phase(
        step,
        device,
        warmups=int(args.train_warmups),
        repetitions=int(args.train_repetitions),
        inference=False,
    )
    result = {
        "label": label,
        "bucket": {"label": bucket.label, "module_count": bucket.module_count, "case_ids": list(bucket.case_ids)},
        "shape": shape,
        "status": measurement["status"],
        "measurement": measurement,
        "optimizer_policy": "fresh_optimizer_matched_to_run1804_hyperparameters",
        "optimizer_inventory": inventory,
        "port_condition_mode": "predicted",
    }
    gc.collect()
    return result


def _read_epoch_metric(path: Path | None, *, epoch: int = DEFAULT_EPOCH) -> dict[str, Any]:
    """Read one schema-light validation field-MSE row from an explicit CSV."""

    if path is None:
        return {"status": "unavailable", "reason": "no --metrics path supplied"}
    source = path.expanduser().resolve()
    if not source.is_file():
        return {"status": "unavailable", "reason": f"metrics file not found: {source}"}
    rows = list(csv.DictReader(source.open("r", newline="", encoding="utf-8")))
    candidates: list[dict[str, Any]] = []
    for row in rows:
        raw_epoch = row.get("epoch", row.get("current_epoch"))
        try:
            if int(float(raw_epoch)) != int(epoch):
                continue
        except (TypeError, ValueError):
            continue
        for key in ("val_field_mse", "validation_field_mse", "val_loss_field", "field_mse"):
            value = _finite(row.get(key))
            if value is not None:
                candidates.append({"epoch": int(epoch), "metric": value, "metric_name": key, "path": str(source)})
                break
    if len(candidates) != 1:
        return {
            "status": "unavailable",
            "reason": f"expected exactly one epoch-{epoch} field-MSE row in {source}; found {len(candidates)}",
            "path": str(source),
        }
    return {"status": "ok", **candidates[0]}


def training_health_gate(path: Path | str | None, *, epoch: int = DEFAULT_EPOCH) -> dict[str, Any]:
    """Validate the explicit Run-1405 training history through ``epoch``.

    The training workflow records gradient/update diagnostics only on selected
    epochs (including epoch 50), so non-diagnostic rows are allowed to carry
    NaN in those columns.  Any row carrying a partial diagnostic snapshot is
    rejected, and the epoch-50 snapshot must be finite with positive evidence
    for every major group.  Core losses are required on every row from epoch 1
    through the target epoch.
    """

    source = None if path is None else Path(path).expanduser().resolve()
    base: dict[str, Any] = {
        "path": None if source is None else str(source),
        "target_epoch": int(epoch),
        "required_epoch_span": [1, int(epoch)],
        "catastrophic_rule": (
            "at most one epoch in the last 10 with val_field_mse > "
            f"{TRAIN_HEALTH_CATASTROPHIC_FACTOR:g}x the last-10 median"
        ),
    }
    if source is None:
        return {
            **base,
            "status": "unavailable",
            "reason": "no --metrics-1405 path supplied",
            "checks": {},
        }
    if not source.is_file():
        return {
            **base,
            "status": "unavailable",
            "reason": f"metrics file not found: {source}",
            "checks": {},
        }

    try:
        with source.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            raw_rows = list(reader)
    except OSError as exc:
        return {
            **base,
            "status": "unavailable",
            "reason": f"cannot read metrics file {source}: {exc}",
            "checks": {},
        }

    required_columns = (*TRAIN_HEALTH_CORE_COLUMNS, *TRAIN_HEALTH_DIAGNOSTIC_COLUMNS)
    missing_columns = sorted(set(required_columns) - set(fieldnames))
    schema_check = {
        "status": "pass" if not missing_columns else "fail",
        "required_columns": list(required_columns),
        "missing_columns": missing_columns,
    }

    rows_by_epoch: dict[int, dict[str, str]] = {}
    duplicate_epochs: list[int] = []
    invalid_epoch_rows: list[int] = []
    for row_number, row in enumerate(raw_rows, start=2):
        raw_value = _finite(row.get("epoch"))
        if raw_value is None or not float(raw_value).is_integer():
            invalid_epoch_rows.append(int(row_number))
            continue
        row_epoch = int(raw_value)
        if row_epoch in rows_by_epoch:
            duplicate_epochs.append(row_epoch)
            continue
        rows_by_epoch[row_epoch] = row

    target_epoch = int(epoch)
    required_epochs = list(range(1, target_epoch + 1))
    missing_epochs = [value for value in required_epochs if value not in rows_by_epoch]
    rows_check = {
        "status": "pass" if not missing_epochs and not duplicate_epochs and not invalid_epoch_rows else "fail",
        "rows_loaded": len(raw_rows),
        "rows_checked": len(required_epochs) - len(missing_epochs),
        "max_epoch": max(rows_by_epoch, default=None),
        "missing_epochs": missing_epochs,
        "duplicate_epochs": sorted(set(duplicate_epochs)),
        "invalid_epoch_rows": invalid_epoch_rows,
    }

    nonfinite_core: dict[str, list[int]] = {column: [] for column in TRAIN_HEALTH_CORE_COLUMNS}
    for row_epoch in required_epochs:
        row = rows_by_epoch.get(row_epoch)
        if row is None:
            continue
        for column in TRAIN_HEALTH_CORE_COLUMNS:
            if _finite(row.get(column)) is None:
                nonfinite_core[column].append(row_epoch)
    nonfinite_core = {column: values for column, values in nonfinite_core.items() if values}
    finite_core_check = {
        "status": "pass" if not nonfinite_core else "fail",
        "nonfinite_by_column": nonfinite_core,
    }

    diagnostic_evidence_epochs: list[int] = []
    partial_diagnostic_epochs: list[int] = []
    for row_epoch in required_epochs:
        row = rows_by_epoch.get(row_epoch)
        if row is None:
            continue
        values = [_finite(row.get(column)) for column in TRAIN_HEALTH_DIAGNOSTIC_COLUMNS]
        present = sum(value is not None for value in values)
        if present:
            diagnostic_evidence_epochs.append(row_epoch)
            if present != len(values):
                partial_diagnostic_epochs.append(row_epoch)
    epoch_row = rows_by_epoch.get(target_epoch, {})
    epoch_diagnostic_values = {
        column: _finite(epoch_row.get(column)) for column in TRAIN_HEALTH_DIAGNOSTIC_COLUMNS
    }
    finite_diagnostics_check = {
        "status": (
            "pass"
            if not partial_diagnostic_epochs
            and all(value is not None for value in epoch_diagnostic_values.values())
            else "fail"
        ),
        "diagnostic_evidence_epochs": diagnostic_evidence_epochs,
        "partial_diagnostic_epochs": partial_diagnostic_epochs,
        "epoch_50_values": epoch_diagnostic_values,
    }

    positive_evidence: dict[str, dict[str, float | None]] = {}
    nonpositive_groups: list[str] = []
    for group in TRAIN_HEALTH_MAJOR_GROUPS:
        gradient = epoch_diagnostic_values.get(f"preclip_gradient_norm_{group}")
        update = epoch_diagnostic_values.get(f"parameter_update_norm_{group}")
        positive_evidence[group] = {"gradient": gradient, "update": update}
        if gradient is None or update is None or gradient <= 0.0 or update <= 0.0:
            nonpositive_groups.append(group)
    positive_groups_check = {
        "status": "pass" if not nonpositive_groups else "fail",
        "required_groups": list(TRAIN_HEALTH_MAJOR_GROUPS),
        "epoch_50": positive_evidence,
        "nonpositive_or_nonfinite_groups": nonpositive_groups,
    }

    first_start, first_end = TRAIN_HEALTH_FIRST_WINDOW
    last_start = target_epoch - (TRAIN_HEALTH_LAST_WINDOW[1] - TRAIN_HEALTH_LAST_WINDOW[0])
    first_epochs = list(range(first_start, first_end + 1))
    last_epochs = list(range(last_start, target_epoch + 1))
    first_values = [
        _finite(rows_by_epoch.get(row_epoch, {}).get("val_field_mse")) for row_epoch in first_epochs
    ]
    last_values = [
        _finite(rows_by_epoch.get(row_epoch, {}).get("val_field_mse")) for row_epoch in last_epochs
    ]
    first_finite = [value for value in first_values if value is not None]
    last_finite = [value for value in last_values if value is not None]
    first_median = float(np.median(first_finite)) if len(first_finite) == len(first_values) == 10 else None
    last_median = float(np.median(last_finite)) if len(last_finite) == len(last_values) == 10 else None
    improvement_check = {
        "status": "pass" if first_median is not None and last_median is not None and last_median < first_median else "fail",
        "first_window_epochs": first_epochs,
        "last_window_epochs": last_epochs,
        "first10_median_val_field_mse": first_median,
        "last10_median_val_field_mse": last_median,
        "criterion": "last-10 val_field_mse median < first-10 val_field_mse median",
    }
    catastrophic_epochs = (
        []
        if last_median is None
        else [
            row_epoch
            for row_epoch, value in zip(last_epochs, last_values)
            if value is not None and value > TRAIN_HEALTH_CATASTROPHIC_FACTOR * last_median
        ]
    )
    catastrophic_check = {
        "status": "pass" if last_median is not None and len(catastrophic_epochs) <= 1 else "fail",
        "threshold": None if last_median is None else float(TRAIN_HEALTH_CATASTROPHIC_FACTOR * last_median),
        "exceeding_epochs": catastrophic_epochs,
        "exceeding_count": len(catastrophic_epochs),
        "allowed_count": 1,
        "criterion": base["catastrophic_rule"],
    }

    checks = {
        "schema": schema_check,
        "rows_through_epoch_50": rows_check,
        "finite_core_train_validation_metrics": finite_core_check,
        "finite_gradient_update_diagnostics": finite_diagnostics_check,
        "positive_major_group_gradient_update_evidence": positive_groups_check,
        "last10_median_below_first10_median": improvement_check,
        "catastrophic_last10_instability": catastrophic_check,
    }
    statuses = [str(check["status"]) for check in checks.values()]
    return {
        **base,
        "status": "pass" if statuses and all(status == "pass" for status in statuses) else "fail",
        "checks": checks,
    }


def fidelity_gate(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the fixed 2x epoch-50 fidelity criterion without fallback substitution."""

    candidate_value = _finite(candidate.get("metric")) if candidate.get("status") == "ok" else None
    reference_value = _finite(reference.get("metric")) if reference.get("status") == "ok" else None
    if candidate_value is None or reference_value is None or reference_value <= 0.0:
        return {
            "status": "unavailable",
            "candidate_metric": candidate_value,
            "reference_metric": reference_value,
            "criterion": "candidate <= 2 * reference",
        }
    passed = bool(candidate_value <= 2.0 * reference_value)
    return {
        "status": "pass" if passed else "fail",
        "candidate_metric": candidate_value,
        "reference_metric": reference_value,
        "threshold": float(2.0 * reference_value),
        "ratio": float(candidate_value / reference_value),
        "criterion": "candidate <= 2 * reference",
    }


def _status_from_values(values: Sequence[float | None], predicate: Callable[[float], bool]) -> dict[str, Any]:
    if not values or any(value is None or not math.isfinite(float(value)) for value in values):
        return {"status": "unavailable", "values": list(values)}
    finite = [float(value) for value in values]
    return {"status": "pass" if all(predicate(value) for value in finite) else "fail", "values": finite}


def continuation_criteria(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the fixed epoch-50 continuation gates without threshold drift."""

    inference = payload.get("inference", [])
    training = payload.get("training", [])

    def rows_for(label: str) -> list[Mapping[str, Any]]:
        return [
            row
            for row in inference
            if isinstance(row, Mapping) and str(row.get("label")) == label
        ] if isinstance(inference, Sequence) else []

    candidate_rows = {str(row.get("case_id")): row for row in rows_for("1405")}
    reference_rows = {str(row.get("case_id")): row for row in rows_for("1804")}
    common_cases = sorted(set(candidate_rows) & set(reference_rows))
    candidate_full = [
        _finite(candidate_rows[case].get("phases", {}).get("full_physical_forward", {}).get("median_seconds"))
        for case in common_cases
    ]
    reference_full = [
        _finite(reference_rows[case].get("phases", {}).get("full_physical_forward", {}).get("median_seconds"))
        for case in common_cases
    ]
    candidate_mean = None if any(value is None for value in candidate_full) or not candidate_full else float(np.mean(candidate_full))
    reference_mean = None if any(value is None for value in reference_full) or not reference_full else float(np.mean(reference_full))
    full_speed = {
        "status": "unavailable" if candidate_mean is None or reference_mean is None else "pass" if candidate_mean <= 0.95 * reference_mean else "fail",
        "candidate_mean_seconds": candidate_mean,
        "reference_mean_seconds": reference_mean,
        "candidate_over_reference": None if candidate_mean is None or reference_mean in (None, 0.0) else float(candidate_mean / reference_mean),
        "criterion": "mean Run-1405 full physical forward <= 0.95 * Run-1804",
        "cases": common_cases,
    }

    def training_row(label: str, bucket: str) -> Mapping[str, Any] | None:
        for row in training if isinstance(training, Sequence) else []:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("label")) == label and str(row.get("bucket", {}).get("label")) == bucket:
                return row
        return None

    candidate_m12 = training_row("1405", "M12")
    reference_m12 = training_row("1804", "M12")
    candidate_step = _finite((candidate_m12 or {}).get("measurement", {}).get("median_seconds"))
    reference_step = _finite((reference_m12 or {}).get("measurement", {}).get("median_seconds"))
    m12_speed = {
        "status": "unavailable" if candidate_step is None or reference_step is None else "pass" if candidate_step <= 0.95 * reference_step else "fail",
        "candidate_median_seconds": candidate_step,
        "reference_median_seconds": reference_step,
        "candidate_over_reference": None if candidate_step is None or reference_step in (None, 0.0) else float(candidate_step / reference_step),
        "criterion": "M12 Run-1405 optimizer-step median <= 0.95 * Run-1804",
    }

    memory_ratios: list[dict[str, Any]] = []
    for case in common_cases:
        candidate_peak = _finite(candidate_rows[case].get("phases", {}).get("full_physical_forward", {}).get("peak_allocated_bytes"))
        reference_peak = _finite(reference_rows[case].get("phases", {}).get("full_physical_forward", {}).get("peak_allocated_bytes"))
        memory_ratios.append(
            {
                "workload": f"full_forward:{case}",
                "candidate_peak_allocated_bytes": candidate_peak,
                "reference_peak_allocated_bytes": reference_peak,
                "candidate_over_reference": None if candidate_peak is None or reference_peak in (None, 0.0) else float(candidate_peak / reference_peak),
            }
        )
    candidate_m12_peak = _finite((candidate_m12 or {}).get("measurement", {}).get("peak_allocated_bytes"))
    reference_m12_peak = _finite((reference_m12 or {}).get("measurement", {}).get("peak_allocated_bytes"))
    memory_ratios.append(
        {
            "workload": "train_step:M12",
            "candidate_peak_allocated_bytes": candidate_m12_peak,
            "reference_peak_allocated_bytes": reference_m12_peak,
            "candidate_over_reference": None if candidate_m12_peak is None or reference_m12_peak in (None, 0.0) else float(candidate_m12_peak / reference_m12_peak),
        }
    )
    ratio_values = [row["candidate_over_reference"] for row in memory_ratios]
    memory_gate = {
        "status": "unavailable" if any(value is None for value in ratio_values) or not ratio_values else "pass" if all(float(value) <= 1.10 for value in ratio_values) else "fail",
        "criterion": "peak allocated memory <= 1.10 * Run-1804 on every controlled full-forward and M12 step workload",
        "workloads": memory_ratios,
    }

    semantic_rows: list[dict[str, Any]] = []
    for case, row in candidate_rows.items():
        semantic = row.get("semantic", {})
        if not isinstance(semantic, Mapping) or semantic.get("status") != "ok":
            continue
        m_active = _finite(semantic.get("M_active"))
        if m_active is None or m_active <= 1.0:
            continue
        r_m = _finite(semantic.get("R_M"))
        r_e = _finite(semantic.get("R_E"))
        semantic_rows.append({"case_id": case, "M_active": m_active, "R_M": r_m, "R_E": r_e})
    grouped_gate = {
        "status": "unavailable" if not semantic_rows or any(row["R_M"] is None or row["R_E"] is None for row in semantic_rows) else "pass" if all(float(row["R_M"]) < 1.0 and float(row["R_E"]) < 1.0 for row in semantic_rows) else "fail",
        "criterion": "actual semantic R_M < 1 and R_E < 1 on every measured multi-module evidence case",
        "cases": semantic_rows,
    }

    fidelity = payload.get("fidelity_gate", {})
    fidelity_status = str(fidelity.get("status", "unavailable")) if isinstance(fidelity, Mapping) else "unavailable"
    health = payload.get("training_health", {})
    health_status = str(health.get("status", "unavailable")) if isinstance(health, Mapping) else "unavailable"
    criteria = {
        "fidelity": {"status": fidelity_status, "criterion": "Run-1405 val field MSE <= 2 * Run-1804 epoch-50 val field MSE"},
        "training_health": {
            "status": health_status,
            "criterion": (
                "Run-1405 metrics.csv has complete epochs 1-50, finite core losses and diagnostics, "
                "positive major-group gradient/update evidence, improving validation median, and "
                "at most one last-10 value above 2x its last-10 median"
            ),
            "path": health.get("path") if isinstance(health, Mapping) else None,
            "checks": health.get("checks", {}) if isinstance(health, Mapping) else {},
            "reason": health.get("reason") if isinstance(health, Mapping) else None,
        },
        "mean_full_forward_speed": full_speed,
        "m12_step_speed": m12_speed,
        "peak_allocated_memory": memory_gate,
        "semantic_grouped_work": grouped_gate,
    }
    statuses = [str(item.get("status")) for item in criteria.values()]
    overall = "pass" if statuses and all(status == "pass" for status in statuses) else "fail" if "fail" in statuses else "unavailable"
    return {
        "status": overall,
        "criteria": criteria,
        "decision": "continue_to_epoch_500_only_if_status_is_pass",
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    protocol = payload.get("protocol", {})
    policies = payload.get("checkpoint_policy", {})
    inference = payload.get("inference", [])
    training = payload.get("training", [])
    gate = payload.get("fidelity_gate", {})
    health = payload.get("training_health", {})
    continuation = payload.get("continuation_criteria", {})
    health_checks = health.get("checks", {}) if isinstance(health, Mapping) else {}

    def health_status(name: str) -> str:
        check = health_checks.get(name, {})
        return str(check.get("status", "unavailable")) if isinstance(check, Mapping) else "unavailable"

    improvement = health_checks.get("last10_median_below_first10_median", {})
    catastrophic = health_checks.get("catastrophic_last10_instability", {})
    lines = [
        "# Run 1405 epoch-50 controlled evidence comparison",
        "",
        "This artifact is a bounded measurement/reporting entry point. It does not allocate a managed run or train a checkpoint. Timed inference disables routing-map materialization; fixed-group maps are collected in one separate untimed debug forward.",
        "",
        "## Checkpoint policy",
        "",
        "| Label | Path | Epoch role | Selection policy |",
        "|---|---|---|---|",
    ]
    for label in COMPARISON_LABELS:
        row = policies.get(label, {}) if isinstance(policies, Mapping) else {}
        lines.append(
            f"| {label} | `{row.get('path', '')}` | {row.get('epoch_role', '')} | {row.get('selection_policy', '')} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Cases: `{', '.join(protocol.get('case_ids', []))}`; Q={protocol.get('query_count')}; receiver chunk={protocol.get('receiver_chunk_size')}; inference={protocol.get('inference_warmups')} warmups/{protocol.get('inference_repetitions')} synchronized repetitions.",
            f"- Training: B={protocol.get('training_batch_size')}, Q={protocol.get('training_query_count')}, exact buckets M1/M12, {protocol.get('training_warmups')} warmup/{protocol.get('training_repetitions')} updates; fresh matched optimizers; predicted-port loss.",
            "- Memory fields are peak allocated and peak reserved CUDA bytes; CPU/planning values remain unavailable.",
            "",
            "## Inference",
            "",
            "| Label | Case | Full physical median (ms) | Prepared P2 median (ms) | Peak alloc (MiB) | Peak reserved (MiB) | P_M | P_E | R_M | R_E | sQ | sM | sE |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in inference if isinstance(inference, Sequence) else []:
        phases = row.get("phases", {})
        full = phases.get("full_physical_forward", {})
        decode = phases.get("prepared_p2_decode", {})
        sem = row.get("semantic", {})
        def ms(value: Any) -> str:
            return "NA" if value is None else f"{1000.0 * float(value):.3f}"
        def mib(value: Any) -> str:
            return "NA" if value is None else f"{float(value) / (1024.0 ** 2):.2f}"
        def scalar(key: str, values: Mapping[str, Any] = sem) -> str:
            value = values.get(key)
            return "NA" if value is None else f"{float(value):.4g}"
        lines.append(
            f"| {row.get('label')} | {row.get('case_id')} | {ms(full.get('median_seconds'))} | {ms(decode.get('median_seconds'))} | {mib(decode.get('peak_allocated_bytes'))} | {mib(decode.get('peak_reserved_bytes'))} | {scalar('P_M')} | {scalar('P_E')} | {scalar('R_M')} | {scalar('R_E')} | {scalar('sQ')} | {scalar('sM')} | {scalar('sE')} |"
        )
    lines.extend(
        [
            "",
            "## Training-step benchmark",
            "",
            "| Label | Bucket | Median step (ms) | Peak alloc (MiB) | Peak reserved (MiB) | Status |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in training if isinstance(training, Sequence) else []:
        measurement = row.get("measurement", {})
        median_value = measurement.get("median_seconds")
        allocated_value = measurement.get("peak_allocated_bytes")
        reserved_value = measurement.get("peak_reserved_bytes")
        median_text = "NA" if median_value is None else f"{1000.0 * float(median_value):.3f}"
        allocated_text = "NA" if allocated_value is None else f"{float(allocated_value) / (1024.0 ** 2):.2f}"
        reserved_text = "NA" if reserved_value is None else f"{float(reserved_value) / (1024.0 ** 2):.2f}"
        lines.append(
            f"| {row.get('label')} | {row.get('bucket', {}).get('label')} | {median_text} | {allocated_text} | {reserved_text} | {row.get('status')} |"
        )
    lines.extend(
        [
            "",
            "## Fidelity gate",
            "",
            f"Run 1405 versus Run 1804 epoch-50 validation field MSE: **{gate.get('status', 'unavailable')}**. Criterion: `{gate.get('criterion', 'candidate <= 2 * reference')}`. Run 1401 is not substituted into this gate.",
            "",
            "## Epoch-50 training health",
            "",
            f"Explicit Run-1405 metrics CSV: `{health.get('path', 'NA') if isinstance(health, Mapping) else 'NA'}`; overall status: **{health.get('status', 'unavailable') if isinstance(health, Mapping) else 'unavailable'}**.",
            f"- Epoch rows through 50: **{health_status('rows_through_epoch_50')}**; finite core train/validation losses and field MSE: **{health_status('finite_core_train_validation_metrics')}**.",
            f"- Finite aggregate/major-group gradient and update diagnostics: **{health_status('finite_gradient_update_diagnostics')}**; positive encoder/backend/head/local_coupling evidence: **{health_status('positive_major_group_gradient_update_evidence')}**.",
            f"- Validation trend: first-10 median `{improvement.get('first10_median_val_field_mse', 'NA') if isinstance(improvement, Mapping) else 'NA'}`; last-10 median `{improvement.get('last10_median_val_field_mse', 'NA') if isinstance(improvement, Mapping) else 'NA'}`; gate **{health_status('last10_median_below_first10_median')}**.",
            f"- Catastrophic rule (at most one last-10 value above 2x its last-10 median): count `{catastrophic.get('exceeding_count', 'NA') if isinstance(catastrophic, Mapping) else 'NA'}`; gate **{health_status('catastrophic_last10_instability')}**.",
            "",
            "## Continuation criteria",
            "",
            f"Overall status: **{continuation.get('status', 'unavailable')}**. The fixed thresholds are evaluated from measured rows only; an unavailable criterion does not pass. Decision: `{continuation.get('decision', 'continue_to_epoch_500_only_if_status_is_pass')}`.",
            "",
            "## Interpretation limits",
            "",
            "- `P_M/P_E/R_M/R_E/sQ/sM/sE` come from exact positive fixed-group semantic supports. They do not prove physical causality or field-value compression.",
            "- Measured CUDA latency and memory are authoritative for runtime claims; semantic ratios alone are not acceleration evidence.",
            "- Missing fixed-group debug arrays are recorded as unavailable; the tool does not infer them from dense counts.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    training_health = training_health_gate(
        None if args.metrics_1405 is None else Path(args.metrics_1405),
    )
    torch, train_bench, dynamic, stage3, loaders = _load_runtime_modules()
    make_batch, select_sample = loaders
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device={args.device}")
    paths = _checkpoint_paths(args)
    specs = [stage3.CheckpointSpec(label=label, path=path) for label, path in paths.items()]
    checkpoints: dict[str, Mapping[str, Any]] = {}
    inference_rows: list[dict[str, Any]] = []
    evidence_maps: dict[str, dict[str, str]] = {}
    map_dir = Path(args.map_dir or (Path(args.output).expanduser().resolve().parent / "maps")).expanduser().resolve()
    training_rows: list[dict[str, Any]] = []

    # Keep checkpoint payloads on CPU for provenance/config reuse, but load and
    # release one GPU model at a time. Absolute peak allocation is then a
    # comparable model/workload peak rather than the sum of three resident
    # models.
    for label, path in paths.items():
        checkpoint = stage3.load_trusted_checkpoint(path, map_location="cpu")
        checkpoints[label] = checkpoint
        if label in {"1405", "1804"}:
            _require_matched_epoch(checkpoint, label)
    try:
        for spec in specs:
            model, _ = stage3._load_model_spec(spec, device)
            dataset, _ = stage3._load_dataset(
                checkpoints[spec.label],
                SimpleNamespace(dataset=args.dataset, split=args.eval_split),
            )
            try:
                for case_id in (args.case_id or DEFAULT_CASE_IDS):
                    row, map_path = _inference_case(
                        label=spec.label,
                        model=model,
                        checkpoint=checkpoints[spec.label],
                        dataset=dataset,
                        case_id=str(case_id),
                        args=args,
                        map_dir=map_dir,
                        torch=torch,
                        dynamic=dynamic,
                        select_sample=select_sample,
                        make_batch=make_batch,
                    )
                    inference_rows.append(row)
                    if map_path is not None:
                        evidence_maps.setdefault(spec.label, {})[str(case_id)] = str(map_path)
            finally:
                close = getattr(dataset, "close", None)
                if callable(close):
                    close()
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        # Exact M1/M12 training buckets cover every explicitly requested
        # comparison model. Run 1401 remains context-only for fidelity and
        # continuation decisions because its supplied checkpoint is not epoch 50.
        train_datasets: dict[str, Any] = {}
        try:
            for label in TRAINING_LABELS:
                train_datasets[label], _ = stage3._load_dataset(
                    checkpoints[label],
                    SimpleNamespace(dataset=args.dataset, split=args.train_split),
                    points_per_case_override=int(args.train_query_count),
                    random_point_sampling_override=False,
                )
                train_datasets[label].include_grid = False
            ordered_datasets = [train_datasets[label] for label in TRAINING_LABELS]
            buckets = _make_exact_training_buckets(
                ordered_datasets,
                batch_size=int(args.train_batch_size),
            )
            reference_training = dict(checkpoints["1804"].get("train_config", {}).get("training", {}))
            reference_schedule = _matched_train_schedule(checkpoints["1804"])
            for label in TRAINING_LABELS:
                for bucket in buckets:
                    # Fresh model per bucket isolates lazy state and optimizer
                    # allocations while preserving the same checkpoint policy.
                    model, _ = stage3._load_model_spec(
                        stage3.CheckpointSpec(label=label, path=paths[label]), device
                    )
                    training_rows.append(
                        _measure_training_bucket(
                            label=label,
                            model=model,
                            checkpoint=checkpoints[label],
                            dataset=train_datasets[label],
                            bucket=bucket,
                            device=device,
                            args=args,
                            shared_training_config=reference_training,
                            shared_schedule=reference_schedule,
                            train_bench=train_bench,
                        )
                    )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
        finally:
            for dataset in train_datasets.values():
                close = getattr(dataset, "close", None)
                if callable(close):
                    close()
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidate_metric = _read_epoch_metric(
        None if args.metrics_1405 is None else Path(args.metrics_1405),
    )
    reference_metric = _read_epoch_metric(
        None if args.metrics_1804 is None else Path(args.metrics_1804),
    )
    gate = fidelity_gate(candidate_metric, reference_metric)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "task": "run1405_epoch50_comparison",
        "status": "complete" if inference_rows else "incomplete",
        "checkpoint_policy": _checkpoint_policy(paths),
        "protocol": _protocol(args),
        "inference": inference_rows,
        "training": training_rows,
        "evidence_maps": evidence_maps,
        "fidelity_metrics": {"1405": candidate_metric, "1804": reference_metric},
        "fidelity_gate": gate,
        "training_health": training_health,
        "debug_api_contract": _debug_api_contract(),
        "limitations": [
            "Run 1401 inference and optimizer steps are explicit non-gating context; it is not a matched epoch-50 fidelity competitor.",
            "Semantic P/R/s statistics are available only when the fixed-group debug API returns all canonical tensors.",
            "Peak memory is reported for CUDA only; CPU/planning records use null.",
            "This bounded evidence benchmark does not create a managed training run, save model weights, or perform a quickcheck run.",
        ],
    }
    payload["continuation_criteria"] = continuation_criteria(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-1405", required=True, type=Path, metavar="PATH")
    parser.add_argument("--checkpoint-1804", required=True, type=Path, metavar="PATH")
    parser.add_argument(
        "--checkpoint-1401",
        required=True,
        type=Path,
        metavar="PATH",
        help="explicit context checkpoint; never inferred and not used for the epoch-50 fidelity gate",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None, help="Markdown report path; defaults beside --output")
    parser.add_argument("--map-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0", help="measurement device; --plan-only does not load/use it")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--receiver-chunk-size", type=int, default=DEFAULT_RECEIVER_CHUNK_SIZE)
    parser.add_argument("--warmups", type=int, default=DEFAULT_INFERENCE_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_INFERENCE_REPETITIONS)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--train-query-count", type=int, default=DEFAULT_TRAIN_QUERY_COUNT)
    parser.add_argument("--train-warmups", type=int, default=DEFAULT_TRAIN_WARMUPS)
    parser.add_argument("--train-repetitions", type=int, default=DEFAULT_TRAIN_REPETITIONS)
    parser.add_argument(
        "--metrics-1405",
        required=True,
        type=Path,
        help="explicit Run-1405 metrics.csv required for the epoch-50 training-health gate",
    )
    parser.add_argument("--metrics-1804", type=Path, default=None, help="explicit metrics.csv containing epoch-50 val_field_mse")
    parser.add_argument("--plan-only", action="store_true", help="validate paths/protocol only; no checkpoint/model/GPU work")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    if args.plan_only:
        _write_json(args.output, plan)
        print(json.dumps(jsonable(plan), indent=2, sort_keys=True))
        return 0
    payload = run_measurement(args)
    _write_json(args.output, payload)
    report = Path(args.report).expanduser().resolve() if args.report is not None else Path(args.output).expanduser().resolve().with_suffix(".md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_markdown_report(payload), encoding="utf-8")
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True))
    return 0 if payload.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPARISON_LABELS",
    "DEFAULT_CASE_IDS",
    "DEFAULT_EPOCH",
    "TRAIN_HEALTH_CORE_COLUMNS",
    "TRAIN_HEALTH_DIAGNOSTIC_COLUMNS",
    "TRAIN_HEALTH_MAJOR_GROUPS",
    "build_parser",
    "build_plan",
    "continuation_criteria",
    "fidelity_gate",
    "main",
    "run_measurement",
    "training_health_gate",
]
