"""Thin Run-1406 versus Dense-1804 epoch-50 evidence entry point.

The command owns measurement/reporting only.  It reuses the maintained
Stage-3 loaders and the existing timing/training-step primitives, while
keeping the Run-1405 fidelity/health approval logic out of this tool.  The
official comparison is exactly two explicitly supplied epoch-50 checkpoints:
Run 1406 and Dense Run 1804.  Run 1401 is neither inferred nor required.

Plan-only validation is safe on a CPU-only host::

    python tools/diagnostics/run_group_control_comparison.py \
        --checkpoint-1406 /path/run1406_epoch_0050.pt \
        --checkpoint-1804 /path/run1804_epoch_0050.pt \
        --output diagnostics/generated/run1406/comparison.json \
        --report docs/reports/HONF_Run1406_Group_Control_Development_Report.md \
        --plan-only

Physical timing is deliberately a separate explicit invocation.  No training,
CUDA allocation, profiler, or routing-map pass occurs in plan-only mode.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_ROOT = Path(__file__).resolve().parent
if str(DIAGNOSTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTIC_ROOT))

import group_control_evidence as group_control
import run_run1405_epoch50_comparison as run1405

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
COMPARISON_LABELS = ("1406", "1804")
TRAINING_LABELS = COMPARISON_LABELS
SPEED_FACTOR = 0.95
MEMORY_FACTOR = 1.10


def _parse_path(raw: str | Path, *, name: str) -> Path:
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


def _jsonable(value: Any) -> Any:
    return run1405.jsonable(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve both official inputs; no historical checkpoint is selected."""

    return {
        "1406": _parse_path(args.checkpoint_1406, name="Run 1406"),
        "1804": _parse_path(args.checkpoint_1804, name="Run 1804"),
    }


def _checkpoint_policy(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        label: {
            "label": label,
            "path": str(path),
            "selection_policy": "explicit_cli_checkpoint",
            "required_epoch": DEFAULT_EPOCH,
            "epoch_role": (
                "matched_epoch_50_candidate" if label == "1406" else "matched_epoch_50_dense_reference"
            ),
        }
        for label, path in paths.items()
    }


def _debug_api_contract() -> dict[str, Any]:
    return {
        "preferred_container": "model output group_control_debug or interaction_aux",
        "accepted_prepared_container": "prepared_state.prepared.interaction_aux/group_control_debug",
        "map_flag": "return_group_control_maps=True (fallback return_routing_maps=True when explicitly supported)",
        "group_count": group_control.GROUP_COUNT,
        "control_dimension": group_control.CONTROL_DIM,
        "arrays": {
            "group_control_module_incidence": "[B,M,K] nonnegative A_m",
            "group_control_environment_incidence": "[B,E,K] nonnegative A_e",
            "group_control_query_routing": "[B,Q,K] nonnegative alpha_qk",
            "group_control_module_overlap": "[B,Q,M] rho_M; derivable from A_m and alpha",
            "group_control_environment_overlap": "[B,Q,E] rho_E; derivable from A_e and alpha",
            "group_control_*_control_moment": "[B,Q,N,D] n; D=16",
            "group_control_h": "[B,K,16] group control state; exact n maps may be derived as alpha*A*h",
        },
        "phase_ledger": {
            "key": "group_control_phase_ledger",
            "phases": ["P0", "P1", "P2", "P2_consistency"],
            "raw_prefix_adapter": {
                "initial_port_": "P0",
                "provisional_": "P1",
                "unprefixed_group_control_*": "P2",
                "port_global_": "P2_consistency",
            },
            "source_types": ["module", "environment"],
            "required_fields": [
                "logical_path_count",
                "unique_pair_count",
                "actual_fine_call_count",
                "module_mlp_rows",
                "environment_geometry_network_rows",
                "environment_content_rows",
                "scalar_control_rows",
                "source_projection_rows",
                "forward_call_count",
                "checkpoint_recompute_count",
                "valid_pair_denominator",
                "padded_pair_denominator",
            ],
            "chunk_policy": "sum numerators/denominators over receiver chunks before ratios; never average chunk ratios",
        },
        "semantics": "learned interaction routing, not physical causality; logical q->group->source paths are distinct from unique executed q->source pairs",
    }


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_epoch": DEFAULT_EPOCH,
        "case_ids": [str(value) for value in (args.case_id or DEFAULT_CASE_IDS)],
        "query_count": int(args.query_count),
        "receiver_chunk_size": int(args.receiver_chunk_size),
        "inference_warmups": int(args.warmups),
        "inference_repetitions": int(args.repetitions),
        "timed_maps": False,
        "timed_profiler": False,
        "debug_map_pass": "one untimed full-Q forward after timing per checkpoint/case",
        "same_gpu_session": True,
        "checkpoint_residency_policy": "load one model at a time; release before loading the next",
        "training_labels": list(TRAINING_LABELS),
        "training_batch_size": int(args.train_batch_size),
        "training_query_count": int(args.train_query_count),
        "training_buckets": ["M1", "M12"],
        "training_warmups": int(args.train_warmups),
        "training_repetitions": int(args.train_repetitions),
        "training_optimizer": "fresh disposable optimizer per model/bucket, matched to Dense 1804",
        "training_port_condition": "predicted",
        "reverse_order_repeat_requested": bool(args.reverse_order_repeat),
        "reverse_order_repeat_policy": "one reverse-order repeat only when explicitly requested for borderline timing",
        "quickcheck_run": False,
        "managed_training": False,
    }


def _decision_rubric() -> dict[str, Any]:
    return {
        "mean_full_forward_speed": "mean Run-1406 full-forward median across cases <= 0.95 * Dense 1804",
        "m12_step_speed": "Run-1406 M12 optimizer-step median <= 0.95 * Dense 1804",
        "peak_allocated_memory": "absolute allocated peak <= 1.10 * Dense 1804 on each full-forward and M12 workload",
        "learning": "finite outputs/parameters/gradients, meaningful updates, improving validation trajectory, no unresolved catastrophic instability when metrics are supplied",
        "execution_semantics": "actual fine calls agree with unique q->source ledger where backend ledger is available",
        "r_ratio": "not a continuation requirement; R_M/R_E may remain at or above one",
        "borderline_rule": "if a timing ratio is within the observed spread of the 0.95 target, request at most one reverse-order repeat; do not manufacture a pass",
        "decision_use": "research-budget evidence only, not a permanent runtime or CI gate",
    }


def _validate_protocol(args: argparse.Namespace) -> None:
    cases = tuple(str(value) for value in (args.case_id or DEFAULT_CASE_IDS))
    if not {"0273", "0653"}.issubset(cases):
        raise ValueError("the Run-1406 protocol requires cases 0273 and 0653")
    exact = {
        "query_count": DEFAULT_QUERY_COUNT,
        "receiver_chunk_size": DEFAULT_RECEIVER_CHUNK_SIZE,
        "warmups": DEFAULT_INFERENCE_WARMUPS,
        "repetitions": DEFAULT_INFERENCE_REPETITIONS,
        "train_batch_size": DEFAULT_TRAIN_BATCH_SIZE,
        "train_query_count": DEFAULT_TRAIN_QUERY_COUNT,
        "train_warmups": DEFAULT_TRAIN_WARMUPS,
        "train_repetitions": DEFAULT_TRAIN_REPETITIONS,
    }
    for name, expected in exact.items():
        value = int(getattr(args, name))
        if value != expected:
            raise ValueError(f"Run-1406 controlled protocol requires {name}={expected}, got {value}")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Validate explicit paths/protocol without importing runtime modules."""

    paths = _checkpoint_paths(args)
    _validate_protocol(args)
    return {
        "schema_version": 1,
        "task": "run1406_group_control_epoch50_comparison",
        "status": "plan_only",
        "checkpoint_policy": _checkpoint_policy(paths),
        "protocol": _protocol(args),
        "decision_rubric": _decision_rubric(),
        "debug_api_contract": _debug_api_contract(),
        "planned_outputs": {
            "comparison_json": str(Path(args.output).expanduser().resolve()),
            "report": str(
                Path(args.report).expanduser().resolve() if args.report else Path(args.output).expanduser().resolve().with_suffix(".md")
            ),
            "map_directory": str(
                Path(args.map_dir or (Path(args.output).expanduser().resolve().parent / "maps")).expanduser().resolve()
            ),
        },
        "limitations": [
            "Plan-only mode does not load checkpoints, datasets, models, CUDA, or routing maps.",
            "The planned debug map is opt-in and excluded from all timed phases.",
            "1405/1401 are historical context only and are not required live comparators.",
            "A board visualizes learned interaction structure; it does not establish physical causality.",
        ],
    }


def _accepted_parameter_names(callable_object: Any) -> set[str]:
    try:
        return set(inspect.signature(callable_object).parameters)
    except (TypeError, ValueError):
        return set()


def _map_kwargs(callable_object: Any, enabled: bool) -> dict[str, Any]:
    """Select only explicit map flags exposed by the current backend."""

    names = _accepted_parameter_names(callable_object)
    kwargs: dict[str, Any] = {}
    if "return_group_control_maps" in names:
        kwargs["return_group_control_maps"] = bool(enabled)
    if "return_routing_maps" in names:
        kwargs["return_routing_maps"] = bool(enabled)
    return kwargs


def _forward(
    model: Any,
    structure: Any,
    query: Any,
    forward_kwargs: Mapping[str, Any],
    *,
    return_prepared_state: bool = False,
    maps: bool = False,
) -> Any:
    kwargs = dict(forward_kwargs)
    kwargs.update(_map_kwargs(model.forward, maps))
    if return_prepared_state:
        kwargs["return_prepared_state"] = True
    return model(structure, query, **kwargs)


def _decode_prepared(model: Any, prepared: Any, query: Any, receiver_chunk: int) -> Any:
    kwargs = {"receiver_chunk_size": int(receiver_chunk)}
    kwargs.update(_map_kwargs(model.decode_prepared, False))
    return model.decode_prepared(prepared, query, **kwargs)


def _extract_prepared(output: Any) -> Any:
    if isinstance(output, Mapping) and "prepared_state" in output:
        return output["prepared_state"]
    prepared = getattr(output, "prepared_state", None)
    if prepared is not None:
        return prepared
    raise KeyError("prepared forward did not return prepared_state")


def _measurement_summary(measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Add explicit median/spread and baseline memory summaries."""

    result = dict(measurement)
    samples = [row for row in measurement.get("samples", []) if isinstance(row, Mapping)]
    complete = [row for row in samples if row.get("status") == "complete"]

    def values(key: str) -> list[float]:
        return [float(value) for row in complete if (value := _finite(row.get(key))) is not None]

    elapsed = values("elapsed_seconds")
    result["spread_seconds"] = None if not elapsed else float(max(elapsed) - min(elapsed))
    for field in (
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "incremental_peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
    ):
        field_values = values(field)
        result[f"median_{field}"] = None if not field_values else float(np.median(field_values))
        result[f"spread_{field}"] = None if not field_values else float(max(field_values) - min(field_values))
    result["timed_maps"] = False
    result["timed_profiler"] = False
    return result


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
    select_sample: Any,
    make_batch: Any,
) -> tuple[dict[str, Any], Path | None]:
    sample = select_sample(dataset, str(case_id), 0)
    query_np = dynamic._query_points(sample, int(args.query_count))
    if len(query_np) != int(args.query_count):
        raise ValueError(f"case {case_id} produced Q={len(query_np)}; expected {args.query_count}")
    batch = make_batch(dict(sample), query_np, torch.device(args.device))
    query = batch["query_xy"]
    forward_kwargs = run1405._phase_forward_kwargs(batch)
    receiver_chunk = int(args.receiver_chunk_size)
    device = torch.device(args.device)

    def full_forward() -> Any:
        with dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
            return _forward(model, batch["structure"], query, forward_kwargs)

    def prepare_one() -> Any:
        with dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
            return _forward(
                model,
                batch["structure"],
                query[:, :1],
                forward_kwargs,
                return_prepared_state=True,
            )

    prepared_holder: dict[str, Any] = {}

    def prepared_decode() -> Any:
        with dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
            return _decode_prepared(model, prepared_holder["prepared"], query, receiver_chunk)

    phases = {
        "full_physical_forward": _measurement_summary(
            run1405._measure_phase(
                full_forward,
                device,
                warmups=int(args.warmups),
                repetitions=int(args.repetitions),
            )
        ),
        "physical_preparation_plus_one_query": _measurement_summary(
            run1405._measure_phase(
                prepare_one,
                device,
                warmups=int(args.warmups),
                repetitions=int(args.repetitions),
            )
        ),
    }
    with torch.inference_mode():
        prepared_output = prepare_one()
    prepared_holder["prepared"] = _extract_prepared(prepared_output)
    phases["prepared_p2_decode"] = _measurement_summary(
        run1405._measure_phase(
            prepared_decode,
            device,
            warmups=int(args.warmups),
            repetitions=int(args.repetitions),
        )
    )
    prepared_holder.clear()
    del prepared_output
    gc.collect()

    map_path: Path | None = None
    semantic: dict[str, Any] = {"status": "unavailable"}
    ledger: dict[str, Any] = {phase: {"status": "unavailable"} for phase in group_control.PHASES}
    debug_error: str | None = None
    try:
        with torch.inference_mode(), dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
            debug_forward_kwargs = {
                **forward_kwargs,
                "return_port_global_consistency": True,
            }
            debug_output = _forward(
                model,
                batch["structure"],
                query,
                debug_forward_kwargs,
                return_prepared_state=True,
                maps=True,
            )
        payload = group_control.collect_group_control_payload(debug_output, query_xy=query)
        arrays, phase_records = group_control.canonicalize_group_control_arrays(payload)
        decode_seconds = phases["prepared_p2_decode"].get("median_seconds")
        debug_metrics = group_control.semantic_metrics(
            arrays,
            prepared_decode_median_ms=None if decode_seconds is None else float(decode_seconds) * 1000.0,
        )
        ledger = group_control.phase_ledger(arrays, phase_records)
        map_dir.mkdir(parents=True, exist_ok=True)
        map_path = group_control.save_group_control_npz(
            map_dir / f"{label}__{case_id}.npz",
            arrays,
            debug_metrics,
            ledger,
            metadata={
                "label": label,
                "case_id": str(case_id),
                "checkpoint_epoch": run1405._epoch_of(checkpoint),
                "prepared_decode_median_ms": None if decode_seconds is None else float(decode_seconds) * 1000.0,
                "timed_maps": False,
                "debug_api": _debug_api_contract(),
            },
        )
        semantic = {"status": "ok", **debug_metrics.values}
        del debug_output
    except Exception as exc:  # noqa: BLE001 - preserve optional map diagnostics
        debug_error = f"{type(exc).__name__}: {exc}"
        semantic = {"status": "unavailable", "reason": debug_error}

    return (
        {
            "label": label,
            "case_id": str(case_id),
            "query_count": int(query.shape[1]),
            "receiver_chunk_size": receiver_chunk,
            "checkpoint_epoch": run1405._epoch_of(checkpoint),
            "phases": phases,
            "semantic": semantic,
            "phase_ledger": ledger,
            "debug_map_path": None if map_path is None else str(map_path),
            "debug_api_error": debug_error,
            "timed_maps": False,
            "timed_profiler": False,
            "route_semantics": "learned interaction routes, not physical causality",
        },
        map_path,
    )


def _run_inference_order(
    *,
    order: Sequence[str],
    paths: Mapping[str, Path],
    checkpoints: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    torch: Any,
    dynamic: Any,
    stage3: Any,
    select_sample: Any,
    make_batch: Any,
    map_dir: Path,
    order_name: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    evidence_maps: dict[str, dict[str, str]] = {}
    device = torch.device(args.device)
    for label in order:
        model, _ = stage3._load_model_spec(stage3.CheckpointSpec(label=label, path=paths[label]), device)
        dataset, _ = stage3._load_dataset(
            checkpoints[label],
            SimpleNamespace(dataset=args.dataset, split=args.eval_split),
        )
        try:
            for case_id in (args.case_id or DEFAULT_CASE_IDS):
                row, map_path = _inference_case(
                    label=label,
                    model=model,
                    checkpoint=checkpoints[label],
                    dataset=dataset,
                    case_id=str(case_id),
                    args=args,
                    map_dir=map_dir / order_name,
                    torch=torch,
                    dynamic=dynamic,
                    select_sample=select_sample,
                    make_batch=make_batch,
                )
                row["measurement_order"] = order_name
                rows.append(row)
                if map_path is not None:
                    evidence_maps.setdefault(label, {})[str(case_id)] = str(map_path)
        finally:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    return rows, evidence_maps


def _training_rows(
    *,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    checkpoints: Mapping[str, Mapping[str, Any]],
    torch: Any,
    train_bench: Any,
    stage3: Any,
) -> list[dict[str, Any]]:
    """Perform real M1/M12 forward/backward/clip/update timing on disposable models."""

    device = torch.device(args.device)
    train_datasets: dict[str, Any] = {}
    training_rows: list[dict[str, Any]] = []
    try:
        for label in TRAINING_LABELS:
            train_datasets[label], _ = stage3._load_dataset(
                checkpoints[label],
                SimpleNamespace(dataset=args.dataset, split=args.train_split),
                points_per_case_override=int(args.train_query_count),
                random_point_sampling_override=False,
            )
            train_datasets[label].include_grid = False
        ordered = [train_datasets[label] for label in TRAINING_LABELS]
        buckets = run1405._make_exact_training_buckets(
            ordered,
            batch_size=int(args.train_batch_size),
        )
        shared_training_config = dict(checkpoints["1804"].get("train_config", {}).get("training", {}))
        shared_schedule = run1405._matched_train_schedule(checkpoints["1804"])
        for label in TRAINING_LABELS:
            for bucket in buckets:
                model, _ = stage3._load_model_spec(
                    stage3.CheckpointSpec(label=label, path=paths[label]),
                    device,
                )
                row = run1405._measure_training_bucket(
                    label=label,
                    model=model,
                    checkpoint=checkpoints[label],
                    dataset=train_datasets[label],
                    bucket=bucket,
                    device=device,
                    args=args,
                    shared_training_config=shared_training_config,
                    shared_schedule=shared_schedule,
                    train_bench=train_bench,
                )
                row["measurement"] = _measurement_summary(row.get("measurement", {}))
                row["update_semantics"] = "one real warmup optimizer update plus three measured forward/backward/clip/update steps"
                training_rows.append(row)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
    finally:
        for dataset in train_datasets.values():
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
    return training_rows


def _timing_borderline(inference_rows: Sequence[Mapping[str, Any]], training_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return workloads whose observed spread touches the five-percent target."""

    borderline: list[str] = []
    candidate = {str(row.get("case_id")): row for row in inference_rows if str(row.get("label")) == "1406"}
    reference = {str(row.get("case_id")): row for row in inference_rows if str(row.get("label")) == "1804"}
    for case_id in sorted(set(candidate) & set(reference)):
        candidate_phase = candidate[case_id].get("phases", {}).get("full_physical_forward", {})
        reference_phase = reference[case_id].get("phases", {}).get("full_physical_forward", {})
        candidate_seconds = _finite(candidate_phase.get("median_seconds"))
        reference_seconds = _finite(reference_phase.get("median_seconds"))
        if candidate_seconds is None or reference_seconds is None or reference_seconds <= 0.0:
            continue
        ratio = candidate_seconds / reference_seconds
        spread = (_finite(candidate_phase.get("spread_seconds")) or 0.0) + (_finite(reference_phase.get("spread_seconds")) or 0.0)
        margin = max(spread / reference_seconds, 0.005)
        if abs(ratio - SPEED_FACTOR) <= margin:
            borderline.append(f"full_forward:{case_id}")

    candidate_m12 = next((row for row in training_rows if str(row.get("label")) == "1406" and str(row.get("bucket", {}).get("label")) == "M12"), None)
    reference_m12 = next((row for row in training_rows if str(row.get("label")) == "1804" and str(row.get("bucket", {}).get("label")) == "M12"), None)
    candidate_measurement = (candidate_m12 or {}).get("measurement", {})
    reference_measurement = (reference_m12 or {}).get("measurement", {})
    candidate_seconds = _finite(candidate_measurement.get("median_seconds"))
    reference_seconds = _finite(reference_measurement.get("median_seconds"))
    if candidate_seconds is not None and reference_seconds is not None and reference_seconds > 0.0:
        ratio = candidate_seconds / reference_seconds
        spread = (_finite(candidate_measurement.get("spread_seconds")) or 0.0) + (_finite(reference_measurement.get("spread_seconds")) or 0.0)
        margin = max(spread / reference_seconds, 0.005)
        if abs(ratio - SPEED_FACTOR) <= margin:
            borderline.append("train_step:M12")
    return borderline


def _read_epoch_metric(path: Path | str | None, *, epoch: int = DEFAULT_EPOCH) -> dict[str, Any]:
    if path is None:
        return {"status": "unavailable", "reason": "no metrics CSV supplied"}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return {"status": "unavailable", "reason": f"metrics file not found: {source}", "path": str(source)}
    try:
        rows = list(csv.DictReader(source.open("r", newline="", encoding="utf-8")))
    except OSError as exc:
        return {"status": "unavailable", "reason": str(exc), "path": str(source)}
    for row in rows:
        try:
            row_epoch = int(float(row.get("epoch", row.get("current_epoch"))))
        except (TypeError, ValueError):
            continue
        if row_epoch != int(epoch):
            continue
        for key in ("val_field_mse", "validation_field_mse", "val_loss_field", "field_mse"):
            value = _finite(row.get(key))
            if value is not None:
                return {"status": "ok", "epoch": int(epoch), "metric": value, "metric_name": key, "path": str(source)}
    return {"status": "unavailable", "reason": f"no finite epoch-{epoch} field-MSE row", "path": str(source)}


def _trajectory_evidence(path: Path | str | None, *, epoch: int = DEFAULT_EPOCH) -> dict[str, Any]:
    """Small learning-evidence summary, not a copied Run-1405 gate."""

    if path is None:
        return {"status": "unavailable", "reason": "no Run-1406 metrics CSV supplied"}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return {"status": "unavailable", "reason": f"metrics file not found: {source}"}
    try:
        rows = list(csv.DictReader(source.open("r", newline="", encoding="utf-8")))
    except OSError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    by_epoch: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        try:
            value = int(float(row.get("epoch", row.get("current_epoch"))))
        except (TypeError, ValueError):
            continue
        by_epoch.setdefault(value, row)
    required = list(range(1, int(epoch) + 1))
    val_values = [_finite(by_epoch.get(index, {}).get("val_field_mse")) for index in required]
    first = [value for value in val_values[:10] if value is not None]
    last = [value for value in val_values[-10:] if value is not None]
    first_median = None if len(first) != 10 else float(np.median(first))
    last_median = None if len(last) != 10 else float(np.median(last))
    catastrophic = [] if last_median is None else [index for index, value in zip(required[-10:], val_values[-10:]) if value is not None and value > 2.0 * last_median]
    diagnostic_columns = [
        column
        for column in (by_epoch.get(int(epoch), {}) if by_epoch else {})
        if ("gradient" in str(column) or "update" in str(column))
    ]
    diagnostic_values = {column: _finite(by_epoch.get(int(epoch), {}).get(column)) for column in diagnostic_columns}
    positive_diagnostics = [value for value in diagnostic_values.values() if value is not None and value > 0.0]
    checks = {
        "rows_through_epoch_50": not any(index not in by_epoch for index in required),
        "finite_validation_field_mse": len(val_values) == len(required) and all(value is not None for value in val_values),
        "improving_validation_median": first_median is not None and last_median is not None and last_median < first_median,
        "catastrophic_last10_count_at_most_one": last_median is not None and len(catastrophic) <= 1,
        "positive_gradient_or_update_evidence": bool(positive_diagnostics),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "path": str(source),
        "target_epoch": int(epoch),
        "checks": checks,
        "first10_median_val_field_mse": first_median,
        "last10_median_val_field_mse": last_median,
        "catastrophic_last10_epochs": catastrophic,
        "diagnostic_values_at_epoch": diagnostic_values,
        "note": "trajectory evidence informs the epoch-50 budget decision; it is not a permanent model-validity gate",
    }


def _rows_for(payload: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    values = payload.get("inference", [])
    return [row for row in values if isinstance(row, Mapping) and str(row.get("label")) == label] if isinstance(values, Sequence) else []


def _training_row(payload: Mapping[str, Any], label: str, bucket: str) -> Mapping[str, Any] | None:
    values = payload.get("training", [])
    if not isinstance(values, Sequence):
        return None
    for row in values:
        if isinstance(row, Mapping) and str(row.get("label")) == label and str(row.get("bucket", {}).get("label")) == bucket:
            return row
    return None


def _ratio_status(candidate: float | None, reference: float | None, factor: float, criterion: str) -> dict[str, Any]:
    if candidate is None or reference is None or reference <= 0.0:
        return {"status": "unavailable", "candidate": candidate, "reference": reference, "criterion": criterion}
    ratio = float(candidate / reference)
    return {
        "status": "pass" if ratio <= factor else "fail",
        "candidate": candidate,
        "reference": reference,
        "ratio": ratio,
        "threshold": factor,
        "criterion": criterion,
    }


def decision_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the written Run-1406 budget rubric from measured rows."""

    candidate_rows = {str(row.get("case_id")): row for row in _rows_for(payload, "1406")}
    reference_rows = {str(row.get("case_id")): row for row in _rows_for(payload, "1804")}
    common_cases = sorted(set(candidate_rows) & set(reference_rows))
    candidate_full = [_finite(candidate_rows[case].get("phases", {}).get("full_physical_forward", {}).get("median_seconds")) for case in common_cases]
    reference_full = [_finite(reference_rows[case].get("phases", {}).get("full_physical_forward", {}).get("median_seconds")) for case in common_cases]
    candidate_mean = None if not candidate_full or any(value is None for value in candidate_full) else float(np.mean(candidate_full))
    reference_mean = None if not reference_full or any(value is None for value in reference_full) else float(np.mean(reference_full))
    full_speed = _ratio_status(
        candidate_mean,
        reference_mean,
        SPEED_FACTOR,
        "mean Run-1406 full-forward median <= 0.95 * Dense 1804",
    )
    full_speed["cases"] = common_cases

    candidate_m12 = _training_row(payload, "1406", "M12")
    reference_m12 = _training_row(payload, "1804", "M12")
    m12_speed = _ratio_status(
        _finite((candidate_m12 or {}).get("measurement", {}).get("median_seconds")),
        _finite((reference_m12 or {}).get("measurement", {}).get("median_seconds")),
        SPEED_FACTOR,
        "Run-1406 M12 optimizer-step median <= 0.95 * Dense 1804",
    )

    memory_workloads: list[dict[str, Any]] = []
    for case in common_cases:
        candidate_memory = _finite(candidate_rows[case].get("phases", {}).get("full_physical_forward", {}).get("peak_allocated_bytes"))
        reference_memory = _finite(reference_rows[case].get("phases", {}).get("full_physical_forward", {}).get("peak_allocated_bytes"))
        memory_workloads.append({"workload": f"full_forward:{case}", **_ratio_status(candidate_memory, reference_memory, MEMORY_FACTOR, "allocated peak <= 1.10 * Dense 1804")})
    candidate_memory = _finite((candidate_m12 or {}).get("measurement", {}).get("peak_allocated_bytes"))
    reference_memory = _finite((reference_m12 or {}).get("measurement", {}).get("peak_allocated_bytes"))
    memory_workloads.append({"workload": "train_step:M12", **_ratio_status(candidate_memory, reference_memory, MEMORY_FACTOR, "allocated peak <= 1.10 * Dense 1804")})
    memory_statuses = [str(row["status"]) for row in memory_workloads]
    memory_gate = {
        "status": "pass" if memory_statuses and all(status == "pass" for status in memory_statuses) else "fail" if "fail" in memory_statuses else "unavailable",
        "criterion": "absolute allocated peak <= 1.10 * Dense 1804 for each full-forward and M12 workload",
        "workloads": memory_workloads,
    }

    trajectory = payload.get("learning_evidence", {}).get("1406", {})
    learning_status = str(trajectory.get("status", "unavailable")) if isinstance(trajectory, Mapping) else "unavailable"
    learning = {
        "status": learning_status,
        "criterion": _decision_rubric()["learning"],
        "evidence": trajectory,
    }

    ledger_rows: list[dict[str, Any]] = []
    for case in common_cases:
        row = candidate_rows[case]
        ledger = row.get("phase_ledger", {})
        p2 = ledger.get("P2", {}) if isinstance(ledger, Mapping) else {}
        source_statuses: list[str] = []
        source_rows: list[dict[str, Any]] = []
        for source_kind in group_control.SOURCE_TYPES:
            source = p2.get(source_kind, {}) if isinstance(p2, Mapping) else {}
            actual = _finite(source.get("actual_fine_call_count")) if isinstance(source, Mapping) else None
            unique = _finite(source.get("unique_pair_count")) if isinstance(source, Mapping) else None
            status = "pass" if actual is not None and unique is not None and actual == unique else "fail" if actual is not None and unique is not None else "unavailable"
            source_statuses.append(status)
            source_rows.append({"source": source_kind, "status": status, "actual_fine_call_count": actual, "unique_pair_count": unique})
        ledger_rows.append({"case_id": case, "status": "pass" if source_statuses and all(status == "pass" for status in source_statuses) else "fail" if "fail" in source_statuses else "unavailable", "sources": source_rows})
    ledger_statuses = [str(row["status"]) for row in ledger_rows]
    semantics = {
        "status": "pass" if ledger_statuses and all(status == "pass" for status in ledger_statuses) else "fail" if "fail" in ledger_statuses else "unavailable",
        "criterion": "P2 actual fine calls equal unique q->source pairs when backend ledger is available; no R<1 requirement",
        "cases": ledger_rows,
    }
    criteria = {
        "mean_full_forward_speed": full_speed,
        "m12_step_speed": m12_speed,
        "peak_allocated_memory": memory_gate,
        "learning": learning,
        "execution_semantics": semantics,
    }
    statuses = [str(item.get("status")) for item in criteria.values()]
    overall = "pass" if statuses and all(status == "pass" for status in statuses) else "fail" if "fail" in statuses else "unavailable"
    return {
        "status": overall,
        "criteria": criteria,
        "decision": "continue_to_epoch_500_only_if_all_evidence_is_pass; otherwise leave at epoch 50 and request a decision",
        "thresholds": {"speed_factor": SPEED_FACTOR, "memory_factor": MEMORY_FACTOR},
        "note": "R_M/R_E below one is deliberately not required for Run 1406 continuation evidence.",
    }


def continuation_criteria(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility name for callers that consume the existing report API."""

    return decision_evidence(payload)


def _ms(value: Any) -> str:
    return "NA" if _finite(value) is None else f"{1000.0 * float(value):.3f}"


def _mib(value: Any) -> str:
    return "NA" if _finite(value) is None else f"{float(value) / (1024.0 ** 2):.2f}"


def _metric(value: Any) -> str:
    return "NA" if _finite(value) is None else f"{float(value):.4g}"


def _markdown_report(payload: Mapping[str, Any]) -> str:
    protocol = payload.get("protocol", {})
    policies = payload.get("checkpoint_policy", {})
    inference = payload.get("inference", [])
    training = payload.get("training", [])
    evidence = payload.get("decision_evidence", {})
    fidelity = payload.get("fidelity_metrics", {})
    learning = payload.get("learning_evidence", {})
    lines = [
        "# Run 1406 Group-Control Development Report",
        "",
        "This is an epoch-50 evidence artifact for the low-dimensional group-control hypothesis. Learned interaction routes are not physical causality. The comparison is reporting evidence, not a permanent runtime or CI gate.",
        "",
        "## Checkpoint policy",
        "",
        "| Label | Path | Required epoch | Role | Selection |",
        "|---|---|---:|---|---|",
    ]
    for label in COMPARISON_LABELS:
        row = policies.get(label, {}) if isinstance(policies, Mapping) else {}
        lines.append(f"| {label} | `{row.get('path', '')}` | {row.get('required_epoch', '')} | {row.get('epoch_role', '')} | {row.get('selection_policy', '')} |")
    lines.extend(
        [
            "",
            "## Controlled protocol",
            "",
            f"- Same-GPU comparison: cases `{', '.join(protocol.get('case_ids', []))}`; Q={protocol.get('query_count')}; receiver chunk={protocol.get('receiver_chunk_size')}; {protocol.get('inference_warmups')} warmups/{protocol.get('inference_repetitions')} synchronized repetitions.",
            "- Timed phases: full physical forward and prepared P2 decode; routing maps and profiler are excluded from timed calls.",
            f"- Training: real fixed M1/M12 buckets, B={protocol.get('training_batch_size')}, Q={protocol.get('training_query_count')}; {protocol.get('training_warmups')} warmup/{protocol.get('training_repetitions')} measured forward/backward/clip/update steps with fresh disposable optimizers.",
            f"- Model residency: `{protocol.get('checkpoint_residency_policy')}`. Reverse-order repeat requested: `{protocol.get('reverse_order_repeat_requested')}`.",
            "",
            "## Inference benchmark",
            "",
            "Each phase reports a median, min-to-max spread, pre-call allocated/reserved baseline, and absolute/incremental allocated/reserved peaks. Values are NA until the physical benchmark is executed.",
            "",
            "| Label | Case | Full median/spread (ms) | P2 median/spread (ms) | Full baseline alloc/res (MiB) | Full abs/inc alloc (MiB) | Full abs/inc reserved (MiB) | P2 baseline alloc/res (MiB) | P2 abs/inc alloc (MiB) | P2 abs/inc reserved (MiB) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in inference if isinstance(inference, Sequence) else []:
        phases = row.get("phases", {})
        full = phases.get("full_physical_forward", {})
        decode = phases.get("prepared_p2_decode", {})
        full_time = f"{_ms(full.get('median_seconds'))}/{_ms(full.get('spread_seconds'))}"
        decode_time = f"{_ms(decode.get('median_seconds'))}/{_ms(decode.get('spread_seconds'))}"
        full_baseline = f"{_mib(full.get('median_baseline_allocated_bytes'))}/{_mib(full.get('median_baseline_reserved_bytes'))}"
        full_alloc = f"{_mib(full.get('peak_allocated_bytes'))}/{_mib(full.get('median_incremental_peak_allocated_bytes'))}"
        full_reserved = f"{_mib(full.get('peak_reserved_bytes'))}/{_mib(full.get('median_incremental_peak_reserved_bytes'))}"
        decode_baseline = f"{_mib(decode.get('median_baseline_allocated_bytes'))}/{_mib(decode.get('median_baseline_reserved_bytes'))}"
        decode_alloc = f"{_mib(decode.get('peak_allocated_bytes'))}/{_mib(decode.get('median_incremental_peak_allocated_bytes'))}"
        decode_reserved = f"{_mib(decode.get('peak_reserved_bytes'))}/{_mib(decode.get('median_incremental_peak_reserved_bytes'))}"
        lines.append(f"| {row.get('label')} | {row.get('case_id')} | {full_time} | {decode_time} | {full_baseline} | {full_alloc} | {full_reserved} | {decode_baseline} | {decode_alloc} | {decode_reserved} |")
    reverse = payload.get("reverse_order_repeat", {})
    if isinstance(reverse, Mapping):
        lines.extend(["", f"Reverse-order repeat: **{reverse.get('status', 'unavailable')}**; borderline workloads: `{reverse.get('borderline_workloads', [])}`."])
    lines.extend(
        [
            "",
            "## Training-step benchmark",
            "",
            "| Label | Bucket | Median/spread (ms) | Baseline alloc/res (MiB) | Absolute/incremental alloc (MiB) | Absolute/incremental reserved (MiB) | Status |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in training if isinstance(training, Sequence) else []:
        measurement = row.get("measurement", {})
        lines.append(
            f"| {row.get('label')} | {row.get('bucket', {}).get('label')} | {_ms(measurement.get('median_seconds'))}/{_ms(measurement.get('spread_seconds'))} | {_mib(measurement.get('median_baseline_allocated_bytes'))}/{_mib(measurement.get('median_baseline_reserved_bytes'))} | {_mib(measurement.get('peak_allocated_bytes'))}/{_mib(measurement.get('median_incremental_peak_allocated_bytes'))} | {_mib(measurement.get('peak_reserved_bytes'))}/{_mib(measurement.get('median_incremental_peak_reserved_bytes'))} | {row.get('status')} |"
        )
    lines.extend(["", "## Logical versus executed work ledger", "", "The board and ledger keep logical q→group→source paths separate from unique q→source candidates. A logical multiplicity is cheap-control work; it is not a fine GPU call.", ""])
    lines.extend(["| Label | Case | Phase | Source | Logical | Unique | Multiplicity | Actual fine calls | MLP rows | Env geometry rows | Env content rows | Scalar rows | Source projections | Forward calls | Recomputation | Valid/padded denominator |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in inference if isinstance(inference, Sequence) else []:
        ledger = row.get("phase_ledger", {})
        for phase in group_control.PHASES:
            phase_row = ledger.get(phase, {}) if isinstance(ledger, Mapping) else {}
            for source_kind in group_control.SOURCE_TYPES:
                source = phase_row.get(source_kind, {}) if isinstance(phase_row, Mapping) else {}
                if not isinstance(source, Mapping):
                    source = {}
                lines.append(
                    f"| {row.get('label')} | {row.get('case_id')} | {phase} | {source_kind} | {_metric(source.get('logical_path_count'))} | {_metric(source.get('unique_pair_count'))} | {_metric(source.get('multiplicity'))} | {_metric(source.get('actual_fine_call_count'))} | {_metric(source.get('module_mlp_rows'))} | {_metric(source.get('environment_geometry_network_rows'))} | {_metric(source.get('environment_content_rows'))} | {_metric(source.get('scalar_control_rows'))} | {_metric(source.get('source_projection_rows'))} | {_metric(source.get('forward_call_count'))} | {_metric(source.get('checkpoint_recompute_count'))} | {_metric(source.get('valid_pair_denominator'))}/{_metric(source.get('padded_pair_denominator'))} |"
                )
    lines.extend(
        [
            "",
            "## Epoch-50 decision evidence",
            "",
            f"Overall status: **{evidence.get('status', 'unavailable')}**. Thresholds are mean full-forward ratio ≤ {SPEED_FACTOR:.2f}, M12 step ratio ≤ {SPEED_FACTOR:.2f}, and allocated peak ratio ≤ {MEMORY_FACTOR:.2f}; R_M/R_E < 1 is not required.",
        ]
    )
    criteria = evidence.get("criteria", {}) if isinstance(evidence, Mapping) else {}
    for name, value in criteria.items() if isinstance(criteria, Mapping) else []:
        lines.append(f"- `{name}`: **{value.get('status', 'unavailable') if isinstance(value, Mapping) else 'unavailable'}** — {value.get('criterion', '') if isinstance(value, Mapping) else ''}")
    lines.extend(["", "## Learning and modest fidelity evidence", ""])
    for label in ("1406", "1804"):
        metric = fidelity.get(label, {}) if isinstance(fidelity, Mapping) else {}
        trajectory = learning.get(label, {}) if isinstance(learning, Mapping) else {}
        metric_text = "NA" if not isinstance(metric, Mapping) else f"{metric.get('metric', 'NA')} ({metric.get('status', 'unavailable')})"
        trajectory_text = trajectory.get("status", "unavailable") if isinstance(trajectory, Mapping) else "unavailable"
        lines.append(f"- Run {label}: epoch-50 val field MSE `{metric_text}`; trajectory evidence **{trajectory_text}**.")
    lines.extend(["- A roughly twofold field-MSE ratio is a severe-regression flag for review, not an automatic convergence verdict."])
    lines.extend(
        [
            "",
            "## Debug/API contract",
            "",
            "The preferred opt-in flag is `return_group_control_maps=True`; the existing routing-map flag is used only when explicitly exposed by the backend. P0/P1/P2/P2_consistency records must report logical and unique numerators, actual calls, module MLP rows, environment geometry/content rows, scalar controls, source projections, forward calls, checkpoint recomputations, and valid/padded denominators. Receiver-chunk numerators and denominators are summed before ratios.",
            "",
            "## Interpretation and limits",
            "",
            "- Empty group centres are unavailable (NaN), never silently placed at an origin.",
            "- The debug map pass is untimed and is not a replacement for measured decode time.",
            "- Unique-pair support can remain dense; lower R is not itself an acceleration result.",
            "- This artifact executed the declared GPU benchmark; it did not launch managed training, a profiler, quickcheck, or continuation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    """Run the explicitly requested physical comparison on one device."""

    torch, train_bench, dynamic, stage3, loaders = run1405._load_runtime_modules()
    make_batch, select_sample = loaders
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device={args.device}")
    paths = _checkpoint_paths(args)
    checkpoints: dict[str, Mapping[str, Any]] = {}
    for label, path in paths.items():
        checkpoint = stage3.load_trusted_checkpoint(path, map_location="cpu")
        run1405._require_matched_epoch(checkpoint, label)
        checkpoints[label] = checkpoint

    map_dir = Path(args.map_dir or (Path(args.output).expanduser().resolve().parent / "maps")).expanduser().resolve()
    inference_rows, evidence_maps = _run_inference_order(
        order=COMPARISON_LABELS,
        paths=paths,
        checkpoints=checkpoints,
        args=args,
        torch=torch,
        dynamic=dynamic,
        stage3=stage3,
        select_sample=select_sample,
        make_batch=make_batch,
        map_dir=map_dir,
        order_name="forward",
    )
    training_rows = _training_rows(
        args=args,
        paths=paths,
        checkpoints=checkpoints,
        torch=torch,
        train_bench=train_bench,
        stage3=stage3,
    )
    reverse_repeat: dict[str, Any] = {
        "status": "not_requested",
        "borderline_workloads": [],
        "rows": [],
        "evidence_maps": {},
    }
    if args.reverse_order_repeat:
        borderline_workloads = _timing_borderline(inference_rows, training_rows)
        reverse_repeat["borderline_workloads"] = borderline_workloads
        if borderline_workloads:
            reverse_rows, reverse_maps = _run_inference_order(
                order=tuple(reversed(COMPARISON_LABELS)),
                paths=paths,
                checkpoints=checkpoints,
                args=args,
                torch=torch,
                dynamic=dynamic,
                stage3=stage3,
                select_sample=select_sample,
                make_batch=make_batch,
                map_dir=map_dir,
                order_name="reverse",
            )
            reverse_repeat.update({"status": "complete", "rows": reverse_rows, "evidence_maps": reverse_maps})
        else:
            reverse_repeat["status"] = "not_run_not_borderline"
            reverse_repeat["reason"] = "the measured ratios were not within their observed spread of the five-percent target"
    learning_evidence = {
        "1406": _trajectory_evidence(args.metrics_1406),
        "1804": _trajectory_evidence(args.metrics_1804),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "task": "run1406_group_control_epoch50_comparison",
        "status": "complete" if inference_rows else "incomplete",
        "checkpoint_policy": _checkpoint_policy(paths),
        "protocol": _protocol(args),
        "inference": inference_rows,
        "reverse_order_repeat": reverse_repeat,
        "training": training_rows,
        "evidence_maps": evidence_maps,
        "reverse_evidence_maps": reverse_repeat["evidence_maps"],
        "fidelity_metrics": {
            "1406": _read_epoch_metric(args.metrics_1406),
            "1804": _read_epoch_metric(args.metrics_1804),
        },
        "learning_evidence": learning_evidence,
        "decision_rubric": _decision_rubric(),
        "debug_api_contract": _debug_api_contract(),
        "limitations": [
            "Run 1405/1401 are not mandatory live comparators for this Run-1406 decision.",
            "Timing and memory claims use same-device measured rows; untimed debug maps are evidence only.",
            "The trajectory summary is a modest epoch-50 learning check, not a permanent approval gate.",
            "No physical causality is inferred from learned routing weights or group centres.",
        ],
    }
    payload["decision_evidence"] = decision_evidence(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-1406", required=True, type=Path, metavar="PATH")
    parser.add_argument("--checkpoint-1804", required=True, type=Path, metavar="PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--map-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0", help="same physical device for both checkpoints")
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
    parser.add_argument("--metrics-1406", type=Path, default=None)
    parser.add_argument("--metrics-1804", type=Path, default=None)
    parser.add_argument("--reverse-order-repeat", action="store_true", help="request one bounded reverse-order repeat")
    parser.add_argument("--plan-only", action="store_true", help="validate paths/protocol only; no runtime/GPU work")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    if args.plan_only:
        _write_json(args.output, plan)
        print(json.dumps(_jsonable(plan), indent=2, sort_keys=True))
        return 0
    payload = run_measurement(args)
    _write_json(args.output, payload)
    report = Path(args.report).expanduser().resolve() if args.report is not None else Path(args.output).expanduser().resolve().with_suffix(".md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_markdown_report(payload), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0 if payload.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPARISON_LABELS",
    "DEFAULT_CASE_IDS",
    "DEFAULT_EPOCH",
    "build_parser",
    "build_plan",
    "continuation_criteria",
    "decision_evidence",
    "main",
    "run_measurement",
]
