"""Matched Run-1407 epoch-50 cost benchmark.

This entry point reuses the maintained Run-1406/Run-1405 evidence helpers and
changes only the comparison set: one explicit Run-1407 checkpoint is measured
against matched Run-1406 and Dense-1804 checkpoints.  Plan-only mode is safe on
CPU and performs no model, CUDA, dataset, or optimizer work.

The physical invocation uses one caller-selected device and keeps only one
model resident at a time.  Inference is fixed to cases 0273/0653, Q=8192,
receiver chunks of 2048, two warmups, and five synchronized repetitions for
full forward, preparation plus one query, and prepared P2 decode.  Training
uses the maintained real predicted-port M1/M12 B48/Q1024 path with one
warmup and three measured forward/backward/clip/update steps.  The training
optimizer is disposable and matched to the Dense-1804 training configuration;
checkpoint optimizer state is not restored or written back.

Support/executed/padded/recomputed rows are copied only from an explicit
normal-executor phase ledger.  Missing actual-row fields remain unavailable;
the tool never relabels a dense denominator or a diagnostic estimate as
executed work.
"""

from __future__ import annotations

import argparse
import gc
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
import run_group_control_comparison as run1406
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
COMPARISON_LABELS = ("1407", "1406", "1804")
TRAINING_LABELS = COMPARISON_LABELS


def _parse_path(raw: str | Path, *, name: str) -> Path:
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} checkpoint does not exist: {path}")
    return path


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
    return {
        "1407": _parse_path(args.checkpoint_1407, name="Run 1407"),
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
                "matched_epoch_50_candidate"
                if label == "1407"
                else "matched_epoch_50_run1406_reference"
                if label == "1406"
                else "matched_epoch_50_dense_reference"
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
        "timed_maps": False,
        "timed_profiler": False,
        "timed_detailed_ledgers": False,
        "training_labels": list(TRAINING_LABELS),
        "training_batch_size": int(args.train_batch_size),
        "training_query_count": int(args.train_query_count),
        "training_buckets": ["M1", "M12"],
        "training_warmups": int(args.train_warmups),
        "training_repetitions": int(args.train_repetitions),
        "training_port_condition": "predicted",
        "optimizer_state_policy": {
            "kind": "fresh_disposable_optimizer",
            "reference_hyperparameters": "Run-1804 checkpoint training config",
            "restore_checkpoint_optimizer_state": False,
            "save_updated_checkpoint": False,
        },
        "checkpoint_residency_policy": "one model resident at a time; release and empty cache before next label/bucket",
        "managed_training": False,
    }


def _validate_protocol(args: argparse.Namespace) -> None:
    cases = tuple(str(value) for value in (args.case_id or DEFAULT_CASE_IDS))
    if set(cases) != set(DEFAULT_CASE_IDS) or len(cases) != len(DEFAULT_CASE_IDS):
        raise ValueError("the Run-1407 matched benchmark requires exactly cases 0273 and 0653")
    expected = {
        "query_count": DEFAULT_QUERY_COUNT,
        "receiver_chunk_size": DEFAULT_RECEIVER_CHUNK_SIZE,
        "warmups": DEFAULT_INFERENCE_WARMUPS,
        "repetitions": DEFAULT_INFERENCE_REPETITIONS,
        "train_batch_size": DEFAULT_TRAIN_BATCH_SIZE,
        "train_query_count": DEFAULT_TRAIN_QUERY_COUNT,
        "train_warmups": DEFAULT_TRAIN_WARMUPS,
        "train_repetitions": DEFAULT_TRAIN_REPETITIONS,
    }
    for name, expected_value in expected.items():
        value = int(getattr(args, name))
        if value != expected_value:
            raise ValueError(f"controlled Run-1407 protocol requires {name}={expected_value}, got {value}")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Validate explicit files/protocol without importing model runtime code."""

    paths = _checkpoint_paths(args)
    _validate_protocol(args)
    output = Path(args.output).expanduser().resolve()
    return {
        "schema_version": 1,
        "task": "run1407_epoch50_matched_benchmark",
        "status": "plan_only",
        "checkpoint_policy": _checkpoint_policy(paths),
        "protocol": _protocol(args),
        "planned_outputs": {
            "comparison_json": str(output),
            "report": str(Path(args.report).expanduser().resolve() if args.report else output.with_suffix(".md")),
            "map_directory": str(Path(args.map_dir or (output.parent / "maps")).expanduser().resolve()),
        },
        "counter_contract": {
            "source": "explicit normal-executor phase ledger from one untimed map/debug pass",
            "fields": ["support_pairs", "executed_rows", "padded_rows", "recomputed_rows"],
            "missing_policy": "unavailable; do not infer actual rows from support or dense denominators",
            "support_semantics": "positive rho and positive source measure, separately from logical paths",
        },
        "limitations": [
            "Plan-only mode does not load checkpoints, datasets, models, CUDA, or optimizers.",
            "Timed inference calls disable maps, detailed ledgers, and profiling; the debug pass is untimed.",
            "Training uses real disposable predicted-port optimizer steps only when the caller executes the physical command.",
            "No managed training run, checkpoint write, continuation, or profiler trace is launched by this tool.",
        ],
    }


def _counter_value(record: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = _finite(record.get(name))
        if value is not None:
            return value
    return None


def _normal_executor_counters(phase_ledger: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize explicit support/executor rows without inventing missing data."""

    result: dict[str, Any] = {}
    records = phase_ledger if isinstance(phase_ledger, Mapping) else {}
    for phase in group_control.PHASES:
        phase_entry = records.get(phase, {})
        phase_result: dict[str, Any] = {}
        for source_kind in group_control.SOURCE_TYPES:
            raw = phase_entry.get(source_kind, {}) if isinstance(phase_entry, Mapping) else {}
            record = raw if isinstance(raw, Mapping) else {}
            support = _counter_value(record, ("support_pair_count", "unique_pair_count", "support_pairs"))
            executed = _counter_value(
                record,
                ("executed_rows", "actual_fine_call_count", "fine_rows_forward", "actual_fine_rows"),
            )
            padded = _counter_value(record, ("padded_rows", "actual_padded_rows", "fine_rows_padded"))
            recomputed = _counter_value(
                record,
                ("recomputed_rows", "checkpoint_recompute_count", "checkpoint_recomputations", "fine_rows_recompute"),
            )
            row = {
                "support_pairs": None if support is None else int(support),
                "executed_rows": None if executed is None else int(executed),
                "padded_rows": None if padded is None else int(padded),
                "recomputed_rows": None if recomputed is None else int(recomputed),
                "logical_paths": _counter_value(record, ("logical_path_count", "logical_paths")),
                "valid_pair_denominator": _counter_value(
                    record, ("valid_pair_denominator", "valid_pairs", "dense_valid_pair_count")
                ),
                "padded_pair_denominator": _counter_value(record, ("padded_pair_denominator", "padded_pair_count")),
                "source": "normal_executor_phase_ledger" if record else "unavailable",
            }
            required = ("support_pairs", "executed_rows", "padded_rows", "recomputed_rows")
            row["status"] = "ok" if all(row[key] is not None for key in required) else "unavailable"
            phase_result[source_kind] = row
        result[phase] = phase_result
    return result


def _measurement_summary(measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Retain medians/spreads and allocated/reserved baseline/peak fields."""

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
        observed = values(field)
        result[f"median_{field}"] = None if not observed else float(np.median(observed))
        result[f"spread_{field}"] = None if not observed else float(max(observed) - min(observed))
    result.update({"timed_maps": False, "timed_profiler": False, "timed_detailed_ledgers": False})
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
) -> dict[str, Any]:
    """Reuse the maintained Run-1406 inference phases and normalize counters."""

    row, _ = run1406._inference_case(
        label=label,
        model=model,
        checkpoint=checkpoint,
        dataset=dataset,
        case_id=str(case_id),
        args=args,
        map_dir=map_dir,
        torch=torch,
        dynamic=dynamic,
        select_sample=select_sample,
        make_batch=make_batch,
    )
    row["normal_executor_counters"] = _normal_executor_counters(row.get("phase_ledger"))
    row["counter_policy"] = "explicit normal-executor rows only; unavailable fields are not inferred"
    return row


def _run_inference(
    *,
    paths: Mapping[str, Path],
    checkpoints: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    torch: Any,
    dynamic: Any,
    stage3: Any,
    select_sample: Any,
    make_batch: Any,
    map_dir: Path,
) -> list[dict[str, Any]]:
    """Measure each label while retaining only one model on the device."""

    rows: list[dict[str, Any]] = []
    device = torch.device(args.device)
    for label in COMPARISON_LABELS:
        model, _ = stage3._load_model_spec(stage3.CheckpointSpec(label=label, path=paths[label]), device)
        dataset, _ = stage3._load_dataset(
            checkpoints[label],
            SimpleNamespace(dataset=args.dataset, split=args.eval_split),
        )
        try:
            for case_id in args.case_id or DEFAULT_CASE_IDS:
                rows.append(
                    _inference_case(
                        label=label,
                        model=model,
                        checkpoint=checkpoints[label],
                        dataset=dataset,
                        case_id=str(case_id),
                        args=args,
                        map_dir=map_dir,
                        torch=torch,
                        dynamic=dynamic,
                        select_sample=select_sample,
                        make_batch=make_batch,
                    )
                )
        finally:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    return rows


def _run_training(
    *,
    paths: Mapping[str, Path],
    checkpoints: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    torch: Any,
    train_bench: Any,
    stage3: Any,
) -> list[dict[str, Any]]:
    """Run real disposable M1/M12 steps one model at a time."""

    device = torch.device(args.device)
    datasets: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    try:
        for label in TRAINING_LABELS:
            datasets[label], _ = stage3._load_dataset(
                checkpoints[label],
                SimpleNamespace(dataset=args.dataset, split=args.train_split),
                points_per_case_override=int(args.train_query_count),
                random_point_sampling_override=False,
            )
            datasets[label].include_grid = False
        buckets = run1405._make_exact_training_buckets(
            [datasets[label] for label in TRAINING_LABELS],
            batch_size=int(args.train_batch_size),
        )
        reference_training = dict(checkpoints["1804"].get("train_config", {}).get("training", {}))
        reference_schedule = run1405._matched_train_schedule(checkpoints["1804"])
        for label in TRAINING_LABELS:
            for bucket in buckets:
                model, _ = stage3._load_model_spec(
                    stage3.CheckpointSpec(label=label, path=paths[label]),
                    device,
                )
                try:
                    row = run1405._measure_training_bucket(
                        label=label,
                        model=model,
                        checkpoint=checkpoints[label],
                        dataset=datasets[label],
                        bucket=bucket,
                        device=device,
                        args=args,
                        shared_training_config=reference_training,
                        shared_schedule=reference_schedule,
                        train_bench=train_bench,
                    )
                    row["measurement"] = _measurement_summary(row["measurement"])
                    row["optimizer_state_policy"] = _protocol(args)["optimizer_state_policy"]
                    row["normal_executor_counters"] = {
                        "status": "unavailable",
                        "reason": "maintained training-step primitive does not return a phase ledger; no diagnostic path was substituted",
                    }
                    rows.append(row)
                finally:
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
    finally:
        for dataset in datasets.values():
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
    return rows


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the explicit physical benchmark on the caller-selected device."""

    _validate_protocol(args)
    paths = _checkpoint_paths(args)
    torch, train_bench, dynamic, stage3, loaders = run1405._load_runtime_modules()
    make_batch, select_sample = loaders
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device={args.device}")

    checkpoints: dict[str, Mapping[str, Any]] = {}
    for label, path in paths.items():
        checkpoint = stage3.load_trusted_checkpoint(path, map_location="cpu")
        run1405._require_matched_epoch(checkpoint, label)
        checkpoints[label] = checkpoint

    output = Path(args.output).expanduser().resolve()
    map_dir = Path(args.map_dir or (output.parent / "maps")).expanduser().resolve()
    inference = _run_inference(
        paths=paths,
        checkpoints=checkpoints,
        args=args,
        torch=torch,
        dynamic=dynamic,
        stage3=stage3,
        select_sample=select_sample,
        make_batch=make_batch,
        map_dir=map_dir,
    )
    training = _run_training(
        paths=paths,
        checkpoints=checkpoints,
        args=args,
        torch=torch,
        train_bench=train_bench,
        stage3=stage3,
    )
    return {
        "schema_version": 1,
        "task": "run1407_epoch50_matched_benchmark",
        "status": "complete" if inference and training else "incomplete",
        "checkpoint_policy": _checkpoint_policy(paths),
        "protocol": _protocol(args),
        "inference": inference,
        "training": training,
        "limitations": [
            "No managed training run, checkpoint write, continuation, quickcheck, or profiler was launched.",
            "Timed calls disable maps/ledgers; counter rows come only from an untimed normal-executor debug pass.",
            "Missing padded/executed/recomputed fields remain unavailable rather than being inferred from support counts.",
            "Training models and optimizers are disposable; checkpoint optimizer state is not restored or persisted.",
        ],
    }


def _ms(value: Any) -> str:
    return "NA" if _finite(value) is None else f"{1000.0 * float(value):.3f}"


def _mib(value: Any) -> str:
    return "NA" if _finite(value) is None else f"{float(value) / (1024.0**2):.2f}"


def _markdown_report(payload: Mapping[str, Any]) -> str:
    protocol = payload.get("protocol", {})
    lines = [
        "# Run 1407 Epoch-50 Matched Benchmark",
        "",
        "This artifact compares explicit epoch-50 Run 1407, Run 1406, and Dense 1804 checkpoints. It is cost/learning evidence, not a CFD-truth or permanent CI gate.",
        "",
        "## Protocol",
        "",
        f"- Cases `{', '.join(protocol.get('case_ids', []))}`; Q={protocol.get('query_count')}; receiver chunk={protocol.get('receiver_chunk_size')}; {protocol.get('inference_warmups')} warmups/{protocol.get('inference_repetitions')} synchronized repetitions.",
        f"- Training: predicted ports, B={protocol.get('training_batch_size')}, Q={protocol.get('training_query_count')}, M1/M12; {protocol.get('training_warmups')} warmup/{protocol.get('training_repetitions')} measured disposable optimizer steps.",
        f"- Residency: `{protocol.get('checkpoint_residency_policy')}`. Optimizer policy: `{protocol.get('optimizer_state_policy', {}).get('kind')}`; restore checkpoint optimizer state `{protocol.get('optimizer_state_policy', {}).get('restore_checkpoint_optimizer_state')}`.",
        "",
        "## Inference cost",
        "",
        "| Label | Case | Full ms | Prep+one query ms | P2 decode ms | Full peak alloc/res MiB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("inference", []):
        if not isinstance(row, Mapping):
            continue
        phases = row.get("phases", {})
        full = phases.get("full_physical_forward", {})
        prep = phases.get("physical_preparation_plus_one_query", {})
        p2 = phases.get("prepared_p2_decode", {})
        lines.append(
            f"| {row.get('label')} | {row.get('case_id')} | {_ms(full.get('median_seconds'))} | {_ms(prep.get('median_seconds'))} | {_ms(p2.get('median_seconds'))} | {_mib(full.get('peak_allocated_bytes'))}/{_mib(full.get('peak_reserved_bytes'))} |"
        )
    lines.extend(
        [
            "",
            "## Normal-executor counters",
            "",
            "Support pairs, executed rows, padded rows, and checkpoint recomputations remain separate. Missing actual-row fields are unavailable.",
            "",
            "| Label | Case | Phase | Source | Support | Executed | Padded | Recomputed | Status |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload.get("inference", []):
        if not isinstance(row, Mapping):
            continue
        counters = row.get("normal_executor_counters", {})
        for phase, phase_row in counters.items() if isinstance(counters, Mapping) else []:
            for source, values in phase_row.items() if isinstance(phase_row, Mapping) else []:
                if not isinstance(values, Mapping):
                    continue
                lines.append(
                    f"| {row.get('label')} | {row.get('case_id')} | {phase} | {source} | {values.get('support_pairs', 'NA')} | {values.get('executed_rows', 'NA')} | {values.get('padded_rows', 'NA')} | {values.get('recomputed_rows', 'NA')} | {values.get('status', 'unavailable')} |"
                )
    lines.extend(
        [
            "",
            "## Training cost",
            "",
            "| Label | Bucket | Median ms | Peak alloc/res MiB | Status |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in payload.get("training", []):
        if not isinstance(row, Mapping):
            continue
        measurement = row.get("measurement", {})
        lines.append(
            f"| {row.get('label')} | {row.get('bucket', {}).get('label')} | {_ms(measurement.get('median_seconds'))} | {_mib(measurement.get('peak_allocated_bytes'))}/{_mib(measurement.get('peak_reserved_bytes'))} | {row.get('status')} |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "The benchmark does not launch training or continue a run by itself. A missing execution ledger is reported as unavailable; support counts are never relabeled as executed or padded rows.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-1407", required=True, type=Path)
    parser.add_argument("--checkpoint-1406", required=True, type=Path)
    parser.add_argument("--checkpoint-1804", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--map-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:1", help="caller-selected physical device")
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
    parser.add_argument("--plan-only", action="store_true")
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
    report = (
        Path(args.report).expanduser().resolve()
        if args.report
        else Path(args.output).expanduser().resolve().with_suffix(".md")
    )
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
    "main",
    "run_measurement",
]
