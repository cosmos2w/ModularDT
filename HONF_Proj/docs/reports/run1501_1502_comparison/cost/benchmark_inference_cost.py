#!/usr/bin/env python3
"""Matched, inference-only cost benchmark for selected ThermalChannel checkpoints.

The primary scope is the maintained ``predict_case`` evaluator (including its
full-grid query loop and CPU result transfer).  The same case is also timed as
one direct ``Q``-point forward, one preparation plus a single query, and a
prepared P2 decode over those same ``Q`` points.  Timing probes run with maps
off; an untimed maps-on pass records exposed pair/row execution ledgers.

This script never constructs an optimizer, calls backward, or writes model
state.  It loads and releases one checkpoint model at a time.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DIAGNOSTIC_DIR = PROJECT_ROOT / "tools" / "diagnostics"
for _path in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "Case_ThermalChannel" / "src",
    DIAGNOSTIC_DIR,
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _jsonable(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() == 1:
                return _jsonable(value.item())
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
    except ImportError:  # pragma: no cover - PyTorch is required by the runner
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (ValueError, TypeError, RuntimeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_case_ids(args: argparse.Namespace) -> list[str]:
    values = [str(value) for value in args.case_id]
    if args.case_list is not None:
        path = Path(args.case_list).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames and "case_id" in reader.fieldnames:
                    values.extend(str(row["case_id"]) for row in reader if row.get("case_id"))
                else:
                    values.extend(
                        field.strip()
                        for row in csv.reader(text.splitlines())
                        for field in row
                        if field.strip() and field.strip().lower() != "case_id"
                    )
        else:
            values.extend(line.strip() for line in text.splitlines() if line.strip())
    result = list(dict.fromkeys(value.strip() for value in values if value.strip()))
    if not result:
        raise ValueError("supply case IDs with --case-id or --case-list")
    return result


def _training_summary(checkpoint_path: Path) -> dict[str, Any]:
    run_dir = next(
        (parent for parent in checkpoint_path.resolve().parents if (parent / "summary.json").is_file()),
        None,
    )
    result: dict[str, Any] = {"run_summary_path": None, "run_summary_status": "unavailable"}
    if run_dir is None:
        return result
    result["run_summary_path"] = str(run_dir / "summary.json")
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["run_summary_error"] = f"{type(exc).__name__}: {exc}"
        return result
    result.update(
        {
            "run_summary_status": "available",
            "run_dir": str(run_dir),
            "epochs": summary.get("epochs"),
            "trainable_parameter_count_recorded": summary.get("trainable_parameter_count"),
            "actual_train_wall_seconds": summary.get("actual_train_wall_seconds"),
            "actual_validation_wall_seconds": summary.get("actual_validation_wall_seconds"),
            "actual_total_epoch_wall_seconds": summary.get("actual_total_epoch_wall_seconds"),
            "peak_cuda_memory_mb_recorded": summary.get("peak_cuda_memory_mb"),
        }
    )
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result["manifest_status"] = manifest.get("status")
            result["manifest_started_at"] = manifest.get("started_at")
            result["manifest_ended_at"] = manifest.get("ended_at")
            result["manifest_last_completed_epoch"] = manifest.get("last_completed_epoch")
            result["launch_resources"] = manifest.get("launch_resources")
        except (OSError, json.JSONDecodeError) as exc:
            result["manifest_error"] = f"{type(exc).__name__}: {exc}"
    software_path = run_dir / "environment" / "software.json"
    if software_path.is_file():
        try:
            software = json.loads(software_path.read_text(encoding="utf-8"))
            result["training_cuda_device_count"] = software.get("cuda_device_count")
            result["training_cuda_devices"] = software.get("cuda_devices")
            result["training_torch_version"] = software.get("torch")
        except (OSError, json.JSONDecodeError):
            pass
    # The trainer summary is used instead of per-epoch CSV timing columns.
    # Resumed Run 1804 contains malformed post-resume timing rows.
    result["timing_source"] = "run summary actual_*_wall_seconds; metrics.csv timing columns not used"
    return result


def _parameter_counts(model: Any) -> dict[str, int]:
    parameters = list(model.parameters())
    return {
        "parameter_count": sum(int(parameter.numel()) for parameter in parameters),
        "trainable_parameter_count": sum(int(parameter.numel()) for parameter in parameters if parameter.requires_grad),
    }


def _ledger_scalars(output: Any) -> dict[str, Any]:
    """Keep scalar support/execution counters from an untimed output only."""
    import torch

    if not isinstance(output, dict):
        return {"output_type": type(output).__name__}
    keys_of_interest = (
        "pair",
        "row",
        "support",
        "executor",
        "geometry",
        "logical",
        "candidate",
        "active",
        "dense",
        "fine",
        "padded",
    )
    flat: dict[str, Any] = {"output_keys": sorted(str(key) for key in output)}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
            return
        if torch.is_tensor(value):
            if value.numel() != 1:
                return
            value = value.detach().cpu().item()
        if isinstance(value, (int, float, bool)):
            leaf = prefix.rsplit(".", 1)[-1].lower()
            if any(token in leaf for token in keys_of_interest):
                flat[prefix] = value
        elif isinstance(value, str) and "executor" in prefix.lower():
            flat[prefix] = value

    for root in ("interaction_aux", "routing_aux", "provisional_interaction_aux"):
        if root in output:
            visit(root, output[root])
    return flat


def _try_measure(
    measure: Any, function: Any, *, torch: Any, device: Any, warmups: int, repetitions: int
) -> dict[str, Any]:
    try:
        return measure(torch, device, function, warmups, repetitions)
    except Exception as exc:  # noqa: BLE001 - preserve the failed exact scope in provenance
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "warmups": int(warmups),
            "repetitions": int(repetitions),
        }


def _measurements_to_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in payload["rows"]:
        for phase, measurement in record.get("phases", {}).items():
            rows.append(
                {
                    "label": record["label"],
                    "checkpoint": record["checkpoint"],
                    "checkpoint_epoch": record["checkpoint_epoch"],
                    "architecture": record["architecture"],
                    "case_id": record["case_id"],
                    "phase": phase,
                    "status": measurement.get("status", "complete"),
                    "median_wall_ms": measurement.get("median_wall_ms"),
                    "p95_wall_ms": measurement.get("p95_wall_ms"),
                    "median_cuda_event_ms": measurement.get("median_cuda_event_ms"),
                    "peak_incremental_allocated_mib": _to_mib(measurement.get("peak_incremental_allocated_bytes")),
                    "peak_incremental_reserved_mib": _to_mib(measurement.get("peak_incremental_reserved_bytes")),
                    "baseline_allocated_mib": _to_mib(_first_sample_value(measurement, "baseline_allocated_bytes")),
                    "baseline_reserved_mib": _to_mib(_first_sample_value(measurement, "baseline_reserved_bytes")),
                    "application_grid_query_count": record.get("application_grid_query_count"),
                    "direct_query_count": record.get("direct_query_count"),
                    "application_query_batch_size": record.get("application_query_batch_size"),
                    "configured_inner_receiver_chunk_size": record.get("configured_inner_receiver_chunk_size"),
                }
            )
    return rows


def _first_sample_value(measurement: dict[str, Any], key: str) -> Any:
    samples = measurement.get("samples", [])
    return samples[0].get(key) if samples else None


def _to_mib(value: Any) -> float | None:
    return None if value is None else float(value) / (1024.0**2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import run_run1405_epoch50_comparison as run1405
    import run_run1501_selected_executor_benchmark as run1501

    torch, _run1405, stage3, make_batch, predict_case, select_sample = run1501._imports()
    specs = stage3.parse_checkpoint_specs(args.checkpoint)
    case_ids = _read_case_ids(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for {device}")
        torch.cuda.set_device(device)
        device_name = torch.cuda.get_device_name(device)
        device_properties = torch.cuda.get_device_properties(device)
        gpu_metadata = {
            "device_name": device_name,
            "total_memory_bytes": int(device_properties.total_memory),
            "capability": [int(device_properties.major), int(device_properties.minor)],
        }
    else:
        gpu_metadata = {"device_name": None, "total_memory_bytes": None, "capability": None}

    payload: dict[str, Any] = {
        "schema_version": 1,
        "task": "run1501_1502_matched_inference_cost",
        "status": "running",
        "protocol": {
            "device": str(device),
            "gpu": gpu_metadata,
            "split": str(args.split),
            "case_ids": case_ids,
            "case_count": len(case_ids),
            "direct_query_count": int(args.query_count),
            "application_query_batch_size": int(args.query_batch_size),
            "inner_receiver_chunk_policy": "checkpoint-native; no runtime override",
            "mixed_teacher_ratio": float(args.mixed_teacher_ratio),
            "warmups": int(args.warmups),
            "repetitions": int(args.repetitions),
            "timing": "synchronized wall time and CUDA events via run1501 benchmark helper",
            "timed_maps": False,
            "ledger_probe": "one untimed maps-on direct forward per checkpoint and case",
            "primary_scope": "channelthermal.evaluation.prepared.predict_case including full-grid evaluation and CPU transfer",
        },
        "checkpoints": {},
        "rows": [],
        "notes": [
            "No optimizer is created, no backward call is made, and no checkpoint/model state is written.",
            "Exactly one model is resident on the selected GPU at a time.",
            "Per-epoch metrics.csv timing columns are not used for training-cost totals; run summary actual_*_wall_seconds are used.",
            "Pair/row counters are collected by an untimed maps-on pass and describe exposed implementation ledgers, not measured arithmetic FLOPs.",
        ],
    }

    for spec_index, spec in enumerate(specs, start=1):
        model, checkpoint = stage3._load_model_spec(spec, device)
        architecture = str(getattr(getattr(model.config, "core_honf", None), "forward_architecture", "legacy_honf"))
        parameter_info = _parameter_counts(model)
        summary_info = _training_summary(Path(spec.path))
        payload["checkpoints"][spec.label] = {
            "path": str(Path(spec.path).resolve()),
            "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
            "architecture": architecture,
            **parameter_info,
            "training_summary": summary_info,
            "executor_policy": getattr(getattr(getattr(model, "core", None), "backend", None), "executor_policy", None),
            "configured_inner_receiver_chunk_size": getattr(getattr(model, "core", None), "receiver_chunk_size", None),
            "runtime_inner_receiver_chunk_override": None,
            "diagnostic_executor_independent": getattr(
                getattr(getattr(model, "core", None), "backend", None),
                "diagnostic_executor_independent",
                None,
            ),
        }
        dataset, dataset_path = stage3._load_dataset(
            checkpoint,
            SimpleNamespace(dataset=args.dataset, split=args.split),
        )
        try:
            model.eval()
            for case_index, case_id in enumerate(case_ids, start=1):
                sample = select_sample(dataset, case_id, 0)
                query_np = run1501._query_points(sample, int(args.query_count))
                batch = make_batch(dict(sample), query_np, device)
                query = batch["query_xy"]
                kwargs = run1405._phase_forward_kwargs(batch)
                # Match the selection/evaluation pass for local port blending.
                kwargs["mixed_teacher_ratio"] = float(args.mixed_teacher_ratio)

                def full_forward(
                    model: Any = model,
                    structure: Any = batch["structure"],
                    query: Any = query,
                    kwargs: dict[str, Any] = kwargs,
                ) -> Any:
                    return model(
                        structure,
                        query,
                        return_prepared_state=False,
                        return_routing_maps=False,
                        **kwargs,
                    )

                def preparation_plus_one_query(
                    model: Any = model,
                    structure: Any = batch["structure"],
                    query: Any = query,
                    kwargs: dict[str, Any] = kwargs,
                ) -> Any:
                    return model(
                        structure,
                        query[:, :1],
                        return_prepared_state=True,
                        return_routing_maps=False,
                        **kwargs,
                    )

                sample_copy = dict(sample)

                def application_evaluator(
                    model: Any = model,
                    sample: dict[str, Any] = sample_copy,
                    predict_case: Any = predict_case,
                    device: Any = device,
                    query_batch_size: int = int(args.query_batch_size),
                    mixed_teacher_ratio: float = float(args.mixed_teacher_ratio),
                ) -> Any:
                    return predict_case(
                        model,
                        sample,
                        device,
                        query_batch_size=query_batch_size,
                        local_port_condition_mode="predicted",
                        mixed_teacher_ratio=mixed_teacher_ratio,
                        return_routing_maps=False,
                        return_topology_signature=False,
                        return_prepared_state=False,
                    )

                configured_inner_chunk = getattr(getattr(model, "core", None), "receiver_chunk_size", None)
                with torch.inference_mode():
                    full_measure = _try_measure(
                        run1501._measure,
                        full_forward,
                        torch=torch,
                        device=device,
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                    )
                    prep_measure = _try_measure(
                        run1501._measure,
                        preparation_plus_one_query,
                        torch=torch,
                        device=device,
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                    )
                    app_measure = _try_measure(
                        run1501._measure,
                        application_evaluator,
                        torch=torch,
                        device=device,
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                    )

                    prepared_state = None
                    decode_measure: dict[str, Any]
                    try:
                        prepared_state = run1501._prepare(model, batch, kwargs)
                        if not callable(getattr(model, "decode_prepared", None)):
                            raise TypeError("model has no decode_prepared method")

                        def prepared_p2_decode(
                            model: Any = model,
                            prepared_state: Any = prepared_state,
                            query: Any = query,
                        ) -> Any:
                            return model.decode_prepared(
                                prepared_state,
                                query,
                                return_routing_maps=False,
                                return_edge_fields=False,
                            )

                        decode_measure = _try_measure(
                            run1501._measure,
                            prepared_p2_decode,
                            torch=torch,
                            device=device,
                            warmups=args.warmups,
                            repetitions=args.repetitions,
                        )
                    except Exception as exc:  # noqa: BLE001 - retain this scope's failure as data
                        decode_measure = {
                            "status": "unavailable",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    del prepared_state

                    ledger: dict[str, Any]
                    try:
                        ledger_output = model(
                            batch["structure"],
                            query,
                            return_prepared_state=False,
                            return_routing_maps=True,
                            **kwargs,
                        )
                        ledger = _ledger_scalars(ledger_output)
                        del ledger_output
                    except Exception as exc:  # noqa: BLE001 - retain this scope's failure as data
                        ledger = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
                if configured_inner_chunk is not None:
                    current_inner_chunk = int(model.core.receiver_chunk_size)
                    if current_inner_chunk != int(configured_inner_chunk):
                        raise RuntimeError(
                            f"inner receiver chunk changed during measurement for {spec.label}: "
                            f"{current_inner_chunk} != {configured_inner_chunk}"
                        )

                application_grid_query_count = int(np.asarray(sample["x_grid"]).size)
                if application_grid_query_count != int(query.shape[1]):
                    raise RuntimeError(
                        f"case {case_id} application grid has {application_grid_query_count} points "
                        f"but the matched direct query has {int(query.shape[1])}"
                    )
                payload["rows"].append(
                    {
                        "label": spec.label,
                        "checkpoint": str(Path(spec.path).resolve()),
                        "checkpoint_epoch": payload["checkpoints"][spec.label]["epoch"],
                        "architecture": architecture,
                        "case_id": case_id,
                        "dataset_path": str(dataset_path) if dataset_path is not None else None,
                        "direct_query_count": int(query.shape[1]),
                        "application_grid_query_count": application_grid_query_count,
                        "configured_inner_receiver_chunk_size": configured_inner_chunk,
                        "runtime_inner_receiver_chunk_override": None,
                        "application_query_batch_size": int(args.query_batch_size),
                        "parameter_count": parameter_info["parameter_count"],
                        "phases": {
                            "application_evaluator": app_measure,
                            "full_forward": full_measure,
                            "preparation_plus_one_query": prep_measure,
                            "prepared_p2_decode": decode_measure,
                        },
                        "untimed_execution_ledger": ledger,
                    }
                )
                if case_index % 10 == 0 or case_index == len(case_ids):
                    print(
                        f"{spec.label}: completed case {case_index}/{len(case_ids)} ({case_id})",
                        flush=True,
                    )
                del batch, kwargs, query, sample
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
        finally:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        # Persist a partial evidence file after each checkpoint to survive
        # external interruption without leaving an ambiguous half-run.
        partial_path = Path(args.output).expanduser().resolve()
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"finished checkpoint {spec_index}/{len(specs)}: {spec.label}", flush=True)

    payload["status"] = "complete"
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeat for each selected best-performance checkpoint",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-list", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--query-count", type=int, default=8192)
    parser.add_argument("--query-batch-size", type=int, default=32768)
    parser.add_argument("--mixed-teacher-ratio", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--device", default="cuda:2")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    _write_csv(csv_path, _measurements_to_rows(payload))
    print(json.dumps({"status": payload["status"], "output": str(output), "csv": str(csv_path)}))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
