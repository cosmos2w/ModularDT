"""Measure Run-1409 full-width and packed execution on explicit anchors."""

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
    from channelthermal.evaluation.prepared import select_sample

    return torch, run1405, stage3, make_batch, select_sample


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


def _set_mode(model: Any, mode: str) -> None:
    setter = getattr(model.core.backend, "set_execution_mode", None)
    if not callable(setter):
        raise TypeError("occupancy backend does not expose callable set_execution_mode")
    setter(mode)


def _forward(model: Any, batch: Mapping[str, Any], kwargs: Mapping[str, Any], *, maps: bool) -> Any:
    return model(
        batch["structure"],
        batch["query_xy"],
        return_prepared_state=maps,
        return_routing_maps=maps,
        **kwargs,
    )


def _measure(torch: Any, device: Any, function: Any, warmups: int, repetitions: int) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmups):
            output = function()
            del output
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_allocated = baseline_reserved = None
        samples: list[float] = []
        for _ in range(repetitions):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                output = function()
                stop.record()
                stop.synchronize()
                samples.append(float(start.elapsed_time(stop)))
            else:
                started = time.perf_counter()
                output = function()
                samples.append(float((time.perf_counter() - started) * 1000.0))
            del output
        if device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = peak_reserved = None
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "milliseconds": samples,
        "median_ms": float(np.median(samples)),
        "mean_ms": float(np.mean(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "incremental_peak_allocated_bytes": None if peak_allocated is None else peak_allocated - baseline_allocated,
        "incremental_peak_reserved_bytes": None if peak_reserved is None else peak_reserved - baseline_reserved,
    }


def _ledger(output: Mapping[str, Any]) -> dict[str, Any]:
    aux = output.get("interaction_aux", {})
    return {
        "kplan": aux.get("occupancy_group_k_plan_per_case"),
        "module_logical_paths": aux.get("group_control_module_logical_paths"),
        "module_unique_pairs": aux.get("group_control_module_unique_pairs"),
        "module_actual_rows": aux.get("group_control_module_fine_rows_forward"),
        "module_padded_rows": aux.get("group_control_module_fine_rows_padded"),
        "environment_logical_paths": aux.get("group_control_environment_logical_paths"),
        "environment_unique_pairs": aux.get("group_control_environment_unique_pairs"),
        "environment_actual_rows": aux.get("group_control_environment_fine_rows_forward"),
        "environment_padded_rows": aux.get("group_control_environment_fine_rows_padded"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    torch, run1405, stage3, make_batch, select_sample = _imports()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, checkpoint_payload = stage3._load_model_spec(
        stage3.CheckpointSpec(label="1409-occupancy", path=checkpoint), device
    )
    if str(model.config.core_honf.forward_architecture) != "occupancy_adaptive_group_control_honf":
        raise RuntimeError("checkpoint is not occupancy_adaptive_group_control_honf")
    dataset, dataset_path = stage3._load_dataset(
        checkpoint_payload, argparse.Namespace(dataset=args.dataset, split=args.split)
    )
    rows: list[dict[str, Any]] = []
    try:
        for case_id in args.case_id:
            sample = select_sample(dataset, str(case_id), 0)
            query = _query_points(sample, int(args.query_count))
            batch = make_batch(dict(sample), query, device)
            kwargs = run1405._phase_forward_kwargs(batch)
            outputs: dict[str, Any] = {}
            timings: dict[str, Any] = {}
            model.eval()
            for mode in ("full_width", "packed"):
                _set_mode(model, mode)
                with torch.inference_mode():
                    outputs[mode] = _forward(model, batch, kwargs, maps=True)
                timings[mode] = _measure(
                    torch,
                    device,
                    lambda current_batch=batch, current_kwargs=kwargs: _forward(
                        model, current_batch, current_kwargs, maps=False
                    ),
                    int(args.warmups),
                    int(args.repetitions),
                )
            _set_mode(model, "full_width")
            with torch.inference_mode():
                deterministic_repeat = _forward(model, batch, kwargs, maps=True)
            delta = (
                outputs["full_width"]["pred_field"].detach().float()
                - outputs["packed"]["pred_field"].detach().float()
            ).abs()
            repeat_delta = (
                outputs["full_width"]["pred_field"].detach().float()
                - deterministic_repeat["pred_field"].detach().float()
            ).abs()
            original_kplan = outputs["full_width"]["interaction_aux"][
                "occupancy_group_k_plan_per_case"
            ]
            repeat_kplan = deterministic_repeat["interaction_aux"][
                "occupancy_group_k_plan_per_case"
            ]
            full_peak = timings["full_width"]["incremental_peak_allocated_bytes"]
            packed_peak = timings["packed"]["incremental_peak_allocated_bytes"]
            rows.append(
                {
                    "case_id": str(case_id),
                    "query_count": int(args.query_count),
                    "prediction_max_abs_difference": float(delta.max().cpu()),
                    "prediction_mean_abs_difference": float(delta.mean().cpu()),
                    "deterministic_repeat_prediction_max_abs_difference": float(
                        repeat_delta.max().cpu()
                    ),
                    "deterministic_repeat_kplan_equal": bool(
                        torch.equal(original_kplan, repeat_kplan)
                    ),
                    "full_width": {"timing": timings["full_width"], "ledger": _ledger(outputs["full_width"])},
                    "packed": {"timing": timings["packed"], "ledger": _ledger(outputs["packed"])},
                    "packed_over_full_median_latency": timings["packed"]["median_ms"] / timings["full_width"]["median_ms"],
                    "packed_over_full_incremental_peak_allocated": None if full_peak is None or packed_peak is None else packed_peak / max(full_peak, 1),
                }
            )
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    return {
        "schema_version": 1,
        "task": "run1409_occupancy_full_packed_benchmark",
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "dataset": str(dataset_path),
        "device": str(device),
        "rows": rows,
        "stochastic_gate_estimator_present": False,
        "stochastic_deterministic_disagreement": "not_applicable_no_stochastic_gate",
        "decision_rule": "enable packed execution only if it is measurably faster while preserving the operator",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=["0273", "0653"])
    parser.add_argument("--query-count", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
