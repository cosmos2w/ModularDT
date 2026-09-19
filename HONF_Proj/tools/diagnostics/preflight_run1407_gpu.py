#!/usr/bin/env python3
# ruff: noqa: I001
"""One real Run-1407 predicted-port update and bounded executor calibration."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
for path in (PROJECT / "src", PROJECT / "Case_ThermalChannel" / "src", PROJECT / "tools" / "diagnostics"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import benchmark_routing_optimization as training_benchmark
from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.training.epoch import make_model_inputs
from channelthermal.training.optimizer import build_forward_optimizer
from channelthermal.workflows.train_forward import (
    build_model_config,
    resolve_auto_internal_mode,
)
from honf_runtime.compat import recursive_to_device, set_seed
from honf_runtime.config_loader import load_config_bundle


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _time(callable_: Any, device: torch.device, *, warmups: int = 2, repetitions: int = 5) -> list[float]:
    for _ in range(warmups):
        callable_()
    _sync(device)
    values: list[float] = []
    for _ in range(repetitions):
        _sync(device)
        started = time.perf_counter()
        callable_()
        _sync(device)
        values.append(1000.0 * (time.perf_counter() - started))
    return values


def _finite_gradients(model: torch.nn.Module) -> tuple[bool, int, float]:
    finite = True
    count = 0
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all())
        count += 1
        squared += float(gradient.double().square().sum())
    return finite, count, math.sqrt(squared)


def _dataset(payload: dict[str, Any], dataset_path: Path, *, query_count: int) -> GlobalChannelThermalDataset:
    dataset_cfg = dict(payload.get("dataset", {}))
    return GlobalChannelThermalDataset(
        str(dataset_path),
        split=str(dataset_cfg.get("train_split", "train")),
        points_per_case=int(query_count),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        seed=int(payload.get("training", {}).get("seed", 0)),
        require_converged=bool(dataset_cfg.get("require_converged", False)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--query-count", type=int, default=1024)
    parser.add_argument("--module-count", type=int, default=12)
    args = parser.parse_args()

    bundle = load_config_bundle(str(args.config.resolve()), overrides={"device": args.device})
    payload = dict(bundle.effective)
    training_cfg = dict(payload.get("training", {}))
    set_seed(int(training_cfg.get("seed", 0)))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = _dataset(payload, args.dataset.resolve(), query_count=int(args.query_count))
    try:
        candidates = sorted(
            str(case_id)
            for case_id, count in zip(
                dataset.selected_case_ids,
                dataset.selected_module_counts,
                strict=True,
            )
            if int(count) == int(args.module_count)
        )
        if not candidates:
            raise ValueError(f"No real training case has exactly M={args.module_count}.")
        case_ids = tuple(candidates[index % len(candidates)] for index in range(int(args.batch_size)))
        bucket = training_benchmark.Bucket(f"M{args.module_count}", int(args.module_count), case_ids)
        checkpoint_like = {"train_config": payload, "epoch": 1}
        loader = training_benchmark._build_loader(
            dataset,
            checkpoint_like,
            bucket,
            batch_size=int(args.batch_size),
        )
        shape = training_benchmark._validate_batch_shape(loader, device, int(args.query_count))
        model_config = build_model_config(payload, dataset)
        model = ChannelThermalHONFModel(model_config).to(device)
        model.set_global_target_normalization(
            dataset.normalizer.stats,
            normalize_targets=bool(payload.get("dataset", {}).get("normalize_targets", False)),
        )
        resolve_auto_internal_mode(model_config, model)
        if model.config.core_honf.forward_architecture != "phase_shared_group_control_honf":
            raise ValueError("Preflight requires phase_shared_group_control_honf.")

        batch = recursive_to_device(next(iter(loader)), device)
        inputs = make_model_inputs(
            batch,
            local_port_condition_mode="predicted",
            mixed_teacher_ratio=0.0,
            return_predicted_port_outputs=True,
            return_port_global_consistency=True,
        )
        model.eval()
        with torch.no_grad():
            output = model(
                **inputs,
                return_routing_maps=True,
                return_prepared_state=True,
            )
        prepared_wrapper = output["prepared_state"]
        prepared = getattr(prepared_wrapper, "prepared", prepared_wrapper)
        query_xy = batch["query_xy"]
        backend = model.core.backend
        original_policy = backend.executor_policy
        original_fraction = backend.module_selected_max_fraction

        def decode() -> dict[str, Any]:
            return model.core.decode_queries(
                prepared,
                query_xy,
                return_routing_maps=True,
                receiver_chunk_size=128,
            )

        with torch.no_grad():
            backend.executor_policy = "rectangular_reference"
            rectangular = decode()
            rectangular_ms = _time(decode, device)
            backend.executor_policy = "hybrid_support"
            backend.module_selected_max_fraction = original_fraction
            hybrid = decode()
            hybrid_ms = _time(decode, device)
            backend.module_selected_max_fraction = 1.0
            selected = decode()
            selected_ms = _time(decode, device)
        backend.executor_policy = original_policy
        backend.module_selected_max_fraction = original_fraction

        optimizer, inventory = build_forward_optimizer(model, training_cfg)
        optimizer_summary = {
            "mode": inventory.get("mode"),
            "weight_decay": inventory.get("weight_decay"),
            "groups": [
                {
                    "name": group.get("name"),
                    "learning_rate": group.get("learning_rate"),
                    "parameter_tensor_count": group.get("parameter_tensor_count"),
                    "trainable_scalar_count": group.get("trainable_scalar_count"),
                }
                for group in inventory.get("groups", [])
            ],
        }
        schedule = training_benchmark._training_schedule(checkpoint_like)
        if schedule[2] != "predicted":
            raise ValueError(f"Run-1407 preflight expected predicted ports, got {schedule[2]!r}.")
        tracked = model.core.backend.router.group_codes
        tracked_before = tracked.detach().clone()
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
        else:
            baseline_allocated = baseline_reserved = 0
        started = time.perf_counter()
        metrics = training_benchmark._run_step(
            model,
            loader,
            device,
            checkpoint_like,
            optimizer=optimizer,
            training_schedule=schedule,
        )
        _sync(device)
        update_seconds = time.perf_counter() - started
        gradients_finite, gradient_count, gradient_norm = _finite_gradients(model)
        tracked_delta = float((tracked.detach() - tracked_before).abs().max())
        result = {
            "schema_version": 1,
            "task": "run1407_real_predicted_port_gpu_preflight",
            "config": str(args.config.resolve()),
            "dataset": str(args.dataset.resolve()),
            "device_argument": str(args.device),
            "shape": shape,
            "bucket": {"label": bucket.label, "case_ids": list(bucket.case_ids)},
            "port_condition_mode": schedule[2],
            "optimizer_inventory": optimizer_summary,
            "update": {
                "elapsed_seconds": update_seconds,
                "metrics": metrics,
                "gradient_tensor_count": gradient_count,
                "gradient_norm": gradient_norm,
                "gradients_finite": gradients_finite,
                "group_code_max_update": tracked_delta,
                "parameters_finite_after": all(
                    bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
                ),
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
                ),
                "peak_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
                ),
            },
            "executor_calibration": {
                "receiver_chunk_size": 128,
                "rectangular_ms": rectangular_ms,
                "hybrid_ms": hybrid_ms,
                "forced_selected_ms": selected_ms,
                "hybrid_vs_rectangular_max_abs": float(
                    (hybrid["pred_field"] - rectangular["pred_field"]).abs().max()
                ),
                "selected_vs_rectangular_max_abs": float(
                    (selected["pred_field"] - rectangular["pred_field"]).abs().max()
                ),
                "hybrid_module_selected": _jsonable(
                    hybrid.get("group_control_module_executor_selected")
                ),
                "forced_module_selected": _jsonable(
                    selected.get("group_control_module_executor_selected")
                ),
                "support_rows": _jsonable(selected.get("group_control_module_support_rows")),
                "executed_rows": _jsonable(selected.get("group_control_module_fine_rows")),
                "padded_rows": _jsonable(selected.get("group_control_module_padded_rows")),
            },
        }
        if not gradients_finite or not result["update"]["parameters_finite_after"]:
            raise RuntimeError("Run-1407 preflight produced non-finite gradients or parameters.")
        if tracked_delta <= 0.0:
            raise RuntimeError("Run-1407 group prototypes did not update in the real predicted-port step.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    finally:
        dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
