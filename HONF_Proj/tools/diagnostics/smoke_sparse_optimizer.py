"""Bounded fresh-optimizer smoke for the exact parent and sparse candidate.

This read-only diagnostic loads one explicit Run 2001 checkpoint, creates a
fresh maintained optimizer for each model and module-width bucket, and runs
one warm update followed by three measured updates.  It never restores the
checkpoint optimizer state, creates a managed run, or saves model weights.
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "Case_ThermalChannel" / "src",
    PROJECT_ROOT / "tools",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import benchmark_routing_optimization as training
import run_dynamic_sparse_routing_study as study
from benchmark_sparse_execution import set_mode
from calibrate_sparse_paircost import science_model
from channelthermal.training.epoch import (
    assemble_channelthermal_loss_terms,
    effective_port_global_weight,
    make_model_inputs,
)
from channelthermal.training.optimizer import build_forward_optimizer

from honf_forward_core.config import ROUTING_TYPED_TEMPERATURE_NAMES
from honf_runtime.compat import recursive_to_device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite_gradients(model: torch.nn.Module) -> tuple[bool, int, float | None]:
    finite = True
    count = 0
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        count += 1
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all())
        squared_norm += float(gradient.double().square().sum())
    return finite, count, math.sqrt(squared_norm)


def _temperature_values(model: torch.nn.Module) -> dict[str, float]:
    parameters = getattr(getattr(model, "core", None), "routing_log_temperatures", {})
    return {
        name: float(parameters[name].detach().cpu())
        for name in ROUTING_TYPED_TEMPERATURE_NAMES
        if name in parameters
    }


def _run_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    inputs: dict[str, Any],
    schedule: tuple[dict[str, Any], dict[str, Any], str, float, float, float, float],
    device: torch.device,
) -> dict[str, Any]:
    training_config, loss_cfg, mode, ratio, internal_weight, interface_weight, predicted_weight = schedule
    optimizer.zero_grad(set_to_none=True)
    temperatures_before = _temperature_values(model)
    baseline_allocated = None
    if device.type == "cuda":
        baseline_allocated = float(torch.cuda.memory_allocated(device) / 2**20)
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    output = model(**inputs)
    terms = assemble_channelthermal_loss_terms(
        output,
        batch,
        model,
        loss_cfg,
        local_port_condition_mode=mode,
        mixed_teacher_ratio=ratio,
        effective_internal_temperature_weight=internal_weight,
        effective_interface_weight=interface_weight,
        predicted_consistency_weight=predicted_weight,
    )
    terms["loss"].backward()
    clip_norm = float(training_config.get("gradient_clip_norm", training_config.get("grad_clip_norm", 0.0)))
    clipped_norm_tensor = None
    if clip_norm > 0.0:
        clipped_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    gradients_finite, gradient_tensor_count, gradient_norm = _finite_gradients(model)
    clipped_norm = None if clipped_norm_tensor is None else float(clipped_norm_tensor)
    temperatures_after = _temperature_values(model)
    temperature_updates = {
        name: temperatures_after[name] - temperatures_before[name]
        for name in temperatures_after
    }
    backend = getattr(getattr(model, "core", None), "backend", None)
    return {
        "elapsed_seconds": float(elapsed),
        "loss": float(terms["loss"].detach()),
        "loss_physical": float(terms["loss_physical"].detach()),
        "loss_paircost": float(terms["loss_paircost"].detach()),
        "gradients_finite": bool(gradients_finite),
        "gradient_tensor_count": int(gradient_tensor_count),
        "gradient_norm_after_clip": float(gradient_norm),
        "clipped_gradient_norm": clipped_norm,
        "actual_qe_backend": getattr(backend, "last_qe_backend", None),
        "qe_backend_reason": getattr(backend, "last_qe_backend_reason", None),
        "temperatures_before": temperatures_before,
        "temperatures_after": temperatures_after,
        "temperature_updates": temperature_updates,
        "peak_allocated_mib": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda" else None
        ),
        "baseline_allocated_mib": baseline_allocated,
        "incremental_peak_allocated_mib": (
            None
            if baseline_allocated is None
            else (
                float(torch.cuda.max_memory_allocated(device) / 2**20)
                - baseline_allocated
            )
        ),
        "peak_reserved_mib": (
            float(torch.cuda.max_memory_reserved(device) / 2**20)
            if device.type == "cuda" else None
        ),
    }


def _fresh_models(
    source: torch.nn.Module,
    device: torch.device,
    backend: str,
    cost_weight: float,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    parent = copy.deepcopy(source).to(device)
    set_mode(parent, "compiled_exact", backend)
    candidate = science_model(parent, device, backend)
    sparsification = candidate.config.core_honf.interface_model.routing.sparsification
    sparsification.cost_weight = float(cost_weight)
    set_mode(candidate, "compiled_exact", backend)
    return parent, candidate


def _optimizer_metadata(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    config = model.config.core_honf.interface_model.routing.sparsification
    return {
        "optimizer": type(optimizer).__name__,
        "parameter_groups": len(optimizer.param_groups),
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        "weight_decays": [float(group["weight_decay"]) for group in optimizer.param_groups],
        "sparsification_enabled": bool(getattr(config, "enabled", False)),
        "sparsification_cost_weight": float(getattr(config, "cost_weight", 0.0)),
    }


def _run_bucket(
    source: torch.nn.Module,
    checkpoint: dict[str, Any],
    dataset: Any,
    bucket: Any,
    *,
    device: torch.device,
    backend: str,
    cost_weight: float,
    warmups: int,
    steps: int,
) -> dict[str, Any]:
    loader = training._build_loader(dataset, checkpoint, bucket, batch_size=len(bucket.case_ids))
    batch = recursive_to_device(next(iter(loader)), device)
    schedule = training._training_schedule(checkpoint)
    _, loss_cfg, mode, ratio, _, _, predicted_weight = schedule
    inputs = make_model_inputs(
        batch,
        local_port_condition_mode=mode,
        mixed_teacher_ratio=ratio,
        return_predicted_port_outputs=predicted_weight > 0.0,
        return_port_global_consistency=effective_port_global_weight(loss_cfg, mode, ratio) != 0.0,
    )
    parent, candidate = _fresh_models(source, device, backend, cost_weight)

    with torch.no_grad():
        parent.eval()
        candidate.eval()
        parent_field = parent(**inputs)["pred_field"]
        candidate_field = candidate(**inputs)["pred_field"]
        field_relative_error = float(
            (candidate_field - parent_field).norm() / parent_field.norm().clamp_min(1.0e-12)
        )
    del parent_field, candidate_field

    records: dict[str, Any] = {
        "bucket": bucket.label,
        "module_count": int(bucket.module_count),
        "case_ids": list(bucket.case_ids),
        "batch_size": len(bucket.case_ids),
        "queries": int(batch["query_xy"].shape[1]),
        "field_relative_error_before_updates": field_relative_error,
        "models": {},
    }
    for label, model in (("exact_parent", parent), ("sparse_candidate", candidate)):
        model.train()
        optimizer, _ = build_forward_optimizer(model, schedule[0])
        model_records: dict[str, Any] = {
            "optimizer": _optimizer_metadata(model, optimizer),
            "warmup_updates": [],
            "measured_updates": [],
        }
        for _ in range(int(warmups)):
            model_records["warmup_updates"].append(
                _run_update(model, optimizer, batch, inputs, schedule, device)
            )
        for _ in range(int(steps)):
            model_records["measured_updates"].append(
                _run_update(model, optimizer, batch, inputs, schedule, device)
            )
        records["models"][label] = model_records
        del optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del parent, candidate, batch, inputs, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("torch", "triton"), default="triton")
    parser.add_argument("--cost-weight", type=float, default=0.0024756277369438542)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--query-points", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.query_points <= 0 or args.warmups < 0 or args.steps <= 0:
        raise ValueError("Batch, query, warmup, and step counts must be valid positive values.")

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the requested optimizer smoke.")
    spec = study.parse_checkpoint_specs([args.checkpoint])[0]
    source, checkpoint = study._load_model_spec(spec, device)
    dataset, dataset_path = study._load_dataset(
        checkpoint,
        SimpleNamespace(dataset=None, split="train"),
        points_per_case_override=int(args.query_points),
        random_point_sampling_override=False,
    )
    dataset.include_grid = False
    buckets = training._make_buckets([dataset], batch_size=int(args.batch_size), requested="both")
    rows = []
    try:
        for bucket in buckets:
            rows.append(
                _run_bucket(
                    source,
                    checkpoint,
                    dataset,
                    bucket,
                    device=device,
                    backend=args.backend,
                    cost_weight=args.cost_weight,
                    warmups=args.warmups,
                    steps=args.steps,
                )
            )
    finally:
        dataset.close()
        del source
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "status": "complete",
        "task": "fresh_sparse_optimizer_smoke",
        "checkpoint": str(spec.path),
        "parent_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "backend": args.backend,
        "execution": "compiled_exact",
        "dataset_path": str(dataset_path),
        "protocol": {
            "batch_size": int(args.batch_size),
            "query_points": int(args.query_points),
            "bucket_policy": "both",
            "warmups": int(args.warmups),
            "measured_updates": int(args.steps),
            "optimizer_state": "fresh for both models; checkpoint optimizer state never restored",
            "comparison": (
                "same physical batches, maintained loss assembly, backend, execution, "
                "and optimizer hyperparameters"
            ),
            "candidate_objective": "physical objective plus one induced-pair-cost term",
            "cost_weight": float(args.cost_weight),
        },
        "rows": rows,
    }
    study.write_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
