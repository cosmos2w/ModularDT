"""Controlled, synchronized timing primitives for disposable study execution."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(call: Callable[[], Any], device: torch.device, *, warmups: int = 3,
            repeats: int = 10) -> dict[str, Any]:
    """Time actual computations, releasing outputs between repetitions."""
    for _ in range(warmups):
        result = call()
        synchronize(device)
        del result
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        result = call()
        synchronize(device)
        durations.append(time.perf_counter() - start)
        del result
    return {
        "warmups": warmups,
        "repeats": repeats,
        "seconds": durations,
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }


def disposable_update(model: Any, batch: Any, *, learning_rate: float = 3e-4,
                      weight_decay: float = 1e-5) -> dict[str, Any]:
    """Apply one full update to a disposable loaded model and measure it.

    Caller must discard this model afterwards; this never saves a checkpoint.
    Materialization is required before calling, and logical B/Q stay unchanged.
    """
    device = batch.module_centers.device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    parameter = next(p for p in model.parameters() if p.requires_grad)
    before = parameter.detach().clone()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch)["pred_field"]
    loss = (prediction - batch.target_field).square().mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "seconds": elapsed,
        "loss": float(loss.detach()),
        "preclip_gradient_norm": float(norm),
        "sampled_parameter_update_norm": float(torch.linalg.vector_norm(parameter.detach() - before)),
        "logical_batch_size": batch.module_present.shape[0],
        "queries_per_case": batch.query_xy.shape[1],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }
