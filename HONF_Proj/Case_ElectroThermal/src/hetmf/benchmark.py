"""Bounded timing-only replay over frozen pilot inputs; writes no duplicate fields."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .io import canonical_json, write_json_atomic
from .resources import require_owned_root
from .validation import validate_solution


def benchmark_batches(root: Path, plan: dict, *, device: str = "cpu",
                      batch_sizes: tuple[int, ...] = (1, 4, 16, 32),
                      repeats: int = 3) -> dict:
    root = require_owned_root(root)
    if plan["kind"] != "pilot" or repeats < 1 or repeats > 3:
        raise ValueError("only bounded pilot timing repeats are allowed")
    if any(b not in (1, 4, 16, 32) for b in batch_sizes):
        raise ValueError("unsupported batch size")
    if device != "cpu":
        raise ValueError("GPU timing requires separate exclusive-device orchestration")
    import torch
    from .gpu_physics import solve_gpu_batch
    torch.set_num_threads(2)
    lookup = {scene["state_id"]: scene for scene in plan["scenes"]}
    scenes = [lookup[sid] for sid in plan["throughput_ids"]]
    if len(scenes) != 256:
        raise ValueError("full 256-state throughput panel required")
    path = root / "diagnostics" / "cpu_batch_timing.jsonl"
    old = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    completed = {(row["batch_size"], row["repeat"], row["first_state_id"])
                 for row in old if row["status"] == "accepted"}
    records = list(old)
    for batch_size in batch_sizes:
        for repeat in range(repeats):
            for start in range(0, len(scenes), batch_size):
                batch = scenes[start:start + batch_size]
                key = (batch_size, repeat, batch[0]["state_id"])
                if key in completed:
                    continue
                started = time.perf_counter()
                try:
                    solutions = solve_gpu_batch(batch, 96, precision="fp64", device="cpu")
                    quality = [validate_solution(scene, solution, residual_limit=2e-10)
                               for scene, solution in zip(batch, solutions)]
                    if not all(q["accepted"] for q in quality):
                        raise RuntimeError(f"timing replay quality failure: {[q['failures'] for q in quality]}")
                    record = {"status": "accepted", "batch_size": batch_size,
                              "repeat": repeat, "first_state_id": batch[0]["state_id"],
                              "count": len(batch), "elapsed_seconds": time.perf_counter() - started,
                              "electrical_iterations": [s["diagnostics"]["electrical_solver"]["total_iterations"] for s in solutions],
                              "thermal_iterations": [s["diagnostics"]["thermal_solver"]["total_iterations"] for s in solutions],
                              "max_true_residual": max(max(s["diagnostics"][key]["true_relative_residual"]
                                                            for key in ("electrical_solver", "thermal_solver"))
                                                       for s in solutions)}
                except Exception as exc:
                    record = {"status": "failed", "batch_size": batch_size,
                              "repeat": repeat, "first_state_id": batch[0]["state_id"],
                              "count": len(batch), "elapsed_seconds": time.perf_counter() - started,
                              "error": repr(exc)}
                    with path.open("a") as handle:
                        handle.write(canonical_json(record) + "\n")
                    raise
                with path.open("a") as handle:
                    handle.write(canonical_json(record) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                records.append(record)
            print(f"CPU batch timing: B{batch_size} repeat {repeat+1}/{repeats} complete", flush=True)
    summary = {"device": "cpu", "threads": 2, "states_per_repeat": 256,
               "repeats": repeats, "batch_sizes": list(batch_sizes),
               "by_batch_size": {str(b): {"elapsed_seconds_total": sum(r["elapsed_seconds"] for r in records
                                                                    if r["status"] == "accepted" and r["batch_size"] == b),
                                          "case_seconds_mean": sum(r["elapsed_seconds"] for r in records
                                                                   if r["status"] == "accepted" and r["batch_size"] == b) / (256 * repeats),
                                          "accepted_evaluations": sum(r["count"] for r in records
                                                                      if r["status"] == "accepted" and r["batch_size"] == b)}
                                 for b in batch_sizes}}
    write_json_atomic(root / "diagnostics" / "cpu_batch_timing_summary.json", summary)
    return summary
