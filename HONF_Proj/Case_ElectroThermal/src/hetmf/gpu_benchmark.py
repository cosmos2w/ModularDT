"""Bounded GPU2 pilot replay without writing duplicate physical fields."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from .io import canonical_json, write_json_atomic
from .resources import GPU2_UUID, gpu2_status, require_owned_root
from .validation import validate_solution


def _append(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _bind_gpu2() -> dict:
    if "torch" in sys.modules:
        raise RuntimeError("GPU2 visibility must be bound before torch import")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, GPU2_UUID):
        raise RuntimeError("CUDA_VISIBLE_DEVICES exposes a different device")
    before = gpu2_status()
    if before["free_mib"] < 16 * 1024:
        raise RuntimeError("GPU2 has less than 16 GiB free; declining concurrent pilot")
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU2_UUID
    return before


def _reference_errors(root: Path, scene: dict, solution: dict) -> dict:
    with h5py.File(root / "raw" / f"{scene['state_id']}_n96.h5") as handle:
        pairs = (("potential", "v"), ("temperature_rise", "theta"),
                 ("q_sheet", "q_sheet"), ("q_contact", "q_contact"),
                 ("terminal_current", "terminal_currents"))
        result = {}
        for key, solved in pairs:
            reference = handle[key][:].astype(np.float64)
            candidate = np.asarray(solution[solved], dtype=np.float64)
            result[key + "_relative_l2"] = float(
                np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1e-14))
    return result


def run_gpu_pilot(root: Path, plan: dict, *, repeats: int = 3,
                  memory_fraction: float = 0.25) -> dict:
    """Qualify FP64/FP32/mixed, then time full B32 and smaller stratified batches.

    The work is capped at 12 qualification solves plus 1,056 timing solves when
    repeats is three. A pre-existing GPU2 user is allowed only because the
    caller has explicitly authorized concurrent pilot use.
    """
    root = require_owned_root(root)
    if plan.get("kind") != "pilot" or repeats != 3 or len(plan["throughput_ids"]) != 256:
        raise ValueError("frozen 256-state pilot and three repeats required")
    if not 0 < memory_fraction <= 0.25:
        raise ValueError("GPU2 pilot memory fraction must be at most 25%")
    before = _bind_gpu2()
    import torch
    from .gpu_physics import BatchSolveError, solve_gpu_batch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("bound physical GPU2 is not the sole visible CUDA device")
    lookup = {scene["state_id"]: scene for scene in plan["scenes"]}
    representative = [lookup[sid] for sid in ("R00", "R05", "R07", "R11")]
    journal = root / "diagnostics" / "gpu2_pilot_timing.jsonl"
    old = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    completed = {(row["phase"], row["precision"], row["batch_size"], row["repeat"],
                  row["first_state_id"]) for row in old if row["status"] == "accepted"}
    for precision in ("fp64", "fp32", "mixed"):
        key = ("qualification", precision, 4, 0, representative[0]["state_id"])
        if key in completed:
            continue
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            solutions = solve_gpu_batch(representative, 96, precision=precision,
                                        device="cuda:0")
            limit = 2e-10 if precision == "fp64" else 2e-6
            quality = [validate_solution(scene, result, residual_limit=limit)
                       for scene, result in zip(representative, solutions)]
            if not all(row["accepted"] for row in quality):
                raise RuntimeError(f"physical qualification failed: {[q['failures'] for q in quality]}")
            errors = [_reference_errors(root, scene, result)
                      for scene, result in zip(representative, solutions)]
            row = {"phase": "qualification", "precision": precision, "batch_size": 4,
                   "repeat": 0, "first_state_id": representative[0]["state_id"],
                   "count": 4, "status": "accepted", "elapsed_seconds": time.perf_counter() - started,
                   "quality": quality, "reference_errors": errors,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        except (BatchSolveError, RuntimeError) as exc:
            row = {"phase": "qualification", "precision": precision, "batch_size": 4,
                   "repeat": 0, "first_state_id": representative[0]["state_id"],
                   "count": 4, "status": "failed", "elapsed_seconds": time.perf_counter() - started,
                   "error": repr(exc)}
        _append(journal, row)
        print(f"GPU2 qualification {precision}: {row['status']}", flush=True)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    if not any(row["phase"] == "qualification" and row["precision"] == "fp64"
               and row["status"] == "accepted" for row in records):
        raise RuntimeError("GPU2 FP64 did not qualify; throughput replay refused")
    full = [lookup[sid] for sid in plan["throughput_ids"]]
    subset = full[::8]
    for batch_size, scenes in ((1, subset), (4, subset), (16, subset), (32, full)):
        for repeat in range(repeats):
            for start in range(0, len(scenes), batch_size):
                batch = scenes[start:start + batch_size]
                key = ("timing", "fp64", batch_size, repeat, batch[0]["state_id"])
                if key in completed:
                    continue
                status = gpu2_status()
                if status["free_mib"] < 8 * 1024:
                    raise RuntimeError("GPU2 free memory fell below 8 GiB at batch boundary")
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                try:
                    solutions = solve_gpu_batch(batch, 96, precision="fp64", device="cuda:0")
                    quality = [validate_solution(scene, result, residual_limit=2e-10)
                               for scene, result in zip(batch, solutions)]
                    if not all(q["accepted"] for q in quality):
                        raise RuntimeError(f"physical quality failure: {[q['failures'] for q in quality]}")
                    row = {"phase": "timing", "precision": "fp64", "batch_size": batch_size,
                           "repeat": repeat, "first_state_id": batch[0]["state_id"],
                           "count": len(batch), "status": "accepted",
                           "elapsed_seconds": time.perf_counter() - started,
                           "electrical_iterations": [s["diagnostics"]["electrical_solver"]["total_iterations"] for s in solutions],
                           "thermal_iterations": [s["diagnostics"]["thermal_solver"]["total_iterations"] for s in solutions],
                           "max_true_residual": max(max(s["diagnostics"][name]["true_relative_residual"]
                                                         for name in ("electrical_solver", "thermal_solver"))
                                                    for s in solutions),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                           "gpu2_free_mib_before_batch": status["free_mib"],
                           "gpu2_other_processes_before_batch":
                               [p for p in status["compute_processes"]
                                if p.split(",", 1)[0].strip() != str(os.getpid())]}
                except (BatchSolveError, RuntimeError) as exc:
                    row = {"phase": "timing", "precision": "fp64", "batch_size": batch_size,
                           "repeat": repeat, "first_state_id": batch[0]["state_id"],
                           "count": len(batch), "status": "failed",
                           "elapsed_seconds": time.perf_counter() - started, "error": repr(exc)}
                    _append(journal, row)
                    raise
                _append(journal, row)
            print(f"GPU2 B{batch_size} repeat {repeat+1}/{repeats} complete", flush=True)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    results = {}
    for batch_size, expected in ((1, 32), (4, 32), (16, 32), (32, 256)):
        rows = [r for r in records if r["phase"] == "timing" and r["batch_size"] == batch_size
                and r["status"] == "accepted"]
        count = sum(r["count"] for r in rows)
        if count != expected * repeats:
            raise RuntimeError(f"GPU2 B{batch_size} panel incomplete: {count}")
        seconds = sum(r["elapsed_seconds"] for r in rows)
        results[str(batch_size)] = {"accepted_evaluations": count,
                                    "complete_seconds_total": seconds,
                                    "complete_seconds_per_case": seconds / count,
                                    "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
                                    "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rows),
                                    "electrical_iterations_p50": float(np.median([i for r in rows for i in r["electrical_iterations"]])),
                                    "electrical_iterations_p90": float(np.quantile([i for r in rows for i in r["electrical_iterations"]], 0.9)),
                                    "thermal_iterations_p50": float(np.median([i for r in rows for i in r["thermal_iterations"]])),
                                    "thermal_iterations_p90": float(np.quantile([i for r in rows for i in r["thermal_iterations"]], 0.9))}
    summary = {"device_uuid": GPU2_UUID, "timing_condition": "concurrent_gpu2_job_authorized; shared-device wall time",
               "gpu2_before": before, "gpu2_after": gpu2_status(), "cpu_threads": 2,
               "memory_fraction_limit": memory_fraction, "precision_selected": "fp64",
               "qualification": [r for r in records if r["phase"] == "qualification"],
               "by_batch_size": results,
               "total_coupled_evaluations": sum(r["count"] for r in records),
               "duplicate_field_files_written": 0}
    write_json_atomic(root / "diagnostics" / "gpu2_pilot_timing_summary.json", summary)
    return summary
