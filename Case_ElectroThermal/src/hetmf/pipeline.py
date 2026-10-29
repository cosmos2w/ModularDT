"""Bounded pilot execution and hash-based resume."""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from .io import (canonical_json, sha256_json, validate_ledger_case,
                 validate_raw_case, write_case_atomic, write_json_atomic)
from .physics import solve_cpu
from .resources import GPU2_UUID, require_budget, require_exclusive_gpu2, require_owned_root
from .validation import audit_persisted, validate_solution


@contextmanager
def single_writer(root: Path):
    path = root / "receipts" / "writer.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another HET-MF writer owns the pilot root") from exc
        yield


def freeze_plan(root: Path, plan: dict, config: dict) -> dict:
    root = require_owned_root(root)
    path = root / "inputs" / f"{plan['kind']}_plan.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != plan:
            raise RuntimeError("frozen input plan differs from requested plan")
    else:
        write_json_atomic(path, plan)
    config_hash = sha256_json(config)
    plan_hash = sha256_json(plan)
    receipt = {"plan_sha256": plan_hash, "config_sha256": config_hash,
               "physical_states": plan["counts"].get("distinct_states", plan["counts"].get("total_states"))}
    write_json_atomic(root / "receipts" / f"{plan['kind']}_freeze.json", receipt)
    return receipt


def _solver_hash(config: dict, grid: int, backend: str) -> str:
    return sha256_json({"physics": config["physics"], "environment": config["environment"],
                        "contacts": config["contacts"], "solver": config["solver"],
                        "grid": [grid, grid], "backend": backend})


def _stage_cases(plan: dict, stage: str) -> list[tuple[dict, int]]:
    lookup = {scene["state_id"]: scene for scene in plan["scenes"]}
    if stage == "representatives":
        return [(lookup[sid], 96) for sid in plan["representative_ids"]]
    if stage == "qualify":
        return [(lookup[sid], grid) for grid in (48, 192)
                for sid in plan["refinement_ids"]]
    if stage == "qualify384":
        return [(lookup[sid], 384) for sid in ("R05", "R07", "R10", "R11")]
    if stage == "interactions":
        ids = [sid for quartet in plan["quartets"]
               for sid in quartet["endpoint_state_ids"] if sid not in plan["representative_ids"]]
        return [(lookup[sid], 96) for sid in ids]
    if stage == "interaction_refine":
        ids = [sid for quartet in (plan["quartets"][0], plan["quartets"][5])
               for sid in quartet["endpoint_state_ids"]]
        return [(lookup[sid], 192) for sid in ids]
    if stage == "throughput":
        return [(lookup[sid], 96) for sid in plan["throughput_ids"]]
    raise ValueError(f"unknown pilot stage {stage}")


def _append(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_pilot_stage(root: Path, plan: dict, config: dict, stage: str, *,
                    backend: str = "cpu", batch_size: int = 1) -> dict:
    root = require_owned_root(root)
    if plan["kind"] != "pilot" or plan.get("formal_generation_authorized"):
        raise ValueError("pilot command cannot process a formal plan")
    if backend not in ("cpu", "gpu_fp64"):
        raise ValueError("unsupported backend")
    if backend == "gpu_fp64":
        require_exclusive_gpu2()
        if "torch" in __import__("sys").modules:
            raise RuntimeError("torch was imported before GPU2 visibility was bound")
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU2_UUID
    reservation = require_budget(
        retained_bytes=int(config["resources"]["retained_budget_gib"] * 2**30),
        temporary_bytes=int(config["resources"]["temporary_budget_gib"] * 2**30),
        minimum_free_bytes=int(config["resources"]["minimum_free_disk_gib"] * 2**30),
    )
    cases = _stage_cases(plan, stage)
    if len(plan["scenes"]) > config["pilot"]["max_distinct_states"]:
        raise RuntimeError("pilot physical-state ceiling exceeded")
    accepted, reused, failed = 0, 0, 0
    from .io import initialize_root

    initialize_root(root, {key: reservation[key] for key in (
        "retained_budget_bytes", "temporary_budget_bytes", "peak_new_bytes",
        "minimum_free_bytes",
    )})
    with single_writer(root):
        for index in range(0, len(cases), batch_size):
            if (root / "receipts" / "STOP").exists():
                break
            batch = cases[index:index + batch_size]
            pending = []
            for scene, grid in batch:
                solver_hash = _solver_hash(config, grid, backend)
                target = root / "raw" / f"{scene['state_id']}_n{grid}.h5"
                if target.exists():
                    validate_raw_case(target, scene, solver_hash)
                    validate_ledger_case(root, target)
                    if not audit_persisted(root, scene, grid)["accepted"]:
                        raise RuntimeError(f"reused case failed post-storage audit: {target}")
                    reused += 1
                else:
                    pending.append((scene, grid, solver_hash))
            batch_started = time.perf_counter()
            if pending and backend == "gpu_fp64":
                # Recheck ownership at each batch boundary. No other process is signalled.
                require_exclusive_gpu2()
                from .gpu_physics import solve_gpu_batch
                if len({grid for _, grid, _ in pending}) != 1:
                    raise RuntimeError("GPU batch must have one native grid")
                solutions = solve_gpu_batch([scene for scene, _, _ in pending], pending[0][1],
                                            precision="fp64", device="cuda:0")
            else:
                solutions = [solve_cpu(scene, grid) for scene, grid, _ in pending]
            saved_records = []
            for (scene, grid, solver_hash), solution in zip(pending, solutions):
                audit = validate_solution(scene, solution,
                                          residual_limit=config["solver"]["reference_true_rtol"] * 2)
                record = {"stage": stage, "state_id": scene["state_id"], "grid": grid,
                          "backend": backend, "input_sha256": sha256_json(scene),
                          "solver_sha256": solver_hash, "quality": audit,
                          "solve_seconds": solution["diagnostics"]["elapsed_s"]}
                if not audit["accepted"]:
                    failed += 1
                    record["status"] = "failed_qualification"
                    _append(root / "diagnostics" / "attempts.jsonl", record)
                    continue
                written = write_case_atomic(root, scene, solution, grid=grid, solver_hash=solver_hash)
                storage = audit_persisted(root, scene, grid)
                record["post_storage"] = storage
                record["raw_bytes"] = written["raw_bytes"]
                record["status"] = "accepted" if storage["accepted"] else "failed_post_storage"
                if not storage["accepted"]:
                    _append(root / "diagnostics" / "attempts.jsonl", record)
                    failed += 1
                    raise RuntimeError(f"post-storage audit failed for {scene['state_id']} n{grid}")
                saved_records.append(record)
                accepted += 1
            batch_seconds = time.perf_counter() - batch_started
            for record in saved_records:
                record["complete_seconds_amortized"] = batch_seconds / len(pending)
                record["complete_batch_seconds"] = batch_seconds
                _append(root / "diagnostics" / "attempts.jsonl", record)
            print(f"{stage}: {min(index + batch_size, len(cases))}/{len(cases)} batch complete, "
                  f"new={accepted} reused={reused} failed={failed}", flush=True)
    summary = {"stage": stage, "requested": len(cases), "accepted_new": accepted,
               "reused": reused, "failed": failed, "reservation_free_bytes": reservation["free_bytes"],
               "complete": accepted + reused == len(cases)}
    write_json_atomic(root / "diagnostics" / f"stage_{stage}_{backend}.json", summary)
    return summary
