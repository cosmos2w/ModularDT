"""Batch launcher for local module thermal surrogate raw cases.

Scope
-----
This script handles **local module** batch generation for Demo 1. It reads
``Configs/config_local_module.json``, creates many seed-varied configs, and
launches ``simulate_local_module_thermal.py`` through a small CPU scheduler.

Inputs and outputs
------------------
Generated configs and logs live under ``Configs/Config_bk``. Raw local cases
are written under ``Data_Saved/LocalModule_Raw/case_*`` and are later packed by
``preprocess_local_module_dataset.py`` for future Stage-A local surrogate
training.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, TextIO

from tqdm.auto import tqdm

from channelthermal_common import (
    SimulationConfig,
    config_from_dict,
    dataclass_to_dict,
    default_config_backup_dir,
    default_config_dir,
    resolve_data_path,
    write_json,
)


# ------------------------------ Editable settings -------------------------------

NUM_LOCAL_CASES = 512
TEMPLATE_CONFIG_NAME = "config_local_module.json"
GENERATED_CONFIG_PREFIX = "config_local_module"
BASE_SEED = 5000
CPU_CONCURRENT_SLOTS = 8
POLL_INTERVAL_SEC = 0.5


@dataclass
class BatchJob:
    case_id: str
    seed: int
    config_path: Path
    config_arg: str
    log_path: Path


@dataclass
class RunningJob:
    job: BatchJob
    slot_name: str
    process: subprocess.Popen[str]
    log_handle: TextIO


def demo_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def generated_config_dir() -> Path:
    return default_config_backup_dir("Configs_local_module")


def generated_log_dir() -> Path:
    return default_config_backup_dir("logs_local_module")


def simulation_script_path() -> Path:
    return Path(__file__).resolve().parent / "simulate_local_module_thermal.py"


def load_template_config() -> SimulationConfig:
    template_path = default_config_dir() / TEMPLATE_CONFIG_NAME
    with template_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return config_from_dict(raw)


def find_next_case_number(root_dir: Path) -> int:
    if not root_dir.exists():
        return 1
    pattern = re.compile(r"^case_local_(\d+)_")
    max_case = 0
    for path in root_dir.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            max_case = max(max_case, int(match.group(1)))
    return max_case + 1


def build_case_id(case_number: int) -> str:
    return f"local_{case_number:04d}"


def build_job_config(template: SimulationConfig, case_id: str, seed: int) -> SimulationConfig:
    """Clone the template config and set the per-case seed/case ID."""
    cfg = config_from_dict(dataclass_to_dict(template))
    cfg.save.case_id = case_id
    cfg.layout.seed = int(seed)
    return cfg.finalize()


def create_jobs() -> List[BatchJob]:
    """Build all local jobs and write resolved per-case configs."""
    template = load_template_config()
    raw_root = resolve_data_path(template.save.root_dir)
    config_dir = generated_config_dir()
    log_dir = generated_log_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    next_case_number = find_next_case_number(raw_root)
    jobs: List[BatchJob] = []
    for idx in range(NUM_LOCAL_CASES):
        case_id = build_case_id(next_case_number + idx)
        seed = BASE_SEED + idx
        cfg = build_job_config(template, case_id, seed)
        config_name = f"{GENERATED_CONFIG_PREFIX}_{case_id}.json"
        config_path = config_dir / config_name
        write_json(config_path, dataclass_to_dict(cfg))
        config_arg = config_path.relative_to(default_config_dir()).as_posix()
        jobs.append(
            BatchJob(
                case_id=case_id,
                seed=seed,
                config_path=config_path,
                config_arg=config_arg,
                log_path=log_dir / f"case_{case_id}.log",
            )
        )
    return jobs


def build_command(job: BatchJob) -> List[str]:
    return [sys.executable, "-u", str(simulation_script_path()), "--config-json", job.config_arg]


def launch_job(job: BatchJob, slot_name: str) -> RunningJob:
    """Start one local simulator subprocess and attach a log file."""
    log_handle = job.log_path.open("w", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        build_command(job),
        cwd=demo_dir(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningJob(job=job, slot_name=slot_name, process=process, log_handle=log_handle)


def main() -> int:
    """Run local jobs across the configured CPU slots."""
    jobs = create_jobs()
    slots = [f"cpu:{idx}" for idx in range(CPU_CONCURRENT_SLOTS)]
    if not slots:
        raise ValueError("CPU_CONCURRENT_SLOTS must be positive.")

    pending: Deque[BatchJob] = deque(jobs)
    running: Dict[str, RunningJob] = {}
    failed_count = 0
    completed_count = 0
    tqdm.write(f"Prepared local module batch: cases={len(jobs)}, slots={len(slots)}")
    tqdm.write(f"Generated configs in: {generated_config_dir()}")
    tqdm.write(f"Logs will be written to: {generated_log_dir()}")

    with tqdm(total=len(jobs), desc="Queued -> running", unit="case", dynamic_ncols=True) as launch_bar:
        with tqdm(total=len(jobs), desc="Completed cases", unit="case", dynamic_ncols=True) as complete_bar:
            while pending or running:
                for slot_name in slots:
                    if not pending or slot_name in running:
                        continue
                    job = pending.popleft()
                    running[slot_name] = launch_job(job, slot_name)
                    launch_bar.update(1)
                    tqdm.write(f"Launched local case: case_id={job.case_id}, seed={job.seed}, slot={slot_name}")

                finished_slots: List[str] = []
                for slot_name, running_job in running.items():
                    return_code = running_job.process.poll()
                    if return_code is None:
                        continue
                    running_job.log_handle.close()
                    finished_slots.append(slot_name)
                    completed_count += 1
                    complete_bar.update(1)
                    if return_code == 0:
                        tqdm.write(f"Completed local case: case_id={running_job.job.case_id}")
                    else:
                        failed_count += 1
                        tqdm.write(
                            "Local case failed: "
                            f"case_id={running_job.job.case_id}, exit_code={return_code}, "
                            f"log={running_job.job.log_path}"
                        )
                for slot_name in finished_slots:
                    del running[slot_name]

                launch_bar.set_postfix(waiting=len(pending), running=len(running))
                complete_bar.set_postfix(failed=failed_count)
                if pending or running:
                    time.sleep(POLL_INTERVAL_SEC)

    if failed_count:
        tqdm.write(f"Local module batch finished with failures: failed={failed_count}, completed={completed_count}")
        return 1
    tqdm.write(f"Local module batch finished successfully: completed={completed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
