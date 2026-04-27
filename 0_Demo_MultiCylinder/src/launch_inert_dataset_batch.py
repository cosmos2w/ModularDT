"""
Batch launcher for inert or active multi-cylinder dataset generation.

Defaults remain inert for backward compatibility. Set DATASET_MODE="active"
and TEMPLATE_CONFIG_NAME/GENERATED_CONFIG_PREFIX to the active names below to
sample active heated/cooled cases.
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
from typing import Deque, Dict, List, Optional, TextIO

from tqdm.auto import tqdm

from multicyl_common import (
    SimulationConfig,
    config_from_dict,
    dataclass_to_dict,
    default_config_backup_dir,
    default_config_backup_log_dir,
    default_config_dir,
    default_data_dir,
    materialize_layout,
    validate_mode_name,
    write_json,
)


# ------------------------------ Editable settings -------------------------------

# NUM_CYLINDER_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8]
# RE_OPTIONS = [20.0, 50.0, 100.0, 160.0, 200.0]

NUM_CYLINDER_OPTIONS = [2, 4, 6]
RE_OPTIONS = [30.0, 120.0, 170.0]
REPEATS_PER_COMBINATION = 6

DATASET_MODE = "inert"  # "inert" | "active"
TEMPLATE_CONFIG_NAME = "config_inert.json"  # for active: "config_active.json"
GENERATED_CONFIG_PREFIX = "config_inert"  # for active: "config_active"

BASE_LAYOUT_SEED = 1000

ENABLE_CPU = True
CPU_CONCURRENT_SLOTS = 2
GPU_IDS = [0, 1]
MAX_CONCURRENT_PER_GPU = 8

POLL_INTERVAL_SEC = 1.0


# ------------------------------- Data classes ----------------------------------


@dataclass
class BatchJob:
    case_id: str
    num_cylinders: int
    re_value: float
    replicate_idx: int
    layout_seed: int
    config_path: Path
    config_arg: str
    log_path: Path


@dataclass
class DeviceSlot:
    slot_name: str
    device: str
    gpu_id: Optional[int]


@dataclass
class RunningJob:
    job: BatchJob
    slot: DeviceSlot
    process: subprocess.Popen[str]
    log_handle: TextIO


# ------------------------------- Path helpers ----------------------------------


def demo_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def generated_config_dir() -> Path:
    return default_config_backup_dir(validate_mode_name(DATASET_MODE))


def generated_log_dir() -> Path:
    return default_config_backup_log_dir(validate_mode_name(DATASET_MODE))


def simulation_script_path() -> Path:
    return Path(__file__).resolve().parent / "simulate_multicylinder_phiflow.py"


# ------------------------------ Config builders --------------------------------


def load_template_config() -> SimulationConfig:
    template_path = default_config_dir() / TEMPLATE_CONFIG_NAME
    with template_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return config_from_dict(raw)


def find_next_case_number() -> int:
    data_root = default_data_dir().resolve()
    if not data_root.exists():
        return 1

    pattern = re.compile(r"^case_(\d+)_")
    max_case = 0
    for path in data_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            max_case = max(max_case, int(match.group(1)))
    return max_case + 1


def build_case_id(case_number: int) -> str:
    return f"{case_number:04d}"


def build_job_config(
    template: SimulationConfig,
    *,
    case_id: str,
    num_cylinders: int,
    re_value: float,
    layout_seed: int,
) -> SimulationConfig:
    cfg = config_from_dict(dataclass_to_dict(template))
    mode = validate_mode_name(DATASET_MODE)
    cfg.mode = mode
    cfg.layout.num_cylinders = num_cylinders
    cfg.layout.seed = layout_seed
    cfg.layout.centers = None
    cfg.layout.heat_powers = None
    cfg.flow.re = re_value
    cfg.save.case_id = case_id
    cfg.execution.device = "cpu"
    cfg.execution.gpu_id = 0
    return materialize_layout(cfg.finalize())


def create_jobs() -> List[BatchJob]:
    template = load_template_config()
    config_dir = generated_config_dir()
    log_dir = generated_log_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    next_case_number = find_next_case_number()
    jobs: List[BatchJob] = []
    layout_seed = BASE_LAYOUT_SEED

    for num_cylinders in NUM_CYLINDER_OPTIONS:
        for re_value in RE_OPTIONS:
            for replicate_idx in range(REPEATS_PER_COMBINATION):
                case_id = build_case_id(next_case_number)
                cfg = build_job_config(
                    template,
                    case_id=case_id,
                    num_cylinders=num_cylinders,
                    re_value=re_value,
                    layout_seed=layout_seed,
                )

                config_name = f"{GENERATED_CONFIG_PREFIX}_{case_id}.json"
                config_path = config_dir / config_name
                write_json(config_path, dataclass_to_dict(cfg))
                config_arg = config_path.relative_to(default_config_dir()).as_posix()

                jobs.append(
                    BatchJob(
                        case_id=case_id,
                        num_cylinders=num_cylinders,
                        re_value=re_value,
                        replicate_idx=replicate_idx,
                        layout_seed=layout_seed,
                        config_path=config_path,
                        config_arg=config_arg,
                        log_path=log_dir / f"case_{case_id}.log",
                    )
                )

                next_case_number += 1
                layout_seed += 1

    return jobs


# ------------------------------ Device scheduler -------------------------------


def build_device_slots() -> List[DeviceSlot]:
    slots: List[DeviceSlot] = []

    if ENABLE_CPU:
        for idx in range(CPU_CONCURRENT_SLOTS):
            slots.append(DeviceSlot(slot_name=f"cpu:{idx}", device="cpu", gpu_id=None))

    for gpu_id in GPU_IDS:
        for idx in range(MAX_CONCURRENT_PER_GPU):
            slots.append(DeviceSlot(slot_name=f"gpu:{gpu_id}:{idx}", device="gpu", gpu_id=gpu_id))

    if not slots:
        raise ValueError("No execution slots configured. Enable CPU slots or provide GPU_IDS.")
    return slots


def build_command(job: BatchJob, slot: DeviceSlot) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        str(simulation_script_path()),
        "--config-json",
        job.config_arg,
        "--device",
        slot.device,
    ]
    if slot.device == "gpu" and slot.gpu_id is not None:
        cmd.extend(["--gpu-id", str(slot.gpu_id)])
    return cmd


def launch_job(job: BatchJob, slot: DeviceSlot) -> RunningJob:
    command = build_command(job, slot)
    log_handle = job.log_path.open("w", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        command,
        cwd=demo_dir(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningJob(job=job, slot=slot, process=process, log_handle=log_handle)


# ------------------------------- Main routine ----------------------------------


def main() -> int:
    jobs = create_jobs()
    slots = build_device_slots()
    pending: Deque[BatchJob] = deque(jobs)
    running: Dict[str, RunningJob] = {}
    launched_count = 0
    completed_count = 0
    failed_count = 0

    tqdm.write(
        "Prepared batch launcher: "
        f"cases={len(jobs)}, slots={len(slots)}, cpu_slots={CPU_CONCURRENT_SLOTS if ENABLE_CPU else 0}, "
        f"gpu_ids={GPU_IDS}, max_concurrent_per_gpu={MAX_CONCURRENT_PER_GPU}"
    )
    tqdm.write(f"Generated configs in: {generated_config_dir()}")
    tqdm.write(f"Logs will be written to: {generated_log_dir()}")

    with tqdm(total=len(jobs), desc="Queued -> running", unit="case", dynamic_ncols=True) as launch_bar:
        with tqdm(total=len(jobs), desc="Completed cases", unit="case", dynamic_ncols=True) as complete_bar:
            with tqdm(total=len(slots), desc="Active slots", unit="slot", dynamic_ncols=True) as active_bar:
                while pending or running:
                    for slot in slots:
                        if not pending or slot.slot_name in running:
                            continue

                        job = pending.popleft()
                        running_job = launch_job(job, slot)
                        running[slot.slot_name] = running_job
                        launched_count += 1
                        launch_bar.update(1)
                        tqdm.write(
                            "Launched case: "
                            f"case_id={job.case_id}, Nc={job.num_cylinders}, Re={job.re_value}, "
                            f"seed={job.layout_seed}, slot={slot.slot_name}, log={job.log_path.name}"
                        )

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
                            tqdm.write(
                                "Completed case: "
                                f"case_id={running_job.job.case_id}, slot={running_job.slot.slot_name}"
                            )
                        else:
                            failed_count += 1
                            tqdm.write(
                                "Case failed: "
                                f"case_id={running_job.job.case_id}, slot={running_job.slot.slot_name}, "
                                f"exit_code={return_code}, log={running_job.job.log_path}"
                            )

                    for slot_name in finished_slots:
                        del running[slot_name]

                    active_bar.n = len(running)
                    active_bar.refresh()
                    launch_bar.set_postfix(waiting=len(pending))
                    complete_bar.set_postfix(running=len(running), failed=failed_count)

                    if pending or running:
                        time.sleep(POLL_INTERVAL_SEC)

    if failed_count:
        tqdm.write(f"Batch finished with failures: failed={failed_count}, completed={completed_count}")
        return 1

    tqdm.write(f"Batch finished successfully: completed={completed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
