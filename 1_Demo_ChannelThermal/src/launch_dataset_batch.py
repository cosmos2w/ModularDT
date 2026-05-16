"""Batch launcher for global channel thermal raw-case generation.

Scope
-----
This script handles **global channel** batch generation. It reads
``Configs/config_channelthermal.json``, materializes many per-case configs, and
launches ``simulate_channelthermal.py`` in CPU/GPU slots.

Inputs and outputs
------------------
Generated configs and logs are written under ``Configs/Config_bk``. Raw cases
are written by the simulator under ``Data_Saved/case_*``. The resulting raw
cases should later be packed with ``preprocess_channelthermal_dataset.py`` for
future Stage-B global hypergraph-organizer training.

Edit the constants below to choose module counts, Reynolds numbers, repeats,
and CPU/GPU slots. Each job writes a materialized config into
``Configs/Config_bk/Configs_channelthermal`` and then launches
``simulate_channelthermal.py``.
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

import _bootstrap_imports  # noqa: F401
from channelthermal_common import (
    SimulationConfig,
    config_from_dict,
    dataclass_to_dict,
    default_config_backup_dir,
    default_config_dir,
    default_data_dir,
    materialize_layout,
    write_json,
)


# ------------------------------ Editable settings -------------------------------

# NUM_MODULE_OPTIONS = [1, 2, 4, 6, 9, 12]
# RE_OPTIONS = [30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 180.0, 200.0]

NUM_MODULE_OPTIONS = [3, 5, 7, 10]
RE_OPTIONS = [80.0, 100.0, 140]

HEAT_POWER_RANGE = (0.5, 2.0)
REPEATS_PER_COMBINATION = 5
LAYOUT_MODE_SEQUENCE = ["mixed", "tandem", "staggered", "random"]

ENABLE_CPU = False
CPU_CONCURRENT_SLOTS = 2
GPU_IDS: List[int] = [1]
MAX_CONCURRENT_PER_GPU = 8

TEMPLATE_CONFIG_NAME = "config_channelthermal.json"
GENERATED_CONFIG_PREFIX = "config_channelthermal"
BASE_LAYOUT_SEED = 1000
POLL_INTERVAL_SEC = 1.0


# ------------------------------- Data classes ----------------------------------


@dataclass
class BatchJob:
    case_id: str
    num_modules: int
    re_value: float
    replicate_idx: int
    layout_seed: int
    layout_mode: str
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
    return default_config_backup_dir("Configs_channelthermal")


def generated_log_dir() -> Path:
    return default_config_backup_dir("logs_channelthermal")


def simulation_script_path() -> Path:
    return Path(__file__).resolve().parent / "simulate_channelthermal.py"


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
    num_modules: int,
    re_value: float,
    layout_seed: int,
    layout_mode: str,
) -> SimulationConfig:
    """Clone the template config and materialize one randomized layout."""
    cfg = config_from_dict(dataclass_to_dict(template))
    cfg.layout.num_modules = int(num_modules)
    cfg.layout.seed = int(layout_seed)
    cfg.layout.layout_mode = str(layout_mode)
    cfg.layout.centers = None
    cfg.layout.heat_powers = None
    cfg.flow.re = float(re_value)
    cfg.thermal.heat_power_min = float(HEAT_POWER_RANGE[0])
    cfg.thermal.heat_power_max = float(HEAT_POWER_RANGE[1])
    cfg.save.case_id = case_id
    cfg.execution.device = "cpu"
    cfg.execution.gpu_id = 0
    return materialize_layout(cfg.finalize())


def create_jobs() -> List[BatchJob]:
    """Build all batch jobs and write their resolved JSON configs."""
    template = load_template_config()
    config_dir = generated_config_dir()
    log_dir = generated_log_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    next_case_number = find_next_case_number()
    jobs: List[BatchJob] = []
    layout_seed = BASE_LAYOUT_SEED
    for num_modules in NUM_MODULE_OPTIONS:
        for re_value in RE_OPTIONS:
            for replicate_idx in range(REPEATS_PER_COMBINATION):
                case_id = build_case_id(next_case_number)
                layout_mode = LAYOUT_MODE_SEQUENCE[replicate_idx % len(LAYOUT_MODE_SEQUENCE)]
                cfg = build_job_config(
                    template,
                    case_id=case_id,
                    num_modules=num_modules,
                    re_value=re_value,
                    layout_seed=layout_seed,
                    layout_mode=layout_mode,
                )
                config_name = f"{GENERATED_CONFIG_PREFIX}_{case_id}.json"
                config_path = config_dir / config_name
                write_json(config_path, dataclass_to_dict(cfg))
                config_arg = config_path.relative_to(default_config_dir()).as_posix()
                jobs.append(
                    BatchJob(
                        case_id=case_id,
                        num_modules=num_modules,
                        re_value=re_value,
                        replicate_idx=replicate_idx,
                        layout_seed=layout_seed,
                        layout_mode=layout_mode,
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
    """Expand editable CPU/GPU settings into concrete scheduler slots."""
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
    command = [
        sys.executable,
        "-u",
        str(simulation_script_path()),
        "--config-json",
        job.config_arg,
        "--device",
        slot.device,
    ]
    if slot.device == "gpu" and slot.gpu_id is not None:
        command.extend(["--gpu-id", str(slot.gpu_id)])
    return command


def launch_job(job: BatchJob, slot: DeviceSlot) -> RunningJob:
    """Start one simulator subprocess and stream stdout/stderr to a log file."""
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
    """Schedule jobs until every generated global case is complete."""
    jobs = create_jobs()
    slots = build_device_slots()
    pending: Deque[BatchJob] = deque(jobs)
    running: Dict[str, RunningJob] = {}
    completed_count = 0
    failed_count = 0

    tqdm.write(
        "Prepared channel thermal batch: "
        f"cases={len(jobs)}, slots={len(slots)}, cpu_slots={CPU_CONCURRENT_SLOTS if ENABLE_CPU else 0}, "
        f"gpu_ids={GPU_IDS}"
    )
    tqdm.write(f"Generated configs in: {generated_config_dir()}")
    tqdm.write(f"Logs will be written to: {generated_log_dir()}")

    with tqdm(total=len(jobs), desc="Queued -> running", unit="case", dynamic_ncols=True) as launch_bar:
        with tqdm(total=len(jobs), desc="Completed cases", unit="case", dynamic_ncols=True) as complete_bar:
            while pending or running:
                for slot in slots:
                    if not pending or slot.slot_name in running:
                        continue
                    job = pending.popleft()
                    running[slot.slot_name] = launch_job(job, slot)
                    launch_bar.update(1)
                    tqdm.write(
                        "Launched case: "
                        f"case_id={job.case_id}, N={job.num_modules}, Re={job.re_value}, "
                        f"layout={job.layout_mode}, seed={job.layout_seed}, slot={slot.slot_name}"
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
                        tqdm.write(f"Completed case: case_id={running_job.job.case_id}, slot={running_job.slot.slot_name}")
                    else:
                        failed_count += 1
                        tqdm.write(
                            "Case failed: "
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
        tqdm.write(f"Batch finished with failures: failed={failed_count}, completed={completed_count}")
        return 1
    tqdm.write(f"Batch finished successfully: completed={completed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
