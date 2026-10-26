"""Case-local storage and GPU ownership checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

DATA2 = Path("/data2")
OWNED_BASE = DATA2 / "wanglz/ModularDT/datasets/hetmf"
EXPECTED_MOUNT_UUID = "a47e4da3-3b56-40a7-baba-b675eeb0e999"
GPU2_UUID = "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39"


def require_data2(*, minimum_free_bytes: int = 50 * 2**30) -> dict:
    """Refuse a root-disk fallback and return measured filesystem capacity."""
    result = subprocess.run(
        ["findmnt", "-n", "-T", str(DATA2), "-o", "TARGET,SOURCE,FSTYPE,UUID"],
        check=True, capture_output=True, text=True,
    )
    fields = result.stdout.split()
    if len(fields) != 4 or fields[0] != str(DATA2) or fields[3] != EXPECTED_MOUNT_UUID:
        raise RuntimeError(f"/data2 mount mismatch: {result.stdout.strip()}")
    if fields[2] != "ext4":
        raise RuntimeError(f"unexpected /data2 filesystem: {fields[2]}")
    usage = shutil.disk_usage(DATA2)
    if usage.free < minimum_free_bytes:
        raise RuntimeError(f"/data2 headroom {usage.free} below {minimum_free_bytes}")
    return {"mount": fields, "total_bytes": usage.total, "free_bytes": usage.free}


def require_budget(*, retained_bytes: int, temporary_bytes: int,
                   minimum_free_bytes: int = 50 * 2**30) -> dict:
    if min(retained_bytes, temporary_bytes) < 0:
        raise ValueError("negative storage budget")
    capacity = require_data2(minimum_free_bytes=minimum_free_bytes)
    if capacity["free_bytes"] - retained_bytes - temporary_bytes < minimum_free_bytes:
        raise RuntimeError("storage budget would breach /data2 headroom")
    return {**capacity, "retained_budget_bytes": retained_bytes,
            "temporary_budget_bytes": temporary_bytes,
            "peak_new_bytes": retained_bytes + temporary_bytes,
            "minimum_free_bytes": minimum_free_bytes}


def require_owned_root(root: Path) -> Path:
    require_data2()
    root = Path(root).resolve(strict=False)
    if root == OWNED_BASE or OWNED_BASE not in root.parents:
        raise ValueError(f"payload root outside owned HET-MF subtree: {root}")
    return root


def gpu2_status() -> dict:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    )
    rows = [[part.strip() for part in line.split(",")] for line in query.stdout.splitlines()]
    gpu = next((row for row in rows if row[1] == GPU2_UUID), None)
    if gpu is None or gpu[0] != "2":
        raise RuntimeError("physical GPU2 UUID/index mismatch")
    jobs = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name",
         "--format=csv,noheader"], check=True, capture_output=True, text=True,
    )
    processes = [line.strip() for line in jobs.stdout.splitlines() if GPU2_UUID in line]
    return {"index": 2, "uuid": GPU2_UUID, "total_mib": int(gpu[2]),
            "free_mib": int(gpu[3]), "utilization_percent": int(gpu[4]),
            "compute_processes": processes}


def require_exclusive_gpu2() -> dict:
    status = gpu2_status()
    required_mib = max(8192, int(status["total_mib"] * 0.2))
    foreign_processes = [row for row in status["compute_processes"]
                         if row.split(",", 1)[0].strip() != str(os.getpid())]
    if foreign_processes or status["free_mib"] < required_mib:
        raise RuntimeError(f"GPU2 not exclusive/available: {status}")
    return status


def link_dataset(root: Path, case_dir: Path, *, replace_owned_links: bool = False) -> dict:
    """Expose the historical pilot through one unambiguous browsing link.

    Pilot subfolders stay beneath ``Dataset/pilot``. Legacy top-level aliases
    are retired separately after their readers and ownership are checked.
    """
    root = require_owned_root(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    case_dir = Path(case_dir).resolve(strict=True)
    if case_dir.name != "Case_ElectroThermal" or not (case_dir / "pyproject.toml").is_file():
        raise ValueError("case directory is not Case_ElectroThermal")
    dataset = case_dir / "Dataset"
    if not dataset.is_dir() or dataset.is_symlink():
        raise ValueError("Dataset must be the case-local real directory")
    ownership = root / "receipts" / "browsing_links.json"
    old = json.loads(ownership.read_text()) if ownership.is_file() else {}
    targets = {"pilot": root}
    made = {}
    for name, target in targets.items():
        dest = dataset / name
        if dest.parent != dataset:
            if dest.parent.exists() and (dest.parent.is_symlink() or not dest.parent.is_dir()):
                raise FileExistsError(f"refusing unsafe link parent: {dest.parent}")
            dest.parent.mkdir(exist_ok=True)
        if dest.is_symlink():
            current = os.readlink(dest)
            if current == str(target):
                made[name] = current
                continue
            if not replace_owned_links or old.get(str(dest)) != current:
                raise FileExistsError(f"refusing to replace unowned link: {dest}")
        elif dest.exists():
            raise FileExistsError(f"refusing to replace regular path: {dest}")
        temporary = dest.parent / f".{dest.name}.hetmf-link-{os.getpid()}"
        temporary.symlink_to(target, target_is_directory=target.is_dir())
        os.replace(temporary, dest)
        made[name] = str(target)
    ownership.parent.mkdir(parents=True, exist_ok=True)
    old.update({str(dataset / k): v for k, v in made.items()})
    ownership.write_text(json.dumps(old, indent=2) + "\n")
    return made


def link_training_artifacts(case_dir: Path, targets: dict[str, Path]) -> dict[str, str]:
    """Expose verified release/training payloads without replacing existing paths."""
    allowed = {"formal_standard", "formal_expanded", "standard.h5", "expanded.h5", "readiness"}
    if not targets or not set(targets) <= allowed:
        raise ValueError("unknown training browsing link")
    case_dir = Path(case_dir).resolve(strict=True)
    dataset = case_dir / "Dataset"
    if (case_dir.name != "Case_ElectroThermal" or not (case_dir / "pyproject.toml").is_file()
            or not dataset.is_dir() or dataset.is_symlink()):
        raise ValueError("expected the real Case_ElectroThermal/Dataset directory")
    checked = {}
    for name, candidate in targets.items():
        target = require_owned_root(candidate)
        if not target.exists():
            raise FileNotFoundError(target)
        if name.endswith(".h5") and not target.is_file():
            raise ValueError("training H5 target must be a file")
        if not name.endswith(".h5") and not target.is_dir():
            raise ValueError("release/diagnostic target must be a directory")
        destination = dataset / name
        if destination.is_symlink():
            if os.readlink(destination) != str(target):
                raise FileExistsError(f"refusing foreign browsing link: {destination}")
        elif destination.exists():
            raise FileExistsError(f"refusing regular browsing path: {destination}")
        checked[name] = target
    for name, target in checked.items():
        destination = dataset / name
        if not destination.is_symlink():
            destination.symlink_to(target, target_is_directory=target.is_dir())
    return {name: str(target) for name, target in checked.items()}


def link_shared_bundle(bundle: Path, case_dir: Path) -> dict[str, str]:
    """Link read-only published inputs; generation keeps its owned-root checks."""
    import h5py
    bundle = Path(bundle).expanduser().resolve(strict=True)
    case_dir = Path(case_dir).expanduser().resolve(strict=True)
    dataset = case_dir / 'Dataset'
    if (case_dir.name != 'Case_ElectroThermal' or not (case_dir / 'pyproject.toml').is_file()
            or dataset.is_symlink() or not dataset.is_dir()):
        raise ValueError('expected a real Case_ElectroThermal/Dataset directory')
    targets = {'standard.h5': bundle / 'standard.h5', 'expanded.h5': bundle / 'expanded.h5'}
    targets.update({name: bundle / 'evidence' / name
                    for name in ('readiness', 'formal_standard', 'formal_expanded', 'pilot')})
    for name, target in targets.items():
        if not target.exists():
            raise FileNotFoundError(target)
        if name.endswith('.h5'):
            with h5py.File(target, 'r') as handle:
                if handle.attrs.get('schema') != 'hetmf_training_h5_v1' or not handle.attrs.get('accepted'):
                    raise ValueError('published file is not an accepted training archive')
        elif not target.is_dir():
            raise ValueError('expected an evidence directory')
        dest = dataset / name
        if dest.is_symlink():
            if dest.resolve(strict=False) != target.resolve():
                raise FileExistsError(f'refusing foreign link: {dest}')
        elif dest.exists():
            raise FileExistsError(f'refusing regular destination: {dest}')
    for name, target in targets.items():
        destination = dataset / name
        if not destination.is_symlink():
            destination.symlink_to(target, target_is_directory=target.is_dir())
    return {name: str(target) for name, target in targets.items()}
