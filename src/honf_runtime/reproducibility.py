"""Seed control and JSON-safe environment provenance."""

from __future__ import annotations

import platform
import random
import subprocess
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and every visible CUDA device."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def environment_snapshot() -> dict[str, Any]:
    """Return compact runtime provenance without host secrets or environment variables."""

    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": cuda_devices,
        "numpy": np.__version__,
        "h5py": _version("h5py"),
        "matplotlib": _version("matplotlib"),
        "honf_project": _version("honf-project"),
        "honf_case_thermalchannel": _version("honf-case-thermalchannel"),
    }


def source_state_snapshot(project_root: str | Path) -> dict[str, Any]:
    """Return commit and dirty-state provenance without storing patch contents."""

    root = Path(project_root).resolve()

    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    repository_root = git("rev-parse", "--show-toplevel")
    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "git_available": commit is not None,
        "repository_root": repository_root,
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "changed_path_count": None if status is None else len(status.splitlines()),
    }
