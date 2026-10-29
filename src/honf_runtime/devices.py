"""Case-neutral PyTorch device selection and validation."""

from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve a requested device and fail early when it is unavailable."""

    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device} was requested, but CUDA is unavailable.")
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; detected {torch.cuda.device_count()} device(s)."
            )
    return device
