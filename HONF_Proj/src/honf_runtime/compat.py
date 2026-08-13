"""Compatibility helpers retained while proven workflows are decomposed.

New orchestration code should prefer the focused modules in ``honf_runtime``.
This module keeps stable JSON, tensor, AMP, checkpoint, and small-network
helpers used by the migrated ThermalChannel implementation. Relative paths are
anchored at ``HONF_Proj`` rather than the old Demo-1 source location.
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np
import torch
import torch.nn as nn

from honf_forward_core.nn import FourierFeatures, MLP

from .devices import resolve_device
from .reproducibility import seed_all


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Backward-compatible name used by copied workflow code.  It now points to the
# new project root, not 1_Demo_ChannelThermal.
DEMO_ROOT = PROJECT_ROOT
EPS = 1.0e-6


def ensure_dir(path: str | Path) -> Path:
    """Ensure dir."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_demo_path(path_like: str | Path | None, *, default: str | Path | None = None) -> Path:
    """Resolve demo path."""

    if path_like is None:
        if default is None:
            raise ValueError("path_like and default cannot both be None.")
        path_like = default
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (DEMO_ROOT / path).resolve()


def read_json(path: str | Path) -> Dict[str, Any]:
    """Read json."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write json."""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_trusted_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a local training checkpoint that may contain config/stat metadata.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    Demo checkpoints are written by the local training scripts and intentionally
    include dictionaries plus NumPy normalization stats, so trusted local loads
    must opt back into full checkpoint unpickling.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def current_timestamp() -> str:
    """Perform the current timestamp operation used by this module."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    """Set seed."""

    seed_all(seed)


def select_device(device_arg: Optional[str] = None) -> torch.device:
    """Select device."""

    return resolve_device(device_arg)


def dataclass_from_dict(cls, payload: Optional[Mapping[str, Any]]):
    """Construct a dataclass and reject unknown non-documentation settings."""
    if payload is None:
        payload = {}
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type.")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(key for key in payload if key not in allowed and not str(key).startswith("_"))
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {unknown}")
    clean = {key: value for key, value in dict(payload).items() if key in allowed}
    return cls(**clean)


def dataclass_to_dict(instance) -> Dict[str, Any]:
    """Perform the dataclass to dict operation used by this module."""

    if not is_dataclass(instance):
        raise TypeError("dataclass_to_dict expects a dataclass instance.")
    return {item.name: getattr(instance, item.name) for item in fields(instance)}


def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Perform the deep update operation used by this module."""

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def decode_string_array(values: Any) -> list[str]:
    """Decode string array."""

    arr = np.asarray(values)
    out: list[str] = []
    for item in arr.reshape(-1):
        out.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return out


def recursive_to_device(value: Any, device: torch.device) -> Any:
    """Perform the recursive to device operation used by this module."""

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: recursive_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_device(item, device) for item in value)
    return value


def count_parameters(model: nn.Module) -> int:
    """Count parameters."""

    total = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        try:
            total += param.numel()
        except ValueError:
            # LazyLinear parameters are initialized by the first real batch.
            continue
    return total


def safe_std_np(std: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Perform the safe std np operation used by this module."""

    std = np.asarray(std, dtype=np.float32)
    return np.where(np.abs(std) < eps, 1.0, std).astype(np.float32)


def safe_std_torch(std: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Perform the safe std torch operation used by this module."""

    return torch.where(std.abs() < eps, torch.ones_like(std), std)


def make_grad_scaler(device: torch.device, enabled: bool):
    """Create grad scaler."""

    amp_enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def autocast_context(device: torch.device, enabled: bool):
    """Perform the autocast context operation used by this module."""

    amp_enabled = bool(enabled and device.type == "cuda")
    if not amp_enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Perform the strip module prefix operation used by this module."""

    return {key.removeprefix("module."): value for key, value in state_dict.items()}


class FourierEncoder(nn.Module):
    """Sin/cos Fourier features for low-dimensional coordinates."""

    def __init__(self, input_dim: int, num_frequencies: int, include_input: bool = True):
        """Initialize FourierEncoder and its required state."""

        super().__init__()
        self.encoder = FourierFeatures(
            int(input_dim),
            int(num_frequencies),
            include_input=bool(include_input),
            angular_scale=2.0 * math.pi,
            ordering="interleaved",
        )

    @property
    def output_dim(self) -> int:
        """Perform the output dim operation used by this module."""

        return self.encoder.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the FourierEncoder tensor transformation described above."""

        return self.encoder(x)
