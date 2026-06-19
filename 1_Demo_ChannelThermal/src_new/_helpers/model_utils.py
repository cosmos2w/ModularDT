"""CHANNELTHERMAL-SPECIFIC utility helpers.

Inputs are ChannelThermal config paths, tensors, arrays, and checkpoint paths;
outputs include resolved paths, JSON payloads, normalized tensors, plotting
artifacts, and small neural-network helper modules. The path and normalization
conventions are specific to Demo 1 ChannelThermal, although some tensor helpers
are domain-agnostic.
"""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np
import torch
import torch.nn as nn


DEMO_ROOT = Path(__file__).resolve().parents[2]
EPS = 1.0e-6


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_demo_path(path_like: str | Path | None, *, default: str | Path | None = None) -> Path:
    if path_like is None:
        if default is None:
            raise ValueError("path_like and default cannot both be None.")
        path_like = default
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (DEMO_ROOT / path).resolve()


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
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
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def select_device(device_arg: Optional[str] = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataclass_from_dict(cls, payload: Optional[Mapping[str, Any]]):
    """Construct a dataclass from a config dict, ignoring unknown keys."""
    if payload is None:
        payload = {}
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type.")
    allowed = {item.name for item in fields(cls)}
    clean = {key: value for key, value in dict(payload).items() if key in allowed}
    return cls(**clean)


def dataclass_to_dict(instance) -> Dict[str, Any]:
    if not is_dataclass(instance):
        raise TypeError("dataclass_to_dict expects a dataclass instance.")
    return {item.name: getattr(instance, item.name) for item in fields(instance)}


def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def decode_string_array(values: Any) -> list[str]:
    arr = np.asarray(values)
    out: list[str] = []
    for item in arr.reshape(-1):
        out.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return out


def recursive_to_device(value: Any, device: torch.device) -> Any:
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
    std = np.asarray(std, dtype=np.float32)
    return np.where(np.abs(std) < eps, 1.0, std).astype(np.float32)


def safe_std_torch(std: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return torch.where(std.abs() < eps, torch.ones_like(std), std)


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, value.shape)
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    sq = (pred - target).square()
    if mask is None:
        return sq.mean()
    return masked_mean(sq, mask)


def masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    if mask is None:
        return torch.softmax(logits, dim=dim)
    mask = mask.to(device=logits.device, dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(EPS)


def make_grad_scaler(device: torch.device, enabled: bool):
    amp_enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def autocast_context(device: torch.device, enabled: bool):
    amp_enabled = bool(enabled and device.type == "cuda")
    if not amp_enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


class FourierEncoder(nn.Module):
    """Sin/cos Fourier features for low-dimensional coordinates."""

    def __init__(self, input_dim: int, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        freq_bands = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + 2 * self.input_dim * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = [x] if self.include_input else []
        for freq in self.freq_bands.to(device=x.device, dtype=x.dtype):
            angle = 2.0 * math.pi * freq * x
            pieces.append(torch.sin(angle))
            pieces.append(torch.cos(angle))
        return torch.cat(pieces, dim=-1)


class MLP(nn.Module):
    """Small configurable MLP used across the local and global models."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        act = nn.GELU if activation == "gelu" else nn.SiLU
        dims = [int(in_dim)] + [int(hidden_dim)] * max(int(num_layers) - 1, 0) + [int(out_dim)]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            is_last = idx == len(dims) - 2
            if not is_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[idx + 1]))
                layers.append(act())
                if dropout > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_loss_curve(csv_path: str | Path, png_path: str | Path, *, title: str = "Loss") -> None:
    """Render a simple loss curve from a CSV written by the training scripts."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    rows = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if rows.size == 0:
        return
    if rows.ndim == 0:
        rows = np.asarray([rows])
    names = rows.dtype.names or ()
    if "epoch" not in names:
        return
    plt.figure(figsize=(7.0, 4.0))
    for name in names:
        if name == "epoch":
            continue
        try:
            values = np.asarray(rows[name], dtype=float)
        except (TypeError, ValueError):
            continue
        if np.any(np.isfinite(values)):
            plt.plot(rows["epoch"], values, label=name)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=160)
    plt.close()
