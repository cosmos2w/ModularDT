from __future__ import annotations

from typing import Dict, List

import numpy as np


CHANNELS = ("u", "v", "p", "omega")


def _series(values: np.ndarray) -> List[float]:
    return [float(x) for x in np.asarray(values).reshape(-1)]


def compute_kpis(field_cycle: np.ndarray) -> Dict:
    arr = np.asarray(field_cycle, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[-1] != 4:
        raise ValueError("field_cycle must have shape [T, H, W, 4].")
    u = arr[..., 0]
    v = arr[..., 1]
    p = arr[..., 2]
    omega = arr[..., 3]

    field_mean = {}
    field_max_abs = {}
    for idx, name in enumerate(CHANNELS):
        field_mean[name] = _series(arr[..., idx].mean(axis=(1, 2)))
        field_max_abs[name] = _series(np.abs(arr[..., idx]).max(axis=(1, 2)))

    kpis = {
        "mean_abs_omega": _series(np.abs(omega).mean(axis=(1, 2))),
        "enstrophy": _series(np.square(omega).mean(axis=(1, 2))),
        "max_abs_omega": _series(np.abs(omega).max(axis=(1, 2))),
        "kinetic_energy": _series((np.square(u) + np.square(v)).mean(axis=(1, 2))),
        "pressure_range": _series(p.max(axis=(1, 2)) - p.min(axis=(1, 2))),
        "field_mean": field_mean,
        "field_max_abs": field_max_abs,
    }
    for name in CHANNELS:
        kpis[f"field_mean_{name}"] = field_mean[name]
        kpis[f"field_max_abs_{name}"] = field_max_abs[name]
    return kpis
