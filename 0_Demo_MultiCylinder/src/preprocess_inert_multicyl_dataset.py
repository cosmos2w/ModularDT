"""Preprocess inert multi-cylinder PhiFlow cases into a canonical neural-field dataset.

This script converts raw saved case directories under ``Data_Saved/`` into a
compact, training-ready representation tailored to forward models of the form:

    Structure -> Organized interaction state -> Behavior manifold
             -> Phase-conditioned neural field reconstruction

The preprocessing pipeline is intentionally modular so it can be extended later
to active thermal cases or richer descriptors without changing the overall
layout.

Main outputs
------------
For each processed case, the script writes a dedicated output directory
containing:

* ``structure.json``: case-level geometric and flow metadata
* ``behavior_summary.npz``: mean field, RMS field, and scalar descriptors
* ``canonical_cycle.npz``: canonical periodic attractor over phase bins
* ``phase_metadata.csv``: frame-wise phase estimates and probe signal
* ``sampled_points.npz``: point-sampled neural-field training tuples
* optional quicklook plots

After all cases are processed, the script also writes:

* ``global_case_index.csv``: summary table for all processed cases
* ``packed_dataset.h5``: one packed HDF5 file containing all processed cases

Usage example
-------------
Run from ``0_Demo_MultiCylinder/``:

python src/preprocess_inert_multicyl_dataset.py \
    --input-root ./Data_Saved \
    --output-root ./Data_Saved/Processed_Inert_Dataset \
    --device cuda:0 \
    --phase-bins 36 \
    --save-cycles 1 \
    --points-per-phase-bin 0 \
    --sampling-mode uniform \
    --compute-pod \
    --quicklook

Notes
-----
* ``points-per-phase-bin = 0`` means "keep all grid points" for each phase bin.
* If ``input-root`` contains ``train/`` and ``test/`` subdirectories, those
  splits are preserved automatically in the output directory structure and the
  HDF5 metadata.
* This script uses ``h5py`` for the final packed HDF5 dataset because HDF5 is
  the target output format and it is the standard lightweight tool for that job.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import hilbert

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "The preprocessing script requires 'h5py' to build the packed HDF5 dataset. "
        "Install it with: pip install h5py"
    ) from exc

import matplotlib.pyplot as plt
from tqdm import tqdm

from multicyl_common import build_uniform_grid, config_from_dict, periodic_offsets, resolve_data_path


# ------------------------------- Data classes ----------------------------------


FIELD_CHANNEL_ORDER = ("u", "v", "p", "omega")


@dataclass
class CaseRecord:
    """A discovered raw case directory and its dataset split label."""

    split: str
    case_dir: Path


@dataclass
class ProbeInfo:
    """Metadata for the automatically selected phase probe."""

    cylinder_index: int
    cylinder_center: Tuple[float, float]
    probe_xy: Tuple[float, float]
    probe_ij: Tuple[int, int]


@dataclass
class CaseData:
    """All in-memory information needed to preprocess one case."""

    record: CaseRecord
    cfg: object
    frame_index: pd.DataFrame
    times: np.ndarray
    frame_ids: np.ndarray
    tensor: torch.Tensor  # [T, H, W, C] on the selected torch device
    cylinder_mask: Optional[np.ndarray]  # [H, W]
    available_fields: Dict[str, bool]
    x_grid: np.ndarray  # [H, W]
    y_grid: np.ndarray  # [H, W]


# ------------------------------ CLI definition ---------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess inert multi-cylinder PhiFlow cases into a canonical dataset.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("./Data_Saved"),
        help="Root directory containing raw case folders, optionally with train/ and test/ subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./Data_Saved/Processed_Inert_Dataset"),
        help="Root directory where processed per-case outputs and the packed HDF5 file will be written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for heavy numerical preprocessing, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--phase-bins",
        type=int,
        default=24,
        help="Number of bins in the canonical periodic attractor representation.",
    )
    parser.add_argument(
        "--canonical-cycle-method",
        choices=["contiguous", "legacy_tau_sort", "time_full_span"],
        default="contiguous",
        help="Method used to build canonical_cycle. 'contiguous' preserves raw-time continuity.",
    )
    parser.add_argument(
        "--min-cycle-frames",
        type=int,
        default=8,
        help="Minimum number of raw frames preferred for a contiguous canonical cycle window.",
    )
    parser.add_argument(
        "--prefer-late-cycle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer later candidate cycle windows when scores are similar.",
    )
    parser.add_argument(
        "--max-allowed-tau-sort-time-jump-factor",
        type=float,
        default=4.0,
        help="Warn when tau sorting would create adjacent time jumps larger than this factor times median dt.",
    )
    parser.add_argument(
        "--canonical-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save canonical-cycle diagnostics and metadata for auditing preprocessing quality.",
    )
    parser.add_argument(
        "--save-cycles",
        type=int,
        default=1,
        help="Number of times to repeat the canonical cycle in the final saved tensors.",
    )
    parser.add_argument(
        "--points-per-phase-bin",
        type=int,
        default=0,
        help="Number of sampled points per phase bin. Use 0 to keep all grid points.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["uniform", "annulus", "wake", "mixed"],
        default="uniform",
        help="Spatial point-sampling strategy for neural-field supervision tuples.",
    )
    parser.add_argument(
        "--annulus-ratio",
        type=float,
        default=0.4,
        help="Fraction of samples drawn from near-cylinder annuli when sampling mode is annulus or mixed.",
    )
    parser.add_argument(
        "--wake-ratio",
        type=float,
        default=0.4,
        help="Fraction of samples drawn from downstream wake windows when sampling mode is wake or mixed.",
    )
    parser.add_argument(
        "--annulus-outer-radius-factor",
        type=float,
        default=2.0,
        help="Outer radius of the near-cylinder annulus, expressed as a multiple of cylinder radius.",
    )
    parser.add_argument(
        "--wake-length-diameters",
        type=float,
        default=6.0,
        help="Length of downstream wake windows, expressed in cylinder diameters.",
    )
    parser.add_argument(
        "--wake-half-width-diameters",
        type=float,
        default=1.5,
        help="Half-width of downstream wake windows, expressed in cylinder diameters.",
    )
    parser.add_argument(
        "--save-full-canonical-cycles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save the full canonical cycle tensor to per-case outputs and HDF5.",
    )
    parser.add_argument(
        "--compute-pod",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute POD descriptors using the Method of Snapshots.",
    )
    parser.add_argument(
        "--pod-rank",
        type=int,
        default=5,
        help="Number of leading POD modes / coefficients to retain when --compute-pod is enabled.",
    )
    parser.add_argument(
        "--quicklook",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save quicklook figures for mean field, RMS field, and phase signal.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove existing packed HDF5 entries for matching case ids when re-running preprocessing.",
    )
    return parser.parse_args()


# ----------------------------- Case discovery ----------------------------------


def discover_case_directories(input_root: Path) -> List[CaseRecord]:
    """Find case directories under the input root, preserving train/test splits."""
    input_root = input_root.expanduser().resolve()
    discovered: List[CaseRecord] = []

    split_candidates = []
    for split_name in ("train", "test"):
        split_dir = input_root / split_name
        if split_dir.exists() and split_dir.is_dir():
            split_candidates.append((split_name, split_dir))

    if split_candidates:
        for split_name, split_dir in split_candidates:
            for case_dir in sorted(split_dir.iterdir()):
                if case_dir.is_dir() and (case_dir / "case_config.json").exists():
                    discovered.append(CaseRecord(split=split_name, case_dir=case_dir))
        return discovered

    for case_dir in sorted(input_root.iterdir()):
        if case_dir.is_dir() and (case_dir / "case_config.json").exists():
            discovered.append(CaseRecord(split="all", case_dir=case_dir))
    return discovered


# ----------------------------- Field loading -----------------------------------


def candidate_field_names(name: str) -> List[str]:
    """Try several capitalization patterns because legacy cases may differ."""
    return [name, name.lower(), name.capitalize(), name.upper()]


def find_field_path(scene_dir: Path, stem: str, frame_id: int) -> Optional[Path]:
    for candidate in candidate_field_names(stem):
        path = scene_dir / f"{candidate}_{frame_id:06d}.npz"
        if path.exists():
            return path
    return None


def load_npz_array(path: Path) -> np.ndarray:
    with np.load(path) as data:
        key = list(data.keys())[-1]
        return np.asarray(data[key])


def normalize_scalar_field(arr: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Normalize saved scalar fields into [H, W] = [ny, nx]."""
    if arr.shape == (ny, nx):
        return arr.astype(np.float32, copy=False)
    if arr.shape == (nx, ny):
        return arr.T.astype(np.float32, copy=False)
    if arr.shape == (ny + 1, nx + 1):
        return arr[:-1, :-1].astype(np.float32, copy=False)
    if arr.shape == (nx + 1, ny + 1):
        return arr[:-1, :-1].T.astype(np.float32, copy=False)
    raise ValueError(
        f"Unsupported scalar field shape {arr.shape}. Expected {(ny, nx)}, {(nx, ny)}, "
        f"{(ny + 1, nx + 1)}, or {(nx + 1, ny + 1)}."
    )


def normalize_vector_field(arr: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Normalize saved vector fields into [H, W, 2] = [ny, nx, 2]."""
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"Unsupported vector field shape {arr.shape}. Expected a trailing vector dimension of size 2.")
    if arr.shape[:2] == (ny, nx):
        return arr.astype(np.float32, copy=False)
    if arr.shape[:2] == (nx, ny):
        return np.transpose(arr, (1, 0, 2)).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported vector field shape {arr.shape}. Expected {(ny, nx, 2)} or {(nx, ny, 2)}.")


def load_case(record: CaseRecord, device: torch.device) -> Optional[CaseData]:
    """Load one case directory into a consistent tensor format."""
    case_dir = record.case_dir
    try:
        with (case_dir / "case_config.json").open("r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        cfg = config_from_dict(raw_cfg)
    except FileNotFoundError:
        print(f"[WARN] Missing case_config.json for {case_dir.name}; skipping case.")
        return None
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Failed to read case_config.json for {case_dir.name}: {exc}")
        return None

    frame_index_path = case_dir / "frame_index.csv"
    if not frame_index_path.exists():
        print(f"[WARN] Missing frame_index.csv for {case_dir.name}; skipping case.")
        return None

    frame_index = pd.read_csv(frame_index_path)
    if frame_index.empty:
        print(f"[WARN] Empty frame_index.csv for {case_dir.name}; skipping case.")
        return None

    frame_ids = frame_index["saved_frame"].to_numpy(dtype=int)
    times = frame_index["time"].to_numpy(dtype=np.float64)
    scene_dir = case_dir / "scene"

    nx = int(cfg.domain.nx)
    ny = int(cfg.domain.ny)
    tensors: List[np.ndarray] = []
    available_fields = {"velocity": True, "pressure": False, "vorticity": False, "cylindermask": False}
    pressure_missing_warned = False
    vorticity_missing_warned = False

    for frame_id in frame_ids:
        vel_path = find_field_path(scene_dir, "velocity", frame_id)
        if vel_path is None:
            print(f"[WARN] Missing velocity field for {case_dir.name} frame {frame_id:06d}; skipping case.")
            return None
        velocity = normalize_vector_field(load_npz_array(vel_path), nx=nx, ny=ny)

        pressure_path = find_field_path(scene_dir, "pressure", frame_id)
        if pressure_path is not None:
            pressure = normalize_scalar_field(load_npz_array(pressure_path), nx=nx, ny=ny)
            available_fields["pressure"] = True
        else:
            pressure = np.zeros((ny, nx), dtype=np.float32)
            if not pressure_missing_warned:
                print(f"[WARN] Pressure missing in {case_dir.name}; filling pressure with zeros.")
                pressure_missing_warned = True

        vorticity_path = find_field_path(scene_dir, "vorticity", frame_id)
        if vorticity_path is not None:
            vorticity = normalize_scalar_field(load_npz_array(vorticity_path), nx=nx, ny=ny)
            available_fields["vorticity"] = True
        else:
            vorticity = np.zeros((ny, nx), dtype=np.float32)
            if not vorticity_missing_warned:
                print(f"[WARN] Vorticity missing in {case_dir.name}; filling vorticity with zeros.")
                vorticity_missing_warned = True

        frame_tensor = np.concatenate(
            [
                velocity,
                pressure[..., None],
                vorticity[..., None],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        tensors.append(frame_tensor)

    cylinder_mask = None
    mask_path = find_field_path(scene_dir, "cylindermask", int(frame_ids[0]))
    if mask_path is not None:
        cylinder_mask = normalize_scalar_field(load_npz_array(mask_path), nx=nx, ny=ny)
        available_fields["cylindermask"] = True

    x_grid, y_grid = build_uniform_grid(cfg)
    return CaseData(
        record=record,
        cfg=cfg,
        frame_index=frame_index,
        times=times,
        frame_ids=frame_ids,
        tensor=torch.from_numpy(np.stack(tensors, axis=0)).to(device=device, dtype=torch.float32),
        cylinder_mask=cylinder_mask,
        available_fields=available_fields,
        x_grid=x_grid.astype(np.float32, copy=False),
        y_grid=y_grid.astype(np.float32, copy=False),
    )


# --------------------------- Phase estimation ----------------------------------


def cylinder_diameter(cfg: object) -> float:
    return 2.0 * float(cfg.domain.cylinder_radius)


def choose_phase_probe(cfg: object, x_grid: np.ndarray, y_grid: np.ndarray) -> ProbeInfo:
    """Choose a probe 2 diameters downstream of the right-most cylinder."""
    centers = np.asarray(cfg.layout.centers or [], dtype=np.float32)
    if centers.size == 0:
        raise ValueError("No cylinder centers found in configuration.")

    cylinder_index = int(np.argmax(centers[:, 0]))
    cx, cy = centers[cylinder_index]
    downstream_dx = 2.0 * cylinder_diameter(cfg)
    probe_x = float((cx + downstream_dx) % cfg.domain.lx)
    probe_y = float(cy % cfg.domain.ly)

    distance2 = (periodic_offsets(x_grid - probe_x, cfg.domain.lx) ** 2) + (periodic_offsets(y_grid - probe_y, cfg.domain.ly) ** 2)
    iy, ix = np.unravel_index(int(np.argmin(distance2)), distance2.shape)

    return ProbeInfo(
        cylinder_index=cylinder_index,
        cylinder_center=(float(cx), float(cy)),
        probe_xy=(probe_x, probe_y),
        probe_ij=(int(iy), int(ix)),
    )


def estimate_frequency_fft(signal_values: np.ndarray, times: np.ndarray) -> float:
    """Estimate a dominant frequency using a simple FFT peak as a robust fallback."""
    centered = signal_values - np.mean(signal_values)
    if centered.size < 4:
        return float("nan")
    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")

    fft_vals = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=dt)
    if freqs.size <= 1:
        return float("nan")

    power = np.abs(fft_vals) ** 2
    power[0] = 0.0
    idx = int(np.argmax(power))
    return float(freqs[idx]) if power[idx] > 0 else float("nan")


def estimate_phase_zero_crossing(signal_values: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fallback phase estimator based on upward zero crossings."""
    centered = signal_values - np.mean(signal_values)
    crossings: List[float] = []
    for i in range(centered.size - 1):
        a = centered[i]
        b = centered[i + 1]
        if a < 0.0 <= b:
            denom = b - a
            alpha = 0.0 if abs(denom) < 1e-12 else (-a) / denom
            crossing_time = float(times[i] + alpha * (times[i + 1] - times[i]))
            crossings.append(crossing_time)

    if len(crossings) < 2:
        tau = np.mod((times - times[0]) / max(times[-1] - times[0], 1e-12), 1.0)
        unwrapped = 2.0 * np.pi * (times - times[0]) / max(times[-1] - times[0], 1e-12)
        return tau.astype(np.float32), unwrapped.astype(np.float32), estimate_frequency_fft(signal_values, times)

    crossings_array = np.asarray(crossings, dtype=np.float64)
    avg_period = float(np.mean(np.diff(crossings_array)))
    if not np.isfinite(avg_period) or avg_period <= 0:
        avg_period = max(float(times[-1] - times[0]) / max(len(crossings_array) - 1, 1), 1e-12)

    unwrapped_phase = np.empty_like(times, dtype=np.float64)
    cycle_idx = 0
    for i, t_val in enumerate(times):
        while cycle_idx + 1 < len(crossings_array) and t_val >= crossings_array[cycle_idx + 1]:
            cycle_idx += 1
        t0 = crossings_array[cycle_idx]
        phase_fraction = (t_val - t0) / avg_period
        unwrapped_phase[i] = 2.0 * np.pi * (cycle_idx + phase_fraction)

    tau = np.mod(unwrapped_phase / (2.0 * np.pi), 1.0)
    dominant_frequency = 1.0 / avg_period
    return tau.astype(np.float32), unwrapped_phase.astype(np.float32), float(dominant_frequency)


def estimate_phase(case_data: CaseData) -> Tuple[np.ndarray, np.ndarray, float, ProbeInfo, np.ndarray]:
    """Estimate frame-wise phase tau in [0, 1) using the v-velocity probe signal."""
    probe = choose_phase_probe(case_data.cfg, case_data.x_grid, case_data.y_grid)
    iy, ix = probe.probe_ij
    signal_values = case_data.tensor[:, iy, ix, 1].detach().cpu().numpy().astype(np.float64, copy=False)
    centered = signal_values - np.mean(signal_values)

    tau: Optional[np.ndarray] = None
    unwrapped_phase: Optional[np.ndarray] = None
    dominant_frequency = float("nan")

    if np.all(np.isfinite(centered)) and np.std(centered) > 1e-8 and centered.size >= 8:
        try:
            analytic = hilbert(centered)
            raw_phase = np.unwrap(np.angle(analytic))
            tau_candidate = np.mod((raw_phase - raw_phase[0]) / (2.0 * np.pi), 1.0)
            slope = np.polyfit(case_data.times, raw_phase, deg=1)[0]
            dominant_frequency = float(slope / (2.0 * np.pi))
            if np.isfinite(dominant_frequency) and abs(dominant_frequency) > 1e-10:
                tau = tau_candidate.astype(np.float32)
                unwrapped_phase = raw_phase.astype(np.float32)
        except Exception:
            tau = None
            unwrapped_phase = None

    if tau is None or unwrapped_phase is None:
        tau, unwrapped_phase, dominant_frequency = estimate_phase_zero_crossing(signal_values, case_data.times)

    if not np.isfinite(dominant_frequency) or dominant_frequency <= 0:
        dominant_frequency = estimate_frequency_fft(signal_values, case_data.times)

    return tau, unwrapped_phase, float(dominant_frequency), probe, signal_values.astype(np.float32)


# --------------------------- Canonical cycle -----------------------------------


def _json_sanitize(value: Any) -> Any:
    """Convert numpy/torch scalar containers into JSON-friendly Python values."""
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_sanitize(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compute_phase_quality(
    times: np.ndarray,
    tau: np.ndarray,
    phase_unwrapped: np.ndarray,
    tensor: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute scalar diagnostics for phase reliability and raw-time continuity."""
    times_np = np.asarray(times, dtype=np.float64)
    tau_np = np.asarray(tau, dtype=np.float64)
    phase_np = np.asarray(phase_unwrapped, dtype=np.float64)
    num_frames = int(times_np.size)
    time_diffs = np.diff(times_np) if num_frames > 1 else np.asarray([], dtype=np.float64)
    tau_diffs = np.diff(tau_np) if tau_np.size > 1 else np.asarray([], dtype=np.float64)
    phase_diffs = np.diff(phase_np) if phase_np.size > 1 else np.asarray([], dtype=np.float64)

    if num_frames > 1 and tau_np.size == num_frames:
        order = np.argsort(tau_np)
        tau_sorted_time_jumps = np.abs(np.diff(times_np[order]))
        largest_time_jump_after_tau_sort = float(np.max(tau_sorted_time_jumps)) if tau_sorted_time_jumps.size else 0.0
    else:
        largest_time_jump_after_tau_sort = 0.0

    quality: Dict[str, float] = {
        "num_frames": float(num_frames),
        "time_span": float(times_np[-1] - times_np[0]) if num_frames > 1 else 0.0,
        "median_dt": float(np.median(time_diffs)) if time_diffs.size else 0.0,
        "tau_negative_time_diffs": float(np.sum(tau_diffs < -1e-6)) if tau_diffs.size else 0.0,
        "phase_unwrapped_negative_diffs": float(np.sum(phase_diffs < -1e-6)) if phase_diffs.size else 0.0,
        "max_abs_phase_step": float(np.max(np.abs(phase_diffs))) if phase_diffs.size else 0.0,
        "mean_abs_phase_step": float(np.mean(np.abs(phase_diffs))) if phase_diffs.size else 0.0,
        "estimated_cycles": float((phase_np[-1] - phase_np[0]) / (2.0 * np.pi)) if phase_np.size > 1 else 0.0,
        "phase_monotonic_fraction": float(np.mean(phase_diffs >= -1e-6)) if phase_diffs.size else 1.0,
        "largest_time_jump_after_tau_sort": largest_time_jump_after_tau_sort,
    }

    if tensor is not None and tensor.ndim == 4 and tensor.shape[0] > 1 and tensor.shape[-1] > 3:
        omega = tensor[..., 3]
        jump_mse = torch.mean((omega[1:] - omega[:-1]).square(), dim=tuple(range(1, omega.ndim)))
        quality["raw_consecutive_omega_jump_mse_mean"] = float(jump_mse.mean().detach().cpu().item())
        quality["raw_consecutive_omega_jump_mse_max"] = float(jump_mse.max().detach().cpu().item())

    return quality


def _field_jump_stats(tensor: torch.Tensor, start_idx: int, end_idx: int) -> Dict[str, float]:
    """Return closure and consecutive-jump diagnostics for an inclusive frame window."""
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    window = tensor[start_idx : end_idx + 1]
    if window.shape[0] == 0:
        return {
            "closure_mse_omega": float("inf"),
            "closure_mse_all": float("inf"),
            "consecutive_jump_mse_mean_omega": float("inf"),
            "consecutive_jump_mse_max_omega": float("inf"),
        }

    closure_all = torch.mean((window[-1] - window[0]).square())
    if window.shape[-1] > 3:
        omega = window[..., 3]
        closure_omega = torch.mean((omega[-1] - omega[0]).square())
        if omega.shape[0] > 1:
            omega_jumps = torch.mean((omega[1:] - omega[:-1]).square(), dim=tuple(range(1, omega.ndim)))
            jump_mean = omega_jumps.mean()
            jump_max = omega_jumps.max()
        else:
            jump_mean = torch.zeros((), dtype=window.dtype, device=window.device)
            jump_max = torch.zeros((), dtype=window.dtype, device=window.device)
    else:
        closure_omega = closure_all
        if window.shape[0] > 1:
            jumps = torch.mean((window[1:] - window[:-1]).square(), dim=tuple(range(1, window.ndim)))
            jump_mean = jumps.mean()
            jump_max = jumps.max()
        else:
            jump_mean = torch.zeros((), dtype=window.dtype, device=window.device)
            jump_max = torch.zeros((), dtype=window.dtype, device=window.device)

    return {
        "closure_mse_omega": float(closure_omega.detach().cpu().item()),
        "closure_mse_all": float(closure_all.detach().cpu().item()),
        "consecutive_jump_mse_mean_omega": float(jump_mean.detach().cpu().item()),
        "consecutive_jump_mse_max_omega": float(jump_max.detach().cpu().item()),
    }


def _detect_upward_crossings(signal_values: Optional[np.ndarray], times: np.ndarray) -> List[Dict[str, float]]:
    if signal_values is None:
        return []
    signal = np.asarray(signal_values, dtype=np.float64)
    times_np = np.asarray(times, dtype=np.float64)
    if signal.size != times_np.size or signal.size < 2 or not np.all(np.isfinite(signal)):
        return []

    centered = signal - np.mean(signal)
    crossings: List[Dict[str, float]] = []
    for i in range(centered.size - 1):
        a = centered[i]
        b = centered[i + 1]
        if a < 0.0 <= b:
            denom = b - a
            alpha = 0.0 if abs(denom) < 1e-12 else float(-a / denom)
            crossings.append(
                {
                    "position": float(i + alpha),
                    "time": float(times_np[i] + alpha * (times_np[i + 1] - times_np[i])),
                }
            )
    return crossings


def _estimate_period_from_inputs(
    times: np.ndarray,
    phase_unwrapped: Optional[np.ndarray],
    crossings: Sequence[Dict[str, float]],
) -> float:
    if len(crossings) >= 2:
        durations = np.diff(np.asarray([c["time"] for c in crossings], dtype=np.float64))
        durations = durations[np.isfinite(durations) & (durations > 0.0)]
        if durations.size:
            return float(np.median(durations))

    if phase_unwrapped is not None:
        phase_np = np.asarray(phase_unwrapped, dtype=np.float64)
        times_np = np.asarray(times, dtype=np.float64)
        if phase_np.size > 1 and times_np.size == phase_np.size:
            cycles = float((phase_np[-1] - phase_np[0]) / (2.0 * np.pi))
            span = float(times_np[-1] - times_np[0])
            if np.isfinite(cycles) and cycles > 1e-6 and span > 0.0:
                return span / cycles
    return float("nan")


def _candidate_cycle_windows(
    times: np.ndarray,
    num_bins: int,
    min_cycle_frames: int,
    phase_unwrapped: Optional[np.ndarray],
    signal_values: Optional[np.ndarray],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]], float]:
    times_np = np.asarray(times, dtype=np.float64)
    num_frames = times_np.size
    crossings = _detect_upward_crossings(signal_values, times_np)
    candidates: List[Dict[str, Any]] = []

    for left, right in zip(crossings[:-1], crossings[1:]):
        start_idx = int(np.ceil(left["position"]))
        end_idx = int(np.floor(right["position"]))
        if 0 <= start_idx < end_idx < num_frames:
            candidates.append(
                {
                    "method": "contiguous_zero_crossing",
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "boundary_start_time": float(left["time"]),
                    "boundary_end_time": float(right["time"]),
                }
            )

    if phase_unwrapped is not None and len(candidates) == 0:
        phase_np = np.asarray(phase_unwrapped, dtype=np.float64)
        if phase_np.size == num_frames and num_frames > 1:
            diffs = np.diff(phase_np)
            start = 0
            for i, diff in enumerate(diffs):
                if diff < -1e-6:
                    if i + 1 - start >= max(2, min_cycle_frames):
                        candidates.append({"method": "contiguous_phase", "start_idx": start, "end_idx": i})
                    start = i + 1
            if num_frames - start >= max(2, min_cycle_frames):
                candidates.append({"method": "contiguous_phase", "start_idx": start, "end_idx": num_frames - 1})

    estimated_period = _estimate_period_from_inputs(times_np, phase_unwrapped, crossings)
    if len(candidates) == 0 and np.isfinite(estimated_period) and estimated_period > 0.0 and num_frames > 1:
        end_time = float(times_np[-1])
        start_time = max(float(times_np[0]), end_time - estimated_period)
        start_idx = int(np.searchsorted(times_np, start_time, side="left"))
        end_idx = num_frames - 1
        if start_idx < end_idx:
            candidates.append(
                {
                    "method": "time_period_fallback",
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "boundary_start_time": start_time,
                    "boundary_end_time": end_time,
                }
            )

    if len(candidates) == 0 and num_frames > 1:
        candidates.append({"method": "time_full_span_fallback", "start_idx": 0, "end_idx": num_frames - 1})

    preferred_count = max(int(num_bins), int(min_cycle_frames), 8)
    good = [c for c in candidates if (int(c["end_idx"]) - int(c["start_idx"]) + 1) >= preferred_count]
    if good:
        candidates = good
    else:
        usable = [c for c in candidates if (int(c["end_idx"]) - int(c["start_idx"]) + 1) >= max(2, int(min_cycle_frames))]
        if usable:
            candidates = usable

    return candidates, crossings, estimated_period


def _score_candidate(
    candidate: Dict[str, Any],
    tensor: torch.Tensor,
    times: np.ndarray,
    signal_values: Optional[np.ndarray],
    estimated_period: float,
    median_dt: float,
    prefer_late_cycle: bool,
) -> Dict[str, Any]:
    start_idx = int(candidate["start_idx"])
    end_idx = int(candidate["end_idx"])
    num_frames = int(end_idx - start_idx + 1)
    duration = float(times[end_idx] - times[start_idx]) if end_idx > start_idx else 0.0
    internal_dt = np.diff(times[start_idx : end_idx + 1])
    max_internal_gap = float(np.max(internal_dt)) if internal_dt.size else 0.0
    stats = _field_jump_stats(tensor, start_idx, end_idx)

    if np.isfinite(estimated_period) and estimated_period > 0.0 and duration > 0.0:
        duration_penalty = abs(duration - estimated_period) / estimated_period
    else:
        duration_penalty = 0.0

    if signal_values is not None:
        signal = np.asarray(signal_values, dtype=np.float64)
        global_std = float(np.std(signal)) + 1e-12
        signal_amplitude = float(np.std(signal[start_idx : end_idx + 1]) / global_std)
    else:
        signal_amplitude = 0.0

    gap_limit = 4.0 * median_dt if median_dt > 0.0 else float("inf")
    gap_penalty = max(0.0, max_internal_gap / gap_limit - 1.0) if np.isfinite(gap_limit) and gap_limit > 0.0 else 0.0
    smooth_mean = stats["consecutive_jump_mse_mean_omega"] + 1e-12
    closure_ratio = stats["closure_mse_omega"] / smooth_mean
    max_jump_ratio = stats["consecutive_jump_mse_max_omega"] / smooth_mean
    late_bonus = (end_idx / max(tensor.shape[0] - 1, 1)) if prefer_late_cycle else 0.0

    score = (
        closure_ratio
        + 0.10 * max_jump_ratio
        + 0.25 * duration_penalty
        + 2.0 * gap_penalty
        - 0.05 * min(signal_amplitude, 4.0)
        - 0.05 * late_bonus
    )

    enriched = dict(candidate)
    enriched.update(stats)
    enriched.update(
        {
            "source_num_frames": num_frames,
            "source_duration": duration,
            "max_internal_time_gap": max_internal_gap,
            "duration_penalty": float(duration_penalty),
            "signal_amplitude_ratio": float(signal_amplitude),
            "candidate_score": float(score),
        }
    )
    return enriched


def _select_cycle_window(
    tensor: torch.Tensor,
    times: np.ndarray,
    num_bins: int,
    min_cycle_frames: int,
    phase_unwrapped: Optional[np.ndarray],
    signal_values: Optional[np.ndarray],
    prefer_late_cycle: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, float]], float]:
    times_np = np.asarray(times, dtype=np.float64)
    candidates, crossings, estimated_period = _candidate_cycle_windows(
        times=times_np,
        num_bins=num_bins,
        min_cycle_frames=min_cycle_frames,
        phase_unwrapped=phase_unwrapped,
        signal_values=signal_values,
    )
    if not candidates:
        raise RuntimeError("No candidate canonical-cycle windows could be constructed.")

    median_dt = float(np.median(np.diff(times_np))) if times_np.size > 1 else 0.0
    scored = [
        _score_candidate(c, tensor, times_np, signal_values, estimated_period, median_dt, prefer_late_cycle)
        for c in candidates
    ]
    scored.sort(key=lambda c: (float(c["candidate_score"]), -int(c["end_idx"]) if prefer_late_cycle else 0))
    return scored[0], scored, crossings, estimated_period


def _interpolate_time_ordered(
    flat: torch.Tensor,
    local_phase: np.ndarray,
    phase_centers: np.ndarray,
    output_shape: Tuple[int, int, int, int],
) -> torch.Tensor:
    """Linearly interpolate flattened fields along a monotone local phase axis."""
    local_phase_np = np.asarray(local_phase, dtype=np.float64)
    finite = np.isfinite(local_phase_np)
    if not np.all(finite):
        keep = np.flatnonzero(finite)
        local_phase_np = local_phase_np[keep]
        flat = flat.index_select(0, torch.as_tensor(keep, device=flat.device, dtype=torch.long))

    if local_phase_np.size == 0:
        raise ValueError("Cannot interpolate canonical cycle from an empty local phase axis.")

    unique_mask = np.ones(local_phase_np.shape, dtype=bool)
    unique_mask[1:] = np.diff(local_phase_np) > 1e-9
    if not np.all(unique_mask):
        keep = np.flatnonzero(unique_mask)
        local_phase_np = local_phase_np[keep]
        flat = flat.index_select(0, torch.as_tensor(keep, device=flat.device, dtype=torch.long))

    if local_phase_np.size == 1:
        canonical = flat[:1].repeat(len(phase_centers), 1)
        return canonical.reshape(output_shape)

    canonical_flat = torch.empty((len(phase_centers), flat.shape[1]), dtype=flat.dtype, device=flat.device)
    for i, phase_bin in enumerate(phase_centers):
        right = int(np.searchsorted(local_phase_np, float(phase_bin), side="right"))
        right = min(max(right, 1), len(local_phase_np) - 1)
        left = right - 1
        phase_left = local_phase_np[left]
        phase_right = local_phase_np[right]
        if phase_right <= phase_left + 1e-12:
            canonical_flat[i] = flat[left]
        else:
            weight = float((float(phase_bin) - phase_left) / (phase_right - phase_left))
            canonical_flat[i] = ((1.0 - weight) * flat[left]) + (weight * flat[right])
    return canonical_flat.reshape(output_shape)


def _legacy_tau_sort_canonical_cycle(
    tensor: torch.Tensor,
    tau: np.ndarray,
    num_bins: int,
) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
    """Original global folded-tau interpolation, kept only for debug comparison."""
    num_frames, height, width, channels = tensor.shape
    flat = tensor.reshape(num_frames, -1)
    phase_centers = (np.arange(num_bins, dtype=np.float32) + 0.5) / float(num_bins)

    order = np.argsort(np.asarray(tau, dtype=np.float64))
    tau_sorted = np.asarray(tau, dtype=np.float64)[order]
    order_torch = torch.as_tensor(order, device=tensor.device, dtype=torch.long)
    flat_sorted = flat.index_select(0, order_torch)

    if np.allclose(tau_sorted, tau_sorted[0]):
        canonical = flat_sorted[:1].repeat(num_bins, 1)
        return phase_centers, canonical.reshape(num_bins, height, width, channels), {
            "canonical_method": "legacy_tau_sort",
            "source_start_frame": int(order[0]) if order.size else 0,
            "source_end_frame": int(order[-1]) if order.size else 0,
        }

    tau_ext = np.concatenate([tau_sorted[-1:] - 1.0, tau_sorted, tau_sorted[:1] + 1.0])
    flat_ext = torch.cat([flat_sorted[-1:], flat_sorted, flat_sorted[:1]], dim=0)
    canonical = _interpolate_time_ordered(
        flat_ext,
        tau_ext,
        phase_centers,
        (num_bins, height, width, channels),
    )
    meta = {
        "canonical_method": "legacy_tau_sort",
        "source_start_frame": int(order[0]) if order.size else 0,
        "source_end_frame": int(order[-1]) if order.size else 0,
    }
    return phase_centers, canonical, meta


def _time_full_span_canonical_cycle(
    tensor: torch.Tensor,
    num_bins: int,
    times: Optional[np.ndarray],
    frame_ids: Optional[np.ndarray],
    method_name: str = "time_full_span",
) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
    num_frames, height, width, channels = tensor.shape
    phase_centers = (np.arange(num_bins, dtype=np.float32) + 0.5) / float(num_bins)
    if times is None:
        times_np = np.arange(num_frames, dtype=np.float64)
    else:
        times_np = np.asarray(times, dtype=np.float64)
    local_phase = (times_np - times_np[0]) / max(float(times_np[-1] - times_np[0]), 1e-12)
    flat = tensor.reshape(num_frames, -1)
    canonical = _interpolate_time_ordered(flat, local_phase, phase_centers, (num_bins, height, width, channels))
    stats = _field_jump_stats(tensor, 0, num_frames - 1)
    source_start_frame = int(frame_ids[0]) if frame_ids is not None and len(frame_ids) else 0
    source_end_frame = int(frame_ids[-1]) if frame_ids is not None and len(frame_ids) else num_frames - 1
    meta = {
        "canonical_method": method_name,
        "source_start_frame": source_start_frame,
        "source_end_frame": source_end_frame,
        "source_start_index": 0,
        "source_end_index": int(num_frames - 1),
        "source_start_time": float(times_np[0]),
        "source_end_time": float(times_np[-1]),
        "source_num_frames": int(num_frames),
        "source_duration": float(times_np[-1] - times_np[0]) if num_frames > 1 else 0.0,
        **stats,
    }
    return phase_centers, canonical, meta


def build_canonical_cycle(
    tensor: torch.Tensor,
    tau: np.ndarray,
    num_bins: int,
    *,
    times: Optional[np.ndarray] = None,
    phase_unwrapped: Optional[np.ndarray] = None,
    signal_values: Optional[np.ndarray] = None,
    frame_ids: Optional[np.ndarray] = None,
    method: str = "contiguous_cycle",
    min_cycle_frames: int = 8,
    prefer_late_cycle: bool = True,
) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
    """Build a canonical cycle without scrambling raw-time continuity.

    The old default sorted every frame by folded tau in [0, 1).  That is unsafe
    when the probe phase is noisy: adjacent tau-sorted samples may come from raw
    times many seconds apart, creating a discontinuous ground-truth cycle.  The
    default path now extracts one contiguous raw-time window, then interpolates
    phase bins inside that window only.  The legacy builder remains available for
    diagnostics and backwards comparisons, but should not be used for training
    data unless the phase estimate is known to be reliable.
    """
    normalized_method = method.strip().lower()
    if normalized_method in {"legacy_tau_sort", "legacy"}:
        return _legacy_tau_sort_canonical_cycle(tensor, tau, num_bins)
    if normalized_method in {"time_full_span", "full_span"}:
        return _time_full_span_canonical_cycle(tensor, num_bins, times, frame_ids, method_name="time_full_span")

    if times is None:
        return _time_full_span_canonical_cycle(
            tensor,
            num_bins,
            np.arange(tensor.shape[0], dtype=np.float64),
            frame_ids,
            method_name="time_full_span_fallback",
        )

    times_np = np.asarray(times, dtype=np.float64)
    num_frames, height, width, channels = tensor.shape
    phase_centers = (np.arange(num_bins, dtype=np.float32) + 0.5) / float(num_bins)

    selected, candidates, crossings, estimated_period = _select_cycle_window(
        tensor=tensor,
        times=times_np,
        num_bins=num_bins,
        min_cycle_frames=min_cycle_frames,
        phase_unwrapped=phase_unwrapped,
        signal_values=signal_values,
        prefer_late_cycle=prefer_late_cycle,
    )

    start_idx = int(selected["start_idx"])
    end_idx = int(selected["end_idx"])
    segment_times = times_np[start_idx : end_idx + 1]
    segment = tensor[start_idx : end_idx + 1]

    local_phase_source = "time"
    if phase_unwrapped is not None:
        phase_segment = np.asarray(phase_unwrapped, dtype=np.float64)[start_idx : end_idx + 1]
        phase_diffs = np.diff(phase_segment)
        phase_span = float(phase_segment[-1] - phase_segment[0]) if phase_segment.size > 1 else 0.0
        if phase_segment.size > 1 and phase_span > 1e-8 and np.all(phase_diffs >= -1e-6):
            local_phase = (phase_segment - phase_segment[0]) / phase_span
            local_phase_source = "phase_unwrapped"
        else:
            local_phase = (segment_times - segment_times[0]) / max(float(segment_times[-1] - segment_times[0]), 1e-12)
    else:
        local_phase = (segment_times - segment_times[0]) / max(float(segment_times[-1] - segment_times[0]), 1e-12)

    flat = segment.reshape(segment.shape[0], -1)
    canonical = _interpolate_time_ordered(flat, local_phase, phase_centers, (num_bins, height, width, channels))

    canonical_jumps: Dict[str, float] = {}
    if canonical.shape[0] > 1 and canonical.shape[-1] > 3:
        omega = canonical[..., 3]
        jumps = torch.mean((omega[1:] - omega[:-1]).square(), dim=tuple(range(1, omega.ndim)))
        canonical_jumps = {
            "canonical_consecutive_jump_mse_mean_omega": float(jumps.mean().detach().cpu().item()),
            "canonical_consecutive_jump_mse_max_omega": float(jumps.max().detach().cpu().item()),
            "canonical_closure_mse_omega": float(torch.mean((omega[-1] - omega[0]).square()).detach().cpu().item()),
        }

    source_start_frame = int(frame_ids[start_idx]) if frame_ids is not None and len(frame_ids) > start_idx else start_idx
    source_end_frame = int(frame_ids[end_idx]) if frame_ids is not None and len(frame_ids) > end_idx else end_idx
    meta: Dict[str, Any] = {
        "canonical_method": str(selected["method"]),
        "local_phase_source": local_phase_source,
        "source_start_frame": source_start_frame,
        "source_end_frame": source_end_frame,
        "source_start_index": int(start_idx),
        "source_end_index": int(end_idx),
        "source_start_time": float(times_np[start_idx]),
        "source_end_time": float(times_np[end_idx]),
        "source_num_frames": int(end_idx - start_idx + 1),
        "source_duration": float(times_np[end_idx] - times_np[start_idx]),
        "estimated_period": float(estimated_period) if np.isfinite(estimated_period) else float("nan"),
        "num_upward_zero_crossings": int(len(crossings)),
        "num_candidate_cycles": int(len(candidates)),
        "candidate_score": float(selected.get("candidate_score", float("nan"))),
        "candidate_summaries": [
            {
                "method": str(c.get("method", "")),
                "start_idx": int(c["start_idx"]),
                "end_idx": int(c["end_idx"]),
                "source_num_frames": int(c.get("source_num_frames", int(c["end_idx"]) - int(c["start_idx"]) + 1)),
                "source_duration": float(c.get("source_duration", 0.0)),
                "closure_mse_omega": float(c.get("closure_mse_omega", float("nan"))),
                "consecutive_jump_mse_mean_omega": float(c.get("consecutive_jump_mse_mean_omega", float("nan"))),
                "candidate_score": float(c.get("candidate_score", float("nan"))),
            }
            for c in candidates[:8]
        ],
        **{k: selected[k] for k in selected if k.startswith("closure_") or k.startswith("consecutive_")},
        **canonical_jumps,
    }

    return phase_centers, canonical, meta


# ------------------------- Behavior descriptors --------------------------------


def compute_case_statistics(
    case_data: CaseData,
    mean_field: torch.Tensor,
    residual: torch.Tensor,
    rms_field: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Compute scalar and field-based descriptors for one case."""
    per_channel_rms_energy = torch.sqrt(torch.mean(residual.square(), dim=(0, 1, 2)))
    fluctuation_energy = torch.mean(residual.square()).reshape(1)

    wake_deficits = []
    u_mean = mean_field[..., 0].detach().cpu().numpy()
    probe_dx = 3.0 * float(case_data.cfg.domain.cylinder_radius)
    probe_dy = 2.0 * float(case_data.cfg.domain.cylinder_radius)

    for cx, cy in case_data.cfg.layout.centers or []:
        dx = periodic_offsets(case_data.x_grid - (cx + probe_dx), case_data.cfg.domain.lx)
        dy = periodic_offsets(case_data.y_grid - cy, case_data.cfg.domain.ly)
        wake_mask = (np.abs(dx) <= probe_dx) & (np.abs(dy) <= probe_dy)
        if np.any(wake_mask):
            deficit = float(case_data.cfg.flow.u_bulk - np.mean(u_mean[wake_mask]))
        else:
            deficit = 0.0
        wake_deficits.append(deficit)

    return {
        "mean_field": mean_field.detach().cpu().numpy().astype(np.float32, copy=False),
        "rms_field": rms_field.detach().cpu().numpy().astype(np.float32, copy=False),
        "per_channel_rms_energy": per_channel_rms_energy.detach().cpu().numpy().astype(np.float32, copy=False),
        "fluctuation_energy": fluctuation_energy.detach().cpu().numpy().astype(np.float32, copy=False),
        "wake_deficits": np.asarray(wake_deficits, dtype=np.float32),
    }


def compute_pod_snapshots(residual: torch.Tensor, rank: int) -> Dict[str, np.ndarray]:
    """Compute POD descriptors using the Method of Snapshots."""
    num_frames = residual.shape[0]
    flattened = residual.reshape(num_frames, -1)
    correlation = (flattened @ flattened.T) / max(num_frames - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(correlation)

    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals.index_select(0, order)
    eigvecs = eigvecs.index_select(1, order)

    positive = eigvals > 1e-12
    eigvals = eigvals[positive][:rank]
    eigvecs = eigvecs[:, positive][:, :rank]

    if eigvals.numel() == 0:
        return {
            "pod_eigenvalues": np.zeros((0,), dtype=np.float32),
            "pod_temporal_coefficients": np.zeros((num_frames, 0), dtype=np.float32),
            "pod_modes": np.zeros((0, residual.shape[1], residual.shape[2], residual.shape[3]), dtype=np.float32),
        }

    normalization = torch.sqrt(eigvals * max(num_frames - 1, 1))
    modes = (flattened.T @ eigvecs) / normalization[None, :]
    coefficients = eigvecs * normalization[None, :]

    return {
        "pod_eigenvalues": eigvals.detach().cpu().numpy().astype(np.float32, copy=False),
        "pod_temporal_coefficients": coefficients.detach().cpu().numpy().astype(np.float32, copy=False),
        "pod_modes": modes.T.reshape(eigvals.shape[0], residual.shape[1], residual.shape[2], residual.shape[3]).detach().cpu().numpy().astype(np.float32, copy=False),
    }


# ----------------------------- Sampling logic ----------------------------------


def build_sampling_masks(case_data: CaseData) -> Dict[str, np.ndarray]:
    """Build reusable spatial masks for structured oversampling."""
    cfg = case_data.cfg
    radius = float(cfg.domain.cylinder_radius)
    diameter = cylinder_diameter(cfg)
    x_grid = case_data.x_grid
    y_grid = case_data.y_grid

    annulus_mask = np.zeros_like(x_grid, dtype=bool)
    wake_mask = np.zeros_like(x_grid, dtype=bool)
    annulus_outer = radius * args_cache["annulus_outer_radius_factor"]
    wake_length = diameter * args_cache["wake_length_diameters"]
    wake_half_width = diameter * args_cache["wake_half_width_diameters"]

    for cx, cy in cfg.layout.centers or []:
        dx = periodic_offsets(x_grid - cx, cfg.domain.lx)
        dy = periodic_offsets(y_grid - cy, cfg.domain.ly)
        rr = np.sqrt(dx * dx + dy * dy)
        annulus_mask |= (rr >= radius) & (rr <= annulus_outer)

        downstream = periodic_offsets(x_grid - cx, cfg.domain.lx)
        wake_mask |= (downstream >= radius) & (downstream <= wake_length) & (np.abs(dy) <= wake_half_width)

    return {"annulus": annulus_mask, "wake": wake_mask}


def sample_indices_from_mask(
    rng: np.random.Generator,
    candidate_indices: np.ndarray,
    quota: int,
) -> np.ndarray:
    if quota <= 0 or candidate_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)
    replace = quota > candidate_indices.size
    return rng.choice(candidate_indices, size=quota, replace=replace)


def sample_phase_points(
    canonical_cycle: np.ndarray,
    phase_centers: np.ndarray,
    case_data: CaseData,
    points_per_phase_bin: int,
    sampling_mode: str,
    annulus_ratio: float,
    wake_ratio: float,
) -> Dict[str, np.ndarray]:
    """Sample pointwise neural-field tuples from the canonical cycle."""
    num_bins, height, width, _ = canonical_cycle.shape
    x_flat = case_data.x_grid.reshape(-1)
    y_flat = case_data.y_grid.reshape(-1)
    masks = build_sampling_masks(case_data)
    annulus_indices = np.flatnonzero(masks["annulus"].reshape(-1))
    wake_indices = np.flatnonzero(masks["wake"].reshape(-1))
    all_indices = np.arange(height * width, dtype=np.int64)
    rng = np.random.default_rng(int(case_data.cfg.layout.seed) + 12345)

    sampled_blocks = {key: [] for key in ("phase_bin", "tau", "x", "y", "u", "v", "p", "omega")}

    for phase_idx in range(num_bins):
        field_flat = canonical_cycle[phase_idx].reshape(-1, canonical_cycle.shape[-1])

        if points_per_phase_bin <= 0:
            chosen = all_indices
        else:
            if sampling_mode == "uniform":
                chosen = sample_indices_from_mask(rng, all_indices, points_per_phase_bin)
            elif sampling_mode == "annulus":
                annulus_quota = int(round(points_per_phase_bin * annulus_ratio))
                uniform_quota = max(points_per_phase_bin - annulus_quota, 0)
                chosen = np.concatenate(
                    [
                        sample_indices_from_mask(rng, annulus_indices, annulus_quota),
                        sample_indices_from_mask(rng, all_indices, uniform_quota),
                    ]
                )
            elif sampling_mode == "wake":
                wake_quota = int(round(points_per_phase_bin * wake_ratio))
                uniform_quota = max(points_per_phase_bin - wake_quota, 0)
                chosen = np.concatenate(
                    [
                        sample_indices_from_mask(rng, wake_indices, wake_quota),
                        sample_indices_from_mask(rng, all_indices, uniform_quota),
                    ]
                )
            else:
                annulus_quota = int(round(points_per_phase_bin * annulus_ratio))
                wake_quota = int(round(points_per_phase_bin * wake_ratio))
                uniform_quota = max(points_per_phase_bin - annulus_quota - wake_quota, 0)
                chosen = np.concatenate(
                    [
                        sample_indices_from_mask(rng, annulus_indices, annulus_quota),
                        sample_indices_from_mask(rng, wake_indices, wake_quota),
                        sample_indices_from_mask(rng, all_indices, uniform_quota),
                    ]
                )

        values = field_flat[chosen]
        sampled_blocks["phase_bin"].append(np.full(chosen.size, phase_idx, dtype=np.int32))
        sampled_blocks["tau"].append(np.full(chosen.size, phase_centers[phase_idx], dtype=np.float32))
        sampled_blocks["x"].append(x_flat[chosen].astype(np.float32))
        sampled_blocks["y"].append(y_flat[chosen].astype(np.float32))
        sampled_blocks["u"].append(values[:, 0].astype(np.float32))
        sampled_blocks["v"].append(values[:, 1].astype(np.float32))
        sampled_blocks["p"].append(values[:, 2].astype(np.float32))
        sampled_blocks["omega"].append(values[:, 3].astype(np.float32))

    return {key: np.concatenate(val, axis=0) if val else np.zeros((0,), dtype=np.float32) for key, val in sampled_blocks.items()}


# ------------------------------- Output writers --------------------------------


args_cache: Dict[str, float] = {}


def progress_enabled() -> bool:
    """Enable tqdm only for interactive terminals."""
    return sys.stdout.isatty()


def report_state(message: str, progress_bar: Optional[tqdm] = None) -> None:
    """Write a concise state message without breaking tqdm rendering."""
    if progress_bar is not None:
        progress_bar.write(message)
    else:
        print(message)


def case_output_dir(output_root: Path, record: CaseRecord) -> Path:
    if record.split in {"train", "test"}:
        return output_root / record.split / record.case_dir.name
    return output_root / record.case_dir.name


def save_structure_json(
    out_dir: Path,
    case_data: CaseData,
    probe: ProbeInfo,
    dominant_frequency: float,
    save_cycles: int,
    phase_quality: Optional[Dict[str, float]] = None,
    canonical_meta: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "case_id": str(case_data.cfg.save.case_id),
        "case_dir_name": case_data.record.case_dir.name,
        "split": case_data.record.split,
        "mode": case_data.cfg.mode,
        "re": float(case_data.cfg.flow.re),
        "num_cylinders": int(case_data.cfg.layout.num_cylinders),
        "cylinder_radius": float(case_data.cfg.domain.cylinder_radius),
        "domain": {
            "nx": int(case_data.cfg.domain.nx),
            "ny": int(case_data.cfg.domain.ny),
            "lx": float(case_data.cfg.domain.lx),
            "ly": float(case_data.cfg.domain.ly),
        },
        "cylinder_centers": case_data.cfg.layout.centers,
        "available_fields": case_data.available_fields,
        "save_cycles": int(save_cycles),
        "probe": {
            "cylinder_index": probe.cylinder_index,
            "cylinder_center": list(probe.cylinder_center),
            "probe_xy": list(probe.probe_xy),
            "probe_ij": list(probe.probe_ij),
        },
        "dominant_frequency": float(dominant_frequency),
        "source_case_dir": str(case_data.record.case_dir.resolve()),
    }
    if phase_quality is not None:
        payload["phase_quality"] = _json_sanitize(phase_quality)
    if canonical_meta is not None:
        payload["canonical_cycle"] = _json_sanitize(canonical_meta)
    with (out_dir / "structure.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_behavior_summary(out_dir: Path, behavior: Dict[str, np.ndarray], pod: Optional[Dict[str, np.ndarray]]) -> None:
    payload = dict(behavior)
    if pod is not None:
        payload.update(pod)
    np.savez_compressed(out_dir / "behavior_summary.npz", **payload)


def save_canonical_cycle(out_dir: Path, canonical_cycle: np.ndarray, phase_centers: np.ndarray, save_full: bool) -> None:
    if not save_full:
        return
    np.savez_compressed(
        out_dir / "canonical_cycle.npz",
        canonical_cycle=canonical_cycle.astype(np.float32),
        phase_bin_centers=phase_centers.astype(np.float32),
        channel_order=np.asarray(FIELD_CHANNEL_ORDER),
    )


def save_phase_metadata(
    out_dir: Path,
    case_data: CaseData,
    tau: np.ndarray,
    phase_unwrapped: np.ndarray,
    signal_values: np.ndarray,
    dominant_frequency: float,
    phase_quality: Optional[Dict[str, float]] = None,
    canonical_meta: Optional[Dict[str, Any]] = None,
    save_diagnostics: bool = True,
) -> None:
    payload = {
        "saved_frame": case_data.frame_ids,
        "time": case_data.times,
        "tau": tau,
        "phase_unwrapped": phase_unwrapped,
        "probe_signal_v": signal_values,
        "dominant_frequency": np.full_like(tau, dominant_frequency, dtype=np.float32),
    }
    if save_diagnostics and canonical_meta is not None:
        selected = np.zeros_like(case_data.frame_ids, dtype=np.int32)
        start_idx = int(canonical_meta.get("source_start_index", -1))
        end_idx = int(canonical_meta.get("source_end_index", -1))
        if 0 <= start_idx <= end_idx < selected.size:
            selected[start_idx : end_idx + 1] = 1
        payload["selected_canonical_window"] = selected
    if save_diagnostics and phase_quality is not None:
        for key, value in phase_quality.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                payload[f"phase_quality_{key}"] = np.full_like(tau, float(value), dtype=np.float32)

    df = pd.DataFrame(payload)
    df.to_csv(out_dir / "phase_metadata.csv", index=False)


def save_canonical_metadata(out_dir: Path, canonical_meta: Dict[str, Any]) -> None:
    with (out_dir / "canonical_cycle_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(canonical_meta), f, indent=2)


def save_sampled_points(out_dir: Path, case_data: CaseData, sampled_points: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        out_dir / "sampled_points.npz",
        case_id=np.asarray(str(case_data.cfg.save.case_id)),
        re=np.asarray(float(case_data.cfg.flow.re), dtype=np.float32),
        num_cylinders=np.asarray(int(case_data.cfg.layout.num_cylinders), dtype=np.int32),
        cylinder_centers=np.asarray(case_data.cfg.layout.centers, dtype=np.float32),
        **sampled_points,
    )


def save_quicklook_plots(
    out_dir: Path,
    case_data: CaseData,
    mean_field: np.ndarray,
    rms_field: np.ndarray,
    signal_values: np.ndarray,
    tau: np.ndarray,
    phase_unwrapped: Optional[np.ndarray] = None,
    canonical_cycle: Optional[np.ndarray] = None,
    canonical_meta: Optional[Dict[str, Any]] = None,
    legacy_cycle: Optional[np.ndarray] = None,
) -> None:
    quicklook_dir = out_dir / "quicklooks"
    quicklook_dir.mkdir(parents=True, exist_ok=True)
    extent = (0.0, case_data.cfg.domain.lx, 0.0, case_data.cfg.domain.ly)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True, dpi=140)
    channel_titles = ["u", "v", "p", "omega"]
    channel_cmaps = ["coolwarm", "coolwarm", "magma", "RdBu_r"]

    for idx, (title, cmap) in enumerate(zip(channel_titles, channel_cmaps)):
        im0 = axes[0, idx].imshow(mean_field[..., idx], origin="lower", extent=extent, cmap=cmap, aspect="equal")
        axes[0, idx].set_title(f"Mean {title}")
        fig.colorbar(im0, ax=axes[0, idx], fraction=0.046, pad=0.04)

        im1 = axes[1, idx].imshow(rms_field[..., idx], origin="lower", extent=extent, cmap="viridis", aspect="equal")
        axes[1, idx].set_title(f"RMS {title}")
        fig.colorbar(im1, ax=axes[1, idx], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.savefig(quicklook_dir / "mean_and_rms_fields.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True, dpi=140, sharex=True)
    axes[0].plot(case_data.times, signal_values, lw=1.8)
    axes[0].set_ylabel("probe v")
    axes[0].set_title("Probe signal")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(case_data.times, tau, lw=1.8)
    axes[1].set_ylabel("tau")
    axes[1].set_title("Estimated phase")
    axes[1].grid(True, alpha=0.3)

    if phase_unwrapped is not None:
        axes[2].plot(case_data.times, phase_unwrapped, lw=1.8)
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("unwrapped phase")
    axes[2].set_title("Unwrapped phase")
    axes[2].grid(True, alpha=0.3)

    if canonical_meta is not None:
        start_time = canonical_meta.get("source_start_time")
        end_time = canonical_meta.get("source_end_time")
        if start_time is not None and end_time is not None:
            for ax in axes:
                ax.axvspan(float(start_time), float(end_time), color="tab:green", alpha=0.16)
    fig.savefig(quicklook_dir / "phase_signal.png")
    plt.close(fig)

    if canonical_cycle is None:
        return

    raw_jump_mean = raw_jump_max = 0.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, dpi=140)
    if case_data.tensor.shape[0] > 1 and case_data.tensor.shape[-1] > 3:
        omega = case_data.tensor[..., 3]
        raw_jumps_t = torch.mean((omega[1:] - omega[:-1]).square(), dim=tuple(range(1, omega.ndim)))
        raw_jumps = raw_jumps_t.detach().cpu().numpy()
        raw_jump_mean = float(raw_jumps.mean())
        raw_jump_max = float(raw_jumps.max())
        axes[0].plot(np.arange(raw_jumps.size), raw_jumps, lw=1.4, label="raw time order")
        axes[0].set_ylabel("omega jump MSE")
        axes[0].set_title(f"Raw consecutive jumps | mean={raw_jump_mean:.3e}, max={raw_jump_max:.3e}")
        axes[0].grid(True, alpha=0.3)
        if canonical_meta is not None:
            start_idx = int(canonical_meta.get("source_start_index", -1))
            end_idx = int(canonical_meta.get("source_end_index", -1))
            if 0 <= start_idx <= end_idx:
                axes[0].axvspan(start_idx, max(start_idx, end_idx - 1), color="tab:green", alpha=0.16)

    canonical_omega = canonical_cycle[..., 3] if canonical_cycle.shape[-1] > 3 else canonical_cycle[..., 0]
    if canonical_omega.shape[0] > 1:
        canonical_jumps = np.mean((canonical_omega[1:] - canonical_omega[:-1]) ** 2, axis=tuple(range(1, canonical_omega.ndim)))
    else:
        canonical_jumps = np.zeros((0,), dtype=np.float32)
    if canonical_jumps.size:
        axes[1].plot(np.arange(canonical_jumps.size), canonical_jumps, lw=1.4, label="contiguous canonical")
    if legacy_cycle is not None:
        legacy_omega = legacy_cycle[..., 3] if legacy_cycle.shape[-1] > 3 else legacy_cycle[..., 0]
        if legacy_omega.shape[0] > 1:
            legacy_jumps = np.mean((legacy_omega[1:] - legacy_omega[:-1]) ** 2, axis=tuple(range(1, legacy_omega.ndim)))
            axes[1].plot(np.arange(legacy_jumps.size), legacy_jumps, lw=1.1, alpha=0.75, label="legacy tau sort")
    can_mean = float(canonical_jumps.mean()) if canonical_jumps.size else 0.0
    can_max = float(canonical_jumps.max()) if canonical_jumps.size else 0.0
    axes[1].set_xlabel("phase-bin edge index")
    axes[1].set_ylabel("omega jump MSE")
    axes[1].set_title(f"Canonical consecutive jumps | mean={can_mean:.3e}, max={can_max:.3e}")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.savefig(quicklook_dir / "canonical_continuity.png")
    plt.close(fig)


def write_case_to_h5(
    h5_file: h5py.File,
    case_data: CaseData,
    probe: ProbeInfo,
    tau: np.ndarray,
    phase_unwrapped: np.ndarray,
    dominant_frequency: float,
    mean_field: np.ndarray,
    rms_field: np.ndarray,
    canonical_cycle: np.ndarray,
    phase_centers: np.ndarray,
    sampled_points: Dict[str, np.ndarray],
    pod: Optional[Dict[str, np.ndarray]],
    save_full_canonical_cycles: bool,
    save_cycles: int,
    phase_quality: Optional[Dict[str, float]] = None,
    canonical_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one case into the packed HDF5 dataset."""
    cases_group = h5_file.require_group("cases")
    case_id = str(case_data.cfg.save.case_id)
    if case_id in cases_group:
        del cases_group[case_id]

    grp = cases_group.create_group(case_id)
    grp.attrs["case_dir_name"] = case_data.record.case_dir.name
    grp.attrs["split"] = case_data.record.split
    grp.attrs["re"] = float(case_data.cfg.flow.re)
    grp.attrs["num_cylinders"] = int(case_data.cfg.layout.num_cylinders)
    grp.attrs["dominant_frequency"] = float(dominant_frequency)
    grp.attrs["probe_cylinder_index"] = int(probe.cylinder_index)
    grp.attrs["source_case_dir"] = str(case_data.record.case_dir.resolve())
    grp.attrs["save_cycles"] = int(save_cycles)
    if canonical_meta is not None:
        grp.attrs["canonical_method"] = str(canonical_meta.get("canonical_method", "unknown"))
        grp.attrs["canonical_cycle_metadata_json"] = json.dumps(_json_sanitize(canonical_meta))
    if phase_quality is not None:
        grp.attrs["phase_quality_json"] = json.dumps(_json_sanitize(phase_quality))

    grp.create_dataset("times", data=case_data.times, compression="gzip")
    grp.create_dataset("tau", data=tau.astype(np.float32), compression="gzip")
    grp.create_dataset("phase_unwrapped", data=phase_unwrapped.astype(np.float32), compression="gzip")
    grp.create_dataset("cylinder_centers", data=np.asarray(case_data.cfg.layout.centers, dtype=np.float32))
    grp.create_dataset("mean_field", data=mean_field.astype(np.float32), compression="gzip")
    grp.create_dataset("rms_field", data=rms_field.astype(np.float32), compression="gzip")
    grp.create_dataset("channel_order", data=np.asarray(FIELD_CHANNEL_ORDER, dtype=h5py.string_dtype(encoding="utf-8")))
    grp.create_dataset("x_grid", data=case_data.x_grid.astype(np.float32), compression="gzip")
    grp.create_dataset("y_grid", data=case_data.y_grid.astype(np.float32), compression="gzip")

    if case_data.cylinder_mask is not None:
        grp.create_dataset("cylinder_mask", data=case_data.cylinder_mask.astype(np.float32), compression="gzip")

    if save_full_canonical_cycles:
        grp.create_dataset("canonical_cycle", data=canonical_cycle.astype(np.float32), compression="gzip")
        grp.create_dataset("phase_bin_centers", data=phase_centers.astype(np.float32), compression="gzip")

    sampled_group = grp.create_group("sampled_points")
    for key, values in sampled_points.items():
        sampled_group.create_dataset(key, data=values, compression="gzip")

    if pod is not None:
        pod_group = grp.create_group("pod")
        for key, values in pod.items():
            pod_group.create_dataset(key, data=values, compression="gzip")


# ---------------------------- Per-case pipeline --------------------------------


def process_case(
    case_data: CaseData,
    args: argparse.Namespace,
    h5_file: h5py.File,
    progress_bar: Optional[tqdm] = None,
) -> Dict[str, object]:
    out_dir = case_output_dir(args.output_root, case_data.record)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_label = f"case_id={case_data.cfg.save.case_id}, case_dir={case_data.record.case_dir.name}"

    report_state(f"[INFO] [{case_label}] Computing mean, residual, and RMS fields.", progress_bar)

    mean_field_t = torch.mean(case_data.tensor, dim=0)
    residual_t = case_data.tensor - mean_field_t.unsqueeze(0)
    rms_field_t = torch.sqrt(torch.mean(residual_t.square(), dim=0))

    report_state(f"[INFO] [{case_label}] Estimating phase signal and dominant frequency.", progress_bar)
    tau, phase_unwrapped, dominant_frequency, probe, signal_values = estimate_phase(case_data)
    phase_quality = compute_phase_quality(case_data.times, tau, phase_unwrapped, tensor=case_data.tensor)

    median_dt = float(phase_quality.get("median_dt", 0.0))
    tau_sort_jump = float(phase_quality.get("largest_time_jump_after_tau_sort", 0.0))
    max_tau_sort_jump = float(args.max_allowed_tau_sort_time_jump_factor) * median_dt if median_dt > 0.0 else float("inf")
    if float(phase_quality.get("phase_unwrapped_negative_diffs", 0.0)) > 0.0:
        report_state(
            (
                f"[WARN] case {case_data.cfg.save.case_id}: Hilbert phase is nonmonotone "
                f"({int(phase_quality['phase_unwrapped_negative_diffs'])} negative steps)."
            ),
            progress_bar,
        )
    if tau_sort_jump > max_tau_sort_jump:
        report_state(
            (
                f"[WARN] case {case_data.cfg.save.case_id}: folded-tau sorting would create a "
                f"{tau_sort_jump:.3g}s adjacent time jump (> {args.max_allowed_tau_sort_time_jump_factor:.1f}x median dt)."
            ),
            progress_bar,
        )

    report_state(f"[INFO] [{case_label}] Building canonical cycle with {args.phase_bins} phase bins.", progress_bar)
    phase_centers, canonical_cycle_t, canonical_meta = build_canonical_cycle(
        case_data.tensor,
        tau,
        args.phase_bins,
        times=case_data.times,
        phase_unwrapped=phase_unwrapped,
        signal_values=signal_values,
        frame_ids=case_data.frame_ids,
        method=args.canonical_cycle_method,
        min_cycle_frames=int(args.min_cycle_frames),
        prefer_late_cycle=bool(args.prefer_late_cycle),
    )
    canonical_meta["phase_quality"] = phase_quality
    report_state(
        (
            f"[INFO] [{case_label}] Canonical cycle method={canonical_meta.get('canonical_method')} "
            f"frames={canonical_meta.get('source_start_frame')}-{canonical_meta.get('source_end_frame')} "
            f"duration={float(canonical_meta.get('source_duration', 0.0)):.3g}."
        ),
        progress_bar,
    )
    raw_jump_mean = float(phase_quality.get("raw_consecutive_omega_jump_mse_mean", 0.0))
    closure = float(canonical_meta.get("closure_mse_omega", canonical_meta.get("canonical_closure_mse_omega", 0.0)))
    if raw_jump_mean > 0.0 and closure > 25.0 * raw_jump_mean:
        report_state(
            (
                f"[WARN] case {case_data.cfg.save.case_id}: selected cycle closure is high "
                f"(omega MSE={closure:.3e}, raw jump mean={raw_jump_mean:.3e})."
            ),
            progress_bar,
        )
    if str(canonical_meta.get("canonical_method", "")).endswith("fallback") or args.canonical_cycle_method == "time_full_span":
        report_state(
            f"[WARN] case {case_data.cfg.save.case_id}: using canonical-cycle fallback method {canonical_meta.get('canonical_method')}.",
            progress_bar,
        )

    report_state(f"[INFO] [{case_label}] Computing behavior statistics.", progress_bar)
    behavior = compute_case_statistics(case_data, mean_field_t, residual_t, rms_field_t)
    if args.compute_pod:
        report_state(f"[INFO] [{case_label}] Computing POD descriptors with rank={args.pod_rank}.", progress_bar)
        pod = compute_pod_snapshots(residual_t, rank=args.pod_rank)
    else:
        pod = None

    mean_field = behavior["mean_field"]
    rms_field = behavior["rms_field"]
    canonical_cycle = canonical_cycle_t.detach().cpu().numpy().astype(np.float32, copy=False)
    legacy_cycle_for_quicklook = None
    if args.quicklook and args.canonical_diagnostic and args.canonical_cycle_method != "legacy_tau_sort":
        _, legacy_cycle_t, legacy_meta = build_canonical_cycle(
            case_data.tensor,
            tau,
            args.phase_bins,
            method="legacy_tau_sort",
        )
        legacy_cycle_for_quicklook = legacy_cycle_t.detach().cpu().numpy().astype(np.float32, copy=False)
        if legacy_cycle_for_quicklook.shape[0] > 1 and legacy_cycle_for_quicklook.shape[-1] > 3:
            legacy_omega = legacy_cycle_for_quicklook[..., 3]
            legacy_jumps = np.mean((legacy_omega[1:] - legacy_omega[:-1]) ** 2, axis=tuple(range(1, legacy_omega.ndim)))
            canonical_meta["legacy_tau_sort_omega_jump_mse_mean"] = float(legacy_jumps.mean())
            canonical_meta["legacy_tau_sort_omega_jump_mse_max"] = float(legacy_jumps.max())
        canonical_meta["legacy_tau_sort_metadata"] = legacy_meta

    canonical_cycle = np.tile(canonical_cycle, (int(args.save_cycles), 1, 1, 1))
    phase_centers = np.concatenate(
        [phase_centers + float(cycle_idx) for cycle_idx in range(int(args.save_cycles))],
        axis=0,
    ).astype(np.float32, copy=False)

    report_state(
        f"[INFO] [{case_label}] Sampling neural-field points using mode='{args.sampling_mode}'.",
        progress_bar,
    )
    sampled_points = sample_phase_points(
        canonical_cycle=canonical_cycle,
        phase_centers=phase_centers,
        case_data=case_data,
        points_per_phase_bin=args.points_per_phase_bin,
        sampling_mode=args.sampling_mode,
        annulus_ratio=args.annulus_ratio,
        wake_ratio=args.wake_ratio,
    )

    report_state(f"[INFO] [{case_label}] Saving processed outputs to {out_dir}.", progress_bar)
    save_structure_json(
        out_dir,
        case_data,
        probe,
        dominant_frequency,
        save_cycles=args.save_cycles,
        phase_quality=phase_quality,
        canonical_meta=canonical_meta,
    )
    save_behavior_summary(out_dir, behavior, pod)
    save_canonical_cycle(out_dir, canonical_cycle, phase_centers, save_full=args.save_full_canonical_cycles)
    if args.canonical_diagnostic:
        save_canonical_metadata(out_dir, canonical_meta)
    save_phase_metadata(
        out_dir,
        case_data,
        tau,
        phase_unwrapped,
        signal_values,
        dominant_frequency,
        phase_quality=phase_quality,
        canonical_meta=canonical_meta,
        save_diagnostics=bool(args.canonical_diagnostic),
    )
    save_sampled_points(out_dir, case_data, sampled_points)

    if args.quicklook:
        report_state(f"[INFO] [{case_label}] Writing quicklook plots.", progress_bar)
        save_quicklook_plots(
            out_dir,
            case_data,
            mean_field,
            rms_field,
            signal_values,
            tau,
            phase_unwrapped=phase_unwrapped,
            canonical_cycle=canonical_cycle,
            canonical_meta=canonical_meta,
            legacy_cycle=legacy_cycle_for_quicklook,
        )

    report_state(f"[INFO] [{case_label}] Appending case to packed HDF5 dataset.", progress_bar)
    write_case_to_h5(
        h5_file=h5_file,
        case_data=case_data,
        probe=probe,
        tau=tau,
        phase_unwrapped=phase_unwrapped,
        dominant_frequency=dominant_frequency,
        mean_field=mean_field,
        rms_field=rms_field,
        canonical_cycle=canonical_cycle,
        phase_centers=phase_centers,
        sampled_points=sampled_points,
        pod=pod,
        save_full_canonical_cycles=args.save_full_canonical_cycles,
        save_cycles=args.save_cycles,
        phase_quality=phase_quality,
        canonical_meta=canonical_meta,
    )

    report_state(
        f"[INFO] [{case_label}] Finished. frames={case_data.tensor.shape[0]}, sampled_points={sampled_points['tau'].size}.",
        progress_bar,
    )

    return {
        "case_id": str(case_data.cfg.save.case_id),
        "case_dir_name": case_data.record.case_dir.name,
        "split": case_data.record.split,
        "re": float(case_data.cfg.flow.re),
        "num_cylinders": int(case_data.cfg.layout.num_cylinders),
        "num_frames": int(case_data.tensor.shape[0]),
        "phase_bins": int(args.phase_bins),
        "save_cycles": int(args.save_cycles),
        "dominant_frequency": float(dominant_frequency),
        "canonical_method": str(canonical_meta.get("canonical_method", "")),
        "canonical_source_start_frame": int(canonical_meta.get("source_start_frame", -1)),
        "canonical_source_end_frame": int(canonical_meta.get("source_end_frame", -1)),
        "canonical_closure_mse_omega": float(canonical_meta.get("closure_mse_omega", canonical_meta.get("canonical_closure_mse_omega", float("nan")))),
        "phase_unwrapped_negative_diffs": int(phase_quality.get("phase_unwrapped_negative_diffs", 0)),
        "largest_time_jump_after_tau_sort": float(phase_quality.get("largest_time_jump_after_tau_sort", 0.0)),
        "fluctuation_energy": float(behavior["fluctuation_energy"][0]),
        "sampled_points": int(sampled_points["tau"].size),
        "output_dir": str(out_dir.resolve()),
    }


# ----------------------------------- Main --------------------------------------


def main() -> None:
    global args_cache
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = resolve_data_path(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[INFO] Using torch device: {device}")
    print(f"[INFO] Reading raw cases from: {args.input_root}")
    print(f"[INFO] Writing processed dataset to: {args.output_root}")

    args_cache = {
        "annulus_outer_radius_factor": float(args.annulus_outer_radius_factor),
        "wake_length_diameters": float(args.wake_length_diameters),
        "wake_half_width_diameters": float(args.wake_half_width_diameters),
    }

    case_records = discover_case_directories(args.input_root)
    if not case_records:
        raise FileNotFoundError(f"No case directories found under input root: {args.input_root}")
    print(f"[INFO] Discovered {len(case_records)} case directories to preprocess.")

    packed_h5_path = args.output_root / "packed_dataset.h5"
    if packed_h5_path.exists() and args.overwrite:
        print(f"[INFO] Overwrite enabled. Removing existing packed dataset: {packed_h5_path}")
        packed_h5_path.unlink()

    global_index_rows: List[Dict[str, object]] = []
    skipped_cases: List[Tuple[str, str]] = []

    with h5py.File(packed_h5_path, "a") as h5_file:
        h5_file.attrs["dataset_type"] = "inert_multicylinder_periodic_attractor"
        h5_file.attrs["phase_bins"] = int(args.phase_bins)
        h5_file.attrs["save_cycles"] = int(args.save_cycles)
        h5_file.attrs["sampling_mode"] = args.sampling_mode
        h5_file.attrs["channel_order"] = json.dumps(FIELD_CHANNEL_ORDER)
        h5_file.attrs["input_root"] = str(args.input_root)
        h5_file.attrs["output_root"] = str(args.output_root)
        h5_file.attrs["canonical_cycle_method"] = str(args.canonical_cycle_method)

        with tqdm(
            total=len(case_records),
            desc="Preprocessing cases",
            unit="case",
            disable=not progress_enabled(),
        ) as case_bar:
            for record in case_records:
                case_bar.set_postfix_str(f"{record.split}/{record.case_dir.name}")
                report_state(
                    f"[INFO] Loading case: split={record.split}, case_dir={record.case_dir.name}",
                    case_bar,
                )
                case_data = load_case(record, device=device)
                if case_data is None:
                    skipped_cases.append((record.case_dir.name, "load_failed"))
                    report_state(f"[WARN] Skipped case after load failure: {record.case_dir.name}", case_bar)
                    case_bar.update(1)
                    continue

                report_state(
                    (
                        f"[INFO] Loaded {record.case_dir.name}: frames={case_data.tensor.shape[0]}, "
                        f"grid={case_data.tensor.shape[1]}x{case_data.tensor.shape[2]}, "
                        f"channels={case_data.tensor.shape[3]}."
                    ),
                    case_bar,
                )

                try:
                    row = process_case(case_data, args, h5_file, progress_bar=case_bar)
                    global_index_rows.append(row)
                except Exception as exc:
                    report_state(f"[WARN] Failed to preprocess {record.case_dir.name}: {exc}", case_bar)
                    skipped_cases.append((record.case_dir.name, str(exc)))
                finally:
                    case_bar.update(1)

        index_group = h5_file.require_group("global_index")
        if "rows" in index_group:
            del index_group["rows"]
        index_payload = pd.DataFrame(global_index_rows).to_json(orient="records")
        index_group.create_dataset("rows", data=np.asarray(index_payload, dtype=h5py.string_dtype(encoding="utf-8")))

    global_index_df = pd.DataFrame(global_index_rows)
    global_index_df.to_csv(args.output_root / "global_case_index.csv", index=False)

    if skipped_cases:
        skipped_df = pd.DataFrame(skipped_cases, columns=["case_dir_name", "reason"])
        skipped_df.to_csv(args.output_root / "skipped_cases.csv", index=False)
        print(f"[INFO] Preprocessing finished with {len(skipped_cases)} skipped cases.")
    else:
        print("[INFO] Preprocessing finished without skipped cases.")

    print(f"[INFO] Wrote packed dataset to: {packed_h5_path}")


if __name__ == "__main__":
    main()
