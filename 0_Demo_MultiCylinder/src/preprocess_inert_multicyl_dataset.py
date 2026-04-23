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
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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


def build_canonical_cycle(tensor: torch.Tensor, tau: np.ndarray, num_bins: int) -> Tuple[np.ndarray, torch.Tensor]:
    """Interpolate the case trajectory onto a uniform phase grid."""
    num_frames, height, width, channels = tensor.shape
    flat = tensor.reshape(num_frames, -1)
    phase_centers = (np.arange(num_bins, dtype=np.float32) + 0.5) / float(num_bins)

    order = np.argsort(tau)
    tau_sorted = tau[order].astype(np.float64)
    order_torch = torch.as_tensor(order, device=tensor.device, dtype=torch.long)
    flat_sorted = flat.index_select(0, order_torch)

    if np.allclose(tau_sorted, tau_sorted[0]):
        canonical = flat_sorted[:1].repeat(num_bins, 1)
        return phase_centers, canonical.reshape(num_bins, height, width, channels)

    tau_ext = np.concatenate([tau_sorted[-1:] - 1.0, tau_sorted, tau_sorted[:1] + 1.0])
    flat_ext = torch.cat([flat_sorted[-1:], flat_sorted, flat_sorted[:1]], dim=0)

    canonical_flat = torch.empty((num_bins, flat.shape[1]), dtype=flat.dtype, device=tensor.device)
    for i, tau_bin in enumerate(phase_centers):
        right = int(np.searchsorted(tau_ext, tau_bin, side="right"))
        left = max(0, right - 1)
        if right >= len(tau_ext):
            right = len(tau_ext) - 1

        tau_left = tau_ext[left]
        tau_right = tau_ext[right]
        if tau_right <= tau_left + 1e-12:
            canonical_flat[i] = flat_ext[left]
        else:
            weight = float((tau_bin - tau_left) / (tau_right - tau_left))
            canonical_flat[i] = ((1.0 - weight) * flat_ext[left]) + (weight * flat_ext[right])

    return phase_centers, canonical_flat.reshape(num_bins, height, width, channels)


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
) -> None:
    df = pd.DataFrame(
        {
            "saved_frame": case_data.frame_ids,
            "time": case_data.times,
            "tau": tau,
            "phase_unwrapped": phase_unwrapped,
            "probe_signal_v": signal_values,
            "dominant_frequency": np.full_like(tau, dominant_frequency, dtype=np.float32),
        }
    )
    df.to_csv(out_dir / "phase_metadata.csv", index=False)


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

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, dpi=140)
    axes[0].plot(case_data.times, signal_values, lw=1.8)
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("probe v")
    axes[0].set_title("Probe signal")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(case_data.times, tau, lw=1.8)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("tau")
    axes[1].set_title("Estimated phase")
    axes[1].grid(True, alpha=0.3)
    fig.savefig(quicklook_dir / "phase_signal.png")
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

    report_state(f"[INFO] [{case_label}] Building canonical cycle with {args.phase_bins} phase bins.", progress_bar)
    phase_centers, canonical_cycle_t = build_canonical_cycle(case_data.tensor, tau, args.phase_bins)

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
    save_structure_json(out_dir, case_data, probe, dominant_frequency, save_cycles=args.save_cycles)
    save_behavior_summary(out_dir, behavior, pod)
    save_canonical_cycle(out_dir, canonical_cycle, phase_centers, save_full=args.save_full_canonical_cycles)
    save_phase_metadata(out_dir, case_data, tau, phase_unwrapped, signal_values, dominant_frequency)
    save_sampled_points(out_dir, case_data, sampled_points)

    if args.quicklook:
        report_state(f"[INFO] [{case_label}] Writing quicklook plots.", progress_bar)
        save_quicklook_plots(out_dir, case_data, mean_field, rms_field, signal_values, tau)

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
