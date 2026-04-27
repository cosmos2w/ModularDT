from __future__ import annotations

"""Training script for the hypergraph-organized neural field model.

This revision trains one primary architecture:
organizer -> behavior / dynamic memory -> hierarchical mean/residual decoder.

The training objective is intentionally residual-focused. Field and residual
reconstruction stay central, mean supervision is optional, and checkpoint
selection can prioritize dynamic quality through a residual-aware metric.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model import HypergraphNeuralFieldModel, ModelConfig, build_model_from_config


# ------------------------------ Utility helpers -------------------------------

DEMO_ROOT = Path(__file__).resolve().parent.parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hypergraph-organized neural field model.")
    parser.add_argument("--config", type=str, default="train_config_template.json", 
                        help="JSON config file name or path.")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Torch device override, for example cpu, cuda, cuda:0.")
    return parser.parse_args()

def default_config_train_dir() -> Path:
    return (DEMO_ROOT / "Config_Train").resolve()


def default_saved_model_dir() -> Path:
    return (DEMO_ROOT / "Saved_Model").resolve()


def resolve_demo_config_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (DEMO_ROOT / path).resolve()


def resolve_train_config_path(config_name_or_path: str) -> Path:
    path = Path(config_name_or_path)
    if path.is_absolute() or path.exists():
        return path.expanduser().resolve()
    return (default_config_train_dir() / config_name_or_path).resolve()


def sort_case_ids(case_ids: Iterable[str]) -> List[str]:
    def key_fn(case_id: str) -> Tuple[int, object]:
        try:
            return (0, int(case_id))
        except (TypeError, ValueError):
            return (1, str(case_id))

    return sorted(case_ids, key=key_fn)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def normalize_loss_scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def round_to_significant_figures(value: torch.Tensor | float, digits: int = 6) -> float:
    scalar = normalize_loss_scalar(value)
    if not math.isfinite(scalar) or scalar == 0.0:
        return scalar
    return float(f"{scalar:.{digits}g}")


def round_loss_metrics(metrics: Dict[str, float], *, digits: int = 6) -> Dict[str, float]:
    return {key: round_to_significant_figures(value, digits) for key, value in metrics.items()}


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def format_metrics(metrics: Dict[str, float], *, keys: Sequence[str]) -> str:
    pieces = []
    for key in keys:
        value = metrics.get(key, float("nan"))
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            pieces.append(f"{key}={float(value):.4e}")
        else:
            pieces.append(f"{key}=nan")
    return ", ".join(pieces)


def format_large_int(value: int) -> str:
    return f"{int(value):,}"


def make_grad_scaler(device: torch.device, enabled: bool):
    amp_enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def autocast_context(device: torch.device, enabled: bool):
    amp_enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=amp_enabled)
    return torch.cuda.amp.autocast(enabled=amp_enabled)


# ------------------------------ Dataset classes -------------------------------


@dataclass
class CaseChunkMeta:
    """Metadata describing one trainable chunk from the HDF5 dataset."""

    case_id: str
    split: str
    start_idx: int
    end_idx: int
    num_points: int
    num_cylinders: int
    re_value: float


class PackedPointChunkDataset(Dataset):
    """Chunked point-sample dataset built from packed_dataset.h5.

    Each dataset item corresponds to a chunk of point samples from a single case.
    Grouping points by case allows the organizer to run once per batch item while
    the neural field decoder evaluates many query points [Q] for that case.
    """

    def __init__(
        self,
        h5_path: Path,
        split: str,
        *,
        is_training: bool,
        points_per_item: int,
        max_num_cylinders: int,
        train_point_fraction: float = 1.0,
        min_points_per_sample: int = 1,
        resample_each_epoch: bool = False,
        base_seed: int = 42,
        randomize_cylinder_order: bool = False,
        point_sampling_mode: str = "uniform",
        wake_uniform_fraction: float = 0.35,
        wake_near_cylinder_fraction: float = 0.20,
        wake_downstream_fraction: float = 0.30,
        wake_high_omega_fraction: float = 0.15,
        wake_near_radius: float = 1.25,
        wake_sigma_y_base: float = 0.50,
        wake_sigma_y_growth: float = 0.20,
        wake_streamwise_decay: float = 8.0,
        wake_high_omega_quantile: float = 0.75,
        wake_sampling_train_only: bool = True,
        use_phase_window_batches: bool = False,
        phase_window_train_only: bool = True,
        phase_window_fraction: float = 0.25,
        phase_window_num_phases: int = 4,
        phase_window_num_xy: int = 256,
        phase_window_phase_mode: str = "stratified_cycle",
        phase_window_xy_sampling: str = "omega_variance",
        phase_window_require_canonical_cycle: bool = False,
    ):
        super().__init__()
        self.h5_path = Path(h5_path).expanduser().resolve()
        self.split = split
        self.is_training = bool(is_training)
        self.points_per_item = int(points_per_item)
        self.max_num_cylinders = int(max_num_cylinders)
        self.train_point_fraction = float(train_point_fraction)
        self.min_points_per_sample = int(min_points_per_sample)
        self.resample_each_epoch = bool(resample_each_epoch)
        self.base_seed = int(base_seed)
        self.randomize_cylinder_order = bool(randomize_cylinder_order)
        
        self.point_sampling_mode = str(point_sampling_mode).strip().lower()
        self.wake_uniform_fraction = float(wake_uniform_fraction)
        self.wake_near_cylinder_fraction = float(wake_near_cylinder_fraction)
        self.wake_downstream_fraction = float(wake_downstream_fraction)
        self.wake_high_omega_fraction = float(wake_high_omega_fraction)
        self.wake_near_radius = float(wake_near_radius)
        self.wake_sigma_y_base = float(wake_sigma_y_base)
        self.wake_sigma_y_growth = float(wake_sigma_y_growth)
        self.wake_streamwise_decay = float(wake_streamwise_decay)
        self.wake_high_omega_quantile = float(wake_high_omega_quantile)
        self.wake_sampling_train_only = bool(wake_sampling_train_only)
        
        self.use_phase_window_batches = bool(use_phase_window_batches)
        self.phase_window_train_only = bool(phase_window_train_only)
        self.phase_window_fraction = float(phase_window_fraction)
        self.phase_window_num_phases = int(phase_window_num_phases)
        self.phase_window_num_xy = int(phase_window_num_xy)
        self.phase_window_phase_mode = str(phase_window_phase_mode).strip().lower()
        self.phase_window_xy_sampling = str(phase_window_xy_sampling).strip().lower()
        self.phase_window_require_canonical_cycle = bool(phase_window_require_canonical_cycle)
        self.current_epoch = 0
        self._h5: Optional[h5py.File] = None

        if not (0.0 < self.train_point_fraction <= 1.0):
            raise ValueError("train_point_fraction must be in (0, 1].")
        if self.min_points_per_sample < 1:
            raise ValueError("min_points_per_sample must be >= 1.")
        if self.point_sampling_mode not in {"uniform", "wake_focused"}:
            raise ValueError("point_sampling_mode must be 'uniform' or 'wake_focused'.")
        if self.phase_window_num_phases < 1:
            raise ValueError("phase_window_num_phases must be >= 1.")
        if self.phase_window_num_xy < 1:
            raise ValueError("phase_window_num_xy must be >= 1.")
        if self.phase_window_phase_mode not in {"stratified_cycle", "local_window"}:
            raise ValueError("phase_window_phase_mode must be 'stratified_cycle' or 'local_window'.")
        if self.phase_window_xy_sampling not in {"uniform", "omega_variance", "wake_focused"}:
            raise ValueError("phase_window_xy_sampling must be 'uniform', 'omega_variance', or 'wake_focused'.")

        self.case_meta: List[CaseChunkMeta] = []
        self.case_ids: List[str] = []
        self.case_lookup: Dict[str, Dict] = {}

        with h5py.File(self.h5_path, "r") as h5_file:
            cases_group = h5_file["cases"]
            for case_id in sort_case_ids(cases_group.keys()):
                grp = cases_group[case_id]
                case_split = grp.attrs.get("split", "all")
                if split not in {"all", case_split}:
                    continue

                centers = np.asarray(grp["cylinder_centers"], dtype=np.float32)
                if centers.shape[0] > self.max_num_cylinders:
                    raise ValueError(
                        f"Case {case_id} has {centers.shape[0]} cylinders but max_num_cylinders={self.max_num_cylinders}."
                    )

                sampled = grp["sampled_points"]
                num_points = int(sampled["tau"].shape[0])
                re_value = float(grp.attrs["re"])
                num_cylinders = int(grp.attrs["num_cylinders"])
                freq = float(grp.attrs["dominant_frequency"])
                mean_field = np.asarray(grp["mean_field"], dtype=np.float32)
                x_grid = np.asarray(grp["x_grid"], dtype=np.float32)
                y_grid = np.asarray(grp["y_grid"], dtype=np.float32)
                dx = float(np.mean(np.diff(x_grid[0]))) if x_grid.shape[1] > 1 else 1.0
                dy = float(np.mean(np.diff(y_grid[:, 0]))) if y_grid.shape[0] > 1 else 1.0
                x0 = float(x_grid[0, 0])
                y0 = float(y_grid[0, 0])
                lx = float((x_grid.max() - x_grid.min()) + dx)
                ly = float((y_grid.max() - y_grid.min()) + dy)
                has_canonical_cycle = "canonical_cycle" in grp and "phase_bin_centers" in grp
                phase_bin_centers = (
                    np.asarray(grp["phase_bin_centers"], dtype=np.float32) if has_canonical_cycle else None
                )

                self.case_lookup[case_id] = {
                    "centers": centers,
                    "re": re_value,
                    "num_cylinders": num_cylinders,
                    "freq": freq,
                    "mean_field": mean_field,
                    "x_grid": x_grid,
                    "y_grid": y_grid,
                    "grid_origin": (x0, y0),
                    "grid_spacing": (dx, dy),
                    "domain_lengths": (lx, ly),
                    "domain_origin": (float(x_grid.min()), float(y_grid.min())),
                    "has_canonical_cycle": has_canonical_cycle,
                    "canonical_cycle": None,
                    "phase_bin_centers": phase_bin_centers,
                }
                self.case_ids.append(case_id)

                if self.points_per_item <= 0:
                    self.case_meta.append(
                        CaseChunkMeta(
                            case_id=case_id,
                            split=case_split,
                            start_idx=0,
                            end_idx=num_points,
                            num_points=num_points,
                            num_cylinders=num_cylinders,
                            re_value=re_value,
                        )
                    )
                else:
                    for start in range(0, num_points, self.points_per_item):
                        end = min(start + self.points_per_item, num_points)
                        self.case_meta.append(
                            CaseChunkMeta(
                                case_id=case_id,
                                split=case_split,
                                start_idx=start,
                                end_idx=end,
                                num_points=(end - start),
                                num_cylinders=num_cylinders,
                                re_value=re_value,
                            )
                        )

    def __len__(self) -> int:
        return len(self.case_meta)

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _rng_for_item(self, idx: int, *, include_epoch: bool, salt: int = 0) -> np.random.Generator:
        epoch_offset = self.current_epoch if include_epoch else 0
        rng_seed = self.base_seed + (1000003 * idx) + (9176 * epoch_offset) + int(salt)
        return np.random.default_rng(rng_seed)

    def _compute_keep_count(self, num_points: int) -> int:
        if self.train_point_fraction >= 1.0:
            return num_points
        keep_count = max(self.min_points_per_sample, int(math.ceil(num_points * self.train_point_fraction)))
        return min(keep_count, num_points)

    def _uniform_subsample_indices(
        self,
        idx: int,
        num_points: int,
        keep_count_override: Optional[int] = None,
    ) -> np.ndarray:
        keep_count = self._compute_keep_count(num_points) if keep_count_override is None else int(keep_count_override)
        keep_count = max(0, min(keep_count, num_points))
        if keep_count <= 0:
            return np.empty((0,), dtype=np.int64)
        if keep_count >= num_points:
            return np.arange(num_points, dtype=np.int64)
        rng = self._rng_for_item(idx, include_epoch=self.resample_each_epoch)
        chosen = np.sort(rng.choice(num_points, size=keep_count, replace=False))
        return chosen.astype(np.int64, copy=False)

    @staticmethod
    def _sample_without_replacement(
        rng: np.random.Generator,
        candidate_idx: np.ndarray,
        count: int,
        selected_mask: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if count <= 0 or candidate_idx.size == 0:
            return np.empty((0,), dtype=np.int64)
        available = candidate_idx[~selected_mask[candidate_idx]]
        if available.size == 0:
            return np.empty((0,), dtype=np.int64)

        take = min(int(count), int(available.size))
        probs = None
        if weights is not None:
            weight_values = np.asarray(weights[available], dtype=np.float64)
            weight_values = np.clip(weight_values, a_min=0.0, a_max=None)
            total = float(weight_values.sum())
            if total > 0.0:
                probs = weight_values / total
        picked = rng.choice(available, size=take, replace=False, p=probs)
        selected_mask[picked] = True
        return picked.astype(np.int64, copy=False)

    def _wake_sampling_counts(self, keep_count: int) -> Dict[str, int]:
        fractions = {
            "uniform": max(self.wake_uniform_fraction, 0.0),
            "near": max(self.wake_near_cylinder_fraction, 0.0),
            "downstream": max(self.wake_downstream_fraction, 0.0),
            "high_omega": max(self.wake_high_omega_fraction, 0.0),
        }
        total_fraction = sum(fractions.values())
        if total_fraction <= 0.0:
            return {"uniform": keep_count, "near": 0, "downstream": 0, "high_omega": 0}

        raw = {key: keep_count * value / total_fraction for key, value in fractions.items()}
        counts = {key: int(math.floor(value)) for key, value in raw.items()}
        remainder = keep_count - sum(counts.values())
        for key, _ in sorted(raw.items(), key=lambda item: item[1] - math.floor(item[1]), reverse=True):
            if remainder <= 0:
                break
            counts[key] += 1
            remainder -= 1
        return counts

    def _wake_focused_subsample_indices(
        self,
        idx: int,
        case_static: Dict,
        chunk_x: np.ndarray,
        chunk_y: np.ndarray,
        chunk_omega: np.ndarray,
        keep_count_override: Optional[int] = None,
    ) -> np.ndarray:
        num_points = int(chunk_x.shape[0])
        keep_count = self._compute_keep_count(num_points) if keep_count_override is None else int(keep_count_override)
        keep_count = max(0, min(keep_count, num_points))
        if keep_count <= 0:
            return np.empty((0,), dtype=np.int64)
        if keep_count >= num_points:
            return np.arange(num_points, dtype=np.int64)

        if self.wake_sampling_train_only and not self.is_training:
            return self._uniform_subsample_indices(idx, num_points, keep_count_override=keep_count)

        rng = self._rng_for_item(idx, include_epoch=self.resample_each_epoch)
        selected_mask = np.zeros((num_points,), dtype=bool)
        counts = self._wake_sampling_counts(keep_count)
        centers = np.asarray(case_static["centers"], dtype=np.float32)
        lx, ly = case_static["domain_lengths"]

        point_idx = np.arange(num_points, dtype=np.int64)
        point_x = chunk_x[None, :]
        point_y = chunk_y[None, :]
        center_x = centers[:, 0:1]
        center_y = centers[:, 1:2]

        dist_to_cyl = periodic_distance_min_image_np(center_x, center_y, point_x, point_y, lx, ly)
        min_dist = dist_to_cyl.min(axis=0)
        near_pool = np.flatnonzero(min_dist <= self.wake_near_radius)

        dx_down = directed_periodic_downstream_delta_np(center_x, point_x, lx)
        dy = periodic_delta_min_image_np(center_y, point_y, ly)
        sigma_y = self.wake_sigma_y_base + self.wake_sigma_y_growth * dx_down
        wake_score = np.exp(-0.5 * (dy / np.maximum(sigma_y, 1e-4)) ** 2) * np.exp(
            -dx_down / max(self.wake_streamwise_decay, 1e-6)
        )
        wake_score = wake_score * (dx_down > 0.0)
        wake_score_max = wake_score.max(axis=0)
        downstream_pool = np.flatnonzero(wake_score_max > 1e-6)

        omega_mag = np.abs(chunk_omega)
        omega_threshold = float(np.quantile(omega_mag, self.wake_high_omega_quantile))
        high_omega_pool = np.flatnonzero(omega_mag >= omega_threshold)

        chosen = [
            self._sample_without_replacement(rng, point_idx, counts["uniform"], selected_mask),
            self._sample_without_replacement(rng, near_pool, counts["near"], selected_mask),
            self._sample_without_replacement(
                rng,
                downstream_pool,
                counts["downstream"],
                selected_mask,
                weights=wake_score_max,
            ),
            self._sample_without_replacement(
                rng,
                high_omega_pool,
                counts["high_omega"],
                selected_mask,
                weights=omega_mag,
            ),
        ]

        chosen_idx = np.concatenate([arr for arr in chosen if arr.size > 0], axis=0) if any(arr.size > 0 for arr in chosen) else np.empty((0,), dtype=np.int64)
        if chosen_idx.size < keep_count:
            fill = self._sample_without_replacement(rng, point_idx, keep_count - chosen_idx.size, selected_mask)
            if fill.size > 0:
                chosen_idx = np.concatenate([chosen_idx, fill], axis=0)

        chosen_idx = np.unique(chosen_idx.astype(np.int64, copy=False))
        if chosen_idx.size < keep_count:
            remaining = point_idx[~selected_mask]
            extra = remaining[: keep_count - chosen_idx.size]
            chosen_idx = np.concatenate([chosen_idx, extra], axis=0)
        return np.sort(chosen_idx[:keep_count]).astype(np.int64, copy=False)

    def _subsample_indices(
        self,
        idx: int,
        case_static: Dict,
        chunk_x: np.ndarray,
        chunk_y: np.ndarray,
        chunk_omega: np.ndarray,
        keep_count_override: Optional[int] = None,
    ) -> np.ndarray:
        if self.point_sampling_mode != "wake_focused":
            return self._uniform_subsample_indices(idx, int(chunk_x.shape[0]), keep_count_override=keep_count_override)
        return self._wake_focused_subsample_indices(
            idx,
            case_static,
            chunk_x,
            chunk_y,
            chunk_omega,
            keep_count_override=keep_count_override,
        )

    @staticmethod
    def _weighted_grid_choice(
        rng: np.random.Generator,
        weights: Optional[np.ndarray],
        count: int,
        num_cells: int,
    ) -> np.ndarray:
        if count <= 0 or num_cells <= 0:
            return np.empty((0,), dtype=np.int64)
        take = min(int(count), int(num_cells))
        probs = None
        if weights is not None:
            flat = np.asarray(weights, dtype=np.float64).reshape(-1)
            if flat.shape[0] != num_cells:
                raise ValueError("phase-window grid weights must match the flattened grid size.")
            flat = np.clip(flat, a_min=0.0, a_max=None)
            total = float(flat.sum())
            if total > 0.0:
                probs = flat / total
        return rng.choice(num_cells, size=take, replace=False, p=probs).astype(np.int64, copy=False)

    def _phase_window_xy_weights(self, case_static: Dict, canonical_cycle: np.ndarray) -> Optional[np.ndarray]:
        mode = self.phase_window_xy_sampling
        if mode == "uniform":
            return None

        omega_variance = np.var(canonical_cycle[..., 3], axis=0).astype(np.float64, copy=False)
        if mode == "omega_variance":
            return omega_variance

        x_grid = np.asarray(case_static["x_grid"], dtype=np.float32)
        y_grid = np.asarray(case_static["y_grid"], dtype=np.float32)
        centers = np.asarray(case_static["centers"], dtype=np.float32)
        if centers.size == 0:
            return omega_variance
        lx, ly = case_static["domain_lengths"]

        flat_x = x_grid.reshape(-1)[None, :]
        flat_y = y_grid.reshape(-1)[None, :]
        center_x = centers[:, 0:1]
        center_y = centers[:, 1:2]

        dist_to_cyl = periodic_distance_min_image_np(center_x, center_y, flat_x, flat_y, lx, ly)
        near_score = np.exp(-0.5 * (dist_to_cyl / max(self.wake_near_radius, 1e-6)) ** 2).max(axis=0)

        dx_down = directed_periodic_downstream_delta_np(center_x, flat_x, lx)
        dy = periodic_delta_min_image_np(center_y, flat_y, ly)
        sigma_y = self.wake_sigma_y_base + self.wake_sigma_y_growth * dx_down
        wake_score = np.exp(-0.5 * (dy / np.maximum(sigma_y, 1e-4)) ** 2) * np.exp(
            -dx_down / max(self.wake_streamwise_decay, 1e-6)
        )
        wake_score = (wake_score * (dx_down > 0.0)).max(axis=0)

        dyn = omega_variance.reshape(-1)
        dyn_norm = dyn / max(float(dyn.max()), 1e-12)
        weights = dyn_norm + 0.5 * near_score + wake_score
        return weights.reshape(x_grid.shape)

    def _sample_phase_window_points(
        self,
        idx: int,
        case_static: Dict,
        h5_case_group: h5py.Group,
        requested_count: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        if requested_count <= 0:
            return None

        canonical_cycle = case_static.get("canonical_cycle")
        phase_bin_centers = case_static.get("phase_bin_centers")
        if canonical_cycle is None and "canonical_cycle" in h5_case_group and "phase_bin_centers" in h5_case_group:
            canonical_cycle = np.asarray(h5_case_group["canonical_cycle"], dtype=np.float32)
            phase_bin_centers = np.asarray(h5_case_group["phase_bin_centers"], dtype=np.float32)

        if canonical_cycle is None or phase_bin_centers is None:
            if self.phase_window_require_canonical_cycle:
                raise RuntimeError(
                    f"Case is missing canonical_cycle/phase_bin_centers, required for phase-window batches."
                )
            return None

        canonical_cycle = np.asarray(canonical_cycle, dtype=np.float32)
        phase_bin_centers = np.asarray(phase_bin_centers, dtype=np.float32).reshape(-1)
        if canonical_cycle.ndim != 4 or canonical_cycle.shape[-1] != 4:
            raise ValueError("canonical_cycle must have shape [phase, H, W, 4].")

        num_phase_bins = int(canonical_cycle.shape[0])
        if num_phase_bins <= 0:
            return None

        n_phases = max(1, min(int(self.phase_window_num_phases), num_phase_bins))
        n_xy = min(int(self.phase_window_num_xy), int(requested_count) // n_phases)
        height, width = canonical_cycle.shape[1:3]
        num_cells = int(height * width)
        n_xy = min(n_xy, num_cells)
        if n_xy <= 0:
            return None

        rng = self._rng_for_item(idx, include_epoch=self.resample_each_epoch, salt=7919)
        base = int(rng.integers(0, num_phase_bins))
        if self.phase_window_phase_mode == "stratified_cycle":
            stride = max(1, num_phase_bins // n_phases)
            phase_idx = (base + np.arange(n_phases, dtype=np.int64) * stride) % num_phase_bins
        else:
            phase_idx = (base + np.arange(n_phases, dtype=np.int64)) % num_phase_bins

        weights = self._phase_window_xy_weights(case_static, canonical_cycle)
        flat_idx = self._weighted_grid_choice(rng, weights, n_xy, num_cells)
        if flat_idx.size == 0:
            return None
        iy, ix = np.unravel_index(flat_idx, (height, width))

        x_grid = np.asarray(case_static["x_grid"], dtype=np.float32)
        y_grid = np.asarray(case_static["y_grid"], dtype=np.float32)
        mean_field = np.asarray(case_static["mean_field"], dtype=np.float32)

        xy_count = int(flat_idx.size)
        repeated_ix = np.repeat(ix, n_phases)
        repeated_iy = np.repeat(iy, n_phases)
        tiled_phase = np.tile(phase_idx, xy_count)

        x = x_grid[repeated_iy, repeated_ix].astype(np.float32, copy=False)
        y = y_grid[repeated_iy, repeated_ix].astype(np.float32, copy=False)
        tau = phase_bin_centers[tiled_phase].astype(np.float32, copy=False)
        targets = canonical_cycle[tiled_phase, repeated_iy, repeated_ix, :].astype(np.float32, copy=False)
        mean_targets = mean_field[repeated_iy, repeated_ix, :].astype(np.float32, copy=False)
        residual_targets = targets - mean_targets
        phase_window_mask = np.ones((targets.shape[0],), dtype=np.float32)

        return {
            "x": x,
            "y": y,
            "tau": tau,
            "targets": targets,
            "mean_targets": mean_targets,
            "residual_targets": residual_targets,
            "phase_window_mask": phase_window_mask,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.case_meta[idx]
        h5_file = self._get_h5()
        grp = h5_file["cases"][meta.case_id]
        sampled = grp["sampled_points"]
        sl = slice(meta.start_idx, meta.end_idx)
        case_static = self.case_lookup[meta.case_id]

        chunk_x = np.asarray(sampled["x"][sl], dtype=np.float32)
        chunk_y = np.asarray(sampled["y"][sl], dtype=np.float32)
        chunk_tau = np.asarray(sampled["tau"][sl], dtype=np.float32)
        chunk_u = np.asarray(sampled["u"][sl], dtype=np.float32)
        chunk_v = np.asarray(sampled["v"][sl], dtype=np.float32)
        chunk_p = np.asarray(sampled["p"][sl], dtype=np.float32)
        chunk_omega = np.asarray(sampled["omega"][sl], dtype=np.float32)

        total_keep_count = self._compute_keep_count(int(chunk_x.shape[0]))
        phase_window_points = None
        phase_window_active = self.use_phase_window_batches and (
            self.is_training or not self.phase_window_train_only
        )
        if phase_window_active:
            fraction = min(max(self.phase_window_fraction, 0.0), 1.0)
            window_count = int(total_keep_count * fraction)
            n_phases = max(1, int(self.phase_window_num_phases))
            window_count = (window_count // n_phases) * n_phases
            phase_window_points = self._sample_phase_window_points(
                idx,
                case_static,
                grp,
                requested_count=window_count,
            )

        window_actual = (
            int(phase_window_points["targets"].shape[0])
            if phase_window_points is not None
            else 0
        )
        normal_count = max(0, total_keep_count - window_actual)
        local_indices = self._subsample_indices(
            idx,
            case_static,
            chunk_x,
            chunk_y,
            chunk_omega,
            keep_count_override=normal_count,
        )

        x_normal = chunk_x[local_indices]
        y_normal = chunk_y[local_indices]
        tau_normal = chunk_tau[local_indices]
        targets_normal = np.stack(
            [
                chunk_u[local_indices],
                chunk_v[local_indices],
                chunk_p[local_indices],
                chunk_omega[local_indices],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

        centers = np.asarray(case_static["centers"], dtype=np.float32)
        centers_valid = centers.copy()
        if self.is_training and self.randomize_cylinder_order and centers_valid.shape[0] > 1:
            rng = self._rng_for_item(idx, include_epoch=True)
            centers_valid = centers_valid[rng.permutation(centers_valid.shape[0])]

        padded_centers = np.zeros((self.max_num_cylinders, 2), dtype=np.float32)
        cyl_mask = np.zeros((self.max_num_cylinders,), dtype=np.float32)
        padded_centers[: centers_valid.shape[0]] = centers_valid
        cyl_mask[: centers_valid.shape[0]] = 1.0

        # Sample mean-field targets at the same spatial points using nearest grid lookup.
        mean_field = case_static["mean_field"]
        x0, y0 = case_static["grid_origin"]
        dx, dy = case_static["grid_spacing"]
        ix = np.clip(np.rint((x_normal - x0) / max(dx, 1e-6)).astype(np.int64), 0, mean_field.shape[1] - 1)
        iy = np.clip(np.rint((y_normal - y0) / max(dy, 1e-6)).astype(np.int64), 0, mean_field.shape[0] - 1)
        mean_targets_normal = mean_field[iy, ix].astype(np.float32, copy=False)
        residual_targets_normal = targets_normal - mean_targets_normal

        x_parts = [x_normal.astype(np.float32, copy=False)]
        y_parts = [y_normal.astype(np.float32, copy=False)]
        tau_parts = [tau_normal.astype(np.float32, copy=False)]
        target_parts = [targets_normal]
        mean_parts = [mean_targets_normal]
        residual_parts = [residual_targets_normal]
        phase_window_parts = [np.zeros((x_normal.shape[0],), dtype=np.float32)]
        if phase_window_points is not None:
            x_parts.append(phase_window_points["x"])
            y_parts.append(phase_window_points["y"])
            tau_parts.append(phase_window_points["tau"])
            target_parts.append(phase_window_points["targets"])
            mean_parts.append(phase_window_points["mean_targets"])
            residual_parts.append(phase_window_points["residual_targets"])
            phase_window_parts.append(phase_window_points["phase_window_mask"])

        x = np.concatenate(x_parts, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(y_parts, axis=0).astype(np.float32, copy=False)
        tau = np.concatenate(tau_parts, axis=0).astype(np.float32, copy=False)
        targets = np.concatenate(target_parts, axis=0).astype(np.float32, copy=False)
        mean_targets = np.concatenate(mean_parts, axis=0).astype(np.float32, copy=False)
        residual_targets = np.concatenate(residual_parts, axis=0).astype(np.float32, copy=False)
        phase_window_mask = np.concatenate(phase_window_parts, axis=0).astype(np.float32, copy=False)

        if x.shape[0] > 1:
            rng = self._rng_for_item(idx, include_epoch=self.resample_each_epoch, salt=1543)
            order = rng.permutation(x.shape[0])
            x = x[order]
            y = y[order]
            tau = tau[order]
            targets = targets[order]
            mean_targets = mean_targets[order]
            residual_targets = residual_targets[order]
            phase_window_mask = phase_window_mask[order]

        return {
            "case_id": meta.case_id,
            "re_values": torch.tensor([meta.re_value], dtype=torch.float32),
            "num_cylinders": torch.tensor([meta.num_cylinders], dtype=torch.float32),
            "centers": torch.from_numpy(padded_centers),
            "cyl_mask": torch.from_numpy(cyl_mask),
            "query_xy": torch.from_numpy(np.stack([x, y], axis=-1)),
            "query_tau": torch.from_numpy(tau[:, None]),
            "field_targets": torch.from_numpy(targets),
            "mean_targets": torch.from_numpy(mean_targets),
            "residual_targets": torch.from_numpy(residual_targets),
            "phase_window_mask": torch.from_numpy(phase_window_mask),
            "freq_target": torch.tensor([case_static["freq"]], dtype=torch.float32),
        }

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


class CanonicalCycleValidationDataset(Dataset):
    """Case-level dataset used for validation / evaluation on canonical cycles."""

    def __init__(self, h5_path: Path, split: str):
        self.h5_path = Path(h5_path).expanduser().resolve()
        self.split = split
        self.case_ids: List[str] = []
        with h5py.File(self.h5_path, "r") as h5_file:
            cases_group = h5_file["cases"]
            for case_id in sort_case_ids(cases_group.keys()):
                grp = cases_group[case_id]
                case_split = grp.attrs.get("split", "all")
                if split in {"all", case_split}:
                    if "canonical_cycle" in grp:
                        self.case_ids.append(case_id)

    def __len__(self) -> int:
        return len(self.case_ids)

    def get_case(self, case_id: str) -> Dict:
        with h5py.File(self.h5_path, "r") as h5_file:
            grp = h5_file["cases"][case_id]
            centers = np.asarray(grp["cylinder_centers"], dtype=np.float32)
            return {
                "case_id": case_id,
                "re": float(grp.attrs["re"]),
                "num_cylinders": int(grp.attrs["num_cylinders"]),
                "dominant_frequency": float(grp.attrs["dominant_frequency"]),
                "centers": centers,
                "canonical_cycle": np.asarray(grp["canonical_cycle"], dtype=np.float32),
                "phase_bin_centers": np.asarray(grp["phase_bin_centers"], dtype=np.float32),
                "x_grid": np.asarray(grp["x_grid"], dtype=np.float32),
                "y_grid": np.asarray(grp["y_grid"], dtype=np.float32),
                "mean_field": np.asarray(grp["mean_field"], dtype=np.float32),
            }


# ------------------------------ Collation logic --------------------------------


def collate_point_chunks(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[str]]:
    """Collate variable-Q point chunks into dense batch tensors.

    Output shapes:
        re_values:      [B, 1]
        num_cylinders:  [B, 1]
        centers:        [B, N_max, 2]
        cyl_mask:       [B, N_max]
        query_xy:       [B, Q_max, 2]
        query_tau:      [B, Q_max, 1]
        field_targets:  [B, Q_max, 4]
        mean_targets:   [B, Q_max, 4]
        residual_targets:[B, Q_max, 4]
        phase_window_mask:[B, Q_max]
        point_mask:     [B, Q_max]
        freq_target:    [B, 1]
    """
    max_points = max(item["query_xy"].shape[0] for item in batch)
    batch_size = len(batch)
    n_max = batch[0]["centers"].shape[0]

    def pad_points(tensor: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        q = tensor.shape[0]
        if q == max_points:
            return tensor
        pad_shape = (max_points - q, *tensor.shape[1:])
        pad = torch.full(pad_shape, fill_value, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=0)

    out: Dict[str, torch.Tensor | List[str]] = {
        "case_id": [item["case_id"] for item in batch],
        "re_values": torch.stack([item["re_values"] for item in batch], dim=0),
        "num_cylinders": torch.stack([item["num_cylinders"] for item in batch], dim=0),
        "centers": torch.stack([item["centers"] for item in batch], dim=0),
        "cyl_mask": torch.stack([item["cyl_mask"] for item in batch], dim=0),
        "query_xy": torch.stack([pad_points(item["query_xy"]) for item in batch], dim=0),
        "query_tau": torch.stack([pad_points(item["query_tau"]) for item in batch], dim=0),
        "field_targets": torch.stack([pad_points(item["field_targets"]) for item in batch], dim=0),
        "mean_targets": torch.stack([pad_points(item["mean_targets"]) for item in batch], dim=0),
        "residual_targets": torch.stack([pad_points(item["residual_targets"]) for item in batch], dim=0),
        "phase_window_mask": torch.stack(
            [
                pad_points(item.get("phase_window_mask", torch.zeros(item["query_xy"].shape[0], dtype=torch.float32)))
                for item in batch
            ],
            dim=0,
        ),
        "freq_target": torch.stack([item["freq_target"] for item in batch], dim=0),
    }

    point_mask = torch.zeros((batch_size, max_points), dtype=torch.float32)
    for i, item in enumerate(batch):
        point_mask[i, : item["query_xy"].shape[0]] = 1.0
    out["point_mask"] = point_mask
    return out


# ------------------------------- Loss functions --------------------------------


def compute_dynamic_energy_loss(
    pred_residual: torch.Tensor,
    target_residual: torch.Tensor,
    point_mask: torch.Tensor,
    channels: Sequence[int],
    eps: float,
    log_space: bool,
    use_smooth_l1: bool,
) -> torch.Tensor:
    channel_list = [int(ch) for ch in channels]
    if not channel_list:
        return pred_residual.new_zeros(())
    if min(channel_list) < 0 or max(channel_list) >= pred_residual.shape[-1]:
        raise ValueError(
            f"dynamic_energy_channels={channel_list} is incompatible with residual channel count "
            f"{pred_residual.shape[-1]}."
        )

    energy_dtype = torch.float32 if pred_residual.dtype in {torch.float16, torch.bfloat16} else pred_residual.dtype
    selected_pred = pred_residual[..., channel_list].to(dtype=energy_dtype)
    selected_tgt = target_residual[..., channel_list].to(device=pred_residual.device, dtype=energy_dtype)
    mask = point_mask.to(device=pred_residual.device, dtype=energy_dtype)
    if mask.ndim == pred_residual.ndim - 1:
        mask = mask.unsqueeze(-1)

    denom = mask.sum(dim=1).clamp_min(1.0) * float(len(channel_list))
    pred_energy = (selected_pred.square() * mask).sum(dim=(1, 2)) / denom.squeeze(-1)
    tgt_energy = (selected_tgt.square() * mask).sum(dim=(1, 2)) / denom.squeeze(-1)

    if log_space:
        pred_metric = torch.log(pred_energy + float(eps))
        tgt_metric = torch.log(tgt_energy + float(eps))
    else:
        pred_metric = pred_energy
        tgt_metric = tgt_energy

    if use_smooth_l1:
        return F.smooth_l1_loss(pred_metric, tgt_metric)
    return F.mse_loss(pred_metric, tgt_metric)


def compute_losses(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    *,
    organizer_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    pred_field = outputs["pred_field"]
    pred_mean = outputs["pred_mean"]
    pred_residual = outputs["pred_residual"]
    freq_pred = outputs["freq_pred"]

    point_mask_2d = batch["point_mask"].to(device=pred_field.device, dtype=pred_field.dtype)
    point_mask = point_mask_2d.unsqueeze(-1)  # [B, Q, 1]

    field_targets = batch["field_targets"]
    mean_targets = batch["mean_targets"]
    residual_targets = batch["residual_targets"]
    freq_target = batch["freq_target"]

    def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = ((pred - target) ** 2) * point_mask
        denom = point_mask.sum().clamp_min(1.0) * pred.shape[-1]
        return diff.sum() / denom

    loss_field = masked_mse(pred_field, field_targets)
    loss_mean = masked_mse(pred_mean, mean_targets)
    loss_residual = masked_mse(pred_residual, residual_targets)
    loss_freq = F.mse_loss(freq_pred, freq_target)
    loss_dynamic_energy = compute_dynamic_energy_loss(
        pred_residual,
        residual_targets,
        point_mask_2d,
        channels=loss_cfg.get("dynamic_energy_channels", [3]),
        eps=float(loss_cfg.get("dynamic_energy_eps", 1.0e-8)),
        log_space=bool(loss_cfg.get("dynamic_energy_log_space", True)),
        use_smooth_l1=bool(loss_cfg.get("dynamic_energy_use_smooth_l1", True)),
    )

    loss_org_concentration, loss_org_entropy = compute_organizer_regularization(
        outputs,
        batch["cyl_mask"],
        device=pred_field.device,
        dtype=pred_field.dtype,
    )

    base_total = combine_weighted_loss_terms(
        loss_cfg,
        loss_field=loss_field,
        loss_mean=loss_mean,
        loss_residual=loss_residual,
        loss_freq=loss_freq,
        loss_dynamic_energy=loss_dynamic_energy,
        loss_org_sparsity=loss_org_concentration,
        loss_org_entropy=loss_org_entropy,
    )

    # Direct organizer supervision (already implemented in this file, but previously unused).
    org_direct = organizer_direct_losses(
        outputs,
        batch,
        me_weight=float(loss_cfg.get("organizer_me_weight", 0.0)),
        mm_weight=float(loss_cfg.get("organizer_mm_weight", 0.0)),
        consistency_weight=float(loss_cfg.get("organizer_consistency_weight", 0.0)),
        eh_factor_weight=float(loss_cfg.get("organizer_eh_factor_weight", 0.0)),
        eh_factor_target_source=str(loss_cfg.get("organizer_eh_factor_target_source", "prior_me")),
        eh_factor_detach_mh=bool(loss_cfg.get("organizer_eh_factor_detach_mh", True)),
        eh_factor_eps=float(loss_cfg.get("organizer_eh_factor_eps", 1.0e-6)),
        mass_align_weight=float(loss_cfg.get("organizer_mass_align_weight", 0.0)),
        mass_align_log_space=bool(loss_cfg.get("organizer_mass_align_log_space", True)),
        mass_align_detach_module=bool(loss_cfg.get("organizer_mass_align_detach_module", True)),
        mass_align_eps=float(loss_cfg.get("organizer_mass_align_eps", 1.0e-6)),
    )
    org_diag = organizer_mass_diagnostics(
        outputs,
        batch,
        eps=float(loss_cfg.get("organizer_mass_align_eps", 1.0e-6)),
    )

    total = base_total + float(organizer_scale) * org_direct["organizer_total"]

    return {
        "loss_total": total,
        "loss_field": loss_field,
        "loss_mean": loss_mean,
        "loss_residual": loss_residual,
        "loss_freq": loss_freq,
        "loss_dynamic_energy": loss_dynamic_energy,
        "loss_org_sparsity": loss_org_concentration,
        "loss_org_entropy": loss_org_entropy,
        "loss_org_direct": org_direct["organizer_total"],
        "loss_org_me": org_direct["organizer_me"],
        "loss_org_mm": org_direct["organizer_mm"],
        "loss_org_consistency": org_direct["organizer_consistency"],
        "loss_org_eh_factor": org_direct["organizer_eh_factor"],
        "loss_org_mass_align": org_direct["organizer_mass_align"],
        **org_diag,
    }


def compute_organizer_regularization(
    outputs: Dict[str, torch.Tensor],
    cyl_mask: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        concentration_loss: lower when assignments are sharper / more selective
        entropy_loss: lower when assignments are lower-entropy
    """
    row_mask = cyl_mask.to(device=device, dtype=dtype)
    A_me = outputs["A_me"].clamp_min(1e-8)  # [B, N, M]
    A_mh = outputs["A_mh"].clamp_min(1e-8)  # [B, N, K]
    A_eh = outputs["A_eh"].clamp_min(1e-8)  # [B, M, K]

    def masked_mean_over_rows(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        while mask.ndim < values.ndim - 1:
            mask = mask.unsqueeze(-1)
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom

    # Concentration surrogate: 1 - sum(p^2), lower when assignments are peaky.
    me_conc = 1.0 - (A_me.square().sum(dim=-1))          # [B, N]
    mh_conc = 1.0 - (A_mh.square().sum(dim=-1))          # [B, N]
    eh_conc = 1.0 - (A_eh.square().sum(dim=-1)).mean()   # scalar-like over [B, M]

    concentration_loss = (
        masked_mean_over_rows(me_conc, row_mask)
        + masked_mean_over_rows(mh_conc, row_mask)
        + eh_conc
    )

    def masked_entropy(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ent = -(values * values.log()).sum(dim=-1)  # sum over assignment dimension
        return masked_mean_over_rows(ent, mask)

    entropy_loss = (
        masked_entropy(A_me, row_mask)
        + masked_entropy(A_mh, row_mask)
        + (-(A_eh * A_eh.log()).sum(dim=-1)).mean()
    )

    return concentration_loss, entropy_loss


def combine_weighted_loss_terms(
    loss_cfg: Dict,
    *,
    loss_field: torch.Tensor | float,
    loss_mean: torch.Tensor | float,
    loss_residual: torch.Tensor | float,
    loss_freq: torch.Tensor | float,
    loss_dynamic_energy: torch.Tensor | float,
    loss_org_sparsity: torch.Tensor | float,
    loss_org_entropy: torch.Tensor | float,
) -> torch.Tensor | float:
    return (
        float(loss_cfg.get("field_mse_weight", 1.0)) * loss_field
        + float(loss_cfg.get("mean_mse_weight", 0.0)) * loss_mean
        + float(loss_cfg.get("residual_mse_weight", 1.0)) * loss_residual
        + float(loss_cfg.get("freq_mse_weight", 0.05)) * loss_freq
        + float(loss_cfg.get("dynamic_energy_weight", 0.0)) * loss_dynamic_energy
        + float(loss_cfg.get("organizer_sparsity_weight", 0.0)) * loss_org_sparsity
        + float(loss_cfg.get("organizer_entropy_weight", 0.0)) * loss_org_entropy
    )


def average_metric_dicts(metric_dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    out: Dict[str, float] = {}
    keys = metric_dicts[0].keys()
    for key in keys:
        vals = [float(row[key]) for row in metric_dicts]
        out[key] = float(sum(vals) / len(vals))
    return out


def estimate_effective_points_per_item(points_per_item: int, point_fraction: float, min_points_per_sample: int) -> int:
    if points_per_item <= 0:
        return max(min_points_per_sample, 1)
    if point_fraction >= 1.0:
        return points_per_item
    return min(points_per_item, max(min_points_per_sample, int(math.ceil(points_per_item * point_fraction))))


def find_nonfinite_tensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, int]:
    bad: Dict[str, int] = {}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        nonfinite = ~torch.isfinite(tensor)
        count = int(nonfinite.sum().item())
        if count > 0:
            bad[name] = count
    return bad


def periodic_delta_min_image_np(src: np.ndarray, dst: np.ndarray, period: float) -> np.ndarray:
    return np.remainder(dst - src + 0.5 * period, period) - 0.5 * period


def periodic_distance_min_image_np(
    src_x: np.ndarray,
    src_y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
    lx: float,
    ly: float,
) -> np.ndarray:
    dx = periodic_delta_min_image_np(src_x, dst_x, lx)
    dy = periodic_delta_min_image_np(src_y, dst_y, ly)
    return np.sqrt(dx * dx + dy * dy)


def directed_periodic_downstream_delta_np(src_x: np.ndarray, dst_x: np.ndarray, period: float) -> np.ndarray:
    return np.remainder(dst_x - src_x, period)


def pairwise_periodic_relative_features(src_xy: torch.Tensor, dst_xy: torch.Tensor) -> torch.Tensor:
    """
    src_xy: [B, N_src, 2] normalized to [0,1]
    dst_xy: [B, N_dst, 2] normalized to [0,1]
    returns: [B, N_src, N_dst, 5] = dx, dy, dist, downstream, upstream
    """
    dx = dst_xy[:, None, :, 0] - src_xy[:, :, None, 0]
    dy = dst_xy[:, None, :, 1] - src_xy[:, :, None, 1]
    dx = (dx + 0.5) % 1.0 - 0.5
    dy = (dy + 0.5) % 1.0 - 0.5
    dist = torch.sqrt(dx.square() + dy.square() + 1e-8)
    downstream = torch.clamp(dx, min=0.0)
    upstream = torch.clamp(-dx, min=0.0)
    return torch.stack([dx, dy, dist, downstream, upstream], dim=-1)


def normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)


def build_module_env_prior(
    module_coords_norm: torch.Tensor,
    env_coords: torch.Tensor,
    cyl_mask: torch.Tensor,
    re_values: torch.Tensor,
    sigma_y_base: float = 0.05,
    sigma_y_growth: float = 0.20,
    decay_x: float = 0.35,
    near_radius: float = 0.08,) -> torch.Tensor:
    """
    Soft wake-like module->environment prior.
    Output: [B, N, M_env], row-normalized over env tokens.
    """
    rel = pairwise_periodic_relative_features(module_coords_norm, env_coords)
    dx = rel[..., 0]
    dy = rel[..., 1]
    dist = rel[..., 2]
    downstream = rel[..., 3]
    re_values = re_values.to(device=module_coords_norm.device, dtype=module_coords_norm.dtype)
    re_values = re_values.reshape(module_coords_norm.shape[0], -1)[:, 0]

    # Optional mild Re effect on wake width.
    re_scale = (re_values / 200.0).clamp(min=0.1, max=1.5)[:, None, None]
    sigma_y = sigma_y_base + sigma_y_growth * downstream * re_scale

    wake = torch.exp(-0.5 * (dy / sigma_y.clamp_min(1e-4)).square()) * torch.exp(-downstream / decay_x)
    wake = wake * (downstream > 0).to(wake.dtype)

    near = torch.exp(-0.5 * (dist / near_radius).square())
    prior = wake + 0.25 * near

    prior = prior * cyl_mask[:, :, None]
    prior = normalize_rows(prior)
    if prior.ndim != 3:
        raise RuntimeError(f"build_module_env_prior produced invalid shape {tuple(prior.shape)}; expected [B, N, M].")
    return prior


def build_module_module_affinity_prior(
    module_coords_norm: torch.Tensor,
    cyl_mask: torch.Tensor,
    sigma_d: float = 0.16,
    sigma_x: float = 0.25,
    sigma_y: float = 0.12,) -> torch.Tensor:
    """
    Soft symmetric module-module interaction affinity.
    Output: [B, N, N], values in [0,1].
    """
    rel = pairwise_periodic_relative_features(module_coords_norm, module_coords_norm)
    dx = rel[..., 0].abs()
    dy = rel[..., 1].abs()
    dist = rel[..., 2]

    affinity = (
        0.5 * torch.exp(-dist / sigma_d)
        + 0.25 * torch.exp(-dx / sigma_x)
        + 0.25 * torch.exp(-dy / sigma_y)
    )

    bsz, n, _ = affinity.shape
    eye = torch.eye(n, device=affinity.device, dtype=affinity.dtype)[None, :, :]
    valid = cyl_mask[:, :, None] * cyl_mask[:, None, :] * (1.0 - eye)
    affinity = affinity * valid
    return affinity


def compute_hyperedge_masses(
    outputs: Dict[str, torch.Tensor],
    structure: Dict[str, torch.Tensor],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    A_mh = outputs["A_mh"]
    A_eh = outputs["A_eh"]
    cyl_mask = structure["cyl_mask"].to(device=A_mh.device, dtype=A_mh.dtype)

    module_mass_raw = (A_mh * cyl_mask[:, :, None]).sum(dim=1)
    module_mass_raw = module_mass_raw / cyl_mask.sum(dim=1, keepdim=True).clamp_min(eps)
    env_mass_raw = A_eh.mean(dim=1)

    module_mass = module_mass_raw / module_mass_raw.sum(dim=-1, keepdim=True).clamp_min(eps)
    env_mass = env_mass_raw / env_mass_raw.sum(dim=-1, keepdim=True).clamp_min(eps)
    return module_mass_raw, env_mass_raw, module_mass, env_mass


def build_eh_factor_target(
    outputs: Dict[str, torch.Tensor],
    structure: Dict[str, torch.Tensor],
    *,
    target_source: str = "prior_me",
    detach_mh: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    A_me = outputs["A_me"]
    A_mh = outputs["A_mh"]
    module_coords_norm = outputs["module_coords_norm"]
    env_coords = outputs["env_coords"]
    cyl_mask = structure["cyl_mask"].to(device=A_mh.device, dtype=A_mh.dtype)
    re_values = structure["re_values"].to(device=A_mh.device, dtype=A_mh.dtype)

    source = str(target_source).strip().lower()
    if source == "prior_me":
        me_for_target = build_module_env_prior(module_coords_norm, env_coords, cyl_mask, re_values)
    elif source == "learned_me":
        me_for_target = A_me.detach()
    else:
        raise ValueError("organizer_eh_factor_target_source must be 'prior_me' or 'learned_me'.")

    mh = A_mh.detach() if detach_mh else A_mh
    raw = torch.einsum("bnm,bnk->bmk", me_for_target, mh)
    denom = raw.sum(dim=-1, keepdim=True)

    _, _, module_mass, _ = compute_hyperedge_masses(outputs, structure, eps=eps)
    fallback = module_mass[:, None, :].expand_as(raw)
    target_eh = torch.where(denom > eps, raw / denom.clamp_min(eps), fallback)
    target_eh = target_eh.clamp_min(eps)
    return target_eh / target_eh.sum(dim=-1, keepdim=True).clamp_min(eps)


def organizer_eh_factor_loss(
    outputs: Dict[str, torch.Tensor],
    structure: Dict[str, torch.Tensor],
    *,
    target_source: str = "prior_me",
    detach_mh: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    A_eh = outputs["A_eh"].clamp_min(eps)
    target_eh = build_eh_factor_target(
        outputs,
        structure,
        target_source=target_source,
        detach_mh=detach_mh,
        eps=eps,
    ).detach()
    return F.kl_div(A_eh.log(), target_eh, reduction="none").sum(dim=-1).mean()


def organizer_mass_align_loss(
    outputs: Dict[str, torch.Tensor],
    structure: Dict[str, torch.Tensor],
    *,
    log_space: bool = True,
    detach_module: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    _, _, module_mass, env_mass = compute_hyperedge_masses(outputs, structure, eps=eps)
    module_target = module_mass.detach() if detach_module else module_mass
    if log_space:
        return F.smooth_l1_loss(torch.log(env_mass + eps), torch.log(module_target + eps))
    return F.smooth_l1_loss(env_mass, module_target)


def organizer_mass_diagnostics(
    outputs: Dict[str, torch.Tensor],
    structure: Dict[str, torch.Tensor],
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    _, _, module_mass, env_mass = compute_hyperedge_masses(outputs, structure, eps=eps)
    env_entropy = -(env_mass.clamp_min(eps) * env_mass.clamp_min(eps).log()).sum(dim=-1)
    module_entropy = -(module_mass.clamp_min(eps) * module_mass.clamp_min(eps).log()).sum(dim=-1)
    diag = {
        "org_env_mass_max": env_mass.max(dim=-1).values.mean(),
        "org_module_mass_max": module_mass.max(dim=-1).values.mean(),
        "org_mass_l1": (env_mass - module_mass).abs().mean(),
        "org_env_effective_hyperedges": torch.exp(env_entropy).mean(),
        "org_module_effective_hyperedges": torch.exp(module_entropy).mean(),
        "org_env_mass_entropy": env_entropy.mean(),
        "org_module_mass_entropy": module_entropy.mean(),
    }
    active_mask = outputs.get("hyper_active_mask")
    if active_mask is not None:
        active = active_mask.to(device=module_mass.device, dtype=module_mass.dtype)
        collapsed = outputs.get("hyper_collapsed_mask", torch.zeros_like(active, dtype=torch.bool)).to(device=module_mass.device)
        duplicate = outputs.get("hyper_duplicate_mask", torch.zeros_like(active, dtype=torch.bool)).to(device=module_mass.device)
        edge_score = outputs.get("hyper_edge_score", torch.zeros_like(active)).to(device=module_mass.device, dtype=module_mass.dtype)
        diag.update(
            {
                "org_active_hyperedges": active.sum(dim=-1).mean(),
                "org_collapsed_hyperedges": collapsed.to(dtype=module_mass.dtype).sum(dim=-1).mean(),
                "org_duplicate_hyperedges": duplicate.to(dtype=module_mass.dtype).sum(dim=-1).mean(),
                "org_inactive_hyperedges": (1.0 - active).sum(dim=-1).mean(),
                "org_mean_edge_score": edge_score.mean(),
            }
        )
    return diag


def organizer_direct_losses(
    outputs: Dict[str, torch.Tensor],
    structure: Dict[str, torch.Tensor],
    me_weight: float = 0.05,
    mm_weight: float = 0.03,
    consistency_weight: float = 0.05,
    eh_factor_weight: float = 0.05,
    eh_factor_target_source: str = "prior_me",
    eh_factor_detach_mh: bool = True,
    eh_factor_eps: float = 1.0e-6,
    mass_align_weight: float = 0.02,
    mass_align_log_space: bool = True,
    mass_align_detach_module: bool = True,
    mass_align_eps: float = 1.0e-6,) -> Dict[str, torch.Tensor]:
    """
    Direct organizer supervision for inert cases.
    Returns dict of scalar losses.
    """
    A_me = outputs["A_me"]                                # [B, N, M]
    A_mh = outputs["A_mh"]                                # [B, N, K]
    A_eh = outputs["A_eh"]                                # [B, M, K]
    module_coords_norm = outputs["module_coords_norm"]    # [B, N, 2]
    env_coords = outputs["env_coords"]                    # [B, M, 2]
    cyl_mask = structure["cyl_mask"]                      # [B, N]
    re_values = structure["re_values"]                    # [B, 1]

    # A) weak geometry-based module->environment prior
    prior_me = build_module_env_prior(module_coords_norm, env_coords, cyl_mask, re_values)
    if prior_me.shape != A_me.shape:
        raise RuntimeError(
            f"build_module_env_prior shape mismatch: prior_me={tuple(prior_me.shape)} vs A_me={tuple(A_me.shape)}"
        )
    kl_me = F.kl_div((A_me.clamp_min(1e-6)).log(), prior_me, reduction="none").sum(dim=-1)
    loss_me = (kl_me * cyl_mask).sum() / cyl_mask.sum().clamp_min(1.0)

    # B) permutation-safe hyperedge supervision via module-module affinity
    pred_mm = torch.matmul(A_mh, A_mh.transpose(1, 2))  # [B, N, N]
    prior_mm = build_module_module_affinity_prior(module_coords_norm, cyl_mask)
    if prior_mm.shape != pred_mm.shape:
        raise RuntimeError(
            f"build_module_module_affinity_prior shape mismatch: prior_mm={tuple(prior_mm.shape)} vs pred_mm={tuple(pred_mm.shape)}"
        )

    n = pred_mm.shape[1]
    eye = torch.eye(n, device=pred_mm.device, dtype=pred_mm.dtype)[None, :, :]
    valid_mm = cyl_mask[:, :, None] * cyl_mask[:, None, :] * (1.0 - eye)
    loss_mm = (((pred_mm - prior_mm) ** 2) * valid_mm).sum() / valid_mm.sum().clamp_min(1.0)

    # C) hypergraph should factorize module-env organization
    pred_me_from_h = torch.matmul(A_mh, A_eh.transpose(1, 2))  # [B, N, M]
    pred_me_from_h = normalize_rows(pred_me_from_h)
    valid_me = cyl_mask[:, :, None].expand_as(A_me)
    loss_cons = (((pred_me_from_h - A_me) ** 2) * valid_me).sum() / valid_me.sum().clamp_min(1.0)

    # D) direct A_eh supervision: environment token e should choose hyperedge k
    # when modules that influence e also belong to k.
    loss_eh_factor = organizer_eh_factor_loss(
        outputs,
        structure,
        target_source=eh_factor_target_source,
        detach_mh=eh_factor_detach_mh,
        eps=eh_factor_eps,
    )

    # E) discourage one-sided collapse by matching normalized module and
    # environment mass per hyperedge.
    loss_mass_align = organizer_mass_align_loss(
        outputs,
        structure,
        log_space=mass_align_log_space,
        detach_module=mass_align_detach_module,
        eps=mass_align_eps,
    )

    total = (
        me_weight * loss_me
        + mm_weight * loss_mm
        + consistency_weight * loss_cons
        + eh_factor_weight * loss_eh_factor
        + mass_align_weight * loss_mass_align
    )
    return {
        "organizer_total": total,
        "organizer_me": loss_me,
        "organizer_mm": loss_mm,
        "organizer_consistency": loss_cons,
        "organizer_eh_factor": loss_eh_factor,
        "organizer_mass_align": loss_mass_align,
    }

# ---------------------------- Validation routine -------------------------------


def build_structure_tensors(case: Dict, max_num_cylinders: int, device: torch.device) -> Dict[str, torch.Tensor]:
    centers = case["centers"]
    padded = np.zeros((1, max_num_cylinders, 2), dtype=np.float32)
    mask = np.zeros((1, max_num_cylinders), dtype=np.float32)
    padded[0, : centers.shape[0]] = centers
    mask[0, : centers.shape[0]] = 1.0
    return {
        "re_values": torch.tensor([[case["re"]]], dtype=torch.float32, device=device),
        "num_cylinders": torch.tensor([[case["num_cylinders"]]], dtype=torch.float32, device=device),
        "centers": torch.from_numpy(padded).to(device=device),
        "cyl_mask": torch.from_numpy(mask).to(device=device),
    }


@torch.no_grad()
def evaluate_canonical_cases(
    model: nn.Module,
    dataset: CanonicalCycleValidationDataset,
    *,
    device: torch.device,
    loss_cfg: Dict,
    validation_cfg: Dict,
    max_num_cylinders: int,
    max_cases: int,
    query_batch_size: int,
    phase_bins_to_eval: int,
    show_progress: bool = False,) -> Dict[str, float]:
    model.eval()
    if len(dataset) == 0:
        return {
            "val_total_loss": float("nan"),
            "val_field_mse": float("nan"),
            "val_mean_mse": float("nan"),
            "val_residual_mse": float("nan"),
            "val_freq_mse": float("nan"),
            "val_dynamic_energy": float("nan"),
            "val_residual_focus": float("nan"),
            "val_loss_org_eh_factor": float("nan"),
            "val_loss_org_mass_align": float("nan"),
            "val_org_env_mass_max": float("nan"),
            "val_org_module_mass_max": float("nan"),
            "val_org_mass_l1": float("nan"),
            "val_org_env_effective_hyperedges": float("nan"),
            "val_org_module_effective_hyperedges": float("nan"),
            "val_org_active_hyperedges": float("nan"),
            "val_org_collapsed_hyperedges": float("nan"),
            "val_org_duplicate_hyperedges": float("nan"),
            "val_org_inactive_hyperedges": float("nan"),
            "val_org_mean_edge_score": float("nan"),
        }

    case_ids = dataset.case_ids[:max_cases] if max_cases > 0 else dataset.case_ids
    total_losses, field_losses, mean_losses, residual_losses, freq_losses, dynamic_energy_losses = [], [], [], [], [], []
    org_diag_buffer: List[Dict[str, float]] = []

    val_iter = case_ids
    val_bar = None
    if show_progress:
        val_bar = tqdm(
            case_ids,
            desc="validation",
            leave=False,
            dynamic_ncols=True,
        )
        val_iter = val_bar

    for case_id in val_iter:
        case = dataset.get_case(case_id)
        structure = build_structure_tensors(case, max_num_cylinders=max_num_cylinders, device=device)
        x_grid = torch.from_numpy(case["x_grid"]).to(device=device)
        y_grid = torch.from_numpy(case["y_grid"]).to(device=device)
        canonical = torch.from_numpy(case["canonical_cycle"]).to(device=device)
        phase_bins = case["phase_bin_centers"]

        step = max(1, len(phase_bins) // max(phase_bins_to_eval, 1))
        chosen_indices = list(range(0, len(phase_bins), step))[:phase_bins_to_eval]
        pred_fields = []
        gt_fields = []
        for idx in chosen_indices:
            tau_val = torch.tensor([phase_bins[idx]], dtype=torch.float32, device=device)
            out = model.reconstruct_full_grid(structure, x_grid, y_grid, tau=tau_val, query_batch_size=query_batch_size)
            pred_fields.append(out["pred_field"][0])
            gt_fields.append(canonical[idx])
        pred_stack = torch.stack(pred_fields, dim=0)
        gt_stack = torch.stack(gt_fields, dim=0)
        loss_field = F.mse_loss(pred_stack, gt_stack)
        field_losses.append(loss_field.item())

        pred_mean = pred_stack.mean(dim=0)
        gt_mean = gt_stack.mean(dim=0)
        loss_mean = F.mse_loss(pred_mean, gt_mean)
        mean_losses.append(loss_mean.item())

        pred_residual = pred_stack - pred_mean.unsqueeze(0)
        gt_residual = gt_stack - gt_mean.unsqueeze(0)
        loss_residual = F.mse_loss(pred_residual, gt_residual)
        residual_losses.append(loss_residual.item())
        flat_mask = torch.ones(
            (1, pred_residual.shape[0] * pred_residual.shape[1] * pred_residual.shape[2]),
            device=device,
            dtype=pred_residual.dtype,
        )
        loss_dynamic_energy = compute_dynamic_energy_loss(
            pred_residual.reshape(1, -1, pred_residual.shape[-1]),
            gt_residual.reshape(1, -1, gt_residual.shape[-1]),
            flat_mask,
            channels=loss_cfg.get("dynamic_energy_channels", [3]),
            eps=float(loss_cfg.get("dynamic_energy_eps", 1.0e-8)),
            log_space=bool(loss_cfg.get("dynamic_energy_log_space", True)),
            use_smooth_l1=bool(loss_cfg.get("dynamic_energy_use_smooth_l1", True)),
        )
        dynamic_energy_losses.append(loss_dynamic_energy.item())

        # Frequency prediction from the behavior head.
        out_once = model.forward(
            structure=structure,
            query_xy=torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)[None, :1024, :],
            query_tau=torch.full((1, 1024, 1), phase_bins[chosen_indices[0]], device=device),
            return_aux=True,
        )
        loss_freq = F.mse_loss(out_once["freq_pred"], torch.tensor([[case["dominant_frequency"]]], device=device))
        freq_losses.append(loss_freq.item())

        loss_org_sparsity, loss_org_entropy = compute_organizer_regularization(
            out_once,
            structure["cyl_mask"],
            device=device,
            dtype=out_once["freq_pred"].dtype,
        )
        base_total = combine_weighted_loss_terms(
            loss_cfg,
            loss_field=loss_field,
            loss_mean=loss_mean,
            loss_residual=loss_residual,
            loss_freq=loss_freq,
            loss_dynamic_energy=loss_dynamic_energy,
            loss_org_sparsity=loss_org_sparsity,
            loss_org_entropy=loss_org_entropy,
        )
        org_direct = organizer_direct_losses(
            out_once,
            structure,
            me_weight=float(loss_cfg.get("organizer_me_weight", 0.0)),
            mm_weight=float(loss_cfg.get("organizer_mm_weight", 0.0)),
            consistency_weight=float(loss_cfg.get("organizer_consistency_weight", 0.0)),
            eh_factor_weight=float(loss_cfg.get("organizer_eh_factor_weight", 0.0)),
            eh_factor_target_source=str(loss_cfg.get("organizer_eh_factor_target_source", "prior_me")),
            eh_factor_detach_mh=bool(loss_cfg.get("organizer_eh_factor_detach_mh", True)),
            eh_factor_eps=float(loss_cfg.get("organizer_eh_factor_eps", 1.0e-6)),
            mass_align_weight=float(loss_cfg.get("organizer_mass_align_weight", 0.0)),
            mass_align_log_space=bool(loss_cfg.get("organizer_mass_align_log_space", True)),
            mass_align_detach_module=bool(loss_cfg.get("organizer_mass_align_detach_module", True)),
            mass_align_eps=float(loss_cfg.get("organizer_mass_align_eps", 1.0e-6)),
        )
        org_diag = organizer_mass_diagnostics(
            out_once,
            structure,
            eps=float(loss_cfg.get("organizer_mass_align_eps", 1.0e-6)),
        )
        org_diag_buffer.append(
            {
                "val_loss_org_eh_factor": normalize_loss_scalar(org_direct["organizer_eh_factor"]),
                "val_loss_org_mass_align": normalize_loss_scalar(org_direct["organizer_mass_align"]),
                "val_org_env_mass_max": normalize_loss_scalar(org_diag["org_env_mass_max"]),
                "val_org_module_mass_max": normalize_loss_scalar(org_diag["org_module_mass_max"]),
                "val_org_mass_l1": normalize_loss_scalar(org_diag["org_mass_l1"]),
                "val_org_env_effective_hyperedges": normalize_loss_scalar(org_diag["org_env_effective_hyperedges"]),
                "val_org_module_effective_hyperedges": normalize_loss_scalar(org_diag["org_module_effective_hyperedges"]),
                "val_org_active_hyperedges": normalize_loss_scalar(org_diag["org_active_hyperedges"]),
                "val_org_collapsed_hyperedges": normalize_loss_scalar(org_diag["org_collapsed_hyperedges"]),
                "val_org_duplicate_hyperedges": normalize_loss_scalar(org_diag["org_duplicate_hyperedges"]),
                "val_org_inactive_hyperedges": normalize_loss_scalar(org_diag["org_inactive_hyperedges"]),
                "val_org_mean_edge_score": normalize_loss_scalar(org_diag["org_mean_edge_score"]),
            }
        )
        loss_total = base_total + org_direct["organizer_total"]
        total_losses.append(float(loss_total.item() if isinstance(loss_total, torch.Tensor) else loss_total))
        
        if val_bar is not None:
            val_bar.set_postfix(
                total=f"{total_losses[-1]:.3e}",
                field_mse=f"{field_losses[-1]:.3e}",
                mean_mse=f"{mean_losses[-1]:.3e}",
                dynamic_energy=f"{dynamic_energy_losses[-1]:.3e}",
                freq_mse=f"{freq_losses[-1]:.3e}",
            )

    if val_bar is not None:
        val_bar.close()

    metrics = {
        "val_total_loss": float(np.mean(total_losses)),
        "val_field_mse": float(np.mean(field_losses)),
        "val_mean_mse": float(np.mean(mean_losses)),
        "val_residual_mse": float(np.mean(residual_losses)),
        "val_freq_mse": float(np.mean(freq_losses)),
        "val_dynamic_energy": float(np.mean(dynamic_energy_losses)),
    }
    metrics.update(average_metric_dicts(org_diag_buffer))
    metrics["val_residual_focus"] = compute_residual_focus_metric(metrics, validation_cfg)
    return metrics


@torch.no_grad()
def evaluate_point_chunks(
    model: nn.Module,
    val_loader: DataLoader,
    *,
    device: torch.device,
    loss_cfg: Dict,
    validation_cfg: Dict,
    show_progress: bool = False,
) -> Dict[str, float]:
    model.eval()
    if len(val_loader) == 0:
        return {
            "val_total_loss": float("nan"),
            "val_field_mse": float("nan"),
            "val_mean_mse": float("nan"),
            "val_residual_mse": float("nan"),
            "val_freq_mse": float("nan"),
            "val_dynamic_energy": float("nan"),
            "val_residual_focus": float("nan"),
            "val_loss_org_eh_factor": float("nan"),
            "val_loss_org_mass_align": float("nan"),
            "val_org_env_mass_max": float("nan"),
            "val_org_module_mass_max": float("nan"),
            "val_org_mass_l1": float("nan"),
            "val_org_env_effective_hyperedges": float("nan"),
            "val_org_module_effective_hyperedges": float("nan"),
            "val_org_active_hyperedges": float("nan"),
            "val_org_collapsed_hyperedges": float("nan"),
            "val_org_duplicate_hyperedges": float("nan"),
            "val_org_inactive_hyperedges": float("nan"),
            "val_org_mean_edge_score": float("nan"),
        }

    metric_buffer: List[Dict[str, float]] = []
    val_iter = val_loader
    val_bar = None
    if show_progress:
        val_bar = tqdm(
            val_loader,
            total=len(val_loader),
            desc="validation",
            leave=False,
            dynamic_ncols=True,
        )
        val_iter = val_bar

    for batch in val_iter:
        structure = {
            "re_values": batch["re_values"].to(device=device),
            "num_cylinders": batch["num_cylinders"].to(device=device),
            "centers": batch["centers"].to(device=device),
            "cyl_mask": batch["cyl_mask"].to(device=device),
        }
        query_xy = batch["query_xy"].to(device=device)
        query_tau = batch["query_tau"].to(device=device)

        for key in [
            "re_values",
            "field_targets",
            "mean_targets",
            "residual_targets",
            "point_mask",
            "freq_target",
            "cyl_mask",
        ]:
            batch[key] = batch[key].to(device=device)

        outputs = model(structure=structure, query_xy=query_xy, query_tau=query_tau, return_aux=True)
        bad_outputs = find_nonfinite_tensors(
            {
                "pred_field": outputs["pred_field"],
                "pred_mean": outputs["pred_mean"],
                "pred_residual": outputs["pred_residual"],
                "freq_pred": outputs["freq_pred"],
                "A_me": outputs["A_me"],
                "A_mh": outputs["A_mh"],
                "A_eh": outputs["A_eh"],
            }
        )
        if bad_outputs:
            raise RuntimeError(f"Non-finite validation outputs detected: {bad_outputs}")
        losses = compute_losses(
            batch,
            outputs,
            loss_cfg,
            organizer_scale=1.0,
        )
        bad_losses = find_nonfinite_tensors(losses)
        if bad_losses:
            raise RuntimeError(f"Non-finite validation losses detected: {bad_losses}")
        row = {
            "val_total_loss": normalize_loss_scalar(losses["loss_total"]),
            "val_field_mse": normalize_loss_scalar(losses["loss_field"]),
            "val_mean_mse": normalize_loss_scalar(losses["loss_mean"]),
            "val_residual_mse": normalize_loss_scalar(losses["loss_residual"]),
            "val_freq_mse": normalize_loss_scalar(losses["loss_freq"]),
            "val_dynamic_energy": normalize_loss_scalar(losses["loss_dynamic_energy"]),
            "val_loss_org_eh_factor": normalize_loss_scalar(losses["loss_org_eh_factor"]),
            "val_loss_org_mass_align": normalize_loss_scalar(losses["loss_org_mass_align"]),
            "val_org_env_mass_max": normalize_loss_scalar(losses["org_env_mass_max"]),
            "val_org_module_mass_max": normalize_loss_scalar(losses["org_module_mass_max"]),
            "val_org_mass_l1": normalize_loss_scalar(losses["org_mass_l1"]),
            "val_org_env_effective_hyperedges": normalize_loss_scalar(losses["org_env_effective_hyperedges"]),
            "val_org_module_effective_hyperedges": normalize_loss_scalar(losses["org_module_effective_hyperedges"]),
            "val_org_active_hyperedges": normalize_loss_scalar(losses["org_active_hyperedges"]),
            "val_org_collapsed_hyperedges": normalize_loss_scalar(losses["org_collapsed_hyperedges"]),
            "val_org_duplicate_hyperedges": normalize_loss_scalar(losses["org_duplicate_hyperedges"]),
            "val_org_inactive_hyperedges": normalize_loss_scalar(losses["org_inactive_hyperedges"]),
            "val_org_mean_edge_score": normalize_loss_scalar(losses["org_mean_edge_score"]),
        }
        row["val_residual_focus"] = compute_residual_focus_metric(row, validation_cfg)
        metric_buffer.append(row)
        if val_bar is not None:
            val_bar.set_postfix(
                total=f"{row['val_total_loss']:.3e}",
                field_mse=f"{row['val_field_mse']:.3e}",
                residual_mse=f"{row['val_residual_mse']:.3e}",
                dynamic_energy=f"{row['val_dynamic_energy']:.3e}",
                residual_focus=f"{row['val_residual_focus']:.3e}",
                freq_mse=f"{row['val_freq_mse']:.3e}",
            )

    if val_bar is not None:
        val_bar.close()

    return average_metric_dicts(metric_buffer)


def compute_validation_selection_metric(val_metrics: Dict[str, float], validation_cfg: Dict) -> float:
    """Residual-focused checkpoint selection metric.

    Supported names:
        val_residual_focus: val_residual_mse + residual_focus_field_weight * val_field_mse
        any direct metric key present in `val_metrics`
    """

    metric_name = str(validation_cfg.get("best_metric_name", "val_residual_focus")).strip()
    if metric_name == "val_residual_focus":
        return compute_residual_focus_metric(val_metrics, validation_cfg)
    if metric_name in val_metrics:
        return float(val_metrics[metric_name])
    raise ValueError(
        f"Unsupported validation.best_metric_name='{metric_name}'. "
        "Use 'val_residual_focus' or a direct validation metric key."
    )


def organizer_ramp_scale(loss_cfg: Dict, epoch: int) -> float:
    ramp_epochs = max(1, int(loss_cfg.get("organizer_ramp_epochs", 1)))
    return min(1.0, float(epoch) / float(ramp_epochs))


def compute_residual_focus_metric(metrics: Dict[str, float], validation_cfg: Dict) -> float:
    residual = float(metrics.get("val_residual_mse", float("nan")))
    field = float(metrics.get("val_field_mse", float("nan")))
    if not (math.isfinite(residual) and math.isfinite(field)):
        return float("nan")
    field_weight = float(validation_cfg.get("residual_focus_field_weight", 0.25))
    return residual + (field_weight * field)

# ------------------------------- Plotting --------------------------------------


def save_loss_curve(history: List[Dict[str, float]], out_path: Path) -> None:
    if not history:
        return

    epochs = [row["epoch"] for row in history]

    def finite_xy(x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        for x, y in zip(x_values, y_values):
            y_float = float(y)
            if math.isfinite(y_float) and y_float > 0.0:
                xs.append(float(x))
                ys.append(y_float)
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140, sharex=True)

    def plot_panel(
        ax,
        title: str,
        train_key: str,
        val_key: str,
        train_label: str,
        val_label: str,
    ) -> None:
        train_values = [row.get(train_key, float("nan")) for row in history]
        train_epochs, train_vals = finite_xy(epochs, train_values)
        if train_epochs:
            ax.plot(train_epochs, train_vals, linestyle="-", linewidth=1.5, label=train_label)
        val_values = [row.get(val_key, float("nan")) for row in history]
        val_epochs, val_vals = finite_xy(epochs, val_values)
        if val_epochs:
            ax.plot(
                val_epochs,
                val_vals,
                linestyle=":",
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                label=val_label,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend()

    plot_panel(axes[0, 0], "Total Loss", "loss_total", "val_total_loss", "Train total", "Val total")
    plot_panel(axes[0, 1], "Field Loss", "loss_field", "val_field_mse", "Train field", "Val field")
    plot_panel(axes[1, 0], "Residual Loss", "loss_residual", "val_residual_mse", "Train residual", "Val residual")
    plot_panel(
        axes[1, 1],
        "Dynamic Energy Loss",
        "loss_dynamic_energy",
        "val_dynamic_energy",
        "Train dynamic energy",
        "Val dynamic energy",
    )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------- Main training ---------------------------------


def main() -> None:
    args = parse_args()
    config_path = resolve_train_config_path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    set_seed(int(cfg["training"].get("seed", 42)))

    case_id = str(cfg["case_id"])
    timestamp = current_timestamp()

    config_train_dir = resolve_demo_config_path(cfg["paths"].get("config_train_dir", default_config_train_dir()))
    saved_model_root = resolve_demo_config_path(cfg["paths"].get("saved_model_dir", default_saved_model_dir()))
    ensure_dir(config_train_dir)
    backup_dir = ensure_dir(config_train_dir / "Configs_bk")
    run_dir = ensure_dir(saved_model_root / f"Case{case_id}_{timestamp}")

    backup_path = backup_dir / f"Config_Train_Case{case_id}_{timestamp}.json"
    write_json(backup_path, cfg)
    write_json(run_dir / "resolved_train_config.json", cfg)

    device = select_device(args.device)
    print(f"\n[setup] device={device}\n")
    dataset_cfg = cfg["dataset"]
    packed_h5_path = resolve_demo_config_path(dataset_cfg["packed_h5_path"])
    if not packed_h5_path.exists():
        raise FileNotFoundError(f"Packed HDF5 dataset not found: {packed_h5_path}")

    model_cfg = ModelConfig.from_dict(cfg["model"])
    max_num_cylinders = int(dataset_cfg.get("max_num_cylinders", model_cfg.max_num_cylinders))
    if model_cfg.max_num_cylinders != max_num_cylinders:
        model_cfg.max_num_cylinders = max_num_cylinders
    print(f"[setup] decoder={model_cfg.decoder_type}")
    print(
        "[setup] hierarchical_perceiver: "
        f"global_layers={model_cfg.perceiver_num_layers_global}, "
        f"local_layers={model_cfg.perceiver_num_layers_local}, "
        f"heads={model_cfg.perceiver_num_heads}, "
        f"head_dim={model_cfg.perceiver_head_dim}, "
        f"topk_env={model_cfg.perceiver_refine_topk_env}, "
        f"topk_mod={model_cfg.perceiver_refine_topk_mod}, "
        f"query_chunk_size={model_cfg.perceiver_query_chunk_size}, "
        f"local_topk_mode={model_cfg.local_topk_mode}"
    )

    train_dataset = PackedPointChunkDataset(
        packed_h5_path,
        split=dataset_cfg.get("train_split", "train"),
        is_training=True,
        points_per_item=int(dataset_cfg.get("points_per_item", 4096)),
        max_num_cylinders=max_num_cylinders,
        train_point_fraction=float(dataset_cfg.get("train_point_fraction", 1.0)),
        min_points_per_sample=int(dataset_cfg.get("min_points_per_sample", 256)),
        resample_each_epoch=bool(dataset_cfg.get("resample_each_epoch", True)),
        base_seed=int(cfg["training"].get("seed", 42)),
        randomize_cylinder_order=bool(dataset_cfg.get("randomize_cylinder_order", False)),
        point_sampling_mode=str(dataset_cfg.get("point_sampling_mode", "uniform")),

        wake_uniform_fraction=float(dataset_cfg.get("wake_uniform_fraction", 0.35)),
        wake_near_cylinder_fraction=float(dataset_cfg.get("wake_near_cylinder_fraction", 0.20)),
        wake_downstream_fraction=float(dataset_cfg.get("wake_downstream_fraction", 0.30)),
        wake_high_omega_fraction=float(dataset_cfg.get("wake_high_omega_fraction", 0.15)),
        wake_near_radius=float(dataset_cfg.get("wake_near_radius", 1.25)),
        wake_sigma_y_base=float(dataset_cfg.get("wake_sigma_y_base", 0.50)),
        wake_sigma_y_growth=float(dataset_cfg.get("wake_sigma_y_growth", 0.20)),
        wake_streamwise_decay=float(dataset_cfg.get("wake_streamwise_decay", 8.0)),
        wake_high_omega_quantile=float(dataset_cfg.get("wake_high_omega_quantile", 0.75)),
        wake_sampling_train_only=bool(dataset_cfg.get("wake_sampling_train_only", True)),

        use_phase_window_batches=bool(dataset_cfg.get("use_phase_window_batches", False)),
        phase_window_train_only=bool(dataset_cfg.get("phase_window_train_only", True)),
        phase_window_fraction=float(dataset_cfg.get("phase_window_fraction", 0.25)),
        phase_window_num_phases=int(dataset_cfg.get("phase_window_num_phases", 4)),
        phase_window_num_xy=int(dataset_cfg.get("phase_window_num_xy", 256)),
        phase_window_phase_mode=str(dataset_cfg.get("phase_window_phase_mode", "stratified_cycle")),
        phase_window_xy_sampling=str(dataset_cfg.get("phase_window_xy_sampling", "omega_variance")),
        phase_window_require_canonical_cycle=bool(dataset_cfg.get("phase_window_require_canonical_cycle", False)),
    )
    validation_cfg = cfg["validation"]
    val_mode = str(validation_cfg.get("mode", "point_chunks")).strip().lower()
    if val_mode not in {"point_chunks", "canonical_full_grid"}:
        raise ValueError("validation.mode must be either 'point_chunks' or 'canonical_full_grid'.")
    best_metric_name = str(validation_cfg.get("best_metric_name", "val_residual_focus")).strip()
    early_stopping_patience = max(0, int(validation_cfg.get("early_stopping_patience", 0)))

    val_dataset = None
    val_loader = None
    canonical_val_dataset = None
    if val_mode == "point_chunks":
        val_dataset = PackedPointChunkDataset(
            packed_h5_path,
            split=dataset_cfg.get("val_split", "test"),
            is_training=False,
            points_per_item=int(validation_cfg.get("points_per_item", dataset_cfg.get("points_per_item", 4096))),
            max_num_cylinders=max_num_cylinders,
            train_point_fraction=float(validation_cfg.get("point_fraction", 1.0)),
            min_points_per_sample=int(validation_cfg.get("min_points_per_sample", dataset_cfg.get("min_points_per_sample", 256))),
            resample_each_epoch=bool(validation_cfg.get("resample_each_eval", False)),
            base_seed=int(validation_cfg.get("seed", cfg["training"].get("seed", 42))),
            randomize_cylinder_order=bool(dataset_cfg.get("randomize_cylinder_order_val", False)),
            point_sampling_mode=str(validation_cfg.get("point_sampling_mode", "uniform")),
            
            wake_uniform_fraction=float(validation_cfg.get("wake_uniform_fraction", dataset_cfg.get("wake_uniform_fraction", 0.35))),
            wake_near_cylinder_fraction=float(validation_cfg.get("wake_near_cylinder_fraction", dataset_cfg.get("wake_near_cylinder_fraction", 0.20))),
            wake_downstream_fraction=float(validation_cfg.get("wake_downstream_fraction", dataset_cfg.get("wake_downstream_fraction", 0.30))),
            wake_high_omega_fraction=float(validation_cfg.get("wake_high_omega_fraction", dataset_cfg.get("wake_high_omega_fraction", 0.15))),
            wake_near_radius=float(validation_cfg.get("wake_near_radius", dataset_cfg.get("wake_near_radius", 1.25))),
            wake_sigma_y_base=float(validation_cfg.get("wake_sigma_y_base", dataset_cfg.get("wake_sigma_y_base", 0.50))),
            wake_sigma_y_growth=float(validation_cfg.get("wake_sigma_y_growth", dataset_cfg.get("wake_sigma_y_growth", 0.20))),
            wake_streamwise_decay=float(validation_cfg.get("wake_streamwise_decay", dataset_cfg.get("wake_streamwise_decay", 8.0))),
            wake_high_omega_quantile=float(validation_cfg.get("wake_high_omega_quantile", dataset_cfg.get("wake_high_omega_quantile", 0.75))),
            wake_sampling_train_only=bool(validation_cfg.get("wake_sampling_train_only", dataset_cfg.get("wake_sampling_train_only", True))),
            
            use_phase_window_batches=bool(validation_cfg.get("use_phase_window_batches", dataset_cfg.get("use_phase_window_batches", False))),
            phase_window_train_only=bool(validation_cfg.get("phase_window_train_only", dataset_cfg.get("phase_window_train_only", True))),
            phase_window_fraction=float(validation_cfg.get("phase_window_fraction", dataset_cfg.get("phase_window_fraction", 0.25))),
            phase_window_num_phases=int(validation_cfg.get("phase_window_num_phases", dataset_cfg.get("phase_window_num_phases", 4))),
            phase_window_num_xy=int(validation_cfg.get("phase_window_num_xy", dataset_cfg.get("phase_window_num_xy", 256))),
            phase_window_phase_mode=str(validation_cfg.get("phase_window_phase_mode", dataset_cfg.get("phase_window_phase_mode", "stratified_cycle"))),
            phase_window_xy_sampling=str(validation_cfg.get("phase_window_xy_sampling", dataset_cfg.get("phase_window_xy_sampling", "omega_variance"))),
            phase_window_require_canonical_cycle=bool(validation_cfg.get("phase_window_require_canonical_cycle", dataset_cfg.get("phase_window_require_canonical_cycle", False))),
        )
    else:
        canonical_val_dataset = CanonicalCycleValidationDataset(
            packed_h5_path,
            split=dataset_cfg.get("val_split", "test"),
        )
    if len(train_dataset) == 0:
        raise RuntimeError(
            f"No training chunks were found in {packed_h5_path} for split='{dataset_cfg.get('train_split', 'train')}'."
        )

    requested_batch_size = int(cfg["training"]["batch_size"])
    points_per_item = int(dataset_cfg.get("points_per_item", 4096))
    train_point_fraction = float(dataset_cfg.get("train_point_fraction", 1.0))
    min_points_per_sample = int(dataset_cfg.get("min_points_per_sample", 256))
    effective_points_per_item = estimate_effective_points_per_item(
        points_per_item,
        point_fraction=train_point_fraction,
        min_points_per_sample=min_points_per_sample,
    )
    # max_physical_queries = int(cfg["training"].get("max_physical_queries_per_step", 131072))
    # if effective_points_per_item > 0 and max_physical_queries > 0:
    #     max_batch_from_queries = max(1, max_physical_queries // effective_points_per_item)
    #     physical_batch_size = min(requested_batch_size, max_batch_from_queries)
    # else:
    #     physical_batch_size = requested_batch_size
    physical_batch_size = requested_batch_size
    accumulation_steps = max(1, math.ceil(requested_batch_size / max(physical_batch_size, 1)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=physical_batch_size,
        shuffle=bool(dataset_cfg.get("shuffle_train_chunks", True)),
        num_workers=int(dataset_cfg.get("train_num_workers", 0)),
        collate_fn=collate_point_chunks,
        pin_memory=torch.cuda.is_available(),
    )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(validation_cfg.get("batch_size", physical_batch_size)),
            shuffle=False,
            num_workers=int(validation_cfg.get("num_workers", dataset_cfg.get("val_num_workers", 0))),
            collate_fn=collate_point_chunks,
            pin_memory=torch.cuda.is_available(),
        )

    model = HypergraphNeuralFieldModel(model_cfg)
    model = model.to(device)

    if device.type == "cuda" and cfg["training"].get("use_data_parallel", True) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    scheduler_name = cfg["training"].get("scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["training"].get("scheduler_t_max", cfg["training"]["epochs"])),
            eta_min=float(cfg["training"].get("scheduler_min_lr", 1e-6)),
        )
    else:
        scheduler = None

    amp_enabled = bool(cfg["training"].get("mixed_precision", True))
    scaler = make_grad_scaler(device, enabled=amp_enabled)

    latest_path = run_dir / "latest_model.pt"
    best_path = run_dir / "best_model.pt"
    history_csv = run_dir / "loss_history.csv"
    loss_curve_path = run_dir / "loss_curve.png"

    log_fields = [
        "epoch",
        "loss_total",
        "loss_field",
        "loss_mean",
        "loss_residual",
        "loss_freq",
        "loss_dynamic_energy",
        "loss_org_sparsity",
        "loss_org_entropy",
        "loss_org_direct",
        "loss_org_me",
        "loss_org_mm",
        "loss_org_consistency",
        "loss_org_eh_factor",
        "loss_org_mass_align",
        "org_env_mass_max",
        "org_module_mass_max",
        "org_mass_l1",
        "org_env_effective_hyperedges",
        "org_module_effective_hyperedges",
        "org_active_hyperedges",
        "org_collapsed_hyperedges",
        "org_duplicate_hyperedges",
        "org_inactive_hyperedges",
        "org_mean_edge_score",
        # "lr",
        "val_total_loss",
        "val_field_mse",
        "val_mean_mse",
        "val_residual_mse",
        "val_freq_mse",
        "val_dynamic_energy",
        "val_residual_focus",
        "val_loss_org_eh_factor",
        "val_loss_org_mass_align",
        "val_org_env_mass_max",
        "val_org_module_mass_max",
        "val_org_mass_l1",
        "val_org_env_effective_hyperedges",
        "val_org_module_effective_hyperedges",
        "val_org_active_hyperedges",
        "val_org_collapsed_hyperedges",
        "val_org_duplicate_hyperedges",
        "val_org_inactive_hyperedges",
        "val_org_mean_edge_score",
        "val_selection_metric",
    ]
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    max_grad_norm = float(cfg["training"].get("gradient_clip_norm", 1.0))
    total_epochs = int(cfg["training"]["epochs"])
    queries_per_step = requested_batch_size * max(effective_points_per_item, 1)
    physical_queries_per_step = physical_batch_size * max(effective_points_per_item, 1)
    env_tokens = int(model_cfg.num_env_tokens_x) * int(model_cfg.num_env_tokens_y)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)

    print(
        f"[setup] train chunks={len(train_dataset)} | physical train batches/epoch={len(train_loader)} "
        f"| optimizer steps/epoch={optimizer_steps_per_epoch} | val mode={val_mode} "
        f"| val units={len(val_dataset) if val_dataset is not None else len(canonical_val_dataset) if canonical_val_dataset is not None else 0}"
    )
    print(f"[setup] run_dir={run_dir}")
    print(f"[setup] saved resolved config to {run_dir / 'resolved_train_config.json'}")
    print(f"[setup] backed up training config to {backup_path}")
    print(
        f"[setup] requested_batch_size={requested_batch_size} | physical_batch_size={physical_batch_size} "
        f"| accumulation_steps={accumulation_steps} | points_per_item={points_per_item}"
    )
    print(
        f"[setup] train_point_fraction={train_point_fraction:.3f} | "
        f"min_points_per_sample={min_points_per_sample} | "
        f"effective_points_per_item~{effective_points_per_item}"
    )
    print(
        f"[setup] phase_window_batches={bool(dataset_cfg.get('use_phase_window_batches', False))} | "
        f"fraction={float(dataset_cfg.get('phase_window_fraction', 0.0)):.3f} | "
        f"num_phases={int(dataset_cfg.get('phase_window_num_phases', 1))} | "
        f"xy_sampling={dataset_cfg.get('phase_window_xy_sampling', 'uniform')}"
    )
    print(
        f"[setup] effective queries/optimizer-step~{format_large_int(queries_per_step)} | "
        f"physical queries/forward~{format_large_int(physical_queries_per_step)} | env_tokens={env_tokens}"
    )
    print(f"[setup] best_metric_name={best_metric_name}")
    risky_dropouts = []
    if float(model_cfg.dropout) <= 0.0:
        risky_dropouts.append("model.dropout")
    if float(model_cfg.perceiver_dropout) <= 0.0:
        risky_dropouts.append("model.perceiver_dropout")
    if float(model_cfg.phase_conditioning_dropout) <= 0.0:
        risky_dropouts.append("model.phase_conditioning_dropout")
    if risky_dropouts:
        print(
            "[warn] Zero dropout detected in "
            + ", ".join(risky_dropouts)
            + ". The current recommended settings keep these at 0.05 for better stability."
        )
    if float(cfg["loss"].get("organizer_sparsity_weight", 0.0)) > 0.0 or float(cfg["loss"].get("organizer_entropy_weight", 0.0)) > 0.0:
        print(
            "[warn] organizer_sparsity_weight / organizer_entropy_weight are nonzero. "
            "The current template disables both because they were not helping stability."
        )
    if requested_batch_size >= 512:
        print(
            "[warn] training.batch_size is large. The updated template defaults to 256 so accumulation can keep "
            "forwards smaller and more stable."
        )
    if accumulation_steps > 1:
        print(
            "[setup] Large requested batch detected, so gradient accumulation is enabled automatically to keep "
            "individual forwards manageable."
        )
    if physical_queries_per_step >= 32768:
        print(
            "[warn] Large physical queries-per-forward can still be slow. If the first batch remains sluggish, "
            "reduce training.batch_size, training.max_physical_queries_per_step, or dataset.points_per_item."
        )
    if physical_queries_per_step >= 32768:
        print(
            "[warn] The hierarchical decoder uses global attention plus local top-k refinement. "
            "If GPU memory is tight, lower dataset.points_per_item or training.batch_size before increasing model width."
        )

    history_rows: List[Dict[str, float]] = []
    best_val_metric = float("inf")
    epochs_without_improvement = 0
    requested_disable_edge = bool(model_cfg.DISABLE_EDGE)
    disable_edge_start_epoch = max(0, int(model_cfg.disable_edge_start_epoch))
    if requested_disable_edge:
        print(
            "[warn] DISABLE_EDGE is enabled. Hard active-edge masking is usually best after organizer losses "
            "have stabilized, or with conservative thresholds."
        )

    for epoch in range(1, total_epochs + 1):
        runtime_disable_edge = requested_disable_edge and epoch >= disable_edge_start_epoch
        runtime_model = model.module if isinstance(model, nn.DataParallel) else model
        runtime_model.set_edge_disable_runtime(runtime_disable_edge)
        model.train()
        train_dataset.set_epoch(epoch)
        if requested_disable_edge:
            print(
                f"[edge-disable] epoch={epoch:03d} active_masking={'on' if runtime_disable_edge else 'off'} "
                f"(start_epoch={disable_edge_start_epoch})"
            )
        print(f"[epoch] {epoch:03d}/{total_epochs:03d} training started")
        train_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"epoch {epoch:03d}/{total_epochs:03d} train",
            leave=False,
            dynamic_ncols=True,
        )
        accumulation_buffer: List[Dict[str, float]] = []
        epoch_metric_buffer: List[Dict[str, float]] = []
        active_accum_steps = accumulation_steps
        epoch_train_metrics = None
        for step_in_epoch, batch in enumerate(train_bar, start=1):
            micro_index = (step_in_epoch - 1) % accumulation_steps
            if micro_index == 0:
                optimizer.zero_grad(set_to_none=True)
                active_accum_steps = min(accumulation_steps, len(train_loader) - step_in_epoch + 1)
                accumulation_buffer = []

            structure = {
                "re_values": batch["re_values"].to(device=device),
                "num_cylinders": batch["num_cylinders"].to(device=device),
                "centers": batch["centers"].to(device=device),
                "cyl_mask": batch["cyl_mask"].to(device=device),
            }
            query_xy = batch["query_xy"].to(device=device)
            query_tau = batch["query_tau"].to(device=device)

            for key in [
                "re_values",
                "field_targets",
                "mean_targets",
                "residual_targets",
                "point_mask",
                "freq_target",
                "cyl_mask",
            ]:
                batch[key] = batch[key].to(device=device)

            org_scale = organizer_ramp_scale(cfg["loss"], epoch)
            with autocast_context(device, enabled=scaler.is_enabled()):
                outputs = model(structure=structure, query_xy=query_xy, query_tau=query_tau, return_aux=True)
                bad_outputs = find_nonfinite_tensors(
                    {
                        "pred_field": outputs["pred_field"],
                        "pred_mean": outputs["pred_mean"],
                        "pred_residual": outputs["pred_residual"],
                        "freq_pred": outputs["freq_pred"],
                        "A_me": outputs["A_me"],
                        "A_mh": outputs["A_mh"],
                        "A_eh": outputs["A_eh"],
                    }
                )
                if bad_outputs:
                    raise RuntimeError(
                        f"Non-finite model outputs detected at epoch={epoch}, step={step_in_epoch}: {bad_outputs}"
                    )
                losses = compute_losses(
                    batch,
                    outputs,
                    cfg["loss"],
                    organizer_scale=org_scale,
                )
                bad_losses = find_nonfinite_tensors(losses)
                if bad_losses:
                    raise RuntimeError(
                        f"Non-finite losses detected at epoch={epoch}, step={step_in_epoch}: {bad_losses}"
                    )
                total_loss = losses["loss_total"] / float(active_accum_steps)

            scaler.scale(total_loss).backward()
            bad_grads = {
                name: int((~torch.isfinite(param.grad)).sum().item())
                for name, param in model.named_parameters()
                if param.grad is not None and not torch.isfinite(param.grad).all()
            }
            if bad_grads:
                raise RuntimeError(
                    f"Non-finite gradients detected at epoch={epoch}, step={step_in_epoch}: {bad_grads}"
                )
            accumulation_buffer.append(
                {
                    "loss_total": normalize_loss_scalar(losses["loss_total"]),
                    "loss_field": normalize_loss_scalar(losses["loss_field"]),
                    "loss_mean": normalize_loss_scalar(losses["loss_mean"]),
                    "loss_residual": normalize_loss_scalar(losses["loss_residual"]),
                    "loss_freq": normalize_loss_scalar(losses["loss_freq"]),
                    "loss_dynamic_energy": normalize_loss_scalar(losses["loss_dynamic_energy"]),
                    "loss_org_sparsity": normalize_loss_scalar(losses["loss_org_sparsity"]),
                    "loss_org_entropy": normalize_loss_scalar(losses["loss_org_entropy"]),
                    "loss_org_direct": normalize_loss_scalar(losses["loss_org_direct"]),
                    "loss_org_me": normalize_loss_scalar(losses["loss_org_me"]),
                    "loss_org_mm": normalize_loss_scalar(losses["loss_org_mm"]),
                    "loss_org_consistency": normalize_loss_scalar(losses["loss_org_consistency"]),
                    "loss_org_eh_factor": normalize_loss_scalar(losses["loss_org_eh_factor"]),
                    "loss_org_mass_align": normalize_loss_scalar(losses["loss_org_mass_align"]),
                    "org_env_mass_max": normalize_loss_scalar(losses["org_env_mass_max"]),
                    "org_module_mass_max": normalize_loss_scalar(losses["org_module_mass_max"]),
                    "org_mass_l1": normalize_loss_scalar(losses["org_mass_l1"]),
                    "org_env_effective_hyperedges": normalize_loss_scalar(losses["org_env_effective_hyperedges"]),
                    "org_module_effective_hyperedges": normalize_loss_scalar(losses["org_module_effective_hyperedges"]),
                    "org_active_hyperedges": normalize_loss_scalar(losses["org_active_hyperedges"]),
                    "org_collapsed_hyperedges": normalize_loss_scalar(losses["org_collapsed_hyperedges"]),
                    "org_duplicate_hyperedges": normalize_loss_scalar(losses["org_duplicate_hyperedges"]),
                    "org_inactive_hyperedges": normalize_loss_scalar(losses["org_inactive_hyperedges"]),
                    "org_mean_edge_score": normalize_loss_scalar(losses["org_mean_edge_score"]),
                }
            )
            train_bar.set_postfix(
                loss=f"{accumulation_buffer[-1]['loss_total']:.3e}",
                residual=f"{accumulation_buffer[-1]['loss_residual']:.3e}",
                dyn_energy=f"{accumulation_buffer[-1]['loss_dynamic_energy']:.3e}",
                org_l1=f"{accumulation_buffer[-1]['org_mass_l1']:.2e}",
                env_effH=f"{accumulation_buffer[-1]['org_env_effective_hyperedges']:.2f}",
                activeH=f"{accumulation_buffer[-1]['org_active_hyperedges']:.1f}",
                freq=f"{accumulation_buffer[-1]['loss_freq']:.3e}",
                accum=f"{micro_index + 1}/{active_accum_steps}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            should_step = ((micro_index + 1) == active_accum_steps) or (step_in_epoch == len(train_loader))
            if not should_step:
                continue

            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            avg_losses = average_metric_dicts(accumulation_buffer)
            epoch_metric_buffer.append(avg_losses)
            epoch_train_metrics = avg_losses

        train_bar.close()
        if epoch_train_metrics is None or not epoch_metric_buffer:
            raise RuntimeError("No optimizer step was completed during the epoch. Check batch sizing and accumulation settings.")
        epoch_train_metrics = average_metric_dicts(epoch_metric_buffer)
        print(
            f"[epoch] {epoch:03d}/{total_epochs:03d} training finished | "
            f"{format_metrics(epoch_train_metrics, keys=['loss_total', 'loss_field', 'loss_mean', 'loss_residual', 'loss_dynamic_energy', 'loss_freq', 'loss_org_direct'])} | "
            f"{format_metrics(epoch_train_metrics, keys=['org_mass_l1', 'org_env_effective_hyperedges', 'org_module_effective_hyperedges', 'org_env_mass_max', 'org_module_mass_max'])} | "
            f"activeH={epoch_train_metrics.get('org_active_hyperedges', float('nan')):.2f} "
            f"collapsedH={epoch_train_metrics.get('org_collapsed_hyperedges', float('nan')):.2f} "
            f"duplicateH={epoch_train_metrics.get('org_duplicate_hyperedges', float('nan')):.2f}"
        )

        if scheduler is not None:
            scheduler.step()
            print(f"[sched] epoch={epoch:03d} lr={optimizer.param_groups[0]['lr']:.6e}")

        val_metrics = {
            "val_total_loss": float("nan"),
            "val_field_mse": float("nan"),
            "val_mean_mse": float("nan"),
            "val_residual_mse": float("nan"),
            "val_freq_mse": float("nan"),
            "val_dynamic_energy": float("nan"),
            "val_residual_focus": float("nan"),
            "val_loss_org_eh_factor": float("nan"),
            "val_loss_org_mass_align": float("nan"),
            "val_org_env_mass_max": float("nan"),
            "val_org_module_mass_max": float("nan"),
            "val_org_mass_l1": float("nan"),
            "val_org_env_effective_hyperedges": float("nan"),
            "val_org_module_effective_hyperedges": float("nan"),
            "val_org_active_hyperedges": float("nan"),
            "val_org_collapsed_hyperedges": float("nan"),
            "val_org_duplicate_hyperedges": float("nan"),
            "val_org_inactive_hyperedges": float("nan"),
            "val_org_mean_edge_score": float("nan"),
        }
        eval_every_epochs = max(1, int(validation_cfg.get("eval_every_epochs", 1)))
        should_run_validation = (epoch == 1) or (epoch % eval_every_epochs == 0)
        if should_run_validation:
            print(f"[epoch] {epoch:03d}/{total_epochs:03d} validation started")
            eval_model = model.module if isinstance(model, nn.DataParallel) else model
            if val_mode == "point_chunks":
                if val_dataset is None or val_loader is None:
                    raise RuntimeError("Point-chunk validation requested but validation dataloader was not initialized.")
                val_dataset.set_epoch(epoch)
                val_metrics = evaluate_point_chunks(
                    model=eval_model,
                    val_loader=val_loader,
                    device=device,
                    loss_cfg=cfg["loss"],
                    validation_cfg=validation_cfg,
                    show_progress=True,
                )
            else:
                if canonical_val_dataset is None:
                    raise RuntimeError("Canonical validation requested but validation dataset was not initialized.")
                val_metrics = evaluate_canonical_cases(
                    model=eval_model,
                    dataset=canonical_val_dataset,
                    device=device,
                    loss_cfg=cfg["loss"],
                    validation_cfg=validation_cfg,
                    max_num_cylinders=max_num_cylinders,
                    max_cases=int(validation_cfg.get("max_cases", len(canonical_val_dataset))),
                    query_batch_size=int(validation_cfg.get("query_batch_size", 32768)),
                    phase_bins_to_eval=int(validation_cfg.get("phase_bins_to_eval", 6)),
                    show_progress=True,
                )
            val_selection_metric = compute_validation_selection_metric(val_metrics, validation_cfg)
            print(
                f"[val] epoch={epoch:03d} total={val_metrics['val_total_loss']:.6e} "
                f"field_mse={val_metrics['val_field_mse']:.6e} "
                f"residual_mse={val_metrics['val_residual_mse']:.6e} "
                f"dynamic_energy={val_metrics['val_dynamic_energy']:.6e} "
                f"residual_focus={val_metrics['val_residual_focus']:.6e} "
                f"mean_mse={val_metrics['val_mean_mse']:.6e} "
                f"freq_mse={val_metrics['val_freq_mse']:.6e} "
                f"org_mass_l1={val_metrics.get('val_org_mass_l1', float('nan')):.6e} "
                f"env_effH={val_metrics.get('val_org_env_effective_hyperedges', float('nan')):.3f} "
                f"mod_effH={val_metrics.get('val_org_module_effective_hyperedges', float('nan')):.3f} "
                f"env_max={val_metrics.get('val_org_env_mass_max', float('nan')):.3f} "
                f"mod_max={val_metrics.get('val_org_module_mass_max', float('nan')):.3f} "
                f"activeH={val_metrics.get('val_org_active_hyperedges', float('nan')):.2f} "
                f"collapsedH={val_metrics.get('val_org_collapsed_hyperedges', float('nan')):.2f} "
                f"duplicateH={val_metrics.get('val_org_duplicate_hyperedges', float('nan')):.2f} "
                f"selection={val_selection_metric:.6e}"
            )
        else:
            val_selection_metric = float("nan")
            print(f"[epoch] {epoch:03d}/{total_epochs:03d} validation skipped")

        epoch_row = {
            "epoch": epoch,
            **round_loss_metrics(
                {
                    "loss_total": epoch_train_metrics["loss_total"],
                    "loss_field": epoch_train_metrics["loss_field"],
                    "loss_mean": epoch_train_metrics["loss_mean"],
                    "loss_residual": epoch_train_metrics["loss_residual"],
                    "loss_freq": epoch_train_metrics["loss_freq"],
                    "loss_dynamic_energy": epoch_train_metrics["loss_dynamic_energy"],
                    "loss_org_sparsity": epoch_train_metrics["loss_org_sparsity"],
                    "loss_org_entropy": epoch_train_metrics["loss_org_entropy"],
                    "loss_org_direct": epoch_train_metrics["loss_org_direct"],
                    "loss_org_me": epoch_train_metrics["loss_org_me"],
                    "loss_org_mm": epoch_train_metrics["loss_org_mm"],
                    "loss_org_consistency": epoch_train_metrics["loss_org_consistency"],
                    "loss_org_eh_factor": epoch_train_metrics["loss_org_eh_factor"],
                    "loss_org_mass_align": epoch_train_metrics["loss_org_mass_align"],
                    "org_env_mass_max": epoch_train_metrics["org_env_mass_max"],
                    "org_module_mass_max": epoch_train_metrics["org_module_mass_max"],
                    "org_mass_l1": epoch_train_metrics["org_mass_l1"],
                    "org_env_effective_hyperedges": epoch_train_metrics["org_env_effective_hyperedges"],
                    "org_module_effective_hyperedges": epoch_train_metrics["org_module_effective_hyperedges"],
                    "org_active_hyperedges": epoch_train_metrics["org_active_hyperedges"],
                    "org_collapsed_hyperedges": epoch_train_metrics["org_collapsed_hyperedges"],
                    "org_duplicate_hyperedges": epoch_train_metrics["org_duplicate_hyperedges"],
                    "org_inactive_hyperedges": epoch_train_metrics["org_inactive_hyperedges"],
                    "org_mean_edge_score": epoch_train_metrics["org_mean_edge_score"],
                }
            ),
            # "lr": float(optimizer.param_groups[0]["lr"]),
            **round_loss_metrics(val_metrics),
            "val_selection_metric": round_to_significant_figures(val_selection_metric),
        }
        history_rows.append(epoch_row)
        with history_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(epoch_row)
        save_loss_curve(history_rows, loss_curve_path)
        print(f"[save] wrote epoch summary to {history_csv}")
        print(f"[save] refreshed loss curve at {loss_curve_path}")

        # Checkpoint selection is configurable so we can prioritize residual fidelity.
        current_val = val_selection_metric
        is_new_best = bool(math.isfinite(current_val) and current_val < best_val_metric)
        if math.isfinite(current_val) and current_val < best_val_metric:
            best_val_metric = current_val
            epochs_without_improvement = 0
        elif should_run_validation and math.isfinite(current_val):
            epochs_without_improvement += 1
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "model_config": {**model_cfg.__dict__, "DISABLE_EDGE": requested_disable_edge},
            "best_metric_name": best_metric_name,
            "val_metrics": val_metrics,
            "val_selection_metric": val_selection_metric,
            "best_val_metric": best_val_metric,
        }
        torch.save(checkpoint, latest_path)
        print(f"[save] wrote epoch checkpoint: {latest_path}")

        if is_new_best:
            torch.save(checkpoint, best_path)
            print(f"[save] new best checkpoint: {best_path} | {best_metric_name}={best_val_metric:.6e}")
        else:
            print(
                f"[save] best checkpoint unchanged | best_val_metric="
                f"{best_val_metric:.6e}" if math.isfinite(best_val_metric) else "[save] best checkpoint unchanged | best_val_metric=nan"
            )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"[stop] early stopping triggered after {epochs_without_improvement} validation epochs "
                f"without improvement in {best_metric_name}"
            )
            break

    train_dataset.close()
    if val_dataset is not None:
        val_dataset.close()
    print(f"Training finished. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
