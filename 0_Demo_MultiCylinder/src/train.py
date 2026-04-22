from __future__ import annotations

"""Training script for the hypergraph-organized neural field model.

This script reads a packed HDF5 dataset, trains the organizer + behavior head +
phase-conditioned neural field decoder end-to-end, and writes only the latest
and best checkpoints to a run-specific directory.

The decoder architecture is selected from the JSON model config through
`model.decoder_type`, for example `mlp_fourier`, `siren`, `deeponet`, or
`structured_perceiver`.
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
        points_per_item: int,
        max_num_cylinders: int,
        train_point_fraction: float = 1.0,
        min_points_per_sample: int = 1,
        resample_each_epoch: bool = False,
        base_seed: int = 42,
    ):
        super().__init__()
        self.h5_path = Path(h5_path).expanduser().resolve()
        self.split = split
        self.points_per_item = int(points_per_item)
        self.max_num_cylinders = int(max_num_cylinders)
        self.train_point_fraction = float(train_point_fraction)
        self.min_points_per_sample = int(min_points_per_sample)
        self.resample_each_epoch = bool(resample_each_epoch)
        self.base_seed = int(base_seed)
        self.current_epoch = 0
        self._h5: Optional[h5py.File] = None

        if not (0.0 < self.train_point_fraction <= 1.0):
            raise ValueError("train_point_fraction must be in (0, 1].")
        if self.min_points_per_sample < 1:
            raise ValueError("min_points_per_sample must be >= 1.")

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

    def _maybe_subsample_indices(self, idx: int, num_points: int) -> np.ndarray:
        if self.train_point_fraction >= 1.0:
            return np.arange(num_points, dtype=np.int64)

        keep_count = max(self.min_points_per_sample, int(math.ceil(num_points * self.train_point_fraction)))
        keep_count = min(keep_count, num_points)
        if keep_count >= num_points:
            return np.arange(num_points, dtype=np.int64)

        epoch_offset = self.current_epoch if self.resample_each_epoch else 0
        rng_seed = self.base_seed + (1000003 * idx) + (9176 * epoch_offset)
        rng = np.random.default_rng(rng_seed)
        chosen = np.sort(rng.choice(num_points, size=keep_count, replace=False))
        return chosen.astype(np.int64, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.case_meta[idx]
        h5_file = self._get_h5()
        grp = h5_file["cases"][meta.case_id]
        sampled = grp["sampled_points"]
        sl = slice(meta.start_idx, meta.end_idx)
        local_size = meta.end_idx - meta.start_idx
        local_indices = self._maybe_subsample_indices(idx, local_size)

        # 1. Fetch the contiguous block from disk into RAM first (Lightning Fast)
        chunk_x = sampled["x"][sl]
        chunk_y = sampled["y"][sl]
        chunk_tau = sampled["tau"][sl]
        chunk_u = sampled["u"][sl]
        chunk_v = sampled["v"][sl]
        chunk_p = sampled["p"][sl]
        chunk_omega = sampled["omega"][sl]

        # 2. Subsample the arrays in memory (Instantaneous)
        x = np.asarray(chunk_x[local_indices], dtype=np.float32)
        y = np.asarray(chunk_y[local_indices], dtype=np.float32)
        tau = np.asarray(chunk_tau[local_indices], dtype=np.float32)
        
        targets = np.stack(
            [
                np.asarray(chunk_u[local_indices], dtype=np.float32),
                np.asarray(chunk_v[local_indices], dtype=np.float32),
                np.asarray(chunk_p[local_indices], dtype=np.float32),
                np.asarray(chunk_omega[local_indices], dtype=np.float32),
            ],
            axis=-1,
        )

        case_static = self.case_lookup[meta.case_id]
        centers = case_static["centers"]
        padded_centers = np.zeros((self.max_num_cylinders, 2), dtype=np.float32)
        cyl_mask = np.zeros((self.max_num_cylinders,), dtype=np.float32)
        padded_centers[: centers.shape[0]] = centers
        cyl_mask[: centers.shape[0]] = 1.0

        # Sample mean-field targets at the same spatial points using nearest grid lookup.
        mean_field = case_static["mean_field"]
        x0, y0 = case_static["grid_origin"]
        dx, dy = case_static["grid_spacing"]
        ix = np.clip(np.rint((x - x0) / max(dx, 1e-6)).astype(np.int64), 0, mean_field.shape[1] - 1)
        iy = np.clip(np.rint((y - y0) / max(dy, 1e-6)).astype(np.int64), 0, mean_field.shape[0] - 1)
        mean_targets = mean_field[iy, ix]
        residual_targets = targets - mean_targets

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
        "freq_target": torch.stack([item["freq_target"] for item in batch], dim=0),
    }

    point_mask = torch.zeros((batch_size, max_points), dtype=torch.float32)
    for i, item in enumerate(batch):
        point_mask[i, : item["query_xy"].shape[0]] = 1.0
    out["point_mask"] = point_mask
    return out


# ------------------------------- Loss functions --------------------------------


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

    point_mask = batch["point_mask"].to(device=pred_field.device, dtype=pred_field.dtype).unsqueeze(-1)  # [B, Q, 1]

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
    )

    total = base_total + float(organizer_scale) * org_direct["organizer_total"]

    return {
        "loss_total": total,
        "loss_field": loss_field,
        "loss_mean": loss_mean,
        "loss_residual": loss_residual,
        "loss_freq": loss_freq,
        "loss_org_sparsity": loss_org_concentration,
        "loss_org_entropy": loss_org_entropy,
        "loss_org_direct": org_direct["organizer_total"],
        "loss_org_me": org_direct["organizer_me"],
        "loss_org_mm": org_direct["organizer_mm"],
        "loss_org_consistency": org_direct["organizer_consistency"],
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
    loss_org_sparsity: torch.Tensor | float,
    loss_org_entropy: torch.Tensor | float,
) -> torch.Tensor | float:
    return (
        loss_cfg["field_mse_weight"] * loss_field
        + loss_cfg["mean_mse_weight"] * loss_mean
        + loss_cfg["residual_mse_weight"] * loss_residual
        + loss_cfg["freq_mse_weight"] * loss_freq
        + loss_cfg["organizer_sparsity_weight"] * loss_org_sparsity
        + loss_cfg["organizer_entropy_weight"] * loss_org_entropy
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


def pairwise_periodic_relative_features(src_xy: torch.Tensor, dst_xy: torch.Tensor) -> torch.Tensor:
    """
    src_xy: [B, N_src, 2] normalized to [0,1]
    dst_xy: [B, N_dst, 2] normalized to [0,1]
    returns: [B, N_src, N_dst, 5] = dx, dy, dist, downstream, upstream
    """
    dx = src_xy[:, :, None, 0] - dst_xy[:, None, :, 0]
    dy = src_xy[:, :, None, 1] - dst_xy[:, None, :, 1]
    dx = (dx + 0.5) % 1.0 - 0.5
    dy = (dy + 0.5) % 1.0 - 0.5
    dist = torch.sqrt(dx.square() + dy.square() + 1e-8)
    downstream = torch.clamp(-dx, min=0.0)
    upstream = torch.clamp(dx, min=0.0)
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

    # Optional mild Re effect on wake width.
    re_scale = (re_values / 200.0).clamp(min=0.1, max=1.5)[:, None, None]
    sigma_y = sigma_y_base + sigma_y_growth * downstream * re_scale

    wake = torch.exp(-0.5 * (dy / sigma_y.clamp_min(1e-4)).square()) * torch.exp(-downstream / decay_x)
    wake = wake * (downstream > 0).to(wake.dtype)

    near = torch.exp(-0.5 * (dist / near_radius).square())
    prior = wake + 0.25 * near

    prior = prior * cyl_mask[:, :, None]
    return normalize_rows(prior)


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


def organizer_direct_losses(
    outputs: Dict[str, torch.Tensor],
    structure: Dict[str, torch.Tensor],
    me_weight: float = 0.05,
    mm_weight: float = 0.03,
    consistency_weight: float = 0.05,) -> Dict[str, torch.Tensor]:
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
    kl_me = F.kl_div((A_me.clamp_min(1e-6)).log(), prior_me, reduction="none").sum(dim=-1)
    loss_me = (kl_me * cyl_mask).sum() / cyl_mask.sum().clamp_min(1.0)

    # B) permutation-safe hyperedge supervision via module-module affinity
    pred_mm = torch.matmul(A_mh, A_mh.transpose(1, 2))  # [B, N, N]
    prior_mm = build_module_module_affinity_prior(module_coords_norm, cyl_mask)

    n = pred_mm.shape[1]
    eye = torch.eye(n, device=pred_mm.device, dtype=pred_mm.dtype)[None, :, :]
    valid_mm = cyl_mask[:, :, None] * cyl_mask[:, None, :] * (1.0 - eye)
    loss_mm = (((pred_mm - prior_mm) ** 2) * valid_mm).sum() / valid_mm.sum().clamp_min(1.0)

    # C) hypergraph should factorize module-env organization
    pred_me_from_h = torch.matmul(A_mh, A_eh.transpose(1, 2))  # [B, N, M]
    pred_me_from_h = normalize_rows(pred_me_from_h)
    loss_cons = (((pred_me_from_h - A_me) ** 2) * cyl_mask[:, :, None]).sum() / (
        cyl_mask[:, :, None].sum().clamp_min(1.0)
    )

    total = me_weight * loss_me + mm_weight * loss_mm + consistency_weight * loss_cons
    return {
        "organizer_total": total,
        "organizer_me": loss_me,
        "organizer_mm": loss_mm,
        "organizer_consistency": loss_cons,
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
            "val_freq_mse": float("nan"),
        }

    case_ids = dataset.case_ids[:max_cases] if max_cases > 0 else dataset.case_ids
    total_losses, field_losses, mean_losses, residual_losses, freq_losses = [], [], [], [], []

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
            loss_org_sparsity=loss_org_sparsity,
            loss_org_entropy=loss_org_entropy,
        )
        org_direct = organizer_direct_losses(
            out_once,
            structure,
            me_weight=float(loss_cfg.get("organizer_me_weight", 0.0)),
            mm_weight=float(loss_cfg.get("organizer_mm_weight", 0.0)),
            consistency_weight=float(loss_cfg.get("organizer_consistency_weight", 0.0)),
        )
        loss_total = base_total + org_direct["organizer_total"]
        total_losses.append(float(loss_total.item() if isinstance(loss_total, torch.Tensor) else loss_total))
        
        if val_bar is not None:
            val_bar.set_postfix(
                total=f"{total_losses[-1]:.3e}",
                field_mse=f"{field_losses[-1]:.3e}",
                mean_mse=f"{mean_losses[-1]:.3e}",
                freq_mse=f"{freq_losses[-1]:.3e}",
            )

    if val_bar is not None:
        val_bar.close()

    return {
        "val_total_loss": float(np.mean(total_losses)),
        "val_field_mse": float(np.mean(field_losses)),
        "val_mean_mse": float(np.mean(mean_losses)),
        "val_freq_mse": float(np.mean(freq_losses)),
    }


@torch.no_grad()
def evaluate_point_chunks(
    model: nn.Module,
    val_loader: DataLoader,
    *,
    device: torch.device,
    loss_cfg: Dict,
    show_progress: bool = False,
) -> Dict[str, float]:
    model.eval()
    if len(val_loader) == 0:
        return {
            "val_total_loss": float("nan"),
            "val_field_mse": float("nan"),
            "val_mean_mse": float("nan"),
            "val_freq_mse": float("nan"),
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
        losses = compute_losses(
            batch,
            outputs,
            loss_cfg,
            organizer_scale=1.0,
        )
        row = {
            "val_total_loss": normalize_loss_scalar(losses["loss_total"]),
            "val_field_mse": normalize_loss_scalar(losses["loss_field"]),
            "val_mean_mse": normalize_loss_scalar(losses["loss_mean"]),
            "val_freq_mse": normalize_loss_scalar(losses["loss_freq"]),
        }
        metric_buffer.append(row)
        if val_bar is not None:
            val_bar.set_postfix(
                total=f"{row['val_total_loss']:.3e}",
                field_mse=f"{row['val_field_mse']:.3e}",
                mean_mse=f"{row['val_mean_mse']:.3e}",
                freq_mse=f"{row['val_freq_mse']:.3e}",
            )

    if val_bar is not None:
        val_bar.close()

    return average_metric_dicts(metric_buffer)

def organizer_ramp_scale(loss_cfg: Dict, epoch: int) -> float:
    ramp_epochs = max(1, int(loss_cfg.get("organizer_ramp_epochs", 1)))
    return min(1.0, float(epoch) / float(ramp_epochs))

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
            if math.isfinite(y_float):
                xs.append(float(x))
                ys.append(y_float)
        return xs, ys

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=140, sharex=True)

    total_ax, field_ax = axes

    train_total = [row.get("loss_total", float("nan")) for row in history]
    test_total = [row.get("val_total_loss", float("nan")) for row in history]
    if any(math.isfinite(float(v)) for v in train_total):
        total_ax.plot(epochs, train_total, linestyle="-", linewidth=1.5, label="Train total")
    test_total_epochs, test_total_vals = finite_xy(epochs, test_total)
    if test_total_epochs:
        total_ax.plot(
            test_total_epochs,
            test_total_vals,
            linestyle=":",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label="Test total",
        )
    total_ax.set_title("Total Loss")
    total_ax.set_xlabel("Epoch")
    total_ax.set_ylabel("Loss")
    total_ax.set_yscale("log")
    total_ax.grid(True, alpha=0.3)
    if total_ax.lines:
        total_ax.legend()

    train_field = [row.get("loss_field", float("nan")) for row in history]
    test_field = [row.get("val_field_mse", float("nan")) for row in history]
    if any(math.isfinite(float(v)) for v in train_field):
        field_ax.plot(epochs, train_field, linestyle="-", linewidth=1.5, label="Train field")
    test_field_epochs, test_field_vals = finite_xy(epochs, test_field)
    if test_field_epochs:
        field_ax.plot(
            test_field_epochs,
            test_field_vals,
            linestyle=":",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label="Test field",
        )
    field_ax.set_title("Field Loss")
    field_ax.set_xlabel("Epoch")
    field_ax.set_ylabel("Loss")
    field_ax.set_yscale("log")
    field_ax.grid(True, alpha=0.3)
    if field_ax.lines:
        field_ax.legend()

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
    print(f"[setup] decoder_type={model_cfg.decoder_type}")
    if model_cfg.decoder_type == "structured_perceiver":
        print(
            "[setup] structured_perceiver: "
            f"layers={model_cfg.perceiver_num_layers}, "
            f"heads={model_cfg.perceiver_num_heads}, "
            f"head_dim={model_cfg.perceiver_head_dim}, "
            f"global_tokens={model_cfg.perceiver_num_global_tokens}, "
            f"relative_bias={model_cfg.perceiver_use_relative_bias}, "
            f"chunk_query_attention={model_cfg.perceiver_chunk_query_attention}"
        )

    train_dataset = PackedPointChunkDataset(
        packed_h5_path,
        split=dataset_cfg.get("train_split", "train"),
        points_per_item=int(dataset_cfg.get("points_per_item", 4096)),
        max_num_cylinders=max_num_cylinders,
        train_point_fraction=float(dataset_cfg.get("train_point_fraction", 1.0)),
        min_points_per_sample=int(dataset_cfg.get("min_points_per_sample", 256)),
        resample_each_epoch=bool(dataset_cfg.get("resample_each_epoch", True)),
        base_seed=int(cfg["training"].get("seed", 42)),
    )
    validation_cfg = cfg["validation"]
    val_mode = str(validation_cfg.get("mode", "point_chunks")).strip().lower()
    if val_mode not in {"point_chunks", "canonical_full_grid"}:
        raise ValueError("validation.mode must be either 'point_chunks' or 'canonical_full_grid'.")

    val_dataset = None
    val_loader = None
    canonical_val_dataset = None
    if val_mode == "point_chunks":
        val_dataset = PackedPointChunkDataset(
            packed_h5_path,
            split=dataset_cfg.get("val_split", "test"),
            points_per_item=int(validation_cfg.get("points_per_item", dataset_cfg.get("points_per_item", 4096))),
            max_num_cylinders=max_num_cylinders,
            train_point_fraction=float(validation_cfg.get("point_fraction", 1.0)),
            min_points_per_sample=int(validation_cfg.get("min_points_per_sample", dataset_cfg.get("min_points_per_sample", 256))),
            resample_each_epoch=bool(validation_cfg.get("resample_each_eval", False)),
            base_seed=int(validation_cfg.get("seed", cfg["training"].get("seed", 42))),
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
    max_physical_queries = int(cfg["training"].get("max_physical_queries_per_step", 131072))
    # if effective_points_per_item > 0:
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
        "loss_org_sparsity",
        "loss_org_entropy",
        "loss_org_direct",
        "loss_org_me",
        "loss_org_mm",
        "loss_org_consistency",
        "lr",
        "val_total_loss",
        "val_field_mse",
        "val_mean_mse",
        "val_freq_mse",
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
        f"[setup] effective queries/optimizer-step~{format_large_int(queries_per_step)} | "
        f"physical queries/forward~{format_large_int(physical_queries_per_step)} | env_tokens={env_tokens}"
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
    if model_cfg.decoder_type == "structured_perceiver" and physical_queries_per_step >= 32768:
        print(
            "[warn] structured_perceiver adds cross-attention over organizer memory. If GPU memory is tight, "
            "lower dataset.points_per_item or training.batch_size before increasing model width."
        )

    history_rows: List[Dict[str, float]] = []
    best_val_metric = float("inf")

    for epoch in range(1, total_epochs + 1):
        model.train()
        train_dataset.set_epoch(epoch)
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
                losses = compute_losses(
                    batch,
                    outputs,
                    cfg["loss"],
                    organizer_scale=org_scale,
                )
                total_loss = losses["loss_total"] / float(active_accum_steps)

            scaler.scale(total_loss).backward()
            accumulation_buffer.append(
                {
                    "loss_total": normalize_loss_scalar(losses["loss_total"]),
                    "loss_field": normalize_loss_scalar(losses["loss_field"]),
                    "loss_mean": normalize_loss_scalar(losses["loss_mean"]),
                    "loss_residual": normalize_loss_scalar(losses["loss_residual"]),
                    "loss_freq": normalize_loss_scalar(losses["loss_freq"]),
                    "loss_org_sparsity": normalize_loss_scalar(losses["loss_org_sparsity"]),
                    "loss_org_entropy": normalize_loss_scalar(losses["loss_org_entropy"]),
                    "loss_org_direct": normalize_loss_scalar(losses["loss_org_direct"]),
                    "loss_org_me": normalize_loss_scalar(losses["loss_org_me"]),
                    "loss_org_mm": normalize_loss_scalar(losses["loss_org_mm"]),
                    "loss_org_consistency": normalize_loss_scalar(losses["loss_org_consistency"]),
                }
            )
            train_bar.set_postfix(
                loss=f"{accumulation_buffer[-1]['loss_total']:.3e}",
                field=f"{accumulation_buffer[-1]['loss_field']:.3e}",
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
            f"{format_metrics(epoch_train_metrics, keys=['loss_total', 'loss_field', 'loss_mean', 'loss_residual', 'loss_freq', 'loss_org_direct'])}"
        )

        if scheduler is not None:
            scheduler.step()
            print(f"[sched] epoch={epoch:03d} lr={optimizer.param_groups[0]['lr']:.6e}")

        val_metrics = {
            "val_total_loss": float("nan"),
            "val_field_mse": float("nan"),
            "val_mean_mse": float("nan"),
            "val_freq_mse": float("nan"),
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
                    max_num_cylinders=max_num_cylinders,
                    max_cases=int(validation_cfg.get("max_cases", len(canonical_val_dataset))),
                    query_batch_size=int(validation_cfg.get("query_batch_size", 32768)),
                    phase_bins_to_eval=int(validation_cfg.get("phase_bins_to_eval", 6)),
                    show_progress=True,
                )
            print(
                f"[val] epoch={epoch:03d} total={val_metrics['val_total_loss']:.6e} "
                f"field_mse={val_metrics['val_field_mse']:.6e} "
                f"mean_mse={val_metrics['val_mean_mse']:.6e} freq_mse={val_metrics['val_freq_mse']:.6e}"
            )
        else:
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
                    "loss_org_sparsity": epoch_train_metrics["loss_org_sparsity"],
                    "loss_org_entropy": epoch_train_metrics["loss_org_entropy"],
                    "loss_org_direct": epoch_train_metrics["loss_org_direct"],
                    "loss_org_me": epoch_train_metrics["loss_org_me"],
                    "loss_org_mm": epoch_train_metrics["loss_org_mm"],
                    "loss_org_consistency": epoch_train_metrics["loss_org_consistency"],
                }
            ),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **round_loss_metrics(val_metrics),
        }
        history_rows.append(epoch_row)
        with history_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(epoch_row)
        save_loss_curve(history_rows, loss_curve_path)
        print(f"[save] wrote epoch summary to {history_csv}")
        print(f"[save] refreshed loss curve at {loss_curve_path}")

        current_val = val_metrics["val_field_mse"]
        is_new_best = bool(math.isfinite(current_val) and current_val < best_val_metric)
        if math.isfinite(current_val) and current_val < best_val_metric:
            best_val_metric = current_val
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "model_config": model_cfg.__dict__,
            "best_val_metric": best_val_metric,
        }
        torch.save(checkpoint, latest_path)
        print(f"[save] wrote epoch checkpoint: {latest_path}")

        if is_new_best:
            torch.save(checkpoint, best_path)
            print(f"[save] new best checkpoint: {best_path} | val_field_mse={best_val_metric:.6e}")
        else:
            print(
                f"[save] best checkpoint unchanged | best_val_metric="
                f"{best_val_metric:.6e}" if math.isfinite(best_val_metric) else "[save] best checkpoint unchanged | best_val_metric=nan"
            )

    train_dataset.close()
    if val_dataset is not None:
        val_dataset.close()
    print(f"Training finished. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
