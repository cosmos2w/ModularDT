from __future__ import annotations

"""
train_gen.py
============

Two-stage training script for the generative multi-cylinder modular-DT model.

This script follows the run-management style of the deterministic demo:
- read a JSON config from Config_Train,
- back up the resolved config,
- create a timestamped run directory,
- write latest/best checkpoints,
- write loss_history.csv / loss_history.json,
- save a loss curve after every epoch.

Training stages
---------------
Stage 1: train ConvResidualAE on canonical-cycle residual fields.
Stage 2: freeze the AE and deterministic modular-DT model, then train a latent
         rectified-flow velocity network conditioned on deterministic organizer
         outputs and deterministic field predictions.

python src/train_gen.py --config train_gen_config_template.json --stage 1 --device cuda:2

The script is intentionally self-contained and uses only the existing packed
HDF5 dataset.  It does not depend on the older sparse-reconstruction baseline.
"""

import argparse
import contextlib
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

from model import ModelConfig, build_model_from_config
from model_gen import (
    ConvResidualAE,
    GridStats,
    LatentEMA,
    LatentRectifiedFlow,
    LatentVelocityUNet,
    build_dense_condition_grid,
    build_global_condition_vector,
    denormalize_grid,
    normalize_grid,
)


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


DEMO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent rectified-flow generator for the multi-cylinder demo.")
    parser.add_argument("--config", type=str, default="train_gen_config_template.json", help="JSON config filename or path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2], help="Optional override for generation.training_stage.")
    parser.add_argument("--reload", action="store_true", help="Resume from latest_model.pt in the newest matching run if possible.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_demo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (DEMO_ROOT / path).resolve()


def resolve_config_path(config_name_or_path: str) -> Path:
    path = Path(config_name_or_path).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    demo_candidate = DEMO_ROOT / path
    if demo_candidate.exists():
        return demo_candidate.resolve()
    for base_name in ("Config_Train", "Configs"):
        candidate = DEMO_ROOT / base_name / path
        if candidate.exists():
            return candidate.resolve()
    return (DEMO_ROOT / "Config_Train" / path).resolve()


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_history_json(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("history", []) if isinstance(payload, dict) else payload
    return list(rows) if isinstance(rows, list) else []


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str | None) -> torch.device:
    if device_arg is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def data_parallel_device_ids(device: torch.device) -> Optional[List[int]]:
    if device.type != "cuda" or torch.cuda.device_count() <= 1:
        return None
    start = 0 if device.index is None else int(device.index)
    return list(range(start, torch.cuda.device_count()))


def maybe_data_parallel(model: nn.Module, device: torch.device, enabled: bool, label: str) -> nn.Module:
    device_ids = data_parallel_device_ids(device) if enabled else None
    if device_ids and len(device_ids) > 1:
        print(f"[setup] using DataParallel for {label} on cuda devices {device_ids}")
        return nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    return model


def sort_case_ids(case_ids: Iterable[str]) -> List[str]:
    def key_fn(case_id: str):
        try:
            return (0, int(case_id))
        except Exception:
            return (1, str(case_id))
    return sorted(case_ids, key=key_fn)


INERT_CHANNEL_ORDER = ("u", "v", "p", "omega")
ACTIVE_CHANNEL_ORDER = ("u", "v", "p", "omega", "temperature")


def decode_string_array(values) -> List[str]:
    arr = np.asarray(values)
    out: List[str] = []
    for item in arr.reshape(-1):
        out.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return out


def channel_order_from_attr(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "mixed":
            return None
        try:
            payload = json.loads(text)
            if isinstance(payload, (list, tuple)):
                return [str(v) for v in payload]
        except json.JSONDecodeError:
            pass
        return [piece.strip() for piece in text.split(",") if piece.strip()]
    return decode_string_array(value)


def get_case_channel_order(grp: h5py.Group, h5_file: h5py.File) -> List[str]:
    if "channel_order" in grp:
        return decode_string_array(grp["channel_order"][...])
    root_order = channel_order_from_attr(h5_file.attrs.get("channel_order"))
    if root_order is not None:
        return root_order
    field_dim = int(grp.attrs.get("field_dim", grp["canonical_cycle"].shape[-1] if "canonical_cycle" in grp else 4))
    return list(ACTIVE_CHANNEL_ORDER if field_dim == 5 else INERT_CHANNEL_ORDER)


def normalize_loss_scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


# -----------------------------------------------------------------------------
# Dataset: canonical-cycle grid snapshots
# -----------------------------------------------------------------------------


@dataclass
class GenCaseMeta:
    """One canonical-cycle snapshot item.

    Each item corresponds to one case and one phase bin from the packed HDF5
    dataset.  This is intentionally case-level rather than point-chunk-level,
    because the latent AE and flow model operate on full regular grids.
    """

    case_id: str
    split: str
    phase_idx: int
    tau: float
    query_time: float
    re_value: float
    num_cylinders: int


class CanonicalResidualGridDataset(Dataset):
    """Dataset returning full-grid residual fields for generative training.

    Output dictionary keys:
        target_grid:  [C, H, W] residual or field target in physical units
        field_grid:   [C, H, W] full ground-truth field
        mean_grid:    [C, H, W] canonical mean field
        x_grid/y_grid:[H, W]
        tau:          [1]
        structure:    Re, cylinder count, padded centers, cylinder mask

    `target_mode` controls what the AE/flow learns:
        residual: field_grid - mean_grid  (recommended)
        field:    field_grid
    """

    def __init__(
        self,
        h5_path: Path,
        split: str,
        *,
        max_num_cylinders: int,
        target_mode: str = "residual",
        phase_stride: int = 1,
        max_cases: int = 0,
        randomize_cylinder_order: bool = False,
        base_seed: int = 42,
        use_heat_power_module_feature: bool = False,
        heat_power_scale: str | float = "auto",
        promote_to_active: bool = False,
        target_channel_order: Optional[Sequence[str]] = None,
        target_field_dim: Optional[int] = None,
        inert_temperature_value: float = 0.0,
        inert_thermal_time_mode: str = "random_active_range",
        active_max_thermal_time: float = 1.0,
    ):
        super().__init__()
        self.h5_path = Path(h5_path).expanduser().resolve()
        self.split = str(split)
        self.max_num_cylinders = int(max_num_cylinders)
        self.target_mode = str(target_mode)
        self.phase_stride = max(1, int(phase_stride))
        self.max_cases = int(max_cases)
        self.randomize_cylinder_order = bool(randomize_cylinder_order)
        self.base_seed = int(base_seed)
        self.use_heat_power_module_feature = bool(use_heat_power_module_feature)
        self.heat_power_scale_config = heat_power_scale
        self.heat_power_scale = 1.0
        self.promote_to_active = bool(promote_to_active)
        self.target_channel_order = list(target_channel_order) if target_channel_order is not None else None
        self.target_field_dim = int(target_field_dim) if target_field_dim is not None else None
        self.inert_temperature_value = float(inert_temperature_value)
        self.inert_thermal_time_mode = str(inert_thermal_time_mode).strip().lower()
        self.active_max_thermal_time = max(float(active_max_thermal_time), 0.0)
        self.channel_order: List[str] = list(INERT_CHANNEL_ORDER)
        self.field_dim = 4
        self.current_epoch = 0
        self._h5: Optional[h5py.File] = None
        self.items: list[GenCaseMeta] = []
        self.case_static: Dict[str, Dict] = {}
        max_abs_heat_power = 0.0

        if self.target_mode not in {"residual", "field"}:
            raise ValueError("target_mode must be 'residual' or 'field'.")
        if self.inert_thermal_time_mode not in {"zero", "random_active_range", "tau"}:
            raise ValueError("inert_thermal_time_mode must be 'zero', 'random_active_range', or 'tau'.")

        with h5py.File(self.h5_path, "r") as h5:
            cases = h5["cases"]
            selected_case_ids = []
            for case_id in sort_case_ids(cases.keys()):
                grp = cases[case_id]
                case_split = grp.attrs.get("split", "all")
                if split not in {"all", case_split}:
                    continue
                if "canonical_cycle" not in grp:
                    continue
                selected_case_ids.append(case_id)
                if self.max_cases > 0 and len(selected_case_ids) >= self.max_cases:
                    break

            for case_id in selected_case_ids:
                grp = cases[case_id]
                centers = np.asarray(grp["cylinder_centers"], dtype=np.float32)
                if centers.shape[0] > self.max_num_cylinders:
                    raise ValueError(f"Case {case_id} has too many cylinders.")
                phase_bins = np.asarray(grp["phase_bin_centers"], dtype=np.float32)
                phase_tau_bins = (
                    np.asarray(grp["phase_tau_centers"], dtype=np.float32)
                    if "phase_tau_centers" in grp
                    else phase_bins
                )
                # Stage-2 conditioning mirrors deterministic training: tau is
                # folded phase_tau; query_time is thermal_time when available,
                # then tau_abs, then legacy tau for inert datasets.
                query_time_bins = (
                    np.asarray(grp["thermal_time_centers"], dtype=np.float32)
                    if "thermal_time_centers" in grp
                    else (
                        np.asarray(grp["tau_abs_centers"], dtype=np.float32)
                        if "tau_abs_centers" in grp
                        else phase_bins
                    )
                )
                re_value = float(grp.attrs["re"])
                num_cyl = int(grp.attrs["num_cylinders"])
                channel_order = get_case_channel_order(grp, h5)
                field_dim = int(grp.attrs.get("field_dim", grp["canonical_cycle"].shape[-1]))
                source_channel_order = channel_order[:field_dim]
                source_field_dim = int(field_dim)
                if self.promote_to_active:
                    if self.target_channel_order is None or self.target_field_dim is None:
                        raise ValueError("promote_to_active requires target_channel_order and target_field_dim.")
                    missing_base = [name for name in INERT_CHANNEL_ORDER if name not in source_channel_order]
                    if missing_base:
                        raise ValueError(f"Cannot promote case {case_id}; missing inert channels {missing_base}.")
                    channel_order = list(self.target_channel_order)[: int(self.target_field_dim)]
                    field_dim = int(self.target_field_dim)
                else:
                    channel_order = source_channel_order
                if not self.items and not self.case_static:
                    self.channel_order = list(channel_order)
                    self.field_dim = int(field_dim)
                elif field_dim != self.field_dim or tuple(channel_order) != tuple(self.channel_order):
                    raise ValueError(
                        f"Mixed channel metadata is not supported in one generative dataset. "
                        f"Expected {self.channel_order} (C={self.field_dim}), got case {case_id} {channel_order} (C={field_dim})."
                    )
                if "heat_powers" in grp:
                    heat_powers = np.asarray(grp["heat_powers"], dtype=np.float32).reshape(-1)
                else:
                    heat_powers = np.zeros((num_cyl,), dtype=np.float32)
                if self.promote_to_active:
                    heat_powers = np.zeros((centers.shape[0],), dtype=np.float32)
                max_abs_heat_power = max(max_abs_heat_power, float(np.max(np.abs(heat_powers))) if heat_powers.size else 0.0)
                max_abs_heat_power = max(
                    max_abs_heat_power,
                    abs(float(grp.attrs.get("thermal_power_min", 0.0))),
                    abs(float(grp.attrs.get("thermal_power_max", 0.0))),
                )
                self.case_static[case_id] = {
                    "centers": centers,
                    "heat_powers": heat_powers,
                    "re": re_value,
                    "num_cylinders": num_cyl,
                    "dominant_frequency": float(grp.attrs.get("dominant_frequency", 0.0)),
                    "x_grid_shape": tuple(grp["x_grid"].shape),
                    "channel_order": channel_order,
                    "source_channel_order": source_channel_order,
                    "source_field_dim": source_field_dim,
                    "field_dim": field_dim,
                }
                for phase_idx in range(0, len(phase_bins), self.phase_stride):
                    tau_value = float(phase_tau_bins[phase_idx])
                    query_time_value = float(query_time_bins[phase_idx])
                    if self.promote_to_active:
                        if self.inert_thermal_time_mode == "zero":
                            query_time_value = 0.0
                        elif self.inert_thermal_time_mode == "tau":
                            query_time_value = tau_value
                        else:
                            rng = np.random.default_rng(self.base_seed + 137 * int(phase_idx) + 1009 * len(self.items))
                            query_time_value = float(rng.uniform(0.0, self.active_max_thermal_time))
                    self.items.append(
                        GenCaseMeta(
                            case_id=case_id,
                            split=str(grp.attrs.get("split", "all")),
                            phase_idx=int(phase_idx),
                            tau=tau_value,
                            query_time=query_time_value,
                            re_value=re_value,
                            num_cylinders=num_cyl,
                        )
                    )
        if str(self.heat_power_scale_config).strip().lower() == "auto":
            self.heat_power_scale = max(float(max_abs_heat_power), 1.0)
        else:
            self.heat_power_scale = max(float(self.heat_power_scale_config), 1e-12)

    def __len__(self) -> int:
        return len(self.items)

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def set_epoch(self, epoch: int) -> None:
        # Used for deterministic cylinder-order randomization per epoch.
        self.current_epoch = int(epoch)

    def _promote_last_dim(self, values: np.ndarray, source_channel_order: Sequence[str]) -> np.ndarray:
        if not self.promote_to_active:
            return values
        if values.shape[-1] == self.field_dim:
            return values
        promoted = np.zeros((*values.shape[:-1], self.field_dim), dtype=np.float32)
        for out_idx, name in enumerate(self.channel_order):
            if name in source_channel_order:
                promoted[..., out_idx] = values[..., source_channel_order.index(name)]
            elif name == "temperature":
                promoted[..., out_idx] = self.inert_temperature_value
            else:
                raise KeyError(f"Cannot promote missing channel {name!r}.")
        return promoted

    def _structure_tensors(self, meta: GenCaseMeta) -> Dict[str, torch.Tensor]:
        static = self.case_static[meta.case_id]
        centers = static["centers"].copy()
        heat_powers = np.asarray(static.get("heat_powers", np.zeros((centers.shape[0],), dtype=np.float32)), dtype=np.float32).copy()
        if self.randomize_cylinder_order and centers.shape[0] > 1:
            rng_seed = self.base_seed + 1000003 * self.current_epoch + 9176 * int(meta.phase_idx)
            rng = np.random.default_rng(rng_seed)
            perm = rng.permutation(centers.shape[0])
            centers = centers[perm]
            heat_powers = heat_powers[perm]

        padded = np.zeros((self.max_num_cylinders, 2), dtype=np.float32)
        mask = np.zeros((self.max_num_cylinders,), dtype=np.float32)
        padded[: centers.shape[0]] = centers
        mask[: centers.shape[0]] = 1.0
        out = {
            "re_values": torch.tensor([meta.re_value], dtype=torch.float32),
            "num_cylinders": torch.tensor([meta.num_cylinders], dtype=torch.float32),
            "centers": torch.from_numpy(padded),
            "cyl_mask": torch.from_numpy(mask),
            "freq_target": torch.tensor([static["dominant_frequency"]], dtype=torch.float32),
        }
        if self.use_heat_power_module_feature:
            padded_powers = np.zeros((self.max_num_cylinders, 1), dtype=np.float32)
            if heat_powers.size:
                padded_powers[: heat_powers.shape[0], 0] = heat_powers / float(self.heat_power_scale)
            out["extra_module"] = torch.from_numpy(padded_powers)
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        meta = self.items[idx]
        h5 = self._get_h5()
        grp = h5["cases"][meta.case_id]
        static = self.case_static[meta.case_id]

        field = np.asarray(grp["canonical_cycle"][meta.phase_idx], dtype=np.float32)  # [H,W,C]
        mean = np.asarray(grp["mean_field"], dtype=np.float32)                        # [H,W,C]
        field = self._promote_last_dim(field, static.get("source_channel_order", static["channel_order"]))
        mean = self._promote_last_dim(mean, static.get("source_channel_order", static["channel_order"]))
        if self.target_mode == "residual":
            target = field - mean
        else:
            target = field

        # Torch convention for conv nets is [C,H,W].
        field_chw = torch.from_numpy(np.moveaxis(field, -1, 0))
        mean_chw = torch.from_numpy(np.moveaxis(mean, -1, 0))
        target_chw = torch.from_numpy(np.moveaxis(target, -1, 0))
        x_grid = torch.from_numpy(np.asarray(grp["x_grid"], dtype=np.float32))
        y_grid = torch.from_numpy(np.asarray(grp["y_grid"], dtype=np.float32))

        out: Dict[str, torch.Tensor | str] = {
            "case_id": meta.case_id,
            "phase_idx": torch.tensor([meta.phase_idx], dtype=torch.long),
            "tau": torch.tensor([meta.tau], dtype=torch.float32),
            "query_time": torch.tensor([meta.query_time], dtype=torch.float32),
            "target_grid": target_chw,
            "field_grid": field_chw,
            "mean_grid": mean_chw,
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
        out.update(self._structure_tensors(meta))
        return out

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def collate_gen_grid(batch: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    """Collate case-level grid snapshots.

    All grids are expected to share H/W and channel count, which is true for the
    packed multi-cylinder demo dataset.
    """
    keys_tensor = [
        "phase_idx", "tau", "query_time", "target_grid", "field_grid", "mean_grid", "x_grid", "y_grid",
        "re_values", "num_cylinders", "centers", "cyl_mask", "freq_target",
    ]
    if any("extra_module" in item for item in batch):
        keys_tensor.append("extra_module")
    out: Dict[str, torch.Tensor | List[str]] = {"case_id": [str(item["case_id"]) for item in batch]}
    for key in keys_tensor:
        out[key] = torch.stack([item[key] for item in batch], dim=0)  # type: ignore[index]
    return out


class MultiSourceGridDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = [ds for ds in datasets if ds is not None and len(ds) > 0]
        if not self.datasets:
            raise ValueError("MultiSourceGridDataset requires at least one non-empty child dataset.")
        self.offsets: List[int] = []
        total = 0
        for ds in self.datasets:
            self.offsets.append(total)
            total += len(ds)
        self.total_len = total
        primary = self.datasets[0]
        self.channel_order = list(getattr(primary, "channel_order", INERT_CHANNEL_ORDER))
        self.field_dim = int(getattr(primary, "field_dim", len(self.channel_order)))

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int):
        idx = int(idx)
        child_idx = max(i for i, offset in enumerate(self.offsets) if offset <= idx)
        return self.datasets[child_idx][idx - self.offsets[child_idx]]

    def set_epoch(self, epoch: int) -> None:
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)  # type: ignore[attr-defined]

    def close(self) -> None:
        for ds in self.datasets:
            if hasattr(ds, "close"):
                ds.close()  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Normalization stats
# -----------------------------------------------------------------------------


@torch.no_grad()
def compute_grid_stats(dataset: CanonicalResidualGridDataset, max_items: int = 0) -> GridStats:
    """Compute channel-wise mean/std over target grids in the training split."""
    count = 0
    sum_c = None
    sumsq_c = None
    n_items = len(dataset) if max_items <= 0 else min(len(dataset), int(max_items))
    for i in tqdm(range(n_items), desc="computing grid stats", leave=False):
        item = dataset[i]
        x = item["target_grid"].float()  # [C,H,W]
        c = x.shape[0]
        flat = x.reshape(c, -1)
        if sum_c is None:
            sum_c = flat.sum(dim=1)
            sumsq_c = flat.square().sum(dim=1)
        else:
            sum_c += flat.sum(dim=1)
            sumsq_c += flat.square().sum(dim=1)
        count += flat.shape[1]
    if sum_c is None or sumsq_c is None:
        raise RuntimeError("Cannot compute stats from an empty dataset.")
    mean = sum_c / float(count)
    var = (sumsq_c / float(count) - mean.square()).clamp_min(1e-8)
    std = var.sqrt().clamp_min(1e-6)
    return GridStats(mean=mean, std=std)


# -----------------------------------------------------------------------------
# Deterministic model conditioning
# -----------------------------------------------------------------------------


def build_structure_from_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    structure = {
        "re_values": batch["re_values"].to(device=device),
        "num_cylinders": batch["num_cylinders"].to(device=device),
        "centers": batch["centers"].to(device=device),
        "cyl_mask": batch["cyl_mask"].to(device=device),
    }
    if "extra_module" in batch:
        structure["extra_module"] = batch["extra_module"].to(device=device)
    return structure


@torch.no_grad()
def deterministic_grid_forward(
    det_model: nn.Module,
    structure: Dict[str, torch.Tensor],
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    tau: torch.Tensor,
    query_time: Optional[torch.Tensor] = None,
    *,
    query_batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Run deterministic model on a full grid for a batch of cases.

    Unlike `reconstruct_full_grid`, this handles B>1 and returns `pred_mean`,
    `pred_residual`, and aux outputs from the first query chunk.
    """
    B, H, W = x_grid.shape
    xy = torch.stack([x_grid.reshape(B, -1), y_grid.reshape(B, -1)], dim=-1)
    tau_full = tau.reshape(B, 1, 1).expand(B, xy.shape[1], 1)
    if query_time is None:
        query_time_full = tau_full
    else:
        query_time_full = query_time.reshape(B, 1, 1).expand(B, xy.shape[1], 1)
    pred_field_chunks, pred_mean_chunks, pred_res_chunks = [], [], []
    aux = None
    for start in range(0, xy.shape[1], int(query_batch_size)):
        end = min(start + int(query_batch_size), xy.shape[1])
        out = det_model(
            structure=structure,
            query_xy=xy[:, start:end],
            query_tau=tau_full[:, start:end],
            query_time=query_time_full[:, start:end],
            return_aux=(aux is None),
        )
        pred_field_chunks.append(out["pred_field"])
        pred_mean_chunks.append(out["pred_mean"])
        pred_res_chunks.append(out["pred_residual"])
        if aux is None:
            aux = {k: v for k, v in out.items() if k not in {"pred_field", "pred_mean", "pred_residual"}}
    pred_field = torch.cat(pred_field_chunks, dim=1).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    pred_mean = torch.cat(pred_mean_chunks, dim=1).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    pred_res = torch.cat(pred_res_chunks, dim=1).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    result = {"pred_field": pred_field, "pred_mean": pred_mean, "pred_residual": pred_res}
    if aux is not None:
        result.update(aux)
    return result


def load_deterministic_model(det_cfg: Dict, device: torch.device) -> tuple[nn.Module, Dict, Path]:
    """Load frozen deterministic model used as the generative conditioner."""
    ckpt_path = resolve_deterministic_checkpoint_path(det_cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Deterministic checkpoint not found: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    if "model_config" in ckpt:
        model_cfg_payload = ckpt["model_config"]
    elif "config" in ckpt and "model" in ckpt["config"]:
        model_cfg_payload = ckpt["config"]["model"]
    else:
        model_cfg_payload = det_cfg.get("model", {})
    model = build_model_from_config(model_cfg_payload)
    state = ckpt.get("model_state_dict", ckpt.get("model", None))
    if state is None:
        raise KeyError("Could not find deterministic model state_dict in checkpoint.")
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    return model, model_cfg_payload, ckpt_path


@torch.no_grad()
def build_stage2_conditions(
    det_model: nn.Module,
    det_model_cfg: Dict,
    batch: Dict[str, torch.Tensor],
    stats: GridStats,
    device: torch.device,
    query_batch_size: int,
    include_field: bool,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute dense and global conditions for one generative training batch."""
    structure = build_structure_from_batch(batch, device)
    if int(det_model_cfg.get("future_module_feature_dim", 0)) > 0 and "extra_module" not in structure:
        bsz, n_max = structure["centers"].shape[:2]
        structure["extra_module"] = torch.zeros(
            (bsz, n_max, int(det_model_cfg.get("future_module_feature_dim", 1))),
            device=device,
            dtype=structure["centers"].dtype,
        )
    x_grid = batch["x_grid"].to(device=device)
    y_grid = batch["y_grid"].to(device=device)
    tau = batch["tau"].to(device=device)
    # The deterministic conditioner receives both aligned coordinates so its
    # temperature head can follow active thermal age.
    query_time = batch.get("query_time", batch["tau"]).to(device=device)
    det_out = deterministic_grid_forward(
        det_model,
        structure,
        x_grid,
        y_grid,
        tau,
        query_time=query_time,
        query_batch_size=query_batch_size,
    )
    cond_grid = build_dense_condition_grid(
        det_mean=det_out["pred_mean"],
        det_residual=det_out["pred_residual"],
        det_field=det_out["pred_field"],
        x_grid=x_grid,
        y_grid=y_grid,
        tau=tau,
        thermal_time=query_time,
        re_values=structure["re_values"],
        stats=stats.to(device, dtype=det_out["pred_mean"].dtype),
        domain_length_x=float(det_model_cfg.get("domain_length_x", 24.0)),
        domain_length_y=float(det_model_cfg.get("domain_length_y", 12.0)),
        re_scale=float(det_model_cfg.get("re_scale", 200.0)),
        include_field=include_field,
    )
    global_cond = build_global_condition_vector(det_out, structure)
    return cond_grid, global_cond, det_out


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------


def infer_grid_shape(dataset: CanonicalResidualGridDataset) -> tuple[int, int, int]:
    sample = dataset[0]
    c, h, w = sample["target_grid"].shape  # type: ignore[index]
    return int(c), int(h), int(w)


@torch.no_grad()
def infer_condition_dims(
    cfg: Dict,
    dataset: CanonicalResidualGridDataset,
    det_model: nn.Module,
    det_model_cfg: Dict,
    stats: GridStats,
    device: torch.device,
) -> tuple[int, int]:
    """Run one tiny deterministic conditioning pass to infer cond-grid channels and global-cond width."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_gen_grid)
    batch = next(iter(loader))
    cond_grid, global_cond, _ = build_stage2_conditions(
        det_model=det_model,
        det_model_cfg=det_model_cfg,
        batch=batch,  # type: ignore[arg-type]
        stats=stats,
        device=device,
        query_batch_size=int(cfg["generation"].get("det_query_batch_size", 32768)),
        include_field=bool(cfg["stage2"]["conditioning"].get("include_pred_field", True)),
    )
    return int(cond_grid.shape[1]), int(global_cond.shape[1])


def build_ae_from_cfg(cfg: Dict, n_fields: int, num_y: int, num_x: int) -> ConvResidualAE:
    arch = cfg["stage1"]["architecture"]
    return ConvResidualAE(
        n_fields=n_fields,
        base_ch=int(arch.get("base_ch", 48)),
        latent_ch=int(arch.get("latent_ch", 96)),
        n_levels=int(arch.get("n_levels", 3)),
        num_res_blocks=int(arch.get("num_res_blocks", 1)),
        num_y=num_y,
        num_x=num_x,
    )


def build_ae_from_stage1_checkpoint(ckpt: Dict) -> ConvResidualAE:
    """Rebuild the AE with the exact architecture saved by stage 1."""
    arch = ckpt.get("ae_config") or ckpt.get("config", {}).get("stage1", {}).get("architecture", {})
    return ConvResidualAE(
        n_fields=int(ckpt.get("n_fields", 4)),
        base_ch=int(arch.get("base_ch", 48)),
        latent_ch=int(arch.get("latent_ch", 96)),
        n_levels=int(arch.get("n_levels", 3)),
        num_res_blocks=int(arch.get("num_res_blocks", 1)),
        num_y=int(ckpt["num_y"]),
        num_x=int(ckpt["num_x"]),
    )


def build_flow_from_cfg(cfg: Dict, ae: ConvResidualAE, cond_ch: int, global_cond_dim: int) -> LatentRectifiedFlow:
    arch = cfg["stage2"]["architecture"]
    velocity = LatentVelocityUNet(
        latent_ch=ae.latent_ch,
        cond_ch=cond_ch,
        global_cond_dim=global_cond_dim,
        base_ch=int(arch.get("base_ch", 192)),
        ch_mult=tuple(arch.get("ch_mult", [1, 2])),
        num_res_blocks=int(arch.get("num_res_blocks", 2)),
        num_heads=int(arch.get("num_heads", 4)),
        dropout=float(arch.get("dropout", 0.0)),
    )
    return LatentRectifiedFlow(ae=ae, velocity_net=velocity, cond_downsample_mode=arch.get("cond_downsample_mode", "area"))


# -----------------------------------------------------------------------------
# Epoch routines
# -----------------------------------------------------------------------------


def train_or_eval_ae_epoch(
    ae: ConvResidualAE,
    loader: DataLoader,
    stats: GridStats,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    training: bool,
    epoch: int,
    loss_weights: Dict,
) -> Dict[str, float]:
    ae.train(training)
    rows = []
    pbar = tqdm(loader, desc=f"AE epoch {epoch:04d} {'train' if training else 'val'}", leave=False)
    for batch in pbar:
        target = batch["target_grid"].to(device=device)
        target_norm = normalize_grid(target, stats.to(device, target.dtype))
        if training:
            optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
        with torch.set_grad_enabled(training):
            recon_norm, _ = ae(target_norm)
            loss_mse = F.mse_loss(recon_norm, target_norm)
            loss_l1 = F.l1_loss(recon_norm, target_norm)
            loss = float(loss_weights.get("mse_weight", 1.0)) * loss_mse + float(loss_weights.get("l1_weight", 0.25)) * loss_l1
        if training:
            loss.backward()
            grad_clip = float(loss_weights.get("grad_clip", 1.0))
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(ae.parameters(), grad_clip)
            optimizer.step()  # type: ignore[union-attr]
        row = {"loss": normalize_loss_scalar(loss), "mse": normalize_loss_scalar(loss_mse), "l1": normalize_loss_scalar(loss_l1)}
        rows.append(row)
        pbar.set_postfix(loss=f"{row['loss']:.3e}")
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()} if rows else {"loss": float("nan"), "mse": float("nan"), "l1": float("nan")}


def train_or_eval_flow_epoch(
    flow: LatentRectifiedFlow,
    det_model: nn.Module,
    det_model_cfg: Dict,
    loader: DataLoader,
    stats: GridStats,
    optimizer: Optional[torch.optim.Optimizer],
    ema: Optional[LatentEMA],
    device: torch.device,
    training: bool,
    epoch: int,
    cfg: Dict,
) -> Dict[str, float]:
    flow.ae.eval().requires_grad_(False)
    flow.velocity_net.train(training)
    rows = []
    pbar = tqdm(loader, desc=f"Flow epoch {epoch:04d} {'train' if training else 'val'}", leave=False)
    velocity_core = unwrap_model(flow.velocity_net)
    ema_context = ema.average_parameters(velocity_core) if (not training and ema is not None) else contextlib.nullcontext()
    with ema_context:
        for batch in pbar:
            target = batch["target_grid"].to(device=device)
            target_norm = normalize_grid(target, stats.to(device, target.dtype))
            cond_grid, global_cond, _ = build_stage2_conditions(
                det_model=det_model,
                det_model_cfg=det_model_cfg,
                batch=batch,
                stats=stats,
                device=device,
                query_batch_size=int(cfg["generation"].get("det_query_batch_size", 32768)),
                include_field=bool(cfg["stage2"]["conditioning"].get("include_pred_field", True)),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
            with torch.set_grad_enabled(training):
                loss, info = flow.training_loss(target_norm, cond_grid, global_cond)
            if training:
                loss.backward()
                grad_clip = float(cfg["stage2"]["training"].get("gradient_clip_norm", 1.0))
                if grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(flow.velocity_net.parameters(), grad_clip)
                optimizer.step()  # type: ignore[union-attr]
                if ema is not None:
                    ema.update(velocity_core)
            row = {"loss": normalize_loss_scalar(loss), "target_rms": info["target_rms"], "pred_rms": info["pred_rms"]}
            rows.append(row)
            pbar.set_postfix(loss=f"{row['loss']:.3e}", pred_rms=f"{row['pred_rms']:.3e}")
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()} if rows else {"loss": float("nan"), "target_rms": float("nan"), "pred_rms": float("nan")}


# -----------------------------------------------------------------------------
# Logging and plotting
# -----------------------------------------------------------------------------


def save_history_csv_json(history: List[Dict[str, float]], csv_path: Path, json_path: Path) -> None:
    if not history:
        return
    keys = sorted({k for row in history for k in row.keys()})
    if "epoch" in keys:
        keys.remove("epoch")
        keys = ["epoch"] + keys
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    write_json(json_path, {"history": history})


def save_loss_plot(history: List[Dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for key, label in [
        ("train_loss", "Train"),
        ("val_loss", "Validation"),
        ("train_mse", "Train MSE"),
        ("val_mse", "Val MSE"),
    ]:
        vals = [row.get(key, float("nan")) for row in history]
        if any(math.isfinite(float(v)) and float(v) > 0 for v in vals):
            ax.plot(epochs, vals, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def find_latest_gen_run(save_root: Path, case_id: str, stage: int) -> Optional[Path]:
    prefix = f"Gen_Case{case_id}_Stage{stage}_"
    candidates = sorted([p for p in save_root.glob(prefix + "*") if p.is_dir()])
    return candidates[-1] if candidates else None


def is_auto_checkpoint_path(path_value: object, placeholder: str) -> bool:
    if path_value is None:
        return True
    text = str(path_value).strip()
    if text == "":
        return True
    if text.lower() in {"auto", "latest", "newest"}:
        return True
    return placeholder in text


def is_auto_stage1_path(path_value: object) -> bool:
    return is_auto_checkpoint_path(path_value, "<STAGE1_RUN>")


def is_auto_deterministic_path(path_value: object) -> bool:
    return is_auto_checkpoint_path(path_value, "<DETERMINISTIC_RUN>")


def find_latest_deterministic_checkpoint(det_cfg: Dict) -> Optional[Path]:
    save_root = resolve_demo_path(det_cfg.get("saved_model_dir", "./Saved_Model"))
    candidates: List[Tuple[str, float, Path]] = []
    for run_dir in save_root.glob("Case*"):
        if not run_dir.is_dir():
            continue
        best_path = run_dir / "best_model.pt"
        latest_path = run_dir / "latest_model.pt"
        ckpt_path = best_path if best_path.exists() else latest_path
        if ckpt_path.exists():
            candidates.append((run_dir.name, ckpt_path.stat().st_mtime, ckpt_path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2].resolve()


def resolve_deterministic_checkpoint_path(det_cfg: Dict) -> Path:
    raw_path = det_cfg.get("checkpoint_path")
    if not is_auto_deterministic_path(raw_path):
        ckpt_path = resolve_demo_path(str(raw_path))
        if ckpt_path.exists():
            return ckpt_path
        raise FileNotFoundError(
            f"Deterministic checkpoint not found: {ckpt_path}. "
            "Set deterministic_model.checkpoint_path to an existing checkpoint, or use 'auto' to load the newest deterministic run."
        )

    ckpt_path = find_latest_deterministic_checkpoint(det_cfg)
    save_root = resolve_demo_path(det_cfg.get("saved_model_dir", "./Saved_Model"))
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No deterministic checkpoint found under {save_root}. "
            "Run deterministic training first, or set deterministic_model.checkpoint_path to a specific checkpoint."
        )
    det_cfg["checkpoint_path"] = str(ckpt_path)
    print(f"[setup] auto-selected newest deterministic checkpoint: {ckpt_path}")
    return ckpt_path


def find_latest_stage1_checkpoint(save_root: Path, case_id: str) -> Optional[Path]:
    prefix = f"Gen_Case{case_id}_Stage1_"
    candidates: List[Tuple[str, float, Path]] = []
    for run_dir in save_root.glob(prefix + "*"):
        if not run_dir.is_dir():
            continue
        best_path = run_dir / "best_model.pt"
        latest_path = run_dir / "latest_model.pt"
        ckpt_path = best_path if best_path.exists() else latest_path
        if ckpt_path.exists():
            candidates.append((run_dir.name, ckpt_path.stat().st_mtime, ckpt_path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2].resolve()


def resolve_stage1_checkpoint_path(stage2_cfg: Dict, save_root: Path, case_id: str) -> Path:
    raw_path = stage2_cfg.get("stage1_checkpoint_path")
    if not is_auto_stage1_path(raw_path):
        stage1_path = resolve_demo_path(str(raw_path))
        if stage1_path.exists():
            return stage1_path
        raise FileNotFoundError(
            f"Stage-1 checkpoint not found: {stage1_path}. "
            "Set stage2.stage1_checkpoint_path to an existing checkpoint, or use 'auto' to load the newest Stage1 run."
        )

    stage1_path = find_latest_stage1_checkpoint(save_root, case_id)
    if stage1_path is None:
        raise FileNotFoundError(
            f"No trained stage-1 checkpoint found under {save_root} for case_id={case_id!r}. "
            "Run stage 1 first, or set stage2.stage1_checkpoint_path to a specific checkpoint."
        )
    stage2_cfg["stage1_checkpoint_path"] = str(stage1_path)
    print(f"[setup] auto-selected newest stage-1 checkpoint: {stage1_path}")
    return stage1_path


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if args.stage is not None:
        cfg.setdefault("generation", {})["training_stage"] = int(args.stage)
    if args.reload:
        cfg.setdefault("training", {})["reload"] = True

    stage = int(cfg["generation"].get("training_stage", 1))
    case_id = str(cfg.get("case_id", "gen"))
    target_mode = str(cfg["generation"].get("target_mode", "residual"))
    if stage == 2 and target_mode != "residual":
        raise ValueError("Stage 2 currently implements deterministic mean + generated residual, so generation.target_mode must be 'residual'.")
    set_seed(int(cfg["training"].get("seed", 42)))
    device = select_device(args.device)

    dataset_cfg = cfg["dataset"]
    packed_path = resolve_demo_path(dataset_cfg["packed_h5_path"])
    if not packed_path.exists():
        raise FileNotFoundError(f"Packed dataset not found: {packed_path}")
    use_inert_effective = bool(dataset_cfg.get("USE_INERT", False))
    inert_packed_path: Optional[Path] = None
    active_max_thermal_time = 1.0
    if use_inert_effective:
        from train import check_inert_active_dataset_compatibility, infer_active_max_thermal_time

        inert_packed_path = resolve_demo_path(dataset_cfg.get("inert_packed_h5_path", "./Data_Saved/Processed_Inert_Dataset/packed_dataset.h5"))
        if not inert_packed_path.exists():
            message = f"USE_INERT requested but inert packed dataset was not found: {inert_packed_path}"
            if bool(dataset_cfg.get("fallback_to_active_only_on_mismatch", True)):
                print(f"[WARN] {message}; falling back to active-only generative training.")
                use_inert_effective = False
            else:
                raise FileNotFoundError(message)
        if use_inert_effective and inert_packed_path is not None:
            ok, warnings = check_inert_active_dataset_compatibility(packed_path, inert_packed_path)
            for warning in warnings:
                print(f"[dataset-compat] {warning}")
            if not ok:
                if bool(dataset_cfg.get("fallback_to_active_only_on_mismatch", True)):
                    print("[WARN] Active/inert datasets are incompatible; falling back to active-only generative training.")
                    use_inert_effective = False
                else:
                    raise ValueError("Active/inert datasets are incompatible for combined generative training.")
        if use_inert_effective:
            configured_max = dataset_cfg.get("active_max_thermal_time", "auto")
            active_max_thermal_time = infer_active_max_thermal_time(packed_path) if str(configured_max).strip().lower() == "auto" else float(configured_max)

    # Build train/validation datasets.  Stage 2 uses the same target grids as
    # stage 1 but adds deterministic conditioning in the epoch loop.
    train_set = CanonicalResidualGridDataset(
        packed_path,
        split=dataset_cfg.get("train_split", "train"),
        max_num_cylinders=int(dataset_cfg.get("max_num_cylinders", 8)),
        target_mode=target_mode,
        phase_stride=int(dataset_cfg.get("train_phase_stride", 1)),
        max_cases=int(dataset_cfg.get("train_max_cases", 0)),
        randomize_cylinder_order=bool(dataset_cfg.get("randomize_cylinder_order", True)),
        base_seed=int(cfg["training"].get("seed", 42)),
        use_heat_power_module_feature=bool(dataset_cfg.get("use_heat_power_module_feature", False)),
        heat_power_scale=dataset_cfg.get("heat_power_scale", "auto"),
    )
    val_set = CanonicalResidualGridDataset(
        packed_path,
        split=dataset_cfg.get("val_split", "test"),
        max_num_cylinders=int(dataset_cfg.get("max_num_cylinders", 8)),
        target_mode=target_mode,
        phase_stride=int(dataset_cfg.get("val_phase_stride", 2)),
        max_cases=int(dataset_cfg.get("val_max_cases", 16)),
        randomize_cylinder_order=False,
        base_seed=int(cfg["training"].get("seed", 42)),
        use_heat_power_module_feature=bool(dataset_cfg.get("use_heat_power_module_feature", False)),
        heat_power_scale=dataset_cfg.get("heat_power_scale", "auto"),
    )
    if use_inert_effective:
        assert inert_packed_path is not None
        inert_train_set = CanonicalResidualGridDataset(
            inert_packed_path,
            split=dataset_cfg.get("inert_split", "all"),
            max_num_cylinders=int(dataset_cfg.get("max_num_cylinders", 8)),
            target_mode=target_mode,
            phase_stride=int(dataset_cfg.get("train_phase_stride", 1)),
            max_cases=int(dataset_cfg.get("inert_train_max_cases", 0)),
            randomize_cylinder_order=bool(dataset_cfg.get("randomize_cylinder_order", True)),
            base_seed=int(cfg["training"].get("seed", 42)) + 777,
            use_heat_power_module_feature=bool(dataset_cfg.get("use_heat_power_module_feature", False)),
            heat_power_scale=dataset_cfg.get("heat_power_scale", "auto"),
            promote_to_active=bool(dataset_cfg.get("promote_inert_to_active", True)),
            target_channel_order=getattr(train_set, "channel_order", ACTIVE_CHANNEL_ORDER),
            target_field_dim=int(getattr(train_set, "field_dim", 5)),
            inert_temperature_value=float(dataset_cfg.get("inert_temperature_value", 0.0)),
            inert_thermal_time_mode=str(dataset_cfg.get("inert_thermal_time_mode", "random_active_range")),
            active_max_thermal_time=active_max_thermal_time,
        )
        train_set = MultiSourceGridDataset([train_set, inert_train_set])
        if bool(dataset_cfg.get("use_inert_for_val", False)):
            inert_val_set = CanonicalResidualGridDataset(
                inert_packed_path,
                split=dataset_cfg.get("inert_split", "all"),
                max_num_cylinders=int(dataset_cfg.get("max_num_cylinders", 8)),
                target_mode=target_mode,
                phase_stride=int(dataset_cfg.get("val_phase_stride", 2)),
                max_cases=int(dataset_cfg.get("inert_val_max_cases", dataset_cfg.get("val_max_cases", 16))),
                randomize_cylinder_order=False,
                base_seed=int(cfg["training"].get("seed", 42)) + 777,
                use_heat_power_module_feature=bool(dataset_cfg.get("use_heat_power_module_feature", False)),
                heat_power_scale=dataset_cfg.get("heat_power_scale", "auto"),
                promote_to_active=bool(dataset_cfg.get("promote_inert_to_active", True)),
                target_channel_order=getattr(train_set, "channel_order", ACTIVE_CHANNEL_ORDER),
                target_field_dim=int(getattr(train_set, "field_dim", 5)),
                inert_temperature_value=float(dataset_cfg.get("inert_temperature_value", 0.0)),
                inert_thermal_time_mode=str(dataset_cfg.get("inert_thermal_time_mode", "random_active_range")),
                active_max_thermal_time=active_max_thermal_time,
            )
            val_set = MultiSourceGridDataset([val_set, inert_val_set])
        else:
            print("[setup] generative validation remains active-only (use_inert_for_val=false).")
    if len(train_set) == 0:
        raise RuntimeError("No generative training snapshots found.")
    n_fields, num_y, num_x = infer_grid_shape(train_set)
    channel_order = list(getattr(train_set, "channel_order", ACTIVE_CHANNEL_ORDER if n_fields == 5 else INERT_CHANNEL_ORDER))
    requested_field_dim = dataset_cfg.get("field_dim", "auto")
    if str(requested_field_dim).strip().lower() not in {"auto", str(n_fields)}:
        raise ValueError(f"dataset.field_dim={requested_field_dim} does not match packed dataset field_dim={n_fields}.")
    cfg.setdefault("dataset", {})["field_dim"] = int(n_fields)
    cfg["dataset"]["channel_order"] = channel_order
    dataset_cfg["field_dim"] = int(n_fields)
    dataset_cfg["channel_order"] = channel_order

    batch_size = int((cfg["stage1"] if stage == 1 else cfg["stage2"])["training"].get("batch_size", 4))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_gen_grid, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_gen_grid, pin_memory=torch.cuda.is_available())

    save_root = ensure_dir(resolve_demo_path(cfg["paths"].get("saved_model_dir", "./Saved_Model_Gen")))
    stage1_path = resolve_stage1_checkpoint_path(cfg["stage2"], save_root, case_id) if stage == 2 else None
    det_ckpt_path = resolve_deterministic_checkpoint_path(cfg["deterministic_model"]) if stage == 2 else None
    config_train_dir = ensure_dir(resolve_demo_path(cfg["paths"].get("config_train_dir", "./Config_Train")))
    backup_dir = ensure_dir(config_train_dir / "Configs_gen_bk")
    timestamp = current_timestamp()
    reload_requested = bool(cfg.get("training", {}).get("reload", False))
    resume_run = find_latest_gen_run(save_root, case_id, stage) if reload_requested else None
    if resume_run is not None and (resume_run / "latest_model.pt").exists():
        run_dir = resume_run
        print(f"[setup] reload requested; resuming newest run: {run_dir}")
    else:
        if reload_requested:
            print("[setup] reload requested, but no latest_model.pt was found; starting a new run.")
        run_dir = ensure_dir(save_root / f"Gen_Case{case_id}_Stage{stage}_{timestamp}")

    resolved_name = "resolved_train_gen_config.json" if not reload_requested else f"resolved_train_gen_config_resume_{timestamp}.json"
    write_json(run_dir / resolved_name, cfg)
    backup_path = backup_dir / f"Config_Gen_Case{case_id}_Stage{stage}_{timestamp}.json"
    write_json(backup_path, cfg)

    # Stats are computed from stage-1 training targets and reused in stage 2.
    stage1_ckpt = None
    if stage == 1:
        stats = compute_grid_stats(train_set, max_items=int(cfg["stage1"].get("stats_max_items", 0)))
    else:
        assert stage1_path is not None
        stage1_ckpt = safe_torch_load(stage1_path, map_location="cpu")
        if int(stage1_ckpt.get("stage", 1)) != 1:
            raise ValueError(f"stage2.stage1_checkpoint_path must point to a stage-1 AE checkpoint, got stage={stage1_ckpt.get('stage')}.")
        stage1_target_mode = str(stage1_ckpt.get("config", {}).get("generation", {}).get("target_mode", target_mode))
        if stage1_target_mode != target_mode:
            raise ValueError(
                f"Stage-1 checkpoint target_mode={stage1_target_mode!r} does not match current target_mode={target_mode!r}."
            )
        if int(stage1_ckpt.get("n_fields", n_fields)) != n_fields or int(stage1_ckpt["num_y"]) != num_y or int(stage1_ckpt["num_x"]) != num_x:
            raise ValueError("Stage-1 checkpoint grid shape does not match the current dataset/config.")
        ckpt_channel_order = [str(v) for v in stage1_ckpt.get("channel_order", channel_order)]
        if ckpt_channel_order != channel_order:
            raise ValueError(f"Stage-1 checkpoint channel_order={ckpt_channel_order} does not match dataset channel_order={channel_order}.")
        stats = GridStats(mean=stage1_ckpt["stats"]["mean"], std=stage1_ckpt["stats"]["std"])

    latest_path = run_dir / "latest_model.pt"
    best_path = run_dir / "best_model.pt"
    history: List[Dict[str, float]] = []
    start_epoch = 1
    best_val = float("inf")

    if stage == 1:
        ae = build_ae_from_cfg(cfg, n_fields=n_fields, num_y=num_y, num_x=num_x).to(device)
        training_cfg = cfg["stage1"]["training"]
        use_data_parallel = bool(training_cfg.get("use_data_parallel", cfg.get("training", {}).get("use_data_parallel", True)))
        ae = maybe_data_parallel(ae, device, use_data_parallel, "stage-1 autoencoder")
        optimizer = torch.optim.AdamW(ae.parameters(), lr=float(training_cfg.get("learning_rate", 2e-4)), weight_decay=float(training_cfg.get("weight_decay", 1e-5)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training_cfg.get("scheduler_t_max", training_cfg.get("epochs", 500))), eta_min=float(training_cfg.get("scheduler_min_lr", 1e-6)))
        flow = None
        det_model = None
        det_model_cfg = None
        ema = None
    else:
        det_model, det_model_cfg, det_ckpt_path = load_deterministic_model(cfg["deterministic_model"], device)
        use_data_parallel = bool(cfg.get("stage2", {}).get("training", {}).get("use_data_parallel", cfg.get("training", {}).get("use_data_parallel", True)))
        det_model = maybe_data_parallel(det_model, device, use_data_parallel, "deterministic conditioner")
        det_field_dim = int(det_model_cfg.get("field_dim", getattr(unwrap_model(det_model), "cfg", ModelConfig()).field_dim))
        if det_field_dim != int(n_fields):
            raise ValueError(
                f"Deterministic checkpoint field_dim={det_field_dim} does not match generative dataset field_dim={n_fields} "
                f"for channel_order={channel_order}."
            )
        if stage1_ckpt is None:
            assert stage1_path is not None
            stage1_ckpt = safe_torch_load(stage1_path, map_location="cpu")
        ae = build_ae_from_stage1_checkpoint(stage1_ckpt).to(device)
        ae.load_state_dict(stage1_ckpt["ae_state_dict"])
        ae.eval().requires_grad_(False)
        cond_ch, global_cond_dim = infer_condition_dims(cfg, train_set, det_model, det_model_cfg, stats, device)
        flow = build_flow_from_cfg(cfg, ae, cond_ch=cond_ch, global_cond_dim=global_cond_dim).to(device)
        training_cfg = cfg["stage2"]["training"]
        use_data_parallel = bool(training_cfg.get("use_data_parallel", cfg.get("training", {}).get("use_data_parallel", True)))
        flow.velocity_net = maybe_data_parallel(flow.velocity_net, device, use_data_parallel, "stage-2 velocity network")
        optimizer = torch.optim.AdamW(flow.velocity_net.parameters(), lr=float(training_cfg.get("learning_rate", 1e-4)), weight_decay=float(training_cfg.get("weight_decay", 1e-5)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training_cfg.get("scheduler_t_max", training_cfg.get("epochs", 1000))), eta_min=float(training_cfg.get("scheduler_min_lr", 1e-6)))
        ema = LatentEMA(unwrap_model(flow.velocity_net), decay=float(cfg["stage2"]["architecture"].get("ema_decay", 0.999)))
        print(f"[setup] inferred cond_ch={cond_ch}, global_cond_dim={global_cond_dim}")

    if reload_requested and latest_path.exists():
        resume_ckpt = safe_torch_load(latest_path, map_location=device)
        if int(resume_ckpt.get("stage", stage)) != stage:
            raise ValueError(f"Cannot resume stage {stage} from checkpoint with stage={resume_ckpt.get('stage')}.")
        if "stats" in resume_ckpt:
            stats = GridStats(mean=resume_ckpt["stats"]["mean"].detach().cpu(), std=resume_ckpt["stats"]["std"].detach().cpu())
        if stage == 1:
            unwrap_model(ae).load_state_dict(resume_ckpt["ae_state_dict"])
        else:
            velocity_core = unwrap_model(flow.velocity_net)
            if int(resume_ckpt.get("cond_ch", velocity_core.cond_ch)) != velocity_core.cond_ch:
                raise ValueError("Cannot resume: checkpoint cond_ch differs from the current deterministic conditioner.")
            if int(resume_ckpt.get("global_cond_dim", velocity_core.global_cond_dim)) != velocity_core.global_cond_dim:
                raise ValueError("Cannot resume: checkpoint global_cond_dim differs from the current deterministic conditioner.")
            velocity_core.load_state_dict(resume_ckpt["velocity_state_dict"])
            if "ae_state_dict" in resume_ckpt:
                unwrap_model(ae).load_state_dict(resume_ckpt["ae_state_dict"])
            if ema is not None and resume_ckpt.get("ema_state_dict") is not None:
                ema.load_state_dict(resume_ckpt["ema_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_val = float(resume_ckpt.get("best_val_loss", best_val))
        history = read_history_json(run_dir / "loss_history.json")
        print(f"[setup] resumed from epoch {start_epoch - 1}; next epoch={start_epoch}")

    total_epochs = int(training_cfg.get("epochs", 500))
    eval_every = max(1, int(training_cfg.get("eval_every_epochs", 1)))

    print(f"[setup] stage={stage} device={device} train_snapshots={len(train_set)} val_snapshots={len(val_set)} grid=({num_y},{num_x}) fields={n_fields}")
    print(f"[setup] run_dir={run_dir}")
    print(f"[setup] saved resolved config to {run_dir / resolved_name}")
    print(f"[setup] backed up training config to {backup_path}")

    for epoch in range(start_epoch, total_epochs + 1):
        train_set.set_epoch(epoch)
        if stage == 1:
            train_metrics = train_or_eval_ae_epoch(ae, train_loader, stats, optimizer, device, True, epoch, cfg["stage1"].get("loss", {}))
            val_metrics = train_or_eval_ae_epoch(ae, val_loader, stats, None, device, False, epoch, cfg["stage1"].get("loss", {})) if epoch == 1 or epoch % eval_every == 0 else {"loss": float("nan"), "mse": float("nan"), "l1": float("nan")}
        else:
            train_metrics = train_or_eval_flow_epoch(flow, det_model, det_model_cfg, train_loader, stats, optimizer, ema, device, True, epoch, cfg)  # type: ignore[arg-type]
            val_metrics = train_or_eval_flow_epoch(flow, det_model, det_model_cfg, val_loader, stats, None, ema, device, False, epoch, cfg) if epoch == 1 or epoch % eval_every == 0 else {"loss": float("nan"), "target_rms": float("nan"), "pred_rms": float("nan")}

        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics.get("loss", float("nan"))),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        for k, v in train_metrics.items():
            row[f"train_{k}"] = float(v)
        for k, v in val_metrics.items():
            row[f"val_{k}"] = float(v)
        history.append(row)
        save_history_csv_json(history, run_dir / "loss_history.csv", run_dir / "loss_history.json")
        save_loss_plot(history, run_dir / "loss_curve.png")

        is_new_best = math.isfinite(row["val_loss"]) and row["val_loss"] < best_val
        if is_new_best:
            best_val = row["val_loss"]

        if stage == 1:
            ae_core = unwrap_model(ae)
            checkpoint = {
                "stage": 1,
                "epoch": epoch,
                "ae_state_dict": ae_core.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val,
                "config": cfg,
                "stats": {"mean": stats.mean.cpu(), "std": stats.std.cpu()},
                "n_fields": n_fields,
                "field_dim": n_fields,
                "channel_order": channel_order,
                "num_y": num_y,
                "num_x": num_x,
                "ae_config": {"base_ch": ae_core.base_ch, "latent_ch": ae_core.latent_ch, "n_levels": ae_core.n_levels, "num_res_blocks": ae_core.num_res_blocks},
                "method": "ConvResidualAE",
            }
        else:
            ae_core = unwrap_model(ae)
            velocity_core = unwrap_model(flow.velocity_net)
            checkpoint = {
                "stage": 2,
                "epoch": epoch,
                "velocity_state_dict": velocity_core.state_dict(),
                "ema_state_dict": ema.state_dict() if ema is not None else None,
                "ae_state_dict": ae_core.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val,
                "config": cfg,
                "stats": {"mean": stats.mean.cpu(), "std": stats.std.cpu()},
                "n_fields": n_fields,
                "field_dim": n_fields,
                "channel_order": channel_order,
                "num_y": num_y,
                "num_x": num_x,
                "ae_config": {"base_ch": ae_core.base_ch, "latent_ch": ae_core.latent_ch, "n_levels": ae_core.n_levels, "num_res_blocks": ae_core.num_res_blocks},
                "cond_ch": velocity_core.cond_ch,
                "global_cond_dim": velocity_core.global_cond_dim,
                "deterministic_checkpoint_path": str(det_ckpt_path),
                "method": "LatentRectifiedFlow_ModularDT",
            }
        torch.save(checkpoint, latest_path)
        if is_new_best:
            torch.save(checkpoint, best_path)

        print(f"[epoch {epoch:04d}] train={row['train_loss']:.6e} val={row['val_loss']:.6e} best={best_val:.6e}")

    train_set.close()
    val_set.close()
    print(f"Training complete. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
