from __future__ import annotations

"""
Train the generative inverse-design model for the inert multi-cylinder demo.

python src/train_inverse.py \
  --config train_inverse_config_template.json \
  --device cuda:0

"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from inverse_kpi import (
    DEFAULT_KPI_NAMES,
    augment_kpi_targets_for_training,
    build_target_spec_vector,
    compute_cycle_kpis,
    kpi_vector_from_dict,
    score_candidate_kpis,
)
from model import build_model_from_config
from model_inverse import (
    HypergraphInverseDesignFlow,
    InverseModelConfig,
    encode_design_vector,
    periodic_min_distance,
)


DEMO_ROOT = Path(__file__).resolve().parent.parent
INERT_CHANNEL_ORDER = ("u", "v", "p", "omega")
ACTIVE_CHANNEL_ORDER = ("u", "v", "p", "omega", "temperature")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train amortized inverse-design rectified flow.")
    parser.add_argument("--config", type=str, default="train_inverse_config_template.json", help="JSON config filename or path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, for example cpu, cuda, cuda:0.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for training.epochs.")
    parser.add_argument("--train-max-cases", type=int, default=None, help="Optional tiny-subset training override.")
    parser.add_argument("--val-max-cases", type=int, default=None, help="Optional tiny-subset validation override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional override for training.num_workers.")
    parser.add_argument("--no-forward-verify", action="store_true", help="Disable optional validation-time forward verification.")
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
    return (DEMO_ROOT / "Config_Train" / path).resolve()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[setup] requested {device_arg}, but CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return device


def sort_case_ids(case_ids: Iterable[str]) -> List[str]:
    def key_fn(case_id: str) -> Tuple[int, Any]:
        try:
            return (0, int(case_id))
        except (TypeError, ValueError):
            return (1, str(case_id))

    return sorted(case_ids, key=key_fn)


def decode_string_array(values: Any) -> List[str]:
    arr = np.asarray(values)
    out: List[str] = []
    for item in arr.reshape(-1):
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def channel_order_from_attr(value: Any) -> Optional[List[str]]:
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


def grid_domain_lengths(x_grid: np.ndarray, y_grid: np.ndarray, grp: Optional[h5py.Group] = None) -> Tuple[float, float]:
    if grp is not None and "lx" in grp.attrs and "ly" in grp.attrs:
        return float(grp.attrs["lx"]), float(grp.attrs["ly"])

    def length(arr: np.ndarray, axis: int) -> float:
        vals = arr[0, :] if axis == 1 and arr.ndim == 2 else arr[:, 0] if arr.ndim == 2 else arr.reshape(-1)
        vals = np.unique(np.asarray(vals, dtype=np.float64))
        vals = vals[np.isfinite(vals)]
        if vals.size <= 1:
            return float(np.nanmax(arr) - np.nanmin(arr))
        diffs = np.diff(np.sort(vals))
        step = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
        return float(np.nanmax(vals) - np.nanmin(vals) + step)

    return length(x_grid, axis=1), length(y_grid, axis=0)


def compute_kpi_stats(vectors: np.ndarray, kpi_names: Sequence[str]) -> Dict[str, Any]:
    arr = np.asarray(vectors, dtype=np.float64)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
    return {
        "names": list(kpi_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "per_kpi": {
            str(name): {"mean": float(mean[i]), "std": float(std[i])}
            for i, name in enumerate(kpi_names)
        },
    }


@dataclass
class InverseCaseRecord:
    case_id: str
    split: str
    re: float
    num_cylinders: int
    centers: np.ndarray
    design_vec: np.ndarray
    mask: np.ndarray
    kpi_dict: Dict[str, float]
    kpi_vector: np.ndarray
    channel_order: List[str]
    field_dim: int
    domain_length_x: float
    domain_length_y: float
    cylinder_radius: float
    min_center_distance: float


class InverseDesignDataset(Dataset):
    """Case-level inverse dataset built from a packed inert HDF5 dataset."""

    def __init__(
        self,
        h5_path: Path,
        split: str,
        *,
        kpi_names: Sequence[str],
        max_num_cylinders: int,
        sort_centers: bool = True,
        normalize_kpis: bool = True,
        target_noise_std: float = 0.0,
        target_augmentation: Optional[Mapping[str, Any]] = None,
        max_cases: int = 0,
        require_inert: bool = True,
        min_center_distance: float = 1.1,
        re_scale: float = 200.0,
    ) -> None:
        self.h5_path = Path(h5_path).expanduser().resolve()
        self.split = str(split)
        self.kpi_names = list(kpi_names)
        self.max_num_cylinders = int(max_num_cylinders)
        self.sort_centers = bool(sort_centers)
        self.normalize_kpis = bool(normalize_kpis)
        self.target_noise_std = float(target_noise_std)
        self.target_augmentation = dict(target_augmentation or {})
        self.min_center_distance_default = float(min_center_distance)
        self.re_scale = float(re_scale)
        self.kpi_stats: Optional[Dict[str, Any]] = None
        self.latent_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self.records: List[InverseCaseRecord] = []
        self._load_records(max_cases=max_cases, require_inert=require_inert)
        if not self.records:
            raise RuntimeError(f"No inverse-design cases found for split='{split}' in {self.h5_path}.")
        self.domain_length_x = float(self.records[0].domain_length_x)
        self.domain_length_y = float(self.records[0].domain_length_y)
        self.case_ids = [record.case_id for record in self.records]
        self.raw_kpi_vectors = np.stack([record.kpi_vector for record in self.records], axis=0).astype(np.float32)

    def _load_records(self, *, max_cases: int, require_inert: bool) -> None:
        with h5py.File(self.h5_path, "r") as h5_file:
            cases_group = h5_file["cases"]
            for case_id in sort_case_ids(cases_group.keys()):
                grp = cases_group[case_id]
                case_split = str(grp.attrs.get("split", "all"))
                if self.split not in {"all", case_split}:
                    continue
                if "canonical_cycle" not in grp or "cylinder_centers" not in grp:
                    continue
                field_dim = int(grp.attrs.get("field_dim", grp["canonical_cycle"].shape[-1]))
                channel_order = get_case_channel_order(grp, h5_file)[:field_dim]
                if require_inert and field_dim > 4 and "temperature" in [c.lower() for c in channel_order]:
                    continue
                centers = np.asarray(grp["cylinder_centers"], dtype=np.float32).reshape(-1, 2)
                if centers.shape[0] > self.max_num_cylinders:
                    continue
                x_grid = np.asarray(grp["x_grid"], dtype=np.float32)
                y_grid = np.asarray(grp["y_grid"], dtype=np.float32)
                lx, ly = grid_domain_lengths(x_grid, y_grid, grp)
                cycle = np.asarray(grp["canonical_cycle"], dtype=np.float32)
                kpi_dict = compute_cycle_kpis(
                    cycle,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    channel_order=channel_order,
                    domain={"lx": lx, "ly": ly},
                )
                kpi_vector = kpi_vector_from_dict(kpi_dict, self.kpi_names)
                design_vec, mask = encode_design_vector(
                    centers,
                    max_num_cylinders=self.max_num_cylinders,
                    domain_length_x=lx,
                    domain_length_y=ly,
                    sort_centers=self.sort_centers,
                )
                min_dist = periodic_min_distance(centers, lx, ly)
                record = InverseCaseRecord(
                    case_id=str(case_id),
                    split=case_split,
                    re=float(grp.attrs.get("re", 100.0)),
                    num_cylinders=int(grp.attrs.get("num_cylinders", centers.shape[0])),
                    centers=centers,
                    design_vec=design_vec,
                    mask=mask,
                    kpi_dict=kpi_dict,
                    kpi_vector=kpi_vector,
                    channel_order=channel_order,
                    field_dim=field_dim,
                    domain_length_x=float(lx),
                    domain_length_y=float(ly),
                    cylinder_radius=float(grp.attrs.get("cylinder_radius", 0.5)),
                    min_center_distance=float(min_dist if math.isfinite(min_dist) else self.min_center_distance_default),
                )
                self.records.append(record)
                if max_cases and len(self.records) >= int(max_cases):
                    break

    def set_kpi_stats(self, stats: Dict[str, Any]) -> None:
        self.kpi_stats = stats

    def attach_latent_cache(self, cache: Dict[str, Dict[str, torch.Tensor]]) -> None:
        missing = [record.case_id for record in self.records if record.case_id not in cache]
        if missing:
            raise KeyError(f"Forward latent cache is missing cases: {missing[:8]}")
        self.latent_cache = cache

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[int(idx)]
        use_augmented_target = self.split == "train" and bool(self.target_augmentation.get("enabled", False))
        kpi_targets = None
        if use_augmented_target:
            kpi_targets = augment_kpi_targets_for_training(
                record.kpi_dict,
                self.kpi_names,
                self.kpi_stats,
                self.target_augmentation,
                np.random,
            )
        drop_constraints = use_augmented_target and float(np.random.random()) < float(self.target_augmentation.get("constraint_dropout_probability", 0.0))
        target_spec = build_target_spec_vector(
            None if use_augmented_target else record.kpi_dict,
            self.kpi_names,
            kpi_targets=kpi_targets,
            stats=self.kpi_stats,
            normalize=self.normalize_kpis,
            re_value=None if drop_constraints else record.re,
            num_cylinders_min=None if drop_constraints else record.num_cylinders,
            num_cylinders_max=None if drop_constraints else record.num_cylinders,
            min_center_distance=None if drop_constraints else self.min_center_distance_default,
            max_num_cylinders=self.max_num_cylinders,
            re_scale=self.re_scale,
            domain_length_scale=max(record.domain_length_x, record.domain_length_y),
        )
        target_vec = np.asarray(target_spec, dtype=np.float32).copy()
        if self.target_noise_std > 0.0:
            k = len(self.kpi_names)
            value_mask = target_vec[k : 2 * k]
            noise = np.random.normal(0.0, self.target_noise_std, size=(k,)).astype(np.float32)
            target_vec[:k] += noise * value_mask

        if self.latent_cache is None:
            raise RuntimeError("Forward latent cache has not been attached to InverseDesignDataset.")
        latent = self.latent_cache[record.case_id]
        return {
            "case_id": record.case_id,
            "target_spec_vector": torch.from_numpy(target_vec),
            "design_vec": torch.from_numpy(record.design_vec.astype(np.float32)),
            "true_count": torch.tensor(record.num_cylinders, dtype=torch.long),
            "behavior_target": latent["behavior_target"].float(),
            "organization_target": latent["organization_target"].float(),
            "re": torch.tensor(record.re, dtype=torch.float32),
            "centers": record.centers.astype(np.float32),
            "kpi_vector": torch.from_numpy(record.kpi_vector.astype(np.float32)),
        }


def collate_inverse(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "case_id": [item["case_id"] for item in batch],
        "target_spec_vector": torch.stack([item["target_spec_vector"] for item in batch], dim=0),
        "design_vec": torch.stack([item["design_vec"] for item in batch], dim=0),
        "true_count": torch.stack([item["true_count"] for item in batch], dim=0),
        "behavior_target": torch.stack([item["behavior_target"] for item in batch], dim=0),
        "organization_target": torch.stack([item["organization_target"] for item in batch], dim=0),
        "re": torch.stack([item["re"] for item in batch], dim=0),
        "kpi_vector": torch.stack([item["kpi_vector"] for item in batch], dim=0),
    }


def load_forward_model(forward_cfg: Mapping[str, Any], device: torch.device) -> Tuple[nn.Module, Dict[str, Any], Path]:
    run_dir = resolve_demo_path(forward_cfg.get("run_dir", "./Saved_Model/Case0010_20260428_084416"))
    checkpoint_name = str(forward_cfg.get("checkpoint_name", "best_model.pt"))
    ckpt_path = run_dir / checkpoint_name
    if not ckpt_path.exists() and checkpoint_name == "best_model.pt":
        ckpt_path = run_dir / "latest_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Forward checkpoint not found: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    if "model_config" in ckpt:
        model_cfg_payload = dict(ckpt["model_config"])
    elif "config" in ckpt and "model" in ckpt["config"]:
        model_cfg_payload = dict(ckpt["config"]["model"])
    else:
        config_name = str(forward_cfg.get("config_name", "resolved_train_config.json"))
        cfg_path = run_dir / config_name
        model_cfg_payload = dict(read_json(cfg_path).get("model", {})) if cfg_path.exists() else {}
    model = build_model_from_config(model_cfg_payload)
    state = ckpt.get("model_state_dict", ckpt.get("model"))
    if state is None:
        raise KeyError(f"Could not find model_state_dict in {ckpt_path}")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, model_cfg_payload, ckpt_path


def build_structure_from_centers(
    centers: np.ndarray,
    *,
    re_value: float,
    max_num_cylinders: int,
    device: torch.device,
    future_module_feature_dim: int = 0,
) -> Dict[str, torch.Tensor]:
    centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    n = min(centers_arr.shape[0], int(max_num_cylinders))
    padded = np.zeros((1, int(max_num_cylinders), 2), dtype=np.float32)
    mask = np.zeros((1, int(max_num_cylinders)), dtype=np.float32)
    if n > 0:
        padded[0, :n] = centers_arr[:n]
        mask[0, :n] = 1.0
    structure = {
        "re_values": torch.tensor([[float(re_value)]], dtype=torch.float32, device=device),
        "num_cylinders": torch.tensor([[float(n)]], dtype=torch.float32, device=device),
        "centers": torch.from_numpy(padded).to(device=device),
        "cyl_mask": torch.from_numpy(mask).to(device=device),
    }
    if int(future_module_feature_dim) > 0:
        structure["extra_module"] = torch.zeros((1, int(max_num_cylinders), int(future_module_feature_dim)), dtype=torch.float32, device=device)
    return structure


def _flatten_first(aux: Mapping[str, torch.Tensor], key: str) -> Optional[torch.Tensor]:
    value = aux.get(key)
    if value is None:
        return None
    return value.detach().float().reshape(value.shape[0], -1)[0].cpu()


def extract_forward_latent_targets(aux: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    behavior = _flatten_first(aux, "behavior_latent")
    if behavior is None:
        raise KeyError("Forward aux output is missing behavior_latent.")

    parts: List[torch.Tensor] = []
    for key in ("mean_latent", "dynamic_global_token"):
        part = _flatten_first(aux, key)
        if part is not None:
            parts.append(part)

    dyn = aux.get("dynamic_hyper_base")
    if dyn is not None and dyn.ndim >= 3:
        dyn_f = dyn.detach().float()
        parts.append(dyn_f.mean(dim=1).reshape(dyn_f.shape[0], -1)[0].cpu())
        parts.append(dyn_f.std(dim=1, unbiased=False).reshape(dyn_f.shape[0], -1)[0].cpu())

    for key in (
        "hyper_strength",
        "hyper_module_mass",
        "hyper_env_mass",
        "hyper_source_coords",
        "hyper_wake_coords",
        "hyper_wake_axis",
        "hyper_wake_extent",
        "hyper_active_mask",
        "hyper_edge_score",
    ):
        part = _flatten_first(aux, key)
        if part is not None:
            parts.append(part)

    for key in ("A_mh_effective", "A_eh_effective"):
        value = aux.get(key)
        if value is not None and value.ndim >= 3:
            tensor = value.detach().float()
            parts.append(tensor.mean(dim=1).reshape(tensor.shape[0], -1)[0].cpu())
            parts.append(tensor.amax(dim=1).reshape(tensor.shape[0], -1)[0].cpu())

    if not parts:
        raise KeyError("Forward aux output did not contain organizer fields.")
    organization = torch.cat(parts, dim=0).float()
    return behavior.float(), organization


@torch.no_grad()
def extract_case_latents(
    forward_model: nn.Module,
    model_cfg: Mapping[str, Any],
    record: InverseCaseRecord,
    *,
    device: torch.device,
    max_num_cylinders: int,
) -> Dict[str, torch.Tensor]:
    future_dim = int(model_cfg.get("future_module_feature_dim", 0))
    structure = build_structure_from_centers(
        record.centers,
        re_value=record.re,
        max_num_cylinders=max_num_cylinders,
        device=device,
        future_module_feature_dim=future_dim,
    )
    query_xy = torch.tensor([[[0.5 * record.domain_length_x, 0.5 * record.domain_length_y]]], dtype=torch.float32, device=device)
    query_tau = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
    out = forward_model(structure=structure, query_xy=query_xy, query_tau=query_tau, query_time=query_tau, return_aux=True)
    behavior, organization = extract_forward_latent_targets(out)
    return {"behavior_target": behavior.cpu(), "organization_target": organization.cpu()}


def build_forward_latent_cache(
    cache_path: Path,
    records: Sequence[InverseCaseRecord],
    *,
    forward_model: nn.Module,
    forward_model_cfg: Mapping[str, Any],
    device: torch.device,
    max_num_cylinders: int,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    needed = {record.case_id for record in records}
    if cache_path.exists() and not force_rebuild:
        cache = safe_torch_load(cache_path, map_location="cpu")
        cached_cases = set(cache.get("case_latents", {}).keys())
        if needed.issubset(cached_cases):
            print(f"[cache] using forward latent cache: {cache_path}")
            return cache
        print(f"[cache] existing cache misses {len(needed - cached_cases)} cases; rebuilding.")

    case_latents: Dict[str, Dict[str, torch.Tensor]] = {}
    iterator: Iterable[InverseCaseRecord] = records
    if tqdm is not None:
        iterator = tqdm(records, desc="forward latent cache", dynamic_ncols=True)
    for record in iterator:
        case_latents[record.case_id] = extract_case_latents(
            forward_model,
            forward_model_cfg,
            record,
            device=device,
            max_num_cylinders=max_num_cylinders,
        )
    first = next(iter(case_latents.values()))
    cache = {
        "case_latents": case_latents,
        "behavior_latent_dim": int(first["behavior_target"].numel()),
        "organization_latent_dim": int(first["organization_target"].numel()),
        "case_ids": sorted(case_latents.keys()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(cache, cache_path)
    print(f"[cache] wrote forward latent cache: {cache_path}")
    return cache


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device=device) if isinstance(value, torch.Tensor) else value
    return out


def scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def mean_rows(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.nanmean([row.get(key, float("nan")) for row in rows])) for key in keys}


def run_supervised_epoch(
    model: HypergraphInverseDesignFlow,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    loss_cfg: Mapping[str, float],
    grad_clip: float = 0.0,
    desc: str,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows: List[Dict[str, float]] = []
    iterator: Iterable[Any] = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in iterator:
        batch_device = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, metrics = model.training_loss(batch_device, loss_weights=loss_cfg)
            if training:
                loss.backward()
                if grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
        row = {
            "loss": scalar(loss),
            "loss_flow": scalar(metrics["loss_flow"]),
            "loss_count": scalar(metrics["loss_count"]),
            "loss_behavior": scalar(metrics["loss_behavior"]),
            "loss_organization": scalar(metrics["loss_organization"]),
            "loss_validity_prior": scalar(metrics["loss_validity_prior"]),
            "count_accuracy": scalar(metrics["count_accuracy"]),
        }
        rows.append(row)
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{row['loss']:.3e}", count_acc=f"{row['count_accuracy']:.2f}")
    return mean_rows(rows)


def make_eval_grid(nx: int, ny: int, lx: float, ly: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = (torch.arange(int(nx), dtype=torch.float32, device=device) + 0.5) * float(lx) / max(int(nx), 1)
    y = (torch.arange(int(ny), dtype=torch.float32, device=device) + 0.5) * float(ly) / max(int(ny), 1)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy


@torch.no_grad()
def predict_cycle_for_centers(
    forward_model: nn.Module,
    forward_model_cfg: Mapping[str, Any],
    centers: np.ndarray,
    *,
    re_value: float,
    max_num_cylinders: int,
    phase_bins: int,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    query_batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
    future_dim = int(forward_model_cfg.get("future_module_feature_dim", 0))
    structure = build_structure_from_centers(
        centers,
        re_value=re_value,
        max_num_cylinders=max_num_cylinders,
        device=device,
        future_module_feature_dim=future_dim,
    )
    x_grid, y_grid = make_eval_grid(nx, ny, lx, ly, device)
    fields: List[np.ndarray] = []
    aux_last: Dict[str, torch.Tensor] = {}
    for tau in np.linspace(0.0, 1.0, int(phase_bins), endpoint=False, dtype=np.float32):
        out = forward_model.reconstruct_full_grid(
            structure,
            x_grid,
            y_grid,
            tau=torch.tensor([float(tau)], dtype=torch.float32, device=device),
            query_time=torch.tensor([float(tau)], dtype=torch.float32, device=device),
            query_batch_size=int(query_batch_size),
        )
        fields.append(out["pred_field"][0].detach().cpu().numpy().astype(np.float32))
        aux_last = {k: v for k, v in out.items() if k not in {"pred_field", "pred_mean", "pred_residual"}}
    return np.stack(fields, axis=0), aux_last


@torch.no_grad()
def run_forward_verification(
    inverse_model: HypergraphInverseDesignFlow,
    forward_model: nn.Module,
    forward_model_cfg: Mapping[str, Any],
    dataset: InverseDesignDataset,
    *,
    device: torch.device,
    validation_cfg: Mapping[str, Any],
    query_batch_size: int,
    full_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Forward-verify sampled inverse designs and return KPI-target mismatch.

    ``val_forward_score`` is a physical validation metric, not a training loss:
    for each held-out target we sample a cylinder layout, run that layout
    through the frozen forward surrogate to reconstruct one canonical flow
    cycle, compute flow KPIs from the predicted fields, and score those KPIs
    against the requested target ranges/bounds. Lower is better; zero means the
    verified flow satisfies all scored KPI targets and hard design constraints.
    """

    num_targets = min(int(validation_cfg.get("forward_verify_num_targets", 0)), len(dataset))
    if num_targets <= 0:
        return {"val_forward_score": float("nan"), "val_self_target_forward_score": float("nan")}
    verifier_cfg = dict(full_config.get("forward_verifier", {}) if isinstance(full_config, Mapping) and isinstance(full_config.get("forward_verifier", {}), Mapping) else {})
    if isinstance(full_config, Mapping) and isinstance(full_config.get("forward_model", {}), Mapping):
        forward_cfg = full_config.get("forward_model", {})
        verifier_cfg = {**dict(forward_cfg), **verifier_cfg}
    if str(verifier_cfg.get("backend", "deterministic")).lower() == "generative" and bool(verifier_cfg.get("generative_enabled", False)):
        from evaluate_inverse import GEN_FORWARD_UNAVAILABLE, load_forward_verifier

        try:
            verifier = load_forward_verifier(full_config or {}, device)
        except Exception as exc:
            raise RuntimeError(GEN_FORWARD_UNAVAILABLE) from exc
        num_samples = max(int(validation_cfg.get("forward_verify_num_samples", 1)), 1)
        phase_bins = max(int(validation_cfg.get("forward_verify_phase_bins", 8)), 1)
        nx = max(int(validation_cfg.get("forward_verify_nx", 48)), 4)
        ny = max(int(validation_cfg.get("forward_verify_ny", 24)), 4)
        scores: List[float] = []
        uncertainties: List[float] = []
        for idx in range(num_targets):
            item = dataset[idx]
            target_vec = item["target_spec_vector"].to(device=device)
            samples = inverse_model.sample_designs(
                target_vec,
                n_samples=num_samples,
                n_steps=int(validation_cfg.get("forward_verify_ode_steps", 16)),
                seed=idx,
                device=device,
            )
            record = dataset.records[idx]
            target_spec = build_target_spec_vector(
                record.kpi_dict,
                dataset.kpi_names,
                re_value=record.re,
                num_cylinders_min=record.num_cylinders,
                num_cylinders_max=record.num_cylinders,
                min_center_distance=dataset.min_center_distance_default,
                max_num_cylinders=dataset.max_num_cylinders,
                return_spec=True,
            )
            for sample_idx, sample in enumerate(samples[: max(1, min(num_samples, 2))]):
                result = verifier.predict_cycle_for_centers(
                    sample["centers"],
                    record.re,
                    phase_bins,
                    nx,
                    ny,
                    query_batch_size,
                    seed=idx * 1000 + sample_idx,
                )
                channel_order = list(result.get("channel_order") or record.channel_order)
                sample_cycles = np.asarray(result.get("cycle_samples"), dtype=np.float32)
                sample_kpis = [
                    compute_cycle_kpis(cycle, x_grid=None, y_grid=None, channel_order=channel_order, domain={"lx": record.domain_length_x, "ly": record.domain_length_y})
                    for cycle in sample_cycles
                ]
                names = sorted({name for row in sample_kpis for name in row.keys()})
                kpis = {name: float(np.nanmean([row.get(name, float("nan")) for row in sample_kpis])) for name in names}
                kpis_std = {name: float(np.nanstd([row.get(name, float("nan")) for row in sample_kpis])) for name in names}
                kpis["num_cylinders"] = int(sample["count"])
                kpis["min_center_distance"] = float(sample["validity"].get("min_pair_distance", 0.0))
                selected = [name for name in target_spec.get("kpi_targets", {}).keys() if name in kpis_std] or list(kpis_std.keys())
                uncertainty = float(np.nanmean([kpis_std[name] for name in selected])) if selected else 0.0
                base_score = score_candidate_kpis(kpis, target_spec)
                total_score = float(base_score["total_score"]) + float(verifier_cfg.get("uncertainty_penalty_weight", 0.05)) * uncertainty
                # Same physical score as the deterministic path, with an
                # optional penalty for KPI uncertainty across stochastic
                # forward-verifier samples.
                scores.append(total_score)
                uncertainties.append(uncertainty)
        result = {
            "val_forward_score": float(np.mean(scores)) if scores else float("nan"),
            "val_gen_forward_score_mean": float(np.mean(scores)) if scores else float("nan"),
            "val_gen_forward_score_std": float(np.std(scores)) if scores else float("nan"),
            "val_gen_kpi_uncertainty": float(np.mean(uncertainties)) if uncertainties else float("nan"),
        }
        result["val_self_target_forward_score"] = result["val_forward_score"]
        return result
    num_samples = max(int(validation_cfg.get("forward_verify_num_samples", 1)), 1)
    phase_bins = max(int(validation_cfg.get("forward_verify_phase_bins", 8)), 1)
    nx = max(int(validation_cfg.get("forward_verify_nx", 48)), 4)
    ny = max(int(validation_cfg.get("forward_verify_ny", 24)), 4)
    scores: List[float] = []
    for idx in range(num_targets):
        item = dataset[idx]
        target_vec = item["target_spec_vector"].to(device=device)
        samples = inverse_model.sample_designs(
            target_vec,
            n_samples=num_samples,
            n_steps=int(validation_cfg.get("forward_verify_ode_steps", 16)),
            seed=idx,
            device=device,
        )
        record = dataset.records[idx]
        for sample in samples[: max(1, min(num_samples, 2))]:
            cycle, _ = predict_cycle_for_centers(
                forward_model,
                forward_model_cfg,
                sample["centers"],
                re_value=record.re,
                max_num_cylinders=dataset.max_num_cylinders,
                phase_bins=phase_bins,
                nx=nx,
                ny=ny,
                lx=record.domain_length_x,
                ly=record.domain_length_y,
                query_batch_size=query_batch_size,
                device=device,
            )
            kpis = compute_cycle_kpis(cycle, x_grid=None, y_grid=None, channel_order=record.channel_order, domain={"lx": record.domain_length_x, "ly": record.domain_length_y})
            kpis["num_cylinders"] = int(sample["count"])
            kpis["min_center_distance"] = float(sample["validity"].get("min_pair_distance", 0.0))
            target_spec = build_target_spec_vector(
                record.kpi_dict,
                dataset.kpi_names,
                re_value=record.re,
                num_cylinders_min=record.num_cylinders,
                num_cylinders_max=record.num_cylinders,
                min_center_distance=dataset.min_center_distance_default,
                max_num_cylinders=dataset.max_num_cylinders,
                return_spec=True,
            )
            score = score_candidate_kpis(kpis, target_spec)
            # Physical meaning: this number measures how far the forward
            # model's predicted wake for the sampled cylinder layout is from
            # the target KPI envelope, after normalizing by each KPI scale.
            scores.append(float(score["total_score"]))
    result = {"val_forward_score": float(np.mean(scores)) if scores else float("nan")}
    result["val_self_target_forward_score"] = result["val_forward_score"]
    return result


@torch.no_grad()
def run_demo_target_forward_verification(
    inverse_model: HypergraphInverseDesignFlow,
    forward_model: nn.Module,
    forward_model_cfg: Mapping[str, Any],
    *,
    target_json_path: Path,
    kpi_names: Sequence[str],
    kpi_stats: Optional[Mapping[str, Any]],
    normalize_kpis: bool,
    max_num_cylinders: int,
    re_scale: float,
    lx: float,
    ly: float,
    device: torch.device,
    validation_cfg: Mapping[str, Any],
    query_batch_size: int,
) -> Dict[str, float]:
    if not target_json_path.exists():
        return {
            "val_demo_target_forward_score": float("nan"),
            "val_demo_target_best_score": float("nan"),
            "val_demo_target_valid_fraction": float("nan"),
            "val_demo_target_mean_count": float("nan"),
            "val_demo_target_min_distance_mean": float("nan"),
        }
    payload = read_json(target_json_path)
    preferences = payload.get("preferences", {}) if isinstance(payload.get("preferences", {}), Mapping) else {}
    min_center_distance = payload.get("min_center_distance", preferences.get("min_center_distance", 1.1))
    target_spec = build_target_spec_vector(
        kpi_names=kpi_names,
        kpi_targets=payload.get("kpis", {}),
        stats=kpi_stats,
        normalize=normalize_kpis,
        re_value=payload.get("Re", payload.get("re", 100.0)),
        num_cylinders_min=payload.get("num_cylinders_min"),
        num_cylinders_max=payload.get("num_cylinders_max"),
        min_center_distance=min_center_distance,
        max_num_cylinders=max_num_cylinders,
        re_scale=re_scale,
        domain_length_scale=max(lx, ly),
        return_spec=True,
    )
    target_spec["preferences"] = dict(preferences)
    if "min_x_span" in preferences:
        target_spec["constraints"]["min_x_span"] = float(preferences["min_x_span"])
    if "min_y_span" in preferences:
        target_spec["constraints"]["min_y_span"] = float(preferences["min_y_span"])
    num_samples = max(int(validation_cfg.get("demo_forward_verify_num_samples", 16)), 1)
    top_k = min(max(int(validation_cfg.get("demo_forward_verify_top_k", 4)), 1), num_samples)
    phase_bins = max(int(validation_cfg.get("forward_verify_phase_bins", 8)), 1)
    nx = max(int(validation_cfg.get("forward_verify_nx", 48)), 4)
    ny = max(int(validation_cfg.get("forward_verify_ny", 24)), 4)
    target_vec = torch.from_numpy(np.asarray(target_spec["vector"], dtype=np.float32)).to(device=device)
    samples = inverse_model.sample_designs(
        target_vec,
        n_samples=num_samples,
        n_steps=int(validation_cfg.get("forward_verify_ode_steps", 16)),
        seed=int(validation_cfg.get("demo_forward_verify_seed", 1729)),
        min_center_distance=float(min_center_distance or 1.1),
        device=device,
    )
    valid_flags = [bool(sample.get("validity", {}).get("valid", False)) for sample in samples]
    counts = [float(sample.get("count", 0)) for sample in samples]
    min_dists = [float(sample.get("validity", {}).get("min_pair_distance", 0.0)) for sample in samples]
    ranked_samples = sorted(
        samples,
        key=lambda sample: (
            0 if bool(sample.get("validity", {}).get("valid", False)) else 1,
            max(0.0, float(preferences.get("min_x_span", 0.0)) - (float(np.ptp(np.asarray(sample["centers"], dtype=np.float32).reshape(-1, 2)[:, 0])) if np.asarray(sample["centers"]).size else 0.0)),
            max(0.0, float(preferences.get("min_y_span", 0.0)) - (float(np.ptp(np.asarray(sample["centers"], dtype=np.float32).reshape(-1, 2)[:, 1])) if np.asarray(sample["centers"]).size else 0.0)),
            -float(sample.get("validity", {}).get("min_pair_distance", 0.0)),
            abs(float(sample.get("count", 0)) - 0.5 * (float(payload.get("num_cylinders_min", 0) or 0) + float(payload.get("num_cylinders_max", max_num_cylinders) or max_num_cylinders))),
        ),
    )
    scores: List[float] = []
    re_value = float(payload.get("Re", payload.get("re", 100.0)))
    for sample in ranked_samples[:top_k]:
        cycle, _ = predict_cycle_for_centers(
            forward_model,
            forward_model_cfg,
            sample["centers"],
            re_value=re_value,
            max_num_cylinders=max_num_cylinders,
            phase_bins=phase_bins,
            nx=nx,
            ny=ny,
            lx=lx,
            ly=ly,
            query_batch_size=query_batch_size,
            device=device,
        )
        kpis = compute_cycle_kpis(cycle, x_grid=None, y_grid=None, channel_order=INERT_CHANNEL_ORDER, domain={"lx": lx, "ly": ly})
        kpis["num_cylinders"] = int(sample["count"])
        kpis["min_center_distance"] = float(sample.get("validity", {}).get("min_pair_distance", 0.0))
        kpis["valid"] = bool(sample.get("validity", {}).get("valid", True))
        centers = np.asarray(sample["centers"], dtype=np.float32).reshape(-1, 2)
        if centers.shape[0] > 1:
            kpis["x_span"] = float(np.max(centers[:, 0]) - np.min(centers[:, 0]))
            kpis["y_span"] = float(np.max(centers[:, 1]) - np.min(centers[:, 1]))
        scores.append(float(score_candidate_kpis(kpis, target_spec)["total_score"]))
    return {
        "val_demo_target_forward_score": float(np.mean(scores)) if scores else float("nan"),
        "val_demo_target_best_score": float(np.min(scores)) if scores else float("nan"),
        "val_demo_target_valid_fraction": float(np.mean(valid_flags)) if valid_flags else float("nan"),
        "val_demo_target_mean_count": float(np.mean(counts)) if counts else float("nan"),
        "val_demo_target_min_distance_mean": float(np.mean(min_dists)) if min_dists else float("nan"),
    }


def save_history_csv(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not history:
        return
    keys = sorted({key for row in history for key in row.keys()})
    if "epoch" in keys:
        keys.remove("epoch")
        keys = ["epoch"] + keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def finite_history_points(
    history: Sequence[Mapping[str, Any]],
    key: str,
    *,
    require_positive: bool = False,
) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []
    for row in history:
        try:
            epoch = int(row["epoch"])
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if require_positive and value <= 0.0:
            continue
        points.append((epoch, value))
    return points


def save_loss_curve(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not history:
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    has_series = False
    for key, label in (("train_loss", "train"), ("val_loss", "validation")):
        points = finite_history_points(history, key, require_positive=True)
        if points:
            epochs, vals = zip(*points)
            ax.plot(epochs, vals, marker="o", markersize=3, linewidth=1.5, label=label)
            has_series = True
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if has_series:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_forward_score_curve(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not history:
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    has_series = False
    all_positive = True
    for key, label in (
        ("val_forward_score", "forward score"),
        ("val_gen_forward_score_mean", "generative mean"),
    ):
        points = finite_history_points(history, key)
        if points:
            epochs, vals = zip(*points)
            ax.plot(epochs, vals, marker="o", markersize=4, linewidth=1.5, label=label)
            all_positive = all_positive and all(v > 0.0 for v in vals)
            has_series = True
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Forward verification score")
    if has_series and all_positive:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if has_series:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def unique_records(*datasets: InverseDesignDataset) -> List[InverseCaseRecord]:
    out: Dict[str, InverseCaseRecord] = {}
    for dataset in datasets:
        for record in dataset.records:
            out[record.case_id] = record
    return [out[key] for key in sort_case_ids(out.keys())]


def main() -> None:
    args = parse_args()
    cfg_path = resolve_config_path(args.config)
    cfg = read_json(cfg_path)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.no_forward_verify:
        cfg.setdefault("validation", {})["forward_verify_every_epochs"] = 0
        cfg.setdefault("validation", {})["demo_forward_verify_every_epochs"] = 0
    if args.train_max_cases is not None:
        cfg.setdefault("dataset", {})["train_max_cases"] = int(args.train_max_cases)
    if args.val_max_cases is not None:
        cfg.setdefault("dataset", {})["val_max_cases"] = int(args.val_max_cases)
    if args.num_workers is not None:
        cfg.setdefault("training", {})["num_workers"] = int(args.num_workers)

    set_seed(int(cfg.get("seed", cfg.get("training", {}).get("seed", 42))))
    device = select_device(args.device)
    dataset_cfg = cfg["dataset"]
    kpi_cfg = cfg["target_kpis"]
    kpi_names = list(kpi_cfg.get("names", DEFAULT_KPI_NAMES))
    max_num_cylinders = int(dataset_cfg.get("max_num_cylinders", 8))
    re_scale = float(cfg.get("inverse_model", {}).get("re_scale", 200.0))
    min_center_distance = float(cfg.get("inverse_model", {}).get("min_center_distance", 1.1))

    packed_path = resolve_demo_path(dataset_cfg["packed_h5_path"])
    if not packed_path.exists():
        raise FileNotFoundError(f"Packed inert dataset not found: {packed_path}")

    train_set = InverseDesignDataset(
        packed_path,
        split=str(dataset_cfg.get("train_split", "train")),
        kpi_names=kpi_names,
        max_num_cylinders=max_num_cylinders,
        sort_centers=bool(dataset_cfg.get("sort_centers", True)),
        normalize_kpis=bool(kpi_cfg.get("normalize", True)),
        target_noise_std=float(kpi_cfg.get("target_noise_std", 0.0)),
        target_augmentation=cfg.get("target_augmentation", {}),
        max_cases=int(dataset_cfg.get("train_max_cases", 0)),
        require_inert=bool(dataset_cfg.get("require_inert", True)),
        min_center_distance=min_center_distance,
        re_scale=re_scale,
    )
    val_set = InverseDesignDataset(
        packed_path,
        split=str(dataset_cfg.get("val_split", "test")),
        kpi_names=kpi_names,
        max_num_cylinders=max_num_cylinders,
        sort_centers=bool(dataset_cfg.get("sort_centers", True)),
        normalize_kpis=bool(kpi_cfg.get("normalize", True)),
        target_noise_std=0.0,
        max_cases=int(dataset_cfg.get("val_max_cases", 0)),
        require_inert=bool(dataset_cfg.get("require_inert", True)),
        min_center_distance=min_center_distance,
        re_scale=re_scale,
    )
    kpi_stats = compute_kpi_stats(train_set.raw_kpi_vectors, kpi_names)
    train_set.set_kpi_stats(kpi_stats)
    val_set.set_kpi_stats(kpi_stats)

    save_root = ensure_dir(resolve_demo_path(cfg["paths"].get("saved_model_dir", "./Saved_Model_Inverse")))
    case_id = str(cfg.get("case_id", "inv001"))
    case_suffix = case_id[3:] if case_id.lower().startswith("inv") else case_id
    run_dir = ensure_dir(save_root / f"CaseInv{case_suffix}_{current_timestamp()}")
    config_train_dir = ensure_dir(resolve_demo_path(cfg["paths"].get("config_train_dir", "./Config_Train")))
    backup_dir = ensure_dir(
        resolve_demo_path(cfg["paths"].get("config_backup_dir", str(config_train_dir / "Configs_inverse_bk")))
    )

    forward_model, forward_model_cfg, forward_ckpt_path = load_forward_model(cfg["forward_model"], device)
    cache_path = run_dir / "inverse_forward_latent_cache.pt"
    cache = build_forward_latent_cache(
        cache_path,
        unique_records(train_set, val_set),
        forward_model=forward_model,
        forward_model_cfg=forward_model_cfg,
        device=device,
        max_num_cylinders=max_num_cylinders,
        force_rebuild=bool(cfg["forward_model"].get("rebuild_latent_cache", False)),
    )
    train_set.attach_latent_cache(cache["case_latents"])
    val_set.attach_latent_cache(cache["case_latents"])

    target_dim = len(kpi_names) * 7 + 5
    inv_cfg = dict(cfg["inverse_model"])
    inv_cfg["target_dim"] = target_dim
    inv_cfg["design_dim"] = max_num_cylinders * 3
    inv_cfg["max_num_cylinders"] = max_num_cylinders
    inv_cfg["behavior_latent_dim"] = int(cache["behavior_latent_dim"])
    inv_cfg["organization_latent_dim"] = int(cache["organization_latent_dim"])
    inv_cfg["domain_length_x"] = float(train_set.domain_length_x)
    inv_cfg["domain_length_y"] = float(train_set.domain_length_y)
    inv_cfg["re_scale"] = re_scale
    cfg["inverse_model"] = inv_cfg
    cfg["target_kpis"]["stats"] = kpi_stats
    cfg["forward_model"]["checkpoint_path"] = str(forward_ckpt_path)
    write_json(run_dir / "resolved_train_inverse_config.json", cfg)
    write_json(backup_dir / f"Config_Inverse_CaseInv{case_suffix}_{current_timestamp()}.json", cfg)

    model = HypergraphInverseDesignFlow(InverseModelConfig.from_dict(inv_cfg)).to(device)
    training_cfg = cfg["training"]
    train_loader = DataLoader(
        train_set,
        batch_size=int(training_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collate_inverse,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(training_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collate_inverse,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-5)),
    )
    epochs = int(training_cfg.get("epochs", 500))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(training_cfg.get("scheduler_min_lr", 1.0e-6)),
    )
    loss_cfg = cfg.get("loss", {})
    validation_cfg = cfg.get("validation", {})
    forward_verify_every = int(validation_cfg.get("forward_verify_every_epochs", 0))
    demo_verify_every = int(validation_cfg.get("demo_forward_verify_every_epochs", 0))
    query_batch_size = int(cfg["forward_model"].get("query_batch_size", 32768))
    grad_clip = float(training_cfg.get("gradient_clip_norm", 1.0))

    best_val = float("inf")
    history: List[Dict[str, Any]] = []
    latest_path = run_dir / "latest_model.pt"
    best_path = run_dir / "best_model.pt"
    print(f"[setup] inverse run dir: {run_dir}")
    print(f"[setup] train cases={len(train_set)} val cases={len(val_set)} target_dim={target_dim}")
    print(f"[setup] behavior_dim={cache['behavior_latent_dim']} organization_dim={cache['organization_latent_dim']}")

    for epoch in range(1, epochs + 1):
        train_metrics = run_supervised_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            loss_cfg=loss_cfg,
            grad_clip=grad_clip,
            desc=f"train {epoch}/{epochs}",
        )
        val_metrics = run_supervised_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            loss_cfg=loss_cfg,
            grad_clip=0.0,
            desc=f"val {epoch}/{epochs}",
        )
        scheduler.step()

        verify_metrics = {"val_forward_score": float("nan"), "val_self_target_forward_score": float("nan")}
        if forward_verify_every > 0 and epoch % forward_verify_every == 0:
            verify_metrics = run_forward_verification(
                model,
                forward_model,
                forward_model_cfg,
                val_set,
                device=device,
                validation_cfg=validation_cfg,
                query_batch_size=query_batch_size,
                full_config=cfg,
            )
        demo_target_json = str(validation_cfg.get("demo_target_json", "")).strip()
        if demo_verify_every > 0 and demo_target_json and epoch % demo_verify_every == 0:
            verify_metrics.update(
                run_demo_target_forward_verification(
                    model,
                    forward_model,
                    forward_model_cfg,
                    target_json_path=resolve_demo_path(demo_target_json),
                    kpi_names=kpi_names,
                    kpi_stats=kpi_stats,
                    normalize_kpis=bool(kpi_cfg.get("normalize", True)),
                    max_num_cylinders=max_num_cylinders,
                    re_scale=re_scale,
                    lx=float(train_set.domain_length_x),
                    ly=float(train_set.domain_length_y),
                    device=device,
                    validation_cfg=validation_cfg,
                    query_batch_size=query_batch_size,
                )
            )

        row: Dict[str, Any] = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row.update(verify_metrics)
        row["train_loss"] = train_metrics.get("loss", float("nan"))
        row["val_loss"] = val_metrics.get("loss", float("nan"))
        history.append(row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "inverse_model_config": model.cfg.to_dict(),
            "config": cfg,
            "kpi_stats": kpi_stats,
            "kpi_names": kpi_names,
            "forward_checkpoint_path": str(forward_ckpt_path),
            "best_val_loss": best_val,
        }
        torch.save(checkpoint, latest_path)
        val_loss = float(row["val_loss"])
        if math.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            checkpoint["best_val_loss"] = best_val
            torch.save(checkpoint, best_path)
            best_note = f"\nnew-best"
        else:
            best_note = ""
        save_history_csv(history, run_dir / "loss_history.csv")
        write_json(run_dir / "loss_history.json", {"history": history})
        save_loss_curve(history, run_dir / "loss_curve.png")
        save_forward_score_curve(history, run_dir / "forward_score_curve.png")
        print(
            f"[epoch {epoch:04d}] train={row['train_loss']:.4e} val={row['val_loss']:.4e} "
            f"count_acc={row.get('val_count_accuracy', float('nan')):.3f} "
            f"forward_score={row.get('val_forward_score', float('nan')):.4e}{best_note}"
        )

    print(f"[done] saved latest checkpoint to {latest_path}")
    if best_path.exists():
        print(f"[done] saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
