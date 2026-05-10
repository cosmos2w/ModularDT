from __future__ import annotations

"""Train the steady ChannelThermal inverse-design generator."""

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-inverse")

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from channelthermal_datasets import CHANNEL_ORDER, H5Normalizer
from channelthermal_model_utils import (
    autocast_context,
    count_parameters,
    current_timestamp,
    ensure_dir,
    load_trusted_checkpoint,
    make_grad_scaler,
    read_json,
    recursive_to_device,
    resolve_demo_path,
    select_device,
    set_seed,
    strip_module_prefix,
    write_json,
)
from model import GlobalChannelThermalModel, GlobalChannelThermalModelConfig, load_local_surrogate_from_checkpoint
from train import local_surrogate_normalization_config, require_local_normalization_stats, warn_if_synthetic_only_local_surrogate
from model_inverse import InverseModelConfig, ThermalInverseDesignFlow, encode_design_vector
from thermal_inverse_kpi import (
    DEFAULT_KPI_NAMES,
    build_target_spec_vector,
    augment_kpi_targets_for_training,
    compute_steady_thermal_kpis,
    kpi_vector_from_dict,
    score_candidate_kpis,
)


DEFAULT_CONFIG_PATH = "./Configs/train_inverse_config_template.json"
CHECKPOINT_PREFERENCE = ("best_predicted_model.pt", "best_model.pt", "latest_model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Demo 1 ChannelThermal inverse-design model.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="JSON config path.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--max-train-cases", type=int, default=None, help="Limit train cases for smoke tests.")
    parser.add_argument("--max-val-cases", type=int, default=None, help="Limit val cases for smoke tests.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches per epoch.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit val batches per epoch.")
    parser.add_argument("--dry-run", action="store_true", help="Load datasets/model, build one batch, then exit.")
    parser.add_argument("--Run_ID", dest="run_id", type=str, default=None, help="Numeric run serial, e.g. 0001.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional descriptive run suffix.")
    return parser.parse_args()


def resolve_config_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return resolve_demo_path(path)


def _decode_string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def decode_string_array(values: Any) -> List[str]:
    return [_decode_string(item) for item in np.asarray(values).reshape(-1)]


def _read_case_config(group: h5py.Group) -> Dict[str, Any]:
    if "case_config_json" not in group:
        return {}
    raw = group["case_config_json"][()]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _domain_from_group(group: h5py.Group) -> Dict[str, float]:
    cfg = _read_case_config(group)
    domain = cfg.get("domain", {}) if isinstance(cfg, dict) else {}
    lx = float(domain.get("lx", np.max(group["x_grid"][...]) if "x_grid" in group else 12.0))
    ly = float(domain.get("ly", np.max(group["y_grid"][...]) if "y_grid" in group else 4.0))
    return {"domain_length_x": lx, "domain_length_y": ly, "lx": lx, "ly": ly}


def _material_params(group: h5py.Group) -> np.ndarray:
    materials = group.get("material_parameters", None)
    if materials is None:
        return np.zeros((6,), dtype=np.float32)
    return np.asarray(
        [
            float(materials.attrs.get("nu", 0.0)),
            float(materials.attrs.get("solid_alpha", 0.0)),
            float(materials.attrs.get("fluid_alpha", 0.0)),
            float(materials.attrs.get("solid_k", 0.0)),
            float(materials.attrs.get("fluid_k", 0.0)),
            float(materials.attrs.get("module_radius", 0.45)),
        ],
        dtype=np.float32,
    )


def _re_uin(group: h5py.Group) -> Tuple[float, float]:
    materials = group.get("material_parameters", None)
    if materials is None:
        return 0.0, 0.0
    return float(materials.attrs.get("re", 0.0)), float(materials.attrs.get("u_in", 0.0))


def _local_disk_query_points(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    size_y, size_x = mask.shape
    xi = np.linspace(-1.0, 1.0, size_x, dtype=np.float32)
    eta = np.linspace(-1.0, 1.0, size_y, dtype=np.float32)
    xx, yy = np.meshgrid(xi, eta)
    return np.stack([xx[mask], yy[mask]], axis=-1).astype(np.float32)


def normalize_run_id(value: Any, fallback: str = "0001") -> str:
    raw = str(value or fallback).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be a numeric serial such as '0001'; got {raw!r}.")
    return f"{int(raw):04d}"


def sanitize_run_suffix(value: Any) -> str:
    raw = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw).strip("_")


@dataclass
class ThermalInverseCaseRecord:
    case_id: str
    split: str
    design_vec: np.ndarray
    true_count: int
    module_centers: np.ndarray
    module_present: np.ndarray
    heat_powers: np.ndarray
    material_params: np.ndarray
    re: float
    u_in: float
    domain_length_x: float
    domain_length_y: float
    module_radius: float
    x_grid: np.ndarray
    y_grid: np.ndarray
    steady_field: np.ndarray
    module_mask: Optional[np.ndarray]
    module_internal_temperature: Optional[np.ndarray]
    module_internal_mask: Optional[np.ndarray]
    module_internal_query_points: np.ndarray
    interface_condition: Optional[np.ndarray]
    interface_target: Optional[np.ndarray]
    kpi_dict: Dict[str, Any]
    kpi_vector: np.ndarray


class ThermalInverseDesignDataset(Dataset):
    def __init__(
        self,
        packed_h5_path: str | Path,
        *,
        split: str = "train",
        kpi_names: Sequence[str] = DEFAULT_KPI_NAMES,
        kpi_stats: Optional[Mapping[str, Any]] = None,
        normalize_targets: bool = True,
        target_augmentation: Optional[Mapping[str, Any]] = None,
        max_num_modules: Optional[int] = None,
        generate_heat_power: bool = False,
        heat_power_scale: float = 1.0,
        sort_centers: bool = True,
        max_cases: int = 0,
        use_all_if_split_missing: bool = True,
        seed: int = 42,
        behavior_latent_dim: int = 96,
        organization_latent_dim: int = 256,
    ) -> None:
        self.path = resolve_demo_path(packed_h5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Packed ChannelThermal dataset not found: {self.path}")
        self.split = str(split)
        self.kpi_names = tuple(kpi_names)
        self.kpi_stats = kpi_stats
        self.normalize_targets = bool(normalize_targets)
        self.target_augmentation = dict(target_augmentation or {})
        self.max_num_modules_config = max_num_modules
        self.generate_heat_power = bool(generate_heat_power)
        self.heat_power_scale = float(heat_power_scale)
        self.sort_centers = bool(sort_centers)
        self.seed = int(seed)
        self.behavior_latent_dim = int(behavior_latent_dim)
        self.organization_latent_dim = int(organization_latent_dim)
        self.latent_targets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.normalizer: Optional[H5Normalizer] = None
        self.records = self._load_records(max_cases=max_cases, use_all_if_split_missing=use_all_if_split_missing)
        if not self.records:
            raise RuntimeError(f"No inverse records found in split {self.split!r} from {self.path}.")

    def _load_records(self, *, max_cases: int, use_all_if_split_missing: bool) -> List[ThermalInverseCaseRecord]:
        records: List[ThermalInverseCaseRecord] = []
        with h5py.File(self.path, "r") as h5:
            self.normalizer = H5Normalizer.from_h5(h5)
            channel_order = decode_string_array(h5.get("channel_order", np.asarray(CHANNEL_ORDER, dtype="S"))[...])
            if "case_ids" in h5 and "splits" in h5:
                case_ids = decode_string_array(h5["case_ids"][...])
                splits = decode_string_array(h5["splits"][...])
            else:
                case_ids = sorted(h5["cases"].keys())
                splits = [_decode_string(h5["cases"][case_id].attrs.get("split", "all")) for case_id in case_ids]
            selected = [idx for idx, item in enumerate(splits) if str(item).lower() == self.split.lower()]
            if not selected and use_all_if_split_missing:
                selected = list(range(len(case_ids)))
            if max_cases and int(max_cases) > 0:
                selected = selected[: int(max_cases)]
            root_max = int(h5.attrs.get("max_modules", 0))
            self.max_num_modules = int(self.max_num_modules_config or root_max or 1)
            self.field_dim = int(h5.attrs.get("field_dim", len(channel_order)))
            self.n_interface_points = int(h5.attrs.get("n_interface_points", 64))
            for idx in selected:
                case_id = str(case_ids[idx])
                group = h5["cases"][case_id]
                domain = _domain_from_group(group)
                material = _material_params(group)
                radius = float(material[5]) if material.shape[0] > 5 and float(material[5]) > 0.0 else 0.45
                re, u_in = _re_uin(group)
                centers = group["module_centers"][...].astype(np.float32)
                present = group["module_present"][...].astype(np.float32)
                heat = group["heat_powers"][...].astype(np.float32) if "heat_powers" in group else np.zeros((centers.shape[0],), dtype=np.float32)
                steady = group["steady_field"][...].astype(np.float32)
                x_grid = group["x_grid"][...].astype(np.float32)
                y_grid = group["y_grid"][...].astype(np.float32)
                internal = group["module_internal_temperature"][...].astype(np.float32) if "module_internal_temperature" in group else None
                internal_mask = group["module_internal_mask"][...].astype(np.uint8) if "module_internal_mask" in group else None
                local_query = _local_disk_query_points(internal_mask) if internal_mask is not None else np.zeros((0, 2), dtype=np.float32)
                interface_condition = group["interface_condition"][...].astype(np.float32) if "interface_condition" in group else None
                interface_target = group["interface_target"][...].astype(np.float32) if "interface_target" in group else None
                module_mask = group["module_mask"][...].astype(np.uint8) if "module_mask" in group else None
                design_vec, _ = encode_design_vector(
                    centers,
                    present,
                    heat,
                    max_num_modules=self.max_num_modules,
                    domain_length_x=float(domain["domain_length_x"]),
                    domain_length_y=float(domain["domain_length_y"]),
                    generate_heat_power=self.generate_heat_power,
                    heat_power_scale=self.heat_power_scale,
                    sort_centers=self.sort_centers,
                )
                kpis = compute_steady_thermal_kpis(
                    steady,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    channel_order=channel_order,
                    module_mask=module_mask,
                    module_centers=centers,
                    module_present=present,
                    heat_powers=heat,
                    module_internal_temperature=internal,
                    module_internal_mask=internal_mask,
                    interface_target=interface_target,
                    interface_condition=interface_condition,
                    domain={**domain, "module_radius": radius},
                    material_params=material,
                )
                active_count = int(np.sum(present > 0.5))
                kpis["num_modules"] = active_count
                kpis["heat_power_total"] = float(np.sum(heat[present > 0.5])) if heat.size else 0.0
                records.append(
                    ThermalInverseCaseRecord(
                        case_id=case_id,
                        split=str(splits[idx]),
                        design_vec=design_vec,
                        true_count=active_count,
                        module_centers=centers,
                        module_present=present,
                        heat_powers=heat,
                        material_params=material,
                        re=re,
                        u_in=u_in,
                        domain_length_x=float(domain["domain_length_x"]),
                        domain_length_y=float(domain["domain_length_y"]),
                        module_radius=radius,
                        x_grid=x_grid,
                        y_grid=y_grid,
                        steady_field=steady,
                        module_mask=module_mask,
                        module_internal_temperature=internal,
                        module_internal_mask=internal_mask,
                        module_internal_query_points=local_query,
                        interface_condition=interface_condition,
                        interface_target=interface_target,
                        kpi_dict=kpis,
                        kpi_vector=kpi_vector_from_dict(kpis, self.kpi_names),
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def set_kpi_stats(self, stats: Optional[Mapping[str, Any]]) -> None:
        self.kpi_stats = stats

    def set_latent_targets(self, latents: Mapping[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        self.latent_targets = {str(k): (np.asarray(v[0], dtype=np.float32), np.asarray(v[1], dtype=np.float32)) for k, v in latents.items()}

    def __getitem__(self, item: int) -> Dict[str, Any]:
        idx = int(item)
        record = self.records[idx]
        rng = np.random.default_rng(self.seed + idx * 104729 + int(torch.randint(0, 2**16, (1,)).item()))
        if self.target_augmentation.get("enabled", False):
            aug = augment_kpi_targets_for_training(
                record.kpi_dict,
                self.kpi_names,
                self.kpi_stats,
                self.target_augmentation,
                rng,
                return_metadata=True,
            )
            kpi_targets = aug["kpi_targets"]
            active_fraction = float(aug["target_active_fraction"])
        else:
            kpi_targets = {
                name: {"mode": "exact", "value": float(record.kpi_dict[name]), "weight": 1.0}
                for name in self.kpi_names
                if name in record.kpi_dict and name not in set(record.kpi_dict.get("unavailable_kpis", []))
            }
            active_fraction = 1.0
        dropout_p = float(self.target_augmentation.get("constraint_dropout_probability", 0.0)) if self.target_augmentation else 0.0
        use_constraints = float(rng.random()) >= dropout_p
        target_spec = build_target_spec_vector(
            kpi_targets=kpi_targets,
            kpi_names=self.kpi_names,
            stats=self.kpi_stats,
            normalize=self.normalize_targets,
            num_modules_min=record.true_count if use_constraints else None,
            num_modules_max=record.true_count if use_constraints else None,
            min_center_distance=float(self.target_augmentation.get("min_center_distance", 0.0)) if self.target_augmentation.get("min_center_distance") else None,
            max_num_modules=self.max_num_modules,
            domain_length_scale=max(record.domain_length_x, record.domain_length_y),
            heat_power_total=float(np.sum(record.heat_powers[record.module_present > 0.5])) if self.generate_heat_power else None,
            heat_power_scale=self.heat_power_scale,
        )
        behavior, organization = self.latent_targets.get(
            record.case_id,
            (
                np.zeros((self.behavior_latent_dim,), dtype=np.float32),
                np.zeros((self.organization_latent_dim,), dtype=np.float32),
            ),
        )
        return {
            "case_id": record.case_id,
            "design_vec": record.design_vec.astype(np.float32),
            "target_spec_vector": np.asarray(target_spec, dtype=np.float32),
            "target_kpi_targets": kpi_targets,
            "kpi_vector": record.kpi_vector.astype(np.float32),
            "true_count": np.asarray([record.true_count], dtype=np.int64),
            "target_active_fraction": np.asarray([active_fraction], dtype=np.float32),
            "behavior_target": behavior.astype(np.float32),
            "organization_target": organization.astype(np.float32),
            "record_index": np.asarray([idx], dtype=np.int64),
        }


def collate_inverse(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"case_id": [item["case_id"] for item in batch]}
    for key in ("design_vec", "target_spec_vector", "kpi_vector", "true_count", "target_active_fraction", "behavior_target", "organization_target", "record_index"):
        out[key] = torch.as_tensor(np.stack([item[key] for item in batch]), dtype=torch.float32 if key != "true_count" and key != "record_index" else torch.long)
    out["true_count"] = out["true_count"].reshape(-1)
    out["target_active_fraction"] = out["target_active_fraction"].reshape(-1)
    out["record_index"] = out["record_index"].reshape(-1)
    return out


def compute_kpi_stats(vectors: np.ndarray, kpi_names: Sequence[str]) -> Dict[str, Any]:
    arr = np.asarray(vectors, dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std = np.where(np.abs(std) < 1.0e-8, 1.0, std).astype(np.float32)
    return {
        "names": list(kpi_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        **{
            str(name): {"mean": float(mean[i]), "std": float(std[i])}
            for i, name in enumerate(kpi_names)
        },
    }


def latest_run_dir(saved_root: Path) -> Path:
    matches = sorted([path for path in saved_root.glob("Run_*") if path.is_dir()])
    if not matches:
        raise FileNotFoundError(f"No Run_* directories found under {saved_root}.")
    return matches[-1]


def resolve_forward_checkpoint(forward_cfg: Mapping[str, Any]) -> Tuple[Path, Path]:
    saved_root = resolve_demo_path(forward_cfg.get("saved_root", "./Saved_Model"))
    run_dir_raw = str(forward_cfg.get("run_dir", "auto"))
    run_dir = latest_run_dir(saved_root) if run_dir_raw.lower() == "auto" else resolve_demo_path(run_dir_raw)
    checkpoint_name = str(forward_cfg.get("checkpoint_name", "auto"))
    names = list(forward_cfg.get("checkpoint_preference", CHECKPOINT_PREFERENCE))
    if checkpoint_name.lower() != "auto":
        names = [checkpoint_name]
    for name in names:
        candidate = run_dir / str(name)
        if candidate.exists():
            return candidate.resolve(), run_dir.resolve()
    raise FileNotFoundError(f"No forward checkpoint found in {run_dir}; tried {names}.")


def _resolved_forward_config(run_dir: Path, checkpoint: Mapping[str, Any], forward_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    config_name = str(forward_cfg.get("config_name", "resolved_train_config.json"))
    if config_name.lower() != "none":
        cfg_path = run_dir / config_name
        if cfg_path.exists():
            return read_json(cfg_path)
    train_cfg = checkpoint.get("train_config", {})
    return dict(train_cfg) if isinstance(train_cfg, Mapping) else {}


def latest_local_surrogate_checkpoint(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    runs = sorted([path for path in root.glob("Run_*") if path.is_dir()])
    for run in reversed(runs):
        for name in ("best_model.pt", "latest_model.pt"):
            candidate = run / name
            if candidate.exists():
                return candidate.resolve()
    return None


def _resolve_local_surrogate_path(
    model_config: GlobalChannelThermalModelConfig,
    forward_cfg: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    resolved_cfg: Mapping[str, Any],
) -> Optional[Path]:
    if not model_config.use_local_surrogate:
        return None
    candidates = [
        forward_cfg.get("local_surrogate_checkpoint_path"),
        checkpoint.get("train_config", {}).get("model", {}).get("local_surrogate_checkpoint_path") if isinstance(checkpoint.get("train_config", {}), Mapping) else None,
        resolved_cfg.get("model", {}).get("local_surrogate_checkpoint_path") if isinstance(resolved_cfg.get("model", {}), Mapping) else None,
    ]
    for candidate in candidates:
        if candidate:
            path = resolve_demo_path(candidate)
            if path.exists():
                return path
    if bool(forward_cfg.get("local_surrogate_auto", True)):
        return latest_local_surrogate_checkpoint(resolve_demo_path(forward_cfg.get("local_surrogate_saved_root", "./Saved_Model_LocalModule")))
    return None


def load_forward_model(forward_cfg: Mapping[str, Any], device: torch.device) -> Tuple[GlobalChannelThermalModel, Dict[str, Any], Path]:
    checkpoint_path, run_dir = resolve_forward_checkpoint(forward_cfg)
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    resolved_cfg = _resolved_forward_config(run_dir, checkpoint, forward_cfg)
    model_payload = checkpoint.get("model_config") or resolved_cfg.get("model", {})
    model_config = GlobalChannelThermalModelConfig.from_dict(dict(model_payload))
    model = GlobalChannelThermalModel(model_config).to(device)
    global_norm_cfg = checkpoint.get("global_normalization_config", {})
    if not isinstance(global_norm_cfg, Mapping):
        global_norm_cfg = resolved_cfg.get("dataset", {})
    model.set_global_target_normalization(
        checkpoint.get("global_normalization_stats", {}),
        normalize_targets=bool(global_norm_cfg.get("normalize_targets", False)),
    )
    local_path = _resolve_local_surrogate_path(model_config, forward_cfg, checkpoint, resolved_cfg)
    if model_config.use_local_surrogate:
        if local_path is None:
            raise ValueError(
                "Forward checkpoint requires model.use_local_surrogate=true, but no local surrogate checkpoint could be resolved. "
                "Set forward_model.local_surrogate_checkpoint_path or enable local_surrogate_auto with Saved_Model_LocalModule runs."
            )
        local_model, local_checkpoint = load_local_surrogate_from_checkpoint(local_path, map_location=device)
        warn_if_synthetic_only_local_surrogate(local_checkpoint)
        normalization_config = local_surrogate_normalization_config(local_checkpoint)
        normalization_stats = require_local_normalization_stats(local_checkpoint, normalization_config)
        local_model.to(device)
        model.set_local_surrogate(
            local_model,
            freeze=bool(model_config.freeze_local_surrogate),
            normalization_config=normalization_config,
            normalization_stats=normalization_stats,
        )
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    model.load_state_dict(strip_module_prefix(state), strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "resolved_config": resolved_cfg,
        "local_surrogate_checkpoint_path": str(local_path) if local_path is not None else None,
    }
    print(
        f"[forward] loaded {checkpoint_path} "
        f"mode={'global+local_surrogate' if model_config.use_local_surrogate else 'global_only'}"
    )
    if local_path is not None:
        print(f"[forward] local surrogate: {local_path}")
    return model, metadata, checkpoint_path


def _normalizer_stats_from_checkpoint(metadata: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool, bool]:
    checkpoint = metadata.get("checkpoint", {})
    train_cfg = checkpoint.get("train_config", {}) if isinstance(checkpoint, Mapping) else {}
    dataset_cfg = train_cfg.get("dataset", {}) if isinstance(train_cfg, Mapping) else {}
    global_norm = checkpoint.get("global_normalization_config", {}) if isinstance(checkpoint, Mapping) else {}
    if not isinstance(global_norm, Mapping):
        global_norm = dataset_cfg
    stats = checkpoint.get("global_normalization_stats", {}) if isinstance(checkpoint, Mapping) else {}
    return dict(stats) if isinstance(stats, Mapping) else {}, bool(global_norm.get("normalize_inputs", False)), bool(global_norm.get("normalize_targets", False))


def _apply_heat_normalization(heat: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    stats, normalize_inputs, _ = _normalizer_stats_from_checkpoint(metadata)
    if not normalize_inputs or "heat_power_mean" not in stats or "heat_power_std" not in stats:
        return heat.astype(np.float32)
    mean = np.asarray(stats["heat_power_mean"], dtype=np.float32)
    std = np.asarray(stats["heat_power_std"], dtype=np.float32)
    std = np.where(np.abs(std) < 1.0e-8, 1.0, std)
    return ((heat.astype(np.float32) - mean) / std).astype(np.float32)


def _denormalize_prediction_array(values: np.ndarray, metadata: Mapping[str, Any], mean_key: str, std_key: str, *alternate: str) -> np.ndarray:
    stats, _, normalize_targets = _normalizer_stats_from_checkpoint(metadata)
    if not normalize_targets:
        return values.astype(np.float32)
    mean = None
    std = None
    for key in (mean_key, *alternate):
        if key in stats:
            mean = np.asarray(stats[key], dtype=np.float32)
            break
    for key in (std_key, *(item.replace("mean", "std") for item in alternate)):
        if key in stats:
            std = np.asarray(stats[key], dtype=np.float32)
            break
    if mean is None or std is None:
        return values.astype(np.float32)
    std = np.where(np.abs(std) < 1.0e-8, 1.0, std)
    return (values.astype(np.float32) * std + mean).astype(np.float32)


def _padded_design_arrays(
    record: ThermalInverseCaseRecord,
    candidate: Mapping[str, Any],
    max_num_modules: int,
    metadata: Mapping[str, Any],
    *,
    generate_heat_power: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
    count = min(int(centers.shape[0]), int(max_num_modules))
    padded_centers = np.zeros((max_num_modules, 2), dtype=np.float32)
    present = np.zeros((max_num_modules,), dtype=np.float32)
    heat = np.zeros((max_num_modules,), dtype=np.float32)
    if count > 0:
        padded_centers[:count] = centers[:count]
        present[:count] = 1.0
        if generate_heat_power and candidate.get("heat_powers") is not None:
            generated_heat = np.asarray(candidate["heat_powers"], dtype=np.float32).reshape(-1)
            heat[:count] = np.resize(generated_heat, count)[:count]
        else:
            active_heat = record.heat_powers[record.module_present > 0.5]
            if active_heat.size == 0:
                active_heat = record.heat_powers
            heat[:count] = np.resize(active_heat.astype(np.float32), count)[:count]
    return padded_centers, present, heat.astype(np.float32), _apply_heat_normalization(heat, metadata)


def _local_module_params_from_raw_heat(record: ThermalInverseCaseRecord, heat_raw: np.ndarray, present: np.ndarray) -> np.ndarray:
    params = np.zeros((heat_raw.shape[0], 7), dtype=np.float32)
    params[:, 0] = heat_raw.astype(np.float32)
    if record.material_params.shape[0] > 3:
        params[:, 1] = float(record.material_params[3])
    if record.material_params.shape[0] > 1:
        params[:, 2] = float(record.material_params[1])
    return params * present[:, None].astype(np.float32)


def predict_candidate_with_forward(
    model: GlobalChannelThermalModel,
    metadata: Mapping[str, Any],
    record: ThermalInverseCaseRecord,
    candidate: Mapping[str, Any],
    device: torch.device,
    *,
    max_num_modules: int,
    generate_heat_power: bool = False,
    query_batch_size: int = 32768,
) -> Dict[str, Any]:
    centers, present, heat_raw, heat = _padded_design_arrays(record, candidate, max_num_modules, metadata, generate_heat_power=generate_heat_power)
    query_xy = np.stack([record.x_grid.reshape(-1), record.y_grid.reshape(-1)], axis=-1).astype(np.float32)
    structure = {
        "re": torch.tensor([[record.re]], dtype=torch.float32, device=device),
        "u_in": torch.tensor([[record.u_in]], dtype=torch.float32, device=device),
        "module_centers": torch.from_numpy(centers[None]).to(device),
        "heat_powers": torch.from_numpy(heat[None]).to(device),
        "module_present": torch.from_numpy(present[None]).to(device),
        "material_params": torch.from_numpy(record.material_params[None]).to(device),
        "domain_length_x": torch.tensor([[record.domain_length_x]], dtype=torch.float32, device=device),
        "domain_length_y": torch.tensor([[record.domain_length_y]], dtype=torch.float32, device=device),
    }
    local_module_params = torch.from_numpy(_local_module_params_from_raw_heat(record, heat_raw, present)[None]).to(device)
    local_query = torch.from_numpy(record.module_internal_query_points[None]).to(device) if record.module_internal_query_points.size else None
    pred_chunks: List[np.ndarray] = []
    first_outputs: Optional[Dict[str, Any]] = None
    with torch.no_grad():
        for start in range(0, query_xy.shape[0], int(query_batch_size)):
            q = torch.from_numpy(query_xy[start : start + int(query_batch_size)][None]).to(device)
            outputs = model(
                structure,
                q,
                interface_condition=None,
                local_module_params=local_module_params,
                teacher_port_tokens=None,
                local_query_points=local_query,
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.0,
            )
            pred_chunks.append(outputs["pred_field"].detach().cpu().numpy()[0])
            if first_outputs is None:
                first_outputs = outputs
    if first_outputs is None:
        raise RuntimeError("Forward prediction produced no chunks.")
    pred_field = np.concatenate(pred_chunks, axis=0).reshape(record.x_grid.shape[0], record.x_grid.shape[1], model.config.field_dim)
    pred_field = _denormalize_prediction_array(pred_field, metadata, "field_mean_by_channel", "field_std_by_channel")
    pred_internal = first_outputs["pred_internal_temperature"].detach().cpu().numpy()[0]
    pred_interface = first_outputs["pred_interface"].detach().cpu().numpy()[0]
    pred_internal = _denormalize_prediction_array(pred_internal, metadata, "internal_temperature_mean", "internal_temperature_std")
    pred_interface = _denormalize_prediction_array(pred_interface, metadata, "interface_target_mean", "interface_target_std", "interface_targets_mean")
    aux = {
        key: value.detach().cpu().numpy()[0] if torch.is_tensor(value) and value.ndim > 0 else value
        for key, value in first_outputs.get("organizer_aux", {}).items()
    }
    return {
        "pred_field_grid": pred_field,
        "pred_internal_temperature": pred_internal,
        "pred_interface": pred_interface,
        "pred_port_condition": first_outputs["pred_port_condition"].detach().cpu().numpy()[0],
        "organizer_aux": aux,
        "centers_padded": centers,
        "module_present": present,
        "heat_powers": heat_raw,
    }


def _summary_vector(value: Any) -> np.ndarray:
    arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr.reshape(-1, 1)
    return np.concatenate(
        [
            np.nanmean(flat, axis=0),
            np.nanstd(flat, axis=0),
            np.nanmin(flat, axis=0),
            np.nanmax(flat, axis=0),
        ],
        axis=0,
    ).astype(np.float32)


def _pad_or_trim(vec: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros((int(dim),), dtype=np.float32)
    if vec.size:
        out[: min(out.size, vec.size)] = vec.reshape(-1)[: out.size]
    return out


def extract_forward_latent_targets(
    prediction: Mapping[str, Any],
    *,
    behavior_dim: int,
    organization_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    behavior_parts = []
    for key in ("pred_internal_temperature", "pred_interface", "pred_port_condition"):
        if key in prediction:
            behavior_parts.append(_summary_vector(prediction[key]))
    behavior = _pad_or_trim(np.concatenate(behavior_parts, axis=0) if behavior_parts else np.zeros((0,), dtype=np.float32), behavior_dim)

    aux = prediction.get("organizer_aux", {})
    org_parts = []
    if isinstance(aux, Mapping):
        for key in ("A_mh", "A_eh", "hyper_state", "hyper_strength", "hyper_module_mass", "hyper_env_mass", "hyper_source_coords", "hyper_thermal_region_coords", "module_centers", "heat_powers"):
            if key in aux:
                org_parts.append(_summary_vector(aux[key]))
    organization = _pad_or_trim(np.concatenate(org_parts, axis=0) if org_parts else np.zeros((0,), dtype=np.float32), organization_dim)
    return behavior, organization


def build_forward_latent_cache(
    model: GlobalChannelThermalModel,
    metadata: Mapping[str, Any],
    records: Sequence[ThermalInverseCaseRecord],
    cache_path: Path,
    device: torch.device,
    *,
    behavior_dim: int,
    organization_dim: int,
    max_num_modules: int,
    generate_heat_power: bool,
    force_rebuild: bool = False,
    query_batch_size: int = 32768,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if cache_path.exists() and not force_rebuild:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            isinstance(payload, Mapping)
            and "latents" in payload
            and int(payload.get("cache_version", -1)) == 2
            and int(payload.get("behavior_dim", -1)) == int(behavior_dim)
            and int(payload.get("organization_dim", -1)) == int(organization_dim)
        ):
            print(f"[latent-cache] loaded {cache_path}")
            return {
                str(k): (np.asarray(v["behavior"], dtype=np.float32), np.asarray(v["organization"], dtype=np.float32))
                for k, v in payload["latents"].items()
            }
        print(f"[latent-cache] ignoring incompatible cache {cache_path}")
    latents: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for record in tqdm(records, desc="latent-cache", unit="case", dynamic_ncols=True, leave=False):
        candidate = {"centers": record.module_centers[record.module_present > 0.5], "count": record.true_count, "heat_powers": record.heat_powers[record.module_present > 0.5]}
        prediction = predict_candidate_with_forward(
            model,
            metadata,
            record,
            candidate,
            device,
            max_num_modules=max_num_modules,
            generate_heat_power=generate_heat_power,
            query_batch_size=query_batch_size,
        )
        latents[record.case_id] = extract_forward_latent_targets(prediction, behavior_dim=behavior_dim, organization_dim=organization_dim)
    ensure_dir(cache_path.parent)
    torch.save(
        {
            "latents": {
                key: {"behavior": value[0], "organization": value[1]}
                for key, value in latents.items()
            },
            "behavior_dim": behavior_dim,
            "organization_dim": organization_dim,
            "cache_version": 2,
            "forward_checkpoint": metadata.get("checkpoint_path"),
        },
        cache_path,
    )
    print(f"[latent-cache] wrote {cache_path}")
    return latents


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return recursive_to_device(dict(batch), device)


def scalar(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def mean_rows(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    return {key: float(np.nanmean([row.get(key, float("nan")) for row in rows])) for key in keys}


def run_supervised_epoch(
    model: ThermalInverseDesignFlow,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Any = None,
    amp: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows: List[Dict[str, float]] = []
    iterator = tqdm(loader, desc="train" if training else "val", unit="batch", dynamic_ncols=True, leave=False)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context(device, amp):
            loss, metrics = model.training_loss(batch, loss_weights=loss_cfg)
        if training:
            clip_norm = float(training_cfg.get("gradient_clip_norm", 1.0))
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
        row = {key: scalar(value) for key, value in metrics.items()}
        rows.append(row)
        iterator.set_postfix(loss=f"{row['loss_total']:.3e}", count=f"{row['count_accuracy']:.2f}")
    return mean_rows(rows)


def sample_diversity(candidates: Sequence[Mapping[str, Any]], max_num_modules: int) -> float:
    if len(candidates) < 2:
        return 0.0
    vecs = []
    for cand in candidates:
        centers = np.asarray(cand.get("centers", []), dtype=np.float32).reshape(-1, 2)
        padded = np.zeros((max_num_modules, 2), dtype=np.float32)
        n = min(centers.shape[0], max_num_modules)
        padded[:n] = centers[:n]
        vecs.append(padded.reshape(-1))
    arr = np.stack(vecs)
    dists = []
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            dists.append(float(np.linalg.norm(arr[i] - arr[j])))
    return float(np.mean(dists)) if dists else 0.0


def run_forward_verification(
    inverse_model: ThermalInverseDesignFlow,
    forward_model: GlobalChannelThermalModel,
    forward_metadata: Mapping[str, Any],
    dataset: ThermalInverseDesignDataset,
    device: torch.device,
    *,
    num_targets: int,
    num_samples: int,
    n_steps: int,
    query_batch_size: int,
    seed: int,
) -> Dict[str, float]:
    if num_targets <= 0 or num_samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(int(num_targets), len(dataset)), replace=False)
    scores = []
    valid = []
    diversity = []
    kpi_errors: Dict[str, List[float]] = {}
    for idx in indices:
        item = dataset[int(idx)]
        record = dataset.records[int(idx)]
        target_spec_vec = item["target_spec_vector"]
        candidates = inverse_model.sample_designs(target_spec_vec, n_samples=int(num_samples), n_steps=int(n_steps), seed=int(seed + idx), device=device)
        diversity.append(sample_diversity(candidates, inverse_model.max_num_modules))
        best_score = float("inf")
        best_valid = 0.0
        for cand in candidates:
            prediction = predict_candidate_with_forward(
                forward_model,
                forward_metadata,
                record,
                cand,
                device,
                max_num_modules=inverse_model.max_num_modules,
                generate_heat_power=inverse_model.cfg.generate_heat_power,
                query_batch_size=query_batch_size,
            )
            kpis = compute_steady_thermal_kpis(
                prediction["pred_field_grid"],
                x_grid=record.x_grid,
                y_grid=record.y_grid,
                channel_order=CHANNEL_ORDER,
                module_centers=prediction["centers_padded"],
                module_present=prediction["module_present"],
                heat_powers=prediction.get("heat_powers", record.heat_powers),
                module_internal_temperature=prediction["pred_internal_temperature"],
                module_internal_mask=record.module_internal_mask,
                interface_target=prediction["pred_interface"],
                interface_condition=None,
                domain={"domain_length_x": record.domain_length_x, "domain_length_y": record.domain_length_y, "module_radius": record.module_radius},
                material_params=record.material_params,
            )
            kpis.update(cand.get("validity", {}))
            kpis["num_modules"] = int(cand.get("count", 0))
            target_spec = {
                "kpi_names": list(dataset.kpi_names),
                "kpi_targets": item.get("target_kpi_targets", {}),
                "kpi_stats": dataset.kpi_stats,
                "constraints": {
                    "num_modules_min": record.true_count,
                    "num_modules_max": record.true_count,
                },
            }
            scored = score_candidate_kpis(kpis, target_spec)
            if scored["total_score"] < best_score:
                best_score = float(scored["total_score"])
                best_valid = 1.0 if bool(cand.get("validity", {}).get("valid", False)) else 0.0
                for name, err in scored.get("per_kpi_errors", {}).items():
                    kpi_errors.setdefault(name, []).append(float(err))
        scores.append(best_score)
        valid.append(best_valid)
    metrics = {
        "val_forward_score": float(np.mean(scores)) if scores else float("nan"),
        "val_forward_validity_rate": float(np.mean(valid)) if valid else float("nan"),
        "val_candidate_diversity": float(np.mean(diversity)) if diversity else float("nan"),
    }
    for name, values in kpi_errors.items():
        metrics[f"val_kpi_error_{name}"] = float(np.mean(values))
    return metrics


def save_history_csv(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not history:
        return
    keys = sorted({key for row in history for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in keys})


def save_loss_curve(history_path: Path, out_path: Path) -> None:
    if not history_path.exists():
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with history_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    epochs = [float(row["epoch"]) for row in rows if row.get("epoch")]
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for key in ("train_loss_total", "val_loss_total", "val_forward_score"):
        values = []
        for row in rows:
            try:
                values.append(float(row.get(key, "nan")))
            except ValueError:
                values.append(float("nan"))
        if values and any(math.isfinite(v) for v in values):
            ax.plot(epochs[: len(values)], values, label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / score")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    *,
    model: ThermalInverseDesignFlow,
    cfg: Dict[str, Any],
    model_config: InverseModelConfig,
    kpi_names: Sequence[str],
    kpi_stats: Mapping[str, Any],
    epoch: int,
    best_metric: float,
    forward_checkpoint: Optional[str],
) -> None:
    torch.save(
        {
            "stage": "channelthermal_inverse_rectified_flow",
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "model_config": model_config.to_dict(),
            "model_state_dict": model.state_dict(),
            "train_config": cfg,
            "kpi_names": list(kpi_names),
            "kpi_stats": dict(kpi_stats),
            "forward_checkpoint": forward_checkpoint,
        },
        path,
    )


def resolve_run_id(args: argparse.Namespace, cfg: Mapping[str, Any]) -> str:
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), Mapping) else {}
    return normalize_run_id(args.run_id or cfg.get("Run_ID") or cfg.get("run_id") or training_cfg.get("Run_ID") or training_cfg.get("run_id"), "0001")


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    cfg = read_json(config_path)
    dataset_cfg = cfg.get("dataset", {})
    target_cfg = cfg.get("target_kpis", {})
    inverse_cfg = dict(cfg.get("inverse_model", {}))
    training_cfg = cfg.get("training", {})
    validation_cfg = cfg.get("validation", {})
    loss_cfg = cfg.get("loss", {})
    set_seed(int(training_cfg.get("seed", cfg.get("seed", 42))))
    device = select_device(args.device or training_cfg.get("device"))

    kpi_names = tuple(target_cfg.get("names", DEFAULT_KPI_NAMES))
    max_train_cases = args.max_train_cases if args.max_train_cases is not None else int(dataset_cfg.get("max_train_cases", 0) or 0)
    max_val_cases = args.max_val_cases if args.max_val_cases is not None else int(dataset_cfg.get("max_val_cases", 0) or 0)
    packed_path = dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5")

    with h5py.File(resolve_demo_path(packed_path), "r") as h5:
        root_max_modules = int(h5.attrs.get("max_modules", 1))
    if str(inverse_cfg.get("max_num_modules", "auto")).lower() == "auto":
        inverse_cfg["max_num_modules"] = root_max_modules
    generate_heat_power = bool(inverse_cfg.get("generate_heat_power", False))
    heat_scale = float(inverse_cfg.get("heat_power_scale", target_cfg.get("heat_power_scale", 1.0)))

    train_dataset = ThermalInverseDesignDataset(
        packed_path,
        split=dataset_cfg.get("train_split", "train"),
        kpi_names=kpi_names,
        normalize_targets=bool(target_cfg.get("normalize", True)),
        target_augmentation=cfg.get("target_augmentation", {}),
        max_num_modules=int(inverse_cfg["max_num_modules"]),
        generate_heat_power=generate_heat_power,
        heat_power_scale=heat_scale,
        sort_centers=bool(dataset_cfg.get("sort_centers", True)),
        max_cases=max_train_cases,
        use_all_if_split_missing=bool(dataset_cfg.get("use_all_if_split_missing", True)),
        seed=int(training_cfg.get("seed", 42)),
        behavior_latent_dim=int(inverse_cfg.get("behavior_latent_dim", 96)),
        organization_latent_dim=int(inverse_cfg.get("organization_latent_dim", 256)),
    )
    val_dataset = ThermalInverseDesignDataset(
        packed_path,
        split=dataset_cfg.get("val_split", "test"),
        kpi_names=kpi_names,
        normalize_targets=bool(target_cfg.get("normalize", True)),
        target_augmentation=cfg.get("target_augmentation", {}),
        max_num_modules=int(inverse_cfg["max_num_modules"]),
        generate_heat_power=generate_heat_power,
        heat_power_scale=heat_scale,
        sort_centers=bool(dataset_cfg.get("sort_centers", True)),
        max_cases=max_val_cases,
        use_all_if_split_missing=bool(dataset_cfg.get("use_all_if_split_missing", True)),
        seed=int(training_cfg.get("seed", 42)) + 1000,
        behavior_latent_dim=int(inverse_cfg.get("behavior_latent_dim", 96)),
        organization_latent_dim=int(inverse_cfg.get("organization_latent_dim", 256)),
    )
    kpi_stats = compute_kpi_stats(np.stack([record.kpi_vector for record in train_dataset.records]), kpi_names)
    train_dataset.set_kpi_stats(kpi_stats)
    val_dataset.set_kpi_stats(kpi_stats)

    sample_record = train_dataset.records[0]
    inverse_cfg["target_dim"] = len(kpi_names) * 7 + 8 if str(inverse_cfg.get("target_dim", "auto")).lower() == "auto" else inverse_cfg.get("target_dim")
    inverse_cfg["domain_length_x"] = float(inverse_cfg.get("domain_length_x", "auto") if str(inverse_cfg.get("domain_length_x", "auto")).lower() != "auto" else sample_record.domain_length_x)
    inverse_cfg["domain_length_y"] = float(inverse_cfg.get("domain_length_y", "auto") if str(inverse_cfg.get("domain_length_y", "auto")).lower() != "auto" else sample_record.domain_length_y)
    inverse_cfg["module_radius"] = float(inverse_cfg.get("module_radius", "auto") if str(inverse_cfg.get("module_radius", "auto")).lower() != "auto" else sample_record.module_radius)
    inverse_cfg["heat_power_scale"] = heat_scale
    model_config = InverseModelConfig.from_dict(inverse_cfg)

    forward_model = None
    forward_metadata: Dict[str, Any] = {}
    if bool(cfg.get("forward_model", {}).get("enabled", True)):
        forward_model, forward_metadata, _ = load_forward_model(cfg.get("forward_model", {}), device)
        cache_name = str(cfg.get("forward_model", {}).get("latent_cache_name", "inverse_forward_latent_cache.pt"))
        cache_path = (resolve_demo_path(cfg.get("paths", {}).get("saved_inverse_root", cfg.get("paths", {}).get("saved_model_dir", "./Saved_Model_Inverse"))) / "cache" / cache_name).resolve()
        latents = build_forward_latent_cache(
            forward_model,
            forward_metadata,
            list(train_dataset.records) + list(val_dataset.records),
            cache_path,
            device,
            behavior_dim=model_config.behavior_latent_dim,
            organization_dim=model_config.organization_latent_dim,
            max_num_modules=model_config.max_num_modules,
            generate_heat_power=model_config.generate_heat_power,
            force_rebuild=bool(cfg.get("forward_model", {}).get("rebuild_latent_cache", False)),
            query_batch_size=int(cfg.get("forward_model", {}).get("query_batch_size", 32768)),
        )
        train_dataset.set_latent_targets(latents)
        val_dataset.set_latent_targets(latents)

    model = ThermalInverseDesignFlow(model_config).to(device)
    print(f"[setup] inverse parameters={count_parameters(model):,}, train_cases={len(train_dataset)}, val_cases={len(val_dataset)}, device={device}")
    print(f"[setup] design variables: centers+mask{' + heat_power' if model_config.generate_heat_power else ''}, max_num_modules={model_config.max_num_modules}")

    if args.dry_run:
        batch = collate_inverse([train_dataset[0], train_dataset[min(1, len(train_dataset) - 1)]])
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            _, metrics = model.training_loss(batch, loss_weights=loss_cfg)
        print("[dry-run] batch keys:", sorted(batch.keys()))
        print("[dry-run] metrics:", {key: scalar(value) for key, value in metrics.items()})
        return 0

    run_id = resolve_run_id(args, cfg)
    suffix = sanitize_run_suffix(args.run_name or cfg.get("case_id") or cfg.get("description", "inverse"))
    run_name = f"Run_{run_id}_{current_timestamp()}" + (f"_{suffix}" if suffix else "")
    saved_root = ensure_dir(resolve_demo_path(cfg.get("paths", {}).get("saved_inverse_root", cfg.get("paths", {}).get("saved_model_dir", "./Saved_Model_Inverse"))))
    run_dir = ensure_dir(saved_root / run_name)
    write_json(run_dir / "resolved_train_inverse_config.json", cfg)
    write_json(run_dir / "kpi_stats.json", kpi_stats)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collate_inverse,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(validation_cfg.get("batch_size", training_cfg.get("batch_size", 64))),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collate_inverse,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get("learning_rate", 1.0e-4)), weight_decay=float(training_cfg.get("weight_decay", 1.0e-5)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.epochs or training_cfg.get("epochs", 1000)), 1),
        eta_min=float(training_cfg.get("scheduler_min_lr", 1.0e-6)),
    )
    scaler = make_grad_scaler(device, bool(training_cfg.get("amp", False)))
    amp = bool(training_cfg.get("amp", False))
    epochs = int(args.epochs or training_cfg.get("epochs", 1000))
    max_train_batches = args.max_train_batches if args.max_train_batches is not None else training_cfg.get("max_train_batches_per_epoch")
    max_val_batches = args.max_val_batches if args.max_val_batches is not None else training_cfg.get("max_val_batches")
    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_verified = float("inf")
    history_path = run_dir / "loss_history.csv"
    for epoch in range(1, epochs + 1):
        train_metrics = run_supervised_epoch(
            model,
            train_loader,
            device,
            loss_cfg,
            training_cfg,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            max_batches=int(max_train_batches) if max_train_batches is not None else None,
        )
        val_metrics = run_supervised_epoch(
            model,
            val_loader,
            device,
            loss_cfg,
            training_cfg,
            optimizer=None,
            scaler=None,
            amp=False,
            max_batches=int(max_val_batches) if max_val_batches is not None else None,
        )
        scheduler.step()
        row: Dict[str, Any] = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        verify_every = int(validation_cfg.get("forward_verify_every_epochs", 0) or 0)
        if forward_model is not None and verify_every > 0 and epoch % verify_every == 0:
            verify_metrics = run_forward_verification(
                model,
                forward_model,
                forward_metadata,
                val_dataset,
                device,
                num_targets=int(validation_cfg.get("forward_verify_num_targets", 4)),
                num_samples=int(validation_cfg.get("forward_verify_num_samples", 4)),
                n_steps=int(validation_cfg.get("forward_verify_n_steps", 16)),
                query_batch_size=int(validation_cfg.get("forward_verify_query_batch_size", cfg.get("forward_model", {}).get("query_batch_size", 32768))),
                seed=int(training_cfg.get("seed", 42)) + epoch,
            )
            row.update(verify_metrics)
            verified = float(verify_metrics.get("val_forward_score", float("inf")))
            if math.isfinite(verified) and verified < best_verified:
                best_verified = verified
                save_checkpoint(
                    run_dir / "best_verified_model.pt",
                    model=model,
                    cfg=cfg,
                    model_config=model_config,
                    kpi_names=kpi_names,
                    kpi_stats=kpi_stats,
                    epoch=epoch,
                    best_metric=best_verified,
                    forward_checkpoint=forward_metadata.get("checkpoint_path"),
                )
        val_loss = float(val_metrics.get("loss_total", float("inf")))
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                run_dir / "best_model.pt",
                model=model,
                cfg=cfg,
                model_config=model_config,
                kpi_names=kpi_names,
                kpi_stats=kpi_stats,
                epoch=epoch,
                best_metric=best_val,
                forward_checkpoint=forward_metadata.get("checkpoint_path"),
            )
        save_checkpoint(
            run_dir / "latest_model.pt",
            model=model,
            cfg=cfg,
            model_config=model_config,
            kpi_names=kpi_names,
            kpi_stats=kpi_stats,
            epoch=epoch,
            best_metric=best_val,
            forward_checkpoint=forward_metadata.get("checkpoint_path"),
        )
        history.append(row)
        save_history_csv(history, history_path)
        if epoch % max(int(training_cfg.get("save_every_epochs", 25)), 1) == 0 or epoch == epochs:
            save_loss_curve(history_path, run_dir / "loss_curve.png")
        print(
            f"[epoch {epoch:04d}] train={train_metrics.get('loss_total', float('nan')):.4e} "
            f"val={val_loss:.4e} forward={row.get('val_forward_score', float('nan')):.4e}"
        )
    save_loss_curve(history_path, run_dir / "loss_curve.png")
    print(f"[done] inverse run saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
