from __future__ import annotations

"""Train the steady ChannelThermal inverse-design generator."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
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
    layout_spread_metrics,
    score_candidate_kpis,
)
from thermal_design_intent import (
    DEFAULT_FIELD_MAP_SHAPE,
    STRUCTURE_INTENT_DIM,
    build_design_intent_arrays,
    build_layout_structure_maps,
    compute_design_intent_score,
    compute_layout_structure_features,
    is_design_intent_payload,
    normalize_intent_payload,
    training_intent_from_record,
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
    parser.add_argument("--run-name", type=str, default=None, help="Deprecated; run directories are now only Run_ID plus timestamp.")
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


def stable_json_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_fingerprint(path_like: str | Path) -> Dict[str, Any]:
    path = resolve_demo_path(path_like)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


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
    structure_intent_vector: np.ndarray
    structure_intent_maps: np.ndarray
    heat_condition_vector: np.ndarray
    heat_condition_mask: np.ndarray
    heat_condition_stats: np.ndarray


HYPERGRAPH_PLAN_FIELDS = [
    "edge_active_or_strength",
    "hyper_strength",
    "module_mass",
    "env_mass",
    "source_x",
    "source_y",
    "thermal_region_x",
    "thermal_region_y",
]
HYPERGRAPH_PLAN_EDGE_ORDERING = "hyper_strength_desc_then_thermal_region_xy"


def hypergraph_plan_dim(max_num_modules: int, num_edges: int) -> int:
    return int(num_edges) * (len(HYPERGRAPH_PLAN_FIELDS) + int(max_num_modules))


def infer_hypergraph_plan_num_edges(plan_dim: int, max_num_modules: int) -> int:
    denom = len(HYPERGRAPH_PLAN_FIELDS) + int(max_num_modules)
    if denom <= 0 or int(plan_dim) <= 0:
        return 0
    return max(int(plan_dim) // denom, 0)


def _aux_array(aux: Mapping[str, Any], key: str, shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    if key not in aux:
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    arr = np.asarray(aux[key], dtype=np.float32)
    out = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.float32)
    slices = tuple(slice(0, min(out.shape[i], arr.shape[i] if i < arr.ndim else 1)) for i in range(len(shape)))
    if arr.ndim < len(shape):
        arr = arr.reshape(arr.shape + (1,) * (len(shape) - arr.ndim))
    if all(s.stop > 0 for s in slices):
        out[slices] = arr[slices]
        mask[slices] = 1.0
    return out, mask


def build_hypergraph_plan_from_forward_prediction(
    prediction: Mapping[str, Any],
    *,
    max_num_modules: int,
    domain_length_x: float,
    domain_length_y: float,
    num_edges: Optional[int] = None,
) -> Dict[str, Any]:
    aux = prediction.get("organizer_aux", {})
    if not isinstance(aux, Mapping):
        aux = {}
    if num_edges is None:
        if "hyper_strength" in aux:
            num_edges = int(np.asarray(aux["hyper_strength"]).reshape(-1).shape[0])
        elif "A_mh" in aux and np.asarray(aux["A_mh"]).ndim >= 2:
            num_edges = int(np.asarray(aux["A_mh"]).shape[-1])
        else:
            num_edges = 0
    k = max(int(num_edges or 0), 0)
    m = int(max_num_modules)
    dim = hypergraph_plan_dim(m, k)
    if k <= 0:
        empty = np.zeros((dim,), dtype=np.float32)
        return {
            "vector": empty,
            "mask": empty.copy(),
            "summary": {},
            "metadata": {
                "hypergraph_plan_num_edges": k,
                "hypergraph_plan_fields": list(HYPERGRAPH_PLAN_FIELDS),
                "hypergraph_plan_dim": dim,
                "hypergraph_plan_edge_ordering": HYPERGRAPH_PLAN_EDGE_ORDERING,
            },
        }

    strength, strength_mask = _aux_array(aux, "hyper_strength", (k,))
    module_mass, module_mask = _aux_array(aux, "hyper_module_mass", (k,))
    env_mass, env_mask = _aux_array(aux, "hyper_env_mass", (k,))
    source, source_mask = _aux_array(aux, "hyper_source_coords", (k, 2))
    thermal, thermal_mask = _aux_array(aux, "hyper_thermal_region_coords", (k, 2))
    a_mh, a_mh_mask = _aux_array(aux, "A_mh", (m, k))
    source_norm = source.copy()
    thermal_norm = thermal.copy()
    source_norm[:, 0] /= max(float(domain_length_x), 1.0e-8)
    source_norm[:, 1] /= max(float(domain_length_y), 1.0e-8)
    thermal_norm[:, 0] /= max(float(domain_length_x), 1.0e-8)
    thermal_norm[:, 1] /= max(float(domain_length_y), 1.0e-8)
    if "active_hyperedge_mask" in aux:
        active_raw = np.asarray(aux["active_hyperedge_mask"], dtype=np.float32).reshape(-1)
        active = np.zeros((k,), dtype=np.float32)
        active[: min(k, active_raw.size)] = active_raw[: min(k, active_raw.size)]
        active_mask = np.zeros((k,), dtype=np.float32)
        active_mask[: min(k, active_raw.size)] = 1.0
    else:
        active = strength.copy()
        active_mask = strength_mask.copy()
    order = np.lexsort((thermal_norm[:, 1], thermal_norm[:, 0], -strength))
    strength = strength[order]
    strength_mask = strength_mask[order]
    module_mass = module_mass[order]
    module_mask = module_mask[order]
    env_mass = env_mass[order]
    env_mask = env_mask[order]
    source_norm = source_norm[order]
    source_mask = source_mask[order]
    thermal_norm = thermal_norm[order]
    thermal_mask = thermal_mask[order]
    active = active[order]
    active_mask = active_mask[order]
    a_mh = a_mh[:, order]
    a_mh_mask = a_mh_mask[:, order]
    edge = np.stack(
        [
            active,
            strength,
            module_mass,
            env_mass,
            source_norm[:, 0],
            source_norm[:, 1],
            thermal_norm[:, 0],
            thermal_norm[:, 1],
        ],
        axis=1,
    ).astype(np.float32)
    edge_mask = np.stack(
        [
            active_mask,
            strength_mask,
            module_mask,
            env_mask,
            source_mask[:, 0],
            source_mask[:, 1],
            thermal_mask[:, 0],
            thermal_mask[:, 1],
        ],
        axis=1,
    ).astype(np.float32)
    vector = np.concatenate([edge.reshape(-1), a_mh.reshape(-1)], axis=0).astype(np.float32)
    mask = np.concatenate([edge_mask.reshape(-1), a_mh_mask.reshape(-1)], axis=0).astype(np.float32)
    active_count = None
    if active_mask.sum() > 0:
        active_count = float(np.sum(active > 0.5))
    elif strength_mask.sum() > 0:
        active_count = float(np.sum(strength > 0.05))
    summary = {
        "edge": edge,
        "edge_mask": edge_mask,
        "A_mh": a_mh,
        "A_mh_mask": a_mh_mask,
        "hyper_strength": strength,
        "module_mass": module_mass,
        "env_mass": env_mass,
        "source_coords": source_norm,
        "thermal_region_coords": thermal_norm,
        "active_edge_count": active_count,
    }
    return {
        "vector": vector,
        "mask": mask,
        "summary": summary,
        "metadata": {
            "hypergraph_plan_num_edges": k,
            "hypergraph_plan_fields": list(HYPERGRAPH_PLAN_FIELDS),
            "hypergraph_plan_dim": int(vector.size),
            "hypergraph_plan_edge_ordering": HYPERGRAPH_PLAN_EDGE_ORDERING,
            "coordinate_normalization": "x/domain_length_x, y/domain_length_y",
        },
    }


def decode_hypergraph_plan_vector(vector: Any, *, max_num_modules: int, num_edges: Optional[int] = None) -> Dict[str, Any]:
    arr = np.asarray(vector if vector is not None else [], dtype=np.float32).reshape(-1)
    if num_edges is None:
        num_edges = infer_hypergraph_plan_num_edges(arr.size, max_num_modules)
    k = max(int(num_edges or 0), 0)
    m = int(max_num_modules)
    dim = hypergraph_plan_dim(m, k)
    padded = np.zeros((dim,), dtype=np.float32)
    if arr.size:
        padded[: min(arr.size, dim)] = arr[: min(arr.size, dim)]
    edge_size = k * len(HYPERGRAPH_PLAN_FIELDS)
    edge = padded[:edge_size].reshape(k, len(HYPERGRAPH_PLAN_FIELDS)) if k else np.zeros((0, len(HYPERGRAPH_PLAN_FIELDS)), dtype=np.float32)
    a_mh = padded[edge_size:].reshape(m, k) if k else np.zeros((m, 0), dtype=np.float32)
    fields = {name: edge[:, idx] if k else np.zeros((0,), dtype=np.float32) for idx, name in enumerate(HYPERGRAPH_PLAN_FIELDS)}
    return {
        "fields": fields,
        "edge": edge,
        "A_mh": a_mh,
        "hyper_strength": fields["hyper_strength"],
        "module_mass": fields["module_mass"],
        "env_mass": fields["env_mass"],
        "source_coords": edge[:, 4:6],
        "thermal_region_coords": edge[:, 6:8],
        "active_edge_count": float(np.sum(fields["edge_active_or_strength"] > 0.5)) if k else 0.0,
        "metadata": {
            "hypergraph_plan_num_edges": k,
            "hypergraph_plan_fields": list(HYPERGRAPH_PLAN_FIELDS),
            "hypergraph_plan_dim": dim,
            "hypergraph_plan_edge_ordering": HYPERGRAPH_PLAN_EDGE_ORDERING,
        },
    }


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
        temperature_limits: Optional[Mapping[str, Any]] = None,
        max_num_modules: Optional[int] = None,
        generate_heat_power: bool = False,
        heat_power_scale: float = 1.0,
        sort_centers: bool = True,
        max_cases: int = 0,
        use_all_if_split_missing: bool = True,
        samples_per_case: int = 1,
        seed: int = 42,
        behavior_latent_dim: int = 96,
        organization_latent_dim: int = 256,
        conditioning_mode: str = "legacy_kpi",
        intent_augmentation: Optional[Mapping[str, Any]] = None,
        structure_conditioning: Optional[Mapping[str, Any]] = None,
        heat_conditioning: Optional[Mapping[str, Any]] = None,
        hypergraph_plan_dim: int = 0,
    ) -> None:
        self.path = resolve_demo_path(packed_h5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Packed ChannelThermal dataset not found: {self.path}")
        self.split = str(split)
        self.kpi_names = tuple(kpi_names)
        self.kpi_stats = kpi_stats
        self.normalize_targets = bool(normalize_targets)
        self.target_augmentation = dict(target_augmentation or {})
        self.temperature_limits = dict(temperature_limits or {})
        self.max_num_modules_config = max_num_modules
        self.generate_heat_power = bool(generate_heat_power)
        self.heat_power_scale = float(heat_power_scale)
        self.sort_centers = bool(sort_centers)
        self.samples_per_case = max(int(samples_per_case), 1)
        self.seed = int(seed)
        self.behavior_latent_dim = int(behavior_latent_dim)
        self.organization_latent_dim = int(organization_latent_dim)
        self.conditioning_mode = str(conditioning_mode).lower().strip()
        self.intent_augmentation = dict(intent_augmentation or {})
        self.structure_conditioning = dict(structure_conditioning or {})
        self.heat_conditioning = dict(heat_conditioning or {})
        self.structure_enabled = bool(self.structure_conditioning.get("enabled", False))
        self.heat_conditioning_enabled = bool(self.heat_conditioning.get("enabled", False))
        self.heat_sort_mode = str(self.heat_conditioning.get("sort_mode", "heat_desc_then_xy")).lower().strip()
        self.kpi_distribution_summary: Optional[Mapping[str, Any]] = None
        self.latent_targets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.hypergraph_plan_dim = max(int(hypergraph_plan_dim or 0), 0)
        self.hypergraph_plan_targets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.normalizer: Optional[H5Normalizer] = None
        self.records = self._load_records(max_cases=max_cases, use_all_if_split_missing=use_all_if_split_missing)
        if not self.records:
            raise RuntimeError(f"No inverse records found in split {self.split!r} from {self.path}.")

    def _slot_order(self, centers: np.ndarray, present: np.ndarray, heat: np.ndarray) -> np.ndarray:
        active = np.flatnonzero(np.asarray(present).reshape(-1) > 0.5)
        if active.size <= 1:
            return active
        active_centers = centers[active]
        active_heat = heat[active] if heat.size >= int(np.max(active)) + 1 else np.zeros((active.size,), dtype=np.float32)
        mode = self.heat_sort_mode
        if mode == "heat_desc_then_xy":
            local = np.lexsort((active_centers[:, 1], active_centers[:, 0], -active_heat))
        elif mode == "xy":
            local = np.lexsort((active_centers[:, 1], active_centers[:, 0]))
        elif mode in {"as_is", "preserve", "dataset"}:
            local = np.arange(active.size)
        else:
            local = np.lexsort((active_centers[:, 1], active_centers[:, 0], -active_heat))
        return active[local]

    def _ordered_layout(self, centers: np.ndarray, present: np.ndarray, heat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = self._slot_order(centers, present, heat) if self.heat_conditioning_enabled else np.flatnonzero(present.reshape(-1) > 0.5)
        out_centers = np.zeros_like(centers, dtype=np.float32)
        out_present = np.zeros_like(present, dtype=np.float32)
        out_heat = np.zeros_like(heat, dtype=np.float32)
        n = min(order.size, out_present.size)
        if n > 0:
            out_centers[:n] = centers[order[:n]]
            out_present[:n] = 1.0
            out_heat[:n] = heat[order[:n]] if heat.size else 0.0
        return out_centers, out_present, out_heat

    def _heat_condition_arrays(self, heat: np.ndarray, present: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_n = int(self.max_num_modules)
        heat = np.asarray(heat, dtype=np.float32).reshape(-1)
        present = np.asarray(present, dtype=np.float32).reshape(-1)
        mask = np.zeros((max_n,), dtype=np.float32)
        heat_norm = np.zeros((max_n,), dtype=np.float32)
        n = min(max_n, heat.size, present.size)
        if n > 0:
            mask[:n] = (present[:n] > 0.5).astype(np.float32)
            heat_norm[:n] = heat[:n] / max(float(self.heat_power_scale), 1.0e-8)
        active_heat = heat[:n][mask[:n] > 0.5] if n else np.asarray([], dtype=np.float32)
        total = float(np.sum(active_heat)) if active_heat.size else 0.0
        mean = float(np.mean(active_heat)) if active_heat.size else 0.0
        std = float(np.std(active_heat)) if active_heat.size else 0.0
        max_heat = float(np.max(active_heat)) if active_heat.size else 0.0
        high_power_fraction = float(np.mean(active_heat >= mean + std)) if active_heat.size and std > 1.0e-8 else 0.0
        stats = np.asarray(
            [
                total / max(float(self.heat_power_scale) * max(float(max_n), 1.0), 1.0e-8),
                mean / max(float(self.heat_power_scale), 1.0e-8),
                std / max(float(self.heat_power_scale), 1.0e-8),
                max_heat / max(float(self.heat_power_scale), 1.0e-8),
                high_power_fraction,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([heat_norm, mask, stats], axis=0).astype(np.float32), mask.astype(np.float32), stats

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
                ordered_centers, ordered_present, ordered_heat = self._ordered_layout(centers, present, heat)
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
                    ordered_centers,
                    ordered_present,
                    ordered_heat,
                    max_num_modules=self.max_num_modules,
                    domain_length_x=float(domain["domain_length_x"]),
                    domain_length_y=float(domain["domain_length_y"]),
                    generate_heat_power=self.generate_heat_power,
                    heat_power_scale=self.heat_power_scale,
                    sort_centers=bool(self.sort_centers and not self.heat_conditioning_enabled),
                )
                structure_vector, _ = compute_layout_structure_features(
                    ordered_centers,
                    ordered_present,
                    ordered_heat,
                    domain_length_x=float(domain["domain_length_x"]),
                    domain_length_y=float(domain["domain_length_y"]),
                    module_radius=radius,
                    max_num_modules=self.max_num_modules,
                    x_bins=int(self.structure_conditioning.get("x_bins", 6)),
                    y_bins=int(self.structure_conditioning.get("y_bins", 4)),
                    pair_distance_bins=int(self.structure_conditioning.get("pair_distance_bins", 6)),
                )
                structure_maps, _ = build_layout_structure_maps(
                    ordered_centers,
                    ordered_present,
                    ordered_heat,
                    domain_length_x=float(domain["domain_length_x"]),
                    domain_length_y=float(domain["domain_length_y"]),
                    module_radius=radius,
                    max_num_modules=self.max_num_modules,
                    shape=DEFAULT_FIELD_MAP_SHAPE,
                )
                heat_condition_vector, heat_condition_mask, heat_condition_stats = self._heat_condition_arrays(ordered_heat, ordered_present)
                kpis = compute_steady_thermal_kpis(
                    steady,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    channel_order=channel_order,
                    module_mask=module_mask,
                    module_centers=ordered_centers,
                    module_present=ordered_present,
                    heat_powers=ordered_heat,
                    module_internal_temperature=internal,
                    module_internal_mask=internal_mask,
                    interface_target=interface_target,
                    interface_condition=interface_condition,
                    domain={**domain, "module_radius": radius},
                    material_params=material,
                    temperature_limits=self.temperature_limits,
                )
                active_count = int(np.sum(ordered_present > 0.5))
                kpis["num_modules"] = active_count
                kpis["heat_power_total"] = float(np.sum(ordered_heat[ordered_present > 0.5])) if ordered_heat.size else 0.0
                records.append(
                    ThermalInverseCaseRecord(
                        case_id=case_id,
                        split=str(splits[idx]),
                        design_vec=design_vec,
                        true_count=active_count,
                        module_centers=ordered_centers,
                        module_present=ordered_present,
                        heat_powers=ordered_heat,
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
                        structure_intent_vector=structure_vector,
                        structure_intent_maps=structure_maps,
                        heat_condition_vector=heat_condition_vector,
                        heat_condition_mask=heat_condition_mask,
                        heat_condition_stats=heat_condition_stats,
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_case

    def set_kpi_stats(self, stats: Optional[Mapping[str, Any]]) -> None:
        self.kpi_stats = stats

    def set_kpi_distribution_summary(self, summary: Optional[Mapping[str, Any]]) -> None:
        self.kpi_distribution_summary = summary

    def set_latent_targets(self, latents: Mapping[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        self.latent_targets = {str(k): (np.asarray(v[0], dtype=np.float32), np.asarray(v[1], dtype=np.float32)) for k, v in latents.items()}

    def set_hypergraph_plan_targets(self, plans: Mapping[str, Tuple[np.ndarray, np.ndarray]], *, plan_dim: Optional[int] = None) -> None:
        if plan_dim is not None:
            self.hypergraph_plan_dim = max(int(plan_dim), 0)
        converted: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for key, value in plans.items():
            vec = _pad_or_trim(np.asarray(value[0], dtype=np.float32), self.hypergraph_plan_dim)
            mask = _pad_or_trim(np.asarray(value[1], dtype=np.float32), self.hypergraph_plan_dim)
            converted[str(key)] = (vec, mask)
        self.hypergraph_plan_targets = converted

    def __getitem__(self, item: int) -> Dict[str, Any]:
        item_idx = int(item)
        physical_idx = item_idx // self.samples_per_case
        sample_idx = item_idx % self.samples_per_case
        record = self.records[physical_idx]
        rng_seed = (self.seed + physical_idx * 104729 + sample_idx * 15485863) % (2**32 - 1)
        rng = np.random.default_rng(rng_seed)
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
            active_kpi_names = list(aug.get("active_kpi_names", []))
            dropped_kpi_names = list(aug.get("dropped_kpi_names", []))
            target_modes_by_kpi = dict(aug.get("target_modes_by_kpi", {}))
            active_kpi_mask = np.asarray(aug.get("active_kpi_mask", []), dtype=np.float32)
            active_kpi_count = int(aug.get("active_kpi_count", len(active_kpi_names)))
        else:
            kpi_targets = {
                name: {"mode": "exact", "value": float(record.kpi_dict[name]), "weight": 1.0}
                for name in self.kpi_names
                if name in record.kpi_dict and name not in set(record.kpi_dict.get("unavailable_kpis", []))
            }
            active_kpi_names = [str(name) for name in self.kpi_names if str(name) in kpi_targets]
            active_set = set(active_kpi_names)
            dropped_kpi_names = [str(name) for name in self.kpi_names if str(name) not in active_set]
            target_modes_by_kpi = {str(name): "exact" for name in active_kpi_names}
            active_kpi_mask = np.asarray([1.0 if str(name) in active_set else 0.0 for name in self.kpi_names], dtype=np.float32)
            active_kpi_count = int(np.sum(active_kpi_mask > 0.5))
            active_fraction = float(active_kpi_count) / max(float(len(self.kpi_names)), 1.0)
        if active_kpi_mask.size != len(self.kpi_names):
            active_set = set(active_kpi_names)
            active_kpi_mask = np.asarray([1.0 if str(name) in active_set else 0.0 for name in self.kpi_names], dtype=np.float32)
            active_kpi_count = int(np.sum(active_kpi_mask > 0.5))
            active_fraction = float(active_kpi_count) / max(float(len(self.kpi_names)), 1.0)
        dropout_p = float(self.target_augmentation.get("constraint_dropout_probability", 0.0)) if self.target_augmentation else 0.0
        use_constraints = not bool(self.target_augmentation.get("enabled", False)) or float(rng.random()) >= dropout_p
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
        if self.conditioning_mode == "design_intent":
            intent_arrays = training_intent_from_record(
                record.kpi_dict,
                true_count=record.true_count,
                domain_length_x=record.domain_length_x,
                domain_length_y=record.domain_length_y,
                max_num_modules=self.max_num_modules,
                rng=rng,
                distribution_summary=self.kpi_distribution_summary,
                augmentation_cfg=self.intent_augmentation,
            )
        else:
            intent_arrays = {
                "design_intent_vector": np.zeros((24,), dtype=np.float32),
                "objective_weight_vector": np.zeros((7,), dtype=np.float32),
                "field_intent_maps": np.zeros((5, DEFAULT_FIELD_MAP_SHAPE[1], DEFAULT_FIELD_MAP_SHAPE[0]), dtype=np.float32),
            }
        behavior, organization = self.latent_targets.get(
            record.case_id,
            (
                np.zeros((self.behavior_latent_dim,), dtype=np.float32),
                np.zeros((self.organization_latent_dim,), dtype=np.float32),
            ),
        )
        hypergraph_plan_target, hypergraph_plan_mask = self.hypergraph_plan_targets.get(
            record.case_id,
            (
                np.zeros((self.hypergraph_plan_dim,), dtype=np.float32),
                np.zeros((self.hypergraph_plan_dim,), dtype=np.float32),
            ),
        )
        structure_vector = np.asarray(record.structure_intent_vector, dtype=np.float32).copy()
        structure_maps = np.asarray(record.structure_intent_maps, dtype=np.float32).copy()
        if self.structure_enabled:
            if self.split.lower() in {"train", "training"}:
                lo, hi = self.structure_conditioning.get("strength_train_range", [0.0, 1.0])
                structure_strength = float(rng.uniform(float(lo), float(hi)))
                noise_std = float(self.structure_conditioning.get("noise_std", 0.0) or 0.0)
                if noise_std > 0.0:
                    structure_vector += rng.normal(0.0, noise_std, size=structure_vector.shape).astype(np.float32)
                drop_p = float(self.structure_conditioning.get("drop_probability", 0.0) or 0.0)
                if drop_p > 0.0 and float(rng.random()) < drop_p:
                    structure_strength = 0.0
            else:
                structure_strength = float(self.structure_conditioning.get("strength_val", 1.0))
        else:
            structure_strength = 0.0
            structure_vector[:] = 0.0
            structure_maps[:] = 0.0
        structure_vector = np.clip(structure_vector, 0.0, 1.0).astype(np.float32)
        if not bool(self.structure_conditioning.get("use_structure_maps", True)):
            structure_maps[:] = 0.0

        heat_condition_vector = np.asarray(record.heat_condition_vector, dtype=np.float32).copy()
        heat_condition_mask = np.asarray(record.heat_condition_mask, dtype=np.float32).copy()
        heat_condition_stats = np.asarray(record.heat_condition_stats, dtype=np.float32).copy()
        if self.heat_conditioning_enabled:
            drop_p = float(self.heat_conditioning.get("drop_probability", 0.0) or 0.0)
            if self.split.lower() in {"train", "training"} and drop_p > 0.0 and float(rng.random()) < drop_p:
                heat_condition_vector[:] = 0.0
                heat_condition_mask[:] = 0.0
                heat_condition_stats[:] = 0.0
        else:
            heat_condition_vector[:] = 0.0
            heat_condition_mask[:] = 0.0
            heat_condition_stats[:] = 0.0
        return {
            "case_id": record.case_id,
            "design_vec": record.design_vec.astype(np.float32),
            "target_spec_vector": np.asarray(target_spec, dtype=np.float32),
            "design_intent_vector": np.asarray(intent_arrays["design_intent_vector"], dtype=np.float32),
            "objective_weight_vector": np.asarray(intent_arrays["objective_weight_vector"], dtype=np.float32),
            "field_intent_maps": np.asarray(intent_arrays["field_intent_maps"], dtype=np.float32),
            "structure_intent_vector": structure_vector,
            "structure_intent_mask": np.asarray([structure_strength], dtype=np.float32),
            "structure_strength": np.asarray([structure_strength], dtype=np.float32),
            "structure_intent_maps": structure_maps.astype(np.float32),
            "heat_condition_vector": heat_condition_vector.astype(np.float32),
            "heat_condition_mask": heat_condition_mask.astype(np.float32),
            "heat_condition_stats": heat_condition_stats.astype(np.float32),
            "target_kpi_targets": kpi_targets,
            "kpi_vector": record.kpi_vector.astype(np.float32),
            "true_count": np.asarray([record.true_count], dtype=np.int64),
            "active_kpi_count": np.asarray([active_kpi_count], dtype=np.float32),
            "target_active_fraction": np.asarray([active_fraction], dtype=np.float32),
            "active_kpi_mask": active_kpi_mask.astype(np.float32),
            "active_kpi_names": active_kpi_names,
            "dropped_kpi_names": dropped_kpi_names,
            "target_modes_by_kpi": target_modes_by_kpi,
            "behavior_target": behavior.astype(np.float32),
            "organization_target": organization.astype(np.float32),
            "hypergraph_plan_target": hypergraph_plan_target.astype(np.float32),
            "hypergraph_plan_mask": hypergraph_plan_mask.astype(np.float32),
            "record_index": np.asarray([physical_idx], dtype=np.int64),
            "physical_case_index": np.asarray([physical_idx], dtype=np.int64),
            "augmentation_sample_index": np.asarray([sample_idx], dtype=np.int64),
        }


def collate_inverse(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"case_id": [item["case_id"] for item in batch]}
    tensor_keys = (
        "design_vec",
        "target_spec_vector",
        "design_intent_vector",
        "objective_weight_vector",
        "field_intent_maps",
        "structure_intent_vector",
        "structure_intent_mask",
        "structure_strength",
        "structure_intent_maps",
        "heat_condition_vector",
        "heat_condition_mask",
        "heat_condition_stats",
        "kpi_vector",
        "true_count",
        "active_kpi_count",
        "target_active_fraction",
        "active_kpi_mask",
        "behavior_target",
        "organization_target",
        "hypergraph_plan_target",
        "hypergraph_plan_mask",
        "record_index",
        "physical_case_index",
        "augmentation_sample_index",
    )
    int_keys = {"true_count", "record_index", "physical_case_index", "augmentation_sample_index"}
    for key in tensor_keys:
        out[key] = torch.as_tensor(np.stack([item[key] for item in batch]), dtype=torch.long if key in int_keys else torch.float32)
    out["true_count"] = out["true_count"].reshape(-1)
    out["active_kpi_count"] = out["active_kpi_count"].reshape(-1)
    out["target_active_fraction"] = out["target_active_fraction"].reshape(-1)
    out["structure_intent_mask"] = out["structure_intent_mask"].reshape(-1)
    out["structure_strength"] = out["structure_strength"].reshape(-1)
    out["record_index"] = out["record_index"].reshape(-1)
    out["physical_case_index"] = out["physical_case_index"].reshape(-1)
    out["augmentation_sample_index"] = out["augmentation_sample_index"].reshape(-1)
    for key in int_keys:
        out[key] = out[key].long()
    out["active_kpi_names"] = [item["active_kpi_names"] for item in batch]
    out["dropped_kpi_names"] = [item["dropped_kpi_names"] for item in batch]
    out["target_modes_by_kpi"] = [item["target_modes_by_kpi"] for item in batch]
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


def compute_kpi_distribution_summary(vectors: np.ndarray, kpi_names: Sequence[str]) -> Dict[str, Any]:
    arr = np.asarray(vectors, dtype=np.float64)
    quantiles = {
        "p01": 1.0,
        "p05": 5.0,
        "p10": 10.0,
        "p25": 25.0,
        "p50": 50.0,
        "p75": 75.0,
        "p90": 90.0,
        "p95": 95.0,
        "p99": 99.0,
    }
    out: Dict[str, Any] = {"names": list(kpi_names), "kpis": {}}
    for idx, name in enumerate(kpi_names):
        values = arr[:, idx] if arr.ndim == 2 and idx < arr.shape[1] else np.asarray([], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            out["kpis"][str(name)] = {"count": 0}
            continue
        entry = {
            "count": int(values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for key, q in quantiles.items():
            entry[key] = float(np.percentile(values, q))
        out["kpis"][str(name)] = entry
    return out


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


def load_first_readable_checkpoint(
    candidates: Sequence[Path],
    *,
    map_location: Any,
    label: str,
) -> Tuple[Path, Dict[str, Any]]:
    errors: List[str] = []
    for path in candidates:
        try:
            checkpoint = load_trusted_checkpoint(path, map_location=map_location)
            if errors:
                print(f"[warning] using {path.name} for {label} after skipping unreadable checkpoint(s).")
            return path, checkpoint
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            print(f"[warning] skipping unreadable {label} checkpoint {path}: {exc}")
    joined = "\n  ".join(errors)
    raise RuntimeError(f"No readable {label} checkpoint found. Tried:\n  {joined}")


def readable_forward_checkpoint_candidates(forward_cfg: Mapping[str, Any]) -> Tuple[List[Path], Path]:
    saved_root = resolve_demo_path(forward_cfg.get("saved_root", "./Saved_Model"))
    run_dir_raw = str(forward_cfg.get("run_dir", "auto"))
    run_dir = latest_run_dir(saved_root) if run_dir_raw.lower() == "auto" else resolve_demo_path(run_dir_raw)
    checkpoint_name = str(forward_cfg.get("checkpoint_name", "auto"))
    names = list(forward_cfg.get("checkpoint_preference", CHECKPOINT_PREFERENCE))
    if checkpoint_name.lower() != "auto":
        names = [checkpoint_name, *[name for name in names if name != checkpoint_name]]
    candidates = [(run_dir / str(name)).resolve() for name in names if (run_dir / str(name)).exists()]
    if not candidates:
        raise FileNotFoundError(f"No forward checkpoint found in {run_dir}; tried {names}.")
    return candidates, run_dir.resolve()


def load_forward_model(forward_cfg: Mapping[str, Any], device: torch.device) -> Tuple[GlobalChannelThermalModel, Dict[str, Any], Path]:
    checkpoint_candidates, run_dir = readable_forward_checkpoint_candidates(forward_cfg)
    checkpoint_path, checkpoint = load_first_readable_checkpoint(checkpoint_candidates, map_location=device, label="forward")
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
    incompatible = model.load_state_dict(strip_module_prefix(state), strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print(f"[forward] missing checkpoint keys ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    if unexpected:
        print(f"[forward] unexpected checkpoint keys ({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")

    def _known_harmless(key: str) -> bool:
        prefixes = (
            "local_surrogate.",
            "local_module_params_",
            "local_port_tokens_",
            "local_internal_temperature_",
            "local_interface_targets_",
            "global_internal_temperature_",
            "global_interface_target_",
        )
        return key.startswith(prefixes)

    major_missing = [key for key in missing if not _known_harmless(key)]
    major_unexpected = [key for key in unexpected if not _known_harmless(key)]
    if (major_missing or major_unexpected) and not bool(forward_cfg.get("allow_state_mismatch", False)):
        raise RuntimeError(
            "Forward checkpoint state_dict mismatch. Set forward_model.allow_state_mismatch=true only after reviewing. "
            f"major_missing={major_missing[:20]}, major_unexpected={major_unexpected[:20]}"
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "resolved_config": resolved_cfg,
        "local_surrogate_checkpoint_path": str(local_path) if local_path is not None else None,
        "local_surrogate_used": bool(model_config.use_local_surrogate),
        "state_missing_keys": missing,
        "state_unexpected_keys": unexpected,
        "resolved_config_hash": stable_json_hash(resolved_cfg),
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
    heat_load_policy: str = "preserve_total_heat",
    fixed_heat_per_module: Optional[float] = None,
    target_heat_power_total: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
    count = min(int(centers.shape[0]), int(max_num_modules))
    padded_centers = np.zeros((max_num_modules, 2), dtype=np.float32)
    present = np.zeros((max_num_modules,), dtype=np.float32)
    heat = np.zeros((max_num_modules,), dtype=np.float32)
    heat_source = "uniform"
    if count > 0:
        padded_centers[:count] = centers[:count]
        present[:count] = 1.0
        if candidate.get("heat_powers") is not None:
            candidate_heat = np.asarray(candidate["heat_powers"], dtype=np.float32).reshape(-1)
            heat[:count] = np.resize(candidate_heat, count)[:count]
            heat_source = "generated_heat_power" if generate_heat_power else "candidate_heat_condition"
        else:
            active_heat = record.heat_powers[record.module_present > 0.5]
            if active_heat.size == 0:
                active_heat = record.heat_powers
            policy = str(heat_load_policy or "preserve_total_heat").lower().strip()
            reference_total = float(np.sum(active_heat)) if active_heat.size else 0.0
            if policy == "preserve_total_heat":
                heat[:count] = reference_total / max(float(count), 1.0)
                heat_source = "preserve_total_heat"
            elif policy == "preserve_per_module_heat":
                n = min(count, int(active_heat.size))
                if n > 0:
                    heat[:n] = active_heat[:n]
                if count > n:
                    remaining = max(reference_total - float(np.sum(heat[:n])), 0.0)
                    heat[n:count] = remaining / max(float(count - n), 1.0)
                heat_source = "preserve_per_module_heat"
            elif policy == "reference_active_heat_resize":
                heat[:count] = np.resize(active_heat.astype(np.float32), count)[:count]
                heat_source = "preserve_per_module_heat"
            elif policy == "fixed_heat_per_module":
                if fixed_heat_per_module is None:
                    raise ValueError("heat_load_policy='fixed_heat_per_module' requires inverse_model.fixed_heat_per_module or target fixed_heat_per_module.")
                heat[:count] = float(fixed_heat_per_module)
                heat_source = "uniform"
            elif policy == "target_heat_power_total":
                total = reference_total if target_heat_power_total is None else float(target_heat_power_total)
                heat[:count] = total / max(float(count), 1.0)
                heat_source = "uniform"
            else:
                raise ValueError(
                    "heat_load_policy must be one of preserve_total_heat, preserve_per_module_heat, "
                    "reference_active_heat_resize, fixed_heat_per_module, target_heat_power_total."
                )
    return padded_centers, present, heat.astype(np.float32), _apply_heat_normalization(heat, metadata), heat_source


def _local_module_params_from_raw_heat(
    record: ThermalInverseCaseRecord,
    heat_raw: np.ndarray,
    present: np.ndarray,
    interface_condition: Optional[np.ndarray] = None,
) -> np.ndarray:
    params = np.zeros((heat_raw.shape[0], 7), dtype=np.float32)
    params[:, 0] = heat_raw.astype(np.float32)
    if record.material_params.shape[0] > 3:
        params[:, 1] = float(record.material_params[3])
    if record.material_params.shape[0] > 1:
        params[:, 2] = float(record.material_params[1])
    if interface_condition is not None:
        cond = np.asarray(interface_condition, dtype=np.float32)
        if cond.ndim >= 3 and cond.shape[0] == heat_raw.shape[0] and cond.shape[-1] >= 7:
            h_index = 7 if cond.shape[-1] >= 8 else 6
            t_out = cond[..., 3]
            h_local = cond[..., h_index]
            params[:, 3] = np.mean(h_local, axis=-1)
            params[:, 4] = np.std(h_local, axis=-1)
            params[:, 5] = np.mean(t_out, axis=-1)
            params[:, 6] = np.std(t_out, axis=-1)
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
    heat_load_policy: str = "preserve_total_heat",
    fixed_heat_per_module: Optional[float] = None,
    target_heat_power_total: Optional[float] = None,
    query_batch_size: int = 32768,
    use_record_interface_condition: bool = False,
) -> Dict[str, Any]:
    centers, present, heat_raw, heat, heat_source = _padded_design_arrays(
        record,
        candidate,
        max_num_modules,
        metadata,
        generate_heat_power=generate_heat_power,
        heat_load_policy=heat_load_policy,
        fixed_heat_per_module=fixed_heat_per_module,
        target_heat_power_total=target_heat_power_total,
    )
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
    record_interface_condition = getattr(record, "interface_condition", None) if bool(use_record_interface_condition) else None
    local_module_params = torch.from_numpy(_local_module_params_from_raw_heat(record, heat_raw, present, record_interface_condition)[None]).to(device)
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
        "heat_load_policy": str(heat_load_policy),
        "heat_source": heat_source,
        "verification_mode": "predicted",
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
    packed_h5_path: str | Path,
    kpi_names: Sequence[str],
    inverse_cache_config: Mapping[str, Any],
    behavior_dim: int,
    organization_dim: int,
    max_num_modules: int,
    generate_heat_power: bool,
    hypergraph_plan_dim_value: int = 0,
    hypergraph_plan_num_edges: int = 0,
    heat_load_policy: str = "preserve_total_heat",
    fixed_heat_per_module: Optional[float] = None,
    target_heat_power_total: Optional[float] = None,
    force_rebuild: bool = False,
    disable_auto_rebuild: bool = False,
    query_batch_size: int = 32768,
) -> Dict[str, Any]:
    record_case_ids = [str(record.case_id) for record in records]
    expected_meta = {
        "cache_version": 5,
        "packed_h5": file_fingerprint(packed_h5_path),
        "forward_checkpoint": file_fingerprint(str(metadata.get("checkpoint_path", ""))),
        "resolved_forward_config_hash": metadata.get("resolved_config_hash"),
        "inverse_config_relevant": dict(inverse_cache_config),
        "inverse_config_hash": stable_json_hash(inverse_cache_config),
        "kpi_names": list(kpi_names),
        "behavior_latent_dim": int(behavior_dim),
        "organization_latent_dim": int(organization_dim),
        "hypergraph_plan_dim": int(hypergraph_plan_dim_value),
        "hypergraph_plan_num_edges": int(hypergraph_plan_num_edges),
        "record_count": int(len(record_case_ids)),
        "record_case_ids_hash": stable_json_hash(record_case_ids),
    }

    def _cache_matches(payload: Mapping[str, Any]) -> bool:
        meta = payload.get("cache_metadata", {})
        return isinstance(meta, Mapping) and meta == expected_meta

    if cache_path.exists() and not force_rebuild:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            isinstance(payload, Mapping)
            and "latents" in payload
            and _cache_matches(payload)
            and int(payload.get("behavior_dim", -1)) == int(behavior_dim)
            and int(payload.get("organization_dim", -1)) == int(organization_dim)
        ):
            print(f"[latent-cache] loaded {cache_path}")
            latents = {
                str(k): (np.asarray(v["behavior"], dtype=np.float32), np.asarray(v["organization"], dtype=np.float32))
                for k, v in payload["latents"].items()
            }
            plans = {
                str(k): (np.asarray(v.get("hypergraph_plan_target", []), dtype=np.float32), np.asarray(v.get("hypergraph_plan_mask", []), dtype=np.float32))
                for k, v in payload["latents"].items()
            }
            return {"latents": latents, "hypergraph_plans": plans, "hypergraph_plan_metadata": payload.get("hypergraph_plan_metadata", {})}
        message = f"[latent-cache] incompatible cache metadata for {cache_path}"
        if disable_auto_rebuild:
            raise RuntimeError(message + "; set forward_model.disable_auto_cache_rebuild=false or rebuild_latent_cache=true.")
        print(message + "; rebuilding")
    latents: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    hypergraph_plans: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    hypergraph_plan_metadata: Dict[str, Any] = {
        "hypergraph_plan_num_edges": int(hypergraph_plan_num_edges),
        "hypergraph_plan_fields": list(HYPERGRAPH_PLAN_FIELDS),
        "hypergraph_plan_dim": int(hypergraph_plan_dim_value),
        "hypergraph_plan_edge_ordering": HYPERGRAPH_PLAN_EDGE_ORDERING,
    }
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
            heat_load_policy=heat_load_policy,
            fixed_heat_per_module=fixed_heat_per_module,
            target_heat_power_total=target_heat_power_total,
            query_batch_size=query_batch_size,
        )
        latents[record.case_id] = extract_forward_latent_targets(prediction, behavior_dim=behavior_dim, organization_dim=organization_dim)
        plan = build_hypergraph_plan_from_forward_prediction(
            prediction,
            max_num_modules=max_num_modules,
            domain_length_x=record.domain_length_x,
            domain_length_y=record.domain_length_y,
            num_edges=int(hypergraph_plan_num_edges) if int(hypergraph_plan_num_edges) > 0 else None,
        )
        vec = _pad_or_trim(np.asarray(plan["vector"], dtype=np.float32), int(hypergraph_plan_dim_value)) if int(hypergraph_plan_dim_value) > 0 else np.asarray(plan["vector"], dtype=np.float32)
        mask = _pad_or_trim(np.asarray(plan["mask"], dtype=np.float32), int(hypergraph_plan_dim_value)) if int(hypergraph_plan_dim_value) > 0 else np.asarray(plan["mask"], dtype=np.float32)
        hypergraph_plans[record.case_id] = (vec, mask)
        if plan.get("metadata"):
            case_summaries = hypergraph_plan_metadata.get("case_summaries", {})
            hypergraph_plan_metadata.update(dict(plan["metadata"]))
            hypergraph_plan_metadata["case_summaries"] = case_summaries
        hypergraph_plan_metadata.setdefault("case_summaries", {})[record.case_id] = {
            "active_edge_count": plan.get("summary", {}).get("active_edge_count") if isinstance(plan.get("summary"), Mapping) else None,
        }
    ensure_dir(cache_path.parent)
    torch.save(
        {
            "latents": {
                key: {
                    "behavior": value[0],
                    "organization": value[1],
                    "hypergraph_plan_target": hypergraph_plans.get(key, (np.zeros((int(hypergraph_plan_dim_value),), dtype=np.float32), np.zeros((int(hypergraph_plan_dim_value),), dtype=np.float32)))[0],
                    "hypergraph_plan_mask": hypergraph_plans.get(key, (np.zeros((int(hypergraph_plan_dim_value),), dtype=np.float32), np.zeros((int(hypergraph_plan_dim_value),), dtype=np.float32)))[1],
                    "hypergraph_plan_summary": hypergraph_plan_metadata.get("case_summaries", {}).get(key, {}),
                }
                for key, value in latents.items()
            },
            "behavior_dim": behavior_dim,
            "organization_dim": organization_dim,
            "hypergraph_plan_dim": int(hypergraph_plan_dim_value),
            "hypergraph_plan_metadata": hypergraph_plan_metadata,
            "cache_version": 5,
            "cache_metadata": expected_meta,
            "forward_checkpoint": metadata.get("checkpoint_path"),
        },
        cache_path,
    )
    print(f"[latent-cache] wrote {cache_path}")
    return {"latents": latents, "hypergraph_plans": hypergraph_plans, "hypergraph_plan_metadata": hypergraph_plan_metadata}


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
    out: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([row.get(key, float("nan")) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        out[key] = float(np.mean(finite)) if finite.size else float("nan")
    return out


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
    kpi_names = [str(name) for name in getattr(loader.dataset, "kpi_names", [])]
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
        if "target_active_fraction" in batch:
            row["target_active_fraction_mean"] = float(batch["target_active_fraction"].detach().float().mean().cpu())
        if "active_kpi_count" in batch:
            row["active_kpi_count_mean"] = float(batch["active_kpi_count"].detach().float().mean().cpu())
        if "active_kpi_mask" in batch:
            mask = batch["active_kpi_mask"].detach().float().cpu()
            if mask.ndim == 2:
                if not kpi_names:
                    kpi_names = [f"kpi_{idx}" for idx in range(mask.shape[1])]
                for idx, name in enumerate(kpi_names[: mask.shape[1]]):
                    row[f"kpi_active_frequency_{name}"] = float(mask[:, idx].mean())
        mode_counts = {"exact": 0, "range": 0, "max": 0, "min": 0}
        mode_total = 0
        for modes_by_kpi in batch.get("target_modes_by_kpi", []) or []:
            if not isinstance(modes_by_kpi, Mapping):
                continue
            for mode in modes_by_kpi.values():
                normalized = str(mode).lower().strip()
                if normalized in {"upper", "at_most"}:
                    normalized = "max"
                elif normalized in {"lower", "at_least"}:
                    normalized = "min"
                if normalized in mode_counts:
                    mode_counts[normalized] += 1
                    mode_total += 1
        denom = max(float(mode_total), 1.0)
        for mode_name, count in mode_counts.items():
            row[f"target_mode_frequency_{mode_name}"] = float(count) / denom
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


def _candidate_layout_metrics(candidate: Mapping[str, Any]) -> Dict[str, float]:
    return layout_spread_metrics(candidate.get("centers", []), num_modules=int(candidate.get("count", 0)))


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
    count_mode: str = "uniform",
    heat_load_policy: str = "preserve_total_heat",
    fixed_heat_per_module: Optional[float] = None,
    target_heat_power_total: Optional[float] = None,
    temperature_limits: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    if num_targets <= 0 or num_samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(int(num_targets), len(dataset)), replace=False)
    scores = []
    valid = []
    diversity_raw = []
    diversity_top = []
    bbox_areas = []
    pair_distances = []
    count_hist: Dict[int, int] = {}
    port_kpis_available = []
    port_condition_predicted = []
    kpi_errors: Dict[str, List[float]] = {}
    for idx in indices:
        item = dataset[int(idx)]
        physical_idx = int(np.asarray(item.get("physical_case_index", [int(idx) // dataset.samples_per_case])).reshape(-1)[0])
        record = dataset.records[physical_idx]
        item_case_id = str(item.get("case_id", ""))
        if item_case_id != record.case_id:
            raise AssertionError(
                f"Forward verification index mismatch: dataset[{int(idx)}] case_id={item_case_id!r}, "
                f"records[{physical_idx}] case_id={record.case_id!r}."
            )
        target_spec_vec = item["target_spec_vector"]
        candidates = inverse_model.sample_designs(
            target_spec_vec,
            n_samples=int(num_samples),
            n_steps=int(n_steps),
            seed=int(seed + idx),
            count_mode=str(count_mode),
            design_intent_vector=item.get("design_intent_vector"),
            objective_weight_vector=item.get("objective_weight_vector"),
            field_intent_maps=item.get("field_intent_maps"),
            structure_intent_vector=item.get("structure_intent_vector"),
            structure_intent_maps=item.get("structure_intent_maps"),
            heat_condition_vector=item.get("heat_condition_vector"),
            heat_condition_mask=item.get("heat_condition_mask"),
            device=device,
        )
        diversity_raw.append(sample_diversity(candidates, inverse_model.max_num_modules))
        best_score = float("inf")
        best_valid = 0.0
        scored_candidates: List[Tuple[float, Mapping[str, Any]]] = []
        for cand in candidates:
            count_hist[int(cand.get("count", 0))] = count_hist.get(int(cand.get("count", 0)), 0) + 1
            spread = _candidate_layout_metrics(cand)
            bbox_areas.append(float(spread["bbox_area"]))
            pair_distances.append(float(spread["mean_pair_distance"]))
            prediction = predict_candidate_with_forward(
                forward_model,
                forward_metadata,
                record,
                cand,
                device,
                max_num_modules=inverse_model.max_num_modules,
                generate_heat_power=inverse_model.cfg.generate_heat_power,
                heat_load_policy=heat_load_policy,
                fixed_heat_per_module=fixed_heat_per_module,
                target_heat_power_total=target_heat_power_total,
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
                interface_condition=prediction.get("pred_port_condition"),
                domain={"domain_length_x": record.domain_length_x, "domain_length_y": record.domain_length_y, "module_radius": record.module_radius},
                material_params=record.material_params,
                temperature_limits=temperature_limits,
            )
            kpis.update(cand.get("validity", {}))
            kpis.update(spread)
            port_kpis_available.append(1.0 if "mean_interface_T_env" in kpis.get("available_kpis", []) else 0.0)
            port_condition_predicted.append(1.0 if prediction.get("pred_port_condition") is not None else 0.0)
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
            scored_candidates.append((float(scored["total_score"]), cand))
            if scored["total_score"] < best_score:
                best_score = float(scored["total_score"])
                best_valid = 1.0 if bool(cand.get("validity", {}).get("valid", False)) else 0.0
                for name, err in scored.get("per_kpi_errors", {}).items():
                    kpi_errors.setdefault(name, []).append(float(err))
        scored_candidates.sort(key=lambda item_score: item_score[0])
        diversity_top.append(sample_diversity([cand for _, cand in scored_candidates[: min(4, len(scored_candidates))]], inverse_model.max_num_modules))
        scores.append(best_score)
        valid.append(best_valid)
    metrics = {
        "val_forward_score": float(np.mean(scores)) if scores else float("nan"),
        "val_forward_score_reconstruct": float(np.mean(scores)) if scores else float("nan"),
        "val_design_intent_score_reconstruct": float(np.mean(scores)) if scores else float("nan"),
        "val_forward_validity_rate": float(np.mean(valid)) if valid else float("nan"),
        "val_candidate_diversity": float(np.mean(diversity_raw)) if diversity_raw else float("nan"),
        "val_candidate_diversity_raw": float(np.mean(diversity_raw)) if diversity_raw else float("nan"),
        "val_candidate_diversity_top": float(np.mean(diversity_top)) if diversity_top else float("nan"),
        "val_count_histogram": json.dumps({str(k): int(v) for k, v in sorted(count_hist.items())}, sort_keys=True),
        "val_mean_bbox_area": float(np.mean(bbox_areas)) if bbox_areas else float("nan"),
        "val_mean_pair_distance": float(np.mean(pair_distances)) if pair_distances else float("nan"),
        "verification_mode": "predicted",
        "predicted_port_condition_available": float(np.mean(port_condition_predicted)) if port_condition_predicted else 0.0,
        "predicted_port_condition_kpis_available": float(np.mean(port_kpis_available)) if port_kpis_available else 0.0,
    }
    for name, values in kpi_errors.items():
        metrics[f"val_kpi_error_{name}"] = float(np.mean(values))
    return metrics


def _safe_metric_name(value: Any) -> str:
    raw = str(value or "target").strip()
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip("_") or "target"


def _span_bounds(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    lo, hi = sorted((float(arr[0]), float(arr[1])))
    return lo, hi


def _target_spec_from_json_for_validation(
    path_like: str | Path,
    *,
    kpi_names: Sequence[str],
    kpi_stats: Mapping[str, Any],
    normalize: bool,
    model: ThermalInverseDesignFlow,
) -> Dict[str, Any]:
    path = resolve_demo_path(path_like)
    payload = read_json(path)
    intent_arrays = build_design_intent_arrays(
        payload,
        max_num_modules=model.max_num_modules,
        domain_length_x=float(model.cfg.domain_length_x),
        domain_length_y=float(model.cfg.domain_length_y),
        heat_power_scale=float(model.cfg.heat_power_scale),
    )
    normalized_intent = intent_arrays["intent"]
    prefs = payload.get("preferences", {}) if isinstance(payload.get("preferences", {}), Mapping) else {}
    kpis = dict(payload.get("kpis", {}))
    if bool(prefs.get("avoid_wall_hotspots", False)) and "wall_hot_area_fraction" not in kpis:
        kpis["wall_hot_area_fraction"] = {"mode": "max", "high": 0.08, "weight": 0.5}
    scenario = normalized_intent.get("scenario", {}) if isinstance(normalized_intent.get("scenario"), Mapping) else {}
    geometry = normalized_intent.get("geometry_constraints", {}) if isinstance(normalized_intent.get("geometry_constraints"), Mapping) else {}
    constraints = {
        "num_modules_min": scenario.get("num_modules_min", payload.get("num_modules_min", payload.get("num_cylinders_min"))),
        "num_modules_max": scenario.get("num_modules_max", payload.get("num_modules_max", payload.get("num_cylinders_max"))),
        "min_center_distance": geometry.get("min_center_distance", payload.get("min_center_distance", prefs.get("min_center_distance"))),
        "wall_clearance": geometry.get("wall_clearance", payload.get("wall_clearance", prefs.get("wall_clearance"))),
        "inlet_clearance": geometry.get("inlet_clearance", payload.get("inlet_clearance", prefs.get("inlet_clearance"))),
        "outlet_clearance": geometry.get("outlet_clearance", payload.get("outlet_clearance", prefs.get("outlet_clearance"))),
        "heat_power_total": scenario.get("heat_power_total", payload.get("heat_power_total")),
    }
    vector = build_target_spec_vector(
        kpi_targets=kpis,
        kpi_names=kpi_names,
        stats=kpi_stats,
        normalize=normalize,
        num_modules_min=constraints["num_modules_min"],
        num_modules_max=constraints["num_modules_max"],
        min_center_distance=constraints["min_center_distance"],
        wall_clearance=constraints["wall_clearance"],
        inlet_clearance=constraints["inlet_clearance"],
        outlet_clearance=constraints["outlet_clearance"],
        heat_power_total=constraints["heat_power_total"],
        max_num_modules=model.max_num_modules,
        domain_length_scale=max(float(model.cfg.domain_length_x), float(model.cfg.domain_length_y)),
        heat_power_scale=float(model.cfg.heat_power_scale),
    )
    thermal_limits_payload = payload.get("temperature_limits")
    if thermal_limits_payload is None and isinstance(normalized_intent.get("thermal_limits"), Mapping):
        thermal_limits_payload = {
            "wall_hot_delta_T": normalized_intent["thermal_limits"].get("wall_hot_delta_T"),
            "outlet_hot_delta_T": normalized_intent["thermal_limits"].get("outlet_hot_delta_T"),
        }
    return {
        "name": payload.get("name", path.stem),
        "vector": vector,
        "kpi_targets": kpis,
        "kpi_names": list(kpi_names),
        "kpi_stats": kpi_stats,
        "constraints": constraints,
        "preferences": prefs,
        "temperature_limits": thermal_limits_payload,
        "target_payload": payload,
        "design_intent": normalized_intent,
        "is_design_intent": bool(is_design_intent_payload(payload)),
        "design_intent_vector": intent_arrays["design_intent_vector"],
        "objective_weight_vector": intent_arrays["objective_weight_vector"],
        "field_intent_maps": intent_arrays["field_intent_maps"],
    }


def run_fixed_target_verification(
    inverse_model: ThermalInverseDesignFlow,
    forward_model: GlobalChannelThermalModel,
    forward_metadata: Mapping[str, Any],
    dataset: ThermalInverseDesignDataset,
    target_spec: Mapping[str, Any],
    device: torch.device,
    *,
    num_samples: int,
    n_steps: int,
    query_batch_size: int,
    seed: int,
    count_mode: str,
    heat_load_policy: str,
    fixed_heat_per_module: Optional[float],
    temperature_limits: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    record = dataset.records[0]
    prefs = target_spec.get("preferences", {}) if isinstance(target_spec.get("preferences"), Mapping) else {}
    candidates = inverse_model.sample_designs(
        target_spec["vector"],
        n_samples=num_samples,
        n_steps=n_steps,
        seed=seed,
        count_mode=str(count_mode),
        x_bounds=_span_bounds(prefs.get("x_span")),
        y_bounds=_span_bounds(prefs.get("y_span")),
        design_intent_vector=target_spec.get("design_intent_vector"),
        objective_weight_vector=target_spec.get("objective_weight_vector"),
        field_intent_maps=target_spec.get("field_intent_maps"),
        structure_intent_vector=target_spec.get("structure_intent_vector"),
        structure_intent_maps=target_spec.get("structure_intent_maps"),
        heat_condition_vector=target_spec.get("heat_condition_vector"),
        heat_condition_mask=target_spec.get("heat_condition_mask"),
        device=device,
    )
    scores = []
    intent_scores = []
    violations = []
    valid = []
    spread_penalties = []
    port_available = []
    for cand in candidates:
        spread = _candidate_layout_metrics(cand)
        prediction = predict_candidate_with_forward(
            forward_model,
            forward_metadata,
            record,
            cand,
            device,
            max_num_modules=inverse_model.max_num_modules,
            generate_heat_power=inverse_model.cfg.generate_heat_power,
            heat_load_policy=heat_load_policy,
            fixed_heat_per_module=fixed_heat_per_module,
            target_heat_power_total=target_spec.get("constraints", {}).get("heat_power_total") if isinstance(target_spec.get("constraints"), Mapping) else None,
            query_batch_size=query_batch_size,
        )
        limits = target_spec.get("temperature_limits") if isinstance(target_spec.get("temperature_limits"), Mapping) else temperature_limits
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
            interface_condition=prediction.get("pred_port_condition"),
            domain={"domain_length_x": record.domain_length_x, "domain_length_y": record.domain_length_y, "module_radius": record.module_radius},
            material_params=record.material_params,
            temperature_limits=limits,
        )
        kpis.update(cand.get("validity", {}))
        kpis.update(spread)
        port_available.append(1.0 if "mean_interface_T_env" in kpis.get("available_kpis", []) else 0.0)
        kpis["num_modules"] = int(cand.get("count", 0))
        scored = score_candidate_kpis(kpis, target_spec)
        intent_scored = compute_design_intent_score(kpis, {"centers": cand.get("centers", [])}, target_spec)
        scores.append(float(scored["total_score"]))
        intent_scores.append(float(intent_scored["design_intent_score"]))
        violations.append(float(scored["kpi_violation"]))
        spread_penalties.append(float(scored.get("spread_preference_penalty", 0.0)))
        valid.append(1.0 if bool(cand.get("validity", {}).get("valid", False)) else 0.0)
    name = _safe_metric_name(target_spec.get("name"))
    return {
        f"val_forward_score_{name}": float(np.min(scores)) if scores else float("nan"),
        f"val_design_intent_score_{name}": float(np.min(intent_scores)) if intent_scores else float("nan"),
        f"validity_rate_{name}": float(np.mean(valid)) if valid else float("nan"),
        f"best_kpi_violation_{name}": float(np.min(violations)) if violations else float("nan"),
        f"best_spread_preference_penalty_{name}": float(np.min(spread_penalties)) if spread_penalties else float("nan"),
        f"predicted_port_condition_kpis_available_{name}": float(np.mean(port_available)) if port_available else 0.0,
    }


def run_forward_guidance_replay(
    inverse_model: ThermalInverseDesignFlow,
    forward_model: GlobalChannelThermalModel,
    forward_metadata: Mapping[str, Any],
    dataset: ThermalInverseDesignDataset,
    device: torch.device,
    *,
    num_intents: int,
    num_candidates_per_intent: int,
    replay_top_k: int,
    n_steps: int,
    seed: int,
    count_mode: str,
    heat_load_policy: str,
    fixed_heat_per_module: Optional[float],
    query_batch_size: int,
) -> List[Dict[str, Any]]:
    if num_intents <= 0 or num_candidates_per_intent <= 0:
        return []
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(int(num_intents), len(dataset)), replace=False)
    replay: List[Dict[str, Any]] = []
    for idx in indices:
        item = dataset[int(idx)]
        physical_idx = int(np.asarray(item.get("physical_case_index", [int(idx) // dataset.samples_per_case])).reshape(-1)[0])
        record = dataset.records[physical_idx]
        candidates = inverse_model.sample_designs(
            item["target_spec_vector"],
            n_samples=int(num_candidates_per_intent),
            n_steps=int(n_steps),
            seed=int(seed + idx),
            count_mode=count_mode,
            design_intent_vector=item.get("design_intent_vector"),
            objective_weight_vector=item.get("objective_weight_vector"),
            field_intent_maps=item.get("field_intent_maps"),
            structure_intent_vector=item.get("structure_intent_vector"),
            structure_intent_maps=item.get("structure_intent_maps"),
            heat_condition_vector=item.get("heat_condition_vector"),
            heat_condition_mask=item.get("heat_condition_mask"),
            device=device,
        )
        scored_rows = []
        for cand in candidates:
            prediction = predict_candidate_with_forward(
                forward_model,
                forward_metadata,
                record,
                cand,
                device,
                max_num_modules=inverse_model.max_num_modules,
                generate_heat_power=inverse_model.cfg.generate_heat_power,
                heat_load_policy=heat_load_policy,
                fixed_heat_per_module=fixed_heat_per_module,
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
                interface_condition=prediction.get("pred_port_condition"),
                domain={"domain_length_x": record.domain_length_x, "domain_length_y": record.domain_length_y, "module_radius": record.module_radius},
                material_params=record.material_params,
                temperature_limits=dataset.temperature_limits,
            )
            kpis.update(cand.get("validity", {}))
            kpis.update(_candidate_layout_metrics(cand))
            kpis["num_modules"] = int(cand.get("count", 0))
            target_spec = {"kpi_names": list(dataset.kpi_names), "kpi_targets": item.get("target_kpi_targets", {}), "kpi_stats": dataset.kpi_stats}
            score = score_candidate_kpis(kpis, target_spec)
            scored_rows.append((float(score["total_score"]), cand))
        scored_rows.sort(key=lambda row: row[0])
        for _, cand in scored_rows[: max(int(replay_top_k), 0)]:
            replay_item = dict(item)
            centers = np.asarray(cand.get("centers", []), dtype=np.float32).reshape(-1, 2)
            present = np.zeros((inverse_model.max_num_modules,), dtype=np.float32)
            present[: min(centers.shape[0], inverse_model.max_num_modules)] = 1.0
            design_vec, _ = encode_design_vector(
                centers,
                present[: centers.shape[0]] if centers.shape[0] else present,
                None,
                max_num_modules=inverse_model.max_num_modules,
                domain_length_x=record.domain_length_x,
                domain_length_y=record.domain_length_y,
                generate_heat_power=inverse_model.cfg.generate_heat_power,
                heat_power_scale=float(inverse_model.cfg.heat_power_scale),
                sort_centers=True,
            )
            replay_item["design_vec"] = design_vec.astype(np.float32)
            replay_item["true_count"] = np.asarray([int(cand.get("count", centers.shape[0]))], dtype=np.int64)
            replay.append(replay_item)
    return replay


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
    for key in ("train_loss_total", "val_loss_total"):
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


def save_forward_score_curve(history_path: Path, out_path: Path) -> None:
    if not history_path.exists():
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with history_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    epochs = [float(row["epoch"]) for row in rows if row.get("epoch")]
    keys = sorted(
        key
        for key in rows[0].keys()
        if key == "val_forward_score" or key.startswith("val_forward_score_")
    )
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    plotted = False
    for key in keys:
        values = []
        for row in rows:
            try:
                values.append(float(row.get(key, "nan")))
            except ValueError:
                values.append(float("nan"))
        if values and any(math.isfinite(v) for v in values):
            ax.plot(epochs[: len(values)], values, marker="o", markersize=3, label=key)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("forward score")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
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
    kpi_distribution_summary: Optional[Mapping[str, Any]] = None,
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
            "kpi_distribution_summary": dict(kpi_distribution_summary or {}),
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
    conditioning_cfg = cfg.get("conditioning", {}) if isinstance(cfg.get("conditioning", {}), Mapping) else {}
    intent_aug_cfg = cfg.get("intent_augmentation", {}) if isinstance(cfg.get("intent_augmentation", {}), Mapping) else {}
    conditioning_dropout_cfg = cfg.get("conditioning_dropout", {}) if isinstance(cfg.get("conditioning_dropout", {}), Mapping) else {}
    structure_conditioning_cfg = cfg.get("structure_conditioning", {}) if isinstance(cfg.get("structure_conditioning", {}), Mapping) else {}
    heat_conditioning_cfg = cfg.get("heat_conditioning", {}) if isinstance(cfg.get("heat_conditioning", {}), Mapping) else {}
    forward_guidance_cfg = cfg.get("forward_guidance", {}) if isinstance(cfg.get("forward_guidance", {}), Mapping) else {}
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
    if bool(heat_conditioning_cfg.get("enabled", False)):
        inverse_cfg.setdefault("generate_heat_power", False)
    if "inverse_mode" not in inverse_cfg and bool(inverse_cfg.get("use_hypergraph_plan", False)):
        inverse_cfg["inverse_mode"] = "layout_flow_with_hypergraph_plan"
    inverse_cfg["inverse_mode"] = str(inverse_cfg.get("inverse_mode", "layout_flow") or "layout_flow").lower().strip()
    inverse_cfg["use_hypergraph_plan"] = inverse_cfg["inverse_mode"] == "layout_flow_with_hypergraph_plan"
    raw_plan_dim = inverse_cfg.get("hypergraph_plan_dim")
    preliminary_plan_dim = int(raw_plan_dim) if raw_plan_dim is not None and str(raw_plan_dim).lower() != "auto" else 0
    generate_heat_power = bool(inverse_cfg.get("generate_heat_power", False))
    heat_load_policy = str(inverse_cfg.get("heat_load_policy", "preserve_total_heat")).lower().strip()
    conditioning_mode = str(conditioning_cfg.get("mode", inverse_cfg.get("conditioning_mode", "legacy_kpi"))).lower().strip()
    inverse_cfg["conditioning_mode"] = conditioning_mode
    inverse_cfg["use_legacy_kpi_vector"] = bool(conditioning_cfg.get("use_legacy_kpi_vector", inverse_cfg.get("use_legacy_kpi_vector", True)))
    inverse_cfg["conditioning_dropout_enabled"] = bool(conditioning_dropout_cfg.get("enabled", False))
    inverse_cfg["conditioning_drop_probability"] = float(conditioning_dropout_cfg.get("drop_probability", 0.0))
    inverse_cfg["structure_conditioning_enabled"] = bool(structure_conditioning_cfg.get("enabled", False))
    inverse_cfg["structure_intent_dim"] = STRUCTURE_INTENT_DIM
    inverse_cfg["structure_intent_map_channels"] = 5 if bool(structure_conditioning_cfg.get("enabled", False)) and bool(structure_conditioning_cfg.get("use_structure_maps", True)) else 0
    inverse_cfg["structure_drop_probability"] = float(structure_conditioning_cfg.get("drop_probability", 0.0))
    inverse_cfg["heat_conditioning_enabled"] = bool(heat_conditioning_cfg.get("enabled", False))
    inverse_cfg["heat_condition_dim"] = int(inverse_cfg["max_num_modules"]) * 2 + 7
    inverse_cfg["heat_drop_probability"] = float(heat_conditioning_cfg.get("drop_probability", 0.0))
    fixed_heat_per_module = inverse_cfg.get("fixed_heat_per_module")
    heat_scale = float(inverse_cfg.get("heat_power_scale", target_cfg.get("heat_power_scale", 1.0)))
    temperature_limits = dict(target_cfg.get("temperature_limits", {}) or {})
    train_aug_cfg = dict(cfg.get("target_augmentation_train", cfg.get("target_augmentation", {})) or {})
    val_aug_cfg = dict(cfg.get("target_augmentation_val", {"enabled": False}) or {})
    train_samples_per_case = max(int(dataset_cfg.get("samples_per_case", 1) or 1), 1)
    val_samples_per_case = max(int(dataset_cfg.get("validation_samples_per_case", 1) or 1), 1)

    train_dataset = ThermalInverseDesignDataset(
        packed_path,
        split=dataset_cfg.get("train_split", "train"),
        kpi_names=kpi_names,
        normalize_targets=bool(target_cfg.get("normalize", True)),
        target_augmentation=train_aug_cfg,
        temperature_limits=temperature_limits,
        max_num_modules=int(inverse_cfg["max_num_modules"]),
        generate_heat_power=generate_heat_power,
        heat_power_scale=heat_scale,
        sort_centers=bool(dataset_cfg.get("sort_centers", True)),
        max_cases=max_train_cases,
        use_all_if_split_missing=bool(dataset_cfg.get("use_all_if_split_missing", True)),
        samples_per_case=train_samples_per_case,
        seed=int(training_cfg.get("seed", 42)),
        behavior_latent_dim=int(inverse_cfg.get("behavior_latent_dim", 96)),
        organization_latent_dim=int(inverse_cfg.get("organization_latent_dim", 256)),
        conditioning_mode=conditioning_mode,
        intent_augmentation=intent_aug_cfg,
        structure_conditioning=structure_conditioning_cfg,
        heat_conditioning=heat_conditioning_cfg,
        hypergraph_plan_dim=preliminary_plan_dim,
    )
    val_dataset = ThermalInverseDesignDataset(
        packed_path,
        split=dataset_cfg.get("val_split", "test"),
        kpi_names=kpi_names,
        normalize_targets=bool(target_cfg.get("normalize", True)),
        target_augmentation=val_aug_cfg,
        temperature_limits=temperature_limits,
        max_num_modules=int(inverse_cfg["max_num_modules"]),
        generate_heat_power=generate_heat_power,
        heat_power_scale=heat_scale,
        sort_centers=bool(dataset_cfg.get("sort_centers", True)),
        max_cases=max_val_cases,
        use_all_if_split_missing=bool(dataset_cfg.get("use_all_if_split_missing", True)),
        samples_per_case=val_samples_per_case,
        seed=int(training_cfg.get("seed", 42)) + 1000,
        behavior_latent_dim=int(inverse_cfg.get("behavior_latent_dim", 96)),
        organization_latent_dim=int(inverse_cfg.get("organization_latent_dim", 256)),
        conditioning_mode=conditioning_mode,
        intent_augmentation={**intent_aug_cfg, "field_preference_dropout": 1.0} if conditioning_mode == "design_intent" else {},
        structure_conditioning=structure_conditioning_cfg,
        heat_conditioning=heat_conditioning_cfg,
        hypergraph_plan_dim=preliminary_plan_dim,
    )
    kpi_stats = compute_kpi_stats(np.stack([record.kpi_vector for record in train_dataset.records]), kpi_names)
    kpi_distribution_summary = compute_kpi_distribution_summary(np.stack([record.kpi_vector for record in train_dataset.records]), kpi_names)
    train_dataset.set_kpi_stats(kpi_stats)
    val_dataset.set_kpi_stats(kpi_stats)
    train_dataset.set_kpi_distribution_summary(kpi_distribution_summary)
    val_dataset.set_kpi_distribution_summary(kpi_distribution_summary)

    sample_record = train_dataset.records[0]
    inverse_cfg["target_dim"] = len(kpi_names) * 7 + 8 if str(inverse_cfg.get("target_dim", "auto")).lower() == "auto" else inverse_cfg.get("target_dim")
    inverse_cfg["domain_length_x"] = float(inverse_cfg.get("domain_length_x", "auto") if str(inverse_cfg.get("domain_length_x", "auto")).lower() != "auto" else sample_record.domain_length_x)
    inverse_cfg["domain_length_y"] = float(inverse_cfg.get("domain_length_y", "auto") if str(inverse_cfg.get("domain_length_y", "auto")).lower() != "auto" else sample_record.domain_length_y)
    inverse_cfg["module_radius"] = float(inverse_cfg.get("module_radius", "auto") if str(inverse_cfg.get("module_radius", "auto")).lower() != "auto" else sample_record.module_radius)
    inverse_cfg["heat_power_scale"] = heat_scale

    forward_model = None
    forward_metadata: Dict[str, Any] = {}
    if bool(cfg.get("forward_model", {}).get("enabled", True)):
        forward_model, forward_metadata, _ = load_forward_model(cfg.get("forward_model", {}), device)
    if bool(inverse_cfg.get("use_hypergraph_plan", False)):
        raw_plan_dim = inverse_cfg.get("hypergraph_plan_dim")
        if raw_plan_dim is None or str(raw_plan_dim).lower() == "auto":
            num_edges = int(getattr(getattr(forward_model, "config", None), "num_hyperedges", inverse_cfg.get("hypergraph_plan_num_edges", 6)) or 6)
            inverse_cfg["hypergraph_plan_num_edges"] = num_edges
            inverse_cfg["hypergraph_plan_dim"] = hypergraph_plan_dim(int(inverse_cfg["max_num_modules"]), num_edges)
        else:
            inverse_cfg["hypergraph_plan_dim"] = int(raw_plan_dim)
            inverse_cfg["hypergraph_plan_num_edges"] = infer_hypergraph_plan_num_edges(int(raw_plan_dim), int(inverse_cfg["max_num_modules"]))
    model_config = InverseModelConfig.from_dict(inverse_cfg)
    train_dataset.hypergraph_plan_dim = int(model_config.hypergraph_plan_dim or 0)
    val_dataset.hypergraph_plan_dim = int(model_config.hypergraph_plan_dim or 0)
    if forward_model is not None:
        cache_name = str(cfg.get("forward_model", {}).get("latent_cache_name", "inverse_forward_latent_cache.pt"))
        cache_path = (resolve_demo_path(cfg.get("paths", {}).get("saved_inverse_root", cfg.get("paths", {}).get("saved_model_dir", "./Saved_Model_Inverse"))) / "cache" / cache_name).resolve()
        cache_payload = build_forward_latent_cache(
            forward_model,
            forward_metadata,
            list(train_dataset.records) + list(val_dataset.records),
            cache_path,
            device,
            packed_h5_path=packed_path,
            kpi_names=kpi_names,
            inverse_cache_config={
                "max_num_modules": model_config.max_num_modules,
                "design_dim": model_config.design_dim,
                "target_dim": model_config.target_dim,
                "generate_heat_power": model_config.generate_heat_power,
                "heat_load_policy": heat_load_policy,
                "center_decode_mode": model_config.center_decode_mode,
                "inverse_mode": model_config.inverse_mode,
                "use_hypergraph_plan": model_config.use_hypergraph_plan,
                "hypergraph_plan_dim": model_config.hypergraph_plan_dim,
                "hypergraph_plan_num_edges": model_config.hypergraph_plan_num_edges,
            },
            behavior_dim=model_config.behavior_latent_dim,
            organization_dim=model_config.organization_latent_dim,
            max_num_modules=model_config.max_num_modules,
            generate_heat_power=model_config.generate_heat_power,
            hypergraph_plan_dim_value=int(model_config.hypergraph_plan_dim or 0),
            hypergraph_plan_num_edges=int(model_config.hypergraph_plan_num_edges or infer_hypergraph_plan_num_edges(int(model_config.hypergraph_plan_dim or 0), model_config.max_num_modules)),
            heat_load_policy=heat_load_policy,
            fixed_heat_per_module=float(fixed_heat_per_module) if fixed_heat_per_module is not None else None,
            force_rebuild=bool(cfg.get("forward_model", {}).get("rebuild_latent_cache", False)),
            disable_auto_rebuild=bool(cfg.get("forward_model", {}).get("disable_auto_cache_rebuild", False)),
            query_batch_size=int(cfg.get("forward_model", {}).get("query_batch_size", 32768)),
        )
        train_dataset.set_latent_targets(cache_payload.get("latents", {}))
        val_dataset.set_latent_targets(cache_payload.get("latents", {}))
        train_dataset.set_hypergraph_plan_targets(cache_payload.get("hypergraph_plans", {}), plan_dim=int(model_config.hypergraph_plan_dim or 0))
        val_dataset.set_hypergraph_plan_targets(cache_payload.get("hypergraph_plans", {}), plan_dim=int(model_config.hypergraph_plan_dim or 0))

    model = ThermalInverseDesignFlow(model_config).to(device)
    print(
        f"[setup] inverse parameters={count_parameters(model):,}, "
        f"train_cases={len(train_dataset.records)} train_items={len(train_dataset)}, "
        f"val_cases={len(val_dataset.records)} val_items={len(val_dataset)}, device={device}"
    )
    print(
        f"[setup] design variables: centers+mask{' + heat_power' if model_config.generate_heat_power else ''}, "
        f"max_num_modules={model_config.max_num_modules}, center_decode_mode={model_config.center_decode_mode}"
    )
    print(
        f"[setup] inverse_mode={model_config.inverse_mode}, use_hypergraph_plan={model_config.use_hypergraph_plan}, "
        f"hypergraph_plan_dim={model_config.hypergraph_plan_dim or 0}"
    )
    print(
        f"[setup] samples_per_case train={train_dataset.samples_per_case}, val={val_dataset.samples_per_case}; "
        f"validation augmentation enabled={bool(val_aug_cfg.get('enabled', False))}; heat_load_policy={heat_load_policy}"
    )

    if args.dry_run:
        batch = collate_inverse([train_dataset[0], train_dataset[min(1, len(train_dataset) - 1)]])
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            _, metrics = model.training_loss(batch, loss_weights=loss_cfg)
        print("[dry-run] batch keys:", sorted(batch.keys()))
        print("[dry-run] metrics:", {key: scalar(value) for key, value in metrics.items()})
        return 0

    run_id = resolve_run_id(args, cfg)
    # Keep model run directories short and stable. Descriptive metadata is
    # already stored in resolved_train_inverse_config.json inside the run.
    run_name = f"Run_{run_id}_{current_timestamp()}"
    saved_root = ensure_dir(resolve_demo_path(cfg.get("paths", {}).get("saved_inverse_root", cfg.get("paths", {}).get("saved_model_dir", "./Saved_Model_Inverse"))))
    run_dir = ensure_dir(saved_root / run_name)
    write_json(run_dir / "resolved_train_inverse_config.json", cfg)
    write_json(run_dir / "kpi_stats.json", kpi_stats)
    write_json(run_dir / "kpi_distribution_summary.json", kpi_distribution_summary)

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
    replay_buffer: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_verified = float("inf")
    best_verified_metric_name = str(validation_cfg.get("best_metric_name", "auto") or "auto")
    history_path = run_dir / "loss_history.csv"
    metrics_history_path = run_dir / "training_metrics.csv"
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
                count_mode=str(validation_cfg.get("forward_verify_count_mode", "uniform")),
                heat_load_policy=heat_load_policy,
                fixed_heat_per_module=float(fixed_heat_per_module) if fixed_heat_per_module is not None else None,
                temperature_limits=temperature_limits,
            )
            row.update(verify_metrics)
            demo_target_jsons = list(validation_cfg.get("demo_target_jsons", []) or [])
            if validation_cfg.get("demo_target_json"):
                demo_target_jsons.append(validation_cfg["demo_target_json"])
            for target_json in demo_target_jsons:
                target_spec = _target_spec_from_json_for_validation(
                    target_json,
                    kpi_names=kpi_names,
                    kpi_stats=kpi_stats,
                    normalize=bool(target_cfg.get("normalize", True)),
                    model=model,
                )
                row.update(
                    run_fixed_target_verification(
                        model,
                        forward_model,
                        forward_metadata,
                        val_dataset,
                        target_spec,
                        device,
                        num_samples=int(validation_cfg.get("demo_forward_verify_num_samples", validation_cfg.get("forward_verify_num_samples", 4))),
                        n_steps=int(validation_cfg.get("demo_forward_verify_n_steps", validation_cfg.get("forward_verify_n_steps", 16))),
                        query_batch_size=int(validation_cfg.get("forward_verify_query_batch_size", cfg.get("forward_model", {}).get("query_batch_size", 32768))),
                        seed=int(training_cfg.get("seed", 42)) + epoch + 1009,
                        count_mode=str(validation_cfg.get("demo_forward_verify_count_mode", validation_cfg.get("forward_verify_count_mode", "uniform"))),
                        heat_load_policy=heat_load_policy,
                        fixed_heat_per_module=float(fixed_heat_per_module) if fixed_heat_per_module is not None else None,
                        temperature_limits=temperature_limits,
                    )
                )
            design_metric_names = sorted(key for key in row if key.startswith("val_design_intent_score_"))
            demo_metric_names = sorted(key for key in row if key.startswith("val_forward_score_") and key != "val_forward_score_reconstruct")
            metric_name = best_verified_metric_name
            if metric_name.lower() in {"auto", ""}:
                metric_name = design_metric_names[0] if conditioning_mode == "design_intent" and design_metric_names else "val_design_intent_score_reconstruct" if conditioning_mode == "design_intent" and "val_design_intent_score_reconstruct" in row else demo_metric_names[0] if demo_metric_names else "val_forward_score_reconstruct"
            if metric_name not in row:
                fallback_metric = design_metric_names[0] if conditioning_mode == "design_intent" and design_metric_names else "val_design_intent_score_reconstruct" if conditioning_mode == "design_intent" and "val_design_intent_score_reconstruct" in row else demo_metric_names[0] if demo_metric_names else "val_forward_score_reconstruct"
                print(f"[checkpoint] best metric {metric_name!r} unavailable; using {fallback_metric!r}.")
                metric_name = fallback_metric
            verified = float(row.get(metric_name, float("inf")))
            if math.isfinite(verified) and verified < best_verified:
                best_verified = verified
                row["best_verified_checkpoint_selected"] = 1
                row["best_verified_metric_name"] = metric_name
                row["best_verified_checkpoint_reason"] = f"lowest {metric_name}={verified:.6g}"
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
                    kpi_distribution_summary=kpi_distribution_summary,
                )
                print(f"[checkpoint] best_verified_model.pt updated: {row['best_verified_checkpoint_reason']}")
            else:
                row["best_verified_checkpoint_selected"] = 0
                row["best_verified_metric_name"] = metric_name
                row["best_verified_checkpoint_reason"] = f"kept previous lowest {metric_name}={best_verified:.6g}"
        if (
            forward_model is not None
            and bool(forward_guidance_cfg.get("enabled", False))
            and epoch >= int(forward_guidance_cfg.get("start_epoch", 200))
            and epoch % max(int(forward_guidance_cfg.get("interval_epochs", 50)), 1) == 0
        ):
            new_replay = run_forward_guidance_replay(
                model,
                forward_model,
                forward_metadata,
                val_dataset,
                device,
                num_intents=int(forward_guidance_cfg.get("num_intents", validation_cfg.get("forward_verify_num_targets", 4))),
                num_candidates_per_intent=int(forward_guidance_cfg.get("num_candidates_per_intent", 16)),
                replay_top_k=int(forward_guidance_cfg.get("replay_top_k", 2)),
                n_steps=int(forward_guidance_cfg.get("n_steps", validation_cfg.get("forward_verify_n_steps", 8))),
                seed=int(training_cfg.get("seed", 42)) + epoch + 4049,
                count_mode=str(forward_guidance_cfg.get("count_mode", validation_cfg.get("forward_verify_count_mode", "uniform"))),
                heat_load_policy=heat_load_policy,
                fixed_heat_per_module=float(fixed_heat_per_module) if fixed_heat_per_module is not None else None,
                query_batch_size=int(forward_guidance_cfg.get("query_batch_size", cfg.get("forward_model", {}).get("query_batch_size", 32768))),
            )
            replay_buffer.extend(new_replay)
            max_replay = int(forward_guidance_cfg.get("max_replay_items", 256))
            if len(replay_buffer) > max_replay:
                replay_buffer = replay_buffer[-max_replay:]
            if replay_buffer:
                model.train(True)
                batch_size = min(int(training_cfg.get("batch_size", 64)), len(replay_buffer))
                replay_batch = collate_inverse(replay_buffer[-batch_size:])
                replay_batch = move_batch_to_device(replay_batch, device)
                optimizer.zero_grad(set_to_none=True)
                replay_loss, replay_metrics = model.training_loss(replay_batch, loss_weights=loss_cfg)
                (float(forward_guidance_cfg.get("replay_weight", 0.2)) * replay_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("gradient_clip_norm", 1.0)))
                optimizer.step()
                row["forward_guidance_replay_items"] = len(replay_buffer)
                row["forward_guidance_replay_loss_total"] = scalar(replay_metrics["loss_total"])
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
                kpi_distribution_summary=kpi_distribution_summary,
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
            kpi_distribution_summary=kpi_distribution_summary,
        )
        history.append(row)
        save_history_csv(history, history_path)
        save_history_csv(history, metrics_history_path)
        if epoch % max(int(training_cfg.get("save_every_epochs", 25)), 1) == 0 or epoch == epochs:
            save_loss_curve(history_path, run_dir / "loss_curve.png")
            save_forward_score_curve(history_path, run_dir / "forward_score_curve.png")
        demo_forward = " ".join(
            f"{key}={float(row[key]):.4e}" for key in sorted(row) if key.startswith("val_forward_score_") and key != "val_forward_score_reconstruct" and isinstance(row.get(key), (int, float))
        )
        print(
            f"[epoch {epoch:04d}] train_loss_total={train_metrics.get('loss_total', float('nan')):.4e} "
            f"val_loss_total={val_loss:.4e} "
            f"loss_hypergraph_plan={train_metrics.get('loss_hypergraph_plan', float('nan')):.4e} "
            f"val_loss_hypergraph_plan={val_metrics.get('loss_hypergraph_plan', float('nan')):.4e} "
            f"val_forward_score_reconstruct={row.get('val_forward_score_reconstruct', float('nan')):.4e} "
            f"{demo_forward} val_candidate_diversity={row.get('val_candidate_diversity', float('nan'))} "
            f"checkpoint={row.get('best_verified_checkpoint_reason', 'not verified this epoch')}"
        )
    save_loss_curve(history_path, run_dir / "loss_curve.png")
    save_forward_score_curve(history_path, run_dir / "forward_score_curve.png")
    print(f"[done] inverse run saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
