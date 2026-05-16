from __future__ import annotations

"""Dataset utilities for target-agnostic design-prior training."""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from layout_search_baselines import LayoutSearchConfig, encode_layout_to_design_vec
    from thermal_inverse_kpi import compute_steady_thermal_kpis, layout_spread_metrics
except Exception:  # pragma: no cover
    from .layout_search_baselines import LayoutSearchConfig, encode_layout_to_design_vec
    from .thermal_inverse_kpi import compute_steady_thermal_kpis, layout_spread_metrics


FALLBACK_KPI_NAMES: Tuple[str, ...] = (
    "max_solid_temperature",
    "pressure_drop",
    "outlet_temperature_nonuniformity",
    "wall_hot_area_fraction",
    "thermal_plume_length",
    "downstream_reheat_index",
)


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _decode_array(values: Any) -> List[str]:
    return [_decode(item) for item in np.asarray(values).reshape(-1)]


def _resolve_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_dir():
        for name in ("design_library.h5", "design_library.hdf5", "design_library.npz", "library.h5", "library.npz"):
            candidate = raw / name
            if candidate.exists():
                return candidate
        matches = sorted(list(raw.glob("*.h5")) + list(raw.glob("*.hdf5")) + list(raw.glob("*.npz")))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No .h5/.hdf5/.npz design library found in {raw}.")
    return raw


def _first_array_npz(data: Mapping[str, Any], keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    return None


def _first_dataset_h5(group: h5py.Group | h5py.File, keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in group and isinstance(group[key], h5py.Dataset):
            return group[key][...]
    return None


def _as_2d(arr: Optional[np.ndarray], n: int = 0) -> np.ndarray:
    if arr is None:
        return np.zeros((int(n), 0), dtype=np.float32)
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1:
        out = out[:, None]
    return out.reshape(out.shape[0], -1).astype(np.float32)


def _normalize(arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    if arr.size == 0 or arr.shape[1] == 0:
        return arr.astype(np.float32), {"mean": [], "std": []}
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std = np.where(np.abs(std) < 1.0e-8, 1.0, std).astype(np.float32)
    return ((np.nan_to_num(arr, nan=0.0) - mean) / std).astype(np.float32), {"mean": mean.tolist(), "std": std.tolist()}


class DesignPriorDataset(Dataset):
    """Dataset for target-agnostic design atlas training.

    Expected arrays:
        design_vec: [N, design_dim]
        hypergraph_vec: [N, hypergraph_dim] optional
        hypergraph_mask: [N, hypergraph_dim] optional
        behavior_vec: [N, behavior_dim] optional
        context_vec: [N, context_dim] optional
        sample_weight: [N] optional
    """

    def __init__(self, path: str | Path, *, behavior_dim: Optional[int] = None, generate_heat_power: bool = False) -> None:
        self.path = _resolve_path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Design prior dataset not found: {self.path}")
        self.generate_heat_power = bool(generate_heat_power)
        self.metadata: Dict[str, Any] = {"path": str(self.path), "warnings": []}
        suffix = self.path.suffix.lower()
        if suffix == ".npz":
            arrays = self._load_npz(self.path)
        elif suffix in {".h5", ".hdf5"}:
            arrays = self._load_h5(self.path)
        else:
            raise ValueError(f"Unsupported design prior dataset suffix: {self.path.suffix}")
        self.design_vec = _as_2d(arrays.get("design_vec"))
        if self.design_vec.shape[0] == 0:
            raise ValueError("design_vec is required and cannot be empty.")
        n = self.design_vec.shape[0]
        self.hypergraph_vec = _as_2d(arrays.get("hypergraph_vec"), n)
        self.hypergraph_mask = _as_2d(arrays.get("hypergraph_mask"), n)
        if self.hypergraph_vec.shape[1] == 0:
            self._warn("hypergraph_vec missing; using hypergraph_dim=0.")
        elif self.hypergraph_mask.shape[1] == 0:
            self.hypergraph_mask = np.ones_like(self.hypergraph_vec, dtype=np.float32)
        self.behavior_vec = _as_2d(arrays.get("behavior_vec"), n)
        if self.behavior_vec.shape[1] == 0:
            fallback = self._fallback_behavior(arrays, behavior_dim=behavior_dim)
            self.behavior_vec = fallback
            if fallback.shape[1] == 0:
                self._warn("behavior_vec missing and no fallback KPI/descriptor arrays found; using behavior_dim=0.")
        if behavior_dim is not None and self.behavior_vec.shape[1] > 0:
            self.behavior_vec = self._pad_or_trim(self.behavior_vec, int(behavior_dim))
        self.context_vec = _as_2d(arrays.get("context_vec"), n)
        if self.context_vec.shape[1] == 0:
            self._warn("context_vec missing; using context_dim=0.")
        sample_weight = arrays.get("sample_weight")
        self.sample_weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1)[:n] if sample_weight is not None else np.ones((n,), dtype=np.float32)
        if self.sample_weight.size < n:
            self.sample_weight = np.pad(self.sample_weight, (0, n - self.sample_weight.size), constant_values=1.0)
        self.source_tag = np.asarray(arrays.get("source_tag", np.asarray([""] * n, dtype=object))).reshape(-1)
        self.case_id = np.asarray(arrays.get("case_id", np.asarray([str(i) for i in range(n)], dtype=object))).reshape(-1)
        self.stats = self.compute_dataset_stats()

    @staticmethod
    def _pad_or_trim(arr: np.ndarray, dim: int) -> np.ndarray:
        out = np.zeros((arr.shape[0], max(dim, 0)), dtype=np.float32)
        if dim > 0 and arr.size:
            out[:, : min(dim, arr.shape[1])] = arr[:, : min(dim, arr.shape[1])]
        return out

    def _warn(self, message: str) -> None:
        warnings.warn(message)
        self.metadata.setdefault("warnings", []).append(message)

    def _load_npz(self, path: Path) -> Dict[str, Any]:
        data = np.load(path, allow_pickle=True)
        arrays = {key: data[key] for key in data.files}
        self.metadata["format"] = "npz"
        self.metadata["keys"] = sorted(data.files)
        return arrays

    def _load_h5(self, path: Path) -> Dict[str, Any]:
        with h5py.File(path, "r") as h5:
            self.metadata["format"] = "h5"
            self.metadata["keys"] = sorted(h5.keys())
            if "design_vec" in h5:
                return self._load_flat_h5(h5)
            if "cases" in h5:
                return self._load_packed_cases_h5(h5)
        raise ValueError(f"No design_vec or cases group found in {path}.")

    def _load_flat_h5(self, h5: h5py.File) -> Dict[str, Any]:
        arrays = {
            "design_vec": _first_dataset_h5(h5, ("design_vec", "design_vectors")),
            "hypergraph_vec": _first_dataset_h5(h5, ("hypergraph_vec", "hypergraph_vectors", "hypergraph_plan_target")),
            "hypergraph_mask": _first_dataset_h5(h5, ("hypergraph_mask", "hypergraph_plan_mask")),
            "behavior_vec": _first_dataset_h5(h5, ("behavior_vec", "behavior_vectors", "behavior_descriptor_vec")),
            "context_vec": _first_dataset_h5(h5, ("context_vec", "context_vectors")),
            "sample_weight": _first_dataset_h5(h5, ("sample_weight", "sample_weights")),
            "kpi_descriptor_vec": _first_dataset_h5(h5, ("kpi_descriptor_vec", "kpi_vec", "kpi_vector")),
            "layout_descriptor_vec": _first_dataset_h5(h5, ("layout_descriptor_vec", "layout_vec")),
        }
        for key in ("source_tag", "case_id"):
            if key in h5:
                arrays[key] = np.asarray(_decode_array(h5[key][...]), dtype=object)
        return arrays

    def _load_packed_cases_h5(self, h5: h5py.File) -> Dict[str, Any]:
        case_ids = sorted(h5["cases"].keys())
        max_modules = int(h5.attrs.get("max_modules", 12))
        design_rows = []
        behavior_rows = []
        context_rows = []
        layout_rows = []
        for case_id in case_ids:
            group = h5["cases"][case_id]
            centers = group["module_centers"][...].astype(np.float32)
            present = group["module_present"][...].astype(np.float32) if "module_present" in group else np.ones((centers.shape[0],), dtype=np.float32)
            heat = group["heat_powers"][...].astype(np.float32) if "heat_powers" in group else None
            domain = {
                "domain_length_x": float(group.attrs.get("domain_length_x", h5.attrs.get("domain_length_x", 12.0))),
                "domain_length_y": float(group.attrs.get("domain_length_y", h5.attrs.get("domain_length_y", 4.0))),
            }
            radius = float(group.attrs.get("module_radius", h5.attrs.get("module_radius", 0.45)))
            cfg = LayoutSearchConfig(max_num_modules=max_modules, domain_length_x=domain["domain_length_x"], domain_length_y=domain["domain_length_y"], module_radius=radius, generate_heat_power=self.generate_heat_power)
            active = centers[present.reshape(-1) > 0.5]
            layout = {"centers": active, "count": int(active.shape[0]), "module_radius": radius, "domain": {**domain, "module_radius": radius}}
            if heat is not None:
                layout["heat_powers"] = heat[present.reshape(-1) > 0.5]
            design_rows.append(encode_layout_to_design_vec(layout, cfg))
            kpis: Dict[str, Any] = {}
            if "steady_field" in group:
                try:
                    kpis = compute_steady_thermal_kpis(
                        group["steady_field"][...],
                        x_grid=group["x_grid"][...] if "x_grid" in group else None,
                        y_grid=group["y_grid"][...] if "y_grid" in group else None,
                        module_centers=centers,
                        module_present=present,
                        heat_powers=heat,
                        module_mask=group["module_mask"][...] if "module_mask" in group else None,
                        domain={**domain, "module_radius": radius},
                    )
                except Exception as exc:
                    self._warn(f"Could not compute KPIs for case {case_id}: {exc}")
            behavior_rows.append([float(kpis.get(name, 0.0) or 0.0) for name in FALLBACK_KPI_NAMES])
            spread = layout_spread_metrics(active, num_modules=active.shape[0])
            layout_rows.append([float(active.shape[0]), float(spread.get("x_coverage", 0.0)), float(spread.get("y_coverage", 0.0)), float(spread.get("bbox_area", 0.0)), float(spread.get("mean_pair_distance", 0.0))])
            context_rows.append([float(group.attrs.get("re", 0.0)), float(group.attrs.get("u_in", 0.0))])
        behavior, norm_stats = _normalize(np.asarray(behavior_rows, dtype=np.float32))
        self.metadata["fallback_behavior_normalization"] = norm_stats
        return {
            "design_vec": np.asarray(design_rows, dtype=np.float32),
            "behavior_vec": behavior,
            "context_vec": np.asarray(context_rows, dtype=np.float32),
            "layout_descriptor_vec": np.asarray(layout_rows, dtype=np.float32),
            "sample_weight": np.ones((len(design_rows),), dtype=np.float32),
            "case_id": np.asarray(case_ids, dtype=object),
            "source_tag": np.asarray(["packed_cases"] * len(design_rows), dtype=object),
        }

    def _fallback_behavior(self, arrays: Mapping[str, Any], *, behavior_dim: Optional[int]) -> np.ndarray:
        candidates = []
        for key in ("kpi_descriptor_vec", "kpi_vec", "kpi_vector", "layout_descriptor_vec"):
            arr = arrays.get(key)
            if arr is not None:
                candidates.append(_as_2d(np.asarray(arr), self.design_vec.shape[0]))
        if not candidates:
            return np.zeros((self.design_vec.shape[0], 0), dtype=np.float32)
        raw = np.concatenate(candidates, axis=1)
        if behavior_dim is not None:
            raw = self._pad_or_trim(raw, int(behavior_dim))
        normalized, stats = _normalize(raw)
        self.metadata["fallback_behavior_normalization"] = stats
        return normalized

    def compute_dataset_stats(self) -> Dict[str, Any]:
        def stats(arr: np.ndarray) -> Dict[str, Any]:
            if arr.size == 0 or arr.shape[1] == 0:
                return {"dim": 0, "mean": [], "std": []}
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            return {"dim": int(arr.shape[1]), "mean": mean.tolist(), "std": np.where(std < 1.0e-8, 1.0, std).tolist()}

        return {
            "num_samples": int(self.design_vec.shape[0]),
            "design": stats(self.design_vec),
            "hypergraph": stats(self.hypergraph_vec),
            "behavior": stats(self.behavior_vec),
            "context": stats(self.context_vec),
            "sample_weight_mean": float(np.mean(self.sample_weight)) if self.sample_weight.size else 1.0,
            "metadata": self.metadata,
        }

    def get_normalization(self) -> Dict[str, Any]:
        return self.stats

    def __len__(self) -> int:
        return int(self.design_vec.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i = int(idx)
        item: Dict[str, Any] = {
            "design_vec": self.design_vec[i],
            "sample_weight": np.asarray([self.sample_weight[i]], dtype=np.float32),
            "case_id": _decode(self.case_id[i]) if i < self.case_id.size else str(i),
            "source_tag": _decode(self.source_tag[i]) if i < self.source_tag.size else "",
        }
        if self.hypergraph_vec.shape[1] > 0:
            item["hypergraph_vec"] = self.hypergraph_vec[i]
            item["hypergraph_mask"] = self.hypergraph_mask[i]
        if self.behavior_vec.shape[1] > 0:
            item["behavior_vec"] = self.behavior_vec[i]
        if self.context_vec.shape[1] > 0:
            item["context_vec"] = self.context_vec[i]
        return item

    @staticmethod
    def collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "case_id": [str(item.get("case_id", "")) for item in batch],
            "source_tag": [str(item.get("source_tag", "")) for item in batch],
        }
        for key in ("design_vec", "hypergraph_vec", "hypergraph_mask", "behavior_vec", "context_vec", "sample_weight"):
            if key in batch[0]:
                out[key] = torch.as_tensor(np.stack([np.asarray(item[key], dtype=np.float32) for item in batch]), dtype=torch.float32)
        if "sample_weight" in out:
            out["sample_weight"] = out["sample_weight"].reshape(-1)
        return out


def collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return DesignPriorDataset.collate_fn(batch)
