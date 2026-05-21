"""ChannelThermal point-query dataset for unified forward training.

This loader reads the existing packed ChannelThermal HDF5 file in place and
converts each sampled case to the sandbox ``BatchData`` contract. It uses the
preprocessed sparse ``sampled_points`` arrays when available and falls back to
sampling dense ``steady_field`` grids when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from unified_types import BatchData


EPS = 1e-6


def _decode_strings(values: Any) -> List[str]:
    arr = np.asarray(values)
    out: List[str] = []
    for item in arr.reshape(-1):
        out.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return out


def _read_scalar_string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _select_split(case_ids: Sequence[str], splits: Sequence[str], split: str) -> List[str]:
    split = str(split).lower()
    if split == "all":
        return list(case_ids)
    return [case_id for case_id, item in zip(case_ids, splits) if str(item).lower() == split]


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    return values.clamp_min(1e-6)


class ChannelThermalPointDataset(Dataset):
    """Sample point-query ChannelThermal batches as ``BatchData`` objects."""

    def __init__(
        self,
        packed_h5_path: str | Path,
        split: str = "train",
        max_cases: Optional[int] = None,
        points_per_case: int = 1024,
        max_num_modules: int = 12,
        field_dim: int = 5,
        normalize_targets: bool = True,
        target_mean: Optional[Sequence[float]] = None,
        target_std: Optional[Sequence[float]] = None,
        seed: int = 0,
    ):
        self.path = Path(packed_h5_path).expanduser()
        if not self.path.is_absolute():
            self.path = (Path.cwd() / self.path).resolve()
        self.split = str(split)
        self.points_per_case = int(points_per_case)
        self.max_num_modules = int(max_num_modules)
        self.field_dim = int(field_dim)
        self.normalize_targets = bool(normalize_targets)
        self.seed = int(seed)
        self._h5: Optional[h5py.File] = None
        self.layout_warnings: List[str] = []

        if not self.path.exists():
            raise FileNotFoundError(f"ChannelThermal packed HDF5 not found: {self.path}")

        with h5py.File(self.path, "r") as h5:
            if "cases" not in h5:
                raise KeyError(f"{self.path} does not contain a 'cases' group.")
            if "case_ids" in h5 and "splits" in h5:
                all_case_ids = _decode_strings(h5["case_ids"][...])
                all_splits = _decode_strings(h5["splits"][...])
            else:
                all_case_ids = sorted(h5["cases"].keys())
                all_splits = [_read_scalar_string(h5["cases"][case_id].attrs.get("split", "all")) for case_id in all_case_ids]
                self.layout_warnings.append("Root case_ids/splits missing; using per-case split attrs.")
            selected = _select_split(all_case_ids, all_splits, self.split)
            if max_cases is not None:
                selected = selected[: int(max_cases)]
            if not selected:
                raise ValueError(f"No ChannelThermal cases found for split={split!r} in {self.path}.")
            self.case_ids = selected
            self.channel_order = (
                _decode_strings(h5["channel_order"][...])
                if "channel_order" in h5
                else ["u", "v", "p", "omega", "temperature"][: self.field_dim]
            )
            self.root_max_modules = int(h5.attrs.get("max_modules", self.max_num_modules))

            norm = h5.get("normalization")
            h5_mean = None
            h5_std = None
            if norm is not None:
                mean_key = "sampled_point_mean_by_channel" if "sampled_point_mean_by_channel" in norm else "field_mean_by_channel"
                std_key = "sampled_point_std_by_channel" if "sampled_point_std_by_channel" in norm else "field_std_by_channel"
                if mean_key in norm and std_key in norm:
                    h5_mean = torch.as_tensor(norm[mean_key][...], dtype=torch.float32)[: self.field_dim]
                    h5_std = torch.as_tensor(norm[std_key][...], dtype=torch.float32)[: self.field_dim]

        self.target_mean = (
            torch.as_tensor(target_mean, dtype=torch.float32)[: self.field_dim]
            if target_mean is not None
            else h5_mean
        )
        self.target_std = (
            _safe_std(torch.as_tensor(target_std, dtype=torch.float32)[: self.field_dim])
            if target_std is not None
            else (_safe_std(h5_std) if h5_std is not None else None)
        )

        if self.normalize_targets and (self.target_mean is None or self.target_std is None):
            self.layout_warnings.append("Target normalization requested but stats are missing; targets remain physical.")

        for warning in self.layout_warnings:
            print(f"[channelthermal_dataset] {warning}")

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def compute_target_stats(self, max_cases: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Compute per-channel target mean/std over selected cases."""
        sums = torch.zeros(self.field_dim, dtype=torch.float64)
        sums_sq = torch.zeros(self.field_dim, dtype=torch.float64)
        count = 0
        case_ids = self.case_ids[: int(max_cases)] if max_cases is not None else self.case_ids
        with h5py.File(self.path, "r") as h5:
            for case_id in case_ids:
                group = h5["cases"][case_id]
                targets = self._read_all_targets_np(group)
                tensor = torch.as_tensor(targets[:, : self.field_dim], dtype=torch.float64)
                sums += tensor.sum(dim=0)
                sums_sq += tensor.square().sum(dim=0)
                count += int(tensor.shape[0])
        if count <= 0:
            raise RuntimeError("No targets found while computing ChannelThermal target stats.")
        mean = sums / float(count)
        var = (sums_sq / float(count)) - mean.square()
        std = torch.sqrt(var.clamp_min(1e-12))
        return {"mean": mean.float(), "std": _safe_std(std.float()), "count": torch.tensor(count)}

    def __getitem__(self, index: int) -> BatchData:
        case_id = self.case_ids[int(index)]
        group = self.h5["cases"][case_id]
        query_xy, target_field = self._sample_points(group, int(index))
        module_centers = self._read_array(group, ["module_centers", "centers"], required=True).astype(np.float32)
        module_present = self._read_array(group, ["module_present", "present"], required=False)
        heat_powers = self._read_array(group, ["heat_powers", "heat_power"], required=False)
        if module_present is None:
            module_present = np.isfinite(module_centers).all(axis=-1).astype(np.float32)
        else:
            module_present = module_present.astype(np.float32)
        if heat_powers is None:
            heat_powers = np.zeros((module_centers.shape[0],), dtype=np.float32)
        else:
            heat_powers = heat_powers.astype(np.float32).reshape(-1)

        lx, ly = self._domain_lengths(group)
        module_centers_padded = np.zeros((self.max_num_modules, 2), dtype=np.float32)
        module_present_padded = np.zeros((self.max_num_modules,), dtype=np.float32)
        module_features = np.zeros((self.max_num_modules, 8), dtype=np.float32)
        count = min(module_centers.shape[0], self.max_num_modules)
        module_centers_padded[:count] = module_centers[:count]
        module_present_padded[:count] = module_present[:count]
        heat = np.zeros((self.max_num_modules,), dtype=np.float32)
        heat[: min(heat_powers.shape[0], self.max_num_modules)] = heat_powers[: self.max_num_modules]
        heat_scale = max(float(np.max(np.abs(heat))) if heat.size else 0.0, 1.0)
        module_features[:, 0] = module_centers_padded[:, 0] / max(lx, EPS)
        module_features[:, 1] = module_centers_padded[:, 1] / max(ly, EPS)
        module_features[:, 2] = 0.45 / max(min(lx, ly), EPS)
        module_features[:, 3] = heat / heat_scale
        module_features[:, 4] = module_present_padded

        if self.normalize_targets and self.target_mean is not None and self.target_std is not None:
            target_field = (target_field - self.target_mean.numpy()) / self.target_std.numpy()

        global_context = np.zeros((8,), dtype=np.float32)
        global_context[0] = 0.0
        global_context[1] = lx
        global_context[2] = ly
        global_context[3] = float(np.sum(module_present_padded))
        global_context[4] = float(np.sum(np.abs(heat)))

        return BatchData(
            module_centers=torch.from_numpy(module_centers_padded),
            module_present=torch.from_numpy(module_present_padded),
            module_features=torch.from_numpy(module_features),
            global_context=torch.from_numpy(global_context),
            query_xy=torch.from_numpy(query_xy.astype(np.float32)),
            query_time=None,
            target_field=torch.from_numpy(target_field.astype(np.float32)),
            case_name="channelthermal",
            metadata={
                "synthetic": False,
                "case_id": str(case_id),
                "source": str(self.path),
                "split": self.split,
                "normalized_targets": bool(self.normalize_targets and self.target_mean is not None),
                "domain_length_x": float(lx),
                "domain_length_y": float(ly),
            },
        )

    def _sample_points(self, group: h5py.Group, item_index: int) -> tuple[np.ndarray, np.ndarray]:
        if "sampled_points" in group:
            samples = np.asarray(group["sampled_points"], dtype=np.float32)
            if samples.ndim != 2 or samples.shape[-1] < 2 + self.field_dim:
                raise ValueError(f"sampled_points has unexpected shape {samples.shape}; expected [N,{2 + self.field_dim}+].")
            point_indices = self._choose_indices(samples.shape[0], item_index)
            selected = samples[point_indices]
            return selected[:, 0:2], selected[:, 2 : 2 + self.field_dim]

        field = self._read_array(group, ["steady_field", "field", "state"], required=True).astype(np.float32)
        if field.ndim == 3 and field.shape[-1] >= self.field_dim:
            dense = field[..., : self.field_dim]
        elif field.ndim == 3 and field.shape[0] >= self.field_dim:
            dense = np.moveaxis(field[: self.field_dim], 0, -1)
            print("[channelthermal_dataset] Transposed dense field from [C,H,W] to [H,W,C].")
        else:
            raise ValueError(f"steady_field has unsupported shape {field.shape}.")
        x_grid = self._read_array(group, ["x_grid", "grid_x"], required=True).astype(np.float32)
        y_grid = self._read_array(group, ["y_grid", "grid_y"], required=True).astype(np.float32)
        flat_field = dense.reshape(-1, dense.shape[-1])
        flat_xy = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1).astype(np.float32)
        point_indices = self._choose_indices(flat_field.shape[0], item_index)
        return flat_xy[point_indices], flat_field[point_indices]

    def _read_all_targets_np(self, group: h5py.Group) -> np.ndarray:
        if "sampled_points" in group:
            samples = np.asarray(group["sampled_points"], dtype=np.float32)
            return samples[:, 2 : 2 + self.field_dim]
        field = self._read_array(group, ["steady_field", "field", "state"], required=True).astype(np.float32)
        if field.ndim == 3 and field.shape[-1] >= self.field_dim:
            return field.reshape(-1, field.shape[-1])[:, : self.field_dim]
        if field.ndim == 3 and field.shape[0] >= self.field_dim:
            return np.moveaxis(field[: self.field_dim], 0, -1).reshape(-1, self.field_dim)
        raise ValueError(f"Could not infer target field layout for shape {field.shape}.")

    def _choose_indices(self, num_points: int, item_index: int) -> np.ndarray:
        if self.points_per_case <= 0 or num_points <= self.points_per_case:
            return np.arange(num_points, dtype=np.int64)
        rng = np.random.default_rng(self.seed + int(item_index) * 104729)
        return rng.choice(num_points, size=self.points_per_case, replace=False)

    def _domain_lengths(self, group: h5py.Group) -> tuple[float, float]:
        if "x_grid" in group and "y_grid" in group:
            x_grid = np.asarray(group["x_grid"], dtype=np.float32)
            y_grid = np.asarray(group["y_grid"], dtype=np.float32)
            dx = float(np.mean(np.diff(x_grid[0]))) if x_grid.ndim == 2 and x_grid.shape[1] > 1 else 0.0
            dy = float(np.mean(np.diff(y_grid[:, 0]))) if y_grid.ndim == 2 and y_grid.shape[0] > 1 else 0.0
            lx = float(x_grid.max() - x_grid.min() + abs(dx))
            ly = float(y_grid.max() - y_grid.min() + abs(dy))
            return lx, ly
        return 12.0, 4.0

    @staticmethod
    def _read_array(group: h5py.Group, aliases: Iterable[str], required: bool) -> Optional[np.ndarray]:
        for key in aliases:
            if key in group:
                return np.asarray(group[key])
        if required:
            raise KeyError(f"None of the expected keys were found in case group: {list(aliases)}")
        print(f"[channelthermal_dataset] Optional keys missing: {list(aliases)}")
        return None


def collate_batchdata(items: Sequence[BatchData]) -> BatchData:
    """Collate a list of ``BatchData`` objects into a batched ``BatchData``."""
    if not items:
        raise ValueError("Cannot collate an empty BatchData list.")
    target_field = None if items[0].target_field is None else torch.stack([item.target_field for item in items], dim=0)
    query_time = None if items[0].query_time is None else torch.stack([item.query_time for item in items], dim=0)
    return BatchData(
        module_centers=torch.stack([item.module_centers for item in items], dim=0),
        module_present=torch.stack([item.module_present for item in items], dim=0),
        module_features=torch.stack([item.module_features for item in items], dim=0),
        global_context=torch.stack([item.global_context for item in items], dim=0),
        query_xy=torch.stack([item.query_xy for item in items], dim=0),
        query_time=query_time,
        target_field=target_field,
        case_name=items[0].case_name,
        metadata={"items": [item.metadata for item in items], "synthetic": False},
    )
