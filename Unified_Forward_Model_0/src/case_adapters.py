"""Lightweight case adapters for the unified forward-model sandbox.

Adapters read a single small batch from existing demo dataset paths when
possible. Missing paths, unavailable HDF5 support, or unknown dataset layouts
fall back to synthetic smoke batches without copying data into the sandbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from unified_types import BatchData


DEFAULT_CHANNEL_PATH = "../1_Demo_ChannelThermal/Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"
DEFAULT_MULTICYLINDER_PATH = "../0_Demo_MultiCylinder/Data_Saved/Processed_MultiCylinder_Dataset/packed_dataset.h5"


def describe_batch(batch: BatchData) -> Dict[str, Any]:
    """Return shape and provenance information for a loaded BatchData object."""
    return {
        "case_name": batch.case_name,
        "synthetic": bool(batch.metadata.get("synthetic", False)),
        "module_centers_shape": _shape(batch.module_centers),
        "module_present_shape": _shape(batch.module_present),
        "module_features_shape": _shape(batch.module_features),
        "global_context_shape": _shape(batch.global_context),
        "query_xy_shape": _shape(batch.query_xy),
        "query_time_shape": _shape(batch.query_time),
        "target_field_shape": _shape(batch.target_field),
        "metadata": batch.metadata,
    }


def make_synthetic_batch(case_name: str = "channel", batch_size: int = 2, points_per_case: int = 192) -> BatchData:
    """Create a deterministic synthetic batch for smoke tests."""
    torch.manual_seed(7 if case_name.lower().startswith("multi") else 5)
    is_multi = case_name.lower().startswith("multi")
    lx, ly = (1.0, 1.0) if is_multi else (12.0, 4.0)
    max_modules = 6 if is_multi else 5

    module_centers = torch.zeros(batch_size, max_modules, 2)
    module_present = torch.zeros(batch_size, max_modules)
    module_features = torch.zeros(batch_size, max_modules, 8)
    for b in range(batch_size):
        count = 3 + (b % 3)
        module_present[b, :count] = 1.0
        xs = torch.linspace(0.18 * lx, 0.82 * lx, count)
        ys = torch.linspace(0.25 * ly, 0.75 * ly, count).roll(b)
        module_centers[b, :count, 0] = xs
        module_centers[b, :count, 1] = ys
        module_features[b, :count, 0:2] = module_centers[b, :count] / torch.tensor([lx, ly])
        module_features[b, :count, 2] = 0.45 if not is_multi else 0.06
        module_features[b, :count, 3] = torch.linspace(0.5, 1.0, count)
        module_features[b, :count, 4] = float(is_multi)
        module_features[b, :count, 5] = torch.arange(count).float() / max(float(count - 1), 1.0)

    query_xy = torch.rand(batch_size, points_per_case, 2) * torch.tensor([lx, ly])
    query_time = torch.rand(batch_size, points_per_case, 1) if is_multi else None
    global_context = torch.zeros(batch_size, 8)
    global_context[:, 0] = 1.0 if is_multi else 0.0
    global_context[:, 1] = lx
    global_context[:, 2] = ly

    delta = query_xy[:, :, None, :] - module_centers[:, None, :, :]
    if is_multi:
        lengths = torch.tensor([lx, ly])
        delta = torch.remainder(delta + 0.5 * lengths, lengths) - 0.5 * lengths
    dist2 = delta.square().sum(dim=-1).clamp_min(1e-6)
    weights = torch.exp(-dist2 / (0.08 if is_multi else 1.0)) * module_present[:, None, :]
    influence = weights.sum(dim=-1)
    phase = torch.zeros_like(influence) if query_time is None else torch.sin(2.0 * torch.pi * query_time[..., 0])
    target = torch.stack(
        [
            torch.sin(query_xy[..., 0]) + 0.1 * influence,
            torch.cos(query_xy[..., 1]) - 0.05 * influence,
            0.2 * influence,
            phase + 0.1 * influence,
            0.5 + influence,
        ],
        dim=-1,
    )

    return BatchData(
        module_centers=module_centers,
        module_present=module_present,
        module_features=module_features,
        global_context=global_context,
        query_xy=query_xy,
        query_time=query_time,
        target_field=target,
        case_name="multicylinder" if is_multi else "channelthermal",
        metadata={"synthetic": True, "domain_length_x": lx, "domain_length_y": ly},
    )


class _BaseAdapter:
    """Shared loader helpers for HDF5-backed demo adapters."""

    def __init__(self, dataset_path: str, case_name: str):
        self.dataset_path = Path(dataset_path)
        self.case_name = case_name

    def load_one_batch(self, batch_size: int = 1, points_per_case: int = 256) -> BatchData:
        if not self.dataset_path.exists():
            print(f"[adapter] Dataset not found: {self.dataset_path}. Using synthetic smoke batch.")
            return make_synthetic_batch(self.case_name, batch_size=batch_size, points_per_case=points_per_case)
        try:
            import h5py  # type: ignore
        except ImportError:
            print("[adapter] h5py is not installed. Using synthetic smoke batch.")
            return make_synthetic_batch(self.case_name, batch_size=batch_size, points_per_case=points_per_case)

        try:
            with h5py.File(self.dataset_path, "r") as handle:
                arrays = self._read_candidate_arrays(handle)
        except Exception as exc:
            print(f"[adapter] Could not read {self.dataset_path}: {exc}. Using synthetic smoke batch.")
            return make_synthetic_batch(self.case_name, batch_size=batch_size, points_per_case=points_per_case)

        batch = self._arrays_to_batch(arrays, batch_size=batch_size, points_per_case=points_per_case)
        if batch is None:
            print("[adapter] HDF5 layout was not recognized. Using synthetic smoke batch.")
            return make_synthetic_batch(self.case_name, batch_size=batch_size, points_per_case=points_per_case)
        return batch

    def _read_candidate_arrays(self, handle: object) -> Dict[str, torch.Tensor]:
        arrays: Dict[str, torch.Tensor] = {}

        def visit(name: str, obj: object) -> None:
            shape = getattr(obj, "shape", None)
            if shape is None or len(shape) == 0:
                return
            low = name.lower()
            wanted = any(
                key in low
                for key in ["module", "center", "query", "point", "field", "target", "context", "tau", "time"]
            )
            if not wanted:
                return
            try:
                data = obj[()]  # type: ignore[index]
            except Exception:
                return
            if hasattr(data, "dtype") and str(data.dtype).startswith(("float", "int", "bool")):
                tensor = torch.as_tensor(data)
                if tensor.numel() > 0 and tensor.ndim <= 4:
                    arrays[name] = tensor

        handle.visititems(visit)  # type: ignore[attr-defined]
        return arrays

    def _arrays_to_batch(
        self,
        arrays: Dict[str, torch.Tensor],
        batch_size: int,
        points_per_case: int,
    ) -> Optional[BatchData]:
        centers = self._find_last_dim(arrays, ["center", "module"], 2)
        query = self._find_last_dim(arrays, ["query", "point", "xy", "coord"], 2)
        target = self._find_last_dim(arrays, ["field", "target", "state"], 5)
        if centers is None or query is None:
            return None

        centers = centers.float()
        if centers.ndim == 2:
            centers = centers.unsqueeze(0)
        centers = centers[:batch_size]
        query = query.float()
        if query.ndim == 2:
            query = query.unsqueeze(0)
        query = query[:batch_size, :points_per_case, :2]
        bsz, modules, _ = centers.shape
        present = torch.isfinite(centers).all(dim=-1).float()
        centers = torch.nan_to_num(centers)
        features = torch.zeros(bsz, modules, 8)
        features[..., 0:2] = centers
        features[..., 2] = 1.0
        global_context = torch.zeros(bsz, 8)

        target_field = None
        if target is not None:
            target = target.float()
            if target.ndim == 2:
                target = target.unsqueeze(0)
            target_field = target[:bsz, : query.shape[1], :5]

        return BatchData(
            module_centers=centers,
            module_present=present,
            module_features=features,
            global_context=global_context,
            query_xy=query,
            query_time=None,
            target_field=target_field,
            case_name=self.case_name,
            metadata={"synthetic": False, "source": str(self.dataset_path), "candidate_arrays": sorted(arrays.keys())[:20]},
        )

    @staticmethod
    def _find_last_dim(arrays: Dict[str, torch.Tensor], name_parts: list[str], dim: int) -> Optional[torch.Tensor]:
        for name, tensor in arrays.items():
            low = name.lower()
            if all(part in low for part in name_parts) and tensor.ndim >= 2 and tensor.shape[-1] >= dim:
                return tensor
        for name, tensor in arrays.items():
            low = name.lower()
            if any(part in low for part in name_parts) and tensor.ndim >= 2 and tensor.shape[-1] >= dim:
                return tensor
        return None


class ChannelThermalAdapter(_BaseAdapter):
    """Adapter for the existing ChannelThermal packed dataset."""

    def __init__(self, dataset_path: str = DEFAULT_CHANNEL_PATH):
        super().__init__(dataset_path, "channelthermal")


class MultiCylinderAdapter(_BaseAdapter):
    """Adapter for the existing MultiCylinder packed dataset."""

    def __init__(self, dataset_path: str = DEFAULT_MULTICYLINDER_PATH):
        super().__init__(dataset_path, "multicylinder")


def _shape(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None
