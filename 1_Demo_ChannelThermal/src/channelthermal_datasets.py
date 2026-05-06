from __future__ import annotations

"""Dataset utilities for Demo 1 Channel Thermal forward models.

The two dataset classes mirror the packed HDF5 files written by the Demo 1
preprocessors. They keep condition arrays and target arrays separate so Stage A
and Stage B training cannot accidentally consume solved interface targets as
inputs.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import h5py
import numpy as np
from torch.utils.data import Dataset

from channelthermal_model_utils import decode_string_array, resolve_demo_path, safe_std_np


LOCAL_DATASET_PATH = "./Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5"
GLOBAL_DATASET_PATH = "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"
CHANNEL_ORDER = ("u", "v", "p", "omega", "temperature")
LOCAL_PORT_INPUT_FEATURE_NAMES = ("theta", "cos_theta", "sin_theta", "T_env", "h")
LOCAL_INTERFACE_TARGET_NAMES = ("T_surface", "q_normal")
GLOBAL_INTERFACE_CONDITION_FEATURE_NAMES = (
    "theta",
    "normal_x",
    "normal_y",
    "T_outside",
    "u_normal",
    "u_tangent",
    "h_proxy",
)
GLOBAL_INTERFACE_TARGET_NAMES = ("T_surface", "q_normal")


def _decode_scalar_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _select_indices(splits: Sequence[str], split: str) -> List[int]:
    split = str(split).lower()
    if split == "all":
        return list(range(len(splits)))
    return [idx for idx, item in enumerate(splits) if str(item).lower() == split]


def _local_disk_query_points(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    size_y, size_x = mask.shape
    xi = np.linspace(-1.0, 1.0, size_x, dtype=np.float32)
    eta = np.linspace(-1.0, 1.0, size_y, dtype=np.float32)
    xx, yy = np.meshgrid(xi, eta)
    return np.stack([xx[mask], yy[mask]], axis=-1).astype(np.float32)


def _read_case_config(group: h5py.Group) -> Dict[str, Any]:
    if "case_config_json" not in group:
        return {}
    raw = group["case_config_json"][()]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _feature_indices(names: Sequence[str], requested: Sequence[str], fallback: Sequence[int]) -> List[int]:
    if names:
        name_to_idx = {str(name): idx for idx, name in enumerate(names)}
        missing = [name for name in requested if name not in name_to_idx]
        if not missing:
            return [name_to_idx[name] for name in requested]
    return [int(idx) for idx in fallback]


class H5Normalizer:
    """Small normalization adapter around the HDF5 ``normalization`` group."""

    def __init__(self, stats: Optional[Dict[str, np.ndarray]] = None):
        self.stats = stats or {}

    @classmethod
    def from_h5(cls, h5_file: h5py.File) -> "H5Normalizer":
        if "normalization" not in h5_file:
            return cls({})
        group = h5_file["normalization"]
        stats = {name: np.asarray(group[name][...], dtype=np.float32) for name in group.keys()}
        return cls(stats)

    def has(self, mean_name: str, std_name: str) -> bool:
        return mean_name in self.stats and std_name in self.stats

    def normalize(self, values: np.ndarray, mean_name: str, std_name: str) -> np.ndarray:
        if not self.has(mean_name, std_name):
            return values.astype(np.float32)
        mean = self.stats[mean_name].astype(np.float32)
        std = safe_std_np(self.stats[std_name])
        return ((values.astype(np.float32) - mean) / std).astype(np.float32)

    def denormalize(self, values: np.ndarray, mean_name: str, std_name: str) -> np.ndarray:
        if not self.has(mean_name, std_name):
            return values.astype(np.float32)
        mean = self.stats[mean_name].astype(np.float32)
        std = safe_std_np(self.stats[std_name])
        return (values.astype(np.float32) * std + mean).astype(np.float32)

    def normalize_fields(self, values: np.ndarray) -> np.ndarray:
        return self.normalize(values, "field_mean_by_channel", "field_std_by_channel")

    def denormalize_fields(self, values: np.ndarray) -> np.ndarray:
        return self.denormalize(values, "field_mean_by_channel", "field_std_by_channel")

    def normalize_module_params(self, values: np.ndarray) -> np.ndarray:
        return self.normalize(values, "module_params_mean", "module_params_std")

    def denormalize_module_params(self, values: np.ndarray) -> np.ndarray:
        return self.denormalize(values, "module_params_mean", "module_params_std")

    def normalize_port_tokens(self, values: np.ndarray) -> np.ndarray:
        return self.normalize(values, "port_tokens_mean", "port_tokens_std")

    def denormalize_port_tokens(self, values: np.ndarray) -> np.ndarray:
        return self.denormalize(values, "port_tokens_mean", "port_tokens_std")

    def normalize_interface_targets(self, values: np.ndarray) -> np.ndarray:
        if self.has("interface_targets_mean", "interface_targets_std"):
            return self.normalize(values, "interface_targets_mean", "interface_targets_std")
        return self.normalize(values, "interface_target_mean", "interface_target_std")

    def denormalize_interface_targets(self, values: np.ndarray) -> np.ndarray:
        if self.has("interface_targets_mean", "interface_targets_std"):
            return self.denormalize(values, "interface_targets_mean", "interface_targets_std")
        return self.denormalize(values, "interface_target_mean", "interface_target_std")

    def normalize_interface_condition(self, values: np.ndarray) -> np.ndarray:
        return self.normalize(values, "interface_condition_mean", "interface_condition_std")

    def denormalize_interface_condition(self, values: np.ndarray) -> np.ndarray:
        return self.denormalize(values, "interface_condition_mean", "interface_condition_std")

    def normalize_internal_temperature(self, values: np.ndarray) -> np.ndarray:
        return self.normalize(values, "internal_temperature_mean", "internal_temperature_std")

    def denormalize_internal_temperature(self, values: np.ndarray) -> np.ndarray:
        return self.denormalize(values, "internal_temperature_mean", "internal_temperature_std")

    def normalize_heat_power(self, values: np.ndarray) -> np.ndarray:
        return self.normalize(values, "heat_power_mean", "heat_power_std")

    def denormalize_heat_power(self, values: np.ndarray) -> np.ndarray:
        return self.denormalize(values, "heat_power_mean", "heat_power_std")


class LocalModuleDataset(Dataset):
    """PyTorch dataset for Stage A local module surrogate training."""

    def __init__(
        self,
        packed_h5_path: str | Path = LOCAL_DATASET_PATH,
        *,
        split: str = "train",
        normalize_inputs: bool = False,
        normalize_targets: bool = False,
        include_grid: bool = True,
    ):
        self.path = resolve_demo_path(packed_h5_path)
        self.split = str(split)
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_targets = bool(normalize_targets)
        self.include_grid = bool(include_grid)
        self._h5: Optional[h5py.File] = None
        if not self.path.exists():
            raise FileNotFoundError(f"Local module packed dataset not found: {self.path}")
        with h5py.File(self.path, "r") as h5:
            self.case_ids = decode_string_array(h5["case_ids"][...])
            self.splits = decode_string_array(h5["splits"][...])
            self.indices = _select_indices(self.splits, self.split)
            self.normalizer = H5Normalizer.from_h5(h5)
            self.module_param_names = decode_string_array(h5.get("module_param_names", np.asarray([], dtype="S"))[...])
            self.port_input_feature_names = decode_string_array(
                h5.get("port_input_feature_names", np.asarray([], dtype="S"))[...]
            )
            self.interface_target_names = decode_string_array(
                h5.get("interface_target_names", np.asarray([], dtype="S"))[...]
            )
            self.module_param_dim = int(h5["module_params"].shape[-1])
            self.port_token_dim = int(h5["port_tokens"].shape[-1])
            self.interface_target_dim = int(h5["interface_targets"].shape[-1])
            self.n_interface_points = int(h5["port_tokens"].shape[1])
            self.num_internal_points = int(h5["internal_query_points"].shape[1])

    def __len__(self) -> int:
        return len(self.indices)

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

    def __getitem__(self, item: int) -> Dict[str, Any]:
        idx = self.indices[int(item)]
        h5 = self.h5
        module_params = h5["module_params"][idx].astype(np.float32)
        port_tokens = h5["port_tokens"][idx].astype(np.float32)
        internal_query_points = h5["internal_query_points"][idx].astype(np.float32)
        internal_temperature_targets = h5["internal_temperature_targets"][idx].astype(np.float32)[..., None]
        interface_targets = h5["interface_targets"][idx].astype(np.float32)

        if self.normalize_inputs:
            module_params = self.normalizer.normalize_module_params(module_params)
            port_tokens = self.normalizer.normalize_port_tokens(port_tokens)
        if self.normalize_targets:
            internal_temperature_targets = self.normalizer.normalize_internal_temperature(internal_temperature_targets)
            interface_targets = self.normalizer.normalize_interface_targets(interface_targets)

        sample: Dict[str, Any] = {
            "module_params": module_params,
            "port_tokens": port_tokens,
            "internal_query_points": internal_query_points,
            "internal_temperature_targets": internal_temperature_targets,
            "interface_targets": interface_targets,
            "case_id": self.case_ids[idx],
        }
        if self.include_grid and "local_grid" in h5 and "local_mask" in h5:
            sample["local_grid"] = h5["local_grid"][idx].astype(np.float32)
            sample["local_mask"] = h5["local_mask"][idx].astype(np.float32)
        return sample


class GlobalChannelThermalDataset(Dataset):
    """PyTorch dataset for Stage B global channel thermal point training."""

    def __init__(
        self,
        packed_h5_path: str | Path = GLOBAL_DATASET_PATH,
        *,
        split: str = "train",
        points_per_case: int | None = 4096,
        normalize_inputs: bool = False,
        normalize_targets: bool = False,
        random_point_sampling: bool = True,
        seed: int = 123,
        include_grid: bool = False,
    ):
        self.path = resolve_demo_path(packed_h5_path)
        self.split = str(split)
        self.points_per_case = points_per_case
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_targets = bool(normalize_targets)
        self.random_point_sampling = bool(random_point_sampling)
        self.seed = int(seed)
        self.include_grid = bool(include_grid)
        self._h5: Optional[h5py.File] = None
        if not self.path.exists():
            raise FileNotFoundError(f"Global channel thermal packed dataset not found: {self.path}")
        with h5py.File(self.path, "r") as h5:
            self.normalizer = H5Normalizer.from_h5(h5)
            self.channel_order = decode_string_array(h5.get("channel_order", np.asarray(CHANNEL_ORDER, dtype="S"))[...])
            self.interface_condition_feature_names = decode_string_array(
                h5.get("interface_condition_feature_names", np.asarray(GLOBAL_INTERFACE_CONDITION_FEATURE_NAMES, dtype="S"))[...]
            )
            self.interface_target_names = decode_string_array(
                h5.get("interface_target_names", np.asarray(GLOBAL_INTERFACE_TARGET_NAMES, dtype="S"))[...]
            )
            if "case_ids" in h5 and "splits" in h5:
                root_case_ids = decode_string_array(h5["case_ids"][...])
                root_splits = decode_string_array(h5["splits"][...])
            else:
                root_case_ids = sorted(h5["cases"].keys())
                root_splits = [_decode_scalar_string(h5["cases"][key].attrs.get("split", "all")) for key in root_case_ids]
            self.case_ids = root_case_ids
            self.splits = root_splits
            root_indices = _select_indices(self.splits, self.split)
            self.indices = [root_indices[idx] for idx in range(len(root_indices))]
            self.selected_case_ids = [self.case_ids[idx] for idx in self.indices]
            self.field_dim = int(h5.attrs.get("field_dim", len(self.channel_order)))
            self.max_num_modules = int(h5.attrs.get("max_modules", 0))
            self.n_interface_points = int(h5.attrs.get("n_interface_points", 0))
            self.local_grid_size = int(h5.attrs.get("local_grid_size", 0))
            self.material_param_dim = 6
            self.target_mode = _decode_scalar_string(h5.attrs.get("target_mode", "unknown"))
            self.require_converged = bool(h5.attrs.get("require_converged", False))
            converged_by_case: Dict[str, bool] = {}
            for case_id in self.case_ids:
                group = h5["cases"][case_id]
                if "converged" in group.attrs:
                    converged_by_case[case_id] = bool(group.attrs["converged"])
                else:
                    cfg = _read_case_config(group)
                    runtime = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
                    converged_by_case[case_id] = bool(runtime.get("converged", False))
            self.converged_by_case = converged_by_case
            self.selected_converged_flags = [self.converged_by_case.get(case_id, False) for case_id in self.selected_case_ids]
            self.num_selected_converged = int(sum(1 for flag in self.selected_converged_flags if flag))
            self.num_selected_unconverged = int(len(self.selected_converged_flags) - self.num_selected_converged)

    def __len__(self) -> int:
        return len(self.selected_case_ids)

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

    def _choose_points(self, samples: np.ndarray, item: int) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if self.points_per_case is None or int(self.points_per_case) <= 0 or samples.shape[0] <= int(self.points_per_case):
            return samples
        count = int(self.points_per_case)
        if self.random_point_sampling:
            rng = np.random.default_rng()
            indices = rng.choice(samples.shape[0], size=count, replace=False)
        else:
            rng = np.random.default_rng(self.seed + int(item) * 104729)
            indices = rng.choice(samples.shape[0], size=count, replace=False)
        return samples[indices]

    def _material_params(self, group: h5py.Group) -> np.ndarray:
        materials = group.get("material_parameters", None)
        if materials is None:
            return np.zeros((6,), dtype=np.float32)
        values = [
            float(materials.attrs.get("nu", 0.0)),
            float(materials.attrs.get("solid_alpha", 0.0)),
            float(materials.attrs.get("fluid_alpha", 0.0)),
            float(materials.attrs.get("solid_k", 0.0)),
            float(materials.attrs.get("fluid_k", 0.0)),
            float(materials.attrs.get("module_radius", 0.0)),
        ]
        return np.asarray(values, dtype=np.float32)

    def _domain_lengths(self, group: h5py.Group) -> tuple[float, float]:
        cfg = _read_case_config(group)
        domain = cfg.get("domain", {}) if isinstance(cfg, dict) else {}
        lx = float(domain.get("lx", np.max(group["x_grid"][...]) if "x_grid" in group else 1.0))
        ly = float(domain.get("ly", np.max(group["y_grid"][...]) if "y_grid" in group else 1.0))
        return lx, ly

    def _teacher_port_tokens(self, interface_condition: np.ndarray) -> np.ndarray:
        requested = ("theta", "normal_x", "normal_y", "T_outside", "h_proxy")
        indices = _feature_indices(self.interface_condition_feature_names, requested, (0, 1, 2, 3, 6))
        return interface_condition[..., indices].astype(np.float32)

    def _local_module_params(
        self,
        heat_powers: np.ndarray,
        interface_condition: np.ndarray,
        material_params: np.ndarray,
        module_present: np.ndarray,
    ) -> np.ndarray:
        t_idx = _feature_indices(self.interface_condition_feature_names, ("T_outside",), (3,))[0]
        h_idx = _feature_indices(self.interface_condition_feature_names, ("h_proxy",), (6,))[0]
        t_out = interface_condition[..., t_idx]
        h_proxy = interface_condition[..., h_idx]
        local_params = np.zeros((heat_powers.shape[0], 7), dtype=np.float32)
        local_params[:, 0] = heat_powers.astype(np.float32)
        local_params[:, 1] = float(material_params[3]) if material_params.shape[0] > 3 else 0.0
        local_params[:, 2] = float(material_params[1]) if material_params.shape[0] > 1 else 0.0
        local_params[:, 3] = np.mean(h_proxy, axis=-1)
        local_params[:, 4] = np.std(h_proxy, axis=-1)
        local_params[:, 5] = np.mean(t_out, axis=-1)
        local_params[:, 6] = np.std(t_out, axis=-1)
        local_params *= module_present[:, None].astype(np.float32)
        return local_params

    def __getitem__(self, item: int) -> Dict[str, Any]:
        case_id = self.selected_case_ids[int(item)]
        group = self.h5["cases"][case_id]
        samples = self._choose_points(group["sampled_points"][...], item)
        query_xy = samples[:, 0:2].astype(np.float32)
        field_targets = samples[:, 2 : 2 + self.field_dim].astype(np.float32)
        module_centers = group["module_centers"][...].astype(np.float32)
        heat_powers = group["heat_powers"][...].astype(np.float32)
        module_present = group["module_present"][...].astype(np.float32)
        interface_condition = group["interface_condition"][...].astype(np.float32)
        interface_target = group["interface_target"][...].astype(np.float32)
        internal_grid = group["module_internal_temperature"][...].astype(np.float32)
        internal_mask = group["module_internal_mask"][...].astype(np.float32)
        local_query_points = _local_disk_query_points(internal_mask)
        internal_points = internal_grid[:, internal_mask.astype(bool)].astype(np.float32)
        material_params = self._material_params(group)
        lx, ly = self._domain_lengths(group)

        teacher_port_tokens = self._teacher_port_tokens(interface_condition)
        local_module_params = self._local_module_params(heat_powers, interface_condition, material_params, module_present)

        if self.normalize_inputs:
            heat_powers = self.normalizer.normalize_heat_power(heat_powers)
            interface_condition = self.normalizer.normalize_interface_condition(interface_condition)
        if self.normalize_targets:
            field_targets = self.normalizer.normalize_fields(field_targets)
            interface_target = self.normalizer.normalize_interface_targets(interface_target)
            internal_grid = self.normalizer.normalize_internal_temperature(internal_grid)
            internal_points = self.normalizer.normalize_internal_temperature(internal_points)

        sample: Dict[str, Any] = {
            "structure": {
                "re": np.asarray([float(group["material_parameters"].attrs.get("re", 0.0))], dtype=np.float32)
                if "material_parameters" in group
                else np.asarray([0.0], dtype=np.float32),
                "u_in": np.asarray([float(group["material_parameters"].attrs.get("u_in", 0.0))], dtype=np.float32)
                if "material_parameters" in group
                else np.asarray([0.0], dtype=np.float32),
                "module_centers": module_centers,
                "heat_powers": heat_powers,
                "module_present": module_present,
                "material_params": material_params,
                "domain_length_x": np.asarray([lx], dtype=np.float32),
                "domain_length_y": np.asarray([ly], dtype=np.float32),
            },
            "query_xy": query_xy,
            "field_targets": field_targets,
            "module_internal_temperature": internal_grid,
            "module_internal_temperature_points": internal_points,
            "module_internal_mask": internal_mask,
            "module_internal_query_points": local_query_points,
            "interface_condition": interface_condition,
            "interface_target": interface_target,
            "teacher_port_tokens": teacher_port_tokens,
            "local_module_params": local_module_params,
            "case_id": case_id,
        }
        if self.include_grid:
            sample["x_grid"] = group["x_grid"][...].astype(np.float32)
            sample["y_grid"] = group["y_grid"][...].astype(np.float32)
            sample["steady_field"] = group["steady_field"][...].astype(np.float32)
            sample["rms_field"] = group["rms_field"][...].astype(np.float32) if "rms_field" in group else np.zeros_like(sample["steady_field"])
        return sample
