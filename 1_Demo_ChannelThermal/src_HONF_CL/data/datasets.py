"""Packed-HDF5 datasets for ChannelThermal global and local training.

The readers expose three task views over the existing processed data:

* :class:`LocalModuleDataset` reads standalone Stage-A module samples.
* :class:`GlobalModuleAlignmentDataset` converts active modules in global
  cases into Stage-A samples.
* :class:`GlobalChannelThermalDataset` provides sampled global field queries
  and the local/interface data needed by the coupled HONF model.

All normalizers can be injected. Training fits a transform from the training
sources and reuses it for validation and inference; this prevents two sources
or two splits from assigning different physical meanings to the same
normalized value.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import h5py
import numpy as np
from torch.utils.data import Dataset

from common.runtime import decode_string_array, resolve_demo_path, safe_std_np


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
    "h_effective",
)
GLOBAL_INTERFACE_TARGET_NAMES = ("T_surface", "q_normal")


def _decode_scalar_string(value: Any) -> str:
    """Perform the decode scalar string operation used by this module."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _select_indices(splits: Sequence[str], split: str) -> List[int]:
    """Perform the select indices operation used by this module."""

    split = str(split).lower()
    if split == "all":
        return list(range(len(splits)))
    return [idx for idx, item in enumerate(splits) if str(item).lower() == split]


def _local_disk_query_points(mask: np.ndarray) -> np.ndarray:
    """Perform the local disk query points operation used by this module."""

    mask = np.asarray(mask).astype(bool)
    size_y, size_x = mask.shape
    xi = np.linspace(-1.0, 1.0, size_x, dtype=np.float32)
    eta = np.linspace(-1.0, 1.0, size_y, dtype=np.float32)
    xx, yy = np.meshgrid(xi, eta)
    return np.stack([xx[mask], yy[mask]], axis=-1).astype(np.float32)


def _read_case_config(group: h5py.Group) -> Dict[str, Any]:
    """Perform the read case config operation used by this module."""

    if "case_config_json" not in group:
        return {}
    raw = group["case_config_json"][()]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _feature_indices(names: Sequence[str], requested: Sequence[str], fallback: Sequence[int]) -> List[int]:
    """Perform the feature indices operation used by this module."""

    if names:
        name_to_idx = {str(name): idx for idx, name in enumerate(names)}
        missing = [name for name in requested if name not in name_to_idx]
        if not missing:
            return [name_to_idx[name] for name in requested]
    return [int(idx) for idx in fallback]


def _append_h_effective_stat_fallback(normalizer: "H5Normalizer", h_proxy_idx: int) -> None:
    """Mirror h_proxy stats for legacy packed files that lack h_effective."""
    mean = normalizer.stats.get("interface_condition_mean")
    std = normalizer.stats.get("interface_condition_std")
    if mean is not None and mean.shape[0] > h_proxy_idx and mean.shape[0] < len(GLOBAL_INTERFACE_CONDITION_FEATURE_NAMES):
        normalizer.stats["interface_condition_mean"] = np.concatenate([mean, mean[h_proxy_idx : h_proxy_idx + 1]]).astype(np.float32)
    if std is not None and std.shape[0] > h_proxy_idx and std.shape[0] < len(GLOBAL_INTERFACE_CONDITION_FEATURE_NAMES):
        normalizer.stats["interface_condition_std"] = np.concatenate([std, std[h_proxy_idx : h_proxy_idx + 1]]).astype(np.float32)


def _condition_with_h_effective_fallback(
    interface_condition: np.ndarray,
    interface_names: Sequence[str],
) -> np.ndarray:
    """Append h_proxy as h_effective for legacy arrays whose metadata was upgraded on load."""
    if "h_effective" not in interface_names:
        return interface_condition.astype(np.float32)
    h_effective_idx = list(interface_names).index("h_effective")
    if interface_condition.shape[-1] > h_effective_idx:
        return interface_condition.astype(np.float32)
    h_proxy_idx = _feature_indices(interface_names, ("h_proxy",), (6,))[0]
    h_proxy = interface_condition[..., h_proxy_idx : h_proxy_idx + 1]
    return np.concatenate([interface_condition.astype(np.float32), h_proxy.astype(np.float32)], axis=-1)


def _local_h_name(interface_names: Sequence[str]) -> str:
    """Perform the local h name operation used by this module."""

    return "h_effective" if "h_effective" in interface_names else "h_proxy"


class H5Normalizer:
    """Apply named mean/std transforms stored in an HDF5 file or checkpoint.

    Values without matching statistics pass through unchanged. A checkpoint
    should be the authoritative source at inference time.
    """

    def __init__(self, stats: Optional[Dict[str, np.ndarray]] = None):
        """Initialize H5Normalizer and its required state."""

        self.stats = stats or {}

    @classmethod
    def from_h5(cls, h5_file: h5py.File) -> "H5Normalizer":
        """Perform the from h5 operation used by this module."""

        if "normalization" not in h5_file:
            return cls({})
        group = h5_file["normalization"]
        stats = {name: np.asarray(group[name][...], dtype=np.float32) for name in group.keys()}
        return cls(stats)

    def has(self, mean_name: str, std_name: str) -> bool:
        """Perform the has operation used by this module."""

        return mean_name in self.stats and std_name in self.stats

    def normalize(self, values: np.ndarray, mean_name: str, std_name: str) -> np.ndarray:
        """Perform the normalize operation used by this module."""

        if not self.has(mean_name, std_name):
            return values.astype(np.float32)
        mean = self.stats[mean_name].astype(np.float32)
        std = safe_std_np(self.stats[std_name])
        return ((values.astype(np.float32) - mean) / std).astype(np.float32)

    def denormalize(self, values: np.ndarray, mean_name: str, std_name: str) -> np.ndarray:
        """Perform the denormalize operation used by this module."""

        if not self.has(mean_name, std_name):
            return values.astype(np.float32)
        mean = self.stats[mean_name].astype(np.float32)
        std = safe_std_np(self.stats[std_name])
        return (values.astype(np.float32) * std + mean).astype(np.float32)

    def normalize_fields(self, values: np.ndarray) -> np.ndarray:
        """Normalize fields."""

        return self.normalize(values, "field_mean_by_channel", "field_std_by_channel")

    def denormalize_fields(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized fields."""

        return self.denormalize(values, "field_mean_by_channel", "field_std_by_channel")

    def normalize_module_params(self, values: np.ndarray) -> np.ndarray:
        """Normalize module params."""

        return self.normalize(values, "module_params_mean", "module_params_std")

    def denormalize_module_params(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized module params."""

        return self.denormalize(values, "module_params_mean", "module_params_std")

    def normalize_port_tokens(self, values: np.ndarray) -> np.ndarray:
        """Normalize port tokens."""

        return self.normalize(values, "port_tokens_mean", "port_tokens_std")

    def denormalize_port_tokens(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized port tokens."""

        return self.denormalize(values, "port_tokens_mean", "port_tokens_std")

    def normalize_interface_targets(self, values: np.ndarray) -> np.ndarray:
        """Normalize interface targets."""

        if self.has("interface_targets_mean", "interface_targets_std"):
            return self.normalize(values, "interface_targets_mean", "interface_targets_std")
        return self.normalize(values, "interface_target_mean", "interface_target_std")

    def denormalize_interface_targets(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized interface targets."""

        if self.has("interface_targets_mean", "interface_targets_std"):
            return self.denormalize(values, "interface_targets_mean", "interface_targets_std")
        return self.denormalize(values, "interface_target_mean", "interface_target_std")

    def normalize_interface_condition(self, values: np.ndarray) -> np.ndarray:
        """Normalize interface condition."""

        return self.normalize(values, "interface_condition_mean", "interface_condition_std")

    def denormalize_interface_condition(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized interface condition."""

        return self.denormalize(values, "interface_condition_mean", "interface_condition_std")

    def normalize_internal_temperature(self, values: np.ndarray) -> np.ndarray:
        """Normalize internal temperature."""

        return self.normalize(values, "internal_temperature_mean", "internal_temperature_std")

    def denormalize_internal_temperature(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized internal temperature."""

        return self.denormalize(values, "internal_temperature_mean", "internal_temperature_std")

    def normalize_heat_power(self, values: np.ndarray) -> np.ndarray:
        """Normalize heat power."""

        return self.normalize(values, "heat_power_mean", "heat_power_std")

    def denormalize_heat_power(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized heat power."""

        return self.denormalize(values, "heat_power_mean", "heat_power_std")


class _RunningFeatureMoments:
    """Accumulate per-feature mean and standard deviation without caching data."""

    def __init__(self, width: int):
        """Initialize _RunningFeatureMoments and its required state."""

        self.width = int(width)
        self.count = 0
        self.total = np.zeros((self.width,), dtype=np.float64)
        self.total_square = np.zeros((self.width,), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        """Perform the update operation used by this module."""

        rows = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        if rows.size == 0:
            return
        finite = np.all(np.isfinite(rows), axis=-1)
        rows = rows[finite]
        if rows.size == 0:
            return
        self.count += int(rows.shape[0])
        self.total += rows.sum(axis=0)
        self.total_square += np.square(rows).sum(axis=0)

    def mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        """Perform the mean std operation used by this module."""

        if self.count == 0:
            return np.zeros((self.width,), dtype=np.float32), np.ones((self.width,), dtype=np.float32)
        mean = self.total / float(self.count)
        variance = np.maximum(self.total_square / float(self.count) - np.square(mean), 0.0)
        return mean.astype(np.float32), safe_std_np(np.sqrt(variance).astype(np.float32))


def fit_local_normalizer(datasets: Iterable[Dataset]) -> H5Normalizer:
    """Fit one Stage-A normalizer over raw training samples from all sources.

    Each input dataset must yield physical (not already normalized) samples.
    The returned transform covers the four Stage-A tensor groups and can be
    injected into local, global-alignment, validation, and evaluation readers.
    """

    module_moments = _RunningFeatureMoments(7)
    port_moments = _RunningFeatureMoments(5)
    internal_moments = _RunningFeatureMoments(1)
    interface_moments = _RunningFeatureMoments(2)
    for dataset in datasets:
        for item in range(len(dataset)):
            sample = dataset[item]
            module_moments.update(sample["module_params"])
            port_moments.update(sample["port_tokens"])
            internal_moments.update(sample["internal_temperature_targets"])
            interface_moments.update(sample["interface_targets"])
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    module_mean, module_std = module_moments.mean_std()
    port_mean, port_std = port_moments.mean_std()
    internal_mean, internal_std = internal_moments.mean_std()
    interface_mean, interface_std = interface_moments.mean_std()
    return H5Normalizer(
        {
            "module_params_mean": module_mean,
            "module_params_std": module_std,
            "port_tokens_mean": port_mean,
            "port_tokens_std": port_std,
            "internal_temperature_mean": internal_mean,
            "internal_temperature_std": internal_std,
            "interface_targets_mean": interface_mean,
            "interface_targets_std": interface_std,
        }
    )


class LocalModuleDataset(Dataset):
    """Read Stage-A module samples from the standalone local packed file."""

    def __init__(
        self,
        packed_h5_path: str | Path = LOCAL_DATASET_PATH,
        *,
        split: str = "train",
        normalize_inputs: bool = False,
        normalize_targets: bool = False,
        include_grid: bool = False,
        normalizer: Optional[H5Normalizer] = None,
    ):
        """Initialize LocalModuleDataset and its required state."""

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
            packed_normalizer = H5Normalizer.from_h5(h5)
            self.normalizer = normalizer if normalizer is not None else packed_normalizer
            self.module_param_names = decode_string_array(h5.get("module_param_names", np.asarray([], dtype="S"))[...])
            self.port_input_feature_names = decode_string_array(
                h5.get("port_input_feature_names", np.asarray([], dtype="S"))[...]
            )
            self.interface_target_names = decode_string_array(
                h5.get("interface_target_names", np.asarray([], dtype="S"))[...]
            )
            self.local_target_roughness_names = decode_string_array(
                h5.get("local_target_roughness_names", np.asarray([], dtype="S"))[...]
            )
            self.solver_types = decode_string_array(h5.get("solver_type", np.asarray(["unknown"] * len(self.case_ids), dtype="S"))[...])
            self.n_active_modes_all = (
                np.asarray(h5["n_active_modes"][...], dtype=np.int32)
                if "n_active_modes" in h5
                else np.full((len(self.case_ids),), -1, dtype=np.int32)
            )
            self.interface_targets_smoothed = bool(h5.attrs.get("interface_targets_smoothed", False))
            self.module_param_dim = int(h5["module_params"].shape[-1])
            self.port_token_dim = int(h5["port_tokens"].shape[-1])
            self.interface_target_dim = int(h5["interface_targets"].shape[-1])
            self.n_interface_points = int(h5["port_tokens"].shape[1])
            self.num_internal_points = int(h5["internal_query_points"].shape[1])

    def __len__(self) -> int:
        """Return the number of samples exposed by this dataset."""

        return len(self.indices)

    def __getstate__(self) -> Dict[str, Any]:
        """Return worker-safe state without an open HDF5 handle."""

        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    @property
    def h5(self) -> h5py.File:
        """Perform the h5 operation used by this module."""

        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def close(self) -> None:
        """Perform the close operation used by this module."""

        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Load and transform one indexed dataset sample."""

        idx = self.indices[int(item)]
        h5 = self.h5
        module_params = h5["module_params"][idx].astype(np.float32)
        port_tokens = h5["port_tokens"][idx].astype(np.float32)
        internal_query_points = h5["internal_query_points"][idx].astype(np.float32)
        internal_temperature_targets = h5["internal_temperature_targets"][idx].astype(np.float32)[..., None]
        interface_targets = h5["interface_targets"][idx].astype(np.float32)
        interface_targets_raw = h5["interface_targets_raw"][idx].astype(np.float32) if "interface_targets_raw" in h5 else None
        local_target_roughness = (
            h5["local_target_roughness"][idx].astype(np.float32)
            if "local_target_roughness" in h5
            else np.zeros((4,), dtype=np.float32)
        )

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
            **({"interface_targets_raw": interface_targets_raw} if interface_targets_raw is not None else {}),
            "local_target_roughness": local_target_roughness,
            "solver_type": self.solver_types[idx] if idx < len(self.solver_types) else "unknown",
            "n_active_modes": np.asarray([self.n_active_modes_all[idx]], dtype=np.int32),
            "case_id": self.case_ids[idx],
        }
        if self.include_grid and "local_grid" in h5 and "local_mask" in h5:
            sample["local_grid"] = h5["local_grid"][idx].astype(np.float32)
            sample["local_mask"] = h5["local_mask"][idx].astype(np.float32)
        return sample


class GlobalModuleAlignmentDataset(Dataset):
    """View processed global channel data as local-surrogate alignment samples.

    Each valid module in each selected global case becomes one Stage-A sample
    with local module parameters, local-format port tokens, internal
    temperature targets, and interface targets. This uses existing processed
    global data only; it does not require a new global preprocessing step.
    """

    def __init__(
        self,
        packed_h5_path: str | Path = GLOBAL_DATASET_PATH,
        *,
        split: str = "train",
        normalize_inputs: bool = False,
        normalize_targets: bool = False,
        include_grid: bool = False,
        normalizer: Optional[H5Normalizer] = None,
    ):
        """Initialize GlobalModuleAlignmentDataset and its required state."""

        self.path = resolve_demo_path(packed_h5_path)
        self.split = str(split)
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_targets = bool(normalize_targets)
        self.include_grid = bool(include_grid)
        self._h5: Optional[h5py.File] = None
        if not self.path.exists():
            raise FileNotFoundError(f"Global channel thermal packed dataset not found: {self.path}")
        self.module_param_names = ["q_internal", "solid_k", "solid_alpha", "h_mean", "h_std", "T_env_mean", "T_env_std"]
        self.port_input_feature_names = list(LOCAL_PORT_INPUT_FEATURE_NAMES)
        self.interface_target_names = list(LOCAL_INTERFACE_TARGET_NAMES)
        self.local_target_roughness_names = []
        with h5py.File(self.path, "r") as h5:
            if "case_ids" in h5 and "splits" in h5:
                case_ids = decode_string_array(h5["case_ids"][...])
                splits = decode_string_array(h5["splits"][...])
            else:
                case_ids = sorted(h5["cases"].keys())
                splits = [_decode_scalar_string(h5["cases"][key].attrs.get("split", "all")) for key in case_ids]
            root_indices = _select_indices(splits, self.split)
            self.case_ids = case_ids
            self.records: List[tuple[str, int]] = []
            for root_idx in root_indices:
                case_id = case_ids[root_idx]
                group = h5["cases"][case_id]
                present = group["module_present"][...].astype(np.float32)
                for module_idx in np.flatnonzero(present > 0.5):
                    self.records.append((case_id, int(module_idx)))
            if self.records:
                sample_group = h5["cases"][self.records[0][0]]
                self.n_interface_points = int(sample_group["interface_condition"].shape[1])
                mask = sample_group["module_internal_mask"][...].astype(np.float32)
                self.num_internal_points = int(np.sum(mask.astype(bool)))
            else:
                self.n_interface_points = int(h5.attrs.get("n_interface_points", 0))
                self.num_internal_points = 0
            self.module_param_dim = 7
            self.port_token_dim = 5
            self.interface_target_dim = 2
            self.normalizer = normalizer if normalizer is not None else self._compute_normalizer(h5)

    def _material_params(self, group: h5py.Group) -> np.ndarray:
        """Perform the material params operation used by this module."""

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
                float(materials.attrs.get("module_radius", 0.0)),
            ],
            dtype=np.float32,
        )

    def _module_params_for(
        self,
        heat_power: float,
        interface_condition: np.ndarray,
        material_params: np.ndarray,
        interface_names: Sequence[str],
    ) -> np.ndarray:
        """Perform the module params for operation used by this module."""

        t_idx = _feature_indices(interface_names, ("T_outside",), (3,))[0]
        h_idx = _feature_indices(interface_names, (_local_h_name(interface_names),), (6,))[0]
        t_env = interface_condition[:, t_idx]
        h = interface_condition[:, h_idx]
        return np.asarray(
            [
                float(heat_power),
                float(material_params[3]) if material_params.shape[0] > 3 else 0.0,
                float(material_params[1]) if material_params.shape[0] > 1 else 0.0,
                float(np.mean(h)),
                float(np.std(h)),
                float(np.mean(t_env)),
                float(np.std(t_env)),
            ],
            dtype=np.float32,
        )

    def _port_tokens_for(self, interface_condition: np.ndarray, interface_names: Sequence[str]) -> np.ndarray:
        """Perform the port tokens for operation used by this module."""

        requested = ("theta", "normal_x", "normal_y", "T_outside", _local_h_name(interface_names))
        indices = _feature_indices(interface_names, requested, (0, 1, 2, 3, 6))
        return interface_condition[:, indices].astype(np.float32)

    def _compute_normalizer(self, h5: h5py.File) -> H5Normalizer:
        """Perform the compute normalizer operation used by this module."""

        if not self.records:
            return H5Normalizer({})
        interface_names = decode_string_array(
            h5.get("interface_condition_feature_names", np.asarray(GLOBAL_INTERFACE_CONDITION_FEATURE_NAMES, dtype="S"))[...]
        )
        if "h_effective" not in interface_names:
            interface_names = list(interface_names) + ["h_effective"]
        module_params = []
        port_tokens = []
        internal_targets = []
        interface_targets = []
        for case_id, module_idx in self.records:
            group = h5["cases"][case_id]
            material = self._material_params(group)
            cond = _condition_with_h_effective_fallback(group["interface_condition"][module_idx].astype(np.float32), interface_names)
            heat = float(group["heat_powers"][module_idx])
            module_params.append(self._module_params_for(heat, cond, material, interface_names))
            port_tokens.append(self._port_tokens_for(cond, interface_names))
            mask = group["module_internal_mask"][...].astype(bool)
            internal_targets.append(group["module_internal_temperature"][module_idx][mask].astype(np.float32).reshape(-1))
            interface_targets.append(group["interface_target"][module_idx].astype(np.float32))
        module_params_arr = np.stack(module_params)
        port_arr = np.stack(port_tokens)
        internal_arr = np.concatenate(internal_targets)
        interface_arr = np.stack(interface_targets)
        return H5Normalizer(
            {
                "module_params_mean": np.mean(module_params_arr, axis=0).astype(np.float32),
                "module_params_std": np.std(module_params_arr, axis=0).astype(np.float32),
                "port_tokens_mean": np.mean(port_arr.reshape(-1, port_arr.shape[-1]), axis=0).astype(np.float32),
                "port_tokens_std": np.std(port_arr.reshape(-1, port_arr.shape[-1]), axis=0).astype(np.float32),
                "internal_temperature_mean": np.asarray([np.mean(internal_arr)], dtype=np.float32),
                "internal_temperature_std": np.asarray([np.std(internal_arr)], dtype=np.float32),
                "interface_targets_mean": np.mean(interface_arr.reshape(-1, interface_arr.shape[-1]), axis=0).astype(np.float32),
                "interface_targets_std": np.std(interface_arr.reshape(-1, interface_arr.shape[-1]), axis=0).astype(np.float32),
            }
        )

    def __len__(self) -> int:
        """Return the number of samples exposed by this dataset."""

        return len(self.records)

    def __getstate__(self) -> Dict[str, Any]:
        """Return worker-safe state without an open HDF5 handle."""

        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    @property
    def h5(self) -> h5py.File:
        """Perform the h5 operation used by this module."""

        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def close(self) -> None:
        """Perform the close operation used by this module."""

        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Load and transform one indexed dataset sample."""

        case_id, module_idx = self.records[int(item)]
        group = self.h5["cases"][case_id]
        interface_names = decode_string_array(
            self.h5.get("interface_condition_feature_names", np.asarray(GLOBAL_INTERFACE_CONDITION_FEATURE_NAMES, dtype="S"))[...]
        )
        if "h_effective" not in interface_names:
            interface_names = list(interface_names) + ["h_effective"]
        material = self._material_params(group)
        cond = _condition_with_h_effective_fallback(group["interface_condition"][module_idx].astype(np.float32), interface_names)
        heat = float(group["heat_powers"][module_idx])
        module_params = self._module_params_for(heat, cond, material, interface_names)
        port_tokens = self._port_tokens_for(cond, interface_names)
        mask = group["module_internal_mask"][...].astype(np.float32)
        internal_query_points = _local_disk_query_points(mask)
        internal_targets = group["module_internal_temperature"][module_idx][mask.astype(bool)].astype(np.float32)[..., None]
        interface_targets = group["interface_target"][module_idx].astype(np.float32)
        if self.normalize_inputs:
            module_params = self.normalizer.normalize_module_params(module_params)
            port_tokens = self.normalizer.normalize_port_tokens(port_tokens)
        if self.normalize_targets:
            internal_targets = self.normalizer.normalize_internal_temperature(internal_targets)
            interface_targets = self.normalizer.normalize_interface_targets(interface_targets)
        sample: Dict[str, Any] = {
            "module_params": module_params.astype(np.float32),
            "port_tokens": port_tokens.astype(np.float32),
            "internal_query_points": internal_query_points.astype(np.float32),
            "internal_temperature_targets": internal_targets.astype(np.float32),
            "interface_targets": interface_targets.astype(np.float32),
            "local_target_roughness": np.zeros((4,), dtype=np.float32),
            "solver_type": "global_alignment",
            "n_active_modes": np.asarray([-1], dtype=np.int32),
            "case_id": f"{case_id}_M{module_idx}",
        }
        if self.include_grid:
            sample["local_grid"] = np.stack(
                np.meshgrid(
                    np.linspace(-1.0, 1.0, mask.shape[1], dtype=np.float32),
                    np.linspace(-1.0, 1.0, mask.shape[0], dtype=np.float32),
                ),
                axis=-1,
            )
            sample["local_mask"] = mask.astype(np.float32)
        return sample


class GlobalChannelThermalDataset(Dataset):
    """Read global cases and sample field-query points for coupled HONF.

    Random sampling is deterministic for a given ``seed``, epoch, and case.
    Call :meth:`set_epoch` before each training epoch to obtain a new but
    reproducible point subset.
    """

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
        include_structure_targets: bool = False,
        require_converged: bool = False,
        normalizer: Optional[H5Normalizer] = None,
    ):
        """Initialize GlobalChannelThermalDataset and its required state."""

        self.path = resolve_demo_path(packed_h5_path)
        self.split = str(split)
        self.points_per_case = points_per_case
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_targets = bool(normalize_targets)
        self.random_point_sampling = bool(random_point_sampling)
        self.seed = int(seed)
        self.epoch = 0
        self.include_grid = bool(include_grid)
        self.include_structure_targets = bool(include_structure_targets)
        self.require_converged = bool(require_converged)
        self._h5: Optional[h5py.File] = None
        if not self.path.exists():
            raise FileNotFoundError(f"Global channel thermal packed dataset not found: {self.path}")
        with h5py.File(self.path, "r") as h5:
            packed_normalizer = H5Normalizer.from_h5(h5)
            self.normalizer = normalizer if normalizer is not None else packed_normalizer
            self.channel_order = decode_string_array(h5.get("channel_order", np.asarray(CHANNEL_ORDER, dtype="S"))[...])
            self.interface_condition_feature_names = decode_string_array(
                h5.get("interface_condition_feature_names", np.asarray(GLOBAL_INTERFACE_CONDITION_FEATURE_NAMES, dtype="S"))[...]
            )
            self._h_effective_fallback_from_h_proxy = "h_effective" not in self.interface_condition_feature_names
            if self._h_effective_fallback_from_h_proxy:
                warnings.warn(
                    "Packed global dataset does not contain h_effective; falling back to h_proxy for local-surrogate h targets. "
                    "Re-run preprocess_channelthermal_dataset.py to store flux-consistent h_effective.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                h_proxy_idx = _feature_indices(self.interface_condition_feature_names, ("h_proxy",), (6,))[0]
                self.interface_condition_feature_names = list(self.interface_condition_feature_names) + ["h_effective"]
                _append_h_effective_stat_fallback(self.normalizer, h_proxy_idx)
            sample_case_id = next(iter(h5["cases"].keys()), None)
            self._has_interface_condition_valid_mask = (
                sample_case_id is not None and "interface_condition_valid_mask" in h5["cases"][sample_case_id]
            )
            if not self._has_interface_condition_valid_mask:
                warnings.warn(
                    "Packed global dataset does not contain interface_condition_valid_mask; using all-ones h_effective "
                    "validity masks. Re-run preprocess_channelthermal_dataset.py to store per-point h supervision masks.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._has_structure_targets = (
                sample_case_id is not None
                and "env_module_influence_target" in h5["cases"][sample_case_id]
                and "module_affinity_target" in h5["cases"][sample_case_id]
            )
            self.structure_target_num_env_tokens = int(h5.attrs.get("structure_target_num_env_tokens", 0))
            if self.include_structure_targets and not self._has_structure_targets:
                warnings.warn(
                    "Packed global dataset does not contain organizer structure targets; using geometry-only "
                    "fallback targets for structure losses. Re-run preprocess_channelthermal_dataset.py to store "
                    "training-only solved-field structure supervision.",
                    RuntimeWarning,
                    stacklevel=2,
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
            if self.require_converged:
                self.selected_case_ids = [
                    case_id for case_id in self.selected_case_ids if self.converged_by_case.get(case_id, False)
                ]
                self.indices = [self.case_ids.index(case_id) for case_id in self.selected_case_ids]
            self.selected_converged_flags = [self.converged_by_case.get(case_id, False) for case_id in self.selected_case_ids]
            self.num_selected_converged = int(sum(1 for flag in self.selected_converged_flags if flag))
            self.num_selected_unconverged = int(len(self.selected_converged_flags) - self.num_selected_converged)

    def __len__(self) -> int:
        """Return the number of samples exposed by this dataset."""

        return len(self.selected_case_ids)

    def __getstate__(self) -> Dict[str, Any]:
        """Return worker-safe state without an open HDF5 handle."""

        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    @property
    def h5(self) -> h5py.File:
        """Perform the h5 operation used by this module."""

        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def close(self) -> None:
        """Perform the close operation used by this module."""

        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic random-sampling stream for one epoch."""

        self.epoch = int(epoch)

    def _choose_point_indices(self, num_samples: int, item: int) -> np.ndarray:
        """Select reproducible point indices for this case and training epoch."""

        if self.points_per_case is None or int(self.points_per_case) <= 0 or num_samples <= int(self.points_per_case):
            return np.arange(num_samples, dtype=np.int64)
        count = int(self.points_per_case)
        if self.random_point_sampling:
            rng = np.random.default_rng(self.seed + int(item) * 104729 + self.epoch * 1000003)
            return rng.choice(num_samples, size=count, replace=False)
        rng = np.random.default_rng(self.seed + int(item) * 104729)
        return rng.choice(num_samples, size=count, replace=False)

    def _material_params(self, group: h5py.Group) -> np.ndarray:
        """Read the six ordered fluid/solid material attributes for one case."""

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
        """Read case domain lengths, falling back to packed-file defaults."""

        cfg = _read_case_config(group)
        domain = cfg.get("domain", {}) if isinstance(cfg, dict) else {}
        lx = float(domain.get("lx", np.max(group["x_grid"][...]) if "x_grid" in group else 1.0))
        ly = float(domain.get("ly", np.max(group["y_grid"][...]) if "y_grid" in group else 1.0))
        return lx, ly

    def _teacher_port_tokens(self, interface_condition: np.ndarray) -> np.ndarray:
        """Extract ``theta,cos,sin,T_env,h`` teacher port columns."""

        requested = ("theta", "normal_x", "normal_y", "T_outside", _local_h_name(self.interface_condition_feature_names))
        indices = _feature_indices(self.interface_condition_feature_names, requested, (0, 1, 2, 3, 6))
        return interface_condition[..., indices].astype(np.float32)

    def _fallback_structure_targets(
        self,
        module_centers: np.ndarray,
        heat_powers: np.ndarray,
        module_present: np.ndarray,
        lx: float,
        ly: float,
    ) -> Dict[str, np.ndarray]:
        """Derive organizer supervision when packed structure targets are absent."""

        nx = 24
        ny = 12
        xs = np.linspace(0.0, float(lx), nx, dtype=np.float32)
        ys = np.linspace(0.0, float(ly), ny, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        env_coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
        centers = np.asarray(module_centers, dtype=np.float32)
        present = np.asarray(module_present, dtype=np.float32)
        heat = np.abs(np.asarray(heat_powers, dtype=np.float32))
        heat_norm = heat / max(float(np.max(heat)) if heat.size else 0.0, 1.0e-6)
        dx = env_coords[:, None, 0] - centers[None, :, 0]
        dy = env_coords[:, None, 1] - centers[None, :, 1]
        dist = np.sqrt(dx * dx + dy * dy + 1.0e-8)
        near = np.exp(-dist / 1.5)
        plume = 0.5 * np.exp(-(np.abs(dy) / 0.9) ** 2) / (1.0 + np.exp(-np.maximum(dx, 0.0) / 2.0))
        score = (near + plume + 0.15 * heat_norm[None, :]) * present[None, :]
        row_sum = score.sum(axis=-1, keepdims=True)
        fallback = present / max(float(np.sum(present)), 1.0)
        env_module = np.where(row_sum > 1.0e-8, score / np.maximum(row_sum, 1.0e-8), fallback[None, :]).astype(np.float32)
        mdx = centers[None, :, 0] - centers[:, None, 0]
        mdy = centers[None, :, 1] - centers[:, None, 1]
        mdist = np.sqrt(mdx * mdx + mdy * mdy + 1.0e-8)
        affinity = (0.6 * np.exp(-mdist / 1.5) + 0.4 * np.exp(-(np.abs(mdy) / 0.9) ** 2)) * present[:, None] * present[None, :]
        np.fill_diagonal(affinity, present)
        affinity = affinity / np.maximum(affinity.sum(axis=-1, keepdims=True), 1.0e-8)
        counts = np.bincount(env_module.argmax(axis=-1), minlength=module_present.shape[0]).astype(np.float32)
        active = np.asarray([float(np.clip(np.sum(counts >= max(2.0, 0.04 * env_coords.shape[0])), 1.0, max(float(np.sum(present)), 1.0)))], dtype=np.float32)
        return {
            "env_token_coords": env_coords,
            "env_module_influence_target": env_module.astype(np.float32),
            "module_affinity_target": affinity.astype(np.float32),
            "env_module_target_mask": np.ones(env_module.shape, dtype=np.float32),
            "module_affinity_target_mask": (present[:, None] * present[None, :]).astype(np.float32),
            "active_edge_count_target": active,
            "has_solved_structure_targets": np.asarray([0.0], dtype=np.float32),
        }

    def _local_module_params(
        self,
        heat_powers: np.ndarray,
        interface_condition: np.ndarray,
        material_params: np.ndarray,
        module_present: np.ndarray,
    ) -> np.ndarray:
        """Assemble Stage-A ``[M,7]`` local parameters from global case arrays."""

        t_idx = _feature_indices(self.interface_condition_feature_names, ("T_outside",), (3,))[0]
        h_idx = _feature_indices(self.interface_condition_feature_names, (_local_h_name(self.interface_condition_feature_names),), (6,))[0]
        t_out = interface_condition[..., t_idx]
        h_local = interface_condition[..., h_idx]
        local_params = np.zeros((heat_powers.shape[0], 7), dtype=np.float32)
        local_params[:, 0] = heat_powers.astype(np.float32)
        local_params[:, 1] = float(material_params[3]) if material_params.shape[0] > 3 else 0.0
        local_params[:, 2] = float(material_params[1]) if material_params.shape[0] > 1 else 0.0
        local_params[:, 3] = np.mean(h_local, axis=-1)
        local_params[:, 4] = np.std(h_local, axis=-1)
        local_params[:, 5] = np.mean(t_out, axis=-1)
        local_params[:, 6] = np.std(t_out, axis=-1)
        local_params *= module_present[:, None].astype(np.float32)
        return local_params

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Load and transform one indexed dataset sample."""

        case_id = self.selected_case_ids[int(item)]
        group = self.h5["cases"][case_id]
        all_samples = group["sampled_points"][...]
        point_indices = self._choose_point_indices(all_samples.shape[0], item)
        samples = np.asarray(all_samples, dtype=np.float32)[point_indices]
        if "sampled_point_weights" in group:
            # Boundary-focused preprocessing stores weights separately from
            # sampled_points so the old [x,y,u,v,p,omega,T] tuple is unchanged.
            point_weights = group["sampled_point_weights"][...].astype(np.float32)[point_indices]
        else:
            point_weights = np.ones((samples.shape[0],), dtype=np.float32)
        if "sampled_point_group" in group:
            point_group = group["sampled_point_group"][...].astype(np.int64)[point_indices]
        else:
            point_group = np.zeros((samples.shape[0],), dtype=np.int64)
        query_xy = samples[:, 0:2].astype(np.float32)
        field_targets = samples[:, 2 : 2 + self.field_dim].astype(np.float32)
        module_centers = group["module_centers"][...].astype(np.float32)
        heat_powers = group["heat_powers"][...].astype(np.float32)
        module_present = group["module_present"][...].astype(np.float32)
        interface_condition = _condition_with_h_effective_fallback(
            group["interface_condition"][...].astype(np.float32),
            self.interface_condition_feature_names,
        )
        if "interface_condition_valid_mask" in group:
            interface_condition_valid_mask = group["interface_condition_valid_mask"][...].astype(np.float32)
        else:
            interface_condition_valid_mask = np.ones(interface_condition.shape[:-1], dtype=np.float32)
        interface_target = group["interface_target"][...].astype(np.float32)
        internal_grid = group["module_internal_temperature"][...].astype(np.float32)
        internal_mask = group["module_internal_mask"][...].astype(np.float32)
        local_query_points = _local_disk_query_points(internal_mask)
        internal_points = internal_grid[:, internal_mask.astype(bool)].astype(np.float32)
        material_params = self._material_params(group)
        lx, ly = self._domain_lengths(group)

        teacher_port_tokens = self._teacher_port_tokens(interface_condition)
        local_module_params = self._local_module_params(heat_powers, interface_condition, material_params, module_present)
        if self.include_structure_targets and self._has_structure_targets:
            env_token_coords = group["structure_env_token_coords"][...].astype(np.float32)
            env_module_target = group["env_module_influence_target"][...].astype(np.float32)
            module_affinity_target = group["module_affinity_target"][...].astype(np.float32)
            active_edge_count_target = (
                group["active_edge_count_target"][...].astype(np.float32)
                if "active_edge_count_target" in group
                else np.asarray([0.0], dtype=np.float32)
            )
            structure_targets = {
                "env_token_coords": env_token_coords,
                "env_module_influence_target": env_module_target,
                "module_affinity_target": module_affinity_target,
                "env_module_target_mask": (module_present[None, :] > 0.5).astype(np.float32) * np.ones_like(env_module_target),
                "module_affinity_target_mask": ((module_present[:, None] > 0.5) & (module_present[None, :] > 0.5)).astype(np.float32),
                "active_edge_count_target": active_edge_count_target.reshape(1),
                "has_solved_structure_targets": np.asarray([1.0], dtype=np.float32),
            }
            if "env_region_label" in group:
                structure_targets["env_region_label"] = group["env_region_label"][...].astype(np.int64)
        elif self.include_structure_targets:
            structure_targets = self._fallback_structure_targets(module_centers, heat_powers, module_present, lx, ly)
        else:
            structure_targets = None

        if self.normalize_inputs:
            heat_powers = self.normalizer.normalize_heat_power(heat_powers)
            interface_condition = self.normalizer.normalize_interface_condition(interface_condition)
        if self.normalize_targets:
            field_targets = self.normalizer.normalize_fields(field_targets)
            interface_target = self.normalizer.normalize_interface_targets(interface_target)
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
            "point_weights": point_weights.astype(np.float32),
            "point_group": point_group,
            "module_internal_temperature_points": internal_points,
            "module_internal_mask": internal_mask,
            "module_internal_query_points": local_query_points,
            "interface_condition": interface_condition,
            "interface_condition_valid_mask": interface_condition_valid_mask,
            "interface_target": interface_target,
            "teacher_port_tokens": teacher_port_tokens,
            "local_module_params": local_module_params,
            "case_id": case_id,
        }
        if structure_targets is not None:
            sample["structure_targets"] = structure_targets
        if self.include_grid:
            sample["x_grid"] = group["x_grid"][...].astype(np.float32)
            sample["y_grid"] = group["y_grid"][...].astype(np.float32)
            sample["steady_field"] = group["steady_field"][...].astype(np.float32)
            sample["rms_field"] = group["rms_field"][...].astype(np.float32) if "rms_field" in group else np.zeros_like(sample["steady_field"])
            if "module_mask" in group:
                sample["module_mask"] = group["module_mask"][...].astype(np.uint8)
        return sample
