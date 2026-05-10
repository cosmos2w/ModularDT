from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import h5py
import numpy as np

from cache import find_cached_job, new_job_id, register_cache, request_hash
from model_registry import ModelRegistry
from render_service import (
    render_error_field_images,
    render_field_images,
    render_internal_temperature_images,
    render_interface_curves,
    render_internal_summary,
    render_organization_domain_overlay,
    render_organization_matrices,
)
from schemas import DesignRequest, ForwardSimulationRequest, ValidationResult
from settings import settings


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _selected_device_arg() -> Optional[str]:
    requested = settings.device.lower().strip()
    return None if requested == "auto" else settings.device


def _distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    dx = float(a["x"]) - float(b["x"])
    dy = float(a["y"]) - float(b["y"])
    return float(math.sqrt(dx * dx + dy * dy))


def _organization_summary(aux: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(aux, Mapping) or "A_mh" not in aux or "A_eh" not in aux:
        return None
    a_mh = np.asarray(aux.get("A_mh"), dtype=np.float32)
    a_eh = np.asarray(aux.get("A_eh"), dtype=np.float32)
    if a_mh.ndim != 2 or a_eh.ndim != 2:
        return None
    module_dom = np.argmax(a_mh, axis=1).astype(int).tolist() if a_mh.size else []
    env_dom = np.argmax(a_eh, axis=1).astype(int).tolist() if a_eh.size else []
    env_coords = np.asarray(aux.get("env_coords", []), dtype=np.float32).reshape(-1, 2) if aux.get("env_coords") is not None else np.zeros((0, 2), dtype=np.float32)
    strength = np.asarray(aux.get("hyper_strength", []), dtype=np.float32).reshape(-1)
    module_mass = np.asarray(aux.get("hyper_module_mass", []), dtype=np.float32).reshape(-1)
    env_mass = np.asarray(aux.get("hyper_env_mass", []), dtype=np.float32).reshape(-1)
    return {
        "A_mh": a_mh.tolist(),
        "A_eh_shape": [int(item) for item in a_eh.shape],
        "env_token_xy": env_coords.tolist() if env_coords.size else None,
        "dominant_module_hyperedge": module_dom,
        "dominant_env_hyperedge": env_dom,
        "hyperedge_strength": strength.tolist() if strength.size else None,
        "hyperedge_module_mass": module_mass.tolist() if module_mass.size else None,
        "hyperedge_env_mass": env_mass.tolist() if env_mass.size else None,
    }


FIELD_ORDER = ("temperature", "u", "v", "p", "omega")
FIELD_CHANNELS = {"u": 0, "v": 1, "p": 2, "omega": 3, "temperature": 4}
ERROR_FIELD_DEFINITION = "range_normalized_absolute_error_v2"


def _robust_reference_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    span = hi - lo
    if span > 1.0e-12:
        return float(span)
    rms = float(np.sqrt(np.mean(finite * finite)))
    return max(rms, 1.0e-8)


def _comparison_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, Dict[str, float]]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    truth_arr = np.asarray(truth, dtype=np.float64)
    out: Dict[str, Dict[str, float]] = {}
    for name in FIELD_ORDER:
        channel = FIELD_CHANNELS[name]
        if channel >= pred_arr.shape[-1] or channel >= truth_arr.shape[-1]:
            continue
        diff = pred_arr[..., channel] - truth_arr[..., channel]
        ref = truth_arr[..., channel]
        finite = np.isfinite(diff) & np.isfinite(ref)
        if not finite.any():
            continue
        diff_f = diff[finite]
        ref_f = ref[finite]
        rmse = float(np.sqrt(np.mean(diff_f * diff_f)))
        span = float(np.nanmax(ref_f) - np.nanmin(ref_f))
        normalizer = span if span > 1.0e-12 else float(np.sqrt(np.mean(ref_f * ref_f)))
        normalizer = max(normalizer, 1.0e-12)
        rel_l2 = float(np.linalg.norm(diff_f) / max(np.linalg.norm(ref_f), 1.0e-12))
        out[name] = {
            "rmse": rmse,
            "nrmse": float(rmse / normalizer),
            "relative_l2": rel_l2,
            "mae": float(np.mean(np.abs(diff_f))),
            "max_abs": float(np.max(np.abs(diff_f))),
            "normalizer": float(normalizer),
        }
    return out


def _relative_error_grid(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return a visually stable relative error map.

    The previous implementation divided by the pointwise truth value. That is
    mathematically valid for strictly positive quantities, but it is misleading
    for channel fields with zeros and sign changes. For display we normalize the
    absolute error by a robust range of the reference field, matching the NRMSE
    intuition used in the metrics panel.
    """
    pred_arr = np.asarray(pred, dtype=np.float32)
    truth_arr = np.asarray(truth, dtype=np.float32)
    rel = np.zeros_like(pred_arr, dtype=np.float32)
    for name in FIELD_ORDER:
        channel = FIELD_CHANNELS[name]
        if channel >= pred_arr.shape[-1] or channel >= truth_arr.shape[-1]:
            continue
        ref = truth_arr[..., channel]
        scale = _robust_reference_scale(ref)
        rel[..., channel] = np.abs(pred_arr[..., channel] - ref) / max(scale, 1.0e-8)
    return rel


def _internal_points(values: np.ndarray, mask: np.ndarray, module_index: int) -> Optional[np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    if arr.size == 0 or mask_bool.size == 0 or module_index < 0 or module_index >= arr.shape[0]:
        return None
    local = np.asarray(arr[module_index], dtype=np.float64)
    if local.ndim >= 3 and local.shape[-1] == 1:
        local = local[..., 0]
    if local.shape == mask_bool.shape:
        return local[mask_bool].reshape(-1)
    return local.reshape(-1)[: int(np.sum(mask_bool))]


def _internal_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, count: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for idx in range(max(int(count), 0)):
        pred_points = _internal_points(pred, mask, idx)
        truth_points = _internal_points(truth, mask, idx)
        if pred_points is None or truth_points is None:
            continue
        n = min(pred_points.size, truth_points.size)
        if n <= 0:
            continue
        diff = pred_points[:n] - truth_points[:n]
        ref = truth_points[:n]
        rmse = float(np.sqrt(np.mean(diff * diff)))
        span = float(np.nanmax(ref) - np.nanmin(ref))
        normalizer = span if span > 1.0e-12 else float(np.sqrt(np.mean(ref * ref)))
        normalizer = max(normalizer, 1.0e-12)
        out[str(idx)] = {
            "rmse": rmse,
            "nrmse": float(rmse / normalizer),
            "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1.0e-12)),
            "mae": float(np.mean(np.abs(diff))),
            "max_abs": float(np.max(np.abs(diff))),
            "normalizer": float(normalizer),
        }
    return out


def _internal_relative_error(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, count: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    out = np.full((max(int(count), 0), int(mask_bool.shape[0]), int(mask_bool.shape[1])), np.nan, dtype=np.float32)
    for idx in range(out.shape[0]):
        pred_points = _internal_points(pred, mask_bool, idx)
        truth_points = _internal_points(truth, mask_bool, idx)
        if pred_points is None or truth_points is None:
            continue
        n = min(pred_points.size, truth_points.size, int(np.sum(mask_bool)))
        ref = truth_points[:n]
        scale = _robust_reference_scale(ref)
        local = np.full(mask_bool.shape, np.nan, dtype=np.float32)
        local_vals = np.abs(pred_points[:n] - ref) / max(scale, 1.0e-8)
        flat_idx = np.flatnonzero(mask_bool.reshape(-1))[:n]
        local.reshape(-1)[flat_idx] = local_vals.astype(np.float32)
        out[idx] = local
    return out


def _internal_scale(*arrays: np.ndarray, mask: np.ndarray, count: int) -> tuple[float, float]:
    values: List[np.ndarray] = []
    for arr in arrays:
        for idx in range(max(int(count), 0)):
            points = _internal_points(arr, mask, idx)
            if points is not None and points.size:
                values.append(points[np.isfinite(points)])
    finite = [item for item in values if item.size]
    if not finite:
        return 0.0, 1.0
    merged = np.concatenate(finite)
    lo = float(np.percentile(merged, 1.0))
    hi = float(np.percentile(merged, 99.0))
    if abs(hi - lo) < 1.0e-12:
        pad = max(abs(hi), 1.0) * 1.0e-3
        return hi - pad, hi + pad
    return lo, hi


def _error_scale(arr: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(arr, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    return 0.0, max(float(np.percentile(finite, 99.0)) if finite.size else 1.0, 1.0e-8)


def _field_scale_overrides(*field_arrays: np.ndarray) -> Dict[str, tuple[float, float]]:
    out: Dict[str, tuple[float, float]] = {}
    for name in FIELD_ORDER:
        channel = FIELD_CHANNELS[name]
        values = []
        for arr in field_arrays:
            arr_np = np.asarray(arr, dtype=np.float32)
            if arr_np.ndim == 3 and channel < arr_np.shape[-1]:
                finite = arr_np[..., channel][np.isfinite(arr_np[..., channel])]
                if finite.size:
                    values.append(finite)
        if not values:
            continue
        merged = np.concatenate(values)
        lo = float(np.percentile(merged, 1.0))
        hi = float(np.percentile(merged, 99.0))
        if name in {"u", "v", "omega"}:
            mag = max(abs(lo), abs(hi), 1.0e-9)
            out[name] = (-mag, mag)
        elif abs(hi - lo) < 1.0e-12:
            pad = max(abs(hi), 1.0) * 1.0e-3
            out[name] = (hi - pad, hi + pad)
        else:
            out[name] = (lo, hi)
    return out


class ThermalInferenceService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._case_summary_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _dataset_path(self) -> Path:
        from channelthermal_model_utils import resolve_demo_path

        return resolve_demo_path("./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5")

    def _load_forward_artifact(self, model_id: str) -> Dict[str, Any]:
        from channelthermal_model_utils import select_device
        from train_inverse import load_forward_model

        entry = self.registry.get_entry(model_id)
        if entry.missing_files:
            raise FileNotFoundError("Model entry has missing files: " + ", ".join(entry.missing_files))
        if not entry.enabled:
            raise ValueError(f"Model {model_id!r} is disabled.")
        with entry.lock:
            if entry.loaded_artifact is not None:
                return entry.loaded_artifact
            device = select_device(_selected_device_arg())
            forward_cfg = {
                "run_dir": str(entry.run_dir),
                "checkpoint_name": entry.checkpoint_path.name,
                "config_name": entry.config_path.name,
                "allow_state_mismatch": bool(entry.raw.get("allow_state_mismatch", False)),
            }
            if entry.raw.get("local_surrogate_checkpoint_path"):
                forward_cfg["local_surrogate_checkpoint_path"] = entry.raw["local_surrogate_checkpoint_path"]
            model, metadata, checkpoint_path = load_forward_model(forward_cfg, device)
            entry.loaded_artifact = {
                "model": model,
                "metadata": metadata,
                "checkpoint_path": checkpoint_path,
                "device": device,
            }
            return entry.loaded_artifact

    @staticmethod
    def _decode_string(value: Any) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    @staticmethod
    def _decode_string_array(values: Any) -> List[str]:
        return [ThermalInferenceService._decode_string(item) for item in np.asarray(values).reshape(-1)]

    @staticmethod
    def _read_case_config(group: h5py.Group) -> Dict[str, Any]:
        if "case_config_json" not in group:
            return {}
        raw = group["case_config_json"][()]
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _local_disk_query_points(mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask).astype(bool)
        if mask.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        size_y, size_x = mask.shape
        xi = np.linspace(-1.0, 1.0, size_x, dtype=np.float32)
        eta = np.linspace(-1.0, 1.0, size_y, dtype=np.float32)
        xx, yy = np.meshgrid(xi, eta)
        return np.stack([xx[mask], yy[mask]], axis=-1).astype(np.float32)

    @staticmethod
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

    @staticmethod
    def _re_uin(group: h5py.Group) -> tuple[float, float]:
        materials = group.get("material_parameters", None)
        if materials is None:
            return 0.0, 0.0
        return float(materials.attrs.get("re", 0.0)), float(materials.attrs.get("u_in", 0.0))

    @classmethod
    def _domain_lengths(cls, group: h5py.Group) -> tuple[float, float]:
        cfg = cls._read_case_config(group)
        domain = cfg.get("domain", {}) if isinstance(cfg, dict) else {}
        lx = float(domain.get("lx", np.max(group["x_grid"][...]) if "x_grid" in group else 12.0))
        ly = float(domain.get("ly", np.max(group["y_grid"][...]) if "y_grid" in group else 6.0))
        return lx, ly

    def _selected_case_ids(self, split: str) -> List[tuple[int, str, str]]:
        split = str(split or "test").lower()
        with h5py.File(self._dataset_path(), "r") as h5:
            if "case_ids" in h5 and "splits" in h5:
                case_ids = self._decode_string_array(h5["case_ids"][...])
                splits = self._decode_string_array(h5["splits"][...])
            else:
                case_ids = sorted(h5["cases"].keys())
                splits = [self._decode_string(h5["cases"][case_id].attrs.get("split", "all")) for case_id in case_ids]
        selected = [(idx, str(case_id), str(splits[idx])) for idx, case_id in enumerate(case_ids) if split == "all" or str(splits[idx]).lower() == split]
        if not selected and split != "all":
            selected = [(idx, str(case_id), str(splits[idx])) for idx, case_id in enumerate(case_ids)]
        return selected

    def _load_record_by_case_id(self, case_id: str, split_label: str = "all") -> Any:
        with h5py.File(self._dataset_path(), "r") as h5:
            group = h5["cases"][str(case_id)]
            material = self._material_params(group)
            re, u_in = self._re_uin(group)
            lx, ly = self._domain_lengths(group)
            module_internal_mask = group["module_internal_mask"][...].astype(np.uint8) if "module_internal_mask" in group else np.zeros((0, 0), dtype=np.uint8)
            radius = float(material[5]) if material.shape[0] > 5 and float(material[5]) > 0.0 else 0.45
            centers = group["module_centers"][...].astype(np.float32)
            present = group["module_present"][...].astype(np.float32)
            heat = group["heat_powers"][...].astype(np.float32) if "heat_powers" in group else np.zeros((centers.shape[0],), dtype=np.float32)
            return SimpleNamespace(
                case_id=str(case_id),
                split=str(split_label),
                true_count=int(np.sum(present > 0.5)),
                module_centers=centers,
                module_present=present,
                heat_powers=heat,
                material_params=material,
                re=re,
                u_in=u_in,
                domain_length_x=float(lx),
                domain_length_y=float(ly),
                module_radius=radius,
                x_grid=group["x_grid"][...].astype(np.float32),
                y_grid=group["y_grid"][...].astype(np.float32),
                steady_field=group["steady_field"][...].astype(np.float32) if "steady_field" in group else None,
                module_mask=group["module_mask"][...].astype(np.uint8) if "module_mask" in group else None,
                module_internal_temperature=group["module_internal_temperature"][...].astype(np.float32)
                if "module_internal_temperature" in group
                else None,
                module_internal_mask=module_internal_mask,
                module_internal_query_points=self._local_disk_query_points(module_internal_mask),
                interface_condition=group["interface_condition"][...].astype(np.float32) if "interface_condition" in group else None,
                interface_target=group["interface_target"][...].astype(np.float32) if "interface_target" in group else None,
            )

    def reference_record(self, split: str, case_index: int = 0, case_id: Optional[str] = None):
        if case_id:
            return self._load_record_by_case_id(str(case_id), split)
        selected = self._selected_case_ids(split)
        if not selected:
            raise RuntimeError(f"No reference records found in split {split!r}.")
        _, selected_case_id, selected_split = selected[min(max(int(case_index), 0), len(selected) - 1)]
        return self._load_record_by_case_id(selected_case_id, selected_split)

    def list_reference_cases(self, split: str = "test", limit: int = 80) -> List[Dict[str, Any]]:
        cache_key = f"{split}:{int(limit)}"
        if cache_key in self._case_summary_cache:
            return self._case_summary_cache[cache_key]
        selected = self._selected_case_ids(split)
        out = []
        with h5py.File(self._dataset_path(), "r") as h5:
            for local_idx, (_, case_id, split_label) in enumerate(selected[: max(int(limit), 1)]):
                group = h5["cases"][case_id]
                centers = group["module_centers"][...].astype(np.float32)
                present = group["module_present"][...].astype(np.float32)
                heat = group["heat_powers"][...].astype(np.float32) if "heat_powers" in group else np.zeros((centers.shape[0],), dtype=np.float32)
                active = present > 0.5
                material = self._material_params(group)
                re, u_in = self._re_uin(group)
                lx, ly = self._domain_lengths(group)
                radius = float(material[5]) if material.shape[0] > 5 and float(material[5]) > 0.0 else 0.45
                out.append(
                    {
                        "index": local_idx,
                        "case_id": str(case_id),
                        "split": str(split_label),
                        "num_modules": int(np.sum(active)),
                        "total_heat_power": float(np.sum(heat[active])) if heat.size else 0.0,
                        "re": float(re),
                        "u_in": float(u_in),
                        "domain_length_x": float(lx),
                        "domain_length_y": float(ly),
                        "module_radius": float(radius),
                        "modules": [
                            {
                                "x": float(centers[i, 0]),
                                "y": float(centers[i, 1]),
                                "heat_power": float(heat[i]) if i < heat.shape[0] else 1.0,
                            }
                            for i in np.flatnonzero(active)
                        ],
                    }
                )
        self._case_summary_cache[cache_key] = out
        return out

    def model_config(self, model_id: str) -> Dict[str, Any]:
        entry = self.registry.get_entry(model_id)
        cfg = entry.load_config_json()
        ref_split = str(entry.raw.get("default_reference_split", "test"))
        ref_index = int(entry.raw.get("default_reference_case_index", 0))
        record = self.reference_record(ref_split, ref_index)
        model_cfg = cfg.get("model", {})
        max_num_modules = model_cfg.get("max_num_modules")
        if str(max_num_modules).lower() == "auto":
            max_num_modules = int(record.module_present.shape[0])
        fields = list(model_cfg.get("field_names", ["u", "v", "p", "omega", "temperature"]))
        return {
            "max_num_modules": int(max_num_modules or record.module_present.shape[0]),
            "domain_length_x": float(record.domain_length_x),
            "domain_length_y": float(record.domain_length_y),
            "module_radius": float(record.module_radius),
            "field_names": fields,
            "default_reference_split": ref_split,
            "default_reference_case_index": ref_index,
            "heat_power_min": float(entry.raw.get("heat_power_min", 0.0)),
            "heat_power_max": float(entry.raw.get("heat_power_max", 3.0)),
            "default_heat_power": float(entry.raw.get("default_heat_power", 1.0)),
            "reference_case": {
                "case_id": record.case_id,
                "num_modules": int(record.true_count),
                "re": float(record.re),
                "u_in": float(record.u_in),
            },
            "model": entry.to_public_dict(),
        }

    def validate(self, request: DesignRequest) -> ValidationResult:
        cfg = self.model_config(request.model_id)
        lx = float(cfg["domain_length_x"])
        ly = float(cfg["domain_length_y"])
        radius = float(cfg["module_radius"])
        max_num = int(cfg["max_num_modules"])
        heat_min = float(cfg["heat_power_min"])
        heat_max = float(cfg["heat_power_max"])
        min_center_distance = float(self.registry.get_entry(request.model_id).raw.get("min_center_distance", 2.0 * radius + 0.05))
        warnings: List[str] = []
        valid = True
        modules = [_model_dump(item) for item in request.modules]
        if not modules:
            valid = False
            warnings.append("At least one heated module is required.")
        if len(modules) > max_num:
            valid = False
            warnings.append(f"Too many modules: {len(modules)} > max_num_modules={max_num}.")
        for idx, module in enumerate(modules):
            x = float(module["x"])
            y = float(module["y"])
            heat = float(module["heat_power"])
            if not math.isfinite(x) or not math.isfinite(y):
                valid = False
                warnings.append(f"Module M{idx} has non-finite coordinates.")
                continue
            if x < radius or x > lx - radius:
                valid = False
                warnings.append(f"Module M{idx} x={x:.3g} violates the inlet/outlet clearance implied by radius={radius:.3g}.")
            if y < radius or y > ly - radius:
                valid = False
                warnings.append(f"Module M{idx} y={y:.3g} violates wall clearance implied by radius={radius:.3g}.")
            if not math.isfinite(heat) or heat < 0.0:
                valid = False
                warnings.append(f"Module M{idx} has invalid heat power.")
            elif heat < heat_min or heat > heat_max:
                warnings.append(f"Module M{idx} heat={heat:.3g} is outside the expected range [{heat_min:.3g}, {heat_max:.3g}].")
        for i, a in enumerate(modules):
            for j in range(i + 1, len(modules)):
                dist = _distance(a, modules[j])
                if dist < min_center_distance:
                    valid = False
                    warnings.append(f"Modules M{i} and M{j} are too close (distance={dist:.3g}, minimum={min_center_distance:.3g}).")
        return ValidationResult(
            valid=valid,
            warnings=warnings,
            max_num_modules=max_num,
            domain_length_x=lx,
            domain_length_y=ly,
            module_radius=radius,
            min_center_distance=min_center_distance,
            heat_power_min=heat_min,
            heat_power_max=heat_max,
            total_heat_power=float(sum(float(item["heat_power"]) for item in modules)),
        )

    def _compute_kpis(self, record: Any, prediction: Mapping[str, Any]) -> Dict[str, Any]:
        from model_inverse import channel_clearance_diagnostics
        from thermal_inverse_kpi import compute_steady_thermal_kpis

        kpis = compute_steady_thermal_kpis(
            prediction["pred_field_grid"],
            x_grid=record.x_grid,
            y_grid=record.y_grid,
            channel_order=("u", "v", "p", "omega", "temperature"),
            module_centers=prediction["centers_padded"],
            module_present=prediction["module_present"],
            heat_powers=prediction.get("heat_powers", record.heat_powers),
            module_internal_temperature=prediction.get("pred_internal_temperature"),
            module_internal_mask=record.module_internal_mask,
            interface_target=prediction.get("pred_interface"),
            interface_condition=prediction.get("pred_port_condition"),
            domain={
                "domain_length_x": record.domain_length_x,
                "domain_length_y": record.domain_length_y,
                "module_radius": record.module_radius,
            },
            material_params=record.material_params,
        )
        centers = prediction["centers_padded"][prediction["module_present"] > 0.5]
        kpis.update(
            channel_clearance_diagnostics(
                centers,
                domain_length_x=record.domain_length_x,
                domain_length_y=record.domain_length_y,
                module_radius=record.module_radius,
            )
        )
        kpis["num_modules"] = int(centers.shape[0])
        kpis["heat_power_total"] = float(np.sum(np.asarray(prediction.get("heat_powers", []), dtype=np.float32)[: centers.shape[0]]))
        return _json_safe(kpis)

    def _cached_forward_result_current(self, job_id: str, request: DesignRequest) -> bool:
        path = self.result_path(job_id)
        if not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as f:
                result = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if "internal_temperature" not in result:
            return False
        comparison = result.get("comparison")
        if request.return_error and isinstance(comparison, Mapping) and comparison.get("available"):
            return comparison.get("error_definition") == ERROR_FIELD_DEFINITION
        internal = result.get("internal_temperature")
        if isinstance(internal, Mapping) and internal.get("available") and internal.get("error_scale") is not None:
            return internal.get("error_definition") == ERROR_FIELD_DEFINITION
        return True

    @staticmethod
    def _matches_reference_record(request: DesignRequest, record: Any) -> bool:
        if request.design_source != "reference_case" or request.re is not None or request.u_in is not None:
            return False
        active = np.flatnonzero(np.asarray(record.module_present, dtype=np.float32) > 0.5)
        modules = [_model_dump(item) for item in request.modules]
        if len(modules) != len(active):
            return False
        for slot, module in zip(active, modules):
            center = np.asarray(record.module_centers[int(slot)], dtype=np.float64)
            heat = float(record.heat_powers[int(slot)]) if int(slot) < len(record.heat_powers) else 1.0
            if abs(float(module["x"]) - float(center[0])) > 1.0e-4:
                return False
            if abs(float(module["y"]) - float(center[1])) > 1.0e-4:
                return False
            if abs(float(module["heat_power"]) - heat) > 1.0e-4:
                return False
        return record.steady_field is not None

    def _comparison_payload(
        self,
        *,
        request: DesignRequest,
        record: Any,
        prediction: Mapping[str, Any],
        job_id: str,
        job_dir: Path,
        modules_for_render: List[Mapping[str, Any]],
        field_scale_overrides: Optional[Mapping[str, tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        if not (request.return_ground_truth or request.return_error):
            return {"available": False, "mode": "inference_only", "reason": "Comparison was not requested."}
        if not self._matches_reference_record(request, record):
            return {
                "available": False,
                "mode": "inference_only",
                "reason": "Ground truth is available only for unmodified dataset reference cases at reference flow conditions.",
            }
        truth = np.asarray(record.steady_field, dtype=np.float32)
        pred = np.asarray(prediction["pred_field_grid"], dtype=np.float32)
        truth_dir = job_dir / "truth_frames"
        error_dir = job_dir / "error_frames"
        truth_meta = render_field_images(
            truth,
            truth_dir,
            modules_for_render,
            domain_length_x=float(record.domain_length_x),
            domain_length_y=float(record.domain_length_y),
            module_radius=float(record.module_radius),
            display_smoothing=bool(request.display_smoothing),
            display_scale=int(request.display_scale),
            render_interpolation=str(request.render_interpolation),
            scale_overrides=field_scale_overrides,
        )
        rel_error = _relative_error_grid(pred, truth)
        error_meta = render_error_field_images(
            rel_error,
            error_dir,
            modules_for_render,
            domain_length_x=float(record.domain_length_x),
            domain_length_y=float(record.domain_length_y),
            module_radius=float(record.module_radius),
            display_smoothing=bool(request.display_smoothing),
            display_scale=int(request.display_scale),
            render_interpolation=str(request.render_interpolation),
        )
        return {
            "available": True,
            "mode": "reference_ground_truth",
            "reason": None,
            "metrics": _comparison_metrics(pred, truth),
            "error_definition": ERROR_FIELD_DEFINITION,
            "error_label": "|prediction - reference| / robust_range(reference)",
            "ground_truth_frame_urls": {name: [f"/api/jobs/{job_id}/files/truth_frames/{name}.png"] for name in truth_meta["fields"].keys()},
            "relative_error_frame_urls": {name: [f"/api/jobs/{job_id}/files/error_frames/{name}.png"] for name in error_meta["fields"].keys()},
            "truth_render": truth_meta,
            "error_render": error_meta,
        }

    def _internal_temperature_payload(
        self,
        *,
        job_id: str,
        job_dir: Path,
        modules_for_render: List[Mapping[str, Any]],
        pred_internal: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        truth_internal: Optional[np.ndarray] = None,
        simulation_internal: Optional[np.ndarray] = None,
        url_prefix: str = "/api/jobs",
    ) -> Dict[str, Any]:
        pred = np.asarray(pred_internal, dtype=np.float32) if pred_internal is not None else np.asarray([], dtype=np.float32)
        mask_arr = np.asarray(mask, dtype=bool) if mask is not None else np.asarray([], dtype=bool)
        count = min(len(modules_for_render), int(pred.shape[0]) if pred.ndim >= 1 else 0)
        if pred.size == 0 or mask_arr.size == 0 or count <= 0:
            return {"available": False, "modules": [], "reason": "No module-internal temperature arrays were returned."}
        truth = np.asarray(truth_internal, dtype=np.float32) if truth_internal is not None else None
        sim = np.asarray(simulation_internal, dtype=np.float32) if simulation_internal is not None else None
        scale_arrays = [pred]
        if truth is not None and truth.size:
            scale_arrays.append(truth)
        if sim is not None and sim.size:
            scale_arrays.append(sim)
        temp_scale = _internal_scale(*scale_arrays, mask=mask_arr, count=count)
        pred_meta = render_internal_temperature_images(
            pred,
            mask_arr,
            job_dir / "internal_temperature" / "inferred",
            modules_for_render,
            scale=temp_scale,
            prefix="inferred",
        )
        truth_meta = None
        if truth is not None and truth.size:
            truth_meta = render_internal_temperature_images(
                truth,
                mask_arr,
                job_dir / "internal_temperature" / "ground_truth",
                modules_for_render,
                scale=temp_scale,
                prefix="truth",
            )
        sim_meta = None
        if sim is not None and sim.size:
            sim_meta = render_internal_temperature_images(
                sim,
                mask_arr,
                job_dir / "internal_temperature" / "simulation",
                modules_for_render,
                scale=temp_scale,
                prefix="simulation",
            )
        comparator = sim if sim is not None and sim.size else truth
        error_meta = None
        metrics = {}
        if comparator is not None and np.asarray(comparator).size:
            rel_error = _internal_relative_error(pred, comparator, mask_arr, count)
            error_scale = _error_scale(rel_error)
            error_meta = render_internal_temperature_images(
                rel_error,
                mask_arr,
                job_dir / "internal_temperature" / "relative_error",
                modules_for_render,
                scale=error_scale,
                error=True,
                prefix="error",
            )
            metrics = _internal_metrics(pred, comparator, mask_arr, count)

        def url_for(kind: str, filename: str) -> str:
            return f"{url_prefix}/{job_id}/files/internal_temperature/{kind}/{filename}"

        modules = []
        for idx in range(count):
            item = {
                "index": idx,
                "label": f"M{idx + 1}",
                "heat_power": float(modules_for_render[idx].get("heat_power", 0.0)),
                "inferred_url": None,
                "ground_truth_url": None,
                "simulation_url": None,
                "relative_error_url": None,
                "metrics": metrics.get(str(idx)),
            }
            pred_file = next((m["file"] for m in pred_meta.get("modules", []) if int(m["index"]) == idx), None)
            if pred_file:
                item["inferred_url"] = url_for("inferred", pred_file)
            if truth_meta:
                truth_file = next((m["file"] for m in truth_meta.get("modules", []) if int(m["index"]) == idx), None)
                if truth_file:
                    item["ground_truth_url"] = url_for("ground_truth", truth_file)
            if sim_meta:
                sim_file = next((m["file"] for m in sim_meta.get("modules", []) if int(m["index"]) == idx), None)
                if sim_file:
                    item["simulation_url"] = url_for("simulation", sim_file)
            if error_meta:
                err_file = next((m["file"] for m in error_meta.get("modules", []) if int(m["index"]) == idx), None)
                if err_file:
                    item["relative_error_url"] = url_for("relative_error", err_file)
            modules.append(item)
        return {
            "available": True,
            "quantity": "module_internal_temperature",
            "count": count,
            "default_visible_count": min(3, count),
            "scale": {"vmin": float(temp_scale[0]), "vmax": float(temp_scale[1]), "label": "T"},
            "error_scale": {"vmin": float(error_meta["scale"]["vmin"]), "vmax": float(error_meta["scale"]["vmax"]), "label": "normalized error"} if error_meta else None,
            "error_definition": ERROR_FIELD_DEFINITION if error_meta else None,
            "modules": modules,
        }

    def infer(self, request: DesignRequest) -> Dict[str, Any]:
        validation = self.validate(request)
        if not validation.valid:
            raise ValueError("Invalid design: " + "; ".join(validation.warnings))
        req_hash = request_hash(request)
        cached = find_cached_job(req_hash)
        if cached and self._cached_forward_result_current(cached, request):
            return {"job_id": cached, "status": "cached", "result_url": f"/api/jobs/{cached}/result"}

        from train_inverse import predict_candidate_with_forward

        entry = self.registry.get_entry(request.model_id)
        artifact = self._load_forward_artifact(request.model_id)
        model = artifact["model"]
        metadata = artifact["metadata"]
        record = self.reference_record(request.reference_split, request.reference_case_index, request.reference_case_id)
        if request.re is not None or request.u_in is not None:
            record = SimpleNamespace(**vars(record))
            record.re = float(record.re if request.re is None else request.re)
            record.u_in = float(record.u_in if request.u_in is None else request.u_in)
        modules = [_model_dump(item) for item in request.modules]
        candidate = {
            "centers": np.asarray([[item["x"], item["y"]] for item in modules], dtype=np.float32),
            "count": len(modules),
            "heat_powers": np.asarray([item["heat_power"] for item in modules], dtype=np.float32),
        }
        prediction = predict_candidate_with_forward(
            model,
            metadata,
            record,
            candidate,
            artifact["device"],
            max_num_modules=int(model.config.max_num_modules),
            generate_heat_power=False,
            heat_load_policy="preserve_per_module_heat",
            query_batch_size=int(entry.raw.get("query_batch_size", 32768)),
        )

        job_id = new_job_id()
        job_dir = settings.cache_dir / job_id
        frames_dir = job_dir / "frames"
        artifacts_dir = job_dir / "artifacts"
        job_dir.mkdir(parents=True, exist_ok=True)
        modules_for_render = [
            {"x": float(item["x"]), "y": float(item["y"]), "heat_power": float(item["heat_power"])}
            for item in modules
        ]
        export_arrays = {
            "pred_field": prediction["pred_field_grid"],
            "pred_internal_temperature": prediction.get("pred_internal_temperature"),
            "pred_interface": prediction.get("pred_interface"),
            "pred_port_condition": prediction.get("pred_port_condition"),
            "centers": prediction["centers_padded"],
            "module_present": prediction["module_present"],
            "heat_powers": prediction.get("heat_powers"),
        }
        reference_match = self._matches_reference_record(request, record)
        field_scale_overrides = (
            _field_scale_overrides(prediction["pred_field_grid"], record.steady_field)
            if reference_match and record.steady_field is not None and (request.return_ground_truth or request.return_error)
            else None
        )
        render_meta = render_field_images(
            prediction["pred_field_grid"],
            frames_dir,
            modules_for_render,
            domain_length_x=float(record.domain_length_x),
            domain_length_y=float(record.domain_length_y),
            module_radius=float(record.module_radius),
            display_smoothing=bool(request.display_smoothing),
            display_scale=int(request.display_scale),
            render_interpolation=str(request.render_interpolation),
            scale_overrides=field_scale_overrides,
        )
        comparison = self._comparison_payload(
            request=request,
            record=record,
            prediction=prediction,
            job_id=job_id,
            job_dir=job_dir,
            modules_for_render=modules_for_render,
            field_scale_overrides=field_scale_overrides,
        )
        active_slots = np.flatnonzero(np.asarray(record.module_present, dtype=np.float32) > 0.5)
        truth_internal = None
        if comparison.get("available") and record.module_internal_temperature is not None:
            truth_internal = np.asarray(record.module_internal_temperature, dtype=np.float32)[active_slots[: len(modules_for_render)]]
        internal_temperature = self._internal_temperature_payload(
            job_id=job_id,
            job_dir=job_dir,
            modules_for_render=modules_for_render,
            pred_internal=prediction.get("pred_internal_temperature"),
            mask=record.module_internal_mask,
            truth_internal=truth_internal,
            url_prefix="/api/jobs",
        )
        if comparison.get("available") and record.steady_field is not None:
            export_arrays["ground_truth_field"] = record.steady_field
            export_arrays["relative_error_field"] = _relative_error_grid(prediction["pred_field_grid"], record.steady_field)
        if truth_internal is not None:
            export_arrays["ground_truth_internal_temperature"] = truth_internal
        np.savez_compressed(job_dir / "fields.npz", **export_arrays)
        artifact_paths = {
            "internal_summary": render_internal_summary(
                prediction.get("pred_internal_temperature"),
                modules_for_render,
                artifacts_dir / "module_internal_temperature_summary.png",
            ),
            "interface_curves": render_interface_curves(
                prediction.get("pred_interface"),
                modules_for_render,
                artifacts_dir / "interface_curves.png",
            ),
            "organization_matrices": render_organization_matrices(
                prediction.get("organizer_aux"),
                artifacts_dir / "organization_matrices.png",
            )
            if request.return_organization
            else None,
            "organization_overlay": render_organization_domain_overlay(
                modules_for_render,
                [item["heat_power"] for item in modules_for_render],
                prediction.get("organizer_aux"),
                artifacts_dir / "organization_domain_overlay.png",
                domain_length_x=float(record.domain_length_x),
                domain_length_y=float(record.domain_length_y),
                module_radius=float(record.module_radius),
                env_token_xy=prediction.get("organizer_aux", {}).get("env_coords") if isinstance(prediction.get("organizer_aux"), Mapping) else None,
            )
            if request.return_organization
            else None,
        }
        artifact_urls = {
            key: f"/api/jobs/{job_id}/files/{path.resolve().relative_to(job_dir.resolve()).as_posix()}"
            for key, path in artifact_paths.items()
            if path is not None and path.exists()
        }
        fields = [name for name in render_meta["fields"].keys()]
        frame_urls = {name: [f"/api/jobs/{job_id}/frames/{name}/000"] for name in fields}
        result = {
            "job_id": job_id,
            "status": "complete",
            "request_hash": req_hash,
            "model": entry.to_public_dict(),
            "validation": _model_dump(validation),
            "reference_case": {
                "case_id": record.case_id,
                "split": record.split,
                "re": float(record.re),
                "u_in": float(record.u_in),
            },
            "domain": {
                "length_x": float(record.domain_length_x),
                "length_y": float(record.domain_length_y),
                "module_radius": float(record.module_radius),
                "resolution_nx": int(record.x_grid.shape[1]),
                "resolution_ny": int(record.x_grid.shape[0]),
            },
            "fields": fields,
            "frame_urls": frame_urls,
            "render": render_meta,
            "comparison": comparison,
            "internal_temperature": internal_temperature,
            "kpis": self._compute_kpis(record, prediction) if request.return_kpis else None,
            "modules": modules_for_render,
            "heat_power_source": "web_per_module",
            "organization": _organization_summary(prediction.get("organizer_aux")) if request.return_organization else None,
            "artifacts": artifact_urls,
            "export_npz_url": f"/api/jobs/{job_id}/export.npz",
        }
        with (job_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(result), f, indent=2)
        register_cache(req_hash, job_id)
        return {"job_id": job_id, "status": "complete", "result_url": f"/api/jobs/{job_id}/result"}

    def result_path(self, job_id: str) -> Path:
        return settings.cache_dir / job_id / "result.json"

    def run_forward_simulation(self, request: ForwardSimulationRequest) -> Dict[str, Any]:
        validation = self.validate(request.design)
        if not validation.valid:
            raise ValueError("Invalid design: " + "; ".join(validation.warnings))
        job_id = new_job_id()
        job_dir = settings.cache_dir / "forward_sim_jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/api/simulate-forward/jobs/{job_id}",
            "result_url": f"/api/simulate-forward/jobs/{job_id}/result",
        }
        self._write_json(job_dir / "status.json", status)
        thread = threading.Thread(target=self._run_forward_simulation_worker, args=(job_id, request), daemon=True)
        thread.start()
        return status

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(dict(payload)), f, indent=2)

    def _simulation_config_payload(self, request: DesignRequest, job_dir: Path, job_id: str) -> Dict[str, Any]:
        template_path = settings.demo_root / "Configs" / "config_channelthermal.json"
        with template_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        record = self.reference_record(request.reference_split, request.reference_case_index, request.reference_case_id)
        modules = [_model_dump(item) for item in request.modules]
        payload.setdefault("layout", {})
        payload["layout"]["num_modules"] = len(modules)
        payload["layout"]["centers"] = [[float(item["x"]), float(item["y"])] for item in modules]
        payload["layout"]["heat_powers"] = [float(item["heat_power"]) for item in modules]
        payload.setdefault("flow", {})
        payload["flow"]["re"] = float(record.re if request.re is None else request.re)
        payload["flow"]["u_in"] = float(record.u_in if request.u_in is None else request.u_in)
        payload.setdefault("domain", {})
        payload["domain"]["lx"] = float(record.domain_length_x)
        payload["domain"]["ly"] = float(record.domain_length_y)
        payload["domain"]["nx"] = int(record.x_grid.shape[1])
        payload["domain"]["ny"] = int(record.x_grid.shape[0])
        payload["domain"]["module_radius"] = float(record.module_radius)
        payload.setdefault("save", {})
        payload["save"]["root_dir"] = str(job_dir / "raw")
        payload["save"]["case_id"] = f"web_{job_id}"
        payload["save"]["tag"] = "web_verify"
        payload.setdefault("execution", {})
        payload["execution"]["device"] = "cpu" if settings.device.lower() == "auto" else settings.device
        return payload

    def _run_forward_simulation_worker(self, job_id: str, request: ForwardSimulationRequest) -> None:
        job_dir = settings.cache_dir / "forward_sim_jobs" / job_id
        status_path = job_dir / "status.json"
        self._write_json(status_path, {"job_id": job_id, "status": "running", "status_url": f"/api/simulate-forward/jobs/{job_id}", "result_url": f"/api/simulate-forward/jobs/{job_id}/result"})
        stdout_path = job_dir / "stdout.txt"
        stderr_path = job_dir / "stderr.txt"
        try:
            config_payload = self._simulation_config_payload(request.design, job_dir, job_id)
            config_path = job_dir / "simulation_config.json"
            self._write_json(config_path, config_payload)
            command = [sys.executable, str(settings.src_dir / "simulate_channelthermal.py"), "--config-json", str(config_path)]
            proc = subprocess.run(
                command,
                cwd=str(settings.demo_root),
                text=True,
                capture_output=True,
                timeout=int(request.max_runtime_seconds),
                check=False,
            )
            stdout_path.write_text(proc.stdout or "", encoding="utf-8")
            stderr_path.write_text(proc.stderr or "", encoding="utf-8")
            if proc.returncode != 0:
                raise RuntimeError(f"Simulation failed with exit code {proc.returncode}.")
            result = self._assemble_simulation_result(job_id, request)
            self._write_json(job_dir / "result.json", result)
            self._write_json(status_path, {"job_id": job_id, "status": "complete", "result_url": f"/api/simulate-forward/jobs/{job_id}/result"})
        except Exception as exc:
            self._write_json(
                status_path,
                {
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(exc),
                    "stdout_tail": self._tail(stdout_path),
                    "stderr_tail": self._tail(stderr_path),
                },
            )

    @staticmethod
    def _tail(path: Path, lines: int = 30) -> List[str]:
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]

    def _assemble_simulation_result(self, job_id: str, request: ForwardSimulationRequest) -> Dict[str, Any]:
        job_dir = settings.cache_dir / "forward_sim_jobs" / job_id
        case_dirs = sorted((job_dir / "raw").glob("case_*"), key=lambda path: path.stat().st_mtime)
        if not case_dirs:
            raise RuntimeError("Simulation completed but no case directory was found.")
        case_dir = case_dirs[-1]
        frames = sorted((case_dir / "scene").glob("frame_*.npz"))
        if not frames:
            raise RuntimeError("Simulation completed but no scene frames were found.")
        final_frame = frames[-1]
        with np.load(final_frame) as data:
            sim_field = np.stack([data["u"], data["v"], data["p"], data["omega"], data["temperature"]], axis=-1).astype(np.float32)
            sim_internal = data["module_internal_temperature"].astype(np.float32) if "module_internal_temperature" in data else None
            sim_internal_mask = data["module_internal_mask"].astype(np.uint8) if "module_internal_mask" in data else None
        record = self.reference_record(request.design.reference_split, request.design.reference_case_index, request.design.reference_case_id)
        modules = [_model_dump(item) for item in request.design.modules]
        pred_field = None
        pred_internal = None
        if request.prediction_job_id:
            pred_npz = settings.cache_dir / request.prediction_job_id / "fields.npz"
            if pred_npz.exists():
                with np.load(pred_npz) as pred_data:
                    pred_field = pred_data["pred_field"].astype(np.float32) if "pred_field" in pred_data else None
                    pred_internal = pred_data["pred_internal_temperature"].astype(np.float32) if "pred_internal_temperature" in pred_data else None
        field_scale_overrides = _field_scale_overrides(sim_field, pred_field) if pred_field is not None else None
        sim_meta = render_field_images(
            sim_field,
            job_dir / "frames",
            modules,
            domain_length_x=float(record.domain_length_x),
            domain_length_y=float(record.domain_length_y),
            module_radius=float(record.module_radius),
            display_smoothing=bool(request.design.display_smoothing),
            display_scale=int(request.design.display_scale),
            render_interpolation=str(request.design.render_interpolation),
            scale_overrides=field_scale_overrides,
        )
        pred_meta = None
        if pred_field is not None:
            pred_meta = render_field_images(
                pred_field,
                job_dir / "prediction_frames",
                modules,
                domain_length_x=float(record.domain_length_x),
                domain_length_y=float(record.domain_length_y),
                module_radius=float(record.module_radius),
                display_smoothing=bool(request.design.display_smoothing),
                display_scale=int(request.design.display_scale),
                render_interpolation=str(request.design.render_interpolation),
                scale_overrides=field_scale_overrides,
            )
        comparison: Dict[str, Any] = {"available": False, "mode": "simulation_only", "reason": "No prediction job was supplied for surrogate-vs-simulation metrics."}
        export_arrays: Dict[str, Any] = {"simulation_field": sim_field}
        if pred_field is not None:
            rel_error = _relative_error_grid(pred_field, sim_field)
            error_scale_overrides = {
                name: (0.0, max(float(np.percentile(rel_error[..., FIELD_CHANNELS[name]][np.isfinite(rel_error[..., FIELD_CHANNELS[name]])], 99.0)), 1.0e-8))
                for name in FIELD_ORDER
                if FIELD_CHANNELS[name] < rel_error.shape[-1] and np.isfinite(rel_error[..., FIELD_CHANNELS[name]]).any()
            }
            error_meta = render_error_field_images(
                rel_error,
                job_dir / "error_frames",
                modules,
                domain_length_x=float(record.domain_length_x),
                domain_length_y=float(record.domain_length_y),
                module_radius=float(record.module_radius),
                display_smoothing=bool(request.design.display_smoothing),
                display_scale=int(request.design.display_scale),
                render_interpolation=str(request.design.render_interpolation),
                scale_overrides=error_scale_overrides,
            )
            comparison = {
                "available": True,
                "mode": "simulation_verification",
                "reason": None,
                "metrics": _comparison_metrics(pred_field, sim_field),
                "error_definition": ERROR_FIELD_DEFINITION,
                "error_label": "|prediction - simulation| / robust_range(simulation)",
                "relative_error_frame_urls": {name: [f"/api/simulate-forward/jobs/{job_id}/files/error_frames/{name}.png"] for name in error_meta["fields"].keys()},
                "error_render": error_meta,
            }
            export_arrays["pred_field"] = pred_field
            export_arrays["relative_error_field"] = rel_error
        internal_temperature = self._internal_temperature_payload(
            job_id=job_id,
            job_dir=job_dir,
            modules_for_render=modules,
            pred_internal=pred_internal,
            mask=sim_internal_mask if sim_internal_mask is not None else record.module_internal_mask,
            simulation_internal=sim_internal,
            url_prefix="/api/simulate-forward/jobs",
        ) if pred_internal is not None else {"available": False, "modules": [], "reason": "No prediction job internal-temperature array was available."}
        if sim_internal is not None:
            export_arrays["simulation_internal_temperature"] = sim_internal
        if pred_internal is not None:
            export_arrays["pred_internal_temperature"] = pred_internal
        np.savez_compressed(job_dir / "simulation_fields.npz", **export_arrays)
        fields = [name for name in sim_meta["fields"].keys()]
        return {
            "job_id": job_id,
            "status": "complete",
            "case_dir": str(case_dir),
            "final_frame": str(final_frame),
            "fields": fields,
            "frame_urls": {name: [f"/api/simulate-forward/jobs/{job_id}/frames/{name}/000"] for name in fields},
            "predicted_frame_urls": {name: [f"/api/simulate-forward/jobs/{job_id}/files/prediction_frames/{name}.png"] for name in pred_meta["fields"].keys()} if pred_meta else {},
            "render": sim_meta,
            "comparison": comparison,
            "internal_temperature": internal_temperature,
            "export_npz_url": f"/api/simulate-forward/jobs/{job_id}/export.npz",
            "stdout_tail": self._tail(job_dir / "stdout.txt"),
            "stderr_tail": self._tail(job_dir / "stderr.txt"),
        }

    def simulation_status_path(self, job_id: str) -> Path:
        return settings.cache_dir / "forward_sim_jobs" / job_id / "status.json"

    def simulation_result_path(self, job_id: str) -> Path:
        return settings.cache_dir / "forward_sim_jobs" / job_id / "result.json"
