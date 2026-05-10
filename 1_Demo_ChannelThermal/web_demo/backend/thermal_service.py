from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import h5py
import numpy as np

from cache import find_cached_job, new_job_id, register_cache, request_hash
from model_registry import ModelRegistry
from render_service import (
    render_field_images,
    render_interface_curves,
    render_internal_summary,
    render_organization_matrices,
)
from schemas import DesignRequest, ValidationResult
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

    def infer(self, request: DesignRequest) -> Dict[str, Any]:
        validation = self.validate(request)
        if not validation.valid:
            raise ValueError("Invalid design: " + "; ".join(validation.warnings))
        req_hash = request_hash(request)
        cached = find_cached_job(req_hash)
        if cached:
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
            generate_heat_power=True,
            heat_load_policy="preserve_total_heat",
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
        np.savez_compressed(
            job_dir / "fields.npz",
            pred_field=prediction["pred_field_grid"],
            pred_internal_temperature=prediction.get("pred_internal_temperature"),
            pred_interface=prediction.get("pred_interface"),
            pred_port_condition=prediction.get("pred_port_condition"),
            centers=prediction["centers_padded"],
            module_present=prediction["module_present"],
            heat_powers=prediction.get("heat_powers"),
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
        )
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
            "kpis": self._compute_kpis(record, prediction) if request.return_kpis else None,
            "modules": modules_for_render,
            "artifacts": artifact_urls,
            "export_npz_url": f"/api/jobs/{job_id}/export.npz",
        }
        with (job_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(result), f, indent=2)
        register_cache(req_hash, job_id)
        return {"job_id": job_id, "status": "complete", "result_url": f"/api/jobs/{job_id}/result"}

    def result_path(self, job_id: str) -> Path:
        return settings.cache_dir / job_id / "result.json"
