"""Local, CPU-only WindFarm demo backed by a saved HONF checkpoint.

The checkpoint is pinned to the newest full-training-population checkpoint at the
time this demo was made. Its original train-fitted transforms and full-access
read path are used unchanged. No dataset or checkpoint is copied or written.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Case_WindFarm" / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from honf_runtime.compat import load_trusted_checkpoint
from windfarm.data import WindFarmNativeView
from windfarm.geometry import D_M, HUB_HEIGHT_D, U_REF_MPS, global_geometry_features, module_geometry
from windfarm.relation_anchored import WindRelationAnchoredAdapter


RUN_NAME = "wind_r15_2_w2_train420_q8192_b24_full1000_v2"
RUN = ROOT / "HONF_Proj/Trained_Results/WindFarm/HONF_Forward_Runs" / RUN_NAME
CHECKPOINT = RUN / "checkpoints/epoch_2500_model.pt"
IDENTITY = RUN / "configs/experiment_identity.json"
DATA = ROOT / "Case_WindFarm/Dataset/links/wind_farm"
MAX_POINTS = 32768
DEFAULT_RESOLUTION = [24, 16, 8]
torch.set_num_threads(4)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Turbine(StrictModel):
    id: int = Field(ge=0)
    x: float
    y: float
    z: float = HUB_HEIGHT_D


class Domain(StrictModel):
    min: tuple[float, float, float]
    max: tuple[float, float, float]


class Wind(StrictModel):
    direction_deg: Literal[270, 285, 300]
    speed_m_s: float = U_REF_MPS


class Reconstruction(StrictModel):
    case_id: str
    turbines: list[Turbine] | None = Field(default=None, min_length=1, max_length=30)
    domain: Domain | None = None
    wind: Wind | None = None
    resolution: tuple[int, int, int] = tuple(DEFAULT_RESOLUTION)


def _scene_dict(case, partition: str) -> dict:
    return {
        "id": case.case,
        "layout_index": case.layout_index,
        "partition": partition,
        "turbines": [
            {"id": i, "x": float(center[0]), "y": float(center[1]), "z": float(center[2])}
            for i, center in enumerate(case.module_centers)
        ],
        "domain": {"min": case.support.lower_D.tolist(), "max": case.support.upper_D.tolist()},
        "wind": {"direction_deg": int(case.wind_direction_deg), "speed_m_s": U_REF_MPS},
        "units": {"coordinates": "D (rotor diameters)", "velocity": "m/s"},
    }


class Demo:
    def __init__(self) -> None:
        import json

        self.identity = json.loads(IDENTITY.read_text())
        provider = self.identity["provider_identity"]
        self.view = WindFarmNativeView(DATA, token_shape=(4, 4, 4))
        self.model = None
        self.lock = threading.Lock()
        self.row_by_case = {}
        self.cases = []
        names = self.view.metadata["case"]
        for partition, rows in (("TRAIN", provider["train_rows"]), ("VALID", provider["validation_rows"])):
            for row in rows:
                row = int(row)
                case_id = str(names[row])
                self.row_by_case[case_id] = (row, partition)
                self.cases.append({
                    "id": case_id,
                    "layout_index": int(self.view.metadata["layout_index"][row]),
                    "wind_direction_deg": int(self.view.metadata["wd_deg"][row]),
                    "turbine_count": int(self.view.metadata["n_turbines"][row]),
                    "partition": partition,
                })
        self.cases.sort(key=lambda item: (item["partition"] != "VALID", item["id"]))

    def meta(self) -> dict:
        return {
            "checkpoint": {
                "run": RUN_NAME,
                "epoch": 2500,
                "path": str(CHECKPOINT),
                "architecture": "relation-anchored joint velocity field",
                "environment_shape": [8, 4, 4],
                "training_cases": 420,
                "validation_cases": 90,
                "status": "completed epoch 2500 on the original full training population",
            },
            "cases": self.cases,
            "defaults": {"case_id": next(item["id"] for item in self.cases
                                          if item["partition"] == "VALID" and 9 <= item["turbine_count"] <= 12),
                         "resolution": DEFAULT_RESOLUTION},
            "controls": {
                "turbine_count": {"min": 1, "max": 30, "training_min": 6, "training_max": 30},
                "wind_direction_deg": {"options": [270, 285, 300]},
                "wind_speed_m_s": {"value": U_REF_MPS, "editable": False},
                "rotor_diameter_m": D_M,
                "hub_height_D": HUB_HEIGHT_D,
                "domain": {"z_min": 0.0, "z_max": 6.25, "description": "Horizontally editable; all turbine centers must remain within the support box."},
                "resolution": {"min": 2, "max": 64, "max_points": MAX_POINTS},
                "coordinates": "rotor diameters",
            },
            "limitations": [
                "The model was trained at 9 m/s and on the three listed direction categories; speed is fixed.",
                "Coordinates are in each direction row's wind-aligned frame; changing direction does not rotate a layout.",
                "Edited layouts and domains are model predictions without CFD references and may be out of distribution.",
                "The latest checkpoint completed 2500 epochs on 420 training cases. TEST cases remain locked.",
            ],
        }

    def case(self, case_id: str):
        try:
            row, partition = self.row_by_case[case_id]
        except KeyError:
            raise HTTPException(404, "Case not found among the available TRAIN/VALID cases.") from None
        return self.view.run(row), partition

    def _load_model(self):
        if self.model is None:
            checkpoint = load_trusted_checkpoint(CHECKPOINT, map_location="cpu")
            if checkpoint["epoch"] != 2500:
                raise RuntimeError("The selected checkpoint is not the expected epoch 2500.")
            provider = checkpoint["experiment_identity"]["provider_identity"]
            self.model = WindRelationAnchoredAdapter.from_state_payload(
                provider["adapter_state"], state_dict=checkpoint["model_state_dict"], device="cpu"
            ).eval()
        return self.model

    @staticmethod
    def _organization(prepared, scene) -> dict:
        directory = prepared.e_access_directory
        coordinates = scene.environment_coordinates[0].detach().cpu().numpy()
        selected = set(np.linspace(0, len(coordinates) - 1, min(12, len(coordinates)), dtype=int).tolist())
        anchors = []
        directory_rows = []
        for i, position in enumerate(coordinates):
            anchor = {"id": i, "position": position.tolist()}
            row = directory.full.row(0, i)
            ids = row.original_id.cpu().tolist()
            is_module = row.is_module.cpu().tolist()
            weights = row.access_weight.cpu().tolist()
            exchange = row.applied_exchange.cpu().tolist()
            self_access = row.structural_self.cpu().tolist()
            members = [
                {"role": "M" if is_m else "E", "id": int(member_id),
                 "weight": float(weight), "exchange": float(gate), "self": float(own)}
                for member_id, is_m, weight, gate, own in zip(ids, is_module, weights, exchange, self_access)
            ]
            directory_rows.append({"id": i, "members": members})
            if i in selected:
                anchor["members"] = members
            anchors.append(anchor)
        return {
            "kind": "e_anchor_directory", "read_mode": "full", "retained_mass": 1.0,
            "anchors": anchors,
            "directory_rows": directory_rows,
            "selection": {
                "selected_anchor_ids": sorted(selected), "shown_rows": len(selected),
                "total_rows": len(coordinates),
                "available_rows": len(directory_rows),
                "description": "Every positive original M/E member in 12 evenly selected full-directory rows is shown.",
            },
            "provenance": {
                "em_stage": prepared.anchored_route_provenance["em"]["stage_identity"],
                "ee_stage": prepared.anchored_route_provenance["ee"]["stage_identity"],
                "meaning": "E receiver <- original M and E members; directory weights include their measures and own-E access.",
            },
            "work": {name: int(value) for name, value in prepared.work_counts.items()},
        }

    def reconstruct(self, request: Reconstruction) -> dict:
        start = time.perf_counter()
        case, partition = self.case(request.case_id)
        original = _scene_dict(case, partition)
        turbines = [item.model_dump() for item in request.turbines] if request.turbines is not None else original["turbines"]
        domain = request.domain.model_dump() if request.domain is not None else original["domain"]
        wind = request.wind.model_dump() if request.wind is not None else original["wind"]
        shape = request.resolution
        if any(n < 2 or n > 64 for n in shape) or np.prod(shape) > MAX_POINTS:
            raise HTTPException(422, f"Each grid dimension must be 2–64, with at most {MAX_POINTS} points.")
        if not np.isclose(wind["speed_m_s"], U_REF_MPS, rtol=0, atol=1e-6):
            raise HTTPException(422, "This checkpoint supports the training inflow speed 9 m/s only.")
        lower = np.array(domain["min"], dtype=np.float64)
        upper = np.array(domain["max"], dtype=np.float64)
        if np.any(upper <= lower) or lower[2] < 0 or upper[2] > 6.25:
            raise HTTPException(422, "Domain bounds must increase; altitude must remain within 0–6.25D.")
        ids = [item["id"] for item in turbines]
        positions = np.array([[item["x"], item["y"], item["z"]] for item in turbines], dtype=np.float32)
        if len(set(ids)) != len(ids):
            raise HTTPException(422, "Turbine IDs must be unique.")
        if not np.allclose(positions[:, 2], HUB_HEIGHT_D, rtol=0, atol=1e-6):
            raise HTTPException(422, "Turbine hub height is fixed at 0.875D.")
        if np.any(positions < lower) or np.any(positions > upper):
            raise HTTPException(422, "Turbine centers must lie inside the domain.")
        native_positions = np.asarray(case.module_centers)
        unchanged = (
            len(turbines) == len(native_positions) and ids == list(range(len(ids)))
            and np.allclose(positions, native_positions, rtol=0, atol=1e-6)
            and np.allclose(lower, case.support.lower_D, rtol=0, atol=1e-6)
            and np.allclose(upper, case.support.upper_D, rtol=0, atol=1e-6)
            and wind["direction_deg"] == int(case.wind_direction_deg)
        )
        support = SimpleNamespace(lower_D=lower, upper_D=upper, extent_D=upper-lower)
        centers, present, features = module_geometry(positions[:, :2], len(turbines))
        live_case = SimpleNamespace(
            case=case.case, index=case.index, layout_index=case.layout_index,
            wind_direction_deg=wind["direction_deg"], module_centers=centers,
            module_present=present, module_features=features, support=support,
            global_context=global_geometry_features(support, wind["direction_deg"], len(turbines)),
        )
        if unchanged:
            native_axes = [case.run.x_m, case.run.y_m, case.run.z_m]
            indices = [np.rint(np.linspace(0, len(axis)-1, n)).astype(int) for axis, n in zip(native_axes, shape)]
            axes = [np.asarray(axis[index], dtype=np.float32) / D_M for axis, index in zip(native_axes, indices)]
        else:
            axes = [np.linspace(low, high, n, dtype=np.float32) for low, high, n in zip(lower, upper, shape)]
        coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        with self.lock, torch.no_grad():
            model = self._load_model()
            scene = model.make_scene(live_case, module_ids=ids)
            prepared = model.core.prepare(scene)
            organization = self._organization(prepared, scene)
            predictions = []
            for begin in range(0, len(coords), 512):
                query = model.make_query(scene, torch.from_numpy(coords[begin:begin+512]))
                result = model.core.decode(prepared, query)
                velocity, _, _ = model.transforms.physical_velocity(result.joint_fields, query.coordinates)
                predictions.append(velocity[0].cpu().numpy())
            prediction = np.concatenate(predictions, axis=0)
        errors = None
        reference = None
        if unchanged:
            ix, iy, iz = indices
            reference = np.asarray(case.run.U_structured[np.ix_(iz, iy, ix)], dtype=np.float32).transpose(2, 1, 0, 3).reshape(-1, 3)
            difference = prediction.astype(np.float64) - reference.astype(np.float64)
            denominator = np.sqrt(np.sum(reference.astype(np.float64) ** 2, axis=0))
            errors = {
                "reference": "native CFD velocity at the same displayed cell centers",
                "mask": "all sampled native volume cells", "units": "m/s",
                "components": {
                    name: {"relative_l2": float(np.linalg.norm(difference[:, i]) / denominator[i]) if denominator[i] else None,
                           "rmse_m_s": float(np.sqrt(np.mean(difference[:, i] ** 2)))}
                    for i, name in enumerate(("Ux", "Uy", "Uz"))
                },
            }
        scene_response = {**original, "turbines": turbines, "domain": domain, "wind": wind}
        return {
            "scene": scene_response,
            "grid": {
                "shape": list(shape), "bounds": domain,
                "axes": {name: axis.tolist() for name, axis in zip(("x", "y", "z"), axes)},
                "order": "x-major, z-fast, then component", "velocity": prediction.reshape(-1).tolist(),
                "reference_velocity": reference.reshape(-1).tolist() if reference is not None else None,
                "fluid_mask": [True] * len(coords) if unchanged else None,
                "reference_profile": {
                    "z_D": model.transforms.background_profile.bin_centers_D.tolist(),
                    "velocity_mps": model.transforms.background_profile.values_mps.tolist(),
                    "source": "checkpoint training vertical profile",
                },
            },
            "organization": organization, "errors": errors,
            "provenance": {
                "run": RUN_NAME, "checkpoint": str(CHECKPOINT), "epoch": 2500,
                "partition": partition, "training_cases": 420, "device": "cpu",
                "inference_seconds": time.perf_counter() - start,
                "unchanged_dataset_case": bool(unchanged),
                "reference_available": bool(unchanged),
                "warning": "Edited scenes have no CFD reference and may be out of distribution. Direction is a category in the wind-aligned frame.",
            },
        }


app = FastAPI(title="WindFarm HONF local demo")
_demo = None
_demo_lock = threading.Lock()


def get_demo() -> Demo:
    global _demo
    with _demo_lock:
        if _demo is None:
            _demo = Demo()
        return _demo


@app.get("/api/meta")
def meta():
    return get_demo().meta()


@app.get("/api/cases/{case_id}")
def case(case_id: str):
    native, partition = get_demo().case(case_id)
    return _scene_dict(native, partition)


@app.post("/api/reconstruct")
def reconstruct(request: Reconstruction):
    return get_demo().reconstruct(request)


app.mount("/", StaticFiles(directory=ROOT / "Case_WindFarm/demo/dist", html=True, check_dir=False), name="frontend")
