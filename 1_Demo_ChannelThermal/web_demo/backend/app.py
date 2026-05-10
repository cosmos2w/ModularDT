from __future__ import annotations

import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from inverse_registry import inverse_registry
from inverse_service import InverseService
from model_registry import registry
from schemas import DesignRequest, ForwardSimulationRequest, ForwardSimulationResponse, InferenceResponse, InverseRunRequest, InverseRunResponse
from settings import settings
from thermal_service import ThermalInferenceService


app = FastAPI(title="ModularDT ChannelThermal Web Demo API")
thermal_service = ThermalInferenceService(registry)
inverse_service = InverseService(inverse_registry, registry)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_job_file(job_dir: Path, rel_path: str) -> Path:
    path = (job_dir / rel_path).resolve()
    try:
        path.relative_to(job_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid file path.") from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return path


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "demo_root": str(settings.demo_root),
        "src_dir": str(settings.src_dir),
        "manifest_path": str(settings.manifest_path),
        "inverse_manifest_path": str(settings.inverse_manifest_path),
    }


@app.get("/api/models")
def list_models():
    registry.reload()
    return {"models": registry.list_public()}


@app.get("/api/models/{model_id}/config")
def model_config(model_id: str):
    try:
        return thermal_service.model_config(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/reference-cases")
def reference_cases(split: str = "test", limit: int = 80):
    try:
        return {"cases": thermal_service.list_reference_cases(split=split, limit=limit)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/design/validate")
def validate(request: DesignRequest):
    try:
        return thermal_service.validate(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/infer", response_model=InferenceResponse)
def infer(request: DesignRequest):
    try:
        return thermal_service.infer(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    path = thermal_service.result_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/jobs/{job_id}/frames/{field}/{frame_id}")
def job_frame(job_id: str, field: str, frame_id: str):
    del frame_id
    path = settings.cache_dir / job_id / "frames" / f"{Path(field).stem}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/files/{rel_path:path}")
def job_file(job_id: str, rel_path: str):
    path = _safe_job_file(settings.cache_dir / job_id, rel_path)
    media = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=path.name)


@app.get("/api/jobs/{job_id}/export.npz")
def job_export(job_id: str):
    path = settings.cache_dir / job_id / "fields.npz"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export not found.")
    return FileResponse(path, media_type="application/octet-stream", filename=f"{job_id}.npz")


@app.post("/api/simulate-forward", response_model=ForwardSimulationResponse)
def simulate_forward(request: ForwardSimulationRequest = Body(...)):
    try:
        return thermal_service.run_forward_simulation(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/simulate-forward/jobs/{job_id}")
def simulation_job_status(job_id: str):
    path = thermal_service.simulation_status_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown simulation job_id: {job_id}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/simulate-forward/jobs/{job_id}/result")
def simulation_job_result(job_id: str):
    path = thermal_service.simulation_result_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Simulation result is not ready: {job_id}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/simulate-forward/jobs/{job_id}/frames/{field}/{frame_id}")
def simulation_job_frame(job_id: str, field: str, frame_id: str):
    del frame_id
    path = settings.cache_dir / "forward_sim_jobs" / job_id / "frames" / f"{Path(field).stem}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/simulate-forward/jobs/{job_id}/files/{rel_path:path}")
def simulation_job_file(job_id: str, rel_path: str):
    path = _safe_job_file(settings.cache_dir / "forward_sim_jobs" / job_id, rel_path)
    media = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=path.name)


@app.get("/api/simulate-forward/jobs/{job_id}/export.npz")
def simulation_job_export(job_id: str):
    path = settings.cache_dir / "forward_sim_jobs" / job_id / "simulation_fields.npz"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Simulation export not found.")
    return FileResponse(path, media_type="application/octet-stream", filename=f"{job_id}_simulation.npz")


@app.get("/api/inverse/models")
def inverse_models():
    try:
        return {"models": inverse_service.list_models()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/inverse/target-presets")
def inverse_target_presets():
    try:
        return {"presets": inverse_service.list_target_presets()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/inverse/kpis")
def inverse_kpis():
    try:
        return {"kpis": inverse_service.list_kpis()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/inverse/run", response_model=InverseRunResponse)
def inverse_run(request: InverseRunRequest = Body(...)):
    try:
        job_id = inverse_service.run_inverse(request)
        return {
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/api/inverse/jobs/{job_id}",
            "result_url": f"/api/inverse/jobs/{job_id}/result",
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/inverse/jobs/{job_id}")
def inverse_job_status(job_id: str):
    try:
        return inverse_service.get_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/inverse/jobs/{job_id}/result")
def inverse_job_result(job_id: str):
    try:
        return inverse_service.get_result(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/inverse/jobs/{job_id}/candidates")
def inverse_candidates(job_id: str):
    try:
        return inverse_service.get_candidates(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/inverse/jobs/{job_id}/debug-files")
def inverse_debug_files(job_id: str):
    try:
        return {"files": inverse_service.debug_files(job_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/inverse/jobs/{job_id}/files/{rel_path:path}")
def inverse_job_file(job_id: str, rel_path: str):
    path = _safe_job_file(settings.inverse_jobs_dir / job_id, rel_path)
    media = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=path.name)
