from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from design_validation import model_limits, validate_design
from generative_service import GenerativeUnavailableError
from inference_service import InferenceService
from model_registry import registry, resolve_phase_bin_config
from schemas import DesignRequest, InferenceResponse
from settings import settings


app = FastAPI(title="ModularDT Multi-Cylinder Web Demo API")
inference_service = InferenceService(registry)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_example_designs():
    path = settings.storage_dir / "example_designs.json"
    if not path.exists():
        return {"examples": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.on_event("startup")
def preload_models() -> None:
    if os.environ.get("MODULARDT_WEB_DEMO_ENABLE_PRELOAD", "0").lower() not in {"1", "true", "yes"}:
        return
    for entry in registry.list_entries():
        if not entry.preload or entry.mode != "deterministic" or entry.missing_files:
            continue
        def _preload(model_id: str = entry.id) -> None:
            current_entry = registry.get_entry(model_id)
            try:
                inference_service.det_service.preload(current_entry)
                registry.set_runtime_error(current_entry.id, None)
            except Exception as exc:
                registry.set_runtime_error(current_entry.id, str(exc))

        thread = threading.Thread(target=_preload, name=f"preload-{entry.id}", daemon=True)
        thread.start()


@app.get("/api/preload-status")
def preload_status():
    return {"models": registry.list_public()}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "demo_root": str(settings.demo_root),
        "src_dir": str(settings.src_dir),
        "manifest_path": str(settings.manifest_path),
    }


@app.get("/api/models")
def list_models():
    registry.reload()
    return {"models": registry.list_public()}


@app.get("/api/example-designs")
def example_designs():
    try:
        return _read_example_designs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models/{model_id}/config")
def model_config(model_id: str):
    try:
        entry = registry.get_entry(model_id)
        config = entry.load_config_json()
        limits = model_limits(config)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        **limits,
        **resolve_phase_bin_config(config, entry.raw),
        "re_scale": limits.get("re_scale"),
        "expected_re_min": entry.raw.get("expected_re_min"),
        "expected_re_max": entry.raw.get("expected_re_max"),
        "fields": ["u", "v", "p", "omega"],
        "mode": entry.mode,
        "stage": entry.stage,
    }


@app.post("/api/design/validate")
def validate(request: DesignRequest):
    try:
        entry = registry.get_entry(request.model_id)
        config = entry.load_config_json()
        return validate_design(request, config, entry.raw)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/infer", response_model=InferenceResponse)
def infer(request: DesignRequest):
    try:
        return inference_service.infer(request)
    except GenerativeUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    path = inference_service.result_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/jobs/{job_id}/frames/{field}/{frame_id}")
def job_frame(job_id: str, field: str, frame_id: str):
    normalized = Path(frame_id).stem
    path = settings.cache_dir / job_id / "frames" / field / f"{normalized}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/export.npz")
def job_export(job_id: str):
    path = settings.cache_dir / job_id / "fields.npz"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export not found.")
    return FileResponse(path, media_type="application/octet-stream", filename=f"{job_id}.npz")
