from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import quote

import numpy as np

from inverse_registry import InverseModelRegistry
from model_registry import ModelRegistry
from schemas import InverseRunRequest
from settings import settings


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


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


def _read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else dict(default)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(dict(payload)), f, indent=2)


def _tail(path: Path, lines: int = 80) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in deque(f, maxlen=max(int(lines), 1))]


def _device_arg() -> str:
    requested = settings.device.lower().strip()
    return "cuda:0" if requested == "auto" else settings.device


def _format_kpi_target(entry: Mapping[str, Any]) -> Dict[str, Any]:
    mode = str(entry.get("mode", "max"))
    target: Dict[str, Any] = {"mode": mode, "weight": float(entry.get("weight", 1.0))}
    if mode == "max":
        target["high"] = entry.get("high", entry.get("value"))
    elif mode == "min":
        target["low"] = entry.get("low", entry.get("value"))
    elif mode == "range":
        target["low"] = entry.get("low")
        target["high"] = entry.get("high")
    else:
        target["value"] = entry.get("value", entry.get("high", entry.get("low")))
    return {key: value for key, value in target.items() if value is not None}


def _file_url(job_id: str, job_dir: Path, path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    rel = path.resolve().relative_to(job_dir.resolve()).as_posix()
    return f"/api/inverse/jobs/{quote(job_id)}/files/{quote(rel, safe='/')}"


class InverseService:
    def __init__(self, inverse_registry: InverseModelRegistry, model_registry: ModelRegistry):
        self.inverse_registry = inverse_registry
        self.model_registry = model_registry
        self._threads: Dict[str, threading.Thread] = {}

    def _job_dir(self, job_id: str) -> Path:
        return settings.inverse_jobs_dir / job_id

    def _status_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "status.json"

    def _write_status(self, job_id: str, status: str, **payload: Any) -> None:
        current = _read_json(self._status_path(job_id))
        current.update({"job_id": job_id, "status": status, "updated_at": _now(), **payload})
        _write_json(self._status_path(job_id), current)

    def list_models(self) -> List[Dict[str, Any]]:
        self.inverse_registry.reload()
        return self.inverse_registry.list_public()

    def list_target_presets(self) -> List[Dict[str, Any]]:
        presets: List[Dict[str, Any]] = []
        for path in sorted(settings.inverse_target_presets_dir.glob("*.json")):
            payload = _read_json(path)
            name = str(payload.get("name", path.stem))
            presets.append({"name": name, "label": name.replace("_", " "), "path": str(path), "target": payload})
        return presets

    def list_kpis(self) -> List[Dict[str, Any]]:
        names: List[str] = []
        for entry in self.inverse_registry.list_entries():
            cfg = _read_json(entry.config_path)
            target_cfg = cfg.get("target_kpis", {}) if isinstance(cfg.get("target_kpis"), Mapping) else {}
            for name in target_cfg.get("names", []):
                if str(name) not in names:
                    names.append(str(name))
        if not names:
            from thermal_inverse_kpi import DEFAULT_KPI_NAMES

            names = [str(name) for name in DEFAULT_KPI_NAMES]
        return [
            {
                "name": name,
                "label": name.replace("_", " "),
                "default_mode": "max",
                "default_weight": 1.0,
            }
            for name in names
        ]

    def _target_payload(self, request: InverseRunRequest) -> Dict[str, Any]:
        enabled = [item for item in request.kpis if item.enabled]
        if not enabled:
            raise ValueError("At least one KPI target row must be enabled.")
        payload: Dict[str, Any] = {
            "name": request.target_name or "web_demo_target",
            "num_modules_min": int(request.constraints.num_modules_min),
            "num_modules_max": int(request.constraints.num_modules_max),
            "min_center_distance": float(request.constraints.min_center_distance),
            "wall_clearance": float(request.constraints.wall_clearance),
            "inlet_clearance": float(request.constraints.inlet_clearance),
            "outlet_clearance": float(request.constraints.outlet_clearance),
            "kpis": {item.name: _format_kpi_target(_model_dump(item)) for item in enabled},
            "preferences": dict(request.preferences or {}),
        }
        if request.constraints.heat_power_total is not None:
            payload["heat_power_total"] = float(request.constraints.heat_power_total)
        return payload

    def _validate_request(self, request: InverseRunRequest) -> None:
        if request.constraints.num_modules_min > request.constraints.num_modules_max:
            raise ValueError("num_modules_min cannot exceed num_modules_max.")
        if request.sampling.n_samples > settings.max_inverse_n_samples:
            raise ValueError(f"n_samples must be <= {settings.max_inverse_n_samples}.")
        inv = self.inverse_registry.get_entry(request.inverse_model_id)
        if inv.missing_files or not inv.enabled:
            raise FileNotFoundError(inv.reason_unavailable() or "Inverse model is unavailable.")
        fwd = self.model_registry.get_entry(request.forward_model_id)
        if fwd.missing_files or not fwd.enabled:
            raise FileNotFoundError(fwd.reason_unavailable() or "Forward model is unavailable.")

    def run_inverse(self, request: InverseRunRequest) -> str:
        self.inverse_registry.reload()
        self.model_registry.reload()
        self._validate_request(request)
        import uuid

        job_id = uuid.uuid4().hex[:16]
        job_dir = self._job_dir(job_id)
        out_dir = job_dir / "evaluate_output"
        job_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        inv = self.inverse_registry.get_entry(request.inverse_model_id)
        fwd = self.model_registry.get_entry(request.forward_model_id)
        target_path = job_dir / "target.json"
        target_payload = self._target_payload(request)
        _write_json(target_path, target_payload)
        request_path = job_dir / "request.json"
        _write_json(request_path, _model_dump(request))
        self._write_status(job_id, "queued", started_at=_now(), request=_model_dump(request), target=target_payload)

        command = [
            sys.executable,
            str(settings.src_dir / "evaluate_inverse.py"),
            "--inverse-run",
            str(inv.run_dir),
            "--checkpoint-name",
            inv.checkpoint_path.name,
            "--target",
            str(target_path),
            "--reference-split",
            str(request.reference_split),
            "--reference-case-index",
            str(int(request.reference_case_index)),
            "--n-samples",
            str(int(request.sampling.n_samples)),
            "--n-steps",
            str(int(request.sampling.n_steps)),
            "--count-mode",
            str(request.sampling.count_mode),
            "--seed",
            str(int(request.sampling.seed)),
            "--device",
            _device_arg(),
            "--output-dir",
            str(out_dir),
            "--forward-run-dir",
            str(fwd.run_dir),
            "--forward-checkpoint-name",
            fwd.checkpoint_path.name,
        ]

        thread = threading.Thread(target=self._run_worker, args=(job_id, command, job_dir, out_dir), daemon=True)
        self._threads[job_id] = thread
        thread.start()
        return job_id

    def _run_worker(self, job_id: str, command: List[str], job_dir: Path, out_dir: Path) -> None:
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        self._write_status(job_id, "running", command=command)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(command, cwd=settings.demo_root, stdout=stdout, stderr=stderr)
            code = proc.wait()
        if code != 0:
            self._write_status(
                job_id,
                "failed",
                returncode=int(code),
                stdout_tail=_tail(stdout_path),
                stderr_tail=_tail(stderr_path),
            )
            return
        try:
            result = self._assemble_result(job_id, job_dir, out_dir, returncode=int(code))
            _write_json(job_dir / "result.json", result)
            self._write_status(job_id, "complete", result_url=f"/api/inverse/jobs/{job_id}/result", returncode=int(code))
        except Exception as exc:
            self._write_status(job_id, "failed", error=str(exc), stdout_tail=_tail(stdout_path), stderr_tail=_tail(stderr_path))

    def _assemble_result(self, job_id: str, job_dir: Path, out_dir: Path, *, returncode: int) -> Dict[str, Any]:
        candidates_payload = _read_json(out_dir / "candidates.json", {"candidates": [], "target": {}})
        summary = _read_json(out_dir / "verification_summary.json")
        target = candidates_payload.get("target", {})
        candidates = candidates_payload.get("candidates", [])
        artifacts = {
            path.stem: _file_url(job_id, job_dir, path)
            for path in sorted(out_dir.glob("*.png"))
        }
        for path in ("candidates.csv", "kpi_scores.csv", "top_candidates.npz", "target_spec_resolved.json", "verification_summary.json"):
            artifacts[Path(path).stem] = _file_url(job_id, job_dir, out_dir / path)
        result = {
            "job_id": job_id,
            "status": "complete",
            "returncode": int(returncode),
            "summary": summary,
            "target": target,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "artifacts": {key: value for key, value in artifacts.items() if value},
            "stdout_tail": _tail(job_dir / "stdout.log", 40),
            "stderr_tail": _tail(job_dir / "stderr.log", 40),
        }
        _write_json(job_dir / "candidates.json", {"target": target, "candidates": candidates})
        return result

    def get_status(self, job_id: str) -> Dict[str, Any]:
        path = self._status_path(job_id)
        if not path.exists():
            raise KeyError(f"Unknown inverse job_id: {job_id}")
        status = _read_json(path)
        if status.get("status") in {"running", "queued"}:
            status["stdout_tail"] = _tail(self._job_dir(job_id) / "stdout.log", 30)
            status["stderr_tail"] = _tail(self._job_dir(job_id) / "stderr.log", 30)
        return status

    def get_result(self, job_id: str) -> Dict[str, Any]:
        path = self._job_dir(job_id) / "result.json"
        if not path.exists():
            raise KeyError(f"Result is not ready for inverse job_id: {job_id}")
        return _read_json(path)

    def get_candidates(self, job_id: str) -> Dict[str, Any]:
        path = self._job_dir(job_id) / "candidates.json"
        if not path.exists():
            result = self.get_result(job_id)
            return {"target": result.get("target", {}), "candidates": result.get("candidates", [])}
        return _read_json(path)
