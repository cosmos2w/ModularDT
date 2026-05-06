from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import quote

import numpy as np

from inference_service import InferenceService
from inverse_registry import InverseModelRegistry
from model_registry import ModelRegistry
from schemas import (
    CandidateSimulationValidationRequest,
    Cylinder,
    DesignRequest,
    GenerativeOptions,
    InverseRunRequest,
    InverseVerificationSpec,
)
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
    if isinstance(value, dict):
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


def _tail(path: Path, lines: int = 60) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in deque(f, maxlen=max(int(lines), 1))]


def _device_arg() -> str:
    requested = settings.device.lower().strip()
    return "cuda:0" if requested == "auto" else settings.device


def _file_url(job_id: str, job_dir: Path, path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    rel = path.resolve().relative_to(job_dir.resolve()).as_posix()
    return f"/api/inverse/jobs/{quote(job_id)}/files/{quote(rel, safe='/')}"


def _score_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _rank_or_none(value: Any) -> Optional[int]:
    if value is None or str(value) == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    sample = int(candidate.get("sample_index", candidate.get("sample", 0)))
    rank = _rank_or_none(candidate.get("rank"))
    if rank is None:
        return f"sample_{sample:03d}"
    return f"rank_{rank:03d}_sample_{sample:03d}"


def _format_kpi_target(entry: Mapping[str, Any]) -> Dict[str, Any]:
    mode = str(entry.get("mode", "exact"))
    target: Dict[str, Any] = {"mode": mode, "weight": float(entry.get("weight", 1.0))}
    for key in ("value", "low", "high"):
        if entry.get(key) is not None:
            target[key] = float(entry[key])
    return target


def _kpi_pass(mode: str, achieved: Optional[float], target: Mapping[str, Any]) -> Optional[bool]:
    if achieved is None:
        return None
    if mode == "range":
        low = target.get("low")
        high = target.get("high")
        return (low is None or achieved >= float(low)) and (high is None or achieved <= float(high))
    if mode == "max":
        high = target.get("high", target.get("value"))
        return None if high is None else achieved <= float(high)
    if mode == "min":
        low = target.get("low", target.get("value"))
        return None if low is None else achieved >= float(low)
    if mode == "exact":
        value = target.get("value")
        if value is None:
            return None
        tolerance = max(abs(float(value)) * 0.05, 1.0e-6)
        return abs(achieved - float(value)) <= tolerance
    return None


def _build_kpi_comparison(candidate: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
    kpis = candidate.get("kpis", {}) if isinstance(candidate.get("kpis"), Mapping) else {}
    sim = candidate.get("simulation_verification", {}) if isinstance(candidate.get("simulation_verification"), Mapping) else {}
    sim_kpis = sim.get("ground_truth_kpis", {}) if isinstance(sim.get("ground_truth_kpis"), Mapping) else {}
    target_kpis = target.get("kpis", {}) if isinstance(target.get("kpis"), Mapping) else {}
    rows: Dict[str, Any] = {}
    for name, raw_entry in target_kpis.items():
        entry = raw_entry if isinstance(raw_entry, Mapping) else {"mode": "exact", "value": raw_entry}
        mode = str(entry.get("mode", "exact"))
        achieved = _score_or_none(kpis.get(name))
        simulated = _score_or_none(sim_kpis.get(name))
        formatted = _format_kpi_target(entry)
        rows[str(name)] = {
            "target": formatted,
            "achieved": achieved,
            "simulation": simulated,
            "pass": _kpi_pass(mode, achieved, formatted),
        }
    return rows


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


class InverseService:
    def __init__(
        self,
        inverse_registry: InverseModelRegistry,
        model_registry: ModelRegistry,
        inference_service: InferenceService,
    ):
        self.inverse_registry = inverse_registry
        self.model_registry = model_registry
        self.inference_service = inference_service
        self._lock = threading.Lock()
        self._simulation_semaphore = threading.Semaphore(max(int(settings.max_concurrent_simulation_jobs), 1))
        self._simulation_threads: Dict[str, threading.Thread] = {}

    def _reload_registries(self) -> None:
        self.inverse_registry.reload()
        self.model_registry.reload()

    def list_models(self) -> List[Dict[str, Any]]:
        self.inverse_registry.reload()
        return self.inverse_registry.list_public()

    def list_target_presets(self) -> List[Dict[str, Any]]:
        presets: List[Dict[str, Any]] = []
        for path in sorted(settings.inverse_target_presets_dir.glob("*.json")):
            payload = _read_json(path)
            name = str(payload.get("name", path.stem))
            presets.append(
                {
                    "name": name,
                    "label": name.replace("_", " "),
                    "path": str(path),
                    "target": payload,
                }
            )
        return presets

    def list_kpis(self) -> List[Dict[str, Any]]:
        self.inverse_registry.reload()
        names: List[str] = []
        for entry in self.inverse_registry.list_entries():
            try:
                cfg = entry.load_config_json()
            except Exception:
                continue
            configured = cfg.get("target_kpis", {}).get("names", []) if isinstance(cfg.get("target_kpis"), Mapping) else []
            for name in configured:
                if str(name) not in names:
                    names.append(str(name))
        if not names:
            from inverse_kpi import DEFAULT_KPI_NAMES

            names = [str(name) for name in DEFAULT_KPI_NAMES]
        return [
            {
                "name": name,
                "label": name.replace("_", " "),
                "default_mode": "range" if name in {"enstrophy", "wake_mixing"} else "max",
                "default_weight": 1.0,
            }
            for name in names
        ]

    def _job_dir(self, job_id: str) -> Path:
        return settings.inverse_jobs_dir / job_id

    def _status_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "status.json"

    def _write_status(self, job_id: str, status: str, **payload: Any) -> None:
        current = _read_json(self._status_path(job_id))
        current.update({"job_id": job_id, "status": status, "updated_at": _now(), **payload})
        _write_json(self._status_path(job_id), current)

    def _validate_request(self, request: InverseRunRequest) -> None:
        enabled_kpis = [kpi for kpi in request.kpis if kpi.enabled]
        if not enabled_kpis:
            raise ValueError("At least one KPI target row must be enabled.")
        if request.constraints.num_cylinders_min > request.constraints.num_cylinders_max:
            raise ValueError("num_cylinders_min cannot exceed num_cylinders_max.")
        if request.sampling.n_samples > settings.max_inverse_n_samples:
            raise ValueError(f"n_samples must be <= {settings.max_inverse_n_samples}.")
        if request.sampling.verify_top_k > settings.max_inverse_verify_top_k:
            raise ValueError(f"verify_top_k must be <= {settings.max_inverse_verify_top_k}.")
        if request.sampling.save_verified_top_k > settings.max_inverse_save_verified_top_k:
            raise ValueError(f"save_verified_top_k must be <= {settings.max_inverse_save_verified_top_k}.")
        if request.sampling.verify_top_k > request.sampling.n_samples:
            raise ValueError("verify_top_k cannot exceed n_samples.")
        if request.sampling.save_verified_top_k > max(request.sampling.verify_top_k, 1):
            raise ValueError("save_verified_top_k cannot exceed verify_top_k.")

        inverse_entry = self.inverse_registry.get_entry(request.inverse_model_id)
        if inverse_entry.missing_files or not inverse_entry.enabled:
            raise FileNotFoundError(inverse_entry.reason_unavailable() or "Inverse model is unavailable.")

        forward_entry = self.model_registry.get_entry(request.verification.forward_verifier_model_id)
        if forward_entry.mode != request.verification.forward_backend:
            raise ValueError(
                "forward_verifier_model_id mode does not match forward_backend: "
                f"{forward_entry.mode} != {request.verification.forward_backend}."
            )
        if forward_entry.missing_files or not forward_entry.enabled:
            raise FileNotFoundError(forward_entry.reason_unavailable() or "Forward verifier model is unavailable.")

        for kpi in enabled_kpis:
            if kpi.mode == "exact" and kpi.value is None:
                raise ValueError(f"KPI {kpi.name!r} exact mode requires value.")
            if kpi.mode == "range" and (kpi.low is None or kpi.high is None):
                raise ValueError(f"KPI {kpi.name!r} range mode requires low and high.")
            if kpi.mode == "max" and kpi.high is None and kpi.value is None:
                raise ValueError(f"KPI {kpi.name!r} max mode requires high.")
            if kpi.mode == "min" and kpi.low is None and kpi.value is None:
                raise ValueError(f"KPI {kpi.name!r} min mode requires low.")

    def _generative_checkpoint_grid(self, forward_entry: Any) -> Optional[tuple[int, int]]:
        raw = getattr(forward_entry, "raw", {}) if forward_entry is not None else {}
        for nx_key, ny_key in (
            ("default_resolution_nx", "default_resolution_ny"),
            ("verifier_nx", "verifier_ny"),
            ("num_x", "num_y"),
        ):
            if raw.get(nx_key) and raw.get(ny_key):
                return int(raw[nx_key]), int(raw[ny_key])

        try:
            import torch

            try:
                ckpt = torch.load(forward_entry.checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(forward_entry.checkpoint_path, map_location="cpu")
            nx = ckpt.get("num_x")
            ny = ckpt.get("num_y")
            if nx and ny:
                return int(nx), int(ny)
        except Exception:
            return None
        return None

    def _normalize_request_for_verifier(self, request: InverseRunRequest) -> InverseRunRequest:
        if request.verification.forward_backend != "generative":
            return request
        forward_entry = self.model_registry.get_entry(request.verification.forward_verifier_model_id)
        grid = self._generative_checkpoint_grid(forward_entry)
        if grid is None:
            return request
        nx, ny = grid
        if int(request.verification.nx) == nx and int(request.verification.ny) == ny:
            return request
        if hasattr(request.verification, "model_copy"):
            verification = request.verification.model_copy(update={"nx": nx, "ny": ny})
            return request.model_copy(update={"verification": verification})
        verification = request.verification.copy(update={"nx": nx, "ny": ny})
        return request.copy(update={"verification": verification})

    def _target_payload(self, request: InverseRunRequest) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        for spec in request.kpis:
            if not spec.enabled:
                continue
            entry: Dict[str, Any] = {"mode": spec.mode, "weight": float(spec.weight)}
            if spec.mode == "exact":
                entry["value"] = float(spec.value)
            elif spec.mode == "range":
                entry["low"] = float(spec.low)
                entry["high"] = float(spec.high)
            elif spec.mode == "max":
                entry["high"] = float(spec.high if spec.high is not None else spec.value)
            elif spec.mode == "min":
                entry["low"] = float(spec.low if spec.low is not None else spec.value)
            elif spec.value is not None:
                entry["value"] = float(spec.value)
            kpis[spec.name] = entry

        preferences: Dict[str, Any] = {"min_center_distance": request.constraints.min_center_distance}
        if request.constraints.min_x_span is not None:
            preferences["min_x_span"] = request.constraints.min_x_span
        if request.constraints.min_y_span is not None:
            preferences["min_y_span"] = request.constraints.min_y_span

        return {
            "name": request.target_name or "web_inverse_target",
            "Re": request.constraints.re,
            "num_cylinders_min": request.constraints.num_cylinders_min,
            "num_cylinders_max": request.constraints.num_cylinders_max,
            "min_center_distance": request.constraints.min_center_distance,
            "kpis": kpis,
            "preferences": preferences,
        }

    def _command(self, request: InverseRunRequest, target_path: Path, output_dir: Path) -> List[str]:
        inverse_entry = self.inverse_registry.get_entry(request.inverse_model_id)
        forward_entry = self.model_registry.get_entry(request.verification.forward_verifier_model_id)
        cmd = [
            sys.executable,
            str(settings.src_dir / "evaluate_inverse.py"),
            "--inverse-run",
            str(inverse_entry.run_dir),
            "--checkpoint",
            inverse_entry.checkpoint_name,
            "--target-json",
            str(target_path),
            "--output-dir",
            str(output_dir),
            "--n-samples",
            str(request.sampling.n_samples),
            "--verify-top-k",
            str(request.sampling.verify_top_k),
            "--save-verified-top-k",
            str(request.sampling.save_verified_top_k),
            "--forward-backend",
            request.verification.forward_backend,
            "--phase-bins",
            str(request.verification.phase_bins),
            "--nx",
            str(request.verification.nx),
            "--ny",
            str(request.verification.ny),
            "--n-steps",
            str(request.sampling.n_steps),
            "--device",
            _device_arg(),
            "--no-simulation-verify",
        ]
        if request.sampling.seed is not None:
            cmd.extend(["--seed", str(request.sampling.seed)])

        if request.constraints.min_x_span is not None:
            cmd.extend(["--prefilter-min-x-span", str(request.constraints.min_x_span)])
        if request.constraints.min_y_span is not None:
            cmd.extend(["--prefilter-min-y-span", str(request.constraints.min_y_span)])
        if request.verification.forward_backend == "deterministic":
            cmd.extend(
                [
                    "--deterministic-run",
                    str(forward_entry.run_dir),
                    "--deterministic-checkpoint",
                    forward_entry.checkpoint_path.name,
                    "--deterministic-config",
                    forward_entry.config_path.name,
                ]
            )
        else:
            cmd.extend(
                [
                    "--generative-run",
                    str(forward_entry.run_dir),
                    "--generative-checkpoint",
                    forward_entry.checkpoint_path.name,
                    "--generative-config",
                    forward_entry.config_path.name,
                    "--generative-num-samples",
                    str(request.verification.generative_num_samples),
                    "--generative-n-steps",
                    str(request.verification.generative_n_steps),
                    "--generative-ode-solver",
                    request.verification.generative_ode_solver,
                    "--uncertainty-penalty-weight",
                    str(request.verification.uncertainty_penalty_weight),
                ]
            )
        return cmd

    def run_inverse(self, request: InverseRunRequest) -> str:
        self._reload_registries()
        request = self._normalize_request_for_verifier(request)
        self._validate_request(request)
        job_id = uuid.uuid4().hex[:16]
        job_dir = self._job_dir(job_id)
        output_dir = job_dir / "evaluate_output"
        job_dir.mkdir(parents=True, exist_ok=True)
        target_payload = self._target_payload(request)
        target_path = job_dir / "target.json"
        cmd = self._command(request, target_path, output_dir)
        _write_json(job_dir / "request.json", _model_dump(request))
        _write_json(target_path, target_payload)
        _write_json(job_dir / "command.json", {"argv": cmd, "cwd": str(settings.demo_root)})
        self._write_status(
            job_id,
            "queued",
            created_at=_now(),
            status_url=f"/api/inverse/jobs/{job_id}",
            result_url=f"/api/inverse/jobs/{job_id}/result",
        )
        thread = threading.Thread(target=self._run_job_thread, args=(job_id, cmd), name=f"inverse-{job_id}", daemon=True)
        thread.start()
        return job_id

    def _run_job_thread(self, job_id: str, cmd: Sequence[str]) -> None:
        job_dir = self._job_dir(job_id)
        stdout_log = job_dir / "stdout.log"
        output_dir = job_dir / "evaluate_output"
        try:
            self._write_status(job_id, "running", started_at=_now(), stdout_log=str(stdout_log))
            env = dict(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            with stdout_log.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    list(cmd),
                    cwd=str(settings.demo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"evaluate_inverse.py failed with exit code {return_code}.")

            self._write_status(job_id, "parsing_results")
            result = self._parse_output(job_id, output_dir)
            self._write_status(
                job_id,
                "complete",
                completed_at=_now(),
                result_url=f"/api/inverse/jobs/{job_id}/result",
                candidate_count=len(result.get("candidates", [])),
            )
        except Exception as exc:
            self._write_status(job_id, "error", completed_at=_now(), error=str(exc), log_tail=_tail(stdout_log))

    def _candidate_dir_for(self, job_dir: Path, candidate: Mapping[str, Any]) -> Optional[Path]:
        sample = int(candidate.get("sample_index", candidate.get("sample", 0)))
        roots = [job_dir / "candidates", job_dir / "evaluate_output" / "candidates"]
        for root in roots:
            matches = sorted(root.glob(f"*sample_{sample:03d}")) if root.exists() else []
            if matches:
                return matches[0]
        return None

    def _candidate_artifacts(self, job_id: str, job_dir: Path, cand_dir: Optional[Path]) -> Dict[str, Any]:
        if cand_dir is None:
            return {}
        images = {
            "ml_flow": _file_url(job_id, job_dir, cand_dir / "ml_flow.png"),
            "ml_cycle": _file_url(job_id, job_dir, cand_dir / "ml_cycle.gif"),
            "organization_physical": _file_url(job_id, job_dir, _first_existing(cand_dir.glob("candidate_*_organization_physical.png"))),
            "organization_matrices": _file_url(job_id, job_dir, _first_existing(cand_dir.glob("candidate_*_organization_matrices.png"))),
            "organization_sankey": _file_url(job_id, job_dir, _first_existing(cand_dir.glob("candidate_*_organization_sankey.png"))),
            "organization_schematic": _file_url(job_id, job_dir, _first_existing(cand_dir.glob("candidate_*_organization_schematic.png"))),
            "simulation_flow": _file_url(job_id, job_dir, cand_dir / "simulation_flow.png"),
            "simulation_cycle": _file_url(job_id, job_dir, cand_dir / "simulation_cycle.gif"),
            "simulation_kpi_comparison": _file_url(job_id, job_dir, cand_dir / "simulation_kpi_comparison.png"),
            "ml_vs_simulation_field": _file_url(job_id, job_dir, cand_dir / "ml_vs_simulation_field.png"),
            "ml_vs_simulation_cycle": _file_url(job_id, job_dir, cand_dir / "ml_vs_simulation_cycle.gif"),
        }
        files = {
            "candidate_result": _file_url(job_id, job_dir, cand_dir / "candidate_result.json"),
            "generated_verifier_cycle": _file_url(job_id, job_dir, cand_dir / "generated_verifier_cycle.npz"),
            "simulation_verification": _file_url(job_id, job_dir, cand_dir / "simulation_verification.json"),
        }
        return {
            "candidate_dir": str(cand_dir),
            "image_urls": {key: value for key, value in images.items() if value},
            "artifact_urls": {key: value for key, value in files.items() if value},
        }

    def _web_candidate(self, job_id: str, raw: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
        job_dir = self._job_dir(job_id)
        cand_dir = self._candidate_dir_for(job_dir, raw)
        sample = int(raw.get("sample_index", raw.get("sample", 0)))
        rank = _rank_or_none(raw.get("rank"))
        sim_status = "complete" if raw.get("simulation_verified") or raw.get("simulation_verification") else "not_started"
        simulation_status_path = self._simulation_status_path(job_id, _candidate_id(raw))
        if simulation_status_path.exists():
            sim_status = _read_json(simulation_status_path).get("status", sim_status)
        candidate = {
            "id": _candidate_id(raw),
            "rank": rank,
            "sample_index": sample,
            "score": _score_or_none(raw.get("score")),
            "centers": raw.get("centers", []),
            "count": int(raw.get("num_cylinders", raw.get("count", len(raw.get("centers", []))))),
            "validity": raw.get("validity", {}),
            "kpis": raw.get("kpis", {}),
            "kpis_std": raw.get("kpis_std", {}),
            "kpi_comparison": _build_kpi_comparison(raw, target),
            "per_kpi_errors": raw.get("per_kpi_errors", {}),
            "constraint_penalty": raw.get("constraint_penalty"),
            "latent_consistency": raw.get("latent_consistency"),
            "verifier_backend": raw.get("verifier_backend"),
            "quick_validation_status": "complete" if raw.get("verified") else raw.get("quick_validation_status", "not_started"),
            "simulation_validation_status": sim_status,
            "simulation_verification": raw.get("simulation_verification"),
            "raw": raw,
        }
        candidate.update(self._candidate_artifacts(job_id, job_dir, cand_dir))
        return candidate

    def _parse_output(self, job_id: str, output_dir: Path) -> Dict[str, Any]:
        job_dir = self._job_dir(job_id)
        candidates_root = output_dir / "candidates"
        if candidates_root.exists():
            shutil.copytree(candidates_root, job_dir / "candidates", dirs_exist_ok=True)
        payload_path = output_dir / "inverse_candidates.json"
        payload = _read_json(payload_path)
        if not payload:
            raise FileNotFoundError(f"Missing evaluator output: {payload_path}")
        target = payload.get("target", _read_json(job_dir / "target.json"))
        raw_candidates = payload.get("candidates", [])
        web_candidates = [self._web_candidate(job_id, raw, target) for raw in raw_candidates]
        web_candidates.sort(
            key=lambda item: (
                item["rank"] is None,
                item["rank"] if item["rank"] is not None else 999999,
                item["sample_index"],
            )
        )
        request_payload = _read_json(job_dir / "request.json")
        domain = self._domain_from_job_request(request_payload)
        top_files = {
            "sampled_layouts": _file_url(job_id, job_dir, output_dir / "sampled_layouts_by_score.png"),
            "kpi_target_vs_achieved": _file_url(job_id, job_dir, output_dir / "kpi_target_vs_achieved.png"),
            "layout_diversity": _file_url(job_id, job_dir, output_dir / "layout_diversity.png"),
            "inverse_candidates_csv": _file_url(job_id, job_dir, output_dir / "inverse_candidates.csv"),
            "inverse_candidates_json": _file_url(job_id, job_dir, output_dir / "inverse_candidates.json"),
        }
        result = {
            "job_id": job_id,
            "status": "complete",
            "target": target,
            "request": request_payload,
            "sampling": request_payload.get("sampling", {}),
            "verification": request_payload.get("verification", {}),
            "constraints": request_payload.get("constraints", {}),
            "domain": domain,
            "inverse_run": payload.get("inverse_run"),
            "checkpoint": payload.get("checkpoint"),
            "forward_checkpoint": payload.get("forward_checkpoint"),
            "forward_verifier_backend": payload.get("forward_verifier_backend"),
            "output_dir": str(output_dir),
            "files": {key: value for key, value in top_files.items() if value},
            "candidates": web_candidates,
        }
        _write_json(job_dir / "result.json", result)
        _write_json(job_dir / "candidates.json", {"candidates": web_candidates})
        return result

    def _domain_from_job_request(self, request_payload: Mapping[str, Any]) -> Dict[str, Any]:
        default_domain = {"length_x": 24.0, "length_y": 12.0}
        try:
            entry = self.inverse_registry.get_entry(str(request_payload.get("inverse_model_id", "")))
            cfg = entry.load_config_json()
            inverse_cfg = cfg.get("inverse_model", {}) if isinstance(cfg.get("inverse_model"), Mapping) else {}
            default_domain = {
                "length_x": float(inverse_cfg.get("domain_length_x", 24.0)),
                "length_y": float(inverse_cfg.get("domain_length_y", 12.0)),
                "max_num_cylinders": int(inverse_cfg.get("max_num_cylinders", 8)),
            }
        except Exception:
            pass
        verification = request_payload.get("verification", {}) if isinstance(request_payload.get("verification"), Mapping) else {}
        default_domain.update(
            {
                "phase_bins": verification.get("phase_bins", 12),
                "resolution_nx": verification.get("nx", 96),
                "resolution_ny": verification.get("ny", 48),
            }
        )
        return default_domain

    def get_status(self, job_id: str) -> Dict[str, Any]:
        status = _read_json(self._status_path(job_id))
        if not status:
            raise KeyError(f"Unknown inverse job_id: {job_id}")
        status["log_tail"] = _tail(self._job_dir(job_id) / "stdout.log")
        return status

    def get_result(self, job_id: str) -> Dict[str, Any]:
        path = self._job_dir(job_id) / "result.json"
        if not path.exists():
            raise KeyError(f"Result is not ready for inverse job_id: {job_id}")
        return _read_json(path)

    def get_candidates(self, job_id: str) -> Dict[str, Any]:
        path = self._job_dir(job_id) / "candidates.json"
        if not path.exists():
            raise KeyError(f"Candidates are not ready for inverse job_id: {job_id}")
        return _read_json(path)

    def _candidate_list(self, job_id: str) -> List[Dict[str, Any]]:
        return list(self.get_candidates(job_id).get("candidates", []))

    def get_candidate(self, job_id: str, candidate_id: str) -> Dict[str, Any]:
        for candidate in self._candidate_list(job_id):
            if self._matches_candidate(candidate, candidate_id):
                return candidate
        raise KeyError(f"Unknown candidate_id: {candidate_id}")

    def _matches_candidate(self, candidate: Mapping[str, Any], candidate_id: str) -> bool:
        cid = str(candidate.get("id", ""))
        sample = int(candidate.get("sample_index", -1))
        rank = candidate.get("rank")
        return candidate_id in {cid, f"sample_{sample:03d}", str(sample), f"rank_{int(rank):03d}" if rank is not None else ""}

    def _replace_candidate(self, job_id: str, next_candidate: Dict[str, Any]) -> None:
        job_dir = self._job_dir(job_id)
        with self._lock:
            candidates_payload = _read_json(job_dir / "candidates.json", {"candidates": []})
            candidates = list(candidates_payload.get("candidates", []))
            replaced = False
            for idx, candidate in enumerate(candidates):
                if self._matches_candidate(candidate, str(next_candidate.get("id", ""))):
                    candidates[idx] = next_candidate
                    replaced = True
                    break
            if not replaced:
                candidates.append(next_candidate)
            _write_json(job_dir / "candidates.json", {"candidates": candidates})
            result_path = job_dir / "result.json"
            if result_path.exists():
                result = _read_json(result_path)
                result["candidates"] = candidates
                _write_json(result_path, result)

    def quick_validate_candidate(
        self,
        job_id: str,
        candidate_id: str,
        verifier_options: Optional[InverseVerificationSpec] = None,
    ) -> Dict[str, Any]:
        candidate = self.get_candidate(job_id, candidate_id)
        if candidate.get("kpis") and candidate.get("quick_validation_status") == "complete":
            return {"status": "cached", "candidate": candidate}

        result = self.get_result(job_id)
        request_payload = result.get("request", {})
        verification_payload = request_payload.get("verification", {}) if isinstance(request_payload.get("verification"), Mapping) else {}
        if verifier_options is None:
            verifier_options = InverseVerificationSpec(**verification_payload)
        re_value = float(result.get("target", {}).get("Re", request_payload.get("constraints", {}).get("re", 100.0)))
        cylinders = [Cylinder(x=float(x), y=float(y)) for x, y in candidate.get("centers", [])]
        design_request = DesignRequest(
            model_id=verifier_options.forward_verifier_model_id,
            mode=verifier_options.forward_backend,
            re=re_value,
            cylinders=cylinders,
            phase_bins=verifier_options.phase_bins,
            resolution_nx=verifier_options.nx,
            resolution_ny=verifier_options.ny,
            field="omega",
            return_hypergraph=True,
            return_kpis=True,
            generative=GenerativeOptions(
                num_samples=verifier_options.generative_num_samples,
                n_steps=verifier_options.generative_n_steps,
                seed=None,
            ),
        )
        response = self.inference_service.infer(design_request)
        forward_result = _read_json(self.inference_service.result_path(response["job_id"]))
        raw = dict(candidate.get("raw") or {})
        raw.update(
            {
                "verified": True,
                "kpis": forward_result.get("kpis") or {},
                "verifier_backend": verifier_options.forward_backend,
                "quick_validation": {
                    "job_id": response["job_id"],
                    "result_url": response["result_url"],
                    "frame_urls": forward_result.get("frame_urls"),
                    "hypergraph": forward_result.get("hypergraph"),
                },
            }
        )
        web_candidate = self._web_candidate(job_id, raw, result.get("target", {}))
        web_candidate["quick_validation_status"] = "complete"
        web_candidate["quick_validation"] = raw["quick_validation"]
        self._replace_candidate(job_id, web_candidate)
        return {"status": "complete", "candidate": web_candidate, "forward_result": forward_result}

    def _simulation_status_path(self, job_id: str, candidate_id: str) -> Path:
        return self._job_dir(job_id) / "simulation_jobs" / candidate_id / "status.json"

    def _write_sim_status(self, job_id: str, candidate_id: str, status: str, **payload: Any) -> Dict[str, Any]:
        path = self._simulation_status_path(job_id, candidate_id)
        current = _read_json(path)
        current.update({"job_id": job_id, "candidate_id": candidate_id, "status": status, "updated_at": _now(), **payload})
        _write_json(path, current)
        return current

    def simulation_validate_candidate(
        self,
        job_id: str,
        candidate_id: str,
        options: CandidateSimulationValidationRequest,
    ) -> Dict[str, Any]:
        candidate = self.get_candidate(job_id, candidate_id)
        cid = str(candidate.get("id", candidate_id))
        status_path = self._simulation_status_path(job_id, cid)
        current = _read_json(status_path)
        if current.get("status") in {"queued", "writing_config", "running_simulation", "preprocessing", "computing_kpis"}:
            return current
        if candidate.get("simulation_validation_status") == "complete":
            return {"job_id": job_id, "candidate_id": cid, "status": "complete", "candidate": candidate}

        self._write_sim_status(job_id, cid, "queued", created_at=_now())
        thread_key = f"{job_id}:{cid}"
        thread = threading.Thread(
            target=self._simulation_thread,
            args=(job_id, cid, candidate, options),
            name=f"inverse-sim-{job_id}-{cid}",
            daemon=True,
        )
        self._simulation_threads[thread_key] = thread
        thread.start()
        return self.get_simulation_status(job_id, cid)

    def _simulation_thread(
        self,
        job_id: str,
        candidate_id: str,
        candidate: Dict[str, Any],
        options: CandidateSimulationValidationRequest,
    ) -> None:
        self._simulation_semaphore.acquire()
        try:
            self._run_simulation_validation(job_id, candidate_id, candidate, options)
        except Exception as exc:
            self._write_sim_status(job_id, candidate_id, "error", completed_at=_now(), error=str(exc))
            next_candidate = dict(candidate)
            next_candidate["simulation_validation_status"] = "error"
            next_candidate["simulation_error"] = str(exc)
            self._replace_candidate(job_id, next_candidate)
        finally:
            self._simulation_semaphore.release()

    def _run_simulation_validation(
        self,
        job_id: str,
        candidate_id: str,
        candidate: Dict[str, Any],
        options: CandidateSimulationValidationRequest,
    ) -> None:
        from evaluate_inverse import (
            _find_single_case_dir,
            _kpi_comparison,
            _load_processed_cycle,
            _run_logged_subprocess,
            plot_candidate_flow,
            plot_kpi_ml_vs_simulation,
            plot_ml_simulation_field_comparison,
            score_candidate_kpis,
            try_save_cycle_gif,
            write_candidate_snapshot,
            write_kpi_comparison_csv,
            write_simulation_config_for_candidate,
        )
        from inverse_kpi import compute_cycle_kpis

        job_dir = self._job_dir(job_id)
        result = self.get_result(job_id)
        raw = dict(candidate.get("raw") or candidate)
        target_payload = result.get("target", {})
        target_spec_payload = _read_json(job_dir / "evaluate_output" / "target_spec.json")
        target_spec = target_spec_payload.get("target_spec") or {
            "kpis": target_payload.get("kpis", {}),
            "constraints": {
                "num_cylinders_min": target_payload.get("num_cylinders_min"),
                "num_cylinders_max": target_payload.get("num_cylinders_max"),
                "min_center_distance": target_payload.get("min_center_distance"),
                **(target_payload.get("preferences", {}) if isinstance(target_payload.get("preferences"), Mapping) else {}),
            },
        }
        domain = result.get("domain", {})
        lx = float(domain.get("length_x", 24.0))
        ly = float(domain.get("length_y", 12.0))
        verification = result.get("verification", {})
        phase_bins = int(options.simulation_phase_bins or verification.get("phase_bins", 12))
        re_value = float(target_payload.get("Re", target_payload.get("re", 100.0)))
        cand_dir = self._candidate_dir_for(job_dir, raw) or (job_dir / "candidates" / candidate_id)
        cand_dir.mkdir(parents=True, exist_ok=True)
        raw_root = cand_dir / "simulation_raw"
        processed_root = cand_dir / "simulation_processed"
        raw_root.mkdir(parents=True, exist_ok=True)
        processed_root.mkdir(parents=True, exist_ok=True)

        args = argparse.Namespace(
            simulation_config_json=None,
            simulation_mode=options.simulation_mode,
            simulation_device=options.simulation_device,
            simulation_gpu_id=options.simulation_gpu_id,
            simulation_preprocess_device=options.simulation_preprocess_device,
            simulation_nx=options.simulation_nx,
            simulation_ny=options.simulation_ny,
            simulation_phase_bins=phase_bins,
            simulation_warmup_cycles=options.simulation_warmup_cycles,
            simulation_save_cycles=options.simulation_save_cycles,
            simulation_frames_per_cycle=options.simulation_frames_per_cycle,
            simulation_dt=options.simulation_dt,
            device=_device_arg(),
        )

        self._write_sim_status(job_id, candidate_id, "writing_config", log_path=str(cand_dir / "simulation.log"))
        config_path = write_simulation_config_for_candidate(raw, args=args, raw_root=raw_root, re_value=re_value, lx=lx, ly=ly)

        sim_runner = (
            "import json, sys; "
            "from pathlib import Path; "
            "sys.path.insert(0, 'src'); "
            "from multicyl_common import config_from_dict; "
            "from simulate_multicylinder_phiflow import run_case; "
            "cfg = config_from_dict(json.load(open(sys.argv[1], 'r', encoding='utf-8'))); "
            "print(f'Prepared configuration: case_id={cfg.save.case_id}, mode={cfg.mode}, device={cfg.execution.device}, gpu_id={cfg.execution.gpu_id}, cylinders={cfg.layout.num_cylinders}, Re={cfg.flow.re}'); "
            "case_dir = run_case(cfg); "
            "print(f'Simulation complete. Saved case to: {case_dir}')"
        )
        sim_log = cand_dir / "simulation.log"
        self._write_sim_status(job_id, candidate_id, "running_simulation", log_path=str(sim_log))
        _run_logged_subprocess(
            [sys.executable, "-c", sim_runner, str(config_path)],
            cwd=settings.demo_root,
            log_path=sim_log,
            label="simulation",
            echo_markers=("Prepared configuration", "Starting simulation", "Created case directory", "Runtime summary", "Simulation complete"),
        )
        case_dir = _find_single_case_dir(raw_root)

        preprocess_log = cand_dir / "simulation_preprocess.log"
        preprocess_device = str(options.simulation_preprocess_device or _device_arg())
        self._write_sim_status(job_id, candidate_id, "preprocessing", log_path=str(preprocess_log))
        _run_logged_subprocess(
            [
                sys.executable,
                "src/preprocess_multicyl_dataset.py",
                "--input-root",
                str(raw_root),
                "--output-root",
                str(processed_root),
                "--device",
                preprocess_device,
                "--phase-bins",
                str(phase_bins),
                "--save-cycles",
                "1",
                "--points-per-phase-bin",
                "0",
                "--sampling-mode",
                "uniform",
                "--save-full-canonical-cycles",
            ],
            cwd=settings.demo_root,
            log_path=preprocess_log,
            label="preprocess",
            echo_markers=("[INFO] Using torch device", "[INFO] Discovered", "[INFO] Loaded", "Canonical cycle method", "Finished."),
        )

        self._write_sim_status(job_id, candidate_id, "computing_kpis", log_path=str(preprocess_log))
        sim_cycle, sim_channel_order = _load_processed_cycle(processed_root, case_dir)
        sim_kpis = compute_cycle_kpis(sim_cycle, x_grid=None, y_grid=None, channel_order=sim_channel_order, domain={"lx": lx, "ly": ly})
        sim_kpis["num_cylinders"] = int(raw.get("num_cylinders", raw.get("count", 0)))
        sim_kpis["min_center_distance"] = float(raw.get("min_pair_distance", 0.0))
        sim_kpis["x_span"] = float(raw.get("x_span", 0.0))
        sim_kpis["y_span"] = float(raw.get("y_span", 0.0))
        sim_kpis["valid"] = bool(raw.get("validity", {}).get("valid", True))
        generated_kpis = raw.get("kpis", {}) if isinstance(raw.get("kpis"), Mapping) else {}
        comparison = _kpi_comparison(generated_kpis, sim_kpis)
        sim_score = score_candidate_kpis(sim_kpis, target_spec)
        generated_score = _score_or_none(raw.get("score"))
        raw["simulation_verified"] = True
        raw["simulation_verification"] = {
            "case_dir": str(case_dir),
            "processed_root": str(processed_root),
            "simulation_log": str(sim_log),
            "preprocess_log": str(preprocess_log),
            "phase_bins": phase_bins,
            "channel_order": sim_channel_order,
            "cycle_shape": list(sim_cycle.shape),
            "generated_kpis": generated_kpis,
            "ground_truth_kpis": sim_kpis,
            "kpi_comparison": comparison,
            "generated_score": generated_score,
            "ground_truth_score": float(sim_score["total_score"]),
            "ground_truth_per_kpi_errors": sim_score.get("per_kpi_errors", {}),
            "score_delta": float(sim_score["total_score"] - generated_score) if generated_score is not None else None,
        }
        np.savez_compressed(cand_dir / "simulation_canonical_cycle.npz", canonical_cycle=sim_cycle.astype(np.float32), channel_order=np.asarray(sim_channel_order))
        centers = np.asarray(raw["centers"], dtype=np.float32).reshape(-1, 2)
        plot_candidate_flow(sim_cycle, centers, cand_dir / "simulation_flow.png", channel_order=sim_channel_order, lx=lx, ly=ly)
        try_save_cycle_gif(sim_cycle, cand_dir / "simulation_cycle.gif", sim_channel_order, centers, lx=lx, ly=ly)
        plot_kpi_ml_vs_simulation(comparison, cand_dir / "simulation_kpi_comparison.png", target_payload=target_payload)
        write_kpi_comparison_csv(comparison, cand_dir / "simulation_kpi_comparison.csv")
        ml_cycle_path = cand_dir / "generated_verifier_cycle.npz"
        if ml_cycle_path.exists():
            with np.load(ml_cycle_path, allow_pickle=True) as data:
                ml_cycle = np.asarray(data["cycle_mean"], dtype=np.float32)
                ml_order = [str(v) for v in np.asarray(data["channel_order"]).reshape(-1)] if "channel_order" in data.files else ["u", "v", "p", "omega"]
            plot_ml_simulation_field_comparison(
                ml_cycle,
                sim_cycle,
                centers,
                cand_dir / "ml_vs_simulation_field.png",
                ml_channel_order=ml_order,
                sim_channel_order=sim_channel_order,
                lx=lx,
                ly=ly,
                gif_path=cand_dir / "ml_vs_simulation_cycle.gif",
            )
        from train_inverse import write_json
        from evaluate_inverse import json_safe

        write_json(cand_dir / "simulation_verification.json", json_safe(raw["simulation_verification"]))
        write_candidate_snapshot(raw, cand_dir)
        web_candidate = self._web_candidate(job_id, raw, target_payload)
        web_candidate["simulation_validation_status"] = "complete"
        self._replace_candidate(job_id, web_candidate)
        self._write_sim_status(
            job_id,
            candidate_id,
            "complete",
            completed_at=_now(),
            candidate=web_candidate,
            result_url=f"/api/inverse/jobs/{job_id}/candidates/{candidate_id}",
        )

    def get_simulation_status(self, job_id: str, candidate_id: str) -> Dict[str, Any]:
        candidate = self.get_candidate(job_id, candidate_id)
        cid = str(candidate.get("id", candidate_id))
        status = _read_json(self._simulation_status_path(job_id, cid))
        if not status:
            status = {
                "job_id": job_id,
                "candidate_id": cid,
                "status": candidate.get("simulation_validation_status", "not_started"),
            }
        log_path = Path(str(status.get("log_path", ""))).expanduser() if status.get("log_path") else None
        status["log_tail"] = _tail(log_path) if log_path else []
        return status

    def safe_file_path(self, job_id: str, relative_path: str) -> Path:
        job_dir = self._job_dir(job_id).resolve()
        candidate = (job_dir / relative_path).resolve()
        if job_dir != candidate and job_dir not in candidate.parents:
            raise ValueError("Path traversal is not allowed.")
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError("File not found.")
        return candidate
