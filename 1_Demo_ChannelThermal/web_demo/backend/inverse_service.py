from __future__ import annotations

from collections import deque
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import quote

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from inverse_registry import InverseModelRegistry
from model_registry import ModelRegistry
from schemas import InverseRunRequest
from settings import settings
from thermal_design_intent import is_design_intent_payload


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


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _parse_number(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(parsed):
        return None
    return int(parsed) if abs(parsed - int(parsed)) < 1.0e-12 else parsed


def _csv_index(rows: List[Dict[str, Any]], key: str = "sample_index") -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        parsed = _parse_number(row.get(key))
        if isinstance(parsed, int):
            out[parsed] = {_key: _parse_number(value) for _key, value in row.items()}
    return out


def load_design_prior_eval_run(run_dir: str | Path) -> Dict[str, Any]:
    """Load a completed design-prior evaluation folder for optional web display.

    This helper is intentionally read-only and is not wired into existing
    inverse endpoints. It summarizes outputs produced by
    ``src/evaluate_design_prior.py`` so a later API route or UI panel can expose
    them without changing the current inverse demo behavior.
    """

    root = Path(run_dir).expanduser()
    summary = _read_json(root / "summary.json")
    top_rows = [{key: _parse_number(value) for key, value in row.items()} for row in _read_csv_rows(root / "candidates_top.csv")]
    all_rows = _read_csv_rows(root / "candidates_all.csv")
    score_rows = [{key: _parse_number(value) for key, value in row.items()} for row in _read_csv_rows(root / "score_vs_forward_calls.csv")]
    methods = summary.get("methods", {}) if isinstance(summary.get("methods"), Mapping) else {}
    artifacts: Dict[str, Any] = {"comparison_plots": {}, "top_candidates": []}
    for name in ("method_comparison.png", "score_vs_calls.png"):
        path = root / name
        if path.exists():
            artifacts["comparison_plots"][path.stem] = str(path)
    top_dir = root / "top_candidates"
    if top_dir.exists():
        grouped: Dict[str, Dict[str, str]] = {}
        for path in sorted(top_dir.glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".png", ".json", ".csv"}:
                continue
            stem = path.stem
            parts = stem.split("_")
            prefix = "_".join(parts[:3]) if len(parts) >= 3 and parts[-2].isdigit() else stem
            grouped.setdefault(prefix, {})[path.name] = str(path)
        artifacts["top_candidates"] = [{"prefix": prefix, "files": files} for prefix, files in sorted(grouped.items())]
    return _json_safe(
        {
            "run_dir": str(root),
            "exists": root.exists(),
            "summary": summary,
            "methods": methods,
            "top_candidates": top_rows,
            "candidate_count_top": len(top_rows),
            "candidate_count_all": len(all_rows),
            "score_vs_forward_calls": score_rows,
            "artifacts": artifacts,
        }
    )


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
        sources = (
            ("inverse_targets_v2", settings.inverse_target_v2_presets_dir),
            ("inverse_targets", settings.inverse_target_presets_dir),
        )
        for source_dir, directory in sources:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.json")):
                payload = _read_json(path)
                name = str(payload.get("name", path.stem))
                mode = "design_intent" if is_design_intent_payload(payload) else "legacy_kpi"
                presets.append(
                    {
                        "name": name,
                        "label": name.replace("_", " "),
                        "path": str(path),
                        "target": payload,
                        "target_mode": mode,
                        "source_dir": source_dir,
                    }
                )
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
        if request.target_mode == "legacy_kpi" and not enabled:
            raise ValueError("At least one KPI target row must be enabled.")
        if request.target_mode == "design_intent":
            constraints = _model_dump(request.constraints)
            structure = _model_dump(request.structure_constraints)
            thermal_limits = _model_dump(request.thermal_limits)
            objective_weights = _model_dump(request.objective_weights)
            heat_loads = _model_dump(request.heat_loads)
            field_preferences = dict(request.field_preferences or {})
            if structure.get("min_x_coverage") is not None:
                structure["x_coverage_min"] = structure["min_x_coverage"]
                field_preferences.setdefault("min_x_coverage", structure["min_x_coverage"])
            if structure.get("min_y_coverage") is not None:
                structure["y_coverage_min"] = structure["min_y_coverage"]
                field_preferences.setdefault("min_y_coverage", structure["min_y_coverage"])
            if structure.get("min_mean_pair_distance") is not None:
                structure["mean_pair_distance_min"] = structure["min_mean_pair_distance"]
                field_preferences.setdefault("min_mean_pair_distance", structure["min_mean_pair_distance"])
            geometry = {
                "min_center_distance": float(constraints.get("min_center_distance", 1.1)),
                "wall_clearance": float(constraints.get("wall_clearance", 0.08)),
                "inlet_clearance": float(constraints.get("inlet_clearance", 0.30)),
                "outlet_clearance": float(constraints.get("outlet_clearance", 0.30)),
                "x_span": structure.get("x_span"),
                "y_span": structure.get("y_span"),
                "keepout_boxes": structure.get("keepout_boxes", []),
                "protected_boxes": structure.get("protected_boxes", []),
                "preferred_boxes": structure.get("preferred_boxes", []),
            }
            sketch_maps = structure.get("sketch_maps")
            if isinstance(sketch_maps, Mapping):
                geometry["sketch_maps"] = dict(sketch_maps)
                for source, dest in (
                    ("preferred", "preferred_region_map"),
                    ("keepout", "keepout_map"),
                    ("protected", "protected_region_map"),
                    ("reference_soft", "reference_layout_soft_map"),
                ):
                    if source in sketch_maps:
                        field_preferences.setdefault(dest, sketch_maps[source])
            payload = {
                "name": request.target_name or "web_demo_design_intent",
                "scenario": {
                    "num_modules_min": int(constraints.get("num_modules_min", 1)),
                    "num_modules_max": int(constraints.get("num_modules_max", constraints.get("num_modules_min", 1))),
                    "heat_load_policy": "preserve_total_heat",
                },
                "geometry_constraints": {key: value for key, value in geometry.items() if value is not None},
                "thermal_limits": {key: value for key, value in thermal_limits.items() if value is not None},
                "objective_weights": objective_weights,
                "field_preferences": field_preferences,
                "structure_constraints": structure,
                "heat_loads": heat_loads,
            }
            if constraints.get("heat_power_total") is not None and payload["heat_loads"].get("total") is None:
                payload["heat_loads"]["total"] = float(constraints["heat_power_total"])
            if enabled:
                payload["kpis"] = {item.name: _format_kpi_target(_model_dump(item)) for item in enabled}
            return payload
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
        checkpoint_name = "best_verified_model.pt" if (inv.run_dir / "best_verified_model.pt").exists() else inv.checkpoint_path.name
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
            checkpoint_name,
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
        if request.guidance_scale is not None:
            command.extend(["--guidance-scale", str(float(request.guidance_scale))])
        if request.diversity_rerank_weight is not None:
            command.extend(["--diversity-rerank-weight", str(float(request.diversity_rerank_weight))])
        if request.candidate_pool_multiplier is not None:
            command.extend(["--candidate-pool-multiplier", str(float(request.candidate_pool_multiplier))])

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
            self._write_status(job_id, result["status"], result_url=f"/api/inverse/jobs/{job_id}/result", returncode=int(code))
        except Exception as exc:
            self._write_status(job_id, "failed", error=str(exc), stdout_tail=_tail(stdout_path), stderr_tail=_tail(stderr_path))

    def _assemble_result(self, job_id: str, job_dir: Path, out_dir: Path, *, returncode: int) -> Dict[str, Any]:
        data_dir = out_dir / "data"
        candidates_path = _first_existing((out_dir / "candidates.json", data_dir / "candidates.json"))
        candidates_payload = _read_json(candidates_path, {"candidates": [], "target": {}}) if candidates_path else {"candidates": [], "target": {}}
        summary_path = _first_existing((out_dir / "verification_summary.json", data_dir / "verification_summary.json"))
        summary = _read_json(summary_path) if summary_path else {}
        target = candidates_payload.get("target", {})
        candidates = self._normalize_candidates(job_id, job_dir, out_dir, candidates_payload.get("candidates", []), target)
        if not candidates:
            candidates = self._fallback_candidates(job_id, job_dir, out_dir, target)
        artifacts = {
            path.relative_to(out_dir).with_suffix("").as_posix().replace("/", "_"): _file_url(job_id, job_dir, path)
            for path in sorted(out_dir.rglob("*.png"))
        }
        for path in sorted(out_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".png"}:
                continue
            if any(token in path.name for token in ("hypergraph_planned", "hypergraph_realized", "hypergraph_mismatch", "hypergraph_overlay", "hypergraph_edge_table")):
                artifacts[path.relative_to(out_dir).with_suffix("").as_posix().replace("/", "_")] = _file_url(job_id, job_dir, path)
        for rel in (
            "candidates.csv",
            "kpi_scores.csv",
            "top_candidates.npz",
            "target_spec_resolved.json",
            "verification_summary.json",
            "data/candidates.csv",
            "data/kpi_scores.csv",
            "data/top_candidates.npz",
            "data/target_spec_resolved.json",
            "data/verification_summary.json",
            "data/all_kpis_verified.csv",
            "data/design_intent_score_breakdown.csv",
        ):
            path = out_dir / rel
            if path.exists():
                artifacts[path.relative_to(out_dir).with_suffix("").as_posix().replace("/", "_")] = _file_url(job_id, job_dir, path)
        status = "complete" if candidates else "complete_with_no_candidates"
        result = {
            "job_id": job_id,
            "status": status,
            "returncode": int(returncode),
            "summary": summary,
            "target": target,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "artifacts": {key: value for key, value in artifacts.items() if value},
            "stdout_tail": _tail(job_dir / "stdout.log", 40),
            "stderr_tail": _tail(job_dir / "stderr.log", 40),
        }
        if not candidates:
            result["debug_files"] = self.debug_files(job_id)
        _write_json(job_dir / "candidates.json", {"target": target, "candidates": candidates})
        return result

    def _target_heat_values(self, target: Mapping[str, Any], count: int) -> Optional[List[float]]:
        count = max(int(count), 0)
        if count <= 0:
            return []
        heat_loads = target.get("heat_loads")
        if not isinstance(heat_loads, Mapping):
            payload = target.get("target_payload")
            heat_loads = payload.get("heat_loads") if isinstance(payload, Mapping) else None
        if not isinstance(heat_loads, Mapping):
            return None
        mode = str(heat_loads.get("mode", "")).lower().strip()
        values = heat_loads.get("values")
        ranges = heat_loads.get("ranges")
        if mode == "per_module" and values is not None:
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            if arr.size:
                return np.resize(arr, max(int(count), 1))[:count].tolist()
        if mode == "per_module_range" and ranges is not None:
            arr = np.asarray(ranges, dtype=np.float32).reshape(-1, 2)
            if arr.size:
                midpoint = np.mean(arr, axis=1)
                return np.resize(midpoint, max(int(count), 1))[:count].tolist()
        if mode == "uniform" and heat_loads.get("value") is not None:
            return [float(heat_loads["value"])] * int(count)
        if mode == "uniform_range" and heat_loads.get("range") is not None:
            arr = np.asarray(heat_loads.get("range"), dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                return [float(np.mean(arr[:2]))] * int(count)
        total = heat_loads.get("total")
        if total is None:
            constraints = target.get("constraints")
            total = constraints.get("heat_power_total") if isinstance(constraints, Mapping) else None
        if mode == "total_only" and total is not None:
            return [float(total) / max(float(count), 1.0)] * int(count)
        return None

    def _normalize_candidates(self, job_id: str, job_dir: Path, out_dir: Path, raw_candidates: Any, target: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(raw_candidates, list):
            return []
        normalized = []
        preview_url = _file_url(job_id, job_dir, _first_existing((out_dir / "layouts" / "candidate_layouts_ranked.png", out_dir / "candidate_layouts_ranked.png")))
        for fallback_rank, raw in enumerate(raw_candidates):
            if not isinstance(raw, Mapping):
                continue
            centers = np.asarray(raw.get("centers", []), dtype=np.float32).reshape(-1, 2)
            count = int(raw.get("count", centers.shape[0]) or centers.shape[0])
            heat = raw.get("heat_powers")
            heat_values = np.asarray(heat, dtype=np.float32).reshape(-1)[:count].tolist() if heat is not None else None
            target_heat = self._target_heat_values(target, count)
            if heat_values is not None and not any(abs(float(value)) > 1.0e-8 for value in heat_values):
                heat_values = None
            artifacts = dict(raw.get("artifacts", {}) or {})
            rank = int(raw.get("rank", fallback_rank))
            prefix = f"candidate_{rank:03d}_hypergraph"
            for key, rel in {
                "hypergraph_planned": f"data/{prefix}_planned.json",
                "hypergraph_realized": f"data/{prefix}_realized.json",
                "hypergraph_mismatch": f"data/{prefix}_mismatch.json",
                "hypergraph_edge_table": f"data/{prefix}_edge_table.csv",
                "hypergraph_overlay": f"plots/diagnostics/{prefix}_overlay.png",
                "hypergraph_mismatch_heatmap": f"plots/diagnostics/{prefix}_mismatch_heatmap.png",
            }.items():
                url = _file_url(job_id, job_dir, out_dir / rel)
                if url:
                    artifacts[key] = url
            if preview_url:
                artifacts.setdefault("preview", preview_url)
            normalized.append(
                {
                    "rank": rank,
                    "sample_index": int(raw.get("sample_index", fallback_rank)),
                    "count": count,
                    "centers": centers[:count].tolist(),
                    "heat_powers": target_heat or heat_values,
                    "heat_power_source": "target_heat_loads" if target_heat else "candidate_or_verified",
                    "valid": bool(raw.get("valid", raw.get("validity", {}).get("valid", False) if isinstance(raw.get("validity"), Mapping) else False)),
                    "total_score": _parse_number(raw.get("total_score")) or 0.0,
                    "design_intent_score": _parse_number(raw.get("design_intent_score")),
                    "kpi_score": _parse_number(raw.get("kpi_score", raw.get("legacy_total_score"))),
                    "hypergraph_consistency_score": _parse_number(raw.get("hypergraph_consistency_score")),
                    "hypergraph_diagnostics_available": bool(raw.get("hypergraph_diagnostics_available", False)),
                    "hypergraph_active_count_error": _parse_number(raw.get("hypergraph_active_count_error")),
                    "hypergraph_source_rmse": _parse_number(raw.get("hypergraph_source_rmse")),
                    "hypergraph_thermal_region_rmse": _parse_number(raw.get("hypergraph_thermal_region_rmse")),
                    "hypergraph_A_mh_l1": _parse_number(raw.get("hypergraph_A_mh_l1")),
                    "verified_kpis": dict(raw.get("verified_kpis", {}) or {}),
                    "score_detail": dict(raw.get("score_detail", {}) or {}),
                    "design_intent_score_detail": dict(raw.get("design_intent_score_detail", {}) or {}),
                    "structure_score_detail": dict(raw.get("structure_score_detail", {}) or {}),
                    "validity": dict(raw.get("validity", {}) or {}),
                    "artifacts": artifacts,
                }
            )
        normalized.sort(key=lambda row: int(row.get("rank", 0)))
        return normalized

    def _fallback_candidates(self, job_id: str, job_dir: Path, out_dir: Path, target: Mapping[str, Any]) -> List[Dict[str, Any]]:
        data_dir = out_dir / "data"
        csv_rows = _read_csv_rows(_first_existing((out_dir / "candidates.csv", data_dir / "candidates.csv")) or Path("__missing__"))
        score_rows = _csv_index(_read_csv_rows(_first_existing((out_dir / "kpi_scores.csv", data_dir / "kpi_scores.csv")) or Path("__missing__")))
        rows_by_sample = _csv_index(csv_rows)
        candidates: List[Dict[str, Any]] = []
        npz_path = _first_existing((out_dir / "top_candidates.npz", data_dir / "top_candidates.npz"))
        if npz_path and npz_path.exists():
            data = np.load(npz_path)
            centers_arr = np.asarray(data.get("centers", np.zeros((0, 0, 2))), dtype=np.float32)
            masks_arr = np.asarray(data.get("masks", np.zeros(centers_arr.shape[:2])), dtype=np.float32)
            scores_arr = np.asarray(data.get("scores", np.zeros((centers_arr.shape[0],))), dtype=np.float32)
            for idx in range(centers_arr.shape[0]):
                mask = masks_arr[idx] > 0.5
                count = int(np.sum(mask))
                row = rows_by_sample.get(idx, {})
                score_row = score_rows.get(idx, {})
                candidates.append(
                    {
                        "rank": int(row.get("rank", idx) or idx),
                        "sample_index": int(row.get("sample_index", idx) or idx),
                        "count": int(row.get("count", count) or count),
                        "centers": centers_arr[idx, mask].tolist(),
                        "heat_powers": self._target_heat_values(target, int(row.get("count", count) or count)),
                        "heat_power_source": "target_heat_loads",
                        "valid": bool(row.get("valid", False) in (True, "True", "true", 1, "1")),
                        "total_score": float(row.get("total_score", scores_arr[idx] if idx < scores_arr.shape[0] else 0.0) or 0.0),
                        "design_intent_score": row.get("design_intent_score"),
                        "kpi_score": row.get("kpi_score"),
                        "hypergraph_consistency_score": _parse_number(row.get("hypergraph_consistency_score")),
                        "hypergraph_diagnostics_available": row.get("hypergraph_consistency_score") not in (None, ""),
                        "hypergraph_active_count_error": _parse_number(row.get("hypergraph_active_count_error")),
                        "hypergraph_source_rmse": _parse_number(row.get("hypergraph_source_rmse")),
                        "hypergraph_thermal_region_rmse": _parse_number(row.get("hypergraph_thermal_region_rmse")),
                        "hypergraph_A_mh_l1": _parse_number(row.get("hypergraph_A_mh_l1")),
                        "verified_kpis": {key: value for key, value in score_row.items() if key not in {"rank", "sample_index", "total_score"}},
                        "score_detail": {},
                        "artifacts": {},
                    }
                )
        elif rows_by_sample:
            for idx, row in rows_by_sample.items():
                score_row = score_rows.get(idx, {})
                candidates.append(
                    {
                        "rank": int(row.get("rank", idx) or idx),
                        "sample_index": idx,
                        "count": int(row.get("count", 0) or 0),
                        "centers": [],
                        "heat_powers": self._target_heat_values(target, int(row.get("count", 0) or 0)),
                        "heat_power_source": "target_heat_loads",
                        "valid": bool(row.get("valid", False) in (True, "True", "true", 1, "1")),
                        "total_score": float(row.get("total_score", 0.0) or 0.0),
                        "design_intent_score": row.get("design_intent_score"),
                        "kpi_score": row.get("kpi_score"),
                        "hypergraph_consistency_score": _parse_number(row.get("hypergraph_consistency_score")),
                        "hypergraph_diagnostics_available": row.get("hypergraph_consistency_score") not in (None, ""),
                        "hypergraph_active_count_error": _parse_number(row.get("hypergraph_active_count_error")),
                        "hypergraph_source_rmse": _parse_number(row.get("hypergraph_source_rmse")),
                        "hypergraph_thermal_region_rmse": _parse_number(row.get("hypergraph_thermal_region_rmse")),
                        "hypergraph_A_mh_l1": _parse_number(row.get("hypergraph_A_mh_l1")),
                        "verified_kpis": {key: value for key, value in score_row.items() if key not in {"rank", "sample_index", "total_score"}},
                        "score_detail": {},
                        "artifacts": {},
                    }
                )
        candidates.sort(key=lambda row: int(row.get("rank", 0)))
        return candidates

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

    def debug_files(self, job_id: str) -> List[Dict[str, Any]]:
        job_dir = self._job_dir(job_id)
        out_dir = job_dir / "evaluate_output"
        if not out_dir.exists():
            raise KeyError(f"Unknown inverse job_id: {job_id}")
        files = []
        for path in sorted(item for item in out_dir.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": path.relative_to(out_dir).as_posix(),
                    "size": int(path.stat().st_size),
                    "url": _file_url(job_id, job_dir, path),
                }
            )
        return files
