from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from settings import settings


def _nested_get(data: Dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parsed_values = [_coerce_positive_int(item) for item in value]
        positive_values = [item for item in parsed_values if item is not None]
        return max(positive_values) if positive_values else None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_phase_bin_config(config: Dict[str, Any], manifest_entry: Dict[str, Any] | None = None) -> Dict[str, Any]:
    manifest_entry = manifest_entry or {}

    max_candidates = [
        ("manifest.max_phase_bins", manifest_entry.get("max_phase_bins")),
        ("config.web_demo.max_phase_bins", _nested_get(config, "web_demo.max_phase_bins")),
        ("config.validation.phase_bins_to_eval", _nested_get(config, "validation.phase_bins_to_eval")),
        ("config.evaluation.cycle.phase_bins", _nested_get(config, "evaluation.cycle.phase_bins")),
    ]
    max_phase_bins = 36
    source = "fallback"
    for candidate_source, candidate_value in max_candidates:
        parsed = _coerce_positive_int(candidate_value)
        if parsed is not None:
            max_phase_bins = parsed
            source = candidate_source
            break

    default_candidates = [
        manifest_entry.get("default_phase_bins"),
        _nested_get(config, "web_demo.default_phase_bins"),
        max_phase_bins,
        36,
    ]
    default_phase_bins = 36
    for candidate_value in default_candidates:
        parsed = _coerce_positive_int(candidate_value)
        if parsed is not None:
            default_phase_bins = min(parsed, max_phase_bins)
            break

    policy = str(manifest_entry.get("phase_bin_policy", "cap")).lower()
    if policy not in {"cap", "reject"}:
        policy = "cap"

    return {
        "default_phase_bins": default_phase_bins,
        "max_phase_bins": max_phase_bins,
        "phase_bin_policy": policy,
        "phase_bin_source": source,
    }


def _resolve_path(path_like: str | Path, *, base: Optional[Path] = None) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


@dataclass
class ModelEntry:
    raw: Dict[str, Any]
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    missing_files: List[str]
    runtime_error: Optional[str] = None
    loaded_artifact: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    @property
    def mode(self) -> str:
        return str(self.raw.get("mode", "deterministic")).lower()

    @property
    def enabled(self) -> bool:
        return bool(self.raw.get("enabled", True))

    @property
    def preload(self) -> bool:
        return bool(self.raw.get("preload", False))

    @property
    def stage(self) -> Optional[int]:
        stage = self.raw.get("stage")
        return int(stage) if stage is not None else None

    @property
    def is_stage2_generative(self) -> bool:
        return self.mode == "generative" and self.stage == 2 and self.enabled

    def load_config_json(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def availability_status(self) -> str:
        if self.runtime_error:
            return "error"
        if self.missing_files:
            return "missing_files"
        if self.mode == "generative" and not self.is_stage2_generative:
            return "stage2_pending"
        if not self.enabled:
            return "disabled"
        return "available"

    def reason_unavailable(self) -> Optional[str]:
        if self.runtime_error:
            return self.runtime_error
        if not self.run_dir.exists():
            return f"Run directory not found: {self.run_dir}"
        if not self.checkpoint_path.exists():
            return f"Checkpoint not found: {self.checkpoint_path}"
        if not self.config_path.exists():
            return f"Config not found: {self.config_path}"
        if self.mode == "generative" and self.stage != 2:
            return "Generative stage-2 checkpoint pending."
        if not self.enabled:
            return "Model entry is disabled in the manifest."
        return None

    def to_public_dict(self) -> Dict[str, Any]:
        status = self.availability_status()
        metadata: Dict[str, Any] = {}
        try:
            cfg = self.load_config_json() if not self.missing_files else {}
            model_cfg = cfg.get("model", {})
            if not model_cfg and "generation" in cfg:
                metadata["generation"] = cfg.get("generation", {})
            metadata.update(
                {
                    "max_num_cylinders": model_cfg.get("max_num_cylinders"),
                    "domain_length_x": model_cfg.get("domain_length_x"),
                    "domain_length_y": model_cfg.get("domain_length_y"),
                    "re_scale": model_cfg.get("re_scale"),
                    "default_resolution_nx": self.raw.get("default_resolution_nx"),
                    "default_resolution_ny": self.raw.get("default_resolution_ny"),
                    **resolve_phase_bin_config(cfg, self.raw),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive metadata path
            metadata["config_error"] = str(exc)

        return {
            "id": self.id,
            "label": self.raw.get("label", self.id),
            "mode": self.mode,
            "enabled": self.enabled,
            "available": status == "available",
            "preload": self.preload,
            "stage": self.stage,
            "run_dir": str(self.run_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "config_path": str(self.config_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "config_exists": self.config_path.exists(),
            "missing_files": list(self.missing_files),
            "reason_unavailable": self.reason_unavailable(),
            "note": self.raw.get("note"),
            "status": status,
            "error": self.runtime_error,
            "metadata": metadata,
        }


class ModelRegistry:
    def __init__(self, manifest_path: Path = settings.manifest_path):
        self.manifest_path = manifest_path
        self._entries: Dict[str, ModelEntry] = {}
        self.reload()

    def reload(self) -> None:
        previous_entries = self._entries
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Model manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        entries: Dict[str, ModelEntry] = {}
        for raw in manifest.get("models", []):
            model_id = str(raw.get("id", "")).strip()
            if not model_id:
                continue
            run_dir = _resolve_path(raw["run_dir"], base=settings.demo_root)
            checkpoint_name = raw.get("checkpoint_name", "best_model.pt")
            config_name = raw.get("config_name", "resolved_train_config.json")
            checkpoint_path = _resolve_path(checkpoint_name, base=run_dir)
            config_path = _resolve_path(config_name, base=run_dir)
            missing = []
            if not run_dir.exists():
                missing.append(str(run_dir))
            if not checkpoint_path.exists():
                missing.append(str(checkpoint_path))
            if not config_path.exists():
                missing.append(str(config_path))
            previous = previous_entries.get(model_id)
            entries[model_id] = ModelEntry(
                raw=dict(raw),
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                missing_files=missing,
                runtime_error=previous.runtime_error if previous else None,
                loaded_artifact=previous.loaded_artifact if previous else None,
            )
        self._entries = entries

    def list_entries(self) -> List[ModelEntry]:
        return list(self._entries.values())

    def list_public(self) -> List[Dict[str, Any]]:
        return [entry.to_public_dict() for entry in self.list_entries()]

    def get_entry(self, model_id: str) -> ModelEntry:
        try:
            return self._entries[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model_id: {model_id}") from exc

    def get_config(self, model_id: str) -> Dict[str, Any]:
        return self.get_entry(model_id).load_config_json()

    def set_runtime_error(self, model_id: str, error: Optional[str]) -> None:
        self.get_entry(model_id).runtime_error = error


registry = ModelRegistry()
