from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from settings import settings


def _resolve_path(path_like: str | Path, *, base: Optional[Path] = None) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _nested_get(data: Dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


@dataclass
class ModelEntry:
    raw: Dict[str, Any]
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    missing_files: List[str]
    loaded_artifact: Any = None
    runtime_error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    @property
    def enabled(self) -> bool:
        return bool(self.raw.get("enabled", True))

    @property
    def preload(self) -> bool:
        return bool(self.raw.get("preload", False))

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
        if not self.enabled:
            return "Model entry is disabled in the manifest."
        return None

    def to_public_dict(self) -> Dict[str, Any]:
        status = self.availability_status()
        metadata: Dict[str, Any] = {}
        try:
            cfg = self.load_config_json() if not self.missing_files else {}
            model_cfg = cfg.get("model", {})
            dataset_cfg = cfg.get("dataset", {})
            metadata.update(
                {
                    "max_num_modules": model_cfg.get("max_num_modules"),
                    "domain_length_x": model_cfg.get("domain_length_x"),
                    "domain_length_y": model_cfg.get("domain_length_y"),
                    "module_radius": model_cfg.get("module_radius"),
                    "field_names": model_cfg.get("field_names", ["u", "v", "p", "omega", "temperature"]),
                    "dataset": dataset_cfg.get("packed_h5_path"),
                    "default_reference_split": self.raw.get("default_reference_split", "test"),
                    "default_reference_case_index": self.raw.get("default_reference_case_index", 0),
                    "default_heat_power": self.raw.get("default_heat_power", 1.0),
                    "heat_power_min": self.raw.get("heat_power_min", 0.0),
                    "heat_power_max": self.raw.get("heat_power_max", 3.0),
                    "local_surrogate": _nested_get(cfg, "model.local_surrogate_checkpoint_path"),
                }
            )
        except Exception as exc:
            metadata["config_error"] = str(exc)
        return {
            "id": self.id,
            "label": self.raw.get("label", self.id),
            "enabled": self.enabled,
            "available": status == "available",
            "run_dir": str(self.run_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "config_path": str(self.config_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "config_exists": self.config_path.exists(),
            "missing_files": list(self.missing_files),
            "reason_unavailable": self.reason_unavailable(),
            "status": status,
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
            checkpoint_name = raw.get("checkpoint_name", "latest_model.pt")
            config_name = raw.get("config_name", "resolved_train_config.json")
            checkpoint_path = _resolve_path(checkpoint_name, base=run_dir)
            config_path = _resolve_path(config_name, base=run_dir)
            missing = []
            for path in (run_dir, checkpoint_path, config_path):
                if not path.exists():
                    missing.append(str(path))
            previous = previous_entries.get(model_id)
            entries[model_id] = ModelEntry(
                raw=dict(raw),
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                missing_files=missing,
                loaded_artifact=previous.loaded_artifact if previous else None,
                runtime_error=previous.runtime_error if previous else None,
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

    def set_runtime_error(self, model_id: str, error: Optional[str]) -> None:
        self.get_entry(model_id).runtime_error = error


registry = ModelRegistry()
