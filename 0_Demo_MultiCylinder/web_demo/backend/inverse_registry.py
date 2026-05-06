from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from settings import settings


def _resolve_path(path_like: str | Path, *, base: Optional[Path] = None) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


@dataclass
class InverseModelEntry:
    raw: Dict[str, Any]
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    missing_files: List[str]

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    @property
    def enabled(self) -> bool:
        return bool(self.raw.get("enabled", True))

    @property
    def preload(self) -> bool:
        return bool(self.raw.get("preload", False))

    @property
    def checkpoint_name(self) -> str:
        return str(self.raw.get("checkpoint_name", self.checkpoint_path.name))

    @property
    def config_name(self) -> str:
        return str(self.raw.get("config_name", self.config_path.name))

    @property
    def default_forward_verifier_id(self) -> Optional[str]:
        value = self.raw.get("default_forward_verifier_id")
        return str(value) if value else None

    def load_config_json(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def availability_status(self) -> str:
        if self.missing_files:
            return "missing_files"
        if not self.enabled:
            return "disabled"
        return "available"

    def reason_unavailable(self) -> Optional[str]:
        if not self.run_dir.exists():
            return f"Run directory not found: {self.run_dir}"
        if not self.checkpoint_path.exists():
            return f"Checkpoint not found: {self.checkpoint_path}"
        if not self.config_path.exists():
            return f"Config not found: {self.config_path}"
        if not self.enabled:
            return "Inverse model entry is disabled in the manifest."
        return None

    def to_public_dict(self) -> Dict[str, Any]:
        status = self.availability_status()
        metadata: Dict[str, Any] = {}
        try:
            cfg = self.load_config_json() if not self.missing_files else {}
            inverse_cfg = cfg.get("inverse_model", {}) if isinstance(cfg.get("inverse_model"), dict) else {}
            target_cfg = cfg.get("target_kpis", {}) if isinstance(cfg.get("target_kpis"), dict) else {}
            metadata = {
                "max_num_cylinders": inverse_cfg.get("max_num_cylinders"),
                "domain_length_x": inverse_cfg.get("domain_length_x"),
                "domain_length_y": inverse_cfg.get("domain_length_y"),
                "min_center_distance": inverse_cfg.get("min_center_distance"),
                "re_scale": inverse_cfg.get("re_scale"),
                "kpi_names": target_cfg.get("names", []),
                "forward_verifier": cfg.get("forward_verifier", {}),
            }
        except Exception as exc:  # pragma: no cover - defensive metadata path
            metadata["config_error"] = str(exc)

        return {
            "id": self.id,
            "label": self.raw.get("label", self.id),
            "enabled": self.enabled,
            "available": status == "available",
            "preload": self.preload,
            "run_dir": str(self.run_dir),
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_path": str(self.checkpoint_path),
            "config_name": self.config_name,
            "config_path": str(self.config_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "config_exists": self.config_path.exists(),
            "missing_files": list(self.missing_files),
            "reason_unavailable": self.reason_unavailable(),
            "status": status,
            "default_forward_verifier_id": self.default_forward_verifier_id,
            "metadata": metadata,
        }


class InverseModelRegistry:
    def __init__(self, manifest_path: Path = settings.inverse_manifest_path):
        self.manifest_path = manifest_path
        self._entries: Dict[str, InverseModelEntry] = {}
        self.reload()

    def reload(self) -> None:
        if not self.manifest_path.exists():
            self._entries = {}
            return
        with self.manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        entries: Dict[str, InverseModelEntry] = {}
        for raw in manifest.get("inverse_models", []):
            model_id = str(raw.get("id", "")).strip()
            if not model_id:
                continue
            run_dir = _resolve_path(raw["run_dir"], base=settings.demo_root)
            checkpoint_name = raw.get("checkpoint_name", "best_model.pt")
            config_name = raw.get("config_name", "resolved_train_inverse_config.json")
            checkpoint_path = _resolve_path(checkpoint_name, base=run_dir)
            config_path = _resolve_path(config_name, base=run_dir)
            missing: List[str] = []
            if not run_dir.exists():
                missing.append(str(run_dir))
            if not checkpoint_path.exists():
                missing.append(str(checkpoint_path))
            if not config_path.exists():
                missing.append(str(config_path))
            entries[model_id] = InverseModelEntry(
                raw=dict(raw),
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                missing_files=missing,
            )
        self._entries = entries

    def list_entries(self) -> List[InverseModelEntry]:
        return list(self._entries.values())

    def list_public(self) -> List[Dict[str, Any]]:
        return [entry.to_public_dict() for entry in self.list_entries()]

    def get_entry(self, model_id: str) -> InverseModelEntry:
        try:
            return self._entries[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown inverse_model_id: {model_id}") from exc


inverse_registry = InverseModelRegistry()
