from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from model_registry import ModelEntry
from settings import settings


@dataclass
class DeterministicArtifact:
    model: Any
    model_config: Dict[str, Any]
    resolved_config: Dict[str, Any]
    checkpoint_path: Path
    device: Any


def _select_device():
    import torch

    requested = settings.device.lower().strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_model_config(checkpoint: Any, resolved_config: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_config", "model_cfg"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        nested_config = checkpoint.get("config")
        if isinstance(nested_config, dict) and isinstance(nested_config.get("model"), dict):
            return nested_config["model"]
    model_cfg = resolved_config.get("model")
    if isinstance(model_cfg, dict):
        return model_cfg
    raise ValueError("Could not find model configuration in checkpoint or resolved config['model'].")


def _extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload is not a dictionary.")
    for key in ("model_state_dict", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    raise ValueError(
        "Checkpoint is missing a model state dict. Expected one of: "
        "'model_state_dict', 'model', or 'state_dict'."
    )


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(key).startswith("module.") for key in state_dict):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict


class DeterministicService:
    def load(self, entry: ModelEntry) -> DeterministicArtifact:
        if entry.mode != "deterministic":
            raise ValueError(f"Model {entry.id!r} is not deterministic.")
        if entry.missing_files:
            raise FileNotFoundError("Model entry has missing files: " + ", ".join(entry.missing_files))

        with entry.lock:
            if entry.loaded_artifact is not None:
                return entry.loaded_artifact

            import torch
            from model import build_model_from_config

            device = _select_device()
            resolved_config = _load_json(entry.config_path)
            checkpoint = torch.load(entry.checkpoint_path, map_location=device)
            model_config = _extract_model_config(checkpoint, resolved_config)
            model = build_model_from_config(model_config)
            state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                raise RuntimeError(f"Failed to load model_state_dict from {entry.checkpoint_path}: {exc}") from exc
            model.to(device)
            model.eval()
            entry.loaded_artifact = DeterministicArtifact(
                model=model,
                model_config=model_config,
                resolved_config=resolved_config,
                checkpoint_path=entry.checkpoint_path,
                device=device,
            )
            return entry.loaded_artifact

    def preload(self, entry: ModelEntry) -> None:
        self.load(entry)
