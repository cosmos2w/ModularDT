"""CHANNELTHERMAL-SPECIFIC checkpoint helpers.

Inputs are NewHONF model state, configuration dictionaries, dataset metadata,
and destination paths. Outputs are trusted local PyTorch checkpoint files and
JSON sidecars. The metadata schema is specific to ChannelThermal Prompt-3
global-field training with local physical coupling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from _helpers.model_utils import ensure_dir, load_trusted_checkpoint, write_json


def save_newhonf_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_newhonf_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> Dict[str, Any]:
    return load_trusted_checkpoint(path, map_location=map_location)


def write_checkpoint_summary(path: str | Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)
