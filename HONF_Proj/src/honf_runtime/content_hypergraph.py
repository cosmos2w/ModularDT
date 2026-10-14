"""Strict, dataset-free loading for the new content-hypergraph family."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .compat import load_trusted_checkpoint
from .registry import require_model_family


def load_content_hypergraph_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Load weights and native transforms without reading datasets or labels.

    Returns (model, physical_adapter, checkpoint). Thermal's physical adapter
    is the model containing the shared core; Wind returns the shared core and
    its native conversion adapter separately.
    """
    payload = load_trusted_checkpoint(path, map_location="cpu")
    identity = payload.get("experiment_identity", {})
    if payload.get("workflow") != "content_hypergraph_joint_v1":
        raise ValueError("This loader accepts only the content-hypergraph joint family.")
    if identity.get("model_family") != "honf_content_hypergraph":
        raise ValueError("Checkpoint family identity is missing or inconsistent.")
    require_model_family("honf_content_hypergraph", "forward")
    recipe = identity.get("recipe")
    if not isinstance(recipe, dict) or recipe.get("schema") != "content_hypergraph_recipe_v1":
        raise ValueError("Missing self-contained content-hypergraph recipe.")
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    if digest != identity.get("recipe_sha256"):
        raise ValueError("Checkpoint recipe checksum is inconsistent.")
    if recipe.get("architecture") != payload.get("arm"):
        raise ValueError("Checkpoint architecture differs from its recorded training arm.")
    provider = identity.get("provider_identity", {})
    if not isinstance(provider.get("model_config"), dict) or not isinstance(provider.get("adapter_state"), dict):
        raise TypeError("Checkpoint lacks its native model and TRAIN transform inventory.")
    if recipe.get("dataset") == "thermal":
        from channelthermal.content_hypergraph import ThermalContentHypergraphAdapter
        model = ThermalContentHypergraphAdapter.from_state_payload(provider["adapter_state"], device=device)
        adapter = model
    elif recipe.get("dataset") == "wind":
        from windfarm.content_hypergraph import WindContentHypergraphAdapter

        from honf_forward_core.content_hypergraph import ContentHypergraphCore
        model = ContentHypergraphCore(**provider["model_config"]).to(device)
        adapter = WindContentHypergraphAdapter.from_state_payload(provider["adapter_state"], device=device)
    else:
        raise ValueError("Unsupported physical dataset in content-hypergraph checkpoint.")
    if model.get_config() != provider["model_config"]:
        raise ValueError("Reconstructed model differs from checkpoint constructor metadata.")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise TypeError("Checkpoint has no model tensor state.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, adapter, payload
