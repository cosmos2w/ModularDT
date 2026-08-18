"""Atomic, provenance-complete inverse checkpoint utilities.

Checkpoints bind the `R,c -> G -> D -> G_hat` hierarchy to its frozen forward
model and immutable data/normalization contracts. They do not alter forward
checkpoint formats or forward run behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner


REQUIRED_PROVENANCE_KEYS = {
    "forward_checkpoint_id",
    "inverse_dataset_version",
    "inverse_dataset_hash",
    "request_schema_version",
    "compact_plan_schema_version",
    "normalization_stats",
}
TOPOLOGY_PROVENANCE_KEYS = {
    "topology_schema_name",
    "topology_schema_version",
    "forward_topology_checkpoint_sha256",
}


def save_inverse_checkpoint(
    path: str | Path,
    *,
    designer: HierarchicalInverseDesigner,
    stage: str,
    epoch: int,
    global_step: int,
    provenance: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    metrics: Mapping[str, float] | None = None,
) -> Path:
    missing = sorted(REQUIRED_PROVENANCE_KEYS - set(provenance))
    if designer.plan_flow.plan_token_mode == "exchangeable_set":
        missing.extend(sorted(TOPOLOGY_PROVENANCE_KEYS - set(provenance)))
    if missing:
        raise ValueError(f"Inverse checkpoint provenance is missing keys: {missing}")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    payload = {
        "checkpoint_schema_name": "honf_hierarchical_inverse",
        "checkpoint_schema_version": 1,
        "model_config": dict(designer.model_config),
        "model_state_dict": designer.state_dict(),
        "stage": str(stage),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "provenance": dict(provenance),
        "metrics": dict(metrics or {}),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_inverse_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=map_location, weights_only=False)
    if checkpoint.get("checkpoint_schema_name") != "honf_hierarchical_inverse":
        raise ValueError("Inverse checkpoint schema mismatch.")
    missing = sorted(REQUIRED_PROVENANCE_KEYS - set(checkpoint.get("provenance", {})))
    if checkpoint.get("model_config", {}).get("plan_token_mode", "indexed") == "exchangeable_set":
        missing.extend(
            sorted(TOPOLOGY_PROVENANCE_KEYS - set(checkpoint.get("provenance", {})))
        )
    if missing:
        raise ValueError(f"Inverse checkpoint provenance is incomplete: {missing}")
    return checkpoint


__all__ = [
    "REQUIRED_PROVENANCE_KEYS",
    "TOPOLOGY_PROVENANCE_KEYS",
    "load_inverse_checkpoint",
    "save_inverse_checkpoint",
]
