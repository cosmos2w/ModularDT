"""Full-case campaign contracts, matched fresh initialization and telemetry.

This module leaves the native thermal losses and physical coupling intact.
Stage stops are independent of the absolute optimization schedule horizon.
"""

from __future__ import annotations

import copy
import os
import subprocess
from typing import Any

import torch

CAMPAIGN_KEYS = {
    "name", "arm", "version", "parent", "schedule_total_epochs",
    "require_full_epoch", "matched_fresh_initialization", "gpu_telemetry",
    "structural_weight", "structural_ramp_start", "structural_ramp_end",
    "microbatch_size",
    "response_stencils",
}
HYPERGRAPH_ARCHITECTURES = frozenset({
    "adaptive_receiver_hypergraph_honf", "overlap_control_hypergraph_honf",
    "local_overlap_hypergraph_honf",
})
ORGANIZER_PREFIXES = ("core.backend.organizer.", "core.backend.control_gain.", "core.backend.control_score.")


def validate_campaign(config: dict[str, Any], *, max_train_batches: int | None = None) -> dict[str, Any]:
    """Reject partial epochs and warm starts in the primary fresh portfolio."""

    settings = copy.deepcopy(config.get("training", {}).get("campaign") or {})
    if not settings:
        return settings
    unknown = set(settings) - CAMPAIGN_KEYS
    if unknown:
        raise ValueError(f"Unknown campaign settings: {sorted(unknown)}")
    training = config.get("training", {})
    if settings.get("require_full_epoch", True) and (
        max_train_batches is not None or training.get("max_train_batches_per_epoch") is not None
    ):
        raise ValueError("Formal campaign epochs must visit the complete native training split; batch caps are smoke checks only.")
    horizon = int(settings.get("schedule_total_epochs", 5000))
    if horizon < int(training.get("epochs", 0)):
        raise ValueError("Campaign schedule horizon must cover the requested stage stop.")
    if settings.get("matched_fresh_initialization", False) and int(training.get("seed", -1)) != 0:
        raise ValueError("The primary matched fresh campaign uses seed 0.")
    settings["schedule_total_epochs"] = horizon
    return settings


def validate_campaign_resume(checkpoint: dict[str, Any], config: dict[str, Any]) -> None:
    """Keep data/task/optimizer policy and absolute schedule unchanged at stage resumes."""

    current = config.get("training", {}).get("campaign")
    if not current:
        return
    saved = checkpoint.get("train_config", {})
    saved_training = saved.get("training", {})
    current_training = config.get("training", {})
    if saved_training.get("campaign") != current:
        raise ValueError("Campaign resume cannot reset or silently amend its schedule/lineage.")
    for section in ("dataset", "loss"):
        if saved.get(section) != config.get(section):
            raise ValueError(f"Campaign resume changed {section}; record a separate training-policy version.")
    for key in ("seed", "learning_rate", "organizer_learning_rate", "weight_decay", "amp", "gradient_clip_norm", "port_curriculum"):
        if saved_training.get(key) != current_training.get(key):
            raise ValueError(f"Campaign resume changed training.{key}.")
    if not checkpoint.get("optimizer_state_dict") or not checkpoint.get("rng_state"):
        raise ValueError("Campaign continuation requires optimizer and RNG checkpoint state.")


def copy_matched_physical_initial_state(target: torch.nn.Module, canonical: torch.nn.Module) -> dict[str, Any]:
    """Copy materialized initial tensors, retaining independently initialized organizers."""

    source = canonical.state_dict()
    current = target.state_dict()
    loaded, excluded, unmatched = [], [], []
    for name, value in current.items():
        if name.startswith(ORGANIZER_PREFIXES):
            excluded.append(name)
            continue
        if name not in source or source[name].shape != value.shape:
            unmatched.append(name)
            continue
        current[name] = source[name].detach().clone()
        loaded.append(name)
    if unmatched:
        raise ValueError(f"Fresh common physical initial state has unmatched tensors: {unmatched}")
    target.load_state_dict(current, strict=True)
    return {"source": "fresh_materialized_B-fine_seed0", "epoch": 0, "loaded": loaded, "organizer_independent": excluded}


def gpu_contention_sample(device: torch.device) -> dict[str, Any]:
    """Record actual device/process occupancy; unavailable data stays explicit."""

    sample: dict[str, Any] = {"pid": os.getpid(), "logical_device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if device.type != "cuda":
        sample["scope"] = "cpu"
        return sample
    try:
        sample["devices"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        sample["compute_processes"] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        sample["unavailable"] = str(error)
    return sample


class CampaignMicrobatchLoader:
    """Split native bucket batches while preserving case-weighted optimizer steps."""

    def __init__(self, loader: Any, microbatch_size: int):
        self.loader = loader
        self.dataset = loader.dataset
        self.microbatch_size = int(microbatch_size)
        if self.microbatch_size <= 0:
            raise ValueError("Campaign microbatch_size must be positive.")

    def __iter__(self):
        for batch_index, batch in enumerate(self.loader):
            count = int(batch["field_targets"].shape[0])
            def sliced(value: Any, start: int, stop: int, count: int = count) -> Any:
                if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
                    return value[start:stop]
                if isinstance(value, dict):
                    return {key: sliced(item, start, stop) for key, item in value.items()}
                if isinstance(value, (list, tuple)) and len(value) == count:
                    return value[start:stop]
                return value
            for start in range(0, count, self.microbatch_size):
                stop = min(start + self.microbatch_size, count)
                result = sliced(batch, start, stop)
                result["_optimizer_start"] = start == 0
                result["_optimizer_boundary"] = stop == count
                result["_accumulation_weight"] = (stop - start) / count
                result["_auxiliary_due"] = batch_index == len(self.loader) - 1 and stop == count
                yield result
