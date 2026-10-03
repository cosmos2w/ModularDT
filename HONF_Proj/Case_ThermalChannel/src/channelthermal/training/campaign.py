"""Full-case campaign contracts, matched fresh initialization and telemetry.

This module leaves the native thermal losses and physical coupling intact.
Stage stops are independent of the absolute optimization schedule horizon.
"""

from __future__ import annotations

import copy
import os
import statistics
import subprocess
from typing import Any

import torch

from .native_denominators import physical_denominator_masses, physical_normalization_weights

CAMPAIGN_KEYS = {
    "name", "arm", "version", "parent", "schedule_total_epochs",
    "require_full_epoch", "matched_fresh_initialization", "gpu_telemetry",
    "structural_weight", "structural_ramp_start", "structural_ramp_end",
    "microbatch_size",
    "response_stencils",
    "native_loss_denominators_start_epoch", "physical_loss_policy_version",
}
HYPERGRAPH_ARCHITECTURES = frozenset({
    "adaptive_receiver_hypergraph_honf", "overlap_control_hypergraph_honf",
    "local_overlap_hypergraph_honf",
})
ORGANIZER_PREFIXES = ("core.backend.organizer.", "core.backend.control_gain.", "core.backend.control_score.")


def structural_calibration_candidate(state: dict, counts: list[int], training_counts: list[int]) -> tuple[int, ...] | None:
    """Select an unsampled training M stratum; mixed boundary batches wait."""

    values = sorted({int(value) for value in training_counts})
    if not values or not counts:
        return None
    number = min(5, len(values))
    strata = [values[index * len(values) // number:(index + 1) * len(values) // number] for index in range(number)]
    if state.get("structural_scale_samples") and not state.get("structural_calibration_samples"):
        raise ValueError("Existing unstratified structural calibration requires an explicit policy-lineage migration.")
    state["structural_calibration_policy"] = "training_module_count_strata_v2"
    state["calibration_policy_version"] = 2
    state.setdefault("structural_calibration_planned_strata", strata)
    if state["structural_calibration_planned_strata"] != strata:
        raise ValueError("Structural calibration training module-count strata changed on resume.")
    sampled = {tuple(sample["stratum"]) for sample in state.get("structural_calibration_samples", [])}
    for stratum in strata:
        if set(counts).issubset(stratum) and tuple(stratum) not in sampled:
            return tuple(stratum)
    return None


def record_structural_calibration(state: dict, stratum: tuple[int, ...], counts: list[int], *,
                                  task_norm: float, cost_norm: float, max_weight: float,
                                  epoch: int, native_batch: int) -> None:
    """Bound pressure by observed task signal and retain its train-only basis."""

    scale = min(.02 * task_norm / cost_norm, max_weight) if task_norm > 1e-10 and cost_norm > 1e-10 else 0.
    state.setdefault("structural_calibration_samples", []).append({
        "stratum": list(stratum), "observed_module_counts": sorted(set(counts)),
        "epoch": epoch, "native_batch": native_batch,
        "task_organizer_gradient_norm": task_norm, "cost_organizer_gradient_norm": cost_norm,
        "coefficient": scale,
    })
    state.setdefault("structural_scale_samples", []).append(scale)
    state["structural_scale"] = float(statistics.median(state["structural_scale_samples"]))
    state["structural_calibration_observed_strata"] = [sample["stratum"] for sample in state["structural_calibration_samples"]]
    state["structural_calibration_complete"] = len(state["structural_calibration_samples"]) == len(state["structural_calibration_planned_strata"])
    state["last_task_organizer_gradient_norm"] = task_norm
    state["last_cost_organizer_gradient_norm"] = cost_norm


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
    if settings.get("native_loss_denominators_start_epoch") is not None:
        if int(settings["native_loss_denominators_start_epoch"]) != 101 or settings.get("physical_loss_policy_version") != 2:
            raise ValueError("Native denominator policy 2 begins at epoch 101 after the common initial 100-epoch screen.")
    elif settings.get("physical_loss_policy_version", 1) != 1:
        raise ValueError("Physical loss policy 2 requires its explicit native denominator activation epoch.")
    return settings


def validate_campaign_resume(checkpoint: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    """Keep data/task/optimizer policy and absolute schedule unchanged at stage resumes."""

    current = config.get("training", {}).get("campaign")
    if not current:
        return
    saved = checkpoint.get("train_config", {})
    saved_training = saved.get("training", {})
    current_training = config.get("training", {})
    amendment = None
    saved_campaign = saved_training.get("campaign") or {}
    if saved_campaign != current:
        policy_keys = {"native_loss_denominators_start_epoch", "physical_loss_policy_version"}
        previous = {key: value for key, value in saved_campaign.items() if key not in policy_keys}
        requested = {key: value for key, value in current.items() if key not in policy_keys}
        if (previous != requested or int(checkpoint.get("epoch", 0)) != 100
                or saved_campaign.get("physical_loss_policy_version", 1) != 1
                or saved_campaign.get("native_loss_denominators_start_epoch") is not None
                or current.get("physical_loss_policy_version") != 2
                or current.get("native_loss_denominators_start_epoch") != 101):
            raise ValueError("Campaign resume cannot reset or silently amend its schedule/lineage.")
        amendment = {"physical_loss_policy_from": 1, "physical_loss_policy_to": 2,
                     "checkpoint_epoch": 100, "activation_epoch": 101,
                     "scope": "explicit common native query/module/valid-port denominators; architecture version unchanged"}
    for section in ("dataset", "loss"):
        if saved.get(section) != config.get(section):
            raise ValueError(f"Campaign resume changed {section}; record a separate training-policy version.")
    for key in ("seed", "learning_rate", "organizer_learning_rate", "weight_decay", "amp", "gradient_clip_norm", "port_curriculum"):
        if saved_training.get(key) != current_training.get(key):
            raise ValueError(f"Campaign resume changed training.{key}.")
    if not checkpoint.get("optimizer_state_dict") or not checkpoint.get("rng_state"):
        raise ValueError("Campaign continuation requires optimizer and RNG checkpoint state.")
    if current_training.get("amp") and not checkpoint.get("scaler_state_dict"):
        raise ValueError("AMP campaign continuation requires gradient-scaler checkpoint state.")
    return amendment


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

    def __init__(self, loader: Any, microbatch_size: int, *, native_loss_denominators: bool = False):
        self.loader = loader
        self.dataset = loader.dataset
        self.microbatch_size = int(microbatch_size)
        self.native_loss_denominators = native_loss_denominators
        if self.microbatch_size <= 0:
            raise ValueError("Campaign microbatch_size must be positive.")

    def __iter__(self):
        for batch_index, batch in enumerate(self.loader):
            count = int(batch["field_targets"].shape[0])
            native_masses = physical_denominator_masses(batch) if self.native_loss_denominators else None
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
                if native_masses is not None:
                    result["_native_loss_normalization_weights"] = physical_normalization_weights(result, native_masses, (stop-start)/count)
                result["_auxiliary_due"] = batch_index == len(self.loader) - 1 and stop == count
                yield result
