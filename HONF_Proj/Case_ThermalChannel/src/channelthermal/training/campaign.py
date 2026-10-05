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
    "heat_null_response",
    "structural_measure_policy_version",
    "organizer_gradient_policy",
    "interface_fit",
}
HYPERGRAPH_ARCHITECTURES = frozenset({
    "adaptive_receiver_hypergraph_honf", "overlap_control_hypergraph_honf",
    "local_overlap_hypergraph_honf", "faithful_receiver_hypergraph_honf", "native_context_tree_honf",
})
NATIVE_CONTEXT_ARCHITECTURES = frozenset({"native_context_tree_honf", "native_context_global_control_honf"})
ORGANIZER_PREFIXES = ("core.backend.organizer.", "core.backend.control_gain.", "core.backend.control_score.",
                      "core.backend.pair_controls.")


def managed_model_matches(saved: dict, current: dict, *, campaign: dict) -> bool:
    """Compare original run models before the native checkpoint resume check.

    The native workflow still validates the exact checkpoint epoch, objective
    amendment, architecture, optimizer and RNG before a resumed update. A run's
    original resolved config stays at policy1 after an approved continuation,
    so this preliminary comparison also supports later policy2 checkpoints.
    """
    if saved == current:
        return True
    previous, requested = copy.deepcopy(saved), copy.deepcopy(current)
    old_core, new_core = previous.get("core_honf", {}), requested.get("core_honf", {})
    if (old_core.get("forward_architecture") != "faithful_receiver_hypergraph_honf"
            or new_core.get("forward_architecture") != "faithful_receiver_hypergraph_honf"):
        return False
    old_options = old_core.setdefault("interface_model", {}).setdefault("hypergraph_options", {})
    new_options = new_core.setdefault("interface_model", {}).setdefault("hypergraph_options", {})
    old_version = old_options.setdefault("structural_measure_policy_version", 1)
    new_version = new_options.setdefault("structural_measure_policy_version", 1)
    if old_version == 1 and new_version == 2:
        if (campaign.get("structural_measure_policy_version") != 2
                or campaign.get("physical_loss_policy_version") != 2
                or campaign.get("native_loss_denominators_start_epoch") != 101):
            return False
        old_options["structural_measure_policy_version"] = 2
    return previous == requested


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
    gradient_policy = settings.get("organizer_gradient_policy", "whole_wrapper_shadow_v1")
    architecture = config.get("model", {}).get("core_honf", {}).get("forward_architecture")
    if gradient_policy not in {"whole_wrapper_shadow_v1", "local_context_shadow_v1", "ordinary_task_v1"}:
        raise ValueError("Unknown organizer gradient policy.")
    if architecture in NATIVE_CONTEXT_ARCHITECTURES:
        from .interface_fit import interface_fit_enabled, validate_interface_fit_campaign
        expected = "local_context_shadow_v1" if architecture == "native_context_tree_honf" else "ordinary_task_v1"
        dataset = config.get("dataset", {})
        development = bool(dataset.get("development_manifest") or dataset.get("development_subset"))
        expected_horizon = 1000 if development else 5000
        if settings.get("interface_fit") or interface_fit_enabled(config.get("model", {})):
            if architecture != "native_context_global_control_honf":
                raise ValueError("Frozen query-interface fits attach only to G-fast.")
            validate_interface_fit_campaign(settings, config)
        elif (gradient_policy != expected or settings.get("parent") is not None or horizon != expected_horizon
                or not settings.get("matched_fresh_initialization")
                or settings.get("heat_null_response") != {"coefficient": 0.0}
                or config.get("model", {}).get("channelthermal", {}).get("global_feature_schema") != "source_local_v3"):
            raise ValueError("Native-context controls require matched fresh source_local_v3 initialization and their declared task-gradient policy.")
        if architecture == "native_context_global_control_honf" and settings.get("structural_weight") != 0.0:
            raise ValueError("Global-C has no structural objective.")
    elif gradient_policy == "ordinary_task_v1":
        raise ValueError("Ordinary task policy is declared only for the native-context Global-C control.")
    elif gradient_policy == "local_context_shadow_v1":
        parent = settings.get("parent")
        if (config.get("model", {}).get("core_honf", {}).get("forward_architecture")
                != "faithful_receiver_hypergraph_honf"
                or not isinstance(parent, dict)
                or set(parent) != {"checkpoint", "epoch", "from", "to"}
                or not isinstance(parent.get("checkpoint"), str) or not parent["checkpoint"].startswith("/")
                or parent.get("epoch") != 200
                or parent.get("from") != "whole_wrapper_shadow_v1" or parent.get("to") != gradient_policy
                or horizon != 1000 or settings.get("physical_loss_policy_version") != 2):
            raise ValueError("Local-context shadow requires an explicit exact-e200 Tree-F child lineage.")
    if settings.get("native_loss_denominators_start_epoch") is not None:
        if int(settings["native_loss_denominators_start_epoch"]) != 101 or settings.get("physical_loss_policy_version") != 2:
            raise ValueError("Native denominator policy 2 begins at epoch 101 after the common initial 100-epoch screen.")
    elif settings.get("physical_loss_policy_version", 1) != 1:
        raise ValueError("Physical loss policy 2 requires its explicit native denominator activation epoch.")
    measure_policy = settings.get("structural_measure_policy_version")
    core = config.get("model", {}).get("core_honf", {})
    option = core.get("interface_model", {}).get("hypergraph_options", {}).get("structural_measure_policy_version", 1)
    if architecture in {"faithful_receiver_hypergraph_honf", "native_context_tree_honf"} and option == 2 and measure_policy != 2:
        raise ValueError("Eligible-mechanism model option 2 requires an explicit reviewed campaign policy.")
    if measure_policy is not None and (measure_policy != 2 or option != 2
            or architecture not in {"faithful_receiver_hypergraph_honf", "native_context_tree_honf"}
            or (architecture == "faithful_receiver_hypergraph_honf"
                and settings.get("native_loss_denominators_start_epoch") != 101)):
        raise ValueError("Eligible-mechanism policy 2 requires its explicit reviewed Tree-F epoch101 option.")
    null = settings.get("heat_null_response")
    if null is not None:
        if architecture in NATIVE_CONTEXT_ARCHITECTURES:
            if null != {"coefficient": 0.0}:
                raise ValueError("Native-context comparison disables the old heat-null TRAIN objective with explicit coefficient zero.")
            return settings
        if not isinstance(null, dict) or null.get("benchmark_verified") is not True:
            raise ValueError("Heat-null auxiliary requires explicit verified benchmark provenance.")
        if (settings.get("native_loss_denominators_start_epoch") != 101
                or null.get("cases_per_epoch", 2) != 2 or null.get("fluid_queries", 256) != 256
                or not 0 < float(null.get("fraction", .1)) <= .25):
            raise ValueError("Heat-null auxiliary follows the reviewed epoch101 two-case Q256 protocol.")
        if config.get("model", {}).get("core_honf", {}).get("forward_architecture") not in {
                "faithful_receiver_hypergraph_honf", "direct_pairwise_control_honf", "three_term_full_access_honf"}:
            raise ValueError("This heat-null amendment requires a reviewed source-local Tree-F/Pair-F/Fine-F backend.")
        if core.get("forward_architecture") == "three_term_full_access_honf" and (
                settings.get("arm") != "Fine-F" or settings.get("structural_weight") != 0.0
                or config.get("model", {}).get("channelthermal", {}).get("global_feature_schema") != "source_local_v3"):
            raise ValueError("Fine-F heat-null exposure requires the source_local_v3 backbone and no structural objective.")
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
        old_gradient = saved_campaign.get("organizer_gradient_policy", "whole_wrapper_shadow_v1")
        new_gradient = current.get("organizer_gradient_policy", "whole_wrapper_shadow_v1")
        gradient_keys = {"organizer_gradient_policy", "parent"}
        previous_gradient = {key: value for key, value in saved_campaign.items() if key not in gradient_keys}
        requested_gradient = {key: value for key, value in current.items() if key not in gradient_keys}
        gradient_transition = (old_gradient == "whole_wrapper_shadow_v1"
            and new_gradient == "local_context_shadow_v1" and int(checkpoint.get("epoch", 0)) == 200
            and previous_gradient == requested_gradient and saved_campaign.get("parent") is None
            and checkpoint.get("model_config", {}).get("core_honf", {}).get("forward_architecture")
                == "faithful_receiver_hypergraph_honf")
        if gradient_transition:
            validate_campaign(config)
            amendment = {"organizer_gradient_policy": {"from": old_gradient, "to": new_gradient,
                "checkpoint_epoch": 200, "activation_epoch": 201, "parent": copy.deepcopy(current["parent"])}}
        elif new_gradient != old_gradient or saved_campaign.get("parent") != current.get("parent"):
            raise ValueError("Campaign resume cannot reset or silently amend its schedule/lineage.")
        if not gradient_transition:
            amendment = _validate_physical_campaign_amendment(checkpoint, saved_campaign, current)
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


def _validate_physical_campaign_amendment(checkpoint: dict, saved_campaign: dict, current: dict) -> dict:
    """Keep the earlier common e101 objective amendment isolated from child policy changes."""

    amendment = None
    if saved_campaign != current:
        policy_keys = {"native_loss_denominators_start_epoch", "physical_loss_policy_version", "heat_null_response",
                       "structural_measure_policy_version"}
        previous = {key: value for key, value in saved_campaign.items() if key not in policy_keys}
        requested = {key: value for key, value in current.items() if key not in policy_keys}
        if (previous != requested or int(checkpoint.get("epoch", 0)) != 100
                or saved_campaign.get("physical_loss_policy_version", 1) != 1
                or saved_campaign.get("native_loss_denominators_start_epoch") is not None
                or (saved_campaign.get("heat_null_response") is not None
                    and not (checkpoint.get("model_config", {}).get("core_honf", {}).get("forward_architecture")
                             in NATIVE_CONTEXT_ARCHITECTURES
                             and saved_campaign.get("heat_null_response") == current.get("heat_null_response") == {"coefficient": 0.0}))
                or (saved_campaign.get("structural_measure_policy_version", 1) != 1
                    and not (checkpoint.get("model_config", {}).get("core_honf", {}).get("forward_architecture")
                             == "native_context_tree_honf"
                             and saved_campaign.get("structural_measure_policy_version")
                             == current.get("structural_measure_policy_version") == 2))
                or current.get("physical_loss_policy_version") != 2
                or current.get("native_loss_denominators_start_epoch") != 101):
            raise ValueError("Campaign resume cannot reset or silently amend its schedule/lineage.")
        amendment = {"physical_loss_policy_from": 1, "physical_loss_policy_to": 2,
                     "checkpoint_epoch": 100, "activation_epoch": 101,
                     "scope": "explicit common native query/module/valid-port denominators; architecture version unchanged"}
        if current.get("heat_null_response") is not None:
            amendment["heat_null_response"] = copy.deepcopy(current["heat_null_response"])
        if current.get("structural_measure_policy_version") == 2 and saved_campaign.get("structural_measure_policy_version", 1) == 1:
            amendment["structural_measure_policy"] = {
                "from": 1, "to": 2, "activation_epoch": 101,
                "scope": "exclude entirely ineligible mechanisms from the physical-measure mean; recalibrate training-only pressure",
            }
    return amendment


def amend_structural_calibration(state: dict, amendment: dict | None) -> None:
    """Preserve the first-screen calibration before the explicit cost change."""
    if not amendment or "structural_measure_policy" not in amendment:
        return
    keys = {key for key in state if key.startswith("structural_calibration_") or key in {
        "structural_scale", "structural_scale_samples", "calibration_policy_version",
        "last_task_organizer_gradient_norm", "last_cost_organizer_gradient_norm"}}
    state["structural_measure_policy_1_calibration"] = {key: copy.deepcopy(state.pop(key)) for key in sorted(keys)}


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


def canonical_initialization_config(model_config: Any) -> Any:
    """Select a fresh reference without leaking architecture-specific options."""
    canonical = copy.deepcopy(model_config)
    if canonical.core_honf.forward_architecture in NATIVE_CONTEXT_ARCHITECTURES:
        canonical.core_honf.forward_architecture = "native_context_global_control_honf"
        options = canonical.core_honf.interface_model.hypergraph_options
        canonical.core_honf.interface_model.hypergraph_options = {key: value for key, value in options.items()
                                                                 if key == "organizer_dim"}
    else:
        canonical.core_honf.forward_architecture = "three_term_full_access_honf"
    return canonical


def heat_null_training_enabled(settings: dict) -> bool:
    """Explicit zero-weight native-context controls perform no null TRAIN calls."""
    null = settings.get("heat_null_response")
    return null is not None and null != {"coefficient": 0.0}


def copy_matched_native_context_initial_state(target: torch.nn.Module, canonical: torch.nn.Module) -> dict[str, Any]:
    """Match materialized physical/context tensors and meaningfully shared controls."""
    shared_controls = ("core.backend.organizer.control_geometry_encoder.", "core.backend.organizer.control_heads.",
                       "core.backend.control_gain.", "core.backend.control_score.")
    source, current = canonical.state_dict(), target.state_dict()
    loaded, excluded = [], []
    for name, value in current.items():
        if name.startswith(ORGANIZER_PREFIXES) and not name.startswith(shared_controls):
            excluded.append(name)
            continue
        if name not in source or source[name].shape != value.shape:
            raise ValueError(f"Fresh common native-context initial tensor is missing or differently shaped: {name}")
        current[name] = source[name].detach().clone()
        loaded.append(name)
    target.load_state_dict(current, strict=True)
    if not all(torch.equal(target.state_dict()[name], source[name]) for name in loaded):
        raise ValueError("Fresh native-context tensor copy failed its exact comparison.")
    if not any(name.startswith(shared_controls[0]) for name in loaded) or not any(name.startswith(shared_controls[1]) for name in loaded):
        raise ValueError("Matched native-context initialization lacks materialized common control geometry/heads.")
    return {"source": "fresh_materialized_Global-C_seed0", "epoch": 0,
            "loaded": [{"name": name, "shape": list(source[name].shape)} for name in loaded],
            "exact_common_tensors_equal": True, "tree_specific_organizer_independent": excluded}


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
    # Keep the legacy device rows intact. total-used includes driver-reserved
    # memory on some systems and must not be interpreted as allocatable free.
    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.reserved", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        sample["device_memory"] = []
        for row in rows:
            index, uuid, free, reserved = (value.strip() for value in row.split(","))
            sample["device_memory"].append({"physical_index": int(index), "uuid": uuid,
                "free_MiB": None if free == "N/A" else float(free),
                "driver_reserved_MiB": None if reserved == "N/A" else float(reserved)})
        sample["device_memory_scope"] = "Explicit nvidia-smi memory.free and memory.reserved; not total-minus-used or allocator cache"
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        sample.pop("device_memory", None)
        sample["device_memory_unavailable"] = str(error)
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
