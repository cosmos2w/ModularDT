"""Opt-in physical response refinement; historical frozen fitting stays separate."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import torch

from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.compat import strip_module_prefix

from .interface_fit import FIT_KIND, FIT_PARAMETER_PREFIX, validate_interface_fit_campaign
from .lean_attachment import MANIFEST_SHA256

REFINEMENT_KIND = "physical_response_refinement_from_fit100"
POLICY_VERSION = "query_physical_v1"
PHYSICAL_PREFIXES = tuple("core.backend." + component + "." for component in (
    "mm_message", "me_message", "em_message", "module_update", "env_update",
    "query_module_message", "query_module_output", "env_query", "env_attention", "env_geometry_bias",
)) + ("core.common.field_head.", "local_coupling.port_head.", "local_coupling.port_refinement_head.")
TRAINABLE_PREFIXES = (FIT_PARAMETER_PREFIX,) + PHYSICAL_PREFIXES
RUNTIME_BUFFER_NAMES = {"_selection_epoch_state", "_selection_total_epochs_state"}
POLICY_DECLARATION = {
    "policy_version": POLICY_VERSION, "backbone_epoch": 1000, "interface_prefit_epoch": 100,
    "admission_mode": "soft", "organizer_learning_rate": 1e-4,
    "physical_learning_rate": 1e-5, "weight_decay": 1e-5,
}


def refinement_enabled(config: dict) -> bool:
    return bool(config.get("training", {}).get("campaign", {}).get("forward_refinement"))


def _model_dict(config: Any) -> dict:
    return config.to_dict() if hasattr(config, "to_dict") else copy.deepcopy(config)


def _options(config: dict) -> dict:
    return config.get("core_honf", {}).get("interface_model", {}).get("hypergraph_options", {})


def _partition_valid(dataset: dict) -> bool:
    partitions = (dataset.get("development_subset") or {}).get("partitions", {})
    return (dataset.get("development_manifest_sha256") == MANIFEST_SHA256
            and partitions.get("train", {}).get("selected_count") == 150
            and partitions.get("test", {}).get("selected_count") == 22)


def validate_refinement_campaign(settings: dict, config: dict) -> None:
    """Reject implicit co-adaptation, data expansion, or curriculum changes."""
    from .response_refinement import validate_response_refinement_declaration

    options = _options(config.get("model", {}))
    parent = settings.get("parent") or {}
    training = config.get("training", {})
    if (settings.get("forward_refinement") != POLICY_DECLARATION
            or settings.get("interface_fit") is not None
            or options.get("tensor_query_interaction") is not True
            or options.get("global_fast_reader") is not True
            or options.get("query_admission_mode") != "soft"
            or options.get("query_interface_parent_epoch", 1000) != 1000
            or settings.get("arm") != {"add": "H-add", "joint": "H-joint"}.get(options.get("query_interaction_mode"))
            or parent.get("kind") != REFINEMENT_KIND or parent.get("epoch") != 100
            or not isinstance(parent.get("checkpoint"), str) or not parent["checkpoint"].startswith("/")
            or settings.get("matched_fresh_initialization") is not False
            or settings.get("schedule_total_epochs") != 1000
            or settings.get("organizer_gradient_policy") != "ordinary_task_v1"
            or settings.get("physical_loss_policy_version") != 2
            or settings.get("native_loss_denominators_start_epoch") != 101
            or settings.get("structural_weight") != 0.0
            or config.get("model", {}).get("channelthermal", {}).get("global_feature_schema") != "source_local_v3"
            or not _partition_valid(config.get("dataset", {}))
            or training.get("seed") != 0 or training.get("amp") is not False
            or training.get("learning_rate") != 1e-5 or training.get("weight_decay") != 1e-5
            or training.get("organizer_learning_rate") is not None
            or training.get("functional_detail_controller_learning_rate") is not None
            or not isinstance(settings.get("response_addendum"), dict)
            or not isinstance(settings.get("response_refinement"), dict)):
        raise ValueError("Refinement requires explicit matched fit100, fixed25 primary data, soft admission and query_physical_v1 policy.")
    validate_response_refinement_declaration(settings)


def refinement_parameter_inventory(model: torch.nn.Module) -> dict:
    """Exact semantic allowlist; every named physical component must be present."""
    named = dict(model.named_parameters())
    missing = [prefix for prefix in TRAINABLE_PREFIXES if not any(name.startswith(prefix) for name in named)]
    if missing:
        raise ValueError(f"Refinement physical components missing: {missing}")
    groups = {"organizer": [], "physical": [], "frozen": []}
    counts = {key: 0 for key in groups}
    complete = True
    for name, parameter in named.items():
        group = ("organizer" if name.startswith(FIT_PARAMETER_PREFIX) else
                 "physical" if name.startswith(PHYSICAL_PREFIXES) else "frozen")
        groups[group].append(name)
        try:
            counts[group] += parameter.numel()
        except ValueError:
            complete = False
    return {"policy_version": POLICY_VERSION, "allowlist": list(TRAINABLE_PREFIXES),
            "parameter_names": {key: sorted(value) for key, value in groups.items()},
            "scalar_counts": counts if complete else None, "materialized": complete,
            "buffer_names": sorted(name for name, _ in model.named_buffers())}


def apply_refinement_policy(model: torch.nn.Module) -> dict:
    """Enable only physical/organizer parameters; activation autograd stays live."""
    inventory = refinement_parameter_inventory(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(TRAINABLE_PREFIXES)
    validate_refinement_trainable_inventory(model, inventory)
    return inventory


def validate_refinement_trainable_inventory(model: torch.nn.Module, expected: dict | None = None) -> dict:
    inventory = refinement_parameter_inventory(model)
    if getattr(model.core.backend.tensor_query_interaction, "admission_mode", None) != "soft":
        raise ValueError("Refinement runtime organizer must use explicit soft admission.")
    intended = sorted(inventory["parameter_names"]["organizer"] + inventory["parameter_names"]["physical"])
    actual = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if actual != intended or (expected is not None and inventory["parameter_names"] != expected["parameter_names"]):
        raise ValueError("Refinement trainable/frozen inventory changed from the exact semantic allowlist.")
    return inventory


def capture_refinement_frozen_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Include nonpersistent normalization buffers as well as frozen parameters."""
    validate_refinement_trainable_inventory(model)
    state = {name: value for name, value in model.named_parameters() if not value.requires_grad}
    state.update({name: value for name, value in model.named_buffers()
                  if name.rsplit(".", 1)[-1] not in RUNTIME_BUFFER_NAMES})
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def assert_refinement_frozen_state(model: torch.nn.Module, frozen: dict[str, torch.Tensor]) -> None:
    current = capture_refinement_frozen_state(model)
    if current.keys() != frozen.keys() or any(not torch.equal(current[name], value) for name, value in frozen.items()):
        raise ValueError("Refinement changed frozen parameters or normalization buffers.")


def _frozen_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(capture_refinement_frozen_state(model).items()):
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_refinement_optimizer(model: torch.nn.Module, training: dict):
    from .optimizer import _optimizer_group_record, _optimizer_inventory_digest

    declaration = training.get("campaign", {}).get("forward_refinement")
    if declaration != POLICY_DECLARATION or training.get("learning_rate") != 1e-5 or training.get("weight_decay") != 1e-5:
        raise ValueError("Refinement optimizer requires its exact declared rates and policy.")
    inventory = validate_refinement_trainable_inventory(model)
    named = dict(model.named_parameters())
    groups, records = [], []
    for name, lr in (("organizer", 1e-4), ("physical", 1e-5)):
        pairs = [(key, named[key]) for key in inventory["parameter_names"][name]]
        groups.append({"params": [parameter for _, parameter in pairs], "lr": lr})
        records.append(_optimizer_group_record(name, lr, pairs))
    optimizer = torch.optim.AdamW(groups, lr=1e-5, weight_decay=1e-5)
    return optimizer, {"mode": "split", "weight_decay": 1e-5, "groups": records,
                       "group_structure_sha256": _optimizer_inventory_digest(records)}


def _identity(config: dict) -> dict:
    """Bind scientific choices, allowing only run length/runtime/I/O changes."""
    training = config.get("training", {})
    training_keys = ("seed", "amp", "learning_rate", "weight_decay", "batch_size", "effective_batch_size",
                     "gradient_accumulation_steps", "query_points", "queries_per_case", "gradient_clip_norm",
                     "port_curriculum", "query_sampling", "optimizer", "lr_scheduler")
    return {"model": config.get("model"), "dataset": config.get("dataset"), "loss": config.get("loss"),
            "campaign": training.get("campaign"),
            "training": {key: training[key] for key in training_keys if key in training}}


def _identity_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(_identity(config), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_refinement_parent(checkpoint: dict, refinement_config: dict) -> None:
    validate_checkpoint_identity(checkpoint, case_id="ThermalChannel", model_family="honf_forward", workflow="forward")
    config = checkpoint.get("train_config", {})
    campaign = config.get("training", {}).get("campaign", {})
    validate_interface_fit_campaign(campaign, config)
    receipt = checkpoint.get("campaign_training_state", {}).get("interface_fit_attachment") or {}
    options = _options(checkpoint.get("model_config", {}))
    requested = _options(refinement_config.get("model", {}))
    if (checkpoint.get("epoch") != 100
            or checkpoint.get("selection_state") != {"epoch": 100, "total_epochs": 1000}
            or receipt.get("kind") != FIT_KIND or receipt.get("inherited_backbone_epoch") != 1000
            or options.get("query_interaction_mode") != requested.get("query_interaction_mode")
            or options.get("query_admission_mode", "curriculum") != "curriculum"
            or not checkpoint.get("rng_state") or not checkpoint.get("optimizer_state_dict")
            or checkpoint.get("campaign_training_state", {}).get("forward_refinement_attachment")):
        raise ValueError("Refinement requires its own historical frozen fit100 parent, never selected fit300 or prior refinement.")


def attach_refinement_checkpoint(checkpoint, *, target_model, target_config, refinement_config,
                                 source_path, source_sha256):
    """Copy every parent value and normalizer, reset only optimization/refinement age."""
    from .response_refinement import response_addendum_identity

    campaign = refinement_config["training"]["campaign"]
    validate_refinement_campaign(campaign, refinement_config)
    validate_refinement_parent(checkpoint, refinement_config)
    if str(source_path) != campaign["parent"]["checkpoint"] or not source_sha256:
        raise ValueError("Refinement parent path/hash must match declared provenance.")
    previous = copy.deepcopy(checkpoint["model_config"])
    requested = _model_dict(target_config)
    if requested != refinement_config["model"]:
        raise ValueError("Refinement target model and launch configuration differ.")
    for payload in (previous, requested):
        _options(payload).pop("query_admission_mode", None)
    if previous != requested:
        raise ValueError("Refinement may change only the explicit admission mode, preserving its own parent architecture.")
    for key in ("dataset", "loss"):
        if refinement_config[key] != checkpoint["train_config"][key]:
            raise ValueError(f"Refinement changed inherited primary {key}.")
    source = strip_module_prefix(checkpoint["model_state_dict"])
    target_model.load_state_dict(source, strict=True)
    if any(not torch.equal(value, target_model.state_dict()[name]) for name, value in source.items()):
        raise ValueError("Refinement failed to retain exact parent values.")
    inventory = apply_refinement_policy(target_model)
    optimizer, optimizer_inventory = build_refinement_optimizer(target_model, refinement_config["training"])
    target_model.set_training_progress(epoch=0, total_epochs=1000)
    receipt = {"kind": REFINEMENT_KIND, "source_checkpoint": str(source_path), "source_checkpoint_sha256": source_sha256,
               "inherited_backbone_epoch": 1000, "interface_prefit_epoch": 100, "refinement_epoch": 0,
               "policy": copy.deepcopy(POLICY_DECLARATION), "parameter_inventory": inventory,
               "frozen_state_sha256": _frozen_digest(target_model),
               "response_addendum_identity": response_addendum_identity(campaign),
               "identity_sha256": _identity_digest(refinement_config),
               "optimizer_moments": "empty; fit-only moments deliberately discarded"}
    child = copy.deepcopy(checkpoint)
    child.update(epoch=0, current_epoch=0, selection_state={"epoch": 0, "total_epochs": 1000},
                 model_config=_model_dict(target_config), model_state_dict=copy.deepcopy(target_model.state_dict()),
                 train_config=copy.deepcopy(refinement_config), optimizer_state_dict=optimizer.state_dict(),
                 optimizer_group_inventory=optimizer_inventory, scaler_state_dict=None,
                 best_metric=float("inf"), best_metrics={})
    previous_state = child.get("campaign_training_state", {})
    state = {"previous_interface_fit_training_state": previous_state, "forward_refinement_attachment": receipt}
    child["campaign_training_state"] = state
    target_model.campaign_training_state = copy.deepcopy(state)
    return child, receipt


def validate_refinement_resume(checkpoint, model, campaign, config=None) -> None:
    """Reject silent policy, auxiliary-data, soft-mode, and normalization drift."""
    from .response_refinement import response_addendum_identity

    validate_checkpoint_identity(checkpoint, case_id="ThermalChannel", model_family="honf_forward", workflow="forward")
    receipt = checkpoint.get("campaign_training_state", {}).get("forward_refinement_attachment") or {}
    current = copy.deepcopy(config if config is not None else checkpoint.get("train_config", {}))
    current.setdefault("training", {})["campaign"] = copy.deepcopy(campaign)
    current["model"] = _model_dict(model.config)
    validate_refinement_campaign(campaign, current)
    validate_refinement_trainable_inventory(model, receipt.get("parameter_inventory"))
    if (receipt.get("kind") != REFINEMENT_KIND or receipt.get("policy") != POLICY_DECLARATION
            or receipt.get("inherited_backbone_epoch") != 1000 or receipt.get("interface_prefit_epoch") != 100
            or receipt.get("refinement_epoch") != checkpoint.get("epoch")
            or checkpoint.get("selection_state") != {"epoch": checkpoint.get("epoch"), "total_epochs": 1000}
            or campaign.get("parent", {}).get("checkpoint") != receipt.get("source_checkpoint")
            or receipt.get("identity_sha256") != _identity_digest(current)
            or receipt.get("response_addendum_identity") != response_addendum_identity(campaign)
            or receipt.get("frozen_state_sha256") != _frozen_digest(model)
            or checkpoint.get("model_config") != current["model"]):
        raise ValueError("Refinement resume changed declared identity, frozen state, or exact fit100 lineage.")
