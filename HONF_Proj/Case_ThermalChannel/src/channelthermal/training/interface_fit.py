"""Frozen G-fast interface fitting with a separate native fit-age lineage."""

from __future__ import annotations

import copy
from typing import Any

import torch

from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.compat import strip_module_prefix

from .lean_attachment import MANIFEST_SHA256

FIT_PARAMETER_PREFIX = "core.backend.tensor_query_interaction."
FIT_KIND = "frozen_G-fast_e1000_interface_fit"
FIT_OPTIONS = {
    "tensor_query_interaction", "query_interaction_mode", "query_distance_alpha",
    "query_distance_length", "query_background_zero_bias", "query_interface_parent_epoch",
}


def interface_fit_enabled(model_config: Any) -> bool:
    payload = model_config.to_dict() if hasattr(model_config, "to_dict") else model_config
    return payload.get("core_honf", {}).get("interface_model", {}).get("hypergraph_options", {}).get("tensor_query_interaction") is True


def freeze_interface_fit_model(model: torch.nn.Module) -> list[str]:
    """Freeze inherited parameters without disabling activation/input autograd."""
    names = []
    for name, parameter in model.named_parameters():
        # Attribute assignment also supports not-yet-materialized lazy layers.
        parameter.requires_grad = name.startswith(FIT_PARAMETER_PREFIX)
        if parameter.requires_grad:
            names.append(name)
    if not names:
        raise ValueError("Interface fit has no declared new query-interface parameters.")
    return sorted(names)


def validate_interface_fit_campaign(settings: dict, config: dict) -> None:
    """Accept this frozen experiment without broadening historical campaigns."""
    declaration = settings.get("interface_fit") or {}
    options = config.get("model", {}).get("core_honf", {}).get("interface_model", {}).get("hypergraph_options", {})
    parent = settings.get("parent") or {}
    dataset = config.get("dataset", {})
    development = dataset.get("development_subset") or {}
    if (not interface_fit_enabled(config.get("model", {}))
            or options.get("global_fast_reader") is not True
            or options.get("query_interaction_mode") not in {"add", "joint"}
            or settings.get("arm") != {"add": "H-add", "joint": "H-joint"}.get(options.get("query_interaction_mode"))
            or options.get("query_interface_parent_epoch", 1000) != 1000
            or declaration.get("backbone_epoch") != 1000
            or declaration.get("parameter_prefix") != FIT_PARAMETER_PREFIX
            or parent.get("kind") != FIT_KIND or parent.get("epoch") != 1000
            or not isinstance(parent.get("checkpoint"), str) or not parent["checkpoint"].startswith("/")
            or settings.get("matched_fresh_initialization") is not False
            or settings.get("schedule_total_epochs") != 1000
            or settings.get("organizer_gradient_policy") != "ordinary_task_v1"
            or settings.get("physical_loss_policy_version") != 2
            or settings.get("native_loss_denominators_start_epoch") != 101
            or settings.get("structural_weight") != 0.0
            or settings.get("heat_null_response") != {"coefficient": 0.0}
            or config.get("model", {}).get("channelthermal", {}).get("global_feature_schema") != "source_local_v3"
            or dataset.get("development_manifest_sha256") != MANIFEST_SHA256
            or development.get("partitions", {}).get("train", {}).get("selected_count") != 150
            or development.get("partitions", {}).get("test", {}).get("selected_count") != 22
            or config.get("training", {}).get("seed") != 0
            or config.get("training", {}).get("amp") is not False
            or config.get("training", {}).get("learning_rate") != 3e-4
            or config.get("training", {}).get("weight_decay") != 1e-5):
        raise ValueError("Interface fit requires declared frozen development G-fast e1000 lineage and its unchanged native objective.")


def validate_interface_fit_parent(checkpoint: dict[str, Any]) -> None:
    validate_checkpoint_identity(checkpoint, case_id="ThermalChannel", model_family="honf_forward", workflow="forward")
    cfg = checkpoint.get("train_config", {})
    options = checkpoint.get("model_config", {}).get("core_honf", {}).get("interface_model", {}).get("hypergraph_options", {})
    dataset = cfg.get("dataset", {})
    partitions = (dataset.get("development_subset") or {}).get("partitions", {})
    if (checkpoint.get("epoch") != 1000
            or checkpoint.get("selection_state") != {"epoch": 1000, "total_epochs": 1000}
            or checkpoint.get("model_config", {}).get("core_honf", {}).get("forward_architecture") != "native_context_global_control_honf"
            or options.get("global_fast_reader") is not True
            or options.get("tensor_source_residual") or options.get("tensor_query_interaction")
            or dataset.get("development_manifest_sha256") != MANIFEST_SHA256
            or partitions.get("train", {}).get("selected_count") != 150
            or partitions.get("test", {}).get("selected_count") != 22
            or cfg.get("training", {}).get("amp") is not False
            or cfg.get("training", {}).get("campaign", {}).get("physical_loss_policy_version") != 2
            or checkpoint.get("model_config", {}).get("channelthermal", {}).get("global_feature_schema") != "source_local_v3"
            or not checkpoint.get("rng_state") or not checkpoint.get("optimizer_state_dict")):
        raise ValueError("Interface fit requires exact development G-fast e1000, not formal or Tensor-H weights.")


def attach_interface_fit_checkpoint(checkpoint, *, target_model, target_config, fit_config,
                                    source_path, source_sha256):
    """Attach zero new controls, freeze the host, and create empty new AdamW state."""
    from .optimizer import build_forward_optimizer

    validate_interface_fit_parent(checkpoint)
    validate_interface_fit_campaign(fit_config["training"]["campaign"], fit_config)
    previous = copy.deepcopy(checkpoint["model_config"])
    requested = copy.deepcopy(target_config.to_dict())
    for payload in (previous, requested):
        options = payload["core_honf"]["interface_model"].setdefault("hypergraph_options", {})
        for key in FIT_OPTIONS:
            options.pop(key, None)
    if previous != requested:
        raise ValueError("Interface fit may add only the declared query-control options.")
    gamma = [(n, p) for n, p in target_model.named_parameters() if n.startswith(FIT_PARAMETER_PREFIX + "gamma.")]
    if not gamma or any(bool(torch.count_nonzero(p)) for _, p in gamma):
        raise ValueError("Interface fit requires zero final gamma heads at attachment.")
    missing, unexpected = target_model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=False)
    if unexpected or any(not n.startswith(FIT_PARAMETER_PREFIX) for n in missing):
        raise ValueError(f"Frozen backbone attachment changed inherited state: missing={missing}, unexpected={unexpected}")
    new_names = freeze_interface_fit_model(target_model)
    source_state = strip_module_prefix(checkpoint["model_state_dict"])
    if any(not torch.equal(v, target_model.state_dict()[n]) for n, v in source_state.items()):
        raise ValueError("Frozen inherited weights or buffers were not copied exactly.")
    for key in ("dataset", "loss"):
        if fit_config[key] != checkpoint["train_config"][key]:
            raise ValueError(f"Interface fit changed inherited {key}.")
    optimizer, inventory = build_forward_optimizer(target_model, fit_config["training"])
    assert not optimizer.state
    receipt = {
        "kind": FIT_KIND, "source_checkpoint": str(source_path), "source_checkpoint_sha256": source_sha256,
        "inherited_backbone_epoch": 1000, "fit_epoch": 0, "fit_schedule_total_epochs": 1000,
        "new_parameter_names": new_names, "frozen_inherited_state_names": sorted(source_state),
        "optimizer_moments": "empty; inherited moments intentionally not allocated",
        "learning_rate": 3e-4, "weight_decay": 1e-5, "lr_scheduler": "none",
    }
    child = copy.deepcopy(checkpoint)
    child.update(epoch=0, current_epoch=0, selection_state={"epoch": 0, "total_epochs": 1000},
                 model_config=target_config.to_dict(), model_state_dict=target_model.state_dict(),
                 train_config=copy.deepcopy(fit_config), optimizer_state_dict=optimizer.state_dict(),
                 optimizer_group_inventory=inventory, scaler_state_dict=None, best_metric=float("inf"), best_metrics={})
    state = child.setdefault("campaign_training_state", {})
    if "matched_continuation_attachment" in state:
        state["frozen_backbone_previous_attachment"] = state.pop("matched_continuation_attachment")
    state["interface_fit_attachment"] = receipt
    target_model.campaign_training_state = copy.deepcopy(state)
    target_model.set_training_progress(epoch=0, total_epochs=1000)
    return child, receipt


def validate_interface_fit_resume(checkpoint, model, campaign) -> None:
    receipt = checkpoint.get("campaign_training_state", {}).get("interface_fit_attachment") or {}
    names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    if (receipt.get("kind") != FIT_KIND or receipt.get("inherited_backbone_epoch") != 1000
            or receipt.get("new_parameter_names") != names
            or any(not n.startswith(FIT_PARAMETER_PREFIX) for n in names)
            or campaign.get("parent", {}).get("checkpoint") != receipt.get("source_checkpoint")):
        raise ValueError("Interface-fit resume lacks its exact frozen-backbone/trainable-name lineage.")
