"""Explicit e500 development attachment without weakening ordinary resume checks."""

from __future__ import annotations

import copy
from typing import Any

import torch

from channelthermal.training.optimizer import build_forward_optimizer
from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.compat import strip_module_prefix

MANIFEST_SHA256 = "933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044"
NEW_PARAMETER_PREFIX = "core.backend.tensor_residual."
PARENT_SHA256 = "1a8ac8d976edc72e55950eb3404145adc4ca124e7a529010e02e1a882c5ff051"


def validate_lean_parent(checkpoint: dict[str, Any]) -> None:
    """Accept only the declared exact development Global-C parent contract."""
    validate_checkpoint_identity(checkpoint, case_id="ThermalChannel", model_family="honf_forward", workflow="forward")
    config = checkpoint.get("train_config", {})
    dataset = config.get("dataset", {})
    campaign = config.get("training", {}).get("campaign", {})
    partitions = (dataset.get("development_subset") or {}).get("partitions", {})
    if (checkpoint.get("epoch") != 500
            or checkpoint.get("selection_state") != {"epoch": 500, "total_epochs": 1000}
            or checkpoint.get("model_config", {}).get("core_honf", {}).get("forward_architecture")
            != "native_context_global_control_honf"
            or dataset.get("development_manifest_sha256") != MANIFEST_SHA256
            or partitions.get("train", {}).get("selected_count") != 150
            or partitions.get("test", {}).get("selected_count") != 22
            or campaign.get("schedule_total_epochs") != 1000
            or campaign.get("physical_loss_policy_version") != 2
            or campaign.get("native_loss_denominators_start_epoch") != 101
            or campaign.get("organizer_gradient_policy") != "ordinary_task_v1"
            or campaign.get("heat_null_response") != {"coefficient": 0.0}
            or config.get("training", {}).get("amp") is not False
            or not checkpoint.get("optimizer_state_dict") or not checkpoint.get("rng_state")):
        raise ValueError("Lean attachment requires exact development Global-C e500 fixed25_v1 with optimizer/RNG and unchanged absolute1000 policy.")
    options = checkpoint["model_config"]["core_honf"]["interface_model"].get("hypergraph_options", {})
    if options.get("tensor_source_residual") or options.get("global_fast_reader"):
        raise ValueError("Attach from the original Global-C parent, not a previously attached child.")


def transplant_optimizer_by_name(source_model, target_model, saved_optimizer, target_optimizer) -> dict[str, Any]:
    """Copy AdamW state by parameter identity; added residuals start without moments.

    Native Global-C uses one historical group in named-parameter traversal order.
    Reject other layouts rather than guessing from the sorted display inventory.
    """
    groups = saved_optimizer.get("param_groups", [])
    if len(groups) != 1 or len(target_optimizer.param_groups) != 1:
        raise ValueError("Lean attachment requires the parent's single native AdamW group.")
    source = [(name, parameter) for name, parameter in source_model.named_parameters() if parameter.requires_grad]
    target = [(name, parameter) for name, parameter in target_model.named_parameters() if parameter.requires_grad]
    old_ids = groups[0]["params"]
    if len(source) != len(old_ids):
        raise ValueError("Parent parameter order/count cannot be reconstructed exactly.")
    source_named = dict(source)
    target_named = dict(target)
    if set(source_named) - set(target_named):
        raise ValueError("Attachment removed common trainable parameters.")
    added = set(target_named) - set(source_named)
    if any(not name.startswith(NEW_PARAMETER_PREFIX) for name in added):
        raise ValueError("Attachment added parameters outside the declared tensor residual.")
    for name, parameter in source:
        if target_named[name].shape != parameter.shape:
            raise ValueError(f"Attachment changed common parameter shape: {name}")
    source_ids = dict(zip((name for name, _ in source), old_ids, strict=True))
    target_state = target_optimizer.state_dict()
    new_ids = target_state["param_groups"][0]["params"]
    copied = {}
    for (name, parameter), new_id in zip(target, new_ids, strict=True):
        if name not in source_ids:
            continue
        old_state = saved_optimizer["state"].get(source_ids[name])
        if old_state is None:
            continue
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in old_state and old_state[key].shape != parameter.shape:
                raise ValueError(f"Optimizer moment shape mismatch: {name}.{key}")
        copied[new_id] = copy.deepcopy(old_state)
    group = copy.deepcopy(groups[0])
    group["params"] = new_ids
    target_optimizer.load_state_dict({"state": copied, "param_groups": [group]})
    return {"common_parameter_tensors": len(source), "restored_moment_tensors": len(copied),
            "common_parameter_tensors_without_moments": len(source) - len(copied),
            "common_parameter_names_without_moments": sorted(name for name in source_ids
                if source_ids[name] not in saved_optimizer["state"]),
            "new_parameter_names": sorted(added), "new_moments": "empty",
            "new_learning_rate": float(group["lr"]), "lr_scheduler": "none; inherited native constant learning rate",
            "absolute_schedule_total_epochs": 1000, "inherited_epochs": 500}


def attach_lean_checkpoint(checkpoint, *, source_model, target_model, target_config, source_sha256, source_path):
    """Return a child checkpoint with identical common work state and explicit lineage."""
    validate_lean_parent(checkpoint)
    previous = copy.deepcopy(source_model.config.to_dict())
    requested = copy.deepcopy(target_config.to_dict())
    previous_options = previous["core_honf"]["interface_model"].setdefault("hypergraph_options", {})
    requested_options = requested["core_honf"]["interface_model"].setdefault("hypergraph_options", {})
    allowed = {"global_fast_reader", "tensor_source_residual", "residual_parent_epoch"}
    for key in allowed:
        previous_options.pop(key, None)
        requested_options.pop(key, None)
    if previous != requested:
        raise ValueError("Lean attachment may change only its declared fast/residual options.")
    options = target_config.to_dict()["core_honf"]["interface_model"].get("hypergraph_options", {})
    if options.get("global_fast_reader") is not True or (
            options.get("tensor_source_residual") and options.get("residual_parent_epoch") != 500):
        raise ValueError("Lean child must use the fast reader and the residual's exact e500 parent age.")
    if options.get("tensor_source_residual"):
        gamma_parameters = [(name, parameter) for name, parameter in target_model.named_parameters()
                            if name.startswith(NEW_PARAMETER_PREFIX + "gamma.")]
        if not gamma_parameters or any(bool(torch.count_nonzero(parameter)) for _, parameter in gamma_parameters):
            raise ValueError("Tensor-H attachment requires exactly zero final gamma projections.")
    source_model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    missing, unexpected = target_model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=False)
    if unexpected or any(not name.startswith(NEW_PARAMETER_PREFIX) for name in missing):
        raise ValueError(f"Attachment state mismatch: missing={missing}, unexpected={unexpected}")
    optimizer, inventory = build_forward_optimizer(target_model, checkpoint["train_config"]["training"])
    receipt = transplant_optimizer_by_name(source_model, target_model, checkpoint["optimizer_state_dict"], optimizer)
    receipt.update(source_checkpoint=str(source_path), source_checkpoint_sha256=source_sha256,
                   kind="matched_development_e500_zero_output_attachment", additional_epoch=0)
    child = copy.deepcopy(checkpoint)
    child.update(model_config=target_config.to_dict(), model_state_dict=target_model.state_dict(),
                 optimizer_state_dict=optimizer.state_dict(), optimizer_group_inventory=inventory)
    child["train_config"]["model"] = target_config.to_dict()
    child.setdefault("campaign_training_state", {})["matched_continuation_attachment"] = receipt
    # Select the best among this child campaign's saved reviews, identically in
    # both arms. The inherited checkpoint is recorded, not a child selector copy.
    child["best_metric"] = float("inf")
    child["best_metrics"] = {}
    return child, receipt
