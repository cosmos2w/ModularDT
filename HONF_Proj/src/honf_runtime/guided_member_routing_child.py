"""State-preserving e500 transfer for the relation-guided reader.

This is a model/objective branch attachment in the maintained training engine.
It does not write a converted copy of the parent checkpoint or alter its run.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


WORKFLOW = "interaction_reused_member_routing_joint_training_v3"


def same_state(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping):
        return isinstance(right, Mapping) and left.keys() == right.keys() and all(
            same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            same_state(a, b) for a, b in zip(left, right))
    return left == right


def prepare_guided_transfer(model, provider, source, current, declaration, *, source_path):
    """Validate the literal parent and preserve compatible learned/training state."""
    if (not isinstance(declaration, Mapping) or declaration.get("source_epoch") != 500
            or Path(declaration["source_checkpoint"]).resolve() != source_path.resolve()
            or source.get("epoch") != 500 or source.get("arm") != "R13-3-Relative"
            or source.get("workflow") != "interaction_reused_member_routing_joint_training_v2"
            or current.get("workflow") != WORKFLOW
            or current.get("training_membership") != "full_train600"
            or model.architecture not in ("R13-4-Full", "R13-4-Guide")):
        raise ValueError("Guided transfer requires the retained Relative e500 and a distinct full-TRAIN600 arm.")
    previous = source["experiment_identity"]
    old_provider, new_provider = previous["provider_identity"], current["provider_identity"]
    for name in ("training_case_ids", "validation_case_ids"):
        if old_provider[name] != new_provider[name]:
            raise ValueError(f"Guided transfer changed {name}.")
    # The versioned provider identity changes for this attachment, so retain
    # its existing dataset/manifest binding explicitly alongside membership.
    if (not isinstance(old_provider.get("source_binding"), Mapping)
            or not same_state(old_provider["source_binding"], new_provider.get("source_binding"))):
        raise ValueError("Guided transfer changed the existing dataset/manifest source binding.")
    for name in ("normalization_stats", "heat_scale", "native_grid_nx_ny", "environment_grid_nx_ny"):
        if not same_state(old_provider["adapter_state"][name], new_provider["adapter_state"][name]):
            raise ValueError(f"Guided transfer changed original TRAIN normalization/adapter field {name}.")
    for name in ("base_model_config", "input_dataset", "input_capability", "entmax_temperature", "query_tile_size"):
        if old_provider["adapter_state"]["core"][name] != new_provider["adapter_state"]["core"][name]:
            raise ValueError(f"Guided transfer changed contextual architecture or precision field {name}.")
    if (previous["optimizer"] != current["optimizer"]
            or previous["optimizer_schedule_contract"] != current["optimizer_schedule_contract"]
            or previous["sampling_dataset_id"] != current["sampling_dataset_id"]
            or previous["engine_config"]["seed"] != current["engine_config"]["seed"]
            or previous["engine_config"]["effective_cases"] != current["engine_config"]["effective_cases"]
            or previous["engine_config"]["microbatch_cases"] != current["engine_config"]["microbatch_cases"]):
        raise ValueError("Guided transfer changed optimizer, schedule or native sampling/packing.")
    state = model.state_dict()
    if state.keys() != source["model_state_dict"].keys() or any(
            value.shape != source["model_state_dict"][name].shape for name, value in state.items()):
        raise ValueError("The initial guided reader must preserve all parent parameter names/shapes.")
    return {
        "kind": "relation_guided_e500_model_objective_branch",
        "source_checkpoint": str(source_path), "source_epoch": 500,
        "source_arm": source["arm"], "target_arm": model.architecture,
        "inherited_state": "all model parameters, named optimizer moments, RNG, sampler, TRAIN normalization and provider numerical clocks",
        "new_parameter_names": [],
        "changes": "relation-weighted member calibration, stable eligible-count entropy; Guide adds early E coverage and paired physical losses",
        "health_comparison_origin": "reset for the changed reader at parent epoch500",
        "native_physical_objective_weights": "unchanged; Guide averages full/restricted losses",
    }
