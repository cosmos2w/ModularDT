"""Exact C-soft e500 attachment and named-state checks for Round5 children."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .moment_hyperedge_protocol import (
    MOMENT_ATTACHMENT_SCHEMA,
    MOMENT_FAMILY,
    MOMENT_FINAL_EPOCH,
    MOMENT_PARENT_EPOCH,
    MOMENT_PROTOCOL,
    MOMENT_SELECTION_INTERVAL,
    MOMENT_WORKFLOW,
    canonical_sha256,
    file_sha256,
    parent_recipe_from_moment_hyperedges,
    validate_moment_hyperedge_recipe,
)


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(array.dtype.str.encode())
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
            digest.update(array.view(np.uint8).tobytes())
        elif isinstance(item, np.generic):
            add(item.item())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda row: str(row)):
                add(str(key))
                add(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(b"tuple\0" if isinstance(item, tuple) else b"list\0")
            digest.update(str(len(item)).encode())
            for child in item:
                add(child)
        elif item is None or type(item) in (bool, int, float, str):
            digest.update(type(item).__name__.encode())
            digest.update(json.dumps(item, allow_nan=False, separators=(",", ":")).encode())
        elif isinstance(item, bytes):
            digest.update(b"bytes\0")
            digest.update(item)
        else:
            raise TypeError(f"Cannot canonically hash state value {type(item).__name__}.")

    add(value)
    return digest.hexdigest()


def _history_sha256(history: Sequence[Mapping[str, Any]]) -> str:
    rows = copy.deepcopy(list(history))
    return canonical_sha256(rows)


def _check_parent_payload(
    source_payload: Mapping[str, Any], recipe: Mapping[str, Any], source_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]:
    value = validate_moment_hyperedge_recipe(recipe)
    path = source_path.expanduser().resolve(strict=True)
    if str(path) != value["source_checkpoint"]:
        raise ValueError("Moment-hyperedge attachment path differs from the pinned C-soft e500 recipe.")
    actual_sha = file_sha256(path)
    if actual_sha != value["source_checkpoint_sha256"]:
        raise ValueError("Moment-hyperedge parent checkpoint differs from its pinned SHA-256.")
    parent_identity = source_payload.get("experiment_identity")
    if not isinstance(parent_identity, Mapping):
        raise TypeError("C-soft e500 checkpoint has no experiment identity.")
    parent_recipe = parent_identity.get("recipe")
    if not isinstance(parent_recipe, Mapping):
        raise TypeError("C-soft e500 checkpoint has no native recipe.")
    if (source_payload.get("workflow") != "inherited_attention_joint_v1"
            or parent_identity.get("workflow") != "inherited_attention_joint_v1"
            or parent_identity.get("model_family") != "inherited_attention_v1"
            or source_payload.get("arm") != "IAG-C"
            or type(source_payload.get("epoch")) is not int
            or source_payload.get("epoch") != MOMENT_PARENT_EPOCH
            or parent_recipe.get("dataset") != value["dataset"]
            or parent_recipe.get("architecture") != "IAG-C"
            or parent_recipe.get("protocol") != "inherited_attention_development_v1"
            or parent_recipe.get("normalization") != "soft"
            or parent_recipe.get("trajectory") != "fresh_soft"):
        raise ValueError("Moment-hyperedge source is not its same-dataset own IAG-C soft e500 checkpoint.")
    if canonical_sha256(parent_recipe) != value["source_recipe_sha256"]:
        raise ValueError("Moment-hyperedge parent recipe digest differs from the pinned e500 recipe.")
    if parent_recipe_from_moment_hyperedges(value) != dict(parent_recipe):
        raise ValueError("Round5 recipe changed a native parent field outside the enumerated core/executor amendment.")
    rows = source_payload.get("history")
    if (not isinstance(rows, list) or len(rows) != MOMENT_PARENT_EPOCH
            or any(not isinstance(row, Mapping) for row in rows)
            or [row.get("epoch") for row in rows] != list(range(1, MOMENT_PARENT_EPOCH + 1))
            or any(row.get("arm") != "IAG-C" for row in rows)):
        raise ValueError("Moment-hyperedge e500 parent must contain its exact history rows 1 through 500.")
    rng = source_payload.get("rng_state")
    if (not isinstance(rng, Mapping) or not {"python", "numpy", "torch_cpu"}.issubset(rng)
            or not torch.is_tensor(rng.get("torch_cpu"))):
        raise ValueError("Moment-hyperedge e500 parent is missing exact Python/NumPy/Torch RNG state.")
    sampler = source_payload.get("sampler_state")
    if (not isinstance(sampler, Mapping) or sampler.get("completed_epoch") != MOMENT_PARENT_EPOCH
            or sampler.get("next_epoch") != MOMENT_PARENT_EPOCH + 1
            or sampler.get("next_case_cursor") != 0
            or sampler.get("seed") != value.get("sampling_seed")):
        raise ValueError("Moment-hyperedge e500 parent sampler is not at the exact epoch boundary.")
    provider_state = source_payload.get("provider_training_state")
    if not isinstance(provider_state, Mapping):
        raise ValueError("Moment-hyperedge e500 parent is missing its native provider training state.")
    parent_recipe_hash = parent_identity.get("recipe_sha256")
    if parent_recipe_hash != value["source_recipe_sha256"]:
        raise ValueError("Moment-hyperedge parent identity recipe hash differs from its recipe content.")
    return parent_identity, parent_recipe, rows


def _copy_common_state_by_name(
    model: torch.nn.Module, source_state: Mapping[str, Any],
) -> tuple[list[str], str, str]:
    target_state = model.state_dict()
    missing = sorted(set(source_state) - set(target_state))
    if missing:
        raise ValueError(f"Moment-hyperedge core dropped common parent state names: {missing}.")
    named_parameters = dict(model.named_parameters())
    born_state_names = sorted(set(target_state) - set(source_state))
    born_parameter_names = sorted(name for name in born_state_names if name in named_parameters)
    if set(born_state_names) != set(born_parameter_names):
        raise ValueError("Moment-hyperedge core introduced unaccounted non-parameter state at attachment.")
    for name, source in source_state.items():
        target = target_state[name]
        if (not torch.is_tensor(source) or not torch.is_tensor(target)
                or source.shape != target.shape or source.dtype != target.dtype):
            raise ValueError(f"Moment-hyperedge common state layout changed for {name!r}.")
        with torch.no_grad():
            target.copy_(source.to(device=target.device))
        if not torch.equal(target.detach().cpu(), source.detach().cpu()):
            raise ValueError(f"Moment-hyperedge parent tensor did not attach exactly for {name!r}.")
    common_digest = _tree_sha256({name: target_state[name] for name in sorted(source_state)})
    born_digest = _tree_sha256({name: target_state[name] for name in born_parameter_names})
    if any(not torch.isfinite(named_parameters[name]).all().item() for name in born_parameter_names):
        raise ValueError("Moment-hyperedge born parameters contain nonfinite values.")
    if any(not named_parameters[name].requires_grad for name in born_parameter_names):
        raise ValueError("Every born moment-hyperedge parameter must train from its first update.")
    return born_parameter_names, common_digest, born_digest


def prepare_moment_hyperedge_child(
    model: torch.nn.Module,
    source_payload: Mapping[str, Any],
    sealed_identity: Mapping[str, Any],
    declaration: Mapping[str, Any],
    *,
    source_path: Path,
    case_keys: Sequence[Any],
    stop_after: int,
) -> dict[str, Any]:
    """Attach exact common C state and record the new parameter birth."""

    if not isinstance(declaration, Mapping):
        raise TypeError("Moment-hyperedge child declaration must be a mapping.")
    identity_recipe = sealed_identity.get("recipe")
    recipe = validate_moment_hyperedge_recipe(identity_recipe)
    expected_declaration = {
        "schema": MOMENT_ATTACHMENT_SCHEMA,
        "family": MOMENT_FAMILY,
        "workflow": MOMENT_WORKFLOW,
        "protocol": MOMENT_PROTOCOL,
        "variant": recipe["variant"],
        "source_checkpoint": recipe["source_checkpoint"],
        "source_checkpoint_sha256": recipe["source_checkpoint_sha256"],
        "source_recipe_sha256": recipe["source_recipe_sha256"],
        "source_epoch": MOMENT_PARENT_EPOCH,
    }
    if any(declaration.get(name) != expected for name, expected in expected_declaration.items()):
        raise ValueError("Moment-hyperedge attachment declaration differs from its sealed recipe.")
    parent_identity, _parent_recipe, history = _check_parent_payload(
        source_payload, recipe, source_path,
    )
    if stop_after < MOMENT_PARENT_EPOCH + 1 or stop_after > MOMENT_FINAL_EPOCH:
        raise ValueError("Moment-hyperedge branch stop must advance its e500 parent within absolute e2500.")
    source_state = source_payload.get("model_state_dict")
    if not isinstance(source_state, Mapping):
        raise TypeError("Moment-hyperedge e500 parent has no model state dictionary.")
    parent_optimizer = source_payload.get("optimizer_state_by_name")
    if not isinstance(parent_optimizer, Mapping):
        raise TypeError("Moment-hyperedge e500 parent has no named optimizer state.")
    optimizer_states = parent_optimizer.get("state_by_name")
    if not isinstance(optimizer_states, Mapping) or not optimizer_states:
        raise ValueError("Moment-hyperedge parent named AdamW state is empty or malformed.")
    schedule = parent_identity.get("optimizer_schedule_contract")
    if not isinstance(schedule, list):
        raise ValueError("Moment-hyperedge parent identity has no optimizer schedule contract.")
    scheduled_names = [name for group in schedule for name in group.get("parameter_names", ())]
    if len(scheduled_names) != len(set(scheduled_names)) or set(scheduled_names) != set(optimizer_states):
        raise ValueError("Moment-hyperedge parent AdamW state does not cover its exact named trainable parameters.")

    born_names, common_state_sha, born_state_sha = _copy_common_state_by_name(model, source_state)
    source_model_names = set(source_state)
    named_parameters = dict(model.named_parameters())
    common_parameter_names = sorted(source_model_names & set(named_parameters))
    if set(optimizer_states) != set(common_parameter_names):
        raise ValueError("Moment-hyperedge parent optimizer names differ from the child's common parameter names.")
    if set(born_names) - set(named_parameters):
        raise ValueError("Moment-hyperedge birth inventory contains a non-parameter tensor.")
    provider_identity = sealed_identity.get("provider_identity")
    parent_provider_identity = parent_identity.get("provider_identity")
    if not isinstance(provider_identity, Mapping) or not isinstance(parent_provider_identity, Mapping):
        raise TypeError("Moment-hyperedge branch must seal both old and new native provider identities.")
    from .moment_hyperedge_factory import validate_moment_provider_pair

    provider_binding = validate_moment_provider_pair(
        parent_provider_identity,
        provider_identity,
        dataset=recipe["dataset"],
        recipe=recipe,
    )
    parent_schedule = parent_identity.get("optimizer_schedule_contract")
    child_schedule = sealed_identity.get("optimizer_schedule_contract")
    if (not isinstance(parent_schedule, list) or not isinstance(child_schedule, list)
            or len(parent_schedule) != len(child_schedule)):
        raise ValueError("Moment-hyperedge continuation must retain its exact AdamW schedule groups.")
    for old_group, new_group in zip(parent_schedule, child_schedule):
        if not isinstance(old_group, Mapping) or not isinstance(new_group, Mapping):
            raise TypeError("Moment-hyperedge optimizer schedule groups must be mappings.")
        if any(old_group.get(key) != new_group.get(key) for key in ("name", "schedule", "weight_decay", "betas", "eps")):
            raise ValueError("Moment-hyperedge continuation changed an AdamW hyperparameter or schedule.")
        old_names = list(old_group.get("parameter_names", ()))
        new_names = list(new_group.get("parameter_names", ()))
        if ([name for name in new_names if name in set(old_names)] != old_names
                or set(new_names) - set(old_names) != set(born_names)):
            raise ValueError("Moment-hyperedge schedule membership is not the exact parent-plus-born parameter set.")
    if (parent_identity.get("sampling_dataset_id") != sealed_identity.get("sampling_dataset_id")
            or parent_identity.get("sampling_identity") != sealed_identity.get("sampling_identity")):
        raise ValueError("Moment-hyperedge attachment changed the native case/query sampling identity.")
    if not case_keys:
        raise ValueError("Moment-hyperedge attachment requires the exact next-epoch native case sequence.")
    return {
        **expected_declaration,
        "parent_identity_sha256": canonical_sha256(parent_identity),
        "parent_provider_identity_sha256": canonical_sha256(parent_provider_identity),
        "child_provider_identity_sha256": canonical_sha256(provider_identity),
        "native_provider_binding": provider_binding,
        "parent_history_rows": len(history),
        "parent_history_sha256": _history_sha256(history),
        "parent_rng_state_sha256": _tree_sha256(source_payload["rng_state"]),
        "parent_sampler_state_sha256": _tree_sha256(source_payload["sampler_state"]),
        "parent_provider_state_sha256": _tree_sha256(source_payload["provider_training_state"]),
        "common_model_state_sha256": common_state_sha,
        "common_model_state_names_sha256": canonical_sha256(sorted(source_state)),
        "common_parameter_names_sha256": canonical_sha256(common_parameter_names),
        "born_parameter_names": born_names,
        "born_parameter_initialization_sha256": born_state_sha,
        "next_epoch_case_keys_sha256": canonical_sha256([str(key) for key in case_keys]),
        "preserve_rng_sampler_and_provider_state": True,
        "history_prefix_preserved": True,
        "selection_policy": {
            "metric": recipe["execution"]["field_metric"],
            "eligible_epochs": [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
                                1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500],
            "tie_break": "earliest_absolute_epoch",
        },
        "new_optimizer_states_at_attachment": 0,
        "exact_common_parent_state_loaded_by_name": True,
        "stop_after": int(stop_after),
        "attachment_effective_epoch": MOMENT_PARENT_EPOCH + 1,
    }


def _optimizer_values_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    return left == right


def verify_moment_hyperedge_child_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source_optimizer_payload: Mapping[str, Any],
    attachment: Mapping[str, Any],
) -> None:
    """Require exact common named AdamW moments and no state for born tensors."""

    from .unified_training import named_optimizer_state

    restored = named_optimizer_state(model, optimizer)
    parent_groups = source_optimizer_payload.get("groups")
    current_groups = restored.get("groups")
    if not isinstance(parent_groups, list) or not isinstance(current_groups, list):
        raise ValueError("Moment-hyperedge parent and child optimizer groups must be named lists.")
    if len(parent_groups) != len(current_groups):
        raise ValueError("Moment-hyperedge continuation changed the number of native AdamW groups.")
    for old, new in zip(parent_groups, current_groups):
        if not isinstance(old, Mapping) or not isinstance(new, Mapping) or old.get("name") != new.get("name"):
            raise ValueError("Moment-hyperedge continuation changed a parent AdamW group name/order.")
        old_names = list(old.get("parameter_names", ()))
        new_names = list(new.get("parameter_names", ()))
        if [name for name in new_names if name in set(old_names)] != old_names:
            raise ValueError("Moment-hyperedge continuation changed common parameter order inside AdamW.")
        if set(new_names) - set(old_names) != set(attachment.get("born_parameter_names", ())):
            raise ValueError("Moment-hyperedge AdamW group contains an unsealed born-parameter set.")
    expected_states = source_optimizer_payload.get("state_by_name")
    observed_states = restored.get("state_by_name")
    if not isinstance(expected_states, Mapping) or not isinstance(observed_states, Mapping):
        raise ValueError("Moment-hyperedge AdamW named state is malformed.")
    if set(observed_states) != set(expected_states):
        raise ValueError("Born moment-hyperedge parameters must have absent AdamW state until their first update.")
    for name in expected_states:
        expected = expected_states[name]
        observed = observed_states[name]
        if not isinstance(expected, Mapping) or not isinstance(observed, Mapping) or set(expected) != set(observed):
            raise ValueError(f"Moment-hyperedge common AdamW state fields changed for {name!r}.")
        for key in expected:
            if not _optimizer_values_equal(expected[key], observed[key]):
                raise ValueError(f"Moment-hyperedge common AdamW state changed for {name}.{key}.")


def verify_moment_hyperedge_resume(
    source_payload: Mapping[str, Any],
    recipe: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate an exact Round5 latest-state resume and its inherited prefix."""

    value = validate_moment_hyperedge_recipe(recipe)
    identity = source_payload.get("experiment_identity")
    parent = source_payload.get("branch_parent")
    if (not isinstance(identity, Mapping) or identity.get("model_family") != MOMENT_FAMILY
            or identity.get("workflow") != MOMENT_WORKFLOW
            or identity.get("protocol") != MOMENT_PROTOCOL
            or identity.get("recipe") != value
            or identity.get("recipe_sha256") != canonical_sha256(value)
            or source_payload.get("workflow") != MOMENT_WORKFLOW
            or source_payload.get("arm") != "IAG-C"
            or type(source_payload.get("epoch")) is not int
            or not MOMENT_PARENT_EPOCH < source_payload["epoch"] <= MOMENT_FINAL_EPOCH
            or not isinstance(parent, Mapping)):
        raise ValueError("Round5 exact resume checkpoint has an incompatible workflow, recipe, arm, or age.")
    attachment = parent.get("moment_hyperedge_child")
    if not isinstance(attachment, Mapping):
        raise ValueError("Round5 exact resume checkpoint has no C-soft e500 attachment receipt.")
    if (parent.get("path") != value["source_checkpoint"]
            or parent.get("sha256") != value["source_checkpoint_sha256"]
            or parent.get("epoch") != MOMENT_PARENT_EPOCH
            or not isinstance(parent.get("identity"), Mapping)
            or canonical_sha256(parent["identity"]) != attachment.get("parent_identity_sha256")):
        raise ValueError("Round5 exact resume changed its pinned C-soft e500 branch identity.")
    for key in ("schema", "family", "workflow", "protocol", "variant", "source_checkpoint",
                "source_checkpoint_sha256", "source_recipe_sha256", "source_epoch"):
        if attachment.get(key) != declaration.get(key):
            raise ValueError(f"Round5 exact resume changed attachment field {key!r}.")
    history = source_payload.get("history")
    if (not isinstance(history, list) or len(history) != source_payload["epoch"]
            or [row.get("epoch") for row in history if isinstance(row, Mapping)]
            != list(range(1, source_payload["epoch"] + 1))):
        raise ValueError("Round5 exact resume history is not the complete 1..saved-age prefix.")
    if (attachment.get("parent_history_rows") != MOMENT_PARENT_EPOCH
            or _history_sha256(history[:MOMENT_PARENT_EPOCH]) != attachment.get("parent_history_sha256")):
        raise ValueError("Round5 exact resume lost the full 500-row inherited parent history.")
    return attachment


__all__ = [
    "prepare_moment_hyperedge_child",
    "verify_moment_hyperedge_child_optimizer",
    "verify_moment_hyperedge_resume",
]
