"""Exact Round4 soft-e500 to trained-sparse IAG-C child guard."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

IAG_SPARSE_CHILD_SCHEMA = "inherited_attention_sparse_child_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _soft_identity(value: Any) -> Any:
    """Erase only the sealed normalization/trajectory identity difference."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key == "sparse_transition":
                continue
            if key == "recipe_sha256":
                result[key] = "<normalization-child-recipe>"
            elif key == "normalization":
                result[key] = "soft"
            elif key in ("trajectory",):
                result[key] = "fresh_soft"
            elif key == "absolute_epoch_transition":
                result[key] = None
            elif key == "scene_executor":
                # The only executor change allowed at attachment is the
                # mathematically identical dense -> selected contraction.
                # _require_same_soft_identity validates direction and values
                # before this comparison-only normalization is applied.
                result[key] = "dense"
            else:
                result[key] = _soft_identity(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_soft_identity(item) for item in value]
    return copy.deepcopy(value)


def _identity_values(value: Any, key_name: str) -> list[Any]:
    if isinstance(value, Mapping):
        found = [item for key, item in value.items() if key == key_name]
        for item in value.values():
            found.extend(_identity_values(item, key_name))
        return found
    if isinstance(value, (tuple, list)):
        found = []
        for item in value:
            found.extend(_identity_values(item, key_name))
        return found
    return []


def _require_same_soft_identity(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, str]:
    source_scene = _identity_values(source, "scene_executor")
    target_scene = _identity_values(target, "scene_executor")
    if not source_scene or not target_scene or any(value not in ("dense", "selected") for value in source_scene + target_scene):
        raise ValueError("IAG-C sparse-child scene executor identity is missing or unsupported.")
    if len(set(source_scene)) != 1 or len(set(target_scene)) != 1:
        raise ValueError("IAG-C sparse-child identity has conflicting scene executor declarations.")
    source_scene_executor = source_scene[0]
    target_scene_executor = target_scene[0]
    if source_scene_executor == "selected" and target_scene_executor == "dense":
        raise ValueError("IAG-C e500 attachment cannot replace selected preparation with the dense executor.")
    source_query = _identity_values(source, "query_executor")
    target_query = _identity_values(target, "query_executor")
    if not source_query or not target_query or set(source_query) != {"indexed"} or set(target_query) != {"indexed"}:
        raise ValueError("IAG-C e500 attachment preserves the indexed query executor exactly.")
    left = _soft_identity(source)
    right = _soft_identity(target)
    if left != right:
        raise ValueError("IAG-C sparse child changes a field beyond normalization and its absolute transition.")
    return {
        "source": str(source_scene_executor),
        "target": str(target_scene_executor),
        "same_probability_function": True,
    }


def _require_state_layout(model: torch.nn.Module, source_state: Any) -> None:
    current = model.state_dict()
    if not isinstance(source_state, Mapping) or set(current) != set(source_state):
        raise ValueError("IAG-C e500 parent and sparse child must have exactly identical state-dict keys.")
    for name, tensor in current.items():
        parent = source_state[name]
        if not torch.is_tensor(parent) or parent.shape != tensor.shape or parent.dtype != tensor.dtype:
            raise ValueError(f"IAG-C e500 state tensor {name!r} differs in shape or dtype.")
    current_parameters = dict(model.named_parameters())
    parent_parameters = {
        name for name in current_parameters if name in source_state
    }
    if parent_parameters != set(current_parameters):
        raise ValueError("IAG-C child must preserve every parameter name from its soft e500 parent.")


def _optimizer_groups_from_identity(identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = identity.get("optimizer_schedule_contract")
    if not isinstance(contract, list):
        raise TypeError("IAG-C parent identity has no sealed optimizer schedule contract.")
    return [
        {"name": str(item["name"]), "parameter_names": list(item["parameter_names"])}
        for item in contract
    ]


def prepare_inherited_attention_sparse_child(
    model: torch.nn.Module,
    source_payload: Mapping[str, Any],
    sealed_identity: Mapping[str, Any],
    declaration: Mapping[str, Any],
    *,
    source_path: Path,
    case_keys: Sequence[Any],
    stop_after: int,
) -> dict[str, Any]:
    """Validate one literal soft IAG-C e500 parent before strict state loading."""
    required = {
        "schema",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_recipe_sha256",
        "target_recipe_sha256",
        "source_epoch",
    }
    if not isinstance(declaration, Mapping) or set(declaration) != required:
        raise ValueError("IAG-C sparse child declaration fields differ from the frozen e500 attachment contract.")
    source_path = source_path.expanduser().resolve(strict=True)
    digest = _sha256(source_path)
    if declaration.get("schema") != IAG_SPARSE_CHILD_SCHEMA:
        raise ValueError("Unsupported IAG-C sparse child declaration schema.")
    if declaration.get("source_checkpoint") != str(source_path):
        raise ValueError("IAG-C declaration must name the exact canonical e500 checkpoint path.")
    if declaration.get("source_checkpoint_sha256") != digest:
        raise ValueError("IAG-C e500 checkpoint checksum differs from the child declaration.")
    if type(declaration.get("source_epoch")) is not int or declaration["source_epoch"] != 500:
        raise ValueError("IAG-C sparse child attaches only to absolute epoch500.")
    if not 501 <= int(stop_after) <= 2500:
        raise ValueError("IAG-C sparse child continuation must stop within absolute epochs501..2500.")
    if source_payload.get("epoch") != 500 or source_payload.get("arm") != "IAG-C":
        raise ValueError("IAG-C sparse child parent checkpoint must be the literal IAG-C epoch500 state.")
    if source_payload.get("workflow") != "inherited_attention_joint_v1":
        raise ValueError("IAG-C sparse child cannot attach from a historical or other-family checkpoint.")
    parent_identity = source_payload.get("experiment_identity")
    if not isinstance(parent_identity, Mapping):
        raise TypeError("IAG-C e500 parent checkpoint has no sealed experiment identity.")
    parent_recipe = parent_identity.get("recipe")
    child_recipe = sealed_identity.get("recipe")
    if not isinstance(parent_recipe, Mapping) or not isinstance(child_recipe, Mapping):
        raise TypeError("IAG-C parent and child identities must both embed their recipes.")
    if (parent_recipe.get("protocol") != "inherited_attention_development_v1"
            or parent_recipe.get("normalization") != "soft"
            or parent_recipe.get("trajectory") != "fresh_soft"):
        raise ValueError("Sparse IAG-C parent must be its own fresh soft segmented trajectory.")
    if (child_recipe.get("protocol") != "inherited_attention_development_v1"
            or child_recipe.get("normalization") != "measure_entmax15"
            or child_recipe.get("trajectory") != "e500_sparse_child"):
        raise ValueError("Sparse IAG-C child must use its sealed segmented measure-entmax trajectory.")
    if declaration.get("source_recipe_sha256") != parent_identity.get("recipe_sha256"):
        raise ValueError("IAG-C declaration does not bind the parent's exact soft recipe digest.")
    if declaration.get("target_recipe_sha256") != sealed_identity.get("recipe_sha256"):
        raise ValueError("IAG-C declaration does not bind the child's exact sparse recipe digest.")
    executor_transition = _require_same_soft_identity(parent_identity, sealed_identity)
    _require_state_layout(model, source_payload.get("model_state_dict"))
    source_optimizer = source_payload.get("optimizer_state_by_name")
    if not isinstance(source_optimizer, Mapping) or not isinstance(source_optimizer.get("state_by_name"), Mapping):
        raise TypeError("IAG-C sparse child requires named AdamW moments from its e500 parent.")
    parent_identity_groups = _optimizer_groups_from_identity(parent_identity)
    child_identity_groups = _optimizer_groups_from_identity(sealed_identity)
    if parent_identity_groups != child_identity_groups:
        raise ValueError("IAG-C sparse child optimizer group names or parameter membership changed.")
    if source_optimizer.get("groups") != parent_identity_groups:
        raise ValueError("IAG-C checkpoint AdamW group membership differs from its own sealed identity.")
    expected_names = {name for name, _ in model.named_parameters()}
    state_by_name = source_optimizer["state_by_name"]
    if set(state_by_name) != expected_names:
        raise ValueError("IAG-C sparse child must inherit one AdamW state for every original trainable parameter.")
    if source_payload.get("provider_training_state", {}) not in ({}, None):
        raise ValueError("IAG-C sparse child does not accept undeclared provider-local training state.")
    history = source_payload.get("history")
    if not isinstance(history, list) or not history or int(history[-1].get("epoch", -1)) != 500:
        raise ValueError("IAG-C parent history must end at the literal epoch500 review checkpoint.")
    if not case_keys:
        raise ValueError("IAG-C sparse child cannot attach without its fixed TRAIN membership.")
    return {
        "schema": IAG_SPARSE_CHILD_SCHEMA,
        "trajectory": "soft_e1_500_then_absolute_sparse_transition_e501_800",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": digest,
        "source_recipe_sha256": declaration["source_recipe_sha256"],
        "target_recipe_sha256": declaration["target_recipe_sha256"],
        "source_epoch": 500,
        "next_epoch": 501,
        "stop_after": int(stop_after),
        "train_case_count": len(case_keys),
        "exact_parameter_layout_verified": True,
        "exact_optimizer_group_contract_verified": True,
        "exact_optimizer_moments_pending_load_verification": True,
        "scene_executor_transition": executor_transition,
        "provider_state_contract": "empty; native sampler age is carried by absolute epoch and restored RNG",
    }


def verify_inherited_attention_sparse_child_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source_optimizer_payload: Mapping[str, Any],
) -> None:
    """Verify all moments and parameter memberships survived named restore."""
    from .unified_training import named_optimizer_state

    restored = named_optimizer_state(model, optimizer)
    expected = source_optimizer_payload
    if restored.get("groups") != expected.get("groups"):
        raise ValueError("IAG-C sparse child restored optimizer groups differ from its exact e500 parent.")
    left = restored.get("state_by_name", {})
    right = expected.get("state_by_name", {})
    if set(left) != set(right):
        raise ValueError("IAG-C sparse child restored AdamW parameter states differ from its exact e500 parent.")
    for name in left:
        if set(left[name]) != set(right[name]):
            raise ValueError(f"IAG-C e500 optimizer state fields changed for {name!r}.")
        for key in left[name]:
            actual, saved = left[name][key], right[name][key]
            if torch.is_tensor(actual) or torch.is_tensor(saved):
                if not (torch.is_tensor(actual) and torch.is_tensor(saved) and torch.equal(actual, saved)):
                    raise ValueError(f"IAG-C e500 optimizer tensor changed for {name}.{key}.")
            elif actual != saved:
                raise ValueError(f"IAG-C e500 optimizer scalar changed for {name}.{key}.")


def verify_inherited_attention_sparse_child_continuation(
    contract: Mapping[str, Any],
    *,
    completed_epoch: int,
    stop_after: int,
) -> None:
    if contract.get("schema") != IAG_SPARSE_CHILD_SCHEMA or contract.get("source_epoch") != 500:
        raise ValueError("IAG-C sparse resume is missing its exact e500 child branch contract.")
    if contract.get("trajectory") != "soft_e1_500_then_absolute_sparse_transition_e501_800":
        raise ValueError("IAG-C sparse resume has a different normalization age contract.")
    if type(completed_epoch) is not int or not 501 <= completed_epoch < stop_after <= 2500:
        raise ValueError("IAG-C sparse same-run resume must advance inside the absolute 501..2500 horizon.")


__all__ = [
    "IAG_SPARSE_CHILD_SCHEMA",
    "prepare_inherited_attention_sparse_child",
    "verify_inherited_attention_sparse_child_continuation",
    "verify_inherited_attention_sparse_child_optimizer",
]
