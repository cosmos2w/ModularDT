"""Exact e500 branch, named-optimizer and absolute-age checks for Round7."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .native_retrieval_protocol import (
    canonical_sha256,
    file_sha256,
    native_retrieval_child_declaration,
    validate_native_retrieval_recipe,
)


def _same_state_tensor(a: Any, b: Any) -> bool:
    return (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype
            and a.shape == b.shape and torch.equal(a.detach().cpu(), b.detach().cpu()))


def _provider_pair(parent: Mapping[str, Any], child: Mapping[str, Any], dataset: str) -> None:
    old = copy.deepcopy(dict(parent))
    new = copy.deepcopy(dict(child))
    if dataset == "thermal":
        for binding in ("model_config", "adapter_state"):
            old_binding = old.get(binding)
            new_binding = new.get(binding)
            if not isinstance(old_binding, Mapping) or not isinstance(new_binding, Mapping):
                raise TypeError("Thermal native provider binding is missing its adapter/core metadata.")
            old_core, new_core = old_binding.get("core"), new_binding.get("core")
            if not isinstance(old_core, Mapping) or not isinstance(new_core, Mapping):
                raise TypeError("Thermal native provider binding has no core config.")
            new_binding["core"] = copy.deepcopy(dict(old_core))
    elif dataset == "wind":
        old_config, new_config = old.get("model_config"), new.get("model_config")
        if not isinstance(old_config, Mapping) or not isinstance(new_config, Mapping):
            raise TypeError("Wind native provider binding is missing its core config.")
        new["model_config"] = copy.deepcopy(dict(old_config))
    else:
        raise ValueError("Native provider identity comparison supports only Thermal and Wind.")
    if canonical_sha256(old) != canonical_sha256(new):
        raise ValueError("Round7 provider changed data, transforms, sampling, native laws or objective beyond its core schema.")


def prepare_native_retrieval_child(
    model: torch.nn.Module,
    provider: Any,
    source_payload: Mapping[str, Any],
    sealed_identity: Mapping[str, Any],
    declaration: Mapping[str, Any],
    *,
    source_path: Path,
    stop_after: int,
) -> dict[str, Any]:
    recipe = validate_native_retrieval_recipe(sealed_identity.get("recipe", {}), verify_source_file=True)
    expected = native_retrieval_child_declaration(recipe)
    if dict(declaration) != expected:
        raise ValueError("Native retrieval branch declaration differs from the exact sealed recipe.")
    if stop_after not in (600, 1000, 1500, 2500):
        raise ValueError("The matched D0 continuation stop must be one of the sealed review ages.")
    source_path = source_path.resolve(strict=True)
    if (source_path != Path(recipe["source_checkpoint"]).resolve(strict=True)
            or file_sha256(source_path) != recipe["source_checkpoint_sha256"]
            or source_payload.get("epoch") != 500 or source_payload.get("arm") != "A2-D"):
        raise ValueError("Native retrieval source is not the exact pinned D0 e500 A2-D checkpoint.")
    source_identity = source_payload.get("experiment_identity")
    if not isinstance(source_identity, Mapping):
        raise TypeError("D0 source checkpoint is missing its experiment identity.")
    if (source_identity.get("recipe_sha256") != recipe["source_recipe_sha256"]
            or canonical_sha256(source_identity.get("recipe")) != recipe["base_recipe_sha256"]
            or canonical_sha256(source_identity.get("provider_identity"))
            != recipe["source_provider_identity_sha256"]):
        raise ValueError("Native retrieval source recipe/provider identity changed from its seal.")
    source_provider = source_identity.get("provider_identity")
    child_provider = sealed_identity.get("provider_identity")
    if not isinstance(source_provider, Mapping) or not isinstance(child_provider, Mapping):
        raise TypeError("Native retrieval branch lacks source or child provider identity.")
    _provider_pair(source_provider, child_provider, recipe["dataset"])

    source_state = source_payload.get("model_state_dict")
    target_state = model.state_dict()
    if not isinstance(source_state, Mapping):
        raise TypeError("D0 source checkpoint has no model state dictionary.")
    if recipe["treatment"] == "D-control":
        if set(target_state) != set(source_state):
            raise ValueError("D-control must preserve the original A2-D state key inventory exactly.")
        model.load_state_dict(source_state, strict=True)
        target_state = model.state_dict()
        added_state = []
    else:
        expected_keys = set(source_state) | {"native_training_epoch"}
        if recipe["dataset"] == "thermal":
            expected_keys = set(source_state) | {"core.native_training_epoch"}
            epoch_key = "core.native_training_epoch"
        else:
            epoch_key = "native_training_epoch"
        if set(target_state) != expected_keys:
            raise ValueError("G-native-M state may add only the one absolute-age buffer.")
        missing = set(target_state) - set(source_state) - {epoch_key}
        if missing:
            raise ValueError(f"G-native-M introduced undeclared state keys: {sorted(missing)}")
        with torch.no_grad():
            for name, value in source_state.items():
                if name not in target_state or target_state[name].shape != value.shape:
                    raise ValueError(f"G-native-M source tensor name/shape mismatch at {name!r}.")
                target_state[name].copy_(value.to(device=target_state[name].device, dtype=target_state[name].dtype))
            target_state[epoch_key].fill_(500)
        if int(target_state[epoch_key]) != 500:
            raise ValueError("G-native-M permission curriculum did not restore absolute e500.")
        added_state = [epoch_key]
    copied = [name for name in source_state if name in target_state]
    if len(copied) != len(source_state) or any(not _same_state_tensor(target_state[name], source_state[name]) for name in copied):
        raise ValueError("Round7 failed exact by-name D0 state inheritance.")

    source_optimizer = source_payload.get("optimizer_state_by_name")
    if not isinstance(source_optimizer, Mapping):
        raise TypeError("D0 checkpoint has no named AdamW state.")
    optimizer_names = {name for name, _parameter in model.named_parameters()}
    source_names = set(source_optimizer.get("state_by_name", {}))
    if optimizer_names != source_names:
        raise ValueError("Round7 child optimizer parameter names differ from the exact D0 named state.")
    actual_specs = tuple(provider.optimizer_groups(model, "A2-D", "joint"))
    from .unified_training import _optimizer_spec_payload

    expected_schedule = source_payload.get("optimizer_schedule_contract")
    observed_schedule = [_optimizer_spec_payload(spec) for spec in actual_specs]
    if (expected_schedule != observed_schedule
            or source_payload.get("optimizer_group_names") != [spec.name for spec in actual_specs]):
        raise ValueError("Round7 continuation changed the exact D0 optimizer group/schedule contract.")
    expected_groups = source_optimizer.get("groups")
    observed_groups = [{"name": spec.name, "parameter_names": list(spec.parameter_names)} for spec in actual_specs]
    if expected_groups != observed_groups:
        raise ValueError("Round7 continuation changed named AdamW parameter-group membership/order.")
    history = source_payload.get("history", ())
    inherited_visits = sum(int(row["case_visits"]) for row in history)
    inherited_updates = sum(int(row["optimizer_updates"]) for row in history)
    if recipe["dataset"] == "thermal" and inherited_updates != 2000:
        raise ValueError("Thermal D0 e500 must retain its exact 4-updates-per-epoch optimizer history.")
    if recipe["dataset"] == "wind" and inherited_updates != 2500:
        raise ValueError("Wind D0 e500 must retain its exact 5-updates-per-epoch optimizer history.")
    amendment = {
        **expected,
        "exact_common_model_state_by_name": True,
        "added_state_keys": added_state,
        "named_optimizer_parent_contract_verified": True,
        "optimizer_state_parameter_count": len(source_names),
        "inherited_case_visits": inherited_visits,
        "inherited_optimizer_updates": inherited_updates,
        "native_environment_value_reads": 0,
    }
    return amendment


def verify_native_retrieval_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source_named_state: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    from .unified_training import named_optimizer_state

    observed = named_optimizer_state(model, optimizer)
    expected_states = source_named_state.get("state_by_name", {})
    if not isinstance(expected_states, Mapping) or set(observed["state_by_name"]) != set(expected_states):
        raise ValueError("Round7 exact inherited AdamW state inventory differs after optimizer construction.")
    for name, expected_state in expected_states.items():
        actual_state = observed["state_by_name"].get(name)
        if not isinstance(actual_state, Mapping) or set(actual_state) != set(expected_state):
            raise ValueError(f"Round7 inherited AdamW state is missing for {name!r}.")
        for key, expected in expected_state.items():
            actual = actual_state[key]
            same = (_same_state_tensor(actual, expected) if torch.is_tensor(expected) else actual == expected)
            if not same:
                raise ValueError(f"Round7 inherited AdamW moment changed at {name!r}/{key}.")
    if observed.get("groups") != source_named_state.get("groups"):
        raise ValueError("Round7 inherited AdamW named group membership changed.")
    result = {
        "exact_inherited_moments_verified": True,
        "new_optimizer_states_at_attachment": 0,
        "optimizer_state_parameter_count": len(expected_states),
    }
    amendment.update(result)
    return result


def verify_native_retrieval_resume(payload: Mapping[str, Any], *, recipe: Mapping[str, Any], stop_after: int) -> None:
    value = validate_native_retrieval_recipe(recipe)
    branch = payload.get("branch_parent")
    contract = branch.get("native_retrieval_child") if isinstance(branch, Mapping) else None
    expected = native_retrieval_child_declaration(value)
    if not isinstance(contract, Mapping) or any(contract.get(key) != item for key, item in expected.items()):
        raise ValueError("Native retrieval resume lost its exact D0 child declaration.")
    completed = payload.get("epoch")
    if type(completed) is not int or completed < 501 or completed >= stop_after or stop_after > 2500:
        raise ValueError("Native retrieval resume must advance its absolute age within e2500.")
    if value["treatment"] == "G-native-M":
        state = payload.get("model_state_dict", {})
        epoch_key = "core.native_training_epoch" if value["dataset"] == "thermal" else "native_training_epoch"
        if int(state[epoch_key]) != completed:
            raise ValueError("Native retrieval checkpoint buffer is stale relative to its absolute epoch.")


__all__ = ["prepare_native_retrieval_child", "verify_native_retrieval_optimizer", "verify_native_retrieval_resume"]
