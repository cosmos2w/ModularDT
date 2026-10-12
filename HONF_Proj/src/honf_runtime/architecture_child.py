"""Strict attachment of native routing/control parameters to an existing P.

This is an architecture child, never an exact resume under a different model
identity. Existing tensors, optimizer groups and physical/data contracts are
retained; only the enumerated zero-output native module is added.
"""
from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

SCHEMA = "native_interaction_routing_child_v1"
PREFIX = "core.native_routing."


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(memoryview(array.numpy()).cast("B"))
    return digest.hexdigest()


def _inventory(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "elements": value.numel()} for name, value in sorted(state.items())}


def _canonical_provider(parent: Mapping[str, Any], child: Mapping[str, Any], mode: str):
    result = copy.deepcopy(dict(child))
    dataset = parent.get("task", parent.get("dataset"))
    if dataset == "ThermalChannel":
        config_key, family_key, mode_key = "model_config", "model_family", "arm"
        family = "thermal_native_curl_interaction_routing_v1"
        old_family = "thermal_interaction_preserving_native_curl_v1"
        config = result.get(config_key)
        parent_config = parent.get(config_key)
        seed = parent_config.get("seed") if isinstance(parent_config, Mapping) else None
        routed_core = config
        old_core = parent_config
        cases = parent.get("training_case_ids")
        expected_cases = 150
        expected_control_width = 32
    elif dataset == "WindFarm":
        config_key, family_key, mode_key = "standalone_model_config", "family", "mode"
        family = "wind_native_interaction_routing_v1"
        old_family = "wind_interaction_preserving_joint_v1"
        config = result.get(config_key)
        parent_config = parent.get(config_key)
        seed = parent_config.get("seed") if isinstance(parent_config, Mapping) else None
        routed_core = config.get("core") if isinstance(config, Mapping) else None
        old_core = parent_config.get("core") if isinstance(parent_config, Mapping) else None
        cases = parent.get("train_rows")
        expected_cases = 72
        expected_control_width = 34
        if parent.get("profile_is_formal_fulltrain") is not False:
            raise ValueError("Native routing requires the segmented Wind parent, not a full-TRAIN migration.")
    else:
        raise ValueError("Native routing children require a ThermalChannel or WindFarm P parent.")
    if (not isinstance(config, dict) or not isinstance(parent_config, Mapping)
            or not isinstance(routed_core, dict) or not isinstance(old_core, Mapping)
            or old_core.get("mode") != "P" or old_core.get("regional_anchors") != 0
            or old_core.get("factorized_affine_execution", False)
            or routed_core.get("mode") != mode or routed_core.get("regional_anchors") != 0
            or routed_core.get("factorized_affine_execution", False)
            or parent.get(family_key) != old_family or result.get(family_key) != family
            or parent.get(mode_key) != "P" or result.get(mode_key) != mode
            or config.get("family") != family or parent_config.get("family") != old_family):
        raise ValueError("Parent and child must bind the literal legacy P and versioned direct R/C families.")
    routing = routed_core.get("native_routing")
    expected = {"schema": "native_interaction_routing_v1", "mode": mode,
                "initialization_seed": seed,
                "policy": "learned_density" if mode == "P-R" else "destination_gain"}
    if mode == "P-R":
        expected.update(score_hidden=32, score_bound=2.0)
    else:
        expected.update(control_hidden=expected_control_width)
    if routing != expected:
        raise ValueError("Native routing configuration differs from the declared v1 scorer/control.")
    routed_core.pop("native_routing")
    routed_core["mode"] = "P"
    config["family"] = old_family
    result[family_key] = old_family
    result[mode_key] = "P"
    if config != parent_config or result != parent:
        raise ValueError("Architecture attachment changed data, normalization, native outputs, losses or provider settings.")
    if (not isinstance(cases, list) or len(cases) != expected_cases
            or len(set(cases)) != len(cases)):
        raise ValueError("Native routing must retain the exact segmented parent TRAIN population.")
    return dataset, cases, expected


def prepare_native_routing_child(
    model: nn.Module, source_payload: Mapping[str, Any], current_identity: Mapping[str, Any],
    declaration: Mapping[str, Any], *, source_sha256: str, case_keys: Sequence[Any],
    stop_after: int,
) -> dict[str, Any]:
    """Validate the explicit child contract and strictly copy inherited state.

    Optimizer construction remains TrainingEngine's responsibility. A child
    optimizer receives the untouched named parent states; new states are absent
    until AdamW's normal first update, rather than assigned a fake prior age.
    """
    required = {"schema", "source_checkpoint_sha256", "source_epoch", "source_arm",
                "target_mode", "added_parameter_names", "review_epochs", "final_epoch",
                "scientific_update_limit", "reason"}
    if set(declaration) != required or declaration.get("schema") != SCHEMA:
        raise ValueError("An architecture child requires its exact versioned declaration.")
    mode = declaration["target_mode"]
    source_epoch = declaration["source_epoch"]
    reviews = declaration["review_epochs"]
    if (mode not in ("P-R", "P-C") or declaration["source_arm"] != "P"
            or type(source_epoch) is not int or source_epoch != 1000
            or source_payload.get("arm") != "P" or source_payload.get("epoch") != source_epoch
            or declaration["source_checkpoint_sha256"] != source_sha256
            or type(declaration["scientific_update_limit"]) is not int
            or not isinstance(declaration["reason"], str) or not declaration["reason"].strip()
            or not isinstance(reviews, list) or reviews != sorted(set(reviews))
            or any(type(value) is not int or value <= source_epoch or value > 2500 for value in reviews)
            or not reviews or stop_after not in reviews
            or declaration["final_epoch"] != reviews[-1]):
        raise ValueError("Native routing source, modes, reviews and bounded age must match the declared child.")
    parent_identity = source_payload.get("experiment_identity")
    if not isinstance(parent_identity, Mapping):
        raise TypeError("Native routing parent lacks its sealed experiment identity.")
    parent_provider = parent_identity.get("provider_identity")
    current_provider = current_identity.get("provider_identity")
    if not isinstance(parent_provider, Mapping) or not isinstance(current_provider, Mapping):
        raise TypeError("Native routing requires both provider identities.")
    dataset, parent_cases, routing_config = _canonical_provider(parent_provider, current_provider, mode)
    if len(case_keys) != len(parent_cases) or set(case_keys) != set(parent_cases):
        raise ValueError("An architecture child cannot resample or shorten its primary TRAIN population.")
    old_state = source_payload.get("model_state_dict")
    if not isinstance(old_state, Mapping) or not all(torch.is_tensor(v) for v in old_state.values()):
        raise TypeError("Native routing parent state must contain literal tensors.")
    current_state = model.state_dict()
    new_names = sorted(set(current_state) - set(old_state))
    declared_names = declaration["added_parameter_names"]
    named = dict(model.named_parameters())
    if (set(old_state) - set(current_state) or new_names != declared_names
            or not new_names or any(not n.startswith(PREFIX) for n in new_names)
            or any(n not in named for n in new_names)
            or any(not p.requires_grad for p in named.values())):
        raise ValueError("Only explicitly enumerated, trainable native-routing parameters may be added.")
    finals = [n for n in new_names if n.split(".")[-2] == "2"]
    expected_final_count = 3 if mode == "P-R" else 6
    if (len(finals) != expected_final_count or any(torch.count_nonzero(current_state[n]).item() for n in finals)
            or any(not torch.isfinite(current_state[n]).all() for n in new_names)):
        raise ValueError("Native routing attachment requires finite privately initialized modules with zero final projections.")
    for name, value in old_state.items():
        candidate = current_state[name]
        if candidate.shape != value.shape or candidate.dtype != value.dtype:
            raise ValueError(f"Inherited state shape/dtype mismatch for {name!r}.")
    normalized_identity = copy.deepcopy(dict(current_identity))
    normalized_identity["provider_identity"] = copy.deepcopy(dict(parent_provider))
    old_engine = copy.deepcopy(dict(parent_identity.get("engine_config", {})))
    new_engine = copy.deepcopy(dict(normalized_identity.get("engine_config", {})))
    for name in ("monitor_every", "monitor_epochs", "checkpoint_epochs", "latest_every", "curve_every", "write_initial_artifacts"):
        old_engine.pop(name, None)
        new_engine.pop(name, None)
    if old_engine != new_engine or new_engine.get("training_mode") != "joint":
        raise ValueError("Architecture children preserve native batches, queries, sampler and absolute schedule.")
    normalized_identity["engine_config"] = copy.deepcopy(parent_identity["engine_config"])
    groups = copy.deepcopy(normalized_identity.get("optimizer_schedule_contract", []))
    for group in groups:
        group["parameter_names"] = [n for n in group["parameter_names"] if n not in new_names]
    normalized_identity["optimizer_schedule_contract"] = groups
    # No permissive arbitrary-key exception: the caller retains parent scientific
    # metadata and gets a separate architecture receipt through branch_parent.
    if normalized_identity != parent_identity:
        raise ValueError("Architecture attachment changed inherited initialization, selection, schedule or sampling metadata.")
    parent_groups = source_payload["optimizer_state_by_name"].get("groups", [])
    source_schedules = source_payload.get("optimizer_schedule_contract")
    if (source_schedules != parent_identity.get("optimizer_schedule_contract")
            or [g["name"] for g in parent_groups] != source_payload.get("optimizer_group_names")
            or any(g["parameter_names"] != s["parameter_names"] for g, s in zip(parent_groups, source_schedules))):
        raise ValueError("Parent named optimizer groups disagree with its schedule identity.")
    child_groups = current_identity["optimizer_schedule_contract"]
    if not all(any(n in g["parameter_names"] for g in child_groups) for n in new_names):
        raise ValueError("Every native routing parameter must belong to the inherited optimizer groups.")
    old_named_states = source_payload["optimizer_state_by_name"].get("state_by_name", {})
    if set(old_named_states) - set(old_state):
        raise ValueError("Parent optimizer contains names outside the inherited model.")
    updates = math.ceil(len(parent_cases) / int(new_engine["effective_cases"]))
    requested = updates * (declaration["final_epoch"] - source_epoch)
    cap = 6000 if dataset == "ThermalChannel" else 4500
    if not 0 < requested <= declaration["scientific_update_limit"] <= cap:
        raise ValueError("Architecture child exceeds its declared scientific update cap.")
    new_state = {name: current_state[name].detach().cpu().clone() for name in new_names}
    merged = {**dict(current_state), **dict(old_state)}
    model.load_state_dict(merged, strict=True)
    for name, value in old_state.items():
        if not torch.equal(model.state_dict()[name].detach().cpu(), value.detach().cpu()):
            raise ValueError(f"Inherited tensor {name!r} changed during attachment.")
    return {**copy.deepcopy(dict(declaration)), "dataset": dataset,
            "routing_configuration": routing_config,
            "inherited_parameter_inventory": _inventory(old_state),
            "new_parameter_inventory": _inventory(new_state),
            "new_parameter_initial_state_sha256": _state_digest(new_state),
            "inherited_state_sha256": _state_digest(old_state),
            "child_initial_state_sha256": _state_digest(model.state_dict()),
            "exact_parent_state_copied": True, "all_parameters_trainable": True,
            "optimizer_mapping": "exact inherited named moments; new parameter states initially absent",
            "parent_provider_identity": copy.deepcopy(dict(parent_provider)),
            "updates_per_epoch": updates, "planned_optimizer_updates": requested,
            "fit_case_keys": list(parent_cases), "initialization": "private seed; zero-output final projections"}


def verify_architecture_child_optimizer(model: nn.Module, optimizer, source_named_state, new_names):
    """Confirm exact copied moments and genuinely fresh newly added state."""
    parameters = dict(model.named_parameters())
    for name, expected in source_named_state["state_by_name"].items():
        actual = optimizer.state.get(parameters[name])
        if actual is None or set(actual) != set(expected):
            raise ValueError(f"Inherited optimizer state is missing or altered for {name!r}.")
        for key, value in expected.items():
            got = actual[key]
            equal = torch.equal(got.detach().cpu(), value.detach().cpu()) if torch.is_tensor(value) else got == value
            if not equal:
                raise ValueError(f"Inherited optimizer state changed for {name!r}/{key}.")
    if any(parameters[name] in optimizer.state for name in new_names):
        raise ValueError("New native parameters must have fresh AdamW state, not invented historical moments.")


def architecture_child_provider_state(provider, source_state: Mapping[str, Any], mode: str):
    """Map only a provider's literal arm label after the strict child check.

    Thermal's unchanged state loader deliberately rejects another arm. The
    architecture attachment changes that label while preserving every saved
    objective/calibration value; ordinary resume still uses its strict loader.
    """
    if mode not in ("P-R", "P-C") or not isinstance(source_state, Mapping):
        raise ValueError("Native child provider state requires its declared mode and a mapping.")
    mapped = copy.deepcopy(dict(source_state))
    if "arm" in mapped:
        if mapped["arm"] != "P":
            raise ValueError("Architecture child provider state must belong to literal P.")
        mapped["arm"] = mode
    state_method = getattr(provider, "training_state_dict", None)
    current = dict(state_method()) if callable(state_method) else {}
    if mapped != current:
        raise ValueError("Architecture child changed non-arm provider state or calibration.")
    return mapped
