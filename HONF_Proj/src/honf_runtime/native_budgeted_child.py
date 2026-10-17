"""Exact Round8 D0 child inheritance and optimizer-state verification."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .native_budgeted_protocol import (
    canonical_sha256,
    file_sha256,
    native_budgeted_child_declaration,
    validate_native_budgeted_recipe,
)


def _same(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a.detach().cpu(), b.detach().cpu()))
    return a == b


def _core(model: torch.nn.Module, dataset: str):
    from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore

    core = getattr(model, "core", None) if dataset == "thermal" else model
    if not isinstance(core, BudgetedNativeCore):
        raise TypeError("G8 child must contain the registered BudgetedNativeCore.")
    return core


def _route_parameter_names(model: torch.nn.Module, dataset: str) -> tuple[str, ...]:
    core = _core(model, dataset)
    full = tuple(name for name, _ in model.named_parameters())
    names = []
    for local in core.budgeted_optimizer_parameter_names():
        match = [name for name in full if name == local or name.endswith("." + local)]
        if len(match) != 1:
            raise ValueError(f"Round8 route parameter {local!r} has no unique qualified model name.")
        names.append(match[0])
    if not names or len(set(names)) != len(names):
        raise ValueError("Round8 route parameter inventory is empty or duplicated.")
    return tuple(names)


def _expected_specs(source_payload: Mapping[str, Any], route_names: tuple[str, ...], treatment: str):

    source = source_payload.get("optimizer_schedule_contract")
    if not isinstance(source, list) or not source:
        raise ValueError("D0 source has no named optimizer schedule contract.")
    if treatment == "D-control":
        return copy.deepcopy(source)
    route = set(route_names)
    expected = []
    for spec in source:
        if not isinstance(spec, Mapping):
            raise TypeError("D0 source optimizer schedule entries must be objects.")
        names = [name for name in spec.get("parameter_names", ()) if name not in route]
        if not names:
            raise ValueError("Extracting Round8 route parameters emptied a source optimizer group.")
        expected.append({**dict(spec), "parameter_names": names})
    original = source[0]
    expected.append({
        "name": "budgeted_native_router",
        "parameter_names": list(route_names),
        "schedule": copy.deepcopy(original["schedule"]),
        "weight_decay": 0.0,
        "betas": copy.deepcopy(original["betas"]),
        "eps": original["eps"],
    })
    return expected


def prepare_native_budgeted_child(
    model: torch.nn.Module, provider: Any, source_payload: Mapping[str, Any],
    sealed_identity: Mapping[str, Any], declaration: Mapping[str, Any], *,
    source_path: Path, stop_after: int,
) -> dict[str, Any]:
    recipe = validate_native_budgeted_recipe(
        sealed_identity.get("recipe", {}), verify_source_file=True, verify_admission=True,
    )
    expected_declaration = native_budgeted_child_declaration(recipe)
    if dict(declaration) != expected_declaration:
        raise ValueError("Round8 child declaration differs from the strict D0 source/admission recipe.")
    if stop_after not in (600, 1000, 1500, 2500):
        raise ValueError("Round8 continuation must stop at a declared e600/e1000/e1500/e2500 review.")
    source_path = source_path.resolve(strict=True)
    if (source_path != Path(recipe["source_checkpoint"]).resolve(strict=True)
            or file_sha256(source_path) != recipe["source_checkpoint_sha256"]
            or source_payload.get("epoch") != 500 or source_payload.get("arm") != "A2-D"
            or source_payload.get("workflow") != "content_hypergraph_joint_v2"):
        raise ValueError("Round8 branch must use the exact recipe-pinned D0 A2-D e500 checkpoint.")
    source_identity = source_payload.get("experiment_identity")
    if not isinstance(source_identity, Mapping):
        raise TypeError("D0 source has no sealed experiment identity.")
    if (source_identity.get("recipe_sha256") != recipe["source_recipe_sha256"]
            or canonical_sha256(source_identity.get("recipe")) != recipe["base_recipe_sha256"]
            or canonical_sha256(source_identity.get("provider_identity"))
            != recipe["source_provider_identity_sha256"]):
        raise ValueError("Round8 D0 source recipe/provider identity differs from its seal.")
    parent_provider = source_identity.get("provider_identity")
    child_provider = sealed_identity.get("provider_identity")
    if not isinstance(parent_provider, Mapping) or not isinstance(child_provider, Mapping):
        raise TypeError("Round8 child has no exact parent or active provider identity.")
    if recipe["treatment"] == "D-control":
        if canonical_sha256(parent_provider) != canonical_sha256(child_provider):
            raise ValueError("D8 provider identity must remain exactly the source D0 provider.")
    else:
        from .native_budgeted_runtime import _round8_provider_pair

        _round8_provider_pair(parent_provider, child_provider, recipe["dataset"])

    source_state = source_payload.get("model_state_dict")
    target_state = model.state_dict()
    if not isinstance(source_state, Mapping):
        raise TypeError("D0 source has no model state dictionary.")
    if recipe["treatment"] == "D-control":
        if set(target_state) != set(source_state):
            raise ValueError("D8 must preserve the complete A2-D source state-key inventory exactly.")
        model.load_state_dict(source_state, strict=True)
        target_state = model.state_dict()
        route_names: tuple[str, ...] = ()
        added_state: list[str] = []
    else:
        core = _core(model, recipe["dataset"])
        route_names = _route_parameter_names(model, recipe["dataset"])
        if recipe["dataset"] == "thermal":
            age_key, macro_key, router_prefix = (
                "core.native_training_epoch", "core.budgeted_macro_step", "core.budgeted_router.",
            )
        else:
            age_key, macro_key, router_prefix = (
                "native_training_epoch", "budgeted_macro_step", "budgeted_router.",
            )
        added_state = sorted(set(target_state) - set(source_state))
        allowed = {age_key, macro_key}
        if not added_state or not set(added_state).issubset(allowed | {
            name for name in target_state if name.startswith(router_prefix)
        }):
            raise ValueError("G8 may add only the route calibration/dual state and declared absolute clocks.")
        if set(source_state) - set(target_state):
            raise ValueError("G8 dropped one or more original A2-D state keys.")
        with torch.no_grad():
            for name, saved in source_state.items():
                current = target_state.get(name)
                if (not torch.is_tensor(current) or current.shape != saved.shape
                        or current.dtype != saved.dtype):
                    raise ValueError(f"G8 source state name/shape/dtype differs at {name!r}.")
                current.copy_(saved.to(device=current.device))
            target_state[age_key].fill_(500)
            target_state[macro_key].zero_()
        core.set_native_training_epoch(500)
    if any(not _same(model.state_dict()[name], saved) for name, saved in source_state.items()):
        raise ValueError("Round8 child failed exact by-name model inheritance from D0 e500.")

    source_optimizer = source_payload.get("optimizer_state_by_name")
    if not isinstance(source_optimizer, Mapping):
        raise TypeError("D0 source lacks named AdamW moments.")
    source_moments = source_optimizer.get("state_by_name")
    if not isinstance(source_moments, Mapping):
        raise TypeError("D0 source named AdamW state has no state_by_name mapping.")
    source_parameter_names = set(source_moments)
    target_parameter_names = {name for name, _ in model.named_parameters()}
    if recipe["treatment"] == "D-control":
        if target_parameter_names != source_parameter_names:
            raise ValueError("D8 named AdamW parameter inventory differs from D0.")
    elif target_parameter_names != source_parameter_names | set(route_names):
        raise ValueError("G8 must preserve every source parameter and add only named budgeted-router parameters.")
    actual_specs = tuple(provider.optimizer_groups(model, "A2-D", "joint"))
    from .unified_training import _optimizer_spec_payload

    expected_schedule = _expected_specs(source_payload, route_names, recipe["treatment"])
    observed_schedule = [_optimizer_spec_payload(spec) for spec in actual_specs]
    if observed_schedule != expected_schedule:
        raise ValueError("Round8 optimizer schedule/groups differ from exact D0 plus the one fresh route group.")
    source_groups = source_optimizer.get("groups")
    expected_groups = [
        {"name": row["name"], "parameter_names": row["parameter_names"]}
        for row in expected_schedule
    ]
    if not isinstance(source_groups, list):
        raise TypeError("D0 source named AdamW moments have no named group list.")
    if recipe["treatment"] == "D-control":
        if source_groups != expected_groups:
            raise ValueError("D8 source named AdamW group membership/order changed.")
    else:
        expected_parent_groups = [
            {"name": row["name"], "parameter_names": [
                name for name in row["parameter_names"] if name not in set(route_names)
            ]}
            for row in source_groups
        ]
        if expected_groups[:-1] != expected_parent_groups:
            raise ValueError("G8 altered source AdamW group names, order or original parameter membership.")
    source_history = source_payload.get("history", ())
    inherited_visits = sum(int(row["case_visits"]) for row in source_history)
    inherited_updates = sum(int(row["optimizer_updates"]) for row in source_history)
    expected_updates = 2000 if recipe["dataset"] == "thermal" else 2500
    if inherited_updates != expected_updates:
        raise ValueError("D0 e500 optimizer history differs from the exact dataset-specific update count.")
    return {
        **expected_declaration,
        "exact_source_model_state_by_name": True,
        "added_state_keys": added_state,
        "source_optimizer_state_parameter_count": len(source_moments),
        "new_route_parameter_names": list(route_names),
        "new_route_moments_at_attachment": 0,
        "source_optimizer_groups_and_schedule_verified": True,
        "inherited_case_visits": inherited_visits,
        "inherited_optimizer_updates": inherited_updates,
    }


def verify_native_budgeted_optimizer(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    source_named_state: Mapping[str, Any], amendment: Mapping[str, Any], *, dataset: str,
) -> dict[str, Any]:
    from .unified_training import named_optimizer_state

    observed = named_optimizer_state(model, optimizer)
    source = source_named_state.get("state_by_name")
    if not isinstance(source, Mapping):
        raise TypeError("Round8 source moments are malformed.")
    route_names = tuple(amendment.get("new_route_parameter_names", ()))
    for name, expected_state in source.items():
        actual = observed["state_by_name"].get(name)
        if not isinstance(actual, Mapping) or set(actual) != set(expected_state):
            raise ValueError(f"Round8 inherited AdamW moment inventory differs at {name!r}.")
        if any(not _same(actual[key], value) for key, value in expected_state.items()):
            raise ValueError(f"Round8 inherited AdamW moment changed at {name!r}.")
    new_states = set(observed["state_by_name"]) - set(source)
    if new_states:
        raise ValueError(f"Round8 fresh route optimizer moments were materialized before update: {sorted(new_states)}.")
    groups = observed.get("groups", [])
    expected_source_groups = source_named_state.get("groups")
    if not isinstance(expected_source_groups, list):
        raise TypeError("Round8 D0 optimizer group membership is malformed.")
    if amendment.get("treatment") == "D-control":
        if groups != expected_source_groups:
            raise ValueError("D8 changed the original AdamW groups or parameter order.")
    elif groups[:-1] != expected_source_groups:
        raise ValueError("G8 changed the original AdamW groups or parameter order.")
    if route_names:
        route_group = next((row for row in groups if row.get("name") == "budgeted_native_router"), None)
        if route_group != {"name": "budgeted_native_router", "parameter_names": list(route_names)}:
            raise ValueError("Round8 route parameters lack the separately named fresh AdamW group.")
    result = {
        "exact_source_adamw_moments_verified": True,
        "fresh_route_optimizer_state_count_at_attachment": len(new_states),
        "route_parameter_count": len(route_names),
        "observed_named_groups": groups,
    }
    if isinstance(amendment, dict):
        amendment.update(result)
    return result


def verify_native_budgeted_child_amendment(
    declaration: Mapping[str, Any], *, recipe: Mapping[str, Any],
    source_payload: Mapping[str, Any], model_state_dict: Mapping[str, Any],
    optimizer_state_by_name: Mapping[str, Any] | None = None,
) -> None:
    """Validate the static D0 child declaration and its measured attachment audit."""
    value = validate_native_budgeted_recipe(recipe, verify_source_file=True, verify_admission=True)
    expected = native_budgeted_child_declaration(value)
    source_state = source_payload.get("model_state_dict")
    source_optimizer = source_payload.get("optimizer_state_by_name")
    source_moments = source_optimizer.get("state_by_name") if isinstance(source_optimizer, Mapping) else None
    source_groups = source_optimizer.get("groups") if isinstance(source_optimizer, Mapping) else None
    history = source_payload.get("history")
    if (not isinstance(declaration, Mapping) or not isinstance(source_state, Mapping)
            or not isinstance(source_moments, Mapping) or not isinstance(source_groups, list)
            or not isinstance(history, list)):
        raise TypeError("Round8 child audit requires the exact source model, named moments, groups and history.")
    inherited_visits = sum(int(row["case_visits"]) for row in history if isinstance(row, Mapping))
    inherited_updates = sum(int(row["optimizer_updates"]) for row in history if isinstance(row, Mapping))
    if any(declaration.get(key) != item for key, item in expected.items()):
        raise ValueError("Round8 checkpoint lost its strict D0 source/admission child declaration.")

    extra_fields = {
        "exact_source_model_state_by_name", "added_state_keys", "source_optimizer_state_parameter_count",
        "new_route_parameter_names", "new_route_moments_at_attachment",
        "source_optimizer_groups_and_schedule_verified", "inherited_case_visits",
        "inherited_optimizer_updates", "exact_source_adamw_moments_verified",
        "fresh_route_optimizer_state_count_at_attachment", "route_parameter_count", "observed_named_groups",
    }
    engine_attachment_fields = {
        "next_epoch", "preserved_case_visits", "preserved_optimizer_updates", "recorded_unix",
    }
    if set(declaration) != set(expected) | extra_fields | engine_attachment_fields:
        raise ValueError("Round8 child amendment has a missing or unexpected attachment-audit field.")
    if (declaration.get("next_epoch") != 501
            or declaration.get("preserved_case_visits") != inherited_visits
            or declaration.get("preserved_optimizer_updates") != inherited_updates):
        raise ValueError("Round8 child engine attachment envelope differs from the e500-to-e501 branch.")
    recorded_unix = declaration.get("recorded_unix")
    if (isinstance(recorded_unix, bool) or not isinstance(recorded_unix, (int, float))
            or not math.isfinite(float(recorded_unix)) or float(recorded_unix) <= 0.0):
        raise ValueError("Round8 child engine attachment timestamp is malformed.")
    if declaration.get("exact_source_model_state_by_name") is not True:
        raise ValueError("Round8 child did not record exact source model inheritance.")
    if declaration.get("source_optimizer_groups_and_schedule_verified") is not True:
        raise ValueError("Round8 child did not verify inherited AdamW groups and schedules.")
    if declaration.get("exact_source_adamw_moments_verified") is not True:
        raise ValueError("Round8 child did not verify exact named AdamW moments.")
    if (declaration.get("new_route_moments_at_attachment") != 0
            or declaration.get("fresh_route_optimizer_state_count_at_attachment") != 0):
        raise ValueError("Round8 route group must start with fresh zero AdamW moments.")
    if declaration.get("source_optimizer_state_parameter_count") != len(source_moments):
        raise ValueError("Round8 child source optimizer parameter count differs from D0 e500.")
    if (declaration.get("inherited_case_visits") != inherited_visits
            or declaration.get("inherited_optimizer_updates") != inherited_updates):
        raise ValueError("Round8 child does not preserve the complete D0 source history exposure.")
    if inherited_updates != (2000 if value["dataset"] == "thermal" else 2500):
        raise ValueError("Round8 child source history has the wrong exact D0 update count.")

    added_state = sorted(set(model_state_dict) - set(source_state))
    if (set(source_state) - set(model_state_dict)
            or declaration.get("added_state_keys") != added_state):
        raise ValueError("Round8 child state-key amendment differs from its exact D0 parent.")
    if value["dataset"] == "thermal":
        age_key, macro_key, route_prefix = (
            "core.native_training_epoch", "core.budgeted_macro_step", "core.budgeted_router.",
        )
    else:
        age_key, macro_key, route_prefix = (
            "native_training_epoch", "budgeted_macro_step", "budgeted_router.",
        )
    allowed_state_additions = {age_key, macro_key} | {
        name for name in added_state if name.startswith(route_prefix)
    }
    if set(added_state) - allowed_state_additions:
        raise ValueError("Round8 child added state outside the route controller and absolute clocks.")
    route_names = declaration.get("new_route_parameter_names")
    if not isinstance(route_names, list) or any(not isinstance(name, str) for name in route_names):
        raise ValueError("Round8 route parameter names do not match the newly born state namespace.")
    if (len(route_names) != len(set(route_names))
            or any(not name.startswith(route_prefix) or name not in model_state_dict
                   or name in source_moments for name in route_names)):
        raise ValueError("Round8 route parameter names do not match the newly born state namespace.")
    if ((value["treatment"] == "D-control" and route_names)
            or (value["treatment"] == "G-budgeted-M" and not route_names)):
        raise ValueError("Round8 D/G child route parameter inventory differs from its declared treatment.")
    if declaration.get("route_parameter_count") != len(route_names):
        raise ValueError("Round8 child route parameter count differs from its named group.")

    expected_groups = copy.deepcopy(source_groups)
    if value["treatment"] == "G-budgeted-M":
        expected_groups.append({"name": "budgeted_native_router", "parameter_names": route_names})
    if declaration.get("observed_named_groups") != expected_groups:
        raise ValueError("Round8 child attachment group order/membership differs from the D0 named optimizer.")
    if optimizer_state_by_name is not None:
        current_groups = optimizer_state_by_name.get("groups")
        if current_groups != expected_groups:
            raise ValueError("Round8 checkpoint named optimizer groups differ from the audited child attachment.")


def verify_native_budgeted_resume(payload: Mapping[str, Any], *, recipe: Mapping[str, Any], stop_after: int) -> None:
    value = validate_native_budgeted_recipe(recipe, verify_source_file=True, verify_admission=True)
    branch = payload.get("branch_parent")
    declaration = branch.get("native_budgeted_child") if isinstance(branch, Mapping) else None
    from .compat import load_trusted_checkpoint

    source_payload = load_trusted_checkpoint(value["source_checkpoint"], map_location="cpu")
    verify_native_budgeted_child_amendment(
        declaration, recipe=value, source_payload=source_payload,
        model_state_dict=payload.get("model_state_dict", {}),
        optimizer_state_by_name=payload.get("optimizer_state_by_name"),
    )
    age = payload.get("epoch")
    if type(age) is not int or age < 501 or age >= stop_after or stop_after > 2500:
        raise ValueError("Round8 exact resume must advance the saved absolute age within e2500.")
    identity = payload.get("experiment_identity", {})
    if identity.get("recipe_sha256") != canonical_sha256(value):
        raise ValueError("Round8 exact resume recipe differs from the original admitted branch.")
    history = payload.get("history")
    if (not isinstance(history, list)
            or [row.get("epoch") for row in history if isinstance(row, Mapping)] != list(range(501, age + 1))):
        raise ValueError("Round8 exact resume history must continuously cover absolute e501 through the checkpoint age.")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("Round8 resume checkpoint has no model state.")
    if value["treatment"] == "G-budgeted-M":
        age_key = "core.native_training_epoch" if value["dataset"] == "thermal" else "native_training_epoch"
        macro_key = "core.budgeted_macro_step" if value["dataset"] == "thermal" else "budgeted_macro_step"
        if age_key not in state or int(state[age_key]) != age:
            raise ValueError("Round8 native absolute-age buffer differs from saved history.")
        updates = sum(int(row.get("optimizer_updates", 0)) for row in history)
        if macro_key not in state or int(state[macro_key]) != updates:
            raise ValueError("Round8 global router macro clock differs from exact saved update history.")


__all__ = [
    "prepare_native_budgeted_child", "verify_native_budgeted_child_amendment",
    "verify_native_budgeted_optimizer", "verify_native_budgeted_resume",
]
