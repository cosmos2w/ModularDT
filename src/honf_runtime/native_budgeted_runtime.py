"""Round8-only provider capability, optimizer grouping, and checkpoint loading."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from types import MethodType
from typing import Any

import torch

from .native_budgeted_protocol import (
    NATIVE_BUDGETED_FAMILY,
    NATIVE_BUDGETED_WORKFLOW,
    canonical_sha256,
    validate_native_budgeted_recipe,
)


def _budgeted_core(model: Any, dataset: str):
    from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore

    core = getattr(model, "core", None) if dataset == "thermal" else model
    if not isinstance(core, BudgetedNativeCore):
        raise TypeError("G8 provider binding requires the registered BudgetedNativeCore.")
    config = core.get_config()
    expected = {
        "schema", "base_model_config", "native_backend", "routing_seed",
        "membership_bias", "lambda0", "environment_value_reader",
    }
    base = config.get("base_model_config")
    if (set(config) != expected or config.get("schema") != "native_budgeted_dispatch_v2"
            or config.get("environment_value_reader") is not False
            or not isinstance(base, Mapping) or base.get("architecture") != "A2-D"):
        raise ValueError("G8 requires the strict no-E A2-D budgeted schema.")
    return core


def budgeted_core_from_parent(parent_model: torch.nn.Module, *, dataset: str,
                              recipe: Mapping[str, Any]):
    """Copy an exact A2-D parent by original parameter name into the Round8 core."""
    from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
    from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore

    value = validate_native_budgeted_recipe(recipe, verify_source_file=True, verify_admission=True)
    parent = getattr(parent_model, "core", None) if dataset == "thermal" else parent_model
    if type(parent) is not ContentHypergraphCoreV2 or parent.get_config().get("architecture") != "A2-D":
        raise TypeError("G8 must attach to the original source A2-D ContentHypergraphCoreV2.")
    init = value["routing_initialization"]
    core = BudgetedNativeCore.from_a2d(
        parent,
        native_backend=value["native_backend"],
        routing_seed=int(init["routing_seed"]),
        membership_bias=float(init["membership_bias"]),
        lambda0=float(init["lambda0"]),
        environment_value_reader=False,
    )
    source_state, target_state = parent.state_dict(), core.state_dict()
    added = set(target_state) - set(source_state)
    allowed = {"native_training_epoch", "budgeted_macro_step"}
    if not added or not added.issubset(allowed | {
        name for name in target_state if name.startswith("budgeted_router.")
    }):
        raise ValueError("Round8 core added undeclared persistent state beyond the native router and clocks.")
    if set(source_state) - set(target_state):
        raise ValueError("Round8 core dropped original A2-D state keys.")
    for name, tensor in source_state.items():
        if not torch.equal(target_state[name].detach().cpu(), tensor.detach().cpu()):
            raise ValueError(f"Round8 core failed exact source state copy at {name!r}.")
    if int(core.native_training_epoch) != 500:
        raise ValueError("Round8 native absolute age must begin at source e500.")
    config = core.get_config()
    if (config.get("native_backend") != value["native_backend"]
            or config.get("routing_seed") != init["routing_seed"]
            or config.get("membership_bias") != init["membership_bias"]
            or config.get("lambda0") != init["lambda0"]
            or config.get("environment_value_reader") is not False):
        raise ValueError("Round8 core constructor differs from the sealed calibration recipe.")
    if dataset == "thermal":
        parent_model.core = core
        return parent_model
    return core


def _install_budgeted_optimizer_groups(provider: Any, model: torch.nn.Module, dataset: str) -> None:
    """Keep every source group verbatim except route-name extraction; append one fresh group."""
    from .unified_training import OptimizerGroupSpec

    core = _budgeted_core(model, dataset)
    if getattr(provider, "_budgeted_optimizer_groups_installed", False):
        raise ValueError("Round8 provider optimizer wrapper was already installed.")
    original = provider.optimizer_groups
    local_route_names = tuple(core.budgeted_optimizer_parameter_names())
    full_names = tuple(name for name, _ in model.named_parameters())
    prefix_by_local: dict[str, str] = {}
    for local in local_route_names:
        matches = [name for name in full_names if name == local or name.endswith("." + local)]
        if len(matches) != 1:
            raise ValueError(f"Round8 route parameter {local!r} has no unique adapter-qualified name.")
        prefix_by_local[local] = matches[0]
    route_names = tuple(prefix_by_local[name] for name in local_route_names)
    if not route_names:
        raise ValueError("Round8 core has no trainable budgeted router parameters.")

    def budgeted_groups(_self, candidate_model, arm, stage):
        if candidate_model is not model:
            raise ValueError("Round8 optimizer groups are bound to the exact D0 provider model object.")
        specs = tuple(original(candidate_model, arm, stage))
        if not specs:
            raise ValueError("Round8 source provider declared no original optimizer groups.")
        route = set(route_names)
        observed_route = {name for spec in specs for name in spec.parameter_names if name in route}
        if observed_route != route:
            raise ValueError("Round8 provider source groups do not cover each new route parameter exactly once.")
        filtered = []
        for spec in specs:
            names = tuple(name for name in spec.parameter_names if name not in route)
            if not names:
                raise ValueError("Round8 source group became empty after extracting route parameters.")
            filtered.append(OptimizerGroupSpec(
                spec.name, names, spec.schedule, weight_decay=spec.weight_decay,
                betas=spec.betas, eps=spec.eps,
            ))
        source = specs[0]
        filtered.append(OptimizerGroupSpec(
            "budgeted_native_router", route_names, source.schedule,
            weight_decay=0.0, betas=source.betas, eps=source.eps,
        ))
        return tuple(filtered)

    provider._budgeted_source_optimizer_groups = original
    provider.optimizer_groups = MethodType(budgeted_groups, provider)
    provider._budgeted_optimizer_groups_installed = True
    provider._budgeted_route_parameter_names = route_names


def bind_budgeted_native_provider(provider: Any, model: Any, *, dataset: str,
                                  recipe: Mapping[str, Any]) -> Any:
    """Enable only the new Round8 core path on the unchanged segmented D0 provider."""
    value = validate_native_budgeted_recipe(recipe, verify_source_file=True, verify_admission=True)
    if dataset != value["dataset"] or dataset not in ("thermal", "wind"):
        raise ValueError("Round8 provider dataset differs from its strict recipe.")
    task_recipe = getattr(provider, "recipe", None)
    if (not isinstance(task_recipe, Mapping) or task_recipe.get("schema") != "content_hypergraph_recipe_v2"
            or task_recipe.get("protocol") != "source_read_development_v1"
            or task_recipe.get("source_reader_variant") != "D0"
            or task_recipe.get("dataset") != dataset or task_recipe.get("architecture") != "A2-D"):
        raise ValueError("Round8 provider binding is restricted to the exact segmented D0 task.")
    core = _budgeted_core(model, dataset)
    if value["treatment"] != "G-budgeted-M":
        raise ValueError("Only G8 receives the Round8 core provider capability.")
    if dataset == "thermal":
        from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
        from channelthermal.training.content_hypergraph_task_v2 import ThermalSourceReadTaskV1

        if (not isinstance(provider, ThermalSourceReadTaskV1)
                or not isinstance(model, ThermalContentHypergraphAdapterV2)
                or provider.model is not model or provider.adapter is not model or model.core is not core):
            raise ValueError("Thermal Round8 must retain the same v2 adapter object in provider/model/core bindings.")
    else:
        from windfarm.training.content_hypergraph_task import (
            WindSourceReadTaskV1,
            _validate_budgeted_wind_core_binding,
        )

        if not isinstance(provider, WindSourceReadTaskV1):
            raise TypeError("Wind Round8 requires the maintained D0 source-read provider.")
        if provider.model is not model:
            raise ValueError("Wind Round8 provider must bind the exact newly wrapped source core.")
        _validate_budgeted_wind_core_binding(task_recipe, core.get_config())
        provider._allow_budgeted_native = True
    _install_budgeted_optimizer_groups(provider, model, dataset)
    return provider


def _round8_provider_pair(parent: Mapping[str, Any], child: Mapping[str, Any], dataset: str) -> None:
    old, new = copy.deepcopy(dict(parent)), copy.deepcopy(dict(child))
    if dataset == "thermal":
        for key in ("model_config", "adapter_state"):
            before, after = old.get(key), new.get(key)
            if not isinstance(before, Mapping) or not isinstance(after, Mapping):
                raise TypeError("Thermal Round8 provider identity is missing adapter/core configuration.")
            old_core, new_core = before.get("core"), after.get("core")
            if not isinstance(old_core, Mapping) or not isinstance(new_core, Mapping):
                raise TypeError("Thermal Round8 provider identity has no nested core configuration.")
            after["core"] = copy.deepcopy(dict(old_core))
    elif dataset == "wind":
        old_config, new_config = old.get("model_config"), new.get("model_config")
        if not isinstance(old_config, Mapping) or not isinstance(new_config, Mapping):
            raise TypeError("Wind Round8 provider identity is missing its core configuration.")
        new["model_config"] = copy.deepcopy(dict(old_config))
    else:
        raise ValueError("Round8 provider identity supports only Thermal or Wind.")
    if canonical_sha256(old) != canonical_sha256(new):
        raise ValueError("Round8 changed source data, transforms, objective or native law outside the core schema.")


def load_native_budgeted_checkpoint(path: str, *, device: torch.device | str = "cpu"):
    """Data-free strict loader for saved Round8 scientific checkpoints."""
    from .compat import load_trusted_checkpoint
    from .registry import require_model_family

    payload = load_trusted_checkpoint(path, map_location="cpu")
    identity = payload.get("experiment_identity", {})
    if (payload.get("workflow") != NATIVE_BUDGETED_WORKFLOW
            or not isinstance(identity, Mapping)
            or identity.get("workflow") != NATIVE_BUDGETED_WORKFLOW
            or identity.get("family") != NATIVE_BUDGETED_FAMILY
            or identity.get("model_family") != NATIVE_BUDGETED_FAMILY):
        raise ValueError("Checkpoint is not a sealed Round8 budgeted native-dispatch member.")
    require_model_family(NATIVE_BUDGETED_FAMILY, "native_budgeted_dispatch_evaluate")
    recipe = validate_native_budgeted_recipe(
        identity.get("recipe", {}), verify_source_file=True, verify_admission=True,
    )
    from .native_budgeted_protocol import canonical_sha256
    if (identity.get("recipe_sha256") != canonical_sha256(recipe)
            or identity.get("protocol") != recipe["protocol"]
            or identity.get("treatment") != recipe["treatment"]
            or payload.get("arm") != "A2-D"
            or type(payload.get("epoch")) is not int
            or not 500 <= payload["epoch"] <= 2500):
        raise ValueError("Round8 family, recipe, arm or absolute age is inconsistent.")
    branch = payload.get("branch_parent")
    child = branch.get("native_budgeted_child") if isinstance(branch, Mapping) else None
    parent_identity = branch.get("identity") if isinstance(branch, Mapping) else None
    if (not isinstance(branch, Mapping) or branch.get("path") != recipe["source_checkpoint"]
            or branch.get("sha256") != recipe["source_checkpoint_sha256"] or branch.get("epoch") != 500
            or not isinstance(parent_identity, Mapping)
            or parent_identity.get("recipe_sha256") != recipe["source_recipe_sha256"]
            or canonical_sha256(parent_identity.get("provider_identity"))
            != recipe["source_provider_identity_sha256"]
            or not isinstance(child, Mapping)):
        raise ValueError("Round8 checkpoint is detached from its exact D0 e500 child declaration.")
    provider = identity.get("provider_identity")
    parent_provider = parent_identity.get("provider_identity")
    if not isinstance(provider, Mapping) or not isinstance(parent_provider, Mapping):
        raise TypeError("Round8 checkpoint lacks a sealed provider/transform identity.")
    if recipe["treatment"] == "D-control":
        if canonical_sha256(provider) != canonical_sha256(parent_provider):
            raise ValueError("D8 changed its original D0 provider identity.")
    else:
        _round8_provider_pair(parent_provider, provider, recipe["dataset"])
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("Round8 checkpoint has no model tensor state.")
    from .native_budgeted_child import verify_native_budgeted_child_amendment

    source_payload = load_trusted_checkpoint(recipe["source_checkpoint"], map_location="cpu")
    verify_native_budgeted_child_amendment(
        child, recipe=recipe, source_payload=source_payload, model_state_dict=state,
        optimizer_state_by_name=payload.get("optimizer_state_by_name"),
    )
    if recipe["dataset"] == "thermal":
        from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2

        adapter_state = provider.get("adapter_state")
        if not isinstance(adapter_state, Mapping):
            raise TypeError("Round8 Thermal checkpoint has no self-contained adapter state.")
        model = ThermalContentHypergraphAdapterV2.from_state_payload(
            adapter_state, device=device, state_dict=state,
        )
        adapter = model
        observed = model.get_config()
    else:
        from windfarm.content_hypergraph import WindContentHypergraphAdapterV3

        from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore

        core_config = provider.get("model_config")
        adapter_state = provider.get("adapter_state")
        if not isinstance(core_config, Mapping) or not isinstance(adapter_state, Mapping):
            raise TypeError("Round8 Wind checkpoint has no core constructor or adapter payload.")
        if recipe["treatment"] == "G-budgeted-M":
            model = BudgetedNativeCore.from_config(core_config).to(device)
        else:
            from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2

            model = ContentHypergraphCoreV2.from_config(core_config).to(device)
        model.load_state_dict(state, strict=True)
        adapter = WindContentHypergraphAdapterV3.from_state_payload(adapter_state, device=device)
        observed = model.get_config()
    expected_config = provider.get("model_config")
    if recipe["dataset"] == "thermal":
        expected_config = provider.get("model_config")
    if observed != expected_config:
        raise ValueError("Round8 reconstructed model/adapter differs from the embedded provider config.")
    model.to(device)
    model.eval()
    if hasattr(adapter, "eval"):
        adapter.eval()
    return model, adapter, payload


__all__ = [
    "bind_budgeted_native_provider", "budgeted_core_from_parent", "load_native_budgeted_checkpoint",
]
