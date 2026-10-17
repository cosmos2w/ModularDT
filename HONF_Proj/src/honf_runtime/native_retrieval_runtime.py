"""Narrow runtime adapters for the opt-in Round7 retrieval family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _retrieval_core(model: Any, dataset: str):
    from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
    from honf_forward_core.native_retrieval.core import NativeRetrievalCore

    core = getattr(model, "core", None) if dataset == "thermal" else model
    config = core.get_config()
    if config.get("schema") != NativeRetrievalCore.schema:
        if type(core) is not ContentHypergraphCoreV2 or config.get("architecture") != "A2-D":
            raise ValueError("D-control must preserve the exact original A2-D core schema.")
        return core
    if not isinstance(core, NativeRetrievalCore):
        raise TypeError("G-native-M provider binding requires the registered NativeRetrievalCore.")
    expected_fields = {
        "schema", "base_model_config", "native_tau_mm", "native_tau_me",
        "native_tau_q", "native_backend", "environment_value_reader",
    }
    if (set(config) != expected_fields or config.get("schema") != "native_retrieval_core_v1"
            or config.get("environment_value_reader") is not False
            or config.get("base_model_config", {}).get("architecture") != "A2-D"):
        raise ValueError("Native retrieval binding must retain the exact A2-D, no-E schema.")
    return core


def _expected_history_routing_state(recipe: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    if recipe["treatment"] == "D-control":
        return {
            "schema": "native_absolute_age_routing_state_v1",
            "treatment": "D-control",
            "backend": "full_direct",
            "absolute_epoch": epoch,
            "access": "full_A2-D_direct",
            "environment_value_reader": False,
        }
    strength = max(0.0, min(1.0, (epoch - 501) / 499.0))
    exploration = 1001 <= epoch <= 2000 and epoch % 8 == 0
    if exploration:
        strength = 0.0
    return {
        "schema": "native_absolute_age_routing_state_v1",
        "treatment": "G-native-M",
        "backend": recipe["native_backend"],
        "absolute_epoch": epoch,
        "exploration": exploration,
        "strength": strength,
        "tau_mm": strength * float(recipe["native_tau_mm"]),
        "tau_me": strength * float(recipe["native_tau_me"]),
        "tau_q": strength * float(recipe["native_tau_q"]),
        "environment_value_reader": False,
    }


def _validate_native_checkpoint_history(
    payload: Mapping[str, Any], identity: Mapping[str, Any], recipe: Mapping[str, Any],
) -> None:
    completed = payload["epoch"]
    rows = payload.get("history")
    cadence = identity.get("training_cadence")
    if not isinstance(rows, list) or not rows or not isinstance(cadence, Mapping):
        raise ValueError("Round7 checkpoint is missing its continuous per-epoch history and D0 cadence seal.")
    ages = [row.get("epoch") if isinstance(row, Mapping) else None for row in rows]
    expected_ages = list(range(501, completed + 1))
    if ages != expected_ages:
        raise ValueError("Round7 history must cover every child absolute epoch from e501 through the checkpoint age.")
    cases = cadence.get("train_case_visits_per_epoch")
    updates = cadence.get("optimizer_updates_per_epoch")
    group_sizes = cadence.get("case_counts_per_update")
    microbatch_cases = cadence.get("microbatch_cases")
    if (type(cases) is not int or cases <= 0 or type(updates) is not int or updates <= 0
            or not isinstance(group_sizes, list) or len(group_sizes) != updates
            or type(microbatch_cases) is not int or microbatch_cases <= 0
            or sum(group_sizes) != cases):
        raise ValueError("Round7 saved training cadence is malformed or differs from the D0 case/update contract.")
    expected_microbatches = sum((int(size) + microbatch_cases - 1) // microbatch_cases for size in group_sizes)
    for row in rows:
        if (row.get("case_visits") != cases or row.get("optimizer_updates") != updates
                or row.get("microbatches") != expected_microbatches
                or row.get("partial_update_cases") != group_sizes[-1]):
            raise ValueError("Round7 per-epoch cases, updates or microbatches differ from its sealed D0 schedule.")
        if row.get("native_routing_state") != _expected_history_routing_state(recipe, int(row["epoch"])):
            raise ValueError("Round7 history has a missing or inconsistent absolute-age routing state.")


def _validate_sampler_binding(
    payload: Mapping[str, Any], identity: Mapping[str, Any], provider: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
) -> None:
    from .unified_training import SamplingKey, _sampling_dataset_id

    source_id = _sampling_dataset_id(parent_identity["provider_identity"])
    child_id = _sampling_dataset_id(provider)
    expected = {
        "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
        "source_sampling_dataset_id": source_id,
        "matches_d0_e500_namespace": True,
        "arm": "A2-D",
        "fit_epoch_is_absolute": True,
        "sampler_epoch_mapping": "identity",
    }
    engine = identity.get("engine_config", {})
    sampler = payload.get("sampler_state")
    epoch = payload.get("epoch")
    if (source_id != child_id
            or identity.get("sampling_dataset_id") != child_id
            or identity.get("sampling_identity") != expected
            or engine.get("sampling_version") != SamplingKey.CASE_EPOCH_VERSION
            or engine.get("deterministic_case_order") is not True
            or not isinstance(sampler, Mapping)
            or sampler.get("completed_epoch") != epoch
            or sampler.get("next_epoch") != epoch + 1
            or sampler.get("absolute_sampling_epoch", epoch) != epoch):
        raise ValueError("Round7 checkpoint changed or detached from the exact D0 absolute case_epoch_v1 sampler.")


def bind_native_retrieval_provider(provider: Any, model: Any, *, dataset: str) -> Any:
    """Bind an existing D0 task to a native core under an explicit capability.

    This helper is intentionally limited to the retained segmented D0 recipe.
    It changes only the active model pointer and the Wind opt-in validator bit;
    all provider-owned data, transforms, targets, sampling and objective code
    remain those of the existing task.
    """
    if dataset not in ("thermal", "wind"):
        raise ValueError("Native retrieval provider binding supports only Thermal and Wind.")
    recipe = getattr(provider, "recipe", None)
    if (not isinstance(recipe, Mapping) or recipe.get("schema") != "content_hypergraph_recipe_v2"
            or recipe.get("protocol") != "source_read_development_v1"
            or recipe.get("source_reader_variant") != "D0"
            or recipe.get("dataset") != dataset or recipe.get("architecture") != "A2-D"):
        raise ValueError("Native retrieval provider binding is restricted to the exact segmented D0 task.")
    core = _retrieval_core(model, dataset)
    is_native = core.get_config().get("schema") == "native_retrieval_core_v1"

    if dataset == "thermal":
        from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
        from channelthermal.training.content_hypergraph_task_v2 import ThermalSourceReadTaskV1

        if not isinstance(provider, ThermalSourceReadTaskV1):
            raise TypeError("Thermal native retrieval requires the existing ThermalSourceReadTaskV1.")
        if (not isinstance(model, ThermalContentHypergraphAdapterV2) or provider.model is not model
                or provider.adapter is not model):
            raise ValueError("Thermal native retrieval must retain the provider's same native adapter object.")
        if model.core is not core or model.architecture != "A2-D":
            raise ValueError("Thermal native adapter/core binding differs from the source-reader A2-D contract.")
    else:
        from windfarm.training.content_hypergraph_task import (
            WindSourceReadTaskV1,
            _validate_native_wind_core_binding,
            _validate_wind_v2_core_binding,
        )

        if not isinstance(provider, WindSourceReadTaskV1):
            raise TypeError("Wind native retrieval requires the existing WindSourceReadTaskV1.")
        if provider.model is not model:
            old_config = getattr(provider.model, "get_config", dict)()
            expected_config = core.get_config().get("base_model_config", core.get_config())
            if old_config != expected_config:
                raise ValueError("Wind provider rebind is not a same-source A2-D to native A2-D replacement.")
            provider.model = model
        if is_native:
            _validate_native_wind_core_binding(provider.recipe, core.get_config())
            provider._allow_native_retrieval = True
        else:
            _validate_wind_v2_core_binding(provider.recipe, core.get_config())
            provider._allow_native_retrieval = False

    if provider.model is not model:
        raise RuntimeError("Native retrieval provider failed to bind the requested model object.")
    if not callable(getattr(provider, "predict_native", None)):
        raise TypeError("Native retrieval provider has no native prediction entrypoint.")
    return provider


def native_core_from_parent(parent_model: torch.nn.Module, *, dataset: str, **options: Any):
    """Construct the opt-in core by name-preserving copy from the exact D0 core."""
    from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
    from honf_forward_core.native_retrieval.core import NativeRetrievalCore

    parent = getattr(parent_model, "core", None) if dataset == "thermal" else parent_model
    if type(parent) is not ContentHypergraphCoreV2:
        raise TypeError("Round7 native branch requires the maintained D0 ContentHypergraphCoreV2 itself.")
    core = NativeRetrievalCore.from_a2d(parent, **options)
    source_state = parent.state_dict()
    target_state = core.state_dict()
    if set(target_state) != set(source_state) | {"native_training_epoch"}:
        raise ValueError("Native branch state differs beyond its single declared epoch buffer.")
    for name, value in source_state.items():
        if not torch.equal(target_state[name].detach().cpu(), value.detach().cpu()):
            raise ValueError(f"Native branch failed exact by-name D0 tensor copy at {name!r}.")
    if int(core.native_training_epoch) != 500:
        raise ValueError("Native branch must start at absolute e500 without changing inherited weights.")
    if dataset == "thermal":
        parent_model.core = core
        return parent_model
    return core


def load_native_retrieval_checkpoint(
    path: str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Reconstruct a Round7 checkpoint and native transforms without datasets."""
    import copy

    from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
    from honf_runtime.compat import load_trusted_checkpoint
    from honf_runtime.registry import require_model_family

    from .native_retrieval_protocol import (
        NATIVE_RETRIEVAL_FAMILY,
        NATIVE_RETRIEVAL_WORKFLOW,
        canonical_sha256,
        native_retrieval_child_declaration,
        validate_native_retrieval_recipe,
    )

    payload = load_trusted_checkpoint(path, map_location="cpu")
    identity = payload.get("experiment_identity", {})
    if (payload.get("workflow") != NATIVE_RETRIEVAL_WORKFLOW
            or not isinstance(identity, Mapping)
            or identity.get("workflow") != NATIVE_RETRIEVAL_WORKFLOW
            or identity.get("model_family") != NATIVE_RETRIEVAL_FAMILY
            or identity.get("family") != NATIVE_RETRIEVAL_FAMILY):
        raise ValueError("Checkpoint is not a sealed Round7 native-retrieval family member.")
    require_model_family(NATIVE_RETRIEVAL_FAMILY, "native_retrieval_evaluate")
    recipe = validate_native_retrieval_recipe(identity.get("recipe", {}))
    if (identity.get("recipe_sha256") != canonical_sha256(recipe)
            or identity.get("treatment") != recipe["treatment"]
            or identity.get("protocol") != recipe["protocol"]
            or payload.get("arm") != "A2-D"
            or type(payload.get("epoch")) is not int
            or not 500 <= int(payload["epoch"]) <= 2500):
        raise ValueError("Round7 family, treatment, arm, absolute age or recipe checksum is inconsistent.")
    branch = payload.get("branch_parent")
    contract = branch.get("native_retrieval_child") if isinstance(branch, Mapping) else None
    parent_identity = branch.get("identity") if isinstance(branch, Mapping) else None
    if (not isinstance(branch, Mapping)
            or branch.get("path") != recipe["source_checkpoint"]
            or branch.get("sha256") != recipe["source_checkpoint_sha256"]
            or branch.get("epoch") != 500
            or not isinstance(parent_identity, Mapping)
            or parent_identity.get("recipe_sha256") != recipe["source_recipe_sha256"]
            or canonical_sha256(parent_identity.get("provider_identity"))
            != recipe["source_provider_identity_sha256"]):
        raise ValueError("Round7 checkpoint is detached from its exact D0 e500 branch parent.")
    expected_contract = native_retrieval_child_declaration(recipe)
    if not isinstance(contract, Mapping):
        raise TypeError("Round7 checkpoint is missing its strict D0 continuation declaration.")
    for name, expected in expected_contract.items():
        if contract.get(name) != expected:
            raise ValueError(f"Round7 continuation declaration field {name!r} is inconsistent.")
    provider = identity.get("provider_identity")
    if not isinstance(provider, Mapping):
        raise TypeError("Round7 checkpoint lacks its native provider and transform binding.")
    if (not isinstance(parent_identity, Mapping)
            or not isinstance(parent_identity.get("provider_identity"), Mapping)):
        raise TypeError("Round7 checkpoint branch parent has no sealed D0 provider identity.")
    if recipe["treatment"] == "D-control":
        if canonical_sha256(provider) != canonical_sha256(parent_identity["provider_identity"]):
            raise ValueError("Round7 D-control changed its original D0 provider or transform identity.")
    else:
        from .native_retrieval_child import _provider_pair

        _provider_pair(parent_identity["provider_identity"], provider, recipe["dataset"])
    _validate_sampler_binding(payload, identity, provider, parent_identity)
    _validate_native_checkpoint_history(payload, identity, recipe)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("Round7 checkpoint has no model tensor state.")

    if recipe["dataset"] == "thermal":
        from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2

        adapter_state = provider.get("adapter_state")
        if not isinstance(adapter_state, Mapping):
            raise TypeError("Round7 Thermal checkpoint is missing its self-contained adapter payload.")
        model = ThermalContentHypergraphAdapterV2.from_state_payload(
            adapter_state, device=device, state_dict=state,
        )
        adapter = model
        observed_model_config = model.get_config()
    else:
        from windfarm.content_hypergraph import WindContentHypergraphAdapterV3

        core_config = provider.get("model_config")
        if not isinstance(core_config, Mapping):
            raise TypeError("Round7 Wind checkpoint is missing its core constructor config.")
        if recipe["treatment"] == "G-native-M":
            from honf_forward_core.native_retrieval.core import NativeRetrievalCore

            model = NativeRetrievalCore.from_config(core_config).to(device)
        else:
            if core_config.get("schema") != ContentHypergraphCoreV2.schema:
                raise ValueError("Round7 D-control must reconstruct the original A2-D core schema.")
            model = ContentHypergraphCoreV2.from_config(core_config).to(device)
        adapter_state = provider.get("adapter_state")
        if not isinstance(adapter_state, Mapping):
            raise TypeError("Round7 Wind checkpoint is missing its self-contained physical adapter payload.")
        adapter = WindContentHypergraphAdapterV3.from_state_payload(adapter_state, device=device)
        model.load_state_dict(state, strict=True)
        observed_model_config = model.get_config()
    if dict(observed_model_config) != dict(provider.get("model_config", {})):
        raise ValueError("Round7 reconstructed core/adapter differs from its sealed provider identity.")
    if recipe["treatment"] == "G-native-M":
        core = _retrieval_core(model, recipe["dataset"])
        epoch_key = "core.native_training_epoch" if recipe["dataset"] == "thermal" else "native_training_epoch"
        if epoch_key not in state or int(state[epoch_key]) != payload["epoch"]:
            raise ValueError("Round7 native absolute-age buffer is stale relative to the saved child history.")
        config = core.get_config()
        if any(config.get(name) != recipe[key] for name, key in (
            ("native_tau_mm", "native_tau_mm"),
            ("native_tau_me", "native_tau_me"),
            ("native_tau_q", "native_tau_q"),
            ("native_backend", "native_backend"),
        )) or config.get("environment_value_reader") is not False:
            raise ValueError("Round7 G-native-M checkpoint core differs from its exact sealed access law.")
        environment_representation = provider.get("environment_representation")
        if not isinstance(environment_representation, Mapping):
            raise ValueError("Primary G-native-M checkpoint lacks its native environment representation binding.")
        if recipe["dataset"] == "thermal":
            if (environment_representation.get("geometry_only") is not True
                    or environment_representation.get("heating_in_scene") is not False
                    or environment_representation.get("temperature_or_flow_target_features") is not False
                    or environment_representation.get("field_sensors", False) is not False):
                raise ValueError("Primary Thermal G-native-M environment representation exposes native value features.")
        elif (environment_representation.get("field_sensors") is not False
                or "geometry-only" not in str(environment_representation.get("measure", "")).casefold()):
            raise ValueError("Primary Wind G-native-M environment representation must bind geometry-only quadrature without value sensors.")
    model.eval()
    if hasattr(adapter, "eval"):
        adapter.eval()
    return model, adapter, copy.deepcopy(dict(payload))


__all__ = [
    "bind_native_retrieval_provider", "load_native_retrieval_checkpoint",
    "native_core_from_parent",
]
