"""Round8 task construction over the exact segmented D0 providers and engine."""

from __future__ import annotations

import copy
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .native_budgeted_protocol import (
    NATIVE_BUDGETED_FAMILY,
    NATIVE_BUDGETED_WORKFLOW,
    canonical_sha256,
    file_sha256,
    native_budgeted_child_declaration,
    validate_native_budgeted_recipe,
)
from .native_budgeted_runtime import bind_budgeted_native_provider, budgeted_core_from_parent
from .registry import require_model_family
from .unified_training import SamplingKey, _sampling_dataset_id


def _configure_root(runtime_root: str | Path | None = None) -> Path:
    project = Path(runtime_root).resolve() if runtime_root is not None else Path(__file__).resolve().parents[2]
    for directory in (project / "src", project / "Case_ThermalChannel/src", project / "Case_WindFarm/src",
                      project / "tools"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    return project


def _loaded_source_binding(project: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_relative_to(project) or not path.is_file():
            continue
        files[path.relative_to(project).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"files": dict(sorted(files.items())), "sha256": canonical_sha256(files)}


def _validate_source_sampler(source_engine, provider_training: Mapping[str, Any], rng_policy: Mapping[str, Any]) -> None:
    """Bind query/case RNG to sampling_seed, independently of model initialization."""
    source_sampling_seed = provider_training.get("sampling_seed")
    if (source_engine.config.sampling_version != rng_policy["sampling_version"]
            or type(source_sampling_seed) is not int
            or source_engine.config.seed != source_sampling_seed):
        raise ValueError("Round8 sampler seed or case_epoch_v1 namespace differs from exact D0 training.")


def build_native_budgeted_task(
    recipe: Mapping[str, Any], *, device: torch.device | str = "cpu",
    runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build D8/G8 without materializing new data or changing the source provider."""
    value = validate_native_budgeted_recipe(
        recipe, verify_source_file=True, verify_admission=True,
    )
    require_model_family(NATIVE_BUDGETED_FAMILY, "native_budgeted_dispatch_train")
    project = _configure_root(runtime_root)
    from content_hypergraph_train import build as build_d0_provider
    from content_hypergraph_train import engine_for as engine_for_d0

    from .content_hypergraph import load_content_hypergraph_checkpoint

    source_path = Path(value["source_checkpoint"]).resolve(strict=True)
    source_model, source_adapter, source_payload = load_content_hypergraph_checkpoint(
        source_path, device=device,
    )
    source_identity = source_payload.get("experiment_identity", {})
    if (source_payload.get("epoch") != 500 or source_payload.get("arm") != "A2-D"
            or source_identity.get("recipe_sha256") != value["source_recipe_sha256"]
            or file_sha256(source_path) != value["source_checkpoint_sha256"]):
        raise ValueError("Round8 source differs from the exact D0 A2-D e500 checkpoint binding.")
    source_provider_identity = source_identity.get("provider_identity")
    if (not isinstance(source_provider_identity, Mapping)
            or canonical_sha256(source_provider_identity) != value["source_provider_identity_sha256"]):
        raise ValueError("Round8 source provider identity differs from the pinned D0 lineage.")

    fresh_model, provider = build_d0_provider(value["base_recipe"], device=str(device))
    fresh_identity = dict(provider.identity_payload())
    if canonical_sha256(fresh_identity) != canonical_sha256(source_provider_identity):
        raise ValueError("Current D0 data, transforms, sampling or objective differ from the exact e500 parent.")
    source_sampling_id = _sampling_dataset_id(source_provider_identity)
    if source_identity.get("sampling_dataset_id") != source_sampling_id:
        raise ValueError("D0 source checkpoint query namespace is not derived from its sealed TRAIN membership.")
    del fresh_model

    provider.model = source_model
    provider.adapter = source_adapter
    if value["treatment"] == "G-budgeted-M":
        model = budgeted_core_from_parent(source_model, dataset=value["dataset"], recipe=value)
        # Thermal wrapping mutates the provider's adapter in place, while the
        # Wind wrapper returns a new core and keeps its physical conversion
        # adapter as a separate object.
        provider.model = model
        bind_budgeted_native_provider(provider, model, dataset=value["dataset"], recipe=value)
    else:
        model = source_model
        if value["dataset"] == "thermal":
            if provider.model is not model or provider.adapter is not source_adapter:
                raise ValueError("D8 Thermal must retain the source adapter and D0 model objects.")
        else:
            if provider.model is not model or provider.adapter is not source_adapter:
                raise ValueError("D8 Wind must retain the source core and native conversion adapter objects.")

    provider_identity = dict(provider.identity_payload())
    if value["treatment"] == "D-control":
        if canonical_sha256(provider_identity) != canonical_sha256(source_provider_identity):
            raise ValueError("D8 changed the exact original D0 provider identity.")
    else:
        from .native_budgeted_runtime import _round8_provider_pair

        _round8_provider_pair(source_provider_identity, provider_identity, value["dataset"])
    source_engine = engine_for_d0(value["base_recipe"], str(device))
    if (source_engine.config.total_epochs != 2500
            or source_engine.config.sampling_version != SamplingKey.CASE_EPOCH_VERSION
            or not source_engine.config.deterministic_case_order):
        raise ValueError("Round8 must preserve the original 2500-age deterministic D0 case_epoch_v1 engine.")
    if (source_engine.config.microbatch_cases != int(value["base_recipe"]["microbatch_size"])
            or source_engine.config.effective_cases != int(value["base_recipe"]["effective_batch_size"])):
        raise ValueError("Round8 engine case/update batch differs from exact D0 source settings.")
    train_cases = tuple(provider.epoch_cases(501, source_engine.config.seed))
    update_slices = source_engine.epoch_update_slices(len(train_cases))
    provider_objective = provider_identity.get("objective", {})
    provider_training = provider_identity.get("training", {})
    if not isinstance(provider_objective, Mapping) or not isinstance(provider_training, Mapping):
        raise TypeError("Round8 provider identity lacks sealed objective or cadence metadata.")

    identity = {
        "family": NATIVE_BUDGETED_FAMILY,
        "model_family": NATIVE_BUDGETED_FAMILY,
        "workflow": NATIVE_BUDGETED_WORKFLOW,
        "protocol": value["protocol"],
        "treatment": value["treatment"],
        "recipe": copy.deepcopy(value),
        "recipe_sha256": canonical_sha256(value),
        "provider_identity": provider_identity,
        "native_access_contract": {
            "module_values": "one dispatched native seed list plus input-defined local protection",
            "environment_values": "not consumed; original contextual exchange and metadata retained",
            "environment_value_reader": False,
            "training_backend": value["native_backend"] if value["treatment"] == "G-budgeted-M" else "full_direct",
            "dense_hard_router_steps": "explicit Round8 training estimator; hard forward field",
        },
        "training_scope": {
            "parameter_scope": "all original A2-D parameters jointly; G8 adds scalar route calibration only",
            "optimizer_state": "exact named D0 AdamW continuation; one fresh named route group for G8",
            "absolute_age": "source e500 plus child continuation e501 through e2500",
            "direct_environment_value_reader": False,
        },
        "sampling_identity": {
            "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            "source_sampling_dataset_id": source_sampling_id,
            "matches_d0_e500_namespace": True,
            "arm": "A2-D",
            "fit_epoch_is_absolute": True,
            "sampler_epoch_mapping": "identity",
        },
        "training_cadence": {
            "absolute_start_epoch": 501,
            "train_case_visits_per_epoch": len(train_cases),
            "optimizer_updates_per_epoch": len(update_slices),
            "case_counts_per_update": [item.stop - item.start for item in update_slices],
            "microbatch_cases": source_engine.config.microbatch_cases,
            "effective_cases": source_engine.config.effective_cases,
            "primary_queries_per_case": provider_training.get("primary_queries_per_case"),
            "response_cadence": provider_objective.get("response_cadence", "not_applicable"),
            "execution": copy.deepcopy(value["execution"]),
        },
        "research_admission": copy.deepcopy(value["research_admission"]),
        "budget_objective": copy.deepcopy(value["budget_objective"]),
        "budget_schedule": copy.deepcopy(value["budget_schedule"]),
        "rng_policy": copy.deepcopy(value["rng_policy"]),
        "loaded_source": _loaded_source_binding(project),
    }
    _validate_source_sampler(source_engine, provider_training, value["rng_policy"])
    return {
        "model": model,
        "provider": provider,
        "provider_identity": provider_identity,
        "identity": identity,
        "engine": source_engine,
        "native_budgeted_child": native_budgeted_child_declaration(value),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": file_sha256(source_path),
        "source_payload": source_payload,
    }


__all__ = ["build_native_budgeted_task"]
