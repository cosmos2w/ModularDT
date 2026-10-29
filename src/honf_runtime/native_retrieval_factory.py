"""Dataset-native Round7 task and TrainingEngine construction."""

from __future__ import annotations

import copy
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .native_retrieval_protocol import (
    NATIVE_RETRIEVAL_FAMILY,
    NATIVE_RETRIEVAL_WORKFLOW,
    canonical_sha256,
    native_retrieval_child_declaration,
    validate_native_retrieval_recipe,
)
from .native_retrieval_runtime import (
    bind_native_retrieval_provider,
    native_core_from_parent,
)
from .registry import require_model_family
from .unified_training import SamplingKey, _sampling_dataset_id


def configure_runtime_root(runtime_root: str | Path | None = None) -> Path:
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


def build_native_retrieval_task(
    recipe: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
    runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the exact D0 provider, then bind its source or native core."""
    value = validate_native_retrieval_recipe(recipe, verify_source_file=True)
    require_model_family(NATIVE_RETRIEVAL_FAMILY, "native_retrieval_train")
    project = configure_runtime_root(runtime_root)

    from content_hypergraph_train import build as build_d0_provider
    from content_hypergraph_train import engine_for as engine_for_d0

    from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2

    from .content_hypergraph import load_content_hypergraph_checkpoint
    from .native_retrieval_protocol import file_sha256

    source_path = Path(value["source_checkpoint"]).resolve(strict=True)
    source_model, source_adapter, source_payload = load_content_hypergraph_checkpoint(
        source_path, device=device,
    )
    source_identity = source_payload.get("experiment_identity", {})
    if (source_payload.get("epoch") != 500 or source_payload.get("arm") != "A2-D"
            or source_identity.get("recipe_sha256") != value["source_recipe_sha256"]):
        raise ValueError("Native task source differs from its exact D0 e500 checkpoint binding.")

    # Dataset construction stays with the maintained Round3 provider builder.
    fresh_model, provider = build_d0_provider(value["base_recipe"], device=str(device))
    fresh_provider_identity = dict(provider.identity_payload())
    expected_source_provider = source_identity.get("provider_identity")
    if (not isinstance(expected_source_provider, Mapping)
            or canonical_sha256(fresh_provider_identity) != canonical_sha256(expected_source_provider)):
        raise ValueError("Current native D0 provider/data/transforms do not match the exact e500 checkpoint binding.")
    source_sampling_dataset_id = _sampling_dataset_id(expected_source_provider)
    if source_identity.get("sampling_dataset_id") != source_sampling_dataset_id:
        raise ValueError("D0 source checkpoint sampling namespace is not the one derived from its TRAIN membership.")

    if value["dataset"] == "thermal":
        model = source_model
        provider.model = model
        provider.adapter = source_adapter
    else:
        model = source_model
        provider.model = model
        provider.adapter = source_adapter
    del fresh_model

    if value["treatment"] == "G-native-M":
        model = native_core_from_parent(
            model,
            dataset=value["dataset"],
            native_tau_mm=float(value["native_tau_mm"]),
            native_tau_me=float(value["native_tau_me"]),
            native_tau_q=float(value["native_tau_q"]),
            native_backend=str(value["native_backend"]),
            environment_value_reader=False,
        )
    else:
        core = getattr(model, "core", model)
        if type(core) is not ContentHypergraphCoreV2 or core.get_config().get("architecture") != "A2-D":
            raise ValueError("D-control must retain the exact original D0 A2-D function.")
    if value["treatment"] == "G-native-M":
        bind_native_retrieval_provider(provider, model, dataset=value["dataset"])
    else:
        # D-control is the original A2-D provider path. Keep its existing
        # provider implementation and validator active; only retarget the
        # references to the exact e500 model and transform objects above.
        if provider.model is not model:
            raise RuntimeError("D-control provider was not rebound to the exact original A2-D model.")
        if value["dataset"] == "thermal":
            if provider.adapter is not model:
                raise ValueError("D-control Thermal adapter must remain the exact original model object.")
        else:
            from windfarm.training.content_hypergraph_task import _validate_wind_v2_core_binding

            if provider.adapter is not source_adapter:
                raise ValueError("D-control Wind transform adapter must remain the exact source adapter object.")
            _validate_wind_v2_core_binding(provider.recipe, model.get_config())
            provider._allow_native_retrieval = False
    provider_identity = dict(provider.identity_payload())
    if value["treatment"] == "D-control" and canonical_sha256(provider_identity) != canonical_sha256(expected_source_provider):
        raise ValueError("D-control provider identity changed from its original D0 source.")
    if value["treatment"] == "G-native-M":
        from .native_retrieval_child import _provider_pair

        _provider_pair(expected_source_provider, provider_identity, value["dataset"])

    engine = engine_for_d0(value["base_recipe"], str(device))
    train_cases = tuple(provider.epoch_cases(501, engine.config.seed))
    update_slices = engine.epoch_update_slices(len(train_cases))
    provider_objective = provider_identity.get("objective", {})
    provider_training = provider_identity.get("training", {})
    if not isinstance(provider_objective, Mapping) or not isinstance(provider_training, Mapping):
        raise TypeError("D0 provider identity is missing its sealed objective or training cadence.")

    identity = {
        "family": NATIVE_RETRIEVAL_FAMILY,
        "model_family": NATIVE_RETRIEVAL_FAMILY,
        "workflow": NATIVE_RETRIEVAL_WORKFLOW,
        "protocol": value["protocol"],
        "treatment": value["treatment"],
        "recipe": copy.deepcopy(value),
        "recipe_sha256": canonical_sha256(value),
        "provider_identity": provider_identity,
        "native_access_contract": {
            "module_values": "original native token values only",
            "environment_values": "not consumed; native metadata and graph only",
            "environment_value_reader": False,
            "reader_backend": value["native_backend"],
        },
        "training_scope": {
            "parameter_scope": "all original A2-D parameters",
            "optimizer_state": "exact named AdamW continuation from D0 e500",
            "absolute_age": "source e500 plus child continuation e501 through e2500",
            "direct_environment_value_reader": False,
        },
        "sampling_identity": {
            "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            "source_sampling_dataset_id": source_sampling_dataset_id,
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
            "microbatch_cases": engine.config.microbatch_cases,
            "effective_cases": engine.config.effective_cases,
            "primary_queries_per_case": provider_training.get(
                "primary_queries_per_case", value["base_recipe"].get("primary_queries")),
            "response_cadence": provider_objective.get("response_cadence", "not_applicable"),
            "execution": copy.deepcopy(value["execution"]),
        },
        "loaded_source": _loaded_source_binding(project),
    }
    if engine.config.total_epochs != 2500 or engine.config.sampling_version != "case_epoch_v1":
        raise ValueError("Round7 must preserve the exact D0 2500-age case_epoch_v1 TrainingEngine schedule.")
    if (engine.config.microbatch_cases != int(value["base_recipe"]["microbatch_size"])
            or engine.config.effective_cases != int(value["base_recipe"]["effective_batch_size"])):
        raise ValueError("Round7 TrainingEngine batch sizes differ from the exact D0 source recipe.")
    # Keep the result's parent payload as a read-only audit reference. No new
    # checkpoint or training artifact is written during construction.
    return {
        "model": model,
        "provider": provider,
        "native_provider": provider,
        "provider_identity": provider_identity,
        "identity": identity,
        "engine": engine,
        "native_retrieval_child": native_retrieval_child_declaration(value),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": file_sha256(source_path),
        "source_payload": source_payload,
    }


__all__ = ["build_native_retrieval_task", "configure_runtime_root"]
