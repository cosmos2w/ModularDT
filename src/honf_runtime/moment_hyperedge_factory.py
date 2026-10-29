"""Native IAG-C task construction and self-contained Round5 checkpoint loading."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .moment_hyperedge_protocol import (
    MOMENT_ATTACHMENT_SCHEMA,
    MOMENT_CORE_SCHEMA,
    MOMENT_FAMILY,
    MOMENT_PARENT_EPOCH,
    MOMENT_PROTOCOL,
    MOMENT_VARIANTS,
    MOMENT_WORKFLOW,
    canonical_sha256,
    parent_recipe_from_moment_hyperedges,
    validate_moment_hyperedge_recipe,
)

_CORE_ONLY_KEYS = (
    "moment_variant",
    "moment_width",
    "processor_hidden_dim",
    "moment_init_seed",
)


def _normalize_round5_core_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Undo exactly the Round5 core/executor additions for parent comparison."""

    result = copy.deepcopy(dict(config))
    if (result.get("schema") != MOMENT_CORE_SCHEMA
            or result.get("architecture") != "IAG-C"
            or result.get("query_executor") != "indexed_projected"
            or result.get("collect_exact_work_counts") is not False
            or result.get("moment_variant") not in MOMENT_VARIANTS
            or result.get("moment_width") != 8
            or result.get("processor_hidden_dim") != 128
            or result.get("moment_init_seed") != 0):
        raise ValueError("Provider identity has an invalid Round5 projected moment-core amendment.")
    result["schema"] = "inherited_attention_core_v1"
    result["query_executor"] = "indexed"
    result.pop("collect_exact_work_counts")
    for name in _CORE_ONLY_KEYS:
        result.pop(name)
    return result


def _normalize_round5_recipe_model(model: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(model))
    if (result.get("schema") != MOMENT_CORE_SCHEMA
            or result.get("query_executor") != "indexed_projected"
            or result.get("collect_exact_work_counts") is not False
            or result.get("moment_variant") not in MOMENT_VARIANTS
            or result.get("moment_width") != 8
            or result.get("processor_hidden_dim") != 128
            or result.get("moment_init_seed") != 0):
        raise ValueError("Provider recipe does not contain the exact Round5 model amendment.")
    result.pop("schema")
    result["query_executor"] = "indexed"
    result.pop("collect_exact_work_counts")
    for name in _CORE_ONLY_KEYS:
        result.pop(name)
    return result


def normalize_moment_provider_identity(
    identity: Mapping[str, Any], *, dataset: str,
) -> dict[str, Any]:
    """Normalize only the explicitly versioned core fields in native identity."""

    result = copy.deepcopy(dict(identity))
    if dataset == "thermal":
        core_paths = (("model_config", "core"), ("adapter_state", "core"))
    elif dataset == "wind":
        core_paths = (("model_config",),)
    else:
        raise ValueError("Round5 native provider identity is restricted to Thermal or Wind.")
    for path in core_paths:
        cursor: Any = result
        for key in path[:-1]:
            cursor = cursor.get(key) if isinstance(cursor, Mapping) else None
        if not isinstance(cursor, Mapping) or not isinstance(cursor.get(path[-1]), Mapping):
            raise ValueError(f"Native provider identity is missing the declared core config at {path!r}.")
        cursor[path[-1]] = _normalize_round5_core_config(cursor[path[-1]])

    recipe = result.get("recipe")
    if not isinstance(recipe, Mapping) or not isinstance(recipe.get("model"), Mapping):
        raise ValueError("Native provider identity omits its exact IAG recipe and model config.")
    normalized_recipe = copy.deepcopy(dict(recipe))
    normalized_recipe["model"] = _normalize_round5_recipe_model(recipe["model"])
    result["recipe"] = normalized_recipe
    return result


def validate_moment_provider_pair(
    parent_identity: Mapping[str, Any],
    child_identity: Mapping[str, Any],
    *,
    dataset: str,
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Require native identity equality beyond Round5's enumerated core fields."""

    child_recipe = validate_moment_hyperedge_recipe(recipe)
    child_model = child_recipe["model"]
    normalized_parent = copy.deepcopy(dict(parent_identity))
    normalized_child = normalize_moment_provider_identity(child_identity, dataset=dataset)

    expected_model = {
        **{key: value for key, value in child_model.items()
           if key not in {"schema", "moment_variant", "moment_width", "processor_hidden_dim",
                          "moment_init_seed", "query_executor", "collect_exact_work_counts"}},
        "schema": MOMENT_CORE_SCHEMA,
        "architecture": "IAG-C",
        "query_executor": "indexed_projected",
        "collect_exact_work_counts": False,
        "moment_variant": child_recipe["variant"],
        "moment_width": 8,
        "processor_hidden_dim": 128,
        "moment_init_seed": 0,
    }
    if dataset == "thermal":
        for keypath in (("model_config", "core"), ("adapter_state", "core")):
            cursor: Any = child_identity
            for key in keypath[:-1]:
                cursor = cursor[key]
            if dict(cursor[keypath[-1]]) != expected_model:
                raise ValueError(f"Thermal provider core config at {keypath!r} differs from the Round5 recipe.")
    else:
        if dict(child_identity.get("model_config", {})) != expected_model:
            raise ValueError("Wind provider core config differs from the Round5 recipe.")

    normalized_parent_recipe = parent_recipe_from_moment_hyperedges(child_recipe)
    if normalized_child.get("recipe") != normalized_parent_recipe:
        raise ValueError("Round5 provider recipe changed a native parent field outside its enumerated core amendment.")
    if canonical_sha256(normalized_child) != canonical_sha256(normalized_parent):
        raise ValueError("Round5 provider identity changed membership, normalization, sampling, objective, or native transforms.")
    return {
        "schema": "moment_hyperedge_native_provider_binding_v1",
        "dataset": dataset,
        "source_provider_identity_sha256": canonical_sha256(normalized_parent),
        "child_provider_identity_sha256": canonical_sha256(dict(child_identity)),
        "normalized_child_provider_identity_sha256": canonical_sha256(normalized_child),
        "normalization": "only the Round5 core schema, variant, dimensions, init seed, projected executor and work-counter fields",
    }


def moment_hyperedge_attachment_declaration(recipe: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_moment_hyperedge_recipe(recipe)
    return {
        "schema": MOMENT_ATTACHMENT_SCHEMA,
        "family": MOMENT_FAMILY,
        "workflow": MOMENT_WORKFLOW,
        "protocol": MOMENT_PROTOCOL,
        "variant": value["variant"],
        "source_checkpoint": value["source_checkpoint"],
        "source_checkpoint_sha256": value["source_checkpoint_sha256"],
        "source_recipe_sha256": value["source_recipe_sha256"],
        "source_epoch": MOMENT_PARENT_EPOCH,
    }


def configure_runtime_root(runtime_root: str | Path) -> Path:
    root = Path(runtime_root).expanduser().resolve(strict=True)
    project = root / "HONF_Proj" if (root / "HONF_Proj").is_dir() else root
    for directory in (project / "src", project / "Case_ThermalChannel/src", project / "Case_WindFarm/src", project / "tools"):
        text = str(directory)
        if directory.is_dir() and text not in sys.path:
            sys.path.insert(0, text)
    return project


def build_moment_hyperedge_task(
    recipe: Mapping[str, Any],
    *,
    device: torch.device | str,
    parent_provider_identity: Mapping[str, Any],
) -> tuple[torch.nn.Module, Any, dict[str, Any], dict[str, Any]]:
    """Build the existing native IAG-C provider with the Round5 core schema."""

    value = validate_moment_hyperedge_recipe(recipe)
    from inherited_attention_train import build
    from .moment_hyperedge_protocol import native_recipe_for_moment_hyperedges

    model, provider = build(native_recipe_for_moment_hyperedges(value), device=str(device))
    identity = dict(provider.identity_payload())
    binding = validate_moment_provider_pair(
        parent_provider_identity,
        identity,
        dataset=value["dataset"],
        recipe=value,
    )
    binding.update({
        "source_checkpoint": value["source_checkpoint"],
        "source_checkpoint_sha256": value["source_checkpoint_sha256"],
        "source_recipe_sha256": value["source_recipe_sha256"],
        "source_epoch": MOMENT_PARENT_EPOCH,
        "variant": value["variant"],
    })
    return model, provider, identity, binding


def loaded_source_binding(project: Path) -> dict[str, Any]:
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


def load_moment_hyperedges_components(
    checkpoint: str | Path,
    *,
    runtime_root: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load the exact native Round5 model/provider pair for evaluation tools."""

    project = configure_runtime_root(runtime_root)
    from .compat import load_trusted_checkpoint

    path = Path(checkpoint).expanduser().resolve(strict=True)
    payload = load_trusted_checkpoint(path, map_location="cpu")
    identity = payload.get("experiment_identity")
    if (not isinstance(identity, Mapping) or payload.get("workflow") != MOMENT_WORKFLOW
            or identity.get("family") != MOMENT_FAMILY
            or identity.get("model_family") != MOMENT_FAMILY
            or identity.get("workflow") != MOMENT_WORKFLOW
            or identity.get("protocol") != MOMENT_PROTOCOL):
        raise ValueError("Checkpoint is not a sealed Round5 moment-hyperedge child.")
    recipe = validate_moment_hyperedge_recipe(identity.get("recipe", {}))
    if canonical_sha256(recipe) != identity.get("recipe_sha256"):
        raise ValueError("Round5 evaluation recipe checksum is inconsistent.")
    if payload.get("arm") != "IAG-C" or type(payload.get("epoch")) is not int:
        raise ValueError("Round5 evaluation requires an absolute-age native IAG-C checkpoint.")
    parent = payload.get("branch_parent")
    if (not isinstance(parent, Mapping) or parent.get("epoch") != MOMENT_PARENT_EPOCH
            or parent.get("path") != recipe["source_checkpoint"]
            or parent.get("sha256") != recipe["source_checkpoint_sha256"]
            or not isinstance(parent.get("identity"), Mapping)):
        raise ValueError("Round5 evaluation checkpoint lost its exact C-soft e500 parent binding.")
    attachment = parent.get("moment_hyperedge_child")
    declaration = moment_hyperedge_attachment_declaration(recipe)
    if not isinstance(attachment, Mapping) or any(
        attachment.get(name) != expected for name, expected in declaration.items()
    ):
        raise ValueError("Round5 evaluation checkpoint has an incompatible attachment declaration.")
    expected_provider = identity.get("provider_identity")
    if not isinstance(expected_provider, Mapping):
        raise ValueError("Round5 checkpoint has no native provider identity.")
    model, provider, observed_provider, binding = build_moment_hyperedge_task(
        recipe,
        device=device,
        parent_provider_identity=parent["identity"].get("provider_identity", {}),
    )
    if canonical_sha256(observed_provider) != canonical_sha256(expected_provider):
        raise ValueError("Round5 evaluation provider identity differs from the saved native binding.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    binding.update({
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_epoch": int(payload["epoch"]),
        "parent_epoch": MOMENT_PARENT_EPOCH,
        "recipe_sha256": identity["recipe_sha256"],
        "loaded_source": loaded_source_binding(project),
    })
    return {
        "model": model,
        "native_provider": provider,
        "recipe": recipe,
        "branch_binding": binding,
    }


__all__ = [
    "build_moment_hyperedge_task",
    "configure_runtime_root",
    "load_moment_hyperedges_components",
    "moment_hyperedge_attachment_declaration",
    "normalize_moment_provider_identity",
    "validate_moment_provider_pair",
]
