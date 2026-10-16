"""Strict, dataset-free loading for the new content-hypergraph family."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .compat import load_trusted_checkpoint
from .registry import require_model_family
from .source_read_protocol import configure_source_read_numerics


def _load_inherited_attention_checkpoint(
    payload: dict[str, Any], *, device: torch.device | str,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Reconstruct the opt-in IAG-C model and native transforms without datasets."""
    from .inherited_attention_protocol import (
        IAG_FAMILY,
        configure_inherited_attention_numerics,
        validate_inherited_attention_recipe,
    )

    identity = payload.get("experiment_identity", {})
    if payload.get("workflow") != "inherited_attention_joint_v1" or identity.get("workflow") != "inherited_attention_joint_v1":
        raise ValueError("IAG-C checkpoint workflow identity is missing or inconsistent.")
    if identity.get("model_family") != IAG_FAMILY or payload.get("arm") != "IAG-C":
        raise ValueError("IAG-C checkpoint family or training-arm identity is missing or inconsistent.")
    require_model_family(IAG_FAMILY, "inherited_attention_evaluate")
    recipe = identity.get("recipe")
    if not isinstance(recipe, dict):
        raise TypeError("IAG-C checkpoint lacks its self-contained recipe.")
    recipe = validate_inherited_attention_recipe(recipe)
    configure_inherited_attention_numerics(recipe["numerical_policy"])
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    if digest != identity.get("recipe_sha256"):
        raise ValueError("IAG-C checkpoint recipe checksum is inconsistent.")
    provider = identity.get("provider_identity")
    if not isinstance(provider, dict) or not isinstance(provider.get("model_config"), dict):
        raise TypeError("IAG-C checkpoint lacks its native model and transform inventory.")
    adapter_state = provider.get("adapter_state")
    if not isinstance(adapter_state, dict):
        raise TypeError("IAG-C checkpoint lacks its dataset-native adapter state.")
    expected_core_config = {
        "schema": "inherited_attention_core_v1",
        "architecture": "IAG-C",
        **dict(recipe["model"]),
    }
    if recipe["dataset"] == "thermal":
        saved_core_config = adapter_state.get("core")
    else:
        saved_core_config = provider.get("model_config")
    if not isinstance(saved_core_config, dict) or saved_core_config != expected_core_config:
        raise ValueError("IAG-C checkpoint core law/executor differs from the exact validated recipe model config.")

    if (recipe["protocol"] == "inherited_attention_development_v1"
            and recipe["normalization"] == "measure_entmax15"):
        if expected_core_config["query_executor"] != "indexed":
            raise ValueError("Sparse development IAG-C checkpoints must preserve the indexed query executor.")
        branch = payload.get("branch_parent")
        child_contract = (
            branch.get("inherited_attention_sparse_child") if isinstance(branch, dict) else None
        )
        if not isinstance(child_contract, dict):
            raise ValueError("Sparse development IAG-C checkpoints must retain their exact soft e500 branch contract.")
        executor_transition = child_contract.get("scene_executor_transition")
        source_executor = executor_transition.get("source") if isinstance(executor_transition, dict) else None
        target_executor = executor_transition.get("target") if isinstance(executor_transition, dict) else None
        valid_executor_transition = (
            source_executor in ("dense", "selected")
            and target_executor in ("dense", "selected")
            and not (source_executor == "selected" and target_executor == "dense")
            and executor_transition.get("same_probability_function") is True
        )
        checkpoint_epoch = payload.get("epoch")
        if (
            child_contract.get("schema") != "inherited_attention_sparse_child_v1"
            or child_contract.get("trajectory") != "soft_e1_500_then_absolute_sparse_transition_e501_800"
            or child_contract.get("source_epoch") != 500
            or child_contract.get("target_recipe_sha256") != identity.get("recipe_sha256")
            or target_executor != expected_core_config["scene_executor"]
            or not valid_executor_transition
            or type(checkpoint_epoch) is not int
            or checkpoint_epoch < 501
        ):
            raise ValueError("Sparse development IAG-C checkpoint no longer proves its sealed e500 transition law.")

    if recipe["dataset"] == "thermal":
        from channelthermal.content_hypergraph import ThermalInheritedAttentionAdapterV1

        model = ThermalInheritedAttentionAdapterV1.from_state_payload(adapter_state, device=device)
        adapter = model
    elif recipe["dataset"] == "wind":
        from windfarm.content_hypergraph import WindInheritedAttentionAdapterV1

        from honf_forward_core.inherited_attention.model import InheritedAttentionCoreV1

        model = InheritedAttentionCoreV1.from_config(provider["model_config"]).to(device)
        adapter = WindInheritedAttentionAdapterV1.from_state_payload(adapter_state, device=device)
    else:  # The recipe validator already rejects this, keep the branch closed.
        raise ValueError("Unsupported physical dataset in IAG-C checkpoint.")
    if model.get_config() != provider["model_config"]:
        raise ValueError("Reconstructed IAG-C core differs from checkpoint constructor metadata.")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise TypeError("IAG-C checkpoint has no model tensor state.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, adapter, payload


def load_content_hypergraph_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Load weights and native transforms without reading datasets or labels.

    Returns (model, physical_adapter, checkpoint). Thermal's physical adapter
    is the model containing the shared core; Wind returns the shared core and
    its native conversion adapter separately.
    """
    payload = load_trusted_checkpoint(path, map_location="cpu")
    if payload.get("workflow") == "inherited_attention_joint_v1":
        return _load_inherited_attention_checkpoint(payload, device=device)
    identity = payload.get("experiment_identity", {})
    workflow = payload.get("workflow")
    version = {"content_hypergraph_joint_v1": 1, "content_hypergraph_joint_v2": 2}.get(workflow)
    if version is None:
        raise ValueError("This loader accepts only the content-hypergraph joint family.")
    if identity.get("model_family") != "honf_content_hypergraph":
        raise ValueError("Checkpoint family identity is missing or inconsistent.")
    require_model_family("honf_content_hypergraph", "forward")
    recipe = identity.get("recipe")
    if not isinstance(recipe, dict) or recipe.get("schema") != f"content_hypergraph_recipe_v{version}":
        raise ValueError("Missing self-contained content-hypergraph recipe.")
    expected_arms = {1: {"A-D", "A-H6"}, 2: {"A2-D", "A2-H6"}}
    if recipe.get("architecture") not in expected_arms[version]:
        raise ValueError("Checkpoint architecture does not belong to its schema version.")
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    if digest != identity.get("recipe_sha256"):
        raise ValueError("Checkpoint recipe checksum is inconsistent.")
    if recipe.get("architecture") != payload.get("arm"):
        raise ValueError("Checkpoint architecture differs from its recorded training arm.")
    configure_source_read_numerics(recipe)
    provider = identity.get("provider_identity", {})
    if not isinstance(provider.get("model_config"), dict) or not isinstance(provider.get("adapter_state"), dict):
        raise TypeError("Checkpoint lacks its native model and TRAIN transform inventory.")
    if recipe.get("dataset") == "thermal":
        from channelthermal.content_hypergraph import ThermalContentHypergraphAdapter
        adapter_type = ThermalContentHypergraphAdapter
        if version == 2:
            from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
            adapter_type = ThermalContentHypergraphAdapterV2
        model = adapter_type.from_state_payload(provider["adapter_state"], device=device)
        adapter = model
    elif recipe.get("dataset") == "wind":
        from windfarm.content_hypergraph import WindContentHypergraphAdapter

        from honf_forward_core.content_hypergraph import ContentHypergraphCore
        if version == 1:
            model = ContentHypergraphCore(**provider["model_config"]).to(device)
            adapter = WindContentHypergraphAdapter.from_state_payload(provider["adapter_state"], device=device)
        else:
            from windfarm.content_hypergraph import WindContentHypergraphAdapterV3

            from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
            model = ContentHypergraphCoreV2.from_config(provider["model_config"]).to(device)
            adapter = WindContentHypergraphAdapterV3.from_state_payload(provider["adapter_state"], device=device)
    else:
        raise ValueError("Unsupported physical dataset in content-hypergraph checkpoint.")
    if model.get_config() != provider["model_config"]:
        raise ValueError("Reconstructed model differs from checkpoint constructor metadata.")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise TypeError("Checkpoint has no model tensor state.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, adapter, payload
