"""Strict, dataset-free loading for the new content-hypergraph family."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .compat import load_trusted_checkpoint
from .registry import require_model_family
from .source_read_protocol import configure_source_read_numerics


def _load_inherited_attention_checkpoint(
    payload: dict[str, Any], *, device: torch.device | str,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Reconstruct an opt-in IAG model and native transforms without datasets."""
    from .inherited_attention_protocol import (
        IAG_ARCHITECTURE_REPRESENTATION,
        IAG_C_ARCHITECTURE,
        IAG_FAMILY,
        IAG_PROTOCOL,
        IAG_Q_ARCHITECTURE,
        configure_inherited_attention_numerics,
        validate_inherited_attention_recipe,
    )

    identity = payload.get("experiment_identity", {})
    if payload.get("workflow") != "inherited_attention_joint_v1" or identity.get("workflow") != "inherited_attention_joint_v1":
        raise ValueError("Inherited-attention checkpoint workflow identity is missing or inconsistent.")
    require_model_family(IAG_FAMILY, "inherited_attention_evaluate")
    recipe = identity.get("recipe")
    if not isinstance(recipe, dict):
        raise TypeError("Inherited-attention checkpoint lacks its self-contained recipe.")
    recipe = validate_inherited_attention_recipe(recipe)
    configure_inherited_attention_numerics(recipe["numerical_policy"])
    architecture = recipe.get("architecture")
    representation = recipe.get("representation")
    expected_representations = IAG_ARCHITECTURE_REPRESENTATION
    if (architecture not in expected_representations
            or representation != expected_representations[architecture]
            or (architecture == IAG_Q_ARCHITECTURE and recipe.get("dataset") != "wind")):
        raise ValueError("Inherited-attention checkpoint has an unsupported dataset/representation pairing.")
    if identity.get("model_family") != IAG_FAMILY or payload.get("arm") != architecture:
        raise ValueError("Inherited-attention checkpoint family or training-arm identity is missing or inconsistent.")
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    if digest != identity.get("recipe_sha256"):
        raise ValueError("Inherited-attention checkpoint recipe checksum is inconsistent.")
    normalization_correction = identity.get("normalization_correction")
    correction_receipt = None
    correction_summary = None
    if normalization_correction is not None:
        from .inherited_attention_child import validate_inherited_attention_normalization_correction_resume

        validate_inherited_attention_normalization_correction_resume(
            normalization_correction, identity,
        )
        correction_receipt = json.loads(normalization_correction["receipt_json"])
        correction_summary = {
            "scope": correction_receipt["scope"],
            "receipt_sha256": normalization_correction["receipt_sha256"],
            "target_loaded_source_sha256": correction_receipt["target_loaded_source_sha256"],
        }
    provider = identity.get("provider_identity")
    if not isinstance(provider, dict) or not isinstance(provider.get("model_config"), dict):
        raise TypeError("Inherited-attention checkpoint lacks its native model and transform inventory.")
    if architecture == IAG_Q_ARCHITECTURE:
        if provider.get("architecture") != architecture or provider.get("representation") != representation:
            raise ValueError("IAG-Q provider identity differs from its recipe representation.")
        if (recipe["dataset"] == "wind"
                and provider.get("schema") != "wind_inherited_attention_provider_v1"):
            raise ValueError("Wind IAG-Q provider schema differs from its native v1 contract.")
    adapter_state = provider.get("adapter_state")
    if not isinstance(adapter_state, dict):
        raise TypeError("Inherited-attention checkpoint lacks its dataset-native adapter state.")
    expected_core_config = {
        "schema": "inherited_attention_core_v1",
        "architecture": architecture,
        **dict(recipe["model"]),
    }
    if recipe["dataset"] == "thermal":
        saved_core_config = adapter_state.get("core")
    else:
        saved_core_config = provider.get("model_config")
    if not isinstance(saved_core_config, dict) or saved_core_config != expected_core_config:
        raise ValueError("Inherited-attention core law/executor differs from the exact validated recipe model config.")

    representation_review = None
    if architecture == IAG_Q_ARCHITECTURE:
        representation_review = identity.get("representation_review")
        if not isinstance(representation_review, dict):
            raise TypeError("IAG-Q checkpoint lacks its sealed representation-review binding.")
        expected_review_keys = {
            "schema", "path", "sha256", "dataset", "decision", "fresh_run_dir",
        }
        review_path = representation_review.get("path")
        fresh_run_dir = representation_review.get("fresh_run_dir")
        review_sha256 = representation_review.get("sha256")
        if (
            set(representation_review) != expected_review_keys
            or representation_review.get("schema") != "inherited_attention_representation_review_v1"
            or representation_review.get("dataset") != "wind"
            or representation_review.get("decision") != "replace_iag_c_with_iag_q"
            or not isinstance(review_path, str)
            or not Path(review_path).is_absolute()
            or str(Path(review_path).resolve()) != review_path
            or not isinstance(fresh_run_dir, str)
            or not Path(fresh_run_dir).is_absolute()
            or str(Path(fresh_run_dir).resolve()) != fresh_run_dir
            or not isinstance(review_sha256, str)
            or len(review_sha256) != 64
            or any(character not in "0123456789abcdef" for character in review_sha256)
        ):
            raise ValueError("IAG-Q representation-review identity is incomplete or noncanonical.")

    if (recipe["protocol"] == IAG_PROTOCOL
            and recipe["normalization"] == "measure_entmax15"):
        if architecture == IAG_C_ARCHITECTURE and expected_core_config["query_executor"] != "indexed":
            raise ValueError("Sparse development IAG-C checkpoints must preserve the indexed query executor.")
        if architecture == IAG_Q_ARCHITECTURE and (
            expected_core_config.get("scene_executor") != "cached_tokens"
            or expected_core_config.get("query_executor") != "cell_selected"
        ):
            raise ValueError("Sparse development IAG-Q checkpoints must preserve cached-token/cell-selected execution.")
        branch = payload.get("branch_parent")
        child_contract = (
            branch.get("inherited_attention_sparse_child") if isinstance(branch, dict) else None
        )
        if not isinstance(child_contract, dict):
            raise ValueError("Sparse development IAG checkpoints must retain their exact soft e500 branch contract.")
        if (correction_summary is not None
                and child_contract.get("normalization_correction") != correction_summary):
            raise ValueError(
                "Sparse child branch correction summary differs from its receipt-bound checkpoint identity."
            )
        if (correction_receipt is not None
                and (correction_receipt.get("source_checkpoint") != branch.get("path")
                     or correction_receipt.get("source_checkpoint_sha256") != branch.get("sha256"))):
            raise ValueError("Normalization-correction receipt differs from its exact e500 checkpoint branch parent.")
        executor_transition = child_contract.get("scene_executor_transition")
        source_executor = executor_transition.get("source") if isinstance(executor_transition, dict) else None
        target_executor = executor_transition.get("target") if isinstance(executor_transition, dict) else None
        checkpoint_epoch = payload.get("epoch")
        if architecture == IAG_C_ARCHITECTURE:
            valid_executor_transition = (
                source_executor in ("dense", "selected")
                and target_executor in ("dense", "selected")
                and not (source_executor == "selected" and target_executor == "dense")
                and executor_transition.get("same_probability_function") is True
            )
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
            if correction_receipt is not None:
                branch = payload.get("branch_parent")
                parent_identity = branch.get("identity") if isinstance(branch, dict) else None
                if not isinstance(parent_identity, dict):
                    raise ValueError("Corrected sparse IAG-C checkpoint is missing its sealed soft e500 parent identity.")
                from .inherited_attention_child import _require_same_soft_identity

                try:
                    observed_transition = _require_same_soft_identity(parent_identity, identity)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Corrected sparse IAG-C parent and child identities differ beyond their sealed transition."
                    ) from exc
                if observed_transition != executor_transition:
                    raise ValueError("IAG-C branch transition differs from its exact parent and child identities.")
        else:
            parent = branch if isinstance(branch, dict) else {}
            parent_identity = parent.get("identity")
            parent_identity = parent_identity if isinstance(parent_identity, dict) else {}
            parent_recipe = parent_identity.get("recipe")
            parent_recipe = parent_recipe if isinstance(parent_recipe, dict) else {}
            parent_model = parent_recipe.get("model")
            parent_model = parent_model if isinstance(parent_model, dict) else {}
            parent_provider = parent_identity.get("provider_identity")
            parent_provider = parent_provider if isinstance(parent_provider, dict) else {}
            parent_review = parent_identity.get("representation_review")
            source_saved_config = parent_provider.get("model_config")
            source_saved_config = source_saved_config if isinstance(source_saved_config, dict) else {}
            required_q_binding = {
                "source_architecture": IAG_Q_ARCHITECTURE,
                "source_representation": "iag_q",
                "target_architecture": IAG_Q_ARCHITECTURE,
                "target_representation": "iag_q",
            }
            valid_q_transition = (
                isinstance(executor_transition, dict)
                and executor_transition.get("source") == "cached_tokens"
                and executor_transition.get("target") == "cached_tokens"
                and executor_transition.get("same_probability_function") is True
            )
            if (
                recipe.get("dataset") != "wind"
                or recipe.get("normalization") != "measure_entmax15"
                or recipe.get("trajectory") != "e500_sparse_child"
                or child_contract.get("schema") != "inherited_attention_sparse_child_v1"
                or child_contract.get("trajectory") != "soft_e1_500_then_absolute_sparse_transition_e501_800"
                or any(child_contract.get(key) != value for key, value in required_q_binding.items())
                or child_contract.get("source_epoch") != 500
                or child_contract.get("target_recipe_sha256") != identity.get("recipe_sha256")
                or child_contract.get("source_recipe_sha256") != parent_identity.get("recipe_sha256")
                or child_contract.get("representation_review_sha256") != representation_review.get("sha256")
                or child_contract.get("source_checkpoint_sha256") != parent.get("sha256")
                or child_contract.get("source_checkpoint") != parent.get("path")
                or parent.get("epoch") != 500
                or parent_identity.get("workflow") != "inherited_attention_joint_v1"
                or parent_identity.get("model_family") != IAG_FAMILY
                or parent_recipe.get("dataset") != "wind"
                or parent_recipe.get("protocol") != IAG_PROTOCOL
                or parent_recipe.get("architecture") != IAG_Q_ARCHITECTURE
                or parent_recipe.get("representation") != "iag_q"
                or parent_recipe.get("normalization") != "soft"
                or parent_recipe.get("trajectory") != "fresh_soft"
                or parent_model.get("normalization") != "soft"
                or parent_provider.get("schema") != "wind_inherited_attention_provider_v1"
                or parent_provider.get("architecture") != IAG_Q_ARCHITECTURE
                or parent_provider.get("representation") != "iag_q"
                or source_saved_config.get("architecture") != IAG_Q_ARCHITECTURE
                or source_saved_config.get("normalization") != "soft"
                or source_saved_config.get("scene_executor") != "cached_tokens"
                or source_saved_config.get("query_executor") != "cell_selected"
                or parent_review != representation_review
                or child_contract.get("exact_parameter_layout_verified") is not True
                or not valid_q_transition
                or type(checkpoint_epoch) is not int
                or checkpoint_epoch < 501
            ):
                raise ValueError("Sparse IAG-Q checkpoint does not prove its exact reviewed IAG-Q e500 parent branch.")
            from .inherited_attention_child import _require_same_soft_identity

            try:
                observed_transition = _require_same_soft_identity(parent_identity, identity)
            except (TypeError, ValueError) as exc:
                raise ValueError("Sparse IAG-Q parent and child identities differ beyond their sealed transition.") from exc
            if observed_transition != executor_transition:
                raise ValueError("Sparse IAG-Q branch transition differs from its exact parent and child identities.")

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
        raise ValueError("Unsupported physical dataset in inherited-attention checkpoint.")
    if model.get_config() != provider["model_config"]:
        raise ValueError("Reconstructed inherited-attention core differs from checkpoint constructor metadata.")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise TypeError("Inherited-attention checkpoint has no model tensor state.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, adapter, payload


def _load_moment_hyperedge_checkpoint(
    payload: dict[str, Any], *, device: torch.device | str,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Reconstruct a Round5 child from its self-contained native bindings."""

    from .moment_hyperedge_factory import (
        moment_hyperedge_attachment_declaration,
        validate_moment_provider_pair,
    )
    from .moment_hyperedge_protocol import (
        MOMENT_FAMILY,
        MOMENT_WORKFLOW,
        canonical_sha256,
        validate_moment_hyperedge_recipe,
    )
    from .moment_hyperedge_child import verify_moment_hyperedge_resume

    identity = payload.get("experiment_identity", {})
    if payload.get("workflow") != MOMENT_WORKFLOW or identity.get("workflow") != MOMENT_WORKFLOW:
        raise ValueError("Moment-hyperedge checkpoint workflow identity is missing or inconsistent.")
    require_model_family(MOMENT_FAMILY, "moment_hyperedges_evaluate")
    recipe = validate_moment_hyperedge_recipe(identity.get("recipe", {}))
    if (identity.get("family") != MOMENT_FAMILY or identity.get("model_family") != MOMENT_FAMILY
            or identity.get("protocol") != recipe["protocol"]
            or identity.get("variant") != recipe["variant"]
            or identity.get("recipe_sha256") != canonical_sha256(recipe)
            or payload.get("arm") != "IAG-C"):
        raise ValueError("Moment-hyperedge family, variant, arm, or recipe identity is inconsistent.")
    declaration = moment_hyperedge_attachment_declaration(recipe)
    verify_moment_hyperedge_resume(payload, recipe, declaration)

    branch = payload["branch_parent"]
    parent_identity = branch["identity"]
    provider = identity.get("provider_identity")
    parent_provider = parent_identity.get("provider_identity")
    if not isinstance(provider, Mapping) or not isinstance(parent_provider, Mapping):
        raise TypeError("Moment-hyperedge checkpoint lacks its native provider bindings.")
    validate_moment_provider_pair(
        parent_provider, provider, dataset=recipe["dataset"], recipe=recipe,
    )
    expected_core_config = {
        "schema": "scene_prepared_moment_hyperedge_core_v1",
        "architecture": "IAG-C",
        **dict(recipe["model"]),
    }
    adapter_state = provider.get("adapter_state")
    if not isinstance(adapter_state, dict):
        raise TypeError("Moment-hyperedge checkpoint lacks its native adapter state.")
    if recipe["dataset"] == "thermal":
        from channelthermal.content_hypergraph import ThermalInheritedAttentionAdapterV1

        if (adapter_state.get("core") != expected_core_config
                or provider.get("model_config", {}).get("core") != expected_core_config):
            raise ValueError("Thermal moment core config differs from the exact Round5 recipe.")
        model = ThermalInheritedAttentionAdapterV1.from_state_payload(adapter_state, device=device)
        adapter = model
    elif recipe["dataset"] == "wind":
        from windfarm.content_hypergraph import WindInheritedAttentionAdapterV1
        from honf_forward_core.inherited_attention.moment_model import MomentHyperedgeCoreV1

        if dict(provider.get("model_config", {})) != expected_core_config:
            raise ValueError("Wind moment core config differs from the exact Round5 recipe.")
        model = MomentHyperedgeCoreV1.from_config(expected_core_config).to(device)
        adapter = WindInheritedAttentionAdapterV1.from_state_payload(adapter_state, device=device)
    else:
        raise ValueError("Unsupported physical dataset in moment-hyperedge checkpoint.")
    if model.get_config() != provider.get("model_config"):
        raise ValueError("Reconstructed moment core/adapter differs from checkpoint constructor metadata.")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise TypeError("Moment-hyperedge checkpoint has no model tensor state.")
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
    if payload.get("workflow") == "joint_native_train_v1":
        from .joint_native_runtime import load_joint_native_checkpoint

        return load_joint_native_checkpoint(str(path), device=device)
    if payload.get("workflow") == "native_retrieval_joint_v1":
        from .native_retrieval_runtime import load_native_retrieval_checkpoint

        return load_native_retrieval_checkpoint(str(path), device=device)
    if payload.get("workflow") == "native_budgeted_dispatch_joint_v2":
        from .native_budgeted_runtime import load_native_budgeted_checkpoint

        return load_native_budgeted_checkpoint(str(path), device=device)
    if payload.get("workflow") == "moment_hyperedges_child_v1":
        return _load_moment_hyperedge_checkpoint(payload, device=device)
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
