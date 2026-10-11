"""Strict, self-contained migration for the mature Wind Run2202 direct path.

This is a compatibility path for one named historical checkpoint. The mapped
P state is the deployed direct computation; copied source optimizer and
training metadata are archival provenance and are not a P resume state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from honf_runtime.compat import load_trusted_checkpoint

from .interaction_preserving import InteractionPreservingWindModel
from .normalization import VelocityNormalizer, VerticalProfileBaseline
from .shared_interaction import WindFarmRefinedInteractionModel

RUN2202_RUN_ID = "Run_2202_20261008_194400_w3_h128_full_train420_q4096_e5000"
RUN2202_SOURCE_WORKFLOW = "unified_interaction_refinement_formal_v1"
RUN2202_SOURCE_FAMILY = "source_resolved_nonlinear_refined"
RUN2202_MIGRATION_SCHEMA = "wind_run2202_direct_p_compatibility_migration_v1"
RUN2202_MIGRATION_LABEL = "Run2202-trained compatibility migration"

RUN2202_EPOCH_CHECKPOINT_SHA256 = {
    1000: "fb04adf28069e71637439eb5e0b86f3dcc61e36fd296d28b297361c07cf1068d",
    2000: "b2e5c6aaab7ab2200f9a2b2db9b1b4060377546d7f025ffaec4b16ad442a2034",
    5000: "5f301944c453c1b319a9cf5c5edc0aaa830802cb119c34ae40907146ff725f08",
}
RUN2202_NORMALIZATION_SHA256 = "8c8d776696074216cd691a9d159899ade5b10135426b709deb19e9251946cd01"
RUN2202_BACKGROUND_PROFILE_SHA256 = "7387c7c9564d01be11c02f0a38408cde13366aeab7d186d21c97fd56cce798ff"
RUN2202_DIRECT_STATE_SHA256 = "5957c1ca8fb62928ba7be2bd26fd7055d7d9323b4f9d17b236c9aebf6f7d9842"
RUN2202_IDENTITY_FILE_SHA256 = "55e54708f755e1792badc02b8e67274f42bfcdc7a80797a8112dfca7372bc4dc"
RUN2202_SOURCE_STATE_FILE_SHA256 = "746e90b5781f6ccd7ca6558e4d259ad3d365715df91d055628efbc2905fc93ce"
RUN2202_NORMALIZATION_DOCUMENT_SHA256 = "041e4b785d3687d954912eb74212c03b9fb8eeda69e26360eb40097db6d79df1"
RUN2202_SOURCE_STATE_DOCUMENT_SHA256 = "6fd7add46a80d24a0e72f88a48c46fd709fff1563b403dcd0d87ff5206080aef"

RUN2202_RECIPE_SHA256 = "a1b6c6691a7e53e8bc6fb4976c809ea69b00bf7deba6c4ed89860b2fc74feedb"
RUN2202_FORMAL_RECIPE_SHA256 = "3d456f34d6b976fa8c0bf61d6f07f7e70718b5e1d4c2840e7dba4eabb0687b4b"
RUN2202_RESOLVED_RECIPE_SHA256 = "fc749ada338460fb1e003eef1437c3ec3f7887baf5cd1cb61739b983b702a2fa"
RUN2202_TRAIN_ROWS_SHA256 = "a2170bb15349462160679632fc074d7f685d7e5d17d09aeeab1797a824c8a282"
RUN2202_VALIDATION_ROWS_SHA256 = "ed7295dd2650687c599c13e29a6bae3492508ec8346517e39f4cacf60d0eff72"
RUN2202_NORMALIZATION_BINDING_SHA256 = "9dc601e4c9954f42706501e78b2a7fecfd4062b4183fb613da93babc89922705"
RUN2202_SOURCE_METADATA_SHA256 = "eac5760d12dbc4329214f7d39df979082f871bc247e52dda57e340a73c68ee1d"

_ALLOWED_SOURCE_EPOCHS = frozenset((1000, 2000, 5000))
_SOURCE_MODEL_CONFIG = {
    "base_width": 16,
    "environment_token_shape": [2, 2, 2],
    "forward_architecture": "source_resolved_nonlinear_refined",
    "hidden": 128,
    "max_sources": 30,
    "message": 128,
    "router_hidden": 32,
}
_EXPECTED_ENVIRONMENT_REPRESENTATION = {
    "token_shape": [2, 2, 2],
    "record_count": 8,
    "measure": "geometry-only support-box quadrature in D^3",
}


@dataclass(frozen=True)
class Run2202DirectModel:
    """A strict old-to-P state mapping for one saved Run2202 checkpoint."""

    model: InteractionPreservingWindModel
    model_config: dict[str, Any]
    mapped_state_dict: dict[str, torch.Tensor]
    mapped_state_sha256: str
    source_checkpoint_payload: Mapping[str, Any]
    source_identity: Mapping[str, Any]
    source_model_state_dict: Mapping[str, torch.Tensor]
    normalizer: VelocityNormalizer
    profile: VerticalProfileBaseline


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise TypeError(f"State value {key!r} is not a tensor.")
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def _state_manifest(state: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Source model state value {key!r} is not a tensor.")
        manifest[key] = {"dtype": str(value.dtype), "shape": list(value.shape)}
    return manifest


def _validate_embedded_normalizer_binding(
    normalization_document: Mapping[str, Any],
    model_config: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    normalization_document_sha256: str,
) -> None:
    """Bind the archival normalizer to P config and the checkpoint-bound profile."""
    if stable_json_sha256(normalization_document) != normalization_document_sha256:
        raise ValueError("Run2202 embedded normalization document digest mismatch.")
    if stable_json_sha256(normalization_document) != RUN2202_NORMALIZATION_DOCUMENT_SHA256:
        raise ValueError("Run2202 embedded normalization document differs from the exact TRAIN artifact.")
    if normalization_document.get("schema_version") != 1:
        raise ValueError("Run2202 embedded normalization schema version must be exactly 1.")
    normalizer = {
        key: value for key, value in normalization_document.items() if key != "vertical_profile_baseline"
    }
    if (
        normalization_document.get("vertical_profile_baseline") != profile
        or model_config.get("background_profile") != profile
        or model_config.get("velocity_transform") != normalizer
    ):
        raise ValueError("Run2202 P config, embedded TRAIN normalizer and saved profile do not agree.")


def _expected_run2202_p_config(
    normalization_document: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Construct the only accepted P config from the pinned source identity."""
    normalizer_payload = {
        key: value for key, value in normalization_document.items() if key != "vertical_profile_baseline"
    }
    normalizer = VelocityNormalizer.from_dict(dict(normalizer_payload))
    background_profile = VerticalProfileBaseline.from_dict(dict(profile))
    if normalizer.to_dict() != normalizer_payload or background_profile.to_dict() != dict(profile):
        raise ValueError("Run2202 TRAIN transforms do not round-trip under their maintained schema.")
    return InteractionPreservingWindModel(
        velocity_transform=normalizer,
        background_profile=background_profile,
        mode="P",
        hidden=128,
        message=128,
        regional_anchors=0,
        depth=2,
        collective_width=64,
        receiver_tile=512,
        seed=seed,
        max_sources=30,
        locality_prior_strength=None,
        factorized_affine_execution=False,
    ).export_config()


def _validate_run2202_p_config(
    config: Mapping[str, Any],
    normalization_document: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    seed: int,
) -> None:
    expected_config = _expected_run2202_p_config(normalization_document, profile, seed=seed)
    if dict(config) != expected_config:
        raise ValueError(
            "Run2202 P config differs from the literal source-bound constructor, E8 tokenization, "
            "receiver features or output law."
        )


def map_run2202_direct_state(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply the existing 52-key direct bridge with strict key/shape/dtype checks."""

    if not isinstance(source_state, Mapping) or not isinstance(target_state, Mapping):
        raise TypeError("Run2202 source and P target states must be mappings.")
    mapping = {
        key: value
        for key, value in source_state.items()
        if key in target_state and key.startswith("core.") and not key.startswith("core.refinement.")
    }
    if len(mapping) != 52 or set(mapping) != set(target_state):
        missing = sorted(set(target_state) - set(mapping))
        extra = sorted(set(mapping) - set(target_state))
        raise ValueError(
            "Run2202 direct state must cover exactly the existing 52-key P bridge; "
            f"mapped={len(mapping)}, target={len(target_state)}, missing={missing}, extra={extra}."
        )
    for key, value in mapping.items():
        target = target_state[key]
        if not torch.is_tensor(value) or not torch.is_tensor(target):
            raise TypeError(f"Run2202 direct state entry {key!r} must be a tensor on both sides.")
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"Run2202 direct tensor {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(target.shape)}."
            )
        if value.dtype != target.dtype:
            raise ValueError(f"Run2202 direct tensor {key!r} dtype {value.dtype} differs from {target.dtype}.")
    return mapping


def _validate_source_identity(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], int]:
    if payload.get("checkpoint_schema_version") != 1:
        raise ValueError("Run2202 checkpoint schema version must be 1.")
    if payload.get("workflow") != "unified_interaction_refinement":
        raise ValueError("Source checkpoint is not a unified Run2202 refinement checkpoint.")
    epoch = payload.get("epoch")
    if type(epoch) is not int or epoch not in _ALLOWED_SOURCE_EPOCHS or payload.get("current_epoch") != epoch:
        raise ValueError("Run2202 migration supports only literal saved epochs 1000, 2000, or 5000.")
    identity = payload.get("experiment_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Run2202 checkpoint lacks its experiment identity.")
    expected_identity = {
        "workflow": RUN2202_SOURCE_WORKFLOW,
        "task": "wind",
        "run_identity": RUN2202_RUN_ID,
        "recipe_sha256": RUN2202_RECIPE_SHA256,
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Source checkpoint experiment identity differs from literal Run2202.")
    engine = identity.get("engine_config")
    if not isinstance(engine, Mapping) or engine.get("total_epochs") != 5000 or engine.get("seed") != 42:
        raise ValueError("Run2202 source schedule or initialization seed is not the saved 5,000-epoch seed-42 recipe.")
    provider = identity.get("provider_identity")
    if not isinstance(provider, Mapping):
        raise TypeError("Run2202 experiment identity lacks its provider identity.")
    expected_provider = {
        "dataset": "WindFarm",
        "family": RUN2202_SOURCE_FAMILY,
        "dataset_scope": "formal_full_original_train",
        "formal_recipe_id": "windfarm_original_seed42_fulltrain_refinement_v1",
        "formal_recipe_sha256": RUN2202_FORMAL_RECIPE_SHA256,
        "resolved_recipe_sha256": RUN2202_RESOLVED_RECIPE_SHA256,
        "normalization_sha256": RUN2202_NORMALIZATION_SHA256,
        "normalization_binding_sha256": RUN2202_NORMALIZATION_BINDING_SHA256,
        "source_metadata_sha256": RUN2202_SOURCE_METADATA_SHA256,
        "train_rows_sha256": RUN2202_TRAIN_ROWS_SHA256,
        "validation_rows_sha256": RUN2202_VALIDATION_ROWS_SHA256,
        "directions_deg": [270.0, 285.0, 300.0],
        "model": _SOURCE_MODEL_CONFIG,
        "environment_representation": _EXPECTED_ENVIRONMENT_REPRESENTATION,
    }
    if any(provider.get(key) != value for key, value in expected_provider.items()):
        mismatches = {
            key: (provider.get(key), value)
            for key, value in expected_provider.items()
            if provider.get(key) != value
        }
        raise ValueError(f"Run2202 provider identity differs from its frozen training lineage: {mismatches}.")
    if len(provider.get("train_layout_indices", ())) != 140:
        raise ValueError("Run2202 source identity must retain all 140 full-TRAIN layouts.")
    if len(provider.get("validation_layout_indices", ())) != 30:
        raise ValueError("Run2202 source identity must retain its 30-layout validation partition.")
    if payload.get("arm") != "full_detail":
        raise ValueError("Run2202 source checkpoint arm is not the mature full-detail model.")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping) or not model_state:
        raise TypeError("Run2202 source checkpoint lacks its learned model state.")
    return identity, provider, epoch


def build_run2202_direct_model(
    checkpoint_payload: Mapping[str, Any],
    normalization_document: Mapping[str, Any],
) -> Run2202DirectModel:
    """Strictly load one saved Run2202 age and map its existing direct tensors to P."""

    identity, provider, _epoch = _validate_source_identity(checkpoint_payload)
    if not isinstance(normalization_document, Mapping):
        raise TypeError("Run2202 normalization document must be a mapping.")
    if stable_json_sha256(provider["background_profile"]) != RUN2202_BACKGROUND_PROFILE_SHA256:
        raise ValueError("Run2202 TRAIN height profile differs from the sealed profile identity.")
    if normalization_document.get("vertical_profile_baseline") != provider["background_profile"]:
        raise ValueError("Embedded normalization and source identity do not contain the same TRAIN height profile.")
    normalizer_payload = {
        key: value for key, value in normalization_document.items() if key != "vertical_profile_baseline"
    }
    normalizer = VelocityNormalizer.from_dict(dict(normalizer_payload))
    profile = VerticalProfileBaseline.from_dict(dict(provider["background_profile"]))
    if normalizer.to_dict() != normalizer_payload or profile.to_dict() != dict(provider["background_profile"]):
        raise ValueError("Run2202 normalizer/profile do not round-trip under the maintained TRAIN schema.")

    old_model = WindFarmRefinedInteractionModel(
        velocity_transform=normalizer,
        background_profile=profile,
        hidden=int(_SOURCE_MODEL_CONFIG["hidden"]),
        message=int(_SOURCE_MODEL_CONFIG["message"]),
        max_sources=int(_SOURCE_MODEL_CONFIG["max_sources"]),
        base_width=int(_SOURCE_MODEL_CONFIG["base_width"]),
        router_hidden=int(_SOURCE_MODEL_CONFIG["router_hidden"]),
    ).cpu().eval()
    source_state = checkpoint_payload["model_state_dict"]
    old_load = old_model.load_state_dict(source_state, strict=True)
    if old_load.missing_keys or old_load.unexpected_keys:
        raise ValueError("Run2202 source weights failed strict loading into their original Full model family.")

    seed = int(identity["engine_config"]["seed"])
    model = InteractionPreservingWindModel(
        velocity_transform=normalizer,
        background_profile=profile,
        mode="P",
        hidden=128,
        message=128,
        regional_anchors=0,
        depth=2,
        collective_width=64,
        receiver_tile=512,
        seed=seed,
        max_sources=30,
        locality_prior_strength=None,
    ).cpu().eval()
    target_state = model.state_dict()
    mapped = map_run2202_direct_state(source_state, target_state)
    mapped_sha = _tensor_state_sha256(mapped)
    loaded = model.load_state_dict(mapped, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Run2202 direct tensors failed strict loading into the P model.")
    if _tensor_state_sha256(model.state_dict()) != mapped_sha:
        raise RuntimeError("P model state changed while installing the Run2202 direct tensors.")
    return Run2202DirectModel(
        model=model,
        model_config=model.export_config(),
        mapped_state_dict={key: value.detach().cpu().contiguous().clone() for key, value in mapped.items()},
        mapped_state_sha256=mapped_sha,
        source_checkpoint_payload=checkpoint_payload,
        source_identity=identity,
        source_model_state_dict=source_state,
        normalizer=normalizer,
        profile=profile,
    )


def build_run2202_migration_document(
    *,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_sha256: str,
    checkpoint_path: str | Path,
    normalization_document: Mapping[str, Any],
    normalization_sha256: str,
    normalization_path: str | Path,
    identity_document: Mapping[str, Any],
    identity_file_sha256: str,
    identity_path: str | Path,
    source_state_document: Mapping[str, Any],
    source_state_file_sha256: str,
    source_state_path: str | Path,
) -> dict[str, Any]:
    """Build the immutable e5000 migration document from hash-bound inputs."""

    if checkpoint_sha256 != RUN2202_EPOCH_CHECKPOINT_SHA256[5000]:
        raise ValueError("Only the hash-pinned Run2202 epoch-5000 checkpoint can create the mature migration.")
    if normalization_sha256 != RUN2202_NORMALIZATION_SHA256:
        raise ValueError("Run2202 migration requires the exact saved TRAIN normalizer/profile artifact hash.")
    identity, provider, epoch = _validate_source_identity(checkpoint_payload)
    if epoch != 5000:
        raise ValueError("The deployable Run2202 compatibility migration is fixed to epoch 5000.")
    if stable_json_sha256(identity_document) != stable_json_sha256(identity):
        raise ValueError("Run2202 run identity artifact differs from the identity embedded in its checkpoint.")
    if identity_file_sha256 != RUN2202_IDENTITY_FILE_SHA256:
        raise ValueError("Run2202 raw run-identity file hash differs from the protected historical artifact.")
    if source_state_file_sha256 != RUN2202_SOURCE_STATE_FILE_SHA256:
        raise ValueError("Run2202 raw source-state file hash differs from the protected historical artifact.")
    if (
        source_state_document.get("commit") != "4660354bfe436cd195587cc04938a31c42618a0e"
        or source_state_document.get("dirty") is not True
        or source_state_document.get("changed_path_count") != 2
        or source_state_document.get("git_available") is not True
    ):
        raise ValueError("Run2202 saved dirty-source provenance differs from the bridge receipt.")
    if normalization_document.get("schema_version") != 1:
        raise ValueError("Run2202 normalization schema version must be exactly 1.")
    if stable_json_sha256(normalization_document) != RUN2202_NORMALIZATION_DOCUMENT_SHA256:
        raise ValueError("Run2202 normalization document differs from the exact hash-bound TRAIN artifact.")
    if normalization_document.get("vertical_profile_baseline") != provider["background_profile"]:
        raise ValueError("Run2202 normalization artifact lacks the checkpoint-bound TRAIN profile.")

    built = build_run2202_direct_model(checkpoint_payload, normalization_document)
    if built.mapped_state_sha256 != RUN2202_DIRECT_STATE_SHA256:
        raise ValueError(
            "Run2202 e5000 mapped tensor digest differs from the reported 52-tensor bridge: "
            f"{built.mapped_state_sha256}."
        )
    source_state = checkpoint_payload["model_state_dict"]
    excluded = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in source_state.items()
        if key not in built.mapped_state_dict
    }
    if len(built.mapped_state_dict) != 52 or len(excluded) != 14:
        raise ValueError("Run2202 state inventory changed; expected 52 deployed and 14 refinement-only tensors.")

    checkpoint_metadata = {
        key: copy.deepcopy(value)
        for key, value in checkpoint_payload.items()
        if key != "model_state_dict"
    }
    required_training_metadata = {
        "experiment_identity",
        "optimizer_state_by_name",
        "optimizer_group_names",
        "optimizer_schedule_contract",
        "provider_training_state",
        "sampler_state",
        "rng_state",
        "history",
    }
    if not required_training_metadata.issubset(checkpoint_metadata):
        raise ValueError(
            "Run2202 checkpoint is missing full training provenance/moments: "
            f"{sorted(required_training_metadata - set(checkpoint_metadata))}."
        )
    if not isinstance(checkpoint_metadata["optimizer_state_by_name"], Mapping):
        raise TypeError("Run2202 optimizer moments must remain a named mapping.")
    if not isinstance(checkpoint_metadata["history"], list) or len(checkpoint_metadata["history"]) != 5000:
        raise ValueError("Run2202 full training history must retain its 5,000 saved epoch records.")

    document = {
        "schema": RUN2202_MIGRATION_SCHEMA,
        "label": RUN2202_MIGRATION_LABEL,
        "model_config": built.model_config,
        "model_state_dict": built.mapped_state_dict,
        "provenance": {
            "source_run_id": RUN2202_RUN_ID,
            "source_epoch": 5000,
            "source_training_exposure": "full original TRAIN: 420 direction rows across 140 layouts",
            "source_checkpoint_sha256": checkpoint_sha256,
            "source_checkpoint_path_at_migration": str(Path(checkpoint_path).resolve()),
            "source_identity_sha256": stable_json_sha256(identity),
            "source_identity_file_sha256": identity_file_sha256,
            "source_identity_path_at_migration": str(Path(identity_path).resolve()),
            "source_identity": copy.deepcopy(dict(identity)),
            "source_identity_file_document": copy.deepcopy(dict(identity_document)),
            "source_identity_file_document_sha256": stable_json_sha256(identity_document),
            "source_state_file_sha256": source_state_file_sha256,
            "source_state_document_sha256": stable_json_sha256(source_state_document),
            "source_state_path_at_migration": str(Path(source_state_path).resolve()),
            "source_state": copy.deepcopy(dict(source_state_document)),
            "normalization_sha256": normalization_sha256,
            "normalization_path_at_migration": str(Path(normalization_path).resolve()),
            "checkpoint_bound_normalization_path": provider.get("normalization_path"),
            "normalization_artifact_matches_checkpoint_binding": True,
            "normalization_document": copy.deepcopy(dict(normalization_document)),
            "normalization_document_sha256": stable_json_sha256(normalization_document),
            "background_profile_sha256": stable_json_sha256(provider["background_profile"]),
            "environment_representation": copy.deepcopy(provider["environment_representation"]),
            "legacy_feature_and_training_identity": copy.deepcopy(dict(provider)),
            "source_model_state_sha256": _tensor_state_sha256(source_state),
            "source_model_state_manifest": _state_manifest(source_state),
            "source_state_dict_excluded_from_p": excluded,
            "source_checkpoint_metadata": checkpoint_metadata,
            "source_optimizer_moments_are_archival_only": True,
            "mapped_direct_state_key_count": len(built.mapped_state_dict),
            "mapped_direct_state_sha256": built.mapped_state_sha256,
            "model_config_sha256": stable_json_sha256(built.model_config),
            "interpretation": (
                "The P tensors retain the Run2202 e5000 direct inference function. Source optimizer moments, "
                "history, sampler/RNG state and training-only refinement tensors are preserved as provenance; "
                "they are not a P-family optimizer resume state."
            ),
        },
    }
    _validate_migration_document(document)
    return document


def _validate_migration_document(document: Mapping[str, Any]) -> None:
    expected_top = {"schema", "label", "model_config", "model_state_dict", "provenance"}
    if set(document) != expected_top or document.get("schema") != RUN2202_MIGRATION_SCHEMA:
        raise ValueError("Run2202 migration document schema or top-level fields are unsupported.")
    if document.get("label") != RUN2202_MIGRATION_LABEL:
        raise ValueError("Run2202 migration label must preserve the historical compatibility lineage.")
    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("Run2202 migration provenance must be a mapping.")
    if (
        provenance.get("source_run_id") != RUN2202_RUN_ID
        or provenance.get("source_epoch") != 5000
        or provenance.get("source_checkpoint_sha256") != RUN2202_EPOCH_CHECKPOINT_SHA256[5000]
        or provenance.get("source_identity_file_sha256") != RUN2202_IDENTITY_FILE_SHA256
        or provenance.get("source_state_file_sha256") != RUN2202_SOURCE_STATE_FILE_SHA256
        or provenance.get("source_state_document_sha256") != RUN2202_SOURCE_STATE_DOCUMENT_SHA256
        or provenance.get("normalization_sha256") != RUN2202_NORMALIZATION_SHA256
        or provenance.get("background_profile_sha256") != RUN2202_BACKGROUND_PROFILE_SHA256
        or provenance.get("mapped_direct_state_key_count") != 52
        or provenance.get("mapped_direct_state_sha256") != RUN2202_DIRECT_STATE_SHA256
        or provenance.get("source_optimizer_moments_are_archival_only") is not True
    ):
        raise ValueError("Run2202 migration provenance is incomplete or differs from the pinned reference.")
    identity = provenance.get("source_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Run2202 migration does not contain its complete source training identity.")
    if stable_json_sha256(identity) != provenance.get("source_identity_sha256"):
        raise ValueError("Run2202 migration source identity digest mismatch.")
    if provenance.get("source_identity_file_sha256") != RUN2202_IDENTITY_FILE_SHA256:
        raise ValueError("Run2202 migration identity artifact hash differs from the pinned source.")
    identity_file_document = provenance.get("source_identity_file_document")
    if (
        not isinstance(identity_file_document, Mapping)
        or stable_json_sha256(identity_file_document) != provenance.get("source_identity_file_document_sha256")
        or stable_json_sha256(identity_file_document) != stable_json_sha256(identity)
    ):
        raise ValueError("Run2202 source checkpoint identity and exact JSON identity artifact do not agree.")
    _validate_source_identity({
        "checkpoint_schema_version": 1,
        "workflow": "unified_interaction_refinement",
        "epoch": 5000,
        "current_epoch": 5000,
        "arm": "full_detail",
        "experiment_identity": identity,
        "model_state_dict": {"sentinel": torch.zeros(1)},
    })
    normalization = provenance.get("normalization_document")
    if (
        not isinstance(normalization, Mapping)
        or stable_json_sha256(normalization) != provenance.get("normalization_document_sha256")
    ):
        raise ValueError("Run2202 migration embedded normalization document is absent or corrupted.")
    if (
        provenance.get("normalization_artifact_matches_checkpoint_binding") is not True
        or provenance.get("normalization_sha256") != RUN2202_NORMALIZATION_SHA256
    ):
        raise ValueError("Run2202 migration does not retain the exact checkpoint-bound TRAIN normalization artifact.")
    if (
        stable_json_sha256(provenance["legacy_feature_and_training_identity"]["background_profile"])
        != RUN2202_BACKGROUND_PROFILE_SHA256
    ):
        raise ValueError("Run2202 migration embedded profile no longer matches the source TRAIN profile.")
    if (
        normalization.get("vertical_profile_baseline")
        != provenance["legacy_feature_and_training_identity"].get("background_profile")
    ):
        raise ValueError("Run2202 migration normalizer and saved profile do not agree.")
    if provenance.get("environment_representation") != _EXPECTED_ENVIRONMENT_REPRESENTATION:
        raise ValueError("Run2202 migration E8 inventory changed.")
    legacy_identity = provenance.get("legacy_feature_and_training_identity")
    if not isinstance(legacy_identity, Mapping) or dict(legacy_identity) != dict(identity["provider_identity"]):
        raise ValueError("Run2202 migration feature/training identity differs from the checkpoint-bound provider.")
    checkpoint_metadata = provenance.get("source_checkpoint_metadata")
    if not isinstance(checkpoint_metadata, Mapping):
        raise TypeError("Run2202 migration lacks its source training checkpoint metadata.")
    if (
        checkpoint_metadata.get("experiment_identity") != identity
        or checkpoint_metadata.get("checkpoint_schema_version") != 1
        or checkpoint_metadata.get("workflow") != "unified_interaction_refinement"
        or checkpoint_metadata.get("arm") != "full_detail"
        or checkpoint_metadata.get("epoch") != 5000
        or checkpoint_metadata.get("current_epoch") != 5000
    ):
        raise ValueError("Source checkpoint metadata and full experiment identity disagree.")
    source_state_document = provenance.get("source_state")
    if (
        not isinstance(source_state_document, Mapping)
        or stable_json_sha256(source_state_document) != RUN2202_SOURCE_STATE_DOCUMENT_SHA256
        or source_state_document.get("commit") != "4660354bfe436cd195587cc04938a31c42618a0e"
        or source_state_document.get("dirty") is not True
        or source_state_document.get("changed_path_count") != 2
        or source_state_document.get("git_available") is not True
    ):
        raise ValueError("Run2202 migration saved source-state provenance differs from the historical receipt.")
    if (
        not isinstance(checkpoint_metadata.get("optimizer_state_by_name"), Mapping)
        or not checkpoint_metadata["optimizer_state_by_name"]
    ):
        raise ValueError("Run2202 optimizer moments are not retained in the migration provenance.")
    if not isinstance(checkpoint_metadata.get("history"), list) or len(checkpoint_metadata["history"]) != 5000:
        raise ValueError("Run2202 five-thousand-epoch history is not retained in the migration provenance.")
    state = document.get("model_state_dict")
    if not isinstance(state, Mapping) or len(state) != 52:
        raise ValueError("Run2202 migrated P state must contain exactly the 52 direct tensors.")
    if _tensor_state_sha256(state) != provenance.get("mapped_direct_state_sha256"):
        raise ValueError("Run2202 migrated P tensor digest differs from its provenance receipt.")
    config = document.get("model_config")
    if not isinstance(config, Mapping) or stable_json_sha256(config) != provenance.get("model_config_sha256"):
        raise ValueError("Run2202 migrated P config is absent or its saved digest changed.")
    saved_profile = legacy_identity.get("background_profile")
    _validate_embedded_normalizer_binding(
        normalization,
        config,
        saved_profile,
        normalization_document_sha256=str(provenance.get("normalization_document_sha256")),
    )
    _validate_run2202_p_config(
        config,
        normalization,
        saved_profile,
        seed=int(identity["engine_config"]["seed"]),
    )
    model = InteractionPreservingWindModel.from_config(config)
    if set(state) != set(model.state_dict()):
        raise ValueError("Run2202 migrated P tensor keys do not exactly match the strict target model.")
    for key, value in state.items():
        target = model.state_dict()[key]
        if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
            raise ValueError(f"Run2202 migrated P tensor {key!r} no longer matches the target model.")
    excluded = provenance.get("source_state_dict_excluded_from_p")
    manifest = provenance.get("source_model_state_manifest")
    if not isinstance(excluded, Mapping) or not isinstance(manifest, Mapping):
        raise TypeError("Run2202 migrated source state inventory is incomplete.")
    if set(state) & set(excluded) or len(excluded) != 14:
        raise ValueError("Run2202 retained training-only state must be the disjoint 14-tensor refinement inventory.")
    reconstructed = {**state, **excluded}
    if _state_manifest(reconstructed) != manifest:
        raise ValueError("Run2202 source model-state manifest differs from the retained direct/refinement tensors.")
    if _tensor_state_sha256(reconstructed) != provenance.get("source_model_state_sha256"):
        raise ValueError("Run2202 full original model state cannot be reconstructed from the migration archive.")


def load_run2202_migration(
    path: str | Path,
    *,
    map_location: Any = "cpu",
) -> tuple[InteractionPreservingWindModel, dict[str, Any]]:
    """Load P from the migration file alone; no source Run2202 file is opened."""

    document = load_trusted_checkpoint(path, map_location=map_location)
    _validate_migration_document(document)
    model = InteractionPreservingWindModel.from_config(document["model_config"])
    loaded = model.load_state_dict(document["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Run2202 P migration did not strictly populate the maintained model.")
    model.eval()
    return model, copy.deepcopy(dict(document["provenance"]))


def migrate_run2202_checkpoint(
    *,
    checkpoint_path: str | Path,
    normalization_path: str | Path,
    output_path: str | Path,
    identity_path: str | Path | None = None,
    source_state_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create the exact e5000 migration, refusing changed or incomplete inputs."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    run_dir = checkpoint_path.parent.parent
    identity_path = (
        Path(identity_path).expanduser().resolve()
        if identity_path
        else run_dir / "configs/experiment_identity.json"
    )
    source_state_path = (
        Path(source_state_path).expanduser().resolve()
        if source_state_path
        else run_dir / "environment/source_state.json"
    )
    normalization_path = Path(normalization_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if checkpoint_path.name != "epoch_5000_model.pt" or run_dir.name != RUN2202_RUN_ID:
        raise ValueError("Migration input must be the literal Run2202 epoch_5000_model.pt path.")
    required_files = (checkpoint_path, normalization_path, identity_path, source_state_path)
    if any(not path.is_file() for path in required_files):
        raise FileNotFoundError(
            "Run2202 checkpoint, exact normalizer, run identity and saved source state are all required."
        )
    checkpoint_hash = file_sha256(checkpoint_path)
    normalization_hash = file_sha256(normalization_path)
    if checkpoint_hash != RUN2202_EPOCH_CHECKPOINT_SHA256[5000]:
        raise ValueError(f"Run2202 checkpoint hash mismatch: {checkpoint_hash}.")
    if normalization_hash != RUN2202_NORMALIZATION_SHA256:
        raise ValueError(f"Run2202 normalization hash mismatch: {normalization_hash}.")

    checkpoint_payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    normalization_document = json.loads(normalization_path.read_text(encoding="utf-8"))
    identity_document = json.loads(identity_path.read_text(encoding="utf-8"))
    source_state_document = json.loads(source_state_path.read_text(encoding="utf-8"))
    document = build_run2202_migration_document(
        checkpoint_payload=checkpoint_payload,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_path=checkpoint_path,
        normalization_document=normalization_document,
        normalization_sha256=normalization_hash,
        normalization_path=normalization_path,
        identity_document=identity_document,
        identity_file_sha256=file_sha256(identity_path),
        identity_path=identity_path,
        source_state_document=source_state_document,
        source_state_file_sha256=file_sha256(source_state_path),
        source_state_path=source_state_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing migration artifact: {output_path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    try:
        torch.save(document, temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        # Linking a complete temporary file is atomic and cannot replace a user file.
        os.link(temporary_name, output_path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return {
        "status": "complete",
        "artifact_path": str(output_path),
        "artifact_sha256": file_sha256(output_path),
        "source_checkpoint_sha256": checkpoint_hash,
        "normalization_sha256": normalization_hash,
        "mapped_direct_state_key_count": document["provenance"]["mapped_direct_state_key_count"],
        "mapped_direct_state_sha256": document["provenance"]["mapped_direct_state_sha256"],
        "source_checkpoint_metadata_retained": True,
        "optimizer_moments_retained_as_archival_provenance": True,
        "external_source_checkpoint_required_to_reload": False,
    }


__all__ = [
    "RUN2202_BACKGROUND_PROFILE_SHA256",
    "RUN2202_DIRECT_STATE_SHA256",
    "RUN2202_EPOCH_CHECKPOINT_SHA256",
    "RUN2202_MIGRATION_LABEL",
    "RUN2202_MIGRATION_SCHEMA",
    "RUN2202_NORMALIZATION_SHA256",
    "RUN2202_RUN_ID",
    "Run2202DirectModel",
    "build_run2202_direct_model",
    "build_run2202_migration_document",
    "file_sha256",
    "load_run2202_migration",
    "map_run2202_direct_state",
    "migrate_run2202_checkpoint",
    "stable_json_sha256",
]
