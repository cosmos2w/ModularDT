"""Strict recipe and child declarations for the Round7 native-token family."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .source_read_protocol import validate_source_read_recipe

NATIVE_RETRIEVAL_FAMILY = "native_retrieval_v1"
NATIVE_RETRIEVAL_WORKFLOW = "native_retrieval_joint_v1"
NATIVE_RETRIEVAL_PROTOCOL = "native_token_retrieval_continuation_v1"
NATIVE_RETRIEVAL_RECIPE_SCHEMA = "native_retrieval_recipe_v1"
NATIVE_RETRIEVAL_CHILD_SCHEMA = "native_retrieval_child_v1"
NATIVE_RETRIEVAL_CHECKPOINT_WORKFLOW = "native_retrieval_joint_v1"
NATIVE_RETRIEVAL_NATIVE_BACKENDS = frozenset(("selected", "dense_masked", "compiled_selected"))
NATIVE_QUERY_THRESHOLD_CANDIDATES = (0.10, 0.20, 0.35)

_RECIPE_FIELDS = frozenset({
    "schema", "family", "model_family", "workflow", "protocol", "dataset",
    "architecture", "treatment", "source_checkpoint", "source_checkpoint_sha256",
    "source_epoch", "source_recipe_sha256", "source_provider_identity_sha256",
    "base_recipe", "base_recipe_sha256", "native_tau_mm", "native_tau_me",
    "native_tau_q", "native_backend", "environment_value_reader",
    "permission_curriculum", "execution",
})


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_path(value: Any, field: str) -> str:
    if (not isinstance(value, str) or not Path(value).is_absolute()
            or str(Path(value).resolve()) != value):
        raise ValueError(f"{field} must be an absolute canonical path.")
    return value


def _source_contract(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    from .compat import load_trusted_checkpoint

    source_path = Path(path).expanduser().resolve(strict=True)
    payload = load_trusted_checkpoint(source_path, map_location="cpu")
    identity = payload.get("experiment_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Native retrieval source checkpoint has no sealed experiment identity.")
    recipe = identity.get("recipe")
    provider = identity.get("provider_identity")
    if (payload.get("workflow") != "content_hypergraph_joint_v2"
            or identity.get("model_family") != "honf_content_hypergraph"
            or identity.get("workflow") != "content_hypergraph_joint_v2"
            or not isinstance(recipe, Mapping) or not isinstance(provider, Mapping)
            or payload.get("arm") != "A2-D" or payload.get("epoch") != 500):
        raise ValueError("Round7 can branch only from a retained content-hypergraph A2-D e500 checkpoint.")
    base_recipe = validate_source_read_recipe(recipe)
    if (base_recipe.get("protocol") != "source_read_development_v1"
            or base_recipe.get("source_reader_variant") != "D0"
            or base_recipe.get("architecture") != "A2-D"
            or base_recipe.get("dataset") not in ("thermal", "wind")):
        raise ValueError("Round7 requires the exact segmented D0 development recipe, not a sparse or formal source.")
    source_digest = file_sha256(source_path)
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise TypeError("D0 source checkpoint is missing its model state dictionary.")
    if not isinstance(payload.get("optimizer_state_by_name"), Mapping):
        raise TypeError("D0 source checkpoint is missing its named optimizer state.")
    if not payload.get("optimizer_group_names") or not payload.get("optimizer_schedule_contract"):
        raise ValueError("D0 source checkpoint is missing its named optimizer-group contract.")
    return payload, copy.deepcopy(dict(identity)), source_digest


def make_native_retrieval_recipe(
    source_checkpoint: str | Path,
    *,
    treatment: str,
    native_backend: str = "selected",
    tau_q: float = 0.20,
) -> dict[str, Any]:
    """Seal one matched Round7 treatment to an exact segmented D0 e500 source."""
    payload, identity, source_digest = _source_contract(source_checkpoint)
    base_recipe = validate_source_read_recipe(identity["recipe"])
    if treatment not in ("D-control", "G-native-M"):
        raise ValueError("Round7 treatment must be D-control or G-native-M.")
    if treatment == "D-control":
        if native_backend != "full_direct" or tau_q != 0.0:
            raise ValueError("D-control retains the original full A2-D reader and has no retrieval thresholds.")
        tau_mm = tau_me = tau_q = 0.0
    else:
        if native_backend not in NATIVE_RETRIEVAL_NATIVE_BACKENDS:
            raise ValueError("G-native-M must declare selected, dense_masked or compiled_selected execution.")
        if type(tau_q) not in (float, int) or float(tau_q) not in NATIVE_QUERY_THRESHOLD_CANDIDATES:
            raise ValueError("G-native-M query threshold must be one of the three TRAIN-only candidates.")
        tau_mm = tau_me = 0.20
        tau_q = float(tau_q)
    recipe = {
        "schema": NATIVE_RETRIEVAL_RECIPE_SCHEMA,
        "family": NATIVE_RETRIEVAL_FAMILY,
        "model_family": NATIVE_RETRIEVAL_FAMILY,
        "workflow": NATIVE_RETRIEVAL_WORKFLOW,
        "protocol": NATIVE_RETRIEVAL_PROTOCOL,
        "dataset": base_recipe["dataset"],
        "architecture": "A2-D",
        "treatment": treatment,
        "source_checkpoint": str(Path(source_checkpoint).expanduser().resolve(strict=True)),
        "source_checkpoint_sha256": source_digest,
        "source_epoch": 500,
        "source_recipe_sha256": str(identity["recipe_sha256"]),
        "source_provider_identity_sha256": canonical_sha256(identity["provider_identity"]),
        "base_recipe": base_recipe,
        "base_recipe_sha256": canonical_sha256(base_recipe),
        "native_tau_mm": tau_mm,
        "native_tau_me": tau_me,
        "native_tau_q": tau_q,
        "native_backend": native_backend,
        "environment_value_reader": False,
        "permission_curriculum": {
            "schema": "native_absolute_age_permission_curriculum_v1",
            "branch_boundary_epoch": 500,
            "transition_epochs": [501, 1000],
            "exploration_epochs": [1001, 2000],
            "exploration_every_absolute_epochs": 8,
            "sparse_epochs": [2001, 2500],
            "terminal_thresholds": {"tau_mm": tau_mm, "tau_me": tau_me, "tau_q": tau_q},
            "exploration_replaces_sparse_epoch": True,
            "direct_environment_value_reader": False,
        },
        "execution": copy.deepcopy(base_recipe["execution"]),
    }
    validate_native_retrieval_recipe(recipe)
    # A new family recipe never grants direct E reads. Keep the argument use
    # explicit so construction cannot silently opt into that pilot.
    assert payload["epoch"] == 500
    return recipe


def validate_native_retrieval_recipe(
    value: Mapping[str, Any], *, verify_source_file: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECIPE_FIELDS:
        raise ValueError("Native retrieval recipe must contain the exact registered field inventory.")
    recipe = copy.deepcopy(dict(value))
    for name, expected in (
        ("schema", NATIVE_RETRIEVAL_RECIPE_SCHEMA), ("family", NATIVE_RETRIEVAL_FAMILY),
        ("model_family", NATIVE_RETRIEVAL_FAMILY), ("workflow", NATIVE_RETRIEVAL_WORKFLOW),
        ("protocol", NATIVE_RETRIEVAL_PROTOCOL), ("architecture", "A2-D"),
    ):
        if recipe.get(name) != expected:
            raise ValueError(f"Native retrieval recipe requires {name}={expected!r}.")
    if recipe.get("dataset") not in ("thermal", "wind"):
        raise ValueError("Native retrieval dataset must be thermal or wind.")
    if recipe.get("treatment") not in ("D-control", "G-native-M"):
        raise ValueError("Native retrieval treatment is not registered.")
    _canonical_path(recipe.get("source_checkpoint"), "source_checkpoint")
    for name in ("source_checkpoint_sha256", "source_recipe_sha256", "source_provider_identity_sha256",
                 "base_recipe_sha256"):
        digest = recipe.get(name)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    if type(recipe.get("source_epoch")) is not int or recipe["source_epoch"] != 500:
        raise ValueError("Native retrieval source must be exact absolute e500.")
    base_recipe = validate_source_read_recipe(recipe.get("base_recipe", {}))
    if (base_recipe.get("dataset") != recipe["dataset"]
            or base_recipe.get("source_reader_variant") != "D0"
            or base_recipe.get("protocol") != "source_read_development_v1"
            or base_recipe.get("architecture") != "A2-D"):
        raise ValueError("Native retrieval base recipe must retain the same dataset's segmented D0 semantics.")
    if canonical_sha256(base_recipe) != recipe["base_recipe_sha256"]:
        raise ValueError("Native retrieval base recipe checksum is inconsistent.")
    if recipe.get("environment_value_reader") is not False:
        raise ValueError("Primary native retrieval cannot consume direct environment values.")
    curriculum = recipe.get("permission_curriculum")
    expected_curriculum = {
        "schema": "native_absolute_age_permission_curriculum_v1",
        "branch_boundary_epoch": 500,
        "transition_epochs": [501, 1000],
        "exploration_epochs": [1001, 2000],
        "exploration_every_absolute_epochs": 8,
        "sparse_epochs": [2001, 2500],
        "terminal_thresholds": {
            "tau_mm": recipe.get("native_tau_mm"),
            "tau_me": recipe.get("native_tau_me"),
            "tau_q": recipe.get("native_tau_q"),
        },
        "exploration_replaces_sparse_epoch": True,
        "direct_environment_value_reader": False,
    }
    if curriculum != expected_curriculum:
        raise ValueError("Native retrieval permission curriculum differs from the absolute-age contract.")
    if recipe["treatment"] == "D-control":
        if (recipe.get("native_backend") != "full_direct"
                or any(recipe.get(key) != 0.0 for key in ("native_tau_mm", "native_tau_me", "native_tau_q"))):
            raise ValueError("D-control must preserve the unchanged full direct A2-D reader.")
    else:
        if (recipe.get("native_backend") not in NATIVE_RETRIEVAL_NATIVE_BACKENDS
                or recipe.get("native_tau_mm") != 0.20 or recipe.get("native_tau_me") != 0.20
                or recipe.get("native_tau_q") not in NATIVE_QUERY_THRESHOLD_CANDIDATES):
            raise ValueError("G-native-M requires fixed member thresholds and one sealed query candidate.")
    for key in ("native_tau_mm", "native_tau_me", "native_tau_q"):
        threshold_value = recipe.get(key)
        if (type(threshold_value) not in (float, int)
                or not math.isfinite(float(threshold_value))
                or not 0.0 <= float(threshold_value) < 0.5):
            raise ValueError(f"{key} must be a finite number in [0,.5).")
    if recipe.get("execution") != base_recipe.get("execution"):
        raise ValueError("Native retrieval must preserve the D0 monitoring, checkpoint and curve cadence.")
    if canonical_sha256(recipe) != canonical_sha256(dict(value)):
        raise ValueError("Native retrieval recipe contains noncanonical values.")
    if verify_source_file:
        source_path = Path(recipe["source_checkpoint"]).resolve(strict=True)
        if file_sha256(source_path) != recipe["source_checkpoint_sha256"]:
            raise ValueError("Native retrieval D0 source checkpoint changed after recipe sealing.")
        _payload, identity, _digest = _source_contract(source_path)
        if (identity.get("recipe_sha256") != recipe["source_recipe_sha256"]
                or canonical_sha256(identity.get("provider_identity"))
                != recipe["source_provider_identity_sha256"]
                or canonical_sha256(identity.get("recipe")) != recipe["base_recipe_sha256"]):
            raise ValueError("Native retrieval source recipe/provider binding differs from the sealed parent.")
    return recipe


def native_retrieval_child_declaration(recipe: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_native_retrieval_recipe(recipe)
    return {
        "schema": NATIVE_RETRIEVAL_CHILD_SCHEMA,
        "treatment": value["treatment"],
        "source_checkpoint": value["source_checkpoint"],
        "source_checkpoint_sha256": value["source_checkpoint_sha256"],
        "source_epoch": 500,
        "source_recipe_sha256": value["source_recipe_sha256"],
        "source_provider_identity_sha256": value["source_provider_identity_sha256"],
        "child_recipe_sha256": canonical_sha256(value),
        "optimizer_policy": "exact_named_adamw_state_and_schedule_continuation",
        "absolute_epoch_policy": "inherited_500_plus_2000_new_epochs",
        "permission_curriculum": copy.deepcopy(value["permission_curriculum"]),
    }


__all__ = [
    "NATIVE_RETRIEVAL_CHILD_SCHEMA",
    "NATIVE_RETRIEVAL_FAMILY",
    "NATIVE_RETRIEVAL_PROTOCOL",
    "NATIVE_RETRIEVAL_RECIPE_SCHEMA",
    "NATIVE_RETRIEVAL_WORKFLOW",
    "canonical_sha256",
    "file_sha256",
    "make_native_retrieval_recipe",
    "native_retrieval_child_declaration",
    "validate_native_retrieval_recipe",
]
