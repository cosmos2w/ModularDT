"""Versioned Round5 scene-prepared moment-hyperedge recipe contracts.

This protocol deliberately wraps the existing native IAG-C development
contract. The native dataset, objective, sampling, and optimizer settings stay
in the recipe; only the family envelope and the opt-in core configuration are
new.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .inherited_attention_protocol import (
    IAG_C_ARCHITECTURE,
    IAG_FAMILY,
    IAG_PROTOCOL,
    IAG_SCHEMA,
    validate_inherited_attention_recipe,
)

MOMENT_FAMILY = "moment_hyperedges_round5"
MOMENT_WORKFLOW = "moment_hyperedges_child_v1"
MOMENT_PROTOCOL = "scene_prepared_moment_hyperedge_development_v1"
MOMENT_RECIPE_SCHEMA = "scene_prepared_moment_hyperedge_recipe_v1"
MOMENT_CORE_SCHEMA = "scene_prepared_moment_hyperedge_core_v1"
MOMENT_ATTACHMENT_SCHEMA = "scene_prepared_moment_hyperedge_child_attachment_v1"
MOMENT_VARIANTS = frozenset(("C0", "M", "H"))
MOMENT_PARENT_EPOCH = 500
MOMENT_FIRST_SELECTION_EPOCH = 600
MOMENT_SELECTION_INTERVAL = 100
MOMENT_FINAL_EPOCH = 2500
MOMENT_WIDTH_PER_HEAD = 8
MOMENT_PROCESSOR_HIDDEN_DIM = 128
MOMENT_INIT_SEED = 0
MOMENT_QUERY_EXECUTOR = "indexed_projected"

_MOMENT_ENVELOPE_FIELDS = frozenset({
    "schema", "family", "model_family", "workflow", "protocol", "variant",
    "source_checkpoint", "source_checkpoint_sha256", "source_recipe_sha256",
    "source_epoch",
})


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def native_recipe_for_moment_hyperedges(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the existing IAG-C recipe view consumed by native builders.

    The model sub-config intentionally retains the new core schema and its
    projected executor. The native recipe envelope is restored so the
    established recipe validator, transforms, and task factories remain the
    single source of dataset and loss semantics.
    """

    recipe = validate_moment_hyperedge_recipe(value)
    native = {key: copy.deepcopy(item) for key, item in recipe.items()
              if key not in _MOMENT_ENVELOPE_FIELDS}
    native.update({
        "schema": IAG_SCHEMA,
        "family": IAG_FAMILY,
        "model_family": IAG_FAMILY,
        "protocol": IAG_PROTOCOL,
    })
    return validate_inherited_attention_recipe(native)


def validate_moment_hyperedge_recipe(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one bounded Round5 development-continuation recipe."""

    if not isinstance(value, Mapping):
        raise TypeError("Moment-hyperedge recipe must be a mapping.")
    recipe = copy.deepcopy(dict(value))
    expected = {
        "schema": MOMENT_RECIPE_SCHEMA,
        "family": MOMENT_FAMILY,
        "model_family": MOMENT_FAMILY,
        "workflow": MOMENT_WORKFLOW,
        "protocol": MOMENT_PROTOCOL,
        "architecture": IAG_C_ARCHITECTURE,
    }
    for name, wanted in expected.items():
        if recipe.get(name) != wanted:
            raise ValueError(f"Moment-hyperedge recipe requires {name}={wanted!r}.")
    variant = recipe.get("variant")
    if variant not in MOMENT_VARIANTS:
        raise ValueError("Moment-hyperedge variant must be C0, M, or H.")
    if recipe.get("source_epoch") != MOMENT_PARENT_EPOCH:
        raise ValueError("Round5 moment-hyperedge branches attach only to their own C-soft e500 parent.")
    parent_path = recipe.get("source_checkpoint")
    if (not isinstance(parent_path, str) or not Path(parent_path).is_absolute()
            or str(Path(parent_path).resolve()) != parent_path):
        raise ValueError("Moment-hyperedge source_checkpoint must be an absolute canonical path.")
    if not _is_sha256(recipe.get("source_checkpoint_sha256")):
        raise ValueError("Moment-hyperedge recipe must bind the source checkpoint SHA-256.")
    if not _is_sha256(recipe.get("source_recipe_sha256")):
        raise ValueError("Moment-hyperedge recipe must bind the source recipe SHA-256.")
    if recipe.get("dataset") not in ("thermal", "wind"):
        raise ValueError("Round5 moment-hyperedge development is restricted to Thermal or Wind.")
    if recipe.get("representation") != "iag_c":
        raise ValueError("Round5 moment-hyperedge development preserves the IAG-C representation.")
    if recipe.get("normalization") != "soft" or recipe.get("trajectory") != "fresh_soft":
        raise ValueError("Round5 branches must preserve their same-dataset soft C parent trajectory.")
    if recipe.get("total_epochs") != MOMENT_FINAL_EPOCH or recipe.get("approved_stop_after") != MOMENT_FINAL_EPOCH:
        raise ValueError("Round5 primary continuations retain the literal absolute e2500 horizon.")

    model = recipe.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("Moment-hyperedge recipe requires the inherited native model configuration.")
    model_expected = {
        "schema": MOMENT_CORE_SCHEMA,
        "moment_variant": variant,
        "moment_width": MOMENT_WIDTH_PER_HEAD,
        "processor_hidden_dim": MOMENT_PROCESSOR_HIDDEN_DIM,
        "moment_init_seed": MOMENT_INIT_SEED,
        "query_executor": MOMENT_QUERY_EXECUTOR,
        "collect_exact_work_counts": False,
    }
    for name, wanted in model_expected.items():
        if model.get(name) != wanted:
            raise ValueError(f"Moment-hyperedge model config requires {name}={wanted!r}.")

    # Delegating the remaining contract catches changed dataset dimensions,
    # splits, query quotas, optimizer values, and output cadence. Only the
    # outer Round5 envelope is translated back to the existing IAG contract.
    native_recipe_for_moment_hyperedges_unchecked(recipe)
    return recipe


def native_recipe_for_moment_hyperedges_unchecked(value: Mapping[str, Any]) -> dict[str, Any]:
    """Internal validator route that avoids recursive public validation."""

    native = {key: copy.deepcopy(item) for key, item in value.items()
              if key not in _MOMENT_ENVELOPE_FIELDS}
    native.update({
        "schema": IAG_SCHEMA,
        "family": IAG_FAMILY,
        "model_family": IAG_FAMILY,
        "protocol": IAG_PROTOCOL,
    })
    return validate_inherited_attention_recipe(native)


def parent_recipe_from_moment_hyperedges(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only the explicit Round5 envelope/core changes to C-soft."""

    native = native_recipe_for_moment_hyperedges(value)
    model = copy.deepcopy(native["model"])
    for name in (
        "schema", "moment_variant", "moment_width", "processor_hidden_dim",
        "moment_init_seed", "collect_exact_work_counts",
    ):
        model.pop(name, None)
    model["query_executor"] = "indexed"
    native["model"] = model
    return validate_inherited_attention_recipe(native)


def make_moment_hyperedge_recipe(
    parent_recipe: Mapping[str, Any],
    source_checkpoint: str | Path,
    variant: str,
) -> dict[str, Any]:
    """Derive the sole allowed R5 envelope and opt-in core config from C-soft."""

    parent = validate_inherited_attention_recipe(parent_recipe)
    if (parent.get("architecture") != IAG_C_ARCHITECTURE
            or parent.get("normalization") != "soft"
            or parent.get("trajectory") != "fresh_soft"
            or parent.get("protocol") != IAG_PROTOCOL):
        raise ValueError("Moment-hyperedge recipe source must be segmented soft IAG-C development.")
    if variant not in MOMENT_VARIANTS:
        raise ValueError("Moment-hyperedge variant must be C0, M, or H.")
    checkpoint = Path(source_checkpoint).expanduser().resolve(strict=True)
    model = copy.deepcopy(parent["model"])
    model.update({
        "schema": MOMENT_CORE_SCHEMA,
        "query_executor": MOMENT_QUERY_EXECUTOR,
        "collect_exact_work_counts": False,
        "moment_variant": variant,
        "moment_width": MOMENT_WIDTH_PER_HEAD,
        "processor_hidden_dim": MOMENT_PROCESSOR_HIDDEN_DIM,
        "moment_init_seed": MOMENT_INIT_SEED,
    })
    recipe = copy.deepcopy(parent)
    recipe.update({
        "schema": MOMENT_RECIPE_SCHEMA,
        "family": MOMENT_FAMILY,
        "model_family": MOMENT_FAMILY,
        "workflow": MOMENT_WORKFLOW,
        "protocol": MOMENT_PROTOCOL,
        "variant": variant,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "source_recipe_sha256": canonical_sha256(parent),
        "source_epoch": MOMENT_PARENT_EPOCH,
        "model": model,
    })
    return validate_moment_hyperedge_recipe(recipe)


def moment_selection_eligible(epoch: int) -> bool:
    """Whether an absolute checkpoint age participates in the secondary selector."""

    return (type(epoch) is int and MOMENT_FIRST_SELECTION_EPOCH <= epoch <= MOMENT_FINAL_EPOCH
            and (epoch - MOMENT_FIRST_SELECTION_EPOCH) % MOMENT_SELECTION_INTERVAL == 0)


__all__ = [
    "MOMENT_ATTACHMENT_SCHEMA",
    "MOMENT_CORE_SCHEMA",
    "MOMENT_FAMILY",
    "MOMENT_FINAL_EPOCH",
    "MOMENT_FIRST_SELECTION_EPOCH",
    "MOMENT_INIT_SEED",
    "MOMENT_PARENT_EPOCH",
    "MOMENT_PROCESSOR_HIDDEN_DIM",
    "MOMENT_PROTOCOL",
    "MOMENT_QUERY_EXECUTOR",
    "MOMENT_RECIPE_SCHEMA",
    "MOMENT_SELECTION_INTERVAL",
    "MOMENT_VARIANTS",
    "MOMENT_WIDTH_PER_HEAD",
    "MOMENT_WORKFLOW",
    "canonical_sha256",
    "file_sha256",
    "make_moment_hyperedge_recipe",
    "moment_selection_eligible",
    "native_recipe_for_moment_hyperedges",
    "parent_recipe_from_moment_hyperedges",
    "validate_moment_hyperedge_recipe",
]
