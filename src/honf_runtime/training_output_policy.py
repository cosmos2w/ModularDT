"""Explicit output-only continuation without relaxing scientific resume bindings."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

OUTPUT_POLICY_SCHEMA = "training_output_policy_amendment_v1"
_ALLOWED_SOURCE_PATHS = frozenset({
    "src/honf_runtime/unified_training.py",
    "src/honf_runtime/training_output_policy.py",
    "src/honf_runtime/content_hypergraph_formal.py",
    "tools/content_hypergraph_train.py",
    "Case_WindFarm/src/windfarm/training/content_hypergraph_task.py",
})


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_changes(saved: Mapping[str, Any], current: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the complete old/new imported-source delta, including new modules."""
    before, after = saved["files"], current["files"]
    return [{"path": path, "old_sha256": before.get(path), "new_sha256": after.get(path)}
            for path in sorted(set(before) | set(after)) if before.get(path) != after.get(path)]


def validate_output_policy_amendment(
    saved: Mapping[str, Any], current: Mapping[str, Any], declaration: Mapping[str, Any],
    *, source_sha256: str, source_epoch: int, stop_after: int, run_directory: str,
) -> dict[str, Any]:
    """Permit only 100-to-10 curves and the audited display/sidecar source patch.

    Validation cadence, selectors, data, transforms, native laws, architecture,
    all optimizer groups/schedules and sampling are compared exactly. Ordinary
    resume keeps its strict identity check and never calls this opt-in helper.
    """
    message = "Invalid output-only continuation; scientific and lineage bindings must remain identical."
    recipe = saved.get("recipe", {})
    horizon = 5000 if recipe.get("dataset") == "thermal" else 2500
    if (not isinstance(declaration, Mapping)
            or declaration.get("schema") != OUTPUT_POLICY_SCHEMA
            or saved.get("workflow") != "content_hypergraph_joint_v2"
            or recipe.get("schema") != "content_hypergraph_recipe_v2"
            or recipe.get("dataset") not in {"thermal", "wind"}
            or recipe.get("architecture") != "A2-D"
            or recipe.get("protocol") != "formal_d_full_train_v1"
            or recipe.get("total_epochs") != horizon or recipe.get("approved_stop_after") != horizon
            or type(source_epoch) is not int or not 0 < source_epoch < horizon
            or type(stop_after) is not int or stop_after != horizon
            or declaration.get("source_epoch") != source_epoch
            or declaration.get("source_checkpoint_sha256") != source_sha256
            or declaration.get("run_directory") != run_directory
            or not isinstance(declaration.get("reason"), str) or not declaration["reason"].strip()):
        raise ValueError(message)
    previous, proposed = saved.get("loaded_source"), current.get("loaded_source")
    if not isinstance(previous, Mapping) or not isinstance(proposed, Mapping):
        raise ValueError(message)
    for binding in (previous, proposed):
        if (not isinstance(binding.get("files"), Mapping)
                or canonical_sha256(dict(binding["files"])) != binding.get("sha256")):
            raise ValueError(message)
    changes = source_changes(previous, proposed)
    if (not changes or any(row["path"] not in _ALLOWED_SOURCE_PATHS for row in changes)
            or any(row["new_sha256"] is None for row in changes)
            or declaration.get("source_changes") != changes
            or declaration.get("saved_loaded_source_sha256") != previous["sha256"]
            or declaration.get("current_loaded_source_sha256") != proposed["sha256"]):
        raise ValueError(message)
    before_recipe, after_recipe = copy.deepcopy(dict(recipe)), copy.deepcopy(dict(current.get("recipe", {})))
    if (before_recipe.get("execution", {}).get("curve_every") != 100
            or after_recipe.get("execution", {}).get("curve_every") != 10):
        raise ValueError(message)
    before_recipe["execution"]["curve_every"] = 10
    if (before_recipe != after_recipe
            or saved.get("recipe_sha256") != canonical_sha256(recipe)
            or current.get("recipe_sha256") != canonical_sha256(after_recipe)):
        raise ValueError(message)
    before_engine = copy.deepcopy(dict(saved.get("engine_config", {})))
    after_engine = copy.deepcopy(dict(current.get("engine_config", {})))
    if before_engine.pop("curve_every", None) != 100 or after_engine.pop("curve_every", 10) != 10:
        raise ValueError(message)
    if after_engine.pop("progress_enabled", True) is not True:
        raise ValueError(message)
    before_engine.pop("progress_enabled", True)
    if before_engine != after_engine:
        raise ValueError(message)
    amended = copy.deepcopy(dict(saved))
    amended["recipe"] = after_recipe
    amended["recipe_sha256"] = current["recipe_sha256"]
    amended["engine_config"] = copy.deepcopy(dict(current["engine_config"]))
    amended["loaded_source"] = copy.deepcopy(dict(proposed))
    provider = amended.get("provider_identity")
    if not isinstance(provider, dict) or provider.get("recipe") != recipe:
        raise ValueError(message)
    provider["recipe"] = copy.deepcopy(after_recipe)
    if amended != current:
        raise ValueError(message)
    return {
        "kind": "explicit_training_output_policy_continuation",
        "schema": OUTPUT_POLICY_SCHEMA,
        "declaration": copy.deepcopy(dict(declaration)),
        "preserved_contract": "weights, moments, history, RNG, sampling, data, normalization, native laws, optimizer, schedule, selectors and validation cadence",
        "output_policy": {"curve_every": 10, "validation_sidecars": "declared_milestones_and_final", "progress_enabled": True},
    }
