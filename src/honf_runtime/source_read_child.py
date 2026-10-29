"""Strict e500 dense-parent attachment for Round3 source-read children.

The child changes only the declared sparse source-read policy and adds named
parameters below the DirectSourceReader source-read namespace. Physical tensors,
provider/data/objective identity, sampler age, schedule and common AdamW moments
remain inherited from the literal development e500 checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .source_read_protocol import (
    SOURCE_READ_DEVELOPMENT,
    SOURCE_READ_NUMERICS,
    validate_source_read_recipe,
)

SCHEMA = "source_read_development_child_v1"
AUXILIARY_FIT_SCHEMA = "source_read_child_auxiliary_fit_v1"
SOURCE_READ_CONFIG_FIELDS = frozenset((
    "source_read_policy", "source_read_executor", "source_read_rank", "source_read_scorer_width",
))
MAX_AUXILIARY_UPDATES = 256
AUXILIARY_OPTIMIZER_ALGORITHM = {
    "name": "AdamW",
    "scope": "inherited_joint_optimizer_new_source_read_parameters_only",
    "learning_rate": 3.0e-4,
    "weight_decay": 1.0e-5,
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "gradient_clip": "global_norm",
    "gradient_clip_norm": 1.0,
    "gradient_clip_scope": "new_source_read_parameters",
    "error_if_nonfinite": True,
    "maximum_updates": MAX_AUXILIARY_UPDATES,
}
_SOURCE_PATH_ALLOWLIST = (
    "src/honf_forward_core/content_hypergraph/model_v2.py",
    "src/honf_forward_core/content_hypergraph/contracts.py",
    "src/honf_forward_core/content_hypergraph/readers_v2.py",
    "src/honf_forward_core/content_hypergraph/source_read_decoder_v1.py",
    "src/honf_forward_core/content_hypergraph/source_read_execution.py",
    "src/honf_forward_core/content_hypergraph/query_chunks_v1.py",
    "src/honf_forward_core/content_hypergraph/source_read_plans.py",
    "src/honf_forward_core/content_hypergraph/source_read_sparse.py",
    "src/honf_forward_core/content_hypergraph/source_read_response.py",
    "src/honf_forward_core/content_hypergraph/source_reader_variants.py",
    "src/honf_runtime/source_read_child.py",
    "src/honf_runtime/source_read_auxiliary.py",
    "src/honf_runtime/source_read_protocol.py",
    "src/honf_runtime/source_read_qualification.py",
    "src/honf_runtime/content_hypergraph.py",
    "src/honf_runtime/unified_training.py",
    "tools/content_hypergraph_train.py",
    "Case_ThermalChannel/src/channelthermal/content_hypergraph.py",
    "Case_ThermalChannel/src/channelthermal/training/content_hypergraph_task_v2.py",
    "Case_WindFarm/src/windfarm/training/content_hypergraph_task.py",
)


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_train_fidelity_screen(value: Any) -> None:
    fields = {
        "schema", "partition", "scope", "pair_denominator", "thresholds",
        "target_union_retention", "target_union_pair_counts", "union_pair_count",
        "physical_pair_count", "query_count", "near_protected_union_pair_count",
        "actual_union_pair_counts", "actual_physical_pair_counts", "actual_union_retention",
        "actual_physical_pair_retention", "target_infeasible_due_to_near_protection",
        "actual_target_overrun_pair_counts", "flow_direct_context_relative_rms",
        "flow_direct_context_head_relative_rms", "thermal_positive_H_endpoint_relative_rms",
        "thermal_kernel_relative_rms", "thermal_temperature_scale",
        "thermal_target_scale_convention", "primary_operating_point", "selection_reason",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Source-read TRAIN fidelity screen fields/schema are not exact.")
    if (value.get("schema") != "source_read_train_fidelity_screen_v1"
            or value.get("partition") != "TRAIN"
            or value.get("scope") != "bounded_dense_teacher_rows_at_most_256_per_train_case"
            or value.get("pair_denominator") != "positive_or_near_protected_physical_pair_union"):
        raise ValueError("Source-read TRAIN fidelity screen has the wrong declared scope.")

    def integer(name: str, *, minimum: int = 0) -> int:
        result = value.get(name)
        if type(result) is not int or result < minimum:
            raise ValueError(f"Source-read TRAIN fidelity screen {name} is malformed.")
        return result

    def numbers(name: str, count: int) -> list[float]:
        result = value.get(name)
        if (not isinstance(result, list) or len(result) != count
                or any(type(item) not in (int, float) or not math.isfinite(item) or item < 0
                       for item in result)):
            raise ValueError(f"Source-read TRAIN fidelity screen {name} is malformed.")
        return [float(item) for item in result]

    thresholds = numbers("thresholds", 2)
    targets = value.get("target_union_retention")
    if targets != [0.75, 0.50]:
        raise ValueError("Source-read TRAIN screen must report the two fixed 75%/50% targets.")
    union_rows = integer("union_pair_count", minimum=1)
    physical_rows = integer("physical_pair_count", minimum=1)
    integer("query_count", minimum=1)
    protected_rows = integer("near_protected_union_pair_count")
    if protected_rows > union_rows or union_rows > physical_rows:
        raise ValueError("Source-read TRAIN screen pair counts are inconsistent.")
    target_counts = value.get("target_union_pair_counts")
    expected_target_counts = [max(1, math.ceil(fraction * union_rows)) for fraction in (0.75, 0.50)]
    if target_counts != expected_target_counts:
        raise ValueError("Source-read TRAIN screen target row counts do not match its fixed cohort union.")
    active_union = value.get("actual_union_pair_counts")
    active_physical = value.get("actual_physical_pair_counts")
    if (not isinstance(active_union, list) or len(active_union) != 2
            or not isinstance(active_physical, list) or len(active_physical) != 2
            or any(type(item) is not int or not 0 <= item <= union_rows for item in active_union)
            or any(type(item) is not int or not 0 <= item <= physical_rows for item in active_physical)):
        raise ValueError("Source-read TRAIN screen achieved pair counts are malformed.")
    actual_union = numbers("actual_union_retention", 2)
    actual_physical = numbers("actual_physical_pair_retention", 2)
    if any(not math.isclose(actual_union[i], active_union[i] / union_rows, rel_tol=0.0, abs_tol=1e-15)
           or not math.isclose(actual_physical[i], active_physical[i] / physical_rows,
                               rel_tol=0.0, abs_tol=1e-15) for i in range(2)):
        raise ValueError("Source-read TRAIN screen achieved retention differs from its counts.")
    near_limited = value.get("target_infeasible_due_to_near_protection")
    overruns = value.get("actual_target_overrun_pair_counts")
    if (not isinstance(near_limited, list) or near_limited != [
            protected_rows > target_counts[0], protected_rows > target_counts[1]]
            or not isinstance(overruns, list) or len(overruns) != 2
            or any(type(item) is not int or item != max(0, active_union[i] - target_counts[i])
                   for i, item in enumerate(overruns))):
        raise ValueError("Source-read TRAIN screen does not disclose near-protection target infeasibility.")

    flow = numbers("flow_direct_context_relative_rms", 2)
    by_head = value.get("flow_direct_context_head_relative_rms")
    if (not isinstance(by_head, list) or len(by_head) != 2
            or any(not isinstance(row, list) or not row for row in by_head)
            or len(by_head[0]) != len(by_head[1])
            or any(type(item) not in (int, float) or not math.isfinite(item) or item < 0
                   for row in by_head for item in row)):
        raise ValueError("Source-read TRAIN screen headwise context distortion is malformed.")
    thermal_endpoint = value.get("thermal_positive_H_endpoint_relative_rms")
    thermal_kernel = value.get("thermal_kernel_relative_rms")
    if (thermal_endpoint is None) != (thermal_kernel is None):
        raise ValueError("Source-read TRAIN screen Thermal distortion metrics must be paired.")
    if thermal_endpoint is not None:
        thermal_endpoint = numbers("thermal_positive_H_endpoint_relative_rms", 2)
        thermal_kernel = numbers("thermal_kernel_relative_rms", 2)
        if (value.get("thermal_temperature_scale") != 1.0
                or value.get("thermal_target_scale_convention") !=
                "native_core_K_is_TRAIN_standardized_temperature_per_heat; auxiliary_sT=1"):
            raise ValueError("Thermal source-read utility targets must retain the native TRAIN temperature scale sT=1.")
    elif (value.get("thermal_temperature_scale") is not None
          or value.get("thermal_target_scale_convention") is not None):
        raise ValueError("Nonthermal source-read screen cannot declare a Thermal target scale.")
    primary = value.get("primary_operating_point")
    reason = value.get("selection_reason")
    if type(primary) is not int or primary not in (0, 1):
        raise ValueError("Source-read TRAIN screen primary operating point is malformed.")
    if primary == 0 and reason == "least_aggressive_useful_75":
        if active_physical[0] >= physical_rows:
            raise ValueError("TRAIN screen calls the 75% point useful without reducing physical reads.")
    elif primary == 0 and reason == "no_useful_point_default75":
        if active_physical[0] < physical_rows:
            raise ValueError("TRAIN screen reports no useful point despite 75% support savings.")
        metrics_75 = [flow[0], *by_head[0]]
        metrics_50 = [flow[1], *by_head[1]]
        if thermal_endpoint is not None:
            metrics_75.extend((thermal_endpoint[0], thermal_kernel[0]))
            metrics_50.extend((thermal_endpoint[1], thermal_kernel[1]))
        if (active_physical[1] < active_physical[0]
                and all(after <= before + 1.0e-12 for after, before in zip(metrics_50, metrics_75))):
            raise ValueError("TRAIN screen ignored a cheaper 50% point with no worse measured distortion.")
    elif primary == 1 and reason == "75_no_savings_use_50_without_worse_screen":
        if active_physical[0] < physical_rows or active_physical[1] >= active_physical[0]:
            raise ValueError("TRAIN screen selected 50% without a cost improvement over an unhelpful 75% point.")
        metrics_75 = [flow[0], *by_head[0]]
        metrics_50 = [flow[1], *by_head[1]]
        if thermal_endpoint is not None:
            metrics_75.extend((thermal_endpoint[0], thermal_kernel[0]))
            metrics_50.extend((thermal_endpoint[1], thermal_kernel[1]))
        if any(after > before + 1.0e-12 for after, before in zip(metrics_50, metrics_75)):
            raise ValueError("TRAIN screen selected 50% despite worse bounded TRAIN fidelity.")
    else:
        raise ValueError("Source-read TRAIN screen selection reason and primary point disagree.")
    if thresholds[0] <= 0 or thresholds[1] <= 0:
        raise ValueError("Source-read TRAIN screen thresholds must be finite and positive.")


def _inventory(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype), "elements": value.numel()}
        for name, value in sorted(state.items())
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _loaded_source_changes(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, dict[str, str | None]]:
    old = parent.get("files")
    new = child.get("files")
    if not isinstance(old, Mapping) or not isinstance(new, Mapping):
        raise ValueError("Source-read child requires complete parent and child loaded-source file maps.")
    for binding in (parent, child):
        if _canonical_sha256(dict(binding["files"])) != binding.get("sha256"):
            raise ValueError("Source-read parent or child loaded-source digest is malformed.")
        for path, digest in binding["files"].items():
            if (not isinstance(path, str) or not path or Path(path).is_absolute()
                    or ".." in Path(path).parts or not _is_sha256(digest)):
                raise ValueError("Source-read loaded-source entries must bind safe relative paths and SHA256 hashes.")
    result: dict[str, dict[str, str | None]] = {}
    for path in sorted(set(old) | set(new)):
        before, after = old.get(path), new.get(path)
        if before == after:
            continue
        if (not isinstance(path, str) or not path or Path(path).is_absolute()
                or ".." in Path(path).parts):
            raise ValueError("Source-read loaded-source paths must be safe project-relative paths.")
        if path not in _SOURCE_PATH_ALLOWLIST:
            raise ValueError(f"Source-read child changed non-reader project source {path!r}.")
        if before is not None and not _is_sha256(before):
            raise ValueError("Source-read parent loaded-source entry has a malformed hash.")
        if after is not None and not _is_sha256(after):
            raise ValueError("Source-read child loaded-source entry has a malformed hash.")
        # This is an additive architecture child. Deleting an imported source
        # module cannot be explained by the declared sparse controller.
        if after is None:
            raise ValueError("Source-read child cannot remove an imported project source file.")
        result[path] = {"parent_sha256": before, "child_sha256": after}
    return result


def _patch_paths(artifact: Path) -> set[str]:
    try:
        text = artifact.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("Source-read source-patch artifact must be a readable UTF-8 unified diff.") from exc
    paths: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("diff --git "):
            continue
        match = re.fullmatch(r"diff --git a/(.+) b/(.+)", line)
        if match is None or match.group(1) != match.group(2):
            raise ValueError("Source-read source patch cannot contain renames or malformed path headers.")
        if match.group(1) in paths:
            raise ValueError("Source-read source patch contains a duplicate project path.")
        paths.add(match.group(1))
    if not paths:
        raise ValueError("Source-read source patch does not identify changed project files.")
    return paths


def _validate_source_change_declaration(
    declaration: Mapping[str, Any], parent_identity: Mapping[str, Any], child_identity: Mapping[str, Any],
) -> dict[str, dict[str, str | None]]:
    declared = declaration.get("allowed_loaded_source_changes")
    old_binding, new_binding = parent_identity.get("loaded_source"), child_identity.get("loaded_source")
    if not isinstance(old_binding, Mapping) or not isinstance(new_binding, Mapping):
        raise ValueError("Source-read child must retain loaded_source in both sealed identities.")
    actual = _loaded_source_changes(old_binding, new_binding)
    if not actual or not isinstance(declared, Mapping) or dict(declared) != actual:
        raise ValueError("Source-read child loaded-source diff differs from its exact declared allowlist.")
    patch_path = Path(str(declaration.get("source_patch_artifact", "")))
    if not patch_path.is_absolute() or not patch_path.is_file():
        raise ValueError("Source-read child must bind an inspected absolute source-patch artifact.")
    patch_bytes = patch_path.read_bytes()
    patch_sha = hashlib.sha256(patch_bytes).hexdigest()
    if patch_sha != declaration.get("source_patch_sha256") or not _is_sha256(patch_sha):
        raise ValueError("Source-read source-patch artifact SHA256 differs from the declaration.")
    if _patch_paths(patch_path) != set(actual):
        raise ValueError("Inspected source patch paths differ from the sealed loaded-source changes.")
    if (declaration.get("parent_loaded_source_sha256") != old_binding.get("sha256")
            or declaration.get("child_loaded_source_sha256") != new_binding.get("sha256")):
        raise ValueError("Source-read child does not bind both loaded-source manifests.")
    return actual


def _recipe_from_identity(identity: Mapping[str, Any], label: str) -> dict[str, Any]:
    provider = identity.get("provider_identity")
    recipe = provider.get("recipe") if isinstance(provider, Mapping) else None
    if not isinstance(recipe, Mapping):
        recipe = identity.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError(f"Source-read {label} identity lacks its explicit source-read recipe.")
    result = validate_source_read_recipe(recipe)
    if result["protocol"] != SOURCE_READ_DEVELOPMENT:
        raise ValueError("Sparse source-read children can branch only from source_read_development_v1.")
    return result


def _canonicalize_recipe(
    parent: Mapping[str, Any], child: Mapping[str, Any], parent_variant: str, target_variant: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(child))
    if result.get("source_reader_variant") != target_variant:
        raise ValueError("Source-read child recipe variant differs from its declaration.")
    numerics = result.get("source_read_numerics", _MISSING)
    if numerics is not _MISSING:
        if target_variant not in ("S-Utility", "S-Complete") or numerics != SOURCE_READ_NUMERICS:
            raise ValueError("Source-read child may add only deterministic_cuda_v1 numerics metadata.")
        parent_numerics = parent.get("source_read_numerics", _MISSING)
        if parent_numerics is _MISSING:
            result.pop("source_read_numerics")
        elif parent_numerics != numerics:
            raise ValueError("Source-read child changed inherited numerical execution metadata.")
    result["source_reader_variant"] = parent_variant
    host = result.get("dense_host_variant", parent_variant)
    if host != parent_variant:
        raise ValueError("Source-read child must name its exact dense host variant.")
    if "dense_host_variant" not in parent:
        result.pop("dense_host_variant", None)
    else:
        result["dense_host_variant"] = parent["dense_host_variant"]
    parent_model = parent.get("model")
    child_model = result.get("model")
    if not isinstance(parent_model, Mapping) or not isinstance(child_model, dict):
        raise ValueError("Source-read dense parent and child recipes require model mappings.")
    expected_policy = "utility" if target_variant == "S-Utility" else "completion"
    if child_model.get("source_read_policy") != expected_policy:
        raise ValueError("Source-read child policy differs from its declared S-Utility/S-Complete variant.")
    expected_fields = {
        "source_read_policy": expected_policy,
        "source_read_executor": child_model.get("source_read_executor"),
        "source_read_rank": child_model.get("source_read_rank", 16),
        "source_read_scorer_width": child_model.get("source_read_scorer_width", 32),
    }
    for name, value in expected_fields.items():
        if name not in SOURCE_READ_CONFIG_FIELDS or value is None:
            raise ValueError("Source-read child config is missing an explicit selected-function policy.")
        if name == "source_read_policy":
            prior = parent_model.get(name)
            if prior not in (None, "full"):
                raise ValueError("Dense source-read parent unexpectedly declares a sparse policy.")
            if prior is None:
                child_model.pop(name, None)
            else:
                child_model[name] = prior
        elif name not in parent_model:
            child_model.pop(name, None)
        elif child_model.get(name) != parent_model[name]:
            raise ValueError(f"Sparse child changed inherited {name} configuration.")
    return result


def _canonicalize_provider_tree(
    parent: Mapping[str, Any], child: Mapping[str, Any], *, parent_variant: str,
    target_variant: str, child_recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Revert only source-read policy/variant metadata before identity equality."""
    result = copy.deepcopy(dict(child))
    recipe = result.get("recipe")
    parent_recipe = parent.get("recipe")
    if isinstance(recipe, Mapping) and isinstance(parent_recipe, Mapping):
        result["recipe"] = _canonicalize_recipe(parent_recipe, recipe, parent_variant, target_variant)

    expected_policy = "utility" if target_variant == "S-Utility" else "completion"

    def normalize(node: Any, old: Any) -> Any:
        if isinstance(node, Mapping) and isinstance(old, Mapping):
            updated = copy.deepcopy(dict(node))
            for key, value in tuple(updated.items()):
                previous = old.get(key, _MISSING)
                if key == "recipe":
                    # The versioned recipe has a stricter schema-aware
                    # canonicalizer above; never normalize it recursively.
                    continue
                if key == "source_reader_variant":
                    if value != target_variant:
                        raise ValueError("Source-read child identity has inconsistent reader-variant metadata.")
                    if previous is _MISSING:
                        updated.pop(key)
                    else:
                        if previous != parent_variant:
                            raise ValueError("Dense parent source-reader variant is inconsistent.")
                        updated[key] = previous
                elif key == "dense_host_variant":
                    if value != parent_variant:
                        raise ValueError("Source-read child identity names a different dense host.")
                    if previous is _MISSING:
                        updated.pop(key)
                    elif previous != parent_variant:
                        raise ValueError("Dense parent host variant is inconsistent.")
                elif key == "source_read_policy":
                    if value != expected_policy:
                        raise ValueError("Sparse source-read policy metadata disagrees with its S variant.")
                    if previous is _MISSING:
                        updated.pop(key)
                    elif previous != "full":
                        raise ValueError("Dense parent must have full-source policy.")
                    else:
                        updated[key] = previous
                elif key in ("source_read_executor", "source_read_rank", "source_read_scorer_width"):
                    expected = child_recipe["model"].get(key, {
                        "source_read_rank": 16, "source_read_scorer_width": 32,
                    }.get(key))
                    if expected is None or value != expected:
                        raise ValueError(f"Sparse child identity has inconsistent {key} metadata.")
                    if previous is _MISSING:
                        updated.pop(key)
                    elif previous != value:
                        raise ValueError(f"Sparse child changed inherited {key} metadata.")
                elif isinstance(value, Mapping) and isinstance(previous, Mapping):
                    updated[key] = normalize(value, previous)
            return updated
        if isinstance(node, list) and isinstance(old, list) and len(node) == len(old):
            return [normalize(value, before) for value, before in zip(node, old)]
        return copy.deepcopy(node)

    # Nested source-read descriptors and duplicate native model configs use the
    # same narrowly enumerated variant/policy metadata as the recipe.
    normalized = normalize(result, parent)
    assert isinstance(normalized, dict)
    return normalized


_MISSING = object()


def _canonicalize_experiment_identity(
    parent: Mapping[str, Any], child: Mapping[str, Any], *, parent_variant: str,
    target_variant: str, child_recipe: Mapping[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(child))
    parent_recipe = parent.get("recipe")
    if isinstance(parent_recipe, Mapping):
        result["recipe"] = _canonicalize_recipe(parent_recipe, result.get("recipe", {}),
                                                 parent_variant, target_variant)
        if result.get("recipe_sha256") != _canonical_sha256(child["recipe"]):
            raise ValueError("Source-read child recipe hash does not match its active recipe.")
        if parent.get("recipe_sha256") != _canonical_sha256(parent_recipe):
            raise ValueError("Dense parent recipe hash does not match its sealed recipe.")
        result["recipe_sha256"] = parent["recipe_sha256"]
    authorization = result.get("source_read_authorization")
    parent_authorization = parent.get("source_read_authorization")
    if isinstance(authorization, Mapping):
        if not isinstance(parent_authorization, Mapping):
            raise ValueError("Sparse source-read child lacks matching parent authorization metadata.")
        authorization = copy.deepcopy(dict(authorization))
        if authorization.get("variant") != target_variant:
            raise ValueError("Active source-read authorization variant differs from the sparse recipe.")
        authorization["variant"] = parent_variant
        result["source_read_authorization"] = authorization
    provider = result.get("provider_identity")
    parent_provider = parent.get("provider_identity")
    if not isinstance(provider, Mapping) or not isinstance(parent_provider, Mapping):
        raise ValueError("Source-read child identity must seal both provider identities.")
    result["provider_identity"] = _canonicalize_provider_tree(
        parent_provider, provider, parent_variant=parent_variant,
        target_variant=target_variant, child_recipe=child_recipe,
    )
    if isinstance(result.get("recipe"), Mapping):
        if result["provider_identity"].get("recipe") != result["recipe"]:
            raise ValueError("Top-level and provider source-read recipes disagree.")

    # The child adds parameters to the existing ordered optimizer groups. All
    # common names, groups, LR/betas/eps/decay and schedule clocks stay exact.
    old_groups = parent.get("optimizer_schedule_contract")
    new_groups = result.get("optimizer_schedule_contract")
    if not isinstance(old_groups, list) or not isinstance(new_groups, list) or len(old_groups) != len(new_groups):
        raise ValueError("Source-read child must preserve optimizer groups and ordering.")
    for old_group, new_group in zip(old_groups, new_groups):
        if not isinstance(old_group, Mapping) or not isinstance(new_group, dict):
            raise ValueError("Source-read optimizer schedule contract is malformed.")
        new_names = new_group.get("parameter_names")
        old_names = old_group.get("parameter_names")
        if not isinstance(new_names, list) or not isinstance(old_names, list):
            raise ValueError("Source-read optimizer groups require ordered named parameters.")
        if [name for name in new_names if name in set(old_names)] != old_names:
            raise ValueError("Source-read child reordered or replaced a physical optimizer parameter.")
        normalized_group = copy.deepcopy(new_group)
        normalized_group["parameter_names"] = list(old_names)
        if normalized_group != old_group:
            raise ValueError("Source-read child changed an inherited optimizer schedule or group setting.")
    result["optimizer_schedule_contract"] = copy.deepcopy(old_groups)
    # loaded_source is the sole other allowed identity difference and is
    # validated separately against its source patch and exact file delta.
    result["loaded_source"] = copy.deepcopy(parent.get("loaded_source"))
    if result != dict(parent):
        raise ValueError("Source-read child changed an unrelated experiment, objective, data or sampling identity.")
    return result


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def prepare_source_read_child(
    model: nn.Module,
    source_payload: Mapping[str, Any],
    current_identity: Mapping[str, Any],
    declaration: Mapping[str, Any],
    *,
    source_sha256: str,
    source_path: str | Path,
    case_keys: Sequence[Any],
    stop_after: int,
) -> dict[str, Any]:
    """Validate an exact development e500 D parent and copy its physical state."""
    required = {
        "schema", "source_checkpoint", "source_checkpoint_sha256", "source_epoch", "source_arm",
        "source_reader_variant", "dense_host_variant", "added_parameter_names", "review_epochs",
        "final_epoch", "auxiliary_fit_mode", "auxiliary_update_limit", "reason",
        "allowed_loaded_source_changes", "parent_loaded_source_sha256", "child_loaded_source_sha256",
        "source_patch_artifact", "source_patch_sha256",
    }
    if not isinstance(declaration, Mapping) or set(declaration) != required or declaration.get("schema") != SCHEMA:
        raise ValueError("A source-read child requires its exact versioned parent/source declaration.")
    resolved_source = Path(source_path).resolve()
    if (declaration.get("source_checkpoint") != str(resolved_source)
            or declaration.get("source_checkpoint_sha256") != source_sha256
            or not _is_sha256(source_sha256)):
        raise ValueError("Source-read child checkpoint path or SHA256 differs from its declaration.")
    parent_identity = source_payload.get("experiment_identity")
    if not isinstance(parent_identity, Mapping):
        raise ValueError("Source-read parent lacks a sealed experiment identity.")
    parent_recipe = _recipe_from_identity(parent_identity, "parent")
    child_recipe = _recipe_from_identity(current_identity, "child")
    provider_parent = parent_identity.get("provider_identity")
    provider_child = current_identity.get("provider_identity")
    if not isinstance(provider_parent, Mapping) or not isinstance(provider_child, Mapping):
        raise ValueError("Source-read parent and child must seal native provider identities.")
    dataset = parent_recipe.get("dataset")
    parent_variant = parent_recipe.get("source_reader_variant")
    target_variant = declaration.get("source_reader_variant")
    if (dataset not in ("thermal", "wind") or parent_recipe.get("dataset") != child_recipe.get("dataset")
            or parent_recipe.get("architecture") != "A2-D" or child_recipe.get("architecture") != "A2-D"
            or parent_variant not in ("D0", "D-scale")
            or target_variant not in ("S-Utility", "S-Complete")
            or child_recipe.get("source_reader_variant") != target_variant
            or child_recipe.get("dense_host_variant", parent_variant) != parent_variant
            or declaration.get("dense_host_variant") != parent_variant
            or parent_recipe.get("model", {}).get("source_value_conditioning", "none")
                != child_recipe.get("model", {}).get("source_value_conditioning", "none")):
        raise ValueError("Source-read child must be a same-protocol A2-D sparse child of its exact D0/D-scale host.")
    if (provider_parent.get("source_reader_variant") != parent_variant
            or provider_child.get("source_reader_variant") != target_variant
            or provider_parent.get("protocol") != SOURCE_READ_DEVELOPMENT
            or provider_child.get("protocol") != SOURCE_READ_DEVELOPMENT
            or provider_parent.get("architecture") != "A2-D"
            or provider_child.get("architecture") != "A2-D"):
        raise ValueError("Provider identity differs from the declared development A2-D parent/child lineage.")
    if source_payload.get("arm") != "A2-D" or declaration.get("source_arm") != "A2-D":
        raise ValueError("Source-read child must branch from the literal A2-D parent arm.")
    if (type(declaration.get("source_epoch")) is not int or declaration["source_epoch"] != 500
            or type(source_payload.get("epoch")) is not int or source_payload["epoch"] != 500):
        raise ValueError("Source-read child must inherit the exact saved e500 dense prefix.")
    if (not isinstance(declaration.get("reason"), str) or not declaration["reason"].strip()
            or type(declaration.get("auxiliary_update_limit")) is not int
            or not 0 <= declaration["auxiliary_update_limit"] <= MAX_AUXILIARY_UPDATES
            or declaration.get("auxiliary_fit_mode") not in ("calibrated", "initialization_only")):
        raise ValueError("Source-read auxiliary fit mode/reason/update ceiling is malformed.")
    if ((declaration["auxiliary_fit_mode"] == "calibrated") != (declaration["auxiliary_update_limit"] > 0)):
        raise ValueError("Calibrated source-read children need a positive <=256 update ceiling.")
    reviews = declaration.get("review_epochs")
    expected_reviews = list(range(600, 2501, 100))
    if (reviews != expected_reviews or declaration.get("final_epoch") != 2500
            or type(stop_after) is not int or stop_after not in expected_reviews):
        raise ValueError("Source-read child reviews and stop must preserve the 501..2500 development trajectory.")
    parent_cases = provider_parent.get("training_case_ids" if dataset == "thermal" else "train_rows")
    expected_size = 150 if dataset == "thermal" else 72
    if (not isinstance(parent_cases, list) or len(parent_cases) != expected_size
            or len(set(parent_cases)) != expected_size or len(case_keys) != expected_size
            or set(case_keys) != set(parent_cases)):
        raise ValueError("Source-read child cannot alter or resample its fixed TRAIN cohort.")
    for identity in (parent_identity, current_identity):
        if identity.get("engine_config", {}).get("training_mode") != "joint":
            raise ValueError("Source-read development parent/child must use the joint native objective engine.")
    source_history = source_payload.get("history")
    if (not isinstance(source_history, list) or len(source_history) != 500
            or [row.get("epoch") for row in source_history if isinstance(row, Mapping)] != list(range(1, 501))
            or any(not isinstance(row, Mapping) or type(row.get("case_visits")) is not int
                   or type(row.get("optimizer_updates")) is not int for row in source_history)):
        raise ValueError("Source-read parent must retain its complete physical history through e500.")
    sampler = source_payload.get("sampler_state")
    if (not isinstance(sampler, Mapping) or sampler.get("completed_epoch") != 500
            or sampler.get("next_epoch") != 501 or sampler.get("next_case_cursor") != 0):
        raise ValueError("Source-read child parent sampler is not sealed at the absolute e500 boundary.")

    old_state = source_payload.get("model_state_dict")
    if (not isinstance(old_state, Mapping) or not old_state
            or not all(torch.is_tensor(value) for value in old_state.values())):
        raise ValueError("Source-read parent model state must contain literal tensors.")
    current_state = model.state_dict()
    if not all(torch.is_tensor(value) for value in current_state.values()):
        raise ValueError("Source-read child model state must contain literal tensors.")
    parameter_names = {name for name, _ in model.named_parameters()}
    added_state_names = sorted(set(current_state) - set(old_state))
    removed_state_names = set(old_state) - set(current_state)
    added_parameter_names = sorted(parameter_names - set(old_state))
    added_buffer_names = sorted(set(dict(model.named_buffers())) - set(old_state))
    declared_names = declaration.get("added_parameter_names")
    prefix = "core.direct_reader.source_read." if dataset == "thermal" else "direct_reader.source_read."
    if (removed_state_names or not added_parameter_names or declared_names != added_parameter_names
            or any(not name.startswith(prefix) for name in added_state_names)
            or any(not name.startswith(prefix) for name in added_parameter_names)
            or any(not parameter.requires_grad for name, parameter in model.named_parameters()
                   if name in added_parameter_names)):
        raise ValueError("Only explicitly enumerated trainable source_read namespace parameters may be added.")
    for name, value in old_state.items():
        candidate = current_state[name]
        if candidate.shape != value.shape or candidate.dtype != value.dtype:
            raise ValueError(f"Source-read child replaced inherited state {name!r} with another shape/dtype.")
    if any(torch.is_floating_point(current_state[name]) and not torch.isfinite(current_state[name]).all()
           for name in added_state_names):
        raise ValueError("New source-read controller state must be finite at attachment.")

    _validate_source_change_declaration(declaration, parent_identity, current_identity)
    _canonicalize_experiment_identity(
        parent_identity, current_identity, parent_variant=parent_variant,
        target_variant=target_variant, child_recipe=child_recipe,
    )
    old_engine = parent_identity.get("engine_config")
    child_engine = current_identity.get("engine_config")
    if (not isinstance(old_engine, Mapping) or not isinstance(child_engine, Mapping)
            or old_engine != child_engine or old_engine.get("total_epochs") != 2500
            or old_engine.get("sampling_version") != "case_epoch_v1"
            or old_engine.get("effective_cases") != parent_recipe.get("effective_batch_size")
            or old_engine.get("seed") != parent_recipe.get("sampling_seed")):
        raise ValueError("Source-read child changed parent sampler, batch, or absolute schedule age.")

    old_groups = source_payload.get("optimizer_state_by_name", {}).get("groups", [])
    source_schedules = source_payload.get("optimizer_schedule_contract")
    if (source_schedules != parent_identity.get("optimizer_schedule_contract")
            or [group.get("name") for group in old_groups] != source_payload.get("optimizer_group_names")
            or len(old_groups) != len(source_schedules)
            or any(group.get("parameter_names") != schedule.get("parameter_names")
                   for group, schedule in zip(old_groups, source_schedules))):
        raise ValueError("Source-read dense parent named optimizer groups disagree with their sealed schedule.")
    current_groups = current_identity.get("optimizer_schedule_contract")
    parent_names = set(old_state)
    if (not isinstance(current_groups, list)
            or not all(any(name in group.get("parameter_names", ()) for group in current_groups)
                       for name in added_parameter_names)):
        raise ValueError("Every new source-read parameter must belong to the inherited joint optimizer group.")
    for before, after in zip(old_groups, current_groups):
        if [name for name in after.get("parameter_names", ()) if name in parent_names] != before["parameter_names"]:
            raise ValueError("Source-read child changed ordered physical optimizer membership.")
    old_optimizer = source_payload.get("optimizer_state_by_name", {}).get("state_by_name", {})
    if not isinstance(old_optimizer, Mapping) or set(old_optimizer) - set(old_state):
        raise ValueError("Source-read parent optimizer contains moments outside its physical model.")

    merged = {**dict(current_state), **dict(old_state)}
    model.load_state_dict(merged, strict=True)
    copied = model.state_dict()
    for name, value in old_state.items():
        if not _same_tensor(copied[name], value):
            raise ValueError(f"Source-read child did not copy inherited tensor {name!r} exactly.")
    child_new_state = {name: copied[name].detach().cpu().clone() for name in added_state_names}
    return {
        **copy.deepcopy(dict(declaration)),
        "dataset": dataset,
        "parent_source_reader_variant": parent_variant,
        "child_source_reader_variant": target_variant,
        "source_read_parameter_prefix": prefix,
        "inherited_parameter_inventory": _inventory(old_state),
        "new_source_read_state_inventory": _inventory(child_new_state),
        "new_source_read_buffer_names": added_buffer_names,
        "new_source_read_buffer_initial_sha256": _state_digest({name: copied[name]
                                                                   for name in added_buffer_names}),
        "new_parameter_names": added_parameter_names,
        "new_parameter_initial_state_sha256": _state_digest({name: child_new_state[name]
                                                               for name in added_parameter_names}),
        "inherited_state_sha256": _state_digest(old_state),
        "child_initial_state_sha256": _state_digest(copied),
        "exact_parent_state_copied": True,
        "inherited_case_visits": sum(int(row["case_visits"]) for row in source_history),
        "inherited_optimizer_updates": sum(int(row["optimizer_updates"]) for row in source_history),
        "updates_per_epoch": math.ceil(expected_size / int(old_engine["effective_cases"])),
        "physical_history_policy": (
            "preserve dense parent history through e500 then append only physical epochs 501..2500"
        ),
        "auxiliary_accounting_policy": (
            "separate source_read_child receipt; never included in physical history or TRAIN visits"
        ),
        "optimizer_mapping": (
            "exact inherited named AdamW moments; source-read moments age only through explicit "
            "auxiliary/new physical updates"
        ),
        "source_changes": _loaded_source_changes(parent_identity["loaded_source"],
                                                   current_identity["loaded_source"]),
    }


def verify_source_read_child_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    source_named_state: Mapping[str, Any],
    new_names: Sequence[str],
    *,
    expected_new_steps: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Verify unchanged common moments and true birth-age state for new names."""
    parameters = dict(model.named_parameters())
    actual = {name: optimizer.state.get(parameters[name]) for name in parameters}
    expected_common = source_named_state.get("state_by_name", {})
    if not isinstance(expected_common, Mapping):
        raise ValueError("Source-read parent optimizer state is malformed.")
    for name, expected in expected_common.items():
        if name not in parameters or actual[name] is None or set(actual[name]) != set(expected):
            raise ValueError(f"Inherited source-read optimizer state is missing for {name!r}.")
        for key, value in expected.items():
            got = actual[name][key]
            equal = (_same_tensor(got, value)
                     if torch.is_tensor(value) else got == value)
            if not equal:
                raise ValueError(f"Inherited source-read optimizer state changed for {name!r}/{key}.")
    new_names = tuple(new_names)
    if expected_new_steps is None:
        if any(parameters[name] in optimizer.state for name in new_names):
            raise ValueError("New source-read AdamW state must begin absent, with no invented e500 age.")
        return {}
    if set(expected_new_steps) - set(new_names):
        raise ValueError("Auxiliary optimizer-step receipt names an undeclared source-read parameter.")
    observed: dict[str, int] = {}
    for name in new_names:
        state = optimizer.state.get(parameters[name])
        expected_step = expected_new_steps.get(name)
        if expected_step is None:
            if state is not None:
                raise ValueError(f"Unreported auxiliary optimizer state exists for {name!r}.")
            continue
        if type(expected_step) is not int or not 1 <= expected_step <= MAX_AUXILIARY_UPDATES:
            raise ValueError("Auxiliary source-read Adam steps must be fresh positive counts at most256.")
        if not isinstance(state, Mapping) or "step" not in state:
            raise ValueError(f"Auxiliary source-read parameter {name!r} lacks its real Adam step state.")
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"Auxiliary source-read Adam state for {name!r} differs from the fixed AdamW contract.")
        step = int(state["step"].item()) if torch.is_tensor(state["step"]) else int(state["step"])
        if step != expected_step:
            raise ValueError(f"Auxiliary source-read Adam age for {name!r} differs from its receipt.")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = state[key]
            if (not torch.is_tensor(moment) or moment.shape != parameters[name].shape
                    or moment.dtype != parameters[name].dtype or not torch.isfinite(moment).all()):
                raise ValueError(f"Auxiliary source-read Adam moment {name!r}/{key} is malformed or nonfinite.")
        observed[name] = step
    return observed


def source_read_child_provider_state(provider, source_state: Mapping[str, Any]) -> dict[str, Any]:
    """Require native provider calibration/state to be exactly inherited."""
    if not isinstance(source_state, Mapping):
        raise ValueError("Source-read dense parent provider state must be a mapping.")
    mapped = copy.deepcopy(dict(source_state))
    state_method = getattr(provider, "training_state_dict", None)
    current = dict(state_method()) if callable(state_method) else {}
    if mapped != current:
        raise ValueError("Source-read child changed native provider state or TRAIN-fitted calibration.")
    return mapped


def _group_snapshot(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    return [{key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
            | {"params": tuple(id(parameter) for parameter in group["params"])}
            for group in optimizer.param_groups]


def run_source_read_child_auxiliary_fit(
    callback: Callable[..., Mapping[str, Any]] | None,
    model: nn.Module,
    provider,
    optimizer: torch.optim.Optimizer,
    source_payload: Mapping[str, Any],
    declaration: Mapping[str, Any],
    *,
    added_parameter_names: Sequence[str],
    train_case_keys: Sequence[Any],
    rng_devices: Sequence[int] = (),
) -> dict[str, Any]:
    """Run an explicit callback on new params only, preserving parent RNG/state.

    The callback owns its target preparation and <=256-update auxiliary fit.
    This wrapper does not build labels or add a teacher-training loop.
    """
    mode = declaration.get("auxiliary_fit_mode")
    limit = declaration.get("auxiliary_update_limit")
    if mode == "initialization_only":
        if callback is not None or limit != 0:
            raise ValueError("Initialization-only source-read children cannot run auxiliary calibration.")
        return {
            "schema": AUXILIARY_FIT_SCHEMA,
            "mode": "initialization_only",
            "partition": "TRAIN",
            "source_checkpoint_sha256": declaration["source_checkpoint_sha256"],
            "source_epoch": 500,
            "train_case_keys": list(train_case_keys),
            "train_case_count": len(train_case_keys),
            "auxiliary_fit_case_keys": [],
            "auxiliary_fit_case_count": 0,
            "auxiliary_fit_case_share": 0.0,
            "teacher_rows": 0,
            "optimizer_updates": 0,
            "updated_parameter_names": [],
            "changed_buffer_names": [],
            "controller_buffer_changes": {},
            "optimizer_steps_by_parameter": {},
            "optimizer_state_policy": "new source-read AdamW state absent until its first physical update",
        }
    if mode != "calibrated" or type(limit) is not int or not 1 <= limit <= MAX_AUXILIARY_UPDATES:
        raise ValueError("Source-read child auxiliary fit mode must explicitly declare a <=256 update cap.")
    if not callable(callback):
        raise ValueError("Calibrated source-read children require the provider/core-owned auxiliary-fit callback.")

    old_state = source_payload.get("model_state_dict", {})
    common_optimizer = source_payload.get("optimizer_state_by_name", {})
    parameters = dict(model.named_parameters())
    names = tuple(added_parameter_names)
    name_set = set(names)
    before_new = {name: parameters[name].detach().cpu().clone() for name in names}
    buffers = dict(model.named_buffers())
    new_buffer_names = declaration.get("new_source_read_buffer_names")
    if (not isinstance(new_buffer_names, list) or new_buffer_names != sorted(set(new_buffer_names))
            or any(name not in buffers for name in new_buffer_names)):
        raise ValueError("Source-read child has malformed new controller-buffer inventory.")
    if _state_digest({name: buffers[name] for name in new_buffer_names}) != declaration.get(
            "new_source_read_buffer_initial_sha256"):
        raise ValueError("Source-read controller buffers changed before auxiliary calibration began.")
    before_buffers = {name: buffers[name].detach().cpu().clone() for name in new_buffer_names}
    state_names_before = set(model.state_dict())
    group_before = _group_snapshot(optimizer)
    requires_grad = {name: parameter.requires_grad for name, parameter in parameters.items()}
    for name, parameter in parameters.items():
        parameter.grad = None
        parameter.requires_grad_(name in name_set)
    rng_state = None
    try:
        model.train(True)
        # torch.random.fork_rng covers device generators; explicit capture and
        # restoration also preserves Python and NumPy generators used by a
        # provider-owned calibration callback.
        with torch.random.fork_rng(devices=list(rng_devices), enabled=True):
            from .unified_training import capture_rng_state, restore_rng_state

            rng_state = capture_rng_state()
            receipt = callback(
                model, provider, optimizer,
                added_parameter_names=names,
                train_case_keys=tuple(train_case_keys),
                max_updates=limit,
                parent_sha256=declaration["source_checkpoint_sha256"],
                source_epoch=500,
            )
            restore_rng_state(rng_state)
    finally:
        for name, parameter in parameters.items():
            parameter.requires_grad_(requires_grad[name])
            parameter.grad = None
        if rng_state is not None:
            from .unified_training import restore_rng_state

            restore_rng_state(rng_state)
    if _group_snapshot(optimizer) != group_before:
        raise ValueError("Source-read auxiliary callback changed joint optimizer group/schedule settings.")
    parameters_after = dict(model.named_parameters())
    buffers_after = dict(model.named_buffers())
    if (set(model.state_dict()) != state_names_before
            or set(parameters_after) != set(parameters)
            or any(parameters_after[name] is not parameter for name, parameter in parameters.items())
            or set(buffers_after) != set(buffers)
            or any(buffers_after[name] is not buffer for name, buffer in buffers.items())):
        raise ValueError("Source-read auxiliary callback replaced parameters or changed model-state structure.")
    buffers = buffers_after
    if not isinstance(receipt, Mapping):
        raise ValueError("Source-read auxiliary callback must return its compact receipt mapping.")
    required_receipt = {
        "schema", "partition", "source_checkpoint_sha256", "source_epoch", "train_case_keys",
        "train_case_count", "auxiliary_fit_case_keys", "auxiliary_fit_case_count",
        "auxiliary_fit_case_share", "teacher_rows", "optimizer_updates",
        "updated_parameter_names", "changed_buffer_names", "optimizer_steps_by_parameter",
        "optimizer_algorithm", "train_fidelity_screen", "gradient_calibration",
    }
    if (not required_receipt.issubset(receipt)
            or set(receipt) - (required_receipt | {"teacher_present_rows"})
            or receipt.get("schema") != AUXILIARY_FIT_SCHEMA):
        raise ValueError("Source-read auxiliary callback receipt fields/schema are not exact.")
    train_ids = list(train_case_keys)
    train_case_count = receipt.get("train_case_count")
    updated = receipt.get("updated_parameter_names")
    steps = receipt.get("optimizer_steps_by_parameter")
    updates = receipt.get("optimizer_updates")
    teacher_rows = receipt.get("teacher_rows")
    teacher_present_rows = receipt.get("teacher_present_rows")
    fit_case_keys = receipt.get("auxiliary_fit_case_keys")
    fit_case_count = receipt.get("auxiliary_fit_case_count")
    fit_case_share = receipt.get("auxiliary_fit_case_share")
    changed_buffers = receipt.get("changed_buffer_names")
    optimizer_algorithm = receipt.get("optimizer_algorithm")
    gradient_calibration = receipt.get("gradient_calibration")
    if (not isinstance(gradient_calibration, Mapping)
            or gradient_calibration.get("schema") != "source_read_gradient_calibration_v1"
            or gradient_calibration.get("partition") != "TRAIN"):
        raise ValueError("Source-read auxiliary fit lacks its fixed TRAIN gradient-calibration receipt.")
    try:
        physical_gradient_norm = float(gradient_calibration["physical_all_inherited_parameter_gradient_l2_norm"])
        auxiliary_gradient_norm = float(gradient_calibration["auxiliary_all_new_parameter_gradient_l2_norm"])
        fixed_coefficient = float(gradient_calibration["fixed_auxiliary_coefficient"])
        realized_gradient_share = float(
            gradient_calibration["realized_weighted_auxiliary_to_physical_gradient_share"]
        )
        maximum_gradient_share = float(gradient_calibration["maximum_gradient_share"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Source-read auxiliary gradient-calibration receipt is incomplete.") from exc
    expected_coefficient = (
        min(1.0, 0.05 * physical_gradient_norm / auxiliary_gradient_norm)
        if auxiliary_gradient_norm > 0 else float("nan")
    )
    expected_gradient_share = (
        fixed_coefficient * auxiliary_gradient_norm / physical_gradient_norm
        if physical_gradient_norm > 0 else 0.0
    )
    if (not all(math.isfinite(value) for value in (
            physical_gradient_norm, auxiliary_gradient_norm, fixed_coefficient,
            realized_gradient_share, maximum_gradient_share))
            or physical_gradient_norm < 0 or auxiliary_gradient_norm <= 0
            or not 0.0 <= fixed_coefficient <= 1.0
            or maximum_gradient_share != 0.05
            or fixed_coefficient != expected_coefficient
            or realized_gradient_share != expected_gradient_share
            or realized_gradient_share < 0.0
            or realized_gradient_share > 0.05 + 1.0e-12):
        raise ValueError("Source-read auxiliary gradient-calibration receipt violates the fixed 5% TRAIN rule.")
    if (not isinstance(optimizer_algorithm, Mapping)
            or dict(optimizer_algorithm) != AUXILIARY_OPTIMIZER_ALGORITHM):
        raise ValueError("Source-read auxiliary optimizer algorithm differs from the fixed TRAIN procedure.")
    _validate_train_fidelity_screen(receipt.get("train_fidelity_screen"))
    actual_changed_buffers = []
    for name, before in before_buffers.items():
        after = buffers[name].detach().cpu()
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Source-read auxiliary callback replaced controller buffer {name!r}.")
        if not _same_tensor(after, before):
            actual_changed_buffers.append(name)
    if (receipt.get("partition") != "TRAIN"
            or receipt.get("source_checkpoint_sha256") != declaration["source_checkpoint_sha256"]
            or receipt.get("source_epoch") != 500
            or receipt.get("train_case_keys") != train_ids
            or type(train_case_count) is not int or train_case_count != len(train_ids)
            or not isinstance(fit_case_keys, list) or not fit_case_keys
            or len(set(fit_case_keys)) != len(fit_case_keys)
            or set(fit_case_keys) - set(train_ids)
            or type(fit_case_count) is not int or fit_case_count != len(fit_case_keys)
            or type(fit_case_share) is not float
            or not math.isclose(fit_case_share, fit_case_count / train_case_count,
                                rel_tol=0.0, abs_tol=1e-15)
            or type(teacher_rows) is not int or teacher_rows <= 0
            or (teacher_present_rows is not None
                and (type(teacher_present_rows) is not int
                     or not 0 <= teacher_present_rows <= teacher_rows))
            or type(updates) is not int or not 1 <= updates <= limit
            or not isinstance(updated, list) or not updated or updated != sorted(set(updated))
            or set(updated) - name_set
            or not isinstance(steps, Mapping)
            or not isinstance(changed_buffers, list)
            or changed_buffers != sorted(set(changed_buffers))
            or changed_buffers != sorted(actual_changed_buffers)
            or set(changed_buffers) - set(new_buffer_names)):
        raise ValueError("Source-read auxiliary fit receipt exceeds its declared TRAIN-only boundary.")
    if set(steps) != set(updated) or any(type(value) is not int or not 1 <= value <= updates
                                         for value in steps.values()):
        raise ValueError("Source-read auxiliary fit Adam-step map differs from its updated parameter names.")
    for name, value in old_state.items():
        if not _same_tensor(model.state_dict()[name], value):
            raise ValueError(f"Source-read auxiliary fit changed inherited physical tensor {name!r}.")
    for name, initial in before_new.items():
        if not torch.isfinite(parameters[name]).all():
            raise ValueError(f"Source-read auxiliary callback produced nonfinite parameter {name!r}.")
        if not torch.equal(parameters[name].detach().cpu(), initial) and name not in set(updated):
            raise ValueError(f"Source-read auxiliary callback changed unreported parameter {name!r}.")
    buffer_changes = {}
    for name, initial in before_buffers.items():
        value = buffers[name]
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise ValueError(f"Source-read auxiliary callback produced nonfinite controller buffer {name!r}.")
        if name in actual_changed_buffers:
            buffer_changes[name] = {
                "before_sha256": _state_digest({name: initial}),
                "after_sha256": _state_digest({name: value.detach().cpu()}),
            }
            if buffer_changes[name]["before_sha256"] == buffer_changes[name]["after_sha256"]:
                raise ValueError(f"Source-read controller buffer {name!r} changed without a byte-level difference.")
    verified_steps = verify_source_read_child_optimizer(
        model, optimizer, common_optimizer, names, expected_new_steps=steps,
    )
    if set(verified_steps) != set(updated):
        raise ValueError("Source-read auxiliary callback did not create exactly its reported Adam moments.")
    source_read_child_provider_state(provider, source_payload.get("provider_training_state", {}))
    return {
        **copy.deepcopy(dict(receipt)),
        "mode": "calibrated",
        "updated_parameter_names": list(updated),
        "changed_buffer_names": list(changed_buffers),
        "controller_buffer_changes": buffer_changes,
        "train_case_count": train_case_count,
        "auxiliary_fit_case_count": fit_case_count,
        "auxiliary_fit_case_share": fit_case_share,
        "optimizer_steps_by_parameter": dict(verified_steps),
        "optimizer_state_policy": "joint AdamW; inherited moments exact, new moments begin at auxiliary birth",
        "physical_rng_restored": True,
        "inherited_model_state_unchanged": True,
        "inherited_optimizer_state_unchanged": True,
    }


def verify_source_read_child_continuation(contract: Mapping[str, Any], stop_after: int) -> None:
    if (contract.get("schema") != SCHEMA
            or not isinstance(contract.get("review_epochs"), list)
            or stop_after not in contract["review_epochs"]
            or stop_after > contract.get("final_epoch", 0)):
        raise ValueError("Source-read child continuation must stay within its sealed reviews and update horizon.")


__all__ = [
    "AUXILIARY_FIT_SCHEMA", "AUXILIARY_OPTIMIZER_ALGORITHM", "MAX_AUXILIARY_UPDATES", "SCHEMA",
    "prepare_source_read_child", "run_source_read_child_auxiliary_fit",
    "source_read_child_provider_state", "verify_source_read_child_continuation",
    "verify_source_read_child_optimizer",
]
