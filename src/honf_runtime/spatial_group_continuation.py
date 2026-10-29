"""Exact-state F13 continuation with explicit, nonscientific horizon changes."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping


WORKFLOW = "spatial_interaction_groups_joint_training_v1"


def spatial_group_horizon_amendment(saved: Mapping, current: Mapping) -> dict:
    """Compare the whole identity after replacing only declared future metadata.

    Model/data/objective, optimizer groups and their original constant schedule,
    sampler, normalization, selection, and output cadence remain exact matches.
    New checkpoints retain the actual future recipe rather than masquerading
    as the old hundred-epoch run.
    """
    old, new = (json.loads(json.dumps(value)) for value in (saved, current))
    message = "F13 full-state continuation changed scientific identity or its declared future horizon."
    if old.get("workflow") != WORKFLOW or new.get("workflow") != WORKFLOW:
        raise ValueError(message)
    before = old.get("engine_config", {}).get("total_epochs")
    after = new.get("engine_config", {}).get("total_epochs")
    if (type(before) is not int or type(after) is not int
            or before != 100 or after != 500
            or old.get("training_membership") != "fixed25_train150_dev22"
            or old.get("optimizer_schedule_contract") != new.get("optimizer_schedule_contract")):
        raise ValueError(message)
    old_provider, new_provider = old.get("provider_identity", {}), new.get("provider_identity", {})
    old_recipe, new_recipe = old_provider.get("recipe", {}), new_provider.get("recipe", {})
    declaration = new_recipe.get("continuation")
    if (not isinstance(declaration, dict) or set(declaration) != {"mode", "source_checkpoint", "source_epoch"}
            or declaration.get("mode") != "full_state" or declaration.get("source_epoch") != before
            or not isinstance(declaration.get("source_checkpoint"), str)
            or old_provider.get("architecture") not in ("F13-0", "F13-1")
            or new_provider.get("architecture") != old_provider.get("architecture")):
        raise ValueError(message)
    schedules = old.get("optimizer_schedule_contract", ())
    if not schedules:
        raise ValueError(message)
    for group in schedules:
        schedule = group.get("schedule", {})
        if (schedule.get("peak_lr") != 3e-4 or schedule.get("final_lr") != 3e-4
                or schedule.get("warmup_start_lr") != 3e-4 or schedule.get("warmup_epochs") != 0
                or schedule.get("total_epochs") != before):
            raise ValueError(message)
    training = new_recipe.get("training", {})
    expected_milestones = list(range(100, after + 1, 100))
    if (training.get("optimizer_schedule_horizon") != before
            or training.get("local_horizon") != after
            or training.get("total_epochs") != after
            or training.get("first_segment_epochs") != after
            or training.get("checkpoint_epochs") != expected_milestones
            or new.get("engine_config", {}).get("checkpoint_epochs") != expected_milestones
            or old.get("engine_config", {}).get("checkpoint_epochs") != list(range(100, before + 1, 100))):
        raise ValueError(message)
    changed = []
    amended = copy.deepcopy(old)

    def replace(container, proposed, key, path):
        if key not in proposed:
            raise ValueError(message)
        if container.get(key) != proposed[key]:
            changed.append(path)
        container[key] = copy.deepcopy(proposed[key])

    replace(amended, new, "config_name", "config_name")
    for key in ("absolute_horizon", "wall_clock_budget_seconds", "clean_save_reserve_seconds"):
        replace(amended["recipe"], new["recipe"], key, "recipe." + key)
    for key in ("total_epochs", "checkpoint_epochs"):
        replace(amended["engine_config"], new["engine_config"], key, "engine_config." + key)
    replace(amended["provider_identity"]["training"], new_provider["training"], "total_epochs",
            "provider_identity.training.total_epochs")
    active = amended["provider_identity"]["recipe"]
    replace(active, new_recipe, "name", "provider_identity.recipe.name")
    replace(active, new_recipe, "continuation", "provider_identity.recipe.continuation")
    storage_keys = {"output_root", "max_output_bytes", "max_peak_temporary_bytes", "minimum_free_headroom_bytes"}
    if set(new_recipe.get("storage", {})) != storage_keys:
        raise ValueError(message)
    replace(active, new_recipe, "storage", "provider_identity.recipe.storage")
    for key in ("local_horizon", "first_segment_epochs", "total_epochs", "checkpoint_epochs",
                "wall_clock_budget_seconds", "clean_save_reserve_seconds", "optimizer_schedule_horizon"):
        replace(active["training"], training, key, "provider_identity.recipe.training." + key)
    if amended != new:
        raise ValueError(message)
    return {
        "kind": "explicit_spatial_group_full_state_continuation",
        "previous_horizon": before,
        "execution_horizon": after,
        "optimizer_schedule_unchanged": True,
        "permitted_future_metadata_changes": changed,
        "source_checkpoint": declaration["source_checkpoint"],
    }
