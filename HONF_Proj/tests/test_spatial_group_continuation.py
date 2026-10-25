from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from honf_runtime.spatial_group_continuation import spatial_group_horizon_amendment
from honf_runtime.unified_training import _resume_identity_amendment


CONFIGS = Path(__file__).resolve().parents[1] / "configs/research"


def _identities():
    old_recipe = json.loads((CONFIGS / "thermal_spatial_groups_f13_1_dev_v1.json").read_text())
    new_recipe = json.loads((CONFIGS / "thermal_spatial_groups_f13_1_dev_mature_v1.json").read_text())
    schedule = {"peak_lr": 3e-4, "warmup_start_lr": 3e-4, "warmup_epochs": 0,
                "hold_through_epoch": 99, "total_epochs": 100, "final_lr": 3e-4}
    old = {
        "workflow": "spatial_interaction_groups_joint_training_v1",
        "training_membership": "fixed25_train150_dev22",
        "config_name": old_recipe["name"],
        "recipe": {"absolute_horizon": 100,
                   "wall_clock_budget_seconds": old_recipe["training"]["wall_clock_budget_seconds"],
                   "clean_save_reserve_seconds": 600},
        "engine_config": {"total_epochs": 100, "checkpoint_epochs": [100],
                          "seed": 0, "effective_cases": 48, "microbatch_cases": 24},
        "optimizer_schedule_contract": [{"name": "joint", "schedule": schedule,
                                         "parameter_names": ["core.weight"]}],
        "provider_identity": {"architecture": "F13-1", "recipe": old_recipe,
                              "training": {"total_epochs": 100, "optimizer_schedule": schedule},
                              "training_case_ids": ["train-a"], "validation_case_ids": ["dev-a"],
                              "adapter_state": {"heat_scale": 1.3164895030821728}},
        "sampler_epoch_mapping": {"parent_age": 0},
        "selection_policy": {"field_metric": "field_score"},
    }
    new = copy.deepcopy(old)
    new["config_name"] = new_recipe["name"]
    new["recipe"].update(absolute_horizon=500,
                         wall_clock_budget_seconds=new_recipe["training"]["wall_clock_budget_seconds"])
    new["engine_config"].update(total_epochs=500, checkpoint_epochs=list(range(100, 501, 100)))
    new["provider_identity"]["recipe"] = new_recipe
    new["provider_identity"]["training"]["total_epochs"] = 500
    return old, new


def test_explicit_extension_retains_actual_future_recipe_and_original_schedule():
    old, new = _identities()
    original = copy.deepcopy(old)
    result = _resume_identity_amendment(old, new, allow_horizon_extension=True)
    assert result["kind"] == "explicit_spatial_group_full_state_continuation"
    assert result["optimizer_schedule_unchanged"] is True
    assert result["previous_horizon"] == 100 and result["execution_horizon"] == 500
    assert old == original
    assert new["provider_identity"]["training"]["total_epochs"] == 500
    assert new["provider_identity"]["training"]["optimizer_schedule"]["total_epochs"] == 100
    with pytest.raises(ValueError):
        _resume_identity_amendment(old, new)


@pytest.mark.parametrize("field", ["seed", "cases", "normalizer", "objective", "schedule", "sampling", "mode"])
def test_future_horizon_cannot_mask_changed_scientific_contract(field):
    old, new = _identities()
    if field == "seed":
        new["engine_config"]["seed"] = 1
    elif field == "cases":
        new["provider_identity"]["validation_case_ids"] = ["other-dev"]
    elif field == "normalizer":
        new["provider_identity"]["adapter_state"]["heat_scale"] = 2.0
    elif field == "objective":
        new["provider_identity"]["recipe"]["objective"]["paired_utility_huber_coefficient"] = 2.0
    elif field == "schedule":
        new["optimizer_schedule_contract"][0]["schedule"]["total_epochs"] = 500
    elif field == "sampling":
        new["sampler_epoch_mapping"]["parent_age"] = 100
    else:
        new["provider_identity"]["recipe"]["continuation"]["mode"] = "fresh_adamw"
    with pytest.raises(ValueError, match="F13 full-state continuation"):
        spatial_group_horizon_amendment(old, new)


def test_existing_resume_remains_exact_without_amendment():
    old, _ = _identities()
    assert _resume_identity_amendment(old, copy.deepcopy(old)) is None
