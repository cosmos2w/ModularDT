"""R13-9 keeps the native matched recipe and starts with its full horizon."""
import copy
from pathlib import Path

import pytest

from honf_runtime.contextual_frontier_training import (
    _engine_config,
    _validate_loaded_contextual_config,
    load_contextual_frontier_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "research"


@pytest.mark.parametrize("name,locator,calibration,membership", [
    ("thermal_r13_9a_e_anchor_inherited_train150_v1", "e_anchor", "inherited", "fixed25_train150_dev22"),
    ("thermal_r13_9a_e_anchor_bounded_tanh_train150_v1", "e_anchor", "bounded_tanh", "fixed25_train150_dev22"),
    ("thermal_r13_9l_local_e_cover_train150_v1", "local_e_cover", "inherited", "fixed25_train150_dev22"),
    ("thermal_r13_9a_e_anchor_train600_formal_v1", "e_anchor", "inherited", "full_train600"),
    ("thermal_r13_9l_local_e_cover_train600_formal_v1", "local_e_cover", "inherited", "full_train600"),
])
def test_scratch_recipe_and_resume_registration(name, locator, calibration, membership):
    config, _ = load_contextual_frontier_config(CONFIGS / (name + ".json"))
    _validate_loaded_contextual_config(config)
    training = config["training"]
    assert (training["membership"], training["local_horizon"], training["first_segment_epochs"],
            training["optimizer_schedule_horizon"]) == (membership, 1000, 100, 1000)
    assert training["checkpoint_epochs"] == list(range(100, 1001, 100))
    assert training["wall_clock_budget_seconds"] == 200000
    assert _engine_config(config).total_epochs == 1000
    assert (config["member_routing"]["model"]["locator_mode"],
            config["member_routing"]["model"]["score_calibration"]) == (locator, calibration)
    assert config["member_routing"]["model"]["retained_mass"] == 0.95
    assert config["initialization"]["source_checkpoint"] is None
    assert config["objective"]["paired_full_restricted"] is True
    assert config["optimizer"]["schedule"] == "constant_config_lr"
    malformed = copy.deepcopy(config)
    malformed["training"]["optimizer_schedule_horizon"] = 500
    with pytest.raises(ValueError):
        _validate_loaded_contextual_config(malformed)


def test_qualification_comparison_changes_only_score_law_and_identity():
    inherited, _ = load_contextual_frontier_config(CONFIGS / "thermal_r13_9a_e_anchor_inherited_train150_v1.json")
    bounded, _ = load_contextual_frontier_config(CONFIGS / "thermal_r13_9a_e_anchor_bounded_tanh_train150_v1.json")
    bounded["name"] = inherited["name"]
    bounded["member_routing"]["model"]["score_calibration"] = "inherited"
    assert bounded == inherited


def test_same_horizon_resume_rejects_optimizer_schedule_or_provider_changes():
    from honf_runtime.unified_training import _resume_identity_amendment

    config, _ = load_contextual_frontier_config(
        CONFIGS / "thermal_r13_9a_e_anchor_train600_formal_v1.json"
    )
    identity = {
        "workflow": "interaction_reused_e_anchored_member_routing_training_v1",
        "engine_config": {"total_epochs": _engine_config(config).total_epochs,
                          "checkpoint_epochs": training_checkpoints(config)},
        "optimizer_schedule": copy.deepcopy(config["optimizer"]),
        "provider_identity": {"recipe": {"training": copy.deepcopy(config["training"])}},
    }
    assert _resume_identity_amendment(identity, copy.deepcopy(identity)) is None
    changed = copy.deepcopy(identity)
    changed["optimizer_schedule"]["lr"] *= 2
    with pytest.raises(ValueError, match="identity"):
        _resume_identity_amendment(identity, changed)
    changed = copy.deepcopy(identity)
    changed["provider_identity"]["recipe"]["training"]["seed"] = 1
    with pytest.raises(ValueError, match="identity"):
        _resume_identity_amendment(identity, changed)


def training_checkpoints(config):
    return config["training"]["checkpoint_epochs"]


@pytest.mark.parametrize("name", [
    "thermal_r13_8a_e_anchor_train150_e500_v1",
    "thermal_r13_8l_local_e_cover_train150_e500_v1",
    "thermal_r13_8a_e_anchor_train600_formal_v1",
    "thermal_r13_8l_local_e_cover_train600_formal_v1",
])
def test_historical_registration_keeps_existing_contract(name):
    config, _ = load_contextual_frontier_config(CONFIGS / (name + ".json"))
    _validate_loaded_contextual_config(config)
    assert config["member_routing"]["model"]["score_calibration"] == "inherited"
    assert config["training"]["optimizer_schedule_horizon"] == 500
    changed = copy.deepcopy(config)
    changed["member_routing"]["model"]["score_calibration"] = "bounded_tanh"
    with pytest.raises(ValueError):
        _validate_loaded_contextual_config(changed)
