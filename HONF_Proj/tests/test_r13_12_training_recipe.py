"""R13-12 discovery, independent confirmation and formal-resume contracts."""
import copy
from pathlib import Path

import pytest

from honf_runtime.contextual_frontier_training import (
    _anchored_schedule_and_model, _validate_loaded_contextual_config, load_contextual_frontier_config,
)


CONFIGS = Path(__file__).resolve().parents[1] / "configs/research"


@pytest.mark.parametrize("mode", ("p", "g"))
@pytest.mark.parametrize("population,phase,seed", ((150, "discovery", 0), (150, "confirmation", 1),
                                               (600, "formal", 0)))
def test_declared_recipe_loads_and_survives_saved_config_validation(mode, population, phase, seed):
    path = CONFIGS / f"thermal_r13_12{mode}_e_anchor_mediated_train{population}_{phase}_v1.json"
    config, _ = load_contextual_frontier_config(path)
    _validate_loaded_contextual_config(config)
    assert config["initialization"]["seed"] == config["training"]["seed"] == seed
    assert config["member_routing"]["model"]["base_model_config"]["init_seed"] == seed
    assert config["training"]["local_horizon"] == config["training"]["optimizer_schedule_horizon"] == 1000
    assert config["training"]["first_segment_epochs"] == (100 if population == 600 else 200)
    assert config["training"]["checkpoint_epochs"] == list(range(100, 1001, 100))


def test_independent_confirmation_seed_does_not_relax_historical_or_formal_seed():
    path = CONFIGS / "thermal_r13_12p_e_anchor_mediated_train600_formal_v1.json"
    config, _ = load_contextual_frontier_config(path)
    config["initialization"].update(seed=1, source_arm="scratch_common_seed1")
    config["training"]["seed"] = 1
    with pytest.raises(ValueError, match="scratch contract"):
        _validate_loaded_contextual_config(config)
    historical, _ = load_contextual_frontier_config(CONFIGS / "thermal_r13_11s_e_anchor_train150_v1.json")
    historical["initialization"].update(seed=1, source_arm="scratch_common_seed1")
    historical["training"]["seed"] = 1
    with pytest.raises(ValueError, match="scratch contract"):
        _validate_loaded_contextual_config(historical)


def test_mediated_architecture_cannot_be_silently_relabelled_as_s():
    config, _ = load_contextual_frontier_config(
        CONFIGS / "thermal_r13_12p_e_anchor_mediated_train150_discovery_v1.json",
    )
    model = copy.deepcopy(config["member_routing"]["model"])
    model["architecture"] = "R13-8-A-EAnchor"
    _, valid = _anchored_schedule_and_model(config["training"], config["initialization"],
                                          model, model["architecture"])
    assert not valid
