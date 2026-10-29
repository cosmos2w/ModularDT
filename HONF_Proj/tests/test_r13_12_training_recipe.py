"""R13-12 discovery, independent confirmation and formal-resume contracts."""
import copy
from pathlib import Path

import pytest

from honf_runtime.contextual_frontier_training import (
    _anchored_schedule_and_model, _validate_loaded_contextual_config, load_contextual_frontier_config,
)


CONFIGS = Path(__file__).resolve().parents[1] / "configs/research"


@pytest.mark.parametrize("mode", ("p", "g", "g_radial"))
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


@pytest.mark.parametrize("mode,law", (("g", "native_cell_radial"), ("g_radial", "ee_head_mean")))
def test_geometry_revision_cannot_silently_relabel_saved_recipe(mode, law):
    config, _ = load_contextual_frontier_config(
        CONFIGS / f"thermal_r13_12{mode}_e_anchor_mediated_train150_discovery_v1.json",
    )
    model = copy.deepcopy(config["member_routing"]["model"])
    model["mediated_geometry_law"] = law
    _, valid = _anchored_schedule_and_model(config["training"], config["initialization"],
                                          model, model["architecture"])
    assert not valid


def test_radial_recipe_checkpoint_loader_preserves_law_and_state(tmp_path):
    import torch
    from honf_runtime.contextual_frontier_evaluation import _contextual_checkpoint_parts, load_contextual_checkpoint
    from honf_runtime.contextual_frontier_training import _build_task, _run_identity

    config, _ = load_contextual_frontier_config(
        CONFIGS / "thermal_r13_12g_radial_e_anchor_mediated_train150_confirmation_v1.json",
    )
    if not Path(config["dataset"]["data_path"]).exists():
        pytest.skip("Thermal source dataset unavailable")
    torch.set_num_threads(1)
    arm = config["member_routing"]["arm"]
    model, provider = _build_task(config, arm, torch.device("cpu"))
    identity = _run_identity(config, provider)
    identity["provider_identity"] = provider.identity_payload()
    payload = {"experiment_identity": identity, "arm": arm,
               "model_state_dict": model.state_dict(), "epoch": 0}
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    loaded, _, _ = load_contextual_checkpoint(checkpoint, "cpu")
    assert loaded.get_config() == model.get_config()
    assert loaded.core.mediated_geometry_law == "native_cell_radial"
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    for section in ("recipe", "adapter_state"):
        bad = copy.deepcopy(payload)
        p = bad["experiment_identity"]["provider_identity"]
        c = p["recipe"]["member_routing"]["model"] if section == "recipe" else p["adapter_state"]["core"]
        c["mediated_geometry_law"] = "ee_head_mean"
        with pytest.raises(ValueError, match="identities must agree"):
            _contextual_checkpoint_parts(bad)
