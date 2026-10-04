"""The new organizer surrogate is an explicit child, never a silent resume."""

import copy
import importlib.util
from pathlib import Path

import pytest
from channelthermal.training.campaign import validate_campaign, validate_campaign_resume


def tree_child():
    saved = {"model": {"core_honf": {"forward_architecture": "faithful_receiver_hypergraph_honf"}},
        "dataset": {"development_manifest_sha256": "fixed"}, "loss": {"field_mse_weight": 1.},
        "training": {"seed": 0, "epochs": 200, "campaign": {"parent": None,
            "schedule_total_epochs": 1000, "physical_loss_policy_version": 2,
            "native_loss_denominators_start_epoch": 101}}}
    checkpoint = {"epoch": 200, "train_config": saved, "model_config": copy.deepcopy(saved["model"]),
                  "optimizer_state_dict": {"state": 1}, "rng_state": {"torch": 1}}
    child = copy.deepcopy(saved)
    child["training"]["epochs"] = 500
    child["training"]["campaign"].update(organizer_gradient_policy="local_context_shadow_v1",
        parent={"checkpoint": "/data/parent/epoch_0200_model.pt", "epoch": 200,
                "from": "whole_wrapper_shadow_v1", "to": "local_context_shadow_v1"})
    return checkpoint, child


def test_exact_e200_child_allows_only_declared_organizer_derivative_change():
    checkpoint, child = tree_child()
    validate_campaign(child)
    amendment = validate_campaign_resume(checkpoint, child)
    assert set(amendment) == {"organizer_gradient_policy"}
    assert amendment["organizer_gradient_policy"]["activation_epoch"] == 201
    assert amendment["organizer_gradient_policy"]["parent"] == child["training"]["campaign"]["parent"]
    later = copy.deepcopy(checkpoint)
    later.update(epoch=300, train_config=copy.deepcopy(child))
    child["training"]["epochs"] = 1000
    assert validate_campaign_resume(later, child) is None


@pytest.mark.parametrize("change", ["epoch", "architecture", "parent", "schedule", "physical", "loss", "membership", "optimizer"])
def test_child_cannot_amend_other_scientific_contracts(change):
    checkpoint, child = tree_child()
    if change == "epoch":
        checkpoint["epoch"] = 100
    elif change == "architecture":
        checkpoint["model_config"]["core_honf"]["forward_architecture"] = "direct_pairwise_control_honf"
    elif change == "parent":
        child["training"]["campaign"]["parent"]["epoch"] = 100
    elif change == "schedule":
        child["training"]["campaign"]["schedule_total_epochs"] = 500
    elif change == "physical":
        child["training"]["campaign"]["physical_loss_policy_version"] = 1
    elif change == "loss":
        child["loss"]["field_mse_weight"] = 2.
    elif change == "membership":
        child["dataset"]["development_manifest_sha256"] = "other"
    elif change == "optimizer":
        checkpoint["optimizer_state_dict"] = {}
    with pytest.raises(ValueError):
        validate_campaign_resume(checkpoint, child)


def test_unknown_gradient_policy_is_rejected():
    _, child = tree_child()
    child["training"]["campaign"]["organizer_gradient_policy"] = "unreviewed"
    with pytest.raises(ValueError, match="Unknown organizer gradient"):
        validate_campaign(child)


def test_child_preparation_preserves_source_profile_and_rejects_same_run_id(tmp_path):
    import sys

    tools = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    try:
        specification = importlib.util.spec_from_file_location("thermal_tree_lite", tools / "thermal_tree_lite.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        checkpoint, _ = tree_child()
        profile = copy.deepcopy(checkpoint["train_config"])
        profile.update(workflow="forward", model_family="honf_forward", run={"id": "3204"})
        before = copy.deepcopy(profile)
        child = module.child_profile(profile, parent_checkpoint=tmp_path / "epoch_0200_model.pt",
                                     run_id="3302", output_root=tmp_path / "runs")
        assert profile == before
        assert child["model"] == before["model"] and child["loss"] == before["loss"]
        assert child["dataset"] == before["dataset"]
        assert child["training"]["epochs"] == 500
        assert child["training"]["campaign"]["schedule_total_epochs"] == 1000
        with pytest.raises(ValueError, match="different numeric run"):
            module.child_profile(profile, parent_checkpoint=tmp_path / "epoch_0200_model.pt",
                                 run_id="3204", output_root=tmp_path / "runs")
    finally:
        sys.path.remove(str(tools))
