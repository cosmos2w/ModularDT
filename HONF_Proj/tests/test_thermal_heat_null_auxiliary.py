"""Scientific negative checks for the opt-in heat-null amendment."""

import copy

import numpy as np
import pytest
import torch
from channelthermal.training.campaign import amend_structural_calibration, validate_campaign, validate_campaign_resume
from channelthermal.training.campaign_null_response import fixed_total_transfer, heat_null_loss


def test_heat_transfer_preserves_total_nonnegative_and_inactive_slots():
    heat = np.asarray([.4, 1.2, 2., 0.])
    transfer, receipt = fixed_total_transfer(heat, [1, 1, 1, 0], seed=4, max_heat=2.)
    np.testing.assert_allclose((heat + transfer).sum(), heat.sum(), atol=1e-14)
    assert (heat + transfer >= 0).all()
    assert transfer[3] == 0 and (heat + transfer <= 2.).all()
    assert receipt["donor"] != receipt["receiver"]
    assert fixed_total_transfer([1., 0.], [1, 0], seed=0) is None


def test_null_supervision_ignores_thermal_channel_and_has_flow_gradients():
    names = ("temperature", "p", "omega", "v", "u")
    before = torch.zeros(4, 5)
    after = torch.zeros(4, 5, requires_grad=True)
    with torch.no_grad():
        after[:, 0] = 1e6
    loss = heat_null_loss(before, after, names, np.ones(5))
    assert float(loss) == 0
    with torch.no_grad():
        after[:, 1] = 2
    heat_null_loss(before, after, names, np.ones(5)).backward()
    assert float(after.grad[:, 0].abs().sum()) == 0
    assert float(after.grad[:, 1].abs().sum()) > 0
    with pytest.raises(ValueError, match="positive"):
        heat_null_loss(before, after, names, np.zeros(5))


def _config():
    return {"model": {"core_honf": {"forward_architecture": "faithful_receiver_hypergraph_honf"}},
        "training": {"seed": 0, "epochs": 100, "campaign": {"schedule_total_epochs": 1000,
            "physical_loss_policy_version": 1}}, "dataset": {}, "loss": {}}


def test_reviewed_null_amendment_only_at_e100_and_cannot_change_later():
    initial = _config()
    resumed = copy.deepcopy(initial)
    resumed["training"]["epochs"] = 500
    resumed["training"]["campaign"].update(physical_loss_policy_version=2,
        native_loss_denominators_start_epoch=101,
        heat_null_response={"benchmark_verified": True, "cases_per_epoch": 2,
                            "fluid_queries": 256, "fraction": .1})
    validate_campaign(resumed)
    checkpoint = {"train_config": initial, "epoch": 100, "optimizer_state_dict": {"state": 1},
        "rng_state": {"torch": 1}}
    amendment = validate_campaign_resume(checkpoint, resumed)
    assert amendment["heat_null_response"]["benchmark_verified"] is True
    checkpoint["epoch"] = 200
    with pytest.raises(ValueError, match="silently amend"):
        validate_campaign_resume(checkpoint, resumed)
    broken = copy.deepcopy(resumed)
    broken["model"]["core_honf"]["forward_architecture"] = "dense_pairwise_field"
    with pytest.raises(ValueError, match="Tree-F/Pair-F"):
        validate_campaign(broken)
    broken = copy.deepcopy(resumed)
    broken["training"]["campaign"]["heat_null_response"]["benchmark_verified"] = False
    with pytest.raises(ValueError, match="provenance"):
        validate_campaign(broken)


def test_eligible_measure_amendment_is_explicit_and_preserves_old_calibration():
    initial = _config()
    current = copy.deepcopy(initial)
    current["model"]["core_honf"]["interface_model"] = {
        "hypergraph_options": {"structural_measure_policy_version": 2}}
    current["training"]["campaign"].update(physical_loss_policy_version=2,
        native_loss_denominators_start_epoch=101, structural_measure_policy_version=2)
    validate_campaign(current)
    undisclosed = copy.deepcopy(current)
    del undisclosed["training"]["campaign"]["structural_measure_policy_version"]
    with pytest.raises(ValueError, match="explicit reviewed campaign"):
        validate_campaign(undisclosed)
    checkpoint = {"train_config": initial, "epoch": 100,
        "optimizer_state_dict": {"state": 1}, "rng_state": {"torch": 1}}
    amendment = validate_campaign_resume(checkpoint, current)
    state = {"structural_scale": .01, "structural_calibration_samples": [{"epoch": 25}],
        "response_coefficient": .2, "heat_null_coefficient": .1}
    amend_structural_calibration(state, amendment)
    assert state["structural_measure_policy_1_calibration"]["structural_scale"] == .01
    assert "structural_scale" not in state
    assert state["response_coefficient"] == .2 and state["heat_null_coefficient"] == .1
    broken = copy.deepcopy(current)
    broken["model"]["core_honf"]["interface_model"]["hypergraph_options"]["structural_measure_policy_version"] = 1
    with pytest.raises(ValueError, match="explicit reviewed"):
        validate_campaign(broken)
