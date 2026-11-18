"""Check native policy execution and reference-only acceptance on a small panel."""

from copy import deepcopy

import pytest
import torch

from channelthermal.inverse.placement_benchmark import _evaluate_shortlist, benchmark_placement
from channelthermal.member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from tests.anchored_fixtures import _base_config
from tests.test_r14_runtime_integration import STATS


def test_reference_rejects_false_feasible_and_unresolved_moves():
    reference = {"baseline": {"J": 0., "P": 0.}, "edit": {"J": -2., "P": .3},
                 "unseen": {"J": -1., "P": 0.}}
    prediction = {"baseline": {"J": 0., "P": 0.}, "edit": {"J": -2., "P": 0.}}
    decision = _evaluate_shortlist(prediction, reference, pressure_allowance=.1, resolution=.1)
    assert decision["model_selected_action"] == "edit"
    assert decision["raw_constraint_violation"]
    assert decision["verified_incumbent"] == "baseline"
    assert decision["full_pool_regret"] == 1.

    reference["edit"] = {"J": -.05, "P": 0.}
    decision = _evaluate_shortlist(prediction, reference, pressure_allowance=.1, resolution=.1)
    assert not decision["raw_constraint_violation"]
    assert decision["verified_incumbent"] == "baseline"

    reference["edit"] = {"J": -.5, "P": 0.}
    decision = _evaluate_shortlist(prediction, reference, pressure_allowance=.1, resolution=.1)
    assert decision["verified_incumbent"] == "edit"
    assert decision["verified_improvement"] == .5
    assert decision["full_pool_regret"] == .5


def native_fixture():
    config = {
        "domain": {"nx": 12, "ny": 6, "lx": 12., "ly": 6., "module_radius": .7},
        "layout": {"num_modules": 3, "centers": [[2., 2.], [5., 3.], [9., 4.]], "heat_powers": [3., 1., 2.]},
        "flow": {"re": 20., "u_in": 1., "nu": None},
        "thermal": {"solid_alpha": .01, "fluid_alpha": .02, "solid_k": 1., "fluid_k": 1.,
                    "t_in": 0., "t_wall": 0.},
        "local_module": {"local_grid_size": 8, "n_interface_points": 64},
    }
    panel = {"schema": "thermal_response_families_v1",
             "cases": [{"case_id": "control", "role": "synthetic", "source_config": config}],
             "actions": []}
    for index in range(3):
        centers = deepcopy(config["layout"]["centers"])
        centers[index][0] += .25
        panel["actions"].append({"case_id": "control", "action": f"move_{index}", "centers": centers,
                                 "physical_delta": {"J": float(index - 1), "P": float(index) / 5}})
    model = ThermalDetailAnchoredMemberRoutingAdapter(STATS, core_config={
        "schema": "interaction_reused_detail_anchored_member_routing_v1",
        "architecture": "R14-Detail-LocalECover", "locator_mode": "local_e_cover",
        "base_model_config": _base_config(), "input_dataset": "thermal",
        "input_capability": "joint_prescribed_records_v1", "query_tile_size": 32,
        "retained_mass": .95, "score_calibration": "inherited",
    }, nx=12, ny=6, environment_nx=4, environment_ny=2).requires_grad_(False).eval()
    return panel, model


def test_matched_native_shortlists_do_not_observe_reference_labels():
    panel, model = native_fixture()
    blind = deepcopy(panel)
    for action in blind["actions"]:
        del action["physical_delta"]
    settings = {"budget": 1, "tau": .4, "pressure_allowance": .2, "resolution": .1}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = benchmark_placement(model, panel, **settings)
        blinded = benchmark_placement(model, blind, **settings)
        illegal = deepcopy(panel)
        illegal["actions"][0]["centers"][0] = illegal["cases"][0]["source_config"]["layout"]["centers"][1]
        with pytest.raises(ValueError, match="disk separation"):
            benchmark_placement(model, illegal, **settings)
    finally:
        torch.set_num_threads(previous_threads)

    for method, row in result["cases"]["control"]["methods"].items():
        unlabelled = blinded["cases"]["control"]["methods"][method]
        assert row["candidate_scores"] == 1
        assert row["shortlist"] == unlabelled["shortlist"]
        assert row["predicted_deltas"] == unlabelled["predicted_deltas"]
        assert "verified_improvement" in row["decision"]
        assert "verified_improvement" not in unlabelled["decision"]
    assert sum(result["cases"]["control"]["methods"]["environment_access"]["module_relevance"]) > 0
    assert result["by_role"]["synthetic"]["uniform"]["reference_family_count"] == 1
    assert blinded["by_role"]["synthetic"]["uniform"]["reference_family_count"] == 0
