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
    assert decision["shortlist_loss"] == 1.
    assert decision["decision_loss"] == 0.

    reference["edit"] = {"J": -.05, "P": 0.}
    decision = _evaluate_shortlist(prediction, reference, pressure_allowance=.1, resolution=.1)
    assert not decision["raw_constraint_violation"]
    assert decision["verified_incumbent"] == "baseline"
    assert decision["decision_loss"] == .05

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
    assert result["cases"]["control"]["methods"]["finite_response"]["proposal_status"] == "uniform fallback"


def test_signed_probes_count_against_budget_and_reference_stays_blind(monkeypatch):
    from channelthermal.inverse import placement_benchmark as placement
    from tools.thermal_placement_benchmark import format_summary

    panel, model = native_fixture()
    config = panel["cases"][0]["source_config"]
    panel["actions"] = []
    responses = {
        "right": ([(0, 0, .25)], -1., .04),
        "left": ([(0, 0, -.25)], 2., .02),
        "down": ([(1, 1, -.25)], -.5, .06),
        "joint": ([(0, 0, .25), (1, 1, -.25)], -2., .12),
    }
    predictions = {}
    for action, (moves, j, p) in responses.items():
        centers = deepcopy(config["layout"]["centers"])
        for slot, axis, shift in moves:
            centers[slot][axis] += shift
        predictions[tuple(tuple(center) for center in centers)] = {"J": j, "P": p}
        panel["actions"].append({"case_id": "control", "action": action, "centers": centers,
                                 "physical_delta": {"J": j, "P": p}})

    def score_known_responses(model, design, *, tau, mode):
        scored = score_native(model, design, tau=tau, mode=mode)
        key = tuple(tuple(center) for center in design.structure["module_centers"][0].tolist())
        scored["functionals"] = predictions.get(key, {"J": 0., "P": 0.})
        return scored

    score_native = placement.score_native_design
    monkeypatch.setattr(placement, "score_native_design", score_known_responses)
    blind = deepcopy(panel)
    for action in blind["actions"]:
        del action["physical_delta"]
    options = {"budget": 3, "tau": .4, "pressure_allowance": .2, "resolution": .1, "full_pool": True}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = benchmark_placement(model, panel, **options)
        blind_result = benchmark_placement(model, blind, **options)
    finally:
        torch.set_num_threads(threads)

    family = result["cases"]["control"]
    blind_family = blind_result["cases"]["control"]
    for arm in family["methods"].values():
        assert arm["candidate_scores"] == arm["work"]["member_routing_read_mode_restricted"] == 3
        assert len(set(arm["shortlist"])) == 3
    signed = family["methods"]["finite_response"]
    assert set(signed["probe_actions"]) == {"right", "down"}
    assert signed["additive_estimates"]["joint"]["J"] == -1.5
    assert signed["predicted_deltas"]["joint"]["J"] == -2.
    assert signed["response_prioritized_actions"] == ["joint"]
    assert signed["shortlist"] == blind_family["methods"]["finite_response"]["shortlist"]
    assert signed["decision"]["verified_incumbent"] == "joint"
    assert signed["decision"]["reference_checks"] == 1
    poll = family["reference_poll"]
    assert poll["shortlist"] == blind_family["reference_poll"]["shortlist"]
    assert poll["candidate_scores"] == 0
    assert poll["decision"]["reference_checks"] == 1
    assert blind_family["reference_poll"]["decision"]["reference_checks"] == 0
    assert family["full_pool"]["candidate_scores"] == len(responses)
    assert family["full_pool"]["response_accuracy"]["correct_j_sign"] == len(responses)
    assert family["full_pool"]["response_accuracy"]["correct_feasible_order"] == 6
    text = format_summary({"Controlled fixture": {"epoch": None, **result}}, "cpu")
    assert "Signed finite response | 3 | 2 | 1/1 | 0/1 | 2 | 0 | 0" in text
    assert "Correct direction on 4/4" in text
