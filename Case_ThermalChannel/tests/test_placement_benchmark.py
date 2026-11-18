"""Check native policy execution and reference-only acceptance on a small panel."""

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path

import numpy as np
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
    from tools.thermal_placement_benchmark import format_curve_summary, format_summary

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
        curve_options = {name: value for name, value in options.items() if name != "budget"}
        curve = placement.benchmark_placement_curve(model, panel, budgets=(1, 2, 3), **curve_options)
        rejected_panel = deepcopy(panel)
        rejected_panel["actions"][-1]["physical_delta"]["P"] = .3
        rejected_curve = placement.benchmark_placement_curve(model, rejected_panel, budgets=(1, 2, 3),
                                                             **{**curve_options, "full_pool": False})
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
    assert [point["mean_verified_improvement"] for point in curve["curves"]["synthetic"]["uniform"]] == [0., .5, 2.]
    assert [point["mean_verified_improvement"] for point in rejected_curve["curves"]["synthetic"]["uniform"]] == [0., .5, 0.]
    assert all("full_pool" not in run["cases"]["control"] for run in curve["runs"][:-1])
    assert "full_pool" in curve["runs"][-1]["cases"]["control"]
    rejected = next(row for row in rejected_curve["critical_steps"] if row["budget"] == 3 and row["policy"] == "uniform")
    assert rejected["recommended"] == "joint" and rejected["retained"] == "baseline"
    assert rejected["reference_delta"]["P"] == .3
    assert "reference pressure exceeded cap" in rejected["events"]
    assert "verified cooling decreased" in rejected["events"]
    signed_step = next(row for row in curve["critical_steps"] if row["budget"] == 3 and row["policy"] == "finite_response")
    assert "signed probes used" in signed_step["events"] and len(signed_step["probe_actions"]) == 2
    assert signed_step["moves"] == [{"slot": 0, "from": [2., 2.], "to": [2.25, 2.]},
                                   {"slot": 1, "from": [5., 3.], "to": [5., 2.75]}]
    report = format_curve_summary({"Controlled fixture": curve}, "cpu")
    assert "| Uniform sample (blue) | 0 | 0.5 | 2 |" in report
    assert "Critical decisions by family" in report


def test_fixed_regional_caps_and_field_curve(monkeypatch):
    from channelthermal.inverse import placement_benchmark as placement
    from channelthermal.inverse import placement_visualization as visualization

    panel, model = native_fixture()
    panel["regional_caps"] = {
        "intake_peak": {"bounds": [.85, 0., 1., .4], "reduction": "max", "allowance": .1},
        "intake_mean": {"bounds": [.85, .6, 1., 1.], "reduction": "mean", "allowance": .1},
        "outlet_spread": {"bounds": [.85, 0., 1., 1.], "reduction": "std", "allowance": .1},
    }
    for row in panel["actions"]:
        row["physical_delta"]["intake_peak"] = .4 if row["action"] == "move_0" else 0.
        row["physical_delta"]["intake_mean"] = .2 if row["action"] == "move_0" else 0.
        row["physical_delta"]["outlet_spread"] = .3 if row["action"] == "move_0" else 0.

    native_score = placement.score_native_design

    def score(model, design, *, tau, mode):
        result = native_score(model, design, tau=tau, mode=mode)
        x, y = design.fluid_xy.unbind(-1)
        result["fields"]["temperature/fluid"]["prediction"] = x + y
        moved = design.structure["module_centers"][0, 0, 0] > 2.
        result["functionals"] = {"J": float(-moved.to(torch.int32)), "P": 0.}
        return result

    monkeypatch.setattr(placement, "score_native_design", score)
    monkeypatch.setattr(visualization, "score_native_design", score)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        curve = placement.benchmark_placement_curve(model, panel, budgets=(2, 3), tau=.4,
                                                    pressure_allowance=.2, resolution=.1)
        run = curve["runs"][-1]
        decision = run["cases"]["control"]["methods"]["uniform"]["decision"]
        assert decision["model_selected_action"] == "move_0"
        assert decision["verified_incumbent"] == "baseline" and not decision["reference_cap_violation"]
        assert decision["reference_violations"] == ["intake_peak", "intake_mean", "outlet_spread"]
        assert run["cases"]["control"]["baseline_prediction"]["intake_peak"] == 13.
        assert run["cases"]["control"]["baseline_prediction"]["intake_mean"] == 16.
        assert run["cases"]["control"]["baseline_prediction"]["outlet_spread"] == pytest.approx(np.sqrt(19 / 6))
        evidence = visualization.collect_placement_fields(model, panel, curve, case_id="control",
                                                           policy="uniform", budget=3)
        proposal = evidence["fields"]["move_0"]
        proposal["reference"] = (proposal["prediction"] + .25, proposal["mask"])
        figure = visualization.plot_placement_fields_and_curve(curve, evidence, budget=3, policy="uniform",
                                                                checkpoint="Synthetic fixture")
        assert any("RMSE 0.25" in axis.get_title() for axis in figure.axes)
        assert any("Error unavailable" in axis.get_title() for axis in figure.axes)
        buffer = BytesIO()
        figure.savefig(buffer, format="pdf")
        assert buffer.getvalue().startswith(b"%PDF")

        invalid = deepcopy(panel)
        invalid["regional_caps"]["intake_peak"]["bounds"] = [.7, .5, .85, .9]
        with pytest.raises(ValueError, match="intersects fixed fluid region"):
            benchmark_placement(model, invalid, budget=2, tau=.4, pressure_allowance=.2, resolution=.1)
    finally:
        torch.set_num_threads(threads)


def test_inspect_checkpoint_reports_saved_r16_identity_without_scoring(monkeypatch, capsys):
    from tools import thermal_placement_benchmark as cli

    panel = {"cases": [{"case_id": "panel-case", "source_case_id": "0002", "family_id": "edit-family", "role": "fit",
                       "source_config": {"domain": {"nx": 128, "ny": 64}}},
                      {"case_id": "synthetic-case", "source_config": {"domain": {"nx": 54, "ny": 24}}}]}
    adapter = {"schema": "test_adapter", "native_grid_nx_ny": [128, 64], "environment_grid_nx_ny": [24, 8]}
    provider = {"adapter_state": adapter, "training_case_ids": ["0002"], "validation_case_ids": ["0684"],
                "r16_10_family": {"arm": "R", "eligible_families": {"new": ["edit-family"]}}}
    saved = {"epoch": 1000, "experiment_identity": {"provider_identity": provider,
             "recipe": {"name": "continuation", "initialization": {"source_parent_age": 3000}}}}
    monkeypatch.setattr(Path, "resolve", lambda path, **kwargs: path)
    monkeypatch.setattr(Path, "read_text", lambda path: json.dumps(panel))
    monkeypatch.setattr(cli, "load_trusted_checkpoint", lambda path, **kwargs: saved)
    assert cli.main(["--inspect-checkpoints", "--checkpoint", "declared-label=/epoch-3000.pt", "--panel", "/panel.json"]) == 0
    result = json.loads(capsys.readouterr().out)["checkpoints"]["declared-label"]
    assert result["epoch"] == 1000 and result["source_parent_epoch"] == 3000
    assert result["r16_arm"] == "R" and result["native_grid_nx_ny"] == [128, 64]
    assert result["panels"][0] == {"case_id": "panel-case", "source_case_id": "0002", "family_id": "edit-family",
                                   "role": "fit", "native_grid_nx_ny": [128, 64], "grid_matches": True,
                                   "recorded_training_case": True, "recorded_validation_case": False,
                                   "recorded_response_groups": ["new"]}
    assert result["panels"][1]["grid_matches"] is False
