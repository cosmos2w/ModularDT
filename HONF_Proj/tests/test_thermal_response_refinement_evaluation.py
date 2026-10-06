"""Scientific denominator, wide arithmetic and finite-pool decision contracts."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from thermal_response_refinement_evaluation import (
    CountedPhaseGraphArrays,
    aggregate_response_rows,
    audit_field_comparability,
    compare_heat_null,
    compare_saved_responses,
    finite_metrics,
    load_atlas_families,
    load_counted_families,
    main,
    prepare_counted_replay,
    save_counted_phase_graph,
    stored_pool_decision,
    validate_heat_records,
    validate_phase_graph_capture,
)


def summary():
    return {"dataset": "fixed.h5", "split": "test", "channel_order": ["T"], "dataset_scope": "development",
            "port_mode": "predicted", "development_manifest_binding": {"manifest_sha256": "fixed"},
            "rows": [{"case_id": f"{index:04d}", "intervention": "normal", "module_count": 3,
                      "metrics": {"fluid/temperature": {"count": 8}}} for index in range(22)]}


def test_field_dashboard_rejects_membership_and_denominator_mismatches():
    original = summary()
    assert audit_field_comparability(original, copy.deepcopy(original))["normal_cases"] == 22
    changed = copy.deepcopy(original)
    changed["rows"][0]["metrics"]["fluid/temperature"]["count"] = 7
    with pytest.raises(ValueError, match="support denominators"):
        audit_field_comparability(original, changed)
    changed = copy.deepcopy(original)
    changed["development_manifest_binding"]["manifest_sha256"] = "other"
    with pytest.raises(ValueError, match="membership"):
        audit_field_comparability(original, changed)


def test_pool_true_maximum_decision_and_unresolved_tie():
    truth = {"baseline": 10.0, "minus": 9.0, "plus": 11.0}
    prediction = {"baseline": 11.0, "minus": 10.0, "plus": 9.0}
    value = stored_pool_decision(truth, prediction, warning_scale=.001)
    assert value["predicted_best_state"] == "plus"
    assert value["measured_best_state"] == "minus"
    assert value["realized_benchmark_regret"] == 2.0
    assert value["choice_correct"] is False
    tied = stored_pool_decision({"a": 9, "b": 9.0001}, {"a": 10, "b": 9}, warning_scale=.001)
    assert tied["measured_best_state"] is None
    assert tied["choice_correct"] is None
    assert tied["unresolved_reference_best_states"] == ["a", "b"]
    with pytest.raises(ValueError, match="identical"):
        stored_pool_decision(truth, {"baseline": 10, "minus": 9})


def test_layouts_and_opposite_directions_are_counted_separately():
    rows = [{"case_id": case, "state": sign, "role": "fluid_fields", "channel": "T",
             "rmse": error, "reference_rms": 2.0, "unit": "native"}
            for case in ("a", "b", "c") for sign, error in (("minus", 1.0), ("plus", 3.0))]
    value = aggregate_response_rows(rows)[0]
    assert value["layout_count"] == 3
    assert value["finite_direction_count"] == 6
    assert value["macro_mean_state_rmse"] == 2.0
    assert value["equal_state_pooled_rmse"] == pytest.approx(np.sqrt(5))
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_response_rows(rows + rows[:1])


def test_finite_metrics_widen_before_subtraction_and_apply_quadrature_mask():
    role = SimpleNamespace(valid_mask=np.array([[True], [True], [False]]),
        quadrature_weights=np.array([1., 3., 1.]), channel_names=("T",), channel_units=("native",))
    base = np.array([[1e8], [1e8], [np.nan]], dtype=np.float32)
    target = np.array([[1e8 + 8], [1e8 + 16], [np.nan]], dtype=np.float32)
    prediction = np.array([[1e8 + 16], [1e8 + 16], [np.nan]], dtype=np.float32)
    result, delta_prediction, delta_reference = finite_metrics(prediction, target, role,
        baseline_prediction=base, baseline_reference=base)
    assert delta_prediction.dtype == np.float64
    assert delta_reference.dtype == np.float64
    assert result["T"]["rmse"] == 4
    assert result["T"]["reference_signed_mean"] == 14
    assert result["T"]["prediction_signed_mean"] == 16


def test_heat_null_uses_identical_steps_and_does_not_promote_floor_limited_reduction():
    parent = {"development_manifest_sha256": "fixed", "query_count": 256, "amplitudes": [.1, .2],
              "train_heat_range": [0, 1], "cases": [
                  {"case_id": f"{index:04d}", "module_count": 3, "eligible": True, "variants": [
                      {"label": "minus", "donors": [0, 2], "fraction_of_feasible_bound": .1, "signed_heat_transfer": -.02},
                      {"label": "plus", "donors": [0, 2], "fraction_of_feasible_bound": .1, "signed_heat_transfer": .02}]}
                  for index in range(22)], "equal_case_channels": {
                      channel: {"mean_rms_change_native": 1e-6} for channel in ("u", "v", "p", "omega")}}
    candidate = copy.deepcopy(parent)
    for channel in candidate["equal_case_channels"]:
        candidate["equal_case_channels"][channel]["mean_rms_change_native"] = 1e-8
    result = compare_heat_null(parent, candidate, numerical_floors={channel: 1e-5 for channel in ("u", "v", "p", "omega")})
    assert result["relative_error_to_zero"] == "undefined"
    assert result["channels"]["p"]["reduction_percent"] == 99
    assert result["channels"]["p"]["at_least_90_percent_reduction"] is None
    candidate["amplitudes"] = [.2]
    with pytest.raises(ValueError, match="amplitudes"):
        compare_heat_null(parent, candidate)
    candidate["amplitudes"] = [.1, .2]
    candidate["cases"][0]["variants"][0]["signed_heat_transfer"] = -.01
    with pytest.raises(ValueError, match="actual signed_heat_transfer"):
        compare_heat_null(parent, candidate)
    candidate["cases"][0]["variants"][0]["signed_heat_transfer"] = -.02
    candidate["cases"][0]["query_xy_sha256"] = "changed"
    with pytest.raises(ValueError, match="reported query_xy_sha256"):
        compare_heat_null(parent, candidate)


def test_heat_record_receiver_mismatch_is_rejected():
    from channelthermal.interaction_evidence.types import DesignState, ModuleState, OperatingContext, RoleOutput

    role = RoleOutput("fluid_fields", np.array([[0., 0.]]), np.array([[1.]]), ("T",), ("native",),
                      np.array([[True]]), np.array([1.]), ("q0",), None, "eulerian")
    design = DesignState("a", "f", "development", (ModuleState("m0", (0., 0.), 1.),))
    record = SimpleNamespace(design=design, context=OperatingContext({"re": 1.}), output=SimpleNamespace(roles={"fluid_fields": role}))
    changed = SimpleNamespace(design=design, context=record.context, output=SimpleNamespace(roles={"fluid_fields": role}))
    changed.output.roles["fluid_fields"] = RoleOutput("fluid_fields", np.array([[1., 0.]]), np.array([[1.]]),
        ("T",), ("native",), np.array([[True]]), np.array([1.]), ("q0",), None, "eulerian")
    with pytest.raises(ValueError, match="query_features"):
        validate_heat_records({"baseline": record, "plus": changed})


def test_real_saved_counted_panel_preserves_missing_baseline_without_solver():
    request = Path("/data/wanglz/ModularDT/thermal_development/tree_faithfulness_20261004/follow_on_reference_request_fixed4.json")
    records = Path("/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/reference/records")
    if not request.exists() or not records.exists():
        pytest.skip("Existing local counted reference panel is unavailable")
    families, _ = load_counted_families(request, records)
    result = prepare_counted_replay(families)
    assert [len(records) for _, records in families] == [2, 3, 3, 3]
    assert result["new_solver_attempts"] == result["new_model_calls"] == 0
    assert result["physical_reference_attempts_cumulative"] == 326
    assert result["families"][0]["candidate_states"] == ["transfer_minus", "transfer_plus"]


def test_atlas_cli_rejects_unasked_direction_and_partial_cohort_before_model_read(tmp_path):
    arguments = ["atlas", "--checkpoint", str(tmp_path / "no_model.pt"), "--dataset", str(tmp_path / "no_data.h5"),
                 "--output-dir", str(tmp_path / "outputs"), "--cohort", "development"]
    for index in range(4):
        arguments += ["--stencil", str(tmp_path / f"family{index}.npz")]
    with pytest.raises(ValueError, match="exactly heat_transfer_minus"):
        main(arguments + ["--variant", "heat_transfer_minus", "--variant", "another_heat_direction"])
    with pytest.raises(ValueError, match="exactly four"):
        load_atlas_families([tmp_path / "one.npz"] * 3, ["heat_transfer_minus", "heat_transfer_plus"], "development")
    assert not (tmp_path / "outputs").exists()


def test_real_atlas_cohorts_are_complete_and_reject_duplicate_physical_family():
    atlas = Path(__file__).resolve().parents[1] / "diagnostics/generated/interactions/physical_response_atlas_20260926/families"
    ids = ("0304", "0320", "0335", "0350")
    paths = [atlas / f"train_{anchor}_responses.npz" for anchor in ids]
    if not all(path.exists() for path in paths):
        pytest.skip("Existing local response-development atlas is unavailable")
    pair = ["heat_transfer_minus", "heat_transfer_plus"]
    assert [anchor for anchor, _ in load_atlas_families(paths, pair, "development")] == list(ids)
    with pytest.raises(ValueError, match="distinct families"):
        load_atlas_families(paths[:-1] + paths[:1], pair, "development")
    with pytest.raises(ValueError, match="physical-family identity"):
        load_atlas_families(paths, pair, "fit")


def test_saved_response_comparison_scales_null_and_rejects_changed_heat_and_queries(tmp_path):
    arrays = tmp_path / "reference.npz"
    identity = {"fluid_fields/query_features": np.array([[0., 1.]]),
                "fluid_fields/valid_mask": np.array([[True]]),
                "fluid_fields/quadrature_weights": np.array([1.]),
                "fluid_fields/channel_names": np.array(["p"]),
                "fluid_fields/channel_units": np.array(["pressure"]),
                "fluid_fields/query_ids": np.array(["q0"]),
                "fluid_fields/receiver_module_ids": np.array([], dtype=str),
                "baseline/fluid_fields/reference": np.array([[1.]]),
                "plus/fluid_fields/reference": np.array([[1.]]),
                "plus/fluid_fields/delta_reference_FP64": np.array([[0.]])}
    np.savez(arrays, **identity)
    metric = {"count": 1, "unit": "pressure", "reference_rms": 0., "reference_signed_mean": 0.,
              "prediction_signed_mean": -.2, "rmse": .2}
    pressure = {"reference": 0., "prediction": -.2, "absolute_error": .2, "unit": "pressure"}
    module = {"reference": -.1, "prediction": -.3, "absolute_error": .2, "unit": "temperature", "unchanged_own_heat": True}
    family = {"case_id": "a", "physical_family_id": "family:a", "arrays": str(arrays),
              "absolute": [{"state": state, "heat": heat, "centres": [[0., 0.], [1., 0.]],
                            "reference_functionals": {"maximum_material_temperature": 10.}}
                           for state, heat in (("baseline", [1., 2.]), ("plus", [1.1, 1.9]))],
              "finite": [{"baseline_state": "baseline", "state": "plus", "roles": {"fluid_fields": {"p": metric}},
                          "pressure_drop_8pct_response": pressure, "module_peak_changes": {"m2": module}}]}
    parent = {"manifest_fingerprint": "fixed", "checkpoint": "parent.pt", "families": [family]}
    candidate = copy.deepcopy(parent)
    candidate["checkpoint"] = "candidate.pt"
    candidate["families"][0]["finite"][0]["roles"]["fluid_fields"]["p"]["rmse"] = .1
    scales = {channel: .5 for channel in ("u", "v", "p", "omega")}
    compared = compare_saved_responses(parent, candidate, common_flow_stds=scales)
    aggregate = compared["primary_response_aggregates"][0]
    assert aggregate["macro_rmse_common_train_scaled"]["parent"] == .4
    assert aggregate["macro_mean_state_rmse_native"]["improvement_percent"] == 50
    assert aggregate["raw_90_percent_reduction"] is False
    assert compared["primary_response_rows"][0]["relative_accuracy_to_zero"] == "undefined"
    assert compared["pressure_drop_8pct_rows"][0]["absolute_error_common_train_scaled"]["parent"] == .4
    assert compared["unchanged_own_heat_module_rows"][0]["module_id"] == "m2"
    candidate["families"][0]["absolute"][1]["heat"] = [1.2, 1.8]
    with pytest.raises(ValueError, match="actual heat"):
        compare_saved_responses(parent, candidate, common_flow_stds=scales)
    candidate = copy.deepcopy(parent)
    changed_arrays = tmp_path / "changed.npz"
    np.savez(changed_arrays, **{**identity, "fluid_fields/query_features": np.array([[1., 1.]])})
    candidate["families"][0]["arrays"] = str(changed_arrays)
    with pytest.raises(ValueError, match="reference/query identity"):
        compare_saved_responses(parent, candidate, common_flow_stds=scales)
    with pytest.raises(ValueError, match="positive and finite"):
        compare_saved_responses(parent, parent, common_flow_stds={**scales, "p": 0})
    np.savez(arrays, **{key: value for key, value in identity.items() if key != "fluid_fields/valid_mask"})
    with pytest.raises(ValueError, match="required reference/query identity keys"):
        compare_saved_responses(parent, parent, common_flow_stds=scales)
    np.savez(arrays, **identity)
    candidate = copy.deepcopy(parent)
    candidate["families"][0]["finite"][0]["pressure_drop_8pct_response"]["unit"] = "other_pressure"
    with pytest.raises(ValueError, match="pressure functional reference"):
        compare_saved_responses(parent, candidate, common_flow_stds=scales)
    candidate = copy.deepcopy(parent)
    candidate["families"][0]["finite"][0]["module_peak_changes"]["m2"]["unit"] = "other_temperature"
    with pytest.raises(ValueError, match="unchanged-own-heat module reference"):
        compare_saved_responses(parent, candidate, common_flow_stds=scales)


def test_counted_phase_capture_rejects_other_states_and_interventions_before_model_read(tmp_path):
    arguments = ["counted", "--request", str(tmp_path / "no_request.json"),
                 "--records-dir", str(tmp_path), "--checkpoint", str(tmp_path / "no_model.pt"),
                 "--output-dir", str(tmp_path / "outputs"), "--capture-phase-graph-case", "0291"]
    with pytest.raises(SystemExit) as rejected:
        main(arguments + ["--intervention", "zero_joint"])
    assert rejected.value.code == 2
    assert not (tmp_path / "outputs").exists()
    baseline = SimpleNamespace(design=SimpleNamespace(physical_family_id="receiver_interaction_fixed4:0291"))
    families = [("0291", {"baseline": baseline, "transfer_plus": baseline})]
    validate_phase_graph_capture(families, "normal", "0291")
    for intervention, case in (("zero_joint", "0291"), ("normal", "0294")):
        with pytest.raises(ValueError, match="normal baseline0291"):
            validate_phase_graph_capture(families, intervention, case)
    with pytest.raises(ValueError, match="exactly one"):
        validate_phase_graph_capture([("0291", {"transfer_plus": baseline})], "normal", "0291")
    baseline.design.physical_family_id = "stored_family:0291"
    with pytest.raises(ValueError, match="counted fixed4"):
        validate_phase_graph_capture(families, "normal", "0291")


def test_counted_phase_graph_retains_first_actual_access_and_exact_provenance(tmp_path):
    from channelthermal.interaction_evidence.types import DesignState, ModuleState, OperatingContext, RoleOutput

    centers = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    xy = np.array([[.25, .5], [.75, .5], [1.25, .5]], dtype=np.float64)
    role = RoleOutput("fluid_fields", xy, np.ones((3, 1)), ("temperature",), ("dataset temperature units",),
                      np.ones((3, 1), dtype=bool), np.ones(3), ("q0", "q1", "q2"), None, "eulerian")
    design = DesignState("0291", "receiver_interaction_fixed4:0291", "development",
                         tuple(ModuleState(f"0291:module:{index}", tuple(center), float(index + 1)) for index, center in enumerate(centers)))
    record = SimpleNamespace(record_id="counted0291_baseline", design=design, context=OperatingContext({"re": 150.}),
                             output=SimpleNamespace(roles={"fluid_fields": role}))
    arrays = CountedPhaseGraphArrays()
    for phase in ("P0", "P1", "P2"):
        arrays[f"phase/{phase}/source_valid/M"] = np.array([[True, True, False]])
        arrays[f"phase/{phase}/source_coords/M"] = np.concatenate((centers, np.zeros((1, 2), dtype=np.float32)))[None]
        arrays[f"phase/{phase}/source_coords/E"] = np.array([[[.5, .5]]], dtype=np.float32)
        for mechanism in ("QM", "QE"):
            arrays[f"access/{phase}/{mechanism}/00000/receivers"] = xy[:2].astype(np.float32)[None]
            arrays[f"access/{phase}/{mechanism}/00001/receivers"] = xy[2:].astype(np.float32)[None]
        arrays[f"access/{phase}/MM/00000/receivers"] = centers[None]
    assert arrays.discarded_arrays == 9
    assert not any("/00001/" in key or "/MM/" in key for key in arrays)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"exact saved checkpoint bytes, no model")
    summary_data = {"checkpoint": str(checkpoint), "checkpoint_epoch": 500, "manifest_fingerprint": "fixed"}
    capture = SimpleNamespace(arrays=arrays)
    saved = save_counted_phase_graph(capture, record, tmp_path, summary_data, checkpoint)
    provenance = saved["phase_graph_provenance"]
    assert provenance["case_id"] == "0291" and provenance["state"] == "baseline"
    assert provenance["heat"] == [1., 2.] and provenance["context"] == {"re": 150.}
    assert provenance["new_wrapper_calls_for_capture"] == 0
    assert provenance["first_P2_QM_matches_counted_fluid_order_FP32"]
    assert provenance["query_identity"]["fluid_fields"]["query_features"]["shape"] == [3, 2]
    assert json.loads(Path(saved["phase_graph_provenance_file"]).read_text()) == provenance
    with np.load(saved["phase_graph_arrays"]) as stored:
        assert json.loads(str(stored["provenance_json"].item())) == provenance
        assert np.array_equal(stored["access/P2/QM/00000/receivers"][0], xy[:2].astype(np.float32))
    arrays["access/P2/QM/00000/receivers"] = xy[1:3].astype(np.float32)[None]
    with pytest.raises(ValueError, match="receiver coordinates/order"):
        save_counted_phase_graph(capture, record, tmp_path, summary_data, checkpoint)
