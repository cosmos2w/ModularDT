"""Scientific denominator, wide arithmetic and finite-pool decision contracts."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from thermal_response_refinement_evaluation import (
    aggregate_response_rows,
    audit_field_comparability,
    finite_metrics,
    load_counted_families,
    prepare_counted_replay,
    stored_pool_decision,
    validate_heat_records,
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
