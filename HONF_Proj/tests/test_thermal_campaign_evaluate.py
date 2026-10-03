"""Physical metrics and input-only development panel selection."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools/thermal_campaign_evaluate.py"
_SPEC = importlib.util.spec_from_file_location("thermal_campaign_evaluate", _PATH)
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


def test_per_channel_physical_metrics_ports_peaks_and_pressure_keep_units_separate():
    x, y = np.meshgrid(np.arange(4), np.arange(3))
    reference = np.zeros((3, 4, 5))
    predicted = reference.copy()
    predicted[..., 0] = 2
    predicted[..., 2] = 2 * x
    sample = {"x_grid": x, "y_grid": y, "steady_field": reference,
        "structure": {"module_centers": np.asarray([[1.5, 1.]]), "module_present": np.asarray([1.]),
                      "material_params": np.asarray([0., 0., 0., 0., 0., .6])},
        "interface_target": np.broadcast_to(np.asarray([2., 5.]), (1, 4, 2)).copy(),
        "module_internal_temperature_points": np.asarray([[10., 20., 30.]]),
        "teacher_port_tokens": np.broadcast_to(np.asarray([0., 1., 0., 20., 7.]), (1, 4, 5)).copy()}
    prediction = {"pred_field_grid": predicted,
        "pred_interface": sample["interface_target"] + np.asarray([1., 2.]),
        "pred_internal_temperature": (sample["module_internal_temperature_points"] + 1)[..., None],
        "pred_port_condition_raw": sample["teacher_port_tokens"] + np.asarray([0., 0., 0., 1., 4.]),
        "pred_port_condition": sample["teacher_port_tokens"] + np.asarray([0., 0., 0., 2., 3.])}
    metrics, masks = evaluation.physical_case_metrics(sample, prediction, ["u", "v", "p", "omega", "temperature"])
    assert metrics["fluid/u"]["rmse"] == 2
    assert metrics["surface_temperature"]["rmse"] == 1
    assert metrics["q_normal_proxy"]["rmse"] == 2
    assert metrics["module_material_peak"]["rmse"] == 1
    assert metrics["initial_port/h_effective"]["rmse"] == 4
    assert metrics["final_port/outside_temperature"]["rmse"] == 2
    assert metrics["inlet_outlet_pressure_difference"]["rmse"] == 6
    assert not (masks["near_mask"] & masks["far_mask"]).any()
    np.testing.assert_array_equal(masks["near_mask"] | masks["far_mask"], masks["fluid_mask"])


def test_equal_case_and_pooled_metrics_differ_without_dropping_failure():
    first = evaluation.physical_errors(np.asarray([1.]), np.zeros(1))
    second = evaluation.physical_errors(np.full(9, 3.), np.zeros(9))
    failed = evaluation.physical_errors(np.asarray([np.nan]), np.zeros(1))
    aggregate = evaluation.aggregate_physical([{"metrics": {"temperature": metric}} for metric in (first, second, failed)])
    result = aggregate["temperature"]
    assert result["equal_case_rmse_mean"] == 2
    assert result["pooled_rmse"] == pytest.approx(np.sqrt(8.2))
    assert result["nonfinite_cases"] == 1
    assert result["pooled_count"] == 10


def test_screen_panel_uses_input_strata_and_excludes_known_duplicate():
    case_ids = ["0273", "0001", "0002", "0003", "0004", "0005", "0006"]
    groups = {case_id: {"module_present": np.ones(1 + index % 3),
                        "heat_powers": np.full(1 + index % 3, index),
                        "material_parameters": SimpleNamespace(attrs={"u_in": index / 10})}
              for index, case_id in enumerate(case_ids)}
    dataset = SimpleNamespace(selected_case_ids=case_ids, h5={"cases": groups})
    panel = evaluation.screen_indices(dataset, count=4)
    assert panel == evaluation.screen_indices(dataset, count=4)
    assert 0 not in panel and len(set(panel)) == 4
    assert {len(groups[case_ids[index]]["module_present"]) for index in panel} == {1, 2, 3}


def test_empty_region_and_bad_shapes_are_explicit():
    metric = evaluation.physical_errors(np.zeros((2, 3)), np.ones((2, 3)), np.zeros((2, 3), dtype=bool))
    assert metric["count"] == 0 and metric["rmse"] is None
    with pytest.raises(ValueError, match="shapes differ"):
        evaluation.physical_errors(np.zeros(2), np.zeros(3))
