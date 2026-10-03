"""Physical metrics and input-only development panel selection."""

import copy
import importlib.util
import json
import os
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


def test_small_context_panel_covers_endpoints_then_interior():
    ids = [f"{index:04d}" for index in range(9)]
    groups = {case_id: {"module_present": np.ones(3), "heat_powers": np.asarray([index]),
        "material_parameters": SimpleNamespace(attrs={"u_in": index})} for index, case_id in enumerate(ids)}
    dataset = SimpleNamespace(selected_case_ids=ids, h5={"cases": groups})
    assert evaluation.screen_indices(dataset, count=3) == [0, 8, 4]


def test_fixed_campaign_screen_preserves_case_identity_across_index_order(tmp_path):
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"split": "test", "case_ids": ["0003", "0001"]}))
    dataset = SimpleNamespace(split="test", selected_case_ids=["0001", "0002", "0003"])
    assert evaluation.fixed_screen_indices(dataset, panel, 2) == [2, 0]
    dataset.selected_case_ids.reverse()
    assert evaluation.fixed_screen_indices(dataset, panel, 2) == [0, 2]
    with pytest.raises(ValueError, match="predeclared size"):
        evaluation.fixed_screen_indices(dataset, panel, 3)


def test_reference_only_mature_stage_keeps_four_cases_and_normal_screen_keeps_all90(tmp_path):
    class Dataset:
        split = "test"
        selected_case_ids = tuple(f"{index:04d}" for index in range(90))

        def __len__(self):
            return len(self.selected_case_ids)

    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"split": "test", "case_ids": ["0003", "0001", "0007", "0009"]}))
    args = SimpleNamespace(stage=500, panel_config=panel, panel_size=4,
        interventions=["geometry_reference_actions", "full_access_fixed_controls"])
    assert evaluation.evaluation_indices(Dataset(), args) == [3, 1, 7, 9]
    args.interventions = ["normal"]
    assert evaluation.evaluation_indices(Dataset(), args) == list(range(90))
    args.interventions = ["normal", "full_access"]
    assert evaluation.evaluation_indices(Dataset(), args) == list(range(90))
    args.stage = 100
    assert evaluation.evaluation_indices(Dataset(), args) == [3, 1, 7, 9]


def test_native_reference_effectiveness_is_not_overwritten_by_missing_anchor_baseline():
    # A reference-only directory has no normal P2 anchor archive. Actual native
    # supports still change, as observed with e100 full access at P0/P1/P2.
    evidence = evaluation.intervention_effectiveness({}, {
        "P0/QE": {"changed_pairs": 77, "max_absolute_weight_change": 1.0},
        "P2/QE": {"changed_pairs": 84, "max_absolute_weight_change": 2.0}})
    assert evidence["effective_pair_intervention"]
    assert evidence["effective_weight_intervention"]
    assert not evidence["effective_anchor_pair_intervention"]
    assert "actual native" in evidence["effective_pair_intervention_scope"]


def test_native_weight_action_and_anchor_support_are_distinct():
    evidence = evaluation.intervention_effectiveness({"QM": 3}, {
        "P0/MM": {"changed_pairs": 0, "max_absolute_weight_change": .5}})
    assert not evidence["effective_pair_intervention"]
    assert evidence["effective_weight_intervention"]
    assert evidence["effective_anchor_pair_intervention"]
    fallback = evaluation.intervention_effectiveness({"QM": 3})
    assert fallback["effective_pair_intervention"]
    assert "unmeasured" in fallback["effective_pair_intervention_scope"]


def test_inverse_trail_atomic_write_keeps_previous_complete_file_on_interruption(tmp_path, monkeypatch):
    from thermal_campaign_heat_inference import atomic_npz, completed_trial
    path = tmp_path / "trial.npz"
    values = np.ones((4, 2))
    atomic_npz(path, iterations=np.arange(4), heat=values,
               observed_predictions=values, held_predictions=values)
    record = {"optimizer_steps": 3}
    assert completed_trial(path, record, steps=3)
    previous = path.read_bytes()

    def interrupted(stream, **_arrays):
        stream.write(b"partial checkpoint")
        raise OSError("simulated interrupted disk write")

    monkeypatch.setattr(np, "savez_compressed", interrupted)
    with pytest.raises(OSError, match="interrupted disk"):
        atomic_npz(path, iterations=np.arange(4))
    assert path.read_bytes() == previous
    assert completed_trial(path, record, steps=3)
    assert not completed_trial(path, record, steps=30)
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"partial archive")
    assert not completed_trial(corrupt, record, steps=3)


@pytest.mark.skipif(not os.environ.get("HONF_ALIGNMENT_THERMAL_CHECKPOINT"), reason="Needs retained native Thermal resources")
def test_native_train_only_summary_calibration_and_held_input_intervention():
    import torch
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.evaluation.loading import load_model
    from thermal_campaign_heat_inference import native_heat_predictor, sensor_panel

    from honf_forward_core.interface_fields.core import InterfaceFieldCore

    model, checkpoint = load_model(Path(os.environ["HONF_ALIGNMENT_THERMAL_CHECKPOINT"]), torch.device("cpu"))
    model.config.core_honf.forward_architecture = "adaptive_receiver_hypergraph_honf"
    model.core = InterfaceFieldCore(model.config.core_honf)
    model.eval()
    dataset_path = checkpoint["train_config"]["dataset"]["packed_h5_path"]
    summary, ids = evaluation.calibrate_training_summary(model, checkpoint, dataset_path, count=2)
    train = GlobalChannelThermalDataset(dataset_path, split="train", points_per_case=1, include_grid=True)
    held = GlobalChannelThermalDataset(dataset_path, split="test", points_per_case=1, include_grid=True)
    assert set(ids).issubset(set(train.selected_case_ids)) and not set(ids).intersection(held.selected_case_ids)
    assert summary.case_counts == {0: 2, 1: 2, 2: 2}
    model.core.backend.training_population_summary = summary
    sample = held[0]
    sensors, _, _, observed, unseen = sensor_panel(sample)
    predictor, _ = native_heat_predictor(model, checkpoint, sample, sensors, observed, unseen)
    heat = torch.as_tensor(sample["structure"]["heat_powers"])
    with torch.no_grad(), evaluation.intervention(model, "fixed_summary"):
        prediction = predictor(heat)
    assert torch.isfinite(prediction["observed"]).all()
    assert model.core.backend.plan_intervention == "normal"
    model.requires_grad_(False)
    caller_heat = heat.clone().requires_grad_()
    with evaluation.intervention(model, "fixed_summary"):
        live = predictor(caller_heat)
        gradient, = torch.autograd.grad(live["observed"].sum(), caller_heat)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    damaged = copy.deepcopy(sample)
    for key in ("steady_field", "interface_target", "module_internal_temperature_points", "teacher_port_tokens"):
        damaged[key][...] = np.nan
    damaged["interface_condition"][..., 3:] = np.nan
    damaged["structure"]["heat_powers"] += 1000
    no_labels, _ = native_heat_predictor(model, checkpoint, damaged, sensors, observed, unseen)
    with torch.no_grad(), evaluation.intervention(model, "fixed_summary"):
        target_free = no_labels(heat)
    for key in ("observed", "held", "peaks", "pressure"):
        torch.testing.assert_close(target_free[key], live[key])
