"""Physical metrics and input-only development panel selection."""

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "tools/thermal_campaign_evaluate.py"
_SPEC = importlib.util.spec_from_file_location("thermal_campaign_evaluate", _PATH)
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


@pytest.mark.parametrize("name,mode", [("tensor_zero_residual", "zero_residual"),
                                     ("tensor_mean_access", "mean_access"),
                                     ("tensor_remove_group", "remove_group")])
def test_tensor_intervention_restores_mode_and_removal_on_exception(name, mode):
    residual = SimpleNamespace(intervention="normal", removed_group=3)
    backend = SimpleNamespace(tensor_residual=residual)
    model = SimpleNamespace(core=SimpleNamespace(backend=backend))
    with pytest.raises(RuntimeError, match="ownership fixture"), evaluation.intervention(model, name):
        assert residual.intervention == mode
        assert residual.removed_group is None
        raise RuntimeError("ownership fixture")
    assert residual.intervention == "normal"
    assert residual.removed_group == 3


def test_tensor_intervention_rejects_base_checkpoint():
    model = SimpleNamespace(core=SimpleNamespace(backend=SimpleNamespace()))
    with pytest.raises(ValueError, match="Tensor-H checkpoint"), evaluation.intervention(model, "tensor_zero_residual"):
        pass


@pytest.mark.parametrize("executor", [None, "dense_masked_reference", "rectangular_subset"])
def test_prediction_work_covers_native_phases_matches_ledger_and_excludes_later_reads(executor):
    from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
    from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
    from honf_forward_core.interface_fields.types import EncodedInterfaceCase

    torch.manual_seed(13)
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn(1, 2, 8, requires_grad=True), env_tokens=torch.randn(1, 3, 8),
        global_token=torch.randn(1, 8), module_centers=torch.rand(1, 2, 2),
        env_coords=torch.rand(1, 3, 2), module_present=torch.ones(1, 2),
        module_features=torch.randn(1, 2, 3), env_features=None,
        env_weights=torch.ones(1, 3), coordinate_scale=torch.ones(1, 1, 2))
    backend = (DensePairwiseField(8, 12, 2, 2) if executor is None else TypedHypergraphField(
        8, 12, 2, 2, architecture="overlap_control_hypergraph_honf", spatial_dim=2,
        module_characteristic_length=.03)).eval()
    if executor:
        backend.organizer.set_epoch(301)
        backend.set_execution_mode(executor, receiver_chunk_size=1)
    query, features = torch.rand(1, 2, 2), torch.rand(1, 2, 6)
    ledgers = []

    def predict():
        outputs = []
        for phase in range(3):
            context = {"interaction_context": SimpleNamespace(phase=f"P{phase}")} if executor else {}
            prepared = backend.prepare(encoded, encoded.module_tokens, **context)
            output, auxiliary = backend.read(prepared, encoded, query, features)
            ledgers.append({**prepared.get("hypergraph_ledger", {}), **auxiliary})
            outputs.append(output)
        return torch.stack(outputs)

    reference = predict()
    expected_gradient = torch.autograd.grad(reference.square().mean(), encoded.module_tokens)[0]
    ledgers.clear()
    prediction, work = evaluation.predict_with_fine_work(backend, predict)
    torch.testing.assert_close(prediction, reference, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(prediction.square().mean(), encoded.module_tokens)[0]
    assert actual_gradient.abs().sum() > 0
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert work["measured"] and "attention" in work["excluded"]
    if executor:
        for route, counts in work["routes"].items():
            assert counts["padded_input_rows"] == sum(int(row[f"hypergraph_{route}_executed_rows"]) for row in ledgers)
            assert counts["calls"] == sum(int(row[f"hypergraph_{route}_fine_calls"]) for row in ledgers)
    else:
        assert work["routes"] == {route: {"padded_input_rows": rows, "calls": 3}
                                  for route, rows in {"MM": 12, "ME": 18, "EM": 18, "QM": 12, "QE": 18}.items()}
    saved = copy.deepcopy(work)
    predict()  # later export/anchor reconstruction must not extend the measured prediction
    assert work == saved
    assert all(not module._forward_hooks for module in (backend.mm_message, backend.env_geometry_bias))
    with pytest.raises(RuntimeError, match="failed prediction"):
        evaluation.predict_with_fine_work(backend, lambda: (_ for _ in ()).throw(RuntimeError("failed prediction")))
    assert not backend.mm_message._forward_hooks


def test_historical_prediction_work_is_explicitly_unmeasured_without_changing_result():
    sentinel = object()
    result, work = evaluation.predict_with_fine_work(SimpleNamespace(), lambda: sentinel)
    assert result is sentinel
    assert work["measured"] is False and work["routes"] is None
    assert "five" in work["reason"].lower()


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


def test_input_declared_downstream_stratum_uses_only_inlet_direction_and_geometry():
    x, y = np.meshgrid(np.linspace(0, 10, 21), np.linspace(-4, 4, 17))
    reference = np.zeros((*x.shape, 5))
    sample = {"x_grid": x, "y_grid": y, "steady_field": reference,
        "structure": {"module_centers": np.asarray([[2., 0.]]), "module_present": np.asarray([1.]),
                      "u_in": np.asarray([1.]), "material_params": np.asarray([0., 0., 0., 0., 0., .5])},
        "interface_target": np.zeros((1, 4, 2)),
        "module_internal_temperature_points": np.zeros((1, 3)),
        "teacher_port_tokens": np.zeros((1, 4, 5))}
    prediction = {"pred_field_grid": reference.copy(), "pred_interface": np.zeros((1, 4, 2)),
        "pred_internal_temperature": np.zeros((1, 3)), "pred_port_condition": np.zeros((1, 4, 5))}
    metrics, masks = evaluation.physical_case_metrics(sample, prediction,
        ["u", "v", "p", "omega", "temperature"])
    expected = (masks["fluid_mask"] & (x - 2. > 1.) & (x - 2. <= 3.) & (np.abs(y) <= 1.))
    np.testing.assert_array_equal(masks["downstream_mask"], expected)
    assert masks["downstream_mask"].any()
    assert not (masks["downstream_mask"] & masks["near_mask"]).any()
    assert not (masks["downstream_mask"] & ~masks["far_mask"]).any()
    assert metrics["downstream/temperature"]["count"] == int(expected.sum())
    assert metrics["downstream/temperature"]["rmse"] == 0.
    sample["structure"]["u_in"] = np.asarray([0.])
    metrics, masks = evaluation.physical_case_metrics(sample, prediction,
        ["u", "v", "p", "omega", "temperature"])
    assert not masks["downstream_mask"].any()
    assert metrics["downstream/temperature"]["count"] == 0
    sample["structure"]["u_in"] = np.asarray([1.])
    sample["structure"]["module_centers"] = np.asarray([[2., 0.], [4.5, 0.]])
    sample["structure"]["module_present"] = np.asarray([1., 1.])
    sample["interface_target"] = np.zeros((2, 4, 2))
    sample["module_internal_temperature_points"] = np.zeros((2, 3))
    sample["teacher_port_tokens"] = np.zeros((2, 4, 5))
    prediction["pred_interface"] = np.zeros((2, 4, 2))
    prediction["pred_internal_temperature"] = np.zeros((2, 3))
    prediction["pred_port_condition"] = np.zeros((2, 4, 5))
    _, masks = evaluation.physical_case_metrics(sample, prediction,
        ["u", "v", "p", "omega", "temperature"])
    # x=3.5 is downstream of source 0, but within 2r of source 1.
    assert masks["near_mask"][8, 7]
    assert not masks["downstream_mask"][8, 7]
    assert not (masks["downstream_mask"] & masks["near_mask"]).any()


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


def test_case_median_and_worst_identity_preserve_ties_and_exclude_failures():
    rows = [{"case_id": identity, "metrics": {"T": evaluation.physical_errors(
        np.full(count, error), np.zeros(count))}}
        for identity, count, error in (("0001", 100, 1.), ("0002", 1, 7.),
                                      ("0003", 1, 7.), ("0004", 1, np.nan))]
    result = evaluation.aggregate_physical(rows)["T"]
    assert result["equal_case_rmse_median"] == 7.
    assert result["worst_case_ids"] == ["0002", "0003"]
    assert result["nonfinite_cases"] == 1
    assert result["pooled_rmse"] != result["equal_case_rmse_median"]


def test_public_native_context_has_exact_keys_and_keeps_nonfinite_inputs_explicit():
    structure = {"re": np.asarray([140.]), "u_in": np.asarray([1.]),
        "heat_powers": np.asarray([1., 2., 0.]),
        "material_params": np.asarray([.006, .01, .02, 1., 1., .45])}
    context = evaluation.native_physical_context(structure)
    assert set(context) == {"re", "u_in", "public_total_heat", "material_params",
                            "module_radius", "unavailable_fields"}
    assert context["re"] == 140. and context["u_in"] == 1.
    assert context["public_total_heat"] == 3. and context["module_radius"] == .45
    assert context["material_params"] == structure["material_params"].tolist()
    assert context["unavailable_fields"] == {}
    structure["re"][0] = np.nan
    structure["heat_powers"][0] = np.inf
    structure["material_params"][-1] = np.nan
    del structure["u_in"]
    context = evaluation.native_physical_context(structure)
    assert context["re"] is None and context["u_in"] is None
    assert context["public_total_heat"] is None and context["module_radius"] is None
    assert context["material_params"][-1] is None
    assert context["unavailable_fields"] == {"re": "nonfinite", "u_in": "missing",
        "public_total_heat": "nonfinite", "material_params": "nonfinite", "module_radius": "nonfinite"}
    json.dumps(context, allow_nan=False)
    overflow = evaluation.native_physical_context({"heat_powers": np.asarray([1e308, 1e308])})
    assert overflow["public_total_heat"] is None
    assert overflow["unavailable_fields"]["public_total_heat"] == "nonfinite"


def test_exact_strata_reuse_role_aggregates_and_reconcile_every_case_with_missing_context():
    one = evaluation.physical_errors(np.ones(1), np.zeros(1))
    nine = evaluation.physical_errors(np.full(9, 3.), np.zeros(9))
    two = evaluation.physical_errors(np.full(2, 2.), np.zeros(2))
    rows = [
        {"case_id": "0001", "module_count": 3, "physical_context": {"re": 80.}, "metrics": {"T": one}},
        {"case_id": "0273", "module_count": 3, "physical_context": {"re": 80.}, "metrics": {"T": nine}},
        {"case_id": "0002", "module_count": 5, "physical_context": {"re": 140.}, "metrics": {"T": two, "p": one}},
        {"case_id": "0003", "module_count": None, "metrics": {"T": two}},
        {"case_id": "0004", "module_count": 5,
         "physical_context": {"re": None, "unavailable_fields": {"re": "nonfinite"}}, "metrics": {"T": one}},
    ]
    strata = evaluation.aggregate_physical_strata(rows)
    assert strata["population_case_count"] == 5
    for dimension in ("by_module_count", "by_re"):
        assert sum(group["case_count"] for group in strata[dimension].values()) == 5
        assert sorted(case for group in strata[dimension].values() for case in group["case_ids"]) == sorted(row["case_id"] for row in rows)
    assert set(strata["by_module_count"]) == {"3", "5", "unavailable_missing"}
    assert set(strata["by_re"]) == {"80.0", "140.0", "unavailable_missing", "unavailable_nonfinite"}
    assert strata["by_module_count_unavailable_cases"] == 1
    assert strata["by_re_unavailable_cases"] == 2
    for key in (strata["by_module_count"]["3"], strata["by_re"]["80.0"]):
        assert key["case_count"] == 2
        assert key["metrics"]["T"]["equal_case_rmse_mean"] == 2.
        assert key["metrics"]["T"]["equal_case_mae_mean"] == 2.
        assert key["metrics"]["T"]["pooled_rmse"] == pytest.approx(np.sqrt(8.2))
        assert key["metrics"]["T"]["pooled_mae"] == 2.8
    assert strata["by_module_count"]["5"]["metrics"]["p"]["cases"] == 1
    assert strata["by_module_count"]["5"]["case_count"] == 2
    # Context failures do not alter the existing all-case metric denominator.
    assert evaluation.aggregate_physical(rows)["T"]["pooled_count"] == 15


def test_summary_strata_preserve_0273_compatibility_and_all_case_aggregates(tmp_path):
    def row(case_id, error, module_count, re=None):
        return {"case_id": case_id, "module_count": module_count, "intervention": "normal",
            "physical_context": {"re": re},
            "metrics": {"T": evaluation.physical_errors(np.asarray([error]), np.zeros(1))}}
    rows = [row("0273", 100., 3, 50.), row("0001", 1., 3, 80.), row("0002", 3., 5)]
    intervention = copy.deepcopy(rows[1])
    intervention["intervention"] = "full_access"
    rows.append(intervention)
    args = SimpleNamespace(stage=500, split="test", executor="dense_masked_reference", panel_config=None)
    evaluation.write_summary(tmp_path, rows, Path("unchanged_parent.pt"), {"epoch": 493},
                             args, Path("dataset.h5"), [0, 1, 2], ["T"])
    summary = json.loads((tmp_path / "summary.json").read_text())
    primary = summary["physical_strata"]["primary_excluding_0273"]
    compatibility = summary["physical_strata"]["compatibility_including_0273"]
    assert primary["population_case_count"] == 2 and compatibility["population_case_count"] == 3
    assert "50.0" not in primary["by_re"] and compatibility["by_re"]["50.0"]["case_ids"] == ["0273"]
    assert primary["by_re"]["unavailable_missing"]["case_ids"] == ["0002"]
    assert summary["primary_excluding_0273"] == evaluation.aggregate_physical(rows[1:3])
    assert summary["compatibility_including_0273"] == evaluation.aggregate_physical(rows[:3])
    assert summary["rows"] == rows


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


@pytest.mark.parametrize("stage", [500, 1000, 5000])
def test_reference_only_mature_stage_keeps_four_cases_and_normal_screen_keeps_all90(tmp_path, stage):
    class Dataset:
        split = "test"
        selected_case_ids = tuple(f"{index:04d}" for index in range(90))

        def __len__(self):
            return len(self.selected_case_ids)

    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"split": "test", "case_ids": ["0003", "0001", "0007", "0009"]}))
    args = SimpleNamespace(stage=stage, panel_config=panel, panel_size=4,
        interventions=["geometry_reference_actions", "full_access_fixed_controls"])
    assert evaluation.evaluation_indices(Dataset(), args) == [3, 1, 7, 9]
    args.interventions = ["normal"]
    assert evaluation.evaluation_indices(Dataset(), args) == list(range(90))
    args.interventions = ["normal", "full_access"]
    assert evaluation.evaluation_indices(Dataset(), args) == list(range(90))
    args.stage = 100
    assert evaluation.evaluation_indices(Dataset(), args) == [3, 1, 7, 9]


def test_formal5000_cli_keeps_original90_and_canonical89_without_inference(tmp_path):
    command = ["--checkpoint", "selected.pt", "--output-dir", str(tmp_path),
               "--stage", "5000", "--evaluation-scope", "formal-full",
               "--save-field-arrays", "panel", "--phase-graph-scope", "panel"]
    args = evaluation.parse_args(command)
    evaluation.configure_evaluation_scope(args, None)
    indices = evaluation.evaluation_indices(list(range(90)), args)
    assert indices == list(range(90))
    assert sum(evaluation.save_case_arrays(args, index, {1, 4, 8, 20}) for index in indices) == 4
    one = evaluation.physical_errors(np.ones(1), np.zeros(1))
    ids = ["0273", *(f"{index:04d}" for index in range(1, 90))]
    rows = [{"case_id": identity, "intervention": "normal", "metrics": {"T": one}} for identity in ids]
    evaluation.write_summary(tmp_path, rows, Path("selected.pt"), {"epoch": 5000},
                             args, Path("dataset.h5"), indices, ["T"])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["requested_stage"] == summary["checkpoint_epoch"] == 5000
    assert summary["completed_normal_cases"] == 90 and summary["dataset_scope"] == "formal-full"
    assert summary["primary_excluding_0273"]["T"]["cases"] == 89
    assert summary["compatibility_including_0273"]["T"]["cases"] == 90
    with pytest.raises(SystemExit):
        evaluation.parse_args([*command, "--development-manifest", "quarter.json"])
    with pytest.raises(SystemExit):
        evaluation.parse_args([*command, "--stage", "1500"])


def test_native_context_scalar_only_keeps_metric_cohort_and_disables_panel_archives(tmp_path):
    from honf_forward_core.evaluation.native_context_evidence import NativeContextEvidence
    from honf_forward_core.interface_fields.common import SharedInterfaceContext

    command = ["--checkpoint", "selected.pt", "--output-dir", str(tmp_path), "--stage", "500",
               "--save-field-arrays", "none", "--native-context-evidence"]
    args = evaluation.parse_args(command)
    assert not args.native_context_scalars_only
    panel = {1, 4, 8, 20}
    assert sum(evaluation.capture_native_context_arrays(args, index, panel) for index in range(22)) == 4
    args = evaluation.parse_args([*command, "--native-context-scalars-only"])
    evaluation.configure_evaluation_scope(args, {"manifest_sha256": "a" * 64})
    assert evaluation.evaluation_indices(list(range(22)), args) == list(range(22))
    assert not any(evaluation.save_case_arrays(args, index, panel) for index in range(22))
    # Constructor-only sentinel: no model initialization or physical forward.
    common = object.__new__(SharedInterfaceContext)
    common.__dict__["coarse_module_source"] = "module_states"
    core = SimpleNamespace(training=False, common=common)
    for index in range(22):
        recorder = NativeContextEvidence(core,
            save_arrays=evaluation.capture_native_context_arrays(args, index, panel))
        recorder._save("physical_rows", torch.ones(2))
        assert not recorder.save_arrays and recorder.arrays == {}
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--checkpoint", "selected.pt", "--output-dir", str(tmp_path),
                               "--stage", "500", "--native-context-scalars-only"])


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
