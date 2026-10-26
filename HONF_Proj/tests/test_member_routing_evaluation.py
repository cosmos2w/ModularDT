from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
for _directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from honf_runtime.member_routing_evaluation import (
    PROBE_NAMES,
    _evaluate_native_split,
    _fixed_probe_evidence,
    _query_indices,
    _route_interventions_enabled,
    _validate_benchmark,
    select_low_m_train_cases,
    select_validation_representatives,
)


def _provider():
    cases = (
        {"case_id": "case-z", "structure": {"module_present": np.ones(3)}},
        {"case_id": "case-a", "structure": {"module_present": np.ones(1)}},
        {"case_id": "case-c", "structure": {"module_present": np.ones(1)}},
        {"case_id": "case-b", "structure": {"module_present": np.ones(2)}},
        {"case_id": "case-y", "structure": {"module_present": np.ones(10)}},
    )
    return SimpleNamespace(training_cases=cases, training_case_ids=tuple(case["case_id"] for case in cases))


def test_low_m_panel_uses_input_counts_then_stable_case_id_and_caps_at_four() -> None:
    provider = _provider()

    assert select_low_m_train_cases(provider, 4) == ("case-a", "case-c", "case-b", "case-z")
    with pytest.raises(ValueError, match="limited"):
        select_low_m_train_cases(provider, 5)


def test_low_m_panel_caps_each_of_m1_and_m2_before_filling_larger_strata() -> None:
    cases = tuple({
        "case_id": case_id,
        "structure": {"module_present": np.ones(count)},
    } for case_id, count in (
        ("case-1-a", 1), ("case-1-b", 1), ("case-1-c", 1),
        ("case-2-a", 2), ("case-2-b", 2), ("case-3-a", 3),
    ))
    provider = SimpleNamespace(training_cases=cases, training_case_ids=tuple(case["case_id"] for case in cases))

    assert select_low_m_train_cases(provider, 4) == (
        "case-1-a", "case-1-b", "case-2-a", "case-2-b",
    )


def test_validation_detail_panel_prefers_fixed_m3_m7_m10_input_strata() -> None:
    counts = (3, 7, 10)
    cases = tuple({
        "case_id": f"val-m{count}",
        "structure": {"module_present": np.ones(count)},
    } for count in counts)
    provider = SimpleNamespace(
        validation_cases=cases,
        validation_case_ids=tuple(case["case_id"] for case in cases),
    )

    assert select_validation_representatives(provider, 3) == (
        "val-m3", "val-m7", "val-m10",
    )


class _MetricModel(torch.nn.Module):
    architecture = "R13-3-Route"


class _MetricProvider:
    def __init__(self):
        self.validation_case_ids = ("case-m1", "case-m3")
        self.validation_cases = tuple({
            "case_id": case_id,
            "structure": {"module_present": np.ones(count)},
        } for case_id, count in (("case-m1", 1), ("case-m3", 3)))
        self.training_case_ids = ("train-m1",)
        self.training_cases = ({"case_id": "train-m1", "structure": {"module_present": np.ones(1)}},)

    def native_evaluation_batches(self, *, split, case_keys, microbatch_size):
        assert split == "validation" and microbatch_size == 1
        for case_id in case_keys:
            yield SimpleNamespace(
                case_keys=(case_id,),
                targets=SimpleNamespace(case_ids=(case_id,)),
                scene_inputs=SimpleNamespace(),
                receivers=SimpleNamespace(),
            )

    def predict_native(self, model, scene, receivers, arm, phase, *, epoch):
        del scene, receivers, epoch
        assert model is self.model and arm == model.architecture and phase == "validation"
        x = torch.linspace(0.0, 1.0, 15)
        coordinates = torch.stack((x, torch.full_like(x, 0.5)), dim=-1)[None]
        query_mask = torch.ones((1, 15), dtype=torch.bool)
        query_mask[0, 7] = False
        prepared = SimpleNamespace(query_batch=SimpleNamespace(
            coordinates=coordinates, query_mask=query_mask,
        ))
        return SimpleNamespace(native_prepared=prepared), {"work": {"native_rows": 1}}

    def member_routing_trace(self, model, prediction, query_indices):
        del model
        coords = prediction.native_prepared.query_batch.coordinates[0]
        return tuple({
            "original_query_row": int(index),
            "coordinate": coords[index].tolist(),
            "module_ids": [11],
            "environment_ids": [21],
        } for index in query_indices)

    def validation_metrics(self, prediction, targets, auxiliary):
        del prediction, auxiliary
        case_id = targets.case_ids[0]
        count = int(case_id.removeprefix("case-m"))
        row = {
            "case_id": case_id,
            "module_count": count,
            "flow_rmse": {name: float(count) for name in ("u", "v", "p", "omega")},
            "roles": {
                "temperature_fluid": {"rmse": float(count + 1)},
                "temperature_surface": {"rmse": float(count + 2)},
                "temperature_material": {"rmse": float(count + 3)},
                "temperature_material_endpoint_maxima": {
                    "per_module": [{"error": float(count)} for _ in range(count)],
                },
            },
            "q_proxy_rmse": float(count + 4),
        }
        return {"case_rows": [row]}

    def work_counts(self, batch, prediction, auxiliary):
        del batch, prediction
        return auxiliary["work"]

    def reduce_native_evaluation_metrics(self, records, *, split, case_keys):
        assert split == "validation"
        rows = [row for record in records for row in record["case_rows"]]
        assert tuple(row["case_id"] for row in rows) == tuple(case_keys)
        return {"per_case": rows, "split": split}


def test_native_split_uses_all_nine_equal_case_metrics_and_counts_actual_work() -> None:
    provider = _MetricProvider()
    model = _MetricModel()
    provider.model = model
    result = _evaluate_native_split(
        model, provider, {"epoch": 100}, "validation", provider.validation_case_ids,
    )

    assert result["case_count"] == 2
    assert result["metrics"]["u"]["mean"] == pytest.approx(2.0)
    assert result["metrics"]["u"]["p90"] == pytest.approx(2.8)
    assert result["metrics"]["u"]["worst"] == pytest.approx(3.0)
    assert result["metrics"]["u"]["by_module_count"]["M1"]["mean"] == pytest.approx(1.0)
    assert result["metrics"]["temperature_material"]["mean"] == pytest.approx(5.0)
    assert result["executed_work_counts"]["native_rows"] == 2


def test_native_route_trace_keeps_union_row_indices_and_physical_coordinates(monkeypatch) -> None:
    provider = _MetricProvider()
    model = _MetricModel()
    provider.model = model
    monkeypatch.setattr(
        "honf_runtime.member_routing_evaluation._native_arrays_with_residuals",
        lambda prediction, batch, case_id: {},
    )

    result = _evaluate_native_split(
        model, provider, {"epoch": 100}, "validation", provider.validation_case_ids,
        export_ids={"case-m3"}, collect_routes=True,
    )

    trace = result["route_traces"]["case-m3"]
    assert trace["query_indices"] == [3, 6, 11]
    assert np.allclose(trace["query_coordinates"], [[3 / 14, 0.5], [6 / 14, 0.5], [11 / 14, 0.5]])
    assert [row["original_query_row"] for row in trace["queries"]] == [3, 6, 11]


def test_fixed_query_selection_uses_native_coordinates_and_skips_masked_rows() -> None:
    x = np.linspace(0.0, 1.0, 15)
    coordinates = np.stack((x, np.full_like(x, 0.5)), axis=-1)[None]
    query_mask = torch.ones((1, 15), dtype=torch.bool)
    query_mask[0, 7] = False
    prediction = SimpleNamespace(native_prepared=SimpleNamespace(
        query_batch=SimpleNamespace(coordinates=torch.as_tensor(coordinates), query_mask=query_mask),
    ))

    # This fluid-only list is deliberately shorter and differently indexed than the prepared union.
    receivers = SimpleNamespace(fluid_xy=torch.tensor([[[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]]))
    indices = _query_indices(receivers, prediction)

    assert len(indices) == len(set(indices)) == 3
    assert 7 not in indices
    assert max(indices) >= 10
    selected_x = coordinates[0, list(indices), 0]
    assert selected_x[0] == pytest.approx(0.2, abs=0.11)
    assert selected_x[1] == pytest.approx(0.5, abs=0.11)
    assert selected_x[2] == pytest.approx(0.8, abs=0.11)


def _native_fixture(case_id="M1", offset=0.0):
    module_count = 1
    target_field = torch.zeros((1, 3, 5), dtype=torch.float32)
    targets = SimpleNamespace(
        case_ids=(case_id,),
        field_targets=target_field,
        point_weights=torch.ones((1, 3)),
        interface_target=torch.zeros((1, module_count, 2, 2)),
        interface_valid_mask=torch.ones((1, module_count, 2)),
        material_targets=torch.zeros((1, module_count, 2)),
    )
    receivers = SimpleNamespace(
        heat=torch.ones((1, module_count)),
        fluid_xy=torch.tensor([[[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]]]),
    )
    prepared = SimpleNamespace(
        stencils={
            "fluid": SimpleNamespace(valid=torch.ones((1, 3), dtype=torch.bool)),
            "surface": SimpleNamespace(valid=torch.ones((1, module_count, 2), dtype=torch.bool)),
            "outside": SimpleNamespace(valid=torch.ones((1, module_count, 2), dtype=torch.bool)),
            "material": SimpleNamespace(valid=torch.ones((1, module_count, 2), dtype=torch.bool)),
        },
        source_present=torch.ones((1, module_count)),
    )
    prediction = SimpleNamespace(
        native_main={
            "pred_field": torch.full((1, 3, 5), float(offset)),
            "pred_interface": torch.full((1, module_count, 2, 2), float(offset)),
            "pred_internal_temperature": torch.full((1, module_count, 2, 1), float(offset)),
        },
        native_prepared=prepared,
    )
    batch = SimpleNamespace(
        targets=targets,
        receivers=receivers,
    )
    return batch, prediction


class _ProbeProvider:
    def member_routing_probe(self, model, batch, base_prediction, query_indices, probe_name):
        del model, batch, query_indices
        value = 1.0 if probe_name == "all_candidates_open" else 2.0
        _batch, prediction = _native_fixture(offset=value)
        return {
            "prediction": base_prediction if probe_name == "all_candidates_open" else prediction,
            "work_counts": {"selected_value_rows": 3, "cheap_score_rows": 8},
            "baseline_selected_member_counts": {"M": 1, "E": 2},
            "selected_member_counts": {"M": 1, "E": 2},
            "route_weight_changes": {"beta_l1": 0.25},
            "trace": ({"module_ids": [17], "environment_ids": [31, 32]},) * 3,
        }


def test_fixed_probes_keep_native_deltas_id_traces_and_work() -> None:
    batch, prediction = _native_fixture()
    base_trace = ({"module_ids": [17], "environment_ids": [31, 32]},) * 3

    result = _fixed_probe_evidence(
        model=object(),
        provider=_ProbeProvider(),
        batches={"M1": batch},
        base_predictions={"M1": prediction},
        query_indices={"M1": (0, 1, 2)},
        base_traces={"M1": base_trace},
    )

    assert tuple(result["M1"]) == PROBE_NAMES
    assert result["M1"]["b_fixed_candidates_scores"]["field_delta_rmse"] > 0
    assert result["M1"]["b_fixed_candidates_scores"]["route_weight_changes"] == {"beta_l1": 0.25}
    assert result["M1"]["b_fixed_candidates_scores"]["work_counts"] == {
        "selected_value_rows": 3,
        "cheap_score_rows": 8,
    }
    assert result["M1"]["all_candidates_open"]["dense_execution_control"] == {
        "expected_same_prediction": True,
        "max_abs_native_component_delta": 0.0,
        "within_1e-5_absolute_tolerance": True,
    }
    assert len(result["M1"]["all_candidates_open"]["trace"]) == 3


def test_geometry_b_probe_reports_unmatched_selected_counts_without_hiding_them() -> None:
    batch, prediction = _native_fixture()

    class Mismatched(_ProbeProvider):
        def member_routing_probe(self, model, batch, base_prediction, query_indices, probe_name):
            result = super().member_routing_probe(model, batch, base_prediction, query_indices, probe_name)
            if probe_name == "geometry_b_comparable_counts":
                result["selected_member_counts"] = {"M": 1, "E": 1}
            return result

    result = _fixed_probe_evidence(
        model=object(), provider=Mismatched(), batches={"M1": batch},
        base_predictions={"M1": prediction}, query_indices={"M1": (0, 1, 2)},
        base_traces={"M1": ({"module_ids": [17]},) * 3},
    )

    geometry = result["M1"]["geometry_b_comparable_counts"]
    assert geometry["selected_member_counts_match_baseline"] is False
    assert geometry["selected_member_counts"] == {"M": 1, "E": 1}


def test_fixed_probes_report_native_field_fidelity_beside_prediction_changes():
    from channelthermal.training.content_hypergraph_task import ThermalContentHypergraphTask
    from tests.test_native_budgeted_postfit import _thermal_stats

    class NativeProbeProvider(_ProbeProvider):
        adapter = SimpleNamespace(normalization_stats=_thermal_stats())
        validation_metrics = ThermalContentHypergraphTask.validation_metrics

    batch, prediction = _native_fixture()
    result = _fixed_probe_evidence(
        model=object(), provider=NativeProbeProvider(), batches={"M1": batch},
        base_predictions={"M1": prediction}, query_indices={"M1": (0, 1, 2)},
        base_traces={"M1": ({"module_ids": [17]},) * 3},
    )["M1"]
    changed = result["geometry_b_comparable_counts"]
    assert len(changed["native_observable_metrics"]) == 9
    assert changed["baseline_native_observable_metrics"]["temperature_material"]["rmse"] == 0.
    assert changed["native_observable_metrics"]["temperature_material"]["rmse"] == 2.
    assert result["all_candidates_open"]["native_observable_metrics"]["u"]["rmse"] == 0.


def test_cost_benchmark_preserves_warmup_repeats_and_all_phases() -> None:
    phases = {name: {"samples_seconds": [0.1, 0.2, 0.3]} for name in (
        "prepare", "cached_decode", "complete_native", "training_macro",
    )}
    result = _validate_benchmark({
        "warmup_count": 1,
        "repetitions": 3,
        "phases": phases,
        "work_counts": {"value_rows": 4},
        "peak_memory_bytes": 1024,
    }, case_id="train-M1")

    assert result["case_id"] == "train-M1"
    assert result["phases"]["prepare"]["samples_seconds"] == [0.1, 0.2, 0.3]
    assert result["phases"]["prepare"]["mean_seconds"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="three measured samples"):
        _validate_benchmark({
            "warmup_count": 1,
            "repetitions": 3,
            "phases": {**phases, "prepare": {"samples_seconds": [0.1]}},
        }, case_id="train-M1")


def test_route_interventions_are_skipped_for_full_arm_only() -> None:
    assert _route_interventions_enabled(SimpleNamespace(architecture="R13-3-Route")) is True
    assert _route_interventions_enabled(SimpleNamespace(architecture="R13-3-Full")) is False
    with pytest.raises(ValueError, match="Unsupported"):
        _route_interventions_enabled(SimpleNamespace(architecture="unknown"))
