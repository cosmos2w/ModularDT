from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest
from honf_runtime.unified_training import _render_loss_curves


def _capture_figures(monkeypatch, output: Path) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    def savefig(figure, destination, *, format, **kwargs):
        del kwargs
        captured.append({
            "format": format,
            "yscales": [axis.get_yscale() for axis in figure.axes],
            "titles": [axis.get_title() for axis in figure.axes],
            "labels": [[line.get_label() for line in axis.lines] for axis in figure.axes],
            "lines": [[line.get_ydata() for line in axis.lines] for axis in figure.axes],
            "texts": [[text.get_text() for text in axis.texts] for axis in figure.axes],
            "title": figure._suptitle.get_text(),
        })
        Path(destination).write_bytes(b"test-render")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", savefig)
    output.mkdir(parents=True, exist_ok=True)
    return captured


def _objective(terms: dict[str, float], matched: list[str], train: float, validation: float):
    return {
        "terms": terms,
        "matched_terms": matched,
        "training_matched_total": train,
        "validation_matched_total": validation,
    }


def test_loss_curves_group_terms_compare_matched_objective_and_write_both_aliases(tmp_path, monkeypatch):
    history = [
        {"epoch": 1, "arm": "adaptive_detail", "case_visits": 600,
         "train_losses": {"reconstruction": 2.0, "base_loss": 0.2, "response": 0.02},
         "validation": {"field_score": 0.5},
         "validation_objective": _objective({"reconstruction": 1.0}, ["reconstruction"], 2.0, 1.0)},
        {"epoch": 2, "arm": "adaptive_detail", "case_visits": 600,
         "train_losses": {"reconstruction": 1.0, "base_loss": 0.1, "response": 0.01},
         "validation": {"field_score": 0.25},
         "validation_objective": _objective({"reconstruction": 0.5}, ["reconstruction"], 1.0, 0.5)},
    ]
    before = copy.deepcopy(history)
    captured = _capture_figures(monkeypatch, tmp_path)

    _render_loss_curves(history, tmp_path, "field_score")

    assert [item["format"] for item in captured] == ["pdf", "png"]
    for item in captured:
        assert len(item["yscales"]) == 6
        assert item["yscales"] == ["log"] * 6
        assert "adaptive_detail" in item["title"] and "600 TRAIN cases/epoch" in item["title"]
        assert "exposed VALIDATION hard inference" in item["title"]
        assert item["titles"] == [
            "Shared-name loss summaries\nTRAIN: mean of per-update means\n"
            "VALIDATION: pooled valid-element mean\nAdaptive TRAIN: 25% fine-path loss replay\n"
            "VALIDATION: hard main path",
            "Prediction losses\nPrediction vs reference; scaled squared error",
            "Physics and response losses\nDiscrete stored-flow residual; response prediction",
            "Held-out field prediction error\nNormalized field MSE for checkpoint selection; not a loss",
            "Approximation and routing losses\nCoarse/fine prediction mismatch; router probability error vs residual-importance target\n"
            "Penalty on expected selected fine reads",
            "Recorded TRAIN work counters\nCounter sums; not measured executor time or savings",
        ]
        assert item["labels"][1] == [
            "TRAIN: Field prediction loss",
            "HELD-OUT VALIDATION: Field prediction loss",
        ]
        assert item["labels"][2] == ["TRAIN: Response prediction loss"]
        assert item["labels"][0] == [
            "TRAIN shared-name sum", "HELD-OUT VALIDATION shared-name sum"]
        assert item["labels"][3] == ["HELD-OUT VALIDATION: Held-out field prediction error"]
        assert all("Counts do not establish sparse executor savings" not in text
                   for text in item["texts"][5])
    metadata = json.loads((tmp_path / "loss_curves_metadata.json").read_text())
    assert "do not establish sparse executor savings" in metadata["panel_definitions"]["train_work"]
    assert "0.05" in metadata["term_definitions"]["reconstruction"]["formula"]
    assert metadata["term_definitions"]["reconstruction"]["weight"].startswith("Overall provider weight 1.0")
    assert "not directly comparable" in metadata["panel_definitions"]["shared_name_loss_summary"]
    assert metadata["directly_comparable"] is False
    assert "macro-update" in metadata["training_reduction"]
    assert "fine-path loss replay" in metadata["prediction_path_comparison"]
    assert (tmp_path / "loss_curves.pdf").read_bytes() == b"test-render"
    assert (tmp_path / "loss_curves.png").read_bytes() == b"test-render"
    assert history == before


def test_loss_curves_mask_invalid_values_without_mutating_saved_history(tmp_path, monkeypatch):
    history = [
        {"epoch": 1, "arm": "adaptive_detail", "case_visits": 600,
         "train_losses": {"reconstruction": 1.0}, "validation": {"field_score": 0.5},
         "validation_objective": _objective({"reconstruction": 0.5}, ["reconstruction"], 1.0, 0.5)},
        {"epoch": 2, "arm": "adaptive_detail", "case_visits": 600,
         "train_losses": {"reconstruction": 0.0}, "validation": {"field_score": 0.0},
         "validation_objective": _objective({"reconstruction": 0.0}, ["reconstruction"], 0.0, 0.0)},
        {"epoch": 3, "arm": "adaptive_detail", "case_visits": 600,
         "train_losses": {"reconstruction": -2.0}, "validation": {"field_score": -0.25},
         "validation_objective": _objective({"reconstruction": -0.25}, ["reconstruction"], -2.0, -0.25)},
    ]
    before = copy.deepcopy(history)
    captured = _capture_figures(monkeypatch, tmp_path)

    _render_loss_curves(history, tmp_path, "field_score")

    for item in captured:
        train_values = item["lines"][1][0]
        validation_values = item["lines"][1][1]
        assert np.ma.getmaskarray(train_values).tolist() == [False, True, True]
        assert np.ma.getmaskarray(validation_values).tolist() == [False, True, True]
        assert any("nonpositive/nonfinite values masked" in text for text in item["texts"][4])
    assert history == before


def test_legacy_field_selector_is_not_fabricated_as_validation_loss(tmp_path, monkeypatch):
    history = [{
        "epoch": 4,
        "arm": "adaptive_detail",
        "train_losses": {"reconstruction": 2.0, "base_loss": 0.1},
        "validation": {"field_score": 0.4},
    }]
    captured = _capture_figures(monkeypatch, tmp_path)

    _render_loss_curves(history, tmp_path, "field_score")

    for item in captured:
        assert item["labels"][1] == [
            "TRAIN: Field prediction loss",
        ]
        assert item["labels"][3] == ["HELD-OUT VALIDATION: Held-out field prediction error"]
        assert "Held-out loss was not recorded in this history." in item["texts"][0][0]
        assert not any(label.startswith("HELD-OUT VALIDATION: Field prediction loss")
                       for label in item["labels"][1])


def test_wind_role_losses_use_plain_names_with_definition(tmp_path, monkeypatch):
    history = [{
        "epoch": 1,
        "arm": "adaptive_detail",
        "train_losses": {
            "native_role/volume": 0.4,
            "native_role/hub_slab": 0.25,
            "native_role/near_turbine": 0.3,
            "native_role/downstream_envelope": 0.2,
        },
        "validation": {"field_score": 0.1},
    }]
    captured = _capture_figures(monkeypatch, tmp_path)

    _render_loss_curves(history, tmp_path, "field_score")

    assert captured[0]["labels"][1] == [
        "TRAIN: Downstream-region velocity loss",
        "TRAIN: Hub-height slab velocity loss",
        "TRAIN: Near-turbine hub-slab velocity loss",
        "TRAIN: Full-volume velocity loss",
    ]
    metadata = json.loads((tmp_path / "loss_curves_metadata.json").read_text())
    assert "s_role^2" in metadata["term_definitions"]["native_role/volume"]["formula"]
    assert metadata["term_definitions"]["native_role/volume"]["weight"] == "0.2 per native role."
    assert "not a physical wake" in metadata["term_definitions"]["native_role/downstream_envelope"]["role_region"]
    assert "hub height" in metadata["term_definitions"]["native_role/hub_slab"]["role_region"]
    assert "D is the reference rotor diameter" in metadata["term_definitions"]["native_role/hub_slab"]["role_region"]
    assert "D = reference rotor diameter" in captured[0]["titles"][1]


def test_round13_member_routing_native_losses_and_zero_read_work_are_explicit(tmp_path, monkeypatch):
    loss_names = (
        "flow/u", "flow/v", "flow/p", "flow/omega", "temperature/fluid",
        "temperature/surface", "temperature/material", "q_proxy",
    )
    work_keys = (
        "core_decode_primary_query_M_selected_value_rows",
        "core_decode_primary_query_E_selected_value_rows",
        "core_decode_primary_query_M_cheap_score_rows",
        "core_decode_primary_query_E_cheap_score_rows",
    )
    history = [
        {
            "epoch": epoch,
            "arm": "R13-3-Route",
            "phase": "joint",
            "case_visits": 600,
            "train_losses": {name: 0.2 / epoch for name in loss_names},
            "validation": {"field_score": 0.4 / epoch, "scope": "canonical_VAL89_exposed", "case_count": 89},
            "work_counts": dict(zip(work_keys, values)),
        }
        for epoch, values in ((1, (6, 3, 40, 60)), (2, (0, 0, 25, 30)))
    ]
    captured = _capture_figures(monkeypatch, tmp_path)

    _render_loss_curves(history, tmp_path, "field_score")

    figure = captured[0]
    assert figure["titles"][1] == "Prediction losses\nPrediction vs reference; scaled squared error"
    assert figure["labels"][1] == [
        "TRAIN: Vorticity loss", "TRAIN: Pressure loss", "TRAIN: Flow u loss", "TRAIN: Flow v loss",
        "TRAIN: Heat-flux proxy loss", "TRAIN: Fluid temperature loss", "TRAIN: Material temperature loss",
        "TRAIN: Surface temperature loss",
    ]
    assert "No separate residual / response objective" in figure["titles"][2]
    assert "No separately fitted residual or response objective" in figure["texts"][2]
    assert "No separate routing penalty" in figure["titles"][4]
    assert figure["yscales"][5] == "symlog"
    work_lines = dict(zip(figure["labels"][5], figure["lines"][5]))
    assert np.asarray(work_lines["M selected value rows (TRAIN)"]).tolist() == [6.0, 0.0]
    assert np.asarray(work_lines["E selected value rows (TRAIN)"]).tolist() == [3.0, 0.0]
    assert np.asarray(work_lines["M cheap-score rows (TRAIN)"]).tolist() == [40.0, 25.0]
    assert np.asarray(work_lines["E cheap-score rows (TRAIN)"]).tolist() == [60.0, 30.0]
    metadata = json.loads((tmp_path / "loss_curves_metadata.json").read_text())
    for name in loss_names:
        assert metadata["term_definitions"][name]["panel"] == "native_prediction"
    assert "No separately fitted residual or response objective" in metadata["panel_definitions"]["physics"]
    assert "core_decode_primary_query_M_selected_value_rows" in metadata["work_counter_definitions"]
    assert "core_decode_primary_query_E_cheap_score_rows" in metadata["work_counter_definitions"]


@pytest.mark.parametrize("arm", [
    "R13-5-ME-Joint95",
    "R13-5A-ME-ConsistentJoint95",
    "R13-5B-ME-OrganizationConditionedJoint95",
])
def test_r13_5_paired_native_losses_and_actual_full_restricted_work_are_explicit(tmp_path, monkeypatch, arm):
    loss_names = (
        "flow/u", "flow/v", "flow/p", "flow/omega", "temperature/fluid",
        "temperature/surface", "temperature/material", "q_proxy",
    )
    work_counts = {
        "core_decode_joint_full_query_M_selected_value_rows": 6,
        "core_decode_joint_restricted_query_M_selected_value_rows": 7,
        "core_decode_joint_full_query_E_selected_value_rows": 3,
        "core_decode_joint_restricted_query_E_selected_value_rows": 5,
        "core_decode_joint_full_query_M_cheap_score_rows": 40,
        "core_decode_joint_restricted_query_M_cheap_score_rows": 25,
        "core_decode_joint_full_query_E_cheap_score_rows": 60,
        "core_decode_joint_restricted_query_E_cheap_score_rows": 30,
        "core_decode_shared_query_joint_address_score_rows": 11,
    }
    history = [{
        "epoch": 6,
        "arm": arm,
        "phase": "joint",
        "case_visits": 150,
        "train_losses": {name: 0.2 for name in loss_names},
        "validation": {"field_score": 0.3, "scope": "fixed25_v1_DEV22_exposed", "case_count": 22},
        "validation_objective": _objective(
            {name: 0.1 for name in loss_names}, list(loss_names), 0.8, 0.4,
        ),
        "work_counts": work_counts,
    }]
    before = copy.deepcopy(history)
    captured = _capture_figures(monkeypatch, tmp_path)

    _render_loss_curves(history, tmp_path, "field_score")

    figure = captured[0]
    assert "paired loss: 0.5 full + 0.5 joint95 restricted" in figure["title"]
    assert figure["titles"][1] == "Prediction losses\nPrediction vs reference; scaled squared error"
    expected_native_train_labels = [
        "TRAIN: Vorticity loss", "TRAIN: Pressure loss", "TRAIN: Flow u loss", "TRAIN: Flow v loss",
        "TRAIN: Heat-flux proxy loss", "TRAIN: Fluid temperature loss", "TRAIN: Material temperature loss",
        "TRAIN: Surface temperature loss",
    ]
    assert [label for label in figure["labels"][1] if label.startswith("TRAIN:")] == expected_native_train_labels
    assert sum(label.startswith("HELD-OUT VALIDATION:") for label in figure["labels"][1]) == len(loss_names)
    assert "No prediction losses saved" not in figure["texts"][1]
    assert "exposed DEV joint95 inference" in figure["title"]
    work_series = dict(zip(figure["labels"][5], figure["lines"][5]))
    assert np.asarray(work_series[
        "M selected value rows (full + joint95 restricted TRAIN)"
    ]).tolist() == [13.0]
    assert np.asarray(work_series[
        "E selected value rows (full + joint95 restricted TRAIN)"
    ]).tolist() == [8.0]
    assert np.asarray(work_series[
        "M cheap-score rows (full + joint95 restricted TRAIN)"
    ]).tolist() == [65.0]
    assert np.asarray(work_series[
        "E cheap-score rows (full + joint95 restricted TRAIN)"
    ]).tolist() == [90.0]
    assert np.asarray(work_series["Shared joint-address score rows (TRAIN)"]).tolist() == [11.0]
    assert "No recorded work counters" not in figure["texts"][5]
    assert "Sum full + joint95 restricted TRAIN counts" in figure["titles"][5]
    assert "Shared address score once" in figure["titles"][5]
    metadata = json.loads((tmp_path / "loss_curves_metadata.json").read_text())
    assert metadata["validation_scope"] == "fixed exposed DEV joint inference; not an independent TEST result"
    assert metadata["training_objective_pairing"]["full_loss_weight"] == 0.5
    assert metadata["training_objective_pairing"]["restricted_loss_weight"] == 0.5
    assert "joint95 restricted" in metadata["training_objective_pairing"]["validation_loss"]
    for name in loss_names:
        assert metadata["term_definitions"][name]["panel"] == "native_prediction"
        assert "0.5 * matching all-eligible full-reader" in metadata["term_definitions"][name]["formula"]
    assert "core_decode_joint_full_query_M_selected_value_rows" in metadata["work_counter_definitions"]
    assert "core_decode_joint_restricted_query_E_cheap_score_rows" in metadata["work_counter_definitions"]
    assert "core_decode_shared_query_joint_address_score_rows" in metadata["work_counter_definitions"]
    assert "full + joint95 restricted" in metadata["paired_work_series"][
        "M selected value rows (full + joint95 restricted TRAIN)"
    ]["aggregation"]
    shared_series = metadata["paired_work_series"]["Shared joint-address score rows (TRAIN)"]
    assert shared_series["source_counters"] == ["core_decode_shared_query_joint_address_score_rows"]
    assert "once per paired decode" in shared_series["aggregation"]
    assert history == before


@pytest.mark.parametrize("arm", [
    "R13-5-M-Joint95",
    "R13-5A-ME-ConsistentJoint95",
    "R13-5B-ME-OrganizationConditionedJoint95",
])
def test_r13_5_full_train_history_uses_canonical_val89_label(tmp_path, monkeypatch, arm):
    history = [{
        "epoch": 1,
        "arm": arm,
        "phase": "joint",
        "case_visits": 600,
        "train_losses": {"flow/u": 0.2},
        "validation": {"field_score": 0.3},
    }]
    captured = _capture_figures(monkeypatch, tmp_path)

    _render_loss_curves(history, tmp_path, "field_score")

    assert "exposed VAL89 joint95 inference" in captured[0]["title"]
    metadata = json.loads((tmp_path / "loss_curves_metadata.json").read_text())
    assert metadata["validation_scope"] == "exposed canonical VAL89 joint inference; not an independent TEST result"
