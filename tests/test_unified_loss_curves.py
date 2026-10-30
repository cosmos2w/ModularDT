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
            "epochs": [[line.get_xdata() for line in axis.lines] for axis in figure.axes],
            "size": tuple(figure.get_size_inches()),
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
        assert len(item["yscales"]) == 5
        assert item["yscales"] == ["log"] * 5
        assert "adaptive_detail" in item["title"] and "600 TRAIN cases/epoch" in item["title"]
        assert item["titles"] == ["Weighted loss totals (different reductions)",
                                  "Prediction losses", "Physics and response losses",
                                  "Exposed validation field error", "Other objective losses"]
        assert item["labels"][1] == [
            "TRAIN: Field prediction loss",
            "HELD-OUT VALIDATION: Field prediction loss",
        ]
        assert item["labels"][2] == ["TRAIN: Response prediction loss"]
        assert item["labels"][0] == [
            "TRAIN: recorded weighted sum", "VALIDATION: recorded total",
            "TRAIN: shared-term subtotal", "VALIDATION: shared-term subtotal"]
        assert np.allclose(item["lines"][0][0], [2.22, 1.11])
        assert np.allclose(item["lines"][0][2], [2.0, 1.0])
        assert item["size"] == (16, 9)
        assert item["labels"][3] == ["HELD-OUT VALIDATION: Held-out field prediction error"]
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
        assert any("nonpositive/nonfinite values masked" in text.lower() for panel in item["texts"] for text in panel)
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
        assert ["TRAIN: Field prediction loss"] in item["labels"]
        assert ["HELD-OUT VALIDATION: Held-out field prediction error"] in item["labels"]
        assert any("Validation objective losses were not recorded" in text for panel in item["texts"] for text in panel)
        assert not any(label.startswith("HELD-OUT VALIDATION: Field prediction loss")
                       for panel in item["labels"] for label in panel)


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


def test_total_panel_before_first_review_and_after_review(tmp_path, monkeypatch):
    history = [{"epoch": epoch, "arm": "R14-Detail-LocalECover", "phase": "joint",
                "train_losses": {"flow/u": 0.2 * epoch, "temperature/fluid": 0.3 * epoch},
                "paired_reader_native_losses": {"full": 1000.0}}
               for epoch in (1, 2, 3)]
    captured = _capture_figures(monkeypatch, tmp_path)
    _render_loss_curves(history[:2], tmp_path, "field_score")
    panel = captured[0]
    assert panel["titles"][0] == "Weighted loss totals (different reductions)"
    assert panel["epochs"][0][0].tolist() == [1, 2]
    assert np.allclose(panel["lines"][0][0], [0.5, 1.0])
    assert "Validation: first review pending" in panel["texts"][0]
    history[2]["validation_objective"] = _objective(
        {"flow/u": 0.4, "temperature/fluid": 0.6},
        ["flow/u", "temperature/fluid"], 1.5, 1.0)
    captured.clear()
    _render_loss_curves(history, tmp_path, "field_score")
    panel = captured[0]
    assert panel["epochs"][0][0].tolist() == [1, 2, 3]
    assert np.allclose(panel["lines"][0][0], [0.5, 1.0, 1.5])
    assert panel["epochs"][0][1].tolist() == [3]
    assert np.allclose(panel["lines"][0][1], [1.0])
    assert len(panel["labels"][0]) == 2


def test_nonfinite_total_is_masked_and_explicit_validation_total_is_used(tmp_path, monkeypatch):
    history = [
        {"epoch": 1, "train_losses": {"flow/u": 1.0, "flow/v": 2.0}},
        {"epoch": 2, "train_losses": {"flow/u": float("nan")}},
        {"epoch": 3, "train_losses": {"flow/u": 2.0, "flow/v": 3.0},
         "validation_objective": {"total": 7.0}},
    ]
    captured = _capture_figures(monkeypatch, tmp_path)
    _render_loss_curves(history, tmp_path, "field_score")
    assert np.ma.getmaskarray(captured[0]["lines"][0][0]).tolist() == [False, True, False]
    assert np.allclose(captured[0]["lines"][0][1], [7.0])


def test_phase_changes_preserve_each_epochs_recorded_total(tmp_path, monkeypatch):
    history = [
        {"epoch": 1, "phase": "warmup", "train_losses": {"flow/u": 1.0},
         "validation_objective": {"terms": {"flow/u": 0.5}}},
        {"epoch": 2, "phase": "joint", "train_losses": {"flow/u": 2.0, "flow/v": 3.0},
         "paired_reader_native_losses": {"full": 1000.0, "restricted": 1000.0},
         "validation_objective": {"terms": {"flow/u": 1.0, "flow/v": 1.5}}},
    ]
    captured = _capture_figures(monkeypatch, tmp_path)
    _render_loss_curves(history, tmp_path, "field_score")
    assert np.allclose(captured[0]["lines"][0][0], [1.0, 5.0])
    assert np.allclose(captured[0]["lines"][0][1], [0.5, 2.5])


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
    assert "Flow losses" in figure["titles"]
    assert "Thermal losses" in figure["titles"]
    assert not any("work" in title.lower() for title in figure["titles"])
    metadata = json.loads((tmp_path / "loss_curves_metadata.json").read_text())
    for name in loss_names:
        assert metadata["term_definitions"][name]["panel"] == "native_prediction"
    assert "No separately fitted residual or response objective" in metadata["panel_definitions"]["physics"]
    assert "core_decode_primary_query_M_selected_value_rows" in metadata["work_counter_definitions"]
    assert "core_decode_primary_query_E_cheap_score_rows" in metadata["work_counter_definitions"]


@pytest.mark.parametrize("arm", ["R13-8-A-EAnchor", "R13-8-L-LocalECover"])
def test_anchored_paired_history_renders_native_losses_and_actual_work(tmp_path, monkeypatch, arm):
    names = ("flow/u", "flow/v", "flow/p", "flow/omega", "temperature/fluid",
             "temperature/surface", "temperature/material", "q_proxy")
    work = {
        "core_decode_joint_full_query_anchor_value_reduction_pairs_M": 10,
        "core_decode_joint_restricted_query_anchor_value_reduction_pairs_M": 7,
        "core_decode_joint_full_query_anchor_value_reduction_pairs_E": 20,
        "core_decode_joint_restricted_query_anchor_value_reduction_pairs_E": 8,
        "core_decode_joint_shared_query_unique_union_raw_score_pairs_M": 11,
        "core_decode_joint_shared_query_unique_union_raw_score_pairs_E": 23,
    }
    history = [{"epoch": 200, "arm": arm, "phase": "joint", "case_visits": 150,
                "train_losses": {name: .2 for name in names},
                "validation": {"scope": "fixed25_v1_DEV22_exposed", "case_count": 22,
                               "field_score": .3},
                "validation_objective": _objective({name: .1 for name in names}, list(names), 1.6, .8),
                "work_counts": work}]
    original = copy.deepcopy(history)
    figure = _capture_figures(monkeypatch, tmp_path)
    _render_loss_curves(history, tmp_path, "field_score")
    assert arm in figure[0]["title"]
    assert "Flow losses" in figure[0]["titles"] and "Thermal losses" in figure[0]["titles"]
    assert not any("work" in title.lower() for title in figure[0]["titles"])
    metadata = json.loads((tmp_path / "loss_curves_metadata.json").read_text())
    assert all(metadata["term_definitions"][name]["panel"] == "native_prediction" for name in names)
    assert metadata["training_objective_pairing"]["full_loss_weight"] == .5
    assert metadata["training_objective_pairing"]["restricted_loss_weight"] == .5
    assert "not latency or executor savings" in metadata["panel_definitions"]["train_work"]
    assert history == original


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
    assert arm in figure["title"]
    native_indices = [i for i, title in enumerate(figure["titles"]) if title in ("Flow losses", "Thermal losses")]
    native_labels = [label for i in native_indices for label in figure["labels"][i]]
    assert sum(label.startswith("TRAIN:") for label in native_labels) == len(loss_names)
    assert sum(label.startswith("HELD-OUT VALIDATION:") for label in native_labels) == len(loss_names)
    assert not any("work" in title.lower() for title in figure["titles"])
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

    assert any("Exposed val89 field error" in title for title in captured[0]["titles"])
    metadata = json.loads((tmp_path / "loss_curves_metadata.json").read_text())
    assert metadata["validation_scope"] == "exposed canonical VAL89 joint inference; not an independent TEST result"
