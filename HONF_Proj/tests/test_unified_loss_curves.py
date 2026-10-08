from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.figure
import numpy as np

from honf_runtime.unified_training import _render_loss_curves


def _capture_figures(monkeypatch, output: Path) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    def savefig(figure, destination, *, format, **kwargs):
        del kwargs
        axes = figure.axes
        captured.append({
            "format": format,
            "yscales": [axis.get_yscale() for axis in axes],
            "lines": [[line.get_ydata() for line in axis.lines] for axis in axes],
            "texts": [[text.get_text() for text in axis.texts] for axis in axes],
            "title": figure._suptitle.get_text(),
        })
        Path(destination).write_bytes(b"test-render")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", savefig)
    output.mkdir(parents=True, exist_ok=True)
    return captured


def test_loss_curves_use_log_scale_and_write_both_aliases(tmp_path, monkeypatch):
    history = [
        {"epoch": 1, "arm": "adaptive_detail", "case_visits": 600,
         "train_losses": {"reconstruction": 2.0}, "validation": {"field_score": 0.5}},
        {"epoch": 2, "arm": "adaptive_detail", "case_visits": 600,
         "train_losses": {"reconstruction": 1.0}, "validation": {"field_score": 0.25}},
    ]
    before = copy.deepcopy(history)
    captured = _capture_figures(monkeypatch, tmp_path)

    _render_loss_curves(history, tmp_path, "field_score")

    assert [item["format"] for item in captured] == ["pdf", "png"]
    assert all(item["yscales"] == ["log", "log"] for item in captured)
    assert all("adaptive_detail" in item["title"] and "600 TRAIN cases/epoch" in item["title"]
               for item in captured)
    assert (tmp_path / "loss_curves.pdf").read_bytes() == b"test-render"
    assert (tmp_path / "loss_curves.png").read_bytes() == b"test-render"
    assert history == before


def test_loss_curves_mask_and_label_nonpositive_values_without_mutating_history(tmp_path, monkeypatch):
    history = [
        {"epoch": 1, "arm": "adaptive_detail", "case_visits": 600,
         "train_losses": {"reconstruction": 1.0}, "validation": {"field_score": 0.5}},
        {"epoch": 2, "arm": "adaptive_detail", "case_visits": 600,
         "train_losses": {"reconstruction": 0.0}, "validation": {"field_score": 0.0}},
        {"epoch": 3, "arm": "adaptive_detail", "case_visits": 600,
         "train_losses": {"reconstruction": -2.0}, "validation": {"field_score": -0.25}},
    ]
    before = copy.deepcopy(history)
    captured = _capture_figures(monkeypatch, tmp_path)

    _render_loss_curves(history, tmp_path, "field_score")

    for item in captured:
        assert item["yscales"] == ["log", "log"]
        train_values = item["lines"][0][0]
        validation_values = item["lines"][1][0]
        assert np.ma.getmaskarray(train_values).tolist() == [False, True, True]
        assert np.ma.getmaskarray(validation_values).tolist() == [False, True, True]
        assert "2 nonpositive/nonfinite value(s) masked" in item["texts"][0][0]
        assert "2 nonpositive/nonfinite value(s) masked" in item["texts"][1][0]
    assert history == before


def test_all_nonpositive_loss_values_are_masked_and_labeled(tmp_path, monkeypatch):
    history = [{"epoch": 1, "case_visits": 2, "train_losses": {"base_loss": 0.0}}]
    captured = _capture_figures(monkeypatch, tmp_path)

    _render_loss_curves(history, tmp_path, "field_score")

    for item in captured:
        assert item["yscales"] == ["log", "log"]
        assert item["lines"][0] == []
        assert "1 nonpositive/nonfinite value(s) masked" in item["texts"][0][0]
