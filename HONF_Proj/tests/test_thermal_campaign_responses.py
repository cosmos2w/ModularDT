"""Finite-response magnitudes and failures must survive reporting."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools/thermal_campaign_responses.py"
_SPEC = importlib.util.spec_from_file_location("thermal_campaign_responses", _PATH)
responses = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(responses)


def test_weighted_response_and_zero_control_retain_physical_units_without_unresolved_ratio():
    target = np.asarray([[1., 0.], [3., 0.]])
    prediction = np.asarray([[2., 2.], [1., 2.]])
    rows = responses.response_channel_metrics(prediction, target, np.ones(2, bool),
        np.asarray([1., 3.]), ["temperature", "p"], ["benchmark_T", "benchmark_p"])
    assert rows["temperature"]["rmse"] == pytest.approx(np.sqrt(3.25))
    assert rows["temperature"]["reference_rms"] == pytest.approx(np.sqrt(7.))
    assert rows["temperature"]["zero_change_rmse"] == rows["temperature"]["reference_rms"]
    assert rows["p"]["rmse"] == 2 and rows["p"]["reference_rms"] == 0
    assert rows["p"]["relative_accuracy"] is None
    assert rows["temperature"]["unit"] == "benchmark_T"


def test_nonfinite_and_empty_responses_are_reported_and_bad_quadrature_rejected():
    target = np.zeros((2, 1))
    rows = responses.response_channel_metrics(np.asarray([[np.nan], [0.]]), target,
        np.ones_like(target, bool), np.ones(2), ["T"], ["K"])
    assert not rows["T"]["finite"] and rows["T"]["rmse"] is None
    rows = responses.response_channel_metrics(target, target, np.zeros_like(target, bool),
        np.ones(2), ["T"], ["K"])
    assert rows["T"]["count"] == 0 and rows["T"]["rmse"] is None
    with pytest.raises(ValueError, match="quadrature"):
        responses.response_channel_metrics(target, target, np.ones_like(target, bool),
            np.asarray([1., -1.]), ["T"], ["K"])
