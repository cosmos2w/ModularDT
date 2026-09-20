from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[1] / "tools/diagnostics/reduce_run1404_1406_1407_1804_accuracy.py"
_SPEC = importlib.util.spec_from_file_location("run_accuracy_reducer", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_metric_summary_reconciles_pooled_and_equal_case_values() -> None:
    rows = [
        {"metric_l2": "0.1", "metric_sse": "1", "metric_target_sse": "99", "metric_num_values": "10"},
        {"metric_l2": "0.2", "metric_sse": "4", "metric_target_sse": "96", "metric_num_values": "20"},
    ]

    summary = _MODULE.metric_summary(rows, "metric")

    assert summary is not None
    assert summary["num_cases"] == 2
    assert summary["equal_case_mean"] == pytest.approx(0.15)
    assert summary["pooled_relative_l2"] == pytest.approx((5 / 195) ** 0.5)
    assert summary["pooled_num_values"] == 30


def test_metric_summary_accepts_physical_relative_l2_suffix() -> None:
    rows = [
        {"physical_relative_l2": "0.25", "physical_sse": "1", "physical_target_sse": "16", "physical_num_values": "4"},
    ]

    summary = _MODULE.metric_summary(rows, "physical")

    assert summary is not None
    assert summary["pooled_relative_l2"] == pytest.approx(0.25)


def test_paired_summary_counts_wins_losses_and_ties_on_common_cases() -> None:
    candidate = {
        "a": {"global_field_fluid_norm_l2": "0.1"},
        "b": {"global_field_fluid_norm_l2": "0.3"},
        "c": {"global_field_fluid_norm_l2": "0.2"},
    }
    baseline = {
        "a": {"global_field_fluid_norm_l2": "0.2"},
        "b": {"global_field_fluid_norm_l2": "0.3"},
        "c": {"global_field_fluid_norm_l2": "0.1"},
        "extra": {"global_field_fluid_norm_l2": "9.0"},
    }

    result = _MODULE.paired_summary(candidate, baseline, "global_field_fluid_norm")

    assert result["num_cases"] == 3
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["ties"] == 1
