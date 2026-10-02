from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from channelthermal.response_control.historical import select_value_recovery_case_rows
from evaluate_value_recovery_pair import _summary_tables, finite_metrics
from run_value_recovery_pair import _remap_cuda_rng_state, _stream_pair_audit


def _row(update: int, value_weight: float) -> dict[str, object]:
    return {
        "completed_update": update,
        "training_family_id": "stored_family:0333",
        "historical_case_id": "0591",
        "training_stencil_index": 4,
        "active_term_weights": {"value": value_weight, "finite": 1.0, "finite_peak": 1.0},
        "training_metadata": {
            "query_seed": 115209219 + update,
            "source_plan_full_access_replay": False,
            "historical_stencil_action_at_source_cursor": "root",
            "effective_pilot_access": "full_access",
            "organizer_frozen": True,
            "shadow_optimization": "disabled",
        },
    }


def test_historical_cuda_zero_rng_stream_maps_to_physical_cuda_two() -> None:
    marker = object()
    assert _remap_cuda_rng_state({"cuda_rng_state_by_model_device": {"cuda:0": marker}}) == {"cuda:2": marker}


def test_current_ordinary_visibility_cuda_two_rng_stream_is_preserved() -> None:
    marker = object()
    assert _remap_cuda_rng_state({"cuda_rng_state_by_model_device": {"cuda:2": marker}}) == {"cuda:2": marker}


def test_pair_audit_allows_only_the_declared_value_weight_difference() -> None:
    audit = _stream_pair_audit([_row(1301, 4.0)], [_row(1301, 8.0)])
    assert audit["paired_stream_equal"] is True
    assert audit["mismatches"] == []


def test_pair_audit_detects_query_stream_divergence() -> None:
    left = _row(1301, 4.0)
    right = _row(1301, 8.0)
    right["training_metadata"]["query_seed"] = 999
    audit = _stream_pair_audit([left], [right])
    assert audit["paired_stream_equal"] is False
    assert any(row["field"] == "training_metadata.query_seed" for row in audit["mismatches"])


def test_value_recovery_panel_covers_every_count_and_focuses_reynolds_tails() -> None:
    strata = {
        count: [(f"{count:02d}-{index:02d}", 50.0 + index * 10.0) for index in range(11)]
        for count in (1, 2, 3, 4, 5, 6, 7, 9, 10, 12)
    }
    rows = select_value_recovery_case_rows(strata, requested=16)
    assert len(rows) == 16
    counts = {int(row["module_count"]) for row in rows}
    assert counts == set(strata)
    for focus in (3, 5, 7, 10):
        selected = [row for row in rows if row["module_count"] == focus]
        assert len(selected) >= 2
        assert {row["selection_role"] for row in selected} >= {"module_count_median_re", "re_quantile_0"}


def test_value_recovery_panel_rejects_insufficient_slots_for_count_coverage() -> None:
    strata = {count: [(f"case-{count}", 90.0)] for count in (1, 2, 3)}
    try:
        select_value_recovery_case_rows(strata, requested=2)
    except ValueError as exc:
        assert "cannot cover" in str(exc)
    else:
        raise AssertionError("A panel smaller than its number of count strata was accepted.")


def test_weighted_metrics_keep_nonfinite_predictions_visible() -> None:
    import numpy as np

    result = finite_metrics(
        np.asarray([1.0, 2.0, np.nan]),
        np.asarray([0.0, 2.0, 3.0]),
        np.asarray([1.0, 3.0, 1.0]),
        np.asarray([True, True, True]),
    )
    assert result["rmse_native_units"] == 0.5
    assert result["finite_prediction_entry_count"] == 2
    assert result["nonfinite_prediction_entry_count"] == 1
    assert result["finite_fraction"] == 2.0 / 3.0


def test_numeric_note_includes_solid_temperature_channel(tmp_path: Path) -> None:
    _summary_tables(
        tmp_path,
        [
            {
                "case_id": "train-case",
                "split": "train",
                "model": "value8",
                "role": "solid_temperature",
                "channel": "temperature",
                "units": "dataset temperature units",
                "quantity": "absolute_baseline",
                "metrics": {
                    "signal_relative_rmse_unitless": 0.1,
                    "finite_prediction_entry_count": 2,
                    "valid_entry_count": 2,
                    "max_absolute_error_native_units": 1.0,
                },
            }
        ],
        [],
        [],
    )
    note = (tmp_path / "numeric_note.md").read_text(encoding="utf-8")
    assert "solid_temperature/temperature" in note
