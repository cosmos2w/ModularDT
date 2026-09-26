from __future__ import annotations

import numpy as np
import pytest

from windfarm.decision_library import (
    build_input_only_w2_cohorts,
    score_finite_library,
    static_geometry_features,
)
from windfarm.splits import GroupSplit


class _OutcomeGuard(dict):
    def __getitem__(self, key):
        if key == "wake_loss_pct":
            raise AssertionError("W2 cohort construction read an outcome target")
        return super().__getitem__(key)


def test_w2_cohorts_are_grouped_from_inputs_without_reading_outcomes() -> None:
    metadata = _OutcomeGuard(
        layout_index=np.repeat(np.arange(6), 2),
        n_turbines=np.full(12, 6),
        wd_deg=np.tile([270.0, 285.0], 6),
        wake_loss_pct=np.arange(12, dtype=np.float32),
    )
    split = GroupSplit(
        train=np.asarray([0, 1, 2, 3]),
        validation=np.asarray([4, 5, 6, 7]),
        test=np.asarray([8, 9, 10, 11]),
        metadata={"seed": 42, "group_key": "layout_index"},
    )

    manifest = build_input_only_w2_cohorts(metadata, split)

    assert manifest["input_columns_read"] == ["layout_index", "n_turbines", "wd_deg"]
    assert manifest["outcome_columns_read"] == []
    assert manifest["outcome_used_for_cohort_construction"] is False
    assert manifest["counts"] == {
        "train": {"eligible_cohorts": 2, "candidate_rows": 4, "distinct_layouts": 2},
        "validation": {"eligible_cohorts": 2, "candidate_rows": 4, "distinct_layouts": 2},
        "test": {"eligible_cohorts": 2, "candidate_rows": 4, "distinct_layouts": 2},
    }


def test_w2_cohort_builder_rejects_overlapping_split_rows() -> None:
    metadata = {
        "layout_index": np.asarray([0, 0, 1]),
        "n_turbines": np.asarray([6, 6, 6]),
        "wd_deg": np.asarray([270.0, 285.0, 270.0]),
    }
    split = GroupSplit(
        train=np.asarray([0, 1]),
        validation=np.asarray([1]),
        test=np.asarray([2]),
        metadata={},
    )

    with pytest.raises(ValueError, match="partition every row exactly once"):
        build_input_only_w2_cohorts(metadata, split)


def test_static_geometry_features_are_input_only_and_finite() -> None:
    turbine_xy = np.asarray(
        [
            [[-0.8, -0.4], [-0.5, 0.2], [-0.1, -0.1], [0.2, 0.4], [0.5, -0.2], [0.8, 0.1]],
            [[-1.0, -0.3], [-0.7, 0.25], [-0.25, -0.15], [0.25, 0.15], [0.7, -0.25], [1.0, 0.3]],
        ],
        dtype=np.float32,
    )
    features, names = static_geometry_features(turbine_xy, [6, 6], [270.0, 285.0])

    assert features.shape == (2, len(names))
    assert features.shape[1] == 23
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[0, :4], [0.2, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(features[1, :4], [0.2, 0.0, 1.0, 0.0])


def test_finite_library_scores_prediction_and_equal_count_controls() -> None:
    frozen = {
        "cohorts": [
            {
                "cohort_id": "validation:n6:wd270",
                "split": "validation",
                "n_turbines": 6,
                "wd_deg": 270,
                "row_indices": [0, 1, 2],
                "candidate_layout_indices": [11, 12, 13],
                "layout_indices": [11, 12, 13],
            },
            {
                "cohort_id": "test:n6:wd270",
                "split": "test",
                "n_turbines": 6,
                "wd_deg": 270,
                "row_indices": [3, 4],
                "candidate_layout_indices": [14, 15],
                "layout_indices": [14, 15],
            },
        ]
    }
    prediction = np.asarray([7.0, 3.0, 5.0, 100.0, -100.0])
    observed = np.asarray([8.0, 5.0, 2.0, -1000.0, 1000.0])

    result = score_finite_library(
        frozen,
        prediction,
        observed,
        split_name="validation",
    )

    assert result["cohort_count"] == 1
    assert result["candidate_rows_scored"] == 3
    assert result["selected_candidate_count"] == 1
    assert result["cohort_selections"][0]["selected_row"] == 1
    assert result["cohort_selections"][0]["selected_layout_index"] == 12
    assert result["cohort_selections"][0]["selected_observed_wake_loss_pct"] == 5.0
    assert result["cohort_selections"][0]["uniform_random_expected_wake_loss_pct"] == 5.0
    assert result["cohort_selections"][0]["selected_minus_uniform_random_expected_pct_points"] == 0.0
    assert result["cohort_selections"][0]["oracle_observed_wake_loss_pct"] == 2.0
    assert result["cohort_selections"][0]["candidate_stored_outcomes"][2]["observed_wake_loss_pct"] == 2.0
    assert result["random_control"]["policy"] == (
        "uniformly choose one stored row per same frozen cohort; exact expected value"
    )
    assert result["statistical_scope"]["sampling_confidence_intervals_reported"] is False
