"""Input-only cohorts and bounded finite-library scoring for WindFarm W2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from .splits import GroupSplit

WIND_DIRECTIONS_DEG = (270.0, 285.0, 300.0)


def build_input_only_w2_cohorts(
    metadata: Mapping[str, Any],
    split: GroupSplit,
    *,
    minimum_distinct_layouts: int = 2,
) -> dict[str, Any]:
    """Freeze candidate cohorts from layout count and direction only.

    This function deliberately accepts no outcome array. It produces a cohort
    manifest before a caller reads ``wake_loss_pct`` or scores candidates.
    """

    if int(minimum_distinct_layouts) < 2:
        raise ValueError("minimum_distinct_layouts must be at least 2")
    required = ("layout_index", "n_turbines", "wd_deg")
    missing = [name for name in required if name not in metadata]
    if missing:
        raise KeyError(f"WindFarm W2 cohort inputs are missing {missing}")
    layout = np.asarray(metadata["layout_index"])
    turbine_count = np.asarray(metadata["n_turbines"])
    direction = np.asarray(metadata["wd_deg"], dtype=np.float64)
    if layout.ndim != 1 or turbine_count.shape != layout.shape or direction.shape != layout.shape:
        raise ValueError("layout_index, n_turbines, and wd_deg must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(layout.astype(np.float64))) or not np.all(np.isfinite(turbine_count.astype(np.float64))):
        raise ValueError("layout_index and n_turbines must contain finite values")
    if not np.all(np.isfinite(direction)):
        raise ValueError("wd_deg must contain finite values")
    if np.any(layout.astype(np.float64) != np.floor(layout.astype(np.float64))):
        raise ValueError("layout_index must contain integer group labels")
    if np.any(turbine_count.astype(np.float64) != np.floor(turbine_count.astype(np.float64))):
        raise ValueError("n_turbines must contain integer counts")

    partitions = {
        "train": np.asarray(split.train, dtype=np.int64),
        "validation": np.asarray(split.validation, dtype=np.int64),
        "test": np.asarray(split.test, dtype=np.int64),
    }
    for name, rows in partitions.items():
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= layout.size):
            raise ValueError(f"{name} split contains an invalid row index")
        if np.unique(rows).size != rows.size:
            raise ValueError(f"{name} split contains duplicate row indices")
    all_rows = np.concatenate(tuple(partitions.values()))
    if all_rows.size != layout.size or np.unique(all_rows).size != layout.size:
        raise ValueError("WindFarm W2 splits must partition every row exactly once")
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(np.asarray(getattr(split, first)).tolist()) & set(np.asarray(getattr(split, second)).tolist()):
            raise ValueError(f"WindFarm W2 {first}/{second} row sets overlap")

    cohorts: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for split_name, rows in partitions.items():
        grouped: dict[tuple[int, float], list[int]] = defaultdict(list)
        for row in rows.tolist():
            grouped[(int(turbine_count[row]), float(direction[row]))].append(int(row))
        eligible = 0
        eligible_rows = 0
        for (count, wd_deg), member_rows in sorted(grouped.items()):
            layout_ids = sorted({int(layout[row]) for row in member_rows})
            if len(layout_ids) < int(minimum_distinct_layouts):
                continue
            eligible += 1
            eligible_rows += len(member_rows)
            wd_label = int(wd_deg) if wd_deg.is_integer() else wd_deg
            cohorts.append(
                {
                    "cohort_id": f"{split_name}:n{count}:wd{wd_label}",
                    "split": split_name,
                    "n_turbines": count,
                    "wd_deg": wd_label,
                    "row_indices": sorted(member_rows),
                    "candidate_layout_indices": [int(layout[row]) for row in sorted(member_rows)],
                    "layout_indices": layout_ids,
                    "distinct_layout_count": len(layout_ids),
                }
            )
        counts[split_name] = {
            "eligible_cohorts": eligible,
            "candidate_rows": eligible_rows,
            "distinct_layouts": int(np.unique(layout[rows]).size),
        }

    return {
        "schema_version": 1,
        "purpose": "input-only finite-library cohort feasibility",
        "grouping_keys": ["split", "n_turbines", "wd_deg"],
        "input_columns_read": ["layout_index", "n_turbines", "wd_deg"],
        "outcome_columns_read": [],
        "outcome_used_for_cohort_construction": False,
        "minimum_distinct_layouts": int(minimum_distinct_layouts),
        "additional_site_constraints": "none specified by the available WindFarm data contract",
        "split_seed": split.metadata.get("seed", 42),
        "split_group_key": split.metadata.get("group_key", "layout_index"),
        "counts": counts,
        "cohorts": cohorts,
    }


def static_geometry_features(
    turbine_xy_D: Any,
    n_turbines: Any,
    wd_deg: Any,
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic scalar-head controls from active input geometry.

    No case/layout ID, descriptor archive column, solved value, or outcome is
    used. Coordinates are already in the documented downstream/crosswind
    frame and are not rotated again.
    """

    coordinates = np.asarray(turbine_xy_D, dtype=np.float64)
    counts = np.asarray(n_turbines, dtype=np.int64)
    directions = np.asarray(wd_deg, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError("turbine_xy_D must have shape [N,M,2]")
    if counts.shape != (coordinates.shape[0],) or directions.shape != counts.shape:
        raise ValueError("n_turbines and wd_deg must align with turbine_xy_D rows")
    if np.any(counts < 2) or np.any(counts > coordinates.shape[1]):
        raise ValueError("n_turbines is outside the active geometry capacity")
    names = [
        "n_turbines_fraction",
        "wd_270_onehot",
        "wd_285_onehot",
        "wd_300_onehot",
        "centroid_x_D",
        "centroid_y_D",
        "spread_x_D",
        "spread_y_D",
        "extent_x_D",
        "extent_y_D",
        "extent_aspect_xy",
        "occupancy_per_bbox_D2",
        "pair_distance_q10_D",
        "pair_distance_q25_D",
        "pair_distance_q50_D",
        "pair_distance_q75_D",
        "pair_distance_q90_D",
        "nearest_distance_min_D",
        "nearest_distance_mean_D",
        "nearest_distance_std_D",
        "nearest_distance_q10_D",
        "nearest_distance_q90_D",
        "cross_covariance_correlation",
    ]
    matrix = np.empty((coordinates.shape[0], len(names)), dtype=np.float64)
    for row, count_value in enumerate(counts.tolist()):
        points = coordinates[row, :count_value]
        if np.any(~np.isfinite(points)):
            raise ValueError(f"active turbine coordinates for row {row} are non-finite")
        x_extent, y_extent = np.ptp(points, axis=0)
        if x_extent <= 0.0 or y_extent <= 0.0:
            raise ValueError(f"active turbine geometry for row {row} has a zero bounding-box extent")
        offsets = points[:, None, :] - points[None, :, :]
        distances = np.sqrt(np.sum(offsets * offsets, axis=-1))
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        pairwise = distances[np.triu_indices(count_value, k=1)]
        covariance = np.cov(points.T, ddof=0)
        denominator = np.sqrt(max(float(covariance[0, 0] * covariance[1, 1]), 0.0))
        correlation = float(covariance[0, 1] / denominator) if denominator > 1.0e-12 else 0.0
        direction_onehot = [float(np.isclose(directions[row], item)) for item in WIND_DIRECTIONS_DEG]
        if sum(direction_onehot) != 1.0:
            raise ValueError(f"wd_deg row {row} is not one of the documented direction categories")
        matrix[row] = np.asarray(
            [
                count_value / 30.0,
                *direction_onehot,
                *points.mean(axis=0),
                *points.std(axis=0),
                x_extent,
                y_extent,
                x_extent / y_extent,
                count_value / (x_extent * y_extent),
                *np.quantile(pairwise, [0.10, 0.25, 0.50, 0.75, 0.90]),
                float(nearest.min()),
                float(nearest.mean()),
                float(nearest.std()),
                *np.quantile(nearest, [0.10, 0.90]),
                correlation,
            ],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("static WindFarm geometry features contain non-finite values")
    return matrix.astype(np.float32), names


def fit_grouped_ridge_head(
    features: Any,
    target: Any,
    layout_groups: Any,
    score_features: Any,
    *,
    device: Any,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    folds: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a standardized ridge scalar head with layout-group CV on train rows."""

    import torch

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    groups = np.asarray(layout_groups)
    x_score = np.asarray(score_features, dtype=np.float64)
    if x.ndim != 2 or x_score.ndim != 2 or x.shape[1] != x_score.shape[1]:
        raise ValueError("ridge feature matrices must be two-dimensional with matching columns")
    if y.shape != (x.shape[0],) or groups.shape != y.shape:
        raise ValueError("ridge target and layout groups must align with training rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(x_score)) or not np.all(np.isfinite(y)):
        raise ValueError("ridge inputs and targets must be finite")
    unique_groups = np.unique(groups)
    alpha_values = tuple(float(value) for value in alphas)
    if len(unique_groups) < int(folds) or int(folds) < 2:
        raise ValueError("grouped ridge CV requires at least one layout per fold and at least two folds")
    if not alpha_values or any(not np.isfinite(value) or value <= 0.0 for value in alpha_values):
        raise ValueError("ridge alphas must be finite positive values")
    target_device = torch.device(device)

    def _tensor(value: np.ndarray):
        return torch.as_tensor(value, dtype=torch.float64, device=target_device)

    def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alpha: float) -> np.ndarray:
        train_t = _tensor(x_train)
        target_t = _tensor(y_train)
        eval_t = _tensor(x_eval)
        mean = train_t.mean(dim=0)
        scale = train_t.std(dim=0, unbiased=False).clamp_min(1.0e-8)
        centered = (train_t - mean) / scale
        centered_eval = (eval_t - mean) / scale
        target_mean = target_t.mean()
        target_centered = target_t - target_mean
        identity = torch.eye(centered.shape[1], dtype=centered.dtype, device=target_device)
        coefficient = torch.linalg.solve(centered.T @ centered + alpha * identity, centered.T @ target_centered)
        result = centered_eval @ coefficient + target_mean
        return result.detach().cpu().numpy()

    rng = np.random.default_rng(int(seed))
    shuffled = unique_groups[rng.permutation(unique_groups.size)]
    held_groups = np.array_split(shuffled, int(folds))
    cv_mse: dict[str, float] = {}
    for alpha in alpha_values:
        fold_sse = 0.0
        fold_rows = 0
        for group_fold in held_groups:
            held = np.isin(groups, group_fold)
            if not np.any(held) or np.all(held):
                raise ValueError("grouped ridge CV generated an empty train or validation fold")
            prediction = _fit_predict(x[~held], y[~held], x[held], alpha)
            fold_sse += float(np.square(prediction - y[held]).sum())
            fold_rows += int(held.sum())
        cv_mse[f"{alpha:g}"] = fold_sse / max(fold_rows, 1)
    selected_alpha = min(alpha_values, key=lambda value: (cv_mse[f"{value:g}"], value))
    scored = _fit_predict(x, y, x_score, selected_alpha)
    return scored.astype(np.float32), {
        "method": "standardized ridge regression",
        "selected_alpha": selected_alpha,
        "candidate_alphas": list(alpha_values),
        "group_cv_folds": int(folds),
        "group_cv_group_key": "layout_index",
        "group_cv_seed": int(seed),
        "group_cv_mse_pct_squared": cv_mse,
        "selected_group_cv_rmse_pct": float(np.sqrt(cv_mse[f"{selected_alpha:g}"])),
        "training_rows": int(x.shape[0]),
        "training_layout_groups": int(unique_groups.size),
        "feature_count": int(x.shape[1]),
        "device": str(target_device),
    }


def regression_metrics(target: Any, prediction: Any) -> dict[str, float | None]:
    """Return empirical scalar prediction metrics in the target's native units."""

    truth = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 1 or pred.shape != truth.shape or truth.size == 0:
        raise ValueError("regression metrics require aligned non-empty one-dimensional arrays")
    if np.any(~np.isfinite(truth)) or np.any(~np.isfinite(pred)):
        raise ValueError("regression metrics require finite truth and prediction values")
    error = pred - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2 = None if denominator <= 0.0 else float(1.0 - np.sum(error**2) / denominator)
    prediction_variance = float(np.sum((pred - pred.mean()) ** 2))
    calibration_slope = (
        None
        if prediction_variance <= 0.0
        else float(np.sum((pred - pred.mean()) * (truth - truth.mean())) / prediction_variance)
    )
    return {
        "rows": int(truth.size),
        "rmse_pct": float(np.sqrt(np.mean(error**2))),
        "mae_pct": float(np.mean(np.abs(error))),
        "mean_prediction_minus_target_pct_points": float(np.mean(error)),
        "diagnostic_target_on_prediction_slope": calibration_slope,
        "r2": r2,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    first = 0
    while first < values.size:
        end = first + 1
        while end < values.size and sorted_values[end] == sorted_values[first]:
            end += 1
        ranks[order[first:end]] = 0.5 * (first + end - 1)
        first = end
    return ranks


def spearman_rank_correlation(target: Any, prediction: Any) -> float | None:
    """Compute Spearman correlation without an optional SciPy dependency."""

    truth = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 1 or pred.shape != truth.shape or truth.size < 2:
        return None
    truth_rank = _rankdata(truth)
    pred_rank = _rankdata(pred)
    if np.std(truth_rank) == 0.0 or np.std(pred_rank) == 0.0:
        return None
    return float(np.corrcoef(truth_rank, pred_rank)[0, 1])


def score_finite_library(
    cohort_manifest: Mapping[str, Any],
    predicted_target: Any,
    observed_target: Any,
    *,
    split_name: str = "validation",
) -> dict[str, Any]:
    """Score one predicted selection per frozen cohort against stored outcomes.

    Random controls and the oracle choose exactly one candidate per eligible
    cohort. They use the same observed candidate set after the prediction-only
    selection is frozen.
    """

    prediction = np.asarray(predicted_target, dtype=np.float64)
    observed = np.asarray(observed_target, dtype=np.float64)
    if prediction.ndim != 1 or observed.shape != prediction.shape:
        raise ValueError("predicted and observed scalar outcomes must be aligned one-dimensional arrays")
    eligible = [
        item for item in cohort_manifest.get("cohorts", [])
        if item.get("split") == str(split_name)
    ]
    if not eligible:
        raise ValueError(f"no eligible frozen W2 cohorts for split {split_name!r}")

    selection_records: list[dict[str, Any]] = []
    selected_truth: list[float] = []
    selected_rank: list[float] = []
    oracle_truth: list[float] = []
    candidates: list[np.ndarray] = []
    cohort_random_expected: list[float] = []
    for item in eligible:
        rows = np.asarray(item["row_indices"], dtype=np.int64)
        if rows.ndim != 1 or rows.size < 2 or np.any(rows < 0) or np.any(rows >= observed.size):
            raise ValueError(f"cohort {item.get('cohort_id')} has invalid candidate rows")
        if np.any(~np.isfinite(prediction[rows])) or np.any(~np.isfinite(observed[rows])):
            raise ValueError(f"cohort {item.get('cohort_id')} includes non-finite scores or outcomes")
        selected_row = min(rows.tolist(), key=lambda row: (prediction[row], int(row)))
        oracle_row = min(rows.tolist(), key=lambda row: (observed[row], int(row)))
        selected_value = float(observed[selected_row])
        target_values = observed[rows]
        random_expected = float(np.mean(target_values))
        better_count = int(np.sum(target_values < selected_value))
        tied_count = int(np.sum(target_values == selected_value))
        normalized_rank = (
            (better_count + 0.5 * max(tied_count - 1, 0)) / float(rows.size - 1)
        )
        selected_truth.append(selected_value)
        selected_rank.append(float(normalized_rank))
        oracle_truth.append(float(observed[oracle_row]))
        candidates.append(rows)
        cohort_random_expected.append(random_expected)
        selection_records.append(
            {
                "cohort_id": str(item["cohort_id"]),
                "n_turbines": int(item["n_turbines"]),
                "wd_deg": item["wd_deg"],
                "candidate_count": int(rows.size),
                "candidate_rows": rows.tolist(),
                "candidate_stored_outcomes": [
                    {
                        "row": int(row),
                        "layout_index": int(layout_index),
                        "predicted_wake_loss_pct": float(prediction[row]),
                        "observed_wake_loss_pct": float(observed[row]),
                    }
                    for row, layout_index in zip(rows.tolist(), item["candidate_layout_indices"])
                ],
                "selected_row": int(selected_row),
                "selected_layout_index": int(
                    item["candidate_layout_indices"][rows.tolist().index(selected_row)]
                ),
                "selected_predicted_wake_loss_pct": float(prediction[selected_row]),
                "selected_observed_wake_loss_pct": selected_value,
                "uniform_random_expected_wake_loss_pct": random_expected,
                "selected_minus_uniform_random_expected_pct_points": selected_value - random_expected,
                "oracle_row": int(oracle_row),
                "oracle_observed_wake_loss_pct": float(observed[oracle_row]),
                "selected_minus_oracle_pct_points": selected_value - float(observed[oracle_row]),
                "selected_normalized_observed_rank": float(normalized_rank),
            }
        )

    selected_mean = float(np.mean(selected_truth))
    random_mean = float(np.mean(cohort_random_expected))
    return {
        "split": str(split_name),
        "selection_direction": "minimize the empirical wake_loss_pct column",
        "target_units": "percent; regret and errors are percentage points",
        "cohort_count": len(eligible),
        "candidate_rows_scored": int(sum(rows.size for rows in candidates)),
        "selected_candidate_count": len(selection_records),
        "reference_outcome_rows_read_for_cohort_audit": int(sum(rows.size for rows in candidates)),
        "no_new_cfd_solves": True,
        "statistical_scope": {
            "benchmark_unit": "input-frozen cohort defined by exact turbine count and wind direction",
            "cohort_outcomes_are_paired": True,
            "cohort_candidates_can_share_layout_index_across_wind_directions": True,
            "aggregate_means_are_descriptive_not_independent_sample_inference": True,
            "sampling_confidence_intervals_reported": False,
        },
        "selected_mean_wake_loss_pct": selected_mean,
        "selected_median_wake_loss_pct": float(np.median(selected_truth)),
        "random_control": {
            "policy": "uniformly choose one stored row per same frozen cohort; exact expected value",
            "mean_of_equal_count_mean_outcomes_pct": random_mean,
            "selected_minus_random_mean_pct_points": selected_mean - random_mean,
            "not_a_sampling_confidence_interval": True,
        },
        "oracle_control": {
            "policy": "minimum stored wake_loss_pct within each cohort; retrospective upper-bound comparator",
            "mean_wake_loss_pct": float(np.mean(oracle_truth)),
            "mean_selected_regret_pct_points": float(np.mean(selected_truth) - np.mean(oracle_truth)),
        },
        "mean_selected_normalized_observed_rank": float(np.mean(selected_rank)),
        "cohort_selections": selection_records,
    }


__all__ = [
    "build_input_only_w2_cohorts",
    "fit_grouped_ridge_head",
    "regression_metrics",
    "score_finite_library",
    "spearman_rank_correlation",
    "static_geometry_features",
]
