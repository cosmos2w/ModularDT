from __future__ import annotations

from pathlib import Path

import pytest

from windfarm.forward_analysis_figures import (
    ENDPOINT_ORDER,
    MODEL_ORDER,
    MetricArtifact,
    bootstrap_layout_direction_means,
    plot_accuracy_overview,
    plot_learning_curves,
    plot_paired_deltas,
    plot_paired_rmse,
    plot_strata,
    validate_endpoint_artifacts,
)


def _cases(offset: float = 0.0) -> list[dict[str, object]]:
    rows = []
    source_index = 0
    for layout in range(30):
        for direction in (270.0, 285.0, 300.0):
            rows.append(
                {
                    "case": f"gen_{layout:04d}_wd{int(direction)}",
                    "source_index": source_index,
                    "layout_index": layout,
                    "wind_direction_deg": direction,
                    "volume_rmse_mps": 0.02 + offset + layout * 1e-4 + (direction - 270.0) * 1e-5,
                }
            )
            source_index += 1
    return rows


def _artifact(model: str, endpoint: str, *, offset: float = 0.0) -> MetricArtifact:
    cases = _cases(offset)
    if endpoint == "epoch2500_validation":
        role = {
            "checkpoint_role": "explicit_checkpoint",
            "checkpoint_epoch": 2500,
            "checkpoint_selector": "epoch_2500_model.pt",
            "validation_only_selection": False,
            "reserved_test_used_for_selection": False,
            "selection_split": None,
        }
    else:
        role = {
            "checkpoint_role": "best_validation_checkpoint",
            "checkpoint_epoch": 20,
            "checkpoint_selector": "best_field.pt",
            "validation_only_selection": True,
            "reserved_test_used_for_selection": False,
            "selection_split": "validation",
        }
    payload = {
        "workflow": "forward",
        "split": "test" if endpoint == "best_test" else "validation",
        "q_volume": 32768,
        "q_band": 8192,
        "sample_seed": 42,
        "rows": len(cases),
        "checkpoint_epoch": role.pop("checkpoint_epoch"),
        "cases": cases,
        "selection_metric_provenance": role,
        "equal_case": {
            "mean_volume_rmse_mps": 0.02 + offset,
            "mean_volume_standardized_mse": 0.004 + offset,
        },
        "pooled": {"volume": {"volume_vector_relative_l2": 0.002 + offset}},
    }
    return MetricArtifact(model, endpoint, Path(f"{model}_{endpoint}.json"), payload)


def _artifacts() -> dict[tuple[str, str], MetricArtifact]:
    return {
        (model, endpoint): _artifact(model, endpoint, offset=(0.001 if model == "dense" else 0.0))
        for model in MODEL_ORDER
        for endpoint in ENDPOINT_ORDER
    }


def test_endpoint_validation_requires_validation_only_best_checkpoint() -> None:
    artifacts = _artifacts()
    audit = validate_endpoint_artifacts(artifacts)
    assert audit["paired_claims_allowed"] is True
    assert audit["rows_by_endpoint"] == {endpoint: 90 for endpoint in ENDPOINT_ORDER}

    bad = _artifacts()
    bad_payload = bad[("classic", "best_test")].payload
    bad_payload["selection_metric_provenance"]["validation_only_selection"] = False
    with pytest.raises(ValueError, match="validation-only"):
        validate_endpoint_artifacts(bad)


def test_layout_bootstrap_is_deterministic_and_uses_three_directions() -> None:
    rows = []
    for layout in range(30):
        for direction in (270.0, 285.0, 300.0):
            rows.append({"layout_index": layout, "dense_minus_classic_volume_rmse_mps": -0.001 * (layout + 1)})
    first = bootstrap_layout_direction_means(rows, replicates=200, seed=17)
    second = bootstrap_layout_direction_means(rows, replicates=200, seed=17)
    assert first == second
    assert first["n_layouts"] == 30
    assert first["directions_per_layout"] == 3
    assert "not population-generalization" in first["descriptive_scope"]


@pytest.mark.parametrize("plotter", [plot_accuracy_overview, plot_paired_rmse])
def test_core_report_figures_render(tmp_path: Path, plotter) -> None:
    pytest.importorskip("matplotlib")
    if plotter is plot_accuracy_overview:
        rows = []
        for endpoint in ENDPOINT_ORDER:
            for model in MODEL_ORDER:
                rows.append(
                    {
                        "endpoint": endpoint,
                        "model": model,
                        "equal_case_physical_rmse_mps": .02 if model == "classic" else .01,
                        "equal_case_standardized_mse": .005 if model == "classic" else .004,
                        "pooled_vector_relative_l2": .003 if model == "classic" else .002,
                    }
                )
        output = plotter(rows, tmp_path / "accuracy.png")
    else:
        rows = []
        for endpoint in ENDPOINT_ORDER:
            for case in range(90):
                classic = .02 + case * 1e-5
                dense = .01 + case * 1e-5
                rows.append(
                    {
                        "endpoint": endpoint,
                        "classic_volume_rmse_mps": classic,
                        "dense_volume_rmse_mps": dense,
                        "dense_minus_classic_volume_rmse_mps": dense - classic,
                    }
                )
        output = plotter(rows, tmp_path / "paired.png")
    assert output.is_file() and output.stat().st_size > 0


def test_learning_delta_and_strata_figures_render(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    histories = {
        model: [
            {"epoch": 1.0, "val_volume_mse": .8, "val_band_mse": .9},
            {"epoch": 10.0, "val_volume_mse": .2, "val_band_mse": .3},
            {"epoch": 2500.0, "val_volume_mse": .01, "val_band_mse": .02},
        ]
        for model in MODEL_ORDER
    }
    for rows in histories.values():
        for row in rows:
            row.update({"loss_total": .1, "train_volume_mse": .1, "train_band_mse": .1})
    learning = plot_learning_curves(histories, [2500], {model: 10 for model in MODEL_ORDER}, tmp_path / "learning.png")
    assert learning.is_file() and learning.stat().st_size > 0

    paired = [
        {
            "endpoint": endpoint,
            "layout_index": layout,
            "classic_volume_rmse_mps": .02,
            "dense_volume_rmse_mps": .01,
            "dense_minus_classic_volume_rmse_mps": -.01,
        }
        for endpoint in ENDPOINT_ORDER
        for layout in range(30)
        for _direction in range(3)
    ]
    bootstrap = [
        {
            "endpoint": endpoint,
            "estimate_mean_dense_minus_classic_volume_rmse_mps": -.01,
            "bootstrap_lower_2_5_percent_mps": -.011,
            "bootstrap_upper_97_5_percent_mps": -.009,
        }
        for endpoint in ENDPOINT_ORDER
    ]
    delta = plot_paired_deltas(paired, bootstrap, tmp_path / "delta.png")
    assert delta.is_file() and delta.stat().st_size > 0

    strata = []
    for endpoint in ENDPOINT_ORDER:
        for family in ("M_bin", "aspect_xy_bin", "direction", "volume_quartile", "mean_nn_D_bin", "min_sep_D_bin", "nn_dispersion_bin"):
            strata.append({"endpoint": endpoint, "stratum": f"{family}=1", "dense_reduction_percent_volume_rmse": 40.0})
    strata_output = plot_strata(strata, tmp_path / "strata.png")
    assert strata_output.is_file() and strata_output.stat().st_size > 0
