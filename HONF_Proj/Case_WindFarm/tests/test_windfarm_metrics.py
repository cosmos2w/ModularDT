"""Focused physical and native-volume metric checks for the WindFarm study."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from windfarm.data import WindFarmNativeView
from windfarm.geometry import support_weights
from windfarm.normalization import VelocityNormalizer
from windfarm.study_spatial import stream_native_errors
from windfarm.workflows.evaluate_forward import _case_metrics


def _normalizer() -> VelocityNormalizer:
    return VelocityNormalizer(
        mean=np.zeros(3, dtype=np.float64),
        std=np.asarray((2.0, 3.0, 4.0), dtype=np.float64),
        safe_std=np.asarray((2.0, 3.0, 4.0), dtype=np.float64),
        u_ref_mps=9.0,
    )


def test_endpoint_metrics_keep_physical_and_standardized_units() -> None:
    normalizer = _normalizer()
    target_physical = np.asarray(
        [[9.0, 0.0, 0.0], [18.0, 3.0, 4.0], [0.0, -3.0, -4.0],
         [9.0, 6.0, 8.0]],
        dtype=np.float32,
    )
    target_standardized = normalizer.normalize(target_physical)
    prediction_physical = target_physical.copy()
    prediction_physical[:, 0] += np.asarray((1.0, -2.0, 3.0, -4.0), dtype=np.float32)
    prediction_standardized = normalizer.normalize(prediction_physical)
    raw_batch = {
        "target_field": target_standardized,
        "query_xy": np.asarray(
            [[0.0, 0.0, 0.875], [1.0, 0.0, 0.875],
             [0.0, 1.0, 0.875], [-1.0, 0.0, 0.875]],
            dtype=np.float32,
        ),
        "query_measure_m3": np.ones(4, dtype=np.float32),
        "module_centers": np.asarray([[0.0, 0.0, 0.875]], dtype=np.float32),
        "metadata": {
            "source_index": 0,
            "case": "metric-case",
            "layout_index": 0,
            "wind_direction_deg": 270.0,
            "n_turbines": 1,
            "shape_nxyz": [4, 2, 2],
            "support_volume_m3": 80.0**3,
            "support_volume_D3": 1.0,
            "support_extent_D": np.asarray((2.0, 2.0, 6.25), dtype=np.float32),
        },
    }
    result = _case_metrics(
        raw_batch=raw_batch,
        prediction_physical=prediction_physical,
        prediction_standardized=prediction_standardized,
        normalizer=normalizer,
        baseline=None,
        volume_queries=3,
        module_present=np.asarray([1.0], dtype=np.float32),
    )

    np.testing.assert_allclose(result["volume_channel_rmse_mps"][0], np.sqrt(14.0 / 3.0), rtol=1e-6)
    np.testing.assert_allclose(
        result["volume_channel_rmse_over_u_ref"][0], np.sqrt(14.0 / 3.0) / 9.0, rtol=1e-6
    )
    assert result["volume_count"] == 3
    assert result["hub_band_count"] == 1
    assert result["downstream_envelope_count"] == 1
    assert result["support_aspect_xy"] == 1.0


def test_native_identity_stream_has_zero_error_and_exact_support_measure() -> None:
    """Exercise the streaming reduction against an actual mmap-backed run."""

    root = Path(__file__).resolve().parents[1] / "Dataset" / "links" / "wind_farm"
    view = WindFarmNativeView(root)
    row = min(
        range(view.n_cases),
        key=lambda index: int(np.prod(view.run(index).shape_nxyz)),
    )
    case = view.run(row)
    run = case.run

    def _nearest(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
        right = np.searchsorted(axis, values, side="left")
        right = np.clip(right, 1, len(axis) - 1)
        left = right - 1
        return left + (np.abs(values - axis[left]) > np.abs(axis[right] - values))

    def identity(coords_D: np.ndarray) -> np.ndarray:
        ix = _nearest(run.x_m, coords_D[:, 0] * 80.0)
        iy = _nearest(run.y_m, coords_D[:, 1] * 80.0)
        iz = _nearest(run.z_m, coords_D[:, 2] * 80.0)
        flat = ix + run.nx * (iy + run.ny * iz)
        return np.asarray(run.U[flat], dtype=np.float64)

    def zero_background(z_D: np.ndarray) -> np.ndarray:
        return np.zeros((len(z_D), 3), dtype=np.float64)

    axis_weights = tuple(axis / 80.0 for axis in support_weights(run.x_m, run.y_m, run.z_m))
    result = stream_native_errors(
        run,
        identity,
        zero_background,
        axis_weights,
        case.module_centers,
        physical_std=np.ones(3, dtype=np.float64),
        chunk_size=65536,
    )

    assert result["volume"]["count"] == run.cell_count
    np.testing.assert_allclose(result["volume"]["quadrature_volume_D3"], case.support.volume_D3, rtol=1e-8)
    np.testing.assert_allclose(result["volume"]["rmse_m_s"], np.zeros(3), atol=1e-12)
    assert result["band"]["count"] > 0
    assert result["downstream"]["count"] > 0


def test_endpoint_metric_input_is_rejecting_non_case_arrays() -> None:
    """Metric code must reject malformed target/prediction arrays honestly."""

    normalizer = _normalizer()
    raw_batch = {
        "target_field": np.zeros((3, 3), dtype=np.float32),
        "query_xy": np.zeros((3, 3), dtype=np.float32),
        "module_centers": np.zeros((1, 3), dtype=np.float32),
        "metadata": {
            "source_index": 0,
            "case": "metric-case",
            "layout_index": 0,
            "wind_direction_deg": 270.0,
            "shape_nxyz": [1, 1, 3],
            "support_volume_m3": 80.0**3,
            "support_volume_D3": 1.0,
        },
    }
    try:
        _case_metrics(
            raw_batch=raw_batch,
            prediction_physical=np.zeros((2, 3), dtype=np.float32),
            prediction_standardized=np.zeros((2, 3), dtype=np.float32),
            normalizer=normalizer,
            baseline=None,
            volume_queries=2,
            module_present=np.asarray([1.0], dtype=np.float32),
        )
    except ValueError as exc:
        assert "expect [Q,3]" in str(exc)
    else:  # pragma: no cover - assertion makes this an ordinary computation test
        raise AssertionError("malformed metric arrays were accepted")


__all__ = [
    "test_endpoint_metric_input_is_rejecting_non_case_arrays",
    "test_endpoint_metrics_keep_physical_and_standardized_units",
    "test_native_identity_stream_has_zero_error_and_exact_support_measure",
]
