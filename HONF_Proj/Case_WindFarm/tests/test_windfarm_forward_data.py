from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from windfarm.data import case_batch, collate_windfarm
from windfarm.geometry import (
    cell_centre_weights,
    environment_representation,
    gather_native_sample,
    support_features,
    support_geometry,
)
from windfarm.normalization import fit_velocity_statistics


def _run(row: int = 0) -> SimpleNamespace:
    x = np.asarray([-8.0, 0.0, 12.0], dtype=np.float32)
    y = np.asarray([-4.0, 3.0], dtype=np.float32)
    z = np.asarray([20.0, 70.0, 130.0], dtype=np.float32)
    shape = (x.size, y.size, z.size)
    values = np.empty((x.size * y.size * z.size, 3), dtype=np.float32)
    for flat in range(values.shape[0]):
        ix = flat % x.size
        iy = (flat // x.size) % y.size
        iz = flat // (x.size * y.size)
        values[flat] = (row + ix, 10.0 * iy + iz, 100.0 * iz + iy)
    return SimpleNamespace(x_m=x, y_m=y, z_m=z, shape_nxyz=shape, U=values)


def test_centre_quadrature_and_geometry_tokens_use_all_three_axes() -> None:
    x = np.asarray([-8.0, 0.0, 12.0])
    y = np.asarray([-4.0, 3.0])
    z = np.asarray([20.0, 70.0, 130.0])
    assert np.isclose(cell_centre_weights(x).sum(), 20.0)
    assert np.isclose(cell_centre_weights(y).sum(), 7.0)
    assert np.isclose(cell_centre_weights(z).sum(), 110.0)
    support = support_geometry(x, y, z)
    representation = environment_representation(support)
    assert representation.coords_D.shape == (512, 3)
    assert representation.features.shape == (512, 7)
    assert np.isclose(representation.weights_D3.sum(), support.volume_D3)
    queries = np.asarray([[0.0, 0.0, 20.0], [0.0, 0.0, 130.0]], dtype=np.float32) / 80.0
    features = support_features(queries, support)
    assert not np.allclose(features[0], features[1])
    assert np.any(np.abs(features[0, :3] - features[1, :3]) > 0.0)


def test_native_indices_gather_same_c_order_target() -> None:
    run = _run()
    sample = gather_native_sample(
        run,
        128,
        np.random.default_rng(42),
        mode="volume",
    )
    expected = run.U[sample.flat_indices].copy()
    assert np.array_equal(sample.velocity_mps, expected)
    assert np.all(sample.coords_D[:, 0] >= run.x_m.min() / 80.0)
    assert np.all(sample.coords_D[:, 2] <= run.z_m.max() / 80.0)


def test_normalization_is_fitted_on_rows_supplied_only() -> None:
    class FakeDataset:
        def run(self, row: int) -> SimpleNamespace:
            return _run(row)

    normalizer, profile = fit_velocity_statistics(
        FakeDataset(),
        [0, 1],
        samples_per_row=64,
        seed=42,
        profile_bins=8,
    )
    assert normalizer.source_rows == 2
    assert normalizer.sample_count_per_row == 64
    assert profile.counts.shape == (8,)
    assert np.all(profile.counts >= 0)
    values = np.asarray([[9.0, 0.0, 0.0]], dtype=np.float32)
    assert normalizer.denormalize(normalizer.normalize(values)).shape == (1, 3)


def test_collation_pads_only_to_batch_maximum() -> None:
    # The production Dataset still computes support features, so validate the
    # dynamic pad directly with two small sample dictionaries here.  Mounted
    # native-array tests exercise the real reader and native metrics separately.
    samples = []
    for count, name in ((6, "case-0"), (9, "case-1")):
        samples.append(
            {
                "module_centers": np.zeros((count, 3), dtype=np.float32),
                "module_present": np.ones(count, dtype=np.float32),
                "module_features": np.ones((count, 2), dtype=np.float32),
                "global_context": np.zeros(11, dtype=np.float32),
                "env_coords": np.zeros((512, 3), dtype=np.float32),
                "env_features": np.zeros((512, 7), dtype=np.float32),
                "env_weights": np.ones(512, dtype=np.float32),
                "query_xy": np.zeros((4, 3), dtype=np.float32),
                "query_features": np.zeros((4, 7), dtype=np.float32),
                "query_time": None,
                "target_field": np.zeros((4, 3), dtype=np.float32),
                "case_name": name,
                "metadata": {},
            }
        )
    batch = collate_windfarm(samples)
    assert batch["module_centers"].shape == (2, 9, 3)
    assert np.all(batch["module_present"][0, 6:] == 0.0)
    assert np.all(np.isfinite(batch["module_centers"]))


def test_case_batch_has_geometry_only_inference_inputs() -> None:
    from windfarm.data import NativeCase

    run = _run()
    support = support_geometry(run.x_m, run.y_m, run.z_m)
    environment = environment_representation(support)
    centers = np.asarray([[0.0, 0.0, 0.875]], dtype=np.float32)
    case = NativeCase(
        index=0,
        case="case-0",
        layout="layout-0",
        layout_index=0,
        wind_direction_deg=270.0,
        n_turbines=1,
        run=run,
        support=support,
        environment=environment,
        module_centers=centers,
        module_present=np.ones(1, dtype=np.float32),
        module_features=np.asarray([[0.5, 0.875]], dtype=np.float32),
        global_context=np.zeros(11, dtype=np.float32),
    )
    batch = case_batch(case, np.asarray([[0.0, 0.0, 0.875]], dtype=np.float32))
    assert tuple(batch.query_xy.shape) == (1, 1, 3)
    assert tuple(batch.env_coords.shape) == (1, 512, 3)
    assert tuple(batch.env_weights.shape) == (1, 512)
    assert batch.target_field is None
