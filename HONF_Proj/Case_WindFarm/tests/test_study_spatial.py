from pathlib import Path

import numpy as np

from windfarm.geometry import support_weights
from windfarm.io import WindFarmDataset
from windfarm.study_spatial import WeightedVelocityErrors, native_coordinates, native_plane


def test_weighted_velocity_reduction_reports_integral_denominator():
    errors = WeightedVelocityErrors()
    target = np.array([[1., 2., 0.], [3., 0., 4.]])
    prediction = target + np.array([[1., -2., 0.], [-1., 0., 2.]])
    errors.add(prediction, target, np.array([2., 3.]))
    result = errors.result([1., 2., 4.], 9.)
    np.testing.assert_allclose(result["squared_error_integral"], [5., 8., 12.])
    np.testing.assert_allclose(result["target_energy_integral"], [29., 8., 48.])
    np.testing.assert_allclose(result["vector_relative_l2"], np.sqrt(25. / 85.))
    np.testing.assert_allclose(result["standardized_mse"], (1. + .4 + .15) / 3.)


def test_actual_native_planes_agree_with_reader_and_support_measure():
    root = Path(__file__).resolve().parents[1] / "Dataset" / "links" / "wind_farm"
    dataset = WindFarmDataset(root)
    counts = np.prod(dataset.array("run_shape"), axis=1)
    order = np.argsort(counts, kind="stable")
    for row in (order[0], order[len(order) // 2], order[-1]):
        run = dataset.run(int(row))
        flat = np.array([0, run.nx - 1, run.nx, run.nx * run.ny, run.cell_count - 1])
        coords = native_coordinates(run, flat, 80.) * 80.
        np.testing.assert_allclose(coords[-1], [run.x_m[-1], run.y_m[-1], run.z_m[-1]], rtol=1e-7)
        np.testing.assert_allclose(coords[2], [run.x_m[0], run.y_m[1], run.z_m[0]], rtol=1e-7)
        for axis, location in (("z", 70.), ("y", 0.), ("x", 0.)):
            plane = native_plane(run, axis, location)
            index = plane["index"]
            direct = {"z": lambda: run.U_structured[index, :, :, :],
                      "y": lambda: run.U_structured[:, index, :, :],
                      "x": lambda: run.U_structured[:, :, index, :]}[axis]()
            np.testing.assert_array_equal(plane["target"], direct)
            assert np.isfinite(plane["coords_D"]).all()
        weights = support_weights(run.x_m, run.y_m, run.z_m)
        expected = (run.x_m[-1] - float(run.x_m[0])) * (run.y_m[-1] - float(run.y_m[0])) * (run.z_m[-1] - float(run.z_m[0]))
        np.testing.assert_allclose(np.prod([w.sum() for w in weights]), expected, rtol=1e-7)
