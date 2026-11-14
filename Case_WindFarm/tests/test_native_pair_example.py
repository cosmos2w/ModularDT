from types import SimpleNamespace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from honf_runtime.state_response import validate_common_coordinates

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_native_pair.py"
spec = importlib.util.spec_from_file_location("wind_native_pair_example", SCRIPT)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_ragged_3d_pressure_difference_and_missing_member_mapping():
    def run(case, direction, axes, offset):
        x, y, z = axes
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        pressure = (xx + 2 * yy + 3 * zz + offset).astype(np.float32)
        return SimpleNamespace(case=case, layout_index=7, wind_direction_deg=direction,
                               x_m=x, y_m=y, z_m=z, p_structured=pressure)

    rows = [run("first", 270, ([0., 1., 3.], [0., 2.], [0., 1.]), 5),
            run("second", 285, ([0., 2.], [0., 1., 2.], [0., .5, 1.]), 9)]
    dataset = SimpleNamespace(run=lambda index: rows[index])
    result = example.compare(dataset, 0, 1, "train")
    assert result["positions_m"] == [[0., 0., 0.], [2., 2., 1.]]
    assert result["pressure_differences"] == pytest.approx([9., 9.])
    assert result["change_in_pressure_difference"] == pytest.approx(0.)
    assert result["persistent_turbine_correspondence"] is False

    positions = np.asarray(result["positions_m"])
    mask = validate_common_coordinates(positions, positions.copy(), [[True, True], [True, False]])
    measures = np.asarray([2., 3.])
    assert measures[mask].sum() == 2.
    with pytest.raises(ValueError, match="coordinates"):
        validate_common_coordinates(positions, positions + 1, [[True, True]])
