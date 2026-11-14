#!/usr/bin/env python3
"""Compare two existing Wind rows at two common wind-relative positions, without fitting.

The positions are selected from the overlap of saved coordinate ranges and
pressure is interpolated from each scene's own cell centres. The
observable is pressure at the last position minus pressure at the first, so
an arbitrary constant pressure offset cancels within each scene. This is a
wind-relative comparison, not an Earth-fixed intervention or a turbine match.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Case_WindFarm/src")]
from honf_runtime.state_response import NativePairScene, select_native_pairs, validate_common_coordinates
from windfarm.io import WindFarmDataset


def compare(dataset, first_index: int, second_index: int, split: str) -> dict:
    runs = [dataset.run(first_index), dataset.run(second_index)]
    scenes = [NativePairScene(str(run.case), split, int(run.layout_index), float(run.wind_direction_deg))
              for run in runs]
    pairs, unmatched = select_native_pairs(scenes, lambda x, y: abs(x - y), lambda x, y: True)
    if not pairs or unmatched:
        raise ValueError("Rows must have distinct directions and the same original layout in the declared split.")
    axes = [tuple(np.asarray(getattr(run, axis + '_m')) for axis in 'xyz') for run in runs]
    bounds = [(max(float(a.min()), float(b.min())), min(float(a.max()), float(b.max())))
              for a, b in zip(*axes)]
    if any(lower >= upper for lower, upper in bounds):
        raise ValueError("The saved coordinate ranges need nonempty overlap on every axis.")
    positions = np.asarray([[lower for lower, _ in bounds], [upper for _, upper in bounds]])
    validate_common_coordinates(positions, positions, [np.ones(2, dtype=bool)] * 2)
    pressure_differences = []
    for run, run_axes in zip(runs, axes):
        interpolator = RegularGridInterpolator(run_axes[::-1], run.p_structured, bounds_error=True)
        values = interpolator(positions[:, ::-1])
        if not np.isfinite(values).all():
            raise ValueError("Saved pressure is not finite at the selected positions.")
        pressure_differences.append(values[1] - values[0])
    return {"case_ids": [row.case_id for row in scenes], "split": split,
            "layout_index": int(runs[0].layout_index),
            "wind_direction_deg": [float(run.wind_direction_deg) for run in runs],
            "frame": "saved downstream/crosswind/vertical wind-relative frame",
            "positions_m": positions.tolist(), "extraction": "trilinear interpolation on each saved field",
            "observable": "pressure(last) - pressure(first)",
            "pressure_differences": pressure_differences,
            "change_in_pressure_difference": pressure_differences[1] - pressure_differences[0],
            "units": "source pressure units", "persistent_turbine_correspondence": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("first_row", type=int)
    parser.add_argument("second_row", type=int)
    parser.add_argument("--split", required=True, help="Declared split of both supplied rows")
    args = parser.parse_args()
    print(json.dumps(compare(WindFarmDataset(args.dataset), args.first_row, args.second_row, args.split), indent=2))


if __name__ == "__main__":
    main()
