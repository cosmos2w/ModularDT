"""Saved Wind velocity references for a common wind-relative observation set."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .splits import make_group_split


def saved_velocity_pair(dataset, first_index, second_index, positions_m):
    """Extract TRAIN velocity references without copying the source dataset.

    Positions and U components use the saved downstream/crosswind/vertical
    frame; direction remains a changed control, not an Earth-fixed transform.
    The equal-position observation measure returned here is an example only,
    not the five-role Wind training objective.
    """
    groups = np.asarray(dataset._arrays['layout_index'])
    split = make_group_split(groups, seed=42)
    train = set(map(int, split.train))
    if first_index == second_index or first_index not in train or second_index not in train:
        raise ValueError('Two distinct rows must belong to the actual grouped TRAIN split.')
    runs = (dataset.run(first_index), dataset.run(second_index))
    if runs[0].layout_index != runs[1].layout_index or runs[0].wind_direction_deg == runs[1].wind_direction_deg:
        raise ValueError('Rows must share a layout and have distinct direction controls.')
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions) or not np.isfinite(positions).all():
        raise ValueError('Positions must be a finite nonempty [receiver, xyz] array in metres.')
    values = []
    for run in runs:
        interpolation = RegularGridInterpolator((run.z_m, run.y_m, run.x_m), run.U_structured,
                                                bounds_error=False, fill_value=np.nan)
        values.append(interpolation(positions[:, ::-1]))
    references = np.stack(values)
    mask = np.isfinite(references).all(axis=0)
    mass = np.zeros_like(mask, dtype=np.float64)
    counts = mask.sum(axis=0)
    for channel in range(3):
        if counts[channel]:
            mass[mask[:, channel], channel] = 1.0 / counts[channel]
    return {'references': references, 'mask': mask, 'masses': mass,
            'case_ids': tuple(run.case for run in runs),
            'layout_index': runs[0].layout_index,
            'direction_deg': tuple(run.wind_direction_deg for run in runs),
            'frame': 'saved wind-relative downstream/crosswind/vertical',
            'channels': ('Ux', 'Uy', 'Uz'), 'units': 'm/s',
            'measure': 'equal mass per valid supplied position, separately per channel'}
