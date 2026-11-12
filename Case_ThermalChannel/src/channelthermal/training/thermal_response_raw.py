"""Read R15-8 raw physical endpoints without touching held-out calibration."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_raw_endpoint(path):
    """Read the converged final frame, complete native receivers and P64 teacher.

    Values retain their saved FP32 precision. The outside stencil's geometric
    validity is separate from the preprocessing-derived h-valid mask.
    """
    path = Path(path)
    config = json.loads((path / 'case_config.json').read_text())
    with (path / 'frame_index.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or int(float(rows[-1]['converged'])) != 1:
        raise ValueError('A converged final physical frame is required')
    with np.load(path / 'scene' / rows[-1]['file'], allow_pickle=False) as frame:
        raw = {key: frame[key].copy() for key in frame.files}
    field = np.stack([raw[k] for k in ('u', 'v', 'p', 'omega', 'temperature')], axis=-1)
    response = raw['interface_response']
    names = [s.decode() if isinstance(s, bytes) else str(s) for s in raw['interface_feature_names']]
    indices = [names.index(s) for s in ('theta', 'normal_x', 'normal_y', 'T_outside', 'T_surface', 'q_normal')]
    theta, nxv, nyv, outside, surface, q = (response[..., i] for i in indices)
    difference = surface - outside
    denominator = np.where(np.abs(difference) < 1.e-3, np.where(difference < 0, -1.e-3, 1.e-3), difference).astype(np.float32)
    h = q / denominator
    h_valid = (np.abs(difference) >= 1.e-3) & np.isfinite(q) & np.isfinite(h) & (h >= 0) & (h <= 1.e4)
    teacher = np.stack((theta, nxv, nyv, outside, np.clip(np.nan_to_num(h, nan=0., posinf=1.e4, neginf=0.), 0., 1.e4)), -1).astype(np.float32)
    d = config['domain']; centers = np.asarray(config['layout']['centers'], dtype=np.float64)
    radius = float(d['module_radius']); delta = min(d['lx']/d['nx'], d['ly']/d['ny'], .15*radius)
    angle = np.arange(response.shape[1], dtype=np.float64) * 2*np.pi / response.shape[1]
    xy = centers[:, None, :] + (radius + delta) * np.stack([np.cos(angle), np.sin(angle)], -1)[None]
    outside_valid = ((xy[..., 0] >= .5*d['lx']/d['nx']) & (xy[..., 0] <= d['lx']-.5*d['lx']/d['nx']) &
                     (xy[..., 1] >= .5*d['ly']/d['ny']) & (xy[..., 1] <= d['ly']-.5*d['ly']/d['ny']))
    return {'raw_case_dir': str(path), 'config': config, 'field': field, 'fluid_mask': ~raw['module_mask'].astype(bool),
            'module_id_grid': raw['module_id'], 'material': raw['module_internal_temperature'],
            'material_mask': raw['module_internal_mask'].astype(bool), 'interface_target': np.stack((surface, q), -1),
            'teacher_ports': teacher, 'teacher_h_valid': h_valid, 'outside_valid': outside_valid}
