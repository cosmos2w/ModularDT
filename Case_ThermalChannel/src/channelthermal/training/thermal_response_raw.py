"""Read R15-8 raw physical endpoints without touching held-out calibration."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..data.finite_responses import raw_endpoint_from_arrays


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
    return raw_endpoint_from_arrays(config, raw, raw_case_dir=path)
