"""Inspection and visualisation helpers for the wind-farm CFD dataset.

This package deliberately stops at dataset profiling, leakage-safe split
metadata, and exploratory figures.  It does not contain a model or a
training workflow.
"""

from .io import WindFarmDataset, discover_volume_root
from .profile import profile_dataset, write_profile_outputs
from .splits import make_group_split
from .visualize import render_case_showcase, select_representative_indices

__all__ = [
    "WindFarmDataset",
    "discover_volume_root",
    "make_group_split",
    "profile_dataset",
    "render_case_showcase",
    "select_representative_indices",
    "write_profile_outputs",
]
