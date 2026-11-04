"""Inspection and native variable-domain data helpers for WindFarm."""

from .data import NativeCase, WindFarmNativeDataset, WindFarmNativeView, case_batch, collate_windfarm
from .geometry import (
    ENV_TOKEN_SHAPE,
    POSITIONAL_SCALE_D,
    cell_centre_weights,
    environment_representation,
    support_features,
    support_geometry,
)
from .io import WindFarmDataset, discover_volume_root
from .normalization import VelocityNormalizer, VerticalProfileBaseline
from .profile import profile_dataset, write_profile_outputs
from .splits import make_group_split
from .visualize import render_case_showcase, select_representative_indices

__all__ = [
    "ENV_TOKEN_SHAPE",
    "POSITIONAL_SCALE_D",
    "NativeCase",
    "VelocityNormalizer",
    "VerticalProfileBaseline",
    "WindFarmDataset",
    "WindRelationAnchoredAdapter",
    "attach_wind_e_cover",
    "WindFarmNativeDataset",
    "WindFarmNativeView",
    "case_batch",
    "cell_centre_weights",
    "collate_windfarm",
    "discover_volume_root",
    "environment_representation",
    "make_group_split",
    "profile_dataset",
    "render_case_showcase",
    "select_representative_indices",
    "support_features",
    "support_geometry",
    "write_profile_outputs",
]


def __getattr__(name):
    # The preprocessing package can be installed without Torch or the core.
    if name in {"WindRelationAnchoredAdapter", "attach_wind_e_cover"}:
        from . import relation_anchored
        return getattr(relation_anchored, name)
    raise AttributeError(name)
