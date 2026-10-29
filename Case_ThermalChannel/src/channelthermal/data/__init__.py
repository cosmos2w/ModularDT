"""Packed-HDF5 readers and normalization for ChannelThermal.

This package contains packed HDF5 dataset readers for ChannelThermal training
and evaluation. Outputs preserve the legacy batch keys.
"""

from .datasets import (
    CHANNEL_ORDER,
    GlobalChannelThermalDataset,
    GlobalModuleAlignmentDataset,
    H5Normalizer,
    LocalModuleDataset,
    fit_local_normalizer,
)
from .collation import ChannelThermalBatchCollator, ModuleCountBucketBatchSampler

__all__ = [
    "CHANNEL_ORDER",
    "ChannelThermalBatchCollator",
    "GlobalChannelThermalDataset",
    "GlobalModuleAlignmentDataset",
    "H5Normalizer",
    "LocalModuleDataset",
    "ModuleCountBucketBatchSampler",
    "fit_local_normalizer",
]
