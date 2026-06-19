"""CHANNELTHERMAL-SPECIFIC data package.

This package contains packed HDF5 dataset readers for ChannelThermal training
and evaluation. Outputs preserve the legacy batch keys.
"""

from .channelthermal_datasets import CHANNEL_ORDER, GlobalChannelThermalDataset

__all__ = ["CHANNEL_ORDER", "GlobalChannelThermalDataset"]
