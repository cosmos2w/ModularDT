"""ChannelThermal domain adapter and coupled HONF model.

This package adapts physical ChannelThermal inputs into reusable HONF tensors
and wraps the core output in the legacy evaluator contract.
"""

from .config import ChannelThermalHONFConfig
from .model import ChannelThermalHONFModel, PreparedChannelThermalCase

__all__ = ["ChannelThermalHONFConfig", "ChannelThermalHONFModel", "PreparedChannelThermalCase"]
