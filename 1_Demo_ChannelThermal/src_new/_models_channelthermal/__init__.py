"""CHANNELTHERMAL-SPECIFIC model package.

This package adapts physical ChannelThermal inputs into reusable HONF tensors
and wraps the core output in the legacy evaluator contract.
"""

from .channelthermal_full_model import ChannelThermalHONFConfig, ChannelThermalHONFModel

__all__ = ["ChannelThermalHONFConfig", "ChannelThermalHONFModel"]
