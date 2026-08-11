"""ChannelThermal Stage-A local surrogate package.

This package contains the copied Stage-A local module surrogate architecture.
The architecture and state-dict keys are intentionally preserved for strict
compatibility with previously trained local checkpoints.
"""

from .model import LocalModuleConfig, LocalModuleSurrogate

__all__ = ["LocalModuleConfig", "LocalModuleSurrogate"]
