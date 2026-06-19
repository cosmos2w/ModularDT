"""CHANNELTHERMAL-SPECIFIC local surrogate package.

This package contains the copied Stage-A local module surrogate architecture.
The architecture and state-dict keys are intentionally preserved for strict
compatibility with previously trained local checkpoints.
"""

from .model_local import LocalModuleConfig, LocalModuleSurrogate, build_local_model_from_config

__all__ = ["LocalModuleConfig", "LocalModuleSurrogate", "build_local_model_from_config"]
