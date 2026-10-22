"""Compatibility resolution for saved dataset-named unified constructors.

Actual transforms and feature contracts belong to their case adapters. New
adapters may register a capability directly without extending this shim.
"""
from importlib import import_module
from honf_forward_core.input_capabilities import (
    LocalInputCapability, local_input_capability, register_local_input_capability,
)

_LEGACY_ADAPTERS = {'thermal': 'channelthermal.local_records',
                    'wind': 'windfarm.local_records'}


def resolve_local_input_capability(name):
    try:
        return local_input_capability(name)
    except ValueError:
        module_name = _LEGACY_ADAPTERS.get(name)
        if module_name is None:
            raise ValueError(f'No case adapter registered for {name!r}.') from None
        adapter = import_module(module_name)
        capability = LocalInputCapability(adapter.feature_contract(), adapter.local_records_scene)
        register_local_input_capability(name, capability)
        return capability
