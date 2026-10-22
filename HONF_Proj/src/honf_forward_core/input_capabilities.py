"""Adapter-owned scene transforms consumed by reusable backbones."""
from dataclasses import dataclass
from typing import Callable, Mapping, Any


@dataclass(frozen=True)
class LocalInputCapability:
    contract: Mapping[str, Any]
    transform: Callable

    def __post_init__(self):
        if not callable(self.transform) or self.contract.get('schema') != 'local_records_context_v1':
            raise ValueError('A local-input capability needs its versioned contract and scene transform.')


_CAPABILITIES = {}


def register_local_input_capability(name, capability):
    if not isinstance(name, str) or not name or not isinstance(capability, LocalInputCapability):
        raise ValueError('Invalid named local-input capability.')
    existing = _CAPABILITIES.get(name)
    if existing is not None and existing != capability:
        raise ValueError(f'Local-input capability {name!r} is already registered differently.')
    _CAPABILITIES[name] = capability


def local_input_capability(name):
    try:
        return _CAPABILITIES[name]
    except KeyError as error:
        raise ValueError(f'No local-input capability registered for {name!r}.') from error
