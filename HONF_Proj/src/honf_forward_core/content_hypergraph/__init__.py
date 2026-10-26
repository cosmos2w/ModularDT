"""Versioned content-hypergraph model exports.

The curated bundle exposes the verified V2 model and loads its contracts on
demand so importing one API does not load unrelated legacy implementations.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ContentHypergraphCoreV2": "model_v2",
    "PreparedAffineQueryV2": "model_v2",
    "CoreOutput": "contracts",
    "PreparedOrganization": "contracts",
    "PreparedOrganizationV2": "contracts",
    "QueryBatch": "contracts",
    "SceneBatch": "contracts",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
