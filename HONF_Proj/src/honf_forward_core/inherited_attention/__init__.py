"""Attention-inherited gathering primitives for the HONF forward core."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "InheritedQueryBatch": "contracts",
    "PreparedAnchorValues": "contracts",
    "PreparedSourceIndexedResponse": "contracts",
    "ReceiverCover": "contracts",
    "RouteScoreSnapshot": "contracts",
    "TypedAssociationAtlas": "contracts",
    "TypedTokenValues": "contracts",
    "with_receiver_cover": "contracts",
    "InheritedAttentionCoreV1": "model",
    "PreparedInheritedOrganization": "model",
    "PreparedInheritedAffineQuery": "model",
    "build_typed_atlas": "atlas",
    "route_raw_logits": "atlas",
    "smoothstep_grid_receiver_cover": "atlas",
    "validate_cover_anchor_identity": "atlas",
    "measure_relative_entmax15": "normalization",
    "measure_relative_entmax15_reference": "normalization",
    "measure_relative_normalize": "normalization",
    "measure_relative_softmax": "normalization",
    "normalize_donor_measure": "normalization",
    "gather_cover_dense_reference": "readers",
    "gather_cover_indexed": "readers",
    "gather_source_response_indexed": "readers",
    "prepare_anchor_values": "readers",
    "prepare_source_indexed_anchor_response": "readers",
}

__all__ = (
    "InheritedAttentionCoreV1",
    "InheritedQueryBatch",
    "PreparedAnchorValues",
    "PreparedInheritedAffineQuery",
    "PreparedInheritedOrganization",
    "PreparedSourceIndexedResponse",
    "ReceiverCover",
    "RouteScoreSnapshot",
    "TypedAssociationAtlas",
    "TypedTokenValues",
    "build_typed_atlas",
    "gather_cover_dense_reference",
    "gather_cover_indexed",
    "gather_source_response_indexed",
    "measure_relative_entmax15",
    "measure_relative_entmax15_reference",
    "measure_relative_normalize",
    "measure_relative_softmax",
    "normalize_donor_measure",
    "prepare_anchor_values",
    "prepare_source_indexed_anchor_response",
    "route_raw_logits",
    "smoothstep_grid_receiver_cover",
    "validate_cover_anchor_identity",
    "with_receiver_cover",
)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
