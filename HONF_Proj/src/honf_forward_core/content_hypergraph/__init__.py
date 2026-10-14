"""Shared target-free A-D/A-H6 attention and content-hypergraph core."""

from .attention import TypedAttentionBlock, explicit_attention
from .contracts import CoreOutput, PreparedOrganization, QueryBatch, SceneBatch
from .model import ContentHypergraphCore
from .response import PreparedAffineQuery

__all__ = [
    "ContentHypergraphCore",
    "CoreOutput",
    "PreparedAffineQuery",
    "PreparedOrganization",
    "QueryBatch",
    "SceneBatch",
    "TypedAttentionBlock",
    "explicit_attention",
]
