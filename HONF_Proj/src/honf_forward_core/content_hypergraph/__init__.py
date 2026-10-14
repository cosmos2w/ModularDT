"""Shared target-free A-D/A-H6 attention and content-hypergraph core."""

from .attention import TypedAttentionBlock, explicit_attention
from .contracts import CoreOutput, PreparedOrganization, PreparedOrganizationV2, QueryBatch, SceneBatch
from .model import ContentHypergraphCore
from .model_v2 import ContentHypergraphCoreV2, PreparedAffineQueryV2
from .response import PreparedAffineQuery

__all__ = [
    "ContentHypergraphCore",
    "ContentHypergraphCoreV2",
    "CoreOutput",
    "PreparedAffineQuery",
    "PreparedOrganization",
    "PreparedOrganizationV2",
    "PreparedAffineQueryV2",
    "QueryBatch",
    "SceneBatch",
    "TypedAttentionBlock",
    "explicit_attention",
]
