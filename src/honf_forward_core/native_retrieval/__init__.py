"""Opt-in native-token hypergraph retrieval for the maintained A2-D path."""

from .contracts import NativeQueryRetrievalPlan, NativeTokenRecords, NativeTokenStore
from .hypergraph import (
    NativeRetrievalHypergraph,
    build_native_hypergraph,
    build_native_retrieval_hypergraph,
)
from .planning import build_query_retrieval_plan, max_product_union, query_seed_permissions
from .route_capture import NativeRouteSnapshot

__all__ = [
    "NativeQueryRetrievalPlan",
    "NativeRetrievalHypergraph",
    "NativeRouteSnapshot",
    "NativeTokenRecords",
    "NativeTokenStore",
    "build_native_hypergraph",
    "build_native_retrieval_hypergraph",
    "build_query_retrieval_plan",
    "max_product_union",
    "query_seed_permissions",
]
