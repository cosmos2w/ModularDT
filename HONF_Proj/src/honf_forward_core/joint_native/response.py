"""Native signed source factors; E never acquires a heating-source axis."""

from ..content_hypergraph.source_read_response import SourceReadAffineFactorsV1

# Reuse the maintained contraction, transpose and explicit-export algebra.
JointNativeSourceFactors = SourceReadAffineFactorsV1

__all__ = ["JointNativeSourceFactors"]
