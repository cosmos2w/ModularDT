"""Reusable contracts for the hierarchical HONF inverse-design stack."""

from .contracts import CompactPlan, FunctionalValue, NamedContext, PhysicalDesign, VerificationResult
from .normalization import ScalarStats, VectorStats, fit_scalar, fit_vector
from .request_schema import (
    GeometryConstraints,
    RELATION_NAMES,
    RELATION_TO_ID,
    RequestCodec,
    RequestTensors,
    RequestToken,
    StructuredRequest,
)

__all__ = [
    "CompactPlan",
    "FunctionalValue",
    "GeometryConstraints",
    "NamedContext",
    "PhysicalDesign",
    "RELATION_NAMES",
    "RELATION_TO_ID",
    "RequestCodec",
    "RequestTensors",
    "RequestToken",
    "ScalarStats",
    "StructuredRequest",
    "VectorStats",
    "VerificationResult",
    "fit_scalar",
    "fit_vector",
]
