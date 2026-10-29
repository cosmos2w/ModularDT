"""Baseline-conditioned, physically indexed interaction-response factors."""

from .factor_operator import AnchoredResponseFactorOperator
from .organizer import InputOnlySupportScorer, masked_support_classification_loss
from .support_search import (
    SupportSearchResult,
    enumerate_unary_pair_candidates,
    group_sparse_support_search,
    select_factor_control,
    size_matched_random_support,
)
from .training import (
    ResponseTrainingExample,
    masked_role_response_loss,
    response_factor_training_step,
    stencil_examples_to_torch,
)
from .types import (
    DEFAULT_OUTPUT_CHANNELS,
    DEFAULT_QUERY_FEATURE_DIMS,
    VALID_EVIDENCE_SOURCES,
    BaselineResponseCache,
    EvidenceSource,
    ReceiverSupport,
    ResponseFactor,
    ResponseQueries,
    ResponseQueryBatch,
    ValidityNeighborhood,
    make_baseline_cache,
)

__all__ = [
    "DEFAULT_OUTPUT_CHANNELS",
    "DEFAULT_QUERY_FEATURE_DIMS",
    "VALID_EVIDENCE_SOURCES",
    "AnchoredResponseFactorOperator",
    "BaselineResponseCache",
    "EvidenceSource",
    "InputOnlySupportScorer",
    "ReceiverSupport",
    "ResponseFactor",
    "ResponseQueries",
    "ResponseQueryBatch",
    "ResponseTrainingExample",
    "SupportSearchResult",
    "ValidityNeighborhood",
    "enumerate_unary_pair_candidates",
    "group_sparse_support_search",
    "make_baseline_cache",
    "masked_role_response_loss",
    "masked_support_classification_loss",
    "response_factor_training_step",
    "select_factor_control",
    "size_matched_random_support",
    "stencil_examples_to_torch",
]
