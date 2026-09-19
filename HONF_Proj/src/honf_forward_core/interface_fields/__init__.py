"""Matched non-legacy interface-field architectures."""

from .core import InterfaceFieldCore
from .fixed_group_pairwise import FixedGroupPairwiseField
from .fixed_group_router import FixedGroupQueryRoute, FixedGroupRouter, FixedGroupState
from .group_control_pairwise import GroupControlPairwiseField
from .group_control_router import (
    GroupQueryRoute,
    LowDimensionalGroupRouter,
    PhaseSharedGroupControl,
    PreparedGroupControl,
    PrototypeAnchoredGroupRouter,
)
from .group_control_support import (
    GROUP_COUNT,
    QUERY_MASK_COUNT,
    QuerySignatureSelection,
    SixBitSourceSupport,
    SixBitSupportIndex,
    SupportExecutionCounts,
    build_six_bit_source_support,
    build_six_bit_support_index,
    live_pair_values,
    pack_six_bit_mask,
    six_bit_mask,
)
from .phase_shared_group_control import PhaseSharedGroupControlPairwiseField
from .regional_response import RegionalResponseField
from .three_term_context import ThreeTermInterfaceContext
from .types import EncodedInterfaceCase, InterfaceRead, PreparedInterfaceField

__all__ = [
    "EncodedInterfaceCase",
    "FixedGroupPairwiseField",
    "FixedGroupQueryRoute",
    "FixedGroupRouter",
    "FixedGroupState",
    "GroupControlPairwiseField",
    "GroupQueryRoute",
    "GROUP_COUNT",
    "InterfaceFieldCore",
    "InterfaceRead",
    "LowDimensionalGroupRouter",
    "QUERY_MASK_COUNT",
    "PhaseSharedGroupControl",
    "PhaseSharedGroupControlPairwiseField",
    "PreparedGroupControl",
    "PreparedInterfaceField",
    "PrototypeAnchoredGroupRouter",
    "QuerySignatureSelection",
    "RegionalResponseField",
    "SixBitSourceSupport",
    "SixBitSupportIndex",
    "SupportExecutionCounts",
    "ThreeTermInterfaceContext",
    "build_six_bit_source_support",
    "build_six_bit_support_index",
    "live_pair_values",
    "pack_six_bit_mask",
    "six_bit_mask",
]
