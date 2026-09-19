"""Matched non-legacy interface-field architectures."""

from .core import InterfaceFieldCore
from .fixed_group_pairwise import FixedGroupPairwiseField
from .fixed_group_router import FixedGroupQueryRoute, FixedGroupRouter, FixedGroupState
from .group_control_pairwise import GroupControlPairwiseField
from .group_control_router import GroupQueryRoute, LowDimensionalGroupRouter, PreparedGroupControl
from .group_control_support import (
    GROUP_COUNT,
    QUERY_MASK_COUNT,
    QuerySignatureSelection,
    SixBitSupportIndex,
    SupportExecutionCounts,
    build_six_bit_support_index,
    live_pair_values,
    pack_six_bit_mask,
    six_bit_mask,
)
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
    "PreparedGroupControl",
    "PreparedInterfaceField",
    "QuerySignatureSelection",
    "RegionalResponseField",
    "SixBitSupportIndex",
    "SupportExecutionCounts",
    "ThreeTermInterfaceContext",
    "build_six_bit_support_index",
    "live_pair_values",
    "pack_six_bit_mask",
    "six_bit_mask",
]
