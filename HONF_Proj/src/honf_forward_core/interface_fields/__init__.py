"""Matched non-legacy interface-field architectures."""

from .core import InterfaceFieldCore
from .fixed_group_pairwise import FixedGroupPairwiseField
from .fixed_group_router import FixedGroupQueryRoute, FixedGroupRouter, FixedGroupState
from .regional_response import RegionalResponseField
from .three_term_context import ThreeTermInterfaceContext
from .types import EncodedInterfaceCase, InterfaceRead, PreparedInterfaceField

__all__ = [
    "EncodedInterfaceCase",
    "FixedGroupPairwiseField",
    "FixedGroupQueryRoute",
    "FixedGroupRouter",
    "FixedGroupState",
    "InterfaceFieldCore",
    "InterfaceRead",
    "PreparedInterfaceField",
    "RegionalResponseField",
    "ThreeTermInterfaceContext",
]
