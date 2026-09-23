"""Matched non-legacy interface-field architectures."""

from .core import InterfaceFieldCore
from .case_group_budget import (
    CaseGroupBudget,
    CaseGroupGate,
    gate_reference_eta_kappa,
    routing_logit_scale,
    sparsification_continuation,
)
from .budgeted_group_control import BudgetedGroupControlPairwiseField, BudgetedGroupRouter
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
from .occupancy_group_control import (
    OccupancyAdaptiveGroupControlPairwiseField,
    OccupancyGroupControlPairwiseField,
)
from .occupancy_group_router import (
    OCCUPANCY_CONTROL_DIM,
    OCCUPANCY_KMAX,
    OccupancyAdaptiveGroupRouter,
    OccupancyGroupPlan,
    OccupancyGroupRouter,
    OccupancyPreparedGroupControl,
    direct_mask_intersection,
    occupancy_mask,
    occupancy_support,
    pack_16bit_mask,
)
from .mass_competitive_group_control import MassCompetitiveGroupControlPairwiseField
from .mass_competitive_router import (
    MASS_COMPETITIVE_EPSILON,
    MassCompetitiveGroupPlan,
    MassCompetitiveGroupRouter,
    MassCompetitivePreparedGroupControl,
    mass_competition,
)
from .sparse_incidence_group_control import SparseIncidenceGroupControlPairwiseField
from .sparse_incidence_router import (
    SparseIncidenceGroupRouter,
    SparseIncidencePreparedGroupControl,
)
from .regional_response import RegionalResponseField
from .three_term_context import ThreeTermInterfaceContext
from .types import EncodedInterfaceCase, InterfaceRead, PreparedInterfaceField


def __getattr__(name: str):
    """Lazily expose the opt-in Run-1408 backend.

    Historical imports should not require the sampled-reader module (or its
    optional implementation dependencies) to be importable.  The factory
    performs the same narrow import only when the new architecture is chosen.
    """

    if name == "HypergraphQuadratureField":
        from .hypergraph_quadrature import HypergraphQuadratureField

        return HypergraphQuadratureField
    raise AttributeError(name)

__all__ = [
    "EncodedInterfaceCase",
    "CaseGroupBudget",
    "CaseGroupGate",
    "BudgetedGroupControlPairwiseField",
    "BudgetedGroupRouter",
    "FixedGroupPairwiseField",
    "FixedGroupQueryRoute",
    "FixedGroupRouter",
    "FixedGroupState",
    "GroupControlPairwiseField",
    "HypergraphQuadratureField",
    "GroupQueryRoute",
    "GROUP_COUNT",
    "InterfaceFieldCore",
    "InterfaceRead",
    "LowDimensionalGroupRouter",
    "QUERY_MASK_COUNT",
    "PhaseSharedGroupControl",
    "PhaseSharedGroupControlPairwiseField",
    "OccupancyAdaptiveGroupControlPairwiseField",
    "OccupancyGroupControlPairwiseField",
    "OccupancyAdaptiveGroupRouter",
    "OccupancyGroupPlan",
    "OccupancyGroupRouter",
    "OccupancyPreparedGroupControl",
    "MassCompetitiveGroupControlPairwiseField",
    "MassCompetitiveGroupPlan",
    "MassCompetitiveGroupRouter",
    "MassCompetitivePreparedGroupControl",
    "SparseIncidenceGroupControlPairwiseField",
    "SparseIncidenceGroupRouter",
    "SparseIncidencePreparedGroupControl",
    "MASS_COMPETITIVE_EPSILON",
    "mass_competition",
    "OCCUPANCY_KMAX",
    "OCCUPANCY_CONTROL_DIM",
    "direct_mask_intersection",
    "occupancy_mask",
    "occupancy_support",
    "pack_16bit_mask",
    "PreparedGroupControl",
    "PreparedInterfaceField",
    "PrototypeAnchoredGroupRouter",
    "QuerySignatureSelection",
    "RegionalResponseField",
    "SixBitSourceSupport",
    "SixBitSupportIndex",
    "SupportExecutionCounts",
    "ThreeTermInterfaceContext",
    "gate_reference_eta_kappa",
    "routing_logit_scale",
    "sparsification_continuation",
    "build_six_bit_source_support",
    "build_six_bit_support_index",
    "live_pair_values",
    "pack_six_bit_mask",
    "six_bit_mask",
]
