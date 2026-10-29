"""Thermal case adapters for the versioned R13-5A/B member readers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .member_routing import ThermalInteractionReusedMemberRoutingAdapter


class ThermalSupplementMemberRoutingAdapter(ThermalInteractionReusedMemberRoutingAdapter):
    """Keep native Thermal mapping while selecting the consistent joint reader."""

    SCHEMA = "thermal_r13_5_supplement_member_routing_adapter_v1"
    CORE_SCHEMA = "interaction_reused_consistent_joint_member_routing_v1"
    ARCHITECTURES = ("R13-5A-ME-ConsistentJoint95",)

    def __init__(self, normalization_stats: Mapping[str, Any], *, core_config=None,
                 core=None, heat_scale: float = 1.0, nx=128, ny=64,
                 environment_nx=24, environment_ny=8):
        if core is None:
            if not isinstance(core_config, Mapping):
                raise TypeError("ConsistentJointMemberRoutingCore requires a saved model configuration.")
            from honf_forward_core.interaction_reused_member_routing.consistent import (
                ConsistentJointMemberRoutingCore,
            )

            core = ConsistentJointMemberRoutingCore.from_config(dict(core_config))
        if getattr(core, "schema", None) != self.CORE_SCHEMA:
            raise ValueError(f"Thermal supplement adapter requires {self.CORE_SCHEMA}.")
        if getattr(core, "architecture", None) not in self.ARCHITECTURES:
            raise ValueError("R13-5A architecture must identify the consistent ME joint95 reader.")
        if getattr(core, "address_mode", None) != "ME":
            raise ValueError("R13-5A uses the existing four-block ME address view.")
        super().__init__(normalization_stats, core=core, heat_scale=heat_scale, nx=nx, ny=ny,
                         environment_nx=environment_nx, environment_ny=environment_ny)


class ThermalOrganizationConditionedMemberRoutingAdapter(ThermalSupplementMemberRoutingAdapter):
    """Keep the A adapter contract while selecting B's row-conditioned core."""

    SCHEMA = "thermal_r13_5_organization_conditioned_member_routing_adapter_v1"
    CORE_SCHEMA = "interaction_reused_organization_conditioned_member_routing_v1"
    ARCHITECTURES = ("R13-5B-ME-OrganizationConditionedJoint95",)

    def __init__(self, normalization_stats: Mapping[str, Any], *, core_config=None,
                 core=None, heat_scale: float = 1.0, nx=128, ny=64,
                 environment_nx=24, environment_ny=8):
        if core is None:
            if not isinstance(core_config, Mapping):
                raise TypeError("OrganizationConditionedMemberRoutingCore requires a saved model configuration.")
            from honf_forward_core.interaction_reused_member_routing.organization_conditioned import (
                OrganizationConditionedMemberRoutingCore,
            )

            core = OrganizationConditionedMemberRoutingCore.from_config(dict(core_config))
        if getattr(core, "schema", None) != self.CORE_SCHEMA:
            raise ValueError(f"Thermal B supplement adapter requires {self.CORE_SCHEMA}.")
        if getattr(core, "architecture", None) not in self.ARCHITECTURES:
            raise ValueError("R13-5B architecture must identify the ME organization-conditioned reader.")
        if getattr(core, "address_mode", None) != "ME":
            raise ValueError("R13-5B uses the existing four-block ME address view.")
        super().__init__(normalization_stats, core=core, heat_scale=heat_scale, nx=nx, ny=ny,
                         environment_nx=environment_nx, environment_ny=environment_ny)


__all__ = [
    "ThermalSupplementMemberRoutingAdapter",
    "ThermalOrganizationConditionedMemberRoutingAdapter",
]
