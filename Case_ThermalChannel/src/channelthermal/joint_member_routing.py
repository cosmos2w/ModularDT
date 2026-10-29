"""Thermal-native adapter for the versioned R13-5 joint M/E reader."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .member_routing import ThermalInteractionReusedMemberRoutingAdapter


class ThermalJointMemberRoutingAdapter(ThermalInteractionReusedMemberRoutingAdapter):
    """Keep Thermal field assembly while selecting the new joint-routing core."""

    SCHEMA = "thermal_interaction_reused_joint_member_routing_adapter_v1"
    CORE_SCHEMA = "interaction_reused_joint_member_routing_v1"
    ARCHITECTURES = ("R13-5-M-Joint95", "R13-5-ME-Joint95")

    def __init__(self, normalization_stats: Mapping[str, Any], *, core_config=None,
                 core=None, heat_scale: float = 1.0, nx=128, ny=64,
                 environment_nx=24, environment_ny=8):
        if core is None:
            if not isinstance(core_config, Mapping):
                raise TypeError("JointMemberRoutingCore requires a saved model configuration.")
            from honf_forward_core.interaction_reused_member_routing.joint import (
                JointMemberRoutingCore,
            )

            core = JointMemberRoutingCore.from_config(dict(core_config))
        if getattr(core, "schema", None) != self.CORE_SCHEMA:
            raise ValueError(f"Thermal joint member routing requires {self.CORE_SCHEMA}.")
        expected_mode = "M" if core.architecture == self.ARCHITECTURES[0] else "ME"
        if core.architecture not in self.ARCHITECTURES or getattr(core, "address_mode", None) != expected_mode:
            raise ValueError("R13-5 architecture and M/ME address mode disagree.")
        super().__init__(normalization_stats, core=core, heat_scale=heat_scale, nx=nx, ny=ny,
                         environment_nx=environment_nx, environment_ny=environment_ny)


__all__ = ["ThermalJointMemberRoutingAdapter"]
