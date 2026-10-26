"""Thermal adapter for relation-guided original-member access."""
from __future__ import annotations

from typing import Any, Mapping

from .member_routing import (
    ThermalInteractionReusedMemberRoutingAdapter,
    ThermalRelativeMemberRoutingAdapter,
)


class ThermalGuidedMemberRoutingAdapter(ThermalRelativeMemberRoutingAdapter):
    """Keep the R13-3 native Thermal mapping for the versioned R13-4 core."""

    SCHEMA = "thermal_interaction_reused_member_routing_adapter_v3"
    CORE_SCHEMA = "interaction_reused_member_routing_v3"
    ARCHITECTURES = ("R13-4-Full", "R13-4-Guide")

    def __init__(self, normalization_stats: Mapping[str, Any], *, core_config=None,
                 core=None, heat_scale: float = 1.0, nx=128, ny=64,
                 environment_nx=24, environment_ny=8):
        if core is None:
            if not isinstance(core_config, Mapping):
                raise TypeError("GuidedMemberRoutingCore requires a saved model configuration.")
            from honf_forward_core.interaction_reused_member_routing.guided import (
                GuidedMemberRoutingCore,
            )
            core = GuidedMemberRoutingCore.from_config(dict(core_config))
        if getattr(core, "schema", None) != self.CORE_SCHEMA:
            raise ValueError(f"Thermal guided member-routing adapter requires {self.CORE_SCHEMA}.")

        # The existing relative constructor dispatches only v1/v2 cores. Bypass
        # that dispatch after constructing the explicitly versioned v3 core.
        ThermalInteractionReusedMemberRoutingAdapter.__init__(
            self, normalization_stats, core=core, heat_scale=heat_scale,
            nx=nx, ny=ny, environment_nx=environment_nx,
            environment_ny=environment_ny,
        )


__all__ = ["ThermalGuidedMemberRoutingAdapter"]
