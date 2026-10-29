"""Thermal case adapter for interaction-reused original-member routing."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .contextual_joint import ThermalContextualJointAdapter


class ThermalInteractionReusedMemberRoutingAdapter(ThermalContextualJointAdapter):
    """Keep native Thermal mapping while dispatching only the new joint core."""

    SCHEMA = "thermal_interaction_reused_member_routing_adapter_v1"
    CORE_SCHEMA = "interaction_reused_member_routing_v1"
    ARCHITECTURES = ("R13-3-Full", "R13-3-Route")

    def __init__(self, normalization_stats: Mapping[str, Any], *, core_config=None,
                 core=None, heat_scale: float = 1.0, nx=128, ny=64,
                 environment_nx=24, environment_ny=8):
        if core is None:
            if not isinstance(core_config, Mapping):
                raise TypeError("InteractionReusedMemberRoutingCore requires a saved model configuration.")
            if core_config.get("schema") == "interaction_reused_member_routing_v2":
                from honf_forward_core.interaction_reused_member_routing.relative import (
                    RelativeMemberRoutingCore,
                )
                core_type = RelativeMemberRoutingCore
            elif core_config.get("schema") == "interaction_reused_faithful_joint_member_routing_v1":
                from honf_forward_core.interaction_reused_member_routing.faithful import FaithfulMECore
                core_type = FaithfulMECore
            elif core_config.get("schema") == "interaction_reused_typed_organization_member_routing_v1":
                from honf_forward_core.interaction_reused_member_routing.typed_organization_core import (
                    TypedOrganizationMemberRoutingCore,
                )
                core_type = TypedOrganizationMemberRoutingCore
            elif core_config.get("schema") == "interaction_reused_single_distribution_member_routing_v1":
                from honf_forward_core.interaction_reused_member_routing.single_distribution_core import (
                    SingleDistributionMemberRoutingCore,
                )
                core_type = SingleDistributionMemberRoutingCore
            elif core_config.get("schema") == "interaction_reused_e_anchored_member_routing_v1":
                from honf_forward_core.interaction_reused_member_routing.anchored import (
                    AnchoredMemberRoutingCore,
                )
                core_type = AnchoredMemberRoutingCore
            else:
                from honf_forward_core.interaction_reused_member_routing import (
                    InteractionReusedMemberRoutingCore,
                )
                core_type = InteractionReusedMemberRoutingCore
            core = core_type.from_config(dict(core_config))
        if getattr(core, "schema", None) != self.CORE_SCHEMA:
            raise ValueError(f"Thermal member-routing adapter requires {self.CORE_SCHEMA}.")
        super().__init__(normalization_stats, core=core, heat_scale=heat_scale, nx=nx, ny=ny,
                         environment_nx=environment_nx, environment_ny=environment_ny)
        if self.architecture not in self.ARCHITECTURES:
            raise ValueError(f"Member-routing architecture must be one of {self.ARCHITECTURES}.")
        if int(getattr(core, "output_dim", -1)) != 4:
            raise ValueError("Thermal member routing requires one four-channel joint physical-field head.")

    def get_config(self):
        return {
            "schema": self.SCHEMA,
            "core": self.core.get_config(),
            "native_grid_nx_ny": [self.nx, self.ny],
            "environment_grid_nx_ny": [self.environment_nx, self.environment_ny],
            "heat_scale": self.heat_scale,
        }

    def state_payload(self):
        return {**self.get_config(), "normalization_stats": self.normalization_stats}

    @classmethod
    def from_state_payload(cls, payload, *, device="cpu", state_dict=None):
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA:
            raise ValueError("Not a Thermal interaction-reused member-routing checkpoint.")
        grid = payload.get("native_grid_nx_ny")
        environment = payload.get("environment_grid_nx_ny")
        if not isinstance(grid, (list, tuple)) or len(grid) != 2:
            raise ValueError("Member-routing checkpoint is missing native grid dimensions.")
        if not isinstance(environment, (list, tuple)) or len(environment) != 2:
            raise ValueError("Member-routing checkpoint is missing environment grid dimensions.")
        model = cls(payload["normalization_stats"], core_config=payload["core"],
                    heat_scale=payload["heat_scale"], nx=grid[0], ny=grid[1],
                    environment_nx=environment[0], environment_ny=environment[1])
        if model.get_config() != {key: payload[key] for key in model.get_config()}:
            raise ValueError("Member-routing adapter payload differs from its reconstructed configuration.")
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
        return model.to(device)

    def apply_native_with_work_counts(self, prepared, physical_heat=None, *, increment=False,
                                      accumulation_dtype=None, intervention=None,
                                      return_trace=False, trace_request=None):
        self.validate_prepared(prepared)
        if increment:
            raise ValueError("Joint fields require separate complete preparations for finite changes.")
        if physical_heat is not None:
            given = torch.as_tensor(physical_heat, device=prepared.scene_batch.module_features.device)
            expected = prepared.structure["heat_powers"]
            if given.shape != expected.shape or not torch.equal(given.to(expected), expected):
                raise ValueError("Heat changed: rebuild the complete joint prepared context.")
        output = self.core.decode(
            prepared.organization,
            prepared.query_batch,
            intervention=intervention,
            return_trace=return_trace,
            trace_request=trace_request,
        )
        fields = getattr(output, "joint_fields", None)
        if fields is None or fields.ndim != 3 or fields.shape[-1] != 4:
            raise ValueError("Member-routing core must return [batch, queries, 4] joint_fields.")
        flow, temperature = fields[..., :3], fields[..., 3:4]
        native = self._extract_native_roles(
            prepared, temperature, flow, accumulation_dtype=accumulation_dtype,
        )
        return native, {str(name): int(value) for name, value in output.work_counts.items()}, output

    def apply_native(self, prepared, physical_heat=None, **kwargs):
        return self.apply_native_with_work_counts(prepared, physical_heat, **kwargs)[0]


class ThermalRelativeMemberRoutingAdapter(ThermalInteractionReusedMemberRoutingAdapter):
    """Thermal-native adapter for the separately versioned relative reader."""

    SCHEMA = "thermal_interaction_reused_member_routing_adapter_v2"
    CORE_SCHEMA = "interaction_reused_member_routing_v2"
    ARCHITECTURES = ("R13-3-Relative",)


class ThermalFaithfulMemberRoutingAdapter(ThermalInteractionReusedMemberRoutingAdapter):
    """Thermal-native adapter for the versioned R13-6 faithful O reader."""

    SCHEMA = "thermal_interaction_reused_faithful_member_routing_adapter_v1"
    CORE_SCHEMA = "interaction_reused_faithful_joint_member_routing_v1"
    ARCHITECTURES = ("R13-6-O-ME-FaithfulJoint95",)


class ThermalTypedOrganizationMemberRoutingAdapter(ThermalInteractionReusedMemberRoutingAdapter):
    """Thermal-native adapter for the versioned R13-6 typed-organization H reader."""

    SCHEMA = "thermal_interaction_reused_typed_organization_member_routing_adapter_v1"
    CORE_SCHEMA = "interaction_reused_typed_organization_member_routing_v1"
    ARCHITECTURES = ("R13-6-H-ME-TypedOrganizationJoint95",)


class ThermalSingleDistributionMemberRoutingAdapter(ThermalInteractionReusedMemberRoutingAdapter):
    """Thermal-native adapter for the R13-7 U single-distribution reader."""

    SCHEMA = "thermal_interaction_reused_single_distribution_member_routing_adapter_v1"
    CORE_SCHEMA = "interaction_reused_single_distribution_member_routing_v1"
    ARCHITECTURES = ("R13-7-U-ME-SingleDistributionJoint95",)


class ThermalAnchoredMemberRoutingAdapter(ThermalInteractionReusedMemberRoutingAdapter):
    """Thermal-native R13-8 E-directory adapter for the A and L locators."""

    SCHEMA = "thermal_interaction_reused_e_anchored_member_routing_adapter_v1"
    CORE_SCHEMA = "interaction_reused_e_anchored_member_routing_v1"
    ARCHITECTURES = ("R13-8-A-EAnchor", "R13-8-L-LocalECover",
                     "R13-12-P-EAnchor", "R13-12-G-EAnchor",
                     "R13-12-P-LocalECover", "R13-12-G-LocalECover")


__all__ = [
    "ThermalFaithfulMemberRoutingAdapter",
    "ThermalAnchoredMemberRoutingAdapter",
    "ThermalInteractionReusedMemberRoutingAdapter",
    "ThermalRelativeMemberRoutingAdapter",
    "ThermalSingleDistributionMemberRoutingAdapter",
    "ThermalTypedOrganizationMemberRoutingAdapter",
]
