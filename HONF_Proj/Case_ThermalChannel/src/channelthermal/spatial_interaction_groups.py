"""Thermal adapter for spatially resolved interaction-group fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contextual_joint import ThermalContextualJointAdapter


class ThermalSpatialInteractionGroupsAdapter(ThermalContextualJointAdapter):
    """Reuse Thermal's native preparation and extraction with the Round13 core."""

    SCHEMA = "thermal_spatial_interaction_groups_adapter_v1"
    CORE_SCHEMA = "spatial_interaction_group_fields_v1"
    ARCHITECTURES = ("F13-0", "F13-1")

    def __init__(
        self,
        normalization_stats: Mapping[str, Any],
        *,
        core_config=None,
        core=None,
        heat_scale: float = 1.0,
        nx=128,
        ny=64,
        environment_nx=24,
        environment_ny=8,
    ):
        if core is None:
            if not isinstance(core_config, Mapping):
                raise TypeError("SpatialInteractionGroupCore requires a saved model configuration.")
            from honf_forward_core.spatial_interaction_groups import SpatialInteractionGroupCore

            core = SpatialInteractionGroupCore.from_config(dict(core_config))
        if getattr(core, "schema", None) != self.CORE_SCHEMA:
            raise ValueError(f"Thermal spatial-group adapter requires {self.CORE_SCHEMA}.")
        super().__init__(
            normalization_stats,
            core=core,
            heat_scale=heat_scale,
            nx=nx,
            ny=ny,
            environment_nx=environment_nx,
            environment_ny=environment_ny,
        )
        if self.architecture not in self.ARCHITECTURES:
            raise ValueError(f"SpatialInteractionGroupCore architecture must be one of {self.ARCHITECTURES}.")

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
            raise ValueError("Not a Thermal spatial-interaction-groups checkpoint.")
        grid = payload.get("native_grid_nx_ny")
        environment = payload.get("environment_grid_nx_ny")
        if not isinstance(grid, (list, tuple)) or len(grid) != 2:
            raise ValueError("Spatial-group checkpoint is missing native grid dimensions.")
        if not isinstance(environment, (list, tuple)) or len(environment) != 2:
            raise ValueError("Spatial-group checkpoint is missing environment grid dimensions.")
        model = cls(
            payload["normalization_stats"],
            core_config=payload["core"],
            heat_scale=payload["heat_scale"],
            nx=grid[0],
            ny=grid[1],
            environment_nx=environment[0],
            environment_ny=environment[1],
        )
        if model.get_config() != {key: payload[key] for key in model.get_config()}:
            raise ValueError("Spatial-group adapter payload differs from its reconstructed configuration.")
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
        return model.to(device)


__all__ = ["ThermalSpatialInteractionGroupsAdapter"]
