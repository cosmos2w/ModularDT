"""Thermal adapter for the versioned InteractionGroupCore architectures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contextual_joint import ThermalContextualJointAdapter


class ThermalInteractionGroupsAdapter(ThermalContextualJointAdapter):
    """Reuse native Thermal preparation while binding the new group core schema."""

    SCHEMA = "thermal_interaction_groups_adapter_v1"

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
        from honf_forward_core.interaction_groups import InteractionGroupCore

        if core is None:
            if not isinstance(core_config, Mapping):
                raise TypeError("InteractionGroupCore requires a saved model configuration.")
            core = InteractionGroupCore.from_config(dict(core_config))
        if getattr(core, "schema", None) != "interaction_groups_v1":
            raise ValueError("Thermal interaction-group adapter requires interaction_groups_v1.")
        super().__init__(
            normalization_stats,
            core=core,
            heat_scale=heat_scale,
            nx=nx,
            ny=ny,
            environment_nx=environment_nx,
            environment_ny=environment_ny,
        )
        if self.architecture not in {"G-EE", "G-F", "G-H", "G-EE-H"}:
            raise ValueError("InteractionGroupCore architecture must be G-EE, G-F, G-H, or G-EE-H.")

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
            raise ValueError("Not a Thermal interaction-groups checkpoint.")
        model = cls(
            payload["normalization_stats"],
            core_config=payload["core"],
            heat_scale=payload["heat_scale"],
            nx=payload["native_grid_nx_ny"][0],
            ny=payload["native_grid_nx_ny"][1],
            environment_nx=payload["environment_grid_nx_ny"][0],
            environment_ny=payload["environment_grid_nx_ny"][1],
        )
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
        return model.to(device)
