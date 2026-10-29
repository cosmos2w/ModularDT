"""Joint physical-field adapter with prescribed controls entering the context.

Only native observation/extraction code is inherited. No legacy affine model is
constructed and no temperature increment or heat-independent flow law is used.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from .content_hypergraph import (
    ThermalContentHypergraphAdapterV2, _stats_json,
)


class ThermalContextualJointAdapter(ThermalContentHypergraphAdapterV2):
    SCHEMA = "thermal_contextual_joint_adapter_v1"
    requires_control_in_prepare = True

    def __init__(self, normalization_stats: Mapping[str, Any], *, core_config=None,
                 core=None, heat_scale: float = 1.0, nx=128, ny=64,
                 environment_nx=24, environment_ny=8):
        nn.Module.__init__(self)
        from honf_forward_core.contextual_organization.model import ContextualOrganizationJointCore
        if heat_scale <= 0:
            raise ValueError("Prescribed heat scale must be positive and TRAIN-derived.")
        self.core = core if core is not None else ContextualOrganizationJointCore.from_config(dict(core_config))
        self.nx, self.ny = int(nx), int(ny)
        self.environment_nx, self.environment_ny = int(environment_nx), int(environment_ny)
        self.normalization_stats = _stats_json(normalization_stats)
        for suffix in ("mean", "std"):
            singular, plural = f"interface_target_{suffix}", f"interface_targets_{suffix}"
            if plural not in self.normalization_stats and singular in self.normalization_stats:
                self.normalization_stats[plural] = self.normalization_stats[singular]
        self.heat_scale = float(heat_scale)
        for name, count in (("field_mean", 5), ("field_std", 5),
                            ("interface_target_std", 2), ("internal_temperature_std", 1)):
            key = {"field_mean": "field_mean_by_channel", "field_std": "field_std_by_channel", "interface_target_std": "interface_targets_std"}.get(name, name)
            self.register_buffer(name, torch.as_tensor(self.normalization_stats[key], dtype=torch.float32).reshape(count))
        # Existing prepared-state checks use this ordinary identity. Actual
        # transforms and control inputs are also checked by the native adapter.
        self.normalizer_key = (self.SCHEMA, repr(self.normalization_stats), self.heat_scale)

    def get_config(self):
        return {"schema": self.SCHEMA, "core": self.core.get_config(),
                "native_grid_nx_ny": [self.nx, self.ny],
                "environment_grid_nx_ny": [self.environment_nx, self.environment_ny],
                "heat_scale": self.heat_scale}

    def state_payload(self):
        return {**self.get_config(), "normalization_stats": self.normalization_stats}

    @classmethod
    def from_state_payload(cls, payload, *, device="cpu", state_dict=None):
        if payload["schema"] != cls.SCHEMA:
            raise ValueError("Not a joint contextual Thermal checkpoint.")
        model = cls(payload["normalization_stats"], core_config=payload["core"],
                    heat_scale=payload["heat_scale"], nx=payload["native_grid_nx_ny"][0],
                    ny=payload["native_grid_nx_ny"][1],
                    environment_nx=payload["environment_grid_nx_ny"][0],
                    environment_ny=payload["environment_grid_nx_ny"][1])
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
        return model.to(device)

    def scene_batch(self, structure, *, scene_key=None):
        if "heat_powers" not in structure:
            raise ValueError("Joint context preparation requires prescribed heat_powers.")
        # Legacy inventory owns geometry features, but its affine boundary
        # assertion is not a restriction of this joint-field adapter.
        geometry = {k: v for k, v in structure.items() if k not in ("t_in", "t_wall")}
        scene = super().scene_batch(geometry, scene_key=scene_key)
        heat = torch.as_tensor(structure["heat_powers"], device=scene.module_features.device,
                               dtype=scene.module_features.dtype)
        if heat.shape != scene.module_mask.shape or not torch.isfinite(heat).all():
            raise ValueError("Prescribed heat must retain the original module columns.")
        raw = scene.environment_features
        operating = scene.operating_features
        boundary = []
        for name in ("t_in", "t_wall"):
            value = torch.as_tensor(structure.get(name, 0.0), device=heat.device, dtype=heat.dtype)
            boundary.append(value.reshape(1).expand(heat.shape[0]).reshape(-1, 1) if value.numel() == 1
                            else value.reshape(heat.shape[0], 1))
        return replace(scene,
            module_features=torch.cat((scene.module_features, (heat / self.heat_scale)[..., None]), -1),
            environment_features=torch.cat((raw[..., :2], raw[..., 4:]), -1),
            operating_features=torch.cat((operating[..., :10], *boundary), -1))

    @staticmethod
    def _prepared_material(scene):
        context = scene.operating_features
        radius = scene.module_features[:, :1, 2] * scene.coordinate_scale[:, 0:1]
        return torch.cat((context[:, 2:5] / 100., context[:, 5:7], radius), -1)

    def prepare_native(self, structure, fluid_xy, *, physical_heat=None, **kwargs):
        if physical_heat is not None:
            structure = {**structure, "heat_powers": physical_heat}
        return super().prepare_native(structure, fluid_xy, **kwargs)

    def _physical_temperature(self, temperature):
        return temperature * self.field_std[4].to(temperature) + self.field_mean[4].to(temperature)

    def apply_native_with_work_counts(self, prepared, physical_heat=None, *, increment=False,
                                      accumulation_dtype=None):
        self.validate_prepared(prepared)
        if increment:
            raise ValueError("Joint fields require separate complete preparations for finite changes.")
        if physical_heat is not None:
            given = torch.as_tensor(physical_heat, device=prepared.scene_batch.module_features.device)
            expected = prepared.structure["heat_powers"]
            if given.shape != expected.shape or not torch.equal(given.to(expected), expected):
                raise ValueError("Heat changed: rebuild the complete joint prepared context.")
        output = self.core.decode(prepared.organization, prepared.query_batch)
        fields = getattr(output, "joint_fields", None)
        if fields is None:
            # CoreOutput compatibility channels are slices of the same head.
            flow, temperature = output.nonlinear_field, output.thermal_temperature
        else:
            flow, temperature = fields[..., :3], fields[..., 3:4]
        return self._extract_native_roles(prepared, temperature, flow,
                   accumulation_dtype=accumulation_dtype), dict(output.work_counts)

    def apply_native(self, prepared, physical_heat=None, **kwargs):
        return self.apply_native_with_work_counts(prepared, physical_heat, **kwargs)[0]
