"""Fresh joint Thermal adapter for the regional hypergraph field core.

This adapter reuses the native Thermal input whitelist, receiver stencils and
shared-grid extraction, while its only learned module is one freshly built
``JointRegionalFieldCore``. Heating remains an application-time control and is
never part of the prepared configuration representation.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from honf_forward_core.interface_fields.joint_regional import JointRegionalFieldCore

from .source_response import (
    CONTEXT_KEYS,
    CONTEXT_WIDTH,
    ENVIRONMENT_WIDTH,
    SOURCE_WIDTH,
    THERMAL_INTERACTION_CONTROL_UNITS,
    ThermalSourceResponse,
    _thermal_interaction_dependency,
)

JOINT_THERMAL_ID = "thermal_joint_regional_v1"
JOINT_THERMAL_CHECKPOINT_SCHEMA = "thermal_joint_regional_checkpoint_v1"
JOINT_THERMAL_CHANNEL_ORDER = ("u", "v", "p", "omega", "temperature")
JOINT_THERMAL_MODES = ("J-direct", "J-geometry", "J-H")


def _validated_normalization_stats(stats: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result = {str(name): np.asarray(value, dtype=np.float32).copy() for name, value in stats.items()}
    required = {
        "field_mean_by_channel": (5,),
        "field_std_by_channel": (5,),
        "interface_targets_std": (2,),
        "internal_temperature_std": (1,),
    }
    if "interface_targets_std" not in result and "interface_target_std" in result:
        result["interface_targets_std"] = result["interface_target_std"].copy()
    if "interface_targets_mean" not in result and "interface_target_mean" in result:
        result["interface_targets_mean"] = result["interface_target_mean"].copy()
    for name, shape in required.items():
        if name not in result or result[name].shape != shape:
            raise ValueError(f"Joint Thermal normalization requires {name} with shape {shape}.")
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Joint Thermal normalization {name} must be finite.")
    if np.any(result["field_std_by_channel"] <= 0) or np.any(result["interface_targets_std"] <= 0):
        raise ValueError("Joint Thermal field and interface scales must be positive.")
    if float(result["internal_temperature_std"].reshape(-1)[0]) <= 0:
        raise ValueError("Joint Thermal material-temperature scale must be positive.")
    return result


class JointThermalRegionalAdapter(ThermalSourceResponse):
    """Thermal native adapter backed only by a new joint regional core.

    ``ThermalSourceResponse`` methods are reused without invoking its legacy
    constructor. In particular, this class owns no dormant source-response
    operator parameters: ``core`` is constructed directly as the new joint
    core, and every active parameter belongs to that core.
    """

    def __init__(
        self,
        *,
        mode: str = "J-H",
        normalization_stats: Mapping[str, Any],
        hidden: int = 128,
        message: int = 128,
        regional_anchors: int = 16,
        depth: int = 2,
        receiver_tile: int = 512,
        nx: int = 128,
        ny: int = 64,
        environment_nx: int = 24,
        environment_ny: int = 8,
        forcing_scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        if mode not in JOINT_THERMAL_MODES:
            raise ValueError(f"Joint Thermal mode must be one of {JOINT_THERMAL_MODES}.")
        if receiver_tile < 1:
            raise ValueError("Joint Thermal receiver tile must be positive.")
        if nx < 2 or ny < 2 or environment_nx < 1 or environment_ny < 1:
            raise ValueError("Joint Thermal native and environment grids must be positive and nontrivial.")
        if not np.isfinite(forcing_scale) or forcing_scale <= 0:
            raise ValueError("Joint Thermal forcing scale must be positive and finite.")

        # Deliberately bypass ThermalSourceResponse.__init__: it creates the
        # older response operator, whose parameters are not part of this model.
        nn.Module.__init__(self)
        self.mode = str(mode)
        self.seed = int(seed)
        self.normalization_stats = _validated_normalization_stats(normalization_stats)
        self.receiver_tile = int(receiver_tile)
        self.nx, self.ny = int(nx), int(ny)
        self.environment_nx, self.environment_ny = int(environment_nx), int(environment_ny)
        self.forcing_scale = float(forcing_scale)
        self.h_effective_eps, self.h_effective_max = 1.0e-3, 1.0e4
        self.environment_flow_context = False
        self.environment_flow_projection = None
        self.core = JointRegionalFieldCore(
            SOURCE_WIDTH,
            CONTEXT_WIDTH,
            ENVIRONMENT_WIDTH,
            spatial_dim=2,
            hidden=int(hidden),
            message=int(message),
            mode=self.mode,
            regional_anchors=int(regional_anchors),
            depth=int(depth),
            field_outputs=4,
            affine_outputs=1,
            query_width=0,
            forcing_scale=self.forcing_scale,
        )
        self.core_config = copy.deepcopy(dict(self.core.config))
        self._joint_model_config = {
            "mode": self.mode,
            "hidden": int(hidden),
            "message": int(message),
            "regional_anchors": int(regional_anchors),
            "depth": int(depth),
            "receiver_tile": self.receiver_tile,
            "nx": self.nx,
            "ny": self.ny,
            "environment_nx": self.environment_nx,
            "environment_ny": self.environment_ny,
            "forcing_scale": self.forcing_scale,
            "seed": self.seed,
        }

    @property
    def interaction_dependency(self):
        return _thermal_interaction_dependency(environment_flow_context=False)

    @property
    def interaction_control_units(self):
        return dict(THERMAL_INTERACTION_CONTROL_UNITS)

    @property
    def interaction_output_laws(self) -> dict[str, str]:
        """Laws for the two distinct output blocks in this joint model."""
        return {"flow": "nonlinear", "temperature_response": "affine"}

    def adapter_config(self) -> dict[str, Any]:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "environment_nx": self.environment_nx,
            "environment_ny": self.environment_ny,
            "forcing_scale": self.forcing_scale,
            "h_effective_eps": self.h_effective_eps,
            "h_effective_max": self.h_effective_max,
            "joint_output_laws": self.interaction_output_laws,
        }

    def _prepare_context_tensors(self, tensors, environment_flow_features):
        """Prepare joint state while declaring only the affine T block as a control response.

        The shared ``ThermalSourceResponse`` helper compares model-level
        ``output_law`` with a single-output dependency. A joint core has both a
        nonlinear field block and an affine temperature block, so validate the
        latter explicitly and attach the Thermal temperature dependency to the
        prepared affine response context.
        """
        dependency = _thermal_interaction_dependency(environment_flow_context=False)
        if getattr(self.core, "response_output_law", None) != dependency.output_law:
            raise ValueError("Joint Thermal affine response block disagrees with its heat dependency.")
        if environment_flow_features is not None:
            raise ValueError("Joint Thermal configuration does not accept predicted-flow environment features.")
        prepared = self.core.prepare_context(**tensors)
        prepared.dependency = dependency
        prepared.model_output_laws = self.interaction_output_laws
        return prepared

    def prepare_native(self, structure, fluid_xy, *, chunk_size=None, **kwargs):
        tile = self.receiver_tile if chunk_size is None else int(chunk_size)
        if tile < 1:
            raise ValueError("Joint Thermal receiver chunk size must be positive.")
        prepared = super().prepare_native(structure, fluid_xy, chunk_size=tile, **kwargs)
        prepared.flow_receivers = fluid_xy
        prepared.flow_receiver_snapshot = fluid_xy.detach().clone()
        return prepared

    def validate_prepared(self, prepared, structure=None):
        super().validate_prepared(prepared, structure=structure)
        receivers = getattr(prepared, "flow_receivers", None)
        snapshot = getattr(prepared, "flow_receiver_snapshot", None)
        if not torch.is_tensor(receivers) or not torch.is_tensor(snapshot) or not torch.equal(receivers, snapshot):
            raise ValueError("Joint Thermal flow receiver catalogue changed; rebuild the prepared state.")

    def _apply_native_fp32(self, prepared, physical_heat, increment, compression):
        """Apply the all-source joint affine read without legacy compression kwargs."""
        if increment and compression is not None:
            raise ValueError("Joint Thermal increments retain every original physical source column.")
        fn = self.core.apply_increment if increment else self.core.apply_forcing
        grid_values = fn(prepared.response, physical_heat.to(prepared.source_present))
        role = {
            name: self._interpolate(grid_values, stencil).reshape(
                grid_values.shape[0], *prepared.role_shapes[name], -1
            )
            for name, stencil in prepared.stencils.items()
        }
        surface = role["surface"] * prepared.source_present[:, :, None, None]
        outside = role["outside"] * prepared.source_present[:, :, None, None]
        q = -prepared.interface_conductivity[:, None, :, None] / prepared.delta[:, None, :, None] * (
            outside - surface
        )
        return role, surface, outside, q

    def apply_native(
        self,
        prepared,
        physical_heat,
        *,
        increment: bool = False,
        compression=None,
        accumulation_dtype=None,
    ) -> dict[str, Any]:
        output = super().apply_native(
            prepared,
            physical_heat,
            increment=increment,
            compression=compression,
            accumulation_dtype=accumulation_dtype,
        )
        if increment:
            # The applied control is heat; it has no edge into configuration flow.
            flow = output["fluid_temperature"].new_zeros((*output["fluid_temperature"].shape[:-1], 4))
        else:
            flow_normalized = self.core.predict_fields(
                prepared.context,
                prepared.flow_receivers,
                chunk_size=self.receiver_tile,
            )
            if flow_normalized.ndim != 3 or flow_normalized.shape[-1] != 4:
                raise ValueError("Joint Thermal field head must return standardized [B,Q,4] u/v/p/omega.")
            means = flow_normalized.new_tensor(self.normalization_stats["field_mean_by_channel"][:4])
            scales = flow_normalized.new_tensor(self.normalization_stats["field_std_by_channel"][:4])
            flow = flow_normalized * scales + means
        output["pred_field"] = torch.cat((flow, output["fluid_temperature"]), dim=-1)
        return output

    def predict_native_sample(self, sample: Mapping[str, Any], device=None) -> dict[str, np.ndarray]:
        device = torch.device(device) if device is not None else next(self.parameters()).device
        structure = {
            key: (
                copy.deepcopy(value)
                if key == "module_source_ids" and not torch.is_tensor(value) and not isinstance(value, np.ndarray)
                else torch.as_tensor(value, device=device, dtype=torch.float32)[None]
            )
            for key, value in sample["structure"].items()
            if key in CONTEXT_KEYS
        }
        fluid_xy = np.stack((np.asarray(sample["x_grid"]).reshape(-1), np.asarray(sample["y_grid"]).reshape(-1)), -1)
        fluid = torch.as_tensor(fluid_xy, device=device, dtype=torch.float32)[None]
        local = torch.as_tensor(sample["module_internal_query_points"], device=device, dtype=torch.float32)
        if local.ndim == 2:
            local = local[None]
        heat = torch.as_tensor(sample["structure"]["heat_powers"], device=device, dtype=torch.float32)[None]
        with torch.no_grad():
            output = self.apply_native(self.prepare_native(structure, fluid, local_query_points=local, ntheta=16), heat)
        ny, nx = np.asarray(sample["x_grid"]).shape
        return {
            "pred_field_grid": output["pred_field"][0].cpu().numpy().reshape(ny, nx, 5),
            "pred_interface": output["pred_interface"][0].cpu().numpy(),
            "pred_internal_temperature": output["pred_internal_temperature"][0, ..., 0].cpu().numpy(),
            "pred_port_condition": output["pred_port_condition"][0].cpu().numpy(),
            "initial_port_status": output["initial_port_status"],
        }

    def prepare_record(self, record, device=None, *, chunk_size=None):
        """Prepare native physical receivers from source IDs and input metadata."""
        device = torch.device(device) if device is not None else next(self.parameters()).device
        modules, context = record.design.modules, record.context.values
        tensor = lambda value: torch.as_tensor(value, device=device, dtype=torch.float32)
        centers = tensor([module.position_xy for module in modules])[None]
        present = tensor([float(module.active) for module in modules])[None]
        material = tensor(
            [context[key] for key in ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius")]
        )[None]
        structure = {
            "module_centers": centers,
            "module_present": present,
            "material_params": material,
            "module_source_ids": tuple(module.module_id for module in modules),
        }
        for key in ("re", "u_in", "domain_length_x", "domain_length_y"):
            structure[key] = tensor([context[key]])[None]
        roles = record.output.roles
        fluid = tensor(roles["fluid_fields"].query_features[:, :2])[None]
        ids = tuple(module.module_id for module in modules)
        interface_rows = [
            np.flatnonzero(np.asarray(roles["interface"].receiver_module_ids) == module_id) for module_id in ids
        ]
        material_rows = [
            np.flatnonzero(np.asarray(roles["solid_temperature"].receiver_module_ids) == module_id)
            for module_id in ids
        ]
        if len({len(rows) for rows in interface_rows}) != 1 or len({len(rows) for rows in material_rows}) != 1:
            raise ValueError("Native Thermal role receivers require equal per-module counts.")
        ntheta = len(interface_rows[0])
        expected_theta = np.arange(ntheta) * (2 * np.pi / ntheta)
        for rows in interface_rows:
            query = np.asarray(roles["interface"].query_features)[rows]
            if not np.allclose(query[:, 0], expected_theta, rtol=0, atol=2.0e-6):
                raise ValueError("Native interface receivers must retain original angular ordering.")
        local = tensor(
            np.stack([np.asarray(roles["solid_temperature"].query_features)[rows, :2] for rows in material_rows])
        )[None]
        prepared = self.prepare_native(
            structure,
            fluid,
            local_query_points=local,
            ntheta=ntheta,
            chunk_size=chunk_size,
        )
        prepared.record_source_ids = ids
        prepared.interface_rows = interface_rows
        prepared.material_rows = material_rows
        prepared.interface_row_count = len(roles["interface"].query_features)
        prepared.material_row_count = len(roles["solid_temperature"].query_features)
        prepared.module_physical_heat = tensor([module.heating for module in modules])[None]
        return prepared

    @staticmethod
    def _record_role_outputs(prepared, output):
        interface = output["pred_interface"].new_empty(prepared.interface_row_count, 2)
        material = output["pred_internal_temperature"].new_empty(prepared.material_row_count, 1)
        for index, rows in enumerate(prepared.interface_rows):
            interface[torch.as_tensor(rows, device=interface.device)] = output["pred_interface"][0, index]
        for index, rows in enumerate(prepared.material_rows):
            material[torch.as_tensor(rows, device=material.device)] = output["pred_internal_temperature"][0, index]
        return interface, material

    def apply_record(self, prepared, physical_heat=None, *, accumulation_dtype=None):
        heat = prepared.module_physical_heat if physical_heat is None else physical_heat
        heat = torch.as_tensor(
            heat,
            device=prepared.source_present.device,
            dtype=accumulation_dtype or prepared.source_present.dtype,
        )
        if heat.ndim == 1:
            heat = heat[None]
        output = self.apply_native(prepared, heat, accumulation_dtype=accumulation_dtype)
        interface, material = self._record_role_outputs(prepared, output)
        return {"fluid_fields": output["pred_field"][0], "interface": interface, "solid_temperature": material}

    def apply_record_increment(
        self,
        prepared,
        delta_heat,
        *,
        compression=None,
        accumulation_dtype=None,
    ):
        if compression is not None:
            raise ValueError("Joint Thermal increments retain every original physical source column.")
        delta_heat = torch.as_tensor(
            delta_heat,
            device=prepared.source_present.device,
            dtype=accumulation_dtype or prepared.source_present.dtype,
        )
        if delta_heat.ndim == 1:
            delta_heat = delta_heat[None]
        output = self.apply_native(
            prepared,
            delta_heat,
            increment=True,
            accumulation_dtype=accumulation_dtype,
        )
        interface, material = self._record_role_outputs(prepared, output)
        return {"fluid_fields": output["pred_field"][0], "interface": interface, "solid_temperature": material}

    def predict_record(self, record, device=None):
        with torch.no_grad():
            output = self.apply_record(self.prepare_record(record, device))
        return {key: value.cpu().numpy() for key, value in output.items()}

    def model_config(self) -> dict[str, Any]:
        return copy.deepcopy(self._joint_model_config)

    def checkpoint_payload(self, *, provider_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "checkpoint_schema": JOINT_THERMAL_CHECKPOINT_SCHEMA,
            "model_family": JOINT_THERMAL_ID,
            "channel_order": list(JOINT_THERMAL_CHANNEL_ORDER),
            "model_config": self.model_config(),
            "adapter_config": self.adapter_config(),
            "model_state_dict": self.state_dict(),
            "global_normalization_stats": copy.deepcopy(self.normalization_stats),
        }
        if provider_identity is not None:
            payload["experiment_identity"] = {"provider_identity": copy.deepcopy(dict(provider_identity))}
        return payload


def load_joint_thermal_checkpoint(source: Mapping[str, Any] | str | Path, device="cpu") -> JointThermalRegionalAdapter:
    """Load one standalone candidate without consulting any legacy model path."""
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = torch.load(Path(source), map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema") != JOINT_THERMAL_CHECKPOINT_SCHEMA:
        raise ValueError("Checkpoint is not a self-contained joint Thermal regional model.")
    if payload.get("model_family") != JOINT_THERMAL_ID:
        raise ValueError("Checkpoint model family differs from the joint Thermal regional adapter.")
    if payload.get("channel_order") != list(JOINT_THERMAL_CHANNEL_ORDER):
        raise ValueError("Checkpoint Thermal channel order is not the native u/v/p/omega/temperature order.")
    if any(key in payload for key in ("parent_checkpoint", "flow_checkpoint", "thermal_parent", "flow_parent")):
        raise ValueError("Joint Thermal checkpoints cannot bind an external learned field parent.")
    config = payload.get("model_config")
    stats = payload.get("global_normalization_stats")
    state = payload.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(stats, Mapping) or not isinstance(state, Mapping):
        raise TypeError("Joint Thermal checkpoint config, normalization and tensors must be mappings.")
    model = JointThermalRegionalAdapter(normalization_stats=stats, **dict(config))
    if any(not name.startswith("core.") for name in state):
        raise ValueError("Joint Thermal checkpoint contains tensors outside its single active core.")
    model.load_state_dict(state, strict=True)
    if any(not name.startswith("core.") for name, _ in model.named_parameters()):
        raise RuntimeError("Joint Thermal adapter registered parameters outside the new joint core.")
    return model.to(device)
