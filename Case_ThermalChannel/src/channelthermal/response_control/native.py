"""Autograd-preserving native ThermalChannel adapter for response fitting.

The adapter keeps only module-slot capacity and static material properties
from a packed case. Re, inlet speed, and domain geometry come from each typed
operating context; module centers and heat come from ``DesignInput``. It never
reads field targets, solved boundary values, or evidence masks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torch.nn.parameter import UninitializedParameter

from channelthermal.model import ChannelThermalHONFModel
from honf_forward_core.interface_fields.capabilities import CAMPAIGN_ARCHITECTURES

from .contracts import AbsolutePrediction, DesignInput, RoleQuery, role_receiver_world_xy

NATIVE_RESPONSE_ARCHITECTURES = frozenset(
    {
        *CAMPAIGN_ARCHITECTURES,
        "dense_pairwise_field",
        "sparse_incidence_group_control_honf",
        # Preserve the established conversion/refit path for historical runs.
        "three_term_full_access_honf",
        "direct_pairwise_control_honf",
    }
)


class _DecodedPreparedCall(nn.Module):
    """Expose the model's maintained prepared-query decoder to functional_call."""

    def __init__(self, target: ChannelThermalHONFModel) -> None:
        super().__init__()
        self.target = target

    def forward(self, prepared: Any, query_xy: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.target.decode_prepared(prepared, query_xy)


def _array_tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.array(value, dtype=np.float32, copy=True), device=device, dtype=dtype)


def _stats_vector(
    stats: Mapping[str, Any],
    mean_key: str,
    std_key: str,
    width: int,
    *,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mean_key not in stats or std_key not in stats:
        raise KeyError(f"Checkpoint normalization is missing {mean_key!r}/{std_key!r}.")
    mean = _array_tensor(stats[mean_key], device=like.device, dtype=like.dtype).reshape(-1)
    std = _array_tensor(stats[std_key], device=like.device, dtype=like.dtype).reshape(-1)
    if mean.numel() < width or std.numel() < width:
        raise ValueError(f"Checkpoint normalization {mean_key!r} has fewer than {width} features.")
    return mean[:width], std[:width].clamp_min(1.0e-8)


def _physical_output(
    values: torch.Tensor,
    stats: Mapping[str, Any],
    mean_key: str,
    std_key: str,
    *,
    normalize_targets: bool,
) -> torch.Tensor:
    if not normalize_targets:
        return values
    mean, std = _stats_vector(stats, mean_key, std_key, values.shape[-1], like=values)
    return values * std + mean


class DifferentiableThermalOperator:
    """Run one prepared native model call and decode typed physical roles.

    ``input_template`` is a packed dataset sample, but the adapter reads only
    its module-slot capacity and static ``material_params`` vector. It does
    not retain or read the
    template's outputs, interface targets, internal targets, or masks.
    """

    def __init__(
        self,
        model: ChannelThermalHONFModel,
        input_template: Mapping[str, Any],
        *,
        dataset_config: Mapping[str, Any],
        normalization_stats: Mapping[str, Any],
        query_batch_size: int = 2048,
        capture_packet_inputs: bool = False,
        organizer_shadow: bool = False,
        organizer_gradient_policy: str = "whole_wrapper_shadow_v1",
    ) -> None:
        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive.")
        self.model = model
        self.dataset_config = dict(dataset_config)
        self.normalization_stats = dict(normalization_stats)
        self.query_batch_size = int(query_batch_size)
        self.capture_packet_inputs = bool(capture_packet_inputs)
        self.organizer_shadow = bool(organizer_shadow)
        self.organizer_gradient_policy = organizer_gradient_policy
        self.last_packet_inputs: Mapping[str, Any] | None = None
        self.normalize_inputs = bool(self.dataset_config.get("normalize_inputs", False))
        self.normalize_targets = bool(self.dataset_config.get("normalize_targets", False))
        parameter = next(
            (
                item
                for item in model.parameters()
                if not isinstance(item, UninitializedParameter)
            ),
            None,
        )
        if parameter is None:
            raise ValueError("A native ThermalChannel model must contain parameters.")
        self.device = parameter.device
        self.dtype = parameter.dtype
        raw_structure = input_template.get("structure")
        if not isinstance(raw_structure, Mapping):
            raise TypeError("input_template must contain a structure mapping.")
        required = {"module_centers", "material_params"}
        missing = required - set(raw_structure)
        if missing:
            raise KeyError(f"Packed input structure is missing static keys: {sorted(missing)}.")
        self.max_modules = int(np.asarray(raw_structure["module_centers"]).shape[0])
        if self.max_modules <= 0:
            raise ValueError("Packed case has no module slots.")
        # Baseline heat and every target array in input_template are ignored.
        self.static_material_params = _array_tensor(
            raw_structure["material_params"], device=self.device, dtype=self.dtype
        ).reshape(1, -1)
        self.field_names = tuple(model.config.channelthermal.field_names)
        if "p" not in self.field_names or "temperature" not in self.field_names:
            raise ValueError("Checkpoint field schema must contain pressure and temperature.")
        architecture = str(model.config.core_honf.forward_architecture)
        if architecture not in NATIVE_RESPONSE_ARCHITECTURES:
            raise ValueError(
                "Response fitting requires one of the supported shared fine-interaction architectures "
                f"{sorted(NATIVE_RESPONSE_ARCHITECTURES)}; "
                f"got {architecture!r}."
            )
        if not model.local_coupling.has_local_surrogate:
            raise RuntimeError("The local surrogate embedded or referenced by this checkpoint must remain attached.")
        if not model._should_use_local_outputs(str(model.config.channelthermal.internal_prediction_mode)):
            raise RuntimeError("Native response fitting requires checkpoint-native local predictions.")
        if not bool(model.config.channelthermal.local_module_params_from_used_ports):
            raise RuntimeError(
                "Native response fitting requires predicted-port T_env/h summaries in local-module inputs."
            )
        self._capture_cover_plans = False
        self._captured_cover_plans: tuple[Any, ...] | None = None

    def capture_anchor_cover_plans(
        self,
        design: DesignInput,
        context: Mapping[str, Any],
        role_queries: Mapping[str, RoleQuery],
    ) -> tuple[Any, ...]:
        """Freeze an attached input-only organizer's baseline decision once."""

        if self.model.config.core_honf.forward_architecture != "dense_pairwise_field":
            raise ValueError("Native cover capture requires the intact Dense checkpoint.")
        if self.model.core.native_interaction_policy is None:
            raise ValueError("Attach an input-only native cover policy before anchor capture.")
        if self._capture_cover_plans:
            raise RuntimeError("Native cover capture cannot be nested.")
        self._captured_cover_plans = None
        self._capture_cover_plans = True
        try:
            with torch.no_grad():
                self(design, context, role_queries)
        finally:
            self._capture_cover_plans = False
        if self._captured_cover_plans is None:
            raise RuntimeError("The native organizer did not supply an applied cover plan.")
        return self._captured_cover_plans

    def _normalize_heat(self, heat: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs:
            return heat
        mean, std = _stats_vector(
            self.normalization_stats, "heat_power_mean", "heat_power_std", 1, like=heat
        )
        return (heat - mean[0]) / std[0]

    def _context_structure(self, context: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        required = (
            "re",
            "u_in",
            "domain_length_x",
            "domain_length_y",
            "nu",
            "solid_alpha",
            "fluid_alpha",
            "solid_k",
            "fluid_k",
            "module_radius",
        )
        missing = [key for key in required if key not in context]
        if missing:
            raise KeyError(f"Operating context is missing native inputs: {missing}.")
        values = {key: float(context[key]) for key in required}
        if not np.isfinite(list(values.values())).all():
            raise ValueError("Native physical context values must be finite.")
        for key, expected in (
            ("domain_length_x", float(self.model.config.core_honf.domain_length_x)),
            ("domain_length_y", float(self.model.config.core_honf.domain_length_y)),
            ("module_radius", float(self.model.config.core_honf.module_radius)),
        ):
            if not np.isclose(values[key], expected, atol=1.0e-6, rtol=1.0e-6):
                raise ValueError(
                    f"Operating {key}={values[key]} differs from checkpoint value {expected}."
                )
        material_names = (
            "nu",
            "solid_alpha",
            "fluid_alpha",
            "solid_k",
            "fluid_k",
            "module_radius",
        )
        material_values = [float(context[name]) for name in material_names]
        if not np.isfinite(material_values).all() or any(value <= 0.0 for value in material_values):
            raise ValueError("All six typed material parameters must be positive and finite.")
        material = _array_tensor(material_values, device=self.device, dtype=self.dtype).reshape(1, -1)
        if not np.isclose(material_values[-1], values["module_radius"], atol=1.0e-6, rtol=1.0e-6):
            raise ValueError("Typed material radius differs from the declared operating radius.")
        # The packed sample supplies the fallback material schema and static
        # local-surrogate seed, but per-family context owns the actual values.
        if self.static_material_params.shape[-1] != len(material_names):
            raise ValueError("Packed static material vector does not match the six-value schema.")
        return {
            "re": _array_tensor([values["re"]], device=self.device, dtype=self.dtype).reshape(1, 1),
            "u_in": _array_tensor([values["u_in"]], device=self.device, dtype=self.dtype).reshape(1, 1),
            "material_params": material,
            "domain_length_x": _array_tensor([values["domain_length_x"]], device=self.device, dtype=self.dtype).reshape(1, 1),
            "domain_length_y": _array_tensor([values["domain_length_y"]], device=self.device, dtype=self.dtype).reshape(1, 1),
        }

    def _module_inputs(
        self, design: DesignInput, context: Mapping[str, Any]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        count = int(design.module_positions.shape[0])
        if count > self.max_modules:
            raise ValueError(f"Design has {count} modules, packed input allows {self.max_modules}.")
        live_centers = design.module_positions.to(device=self.device, dtype=self.dtype)
        live_heat = design.module_heating.to(device=self.device, dtype=self.dtype)
        pad = self.max_modules - count
        centers = F.pad(live_centers, (0, 0, 0, pad)).unsqueeze(0)
        physical_heat = F.pad(live_heat, (0, pad)).unsqueeze(0)
        present = F.pad(
            design.module_present.to(device=self.device, dtype=self.dtype), (0, pad)
        ).unsqueeze(0)
        structure = self._context_structure(context)
        structure.update(
            {
                "module_centers": centers,
                "heat_powers": self._normalize_heat(physical_heat),
                "module_present": present,
            }
        )
        material = structure["material_params"]
        local_params = physical_heat.new_zeros((1, self.max_modules, 7))
        local_params[..., 0] = physical_heat
        if material.shape[-1] > 3:
            local_params[..., 1] = material[:, None, 3]
        if material.shape[-1] > 1:
            local_params[..., 2] = material[:, None, 1]
        local_params = local_params * present.unsqueeze(-1)
        return structure, local_params

    @staticmethod
    def _rows_by_slot(
        query: RoleQuery,
        module_count: int,
        active_slots: tuple[bool, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if query.receiver_slots is None:
            raise ValueError(f"Material role {query.role!r} needs receiver module slots.")
        active = active_slots or tuple(True for _ in range(module_count))
        if len(active) != module_count:
            raise ValueError("Active-slot mask must cover every design module slot.")
        slots = torch.as_tensor(query.receiver_slots, device=query.query_features.device, dtype=torch.long)
        rows = tuple(torch.nonzero(slots == index, as_tuple=False).reshape(-1) for index in range(module_count))
        if any(row.numel() == 0 for index, row in enumerate(rows) if active[index]):
            raise ValueError(f"Every active module needs {query.role!r} receiver rows.")
        if any(row.numel() > 0 for index, row in enumerate(rows) if not active[index]):
            raise ValueError(f"Inactive module slots cannot receive {query.role!r} rows.")
        return rows

    def _interface_inputs(
        self,
        query: RoleQuery,
        module_count: int,
        active_slots: tuple[bool, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        rows = self._rows_by_slot(query, module_count, active_slots)
        counts = {int(row.numel()) for index, row in enumerate(rows) if active_slots[index]}
        if not counts:
            raise ValueError("Native interface inputs require at least one active module.")
        if len(counts) != 1:
            raise ValueError("Interface receivers need the same port count for each active module.")
        ports = counts.pop()
        if query.query_features.shape[1] < 3:
            raise ValueError("Interface role needs [theta, normal_x, normal_y] geometry features.")
        geometry = query.query_features[:, :3].to(device=self.device, dtype=self.dtype)
        condition = torch.zeros(
            (1, self.max_modules, ports, 8), device=self.device, dtype=self.dtype
        )
        for slot, indices in enumerate(rows):
            if active_slots[slot]:
                condition[0, slot, :, :3] = geometry.index_select(0, indices)
        if self.normalize_inputs:
            mean, std = _stats_vector(
                self.normalization_stats,
                "interface_condition_mean",
                "interface_condition_std",
                3,
                like=condition,
            )
            condition[..., :3] = (condition[..., :3] - mean) / std
        # Match the maintained dataset's token mapping while withholding all
        # observed temperature, heat-transfer, and flux channels.
        ports_only = torch.cat([condition[..., :4], condition[..., 7:8]], dim=-1)
        return condition, ports_only, rows

    def _solid_inputs(
        self,
        query: RoleQuery,
        module_count: int,
        active_slots: tuple[bool, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        rows = self._rows_by_slot(query, module_count, active_slots)
        counts = {int(row.numel()) for index, row in enumerate(rows) if active_slots[index]}
        if not counts:
            raise ValueError("Native solid inputs require at least one active module.")
        if len(counts) != 1:
            raise ValueError("Solid material queries need an equal point count per active module.")
        points = query.query_features[:, :2].to(device=self.device, dtype=self.dtype)
        first_active = next(index for index, is_active in enumerate(active_slots) if is_active)
        first = points.index_select(0, rows[first_active])
        for slot, indices in enumerate(rows):
            if not active_slots[slot] or slot == first_active:
                continue
            candidate = points.index_select(0, indices)
            if candidate.shape != first.shape or not torch.allclose(candidate, first, atol=1.0e-7, rtol=0.0):
                raise ValueError("Native local solid heads require a shared material-local point order.")
        return first.unsqueeze(0), rows

    def __call__(
        self,
        design: DesignInput,
        context: Mapping[str, Any],
        role_queries: Mapping[str, RoleQuery],
        *,
        fixed_cover_plans: tuple[Any, ...] | None = None,
        cover_plan_builder: Any | None = None,
        detach_model_parameters: bool = False,
    ) -> AbsolutePrediction:
        if fixed_cover_plans is not None and cover_plan_builder is not None:
            raise ValueError("Pass either fixed_cover_plans or cover_plan_builder, not both.")
        if fixed_cover_plans is not None or cover_plan_builder is not None:
            if self.model.config.core_honf.forward_architecture != "dense_pairwise_field":
                raise ValueError("Frozen native covers require the intact Dense interface-field model.")
            if fixed_cover_plans is not None and len(fixed_cover_plans) != 1:
                raise ValueError("The native Thermal adapter expects one frozen plan for its one-case batch.")
        expected_roles = {"fluid_fields", "interface", "solid_temperature"}
        if set(role_queries) != expected_roles:
            raise ValueError(f"Native callback expects exactly {sorted(expected_roles)} roles.")
        if detach_model_parameters:
            design = DesignInput(
                module_positions=design.module_positions.detach(),
                module_heating=design.module_heating.detach(),
                module_present=design.module_present.detach(),
            )
            context = {
                str(key): value.detach() if torch.is_tensor(value) else value
                for key, value in context.items()
            }
            role_queries = {
                str(name): replace(query, query_features=query.query_features.detach())
                for name, query in role_queries.items()
            }
        fluid = role_queries["fluid_fields"]
        interface = role_queries["interface"]
        solid = role_queries["solid_temperature"]
        if fluid.coordinate_kind != "eulerian" or fluid.query_features.shape[1] < 2:
            raise ValueError("Fluid fields must use fixed Eulerian xy coordinates.")
        if tuple(fluid.channel_names) != self.field_names:
            raise ValueError("Checkpoint field channels and typed evidence channels do not align.")
        if len(solid.channel_names) != 1 or interface.query_features.shape[1] < 3:
            raise ValueError("Native internal/interface role schemas are unsupported.")

        module_count = int(design.module_positions.shape[0])
        structure, local_params = self._module_inputs(design, context)
        active_slots = tuple(bool(value) for value in design.module_present.detach().bool().cpu().tolist())
        interface_condition, teacher_ports, interface_rows = self._interface_inputs(
            interface, module_count, active_slots
        )
        local_query, solid_rows = self._solid_inputs(solid, module_count, active_slots)
        field_coordinates = fluid.query_features[:, :2].to(device=self.device, dtype=self.dtype)
        if field_coordinates.shape[0] == 0:
            raise ValueError("Fluid query set must be nonempty.")

        output_chunks: list[torch.Tensor] = []
        prepared = None
        first_output: Mapping[str, Any] | None = None
        detached_parameters = (
            {name: parameter.detach() for name, parameter in self.model.named_parameters()}
            if detach_model_parameters else None
        )
        detached_buffers = (
            {name: buffer.detach().clone() for name, buffer in self.model.named_buffers()}
            if detach_model_parameters else None
        )
        decode_call = _DecodedPreparedCall(self.model) if detach_model_parameters else None
        decode_parameters = (
            {f"target.{name}": value for name, value in detached_parameters.items()}
            if detached_parameters is not None else None
        )
        decode_buffers = (
            {f"target.{name}": value for name, value in detached_buffers.items()}
            if detached_buffers is not None else None
        )
        for start in range(0, int(field_coordinates.shape[0]), self.query_batch_size):
            query_chunk = field_coordinates[start : start + self.query_batch_size].unsqueeze(0)
            if prepared is None:
                call_args = (structure, query_chunk)
                call_kwargs = {
                    "interface_condition": interface_condition,
                    "local_module_params": local_params,
                    "teacher_port_tokens": teacher_ports,
                    "local_query_points": local_query,
                    "local_port_condition_mode": "predicted",
                    "mixed_teacher_ratio": 0.0,
                    "fixed_cover_plans": fixed_cover_plans,
                    "return_prepared_state": True,
                    "return_packet_inputs": self.capture_packet_inputs,
                    "cover_plan_builder": cover_plan_builder,
                }
                if detach_model_parameters:
                    assert detached_parameters is not None and detached_buffers is not None
                    output = functional_call(
                        self.model,
                        (detached_parameters, detached_buffers),
                        call_args,
                        call_kwargs,
                        strict=True,
                    )
                elif self.organizer_shadow:
                    from honf_forward_core.training.hypergraph_shadow import hard_value_soft_hypergraph_forward
                    output = hard_value_soft_hypergraph_forward(self.model, *call_args,
                        gradient_policy=self.organizer_gradient_policy, **call_kwargs)
                else:
                    output = self.model(*call_args, **call_kwargs)
                prepared = output["prepared_state"]
                first_output = output
                self.last_packet_inputs = output.get("packet_inputs")
                if self._capture_cover_plans:
                    backend_state = getattr(getattr(prepared, "prepared", None), "backend_state", None)
                    plans = backend_state.get("cover_plans") if isinstance(backend_state, dict) else None
                    if plans is None:
                        raise RuntimeError("The native P2 preparation did not apply a cover plan.")
                    self._captured_cover_plans = tuple(plans)
            else:
                if detach_model_parameters:
                    assert decode_call is not None and decode_parameters is not None and decode_buffers is not None
                    output = functional_call(
                        decode_call,
                        (decode_parameters, decode_buffers),
                        (prepared, query_chunk),
                        strict=True,
                    )
                else:
                    output = self.model.decode_prepared(prepared, query_chunk)
            output_chunks.append(output["pred_field"].squeeze(0))

        field = _physical_output(
            torch.cat(output_chunks, dim=0),
            self.normalization_stats,
            "field_mean_by_channel",
            "field_std_by_channel",
            normalize_targets=self.normalize_targets,
        )
        if first_output is None:
            raise RuntimeError("The native field preparation pass did not execute.")
        internal = _physical_output(
            first_output["pred_internal_temperature"].squeeze(0),
            self.normalization_stats,
            "internal_temperature_mean",
            "internal_temperature_std",
            normalize_targets=self.normalize_targets,
        )
        interface_values = _physical_output(
            first_output["pred_interface"].squeeze(0),
            self.normalization_stats,
            "interface_targets_mean" if "interface_targets_mean" in self.normalization_stats else "interface_target_mean",
            "interface_targets_std" if "interface_targets_std" in self.normalization_stats else "interface_target_std",
            normalize_targets=self.normalize_targets,
        )

        solid_rows_flat: list[torch.Tensor | None] = [None] * int(solid.query_features.shape[0])
        interface_rows_flat: list[torch.Tensor | None] = [None] * int(interface.query_features.shape[0])
        for slot, indices in enumerate(solid_rows):
            for offset, row_index in enumerate(indices.tolist()):
                solid_rows_flat[int(row_index)] = internal[slot, offset, :]
        for slot, indices in enumerate(interface_rows):
            for offset, row_index in enumerate(indices.tolist()):
                interface_rows_flat[int(row_index)] = interface_values[slot, offset, :]
        if any(value is None for value in solid_rows_flat + interface_rows_flat):
            raise ValueError("A material role contains rows without an active receiver slot.")
        solid_values = torch.stack([value for value in solid_rows_flat if value is not None], dim=0)
        interface_values_aligned = torch.stack(
            [value for value in interface_rows_flat if value is not None], dim=0
        )
        if solid_values.shape != (solid.query_features.shape[0], len(solid.channel_names)):
            raise ValueError("Native internal outputs do not align with typed solid receiver rows.")
        if interface_values_aligned.shape != (interface.query_features.shape[0], len(interface.channel_names)):
            raise ValueError("Native interface outputs do not align with typed port receiver rows.")

        radius = float(self.model.config.core_honf.module_radius)
        world_xy = {
            name: role_receiver_world_xy(query, design, module_radius=radius)
            for name, query in role_queries.items()
        }
        return AbsolutePrediction(
            role_values={
                "fluid_fields": field,
                "interface": interface_values_aligned,
                "solid_temperature": solid_values,
            },
            receiver_world_xy=world_xy,
        )

    def predict_with_frozen_topology(
        self,
        design: DesignInput,
        context: Mapping[str, Any],
        role_queries: Mapping[str, RoleQuery],
        *,
        fixed_cover_plans: tuple[Any, ...],
    ) -> AbsolutePrediction:
        """Use the real core permission path for one fresh continuous state."""

        return self(
            design,
            context,
            role_queries,
            fixed_cover_plans=fixed_cover_plans,
        )


__all__ = ["DifferentiableThermalOperator"]
