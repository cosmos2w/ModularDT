"""Opt-in source-response ThermalChannel adapter with native grid extraction.

The response context contains geometry and prescribed coefficients only.
Heating is supplied to the prepared operator separately, in physical units.
No Stage-A, reference generator or retained thermal wrapper is executed here.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from honf_forward_core.interface_fields.source_response_operator import SourceResponseOperator

SOURCE_RESPONSE_ID = "thermal_source_response_v1"
SOURCE_RESPONSE_CAPABILITY = "channelthermal_shared_grid_affine_source_v1"
SOURCE_WIDTH, CONTEXT_WIDTH, ENVIRONMENT_WIDTH = 8, 14, 8
FIELD_ORDER = ("u", "v", "p", "omega", "temperature")
CONTEXT_KEYS = (
    "module_centers",
    "module_present",
    "material_params",
    "re",
    "u_in",
    "domain_length_x",
    "domain_length_y",
    "module_source_ids",
    "t_in",
    "t_wall",
)


@dataclass
class NativeStencil:
    indices: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor


@dataclass
class PreparedNativeResponse:
    response: Any
    context: Any
    owner: int
    structure: dict
    input_snapshot: dict
    stencils: dict[str, NativeStencil]
    grid_indices: torch.Tensor
    role_shapes: dict[str, tuple]
    source_present: torch.Tensor
    theta: torch.Tensor
    delta: torch.Tensor
    interface_conductivity: torch.Tensor
    receiver_snapshots: tuple


class ThermalSourceResponse(nn.Module):
    """One common grid-temperature operator and physical role extraction.

    The zero offset is source-qualified for the active zero inlet/wall and
    zero initial temperature benchmark. A nonzero boundary capability requires
    another explicit adapter, not a changed normalized mean.
    """

    def __init__(
        self,
        core_config=None,
        *,
        nx=128,
        ny=64,
        environment_nx=24,
        environment_ny=8,
        forcing_scale=1.0,
        h_effective_eps=1.0e-3,
        h_effective_max=1.0e4,
    ):
        super().__init__()
        if nx < 2 or ny < 2 or not np.isfinite(forcing_scale) or forcing_scale <= 0:
            raise ValueError("Native grid and frozen forcing scale must be positive.")
        config = dict(core_config or {})
        if "forcing_scale" in config and float(config["forcing_scale"]) != float(forcing_scale):
            raise ValueError("Core and adapter frozen forcing scales differ; physical heat units must agree.")
        for name, width in (
            ("source_width", SOURCE_WIDTH),
            ("context_width", CONTEXT_WIDTH),
            ("environment_width", ENVIRONMENT_WIDTH),
        ):
            if name in config and config[name] != width:
                raise ValueError(f"Thermal adapter requires {name}={width}.")
            config[name] = width
        if config.get("spatial_dim", 2) != 2 or config.get("output_width", 1) != 1:
            raise ValueError("Thermal response is a 2-D shared scalar-temperature kernel.")
        if not config.get("zero_offset", True):
            raise ValueError("Active zero-boundary thermal capability requires zero offset.")
        config.update(spatial_dim=2, output_width=1, zero_offset=True, forcing_scale=float(forcing_scale))
        self.core = SourceResponseOperator(**config)
        self.core_config = dict(self.core.config)
        self.nx, self.ny = int(nx), int(ny)
        self.environment_nx, self.environment_ny = int(environment_nx), int(environment_ny)
        self.forcing_scale = float(forcing_scale)
        self.h_effective_eps, self.h_effective_max = float(h_effective_eps), float(h_effective_max)

    def adapter_config(self):
        return {
            "nx": self.nx,
            "ny": self.ny,
            "environment_nx": self.environment_nx,
            "environment_ny": self.environment_ny,
            "forcing_scale": self.forcing_scale,
            "h_effective_eps": self.h_effective_eps,
            "h_effective_max": self.h_effective_max,
        }

    @staticmethod
    def _column(structure, name, like, fallback):
        value = structure.get(name)
        return (
            like.new_full((like.shape[0], 1), fallback) if value is None else value.reshape(like.shape[0], 1).to(like)
        )

    def context_tensors(self, structure):
        """Build the actual input whitelist; never inspect a heat/target tensor."""
        centers = structure["module_centers"].float()
        for name in ("t_in", "t_wall"):
            if name in structure:
                boundary = torch.as_tensor(structure[name], device=centers.device)
                if not bool(torch.isfinite(boundary).all()) or not bool((boundary == 0).all()):
                    raise ValueError("This affine source capability requires zero inlet/wall temperature.")
        present = structure["module_present"].to(centers)
        if centers.ndim != 3 or centers.shape[-1] != 2 or present.shape != centers.shape[:2]:
            raise ValueError("Thermal geometry requires [B,M,2] centers and [B,M] mask.")
        material = structure["material_params"].to(centers)
        if material.ndim == 1:
            material = material[None].expand(centers.shape[0], -1)
        if material.shape != (centers.shape[0], 6):
            raise ValueError("Thermal material order is [nu,solid_alpha,fluid_alpha,solid_k,fluid_k,radius].")
        if not bool(torch.isfinite(material).all()) or not bool((material > 0).all()):
            raise ValueError("Native thermal coefficients and radius must be positive and finite.")
        lengths = torch.cat(
            (
                self._column(structure, "domain_length_x", centers, 12.0),
                self._column(structure, "domain_length_y", centers, 6.0),
            ),
            -1,
        )
        if not bool(torch.isfinite(lengths).all()) or not bool((lengths > 0).all()):
            raise ValueError("Domain lengths must be positive and finite.")
        count = present.sum(1, keepdim=True)
        radius = material[:, 5:6]
        descriptors = material[:, 1:5] * centers.new_tensor([100.0, 100.0, 1.0, 1.0])
        sources = torch.cat(
            (
                centers / lengths[:, None],
                (radius / lengths)[:, None].expand(-1, centers.shape[1], -1),
                descriptors[:, None].expand(-1, centers.shape[1], -1),
            ),
            -1,
        )
        context = torch.cat(
            (
                self._column(structure, "re", centers, 0.0) / 100.0,
                self._column(structure, "u_in", centers, 0.0),
                material[:, :5] * centers.new_tensor([100.0, 100.0, 100.0, 1.0, 1.0]),
                radius,
                lengths / centers.new_tensor([12.0, 6.0]),
                count / 12.0,
                count / lengths.prod(-1, keepdim=True),
                centers.new_zeros(centers.shape[0], 2),
            ),
            -1,
        )
        axis_x = (
            torch.arange(self.environment_nx, device=centers.device, dtype=centers.dtype) + 0.5
        ) / self.environment_nx
        axis_y = (
            torch.arange(self.environment_ny, device=centers.device, dtype=centers.dtype) + 0.5
        ) / self.environment_ny
        yy, xx = torch.meshgrid(axis_y, axis_x, indexing="ij")
        env_normal = torch.stack((xx, yy), -1).reshape(1, -1, 2).expand(centers.shape[0], -1, -1)
        env_coords = env_normal * lengths[:, None]
        distance = torch.linalg.vector_norm(env_coords[:, :, None] - centers[:, None], dim=-1)
        live_distance = torch.where(present[:, None] > 0.5, distance, distance.new_full((), float("inf")))
        nearest = live_distance.amin(-1, keepdim=True) / radius[:, None]
        nearest = torch.where(torch.isfinite(nearest), nearest, nearest.new_full((), 10.0))
        occupancy = (torch.exp(-distance.square() / radius[:, None].square()) * present[:, None]).sum(-1, keepdim=True)
        boundary_distance = torch.cat((env_normal, 1.0 - env_normal), -1)
        env_tokens = torch.cat((env_normal, nearest - 1.0, occupancy, boundary_distance), -1)
        source_lengths = radius.expand(-1, centers.shape[1])
        source_measures = torch.pi * source_lengths.square() * present
        env_measures = lengths.prod(-1, keepdim=True).expand(-1, env_coords.shape[1]) / env_coords.shape[1]
        source_ids = structure.get("module_source_ids")
        if source_ids is not None and not torch.is_tensor(source_ids):
            # Typed string IDs are kept in the adapter catalogue; the generic
            # tensor operator receives its exact ordered slot index.
            if len(source_ids) != centers.shape[1]:
                raise ValueError("Source ID catalogue does not match active/padded slots.")
            source_ids = torch.arange(centers.shape[1], device=centers.device)[None].expand(centers.shape[0], -1)
        return {
            "sources": sources,
            "context": context,
            "centers": centers,
            "present": present,
            "lengths": lengths,
            "source_lengths": source_lengths,
            "source_measures": source_measures,
            "environment_tokens": env_tokens,
            "environment_coords": env_coords,
            "environment_present": present.new_ones(present.shape[0], env_coords.shape[1]),
            "environment_measures": env_measures,
            "source_ids": source_ids,
        }

    def prepare_context(self, structure):
        return self.core.prepare_context(**self.context_tensors(structure))

    def native_stencil(self, xy, lengths):
        """Geometry-only equivalent of the generator's cell-center bilinear_sample."""
        shape = xy.shape
        flat = xy.reshape(shape[0], -1, 2)
        if not bool(torch.isfinite(flat).all()):
            raise ValueError("Native receivers must be finite physical coordinates.")
        grid = flat / (lengths[:, None] / flat.new_tensor([self.nx, self.ny])) - 0.5
        # Exact native cell centers may move a few ulps during FP32 division.
        # Snap only near integers, retaining the original coordinate VJP.
        rounded = grid.round()
        near = (grid - rounded).abs() <= 8 * torch.finfo(grid.dtype).eps * max(self.nx, self.ny)
        grid = grid + torch.where(near, rounded - grid, grid.new_zeros(())).detach()
        valid = ((grid >= 0) & (grid <= grid.new_tensor([self.nx - 1, self.ny - 1]))).all(-1)
        clipped = torch.minimum(torch.maximum(grid, grid.new_zeros(())), grid.new_tensor([self.nx - 1, self.ny - 1]))
        low = clipped.floor().long()
        high = torch.minimum(low + 1, low.new_tensor([self.nx - 1, self.ny - 1]))
        weight = clipped - low.to(clipped)
        wx, wy = weight.unbind(-1)
        ix, iy = low.unbind(-1)
        jx, jy = high.unbind(-1)
        indices = torch.stack((iy * self.nx + ix, iy * self.nx + jx, jy * self.nx + ix, jy * self.nx + jx), -1)
        weights = torch.stack(((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy), -1)
        return NativeStencil(indices, weights, valid)

    def prepare_native(self, structure, fluid_xy, *, local_query_points=None, ntheta=64, chunk_size=512):
        tensors = self.context_tensors(structure)
        context = self.core.prepare_context(**tensors)
        centers, present, lengths = tensors["centers"], tensors["present"], tensors["lengths"]
        radius = tensors["source_lengths"][:, :1]
        delta = torch.minimum((lengths / lengths.new_tensor([self.nx, self.ny])).amin(-1, keepdim=True), 0.15 * radius)
        theta = torch.arange(int(ntheta), device=centers.device, dtype=centers.dtype) * (2 * torch.pi / int(ntheta))
        normals = torch.stack((theta.cos(), theta.sin()), -1)
        roles = {
            "fluid": fluid_xy.to(centers),
            "surface": centers[:, :, None] + radius[:, None, :, None] * normals[None, None],
            "outside": centers[:, :, None] + (radius + delta)[:, None, :, None] * normals[None, None],
        }
        if local_query_points is not None:
            local = local_query_points.to(centers)
            if local.ndim == 3:
                local = local[:, None].expand(-1, centers.shape[1], -1, -1)
            if local.ndim != 4 or local.shape[:2] != centers.shape[:2]:
                raise ValueError("Material queries must be [B,Ql,2] or [B,M,Ql,2].")
            roles["material"] = centers[:, :, None] + radius[:, None, :, None] * local
        stencils = {name: self.native_stencil(xy, lengths) for name, xy in roles.items()}
        # Deduplicate within each case, not across unrelated minibatch layouts.
        # Zero-weight neighbours are unnecessary for ordinary training, but
        # remain necessary when interpolation-weight derivatives are live.
        # Keeping all four in that scope preserves receiver/geometry VJPs.
        unions = []
        for batch_index in range(centers.shape[0]):
            selected = []
            for name, s in stencils.items():
                needed = s.weights[batch_index] != 0
                if s.weights.requires_grad:
                    needed = torch.ones_like(needed)
                if name != "fluid":
                    active = present[batch_index, :, None].expand(*roles[name].shape[1:-1]).reshape(-1)
                    needed = needed & active[:, None].bool()
                needed = needed & s.valid[batch_index, :, None]
                selected.append(s.indices[batch_index][needed])
            union = torch.unique(torch.cat(selected), sorted=True)
            if union.numel() == 0:
                union = torch.zeros(1, dtype=torch.long, device=centers.device)
            unions.append(union)
            for s in stencils.values():
                indices = s.indices[batch_index]
                mapped = torch.searchsorted(union, indices)
                exists = (mapped < union.numel()) & (union[mapped.clamp_max(union.numel() - 1)] == indices)
                s.indices[batch_index] = torch.where(exists, mapped, torch.zeros_like(mapped))
        width = max(u.numel() for u in unions)
        union = torch.stack([torch.cat((u, u[:1].expand(width - u.numel()))) for u in unions])
        xy = torch.stack((union % self.nx + 0.5, union // self.nx + 0.5), -1).to(centers)
        grid_receivers = xy * (lengths[:, None] / lengths.new_tensor([self.nx, self.ny]))
        response = self.core.prepare_receivers(context, grid_receivers, receiver_ids=union, chunk_size=chunk_size)
        material = structure["material_params"].to(centers)
        if material.ndim == 1:
            material = material[None].expand(centers.shape[0], -1)
        conductivity = (
            2 * material[:, 3:4] * material[:, 4:5] / (material[:, 3:4] + material[:, 4:5]).clamp_min(1.0e-12)
        )
        snapshot = {
            k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in structure.items()
            if k in CONTEXT_KEYS
        }
        return PreparedNativeResponse(
            response,
            context,
            id(self),
            structure,
            snapshot,
            stencils,
            union,
            {k: v.shape[1:-1] for k, v in roles.items()},
            present,
            theta,
            delta,
            conductivity,
            tuple((v, v.detach().clone()) for v in (fluid_xy, local_query_points) if torch.is_tensor(v)),
        )

    @staticmethod
    def _interpolate(values, stencil):
        # values [B,N,O] or explicit kernel [B,N,M,O]
        batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
        gathered = values[batch, stencil.indices]
        weights = stencil.weights.reshape(*stencil.weights.shape, *([1] * (values.ndim - 2)))
        valid = stencil.valid.reshape(*stencil.valid.shape, *([1] * (values.ndim - 2)))
        return (gathered * weights).sum(2) * valid

    def validate_prepared(self, prepared, structure=None):
        if not isinstance(prepared, PreparedNativeResponse) or prepared.owner != id(self):
            raise ValueError("Native response preparation belongs to another adapter/request.")
        if any(not torch.equal(value, snapshot) for value, snapshot in prepared.receiver_snapshots):
            raise ValueError("Native receiver catalogue changed; rebuild the response operator.")
        structure = prepared.structure if structure is None else structure
        current = {k: v for k, v in structure.items() if k in CONTEXT_KEYS}
        equal = lambda a, b: (
            torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a, b)
            if torch.is_tensor(a) or torch.is_tensor(b)
            else a == b
        )
        if set(current) != set(prepared.input_snapshot) or any(
            not equal(current[k], v) for k, v in prepared.input_snapshot.items()
        ):
            raise ValueError("Geometry/context changed; rebuild the response operator.")

    def apply_native(self, prepared, physical_heat, *, increment=False, compression=None):
        self.validate_prepared(prepared)
        fn = self.core.apply_increment if increment else self.core.apply_forcing
        grid_values = fn(
            prepared.response,
            physical_heat.to(prepared.source_present),
            **({"compression": compression} if increment else {}),
        )
        role = {
            k: self._interpolate(grid_values, s).reshape(grid_values.shape[0], *prepared.role_shapes[k], -1)
            for k, s in prepared.stencils.items()
        }
        surface = role["surface"] * prepared.source_present[:, :, None, None]
        outside = role["outside"] * prepared.source_present[:, :, None, None]
        q = -prepared.interface_conductivity[:, None, :, None] / prepared.delta[:, None, :, None] * (outside - surface)
        jump = surface - outside
        sign = torch.where(jump < 0, -torch.ones_like(jump), torch.ones_like(jump))
        denom = torch.where(jump.abs() < self.h_effective_eps, sign * self.h_effective_eps, jump)
        raw_h = q / denom
        h = torch.nan_to_num(raw_h, nan=0.0, posinf=self.h_effective_max, neginf=0.0).clamp(0, self.h_effective_max)
        valid = (
            (jump.abs() >= self.h_effective_eps)
            & torch.isfinite(q)
            & torch.isfinite(raw_h)
            & (raw_h >= 0)
            & (raw_h <= self.h_effective_max)
        )
        theta = prepared.theta
        angular = torch.stack((theta, theta.cos(), theta.sin()), -1)[None, None].expand(
            surface.shape[0], surface.shape[1], -1, -1
        )
        output = {
            "fluid_temperature": role["fluid"],
            "pred_interface": torch.cat((surface, q), -1),
            "pred_port_condition": torch.cat((angular, outside, h), -1) * prepared.source_present[:, :, None, None],
            "h_proxy": q.abs() / (jump.abs() + 1.0e-6),
            "h_effective_valid_mask": valid[..., 0] & prepared.source_present[:, :, None].bool(),
            "initial_port_status": "not_applicable_no_port_refinement_trajectory",
            "native_neural_receiver_rows": int(prepared.grid_indices.numel()),
        }
        if "material" in role:
            output["pred_internal_temperature"] = role["material"] * prepared.source_present[:, :, None, None]
            output["module_material_peak"] = output["pred_internal_temperature"][..., 0].amax(-1)
        return output

    def export_native_kernels(self, prepared):
        self.validate_prepared(prepared)
        grid = self.core.export_response_operator(prepared.response)
        kernel = grid["kernel"] if "kernel" in grid else grid["K"]
        roles = {
            k: self._interpolate(kernel, s).reshape(kernel.shape[0], *prepared.role_shapes[k], *kernel.shape[2:])
            for k, s in prepared.stencils.items()
        }
        roles["q_normal"] = (
            -prepared.interface_conductivity[:, None, :, None, None]
            / prepared.delta[:, None, :, None, None]
            * (roles["outside"] - roles["surface"])
        )
        return roles

    def forward(self, structure, physical_heat, fluid_xy, **kwargs):
        return self.apply_native(self.prepare_native(structure, fluid_xy, **kwargs), physical_heat)


class SourceResponseThermalModel(nn.Module):
    """New thermal response core plus exactly one frozen D-sep flow reader."""

    def __init__(self, thermal, flow, normalization_stats):
        super().__init__()
        if flow.policy != "D-sep":
            raise ValueError("New response composition requires the D-sep flow policy.")
        self.thermal, self.flow = thermal, flow
        self.flow.requires_grad_(False).eval()
        self.normalization_stats = normalization_stats

    def train(self, mode=True):
        super().train(mode)
        self.flow.eval()
        return self

    def prepare_native(self, structure, fluid_xy, **kwargs):
        return {
            "thermal": self.thermal.prepare_native(structure, fluid_xy, **kwargs),
            "flow": self.flow.prepare_flow(structure),
            "query_xy": fluid_xy,
            "query_snapshot": fluid_xy.detach().clone(),
        }

    def apply_native(self, prepared, physical_heat):
        if not torch.equal(prepared["query_xy"], prepared["query_snapshot"]):
            raise ValueError("Receiver catalogue changed; rebuild the complete prepared response.")
        output = self.thermal.apply_native(prepared["thermal"], physical_heat)
        flow = self.flow.read_flow(prepared["flow"], prepared["query_xy"])
        mean = flow.new_tensor(self.normalization_stats["field_mean_by_channel"])[:4]
        std = flow.new_tensor(self.normalization_stats["field_std_by_channel"])[:4]
        output["pred_field"] = torch.cat((flow * std + mean, output["fluid_temperature"]), -1)
        return output

    def predict_native_sample(self, sample, device=None):
        """Physical raw sample in; physical numpy predictions out; no target read."""
        device = torch.device(device) if device is not None else next(self.parameters()).device
        structure = {
            k: (
                copy.deepcopy(v)
                if k == "module_source_ids" and not torch.is_tensor(v) and not isinstance(v, np.ndarray)
                else torch.as_tensor(v, device=device, dtype=torch.float32)[None]
            )
            for k, v in sample["structure"].items()
            if k in CONTEXT_KEYS
        }
        xy = np.stack((np.asarray(sample["x_grid"]).reshape(-1), np.asarray(sample["y_grid"]).reshape(-1)), -1)
        fluid = torch.as_tensor(xy, device=device, dtype=torch.float32)[None]
        local = torch.as_tensor(sample["module_internal_query_points"], device=device, dtype=torch.float32)
        if local.ndim == 2:
            local = local[None]
        ports = int(sample["interface_condition"].shape[-2])
        heat = torch.as_tensor(sample["structure"]["heat_powers"], device=device, dtype=torch.float32)[None]
        with torch.no_grad():
            output = self.apply_native(
                self.prepare_native(structure, fluid, local_query_points=local, ntheta=ports), heat
            )
        ny, nx = np.asarray(sample["x_grid"]).shape
        prediction = {
            "pred_field_grid": output["pred_field"][0].cpu().numpy().reshape(ny, nx, 5),
            "pred_interface": output["pred_interface"][0].cpu().numpy(),
            "pred_internal_temperature": output["pred_internal_temperature"][0, ..., 0].cpu().numpy(),
            "pred_port_condition": output["pred_port_condition"][0].cpu().numpy(),
            "initial_port_status": output["initial_port_status"],
        }
        return prediction

    def prepare_record(self, record, device=None, *, chunk_size=512):
        """Prepare exact typed receivers from input metadata, never role values."""
        device = torch.device(device) if device is not None else next(self.parameters()).device
        modules = record.design.modules
        context = record.context.values
        tensor = lambda value: torch.as_tensor(value, device=device, dtype=torch.float32)
        centers = tensor([module.position_xy for module in modules])[None]
        present = tensor([float(module.active) for module in modules])[None]
        material = tensor(
            [context[k] for k in ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius")]
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
        interface_rows = [np.flatnonzero(np.asarray(roles["interface"].receiver_module_ids) == mid) for mid in ids]
        material_rows = [
            np.flatnonzero(np.asarray(roles["solid_temperature"].receiver_module_ids) == mid) for mid in ids
        ]
        if len({len(x) for x in interface_rows}) != 1 or len({len(x) for x in material_rows}) != 1:
            raise ValueError("Native typed roles require equal per-module receiver counts.")
        ntheta = len(interface_rows[0])
        expected = np.arange(ntheta) * 2 * np.pi / ntheta
        for rows in interface_rows:
            query = np.asarray(roles["interface"].query_features)[rows]
            if not np.allclose(query[:, 0], expected, rtol=0, atol=2.0e-6):
                raise ValueError("Native port receivers must use original uniform angular ordering.")
        local = tensor(
            np.stack([np.asarray(roles["solid_temperature"].query_features)[rows, :2] for rows in material_rows])
        )[None]
        prepared = self.prepare_native(structure, fluid, local_query_points=local, ntheta=ntheta, chunk_size=chunk_size)
        prepared.update(
            record_source_ids=ids,
            interface_rows=interface_rows,
            material_rows=material_rows,
            interface_row_count=len(roles["interface"].query_features),
            material_row_count=len(roles["solid_temperature"].query_features),
            module_physical_heat=tensor([module.heating for module in modules])[None],
        )
        return prepared

    def apply_record(self, prepared, physical_heat=None):
        heat = prepared["module_physical_heat"] if physical_heat is None else physical_heat
        heat = torch.as_tensor(
            heat, device=prepared["thermal"].source_present.device, dtype=prepared["thermal"].source_present.dtype
        )
        if heat.ndim == 1:
            heat = heat[None]
        output = self.apply_native(prepared, heat)
        interface = output["pred_interface"].new_empty(prepared["interface_row_count"], 2)
        material = output["pred_internal_temperature"].new_empty(prepared["material_row_count"], 1)
        for index, rows in enumerate(prepared["interface_rows"]):
            interface[torch.as_tensor(rows, device=interface.device)] = output["pred_interface"][0, index]
        for index, rows in enumerate(prepared["material_rows"]):
            material[torch.as_tensor(rows, device=material.device)] = output["pred_internal_temperature"][0, index]
        return {"fluid_fields": output["pred_field"][0], "interface": interface, "solid_temperature": material}

    def apply_record_increment(self, prepared, delta_heat, *, compression=None):
        delta_heat = torch.as_tensor(
            delta_heat, device=prepared["thermal"].source_present.device, dtype=prepared["thermal"].source_present.dtype
        )
        if delta_heat.ndim == 1:
            delta_heat = delta_heat[None]
        output = self.thermal.apply_native(prepared["thermal"], delta_heat, increment=True, compression=compression)
        interface = output["pred_interface"].new_empty(prepared["interface_row_count"], 2)
        material = output["pred_internal_temperature"].new_empty(prepared["material_row_count"], 1)
        for index, rows in enumerate(prepared["interface_rows"]):
            interface[torch.as_tensor(rows, device=interface.device)] = output["pred_interface"][0, index]
        for index, rows in enumerate(prepared["material_rows"]):
            material[torch.as_tensor(rows, device=material.device)] = output["pred_internal_temperature"][0, index]
        temperature = output["fluid_temperature"][0]
        # The flow input signature omits heating; its exact increment is zero.
        fluid = torch.cat((temperature.new_zeros(temperature.shape[0], 4), temperature), -1)
        return {"fluid_fields": fluid, "interface": interface, "solid_temperature": material}

    def predict_record(self, record, device=None):
        with torch.no_grad():
            output = self.apply_record(self.prepare_record(record, device))
        return {k: v.cpu().numpy() for k, v in output.items()}


def load_source_response_model(path, device="cpu", checkpoint=None):
    """Distinct identity; load one learned flow state and never a thermal parent."""
    from channelthermal.dependency_flow import ThermalFlowReader
    from channelthermal.training.checkpoints import _file_sha256
    from honf_runtime.checkpoints import validate_checkpoint_identity
    from honf_runtime.compat import load_trusted_checkpoint

    saved = checkpoint if checkpoint is not None else load_trusted_checkpoint(path, map_location="cpu")
    validate_checkpoint_identity(saved, case_id="ThermalChannel", model_family="honf_forward", workflow="forward")
    if (
        saved.get("source_response_identity") != SOURCE_RESPONSE_ID
        or saved.get("case_capability") != SOURCE_RESPONSE_CAPABILITY
    ):
        raise ValueError("Checkpoint does not declare the native affine source-response capability.")
    if saved.get("channel_order") != list(FIELD_ORDER):
        raise ValueError("Source-response checkpoint field order differs.")
    config = saved["source_response_config"]
    if "fit_identity" in saved:
        identity = saved["fit_identity"]
        mode, recipe = identity.get("mode"), identity.get("recipe", {})
        if (
            mode not in ("direct", "group")
            or config["core"].get("mode") != mode
            or recipe.get("core_configs", {}).get(mode) != config["core"]
            or recipe.get("adapter_config") != config.get("adapter", {})
        ):
            raise ValueError("Source-response core/adapter/mode differ from the sealed fit identity recipe.")
    flow_path = Path(saved["flow_checkpoint"])
    if _file_sha256(flow_path) != saved["flow_checkpoint_sha256"]:
        raise ValueError("Frozen flow checkpoint identity changed.")
    flow_saved = load_trusted_checkpoint(flow_path, map_location="cpu")
    from channelthermal.dependency_flow import CASE_CAPABILITY, DEPENDENCY_ID

    if (
        flow_saved.get("dependency_policy") != "D-sep"
        or flow_saved.get("dependency_identity") != DEPENDENCY_ID
        or flow_saved.get("case_capability") != CASE_CAPABILITY
    ):
        raise ValueError("Frozen flow partner must declare the audited D-sep dependency identity/capability.")
    if set(saved["global_normalization_stats"]) != set(flow_saved["global_normalization_stats"]):
        raise ValueError("Thermal/flow TRAIN normalization key sets differ.")
    for key, value in saved["global_normalization_stats"].items():
        if not np.array_equal(np.asarray(value), np.asarray(flow_saved["global_normalization_stats"][key])):
            raise ValueError("Thermal/flow TRAIN normalization metadata differs.")
    left = saved["train_config"]["dataset"]
    right = flow_saved["train_config"]["dataset"]
    expected_manifest = "933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044"
    for dataset in (left, right):
        subset = dataset.get("development_subset", {})
        if (
            dataset.get("development_manifest_sha256") != expected_manifest
            or subset.get("manifest_sha256") != expected_manifest
        ):
            raise ValueError("Source response requires the literal fixed25_v1 primary development membership.")
        for split, count in (("train", 150), ("test", 22)):
            ids = subset.get("partitions", {}).get(split, {}).get("case_ids", ())
            if len(ids) != count or len(set(ids)) != count:
                raise ValueError("Source response requires bound fixed25_v1 150/22 primary case IDs.")
    for key in ("development_manifest", "development_manifest_sha256", "development_subset", "dataset_fingerprint"):
        if left.get(key) != right.get(key):
            raise ValueError("Thermal/flow primary development membership differs.")
    flow = ThermalFlowReader("D-sep", flow_saved["flow_reader_config"])
    flow.load_state_dict(flow_saved["flow_state_dict"], strict=True)
    thermal = ThermalSourceResponse(config["core"], **config.get("adapter", {}))
    thermal.load_state_dict(saved["thermal_state_dict"], strict=True)
    model = SourceResponseThermalModel(thermal, flow, saved["global_normalization_stats"]).to(device).eval()
    return model, saved
