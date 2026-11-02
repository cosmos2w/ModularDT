"""Differentiable native Thermal observations for frozen design studies.

The caller owns the record and the design tensor.  Every call prepares a new
scene, so moved receivers and contextual organization follow the design.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ThermalDesignEvaluation:
    fields: dict[str, torch.Tensor]
    objective: torch.Tensor
    sampled_max: torch.Tensor
    pressure_drop: torch.Tensor
    valid_material_count: int
    prepared: Any
    work: dict[str, int]
    core_output: Any


class ThermalDesignEvaluator:
    """Evaluate native record receivers, with live centers and fixed controls."""

    def __init__(self, model, record):
        self.model = model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.record = record
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.ids = tuple(m.module_id for m in record.design.modules)
        roles = record.output.roles
        fluid = np.asarray(roles["fluid_fields"].query_features)[:, :2]
        self.fluid = self._tensor(fluid)[None]
        material = roles["solid_temperature"]
        material_ids = np.asarray(material.receiver_module_ids)
        rows = [np.flatnonzero(material_ids == module_id) for module_id in self.ids]
        if not rows or len({len(row) for row in rows}) != 1 or not len(rows[0]):
            raise ValueError("Native material receivers must have equal nonzero counts per original module.")
        self.local = self._tensor(np.stack([np.asarray(material.query_features)[row, :2] for row in rows]))[None]
        interface = roles["interface"]
        counts = [np.count_nonzero(np.asarray(interface.receiver_module_ids) == i) for i in self.ids]
        if len(set(counts)) != 1 or counts[0] != 64:
            raise ValueError("Expected the original 64 interface angles for each module.")
        self.tau = 0.05 * float(model.internal_temperature_std[0].detach().cpu())
        if self.tau <= 0 or not math.isfinite(self.tau):
            raise ValueError("Invalid checkpoint internal temperature scale.")

    def _tensor(self, value):
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            value = value.copy()
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def evaluate_design(self, centers: torch.Tensor, *, heat=None, read_mode="full",
                        context_message_removal=None, access_intervention=None):
        if centers.shape != (len(self.ids), 2):
            raise ValueError("Centers must preserve original module order and have shape [M,2].")
        if centers.device != self.device or centers.dtype != self.dtype:
            raise ValueError("Centers must use the frozen model's device and dtype.")
        if read_mode not in ("full", "restricted"):
            raise ValueError("Explicit read mode must be full or restricted.")
        context = self.record.context.values
        structure = {
            "module_centers": centers[None],
            "module_present": self._tensor([float(m.active) for m in self.record.design.modules])[None],
            "module_source_ids": self.ids,
            "material_params": self._tensor([context[k] for k in ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius")])[None],
            "heat_powers": (self._tensor([m.heating for m in self.record.design.modules]) if heat is None else heat)[None],
        }
        for key in ("re", "u_in", "domain_length_x", "domain_length_y", "t_in", "t_wall"):
            if key in context:
                structure[key] = self._tensor([context[key]])[None]
        prepared = self.model.prepare_native(structure, self.fluid, local_query_points=self.local, ntheta=64, **(
            {"context_message_removal": context_message_removal} if context_message_removal else {}))
        native, work, core = self.model.apply_native_with_work_counts(
            prepared, structure["heat_powers"], intervention={"read_mode": read_mode, **(
                {"access_intervention": access_intervention} if access_intervention else {})})
        values = native["pred_internal_temperature"][..., 0]
        valid = prepared.stencils["material"].valid.reshape(values.shape) & prepared.scene_batch.module_mask[..., None]
        selected = values[valid]
        if selected.numel() == 0:
            raise ValueError("No valid material receiver remains at this design.")
        objective = self.tau * (torch.logsumexp(selected / self.tau, dim=0) - math.log(selected.numel()))
        # Native cell-centre pressure bands and geometry-dependent fluid mask.
        xy = self.fluid[0]
        radius = float(context["module_radius"])
        fluid = ((xy[:, None] - centers[None]).square().sum(-1) >= radius ** 2).all(-1)
        fluid = fluid & prepared.stencils["fluid"].valid[0]
        left = fluid & (xy[:, 0] <= 0.08 * float(context["domain_length_x"]))
        right = fluid & (xy[:, 0] >= 0.92 * float(context["domain_length_x"]))
        if not bool(left.any()) or not bool(right.any()):
            raise ValueError("Native pressure band has no valid fluid receiver.")
        pressure = native["pred_field"][0, :, 2]
        return ThermalDesignEvaluation(native, objective, selected.max(), pressure[left].mean() - pressure[right].mean(),
                                       selected.numel(), prepared, work, core)
