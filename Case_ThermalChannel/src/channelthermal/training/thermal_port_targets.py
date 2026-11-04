"""Native 64-angle teacher-port target construction."""
from dataclasses import dataclass, fields, replace
import numpy as np
import torch
from ..native_observations import _column, native_cell_center_stencil
from .content_hypergraph_task import ThermalContentTargets


@dataclass(frozen=True)
class ThermalPortTargets(ThermalContentTargets):
    teacher_ports: torch.Tensor | None = None
    teacher_h_valid: torch.Tensor | None = None
    outside_valid: torch.Tensor | None = None


def with_port_targets(self, batch):
    width = batch.scene_inputs.structure["module_present"].shape[1]
    cases = [self._port_cases[case_id] for case_id in batch.targets.case_ids]
    ports = torch.as_tensor(np.stack([case["teacher_port_tokens"][:width] for case in cases]),
                            device=self.device)
    valid = torch.as_tensor(np.stack([case["interface_condition_valid_mask"][:width] for case in cases]),
                            device=self.device)
    structure = batch.scene_inputs.structure
    centers = structure["module_centers"]
    lengths = torch.cat((_column(structure, "domain_length_x", centers, 12.0),
                         _column(structure, "domain_length_y", centers, 6.0)), -1)
    radius = structure["material_params"][:, 5:6]
    delta = torch.minimum((lengths / lengths.new_tensor([self.model.nx, self.model.ny])).amin(-1, keepdim=True),
                          .15 * radius)
    theta = torch.arange(64, device=centers.device, dtype=centers.dtype) * (2 * torch.pi / 64)
    normals = torch.stack((theta.cos(), theta.sin()), -1)
    points = centers[:, :, None] + (radius + delta)[:, None, :, None] * normals[None, None]
    outside_valid = native_cell_center_stencil(points, lengths, nx=self.model.nx,
                                               ny=self.model.ny).valid.reshape(ports.shape[:-1])
    base = {field.name: getattr(batch.targets, field.name) for field in fields(ThermalContentTargets)}
    return replace(batch, targets=ThermalPortTargets(**base, teacher_ports=ports,
                   teacher_h_valid=valid, outside_valid=outside_valid))
