"""One-way, organization-controlled Thermal port and local readout."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from honf_runtime.compat import safe_std_np
from .local_coupling import (LocalSurrogateCoupling, build_local_module_params_from_global,
                             update_local_module_params_from_ports)
from .native_observations import interpolate_native


def fit_port_statistics(cases):
    """Fit physical T/h moments exclusively from the supplied TRAIN cases."""
    values = [[], []]
    for case in cases:
        ports = np.asarray(case["teacher_port_tokens"])
        present = np.asarray(case["structure"]["module_present"]) > .5
        valid = np.asarray(case["interface_condition_valid_mask"]) > .5
        for channel in range(2):
            v = ports[..., 3 + channel]
            mask = present[:, None] & np.isfinite(v)
            if channel == 1:
                mask &= valid
            values[channel].append(v[mask])
    columns = [np.concatenate(parts) for parts in values]
    if any(not len(column) for column in columns):
        raise ValueError("TRAIN must contain valid outside-temperature and h observations.")
    mean = np.array([column.mean(dtype=np.float64) for column in columns], dtype=np.float32)
    std = safe_std_np(np.array([column.std(dtype=np.float64) for column in columns], dtype=np.float32))
    return {"mean": mean.tolist(), "std": std.tolist()}


class ThermalPortHead(nn.Module):
    """Predict physical outside T and nonnegative h from organized field reads."""

    def __init__(self, statistics, hidden_dim=64):
        super().__init__()
        self.register_buffer("mean", torch.tensor(statistics["mean"], dtype=torch.float32))
        self.register_buffer("std", torch.tensor(statistics["std"], dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(7, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, 2))

    def forward(self, outside, prescribed, present):
        ntheta = outside.shape[-1]
        theta = torch.arange(ntheta, device=outside.device, dtype=outside.dtype) * (2 * torch.pi / ntheta)
        geometry = torch.stack((theta, theta.cos(), theta.sin()), -1)
        geometry = geometry.view(1, 1, ntheta, 3).expand(*outside.shape, 3)
        # Prescribed heat/material values are scaled without reference to targets.
        p = torch.sign(prescribed) * torch.log1p(prescribed.abs())
        p = p[..., None, :].expand(*outside.shape, 3)
        x = torch.cat((((outside - self.mean[0]) / self.std[0])[..., None], p, geometry), -1)
        raw = self.net(x)
        temperature = outside + self.std[0] * raw[..., 0]
        h = F.softplus(raw[..., 1] + self.mean[1] / self.std[1]) * self.std[1]
        ports = torch.cat((geometry, temperature[..., None], h[..., None]), -1)
        return ports * present[..., None, None].to(ports)


class ThermalCalibratedPorts(nn.Module):
    """Inference-only raw organized outside temperature and TRAIN mean h."""

    def __init__(self, mean_h):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.0, mean_h], dtype=torch.float32))

    def forward(self, outside, prescribed, present):
        n = outside.shape[-1]
        theta = torch.arange(n, dtype=outside.dtype, device=outside.device) * (2 * torch.pi / n)
        geometry = torch.stack((theta, theta.cos(), theta.sin()), -1).view(1, 1, n, 3).expand(*outside.shape, 3)
        h = torch.ones_like(outside) * self.mean[1].to(outside)
        ports = torch.cat((geometry, outside[..., None], h[..., None]), dim=-1)
        return ports * present[..., None, None].to(ports)


def organized_outside64(adapter, prepared):
    """Read native outside receivers through the existing prepared organization."""
    adapter.validate_prepared(prepared)
    query, union, stencils, *rest = adapter._query_union(
        # Native curl stencils require at least one fluid receiver. Retaining
        # one existing receiver avoids rereading the complete primary Q set.
        prepared.scene_batch, prepared.fluid_xy[:, :1], None, 64,
    )
    shape = rest[-1]["outside"]
    output = adapter.core.decode(prepared.organization, query, intervention={"read_mode": "restricted"})
    temperature = adapter._physical_temperature(output.joint_fields[..., 3])
    outside = interpolate_native(temperature[..., None], stencils["outside"])
    outside = outside.reshape(prepared.source_present.shape[0], *shape)
    valid = stencils["outside"].valid.reshape_as(outside)
    work = {str(k): int(v) for k, v in output.work_counts.items()}
    work["native_union_rows"] = int(union.unique_counts.sum())
    return outside, valid, work


def port_loss(ports, target, present, outside_valid, h_valid, std):
    """Per-case normalized T/h squared errors, including empty-mask cases."""
    losses = []
    for channel in range(2):
        truth = target[..., 3 + channel].to(ports)
        valid = present[..., None].bool() & outside_valid.bool() & torch.isfinite(truth)
        if channel == 1:
            valid &= h_valid.bool()
        error = (ports[..., 3 + channel] - torch.nan_to_num(truth)) / std[channel].to(ports)
        losses.append((error.square() * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1))
    return tuple(losses)


class FrozenThermalLocal(nn.Module):
    """Strictly loaded local asset; frozen weights retain derivatives of inputs."""

    def __init__(self, checkpoint_path):
        super().__init__()
        self.coupling = LocalSurrogateCoupling(hidden_dim=8, local_surrogate_latent_dim=8,
                                               dropout=0.0, use_layer_norm=False)
        checkpoint = self.coupling.attach_from_checkpoint(checkpoint_path, freeze=True)
        if int(checkpoint.get("epoch", -1)) != 4975:
            raise ValueError("Expected the specified epoch-4975 local checkpoint.")
        for parameter in self.coupling.parameters():
            parameter.requires_grad_(False)
        self.coupling.eval()

    def forward(self, structure, heat, ports, local_query_points):
        present = structure["module_present"]
        params = build_local_module_params_from_global(
            heat, None, structure["material_params"], present)
        params = update_local_module_params_from_ports(params, ports, present)
        return self.coupling.call_local_surrogate(params, ports, local_query_points, present)

    def train(self, mode=True):
        super().train(mode)
        self.coupling.eval()
        return self
