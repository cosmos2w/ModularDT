"""Differentiable, full-receiver Thermal design functionals.

The caller supplies the original native receiver catalogue and design masks.
This module neither owns nor changes model parameters.
"""
from __future__ import annotations

import math
import torch


def native_design_functionals(material_temperature, material_valid, pressure,
                              fluid_valid, fluid_x, domain_length_x, tau):
    """Return normalized material smooth maximum and inlet-minus-outlet pressure.

    Inputs describe one design, including its complete canonical material and
    fluid receiver sets. Reductions are in FP64 and retain autograd paths.
    """
    if not math.isfinite(float(tau)) or float(tau) <= 0:
        raise ValueError('The material smoothing scale must be positive and finite.')
    material_valid = torch.as_tensor(material_valid, device=material_temperature.device, dtype=torch.bool)
    fluid_valid = torch.as_tensor(fluid_valid, device=pressure.device, dtype=torch.bool)
    x = torch.as_tensor(fluid_x, device=pressure.device)
    if material_valid.shape != material_temperature.shape or fluid_valid.shape != pressure.shape or x.shape != pressure.shape:
        raise ValueError('Each native validity mask and coordinate must match its receivers.')
    length = float(domain_length_x)
    if not math.isfinite(length) or length <= 0:
        raise ValueError('The domain length must be positive and finite.')
    values = material_temperature[material_valid].to(torch.float64)
    left = fluid_valid & (x <= .08 * length)
    right = fluid_valid & (x >= .92 * length)
    if not values.numel() or not bool(left.any()) or not bool(right.any()):
        raise ValueError('Full material and both fluid pressure bands require valid receivers.')
    if not torch.isfinite(values).all() or not torch.isfinite(pressure[left]).all() or not torch.isfinite(pressure[right]).all():
        raise ValueError('A valid native functional receiver is nonfinite.')
    objective = float(tau) * (torch.logsumexp(values / float(tau), dim=0) - math.log(values.numel()))
    p = pressure.to(torch.float64)
    return objective, p[left].mean() - p[right].mean()
