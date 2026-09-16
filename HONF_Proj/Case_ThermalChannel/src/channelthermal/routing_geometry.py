"""Known channel boundary descriptors for the generic routing index.

The formal dataset supplies no validated internal barrier field. Additional
resistance is neutral; boundary descriptors are geometric input, not solver truth.
"""

from __future__ import annotations

import math

import torch

from .environment import ChannelThermalEnvironmentBuilder


class ChannelThermalRoutingGeometry:
    """Ephemeral geometry reconstructed by the case adapter on each forward."""

    feature_dim = 7

    def __init__(self, domain_length_x: float, domain_length_y: float, module_radius: float):
        values = (domain_length_x, domain_length_y, module_radius)
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
            raise ValueError("Channel routing dimensions and radius must be finite and positive.")
        self.domain_length_x = float(domain_length_x)
        self.domain_length_y = float(domain_length_y)
        self.bounds = ((0.0, self.domain_length_x), (0.0, self.domain_length_y))
        self.length_scale = (4.0 * float(module_radius),) * 2

    def features(self, points: torch.Tensor) -> torch.Tensor:
        if points.shape[-1] != 2:
            raise ValueError("ThermalChannel routing requires two-dimensional coordinates.")
        boundary = ChannelThermalEnvironmentBuilder.query_features(
            points, domain_length_x=self.domain_length_x, domain_length_y=self.domain_length_y,
        )
        y = points[..., 1:2]
        centerline = (1.0 - (y - 0.5 * self.domain_length_y).abs()
                      / (0.5 * self.domain_length_y)).clamp(0.0, 1.0)
        return torch.cat((boundary, centerline), dim=-1)

    def resistance(self, a: torch.Tensor, b: torch.Tensor, relation_type: str) -> torch.Tensor:
        del relation_type
        # Retain a differentiable identity with respect to both endpoints.
        return (a - b).sum(dim=-1) * 0.0
