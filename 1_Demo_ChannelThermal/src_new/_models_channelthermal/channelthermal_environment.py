"""CHANNELTHERMAL-SPECIFIC environment feature builder.

Inputs are a batch size, domain lengths, and environment-token grid counts.
Outputs are physical environment coordinates plus generic feature tensors for
the CORE HONF. The features encode ChannelThermal wall, inlet, outlet, and
centerline context outside the reusable HONF core.

Environment feature columns:
0. normalized x
1. normalized y
2. bottom-wall distance normalized by channel height
3. top-wall distance normalized by channel height
4. inlet distance normalized by channel length
5. outlet distance normalized by channel length
6. centerline proximity, 1 at centerline and 0 near walls
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChannelThermalEnvironment:
    env_coords: torch.Tensor
    env_features: torch.Tensor


class ChannelThermalEnvironmentBuilder:
    """Build cell-centered ChannelThermal environment tokens."""

    feature_names = (
        "x_norm",
        "y_norm",
        "bottom_wall_distance_norm",
        "top_wall_distance_norm",
        "inlet_distance_norm",
        "outlet_distance_norm",
        "centerline_proximity",
    )

    def __call__(
        self,
        *,
        batch_size: int,
        num_env_tokens_x: int,
        num_env_tokens_y: int,
        domain_length_x: float,
        domain_length_y: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ChannelThermalEnvironment:
        nx = max(int(num_env_tokens_x), 1)
        ny = max(int(num_env_tokens_y), 1)
        lx = max(float(domain_length_x), 1.0e-6)
        ly = max(float(domain_length_y), 1.0e-6)
        xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) / float(nx) * lx
        ys = (torch.arange(ny, device=device, dtype=dtype) + 0.5) / float(ny) * ly
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        centerline = 1.0 - (y - 0.5 * ly).abs() / max(0.5 * ly, 1.0e-6)
        features = torch.cat(
            [
                x / lx,
                y / ly,
                y / ly,
                (ly - y) / ly,
                x / lx,
                (lx - x) / lx,
                centerline.clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        return ChannelThermalEnvironment(
            env_coords=coords.unsqueeze(0).expand(batch_size, -1, -1),
            env_features=features.unsqueeze(0).expand(batch_size, -1, -1),
        )
