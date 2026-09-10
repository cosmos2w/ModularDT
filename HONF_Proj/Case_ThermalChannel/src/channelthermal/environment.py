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
    """Environment coordinates ``[B,E,2]`` and features ``[B,E,7]``."""

    env_coords: torch.Tensor
    env_features: torch.Tensor
    # Optional deterministic region IDs aligned with ``env_coords``.  The
    # regional-response experiment requests these from the case adapter;
    # existing families leave them unset and keep their original route.
    env_region_ids: torch.Tensor | None = None


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
    query_feature_names = (
        "x_norm",
        "y_norm",
        "bottom_wall_distance_norm",
        "top_wall_distance_norm",
        "inlet_distance_norm",
        "outlet_distance_norm",
    )

    @staticmethod
    def region_ids_from_coordinates(
        coordinates: torch.Tensor,
        *,
        num_env_tokens_x: int,
        num_env_tokens_y: int,
        block_shape: tuple[int, int] | list[int],
    ) -> torch.Tensor:
        """Derive physical grid-region IDs without relying on token order.

        The current adapter emits a cell-centred rectangular grid, with x
        varying fastest in its flattened representation.  Region assignment
        is nevertheless based on sorted physical coordinate ranks so a token
        permutation leaves the IDs permuted with the tokens, and exact
        duplicate samples share one region.  The metadata dimensions provide
        the expected number of blocks; no reshape of an arbitrary sequence is
        performed.
        """

        if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape [E, 2].")
        if not isinstance(block_shape, (tuple, list)) or len(block_shape) != 2:
            raise ValueError("block_shape must contain two positive integers.")
        if any(isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0 for value in block_shape):
            raise ValueError("block_shape must contain two positive integers.")
        nx = int(num_env_tokens_x)
        ny = int(num_env_tokens_y)
        if nx <= 0 or ny <= 0:
            raise ValueError("Environment grid dimensions must be positive.")
        block_x, block_y = (int(block_shape[0]), int(block_shape[1]))
        x_values, x_rank = torch.unique(coordinates[:, 0], sorted=True, return_inverse=True)
        y_values, y_rank = torch.unique(coordinates[:, 1], sorted=True, return_inverse=True)
        # This builder owns a rectangular grid, so every configured physical
        # row/column must be represented.  Exact duplicate quadrature samples
        # are still fine because ``unique`` removes them before ranking.
        if int(x_values.numel()) != nx or int(y_values.numel()) != ny:
            raise ValueError(
                "Environment coordinates do not match the configured rectangular grid metadata."
            )
        regions_x = (nx + block_x - 1) // block_x
        region_x = x_rank.to(torch.long) // block_x
        region_y = y_rank.to(torch.long) // block_y
        return region_y * regions_x + region_x

    @staticmethod
    def query_features(
        query_xy: torch.Tensor,
        *,
        domain_length_x: float,
        domain_length_y: float,
    ) -> torch.Tensor:
        """Build the case-owned rectangular query features ``[B,Q,6]``."""

        lx = max(float(domain_length_x), 1.0e-6)
        ly = max(float(domain_length_y), 1.0e-6)
        x = query_xy[..., 0:1]
        y = query_xy[..., 1:2]
        return torch.cat(
            [x / lx, y / ly, y / ly, (ly - y) / ly, x / lx, (lx - x) / lx],
            dim=-1,
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
        response_region_block_shape: tuple[int, int] | list[int] | None = None,
    ) -> ChannelThermalEnvironment:
        """Build a cell-centered ``nx*ny`` environment grid for each case."""

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
        env_region_ids = None
        if response_region_block_shape is not None:
            ids = self.region_ids_from_coordinates(
                coords,
                num_env_tokens_x=nx,
                num_env_tokens_y=ny,
                block_shape=response_region_block_shape,
            )
            env_region_ids = ids.unsqueeze(0).expand(batch_size, -1)
        return ChannelThermalEnvironment(
            env_coords=coords.unsqueeze(0).expand(batch_size, -1, -1),
            env_features=features.unsqueeze(0).expand(batch_size, -1, -1),
            env_region_ids=env_region_ids,
        )
