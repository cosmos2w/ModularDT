"""Thermal E-lattice coverage carried by native query batches."""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from honf_forward_core.content_hypergraph.contracts import QueryBatch, SceneBatch


def _require(condition: Tensor, message: str) -> None:
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _slot_grid(scene: SceneBatch, batch_index: int, nx: int, ny: int) -> tuple[Tensor, Tensor, Tensor]:
    """Map physical lattice coordinates to prepared E slots without ID/order assumptions."""
    coordinates = scene.environment_coordinates[batch_index]
    mask = scene.environment_mask[batch_index]
    origin = (scene.coordinate_origin[batch_index] if scene.coordinate_origin is not None
              else coordinates.new_zeros(2))
    if scene.coordinate_scale is None:
        raise ValueError("Thermal E coverage requires the declared physical coordinate scale.")
    extent = scene.coordinate_scale[batch_index]
    axis_x = origin[0] + (torch.arange(nx, device=coordinates.device, dtype=coordinates.dtype) + 0.5) * extent[0] / nx
    axis_y = origin[1] + (torch.arange(ny, device=coordinates.device, dtype=coordinates.dtype) + 0.5) * extent[1] / ny

    ix = (coordinates[:, 0, None] - axis_x[None]).abs().argmin(-1)
    iy = (coordinates[:, 1, None] - axis_y[None]).abs().argmin(-1)
    linear = iy * nx + ix
    expected = torch.stack((axis_x[ix], axis_y[iy]), dim=-1)
    tolerance = extent.abs().amax() * (16 * torch.finfo(coordinates.dtype).eps)
    _require(((coordinates - expected).abs().amax(-1) <= tolerance).all(),
             "Prepared Thermal E coordinates do not match the declared regular lattice.")
    _require(mask.all(), "Thermal E coverage requires the complete declared E lattice.")
    counts = torch.zeros((nx * ny,), device=coordinates.device, dtype=torch.long)
    counts.scatter_add_(0, linear, torch.ones_like(linear))
    _require((counts == 1).all(), "Prepared Thermal E slots do not form a one-to-one lattice mapping.")
    slot_by_grid = torch.empty((nx * ny,), device=coordinates.device, dtype=torch.long)
    slot_by_grid.scatter_(0, linear, torch.arange(coordinates.shape[0], device=coordinates.device))
    return slot_by_grid.reshape(ny, nx), axis_x, axis_y


def attach_thermal_e_anchor_query(
    query: QueryBatch,
    scene: SceneBatch,
    locator_mode: str,
    *,
    environment_nx: int = 24,
    environment_ny: int = 8,
) -> QueryBatch:
    """Attach A or L sparse E-slot coverage using live physical coordinates.

    ``e_anchor`` assigns one containing E cell with half-open interior
    ownership and a closed final endpoint. ``local_e_cover`` blends up to four
    adjacent center anchors with tensor-product cubic smoothstep weights; the
    outer half-cells clamp to their edge center. Indices are looked up through
    the scene's actual coordinate-to-slot map, so IDs and tensor order are not
    treated as grid coordinates.
    """
    if locator_mode not in {"e_anchor", "local_e_cover"}:
        raise ValueError("Thermal E locator_mode must be 'e_anchor' or 'local_e_cover'.")
    if (environment_nx < 2 or environment_ny < 2
            or scene.environment_coordinates.shape[1] != environment_nx * environment_ny):
        raise ValueError("Thermal E locator dimensions differ from the prepared environment catalogue.")
    if query.coordinates.ndim != 3 or query.coordinates.shape[0] != scene.batch_size or query.coordinates.shape[-1] != 2:
        raise ValueError("Thermal E coverage requires query coordinates with shape [B,Q,2].")
    if query.query_mask is None:
        query_mask = torch.ones(query.coordinates.shape[:2], device=query.coordinates.device, dtype=torch.bool)
    else:
        query_mask = query.query_mask

    anchor_indices: list[Tensor] = []
    anchor_weights: list[Tensor] = []
    for batch_index in range(scene.batch_size):
        slot_by_grid, axis_x, axis_y = _slot_grid(scene, batch_index, environment_nx, environment_ny)
        coordinates = query.coordinates[batch_index]
        origin = (scene.coordinate_origin[batch_index] if scene.coordinate_origin is not None
                  else coordinates.new_zeros(2))
        extent = scene.coordinate_scale[batch_index]
        upper = origin + extent
        tolerance = extent.abs().amax() * (16 * torch.finfo(coordinates.dtype).eps)
        in_domain = (
            (coordinates[:, 0] >= origin[0] - tolerance)
            & (coordinates[:, 0] <= upper[0] + tolerance)
            & (coordinates[:, 1] >= origin[1] - tolerance)
            & (coordinates[:, 1] <= upper[1] + tolerance)
        )
        _require((~query_mask[batch_index] | in_domain).all(),
                 "A valid Thermal native query lies outside the physical E coverage bounds.")
        x = coordinates[:, 0].clamp(min=origin[0], max=upper[0])
        y = coordinates[:, 1].clamp(min=origin[1], max=upper[1])

        if locator_mode == "e_anchor":
            step_x = (axis_x[1] - axis_x[0])
            step_y = (axis_y[1] - axis_y[0])
            ix = torch.floor((x - origin[0]) / step_x).to(torch.long).clamp(0, environment_nx - 1)
            iy = torch.floor((y - origin[1]) / step_y).to(torch.long).clamp(0, environment_ny - 1)
            indices = slot_by_grid[iy, ix][:, None]
            weights = coordinates.new_ones((coordinates.shape[0], 1))
        else:
            ix_right = torch.searchsorted(axis_x.contiguous(), x.detach().contiguous(), right=True)
            iy_right = torch.searchsorted(axis_y.contiguous(), y.detach().contiguous(), right=True)
            ix_right = ix_right.clamp(1, environment_nx - 1)
            iy_right = iy_right.clamp(1, environment_ny - 1)
            ix_left, iy_left = ix_right - 1, iy_right - 1
            tx = ((x - axis_x[ix_left]) / (axis_x[ix_right] - axis_x[ix_left])).clamp(0, 1)
            ty = ((y - axis_y[iy_left]) / (axis_y[iy_right] - axis_y[iy_left])).clamp(0, 1)
            sx = 3 * tx.square() - 2 * tx.pow(3)
            sy = 3 * ty.square() - 2 * ty.pow(3)
            indices = torch.stack((
                slot_by_grid[iy_left, ix_left], slot_by_grid[iy_left, ix_right],
                slot_by_grid[iy_right, ix_left], slot_by_grid[iy_right, ix_right],
            ), dim=-1)
            weights = torch.stack((
                (1 - sx) * (1 - sy), sx * (1 - sy),
                (1 - sx) * sy, sx * sy,
            ), dim=-1)
            # Remove zero-weight and duplicate boundary entries while keeping
            # a fixed four-slot carrier for query slicing and checkpointing.
            keep = weights > 0
            for index in range(1, 4):
                duplicate = ((indices[..., index, None] == indices[..., :index])
                             & keep[..., :index]).any(-1)
                keep[..., index] &= ~duplicate
            weights = torch.where(keep, weights, torch.zeros_like(weights))
            indices = torch.where(keep, indices, torch.full_like(indices, -1))

        valid = query_mask[batch_index]
        anchor_indices.append(torch.where(valid[:, None], indices, torch.full_like(indices, -1)))
        anchor_weights.append(torch.where(valid[:, None], weights, torch.zeros_like(weights)))

    return replace(
        query,
        anchor_indices=torch.stack(anchor_indices),
        anchor_weights=torch.stack(anchor_weights),
    )


__all__ = ["attach_thermal_e_anchor_query"]
