"""Resolved, prescribed Thermal cell-center inputs for the original E records."""
from __future__ import annotations

import torch

_indices = torch.zeros((192, 8, 6), dtype=torch.long)
_valid = torch.zeros((192, 8, 6), dtype=torch.bool)
for _ey in range(8):
    for _ex in range(24):
        _xs = [i for i in range(128) if ((2 * i + 1) * 24 // 256) == _ex]
        for _row in range(8):
            for _col, _x in enumerate(_xs):
                _indices[_ey * 24 + _ex, _row, _col] = (_ey * 8 + _row) * 128 + _x
                _valid[_ey * 24 + _ex, _row, _col] = True


def prescribed_environment_patch(centers, present, geometry, heat, material, *, heat_scale, physical):
    """Return [B,192,8,6,7] patches in y/x order, computed in FP64.

    Inputs contain only prescribed configuration. Ownership follows the source
    grid and original-order, inclusive, last-owner-wins disk rasterization.
    """
    batch, modules = present.shape
    if (centers.shape != (batch, modules, 2) or geometry.shape != (batch, 5)
            or heat.shape != (batch, modules) or material.shape != (batch, 4)
            or any(t.dtype != torch.float64 for t in (centers, geometry, heat, material))):
        raise ValueError("Prescribed patches require FP64 geometry, heat and material in original slots.")
    if (not all(bool(torch.isfinite(t).all()) for t in (centers, geometry, heat, material))
            or bool((geometry[:, :3] <= 0).any()) or bool((material <= 0).any())
            or not bool((geometry[:, 3:] == geometry.new_tensor([128, 64])).all()) or heat_scale <= 0):
        raise ValueError("Invalid prescribed patch geometry, coefficients, grid or heat scale.")
    lx, ly, radius = geometry[:, 0], geometry[:, 1], geometry[:, 2]
    x = torch.arange(128, dtype=centers.dtype, device=centers.device)[None] * (lx[:, None] / 128) + .5 * lx[:, None] / 128
    y = torch.arange(64, dtype=centers.dtype, device=centers.device)[None] * (ly[:, None] / 64) + .5 * ly[:, None] / 64
    owner = torch.full((batch, 64, 128), -1, dtype=torch.long, device=centers.device)
    for m in range(modules):
        inside = (torch.hypot(x[:, None, :] - centers[:, m, 0, None, None],
                              y[:, :, None] - centers[:, m, 1, None, None]) <= radius[:, None, None])
        owner = torch.where(inside & present[:, m, None, None].bool(), m, owner)
    solid = owner >= 0
    source = torch.gather(heat, 1, owner.clamp_min(0).reshape(batch, -1)).reshape_as(owner) * solid
    alpha_s, alpha_f, k_s, k_f = material.unbind(-1)
    index = _indices.to(centers.device)
    valid = _valid.to(centers.device)[None].expand(batch, -1, -1, -1)
    flat = index.reshape(-1)
    xx = x[:, (flat % 128)].reshape(batch, 192, 8, 6)
    yy = y[:, (flat // 128)].reshape(batch, 192, 8, 6)
    region = torch.arange(192, device=centers.device)
    ex, ey = (region % 24).reshape(1, 192, 1, 1), (region // 24).reshape(1, 192, 1, 1)
    relative_x = (xx - (ex + .5) * lx[:, None, None, None] / 24) / (lx[:, None, None, None] / 24)
    relative_y = (yy - (ey + .5) * ly[:, None, None, None] / 8) / (ly[:, None, None, None] / 8)
    occupied = solid.reshape(batch, -1)[:, flat].reshape(batch, 192, 8, 6) & valid
    if not physical:
        occupied = torch.zeros_like(occupied)
    q = source.reshape(batch, -1)[:, flat].reshape(batch, 192, 8, 6) / heat_scale
    return torch.stack((valid, relative_x * valid, relative_y * valid, occupied,
        occupied * ((k_s - k_f) / (k_s + k_f))[:, None, None, None],
        occupied * ((alpha_s - alpha_f) / (alpha_s + alpha_f))[:, None, None, None],
        q * occupied), -1).to(centers)
