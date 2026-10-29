"""Supervision-side discrete balance; never solve or integrate temperature.

Stored TRAIN velocity is allowed here as a known stencil coefficient and is
excluded from the inference adapter. The operator matches the shared-grid
generator, including boundary replacement and solid-only forcing.
"""

from dataclasses import dataclass

import torch


@dataclass
class DiscreteThermalBalance:
    coefficients: torch.Tensor  # [B,ny,nx,5]: self,left,right,down,up
    source_slots: torch.Tensor  # [B,ny,nx], -1 fluid
    present: torch.Tensor

    @classmethod
    def from_training_fields(cls, u, v, module_ids, material, lengths, present):
        if u.ndim == 2:
            u, v, module_ids = u[None], v[None], module_ids[None]
        if u.shape != v.shape or u.shape != module_ids.shape:
            raise ValueError("Discrete coefficient fields require the same [B,ny,nx] grid.")
        batch, ny, nx = u.shape
        material = material.reshape(batch, 6).to(u)
        lengths = lengths.reshape(batch, 2).to(u)
        fluid = module_ids < 0
        alpha = torch.where(fluid, material[:, 2, None, None], material[:, 1, None, None])
        dx, dy = lengths[:, 0, None, None] / nx, lengths[:, 1, None, None] / ny
        ax = 0.5 * (alpha[:, :, 1:] + alpha[:, :, :-1]) / dx.square()
        ay = 0.5 * (alpha[:, 1:] + alpha[:, :-1]) / dy.square()
        c = u.new_zeros(batch, ny, nx, 5)
        c[:, :, :-1, 0] += ax
        c[:, :, 1:, 0] += ax
        c[:, :, :-1, 2] -= ax
        c[:, :, 1:, 1] -= ax
        c[:, :-1, :, 0] += ay
        c[:, 1:, :, 0] += ay
        c[:, :-1, :, 4] -= ay
        c[:, 1:, :, 3] -= ay
        c[..., 0] += (u.abs() / dx + v.abs() / dy) * fluid
        c[..., 1] -= u.clamp_min(0) / dx * fluid
        c[..., 2] -= (-u).clamp_min(0) / dx * fluid
        c[..., 3] -= v.clamp_min(0) / dy * fluid
        c[..., 4] -= (-v).clamp_min(0) / dy * fluid
        # Overwriting temperature boundaries in the generator replaces PDE
        # rows. Wall assignment follows inlet/outlet assignment, so corners
        # obey the wall Dirichlet condition.
        c[:, :, 0] = 0
        c[:, :, 0, 0] = 1
        c[:, :, -1] = 0
        c[:, :, -1, 0] = 1
        c[:, :, -1, 1] = -1
        c[:, 0] = 0
        c[:, 0, :, 0] = 1
        c[:, -1] = 0
        c[:, -1, :, 0] = 1
        return cls(c, module_ids.long(), present.to(u))

    def apply(self, temperature):
        """Apply A to fields/kernels [B,ny,nx] or [B,ny,nx,S]."""
        if temperature.shape[:3] != self.coefficients.shape[:3]:
            raise ValueError("Temperature grid does not match the discrete stencil.")
        if temperature.ndim == 3:
            temperature = temperature[..., None]
            squeeze = True
        elif temperature.ndim == 4:
            squeeze = False
        else:
            raise ValueError("Discrete fields must have three or four axes.")
        left = torch.cat((temperature[:, :, :1], temperature[:, :, :-1]), 2)
        right = torch.cat((temperature[:, :, 1:], temperature[:, :, -1:]), 2)
        down = torch.cat((temperature[:, :1], temperature[:, :-1]), 1)
        up = torch.cat((temperature[:, 1:], temperature[:, -1:]), 1)
        result = sum(
            self.coefficients[..., k, None] * field for k, field in enumerate((temperature, left, right, down, up))
        )
        return result[..., 0] if squeeze else result

    def forcing(self, physical_heat):
        if physical_heat.ndim != 2 or physical_heat.shape != self.present.shape:
            raise ValueError("Physical forcing requires [B,M] amplitudes.")
        slots = self.source_slots.clamp_min(0)
        heat = physical_heat.gather(1, slots.reshape(slots.shape[0], -1)).reshape(slots.shape)
        source = heat * (self.source_slots >= 0)
        source = source.clone()
        source[:, :, 0] = 0
        source[:, :, -1] = 0
        source[:, 0] = 0
        source[:, -1] = 0
        return source

    def forcing_columns(self):
        slots = torch.arange(self.present.shape[1], device=self.source_slots.device)
        source = (self.source_slots[..., None] == slots) * self.present[:, None, None]
        source = source.to(self.coefficients).clone()
        source[:, :, 0] = 0
        source[:, :, -1] = 0
        source[:, 0] = 0
        source[:, -1] = 0
        return source

    def row_scale(self):
        return self.coefficients.abs().sum(-1).clamp_min(torch.finfo(self.coefficients.dtype).eps)

    def adjoint(self, values):
        """Exact transpose of the implemented boundary-aware local stencil."""
        if values.ndim == 3:
            values = values[..., None]
            squeeze = True
        else:
            squeeze = False
        c = self.coefficients
        result = c[..., 0, None] * values
        result = result.clone()
        result[:, :, :-1] += c[:, :, 1:, 1, None] * values[:, :, 1:]
        result[:, :, 1:] += c[:, :, :-1, 2, None] * values[:, :, :-1]
        result[:, :-1] += c[:, 1:, :, 3, None] * values[:, 1:]
        result[:, 1:] += c[:, :-1, :, 4, None] * values[:, :-1]
        # apply() uses repeated edge neighbours; factory boundary rows zero
        # these coefficients, but include them for the general exact adjoint.
        result[:, :, 0] += c[:, :, 0, 1, None] * values[:, :, 0]
        result[:, :, -1] += c[:, :, -1, 2, None] * values[:, :, -1]
        result[:, 0] += c[:, 0, :, 3, None] * values[:, 0]
        result[:, -1] += c[:, -1, :, 4, None] * values[:, -1]
        return result[..., 0] if squeeze else result

    def residual(self, temperature, physical_heat, *, normalize_rows=False):
        value = self.apply(temperature) - self.forcing(physical_heat)
        return value / self.row_scale() if normalize_rows else value

    def kernel_residual(self, kernels, *, normalize_rows=True):
        value = self.apply(kernels) - self.forcing_columns()
        return value / self.row_scale()[..., None] if normalize_rows else value


def module_grid_ids(x_grid, y_grid, centers, present, radius):
    """Geometry-only source deposition slots; later modules overwrite overlaps."""
    if x_grid.ndim == 2:
        x_grid, y_grid = x_grid[None], y_grid[None]
    ids = torch.full_like(x_grid, -1, dtype=torch.long)
    radius = radius.reshape(centers.shape[0], 1, 1)
    for index in range(centers.shape[1]):
        dist = (x_grid - centers[:, index, 0, None, None]).square() + (
            y_grid - centers[:, index, 1, None, None]
        ).square()
        mask = (dist <= radius.square()) & present[:, index, None, None].bool()
        ids = torch.where(mask, ids.new_full((), index), ids)
    return ids


def sampled_kernel_residual(core, context, balance, row_indices, lengths, *, chunk_size=512):
    """Evaluate sampled A K-B for every active source; no physical inversion.

    row_indices are original C-order ny*nx grid indices [B,R]. Coefficients
    come from supervision-side stored TRAIN u/v, never the inference encoder.
    Returns a row-normalized [B,R,M] residual and its actual neural row count.
    """
    batch, ny, nx = balance.source_slots.shape
    if row_indices.ndim != 2 or row_indices.shape[0] != batch:
        raise ValueError("Sampled residual rows require [B,R] native C-order indices.")
    if bool(((row_indices < 0) | (row_indices >= ny * nx)).any()):
        raise ValueError("Residual row lies outside the native grid.")
    ix, iy = row_indices % nx, row_indices // nx
    stencil = torch.stack(
        (
            row_indices,
            iy * nx + (ix - 1).clamp_min(0),
            iy * nx + (ix + 1).clamp_max(nx - 1),
            (iy - 1).clamp_min(0) * nx + ix,
            (iy + 1).clamp_max(ny - 1) * nx + ix,
        ),
        -1,
    )
    unions = [torch.unique(stencil[b], sorted=True) for b in range(batch)]
    width = max(u.numel() for u in unions)
    native_ids = torch.stack([torch.cat((u, u[:1].expand(width - u.numel()))) for u in unions])
    gather = torch.stack([torch.searchsorted(u, stencil[b]) for b, u in enumerate(unions)])
    native = torch.stack((native_ids % nx + 0.5, native_ids // nx + 0.5), -1).to(lengths)
    query = native * (lengths[:, None] / lengths.new_tensor([nx, ny]))
    grid_K = core.read_kernel(context, query, receiver_ids=native_ids, chunk_size=chunk_size)[..., 0]
    batch_indices = torch.arange(batch, device=gather.device)[:, None, None]
    K = grid_K[batch_indices, gather]
    coefficients = balance.coefficients.reshape(batch, -1, 5).gather(1, row_indices[..., None].expand(-1, -1, 5))
    AK = (K * coefficients[..., None]).sum(2)
    slots = balance.source_slots.reshape(batch, -1).gather(1, row_indices)
    modules = torch.arange(balance.present.shape[1], device=slots.device)
    B = (slots[..., None] == modules) * balance.present[:, None]
    boundary = (ix == 0) | (ix == nx - 1) | (iy == 0) | (iy == ny - 1)
    B = B.to(AK) * (~boundary[..., None])
    residual = (AK - B) / coefficients.abs().sum(-1).clamp_min(torch.finfo(AK.dtype).eps)[..., None]
    return residual, {
        "operator_rows": int(row_indices.numel()),
        "kernel_columns": int(balance.present.sum()),
        "neural_stencil_receiver_rows": int(query.shape[0] * query.shape[1]),
        "unique_stencil_receiver_rows": sum(int(u.numel()) for u in unions),
        "per_case_unique_receiver_rows": [int(u.numel()) for u in unions],
        "physical_solves": 0,
    }
