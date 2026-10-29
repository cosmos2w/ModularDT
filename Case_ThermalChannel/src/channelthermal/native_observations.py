"""Model-neutral geometry and interpolation helpers for Thermal observations.

The helpers in this module own only prescribed geometry and native receiver
extraction. They are shared by the historical source-response adapter and the
content-hypergraph adapter; no learned model or reference field is constructed
here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass
class NativeStencil:
    """Four-cell interpolation stencil on the native cell-centre grid."""

    indices: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor


@dataclass
class NativeGridUnion:
    """One deduplicated cell union with role-local interpolation pullbacks."""

    grid_indices: torch.Tensor
    grid_coordinates: torch.Tensor
    stencils: dict[str, NativeStencil]
    unique_counts: torch.Tensor
    mandatory_to_union: torch.Tensor | None = None


def _column(structure, name: str, like: torch.Tensor, fallback: float) -> torch.Tensor:
    value = structure.get(name)
    if value is None:
        return like.new_full((like.shape[0], 1), fallback)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype).reshape(like.shape[0], 1)


def thermal_context_tensors(
    structure: Mapping[str, object],
    *,
    environment_nx: int = 24,
    environment_ny: int = 8,
    preserve_dtype: bool = False,
) -> dict[str, torch.Tensor]:
    """Build the target-free Thermal module/environment token inventory.

    ``preserve_dtype=False`` keeps the historical FP32 behavior. The new
    adapter opts in to dtype preservation so coordinate derivatives and
    FP64 software fixtures do not get silently narrowed.
    """

    centers = torch.as_tensor(structure["module_centers"])
    if not centers.is_floating_point():
        centers = centers.to(torch.float32)
    elif not preserve_dtype:
        centers = centers.float()
    for name in ("t_in", "t_wall"):
        if name in structure:
            boundary = torch.as_tensor(structure[name], device=centers.device)
            if not bool(torch.isfinite(boundary).all()) or not bool((boundary == 0).all()):
                raise ValueError("This affine source capability requires zero inlet/wall temperature.")
    present = torch.as_tensor(structure["module_present"], device=centers.device, dtype=centers.dtype)
    if centers.ndim != 3 or centers.shape[-1] != 2 or present.shape != centers.shape[:2]:
        raise ValueError("Thermal geometry requires [B,M,2] centers and [B,M] mask.")
    material = torch.as_tensor(structure["material_params"], device=centers.device, dtype=centers.dtype)
    if material.ndim == 1:
        material = material[None].expand(centers.shape[0], -1)
    if material.shape != (centers.shape[0], 6):
        raise ValueError("Thermal material order is [nu,solid_alpha,fluid_alpha,solid_k,fluid_k,radius].")
    if not bool(torch.isfinite(material).all()) or not bool((material > 0).all()):
        raise ValueError("Native thermal coefficients and radius must be positive and finite.")
    lengths = torch.cat(
        (
            _column(structure, "domain_length_x", centers, 12.0),
            _column(structure, "domain_length_y", centers, 6.0),
        ),
        -1,
    )
    if not bool(torch.isfinite(lengths).all()) or not bool((lengths > 0).all()):
        raise ValueError("Domain lengths must be positive and finite.")
    if environment_nx < 1 or environment_ny < 1:
        raise ValueError("Thermal geometry catalogue dimensions must be positive.")

    count = present.sum(1, keepdim=True)
    radius = material[:, 5:6]
    descriptors = material[:, 1:5] * centers.new_tensor([100.0, 100.0, 1.0, 1.0])
    sources = torch.cat(
        (
            centers / lengths[:, None],
            (radius / lengths)[:, None].expand(-1, centers.shape[1], -1),
            descriptors[:, None].expand(-1, centers.shape[1], -1),
        ),
        -1,
    )
    context = torch.cat(
        (
            _column(structure, "re", centers, 0.0) / 100.0,
            _column(structure, "u_in", centers, 0.0),
            material[:, :5] * centers.new_tensor([100.0, 100.0, 100.0, 1.0, 1.0]),
            radius,
            lengths / centers.new_tensor([12.0, 6.0]),
            count / 12.0,
            count / lengths.prod(-1, keepdim=True),
            centers.new_zeros(centers.shape[0], 2),
        ),
        -1,
    )

    axis_x = (torch.arange(environment_nx, device=centers.device, dtype=centers.dtype) + 0.5) / environment_nx
    axis_y = (torch.arange(environment_ny, device=centers.device, dtype=centers.dtype) + 0.5) / environment_ny
    yy, xx = torch.meshgrid(axis_y, axis_x, indexing="ij")
    env_normal = torch.stack((xx, yy), -1).reshape(1, -1, 2).expand(centers.shape[0], -1, -1)
    env_coords = env_normal * lengths[:, None]
    distance = torch.linalg.vector_norm(env_coords[:, :, None] - centers[:, None], dim=-1)
    live_distance = torch.where(present[:, None] > 0.5, distance, distance.new_full((), float("inf")))
    nearest = live_distance.amin(-1, keepdim=True) / radius[:, None]
    nearest = torch.where(torch.isfinite(nearest), nearest, nearest.new_full((), 10.0))
    occupancy = (torch.exp(-distance.square() / radius[:, None].square()) * present[:, None]).sum(-1, keepdim=True)
    boundary_distance = torch.cat((env_normal, 1.0 - env_normal), -1)
    env_tokens = torch.cat((env_normal, nearest - 1.0, occupancy, boundary_distance), -1)
    source_lengths = radius.expand(-1, centers.shape[1])
    source_measures = torch.pi * source_lengths.square() * present
    env_measures = lengths.prod(-1, keepdim=True).expand(-1, env_coords.shape[1]) / env_coords.shape[1]

    source_ids = structure.get("module_source_ids")
    if source_ids is None:
        source_ids = torch.arange(centers.shape[1], device=centers.device, dtype=torch.long)[None].expand(
            centers.shape[0], -1
        )
    elif not torch.is_tensor(source_ids):
        if len(source_ids) != centers.shape[1]:
            raise ValueError("Source ID catalogue does not match active/padded slots.")
        # Typed string IDs remain adapter metadata; tensor IDs are slot-local
        # correspondence keys and are never embedded by the shared model.
        source_ids = torch.arange(centers.shape[1], device=centers.device, dtype=torch.long)[None].expand(
            centers.shape[0], -1
        )
    else:
        source_ids = source_ids.to(device=centers.device, dtype=torch.long)
        if source_ids.ndim == 1:
            source_ids = source_ids[None].expand(centers.shape[0], -1)
        if source_ids.shape != present.shape:
            raise ValueError("Thermal source IDs must match the padded [B,M] module slots.")
    source_ids = torch.where(present > 0.5, source_ids, source_ids.new_full((), -1))
    return {
        "sources": sources,
        "context": context,
        "centers": centers,
        "present": present,
        "lengths": lengths,
        "source_lengths": source_lengths,
        "source_measures": source_measures,
        "environment_tokens": env_tokens,
        "environment_coords": env_coords,
        "environment_present": present.new_ones(present.shape[0], env_coords.shape[1]),
        "environment_measures": env_measures,
        "source_ids": source_ids,
    }


def native_cell_center_stencil(xy: torch.Tensor, lengths: torch.Tensor, *, nx: int, ny: int) -> NativeStencil:
    """Construct the maintained cell-centre bilinear interpolation stencil."""

    if nx < 2 or ny < 2 or xy.ndim < 3 or xy.shape[0] != lengths.shape[0] or xy.shape[-1] != 2:
        raise ValueError("Native interpolation requires [B,...,2] receivers and a grid of at least 2x2 cells.")
    if lengths.shape != (xy.shape[0], 2) or not xy.is_floating_point():
        raise ValueError("Native interpolation lengths must be floating [B,2].")
    flat = xy.reshape(xy.shape[0], -1, 2)
    if not bool(torch.isfinite(flat).all()) or not bool(torch.isfinite(lengths).all()) or bool((lengths <= 0).any()):
        raise ValueError("Native receivers and domain lengths must be finite; lengths must be positive.")
    grid = flat / (lengths.to(flat)[:, None] / flat.new_tensor([nx, ny])) - 0.5
    rounded = grid.round()
    near = (grid - rounded).abs() <= 8 * torch.finfo(grid.dtype).eps * max(nx, ny)
    # Snapping is a discrete indexing aid; the original coordinate VJP remains.
    grid = grid + torch.where(near, rounded - grid, grid.new_zeros(())).detach()
    valid = ((grid >= 0) & (grid <= grid.new_tensor([nx - 1, ny - 1]))).all(-1)
    clipped = torch.minimum(torch.maximum(grid, grid.new_zeros(())), grid.new_tensor([nx - 1, ny - 1]))
    low = clipped.floor().long()
    high = torch.minimum(low + 1, low.new_tensor([nx - 1, ny - 1]))
    weight = clipped - low.to(clipped)
    wx, wy = weight.unbind(-1)
    ix, iy = low.unbind(-1)
    jx, jy = high.unbind(-1)
    indices = torch.stack((iy * nx + ix, iy * nx + jx, jy * nx + ix, jy * nx + jx), -1)
    weights = torch.stack(((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy), -1)
    return NativeStencil(indices, weights, valid)


def interpolate_native(values: torch.Tensor, stencil: NativeStencil) -> torch.Tensor:
    """Apply observation-local interpolation weights to values on a grid union."""

    if values.ndim < 3 or values.shape[0] != stencil.indices.shape[0]:
        raise ValueError("Native interpolation values must be [B,N,...] and match stencil batch size.")
    if stencil.indices.shape != stencil.weights.shape or stencil.valid.shape != stencil.indices.shape[:-1]:
        raise ValueError("Native interpolation stencil axes are inconsistent.")
    if stencil.indices.numel() and (bool((stencil.indices < 0).any()) or bool((stencil.indices >= values.shape[1]).any())):
        raise ValueError("Native interpolation stencil points outside its prepared receiver union.")
    batch = torch.arange(values.shape[0], device=values.device).reshape(
        values.shape[0], *([1] * (stencil.indices.ndim - 1))
    )
    gathered = values[batch, stencil.indices]
    extra_dims = values.ndim - 2
    weights = stencil.weights.to(values.dtype).reshape(*stencil.weights.shape, *([1] * extra_dims))
    valid = stencil.valid.reshape(*stencil.valid.shape, *([1] * extra_dims))
    return (gathered * weights).sum(-extra_dims - 1) * valid


def build_native_solid_mask(
    module_centers: torch.Tensor,
    module_present: torch.Tensor,
    module_radius: torch.Tensor,
    lengths: torch.Tensor,
    *,
    nx: int,
    ny: int,
) -> torch.Tensor:
    """Rebuild the target-free circular-module mask for the current geometry."""

    if module_centers.ndim != 3 or module_centers.shape[-1] != 2:
        raise ValueError("Thermal module centres must be [B,M,2].")
    batch, modules, _ = module_centers.shape
    if module_present.shape != (batch, modules) or lengths.shape != (batch, 2):
        raise ValueError("Thermal geometry mask inputs have inconsistent batch/module axes.")
    radius = torch.as_tensor(module_radius, device=module_centers.device, dtype=module_centers.dtype)
    if radius.ndim == 1:
        radius = radius[:, None]
    if radius.shape not in ((batch, 1), (batch, modules)):
        raise ValueError("Thermal module radius must be [B,1] or [B,M].")
    x = (torch.arange(nx, device=module_centers.device, dtype=module_centers.dtype) + 0.5) / nx
    y = (torch.arange(ny, device=module_centers.device, dtype=module_centers.dtype) + 0.5) / ny
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None] * lengths[:, None, None, :].to(module_centers)
    solid = torch.zeros((batch, ny, nx), dtype=torch.bool, device=module_centers.device)
    for module in range(modules):
        distance_squared = (grid - module_centers[:, module, None, None, :]).square().sum(-1)
        module_radius = radius[:, 0] if radius.shape[1] == 1 else radius[:, module]
        solid |= (distance_squared <= module_radius[:, None, None].square()) & (module_present[:, module, None, None] > 0.5)
    return solid


def merge_native_grid_receiver_union(
    stencils: Mapping[str, NativeStencil],
    role_coordinates: Mapping[str, torch.Tensor],
    module_present: torch.Tensor,
    lengths: torch.Tensor,
    *,
    nx: int,
    ny: int,
    module_roles: tuple[str, ...] = ("surface", "outside", "material"),
    mandatory_grid_indices: torch.Tensor | None = None,
    mandatory_counts: torch.Tensor | None = None,
) -> NativeGridUnion:
    """Merge interpolation and optional native-curl cells into one read union.

    Observation coordinates, validity, and interpolation weights remain in
    their per-role stencils. Only duplicate *grid values* are coalesced, so
    coincident observations with distinct derivative ownership retain their
    own pullbacks.
    """

    if set(stencils) != set(role_coordinates):
        raise ValueError("Every Thermal observation role must have one matching interpolation stencil.")
    if module_present.ndim != 2 or lengths.shape != (module_present.shape[0], 2):
        raise ValueError("Thermal union requires [B,M] presence and [B,2] lengths.")
    batch = int(module_present.shape[0])
    if mandatory_grid_indices is not None:
        if mandatory_grid_indices.ndim != 2 or mandatory_grid_indices.shape[0] != batch:
            raise ValueError("Mandatory native rows must be padded [B,N] grid indices.")
        if mandatory_counts is None or mandatory_counts.shape != (batch,):
            raise ValueError("Mandatory native rows require per-case valid counts.")
    elif mandatory_counts is not None:
        raise ValueError("Mandatory native counts require their grid indices.")

    unions: list[torch.Tensor] = []
    counts: list[int] = []
    for batch_index in range(batch):
        selected = []
        if mandatory_grid_indices is not None:
            count = int(mandatory_counts[batch_index].item())
            if count < 1 or count > mandatory_grid_indices.shape[1]:
                raise ValueError("Mandatory native receiver count is outside its padded extent.")
            selected.append(mandatory_grid_indices[batch_index, :count])
        for name, stencil in stencils.items():
            needed = stencil.weights[batch_index] != 0
            if stencil.weights.requires_grad:
                needed = torch.ones_like(needed)
            if name in module_roles:
                role_shape = role_coordinates[name].shape[1:-1]
                if not role_shape or role_shape[0] != module_present.shape[1]:
                    raise ValueError(f"Thermal module role {name!r} lost its [B,M,...,2] geometry axis.")
                active_shape = (module_present.shape[1],) + (1,) * (len(role_shape) - 1)
                active = module_present[batch_index].reshape(active_shape).expand(role_shape).reshape(-1)
                needed = needed.reshape(-1, 4) & active[:, None].bool()
            else:
                needed = needed.reshape(-1, 4)
            needed = needed & stencil.valid[batch_index].reshape(-1, 1)
            selected.append(stencil.indices[batch_index].reshape(-1, 4)[needed])
        if not selected:
            union = torch.zeros(1, dtype=torch.long, device=module_present.device)
        else:
            candidates = torch.cat([item.reshape(-1) for item in selected])
            union = torch.unique(candidates, sorted=True) if candidates.numel() else torch.zeros(
                1, dtype=torch.long, device=module_present.device
            )
        unions.append(union)
        counts.append(int(union.numel()))

    width = max(counts)
    grid_indices = torch.stack(
        [torch.cat((union, union[:1].expand(width - union.numel()))) for union in unions]
    )
    spacing = lengths / lengths.new_tensor([nx, ny])
    grid_xy = torch.stack((grid_indices % nx + 0.5, grid_indices // nx + 0.5), -1).to(lengths)
    grid_coordinates = grid_xy * spacing[:, None]
    remapped = {}
    for name, stencil in stencils.items():
        new_indices = []
        for batch_index, union in enumerate(unions):
            mapped = torch.searchsorted(union, stencil.indices[batch_index])
            safe = mapped.clamp_max(union.numel() - 1)
            found = (mapped < union.numel()) & (union[safe] == stencil.indices[batch_index])
            new_indices.append(torch.where(found, mapped, torch.zeros_like(mapped)))
        remapped[name] = NativeStencil(torch.stack(new_indices), stencil.weights, stencil.valid)
    mandatory_to_union = None
    if mandatory_grid_indices is not None:
        mapped_rows = []
        for batch_index, union in enumerate(unions):
            mapped = torch.searchsorted(union, mandatory_grid_indices[batch_index])
            safe = mapped.clamp_max(union.numel() - 1)
            found = (mapped < union.numel()) & (union[safe] == mandatory_grid_indices[batch_index])
            mapped_rows.append(torch.where(found, mapped, torch.zeros_like(mapped)))
        mandatory_to_union = torch.stack(mapped_rows)
    return NativeGridUnion(
        grid_indices=grid_indices,
        grid_coordinates=grid_coordinates,
        stencils=remapped,
        unique_counts=torch.tensor(counts, device=module_present.device, dtype=torch.long),
        mandatory_to_union=mandatory_to_union,
    )
