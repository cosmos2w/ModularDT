"""The ThermalChannel generator's discrete cell-centred vorticity operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

GENERATOR_CURL_ID = "channelthermal_numpy_gradient_cell_centered_v1"


@dataclass(frozen=True)
class NearObstacleCurlStencil:
    """Input-selected centres and their exact five-point curl query layout."""

    query_xy: np.ndarray
    solid_nodes: np.ndarray
    x_span: np.ndarray
    y_span: np.ndarray
    valid: np.ndarray
    omega_target: np.ndarray
    center_indices: np.ndarray
    candidate_count: int


def generator_discrete_vorticity(
    u: np.ndarray, v: np.ndarray, solid_mask: np.ndarray, dx: float, dy: float,
) -> np.ndarray:
    """Reproduce ``compute_vorticity`` and its caller's solid-cell rule."""

    u = np.asarray(u)
    v = np.asarray(v)
    solid = np.asarray(solid_mask, dtype=bool)
    if u.ndim != 2 or v.shape != u.shape or solid.shape != u.shape:
        raise ValueError("u, v and solid_mask must be equal-shaped two-dimensional arrays.")
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        raise ValueError("Grid spacings must be finite and positive.")
    dvdx = np.gradient(v, float(dx), axis=1, edge_order=1)
    dudy = np.gradient(u, float(dy), axis=0, edge_order=1)
    omega = dvdx - dudy
    omega[solid] = 0
    return omega


def build_near_obstacle_curl_stencil(
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    module_centers: np.ndarray,
    module_present: np.ndarray,
    module_mask: np.ndarray,
    omega_target: np.ndarray,
    radius: float,
    dx: float,
    dy: float,
    max_centers: int = 128,
    seed: int = 0,
) -> NearObstacleCurlStencil:
    """Choose at most ``max_centers`` fluid grid nodes within one radius.

    For every selected centre, query the centre plus its x/y derivative
    neighbours. Boundary neighbours duplicate the centre as required by the
    generator's first-order one-sided ``numpy.gradient`` boundary stencil.
    Selection depends only on supplied geometry and a fixed seed.
    """

    x = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    solid = np.asarray(module_mask, dtype=bool)
    centers = np.asarray(module_centers, dtype=np.float64)
    present = np.asarray(module_present, dtype=bool)
    target = np.asarray(omega_target)
    if x.ndim != 2 or y.shape != x.shape or solid.shape != x.shape or target.shape != x.shape:
        raise ValueError("Grids, module mask and omega target must have equal two-dimensional shapes.")
    if centers.ndim != 2 or centers.shape[1] != 2 or present.shape != (centers.shape[0],):
        raise ValueError("module_centers and module_present have incompatible shapes.")
    if not np.any(present):
        raise ValueError("Near-obstacle selection requires at least one present module.")
    if not np.isfinite(radius) or radius <= 0 or max_centers < 1:
        raise ValueError("radius and max_centers must be positive.")
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        raise ValueError("Grid spacings must be finite and positive.")

    active = centers[present]
    distance = np.full(x.shape, np.inf, dtype=np.float64)
    for center in active:
        distance = np.minimum(distance, np.hypot(x - center[0], y - center[1]) - float(radius))
    eligible = np.flatnonzero((~solid & (distance >= 0.0) & (distance <= float(radius))).reshape(-1))
    if eligible.size == 0:
        raise ValueError("The input-defined one-radius near-obstacle band has no fluid grid centres.")

    take = min(int(max_centers), int(eligible.size))
    if eligible.size > take:
        rng = np.random.default_rng(int(seed))
        selected = np.sort(rng.choice(eligible, size=take, replace=False))
    else:
        selected = eligible
    n_y, n_x = x.shape
    selected_y, selected_x = np.divmod(selected, n_x)
    left_x = np.maximum(selected_x - 1, 0)
    right_x = np.minimum(selected_x + 1, n_x - 1)
    down_y = np.maximum(selected_y - 1, 0)
    up_y = np.minimum(selected_y + 1, n_y - 1)
    node_indices = np.stack((
        selected,
        selected_y * n_x + left_x,
        selected_y * n_x + right_x,
        down_y * n_x + selected_x,
        up_y * n_x + selected_x,
    ), axis=1)
    valid = np.zeros(int(max_centers), dtype=bool)
    valid[:take] = True
    if take < int(max_centers):
        node_indices = np.pad(node_indices, ((0, int(max_centers) - take), (0, 0)), mode="edge")
        selected = np.pad(selected, (0, int(max_centers) - take), mode="edge")

    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    query_xy = np.stack((x_flat[node_indices], y_flat[node_indices]), axis=-1).astype(np.float32)
    solid_nodes = solid.reshape(-1)[node_indices]
    x_span = ((right_x - left_x) * float(dx)).astype(np.float32)
    y_span = ((up_y - down_y) * float(dy)).astype(np.float32)
    if take < int(max_centers):
        x_span = np.pad(x_span, (0, int(max_centers) - take), mode="edge")
        y_span = np.pad(y_span, (0, int(max_centers) - take), mode="edge")
    if np.any(x_span <= 0) or np.any(y_span <= 0):
        raise ValueError("Selected stencil centre lacks a valid x or y neighbour.")
    return NearObstacleCurlStencil(
        query_xy=query_xy.reshape(-1, 2),
        solid_nodes=solid_nodes,
        x_span=x_span,
        y_span=y_span,
        valid=valid,
        omega_target=target.reshape(-1)[selected].astype(np.float32),
        center_indices=selected.astype(np.int64),
        candidate_count=int(eligible.size),
    )


def torch_normalized_curl_from_stencil(
    normalized_flow: torch.Tensor,
    solid_nodes: torch.Tensor,
    x_span: torch.Tensor,
    y_span: torch.Tensor,
    field_mean: torch.Tensor,
    field_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centre omega and discrete curl in the same normalized units.

    ``normalized_flow`` has shape ``[B, N, 5, 4]`` with node order
    centre/left/right/down/up and channel order u/v/p/omega.
    """

    if normalized_flow.ndim != 4 or normalized_flow.shape[-2:] != (5, 4):
        raise ValueError("normalized_flow must have shape [B, N, 5, 4].")
    if solid_nodes.shape != normalized_flow.shape[:-1]:
        raise ValueError("solid_nodes must have shape [B, N, 5].")
    if x_span.shape != normalized_flow.shape[:2] or y_span.shape != normalized_flow.shape[:2]:
        raise ValueError("x_span and y_span must have shape [B, N].")
    mean = normalized_flow.new_tensor(field_mean).reshape(-1)[:4]
    std = normalized_flow.new_tensor(field_std).reshape(-1)[:4]
    if mean.numel() != 4 or std.numel() != 4 or torch.any(std <= 0):
        raise ValueError("Flow normalization requires four finite means and positive scales.")
    physical_u = normalized_flow[..., 0] * std[0] + mean[0]
    physical_v = normalized_flow[..., 1] * std[1] + mean[1]
    physical_u = torch.where(solid_nodes, torch.zeros_like(physical_u), physical_u)
    physical_v = torch.where(solid_nodes, torch.zeros_like(physical_v), physical_v)
    dvdx = (physical_v[..., 2] - physical_v[..., 1]) / x_span
    dudy = (physical_u[..., 4] - physical_u[..., 3]) / y_span
    curl_normalized = (dvdx - dudy - mean[3]) / std[3]
    omega_normalized = normalized_flow[..., 0, 3]
    return omega_normalized, curl_normalized


NATIVE_CURL_READOUT_LAW = "native_curl_cell_centred_v1"


def native_curl_readout_contract():
    """The explicit output law and target-free receiver-metadata contract."""
    return {
        "law": NATIVE_CURL_READOUT_LAW,
        "operator_id": GENERATOR_CURL_ID,
        "native_grid_shape_ny_nx": [64, 128],
        "derivative_node_order": ["centre", "left", "right", "down", "up"],
        "boundary_rule": "numpy.gradient edge_order=1; one-sided boundary differences and centred interior differences",
        "solid_rule": "saved geometry mask; physical u/v zero on solid stencil nodes and physical omega zero at solid centres before normalization",
        "receiver_input": "saved boolean [B,ny,nx] native_solid_mask; outside learned context",
        "receiver_domain": "native cell centres only; off-grid requests rejected",
        "node_union": "independent per-case ordered unique centres before neighbours; duplicate-first-node batch padding",
        "omega_projection_rows": 0,
        "omega_supervision_backpropagates_to_velocity": True,
        "learned_context_mask_input": False,
    }


@dataclass(frozen=True)
class NativeCurlReadStencil:
    query_xy: torch.Tensor
    node_indices: torch.Tensor
    gather_indices: torch.Tensor
    solid_nodes: torch.Tensor
    x_span: torch.Tensor
    y_span: torch.Tensor
    unique_counts: torch.Tensor


def build_native_curl_read_stencil(fluid_xy, lengths, native_solid_mask, *, nx, ny):
    """Build a target-free ordered node union for the generator's five-point curl.

    Primary receivers must be native cell centres. Their first-occurrence
    order precedes all added neighbours, so original query zero stays zero.
    Cases have independent unions, padded by repeating their first node.
    The saved geometry mask is receiver metadata, never learned context.
    """
    if nx < 2 or ny < 2:
        raise ValueError("Native curl requires at least two cells on each axis.")
    if fluid_xy.ndim != 3 or fluid_xy.shape[-1] != 2 or fluid_xy.shape[1] < 1:
        raise ValueError("Native curl receivers must be nonempty [B,Q,2].")
    batch, queries, _ = fluid_xy.shape
    if lengths.shape != (batch, 2) or not bool(torch.isfinite(lengths).all()) or bool((lengths <= 0).any()):
        raise ValueError("Native curl domain lengths must be finite positive [B,2].")
    if not torch.is_tensor(native_solid_mask) or native_solid_mask.dtype != torch.bool:
        raise TypeError("Native curl requires the saved boolean geometry mask.")
    if native_solid_mask.shape != (batch, ny, nx):
        raise ValueError("Native curl geometry mask must be [B,ny,nx].")
    if native_solid_mask.requires_grad:
        raise ValueError("A geometry mask cannot carry learned values or gradients.")
    if fluid_xy.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(fluid_xy).all()):
        raise ValueError("Native curl receivers require finite float32/float64 coordinates.")
    lengths = lengths.to(fluid_xy)
    mask = native_solid_mask.to(device=fluid_xy.device)
    spacing = lengths / lengths.new_tensor([nx, ny])
    coordinate = fluid_xy / spacing[:, None] - 0.5
    integer = coordinate.round()
    tolerance = min(8 * torch.finfo(fluid_xy.dtype).eps * max(nx, ny), 1.0e-3)
    if bool(((coordinate - integer).abs() > tolerance).any()):
        raise ValueError("Native curl is defined only on native cell-centred receivers.")
    ix, iy = integer[..., 0].long(), integer[..., 1].long()
    if bool(((ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)).any()):
        raise ValueError("Native curl receiver lies outside the native grid.")
    left, right = (ix - 1).clamp_min(0), (ix + 1).clamp_max(nx - 1)
    down, up = (iy - 1).clamp_min(0), (iy + 1).clamp_max(ny - 1)
    primary = iy * nx + ix
    nodes = torch.stack((primary, iy * nx + left, iy * nx + right,
                         down * nx + ix, up * nx + ix), dim=-1)
    unions, gather, counts = [], [], []
    for row in range(batch):
        # Column-major flattening places every primary before any neighbour.
        ordered_input = nodes[row].transpose(0, 1).reshape(-1)
        unique, inverse = torch.unique(ordered_input, sorted=True, return_inverse=True)
        first = torch.full_like(unique, ordered_input.numel())
        first.scatter_reduce_(0, inverse, torch.arange(ordered_input.numel(), device=fluid_xy.device),
                              reduce="amin", include_self=True)
        order = first.argsort()
        reverse = torch.empty_like(order)
        reverse[order] = torch.arange(order.numel(), device=fluid_xy.device)
        union = unique[order]
        unions.append(union)
        gather.append(reverse[inverse].reshape(5, queries).transpose(0, 1))
        counts.append(union.numel())
    width = max(counts)
    node_indices = torch.stack([torch.cat((union, union[:1].expand(width - union.numel())))
                                for union in unions])
    grid_xy = torch.stack((node_indices % nx + 0.5, node_indices // nx + 0.5), dim=-1).to(fluid_xy)
    query_xy = grid_xy * spacing[:, None]
    batch_indices = torch.arange(batch, device=fluid_xy.device)[:, None, None]
    solid_nodes = mask.reshape(batch, -1)[batch_indices, nodes]
    return NativeCurlReadStencil(
        query_xy=query_xy, node_indices=node_indices,
        gather_indices=torch.stack(gather), solid_nodes=solid_nodes,
        x_span=(right - left).to(fluid_xy) * spacing[:, 0:1],
        y_span=(up - down).to(fluid_xy) * spacing[:, 1:2],
        unique_counts=torch.tensor(counts, device=fluid_xy.device, dtype=torch.long),
    )


def physical_uvp_native_curl(normalized_uvp, stencil, field_mean, field_std):
    """Return physical u/v/p/omega, coupling omega supervision to velocity.

    Solid velocities are physically zero before taking derivatives. Physical
    curl is also zero at solid centres, before any downstream normalization.
    Pressure retains its own nonlinear read. No omega projection exists.
    """
    if normalized_uvp.ndim != 3 or normalized_uvp.shape[-1] != 3:
        raise ValueError("Native curl needs three actual nonlinear u/v/p outputs.")
    if normalized_uvp.dtype not in (torch.float32, torch.float64):
        raise ValueError("Native curl derivative arithmetic requires float32/float64 fields.")
    if normalized_uvp.shape[:2] != stencil.node_indices.shape:
        raise ValueError("Nonlinear outputs do not match the bound native node union.")
    mean = torch.as_tensor(field_mean, device=normalized_uvp.device, dtype=normalized_uvp.dtype).reshape(-1)
    std = torch.as_tensor(field_std, device=normalized_uvp.device, dtype=normalized_uvp.dtype).reshape(-1)
    if mean.numel() < 4 or std.numel() < 4 or not bool(torch.isfinite(mean[:4]).all()) or not bool(torch.isfinite(std[:4]).all()) or bool((std[:4] <= 0).any()):
        raise ValueError("Native curl requires four finite means and positive physical scales.")
    physical = normalized_uvp * std[:3] + mean[:3]
    batch = torch.arange(physical.shape[0], device=physical.device)[:, None, None]
    sampled = physical[batch, stencil.gather_indices]
    u = torch.where(stencil.solid_nodes, torch.zeros_like(sampled[..., 0]), sampled[..., 0])
    v = torch.where(stencil.solid_nodes, torch.zeros_like(sampled[..., 1]), sampled[..., 1])
    curl = (v[..., 2] - v[..., 1]) / stencil.x_span - (u[..., 4] - u[..., 3]) / stencil.y_span
    curl = torch.where(stencil.solid_nodes[..., 0], torch.zeros_like(curl), curl)
    return torch.stack((u[..., 0], v[..., 0], sampled[..., 0, 2], curl), dim=-1)
