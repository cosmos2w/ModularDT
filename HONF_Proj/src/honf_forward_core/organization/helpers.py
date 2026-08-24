"""CORE HONF hypergraph organizer.

Inputs are encoded generic module tokens, environment tokens, module centers,
environment coordinates, and a module-present mask. Outputs are A_me, A_mh,
A_eh, hyperedge states, source/region coordinates, mechanism descriptors, and
diagnostics used by the field decoder and visualizers. This module is reusable
across domains and contains no ChannelThermal-specific wall or inlet logic.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import UnifiedForwardConfig
from ..routing import normalize_assignment


EPS = 1e-6


def _masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """Normalize logits along ``dim`` while assigning masked entries zero mass."""

    if mask is None:
        return torch.softmax(logits, dim=dim)
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    masked = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
    out = torch.softmax(masked, dim=dim) * mask
    return out / out.sum(dim=dim, keepdim=True).clamp_min(EPS)


def _as_batched_coords(coords: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast shared ``[N,2]`` coordinates to ``[B,N,2]`` when needed."""

    if coords.ndim == 2:
        return coords.unsqueeze(0).expand(batch_size, -1, -1)
    return coords


def _relative_delta(src: torch.Tensor, dst: torch.Tensor, cfg: UnifiedForwardConfig) -> torch.Tensor:
    """Compute ``dst-src``, using minimum-image offsets for periodic domains."""

    delta = dst - src
    periodic_axes = cfg.periodic_dimensions()
    if periodic_axes:
        scale_x, scale_y = cfg.spatial_scale()
        lengths = torch.tensor(
            [max(scale_x, EPS), max(scale_y, EPS)],
            device=delta.device,
            dtype=delta.dtype,
        )
        wrapped = torch.remainder(delta + 0.5 * lengths, lengths) - 0.5 * lengths
        mask = torch.tensor(
            [axis in periodic_axes for axis in range(2)],
            device=delta.device,
            dtype=torch.bool,
        )
        delta = torch.where(mask, wrapped, delta)
    return delta


def _weighted_coords(coords: torch.Tensor, weights: torch.Tensor, cfg: UnifiedForwardConfig) -> torch.Tensor:
    """Reduce node coordinates ``[B,N,2]`` into ``K`` weighted centroids."""

    denom = weights.sum(dim=1).clamp_min(EPS).unsqueeze(-1)
    periodic_axes = cfg.periodic_dimensions()
    if not periodic_axes:
        return torch.einsum("bnk,bnd->bkd", weights, coords) / denom

    scale_x, scale_y = cfg.spatial_scale()
    lengths = torch.tensor(
        [max(scale_x, EPS), max(scale_y, EPS)],
        device=coords.device,
        dtype=coords.dtype,
    )
    if periodic_axes != (0, 1):
        normalized_weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(EPS)
        outputs = []
        for axis in range(2):
            if axis in periodic_axes:
                angles = 2.0 * math.pi * coords[..., axis] / lengths[axis]
                sin_sum = torch.einsum("bnk,bn->bk", normalized_weights, torch.sin(angles))
                cos_sum = torch.einsum("bnk,bn->bk", normalized_weights, torch.cos(angles))
                mean = torch.remainder(torch.atan2(sin_sum, cos_sum) / (2.0 * math.pi) * lengths[axis], lengths[axis])
            else:
                mean = torch.einsum("bnk,bn->bk", weights, coords[..., axis]) / denom.squeeze(-1)
            outputs.append(mean)
        return torch.stack(outputs, dim=-1)
    angles = 2.0 * math.pi * coords[:, None, :, :] / lengths
    weight_t = (weights / weights.sum(dim=1, keepdim=True).clamp_min(EPS)).transpose(1, 2).unsqueeze(-1)
    sin_sum = (weight_t * torch.sin(angles)).sum(dim=2)
    cos_sum = (weight_t * torch.cos(angles)).sum(dim=2)
    mean_angle = torch.atan2(sin_sum, cos_sum)
    return torch.remainder(mean_angle / (2.0 * math.pi) * lengths, lengths)


def _weighted_scale(
    coords: torch.Tensor,
    weights: torch.Tensor,
    centroids: torch.Tensor,
    cfg: UnifiedForwardConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return diagonal variance and scale for weighted node coordinates."""

    delta = _relative_delta(centroids[:, None, :, :], coords[:, :, None, :], cfg)
    variance = torch.einsum("bnk,bnkd->bkd", weights, delta.square())
    return variance, torch.sqrt(variance.clamp_min(EPS))


def _assignment_purity(assignment: torch.Tensor) -> torch.Tensor:
    """Measure the winner-owned fraction of every assignment column."""

    winners = assignment.argmax(dim=-1)
    winner_mask = torch.nn.functional.one_hot(winners, num_classes=assignment.shape[-1]).to(assignment.dtype)
    numerator = (assignment * winner_mask).sum(dim=1)
    return numerator / assignment.sum(dim=1).clamp_min(EPS)


def _stabilize_all_edge_softmax_assignment(
    assignment: torch.Tensor,
    *,
    mass_fraction_floor: float,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep every all-edge softmax candidate strictly above its viability floor.

    The affine simplex map reserves the same lower-bound probability for each
    anonymous slot and leaves the remainder in the learned assignment. It is
    permutation equivariant, preserves unit active-token rows, and introduces
    no parameters or auxiliary regularization.
    """

    capacity = int(assignment.shape[-1])
    lower_bound = float(mass_fraction_floor) * 1.001
    if capacity * lower_bound >= 1.0:
        raise ValueError(
            "All-edge softmax viability requires "
            "edge_capacity * mass_fraction_floor < 1."
        )
    stabilized = assignment * (1.0 - capacity * lower_bound) + lower_bound
    if token_mask is not None:
        stabilized = stabilized * token_mask.to(
            device=assignment.device,
            dtype=assignment.dtype,
        )
    return stabilized


def _scheduled_stabilized_assignment(
    logits: torch.Tensor,
    *,
    entmax_blend: float,
    mass_fraction_floor: float,
    mask: Optional[torch.Tensor] = None,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Blend viability-stabilized softmax continuously into exact entmax15.

    The exact endpoints are intentional: the soft endpoint protects every
    anonymous candidate while roles are forming, and the sparse endpoint adds
    no artificial probability floor once the schedule has completed.
    """

    blend = min(max(float(entmax_blend), 0.0), 1.0)
    if blend >= 1.0:
        return normalize_assignment(
            logits,
            mode="entmax15",
            mask=mask,
        )
    soft_assignment = normalize_assignment(
        logits,
        mode="softmax",
        mask=mask,
    )
    stabilized_softmax = _stabilize_all_edge_softmax_assignment(
        soft_assignment,
        mass_fraction_floor=mass_fraction_floor,
        token_mask=token_mask,
    )
    if blend <= 0.0:
        return stabilized_softmax
    sparse_assignment = normalize_assignment(
        logits,
        mode="entmax15",
        mask=mask,
    )
    assignment = (1.0 - blend) * stabilized_softmax + blend * sparse_assignment
    if mask is not None:
        assignment = assignment * torch.broadcast_to(
            mask.to(device=logits.device, dtype=torch.bool),
            logits.shape,
        ).to(dtype=assignment.dtype)
    return assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(EPS)


def _masked_mass_statistics(
    mass: torch.Tensor,
    token_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-case minimum, p05, and mean retained token mass."""

    minima = []
    p05s = []
    means = []
    for batch_index in range(mass.shape[0]):
        values = mass[batch_index]
        if token_mask is not None:
            values = values[token_mask[batch_index].to(device=mass.device, dtype=torch.bool)]
        if values.numel() == 0:
            values = mass.new_zeros(1)
        minima.append(values.amin())
        p05s.append(torch.quantile(values.float(), 0.05).to(dtype=mass.dtype))
        means.append(values.mean())
    return torch.stack(minima), torch.stack(p05s), torch.stack(means)


def _descriptor_first_features(
    hyper_source_coords: torch.Tensor,
    hyper_source_scale: torch.Tensor,
    hyper_region_coords: torch.Tensor,
    hyper_region_scale: torch.Tensor,
    hyper_module_mass: torch.Tensor,
    hyper_env_mass: torch.Tensor,
    hyper_module_purity: torch.Tensor,
    hyper_env_purity: torch.Tensor,
    edge_active_mask: torch.Tensor,
    cfg: UnifiedForwardConfig,
) -> torch.Tensor:
    """Build normalized mechanism descriptors with geometry as primary state."""

    scale_x, scale_y = cfg.spatial_scale()
    scales = hyper_source_coords.new_tensor([max(scale_x, EPS), max(scale_y, EPS)])
    displacement = _relative_delta(hyper_source_coords, hyper_region_coords, cfg)
    distance_scale = max(math.sqrt(scale_x**2 + scale_y**2), EPS)
    distance = torch.sqrt(displacement.square().sum(dim=-1, keepdim=True) + EPS) / distance_scale
    return torch.cat(
        [
            hyper_source_coords / scales,
            hyper_source_scale / scales,
            hyper_region_coords / scales,
            hyper_region_scale / scales,
            displacement / scales,
            distance,
            hyper_module_mass.unsqueeze(-1),
            hyper_env_mass.unsqueeze(-1),
            hyper_module_purity.unsqueeze(-1),
            hyper_env_purity.unsqueeze(-1),
            edge_active_mask.unsqueeze(-1),
        ],
        dim=-1,
    )


def _mechanism_descriptors(
    hyper_source_coords: torch.Tensor,
    hyper_region_coords: torch.Tensor,
    hyper_module_mass: torch.Tensor,
    hyper_env_mass: torch.Tensor,
    hyper_strength: torch.Tensor,
    module_mass_raw: torch.Tensor,
    env_mass_raw: torch.Tensor,
    module_present: torch.Tensor,
    env_count: int,
    cfg: UnifiedForwardConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generic source-region descriptors for hyperedge field mechanisms.

    These features describe where a hyperedge draws from, where its field
    context is concentrated, and how much module/environment mass it carries.
    They deliberately avoid case-specific wall, plume, or thermal rules.
    """

    scale_x, scale_y = cfg.spatial_scale()
    lx = max(scale_x, EPS)
    ly = max(scale_y, EPS)
    diag = max(math.sqrt(lx * lx + ly * ly), EPS)
    displacement = _relative_delta(hyper_source_coords, hyper_region_coords, cfg)
    dx = displacement[..., 0:1]
    dy = displacement[..., 1:2]
    if 0 in cfg.periodic_dimensions():
        downstream = torch.remainder(hyper_region_coords[..., 0:1] - hyper_source_coords[..., 0:1], lx) / lx
        upstream = torch.remainder(hyper_source_coords[..., 0:1] - hyper_region_coords[..., 0:1], lx) / lx
    else:
        downstream = torch.relu(dx) / lx
        upstream = torch.relu(-dx) / lx
    lateral = dy.abs() / ly
    distance = torch.sqrt(dx.square() + dy.square() + EPS) / diag
    mechanism_geometry_features = torch.cat(
        [
            hyper_source_coords[..., 0:1] / lx,
            hyper_source_coords[..., 1:2] / ly,
            hyper_region_coords[..., 0:1] / lx,
            hyper_region_coords[..., 1:2] / ly,
            dx / lx,
            dy / ly,
            distance,
            downstream,
            upstream,
            lateral,
        ],
        dim=-1,
    )

    module_count = module_present.sum(dim=-1, keepdim=True).clamp_min(1.0)
    env_count_t = module_present.new_tensor(float(max(env_count, 1)))
    module_raw_norm = module_mass_raw / module_count
    env_raw_norm = env_mass_raw / env_count_t
    module_raw_log = torch.log1p(module_mass_raw) / torch.log1p(module_count)
    env_raw_log = torch.log1p(env_mass_raw) / torch.log1p(env_count_t)
    mechanism_mass_features = torch.stack(
        [
            hyper_module_mass,
            hyper_env_mass,
            hyper_strength,
            module_raw_norm,
            env_raw_norm,
            module_raw_log,
            env_raw_log,
        ],
        dim=-1,
    )
    mechanism_raw_features = torch.cat([mechanism_geometry_features, mechanism_mass_features], dim=-1)
    return mechanism_geometry_features, mechanism_mass_features, mechanism_raw_features, distance, downstream, lateral


def deterministic_slot_codes(
    capacity: int,
    hidden_dim: int,
    *,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate zero-mean, non-trainable candidate codes as ``[K,H]``."""

    count = int(capacity)
    width = int(hidden_dim)
    if count <= 0 or width <= 0:
        raise ValueError("Slot-code capacity and hidden dimension must be positive.")
    positions = (torch.arange(count, device=device, dtype=dtype) + 0.5) / float(count)
    dimensions = torch.arange(width, device=device)
    if mode == "sinusoidal":
        frequencies = torch.div(dimensions, 2, rounding_mode="floor").to(dtype=dtype) + 1.0
        angles = 2.0 * math.pi * positions[:, None] * frequencies[None, :]
        codes = torch.where((dimensions % 2)[None, :] == 0, torch.sin(angles), torch.cos(angles))
    elif mode == "low_discrepancy":
        golden = (math.sqrt(5.0) - 1.0) / 2.0
        codes = torch.frac(
            (torch.arange(count, device=device, dtype=dtype)[:, None] + 1.0)
            * (dimensions.to(dtype=dtype)[None, :] + 1.0)
            * golden
        )
        codes = 2.0 * codes - 1.0
    else:
        raise ValueError(f"Unsupported slot code mode: {mode!r}.")
    codes = codes - codes.mean(dim=0, keepdim=True)
    return codes / torch.sqrt(codes.square().mean().clamp_min(EPS))

