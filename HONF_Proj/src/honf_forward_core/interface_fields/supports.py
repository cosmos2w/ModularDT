"""Sparse cubic-support geometry for the interface-centred HONF.

The routines in this module construct geometry only.  They deliberately do
not contain neural scoring, group states, or a learned group count.  A layout
is represented by the supports touched by the physical module ports and by
sparse incidence lists into those supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional

import torch


@dataclass(frozen=True)
class SparseSupportLayout:
    """Flattened variable-size support geometry for a batch of layouts.

    Source indices in the incidence tensors address flattened padded source
    storage (`batch * source_width + source_slot`).  Group indices address the
    unpadded arrays in this object.  ``case_group_offsets`` therefore provides
    the group range for each physical case without padding every case to the
    largest number of groups in the batch.
    """

    lattice_keys: torch.Tensor
    centres: torch.Tensor
    group_batch: torch.Tensor
    case_group_offsets: torch.Tensor
    module_group_indices: torch.Tensor
    module_support_weight: torch.Tensor
    environment_group_indices: torch.Tensor
    environment_geometric_weight: torch.Tensor
    occupancy: torch.Tensor
    occupancy_envelope: torch.Tensor
    covered_volume_ratio: torch.Tensor
    origin: torch.Tensor
    spacing: torch.Tensor
    batch_size: int
    module_width: int
    environment_width: int
    spatial_dimension: int

    @property
    def num_groups(self) -> int:
        return int(self.lattice_keys.shape[0])

    @property
    def group_count(self) -> int:
        return self.num_groups

    @property
    def module_count(self) -> int:
        return self.module_width

    @property
    def environment_count(self) -> int:
        return self.environment_width

    @property
    def spatial_dim(self) -> int:
        return self.spatial_dimension

    @property
    def module_geometric_weights(self) -> torch.Tensor:
        """Plural alias matching generic incidence-list terminology."""

        return self.module_support_weight

    @property
    def environment_geometric_weights(self) -> torch.Tensor:
        """Plural alias matching generic incidence-list terminology."""

        return self.environment_geometric_weight


@dataclass(frozen=True)
class SparseReceiverIncidence:
    """Sparse active-support lookup for a padded receiver tensor."""

    receiver_group_indices: torch.Tensor
    support_weight: torch.Tensor
    degree: torch.Tensor
    batch_size: int
    receiver_width: int

    @property
    def geometric_weights(self) -> torch.Tensor:
        return self.support_weight

    @property
    def geometric_weight(self) -> torch.Tensor:
        return self.support_weight


def cubic_bspline(relative_coordinate: torch.Tensor) -> torch.Tensor:
    """Evaluate the centred cardinal cubic B-spline ``B_3`` elementwise."""

    distance = relative_coordinate.abs()
    inner = (4.0 - 6.0 * distance.square() + 3.0 * distance.pow(3)) / 6.0
    outer = (2.0 - distance).clamp_min(0.0).pow(3) / 6.0
    return torch.where(distance < 1.0, inner, torch.where(distance < 2.0, outer, torch.zeros_like(distance)))


def tensor_product_cubic_support(
    coordinates: torch.Tensor,
    centres: torch.Tensor,
    spacing: torch.Tensor | float,
) -> torch.Tensor:
    """Evaluate tensor-product cubic support for pairwise-aligned points."""

    spacing_tensor = torch.as_tensor(spacing, device=coordinates.device, dtype=coordinates.dtype)
    return cubic_bspline((coordinates - centres) / spacing_tensor).prod(dim=-1)


def _origins_for_batch(
    origin: Optional[torch.Tensor],
    *,
    batch_size: int,
    dimension: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if origin is None:
        return reference.new_zeros((batch_size, dimension))
    origins = torch.as_tensor(origin, device=reference.device, dtype=reference.dtype)
    if origins.shape == (dimension,):
        return origins.unsqueeze(0).expand(batch_size, -1)
    if origins.shape == (batch_size, dimension):
        return origins
    raise ValueError(
        f"origin must have shape [{dimension}] or [{batch_size},{dimension}], got {tuple(origins.shape)}."
    )


def _candidate_offsets(dimension: int, device: torch.device) -> torch.Tensor:
    # floor(x / Delta) + {-1, 0, 1, 2} contains every support with nonzero
    # cubic overlap at a generic point.  Keys at an exact outer boundary have
    # zero weight and enter/leave continuously.
    return torch.tensor(list(product((-1, 0, 1, 2), repeat=dimension)), device=device, dtype=torch.long)


@dataclass(frozen=True)
class _Candidates:
    batch: torch.Tensor
    source: torch.Tensor
    keys: torch.Tensor
    support_weight: torch.Tensor


def _enumerate_candidates(
    coordinates: torch.Tensor,
    source_batch: torch.Tensor,
    source_indices: torch.Tensor,
    origins: torch.Tensor,
    spacing: torch.Tensor,
) -> _Candidates:
    """Enumerate only the local ``4**d`` lattice candidates per source."""

    dimension = int(coordinates.shape[-1])
    offsets = _candidate_offsets(dimension, coordinates.device)
    normalized = (coordinates - origins.index_select(0, source_batch)) / spacing
    # The integer topology is intentionally nondifferentiable.  Smooth support
    # values retain coordinate gradients within each local candidate set.
    base_keys = torch.floor(normalized.detach()).to(torch.long)
    keys = base_keys[:, None, :] + offsets[None, :, :]
    relative = normalized[:, None, :] - keys.to(dtype=normalized.dtype)
    weights = cubic_bspline(relative).prod(dim=-1)
    candidate_count = int(offsets.shape[0])
    return _Candidates(
        batch=source_batch.repeat_interleave(candidate_count),
        source=source_indices.repeat_interleave(candidate_count),
        keys=keys.reshape(-1, dimension),
        support_weight=weights.reshape(-1),
    )


def _match_active_groups(
    active_table: torch.Tensor,
    candidate_table: torch.Tensor,
) -> torch.Tensor:
    """Match candidate ``[batch,key...]`` rows to active rows without all-pairs work."""

    candidate_count = int(candidate_table.shape[0])
    if candidate_count == 0:
        return candidate_table.new_empty((0,), dtype=torch.long)
    if active_table.shape[0] == 0:
        return candidate_table.new_full((candidate_count,), -1, dtype=torch.long)
    joined = torch.cat([active_table, candidate_table], dim=0)
    _, inverse = torch.unique(joined, dim=0, sorted=True, return_inverse=True)
    active_inverse = inverse[: active_table.shape[0]]
    candidate_inverse = inverse[active_table.shape[0] :]
    unique_to_group = inverse.new_full((int(inverse.max().item()) + 1,), -1)
    unique_to_group[active_inverse] = torch.arange(active_table.shape[0], device=active_table.device)
    return unique_to_group.index_select(0, candidate_inverse)


def _empty_layout(
    *,
    port_coordinates: torch.Tensor,
    environment_coordinates: torch.Tensor,
    origins: torch.Tensor,
    spacing: torch.Tensor,
) -> SparseSupportLayout:
    batch_size, module_width, _, dimension = port_coordinates.shape
    environment_width = int(environment_coordinates.shape[1])
    long_empty = torch.empty((0,), device=port_coordinates.device, dtype=torch.long)
    float_empty = torch.empty((0,), device=port_coordinates.device, dtype=port_coordinates.dtype)
    return SparseSupportLayout(
        lattice_keys=torch.empty((0, dimension), device=port_coordinates.device, dtype=torch.long),
        centres=torch.empty((0, dimension), device=port_coordinates.device, dtype=port_coordinates.dtype),
        group_batch=long_empty,
        case_group_offsets=torch.zeros((batch_size + 1,), device=port_coordinates.device, dtype=torch.long),
        module_group_indices=torch.empty((2, 0), device=port_coordinates.device, dtype=torch.long),
        module_support_weight=float_empty,
        environment_group_indices=torch.empty((2, 0), device=port_coordinates.device, dtype=torch.long),
        environment_geometric_weight=float_empty,
        occupancy=float_empty,
        occupancy_envelope=float_empty,
        covered_volume_ratio=float_empty,
        origin=origins,
        spacing=spacing,
        batch_size=batch_size,
        module_width=module_width,
        environment_width=environment_width,
        spatial_dimension=dimension,
    )


def build_sparse_support_layout(
    port_coordinates: torch.Tensor,
    module_present: torch.Tensor,
    environment_coordinates: torch.Tensor,
    environment_weights: torch.Tensor,
    *,
    spacing: torch.Tensor | float,
    port_quadrature_weights: Optional[torch.Tensor] = None,
    origin: Optional[torch.Tensor] = None,
) -> SparseSupportLayout:
    """Build occupied groups and sparse module/environment incidences.

    Args:
        port_coordinates: Physical footprint samples ``[B,M,P,d]``.
        module_present: Active-module mask ``[B,M]``.
        environment_coordinates: Environmental quadrature points ``[B,E,d]``.
        environment_weights: Positive physical volume weights ``[B,E]``.
        spacing: Physical lattice spacing ``Delta``.
        port_quadrature_weights: Optional nonnegative footprint quadrature
            ``[B,M,P]``.  Equal weights are used when omitted.
        origin: Shared ``[d]`` or case-local ``[B,d]`` lattice origin.

    The work prior to neural processing is proportional to
    ``4**d * (active ports + environment points)`` plus coalescing/search.  No
    dense module-by-group or environment-by-group array is materialized.
    """

    if port_coordinates.ndim != 4:
        raise ValueError(f"port_coordinates must be [B,M,P,d], got {tuple(port_coordinates.shape)}.")
    if environment_coordinates.ndim != 3:
        raise ValueError(
            f"environment_coordinates must be [B,E,d], got {tuple(environment_coordinates.shape)}."
        )
    batch_size, module_width, port_count, dimension = port_coordinates.shape
    if module_present.shape != (batch_size, module_width):
        raise ValueError(
            f"module_present must be [{batch_size},{module_width}], got {tuple(module_present.shape)}."
        )
    if environment_coordinates.shape[0] != batch_size or environment_coordinates.shape[-1] != dimension:
        raise ValueError("Port and environment coordinates must share batch size and spatial dimension.")
    environment_width = int(environment_coordinates.shape[1])
    if environment_weights.shape != (batch_size, environment_width):
        raise ValueError(
            f"environment_weights must be [{batch_size},{environment_width}], got {tuple(environment_weights.shape)}."
        )
    if port_coordinates.device != environment_coordinates.device:
        raise ValueError("Port and environment coordinates must be on the same device.")
    if not torch.is_floating_point(port_coordinates) or not torch.is_floating_point(environment_coordinates):
        raise ValueError("Support coordinates must use a floating-point dtype.")

    spacing_tensor = torch.as_tensor(spacing, device=port_coordinates.device, dtype=port_coordinates.dtype)
    if spacing_tensor.numel() != 1 or not bool(torch.isfinite(spacing_tensor).all()) or float(spacing_tensor) <= 0.0:
        raise ValueError("spacing must be one finite positive scalar.")
    spacing_tensor = spacing_tensor.reshape(())
    origins = _origins_for_batch(
        origin,
        batch_size=batch_size,
        dimension=dimension,
        reference=port_coordinates,
    )

    env_weights = environment_weights.to(device=port_coordinates.device, dtype=port_coordinates.dtype)
    if not bool(torch.isfinite(env_weights).all()) or bool((env_weights <= 0.0).any()):
        raise ValueError("environment_weights must be finite and strictly positive.")
    if port_quadrature_weights is None:
        port_weights = port_coordinates.new_ones((batch_size, module_width, port_count))
    else:
        if port_quadrature_weights.shape != (batch_size, module_width, port_count):
            raise ValueError(
                "port_quadrature_weights must match the first three port_coordinates dimensions."
            )
        port_weights = port_quadrature_weights.to(device=port_coordinates.device, dtype=port_coordinates.dtype)
        if not bool(torch.isfinite(port_weights).all()) or bool((port_weights < 0.0).any()):
            raise ValueError("port_quadrature_weights must be finite and nonnegative.")

    active_modules = torch.nonzero(module_present.to(device=port_coordinates.device) > 0.5, as_tuple=False)
    if active_modules.shape[0] == 0:
        return _empty_layout(
            port_coordinates=port_coordinates,
            environment_coordinates=environment_coordinates,
            origins=origins,
            spacing=spacing_tensor,
        )
    active_batch = active_modules[:, 0]
    active_slot = active_modules[:, 1]
    active_ports = port_coordinates[active_batch, active_slot]
    active_port_weights = port_weights[active_batch, active_slot]
    weight_sums = active_port_weights.sum(dim=-1, keepdim=True)
    if bool((weight_sums <= 0.0).any()):
        raise ValueError("Every active module must have positive total port quadrature weight.")
    normalized_port_weights = active_port_weights / weight_sums

    port_source = (active_batch * module_width + active_slot).repeat_interleave(port_count)
    port_batch = active_batch.repeat_interleave(port_count)
    port_candidates = _enumerate_candidates(
        active_ports.reshape(-1, dimension),
        port_batch,
        port_source,
        origins,
        spacing_tensor,
    )
    candidate_count_per_point = 4**dimension
    port_quadrature = normalized_port_weights.reshape(-1).repeat_interleave(candidate_count_per_point)
    candidate_membership = port_quadrature * port_candidates.support_weight
    retained_port = candidate_membership > 0.0
    retained_batch = port_candidates.batch[retained_port]
    retained_source = port_candidates.source[retained_port]
    retained_keys = port_candidates.keys[retained_port]
    retained_membership = candidate_membership[retained_port]

    active_table, candidate_to_group = torch.unique(
        torch.cat([retained_batch[:, None], retained_keys], dim=-1),
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    group_batch = active_table[:, 0]
    lattice_keys = active_table[:, 1:]
    num_groups = int(lattice_keys.shape[0])
    centres = origins.index_select(0, group_batch) + spacing_tensor * lattice_keys.to(port_coordinates.dtype)

    module_group_rows = torch.stack([retained_source, candidate_to_group], dim=-1)
    unique_module_group, module_inverse = torch.unique(
        module_group_rows, dim=0, sorted=True, return_inverse=True
    )
    module_support_weight = port_coordinates.new_zeros((unique_module_group.shape[0],))
    module_support_weight.index_add_(0, module_inverse, retained_membership)
    module_group_indices = unique_module_group.transpose(0, 1).contiguous()
    occupancy = port_coordinates.new_zeros((num_groups,))
    occupancy.index_add_(0, module_group_indices[1], module_support_weight)
    occupancy_envelope = -torch.expm1(-(float(4**dimension)) * occupancy)

    env_batch = torch.arange(batch_size, device=port_coordinates.device).repeat_interleave(environment_width)
    env_source = torch.arange(
        batch_size * environment_width, device=port_coordinates.device, dtype=torch.long
    )
    env_candidates = _enumerate_candidates(
        environment_coordinates.to(dtype=port_coordinates.dtype).reshape(-1, dimension),
        env_batch,
        env_source,
        origins,
        spacing_tensor,
    )
    retained_env_candidate = env_candidates.support_weight > 0.0
    env_candidate_table = torch.cat(
        [env_candidates.batch[retained_env_candidate, None], env_candidates.keys[retained_env_candidate]], dim=-1
    )
    matched_groups = _match_active_groups(active_table, env_candidate_table)
    env_volume = env_weights.reshape(-1).repeat_interleave(candidate_count_per_point)[retained_env_candidate]
    env_geometric = env_volume * env_candidates.support_weight[retained_env_candidate]
    matched = (matched_groups >= 0) & (env_geometric > 0.0)
    environment_group_indices = torch.stack(
        [env_candidates.source[retained_env_candidate][matched], matched_groups[matched]], dim=0
    )
    environment_geometric_weight = env_geometric[matched]
    covered_volume_ratio = port_coordinates.new_zeros((num_groups,))
    covered_volume_ratio.index_add_(
        0,
        environment_group_indices[1],
        environment_geometric_weight / spacing_tensor.pow(dimension),
    )

    groups_per_case = torch.bincount(group_batch, minlength=batch_size)
    case_group_offsets = torch.cat(
        [torch.zeros((1,), device=port_coordinates.device, dtype=torch.long), groups_per_case.cumsum(dim=0)]
    )
    return SparseSupportLayout(
        lattice_keys=lattice_keys,
        centres=centres,
        group_batch=group_batch,
        case_group_offsets=case_group_offsets,
        module_group_indices=module_group_indices,
        module_support_weight=module_support_weight,
        environment_group_indices=environment_group_indices,
        environment_geometric_weight=environment_geometric_weight,
        occupancy=occupancy,
        occupancy_envelope=occupancy_envelope,
        covered_volume_ratio=covered_volume_ratio,
        origin=origins,
        spacing=spacing_tensor,
        batch_size=batch_size,
        module_width=module_width,
        environment_width=environment_width,
        spatial_dimension=dimension,
    )


def lookup_sparse_supports(
    layout: SparseSupportLayout,
    receiver_coordinates: torch.Tensor,
) -> SparseReceiverIncidence:
    """Return only active cubic-support incidences for each receiver."""

    if receiver_coordinates.ndim != 3:
        raise ValueError(f"receiver_coordinates must be [B,Q,d], got {tuple(receiver_coordinates.shape)}.")
    batch_size, receiver_width, dimension = receiver_coordinates.shape
    if batch_size != layout.batch_size or dimension != layout.spatial_dimension:
        raise ValueError("Receiver coordinates do not match the support layout batch/dimension.")
    if receiver_coordinates.device != layout.centres.device:
        raise ValueError("Receiver coordinates and support layout must be on the same device.")
    receiver_batch = torch.arange(batch_size, device=receiver_coordinates.device).repeat_interleave(receiver_width)
    receiver_source = torch.arange(
        batch_size * receiver_width, device=receiver_coordinates.device, dtype=torch.long
    )
    candidates = _enumerate_candidates(
        receiver_coordinates.reshape(-1, dimension),
        receiver_batch,
        receiver_source,
        layout.origin,
        layout.spacing,
    )
    nonzero = candidates.support_weight > 0.0
    candidate_table = torch.cat([candidates.batch[nonzero, None], candidates.keys[nonzero]], dim=-1)
    active_table = torch.cat([layout.group_batch[:, None], layout.lattice_keys], dim=-1)
    matched_group = _match_active_groups(active_table, candidate_table)
    matched = matched_group >= 0
    receiver_indices = candidates.source[nonzero][matched]
    support_weight = candidates.support_weight[nonzero][matched]
    receiver_group_indices = torch.stack([receiver_indices, matched_group[matched]], dim=0)
    flat_degree = torch.bincount(receiver_indices, minlength=batch_size * receiver_width)
    degree = flat_degree.reshape(batch_size, receiver_width)
    return SparseReceiverIncidence(
        receiver_group_indices=receiver_group_indices,
        support_weight=support_weight,
        degree=degree,
        batch_size=batch_size,
        receiver_width=receiver_width,
    )


# Concise public name used by the backend's continuous reader.
lookup_receivers = lookup_sparse_supports
