"""Case-owned environmental response hierarchies.

The hierarchy in this module is deliberately small and mostly geometric.  It
contains the integer topology needed by a batched tree walk and the physical
boxes used by the walk; learned tensors are produced by the backend on every
preparation pass.  In particular, this module does not infer a grid from token
order and it never stores a query-by-node incidence matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import torch


def _as_positive_shape(value: Sequence[int] | int, dimension: int, *, name: str) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        result = (int(value),) * dimension
    else:
        try:
            result = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain positive integers.") from exc
        if len(result) != dimension:
            raise ValueError(f"{name} must contain {dimension} entries.")
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive integers.")
    return result


def _broadcast_batch_tensor(
    tensor: torch.Tensor,
    batch: int,
    trailing_ndim: int,
    *,
    name: str,
) -> torch.Tensor:
    """Return a tensor with an explicit leading batch axis."""

    if tensor.ndim == trailing_ndim:
        return tensor.unsqueeze(0).expand((batch,) + tuple(tensor.shape))
    if tensor.ndim == trailing_ndim + 1 and int(tensor.shape[0]) in (1, batch):
        return tensor.expand((batch,) + tuple(tensor.shape[1:]))
    raise ValueError(f"{name} has incompatible shape {tuple(tensor.shape)} for batch {batch}.")


@dataclass(frozen=True)
class EnvironmentHierarchy:
    """Lightweight rectangular-tree metadata supplied by a case adapter.

    ``fine_to_leaf`` contains global node indices, rather than compact leaf
    slots.  This makes the same metadata useful for reductions and traversal.
    It may be ``[E]``/``[N]`` for one geometry or ``[B,E]``/``[B,N,D]`` when
    an adapter has already expanded a geometry over a batch.  The integer
    topology is shared over that batch.  Floating bounds and optional static
    occupancy are intentionally retained as tensors so a caller can move the
    complete object with :meth:`to` without rebuilding it.
    """

    fine_to_leaf: torch.Tensor
    parent_index: torch.Tensor
    children: torch.Tensor
    level: torch.Tensor
    level_offsets: torch.Tensor
    bounds_min: torch.Tensor
    bounds_max: torch.Tensor
    leaf_nodes: torch.Tensor
    node_valid: torch.Tensor | None = None
    grid_shape: tuple[int, ...] = ()
    block_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("fine_to_leaf", "parent_index", "children", "level", "level_offsets", "leaf_nodes"):
            value = getattr(self, name)
            if not torch.is_tensor(value):
                raise TypeError(f"EnvironmentHierarchy.{name} must be a tensor.")
        for name in ("bounds_min", "bounds_max"):
            value = getattr(self, name)
            if not torch.is_tensor(value) or value.ndim not in (2, 3):
                raise ValueError(f"EnvironmentHierarchy.{name} must have shape [N,D] or [B,N,D].")
        node_count = int(self.parent_index.numel())
        if self.parent_index.ndim != 1 or self.level.ndim != 1 or self.level.shape != self.parent_index.shape:
            raise ValueError("parent_index and level must both have shape [N].")
        if self.children.ndim != 2 or int(self.children.shape[0]) != node_count:
            raise ValueError("children must have shape [N,max_children].")
        if self.bounds_min.shape[-2] != node_count or self.bounds_max.shape != self.bounds_min.shape:
            raise ValueError("Hierarchy bounds must align with the node count.")
        if self.level_offsets.ndim != 1 or int(self.level_offsets[0].item()) != 0:
            raise ValueError("level_offsets must be a one-dimensional cumulative offset tensor.")
        if int(self.level_offsets[-1].item()) != node_count:
            raise ValueError("level_offsets must end at the node count.")
        if self.fine_to_leaf.ndim not in (1, 2):
            raise ValueError("fine_to_leaf must have shape [E] or [B,E].")
        if self.leaf_nodes.ndim != 1:
            raise ValueError("leaf_nodes must have shape [n_leaf].")
        if self.node_valid is not None:
            if self.node_valid.ndim not in (1, 2):
                raise ValueError("node_valid must have shape [N] or [B,N].")
            if int(self.node_valid.shape[-1]) != node_count:
                raise ValueError("node_valid must align with the node count.")
        if self.grid_shape and any(int(item) <= 0 for item in self.grid_shape):
            raise ValueError("grid_shape must contain positive integers.")
        if self.block_shape and any(int(item) <= 0 for item in self.block_shape):
            raise ValueError("block_shape must contain positive integers.")

    @property
    def num_nodes(self) -> int:
        return int(self.parent_index.numel())

    @property
    def node_count(self) -> int:
        return self.num_nodes

    @property
    def num_levels(self) -> int:
        return int(self.level_offsets.numel()) - 1

    @property
    def level_sizes(self) -> tuple[int, ...]:
        offsets = self.level_offsets.detach().cpu().tolist()
        return tuple(int(stop - start) for start, stop in zip(offsets[:-1], offsets[1:], strict=True))

    @property
    def leaf_ids(self) -> torch.Tensor:
        """Alias used by adapters that describe sample-to-leaf membership."""

        return self.fine_to_leaf

    @property
    def parent_indices(self) -> torch.Tensor:
        return self.parent_index

    @property
    def child_indices(self) -> torch.Tensor:
        return self.children

    @property
    def node_levels(self) -> torch.Tensor:
        return self.level

    @property
    def centers(self) -> torch.Tensor:
        return 0.5 * (self.bounds_min + self.bounds_max)

    @property
    def radii(self) -> torch.Tensor:
        span = 0.5 * (self.bounds_max - self.bounds_min)
        # Degenerate internal cells can occur in a partially filled rectangular
        # grid.  Leaves never use a radius; this positive fallback keeps their
        # metadata well-defined for diagnostics and unary internal nodes stable.
        return torch.linalg.vector_norm(span, dim=-1).clamp_min(torch.finfo(span.dtype).eps)

    def level_slice(self, level: int) -> slice:
        level = int(level)
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"level must be between 0 and {self.num_levels - 1}.")
        start = int(self.level_offsets[level].item())
        stop = int(self.level_offsets[level + 1].item())
        return slice(start, stop)

    def expand_batch(self, batch: int) -> "EnvironmentHierarchy":
        """Expand shared metadata to ``batch`` without copying storage."""

        batch = int(batch)
        if batch <= 0:
            raise ValueError("batch must be positive.")
        current_batch = int(self.fine_to_leaf.shape[0]) if self.fine_to_leaf.ndim == 2 else 1
        if (
            self.fine_to_leaf.ndim == 2
            and current_batch == batch
            and self.bounds_min.ndim == 3
            and int(self.bounds_min.shape[0]) == batch
        ):
            return self
        fine = _broadcast_batch_tensor(self.fine_to_leaf, batch, 1, name="fine_to_leaf")
        bounds_min = _broadcast_batch_tensor(self.bounds_min, batch, 2, name="bounds_min")
        bounds_max = _broadcast_batch_tensor(self.bounds_max, batch, 2, name="bounds_max")
        valid = None
        if self.node_valid is not None:
            valid = _broadcast_batch_tensor(self.node_valid, batch, 1, name="node_valid")
        return replace(self, fine_to_leaf=fine, bounds_min=bounds_min, bounds_max=bounds_max, node_valid=valid)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "EnvironmentHierarchy":
        """Move all tensor metadata while preserving the immutable topology."""

        target = torch.device(device)
        if all(value.device == target for value in (
            self.fine_to_leaf,
            self.parent_index,
            self.children,
            self.level,
            self.level_offsets,
            self.bounds_min,
            self.bounds_max,
            self.leaf_nodes,
        )) and (self.node_valid is None or self.node_valid.device == target):
            return self

        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(device=device, non_blocking=non_blocking)

        return replace(
            self,
            fine_to_leaf=self.fine_to_leaf.to(device=target, non_blocking=non_blocking),
            parent_index=self.parent_index.to(device=target, non_blocking=non_blocking),
            children=self.children.to(device=target, non_blocking=non_blocking),
            level=self.level.to(device=target, non_blocking=non_blocking),
            level_offsets=self.level_offsets.to(device=target, non_blocking=non_blocking),
            bounds_min=self.bounds_min.to(device=target, non_blocking=non_blocking),
            bounds_max=self.bounds_max.to(device=target, non_blocking=non_blocking),
            leaf_nodes=self.leaf_nodes.to(device=target, non_blocking=non_blocking),
            node_valid=move(self.node_valid),
        )


@dataclass(frozen=True)
class HierarchyGeometry:
    """Live mass and centroid metadata reusable across preparation passes."""

    mass: torch.Tensor
    coordinates: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.mass.ndim not in (1, 2) or self.coordinates.ndim not in (2, 3):
            raise ValueError("HierarchyGeometry tensors must be batched or unbatched.")
        if self.valid.shape != self.mass.shape:
            raise ValueError("HierarchyGeometry.valid must align with mass.")
        if self.coordinates.shape[:-1] != self.mass.shape:
            raise ValueError("HierarchyGeometry.coordinates must align with mass.")

    @property
    def centroids(self) -> torch.Tensor:
        return self.coordinates

    def expand_batch(self, batch: int) -> "HierarchyGeometry":
        batch = int(batch)
        if batch <= 0:
            raise ValueError("batch must be positive.")
        mass = _broadcast_batch_tensor(self.mass, batch, 1, name="geometry.mass")
        coordinates = _broadcast_batch_tensor(self.coordinates, batch, 2, name="geometry.coordinates")
        valid = _broadcast_batch_tensor(self.valid, batch, 1, name="geometry.valid")
        if (
            self.mass.ndim == 2
            and int(self.mass.shape[0]) == batch
            and self.coordinates.ndim == 3
            and int(self.coordinates.shape[0]) == batch
        ):
            return self
        return replace(self, mass=mass, coordinates=coordinates, valid=valid)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "HierarchyGeometry":
        target = torch.device(device)
        if self.mass.device == target and self.coordinates.device == target and self.valid.device == target:
            return self
        return replace(
            self,
            mass=self.mass.to(device=target, non_blocking=non_blocking),
            coordinates=self.coordinates.to(device=target, non_blocking=non_blocking),
            valid=self.valid.to(device=target, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class HierarchyStatistics:
    """Weighted, pre-update sufficient statistics for every tree node."""

    mass: torch.Tensor
    coordinates: torch.Tensor
    environment: torch.Tensor
    response: torch.Tensor
    valid: torch.Tensor

    @property
    def centroids(self) -> torch.Tensor:
        return self.coordinates

    @property
    def env_mean(self) -> torch.Tensor:
        return self.environment

    @property
    def response_mean(self) -> torch.Tensor:
        return self.response


def _normalise_coordinates(coordinates: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    if coordinates.ndim == 2:
        return coordinates.unsqueeze(0), 1, int(coordinates.shape[0]), int(coordinates.shape[1])
    if coordinates.ndim == 3:
        return coordinates, int(coordinates.shape[0]), int(coordinates.shape[1]), int(coordinates.shape[2])
    raise ValueError("coordinates must have shape [E,D] or [B,E,D].")


def _normalise_weights(weights: torch.Tensor | None, batch: int, environment: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if weights is None:
        return None
    if weights.ndim == 1:
        if int(weights.shape[0]) != environment:
            raise ValueError("weights does not align with coordinates.")
        result = weights.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1)
    elif weights.ndim == 2 and tuple(weights.shape) in ((batch, environment), (1, environment)):
        result = weights.to(device=device, dtype=dtype).expand(batch, -1)
    else:
        raise ValueError("weights must have shape [E] or [B,E].")
    if not torch.isfinite(result).all() or torch.any(result < 0):
        raise ValueError("environment weights must be nonnegative.")
    return result


def _normalise_bounds(bounds: Sequence[Sequence[float]] | torch.Tensor | None, dimension: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if bounds is None:
        return None
    tensor = torch.as_tensor(bounds, device=device, dtype=dtype)
    if tensor.shape == (dimension, 2):
        result = tensor
    elif tensor.shape == (2, dimension):
        result = tensor.transpose(0, 1)
    else:
        raise ValueError(f"bounds must have shape [{dimension},2] or [2,{dimension}].")
    if not torch.isfinite(result).all() or torch.any(result[:, 1] <= result[:, 0]):
        raise ValueError("bounds must contain finite increasing intervals.")
    return result


def prepare_hierarchy_geometry(
    hierarchy: EnvironmentHierarchy,
    weights: torch.Tensor,
    coordinates: torch.Tensor,
) -> HierarchyGeometry:
    """Compute live node masses and centroids once for an encoded case."""

    coords, batch, environment, dimension = _normalise_coordinates(coordinates)
    batch_weights = _normalise_weights(
        weights,
        batch,
        environment,
        device=coords.device,
        dtype=coords.dtype,
    )
    if batch_weights is None:
        batch_weights = coords.new_ones(batch, environment)
    hierarchy = hierarchy.to(coords.device).expand_batch(batch)
    if int(hierarchy.fine_to_leaf.shape[-1]) != environment:
        raise ValueError("Hierarchy geometry membership must align with coordinates.")
    fine_to_leaf = hierarchy.fine_to_leaf.to(device=coords.device, dtype=torch.long)
    node_count = hierarchy.num_nodes
    mass = coords.new_zeros(batch, node_count)
    mass.scatter_add_(1, fine_to_leaf, batch_weights)
    coordinate_sum = coords.new_zeros(batch, node_count, dimension)
    coordinate_index = fine_to_leaf[..., None].expand(-1, -1, dimension)
    coordinate_sum.scatter_add_(1, coordinate_index, batch_weights[..., None] * coords)
    for level_number in range(1, hierarchy.num_levels):
        child_start = int(hierarchy.level_offsets[level_number - 1].item())
        child_stop = int(hierarchy.level_offsets[level_number].item())
        child_nodes = torch.arange(child_start, child_stop, device=coords.device, dtype=torch.long)
        parent_nodes = hierarchy.parent_index[child_nodes].to(device=coords.device, dtype=torch.long)
        mass.index_add_(1, parent_nodes, mass.index_select(1, child_nodes))
        coordinate_sum.index_add_(1, parent_nodes, coordinate_sum.index_select(1, child_nodes))
    valid = mass > 0
    if hierarchy.node_valid is not None:
        valid = valid & (hierarchy.node_valid.to(device=coords.device) > 0.5)
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    centroids = coordinate_sum / safe_mass.to(dtype=coords.dtype)[..., None]
    centroids = centroids.masked_fill(~valid[..., None], 0.0)
    mass = mass.masked_fill(~valid, 0.0)
    return HierarchyGeometry(mass, centroids, valid)


def _axis_ranks(
    values: torch.Tensor,
    size: int,
    *,
    bound: torch.Tensor | None,
) -> torch.Tensor:
    """Build discrete ranks from detached geometry; values remain live elsewhere."""

    detached = values.detach()
    if bound is not None:
        edges = torch.linspace(bound[0], bound[1], size + 1, device=values.device, dtype=values.dtype)
        rank = torch.searchsorted(edges[1:-1].detach(), detached)
        return rank.clamp_(0, size - 1).to(torch.long)
    _, inverse = torch.unique(detached, sorted=True, return_inverse=True)
    if int(inverse.max().item()) + 1 > size:
        raise ValueError("coordinates contain more unique ranks than grid_shape allows.")
    return inverse.to(torch.long)


def _flat_axis_first(ranks: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    result = torch.zeros_like(ranks[..., 0], dtype=torch.long)
    stride = 1
    for axis, size in enumerate(shape):
        result = result + ranks[..., axis] * stride
        stride *= int(size)
    return result


def _unflatten_axis_first(indices: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    parts = []
    remainder = indices.to(torch.long)
    for size in shape:
        parts.append(remainder % int(size))
        remainder = torch.div(remainder, int(size), rounding_mode="floor")
    return torch.stack(parts, dim=-1)


def _node_topology(
    grid_shape: tuple[int, ...],
    block_shape: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    level_shapes = [grid_shape]
    while any(int(size) > 1 for size in level_shapes[-1]):
        current_shape = level_shapes[-1]
        next_shape = tuple(
            (int(size) + int(block) - 1) // int(block)
            for size, block in zip(current_shape, block_shape, strict=True)
        )
        if next_shape == current_shape:
            raise ValueError(
                "block_shape must exceed one on every grid axis whose size exceeds one."
            )
        level_shapes.append(next_shape)
    level_sizes = tuple(int(torch.tensor(shape).prod().item()) for shape in level_shapes)
    offsets_list = [0]
    for size in level_sizes:
        offsets_list.append(offsets_list[-1] + size)
    offsets = torch.tensor(offsets_list, device=device, dtype=torch.long)
    node_count = offsets_list[-1]
    level = torch.cat(
        [torch.full((size,), level_number, device=device, dtype=torch.long) for level_number, size in enumerate(level_sizes)]
    )
    parent = torch.full((node_count,), -1, device=device, dtype=torch.long)
    max_children = 1
    for block in block_shape:
        max_children *= int(block)
    children = torch.full((node_count, max_children), -1, device=device, dtype=torch.long)
    for level_number in range(1, len(level_shapes)):
        child_shape = level_shapes[level_number - 1]
        parent_shape = level_shapes[level_number]
        parent_offset = offsets_list[level_number]
        child_offset = offsets_list[level_number - 1]
        parent_count = level_sizes[level_number]
        parent_coords = _unflatten_axis_first(torch.arange(parent_count, device=device), parent_shape)
        slot = 0
        # A fixed child-slot ordering makes diagnostics and tests deterministic.
        import itertools

        for local_child in itertools.product(*(range(int(item)) for item in block_shape)):
            child_coords = parent_coords * torch.tensor(block_shape, device=device, dtype=torch.long) + torch.tensor(local_child, device=device, dtype=torch.long)
            in_bounds = torch.ones(parent_count, device=device, dtype=torch.bool)
            for axis, size in enumerate(child_shape):
                in_bounds &= child_coords[:, axis] < int(size)
            child_flat = _flat_axis_first(child_coords.clamp_min(0), child_shape) + child_offset
            parent_flat = torch.arange(parent_count, device=device, dtype=torch.long) + parent_offset
            children[parent_flat, slot] = torch.where(in_bounds, child_flat, torch.full_like(child_flat, -1))
            parent[child_flat[in_bounds]] = parent_flat[in_bounds]
            slot += 1
    leaf_nodes = torch.arange(level_sizes[0], device=device, dtype=torch.long) + offsets[0]
    return parent, children, level, offsets, leaf_nodes, tuple(level_sizes)


def _bounds_from_geometry(
    coordinates: torch.Tensor,
    fine_to_leaf: torch.Tensor,
    grid_shape: tuple[int, ...],
    block_shape: tuple[int, ...],
    level_offsets: torch.Tensor,
    bounds: torch.Tensor | None,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, environment, dimension = coordinates.shape
    node_count = int(level_offsets[-1].item())
    leaf_count = int(level_offsets[1].item())
    if bounds is not None:
        edges = [
            torch.linspace(bounds[axis, 0], bounds[axis, 1], int(grid_shape[axis]) + 1, device=coordinates.device, dtype=coordinates.dtype)
            for axis in range(dimension)
        ]
        leaf_coords = _unflatten_axis_first(torch.arange(leaf_count, device=coordinates.device), grid_shape)
        leaf_min = torch.stack([edges[axis][leaf_coords[:, axis]] for axis in range(dimension)], dim=-1)
        leaf_max = torch.stack([edges[axis][leaf_coords[:, axis] + 1] for axis in range(dimension)], dim=-1)
        bounds_min = coordinates.new_zeros(batch, node_count, dimension)
        bounds_max = coordinates.new_zeros(batch, node_count, dimension)
        bounds_min[:, :leaf_count] = leaf_min
        bounds_max[:, :leaf_count] = leaf_max
        # Parent bounds are filled by the common topology reduction below.
    else:
        inf = torch.tensor(float("inf"), device=coordinates.device, dtype=coordinates.dtype)
        bounds_min = coordinates.new_full((batch, node_count, dimension), inf)
        bounds_max = coordinates.new_full((batch, node_count, dimension), -inf)
        source_index = fine_to_leaf
        source_index_expanded = source_index[..., None].expand(-1, -1, dimension)
        active_coordinates_min = coordinates.masked_fill(~active[..., None], inf)
        active_coordinates_max = coordinates.masked_fill(~active[..., None], -inf)
        bounds_min = bounds_min.scatter_reduce(
            1, source_index_expanded, active_coordinates_min, reduce="amin", include_self=True
        )
        bounds_max = bounds_max.scatter_reduce(
            1, source_index_expanded, active_coordinates_max, reduce="amax", include_self=True
        )
    # CUDA does not provide a bool ``scatter_reduce_(amax)`` kernel on the
    # supported runtime.  Reduce integer occupancy codes and convert back to
    # bool after the reduction; this preserves the same zero-weight semantics.
    occupied_codes = torch.zeros(batch, node_count, device=coordinates.device, dtype=torch.long)
    occupied_codes.scatter_reduce_(
        1,
        fine_to_leaf,
        active.to(dtype=torch.long),
        reduce="amax",
        include_self=True,
    )
    occupied = occupied_codes > 0
    return bounds_min, bounds_max, occupied


def _complete_parent_bounds(
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    occupied: torch.Tensor,
    children: torch.Tensor,
    level_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for level_number in range(1, int(level_offsets.numel()) - 1):
        start = int(level_offsets[level_number].item())
        stop = int(level_offsets[level_number + 1].item())
        parent_children = children[start:stop]
        safe_children = parent_children.clamp_min(0)
        child_valid = parent_children >= 0
        selected_min = bounds_min[:, safe_children, :].masked_fill(~child_valid[None, :, :, None], float("inf"))
        selected_max = bounds_max[:, safe_children, :].masked_fill(~child_valid[None, :, :, None], float("-inf"))
        parent_min = selected_min.amin(dim=2)
        parent_max = selected_max.amax(dim=2)
        parent_occupied = occupied[:, safe_children].logical_and(child_valid[None]).any(dim=2)
        bounds_min = torch.cat([bounds_min[:, :start], parent_min, bounds_min[:, stop:]], dim=1)
        bounds_max = torch.cat([bounds_max[:, :start], parent_max, bounds_max[:, stop:]], dim=1)
        occupied = torch.cat([occupied[:, :start], parent_occupied, occupied[:, stop:]], dim=1)
    invalid = ~occupied
    bounds_min = bounds_min.masked_fill(invalid[..., None], 0.0)
    bounds_max = bounds_max.masked_fill(invalid[..., None], 0.0)
    return bounds_min, bounds_max, occupied


def build_rectangular_environment_hierarchy(
    coordinates: torch.Tensor,
    grid_shape: Sequence[int] | int | None = None,
    block_shape: Sequence[int] | int = 2,
    *,
    num_env_tokens_x: int | None = None,
    num_env_tokens_y: int | None = None,
    env_weights: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    bounds: Sequence[Sequence[float]] | torch.Tensor | None = None,
) -> EnvironmentHierarchy:
    """Build a rank-based rectangular hierarchy from physical coordinates.

    The builder accepts a single ``[E,D]`` coordinate set or a batched
    ``[B,E,D]`` set.  Ranks come from detached coordinates (discrete metadata),
    while supplied coordinates and masses remain live in the backend's
    per-preparation reductions.  If physical ``bounds`` are supplied, node
    boxes use the actual grid-cell unions rather than fictitious padding; if
    they are omitted, boxes are the min/max of the supplied source samples.
    """

    coords, batch, environment, dimension = _normalise_coordinates(coordinates)
    if grid_shape is None:
        if num_env_tokens_x is not None and num_env_tokens_y is not None:
            grid_shape = (int(num_env_tokens_x), int(num_env_tokens_y))
        else:
            raise ValueError("grid_shape or both num_env_tokens_x and num_env_tokens_y are required.")
    grid = _as_positive_shape(grid_shape, dimension, name="grid_shape")
    block = _as_positive_shape(block_shape, dimension, name="block_shape")
    if weights is not None and env_weights is not None:
        raise ValueError("Pass only one of env_weights and weights.")
    source_weights = env_weights if env_weights is not None else weights
    batch_weights = _normalise_weights(
        source_weights,
        batch,
        environment,
        device=coords.device,
        dtype=coords.dtype,
    )
    active_coordinates = (
        torch.ones(batch, environment, device=coords.device, dtype=torch.bool)
        if batch_weights is None
        else batch_weights > 0
    )
    if not torch.isfinite(coords.masked_select(active_coordinates[..., None])).all():
        raise ValueError("active coordinates must be finite.")
    domain_bounds = _normalise_bounds(bounds, dimension, device=coords.device, dtype=coords.dtype)

    ranks = torch.zeros(batch, environment, dimension, device=coords.device, dtype=torch.long)
    active_for_rank = None if batch_weights is None else batch_weights > 0
    for batch_index in range(batch):
        active = (
            torch.ones(environment, device=coords.device, dtype=torch.bool)
            if active_for_rank is None
            else active_for_rank[batch_index]
        )
        for axis, size in enumerate(grid):
            active_indices = torch.nonzero(active, as_tuple=False).flatten()
            if active_indices.numel() == 0:
                continue
            active_ranks = _axis_ranks(
                coords[batch_index, :, axis][active],
                size,
                bound=None if domain_bounds is None else domain_bounds[axis],
            )
            full_ranks = torch.zeros(environment, device=coords.device, dtype=torch.long)
            full_ranks[active_indices] = active_ranks
            ranks[batch_index, :, axis] = full_ranks
            # Zero-weight padding may be outside the domain.  Its integer
            # location is immaterial because reduction masks its mass.
            if not bool(active.all()):
                ranks[batch_index, ~active, axis] = 0

    parent, children, level, level_offsets, leaf_nodes, level_sizes = _node_topology(
        grid, block, device=coords.device
    )
    fine_to_leaf = _flat_axis_first(ranks, grid)
    bounds_min, bounds_max, occupied = _bounds_from_geometry(
        coords,
        fine_to_leaf,
        grid,
        block,
        level_offsets,
        domain_bounds,
        active_coordinates,
    )
    bounds_min, bounds_max, occupied = _complete_parent_bounds(
        bounds_min, bounds_max, occupied, children, level_offsets
    )
    return EnvironmentHierarchy(
        fine_to_leaf=fine_to_leaf if batch > 1 else fine_to_leaf[0],
        parent_index=parent,
        children=children,
        level=level,
        level_offsets=level_offsets,
        bounds_min=bounds_min if batch > 1 else bounds_min[0],
        bounds_max=bounds_max if batch > 1 else bounds_max[0],
        leaf_nodes=leaf_nodes,
        node_valid=occupied if batch > 1 else occupied[0],
        grid_shape=grid,
        block_shape=block,
    )


def reduce_preupdate_statistics(
    hierarchy: EnvironmentHierarchy,
    environment: torch.Tensor,
    response: torch.Tensor,
    weights: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    geometry: HierarchyGeometry | None = None,
) -> HierarchyStatistics:
    """Reduce fine pre-update ``e`` and ``a`` statistics bottom-up.

    Every node sum is formed from fine values through child sums.  No
    already-updated child state is averaged, so the caller can apply one
    shared nonlinear ``env_update`` to the resulting node means.
    """

    if environment.ndim != 3 or response.ndim != 3 or coordinates.ndim != 3:
        raise ValueError("environment, response, and coordinates must be batched rank-3 tensors.")
    batch, fine_count, env_hidden = environment.shape
    if tuple(response.shape[:2]) != (batch, fine_count) or tuple(coordinates.shape[:2]) != (batch, fine_count):
        raise ValueError("Fine environment, response, and coordinate axes must align.")
    if weights.ndim == 1:
        weights = weights.unsqueeze(0).expand(batch, -1)
    elif weights.ndim == 2 and tuple(weights.shape) == (1, fine_count):
        weights = weights.expand(batch, -1)
    if weights.ndim != 2 or tuple(weights.shape) != (batch, fine_count):
        raise ValueError("weights must have shape [E] or [B,E].")
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("environment weights must be nonnegative.")
    hierarchy = hierarchy.to(environment.device).expand_batch(batch)
    fine_to_leaf = hierarchy.fine_to_leaf.to(device=environment.device, dtype=torch.long)
    node_count = hierarchy.num_nodes
    if geometry is None:
        geometry = prepare_hierarchy_geometry(hierarchy, weights, coordinates)
    else:
        geometry = geometry.to(environment.device).expand_batch(batch)
        if int(geometry.mass.shape[-1]) != node_count:
            raise ValueError("HierarchyGeometry does not align with hierarchy nodes.")
        if int(geometry.coordinates.shape[-1]) != int(coordinates.shape[-1]):
            raise ValueError("HierarchyGeometry does not align with coordinate dimension.")
    mass = geometry.mass.to(dtype=environment.dtype)
    valid = geometry.valid.to(device=environment.device) & (mass > 0)
    environment_sum = environment.new_zeros(batch, node_count, env_hidden)
    response_sum = response.new_zeros(batch, node_count, response.shape[-1])
    fine_weights = weights.to(dtype=environment.dtype)
    env_index = fine_to_leaf[..., None].expand(-1, -1, env_hidden)
    response_index = fine_to_leaf[..., None].expand(-1, -1, int(response.shape[-1]))
    environment_sum.scatter_add_(1, env_index, fine_weights[..., None] * environment)
    response_sum.scatter_add_(1, response_index, weights.to(dtype=response.dtype)[..., None] * response)

    for level_number in range(1, hierarchy.num_levels):
        child_start = int(hierarchy.level_offsets[level_number - 1].item())
        child_stop = int(hierarchy.level_offsets[level_number].item())
        child_nodes = torch.arange(child_start, child_stop, device=environment.device, dtype=torch.long)
        parent_nodes = hierarchy.parent_index[child_nodes].to(device=environment.device, dtype=torch.long)
        environment_sum.index_add_(1, parent_nodes, environment_sum.index_select(1, child_nodes))
        response_sum.index_add_(1, parent_nodes, response_sum.index_select(1, child_nodes))
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    environment_mean = environment_sum / safe_mass[..., None]
    response_mean = response_sum / safe_mass[..., None]
    environment_mean = environment_mean.masked_fill(~valid[..., None], 0.0)
    response_mean = response_mean.masked_fill(~valid[..., None], 0.0)
    coordinate_mean = geometry.coordinates.to(device=coordinates.device)
    coordinate_mean = coordinate_mean.masked_fill(~valid[..., None], 0.0)
    mass = mass.masked_fill(~valid, 0.0)
    return HierarchyStatistics(mass, coordinate_mean, environment_mean, response_mean, valid)


__all__ = [
    "EnvironmentHierarchy",
    "HierarchyGeometry",
    "HierarchyStatistics",
    "build_rectangular_environment_hierarchy",
    "prepare_hierarchy_geometry",
    "reduce_preupdate_statistics",
]
