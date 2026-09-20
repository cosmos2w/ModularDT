"""Case-neutral sampling helpers for a regular two-dimensional environment grid.

The ThermalChannel adapter owns the physical domain, centre coordinates,
quadrature masses, and the mapping from its token sequence to the lattice.  A
``RegularGridLayout`` carries that metadata into the generic HONF reader.  It
does not infer a grid from token order and it deliberately rejects layouts
that are not a complete rectangular lattice.

Coordinates are always in physical ``(x, y)`` order.  ``token_to_grid`` uses
``(row, column) == (y-index, x-index)`` order, matching PyTorch's ``(H, W)``
layout.  Values passed to the sampler are packed as ``[B, C, Ny, Nx]`` and
sample points as ``[B, ..., 2]`` (or ``[..., 2]`` when shared by the batch).
The centre hull, rather than the physical rectangle, is used for the
``align_corners=True`` normalisation.  A singleton axis is constant and maps
to normalised coordinate zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

Tensor = torch.Tensor

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _as_axis(value: Tensor | Sequence[float], *, name: str) -> Tensor:
    axis = torch.as_tensor(value)
    if axis.ndim != 1 or axis.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor.")
    if not torch.is_floating_point(axis):
        axis = axis.to(dtype=torch.get_default_dtype())
    if not bool(torch.isfinite(axis).all()):
        raise ValueError(f"{name} must contain only finite coordinates.")
    if axis.numel() > 1 and not bool((torch.diff(axis) > 0).all()):
        raise ValueError(f"{name} must be strictly increasing with no duplicate centres.")
    if axis.numel() > 2:
        spacing = torch.diff(axis)
        # ``grid_sample(align_corners=True)`` represents a uniformly spaced
        # lattice.  Reject an irregular coordinate list instead of silently
        # applying the endpoint-affine map to the wrong cells.
        tolerance = 64.0 * torch.finfo(axis.dtype).eps
        if not torch.allclose(spacing, spacing[:1], rtol=tolerance, atol=tolerance):
            raise ValueError(f"{name} must be uniformly spaced for regular-grid interpolation.")
    return axis


def _as_physical_bounds(
    bounds: Tensor | Sequence[Sequence[float]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return bounds as ``[[x_min, y_min], [x_max, y_max]]``.

    The generic representation is lower/upper rows.  A pair of lower and
    upper vectors is accepted as well, which keeps adapter construction clear:
    ``(lower, upper)``.  Axis-wise ``((x_min, x_max), (y_min, y_max))`` input
    is available through :meth:`RegularGridLayout.from_axis_bounds`.
    """

    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raw = torch.as_tensor(bounds, dtype=dtype, device=device)
        if raw.shape != (2, 2):
            raise ValueError(
                "physical_bounds must have shape [2,2] as lower/upper rows "
                "or be a pair of lower and upper vectors."
            )
    else:
        raw = torch.as_tensor(bounds, dtype=dtype, device=device)
        if raw.shape != (2, 2):
            raise ValueError(
                "physical_bounds must contain two length-2 lower/upper vectors."
            )
    return raw


@dataclass(frozen=True)
class InterpolationIncidence:
    """Four-corner incidence for one set of physical sample points.

    ``corner_indices[..., r, :]`` stores ``(row, column)`` lattice indices in
    the order ``(top-left, top-right, bottom-left, bottom-right)``.  The
    corresponding non-negative ``corner_weights`` sum to one, including when
    an axis is singleton (then repeated corners carry the appropriate total
    mass).  ``cell_indices`` stores the lower-left cell as ``(row, column)``;
    on a singleton or upper boundary it is the final valid cell with a weight
    of one on the boundary corner.
    """

    corner_indices: Tensor
    corner_weights: Tensor
    normalized_points: Tensor
    cell_indices: Tensor
    grid_shape: tuple[int, int] | None = None

    @property
    def corner_linear_indices(self) -> Tensor:
        """Return flattened row-major corner indices when available.

        The final coordinate is intentionally not cached in the layout so the
        incidence remains a lightweight diagnostic object.  This property
        infers the width from the largest column in the incidence; callers
        that need exact linear indices should use
        :meth:`RegularGridLayout.linearize_indices` instead.
        """

        # This property is mainly convenient for diagnostics.  The exact
        # layout-aware variant is supplied by ``RegularGridLayout`` below.
        columns = self.corner_indices[..., 1]
        if self.grid_shape is None:
            width = int(columns.max().detach().item()) + 1 if columns.numel() else 1
        else:
            width = int(self.grid_shape[1])
        return self.corner_indices[..., 0] * width + columns


@dataclass(frozen=True)
class GridSampleResult:
    """Sampled values together with the executed interpolation incidence."""

    values: Tensor
    incidence: InterpolationIncidence


@dataclass(frozen=True)
class RegularGridLayout:
    """Validated metadata for a complete regular rectangular 2-D lattice.

    Parameters
    ----------
    physical_bounds:
        Lower/upper rows ``[[x_min, y_min], [x_max, y_max]]``.  The values
        describe the physical rectangle and may be wider than the centre
        hull, as is the case for a cell-centred grid.
    centre_axes:
        Two strictly increasing one-dimensional tensors ``(x_centres,
        y_centres)``.  Coordinates use physical ``(x, y)`` order.
    token_to_grid:
        Integer ``[E, 2]`` mapping from adapter token index to ``(row, col)``
        lattice index.  Every lattice cell must occur exactly once.
    weights:
        Strictly positive finite token masses ``[E]`` in adapter token order.

    The metadata is intentionally explicit.  In particular, no reshape of an
    arbitrary token sequence is attempted, so a consistently permuted token
    list remains equivalent after ``tokens_to_grid``.
    """

    physical_bounds: Tensor | Sequence[Sequence[float]]
    centre_axes: tuple[Tensor, Tensor] | Sequence[Tensor]
    token_to_grid: Tensor
    weights: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.centre_axes, (tuple, list)) or len(self.centre_axes) != 2:
            raise ValueError("centre_axes must be a pair (x_centres, y_centres).")
        x_axis = _as_axis(self.centre_axes[0], name="centre_axes[0] (x centres)")
        y_axis = _as_axis(self.centre_axes[1], name="centre_axes[1] (y centres)")
        # One metadata dtype/device makes all coordinate transforms explicit;
        # adapter-owned axes are not autograd state, so this cast is safe.
        y_axis = y_axis.to(device=x_axis.device, dtype=x_axis.dtype)
        bounds = _as_physical_bounds(
            self.physical_bounds,
            dtype=x_axis.dtype,
            device=x_axis.device,
        )
        if not bool(torch.isfinite(bounds).all()):
            raise ValueError("physical_bounds must contain only finite values.")
        if not bool((bounds[1] > bounds[0]).all()):
            raise ValueError("physical_bounds must have strictly positive x/y extents.")
        lower, upper = bounds[0], bounds[1]
        for axis, axis_name, coordinate in (
            (x_axis, "x", 0),
            (y_axis, "y", 1),
        ):
            if bool((axis < lower[coordinate]).any()) or bool((axis > upper[coordinate]).any()):
                raise ValueError(
                    f"{axis_name} centre coordinates must lie inside the supplied physical bounds."
                )

        mapping = torch.as_tensor(self.token_to_grid)
        if mapping.ndim != 2 or mapping.shape[-1] != 2:
            raise ValueError("token_to_grid must have shape [E, 2] as (row, column) indices.")
        if mapping.dtype not in _INTEGER_DTYPES:
            raise ValueError("token_to_grid must use an integer dtype.")
        mapping = mapping.to(dtype=torch.long)

        weights = torch.as_tensor(self.weights, dtype=x_axis.dtype, device=x_axis.device)
        if weights.ndim != 1 or int(weights.shape[0]) != int(mapping.shape[0]):
            raise ValueError("weights must have shape [E] matching token_to_grid.")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("weights must contain finite strictly positive masses.")

        ny, nx = int(y_axis.numel()), int(x_axis.numel())
        if int(mapping.shape[0]) != nx * ny:
            raise ValueError(
                "token_to_grid must contain exactly Ny*Nx tokens for a complete rectangular grid."
            )
        rows, cols = mapping.unbind(dim=-1)
        if bool((rows < 0).any()) or bool((rows >= ny).any()) or bool((cols < 0).any()) or bool((cols >= nx).any()):
            raise ValueError("token_to_grid contains an index outside the declared centre axes.")
        linear = rows * nx + cols
        if not torch.equal(torch.sort(linear).values, torch.arange(nx * ny, device=linear.device)):
            raise ValueError(
                "token_to_grid must be a one-to-one mapping covering every rectangular grid cell."
            )

        # Keep axes and bounds in the same metadata dtype/device.  Values are
        # cast to the sampled bank's dtype at the call site, preserving the
        # layout as adapter-owned data rather than autograd state.
        object.__setattr__(self, "centre_axes", (x_axis, y_axis))
        object.__setattr__(self, "physical_bounds", bounds)
        object.__setattr__(self, "token_to_grid", mapping)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def from_axis_bounds(
        cls,
        *,
        axis_bounds: Sequence[Sequence[float]] | Tensor,
        centre_axes: tuple[Tensor, Tensor] | Sequence[Tensor],
        token_to_grid: Tensor,
        weights: Tensor,
    ) -> RegularGridLayout:
        """Construct from ``((x_min, x_max), (y_min, y_max))`` bounds."""

        raw = torch.as_tensor(axis_bounds)
        if raw.shape != (2, 2):
            raise ValueError("axis_bounds must have shape [2,2] as x/y lower/upper pairs.")
        lower_upper = torch.stack((raw[:, 0], raw[:, 1]), dim=0)
        return cls(
            physical_bounds=lower_upper,
            centre_axes=centre_axes,
            token_to_grid=token_to_grid,
            weights=weights,
        )

    @property
    def center_axes(self) -> tuple[Tensor, Tensor]:
        """US-spelling alias for :attr:`centre_axes`."""

        return self.centre_axes

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return ``(Ny, Nx)`` in tensor/grid order."""

        return int(self.centre_axes[1].numel()), int(self.centre_axes[0].numel())

    @property
    def num_tokens(self) -> int:
        return int(self.token_to_grid.shape[0])

    @property
    def centre_hull(self) -> Tensor:
        """Return lower/upper rows for the centre-coordinate hull."""

        x_axis, y_axis = self.centre_axes
        return torch.stack(
            (
                torch.stack((x_axis[0], y_axis[0])),
                torch.stack((x_axis[-1], y_axis[-1])),
            ),
            dim=0,
        )

    @property
    def center_hull(self) -> Tensor:
        """US-spelling alias for :attr:`centre_hull`."""

        return self.centre_hull

    @property
    def token_order(self) -> Tensor:
        """Token indices sorted into row-major lattice order."""

        _, nx = self.grid_shape
        linear = self.token_to_grid[:, 0] * nx + self.token_to_grid[:, 1]
        return torch.argsort(linear)

    @property
    def linear_token_indices(self) -> Tensor:
        """Flattened row-major lattice index for every adapter token."""

        _, nx = self.grid_shape
        return self.token_to_grid[:, 0] * nx + self.token_to_grid[:, 1]

    def linearize_indices(self, indices: Tensor) -> Tensor:
        """Convert ``(..., 2)`` ``(row, column)`` indices to row-major IDs."""

        if indices.shape[-1] != 2:
            raise ValueError("indices must end with a (row, column) coordinate.")
        rows, cols = indices[..., 0], indices[..., 1]
        ny, nx = self.grid_shape
        if bool((rows < 0).any()) or bool((rows >= ny).any()) or bool((cols < 0).any()) or bool((cols >= nx).any()):
            raise ValueError("indices contain a coordinate outside the grid.")
        return rows * nx + cols

    @property
    def weights_grid(self) -> Tensor:
        """Return adapter masses as a canonical ``[Ny, Nx]`` grid."""

        return self.weights.index_select(
            0,
            self.token_order.to(device=self.weights.device),
        ).reshape(self.grid_shape)

    def tokens_to_grid(self, tokens: Tensor) -> Tensor:
        """Pack token-order values ``[B,E,C]`` into ``[B,C,Ny,Nx]``.

        A batchless ``[E,C]`` input is accepted and returns ``[C,Ny,Nx]``.
        The operation uses the supplied token mapping, so physically identical
        data is invariant to a consistent token permutation.
        """

        if tokens.ndim == 2:
            tokens_b = tokens.unsqueeze(0)
            batchless = True
        elif tokens.ndim == 3:
            tokens_b = tokens
            batchless = False
        else:
            raise ValueError("tokens must have shape [E,C] or [B,E,C].")
        if int(tokens_b.shape[1]) != self.num_tokens:
            raise ValueError("tokens environment axis must match token_to_grid.")
        ordered = tokens_b.index_select(1, self.token_order.to(device=tokens_b.device))
        batch, _, channels = ordered.shape
        ny, nx = self.grid_shape
        grid = ordered.reshape(batch, ny, nx, channels).permute(0, 3, 1, 2).contiguous()
        return grid[0] if batchless else grid

    def grid_to_tokens(self, grid: Tensor) -> Tensor:
        """Unpack ``[B,C,Ny,Nx]`` values into adapter token order ``[B,E,C]``."""

        if grid.ndim == 3:
            grid_b = grid.unsqueeze(0)
            batchless = True
        elif grid.ndim == 4:
            grid_b = grid
            batchless = False
        else:
            raise ValueError("grid must have shape [C,Ny,Nx] or [B,C,Ny,Nx].")
        ny, nx = self.grid_shape
        if tuple(grid_b.shape[-2:]) != (ny, nx):
            raise ValueError("grid spatial shape must match centre_axes.")
        flat = grid_b.permute(0, 2, 3, 1).reshape(grid_b.shape[0], self.num_tokens, grid_b.shape[1])
        token_values = flat.index_select(1, self.linear_token_indices.to(device=grid_b.device))
        return token_values[0] if batchless else token_values

    def physical_to_centre_points(self, points: Tensor) -> Tensor:
        """Map physical-rectangle coordinates to the centre-coordinate hull.

        This is the inward map used by the proposed site generator.  It is
        separate from ``normalise_points`` because the latter accepts points
        already expressed in physical centre coordinates.
        """

        if points.shape[-1] != 2:
            raise ValueError("points must end with an (x, y) coordinate.")
        points = points.to(device=self.centre_axes[0].device, dtype=self.centre_axes[0].dtype)
        lower, upper = self.physical_bounds[0], self.physical_bounds[1]
        if not bool(torch.isfinite(points).all()):
            raise ValueError("physical points must contain only finite coordinates.")
        scale = (upper - lower).abs().clamp_min(1.0)
        tolerance = 32.0 * torch.finfo(points.dtype).eps * scale
        if bool((points < lower - tolerance).any()) or bool((points > upper + tolerance).any()):
            raise ValueError("physical points must lie inside the supplied physical bounds.")
        x_axis, y_axis = self.centre_axes
        mapped = []
        for coordinate, axis in ((points[..., 0], x_axis), (points[..., 1], y_axis)):
            if axis.numel() == 1:
                mapped.append(torch.full_like(coordinate, axis[0]))
            else:
                mapped.append(
                    axis[0]
                    + (axis[-1] - axis[0]) * (coordinate - lower[len(mapped)]) / (upper[len(mapped)] - lower[len(mapped)])
                )
        return torch.stack(mapped, dim=-1)

    def normalize_points(self, points: Tensor) -> Tensor:
        """Map centre-coordinate points to ``[-1, 1]`` for ``grid_sample``."""

        if points.shape[-1] != 2:
            raise ValueError("points must end with an (x, y) coordinate.")
        x_axis, y_axis = self.centre_axes
        points = points.to(device=x_axis.device, dtype=x_axis.dtype)
        hull = self.centre_hull
        if not bool(torch.isfinite(points).all()):
            raise ValueError("sample points must contain only finite coordinates.")
        # Reject out-of-hull sites rather than allowing padding to fabricate a
        # physical value.  The tolerance only absorbs endpoint round-off.
        scale = (hull[1] - hull[0]).abs().clamp_min(1.0)
        tolerance = 32.0 * torch.finfo(points.dtype).eps * scale
        if bool((points < hull[0] - tolerance).any()) or bool((points > hull[1] + tolerance).any()):
            raise ValueError("sample points must lie inside the environmental centre hull.")
        normalized = []
        for coordinate, axis in ((points[..., 0], x_axis), (points[..., 1], y_axis)):
            if axis.numel() == 1:
                normalized.append(torch.zeros_like(coordinate))
            else:
                normalized.append(2.0 * (coordinate - axis[0]) / (axis[-1] - axis[0]) - 1.0)
        return torch.stack(normalized, dim=-1)

    # British spelling is used in the design document; keep both forms public.
    normalise_points = normalize_points

    def interpolation_incidence(self, points: Tensor) -> InterpolationIncidence:
        """Return four-corner indices and bilinear weights for ``points``."""

        if points.shape[-1] != 2:
            raise ValueError("points must end with an (x, y) coordinate.")
        points = points.to(device=self.centre_axes[0].device, dtype=self.centre_axes[0].dtype)
        normalized = self.normalize_points(points)
        x_axis, y_axis = self.centre_axes
        ny, nx = self.grid_shape

        if nx == 1:
            column0 = torch.zeros_like(points[..., 0], dtype=torch.long)
            column1 = column0
            tx = torch.zeros_like(points[..., 0])
        else:
            x_position = (points[..., 0] - x_axis[0]) / (x_axis[-1] - x_axis[0]) * float(nx - 1)
            column0 = torch.floor(x_position).to(torch.long).clamp(0, nx - 2)
            column1 = column0 + 1
            tx = x_position - column0.to(x_position.dtype)

        if ny == 1:
            row0 = torch.zeros_like(points[..., 1], dtype=torch.long)
            row1 = row0
            ty = torch.zeros_like(points[..., 1])
        else:
            y_position = (points[..., 1] - y_axis[0]) / (y_axis[-1] - y_axis[0]) * float(ny - 1)
            row0 = torch.floor(y_position).to(torch.long).clamp(0, ny - 2)
            row1 = row0 + 1
            ty = y_position - row0.to(y_position.dtype)

        corner_indices = torch.stack(
            (
                torch.stack((row0, column0), dim=-1),
                torch.stack((row0, column1), dim=-1),
                torch.stack((row1, column0), dim=-1),
                torch.stack((row1, column1), dim=-1),
            ),
            dim=-2,
        )
        corner_weights = torch.stack(
            ((1.0 - ty) * (1.0 - tx), (1.0 - ty) * tx, ty * (1.0 - tx), ty * tx),
            dim=-1,
        )
        cell_indices = torch.stack((row0, column0), dim=-1)
        return InterpolationIncidence(
            corner_indices=corner_indices,
            corner_weights=corner_weights,
            normalized_points=normalized,
            cell_indices=cell_indices,
            grid_shape=(ny, nx),
        )

    # Concise method names are useful in the hot reader and remain explicit.
    incidence = interpolation_incidence


def _prepare_values(values: Tensor, layout: RegularGridLayout) -> tuple[Tensor, bool]:
    if values.ndim == 3:
        values_b = values.unsqueeze(0)
        batchless = True
    elif values.ndim == 4:
        values_b = values
        batchless = False
    else:
        raise ValueError("values must have shape [C,Ny,Nx] or [B,C,Ny,Nx].")
    if not torch.is_floating_point(values_b):
        raise ValueError("values must use a floating-point dtype.")
    if tuple(values_b.shape[-2:]) != layout.grid_shape:
        raise ValueError("values spatial shape must match the regular-grid layout.")
    return values_b, batchless


def _prepare_points(points: Tensor, *, batch_size: int, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, bool]:
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [...,2] with physical (x,y) coordinates.")
    points = points.to(device=device, dtype=dtype)
    # A leading batch axis is recognised only when it matches the values
    # batch.  Otherwise one shared set of points is broadcast to all batches.
    has_batch = points.ndim >= 3 and int(points.shape[0]) == batch_size
    if has_batch:
        return points, True
    return points.unsqueeze(0).expand(batch_size, *points.shape), False


def _reference_values(values_b: Tensor, incidence: InterpolationIncidence, layout: RegularGridLayout) -> Tensor:
    """Evaluate a corner incidence against a packed ``[B,C,Ny,Nx]`` bank."""

    batch, channels = int(values_b.shape[0]), int(values_b.shape[1])
    query_shape = tuple(incidence.corner_weights.shape[1:-1])
    count = 1
    for size in query_shape:
        count *= int(size)
    corner_indices = incidence.corner_indices.reshape(batch, count, 4, 2)
    corner_weights = incidence.corner_weights.reshape(batch, count, 4).to(dtype=values_b.dtype)
    linear = layout.linearize_indices(corner_indices).reshape(batch, count, 4)
    flat = values_b.reshape(batch, channels, -1).transpose(1, 2)
    gathered = flat.gather(
        1,
        linear.unsqueeze(-1).expand(batch, count, 4, channels).reshape(batch, count * 4, channels),
    ).reshape(batch, count, 4, channels)
    sampled = (gathered * corner_weights.unsqueeze(-1)).sum(dim=-2)
    return sampled.reshape(batch, *query_shape, channels)


def sample_regular_grid_reference(
    values: Tensor,
    points: Tensor,
    layout: RegularGridLayout,
    *,
    return_incidence: bool = False,
) -> Tensor | GridSampleResult:
    """Sample with an explicit four-corner bilinear interpolation reference.

    The reference is intentionally independent of ``torch.nn.functional`` and
    is used for arithmetic/gradient checks.  It has the same centre-hull and
    singleton-axis convention as :func:`sample_regular_grid`.
    """

    values_b, batchless_values = _prepare_values(values, layout)
    points_b, _ = _prepare_points(
        points,
        batch_size=int(values_b.shape[0]),
        dtype=values_b.dtype,
        device=values_b.device,
    )
    incidence = layout.interpolation_incidence(points_b)
    sampled = _reference_values(values_b, incidence, layout)
    if batchless_values:
        sampled_out = sampled[0]
        incidence_out = InterpolationIncidence(
            corner_indices=incidence.corner_indices[0],
            corner_weights=incidence.corner_weights[0],
            normalized_points=incidence.normalized_points[0],
            cell_indices=incidence.cell_indices[0],
            grid_shape=incidence.grid_shape,
        )
    else:
        sampled_out = sampled
        incidence_out = incidence
    if return_incidence:
        return GridSampleResult(values=sampled_out, incidence=incidence_out)
    return sampled_out


def sample_regular_grid(
    values: Tensor,
    points: Tensor,
    layout: RegularGridLayout,
    *,
    return_incidence: bool = False,
) -> Tensor | GridSampleResult:
    """Sample a regular bank through ``grid_sample(align_corners=True)``.

    ``values`` is ``[B,C,Ny,Nx]`` (or batchless ``[C,Ny,Nx]``), while
    ``points`` is ``[B,...,2]`` or a shared ``[...,2]`` tensor.  The output is
    ``[B,...,C]`` (or ``[...,C]`` for a batchless bank).  Every point is
    validated against the centre hull before sampling, so the sampler never
    silently falls back to fabricated padding values.
    """

    values_b, batchless_values = _prepare_values(values, layout)
    points_b, _ = _prepare_points(
        points,
        batch_size=int(values_b.shape[0]),
        dtype=values_b.dtype,
        device=values_b.device,
    )
    normalized = layout.normalize_points(points_b).to(dtype=values_b.dtype, device=values_b.device)
    batch, channels = int(values_b.shape[0]), int(values_b.shape[1])
    query_shape = tuple(points_b.shape[1:-1])
    count = 1
    for size in query_shape:
        count *= int(size)
    # grid_sample consumes [B,H_out,W_out,2].  Flatten arbitrary query axes
    # into W_out while retaining a trivial H_out.
    grid = normalized.reshape(batch, 1, count, 2)
    sampled = F.grid_sample(
        values_b,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled[:, :, 0, :].transpose(1, 2).reshape(batch, *query_shape, channels)
    if batchless_values:
        sampled_out = sampled[0]
    else:
        sampled_out = sampled
    if return_incidence:
        incidence = layout.interpolation_incidence(points_b)
        if batchless_values:
            incidence = InterpolationIncidence(
                corner_indices=incidence.corner_indices[0],
                corner_weights=incidence.corner_weights[0],
                normalized_points=incidence.normalized_points[0],
                cell_indices=incidence.cell_indices[0],
                grid_shape=incidence.grid_shape,
            )
        return GridSampleResult(values=sampled_out, incidence=incidence)
    return sampled_out


# Names that make the reference relationship obvious at call sites.
interpolate_regular_grid = sample_regular_grid
interpolate_regular_grid_reference = sample_regular_grid_reference


__all__ = [
    "GridSampleResult",
    "InterpolationIncidence",
    "RegularGridLayout",
    "interpolate_regular_grid",
    "interpolate_regular_grid_reference",
    "sample_regular_grid",
    "sample_regular_grid_reference",
]
