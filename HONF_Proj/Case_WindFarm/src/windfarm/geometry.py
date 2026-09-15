"""Geometry and native sampling helpers for the WindFarm forward view.

The volume export contains cell centres rather than a mesh.  This module keeps
that distinction explicit: support boxes and node-associated quadrature are
derived from the exact native axes, while all supervised coordinates are
gathered from the same run as their target values.  No field is resampled to a
shared cube and the environmental representation contains geometry only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

D_M = 80.0
HUB_HEIGHT_M = 70.0
U_REF_MPS = 9.0
HUB_HEIGHT_D = HUB_HEIGHT_M / D_M
POSITIONAL_SCALE_D = np.asarray((50.0, 38.0, 6.25), dtype=np.float64)
ENV_TOKEN_SHAPE = (16, 8, 4)
ENV_TOKEN_COUNT = int(np.prod(ENV_TOKEN_SHAPE))
TARGET_COMPONENTS = ("Ux", "Uy", "Uz")
WIND_DIRECTIONS_DEG = (270.0, 285.0, 300.0)


class GeometryError(ValueError):
    """Raised when native axes or geometry metadata are not usable."""


@dataclass(frozen=True)
class SupportGeometry:
    """Cell-centre support and fixed-unit geometry for one native run."""

    lower_m: np.ndarray
    upper_m: np.ndarray
    lower_D: np.ndarray
    upper_D: np.ndarray
    extent_D: np.ndarray
    volume_m3: float
    volume_D3: float

    @property
    def lower(self) -> np.ndarray:
        """Compatibility alias for the lower support bound in metres."""

        return self.lower_m

    @property
    def upper(self) -> np.ndarray:
        """Compatibility alias for the upper support bound in metres."""

        return self.upper_m


@dataclass(frozen=True)
class EnvironmentRepresentation:
    """Geometry-only environmental tokens and their adapter-owned measure."""

    coords_D: np.ndarray
    features: np.ndarray
    weights_D3: np.ndarray
    token_shape: tuple[int, int, int]

    @property
    def weights(self) -> np.ndarray:
        """Compatibility alias for weights in rotor-diameter cubed."""

        return self.weights_D3


@dataclass(frozen=True)
class NativeQuerySample:
    """A native target gather and its node quadrature information."""

    flat_indices: np.ndarray
    coords_D: np.ndarray
    velocity_mps: np.ndarray
    node_weights_m3: np.ndarray
    distribution_volume_m3: float
    mode: Literal["volume", "hub_band"]


def _axis_array(axis: Any, *, name: str) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise GeometryError(f"{name} must be a non-empty one-dimensional axis")
    if not np.all(np.isfinite(values)):
        raise GeometryError(f"{name} contains non-finite cell centres")
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise GeometryError(f"{name} must be strictly increasing")
    return values


def support_geometry(
    x_m: Any,
    y_m: Any,
    z_m: Any,
    *,
    diameter_m: float = D_M,
) -> SupportGeometry:
    """Build the explicit support box from exact cell-centre axes.

    The resulting bounds describe the represented centre support.  They are
    not asserted to be the unknown outer CFD face locations.
    """

    diameter = float(diameter_m)
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise GeometryError("diameter_m must be finite and positive")
    axes = tuple(_axis_array(axis, name=name) for name, axis in (("x", x_m), ("y", y_m), ("z", z_m)))
    lower_m = np.asarray([axis[0] for axis in axes], dtype=np.float64)
    upper_m = np.asarray([axis[-1] for axis in axes], dtype=np.float64)
    extent_m = upper_m - lower_m
    if np.any(extent_m <= 0.0):
        raise GeometryError("support axes must span a positive centre box")
    lower_D = lower_m / diameter
    upper_D = upper_m / diameter
    extent_D = extent_m / diameter
    volume_m3 = float(np.prod(extent_m, dtype=np.float64))
    return SupportGeometry(
        lower_m=lower_m,
        upper_m=upper_m,
        lower_D=lower_D,
        upper_D=upper_D,
        extent_D=extent_D,
        volume_m3=volume_m3,
        volume_D3=float(volume_m3 / diameter**3),
    )


def cell_centre_weights(axis: Any) -> np.ndarray:
    """Return centre-box support weights for one strictly increasing axis.

    The first and last support edges are the first and last centres.  Interior
    edges are midpoints, so the weights integrate over the known centre box
    and sum to ``axis[-1] - axis[0]``.  These are not guessed finite-volume
    cell volumes.
    """

    values = _axis_array(axis, name="axis")
    if values.size == 1:
        return np.ones(1, dtype=np.float64)
    weights = np.empty(values.size, dtype=np.float64)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    if values.size > 2:
        weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    return weights


def support_weights(x_m: Any, y_m: Any, z_m: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return factorised centre-box weights in metres."""

    return cell_centre_weights(x_m), cell_centre_weights(y_m), cell_centre_weights(z_m)


def support_features(
    coords_D: Any,
    support: SupportGeometry,
    *,
    positional_scale_D: Any = POSITIONAL_SCALE_D,
) -> np.ndarray:
    """Compute seven known support features for arbitrary coordinates.

    The first three entries are lower-face distances in fixed D-scaled units,
    the next three are upper-face distances, and the final entry is absolute
    normalized altitude.  Coordinates are expected in rotor-diameter units.
    """

    coords = np.asarray(coords_D, dtype=np.float64)
    if coords.shape[-1:] != (3,):
        raise GeometryError(f"coords_D must end in 3 coordinates, got {coords.shape}")
    scale = np.asarray(positional_scale_D, dtype=np.float64).reshape(-1)
    if scale.shape != (3,) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise GeometryError("positional_scale_D must contain three finite positive values")
    relative_lower = (coords - support.lower_D) / scale
    relative_upper = (support.upper_D - coords) / scale
    absolute_height = coords[..., 2:3] / scale[2]
    return np.concatenate((relative_lower, relative_upper, absolute_height), axis=-1).astype(np.float32)


def environment_representation(
    support: SupportGeometry,
    *,
    token_shape: tuple[int, int, int] = ENV_TOKEN_SHAPE,
    positional_scale_D: Any = POSITIONAL_SCALE_D,
) -> EnvironmentRepresentation:
    """Construct deterministic cell-centred geometry tokens for one support.

    Token coordinates are in D units and weights sum to the represented
    centre-box volume in D³.  No solved field or target statistic is read.
    """

    shape = tuple(int(value) for value in token_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise GeometryError("token_shape must contain three positive dimensions")
    fractions = tuple(
        (np.arange(size, dtype=np.float64) + 0.5) / float(size) for size in shape
    )
    # meshgrid returns z/y/x in this order only if requested explicitly; the
    # final stack below is always [x, y, z].
    x_fraction, y_fraction, z_fraction = np.meshgrid(*fractions, indexing="ij")
    coords_D = np.stack(
        (
            support.lower_D[0] + support.extent_D[0] * x_fraction,
            support.lower_D[1] + support.extent_D[1] * y_fraction,
            support.lower_D[2] + support.extent_D[2] * z_fraction,
        ),
        axis=-1,
    ).reshape(-1, 3)
    weights_D3 = np.full(coords_D.shape[0], support.volume_D3 / coords_D.shape[0], dtype=np.float32)
    features = support_features(coords_D, support, positional_scale_D=positional_scale_D)
    return EnvironmentRepresentation(
        coords_D=coords_D.astype(np.float32),
        features=features,
        weights_D3=weights_D3,
        token_shape=shape,
    )


def _categorical_sample(weights: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.empty((0,), dtype=np.int64)
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise GeometryError("categorical weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise GeometryError("categorical weights have zero total mass")
    cdf = np.cumsum(values, dtype=np.float64)
    draws = rng.random(int(count), dtype=np.float64) * total
    return np.searchsorted(cdf, draws, side="right").astype(np.int64, copy=False)


def sample_native_indices(
    x_m: Any,
    y_m: Any,
    z_m: Any,
    count: int,
    rng: np.random.Generator,
    *,
    mode: Literal["volume", "hub_band"] = "volume",
    hub_height_m: float = HUB_HEIGHT_M,
    diameter_m: float = D_M,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample exact native flat indices from volume or hub-band measure."""

    x_values, y_values, z_values = (
        _axis_array(axis, name=name) for name, axis in (("x", x_m), ("y", y_m), ("z", z_m))
    )
    wx, wy, wz = support_weights(x_values, y_values, z_values)
    if mode == "hub_band":
        half_band = 0.5 * float(diameter_m)
        z_mask = np.abs(z_values - float(hub_height_m)) <= half_band
        wz = np.where(z_mask, wz, 0.0)
        if not np.any(z_mask):
            raise GeometryError("hub-height band has no native z centres")
    elif mode != "volume":
        raise ValueError(f"unknown native sampling mode {mode!r}")
    ix = _categorical_sample(wx, int(count), rng)
    iy = _categorical_sample(wy, int(count), rng)
    iz = _categorical_sample(wz, int(count), rng)
    flat = ix + int(x_values.size) * (iy + int(y_values.size) * iz)
    node_weights = wx[ix] * wy[iy] * wz[iz]
    if mode == "hub_band":
        distribution_volume = float(wx.sum() * wy.sum() * wz.sum())
    else:
        distribution_volume = float(wx.sum() * wy.sum() * wz.sum())
    return flat.astype(np.int64, copy=False), node_weights.astype(np.float64, copy=False), distribution_volume


def gather_native_sample(
    run: Any,
    count: int,
    rng: np.random.Generator,
    *,
    mode: Literal["volume", "hub_band"] = "volume",
    hub_height_m: float = HUB_HEIGHT_M,
    diameter_m: float = D_M,
) -> NativeQuerySample:
    """Sample and gather native coordinates/velocity from one :class:`RunView`."""

    flat, node_weights, distribution_volume_m3 = sample_native_indices(
        run.x_m,
        run.y_m,
        run.z_m,
        count,
        rng,
        mode=mode,
        hub_height_m=hub_height_m,
        diameter_m=diameter_m,
    )
    nx, ny, _nz = (int(value) for value in run.shape_nxyz)
    ix = flat % nx
    yz = flat // nx
    iy = yz % ny
    iz = yz // ny
    coords_D = np.column_stack((run.x_m[ix], run.y_m[iy], run.z_m[iz])).astype(np.float32) / float(diameter_m)
    # A copy makes the result writable and avoids exposing read-only mmap views
    # to torch; it is bounded by the requested query count.
    velocity = np.asarray(run.U[flat], dtype=np.float32).copy()
    return NativeQuerySample(
        flat_indices=flat,
        coords_D=coords_D,
        velocity_mps=velocity,
        node_weights_m3=node_weights,
        distribution_volume_m3=float(distribution_volume_m3),
        mode=mode,
    )


def module_geometry(
    turbine_xy_D: Any,
    n_turbines: int,
    *,
    hub_height_D: float = HUB_HEIGHT_D,
    rotor_radius_D: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite active turbine centres, mask, and known module features."""

    xy = np.asarray(turbine_xy_D, dtype=np.float64)
    count = int(n_turbines)
    if xy.ndim != 2 or xy.shape[1] != 2 or count <= 0 or count > xy.shape[0]:
        raise GeometryError("turbine_xy_D must be [M,2] and n_turbines must select active rows")
    active_xy = xy[:count]
    if np.any(~np.isfinite(active_xy)):
        raise GeometryError("active turbine coordinates contain non-finite values")
    centers = np.column_stack((active_xy, np.full(count, float(hub_height_D), dtype=np.float64))).astype(np.float32)
    present = np.ones(count, dtype=np.float32)
    features = np.column_stack(
        (
            np.full(count, float(rotor_radius_D), dtype=np.float64),
            np.full(count, float(hub_height_D), dtype=np.float64),
        )
    ).astype(np.float32)
    return centers, present, features


def global_geometry_features(
    support: SupportGeometry,
    wind_direction_deg: float,
    n_turbines: int,
    *,
    reference_speed_mps: float = U_REF_MPS,
    positional_scale_D: Any = POSITIONAL_SCALE_D,
    direction_categories: tuple[float, ...] = WIND_DIRECTIONS_DEG,
    max_turbines: float = 30.0,
) -> np.ndarray:
    """Build the documented width-11 known global context vector."""

    scale = np.asarray(positional_scale_D, dtype=np.float64).reshape(3)
    one_hot = np.zeros(len(direction_categories), dtype=np.float64)
    direction = float(wind_direction_deg)
    matches = np.flatnonzero(np.isclose(np.asarray(direction_categories), direction))
    if matches.size != 1:
        raise GeometryError(f"wind direction {direction!r} is not one of {direction_categories!r}")
    one_hot[int(matches[0])] = 1.0
    values = np.concatenate(
        (
            one_hot,
            np.asarray([float(n_turbines) / float(max_turbines), float(reference_speed_mps) / float(U_REF_MPS)]),
            support.lower_D / scale,
            support.extent_D / scale,
        )
    )
    if values.shape != (11,) or not np.all(np.isfinite(values)):
        raise GeometryError("global geometry feature construction failed")
    return values.astype(np.float32)


__all__ = [
    "D_M",
    "ENV_TOKEN_COUNT",
    "ENV_TOKEN_SHAPE",
    "HUB_HEIGHT_D",
    "HUB_HEIGHT_M",
    "POSITIONAL_SCALE_D",
    "TARGET_COMPONENTS",
    "U_REF_MPS",
    "WIND_DIRECTIONS_DEG",
    "EnvironmentRepresentation",
    "GeometryError",
    "NativeQuerySample",
    "SupportGeometry",
    "cell_centre_weights",
    "environment_representation",
    "gather_native_sample",
    "global_geometry_features",
    "module_geometry",
    "sample_native_indices",
    "support_features",
    "support_geometry",
    "support_weights",
]
