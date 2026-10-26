"""Case-neutral geometry helpers for routing affinities.

The generic router consumes descriptors and a finite optional resistance
penalty.  It does not know whether a descriptor represents a wall, inlet,
material, or any other physical concept; those meanings stay in the adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .types import RoutingGeometry


def stable_descriptor_norm(values: torch.Tensor) -> torch.Tensor:
    """Bound descriptor norm below one without a near-zero division."""

    if values.ndim < 1:
        raise ValueError("Descriptor values must have at least one dimension.")
    squared_norm = values.square().sum(dim=-1, keepdim=True)
    return values / torch.sqrt(1.0 + squared_norm)


def geometry_features(
    geometry: RoutingGeometry | Any | None,
    points: torch.Tensor,
    *,
    feature_dim: int | None = None,
) -> torch.Tensor:
    """Evaluate adapter descriptors and validate their point-aligned shape.

    A missing provider is represented by an explicitly requested zero-width or
    ``feature_dim`` zero tensor.  This keeps generic numerical fixtures
    dimension-neutral while making the production adapter's features explicit.
    """

    if points.ndim < 1:
        raise ValueError("Routing points must have at least one coordinate dimension.")
    if geometry is None:
        width = 0 if feature_dim is None else int(feature_dim)
        if width < 0:
            raise ValueError("feature_dim must be nonnegative.")
        return points.new_zeros((*points.shape[:-1], width))
    method = getattr(geometry, "features", None)
    if method is None or not callable(method):
        raise TypeError("Routing geometry must provide callable features(points).")
    result = method(points)
    if not isinstance(result, torch.Tensor):
        result = points.new_tensor(result)
    else:
        result = result.to(device=points.device, dtype=points.dtype)
    expected_prefix = tuple(points.shape[:-1])
    if result.ndim != points.ndim or tuple(result.shape[:-1]) != expected_prefix:
        raise ValueError("Routing geometry features must align with points on all leading dimensions.")
    if feature_dim is not None and int(result.shape[-1]) != int(feature_dim):
        raise ValueError(
            f"Routing geometry returned {int(result.shape[-1])} features; expected {int(feature_dim)}."
        )
    return result


def geometry_length_scale(
    geometry: RoutingGeometry | Any | None,
    points: torch.Tensor,
    *,
    fallback: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a positive per-coordinate routing scale on ``points``' device."""

    raw = getattr(geometry, "length_scale", None) if geometry is not None else fallback
    if raw is None:
        if fallback is None:
            raise ValueError("Routing geometry must provide a positive length_scale.")
        raw = fallback
    scale = torch.as_tensor(raw, device=points.device, dtype=points.dtype)
    dimension = int(points.shape[-1])
    if scale.ndim == 0:
        scale = scale.expand(dimension)
    if scale.ndim != 1 or int(scale.shape[0]) != dimension:
        raise ValueError(f"Routing length_scale must contain one value per coordinate ({dimension}).")
    # Provider metadata is validated at adapter construction.  Avoid a
    # content-dependent CUDA synchronization for every receiver chunk; retain
    # the inexpensive eager check for CPU fixtures and static metadata.
    if scale.device.type != "cuda" and (not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any())):
        raise ValueError("Routing length_scale must be finite and strictly positive.")
    return scale


def geometry_bounds(
    geometry: RoutingGeometry | Any,
    points: torch.Tensor,
) -> torch.Tensor:
    """Return adapter bounds as a ``[d,2]`` tensor after basic validation."""

    if geometry is None or not hasattr(geometry, "bounds"):
        raise ValueError("Routing geometry must provide bounds for bounded candidates.")
    raw = torch.as_tensor(geometry.bounds, device=points.device, dtype=points.dtype)
    dimension = int(points.shape[-1])
    if tuple(raw.shape) != (dimension, 2):
        raise ValueError(f"Routing bounds must have shape [{dimension}, 2].")
    if raw.device.type != "cuda" and (not bool(torch.isfinite(raw).all()) or bool((raw[:, 1] <= raw[:, 0]).any())):
        raise ValueError("Routing bounds must be finite and strictly increasing.")
    return raw


def geometry_resistance(
    geometry: RoutingGeometry | Any | None,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    relation_type: str = "routing",
) -> torch.Tensor:
    """Evaluate an adapter resistance or return an explicit neutral zero.

    The result follows the broadcasted leading shape of ``starts`` and
    ``ends``.  The final coordinate dimension is removed.  A finite negative
    value is rejected rather than interpreted as an invented physical gain.
    """

    if starts.ndim < 1 or ends.ndim < 1 or int(starts.shape[-1]) != int(ends.shape[-1]):
        raise ValueError("Resistance endpoints must have matching coordinate dimensions.")
    leading = torch.broadcast_shapes(starts.shape[:-1], ends.shape[:-1])
    if geometry is None:
        return starts.new_zeros(leading)
    method = getattr(geometry, "resistance", None)
    if method is None or not callable(method):
        return starts.new_zeros(leading)
    # The maintained ThermalChannel provider requires relation_type.  The
    # positional fallback accommodates a small numerical fixture with the
    # same semantics but a positional-only implementation.
    try:
        resistance = method(starts, ends, relation_type=relation_type)
    except TypeError as keyword_error:
        try:
            resistance = method(starts, ends, relation_type)
        except TypeError:
            raise keyword_error
    if not isinstance(resistance, torch.Tensor):
        resistance = starts.new_tensor(resistance)
    else:
        resistance = resistance.to(device=starts.device, dtype=starts.dtype)
    try:
        resistance = torch.broadcast_to(resistance, leading)
    except RuntimeError as error:
        raise ValueError("Routing resistance must align with endpoint leading dimensions.") from error
    if resistance.device.type != "cuda":
        if not bool(torch.isfinite(resistance).all()):
            raise ValueError("Routing resistance must be finite.")
        if bool((resistance < 0).any()):
            raise ValueError("Routing resistance must be nonnegative.")
    return resistance


def segment_resistance(
    geometry: RoutingGeometry | Any | None,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    relation_type: str = "routing",
    samples: int = 8,
) -> torch.Tensor:
    """Sample an optional pointwise resistance field along a line segment.

    This is the fixed midpoint rule from Equation (2): one quadrature is
    applied to each segment and the result is scaled by its endpoint distance
    divided by the geometric mean routing length.  Adapters that only expose
    endpoint ``resistance(a, b)`` use that finite penalty directly.  Missing
    resistance data is an explicit neutral zero, never a fabricated barrier.
    """

    if starts.ndim < 1 or ends.ndim < 1 or int(starts.shape[-1]) != int(ends.shape[-1]):
        raise ValueError("Segment endpoints must have matching coordinate dimensions.")
    if int(samples) <= 0:
        raise ValueError("Segment resistance samples must be positive.")
    field = getattr(geometry, "resistance_field", None) if geometry is not None else None
    if field is None or not callable(field):
        return geometry_resistance(geometry, starts, ends, relation_type=relation_type)
    leading = torch.broadcast_shapes(starts.shape[:-1], ends.shape[:-1])
    starts_b = torch.broadcast_to(starts, (*leading, starts.shape[-1]))
    ends_b = torch.broadcast_to(ends, (*leading, ends.shape[-1]))
    scale = geometry_length_scale(geometry, starts_b)
    ell0 = torch.exp(torch.log(scale).mean())
    sample_positions = (
        torch.arange(int(samples), device=starts.device, dtype=starts.dtype) + 0.5
    ) / float(samples)
    points = starts_b[..., None, :] + (ends_b - starts_b)[..., None, :] * sample_positions.reshape(
        *([1] * len(leading)), int(samples), 1
    )
    try:
        values = field(points, relation_type=relation_type)
    except TypeError as keyword_error:
        try:
            values = field(points, relation_type)
        except TypeError:
            raise keyword_error
    if not isinstance(values, torch.Tensor):
        values = starts.new_tensor(values)
    else:
        values = values.to(device=starts.device, dtype=starts.dtype)
    expected_shape = (*leading, int(samples))
    if tuple(values.shape) != expected_shape:
        raise ValueError("Pointwise routing resistance must return [...,samples] values.")
    if values.device.type != "cuda":
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Pointwise routing resistance must be finite.")
        if bool((values < 0).any()):
            raise ValueError("Pointwise routing resistance must be nonnegative.")
    weighted = values.mean(dim=-1)
    distance = torch.linalg.vector_norm(ends_b - starts_b, dim=-1)
    return distance / ell0 * weighted


# Names used by numerical fixtures and case adapters can choose either word;
# both point to the same exactly-once midpoint quadrature.
sample_segment_resistance = segment_resistance
line_resistance = segment_resistance


def routing_affinity(
    source_descriptors: torch.Tensor,
    source_coords: torch.Tensor,
    hub_descriptors: torch.Tensor,
    hub_coords: torch.Tensor,
    propensity: torch.Tensor,
    length_scale: torch.Tensor | Sequence[float],
    *,
    resistance: torch.Tensor | None = None,
    content_scale: float = 2.0,
    geometry_scale: float = 0.25,
    propensity_scale: float = 0.25,
) -> torch.Tensor:
    """Compute the bounded shared source/query-to-hub affinity.

    This is Equation (1) from the mathematical design.  The returned tensor
    has shape ``[B,N,K]`` and contains only scalar route logits; it never
    carries source values to the field head.
    """

    if source_descriptors.ndim != 3 or hub_descriptors.ndim != 3:
        raise ValueError("Routing descriptors must have shape [B,N,D] and [B,K,D].")
    if source_coords.ndim != 3 or hub_coords.ndim != 3:
        raise ValueError("Routing coordinates must have shape [B,N,d] and [B,K,d].")
    batch, source_count, descriptor_dim = source_descriptors.shape
    if tuple(hub_descriptors.shape[:1]) != (batch,) or int(hub_descriptors.shape[-1]) != descriptor_dim:
        raise ValueError("Source and hub descriptors must share batch and descriptor dimensions.")
    if tuple(source_coords.shape[:2]) != (batch, source_count):
        raise ValueError("Source coordinates must align with source descriptors.")
    if int(hub_coords.shape[0]) != batch or int(hub_coords.shape[1]) != int(hub_descriptors.shape[1]):
        raise ValueError("Hub coordinates must align with hub descriptors.")
    if tuple(propensity.shape) != (batch, int(hub_descriptors.shape[1])):
        raise ValueError("Hub propensity must have shape [B,K].")
    scale = torch.as_tensor(length_scale, device=source_coords.device, dtype=source_coords.dtype)
    if scale.ndim == 1:
        if int(scale.shape[0]) != int(source_coords.shape[-1]):
            raise ValueError("Routing length scale must match coordinate dimension.")
        scale = scale.reshape(1, 1, 1, -1)
    elif scale.ndim == 2:
        if tuple(scale.shape) != (batch, int(source_coords.shape[-1])):
            raise ValueError("Batched routing length scale must have shape [B,d].")
        scale = scale[:, None, None, :]
    else:
        raise ValueError("Routing length scale must have shape [d] or [B,d].")
    if scale.device.type != "cuda" and (not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any())):
        raise ValueError("Routing length scale must be finite and strictly positive.")
    delta = (source_coords[:, :, None, :] - hub_coords[:, None, :, :]) / scale
    distance_term = torch.log1p(delta.square().sum(dim=-1))
    logits = (
        float(content_scale) * torch.einsum("bnd,bkd->bnk", source_descriptors, hub_descriptors)
        - float(geometry_scale) * distance_term
        + float(propensity_scale) * propensity[:, None, :]
    )
    if resistance is not None:
        if tuple(resistance.shape) != tuple(logits.shape):
            raise ValueError("Routing resistance must have shape [B,N,K].")
        logits = logits - resistance
    return logits
