"""Candidate generation and shared routing descriptors.

The descriptor maps are shared by all routed strategies.  Candidate motion is
kept here as a small scalar routing operation so that the physical readers do
not need to know how a candidate was generated.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .geometry import routing_affinity, stable_descriptor_norm


class _LazySiLUProjection(nn.Module):
    """Two-layer SiLU map for adapter-defined descriptor widths."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LazyLinear(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class RoutingDescriptorMaps(nn.Module):
    """Shared live M/E/Q routing maps used by the delayed fine reader.

    The three descriptor maps intentionally have separate parameters because
    module sources, environment sources, and receivers have different fine
    states/features.  They all produce the same bounded descriptor width and
    share one scalar propensity head for module-induced candidate hubs.  This
    class owns routing metadata only; it has no fine field-value path.
    """

    def __init__(self, source_dim: int, descriptor_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        if int(source_dim) <= 0 or int(descriptor_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("Routed router dimensions must be positive.")
        self.source_dim = int(source_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.module_descriptor = _LazySiLUProjection(hidden_dim, descriptor_dim)
        self.environment_descriptor = _LazySiLUProjection(hidden_dim, descriptor_dim)
        self.query_descriptor = _LazySiLUProjection(hidden_dim, descriptor_dim)
        self.propensity = _LazySiLUProjection(hidden_dim, 1)

    @staticmethod
    def _global_per_point(values: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        if global_features.ndim != 2 or int(global_features.shape[0]) != int(values.shape[0]):
            raise ValueError("global_features must have shape [B,G] aligned with values.")
        return global_features[:, None, :].expand(-1, values.shape[1], -1)

    def module_map(
        self,
        module_states: torch.Tensor,
        module_geometry: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        return stable_descriptor_norm(
            self.module_descriptor(
                torch.cat((module_states, module_geometry, self._global_per_point(module_states, global_features)), dim=-1)
            )
        )

    def environment_map(
        self,
        environment_states: torch.Tensor,
        environment_geometry: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        return stable_descriptor_norm(
            self.environment_descriptor(
                torch.cat(
                    (environment_states, environment_geometry, self._global_per_point(environment_states, global_features)),
                    dim=-1,
                )
            )
        )

    def query_map(
        self,
        query_features: torch.Tensor,
        query_geometry: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        return stable_descriptor_norm(
            self.query_descriptor(
                torch.cat((query_features, query_geometry, self._global_per_point(query_features, global_features)), dim=-1)
            )
        )

    def hub_propensity(
        self,
        hub_descriptors: torch.Tensor,
        hub_geometry: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        raw = torch.cat(
            (hub_descriptors, hub_geometry, self._global_per_point(hub_descriptors, global_features)),
            dim=-1,
        )
        return torch.tanh(self.propensity(raw).squeeze(-1))


def fixed_data_mean_shift(
    source_coords: torch.Tensor,
    source_descriptors: torch.Tensor,
    length_scale: torch.Tensor,
    *,
    source_valid: torch.Tensor | None = None,
    steps: int = 3,
    feature_bandwidth: float = 1.0,
    resistance_fn: Any | None = None,
    return_trajectory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Move every module-seeded candidate through fixed-data mean shift.

    ``source_coords`` and ``source_descriptors`` are the fixed samples of the
    distribution.  The candidate state is updated exactly ``steps`` times,
    while the fixed samples remain live in the autograd graph.  The canonical
    Run 2100 profile requires ``steps == 3``; the explicit argument is useful
    for the T=0 numerical contract and for small unit fixtures.  No candidate
    identity is merged or removed.

    ``resistance_fn`` is an optional runtime geometry callback.  It receives
    broadcasted ``[B,K,N,d]`` candidate/source endpoints and must return
    ``[B,K,N]`` finite penalties.  It is called once per iteration, after the
    current candidate coordinates are formed.

    Returns the final ``[B,K,d+D]`` states and, when requested, a trajectory
    with the initial state followed by each update at ``[B,steps+1,K,d+D]``.
    The implementation uses log-space softmax and a finite mask sentinel, so
    an all-invalid source row contributes zero rather than NaN.
    """

    if not isinstance(source_coords, torch.Tensor) or source_coords.ndim != 3:
        raise ValueError("source_coords must have shape [B,N,d].")
    if not isinstance(source_descriptors, torch.Tensor) or source_descriptors.ndim != 3:
        raise ValueError("source_descriptors must have shape [B,N,D].")
    if tuple(source_descriptors.shape[:2]) != tuple(source_coords.shape[:2]):
        raise ValueError("Mean-shift source coordinates and descriptors must align on [B,N].")
    if not source_coords.is_floating_point() or not source_descriptors.is_floating_point():
        raise TypeError("Mean-shift source coordinates and descriptors must be floating point.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("Mean-shift steps must be a nonnegative integer.")
    bandwidth = float(feature_bandwidth)
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("Mean-shift feature bandwidth must be finite and positive.")

    batch, source_count, spatial_dim = (int(v) for v in source_coords.shape)
    scale = torch.as_tensor(length_scale, device=source_coords.device, dtype=source_coords.dtype)
    if scale.ndim == 1:
        if int(scale.shape[0]) != spatial_dim:
            raise ValueError("Mean-shift length_scale must match the coordinate dimension.")
        scale = scale.reshape(1, 1, spatial_dim)
    elif scale.ndim == 2:
        if tuple(scale.shape) != (batch, spatial_dim):
            raise ValueError("Batched mean-shift length_scale must have shape [B,d].")
        scale = scale[:, None, :]
    else:
        raise ValueError("Mean-shift length_scale must have shape [d] or [B,d].")
    if scale.device.type != "cuda" and (
        not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any())
    ):
        raise ValueError("Mean-shift length_scale must be finite and strictly positive.")

    if source_valid is None:
        valid = torch.ones((batch, source_count), device=source_coords.device, dtype=torch.bool)
    else:
        if tuple(source_valid.shape) != (batch, source_count):
            raise ValueError("source_valid must align with mean-shift source samples.")
        valid = source_valid.to(device=source_coords.device, dtype=torch.bool)

    fixed_samples = torch.cat((source_coords / scale, source_descriptors), dim=-1)
    attractors = fixed_samples.clone()
    trajectory: list[torch.Tensor] | None = [attractors] if return_trajectory else None

    # A zero-source fixture is valid for the generic empty-module contract.
    # The candidate bank is retained (with no active source) and no softmax is
    # attempted over an empty final axis.
    if source_count == 0:
        if return_trajectory:
            assert trajectory is not None
            trajectory.extend([attractors] * int(steps))
            return attractors, torch.stack(trajectory, dim=1)
        return attractors, None

    finite_sentinel = torch.finfo(source_coords.dtype).min
    for _ in range(int(steps)):
        candidate_positions = attractors[..., :spatial_dim]
        candidate_features = attractors[..., spatial_dim:]
        position_delta = candidate_positions[:, :, None, :] - fixed_samples[:, None, :, :spatial_dim]
        feature_delta = candidate_features[:, :, None, :] - fixed_samples[:, None, :, spatial_dim:]
        log_kernel = -0.5 * position_delta.square().sum(dim=-1)
        log_kernel = log_kernel - (0.5 / (bandwidth * bandwidth)) * feature_delta.square().sum(dim=-1)
        if resistance_fn is not None:
            starts = (candidate_positions * scale)[:, :, None, :].expand(-1, -1, source_count, -1)
            ends = source_coords[:, None, :, :].expand(-1, int(attractors.shape[1]), -1, -1)
            resistance = resistance_fn(starts, ends)
            if not isinstance(resistance, torch.Tensor):
                resistance = source_coords.new_tensor(resistance)
            resistance = resistance.to(device=source_coords.device, dtype=source_coords.dtype)
            if tuple(resistance.shape) != tuple(log_kernel.shape):
                raise ValueError("Mean-shift resistance must have shape [B,K,N].")
            log_kernel = log_kernel - resistance
        valid_kernel = valid[:, None, :].expand_as(log_kernel)
        masked_kernel = torch.where(
            valid_kernel,
            log_kernel,
            log_kernel.new_full((), finite_sentinel),
        )
        probabilities = torch.softmax(masked_kernel, dim=-1)
        probabilities = probabilities * valid_kernel.to(dtype=probabilities.dtype)
        probability_mass = probabilities.sum(dim=-1, keepdim=True)
        probabilities = probabilities / probability_mass.clamp_min(torch.finfo(probabilities.dtype).tiny)
        updated = torch.matmul(probabilities, fixed_samples)
        has_source = valid.any(dim=-1)[:, None, None]
        attractors = torch.where(has_source, updated, attractors)
        if trajectory is not None:
            trajectory.append(attractors)

    if trajectory is None:
        return attractors, None
    return attractors, torch.stack(trajectory, dim=1)


# Keep the name used by the in-progress routed backend as a compatibility
# alias.  The descriptive name makes clear that this module owns shared
# M/E/Q descriptor maps and does not itself execute field interactions.
RoutedRoutingRouter = RoutingDescriptorMaps


__all__ = [
    "RoutedRoutingRouter",
    "RoutingDescriptorMaps",
    "fixed_data_mean_shift",
    "routing_affinity",
]
