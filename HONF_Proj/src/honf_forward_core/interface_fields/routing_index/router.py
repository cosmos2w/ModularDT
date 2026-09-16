"""Module-induced candidate hubs and shared routing descriptors."""

from __future__ import annotations

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


# Keep the name used by the in-progress routed backend as a compatibility
# alias.  The descriptive name makes clear that this module owns shared
# M/E/Q descriptor maps and does not itself execute field interactions.
RoutedRoutingRouter = RoutingDescriptorMaps


__all__ = [
    "RoutedRoutingRouter",
    "RoutingDescriptorMaps",
    "routing_affinity",
]
