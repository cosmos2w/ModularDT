"""Dense joint-response preparation with a shared regional environmental read."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dense_pairwise import DensePairwiseField
from .types import EncodedInterfaceCase


@dataclass(frozen=True)
class RegionalPool:
    """Padded batch representation of nonempty weighted regions.

    Region slots are sorted by the adapter-supplied integer ID.  A slot with
    ``valid=False`` is padding and has zero mass and ID ``-1``.  The padded
    representation keeps batched attention rectangular while the reduction
    itself omits empty groups.
    """

    values: torch.Tensor
    region_ids: torch.Tensor
    mass: torch.Tensor
    centroids: torch.Tensor | None
    valid: torch.Tensor


def pool_region_weighted(
    values: torch.Tensor,
    region_ids: torch.Tensor,
    weights: torch.Tensor,
    coordinates: torch.Tensor | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
) -> RegionalPool:
    """Pool values, mass and optionally coordinates by weighted region IDs.

    ``values`` has shape ``[B, E, ...]``; IDs and weights may be ``[E]`` or
    ``[B, E]``.  IDs do not need to be contiguous, and repeated IDs are
    reduced together.  Negative IDs and zero-weight entries are treated as
    padding; groups with no positive mass are omitted.  This deliberately
    uses adapter-supplied memberships and performs no grouping search.
    """

    if values.ndim < 2:
        raise ValueError("values must have shape [batch, environment, ...].")
    batch, environment = int(values.shape[0]), int(values.shape[1])

    def _batch_axis(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim == 1:
            if int(tensor.shape[0]) != environment:
                raise ValueError(f"{name} has {tensor.shape[0]} entries, expected {environment}.")
            return tensor.unsqueeze(0).expand(batch, -1)
        if tensor.ndim == 2 and tuple(tensor.shape) == (batch, environment):
            return tensor
        raise ValueError(f"{name} must have shape [E] or [B, E].")

    ids = _batch_axis(region_ids, "region_ids").to(device=values.device, dtype=torch.long)
    batch_weights = _batch_axis(weights, "weights").to(device=values.device)
    if torch.any(batch_weights < 0):
        raise ValueError("region pooling weights must be nonnegative.")
    if valid_mask is None:
        mask = torch.ones_like(batch_weights, dtype=torch.bool)
    else:
        mask = _batch_axis(valid_mask, "valid_mask") > 0.5
    mask = mask & (ids >= 0) & (batch_weights > 0)

    batch_coordinates: torch.Tensor | None = None
    if coordinates is not None:
        if coordinates.ndim == 2:
            if int(coordinates.shape[0]) != environment:
                raise ValueError("coordinates has the wrong environment axis length.")
            batch_coordinates = coordinates.unsqueeze(0).expand(batch, -1, -1)
        elif coordinates.ndim == 3 and tuple(coordinates.shape[:2]) == (batch, environment):
            batch_coordinates = coordinates
        else:
            raise ValueError("coordinates must have shape [E, spatial_dim] or [B, E, spatial_dim].")
        batch_coordinates = batch_coordinates.to(device=values.device)

    value_dtype = values.dtype
    weighted_values: list[torch.Tensor] = []
    output_ids: list[torch.Tensor] = []
    output_mass: list[torch.Tensor] = []
    output_centroids: list[torch.Tensor] | None = [] if batch_coordinates is not None else None
    output_valid_counts: list[int] = []
    for batch_index in range(batch):
        selected = mask[batch_index]
        selected_ids = ids[batch_index, selected]
        selected_weights = batch_weights[batch_index, selected].to(dtype=value_dtype)
        if selected_ids.numel() == 0:
            unique_ids = ids.new_empty((0,))
            inverse = ids.new_empty((0,))
        else:
            unique_ids, inverse = torch.unique(selected_ids, sorted=True, return_inverse=True)
        region_count = int(unique_ids.numel())
        output_valid_counts.append(region_count)
        mass = values.new_zeros(region_count)
        if region_count:
            mass.index_add_(0, inverse, selected_weights)
        pooled = values.new_zeros((region_count, *values.shape[2:]))
        if region_count:
            weighted = values[batch_index, selected] * selected_weights.reshape(
                (-1,) + (1,) * (values.ndim - 2)
            )
            pooled.index_add_(0, inverse, weighted)
            pooled = pooled / mass.reshape((-1,) + (1,) * (values.ndim - 2))
        weighted_values.append(pooled)
        output_ids.append(unique_ids)
        output_mass.append(mass)
        if output_centroids is not None:
            coords = batch_coordinates[batch_index, selected]
            centroid = coords.new_zeros((region_count, coords.shape[-1]))
            if region_count:
                coordinate_weights = batch_weights[batch_index, selected].to(coords.dtype)
                centroid.index_add_(0, inverse, coordinate_weights[:, None] * coords)
                centroid = centroid / mass.to(coords.dtype)[:, None]
            output_centroids.append(centroid)

    max_regions = max(output_valid_counts, default=0)
    padded_values = values.new_zeros((batch, max_regions, *values.shape[2:]))
    padded_ids = ids.new_full((batch, max_regions), -1)
    padded_mass = values.new_zeros((batch, max_regions))
    padded_valid = torch.zeros((batch, max_regions), device=values.device, dtype=torch.bool)
    padded_centroids = (
        batch_coordinates.new_zeros((batch, max_regions, batch_coordinates.shape[-1]))
        if batch_coordinates is not None
        else None
    )
    for batch_index, region_count in enumerate(output_valid_counts):
        if region_count == 0:
            continue
        padded_values[batch_index, :region_count] = weighted_values[batch_index]
        padded_ids[batch_index, :region_count] = output_ids[batch_index]
        padded_mass[batch_index, :region_count] = output_mass[batch_index]
        padded_valid[batch_index, :region_count] = True
        if padded_centroids is not None and output_centroids is not None:
            padded_centroids[batch_index, :region_count] = output_centroids[batch_index]
    return RegionalPool(padded_values, padded_ids, padded_mass, padded_centroids, padded_valid)


class RegionalResponseField(DensePairwiseField):
    """Dense fine messages followed by one shared response state per region."""

    def __init__(
        self,
        hidden_dim: int,
        message_hidden_dim: int,
        num_heads: int,
        fourier_frequencies: int,
        *,
        response_region_block_shape: tuple[int, ...] = (2, 2),
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            hidden_dim,
            message_hidden_dim,
            num_heads,
            fourier_frequencies,
            activation_checkpointing=activation_checkpointing,
        )
        # Membership is owned by the adapter.  Keep accepting the setting so
        # a profile can be shared while older integration code is updated;
        # no grouping is inferred from this value in the reusable backend.
        self.response_region_block_shape = tuple(int(value) for value in response_region_block_shape)

    def _resolve_region_ids(
        self,
        encoded: EncodedInterfaceCase,
        supplied_region_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        region_ids = supplied_region_ids
        if region_ids is None:
            region_ids = getattr(encoded, "env_region_ids", None)
        if region_ids is None:
            raise ValueError(
                "regional_response_honf requires adapter-supplied env_region_ids aligned with env_coords."
            )
        return region_ids.to(device=encoded.env_tokens.device, dtype=torch.long)

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        region_ids: torch.Tensor | None = None,
        return_routing_maps: bool = False,
    ) -> dict[str, torch.Tensor]:
        del return_routing_maps
        fine = self.prepare_fine_messages(encoded, module_states)
        ids = self._resolve_region_ids(encoded, region_ids)
        combined = torch.cat([fine["env_tokens"], fine["environment_messages"]], dim=-1)
        pooled = pool_region_weighted(
            combined,
            ids,
            encoded.env_weights,
            encoded.env_coords,
        )
        hidden = int(fine["env_tokens"].shape[-1])
        pooled_env = pooled.values[..., :hidden]
        pooled_em = pooled.values[..., hidden:]
        region_global = encoded.global_token[:, None, :].expand(-1, pooled_env.shape[1], -1)
        regional_state = pooled_env + self.env_update(
            torch.cat([pooled_env, pooled_em, region_global], dim=-1)
        )
        regional_keys, regional_values = self.project_environment_sources(regional_state)
        if pooled.centroids is None:
            raise RuntimeError("Regional response preparation requires environmental coordinates.")
        return {
            "module_tokens": fine["module_tokens"],
            "regional_response_states": regional_state,
            "regional_coords": pooled.centroids,
            "regional_weights": pooled.mass,
            "regional_valid": pooled.valid,
            "regional_ids": pooled.region_ids,
            # These are live graph tensors.  They are intentionally created
            # afresh for each physical preparation pass.
            "regional_keys": regional_keys,
            "regional_values": regional_values,
        }

    def read(
        self,
        state: dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        module_context = self.read_module(state, encoded, receivers, receiver_features)
        regional_context, env_attention = self.read_regional(
            state,
            encoded,
            receivers,
            receiver_features,
            return_routing_maps=bool(return_routing_maps),
        )
        aux: dict[str, torch.Tensor] = {
            "dense_module_context_norm": torch.linalg.vector_norm(module_context, dim=-1),
            "regional_context_norm": torch.linalg.vector_norm(regional_context, dim=-1),
            "regional_response_count": state["regional_valid"].sum(dim=1),
        }
        if return_routing_maps and env_attention is not None:
            aux["regional_environment_attention"] = env_attention
        return module_context + regional_context, aux

    def read_regional(
        self,
        state: dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read only the shared regional environmental response."""

        return self.read_environment_projected(
            state["regional_keys"],
            state["regional_values"],
            encoded,
            receivers,
            receiver_features,
            state["regional_coords"],
            state["regional_weights"],
            source_mask=state["regional_valid"],
            return_routing_maps=bool(return_routing_maps),
        )


__all__ = [
    "RegionalPool",
    "RegionalResponseField",
    "pool_region_weighted",
]
