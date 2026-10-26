"""Opt-in snapshots of the actual staged contextualizer routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

from .contracts import _validate_ids, tensor_versions

RouteName = Literal["mm", "me", "em", "ee"]


@dataclass(frozen=True)
class NativeRouteSnapshot:
    """A typed route's live score tensors and exact native correspondence.

    ``raw_scores`` includes the live bounded temperature through ``query_scaled``
    and the route's geometry bias. It excludes donor log measure, so the same
    scores can define density-relative membership without mixing physical
    quadrature weights into the screening threshold.
    """

    route: RouteName
    query_scaled: Tensor  # [B,H,D,Dh]
    key_normalized: Tensor  # [B,H,S,Dh]
    geometry_bias: Tensor  # [B,H,D,S]
    raw_scores: Tensor  # [B,H,D,S], excludes donor log measure
    allowed: Tensor  # bool [B,D,S], includes route self/padding rules
    destination_mask: Tensor  # bool [B,D]
    donor_mask: Tensor  # bool [B,S], scene presence before positive-measure route filtering
    destination_ids: Tensor  # int [B,D]
    donor_ids: Tensor  # int [B,S]
    donor_measure: Tensor  # native measure [B,S]
    measure_convention: str
    stage_identity: str
    owner_token: int | None = None
    parameter_versions: tuple[tuple[int, int], ...] = ()
    buffer_versions: tuple[tuple[int, int], ...] = ()
    scene_identity: tuple[Any, Any, Any] | None = None
    scene_tensor_versions: tuple[tuple[int, int, int], ...] = ()
    model_config_key: tuple[Any, ...] = ()
    _tensor_versions: tuple[tuple[int, int, int], ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.route not in ("mm", "me", "em", "ee"):
            raise ValueError("route must be one of mm, me, em, or ee.")
        if self.query_scaled.ndim != 4 or self.key_normalized.ndim != 4:
            raise ValueError("Captured Q/K must have shapes [B,H,D,Dh] and [B,H,S,Dh].")
        batch, heads, destinations, head_width = self.query_scaled.shape
        if self.key_normalized.shape[:2] != (batch, heads) or self.key_normalized.shape[-1] != head_width:
            raise ValueError("Captured Q/K batch, head, and feature axes must match.")
        donors = self.key_normalized.shape[2]
        expected = (batch, heads, destinations, donors)
        if self.geometry_bias.shape != expected or self.raw_scores.shape != expected:
            raise ValueError("Captured route scores must have shape [B,H,D,S].")
        if self.allowed.shape != (batch, destinations, donors) or self.allowed.dtype != torch.bool:
            raise ValueError("allowed must be boolean [B,D,S].")
        if self.destination_mask.shape != (batch, destinations) or self.destination_mask.dtype != torch.bool:
            raise ValueError("destination_mask must be boolean [B,D].")
        if self.donor_mask.shape != (batch, donors) or self.donor_mask.dtype != torch.bool:
            raise ValueError("donor_mask must be boolean [B,S].")
        if self.destination_ids.shape != (batch, destinations) or self.donor_ids.shape != (batch, donors):
            raise ValueError("Native IDs must match destination/donor axes.")
        if self.destination_ids.dtype not in (torch.int32, torch.int64) or self.donor_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("Native IDs must use integer tensor dtypes.")
        if self.donor_measure.shape != (batch, donors):
            raise ValueError("donor_measure must have shape [B,S].")
        _validate_ids(self.destination_ids, self.destination_mask, "destination native IDs")
        _validate_ids(self.donor_ids, self.donor_mask, "donor native IDs")
        valid_route_pairs = (
            self.destination_mask[:, :, None]
            & self.donor_mask[:, None, :]
            & (self.donor_measure > 0)[:, None, :]
        )
        if self.allowed.device.type == "cuda":
            torch._assert_async(
                (~self.allowed | valid_route_pairs).all(),
                "Captured route cannot allow padded destination or donor records.",
            )
            torch._assert_async(
                (~self.allowed[:, None, :, :] | torch.isfinite(self.raw_scores)).all(),
                "Captured eligible route scores must be finite.",
            )
        else:
            if bool((self.allowed & ~valid_route_pairs).any()):
                raise ValueError("Captured route cannot allow padded destination or donor records.")
            if bool((self.allowed[:, None, :, :] & ~torch.isfinite(self.raw_scores)).any()):
                raise ValueError("Captured eligible route scores must be finite.")
        if not self.measure_convention:
            raise ValueError("measure_convention must be explicit.")
        if not self.stage_identity:
            raise ValueError("stage_identity must identify the captured stage and route.")
        object.__setattr__(
            self,
            "_tensor_versions",
            tensor_versions(
                self.query_scaled,
                self.key_normalized,
                self.geometry_bias,
                self.raw_scores,
                self.allowed,
                self.destination_mask,
                self.donor_mask,
                self.destination_ids,
                self.donor_ids,
                self.donor_measure,
            ),
        )

    def assert_current(self) -> None:
        """Reject in-place changes to a captured route or its native IDs."""
        current = tensor_versions(
            self.query_scaled,
            self.key_normalized,
            self.geometry_bias,
            self.raw_scores,
            self.allowed,
            self.destination_mask,
            self.donor_mask,
            self.destination_ids,
            self.donor_ids,
            self.donor_measure,
        )
        if current != self._tensor_versions:
            raise ValueError("Captured native route scores, masks, measures, or IDs changed in place.")


def make_native_route_snapshot(
    *,
    route: RouteName,
    query_scaled: Tensor,
    key_normalized: Tensor,
    geometry_bias: Tensor,
    allowed: Tensor,
    destination_mask: Tensor,
    donor_mask: Tensor,
    destination_ids: Tensor,
    donor_ids: Tensor,
    donor_measure: Tensor,
    stage_identity: str,
    owner_token: int | None = None,
    parameter_versions: tuple[tuple[int, int], ...] = (),
    buffer_versions: tuple[tuple[int, int], ...] = (),
    scene_identity: tuple[Any, Any, Any] | None = None,
    scene_tensor_versions: tuple[tuple[int, int, int], ...] = (),
    model_config_key: tuple[Any, ...] = (),
) -> NativeRouteSnapshot:
    """Build a snapshot from the exact Q/K and geometry used by staged SDPA."""
    raw_scores = torch.matmul(query_scaled, key_normalized.transpose(-2, -1)) + geometry_bias
    return NativeRouteSnapshot(
        route=route,
        query_scaled=query_scaled,
        key_normalized=key_normalized,
        geometry_bias=geometry_bias,
        raw_scores=raw_scores,
        allowed=allowed,
        destination_mask=destination_mask,
        donor_mask=donor_mask,
        destination_ids=destination_ids,
        donor_ids=donor_ids,
        donor_measure=donor_measure,
        measure_convention="unnormalized_native_donor_measure_in_attention_softmax",
        stage_identity=stage_identity,
        owner_token=owner_token,
        parameter_versions=parameter_versions,
        buffer_versions=buffer_versions,
        scene_identity=scene_identity,
        scene_tensor_versions=scene_tensor_versions,
        model_config_key=model_config_key,
    )
