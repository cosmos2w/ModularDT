"""Small indexed execution paths for prepared query-to-anchor covers."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .atlas import validate_cover_anchor_identity
from .contracts import PreparedProjectedAnchorValues, ReceiverCover


def _weighted_projected_cover_sum(values: Tensor, indices: Tensor, weights: Tensor, validity: Tensor) -> Tensor:
    """Tensor-only selected gather and weighted reduction used as reference."""
    batch = values.shape[0]
    batch_indices = torch.arange(batch, device=values.device)[:, None, None].expand_as(indices)
    selected = values[batch_indices, indices]
    effective_weights = weights * validity.to(weights.dtype)
    return (selected * effective_weights[..., None]).sum(dim=2)


def compile_projected_cover_sum(
    *,
    fullgraph: bool = True,
    dynamic: bool = False,
) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    """Opt in to compiling only the indexed weighted-reduction tensor function.

    The standard reader remains eager. A caller must explicitly request this
    wrapper and pass it as ``weighted_sum_executor`` to
    :func:`gather_projected_cover_indexed`; no module or model graph is compiled.
    ``fullgraph=True`` makes graph breaks fail instead of silently falling back
    to eager sections.
    """
    return torch.compile(_weighted_projected_cover_sum, fullgraph=fullgraph, dynamic=dynamic)


def gather_projected_cover_indexed(
    anchor_values: PreparedProjectedAnchorValues | Tensor,
    cover: ReceiverCover,
    *,
    anchor_ids: Tensor | None = None,
    validate_identity: bool = True,
    expected_frame_key=None,
    expected_scene_version=None,
    expected_owner_token=None,
    weighted_sum_executor: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    """Read one fused ``[B,E,C]`` payload through a sparse receiver cover.

    The reference indexes only the selected anchors, then multiplies by live
    cover weights and reduces the slot axis. Autograd therefore accumulates
    value gradients at duplicate anchors and retains the derivative through
    cover weights to query and selected anchor coordinates.
    """
    if isinstance(anchor_values, PreparedProjectedAnchorValues):
        payload = anchor_values
        values = payload.values
        anchor_ids = payload.anchor_ids
        payload.validate_binding(
            frame_key=expected_frame_key if expected_frame_key is not None else cover.frame_key,
            scene_version=expected_scene_version,
            owner_token=expected_owner_token,
        )
        if cover.scene_version is not None and payload.scene_version != cover.scene_version:
            raise ValueError("Projected anchor payload and receiver cover belong to different scene versions.")
    else:
        values = anchor_values
        if any(value is not None for value in (expected_scene_version, expected_owner_token)):
            raise ValueError("Scene-bound projected gathers require PreparedProjectedAnchorValues.")
        if expected_frame_key is not None and expected_frame_key != cover.frame_key:
            raise ValueError("Receiver cover uses a different coordinate frame.")
    if values.ndim != 3 or values.shape[0] != cover.batch_size:
        raise ValueError("anchor_values must have shape [B,E,C] and match the cover batch.")
    if values.device != cover.anchor_indices.device or values.device != cover.weights.device:
        raise ValueError("Projected anchor values and receiver-cover tensors must share one device.")
    anchors = values.shape[1]
    if validate_identity:
        if anchor_ids is None:
            raise ValueError("anchor_ids are required for a validated projected gather.")
        validate_cover_anchor_identity(cover, anchor_ids)
        if anchors != anchor_ids.shape[1] and cover.anchor_indices.numel() and (
            (cover.anchor_indices < 0).any() or (cover.anchor_indices >= anchors).any()
        ):
            raise ValueError("Receiver cover contains an invalid anchor offset.")
    else:
        cover.validate_unchanged()
        if not cover._has_validation_seal(anchor_count=anchors, unchanged_checked=True) and cover.anchor_indices.numel() and (
            (cover.anchor_indices < 0).any() or (cover.anchor_indices >= anchors).any()
        ):
            raise ValueError("Receiver cover contains an invalid anchor offset.")
    indices = cover.anchor_indices.to(torch.long)
    executor = _weighted_projected_cover_sum if weighted_sum_executor is None else weighted_sum_executor
    return executor(values, indices, cover.weights, cover.validity)
