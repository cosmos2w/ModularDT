"""Reassociated IAG-C value gathering and source-indexed affine transport."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .atlas import validate_cover_anchor_identity
from .contracts import (
    PreparedAnchorValues,
    PreparedSourceIndexedResponse,
    ReceiverCover,
    TypedAssociationAtlas,
    TypedTokenValues,
)
from .normalization import normalize_donor_measure


def _require_same_ids(name: str, left: Tensor, right: Tensor) -> None:
    if left.shape != right.shape or not torch.equal(left, right):
        raise ValueError(f"{name} do not match in both identity and order.")


def _require_same_bool(name: str, left: Tensor, right: Tensor) -> None:
    if left.shape != right.shape or not torch.equal(left, right):
        raise ValueError(f"{name} do not match.")


def _contract_atlas_edges(
    coefficients: Tensor,
    token_values: Tensor,
    *,
    positive_support: Tensor,
    candidate_support: Tensor,
    destination_valid: Tensor,
    executor: Literal["dense", "selected"],
) -> tuple[Tensor, Tensor, Tensor]:
    """Contract typed donor values and report dense/selected edge work."""
    if executor not in ("dense", "selected"):
        raise ValueError("executor must be 'dense' or 'selected'.")
    if coefficients.ndim != 4 or positive_support.shape != coefficients.shape:
        raise ValueError("Atlas coefficients/support must have shape [B,H,E,N].")
    try:
        candidate_support = torch.broadcast_to(candidate_support, coefficients.shape)
    except RuntimeError as error:
        raise ValueError("candidate_support must broadcast to [B,H,E,N].") from error
    batch, heads, anchors, donors = coefficients.shape
    if token_values.ndim != 4 or token_values.shape[:3] != (batch, heads, donors):
        raise ValueError("Typed values must have shape [B,H,N,D] matching the atlas.")
    if destination_valid.shape != (batch, anchors):
        raise ValueError("Destination validity must have shape [B,E].")

    candidate_count = candidate_support.sum(dim=-1)
    if executor == "dense":
        contracted = torch.einsum("bhan,bhnd->bhad", coefficients, token_values)
        executed_count = torch.full_like(candidate_count, donors)
    else:
        edge_indices = torch.nonzero(positive_support, as_tuple=False)
        if edge_indices.numel():
            batch_index, head_index, anchor_index, donor_index = edge_indices.unbind(dim=-1)
            edge_coefficients = coefficients[
                batch_index, head_index, anchor_index, donor_index
            ]
            edge_values = token_values[batch_index, head_index, donor_index]
            edge_contributions = edge_values * edge_coefficients[:, None]
            destination_rows = (batch_index * heads + head_index) * anchors + anchor_index
            flat_contracted = edge_contributions.new_zeros(
                batch * heads * anchors, edge_contributions.shape[-1]
            )
            flat_contracted = flat_contracted.index_add(
                0, destination_rows, edge_contributions
            )
            contracted = flat_contracted.reshape(
                batch, heads, anchors, edge_contributions.shape[-1]
            )
            executed_count = torch.bincount(
                destination_rows, minlength=batch * heads * anchors
            ).reshape(batch, heads, anchors)
        else:
            # Preserve a valid empty graph for fully masked typed routes.
            contracted = token_values.sum(dim=2, keepdim=True).expand(
                batch, heads, anchors, token_values.shape[-1]
            ) * 0.0
            executed_count = torch.zeros_like(candidate_count)
    candidate_count = candidate_count * destination_valid[:, None, :].to(candidate_count.dtype)
    return contracted, executed_count, candidate_count


def prepare_anchor_values(
    atlas: TypedAssociationAtlas,
    token_values: TypedTokenValues,
    *,
    executor: Literal["dense", "selected"] = "dense",
) -> PreparedAnchorValues:
    """Compute ``P @ V`` and record dense or selected-edge work.

    The dense executor is the default identical-function control.  The
    selected executor gathers exactly every positive edge of the same current
    probability tensor, with no threshold.  ``executed_edge_counts`` records
    work per ``[B,H,E]`` row; it is separate from the query-side cover gather.
    """
    if atlas.kind != token_values.kind:
        raise ValueError("An association atlas and payload values must have the same physical type.")
    batch, heads, _anchors, donors = atlas.probabilities.shape
    if token_values.values.shape[:3] != (batch, heads, donors):
        raise ValueError("Typed token values must match the atlas batch/head/donor axes.")
    _require_same_ids("Atlas donor IDs and token value IDs", atlas.donor_ids, token_values.token_ids)
    typed_valid = token_values.valid & (token_values.measure > 0)
    _require_same_bool("Atlas and token donor validity", atlas.donor_valid, typed_valid)
    typed_measure = normalize_donor_measure(token_values.measure, token_values.valid)
    if not torch.allclose(typed_measure, atlas.normalized_measure, atol=0.0, rtol=0.0):
        raise ValueError("Atlas and token values were prepared with different donor measures.")
    values, executed_counts, candidate_counts = _contract_atlas_edges(
        atlas.probabilities,
        token_values.values,
        positive_support=atlas.probabilities > 0,
        candidate_support=atlas.donor_valid[:, None, None, :],
        destination_valid=atlas.destination_valid,
        executor=executor,
    )
    return PreparedAnchorValues(
        kind=atlas.kind,
        values=values,
        anchor_ids=atlas.anchor_ids,
        executed_edge_counts=executed_counts,
        candidate_edge_counts=candidate_counts,
    )


def _gather_cover_values(values: Tensor, cover: ReceiverCover) -> Tensor:
    """Index ``[B,H,E,D]`` directly into compact ``[B,Q,C,H,D]`` values.

    Avoiding an expanded ``[B,H,Q,E,D]`` view matters for backward too: its
    expand/gather graph can materialize the full route-shaped gradient even
    when the forward operation reads only a few anchors.
    """
    batch = cover.batch_size
    anchor_indices = cover.anchor_indices.to(torch.long)
    batch_indices = torch.arange(batch, device=values.device)[:, None, None].expand_as(anchor_indices)
    by_anchor = values.permute(0, 2, 1, 3)
    return by_anchor[batch_indices, anchor_indices]


def _validate_cover_for_gather(
    cover: ReceiverCover,
    anchor_count: int,
    anchor_ids: Tensor | None,
    *,
    validate_identity: bool,
) -> None:
    """Run strict version guards and only repeat semantic checks when unsealed."""
    if validate_identity:
        if anchor_ids is None:
            raise ValueError("anchor_ids are required for a validated receiver gather.")
        validate_cover_anchor_identity(cover, anchor_ids)
        if anchor_count != anchor_ids.shape[1] and cover.anchor_indices.numel() and (
            (cover.anchor_indices < 0).any() or (cover.anchor_indices >= anchor_count).any()
        ):
            raise ValueError("Receiver cover contains an invalid anchor offset.")
        return
    cover.validate_unchanged()
    if cover._has_validation_seal(anchor_count=anchor_count, unchanged_checked=True):
        return
    if cover.anchor_indices.numel() and (
        (cover.anchor_indices < 0).any() or (cover.anchor_indices >= anchor_count).any()
    ):
        raise ValueError("Receiver cover contains an invalid anchor offset.")


def gather_cover_indexed(
    anchor_values: PreparedAnchorValues | Tensor,
    cover: ReceiverCover,
    *,
    anchor_ids: Tensor | None = None,
    validate_identity: bool = True,
) -> Tensor:
    """Read only the cover's indexed anchors, returning ``[B,Q,H,D]``."""
    if isinstance(anchor_values, PreparedAnchorValues):
        values = anchor_values.values
        anchor_ids = anchor_values.anchor_ids
    else:
        values = anchor_values
    if values.ndim != 4 or values.shape[0] != cover.batch_size:
        raise ValueError("anchor_values must have shape [B,H,E,D] and match the cover batch.")
    _batch, _heads, anchors, _width = values.shape
    _validate_cover_for_gather(cover, anchors, anchor_ids, validate_identity=validate_identity)
    selected = _gather_cover_values(values, cover)
    weights = cover.weights[:, :, :, None, None]
    weights = weights * cover.validity[:, :, :, None, None].to(weights.dtype)
    return (selected * weights).sum(dim=2)


def gather_cover_dense_reference(
    anchor_values: PreparedAnchorValues | Tensor,
    cover: ReceiverCover,
    *,
    anchor_ids: Tensor | None = None,
    validate_identity: bool = True,
) -> Tensor:
    """Literal dense-cover contraction for arithmetic and gradient references."""
    if isinstance(anchor_values, PreparedAnchorValues):
        values = anchor_values.values
        anchor_ids = anchor_values.anchor_ids
    else:
        values = anchor_values
    if values.ndim != 4 or values.shape[0] != cover.batch_size:
        raise ValueError("anchor_values must have shape [B,H,E,D] and match the cover batch.")
    batch, _heads, anchors, _width = values.shape
    _validate_cover_for_gather(cover, anchors, anchor_ids, validate_identity=validate_identity)
    effective_weights = cover.weights * cover.validity.to(cover.weights.dtype)
    dense_cover = values.new_zeros(batch, cover.query_count, anchors).scatter_add(
        dim=-1,
        index=cover.anchor_indices.to(torch.long),
        src=effective_weights.to(values.dtype),
    )
    return torch.einsum("bqe,behd->bqhd", dense_cover, values.permute(0, 2, 1, 3))


def prepare_source_indexed_anchor_response(
    atlas: TypedAssociationAtlas,
    carriers: Tensor,
    controls: Tensor,
    *,
    executor: Literal["dense", "selected"] = "dense",
) -> PreparedSourceIndexedResponse:
    """Contract ``(P / m) * carrier * control`` at scene anchors.

    ``carriers`` has shape ``[B,H,N,R]`` and ``controls`` is ``[B,N]``. The
    measure division is exact on valid donors and returns zero for invalid
    donors; no query-by-source coefficient matrix is constructed.
    """
    if atlas.kind != "module":
        raise ValueError("Source-indexed control transport requires a module atlas.")
    batch, heads, _anchors, donors = atlas.probabilities.shape
    if carriers.ndim != 4 or carriers.shape[:3] != (batch, heads, donors):
        raise ValueError("carriers must have shape [B,H,M,R] matching the module atlas.")
    if controls.shape != (batch, donors):
        raise ValueError("controls must have shape [B,M] matching the module atlas.")
    measure = atlas.normalized_measure
    safe_measure = torch.where(atlas.donor_valid, measure, torch.ones_like(measure))
    density = atlas.probabilities / safe_measure[:, None, None, :]
    density = density * atlas.donor_valid[:, None, None, :].to(density.dtype)
    source_coefficients = density * controls.to(density.dtype)[:, None, None, :]
    anchor_response, executed_counts, candidate_counts = _contract_atlas_edges(
        source_coefficients,
        carriers,
        positive_support=atlas.probabilities > 0,
        candidate_support=atlas.donor_valid[:, None, None, :],
        destination_valid=atlas.destination_valid,
        executor=executor,
    )
    return PreparedSourceIndexedResponse(
        values=anchor_response,
        anchor_ids=atlas.anchor_ids,
        executed_edge_counts=executed_counts,
        candidate_edge_counts=candidate_counts,
    )


def gather_source_response_indexed(
    anchor_response: PreparedSourceIndexedResponse | Tensor,
    cover: ReceiverCover,
    phi: Tensor,
    *,
    anchor_ids: Tensor | None = None,
    validate_identity: bool = True,
) -> Tensor:
    """Apply heat-independent query factors to covered anchor responses."""
    if isinstance(anchor_response, PreparedSourceIndexedResponse):
        values = anchor_response.values
        anchor_ids = anchor_response.anchor_ids
    else:
        values = anchor_response
    if values.ndim != 4 or values.shape[0] != cover.batch_size:
        raise ValueError("anchor_response must have shape [B,H,E,R].")
    batch, heads, anchors, rank = values.shape
    if phi.shape != (batch, cover.query_count, heads, rank):
        raise ValueError("phi must have shape [B,Q,H,R] matching the response and cover.")
    _validate_cover_for_gather(cover, anchors, anchor_ids, validate_identity=validate_identity)
    selected = _gather_cover_values(values, cover)
    weights = cover.weights[:, :, :, None, None]
    weights = weights * cover.validity[:, :, :, None, None].to(weights.dtype)
    query_response = (selected * weights).sum(dim=2)
    return (query_response * phi).sum(dim=(-1, -2))
