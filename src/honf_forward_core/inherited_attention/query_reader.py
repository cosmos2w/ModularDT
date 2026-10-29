"""Parameter-free IAG-Q: inherited support and batched cell-token attention.

Cells are signatures of the geometry cover's anchor offsets and validity,
never clusters fitted to receivers or fields. One geometry-only grouping is
priced per request. Candidate unions are compacted once per cell, across
heads; all queries in that cell then share cached K/V matrix operands.
There is no query/donor/128-wide pair-message tensor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .atlas import validate_cover_anchor_identity
from .contracts import ReceiverCover, TypedAssociationAtlas, TypedTokenValues
from .normalization import normalize_donor_measure


def _stamp(tensors: tuple[Tensor, ...]) -> tuple:
    return tuple((id(t), t.data_ptr(), int(t._version)) for t in tensors)


@dataclass(frozen=True)
class CellQueryPlan:
    """One request's geometry grouping, with no learned or target inputs."""

    cover_owner: int
    group_batch: Tensor
    group_anchors: Tensor
    group_anchor_validity: Tensor
    query_indices: Tensor
    query_validity: Tensor
    inverse: Tensor
    within_group: Tensor
    counts: Tensor
    versions: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "versions", _stamp(self.tensors()))

    def tensors(self) -> tuple[Tensor, ...]:
        return (self.group_batch, self.group_anchors, self.group_anchor_validity,
                self.query_indices, self.query_validity, self.inverse,
                self.within_group, self.counts)

    def validate(self, cover: ReceiverCover) -> None:
        cover.validate_unchanged()
        if id(cover) != self.cover_owner or self.versions != _stamp(self.tensors()):
            raise ValueError("Cell query plan or its owning receiver cover changed.")


def build_cell_query_plan(cover: ReceiverCover) -> CellQueryPlan:
    """Group whole anchor signatures once; never sort query donor scores."""
    cover.validate_unchanged()
    batch, queries, slots = cover.anchor_indices.shape
    if queries < 1:
        raise ValueError("A cell query plan requires at least one receiver.")
    device = cover.anchor_indices.device
    batch_ids = torch.arange(batch, device=device)[:, None].expand(batch, queries)
    signatures = torch.cat((batch_ids.reshape(-1, 1),
                            cover.anchor_indices.reshape(-1, slots),
                            cover.validity.to(torch.long).reshape(-1, slots)), dim=1)
    cells, inverse, counts = torch.unique(signatures, dim=0, sorted=True,
                                         return_inverse=True, return_counts=True)
    order = torch.argsort(inverse, stable=True)
    starts = counts.cumsum(0) - counts
    positions = torch.arange(order.numel(), device=device) - torch.repeat_interleave(starts, counts)
    maximum_queries = int(counts.max())
    packed = torch.zeros((cells.shape[0], maximum_queries), dtype=torch.long, device=device)
    packed[inverse[order], positions] = order
    within_group = torch.empty_like(order)
    within_group[order] = positions
    valid = torch.arange(maximum_queries, device=device)[None, :] < counts[:, None]
    if cover.query_mask is not None:
        valid = valid & cover.query_mask.reshape(-1)[packed]
    return CellQueryPlan(id(cover), cells[:, 0], cells[:, 1:slots + 1],
                         cells[:, slots + 1:].to(torch.bool), packed, valid,
                         inverse, within_group, counts)


def _weights(beta: Tensor, scores: Tensor, allowed: Tensor) -> Tensor:
    """Stable beta*exp(score), with exact zeros and explicit empty rows."""
    positive = (beta > 0) & allowed
    safe_beta = torch.where(positive, beta, torch.ones_like(beta))
    logits = scores + torch.log(safe_beta)
    has_donor = positive.any(-1, keepdim=True)
    masked = logits.masked_fill(~positive, -torch.inf)
    masked = torch.where(has_donor, masked, torch.zeros_like(masked))
    return torch.softmax(masked, dim=-1) * positive.to(beta.dtype)


def _check(atlas: TypedAssociationAtlas, values: TypedTokenValues, q: Tensor,
           keys: Tensor, cover: ReceiverCover) -> None:
    cover.validate_unchanged()
    batch, heads, _anchors, donors = atlas.probabilities.shape
    if (atlas.kind != values.kind or not torch.equal(atlas.donor_ids, values.token_ids)
            or q.shape[:3] != (batch, heads, cover.query_count)
            or keys.shape != values.values.shape or keys.shape[:3] != (batch, heads, donors)
            or q.shape[-1] != keys.shape[-1]):
        raise ValueError("IAG-Q query/key/value/atlas typed axes or IDs differ.")
    if any(t.device != q.device for t in (keys, values.values, atlas.probabilities, cover.weights)):
        raise ValueError("IAG-Q factors must share one model device.")
    if any(t.dtype != q.dtype for t in (keys, values.values, atlas.probabilities, cover.weights)):
        raise ValueError("IAG-Q query/key/value/atlas arithmetic dtypes differ.")
    if not torch.equal(atlas.donor_valid, values.valid & (values.measure > 0)):
        raise ValueError("IAG-Q atlas and final typed values disagree on donor validity.")
    if not torch.equal(atlas.normalized_measure, normalize_donor_measure(values.measure, values.valid)):
        raise ValueError("IAG-Q atlas and final typed values have different positive measures.")
    validate_cover_anchor_identity(cover, atlas.anchor_ids)


def query_conditioned_gather(
    atlas: TypedAssociationAtlas,
    values: TypedTokenValues,
    query_scaled: Tensor,
    key_normalized: Tensor,
    cover: ReceiverCover,
    *,
    executor: str,
    plan: CellQueryPlan | None = None,
    return_trace: bool = False,
) -> tuple[Tensor, dict[str, int], dict[str, Any] | None]:
    """Return [B,Q,H,D] contexts using the same beta-weighted query law.

    ``dense_reference`` is a literal scalar query/donor attention control,
    with cached typed K/V. ``cell_selected`` gathers K/V once per structural
    cell union and performs batched matrix products, including reported
    padding. Neither executor creates nonlinear pair values.
    """
    _check(atlas, values, query_scaled, key_normalized, cover)
    batch, heads, queries, width = query_scaled.shape
    donors = key_normalized.shape[2]
    trace = None
    if executor not in ("cell_selected", "dense_reference"):
        raise ValueError("Unknown IAG-Q query executor.")
    if not donors or not queries:
        zero = (query_scaled.sum() + key_normalized.sum() + values.values.sum()
                + atlas.probabilities.sum()) * 0.0
        return query_scaled.new_zeros(batch, queries, heads, width) + zero, {
            "query_candidate_dot_products": 0, "query_positive_associations": 0,
            "query_padded_candidate_dot_products": 0, "query_cell_key_value_rows": 0,
            "query_cell_count": 0, "query_max_candidate_width": 0,
            "query_max_cell_queries": 0, "query_atlas_scalar_reads": 0,
        }, None
    trace_queries = min(16, queries)
    if executor == "dense_reference":
        b = torch.arange(batch, device=query_scaled.device)[:, None, None]
        rows = atlas.probabilities.permute(0, 2, 1, 3)[b, cover.anchor_indices]
        weights = cover.weights * cover.validity.to(cover.weights.dtype)
        beta = (rows * weights[..., None, None]).sum(2).permute(0, 2, 1, 3)
        scores = torch.matmul(query_scaled, key_normalized.transpose(-1, -2))
        allowed = atlas.donor_valid[:, None, None, :].expand_as(beta)
        if cover.query_mask is not None:
            allowed = allowed & cover.query_mask[:, None, :, None]
        probabilities = _weights(beta, scores, allowed)
        result = torch.matmul(probabilities, values.values).transpose(1, 2)
        count = batch * heads * queries * donors
        work = {"query_candidate_dot_products": count,
                "query_positive_associations": int((probabilities > 0).sum()),
                "query_padded_candidate_dot_products": count,
                "query_cell_key_value_rows": 2 * batch * heads * donors,
                "query_cell_count": batch, "query_max_candidate_width": donors,
                "query_max_cell_queries": queries,
                "query_atlas_scalar_reads": batch * queries * cover.slots * heads * donors}
        if return_trace:
            trace = {"query_positions": torch.arange(trace_queries, device=scores.device),
                     "candidate_indices": torch.arange(donors, device=scores.device)[None, None].expand(batch, trace_queries, -1),
                     "candidate_validity": atlas.donor_valid[:, None].expand(-1, trace_queries, -1),
                     "probabilities": probabilities[:, :, :trace_queries].transpose(1, 2),
                     "beta": beta[:, :, :trace_queries].transpose(1, 2)}
        return result, work, trace

    if plan is None:
        plan = build_cell_query_plan(cover)
    plan.validate(cover)
    cells, packed_queries = plan.query_indices.shape
    # Only cell/anchor/donor scalar rows are gathered here. K/V are gathered
    # once per cell below, without an additional receiver axis.
    cell_p = atlas.probabilities.permute(0, 2, 1, 3)[
        plan.group_batch[:, None], plan.group_anchors].permute(0, 2, 1, 3)
    cell_p = cell_p * plan.group_anchor_validity[:, None, :, None].to(cell_p.dtype)
    union = (cell_p > 0).any(dim=(1, 2))
    counts = union.sum(-1)
    maximum = int(counts.max())
    # An all-empty typed atlas is legal (e.g. empty modules).
    if maximum == 0:
        zero = (query_scaled.sum() + key_normalized.sum() + values.values.sum()
                + atlas.probabilities.sum()) * 0.0
        return query_scaled.new_zeros(batch, queries, heads, width) + zero, {
            "query_candidate_dot_products": 0, "query_positive_associations": 0,
            "query_padded_candidate_dot_products": 0, "query_cell_key_value_rows": 0,
            "query_cell_count": cells, "query_max_candidate_width": 0,
            "query_max_cell_queries": packed_queries, "query_atlas_scalar_reads": 0,
        }, None
    cell_index, donor_index = torch.nonzero(union, as_tuple=True)
    ranks = union.to(torch.long).cumsum(-1) - 1
    candidates = torch.zeros(cells, maximum, dtype=torch.long, device=query_scaled.device)
    candidates[cell_index, ranks[cell_index, donor_index]] = donor_index
    candidate_valid = torch.arange(maximum, device=query_scaled.device)[None, :] < counts[:, None]
    p = cell_p.gather(-1, candidates[:, None, None, :].expand(-1, heads, cover.slots, -1))
    p = p * candidate_valid[:, None, None, :].to(p.dtype)
    flat_q = query_scaled.transpose(1, 2).reshape(batch * queries, heads, width)
    q = flat_q[plan.query_indices].permute(0, 2, 1, 3)
    k = key_normalized.permute(0, 2, 1, 3)[plan.group_batch[:, None], candidates].permute(0, 2, 1, 3)
    v = values.values.permute(0, 2, 1, 3)[plan.group_batch[:, None], candidates].permute(0, 2, 1, 3)
    lambdas = (cover.weights * cover.validity.to(cover.weights.dtype)).reshape(-1, cover.slots)[plan.query_indices]
    beta = torch.einsum("gqs,ghsn->ghqn", lambdas, p)
    scores = torch.matmul(q, k.transpose(-1, -2))
    allowed = candidate_valid[:, None, None, :] & plan.query_validity[:, None, :, None]
    probabilities = _weights(beta, scores, allowed)
    contexts = torch.matmul(probabilities, v)
    # Restore exact caller order; padded slots never become physical queries.
    result = contexts[plan.inverse, :, plan.within_group].reshape(batch, queries, heads, width)
    logical = int((counts * plan.counts).sum()) * heads
    work = {"query_candidate_dot_products": logical,
            "query_positive_associations": int((probabilities > 0).sum()),
            "query_padded_candidate_dot_products": cells * heads * packed_queries * maximum,
            "query_cell_key_value_rows": 2 * cells * heads * maximum,
            "query_cell_count": cells, "query_max_candidate_width": maximum,
            "query_max_cell_queries": packed_queries,
            "query_atlas_scalar_reads": cells * heads * cover.slots * maximum}
    if return_trace:
        ids = (torch.arange(batch, device=query_scaled.device)[:, None] * queries
               + torch.arange(trace_queries, device=query_scaled.device)[None, :]).reshape(-1)
        groups, positions = plan.inverse[ids], plan.within_group[ids]
        trace = {"query_positions": torch.arange(trace_queries, device=query_scaled.device),
                 "candidate_indices": candidates[groups].reshape(batch, trace_queries, maximum),
                 "candidate_validity": candidate_valid[groups].reshape(batch, trace_queries, maximum),
                 "probabilities": probabilities[groups, :, positions].reshape(batch, trace_queries, heads, maximum),
                 "beta": beta[groups, :, positions].reshape(batch, trace_queries, heads, maximum)}
    return result, work, trace
