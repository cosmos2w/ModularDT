"""Exact support-union block metadata for sparse-incidence readers.

The support metadata in this module is deliberately small and observed-only.
Positive source and query memberships are packed into integer words, then only
the query signatures present in the current receiver chunk are visited.  No
``2**K`` lookup table, group-expanded pair list, or live numeric tensor is
created here.  The returned indices are integer metadata; callers must still
compute memberships, overlaps, controls, and values from the original live
tensors inside each block.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

_MAX_MASK_BITS = 63


def pack_positive_support(values: torch.Tensor) -> torch.Tensor:
    """Pack positive incidence into one detached integer word per row.

    ``torch.long`` is signed, so bit 62 is the highest portable positive bit.
    Run 1501 uses K=12.  The bound keeps this helper safe for small synthetic
    mixed-width tests without allocating an exponential table.
    """

    if not isinstance(values, torch.Tensor) or values.ndim != 3:
        raise ValueError("incidence must have shape [B,S,K].")
    if not (values.is_floating_point() or values.dtype == torch.bool):
        raise TypeError("incidence must be floating point or boolean.")
    if values.is_floating_point():
        if not bool(torch.isfinite(values).all()):
            raise ValueError("incidence must contain only finite values.")
        if bool((values < 0).any()):
            raise ValueError("incidence must be nonnegative.")
    group_count = int(values.shape[-1])
    if group_count <= 0 or group_count > _MAX_MASK_BITS:
        raise ValueError(
            f"incidence group width must be in [1,{_MAX_MASK_BITS}], got {group_count}."
        )
    shifts = torch.arange(group_count, device=values.device, dtype=torch.long)
    powers = torch.ones_like(shifts).bitwise_left_shift(shifts)
    # Positive support is discrete dispatch metadata by construction.  The
    # integer result cannot retain an autograd connection to live assignments.
    return (values.gt(0).to(dtype=torch.long) * powers).sum(dim=-1).detach()


@dataclass(frozen=True)
class SupportBlock:
    """One disjoint query-signature block and its unique source union."""

    batch_index: int
    query_signature: int
    query_index: torch.Tensor
    source_index: torch.Tensor

    @property
    def query_count(self) -> int:
        return int(self.query_index.numel())

    @property
    def source_count(self) -> int:
        return int(self.source_index.numel())

    @property
    def executed_rows(self) -> int:
        return self.query_count * self.source_count


def observed_support_blocks(
    query_assignment: torch.Tensor,
    source_membership: torch.Tensor,
    *,
    source_measure: torch.Tensor | None = None,
) -> Iterator[SupportBlock]:
    """Yield blocks for only the query signatures observed in this chunk.

    Query rows with one signature are grouped in stable original order.  The
    source union is computed by direct bit intersection and each source slot
    appears at most once in a block.  A zero-signature query yields an empty
    block, allowing the caller to preserve the exact zero-support boundary
    without evaluating a synthetic epsilon path.
    """

    if not isinstance(query_assignment, torch.Tensor) or query_assignment.ndim != 3:
        raise ValueError("query_assignment must have shape [B,Q,K].")
    if not isinstance(source_membership, torch.Tensor) or source_membership.ndim != 3:
        raise ValueError("source_membership must have shape [B,S,K].")
    if tuple(query_assignment.shape[:1]) != tuple(source_membership.shape[:1]):
        raise ValueError("query_assignment and source_membership must share batch size.")
    if int(query_assignment.shape[-1]) != int(source_membership.shape[-1]):
        raise ValueError("query_assignment and source_membership must share group width.")
    if query_assignment.device != source_membership.device:
        raise ValueError("query_assignment and source_membership must share device.")
    if source_measure is not None:
        if tuple(source_measure.shape) != tuple(source_membership.shape[:2]):
            raise ValueError("source_measure must align with source_membership as [B,S].")
        if source_measure.device != source_membership.device:
            raise ValueError("source_measure must share source_membership device.")

    query_masks = pack_positive_support(query_assignment)
    source_masks = pack_positive_support(source_membership)
    source_valid = (
        torch.ones_like(source_masks, dtype=torch.bool)
        if source_measure is None
        else source_measure > 0
    )
    for batch_index in range(int(query_masks.shape[0])):
        masks = query_masks[batch_index]
        # unique() enumerates observed values only; it never creates a mask
        # table proportional to the registered group capacity.
        for signature in torch.unique(masks, sorted=True).tolist():
            signature = int(signature)
            query_index = torch.nonzero(masks == signature, as_tuple=False).reshape(-1)
            reachable = (
                source_masks[batch_index].bitwise_and(signature).ne(0)
                & source_valid[batch_index]
            )
            source_index = torch.nonzero(reachable, as_tuple=False).reshape(-1)
            yield SupportBlock(
                batch_index=batch_index,
                query_signature=signature,
                query_index=query_index,
                source_index=source_index,
            )


__all__ = ["SupportBlock", "observed_support_blocks", "pack_positive_support"]
