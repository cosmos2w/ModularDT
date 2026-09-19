"""Exact six-bit support metadata for phase-shared group control.

Run 1407 keeps the positive source/query incidence arithmetic from Run 1406,
but the executor needs the union of sources touched by a query rather than a
list of ``(query, group, source)`` paths.  With the formal ``K=6`` profile,
the positive group incidence of a source and a query fit in one six-bit word::

    source_mask[s] = sum(2**k for k when A[s, k] > 0)
    query_mask[q]  = sum(2**k for k when alpha[q, k] > 0)

Their source support is exactly ``source_mask[s] & query_mask[q] != 0``.  This
module builds the 64-entry lookup table once for a prepared physical state.
The table is integer/bool metadata only.  It never replaces live memberships,
query assignments, source values, or control moments.

The helper intentionally does not generalize the table to arbitrary ``K``.
The formal phase-shared profile is K=6, and an accidental ``2**K`` allocation
for a large experimental K would be an unsafe change in executor behavior.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

GROUP_COUNT = 6
QUERY_MASK_COUNT = 1 << GROUP_COUNT


def _validate_incidence(values: torch.Tensor, name: str) -> tuple[int, int, int]:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape [B,S,6] or [B,Q,6].")
    if int(values.shape[-1]) != GROUP_COUNT:
        raise ValueError(f"{name} must have exactly K={GROUP_COUNT} group columns; got {int(values.shape[-1])}.")
    if not (values.is_floating_point() or values.dtype == torch.bool):
        raise TypeError(f"{name} must be floating point or boolean.")
    # Source/query assignments are entmax outputs and are nonnegative by
    # contract.  Refusing negative values prevents a signed value from being
    # silently interpreted as absent by the bit metadata.
    if values.is_floating_point():
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"{name} must contain only finite values.")
        if bool((values < 0).any()):
            raise ValueError(f"{name} must be nonnegative.")
    return int(values.shape[0]), int(values.shape[1]), int(values.shape[2])


def _as_valid_source_mask(
    source_valid: torch.Tensor | None,
    *,
    source_measure: torch.Tensor | None,
    batch: int,
    source_count: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Normalize optional active-source and positive-measure masks."""

    if source_valid is None:
        valid = torch.ones(
            (batch, source_count),
            device=reference.device,
            dtype=torch.bool,
        )
    else:
        if not isinstance(source_valid, torch.Tensor):
            raise TypeError("source_valid must be a torch.Tensor when provided.")
        if tuple(source_valid.shape) != (batch, source_count):
            raise ValueError(f"source_valid must have shape [{batch},{source_count}], got {tuple(source_valid.shape)}.")
        if source_valid.device != reference.device:
            raise ValueError("source_valid must be on the same device as source_membership.")
        if source_valid.dtype == torch.bool:
            valid = source_valid
        elif source_valid.is_floating_point() or source_valid.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            if not bool(torch.isfinite(source_valid).all()):
                raise ValueError("source_valid must contain only finite values.")
            valid = source_valid > 0
        else:
            raise TypeError("source_valid must be boolean or floating point.")

    if source_measure is not None:
        if not isinstance(source_measure, torch.Tensor):
            raise TypeError("source_measure must be a torch.Tensor when provided.")
        if tuple(source_measure.shape) != (batch, source_count):
            raise ValueError(
                f"source_measure must have shape [{batch},{source_count}], got {tuple(source_measure.shape)}."
            )
        if source_measure.device != reference.device:
            raise ValueError("source_measure must be on the same device as source_membership.")
        if not (
            source_measure.is_floating_point()
            or source_measure.dtype == torch.bool
            or source_measure.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            raise TypeError("source_measure must be boolean or floating point.")
        if source_measure.is_floating_point() or source_measure.dtype != torch.bool:
            if not bool(torch.isfinite(source_measure).all()):
                raise ValueError("source_measure must contain only finite values.")
            valid = valid & (source_measure > 0)
        else:
            valid = valid & source_measure
    return valid


def six_bit_mask(
    incidence: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack positive K=6 incidence into a ``torch.long`` six-bit word.

    ``valid`` is useful for padded module sources.  It masks complete source
    rows before packing; query masks normally call this function without a
    validity mask.  The returned metadata is integer-valued and has no
    autograd connection by construction.
    """

    batch, width, _ = _validate_incidence(incidence, "incidence")
    if valid is not None:
        if not isinstance(valid, torch.Tensor):
            raise TypeError("valid must be a torch.Tensor when provided.")
        if tuple(valid.shape) != (batch, width):
            raise ValueError(f"valid must have shape [{batch},{width}].")
        if valid.device != incidence.device:
            raise ValueError("valid must be on the same device as incidence.")
        if valid.dtype == torch.bool:
            valid_rows = valid
        elif valid.is_floating_point() or valid.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            if not bool(torch.isfinite(valid).all()):
                raise ValueError("valid must contain only finite values.")
            valid_rows = valid > 0
        else:
            raise TypeError("valid must be boolean or floating point.")
    else:
        valid_rows = None

    positive = incidence > 0
    if valid_rows is not None:
        positive = positive & valid_rows[..., None]
    shifts = torch.arange(
        GROUP_COUNT,
        device=incidence.device,
        dtype=torch.long,
    )
    powers = torch.ones_like(shifts).bitwise_left_shift(shifts)
    return (positive.to(torch.long) * powers).sum(dim=-1)


# The descriptive alias is convenient at call sites that work with both
# source and query words.
pack_six_bit_mask = six_bit_mask


@dataclass(frozen=True)
class QuerySignatureSelection:
    """One query-signature group and its unique source selection.

    ``query_index`` preserves the original query order within this signature;
    ``source_index`` is sorted by the original padded source slot.  The two
    vectors describe a rectangular block only.  No group axis or q-k-s path
    list is materialized.
    """

    batch_index: int
    support_mask: int
    query_index: torch.Tensor
    source_index: torch.Tensor

    @property
    def query_count(self) -> int:
        return int(self.query_index.numel())

    @property
    def source_count(self) -> int:
        return int(self.source_index.numel())

    @property
    def support_pair_count(self) -> int:
        return self.query_count * self.source_count


@dataclass(frozen=True)
class SupportExecutionCounts:
    """Separate semantic support counts from measured executor row counts.

    All fields are per-query tensors.  ``support_pairs`` is metadata-derived;
    ``executed_rows`` and ``padded_rows`` must be supplied by the actual
    selected runtime policy.  Keeping them separate prevents a diagnostic or
    gathered counter from being mislabeled as rectangular execution work.
    """

    logical_paths: torch.Tensor
    support_pairs: torch.Tensor
    executed_rows: torch.Tensor
    padded_rows: torch.Tensor
    recomputed_rows: torch.Tensor

    @property
    def logical_path_count(self) -> torch.Tensor:
        return self.logical_paths

    @property
    def support_pair_count(self) -> torch.Tensor:
        return self.support_pairs

    @property
    def executed_row_count(self) -> torch.Tensor:
        return self.executed_rows

    @property
    def padded_row_count(self) -> torch.Tensor:
        return self.padded_rows


@dataclass(frozen=True)
class SixBitSupportIndex:
    """Prepared exact support metadata for one source type and query batch.

    ``source_masks`` is the raw positive-incidence word from ``A``.  The
    separate ``source_valid`` mask is applied to ``support_table`` so padded
    rows cannot execute while logical path counts remain faithful to the
    source-assignment arithmetic.
    """

    source_masks: torch.Tensor
    query_masks: torch.Tensor
    source_valid: torch.Tensor
    support_table: torch.Tensor
    support_count_table: torch.Tensor
    logical_path_count_table: torch.Tensor
    signature_query_index: torch.Tensor
    signature_query_ptr: torch.Tensor
    signature_query_count: torch.Tensor

    def __post_init__(self) -> None:
        if self.source_masks.ndim != 2 or self.source_masks.dtype != torch.long:
            raise ValueError("source_masks must be a [B,S] torch.long tensor.")
        if self.query_masks.ndim != 2 or self.query_masks.dtype != torch.long:
            raise ValueError("query_masks must be a [B,Q] torch.long tensor.")
        batch, source_count = (int(value) for value in self.source_masks.shape)
        if int(self.query_masks.shape[0]) != batch:
            raise ValueError("source_masks and query_masks must share their batch dimension.")
        if self.source_valid.shape != self.source_masks.shape or self.source_valid.dtype != torch.bool:
            raise ValueError("source_valid must be a boolean tensor aligned with source_masks.")
        if self.support_table.shape != (batch, QUERY_MASK_COUNT, source_count):
            raise ValueError("support_table must have shape [B,64,S].")
        if self.support_table.dtype != torch.bool:
            raise ValueError("support_table must be boolean metadata.")
        if self.support_count_table.shape != (batch, QUERY_MASK_COUNT):
            raise ValueError("support_count_table must have shape [B,64].")
        if self.logical_path_count_table.shape != (batch, QUERY_MASK_COUNT):
            raise ValueError("logical_path_count_table must have shape [B,64].")
        if self.signature_query_index.shape != self.query_masks.shape:
            raise ValueError("signature_query_index must have shape [B,Q].")
        if self.signature_query_index.dtype != torch.long:
            raise ValueError("signature_query_index must be torch.long metadata.")
        if self.signature_query_ptr.shape != (batch, QUERY_MASK_COUNT + 1):
            raise ValueError("signature_query_ptr must have shape [B,65].")
        if self.signature_query_count.shape != (batch, QUERY_MASK_COUNT):
            raise ValueError("signature_query_count must have shape [B,64].")
        tensors = (
            self.query_masks,
            self.source_valid,
            self.support_table,
            self.support_count_table,
            self.logical_path_count_table,
            self.signature_query_index,
            self.signature_query_ptr,
            self.signature_query_count,
        )
        if any(value.device != self.source_masks.device for value in tensors):
            raise ValueError("All support metadata tensors must share one device.")

    @property
    def source_support_table(self) -> torch.Tensor:
        """Descriptive alias for the [B,64,S] support table."""

        return self.support_table

    @property
    def active_source_masks(self) -> torch.Tensor:
        """Return source masks after valid/padded-source filtering."""

        return torch.where(self.source_valid, self.source_masks, torch.zeros_like(self.source_masks))

    @property
    def support_mask_table(self) -> torch.Tensor:
        return self.support_table

    @property
    def support_pairs_per_query(self) -> torch.Tensor:
        return self.support_count_table.gather(1, self.query_masks)

    @property
    def logical_paths_per_query(self) -> torch.Tensor:
        return self.logical_path_count_table.gather(1, self.query_masks)

    @property
    def support_pair_count(self) -> torch.Tensor:
        return self.support_pairs_per_query.sum()

    @property
    def logical_path_count(self) -> torch.Tensor:
        return self.logical_paths_per_query.sum()

    @property
    def query_count(self) -> int:
        return int(self.query_masks.shape[1])

    @property
    def source_count(self) -> int:
        return int(self.source_masks.shape[1])

    @property
    def batch_count(self) -> int:
        return int(self.source_masks.shape[0])

    def query_support(self, batch_index: int, query_index: int) -> torch.Tensor:
        """Return one query's boolean source support without live weights."""

        batch = int(batch_index)
        query = int(query_index)
        return self.support_table[batch, self.query_masks[batch, query]]

    def source_indices_for_mask(self, batch_index: int, support_mask: int) -> torch.Tensor:
        """Return sorted padded source slots for one of the 64 masks."""

        batch = int(batch_index)
        mask = int(support_mask)
        if not 0 <= mask < QUERY_MASK_COUNT:
            raise ValueError(f"support_mask must be in [0,{QUERY_MASK_COUNT - 1}].")
        return torch.nonzero(self.support_table[batch, mask], as_tuple=False).reshape(-1)

    def query_indices_for_mask(self, batch_index: int, support_mask: int) -> torch.Tensor:
        """Return original query slots in stable order for one signature."""

        batch = int(batch_index)
        mask = int(support_mask)
        if not 0 <= mask < QUERY_MASK_COUNT:
            raise ValueError(f"support_mask must be in [0,{QUERY_MASK_COUNT - 1}].")
        start = int(self.signature_query_ptr[batch, mask])
        stop = int(self.signature_query_ptr[batch, mask + 1])
        return self.signature_query_index[batch, start:stop]

    def signature_selection(
        self,
        batch_index: int,
        support_mask: int,
    ) -> QuerySignatureSelection:
        """Return query/source metadata for one signature, including empties."""

        mask = int(support_mask)
        return QuerySignatureSelection(
            batch_index=int(batch_index),
            support_mask=mask,
            query_index=self.query_indices_for_mask(batch_index, mask),
            source_index=self.source_indices_for_mask(batch_index, mask),
        )

    def iter_signature_selections(
        self,
        *,
        include_empty: bool = False,
    ) -> Iterator[QuerySignatureSelection]:
        """Yield query-signature/source blocks in deterministic mask order."""

        for batch in range(self.batch_count):
            for mask in range(QUERY_MASK_COUNT):
                selection = self.signature_selection(batch, mask)
                if include_empty or selection.query_count > 0:
                    yield selection

    def selected_pair_indices(self, query_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return unique ``[batch,query,source]`` pairs, never q-k-s paths.

        This compact reference helper is useful for numerical tests and small
        selected-source readers.  The production reader can instead consume
        :meth:`iter_signature_selections` and evaluate one rectangular block
        per signature without constructing this pair list.
        """

        if query_mask is None:
            selected = torch.ones_like(self.query_masks, dtype=torch.bool)
        else:
            if tuple(query_mask.shape) != tuple(self.query_masks.shape):
                raise ValueError("query_mask must align with query_masks as [B,Q].")
            if query_mask.device != self.query_masks.device or query_mask.dtype != torch.bool:
                raise ValueError("query_mask must be boolean and on the support-index device.")
            selected = query_mask
        gathered = self.support_table.gather(
            1,
            self.query_masks[..., None].expand(-1, -1, self.source_count),
        )
        gathered = gathered & selected[..., None]
        pairs = torch.nonzero(gathered, as_tuple=False)
        if pairs.numel() == 0:
            return torch.empty((0, 3), device=self.source_masks.device, dtype=torch.long)
        return pairs

    def execution_counts(
        self,
        executed_rows: torch.Tensor,
        padded_rows: torch.Tensor,
        *,
        recomputed_rows: torch.Tensor | None = None,
    ) -> SupportExecutionCounts:
        """Attach measured runtime row counts without overwriting support counts."""

        expected_shape = tuple(self.query_masks.shape)

        def _rows(value: torch.Tensor, name: str) -> torch.Tensor:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor with shape {expected_shape}.")
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}.")
            if value.device != self.source_masks.device:
                raise ValueError(f"{name} must share the support-index device.")
            if not (value.is_floating_point() or value.dtype in (torch.int32, torch.int64)):
                raise TypeError(f"{name} must be numeric.")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values.")
            if bool((value < 0).any()):
                raise ValueError(f"{name} must be nonnegative.")
            return value

        executed = _rows(executed_rows, "executed_rows")
        padded = _rows(padded_rows, "padded_rows")
        recomputed = (
            torch.zeros_like(executed) if recomputed_rows is None else _rows(recomputed_rows, "recomputed_rows")
        )
        return SupportExecutionCounts(
            logical_paths=self.logical_paths_per_query,
            support_pairs=self.support_pairs_per_query,
            executed_rows=executed,
            padded_rows=padded,
            recomputed_rows=recomputed,
        )


def _query_signature_metadata(query_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build stable [B,Q] query order and [B,65] mask pointers."""

    batch, query_count = (int(value) for value in query_masks.shape)
    order = torch.argsort(query_masks, dim=-1, stable=True)
    counts = torch.zeros(
        (batch, QUERY_MASK_COUNT),
        device=query_masks.device,
        dtype=torch.long,
    )
    counts.scatter_add_(
        1,
        query_masks,
        torch.ones((batch, query_count), device=query_masks.device, dtype=torch.long),
    )
    pointers = torch.cat(
        [
            torch.zeros((batch, 1), device=query_masks.device, dtype=torch.long),
            counts.cumsum(dim=-1),
        ],
        dim=-1,
    )
    return order, pointers, counts


def build_six_bit_support_index(
    source_membership: torch.Tensor,
    query_assignment: torch.Tensor,
    *,
    source_valid: torch.Tensor | None = None,
    source_measure: torch.Tensor | None = None,
) -> SixBitSupportIndex:
    """Build exact K=6 source support and query-signature metadata.

    Parameters are live tensors, but only positive-incidence comparisons enter
    the returned integer metadata.  ``source_valid`` and positive
    ``source_measure`` mask padded/inactive source slots.  The original live
    tensors are deliberately not stored in the result, so the index cannot
    accidentally become a detached replacement for ``rho`` or moments.
    """

    source_batch, source_count, _ = _validate_incidence(source_membership, "source_membership")
    query_batch, _, _ = _validate_incidence(query_assignment, "query_assignment")
    if source_batch != query_batch:
        raise ValueError("source_membership and query_assignment must share their batch dimension.")
    if source_membership.device != query_assignment.device:
        raise ValueError("source_membership and query_assignment must share one device.")
    valid = _as_valid_source_mask(
        source_valid,
        source_measure=source_measure,
        batch=source_batch,
        source_count=source_count,
        reference=source_membership,
    )
    # ``source_masks`` is the direct m_s definition.  Validity is retained as
    # a separate mask and applied only when constructing executable support;
    # this keeps logical-incidence metadata faithful to positive A entries.
    source_masks = six_bit_mask(source_membership)
    query_masks = six_bit_mask(query_assignment)
    all_masks = torch.arange(
        QUERY_MASK_COUNT,
        device=source_membership.device,
        dtype=torch.long,
    )
    support_table = (source_masks[:, None, :].bitwise_and(all_masks[None, :, None]) != 0) & valid[:, None, :]
    support_count_table = support_table.sum(dim=-1, dtype=torch.long)

    # Logical paths count positive query/group incidences separately from the
    # unique source support.  It is metadata derived from the same source and
    # query masks, not a row count from either executor.
    powers = torch.ones((GROUP_COUNT,), device=source_membership.device, dtype=torch.long)
    powers = powers.bitwise_left_shift(torch.arange(GROUP_COUNT, device=powers.device))
    source_group_incidence = ((source_masks[..., None].bitwise_and(powers[None, None, :])) != 0).sum(
        dim=1, dtype=torch.long
    )
    query_group_incidence = all_masks[:, None].bitwise_and(powers[None, :]) != 0
    logical_path_count_table = torch.einsum(
        "mk,bk->bm",
        query_group_incidence.to(torch.long),
        source_group_incidence,
    )
    signature_query_index, signature_query_ptr, signature_query_count = _query_signature_metadata(query_masks)
    return SixBitSupportIndex(
        source_masks=source_masks,
        query_masks=query_masks,
        source_valid=valid,
        support_table=support_table,
        support_count_table=support_count_table,
        logical_path_count_table=logical_path_count_table,
        signature_query_index=signature_query_index,
        signature_query_ptr=signature_query_ptr,
        signature_query_count=signature_query_count,
    )


def live_pair_values(
    query_assignment: torch.Tensor,
    source_membership: torch.Tensor,
    group_control: torch.Tensor,
    pair_indices: torch.Tensor,
    *,
    chunk_size: int = 131_072,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate live ``rho`` and control moments for unique source pairs.

    ``pair_indices`` has shape ``[P,3]`` with columns ``(batch,query,source)``.
    The implementation gathers ``[P,K]`` chunks and contracts them directly;
    it never builds a ``[B,Q,K,S]`` or ``[Q,K,S]`` q-k-s tensor.  Gradients
    remain connected to all three live inputs.
    """

    query_batch, _, _ = _validate_incidence(query_assignment, "query_assignment")
    source_batch, source_count, _ = _validate_incidence(source_membership, "source_membership")
    if query_batch != source_batch:
        raise ValueError("query_assignment and source_membership must share their batch dimension.")
    if not isinstance(group_control, torch.Tensor) or group_control.ndim != 3:
        raise ValueError("group_control must have shape [B,6,D].")
    if tuple(group_control.shape[:2]) != (source_batch, GROUP_COUNT):
        raise ValueError("group_control must align with [B,6].")
    if group_control.device != query_assignment.device or group_control.device != source_membership.device:
        raise ValueError("All live pair inputs must share one device.")
    if not group_control.is_floating_point():
        raise TypeError("group_control must be floating point.")
    if not isinstance(pair_indices, torch.Tensor) or pair_indices.ndim != 2 or int(pair_indices.shape[-1]) != 3:
        raise ValueError("pair_indices must have shape [P,3] as (batch,query,source).")
    if pair_indices.dtype != torch.long:
        raise TypeError("pair_indices must use torch.long metadata.")
    if pair_indices.device != query_assignment.device:
        raise ValueError("pair_indices must share the live-input device.")
    if isinstance(chunk_size, bool) or int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    query_count = int(query_assignment.shape[1])
    pair_count = int(pair_indices.shape[0])
    if pair_count:
        if bool((pair_indices[:, 0] < 0).any()) or bool((pair_indices[:, 0] >= query_batch).any()):
            raise IndexError("pair_indices contains an invalid batch index.")
        if bool((pair_indices[:, 1] < 0).any()) or bool((pair_indices[:, 1] >= query_count).any()):
            raise IndexError("pair_indices contains an invalid query index.")
        if bool((pair_indices[:, 2] < 0).any()) or bool((pair_indices[:, 2] >= source_count).any()):
            raise IndexError("pair_indices contains an invalid source index.")

    rho_chunks: list[torch.Tensor] = []
    moment_chunks: list[torch.Tensor] = []
    for start in range(0, pair_count, int(chunk_size)):
        stop = min(start + int(chunk_size), pair_count)
        batches, queries, sources = pair_indices[start:stop].unbind(dim=-1)
        alpha_pair = query_assignment[batches, queries]
        membership_pair = source_membership[batches, sources]
        controls = group_control[batches]
        rho_chunks.append((alpha_pair * membership_pair).sum(dim=-1))
        moment_chunks.append(torch.einsum("pk,pk,pkd->pd", alpha_pair, membership_pair, controls))
    if not rho_chunks:
        return (
            query_assignment.new_empty((0,)),
            group_control.new_empty((0, int(group_control.shape[-1]))),
        )
    return torch.cat(rho_chunks, dim=0), torch.cat(moment_chunks, dim=0)


__all__ = [
    "GROUP_COUNT",
    "QUERY_MASK_COUNT",
    "QuerySignatureSelection",
    "SixBitSupportIndex",
    "SupportExecutionCounts",
    "build_six_bit_support_index",
    "live_pair_values",
    "pack_six_bit_mask",
    "six_bit_mask",
]
