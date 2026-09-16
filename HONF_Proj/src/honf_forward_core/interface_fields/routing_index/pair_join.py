"""GPU-friendly two-hop source/query join and positive-pair coalescing."""

from __future__ import annotations

import math
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from .types import InvertedSourceIncidence, PackedPairs, TypedSourceIncidence


def _empty_pairs(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedPairs:
    empty_index = torch.empty((0,), device=device, dtype=torch.long)
    return PackedPairs(empty_index, empty_index, empty_index, torch.empty((0,), device=device, dtype=dtype), 0, 0)


def build_inverted_source_incidence(
    membership: torch.Tensor,
    *,
    source_weights: torch.Tensor | None = None,
    source_valid: torch.Tensor | None = None,
) -> InvertedSourceIncidence:
    """Build positive ``(batch, source, hub)`` entries for repeated joins.

    ``values`` is an advanced-indexing view of the original membership tensor
    and therefore carries its gradient.  Integer incidence indices are the
    only objects sorted/grouped by the execution compiler.
    """

    if membership.ndim != 3:
        raise ValueError("membership must have shape [B,N,K].")
    batch, source_count, hub_count = (int(v) for v in membership.shape)
    if source_weights is not None:
        if tuple(source_weights.shape) != (batch, source_count):
            raise ValueError("source_weights must align with membership on [B,N].")
        source_positive = source_weights.to(device=membership.device) > 0.0
    else:
        source_positive = None
    if source_valid is not None:
        if tuple(source_valid.shape) != (batch, source_count):
            raise ValueError("source_valid must align with membership on [B,N].")
        valid = source_valid.to(device=membership.device, dtype=torch.bool)
    else:
        valid = None
    positive = membership > 0.0
    if source_positive is not None:
        positive = positive & source_positive[..., None]
    if valid is not None:
        positive = positive & valid[..., None]
    rows = torch.nonzero(positive, as_tuple=False)
    if rows.numel() == 0:
        empty = torch.empty((0,), device=membership.device, dtype=torch.long)
        values = membership.reshape(-1)[:0]
        batch_count = batch
        offsets = torch.zeros(
            (batch_count * hub_count + 1,),
            device=membership.device,
            dtype=torch.long,
        )
        return InvertedSourceIncidence(empty, empty, empty, values, source_count, hub_count, offsets)
    batch_index, source_index, hub_index = rows.unbind(dim=1)
    values = membership[batch_index, source_index, hub_index]
    key = batch_index * hub_count + hub_index
    order = torch.argsort(key, stable=True)
    batch_index = batch_index[order]
    source_index = source_index[order]
    hub_index = hub_index[order]
    values = values[order]
    counts = torch.bincount(key, minlength=batch * hub_count)
    offsets = torch.cat(
        (
            torch.zeros((1,), device=membership.device, dtype=torch.long),
            counts.cumsum(dim=0),
        ),
        dim=0,
    )
    return InvertedSourceIncidence(
        batch_index=batch_index,
        source_index=source_index,
        hub_index=hub_index,
        values=values,
        source_count=source_count,
        hub_count=hub_count,
        group_offsets=offsets,
    )


def _join_paths(
    query_density: torch.Tensor,
    source_incidence: InvertedSourceIncidence,
    source_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Enumerate positive query/source paths sharing a hub.

    No query-by-hub-by-source tensor is formed.  A cartesian product is
    expanded only for groups containing both a query edge and a source edge;
    the returned scalar path weights remain connected to all three live route
    tensors.
    """

    batch, _, hub_count = (int(v) for v in query_density.shape)
    query_positive = query_density > 0.0
    query_rows = torch.nonzero(query_positive, as_tuple=False)
    source_rows = source_incidence.batch_index
    if query_rows.numel() == 0 or int(source_rows.numel()) == 0:
        empty_index = torch.empty((0,), device=query_density.device, dtype=torch.long)
        empty_value = query_density.reshape(-1)[:0]
        return empty_index, empty_index, empty_index, empty_value, 0

    query_batch, query_index, query_hub = query_rows.unbind(dim=1)
    query_values = query_density[query_batch, query_index, query_hub]
    query_key = query_batch * hub_count + query_hub
    # Source entries are sorted and grouped when the prepared index is built.
    # A fixture can omit offsets; in that case construct the same compact view
    # once for this call rather than relying on ungrouped edge order.
    source_batch = source_incidence.batch_index
    source_index = source_incidence.source_index
    source_hub = source_incidence.hub_index
    source_values = source_incidence.values
    expected_group_count = batch * hub_count
    group_offsets = source_incidence.group_offsets
    if group_offsets is None or int(group_offsets.numel()) != expected_group_count + 1:
        source_key = source_batch * hub_count + source_hub
        source_order = torch.argsort(source_key, stable=True)
        source_batch = source_batch[source_order]
        source_index = source_index[source_order]
        source_values = source_values[source_order]
        source_key = source_key[source_order]
        source_sizes = torch.bincount(source_key, minlength=expected_group_count)
        group_offsets = torch.cat(
            (
                torch.zeros((1,), device=query_density.device, dtype=torch.long),
                source_sizes.cumsum(dim=0),
            ),
            dim=0,
        )
    else:
        source_sizes = group_offsets[1:] - group_offsets[:-1]
    query_order = torch.argsort(query_key, stable=True)
    grouped_query_key = query_key[query_order]
    grouped_query_batch = query_batch[query_order]
    grouped_query_index = query_index[query_order]
    grouped_query_values = query_values[query_order]
    query_sizes = torch.bincount(grouped_query_key, minlength=expected_group_count)
    group_count = expected_group_count
    path_sizes = query_sizes * source_sizes
    raw_path_count_tensor = path_sizes.sum()
    raw_path_count = int(raw_path_count_tensor)
    if raw_path_count == 0:
        empty_index = torch.empty((0,), device=query_density.device, dtype=torch.long)
        empty_value = query_density.reshape(-1)[:0]
        return empty_index, empty_index, empty_index, empty_value, 0

    group_index = torch.arange(group_count, device=query_density.device, dtype=torch.long)
    group_for_path = torch.repeat_interleave(group_index, path_sizes)
    path_starts = path_sizes.cumsum(dim=0) - path_sizes
    path_offset = torch.arange(raw_path_count, device=query_density.device, dtype=torch.long)
    path_offset = path_offset - torch.repeat_interleave(path_starts, path_sizes)
    source_size_for_group = source_sizes[group_for_path]
    query_offset = torch.div(path_offset, source_size_for_group, rounding_mode="floor")
    source_offset = torch.remainder(path_offset, source_size_for_group)
    query_starts = query_sizes.cumsum(dim=0) - query_sizes
    source_starts = group_offsets[:-1]
    query_row = query_starts[group_for_path] + query_offset
    source_row = source_starts[group_for_path] + source_offset

    path_batch = grouped_query_batch[query_row]
    path_receiver = grouped_query_index[query_row]
    path_source = source_index[source_row]
    # Keep scalar route algebra in sufficient precision even when the H-wide
    # neural states remain FP32.  This matters for tiny positive source masses
    # whose paths can underflow before duplicate paths are coalesced.  Casts
    # preserve gradients to source weights, memberships, and query densities.
    scalar_dtype = torch.float64 if (
        source_weights.dtype in {torch.float16, torch.bfloat16, torch.float32}
        or source_values.dtype in {torch.float16, torch.bfloat16, torch.float32}
        or grouped_query_values.dtype in {torch.float16, torch.bfloat16, torch.float32}
    ) else torch.promote_types(source_weights.dtype, grouped_query_values.dtype)
    path_prior = (
        source_weights[path_batch, path_source].to(dtype=scalar_dtype)
        * source_values[source_row].to(dtype=scalar_dtype)
        * grouped_query_values[query_row].to(dtype=scalar_dtype)
    )
    return path_batch, path_receiver, path_source, path_prior, raw_path_count


def _join_path_indices(
    query_density: torch.Tensor,
    source_incidence: InvertedSourceIncidence,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Enumerate positive two-hop paths as integer topology only.

    This is the topology half of :func:`_join_paths`.  It intentionally never
    gathers a route value, so duplicate path expansion does not retain an
    ``omega * A * d`` autograd graph before pair coalescing.
    """

    batch, _, hub_count = (int(v) for v in query_density.shape)
    query_rows = torch.nonzero(query_density > 0.0, as_tuple=False)
    if query_rows.numel() == 0 or int(source_incidence.batch_index.numel()) == 0:
        empty = torch.empty((0,), device=query_density.device, dtype=torch.long)
        return empty, empty, empty, 0

    query_batch, query_index, query_hub = query_rows.unbind(dim=1)
    query_key = query_batch * hub_count + query_hub
    source_batch = source_incidence.batch_index
    source_index = source_incidence.source_index
    source_hub = source_incidence.hub_index
    expected_group_count = batch * hub_count
    group_offsets = source_incidence.group_offsets
    if group_offsets is None or int(group_offsets.numel()) != expected_group_count + 1:
        source_key = source_batch * hub_count + source_hub
        source_order = torch.argsort(source_key, stable=True)
        source_index = source_index[source_order]
        source_key = source_key[source_order]
        source_sizes = torch.bincount(source_key, minlength=expected_group_count)
        group_offsets = torch.cat(
            (
                torch.zeros((1,), device=query_density.device, dtype=torch.long),
                source_sizes.cumsum(dim=0),
            ),
            dim=0,
        )
    else:
        source_sizes = group_offsets[1:] - group_offsets[:-1]

    query_order = torch.argsort(query_key, stable=True)
    grouped_query_key = query_key[query_order]
    grouped_query_batch = query_batch[query_order]
    grouped_query_index = query_index[query_order]
    query_sizes = torch.bincount(grouped_query_key, minlength=expected_group_count)
    path_sizes = query_sizes * source_sizes
    raw_path_count = int(path_sizes.sum())
    if raw_path_count == 0:
        empty = torch.empty((0,), device=query_density.device, dtype=torch.long)
        return empty, empty, empty, 0

    group_index = torch.arange(expected_group_count, device=query_density.device, dtype=torch.long)
    group_for_path = torch.repeat_interleave(group_index, path_sizes)
    path_starts = path_sizes.cumsum(dim=0) - path_sizes
    path_offset = torch.arange(raw_path_count, device=query_density.device, dtype=torch.long)
    path_offset = path_offset - torch.repeat_interleave(path_starts, path_sizes)
    source_size_for_group = source_sizes[group_for_path]
    query_offset = torch.div(path_offset, source_size_for_group, rounding_mode="floor")
    source_offset = torch.remainder(path_offset, source_size_for_group)
    query_starts = query_sizes.cumsum(dim=0) - query_sizes
    source_starts = group_offsets[:-1]
    query_row = query_starts[group_for_path] + query_offset
    source_row = source_starts[group_for_path] + source_offset
    return (
        grouped_query_batch[query_row],
        grouped_query_index[query_row],
        source_index[source_row],
        raw_path_count,
    )


def _materialize_live_source_incidence(
    source_incidence: InvertedSourceIncidence,
    batch: int,
    *,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild filtered live ``A`` and its authoritative support mask.

    ``InvertedSourceIncidence`` is the source of truth for both support and
    values.  Reconstructing from its live values handles an incidence that was
    filtered by source validity or measure, including duplicate entries, while
    keeping gradients connected to the original membership tensor.
    """

    source_count = int(source_incidence.source_count)
    hub_count = int(source_incidence.hub_count)
    flat_key = (
        (source_incidence.batch_index * source_count + source_incidence.source_index)
        * hub_count
        + source_incidence.hub_index
    )
    flat_size = int(batch) * source_count * hub_count
    values = source_incidence.values if dtype is None else source_incidence.values.to(dtype=dtype)
    live_flat = values.new_zeros((flat_size,))
    live_flat = live_flat.scatter_add(0, flat_key, values)
    support_flat = torch.zeros((flat_size,), device=flat_key.device, dtype=torch.bool)
    support_flat = support_flat.scatter(
        0,
        flat_key,
        torch.ones_like(flat_key, dtype=torch.bool),
    )
    return (
        live_flat.reshape(batch, source_count, hub_count),
        support_flat.reshape(batch, source_count, hub_count),
    )


def _pair_prior_batched_block(
    query_density: torch.Tensor,
    source_membership: torch.Tensor,
    source_weights: torch.Tensor,
    batch_index: torch.Tensor,
    receiver_index: torch.Tensor,
    source_index: torch.Tensor,
    query_start: int,
    query_end: int,
    source_start: int,
    source_end: int,
) -> torch.Tensor:
    """Gather selected live priors from one bounded scalar FP64 block."""

    block = torch.bmm(
        query_density[:, query_start:query_end, :],
        source_membership[:, source_start:source_end, :].transpose(1, 2),
    )
    selected = block[
        batch_index,
        receiver_index - int(query_start),
        source_index - int(source_start),
    ]
    return source_weights[batch_index, source_index].to(dtype=torch.float64) * selected


def _pair_prior_tile(
    source_weights: torch.Tensor,
    source_membership: torch.Tensor,
    source_support: torch.Tensor,
    query_density: torch.Tensor,
    batch_index: torch.Tensor,
    receiver_index: torch.Tensor,
    source_index: torch.Tensor,
) -> torch.Tensor:
    """Compute exact live priors for one unique-pair tile.

    The gathers are deliberately inside this function because callers may
    wrap it in non-reentrant checkpointing.  Support masks are integer
    topology decisions; the selected values remain differentiable.
    """

    source_values = source_membership[batch_index, source_index]
    source_values = torch.where(
        source_support[batch_index, source_index],
        source_values,
        torch.zeros_like(source_values),
    )
    query_values = query_density[batch_index, receiver_index]
    query_values = torch.where(
        query_values > 0.0,
        query_values,
        torch.zeros_like(query_values),
    )
    scalar_dtype = torch.float64 if (
        source_weights.dtype in {torch.float16, torch.bfloat16, torch.float32}
        or source_membership.dtype in {torch.float16, torch.bfloat16, torch.float32}
        or query_density.dtype in {torch.float16, torch.bfloat16, torch.float32}
    ) else torch.promote_types(source_weights.dtype, query_density.dtype)
    return (
        source_weights[batch_index, source_index].to(dtype=scalar_dtype)
        * (
            source_values.to(dtype=scalar_dtype)
            * query_values.to(dtype=scalar_dtype)
        ).sum(dim=-1)
    )


def compile_two_hop_pairs_reference(
    query_density: torch.Tensor,
    source: TypedSourceIncidence | InvertedSourceIncidence,
    source_weights: torch.Tensor | None = None,
    *,
    query_count: int | None = None,
) -> PackedPairs:
    """Compile and deduplicate positive receiver/source pairs.

    ``query_density`` is the live ``d`` from source-measure sparsemax, while
    the source argument supplies live ordinary membership values ``A``.  The
    executable prior is exactly

    ``Pi[q, i] = omega[i] * sum_k A[i, k] * d[q, k]``.

    A pair reached through multiple hubs is returned once, with path weights
    summed by a differentiable ``scatter_add``.  Fine QM/QE code can therefore
    gather the returned indices without evaluating duplicate micro-messages.
    """

    if query_density.ndim != 3:
        raise ValueError("query_density must have shape [B,Q,K].")
    batch, receiver_count, hub_count = (int(v) for v in query_density.shape)
    if query_count is not None and int(query_count) != receiver_count:
        raise ValueError("query_count does not match query_density.")
    if isinstance(source, TypedSourceIncidence):
        incidence = source.inverted
        weights = source.source_weights if source_weights is None else source_weights
    elif isinstance(source, InvertedSourceIncidence):
        incidence = source
        if source_weights is None:
            raise ValueError("source_weights is required with an inverted incidence alone.")
        weights = source_weights
    else:
        raise TypeError("source must be TypedSourceIncidence or InvertedSourceIncidence.")
    if tuple(weights.shape) != (batch, incidence.source_count):
        raise ValueError("source_weights must align with query batch and source count.")
    if incidence.hub_count != hub_count:
        raise ValueError("Source incidence hub count must match query_density.")
    if incidence.values.device != query_density.device or weights.device != query_density.device:
        raise ValueError("Query, source incidence, and source weights must share a device.")
    if receiver_count == 0 or incidence.source_count == 0:
        return _empty_pairs(device=query_density.device, dtype=query_density.dtype)

    path_batch, path_receiver, path_source, path_prior, raw_path_count = _join_paths(
        query_density,
        incidence,
        weights,
    )
    if raw_path_count == 0:
        return _empty_pairs(device=query_density.device, dtype=query_density.dtype)

    # Integer keys are used solely for grouping.  The prior values are sorted
    # with those keys and then accumulated without detaching their graph.
    pair_key = (path_batch * receiver_count + path_receiver) * incidence.source_count + path_source
    order = torch.argsort(pair_key, stable=True)
    sorted_key = pair_key[order]
    unique_key, inverse = torch.unique_consecutive(sorted_key, return_inverse=True)
    unique_count = int(unique_key.numel())
    prior = torch.zeros(
        (unique_count,),
        device=path_prior.device,
        dtype=path_prior.dtype,
    )
    prior = prior.scatter_add(0, inverse, path_prior[order])
    unique_batch = torch.div(unique_key, receiver_count * incidence.source_count, rounding_mode="floor")
    remainder = torch.remainder(unique_key, receiver_count * incidence.source_count)
    unique_receiver = torch.div(remainder, incidence.source_count, rounding_mode="floor")
    unique_source = torch.remainder(remainder, incidence.source_count)
    return PackedPairs(
        batch_index=unique_batch,
        receiver_index=unique_receiver,
        source_index=unique_source,
        prior=prior,
        raw_path_count=raw_path_count,
        unique_pair_count=unique_count,
    )


def compile_two_hop_pairs_optimized(
    query_density: torch.Tensor,
    source: TypedSourceIncidence | InvertedSourceIncidence,
    source_weights: torch.Tensor | None = None,
    *,
    query_count: int | None = None,
    pair_tile_size: int = 16384,
    use_checkpoint: bool = True,
) -> PackedPairs:
    """Compile the same pairs while delaying live prior algebra until tiles.

    Positive topology is built and coalesced with integer indices first.  The
    exact prior for each unique pair is then evaluated as
    ``omega[i] * sum_k A[i,k] * d[q,k]`` in bounded tiles.  The optimized path
    is intentionally separate from :func:`compile_two_hop_pairs_reference`
    until its numerical and gradient equivalence is reviewed.

    ``use_checkpoint`` checkpoints the tile function with
    ``use_reentrant=False`` when gradients are enabled.  All gathers occur in
    that function, so the temporary ``[pair_tile, K]`` route values are not
    retained by the forward graph.  Releasing the integer topology workspace
    before these tiles is a memory-lifetime optimization only; it carries no
    latency claim until an execution benchmark is run.
    """

    if query_density.ndim != 3:
        raise ValueError("query_density must have shape [B,Q,K].")
    if isinstance(pair_tile_size, bool) or int(pair_tile_size) <= 0:
        raise ValueError("pair_tile_size must be a positive integer.")
    batch, receiver_count, hub_count = (int(v) for v in query_density.shape)
    if query_count is not None and int(query_count) != receiver_count:
        raise ValueError("query_count does not match query_density.")
    if isinstance(source, TypedSourceIncidence):
        incidence = source.inverted
        weights = source.source_weights if source_weights is None else source_weights
    elif isinstance(source, InvertedSourceIncidence):
        incidence = source
        if source_weights is None:
            raise ValueError("source_weights is required with an inverted incidence alone.")
        weights = source_weights
    else:
        raise TypeError("source must be TypedSourceIncidence or InvertedSourceIncidence.")
    if tuple(weights.shape) != (batch, incidence.source_count):
        raise ValueError("source_weights must align with query batch and source count.")
    if incidence.hub_count != hub_count:
        raise ValueError("Source incidence hub count must match query_density.")
    if incidence.values.device != query_density.device or weights.device != query_density.device:
        raise ValueError("Query, source incidence, and source weights must share a device.")
    if receiver_count == 0 or incidence.source_count == 0:
        return _empty_pairs(device=query_density.device, dtype=query_density.dtype)

    path_batch, path_receiver, path_source, raw_path_count = _join_path_indices(
        query_density,
        incidence,
    )
    if raw_path_count == 0:
        return _empty_pairs(device=query_density.device, dtype=query_density.dtype)

    pair_key = (path_batch * receiver_count + path_receiver) * incidence.source_count + path_source
    order = torch.argsort(pair_key, stable=True)
    unique_key = torch.unique_consecutive(pair_key[order])
    unique_count = int(unique_key.numel())
    unique_batch = torch.div(unique_key, receiver_count * incidence.source_count, rounding_mode="floor")
    remainder = torch.remainder(unique_key, receiver_count * incidence.source_count)
    unique_receiver = torch.div(remainder, incidence.source_count, rounding_mode="floor")
    unique_source = torch.remainder(remainder, incidence.source_count)
    # The raw integer topology has served its purpose once unique keys and
    # indices exist.  Release it before constructing live prior tiles; this
    # bounds overlap between duplicate-path workspaces and route algebra.
    del path_batch, path_receiver, path_source, pair_key, order

    source_membership, source_support = _materialize_live_source_incidence(incidence, batch)
    prior_tiles: list[torch.Tensor] = []
    tile_size = int(pair_tile_size)
    for start in range(0, unique_count, tile_size):
        end = min(start + tile_size, unique_count)
        tile_args = (
            weights,
            source_membership,
            source_support,
            query_density,
            unique_batch[start:end],
            unique_receiver[start:end],
            unique_source[start:end],
        )
        has_live_grad = any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for value in tile_args[:4]
        )
        if use_checkpoint and torch.is_grad_enabled() and has_live_grad:
            prior_tile = checkpoint(_pair_prior_tile, *tile_args, use_reentrant=False)
        else:
            prior_tile = _pair_prior_tile(*tile_args)
        prior_tiles.append(prior_tile)
    prior = torch.cat(prior_tiles, dim=0)
    return PackedPairs(
        batch_index=unique_batch,
        receiver_index=unique_receiver,
        source_index=unique_source,
        prior=prior,
        raw_path_count=raw_path_count,
        unique_pair_count=unique_count,
    )


def compile_two_hop_pairs_batched(
    query_density: torch.Tensor,
    source: TypedSourceIncidence | InvertedSourceIncidence,
    source_weights: torch.Tensor | None = None,
    *,
    query_count: int | None = None,
    scalar_tile_size: int = 262144,
    use_checkpoint: bool = False,
) -> PackedPairs:
    """Compile exact priors with bounded batched FP64 scalar products.

    The integer positive-path union is intentionally the same as the current
    optimized compiler.  Once unique pair indices exist, each occupied
    query/source block computes a bounded ``d @ A.T`` scalar tile and gathers
    only the requested unique pairs.  The block is never coupled to the fine
    neural tile size and no ``[B,Q,K,N]`` tensor is formed.  Checkpointing is
    opt-in because the scalar blocks are small relative to the avoided
    per-pair route tiles; it remains available for larger workloads.
    """

    if query_density.ndim != 3:
        raise ValueError("query_density must have shape [B,Q,K].")
    if isinstance(scalar_tile_size, bool) or int(scalar_tile_size) <= 0:
        raise ValueError("scalar_tile_size must be a positive integer.")
    batch, receiver_count, hub_count = (int(v) for v in query_density.shape)
    scalar_tile_size = int(scalar_tile_size)
    if scalar_tile_size < batch:
        raise ValueError("scalar_tile_size must be at least the query batch size.")
    if query_count is not None and int(query_count) != receiver_count:
        raise ValueError("query_count does not match query_density.")
    if isinstance(source, TypedSourceIncidence):
        incidence = source.inverted
        weights = source.source_weights if source_weights is None else source_weights
    elif isinstance(source, InvertedSourceIncidence):
        incidence = source
        if source_weights is None:
            raise ValueError("source_weights is required with an inverted incidence alone.")
        weights = source_weights
    else:
        raise TypeError("source must be TypedSourceIncidence or InvertedSourceIncidence.")
    if tuple(weights.shape) != (batch, incidence.source_count):
        raise ValueError("source_weights must align with query batch and source count.")
    if incidence.hub_count != hub_count:
        raise ValueError("Source incidence hub count must match query_density.")
    if incidence.values.device != query_density.device or weights.device != query_density.device:
        raise ValueError("Query, source incidence, and source weights must share a device.")
    if receiver_count == 0 or incidence.source_count == 0:
        return _empty_pairs(device=query_density.device, dtype=query_density.dtype)

    path_batch, path_receiver, path_source, raw_path_count = _join_path_indices(
        query_density,
        incidence,
    )
    if raw_path_count == 0:
        return _empty_pairs(device=query_density.device, dtype=query_density.dtype)

    pair_key = (path_batch * receiver_count + path_receiver) * incidence.source_count + path_source
    order = torch.argsort(pair_key, stable=True)
    unique_key = torch.unique_consecutive(pair_key[order])
    unique_count = int(unique_key.numel())
    unique_batch = torch.div(unique_key, receiver_count * incidence.source_count, rounding_mode="floor")
    remainder = torch.remainder(unique_key, receiver_count * incidence.source_count)
    unique_receiver = torch.div(remainder, incidence.source_count, rounding_mode="floor")
    unique_source = torch.remainder(remainder, incidence.source_count)
    del path_batch, path_receiver, path_source, pair_key, order

    # The source incidence is authoritative for support.  Materialize its
    # live values in FP64 before any block product, so duplicate incidence
    # rows are accumulated in the same scalar precision as the prior.
    source_membership, source_support = _materialize_live_source_incidence(
        incidence,
        batch,
        dtype=torch.float64,
    )
    query_values = torch.where(
        query_density > 0.0,
        query_density,
        torch.zeros_like(query_density),
    ).to(dtype=torch.float64)
    source_values = torch.where(
        source_support,
        source_membership,
        torch.zeros_like(source_membership),
    )

    # Keep each FP64 scalar block within the requested element budget.  When
    # the source axis fits, retaining it whole minimizes the number of blocks
    # for the common Q<=128, N~=192 training shape.  Larger axes use a
    # roughly square bounded block.
    entries_per_batch = max(1, scalar_tile_size // batch)
    if incidence.source_count <= entries_per_batch:
        source_tile = incidence.source_count
        query_tile = max(1, min(receiver_count, entries_per_batch // source_tile))
    else:
        query_tile = max(1, min(receiver_count, int(math.sqrt(entries_per_batch))))
        source_tile = max(1, min(incidence.source_count, entries_per_batch // query_tile))
    if batch * query_tile * source_tile > scalar_tile_size:
        raise RuntimeError("Internal scalar tile exceeds scalar_tile_size.")
    source_block_count = (incidence.source_count + source_tile - 1) // source_tile
    query_block = torch.div(unique_receiver, query_tile, rounding_mode="floor")
    source_block = torch.div(unique_source, source_tile, rounding_mode="floor")
    block_key = query_block * source_block_count + source_block
    block_order = torch.argsort(block_key, stable=True)
    sorted_block_key = block_key[block_order]
    occupied_blocks, block_counts = torch.unique_consecutive(
        sorted_block_key,
        return_counts=True,
    )
    # Block metadata is tiny compared with the scalar operands.  Transfer it
    # once so the bounded block loop does not synchronize on every index.
    block_metadata = torch.stack((occupied_blocks, block_counts), dim=1).detach().cpu().tolist()
    selected_positions: list[torch.Tensor] = []
    selected_priors: list[torch.Tensor] = []
    start = 0
    for block_id, block_size in block_metadata:
        end = start + block_size
        positions = block_order[start:end]
        query_start = (block_id // source_block_count) * query_tile
        query_end = min(query_start + query_tile, receiver_count)
        source_start = (block_id % source_block_count) * source_tile
        source_end = min(source_start + source_tile, incidence.source_count)
        block_args = (
            query_values,
            source_values,
            weights,
            unique_batch[positions],
            unique_receiver[positions],
            unique_source[positions],
        )
        has_live_grad = any(value.requires_grad for value in block_args[:3])
        if use_checkpoint and torch.is_grad_enabled() and has_live_grad:
            prior_values = checkpoint(
                partial(
                    _pair_prior_batched_block,
                    query_start=query_start,
                    query_end=query_end,
                    source_start=source_start,
                    source_end=source_end,
                ),
                *block_args,
                use_reentrant=False,
            )
        else:
            prior_values = _pair_prior_batched_block(
                *block_args,
                query_start,
                query_end,
                source_start,
                source_end,
            )
        selected_positions.append(positions)
        selected_priors.append(prior_values)
        start = end
    positions = torch.cat(selected_positions, dim=0)
    values = torch.cat(selected_priors, dim=0)
    prior = values.new_zeros((unique_count,)).scatter(0, positions, values)
    return PackedPairs(
        batch_index=unique_batch,
        receiver_index=unique_receiver,
        source_index=unique_source,
        prior=prior,
        raw_path_count=raw_path_count,
        unique_pair_count=unique_count,
    )


# Keep the historical public default/reference compiler unchanged while the
# compact implementation is benchmarked and reviewed independently.
compile_two_hop_pairs = compile_two_hop_pairs_reference


# ``compile_routed_pairs`` is intentionally a semantic alias: call sites can
# use either the mathematical two-hop or execution-oriented wording.
compile_routed_pairs = compile_two_hop_pairs
