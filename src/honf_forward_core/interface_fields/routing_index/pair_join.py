"""GPU-friendly two-hop source/query join and positive-pair coalescing."""

from __future__ import annotations

import math
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from .types import CompiledPairs, InvertedSourceIncidence, PackedPairs, TypedSourceIncidence


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


def _resolve_compiled_source(
    query_density: torch.Tensor,
    source: TypedSourceIncidence | InvertedSourceIncidence,
    source_weights: torch.Tensor | None,
) -> tuple[InvertedSourceIncidence, torch.Tensor]:
    """Validate and normalize source arguments for the union compiler."""

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
    batch, _, hub_count = (int(v) for v in query_density.shape)
    if tuple(weights.shape) != (batch, incidence.source_count):
        raise ValueError("source_weights must align with query batch and source count.")
    if incidence.hub_count != hub_count:
        raise ValueError("Source incidence hub count must match query_density.")
    if incidence.values.device != query_density.device or weights.device != query_density.device:
        raise ValueError("Query, source incidence, and source weights must share a device.")
    return incidence, weights


def _compiled_union_tile_sizes(
    batch: int,
    receiver_count: int,
    source_count: int,
    hub_count: int,
    budget: int,
) -> tuple[int, int, int]:
    """Choose bounded ``(Q,N,K)`` Boolean-union tiles.

    The bound is on the largest temporary Boolean block, including the batch
    dimension.  Hub tiling is required when K is wider than the budget; this
    keeps the implementation valid for unusually large candidate banks.
    """

    if isinstance(budget, bool) or int(budget) <= 0:
        raise ValueError("support_tile_size must be a positive integer.")
    if batch <= 0 or receiver_count <= 0 or source_count <= 0 or hub_count <= 0:
        return 1, 1, 1
    budget = int(budget)
    if budget < batch:
        raise ValueError("support_tile_size must be at least the query batch size.")
    per_batch = max(1, budget // batch)
    # A compact cube is a useful starting point, but all final dimensions are
    # reduced explicitly below so rounding can never exceed the budget.
    hub_tile = min(hub_count, max(1, int(per_batch ** (1.0 / 3.0))))
    receiver_tile = min(
        receiver_count,
        max(1, int((per_batch / float(max(hub_tile, 1))) ** 0.5)),
    )
    source_tile = min(
        source_count,
        max(1, per_batch // max(receiver_tile * hub_tile, 1)),
    )
    while receiver_tile * source_tile * hub_tile > per_batch:
        if source_tile > 1:
            source_tile -= 1
        elif receiver_tile > 1:
            receiver_tile -= 1
        elif hub_tile > 1:
            hub_tile -= 1
        else:
            break
    return receiver_tile, source_tile, hub_tile


def _compiled_complete_prior(
    query_density: torch.Tensor,
    source_membership: torch.Tensor,
    source_support: torch.Tensor,
    source_weights: torch.Tensor,
    complete_row_index: torch.Tensor,
    *,
    scalar_tile_size: int,
) -> torch.Tensor:
    """Evaluate Eq. (1) for complete rows without pair-index materialization."""

    batch, receiver_count, hub_count = (int(v) for v in query_density.shape)
    source_count = int(source_membership.shape[1])
    complete_count = int(complete_row_index.numel())
    scalar_dtype = torch.float64
    if complete_count == 0:
        return source_membership.new_empty((0, source_count), dtype=scalar_dtype)
    if isinstance(scalar_tile_size, bool) or int(scalar_tile_size) <= 0:
        raise ValueError("scalar_tile_size must be a positive integer.")
    scalar_tile_size = int(scalar_tile_size)
    if hub_count == 0:
        return source_membership.new_zeros((complete_count, source_count), dtype=scalar_dtype)

    query_values = torch.where(
        query_density > 0.0,
        query_density,
        torch.zeros_like(query_density),
    ).to(dtype=scalar_dtype)
    source_values = torch.where(
        source_support,
        source_membership,
        torch.zeros_like(source_membership),
    ).to(dtype=scalar_dtype)
    weights = source_weights.to(dtype=scalar_dtype)
    if complete_count == batch * receiver_count:
        # Common complete-support workload: reuse each case's source bank
        # across all its queries rather than launching one mm per case.
        # Tile the batch as well when a source bank alone exceeds the budget.
        n_tile = max(1, min(source_count, scalar_tile_size // max(hub_count, 1)))
        b_tile = max(1, min(batch, scalar_tile_size // max(n_tile * hub_count, n_tile, 1)))
        batch_outputs = []
        for b0 in range(0, batch, b_tile):
            b1 = min(batch, b0 + b_tile)
            q_tile = max(1, scalar_tile_size // ((b1-b0) * max(n_tile, hub_count, 1)))
            query_outputs = []
            for q0 in range(0, receiver_count, q_tile):
                q1 = min(receiver_count, q0 + q_tile)
                source_outputs = []
                for n0 in range(0, source_count, n_tile):
                    n1 = min(source_count, n0 + n_tile)
                    source_outputs.append(torch.bmm(
                        query_values[b0:b1, q0:q1],
                        source_values[b0:b1, n0:n1].transpose(1, 2),
                    ) * weights[b0:b1, None, n0:n1])
                query_outputs.append(torch.cat(source_outputs, dim=-1))
            batch_outputs.append(torch.cat(query_outputs, dim=1))
        return torch.cat(batch_outputs, dim=0).reshape(complete_count, source_count)

    # Keep the largest bmm within the scalar workspace budget.  The returned
    # complete prior is compact in rows but can still be wide in source count;
    # callers use it as scalar metadata, never as a fine feature tile.
    source_tile = max(1, min(source_count, scalar_tile_size // max(hub_count, 1)))
    # Source [N,K] is reused by rectangular mm; no [Q,N,K] product exists.
    # Bound each source, query and output scalar workspace independently.
    complete_batches = torch.div(complete_row_index, receiver_count, rounding_mode="floor")
    counts = torch.bincount(complete_batches, minlength=batch)
    full_indices = torch.nonzero(counts == receiver_count, as_tuple=False).flatten()
    output = source_values.new_zeros((complete_count, source_count))
    full_case_mask = torch.zeros(batch, device=query_values.device, dtype=torch.bool)
    if int(full_indices.numel()):
        group_count = int(full_indices.numel())
        rectangular = _compiled_complete_prior(
            query_values.index_select(0, full_indices),
            source_values.index_select(0, full_indices),
            source_support.index_select(0, full_indices),
            weights.index_select(0, full_indices),
            torch.arange(group_count * receiver_count, device=query_values.device),
            scalar_tile_size=scalar_tile_size,
        )
        full_case_mask[full_indices] = True
        full_positions = torch.nonzero(full_case_mask[complete_batches], as_tuple=False).flatten()
        output = output.index_copy(0, full_positions, rectangular)
    remaining_positions = torch.nonzero(~full_case_mask[complete_batches], as_tuple=False).flatten()
    # Irregular complete rows still have implicit source banks. Gather only
    # bounded scalar incidence blocks, never hidden states or duplicate paths.
    # Packing across physical cases removes a Python/mm launch per case.
    packed_row_tile = max(1, scalar_tile_size // max(source_tile * hub_count, 1))
    for r0 in range(0, int(remaining_positions.numel()), packed_row_tile):
        positions = remaining_positions[r0:r0 + packed_row_tile]
        flat_rows = complete_row_index.index_select(0, positions)
        batch_ids = torch.div(flat_rows, receiver_count, rounding_mode="floor")
        query_ids = torch.remainder(flat_rows, receiver_count)
        query_block = query_values[batch_ids, query_ids].unsqueeze(1)
        source_outputs = []
        for n0 in range(0, source_count, source_tile):
            n1 = min(source_count, n0 + source_tile)
            source_block = source_values[batch_ids, n0:n1]
            product = torch.bmm(query_block, source_block.transpose(1, 2)).squeeze(1)
            source_outputs.append(product * weights[batch_ids, n0:n1])
        output = output.index_copy(0, positions, torch.cat(source_outputs, dim=1))
    return output


def _compiled_partial_union(
    query_density: torch.Tensor,
    source_support: torch.Tensor,
    occupied_hubs: torch.Tensor,
    complete_rows: torch.Tensor,
    *,
    support_tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build the exact partial-row CSR union with bounded Boolean tiles."""

    batch, receiver_count, hub_count = (int(v) for v in query_density.shape)
    source_count = int(source_support.shape[1])
    receiver_tile, source_tile, hub_tile = _compiled_union_tile_sizes(
        batch,
        receiver_count,
        source_count,
        hub_count,
        support_tile_size,
    )
    query_support = (query_density > 0.0) & occupied_hubs[:, None, :]
    # Complete rows are certified before entering the union.  Clearing them
    # here keeps their work out of the emitted CSR even if a future tile shape
    # visits their batch row.
    query_support = query_support & (~complete_rows[..., None])
    partial_mask = ~complete_rows
    if not bool(partial_mask.any()):
        empty_ptr = torch.zeros(
            (batch * receiver_count + 1,),
            device=query_density.device,
            dtype=torch.long,
        )
        empty_source = torch.empty((0,), device=query_density.device, dtype=torch.long)
        # No Boolean candidate scan is needed when the pre-join certificate
        # covers every row.
        return empty_ptr, empty_source, empty_source, 0

    # Pack each 63-hub word into one int64.  The highest bit is deliberately
    # left unused so every packed word is a nonnegative signed int64 on both
    # CPU and CUDA.  A word intersection is an exact support test; it does not
    # threshold or otherwise approximate the live floating route values.
    word_bits = 63
    word_count = (hub_count + word_bits - 1) // word_bits

    def pack_support(mask: torch.Tensor) -> torch.Tensor:
        packed = torch.zeros(
            (*mask.shape[:-1], word_count),
            device=mask.device,
            dtype=torch.int64,
        )
        # Convert only a few words at a time.  A single ``mask.to(int64)``
        # over a very wide candidate bank would create an avoidable
        # ``[B,N,K]`` 64-bit temporary and defeat the bounded support reader.
        words_per_chunk = 4
        expand_prefix = (1,) * (mask.ndim - 1)
        for word_start in range(0, word_count, words_per_chunk):
            hub_start = word_start * word_bits
            hub_end = min(hub_count, (word_start + words_per_chunk) * word_bits)
            width = hub_end - hub_start
            indices = torch.arange(
                hub_start,
                hub_end,
                device=mask.device,
                dtype=torch.long,
            )
            local_words = torch.div(indices, word_bits, rounding_mode="floor")
            local_bits = torch.remainder(indices, word_bits)
            bit_value = torch.bitwise_left_shift(
                torch.ones_like(local_bits, dtype=torch.int64),
                local_bits,
            )
            packed.scatter_add_(
                -1,
                local_words.reshape(*expand_prefix, width).expand(
                    *mask.shape[:-1], width
                ),
                mask[..., hub_start:hub_end].to(dtype=torch.int64)
                * bit_value.reshape(*expand_prefix, width),
            )
        return packed

    # Word packing removes the repeated per-hub Boolean launches for the
    # current candidate banks.  Pathological K larger than the scalar support
    # budget keeps the tiled Boolean implementation below, where a single
    # source/query block can still be formed without exceeding the budget.
    use_bitset = word_count <= int(support_tile_size)
    partial_row_ids = torch.nonzero(partial_mask.reshape(-1), as_tuple=False).flatten()

    if use_bitset:
        query_support_flat = query_support.reshape(batch * receiver_count, hub_count)
        row_count = int(partial_row_ids.numel())
        word_budget = max(1, int(support_tile_size) // max(word_count, 1))
        if source_count <= word_budget:
            bitset_source_tile = source_count
            bitset_row_tile = max(1, min(row_count, word_budget // max(bitset_source_tile, 1)))
        else:
            bitset_source_tile = max(1, min(source_count, int(math.sqrt(word_budget))))
            bitset_row_tile = max(1, min(row_count, word_budget // bitset_source_tile))
        while (
            bitset_row_tile * bitset_source_tile * word_count
            > int(support_tile_size)
        ):
            if bitset_source_tile > 1:
                bitset_source_tile -= 1
            elif bitset_row_tile > 1:
                bitset_row_tile -= 1
            else:
                break

        def scan_bitset_blocks():
            for row_start in range(0, row_count, bitset_row_tile):
                row_end = min(row_start + bitset_row_tile, row_count)
                row_ids = partial_row_ids[row_start:row_end]
                row_batch = torch.div(row_ids, receiver_count, rounding_mode="floor")
                # Pack only this receiver tile so a very large Q does not
                # create an unbounded [B*Q, words] support workspace.
                query_block = pack_support(query_support_flat.index_select(0, row_ids))
                for source_start in range(0, source_count, bitset_source_tile):
                    source_end = min(source_start + bitset_source_tile, source_count)
                    # Gather only the current source tile.  In particular, a
                    # large source bank must not be copied for every row tile
                    # before its source axis is sliced.
                    source_block = pack_support(
                        source_support[:, source_start:source_end].index_select(0, row_batch)
                    )
                    support_block = torch.bitwise_and(
                        query_block[:, None, :], source_block
                    ).ne(0).any(dim=-1)
                    hits = torch.nonzero(support_block, as_tuple=False)
                    if hits.numel() == 0:
                        continue
                    yield (
                        row_ids.index_select(0, hits[:, 0]),
                        hits[:, 1] + source_start,
                    )

        scan_blocks = scan_bitset_blocks
    else:
        scan_blocks = None

    def scan_tiled_blocks():
        """Yield unique ``(row, source)`` entries in bounded tiles."""

        for query_start in range(0, receiver_count, receiver_tile):
            query_end = min(query_start + receiver_tile, receiver_count)
            query_width = query_end - query_start
            for source_start in range(0, source_count, source_tile):
                source_end = min(source_start + source_tile, source_count)
                source_width = source_end - source_start
                support_block = torch.zeros(
                    (batch, query_width, source_width),
                    device=query_density.device,
                    dtype=torch.bool,
                )
                for hub_start in range(0, hub_count, hub_tile):
                    hub_end = min(hub_start + hub_tile, hub_count)
                    query_block = query_support[:, query_start:query_end, hub_start:hub_end]
                    source_block = source_support[:, source_start:source_end, hub_start:hub_end]
                    support_block = support_block | (
                        query_block[:, :, None, :] & source_block[:, None, :, :]
                    ).any(dim=-1)
                support_block = support_block & partial_mask[:, query_start:query_end, None]
                rows = torch.nonzero(support_block, as_tuple=False)
                if rows.numel() == 0:
                    continue
                yield (
                    rows[:, 0] * receiver_count + rows[:, 1] + query_start,
                    rows[:, 2] + source_start,
                )

    if scan_blocks is None:
        scan_blocks = scan_tiled_blocks

    row_counts = torch.zeros(
        (batch * receiver_count,),
        device=query_density.device,
        dtype=torch.long,
    )
    # First pass determines row lengths.  The second pass below writes each
    # source directly at its row-owned CSR offset, so no global sort is needed.
    for flat_rows, _ in scan_blocks():
        row_counts.index_add_(
            0,
            flat_rows,
            torch.ones_like(flat_rows, dtype=torch.long),
        )
    row_ptr = torch.cat(
        (
            torch.zeros((1,), device=query_density.device, dtype=torch.long),
            row_counts.cumsum(dim=0),
        ),
        dim=0,
    )
    total = int(row_counts.sum())
    if total == 0:
        sources = torch.empty((0,), device=query_density.device, dtype=torch.long)
        rows = torch.empty((0,), device=query_density.device, dtype=torch.long)
    else:
        sources = torch.empty((total,), device=query_density.device, dtype=torch.long)
        write_offsets = row_ptr[:-1].clone()
        for flat_rows, flat_sources in scan_blocks():
            # ``flat_rows`` can contain several source entries for one row in
            # a source tile.  Add each entry's within-tile rank so writes are
            # row-owned and collision-free; a plain row offset would leave
            # uninitialized CSR slots when a tile contains multiple sources.
            _, local_counts = torch.unique_consecutive(flat_rows, return_counts=True)
            local_starts = local_counts.cumsum(dim=0) - local_counts
            local_rank = torch.arange(
                int(flat_rows.numel()),
                device=flat_rows.device,
                dtype=torch.long,
            ) - torch.repeat_interleave(local_starts, local_counts)
            destination = write_offsets.index_select(0, flat_rows) + local_rank
            sources.index_copy_(0, destination, flat_sources)
            write_offsets.index_add_(
                0,
                flat_rows,
                torch.ones_like(flat_rows, dtype=torch.long),
            )
        rows = torch.repeat_interleave(
            torch.arange(batch * receiver_count, device=query_density.device, dtype=torch.long),
            row_counts,
        )
    # Count the candidate Boolean examinations for rows that actually need a
    # union.  Complete rows were dispatched by the certificate above.
    examined = int(partial_mask.sum()) * source_count * hub_count
    return row_ptr, sources, rows, examined


def _compiled_partial_prior(
    query_density: torch.Tensor,
    source_membership: torch.Tensor,
    source_support: torch.Tensor,
    source_weights: torch.Tensor,
    row_ids: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    scalar_tile_size: int,
) -> torch.Tensor:
    """Evaluate live priors for CSR-selected pairs in bounded scalar tiles."""

    count = int(source_ids.numel())
    if count == 0:
        return source_membership.new_empty((0,), dtype=torch.float64)
    _, receiver_count, _ = (int(v) for v in query_density.shape)
    scalar_tile_size = int(scalar_tile_size)
    hub_count = int(query_density.shape[-1])
    pair_tile = max(1, min(count, scalar_tile_size // max(hub_count, 1)))
    outputs: list[torch.Tensor] = []
    for start in range(0, count, pair_tile):
        end = min(start + pair_tile, count)
        row_block = row_ids[start:end]
        source_block = source_ids[start:end]
        batch_index = torch.div(row_block, receiver_count, rounding_mode="floor")
        receiver_index = torch.remainder(row_block, receiver_count)
        # Cast only this bounded pair tile.  In particular, avoid a full
        # FP64 [B,N,K] source copy when a source bank exceeds the scalar
        # workspace budget.
        query_values = query_density[batch_index, receiver_index]
        query_values = torch.where(
            query_values > 0.0,
            query_values,
            torch.zeros_like(query_values),
        ).to(dtype=torch.float64)
        source_values = source_membership[batch_index, source_block]
        source_values = torch.where(
            source_support[batch_index, source_block],
            source_values,
            torch.zeros_like(source_values),
        ).to(dtype=torch.float64)
        product = (query_values * source_values).sum(dim=-1)
        outputs.append(
            product * source_weights[batch_index, source_block].to(dtype=torch.float64)
        )
    return torch.cat(outputs, dim=0)


def _validate_compiled_priors(
    prior: torch.Tensor,
    valid: torch.Tensor | None,
    *,
    label: str,
) -> None:
    """Reject selected route products that are nonpositive or nonfinite.

    Boolean support is a topology decision, so a positive path must still
    carry a positive scalar prior when the selected route is evaluated.  A
    product that underflows to zero would otherwise be silently passed to
    ``log`` by the QE reader (or to a zero-weight QM reduction), changing the
    mathematical route while hiding the numerical failure.  Invalid padded
    complete slots are excluded through ``valid``.
    """

    selected = prior if valid is None else prior[valid]
    if selected.numel() == 0:
        return
    bad = (~torch.isfinite(selected)) | (selected <= 0.0)
    if bool(bad.any()):
        bad_count = int(bad.sum())
        raise FloatingPointError(
            f"compiled {label} prior contains {bad_count} selected values "
            "that are nonpositive or nonfinite; increase scalar precision "
            "or remove the underflowing route support"
        )


def compile_two_hop_pairs_compiled(
    query_density: torch.Tensor,
    source: TypedSourceIncidence | InvertedSourceIncidence,
    source_weights: torch.Tensor | None = None,
    *,
    query_count: int | None = None,
    support_tile_size: int = 262144,
    scalar_tile_size: int = 262144,
) -> CompiledPairs:
    """Compile an exact two-hop union into complete rows plus partial CSR.

    The completeness certificate is evaluated before any join.  A row that
    activates every occupied source hub gets an implicit valid source range;
    its live priors are evaluated by bounded scalar products.  Other rows use
    a tiled Boolean union over authoritative positive incidence, so no
    ``[B,Q,K,N]`` tensor and no duplicated ``(q,k,i)`` path list is formed.
    """

    if query_density.ndim != 3:
        raise ValueError("query_density must have shape [B,Q,K].")
    batch, receiver_count, hub_count = (int(v) for v in query_density.shape)
    if query_count is not None and int(query_count) != receiver_count:
        raise ValueError("query_count does not match query_density.")
    incidence, weights = _resolve_compiled_source(query_density, source, source_weights)
    source_count = int(incidence.source_count)
    if isinstance(scalar_tile_size, bool) or int(scalar_tile_size) <= 0:
        raise ValueError("scalar_tile_size must be a positive integer.")
    if int(scalar_tile_size) < max(1, hub_count):
        raise ValueError("scalar_tile_size must be at least the hub count.")
    if isinstance(support_tile_size, bool) or int(support_tile_size) <= 0:
        raise ValueError("support_tile_size must be a positive integer.")
    complete_rows = torch.zeros(
        (batch, receiver_count),
        device=query_density.device,
        dtype=torch.bool,
    )
    empty_row_ptr = torch.zeros(
        (batch * receiver_count + 1,),
        device=query_density.device,
        dtype=torch.long,
    )
    empty_index = torch.empty((0,), device=query_density.device, dtype=torch.long)
    source_valid = torch.zeros(
        (batch, source_count),
        device=query_density.device,
        dtype=torch.bool,
    )
    empty_prior = query_density.new_empty((0, source_count), dtype=torch.float64)
    if receiver_count == 0 or source_count == 0 or hub_count == 0:
        return CompiledPairs(
            complete_rows=complete_rows,
            complete_row_index=empty_index,
            complete_prior=empty_prior,
            partial_row_ptr=empty_row_ptr,
            partial_source_index=empty_index,
            partial_prior=query_density.new_empty((0,), dtype=torch.float64),
            source_valid=source_valid,
            raw_path_count=0,
            unique_pair_count=0,
            support_exam_count=0,
            scalar_exam_count=0,
        )

    # The incidence index is authoritative for support.  Rebuilding its live
    # rectangular view also handles filtered source rows and hand-authored
    # duplicate incidence entries without changing their gradient path.
    source_membership, source_support = _materialize_live_source_incidence(
        incidence,
        batch,
    )
    source_support = source_support & (weights > 0.0)[..., None]
    source_membership = torch.where(
        source_support,
        source_membership,
        torch.zeros_like(source_membership),
    )
    source_valid = source_support.any(dim=-1)
    occupied_hubs = source_valid.new_zeros((batch, hub_count))
    occupied_hubs = source_support.any(dim=1)
    query_support = (query_density > 0.0) & occupied_hubs[:, None, :]
    complete_rows = occupied_hubs.any(dim=-1, keepdim=True) & (
        (~occupied_hubs[:, None, :]) | query_support
    ).all(dim=-1)
    certificate_complete_rows = complete_rows

    # Logical raw paths are counted directly from hub edge cardinalities.  No
    # path tensor is allocated, and complete rows contribute to the audit even
    # though they take the implicit dispatch.
    query_hub_counts = query_support.sum(dim=1)
    source_hub_counts = source_support.sum(dim=1)
    raw_path_count = int((query_hub_counts * source_hub_counts).sum())

    # When the pre-join certificate covers every row, skip the CSR scan and
    # its row-count/promotion bookkeeping entirely.  The complete prior still
    # uses the same bounded scalar products below.
    all_certificate_complete = bool(certificate_complete_rows.all())
    if all_certificate_complete:
        partial_row_ptr = empty_row_ptr
        partial_source_index = empty_index
        partial_row_ids = empty_index
        support_exam_count = 0
        partial_complete_flat = torch.zeros(
            (batch * receiver_count,),
            device=query_density.device,
            dtype=torch.bool,
        )
        complete_rows = certificate_complete_rows
    else:
        partial_row_ptr, partial_source_index, partial_row_ids, support_exam_count = _compiled_partial_union(
            query_density,
            source_support,
            occupied_hubs,
            certificate_complete_rows,
            support_tile_size=int(support_tile_size),
        )
        # Some rows can be complete even when the sufficient pre-join
        # certificate is false (for example, an inactive hub has no source
        # incidence). Promote those rows after the exact Boolean union so the
        # reader still gets implicit complete dispatch wherever valid.
        partial_counts = partial_row_ptr[1:] - partial_row_ptr[:-1]
        source_count_by_row = source_valid.sum(dim=-1).repeat_interleave(receiver_count)
        partial_complete_flat = (
            (~certificate_complete_rows).reshape(-1)
            & (partial_counts > 0)
            & (partial_counts == source_count_by_row)
        )
        complete_rows = certificate_complete_rows | partial_complete_flat.reshape(batch, receiver_count)
    complete_row_index = torch.nonzero(
        complete_rows.reshape(-1),
        as_tuple=False,
    ).flatten()
    if bool(partial_complete_flat.any()):
        partial_keep = ~partial_complete_flat.index_select(0, partial_row_ids)
        partial_row_ids = partial_row_ids[partial_keep]
        partial_source_index = partial_source_index[partial_keep]
        partial_counts = torch.zeros_like(partial_counts)
        partial_counts.index_add_(
            0,
            partial_row_ids,
            torch.ones_like(partial_row_ids, dtype=torch.long),
        )
        partial_row_ptr = torch.cat(
            (
                torch.zeros((1,), device=query_density.device, dtype=torch.long),
                partial_counts.cumsum(dim=0),
            ),
            dim=0,
        )
    # The Boolean union returns row IDs only to keep its output order explicit;
    # priors use those IDs while the public representation retains row pointers
    # and source indices only.
    partial_prior = _compiled_partial_prior(
        query_density,
        source_membership,
        source_support,
        weights,
        partial_row_ids,
        partial_source_index,
        scalar_tile_size=int(scalar_tile_size),
    )
    complete_prior = _compiled_complete_prior(
        query_density,
        source_membership,
        source_support,
        weights,
        complete_row_index,
        scalar_tile_size=int(scalar_tile_size),
    )
    _validate_compiled_priors(
        partial_prior,
        None,
        label="partial",
    )
    complete_batch = torch.div(
        complete_row_index,
        receiver_count,
        rounding_mode="floor",
    )
    complete_pair_count = int(source_valid.index_select(0, complete_batch).sum())
    complete_valid = source_valid.index_select(0, complete_batch) if complete_pair_count else None
    _validate_compiled_priors(
        complete_prior,
        complete_valid,
        label="complete",
    )
    unique_pair_count = complete_pair_count + int(partial_source_index.numel())
    scalar_exam_count = complete_pair_count * hub_count + int(partial_source_index.numel()) * hub_count
    return CompiledPairs(
        complete_rows=complete_rows,
        complete_row_index=complete_row_index,
        complete_prior=complete_prior,
        partial_row_ptr=partial_row_ptr,
        partial_source_index=partial_source_index,
        partial_prior=partial_prior,
        source_valid=source_valid,
        raw_path_count=raw_path_count,
        unique_pair_count=unique_pair_count,
        support_exam_count=support_exam_count,
        scalar_exam_count=scalar_exam_count,
        complete_pair_count_hint=complete_pair_count,
    )


# Explicit aliases keep the operation discoverable for callers that use the
# plan's ``exact`` or ``CSR`` terminology without changing the historical
# ``compile_two_hop_pairs`` default/reference function.
compile_two_hop_pairs_exact = compile_two_hop_pairs_compiled
compile_two_hop_pairs_union_csr = compile_two_hop_pairs_compiled


# Keep the historical public default/reference compiler unchanged while the
# compact implementation is benchmarked and reviewed independently.
compile_two_hop_pairs = compile_two_hop_pairs_reference


# ``compile_routed_pairs`` is intentionally a semantic alias: call sites can
# use either the mathematical two-hop or execution-oriented wording.
compile_routed_pairs = compile_two_hop_pairs
