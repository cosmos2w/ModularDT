"""GPU-friendly two-hop source/query join and positive-pair coalescing."""

from __future__ import annotations

import torch

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


def compile_two_hop_pairs(
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


# ``compile_routed_pairs`` is intentionally a semantic alias: call sites can
# use either the mathematical two-hop or execution-oriented wording.
compile_routed_pairs = compile_two_hop_pairs
